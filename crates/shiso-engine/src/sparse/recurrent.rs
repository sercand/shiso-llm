//! The gated recurrent block, as this kernel set runs it.
//!
//! Three slots in four hold one of these. It projects one fused query/key/value
//! row, runs a short causal convolution over it, normalizes the query and the
//! key, derives a per-head decay and write strength, and then walks a square
//! state matrix per value head forward over the chunk.
//!
//! # Why there are two recurrences and not one
//!
//! The sequential kernel is correct at any row count and is the only one that
//! produces the per-position checkpoints a speculative rollback copies back. The
//! chunked one is a prefill-only path: it is faster over many rows, it produces
//! no per-position state, and its sums are *different* sums rather than a
//! reassociation of the same ones. So the gate that chooses it has three terms,
//! and the middle one is the important one: a run that any rollback could reach
//! stays on the sequential kernel even if the row-count threshold is set below
//! it.

use shiso_core::error::CoreResult;

use crate::sparse::bind::{Args, LayerRef, Operand, Role, WeightRef};
use crate::sparse::kernels::{self, Entry, MOD_GDN, MOD_GDN_TC};
use crate::sparse::matmul::{self, Bound, Device};
use crate::sparse::moe::RowOrigin;
use crate::sparse::mtp::SpecDepth;
use crate::sparse::plan::Plan;
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::weights::RecurrentDesc;

/// When the chunked recurrence is allowed to run.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPolicy {
    /// Rows from which the chunked kernel is considered at all.
    ///
    /// Zero disables it entirely, which is what a build with no chunked kernel
    /// or a caller who wants the sequential numerics everywhere asks for.
    pub min_rows: u32,
    /// Whether the device could actually grant the chunked kernel's shared
    /// memory. It asks for nearly a hundred kilobytes, which not every device
    /// will give a block.
    pub available: bool,
}

impl ChunkPolicy {
    /// The default: the chunked kernel from twice its own run width.
    ///
    /// Twice the run width rather than once, so that a chunk that takes this
    /// path has at least two full runs to amortize its setup over.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min_rows: 2 * kernels::GDN_CHUNK,
            available: true,
        }
    }

    /// Whether a chunk of `n_tok` rows takes the chunked kernel at `spec`
    /// drafts per step.
    ///
    /// The depth is an argument rather than a field because it is the same
    /// number that sizes the rollback slabs on every session, and two copies of
    /// it would be two chances for a run to take the chunked kernel while a
    /// rollback expects records it never wrote.
    #[must_use]
    pub const fn takes_chunk(&self, n_tok: u32, spec: SpecDepth) -> bool {
        self.min_rows != 0 && n_tok >= self.min_rows && n_tok > spec.verify_rows() && self.available
    }
}

impl Default for ChunkPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// The two decisions that together choose which recurrence runs and what it
/// records.
///
/// One value rather than two arguments, because they are not independent: the
/// chunked kernel writes no per-position state, so a policy that took the
/// chunked path at a depth that can roll back would restore a position nothing
/// recorded. [`Recurrence::takes_chunk`] is the one place that decision is made.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Recurrence {
    /// When the chunked kernel is allowed to run.
    pub chunk: ChunkPolicy,
    /// Drafts one step may carry, which is what the sequential kernel records.
    pub spec: SpecDepth,
    /// Whether the rows are one sequence's consecutive positions or one row per
    /// sequence.
    ///
    /// It is here rather than beside `n_tok` because it is a third term of the
    /// same decision: unrelated rows carry a state each, so they take the
    /// batched kernels, which are the single-token body of the sequential ones
    /// with every read offset by its row and no chained scan at all.
    pub origin: RowOrigin,
}

impl Recurrence {
    /// Whether a chunk of `n_tok` rows takes the chunked kernel.
    ///
    /// Never, for unrelated rows: the chunked kernel walks one state forward
    /// over consecutive positions, and a batch's rows have neither one state
    /// nor consecutive positions.
    #[must_use]
    pub const fn takes_chunk(&self, n_tok: u32) -> bool {
        matches!(self.origin, RowOrigin::Sequential) && self.chunk.takes_chunk(n_tok, self.spec)
    }

    /// Whether the rows are one row per sequence.
    #[must_use]
    pub const fn is_batched(&self) -> bool {
        matches!(self.origin, RowOrigin::Unrelated)
    }
}

/// One recurrent block, in launch order.
///
/// The two normalizations are one kernel launched twice, at the two offsets the
/// fused projection puts the query and the key rows at: the value rows are used
/// as they come, which is why there is no third. The offsets are arguments
/// rather than pointer advances, which is why both launches read the same
/// buffer.
///
/// `layer` is the slot index and `slot` its rank among recurrent slots: the
/// first names the weights, the second names the session state, and the two are
/// different numbers because three slots in four hold one of these.
///
/// # Errors
///
/// Whatever [`matmul::project`] reports for a weight this build cannot project.
#[expect(
    clippy::too_many_lines,
    reason = "one statement per kernel argument is the point: these bodies are \
              transcriptions of parameter lists, and a split that broke a list \
              across two functions would hide the order being transcribed"
)]
pub fn block(
    shape: &SparseShape,
    desc: &RecurrentDesc,
    layer: LayerRef,
    slot: u32,
    policy: Recurrence,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let spec = policy.spec;
    let at = |role: Role| WeightRef::layer(layer, role);
    // The two arguments the rollback needs, as one decision: a slab and a
    // record count that disagreed would be a kernel writing past the end of a
    // buffer, so they are derived together from the one depth.
    let conv_ckpt = rollback_args(spec, Operand::RecurrentConvCkpt(slot));
    let state_ckpt = rollback_args(spec, Operand::RecurrentStateCkpt(slot));
    let mut plan = Plan::new();
    matmul::project(
        &mut plan,
        desc.qkv,
        Bound::new(
            at(Role::RecurrentQkv),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::GdnQkv),
        ),
        n_tok,
        device,
    )?;
    if let RowOrigin::Segmented(runs) = policy.origin {
        // Each run is its own sequence's consecutive positions: the scalar
        // convolution's body over that run's rows, from that session's window,
        // with that session's rollback records - one launch for every run.
        let (ckpt, cap) = rollback_args(spec, Operand::RecurrentConvCkpts(slot));
        plan.launch(
            Entry::plain(MOD_GDN, "q4e_gdn_conv_seg_kernel"),
            kernels::gdn_conv_batch(shape.gdn_in_dim(), runs.count()),
            // `out, states, x, w, channels, kernel, run_lens, ckpts, ckpt_cap`
            Args::of(&[
                Operand::Scratch(Buf::GdnConv),
                Operand::RecurrentConvs(slot),
                Operand::Scratch(Buf::GdnQkv),
                Operand::Weight(at(Role::RecurrentConv1d)),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U32(shape.gdn.conv_kernel),
                Operand::U64(runs.packed_lens()),
                ckpt,
                cap,
            ]),
        );
    } else if policy.is_batched() {
        plan.launch(
            Entry::plain(MOD_GDN, "q4e_gdn_conv_batch_kernel"),
            kernels::gdn_conv_batch(shape.gdn_in_dim(), n_tok),
            // `out, states, x, w, channels, kernel` - one window per row,
            // dereferenced by `blockIdx.y`. There is no row count and no
            // checkpoint slab: each row is exactly one token, so there is
            // nothing to scan and nothing to record.
            Args::of(&[
                Operand::Scratch(Buf::GdnConv),
                Operand::RecurrentConvs(slot),
                Operand::Scratch(Buf::GdnQkv),
                Operand::Weight(at(Role::RecurrentConv1d)),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U32(shape.gdn.conv_kernel),
            ]),
        );
    } else if policy.takes_chunk(n_tok) {
        // A prefill chunk convolves in parallel over tokens and then rewrites
        // the carried window; like the recurrence it records no per-position
        // state, which is what the chunked path already guarantees it is never
        // asked for. Measured on a GB10 at an 8192-row chunk: the sequential
        // walk was 3.9 ms a layer on 40 blocks.
        plan.launch(
            Entry::plain(MOD_GDN_TC, "q4e_gdn_conv_par_kernel"),
            kernels::gdn_conv_par(shape.gdn_in_dim(), n_tok),
            // `out, state, x, w, channels, kernel, n_tok`
            Args::of(&[
                Operand::Scratch(Buf::GdnConv),
                Operand::RecurrentConv(slot),
                Operand::Scratch(Buf::GdnQkv),
                Operand::Weight(at(Role::RecurrentConv1d)),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U32(shape.gdn.conv_kernel),
                Operand::U32(n_tok),
            ]),
        );
        plan.launch(
            Entry::plain(MOD_GDN_TC, "q4e_gdn_conv_tail_kernel"),
            kernels::gdn_conv(shape.gdn_in_dim()),
            // `state, x, channels, kernel, n_tok`
            Args::of(&[
                Operand::RecurrentConv(slot),
                Operand::Scratch(Buf::GdnQkv),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U32(shape.gdn.conv_kernel),
                Operand::U32(n_tok),
            ]),
        );
    } else {
        plan.launch(
            Entry::plain(MOD_GDN, "q4e_gdn_conv_kernel"),
            kernels::gdn_conv(shape.gdn_in_dim()),
            // `out, state, x, w, channels, kernel, n_tok, ckpt, ckpt_cap` - the
            // checkpoint slab is the speculative rollback's, absent on a plain
            // pass.
            Args::of(&[
                Operand::Scratch(Buf::GdnConv),
                Operand::RecurrentConv(slot),
                Operand::Scratch(Buf::GdnQkv),
                Operand::Weight(at(Role::RecurrentConv1d)),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U32(shape.gdn.conv_kernel),
                Operand::U32(n_tok),
                conv_ckpt.0,
                conv_ckpt.1,
            ]),
        );
    }
    // `qk, head_dim, n_heads, stride, offset` - the query rows start the fused
    // projection and the key rows follow them.
    for offset in [0, shape.gdn_qk_dim()] {
        plan.launch(
            Entry::plain(MOD_GDN, "q4e_gdn_l2norm_kernel"),
            kernels::gdn_l2norm(shape.gdn.qk_heads, n_tok),
            Args::of(&[
                Operand::Scratch(Buf::GdnConv),
                Operand::U32(shape.gdn.head_dim),
                Operand::U32(shape.gdn.qk_heads),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U32(offset),
            ]),
        );
    }
    matmul::project(
        &mut plan,
        desc.alpha,
        Bound::new(
            at(Role::RecurrentAlpha),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::GdnAlphaProj),
        ),
        n_tok,
        device,
    )?;
    matmul::project(
        &mut plan,
        desc.beta,
        Bound::new(
            at(Role::RecurrentBeta),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::GdnBetaProj),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::plain(MOD_GDN, "q4e_gdn_gates_kernel"),
        kernels::gdn_gates(shape.gdn.v_heads, n_tok),
        // `decay, beta, alpha_proj, beta_proj, dt_bias, a, n_head, n_tok`
        Args::of(&[
            Operand::Scratch(Buf::GdnDecay),
            Operand::Scratch(Buf::GdnBeta),
            Operand::Scratch(Buf::GdnAlphaProj),
            Operand::Scratch(Buf::GdnBetaProj),
            Operand::Weight(at(Role::RecurrentDecayBias)),
            Operand::Weight(at(Role::RecurrentDecayBase)),
            Operand::U32(shape.gdn.v_heads),
            Operand::U32(n_tok),
        ]),
    );
    if let RowOrigin::Segmented(runs) = policy.origin {
        // The same, for the recurrence: one sequential walk per run, every run
        // in one launch. Launched once per run instead, the walks were 32
        // blocks each on a 48-SM device and 10 ms of a four-sequence verify
        // step on a GB10.
        let (ckpt, cap) = rollback_args(spec, Operand::RecurrentStateCkpts(slot));
        plan.launch(
            Entry::plain(MOD_GDN, "q4e_gdn_recurrent_seg_kernel"),
            kernels::gdn_recurrent_batch(shape.gdn.head_dim, shape.gdn.v_heads, runs.count()),
            // `attn_out, states, qkv, decay, beta, head_dim, n_head_k,
            //  n_head_v, k_offset, v_offset, stride, run_lens, ckpts, ckpt_cap`
            Args::of(&[
                Operand::Scratch(Buf::GdnAttn),
                Operand::RecurrentStates(slot),
                Operand::Scratch(Buf::GdnConv),
                Operand::Scratch(Buf::GdnDecay),
                Operand::Scratch(Buf::GdnBeta),
                Operand::U32(shape.gdn.head_dim),
                Operand::U32(shape.gdn.qk_heads),
                Operand::U32(shape.gdn.v_heads),
                Operand::U32(shape.gdn_qk_dim()),
                Operand::U32(2 * shape.gdn_qk_dim()),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U64(runs.packed_lens()),
                ckpt,
                cap,
            ]),
        );
    } else if policy.is_batched() {
        plan.launch(
            Entry::plain(MOD_GDN, "q4e_gdn_recurrent_batch_kernel"),
            kernels::gdn_recurrent_batch(shape.gdn.head_dim, shape.gdn.v_heads, n_tok),
            // `attn_out, states, qkv, decay, beta, head_dim, n_head_k,
            //  n_head_v, k_offset, v_offset, stride` - one state matrix per
            // row, and no row count: `blockIdx.y` is the row and each row is
            // one token.
            Args::of(&[
                Operand::Scratch(Buf::GdnAttn),
                Operand::RecurrentStates(slot),
                Operand::Scratch(Buf::GdnConv),
                Operand::Scratch(Buf::GdnDecay),
                Operand::Scratch(Buf::GdnBeta),
                Operand::U32(shape.gdn.head_dim),
                Operand::U32(shape.gdn.qk_heads),
                Operand::U32(shape.gdn.v_heads),
                Operand::U32(shape.gdn_qk_dim()),
                Operand::U32(2 * shape.gdn_qk_dim()),
                Operand::U32(shape.gdn_in_dim()),
            ]),
        );
    } else if policy.takes_chunk(n_tok) {
        // The chunked recurrence, on tensor cores: the half of each run's work
        // that does not depend on the state at its start for every run at once,
        // then the carry, one block per head. Measured on a GB10 at an
        // 8192-row chunk against the CUDA-core chunked kernel it replaces:
        // 21.3 -> 6.4 ms a layer, outputs within 5e-4 relative L2. Neither
        // writes per-position state, which is why neither carries a
        // checkpoint slab.
        plan.launch(
            Entry::plain(MOD_GDN_TC, "q4e_gdn_tc_prep_kernel"),
            kernels::gdn_tc_prep(shape.gdn.v_heads, n_tok),
            // `u, w, p, g, qkv, decay, beta, n_head_k, n_head_v, k_offset,
            //  v_offset, stride, n_tok`
            Args::of(&[
                Operand::Scratch(Buf::GdnTcU),
                Operand::Scratch(Buf::GdnTcW),
                Operand::Scratch(Buf::GdnTcP),
                Operand::Scratch(Buf::GdnTcG),
                Operand::Scratch(Buf::GdnConv),
                Operand::Scratch(Buf::GdnDecay),
                Operand::Scratch(Buf::GdnBeta),
                Operand::U32(shape.gdn.qk_heads),
                Operand::U32(shape.gdn.v_heads),
                Operand::U32(shape.gdn_qk_dim()),
                Operand::U32(2 * shape.gdn_qk_dim()),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U32(n_tok),
            ]),
        );
        plan.launch(
            Entry::plain(MOD_GDN_TC, "q4e_gdn_tc_scan_kernel"),
            kernels::gdn_tc_scan(shape.gdn.v_heads),
            // `attn_out, state, qkv, u, w, p, g, n_head_k, n_head_v,
            //  k_offset, stride, n_tok`
            Args::of(&[
                Operand::Scratch(Buf::GdnAttn),
                Operand::RecurrentState(slot),
                Operand::Scratch(Buf::GdnConv),
                Operand::Scratch(Buf::GdnTcU),
                Operand::Scratch(Buf::GdnTcW),
                Operand::Scratch(Buf::GdnTcP),
                Operand::Scratch(Buf::GdnTcG),
                Operand::U32(shape.gdn.qk_heads),
                Operand::U32(shape.gdn.v_heads),
                Operand::U32(shape.gdn_qk_dim()),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U32(n_tok),
            ]),
        );
    } else {
        plan.launch(
            Entry::plain(MOD_GDN, "q4e_gdn_recurrent_kernel"),
            kernels::gdn_recurrent(shape.gdn.head_dim, shape.gdn.v_heads),
            // `attn_out, state, qkv, decay, beta, head_dim, n_head_k, n_head_v,
            //  k_offset, v_offset, stride, n_tok, ckpt, ckpt_cap`
            Args::of(&[
                Operand::Scratch(Buf::GdnAttn),
                Operand::RecurrentState(slot),
                Operand::Scratch(Buf::GdnConv),
                Operand::Scratch(Buf::GdnDecay),
                Operand::Scratch(Buf::GdnBeta),
                Operand::U32(shape.gdn.head_dim),
                Operand::U32(shape.gdn.qk_heads),
                Operand::U32(shape.gdn.v_heads),
                Operand::U32(shape.gdn_qk_dim()),
                Operand::U32(2 * shape.gdn_qk_dim()),
                Operand::U32(shape.gdn_in_dim()),
                Operand::U32(n_tok),
                state_ckpt.0,
                state_ckpt.1,
            ]),
        );
    }
    matmul::project(
        &mut plan,
        desc.gate,
        Bound::new(
            at(Role::RecurrentGate),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::GdnGate),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::plain(MOD_GDN, "q4e_gdn_out_gate_kernel"),
        kernels::gdn_out_gate(shape.gdn.v_heads, n_tok),
        // `out, attn, z, w, head_dim, n_head, eps`
        Args::of(&[
            Operand::Scratch(Buf::GdnGated),
            Operand::Scratch(Buf::GdnAttn),
            Operand::Scratch(Buf::GdnGate),
            Operand::Weight(at(Role::RecurrentNorm)),
            Operand::U32(shape.gdn.head_dim),
            Operand::U32(shape.gdn.v_heads),
            Operand::f32(shape.rms_eps),
        ]),
    );
    matmul::project(
        &mut plan,
        desc.out,
        Bound::new(
            at(Role::RecurrentOut),
            Operand::Scratch(Buf::GdnGated),
            Operand::Scratch(Buf::BlockOut),
        ),
        n_tok,
        device,
    )?;
    Ok(plan)
}

/// The slab and the record count one per-position checkpoint argument pair
/// carries, or the null pair on a pass that drafts nothing.
///
/// A pair rather than two calls, because the kernels test the pointer and then
/// index it by the count: a slab with a zero count records nothing and a null
/// pointer with a count is a write through zero.
const fn rollback_args(spec: SpecDepth, slab: Operand) -> (Operand, Operand) {
    if spec.is_speculative() {
        (slab, Operand::U32(spec.rollback_slots()))
    } else {
        (Operand::Null, Operand::U32(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_a_rollback_could_reach_stays_on_the_sequential_kernel() {
        // The threshold alone would take the chunked path at five rows; the
        // draft depth is what keeps it sequential, because the chunked kernel
        // records no per-position state for a rollback to restore. Nine rows is
        // exactly a verify batch at depth eight and is still refused; ten is one
        // row wider than any verify pass can be and is allowed.
        let policy = ChunkPolicy {
            min_rows: 4,
            available: true,
        };
        let deep = SpecDepth::new(8);
        assert!(!policy.takes_chunk(5, deep));
        assert!(!policy.takes_chunk(9, deep));
        assert!(policy.takes_chunk(10, deep));
        // Without speculation the threshold is the only term left.
        assert!(policy.takes_chunk(5, SpecDepth::NONE));
    }

    #[test]
    fn a_zero_threshold_disables_the_chunked_kernel_at_every_row_count() {
        let policy = ChunkPolicy {
            min_rows: 0,
            available: true,
        };
        assert!(!policy.takes_chunk(512, SpecDepth::NONE));
    }

    #[test]
    fn a_device_that_cannot_grant_the_shared_memory_never_takes_the_chunked_path() {
        let policy = ChunkPolicy {
            available: false,
            ..ChunkPolicy::new()
        };
        assert!(!policy.takes_chunk(512, SpecDepth::NONE));
        assert!(ChunkPolicy::new().takes_chunk(512, SpecDepth::NONE));
    }

    #[test]
    fn decode_never_reaches_the_chunked_kernel_under_the_default_policy() {
        let policy = ChunkPolicy::new();
        assert!(!policy.takes_chunk(1, SpecDepth::NONE));
        assert!(!policy.takes_chunk(127, SpecDepth::NONE));
        assert!(policy.takes_chunk(128, SpecDepth::NONE));
    }

    #[test]
    fn a_pass_that_drafts_nothing_hands_the_recurrence_a_null_slab_and_a_zero_count() {
        // Not a slab with a zero count: a session that does not speculate has no
        // records allocated at all, and a pointer to one would be an address the
        // binder cannot produce.
        assert_eq!(
            rollback_args(SpecDepth::NONE, Operand::RecurrentConvCkpt(0)),
            (Operand::Null, Operand::U32(0))
        );
        assert_eq!(
            rollback_args(SpecDepth::new(4), Operand::RecurrentConvCkpt(2)),
            (Operand::RecurrentConvCkpt(2), Operand::U32(4))
        );
    }
}

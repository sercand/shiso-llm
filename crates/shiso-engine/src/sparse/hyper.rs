//! The hyper-connection streams: how a block reads the residual and how its
//! output gets back in.
//!
//! The residual here is not one vector per token but [`SparseShape::streams`]
//! of them, laid out stream-outer so one token is a contiguous run of
//! `streams * hidden` floats. Every block reads a single mixed vector and
//! writes a single contribution, so something has to fold the streams down on
//! the way in and fan the contribution back out on the way out. That is what a
//! *mix* and a *combine* are, and between them they are the only launches every
//! single layer performs whatever block it holds.
//!
//! # The mix and the combine are one chain, not two operations
//!
//! The combine that closes a block does two things in one pass over the
//! residual: it adds the block's contribution back into the streams, and it
//! normalizes the result with the *next* block's norm weight. So the next mix
//! usually has no norm to do - its input is already normed - and the chain
//! breaks in exactly three places: the first island of a pass, because nothing
//! preceded it; an island whose n-gram block reads and rewrites the residual
//! between the update and the mix; and the last layer, whose successor is
//! either the collapsing mix or nothing at all.
//!
//! [`MixInput`] is that chain expressed as a type, so a call site cannot leave
//! it ambiguous.

use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Args, LayerRef, Operand, Role, WeightRef};
use crate::sparse::kernels::{self, Elem, Entry, MOD_BF16, MOD_HC, TemplateArgs};
use crate::sparse::matmul::{self, Bound, Device, NormedStreams};
use crate::sparse::plan::Plan;
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::weights::{MixDesc, WeightDesc};

/// Whether a mix has to normalize its input, or was handed it already normed.
///
/// An enum rather than a `bool`, because the two readings of `false` - "do not
/// normalize" and "there is nothing to normalize yet" - are the difference
/// between a correct pass and one that runs its first block on an unnormalized
/// residual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixInput {
    /// The previous combine left the normed residual behind, so this mix reads
    /// it and launches no norm of its own.
    AlreadyNormed,
    /// Nothing normed this residual, so the mix normalizes it first.
    Raw,
}

/// Whether the normalization chain is whole across one seam.
///
/// The decision, without the weight: [`chain`] answers it from the layer
/// structure alone, and the pass turns a whole link into the weight the
/// combine binds. Two types rather than one because the two are decided in
/// different places and by different things.
///
/// [`chain`]: crate::sparse::island::chain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainLink {
    /// The combine leaves the next block's normed input behind.
    Chained,
    /// The chain breaks here.
    Broken,
}

/// What the combine that closes a block leaves behind for whatever runs next.
///
/// A role rather than a flag: the combine binds the next mix's norm weight, so
/// "chained" has to say *which* weight, and "broken" is the null pointer the
/// launcher passes in its place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextNorm {
    /// The weight of the next block's mix, which this combine applies in the
    /// same pass that updates the residual.
    Chained(WeightRef),
    /// Nothing: the chain breaks here, and whatever runs next normalizes for
    /// itself.
    Broken,
}

impl NextNorm {
    /// The operand the launcher passes for the next norm.
    #[must_use]
    pub const fn operand(self) -> Operand {
        match self {
            Self::Chained(weight) => Operand::Weight(weight),
            Self::Broken => Operand::Null,
        }
    }
}

/// Which four tensors one stream mix reads.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixRoles {
    /// The layer that holds them.
    pub layer: LayerRef,
    /// The per-stream norm.
    pub norm: Role,
    /// The contraction.
    pub down: Role,
    /// The expansion.
    pub up: Role,
    /// The injection logits, absent on the collapsing mix.
    pub inject: Option<Role>,
}

impl MixRoles {
    /// The token mixer's stream mix, for one layer.
    #[must_use]
    pub const fn token_mixer(layer: LayerRef) -> Self {
        Self {
            layer,
            norm: Role::AttnMixNorm,
            down: Role::AttnMixDown,
            up: Role::AttnMixUp,
            inject: Some(Role::AttnMixInject),
        }
    }

    /// The routed feed-forward's stream mix, for one layer.
    #[must_use]
    pub const fn experts(layer: LayerRef) -> Self {
        Self {
            layer,
            norm: Role::FfnMixNorm,
            down: Role::FfnMixDown,
            up: Role::FfnMixUp,
            inject: Some(Role::FfnMixInject),
        }
    }

    /// The mix that collapses the streams at the end of the draft head's pass.
    #[must_use]
    pub const fn draft_output() -> Self {
        Self {
            layer: LayerRef::draft(),
            norm: Role::OutputMixNorm,
            down: Role::OutputMixDown,
            up: Role::OutputMixUp,
            inject: None,
        }
    }

    /// The mix that collapses the streams at the end of a pass.
    #[must_use]
    pub const fn output() -> Self {
        Self {
            layer: LayerRef::target(0),
            norm: Role::OutputMixNorm,
            down: Role::OutputMixDown,
            up: Role::OutputMixUp,
            inject: None,
        }
    }

    /// One of these roles as a reference into the layer that holds it.
    #[must_use]
    pub const fn at(&self, role: Role) -> WeightRef {
        WeightRef {
            layer: self.layer,
            role,
            part: crate::sparse::weights::WeightPart::Whole,
        }
    }
}

/// The broadcast that opens a pass: the embedding tiled across every stream.
#[must_use]
pub fn init(shape: &SparseShape, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_HC, "q4e_hc_init_kernel"),
        kernels::hc_init(shape.hidden, shape.streams, n_tok),
        // `res, embed, n_embd, n_hc, n_tok`
        Args::of(&[
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Embed),
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
            Operand::U32(n_tok),
        ]),
    );
    plan
}

/// The same broadcast with a per-stream term added, which is how a draft head
/// starts from the target's hidden state rather than from an embedding alone.
///
/// `embed` is the per-token term and `hidden` the per-stream one: the draft
/// head's two projections, of the token it was given and of the target's last
/// hidden state, each already at the width the residual wants. Both are
/// operands rather than fixed buffers because neither is where a target pass
/// leaves its embedding - this launch opens a pass that starts from two
/// projections rather than from a gather.
#[must_use]
pub fn init_add(shape: &SparseShape, embed: Operand, hidden: Operand, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_HC, "q4e_hc_init_add_kernel"),
        kernels::hc_init(shape.hidden, shape.streams, n_tok),
        // `res, embed, h, n_embd, n_hc, n_tok`
        Args::of(&[
            Operand::Scratch(Buf::Residual),
            embed,
            hidden,
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
            Operand::U32(n_tok),
        ]),
    );
    plan
}

/// The stream norm, which three call sites share: a mix whose chain broke, and
/// the n-gram block's three normalizations.
///
/// `dst` and `src` are separate because one of those call sites norms a buffer
/// into itself and another norms the residual into a buffer of its own.
#[must_use]
pub fn norm(
    shape: &SparseShape,
    weight: WeightRef,
    src: Operand,
    dst: Operand,
    n_tok: u32,
) -> Plan {
    norm_of(
        shape.hidden,
        shape.streams,
        shape.rms_eps,
        weight,
        src,
        dst,
        n_tok,
    )
}

/// The same normalization at a width and a group count of the caller's.
///
/// The stream norm above is this with the two read off the shape. The draft
/// head needs neither: it normalizes a token embedding as **one** group of the
/// residual width, and a whole multi-stream vector as **one** group of the
/// multi-stream width - a single RMS over all four streams together, which is
/// what the weight it carries is wide enough for. Reading the group count off
/// the shape there would normalize each stream separately against a quarter of
/// the weight, which is four normalizations of the right width and the wrong
/// numbers.
#[must_use]
pub fn norm_of(
    width: u32,
    groups: u32,
    eps: f32,
    weight: WeightRef,
    src: Operand,
    dst: Operand,
    n_tok: u32,
) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::template(MOD_HC, "q4e_hc_norm_kernel", TemplateArgs::Elem(Elem::F32)),
        kernels::hc_norm(groups, n_tok),
        // `out, res, w, n_embd, n_hc, eps`
        Args::of(&[
            dst,
            src,
            Operand::Weight(weight),
            Operand::U32(width),
            Operand::U32(groups),
            Operand::f32(eps),
        ]),
    );
    plan
}

/// The stream norm of a mix, into the precision the chunk's mixes run at.
fn stream_norm(
    shape: &SparseShape,
    weight: WeightRef,
    src: Operand,
    dst: Operand,
    n_tok: u32,
    elem: Elem,
) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::template(MOD_HC, "q4e_hc_norm_kernel", TemplateArgs::Elem(elem)),
        kernels::hc_norm(shape.streams, n_tok),
        // `out, res, w, n_embd, n_hc, eps`
        Args::of(&[
            dst,
            src,
            Operand::Weight(weight),
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
            Operand::f32(shape.rms_eps),
        ]),
    );
    plan
}

/// One stream mix.
///
/// The order is the whole of it: normalize (when the chain broke), contract to
/// the low-rank vector, scale by the stream count and activate, expand back,
/// collapse the gated streams into the block's input, and - for the two mixes
/// that have one - project the per-stream logits the matching combine injects.
///
/// The scale sits *inside* the activation, before the expansion. Moving it
/// outside changes the gate materially rather than by a rounding step, which is
/// why it is a launch of its own rather than a factor folded into a weight.
///
/// # Errors
///
/// Whatever [`matmul::project`] reports for a weight this build cannot
/// project.
///
/// `elem` is the precision the normed residual and the expansion are held in,
/// which [`matmul::half_streams`] decides once for a whole pass: every combine
/// in the pass writes the normed residual the next mix reads, so the two have
/// to agree.
pub fn mix(
    shape: &SparseShape,
    desc: &MixDesc,
    roles: MixRoles,
    input: MixInput,
    n_tok: u32,
    device: Device,
    elem: Elem,
) -> CoreResult<Plan> {
    let mut plan = Plan::new();
    if input == MixInput::Raw {
        plan.append(stream_norm(
            shape,
            roles.at(roles.norm),
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Normed),
            n_tok,
            elem,
        ));
    }
    if elem == Elem::F32 && fuses(shape, desc, n_tok, device) {
        // The previous combine left the scale alone on a pass whose mixes
        // all norm as they read; a mix that normed for itself reads its copy.
        let scaled = input == MixInput::AlreadyNormed && device.normed == NormedStreams::Scaled;
        plan.append(fused_mix(shape, desc, roles, n_tok, scaled));
        return Ok(plan);
    }
    if elem == Elem::F32 && step_fuses(shape, desc, device) {
        plan.append(fused_step_mix(shape, desc, roles, n_tok));
        return Ok(plan);
    }
    let down = Bound::new(
        roles.at(roles.down),
        Operand::Scratch(Buf::Normed),
        Operand::Scratch(Buf::LowRank),
    );
    match elem {
        Elem::F16 => {
            matmul::project_on_blas(&mut plan, desc.down, down, n_tok, device, (elem, Elem::F32))?;
        }
        Elem::F32 => matmul::project(&mut plan, desc.down, down, n_tok, device)?,
    }
    plan.launch(
        Entry::plain(MOD_HC, "q4e_scale_silu_kernel"),
        kernels::elementwise(u64::from(n_tok) * u64::from(shape.stream_rank)),
        // `x, inv_hc, n` - in place, and the reciprocal is the stream count's.
        Args::of(&[
            Operand::Scratch(Buf::LowRank),
            Operand::f32(stream_reciprocal(shape.streams)),
            Operand::U64(u64::from(n_tok) * u64::from(shape.stream_rank)),
        ]),
    );
    let up = Bound::new(
        roles.at(roles.up),
        Operand::Scratch(Buf::LowRank),
        Operand::Scratch(Buf::Expanded),
    );
    match elem {
        Elem::F16 => {
            matmul::project_on_blas(&mut plan, desc.up, up, n_tok, device, (Elem::F32, elem))?;
        }
        Elem::F32 => matmul::project(&mut plan, desc.up, up, n_tok, device)?,
    }
    match elem {
        Elem::F32 => plan.launch(
            Entry::template(MOD_HC, "q4e_hc_collapse_kernel", TemplateArgs::Elem(elem)),
            kernels::hc_collapse(n_tok),
            // `out, xn, up, n_embd, n_hc`
            Args::of(&[
                Operand::Scratch(Buf::Mixed),
                Operand::Scratch(Buf::Normed),
                Operand::Scratch(Buf::Expanded),
                Operand::U32(shape.hidden),
                Operand::U32(shape.streams),
            ]),
        ),
        // The half-precision copy of the block input is written here, where
        // the block input is produced, for every projection that reads it on
        // the half-precision tier: see [`Device::half_block_input`].
        Elem::F16 => plan.launch(
            Entry::plain(MOD_HC, "q4e_hc_collapse_f16_dual_kernel"),
            kernels::hc_collapse(n_tok),
            // `out, out_h, xn, up, n_embd, n_hc`
            Args::of(&[
                Operand::Scratch(Buf::Mixed),
                Operand::Scratch(Buf::MixedHalf),
                Operand::Scratch(Buf::Normed),
                Operand::Scratch(Buf::Expanded),
                Operand::U32(shape.hidden),
                Operand::U32(shape.streams),
            ]),
        ),
    }
    if let (Some(inject), Some(role)) = (desc.inject, roles.inject) {
        let bound = Bound::new(
            roles.at(role),
            Operand::Scratch(Buf::Normed),
            Operand::Scratch(Buf::Inject),
        );
        match elem {
            Elem::F16 => {
                matmul::project_on_blas(
                    &mut plan,
                    inject,
                    bound,
                    n_tok,
                    device,
                    (elem, Elem::F32),
                )?;
            }
            Elem::F32 => matmul::project(&mut plan, inject, bound, n_tok, device)?,
        }
    }
    Ok(plan)
}

/// Streams the fused expansion and collapse is written for
/// (`Q4E_BF_COLLAPSE_HC`).
const FUSED_STREAMS: u32 = 4;
/// Block-input elements one fused-collapse tile covers (`Q4E_BF_COLLAPSE_E`).
const FUSED_COLLAPSE_E: u32 = 32;

/// Whether a mix over single-precision streams takes [`fused_mix`]: a
/// prompt pass whose three brain-float projections the plain prompt GEMM
/// would run, at the stream count and widths the fused kernels are written
/// for.
pub(crate) fn fuses(shape: &SparseShape, desc: &MixDesc, n_tok: u32, device: Device) -> bool {
    let wide = shape.stream_dim();
    shape.streams == FUSED_STREAMS
        && shape.hidden.is_multiple_of(FUSED_COLLAPSE_E)
        && desc.down.in_dim == wide
        && desc.up.out_dim == wide
        && desc.up.in_dim == desc.down.out_dim
        && matmul::brain_plain_gemm(desc.down, n_tok, device)
        && matmul::brain_plain_gemm(desc.up, n_tok, device)
        && desc.inject.is_none_or(|inject| {
            inject.in_dim == wide && matmul::brain_plain_gemm(inject, n_tok, device)
        })
}

/// Whether a mix of a pass of `n_tok` rows reads the residual through the
/// norming launches when the combine before it leaves only the scale: the
/// prompt pass's fused mix. [`NormedStreams::Scaled`] is only planned when
/// every mix of the pass does.
#[must_use]
pub fn norms_on_read(shape: &SparseShape, desc: &MixDesc, n_tok: u32, device: Device) -> bool {
    fuses(shape, desc, n_tok, device)
}

/// The stream mix of a prompt pass in three launches where it took five,
/// each output the bits the launches it replaces write
/// (`kernels/cuda/q4e/bf16.cu`):
///
/// - the contraction and the injection logits as one GEMM whose rows are
///   both weights' (`q4e_bf16_gemm_pair_kernel`): the injection's own read
///   of the normed streams - all of them, for four outputs - is gone, and its
///   rows fit the contraction's last tile;
/// - the activation, as before;
/// - the expansion with the collapse in its epilogue
///   (`q4e_bf16_gemm_collapse_kernel`): the expanded streams, the widest
///   activation of the pass, are never written or read back.
///
/// Measured on a GB10 (kernel bench, wall clock, medians of three) at a
/// 4096-row chunk: contraction and injection 1.98 -> 1.03-1.12 ms, expansion
/// and collapse 3.17-3.33 -> 1.32-1.44 ms.
///
/// With `scaled` the two GEMMs read the residual and the combine's scale in
/// place of the normed streams and norm each value as they read it
/// (`q4e_bf16_gemm_{pair,collapse}_norm_kernel`), the operands - and so the
/// bits - of the stored copy, which the combine then never writes. Measured
/// at a 4096-row chunk: the combine 2.45 -> ~1.7 ms with the reads unchanged.
fn fused_mix(
    shape: &SparseShape,
    desc: &MixDesc,
    roles: MixRoles,
    n_tok: u32,
    scaled: bool,
) -> Plan {
    let mut plan = Plan::new();
    let tile = matmul::BF16_GEMM_TILE;
    let gemm_launch = |x: u32| {
        kernels::Launch::new(
            shiso_core::Extent3::new(x, n_tok.div_ceil(tile), 1),
            matmul::BF16_GEMM_THREADS,
            matmul::BF16_GEMM_SMEM,
        )
    };
    match (desc.inject, roles.inject) {
        (Some(inject), Some(role)) if scaled => plan.launch(
            Entry::plain(MOD_BF16, "q4e_bf16_gemm_pair_norm_kernel"),
            gemm_launch((desc.down.out_dim + inject.out_dim).div_ceil(tile)),
            // `out_a, w_a, out_dim_a, out_b, w_b, out_dim_b, res, scale, nw,
            //  n_embd, in_dim, rows`
            Args::of(&[
                Operand::Scratch(Buf::LowRank),
                Operand::Weight(roles.at(roles.down)),
                Operand::U32(desc.down.out_dim),
                Operand::Scratch(Buf::Inject),
                Operand::Weight(roles.at(role)),
                Operand::U32(inject.out_dim),
                Operand::Scratch(Buf::Residual),
                Operand::Scratch(Buf::NormScale),
                Operand::Weight(roles.at(roles.norm)),
                Operand::U32(shape.hidden),
                Operand::U32(desc.down.in_dim),
                Operand::U32(n_tok),
            ]),
        ),
        (Some(inject), Some(role)) => plan.launch(
            Entry::plain(MOD_BF16, "q4e_bf16_gemm_pair_kernel"),
            gemm_launch((desc.down.out_dim + inject.out_dim).div_ceil(tile)),
            // `out_a, w_a, out_dim_a, out_b, w_b, out_dim_b, x, in_dim, rows`
            Args::of(&[
                Operand::Scratch(Buf::LowRank),
                Operand::Weight(roles.at(roles.down)),
                Operand::U32(desc.down.out_dim),
                Operand::Scratch(Buf::Inject),
                Operand::Weight(roles.at(role)),
                Operand::U32(inject.out_dim),
                Operand::Scratch(Buf::Normed),
                Operand::U32(desc.down.in_dim),
                Operand::U32(n_tok),
            ]),
        ),
        _ => plan.launch(
            Entry::plain(MOD_BF16, "q4e_bf16_gemm_kernel"),
            gemm_launch(desc.down.out_dim.div_ceil(tile)),
            // `out, w, x, in_dim, out_dim, rows`
            Args::of(&[
                Operand::Scratch(Buf::LowRank),
                Operand::Weight(roles.at(roles.down)),
                Operand::Scratch(Buf::Normed),
                Operand::U32(desc.down.in_dim),
                Operand::U32(desc.down.out_dim),
                Operand::U32(n_tok),
            ]),
        ),
    }
    plan.launch(
        Entry::plain(MOD_HC, "q4e_scale_silu_kernel"),
        kernels::elementwise(u64::from(n_tok) * u64::from(shape.stream_rank)),
        Args::of(&[
            Operand::Scratch(Buf::LowRank),
            Operand::f32(stream_reciprocal(shape.streams)),
            Operand::U64(u64::from(n_tok) * u64::from(shape.stream_rank)),
        ]),
    );
    if scaled {
        plan.launch(
            Entry::plain(MOD_BF16, "q4e_bf16_gemm_collapse_norm_kernel"),
            gemm_launch(shape.hidden / FUSED_COLLAPSE_E),
            // `out, w_up, x, res, scale, nw, in_dim, n_embd, n_hc, rows`
            Args::of(&[
                Operand::Scratch(Buf::Mixed),
                Operand::Weight(roles.at(roles.up)),
                Operand::Scratch(Buf::LowRank),
                Operand::Scratch(Buf::Residual),
                Operand::Scratch(Buf::NormScale),
                Operand::Weight(roles.at(roles.norm)),
                Operand::U32(desc.up.in_dim),
                Operand::U32(shape.hidden),
                Operand::U32(shape.streams),
                Operand::U32(n_tok),
            ]),
        );
        return plan;
    }
    plan.launch(
        Entry::plain(MOD_BF16, "q4e_bf16_gemm_collapse_kernel"),
        gemm_launch(shape.hidden / FUSED_COLLAPSE_E),
        // `out, w_up, x, xn, in_dim, n_embd, n_hc, rows`
        Args::of(&[
            Operand::Scratch(Buf::Mixed),
            Operand::Weight(roles.at(roles.up)),
            Operand::Scratch(Buf::LowRank),
            Operand::Scratch(Buf::Normed),
            Operand::U32(desc.up.in_dim),
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
            Operand::U32(n_tok),
        ]),
    );
    plan
}

/// The low-rank width the fused step mix is written for (`Q4E_BF_MIX_RANK`).
const STEP_MIX_RANK: u32 = 320;
/// Block-input elements one fused step block covers (`Q4E_BF_MIX_E`).
const STEP_MIX_E: u32 = 8;
/// Rows one fused step launch takes; a wider pass runs as passes of it.
const STEP_MIX_ROWS: u32 = 8;

/// Whether a mix over single-precision streams on a step or verify pass
/// takes [`fused_step_mix`]: brain-float projections on a device with the
/// brain-float step kernels, at the widths the fused step is written for.
/// A function of the weights and the pass role alone - never of the row
/// count - because the fused expansion sums a row in its own order, and
/// every step row of every width has to be summed that one way.
fn step_fuses(shape: &SparseShape, desc: &MixDesc, device: Device) -> bool {
    let wide = shape.stream_dim();
    !device.row_invariant()
        && matches!(device.prompt_matmul, matmul::PromptMatmul::Tensor)
        && shape.streams == FUSED_STREAMS
        && shape.hidden.is_multiple_of(STEP_MIX_E)
        && desc.down.dtype == shiso_core::DType::BF16
        && desc.up.dtype == shiso_core::DType::BF16
        && desc.down.in_dim == wide
        && desc.down.out_dim == STEP_MIX_RANK
        && desc.up.in_dim == STEP_MIX_RANK
        && desc.up.out_dim == wide
        && desc
            .inject
            .is_none_or(|inject| inject.dtype == shiso_core::DType::BF16 && inject.in_dim == wide)
}

/// The stream mix of a step or verify pass in two launches where it took
/// five, in passes of [`STEP_MIX_ROWS`] rows (`kernels/cuda/q4e/bf16.cu`):
///
/// - the contraction and the injection logits stacked in one step launch
///   (`q4e_bf16_{gemv,rows}_pair_kernel`), each row the bits its own launch
///   writes - both are cut into the same slices - and the injection's launch
///   gone, when the two share a slice count;
/// - the activation, the expansion and the collapse in one launch
///   (`q4e_bf16_mix_step_kernel`), reading the 640-byte expansion rows four
///   to a warp.
///
/// Measured on a GB10 (kernel bench, wall clock, a 2 GiB working set):
/// contraction and injection 34-37 -> 31-32 us at one row and 38 -> 32-33 us
/// at four; the fused step 30-32 us at one row and 34 us at four, where the
/// expansion alone took 29-31 us and the activation and the collapse were two
/// launches more.
fn fused_step_mix(shape: &SparseShape, desc: &MixDesc, roles: MixRoles, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    let wide = shape.stream_dim();
    let mut done = 0;
    while done < n_tok {
        let rows = (n_tok - done).min(STEP_MIX_ROWS);
        let at = |buf: Buf, width: u32| {
            let bytes = u64::from(done) * u64::from(width) * 4;
            if bytes == 0 {
                Operand::Scratch(buf)
            } else {
                Operand::ScratchFrom(buf, bytes)
            }
        };
        let splits = matmul::bf16_split(desc.down.in_dim, desc.down.out_dim);
        let per_block = matmul::BF16_STEP_WARPS / splits * matmul::bf16_warp_rows(rows);
        let stacked = desc.inject.filter(|inject| {
            matmul::bf16_split(inject.in_dim, inject.out_dim) == splits
                && matmul::bf16_split(desc.down.in_dim, desc.down.out_dim + inject.out_dim)
                    == splits
        });
        if let (Some(inject), Some(role)) = (stacked, roles.inject) {
            let total = desc.down.out_dim + inject.out_dim;
            plan.launch(
                Entry::plain(
                    MOD_BF16,
                    if rows == 1 {
                        "q4e_bf16_gemv_pair_kernel"
                    } else {
                        "q4e_bf16_rows_pair_kernel"
                    },
                ),
                kernels::Launch::linear(total.div_ceil(per_block), matmul::BF16_STEP_THREADS),
                // `out_a, w_a, out_dim_a, out_b, w_b, out_dim_b, x, in_dim,
                //  splits, rows`
                Args::of(&[
                    at(Buf::LowRank, desc.down.out_dim),
                    Operand::Weight(roles.at(roles.down)),
                    Operand::U32(desc.down.out_dim),
                    at(Buf::Inject, inject.out_dim),
                    Operand::Weight(roles.at(role)),
                    Operand::U32(inject.out_dim),
                    at(Buf::Normed, wide),
                    Operand::U32(desc.down.in_dim),
                    Operand::U32(splits),
                    Operand::U32(rows),
                ]),
            );
        } else {
            let down = roles.at(roles.down);
            plan.append(step_projection(
                desc.down,
                down,
                at(Buf::Normed, wide),
                at(Buf::LowRank, desc.down.out_dim),
                rows,
            ));
            if let (Some(inject), Some(role)) = (desc.inject, roles.inject) {
                let dst = at(Buf::Inject, inject.out_dim);
                plan.append(step_projection(
                    inject,
                    roles.at(role),
                    at(Buf::Normed, wide),
                    dst,
                    rows,
                ));
            }
        }
        plan.launch(
            Entry::plain(MOD_BF16, "q4e_bf16_mix_step_kernel"),
            kernels::Launch::linear(shape.hidden / STEP_MIX_E, matmul::BF16_STEP_THREADS),
            // `out, w_up, x, xn, inv_hc, in_dim, n_embd, n_hc, rows`
            Args::of(&[
                at(Buf::Mixed, shape.hidden),
                Operand::Weight(roles.at(roles.up)),
                at(Buf::LowRank, desc.down.out_dim),
                at(Buf::Normed, wide),
                Operand::f32(stream_reciprocal(shape.streams)),
                Operand::U32(desc.up.in_dim),
                Operand::U32(shape.hidden),
                Operand::U32(shape.streams),
                Operand::U32(rows),
            ]),
        );
        done += rows;
    }
    plan
}

/// One brain-float step projection of at most eight rows: `src` through
/// `weight` into `dst`, as [`matmul::project`] plans it on a tensor-core
/// device.
fn step_projection(
    desc: WeightDesc,
    weight: WeightRef,
    src: Operand,
    dst: Operand,
    rows: u32,
) -> Plan {
    let splits = matmul::bf16_split(desc.in_dim, desc.out_dim);
    let per_block = matmul::BF16_STEP_WARPS / splits * matmul::bf16_warp_rows(rows);
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(
            MOD_BF16,
            if rows == 1 {
                "q4e_bf16_gemv_kernel"
            } else {
                "q4e_bf16_rows_kernel"
            },
        ),
        kernels::Launch::linear(desc.out_dim.div_ceil(per_block), matmul::BF16_STEP_THREADS),
        // `out, w, x, in_dim, out_dim, splits, rows`
        Args::of(&[
            dst,
            Operand::Weight(weight),
            src,
            Operand::U32(desc.in_dim),
            Operand::U32(desc.out_dim),
            Operand::U32(splits),
            Operand::U32(rows),
        ]),
    );
    plan
}

/// The fused residual update and next-block norm that closes a block.
///
/// One launch, not two. Running an add and then a norm instead would round the
/// residual twice - once when the sum is stored and again when it is read back -
/// so the residual the next layer sees would not be the one this layer
/// intended.
///
/// `next` is the *caller's* to decide: it changes which weight the launch binds
/// and nothing about the grid, and where the chain breaks it is the null
/// pointer that turns this launch into the residual update alone.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the stream width is wider than the fused
/// kernel's fixed block can cover, which its launcher refuses outright rather
/// than leaving a tail of the residual unwritten.
///
/// `elem` is the precision the normed residual is written in, which is the one
/// the next mix reads it at; see [`mix`].
pub fn combine(
    shape: &SparseShape,
    next: NextNorm,
    n_tok: u32,
    elem: Elem,
    normed: NormedStreams,
) -> CoreResult<Plan> {
    if !kernels::hc_combine_fits(shape.hidden) {
        return Err(CoreError::Unsupported {
            what: "a residual stream wider than the fused combine's block can cover",
            backend: "engine",
        });
    }
    let mut plan = Plan::new();
    // A chained combine on a pass whose mixes norm as they read leaves the
    // scale alone: the same residual update, the norm's scale, and no normed
    // copy (`q4e_hc_combine_scale_kernel`). The next mix applies its own norm
    // weight - the one `next` names - as it reads.
    if let (NextNorm::Chained(_), NormedStreams::Scaled, Elem::F32) = (next, normed, elem) {
        plan.launch(
            Entry::plain(MOD_HC, "q4e_hc_combine_scale_kernel"),
            kernels::hc_combine(shape.streams, n_tok),
            // `res, scale, block_out, inject, n_embd, n_hc, eps`
            Args::of(&[
                Operand::Scratch(Buf::Residual),
                Operand::Scratch(Buf::NormScale),
                Operand::Scratch(Buf::BlockOut),
                Operand::Scratch(Buf::Inject),
                Operand::U32(shape.hidden),
                Operand::U32(shape.streams),
                Operand::f32(shape.rms_eps),
            ]),
        );
        return Ok(plan);
    }
    plan.launch(
        Entry::template(
            MOD_HC,
            "q4e_hc_combine_norm_kernel",
            TemplateArgs::Elem(elem),
        ),
        kernels::hc_combine(shape.streams, n_tok),
        // `res, xn, block_out, inject, w, n_embd, n_hc, eps`
        Args::of(&[
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Normed),
            Operand::Scratch(Buf::BlockOut),
            Operand::Scratch(Buf::Inject),
            next.operand(),
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
            Operand::f32(shape.rms_eps),
        ]),
    );
    Ok(plan)
}

/// The mix that ends a pass.
///
/// It collapses the streams for the vocabulary projection and stands in for an
/// output norm; there is no separate output-norm tensor, and nothing to inject
/// back into a residual - which is why its descriptor has no injection weight
/// and this is the one mix whose input is never already normed.
///
/// # Errors
///
/// Whatever [`mix`] reports.
pub fn collapse(
    shape: &SparseShape,
    desc: &MixDesc,
    roles: MixRoles,
    n_tok: u32,
    device: Device,
    elem: Elem,
) -> CoreResult<Plan> {
    if desc.inject.is_some() {
        return Err(CoreError::InvalidValue {
            key: "output mix",
            value: "the collapsing mix has no residual to inject into".to_owned(),
        });
    }
    mix(shape, desc, roles, MixInput::Raw, n_tok, device, elem)
}

/// The reciprocal of the stream count, which the activation scales by.
///
/// Narrowed through a sixteen-bit conversion rather than cast: the stream count
/// is a handful, the conversion is exact there, and a count that did not fit
/// would be a geometry error rather than a silently rounded scale.
fn stream_reciprocal(streams: u32) -> f32 {
    let streams = f32::from(u16::try_from(streams).unwrap_or(u16::MAX));
    if streams == 0.0 { 0.0 } else { 1.0 / streams }
}

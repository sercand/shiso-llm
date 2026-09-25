//! The routed feed-forward: a router over hundreds of experts, plus a shared
//! expert every token pays for.
//!
//! Every layer holds one. The router scores every expert for every token and
//! keeps the best handful; each surviving (token, expert) pair becomes a
//! *column*; the columns run through the chosen experts' gate, up, activation
//! and down; and the results are recombined into one contribution per token,
//! with the shared expert's own output added on top.
//!
//! # Where the arithmetic is, and why it is worth isolating
//!
//! Almost nothing in this file is a shape. The column count, the sorted layout,
//! the grid the down projection runs on, the threshold that decides whether
//! columns are grouped by expert - all of them are arithmetic over the routing,
//! and every one of them is checkable without a device. That matters more here
//! than anywhere else in the pass, because the failure mode of getting the
//! routing wrong is not a crash: it is a token answered by the wrong ten
//! experts, which reads as a model that is merely a bit worse.
//!
//! # Two expert stages, and one of them this build cannot reach
//!
//! A trellis expert stack runs gate, up, activation and down as one cooperative
//! launch per expert group. Anything else runs them staged - and the staged
//! gate-and-up belongs to a different kernel set than the one vendored here, so
//! it is a typed refusal naming the missing companion rather than a slower path
//! that quietly produces different numbers.
//!
//! That is also why [`block`] takes no [`RowOrigin`]. The fused launch groups
//! the expert slots *one token* was routed to - its fan-out counts
//! [`MoeShape::used`] and its grid is one group per expert, neither of which
//! depends on the row count - so what the rows beside a token routed to changes
//! neither which blocks run nor what any of them sums. The grouping that does
//! depend on the rows being related is the down projection's, which sweeps one
//! expert's row across every column routed to it, and that is [`RowOrigin`]'s
//! own question in [`down`]. A gate on the fused launch could only ever turn a
//! decode batch into the refusal above, because there is no second path for it
//! to fall to.
//!
//! [`MoeShape::used`]: crate::sparse::shape::MoeShape::used

use shiso_core::DType;
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Args, LayerRef, Operand, Role, WeightRef, WeightSet};
use crate::sparse::exl3::{self, Fanout};
use crate::sparse::kernels::{self, Entry, MOD_EXL3, MOD_MOE, TemplateArgs};
use crate::sparse::matmul::{self, Bound, Device};
use crate::sparse::plan::{Branch, Plan, Step, TrellisOp};
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::weights::{ExpertDesc, WeightPart};

/// Rows from which columns are grouped by expert before the down projection.
///
/// The grouped kernel reads one expert's row into registers once and sweeps
/// every column routed to it; the per-column kernel reads the whole row again
/// for each column. At one token those are the same traffic, and at a
/// five-hundred-row chunk the second is five hundred times too much.
pub const GROUP_MIN_ROWS: u32 = 4;

/// Rows from which columns are grouped when a chunk's rows belong to unrelated
/// sequences.
///
/// Consecutive tokens of one sequence route to overlapping experts, so grouping
/// them pays from a handful of rows. The rows of a batched decode step are
/// unrelated sequences: there the grouping gains very little and changes the
/// rounding of a path that is otherwise identical to a single-sequence decode,
/// so it waits for a row count that is a real chunk.
pub const GROUP_MIN_UNRELATED_ROWS: u32 = 32;

/// Whether a chunk's rows come from one sequence or from several.
///
/// An enum rather than a `bool` because the threshold it selects differs by a
/// factor of eight, and at a call site `true` says nothing about which way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RowOrigin {
    /// One sequence's consecutive positions: a prefill chunk, or a speculative
    /// verify.
    ///
    /// The default because it is the only origin a pass of one row can have,
    /// and one row is what most passes are.
    #[default]
    Sequential,
    /// One row per sequence, from sequences that have nothing to do with each
    /// other.
    Unrelated,
    /// Several sequences' runs of consecutive positions, one run each, in the
    /// order of the pass's sessions: a speculative verify for a batch of
    /// sequences, which reads the weights once for all of them.
    ///
    /// Every launch that carries a sequence's state runs once per run, as the
    /// scalar kernel over that run's rows and that sequence's state - which is
    /// what makes each run's rows the arithmetic its own verify pass would do.
    Segmented(Segments),
}

/// The runs of a segmented pass: how many consecutive rows each session has,
/// in pass order.
///
/// A fixed array rather than a vector so the origin stays `Copy`: at most one
/// run per decode-batch row, each at most a verify's width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Segments {
    lens: [u8; crate::sparse::stack::MAX_DECODE_BATCH as usize],
    count: u8,
}

impl Segments {
    /// The runs, in pass order.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for no runs, more runs than a decode batch
    /// holds, an empty run, or one wider than a byte counts.
    pub fn new(lens: &[u32]) -> CoreResult<Self> {
        let mut out = Self::default();
        if lens.is_empty() || lens.len() > out.lens.len() {
            return Err(CoreError::ShapeMismatch {
                field: "segmented pass runs",
                expected: out.lens.len() as u64,
                got: lens.len() as u64,
            });
        }
        for (slot, len) in out.lens.iter_mut().zip(lens) {
            *slot =
                u8::try_from(*len)
                    .ok()
                    .filter(|len| *len > 0)
                    .ok_or(CoreError::ShapeMismatch {
                        field: "segmented pass run rows",
                        expected: u64::from(u8::MAX),
                        got: u64::from(*len),
                    })?;
        }
        out.count = u8::try_from(lens.len()).unwrap_or(u8::MAX);
        Ok(out)
    }

    /// Runs in the pass.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count as u32
    }

    /// Rows run `run` covers.
    #[must_use]
    pub fn len(&self, run: u32) -> u32 {
        self.lens.get(run as usize).map_or(0, |len| u32::from(*len))
    }

    /// The pass row run `run` starts at.
    #[must_use]
    pub fn start(&self, run: u32) -> u32 {
        (0..run).map(|earlier| self.len(earlier)).sum()
    }

    /// Rows in the whole pass.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.start(self.count())
    }

    /// The run lengths as the segmented kernels take them: one byte per run,
    /// run 0 in the low byte.
    ///
    /// By value, because a pass is at most eight runs of at most a byte's rows
    /// each - which [`Segments::new`] enforces - so the table fits one kernel
    /// argument and needs no buffer the host fills before every pass.
    #[must_use]
    pub fn packed_lens(&self) -> u64 {
        self.lens
            .iter()
            .take(self.count as usize)
            .enumerate()
            .fold(0, |packed, (run, len)| {
                packed | (u64::from(*len) << (8 * run))
            })
    }
}

impl RowOrigin {
    /// Whether a chunk of `n_tok` rows groups its columns by expert.
    #[must_use]
    pub const fn groups(self, n_tok: u32) -> bool {
        match self {
            Self::Sequential => n_tok >= GROUP_MIN_ROWS,
            Self::Unrelated | Self::Segmented(_) => n_tok >= GROUP_MIN_UNRELATED_ROWS,
        }
    }
}

/// The router's own stage: one projection, then the selection.
///
/// The selection kernel stages the whole logit row in shared memory, one float
/// per expert, which is why the expert count reaches its launch.
///
/// # Errors
///
/// Whatever [`matmul::project`] reports for the router's projection.
pub fn route(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let mut plan = router_logits(desc, layer, n_tok, device)?;
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_route_kernel"),
        kernels::moe_route(shape.moe.experts, n_tok),
        // `ids, weights, logits, n_expert, n_used`
        Args::of(&[
            Operand::Scratch(Buf::MoeIds),
            Operand::Scratch(Buf::MoeWeights),
            Operand::Scratch(Buf::MoeRouter),
            Operand::U32(shape.moe.experts),
            Operand::U32(shape.moe.used),
        ]),
    );
    Ok(plan)
}

/// The router's projection alone, into [`Buf::MoeRouter`].
///
/// # Errors
///
/// Whatever [`matmul::project`] reports.
pub fn router_logits(
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let mut plan = Plan::new();
    matmul::project(
        &mut plan,
        desc.router,
        Bound::new(
            WeightRef::layer(layer, Role::Router),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::MoeRouter),
        ),
        n_tok,
        device,
    )?;
    Ok(plan)
}

/// Tokens up to which the routed selection, the sort and its inverse run as
/// one launch (`Q4E_ROUTE_SORT_TOKENS`).
pub const ROUTE_SORT_MAX_TOKENS: u32 = 32;
/// Experts the one-launch selection holds in registers
/// (`Q4E_ROUTE_SORT_EXPERTS`).
const ROUTE_SORT_MAX_EXPERTS: u32 = 512;

/// Whether a pass takes [`route_sort`].
#[must_use]
pub const fn route_sorts(shape: &SparseShape, n_tok: u32) -> bool {
    n_tok <= ROUTE_SORT_MAX_TOKENS
        && shape.moe.experts <= ROUTE_SORT_MAX_EXPERTS
        && shape.moe.experts.is_multiple_of(32)
        && shape.moe.used <= 16
}

/// The routed selection, the sort by expert and the sort's inverse, as one
/// block: the same ids, weights, bounds and active list as `route`'s selection
/// followed by [`sort`], and the inverse the decode-shaped prep reads.
///
/// At a decode step those were seven launches of a block or less each, 17 us a
/// layer of dependent latency in front of the routed projections.
#[must_use]
pub fn route_sort(shape: &SparseShape, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_route_sort_small_kernel"),
        kernels::Launch::linear(1, 32 * ROUTE_SORT_MAX_TOKENS),
        // `ids, weights, logits, bounds, active, sorted, pos, n_tok,
        //  n_expert, n_used`
        Args::of(&[
            Operand::Scratch(Buf::MoeIds),
            Operand::Scratch(Buf::MoeWeights),
            Operand::Scratch(Buf::MoeRouter),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeWsPos),
            Operand::U32(n_tok),
            Operand::U32(shape.moe.experts),
            Operand::U32(shape.moe.used),
        ]),
    );
    plan
}

/// The sort that groups the routed columns by expert.
///
/// Three passes - count, scan, scatter - over the columns rather than over the
/// tokens, because a column is what an expert receives. The scan is one warp in
/// one block: it runs over the expert count and is serial by construction.
///
/// The per-expert count row is the scatter's cursor as well as the count pass's
/// destination: the scan has already read it into the bounds by then, and the
/// scatter needs one counter per expert and nothing else.
#[must_use]
pub fn sort(shape: &SparseShape, n_tok: u32) -> Plan {
    let rows = shape.moe.slot_rows(n_tok);
    let columns = u32::try_from(rows).unwrap_or(u32::MAX);
    let mut plan = Plan::new();
    // The histogram is accumulated with `atomicAdd` and the buffer outlives the
    // layer, so without this the counts of every earlier layer are still in it:
    // expert spans grow past the staging a group has, and the fused launch
    // starts producing NaN rows around the fifth routed block.
    plan.push(Step::Clear {
        buffer: Operand::Scratch(Buf::MoeCounts),
    });
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_sort_count_kernel"),
        kernels::moe_sort_count(rows),
        // `counts, ids, n_rows, n_expert`
        Args::of(&[
            Operand::Scratch(Buf::MoeCounts),
            Operand::Scratch(Buf::MoeIds),
            Operand::U32(columns),
            Operand::U32(shape.moe.experts),
        ]),
    );
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_sort_scan_kernel"),
        kernels::moe_sort_scan(),
        // `bounds, active, counts, n_expert`
        Args::of(&[
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            Operand::Scratch(Buf::MoeCounts),
            Operand::U32(shape.moe.experts),
        ]),
    );
    // The same row is the scatter's per-expert cursor. The scan has read the
    // counts into the bounds by now, so zeroing here is what makes the cursor a
    // cursor rather than the histogram it was a moment ago.
    plan.push(Step::Clear {
        buffer: Operand::Scratch(Buf::MoeCounts),
    });
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_sort_scatter_kernel"),
        kernels::moe_sort_scatter(rows),
        // `sorted, cursor, bounds, ids, n_rows, n_expert`
        Args::of(&[
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeCounts),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeIds),
            Operand::U32(columns),
            Operand::U32(shape.moe.experts),
        ]),
    );
    plan
}

/// The stored element type ids the routed down projection was instantiated for.
///
/// Two, and they are the two an expert stack is shipped in when it is not a
/// trellis. A third would be a template instantiation this build does not
/// carry, which is why this is a lookup that can fail rather than a cast.
#[must_use]
pub const fn down_template_type(dtype: DType) -> Option<u32> {
    match dtype {
        DType::Q5_1 => Some(7),
        DType::Q8_0 => Some(8),
        _ => None,
    }
}

/// The experts' own stage, for a trellis stack: one cooperative launch.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when no built tiling shape divides the expert
/// weights, and whatever [`exl3::check_shape`] reports for a weight the codec
/// cannot tile at all.
pub fn fused_experts(
    shape: &SparseShape,
    layer: LayerRef,
    bits: u32,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    exl3::check_shape(shape.hidden, shape.moe.intermediate, bits)?;
    let fanout = Fanout::Grouped {
        slots: shape.moe.used,
        input_per_slot: true,
    };
    let tiling = exl3::select_shape(shape.hidden, shape.moe.intermediate, bits, fanout).ok_or(
        CoreError::Unsupported {
            what: "an expert stack no built tiling shape divides",
            backend: "engine",
        },
    )?;
    let grid = exl3::routed_grid(
        device.max_co_resident,
        device.max_co_resident,
        shape.moe.experts,
    );
    // The staging is per group and per token, and an undersized one is silent
    // out-of-bounds corruption rather than a failed launch - so its three
    // buffers are carved at offsets derived from the grid rather than guessed
    // at. The reservation itself belongs to the pass, not to this launch: it
    // cannot be made while a stream is capturing, and this launch sits inside
    // an island that may be.
    let staging = exl3::routed_staging(grid, n_tok, shape.hidden, shape.moe.intermediate);
    let at = |role: Role, span| Operand::Weight(WeightRef::layer(layer, role).spanning(span));
    let width = |value: u32| Operand::I32(i32::try_from(value).unwrap_or(i32::MAX));
    let mut plan = Plan::new();
    plan.push(Step::Trellis {
        op: TrellisOp::Routed,
        entry: Entry::template(MOD_EXL3, kernels::EXL3_MOE, TemplateArgs::TrellisBits(bits)),
        shape: tiling,
        bits,
        blocks: grid.group_size,
        groups: grid.groups,
        // The kernel's own width, not the tiling shape's: this entry point is
        // `__launch_bounds__` at `EXL3_MOE_BLOCK_DIM` and divides its work by
        // that number, so a narrower block from a shape that happens to carry
        // one would leave most of each tile unwritten.
        block_dim: kernels::EXL3_MOE_BLOCK_DIM,
        smem: kernels::EXL3_SMEM_MAX,
        staging: exl3::Staging::of(staging.bytes),
        // `hidden_state, temp_state, temp_intermediate, temp_act, slot_out,
        //  gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh,
        //  down_trellis, down_suh, down_svh, expert_bounds, slot_sorted,
        //  num_tokens, hidden_dim, intermediate_dim, num_experts,
        //  num_experts_per_tok, locks`
        //
        // The routing is the expert-sorted column map the sort stage built:
        // expert `e` owns the sorted slots between `bounds[e]` and
        // `bounds[e + 1]`, and the launch writes each assignment's unweighted
        // contribution at its own slot, which is what lets the combine apply
        // the routing weights in a fixed order.
        args: Args::of(&[
            Operand::Scratch(Buf::Mixed),
            Operand::CoopStaging(staging.state),
            Operand::CoopStaging(staging.intermediate),
            Operand::CoopStaging(staging.act),
            Operand::Scratch(Buf::MoeDown),
            at(Role::ExpertGate, WeightPart::Tiles),
            at(Role::ExpertGate, WeightPart::InputScales),
            at(Role::ExpertGate, WeightPart::OutputScales),
            at(Role::ExpertUp, WeightPart::Tiles),
            at(Role::ExpertUp, WeightPart::InputScales),
            at(Role::ExpertUp, WeightPart::OutputScales),
            at(Role::ExpertDown, WeightPart::Tiles),
            at(Role::ExpertDown, WeightPart::InputScales),
            at(Role::ExpertDown, WeightPart::OutputScales),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeSorted),
            width(n_tok),
            width(shape.hidden),
            width(shape.moe.intermediate),
            width(shape.moe.experts),
            width(shape.moe.used),
            Operand::CoopLocks,
        ]),
    });
    Ok(plan)
}

/// Rows up to which a target layer's routed block takes the grouped launches.
///
/// None do. Every target pass - a decode step, a verify pass, a decode batch
/// and a prefill chunk - takes the weight-stationary launches, whose per-row
/// arithmetic does not depend on how many rows share the pass: an m16 MMA row
/// accumulates over K in the same order whatever the other fifteen hold. So a
/// token decodes to the same logits alone, speculatively, or beside other
/// sequences. The grouped launches reduce in another order, and a plain step
/// through them drifts from a verify pass through these within a few hundred
/// tokens of greedy text (measured). What that costs is the plain single-row
/// step: 25.4 tok/s here against 29.2 grouped, on a GB10.
pub const GROUPED_MAX_ROWS: u32 = 0;

/// Rows up to which the draft head's routed block takes the grouped launches.
///
/// A draft's logits are a guess the target verifies, so no answer depends on
/// its bits and it keeps the faster single-row launches.
pub const DRAFT_GROUPED_MAX_ROWS: u32 = 1;

/// Rows up to which a weight set's routed block takes the grouped launches.
#[must_use]
pub const fn grouped_max(set: WeightSet) -> u32 {
    match set {
        WeightSet::Target => GROUPED_MAX_ROWS,
        WeightSet::Draft => DRAFT_GROUPED_MAX_ROWS,
    }
}

/// Rows from which the routed block takes the weight-stationary launches.
///
/// The same line the reference draws. Below it a chunk is a verify pass of two
/// or three rows, where the cooperative launch is already bound by the
/// weights it streams.
pub const WS_MIN_ROWS: u32 = 4;

/// The experts' own stage for a prefill chunk: three ordinary launches.
///
/// The fused cooperative launch streams every expert's weights once per 16 of
/// that expert's rows and pays a grid barrier between its phases. Here a block
/// owns one (expert, 128-column tile), decodes each weight tile once and
/// multiplies it against all of that expert's rows in 64-row passes. Measured
/// on this checkpoint at a 2048-row chunk on a GB10: 28 ms a layer fused
/// against 9.4 ms here; the chunk's result is the same up to the order of the
/// multiply-adds (relative L2 5e-5).
///
/// The launches are `prep` (each routed column's row, scaled and
/// Hadamard-transformed for gate and up, in expert order), `gateup` (both
/// projections, the activation, and the down projection's input transform as
/// the epilogue) and `down`, which stores each assignment's unweighted output
/// at its own slot - the layout [`combine`] sums in a fixed order, exactly as
/// it does for the fused launch.
///
/// # Errors
///
/// Whatever [`exl3::check_shape`] reports, and [`CoreError::ShapeMismatch`]
/// for widths the 128-column tiles and the 64-wide stages do not divide.
pub fn ws_experts(shape: &SparseShape, layer: LayerRef, bits: u32, n_tok: u32) -> CoreResult<Plan> {
    let hidden = shape.hidden;
    let inter = shape.moe.intermediate;
    exl3::check_shape(hidden, inter, bits)?;
    let tile = kernels::EXL3_MOE_WS_TILE_N;
    if !hidden.is_multiple_of(tile) || !inter.is_multiple_of(tile) {
        return Err(CoreError::ShapeMismatch {
            field: "weight-stationary expert tile",
            expected: u64::from(tile),
            got: u64::from(if hidden.is_multiple_of(tile) {
                inter
            } else {
                hidden
            }),
        });
    }
    let at = |role: Role, span| Operand::Weight(WeightRef::layer(layer, role).spanning(span));
    let width = |value: u32| Operand::I32(i32::try_from(value).unwrap_or(i32::MAX));
    let threads = kernels::EXL3_MOE_WS_THREADS;
    let rows = n_tok.saturating_mul(shape.moe.used);
    let experts = shape.moe.experts;
    let bits_key = TemplateArgs::TrellisBits(bits);
    let mut plan = Plan::new();
    // The sort's inverse, then one warp per (token, 128 columns of the input),
    // which reads each token once and writes every expert row it routed to.
    plan.launch(
        Entry::template(
            MOD_EXL3,
            kernels::EXL3_MOE_WS_INVERT,
            TemplateArgs::Untemplated,
        ),
        kernels::Launch::linear(
            kernels::blocks_for(u64::from(rows), u64::from(threads)),
            threads,
        ),
        // `pos, slot_sorted, n_rows`
        Args::of(&[
            Operand::Scratch(Buf::MoeWsPos),
            Operand::Scratch(Buf::MoeSorted),
            width(rows),
        ]),
    );
    let warps = u64::from(n_tok) * u64::from(hidden / tile);
    plan.launch(
        Entry::template(
            MOD_EXL3,
            kernels::EXL3_MOE_WS_PREP_TOK,
            TemplateArgs::Untemplated,
        ),
        kernels::Launch::linear(kernels::blocks_for(warps, u64::from(threads / 32)), threads),
        // `x, a_g, a_u, suh_g, suh_u, ids, pos, n_tok, n_used, k`
        Args::of(&[
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::MoeWsGate),
            Operand::Scratch(Buf::MoeWsUp),
            at(Role::ExpertGate, WeightPart::InputScales),
            at(Role::ExpertUp, WeightPart::InputScales),
            Operand::Scratch(Buf::MoeIds),
            Operand::Scratch(Buf::MoeWsPos),
            width(n_tok),
            width(shape.moe.used),
            width(hidden),
        ]),
    );
    plan.launch(
        Entry::template(MOD_EXL3, kernels::EXL3_MOE_WS_GATEUP, bits_key),
        kernels::Launch::new(
            shiso_core::Extent3::linear(experts.saturating_mul(inter / tile)),
            threads,
            kernels::exl3_moe_ws_smem(bits, true),
        ),
        // `a_g, a_u, act, gate_trellis, gate_svh, up_trellis, up_svh,
        //  down_suh, bounds, hidden, inter`
        Args::of(&[
            Operand::Scratch(Buf::MoeWsGate),
            Operand::Scratch(Buf::MoeWsUp),
            Operand::Scratch(Buf::MoeWsAct),
            at(Role::ExpertGate, WeightPart::Tiles),
            at(Role::ExpertGate, WeightPart::OutputScales),
            at(Role::ExpertUp, WeightPart::Tiles),
            at(Role::ExpertUp, WeightPart::OutputScales),
            at(Role::ExpertDown, WeightPart::InputScales),
            Operand::Scratch(Buf::MoeBounds),
            width(hidden),
            width(inter),
        ]),
    );
    plan.launch(
        Entry::template(MOD_EXL3, kernels::EXL3_MOE_WS_DOWN_H, bits_key),
        kernels::Launch::new(
            shiso_core::Extent3::linear(experts.saturating_mul(hidden / tile)),
            threads,
            kernels::exl3_moe_ws_smem(bits, false),
        ),
        // `act, slot_out, down_trellis, down_svh, bounds, slot_sorted, hidden,
        //  inter`
        Args::of(&[
            Operand::Scratch(Buf::MoeWsAct),
            Operand::Scratch(Buf::MoeDown),
            at(Role::ExpertDown, WeightPart::Tiles),
            at(Role::ExpertDown, WeightPart::OutputScales),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeSorted),
            width(hidden),
            width(inter),
        ]),
    );
    Ok(plan)
}

/// Rows up to which the routed block takes the decode-shaped launches.
///
/// They are the weight-stationary GEMMs cut into one n16 column group a warp
/// over the *active* experts, with the Hadamard epilogues in launches of their
/// own, and a pass through them is bit-identical to one through
/// [`ws_experts`] - so this line is drawn on speed alone. The weight-stationary
/// grid is sized by the expert count and holds 64-row staging: one token's
/// fifty busy blocks of 2,560 ran two waves on forty-eight SMs, at 55% of the
/// bandwidth the layer's weights cost (measured on a GB10, 185 us a layer).
/// Past a real chunk the 64-row passes win, as each expert has rows enough to
/// fill them.
pub const WSD_MAX_ROWS: u32 = 32;

/// The experts' own stage for a decode step, a verify pass or a decode batch:
/// the same prep as [`ws_experts`], then the decode-shaped gate and up, their
/// epilogue, and the down projection. [`wsd_combine`] finishes it. The sort's
/// inverse is [`route_sort`]'s, which every pass this takes runs first.
///
/// # Errors
///
/// As [`ws_experts`].
#[expect(
    clippy::too_many_lines,
    reason = "one statement per kernel argument is the point: these bodies are \
              transcriptions of parameter lists, and a split that broke a list \
              across two functions would hide the order being transcribed"
)]
pub fn wsd_experts(
    shape: &SparseShape,
    layer: LayerRef,
    bits: u32,
    n_tok: u32,
) -> CoreResult<Plan> {
    let hidden = shape.hidden;
    let inter = shape.moe.intermediate;
    exl3::check_shape(hidden, inter, bits)?;
    let tile = kernels::EXL3_MOE_WS_TILE_N;
    let group = 16 * kernels::EXL3_MOE_WSD_WARPS;
    if !hidden.is_multiple_of(tile) || !inter.is_multiple_of(tile) || !tile.is_multiple_of(group) {
        return Err(CoreError::ShapeMismatch {
            field: "decode-shaped expert tile",
            expected: u64::from(tile),
            got: u64::from(if hidden.is_multiple_of(tile) {
                inter
            } else {
                hidden
            }),
        });
    }
    let at = |role: Role, span| Operand::Weight(WeightRef::layer(layer, role).spanning(span));
    let width = |value: u32| Operand::I32(i32::try_from(value).unwrap_or(i32::MAX));
    let threads = kernels::EXL3_MOE_WS_THREADS;
    let rows = n_tok.saturating_mul(shape.moe.used);
    let experts = shape.moe.experts;
    // No more experts can be busy than there are routed columns.
    let busy = rows.min(experts);
    if shape.moe.used > kernels::EXL3_MOE_WSD_MAX_USED {
        return Err(CoreError::ShapeMismatch {
            field: "decode-shaped combine slots",
            expected: u64::from(kernels::EXL3_MOE_WSD_MAX_USED),
            got: u64::from(shape.moe.used),
        });
    }
    let bits_key = TemplateArgs::TrellisBits(bits);
    let block = kernels::EXL3_MOE_WSD_WARPS * 32;
    let mut plan = Plan::new();
    let warps = u64::from(rows) * u64::from(hidden / tile);
    plan.launch(
        Entry::template(
            MOD_EXL3,
            kernels::EXL3_MOE_WSD_PREP,
            TemplateArgs::Untemplated,
        ),
        kernels::Launch::linear(kernels::blocks_for(warps, u64::from(threads / 32)), threads),
        // `x, a_g, a_u, suh_g, suh_u, ids, pos, n_tok, n_used, k`
        Args::of(&[
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::MoeWsGate),
            Operand::Scratch(Buf::MoeWsUp),
            at(Role::ExpertGate, WeightPart::InputScales),
            at(Role::ExpertUp, WeightPart::InputScales),
            Operand::Scratch(Buf::MoeIds),
            Operand::Scratch(Buf::MoeWsPos),
            width(n_tok),
            width(shape.moe.used),
            width(hidden),
        ]),
    );
    plan.launch(
        Entry::template(MOD_EXL3, kernels::EXL3_MOE_WSD_GATEUP, bits_key),
        kernels::Launch::new(
            shiso_core::Extent3::linear(busy.saturating_mul(inter / group)),
            block,
            kernels::exl3_moe_wsd_smem(bits, true),
        ),
        // `a_g, a_u, raw_g, raw_u, gate_trellis, up_trellis, bounds, active,
        //  n_expert, hidden, inter`
        Args::of(&[
            Operand::Scratch(Buf::MoeWsGate),
            Operand::Scratch(Buf::MoeWsUp),
            Operand::Scratch(Buf::MoeGate),
            Operand::Scratch(Buf::MoeUp),
            at(Role::ExpertGate, WeightPart::Tiles),
            at(Role::ExpertUp, WeightPart::Tiles),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            width(experts),
            width(hidden),
            width(inter),
        ]),
    );
    let act_warps = u64::from(rows) * u64::from(inter / tile);
    plan.launch(
        Entry::template(
            MOD_EXL3,
            kernels::EXL3_MOE_WSD_ACT,
            TemplateArgs::Untemplated,
        ),
        kernels::Launch::linear(
            kernels::blocks_for(act_warps, u64::from(threads / 32)),
            threads,
        ),
        // `raw_g, raw_u, act, gate_svh, up_svh, down_suh, ids, slot_sorted,
        //  bounds, n_expert, n_rows, inter`
        Args::of(&[
            Operand::Scratch(Buf::MoeGate),
            Operand::Scratch(Buf::MoeUp),
            Operand::Scratch(Buf::MoeWsAct),
            at(Role::ExpertGate, WeightPart::OutputScales),
            at(Role::ExpertUp, WeightPart::OutputScales),
            at(Role::ExpertDown, WeightPart::InputScales),
            Operand::Scratch(Buf::MoeIds),
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeBounds),
            width(experts),
            width(rows),
            width(inter),
        ]),
    );
    plan.launch(
        Entry::template(MOD_EXL3, kernels::EXL3_MOE_WSD_DOWN, bits_key),
        kernels::Launch::new(
            shiso_core::Extent3::linear(busy.saturating_mul(hidden / group)),
            block,
            kernels::exl3_moe_wsd_smem(bits, false),
        ),
        // `act, raw_d, down_trellis, bounds, active, n_expert, hidden, inter`
        Args::of(&[
            Operand::Scratch(Buf::MoeWsAct),
            Operand::Scratch(Buf::MoeDown),
            at(Role::ExpertDown, WeightPart::Tiles),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            width(experts),
            width(hidden),
            width(inter),
        ]),
    );
    Ok(plan)
}

/// The down epilogue and the weighted sum of [`wsd_experts`]' raw rows, with
/// the gated shared expert added in the same pass, as [`ws_combine`] adds it.
#[must_use]
pub fn wsd_combine(shape: &SparseShape, layer: LayerRef, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    let chunks = shape.hidden / kernels::EXL3_MOE_WS_TILE_N;
    plan.launch(
        Entry::template(
            MOD_EXL3,
            kernels::EXL3_MOE_WSD_COMBINE,
            TemplateArgs::Untemplated,
        ),
        // One warp per slot: the geometry refuses a count past the kernel's
        // static staging rather than leaving slots out of the sum.
        kernels::Launch::new(
            shiso_core::Extent3::new(chunks, n_tok, 1),
            32 * shape.moe.used,
            0,
        ),
        // `out, raw_d, down_svh, ids, pos, weights, shared, logit, n_embd,
        //  n_used`
        Args::of(&[
            Operand::Scratch(Buf::BlockOut),
            Operand::Scratch(Buf::MoeDown),
            Operand::Weight(
                WeightRef::layer(layer, Role::ExpertDown).spanning(WeightPart::OutputScales),
            ),
            Operand::Scratch(Buf::MoeIds),
            Operand::Scratch(Buf::MoeWsPos),
            Operand::Scratch(Buf::MoeWeights),
            Operand::Scratch(Buf::SharedOut),
            Operand::Scratch(Buf::SharedLogit),
            Operand::I32(i32::try_from(shape.hidden).unwrap_or(i32::MAX)),
            Operand::I32(i32::try_from(shape.moe.used).unwrap_or(i32::MAX)),
        ]),
    );
    plan
}

/// Rows from which the routed block takes the fused launch.
///
/// Below it - which is a decode step and nothing else - the block runs as two
/// grouped launches instead. See [`grouped_experts`] for why.
pub const FUSED_MIN_ROWS: u32 = 2;

/// The experts' own stage for a single row, as two grouped launches.
///
/// # Why one token does not want the fused launch
///
/// The fused kernel gives each expert a *group* of blocks and walks the groups
/// over the experts that received columns. Its group count comes from the
/// device - on a forty-eight-block device that is six groups of eight - so ten
/// active experts go round twice, and the second round runs four experts wide
/// on a device sized for six. Measured on this box against a standalone A/B of
/// the two paths on this checkpoint's widths: 165 us a layer fused, 97 us
/// grouped, where 97 us is what the 23 MB a layer streams costs at the memory
/// bandwidth. The reference draws the same line, at the same place, for the
/// same reason.
///
/// The grouped form gives each *slot* its own group, so every expert the token
/// routed to streams at once: the gate and the up projections are twenty slots
/// behind one launch and the down projection is ten behind another. It also
/// needs no expert-sorted column map, because a slot names its matrix directly
/// - so a decode step skips the sort's four launches as well.
///
/// # Why it is a single row only
///
/// A slot's input is one row. Several rows per slot would mean replicating the
/// activation once per (row, expert) before the launch and reading each
/// expert's weights once per column rather than once - which is the trade the
/// fused kernel exists to avoid, and at a prefill chunk it is hundreds of times
/// the traffic. So this is the decode arm, and [`block`] takes it on the row
/// count rather than offering it as a choice.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when no built tiling shape divides the expert
/// weights, and whatever [`exl3::check_shape`] reports for a weight the codec
/// cannot tile at all.
#[expect(
    clippy::too_many_lines,
    reason = "one statement per kernel argument is the point: these bodies are \
              transcriptions of parameter lists, and a split that broke a list \
              across two functions would hide the order being transcribed"
)]
pub fn grouped_experts(
    shape: &SparseShape,
    layer: LayerRef,
    bits: u32,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let hidden = shape.hidden;
    let inter = shape.moe.intermediate;
    let used = shape.moe.used;
    let columns = used.saturating_mul(n_tok);
    exl3::check_shape(hidden, inter, bits)?;
    exl3::check_shape(inter, hidden, bits)?;
    let at = |role: Role, span| Operand::Weight(WeightRef::layer(layer, role).spanning(span));
    let width = |value: u32| Operand::I32(i32::try_from(value).unwrap_or(i32::MAX));
    let unsupported = || CoreError::Unsupported {
        what: "an expert stack no built tiling shape divides",
        backend: "engine",
    };
    let mut plan = Plan::new();

    // A single row's slots all read that row. Several rows' slots each read
    // their own token's row, replicated once per slot, because that is the one
    // input layout the launch takes besides a shared one - and a replica is the
    // same bytes, so a row's slot computes exactly what it would alone.
    let input = if n_tok == 1 {
        Operand::Scratch(Buf::Mixed)
    } else {
        let n = u64::from(columns) * u64::from(hidden);
        plan.launch(
            Entry::plain(MOD_MOE, "q4e_moe_rows_per_slot_kernel"),
            kernels::elementwise(n),
            // `out, x, in_dim, n_used, n`
            Args::of(&[
                Operand::Scratch(Buf::MoeRep),
                Operand::Scratch(Buf::Mixed),
                Operand::U32(hidden),
                Operand::U32(used),
                Operand::U64(n),
            ]),
        );
        Operand::Scratch(Buf::MoeRep)
    };

    // Gate and up share their input and their slot list, so they are one launch
    // over twice the slots: slot `j` past the split is the second tensor's slot
    // `j - split`, reading the same expert id and writing the other output.
    //
    // The tiling and the blocks per slot are the ones a single row's launch
    // gets, whatever the row count, and the extra slots run in rounds: see
    // [`exl3::mgemm_grid_rounds`].
    let basis = used.saturating_mul(2);
    let slots = columns.saturating_mul(2);
    let pair = Fanout::Grouped {
        slots: basis,
        input_per_slot: false,
    };
    let tiling = exl3::select_shape(hidden, inter, bits, pair).ok_or_else(unsupported)?;
    let grid = exl3::mgemm_grid_rounds(hidden, inter, tiling, slots, basis, device.max_co_resident);
    plan.push(Step::Trellis {
        op: TrellisOp::Pair,
        entry: matmul::trellis_entry(kernels::EXL3_MGEMM, bits, tiling),
        shape: tiling,
        bits,
        blocks: grid.per_group,
        groups: grid.concurrency,
        block_dim: tiling.block_dim(),
        smem: kernels::EXL3_SMEM_MAX,
        // One transformed slab per slot, both tensors' slots counted: the
        // kernel indexes this by the *global* slot, so sizing it for one
        // tensor's worth is a write past the end rather than a failed launch.
        staging: exl3::Staging::hadamard(slots, 1, hidden),
        // `A, B_base, B_stride, C, size_m, size_k, size_n, locks, suh_base,
        //  A_had, svh_base, B_indices, bszm_in, bszm_out, B_base2, suh_base2,
        //  svh_base2, C2, split`
        args: Args::of(&[
            input,
            at(Role::ExpertGate, WeightPart::Tiles),
            Operand::U64(exl3::expert_tile_stride(hidden, inter, bits)),
            Operand::Scratch(Buf::MoeGate),
            Operand::U32(1),
            Operand::U32(hidden),
            Operand::U32(inter),
            Operand::CoopLocks,
            at(Role::ExpertGate, WeightPart::InputScales),
            Operand::CoopStaging(0),
            at(Role::ExpertGate, WeightPart::OutputScales),
            Operand::Scratch(Buf::MoeIds),
            // One input shared by every slot, or one per slot.
            Operand::I32(if n_tok == 1 {
                1
            } else {
                i32::try_from(columns).unwrap_or(i32::MAX)
            }),
            width(slots),
            at(Role::ExpertUp, WeightPart::Tiles),
            at(Role::ExpertUp, WeightPart::InputScales),
            at(Role::ExpertUp, WeightPart::OutputScales),
            Operand::Scratch(Buf::MoeUp),
            width(columns),
        ]),
    });

    plan.append(activate(
        shape,
        Operand::Scratch(Buf::MoeGate),
        Operand::Scratch(Buf::MoeUp),
        Operand::Scratch(Buf::MoeMid),
        n_tok,
    ));

    // The down projection is one tensor, and its slots do *not* share an input:
    // each reads the row its own expert activated.
    let down = Fanout::Grouped {
        slots: used,
        input_per_slot: true,
    };
    let tiling = exl3::select_shape(inter, hidden, bits, down).ok_or_else(unsupported)?;
    let grid =
        exl3::mgemm_grid_rounds(inter, hidden, tiling, columns, used, device.max_co_resident);
    plan.push(Step::Trellis {
        op: TrellisOp::Fanout,
        entry: matmul::trellis_entry(kernels::EXL3_MGEMM, bits, tiling),
        shape: tiling,
        bits,
        blocks: grid.per_group,
        groups: grid.concurrency,
        block_dim: tiling.block_dim(),
        smem: kernels::EXL3_SMEM_MAX,
        staging: exl3::Staging::hadamard(columns, 1, inter),
        // The same argument block with no second tensor: the null base is what
        // makes the split unreachable, and the split is still the slot count so
        // that a read of it cannot address past the list.
        args: Args::of(&[
            Operand::Scratch(Buf::MoeMid),
            at(Role::ExpertDown, WeightPart::Tiles),
            Operand::U64(exl3::expert_tile_stride(inter, hidden, bits)),
            Operand::Scratch(Buf::MoeDown),
            Operand::U32(1),
            Operand::U32(inter),
            Operand::U32(hidden),
            Operand::CoopLocks,
            at(Role::ExpertDown, WeightPart::InputScales),
            Operand::CoopStaging(0),
            at(Role::ExpertDown, WeightPart::OutputScales),
            Operand::Scratch(Buf::MoeIds),
            width(columns),
            width(columns),
            Operand::Null,
            Operand::Null,
            Operand::Null,
            Operand::Null,
            width(columns),
        ]),
    });
    Ok(plan)
}

/// Output rows one affine expert threadgroup covers (`Q4E_AFFINE_ROWS`).
const AFFINE_EXPERT_ROWS: u32 = 8;

/// Weight rows one tile of the routed prompt kernels covers
/// (`Q4E_GEMM_EXPERT_ROWS`).
const AFFINE_GEMM_EXPERT_ROWS: u32 = 64;

/// Output rows one threadgroup of the affine routed down projection covers.
///
/// A contraction that is a multiple of 128 up to 1024 - the intermediate
/// width, 640 here - is split by rows, sixteen a threadgroup
/// (`Q4E_DOWN_SPLIT_ROWS`); any other takes the dense layout's eight.
#[must_use]
pub const fn affine_down_rows(in_dim: u32) -> u32 {
    if in_dim.is_multiple_of(128) && in_dim <= 1024 {
        16
    } else {
        AFFINE_EXPERT_ROWS
    }
}

/// The expert stages over an affine or a block-scaled FP8 bank: gate and up in
/// one launch over the sorted columns, the activation, and the down
/// projection, every stage reaching an expert's matrices through the layer's
/// expert-to-slot map.
///
/// The grid's second axis is one threadgroup per active expert, bounded by
/// the expert count and by the column count, whichever is smaller; the tail
/// of it returns once the active list runs out.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the three banks are not all affine at the
/// widths the expert shape names.
#[expect(
    clippy::too_many_lines,
    reason = "one statement per kernel argument is the point: these bodies are \
              transcriptions of parameter lists, two formats' worth"
)]
pub fn affine_experts(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let hidden = shape.hidden;
    let inter = shape.moe.intermediate;
    let widths = [
        (desc.gate, hidden, inter),
        (desc.up, hidden, inter),
        (desc.down, inter, hidden),
    ];
    let fp8 = desc.gate.dtype == DType::Fp8E4M3Block128;
    if widths.iter().any(|(w, i, o)| {
        !(w.dtype.is_affine() || fp8)
            || w.dtype != desc.gate.dtype
            || w.in_dim != *i
            || w.out_dim != *o
    }) {
        return Err(CoreError::Unsupported {
            what: "a routed expert stage whose three banks disagree on format or width",
            backend: "engine",
        });
    }
    let columns = u32::try_from(shape.moe.slot_rows(n_tok)).unwrap_or(u32::MAX);
    let groups = columns.min(shape.moe.experts);
    let at = |role: Role, span| Operand::Weight(WeightRef::layer(layer, role).spanning(span));
    let map = Operand::ExpertMap(layer.index);
    // The block-scaled kernels run a simdgroup per output row; the affine
    // ones four rows a simdgroup.
    let launch = |out_dim: u32| {
        kernels::Launch::new(
            shiso_core::Extent3::new(out_dim.div_ceil(AFFINE_EXPERT_ROWS), groups, 1),
            if fp8 {
                32 * AFFINE_EXPERT_ROWS
            } else {
                matmul::AFFINE_THREADS
            },
            0,
        )
    };
    let mut plan = Plan::new();
    let tail = |in_dim: u32, out_dim: u32| {
        [
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            map,
            Operand::U32(in_dim),
            Operand::U32(out_dim),
            Operand::U32(shape.moe.experts),
            Operand::U32(shape.moe.used),
        ]
    };
    if fp8 {
        // `out_gate, out_up, gate_w, gate_s, up_w, up_s, x, sorted, bounds,
        //  active, map, in_dim, out_dim, n_expert, n_used`
        let mut ops = vec![
            Operand::Scratch(Buf::MoeGate),
            Operand::Scratch(Buf::MoeUp),
            at(Role::ExpertGate, WeightPart::Whole),
            at(Role::ExpertGate, WeightPart::BlockScales),
            at(Role::ExpertUp, WeightPart::Whole),
            at(Role::ExpertUp, WeightPart::BlockScales),
            Operand::Scratch(Buf::Mixed),
        ];
        ops.extend(tail(hidden, inter));
        plan.launch(
            Entry::plain(MOD_MOE, "q4e_fp8_experts_gate_up_kernel"),
            launch(inter),
            Args::of(&ops),
        );
        plan.append(activate(
            shape,
            Operand::Scratch(Buf::MoeGate),
            Operand::Scratch(Buf::MoeUp),
            Operand::Scratch(Buf::MoeMid),
            n_tok,
        ));
        // `out, down_w, down_s, mid, sorted, bounds, active, map, in_dim,
        //  out_dim, n_expert, n_used`
        let mut ops = vec![
            Operand::Scratch(Buf::MoeDown),
            at(Role::ExpertDown, WeightPart::Whole),
            at(Role::ExpertDown, WeightPart::BlockScales),
            Operand::Scratch(Buf::MoeMid),
        ];
        ops.extend(tail(inter, hidden));
        plan.launch(
            Entry::plain(MOD_MOE, "q4e_fp8_experts_down_kernel"),
            launch(hidden),
            Args::of(&ops),
        );
        return Ok(plan);
    }
    // A prompt pass takes its expert's columns 32 at a time on the matrix
    // units: a column's result then depends on the shape alone, which is the
    // prompt's promise, and the weights are staged once a tile rather than
    // once a column. A step keeps the one-column kernels its single-row
    // reference runs.
    let gemm = device.row_invariant();
    let tiled = |out_dim: u32| {
        kernels::Launch::new(
            shiso_core::Extent3::new(out_dim.div_ceil(AFFINE_GEMM_EXPERT_ROWS), groups, 1),
            matmul::AFFINE_GEMM_THREADS,
            0,
        )
    };
    plan.launch(
        Entry::plain(
            MOD_MOE,
            if gemm {
                "q4e_affine_experts_gemm_gate_up_kernel"
            } else {
                "q4e_affine_experts_gate_up_kernel"
            },
        ),
        if gemm {
            // The gate's tiles, then the up's.
            tiled(2 * inter.div_ceil(AFFINE_GEMM_EXPERT_ROWS) * AFFINE_GEMM_EXPERT_ROWS)
        } else {
            launch(inter)
        },
        // `out_gate, out_up, gate_w, gate_s, gate_b, up_w, up_s, up_b, x,
        //  sorted, bounds, active, map, in_dim, out_dim, n_expert, n_used`
        Args::of(&[
            Operand::Scratch(Buf::MoeGate),
            Operand::Scratch(Buf::MoeUp),
            at(Role::ExpertGate, WeightPart::Whole),
            at(Role::ExpertGate, WeightPart::AffineScales),
            at(Role::ExpertGate, WeightPart::AffineBiases),
            at(Role::ExpertUp, WeightPart::Whole),
            at(Role::ExpertUp, WeightPart::AffineScales),
            at(Role::ExpertUp, WeightPart::AffineBiases),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            map,
            Operand::U32(hidden),
            Operand::U32(inter),
            Operand::U32(shape.moe.experts),
            Operand::U32(shape.moe.used),
        ]),
    );
    plan.append(activate(
        shape,
        Operand::Scratch(Buf::MoeGate),
        Operand::Scratch(Buf::MoeUp),
        Operand::Scratch(Buf::MoeMid),
        n_tok,
    ));
    plan.launch(
        Entry::plain(
            MOD_MOE,
            if gemm {
                "q4e_affine_experts_gemm_down_kernel"
            } else {
                "q4e_affine_experts_down_kernel"
            },
        ),
        if gemm {
            tiled(hidden)
        } else {
            kernels::Launch::new(
                shiso_core::Extent3::new(hidden.div_ceil(affine_down_rows(inter)), groups, 1),
                matmul::AFFINE_THREADS,
                0,
            )
        },
        // `out, down_w, down_s, down_b, mid, sorted, bounds, active, map,
        //  in_dim, out_dim, n_expert, n_used`
        Args::of(&[
            Operand::Scratch(Buf::MoeDown),
            at(Role::ExpertDown, WeightPart::Whole),
            at(Role::ExpertDown, WeightPart::AffineScales),
            at(Role::ExpertDown, WeightPart::AffineBiases),
            Operand::Scratch(Buf::MoeMid),
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            map,
            Operand::U32(inter),
            Operand::U32(hidden),
            Operand::U32(shape.moe.experts),
            Operand::U32(shape.moe.used),
        ]),
    );
    Ok(plan)
}

/// The activation between the up projection and the down one.
///
/// `out, gate, up, n`. The three buffers are the caller's because this kernel
/// serves two stages with different ones: the routed stage runs it over every
/// routed column, and the shared expert over every token.
#[must_use]
pub fn activate(shape: &SparseShape, gate: Operand, up: Operand, out: Operand, n_tok: u32) -> Plan {
    let n = shape.moe.slot_rows(n_tok) * u64::from(shape.moe.intermediate);
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_swiglu_kernel"),
        kernels::elementwise(n),
        Args::of(&[out, gate, up, Operand::U64(n)]),
    );
    plan
}

/// The routed down projection, which is where the column grouping shows up.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a stored element type the down template was
/// not instantiated for.
pub fn down(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    origin: RowOrigin,
    n_tok: u32,
) -> CoreResult<Plan> {
    let stored = down_template_type(desc.down.dtype).ok_or(CoreError::Unsupported {
        what: "a routed down projection at a stored element type this build never instantiated",
        backend: "engine",
    })?;
    let rows = u32::try_from(shape.moe.slot_rows(n_tok)).unwrap_or(u32::MAX);
    let blocks = matmul::quant_blocks_per_row(shape.moe.intermediate)
        .and_then(|blocks| u32::try_from(blocks).ok())
        .ok_or(CoreError::ShapeMismatch {
            field: "expert intermediate width",
            expected: 32,
            got: u64::from(shape.moe.intermediate),
        })?;
    let expert_bytes = crate::sparse::weights::Weight::payload_bytes(
        desc.down.dtype,
        desc.down.in_dim,
        desc.down.out_dim,
    )
    .ok_or(CoreError::ShapeMismatch {
        field: "expert slab bytes",
        expected: 1,
        got: 0,
    })?;
    let expert_bytes = Operand::U32(u32::try_from(expert_bytes).unwrap_or(u32::MAX));
    let weight = Operand::Weight(WeightRef::layer(layer, Role::ExpertDown));
    let mut plan = Plan::new();
    if origin.groups(n_tok) {
        plan.launch(
            Entry::template(
                MOD_MOE,
                "q4e_moe_down_grouped_kernel",
                TemplateArgs::StoredTypeBlocks { stored, blocks },
            ),
            kernels::moe_down_grouped(shape.hidden, rows, shape.moe.experts),
            // `dst, W, x, sorted_cols, bounds, active, n_expert, out_dim,
            //  in_dim, expert_bytes`
            Args::of(&[
                Operand::Scratch(Buf::MoeDown),
                weight,
                Operand::Scratch(Buf::MoeMid),
                Operand::Scratch(Buf::MoeSorted),
                Operand::Scratch(Buf::MoeBounds),
                Operand::Scratch(Buf::MoeActive),
                Operand::U32(shape.moe.experts),
                Operand::U32(shape.hidden),
                Operand::U32(shape.moe.intermediate),
                expert_bytes,
            ]),
        );
    } else {
        plan.launch(
            Entry::template(
                MOD_MOE,
                "q4e_moe_down_kernel",
                TemplateArgs::StoredType(stored),
            ),
            kernels::moe_down(shape.hidden, rows),
            // `dst, W, x, ids, out_dim, in_dim, n_used, expert_bytes`
            Args::of(&[
                Operand::Scratch(Buf::MoeDown),
                weight,
                Operand::Scratch(Buf::MoeMid),
                Operand::Scratch(Buf::MoeIds),
                Operand::U32(shape.hidden),
                Operand::U32(shape.moe.intermediate),
                Operand::U32(shape.moe.used),
                expert_bytes,
            ]),
        );
    }
    Ok(plan)
}

/// The weighted sum of the weight-stationary block's half slot outputs, with
/// the shared expert's gated contribution added in the same pass.
///
/// The shared expert's projections have to have run first: this reads their
/// output and their gate logit, where [`shared`] adds them onto a combine that
/// already happened.
#[must_use]
pub fn ws_combine(shape: &SparseShape, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    let columns = u64::from(shape.hidden / 4);
    plan.launch(
        Entry::template(
            MOD_EXL3,
            kernels::EXL3_MOE_WS_COMBINE,
            TemplateArgs::Untemplated,
        ),
        kernels::Launch::new(
            shiso_core::Extent3::new(kernels::blocks_for(columns, 256), n_tok, 1),
            256,
            0,
        ),
        // `out, slot_out, weights, shared, logit, n_embd, n_used`
        Args::of(&[
            Operand::Scratch(Buf::BlockOut),
            Operand::Scratch(Buf::MoeDown),
            Operand::Scratch(Buf::MoeWeights),
            Operand::Scratch(Buf::SharedOut),
            Operand::Scratch(Buf::SharedLogit),
            Operand::I32(i32::try_from(shape.hidden).unwrap_or(i32::MAX)),
            Operand::I32(i32::try_from(shape.moe.used).unwrap_or(i32::MAX)),
        ]),
    );
    plan
}

/// The recombination of every column into one contribution per token.
#[must_use]
pub fn combine(shape: &SparseShape, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_moe_combine_kernel"),
        kernels::moe_combine(shape.hidden, n_tok),
        // `out, down, weights, n_embd, n_used`
        Args::of(&[
            Operand::Scratch(Buf::BlockOut),
            Operand::Scratch(Buf::MoeDown),
            Operand::Scratch(Buf::MoeWeights),
            Operand::U32(shape.hidden),
            Operand::U32(shape.moe.used),
        ]),
    );
    plan
}

/// The shared expert's projections: dense over every token, with a scalar gate
/// of its own, into its output and its gate logit.
///
/// # Errors
///
/// Whatever [`matmul::project_pair`] and [`matmul::project`] report.
pub fn shared_projections(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let mut plan = shared_expert(shape, desc, layer, n_tok, device)?;
    plan.append(shared_logit(desc, layer, n_tok, device)?);
    Ok(plan)
}

/// The shared expert's three projections and its activation, into
/// [`Buf::SharedOut`]: everything of [`shared_projections`] but the gate logit.
///
/// # Errors
///
/// Whatever [`matmul::project_pair`] and [`matmul::project`] report.
pub fn shared_expert(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let at = |role: Role| WeightRef::layer(layer, role);
    let n = u64::from(n_tok) * u64::from(shape.moe.intermediate);
    let mut plan = Plan::new();
    matmul::project_pair(
        &mut plan,
        desc.shared_gate,
        desc.shared_up,
        Bound::new(
            at(Role::SharedGate),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::SharedGate),
        ),
        Bound::new(
            at(Role::SharedUp),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::SharedUp),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_swiglu_kernel"),
        kernels::elementwise(n),
        // `out, gate, up, n`
        Args::of(&[
            Operand::Scratch(Buf::SharedMid),
            Operand::Scratch(Buf::SharedGate),
            Operand::Scratch(Buf::SharedUp),
            Operand::U64(n),
        ]),
    );
    matmul::project(
        &mut plan,
        desc.shared_down,
        Bound::new(
            at(Role::SharedDown),
            Operand::Scratch(Buf::SharedMid),
            Operand::Scratch(Buf::SharedOut),
        ),
        n_tok,
        device,
    )?;
    Ok(plan)
}

/// The shared expert's gate logit, into [`Buf::SharedLogit`].
///
/// # Errors
///
/// Whatever [`matmul::project`] reports.
pub fn shared_logit(
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let at = |role: Role| WeightRef::layer(layer, role);
    let mut plan = Plan::new();
    matmul::project(
        &mut plan,
        desc.shared_router,
        Bound::new(
            at(Role::SharedRouter),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::SharedLogit),
        ),
        n_tok,
        device,
    )?;
    Ok(plan)
}

/// The shared expert, added onto the routed contribution already in place.
///
/// # Errors
///
/// Whatever [`shared_projections`] reports.
pub fn shared(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let mut plan = shared_projections(shape, desc, layer, n_tok, device)?;
    plan.launch(
        Entry::plain(MOD_MOE, "q4e_shared_add_kernel"),
        kernels::moe_combine(shape.hidden, n_tok),
        // `out, shared, logit, n_embd`
        Args::of(&[
            Operand::Scratch(Buf::BlockOut),
            Operand::Scratch(Buf::SharedOut),
            Operand::Scratch(Buf::SharedLogit),
            Operand::U32(shape.hidden),
        ]),
    );
    Ok(plan)
}

/// One routed feed-forward, in launch order.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the expert stack is stored in a format whose
/// staged gate-and-up belongs to a kernel set this build does not carry, and
/// whatever the stages report.
pub fn block(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    if desc.gate.dtype.is_affine() || desc.gate.dtype == DType::Fp8E4M3Block128 {
        // A decode-sized pass routes and sorts in one launch, which is the
        // same answer as the four-launch route and sort bit for bit - so a
        // prompt pass narrow enough to take it still routes a row as a wider
        // one does - at a sixth of the dependent launches.
        let mut plan = if route_sorts(shape, n_tok) {
            let mut plan = router_logits(desc, layer, n_tok, device)?;
            plan.append(route_sort(shape, n_tok));
            plan
        } else {
            let mut plan = route(shape, desc, layer, n_tok, device)?;
            plan.append(sort(shape, n_tok));
            plan
        };
        // The target's banks may be streamed; a draft head's never are.
        if layer.set == WeightSet::Target {
            plan.push(Step::Fetch {
                layer: layer.index,
                ids: u32::try_from(shape.moe.slot_rows(n_tok)).unwrap_or(u32::MAX),
            });
        }
        plan.append(affine_experts(shape, desc, layer, n_tok, device)?);
        plan.append(combine(shape, n_tok));
        plan.append(shared(shape, desc, layer, n_tok, device)?);
        return Ok(plan);
    }
    let Some(bits) = matmul::trellis_bits(desc.gate.dtype) else {
        return Err(CoreError::Unsupported {
            what: "a staged expert gate and up projection, whose kernel is not in this kernel set",
            backend: "engine",
        });
    };
    // The shared expert reads only the block input and writes only its own
    // buffers, and on the decode-shaped launches its projections are latency
    // bound - 35 us a layer at a third of the bandwidth - so it runs as a
    // branch beside the router and the routed launches. Only when those carve
    // none of the per-stream scratch the branch's launches use: a router over
    // more than sixteen rows widens its weight into it outside a capture, and
    // the two corrupted each other. The gate logit is a half-precision
    // projection, whose split sums share a buffer with the router's, so it
    // stays on the main stream.
    let wsd = n_tok <= WSD_MAX_ROWS && route_sorts(shape, n_tok);
    if n_tok > grouped_max(layer.set)
        && matmul::pair_decoded(&desc.shared_gate, n_tok, device)
        && matmul::pair_decoded(&desc.shared_down, n_tok, device)
        && let Some(main) = {
            let main = if wsd {
                let mut main = router_logits(desc, layer, n_tok, device)?;
                main.append(route_sort(shape, n_tok));
                main.append(wsd_experts(shape, layer, bits, n_tok)?);
                main
            } else {
                let mut main = route(shape, desc, layer, n_tok, device)?;
                main.append(sort(shape, n_tok));
                main.append(ws_experts(shape, layer, bits, n_tok)?);
                main
            };
            (main.staging().bytes == 0).then_some(main)
        }
    {
        let mut plan = Plan::new();
        plan.push(Step::Branch(Branch::Fork));
        plan.append(shared_expert(shape, desc, layer, n_tok, device)?);
        plan.push(Step::Branch(Branch::Resume));
        plan.append(main);
        plan.push(Step::Branch(Branch::Join));
        plan.append(shared_logit(desc, layer, n_tok, device)?);
        plan.append(if wsd {
            wsd_combine(shape, layer, n_tok)
        } else {
            ws_combine(shape, n_tok)
        });
        return Ok(plan);
    }
    if n_tok <= grouped_max(layer.set) {
        let mut plan = route(shape, desc, layer, n_tok, device)?;
        plan.append(grouped_experts(shape, layer, bits, n_tok, device)?);
        plan.append(combine(shape, n_tok));
        plan.append(shared(shape, desc, layer, n_tok, device)?);
        return Ok(plan);
    }
    // The weight-stationary path writes its slot outputs as halves, and its
    // combine adds the shared expert in the same pass, so the shared expert's
    // projections run first.
    if wsd {
        let mut plan = router_logits(desc, layer, n_tok, device)?;
        plan.append(route_sort(shape, n_tok));
        plan.append(wsd_experts(shape, layer, bits, n_tok)?);
        plan.append(shared_projections(shape, desc, layer, n_tok, device)?);
        plan.append(wsd_combine(shape, layer, n_tok));
        return Ok(plan);
    }
    let mut plan = route(shape, desc, layer, n_tok, device)?;
    plan.append(sort(shape, n_tok));
    plan.append(ws_experts(shape, layer, bits, n_tok)?);
    plan.append(shared_projections(shape, desc, layer, n_tok, device)?);
    plan.append(ws_combine(shape, n_tok));
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_draft_heads_single_row_takes_the_grouped_launches() {
        // A target row must reduce in one order at every pass width; a draft's
        // guess is verified, so its one row keeps the faster launches.
        assert_eq!(grouped_max(WeightSet::Target), 0);
        assert_eq!(grouped_max(WeightSet::Draft), 1);
    }

    #[test]
    fn a_routed_column_is_one_token_and_one_expert_slot() {
        let moe = crate::sparse::shape::MoeShape::new(512, 10, 1, 640);
        assert_eq!(moe.slot_rows(1), 10);
        assert_eq!(moe.slot_rows(512), 5_120);
    }

    #[test]
    fn unrelated_rows_wait_for_a_real_chunk_before_they_group() {
        assert!(!RowOrigin::Sequential.groups(3));
        assert!(RowOrigin::Sequential.groups(4));
        assert!(!RowOrigin::Unrelated.groups(8));
        assert!(RowOrigin::Unrelated.groups(32));
    }

    #[test]
    fn a_down_projection_outside_the_instantiated_types_is_refused_by_name() {
        assert_eq!(down_template_type(DType::Q8_0), Some(8));
        assert_eq!(down_template_type(DType::Q5_1), Some(7));
        assert_eq!(down_template_type(DType::Q4_K), None);
    }
}

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
use crate::sparse::kernels::{
    self, Entry, MOD_EXL3, MOD_MODELOPT, MOD_MOE, MOD_NVFP4_A4, TemplateArgs,
};
use crate::sparse::matmul::{self, Bound, Device};
use crate::sparse::plan::{Branch, Plan, Step, TrellisOp};
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::stack::{MAX_DECODE_BATCH, MAX_PROMPT_RUNS};
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
#[expect(
    clippy::large_enum_variant,
    reason = "the origin is `Copy` so every attention context can carry it, and it \
              is copied into the plan of every layer of a pass - a few hundred bytes \
              a copy, a few dozen copies a pass, against launches that cost \
              microseconds each; a boxed run table would trade that for an \
              allocation per pass and a clone wherever a copy is taken now"
)]
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

/// The runs of a segmented pass: where each session's consecutive rows start,
/// in pass order.
///
/// Row offsets rather than lengths: a prompt-role pass reads each scoring
/// session's whole tail as one run - hundreds of rows, up to
/// [`MAX_PROMPT_RUNS`] runs - and what every consumer asks is where a run
/// starts or ends. A fixed array rather than a vector so the origin stays
/// `Copy`. A verify's kernels take the lengths packed into one argument
/// instead, which [`Segments::packed_lens`] produces and refuses past a
/// verify's own bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segments {
    /// `starts[r]` is run `r`'s first pass row; `starts[count]` is the pass's
    /// row count.
    starts: [u32; MAX_PROMPT_RUNS as usize + 1],
    count: u32,
}

impl Default for Segments {
    fn default() -> Self {
        Self {
            starts: [0; MAX_PROMPT_RUNS as usize + 1],
            count: 0,
        }
    }
}

impl Segments {
    /// The runs, in pass order.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for no runs, more than
    /// [`MAX_PROMPT_RUNS`], an empty run, or more rows than a `u32` counts.
    pub fn new(lens: &[u32]) -> CoreResult<Self> {
        let mut out = Self::default();
        if lens.is_empty() || lens.len() > MAX_PROMPT_RUNS as usize {
            return Err(CoreError::ShapeMismatch {
                field: "segmented pass runs",
                expected: u64::from(MAX_PROMPT_RUNS),
                got: lens.len() as u64,
            });
        }
        let mut at = 0_u32;
        for (slot, len) in out.starts.iter_mut().skip(1).zip(lens) {
            if *len == 0 {
                return Err(CoreError::ShapeMismatch {
                    field: "segmented pass run rows",
                    expected: 1,
                    got: 0,
                });
            }
            at = at.checked_add(*len).ok_or(CoreError::ShapeMismatch {
                field: "segmented pass rows",
                expected: u64::from(u32::MAX),
                got: u64::from(at) + u64::from(*len),
            })?;
            *slot = at;
        }
        out.count = u32::try_from(lens.len()).unwrap_or(MAX_PROMPT_RUNS);
        Ok(out)
    }

    /// Runs in the pass.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// The pass row run `run` starts at; the pass's row count past the last.
    #[must_use]
    pub fn start(&self, run: u32) -> u32 {
        self.starts
            .get(run.min(self.count) as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Rows run `run` covers, zero past the last.
    #[must_use]
    pub fn len(&self, run: u32) -> u32 {
        if run >= self.count {
            0
        } else {
            self.start(run + 1) - self.start(run)
        }
    }

    /// Rows in the whole pass.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.start(self.count)
    }

    /// The pass row of run `run`'s last position: the row its logits and its
    /// staged draft row come from.
    ///
    /// Every caller asks for a run below [`Segments::count`]. Past the last
    /// run this is not an error but the pass's own last row, which is the
    /// last run's, because [`Segments::start`] clamps to the pass's row count
    /// there and [`Segments::len`] is zero; zero for a pass with no rows.
    #[must_use]
    pub fn last_row(&self, run: u32) -> u32 {
        self.start(run)
            .saturating_add(self.len(run))
            .saturating_sub(1)
    }

    /// The run lengths as a verify's segmented kernels take them: one byte per
    /// run, run 0 in the low byte.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for more runs than a decode batch or a run
    /// wider than a byte counts: the table is one kernel argument, and a
    /// prompt-role pass never asks for it (its recurrences run per run).
    pub fn packed_lens(&self) -> CoreResult<u64> {
        if self.count > MAX_DECODE_BATCH {
            return Err(CoreError::Unsupported {
                what: "a segmented verify of more runs than one kernel argument packs",
                backend: "engine",
            });
        }
        (0..self.count).try_fold(0_u64, |packed, run| {
            let len = u8::try_from(self.len(run)).map_err(|_| CoreError::Unsupported {
                what: "a segmented verify run wider than one byte counts",
                backend: "engine",
            })?;
            Ok(packed | (u64::from(len) << (8 * run)))
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
/// Tokens up to which [`route_sort`] takes the tiny entry
/// (`Q4E_ROUTE_TINY_TOKENS`).
pub const ROUTE_TINY_MAX_TOKENS: u32 = 8;
/// Threads the tiny entry runs (`Q4E_ROUTE_TINY_THREADS`).
const ROUTE_TINY_THREADS: u32 = 256;
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
pub fn route_sort(shape: &SparseShape, n_tok: u32, device: Device) -> Plan {
    // The tiny entry is the small one's results at a shorter serial chain
    // (GB10: 12.7 -> 7.8 us at one token), for the widths it takes, on a
    // kernel set that has it.
    let (entry, threads) = if matches!(
        device.route_sort,
        matmul::RouteSort::Tiny | matmul::RouteSort::Fused
    ) && n_tok <= ROUTE_TINY_MAX_TOKENS
    {
        ("q4e_moe_route_sort_tiny_kernel", ROUTE_TINY_THREADS)
    } else {
        (
            "q4e_moe_route_sort_small_kernel",
            32 * ROUTE_SORT_MAX_TOKENS,
        )
    };
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_MOE, entry),
        kernels::Launch::linear(1, threads),
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

/// Rows up to which a step or verify pass can take [`fused_route`]
/// (`Q4E_FR_ROWS`): the tiny selection's width.
pub const FUSED_ROUTE_MAX_TOKENS: u32 = ROUTE_TINY_MAX_TOKENS;
/// Words of [`Buf::MoeSync`] the fused route launch orders its work through
/// (`Q4E_FR_SYNC_*`).
pub const FUSED_ROUTE_SYNC_WORDS: u32 = 4;
/// Warps a block of the fused route launch runs (`Q4E_FR_WARPS`): each item
/// is eight shared gate-and-up rows, eight slices of router rows, or sixteen
/// shared down rows.
const FUSED_ROUTE_WARPS: u32 = 8;
/// The widest block input the fused route launch stages a shared row's scales
/// for (`Q4E_FR_MX_SCALES` groups of 32).
const FUSED_ROUTE_MAX_IN: u32 = 5_120;
/// The contraction width the shared gate and up must be whole chunks of
/// (`Q4E_MX_CHUNK`).
const FUSED_ROUTE_IN_ALIGN: u32 = 512;
/// Rows of the shared down projection one fused route item covers: two a
/// warp (`q4e_mx_rows<NT, 2, 1>`).
const FUSED_ROUTE_DOWN_ROWS: u32 = 2 * FUSED_ROUTE_WARPS;
/// Dynamic shared memory a fused route block reserves and never touches: with
/// the 11 KB it declares, two blocks fill a GB10 multiprocessor's 100 KB.
///
/// A throttle, measured: at the three blocks an SM its registers allow, the
/// gate-and-up items' row pairs and the router's slices all in flight at once
/// queued the memory system so deep that every item took longer, and the
/// launch took 46.5 us at one row; at two, 39.3 us (two rows 41.6 -> 39.5).
/// From three rows on the verify entry's own registers hold it near two
/// blocks and the reservation measured no better (eight rows 54.2 -> 57.5,
/// with less of the multiprocessor left to the first-level cache the rows'
/// activations are read through), so only the passes of at most
/// [`FUSED_ROUTE_THROTTLED`] rows take it. A launch parameter, not a sum: no
/// value moves with it. Below the 48 KB a launch takes without opting in.
pub const FUSED_ROUTE_SMEM: u32 = 36_000;
/// The widest pass that takes [`FUSED_ROUTE_SMEM`].
pub const FUSED_ROUTE_THROTTLED: u32 = 2;

/// Whether a pass takes [`fused_route`] in place of the router's projection,
/// [`route_sort`], the shared expert's branch and its gate logit.
///
/// Only where it reproduces them bit for bit: a step or verify pass (a prompt
/// pass's router takes the tensor-core GEMM, whose sums differ) of at most
/// [`FUSED_ROUTE_MAX_TOKENS`] rows that [`route_sorts`], on a device whose
/// brain-float projections take [`kernels::MOD_BF16`]'s step kernels, with a
/// brain-float router and gate and an MXFP8 shared expert at the widths the
/// launch is written for. A pass that misses any of these keeps the separate
/// launches - and since both reproduce the same kernels' arithmetic, which
/// one a pass takes moves no bit.
#[must_use]
pub fn route_fuses(shape: &SparseShape, desc: &ExpertDesc, n_tok: u32, device: Device) -> bool {
    let hidden = shape.hidden;
    let inter = shape.moe.intermediate;
    let bf16 = |w: crate::sparse::weights::WeightDesc, out_dim: u32| {
        w.dtype == DType::BF16 && w.in_dim == hidden && w.out_dim == out_dim
    };
    let mx = |w: crate::sparse::weights::WeightDesc, in_dim: u32, out_dim: u32| {
        w.dtype == DType::Mxfp8E4M3G32 && w.in_dim == in_dim && w.out_dim == out_dim
    };
    device.route_sort == matmul::RouteSort::Fused
        && matches!(device.prompt_matmul, matmul::PromptMatmul::Tensor)
        && !device.row_invariant()
        && (1..=FUSED_ROUTE_MAX_TOKENS).contains(&n_tok)
        && route_sorts(shape, n_tok)
        && hidden.is_multiple_of(FUSED_ROUTE_IN_ALIGN)
        && hidden <= FUSED_ROUTE_MAX_IN
        && inter.is_multiple_of(32)
        && bf16(desc.router, shape.moe.experts)
        && bf16(desc.shared_router, 1)
        && mx(desc.shared_gate, hidden, inter)
        && mx(desc.shared_up, hidden, inter)
        && mx(desc.shared_down, inter, hidden)
}

/// The router's projection, the shared expert's gate logit, the routed
/// selection and sort, and the shared expert's gate, up, activation and down,
/// as one launch over the block input: what [`router_logits`], [`route_sort`],
/// [`shared_expert`] and [`shared_logit`] write, into the same buffers, bit
/// for bit. The caller checks [`route_fuses`] first.
///
/// At a decode step those were seven launches, two of them on a branch: the
/// router and the shared expert's gate and up side by side (33 and 35 us in
/// the model), the one-block selection behind the router (11 us), the
/// activation, the shared down projection, and the gate logit after the join.
/// Here they are the work items of one grid (`q4e_ffn_route_kernel`): the
/// router item that finishes last runs the selection while the shared down
/// items stream, and nothing the pass reads afterwards changes. GB10, kernel
/// bench over 24 layers' weights: one row 51.0 -> 39.3 us, two 59.8 -> 39.5,
/// five 65.1 -> 46.8, eight 74.1 -> 54.2.
#[must_use]
pub fn fused_route(shape: &SparseShape, desc: &ExpertDesc, layer: LayerRef, n_tok: u32) -> Plan {
    let hidden = shape.hidden;
    let inter = shape.moe.intermediate;
    // The slice counts the brain-float step kernels take for these weights
    // (`matmul::project` hands them the same), which fix each logit's sum.
    let splits = matmul::bf16_split(desc.router.in_dim, desc.router.out_dim);
    let gate_splits = matmul::bf16_split(desc.shared_router.in_dim, desc.shared_router.out_dim);
    let blocks = inter.div_ceil(FUSED_ROUTE_WARPS)
        + shape.moe.experts * splits / FUSED_ROUTE_WARPS
        + 1
        + hidden.div_ceil(FUSED_ROUTE_DOWN_ROWS);
    let at = |role: Role| WeightRef::layer(layer, role);
    let scales = |role: Role| Operand::Weight(at(role).spanning(WeightPart::GroupScales));
    let entry = if n_tok == 1 {
        "q4e_ffn_route_kernel"
    } else {
        "q4e_ffn_route_rows_kernel"
    };
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(kernels::MOD_FFN, entry),
        kernels::Launch::new(
            shiso_core::Extent3::linear(blocks),
            FUSED_ROUTE_WARPS * kernels::WARP,
            if n_tok <= FUSED_ROUTE_THROTTLED {
                FUSED_ROUTE_SMEM
            } else {
                0
            },
        ),
        // `logits, gate_logit, mid, shared_out, ids, weights, bounds, active,
        //  sorted, pos, sync, router, gate, g_codes, g_scales, u_codes,
        //  u_scales, d_codes, d_scales, x, in_dim, n_expert, n_used, inter,
        //  splits, gate_splits, rows`
        Args::of(&[
            Operand::Scratch(Buf::MoeRouter),
            Operand::Scratch(Buf::SharedLogit),
            Operand::Scratch(Buf::SharedMid),
            Operand::Scratch(Buf::SharedOut),
            Operand::Scratch(Buf::MoeIds),
            Operand::Scratch(Buf::MoeWeights),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeWsPos),
            Operand::Scratch(Buf::MoeSync),
            Operand::Weight(at(Role::Router)),
            Operand::Weight(at(Role::SharedRouter)),
            Operand::Weight(at(Role::SharedGate)),
            scales(Role::SharedGate),
            Operand::Weight(at(Role::SharedUp)),
            scales(Role::SharedUp),
            Operand::Weight(at(Role::SharedDown)),
            scales(Role::SharedDown),
            Operand::Scratch(Buf::Mixed),
            Operand::U32(hidden),
            Operand::U32(shape.moe.experts),
            Operand::U32(shape.moe.used),
            Operand::U32(inter),
            Operand::U32(splits),
            Operand::U32(gate_splits),
            Operand::U32(n_tok),
        ]),
    );
    plan
}

/// [`combine`] then [`shared_add`] as one launch (`q4e_ffn_combine_kernel`):
/// the same expressions in the same order, the routed sum never stored in
/// between - the close of a block whose shared expert [`fused_route`] ran.
#[must_use]
pub fn fused_close(shape: &SparseShape, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(kernels::MOD_FFN, "q4e_ffn_combine_kernel"),
        kernels::moe_combine(shape.hidden, n_tok),
        // `out, down, weights, shared, logit, n_embd, n_used`
        Args::of(&[
            Operand::Scratch(Buf::BlockOut),
            Operand::Scratch(Buf::MoeDown),
            Operand::Scratch(Buf::MoeWeights),
            Operand::Scratch(Buf::SharedOut),
            Operand::Scratch(Buf::SharedLogit),
            Operand::U32(shape.hidden),
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

/// Weight rows one block of the NVFP4 prompt kernels covers (`Q4E_MG_BM`):
/// 256 of down, or the same 128 (`Q4E_MG_HALF`) of gate and of up.
const NVFP4_GEMM_ROWS: u32 = 256;

/// Blocks one row tile of the NVFP4 prompt kernels is split over, column tile
/// `z` of every `NVFP4_GEMM_SPLIT` to block `z` (`Q4E_MG_SPLIT`).
const NVFP4_GEMM_SPLIT: u32 = 4;

/// Threads one block of the NVFP4 prompt kernels runs (`Q4E_MG_THREADS`):
/// sixteen warps, sixteen rows each.
const NVFP4_GEMM_THREADS: u32 = 512;

/// Dynamic shared memory one block of the NVFP4 prompt kernels takes
/// (`Q4E_MG_SMEM`): two weight stages, two activation steps, and the output
/// tile staged over them.
const NVFP4_GEMM_SMEM: u32 = 92_416;

/// Contraction widths the NVFP4 prompt kernels take: whole multiples of
/// their two activation steps of 64.
const NVFP4_GEMM_K_ALIGN: u32 = 128;

/// Weight rows one block of the W4A4 prompt kernels covers (`LP_A4_BW`): 128
/// of down, or the same 64 of gate and of up.
const NVFP4_A4_ROWS: u32 = 128;

/// Blocks one row tile of the W4A4 prompt kernels is split over
/// (`LP_A4_SPLIT`).
const NVFP4_A4_SPLIT: u32 = 4;

/// Threads one block of the W4A4 prompt kernels runs (`LP_A4_THREADS`).
const NVFP4_A4_THREADS: u32 = 256;

/// Columns one W4A4 tile covers (`LP_A4_BN`).
const NVFP4_A4_COLS: u32 = 128;

/// Dynamic shared memory one block of the W4A4 prompt kernels takes
/// (`LP_A4_SMEM`): two stages of 128 weight rows and 128 columns at 256
/// contraction elements (codes and scales of both), and three words a column.
const NVFP4_A4_SMEM: u32 =
    2 * (128 * 128 + 128 * 16 + NVFP4_A4_COLS * 128 + NVFP4_A4_COLS * 16) + NVFP4_A4_COLS * 12;

/// Rows one block of the NVFP4 quantizer covers, a warp each.
const NVFP4_A4_QUANT_ROWS: u32 = 4;

/// Bytes of `rows` rows of width `k` quantized for the W4A4 kernels: codes,
/// block scales and one `f32` global a row.
const fn nvfp4_a4_bytes(rows: u64, k: u32) -> u64 {
    rows * (k as u64 / 2 + k as u64 / 16 + 4)
}

/// Whether a routed stage takes the W4A4 kernels: a target layer's NVFP4
/// bank in a prompt pass under
/// [`matmul::PrefillActivations::Lowp`]. The draft head's resident experts
/// keep the weight-only kernels, as the reference keeps them.
#[must_use]
pub fn lowp_routed(desc: &ExpertDesc, layer: LayerRef, device: Device) -> bool {
    device.lowp_prompt() && layer.set == WeightSet::Target && desc.gate.dtype == DType::Nvfp4Gs16
}

/// The W4A4 routed stage of a prompt pass: the token rows quantized, gate and
/// up with the `SwiGLU` fused, the intermediate quantized, down. The quantized
/// rows take the weight-only stage's gate and up buffers, which this stage
/// does not otherwise write: a quantized row is a ninth of the bytes of the
/// `f32` rows they were sized for. The intermediate and the routed rows are
/// brain floats in the buffers sized for their `f32` forms, and the combine
/// that reads them is [`combine_lowp`].
///
/// # Errors
///
/// [`CoreError::Unsupported`] when a width is not a multiple of 128.
fn lowp_experts(
    shape: &SparseShape,
    layer: LayerRef,
    n_tok: u32,
    groups: u32,
    map: Operand,
) -> CoreResult<Plan> {
    let hidden = shape.hidden;
    let inter = shape.moe.intermediate;
    if !(hidden.is_multiple_of(NVFP4_GEMM_K_ALIGN) && inter.is_multiple_of(NVFP4_GEMM_K_ALIGN)) {
        return Err(CoreError::Unsupported {
            what: "a W4A4 prompt expert stage whose widths are not multiples of 128",
            backend: "engine",
        });
    }
    let columns = u32::try_from(shape.moe.slot_rows(n_tok)).unwrap_or(u32::MAX);
    debug_assert!(
        nvfp4_a4_bytes(u64::from(n_tok), hidden) <= Buf::MoeGate.bytes(shape, n_tok, 1, 0)
            && nvfp4_a4_bytes(u64::from(columns), inter) <= Buf::MoeUp.bytes(shape, n_tok, 1, 0),
        "the quantized rows fit the buffers they borrow"
    );
    let at = |role: Role, span| Operand::Weight(WeightRef::layer(layer, role).spanning(span));
    let quant = |entry: &'static str, q: Buf, x: Buf, rows: u32, k: u32| {
        let mut plan = Plan::new();
        // `q, x, static_gs, static_idx, rows, k`: dynamic globals.
        plan.launch(
            Entry::plain(MOD_NVFP4_A4, entry),
            kernels::Launch::linear(rows.div_ceil(NVFP4_A4_QUANT_ROWS), 32 * NVFP4_A4_QUANT_ROWS),
            Args::of(&[
                Operand::Scratch(q),
                Operand::Scratch(x),
                Operand::Null,
                Operand::Null,
                Operand::U32(rows),
                Operand::U32(k),
            ]),
        );
        plan
    };
    let tiles = |rows_per: u32, out_dim: u32| {
        kernels::Launch::new(
            shiso_core::Extent3::new(NVFP4_A4_SPLIT * out_dim.div_ceil(rows_per), groups, 1),
            NVFP4_A4_THREADS,
            NVFP4_A4_SMEM,
        )
    };
    let tail = |in_dim: u32, out_dim: u32, rows: u32| {
        [
            Operand::Scratch(Buf::MoeSorted),
            Operand::Scratch(Buf::MoeBounds),
            Operand::Scratch(Buf::MoeActive),
            map,
            Operand::U32(in_dim),
            Operand::U32(out_dim),
            Operand::U32(shape.moe.experts),
            Operand::U32(shape.moe.used),
            Operand::U32(rows),
        ]
    };
    let mut plan = quant(
        "q4e_nvfp4_a4_quant_kernel",
        Buf::MoeGate,
        Buf::Mixed,
        n_tok,
        hidden,
    );
    // `mid, gate_w, gate_s, gate_g, up_w, up_s, up_g, xq, sorted, bounds,
    //  active, map, in_dim, out_dim, n_expert, n_used, rows`
    let mut ops = vec![
        Operand::Scratch(Buf::MoeMid),
        at(Role::ExpertGate, WeightPart::Whole),
        at(Role::ExpertGate, WeightPart::GroupScales),
        at(Role::ExpertGate, WeightPart::GlobalScale),
        at(Role::ExpertUp, WeightPart::Whole),
        at(Role::ExpertUp, WeightPart::GroupScales),
        at(Role::ExpertUp, WeightPart::GlobalScale),
        Operand::Scratch(Buf::MoeGate),
    ];
    ops.extend(tail(hidden, inter, n_tok));
    plan.launch(
        Entry::plain(MOD_NVFP4_A4, "q4e_nvfp4_a4_gate_up_kernel"),
        tiles(NVFP4_A4_ROWS / 2, inter),
        Args::of(&ops),
    );
    plan.append(quant(
        "q4e_nvfp4_a4_quant_bf16_kernel",
        Buf::MoeUp,
        Buf::MoeMid,
        columns,
        inter,
    ));
    // `out, down_w, down_s, down_g, midq, sorted, bounds, active, map,
    //  in_dim, out_dim, n_expert, n_used, rows`
    let mut ops = vec![
        Operand::Scratch(Buf::MoeDown),
        at(Role::ExpertDown, WeightPart::Whole),
        at(Role::ExpertDown, WeightPart::GroupScales),
        at(Role::ExpertDown, WeightPart::GlobalScale),
        Operand::Scratch(Buf::MoeUp),
    ];
    ops.extend(tail(inter, hidden, columns));
    plan.launch(
        Entry::plain(MOD_NVFP4_A4, "q4e_nvfp4_a4_down_kernel"),
        tiles(NVFP4_A4_ROWS, hidden),
        Args::of(&ops),
    );
    Ok(plan)
}

/// The recombination of the W4A4 stage's brain-float routed rows into one
/// `f32` contribution per token, in slot order as [`combine`] sums.
#[must_use]
pub fn combine_lowp(shape: &SparseShape, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_NVFP4_A4, "q4e_nvfp4_a4_combine_kernel"),
        kernels::Launch::new(
            shiso_core::Extent3::new(shape.hidden.div_ceil(512), n_tok, 1),
            256,
            0,
        ),
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

/// Output rows a block of the NVFP4 step kernels covers: gate and up one row
/// a warp (`Q4E_NVFP4_GU_ROWS`), down four (`Q4E_NVFP4_DN_ROWS`), four warps
/// a block.
const NVFP4_STEP_GATE_UP_ROWS: u32 = 4;
/// See [`NVFP4_STEP_GATE_UP_ROWS`].
const NVFP4_STEP_DOWN_ROWS: u32 = 16;

/// Threads a block of the NVFP4 step kernels runs (`Q4E_NVFP4_WARPS`).
const NVFP4_STEP_THREADS: u32 = 128;

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

/// The expert stages over an affine, a block-scaled FP8 or an NVFP4 bank: gate and up in
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
    let nvfp4 = desc.gate.dtype == DType::Nvfp4Gs16;
    if widths.iter().any(|(w, i, o)| {
        !(w.dtype.is_affine() || fp8 || nvfp4)
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
    // A draft head's banks are resident even when the target's stream.
    let map = match layer.set {
        WeightSet::Target => Operand::ExpertMap(layer.index),
        WeightSet::Draft => Operand::ResidentExpertMap,
    };
    // The block-scaled kernels run a simdgroup per output row; the affine
    // ones four rows a simdgroup.
    let launch = |out_dim: u32| {
        kernels::Launch::new(
            shiso_core::Extent3::new(out_dim.div_ceil(AFFINE_EXPERT_ROWS), groups, 1),
            if fp8 || nvfp4 {
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
    if nvfp4 && lowp_routed(desc, layer, device) {
        return lowp_experts(shape, layer, n_tok, groups, map);
    }
    if nvfp4 {
        // A prompt pass takes the tensor-core tiles: a column's result then
        // depends on the shape alone, which is the prompt's promise, and an
        // expert's weights are decoded once a tile rather than once a column.
        // A step keeps the warp-per-row kernels.
        let gemm = device.row_invariant();
        if gemm
            && !(hidden.is_multiple_of(NVFP4_GEMM_K_ALIGN)
                && inter.is_multiple_of(NVFP4_GEMM_K_ALIGN))
        {
            return Err(CoreError::Unsupported {
                what: "an NVFP4 prompt expert stage whose widths are not multiples of 128",
                backend: "engine",
            });
        }
        let (gate_up, down) = if gemm {
            (
                "q4e_nvfp4_experts_gemm_gate_up_kernel",
                "q4e_nvfp4_experts_gemm_down_kernel",
            )
        } else {
            // A step applies the activation as gate and up are stored: the
            // activation launch's bits, without the two intermediate rows. A
            // wider pass holds each row in registers across an expert's
            // columns; the two entries sum a row identically, so the choice
            // by row count moves no bit.
            (
                if n_tok == 1 {
                    "q4e_nvfp4_experts_gate_up_act_kernel"
                } else {
                    "q4e_nvfp4_experts_gate_up_act_held_kernel"
                },
                "q4e_nvfp4_experts_down_kernel",
            )
        };
        // The prompt grid: a gate-and-up block covers half the rows a down
        // block does, of both matrices, and every row tile is split over
        // `NVFP4_GEMM_SPLIT` blocks. The step grid: rows a block, one block
        // per active expert on the second axis either way.
        let (gate_up_launch, down_launch) = if gemm {
            let tiles = |rows: u32| {
                kernels::Launch::new(
                    shiso_core::Extent3::new(NVFP4_GEMM_SPLIT * rows, groups, 1),
                    NVFP4_GEMM_THREADS,
                    NVFP4_GEMM_SMEM,
                )
            };
            (
                tiles(inter.div_ceil(NVFP4_GEMM_ROWS / 2)),
                tiles(hidden.div_ceil(NVFP4_GEMM_ROWS)),
            )
        } else {
            let rows = |out_dim: u32, per: u32| {
                kernels::Launch::new(
                    shiso_core::Extent3::new(out_dim.div_ceil(per), groups, 1),
                    NVFP4_STEP_THREADS,
                    0,
                )
            };
            (
                rows(inter, NVFP4_STEP_GATE_UP_ROWS),
                rows(hidden, NVFP4_STEP_DOWN_ROWS),
            )
        };
        // `out_gate, out_up, gate_w, gate_s, gate_g, up_w, up_s, up_g, x,
        //  sorted, bounds, active, map, in_dim, out_dim, n_expert, n_used`,
        // and on a step the one output `out_mid` in place of the first two.
        let outputs: &[Operand] = if gemm {
            &[Operand::Scratch(Buf::MoeGate), Operand::Scratch(Buf::MoeUp)]
        } else {
            &[Operand::Scratch(Buf::MoeMid)]
        };
        let mut ops = outputs.to_vec();
        ops.extend([
            at(Role::ExpertGate, WeightPart::Whole),
            at(Role::ExpertGate, WeightPart::GroupScales),
            at(Role::ExpertGate, WeightPart::GlobalScale),
            at(Role::ExpertUp, WeightPart::Whole),
            at(Role::ExpertUp, WeightPart::GroupScales),
            at(Role::ExpertUp, WeightPart::GlobalScale),
            Operand::Scratch(Buf::Mixed),
        ]);
        ops.extend(tail(hidden, inter));
        plan.launch(
            Entry::plain(MOD_MODELOPT, gate_up),
            gate_up_launch,
            Args::of(&ops),
        );
        if gemm {
            plan.append(activate(
                shape,
                Operand::Scratch(Buf::MoeGate),
                Operand::Scratch(Buf::MoeUp),
                Operand::Scratch(Buf::MoeMid),
                n_tok,
            ));
        }
        // `out, down_w, down_s, down_g, mid, sorted, bounds, active, map,
        //  in_dim, out_dim, n_expert, n_used`
        let mut ops = vec![
            Operand::Scratch(Buf::MoeDown),
            at(Role::ExpertDown, WeightPart::Whole),
            at(Role::ExpertDown, WeightPart::GroupScales),
            at(Role::ExpertDown, WeightPart::GlobalScale),
            Operand::Scratch(Buf::MoeMid),
        ];
        ops.extend(tail(inter, hidden));
        plan.launch(
            Entry::plain(MOD_MODELOPT, down),
            down_launch,
            Args::of(&ops),
        );
        return Ok(plan);
    }
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
            Entry::plain(MOD_MODELOPT, "q4e_fp8_experts_gate_up_kernel"),
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
            Entry::plain(MOD_MODELOPT, "q4e_fp8_experts_down_kernel"),
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
    plan.append(shared_add(shape, n_tok));
    Ok(plan)
}

/// The shared expert's gated output added onto [`Buf::BlockOut`].
#[must_use]
pub fn shared_add(shape: &SparseShape, n_tok: u32) -> Plan {
    let mut plan = Plan::new();
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
    plan
}

/// `main` followed by the shared expert.
///
/// On a step or verify pass whose shared expert is MXFP8 group launches, the
/// shared expert runs as a branch beside `main` (the router, the selection
/// and the routed stages), as the trellis path runs its own: it reads only
/// the block input and writes only its own buffers, and on its own it is a
/// chain of latency-bound launches (the router's selection alone is one
/// block). Only when neither side carves per-stream scratch, which the two
/// would share. Its gate logit and the add stay after the join, in the order
/// the serial plan has them, so no value moves.
fn with_shared(
    shape: &SparseShape,
    desc: &ExpertDesc,
    layer: LayerRef,
    n_tok: u32,
    device: Device,
    mut main: Plan,
) -> CoreResult<Plan> {
    if matmul::groups(&[desc.shared_gate, desc.shared_up], device)
        && desc.shared_down.dtype == DType::Mxfp8E4M3G32
    {
        let branch = shared_expert(shape, desc, layer, n_tok, device)?;
        if branch.staging().bytes == 0 && main.staging().bytes == 0 {
            let mut forked = Plan::new();
            forked.push(Step::Branch(Branch::Fork));
            forked.append(branch);
            forked.push(Step::Branch(Branch::Resume));
            forked.append(main);
            forked.push(Step::Branch(Branch::Join));
            forked.append(shared_logit(desc, layer, n_tok, device)?);
            forked.append(shared_add(shape, n_tok));
            return Ok(forked);
        }
    }
    main.append(shared(shape, desc, layer, n_tok, device)?);
    Ok(main)
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
    if desc.gate.dtype.is_affine()
        || matches!(desc.gate.dtype, DType::Fp8E4M3Block128 | DType::Nvfp4Gs16)
    {
        // A step or verify pass whose router and shared expert the fused
        // route launch reads computes all of them, the selection included,
        // in that one launch; the routed stages follow it on the same stream
        // and the shared expert's output and gate logit are in place for the
        // add.
        if route_fuses(shape, desc, n_tok, device) {
            let mut plan = fused_route(shape, desc, layer, n_tok);
            if layer.set == WeightSet::Target {
                plan.push(Step::Fetch {
                    layer: layer.index,
                    ids: u32::try_from(shape.moe.slot_rows(n_tok)).unwrap_or(u32::MAX),
                });
            }
            plan.append(affine_experts(shape, desc, layer, n_tok, device)?);
            plan.append(fused_close(shape, n_tok));
            return Ok(plan);
        }
        // A decode-sized pass routes and sorts in one launch, which is the
        // same answer as the four-launch route and sort bit for bit - so a
        // prompt pass narrow enough to take it still routes a row as a wider
        // one does - at a sixth of the dependent launches.
        let mut plan = if route_sorts(shape, n_tok) {
            let mut plan = router_logits(desc, layer, n_tok, device)?;
            plan.append(route_sort(shape, n_tok, device));
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
        plan.append(if lowp_routed(desc, layer, device) {
            combine_lowp(shape, n_tok)
        } else {
            combine(shape, n_tok)
        });
        return with_shared(shape, desc, layer, n_tok, device, plan);
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
                main.append(route_sort(shape, n_tok, device));
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
        plan.append(route_sort(shape, n_tok, device));
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

    /// The published geometry, four layers of it.
    fn sparse_shape() -> SparseShape {
        use crate::shape::{GdnShape, LayerKind};
        use crate::sparse::shape::{GqaShape, IndexerShape, MoeShape, MropeSections};
        SparseShape {
            layers: vec![
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Attn,
            ],
            hidden: 2_560,
            vocab: 248_320,
            streams: 4,
            stream_rank: 320,
            gqa: GqaShape::new(24, 2, 256, 64),
            gdn: GdnShape::new(48, 16, 128, 4),
            moe: MoeShape::new(512, 10, 1, 640),
            indexer: IndexerShape::new(4, 128, 2_048, 4),
            ple: None,
            mrope: MropeSections::new(11, 11, 10),
            rms_eps: 1.0e-6,
            rope_theta: 1.0e7,
        }
    }

    /// An NVFP4 routed bank at the published widths; the rest of the
    /// description is never read by the routed stages.
    fn nvfp4_desc() -> ExpertDesc {
        use crate::sparse::weights::WeightDesc;
        let w = |dtype, in_dim, out_dim| WeightDesc::new(dtype, in_dim, out_dim);
        ExpertDesc {
            router: w(DType::BF16, 2_560, 512),
            gate: w(DType::Nvfp4Gs16, 2_560, 640),
            up: w(DType::Nvfp4Gs16, 2_560, 640),
            down: w(DType::Nvfp4Gs16, 640, 2_560),
            shared_router: w(DType::BF16, 2_560, 1),
            shared_gate: w(DType::Mxfp8E4M3G32, 2_560, 640),
            shared_up: w(DType::Mxfp8E4M3G32, 2_560, 640),
            shared_down: w(DType::Mxfp8E4M3G32, 640, 2_560),
        }
    }

    fn entries(plan: &Plan) -> Vec<&'static str> {
        plan.steps()
            .iter()
            .filter_map(Step::entry)
            .map(|e| e.name())
            .collect()
    }

    #[test]
    fn only_a_target_prompt_pass_under_lowp_takes_the_w4a4_stages() {
        let shape = sparse_shape();
        let desc = nvfp4_desc();
        let base = Device {
            prefill_activations: matmul::PrefillActivations::Lowp,
            ..Device::default()
        };
        let prompt = Device {
            role: matmul::PassRole::Prompt { snapshot: None },
            ..base
        };
        let target = LayerRef::target(0);
        for n_tok in [1, 64, 2_048] {
            let plan = affine_experts(&shape, &desc, target, n_tok, prompt).expect("plans");
            assert_eq!(
                entries(&plan),
                [
                    "q4e_nvfp4_a4_quant_kernel",
                    "q4e_nvfp4_a4_gate_up_kernel",
                    "q4e_nvfp4_a4_quant_bf16_kernel",
                    "q4e_nvfp4_a4_down_kernel",
                ],
                "{n_tok} tokens"
            );
            assert!(lowp_routed(&desc, target, prompt));
        }
        // A step under the option, a prompt without it, and the draft head's
        // resident bank keep the weight-only stages.
        for (device, layer) in [
            (base, target),
            (
                Device {
                    prefill_activations: matmul::PrefillActivations::A16,
                    ..prompt
                },
                target,
            ),
            (prompt, LayerRef::draft()),
        ] {
            assert!(!lowp_routed(&desc, layer, device));
            let plan = affine_experts(&shape, &desc, layer, 64, device).expect("plans");
            assert!(
                entries(&plan).iter().all(|e| !e.contains("_a4_")),
                "{:?}",
                entries(&plan)
            );
        }
    }

    #[test]
    fn the_w4a4_quantized_rows_fit_the_buffers_they_borrow() {
        let shape = sparse_shape();
        for rows in [1u32, 64, 2_048, crate::sparse::MAX_PREFILL_CHUNK] {
            let columns = shape.moe.slot_rows(rows);
            assert!(
                nvfp4_a4_bytes(u64::from(rows), shape.hidden)
                    <= Buf::MoeGate.bytes(&shape, rows, 1, 0)
            );
            assert!(
                nvfp4_a4_bytes(columns, shape.moe.intermediate)
                    <= Buf::MoeUp.bytes(&shape, rows, 1, 0)
            );
        }
    }

    #[test]
    fn segments_carry_up_to_sixty_four_runs_of_any_width() {
        let runs = Segments::new(&[300, 5, 1]).expect("three runs");
        assert_eq!((runs.count(), runs.rows()), (3, 306));
        assert_eq!([runs.start(0), runs.start(1), runs.start(2)], [0, 300, 305]);
        assert_eq!(
            [runs.last_row(0), runs.last_row(1), runs.last_row(2)],
            [299, 304, 305]
        );
        assert_eq!(runs.len(3), 0, "past the last run there are no rows");
        let widest = vec![1_u32; MAX_PROMPT_RUNS as usize];
        assert_eq!(Segments::new(&widest).expect("the widest pass").count(), 64);
    }

    #[test]
    fn segments_refuse_no_runs_an_empty_run_and_one_run_too_many() {
        Segments::new(&[]).expect_err("no runs");
        Segments::new(&[4, 0]).expect_err("an empty run");
        let over = vec![1_u32; MAX_PROMPT_RUNS as usize + 1];
        Segments::new(&over).expect_err("sixty-five runs");
    }

    #[test]
    fn a_verify_packs_its_run_table_as_before_and_a_prompt_pass_cannot() {
        let verify = Segments::new(&[3, 4]).expect("a verify");
        assert_eq!(verify.packed_lens().expect("it fits"), 0x0403);
        Segments::new(&[256])
            .expect("one prompt run")
            .packed_lens()
            .expect_err("a run wider than a byte counts");
        Segments::new(&[1; 9])
            .expect("nine runs")
            .packed_lens()
            .expect_err("more runs than a verify batch");
    }

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

//! The attention block, dense and selected-block alike.
//!
//! One slot in four holds one of these. It projects a query row that carries its
//! own per-head output gate in the same tensor, normalizes and rotates the
//! query, projects and stores a key and a value, keeps the indexer current, runs
//! attention, applies the gate and projects back to the residual width.
//!
//! # Six attention kernels, one preference order
//!
//! The dense entry has a preference order with three shape preconditions ahead
//! of an always-correct fallback, and the sparse entry has its own. Every one of
//! those preconditions is a *shape* question - a row count, a head width, a
//! shared-memory grant - so all of them are answered here, by name, and the
//! kernel that runs is the first whose preconditions hold. A path taken for the
//! wrong reason is a wrong answer rather than a slow one, because they do not
//! read the same keys: the sparse ones read only the blocks the indexer
//! selected.
//!
//! One entry sits above the order rather than inside it. Every kernel in it but
//! the batched one stages several queries under a single page table, and the
//! rows of a decode batch belong to different sequences; so a batched pass
//! takes the batched entry whatever the grants say, and that entry is the dense
//! one, because this kernel set ships no batched selected-block attention.

use shiso_core::error::CoreResult;

use crate::sparse::bind::{Args, LayerRef, Operand, Role, WeightRef};
use crate::sparse::indexer::{self, Scorer};
use crate::sparse::inputs::MROPE_PREFIX_BYTES;
use crate::sparse::kernels::{self, Entry, MOD_EXACT, MOD_QSA, MOD_QSA_SPARSE};
use crate::sparse::matmul::{self, Bound, Device, PassRole};
use crate::sparse::moe::{RowOrigin, Segments};
use crate::sparse::paged::Plane;
use crate::sparse::plan::Plan;
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::weights::AttnDesc;

/// What an attention block needs that its geometry does not say.
///
/// Deliberately NOT `#[non_exhaustive]`: the stack builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnSlot {
    /// The layer, which names the weights.
    pub layer: LayerRef,
    /// The slot's rank among attention slots, which names the cache planes.
    pub rank: u32,
    /// Cache **rows** the pool holds - pages times [`shiso_core::page::PAGE_TOKENS`],
    /// not pages.
    ///
    /// The store kernel refuses a row past this, which is what keeps a page id
    /// the host got wrong from writing over another sequence. Handing it the
    /// page count instead is the same refusal with the bound a factor of
    /// `PAGE_TOKENS` too low: every write lands silently nowhere from the first
    /// session whose page id reaches `pool_pages / PAGE_TOKENS`, its attention
    /// reads a plane of zeros, and the answer is fluent and unrelated. Nothing
    /// fails, which is why the unit is in the name and in this note.
    pub pool_slots: u32,
    /// Positions this run's context reaches.
    ///
    /// The selection's score buffer is strided by it: see
    /// [`crate::sparse::shape::IndexerShape::score_blocks`].
    pub ctx: u32,
}

/// Which attention kernel a call takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttnPath {
    /// The always-correct kernel: one block per (head, token), every visible
    /// key read. Nothing about it is conditional, which is why it is what every
    /// other path falls back to.
    Dense,
    /// The tiled kernel, which stages key tiles in shared memory. Faster over a
    /// chunk, and only reachable at a row count and head width its tiles cover.
    Tiled,
    /// The tensor-core kernel over the whole causal range, one block per
    /// (row, key head), the result written directly.
    TensorCore,
    /// The split-key kernel, which cuts the context into fixed slices and folds
    /// them. A decode path: its slice count is fixed so that a captured graph
    /// sees the same geometry at every context length.
    Split,
    /// The gathered kernel, which reads only the blocks the indexer selected.
    Gathered,
    /// The batched kernel: the always-correct one with one page table and one
    /// position per row.
    ///
    /// It sits above the whole ladder rather than inside it, because every
    /// other entry here stages several queries under one page table and a
    /// batch's rows do not share one. It is also the only batched attention
    /// entry this kernel set ships, which is why a batched pass is a dense
    /// pass.
    Batched,
    /// The tensor-core kernel over the selected blocks.
    GatheredTensorCore,
}

/// One row count per run of a segmented pass, in run order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunRows([u32; crate::sparse::stack::MAX_PROMPT_RUNS as usize]);

impl Default for RunRows {
    fn default() -> Self {
        Self([0; crate::sparse::stack::MAX_PROMPT_RUNS as usize])
    }
}

impl RunRows {
    /// Run `run`'s count, zero past the last run.
    #[must_use]
    pub fn get(&self, run: u32) -> u32 {
        self.0.get(run as usize).copied().unwrap_or(0)
    }

    /// Sets run `run`'s count; a run past the cap has none to set.
    pub fn set(&mut self, run: u32, rows: u32) {
        if let Some(slot) = self.0.get_mut(run as usize) {
            *slot = rows;
        }
    }
}

/// What the caller knows about this call that the shape does not say.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AttnContext {
    /// Whether the indexer's selection is live for this pass's last row; the
    /// rows in front of it that are not yet past the budget are counted by
    /// [`AttnContext::dense_rows`].
    ///
    /// It is not a property of the block: it is a property of how much context
    /// the session has accumulated, and it is the same answer for every layer of
    /// one pass, which is why the caller supplies it rather than each block
    /// asking.
    pub selected: bool,
    /// Whether the device granted the tensor-core attention kernel its shared
    /// memory, which is more than three quarters of the per-block maximum.
    pub tensor_core: bool,
    /// Whether the device granted the tiled kernel its shared memory.
    pub tiled: bool,
    /// Whether the rows are one sequence's consecutive positions or one row per
    /// sequence.
    ///
    /// Here rather than beside the row count for the same reason `selected` is:
    /// it is a property of the pass rather than of the block, it is the same
    /// answer for every layer, and it decides which of two kernels the store
    /// and the attention are.
    pub origin: RowOrigin,
    /// For a batch of unrelated rows or the runs of a segmented pass, which
    /// rows' or runs' own contexts have outgrown the indexer budget: bit `r`
    /// is row or run `r`.
    ///
    /// A batch's rows are separate sequences, so whether a row attends to a
    /// selection is that row's question, answered from its own frontier - and a
    /// row that is past the budget must read the same selection a single-row
    /// step of that sequence would, or batching it would change its text.
    /// Ignored for one sequence's rows, whose answer is `selected`.
    pub batch_selected: u64,
    /// For one sequence's rows when `selected` is set: how many rows at the
    /// front of the pass still sit within the budget, counted from each row's
    /// own position.
    ///
    /// `selected` is the pass's last row's answer, and a pass that crosses the
    /// budget has rows on both sides of it. Those in front are attended densely
    /// and only the rest over a selection, because that is what each row's own
    /// single-row step does - and a verify pass is only a verify if every row
    /// is computed as that step would compute it. Below the budget a selection
    /// is every visible key, so sending those rows through the selected-block
    /// kernel instead is the same attention summed by a different kernel:
    /// close, not equal, and a speculative greedy run then leaves a plain one's
    /// text at the next near tie. Zero, the default, is a pass wholly past it.
    pub dense_rows: u32,
    /// The same count for each run of a segmented pass, where run `r`'s entry
    /// is read when its bit of `batch_selected` is set.
    pub run_dense_rows: RunRows,
    /// Which reference pass the rows must reproduce. A
    /// [`PassRole::Prompt`] pass's rows are attended exactly as they would be
    /// in a pass of any other width. The kernel is then chosen as for the
    /// widest pass - the tensor-core entries, whose blocks are one row each, at
    /// one slice - rather than from this pass's row count, so the fourteen
    /// rows at the end of a prompt are summed as they would have been in the
    /// middle of one. See [`crate::sparse::matmul::Device::role`].
    pub role: PassRole,
}

/// Query heads that share one key head.
fn groups(shape: &SparseShape) -> u32 {
    if shape.gqa.kv_heads == 0 {
        return 0;
    }
    shape.gqa.q_heads / shape.gqa.kv_heads
}

/// Whether the selected-block kernels can run on this geometry at all.
///
/// Three preconditions, and the middle one is why a stack whose query heads do
/// not divide evenly among its key heads has no sparse path: the selection is
/// per key head, and a group that straddled two of them would read another
/// head's keys.
#[must_use]
pub fn selected_eligible(shape: &SparseShape) -> bool {
    shape.gqa.head_dim == kernels::ATTN_SPLIT_HD
        && shape.gqa.kv_heads != 0
        && shape.gqa.q_heads.is_multiple_of(shape.gqa.kv_heads)
        && groups(shape) <= kernels::ATTN_QG
}

/// Which kernel this call takes.
///
/// The order is the preference order. The selected-block entry comes first when
/// a selection exists *and* the geometry supports it; below it is the dense
/// ladder, which is itself three conditional paths above an unconditional one.
#[must_use]
pub fn path(shape: &SparseShape, ctx: AttnContext, n_tok: u32) -> AttnPath {
    let head_dim = shape.gqa.head_dim;
    let n_tok = if matches!(ctx.role, PassRole::Prompt { .. }) {
        u32::MAX
    } else {
        n_tok
    };
    // Above every precondition below it: rows from different sequences read
    // different page tables, and no other entry in this set takes more than
    // one.
    if ctx.origin == RowOrigin::Unrelated {
        return AttnPath::Batched;
    }
    if ctx.selected && selected_eligible(shape) {
        return if ctx.tensor_core {
            AttnPath::GatheredTensorCore
        } else {
            AttnPath::Gathered
        };
    }
    if ctx.tensor_core
        && n_tok >= kernels::ATTN_QT
        && head_dim == kernels::TC_HD
        && shape.gqa.kv_heads != 0
        && shape.gqa.q_heads.is_multiple_of(shape.gqa.kv_heads)
        && groups(shape) <= kernels::ATTN_QG
    {
        return AttnPath::TensorCore;
    }
    if ctx.tiled && kernels::qsa_tiled_eligible(n_tok, head_dim) {
        return AttnPath::Tiled;
    }
    // The split path is bounded from above by the tiled path's floor rather
    // than merely preferred below it: its partial slab is allocated once, for
    // the widest batch it can take, so that a captured graph never sees the
    // address move.
    if head_dim == kernels::ATTN_SPLIT_HD
        && shape.gqa.kv_heads != 0
        && shape.gqa.q_heads.is_multiple_of(shape.gqa.kv_heads)
        && n_tok < kernels::ATTN_QT
    {
        return AttnPath::Split;
    }
    AttnPath::Dense
}

/// The attention launches alone, without the projections around them.
///
/// Split out because the choice between the six is the part a test asks about
/// and the projections around it are the same either way.
#[must_use]
pub fn attend(shape: &SparseShape, slot: AttnSlot, ctx: AttnContext, n_tok: u32) -> Plan {
    attend_at(shape, slot, &ctx, n_tok, RowAt::WHOLE)
}

/// [`attend`] over the rows `at` names.
#[expect(
    clippy::too_many_lines,
    reason = "one statement per kernel argument is the point: these bodies are \
              transcriptions of parameter lists, and a split that broke a list \
              across two functions would hide the order being transcribed"
)]
fn attend_at(
    shape: &SparseShape,
    slot: AttnSlot,
    ctx: &AttnContext,
    n_tok: u32,
    at: RowAt,
) -> Plan {
    let q_width = u64::from(shape.gqa.q_heads) * u64::from(shape.gqa.head_dim);
    let mut plan = Plan::new();
    let keys = Operand::CachePlane {
        rank: slot.rank,
        plane: Plane::Key,
    };
    let values = Operand::CachePlane {
        rank: slot.rank,
        plane: Plane::Value,
    };
    let out = at.of(Buf::AttnOut, q_width);
    let part = Operand::Scratch(Buf::AttnPartial);
    let query = at.of(Buf::AttnQuery, q_width);
    let positions = at.of(Buf::Positions, 1);
    let pages = at.pages;
    let tokens = at.of(Buf::IdxTokens, u64::from(kernels::IDX_WIDTH));
    let token_count = at.of(Buf::IdxTokenCount, 1);
    // `out, part, n_splits`
    let fold = |plan: &mut Plan, splits: u32| {
        plan.launch(
            Entry::plain(MOD_QSA_SPARSE, "q4e_qsa_attention_combine_kernel"),
            kernels::qsa_attention_combine(shape.gqa.q_heads, n_tok),
            Args::of(&[out, part, Operand::U32(splits)]),
        );
    };
    // `out, k_cache, v_cache, q, head_dim, n_head, n_head_kv, pos, pages, n_tok`
    let dense = Args::of(&[
        out,
        keys,
        values,
        query,
        Operand::U32(shape.gqa.head_dim),
        Operand::U32(shape.gqa.q_heads),
        Operand::U32(shape.gqa.kv_heads),
        positions,
        pages,
        Operand::U32(n_tok),
    ]);
    match path(shape, *ctx, n_tok) {
        // `out, k_cache, v_cache, q, head_dim, n_head, n_head_kv, pos,
        //  pages_arr, n_tok` - the scalar list with the page table replaced by
        // the per-row array, which is the whole of the difference.
        AttnPath::Batched => plan.launch(
            Entry::plain(MOD_QSA, "q4e_qsa_attention_batch_kernel"),
            kernels::qsa_attention_batch(shape.gqa.q_heads, n_tok, shape.gqa.head_dim),
            Args::of(&[
                out,
                keys,
                values,
                query,
                Operand::U32(shape.gqa.head_dim),
                Operand::U32(shape.gqa.q_heads),
                Operand::U32(shape.gqa.kv_heads),
                positions,
                Operand::PageTables,
                Operand::U32(n_tok),
            ]),
        ),
        AttnPath::Dense => plan.launch(
            Entry::plain(MOD_QSA, "q4e_qsa_attention_kernel"),
            kernels::qsa_attention(shape.gqa.q_heads, n_tok, shape.gqa.head_dim),
            dense,
        ),
        AttnPath::Tiled => plan.launch(
            Entry::plain(MOD_QSA, "q4e_qsa_attention_tiled_kernel"),
            kernels::qsa_attention_tiled(shape.gqa.q_heads, n_tok, shape.gqa.head_dim),
            dense,
        ),
        // One slice, so the kernel writes the destination itself and there is
        // nothing to fold. The dense entry reads the whole causal range, which
        // is what the absent selection list says. Both tensor-core paths take
        // the exact entry: the ported one rounds every softmax weight to f16
        // before the P V product, an error the f32 denominator does not share
        // (1.6e-4 against 6e-7 on the kernel case's short rows). It is the
        // ported kernel's arguments and grid, so it is a drop-in.
        AttnPath::TensorCore => plan.launch(
            Entry::plain(MOD_EXACT, "q4e_qsa_attention_tc_exact_kernel"),
            kernels::qsa_attention_tc_exact(shape.gqa.kv_heads, n_tok, 1),
            tc_args(shape, slot, 1, false, at),
        ),
        AttnPath::Split => {
            plan.launch(
                Entry::plain(MOD_QSA_SPARSE, "q4e_qsa_attention_split_kernel"),
                kernels::qsa_attention_split(shape.gqa.kv_heads, n_tok),
                // `out, part, k_cache, v_cache, q, n_head, n_head_kv, pos,
                //  pages, n_tok, n_splits`
                Args::of(&[
                    out,
                    part,
                    keys,
                    values,
                    query,
                    Operand::U32(shape.gqa.q_heads),
                    Operand::U32(shape.gqa.kv_heads),
                    positions,
                    pages,
                    Operand::U32(n_tok),
                    Operand::U32(kernels::ATTN_SPLITS),
                ]),
            );
            fold(&mut plan, kernels::ATTN_SPLITS);
        }
        AttnPath::Gathered => {
            plan.launch(
                Entry::plain(MOD_QSA_SPARSE, "q4e_qsa_attention_gather_kernel"),
                kernels::qsa_attention_gather(shape.gqa.kv_heads, n_tok),
                // `part, k_cache, v_cache, q, tokens, n_sel, pages, width,
                //  n_head, n_head_kv, n_splits` - it writes only the slabs, so
                // the fold below is what writes the destination.
                Args::of(&[
                    part,
                    keys,
                    values,
                    query,
                    tokens,
                    token_count,
                    pages,
                    Operand::U32(kernels::IDX_WIDTH),
                    Operand::U32(shape.gqa.q_heads),
                    Operand::U32(shape.gqa.kv_heads),
                    Operand::U32(kernels::ATTN_GSPLITS),
                ]),
            );
            fold(&mut plan, kernels::ATTN_GSPLITS);
        }
        AttnPath::GatheredTensorCore => {
            let splits = kernels::tc_splits_sparse(
                shape.gqa.kv_heads,
                if matches!(ctx.role, PassRole::Prompt { .. }) {
                    u32::MAX
                } else {
                    n_tok
                },
            );
            plan.launch(
                Entry::plain(MOD_EXACT, "q4e_qsa_attention_tc_exact_kernel"),
                kernels::qsa_attention_tc_exact(shape.gqa.kv_heads, n_tok, splits),
                tc_args(shape, slot, splits, true, at),
            );
            // A single slice wrote the destination itself; folding anyway would
            // double every value.
            if splits > 1 {
                fold(&mut plan, splits);
            }
        }
    }
    plan
}

/// The rest of an attention block for a segmented pass, from the key and value
/// store on: every launch that addresses a sequence's cache runs once per run,
/// as the launch that sequence's own verify pass would make over those rows.
#[expect(
    clippy::too_many_lines,
    reason = "one statement per kernel argument is the point: these bodies are \
              transcriptions of parameter lists, and a split that broke a list \
              across two functions would hide the order being transcribed"
)]
fn segmented(
    shape: &SparseShape,
    desc: &AttnDesc,
    slot: AttnSlot,
    ctx: &AttnContext,
    runs: &Segments,
    device: Device,
    mut plan: Plan,
) -> CoreResult<Plan> {
    let at = |role: Role| WeightRef::layer(slot.layer, role);
    let n_tok = runs.rows();
    let kv_width = u64::from(shape.kv_dim()) * 4;
    for run in 0..runs.count() {
        let first = u64::from(runs.start(run));
        let rows = runs.len(run);
        plan.launch(
            Entry::plain(MOD_QSA, "q4e_qsa_store_kv_kernel"),
            kernels::qsa_store_kv(shape.gqa.kv_heads, rows, shape.gqa.head_dim),
            // `k_cache, v_cache, k, v, kw, head_dim, n_head_kv, n_rot,
            //  rope_base, pos, mrope, sec_t, sec_h, sec_w, pages, pool_slots,
            //  eps`
            Args::of(&[
                Operand::CachePlane {
                    rank: slot.rank,
                    plane: Plane::Key,
                },
                Operand::CachePlane {
                    rank: slot.rank,
                    plane: Plane::Value,
                },
                Operand::ScratchFrom(Buf::AttnKey, first * kv_width),
                Operand::ScratchFrom(Buf::AttnValue, first * kv_width),
                Operand::Weight(at(Role::AttnKeyNorm)),
                Operand::U32(shape.gqa.head_dim),
                Operand::U32(shape.gqa.kv_heads),
                Operand::U32(shape.gqa.n_rot),
                Operand::f32(shape.rope_theta),
                Operand::ScratchFrom(Buf::Positions, first * 4),
                Operand::ScratchFrom(
                    Buf::Mrope,
                    MROPE_PREFIX_BYTES + first * crate::sparse::inputs::MROPE_ROW_BYTES,
                ),
                Operand::U32(shape.mrope.time),
                Operand::U32(shape.mrope.height),
                Operand::U32(shape.mrope.width),
                Operand::PagesOf(run),
                Operand::U32(slot.pool_slots),
                Operand::f32(shape.rms_eps),
            ]),
        );
    }
    let indexed = indexer::IndexerSlot::new(slot.layer, slot.rank, slot.ctx);
    plan.append(indexer::store(
        shape,
        &desc.indexer,
        indexed,
        ctx.origin,
        n_tok,
        device,
    )?);
    let scorer = Scorer::choose(shape.indexer.heads, ctx.tensor_core);
    if ctx.batch_selected != 0 {
        plan.append(indexer::select_queries(
            shape,
            &desc.indexer,
            indexed,
            n_tok,
            device,
        )?);
    }
    for run in 0..runs.count() {
        let first = runs.start(run);
        let rows = runs.len(run);
        let selected = ctx
            .batch_selected
            .checked_shr(run)
            .is_some_and(|bits| bits & 1 == 1);
        let dense = if selected {
            ctx.run_dense_rows.get(run).min(rows)
        } else {
            rows
        };
        if dense < rows {
            plan.append(indexer::select_rows(
                shape,
                indexed,
                scorer,
                indexer::RowSpan {
                    base: first + dense,
                    n: rows - dense,
                    pages: Operand::PagesOf(run),
                },
            ));
        }
        let one = AttnContext {
            selected,
            origin: RowOrigin::Sequential,
            ..*ctx
        };
        plan.append(attend_across(
            shape,
            slot,
            &one,
            rows,
            dense,
            RowAt {
                row: first,
                pages: Operand::PagesOf(run),
            },
        ));
    }
    plan.launch(
        Entry::plain(MOD_QSA, "q4e_qsa_gate_kernel"),
        kernels::elementwise(u64::from(n_tok) * u64::from(shape.q_dim())),
        // `x, gate, n` - in place over attention's own output.
        Args::of(&[
            Operand::Scratch(Buf::AttnOut),
            Operand::Scratch(Buf::AttnGate),
            Operand::U64(u64::from(n_tok) * u64::from(shape.q_dim())),
        ]),
    );
    matmul::project(
        &mut plan,
        desc.output,
        Bound::new(
            at(Role::AttnOutput),
            Operand::Scratch(Buf::AttnOut),
            Operand::Scratch(Buf::BlockOut),
        ),
        n_tok,
        device,
    )?;
    Ok(plan)
}

/// Where one row of a batch lives in the pass's buffers.
///
/// A batch's row `r` is its own sequence: its query, output, position and
/// selection rows are row `r` of the pass's, and its page table is its own.
#[derive(Debug, Clone, Copy)]
struct RowAt {
    row: u32,
    pages: Operand,
}

impl RowAt {
    /// The whole pass, one sequence's rows from the first.
    const WHOLE: Self = Self {
        row: 0,
        pages: Operand::Pages,
    };

    /// Byte offset of this row in a buffer whose rows are `width` elements of
    /// four bytes.
    fn offset(self, width: u64) -> u64 {
        u64::from(self.row) * width * 4
    }

    /// `buf` from this row on.
    fn of(self, buf: Buf, width: u64) -> Operand {
        if self.row == 0 {
            Operand::Scratch(buf)
        } else {
            Operand::ScratchFrom(buf, self.offset(width))
        }
    }
}

/// One sequence's `rows` rows from `at`, the first `dense` of them attended
/// over the whole context and the rest over their selections.
///
/// Two launches only for the one pass per sequence that crosses the indexer
/// budget; on either side of it `dense` is zero or `rows` and this is the one
/// launch it always was. See [`AttnContext::dense_rows`] for why the rows in
/// front cannot simply ride along with the selected ones.
fn attend_across(
    shape: &SparseShape,
    slot: AttnSlot,
    ctx: &AttnContext,
    rows: u32,
    dense: u32,
    at: RowAt,
) -> Plan {
    let mut plan = Plan::new();
    if dense > 0 {
        let within = AttnContext {
            selected: false,
            ..*ctx
        };
        plan.append(attend_at(shape, slot, &within, dense, at));
    }
    if dense < rows {
        let past = RowAt {
            row: at.row + dense,
            pages: at.pages,
        };
        plan.append(attend_at(shape, slot, ctx, rows - dense, past));
    }
    plan
}

/// One batch row's attention, launched exactly as a single-row step of its
/// sequence launches it: the split kernel over the whole cache, or - for a row
/// whose context has outgrown the budget - the selected-block kernel over the
/// selection the indexer just made for that row alone.
fn attend_one(shape: &SparseShape, slot: AttnSlot, ctx: &AttnContext, row: u32) -> Plan {
    let at = RowAt {
        row,
        pages: Operand::PagesOf(row),
    };
    let selected = ctx
        .batch_selected
        .checked_shr(row)
        .is_some_and(|bits| bits & 1 == 1);
    let one = AttnContext {
        selected,
        origin: RowOrigin::Sequential,
        ..*ctx
    };
    attend_at(shape, slot, &one, 1, at)
}

/// The tensor-core entry's argument list.
///
/// `out, part, k_cache, v_cache, q, tokens, n_sel, pos, pages, width, n_head,
/// n_head_kv, n_splits`. On the dense entry the selection list is absent and
/// the kernel reads the whole causal range instead, which is what the two null
/// pointers say.
///
/// # The first two arguments are exclusive
///
/// The kernel decides what to write by asking whether `part` is null: with a
/// partial buffer it stores an unnormalized slab for the combine, and without
/// one it normalizes and writes the destination itself. So exactly one of the
/// two is a real buffer, and which one is what the split count means. Handing
/// it both is not an over-specification - it is the slab branch, at a split
/// count whose caller then does not fold, so the destination keeps whatever it
/// held and the block contributes the previous chunk's attention.
fn tc_args(shape: &SparseShape, slot: AttnSlot, splits: u32, selected: bool, at: RowAt) -> Args {
    let q_width = u64::from(shape.gqa.q_heads) * u64::from(shape.gqa.head_dim);
    let (tokens, counts) = if selected {
        (
            at.of(Buf::IdxTokens, u64::from(kernels::IDX_WIDTH)),
            at.of(Buf::IdxTokenCount, 1),
        )
    } else {
        (Operand::Null, Operand::Null)
    };
    let (out, part) = if splits == 1 {
        (at.of(Buf::AttnOut, q_width), Operand::Null)
    } else {
        (Operand::Null, Operand::Scratch(Buf::AttnPartial))
    };
    Args::of(&[
        out,
        part,
        Operand::CachePlane {
            rank: slot.rank,
            plane: Plane::Key,
        },
        Operand::CachePlane {
            rank: slot.rank,
            plane: Plane::Value,
        },
        at.of(Buf::AttnQuery, q_width),
        tokens,
        counts,
        at.of(Buf::Positions, 1),
        at.pages,
        Operand::U32(kernels::IDX_WIDTH),
        Operand::U32(shape.gqa.q_heads),
        Operand::U32(shape.gqa.kv_heads),
        Operand::U32(splits),
    ])
}

/// One attention block, in launch order.
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
    desc: &AttnDesc,
    slot: AttnSlot,
    ctx: AttnContext,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let at = |role: Role| WeightRef::layer(slot.layer, role);
    let mut plan = Plan::new();
    matmul::project(
        &mut plan,
        desc.q,
        Bound::new(
            at(Role::AttnQuery),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::AttnQueryFull),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::plain(MOD_QSA, "q4e_qsa_q_norm_rope_kernel"),
        kernels::qsa_q_norm_rope(shape.gqa.q_heads, n_tok, shape.gqa.head_dim),
        // `q_out, gate_out, qkv, w, head_dim, n_head, n_rot, rope_base, mrope,
        //  sec_t, sec_h, sec_w, eps` - the query projection carries the per-head
        // output gate in its own rows, and this is what splits the two apart.
        Args::of(&[
            Operand::Scratch(Buf::AttnQuery),
            Operand::Scratch(Buf::AttnGate),
            Operand::Scratch(Buf::AttnQueryFull),
            Operand::Weight(at(Role::AttnQueryNorm)),
            Operand::U32(shape.gqa.head_dim),
            Operand::U32(shape.gqa.q_heads),
            Operand::U32(shape.gqa.n_rot),
            Operand::f32(shape.rope_theta),
            Operand::ScratchFrom(Buf::Mrope, MROPE_PREFIX_BYTES),
            Operand::U32(shape.mrope.time),
            Operand::U32(shape.mrope.height),
            Operand::U32(shape.mrope.width),
            Operand::f32(shape.rms_eps),
        ]),
    );
    matmul::project(
        &mut plan,
        desc.k,
        Bound::new(
            at(Role::AttnKey),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::AttnKey),
        ),
        n_tok,
        device,
    )?;
    matmul::project(
        &mut plan,
        desc.v,
        Bound::new(
            at(Role::AttnValue),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::AttnValue),
        ),
        n_tok,
        device,
    )?;
    // The batched store is the scalar one with `pages` replaced by the per-row
    // array; every other parameter, and the grid, is the same.
    let batched = ctx.origin == RowOrigin::Unrelated;
    if let RowOrigin::Segmented(runs) = ctx.origin {
        return segmented(shape, desc, slot, &ctx, &runs, device, plan);
    }
    let (store_entry, store_launch, store_pages) = if batched {
        (
            "q4e_qsa_store_kv_batch_kernel",
            kernels::qsa_store_kv_batch(shape.gqa.kv_heads, n_tok, shape.gqa.head_dim),
            Operand::PageTables,
        )
    } else {
        (
            "q4e_qsa_store_kv_kernel",
            kernels::qsa_store_kv(shape.gqa.kv_heads, n_tok, shape.gqa.head_dim),
            Operand::Pages,
        )
    };
    plan.launch(
        Entry::plain(MOD_QSA, store_entry),
        store_launch,
        // `k_cache, v_cache, k, v, kw, head_dim, n_head_kv, n_rot, rope_base,
        //  pos, mrope, sec_t, sec_h, sec_w, pages, pool_slots, eps`
        Args::of(&[
            Operand::CachePlane {
                rank: slot.rank,
                plane: Plane::Key,
            },
            Operand::CachePlane {
                rank: slot.rank,
                plane: Plane::Value,
            },
            Operand::Scratch(Buf::AttnKey),
            Operand::Scratch(Buf::AttnValue),
            Operand::Weight(at(Role::AttnKeyNorm)),
            Operand::U32(shape.gqa.head_dim),
            Operand::U32(shape.gqa.kv_heads),
            Operand::U32(shape.gqa.n_rot),
            Operand::f32(shape.rope_theta),
            Operand::Scratch(Buf::Positions),
            Operand::ScratchFrom(Buf::Mrope, MROPE_PREFIX_BYTES),
            Operand::U32(shape.mrope.time),
            Operand::U32(shape.mrope.height),
            Operand::U32(shape.mrope.width),
            store_pages,
            Operand::U32(slot.pool_slots),
            Operand::f32(shape.rms_eps),
        ]),
    );
    let indexed = indexer::IndexerSlot::new(slot.layer, slot.rank, slot.ctx);
    plan.append(indexer::store(
        shape,
        &desc.indexer,
        indexed,
        ctx.origin,
        n_tok,
        device,
    )?);
    let scorer = Scorer::choose(shape.indexer.heads, ctx.tensor_core);
    if batched {
        // Each row of a batch is attended exactly as a single-row step of its
        // own sequence would attend it: its own selection when its own context
        // is past the budget, the split kernel over its own cache otherwise.
        if ctx.batch_selected != 0 {
            plan.append(indexer::select_queries(
                shape,
                &desc.indexer,
                indexed,
                n_tok,
                device,
            )?);
        }
        for row in 0..n_tok {
            if ctx
                .batch_selected
                .checked_shr(row)
                .is_some_and(|bits| bits & 1 == 1)
            {
                plan.append(indexer::select_rows(
                    shape,
                    indexed,
                    scorer,
                    indexer::RowSpan::one(row),
                ));
            }
            plan.append(attend_one(shape, slot, &ctx, row));
        }
    } else {
        let dense = if ctx.selected {
            ctx.dense_rows.min(n_tok)
        } else {
            n_tok
        };
        if dense < n_tok {
            plan.append(indexer::select_queries(
                shape,
                &desc.indexer,
                indexed,
                n_tok,
                device,
            )?);
            plan.append(indexer::select_rows(
                shape,
                indexed,
                scorer,
                indexer::RowSpan {
                    base: dense,
                    n: n_tok - dense,
                    pages: Operand::Pages,
                },
            ));
        }
        plan.append(attend_across(shape, slot, &ctx, n_tok, dense, RowAt::WHOLE));
    }
    plan.launch(
        Entry::plain(MOD_QSA, "q4e_qsa_gate_kernel"),
        kernels::elementwise(u64::from(n_tok) * u64::from(shape.q_dim())),
        // `x, gate, n` - in place over attention's own output.
        Args::of(&[
            Operand::Scratch(Buf::AttnOut),
            Operand::Scratch(Buf::AttnGate),
            Operand::U64(u64::from(n_tok) * u64::from(shape.q_dim())),
        ]),
    );
    matmul::project(
        &mut plan,
        desc.output,
        Bound::new(
            at(Role::AttnOutput),
            Operand::Scratch(Buf::AttnOut),
            Operand::Scratch(Buf::BlockOut),
        ),
        n_tok,
        device,
    )?;
    Ok(plan)
}

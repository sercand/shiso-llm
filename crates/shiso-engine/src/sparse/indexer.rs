//! The indexer: the narrow tower that decides which of the context an
//! attention block actually reads.
//!
//! It is a second, much smaller attention problem run in front of the real one.
//! Every position projects one raw key row; every block of
//! [`IndexerShape::compress`] consecutive positions is mean-pooled into one
//! pooled row; and when the context has outgrown the selection budget, the
//! block's queries are scored against those pooled rows, the best blocks are
//! kept, and the kept blocks are expanded into the position list the attention
//! kernel then gathers.
//!
//! # A raw key lives until its block is pooled, and no longer
//!
//! The pooled pass is the only reader of a raw key. So the page keeps only the
//! pooled plane, and the raw keys a later pass still needs - those of the
//! block a session's frontier stands inside - sit in the session's own ring
//! ([`crate::sparse::paged::ring_bytes`]). The pass reads the raw keys of its
//! own positions straight from its key projection, rounded as a cache store
//! would round them, and only the block it starts inside reaches back into the
//! ring; one launch pools, then refills the ring with the pass's last rows.
//!
//! # The storing half runs on every pass and the selecting half does not
//!
//! Keeping the pooled plane and the ring current is not optional: a forward
//! pass that skipped it would leave a hole in the table for the positions it
//! wrote, and a later pass would select blocks whose rows were never filled
//! in. The
//! *selection* is optional, and below the budget it is not merely optional but
//! wrong to take: the selection there would be every visible token, and the
//! dense kernel covers that exactly and faster.
//!
//! [`IndexerShape::compress`]: crate::sparse::shape::IndexerShape::compress

use shiso_core::error::CoreResult;

use crate::sparse::bind::{Args, LayerRef, Operand, Role, WeightRef};
use crate::sparse::inputs::{MROPE_PREFIX_BYTES, batch_lookback_offset};
use crate::sparse::kernels::{self, Entry, I32_BYTES, IDX_NO_TAP, MOD_IDX, MOD_IDXRING};
use crate::sparse::matmul::{self, Bound, Device};
use crate::sparse::moe::RowOrigin;
use crate::sparse::paged::Plane;
use crate::sparse::plan::Plan;
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::weights::IndexerDesc;

/// Which of the two scorers a head count reaches.
///
/// The tensor-core one is written for one head count and needs a
/// shared-memory grant the device may refuse; the scalar one is correct at any
/// head count. They tile the key blocks at different widths, so their grids are
/// not interchangeable - which is why this is a decision with a name rather
/// than a flag on one function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scorer {
    /// The scalar scorer, whose shared memory grows with the head count.
    Scalar,
    /// The tensor-core scorer.
    TensorCore,
}

impl Scorer {
    /// Which scorer runs at this head count, given whether the wider
    /// shared-memory grant succeeded.
    #[must_use]
    pub const fn choose(heads: u32, tensor_core_available: bool) -> Self {
        if heads == kernels::IDX_TC_HEADS && tensor_core_available {
            Self::TensorCore
        } else {
            Self::Scalar
        }
    }
}

/// Which indexer a plan is for, and in what run.
///
/// Three numbers that always travel together and that nothing else can supply:
/// two name the weights and the cache planes, and the third is a fact about the
/// run rather than about the model, which is why it cannot be read off the
/// shape. The fourth is optional and rare: a copy of the raw key ring into a
/// checkpoint, which only the pass a snapshot is cut inside asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct IndexerSlot {
    /// The layer, which names the weights.
    pub layer: LayerRef,
    /// The slot's rank among attention slots, which names the cache planes and
    /// the session's raw key ring.
    pub rank: u32,
    /// Positions this run's context reaches, which is what the score buffer is
    /// strided by. See [`crate::sparse::shape::IndexerShape::score_blocks`].
    pub ctx: u32,
    /// The ring copy this pass makes into the snapshot buffer, if any.
    pub tap: Option<RingTap>,
}

impl IndexerSlot {
    /// One indexer's identity.
    #[must_use]
    pub const fn new(layer: LayerRef, rank: u32, ctx: u32) -> Self {
        Self {
            layer,
            rank,
            ctx,
            tap: None,
        }
    }

    /// The same indexer, also copying its ring into a checkpoint as it stands
    /// before one row of the pass.
    #[must_use]
    pub const fn with_tap(mut self, tap: Option<RingTap>) -> Self {
        self.tap = tap;
        self
    }
}

/// A copy of one rank's raw key ring into the stack's snapshot buffer, as the
/// ring stands before row `row` of a one-session pass: the raw keys of the
/// block holding that row's position, from its first position up to it.
///
/// It exists for one pass: the draft head's warm-up over a prompt chunk that a
/// grid checkpoint is cut inside. That checkpoint stands where the head has
/// written up to the position before the cut, and the warm-up writes on past
/// it, so the only moment those raw keys exist is inside this launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RingTap {
    /// The pass row the copy is taken before.
    pub row: u32,
    /// Where this rank's ring section starts in the snapshot buffer.
    pub offset: u64,
}

/// The storing half: keep this pass's pooled key rows and the session's raw key
/// ring current.
///
/// The key projection, then the pooled pass over it: once for a chunk, once
/// per run of a segmented pass, once per row of a batch. Its launch addresses
/// the cache through a page table and the ring through a session, which is what
/// makes it the part of an attention block that cannot be shared between two
/// sessions. Neither half is optional: a pass that skipped it would leave a
/// hole in the table for the positions it wrote, and a later pass would select
/// blocks whose rows were never filled in.
///
/// # Errors
///
/// Whatever [`matmul::project`] reports for the key projection.
pub fn store(
    shape: &SparseShape,
    desc: &IndexerDesc,
    slot: IndexerSlot,
    origin: RowOrigin,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let mut plan = Plan::new();
    matmul::project(
        &mut plan,
        desc.k,
        Bound::new(
            WeightRef::layer(slot.layer, Role::IndexerKey),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::IdxKey),
        ),
        n_tok,
        device,
    )?;
    plan.append(pool(shape, slot, &origin, n_tok));
    Ok(plan)
}

/// The pooled key pass, once for a chunk, once per run and once per row of a
/// batch.
///
/// It has no batched sibling and could not have one without a different kernel:
/// its grid and its block range are derived from the first and the last position
/// of the launch - `pos[0] / r` to `(pos[n_tok - 1] + 1) / r` - which for rows
/// that belong to different sequences is not a range at all. So a batch launches
/// it per row, each over one position, one page table, one ring and one
/// lookback group.
///
/// One block per launch at one row, on a kernel whose body is an average, a norm
/// and a rotation over a hundred and twenty-eight values: the launches cost more
/// than the work, which is exactly the trade the other five batched kernels were
/// written to avoid and exactly the one that does not matter here, because it is
/// one launch per row per attention slot rather than one per row per layer. It
/// is also the only launch of the storing half: the raw keys no longer have a
/// store of their own.
fn pool(shape: &SparseShape, slot: IndexerSlot, origin: &RowOrigin, n_tok: u32) -> Plan {
    let IndexerSlot {
        layer, rank, tap, ..
    } = slot;
    let mut plan = Plan::new();
    let batched = *origin == RowOrigin::Unrelated;
    let segments = match origin {
        RowOrigin::Segmented(runs) => Some(runs),
        RowOrigin::Sequential | RowOrigin::Unrelated => None,
    };
    let rows = match segments {
        Some(runs) => runs.count(),
        None if batched => n_tok,
        None => 1,
    };
    let key_row = u64::from(shape.indexer.head_dim) * 4;
    for row in 0..rows {
        // A chunk's launch covers every row and reads the table whole, from the
        // prefix the writer put in front of the chunk's first position. A
        // batched row's covers one position and reads its own group, and a
        // segmented run's covers its own rows from its own group. Each reads
        // its own rows of the key projection and its own session's ring.
        let (pos, pages, table, covered, keys, ring) = if let Some(runs) = segments {
            let first = u64::from(runs.start(row));
            (
                Operand::ScratchFrom(Buf::Positions, first * I32_BYTES),
                Operand::PagesOf(row),
                Operand::ScratchFrom(
                    Buf::Mrope,
                    crate::sparse::inputs::segment_lookback_offset(runs, row),
                ),
                runs.len(row),
                Operand::ScratchFrom(Buf::IdxKey, first * key_row),
                Operand::IndexerRingOf { rank, row },
            )
        } else if batched {
            (
                Operand::ScratchFrom(Buf::Positions, u64::from(row) * I32_BYTES),
                Operand::PagesOf(row),
                Operand::ScratchFrom(Buf::Mrope, batch_lookback_offset(n_tok, row)),
                1,
                Operand::ScratchFrom(Buf::IdxKey, u64::from(row) * key_row),
                Operand::IndexerRingOf { rank, row },
            )
        } else {
            (
                Operand::Scratch(Buf::Positions),
                Operand::Pages,
                Operand::Scratch(Buf::Mrope),
                n_tok,
                Operand::Scratch(Buf::IdxKey),
                Operand::IndexerRing(rank),
            )
        };
        // A pass with no tap still hands the kernel a real buffer for the copy
        // it will not make - its own ring - so no backend has to bind a null.
        let (snap, snap_row) = match tap {
            Some(tap) if segments.is_none() && !batched => {
                (Operand::Snapshot(tap.offset), Operand::U32(tap.row))
            }
            _ => (ring, Operand::U32(IDX_NO_TAP)),
        };
        plan.launch(
            Entry::plain(MOD_IDXRING, "q4e_idx_ring_pool_kernel"),
            kernels::idx_ring_pool(covered, shape.indexer.compress),
            // `pooled, ring, k, w, pos, pages, n_tok, n_rot, rope_base,
            //  mrope_all, sec_t, sec_h, sec_w, eps, snap, snap_row` - and this
            // is the one launch that reads the rotary table from a lookback
            // rather than from a row's own position.
            Args::of(&[
                Operand::CachePlane {
                    rank,
                    plane: Plane::IndexerPooled,
                },
                ring,
                keys,
                Operand::Weight(WeightRef::layer(layer, Role::IndexerKeyNorm)),
                pos,
                pages,
                Operand::U32(covered),
                Operand::U32(shape.gqa.n_rot),
                Operand::f32(shape.rope_theta),
                table,
                Operand::U32(shape.mrope.time),
                Operand::U32(shape.mrope.height),
                Operand::U32(shape.mrope.width),
                Operand::f32(shape.rms_eps),
                snap,
                snap_row,
            ]),
        );
    }
    plan
}

/// The selecting half: score the pooled blocks, keep the best, expand them into
/// positions.
///
/// A chunk wider than one selection pass is selected in passes of that width,
/// each bounded by the position of its own last row. Each pass is its own set
/// of launches, which is why they are planned rather than collapsed into one:
/// the grids differ on the last, narrower pass, and every launch in a pass
/// reads that pass's own rows.
///
/// # Errors
///
/// Whatever [`matmul::project`] reports for the query projection.
pub fn select(
    shape: &SparseShape,
    desc: &IndexerDesc,
    slot: IndexerSlot,
    scorer: Scorer,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let mut plan = select_queries(shape, desc, slot, n_tok, device)?;
    plan.append(select_rows(
        shape,
        slot,
        scorer,
        RowSpan {
            base: 0,
            n: n_tok,
            pages: Operand::Pages,
        },
    ));
    Ok(plan)
}

/// The rows a selection covers: `n` rows of the pass from row `base`, all of
/// them one sequence's, read through `pages`.
///
/// A chunk's selection is its rows from the first, through the pass's one
/// table. A batch's is one row at a time, through that row's own: its rows
/// are unrelated sequences, and a selection bounds its scores by the position
/// of the last row it covers.
#[derive(Debug, Clone, Copy)]
pub struct RowSpan {
    /// The first row.
    pub base: u32,
    /// Rows covered.
    pub n: u32,
    /// The page table they read.
    pub pages: Operand,
}

impl RowSpan {
    /// Row `row` of a batch, alone, through its own page table.
    #[must_use]
    pub const fn one(row: u32) -> Self {
        Self {
            base: row,
            n: 1,
            pages: Operand::PagesOf(row),
        }
    }
}

/// The query half of the selection: every row's indexer query, projected,
/// normed and rotated. Per-row arithmetic, so a batch runs it once over all
/// its rows.
///
/// # Errors
///
/// Whatever [`matmul::project`] reports for the query projection.
pub fn select_queries(
    shape: &SparseShape,
    desc: &IndexerDesc,
    slot: IndexerSlot,
    n_tok: u32,
    device: Device,
) -> CoreResult<Plan> {
    let IndexerSlot { layer, .. } = slot;
    let mut plan = Plan::new();
    matmul::project(
        &mut plan,
        desc.q,
        Bound::new(
            WeightRef::layer(layer, Role::IndexerQuery),
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::IdxQuery),
        ),
        n_tok,
        device,
    )?;
    plan.launch(
        Entry::plain(MOD_IDX, "q4e_idx_q_kernel"),
        kernels::idx_q(shape.indexer.heads, n_tok),
        // `qn, q, w, mrope, sec_t, sec_h, sec_w, n_head, n_rot, rope_base, eps`
        Args::of(&[
            Operand::Scratch(Buf::IdxQueryNorm),
            Operand::Scratch(Buf::IdxQuery),
            Operand::Weight(WeightRef::layer(layer, Role::IndexerQueryNorm)),
            Operand::ScratchFrom(Buf::Mrope, MROPE_PREFIX_BYTES),
            Operand::U32(shape.mrope.time),
            Operand::U32(shape.mrope.height),
            Operand::U32(shape.mrope.width),
            Operand::U32(shape.indexer.heads),
            Operand::U32(shape.gqa.n_rot),
            Operand::f32(shape.rope_theta),
            Operand::f32(shape.rms_eps),
        ]),
    );
    Ok(plan)
}

/// The scoring half of the selection over the rows `span` names: score the
/// pooled blocks, keep the best, expand them into positions, each row's lists
/// landing at that row of the pass.
#[must_use]
pub fn select_rows(shape: &SparseShape, slot: IndexerSlot, scorer: Scorer, span: RowSpan) -> Plan {
    let IndexerSlot { rank, ctx, .. } = slot;
    let RowSpan {
        base,
        n: n_tok,
        pages,
    } = span;
    let mut plan = Plan::new();
    // Two different block counts, and swapping them is silent: `ranked` is how
    // many blocks the scorer ranks, and how wide one row of the score buffer
    // is; `keep` is how many of them survive the top-k. Below the budget every
    // visible block fits in either, which is why a stride taken from the budget
    // answers correctly until the context passes it - and then selects out of
    // the oldest `top_k` positions for ever after, with nothing failing.
    let ranked = shape.indexer.score_blocks(ctx);
    let keep = shape.indexer.top_k / kernels::IDX_R.max(1);
    let mut done = 0;
    while done < n_tok {
        let rows = (n_tok - done).min(kernels::IDX_QUERY_BLOCK);
        // Every pass reads its own query rows and its own positions; the score
        // bound is the position of the pass's last row, which is one slot of
        // the position row rather than a buffer of its own.
        let first = base + done;
        let qn = Operand::ScratchFrom(
            Buf::IdxQueryNorm,
            u64::from(first) * u64::from(shape.indexer.heads) * u64::from(kernels::IDX_DIM) * 2,
        );
        let pos = Operand::ScratchFrom(Buf::Positions, u64::from(first) * 4);
        let last = Operand::ScratchFrom(Buf::Positions, u64::from(first + rows - 1) * 4);
        // The selection each pass produces is its own rows' and is read by the
        // attention after every pass has run, so it lands at the pass's own row
        // offset. The scores are consumed by the very next launch and can share
        // one slab. Written at offset zero, every pass after the first
        // overwrote rows 0..512 and the rest of a 2048-row chunk attended over
        // whatever token lists an earlier request had left there.
        let row = |width: u64| u64::from(first) * width * 4;
        let selected = Operand::ScratchFrom(Buf::IdxSelected, row(u64::from(keep)));
        let count = Operand::ScratchFrom(Buf::IdxCount, row(1));
        let tokens = Operand::ScratchFrom(Buf::IdxTokens, row(u64::from(kernels::IDX_WIDTH)));
        let token_count = Operand::ScratchFrom(Buf::IdxTokenCount, row(1));
        match scorer {
            // `score, qn, pooled, pos, pages, n_head, n_tok, n_blocks_stride, pos_last`
            Scorer::Scalar => plan.launch(
                Entry::plain(MOD_IDX, "q4e_idx_score_kernel"),
                kernels::idx_score(shape.indexer.heads, rows, ranked),
                Args::of(&[
                    Operand::Scratch(Buf::IdxScore),
                    qn,
                    Operand::CachePlane {
                        rank,
                        plane: Plane::IndexerPooled,
                    },
                    pos,
                    pages,
                    Operand::U32(shape.indexer.heads),
                    Operand::U32(rows),
                    Operand::U32(ranked),
                    last,
                ]),
            ),
            // The tensor-core scorer is written for one head count, so it takes
            // no head argument at all.
            Scorer::TensorCore => plan.launch(
                Entry::plain(MOD_IDX, "q4e_idx_score_tc_kernel"),
                kernels::idx_score_tc(rows, ranked),
                Args::of(&[
                    Operand::Scratch(Buf::IdxScore),
                    qn,
                    Operand::CachePlane {
                        rank,
                        plane: Plane::IndexerPooled,
                    },
                    pos,
                    pages,
                    Operand::U32(rows),
                    Operand::U32(ranked),
                    last,
                ]),
            ),
        }
        plan.launch(
            Entry::plain(MOD_IDX, "q4e_idx_topk_kernel"),
            kernels::idx_topk(rows),
            // `sel, cnt, score, n_blocks_stride, pos_last, k`
            Args::of(&[
                selected,
                count,
                Operand::Scratch(Buf::IdxScore),
                Operand::U32(ranked),
                last,
                Operand::U32(keep),
            ]),
        );
        plan.launch(
            Entry::plain(MOD_IDX, "q4e_idx_expand_kernel"),
            kernels::idx_expand(rows),
            // `tokens, n_sel, sel, cnt, pos, k, width`
            Args::of(&[
                tokens,
                token_count,
                selected,
                count,
                pos,
                Operand::U32(keep),
                Operand::U32(kernels::IDX_WIDTH),
            ]),
        );
        done = done.saturating_add(rows);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tensor_core_scorer_is_reached_at_one_head_count_and_only_when_granted() {
        assert_eq!(Scorer::choose(4, true), Scorer::TensorCore);
        assert_eq!(Scorer::choose(4, false), Scorer::Scalar);
        assert_eq!(Scorer::choose(8, true), Scorer::Scalar);
    }

    /// The scorer ranks every block the context holds, not the budget's worth.
    ///
    /// The two are the same number until the context outgrows the budget -
    /// which is the only region where the sparse path runs at all - so a stride
    /// taken from the budget is right everywhere it is ever checked by hand and
    /// wrong everywhere it matters: the blocks past the budget are never
    /// ranked, so the selection is the oldest `top_k` positions, and nothing
    /// fails.
    #[test]
    fn the_scorer_ranks_the_whole_context_and_not_the_selection_budget() {
        // The shipping indexer: 2,048 positions kept, pooled four to a block.
        let indexer = crate::sparse::shape::IndexerShape::new(4, 128, 2_048, 4);
        assert_eq!(indexer.budget_blocks(), 512);
        assert_eq!(indexer.score_blocks(2_048), 513);
        assert_eq!(indexer.score_blocks(16_384), 4_097);
        assert!(
            indexer.score_blocks(16_384) > indexer.budget_blocks(),
            "a context past the budget must rank more blocks than it keeps"
        );
        // And the degenerate geometry answers zero rather than dividing.
        assert_eq!(
            crate::sparse::shape::IndexerShape::new(4, 128, 2_048, 0).score_blocks(16_384),
            0
        );
    }
}

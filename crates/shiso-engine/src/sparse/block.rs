//! A block drafter beside the sparse stack: [`crate::block_draft`]'s drafter,
//! fed from this stack's passes and drafting through its embedding table and
//! vocabulary head.
//!
//! # What the stack owes the drafter
//!
//! Its taps: for every row of every target pass, each tapped layer's output
//! residual, which here is several streams wide. The drafter reads one row a
//! layer - the mean over the streams, which is what it was trained against - so
//! the stack writes that mean, not the streams, into [`Buf::BlockTaps`]: one
//! launch at the end of each tapped layer's routed island, inside the island,
//! so a captured island replays it with everything else and a replayed pass
//! taps exactly as an encoded one does. A stack with no drafter plans none of
//! these launches, and its passes are the passes it always ran.
//!
//! Then, after every target pass, the drafter absorbs the pass's rows at the
//! positions they ran at: a prompt chunk's, a decode step's, a verify's. A
//! verify's rejected rows are absorbed too, at positions past where the session
//! is wound back to, and that is not a leak: the next block the drafter
//! proposes writes its own keys and values at exactly those positions before
//! any query reads them, as [`crate::block_draft`] lays out for the dense lane.
//!
//! # Where its cache lives
//!
//! In a ring of a few pages per session ([`ring_pages`]), not in the stack's
//! pool. The drafter attends over its window and no further, so a key older
//! than that is never read again: one page table shared by every session
//! ([`ring_table`]) folds page `p` onto ring page `p % ring_pages`, and the ring
//! is deep enough that the pages a draft reads never fold onto each other.
//! Riding along with every pool page instead cost the pool about 40% of its
//! tokens on Qwen3.8-Flash (5.2 MB of drafter keys and values beside 6.5 MB of
//! the target's per page), all of it for positions the drafter cannot see.
//!
//! What a resumed session needs of the ring travels in its checkpoint blob,
//! after the indexer's ring: a resume restores the recurrent state at the
//! position it lands on, and the drafter's window behind that position is the
//! same kind of state. A resume can only land on a checkpoint, so every
//! resumed conversation drafts from the context it left off with.
//!
//! # What the stack lends it
//!
//! The embedding gather and the vocabulary projection, as plans over this
//! stack's own scratch ([`SparseTarget`]), and a sliced top on the device, so
//! the selector reads a few hundred candidates a drafted row - merged on the
//! host to its sixteen - rather than a megabyte a row and a host scan of it.

use shiso_core::DeviceView;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::DraftHint;

use std::sync::Arc;

use shiso_core::Backend;
use shiso_core::buffer::{DeviceBuffer, alloc_tagged};
use shiso_core::ledger::{self, MemTag};
use shiso_core::page::{PAGE_TOKENS, pages_for};

use crate::block_draft::{
    ABSORB_ROWS, BesideTarget, BlockDraftFootprint, BlockDraftParts, BlockDrafter, Candidates,
    DraftCache, DraftTarget,
};
use crate::kv::DeviceKv;
use crate::sparse::bind::{Args, Operand, Role, WeightRef};
use crate::sparse::gather;
use crate::sparse::kernels::{self, Entry, F32_BYTES, GDN_CHUNK, MOD_MISC};
use crate::sparse::matmul::{self, Bound};
use crate::sparse::moe::RowOrigin;
use crate::sparse::plan::Plan;
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::stack::{Pass, SparseStack};

/// Pages of one session's drafter ring.
///
/// Deep enough that no two pages a read can reach fold onto one ring page. A
/// draft at `at` reads from `at + 1 - window` to its block's last row, and a
/// checkpoint cut on the recurrence's grid inside a prompt chunk is taken after
/// the drafter absorbed the whole chunk, up to [`GDN_CHUNK`] rows past it - so
/// the span that must survive is the window and the wider of those two, and
/// it can straddle one page more than it fills.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a drafter with no window, which attends over
/// the whole context and so has no ring that would do.
pub(crate) fn ring_pages(footprint: &BlockDraftFootprint) -> CoreResult<u32> {
    if footprint.window() == 0 {
        return Err(CoreError::Unsupported {
            what: "a block drafter that attends over the whole context beside a stack that \
                   keeps only a drafter's window",
            backend: "engine",
        });
    }
    Ok(ring_pages_for(footprint.window(), footprint.block()))
}

/// [`ring_pages`] for a window and a block.
fn ring_pages_for(window: u32, block: u32) -> u32 {
    window
        .saturating_add(block.max(GDN_CHUNK))
        .div_ceil(PAGE_TOKENS)
        .saturating_add(1)
}

/// Bytes of one session's drafter ring, which is also what it adds to the
/// session's checkpoint blob.
///
/// # Errors
///
/// As [`ring_pages`].
pub(crate) fn ring_bytes(footprint: &BlockDraftFootprint) -> CoreResult<u64> {
    Ok(u64::from(ring_pages(footprint)?).saturating_mul(footprint.page_bytes()))
}

/// Opens one session's drafter ring, zeroed.
///
/// Zeroed although no draft reads a row nothing wrote: a fresh allocation is
/// the last tenant's bytes, and a ring restored from a blob is overwritten
/// whole anyway, so this costs one clear per session for a guarantee that does
/// not depend on that argument staying true.
///
/// # Errors
///
/// As [`ring_pages`], and [`CoreError::Backend`] when an allocation fails.
pub(crate) fn open_ring(
    backend: &Arc<dyn Backend>,
    footprint: &BlockDraftFootprint,
) -> CoreResult<DeviceKv> {
    let pages = ring_pages(footprint)?;
    let ring = {
        let _tag = ledger::scope(MemTag::KvDraft);
        DeviceKv::open(backend, &footprint.layout(), pages)?
    };
    for run in ring.runs() {
        backend.fill_zero(run)?;
    }
    Ok(ring)
}

/// The page table every session's drafter ring is read through: entry `p` is
/// ring page `p % ring_pages`, for every page of a `ctx`-position session.
///
/// One table for every session, because the ring pages are each session's own
/// runs and the fold is the same arithmetic for all of them.
///
/// # Errors
///
/// As [`ring_pages`], and [`CoreError::Backend`] when the allocation or the
/// upload fails.
pub(crate) fn ring_table(
    backend: &Arc<dyn Backend>,
    footprint: &BlockDraftFootprint,
    ctx: u32,
) -> CoreResult<DeviceBuffer> {
    let ring = ring_pages(footprint)?;
    let entries = pages_for(ctx);
    let bytes: Vec<u8> = (0..entries)
        .flat_map(|page| i32::try_from(page % ring).unwrap_or(0).to_le_bytes())
        .collect();
    let table = alloc_tagged(backend, MemTag::PageTables, bytes.len().max(4) as u64)?;
    backend.copy_h2d(table.view().slice(0, bytes.len() as u64)?, &bytes)?;
    Ok(table)
}

/// The launch that writes layer `layer`'s taps for a pass of `n_tok` rows, or
/// `None` when the drafter does not tap that layer.
#[must_use]
pub(crate) fn tap(
    shape: &SparseShape,
    footprint: &BlockDraftFootprint,
    layer: u32,
    n_tok: u32,
) -> Option<Plan> {
    let slot = footprint.taps().iter().position(|&t| t == layer)?;
    let n_taps = u32::try_from(footprint.taps().len()).unwrap_or(u32::MAX);
    let slot = u32::try_from(slot).unwrap_or(u32::MAX);
    let mut plan = Plan::new();
    plan.launch(
        Entry::plain(MOD_MISC, "q4e_stream_mean_rows_kernel"),
        kernels::elementwise(u64::from(n_tok) * u64::from(shape.hidden)),
        // `dst, res, n_embd, n_hc, n_tok, dst_stride, dst_offset`
        Args::of(&[
            Operand::Scratch(Buf::BlockTaps),
            Operand::Scratch(Buf::Residual),
            Operand::U32(shape.hidden),
            Operand::U32(shape.streams),
            Operand::U32(n_tok),
            Operand::U32(n_taps * shape.hidden),
            Operand::U32(slot * shape.hidden),
        ]),
    );
    Some(plan)
}

/// Slices each drafted logits row's top is taken over: a block each, so seven
/// rows keep 224 blocks in flight on a 48-SM device.
const TOP_SLICES: u32 = 32;

/// The sparse stack as a block drafter's target, for one session's draft.
struct SparseTarget<'a> {
    stack: &'a SparseStack,
    session: SessionId,
}

impl DraftTarget for SparseTarget<'_> {
    fn embed(
        &self,
        backend: &dyn Backend,
        ids: &[TokenId],
        _drafter: &crate::layers::Scratch,
        dst: DeviceView<'_>,
    ) -> CoreResult<()> {
        let stack = self.stack;
        let n = u32::try_from(ids.len()).unwrap_or(u32::MAX);
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|id| i32::try_from(id.get()).unwrap_or(0).to_le_bytes())
            .collect();
        backend.copy_h2d(
            stack.scratch_view(Buf::Ids)?.slice(0, bytes.len() as u64)?,
            &bytes,
        )?;
        let plan = gather::rows(
            &stack.shape,
            stack.desc.token_embd.dtype,
            gather::Bound::new(
                Operand::Scratch(Buf::Ids),
                Operand::Weight(WeightRef::model(Role::TokenEmbd)),
                Operand::Scratch(Buf::Embed),
            ),
            n,
        )?;
        stack.executor(self.session)?.run_steps(plan.steps())?;
        let row = u64::from(stack.shape.hidden) * F32_BYTES;
        backend.copy_d2d(
            dst,
            stack
                .scratch_view(Buf::Embed)?
                .slice(0, u64::from(n) * row)?,
        )
    }

    fn candidates(
        &self,
        backend: &dyn Backend,
        normed: DeviceView<'_>,
        rows: u32,
        k: u32,
    ) -> CoreResult<Vec<Candidates>> {
        let stack = self.stack;
        let row = u64::from(stack.shape.hidden) * F32_BYTES;
        backend.copy_d2d(
            stack
                .scratch_view(Buf::Mixed)?
                .slice(0, u64::from(rows) * row)?,
            normed,
        )?;
        // Through the narrowed head when the build has one: a candidate the
        // slice cannot reach is a guess the target rejects, which costs a
        // verify row and never an answer, and the full head is most of what a
        // draft would otherwise cost (2.2 of 5.3 ms a block on a GB10).
        let narrow = stack.resident()?.weights.block_vocabulary.as_ref();
        let (role, output, width) = match narrow {
            Some(slice) => (
                Role::BlockVocabulary,
                crate::sparse::weights::WeightDesc::from(slice.weight()),
                slice.rows(),
            ),
            None => (Role::Output, stack.desc.output, stack.shape.vocab),
        };
        let mut plan = Plan::new();
        matmul::project(
            &mut plan,
            output,
            Bound::new(
                WeightRef::model(role),
                Operand::Scratch(Buf::Mixed),
                Operand::Scratch(Buf::Logits),
            ),
            rows,
            stack.device(),
        )?;
        // Each slice's best `k`, merged below: the whole row's best `k`, in
        // the same order, read as `slices * k` entries a row rather than the
        // row. One block a slice keeps every SM busy over seven rows.
        let slices = TOP_SLICES;
        let entries = u64::from(rows) * u64::from(slices) * u64::from(k);
        let ids_at = entries * F32_BYTES;
        let room = stack.scratch_view(Buf::RowTop)?.len();
        if 2 * ids_at > room {
            return Err(CoreError::ShapeMismatch {
                field: "block drafter candidate slices",
                expected: room,
                got: 2 * ids_at,
            });
        }
        plan.launch(
            Entry::plain(MOD_MISC, "q4e_top_slices_kernel"),
            kernels::Launch::new(
                shiso_core::Extent3::new(slices, rows, 1),
                kernels::BLOCK_LINEAR,
                0,
            ),
            // `logits, n_vocab, k, slices, values, ids`
            Args::of(&[
                Operand::Scratch(Buf::Logits),
                Operand::U32(width),
                Operand::U32(k),
                Operand::U32(slices),
                Operand::Scratch(Buf::RowTop),
                Operand::ScratchFrom(Buf::RowTop, ids_at),
            ]),
        );
        let (plan, _) = crate::sparse::passes::with_reservation(plan, Vec::new());
        plan.validate()?;
        stack.executor(self.session)?.run_steps(plan.steps())?;
        let mut bytes = vec![0u8; usize::try_from(2 * ids_at).unwrap_or(0)];
        backend.copy_d2h(
            stack.scratch_view(Buf::RowTop)?.slice(0, 2 * ids_at)?,
            &mut bytes,
        )?;
        let (values, ids) = bytes.split_at(usize::try_from(ids_at).unwrap_or(0));
        let values = values.as_chunks::<4>().0;
        let ids = ids.as_chunks::<4>().0;
        let per_row = (slices * k) as usize;
        // A slice's rows are its ids' positions; the whole head's are the ids.
        let id_of = |local: u32| match narrow {
            Some(slice) => slice.id(local),
            None => (local < width).then_some(local),
        };
        Ok((0..rows as usize)
            .map(|r| {
                let at = r * per_row..(r + 1) * per_row;
                let mut row: Vec<(f32, u32)> = values[at.clone()]
                    .iter()
                    .zip(&ids[at])
                    .filter_map(|(v, i)| {
                        id_of(u32::from_le_bytes(*i)).map(|id| (f32::from_le_bytes(*v), id))
                    })
                    .collect();
                row.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                row.truncate(k as usize);
                Candidates {
                    ids: row.iter().map(|(_, id)| *id).collect(),
                    logits: row.iter().map(|(v, _)| *v).collect(),
                }
            })
            .collect())
    }
}

impl SparseStack {
    /// Makes `parts` this stack's block drafter.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] on a stack built without a speculative depth,
    /// whose sessions keep no rollback records, on one that carries a draft
    /// head - a stack drafts one way - and on one whose options did not name
    /// this drafter's footprint, whose passes therefore write no taps and
    /// whose sessions open no ring for it;
    /// [`CoreError::ShapeMismatch`] for a drafter this stack cannot feed.
    pub fn bind_block_drafter(&mut self, parts: BlockDraftParts) -> CoreResult<()> {
        if !self.opts.spec.is_speculative() {
            return Err(CoreError::Unsupported {
                what: "a block drafter on a stack whose sessions were opened with no rollback \
                       records, which is what a speculative depth of zero asks for",
                backend: "engine",
            });
        }
        if self.has_draft_head() {
            return Err(CoreError::Unsupported {
                what: "a block drafter on a stack that already carries a draft head",
                backend: "engine",
            });
        }
        let footprint = parts.footprint(self.shape.vocab, ABSORB_ROWS)?;
        if self.opts.block_draft != Some(footprint) {
            return Err(CoreError::Unsupported {
                what: "a block drafter the stack was not built for: its options name another \
                       footprint or none, so its passes tap other layers and its sessions keep \
                       no ring for the drafter",
                backend: "engine",
            });
        }
        let layers = u32::try_from(self.shape.layers.len()).unwrap_or(u32::MAX);
        if self.block_table.is_none() {
            return Err(CoreError::Unsupported {
                what: "a block drafter on a stack that opened no ring table for it",
                backend: "engine",
            });
        }
        let drafter = BlockDrafter::new_beside(
            &self.backend,
            parts,
            BesideTarget {
                hidden: self.shape.hidden,
                layers,
                vocab: self.shape.vocab,
            },
        )?;
        self.block = Some(drafter);
        Ok(())
    }

    /// Whether a block drafter is bound.
    #[must_use]
    pub const fn has_block_drafter(&self) -> bool {
        self.block.is_some()
    }

    /// Whether anything drafts for this stack's sessions: a draft head or a
    /// block drafter.
    #[must_use]
    pub fn drafts(&self) -> bool {
        self.has_draft_head() || self.has_block_drafter()
    }

    /// `session`'s drafter ring, read through the shared ring table.
    fn ring_cache(&self, session: SessionId) -> CoreResult<DraftCache<'_>> {
        let missing = CoreError::Unsupported {
            what: "a block draft for a session opened with no drafter ring",
            backend: "engine",
        };
        let kv = self.session(session)?.block_ring.as_ref().ok_or(missing)?;
        let pages = self
            .block_table
            .as_ref()
            .ok_or(CoreError::Unsupported {
                what: "a block draft on a stack that opened no ring table",
                backend: "engine",
            })?
            .view();
        Ok(DraftCache { kv, pages })
    }

    /// Hands the drafter every row of the target pass that just ran, each run
    /// of it at the positions its session ran it at.
    ///
    /// Called with the sessions' frontiers where the pass found them: a pass
    /// commits nothing itself, so a session's frontier is still its first
    /// row's position.
    pub(crate) fn absorb_pass(&self, sessions: &[SessionId], pass: &Pass) -> CoreResult<()> {
        let Some(drafter) = self.block.as_ref() else {
            return Ok(());
        };
        // Traced, the pass is waited for first, so the time logged below is
        // the absorb's own rather than the tail of the pass it follows.
        if tracing::enabled!(tracing::Level::TRACE) {
            self.backend.sync()?;
        }
        let started = std::time::Instant::now();
        let taps = self.scratch_view(Buf::BlockTaps)?;
        let absorb = |session: SessionId, from: u32, n: u32| -> CoreResult<()> {
            let live = self.session(session)?;
            if !live.spec_depth().is_speculative() {
                return Ok(());
            }
            let cache = self.ring_cache(session)?;
            drafter.absorb(&self.backend, taps, cache, (from, n), live.written())
        };
        let absorbed = match pass.origin {
            RowOrigin::Sequential => {
                let session = *sessions.first().ok_or(CoreError::ShapeMismatch {
                    field: "pass sessions",
                    expected: 1,
                    got: 0,
                })?;
                absorb(session, 0, pass.rows)
            }
            RowOrigin::Unrelated => {
                for (row, &session) in sessions.iter().enumerate() {
                    absorb(session, u32::try_from(row).unwrap_or(u32::MAX), 1)?;
                }
                Ok(())
            }
            RowOrigin::Segmented(runs) => {
                for (run, &session) in (0..runs.count()).zip(sessions) {
                    absorb(session, runs.start(run), runs.len(run))?;
                }
                Ok(())
            }
        };
        tracing::trace!(
            rows = pass.rows,
            absorb_ms = started.elapsed().as_secs_f64() * 1e3,
            "block drafter absorbed a pass"
        );
        absorbed
    }

    /// The block drafter's drafts behind `first`, which sits at `frontier`:
    /// at most `budget`, cut at the first the drafter doubts.
    // None: the drafter did not run - no budget, a round drafting is not
    // expected to pay for, or a block that would pass the context - which a
    // round's cost estimate tells apart from a block cut to nothing.
    pub(crate) fn block_drafts(
        &mut self,
        session: SessionId,
        first: TokenId,
        frontier: u32,
        hint: DraftHint,
        budget: u32,
    ) -> CoreResult<Option<Vec<TokenId>>> {
        if budget == 0 {
            return Ok(None);
        }
        let generated_from = self.session(session)?.written();
        let pays = match self.block.as_mut() {
            Some(drafter) => {
                drafter.track(session, generated_from);
                drafter.pays(session)
            }
            None => return Ok(None),
        };
        if !pays {
            return Ok(None);
        }
        // The block writes its own keys and values at every position it
        // proposes, in the session's ring, which needs no claim. A block that
        // would run past the context drafts nothing; the target's last few
        // positions decode one at a time.
        let reach = frontier.saturating_add(self.block.as_ref().map_or(0, BlockDrafter::block));
        if reach > self.opts.engine.ctx {
            return Ok(None);
        }
        let drafter = self.block.as_ref().ok_or(CoreError::Unsupported {
            what: "a block draft on a stack that carries no block drafter",
            backend: "engine",
        })?;
        let cache = self.ring_cache(session)?;
        let target = SparseTarget {
            stack: self,
            session,
        };
        let started = std::time::Instant::now();
        let (mut drafts, sure) =
            drafter.propose(&self.backend, &target, cache, (first, frontier), hint)?;
        // Cut at the first guess the drafter itself doubts, as a head's chain
        // stops there: every guess after it can only be kept if it is, and
        // each is a verify row, which on this stack is what a drafted round
        // mostly costs.
        let threshold = self.opts.draft_confidence;
        let sure_run = if threshold > 0.0 {
            sure.iter().take_while(|p| **p >= threshold).count()
        } else {
            drafts.len()
        };
        drafts.truncate(sure_run.min(budget as usize));
        tracing::debug!(
            %session,
            propose_ms = started.elapsed().as_secs_f64() * 1e3,
            kept = drafts.len(),
            "block drafter proposed"
        );
        Ok(Some(drafts))
    }
}

#[cfg(test)]
mod tests {
    use super::{GDN_CHUNK, PAGE_TOKENS, ring_pages_for};

    /// Whether the positions `lo .. hi` fold onto distinct ring pages.
    fn distinct(lo: u32, hi: u32, ring: u32) -> bool {
        let pages: Vec<u32> = (lo / PAGE_TOKENS..=(hi - 1) / PAGE_TOKENS).collect();
        let mut folded: Vec<u32> = pages.iter().map(|p| p % ring).collect();
        folded.sort_unstable();
        folded.dedup();
        folded.len() == pages.len()
    }

    #[test]
    fn no_page_a_draft_or_a_grid_checkpoint_reads_folds_onto_another() {
        for (window, block) in [(2048, 8), (2048, 16), (1000, 8), (4096, 8), (256, 8)] {
            let ring = ring_pages_for(window, block);
            for at in window..window + 4 * PAGE_TOKENS {
                // A draft at `at` reads from `at + 1 - window` through its
                // block's last row.
                assert!(
                    distinct(at + 1 - window, at + block, ring),
                    "window {window} block {block}: a draft at {at} folds"
                );
                // A grid checkpoint at `at` keeps its window while the chunk
                // runs up to a grid's width past it.
                assert!(
                    distinct(at + 1 - window, at + GDN_CHUNK, ring),
                    "window {window} block {block}: a checkpoint at {at} folds"
                );
            }
        }
    }

    #[test]
    fn the_qwen38_drafter_keeps_ten_pages_a_session() {
        assert_eq!(ring_pages_for(2048, 8), 10);
    }
}

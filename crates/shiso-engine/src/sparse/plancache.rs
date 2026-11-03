//! Reusing a step's plan from one token to the next.
//!
//! Planning a decode step walks every layer of the stack and builds about
//! fifteen hundred steps, and it cost 120-130 us of host time a token on a
//! GB10 - time the device spent idle, because a step is launched after it is
//! planned. And the plan it builds is the plan it built a token earlier: a
//! sequential pass names its buffers symbolically and reads every position out
//! of the inputs it is handed, so for one session at one row count the plan
//! changes only where [`SparseStack::sequential_attention`](crate::sparse::stack::SparseStack::sequential_attention) says it does -
//! when the indexer's selection goes live, and on the one step whose rows
//! straddle that point.
//!
//! So a step pass is kept, keyed by exactly the facts it was planned from, and
//! handed back the next time those facts recur. The one part of a pass that is
//! per position - the n-gram gather - is replanned every time; it is a few
//! integers.
//!
//! # What the key has to hold
//!
//! Everything [`SparseStack::plan_chunk_media`](crate::sparse::stack::SparseStack::plan_chunk_media) reads that can differ between
//! two calls on one stack: the session (whose handle the islands are filed
//! under), the row count, the logit rows, and the two frontier facts. The
//! stack's geometry, options and weights are fixed at construction. A media
//! chunk and a prompt pass are never cached: the first plans a splice of a
//! request's own encoder block, and the second is read once.
//!
//! A cache that keyed on less would hand a step the plan of another row count
//! or another session, and every launch of it would succeed.

use shiso_core::ids::SessionId;

use crate::sparse::stack::Pass;

/// Step passes kept at once.
///
/// One for the decode step, and room for a speculative build's verify and its
/// one-row fallback beside it, so the two do not evict each other every token;
/// a fourth for a second session decoding alone between batches.
const KEPT: usize = 4;

/// The facts a sequential step pass was planned from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlanKey {
    /// The session the pass is issued for, whose handle its islands are filed
    /// under.
    pub(crate) session: SessionId,
    /// Rows in the pass.
    pub(crate) rows: u32,
    /// Rows the vocabulary projection runs on.
    pub(crate) logit_rows: u32,
    /// Whether the attention reads the indexer's selection.
    pub(crate) selected: bool,
    /// Front rows that still attend densely.
    pub(crate) dense_rows: u32,
}

/// The kept passes, most recently used last.
#[derive(Debug, Default)]
pub(crate) struct PlanCache {
    kept: Vec<(PlanKey, Pass)>,
}

impl PlanCache {
    /// Takes the pass planned for `key` out of the cache, when there is one.
    ///
    /// Taken rather than borrowed: the caller fills in the gather and issues
    /// it, and hands it back with [`PlanCache::put`] once it has run. A pass
    /// whose run failed is simply never handed back.
    pub(crate) fn take(&mut self, key: PlanKey) -> Option<Pass> {
        let at = self.kept.iter().position(|(kept, _)| *kept == key)?;
        Some(self.kept.remove(at).1)
    }

    /// Keeps `pass` for `key`, dropping the least recently used pass past
    /// [`KEPT`].
    ///
    /// The pass must hold no gather: one left in would be issued for the
    /// position it was hashed at, which is a token behind by the time it is
    /// taken again. So a pass that still holds one is not kept.
    pub(crate) fn put(&mut self, key: PlanKey, pass: Pass) {
        if !pass.gather.is_empty() {
            return;
        }
        self.kept.retain(|(kept, _)| *kept != key);
        if self.kept.len() >= KEPT {
            self.kept.remove(0);
        }
        self.kept.push((key, pass));
    }

    /// Drops every pass kept for `session`.
    pub(crate) fn forget(&mut self, session: SessionId) {
        self.kept.retain(|(kept, _)| kept.session != session);
    }

    /// Drops every kept pass.
    pub(crate) fn clear(&mut self) {
        self.kept.clear();
    }

    /// Passes kept.
    pub(crate) fn len(&self) -> usize {
        self.kept.len()
    }
}

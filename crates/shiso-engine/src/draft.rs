//! The speculative step this stack had none of: a draft head, its own cache,
//! and the acceptance loop that commits what the target agreed with.
//!
//! One decode step through this stack can commit more than one token. A draft
//! head predicts the token *after* the next one, a chain of those gives `k`
//! guesses, the target verifies all `1 + k` in one pass, and the longest prefix
//! whose greedy pick agrees is committed.
//!
//! # Ruling: the decidable half is shared, the pass is not lifted
//!
//! Task 3.3 Step 1 asked whether to lift [`crate::sparse::draft`]'s acceptance
//! loop or to share it. **Shared, in place, and only the decidable half.**
//!
//! [`crate::mtp`] - which is `sparse/mtp.rs`, re-exported by `lib.rs` - is
//! already the part of speculation that is host arithmetic over tokens:
//! bounding `k`, deciding how much of a draft the target agreed with, and
//! naming the record a rejection copies back. Its only import is `shiso_core`;
//! it names no stack, no planner and no buffer. That is the half which is
//! *silently* wrong when it is wrong - a build that accepts one token too many
//! answers fluently from a state the target never computed - so it is exactly
//! the half that must not exist twice. [`crate::sparse::mtp::StepOutcome`],
//! [`crate::sparse::mtp::accepted_prefix`] and
//! [`crate::sparse::mtp::Rollback`] below are that module's, not copies of it.
//!
//! The *pass* half is not lifted, and could not be cheaply. Every line of
//! `sparse/draft.rs`'s `plan_draft` is written in `SparseStack`'s planner
//! vocabulary - `Plan`, `Operand`, `WeightRef`, `LayerRef`, `Buf`, `Island` -
//! and this stack has no planner at all: it has [`crate::layers::forward`], a
//! straight-line launcher. Lifting would mean giving [`HybridStack`] a planner,
//! which is a rewrite of the crate and lands in what Phase 4 owns.
//!
//! **Cost if this ruling is wrong.** Sharing is wrong only if `sparse/mtp.rs`
//! later grows a `SparseStack`-shaped assumption and this lane silently
//! inherits it. The guard is that the module imports nothing but `shiso_core`,
//! so such an assumption cannot be added without adding an import to it -
//! a visible edit in that file, not a behaviour change here. The alternative
//! considered and rejected was moving the file to `crate::mtp`: sixteen call
//! sites across four crates, all in the sparse lane, for a shorter path and no
//! behavioural gain.
//!
//! # The head is a target attention layer with two normalizations in front
//!
//! `mtp.fc` projects the concatenation of a normed token embedding and a normed
//! target hidden state down to the residual width, and what follows is
//! tensor-for-tensor a target attention slot: `q_proj` doubled by this family's
//! folded output gate, `k`/`v` at four heads, `q_norm` and `k_norm` at
//! `head_dim`, and a gated feed-forward. So the block runs through the family's
//! own [`crate::blocks::BlockKernels::attention`] and [`crate::ffn::run`], at
//! the target's own widths, and nothing here re-derives a launch geometry.
//!
//! # Why the head's cache is its own buffers and the target's page table
//!
//! The head's block is an attention block on a stack that is three recurrent
//! slots to one attention slot, so its keys and values have nowhere to live
//! among the target's: every run in that pool belongs to a slot that writes it.
//! It therefore gets a pool of its own, two runs deep.
//!
//! What it does *not* get is a page table of its own. It writes the same
//! positions the target does, so the target session's table maps them to the
//! same page ids, and page id `p` in the head's pool is the head's own storage
//! for whatever position the target keeps at page `p`. Two sessions sharing a
//! target page share a draft page, which is consistent: the head's row at a
//! position is a function of the prefix, and a shared page means a shared
//! prefix.
//!
//! The reason this is safe to be approximate about at all is that **nothing
//! about correctness depends on the head's cache**. A head reading a stale row
//! proposes a worse token, the target rejects it, and the answer is the one
//! plain decode would have given. Only the acceptance rate is at stake.
//!
//! # What a rejection has to put back, and what it does not
//!
//! The cache rows a rejected draft wrote sit past the frontier and are
//! overwritten before anything reads them. The recurrent state is the one thing
//! that cannot be left: a state matrix that walked three tokens too far is what
//! every token after it is then conditioned on, and nothing fails. That is what
//! [`crate::blocks::StateTrail`] records and what
//! [`crate::sparse::mtp::Rollback`] indexes.

use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, alloc};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::Speculated;

use crate::attn::{Attn, AttnCache, AttnGeometry, Positions};
use crate::blocks::AttnIo;
use crate::ffn::Ffn;
use crate::kv::{DeviceKv, KvLayout};
use crate::launch::F32_BYTES;
use crate::layers::{self, Geometry, ResidualIndex, Scratch};
use crate::moe::{Moe, MoeScratch};
use crate::mtp::{DraftSource, SpecDepth, StepOutcome};
use crate::projection::{self, Weight};
use crate::shape::MoeShape;
use crate::stack::{HybridStack, MAX_DECODE_BATCH};

/// Everything a draft head carries that is not the target's.
///
/// Deliberately NOT `#[non_exhaustive]`: the crate that knows a checkpoint
/// family builds one of these, field by field, out of tensors it has already
/// imported. This crate names no family and does no importing.
#[derive(Debug)]
pub struct DraftParts {
    /// `2 * hidden -> hidden`: the pair-to-layer projection.
    pub fc: Weight,
    /// Norm over the token embedding the head starts from, `[hidden]`.
    pub norm_embed: DeviceBuffer,
    /// Norm over the target hidden state it starts from, `[hidden]`.
    pub norm_hidden: DeviceBuffer,
    /// Norm the head's block reads, `[hidden]`.
    pub input_norm: DeviceBuffer,
    /// Norm the head's feed-forward reads, `[hidden]`.
    pub ffn_norm: DeviceBuffer,
    /// Norm in front of the shared vocabulary projection, `[hidden]`.
    pub output_norm: DeviceBuffer,
    /// The head's single attention block.
    pub attn: Attn,
    /// The head's single feed-forward.
    ///
    /// For a routed head this is the SHARED expert, because that is what it is:
    /// an always-on dense feed-forward beside the routed ones. See
    /// [`DraftParts::moe`].
    pub ffn: Ffn,
    /// The head's routed feed-forward, when it has one.
    ///
    /// `None` is a dense head, which is what the `qwen35` sidecar ships and
    /// what this type described until the routed family gained a head of its
    /// own. `Some` carries the block and the widths it is launched over -
    /// paired, because a [`MoeShape`] that disagreed with the block's stacks
    /// would size the scratch for one expert count and stride the slabs by
    /// another.
    ///
    /// The head's routed block is **the target's architecture at the target's
    /// widths**: 256 experts, 8 per token. What differs is the codec - the
    /// extracted `Qwen3.6-35B-A3B-MTP-4bit` head is MLX affine 4-bit group 64
    /// where the target's GGUF is `Q4_K`/`Q5_K` - and the codec is a property
    /// of the slabs, so nothing here has to say so.
    pub moe: Option<(Moe, MoeShape)>,
}

/// A draft head resident on the device, with the cache and scratch it runs in.
#[derive(Debug)]
pub struct DraftHead {
    parts: DraftParts,
    /// The head's own key and value runs, one attention slot deep.
    kv: DeviceKv,
    /// The head's own activation rows.
    ///
    /// Not the target's. The head runs *between* target passes and reads the
    /// residual the last one left, so a head writing into the target's rows
    /// would overwrite the very thing it was handed.
    scratch: Scratch,
    /// The widths the head's block launches at, which are a target attention
    /// slot's.
    geometry: AttnGeometry,
    /// The target residual row the next chain starts from.
    ///
    /// Copied out of the target's stream rather than read in place, because the
    /// verify pass that produced it is the same scratch the next verify pass
    /// overwrites.
    staged: DeviceBuffer,
    /// Which session [`DraftHead::staged`] holds a current row for, if any.
    ///
    /// A session handle rather than a boolean, and that is the whole point. The
    /// flag answers "may the next chain skip staging?", and the honest answer
    /// depends on *whose* chain: the head is one buffer shared by every session
    /// on the stack, so a row staged for A is stale for B even though no pass
    /// ran in between. Two consecutive speculative steps for two sessions is a
    /// reachable schedule - `scheduler` takes the speculative path whenever
    /// exactly one row is runnable, so two sessions whose ticks serialise
    /// alternate single-row speculative steps with nothing between them - and
    /// under a boolean the second would have drafted from the first's hidden
    /// state.
    ///
    /// Cleared outright by every pass that is not a verify
    /// ([`DraftHead::stale`]). A plain decode step leaves a residual of its own,
    /// and a chain that drafted from a row two steps old would propose from a
    /// hidden state the session has moved past - fluently, and at the right
    /// shape.
    staged_for: Option<SessionId>,
    /// The head's OWN residual for the row its last pass wrote.
    ///
    /// Copied out rather than read in place, because a chain step's first
    /// launch is the embedding gather and it writes the very buffer the
    /// previous step left its residual in. Reading it in place would norm the
    /// embedding twice and hand `fc` two copies of the same term - at exactly
    /// the right shape, through a launch that succeeds.
    own: DeviceBuffer,
    /// One `[n_tok, 2 * hidden]` row of the two normed terms, side by side.
    ///
    /// A buffer of its own because `fc` reads them as one row of `2 * hidden`
    /// columns, and the two norms that fill it write half a row each. Writing
    /// them into two separate buffers would mean a concatenating copy per row
    /// before every projection.
    pair: DeviceBuffer,
    /// The chunk's ids and positions, as `int32` rows.
    staging: Vec<u8>,
    /// The routed block's own scratch, when the head has one.
    ///
    /// Not the target's. The head runs between target passes over rows of its
    /// own, and the routed scratch holds the sorted column map and the expert
    /// activations for the pass that is running - so a head sharing the
    /// target's would overwrite the routing of the verify pass it sits between.
    moe_scratch: Option<MoeScratch>,
}

impl DraftHead {
    /// Makes a bound head resident, with a cache and scratch of its own.
    ///
    /// `rows` is the widest pass the head will run, which is the prompt chunk
    /// it is warmed over rather than the chain it drafts: a chain is one row.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the head's projections are not the
    /// target's widths - it is a *second model* over the same residual stream,
    /// and a head built at another width would bind into launches sized by the
    /// target's geometry and succeed at every one - and
    /// [`CoreError::Backend`] when an allocation fails.
    pub fn new(
        backend: &Arc<dyn Backend>,
        parts: DraftParts,
        geometry: &Geometry,
        pool_pages: u32,
        rows: u32,
    ) -> CoreResult<Self> {
        let widths = *geometry.attn.first().ok_or(CoreError::ShapeMismatch {
            field: "attention slots the head's block is shaped like",
            expected: 1,
            got: 0,
        })?;
        let pair = u64::from(geometry.hidden).saturating_mul(2);
        if u64::from(parts.fc.in_dim) != pair || parts.fc.out_dim != geometry.hidden {
            return Err(CoreError::ShapeMismatch {
                field: "draft head fc projection",
                expected: pair,
                got: u64::from(parts.fc.in_dim),
            });
        }
        // Two runs, one key and one value, for the head's single attention
        // slot - sized by the cache's own constructor rather than by the same
        // three multiplications written out again here. They were written out
        // once, with `F32_BYTES` where the cache uses a two-byte element, and
        // the pool came out twice the size it needed with nothing to notice:
        // an over-large buffer is still in bounds.
        let kv = DeviceKv::open(
            backend,
            &KvLayout::for_one_attn_slot(widths.n_head_kv, widths.head_dim),
            pool_pages,
        )?;
        let scratch = Scratch::new(backend, geometry, rows, 1)?;
        // Sized for the widest pass the head runs, exactly as the activation
        // scratch above is: a chain step is one row, and a warm over a prompt
        // chunk is `rows`.
        let moe_scratch = parts
            .moe
            .as_ref()
            .map(|(_, shape)| MoeScratch::new(backend, *shape, geometry.hidden, rows))
            .transpose()?;
        let hidden = u64::from(geometry.hidden).saturating_mul(F32_BYTES);
        let staged = alloc(backend, hidden)?;
        let own = alloc(backend, hidden)?;
        let pair = alloc(
            backend,
            hidden.saturating_mul(2).saturating_mul(u64::from(rows)),
        )?;
        tracing::info!(
            hidden = geometry.hidden,
            heads = widths.n_head,
            pool_pages,
            rows,
            "draft head resident"
        );
        Ok(Self {
            parts,
            kv,
            scratch,
            geometry: widths,
            staged,
            own,
            staged_for: None,
            pair,
            staging: Vec::new(),
            moe_scratch,
        })
    }

    /// Marks the staged row as belonging to a pass the session has moved past.
    ///
    /// Unconditional, and for every session: a pass that is not a verify
    /// overwrites the target scratch the staged row was copied out of, so
    /// there is no session whose row survives it.
    pub(crate) const fn stale(&mut self) {
        self.staged_for = None;
    }
}

impl HybridStack {
    /// Makes `head` this stack's draft head.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when the stack was built without a
    /// speculative depth, which is the setting that sized every session's
    /// rollback records: a head bound to a stack whose sessions keep no trail
    /// would draft, verify, and have nothing to wind back to.
    pub fn bind_draft(&mut self, head: DraftHead) -> CoreResult<()> {
        if !self.opts.spec.is_speculative() {
            return Err(CoreError::Unsupported {
                what: "a draft head on a stack whose sessions were opened with no rollback \
                       records, which is what a speculative depth of zero asks for",
                backend: "engine",
            });
        }
        self.draft = Some(head);
        Ok(())
    }

    /// Whether a draft head is resident, which is what makes a step
    /// speculative.
    #[must_use]
    pub fn draft_resident(&self) -> bool {
        self.draft.is_some()
    }

    /// How deep this stack may draft.
    #[must_use]
    pub const fn spec_depth(&self) -> SpecDepth {
        self.opts.spec
    }

    /// One speculative decode step: draft `k`, verify all `k + 1` in one target
    /// pass, commit the prefix the target agreed with.
    ///
    /// `first` is the token the caller sampled from the previous pass, which
    /// this step commits unconditionally at the session's frontier. `eos` stops
    /// the accepted prefix: a draft may continue past the end of a turn, and
    /// committing what follows would put tokens after the end of the answer.
    ///
    /// # Greedy only
    ///
    /// A guess is accepted by comparing it with the target's *argmax*. That is
    /// exactly right when the caller samples greedily and wrong otherwise, and
    /// [`shiso_core::model::Speculating`] says so on the trait.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] on a stack with no head or a session opened
    /// without rollback records, [`CoreError::UnknownSession`], and whatever
    /// the draft, the verify pass and the rewind report.
    pub fn speculate(
        &mut self,
        session: SessionId,
        first: TokenId,
        eos: TokenId,
    ) -> CoreResult<Speculated> {
        let live = self
            .sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let frontier = live.written();
        let depth = live.spec_depth();
        // The narrowest of three ceilings: what the session may draft, what the
        // context has room for, and what the logits block has rows for. The
        // third is this stack's own and is easy to forget: `write_logits`
        // indexes a block of `MAX_DECODE_BATCH` rows, and a verify pass one row
        // wider would project over the row after the last one.
        let budget = crate::mtp::draft_budget(
            depth,
            frontier,
            self.opts.ctx,
            MAX_DECODE_BATCH.min(crate::mtp::MAX_DRAFT.saturating_add(1)),
        );
        self.stage_hidden(session)?;
        // The chain writes cache rows at positions past the frontier, and the
        // page table the head reads is the target session's. A position the
        // table has not claimed resolves to whatever page id was last in that
        // slot, so the head would write its keys into another session's page
        // and read them back - fluently, at the right shape, from the wrong
        // conversation. Claimed here, once, for every position the chain can
        // reach.
        // The cache by a second handle, so a claim can evict from it while the
        // session it claims for is borrowed.
        let cache = Arc::clone(&self.resident()?.cache);
        let room = crate::kv::Room {
            cache: &cache,
            now: self.now(),
        };
        self.sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .reserve_to(frontier.saturating_add(budget), room)?;
        let clock = std::time::Instant::now();
        let drafts = self.chain(session, first, frontier, budget)?;
        let chained = clock.elapsed();

        // The verify pass: the committed token and every guess behind it, at
        // consecutive positions from the frontier. Row `i` of it is the
        // target's own pick for the position `drafts[i]` proposed.
        let mut rows = Vec::with_capacity(drafts.len().saturating_add(1));
        rows.push(first);
        rows.extend_from_slice(&drafts);
        let verify_rows = u32::try_from(rows.len()).unwrap_or(1);
        let clock = std::time::Instant::now();
        self.verify(session, &rows)?;
        let verified = clock.elapsed();
        let clock = std::time::Instant::now();
        let picks = self.read_picks(verify_rows)?;
        let read = clock.elapsed();
        // What the head guessed against what the target actually wanted, at the
        // same positions. A head that is merely WEAK proposes plausible tokens
        // and misses; a head that is wired wrong proposes tokens with no
        // relation to the target's, and the two look identical in an acceptance
        // rate. This is the one line that tells them apart - it is what showed
        // that the dense head locks onto " the"/" return" whatever the context,
        // which is not what corrupt weights look like.
        //
        // A case that wants to read it has to install a subscriber; the test
        // harness does not do so by default.
        tracing::debug!(
            drafts = ?drafts.iter().map(|t| t.get()).collect::<Vec<_>>(),
            wanted = ?picks.iter().take(drafts.len()).map(|t| t.get()).collect::<Vec<_>>(),
            "the chain's guesses against the target's picks"
        );

        let outcome = StepOutcome::settle(
            first,
            &drafts,
            picks.get(..drafts.len()).unwrap_or(&[]),
            eos,
            (!drafts.is_empty()).then_some(DraftSource::Head),
        )?;
        let accepted = outcome.logits_row;
        // The rewind before the commit: the commit registers the span the
        // session wrote, and a session that committed `1 + k` positions and
        // then wound back to `1 + j` would have handed the cache a span it does
        // not stand on.
        if let Some(rewind) = outcome.rollback {
            let live = self
                .sessions
                .get(&session)
                .ok_or(CoreError::UnknownSession(session))?;
            live.rewind_states(&self.backend, rewind)?;
            let keep = frontier.saturating_add(accepted).saturating_add(1);
            self.sessions
                .get_mut(&session)
                .ok_or(CoreError::UnknownSession(session))?
                .rewind_to(keep);
            tracing::debug!(%session, keep, slot = rewind.slot(), "speculative state rolled back");
        }
        self.commit_step(session)?;
        // The row the next chain starts from is the residual of the last
        // COMMITTED position, which is row `accepted` of the pass that just
        // ran.
        self.stage_row(session, accepted)?;
        let clock = std::time::Instant::now();
        self.reflush(session, frontier, &outcome.committed)?;
        // What a speculative step spends its time on, by phase. A cycle that is
        // slower than the plain step it replaces is either a head that costs
        // too much or a verify pass that amortizes too little, and an
        // acceptance rate cannot tell those apart.
        tracing::debug!(
            chain_ms = chained.as_secs_f64() * 1e3,
            verify_ms = verified.as_secs_f64() * 1e3,
            read_picks_ms = read.as_secs_f64() * 1e3,
            reflush_ms = clock.elapsed().as_secs_f64() * 1e3,
            rows = verify_rows,
            accepted,
            "speculative step"
        );

        // What the caller has not seen: the guesses that were kept, then the
        // target's own pick at the row the step settled on. `first` is left out
        // because it was already the caller's.
        let mut produced = outcome.committed.get(1..).unwrap_or(&[]).to_vec();
        produced.push(
            *picks
                .get(accepted as usize)
                .ok_or(CoreError::ShapeMismatch {
                    field: "verify rows",
                    expected: u64::from(accepted).saturating_add(1),
                    got: u64::try_from(picks.len()).unwrap_or(u64::MAX),
                })?,
        );
        Ok(Speculated {
            produced,
            accepted,
            drafted: u32::try_from(drafts.len()).unwrap_or(u32::MAX),
        })
    }

    /// The head's chain of guesses behind `first`.
    ///
    /// The head at position `p` reads the target's hidden state for `p` and the
    /// token at `p + 1`, and predicts the token after that. So the first guess
    /// is drafted at the last committed position with `first` as its token, and
    /// each one after it moves one position along and reads the head's own
    /// output.
    fn chain(
        &mut self,
        session: SessionId,
        first: TokenId,
        frontier: u32,
        budget: u32,
    ) -> CoreResult<Vec<TokenId>> {
        let mut drafts = Vec::with_capacity(budget as usize);
        let mut token = first;
        let mut at = frontier.saturating_sub(1);
        // The first guess reads the row the last target pass staged; every one
        // after it reads the head's own residual, which its previous pass left
        // in row zero.
        let mut from = Hidden::Staged;
        for _ in 0..budget {
            let pick = self.draft_pass(session, &[token], at, from, true)?;
            let pick = pick.ok_or(CoreError::Unsupported {
                what: "a draft pass that was asked for a token and produced none",
                backend: "engine",
            })?;
            drafts.push(pick);
            token = pick;
            at = at.saturating_add(1);
            from = Hidden::Own;
        }
        Ok(drafts)
    }

    /// Runs the verify pass and projects every row of it.
    ///
    /// One pass over `1 + k` rows rather than `1 + k` passes of one row, which
    /// is the whole arithmetic of speculation: every projection reads its
    /// weight once for all the rows, and a decode step is where its bytes go.
    fn verify(&mut self, session: SessionId, rows: &[TokenId]) -> CoreResult<()> {
        self.run_verify_rows(session, rows)?;
        for row in 0..u32::try_from(rows.len()).unwrap_or(1) {
            self.write_logits_row(row, row)?;
        }
        Ok(())
    }

    /// The greedy pick of every row of the pass that just ran.
    fn read_picks(&mut self, rows: u32) -> CoreResult<Vec<TokenId>> {
        let vocab = self.geometry.vocab;
        let row_bytes = u64::from(vocab).saturating_mul(F32_BYTES);
        let want = row_bytes.saturating_mul(u64::from(rows));
        let resident = self.resident()?;
        let view = resident.scratch.logits().slice(0, want)?;
        let mut bytes = vec![0u8; usize::try_from(want).unwrap_or(0)];
        self.backend.copy_d2h(view, &mut bytes)?;
        Ok(bytes
            .chunks_exact(usize::try_from(row_bytes).unwrap_or(0))
            .map(|row| crate::sampling::argmax(&as_floats(row)))
            .collect())
    }

    /// Copies the target residual the next chain reads out of this session's
    /// last pass.
    ///
    /// Skipped only when the head already holds a row staged **for this
    /// session**, which is the case on every step of a chain but the first
    /// after a prefill or a plain decode.
    ///
    /// **This session's last pass, not the stack's, and this session's staged
    /// row, not the stack's.** Two things had to move for that to be true and
    /// only one of them is the residual pointer. Phase 4 put that pointer on
    /// [`crate::sessions::PassRow`], so the *row* a stage reads is the right
    /// session's; the *decision* to stage stayed a stack-wide latch on the
    /// head, and a latch is enough on its own. [`HybridStack::speculate`] sets
    /// it at the end of every step, and a verify pass does not clear it by
    /// design, so `speculate(A)` followed by `speculate(B)` with no plain or
    /// batched pass in between would have skipped staging and drafted B's
    /// first guess from A's hidden state. That schedule is reachable:
    /// `scheduler` takes the speculative path whenever exactly one row is
    /// runnable, so two sessions whose ticks serialise alternate single-row
    /// speculative steps with nothing between them.
    ///
    /// Keying the latch by [`SessionId`] is what prevents it. The target
    /// verifies every guess, so the symptom would have been an acceptance rate
    /// quietly going to nothing rather than a wrong answer - which is exactly
    /// why it has to be prevented here rather than caught downstream.
    fn stage_hidden(&mut self, session: SessionId) -> CoreResult<()> {
        if self
            .draft
            .as_ref()
            .is_some_and(|head| head.staged_for == Some(session))
        {
            return Ok(());
        }
        let row = self.last_pass(session)?.row();
        self.stage_row(session, row)
    }

    /// Where `session`'s last pass left its residual row.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] for a handle this stack does not hold, and
    /// [`CoreError::Unsupported`] for a session no pass has run - a head asked
    /// to draft from one would otherwise read a row the allocator last wrote.
    fn last_pass(&self, session: SessionId) -> CoreResult<crate::sessions::PassRow> {
        self.sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .last_pass()
            .ok_or(CoreError::Unsupported {
                what: "a draft from a session no forward pass has run, which has no \
                       residual row to start from",
                backend: "engine",
            })
    }

    /// [`HybridStack::stage_hidden`], reachable from `stack_tests.rs`.
    ///
    /// Test-only, and a window rather than a seam: the staging latch is the one
    /// part of the draft head a host-side case can reach at all, because every
    /// other step of a speculative round needs a read-back and the recording
    /// device refuses those by design.
    #[cfg(test)]
    pub(crate) fn stage_hidden_for_test(&mut self, session: SessionId) -> CoreResult<()> {
        self.stage_hidden(session)
    }

    /// Which session the head's staged row belongs to, if any.
    ///
    /// Test-only, for the same reason [`HybridStack::stage_hidden_for_test`]
    /// is: this is the state the latch keeps and nothing else observes it.
    #[cfg(test)]
    pub(crate) fn staged_session_for_test(&self) -> Option<SessionId> {
        self.draft.as_ref().and_then(|head| head.staged_for)
    }

    /// Copies row `row` of this session's last pass into the head's staged row.
    fn stage_row(&mut self, session: SessionId, row: u32) -> CoreResult<()> {
        let hidden = u64::from(self.geometry.hidden).saturating_mul(F32_BYTES);
        if self.draft.is_none() {
            return Ok(());
        }
        let residual = self.last_pass(session)?.residual();
        let Self {
            backend,
            resident,
            draft,
            ..
        } = self;
        let Some(head) = draft.as_mut() else {
            return Ok(());
        };
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let src = resident
            .scratch
            .stream_at(residual)?
            .slice(u64::from(row).saturating_mul(hidden), hidden)?;
        backend.copy_d2d(head.staged.view(), src)?;
        head.staged_for = Some(session);
        Ok(())
    }

    /// Fills the head's own cache over a chunk of prompt the target has just
    /// read.
    ///
    /// The head at position `p` reads the target's hidden state for `p` and the
    /// token at `p + 1`, so a chunk covering `[start, start + n)` can answer
    /// for `[start, start + n - 1)` out of its own rows. Its last position
    /// needs the token after it, which is the next chunk's first; that one row
    /// is left cold, because a cold row per 512-token chunk costs one worse
    /// draft and a boundary pass costs a launch on every chunk of every prompt.
    ///
    /// One pass, not several. The rows this reads are the ones the chunk left
    /// in the target's residual, and only a target pass overwrites them.
    ///
    /// Nothing at all on a stack with no head, which is what makes this safe to
    /// call unconditionally from the prefill loop.
    ///
    /// # Errors
    ///
    /// Whatever the head's pass reports.
    pub(crate) fn warm_draft(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        start: u32,
    ) -> CoreResult<()> {
        let Some(span) = tokens.get(1..).filter(|rest| !rest.is_empty()) else {
            return Ok(());
        };
        if self.draft.is_none() {
            return Ok(());
        }
        let span = span.to_vec();
        self.draft_pass(session, &span, start, Hidden::TargetRows, false)
            .map(|_| ())
    }

    /// Rewrites the head's cache for the rows this step committed, out of the
    /// target's own hidden states.
    ///
    /// The chain wrote those rows already, but it wrote them from the head's
    /// *own* states: the first guess reads the target's staged row and every
    /// guess after it reads what the head itself produced. That is off the
    /// distribution the head was trained against and it compounds, so a chain
    /// left in the cache makes the next chain worse and the one after that
    /// worse again.
    ///
    /// It costs one head pass over the accepted rows, with no vocabulary
    /// projection - which is the widest matmul in the model and the only
    /// expensive part of a draft.
    fn reflush(
        &mut self,
        session: SessionId,
        frontier: u32,
        committed: &[TokenId],
    ) -> CoreResult<()> {
        // Row `i` of the verify pass is position `frontier + i`, and the head
        // at that position reads the token after it - which is
        // `committed[i + 1]`, still inside this step's own rows. The last
        // committed position has no such token yet: it is the one the next
        // chain starts from, out of the row just staged.
        let Some(tokens) = committed.get(1..) else {
            return Ok(());
        };
        if tokens.is_empty() {
            return Ok(());
        }
        self.draft_pass(session, tokens, frontier, Hidden::TargetRows, false)
            .map(|_| ())
    }
}

/// Where one draft pass's hidden term comes from.
///
/// Three sources and not a boolean, because two of them are one row and the
/// third is a run of them - and the failure a boolean allowed was silent: a
/// reflush over a single accepted row took the "one row" branch and read the
/// row the NEXT step starts from, which is the right shape, the right buffer
/// and one position too far along.
#[derive(Debug, Clone, Copy)]
enum Hidden {
    /// The target residual row the last verify staged: a chain's first step.
    Staged,
    /// The head's own residual from its previous pass: a chain step past the
    /// first.
    Own,
    /// The target's residual rows of the pass that just ran, row for row with
    /// this pass's: a warm over a prompt chunk, or a reflush over the rows a
    /// step committed.
    TargetRows,
}

/// Reads a logits row back as floats.
///
/// A copy rather than a cast: the bytes come off the device with no alignment
/// promise, and `bytemuck` would refuse them at exactly the widths that matter.
fn as_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

impl HybridStack {
    /// Runs the head over `tokens` at consecutive positions from `start`, and
    /// returns the token it proposes when one was asked for.
    ///
    /// `from_target` says which hidden state the pass opens from: the staged
    /// target residual, or the head's own from its previous pass. The session
    /// is *not* advanced - a draft is a proposal, and the verify pass is what
    /// commits.
    ///
    /// `pick` is `false` for a pass that is only filling the head's cache.
    /// `None` is what that asks for rather than a refusal: such a pass runs no
    /// vocabulary projection, so there is no token to hand back, and a
    /// placeholder would be a token a caller could draft with.
    #[expect(
        clippy::too_many_lines,
        reason = "the head's pass is one sequence of stages and reads as one: the two \
                  terms it opens with are only meaningful beside each other, and a split \
                  between them would hide that they are added"
    )]
    // `None` is the answer to `pick: false` rather than a refusal, which the
    // doc above says at length. Restated here because `scripts/api_gate.sh`
    // walks up from the signature and stops at the first line that is neither
    // a comment nor a one-line attribute - so the `#[expect]` block between
    // the two hides the doc from it.
    fn draft_pass(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        start: u32,
        from: Hidden,
        pick: bool,
    ) -> CoreResult<Option<TokenId>> {
        let n_tok = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if n_tok == 0 {
            return Err(CoreError::ShapeMismatch {
                field: "draft rows",
                expected: 1,
                got: 0,
            });
        }
        // Everything the pass reads is resolved before a launch, so that a
        // stack with no head refuses here rather than half way down the block.
        let hidden = u64::from(self.geometry.hidden).saturating_mul(F32_BYTES);
        let eps = self.geometry.eps;
        let pages = self
            .sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .kv()
            .view();
        // This session's own last pass, which for a `Hidden::TargetRows` read
        // is the verify pass whose rows the head is being flushed over.
        let residual = self.last_pass(session)?.residual();
        let Self {
            backend,
            geometry,
            kernels,
            resident,
            draft,
            ..
        } = self;
        let head = draft.as_mut().ok_or(CoreError::Unsupported {
            what: "a draft pass on a stack that carries no draft head",
            backend: "engine",
        })?;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let scratch = &head.scratch;

        // The three int32 rows the block reads. The bound is the position row
        // again: the head's attention is causal exactly as the target's is.
        crate::stack::upload_row(
            backend.as_ref(),
            &mut head.staging,
            scratch.ids(),
            tokens.iter().map(|t| i32::try_from(*t).unwrap_or(0)),
        )?;
        for dst in [scratch.positions(), scratch.bound()] {
            crate::stack::upload_row(
                backend.as_ref(),
                &mut head.staging,
                dst,
                (0..n_tok).map(|t| i32::try_from(start.saturating_add(t)).unwrap_or(i32::MAX)),
            )?;
        }

        // The token term, into the head's own stream buffer zero, and the
        // hidden term beside it: one pair row per token, which the projection
        // reads as `2 * hidden` columns.
        let stream = scratch.stream_at(ResidualIndex::FIRST)?;
        layers::embed_rows(
            backend.as_ref(),
            &resident.weights,
            geometry,
            scratch,
            stream,
            n_tok,
        )?;
        let pair = head.pair.view();
        for row in 0..n_tok {
            let at = u64::from(row).saturating_mul(hidden);
            let both = u64::from(row).saturating_mul(hidden).saturating_mul(2);
            layers::rms_norm_mul(
                backend.as_ref(),
                stream.slice(at, hidden)?,
                head.parts.norm_embed.view(),
                pair.slice(both, hidden)?,
                geometry.hidden,
                1,
                eps,
            )?;
            // Where the head's hidden term comes from. A chain step past the
            // first reads the head's OWN residual for the row before it, which
            // its previous pass left in row zero; everything else reads the
            // target's, which is the staged row for a chain's first step and
            // the verify pass's own rows for a reflush or a warm.
            let src = match from {
                Hidden::Staged => head.staged.view(),
                Hidden::Own => head.own.view(),
                Hidden::TargetRows => resident.scratch.stream_at(residual)?.slice(at, hidden)?,
            };
            layers::rms_norm_mul(
                backend.as_ref(),
                src,
                head.parts.norm_hidden.view(),
                pair.slice(both.saturating_add(hidden), hidden)?,
                geometry.hidden,
                1,
                eps,
            )?;
        }
        projection::projection(
            backend.as_ref(),
            &head.parts.fc,
            n_tok,
            pair.slice(0, hidden.saturating_mul(2).saturating_mul(u64::from(n_tok)))?,
            stream,
        )?;

        // The block, through the family's own launches at the target's widths.
        layers::rms_norm_mul(
            backend.as_ref(),
            stream,
            head.parts.input_norm.view(),
            scratch.normed(),
            geometry.hidden,
            n_tok,
            eps,
        )?;
        let (k, v) = head.kv.layer_runs(0)?;
        kernels.attention(
            backend.as_ref(),
            &head.parts.attn,
            &head.geometry,
            scratch.attn(),
            AttnIo {
                input: scratch.normed(),
                pos: Positions::Linear(scratch.positions()),
                bound: scratch.bound(),
                block_out: scratch.block_out(),
            },
            // One table for every row: a draft chain is one session's own
            // consecutive positions, whatever the target's last pass carried.
            &AttnCache {
                k,
                v,
                pages: crate::attn::Pages::shared(pages),
            },
            n_tok,
        )?;
        layers::add_rms_norm_mul(
            backend.as_ref(),
            stream,
            scratch.block_out(),
            head.parts.ffn_norm.view(),
            scratch.stream_at(ResidualIndex::SECOND)?,
            scratch.normed(),
            geometry.hidden,
            n_tok,
            eps,
        )?;
        // The routed half when the head has one, and the dense feed-forward
        // when it does not. `head.parts.ffn` is the shared expert in the first
        // case and the whole feed-forward in the second, which is the same
        // field meaning the same thing the target's `LayerWeights` does.
        match (head.parts.moe.as_ref(), head.moe_scratch.as_ref()) {
            (Some((moe, shape)), Some(moe_scratch)) => crate::moe::run(
                backend.as_ref(),
                moe,
                &head.parts.ffn,
                *shape,
                moe_scratch,
                crate::moe::MoeIo {
                    input: scratch.normed(),
                    block_out: scratch.block_out(),
                },
                n_tok,
            )?,
            // A head that stated a routed block and got no scratch for it is a
            // construction this type cannot produce - both come from `new` -
            // so it is refused rather than run as a dense head, which would
            // draft from a feed-forward that is a quarter of the one the head
            // was trained with.
            (Some(_), None) => {
                return Err(CoreError::Unsupported {
                    what: "a routed draft head with no routed scratch",
                    backend: "engine",
                });
            }
            (None, _) => {
                let [gate, up, mid] = scratch.ffn_rows();
                crate::ffn::run(
                    backend.as_ref(),
                    &head.parts.ffn,
                    geometry.activation,
                    scratch.normed(),
                    gate,
                    up,
                    mid,
                    scratch.block_out(),
                    n_tok,
                )?;
            }
        }
        // Into stream zero, because that is where the next chain step's hidden
        // term comes from and where a `pick: false` pass leaves nothing anybody
        // reads.
        layers::add_rms_norm_mul(
            backend.as_ref(),
            scratch.stream_at(ResidualIndex::SECOND)?,
            scratch.block_out(),
            head.parts.output_norm.view(),
            stream,
            scratch.normed(),
            geometry.hidden,
            n_tok,
            eps,
        )?;
        // The head's own residual for the last row of this pass, for the chain
        // step that reads it next.
        backend.copy_d2d(
            head.own.view(),
            stream.slice(
                u64::from(n_tok.saturating_sub(1)).saturating_mul(hidden),
                hidden,
            )?,
        )?;
        if !pick {
            return Ok(None);
        }
        // Through the target's own vocabulary projection: the head shares it
        // exactly as it shares the embedding table, and a draft is one token -
        // the last row of the pass, for the reason a chunk's logits are its
        // last row.
        layers::logits_row(
            backend.as_ref(),
            &resident.weights,
            geometry,
            scratch,
            n_tok.saturating_sub(1),
            0,
        )?;
        let row_bytes = u64::from(geometry.vocab).saturating_mul(F32_BYTES);
        let mut bytes = vec![0u8; usize::try_from(row_bytes).unwrap_or(0)];
        backend.copy_d2h(scratch.logits().slice(0, row_bytes)?, &mut bytes)?;
        Ok(Some(crate::sampling::argmax(&as_floats(&bytes))))
    }
}

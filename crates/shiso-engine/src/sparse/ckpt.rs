//! The sparse stack's half of the checkpoint tier: what its blob is made of,
//! when a span is committed, and how a stored prefix is stood back up.
//!
//! [`crate::ckpt`] owns the tier itself - the policy, the prefix scan, the
//! refusal vocabulary and the [`Checkpointed`](crate::ckpt::Checkpointed)
//! contract - and none of that is restated here. What is here is the two things
//! the sparse stack answers differently from the dense one, and they are both
//! about the *content* of a checkpoint rather than about when one is taken.
//!
//! # A sparse resume is not key/value rows alone
//!
//! The dense family's blob is the recurrent stack and nothing else: everything
//! else a resumed session needs is either in the cache pages or is recomputed by
//! the next step. This family carries two more things, and neither can be
//! reconstructed:
//!
//! - the n-gram block's dilated convolution window, which the block *reads*
//!   before it writes and which therefore has to arrive holding the positions
//!   the original run left in it;
//! - the residual row a speculative step staged for the draft head. Without it a
//!   resumed session cannot draft on its first step, and the first step is the
//!   one a dropped row changes: what the head would have proposed there was
//!   computed from a residual nothing else records.
//!
//! The frontier logits row is committed beside the blob rather than inside it,
//! which is the same arrangement the dense tier uses and the reason
//! [`crate::sparse::paged::ckpt_bytes`] excludes it.
//!
//! # The layout guard says which of those a file carries
//!
//! [`shiso_kvstore::PathLayout`] already has the three words this family needs:
//! `ple_window_elements` and `hc_width`, both zero in a dense file, and
//! `draft_head`, which the store reads as "the state section ends with one row
//! of `hc_width`". A build that grew one of those planes therefore cannot read a
//! file written before it had one, which is the failure the guard exists for -
//! loading another geometry's rows answers fluently rather than failing.
//!
//! The pending row's *position* is not a guard word and does not need to be. A
//! checkpoint is only ever taken at a session's frontier, and the row staged
//! there is always the last committed one, so the position is `tokens - 1` and a
//! word carrying it could only ever disagree.

use std::sync::Arc;
use std::time::Duration;

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::media::{KEY_ID_FLOOR, MediaKey, MediaSpan};
use shiso_core::model::Logits;
use shiso_kvstore::{
    CacheStats, CommitOpts, KvCache, PathLayout, PersistRefusal, PumpReport, Reason,
};

use crate::ckpt::{Checkpoint, CkptConfig, ResumeInfo, SaveOutcome, source_of};
use crate::ckpt_layout::CkptIdentity;
use crate::shape::LayerKind;
use crate::sparse::kernels::F32_BYTES;
use crate::sparse::mtp::SpecDepth;
use crate::sparse::paged::{self, PagePlanes, conv_state_bytes, ple_window_bytes, state_bytes};
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::stack::{MAX_PREFILL_CHUNK, SessionKey, SparseSession, SparseStack};

/// Bytes per stored key/value element: this family's cache is half precision end
/// to end, exactly as the dense one's is.
const CACHE_ELEMENT_BYTES: u32 = 2;

/// The geometry guard a stored payload carries, derived from the running build.
///
/// Every number is either a count from the shape or a width the cache was built
/// with, so a file this build writes and a file this build will accept are
/// described by one function.
#[must_use]
pub(crate) fn path_layout(
    shape: &SparseShape,
    ctx: u32,
    spec: SpecDepth,
    identity: CkptIdentity,
) -> PathLayout {
    PathLayout {
        model_id: identity.model_id,
        quant_bits: identity.quant_bits,
        ctx_size: ctx,
        kv_width: shape.kv_dim().saturating_mul(CACHE_ELEMENT_BYTES),
        n_layers: shape.n_layers(),
        n_attn_layers: shape.count(LayerKind::Attn),
        n_gdn_layers: shape.count(LayerKind::Gdn),
        vocab: shape.vocab,
        gdn_conv_elements: elements(conv_state_bytes(shape)),
        gdn_state_elements: elements(state_bytes(shape)),
        // Both are zero in the dense family's layout, which is what keeps the
        // two families' files from reading each other even before the model tag
        // is compared.
        ple_window_elements: elements(ple_window_bytes(shape)),
        hc_width: shape.stream_dim(),
        indexer: true,
        // Not "this family has a draft head" but "this payload carries its
        // staged row". The store derives the state section's length from this
        // word, so a build that speculates and one that does not write files of
        // different lengths and each refuses to read the other's - which is the
        // right outcome: a row restored onto a session that has nowhere to keep
        // it, or a row a speculating session invented, are both answers from
        // numbers nothing computed.
        draft_head: spec.is_speculative(),
    }
}

/// How many single-precision elements a byte count is, narrowed to the width a
/// guard word carries.
///
/// Saturating rather than checked: the value is a guard, and a geometry wide
/// enough to overflow it would have failed to allocate its pool long before
/// anything asked for a checkpoint.
fn elements(bytes: u64) -> u32 {
    u32::try_from(bytes / F32_BYTES).unwrap_or(u32::MAX)
}

/// What kind of boundary a committed span ends at, which decides whether it
/// is worth a checkpoint and what the disk tier files it under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Boundary {
    /// A decode step or a chunk end the stack happened to stop at.
    Step,
    /// The end of a prompt: where the same prompt, or the next turn of its
    /// conversation, resumes from.
    PromptEnd,
    /// A message boundary or the point a prompt branched off a path the cache
    /// held: shared by many sessions, so worth a checkpoint of its own.
    Shared,
}

impl Boundary {
    /// Whether a rule earned this boundary, rather than the stack stopping
    /// there by chance.
    const fn earned(self) -> bool {
        !matches!(self, Self::Step)
    }
}

impl SparseStack {
    /// Whether a checkpoint directory has been opened.
    #[must_use]
    pub const fn ckpt_enabled(&self) -> bool {
        self.ckpt.is_some()
    }

    /// What the cache underneath this stack currently holds.
    #[must_use]
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.resident
            .as_ref()
            .map(|resident| resident.cache.stats())
    }

    /// The cache underneath this stack.
    #[must_use]
    pub fn cache(&self) -> Option<Arc<KvCache>> {
        self.resident
            .as_ref()
            .map(|resident| Arc::clone(&resident.cache))
    }

    /// Opens the checkpoint tier over `config.dir`, tagged with `identity`.
    ///
    /// The tier is part of the cache rather than a layer above it, so turning it
    /// on rebuilds the cache with a disk directory and the layout its keys name.
    /// That is why it is refused while sessions are open: a session holds page
    /// references into the pool the old cache owns, and the new one has its own.
    ///
    /// `identity` is not a caller's choice. It comes from the crate that loaded
    /// the checkpoint, which is the only party that knows which family wrote the
    /// weights and how they were quantized.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when a session is open or the stack has been
    /// torn down, [`CoreError::InvalidValue`] for a zero budget, and
    /// [`CoreError::Backend`] when the directory cannot be opened.
    pub fn ckpt_enable(&mut self, config: &CkptConfig, identity: CkptIdentity) -> CoreResult<()> {
        if !self.sessions.is_empty() {
            return Err(CoreError::Unsupported {
                what: "opening a checkpoint directory while sessions are already running",
                backend: "engine",
            });
        }
        let layout = path_layout(&self.shape, self.opts.engine.ctx, self.opts.spec, identity);
        // The planes the pool was allocated with, which include a draft head's
        // own four when one is resident: a geometry that left them out handed
        // the store a page buffer one draft layer short of the page, and every
        // store from a build with a head failed on its first page.
        let planes = PagePlanes {
            indexer: true,
            draft: self.has_draft_head(),
        };
        let geometry =
            paged::cache_geometry(&self.shape, planes, self.opts.spec).with_layout(layout);
        let (pool_pages, ckpt_slots) = (
            self.opts.engine.kv_pool_pages,
            self.opts.engine.kv_ckpt_slots,
        );
        let disk = config.options.disk(
            config.dir.clone(),
            geometry.page_bytes_total(pool_pages),
            self.backend.numerics_id(),
        )?;
        let budget_bytes = disk.budget_bytes;
        let numerics = disk.numerics.clone();
        let resident = self.resident.as_mut().ok_or(CoreError::Unsupported {
            what: "a stack whose device state teardown already released",
            backend: "engine",
        })?;
        let cache = KvCache::open(
            &geometry,
            Box::new(resident.kv.clone()),
            pool_pages,
            ckpt_slots,
            Some(disk),
        )
        .map_err(cache_failure)?;
        resident.cache = Arc::new(cache);
        // Sized as one blob and tagged as the recurrent state it holds a copy
        // of. The pool was sized before any tier opened, so this comes out of
        // what the pool left, never out of the pool.
        resident.snapshot = Some(shiso_core::buffer::alloc_tagged(
            &self.backend,
            shiso_core::ledger::MemTag::RecurrentState,
            geometry.ckpt_bytes,
        )?);
        crate::sparse::memory::watch_cache(
            &self.backend,
            &resident.cache,
            resident.kv.page_bytes(),
            geometry.ckpt_bytes,
        );
        self.ckpt = Some(Checkpoint {
            options: config.options,
        });
        tracing::info!(
            dir = %config.dir.display(),
            min_tokens = config.options.min_tokens,
            budget_bytes,
            model_id = identity.model_id,
            quant_bits = identity.quant_bits,
            %numerics,
            "sparse checkpoint tier open"
        );
        Ok(())
    }

    /// Registers the span a session just wrote, and admits a checkpoint when the
    /// span ends at a boundary worth one.
    ///
    /// The snapshot is asked for before it is taken, for the reason the dense
    /// stack gives: a blob is a multi-megabyte read back off the device, so
    /// paying it on every committed span would cost more than the prefill it
    /// saves. So a snapshot is only taken at a boundary worth one on its own
    /// terms - a prompt's end, a message boundary, a branch point, or a stride
    /// multiple - which are the positions a later request can resume from.
    ///
    /// `boundary` is the caller's to answer and not this function's to infer,
    /// because a prompt's last chunk and a decode step look identical from here:
    /// both write one logits row, so a checkpoint earned by *having* a row is a
    /// checkpoint earned by every token - the whole recurrent stack read back to
    /// the host per token, which is most of a decode step.
    ///
    /// `frontier_row` names which row of the logits block this session's last
    /// position produced. A checkpoint admitted without one cannot be resumed
    /// from at the very end of a prompt without re-running a step.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], whatever the device reports for the
    /// snapshot reads, and whatever the cache reports for a span it could not
    /// register.
    pub(crate) fn commit(
        &mut self,
        session: SessionId,
        boundary: Boundary,
        frontier_row: Option<u32>,
    ) -> CoreResult<()> {
        let now = self.now();
        let vocab_row = u64::from(self.shape.vocab).saturating_mul(F32_BYTES);
        let blob_bytes = paged::ckpt_bytes(&self.shape, self.opts.spec);
        let Self {
            backend,
            resident,
            sessions,
            staging,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a stack whose device state teardown already released",
            backend: "engine",
        })?;
        let live = sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let q = live.written();
        let earned = boundary.earned();
        let periodic = q > 0 && q.is_multiple_of(MAX_PREFILL_CHUNK);
        // A scoring session's prompt end is answered and dropped, and its
        // stride points are what the shared boundary already covers better:
        // each would be a full recurrent read-back for a checkpoint nothing
        // resumes. Only a shared boundary - where its sibling questions
        // diverge - is worth one.
        let scoring = live.scoring;
        let wanted = !scoring || boundary == Boundary::Shared;
        // A speculating session's blob carries its staged draft row, and a
        // resume marks that row usable because the blob carried it. So a session
        // that has not staged one is not checkpointed at all rather than
        // checkpointed with zeros that a later run would draft from.
        let staged = live.spec_depth().is_speculative() == live.pending_draft().is_some();
        // A prompt's end and a shared boundary are admitted without asking the
        // cache where this path branches: the cache answers with the earliest
        // stop, and every rendered prompt shares at least its template's opening
        // with one before it, so the branch lay behind every prompt's end but the
        // first one a process read. A stride position still has to be the
        // cache's own stop. A session that read media is checkpointed only
        // under a content key - its placeholder runs cannot tell two pictures
        // apart - and never inside a run, where no later pass could continue.
        let admit = (earned || periodic)
            && wanted
            && staged
            && keyed_at(live, q)
            && (earned
                || resident.cache.next_admission(
                    live.pages.frontier(),
                    live.cache_path(),
                    &[],
                    MAX_PREFILL_CHUNK,
                ) == Some(q));
        if admit {
            snapshot(backend.as_ref(), live, blob_bytes, &mut staging.ckpt)?;
            if let Some(row) = frontier_row {
                let at = u64::from(row).saturating_mul(vocab_row);
                let view = resident.scratch.view(Buf::Logits)?.slice(at, vocab_row)?;
                staging
                    .logits
                    .resize(usize::try_from(view.len()).unwrap_or(0), 0);
                backend.copy_d2h(view, &mut staging.logits)?;
            }
        }
        let reason = match boundary {
            Boundary::Shared => Reason::AgentSystem,
            _ if live.resumed_at == 0 => Reason::Cold,
            _ => Reason::Continued,
        };
        let mut opts = CommitOpts::span(&[]);
        if admit {
            opts = opts.checkpoint(staging.ckpt.as_slice(), earned);
            if frontier_row.is_some() {
                opts = opts.logits_row(staging.logits.as_slice());
            }
            // A scoring session's shared boundary serves the questions of the
            // request that is reading it now, from memory; the disk tier
            // would only be paying for a prefix no later process asks for.
            if earned && !scoring {
                // Written behind: this queues the checkpoint and pins the pages
                // the disk does not have yet, and the pump copies them later.
                opts = opts.persist(reason);
            }
        }
        // Split borrows rather than a copy of the path: this runs once per
        // committed span, a decode step is one span, and a session's history is
        // as long as the conversation - so copying it here would make every
        // token cost a pass over the whole turn.
        let (frontier, history) = live.commit_parts();
        resident
            .cache
            .commit(frontier, history, q, opts, now)
            .map_err(cache_failure)
    }

    /// Commits the span a prompt's final pass wrote up to its snapshot row -
    /// `lead` is those rows' tokens - with the state the pass left in the
    /// snapshot buffer as its checkpoint, filed and persisted as the prompt's
    /// end would have been. `false` when there was nothing to commit it with.
    ///
    /// A speculating session's blob carries the draft row of the position
    /// before the snapshot, which the pass's residual still holds.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and whatever the device or the cache
    /// report.
    pub(crate) fn commit_snapshot(
        &mut self,
        session: SessionId,
        lead: &[TokenId],
    ) -> CoreResult<bool> {
        let Some(row) = u64::try_from(lead.len()).ok().filter(|row| *row > 0) else {
            return Ok(false);
        };
        let now = self.now();
        let blob_bytes = paged::ckpt_bytes(&self.shape, self.opts.spec);
        let draft_at = paged::snapshot_draft_offset(&self.shape);
        let width = paged::draft_row_bytes(&self.shape);
        let Self {
            backend,
            resident,
            sessions,
            staging,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a stack whose device state teardown already released",
            backend: "engine",
        })?;
        let Some(snapshot) = resident.snapshot.as_ref() else {
            return Ok(false);
        };
        let live = sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        if live.spec_depth().is_speculative() {
            let residual = resident
                .scratch
                .view(Buf::Residual)?
                .slice((row - 1).saturating_mul(width), width)?;
            backend.copy_d2d(snapshot.view().slice(draft_at, width)?, residual)?;
        }
        staging
            .ckpt
            .resize(usize::try_from(blob_bytes).unwrap_or(usize::MAX), 0);
        backend.copy_d2h(snapshot.view().slice(0, blob_bytes)?, &mut staging.ckpt)?;
        let q = live
            .written()
            .saturating_add(u32::try_from(row).unwrap_or(u32::MAX));
        if !keyed_at(live, q) {
            return Ok(false);
        }
        let path = live.cache_path_with(lead);
        let reason = if live.resumed_at == 0 {
            Reason::Cold
        } else {
            Reason::Continued
        };
        let opts = CommitOpts::span(&[])
            .checkpoint(staging.ckpt.as_slice(), true)
            .persist(reason);
        let (frontier, _) = live.commit_parts();
        resident
            .cache
            .commit(frontier, &path, q, opts, now)
            .map_err(cache_failure)?;
        Ok(true)
    }

    /// Stores the checkpoint at the session's frontier, taking one first when
    /// the frontier has none. See [`crate::ckpt::Checkpointed::kv_save`].
    ///
    /// The end of a turn is the case that takes one: the frontier moved one
    /// decoded token at a time and none of those steps took a snapshot. The
    /// last sampled token has not been run, so the checkpoint is at the last
    /// position the session *wrote* - the next turn's prompt supplies that token
    /// again - and a speculating session is only snapshotted when its staged
    /// draft row belongs to that position.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and whatever the device or the cache report.
    pub fn ckpt_save(&mut self, session: SessionId, reason: Reason) -> CoreResult<SaveOutcome> {
        let Some(Checkpoint { options }) = self.ckpt else {
            return Ok(SaveOutcome::Disabled);
        };
        let now = self.now();
        let blob_bytes = paged::ckpt_bytes(&self.shape, self.opts.spec);
        let Self {
            backend,
            resident,
            sessions,
            staging,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a stack whose device state teardown already released",
            backend: "engine",
        })?;
        let live = sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let q = live.written();
        if !keyed_at(live, q) {
            return Ok(SaveOutcome::CarriesMedia);
        }
        // The floor is checked before the snapshot: below it the store would
        // refuse the entry, and the snapshot is the expensive half.
        let min = options.min_tokens;
        if q < min {
            return Ok(SaveOutcome::Refused(PersistRefusal::BelowMinTokens {
                tokens: q,
                min,
            }));
        }
        let standing = resident
            .cache
            .frontier_ckpt(live.pages.frontier())
            .is_some_and(|ckpt| ckpt.pos == q);
        if !standing {
            if live.spec_depth().is_speculative() && !live.staged_row_current() {
                return Ok(SaveOutcome::NoDraftRow);
            }
            snapshot(backend.as_ref(), live, blob_bytes, &mut staging.ckpt)?;
            let (frontier, history) = live.commit_parts();
            resident
                .cache
                .commit(
                    frontier,
                    history,
                    q,
                    CommitOpts::span(&[]).checkpoint(staging.ckpt.as_slice(), true),
                    now,
                )
                .map_err(cache_failure)?;
        }
        let outcome = resident
            .cache
            .persist(live.pages.frontier(), reason)
            .map_err(cache_failure)?;
        let outcome = SaveOutcome::from_persist(outcome);
        tracing::debug!(%session, tokens = q, ?reason, ?outcome, "a session's frontier was offered to disk");
        Ok(outcome)
    }

    /// Stores every open session. See
    /// [`crate::ckpt::Checkpointed::kv_shutdown_sweep`].
    pub fn ckpt_shutdown_sweep(&mut self) -> usize {
        if self.ckpt.is_none() {
            return 0;
        }
        let mut stored = 0;
        for session in self.order.clone() {
            match self.ckpt_save(session, Reason::Shutdown) {
                Ok(outcome) if outcome.stored().is_some() => stored += 1,
                Ok(outcome) => {
                    tracing::debug!(%session, ?outcome, "session not stored on the way down");
                }
                Err(error) => {
                    // Logged, never propagated: a session that cannot be stored
                    // costs its next request a prefill, and a sweep that stopped
                    // here would cost every session behind it the same.
                    tracing::warn!(%session, %error, "could not store a session on the way down");
                }
            }
        }
        tracing::info!(stored, "sparse shutdown sweep complete");
        stored
    }

    /// Records the message boundaries of the prompt `session` is about to read.
    /// See [`crate::ckpt::Checkpointed::kv_set_hints`].
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`].
    pub fn ckpt_set_hints(&mut self, session: SessionId, hints: &[u32]) -> CoreResult<()> {
        let live = self
            .sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        live.hints.clear();
        live.hints.extend_from_slice(hints);
        live.hints.sort_unstable();
        live.hints.dedup();
        Ok(())
    }

    /// Marks `session` as reading a prompt only to score it. See
    /// [`crate::ckpt::Checkpointed::kv_set_scoring`].
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`].
    pub fn ckpt_set_scoring(&mut self, session: SessionId) -> CoreResult<()> {
        self.sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .scoring = true;
        Ok(())
    }

    /// Files `session`'s path under a media prompt's content key. See
    /// [`crate::ckpt::Checkpointed::kv_set_media_key`].
    ///
    /// Set before anything is read: the key names the prompt's positions from
    /// zero, and a session that already wrote some has filed them under its
    /// bare ids.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::Unsupported`] for a session
    /// that has already written, and [`CoreError::InvalidValue`] for a key that
    /// is not `prompt` with only its spans rewritten.
    pub fn ckpt_set_media_key(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
        key: MediaKey<'_>,
    ) -> CoreResult<()> {
        let live = self
            .sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        if live.written() != 0 {
            return Err(CoreError::Unsupported {
                what: "a media key on a session that has already written",
                backend: "engine",
            });
        }
        let path = key.path();
        let differs = path.len() != prompt.len()
            || path
                .iter()
                .zip(prompt)
                .any(|(keyed, own)| keyed != own && keyed.get() < KEY_ID_FLOOR);
        if differs {
            return Err(CoreError::InvalidValue {
                key: "media cache key",
                value: format!(
                    "a {}-token key for a {}-token prompt that differs outside its spans",
                    path.len(),
                    prompt.len()
                ),
            });
        }
        // Marked media now rather than when a chunk splices: a resume can cover
        // every picture, and the tail it leaves is then read as text - but the
        // path still carries placeholder runs, and dropping the key must never
        // leave it checkpointable under them.
        live.media = true;
        live.key = Some(SessionKey {
            prompt: path.to_vec(),
            spans: key.spans().to_vec(),
            path: Vec::with_capacity(path.len()),
        });
        Ok(())
    }

    /// Copies up to `max_bytes` of queued checkpoint pages off the device.
    ///
    /// # Errors
    ///
    /// Whatever the device reports for a copy.
    pub fn ckpt_pump(&self, max_bytes: u64) -> CoreResult<PumpReport> {
        match self.resident.as_ref() {
            Some(resident) => resident.cache.pump(max_bytes).map_err(cache_failure),
            None => Ok(PumpReport::default()),
        }
    }

    /// Copies everything queued and waits up to `timeout` for it to land.
    #[must_use]
    pub fn ckpt_drain(&self, timeout: Duration) -> bool {
        self.resident
            .as_ref()
            .is_none_or(|resident| resident.cache.drain(timeout))
    }

    /// The path a resume of `prompt` looks up, and the pictures on it.
    ///
    /// A keyed session matches under its key and reads under its own ids; the
    /// two are the same length and differ only inside the spans, which
    /// [`Self::ckpt_set_media_key`] checked.
    fn resume_path(
        &self,
        session: SessionId,
        prompt: &[TokenId],
    ) -> CoreResult<(Vec<TokenId>, Vec<MediaSpan>)> {
        match &self.session(session)?.key {
            Some(key) if key.prompt.len() == prompt.len() => {
                Ok((key.prompt.clone(), key.spans.clone()))
            }
            Some(_) => Err(CoreError::InvalidValue {
                key: "media cache key",
                value: "a resume of a prompt other than the one the key was set for".into(),
            }),
            None => Ok((prompt.to_vec(), Vec::new())),
        }
    }

    /// Stands `session` on the deepest usable checkpoint of `prompt`. See
    /// [`crate::ckpt::Checkpointed::kv_resume`].
    ///
    /// `Ok(None)` is a cold start and not a refusal: no directory is open, or
    /// nothing stored matched. The caller prefills, exactly as it would have
    /// with no checkpoint tier at all.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and [`CoreError::Backend`] when the device
    /// or the cache fails while the prefix is being installed.
    pub fn ckpt_resume(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
    ) -> CoreResult<Option<ResumeInfo<'_>>> {
        if self.ckpt.is_none() {
            return Ok(None);
        }
        if !self.sessions.contains_key(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        let now = self.now();
        let capacity = self.opts.engine.ctx;
        let vocab = self.shape.vocab;
        let cache = Arc::clone(&self.resident()?.cache);
        let (lookup, spans) = self.resume_path(session, prompt)?;
        let Some(resume) = cache.resume(&lookup, now).map_err(cache_failure)? else {
            return Ok(None);
        };
        let Some(reference) = resume.ckpt else {
            return Ok(None);
        };
        let reused = resume.ckpt_pos.min(capacity);
        // The rotary stream a resumed session continues from is a fact about
        // the pictures behind it, and the key pins those: the same key path is
        // the same pictures, so the shift is recomputed here rather than
        // stored. A checkpoint inside a run is never admitted; one found there
        // anyway is dropped - which returns its pages - rather than continued
        // halfway through an item.
        let Some(rope_shift) =
            crate::vision::rotary_shift_at(&spans, reused, crate::sparse::media::MEDIA_POSITIONS)
        else {
            tracing::warn!(
                reused,
                "a stored media prefix ends inside a picture; prefilling instead"
            );
            return Ok(None);
        };
        let whole = u32::try_from(prompt.len()).unwrap_or(u32::MAX);
        let want_row = u64::from(vocab).saturating_mul(F32_BYTES);
        let row = if reused == whole {
            let mut row = vec![0u8; usize::try_from(want_row).unwrap_or(usize::MAX)];
            cache
                .ckpt_logits_into(reference, &mut row)
                .map_err(cache_failure)?;
            Some(row)
        } else {
            None
        };
        let mut blob =
            vec![0u8; usize::try_from(cache.geometry().ckpt_bytes).unwrap_or(usize::MAX)];
        cache
            .ckpt_state_into(reference, &mut blob)
            .map_err(cache_failure)?;
        let tokens = prompt[..usize::try_from(reused).unwrap_or(prompt.len())].to_vec();
        let source = source_of(resume.source);
        {
            // Split borrows: the frontier and the per-session state belong to the
            // session, and the bytes move through the backend.
            let Self {
                backend, sessions, ..
            } = self;
            let live = sessions
                .get_mut(&session)
                .ok_or(CoreError::UnknownSession(session))?;
            live.pages.adopt(resume.frontier)?;
            live.history.clear();
            live.history.extend_from_slice(&tokens);
            if let Some(key) = live.key.as_mut() {
                key.path.clear();
                key.path.extend_from_slice(&lookup[..tokens.len()]);
            }
            live.rope_shift = rope_shift;
            live.written = reused;
            live.resumed_at = reused;
            restore_state(backend.as_ref(), live, &blob)?;
            if let Some(spec) = live.spec.as_mut() {
                spec.staged_for = reused;
            }
        }
        // The frontier row goes exactly where a freshly computed one would be, so
        // a caller cannot tell a resumed row from a prefilled one.
        let logits = match row {
            Some(row) => {
                let view = self
                    .resident()?
                    .scratch
                    .view(Buf::Logits)?
                    .slice(0, want_row)?;
                self.backend.copy_h2d(view, &row)?;
                let view = self
                    .resident()?
                    .scratch
                    .view(Buf::Logits)?
                    .slice(0, want_row)?;
                Some(Logits::new(view, 1, vocab)?)
            }
            None => None,
        };
        tracing::info!(
            tokens = reused,
            ?source,
            whole,
            "resumed a stored sparse prefix"
        );
        Ok(Some(ResumeInfo::new(reused, tokens, logits, source)))
    }
}

/// Whether `live` may be checkpointed at `q`: a text session anywhere, and a
/// media session only under a content key and never inside one of its runs.
fn keyed_at(live: &SparseSession, q: u32) -> bool {
    match &live.key {
        None => !live.media,
        Some(key) => !key.spans.iter().any(|span| span.splits(q)),
    }
}

/// Copies the session's whole restorable state into `blob`.
///
/// The layout is the one [`restore_state`] reads back, and the two agree because
/// this is the only function that writes it: every recurrent slot in ordinal
/// order, each slot's convolution window followed by its state matrix; then the
/// n-gram block's dilated window; then the pending draft row.
///
/// The draft row is present exactly when the stack speculates, which is the same
/// condition the stored layout's draft-head word carries, so the blob's length
/// is always the one the store derives from that word. A speculating session
/// with no row staged is never admitted in the first place - see
/// [`SparseStack::commit`].
fn snapshot(
    backend: &dyn shiso_core::backend::Backend,
    session: &SparseSession,
    blob_bytes: u64,
    blob: &mut Vec<u8>,
) -> CoreResult<()> {
    blob.clear();
    for slot in &session.states {
        append(backend, blob, slot.conv.view())?;
        append(backend, blob, slot.state.view())?;
    }
    if let Some(window) = session.ngram_window() {
        append(backend, blob, window)?;
    }
    if let Some(row) = session.pending_draft() {
        append(backend, blob, row)?;
    }
    let got = u64::try_from(blob.len()).unwrap_or(u64::MAX);
    if got != blob_bytes {
        return Err(CoreError::ShapeMismatch {
            field: "checkpoint blob",
            expected: blob_bytes,
            got,
        });
    }
    Ok(())
}

/// Reads one device range onto the end of `blob`.
fn append(
    backend: &dyn shiso_core::backend::Backend,
    blob: &mut Vec<u8>,
    view: shiso_core::buffer::DeviceView<'_>,
) -> CoreResult<()> {
    let start = blob.len();
    let len = usize::try_from(view.len()).unwrap_or(0);
    blob.resize(start.saturating_add(len), 0);
    backend.copy_d2h(view, &mut blob[start..])
}

/// Copies a checkpoint blob back into a session, in the order [`snapshot`] wrote
/// it.
///
/// Nothing is interpreted: each piece is a byte range whose length the device
/// buffer itself states, and a blob that does not end exactly where the last
/// buffer does is refused rather than partially applied.
///
/// The pending draft row is read exactly when this session can hold one, and
/// that is not a guess about the file: a stack that speculates and one that does
/// not write different `draft_head` words, the store derives the state section's
/// length from that word, and a file written by the other kind is refused before
/// a byte of it reaches here. So "this session speculates" and "this blob ends
/// with a draft row" are the same fact, and the length check below is what says
/// so out loud.
fn restore_state(
    backend: &dyn shiso_core::backend::Backend,
    session: &mut SparseSession,
    blob: &[u8],
) -> CoreResult<()> {
    let mut at = 0usize;
    let views: Vec<_> = session
        .states
        .iter()
        .flat_map(|slot| [slot.conv.view(), slot.state.view()])
        .chain(
            session
                .ngram_window
                .as_ref()
                .map(shiso_core::buffer::DeviceBuffer::view),
        )
        .collect();
    for view in views {
        at = write_view(backend, view, blob, at)?;
    }
    if let Some(spec) = session.spec.as_mut() {
        let row = usize::try_from(spec.pending.len()).unwrap_or(usize::MAX);
        let end = at.saturating_add(row);
        let staged = blob.get(at..end).ok_or(CoreError::ShapeMismatch {
            field: "restored draft row",
            expected: u64::try_from(end).unwrap_or(u64::MAX),
            got: u64::try_from(blob.len()).unwrap_or(u64::MAX),
        })?;
        backend.copy_h2d(spec.pending.view(), staged)?;
        // Usable because the blob carried it: a file written by a build that
        // does not speculate is shorter, and the store refuses it before a byte
        // of it reaches here.
        spec.pending_valid = true;
        at = end;
    }
    let end = at;
    if end != blob.len() {
        return Err(CoreError::ShapeMismatch {
            field: "restored session state",
            expected: u64::try_from(blob.len()).unwrap_or(u64::MAX),
            got: u64::try_from(end).unwrap_or(u64::MAX),
        });
    }
    Ok(())
}

/// Writes the range of `blob` starting at `at` that fills `view`, and reports
/// where the next one starts.
fn write_view(
    backend: &dyn shiso_core::backend::Backend,
    view: shiso_core::buffer::DeviceView<'_>,
    blob: &[u8],
    at: usize,
) -> CoreResult<usize> {
    let len = usize::try_from(view.len()).unwrap_or(usize::MAX);
    let end = at.saturating_add(len);
    let bytes = blob.get(at..end).ok_or(CoreError::ShapeMismatch {
        field: "restored session state",
        expected: u64::try_from(end).unwrap_or(u64::MAX),
        got: u64::try_from(blob.len()).unwrap_or(u64::MAX),
    })?;
    backend.copy_h2d(view, bytes)?;
    Ok(end)
}

/// Presents a cache failure in the core vocabulary.
fn cache_failure(source: shiso_kvstore::KvError) -> CoreError {
    CoreError::Backend {
        backend: "engine",
        op: "kv cache",
        source: Box::new(source),
    }
}

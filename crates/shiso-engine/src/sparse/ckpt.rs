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

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::Logits;
use shiso_kvstore::{CacheStats, CommitOpts, KvCache, PathLayout, Reason, SpillConfig};
use shiso_kvstore::{SpillOutcome, SpillRefusal};

use crate::ckpt::{Checkpoint, CkptConfig, ResumeInfo, SaveOutcome, scan_prefixes};
use crate::ckpt_layout::CkptIdentity;
use crate::shape::LayerKind;
use crate::sparse::kernels::F32_BYTES;
use crate::sparse::mtp::SpecDepth;
use crate::sparse::paged::{self, PagePlanes, conv_state_bytes, ple_window_bytes, state_bytes};
use crate::sparse::scratch::Buf;
use crate::sparse::shape::SparseShape;
use crate::sparse::stack::{MAX_PREFILL_CHUNK, SparseSession, SparseStack};

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

    /// Opens the checkpoint tier over `config.dir`, tagged with `identity`.
    ///
    /// The tier is part of the cache rather than a layer above it, so turning it
    /// on rebuilds the cache with a spill directory and the byte description a
    /// stored payload needs. That is why it is refused while sessions are open:
    /// a session holds page references into the pool the old cache owns, and the
    /// new one has its own.
    ///
    /// `identity` is not a caller's choice. It comes from the crate that loaded
    /// the checkpoint, which is the only party that knows which family wrote the
    /// weights and how they were quantized.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when a session is open or the stack has been
    /// torn down, and [`CoreError::Backend`] when the directory cannot be
    /// opened.
    pub fn ckpt_enable(&mut self, config: &CkptConfig, identity: CkptIdentity) -> CoreResult<()> {
        if !self.sessions.is_empty() {
            return Err(CoreError::Unsupported {
                what: "opening a checkpoint directory while sessions are already running",
                backend: "engine",
            });
        }
        let layout = path_layout(&self.shape, self.opts.engine.ctx, self.opts.spec, identity);
        let geometry =
            paged::cache_geometry(&self.shape, PagePlanes::indexer_only(), self.opts.spec)
                .with_layout(layout);
        let mut spill = SpillConfig::new(config.dir.clone(), config.options.budget_mb);
        spill.reject_different_quant = config.options.reject_different_quant;
        let (pool_pages, ckpt_slots) = (
            self.opts.engine.kv_pool_pages,
            self.opts.engine.kv_ckpt_slots,
        );
        let resident = self.resident.as_mut().ok_or(CoreError::Unsupported {
            what: "a stack whose device state teardown already released",
            backend: "engine",
        })?;
        let cache = KvCache::open(
            &geometry,
            Box::new(resident.kv.clone()),
            pool_pages,
            ckpt_slots,
            Some(spill),
        )
        .map_err(cache_failure)?;
        resident.cache = Arc::new(cache);
        self.ckpt = Some(Checkpoint {
            options: config.options,
            identity,
        });
        tracing::info!(
            dir = %config.dir.display(),
            min_tokens = config.options.min_tokens,
            budget_mb = config.options.budget_mb,
            model_id = identity.model_id,
            quant_bits = identity.quant_bits,
            "sparse checkpoint tier open"
        );
        Ok(())
    }

    /// Registers the span a session just wrote, and admits a checkpoint when the
    /// cache says the position is worth one.
    ///
    /// The snapshot is asked for before it is taken, for the reason the dense
    /// stack gives: a blob is a multi-megabyte read back off the device, so
    /// paying it on every committed span would cost more than the prefill it
    /// saves. So a snapshot is only considered at a boundary worth one on its own
    /// terms - the end of a prompt, or a stride multiple - which are the
    /// positions a later request can actually resume from.
    ///
    /// `earned` says the span ends a prompt, which is the boundary a later
    /// request would resume from. It is the caller's to answer and not this
    /// function's to infer, because the two callers look identical from here:
    /// a prompt's last chunk and a decode step both write one logits row, so a
    /// checkpoint earned by *having* a row is a checkpoint earned by every
    /// token - and that is the whole recurrent stack read back to the host per
    /// token, which is most of a decode step.
    ///
    /// `frontier_row` names which row of the logits block this session's last
    /// position produced. A checkpoint admitted without one cannot be resumed
    /// from without re-running a step.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], whatever the device reports for the
    /// snapshot reads, and whatever the cache reports for a span it could not
    /// register.
    pub(crate) fn commit(
        &mut self,
        session: SessionId,
        earned: bool,
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
        // A prompt's final chunk is a boundary on its own terms - it is where a
        // later request carrying the same prompt would want to resume - so it is
        // *earned*, and an earned checkpoint survives an eviction pass that
        // takes the speculative ones back. Every other boundary this stack takes
        // is a chunk end it happened to stop at.
        let periodic = q > 0 && q.is_multiple_of(MAX_PREFILL_CHUNK);
        // A speculating session's blob carries its staged draft row, and a
        // resume marks that row usable because the blob carried it. So a session
        // that has not staged one is not checkpointed at all rather than
        // checkpointed with zeros that a later run would draft from.
        let staged = live.spec_depth().is_speculative() == live.pending_draft().is_some();
        let admit = (earned || periodic)
            && staged
            && resident.cache.next_admission(
                live.pages.frontier(),
                &live.history,
                &[],
                MAX_PREFILL_CHUNK,
            ) == Some(q);
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
        let mut opts = CommitOpts::span(&[]);
        if admit {
            opts = opts.checkpoint(staging.ckpt.as_slice(), earned);
            if frontier_row.is_some() {
                opts = opts.logits_row(staging.logits.as_slice());
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

    /// Stores the session's prefix at its frontier. See
    /// [`crate::ckpt::Checkpointed::kv_save`].
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and [`CoreError::Backend`] when the store
    /// fails for a reason that is not a refusal.
    pub fn ckpt_save(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        prompt_text: &[u8],
        reason: Reason,
    ) -> CoreResult<SaveOutcome> {
        let Some(ckpt) = self.ckpt else {
            return Ok(SaveOutcome::Disabled);
        };
        let live = self
            .sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let written = live.written();
        let offered = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        // The frontier gate, and it compares the path rather than only its
        // length: a caller holding the right number of the wrong tokens would
        // otherwise file one conversation's state under another's key.
        if offered != written || tokens != live.history() {
            return Ok(SaveOutcome::NotAtFrontier {
                tokens: offered,
                written,
            });
        }
        if offered < ckpt.options.min_tokens {
            return Ok(SaveOutcome::BelowMinTokens {
                tokens: offered,
                min: ckpt.options.min_tokens,
            });
        }
        if prompt_text.is_empty() {
            return Ok(SaveOutcome::EmptyText);
        }
        let node = live.pages.frontier().node();
        let cache = &self.resident()?.cache;
        let outcome = cache
            .spill(node, prompt_text, reason)
            .map_err(cache_failure)?;
        match outcome {
            SpillOutcome::Written { id, bytes } => {
                tracing::debug!(entry = %id, bytes, tokens = offered, ?reason, "checkpoint written");
                Ok(SaveOutcome::Written(id))
            }
            SpillOutcome::AlreadyStored { id } => Ok(SaveOutcome::AlreadyStored(id)),
            SpillOutcome::Refused {
                reason: SpillRefusal::StaleLogits,
            } => Ok(SaveOutcome::NoFrontierLogits),
            SpillOutcome::Refused { reason } => Ok(SaveOutcome::StoreRefused(reason)),
            // The store's answer set is open, and an answer this build cannot
            // name is not a refusal it may quietly report as one.
            other => {
                tracing::error!(
                    ?other,
                    "the store answered in terms this build has no name for"
                );
                Err(CoreError::Unsupported {
                    what: "a spill outcome this build cannot report",
                    backend: "engine",
                })
            }
        }
    }

    /// Stores every open session. See
    /// [`crate::ckpt::Checkpointed::kv_shutdown_sweep`].
    pub fn ckpt_shutdown_sweep(
        &mut self,
        text_of: &mut dyn FnMut(SessionId) -> Option<(Vec<TokenId>, Vec<u8>)>,
    ) -> usize {
        if self.ckpt.is_none() {
            return 0;
        }
        let mut written = 0;
        for session in self.order.clone() {
            let Some((tokens, text)) = text_of(session) else {
                continue;
            };
            match self.ckpt_save(session, &tokens, &text, Reason::Shutdown) {
                Ok(outcome) if outcome.stored().is_some() => written += 1,
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
        tracing::info!(written, "sparse shutdown sweep complete");
        written
    }

    /// Stands `session` on the longest stored prefix of `prompt_text`. See
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
        prompt_text: &[u8],
    ) -> CoreResult<Option<ResumeInfo<'_>>> {
        let Some(ckpt) = self.ckpt else {
            return Ok(None);
        };
        if !self.sessions.contains_key(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        let now = self.now();
        let capacity = self.opts.engine.ctx;
        let vocab = self.shape.vocab;
        let cache = Arc::clone(&self.resident()?.cache);

        let scan = scan_prefixes(&cache.spill_entries(), prompt_text, ckpt.gate(capacity));
        for id in scan.foreign() {
            match cache.forget_spill(id) {
                Ok(_) => tracing::info!(entry = %id, "discarded an entry written by another model"),
                Err(error) => {
                    tracing::warn!(entry = %id, %error, "could not discard a foreign entry");
                }
            }
        }
        let Some(best) = scan.best() else {
            return Ok(None);
        };
        let Some(restored) = cache.restore(best.id(), now).map_err(cache_failure)? else {
            return Ok(None);
        };
        let Some(reference) = restored.ckpt else {
            // The pages came back but no slot could hold the state. A transient
            // shortage rather than a bad file, so the entry stays and the request
            // prefills.
            tracing::debug!(entry = %best.id(), "no checkpoint slot for a restored prefix");
            return Ok(None);
        };
        let Some(row) = restored.logits else {
            // A payload with no frontier row cannot be resumed from without
            // re-running a step, which is the one thing this tier promises not to
            // do. Nothing this build writes is like that, so the file is one it
            // can never use.
            tracing::warn!(entry = %best.id(), "a stored prefix carries no frontier row; discarding");
            let _discarded = cache.forget_spill(best.id());
            return Ok(None);
        };
        let want_row = u64::from(vocab).saturating_mul(F32_BYTES);
        if u64::try_from(row.len()).unwrap_or(u64::MAX) != want_row {
            return Err(CoreError::ShapeMismatch {
                field: "restored logits row",
                expected: want_row,
                got: u64::try_from(row.len()).unwrap_or(u64::MAX),
            });
        }

        let mut blob =
            vec![0u8; usize::try_from(cache.geometry().ckpt_bytes).unwrap_or(usize::MAX)];
        cache
            .ckpt_state_into(reference, &mut blob)
            .map_err(cache_failure)?;
        let tokens = restored.tokens;
        let reused = u32::try_from(tokens.len()).unwrap_or(u32::MAX);

        {
            // Split borrows: the frontier and the per-session state belong to the
            // session, and the bytes move through the backend.
            let Self {
                backend, sessions, ..
            } = self;
            let live = sessions
                .get_mut(&session)
                .ok_or(CoreError::UnknownSession(session))?;
            live.pages.adopt(restored.frontier)?;
            live.history.clear();
            live.history.extend_from_slice(&tokens);
            live.written = reused.min(capacity);
            restore_state(backend.as_ref(), live, &blob)?;
        }
        // The frontier row goes exactly where a freshly computed one would be, so
        // a caller cannot tell a resumed row from a prefilled one.
        {
            let logits = self
                .resident()?
                .scratch
                .view(Buf::Logits)?
                .slice(0, want_row)?;
            self.backend.copy_h2d(logits, &row)?;
        }
        tracing::info!(
            entry = %best.id(),
            tokens = reused,
            text_bytes = best.text_bytes(),
            "resumed a stored sparse prefix"
        );
        let logits = self
            .resident()?
            .scratch
            .view(Buf::Logits)?
            .slice(0, want_row)?;
        Ok(Some(ResumeInfo::new(
            reused,
            best.text_bytes(),
            tokens,
            Logits::new(logits, 1, vocab)?,
        )))
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

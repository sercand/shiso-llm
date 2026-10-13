//! Checkpointing a session to disk, and standing one back up from it.
//!
//! The cache below this keeps a prefix for as long as memory allows. Its disk
//! tier is what is left when memory does not, and when the process does not: a
//! session's key/value pages and its recurrent state at one exact position,
//! stored under the token path that produced them, so the next process to see a
//! prompt starting with that path answers without prefilling it again.
//!
//! [`crate::ckpt_layout`] says which build may read a file. This module says
//! when a stack stores one and how a stored one is stood back up; the files
//! themselves are `shiso-kvstore`'s.
//!
//! # Off until somebody turns it on
//!
//! A stack has no disk tier until [`Checkpointed::kv_enable`] is called with a
//! directory. There is no environment variable and no default path: a cache
//! directory is a deployment decision, and a process that was never told where
//! to put one must not invent a place.
//!
//! # Keyed by tokens
//!
//! An entry is named by the token path it holds and the build that computed
//! it, never by rendered text. Two renderings of one conversation that tokenize
//! the same share entries, a rendering that drops an old turn's reasoning simply
//! stops matching where the tokens stop matching, and a resume can never land a
//! session on a path its prompt does not start with.
//!
//! # Nothing is re-run on a resume
//!
//! An entry at the very end of a prompt carries the frontier logits row, so a
//! resumed session is already standing on the answer its next sample reads.
//! Re-running the last step to recover that row would be slower and different:
//! the row would be recomputed from restored state rather than being the row the
//! original run produced.
//!
//! # Writes are behind
//!
//! Storing queues the checkpoint and returns: the pages still to copy are pinned,
//! [`Checkpointed::kv_pump`] copies a few between steps, and the store's writer
//! thread does every file operation. A turn is never slower for being stored
//! than by the snapshot its checkpoint needed anyway.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use shiso_core::backend::Backend;
use shiso_core::buffer::DeviceView;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::media::MediaKey;
use shiso_core::model::{Logits, Model};
use shiso_kvstore::{
    CacheStats, CommitOpts, DiskConfig, EntryKey, KvCache, PersistOutcome, PersistRefusal,
    PumpReport, Reason, ReuseSource,
};

use crate::ckpt_layout::{CkptIdentity, path_layout};
use crate::kv;
use crate::launch::F32_BYTES;
use crate::sessions::Session;
use crate::stack::{HybridStack, cache_failure, snapshot};

/// Checkpoints shorter than this are not worth storing, by default.
///
/// Below it the files cost more to write and read than the prefill they save,
/// and a directory full of such entries prices the ones that matter out of their
/// budget.
pub const CKPT_MIN_TOKENS_DEFAULT: u32 = shiso_kvstore::DEFAULT_MIN_TOKENS;

/// The default disk budget, as a multiple of the key/value pool's bytes.
///
/// The disk has to hold more than memory does to be worth having - an evicted
/// session is only resumable if its pages are still on disk - and twice the
/// pool keeps every resident session plus as many again that memory gave up.
pub const CKPT_BUDGET_POOL_MULTIPLE: u64 = 2;

/// Bytes in a mebibyte.
const MIB: u64 = 1024 * 1024;

/// What the disk tier does once it is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CkptOptions {
    /// Checkpoints shorter than this are neither stored nor looked up.
    pub min_tokens: u32,
    /// Bytes the directory may hold, in mebibytes. `None` is
    /// [`CKPT_BUDGET_POOL_MULTIPLE`] times the key/value pool's bytes, which is
    /// a derived answer and not a fallback: zero is refused.
    pub budget_mb: Option<u64>,
    /// Host bytes copied off the device and not yet written, in mebibytes.
    pub staging_mb: u64,
    /// Checkpoints that may wait for the pump before the least valuable is
    /// dropped.
    pub max_jobs: u32,
}

impl CkptOptions {
    /// The defaults: the store's floor, a budget of twice the pool, the store's
    /// staging and queue.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min_tokens: CKPT_MIN_TOKENS_DEFAULT,
            budget_mb: None,
            staging_mb: shiso_kvstore::DEFAULT_STAGING_MB,
            max_jobs: shiso_kvstore::DEFAULT_MAX_JOBS,
        }
    }

    /// The store's configuration for a pool of `pool_bytes`, on a device whose
    /// arithmetic is named `numerics`.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a zero budget, floor or staging size:
    /// each would silently turn the tier into something the operator did not
    /// ask for.
    pub(crate) fn disk(
        &self,
        dir: PathBuf,
        pool_bytes: u64,
        numerics: String,
    ) -> CoreResult<DiskConfig> {
        let budget = match self.budget_mb {
            Some(0) => {
                return Err(CoreError::InvalidValue {
                    key: "checkpoint disk budget (MiB)",
                    value: "0".to_owned(),
                });
            }
            Some(mb) => mb.saturating_mul(MIB),
            None => pool_bytes.saturating_mul(CKPT_BUDGET_POOL_MULTIPLE),
        };
        if self.staging_mb == 0 {
            return Err(CoreError::InvalidValue {
                key: "checkpoint disk staging (MiB)",
                value: "0".to_owned(),
            });
        }
        let mut config = DiskConfig::new(dir, budget);
        config.min_tokens = self.min_tokens.max(1);
        config.staging_bytes = self.staging_mb.saturating_mul(MIB);
        config.max_jobs = self.max_jobs.max(1);
        config.numerics = numerics;
        Ok(config)
    }
}

impl Default for CkptOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Where the disk tier lives and how it behaves.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CkptConfig {
    /// The directory, created if it is absent.
    pub dir: PathBuf,
    /// What the tier does once it is on.
    pub options: CkptOptions,
}

impl CkptConfig {
    /// A tier rooted at `dir` with the default behaviour.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            options: CkptOptions::new(),
        }
    }

    /// The same tier with `options` in place of the defaults.
    #[must_use]
    pub const fn with_options(mut self, options: CkptOptions) -> Self {
        self.options = options;
        self
    }
}

/// The tier as a live stack holds it: what it does. Who wrote its entries is in
/// every key, not here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Checkpoint {
    pub(crate) options: CkptOptions,
}

/// What a save did.
///
/// Not a `bool`. Every refusal here is something an operator debugging a cache
/// that never fills wants to read directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaveOutcome {
    /// Queued for the writer under this key.
    Queued(EntryKey),
    /// The same entry was already queued or on disk.
    AlreadyStored(EntryKey),
    /// The stack has no disk tier: nobody called [`Checkpointed::kv_enable`].
    Disabled,
    /// The session read media with no content key, or stands inside one of
    /// its pictures: a run of placeholder tokens is the same id repeated, so
    /// such a path cannot tell two pictures apart and is never stored (see
    /// [`Checkpointed::kv_set_media_key`]).
    CarriesMedia,
    /// A speculating session with no staged draft row: a checkpoint would carry
    /// a row nothing computed.
    NoDraftRow,
    /// The store declined, in its own words.
    Refused(PersistRefusal),
}

impl SaveOutcome {
    /// The key an entry is now stored under, whether this call queued it or
    /// found it.
    #[must_use]
    pub const fn stored(&self) -> Option<EntryKey> {
        match self {
            Self::Queued(key) | Self::AlreadyStored(key) => Some(*key),
            _ => None,
        }
    }

    /// The outcome a store's answer maps to.
    #[must_use]
    pub const fn from_persist(outcome: PersistOutcome) -> Self {
        match outcome {
            PersistOutcome::Queued { key, .. } => Self::Queued(key),
            PersistOutcome::AlreadyStored { key } => Self::AlreadyStored(key),
            PersistOutcome::Refused(why) => Self::Refused(why),
            // The store's answer set is open; one this build cannot name is
            // reported as the most conservative refusal rather than invented.
            _ => Self::Refused(PersistRefusal::NotConfigured),
        }
    }
}

/// Where a resumed prefix came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResumeSource {
    /// A checkpoint the in-memory tree still held.
    Memory,
    /// A checkpoint read back from disk.
    Disk,
}

/// What a resume recovered.
///
/// Not `#[non_exhaustive]`: a resume hit is something a *fake* implementation of
/// the trait has to be able to return, and a fake that can only ever report a
/// miss leaves the hit path testable by nothing but a real store on a real
/// device.
#[derive(Debug)]
pub struct ResumeInfo<'a> {
    /// Positions the session now stands at, which is the prefill it does not pay.
    pub tokens_reused: u32,
    /// The token path the session stands on, position zero first: always the
    /// first `tokens_reused` tokens of the prompt it was resumed for.
    pub tokens: Vec<TokenId>,
    /// The frontier logits row, present exactly when the resume covered the
    /// whole prompt: the row the original run produced, which the caller samples
    /// the first token from.
    pub frontier_logits: Option<Logits<'a>>,
    /// Where the prefix came from.
    pub source: ResumeSource,
}

impl<'a> ResumeInfo<'a> {
    /// What a resume recovered, given directly: the real path builds one from
    /// the cache, and this exists so a test double can report a hit.
    #[must_use]
    pub const fn new(
        tokens_reused: u32,
        tokens: Vec<TokenId>,
        frontier_logits: Option<Logits<'a>>,
        source: ResumeSource,
    ) -> Self {
        Self {
            tokens_reused,
            tokens,
            frontier_logits,
            source,
        }
    }
}

/// Checkpointing, layered on [`Model`] rather than added to it.
///
/// [`Model`] is the contract a scheduler holds every model through, and not every
/// model has a disk tier. A separate trait keeps these methods off the models that
/// do not implement them, and a scheduler that holds a `&mut dyn Checkpointed` can
/// still hand it to anything taking a `&mut dyn Model`.
pub trait Checkpointed: Model {
    /// Turns the tier on over `config.dir`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when sessions are already open - the tier is a
    /// property of the cache the sessions draw pages from - [`CoreError::InvalidValue`]
    /// for a zero budget, and [`CoreError::Backend`] when the directory cannot be
    /// opened.
    fn kv_enable(&mut self, config: &CkptConfig) -> CoreResult<()>;

    /// Whether a directory has been opened.
    fn kv_checkpoint_enabled(&self) -> bool;

    /// Stands `session` on the deepest stored checkpoint whose path `prompt`
    /// starts with - in memory or on disk - and reports how far that reaches.
    ///
    /// `Ok(None)` is a cold start: nothing matched, or what matched was not
    /// usable. Every other failure is an `Err`.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and [`CoreError::Backend`] when the device or
    /// the cache fails while the prefix is being installed.
    fn kv_resume(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
    ) -> CoreResult<Option<ResumeInfo<'_>>>;

    /// Stores the checkpoint at the session's frontier - taking one there first
    /// when the frontier has none, as at the end of a turn - under `reason`.
    ///
    /// The path is the session's own history, which is what makes the key
    /// right by construction: there is no caller-supplied path to disagree with
    /// the state.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and whatever the device reports for the
    /// snapshot.
    fn kv_save(&mut self, session: SessionId, reason: Reason) -> CoreResult<SaveOutcome>;

    /// Message boundaries of the prompt `session` is about to read, as absolute
    /// positions: the stack stops a chunk at each and takes a checkpoint there,
    /// so a later request sharing only that much can resume from it.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`].
    fn kv_set_hints(&mut self, session: SessionId, hints: &[u32]) -> CoreResult<()> {
        let _ = (session, hints);
        Ok(())
    }

    /// Marks `session` as reading its prompt only to score the last row.
    ///
    /// Such a prompt's end is answered and dropped, so the stack takes no
    /// checkpoint there, none at its stride points, and stores nothing on
    /// disk; the shared boundaries its hints name are still checkpointed in
    /// memory, which is how sibling questions over one input resume each
    /// other. A no-op by default: a stack that does not distinguish keeps
    /// checkpointing as it would for any prompt.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`].
    fn kv_set_scoring(&mut self, session: SessionId) -> CoreResult<()> {
        let _ = session;
        Ok(())
    }

    /// Files `session`'s path under `key`, the content key of the media prompt
    /// `prompt` it is about to resume or read, so its prefix can be stored and
    /// resumed like a text prompt's.
    ///
    /// Refused by default, and the refusal is the answer a host must honour: a
    /// model that cannot key its cache files a media prompt under placeholder
    /// runs that are the same for any picture, so the host keeps such a
    /// session off the tier altogether.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] from a model that cannot key its cache, and
    /// whatever the implementation refuses - see
    /// [`crate::sparse::SparseStack::ckpt_set_media_key`].
    fn kv_set_media_key(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
        key: MediaKey<'_>,
    ) -> CoreResult<()> {
        let _ = (session, prompt, key);
        Err(CoreError::Unsupported {
            what: "a content-keyed prefix for a media prompt",
            backend: "checkpoint",
        })
    }

    /// Copies up to `max_bytes` of queued checkpoint pages off the device for
    /// the writer. The host calls this between steps.
    ///
    /// # Errors
    ///
    /// Whatever the device reports for a copy.
    fn kv_pump(&mut self, max_bytes: u64) -> CoreResult<PumpReport> {
        let _ = max_bytes;
        Ok(PumpReport::default())
    }

    /// Copies everything queued and waits up to `timeout` for it to land.
    /// Returns whether it all did.
    fn kv_drain(&mut self, timeout: Duration) -> bool {
        let _ = timeout;
        true
    }

    /// Reports a measured prefill: `tokens` computed in `seconds`. What a disk
    /// resume is priced against.
    fn kv_observe_prefill(&mut self, tokens: u32, seconds: f64) {
        let _ = (tokens, seconds);
    }

    /// The cache underneath this model, for a caller that reads its numbers or
    /// warms its disk tier from another thread.
    fn kv_cache(&self) -> Option<Arc<KvCache>> {
        None
    }

    /// What the cache underneath this model currently holds.
    ///
    /// Defaulted to `None` rather than required: a model with no disk tier has
    /// nothing to report, and a zero would read as "measured, and empty".
    fn kv_cache_stats(&self) -> Option<CacheStats> {
        None
    }

    /// Stores every open session on the way down, and reports how many were
    /// queued or already stored.
    fn kv_shutdown_sweep(&mut self) -> usize;
}

/// Maps the cache's resume source onto this crate's vocabulary.
#[must_use]
pub(crate) const fn source_of(source: ReuseSource) -> ResumeSource {
    match source {
        ReuseSource::Disk => ResumeSource::Disk,
        _ => ResumeSource::Memory,
    }
}

// ---------------------------------------------------------------------------
// the dense stack's half
// ---------------------------------------------------------------------------

impl HybridStack {
    /// Whether a checkpoint directory has been opened.
    #[must_use]
    pub const fn ckpt_enabled(&self) -> bool {
        self.ckpt.is_some()
    }

    /// The cache underneath this stack.
    #[must_use]
    pub fn cache(&self) -> Option<Arc<KvCache>> {
        self.resident
            .as_ref()
            .map(|resident| Arc::clone(&resident.cache))
    }

    /// Opens the disk tier over `config.dir`, tagged with `identity`.
    ///
    /// The tier is part of the cache rather than a layer above it, so turning it
    /// on rebuilds the cache with a disk directory and the layout its keys name.
    /// That is why it is refused while sessions are open: a session holds page
    /// references into the pool the old cache owns, and the new one has its own.
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
        let layout = path_layout(&self.shape, &self.geometry, self.opts.ctx, identity);
        let geometry = kv::geometry(&self.shape).with_layout(layout);
        let (pool_pages, ckpt_slots) = (self.opts.kv_pool_pages, self.opts.kv_ckpt_slots);
        let pool_bytes = geometry.page_bytes_total(pool_pages);
        let disk =
            config
                .options
                .disk(config.dir.clone(), pool_bytes, self.backend.numerics_id())?;
        let budget_bytes = disk.budget_bytes;
        let resident = self.resident.as_mut().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
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
        self.ckpt = Some(Checkpoint {
            options: config.options,
        });
        tracing::info!(
            dir = %config.dir.display(),
            min_tokens = config.options.min_tokens,
            budget_bytes,
            model_id = identity.model_id,
            quant_bits = identity.quant_bits,
            "checkpoint tier open"
        );
        Ok(())
    }

    /// Stores the session's frontier. See [`Checkpointed::kv_save`].
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and whatever the device or the cache report.
    pub fn ckpt_save(&mut self, session: SessionId, reason: Reason) -> CoreResult<SaveOutcome> {
        let Some(Checkpoint { options }) = self.ckpt else {
            return Ok(SaveOutcome::Disabled);
        };
        let now = self.now();
        let Self {
            backend,
            resident,
            sessions,
            staging,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let live = sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        if live.carries_media() {
            return Ok(SaveOutcome::CarriesMedia);
        }
        let q = live.written();
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
            .frontier_ckpt(live.kv().frontier())
            .is_some_and(|ckpt| ckpt.pos == q);
        if !standing {
            // The end of a turn: the frontier moved one decoded token at a time
            // and none of those steps took a snapshot, so one is taken here.
            snapshot(backend.as_ref(), live, &mut staging.ckpt)?;
            let (history, kv) = live.commit_parts();
            resident
                .cache
                .commit(
                    kv.frontier_mut(),
                    history,
                    q,
                    CommitOpts::span(&[]).checkpoint(staging.ckpt.as_slice(), true),
                    now,
                )
                .map_err(cache_failure)?;
        }
        let outcome = resident
            .cache
            .persist(live.kv().frontier(), reason)
            .map_err(cache_failure)?;
        Ok(SaveOutcome::from_persist(outcome))
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

    /// Stores every open session. See [`Checkpointed::kv_shutdown_sweep`].
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
                    // costs its next request a prefill, and a shutdown that
                    // stopped here would cost every session behind it the same.
                    tracing::warn!(%session, %error, "could not store a session on the way down");
                }
            }
        }
        tracing::info!(stored, "shutdown sweep complete");
        stored
    }

    /// Records the shared boundaries of the prompt `session` is about to read.
    /// See [`Checkpointed::kv_set_hints`].
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`].
    pub fn ckpt_set_hints(&mut self, session: SessionId, hints: &[u32]) -> CoreResult<()> {
        self.sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .set_hints(hints);
        Ok(())
    }

    /// Marks `session` as reading its prompt only to score it. See
    /// [`Checkpointed::kv_set_scoring`].
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`].
    pub fn ckpt_set_scoring(&mut self, session: SessionId) -> CoreResult<()> {
        self.sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .set_scoring();
        Ok(())
    }

    /// Stands `session` on the deepest usable checkpoint of `prompt`. See
    /// [`Checkpointed::kv_resume`].
    ///
    /// `Ok(None)` is a cold start and not a refusal: nothing stored matched.
    /// The caller prefills, exactly as it would have with nothing stored.
    ///
    /// No directory need be open. The cache keeps the checkpoints a commit
    /// admits in memory whether or not it has a disk tier, and the disk is
    /// only where they also persist; a server started without one still
    /// stands a sibling question on the state its first question just read.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and [`CoreError::Backend`] when the device or
    /// the cache fails while the prefix is being installed.
    pub fn ckpt_resume(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
    ) -> CoreResult<Option<ResumeInfo<'_>>> {
        if !self.sessions.contains_key(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        let now = self.now();
        let capacity = self.opts.ctx;
        let vocab = self.geometry.vocab;
        let slots = self.recurrent_slots;
        let cache = Arc::clone(&self.resident()?.cache);
        let Some(resume) = cache.resume(prompt, now).map_err(cache_failure)? else {
            return Ok(None);
        };
        let Some(reference) = resume.ckpt else {
            return Ok(None);
        };
        let reused = resume.ckpt_pos;
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
            let Self {
                backend, sessions, ..
            } = self;
            let live = sessions
                .get_mut(&session)
                .ok_or(CoreError::UnknownSession(session))?;
            live.adopt(resume.frontier, tokens.clone(), capacity)?;
            live.set_resumed_at(reused);
            restore_stack(backend.as_ref(), live, slots, &blob)?;
        }
        let logits = match row {
            Some(row) => {
                let view = self.resident()?.scratch.logits().slice(0, want_row)?;
                self.backend.copy_h2d(view, &row)?;
                let view = self.resident()?.scratch.logits().slice(0, want_row)?;
                Some(Logits::new(view, 1, vocab)?)
            }
            None => None,
        };
        tracing::info!(tokens = reused, ?source, "resumed a stored prefix");
        Ok(Some(ResumeInfo::new(reused, tokens, logits, source)))
    }
}

/// Copies a checkpoint blob back into a session's recurrent stack.
///
/// Slot by slot in ordinal order, each slot's convolution window followed by its state
/// matrix, which is the order the snapshot wrote them in. Nothing is interpreted: each
/// half is a byte range whose length the device buffer itself states, and a blob that
/// does not end exactly where the last buffer does is refused rather than partially
/// applied.
fn restore_stack(
    backend: &dyn Backend,
    session: &mut Session,
    slots: usize,
    blob: &[u8],
) -> CoreResult<()> {
    let mut at = 0usize;
    for ordinal in 0..u32::try_from(slots).unwrap_or(u32::MAX) {
        let (conv, state) = session.gdn_halves(ordinal)?;
        at = write_view(backend, conv, blob, at)?;
        at = write_view(backend, state, blob, at)?;
    }
    if at != blob.len() {
        return Err(CoreError::ShapeMismatch {
            field: "restored recurrent state",
            expected: u64::try_from(blob.len()).unwrap_or(u64::MAX),
            got: u64::try_from(at).unwrap_or(u64::MAX),
        });
    }
    Ok(())
}

/// Writes the range of `blob` starting at `at` that fills `view`, and reports where
/// the next one starts.
fn write_view(
    backend: &dyn Backend,
    view: DeviceView<'_>,
    blob: &[u8],
    at: usize,
) -> CoreResult<usize> {
    let len = usize::try_from(view.len()).unwrap_or(usize::MAX);
    let end = at.saturating_add(len);
    let bytes = blob.get(at..end).ok_or(CoreError::ShapeMismatch {
        field: "restored recurrent state",
        expected: u64::try_from(end).unwrap_or(u64::MAX),
        got: u64::try_from(blob.len()).unwrap_or(u64::MAX),
    })?;
    backend.copy_h2d(view, bytes)?;
    Ok(end)
}

#[cfg(test)]
mod tests;

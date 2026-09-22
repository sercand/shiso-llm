//! Checkpointing a session to disk, and standing one back up from it.
//!
//! The cache below this keeps a prefix for as long as memory allows. This tier is
//! what is left when memory does not, and when the process does not: a session's
//! whole prefix - its key/value pages, its recurrent stack and the logits row it is
//! standing on - written to a file named by the rendered text it caches, so that the
//! next process to see that text answers without prefilling it again.
//!
//! [`crate::ckpt_layout`] says what the bytes are. This module says when they are
//! written, which stored entry is worth reading, and what happens to one that is not.
//!
//! # Off until somebody turns it on
//!
//! A stack has no checkpoint tier until [`HybridStack::ckpt_enable`] is called with a
//! directory. There is no environment variable and no default path: a cache
//! directory is a deployment decision, and a process that was never told where to put
//! one must not invent a place.
//!
//! # The engine hashes the bytes it is handed
//!
//! An entry is keyed by the sha1 of the *rendered* prompt text, and this crate has no
//! tokenizer, so both [`HybridStack::ckpt_save`] and [`HybridStack::ckpt_resume`] take
//! that text as bytes and the caller owns the rendering. The two sides must render
//! identically or nothing ever matches - which is a cold start, never a wrong answer,
//! because a key that was derived differently simply names a file that is not there.
//!
//! # Why nothing is re-run on a resume
//!
//! The stored payload carries the frontier logits row, so a resumed session is
//! already standing on the answer its next sample reads. Re-running the last step to
//! recover that row would be both slower and *different*: the row would be recomputed
//! from restored state rather than being the row the original run produced, and
//! nothing downstream could tell the two apart.
//!
//! # What a refusal says
//!
//! [`SaveOutcome`] names why nothing was written rather than answering `false`. Six of
//! its variants are ordinary conditions an operator watching a cold cache wants to
//! read in a log line, and collapsing them into one bit would make "the prompt was too
//! short" indistinguishable from "the session was not at its frontier".

use std::path::PathBuf;
use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::buffer::DeviceView;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::{Logits, Model};
use shiso_kvstore::CacheStats;
use shiso_kvstore::{
    KvCache, Reason, SpillConfig, SpillEntry, SpillId, SpillOutcome, SpillRefusal,
};

use crate::ckpt_layout::{CkptIdentity, path_layout};
use crate::kv;
use crate::launch::F32_BYTES;
use crate::sessions::Session;
use crate::stack::{HybridStack, cache_failure};

/// Prefixes shorter than this are not worth a checkpoint file.
///
/// Below it the file costs more to write and read than the prefill it saves, and a
/// directory full of such entries prices the ones that matter out of their budget.
pub const CKPT_MIN_TOKENS_DEFAULT: u32 = 512;

/// The default directory ceiling, in mebibytes.
pub const CKPT_BUDGET_MB_DEFAULT: u64 = 4096;

/// What the checkpoint tier does once it is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CkptOptions {
    /// Prefixes shorter than this are not stored.
    ///
    /// The one intrinsic gate. How *long* a prefix to store - whether to trim to a
    /// message boundary, how often to refresh a growing conversation - is a
    /// scheduler's decision and is not made here.
    pub min_tokens: u32,
    /// Whether an entry written by a build with a different weight quantization is
    /// refused.
    ///
    /// True is the safe answer and the default: the same prompt through a 4-bit and a
    /// 2-bit build produces different key/value rows, so a cross-load answers from
    /// rows that were never computed for these weights.
    pub reject_different_quant: bool,
    /// Bytes the directory may hold, in mebibytes.
    pub budget_mb: u64,
}

impl CkptOptions {
    /// The defaults: a 512-token floor, quantization mismatches refused, a 4 GiB
    /// directory.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min_tokens: CKPT_MIN_TOKENS_DEFAULT,
            reject_different_quant: true,
            budget_mb: CKPT_BUDGET_MB_DEFAULT,
        }
    }
}

impl Default for CkptOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Where the checkpoint tier lives and how it behaves.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CkptConfig {
    /// The directory entries are written to, created if it is absent.
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

/// The tier as a live stack holds it: what it does, and who wrote its entries.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Checkpoint {
    pub(crate) options: CkptOptions,
    pub(crate) identity: CkptIdentity,
}

/// Why a save did not produce a file, or which file it produced.
///
/// Not a `bool`. Every refusal here is something an operator debugging a cache that
/// never fills wants to read directly, and the four that are ordinary conditions -
/// disabled, too short, not at the frontier, no text - are indistinguishable once
/// they have been folded into one bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaveOutcome {
    /// The prefix was written under this key.
    Written(SpillId),
    /// An equivalent entry was already stored, so nothing was written.
    AlreadyStored(SpillId),
    /// The stack has no checkpoint tier: nobody called [`HybridStack::ckpt_enable`].
    Disabled,
    /// The prefix is shorter than the floor.
    BelowMinTokens {
        /// Tokens offered.
        tokens: u32,
        /// The floor they fell under.
        min: u32,
    },
    /// The tokens handed over are not the ones the session is standing on, so the
    /// state at its frontier is not the state they describe.
    NotAtFrontier {
        /// Tokens offered.
        tokens: u32,
        /// Positions the session has actually written.
        written: u32,
    },
    /// No text to key the entry by, so no later request could ever name it.
    EmptyText,
    /// The checkpoint at the frontier has no logits row, so a resume from it would
    /// have to re-run a step to get one. See the module docs.
    NoFrontierLogits,
    /// The store declined, in its own words.
    StoreRefused(SpillRefusal),
}

impl SaveOutcome {
    /// The key an entry is now stored under, whether this call wrote it or found it.
    #[must_use]
    pub const fn stored(&self) -> Option<SpillId> {
        match self {
            Self::Written(id) | Self::AlreadyStored(id) => Some(*id),
            _ => None,
        }
    }
}

/// What a resume recovered.
///
/// Not `#[non_exhaustive]`, for the reason the workspace settled on: a resume hit
/// is something a *fake* implementation of the trait has to be able to return, and
/// a fake that can only ever report a miss leaves the hit path testable by nothing
/// but a real store on a real device. That gap was found by the scheduler, which
/// could test every other branch of its checkpoint handling and not that one.
#[derive(Debug)]
pub struct ResumeInfo<'a> {
    /// Positions the session now stands at, which is the prefill it does not pay.
    pub tokens_reused: u32,
    /// Bytes of the request text the stored entry covers. The caller tokenizes the
    /// remainder and prefills only that.
    pub text_bytes: u64,
    /// The token path the entry covered, position zero first.
    pub tokens: Vec<TokenId>,
    /// The frontier logits row the payload carried - no decode step is re-run on a
    /// resume, so this is the row the original run produced.
    pub frontier_logits: Logits<'a>,
}

impl<'a> ResumeInfo<'a> {
    /// What a resume recovered, given directly.
    ///
    /// The real path builds one from a payload; this exists so a test double can
    /// report a hit, which is the branch a caller is most likely to get wrong and
    /// least able to reach otherwise.
    #[must_use]
    pub const fn new(
        tokens_reused: u32,
        text_bytes: u64,
        tokens: Vec<TokenId>,
        frontier_logits: Logits<'a>,
    ) -> Self {
        Self {
            tokens_reused,
            text_bytes,
            tokens,
            frontier_logits,
        }
    }
}

/// Checkpointing, layered on [`Model`] rather than added to it.
///
/// [`Model`] is the contract a scheduler holds every model through, and not every
/// model has a disk tier. A separate trait keeps the five methods off the models that
/// do not implement them, and a scheduler that holds a `&mut dyn Checkpointed` can
/// still hand it to anything taking a `&mut dyn Model`.
pub trait Checkpointed: Model {
    /// Turns the tier on over `config.dir`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when sessions are already open - the tier is a
    /// property of the cache the sessions draw pages from, so it is chosen before
    /// anything stands on one - and [`CoreError::Backend`] when the directory cannot
    /// be opened.
    fn kv_enable(&mut self, config: &CkptConfig) -> CoreResult<()>;

    /// Whether a directory has been opened.
    fn kv_checkpoint_enabled(&self) -> bool;

    /// Looks for a stored prefix of `prompt_text` and stands `session` on it.
    ///
    /// `Ok(None)` is a cold start: nothing matched, or what matched was not usable.
    /// Every other failure is an `Err`.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and [`CoreError::Backend`] when the device or
    /// the cache fails while the prefix is being installed.
    fn kv_resume(
        &mut self,
        session: SessionId,
        prompt_text: &[u8],
    ) -> CoreResult<Option<ResumeInfo<'_>>>;

    /// Stores the session's prefix at its frontier, keyed by `prompt_text`.
    ///
    /// `tokens` is the path the caller believes the session stands on; a session that
    /// stands somewhere else is [`SaveOutcome::NotAtFrontier`] rather than a file
    /// describing a position nothing ever reached.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and [`CoreError::Backend`] when the store fails
    /// for a reason that is not a refusal.
    fn kv_save(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        prompt_text: &[u8],
        reason: Reason,
    ) -> CoreResult<SaveOutcome>;

    /// What the cache underneath this model currently holds.
    ///
    /// Defaulted to `None` rather than required: a model with no disk tier has
    /// nothing to report, and a zero would read as "measured, and empty" - which
    /// is the one answer an operator must not be given about a cache that is
    /// actually working.
    fn kv_cache_stats(&self) -> Option<CacheStats> {
        None
    }

    /// Stores every resident session on the way down, and reports how many files
    /// were written.
    ///
    /// `text_of` supplies each session's rendered text and token path, because this
    /// crate has no tokenizer. Per-session failures are logged and skipped: a sweep
    /// that stopped at the first failure would leave the rest of a shutdown's work
    /// undone for no gain.
    fn kv_shutdown_sweep(
        &mut self,
        text_of: &mut dyn FnMut(SessionId) -> Option<(Vec<TokenId>, Vec<u8>)>,
    ) -> usize;
}

// ---------------------------------------------------------------------------
// the prefix scan
// ---------------------------------------------------------------------------

/// What a stored entry has to be before it is worth reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gate {
    /// The floor a stored path has to reach.
    min_tokens: u32,
    /// The family tag this build writes.
    model_id: u8,
    /// The quantization this build was compiled against, or `None` when a
    /// disagreement is tolerated.
    quant_bits: Option<u8>,
    /// Tokens a session here may hold. A path prefilled under a larger context can be
    /// longer than there is room for.
    ctx: u32,
}

/// One stored entry that names a prefix of the request text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Candidate {
    /// The key it is stored under.
    id: SpillId,
    /// Bytes of the request text it covers.
    text_bytes: u64,
    /// Tokens the stored path holds.
    tokens: u32,
}

/// What a scan over the directory found.
///
/// The fields stay private and are read through accessors, because there are
/// now two tiers reading one scan and a field a caller could assign would let
/// one of them answer from a candidate it made up.
#[derive(Debug, Default)]
pub(crate) struct Scan {
    /// The entry to read, if any.
    best: Option<Candidate>,
    /// Entries standing at a key this build owns while holding another family's
    /// bytes. They are unlinked: this build's next save under that key would have to
    /// replace them anyway, and until then they are a file nothing can use.
    foreign: Vec<SpillId>,
}

impl Scan {
    /// The entry worth reading, if any.
    pub(crate) const fn best(&self) -> Option<Candidate> {
        self.best
    }

    /// The entries standing at a key this build owns while holding another
    /// family's bytes, which the caller unlinks.
    pub(crate) fn foreign(&self) -> Vec<SpillId> {
        self.foreign.clone()
    }
}

impl Candidate {
    /// The key this entry is stored under.
    pub(crate) const fn id(&self) -> SpillId {
        self.id
    }

    /// Bytes of the request text it covers.
    pub(crate) const fn text_bytes(&self) -> u64 {
        self.text_bytes
    }
}

/// Picks the stored entry that covers the longest prefix of `text`.
///
/// Confirming a candidate means hashing `text[..entry.text_bytes]` and comparing it
/// against the key the entry is filed under. That comparison is doing two jobs at
/// once: it is how the entry is found, and it is also the proof that the stored text
/// really is a byte prefix of this request, because the key is the hash of exactly
/// those bytes and of nothing else.
///
/// The cost is one sha1 pass per *distinct stored length*, not per prefix position:
/// only lengths some entry claims are hashed. A scheduler that resumes and then saves
/// the same request should hold on to the answer rather than asking twice.
///
/// Longest wins, and more tokens breaks a tie - two entries covering the same bytes
/// can still differ in how far their token path runs, and the longer path is the one
/// that saves more prefill.
pub(crate) fn scan_prefixes(entries: &[SpillEntry], text: &[u8], gate: Gate) -> Scan {
    let mut scan = Scan::default();
    let len = u64::try_from(text.len()).unwrap_or(u64::MAX);
    for entry in entries {
        if entry.text_bytes == 0 || entry.text_bytes > len {
            continue;
        }
        let at = usize::try_from(entry.text_bytes).unwrap_or(usize::MAX);
        let Some(prefix) = text.get(..at) else {
            continue;
        };
        if SpillId::of(prefix) != entry.id {
            continue;
        }
        // From here the entry is *ours by name*: it is filed under a key this
        // request would itself write. Whether its bytes are usable is the next
        // question, and a foreign family's answer is to unlink rather than to skip.
        if entry.model_id != gate.model_id {
            scan.foreign.push(entry.id);
            continue;
        }
        if entry.tokens < gate.min_tokens
            || entry.tokens > gate.ctx
            || entry.ctx_size > gate.ctx
            || gate.quant_bits.is_some_and(|bits| bits != entry.quant_bits)
        {
            // Kept, not unlinked: every one of these is a disagreement with *this*
            // configuration, and the same build started with another one reads the
            // file. The key names the text, not the toggles.
            continue;
        }
        let candidate = Candidate {
            id: entry.id,
            text_bytes: entry.text_bytes,
            tokens: entry.tokens,
        };
        let better = scan.best.is_none_or(|best| {
            (candidate.text_bytes, candidate.tokens) > (best.text_bytes, best.tokens)
        });
        if better {
            scan.best = Some(candidate);
        }
    }
    scan
}

// ---------------------------------------------------------------------------
// the stack's half
// ---------------------------------------------------------------------------

impl HybridStack {
    /// Whether a checkpoint directory has been opened.
    #[must_use]
    pub const fn ckpt_enabled(&self) -> bool {
        self.ckpt.is_some()
    }

    /// Opens the checkpoint tier over `config.dir`, tagged with `identity`.
    ///
    /// The tier is part of the cache rather than a layer above it - the cache is what
    /// walks a path's pages and holds its recurrent state - so turning it on rebuilds
    /// the cache with a spill directory and the byte description a stored payload
    /// needs. That is why it is refused while sessions are open: a session holds page
    /// references into the pool the old cache owns, and the new one has its own.
    ///
    /// `identity` is not a caller's choice. It comes from the crate that loaded the
    /// checkpoint, which is the only party that knows which family wrote the weights
    /// and how they were quantized.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when a session is open or the stack has been torn
    /// down, and [`CoreError::Backend`] when the directory cannot be opened.
    pub fn ckpt_enable(&mut self, config: &CkptConfig, identity: CkptIdentity) -> CoreResult<()> {
        if !self.sessions.is_empty() {
            return Err(CoreError::Unsupported {
                what: "opening a checkpoint directory while sessions are already running",
                backend: "engine",
            });
        }
        let layout = path_layout(&self.shape, &self.geometry, self.opts.ctx, identity);
        let geometry = kv::geometry(&self.shape).with_layout(layout);
        let mut spill = SpillConfig::new(config.dir.clone(), config.options.budget_mb);
        spill.reject_different_quant = config.options.reject_different_quant;
        let (pool_pages, ckpt_slots) = (self.opts.kv_pool_pages, self.opts.kv_ckpt_slots);
        let resident = self.resident.as_mut().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
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
            "checkpoint tier open"
        );
        Ok(())
    }

    /// Stores the session's prefix at its frontier. See [`Checkpointed::kv_save`].
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and [`CoreError::Backend`] when the store fails
    /// for a reason that is not a refusal.
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
        // The frontier gate, and it compares the path rather than only its length: a
        // caller holding the right number of the wrong tokens would otherwise file a
        // conversation's state under another conversation's key.
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
        let node = live.kv().frontier().node();
        let cache = &self.resident()?.cache;
        // An entry already at this key holding bytes this build cannot use is
        // replaced by the write itself: the store checks the existing file's model,
        // quantization and context before it accepts it as a duplicate, and unlinks
        // it when it is not one. So there is no unlink-and-retry to do here.
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
            // The store's answer set is open, and an answer this build cannot name is
            // not a refusal it may quietly report as one: a save that silently became
            // "nothing was written, for one of the reasons you know about" would hide
            // a store doing something new.
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

    /// Stores every open session. See [`Checkpointed::kv_shutdown_sweep`].
    pub fn ckpt_shutdown_sweep(
        &mut self,
        text_of: &mut dyn FnMut(SessionId) -> Option<(Vec<TokenId>, Vec<u8>)>,
    ) -> usize {
        if self.ckpt.is_none() {
            return 0;
        }
        let mut written = 0;
        // A copy of the open order, because `text_of` and the save below both run
        // while the session table would otherwise be borrowed.
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
                    // Logged, never propagated: a session that cannot be stored costs
                    // its next request a prefill, and a shutdown that stopped here
                    // would cost every session behind it the same.
                    tracing::warn!(%session, %error, "could not store a session on the way down");
                }
            }
        }
        tracing::info!(written, "shutdown sweep complete");
        written
    }

    /// Stands `session` on the longest stored prefix of `prompt_text`.
    /// See [`Checkpointed::kv_resume`].
    ///
    /// `Ok(None)` is a cold start and not a refusal: no directory is open, or
    /// nothing stored matched. The caller prefills, exactly as it would have
    /// with no checkpoint tier at all.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and [`CoreError::Backend`] when the device or
    /// the cache fails while the prefix is being installed.
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
        let capacity = self.opts.ctx;
        let vocab = self.geometry.vocab;
        let slots = self.recurrent_slots;
        let cache = Arc::clone(&self.resident()?.cache);

        let scan = scan_prefixes(&cache.spill_entries(), prompt_text, ckpt.gate(capacity));
        for id in scan.foreign {
            match cache.forget_spill(id) {
                Ok(_) => {
                    tracing::info!(entry = %id, "discarded an entry written by another model");
                }
                Err(error) => {
                    tracing::warn!(entry = %id, %error, "could not discard a foreign entry");
                }
            }
        }
        let Some(best) = scan.best else {
            return Ok(None);
        };
        let Some(restored) = cache.restore(best.id, now).map_err(cache_failure)? else {
            return Ok(None);
        };

        let Some(reference) = restored.ckpt else {
            // The pages came back but no slot could hold the recurrent state. That is
            // a transient shortage rather than a bad file, so the entry stays and the
            // request prefills.
            tracing::debug!(entry = %best.id, "no checkpoint slot for a restored prefix");
            return Ok(None);
        };
        let Some(row) = restored.logits else {
            // A payload with no frontier row cannot be resumed from without re-running
            // a step, which is the one thing this tier promises not to do. Nothing
            // this build writes is like that, so the file is one it can never use.
            tracing::warn!(entry = %best.id, "a stored prefix carries no frontier row; discarding");
            let _discarded = cache.forget_spill(best.id);
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
            // Split borrows: the frontier and the recurrent stack belong to the
            // session, and the bytes move through the backend, so both are held at
            // once rather than one accessor at a time.
            let Self {
                backend, sessions, ..
            } = self;
            let live = sessions
                .get_mut(&session)
                .ok_or(CoreError::UnknownSession(session))?;
            live.adopt(restored.frontier, tokens.clone(), capacity)?;
            restore_stack(backend.as_ref(), live, slots, &blob)?;
        }
        // The frontier row goes exactly where a freshly computed one would be, so a
        // caller cannot tell a resumed row from a prefilled one - which is the point.
        {
            let logits = self.resident()?.scratch.logits().slice(0, want_row)?;
            self.backend.copy_h2d(logits, &row)?;
        }

        tracing::info!(
            entry = %best.id,
            tokens = reused,
            text_bytes = best.text_bytes,
            "resumed a stored prefix"
        );
        let logits = self.resident()?.scratch.logits().slice(0, want_row)?;
        Ok(Some(ResumeInfo {
            tokens_reused: reused,
            text_bytes: best.text_bytes,
            tokens,
            frontier_logits: Logits::new(logits, 1, vocab)?,
        }))
    }
}

impl Checkpoint {
    /// What a stored entry has to be for this build, running at `ctx` tokens of
    /// context, to read it.
    ///
    /// `pub(crate)` rather than private because the sparse stack's own
    /// checkpoint tier is a second caller of [`scan_prefixes`], and the
    /// alternative - exposing [`Gate`]'s four fields so it could build one -
    /// would let a caller state a gate that disagrees with the tier it came
    /// from.
    pub(crate) fn gate(self, ctx: u32) -> Gate {
        Gate {
            min_tokens: self.options.min_tokens,
            model_id: self.identity.model_id,
            quant_bits: self
                .options
                .reject_different_quant
                .then_some(self.identity.quant_bits),
            ctx,
        }
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

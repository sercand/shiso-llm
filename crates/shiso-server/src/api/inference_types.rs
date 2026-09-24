//! The wire shapes the informational routes answer with.
//!
//! These are the only serde types in the crate, and that is on purpose: the
//! conversation IR carries no serde derives, because each surface owns its own
//! wire structs and parses into the neutral types rather than deriving them.
//! The routes here have no neutral counterpart - there is no IR type for "the
//! list of models this server serves" - so their wire shape is their only
//! shape, and it lives beside the handlers that build it.

use serde::Serialize;

/// The provider block a model listing carries.
#[derive(Debug, Clone, Serialize)]
pub struct TopProvider {
    /// Tokens one session may hold.
    pub context_length: u32,
    /// The largest completion this server will generate.
    pub max_completion_tokens: u32,
    /// Whether outputs pass through a moderation layer. This server has none.
    pub is_moderated: bool,
}

/// One entry of the model listing.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// The name clients put in their `model` field.
    pub id: String,
    /// The discriminant the listing format requires.
    pub object: &'static str,
    /// Seconds since the epoch. The process start, because a checkpoint on disk
    /// has no creation date a client could act on.
    pub created: u64,
    /// Who published it. This server does not know, and says so by naming
    /// itself.
    pub owned_by: &'static str,
    /// The same string as `id`, which is what the format expects.
    pub name: String,
    /// Tokens one session may hold.
    pub context_length: u32,
    /// Provider-level limits.
    pub top_provider: TopProvider,
    /// Request fields this server honours.
    ///
    /// The list names what will exist once the two adapter tasks land, because
    /// a client reads it to decide what to send and a list that grew later
    /// would have it sending less than the server can do. Nothing here is a
    /// field the IR cannot carry.
    pub supported_parameters: Vec<&'static str>,
}

/// The `/v1/models` body.
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    /// The discriminant the listing format requires.
    pub object: &'static str,
    /// The models. Exactly one, because this binary serves one checkpoint.
    pub data: Vec<ModelInfo>,
}

/// The disk tier's section of the cache report.
///
/// `entries` is a count and not a list: an entry is keyed by the hash of a
/// conversation's token prefix, and listing them would put a fingerprint of
/// other people's conversations in the body of an operator endpoint.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DiskReport {
    /// Whether a disk directory was opened.
    pub enabled: bool,
    /// Where it is, when it exists.
    pub dir: Option<String>,
    /// The budget it was opened with.
    pub budget_bytes: u64,
    /// The prefix floor below which nothing is stored or looked up.
    pub min_tokens: u32,
    /// Whether the numbers below were read from the live store rather than
    /// left at zero because nothing here holds the cache.
    pub measured: bool,
    /// Bytes its files occupy.
    pub total_bytes: u64,
    /// Bytes promised to writes still queued.
    pub reserved_bytes: u64,
    /// Entries a lookup can find.
    pub entries: u32,
    /// Entries accepted and not yet on disk.
    pub pending_entries: u32,
    /// Page files.
    pub pages: u32,
    /// Checkpoints waiting for their pages to be copied off the device.
    pub queued_jobs: u32,
    /// Host bytes copied off the device and not yet written.
    pub staged_bytes: u64,
    /// Entries the budget released.
    pub evictions: u64,
    /// Bytes written since the process started: the drive-wear figure.
    pub bytes_written: u64,
    /// Bytes read back by restores since the process started.
    pub bytes_read: u64,
    /// Entry files written, per reason.
    pub entries_written: std::collections::BTreeMap<&'static str, u64>,
    /// Writes that failed.
    pub write_failures: u64,
    /// Checkpoints dropped because the write queue was full.
    pub jobs_dropped: u64,
    /// Restores that stood a session up from disk.
    pub restores: u64,
    /// Restores that found a file unusable and fell back to prefill.
    pub restore_failures: u64,
    /// Lookups whose restore the cost model judged slower than a prefill.
    pub restores_declined: u64,
    /// Seconds spent restoring.
    pub restore_seconds: f64,
    /// Seconds of prefill those restores replaced, by the prefill estimate.
    pub restore_prefill_seconds: f64,
    /// Read throughput the cost model assumes, bytes per second.
    pub read_bps: f64,
    /// Prefetches started for queued requests.
    pub prefetches: u64,
}

/// What the scheduler has made of the cache.
///
/// Every number here is counted by the coordinator as it runs, which is why
/// they are present even when nothing can read the store itself: they are facts
/// about what this process asked for and got, not about what is on the disk.
#[derive(Debug, Clone, Serialize)]
pub struct EngineCacheReport {
    /// Generations that were given a session.
    pub requests: u64,
    /// Generations that resumed from a stored prefix.
    pub hits: u64,
    /// The share of generations that resumed, to two decimal places.
    pub hit_rate: f64,
    /// Prompt tokens a resume made unnecessary.
    pub tokens_reused: u64,
    /// Prompt tokens that went through a forward pass.
    pub tokens_prefilled: u64,
    /// Frontiers this process offered to the disk tier and had stored.
    pub writes: u64,
    /// Resumes served from checkpoints the in-memory tree held.
    pub resumes_memory: u64,
    /// Resumes served from the disk tier.
    pub resumes_disk: u64,
    /// The prefill rate the resume cost model assumes, seconds per token.
    pub prefill_s_per_token: f64,
    /// Sequences resident now.
    pub resident: u64,
    /// Sequences that may be resident at once.
    pub capacity: u64,
}

/// The `/cache` body.
///
/// Two sections: what is on the disk, and what the scheduler did with it. They
/// are grouped under keys rather than spread across the top level so that a
/// number can be added to one without a client having to guess which tier it
/// belongs to.
#[derive(Debug, Clone, Serialize)]
pub struct CacheReport {
    /// The disk tier.
    pub disk: DiskReport,
    /// The scheduler's own account of it.
    pub engine: EngineCacheReport,
}

/// The request fields the adapters will honour.
pub const SUPPORTED_PARAMETERS: &[&str] = &[
    "tools",
    "tool_choice",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "stop",
    "seed",
    "stream",
    "reasoning_effort",
];

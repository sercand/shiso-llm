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
/// `entries` is a count and not a list, because a count is what the store
/// publishes: an entry is keyed by the hash of a rendered prompt prefix, and
/// listing them would put fragments of other people's conversations in the
/// body of an operator endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct DiskReport {
    /// Whether a spill directory was opened.
    pub enabled: bool,
    /// Where it is, when it exists.
    pub dir: Option<String>,
    /// The budget it was opened with.
    pub budget_bytes: u64,
    /// The prefix floor below which nothing is stored.
    pub min_tokens: u32,
    /// Whether the three numbers below were read from the live store.
    ///
    /// They can only be read by whoever holds the cache, and in a running
    /// server that is the model, on the one thread that owns it - a handler
    /// with a second reference to it could read device-backed state while a
    /// forward pass was writing it. So a server whose tier is on still reports
    /// `measured: false` and zeroes here, and an operator reading the body can
    /// tell "nothing is stored" from "nobody here can count what is stored".
    pub measured: bool,
    /// What it currently holds.
    pub total_bytes: u64,
    /// How many entries a read would accept.
    pub entries: u32,
    /// Entries this process has deleted to stay inside the budget.
    pub evictions: u64,
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
    /// Prefixes this process wrote.
    pub writes: u64,
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

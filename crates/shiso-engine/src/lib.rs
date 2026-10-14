//! shiso-engine - backend-agnostic orchestration: the hybrid recurrent/attention
//! forward pass, the sampler, sessions, and the engine's half of the KV cache.
//!
//! Three things live here, and they are one crate because they meet on every
//! token:
//!
//! - **The forward pass** ([`HybridStack`] and its per-block siblings) drives one
//!   layer at a time through a [`shiso_core::Backend`] and never looks at a device API.
//! - **The sampler** ([`sampling`]) is host code over a logits row, ported stage
//!   for stage so that a stochastic sample advances the generator exactly once and
//!   a greedy one does not touch it at all.
//! - Sessions own the device state a conversation cannot be resumed without: the
//!   recurrent stack, its conv windows, and a frontier into the KV cache.
//! - **The checkpoint tier** ([`ckpt`]) writes a session's whole prefix to disk and
//!   stands one back up from it, so a prompt seen by an earlier process is answered
//!   without being prefilled again. It is off until a caller names a directory.
//!
//! # What this crate does not own
//!
//! The KV cache - pages, refcounts, victim choice, checkpoints, the disk tier - is
//! `shiso-kvstore`'s. This crate derives the geometry a stack's cache needs, moves
//! page bytes between the pool and device memory, and hands each session a
//! frontier; it does not get to decide which span is a victim.
//!
//! # And it does not know which checkpoint it is running
//!
//! Nothing here names a model. A [`HybridStack`] is built from a
//! [`shape::HybridShape`] and weights that are already resident, so which file
//! they came from, what its tensors were called and which token ends a turn are
//! all decisions belonging to a crate above this one. A second checkpoint family
//! with the same block structure is a second crate up there, and no edit down
//! here.
//!
//! [`shiso_kvstore::Frontier`]: shiso_kvstore::Frontier
#![forbid(unsafe_code)]

pub mod attn;
pub mod block_draft;
pub mod blocks;
pub mod ckpt;
mod ckpt_layout;
pub mod dense;
pub mod draft;
pub mod ffn;
pub mod gdn;
mod kv;
pub mod launch;
pub mod layers;
mod media_path;
pub mod moe;
pub mod moe_biased;
/// Speculative decode's decidable half, shared by both stacks.
///
/// A re-export rather than a module of its own, and that is Task 3.3 Step 1's
/// ruling written where a reader meets it. [`sparse::mtp`] is already the half
/// of speculation that is host arithmetic over tokens - bounding `k`, deciding
/// how much of a draft the target agreed with, naming the record a rejection
/// copies back - and its only import is `shiso_core`. Two stacks reading one
/// copy of that is the point; moving the file would edit sixteen call sites
/// across four crates in a lane this phase does not own, and would buy nothing
/// but a shorter path.
pub use sparse::mtp;
mod options;
pub mod ple_rows;
pub mod projection;
pub mod quantize;
#[cfg(test)]
mod recorder;
pub mod rotation;
mod rows;
mod sampling;
mod sampling_probs;
mod sessions;
pub mod shape;
pub mod short_conv;
pub mod sparse;
pub mod ssm;
mod stack;
pub mod trace;
pub mod tune;
pub mod vision;

pub use ckpt::{
    CKPT_BUDGET_POOL_MULTIPLE, CKPT_MIN_TOKENS_DEFAULT, Checkpointed, CkptConfig, CkptOptions,
    ResumeInfo, ResumeSource, SaveOutcome,
};
pub use ckpt_layout::CkptIdentity;
pub use draft::{DraftHead, DraftParts};
pub use options::{EngineOptions, KvCheck, KvFit, PoolCost};
pub use sampling::{argmax, keyed_sample, rng_f32, rng_next, sample_logits, suppress};
pub use sampling_probs::{build_probabilities, sample_probabilities};
/// What the cache holds, re-exported so a model crate can forward it without
/// taking a dependency on the cache crate: the type is part of this crate's own
/// contract the moment `Checkpointed` reports it.
pub use shiso_kvstore::CacheStats;
// The store's own vocabulary, re-exported: a `Reason` is what an entry is filed
// under, an `EntryKey` is its name, a `PersistRefusal` is the store's half of a
// refused save and a `PumpReport` is what a pump did. A model crate that speaks only
// to the checkpoint tier should not have to depend on the store to name them.
pub use shiso_kvstore::{EntryKey, KvCache, PersistRefusal, PumpReport, Reason};
pub use stack::{HybridStack, MAX_DECODE_BATCH, MAX_PREFILL_CHUNK, MAX_PROMPT_RUNS};

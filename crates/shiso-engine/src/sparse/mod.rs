//! The sparse mixture-of-experts hybrid: hyper-connected streams, a routed
//! feed-forward, and attention that reads a selected part of its context.
//!
//! This is a second block family alongside the dense hybrid in [`crate::attn`],
//! [`crate::ffn`] and [`crate::gdn`], not a variation on it. The residual is
//! several streams rather than one; the feed-forward is a router over hundreds
//! of experts plus a shared one; the attention block carries an indexer that
//! decides which key blocks it reads; and one slot may fold in an n-gram
//! embedding block. The recurrent block is the same idea as the dense family's
//! and a different kernel set, because these two families' kernels were written
//! separately and rule one of this port is that a kernel is copied rather than
//! reinterpreted.
//!
//! # Nothing here names a checkpoint family
//!
//! There is no constant table, no config reader and no tensor-name vocabulary
//! below. A [`shape::SparseShape`] is filled in by whichever crate can read a
//! particular checkpoint's config, and a [`weights::SparseWeights`] is built by
//! whichever crate knows what that checkpoint's tensors are called. What is
//! here is the arithmetic that would be the same for a second family with the
//! same blocks and different widths.
//!
//! The kernel *module* keys and entry names in [`kernels`] are the one place
//! that reads otherwise, and they are not a family name: they are the identity
//! of a compiled kernel set, which the vendored kernels themselves carry and
//! which both sides of that seam have to spell the same way. The dense family's
//! kernel set is named the same way in [`crate::launch`], for the same reason.
//!
//! # What runs, and what is decided
//!
//! Most of this module is *decisions*: which kernel a weight is projected by,
//! what grid it launches on, whether a chunk is wide enough to take the tiled
//! attention, when the indexer's selection is worth running, how many blocks a
//! cooperative launch may use. Every one of those is a pure function over a
//! shape and a row count, so every one of them is checkable on a machine with
//! no device - which is the only way they get checked before there is hardware
//! to run them on.
//!
//! [`stack::SparseStack`] is what puts them in order.

pub mod bind;
pub mod capture;
pub mod ckpt;
pub mod draft;
pub mod exec;
pub mod exl3;
pub mod gather;
pub mod headslice;
pub mod hyper;
pub mod indexer;
pub mod inputs;
pub mod island;
pub mod kernels;
pub mod matmul;
pub mod media;
pub mod moe;
pub mod mtp;
pub mod paged;
pub mod passes;
pub mod plan;
pub mod ple;
pub mod qsa;
pub mod recurrent;
pub mod rows;
pub mod run;
pub mod scratch;
pub mod shape;
pub mod stack;
pub mod tap;
pub mod weights;

pub use bind::{Args, Binder, Operand, Role, WeightRef};
pub use exec::{Exec, PassReport};
pub use inputs::{DecodeRow, InputRows, MROPE_AXES, MROPE_PREFIX_BYTES};
pub use island::{Island, IslandContext, IslandKey, LayerChain};
pub use mtp::{
    DraftSource, FailRing, MAX_DRAFT, NGRAM_MAX_MATCH, NGRAM_MIN_MATCH, NgramPolicy, Proposal,
    ProposalId, Rollback, SpecDepth, StepOutcome,
};
pub use plan::{Plan, Step};
pub use ple::{NgramTable, NgramTier, PleGather, PleRows};
pub use rows::{Layout, PtrArray};
pub use shape::{IndexerShape, MoeShape, MropeSections, PleShape, SparseShape};
pub use stack::{
    MAX_DECODE_BATCH, MAX_PREFILL_CHUNK, Pass, SparseOptions, SparseSession, SparseStack,
};
pub use tap::{ResidualTap, TapPoint, TapRecord};
pub use weights::{
    ExpertBank, Indexer, NgramBlock, Recurrent, SparseAttn, SparseBlock, SparseLayer,
    SparseWeights, StreamMix, Weight, WeightDesc,
};

//! The options every model in this crate needs, defined once.
//!
//! A struct with defaults rather than a builder chain: every field has a working
//! default, the set is small, and a chain of setters cannot express "these three
//! are the whole set". The same struct is what a second architecture in this crate
//! would reuse, which is the only reason it is a separate type at all - three
//! fields duplicated across two option structs would be three chances for the
//! defaults to drift apart.
//!
//! # Why the KV numbers are here and the geometry is not
//!
//! `kv_pool_pages` and `kv_ckpt_slots` are counts of *cache units*, so a caller can
//! set them without knowing a model's page size. The byte sizes those counts are
//! multiplied by are derived from the shape profile in [`crate::kv`], because a
//! page that is the wrong size for the layer pattern is not a cache unit at all.

use crate::mtp::SpecDepth;

/// Everything a model in this crate needs to be built, in one place.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct EngineOptions {
    /// Tokens one session may hold, prompt included.
    ///
    /// This is the cache's ceiling, not the model's: these checkpoints have no
    /// positional limit of their own, so a longer context costs pool pages and
    /// nothing else.
    pub ctx: u32,
    /// Pages in the KV pool, which is the cache's whole memory budget expressed
    /// in [`crate::kv::PAGE_TOKENS`]-token units.
    ///
    /// Pages are claimed lazily, so this is a ceiling on what the cache may hold
    /// rather than an allocation: an over-generous number costs a vector of
    /// refcounts.
    pub kv_pool_pages: u32,
    /// Checkpoint slots, which bounds how many resumable positions the cache can
    /// carry at once.
    ///
    /// A slot's blob is allocated when the slot is first filled, so this is the
    /// one number that bounds the recurrent-state tier directly.
    pub kv_ckpt_slots: u32,
    /// How many tokens a step may draft, which is also what sizes the
    /// per-row recurrent records every session allocates.
    ///
    /// [`SpecDepth::NONE`] is the default and is not a placeholder: a
    /// speculative session carries one state matrix per drafted row per
    /// recurrent slot, which on this family is 2 MiB times the depth times
    /// twenty-four, and a build with no draft head would be paying all of it
    /// for records nothing writes.
    pub spec: SpecDepth,
}

impl EngineOptions {
    /// The defaults a server starts with: 4096 tokens, 512 pool pages, 16
    /// checkpoint slots, no spill tier, no speculation.
    ///
    /// The pool default is 512 pages, which at this page size is 4 GiB of address
    /// space that is only touched as sessions fill it.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ctx: 4096,
            kv_pool_pages: 512,
            kv_ckpt_slots: 16,
            spec: SpecDepth::NONE,
        }
    }
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self::new()
    }
}

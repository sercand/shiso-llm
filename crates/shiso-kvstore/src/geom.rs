//! Cache geometry, injected by the engine.
//!
//! The store is model-agnostic: it stores no model constants at all. Page and
//! checkpoint sizes arrive in [`CacheGeometry`], derived by whichever engine is
//! driving the crate, so each architecture hardcodes none of its own numbers here.
//!
//! # Who derives what
//!
//! The engine computes `page_bytes` and `ckpt_bytes` from its own shape profile and
//! passes them in; this crate only validates that what arrived can be built from and
//! then does arithmetic with it. That separation is the whole reason the same cache
//! serves a 24-layer recurrent model and a 48-layer attention model without either
//! appearing here:
//!
//! - `page_bytes` = attention layers x 2 (K and V) x `page_tokens` x
//!   (kv heads x head dim) x bytes per element, plus whatever extra planes
//!   (indexer, draft rows) an architecture keeps page-aligned.
//! - `ckpt_bytes` = recurrent layers x ((conv kernel - 1) x conv-window width +
//!   value heads x head dim^2) x 4, plus the vocab-sized logits row, plus any
//!   per-architecture side matrices. A family with no recurrent layers derives
//!   zero here, and that is a geometry this crate serves: the checkpoint is then
//!   the frontier logits row alone, and what it buys is still the prefill below it.
//!
//! Both are opaque totals here: the cache stores and prices the bytes, and never
//! interprets them.
//!
//! `prefill_s_per_token` is measured, not derived, and `decay_s` is the one scale in
//! this crate with a defensible default (10 minutes), because it is a property of how
//! quickly a session goes stale rather than of a model.

use crate::error::KvError;

/// Sizes and rates the cache is built from, supplied by the engine.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CacheGeometry {
    /// Positions per page. A power of two, because a page is found by dividing.
    pub page_tokens: u32,
    /// Bytes of one page's key/value rows.
    pub page_bytes: u64,
    /// Bytes of one checkpoint's state blob: the recurrent stack's conv windows and
    /// state matrices, exactly as the engine snapshots them.
    ///
    /// Zero is a legitimate value and means the model has no recurrent state; a
    /// checkpoint is then the frontier logits row alone. Such a checkpoint is still
    /// worth admitting, because what it saves is the prefill below it.
    ///
    /// The prompt-final logits row a refresh boundary carries is held alongside the
    /// blob and is not counted here, because the store keeps the two apart -- one is
    /// restored into the recurrent stack, the other is a sampler's answer. An engine
    /// that folds the row into its own blob should say so in `PathLayout` by leaving
    /// `vocab` at zero; the disk tier stores the two as separate sections of an entry.
    pub ckpt_bytes: u64,
    /// Measured prefill seconds per token: the numerator of both utility
    /// formulas. A property of the machine, not of a model.
    pub prefill_s_per_token: f64,
    /// The recency scale both utility formulas decay against.
    pub decay_s: f64,
    /// What a serialized path looks like, when this cache is allowed to write one.
    ///
    /// `None` means the disk tier cannot be opened: the cache stores nothing
    /// and refuses to say a path is restorable, because it has no way to describe the
    /// bytes it would be describing. See [`Self::with_layout`].
    pub layout: Option<PathLayout>,
}

impl CacheGeometry {
    /// The geometry of a cache whose pages hold `page_tokens` positions, whose key
    /// value plane is `page_bytes` per page, and whose checkpoint plane is
    /// `ckpt_bytes` per checkpoint.
    ///
    /// A constructor rather than a struct literal because the type is
    /// `#[non_exhaustive]`: another derived size can be added later without every
    /// caller's literal failing to compile. The three numbers come from the
    /// derivations in the module docs, which the engine owns; this crate only
    /// checks that what arrived can be built from ([`Self::validate`]) and then
    /// prices bytes with it.
    #[must_use]
    pub const fn new(
        page_tokens: u32,
        page_bytes: u64,
        ckpt_bytes: u64,
        prefill_s_per_token: f64,
        decay_s: f64,
    ) -> Self {
        Self {
            page_tokens,
            page_bytes,
            ckpt_bytes,
            prefill_s_per_token,
            decay_s,
            layout: None,
        }
    }

    /// Adds the byte description a stored path's identity is built from.
    ///
    /// A builder rather than a seventh constructor argument because the two numbers
    /// the cache prices bytes with are needed by every cache and the fourteen the
    /// disk tier's identity folds in are needed by exactly one feature.
    #[must_use]
    pub const fn with_layout(mut self, layout: PathLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    /// Rejects a geometry the cache cannot be built from.
    ///
    /// `page_tokens` must be a power of two because the page a position lives in
    /// is found by dividing, and a zero size would make every page free and every
    /// utility calculation divide by zero. Those are construction errors, so they
    /// are reported rather than papered over.
    ///
    /// `ckpt_bytes` is not checked at all: zero means the model has no recurrent
    /// state, which is a shape the cache serves rather than a construction error.
    ///
    /// # Errors
    ///
    /// [`KvError::bad_geometry`] when `page_tokens` is zero or not a power of
    /// two, when `page_bytes` is zero, or when either rate is NaN or not positive.
    pub fn validate(&self) -> Result<(), KvError> {
        if self.page_tokens == 0 {
            return Err(KvError::bad_geometry("page_tokens", "is zero"));
        }
        if !self.page_tokens.is_power_of_two() {
            return Err(KvError::bad_geometry(
                "page_tokens",
                "must be a power of two, because a page is found by dividing a position",
            ));
        }
        if self.page_bytes == 0 {
            return Err(KvError::bad_geometry("page_bytes", "is zero"));
        }
        // `ckpt_bytes` is deliberately *not* checked: zero is the honest answer for a
        // model with no recurrent state, whose checkpoint is the frontier logits row
        // alone. Refusing it would make a purely attentional family re-prefill every
        // prompt from position zero.
        if self.prefill_s_per_token.is_nan() || self.prefill_s_per_token <= 0.0 {
            return Err(KvError::bad_geometry(
                "prefill_s_per_token",
                "must be positive, because it is the numerator of every utility",
            ));
        }
        if self.decay_s.is_nan() || self.decay_s <= 0.0 {
            return Err(KvError::bad_geometry(
                "decay_s",
                "must be positive, because it is a divisor",
            ));
        }
        Ok(())
    }

    /// The page covering `position`.
    #[must_use]
    pub const fn page_of(&self, position: u32) -> u32 {
        position / self.page_tokens
    }

    /// The first position inside the page `page` covers.
    #[must_use]
    pub const fn first_position(&self, page: u32) -> u32 {
        page * self.page_tokens
    }

    /// Whether `position` starts a page.
    ///
    /// This is the copy-on-write test: a checkpoint at a page start resumes
    /// without touching a shared page, while one inside a page needs a private
    /// copy of the page its last row sits in.
    #[must_use]
    pub const fn is_page_start(&self, position: u32) -> bool {
        position.is_multiple_of(self.page_tokens)
    }

    /// Pages needed to cover positions `[0, n_pos)`.
    #[must_use]
    pub const fn pages_for(&self, n_pos: u32) -> u32 {
        let per = self.page_tokens;
        n_pos.div_ceil(per)
    }

    /// Bytes for `pages` pages, saturating so a large pool cannot wrap.
    #[must_use]
    pub const fn page_bytes_total(&self, pages: u32) -> u64 {
        // The one `as` in the crate, and the only place it is unavoidable: `u32` to `u64`
        // is infallible, and a `const fn` cannot call `From`.
        self.page_bytes.saturating_mul(pages as u64)
    }

    /// The mebibyte figure a page costs: the denominator of node utility.
    #[must_use]
    pub fn page_mib(&self) -> f64 {
        mib(self.page_bytes)
    }

    /// The mebibyte figure a checkpoint's state blob costs.
    ///
    /// This is a size and nothing else, so it is zero for a model with no
    /// recurrent state. Checkpoint utility divides by it everywhere that number
    /// is positive and substitutes a constant where it is not, because a
    /// division by zero would rank every stateless checkpoint as infinitely
    /// valuable; the substitution belongs with the formula, in [`crate::evict`],
    /// rather than here, where it would make this method report a cost the
    /// geometry does not have.
    #[must_use]
    pub fn ckpt_mib(&self) -> f64 {
        mib(self.ckpt_bytes)
    }
}

/// Bytes expressed in mebibytes, the unit every utility formula divides by.
#[must_use]
pub(crate) fn mib(bytes: u64) -> f64 {
    f64::from(crate::ids::to_u32_wide(bytes)) / (1024.0 * 1024.0)
}

/// What a serialized path is made of, in the engine's own units.
///
/// [`CacheGeometry`] carries the two totals this crate prices bytes with. A path
/// written to disk has to say more than that: a reader re-adopting those bytes needs
/// to know they came from the same *layout*, not merely from a cache with the same
/// page size, or it will happily load another architecture's key/value rows and
/// answer confidently with them. This is the field set that check needs, kept in its
/// own type so the numbers the cache itself uses stay exactly two.
///
/// Every field is an element count or a toggle the engine derives from its shape
/// profile. This crate never computes any of them and never interprets them; it
/// writes them into a header and compares them on the way back.
///
/// The field set is the on-disk word set, so it is deliberately *not*
/// `#[non_exhaustive]`, for the same reason [`crate::PathHeader`] is not: a field
/// cannot be added here without a version bump that every reader in the directory
/// would fail, and the engine is the only party that can state one - a constructor or
/// an unbuildable literal would hide that a caller left one of the words unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathLayout {
    /// Which build's geometry a stored path assumes. A reader that does not
    /// recognize the value refuses the file rather than guessing.
    pub model_id: u8,
    /// Quantization width the page bytes were written with. Two widths are not
    /// interchangeable, so this is checked before a single byte is read.
    pub quant_bits: u8,
    /// Context size the path was prefilled under, which a reader with a smaller one
    /// cannot use.
    pub ctx_size: u32,
    /// Bytes of one position's row across every key/value buffer.
    pub kv_width: u32,
    /// Layers the page section interleaves.
    pub n_layers: u32,
    /// Layers that contribute key/value rows.
    pub n_attn_layers: u32,
    /// Layers that contribute recurrent state.
    pub n_gdn_layers: u32,
    /// Vocabulary size, which is the logits row's length in `f32`s.
    pub vocab: u32,
    /// Elements of one recurrent layer's convolution window.
    pub gdn_conv_elements: u32,
    /// Elements of one recurrent layer's state matrix.
    pub gdn_state_elements: u32,
    /// Elements of a dilated side-window block; zero when the architecture has none.
    pub ple_window_elements: u32,
    /// Width of the side plane a draft head writes; zero when there is none.
    pub hc_width: u32,
    /// Whether the page section carries indexer caches. An operator toggle, so a
    /// disagreement is a configuration difference rather than a foreign file.
    pub indexer: bool,
    /// Whether the page section carries a draft head. Also an operator toggle.
    pub draft_head: bool,
}

/// Counting the pages a span needs, including the last partial one.
#[cfg(test)]
mod tests {
    use super::CacheGeometry;

    /// task-02/task-06's numbers for qwen35: 256 tokens per page, 8 MiB of f16
    /// K/V per page, 51.2 MiB of recurrent state plus logits row per checkpoint.
    fn geometry(page_tokens: u32) -> CacheGeometry {
        CacheGeometry::new(page_tokens, 8_388_608, 53_684_224, 0.0014, 600.0)
    }

    /// A span that ends exactly on a page boundary does not need another page.
    #[test]
    fn pages_are_counted_up_to_the_last_partial_one() {
        let geometry = geometry(256);
        assert_eq!(geometry.pages_for(0), 0);
        assert_eq!(geometry.pages_for(1), 1);
        assert_eq!(geometry.pages_for(256), 1);
        assert_eq!(geometry.pages_for(257), 2);
        assert_eq!(geometry.pages_for(700), 3);
    }

    #[test]
    fn a_page_is_found_by_dividing_and_the_first_position_by_multiplying() {
        let geometry = geometry(256);
        assert_eq!(geometry.page_of(0), 0);
        assert_eq!(geometry.page_of(255), 0);
        assert_eq!(geometry.page_of(256), 1);
        assert_eq!(geometry.page_of(700), 2);
        assert_eq!(geometry.first_position(2), 512);
        assert!(geometry.is_page_start(512));
        assert!(!geometry.is_page_start(513));
    }

    #[test]
    fn sizes_are_priced_in_mebibytes_because_that_is_the_denominator() {
        let geometry = geometry(256);
        assert!((geometry.page_mib() - 8.0).abs() < 1e-9);
        assert!((geometry.ckpt_mib() - 51.2).abs() < 0.01);
        assert_eq!(geometry.page_bytes_total(2), 16_777_216);
    }

    #[test]
    fn a_geometry_that_cannot_be_built_from_is_refused() {
        assert!(geometry(0).validate().is_err(), "no page size, no pages");
        let mut ragged = geometry(256);
        ragged.page_tokens = 300;
        assert!(ragged.validate().is_err());
        let mut no_pages = geometry(256);
        no_pages.page_bytes = 0;
        assert!(no_pages.validate().is_err());
        assert!(geometry(256).validate().is_ok());
    }

    #[test]
    fn a_stateless_geometry_is_accepted() {
        let mut no_state = geometry(256);
        no_state.ckpt_bytes = 0;
        assert!(
            no_state.validate().is_ok(),
            "a model with no recurrent state has nothing to put in a blob, and its \
             checkpoint -- the frontier logits row alone -- is still worth admitting"
        );
    }
}

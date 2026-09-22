//! What a prompt carries when some of its tokens are pieces of a picture.
//!
//! # Why this is here and not in the crate that splices it
//!
//! The rows an encoder produces have to travel from whatever ran the encoder,
//! across the one contract every loaded model answers, to whatever runs the
//! forward pass. That is three crates, and the middle one is this one. A block
//! of embedding rows, the grids that describe them and the placeholder ids they
//! replace are the same three facts for any tower and any checkpoint family, so
//! they are stated once, here, in the vocabulary every layer already speaks.
//!
//! Nothing in this module names a model, a tower or a container. It does not
//! know what a patch is, it never sees a pixel, and it computes no geometry: it
//! carries a row block and the counts that make the block addressable, and it
//! refuses the one disagreement that would otherwise pass silently.
//!
//! # The disagreement this exists to catch
//!
//! [`VisionPrefill::new`] refuses a block whose row count is not the sum of the
//! items' placeholder counts. Nothing downstream re-expands a placeholder run,
//! so the count the items carry is the count the rendered prompt used, and a
//! mismatch between the two is the one media failure that produces a fluent
//! answer rather than an error: every buffer is the right size, every kernel
//! launches, and the answer is about a rearranged picture.
//!
//! # Why the capability is a trait and not a method every model refuses
//!
//! [`MediaModel`] is a separate contract reached through [`Model::media_mut`],
//! which answers `None` for a model with no media path at all. A pair of
//! defaulted methods on [`Model`] returning "unsupported" would put the same
//! question at every call site and answer it one prefill too late — after a
//! session was opened and a prompt was read. Here the question is a type: a
//! caller that holds `Some` holds a model that splices, and a caller that holds
//! `None` knows it at admission, before anything has been paid for.

use crate::buffer::DeviceView;
use crate::error::{CoreError, CoreResult};
use crate::ids::{SessionId, TokenId};
use crate::model::{Logits, Model};

/// Bytes one single-precision element occupies, which is how a row block's
/// height is recovered from its length.
const F32_BYTES: u64 = 4;

/// One media item's post-merge grid.
///
/// Post-merge is the whole of it: the grid here counts the tokens the item
/// occupies in the prompt, not the patches a tower consumed. The two differ by
/// the square of the merge size, and using the pre-merge grid would give every
/// image several times the position budget it was trained with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionGrid {
    t_len: u32,
    gh: u32,
    gw: u32,
}

impl VisionGrid {
    /// A grid of `t_len` temporal groups over `gh` by `gw` merged positions.
    ///
    /// A zero extent is clamped to one rather than refused. The caller that
    /// passes one has a broken item and this is not where that is diagnosed;
    /// what matters here is that a walk over the grid terminates and divides by
    /// nothing.
    #[must_use]
    pub const fn new(t_len: u32, gh: u32, gw: u32) -> Self {
        Self {
            t_len: if t_len == 0 { 1 } else { t_len },
            gh: if gh == 0 { 1 } else { gh },
            gw: if gw == 0 { 1 } else { gw },
        }
    }

    /// Temporal groups. One for a still image.
    #[must_use]
    pub const fn t_len(&self) -> u32 {
        self.t_len
    }

    /// Merged rows per temporal group.
    #[must_use]
    pub const fn grid_h(&self) -> u32 {
        self.gh
    }

    /// Merged columns per temporal group.
    #[must_use]
    pub const fn grid_w(&self) -> u32 {
        self.gw
    }

    /// Placeholder tokens this item occupies, which is also the number of
    /// embedding rows it produced.
    #[must_use]
    pub const fn pad_count(&self) -> u32 {
        self.t_len.saturating_mul(self.gh).saturating_mul(self.gw)
    }
}

/// The two placeholder token ids a rendered prompt can carry.
///
/// Both are optional because a vocabulary may declare one and not the other, and
/// a model with neither has no media path at all. They are one type rather than
/// two arguments so that no call site can pass them in the wrong order or match
/// only the first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PadTokens {
    image: Option<TokenId>,
    video: Option<TokenId>,
}

impl PadTokens {
    /// The pair as a vocabulary reports it.
    #[must_use]
    pub const fn new(image: Option<TokenId>, video: Option<TokenId>) -> Self {
        Self { image, video }
    }

    /// The still-image placeholder.
    #[must_use]
    pub const fn image(&self) -> Option<TokenId> {
        self.image
    }

    /// The clip placeholder.
    #[must_use]
    pub const fn video(&self) -> Option<TokenId> {
        self.video
    }

    /// Whether either placeholder exists, which is whether a prompt can carry
    /// media at all.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.image.is_some() || self.video.is_some()
    }

    /// Whether `token` is a placeholder. Both are matched, always.
    ///
    /// Matching only the still-image id walks a clip's run one text token at a
    /// time: every position after it shifts, no encoder row is spliced over any
    /// of its tokens, and the model answers about the placeholder embedding it
    /// was left holding.
    #[must_use]
    pub fn matches(&self, token: TokenId) -> bool {
        Some(token) == self.image || Some(token) == self.video
    }
}

/// One request's encoder output, with the grids that describe it.
///
/// Built by whatever ran the tower and handed to a prefill call. The rows are
/// borrowed rather than owned: they live in the encoder's own buffer for the
/// length of the prefill and are not copied on the way through.
#[derive(Debug, Clone, Copy)]
pub struct VisionPrefill<'a> {
    rows: DeviceView<'a>,
    grids: &'a [VisionGrid],
    pads: PadTokens,
    row_count: u32,
}

impl<'a> VisionPrefill<'a> {
    /// Binds an encoder output to the items it came from.
    ///
    /// `rows` is `[sum of placeholder counts, hidden]` single-precision, in the
    /// order the items appear in the prompt. `hidden` is the LANGUAGE model's
    /// residual width, which is what a tower's merger projects to.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the block's row count is not the sum of
    /// the items' placeholder counts — the mismatch this whole path exists to
    /// catch, and the one a caller maps to a bad request — and
    /// [`CoreError::Unsupported`] when items were supplied but the vocabulary
    /// declares no placeholder token, which would leave every row unspliced.
    pub fn new(
        rows: DeviceView<'a>,
        grids: &'a [VisionGrid],
        pads: PadTokens,
        hidden: u32,
    ) -> CoreResult<Self> {
        let expected: u64 = grids
            .iter()
            .map(|grid| u64::from(grid.pad_count()))
            .sum::<u64>();
        let row_bytes = u64::from(hidden).saturating_mul(F32_BYTES);
        // `hidden` of zero is a caller error rather than a division to perform,
        // and it reaches the same refusal below as any other wrong row count:
        // no row is a whole number of zero bytes, so the count is zero and the
        // mismatch names it.
        let got = rows.len().checked_div(row_bytes).unwrap_or(0);
        if got != expected {
            return Err(CoreError::ShapeMismatch {
                field: "vision embedding rows",
                expected,
                got,
            });
        }
        if !grids.is_empty() && !pads.any() {
            return Err(CoreError::Unsupported {
                what: "media in a prompt whose vocabulary declares no placeholder token",
                backend: "media",
            });
        }
        Ok(Self {
            rows,
            grids,
            pads,
            row_count: u32::try_from(expected).unwrap_or(u32::MAX),
        })
    }

    /// The encoder rows, whole.
    #[must_use]
    pub const fn rows(&self) -> DeviceView<'a> {
        self.rows
    }

    /// Rows in the block, which is also the placeholder count of the prompt.
    #[must_use]
    pub const fn row_count(&self) -> u32 {
        self.row_count
    }

    /// The items, in prompt order.
    #[must_use]
    pub const fn grids(&self) -> &'a [VisionGrid] {
        self.grids
    }

    /// The placeholder ids this prompt was rendered with.
    #[must_use]
    pub const fn pads(&self) -> PadTokens {
        self.pads
    }

    /// Whether this carries anything at all. An empty one is not an error: a
    /// caller that builds one request shape for every request gets the text
    /// path back by handing over no items.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.grids.is_empty()
    }
}

/// A loaded model that can splice encoder rows into a prompt.
///
/// Separate from [`Model`] because it is a capability a family either has or
/// does not, and reached only through [`Model::media_mut`], so a model without
/// it cannot be asked. A request with no media takes [`Model::prefill_chunk`]
/// and [`Model::prefill_last`] rather than passing an empty block through here:
/// the text path is pinned to a recorded vector and is not a degenerate case of
/// this one.
///
/// # Why there is one method and not a chunk/last pair
///
/// A placeholder run's positions are not the linear ones — after a single image
/// the running position is already a different number — and which encoder row
/// the splice has reached is part of the same running state. A prompt handed
/// over in pieces would need that state to survive between calls, and an engine
/// that rebuilds it per call would restart both, silently: the request succeeds,
/// the answer is fluent, and the picture in it is rearranged. So the contract
/// takes the whole prompt in one call, and a caller cannot express the split
/// that would be wrong. Splitting the work into device-sized pieces is the
/// implementation's business and happens below this line.
pub trait MediaModel: Model {
    /// Advances a session by the WHOLE of a media-carrying prompt, splicing
    /// `media` over the placeholder positions, and returns the last-token row.
    ///
    /// # Errors
    ///
    /// As [`Model::prefill_last`], plus [`CoreError::ShapeMismatch`] when the
    /// prompt's placeholder run and the block's rows disagree — the count the
    /// items carry is the count the rendered prompt used, and nothing below
    /// re-expands a run — and [`CoreError::Unsupported`] when the block names a
    /// placeholder the vocabulary does not declare.
    fn prefill_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>>;
}

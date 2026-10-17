//! Token layout of one sample: per-mode ids, the order the single-stream
//! tail reads the modes in, and the padded prompt length.
//!
//! A FLUX.3 sample is three token groups ("modes"): the noisy `image`, the
//! reference `image_cond`, and the `text`. The per-mode blocks see each
//! group alone; the single-stream tail sees them concatenated per sample in
//! mode-name order. Rotary ids are four-axis: image tokens carry their
//! latent row and column on the middle axes, scaled by one half; references
//! carry a negative index on the last axis so the model can tell them apart;
//! text tokens carry only their position on the last axis.

use std::ops::Range;

use shiso_vision::generator::ImageGenError;

/// Scale on the two spatial rotary axes of image tokens.
const IMAGE_ROPE_SCALE: f32 = 0.5;
/// Text lengths are padded to a multiple of this.
const PAD_MULTIPLE: usize = 80;
/// The longest padded prompt the text encoder reads.
const MAX_TEXT_TOKENS: usize = 4096;

/// One token group of a sample. Declaration order is mode-name order
/// (`image` < `image_cond` < `text`), the order the tail concatenates in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Mode {
    /// The noisy image tokens being denoised.
    Image,
    /// The VAE-encoded reference images.
    ImageCond,
    /// The text conditioning.
    Text,
}

/// Where each mode's tokens sit inside its own per-mode tensor, for one
/// sample. An empty range is a mode the sample does not have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Pack {
    /// Rows of the image tensor this sample owns.
    pub image: Range<usize>,
    /// Rows of the reference tensor this sample owns (empty without references).
    pub image_cond: Range<usize>,
    /// Rows of the text tensor this sample owns.
    pub text: Range<usize>,
}

impl Pack {
    /// The pack of a single sample whose per-mode tensors hold exactly its
    /// own `image`, `image_cond` and `text` rows.
    #[must_use]
    pub(crate) fn single(image: usize, image_cond: usize, text: usize) -> Self {
        Self {
            image: 0..image,
            image_cond: 0..image_cond,
            text: 0..text,
        }
    }

    /// The non-empty modes in the order the single-stream tail concatenates
    /// them (sorted by mode name), each with its rows in its own tensor. A
    /// mode without tokens is left out: the reference runs a phantom token
    /// through its projections but never attends it.
    #[must_use]
    pub(crate) fn single_stream_order(&self) -> Vec<(Mode, Range<usize>)> {
        [
            (Mode::Image, &self.image),
            (Mode::ImageCond, &self.image_cond),
            (Mode::Text, &self.text),
        ]
        .into_iter()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(mode, rows)| (mode, rows.clone()))
        .collect()
    }

    /// Tokens in the concatenated single-stream sequence.
    #[must_use]
    pub(crate) fn stream_len(&self) -> usize {
        self.image.len() + self.image_cond.len() + self.text.len()
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "latent grid coordinates are far below f32's exact-integer range"
)]
fn grid_ids(h: usize, w: usize, l: f32) -> Vec<[f32; 4]> {
    (0..h)
        .flat_map(|row| {
            (0..w).map(move |col| {
                [
                    0.0,
                    row as f32 * IMAGE_ROPE_SCALE,
                    col as f32 * IMAGE_ROPE_SCALE,
                    l,
                ]
            })
        })
        .collect()
}

/// Rotary ids of an `h` x `w` latent grid, row-major: `(0, row/2, col/2, 0)`.
#[must_use]
pub(crate) fn image_ids(h: usize, w: usize) -> Vec<[f32; 4]> {
    grid_ids(h, w, 0.0)
}

/// Rotary ids of reference number `index`: the image grid's, with the last
/// axis `-(1 + index)`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "a reference index is at most ten"
)]
pub(crate) fn reference_ids(h: usize, w: usize, index: usize) -> Vec<[f32; 4]> {
    grid_ids(h, w, -((1 + index) as f32))
}

/// Rotary ids of `len` text tokens, pad rows included: `(0, 0, 0, position)`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "text positions stay below 4096, exact in f32"
)]
pub(crate) fn text_ids(len: usize) -> Vec<[f32; 4]> {
    (0..len).map(|pos| [0.0, 0.0, 0.0, pos as f32]).collect()
}

/// The text length `real` prompt tokens are padded to: the next multiple of
/// 80, capped at 4096.
///
/// # Errors
/// [`ImageGenError::EmptyPrompt`] for no tokens, and
/// [`ImageGenError::PromptTooLong`] over 4096: a longer prompt is refused,
/// never cut, since cutting would drop the assistant opener the model is
/// conditioned on.
pub(crate) fn padded_len(real: usize) -> Result<usize, ImageGenError> {
    if real == 0 {
        return Err(ImageGenError::EmptyPrompt);
    }
    if real > MAX_TEXT_TOKENS {
        return Err(ImageGenError::PromptTooLong {
            tokens: u32::try_from(real).unwrap_or(u32::MAX),
            max: u32::try_from(MAX_TEXT_TOKENS).unwrap_or(u32::MAX),
        });
    }
    Ok(real.next_multiple_of(PAD_MULTIPLE).min(MAX_TEXT_TOKENS))
}

//! Reference-image preparation for Qwen-Image edits.
//!
//! A reference reaches the model twice: the vision tower reads it as
//! picture context for the text encoder, and the VAE encodes it into the
//! latent tokens that stand in the transformer's sequence. Both read the
//! same resize, so each image slot of the encoder (a 32-pixel tile) lines up
//! with the 2x2 group of latent tokens (16-pixel tiles) that replaces it.
//!
//! The condition size follows the reference: the area of 1024x1024 at the
//! picture's own aspect ratio, each side rounded (ties to even) to a
//! multiple of 32, whatever the picture's own size. The tower's copy has
//! its alpha composited over white, as the checkpoint was trained; the
//! VAE's copy keeps all four channels.
//!
//! The resampling filter is the `image` crate's Lanczos3 over straight
//! alpha: the reference's kernel in a different implementation (which
//! resamples premultiplied), so resized pixels can differ in the last bits,
//! and at the edge of a transparent region by more.

use image::imageops::FilterType;
use image::{ImageDecoder, ImageReader, RgbaImage};
use sha2::{Digest, Sha256};
use shiso_vision::generator::{ImageGenError, InputReferenceImage};

use super::vae::{LatentGrid, SPATIAL_FACTOR};

/// The most references one edit carries (the model card's limit).
pub(crate) const MAX_REFERENCES: usize = 10;
/// Pixels a prepared reference holds, before rounding to the grid.
pub(crate) const CONDITION_AREA: u64 = 1024 * 1024;
/// Condition sides are multiples of this: one encoder image slot.
const SLOT: u32 = 32;
/// The shortest side a reference may have.
const MIN_SOURCE_SIDE: u32 = 16;
/// The sides the model serves; an omitted output size is clamped to them.
const SERVED_SIDES: std::ops::RangeInclusive<u32> = 256..=2048;

/// The `(width, height)` a `width` x `height` reference is conditioned at.
///
/// # Errors
/// [`ImageGenError::ReferenceImage`] for a side under 16 pixels, or an
/// aspect ratio so extreme that a side rounds to no slot.
#[expect(clippy::cast_precision_loss, reason = "the area is 2^20, exact in f64")]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded non-negative slot counts of at most a few thousand"
)]
pub(crate) fn condition_size(width: u32, height: u32) -> Result<(u32, u32), ImageGenError> {
    if width < MIN_SOURCE_SIDE || height < MIN_SOURCE_SIDE {
        return Err(ImageGenError::ReferenceImage(format!(
            "a reference image is {width}x{height}; every side must be at least \
             {MIN_SOURCE_SIDE} pixels"
        )));
    }
    let ratio = f64::from(width) / f64::from(height);
    let w = (CONDITION_AREA as f64 * ratio).sqrt();
    let h = w / ratio;
    // Python's `round` takes ties to even, and so must this.
    let slots = |side: f64| (side / f64::from(SLOT)).round_ties_even() as u32;
    let (cols, rows) = (slots(w), slots(h));
    if cols == 0 || rows == 0 {
        return Err(ImageGenError::ReferenceImage(format!(
            "a {width}x{height} reference image is too elongated: at the model's condition \
             area its short side is under one {SLOT}-pixel slot"
        )));
    }
    Ok((cols * SLOT, rows * SLOT))
}

/// One reference at its condition size, in both forms the model reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreparedReference {
    /// Condition width in pixels.
    pub width: u32,
    /// Condition height in pixels.
    pub height: u32,
    /// RGBA8, row-major (the VAE's input).
    pub rgba: Vec<u8>,
    /// RGB8, row-major, alpha composited over white (the tower's input).
    pub rgb: Vec<u8>,
}

impl PreparedReference {
    /// The encoder's image slots, `(rows, columns)`.
    pub(crate) fn slots(&self) -> (usize, usize) {
        ((self.height / SLOT) as usize, (self.width / SLOT) as usize)
    }

    /// The latent grid the VAE encodes this reference to.
    pub(crate) fn latent_grid(&self) -> LatentGrid {
        LatentGrid {
            h: self.height as usize / SPATIAL_FACTOR,
            w: self.width as usize / SPATIAL_FACTOR,
        }
    }
}

/// Refuses an edit of no references or more than `max`.
///
/// # Errors
/// [`ImageGenError::ReferenceImage`] naming the count and the limit.
pub(crate) fn check_count(n: usize, max: usize) -> Result<(), ImageGenError> {
    if n == 0 {
        return Err(ImageGenError::ReferenceImage(
            "an edit needs at least one reference image".to_string(),
        ));
    }
    if n > max {
        return Err(ImageGenError::ReferenceImage(format!(
            "{n} reference images were given; at most {max} are accepted"
        )));
    }
    Ok(())
}

/// Decodes reference number `index` to RGBA with its EXIF orientation
/// applied.
fn decode(reference: &InputReferenceImage, index: usize) -> Result<RgbaImage, ImageGenError> {
    let fail = |e: image::ImageError| {
        ImageGenError::ReferenceImage(format!(
            "reference image {} does not decode: {e}",
            index + 1
        ))
    };
    let mut decoder = ImageReader::new(std::io::Cursor::new(&reference.bytes))
        .with_guessed_format()
        .map_err(|e| {
            ImageGenError::ReferenceImage(format!(
                "reference image {} is unreadable: {e}",
                index + 1
            ))
        })?
        .into_decoder()
        .map_err(fail)?;
    let orientation = decoder.orientation().map_err(fail)?;
    let mut picture = image::DynamicImage::from_decoder(decoder).map_err(fail)?;
    picture.apply_orientation(orientation);
    Ok(picture.to_rgba8())
}

/// `channel` at coverage `alpha` over white, rounded.
fn over_white(channel: u8, alpha: u8) -> u8 {
    let (c, a) = (u32::from(channel), u32::from(alpha));
    u8::try_from((c * a + 255 * (255 - a) + 127) / 255).unwrap_or(u8::MAX)
}

/// Decodes and resizes reference number `index` (0-based, for messages).
///
/// # Errors
/// [`ImageGenError::ReferenceImage`] when it does not decode or
/// [`condition_size`] refuses its shape.
pub(crate) fn prepare(
    reference: &InputReferenceImage,
    index: usize,
) -> Result<PreparedReference, ImageGenError> {
    let picture = decode(reference, index)?;
    let (width, height) = condition_size(picture.width(), picture.height())?;
    let resized = if picture.dimensions() == (width, height) {
        picture
    } else {
        image::imageops::resize(&picture, width, height, FilterType::Lanczos3)
    };
    let rgba = resized.into_raw();
    let rgb = rgba
        .chunks_exact(4)
        .flat_map(|p| [0, 1, 2].map(|c| over_white(p[c], p[3])))
        .collect();
    Ok(PreparedReference {
        width,
        height,
        rgba,
        rgb,
    })
}

/// The size an edit takes when its caller names none: the last reference's
/// condition size, each side clamped to what the model serves.
///
/// # Errors
/// [`ImageGenError::ReferenceImage`] for no references, too many, or a last
/// one that does not decode or has no condition size.
pub(crate) fn default_size(
    references: &[InputReferenceImage],
) -> Result<(u32, u32), ImageGenError> {
    check_count(references.len(), MAX_REFERENCES)?;
    let index = references.len() - 1;
    let picture = decode(&references[index], index)?;
    let (w, h) = condition_size(picture.width(), picture.height())?;
    let clamp = |side: u32| side.clamp(*SERVED_SIDES.start(), *SERVED_SIDES.end());
    Ok((clamp(w), clamp(h)))
}

/// A digest of the prepared references, sizes and order included: what an
/// edit's encoder features depend on beside the prompt.
pub(crate) fn references_hash(prepared: &[&PreparedReference]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for reference in prepared {
        hasher.update(reference.width.to_le_bytes());
        hasher.update(reference.height.to_le_bytes());
        hasher.update(&reference.rgba);
    }
    hasher.finalize().into()
}

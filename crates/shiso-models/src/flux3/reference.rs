//! Reference-image preparation for FLUX.3 edits.
//!
//! A reference reaches the transformer as VAE tokens, so its pixels must sit
//! on the 16-pixel latent grid and stay within what the model was trained to
//! read. One reference is fitted under 4 MiB pixels and reflect-padded to a
//! canvas on the grid (the output is cropped back to the fitted box when the
//! caller asks for the canvas size); several are each fitted under 1 MP and
//! centre-cropped to the grid, so ten of them stay affordable.
//!
//! The arithmetic follows the reference implementation exactly where it can
//! be pinned (the canvas search, padding by index folding, the crop box);
//! the resampling filter is the `image` crate's Lanczos3, which is the same
//! kernel as the reference's but a different implementation, so resized pixels
//! can differ in the last bit or two.

use image::imageops::FilterType;
use image::{ImageDecoder, ImageReader, RgbImage};
use shiso_vision::generator::{ImageGenError, InputReferenceImage};

/// Pixels the padded canvas of a single reference may hold.
const MATCH_MAX_PIXELS: u64 = 4 * 1024 * 1024;
/// Pixels each reference of a multi-reference edit may hold.
const MULTI_REFERENCE_MAX_PIXELS: u64 = 1024 * 1024;
/// The latent grid: an 8x VAE times the 2x2 patch.
const GRID: u32 = 16;
/// The smallest canvas side.
const MIN_SIDE: u32 = 64;
/// The largest canvas side.
const MAX_SIDE: u32 = 4096;
/// The most references one edit carries.
pub(crate) const MAX_REFERENCES: usize = 10;

fn ceil_grid(v: u32) -> u32 {
    v.div_ceil(GRID) * GRID
}

/// The `(fitted size, canvas)` of a single `width` x `height` reference: the
/// canvas is the fitted size rounded up to 16 within `64..=4096`, and holds
/// at most 4 MiB pixels (the scale shrinks by half a percent until it does).
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "image sides are at most a few thousand, exact in f64"
)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the fitted sides are rounded positive values of at most MAX_SIDE"
)]
pub(crate) fn reference_canvas(width: u32, height: u32) -> ((u32, u32), (u32, u32)) {
    let (w, h) = (f64::from(width), f64::from(height));
    let mut scale = 1.0f64
        .min((MATCH_MAX_PIXELS as f64 / (w * h)).sqrt())
        .min(f64::from(MAX_SIDE) / w.max(h));
    loop {
        // Python's `round` takes ties to even, and so must this.
        let fit = (
            ((w * scale).round_ties_even() as u32).max(1),
            ((h * scale).round_ties_even() as u32).max(1),
        );
        let canvas = (
            ceil_grid(fit.0).clamp(MIN_SIDE, MAX_SIDE),
            ceil_grid(fit.1).clamp(MIN_SIDE, MAX_SIDE),
        );
        if u64::from(canvas.0) * u64::from(canvas.1) <= MATCH_MAX_PIXELS {
            return (fit, canvas);
        }
        scale *= 0.995;
    }
}

/// The source index a padded coordinate `i` reads from a side of `n`
/// pixels, folding as often as the pad is wide: symmetric repeats the edge
/// pixel (period `2n`), reflect does not (period `2(n-1)`).
fn fold(i: i64, n: i64, reflect: bool) -> usize {
    let period = if reflect { 2 * (n - 1) } else { 2 * n };
    let r = i.rem_euclid(period);
    let index = if r < n {
        r
    } else {
        period - r - i64::from(!reflect)
    };
    usize::try_from(index).unwrap_or(0)
}

/// A single reference prepared for the canvas: fitted by
/// [`reference_canvas`], then centred on the canvas and padded by reflection
/// (by symmetric mirroring when a side is 16 or shorter, since a reflection
/// needs a pad shorter than the side). Returns the canvas picture and, for a
/// padded one, the `[left, top, right, bottom]` box of the fitted picture
/// inside it.
#[must_use]
pub(crate) fn prepare_single_reference(img: &RgbImage) -> (RgbImage, Option<[u32; 4]>) {
    let (fit, (w, h)) = reference_canvas(img.width(), img.height());
    let fitted = if fit == img.dimensions() {
        img.clone()
    } else {
        image::imageops::resize(img, fit.0, fit.1, FilterType::Lanczos3)
    };
    if (w, h) == fitted.dimensions() {
        return (fitted, None);
    }
    let (left, top) = ((w - fit.0) / 2, (h - fit.1) / 2);
    let reflect = fit.0.min(fit.1) > GRID;
    let canvas = RgbImage::from_fn(w, h, |x, y| {
        let sx = fold(i64::from(x) - i64::from(left), i64::from(fit.0), reflect);
        let sy = fold(i64::from(y) - i64::from(top), i64::from(fit.1), reflect);
        *fitted.get_pixel(
            u32::try_from(sx).unwrap_or(0),
            u32::try_from(sy).unwrap_or(0),
        )
    });
    (canvas, Some([left, top, left + fit.0, top + fit.1]))
}

/// One reference of a multi-reference edit: fitted under 1 MP (sides
/// truncated), then centre-cropped to the 16-pixel grid. The picture must be
/// at least 16 pixels on each side ([`check_reference_size`]).
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "image sides are at most a few thousand, exact in f64"
)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the scaled sides are truncated positive values below the original sides"
)]
pub(crate) fn prepare_multi_reference(img: &RgbImage) -> RgbImage {
    let (mut w, mut h) = img.dimensions();
    let fitted;
    let mut view = img;
    if u64::from(w) * u64::from(h) > MULTI_REFERENCE_MAX_PIXELS {
        let scale = (MULTI_REFERENCE_MAX_PIXELS as f64 / (f64::from(w) * f64::from(h))).sqrt();
        w = (f64::from(w) * scale) as u32;
        h = (f64::from(h) * scale) as u32;
        fitted = image::imageops::resize(img, w, h, FilterType::Lanczos3);
        view = &fitted;
    }
    let (cw, ch) = (w / GRID * GRID, h / GRID * GRID);
    let (left, top) = ((w - cw) / 2, (h - ch) / 2);
    image::imageops::crop_imm(view, left, top, cw, ch).to_image()
}

/// Refuses a picture too small to hold one latent cell.
///
/// # Errors
/// [`ImageGenError::ReferenceImage`] naming the size.
pub(crate) fn check_reference_size(img: &RgbImage, index: usize) -> Result<(), ImageGenError> {
    if img.width() < GRID || img.height() < GRID {
        return Err(ImageGenError::ReferenceImage(format!(
            "reference image {} is {}x{}; every side must be at least {GRID} pixels",
            index + 1,
            img.width(),
            img.height()
        )));
    }
    Ok(())
}

/// Refuses more than [`MAX_REFERENCES`] references.
///
/// # Errors
/// [`ImageGenError::ReferenceImage`] naming the count.
pub(crate) fn check_reference_count(count: usize) -> Result<(), ImageGenError> {
    if count > MAX_REFERENCES {
        return Err(ImageGenError::ReferenceImage(format!(
            "{count} reference images were given; at most {MAX_REFERENCES} are accepted"
        )));
    }
    Ok(())
}

/// Decodes an encoded reference to RGB with its EXIF orientation applied.
///
/// # Errors
/// [`ImageGenError::ReferenceImage`] when the bytes are not an image the
/// build decodes.
pub(crate) fn decode_reference(bytes: &[u8], index: usize) -> Result<RgbImage, ImageGenError> {
    let fail = |e: image::ImageError| {
        ImageGenError::ReferenceImage(format!(
            "reference image {} does not decode: {e}",
            index + 1
        ))
    };
    let mut decoder = ImageReader::new(std::io::Cursor::new(bytes))
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
    Ok(picture.to_rgb8())
}

/// The size an edit takes when its caller names none: 1024x1024 without
/// references, a single reference's canvas, or the first of several
/// references' prepared size.
///
/// A generator is handed a final size, so whoever fills in an omitted one
/// calls this first; a job of exactly a single reference's canvas size
/// comes back cropped to that reference's own size.
///
/// # Errors
/// [`ImageGenError::ReferenceImage`] for too many references, one that does
/// not decode or one under 16 pixels on a side.
pub fn default_size_for_references(
    references: &[InputReferenceImage],
) -> Result<(u32, u32), ImageGenError> {
    check_reference_count(references.len())?;
    let Some(first) = references.first() else {
        return Ok((1024, 1024));
    };
    let picture = decode_reference(&first.bytes, 0)?;
    check_reference_size(&picture, 0)?;
    if references.len() == 1 {
        Ok(reference_canvas(picture.width(), picture.height()).1)
    } else {
        Ok(prepare_multi_reference(&picture).dimensions())
    }
}

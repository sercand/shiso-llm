//! The host-side preprocessing stage: photo in, 1024-square RGB out.
//!
//! The rules are the reference implementation's, reproduced exactly rather
//! than approximately, because the encoders were trained on its output: the
//! short side is scaled to the model's input size *before* matting, the alpha
//! is eroded by one pixel so no background-coloured fringe survives, the crop
//! is a square of `crop_margin` times the object's bounding box, and the result
//! is composited over black.
//!
//! # Why the order matters
//!
//! Matting after the first resize is not an optimisation: a background
//! remover trained at one scale gives a different mask at another, and the
//! reference mattes at the model's scale. Eroding after matting rather than
//! before is what removes the fringe the remover's soft edge leaves behind.
//! Both are observable in the output, so both are fixed here.

use image::{Rgba, RgbaImage};

/// The model's input side: every preprocessed picture is this square.
pub const IMAGE_SIZE: u32 = 1024;

/// The longest side the first resize may produce.
///
/// A picture whose short side is tiny but whose long side is enormous - a
/// 100000x1 strip is a few hundred compressed bytes - would otherwise scale to
/// a hundred million pixels wide and abort the process on the allocation. The
/// reference lets its allocator raise; a server that a single request can kill
/// has to refuse instead.
pub const MAX_SCALED_SIDE: u32 = 65_536;

/// The largest area the first resize may produce, in pixels.
pub const MAX_SCALED_PIXELS: u64 = 64_000_000;

/// A preprocessing refusal.
#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    /// The bytes are not a decodable PNG or JPEG.
    #[error("image could not be decoded: {0}")]
    Decode(String),
    /// The alpha channel is empty after matting and erosion: there is no
    /// object to crop to.
    #[error("no object found in the image (the alpha channel is empty)")]
    EmptyAlpha,
    /// The image has no real alpha and no background remover is loaded.
    #[error("the image has no alpha and no background remover is loaded")]
    NoMatte,
    /// The background remover failed, or answered the wrong number of values.
    #[error("{0}")]
    Matting(String),
    /// Scaling the short side to the model's input size would produce a
    /// picture no host can hold.
    #[error(
        "a {width}x{height} picture scales to {scaled_width}x{scaled_height}, over this stage's \
         {MAX_SCALED_SIDE}-pixel side or {MAX_SCALED_PIXELS}-pixel area ceiling"
    )]
    TooLarge {
        /// The source width.
        width: u32,
        /// The source height.
        height: u32,
        /// The width after the first resize.
        scaled_width: u32,
        /// The height after the first resize.
        scaled_height: u32,
    },
}

/// Background removal, as the preprocessing stage needs it.
///
/// A seam rather than a concrete model so the stage can be tested without a
/// device, and so the remover can be swapped or left out on a host that only
/// ever sees cut-outs.
pub trait Matting: Send + Sync + std::fmt::Debug {
    /// The alpha channel for `rgb`, one byte per pixel in row-major order.
    ///
    /// # Errors
    ///
    /// A human-readable reason the remover could not run.
    fn alpha(&self, rgb: &image::RgbImage) -> Result<Vec<u8>, String>;
}

/// What the encoders read: a 1024-square RGB image, background composited to
/// black.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preprocessed {
    /// The picture the encoders see.
    pub rgb: image::RgbImage,
}

/// Decodes PNG or JPEG bytes into an RGBA image.
///
/// # Errors
///
/// [`PreprocessError::Decode`] for bytes the `image` crate cannot read.
pub fn decode_rgba(bytes: &[u8]) -> Result<RgbaImage, PreprocessError> {
    image::load_from_memory(bytes)
        .map(|image| image.to_rgba8())
        .map_err(|error| PreprocessError::Decode(error.to_string()))
}

/// Turns encoded picture bytes into the square the encoders read.
///
/// `crop_margin` is already resolved: the caller has turned the request's mode
/// into a margin, so this stage does not need to know what a mode is.
///
/// # Errors
///
/// [`PreprocessError::Decode`] for bytes that are not a picture,
/// [`PreprocessError::NoMatte`] when the picture has no real alpha and no
/// remover is loaded, [`PreprocessError::Matting`] when the remover failed,
/// and
/// [`PreprocessError::EmptyAlpha`] when nothing opaque is left to crop to.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the scaled side is a positive product of a u32 dimension and a positive ratio, \
              rounded and floored at 1, so it is a small positive integer"
)]
pub fn preprocess(
    image: &[u8],
    crop_margin: f32,
    matting: Option<&dyn Matting>,
) -> Result<Preprocessed, PreprocessError> {
    let decoded = decode_rgba(image)?;
    let (width, height) = decoded.dimensions();
    // The short side becomes the model's input size first, so matting runs at
    // the scale the remover was trained at and the crop arithmetic below works
    // on the picture the encoders will actually see.
    let scale = f64::from(IMAGE_SIZE) / f64::from(width.min(height));
    // Python's `round` is half-to-even, and the reference uses it: a 2049x2048
    // picture scales to 1024.5 wide, which the reference rounds *down* to
    // 1024. Rounding half away from zero would shift the crop by a column.
    let scaled_w = ((f64::from(width) * scale).round_ties_even() as u32).max(1);
    let scaled_h = ((f64::from(height) * scale).round_ties_even() as u32).max(1);
    if scaled_w > MAX_SCALED_SIDE
        || scaled_h > MAX_SCALED_SIDE
        || u64::from(scaled_w) * u64::from(scaled_h) > MAX_SCALED_PIXELS
    {
        return Err(PreprocessError::TooLarge {
            width,
            height,
            scaled_width: scaled_w,
            scaled_height: scaled_h,
        });
    }
    let mut working = resize_lanczos3(&decoded, scaled_w, scaled_h);

    // The reference asks whether Pillow's *mode* is RGBA and some alpha is not
    // 255. `to_rgba8` cannot answer that: it turns a palette PNG with a `tRNS`
    // chunk into RGBA, which Pillow keeps as mode `P` and therefore mattes. The
    // PNG header's colour type is what tells the two apart.
    if !(is_rgba_png(image) && has_transparent_pixel(&working)) {
        let matting = matting.ok_or(PreprocessError::NoMatte)?;
        let rgb = to_rgb(&working);
        let alpha = matting.alpha(&rgb).map_err(PreprocessError::Matting)?;
        let expected = (working.width() * working.height()) as usize;
        if alpha.len() != expected {
            return Err(PreprocessError::Matting(format!(
                "the background remover answered {} alpha values for {expected} pixels",
                alpha.len()
            )));
        }
        for (pixel, value) in working.pixels_mut().zip(alpha) {
            pixel.0[3] = value;
        }
    }

    min_filter3(&mut working);
    let bbox = alpha_bbox(&working).ok_or(PreprocessError::EmptyAlpha)?;
    let (left, top, width, height) = crop_box(bbox, crop_margin);
    let cropped = crop_with_pad(&working, left, top, width, height);
    let square = resize_lanczos3(&cropped, IMAGE_SIZE, IMAGE_SIZE);
    Ok(Preprocessed {
        rgb: composite_over_black(&square),
    })
}

/// Whether the encoded picture is a true-colour PNG with an alpha channel.
///
/// PNG's IHDR colour type lives at byte 25 of the file: an 8-byte signature, a
/// 4-byte chunk length, the 4-byte `IHDR` tag, the 4-byte width, the 4-byte
/// height, one byte of bit depth, then the colour type. Type 6 is RGBA, which
/// is the only PNG Pillow opens as mode `RGBA`; a palette PNG with a `tRNS`
/// chunk is mode `P` and is matted by the reference even though it carries
/// transparency.
#[must_use]
pub(crate) fn is_rgba_png(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n") && bytes.get(25) == Some(&6)
}

/// Whether any pixel is not fully opaque.
///
/// A fully opaque alpha is the same as no alpha at all: the reference treats
/// it that way, so an RGBA PNG saved with an all-255 alpha still goes through
/// matting rather than being cropped to its whole frame.
#[must_use]
pub(crate) fn has_transparent_pixel(image: &RgbaImage) -> bool {
    image.pixels().any(|pixel| pixel.0[3] < 255)
}

/// The smallest rectangle around the opaque pixels, in source pixels, with
/// both edges inclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Bbox {
    /// Leftmost opaque column.
    pub left: u32,
    /// Topmost opaque row.
    pub top: u32,
    /// Rightmost opaque column.
    pub right: u32,
    /// Bottommost opaque row.
    pub bottom: u32,
}

/// The bounding box of every pixel whose alpha is not zero.
///
/// Returns `None` when no pixel is opaque, which is the reference's "no object
/// found" case: a zero-Gaussian splat is not a useful answer, so the caller
/// refuses instead.
#[must_use]
pub(crate) fn alpha_bbox(image: &RgbaImage) -> Option<Bbox> {
    let (width, height) = image.dimensions();
    let mut left = width;
    let mut top = height;
    let mut right = 0u32;
    let mut bottom = 0u32;
    let mut found = false;
    for y in 0..height {
        for x in 0..width {
            if image.get_pixel(x, y).0[3] == 0 {
                continue;
            }
            found = true;
            left = left.min(x);
            right = right.max(x);
            top = top.min(y);
            bottom = bottom.max(y);
        }
    }
    found.then_some(Bbox {
        left,
        top,
        right,
        bottom,
    })
}

/// The square crop around `bbox`, in source pixels.
///
/// Returns `(left, top, width, height)`. The side is `crop_margin` times the
/// longer edge of the bounding box, and the edges are truncated toward zero
/// exactly as the reference's `int()` does - a floor would move the crop on
/// the upper and left edges by a pixel whenever the centre lands below an
/// integer.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the crop geometry is float by construction and truncated toward zero on purpose; \
              image dimensions are far below f32's exact-integer range and the result is floored \
              at one pixel per side"
)]
pub(crate) fn crop_box(bbox: Bbox, margin: f32) -> (i64, i64, u32, u32) {
    let center_x = (bbox.left + bbox.right) as f32 / 2.0;
    let center_y = (bbox.top + bbox.bottom) as f32 / 2.0;
    let span = (bbox.right - bbox.left).max(bbox.bottom - bbox.top) as f32;
    let half = span / 2.0 * margin;
    let left = (center_x - half) as i64;
    let top = (center_y - half) as i64;
    let right = (center_x + half) as i64;
    let bottom = (center_y + half) as i64;
    let width = (right - left).max(1) as u32;
    let height = (bottom - top).max(1) as u32;
    (left, top, width, height)
}

/// Copies the `width`x`height` rectangle at `(left, top)`, padding with
/// transparent pixels wherever it leaves the source.
///
/// Padding rather than refusing is what makes an object that touches the frame
/// edge a legitimate photograph: the margin around it is background, and the
/// reference pads it the same way.
#[must_use]
pub(crate) fn crop_with_pad(
    image: &RgbaImage,
    left: i64,
    top: i64,
    width: u32,
    height: u32,
) -> RgbaImage {
    let mut out = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));
    let (src_w, src_h) = image.dimensions();
    for dst_y in 0..height {
        let src_y = top + i64::from(dst_y);
        if src_y < 0 || src_y >= i64::from(src_h) {
            continue;
        }
        for dst_x in 0..width {
            let src_x = left + i64::from(dst_x);
            if src_x < 0 || src_x >= i64::from(src_w) {
                continue;
            }
            // Both are inside the source's bounds, so the conversion cannot
            // fail; the `expect` names that rather than casting.
            let (Ok(x), Ok(y)) = (u32::try_from(src_x), u32::try_from(src_y)) else {
                continue;
            };
            out.put_pixel(dst_x, dst_y, *image.get_pixel(x, y));
        }
    }
    out
}

/// Erodes the alpha channel by one pixel, in place.
///
/// A 3x3 minimum with edge clamping, which is Pillow's `MinFilter(3)`: every
/// pixel takes the darkest alpha in its neighbourhood, so a one-pixel fringe of
/// background-coloured pixels around the object loses its alpha. Read from a
/// copy so the result does not feed itself.
pub(crate) fn min_filter3(image: &mut RgbaImage) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return;
    }
    let alpha: Vec<u8> = image.pixels().map(|pixel| pixel.0[3]).collect();
    let at = |x: i64, y: i64| -> u8 {
        // The clamp makes both conversions infallible; the fallback is the
        // first pixel and is never reached.
        let x = u32::try_from(x.clamp(0, i64::from(width) - 1)).unwrap_or(0);
        let y = u32::try_from(y.clamp(0, i64::from(height) - 1)).unwrap_or(0);
        alpha[(y * width + x) as usize]
    };
    for y in 0..height {
        for x in 0..width {
            let mut minimum = 255u8;
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    minimum = minimum.min(at(i64::from(x) + dx, i64::from(y) + dy));
                }
            }
            image.get_pixel_mut(x, y).0[3] = minimum;
        }
    }
}

/// Composites the picture over black using its alpha, dropping the alpha.
///
/// The rounding is Pillow's `paste(mask=...)`: `(channel * alpha + 127) / 255`,
/// which is the nearest integer rather than a truncation.
#[must_use]
pub(crate) fn composite_over_black(image: &RgbaImage) -> image::RgbImage {
    image::RgbImage::from_fn(image.width(), image.height(), |x, y| {
        let pixel = image.get_pixel(x, y);
        let alpha = u32::from(pixel.0[3]);
        // The quotient is at most 255 by construction, so the conversion is
        // infallible; the fallback names that rather than casting.
        let blend =
            |channel: u8| u8::try_from((u32::from(channel) * alpha + 127) / 255).unwrap_or(255);
        image::Rgb([blend(pixel.0[0]), blend(pixel.0[1]), blend(pixel.0[2])])
    })
}

/// The RGB channels of an RGBA image.
fn to_rgb(image: &RgbaImage) -> image::RgbImage {
    image::RgbImage::from_fn(image.width(), image.height(), |x, y| {
        let pixel = image.get_pixel(x, y);
        image::Rgb([pixel.0[0], pixel.0[1], pixel.0[2]])
    })
}

/// The Lanczos-3 window function: `sinc(x) * sinc(x / 3)` inside its support.
///
/// Pillow's `LANCZOS` filter, which the reference uses for every resize in the
/// stage. Three lobes each way is what makes the filter's support 3 source
/// pixels, which is what the window arithmetic below assumes.
fn lanczos3(x: f64) -> f64 {
    if x.abs() >= 3.0 {
        return 0.0;
    }
    let sinc = |value: f64| {
        if value == 0.0 {
            1.0
        } else {
            let angle = std::f64::consts::PI * value;
            angle.sin() / angle
        }
    };
    sinc(x) * sinc(x / 3.0)
}

/// One output pixel's source window and weights along one axis.
struct AxisWeights {
    /// First source index the window covers.
    start: usize,
    /// Normalised weights, one per source index from `start`.
    values: Vec<f64>,
}

/// The source window and normalised weights for every output index.
///
/// This is Pillow's `precompute_coeffs`: the window is the filter's support
/// widened by the downscale factor, and the weights are normalised so a
/// constant image survives the pass exactly.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the window arithmetic is float by construction, as in the reference: lengths are \
              image dimensions and the bounds are clamped into the source before use"
)]
fn axis_weights(src_len: usize, dst_len: usize) -> Vec<AxisWeights> {
    let scale = src_len as f64 / dst_len as f64;
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    let mut out = Vec::with_capacity(dst_len);
    for index in 0..dst_len {
        let center = (index as f64 + 0.5) * scale;
        // `as i64` truncates toward zero, which is what Pillow's C cast does;
        // the clamp to the image makes the negative case moot.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(src_len as i64) as usize;
        let mut values: Vec<f64> = (xmin..xmax)
            .map(|x| lanczos3((x as f64 - center + 0.5) / filter_scale))
            .collect();
        if values.is_empty() {
            // A window that rounds away entirely would divide by zero; the
            // nearest source pixel is the only defensible answer, and it
            // cannot happen for a non-empty source.
            out.push(AxisWeights {
                start: xmin.min(src_len.saturating_sub(1)),
                values: vec![1.0],
            });
            continue;
        }
        let total: f64 = values.iter().sum();
        if total != 0.0 {
            for weight in &mut values {
                *weight /= total;
            }
        }
        out.push(AxisWeights {
            start: xmin,
            values,
        });
    }
    out
}

/// A channel value rounded to the nearest integer and clamped to `0..=255`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to 0..=255 and rounded before the cast, so it is a whole \
              number in range"
)]
fn to_channel(value: f64) -> u8 {
    value.clamp(0.0, 255.0).round() as u8
}

/// Resizes with Pillow's Lanczos-3 geometry.
///
/// Separable, and streamed along the vertical axis: each source row is
/// horizontally resampled once, lazily, and kept only while some output row
/// still needs it. A naive two-pass resize would hold a full-width `f64`
/// intermediate, which for an 8000x6000 photograph is larger than the picture
/// itself; this holds one vertical window instead.
pub(crate) fn resize_lanczos3(src: &RgbaImage, width: u32, height: u32) -> RgbaImage {
    let (src_w, src_h) = src.dimensions();
    if src_w == 0 || src_h == 0 || width == 0 || height == 0 {
        return RgbaImage::new(width, height);
    }
    let horizontal = axis_weights(src_w as usize, width as usize);
    let vertical = axis_weights(src_h as usize, height as usize);
    let mut out = RgbaImage::new(width, height);
    let mut cache: std::collections::VecDeque<(usize, Vec<f64>)> =
        std::collections::VecDeque::new();
    let mut next_source_row = 0usize;
    let mut accumulator = vec![0.0f64; width as usize * 4];
    for (y, weights) in vertical.iter().enumerate() {
        let end = weights.start + weights.values.len();
        while next_source_row < end {
            let row = horizontal_row(src, next_source_row, &horizontal);
            cache.push_back((next_source_row, row));
            next_source_row += 1;
        }
        while cache
            .front()
            .is_some_and(|(index, _)| *index < weights.start)
        {
            cache.pop_front();
        }
        accumulator.fill(0.0);
        for (offset, &weight) in weights.values.iter().enumerate() {
            let Some((_, row)) = cache.get(offset) else {
                continue;
            };
            for (sum, value) in accumulator.iter_mut().zip(row) {
                *sum += weight * value;
            }
        }
        // The output row index is a `usize` from `enumerate` over `height`
        // entries, so it fits the image's own `u32` height.
        let Ok(y) = u32::try_from(y) else {
            continue;
        };
        for x in 0..width {
            let base = x as usize * 4;
            out.put_pixel(
                x,
                y,
                Rgba([
                    to_channel(accumulator[base]),
                    to_channel(accumulator[base + 1]),
                    to_channel(accumulator[base + 2]),
                    to_channel(accumulator[base + 3]),
                ]),
            );
        }
    }
    out
}

/// One source row resampled to the output width, as `f64` channels.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the source index is a window offset bounded by the row's own length"
)]
fn horizontal_row(src: &RgbaImage, y: usize, weights: &[AxisWeights]) -> Vec<f64> {
    let mut row = vec![0.0f64; weights.len() * 4];
    for (x, window) in weights.iter().enumerate() {
        for (offset, &weight) in window.values.iter().enumerate() {
            let pixel = src.get_pixel((window.start + offset) as u32, y as u32);
            for channel in 0..4 {
                row[x * 4 + channel] += weight * f64::from(pixel.0[channel]);
            }
        }
    }
    row
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "fixtures are built from small literal dimensions, so every narrowing is exact"
)]
mod tests {
    use image::{Rgb, RgbImage, Rgba};

    use super::*;

    fn image_with_alpha(width: u32, height: u32, opaque: &[(u32, u32)]) -> RgbaImage {
        let mut image = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));
        for &(x, y) in opaque {
            image.put_pixel(x, y, Rgba([255, 255, 255, 255]));
        }
        image
    }

    #[test]
    fn a_fully_opaque_alpha_is_not_a_real_alpha() {
        let image = RgbaImage::from_pixel(4, 4, Rgba([1, 2, 3, 255]));
        assert!(!has_transparent_pixel(&image));
        let mut image = image;
        image.put_pixel(0, 0, Rgba([1, 2, 3, 254]));
        assert!(has_transparent_pixel(&image));
    }

    #[test]
    fn only_a_true_colour_png_counts_as_rgba() {
        // A 1x1 RGBA PNG's IHDR colour type is 6.
        let rgba = png_with_alpha(1, 1, 0..1);
        assert!(is_rgba_png(&rgba));
        // The same picture without an alpha channel is colour type 2.
        let rgb = opaque_png(1, 1);
        assert!(!is_rgba_png(&rgb));
        // Anything that is not a PNG cannot be Pillow's mode `RGBA`.
        assert!(!is_rgba_png(b"\xff\xd8\xff\xe0"));
    }

    #[test]
    fn an_empty_alpha_is_refused_not_ignored() {
        let image = image_with_alpha(4, 4, &[]);
        assert!(alpha_bbox(&image).is_none());
    }

    #[test]
    fn the_bbox_is_the_smallest_rectangle_around_the_alpha() {
        let image = image_with_alpha(8, 8, &[(2, 3), (5, 3), (2, 6), (5, 6)]);
        let bbox = alpha_bbox(&image).expect("non-empty");
        assert_eq!((bbox.left, bbox.top, bbox.right, bbox.bottom), (2, 3, 5, 6));
    }

    #[test]
    fn the_crop_box_truncates_like_python_int() {
        // bbox x 2..=5, y 3..=6: span = 3, half = 3 / 2 * 1.2 = 1.8
        // left = int(3.5 - 1.8) = int(1.7) = 1, right = int(5.3) = 5 -> 4 wide
        // top  = int(4.5 - 1.8) = int(2.7) = 2, bottom = int(6.3) = 6 -> 4 tall
        let bbox = Bbox {
            left: 2,
            top: 3,
            right: 5,
            bottom: 6,
        };
        assert_eq!(crop_box(bbox, 1.2), (1, 2, 4, 4));
    }

    #[test]
    fn a_negative_edge_truncates_toward_zero_not_down() {
        // centre 0.5, half 1.8: left = int(-1.3) = -1 (a floor would say -2),
        // right = int(2.3) = 2 -> 3 wide.
        let bbox = Bbox {
            left: 0,
            top: 0,
            right: 1,
            bottom: 1,
        };
        assert_eq!(crop_box(bbox, 3.6), (-1, -1, 3, 3));
    }

    #[test]
    fn a_taller_than_wide_box_takes_its_height() {
        // span = max(1, 5) = 5, half = 2.5 * 1.7 = 4.25
        // centre (1.5, 3.5): left = int(-2.75) = -2, right = int(5.75) = 5
        // top = int(-0.75) = 0, bottom = int(7.75) = 7
        let bbox = Bbox {
            left: 1,
            top: 1,
            right: 2,
            bottom: 6,
        };
        assert_eq!(crop_box(bbox, 1.7), (-2, 0, 7, 7));
    }

    #[test]
    fn a_crop_that_leaves_the_image_is_padded_transparent() {
        let image = image_with_alpha(4, 4, &[(0, 0)]);
        let cropped = crop_with_pad(&image, -2, -2, 8, 8);
        assert_eq!(cropped.dimensions(), (8, 8));
        assert_eq!(
            cropped.get_pixel(0, 0).0[3],
            0,
            "outside the source is transparent"
        );
        assert_eq!(
            cropped.get_pixel(2, 2).0[3],
            255,
            "the source lands at the offset"
        );
    }

    #[test]
    fn a_tiny_image_does_not_produce_a_zero_side_crop() {
        let image = image_with_alpha(1, 1, &[(0, 0)]);
        let bbox = alpha_bbox(&image).expect("non-empty");
        let (_, _, width, height) = crop_box(bbox, 1.2);
        assert!(width >= 1 && height >= 1, "{width}x{height}");
    }

    #[test]
    fn a_png_decodes_to_opaque_rgba() {
        let mut buffer = std::io::Cursor::new(Vec::new());
        let rgb = RgbImage::from_pixel(2, 2, Rgb([10, 20, 30]));
        image::DynamicImage::ImageRgb8(rgb)
            .write_to(&mut buffer, image::ImageFormat::Png)
            .expect("encodes");
        let decoded = decode_rgba(buffer.get_ref()).expect("decodes");
        assert_eq!(decoded.dimensions(), (2, 2));
        assert!(!has_transparent_pixel(&decoded));
    }

    #[test]
    fn bytes_that_are_not_an_image_are_a_decode_error() {
        assert!(matches!(
            decode_rgba(b"not an image"),
            Err(PreprocessError::Decode(_))
        ));
    }

    #[test]
    fn a_constant_image_survives_the_resample() {
        let image = RgbaImage::from_pixel(17, 13, Rgba([10, 20, 30, 200]));
        let out = resize_lanczos3(&image, 9, 7);
        for pixel in out.pixels() {
            assert_eq!(pixel.0, [10, 20, 30, 200]);
        }
    }

    #[test]
    fn the_filter_has_three_lobes_and_is_normalised_at_zero() {
        assert!((lanczos3(0.0) - 1.0).abs() < 1e-12);
        assert!(lanczos3(3.0).abs() < 1e-12, "the support ends at 3");
        assert!(lanczos3(4.0).abs() < 1e-12);
        assert!(lanczos3(1.0).abs() < 1e-12, "sinc(1) is zero");
    }

    #[test]
    fn downscaling_tracks_a_pillow_golden() {
        // A 4x4 grey ramp scaled to 2x2. The expected values are what Pillow's
        // Image.LANCZOS produces for this exact input, and this implementation
        // reproduces them exactly. Pillow's 8-bit path quantises its
        // coefficients to 22-bit fixed point and rounds to `u8` between the two
        // passes where this one keeps `f64` throughout, so the two can differ
        // by one unit on other inputs; the geometry, which is what the model
        // depends on, is identical. The comparison is exact because this
        // implementation is deterministic.
        let mut image = RgbaImage::new(4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let value = (x * 60 + y * 15) as u8;
                image.put_pixel(x, y, Rgba([value, value, value, 255]));
            }
        }
        let out = resize_lanczos3(&image, 2, 2);
        let got: Vec<u8> = out.pixels().map(|pixel| pixel.0[0]).collect();
        assert_eq!(got, [40u8, 156, 69, 185]);
    }

    #[test]
    fn upscaling_a_constant_image_is_exact() {
        let image = RgbaImage::from_pixel(3, 2, Rgba([7, 8, 9, 255]));
        let out = resize_lanczos3(&image, 11, 7);
        assert_eq!(out.dimensions(), (11, 7));
        for pixel in out.pixels() {
            assert_eq!(pixel.0, [7, 8, 9, 255]);
        }
    }

    #[test]
    fn a_one_pixel_source_does_not_divide_by_zero() {
        let image = RgbaImage::from_pixel(1, 1, Rgba([1, 2, 3, 4]));
        let out = resize_lanczos3(&image, 4, 4);
        for pixel in out.pixels() {
            assert_eq!(pixel.0, [1, 2, 3, 4]);
        }
    }

    #[test]
    fn the_erode_shrinks_the_alpha_by_one_pixel() {
        let mut image = RgbaImage::from_pixel(5, 5, Rgba([0, 0, 0, 0]));
        for y in 1..3 {
            for x in 1..3 {
                image.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        min_filter3(&mut image);
        assert_eq!(
            image.get_pixel(1, 1).0[3],
            0,
            "an edge pixel loses its alpha"
        );
    }

    #[test]
    fn compositing_over_black_uses_the_alpha() {
        let mut image = RgbaImage::from_pixel(1, 1, Rgba([200, 100, 50, 128]));
        assert_eq!(
            composite_over_black(&image).get_pixel(0, 0).0,
            [100, 50, 25]
        );
        image.put_pixel(0, 0, Rgba([200, 100, 50, 255]));
        assert_eq!(
            composite_over_black(&image).get_pixel(0, 0).0,
            [200, 100, 50]
        );
        image.put_pixel(0, 0, Rgba([200, 100, 50, 1]));
        assert_eq!(composite_over_black(&image).get_pixel(0, 0).0, [1, 0, 0]);
    }

    /// A gradient rather than a flat colour, so a resize that silently
    /// dropped the picture would not still pass.
    fn opaque_png(width: u32, height: u32) -> Vec<u8> {
        let image = RgbImage::from_fn(width, height, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        encode(&image::DynamicImage::ImageRgb8(image))
    }

    /// An RGBA picture whose alpha is 255 inside `range` and 0 outside.
    fn png_with_alpha(width: u32, height: u32, range: std::ops::Range<u32>) -> Vec<u8> {
        let image = RgbaImage::from_fn(width, height, |x, y| {
            let inside = range.contains(&x) && range.contains(&y);
            let alpha = if inside { 255 } else { 0 };
            Rgba([(x % 256) as u8, (y % 256) as u8, 90, alpha])
        });
        encode(&image::DynamicImage::ImageRgba8(image))
    }

    fn encode(image: &image::DynamicImage) -> Vec<u8> {
        let mut buffer = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut buffer, image::ImageFormat::Png)
            .expect("encodes");
        buffer.into_inner()
    }

    /// A matte that marks a square, built from the dimensions it is handed:
    /// the matte runs after the first resize, so the image is 1024 on its
    /// short side and may be wider than that.
    #[derive(Debug)]
    struct SquareMatte {
        lo: u32,
        hi: u32,
        /// The dimensions the matte was handed, so a test can prove the stage
        /// resized *before* matting rather than after.
        seen: std::sync::Mutex<Option<(u32, u32)>>,
    }

    impl SquareMatte {
        fn new(lo: u32, hi: u32) -> Self {
            Self {
                lo,
                hi,
                seen: std::sync::Mutex::new(None),
            }
        }
    }

    impl Matting for SquareMatte {
        fn alpha(&self, rgb: &RgbImage) -> Result<Vec<u8>, String> {
            let (width, height) = rgb.dimensions();
            *self.seen.lock().expect("the matte's own lock") = Some((width, height));
            let mut alpha = vec![0u8; (width * height) as usize];
            for y in self.lo..self.hi.min(height) {
                for x in self.lo..self.hi.min(width) {
                    alpha[(y * width + x) as usize] = 255;
                }
            }
            Ok(alpha)
        }
    }

    #[test]
    fn a_real_alpha_channel_skips_the_matte() {
        let png = png_with_alpha(1024, 1024, 300..724);
        let out = preprocess(&png, 1.2, None).expect("no matte needed");
        assert_eq!(out.rgb.dimensions(), (IMAGE_SIZE, IMAGE_SIZE));
    }

    #[test]
    fn an_opaque_photo_without_a_matte_is_refused() {
        let png = opaque_png(1024, 1024);
        let err = preprocess(&png, 1.7, None).expect_err("no matte available");
        assert!(matches!(err, PreprocessError::NoMatte), "{err}");
    }

    #[test]
    fn the_output_is_always_square_and_black_outside_the_object() {
        let matte = SquareMatte::new(200, 824);
        let out = preprocess(&opaque_png(1200, 800), 1.2, Some(&matte)).expect("runs");
        assert_eq!(out.rgb.dimensions(), (IMAGE_SIZE, IMAGE_SIZE));
        assert_eq!(out.rgb.get_pixel(0, 0).0, [0, 0, 0], "corner is background");
    }

    #[test]
    fn a_large_photo_is_scaled_before_the_crop() {
        let matte = SquareMatte::new(400, 700);
        let out = preprocess(&opaque_png(4000, 3000), 1.7, Some(&matte)).expect("runs");
        assert_eq!(out.rgb.dimensions(), (IMAGE_SIZE, IMAGE_SIZE));
    }

    #[test]
    fn an_object_touching_the_edge_is_padded_not_refused() {
        let matte = SquareMatte::new(0, 300);
        let out = preprocess(&opaque_png(1024, 1024), 1.7, Some(&matte)).expect("runs");
        assert_eq!(out.rgb.dimensions(), (IMAGE_SIZE, IMAGE_SIZE));
    }

    #[test]
    fn a_matte_that_finds_nothing_is_refused() {
        let matte = SquareMatte::new(0, 0);
        let err = preprocess(&opaque_png(512, 512), 1.2, Some(&matte))
            .expect_err("an empty matte leaves nothing to crop");
        assert!(matches!(err, PreprocessError::EmptyAlpha));
    }

    #[test]
    fn a_matte_of_the_wrong_length_is_refused() {
        #[derive(Debug)]
        struct Short;
        impl Matting for Short {
            fn alpha(&self, _rgb: &RgbImage) -> Result<Vec<u8>, String> {
                Ok(vec![255; 4])
            }
        }
        let err = preprocess(&opaque_png(512, 512), 1.2, Some(&Short))
            .expect_err("a short matte is a bug, not a picture");
        assert!(matches!(err, PreprocessError::Matting(_)));
    }
}

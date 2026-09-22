//! Still-image preprocessing: decode, orient, resize, normalize, group.
//!
//! The output is a [`VisionItem`] with one temporal group. That is not a
//! special case dressed up as the general one - the tower's patch dimension
//! spans `temporal_patch_size` frames whether or not the input moves, so a
//! still fills that axis by replicating itself and is genuinely the degenerate
//! video. The packing loop below is shared with the video path for exactly
//! that reason: two copies of it would be two places for the channel/time/row
//! offset to drift.

use std::num::NonZeroU64;

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits, RgbImage};

use crate::config::{ImageLimits, PatchGeometry, PreprocessSpec, Resample, ResizeRule};
use crate::error::VisionError;
use crate::item::VisionItem;

/// Long-side cap applied ONLY when nothing else bounds the image, i.e. the
/// caller declared no area bound because the checkpoint shipped no
/// preprocessor config and the operator set no override.
///
/// This was once an unconditional ceiling, and as an unconditional ceiling it
/// silently discarded most of the resolution a modern checkpoint permits: a
/// tower declaring an area bound of 4096x4096 still had every input clamped to
/// 1280 on the long side, roughly a tenth of the permitted area. Detail-bearing
/// inputs - documents, charts, dense screenshots - paid for that directly, and
/// nothing logged it. It survives as a fallback because a checkpoint that
/// declares nothing still needs a defined answer.
const FALLBACK_MAX_DIM: u32 = 1280;

/// Absolute long-side ceiling, applied even when an area bound is in force.
///
/// An area bound alone permits an unbounded long side, because a 1xN strip
/// satisfies any area. This is the safety net the fallback clamp was
/// informally providing before it became a fallback, and it sits deliberately
/// far above any sane vision input.
const ABS_MAX_DIM: u32 = 4096;

/// Decoder limit: reject a header declaring more than this on either side,
/// before a single pixel is allocated.
///
/// Everything is resized down to at most [`ABS_MAX_DIM`] anyway, so this only
/// has to sit above any real camera image; 16384 is about four times the long
/// side of a 50-megapixel photograph. The check matters because a 40-byte PNG
/// header can declare 65535x65535, which is roughly 13 GB of RGB, and the
/// bytes arrive from an unauthenticated caller.
const DECODE_MAX_SIDE: u32 = 16_384;

/// Turns one encoded image into patch rows for the vision tower.
///
/// Takes decoded bytes rather than a `data:` URI string. Unwrapping the
/// transport - base64, a URI header, a fetched body - belongs to the wire
/// adapter, which already has to tell those apart to reject remote URLs; making
/// every caller re-encode bytes into a string so this function can decode them
/// again would be pure waste.
///
/// The tower supplies `spec`. The geometry is only half of it: the resize rule
/// and the normalization statistics differ between families that agree on the
/// patch size, so a preprocessor that took only a geometry would have to guess
/// the rest from the architecture string.
///
/// # Errors
///
/// [`VisionError::EmptyPayload`] for no bytes; and [`VisionError::ImageDecode`]
/// when the bytes are not a decodable picture or declare a size past the
/// decoder limits, with the decoder's own complaint as the error source. A
/// geometry that cannot drive a preprocessor is refused when the spec is built,
/// not here.
pub fn preprocess_image(
    image: &[u8],
    spec: &PreprocessSpec,
    limits: ImageLimits,
) -> Result<VisionItem, VisionError> {
    let geom = spec.geometry;
    let decoded = decode_image(image, limits)?;
    let (th, tw) = target_size_for(decoded.height(), decoded.width(), spec, limits.max_pixels());
    let resized = image::imageops::resize(&decoded, tw, th, filter(spec.resample));

    let ps = geom.patch_size();
    let grid_h = side_as_usize(th) / ps;
    let grid_w = side_as_usize(tw) / ps;
    // The still occupies every temporal slot of its single group. This is the
    // one place the replication happens; the video path passes distinct frames
    // into the same function.
    let frames: Vec<&RgbImage> =
        std::iter::repeat_n(&resized, geom.temporal_patch_size()).collect();
    let pixels = pack_group(&frames, spec, grid_h, grid_w);
    VisionItem::new(
        pixels.into_boxed_slice(),
        geom.patch_dim(),
        1,
        grid_h,
        grid_w,
    )
}

/// Decodes bytes into RGB, under the caller's limits and with the orientation
/// tag applied.
///
/// ## The orientation tag is applied, and that is not cosmetic
///
/// A camera writes the sensor's raw pixels and records how to turn them upright
/// in a metadata tag rather than rotating the data, so a phone photograph is
/// very often stored sideways. Decoding without the tag hands the model an
/// image a quarter turn from how its owner saw it - and nothing errors and
/// nothing looks broken, the model simply answers about a sideways picture.
/// Every viewer the user compares against honours the tag, so ignoring it makes
/// "what the model saw" and "what the user saw" silently disagree on a large
/// share of real photographs.
///
/// The tag lives on the DECODER and is gone once the pixels are out, which is
/// why this reads it before finishing the decode. A format that carries no
/// orientation reports no transform, so this is a no-op for PNG and for any
/// untagged JPEG.
fn decode_image(bytes: &[u8], limits: ImageLimits) -> Result<RgbImage, VisionError> {
    if bytes.is_empty() {
        return Err(VisionError::EmptyPayload);
    }
    let format = image::guess_format(bytes).unwrap_or(ImageFormat::Jpeg);
    // Decoded through a reader rather than the convenience free function so
    // the limits are ours. The free function is not unlimited, but it sets no
    // DIMENSION cap at all, and its allocation cap is documented as advisory.
    // The dimension limit is what rejects a declared 65535x65535 from the
    // header, before any buffer is reserved.
    let mut reader = ImageReader::new(std::io::Cursor::new(bytes));
    reader.set_format(format);
    let mut decode_limits = Limits::default();
    decode_limits.max_image_width = Some(DECODE_MAX_SIDE);
    decode_limits.max_image_height = Some(DECODE_MAX_SIDE);
    decode_limits.max_alloc = Some(limits.max_decoded_bytes());
    reader.limits(decode_limits);

    let mut decoder = reader
        .into_decoder()
        .map_err(|source| VisionError::ImageDecode { source })?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut picture = DynamicImage::from_decoder(decoder)
        .map_err(|source| VisionError::ImageDecode { source })?;
    if orientation != image::metadata::Orientation::NoTransforms {
        // Worth a line: it changes what the model sees and is invisible in the
        // output, so if it ever fires wrongly this is the only trace of it.
        tracing::debug!(?orientation, "applying the image's orientation tag");
        picture.apply_orientation(orientation);
    }
    Ok(picture.to_rgb8())
}

/// Chooses the resized `(height, width)`, in pixels, under the spec's rule.
///
/// Whichever rule chose the size, the caller's `max_pixels` is a HARD cap
/// applied afterwards: it is a per-deployment limit on what one request may
/// cost, so a tower whose own rule asks for more is shrunk rather than
/// obeyed. That is the one thing the two rules have in common, and it is
/// written once here rather than inside each of them.
pub(crate) fn target_size_for(
    orig_h: u32,
    orig_w: u32,
    spec: &PreprocessSpec,
    max_pixels: Option<NonZeroU64>,
) -> (u32, u32) {
    let grid_unit = grid_unit_pixels(spec.geometry).max(1);
    let (target_h, target_w) = match spec.resize {
        ResizeRule::GridUnit => grid_unit_target(orig_h, orig_w, grid_unit, max_pixels),
        ResizeRule::PatchBudget { max_patches } => {
            patch_budget_target(orig_h, orig_w, spec.geometry, grid_unit, max_patches)
        }
    };
    shrink_under_cap(target_h, target_w, orig_h, orig_w, grid_unit, max_pixels)
}

/// The Qwen-family target: fit under the ceilings, snap, never grow.
///
/// The result satisfies, in this order:
///
/// - the area bound after grid snapping, when the caller declared one;
///   otherwise the long side is clamped to [`FALLBACK_MAX_DIM`];
/// - the long side never exceeds [`ABS_MAX_DIM`], bound or not;
/// - both sides are multiples of `grid_unit`;
/// - the aspect ratio is preserved to within one grid unit of rounding;
/// - nothing is ever scaled UP. A small image under a generous bound stays
///   small. That diverges from processors that honour a minimum-pixel setting
///   by upscaling, and it is deliberate: inventing pixels costs tokens and adds
///   no information.
fn grid_unit_target(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<NonZeroU64>,
) -> (u32, u32) {
    let long_side = f64::from(orig_h.max(orig_w)).max(1.0);
    let area = (f64::from(orig_h) * f64::from(orig_w)).max(1.0);
    let bound_scale = match max_pixels {
        // A declared AREA bound governs on its own; it does not intersect with
        // the fallback clamp. See `ImageLimits::with_max_pixels` for why.
        Some(pixels) => (widen(pixels.get()) / area).sqrt(),
        None => f64::from(FALLBACK_MAX_DIM) / long_side,
    };
    let abs_scale = f64::from(ABS_MAX_DIM) / long_side;
    let scale = bound_scale.min(abs_scale).min(1.0);

    let unit = f64::from(grid_unit.max(1));
    let target_h = round_u32(f64::from(orig_h) * scale / unit).max(1) * grid_unit.max(1);
    let target_w = round_u32(f64::from(orig_w) * scale / unit).max(1) * grid_unit.max(1);
    (target_h, target_w)
}

/// transformers' `get_aspect_ratio_preserving_size`, to the letter.
///
/// The budget is on PATCHES, not on pixels, and it is spent: a picture smaller
/// than the budget allows is scaled UP to fill it. That is the deliberate
/// opposite of [`grid_unit_target`], and it is why [`ABS_MAX_DIM`] is not
/// applied here - the budget already bounds the area, and the zero-side
/// fallback bounds the long side at `(max_patches / pool^2) * side_mult`, so
/// there is no unbounded direction for the ceiling to guard.
///
/// See [`ResizeRule::PatchBudget`] for the quoted fallback rule and its source.
/// `side_mult` is `grid_unit_pixels(geom).max(1)`, computed once by
/// [`target_size_for`] - which needs the same floored value for the other
/// rule and for the cap - rather than re-derived here.
fn patch_budget_target(
    orig_h: u32,
    orig_w: u32,
    geom: PatchGeometry,
    side_mult: u32,
    max_patches: u32,
) -> (u32, u32) {
    let patch = u64::try_from(geom.patch_size()).unwrap_or(1).max(1);
    let total_px = (f64::from(orig_h) * f64::from(orig_w)).max(1.0);
    let target_px = widen(
        u64::from(max_patches)
            .saturating_mul(patch)
            .saturating_mul(patch),
    );
    let factor = (target_px / total_px).sqrt();

    let unit = f64::from(side_mult);
    let mut target_h = floor_u32(f64::from(orig_h) * factor / unit).saturating_mul(side_mult);
    let mut target_w = floor_u32(f64::from(orig_w) * factor / unit).saturating_mul(side_mult);
    if target_h != 0 && target_w != 0 {
        return (target_h, target_w);
    }

    // The two ideal sides multiply out to the whole budget, so they cannot both
    // fall under one unit for a frame with pixels in it; the reference raises
    // on that case and this returns the smallest representable target instead.
    // A decoded frame that reached here has pixels, and a request task is never
    // the right place to discover otherwise.
    let pool = u32::try_from(geom.spatial_merge_size()).unwrap_or(1).max(1);
    let max_side = (max_patches / pool.saturating_mul(pool))
        .saturating_mul(side_mult)
        .max(side_mult);
    if target_h == 0 && target_w == 0 {
        return (side_mult, side_mult);
    }
    if target_h == 0 {
        target_h = side_mult;
        target_w = (orig_w / orig_h.max(1))
            .saturating_mul(side_mult)
            .clamp(side_mult, max_side);
    } else {
        target_w = side_mult;
        target_h = (orig_h / orig_w.max(1))
            .saturating_mul(side_mult)
            .clamp(side_mult, max_side);
    }
    (target_h, target_w)
}

/// Brings a chosen target back under the caller's area cap, a grid unit at a
/// time.
///
/// Rounding each axis to a grid line can push both of them past the continuous
/// area scale, so a declared hard cap has to survive the quantisation step as
/// well as the scaling one. Shrink a grid unit at a time, taking whichever axis
/// leaves the closer source aspect ratio. One grid cell is the smallest
/// representable target, so a bound below that is normalized to the unavoidable
/// minimum rather than refused.
fn shrink_under_cap(
    mut target_h: u32,
    mut target_w: u32,
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<NonZeroU64>,
) -> (u32, u32) {
    let Some(max_pixels) = max_pixels else {
        return (target_h, target_w);
    };
    let unit_px = u64::from(grid_unit.max(1));
    let max_area = max_pixels.get().max(unit_px * unit_px);
    let area_of = |h: u32, w: u32| u64::from(h) * u64::from(w);
    let aspect_error = |h: u32, w: u32| {
        if orig_h == 0 || h == 0 {
            0.0
        } else {
            ((f64::from(w) / f64::from(h)) - (f64::from(orig_w) / f64::from(orig_h))).abs()
        }
    };
    while area_of(target_h, target_w) > max_area {
        let shorter_h = target_h.checked_sub(grid_unit).filter(|&h| h >= grid_unit);
        let shorter_w = target_w.checked_sub(grid_unit).filter(|&w| w >= grid_unit);
        match (shorter_h, shorter_w) {
            (Some(h), Some(w)) => {
                // A tie is broken toward the larger surviving area, so the
                // shrink loop never gives up more resolution than the cap asks
                // for. Compared by total order rather than by `==`, which keeps
                // the tie-break deterministic even for a degenerate ratio.
                let take_h = match aspect_error(h, target_w).total_cmp(&aspect_error(target_h, w)) {
                    std::cmp::Ordering::Less => true,
                    std::cmp::Ordering::Equal => area_of(h, target_w) >= area_of(target_h, w),
                    std::cmp::Ordering::Greater => false,
                };
                if take_h {
                    target_h = h;
                } else {
                    target_w = w;
                }
            }
            (Some(h), None) => target_h = h,
            (None, Some(w)) => target_w = w,
            (None, None) => break,
        }
    }
    (target_h, target_w)
}

/// Packs one temporal group of already-resized frames into patch rows.
///
/// `frames` holds exactly `temporal_patch_size` images, all sized
/// `grid_h * patch_size` by `grid_w * patch_size`. One patch row is
/// `temporal_patch_size * patch^2 * 3` values, and the offset within it is
/// `t * (ps^2 * 3) + y * (ps * 3) + x * 3 + c` - **channel fastest**.
///
/// That ordering is not a choice. It is the axis order of the patch projection
/// in the checkpoint this reads: the tensor is written `[out, t, y, x, c]`, so a
/// patch row has to arrive with its channels innermost or the projection pairs
/// every value with the wrong weight. The two orders are the same numbers in a
/// different arrangement, which is why nothing else catches this - every shape
/// agrees, every kernel launches, and the model describes a scrambled picture.
///
/// An earlier version of this function emitted channel-slowest, which is the
/// convention a framework-native export of the same layer would use. If a
/// checkpoint ever ships that way, the permutation belongs in the loader for
/// that one tensor, not here: this function's contract is the row order the
/// encoder multiplies, and there is only one of those.
///
/// Pixels are read out of the frame's raw byte buffer rather than through
/// per-pixel accessors: the loop touches every byte of every frame once, and
/// the bounds check per component is the whole cost of the inner loop.
pub(crate) fn pack_group(
    frames: &[&RgbImage],
    spec: &PreprocessSpec,
    grid_h: usize,
    grid_w: usize,
) -> Vec<f32> {
    let geom = spec.geometry;
    let ps = geom.patch_size();
    let tp = geom.temporal_patch_size();
    let patch_dim = geom.patch_dim();
    let stride = grid_w * ps * 3;
    let mut pixels = vec![0.0f32; grid_h * grid_w * patch_dim];
    for ph in 0..grid_h {
        for pw in 0..grid_w {
            let patch_base = (ph * grid_w + pw) * patch_dim;
            for (t, frame) in frames.iter().take(tp).enumerate() {
                let raw = frame.as_raw();
                let time_base = patch_base + t * ps * ps * 3;
                for py in 0..ps {
                    let row_base = (ph * ps + py) * stride + pw * ps * 3;
                    let out_row = time_base + py * ps * 3;
                    for px in 0..ps {
                        for (c, (&mean, &deviation)) in
                            spec.mean.iter().zip(spec.std.iter()).enumerate()
                        {
                            let byte = raw.get(row_base + px * 3 + c).copied().unwrap_or_default();
                            let value = f32::from(byte) / 255.0;
                            pixels[out_row + px * 3 + c] = (value - mean) / deviation;
                        }
                    }
                }
            }
        }
    }
    pixels
}

/// The `image` filter a [`Resample`] names.
///
/// Total by construction: the enum holds the one cubic this crate can run, and
/// the mapping lives here so the resize calls in both preprocessing paths read
/// the same value out of the spec instead of each naming a filter.
pub(crate) const fn filter(resample: Resample) -> image::imageops::FilterType {
    match resample {
        Resample::CatmullRom => image::imageops::FilterType::CatmullRom,
    }
}

/// The grid unit as a pixel count.
///
/// `PatchGeometry` bounds both of its factors, so the product is far inside
/// 32 bits and the saturating arm is unreachable; it saturates rather than
/// panicking because a panic inside a request task is never the better failure.
pub(crate) fn grid_unit_pixels(geom: PatchGeometry) -> u32 {
    u32::try_from(geom.grid_unit()).unwrap_or(u32::MAX)
}

/// Narrows an image side to a `usize`.
///
/// Sides here are bounded by [`DECODE_MAX_SIDE`] long before this runs and this
/// crate does not build for a pointer narrower than 32 bits, so the saturating
/// arm is unreachable.
pub(crate) fn side_as_usize(side: u32) -> usize {
    usize::try_from(side).unwrap_or(usize::MAX)
}

/// Widens an exact count into the float domain the scale arithmetic works in.
///
/// Counts here are pixel areas and frame counts, both far below the 53-bit
/// range where a `f64` still represents every integer exactly.
#[expect(
    clippy::cast_precision_loss,
    reason = "areas and frame counts stay far below f64's exact-integer range"
)]
pub(crate) fn widen(count: u64) -> f64 {
    count as f64
}

/// Rounds a non-negative scale result DOWN to a `u32`, saturating at both ends.
///
/// The patch-budget rule floors rather than rounds, and the difference is not a
/// detail: rounding up there would spend more than the budget allows, which the
/// reference treats as an error rather than as a near miss.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped into the u32 range on the lines above, so the conversion is exact"
)]
pub(crate) fn floor_u32(value: f64) -> u32 {
    let floored = value.floor();
    if !floored.is_finite() || floored <= 0.0 {
        return 0;
    }
    if floored >= f64::from(u32::MAX) {
        return u32::MAX;
    }
    floored as u32
}

/// Rounds a non-negative scale result to the nearest `u32`, saturating at both
/// ends so a degenerate input produces a clamped size rather than a panic.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped into the u32 range on the lines above, so the conversion is exact"
)]
pub(crate) fn round_u32(value: f64) -> u32 {
    let rounded = value.round();
    if !rounded.is_finite() || rounded <= 0.0 {
        return 0;
    }
    if rounded >= f64::from(u32::MAX) {
        return u32::MAX;
    }
    rounded as u32
}

#[cfg(test)]
#[path = "image_tests.rs"]
mod tests;

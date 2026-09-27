//! Tests for still-image preprocessing.
//!
//! The sizing arithmetic gets tested as arithmetic - directly, at its
//! boundaries - rather than only through a decoded fixture, because the target
//! size is what decides an image's token cost and a fixture can only ever
//! sample one point of it.

use std::num::NonZeroU64;

use super::{ABS_MAX_DIM, DECODE_MAX_SIDE, decode_image, preprocess_image, target_size_for};
use crate::config::{ImageLimits, PreprocessSpec};
use crate::error::VisionError;
use crate::testing::{chain, encoded, png, vision_config};

/// The shipped grid unit: a 16-pixel patch under a 2x2 merger.
const GU: u32 = 32;

/// Gemma 4's grid unit: a 16-pixel patch under a 3x3 pooler.
const GEMMA_UNIT: u32 = 48;

/// An area bound, as the sizing function takes it.
fn bound(pixels: u64) -> Option<NonZeroU64> {
    NonZeroU64::new(pixels)
}

/// The Qwen-family spec the recorded checkpoint implies: the grid-unit rule
/// over the geometry in `config.json`.
fn qwen() -> PreprocessSpec {
    PreprocessSpec::from_vision_config(&vision_config())
        .expect("the recorded geometry drives a preprocessor")
}

/// The grid-unit target for the recorded checkpoint's geometry.
///
/// Every sizing case below goes through the spec rather than through a bare
/// grid unit, because the spec is what the callers now hand over and a rule
/// that only ever runs from a hand-built argument is a rule nothing exercises.
fn qwen_size(orig_h: u32, orig_w: u32, max_pixels: Option<NonZeroU64>) -> (u32, u32) {
    target_size_for(orig_h, orig_w, &qwen(), max_pixels)
}

/// The patch-budget target under Gemma 4's spec.
fn gemma_size(orig_h: u32, orig_w: u32, max_pixels: Option<NonZeroU64>) -> (u32, u32) {
    target_size_for(orig_h, orig_w, &PreprocessSpec::gemma4(), max_pixels)
}

/// Merged vision tokens the language model sees for a target of this size.
fn merged_tokens(h: u32, w: u32) -> u32 {
    (h / 16) * (w / 16) / 4
}

// -- sizing -----------------------------------------------------------------

#[test]
fn a_small_unbounded_image_only_moves_to_the_nearest_grid() {
    assert_eq!(qwen_size(100, 150, None), (96, 160));
}

#[test]
fn an_unbounded_large_image_uses_the_fallback_long_side() {
    assert_eq!(qwen_size(2000, 3000, None), (864, 1280));
}

#[test]
fn a_square_area_bound_snaps_to_its_exact_grid() {
    assert_eq!(qwen_size(1254, 1254, bound(512 * 512)), (512, 512));
}

/// Non-regression: a checkpoint shipping no preprocessor config must behave
/// exactly as it did before area bounds existed.
#[test]
fn no_declared_bound_keeps_the_fallback_clamp() {
    let (h, w) = qwen_size(1344, 1344, None);
    assert_eq!((h, w), (1280, 1280));
    assert_eq!(merged_tokens(h, w), 1600);
}

/// A declared bound must be able to RAISE the resolution past the fallback
/// clamp. Intersecting the two - which is the intuitive implementation - meant
/// a tower permitting 4096x4096 could never exceed 1280 on the long side, so
/// its own declared bound could only ever lower the resolution.
#[test]
fn a_declared_area_bound_raises_above_the_fallback() {
    let (h, w) = qwen_size(2048, 2048, bound(16_777_216));
    assert_eq!(
        (h, w),
        (2048, 2048),
        "2048x2048 sits inside a 4096x4096 bound"
    );
    assert!(
        h > 1280,
        "a declared bound must be able to raise past the fallback"
    );
}

#[test]
fn a_declared_area_bound_still_downscales_when_exceeded() {
    let (h, w) = qwen_size(8192, 8192, bound(16_777_216));
    assert!(u64::from(h) * u64::from(w) <= 16_777_216);
    assert!(
        h > 1280,
        "downscaled to the bound, not to the fallback clamp"
    );
}

/// The continuous scale can land just below a grid line on BOTH axes, and
/// rounding each to nearest then puts the area back over a cap the operator
/// declared as hard. The shrink loop is what survives that quantisation.
#[test]
fn a_declared_area_bound_survives_grid_rounding() {
    let max_pixels = 512 * 512;
    let (h, w) = qwen_size(85, 2740, bound(max_pixels));
    assert_eq!((h, w), (96, 2720));
    assert!(u64::from(h) * u64::from(w) <= max_pixels);
}

#[test]
fn an_operator_override_can_lower_below_the_fallback() {
    let (h, w) = qwen_size(1344, 1344, bound(256 * 256));
    assert!(u64::from(h) * u64::from(w) <= 256 * 256);
    assert!(h < 1280);
}

/// An area bound alone permits an unbounded long side, because a 1xN strip
/// satisfies any area. The absolute ceiling is the guard.
#[test]
fn the_absolute_ceiling_bounds_a_pathological_aspect_ratio() {
    let (_h, w) = qwen_size(32, 1_000_000, bound(16_777_216));
    assert!(
        w <= ABS_MAX_DIM,
        "long side {w} escaped the absolute ceiling"
    );
}

/// A tiny image under a huge bound stays tiny. This diverges from processors
/// that honour a minimum-pixel setting by scaling up, and the divergence is
/// deliberate - but it must not drift silently.
#[test]
fn a_small_image_is_never_scaled_up_to_a_large_declared_bound() {
    assert_eq!(qwen_size(64, 64, bound(16_777_216)), (64, 64));
}

#[test]
fn sides_are_always_grid_multiples() {
    for (oh, ow, max_pixels) in [
        (1080u32, 1920u32, None),
        (1080, 1920, bound(16_777_216)),
        (450, 300, None),
        (33, 33, bound(65_536)),
    ] {
        let (h, w) = qwen_size(oh, ow, max_pixels);
        assert_eq!(h % GU, 0, "h={h} is not a multiple of {GU}");
        assert_eq!(w % GU, 0, "w={w} is not a multiple of {GU}");
        assert!(h >= GU && w >= GU, "degenerate target {h}x{w}");
    }
}

#[test]
fn the_aspect_ratio_survives_grid_rounding() {
    let (h, w) = qwen_size(1080, 1920, bound(16_777_216));
    let want = 1920.0_f64 / 1080.0;
    let got = f64::from(w) / f64::from(h);
    assert!((got - want).abs() < 0.05, "aspect drifted: {got} vs {want}");
}

// -- the patch-budget rule --------------------------------------------------

/// The patch budget Gemma 4 declares: 280 soft tokens under a 3x3 pooler.
const GEMMA_PATCHES: u64 = 2520;

/// The spec's normalization statistics, channel by channel.
///
/// Compared with a tolerance rather than by array equality, which is what
/// `float_cmp` asks for even when both sides are written as literals.
fn assert_affine(spec: &PreprocessSpec, mean: [f32; 3], std: [f32; 3]) {
    for (channel, (&got, want)) in spec.mean.iter().zip(mean).enumerate() {
        assert!(
            (got - want).abs() < f32::EPSILON,
            "mean[{channel}] is {got}"
        );
    }
    for (channel, (&got, want)) in spec.std.iter().zip(std).enumerate() {
        assert!((got - want).abs() < f32::EPSILON, "std[{channel}] is {got}");
    }
}

/// Patches a target of this size costs, which is what the budget bounds.
fn patches(h: u32, w: u32) -> u64 {
    u64::from(h / 16) * u64::from(w / 16)
}

/// Sizes taken from `get_aspect_ratio_preserving_size` in transformers 5.17.0,
/// `models/gemma4/image_processing_gemma4.py`, run against the same inputs.
///
/// These are the numbers the reference produces, not the numbers the plan
/// predicted: the 1000x1000 case in particular scales to 768 rather than 720,
/// because `sqrt(645120 / 1e6) * 1000 / 48` is 16.73 and floors to 16.
#[test]
fn patch_budget_matches_transformers() {
    // factor = sqrt(2520 * 16^2 / (640 * 480)) ~ 1.4491, so the ideal sides are
    // 695.6 x 927.4 and each floors to a whole 48-pixel unit.
    assert_eq!(gemma_size(480, 640, None), (672, 912));
    assert_eq!(gemma_size(1000, 1000, None), (768, 768));
    assert_eq!(gemma_size(100, 4000, None), (96, 5040));
    assert_eq!(gemma_size(4000, 100, None), (5040, 96));
}

/// The rule scales a small image UP to fill its budget, which is exactly what
/// the grid-unit rule refuses to do. The two rules disagreeing here is the
/// whole reason there are two of them.
#[test]
fn the_patch_budget_rule_upscales_where_the_grid_unit_rule_does_not() {
    assert_eq!(gemma_size(64, 64, None), (768, 768));
    assert_eq!(qwen_size(64, 64, None), (64, 64));
}

/// A strip whose short side does not reach one whole 48-pixel unit after
/// scaling. Quoting the reference, `image_processing_gemma4.py`:
///
/// ```text
/// max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
/// if target_height == 0:
///     target_height = side_mult
///     target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
/// elif target_width == 0:
///     target_width = side_mult
///     target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
/// ```
#[test]
fn a_patch_budget_strip_falls_back_to_one_unit_on_the_short_side() {
    // 4000x10: the ideal height is 40.2 px, less than one 48-pixel unit.
    assert_eq!(gemma_size(10, 4000, None), (48, 13_440));
    assert_eq!(gemma_size(4000, 10, None), (13_440, 48));
    // The fallback cannot smuggle an over-budget target past the cap: the
    // reference raises in that case, and the arithmetic here must make the
    // raise unreachable rather than leave it to be caught later.
    assert_eq!(patches(48, 13_440), GEMMA_PATCHES);
}

/// The budget is a budget: nothing the rule returns may cost more patches than
/// it allows, at any aspect ratio.
#[test]
fn the_patch_budget_target_never_exceeds_the_budget() {
    for (oh, ow) in [
        (480u32, 640u32),
        (1000, 1000),
        (1, 1),
        (10, 4000),
        (4000, 10),
        (3, 1),
        (2160, 3840),
        (7, 9),
    ] {
        let (h, w) = gemma_size(oh, ow, None);
        assert!(
            patches(h, w) <= GEMMA_PATCHES,
            "{oh}x{ow} became {h}x{w}, which is {} patches",
            patches(h, w)
        );
        assert_eq!(h % GEMMA_UNIT, 0, "h={h} is not a multiple of {GEMMA_UNIT}");
        assert_eq!(w % GEMMA_UNIT, 0, "w={w} is not a multiple of {GEMMA_UNIT}");
        assert!(h >= GEMMA_UNIT && w >= GEMMA_UNIT, "degenerate {h}x{w}");
    }
}

/// The operator's area cap outranks the budget. A rule that honoured its own
/// budget and ignored the cap would let a per-deployment limit be overridden by
/// whichever tower happened to be loaded.
#[test]
fn a_declared_area_bound_still_caps_the_patch_budget() {
    let max_pixels = 256 * 256;
    let (h, w) = gemma_size(480, 640, bound(max_pixels));
    assert!(
        u64::from(h) * u64::from(w) <= max_pixels,
        "{h}x{w} escaped the declared cap"
    );
    assert_eq!(h % GEMMA_UNIT, 0);
    assert_eq!(w % GEMMA_UNIT, 0);
}

/// End to end: the grid and the pad run a Gemma tower will be handed for a
/// 640x480 picture. The pad run is what the prompt reserves and what the
/// encoder must return, so it is pinned rather than derived twice.
#[test]
fn a_gemma_image_reaches_the_documented_grid_and_pad_run() {
    let spec = PreprocessSpec::gemma4();
    let item = preprocess_image(&png(640, 480, [12, 34, 56]), &spec, ImageLimits::default())
        .expect("a 640x480 PNG preprocesses");
    assert_eq!(item.grid(), (1, 42, 57));
    assert_eq!(item.pad_count(3), 266);
    // Gemma's temporal patch size is one, so a patch row is a single frame.
    assert_eq!(item.row_len(), 3 * 16 * 16);
}

/// Gemma maps the byte range onto [-1, 1] here, because its reference does it
/// one step later - inside the tower, at `modeling_gemma4.py:616`, and at
/// `mlx_vlm/models/gemma4/vision.py:319` and `:328` - and shiso's tower has no
/// such step. `processor_config.json` saying `do_normalize: false` is therefore
/// not the whole story; see [`PreprocessSpec::gemma4`].
///
/// Pinned as pixels rather than only as the two constants: `2 * (x - 0.5)` and
/// `(x - 0.5) / 0.5` have to agree on the values the tower multiplies, and that
/// is what this checks.
#[test]
fn the_gemma_spec_maps_pixels_onto_the_signed_unit_range() {
    let spec = PreprocessSpec::gemma4();
    assert_affine(&spec, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    let item = preprocess_image(&png(96, 96, [255, 0, 0]), &spec, ImageLimits::default())
        .expect("a solid red PNG preprocesses");
    let row = item.row(0).expect("the first row exists");
    for (offset, value) in row.iter().enumerate() {
        // What the reference's `2 * (pixel_values - 0.5)` produces for the same
        // bytes: red saturated, the other two channels at zero.
        let rescaled = if offset % 3 == 0 { 1.0f32 } else { 0.0 };
        let want = 2.0 * (rescaled - 0.5);
        assert!(
            (value - want).abs() < f32::EPSILON,
            "value {offset} is {value}, want {want}"
        );
    }
}

/// The Qwen spec must still map the byte range onto [-1, 1]; the committed
/// vision goldens were recorded under it.
#[test]
fn the_qwen_spec_keeps_its_normalization_and_filter() {
    let spec = qwen();
    assert_affine(&spec, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    assert_eq!(spec.resample, crate::config::Resample::CatmullRom);
    assert_eq!(spec.resize, crate::config::ResizeRule::GridUnit);
}

/// The grid-unit rule is untouched by the arrival of a second rule. Restated as
/// a table because the qwen35 vision goldens are recorded against exactly these
/// sizes, and a silent change to any of them re-records a golden instead of
/// failing.
#[test]
fn grid_unit_rule_is_unchanged() {
    for (oh, ow, cap, want) in [
        (100u32, 150u32, None, (96u32, 160u32)),
        (2000, 3000, None, (864, 1280)),
        (1344, 1344, None, (1280, 1280)),
        (64, 64, None, (64, 64)),
        (1254, 1254, bound(512 * 512), (512, 512)),
        (85, 2740, bound(512 * 512), (96, 2720)),
        (2048, 2048, bound(16_777_216), (2048, 2048)),
        (64, 64, bound(16_777_216), (64, 64)),
    ] {
        assert_eq!(qwen_size(oh, ow, cap), want, "{oh}x{ow} under {cap:?}");
    }
}

// -- geometry guards --------------------------------------------------------

/// Config parsing reports a missing key as zero, so a checkpoint whose vision
/// block omits `patch_size` arrives here as a divisor. Dividing by it panics
/// inside a request task, which is a remote crash from a request body.
#[test]
fn a_zero_patch_size_is_a_named_error_not_a_divide_by_zero() {
    let mut vcfg = vision_config();
    vcfg.patch_size = 0;
    let error = PreprocessSpec::from_vision_config(&vcfg)
        .expect_err("a zero patch size cannot drive a preprocessor");
    assert!(error.to_string().contains("patch_size"), "{error}");
}

/// The grid unit is the other divisor, and a zero here makes it zero in the
/// scale computation, producing a 0x0 target.
#[test]
fn a_zero_spatial_merge_size_is_a_named_error() {
    let mut vcfg = vision_config();
    vcfg.spatial_merge_size = 0;
    let error = PreprocessSpec::from_vision_config(&vcfg)
        .expect_err("a zero merge size cannot drive a preprocessor");
    assert!(error.to_string().contains("spatial_merge_size"), "{error}");
}

/// A zero temporal patch size collapses the row length to zero, yielding an
/// empty pixel buffer the encoder would then have to catch.
#[test]
fn a_zero_temporal_patch_size_is_a_named_error() {
    let mut vcfg = vision_config();
    vcfg.temporal_patch_size = 0;
    let error = PreprocessSpec::from_vision_config(&vcfg)
        .expect_err("a zero temporal patch size cannot drive a preprocessor");
    assert!(error.to_string().contains("temporal_patch_size"), "{error}");
}

/// The guards must not have broken the working path.
#[test]
fn a_red_pixel_produces_the_exact_grid_and_normalized_channel_layout() {
    let vcfg = vision_config();
    let item = preprocess_image(&png(1, 1, [255, 0, 0]), &qwen(), ImageLimits::default())
        .expect("a 1x1 PNG preprocesses");
    let ps = usize::try_from(vcfg.patch_size).expect("a sane patch size");
    let tp = usize::try_from(vcfg.temporal_patch_size).expect("a sane temporal patch size");
    let row_len = 3 * tp * ps * ps;

    assert_eq!(
        item.grid(),
        (1, 2, 2),
        "a 1x1 source snaps up to one grid cell"
    );
    assert_eq!(item.row_len(), row_len);
    assert_eq!(item.as_flat().len(), 2 * 2 * row_len);

    // Red maps to +1 on the red channel and -1 on the other two. Channel is the
    // FASTEST axis, so the three arrive interleaved at every pixel rather than as
    // three contiguous spans - which is the whole point of this assertion: the
    // other arrangement holds exactly the same numbers, so a test that only
    // counted them would pass against a row the projection reads backwards.
    for index in 0..item.rows() {
        let row = item.row(index).expect("every row is addressable");
        for (offset, value) in row.iter().enumerate() {
            let want = if offset % 3 == 0 { 1.0 } else { -1.0 };
            assert!(
                (value - want).abs() < f32::EPSILON,
                "row {index} value {offset} is {value}, want {want}: channels must interleave"
            );
        }
    }
}

/// A still fills every temporal slot with the same pixels. That is the
/// behaviour a video must NOT have, and the two are checked against each other
/// so neither can quietly adopt the other's.
#[test]
fn a_still_replicates_itself_across_the_temporal_axis() {
    let vcfg = vision_config();
    let item = preprocess_image(&png(8, 8, [10, 200, 30]), &qwen(), ImageLimits::default())
        .expect("a small PNG preprocesses");
    let ps = usize::try_from(vcfg.patch_size).expect("a sane patch size");
    let row = item.row(0).expect("the first row exists");
    // Time is the slowest axis of a row, so the second temporal slot starts a
    // whole patch plane in - `patch^2` pixels of three interleaved channels.
    let slot = ps * ps * 3;
    for offset in 0..slot {
        assert!(
            (row[offset] - row[slot + offset]).abs() < f32::EPSILON,
            "temporal slot {offset} differs; a still must replicate"
        );
    }
}

// -- decode refusals --------------------------------------------------------

#[test]
fn an_empty_payload_is_refused_before_decoding() {
    assert!(matches!(
        preprocess_image(&[], &qwen(), ImageLimits::default()),
        Err(VisionError::EmptyPayload)
    ));
}

#[test]
fn undecodable_bytes_fail_as_a_decode_error() {
    let error = preprocess_image(b"not-an-image", &qwen(), ImageLimits::default())
        .expect_err("arbitrary bytes are not a picture");
    assert!(matches!(error, VisionError::ImageDecode { .. }), "{error}");
}

/// A PNG whose header declares 65535x65535 - about 13 GB of RGB - in a file of
/// seventy bytes. The dimension limit rejects it FROM THE HEADER, before any
/// pixel buffer is reserved.
///
/// The assertion names the limit message specifically rather than accepting any
/// error: the fixture is also a truncated PNG, so a decoder with no limits at
/// all would still fail, just later and after reserving the buffer. Only this
/// message proves the header check fired first.
#[test]
fn decode_bomb_dimensions_are_rejected_before_allocation() {
    let mut png_bytes: Vec<u8> = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr: Vec<u8> = b"IHDR".to_vec();
    ihdr.extend_from_slice(&65535u32.to_be_bytes());
    ihdr.extend_from_slice(&65535u32.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let crc = {
        let mut c: u32 = 0xFFFF_FFFF;
        for &byte in &ihdr {
            c ^= u32::from(byte);
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    (c >> 1) ^ 0xEDB8_8320
                } else {
                    c >> 1
                };
            }
        }
        !c
    };
    png_bytes.extend_from_slice(&13u32.to_be_bytes());
    png_bytes.extend_from_slice(&ihdr);
    png_bytes.extend_from_slice(&crc.to_be_bytes());

    // The fixture must exceed the limit under test or it proves nothing.
    const { assert!(65535 > DECODE_MAX_SIDE) };
    let error = preprocess_image(&png_bytes, &qwen(), ImageLimits::default())
        .expect_err("a declared 65535x65535 must be refused");
    let text = chain(&error);
    assert!(text.contains("Image size exceeds limit"), "{text}");
}

// -- orientation ------------------------------------------------------------

/// Mean red channel over a pixel rectangle.
fn region_red(img: &image::RgbImage, xs: u32, ys: u32, xe: u32, ye: u32) -> f32 {
    let mut sum = 0f32;
    let mut count = 0f32;
    for y in ys..ye.min(img.height()) {
        for x in xs..xe.min(img.width()) {
            sum += f32::from(img.get_pixel(x, y)[0]);
            count += 1.0;
        }
    }
    sum / count.max(1.0)
}

/// A minimal big-endian metadata segment declaring orientation 6, "rotate a
/// quarter turn clockwise to display". Written by hand because every encoder
/// that would emit one also normalizes the pixels, which is the behaviour under
/// test.
const ORIENTATION_6: &[u8] = &[
    0xff, 0xe1, 0x00, 0x22, b'E', b'x', b'i', b'f', 0x00, 0x00, b'M', b'M', 0x00, 0x2a, 0x00, 0x00,
    0x00, 0x08, 0x00, 0x01, 0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
];

/// Splices the orientation segment in immediately after the start-of-image
/// marker.
fn with_orientation_tag(jpeg: &[u8]) -> Vec<u8> {
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(ORIENTATION_6);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// A half-red, half-blue JPEG, optionally carrying an orientation tag that says
/// "rotate a quarter turn clockwise to display".
fn split_jpeg(tagged: bool) -> Vec<u8> {
    let mut image = image::RgbImage::new(64, 64);
    for (_, y, pixel) in image.enumerate_pixels_mut() {
        *pixel = if y < 32 {
            image::Rgb([255, 0, 0])
        } else {
            image::Rgb([0, 0, 255])
        };
    }
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 100)
        .encode_image(&image)
        .expect("encoding the orientation fixture");
    if tagged {
        with_orientation_tag(&jpeg)
    } else {
        jpeg
    }
}

/// The untagged control: nothing to apply, so red stays on top. Every PNG and
/// every untagged JPEG must decode exactly as it did before orientation
/// handling existed.
#[test]
fn an_untagged_image_is_not_rotated() {
    let img = decode_image(&split_jpeg(false), ImageLimits::default()).expect("decode");
    let top = region_red(&img, 6, 3, 58, 22);
    let bottom = region_red(&img, 6, 42, 58, 61);
    assert!(
        top > 200.0 && bottom < 60.0,
        "expected red above blue; got top={top:.0} bottom={bottom:.0}"
    );
}

/// The tagged image must come out rotated. Orientation 6 moves the stored top
/// edge to the right, so the red half ends up on the RIGHT. Ignore the tag and
/// nothing errors and nothing looks broken - the model simply answers about a
/// sideways picture.
#[test]
fn a_tagged_image_is_rotated_upright() {
    let img = decode_image(&split_jpeg(true), ImageLimits::default()).expect("decode");
    let left = region_red(&img, 3, 6, 22, 58);
    let right = region_red(&img, 42, 6, 61, 58);
    assert!(
        right > 200.0 && left < 60.0,
        "orientation 6 should put red on the RIGHT; got left={left:.0} right={right:.0}"
    );
}

/// Rotation happens BEFORE the grid snap, so a landscape image tagged as
/// portrait is measured at its displayed shape. The asymmetric fixture makes an
/// ignored or late-applied tag swap the grid back instead of hiding behind
/// equal axes.
#[test]
fn rotation_precedes_grid_sizing() {
    let landscape = image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]));
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
        .encode_image(&landscape)
        .expect("encoding the asymmetric fixture");
    let tagged = with_orientation_tag(&jpeg);

    let decoded = decode_image(&tagged, ImageLimits::default()).expect("decode");
    assert_eq!((decoded.width(), decoded.height()), (32, 64));

    let item = preprocess_image(&tagged, &qwen(), ImageLimits::default())
        .expect("the tagged fixture preprocesses");
    assert_eq!(item.grid(), (1, 4, 2));
}

/// The bound carried in the limits must reach the sizing step; a bound that is
/// accepted and then ignored is worse than one that is refused.
#[test]
fn the_declared_limit_reaches_the_sizing_step() {
    let source = png(1024, 1024, [200, 100, 50]);
    let unbounded = preprocess_image(&source, &qwen(), ImageLimits::default()).expect("unbounded");
    let bounded = preprocess_image(
        &source,
        &qwen(),
        ImageLimits::default()
            .with_max_pixels(NonZeroU64::new(128 * 128).expect("a non-zero bound")),
    )
    .expect("bounded");
    let (_, big_h, _) = unbounded.grid();
    let (_, small_h, _) = bounded.grid();
    assert!(
        small_h < big_h,
        "the bound did not shrink {big_h} to {small_h}"
    );
}

/// WebP and BMP decode in process, like PNG: the same picture in each format
/// becomes the same item, so nothing downstream can tell which one was sent.
#[test]
fn webp_and_bmp_decode_to_the_same_item_as_png() {
    let rgb = [30, 144, 255];
    let want = preprocess_image(&png(64, 48, rgb), &qwen(), ImageLimits::default())
        .expect("a PNG preprocesses");
    for format in [image::ImageFormat::WebP, image::ImageFormat::Bmp] {
        let got = preprocess_image(
            &encoded(64, 48, rgb, format),
            &qwen(),
            ImageLimits::default(),
        )
        .unwrap_or_else(|error| panic!("{format:?} did not preprocess: {error}"));
        assert_eq!(got.grid(), want.grid(), "{format:?} grid");
        assert_eq!(got.as_flat(), want.as_flat(), "{format:?} pixels");
    }
}

//! Tests for frame sampling, container dispatch and temporal grouping.
//!
//! The sampling arithmetic is where a video's token count comes from, so it is
//! tested as arithmetic - directly, at the boundaries - rather than only
//! through a decoded fixture.

use std::num::NonZeroU64;

use super::{
    DEFAULT_FPS, DecodedClip, DecodedFrame, decode_clip, preprocess_video, sample_indices,
    sniff_container,
};
use crate::config::{ImageLimits, PreprocessSpec};
use crate::error::{Container, VisionError};
use crate::testing::{gif, vision_config};

/// The Qwen-family spec the recorded checkpoint implies.
fn qwen() -> PreprocessSpec {
    PreprocessSpec::from_vision_config(&vision_config())
        .expect("the recorded geometry drives a preprocessor")
}

/// Decodes a GIF and samples it to `target_fps`, which is the whole in-process
/// path a caller runs.
fn frames_of(bytes: &[u8], target_fps: f64, temporal_patch_size: usize) -> Vec<DecodedFrame> {
    decode_clip(bytes, target_fps, None)
        .expect("a generated GIF decodes")
        .into_sampled(target_fps, temporal_patch_size)
}

// -- sampling ---------------------------------------------------------------

/// The count must be a whole number of temporal groups whenever one full group
/// is achievable: a partial group has no representation in the patch tensor, so
/// an odd count at a temporal patch size of two is a latent shape bug.
///
/// The one exception is a clip with FEWER frames than a single group. There,
/// sampling reports honestly what it has and `preprocess_video` is what refuses
/// it - rounding one frame down to zero would turn a describable error into an
/// empty selection the caller has to reverse-engineer.
#[test]
fn the_frame_count_is_always_a_whole_number_of_groups() {
    const TP: usize = 2;
    for n in 1..64usize {
        for fps in [1.0f64, 2.0, 5.0, 7.5, 30.0] {
            let got = sample_indices(n, 30.0, fps, 4, 768, TP);
            if n < TP {
                assert_eq!(got.len(), n, "n={n}: too short for a group, report as-is");
                continue;
            }
            assert_eq!(
                got.len() % TP,
                0,
                "n={n} fps={fps} produced {} frames, not whole groups",
                got.len()
            );
            assert!(got.len() >= TP, "n={n} fps={fps} dropped below one group");
            assert!(got.len() <= n, "n={n} fps={fps} sampled more than it had");
        }
    }
}

#[test]
fn indices_are_in_range_and_strictly_increasing() {
    let got = sample_indices(100, 30.0, 2.0, 4, 768, 2);
    assert!(!got.is_empty());
    for pair in got.windows(2) {
        assert!(
            pair[0] < pair[1],
            "indices repeated or went backwards: {got:?}"
        );
    }
    assert!(*got.last().expect("non-empty") < 100);
}

/// A 30fps, 100-frame clip is 3.33 s; at 2fps that is about seven frames,
/// rounded down to six for three whole groups.
#[test]
fn a_clip_is_sampled_to_the_requested_rate() {
    let got = sample_indices(100, 30.0, 2.0, 4, 768, 2);
    assert_eq!(got.len(), 6, "got {got:?}");
    // Spread across the whole clip, not clustered at the start.
    assert_eq!(got.first().copied(), Some(0));
    assert_eq!(got.last().copied(), Some(99));
}

/// Never upsample. Asking for 30fps of a clip that has four frames returns
/// those four, not four repeated: repeats inflate the token count with no new
/// information, and the request would be charged for frames that do not exist.
#[test]
fn a_short_clip_is_never_padded_up_to_the_minimum() {
    assert_eq!(sample_indices(4, 2.0, 30.0, 16, 768, 2), vec![0, 1, 2, 3]);
}

#[test]
fn the_frame_ceiling_is_honoured() {
    assert_eq!(sample_indices(10_000, 30.0, 30.0, 4, 768, 2).len(), 768);
}

#[test]
fn an_inverted_frame_band_uses_the_ceiling() {
    let got = sample_indices(100, 30.0, 30.0, 16, 4, 2);
    assert_eq!(got.len(), 4);
    assert_eq!(got.first().copied(), Some(0));
    assert_eq!(got.last().copied(), Some(99));
}

/// A degenerate rate must not divide by zero and must not return nothing; it
/// falls back to the checkpoint default rather than failing the request.
#[test]
fn degenerate_rates_fall_back_rather_than_exploding() {
    for (native, target, valid_native, valid_target) in [
        (0.0f64, 1.0f64, DEFAULT_FPS, 1.0),
        (30.0, 0.0, 30.0, DEFAULT_FPS),
        (f64::NAN, 1.0, DEFAULT_FPS, 1.0),
        (30.0, f64::INFINITY, 30.0, DEFAULT_FPS),
    ] {
        assert_eq!(
            sample_indices(100, native, target, 4, 768, 2),
            sample_indices(100, valid_native, valid_target, 4, 768, 2),
            "native={native} target={target}"
        );
    }
}

#[test]
fn an_empty_clip_samples_to_nothing() {
    assert!(sample_indices(0, 30.0, 2.0, 4, 768, 2).is_empty());
}

/// A single frame cannot fill a two-frame group. Sampling reports what it has
/// and `preprocess_video` refuses it; both halves are checked so the two cannot
/// drift into disagreeing.
#[test]
fn one_frame_cannot_make_a_group() {
    let got = sample_indices(1, 30.0, 2.0, 4, 768, 2);
    assert_eq!(got.len(), 1, "sampling reports the single frame it has");
    assert!(got.len() < 2, "and it is short of one group");
}

// -- container dispatch -----------------------------------------------------

#[test]
fn an_animated_gif_decodes_to_its_frames_at_its_own_rate() {
    let clip = decode_clip(&gif(6, 64), 10.0, None).expect("decode");
    assert_eq!(clip.frames().len(), 6);
    // 100 ms per frame is 10 fps.
    assert!((clip.fps() - 10.0).abs() < 0.5, "fps was {}", clip.fps());
}

/// Magic bytes win over the declared type. A client that sends a GIF labelled
/// `video/mp4` still reaches the in-process decoder rather than needlessly
/// spawning a process.
#[test]
fn the_magic_bytes_decide_the_container() {
    assert_eq!(sniff_container(&gif(2, 8)), Container::Gif);
    assert_eq!(
        sniff_container(b"\x00\x00\x00\x20ftypmp42____"),
        Container::Mp4
    );
    assert_eq!(
        sniff_container(&[0x1a, 0x45, 0xdf, 0xa3, 0, 0]),
        Container::Matroska
    );
    assert_eq!(sniff_container(b"RIFFxxxxAVI "), Container::Avi);
    assert_eq!(sniff_container(b"nothing at all"), Container::Unknown);
}

#[test]
fn an_empty_payload_is_refused_before_dispatch() {
    assert!(matches!(
        decode_clip(&[], 10.0, None),
        Err(VisionError::EmptyPayload)
    ));
}

/// The container that needs the subprocess backend must be named, and so must
/// the flag that would allow it. "Not a decodable animation" leaves the
/// operator guessing whether the file is corrupt.
#[test]
fn a_container_needing_the_decode_chain_is_refused_by_name() {
    let error = decode_clip(b"\x00\x00\x00\x20ftypmp42____", 10.0, None)
        .expect_err("no policy means no decode");
    match &error {
        VisionError::VideoDecodeDisabled { container } => {
            assert_eq!(*container, Container::Mp4);
        }
        other => panic!("expected a policy refusal, got {other}"),
    }
    let text = error.to_string();
    assert!(text.contains("mp4"), "the container is not named: {text}");
    assert!(
        text.contains("--video-allow-decode"),
        "no remedy offered: {text}"
    );
}

#[test]
fn garbage_is_an_error_rather_than_a_panic() {
    assert!(decode_clip(b"GIF89a-but-not-really", 10.0, None).is_err());
    assert!(decode_clip(b"\x00\x01\x02\x03\x04", 10.0, None).is_err());
}

// -- grouping ---------------------------------------------------------------

/// The shape contract the encoder depends on: one row per patch per group,
/// each of `3 * temporal_patch_size * patch^2` values.
#[test]
fn grouping_produces_correctly_shaped_rows() {
    let frames = frames_of(&gif(8, 64), 10.0, 2);
    let item = preprocess_video(&frames, &qwen(), ImageLimits::default()).expect("preprocess");
    assert_eq!(
        item.grid(),
        (4, 4, 4),
        "8 frames at tp=2 is 4 groups of 64/16"
    );
    assert_eq!(item.row_len(), 3 * 2 * 16 * 16);
    assert_eq!(item.rows(), 4 * 4 * 4);
    assert_eq!(item.as_flat().len(), item.rows() * item.row_len());
}

/// Token accounting, which is what the prompt-side pad run must match.
#[test]
fn pad_count_is_groups_times_the_merged_plane() {
    let frames = frames_of(&gif(8, 64), 10.0, 2);
    let item = preprocess_video(&frames, &qwen(), ImageLimits::default()).expect("preprocess");
    assert_eq!(item.pad_count(2), 4 * 4);
    assert_eq!(item.pad_count(1), 4 * 16);
}

/// Distinct frames must land in distinct temporal slots. If grouping ever
/// duplicated one frame across the temporal axis - the way a STILL deliberately
/// does - a video would silently become a slideshow of doubled stills, and
/// every shape check above would still pass.
#[test]
fn the_frames_of_a_group_are_actually_different() {
    let frames = frames_of(&gif(4, 64), 10.0, 2);
    let item = preprocess_video(&frames, &qwen(), ImageLimits::default()).expect("preprocess");
    let row = item.row(0).expect("the first row exists");
    let plane = 16 * 16;
    assert!(
        (row[0] - row[plane]).abs() > f32::EPSILON,
        "the two temporal slots hold identical pixels; the frames were duplicated"
    );
}

/// Consecutive groups must differ too: group one must not be a copy of group
/// zero.
#[test]
fn consecutive_groups_hold_different_frames() {
    let frames = frames_of(&gif(4, 64), 10.0, 2);
    let item = preprocess_video(&frames, &qwen(), ImageLimits::default()).expect("preprocess");
    let (grid_t, grid_h, grid_w) = item.grid();
    assert_eq!(grid_t, 2);
    let first = item.row(0).expect("group 0 row 0");
    let second = item.row(grid_h * grid_w).expect("group 1 row 0");
    assert!(
        (first[0] - second[0]).abs() > f32::EPSILON,
        "group 1 repeated group 0"
    );
}

/// The area bound applies to video exactly as it does to stills. A video is far
/// more expensive per item, so a bound that silently did not apply here would
/// be the worse omission.
#[test]
fn the_area_bound_shrinks_the_grid_but_not_the_temporal_extent() {
    let frames = frames_of(&gif(4, 256), 10.0, 2);
    let big = preprocess_video(&frames, &qwen(), ImageLimits::default()).expect("unbounded");
    let small = preprocess_video(
        &frames,
        &qwen(),
        ImageLimits::default().with_max_pixels(NonZeroU64::new(64 * 64).expect("non-zero")),
    )
    .expect("bounded");
    let (big_t, big_h, _) = big.grid();
    let (small_t, small_h, _) = small.grid();
    assert!(
        small_h < big_h,
        "the bound did not shrink {big_h} to {small_h}"
    );
    assert_eq!(small_t, big_t, "the bound is spatial, not temporal");
}

// -- refusals ---------------------------------------------------------------

#[test]
fn a_single_frame_clip_is_refused_rather_than_treated_as_a_still() {
    let frames = frames_of(&gif(1, 64), 10.0, 2);
    let error = preprocess_video(&frames, &qwen(), ImageLimits::default())
        .expect_err("one frame is not a temporal group");
    assert!(
        matches!(
            error,
            VisionError::ShortClip {
                frames: 1,
                temporal_patch_size: 2
            }
        ),
        "{error}"
    );
}

#[test]
fn an_empty_frame_sequence_is_refused() {
    assert!(matches!(
        preprocess_video(&[], &qwen(), ImageLimits::default()),
        Err(VisionError::NoFrames)
    ));
}

/// Sizing is decided once from the first frame, so frames that disagree about
/// their size have no single grid. Silently resizing them all to the first
/// frame's shape would distort every later frame with nothing to say so.
#[test]
fn frames_that_disagree_about_their_size_are_refused() {
    let frames = vec![
        DecodedFrame::from_rgb8(32, 32, vec![0u8; 32 * 32 * 3]).expect("a square frame"),
        DecodedFrame::from_rgb8(32, 64, vec![0u8; 32 * 64 * 3]).expect("a taller frame"),
    ];
    let error = preprocess_video(&frames, &qwen(), ImageLimits::default())
        .expect_err("mismatched frames have no single grid");
    assert!(
        matches!(error, VisionError::FrameSizeMismatch { index: 1, .. }),
        "{error}"
    );
}

/// The geometry is checked when the spec is built, which is before any frame
/// can reach a divisor. Still asserted from the video side: the guard moved,
/// and a move is exactly when a guard goes missing.
#[test]
fn invalid_geometry_is_refused_before_dividing_by_it() {
    for mutate in [
        |c: &mut shiso_core::VisionConfig| c.temporal_patch_size = 0,
        |c: &mut shiso_core::VisionConfig| c.patch_size = 0,
        |c: &mut shiso_core::VisionConfig| c.spatial_merge_size = 0,
    ] {
        let mut vcfg = vision_config();
        mutate(&mut vcfg);
        assert!(PreprocessSpec::from_vision_config(&vcfg).is_err());
    }
}

#[test]
fn a_frame_whose_bytes_do_not_match_its_size_is_refused() {
    assert!(matches!(
        DecodedFrame::from_rgb8(8, 8, vec![0u8; 10]),
        Err(VisionError::FrameShape {
            expected: 192,
            got: 10,
            ..
        })
    ));
}

#[test]
fn a_clip_cannot_exist_without_frames() {
    assert!(matches!(
        DecodedClip::new(Vec::new(), 10.0),
        Err(VisionError::NoFrames)
    ));
}

/// A clip that declares a nonsense rate still samples, because the sampler
/// needs a defined divisor rather than a failed request.
#[test]
fn a_clip_normalizes_a_degenerate_rate() {
    let frame = DecodedFrame::from_rgb8(2, 2, vec![0u8; 12]).expect("a tiny frame");
    let clip = DecodedClip::new(vec![frame], f64::NAN).expect("one frame is a clip");
    assert!((clip.fps() - DEFAULT_FPS).abs() < f64::EPSILON);
}

/// Sampling moves the kept frames out and drops the rest, so the survivors are
/// the ones the sampler chose and nothing else is still resident.
#[test]
fn sampling_keeps_exactly_the_chosen_frames() {
    let clip = decode_clip(&gif(8, 16), 10.0, None).expect("decode");
    let expected = sample_indices(8, clip.fps(), 2.5, 4, 768, 2).len();
    assert_eq!(clip.into_sampled(2.5, 2).len(), expected);
}

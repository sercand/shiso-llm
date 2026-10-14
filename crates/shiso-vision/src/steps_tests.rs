//! The in-process steps over the committed container fixtures.
//!
//! The OS decoder cases open a real device session, so they are `#[ignore]`d and
//! macOS-only; everything else runs anywhere.

use super::*;
use crate::chain::VideoDecodePolicy;
#[cfg(feature = "decode-av1")]
use crate::chain::run_chain;
#[cfg(feature = "decode-av1")]
use crate::demux;
use crate::error::Container;
use crate::video::decode_clip;

const H264_MP4: &[u8] = include_bytes!("../tests/fixtures/h264.mp4");
#[cfg(target_os = "macos")]
const H264_BFRAMES: &[u8] = include_bytes!("../tests/fixtures/h264_bframes.mp4");
const FRAGMENTED: &[u8] = include_bytes!("../tests/fixtures/h264_fragmented.mp4");
#[cfg(feature = "decode-av1")]
const AV1_MP4: &[u8] = include_bytes!("../tests/fixtures/av1.mp4");

#[test]
fn a_container_the_demuxer_declines_names_demux_in_the_refusal() {
    // No ffmpeg in this chain, so nothing can read a fragmented MP4.
    let policy = VideoDecodePolicy::default();
    let error = decode_clip(FRAGMENTED, 2.0, Some(&policy)).expect_err("nothing reads it");
    match &error {
        VisionError::NoVideoDecoder {
            container, tried, ..
        } => {
            assert_eq!(*container, Container::Mp4);
            assert_eq!(tried.first(), Some(&"demux"));
        }
        other => panic!("expected NoVideoDecoder, got {other}"),
    }
}

#[test]
fn a_broken_index_is_refused_not_retried() {
    // Cut inside the moov so the sample tables are incomplete.
    let cut = H264_MP4
        .get(..H264_MP4.len() / 3)
        .expect("fixture is long enough");
    let policy = VideoDecodePolicy::default();
    let error = decode_clip(cut, 2.0, Some(&policy)).expect_err("malformed");
    assert!(
        matches!(
            error,
            VisionError::Demux { .. } | VisionError::NoVideoDecoder { .. }
        ),
        "got {error}"
    );
}

#[cfg(feature = "decode-av1")]
#[test]
fn the_rust_step_decodes_an_av1_mp4() {
    let track = demux::open(AV1_MP4).expect("demuxes");
    let policy = VideoDecodePolicy::default();
    let input = StepInput::new(AV1_MP4, Container::Mp4, 10.0, &policy);
    let step = Av1Step(&track);
    let clip = run_chain(&[&step], &input).expect("decodes");
    let plan = SamplePlan::new(&track, 10.0, policy.max_frames());
    assert_eq!(clip.frames().len(), plan.kept());
    for frame in clip.frames() {
        assert_eq!((frame.width(), frame.height()), (64, 64));
    }
    assert!(clip.fps() > 0.0);
}

#[cfg(feature = "decode-av1")]
#[test]
fn the_rust_step_declines_other_codecs_by_name() {
    let track = demux::open(H264_MP4).expect("demuxes");
    let policy = VideoDecodePolicy::default();
    let input = StepInput::new(H264_MP4, Container::Mp4, 10.0, &policy);
    let step = Av1Step(&track);
    let error = run_chain(&[&step], &input).expect_err("h264 is not its codec");
    assert!(
        matches!(&error, VisionError::NoVideoDecoder { tried, .. } if tried == &["rust"]),
        "got {error}"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "opens a VideoToolbox session"]
fn the_os_step_decodes_h264_in_display_order() {
    let policy = VideoDecodePolicy::default();
    for bytes in [H264_MP4, H264_BFRAMES] {
        let clip = decode_clip(bytes, 10.0, Some(&policy)).expect("decodes");
        assert!(clip.frames().len() >= 4, "{} frames", clip.frames().len());
        for frame in clip.frames() {
            assert_eq!((frame.width(), frame.height()), (64, 64));
        }
    }
}

#[test]
fn the_output_cap_ends_the_request() {
    let mut frames = Vec::new();
    let frame = HwFrame::new(2, 2, vec![0; 12], HwTimestamp::new(0)).expect("2x2");
    let error = take(&mut frames, vec![frame], 0, 8).expect_err("12 bytes exceed 8");
    assert!(matches!(error, VisionError::DecoderOutputCap { cap: 8 }));
}

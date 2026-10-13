//! `VideoToolbox` device tests: the committed fixtures decoded by the OS
//! decoder and compared with ffmpeg's decode of the same streams.
//!
//! `#[ignore]`: they open a real decoder session. Run them with
//!
//! ```text
//! cargo test -p shiso-media-sys --locked --test videotoolbox -- --ignored
//! ```
//!
//! The reference frames are not committed (`test-vectors/` is gitignored); a
//! test that needs them and finds none says so on stderr and returns, so read
//! the output before calling a run a pass. The fixture test at the bottom needs
//! no device and always runs.
#![cfg(target_os = "macos")]
#![expect(
    clippy::print_stderr,
    reason = "device tests: what was measured, and why a check did not run, belong on stderr"
)]

mod support;

use shiso_media_sys::{HwCodec, HwDecoder, HwDecoderDesc, HwError, HwFrame, HwTimestamp};
use support::{Clip, FRAMES, clip, cropped, psnr, reference};

/// The floor every frame is held to (the spec's): above what a wrong plane,
/// stride or colour matrix scores (a BT.601 or full-range conversion of these
/// fixtures measures 29 and 26 dB), and below what two correct conversions'
/// rounding costs (measured about 50 dB; `support` says how the references
/// are made).
const PSNR_FLOOR: f64 = 40.0;

const ALL: [HwCodec; 4] = [HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1];

/// Opens a decoder for `clip` at the fixture's size.
fn open(clip: &Clip) -> Result<HwDecoder, HwError> {
    open_at(clip, clip.width, clip.height)
}

fn open_at(clip: &Clip, width: u32, height: u32) -> Result<HwDecoder, HwError> {
    HwDecoder::open(&HwDecoderDesc::new(clip.codec, &clip.config, width, height))
}

/// Opens `codec`'s fixture, or says on stderr why this machine cannot and
/// returns `None`. Only AV1 may be missing (no AV1 decoder before M3).
fn open_or_skip(clip: &Clip) -> Option<HwDecoder> {
    match open(clip) {
        Ok(decoder) => Some(decoder),
        Err(HwError::Unsupported { reason, .. }) if clip.codec == HwCodec::Av1 => {
            eprintln!("SKIPPED av1: VideoToolbox has no AV1 decoder on this machine: {reason}");
            None
        }
        Err(e) => panic!("{}: open: {e}", clip.codec),
    }
}

/// Feeds every sample, keeping those `keep` picks (by decode index), and
/// returns every frame in the order `decode` and then `flush` handed it back.
fn run(decoder: &mut HwDecoder, clip: &Clip, keep: impl Fn(usize) -> bool) -> Vec<HwFrame> {
    let mut frames = Vec::new();
    for (index, (sample, pts)) in clip.samples.iter().enumerate() {
        let out = decoder
            .decode(sample, HwTimestamp::new(*pts), keep(index))
            .unwrap_or_else(|e| panic!("{} sample {index} (pts {pts}): {e}", clip.codec));
        frames.extend(out);
    }
    frames.extend(decoder.flush().expect("flush"));
    frames
}

fn ticks(frames: &[HwFrame]) -> Vec<i64> {
    frames.iter().map(|f| f.pts().ticks()).collect()
}

fn display_order(clip: &Clip) -> Vec<i64> {
    let mut pts: Vec<i64> = clip.samples.iter().map(|(_, pts)| *pts).collect();
    pts.sort_unstable();
    pts
}

fn decodes_against_reference(clip: &Clip) {
    let clip = clip.clone();
    let codec = clip.name;
    let Some(mut decoder) = open_or_skip(&clip) else {
        return;
    };
    let frames = run(&mut decoder, &clip, |_| true);

    // Every frame, once, in display order across all the `decode` calls and
    // the `flush` together.
    assert_eq!(frames.len(), FRAMES, "{codec}: one frame per sample");
    assert_eq!(
        ticks(&frames),
        display_order(&clip),
        "{codec}: display order"
    );
    for frame in &frames {
        assert_eq!(
            (frame.width(), frame.height()),
            (clip.width, clip.height),
            "{codec}"
        );
    }

    let reference = match reference(&clip) {
        Ok(reference) => reference,
        Err(why) => {
            eprintln!(
                "SKIPPED {codec} PSNR check (decode and order were checked): no reference frames: {why}"
            );
            return;
        }
    };
    let frame_bytes = frame_bytes(&clip);
    assert_eq!(
        reference.len(),
        frame_bytes * FRAMES,
        "{codec}: reference size"
    );
    let scores: Vec<f64> = frames
        .iter()
        .zip(reference.chunks_exact(frame_bytes))
        .map(|(frame, want)| psnr(frame.rgb(), want))
        .collect();
    let worst = scores.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!("{codec}: {FRAMES} frames in display order, PSNR {scores:.2?}, worst {worst:.2} dB");
    for (index, score) in scores.iter().enumerate() {
        assert!(
            *score >= PSNR_FLOOR,
            "{codec} frame {index}: PSNR {score:.2} dB < {PSNR_FLOOR}"
        );
    }
}

#[test]
#[ignore = "opens a VideoToolbox session"]
fn h264_high_with_b_frames_matches_reference() {
    decodes_against_reference(&clip(HwCodec::H264));
}

#[test]
#[ignore = "opens a VideoToolbox session"]
fn hevc_with_b_frames_matches_reference() {
    decodes_against_reference(&clip(HwCodec::Hevc));
}

#[test]
#[ignore = "opens a VideoToolbox session"]
fn vp9_matches_reference() {
    decodes_against_reference(&clip(HwCodec::Vp9));
}

#[test]
#[ignore = "opens a VideoToolbox session; skips (saying so) where there is no AV1 decoder"]
fn av1_matches_reference() {
    decodes_against_reference(&clip(HwCodec::Av1));
}

/// 170x98 is coded as 176x112; the frames come out at the cropped size the
/// SPS declares, with the picture where the reference has it.
#[test]
#[ignore = "opens a VideoToolbox session"]
fn h264_crop_is_honoured() {
    decodes_against_reference(&cropped(HwCodec::H264));
}

#[test]
#[ignore = "opens a VideoToolbox session"]
fn hevc_conformance_window_is_honoured() {
    decodes_against_reference(&cropped(HwCodec::Hevc));
}

fn frame_bytes(clip: &Clip) -> usize {
    usize::try_from(clip.width * clip.height * 3).expect("small")
}

/// A sample decoded with `keep = false` still feeds later frames but is not
/// returned, and the frames that are kept still come out in display order.
#[test]
#[ignore = "opens a VideoToolbox session"]
fn dropped_frames_are_decoded_but_not_returned() {
    for codec in ALL {
        let clip = clip(codec);
        let Some(mut decoder) = open_or_skip(&clip) else {
            continue;
        };
        // Odd decode indices: in the IBBP streams these include B-frames that
        // reference the dropped anchors.
        let frames = run(&mut decoder, &clip, |index| index % 2 == 1);
        let mut want: Vec<i64> = clip
            .samples
            .iter()
            .skip(1)
            .step_by(2)
            .map(|(_, pts)| *pts)
            .collect();
        want.sort_unstable();
        assert_eq!(ticks(&frames), want, "{codec}");

        // The kept frames are right, so the dropped ones they reference were
        // decoded.
        let Ok(reference) = reference(&clip) else {
            eprintln!("SKIPPED {codec} PSNR check of the kept frames: no reference frames");
            continue;
        };
        let frame_bytes = frame_bytes(&clip);
        let order = display_order(&clip);
        for frame in &frames {
            let index = order
                .iter()
                .position(|pts| *pts == frame.pts().ticks())
                .expect("a sample's pts");
            let want = &reference[index * frame_bytes..(index + 1) * frame_bytes];
            let score = psnr(frame.rgb(), want);
            assert!(
                score >= PSNR_FLOOR,
                "{codec} display frame {index}: {score:.2} dB"
            );
        }
    }
}

/// A sample whose length prefix overruns it is refused before it reaches the
/// decoder; a well-framed sample of noise reaches the decoder and comes back
/// as `Failed`. Neither panics.
#[test]
#[ignore = "opens a VideoToolbox session"]
fn garbage_samples_fail() {
    let clip = clip(HwCodec::H264);
    let mut decoder = open(&clip).expect("open");
    let (first, pts) = &clip.samples[0];
    decoder
        .decode(first, HwTimestamp::new(*pts), true)
        .expect("the keyframe decodes");

    let truncated = &first[..first.len() / 2];
    let err = decoder
        .decode(truncated, HwTimestamp::new(10_000), true)
        .expect_err("a truncated sample");
    assert!(matches!(err, HwError::Failed { .. }), "{err}");

    // A well-framed IDR slice NAL unit whose payload is noise. Measured:
    // the decoder's handler reports kVTVideoDecoderBadDataErr.
    let mut garbage = vec![0, 0, 0, 64, 0x65];
    garbage.extend((0u8..63).map(|i| i.wrapping_mul(37) ^ 0x5a));
    let result = decoder
        .decode(&garbage, HwTimestamp::new(20_000), true)
        .and_then(|_| decoder.flush());
    eprintln!("h264 noise slice: {result:?}");
    assert!(matches!(result, Err(HwError::Failed { .. })), "{result:?}");

    for codec in [HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1] {
        let clip = support::clip(codec);
        let Some(mut decoder) = open_or_skip(&clip) else {
            continue;
        };
        let noise: Vec<u8> = (0u8..96).map(|i| i.wrapping_mul(53) ^ 0x3c).collect();
        // Framed for HEVC; VP9 and AV1 samples carry their own syntax.
        let sample = if codec == HwCodec::Hevc {
            [&[0, 0, 0, 96][..], &noise].concat()
        } else {
            noise
        };
        let result = decoder
            .decode(&sample, HwTimestamp::new(0), true)
            .and_then(|_| decoder.flush());
        eprintln!("{codec} noise sample: {result:?}");
        assert!(
            matches!(result, Err(HwError::Failed { .. })),
            "{codec}: {result:?}"
        );
    }
}

/// The size in `HwDecoderDesc` is the container's claim. H.264 and HEVC take
/// their size from the SPS and AV1 from its sequence header, so a wrong but
/// possible claim changes nothing; VP9 has no header in its record to
/// overrule it, and the decoder refuses its frames (`Failed`). A size no
/// picture can have is refused up front, also `Failed`.
#[test]
#[ignore = "opens a VideoToolbox session"]
fn frame_size_comes_from_the_bitstream() {
    for codec in ALL {
        let clip = clip(codec);
        for (width, height) in [(0, clip.height), (clip.width, 0), (u32::MAX, clip.height)] {
            let err = open_at(&clip, width, height).expect_err("an impossible size");
            assert!(
                matches!(err, HwError::Failed { .. }),
                "{codec} {width}x{height}: {err}"
            );
        }
        if open_or_skip(&clip).is_none() {
            continue;
        }
        let mut decoder = open_at(&clip, 320, 200).expect("a wrong but possible size opens");
        if codec == HwCodec::Vp9 {
            let (sample, pts) = &clip.samples[0];
            let result = decoder
                .decode(sample, HwTimestamp::new(*pts), true)
                .and_then(|_| decoder.flush());
            assert!(matches!(result, Err(HwError::Failed { .. })), "{result:?}");
            continue;
        }
        let frames = run(&mut decoder, &clip, |_| true);
        assert_eq!(frames.len(), FRAMES, "{codec}");
        for frame in &frames {
            assert_eq!(
                (frame.width(), frame.height()),
                (clip.width, clip.height),
                "{codec}"
            );
        }
    }
}

/// A configuration record that is not one - the wrong version, cut short, or
/// with parameter sets `CoreMedia` cannot parse - is malformed input, refused
/// as `Failed` (not `Unsupported`, which would hand the same bytes to the
/// next decoder).
#[test]
#[ignore = "opens a VideoToolbox session"]
fn bad_avcc_is_refused() {
    let clip = clip(HwCodec::H264);
    let open = |config: &[u8]| {
        HwDecoder::open(&HwDecoderDesc::new(
            clip.codec,
            config,
            clip.width,
            clip.height,
        ))
    };

    let mut wrong_version = clip.config.clone();
    wrong_version[0] = 7;
    let mut noisy_sps = clip.config.clone();
    for (i, byte) in noisy_sps.iter_mut().enumerate().skip(9).take(12) {
        *byte = u8::try_from(i).expect("small").wrapping_mul(91) ^ 0xa5;
    }
    let mut overlong_sps = clip.config.clone();
    overlong_sps[7] = 0xff;
    for (what, config) in [
        ("version 7", &wrong_version[..]),
        ("truncated", &clip.config[..3]),
        ("empty", &[][..]),
        ("SPS length past the end", &overlong_sps[..]),
        ("SPS bytes replaced by noise", &noisy_sps[..]),
    ] {
        let result = open(config);
        eprintln!("avcC {what}: {result:?}");
        assert!(
            matches!(result, Err(HwError::Failed { .. })),
            "{what}: {result:?}"
        );
    }
}

/// Dropping a session with frames in flight releases it cleanly, and a new
/// session decodes the whole stream afterwards.
#[test]
#[ignore = "opens a VideoToolbox session"]
fn dropping_mid_stream_is_clean() {
    for codec in ALL {
        let clip = clip(codec);
        for _ in 0..3 {
            let Some(mut decoder) = open_or_skip(&clip) else {
                return;
            };
            for (sample, pts) in clip.samples.iter().take(4) {
                decoder
                    .decode(sample, HwTimestamp::new(*pts), true)
                    .expect("decode");
            }
            drop(decoder);
        }
        let mut decoder = open(&clip).expect("open");
        assert_eq!(run(&mut decoder, &clip, |_| true).len(), FRAMES, "{codec}");
    }
}

/// A session can move to another thread between samples.
#[test]
#[ignore = "opens a VideoToolbox session"]
fn a_session_moves_between_threads() {
    let clip = clip(HwCodec::H264);
    let mut decoder = open(&clip).expect("open");
    let (first, rest) = clip.samples.split_at(4);
    let rest = rest.to_vec();
    let mut frames = Vec::new();
    for (sample, pts) in first {
        frames.extend(
            decoder
                .decode(sample, HwTimestamp::new(*pts), true)
                .expect("decode"),
        );
    }
    let (decoder, more) = std::thread::spawn(move || {
        let mut frames = Vec::new();
        for (sample, pts) in &rest {
            frames.extend(
                decoder
                    .decode(sample, HwTimestamp::new(*pts), true)
                    .expect("decode"),
            );
        }
        (decoder, frames)
    })
    .join()
    .expect("the decoding thread");
    let mut decoder = decoder;
    frames.extend(more);
    frames.extend(decoder.flush().expect("flush"));
    assert_eq!(ticks(&frames), display_order(&clip));
}

#[test]
#[ignore = "asks VideoToolbox what it can decode"]
fn probe_reports_hardware_codecs() {
    let probe = shiso_media_sys::probe().expect("VideoToolbox is always present on macOS");
    eprintln!("probe: {probe:?}");
    assert_eq!(probe.decoder, "VideoToolbox");
    assert!(probe.codecs.contains(&HwCodec::H264), "{probe:?}");
}

/// The fixtures parse into what the decoder tests assume, so a broken fixture
/// fails here and not as a confusing decode error.
#[test]
fn fixtures_split_into_one_sample_per_frame() {
    for codec in ALL {
        let clip = clip(codec);
        assert_eq!(clip.samples.len(), FRAMES, "{codec}");
        let mut pts = display_order(&clip);
        pts.dedup();
        assert_eq!(pts.len(), FRAMES, "{codec}: distinct timestamps");
        assert!(!clip.config.is_empty(), "{codec}");
    }
    // The H.264 and HEVC fixtures really have B-frames: decode order is not
    // display order.
    for codec in [HwCodec::H264, HwCodec::Hevc] {
        let clip = clip(codec);
        let decode: Vec<i64> = clip.samples.iter().map(|(_, pts)| *pts).collect();
        assert_ne!(decode, display_order(&clip), "{codec}");
    }
}

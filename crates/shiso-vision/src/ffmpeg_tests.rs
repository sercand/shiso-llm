//! Tests for the subprocess decode backend.
//!
//! The ones that need a real decoder generate their own clip with it and skip
//! when it is absent, so the suite stays green on a machine without it - the
//! same condition the backend itself has to survive.

use std::num::{NonZeroU64, NonZeroUsize};

use super::{Availability, FfmpegPolicy, decode_frames, probe, split_png_stream};
use crate::error::VisionError;

/// A policy pointing at whatever decoder is on `PATH`.
fn enabled() -> FfmpegPolicy {
    FfmpegPolicy::default()
}

/// A policy pointing at a path that cannot exist.
fn missing() -> FfmpegPolicy {
    FfmpegPolicy::new(std::path::PathBuf::from(
        "/nonexistent/definitely-not-a-decoder",
    ))
}

fn have_decoder() -> bool {
    matches!(probe(Some(&enabled())), Availability::Ready { .. })
}

fn nonzero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("a non-zero cap")
}

/// Encodes a short clip with the decoder itself and hands back the container
/// bytes.
///
/// The temporary directory is unique per CALL rather than per parameter set:
/// keying it on the parameters collided between tests that differ only in frame
/// size, and since each call removes its own directory, one test deleted
/// another's clip mid-decode.
fn make_clip(seconds: u32, fps: u32, size: &str, codec: &str) -> Option<Vec<u8>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);

    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "shiso-vision-clip-{}-{n}-{seconds}-{fps}-{size}-{codec}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join("clip.mp4");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=size={size}:rate={fps}:duration={seconds}"),
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&out)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;
    let bytes = if status.success() {
        std::fs::read(&out).ok()
    } else {
        None
    };
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

// -- the default posture ----------------------------------------------------

/// The disabled state is the absence of a policy, so there is no value to
/// assert `enabled == false` on. What there is instead is this: the type system
/// makes a decode without a policy impossible, and the probe says so by name.
#[test]
fn no_policy_probes_as_disabled_rather_than_missing() {
    assert_eq!(probe(None), Availability::Disabled);
}

// -- availability reporting -------------------------------------------------

#[test]
fn probing_a_missing_binary_reports_it_by_name() {
    match probe(Some(&missing())) {
        Availability::Missing { binary, reason } => {
            assert!(
                binary
                    .to_string_lossy()
                    .contains("definitely-not-a-decoder"),
                "{binary:?}"
            );
            assert!(!reason.is_empty(), "the reason must say something");
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

/// A misconfigured deployment must fail with the BINARY named, not with
/// something that reads like a corrupt video: the two have completely different
/// fixes, and an error that points at the upload sends the operator to the
/// wrong place entirely.
#[test]
fn a_missing_binary_fails_the_decode_by_name() {
    let error = decode_frames(b"x", 2.0, &missing()).expect_err("a missing binary cannot decode");
    match &error {
        VisionError::DecoderSpawn { binary, .. } => {
            assert!(
                binary
                    .to_string_lossy()
                    .contains("definitely-not-a-decoder"),
                "{binary:?}"
            );
        }
        other => panic!("expected a spawn failure, got {other}"),
    }
    let text = error.to_string();
    assert!(
        text.contains("ffmpeg installed"),
        "no remedy offered: {text}"
    );
}

// -- real decoding ----------------------------------------------------------

#[test]
fn a_clip_decodes_to_frames_at_the_requested_rate() {
    if !have_decoder() {
        tracing::warn!("skipping: no decoder on PATH");
        return;
    }
    let clip = make_clip(3, 10, "320x240", "libx264").expect("encode");
    let frames = decode_frames(&clip, 2.0, &enabled()).expect("decode");
    // Three seconds at 2fps. The rate filter can emit an extra boundary frame,
    // so this is a band rather than an exact count; pinning it exactly would
    // make the test hostage to a filter-timing detail.
    assert!(
        (5..=7).contains(&frames.len()),
        "expected about 6 frames, got {}",
        frames.len()
    );
    assert_eq!(
        (frames[0].width(), frames[0].height()),
        (320, 240),
        "the decoder resized the clip"
    );
}

#[test]
fn the_requested_rate_changes_the_frame_count() {
    if !have_decoder() {
        tracing::warn!("skipping: no decoder on PATH");
        return;
    }
    let clip = make_clip(4, 10, "160x120", "libx264").expect("encode");
    let slow = decode_frames(&clip, 1.0, &enabled()).expect("1 fps");
    let fast = decode_frames(&clip, 4.0, &enabled()).expect("4 fps");
    assert!(
        fast.len() > slow.len(),
        "4 fps gave {} frames, 1 fps gave {}",
        fast.len(),
        slow.len()
    );
}

/// Frames must be distinct. A backend that returned the same frame N times
/// would pass every count and dimension check while feeding the model a still.
#[test]
fn decoded_frames_are_not_all_identical() {
    if !have_decoder() {
        tracing::warn!("skipping: no decoder on PATH");
        return;
    }
    let clip = make_clip(3, 10, "160x120", "libx264").expect("encode");
    let frames = decode_frames(&clip, 2.0, &enabled()).expect("decode");
    assert!(frames.len() >= 2);
    assert_ne!(
        frames[0].as_rgb8(),
        frames[frames.len() - 1].as_rgb8(),
        "first and last frame are byte-identical; the clip did not advance"
    );
}

#[test]
fn the_frame_cap_is_honoured() {
    if !have_decoder() {
        tracing::warn!("skipping: no decoder on PATH");
        return;
    }
    let clip = make_clip(5, 10, "160x120", "libx264").expect("encode");
    let policy = enabled().with_max_frames(nonzero_usize(3));
    assert_eq!(
        decode_frames(&clip, 10.0, &policy).expect("decode").len(),
        3
    );
}

#[test]
fn an_output_cap_smaller_than_the_clip_is_refused() {
    if !have_decoder() {
        tracing::warn!("skipping: no decoder on PATH");
        return;
    }
    let clip = make_clip(3, 10, "320x240", "libx264").expect("encode");
    let policy = enabled().with_max_output_bytes(nonzero_usize(1024));
    let error = decode_frames(&clip, 2.0, &policy).expect_err("the cap must bite");
    assert!(
        matches!(error, VisionError::DecoderOutputCap { cap: 1024 }),
        "{error}"
    );
}

/// Garbage must produce the decoder's own complaint, not a panic or a hang.
#[test]
fn a_non_video_payload_fails_cleanly() {
    if !have_decoder() {
        tracing::warn!("skipping: no decoder on PATH");
        return;
    }
    let error = decode_frames(b"this is not a video at all", 2.0, &enabled())
        .expect_err("arbitrary bytes are not a container");
    assert!(
        matches!(error, VisionError::DecoderFailed { .. }),
        "{error}"
    );
}

/// An HEVC clip in the same container: the case a linked H.264-only decoder
/// could not have served, and the reason this backend exists at all.
#[test]
fn an_hevc_clip_decodes_too_when_the_encoder_is_available() {
    if !have_decoder() {
        tracing::warn!("skipping: no decoder on PATH");
        return;
    }
    let Some(clip) = make_clip(2, 10, "160x120", "libx265") else {
        tracing::warn!("skipping: this decoder build has no HEVC encoder");
        return;
    };
    let frames = decode_frames(&clip, 2.0, &enabled()).expect("decode hevc");
    assert!(!frames.is_empty());
    assert_eq!((frames[0].width(), frames[0].height()), (160, 120));
}

/// A decoder that hangs must not hold a request thread for its full budget.
#[cfg(unix)]
#[test]
fn a_hanging_decoder_is_killed_at_its_deadline() {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);

    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("shiso-vision-hang-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let binary = dir.join("hanging-decoder");
    std::fs::write(&binary, "#!/bin/sh\nexec sleep 10\n").expect("write the stub");
    let mut permissions = std::fs::metadata(&binary)
        .expect("stat the stub")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&binary, permissions).expect("make the stub runnable");

    let policy = FfmpegPolicy::new(binary)
        .with_timeout_secs(NonZeroU64::new(1).expect("a non-zero timeout"));

    // The stub was written by this thread a moment ago, and a *different*
    // thread's `Command::spawn` between that write and this one inherits the
    // descriptor across its fork - which makes the file busy until that child
    // execs, and this spawn fails with a message about ffmpeg not being
    // installed. It is a race between tests rather than anything about the
    // watchdog, it only happens under a parallel run, and the answer is to try
    // again rather than to quarantine the test: the condition is transient by
    // construction and clears in microseconds.
    let mut attempts = 0;
    let (error, started) = loop {
        let started = std::time::Instant::now();
        let error = decode_frames(b"input", 2.0, &policy).expect_err("the watchdog must fire");
        if matches!(error, VisionError::DecoderTimeout { .. }) || attempts == 20 {
            break (error, started);
        }
        attempts += 1;
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        matches!(error, VisionError::DecoderTimeout { secs: 1 }),
        "{error}"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(3));
}

// -- stream splitting -------------------------------------------------------

#[test]
fn an_empty_stream_splits_to_nothing() {
    assert!(
        split_png_stream(b"")
            .expect("an empty stream is not an error")
            .is_empty()
    );
}

/// Frames arrive concatenated on one pipe, and the signature is what separates
/// them - a payload byte pattern must not be mistaken for a frame boundary.
#[test]
fn a_stream_of_two_images_splits_into_two_frames() {
    let mut buf = Vec::new();
    for pixel in [[1u8, 2, 3], [4, 5, 6]] {
        let image = image::RgbImage::from_pixel(4, 3, image::Rgb(pixel));
        let mut one = Vec::new();
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut std::io::Cursor::new(&mut one), image::ImageFormat::Png)
            .expect("encode");
        buf.extend_from_slice(&one);
    }
    let frames = split_png_stream(&buf).expect("split");
    assert_eq!(frames.len(), 2);
    assert_eq!((frames[0].width(), frames[0].height()), (4, 3));
    assert_eq!(frames[0].as_rgb8().get(..3), Some(&[1u8, 2, 3][..]));
    assert_eq!(frames[1].as_rgb8().get(..3), Some(&[4u8, 5, 6][..]));
}

#[test]
fn run_decoder_returns_raw_output_for_arbitrary_output_args() {
    if !have_decoder() {
        return;
    }
    // A generated one-second sine through the audio arguments the audio crate
    // will use. Raw f32 mono at 16 kHz is exactly 64000 bytes per second.
    let wav = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
        ])
        .args(["-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"])
        .output()
        .expect("the decoder generates a tone")
        .stdout;
    let out = super::run_decoder(
        &enabled(),
        &[
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
        ],
        &wav,
    )
    .expect("the tone decodes");
    assert_eq!(out.len(), 64_000);
}

/// An m4a whose index trails its media, as the decoder itself and most phone
/// recorders write one, still decodes over the pipe.
#[test]
fn an_audio_file_with_a_trailing_index_decodes_over_the_pipe() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    if !have_decoder() {
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "shiso-vision-m4a-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("a temp dir");
    let out = dir.join("tone.m4a");
    let made = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=60",
        ])
        .args(["-c:a", "aac", "-b:a", "64k"])
        .arg(&out)
        .status()
        .expect("the decoder runs");
    assert!(made.success(), "the tone encodes");
    let bytes = std::fs::read(&out).expect("the tone reads");
    let _ = std::fs::remove_dir_all(&dir);
    let moov = bytes
        .windows(4)
        .position(|w| w == b"moov")
        .expect("an index");
    let mdat = bytes.windows(4).position(|w| w == b"mdat").expect("media");
    assert!(moov > mdat, "the fixture's index trails its media");
    let raw = super::run_decoder(
        &enabled(),
        &[
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
        ],
        &bytes,
    )
    .expect("it decodes");
    // A minute at 16 kHz, give or take the encoder's priming. Sixty seconds
    // because a clip small enough to fit the decoder's probe buffer demuxes
    // from a pipe anyway, and would not show the trailing index at all.
    let samples = raw.len() / 4;
    assert!((958_000..=962_000).contains(&samples), "{samples} samples");
}

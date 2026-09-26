//! Mel parity against the independent reference extractors.
//!
//! The fixtures live under the gitignored `test-vectors/references/audio/` and
//! are captured by `scripts/reference/audio_reference.py mel`. A missing
//! fixture skips the case with a line on stdout: read the output, because a
//! skipped case and a passing one look alike.

use std::path::{Path, PathBuf};

use shiso_audio::{
    Filterbank, Framing, LogScale, MelFrontend, MelScale, MelSpec, Preemphasis, Spectrum, Window,
};

// The budgets are about twice what was measured on 2026-09-25 (M1 Max,
// reference and versions as recorded in test-vectors/README.md). The residual is the
// reference's f32 STFT and f32 filterbank against this crate's f64 transform
// and f64 accumulation: it is rounding, not a disagreement.
//
// Centred: worst 4.6e-4 (speech-en), 3.3e-5 (weather-es); mean 9.3e-7 and
// 3.5e-7. The one large residual sits in a near-silent bin, where the 2^-24 log
// guard magnifies a relative rounding error of the power sum into log units.
const BUDGET_CENTRED_MAX: f32 = 1e-3;
const BUDGET_CENTRED_MEAN: f64 = 2e-6;
// Left-padded: worst 2.8e-5 (speech-en), 2.5e-5 (weather-es); mean 1.9e-7 and
// 2.0e-7. The 1e-3 log guard keeps every bin well above the rounding floor.
const BUDGET_LEFT_MAX: f32 = 6e-5;
const BUDGET_LEFT_MEAN: f64 = 5e-7;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/audio")
}

#[expect(clippy::print_stdout, reason = "the skip report is this test's output")]
fn say(line: &str) {
    println!("{line}");
}

fn spec(style: &str) -> MelSpec {
    match style {
        "centred" => MelSpec {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 128,
            window: Window::HannSymmetric,
            framing: Framing::Centred,
            preemphasis: Preemphasis::Signal(0.97),
            spectrum: Spectrum::Power,
            filterbank: Filterbank::Computed {
                scale: MelScale::Slaney,
                slaney_norm: true,
                f_min: 0.0,
                f_max: 8_000.0,
            },
            log_scale: LogScale::Natural {
                guard: 2f32.powi(-24),
            },
        },
        "left-padded" => MelSpec {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 320,
            hop_length: 160,
            n_mels: 128,
            window: Window::HannPeriodic,
            framing: Framing::LeftPadded {
                pad_to_multiple: 128,
            },
            preemphasis: Preemphasis::None,
            spectrum: Spectrum::Magnitude,
            filterbank: Filterbank::Computed {
                scale: MelScale::Htk,
                slaney_norm: false,
                f_min: 0.0,
                f_max: 8_000.0,
            },
            log_scale: LogScale::Natural { guard: 1e-3 },
        },
        other => panic!("unknown style {other}"),
    }
}

/// 16-bit PCM mono WAV, the only shape the capture script writes.
fn read_clip(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("the clip reads");
    let data = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("a data chunk")
        + 8;
    bytes[data..]
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32_768.0)
        .collect()
}

fn case(clip: &str, style: &str, budget_max: f32, budget_mean: f64) {
    let meta_path = root().join(format!("mel-{clip}-{style}.json"));
    let Ok(meta) = std::fs::read_to_string(&meta_path) else {
        say(&format!(
            "SKIP mel {clip}/{style}: {} is absent; run audio_reference.py mel",
            meta_path.display()
        ));
        return;
    };
    let meta: serde_json::Value = serde_json::from_str(&meta).expect("the metadata is JSON");
    let reference: Vec<f32> = std::fs::read(root().join(format!("mel-{clip}-{style}.f32")))
        .expect("the block reads")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let samples = read_clip(&root().join(format!("clips/{clip}.wav")));
    let mel = MelFrontend::new(spec(style))
        .expect("the spec is valid")
        .compute(&samples);
    let total = usize::try_from(meta["frames_total"].as_u64().expect("total")).expect("fits");
    let valid = usize::try_from(meta["frames_valid"].as_u64().expect("valid")).expect("fits");
    assert_eq!(
        (mel.total(), mel.valid()),
        (total, valid),
        "{clip}/{style} geometry"
    );
    let (mut worst, mut sum) = (0.0f32, 0.0f64);
    for (a, b) in mel.as_slice().iter().zip(&reference) {
        let d = (a - b).abs();
        worst = worst.max(d);
        sum += f64::from(d);
    }
    let mean = sum / f64::from(u32::try_from(reference.len()).expect("fits"));
    say(&format!(
        "mel {clip}/{style}: worst {worst:.3e}, mean {mean:.3e} over {valid} frames"
    ));
    assert!(worst <= budget_max, "{clip}/{style}: worst {worst}");
    assert!(mean <= budget_mean, "{clip}/{style}: mean {mean}");
}

#[test]
fn centred_frontend_matches_the_reference() {
    for clip in ["weather-es", "speech-en", "speech-en-odd"] {
        case(clip, "centred", BUDGET_CENTRED_MAX, BUDGET_CENTRED_MEAN);
    }
}

#[test]
fn left_padded_frontend_matches_the_reference() {
    for clip in ["weather-es", "speech-en", "speech-en-odd"] {
        case(clip, "left-padded", BUDGET_LEFT_MAX, BUDGET_LEFT_MEAN);
    }
}

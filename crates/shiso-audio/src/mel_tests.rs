//! Frontend geometry and arithmetic that can be checked without a reference.

use super::{
    Filterbank, Framing, LogScale, MelFrontend, MelScale, MelSpec, Preemphasis, Spectrum, Window,
    apply_max_floor,
};
use crate::num::{f32_of, f64_of};

/// The Nemotron-style frontend: centred framing, symmetric window, power.
pub(crate) fn centred() -> MelSpec {
    MelSpec {
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
    }
}

/// The Gemma-style frontend: left padding, periodic window, magnitude, HTK.
pub(crate) fn left_padded() -> MelSpec {
    MelSpec {
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
    }
}

/// The reflect-framed frontend: mirrored padding, periodic window, power,
/// Slaney scale, the whole-clip floor-and-rescale log.
pub(crate) fn centred_reflect() -> MelSpec {
    MelSpec {
        sample_rate: 16_000,
        n_fft: 400,
        win_length: 400,
        hop_length: 160,
        n_mels: 128,
        window: Window::HannPeriodic,
        framing: Framing::CentredReflect,
        preemphasis: Preemphasis::None,
        spectrum: Spectrum::Power,
        filterbank: Filterbank::Computed {
            scale: MelScale::Slaney,
            slaney_norm: true,
            f_min: 0.0,
            f_max: 8_000.0,
        },
        log_scale: LogScale::Log10MaxFloor,
    }
}

fn tone(freq: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| f32_of((2.0 * std::f64::consts::PI * freq * f64_of(i) / 16_000.0).sin() * 0.5))
        .collect()
}

#[test]
fn centred_frame_counts() {
    let fe = MelFrontend::new(centred()).expect("valid");
    for (len, valid) in [(0, 0), (50, 0), (160, 1), (16_000, 100), (16_123, 100)] {
        let mel = fe.compute(&vec![0.1; len]);
        assert_eq!(mel.valid(), valid, "len {len}");
        assert_eq!(mel.total(), 1 + len / 160, "len {len}");
        assert_eq!(fe.valid_frames(len), valid);
        // Rows past the valid count are zeroed, as the reference masks them.
        for t in mel.valid()..mel.total() {
            assert!(mel.row(t).iter().all(|v| *v == 0.0));
        }
    }
}

#[test]
fn left_padded_frame_counts() {
    let fe = MelFrontend::new(left_padded()).expect("valid");
    // 16000 samples -> padded to 16000 (already a multiple of 128);
    // total = (16000 + 160 - 321)/160 + 1 = 99; valid t: t*160 + 320 < 160 + 16000 -> t <= 98.
    let mel = fe.compute(&vec![0.1; 16_000]);
    assert_eq!((mel.total(), mel.valid()), (99, 99));
    // 16001 -> padded to 16128: total = (16128 + 160 - 321)/160 + 1 = 100;
    // valid t: t*160 + 320 < 160 + 16001 -> t <= 99 -> 100.
    let mel = fe.compute(&vec![0.1; 16_001]);
    assert_eq!((mel.total(), mel.valid()), (100, 100));
    // 16161 -> padded to 16256: total = (16256 + 160 - 321)/160 + 1 = 101; valid t <= 100 -> 101.
    let mel = fe.compute(&vec![0.1; 16_161]);
    assert_eq!((mel.total(), mel.valid()), (101, 101));
    assert_eq!(fe.compute(&[0.1; 100]).total(), 0);
}

#[test]
fn a_tone_peaks_in_its_mel_band() {
    for spec in [centred(), left_padded()] {
        let fe = MelFrontend::new(spec).expect("valid");
        let mel = fe.compute(&tone(1_000.0, 16_000));
        let row = mel.row(50);
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("bins");
        // 1 kHz lies in the lower third of a 0-8 kHz mel axis under both scales.
        assert!((20..50).contains(&peak), "peak bin {peak}");
    }
}

#[test]
fn slaney_bank_rows_integrate_to_the_norm() {
    // With Slaney normalisation each triangle has unit area in Hz, so its sum
    // over bins times the bin width is about 1.
    let fe = MelFrontend::new(centred()).expect("valid");
    let bin_hz = 16_000.0 / 512.0;
    for m in [10, 60, 120] {
        let area: f64 = fe.bank_row(m).iter().map(|w| f64::from(*w) * bin_hz).sum();
        assert!((area - 1.0).abs() < 0.08, "row {m}: area {area}");
    }
}

#[test]
fn inconsistent_specs_are_refused() {
    let mut s = centred();
    s.win_length = 600;
    assert!(MelFrontend::new(s).is_err());
    let mut s = left_padded();
    s.preemphasis = Preemphasis::Signal(0.97);
    assert!(MelFrontend::new(s).is_err());
    let mut s = centred();
    s.filterbank = Filterbank::Given(vec![0.0; 10]);
    assert!(MelFrontend::new(s).is_err());
    let mut s = centred();
    s.log_scale = LogScale::Natural { guard: 0.0 };
    assert!(MelFrontend::new(s).is_err());
    let mut s = centred_reflect();
    s.preemphasis = Preemphasis::Signal(0.97);
    assert!(MelFrontend::new(s).is_err());
}

#[test]
fn a_given_bank_is_used_verbatim() {
    let mut s = centred();
    let mut bank = vec![0.0f32; 128 * 257];
    bank[5 * 257 + 32] = 1.0; // mel row 5 reads bin 32 (1 kHz) only
    s.filterbank = Filterbank::Given(bank);
    let fe = MelFrontend::new(s).expect("valid");
    let mel = fe.compute(&tone(1_000.0, 16_000));
    assert!(mel.row(50)[5] > mel.row(50)[6] + 10.0);
}

#[test]
fn the_reflect_frontend_drops_the_last_frame() {
    let fe = MelFrontend::new(centred_reflect()).expect("valid");
    let signal: Vec<f32> = (0..1_600)
        .map(|i| f32_of((f64_of(i) * 0.013).sin() * 0.4))
        .collect();
    let mel = fe.compute(&signal);
    assert_eq!(fe.valid_frames(1_600), 10);
    assert_eq!((mel.total(), mel.valid()), (11, 10));
    // The dropped frame is never computed, so it stays zero like every other
    // row past `valid`.
    assert!(mel.row(10).iter().all(|v| *v == 0.0));
}

#[test]
fn the_log10_max_floor_floors_at_the_max_minus_eight() {
    let mels = 4;
    let valid = 3;
    let mut data: Vec<f32> = vec![
        -5.0, -5.0, -5.0, -5.0, //
        50.0, -3.0, -3.0, -3.0, // one huge bin
        -6.0, -6.0, -6.0, -6.0, //
        0.0, 0.0, 0.0, 0.0, // past `valid`: left alone
    ];
    apply_max_floor(&mut data, mels, valid);
    let floored = f32_of((50.0 - 8.0 + 4.0) / 4.0);
    let peak = f32_of((50.0 + 4.0) / 4.0);
    let want = [
        floored, floored, floored, floored, //
        peak, floored, floored, floored, //
        floored, floored, floored, floored, //
        0.0, 0.0, 0.0, 0.0,
    ];
    for (i, (got, want)) in data.iter().zip(&want).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "row {} col {}: {got} != {want}",
            i / mels,
            i % mels
        );
    }
}

#[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
fn say(line: &str) {
    eprintln!("{line}");
}

// Measured 2026-09-27 (M1 Max) against `qwen3asr-speech-en-mel.f32`: worst
// 3.65e-5, mean 1.0e-7 over 3000 * 128 values. Budgets are about twice that,
// following the convention `mel_reference.rs` sets for the other two
// frontends. The residual is the reference's own float32 STFT, filterbank and
// clamp against this crate's f64 transform and f64 accumulation: eight of the
// 384,000 values land a near-silent bin, where the log turns a tiny relative
// difference in the mel energy into a visible difference in log units - the
// same failure mode already documented on the centred frontend above.
const QWEN3ASR_BUDGET_MAX: f32 = 8e-5;
const QWEN3ASR_BUDGET_MEAN: f64 = 3e-7;

#[test]
#[ignore = "needs the qwen3asr mel reference fixture under test-vectors/references/audio"]
fn qwen3asr_mel_matches_the_reference() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/audio");
    let meta_path = root.join("qwen3asr-speech-en-mel.json");
    let Ok(meta) = std::fs::read_to_string(&meta_path) else {
        say(&format!(
            "SKIP qwen3asr_mel_matches_the_reference: {} is absent; capture it first",
            meta_path.display()
        ));
        return;
    };
    let meta: serde_json::Value = serde_json::from_str(&meta).expect("the metadata is JSON");
    let frames = usize::try_from(meta["frames"].as_u64().expect("a frame count")).expect("fits");
    let reference: Vec<f32> = std::fs::read(root.join("qwen3asr-speech-en-mel.f32"))
        .expect("the reference block reads")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let clip = std::fs::read(root.join("clips/speech-en.wav")).expect("the clip reads");
    let pcm = crate::wav::parse_wav(&clip).expect("a well-formed WAV");
    let samples = crate::resample::resample(pcm.samples(), pcm.rate(), crate::pcm::MODEL_RATE)
        .expect("16 kHz resamples");
    let fe = MelFrontend::new(centred_reflect()).expect("valid");
    let mel = fe.compute(&samples);
    assert_eq!(mel.valid(), frames, "frame count");
    assert_eq!(reference.len(), frames * mel.n_mels(), "reference shape");
    let got = &mel.as_slice()[..frames * mel.n_mels()];
    let (mut worst, mut sum) = (0.0f32, 0.0f64);
    for (a, b) in got.iter().zip(&reference) {
        let d = (a - b).abs();
        worst = worst.max(d);
        sum += f64::from(d);
    }
    let mean = sum / f64::from(u32::try_from(got.len()).expect("fits"));
    say(&format!(
        "qwen3asr mel: worst {worst:.3e}, mean {mean:.3e} over {frames} frames"
    ));
    assert!(worst <= QWEN3ASR_BUDGET_MAX, "worst abs difference {worst}");
    assert!(mean <= QWEN3ASR_BUDGET_MEAN, "mean abs difference {mean}");
}

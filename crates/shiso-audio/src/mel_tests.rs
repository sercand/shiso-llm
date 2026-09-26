//! Frontend geometry and arithmetic that can be checked without a reference.

use super::{Filterbank, Framing, MelFrontend, MelScale, MelSpec, Preemphasis, Spectrum, Window};
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
        log_guard: 2f32.powi(-24),
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
        log_guard: 1e-3,
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
    s.log_guard = 0.0;
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

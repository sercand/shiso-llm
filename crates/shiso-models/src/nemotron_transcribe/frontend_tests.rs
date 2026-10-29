//! The window plan, the frontend against the reference's raw log-mel, and
//! the limits a clip is held to.

#![expect(
    clippy::cast_precision_loss,
    reason = "a mean over a few hundred thousand cells needs no exact mantissa"
)]

use shiso_audio::{AudioError, MelFrontend, Pcm};

use super::super::config::{BlockDims, PeeConfig};
use super::super::encoder_tests::{reference, refs};
use super::{audio_tokens, chunk_plan, mel_spec};

fn config() -> PeeConfig {
    let block = BlockDims {
        d_model: 1280,
        heads: 10,
        head_dim: 128,
        layers: 1,
        ff: 5120,
        rot_dims: 64,
        qk_norm: true,
    };
    PeeConfig {
        asr: block,
        diar: block,
        ctc: block,
        mels: 128,
        stack: 8,
        speakers: 8,
        head_hidden: 192,
        fusion_dim: 2688,
        threshold: 0.5,
        ctc_vocab: 23_863,
        ctc_upsample: 2,
        ctc_residual: 0.1,
        window_core: 500,
        window_context: 50,
        sample_rate: 16_000,
        n_fft: 512,
        win_length: 400,
        hop_length: 160,
    }
}

#[test]
fn chunk_plan_covers_every_frame_once() {
    let c = config();
    for frames in [1usize, 2_999, 3_000, 3_001, 4_000, 4_001, 6_000, 360_000] {
        let plan = chunk_plan(frames, &c);
        let mut next = 0usize;
        let mut rows = 0usize;
        for span in &plan {
            assert_eq!(span.start, next, "{frames} frames: a gap or an overlap");
            assert!(span.enc_start <= span.start && span.end <= span.enc_end);
            assert!(span.left_drop + span.core <= span.len().div_ceil(8));
            next = span.end;
            rows += span.core;
        }
        assert_eq!(next, frames, "{frames} frames: the cores end early");
        assert_eq!(
            rows,
            audio_tokens(frames),
            "{frames} frames: rows dropped or doubled"
        );
    }
}

#[test]
fn one_hour_is_accepted_and_one_second_more_is_refused() {
    let tower = super::super::tower::limits_for_tests();
    let spec = mel_spec(&config(), &vec![0.0; 128 * 257]);
    let pcm = |seconds: usize| Pcm::new(vec![0.0; seconds * 16_000], 16_000).expect("pcm");
    assert!(shiso_audio::clip_for(&tower(spec.clone()), &pcm(3600)).is_ok());
    let err = shiso_audio::clip_for(&tower(spec), &pcm(3601)).expect_err("past an hour");
    assert!(matches!(err, AudioError::TooLong { .. }), "{err:?}");
}

#[test]
#[ignore = "needs the speech encoder export and the reference captures"]
#[expect(
    clippy::print_stderr,
    reason = "the measured agreement is the test's report"
)]
fn mel_matches_reference_short_en() {
    let Some((_, weights_fb)) = reference() else {
        return;
    };
    let Some(dir) = refs("short_en/verbatim") else {
        return;
    };
    let want = super::super::encoder_tests::floats(&dir.join("mel_raw.f32"));
    let pcm = super::super::encoder_tests::clip_pcm("short_en");
    let frontend = MelFrontend::new(mel_spec(&config(), &weights_fb)).expect("spec");
    let mel = frontend.compute(pcm.samples());
    assert_eq!(mel.valid() * 128, want.len(), "frame count");
    let diffs: Vec<f32> = mel.as_slice()[..want.len()]
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let (at, worst) = diffs
        .iter()
        .copied()
        .enumerate()
        .fold((0, 0.0f32), |m, (i, d)| if d > m.1 { (i, d) } else { m });
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    eprintln!(
        "mel worst |diff| {worst} at frame {} bin {} (value {}), mean {mean}",
        at / 128,
        at % 128,
        want[at]
    );
    // f64 against the reference's f32 transform: the worst cell is a
    // near-silent bin where the 2^-24 guard dominates.
    assert!(mean <= 1e-4, "mean log-mel difference {mean}");
    assert!(worst <= 1e-2, "worst log-mel difference {worst}");
}

//! The window plan, the frontend against the reference's raw log-mel, and
//! the limits a clip is held to.

#![expect(
    clippy::cast_precision_loss,
    reason = "a mean over a few hundred thousand cells needs no exact mantissa"
)]

use shiso_audio::{AudioError, MelFrames, MelFrontend, Pcm};

use super::super::config::{BlockDims, PeeConfig};
use super::super::encoder_tests::{reference, refs};
use super::{audio_tokens, chunk_plan, mel_spec, normalize, stack_rows};

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

/// The limit is an hour, and a clip at the limit is taken while one sample
/// past it is not. The boundary is checked under a two-second limit: an
/// hour of mel in a debug build took 4.3 s and a gigabyte, and the limit
/// check is the same comparison at any length.
/// Over a recording of several windows whose mel frame `f` holds `f + 1`,
/// every kept encoder row is the stack of the eight frames it stands for in
/// the whole recording - the windows' left drops line up with the stacking -
/// and every window sees 4 s of context each side where the recording has it.
#[test]
fn every_kept_row_stacks_its_own_frames() {
    let c = config();
    for frames in [4_001usize, 8_000, 12_345, 20_003] {
        let ramp: Vec<f32> = (1..=frames).map(|f| f as f32).collect();
        let plan = chunk_plan(frames, &c);
        assert!(plan.len() > 1, "{frames} frames are several windows");
        let mut kept: Vec<f32> = Vec::new();
        for win in &plan {
            let context = c.window_context * c.stack;
            assert_eq!(win.start - win.enc_start, win.start.min(context));
            assert_eq!(win.right(), (frames - win.end).min(context));
            let stacked = stack_rows(&ramp, 1, win.enc_start, win.len(), c.stack);
            kept.extend_from_slice(
                &stacked[win.left_drop * c.stack..(win.left_drop + win.core) * c.stack],
            );
        }
        assert_eq!(kept.len(), audio_tokens(frames) * c.stack);
        for (row, group) in kept.chunks_exact(c.stack).enumerate() {
            for (i, v) in group.iter().enumerate() {
                let frame = row * c.stack + i;
                let want = if frame < frames {
                    (frame + 1) as f32
                } else {
                    0.0
                };
                assert!(
                    (v - want).abs() < f32::EPSILON,
                    "{frames} frames: row {row} slot {i} holds {v}, not {want}"
                );
            }
        }
    }
}

#[test]
fn normalize_centres_and_scales_each_bin() {
    // Bin 0 constant, bin 1 holds 1 and 3: mean 2, unbiased std sqrt(2).
    let mel = MelFrames::from_rows(vec![5.0, 1.0, 5.0, 3.0], 2).expect("rows");
    let got = normalize(&mel);
    let scale = 1.0 / (2.0f64.sqrt() + 1e-5);
    let want = [0.0, -scale, 0.0, scale];
    for (g, w) in got.iter().zip(want) {
        assert!((f64::from(*g) - w).abs() < 1e-6, "{got:?} vs {want:?}");
    }
    // Unbiased: three rows divide the squared deviations by two.
    let mel = MelFrames::from_rows(vec![0.0, 3.0, 6.0], 1).expect("rows");
    let got = normalize(&mel);
    let scale = 1.0 / (9.0f64.sqrt() + 1e-5);
    assert!((f64::from(got[2]) - 3.0 * scale).abs() < 1e-6, "{got:?}");
}

#[test]
fn stack_rows_pads_the_last_group_with_zeros() {
    let rows: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    // Rows 1..4 of four two-bin rows: three rows, padded to one group of 8.
    let got = stack_rows(&rows, 2, 1, 3, 8);
    let mut want = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    want.resize(16, 0.0);
    assert_eq!(got, want);
    // A whole group is not padded.
    assert_eq!(stack_rows(&rows, 1, 0, 8, 8), rows);
}

#[test]
fn a_clip_at_the_limit_is_accepted_and_one_sample_more_is_refused() {
    use super::super::tower::{MAX_SECONDS_FOR_TESTS, limits_for_tests};
    assert!((MAX_SECONDS_FOR_TESTS - 3600.0).abs() < f64::EPSILON);
    let spec = mel_spec(&config(), &vec![0.0; 128 * 257]);
    let pcm = |samples: usize| Pcm::new(vec![0.0; samples], 16_000).expect("pcm");
    let tower = limits_for_tests(2.0);
    assert!(shiso_audio::clip_for(&tower(spec.clone()), &pcm(32_000)).is_ok());
    let err = shiso_audio::clip_for(&tower(spec.clone()), &pcm(32_001)).expect_err("past 2 s");
    assert!(matches!(err, AudioError::TooLong { .. }), "{err:?}");
    // Past the real limit the refusal comes before any frame is computed.
    let hour = limits_for_tests(MAX_SECONDS_FOR_TESTS);
    let err = shiso_audio::clip_for(&hour(spec), &pcm(3601 * 16_000)).expect_err("past an hour");
    assert!(matches!(err, AudioError::TooLong { .. }), "{err:?}");
}

/// A clip with no whole mel frame makes no row and is refused as empty; the
/// first 10 ms frame makes one row, as the reference's ceiling division of
/// mel frames by eight does, so 50 ms is one row rather than none.
#[test]
fn tower_refuses_a_clip_under_one_token() {
    let tower = super::super::tower::limits_for_tests(super::super::tower::MAX_SECONDS_FOR_TESTS);
    let spec = mel_spec(&config(), &vec![0.0; 128 * 257]);
    let pcm = |samples: usize| Pcm::new(vec![0.01; samples], 16_000).expect("pcm");
    let err = shiso_audio::clip_for(&tower(spec.clone()), &pcm(0)).expect_err("no samples");
    assert!(matches!(err, AudioError::Empty), "{err:?}");
    let clip = shiso_audio::clip_for(&tower(spec), &pcm(800)).expect("50 ms");
    assert_eq!(clip.tokens(), 1);
}

#[test]
#[ignore = "needs the speech encoder export and the reference captures"]
#[expect(
    clippy::print_stderr,
    reason = "the measured agreement is the test's report"
)]
fn mel_matches_reference_short_en() {
    let Some((c, weights_fb)) = reference() else {
        return;
    };
    let Some(dir) = refs("short_en/verbatim") else {
        return;
    };
    let want = super::super::encoder_tests::floats(&dir.join("mel_raw.f32"));
    let pcm = super::super::encoder_tests::clip_pcm("short_en");
    // The export's own preprocessor numbers, not the typed copy above: a
    // frontend that differs from the checkpoint's must fail here for that.
    let frontend = MelFrontend::new(mel_spec(&c, &weights_fb)).expect("spec");
    let mel = frontend.compute(pcm.samples());
    assert_eq!(mel.valid() * c.mels, want.len(), "frame count");
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
        at / c.mels,
        at % c.mels,
        want[at]
    );
    // f64 against the reference's f32 transform: the worst cell is a
    // near-silent bin where the 2^-24 guard dominates.
    assert!(mean <= 1e-4, "mean log-mel difference {mean}");
    assert!(worst <= 1e-2, "worst log-mel difference {worst}");
}

//! The timestamp head on the device against the reference's
//! log-probabilities, and the word times it leads to.
//!
//! Needs a Metal device, `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER` and
//! the captures' alignment dumps; each case says on stderr why it skipped.

#![cfg(target_os = "macos")]

use std::path::Path;

use shiso_audio::transcript::Word;

use super::{ClassRows, bf16_round, evidence};
use crate::audio_ops::Ops;
use crate::audio_ops::store::TensorStore;
use crate::nemotron_transcribe::align::tests::{duration, reference_words};
use crate::nemotron_transcribe::align::{Activity, AlignDesc, align, targets};
use crate::nemotron_transcribe::config::PeeConfig;
use crate::nemotron_transcribe::encoder_tests::floats;
use crate::nemotron_transcribe::spm::tests::{aligned_refs, tokenizer};
use crate::nemotron_transcribe::weights::PeeWeights;

#[test]
fn bf16_rounding_is_to_nearest_even() {
    let round = |bits: u32| bf16_round(f32::from_bits(bits)).to_bits();
    assert_eq!(round(0x3f80_0000), 0x3f80_0000);
    assert_eq!(round(0x3f80_8000), 0x3f80_0000, "a tie to the even below");
    assert_eq!(round(0x3f81_8000), 0x3f82_0000, "a tie to the even above");
    assert_eq!(round(0x3f80_8001), 0x3f81_0000, "past the tie");
    assert!(bf16_round(f32::NAN).is_nan());
}

/// The export, its config and its weights on the device.
fn loaded() -> Option<(TensorStore, PeeConfig, PeeWeights, Ops)> {
    let path = std::env::var_os("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER")?;
    let backend = crate::audio_ops::testing::device()?;
    let ops = Ops::new(backend);
    let store = TensorStore::open_gguf(Path::new(&path)).expect("opens");
    let c = PeeConfig::from_gguf(store.gguf().expect("gguf")).expect("config");
    let w = PeeWeights::load(&ops, &store, &c).expect("the encoder loads");
    Some((store, c, w, ops))
}

/// Words compared with the reference's: how many match exactly, and the
/// worst start and end difference of the rest, seconds.
fn compare(got: &[Word], want: &[(String, f64, f64)]) -> (usize, f64) {
    assert_eq!(got.len(), want.len(), "word count");
    let mut exact = 0;
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.text, w.0, "the words in order");
        if g.start.to_bits() == w.1.to_bits() && g.end.to_bits() == w.2.to_bits() {
            exact += 1;
        } else {
            worst = worst.max((g.start - w.1).abs()).max((g.end - w.2).abs());
        }
    }
    (exact, worst)
}

/// The words of the capture's transcript timed with `log_probs`, against
/// the reference's speaker activity.
fn words_with(dir: &Path, log_probs: &[f32], t: &super::super::align::Targets) -> Vec<Word> {
    let rows80 = floats(&dir.join("diar_probs.f32"));
    let rows10 = floats(&dir.join("spk_probs10.f32"));
    align(
        &AlignDesc {
            targets: t,
            log_probs,
            frames: log_probs.len() / t.columns.len(),
            activity: Some(Activity {
                rows80: &rows80,
                rows10: &rows10,
                speakers: 8,
            }),
            duration: duration(dir),
        },
        2,
    )
    .expect("aligns")
}

#[expect(clippy::print_stderr, reason = "the measured agreement is the report")]
fn head_matches(rel: &str) {
    let Some(dir) = aligned_refs(rel) else {
        return;
    };
    let Some(tok) = tokenizer() else {
        return;
    };
    let Some((store, c, w, ops)) = loaded() else {
        eprintln!("SKIP: no Metal device or no encoder export");
        return;
    };
    let speech = floats(&dir.join("asr_out.f32"));
    let ev = evidence(&ops, &w.ctc, &c, &speech).expect("the head runs");
    let want_lse = floats(&dir.join("ctc_lse.f32"));
    assert_eq!(ev.frames(), want_lse.len(), "{rel}: frames");
    let transcript =
        std::fs::read_to_string(dir.join("ctc_transcript.txt")).expect("the transcript reads");
    let t = targets(&tok, &transcript, ev.frames()).expect("targets");
    let classes = ClassRows::read(&store, &t.columns, c.ctc.d_model).expect("rows");
    let got = ev.log_probs(&classes);
    let want = floats(&dir.join("ctc_logprobs.f32"));
    assert_eq!(got.len(), want.len(), "{rel}: log-probabilities");
    let worst_lp = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Past one bf16 step of the reference's value: what is left once a
    // rounding that went the other way is forgiven.
    let worst_past_step = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs() - bf16_step(*b))
        .fold(0.0f32, f32::max);
    let worst_lse = ev
        .lse
        .iter()
        .zip(&want_lse)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (exact, worst_t) = compare(&words_with(&dir, &got, &t), &reference_words(&dir));
    eprintln!(
        "{rel}: log-prob |d| {worst_lp:.4} ({worst_past_step:.4} past one bf16 step), \
         log-sum-exp |d| {worst_lse:.4}; words timed with ours: {exact} of {} exact, the rest \
         within {worst_t:.3} s",
        t.words.len()
    );
    assert!(
        worst_lse <= LSE_BUDGET,
        "{rel}: log-sum-exp off by {worst_lse}"
    );
    assert!(
        worst_past_step <= LOG_PROB_BUDGET,
        "{rel}: log-probability off by {worst_lp}, {worst_past_step} past one bf16 step"
    );
    assert!(
        exact * 100 >= t.words.len() * 99 && worst_t <= 0.04,
        "{rel}: {exact} of {} words exact, the rest within {worst_t} s",
        t.words.len()
    );
}

/// One bf16 step at `v`: the spacing of bf16 values around it.
fn bf16_step(v: f32) -> f32 {
    let exponent = (v.abs().to_bits() >> 23).cast_signed() - 127;
    2f32.powi(exponent - 7)
}

/// The reference's CPU run multiplies its f16 weights with f16 activations,
/// so its logits carry rounding ours do not. Measured worst on the M4:
/// `short_en` 0.0027, `meeting_60s` 0.0316, `meeting_10m` 0.0524.
const LSE_BUDGET: f32 = 0.1;
/// Worst log-probability difference past one bf16 step of the reference's
/// value (both sides round to bf16, and a value near a rounding boundary
/// lands on either side). Measured worst on the M4: `short_en` 0.0000,
/// `meeting_60s` 0.0312, `meeting_10m` 0.0469; the worst raw difference is one
/// step, 0.125, on all three. Every word time matched the reference's to the
/// millisecond on all three.
const LOG_PROB_BUDGET: f32 = 0.1;

#[test]
#[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
fn short_en_head_matches_reference() {
    head_matches("short_en/verbatim");
}

#[test]
#[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
fn meeting_60s_head_matches_reference() {
    head_matches("meeting_60s/verbatim");
}

#[test]
#[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
fn meeting_10m_head_matches_reference() {
    head_matches("meeting_10m/verbatim");
}

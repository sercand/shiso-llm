//! The arithmetic of a scored answer.
//!
//! A scoring pass hands back the log-probability of each label at the answer
//! position; everything a caller sees is derived from that vector here, in
//! `f64`, so a probability that must sum to one does and a confidence that
//! must sit in `[0, 1]` cannot drift out of it on a float edge.

use shiso_core::TokenId;

/// The log-probability of each `labels` entry under the full-vocab logits
/// `row` of the answer position.
///
/// The vocabulary-wide normalizer is the same for every label and cancels in
/// the softmax over labels; the mass is not a softmax artifact, so the row's
/// log-sum-exp is what turns raw logits into real log-probabilities.
#[must_use]
pub fn label_logprobs(row: &[f32], labels: &[TokenId]) -> Vec<f64> {
    let norm = logsumexp(row);
    labels
        .iter()
        .map(|label| {
            let at = usize::try_from(label.get()).unwrap_or(usize::MAX);
            // A label id past the row is not a caller's mistake to make: the
            // HTTP layer checked the id against the vocabulary before queueing.
            // Clamping to the last row entry keeps a mismatch a wrong number
            // the scheduler test would catch, never an index panic on the
            // engine thread.
            let logit = f64::from(row.get(at).copied().unwrap_or(f32::NEG_INFINITY));
            logit - norm
        })
        .collect()
}

/// The softmax of `logprobs / temperature` over the labels.
///
/// The caller guarantees `temperature > 0`; a zero or negative temperature is
/// refused at the wire, not softened here.
#[must_use]
pub fn softmax(logprobs: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logprobs.iter().map(|lp| lp / temperature).collect();
    let max = scaled
        .iter()
        .fold(f64::NEG_INFINITY, |best, value| best.max(*value));
    let exps: Vec<f64> = scaled.iter().map(|value| (value - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|value| value / sum).collect()
}

/// The probability mass the labels carry together: `Σ exp(logprob)`.
///
/// Independent of the temperature by construction — a softmax over the labels
/// would always sum to one and say nothing about how much of the distribution
/// lives outside them.
#[must_use]
pub fn label_mass(logprobs: &[f64]) -> f64 {
    logprobs.iter().map(|lp| lp.exp()).sum()
}

/// The expected level of a score answer: `Σ i · p_i`.
#[expect(
    clippy::cast_precision_loss,
    reason = "level indices are single digits and exact in f64"
)]
#[must_use]
pub fn expected_score(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(index, p)| index as f64 * p)
        .sum()
}

/// How far a choice answer stands above a uniform draw:
/// `clamp01((n · max(p) − 1) / (n − 1))`, `1.0` when there is one label, and
/// `0.0` when there are none (the formula has no top to take).
///
/// A uniform draw over `n` labels has `max = 1/n` and scores 0; putting all
/// mass on one label scores 1. `choice_confidence_of_a_two_label_draw_is_twice_the_top_minus_one`
/// below is the published worked example: `[0.1, 0.9]` over two labels is
/// `2 · 0.9 − 1 = 0.8`.
#[expect(
    clippy::cast_precision_loss,
    reason = "label counts are single digits and exact in f64"
)]
#[must_use]
pub fn choice_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.is_empty() {
        return 0.0;
    }
    if probabilities.len() == 1 {
        return 1.0;
    }
    let top = probabilities
        .iter()
        .fold(f64::NEG_INFINITY, |best, p| best.max(*p));
    let n = probabilities.len() as f64;
    ((n * top - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

/// How concentrated a score answer is around its top level:
/// `max(0, 1 − spread / uniform_spread)`.
///
/// `spread = Σ p_i · |i − top|` is the expected distance from the top level;
/// `uniform_spread = Σ |i − (n − 1) / 2| / n` is the same distance for a
/// uniform draw, so a tight answer scores near 1 and a draw split across the
/// scale scores 0. The three published worked examples are pinned by the tests
/// below: `[0, .14, .86, 0, 0]` is `1 − .14/1.2`, `[0, 0, .48, .52]` is `.52`,
/// and `[.5, 0, 0, 0, .5]` is `0` — a split tail, not a negative number. A
/// vector with fewer than two levels scores `1.0`: its uniform denominator
/// does not exist, and one level is as concentrated as an answer gets.
#[expect(
    clippy::cast_precision_loss,
    reason = "level indices and counts are single digits and exact in f64"
)]
#[must_use]
pub fn score_confidence(probabilities: &[f64]) -> f64 {
    let n = probabilities.len();
    if n <= 1 {
        return 1.0;
    }
    let top = probabilities
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |best, (index, p)| {
            if *p > best.1 { (index, *p) } else { best }
        })
        .0;
    // `abs_diff` rather than subtraction of the indices: `i` runs below `top`
    // as often as above it, and unsigned subtraction would wrap there.
    let spread: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| p * i.abs_diff(top) as f64)
        .sum();
    let mid = (n - 1) as f64 / 2.0;
    let uniform_spread: f64 = (0..n).map(|i| ((i as f64) - mid).abs()).sum::<f64>() / n as f64;
    (1.0 - spread / uniform_spread).max(0.0)
}

/// `log Σ exp(row)` in `f64`, shifted by the row's maximum so a wide logit
/// range neither underflows nor overflows.
fn logsumexp(row: &[f32]) -> f64 {
    let max = row
        .iter()
        .fold(f32::NEG_INFINITY, |best, value| best.max(*value));
    if max == f32::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let max = f64::from(max);
    let sum: f64 = row
        .iter()
        .map(|value| (f64::from(*value) - max).exp())
        .sum();
    max + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // The row is an f32 row because a logits row is one; ln 3 rounded into it
    // is the exact value the assertions then expect.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "ln 3 rounded into the f32 a logits row is stored in"
    )]
    fn label_logprobs_are_row_value_minus_logsumexp() {
        // A row of [0, ln 3, 0, 0] over vocab {A:0, B:1}: logsumexp = ln(1+3+1+1) = ln 6.
        let row = vec![0.0f32, (3.0f64).ln() as f32, 0.0, 0.0];
        let labels = [TokenId::new(0), TokenId::new(1)];
        let lp = label_logprobs(&row, &labels);
        assert!((lp[0] - (1.0f64 / 6.0).ln()).abs() < 1e-6);
        assert!((lp[1] - (3.0f64 / 6.0).ln()).abs() < 1e-6);
    }

    #[test]
    fn softmax_at_unit_temperature_is_plain_softmax_over_labels() {
        // lp = [0, ln 3] -> [0.25, 0.75].
        let lp = vec![0.0, (3.0f64).ln()];
        let p = softmax(&lp, 1.0);
        assert!((p[0] - 0.25).abs() < 1e-9);
        assert!((p[1] - 0.75).abs() < 1e-9);
    }

    #[test]
    fn softmax_scales_by_temperature() {
        // lp = [0, ln 3] with T = 2: softmax([0, (ln 3)/2]) = softmax([0, ln √3])
        // = [1/(1+√3), √3/(1+√3)].
        let lp = vec![0.0, (3.0f64).ln()];
        let p = softmax(&lp, 2.0);
        let root3 = 3.0f64.sqrt();
        assert!((p[0] - 1.0 / (1.0 + root3)).abs() < 1e-9);
        assert!((p[1] - root3 / (1.0 + root3)).abs() < 1e-9);
    }

    #[test]
    fn label_mass_is_the_sum_of_label_probabilities() {
        let lp = vec![(0.2f64).ln(), (0.3f64).ln()];
        assert!((label_mass(&lp) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn expected_score_weights_levels_by_index() {
        // [0, .14, .86, 0, 0] → .14·1 + .86·2 = 1.86.
        let p = vec![0.0, 0.14, 0.86, 0.0, 0.0];
        assert!((expected_score(&p) - 1.86).abs() < 1e-12);
    }

    #[test]
    fn choice_confidence_of_a_uniform_draw_is_zero() {
        assert!(choice_confidence(&[0.25; 4]).abs() < 1e-9);
    }

    #[test]
    fn choice_confidence_of_a_two_label_draw_is_twice_the_top_minus_one() {
        assert!((choice_confidence(&[0.1, 0.9]) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn choice_confidence_of_a_single_label_question_is_one() {
        assert!((choice_confidence(&[1.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn score_confidence_of_the_published_tight_example() {
        // [0, .14, .86, 0, 0]: top = 2, spread = .14, uniform = 6/5 = 1.2.
        let p = vec![0.0, 0.14, 0.86, 0.0, 0.0];
        assert!((score_confidence(&p) - (1.0 - 0.14 / 1.2)).abs() < 1e-9);
    }

    #[test]
    fn score_confidence_of_the_published_split_top_example() {
        // [0, 0, .48, .52]: top = 3, spread = .48, uniform = 4/4 = 1.0 → .52.
        let p = vec![0.0, 0.0, 0.48, 0.52];
        assert!((score_confidence(&p) - 0.52).abs() < 1e-9);
    }

    #[test]
    fn score_confidence_of_a_split_tails_example_is_zero() {
        // [.5, 0, 0, 0, .5]: first argmax = 0, spread = 2 > uniform 1.2 → 0, not negative.
        let p = vec![0.5, 0.0, 0.0, 0.0, 0.5];
        assert!(score_confidence(&p).abs() < 1e-9);
    }
}

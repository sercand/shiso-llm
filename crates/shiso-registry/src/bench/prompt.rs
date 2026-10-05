//! Synthetic prompts of a chosen token length.
//!
//! The bench has no tokenizer: it asks the server how many tokens two filler
//! texts render to and solves for the words a target length needs. That also
//! counts the chat template or the decision framing, which a local tokenizer
//! would have to reproduce.

use std::future::Future;

use anyhow::bail;

/// Deterministic filler prose, different for every run.
///
/// The run number leads the text so no two runs share a prompt prefix: the
/// server's prefix cache would otherwise answer most of a repeated prompt and
/// the prefill rate would measure the cache. `lead` is the instruction the
/// text opens with.
pub(super) fn filler(lead: &str, words: usize, run: u32) -> String {
    const WORDS: [&str; 32] = [
        "river", "lantern", "copper", "meadow", "signal", "harbor", "orchard", "granite",
        "whisper", "compass", "timber", "glacier", "ember", "violet", "canyon", "ledger", "falcon",
        "prism", "cinder", "thistle", "beacon", "mosaic", "quarry", "saddle", "tundra", "velvet",
        "anchor", "bramble", "cobalt", "dune", "fable", "garnet",
    ];
    // A fixed linear congruential sequence: reproducible, and varied enough
    // that the tokenizer does not fold the text into a few long tokens.
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15 ^ u64::from(run);
    let mut text = format!("Run {run}. {lead}\n");
    for i in 0..words {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let idx = usize::try_from(state >> 59).unwrap_or(0) % WORDS.len();
        text.push_str(WORDS[idx]);
        text.push(if i % 12 == 11 { '.' } else { ' ' });
    }
    text
}

/// How many filler words make a prompt of a given token count, for this
/// model's tokenizer and framing.
pub(super) struct WordRate {
    pub(super) tokens_per_word: f64,
    pub(super) overhead_tokens: f64,
}

impl WordRate {
    /// Two short prompts give the tokens per filler word and the framing's
    /// fixed cost. `count` sends `filler(lead, words, run)` and returns the
    /// prompt tokens the server reported.
    pub(super) async fn calibrate<F, Fut>(count: F) -> anyhow::Result<Self>
    where
        F: Fn(usize, u32) -> Fut,
        Fut: Future<Output = anyhow::Result<u64>>,
    {
        let (small, large) = (64_usize, 512_usize);
        let ta = count(small, 1000).await?;
        let tb = count(large, 1001).await?;
        if tb <= ta {
            bail!("the server reported {ta} and {tb} prompt tokens for a short and a long prompt");
        }
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let rate = (tb - ta) as f64 / (large - small) as f64;
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let overhead = ta as f64 - rate * small as f64;
        Ok(Self {
            tokens_per_word: rate,
            overhead_tokens: overhead.max(0.0),
        })
    }

    pub(super) fn words_for(&self, tokens: u64) -> usize {
        #[expect(
            clippy::cast_precision_loss,
            reason = "prompt sizes are far below 2^52"
        )]
        let want = (tokens as f64 - self.overhead_tokens).max(1.0) / self.tokens_per_word;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a positive word count well inside usize"
        )]
        let words = want.round() as usize;
        words.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filler_differs_between_runs_from_the_first_token() {
        let (a, b) = (filler("Go.", 50, 1), filler("Go.", 50, 2));
        assert_ne!(a.split_whitespace().nth(1), b.split_whitespace().nth(1));
        assert_eq!(filler("Go.", 50, 1), a);
    }

    #[tokio::test]
    async fn calibration_solves_for_words_per_token() {
        // 20 framing tokens and 1.5 tokens a word.
        let rate = WordRate::calibrate(|words, _| async move {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss,
                reason = "test"
            )]
            let tokens = (20.0 + 1.5 * words as f64) as u64;
            Ok(tokens)
        })
        .await
        .expect("calibrates");
        assert!((rate.tokens_per_word - 1.5).abs() < 1e-9);
        assert_eq!(rate.words_for(320), 200);
    }
}

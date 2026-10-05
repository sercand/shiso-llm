//! Chat: prefill and decode tokens a second at each prompt length.

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::client::Server;
use super::prompt::{WordRate, filler};
use super::{BenchDesc, BenchLog, BenchResult, median, round2, run_label, say};

/// The server has no `min_tokens` stage yet, so the decode length is the
/// model's choice; an open-ended writing task keeps it going past the budget,
/// where a "summarize" ends after a few dozen tokens.
const LEAD: &str = "Write a long, detailed story that uses every word in the notes below, at least two thousand words.";

/// One prompt size's measurement, the median over the runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatResult {
    /// Prompt tokens the server computed (cache hits excluded).
    pub(crate) prompt_tokens: u64,
    /// Tokens generated per run (the shortest run's count).
    pub(crate) decode_tokens: u64,
    /// Prompt tokens per second, from the server's time to first token.
    pub(crate) prefill_tokens_per_second: f64,
    /// Generated tokens per second after the first.
    pub(crate) decode_tokens_per_second: f64,
    /// Time from accepting the request to the first token.
    pub(crate) time_to_first_token_ms: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Sample {
    pub(super) prompt_tokens: u64,
    pub(super) decode_tokens: u64,
    pub(super) ttft_ms: f64,
    pub(super) decode_tps: f64,
}

impl Sample {
    pub(super) fn median(samples: &[Self], decode_tokens: u64) -> ChatResult {
        #[expect(
            clippy::cast_precision_loss,
            reason = "token counts are far below 2^52"
        )]
        let prefill = median(
            samples
                .iter()
                .map(|s| s.prompt_tokens as f64 / (s.ttft_ms / 1000.0))
                .collect(),
        );
        ChatResult {
            prompt_tokens: samples.iter().map(|s| s.prompt_tokens).max().unwrap_or(0),
            decode_tokens: samples
                .iter()
                .map(|s| s.decode_tokens)
                .min()
                .unwrap_or(decode_tokens),
            prefill_tokens_per_second: round2(prefill),
            decode_tokens_per_second: round2(median(
                samples.iter().map(|s| s.decode_tps).collect(),
            )),
            time_to_first_token_ms: round2(median(samples.iter().map(|s| s.ttft_ms).collect())),
        }
    }
}

async fn complete(server: &Server, prompt: &str, tokens: u64) -> anyhow::Result<Value> {
    let body = json!({
        "model": server.model_id,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": tokens,
        // Asked for so a server that has the stage decodes the full budget;
        // one that lacks it ignores the field, and the record keeps the
        // shortest decode a run actually made.
        "min_tokens": tokens,
        "temperature": 0.0,
        "stream": false,
    });
    let (value, _) = server.post_json("/v1/chat/completions", &body).await?;
    Ok(value["usage"].clone())
}

async fn measure(server: &Server, prompt: &str, tokens: u64) -> anyhow::Result<Sample> {
    let usage = complete(server, prompt, tokens).await?;
    let timings = &usage["timings"];
    let (Some(ttft_ms), Some(decode_tps)) = (
        timings["time_to_first_token_ms"].as_f64(),
        timings["decode_tokens_per_second"].as_f64(),
    ) else {
        bail!("the server does not report usage.timings; it needs a newer shiso build");
    };
    let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    let decode_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
    if ttft_ms <= 0.0 || decode_tps <= 0.0 || decode_tokens < 2 {
        bail!("the server reported no usable timing ({usage})");
    }
    if decode_tokens < tokens {
        tracing::warn!(decode_tokens, asked = tokens, "the turn ended early");
    }
    Ok(Sample {
        prompt_tokens: prompt_tokens.saturating_sub(cached),
        decode_tokens,
        ttft_ms,
        decode_tps,
    })
}

/// Calibrates the prompt length, then measures every prompt size.
pub(super) async fn run(
    server: &Server,
    desc: &BenchDesc,
    log: Option<&BenchLog>,
) -> anyhow::Result<Vec<BenchResult>> {
    if desc.decode_tokens < 2 {
        bail!("a chat bench needs --decode-tokens >= 2");
    }
    // A prompt that leaves no room for the decode budget would end at the
    // context limit and record a decode that measured the limit instead.
    if let Some(p) = desc
        .prompt_tokens
        .iter()
        .find(|p| **p + desc.decode_tokens > server.context_length)
    {
        bail!(
            "prompt size {p} plus {} decode tokens exceeds the server's {}-token context; restart it with a larger --ctx",
            desc.decode_tokens,
            server.context_length
        );
    }
    let words = WordRate::calibrate(|words, run| async move {
        let usage = complete(server, &filler(LEAD, words, run), 1).await?;
        Ok(usage["prompt_tokens"].as_u64().unwrap_or(0))
    })
    .await?;
    say(
        log,
        format!(
            "Calibrated: {:.2} tokens per filler word, {:.0} template tokens",
            words.tokens_per_word, words.overhead_tokens
        ),
    );

    let mut results = Vec::new();
    for &target in &desc.prompt_tokens {
        let word_count = words.words_for(target);
        let mut samples = Vec::new();
        // The first run is discarded: it pays for kernel compilation, graph
        // capture and cold pages, none of which a steady server repeats.
        for run in 0..=desc.runs {
            let sample = measure(server, &filler(LEAD, word_count, run), desc.decode_tokens)
                .await
                .with_context(|| format!("prompt size {target}, run {run}"))?;
            say(
                log,
                format!(
                    "  {target} tokens, {}: {} prompt tokens, ttft {:.0} ms, decode {:.1} tok/s",
                    run_label(run),
                    sample.prompt_tokens,
                    sample.ttft_ms,
                    sample.decode_tps
                ),
            );
            if run > 0 {
                samples.push(sample);
            }
        }
        let result = Sample::median(&samples, desc.decode_tokens);
        say(
            log,
            format!(
                "{target} tokens: prefill {:.0} tok/s, decode {:.1} tok/s (median of {})",
                result.prefill_tokens_per_second,
                result.decode_tokens_per_second,
                samples.len()
            ),
        );
        results.push(BenchResult::Chat(result));
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_and_prefill_rate_come_from_the_runs() {
        let s = |ttft_ms: f64, decode_tps: f64| Sample {
            prompt_tokens: 1000,
            decode_tokens: 128,
            ttft_ms,
            decode_tps,
        };
        let r = Sample::median(&[s(500.0, 40.0), s(250.0, 44.0), s(1000.0, 42.0)], 128);
        assert!((r.prefill_tokens_per_second - 2000.0).abs() < 1e-9);
        assert!((r.decode_tokens_per_second - 42.0).abs() < 1e-9);
        assert!((r.time_to_first_token_ms - 500.0).abs() < 1e-9);
    }
}

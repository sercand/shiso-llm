//! Decision: how long one decision takes at each prompt length.
//!
//! A decision is a prefill and one scoring step, so its latency is the cost a
//! caller waits on; the prompt rate it implies is reported beside it because
//! it is the number that carries over to other prompt lengths.

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::client::Server;
use super::prompt::{WordRate, filler};
use super::{BenchDesc, BenchLog, BenchResult, median, round2, run_label, say};

const LEAD: &str = "Read the notes below, then answer the question about them.";

/// One prompt size's measurement, the median over the runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionResult {
    /// Prompt tokens the server scored.
    pub(crate) prompt_tokens: u64,
    /// Wall time of one `/v1/decisions` request with one yes/no question.
    pub(crate) latency_ms: f64,
    /// Prompt tokens per second of that latency.
    pub(crate) prompt_tokens_per_second: f64,
}

/// One yes/no question over `input`; the smallest decision a caller can ask.
async fn decide(server: &Server, input: &str) -> anyhow::Result<(u64, f64)> {
    let body = json!({
        "model": server.model_id,
        "input": input,
        "questions": [{
            "id": "q1",
            "type": "yes_no",
            "question": "The notes mention a river.",
        }],
    });
    let (value, elapsed): (Value, _) = server.post_json("/v1/decisions", &body).await?;
    let tokens = value["usage"]["prompt_tokens"]
        .as_u64()
        .context("the decision answer carries no usage.prompt_tokens")?;
    Ok((tokens, elapsed.as_secs_f64() * 1000.0))
}

pub(super) async fn run(
    server: &Server,
    desc: &BenchDesc,
    log: Option<&BenchLog>,
) -> anyhow::Result<Vec<BenchResult>> {
    if let Some(p) = desc
        .prompt_tokens
        .iter()
        .find(|p| **p >= server.context_length)
    {
        bail!(
            "prompt size {p} does not fit the server's {}-token context; restart it with a larger --ctx",
            server.context_length
        );
    }
    let words = WordRate::calibrate(|words, run| async move {
        Ok(decide(server, &filler(LEAD, words, run)).await?.0)
    })
    .await?;
    say(
        log,
        format!(
            "Calibrated: {:.2} tokens per filler word, {:.0} framing tokens",
            words.tokens_per_word, words.overhead_tokens
        ),
    );

    let mut results = Vec::new();
    for &target in &desc.prompt_tokens {
        let word_count = words.words_for(target);
        let mut samples = Vec::new();
        for run in 0..=desc.runs {
            let (tokens, ms) = decide(server, &filler(LEAD, word_count, run))
                .await
                .with_context(|| format!("prompt size {target}, run {run}"))?;
            say(
                log,
                format!(
                    "  {target} tokens, {}: {tokens} prompt tokens, {ms:.0} ms",
                    run_label(run)
                ),
            );
            if run > 0 {
                samples.push((tokens, ms));
            }
        }
        let latency = median(samples.iter().map(|s| s.1).collect());
        #[expect(
            clippy::cast_precision_loss,
            reason = "token counts are far below 2^52"
        )]
        let rate = median(
            samples
                .iter()
                .map(|(t, ms)| *t as f64 / (ms / 1000.0))
                .collect(),
        );
        let result = DecisionResult {
            prompt_tokens: samples.iter().map(|s| s.0).max().unwrap_or(0),
            latency_ms: round2(latency),
            prompt_tokens_per_second: round2(rate),
        };
        say(
            log,
            format!(
                "{target} tokens: {:.0} ms per decision, {:.0} prompt tok/s (median of {})",
                result.latency_ms,
                result.prompt_tokens_per_second,
                samples.len()
            ),
        );
        results.push(BenchResult::Decision(result));
    }
    Ok(results)
}

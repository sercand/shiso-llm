//! The HTTP side of a bench: one running `shiso serve`, and the entry of its
//! model listing that serves the task being measured.

use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde_json::Value;

use super::BenchKind;

/// Image defaults the generator's listing entry reports.
#[derive(Debug, Clone, Copy)]
pub(super) struct ImageDefaults {
    pub(super) steps: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

/// The served model a bench sends its requests to.
pub(super) struct Server {
    client: reqwest::Client,
    base: String,
    /// The id the requests name in `model`.
    pub(super) model_id: String,
    /// Tokens one session may hold; zero for a model without a context.
    pub(super) context_length: u64,
    /// The generator's defaults, on an image entry.
    pub(super) image: Option<ImageDefaults>,
}

impl Server {
    /// Reads the listing and picks the entry that serves `kind`.
    ///
    /// One server can list a language model, a recognizer and a generator
    /// side by side, so the entry is chosen by what it reports rather than by
    /// position: a context for a language model, image defaults for a
    /// generator, neither for a recognizer.
    pub(super) async fn connect(url: &str, kind: BenchKind) -> anyhow::Result<Self> {
        let base = url.trim_end_matches('/').to_string();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(1800))
            .build()?;
        let models: Value = client
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .with_context(|| format!("reaching {base}; is `shiso serve` running?"))?
            .error_for_status()?
            .json()
            .await?;
        let entries = models["data"].as_array().cloned().unwrap_or_default();
        let ctx = |e: &Value| e["context_length"].as_u64().unwrap_or(0);
        let entry = entries
            .iter()
            .find(|e| match kind {
                BenchKind::Chat | BenchKind::Decision => ctx(e) > 0,
                BenchKind::Image => e["image"].is_object(),
                BenchKind::Transcription => ctx(e) == 0 && !e["image"].is_object(),
            })
            .with_context(|| {
                format!(
                    "the server at {base} lists no model that serves {kind}; the image entry needs a shiso build that reports its defaults"
                )
            })?;
        let image = entry["image"].is_object().then(|| {
            let n = |k: &str| {
                entry["image"][k]
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0)
            };
            ImageDefaults {
                steps: n("steps"),
                width: n("width"),
                height: n("height"),
            }
        });
        Ok(Self {
            client,
            base,
            model_id: entry["id"]
                .as_str()
                .context("the listing entry has no id")?
                .to_string(),
            context_length: ctx(entry),
            image,
        })
    }

    /// Posts a JSON body and returns the parsed answer and the wall time it
    /// took, refusing any status but success.
    pub(super) async fn post_json(
        &self,
        path: &str,
        body: &Value,
    ) -> anyhow::Result<(Value, Duration)> {
        let started = Instant::now();
        let resp = self
            .client
            .post(format!("{}{path}", self.base))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        let value: Value = resp.json().await?;
        let elapsed = started.elapsed();
        if !status.is_success() {
            bail!("{path} answered {status}: {value}");
        }
        Ok((value, elapsed))
    }

    /// Posts a multipart form and returns the response, still streaming,
    /// with the instant the request was sent.
    pub(super) async fn post_form(
        &self,
        path: &str,
        form: reqwest::multipart::Form,
    ) -> anyhow::Result<(reqwest::Response, Instant)> {
        let started = Instant::now();
        let resp = self
            .client
            .post(format!("{}{path}", self.base))
            .multipart(form)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("{path} answered {status}: {text}");
        }
        Ok((resp, started))
    }
}

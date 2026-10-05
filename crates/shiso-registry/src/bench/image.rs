//! Image: seconds per image at a size and step count.
//!
//! The steps are the server's (`--image-default-steps`; the wire has no
//! field for them), read from its listing so the record says what the time
//! was measured at. The size is read back from the PNG the server returns,
//! because the server rounds a requested size to what the model can render.

use anyhow::{Context, bail};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::Server;
use super::{BenchDesc, BenchLog, BenchResult, median, round2, run_label, say};

/// One size's measurement, the median over the runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImageResult {
    /// Output width in pixels.
    pub(crate) width: u32,
    /// Output height in pixels.
    pub(crate) height: u32,
    /// Denoising steps per image.
    pub(crate) steps: u32,
    /// Wall time of one single-image request.
    pub(crate) seconds_per_image: f64,
}

/// The width and height in a PNG's header.
fn png_size(bytes: &[u8]) -> Option<(u32, u32)> {
    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < 24 || !bytes.starts_with(SIGNATURE) || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let read =
        |at: usize| u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    Some((read(16), read(20)))
}

async fn generate(
    server: &Server,
    size: Option<&str>,
    run: u32,
) -> anyhow::Result<((u32, u32), f64)> {
    let mut body = json!({
        "model": server.model_id,
        // A different prompt and seed each run, so nothing a server might
        // cache between identical requests answers the next one.
        "prompt": format!("Run {run}. A watercolor lighthouse on a sea cliff at dawn, gulls, soft light."),
        "n": 1,
        "seed": run,
        "response_format": "b64_json",
        "output_format": "png",
    });
    if let Some(size) = size {
        body["size"] = json!(size);
    }
    let (value, elapsed) = server.post_json("/v1/images/generations", &body).await?;
    let b64 = value["data"][0]["b64_json"]
        .as_str()
        .context("the image answer carries no data[0].b64_json")?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .context("the image is not base64")?;
    let dims = png_size(&bytes).context("the image is not a PNG")?;
    Ok((dims, elapsed.as_secs_f64()))
}

pub(super) async fn run(
    server: &Server,
    desc: &BenchDesc,
    log: Option<&BenchLog>,
) -> anyhow::Result<Vec<BenchResult>> {
    let defaults = server
        .image
        .context("the image entry reports no defaults")?;
    if defaults.steps == 0 {
        bail!("the server reports zero denoising steps");
    }
    let sizes: Vec<Option<&str>> = if desc.image_sizes.is_empty() {
        vec![None]
    } else {
        desc.image_sizes.iter().map(|s| Some(s.as_str())).collect()
    };
    let mut results = Vec::new();
    for size in sizes {
        let label = size.map_or_else(
            || format!("{}x{} (server default)", defaults.width, defaults.height),
            ToString::to_string,
        );
        let mut seconds = Vec::new();
        let mut dims = (0, 0);
        for run in 0..=desc.runs {
            let (got, s) = generate(server, size, run)
                .await
                .with_context(|| format!("size {label}, run {run}"))?;
            say(
                log,
                format!(
                    "  {label}, {}: {}x{} in {s:.2} s",
                    run_label(run),
                    got.0,
                    got.1
                ),
            );
            if run > 0 {
                seconds.push(s);
                dims = got;
            }
        }
        let result = ImageResult {
            width: dims.0,
            height: dims.1,
            steps: defaults.steps,
            seconds_per_image: round2(median(seconds)),
        };
        say(
            log,
            format!(
                "{}x{}, {} steps: {:.2} s per image",
                result.width, result.height, result.steps, result.seconds_per_image
            ),
        );
        results.push(BenchResult::Image(result));
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_size_is_read_from_the_png_header() {
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        png.extend_from_slice(&1024u32.to_be_bytes());
        png.extend_from_slice(&768u32.to_be_bytes());
        assert_eq!(png_size(&png), Some((1024, 768)));
        assert_eq!(png_size(b"GIF89a"), None);
    }
}

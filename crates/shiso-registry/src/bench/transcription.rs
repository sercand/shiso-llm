//! Transcription: how much faster than real time a clip is transcribed, and
//! how soon the first text arrives.
//!
//! The clip is a WAV file given with `--audio`, or, on macOS, speech the
//! system synthesizer reads from a fixed passage. Its length comes from the
//! WAV header: the server's own duration is rounded up to whole seconds, too
//! coarse for a ratio over a one-minute clip.

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::client::Server;
use super::{BenchDesc, BenchLog, BenchResult, median, round2, run_label, say};

/// About a minute of speech: long enough that the fixed cost of a request is
/// small beside the work, short enough to repeat a few times.
const PASSAGE: &str = "The lighthouse keeper climbed the spiral stairs every evening at dusk. \
    From the gallery she could see the harbor, the fishing boats returning with their nets, \
    and the long line of the breakwater where children waited for their fathers. \
    She trimmed the wick, polished the great lens, and wrote the weather in her ledger: \
    wind from the southwest, a falling glass, rain likely before midnight. \
    Some nights the fog came in so thick that the beam seemed to stop an arm's length from the glass, \
    and then she rang the bell by hand, once every thirty seconds, until the morning. \
    In forty years she had missed only one evening, the night her daughter was born, \
    and even then her husband had kept the light burning in her place. \
    When the harbor board finally sent an engineer to install an automatic lamp, \
    she showed him every gear and spring, and then walked down the stairs for the last time.";

/// One clip's measurement, the median over the runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TranscriptionResult {
    /// The clip: its file name, or `synthesized speech`.
    pub(crate) clip: String,
    /// The clip's length.
    pub(crate) audio_seconds: f64,
    /// Wall time from sending the request to the final event.
    pub(crate) wall_seconds: f64,
    /// Audio seconds transcribed per wall second: 30 is thirty times faster
    /// than real time.
    pub(crate) realtime_multiple: f64,
    /// Wall time to the first text delta of the stream.
    pub(crate) first_text_ms: f64,
}

/// The length of a PCM WAV file, from its `fmt ` byte rate and `data` size.
fn wav_seconds(bytes: &[u8]) -> Option<f64> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let u32_at = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let mut at = 12;
    let mut byte_rate = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32_at(at + 4)? as usize;
        let body = at + 8;
        if id == b"fmt " {
            byte_rate = u32_at(body + 8);
        } else if id == b"data" {
            let rate = byte_rate.filter(|r| *r > 0)?;
            // A streamed writer may leave the size unset; the rest of the
            // file is the data then.
            let len = size.min(bytes.len() - body);
            return Some(f64::from(u32::try_from(len).ok()?) / f64::from(rate));
        }
        // Chunks are padded to an even length.
        at = body + size + (size & 1);
    }
    None
}

/// The clip to transcribe and its name for the record.
fn load_clip(desc: &BenchDesc) -> anyhow::Result<(Vec<u8>, String)> {
    if let Some(path) = &desc.audio {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let name = path
            .file_name()
            .map_or_else(|| "clip".to_string(), |n| n.to_string_lossy().to_string());
        return Ok((bytes, name));
    }
    if !cfg!(target_os = "macos") {
        bail!("a transcription bench needs --audio <file.wav> on this host");
    }
    let out = std::env::temp_dir().join(format!("shiso-bench-{}.wav", std::process::id()));
    let status = Command::new("say")
        .arg("-o")
        .arg(&out)
        .args(["--file-format=WAVE", "--data-format=LEI16@16000", PASSAGE])
        .status()
        .context("running `say` to synthesize the clip; pass --audio instead")?;
    let bytes = fs::read(&out);
    let _ = fs::remove_file(&out);
    if !status.success() {
        bail!("`say` failed to synthesize the clip; pass --audio instead");
    }
    Ok((bytes?, "synthesized speech".to_string()))
}

/// One streamed transcription: (wall seconds, ms to the first text).
async fn transcribe(server: &Server, clip: &[u8], name: &Path) -> anyhow::Result<(f64, f64)> {
    let part = reqwest::multipart::Part::bytes(clip.to_vec())
        .file_name(name.display().to_string())
        .mime_str("audio/wav")?;
    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", server.model_id.clone())
        .text("response_format", "json")
        .text("stream", "true");
    let (mut resp, started) = server.post_form("/v1/audio/transcriptions", form).await?;
    let mut buffer = String::new();
    let mut first_text = None;
    while let Some(chunk) = resp.chunk().await? {
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(end) = buffer.find("\n\n") {
            let event: String = buffer.drain(..end + 2).collect();
            for data in event.lines().filter_map(|l| l.strip_prefix("data:")) {
                let data = data.trim();
                if data == "[DONE]" || data.is_empty() {
                    continue;
                }
                let value: Value = serde_json::from_str(data)
                    .with_context(|| format!("parsing the stream event {data}"))?;
                match value["type"].as_str() {
                    Some("transcript.text.delta") if first_text.is_none() => {
                        first_text = Some(started.elapsed());
                    }
                    Some("transcript.text.done") => {
                        let wall = started.elapsed();
                        let first = first_text.unwrap_or(wall);
                        return Ok((wall.as_secs_f64(), first.as_secs_f64() * 1000.0));
                    }
                    Some("error") => bail!("the transcription failed: {}", value["error"]),
                    _ => {}
                }
            }
        }
    }
    bail!("the stream ended without a transcript.text.done event")
}

pub(super) async fn run(
    server: &Server,
    desc: &BenchDesc,
    log: Option<&BenchLog>,
) -> anyhow::Result<Vec<BenchResult>> {
    let (clip, name) = load_clip(desc)?;
    let audio_seconds = wav_seconds(&clip)
        .filter(|s| *s > 0.0)
        .context("the clip is not a PCM WAV file with a readable length")?;
    say(log, format!("Clip: {name}, {audio_seconds:.1} s"));
    let mut walls = Vec::new();
    let mut firsts = Vec::new();
    for run in 0..=desc.runs {
        let (wall, first) = transcribe(server, &clip, Path::new("bench.wav"))
            .await
            .with_context(|| format!("run {run}"))?;
        say(
            log,
            format!(
                "  {}: {wall:.2} s, first text after {first:.0} ms ({:.1}x real time)",
                run_label(run),
                audio_seconds / wall
            ),
        );
        if run > 0 {
            walls.push(wall);
            firsts.push(first);
        }
    }
    let wall = median(walls);
    let result = TranscriptionResult {
        clip: name,
        audio_seconds: round2(audio_seconds),
        wall_seconds: round2(wall),
        realtime_multiple: round2(audio_seconds / wall),
        first_text_ms: round2(median(firsts)),
    };
    say(
        log,
        format!(
            "{:.1} s of audio: {:.1}x real time, first text after {:.0} ms",
            result.audio_seconds, result.realtime_multiple, result.first_text_ms
        ),
    );
    Ok(vec![BenchResult::Transcription(result)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_length_comes_from_the_wav_header() {
        // 16 kHz mono 16-bit: 32000 bytes a second; two seconds of data,
        // after a LIST chunk the reader has to step over.
        let mut wav = b"RIFF\0\0\0\0WAVE".to_vec();
        wav.extend_from_slice(b"fmt \x10\0\0\0");
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&32_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"LIST\x03\0\0\0abc\0");
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&64_000u32.to_le_bytes());
        wav.extend(std::iter::repeat_n(0u8, 64_000));
        let s = wav_seconds(&wav).expect("readable");
        assert!((s - 2.0).abs() < 1e-9, "{s}");
        assert_eq!(wav_seconds(b"ID3 not a wav"), None);
    }
}

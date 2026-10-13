//! Embedding latency: one input and batches of inputs through an embedding
//! checkpoint, per modality, as the server's worker runs them.
//!
//! Text is synthetic ids drawn from the vocabulary with a fixed seed: the
//! pass costs the same whatever the ids say, and a length is then exactly
//! the length asked for. Media are the reference fixtures under
//! `test-vectors/references/embedding-gemma2/`, decoded once, before timing.
//!
//! Each case discards one warm-up and reports the median of the runs with
//! their spread; preprocessing (resize, mel framing) is outside the timed
//! span, as it is outside the worker in the server.

#![expect(clippy::print_stdout, reason = "the timings are the tool's output")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use shiso_core::{Backend, TokenId};
use shiso_metal::MetalBackend;
use shiso_models::embedder::{EmbedPart, EmbedderDesc, open_embedder};
use shiso_vision::{DecodedFrame, ImageLimits, preprocess_image, preprocess_video};

/// CLI arguments.
#[derive(Debug, Parser)]
#[command(
    name = "embedding",
    about = "Times an embedding checkpoint per modality"
)]
struct Args {
    /// The text GGUF.
    #[arg(long)]
    model: PathBuf,
    /// The projector GGUF, for the media cases.
    #[arg(long)]
    mmproj: Option<PathBuf>,
    /// Timed runs per case, after one warm-up.
    #[arg(long, default_value_t = 7)]
    runs: usize,
    /// Writes the first input's vector of the last run here, one value a
    /// line, for comparing two builds' answers.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Only these cases, comma-separated.
    #[arg(long)]
    only: Option<String>,
    /// Writes one GPU-time-per-kernel report per case into this directory,
    /// from one extra pass after the timed ones.
    #[arg(long)]
    profile: Option<PathBuf>,
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/embedding-gemma2")
}

/// `len` body ids from a fixed-seed generator, inside ordinary vocabulary.
fn text(len: usize, seed: u64) -> Vec<EmbedPart> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let ids = (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let id = 1000 + u32::try_from(s % 200_000).unwrap_or(0);
            TokenId::new(id)
        })
        .collect();
    vec![EmbedPart::Text(ids)]
}

/// Median and spread (max/min - 1) of `runs` timed calls, in ms.
fn time(runs: usize, mut f: impl FnMut() -> Result<()>) -> Result<(f64, f64)> {
    f()?;
    let mut ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let began = Instant::now();
        f()?;
        ms.push(began.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    let median = ms[ms.len() / 2];
    let spread = ms[ms.len() - 1] / ms[0] - 1.0;
    Ok((median, spread))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let backend: Arc<dyn Backend> = MetalBackend::new().context("the Metal device")?;
    let mut model = open_embedder(
        &EmbedderDesc {
            model: &args.model,
            mmproj: args.mmproj.as_deref(),
            sidecar: None,
        },
        &backend,
    )?;
    let frontend = model.frontend();

    let mut cases: Vec<(String, Vec<Vec<EmbedPart>>)> = Vec::new();
    for len in [8usize, 32, 128, 512, 2048, 8000] {
        cases.push((format!("text-{len}-n1"), vec![text(len, 1)]));
    }
    for len in [8usize, 32, 128, 512] {
        for n in [4usize, 16] {
            let batch = (0..n).map(|i| text(len, 7 + i as u64)).collect();
            cases.push((format!("text-{len}-n{n}"), batch));
        }
    }
    if let Some(spec) = frontend.image.as_ref() {
        let png = std::fs::read(fixtures().join("image-synthetic.png"))?;
        let item = preprocess_image(&png, spec, ImageLimits::default())?;
        let one = vec![EmbedPart::Image(item)];
        cases.push(("image-640x480-n1".into(), vec![one.clone()]));
        cases.push(("image-640x480-n4".into(), vec![one; 4]));
    }
    if let Some(spec) = frontend.video.as_ref() {
        let raw = std::fs::read(fixtures().join("video-synthetic-frames.u8"))?;
        let (w, h) = (320usize, 240usize);
        let frames = raw
            .chunks_exact(w * h * 3)
            .take(4)
            .map(|f| {
                let frame = DecodedFrame::from_rgb8(320, 240, f.to_vec())?;
                Ok(preprocess_video(
                    std::slice::from_ref(&frame),
                    &spec.frames,
                    ImageLimits::default(),
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        cases.push((
            "video-4frames-n1".into(),
            vec![vec![EmbedPart::Video(frames)]],
        ));
    }
    if let Some(tower) = frontend.audio.as_ref() {
        let wav = std::fs::read(fixtures().join("audio-speech-10s.wav"))?;
        let pcm = shiso_audio::decode_audio(
            &wav,
            &shiso_audio::DecodeLimits::new(std::num::NonZeroU32::new(60).context("nonzero")?),
            None,
        )?;
        let clip = shiso_audio::clip_for(tower.as_ref(), &pcm)?;
        let one = vec![EmbedPart::Audio(clip)];
        cases.push(("audio-10s-n1".into(), vec![one.clone()]));
        cases.push(("audio-10s-n4".into(), vec![one; 4]));
    }

    println!(
        "{:<20} {:>8} {:>10} {:>10} {:>8} {:>10}",
        "case", "tokens", "median ms", "ms/input", "spread", "tok/s"
    );
    for (name, inputs) in &cases {
        if args
            .only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|w| w == name))
        {
            continue;
        }
        let mut tokens = 0u64;
        let (median, spread) = time(args.runs, || {
            let out = model.embed_batch(inputs)?;
            if let Some(path) = args.dump.as_ref() {
                let v: Vec<String> = out[0].vector.iter().map(ToString::to_string).collect();
                std::fs::write(path, v.join("\n"))?;
            }
            tokens = out.iter().map(|e| u64::from(e.usage.total())).sum();
            Ok(())
        })?;
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let (per, tps) = (median / inputs.len() as f64, tokens as f64 / (median / 1e3));
        println!(
            "{name:<20} {tokens:>8} {median:>10.2} {per:>10.2} {:>7.1}% {tps:>10.0}",
            spread * 100.0
        );
        if let Some(dir) = args.profile.as_deref() {
            profile_case(dir, backend.as_ref(), name, inputs.len(), || {
                model.embed_batch(inputs)?;
                Ok(())
            })?;
        }
    }
    Ok(())
}

/// One more pass of a case inside a profile region, written as
/// `<dir>/<name>.json` and printed as a table.
fn profile_case(
    dir: &Path,
    backend: &dyn Backend,
    name: &str,
    inputs: usize,
    pass: impl FnOnce() -> Result<()>,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let ((), report) = shiso_bench::profile::scoped(backend, pass)?;
    shiso_bench::profile::write(
        &dir.join(format!("{name}.json")),
        &serde_json::json!({ "case": name, "inputs": inputs }),
        &report,
    )?;
    println!("{}", report.table(15));
    Ok(())
}

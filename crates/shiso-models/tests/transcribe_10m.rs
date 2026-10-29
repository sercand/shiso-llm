//! Ten minutes of a meeting, end to end on the Metal device: the speech
//! encoder, the decoder's prefill over its rows and speculative greedy decode
//! to the end of the turn, each timed. A measurement with one assertion:
//! the run finishes inside the two-minute bar.
//!
//! Needs `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE`,
//! `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER`, the reference captures and
//! the clip:
//! `cargo test --release -p shiso-models --test transcribe_10m -- --ignored --nocapture`.
#![cfg(target_os = "macos")]
#![expect(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "a rate in a report needs no exact mantissa, and a device run reads best as one sequence"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_audio::{AudioClip, AudioTower, MelFrontend, Pcm};
use shiso_core::media::{MediaModel, PadTokens, VisionPrefill};
use shiso_core::{Backend, Model, TokenId};
use shiso_engine::EngineOptions;
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_models::nemotron_h::{GgufResidency, NemotronHConfig, NemotronHModel};
use shiso_models::nemotron_transcribe::NemotronTranscribeTower;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

fn words(path: &Path) -> Vec<u32> {
    std::fs::read(path)
        .expect("reads")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn wav(path: &Path) -> Pcm {
    let bytes = std::fs::read(path).expect("the clip reads");
    let mut at = 12usize;
    loop {
        let len = u32::from_le_bytes([bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]])
            as usize;
        if &bytes[at..at + 4] == b"data" {
            let samples = bytes[at + 8..at + 8 + len]
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect();
            return Pcm::new(samples, 16_000).expect("16 kHz");
        }
        at += 8 + len + (len & 1);
    }
}

#[test]
#[ignore = "requires a Metal device, both speech exports, the reference captures and the clip"]
#[expect(clippy::print_stderr, reason = "the timings are the test's report")]
fn ten_minutes_transcribe_inside_two_minutes() {
    let (Some(decoder), Some(encoder)) = (
        env_path("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE"),
        env_path("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER"),
    ) else {
        eprintln!("SKIP: the two speech exports are not named");
        return;
    };
    let clip = env_path("SHISO_TEST_CLIP_10M").unwrap_or_else(|| {
        env_path("HOME")
            .expect("HOME")
            .join("models/clips/meeting_10m.wav")
    });
    let refs = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/nemotron_transcribe/meeting_10m/verbatim");
    if !refs.is_dir() || !clip.is_file() {
        eprintln!(
            "SKIP: no capture at {} or no clip at {}",
            refs.display(),
            clip.display()
        );
        return;
    }
    let k: u32 = std::env::var("SHISO_TEST_SPEC_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let slots: u32 = std::env::var("SHISO_TEST_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1600);
    let rows: u32 = std::env::var("SHISO_TEST_PREFILL_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");

    let t0 = Instant::now();
    let tower = NemotronTranscribeTower::open(&backend, &encoder).expect("the encoder loads");
    let config = NemotronHConfig::from_gguf_path(&decoder).expect("the decoder reads");
    let mut opts = EngineOptions::default();
    opts.ctx = 16_384;
    opts.prefill_rows = rows;
    opts.spec = SpecDepth::new(k);
    let mut model = NemotronHModel::load_gguf(
        &backend,
        &decoder,
        &config,
        opts,
        GgufResidency::Streamed { slots },
        true,
    )
    .expect("the decoder loads");
    let load = t0.elapsed().as_secs_f64();
    if let Some(info) = backend.memory_info() {
        eprintln!(
            "device memory after load: {:.1} GiB free of {:.1} GiB",
            info.free as f64 / f64::from(1u32 << 30),
            info.total as f64 / f64::from(1u32 << 30)
        );
    }

    // The request: decode, frame, encode, prefill, decode.
    let started = Instant::now();
    let pcm = wav(&clip);
    let mel = MelFrontend::new(tower.mel_spec().clone())
        .expect("spec")
        .compute(pcm.samples());
    let tokens = tower.tokens_for(mel.valid()).expect("a clip of audio");
    let clips = [AudioClip::new(mel, tokens).expect("clip")];
    let audio_rows = tower.encode(&backend, &clips).expect("encodes");
    backend.sync().expect("sync");
    let encode = started.elapsed().as_secs_f64();

    let pad = TokenId::new(0);
    let mut prompt: Vec<TokenId> = words(&refs.join("prompt_prefix.i32"))
        .into_iter()
        .map(TokenId::new)
        .collect();
    prompt.extend(std::iter::repeat_n(pad, tokens));
    prompt.extend(
        words(&refs.join("prompt_suffix.i32"))
            .into_iter()
            .map(TokenId::new),
    );
    let media = VisionPrefill::new(
        audio_rows.rows.view(),
        &audio_rows.grids,
        PadTokens::new(None, None).with_audio(Some(pad)),
        2688,
    )
    .expect("the rows match the run");
    let session = model.open_session().expect("session");
    let region = std::env::var_os("SHISO_TEST_PROFILE_PREFILL").map(|_| {
        shiso_core::profile::ProfileRegion::open(
            backend
                .as_profiler()
                .expect("the Metal device times launches"),
        )
        .expect("a profile region opens")
    });
    let t1 = Instant::now();
    let mut last = {
        let logits = model
            .prefill_media(session, &prompt, &media)
            .expect("prefill");
        let view = logits.row(0).expect("row");
        let mut raw = vec![0u8; usize::try_from(view.len()).expect("fits")];
        backend.copy_d2h(view, &mut raw).expect("d2h");
        let row: &[f32] = bytemuck::cast_slice(&raw);
        u32::try_from(
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .expect("non-empty"),
        )
        .expect("fits")
    };
    let prefill = t1.elapsed().as_secs_f64();
    if let Some(region) = region {
        eprintln!("{}", region.finish().expect("the region closes").table(16));
        let s = model.expert_stats().unwrap_or_default();
        eprintln!(
            "prefill reads {} MiB in {:.1} s, drain {:.1} s, {} fetches, {} read ahead, {} used, {:.1} s wall",
            s.read_bytes >> 20,
            s.read_seconds,
            s.drain_seconds,
            s.fetches,
            s.prefetch_reads,
            s.prefetched,
            prefill
        );
        return;
    }
    let before = model.expert_stats().unwrap_or_default();

    let t2 = Instant::now();
    let mut out = vec![last];
    let (mut accepted, mut drafted) = (0u32, 0u32);
    while last != 11 && out.len() < 6000 {
        let step = model
            .spec_mut()
            .expect("a bound head")
            .speculate(session, TokenId::new(last))
            .expect("speculate");
        accepted += step.accepted;
        drafted += step.drafted;
        for t in step.produced {
            last = t.get();
            out.push(last);
            if last == 11 {
                break;
            }
        }
    }
    let decode = t2.elapsed().as_secs_f64();
    let total = started.elapsed().as_secs_f64();
    let after = model.expert_stats().unwrap_or_default();
    let reference: Vec<u32> = std::fs::read_to_string(refs.join("picked.txt"))
        .expect("picks")
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let same_prefix = out
        .iter()
        .zip(&reference)
        .take_while(|(a, b)| a == b)
        .count();
    if let Some(path) = env_path("SHISO_TEST_DUMP_IDS") {
        let text = out
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(path, text).expect("the ids write");
    }
    eprintln!(
        "load {load:.1} s | request {total:.1} s = encode {encode:.1} s + prefill {prefill:.1} s ({} rows) \\
         + decode {decode:.1} s ({} tokens, {:.1} tok/s, accepted {accepted} of {drafted}) | \\
         decode hit {:.3}, read {:.1} s ({} MiB), drain {:.1} s | reference {} tokens, first {same_prefix} identical",
        prompt.len(),
        out.len(),
        out.len() as f64 / decode,
        (after.hits - before.hits) as f64 / (after.wanted - before.wanted).max(1) as f64,
        after.read_seconds - before.read_seconds,
        (after.read_bytes - before.read_bytes) >> 20,
        after.drain_seconds - before.drain_seconds,
        reference.len(),
    );
    assert!(total <= 120.0, "ten minutes took {total:.1} s");
}

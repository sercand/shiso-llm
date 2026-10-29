//! The speech decoder's GGUF export on the Metal device, its routed experts
//! streamed from disk.
//!
//! Needs `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE` naming the decoder `.gguf`:
//! `cargo test --release -p shiso-models --test nemotron_h_gguf_metal -- --ignored`.
#![cfg(target_os = "macos")]
#![expect(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "a rate in a report needs no exact mantissa, and a device run reads best as one sequence"
)]

use std::path::PathBuf;
use std::sync::Arc;

use shiso_core::{Backend, Model};
use shiso_engine::EngineOptions;
use shiso_modelfile::GgufFile;
use shiso_models::nemotron_h::{GgufResidency, NemotronHConfig, NemotronHModel};

/// Slots the tests stream through: a little over half of the 23 * 128
/// routed experts, the share this decoder gets on a 24 GB machine.
const SLOTS: u32 = 1600;

/// The decoder export, or `None` (said on stderr) when this box has none.
#[expect(clippy::print_stderr, reason = "a skipped device test says why")]
fn decoder() -> Option<PathBuf> {
    let Some(path) = std::env::var_os("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE").map(PathBuf::from)
    else {
        eprintln!("SKIP: SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE is unset");
        return None;
    };
    if !shiso_metal::metal_available() {
        eprintln!("SKIP: no Metal device on this machine");
        return None;
    }
    Some(path)
}

fn metal() -> Arc<dyn Backend> {
    shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles")
}

fn load(backend: &Arc<dyn Backend>, path: &std::path::Path) -> NemotronHModel {
    load_with(backend, path, |_| {})
}

fn load_with(
    backend: &Arc<dyn Backend>,
    path: &std::path::Path,
    tune: impl FnOnce(&mut EngineOptions),
) -> NemotronHModel {
    let file = GgufFile::open(path).expect("the export parses");
    let config = NemotronHConfig::from_gguf(&file).expect("the export's geometry reads");
    drop(file);
    let mut opts = EngineOptions::default();
    opts.ctx = 8192;
    tune(&mut opts);
    NemotronHModel::load_gguf(
        backend,
        path,
        &config,
        opts,
        GgufResidency::Streamed { slots: SLOTS },
        std::env::var("SHISO_TEST_PREFETCH").map_or(true, |v| v != "0"),
    )
    .expect("the export loads with its experts streamed")
}

#[test]
#[ignore = "requires a Metal device and the speech decoder export"]
fn loads_and_reports_info() {
    let Some(path) = decoder() else { return };
    let model = load(&metal(), &path);
    let info = model.info();
    assert_eq!(info.vocab, 131_072);
    assert_eq!(info.hidden, 2688);
    // Each routed layer joins the mixing layer before it: 52 - 23 slots.
    assert_eq!(info.n_layers, 29);
    assert!(model.expert_stats().is_some(), "the banks are streamed");
}

/// The reference's capture of one prompt: its token ids around the audio, the
/// fused audio rows it spliced in, its first logits row and its greedy picks.
struct Capture {
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    rows: Vec<f32>,
    logits0: Vec<f32>,
    picked: Vec<u32>,
}

fn capture(dir: &std::path::Path) -> Option<Capture> {
    let words = |name: &str| -> Option<Vec<u32>> {
        let raw = std::fs::read(dir.join(name)).ok()?;
        Some(
            raw.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    };
    let floats = |name: &str| -> Option<Vec<f32>> {
        let raw = std::fs::read(dir.join(name)).ok()?;
        Some(
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    };
    let picked = std::fs::read_to_string(dir.join("picked.txt"))
        .ok()?
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    Some(Capture {
        prefix: words("prompt_prefix.i32")?,
        suffix: words("prompt_suffix.i32")?,
        rows: floats("fused.f32")?,
        logits0: floats("logits0.f32")?,
        picked,
    })
}

#[test]
#[ignore = "requires a Metal device, the speech decoder export and the reference captures"]
#[expect(
    clippy::print_stderr,
    reason = "the measured agreement is the test's report"
)]
fn reference_audio_prompt_matches_logits_and_greedy() {
    use shiso_core::media::{MediaModel, PadTokens, VisionGrid, VisionPrefill};
    use shiso_core::{DecodeStep, TokenId, alloc};

    let Some(path) = decoder() else { return };
    let refs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/nemotron_transcribe/short_en/verbatim");
    let Some(cap) = capture(&refs) else {
        eprintln!("SKIP: no reference capture at {}", refs.display());
        return;
    };
    let backend = metal();
    let mut model = load(&backend, &path);
    let n_rows = u32::try_from(cap.rows.len() / 2688).expect("fits");
    // Placeholder id 0: the reference never writes it in a prompt, and every
    // row it marks is overwritten by an audio row before anything reads it.
    let pad = TokenId::new(0);
    let mut tokens: Vec<TokenId> = cap.prefix.iter().map(|&t| TokenId::new(t)).collect();
    tokens.extend(std::iter::repeat_n(pad, n_rows as usize));
    tokens.extend(cap.suffix.iter().map(|&t| TokenId::new(t)));
    let rows = alloc(&backend, (cap.rows.len() * 4) as u64).expect("alloc");
    backend
        .copy_h2d(rows.view(), bytemuck::cast_slice(&cap.rows))
        .expect("h2d");
    let grids = [VisionGrid::new(1, 1, n_rows)];
    let media = VisionPrefill::new(
        rows.view(),
        &grids,
        PadTokens::new(None, None).with_audio(Some(pad)),
        2688,
    )
    .expect("the rows match the run");
    let session = model.open_session().expect("session");
    let read = |backend: &Arc<dyn Backend>, view: shiso_core::DeviceView<'_>| -> Vec<f32> {
        let mut raw = vec![0u8; usize::try_from(view.len()).expect("fits")];
        backend.copy_d2h(view, &mut raw).expect("d2h");
        bytemuck::cast_slice(&raw).to_vec()
    };
    let first = {
        let logits = model
            .prefill_media(session, &tokens, &media)
            .expect("prefill");
        read(&backend, logits.row(0).expect("row"))
    };
    let (mut dot, mut na, mut nb, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (a, b) in first.iter().zip(&cap.logits0) {
        dot += f64::from(*a) * f64::from(*b);
        na += f64::from(*a) * f64::from(*a);
        nb += f64::from(*b) * f64::from(*b);
        worst = worst.max((a - b).abs());
    }
    let cosine = dot / (na.sqrt() * nb.sqrt());
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| u32::try_from(i).expect("fits"))
            .expect("non-empty")
    };
    let log_softmax = |v: &[f32]| -> Vec<f64> {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f64 = v.iter().map(|x| f64::from(x - m).exp()).sum();
        v.iter().map(|x| f64::from(x - m) - z.ln()).collect()
    };
    let (ours_lp, ref_lp) = (log_softmax(&first), log_softmax(&cap.logits0));
    let kl: f64 = ref_lp
        .iter()
        .zip(&ours_lp)
        .map(|(r, o)| r.exp() * (r - o))
        .sum();
    let mut order: Vec<usize> = (0..ref_lp.len()).collect();
    order.sort_by(|a, b| ref_lp[*b].total_cmp(&ref_lp[*a]));
    let top_nats = order[..64]
        .iter()
        .map(|&i| (ref_lp[i] - ours_lp[i]).abs())
        .fold(0.0f64, f64::max);
    eprintln!(
        "first row: cosine {cosine:.6}, max |diff| {worst:.4}, KL(ref||ours) {kl:.6} nats, top-64 worst {top_nats:.4} nats"
    );
    // Teacher-forced: every step reads the reference's previous pick, so one
    // near-tie that tips the other way (the reference quantizes activations
    // to 8 bits before every quantized matmul; this port does not) is one
    // step's disagreement, not a shifted tail.
    let mut worst_gap = 0.0f64;
    let mut agreed = 0usize;
    let mut row = first.clone();
    for (step, &want) in cap.picked.iter().enumerate() {
        let lp = log_softmax(&row);
        let pick = argmax(&row);
        let gap = lp[pick as usize] - lp[want as usize];
        worst_gap = worst_gap.max(gap);
        if pick == want {
            agreed += 1;
        } else {
            eprintln!(
                "step {step}: ours {pick} ({:.3} nats), reference {want} ({:.3} nats)",
                lp[pick as usize], lp[want as usize]
            );
        }
        if want == 11 {
            break;
        }
        let logits = model
            .decode_batch(&[DecodeStep {
                session,
                token: TokenId::new(want),
            }])
            .expect("decode");
        row = read(&backend, logits.row(0).expect("row"));
    }
    eprintln!(
        "greedy: {agreed}/{} agree, worst gap {worst_gap:.3} nats",
        cap.picked.len()
    );
    if let Some(stats) = model.expert_stats() {
        eprintln!(
            "experts: wanted {} hits {} reads {} ({} MiB in {:.2} s)",
            stats.wanted,
            stats.hits,
            stats.reads,
            stats.read_bytes >> 20,
            stats.read_seconds
        );
    }
    assert_eq!(argmax(&first), cap.picked[0], "the first pick");
    assert!(
        worst_gap <= 0.5,
        "a reference pick sat {worst_gap} nats under this port's choice"
    );
}

/// Prefills the reference prompt and decodes `steps` greedy tokens (an
/// end-of-turn is fed back like any other), timing the decode. A measurement,
/// not an assertion: its numbers go in the run's report.
#[test]
#[ignore = "requires a Metal device, the speech decoder export and the reference captures"]
#[expect(clippy::print_stderr, reason = "the measurement is the test's report")]
fn decode_rate() {
    use shiso_core::media::{MediaModel, PadTokens, VisionGrid, VisionPrefill};
    use shiso_core::{DecodeStep, TokenId, alloc};

    let Some(path) = decoder() else { return };
    let refs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/nemotron_transcribe/meeting_60s/verbatim");
    let Some(cap) = capture(&refs) else {
        eprintln!("SKIP: no reference capture at {}", refs.display());
        return;
    };
    let steps: usize = std::env::var("SHISO_TEST_DECODE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let backend = metal();
    let mut model = load(&backend, &path);
    let n_rows = u32::try_from(cap.rows.len() / 2688).expect("fits");
    let pad = TokenId::new(0);
    let mut tokens: Vec<TokenId> = cap.prefix.iter().map(|&t| TokenId::new(t)).collect();
    tokens.extend(std::iter::repeat_n(pad, n_rows as usize));
    tokens.extend(cap.suffix.iter().map(|&t| TokenId::new(t)));
    let rows = alloc(&backend, (cap.rows.len() * 4) as u64).expect("alloc");
    backend
        .copy_h2d(rows.view(), bytemuck::cast_slice(&cap.rows))
        .expect("h2d");
    let grids = [VisionGrid::new(1, 1, n_rows)];
    let media = VisionPrefill::new(
        rows.view(),
        &grids,
        PadTokens::new(None, None).with_audio(Some(pad)),
        2688,
    )
    .expect("the rows match the run");
    let session = model.open_session().expect("session");
    let started = std::time::Instant::now();
    let mut last = {
        let logits = model
            .prefill_media(session, &tokens, &media)
            .expect("prefill");
        argmax_row(&backend, logits.row(0).expect("row"))
    };
    let prefill = started.elapsed().as_secs_f64();
    let before = model.expert_stats().unwrap_or_default();
    eprintln!(
        "prefill streamed {} MiB in {:.2} s of reads, {:.2} s draining, {} fetches",
        before.read_bytes >> 20,
        before.read_seconds,
        before.drain_seconds,
        before.fetches
    );
    let region = std::env::var_os("SHISO_TEST_PROFILE").map(|_| {
        shiso_core::profile::ProfileRegion::open(
            backend
                .as_profiler()
                .expect("the Metal device times launches"),
        )
        .expect("a profile region opens")
    });
    let started = std::time::Instant::now();
    for _ in 0..steps {
        let logits = model
            .decode_batch(&[DecodeStep {
                session,
                token: TokenId::new(last),
            }])
            .expect("decode");
        last = argmax_row(&backend, logits.row(0).expect("row"));
    }
    let decode = started.elapsed().as_secs_f64();
    if let Some(region) = region {
        eprintln!("{}", region.finish().expect("the region closes").table(25));
    }
    let after = model.expert_stats().unwrap_or_default();
    let wanted = after.wanted - before.wanted;
    let hits = after.hits - before.hits;
    eprintln!(
        "prefill {} tokens in {prefill:.2} s ({:.0} tok/s); decode {steps} in {decode:.2} s ({:.2} tok/s); \
         hit rate {:.3}; {:.0} MiB read per token; read {:.2} s, drain {:.2} s; \
         {} guessed, {} of them used",
        tokens.len(),
        tokens.len() as f64 / prefill,
        steps as f64 / decode,
        hits as f64 / wanted.max(1) as f64,
        (after.read_bytes - before.read_bytes) as f64 / steps as f64 / 1_048_576.0,
        after.read_seconds - before.read_seconds,
        after.drain_seconds - before.drain_seconds,
        after.prefetch_reads - before.prefetch_reads,
        after.prefetched - before.prefetched,
    );
}

fn argmax_row(backend: &Arc<dyn Backend>, view: shiso_core::DeviceView<'_>) -> u32 {
    let mut raw = vec![0u8; usize::try_from(view.len()).expect("fits")];
    backend.copy_d2h(view, &mut raw).expect("d2h");
    let row: &[f32] = bytemuck::cast_slice(&raw);
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| u32::try_from(i).expect("fits"))
        .expect("non-empty")
}

/// The first logits row of the 60-second reference prompt, prefilled at
/// `rows` rows a pass.
fn first_row_at(
    backend: &Arc<dyn Backend>,
    path: &std::path::Path,
    cap: &Capture,
    rows: u32,
) -> (Vec<f32>, f64) {
    use shiso_core::media::{MediaModel, PadTokens, VisionGrid, VisionPrefill};
    use shiso_core::{TokenId, alloc};

    let mut model = load_with(backend, path, |opts| opts.prefill_rows = rows);
    let n_rows = u32::try_from(cap.rows.len() / 2688).expect("fits");
    let pad = TokenId::new(0);
    let mut tokens: Vec<TokenId> = cap.prefix.iter().map(|&t| TokenId::new(t)).collect();
    tokens.extend(std::iter::repeat_n(pad, n_rows as usize));
    tokens.extend(cap.suffix.iter().map(|&t| TokenId::new(t)));
    let block = alloc(backend, (cap.rows.len() * 4) as u64).expect("alloc");
    backend
        .copy_h2d(block.view(), bytemuck::cast_slice(&cap.rows))
        .expect("h2d");
    let grids = [VisionGrid::new(1, 1, n_rows)];
    let media = VisionPrefill::new(
        block.view(),
        &grids,
        PadTokens::new(None, None).with_audio(Some(pad)),
        2688,
    )
    .expect("the rows match the run");
    let session = model.open_session().expect("session");
    let profile = std::env::var_os("SHISO_TEST_PROFILE").is_some();
    let region = profile.then(|| {
        shiso_core::profile::ProfileRegion::open(
            backend
                .as_profiler()
                .expect("the Metal device times launches"),
        )
        .expect("a profile region opens")
    });
    let started = std::time::Instant::now();
    let logits = model
        .prefill_media(session, &tokens, &media)
        .expect("prefill");
    let view = logits.row(0).expect("row");
    if let Some(region) = region {
        let report = region.finish().expect("the region closes");
        #[expect(clippy::print_stderr, reason = "the profile is the run's report")]
        {
            eprintln!("{}", report.table(20));
        }
    }
    let mut raw = vec![0u8; usize::try_from(view.len()).expect("fits")];
    backend.copy_d2h(view, &mut raw).expect("d2h");
    (
        bytemuck::cast_slice(&raw).to_vec(),
        started.elapsed().as_secs_f64(),
    )
}

#[test]
#[ignore = "requires a Metal device, the speech decoder export and the reference captures"]
#[expect(clippy::print_stderr, reason = "the timings are the test's report")]
fn a_wider_prefill_pass_answers_the_same() {
    let Some(path) = decoder() else { return };
    let refs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/nemotron_transcribe/meeting_60s/verbatim");
    let Some(cap) = capture(&refs) else {
        eprintln!("SKIP: no reference capture at {}", refs.display());
        return;
    };
    let backend = metal();
    let (narrow, narrow_s) = first_row_at(&backend, &path, &cap, 512);
    let (wide, wide_s) = first_row_at(&backend, &path, &cap, 4096);
    eprintln!("prefill at 512 rows {narrow_s:.2} s, at 4096 rows {wide_s:.2} s");
    assert!(
        narrow
            .iter()
            .zip(&wide)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the answer depends on the pass width"
    );
}

/// Prefills `cap`'s prompt on `model` and returns the first pick.
fn prefill_capture(
    backend: &Arc<dyn Backend>,
    model: &mut NemotronHModel,
    cap: &Capture,
) -> (shiso_core::SessionId, u32) {
    use shiso_core::media::{MediaModel, PadTokens, VisionGrid, VisionPrefill};
    use shiso_core::{TokenId, alloc};

    let n_rows = u32::try_from(cap.rows.len() / 2688).expect("fits");
    let pad = TokenId::new(0);
    let mut tokens: Vec<TokenId> = cap.prefix.iter().map(|&t| TokenId::new(t)).collect();
    tokens.extend(std::iter::repeat_n(pad, n_rows as usize));
    tokens.extend(cap.suffix.iter().map(|&t| TokenId::new(t)));
    let rows = alloc(backend, (cap.rows.len() * 4) as u64).expect("alloc");
    backend
        .copy_h2d(rows.view(), bytemuck::cast_slice(&cap.rows))
        .expect("h2d");
    let grids = [VisionGrid::new(1, 1, n_rows)];
    let media = VisionPrefill::new(
        rows.view(),
        &grids,
        PadTokens::new(None, None).with_audio(Some(pad)),
        2688,
    )
    .expect("the rows match the run");
    let session = model.open_session().expect("session");
    let first = {
        let logits = model
            .prefill_media(session, &tokens, &media)
            .expect("prefill");
        argmax_row(backend, logits.row(0).expect("row"))
    };
    (session, first)
}

/// Greedy decode of `cap`'s prompt to the end of the turn (at most `limit`
/// tokens), plainly or speculating `k` deep: the tokens, the seconds, and the
/// accepted and drafted counts.
fn transcribe_ids(
    path: &std::path::Path,
    cap: &Capture,
    k: u32,
    limit: usize,
) -> (Vec<u32>, f64, u32, u32) {
    use shiso_core::{DecodeStep, TokenId};

    let backend = metal();
    let mut model = load_with(&backend, path, |opts| {
        opts.spec = shiso_engine::sparse::mtp::SpecDepth::new(k);
    });
    let (session, mut last) = prefill_capture(&backend, &mut model, cap);
    let mut out = vec![last];
    let (mut accepted, mut drafted) = (0u32, 0u32);
    let region = (k > 0 && std::env::var_os("SHISO_TEST_PROFILE").is_some()).then(|| {
        shiso_core::profile::ProfileRegion::open(
            backend
                .as_profiler()
                .expect("the Metal device times launches"),
        )
        .expect("a profile region opens")
    });
    let before = model.expert_stats().unwrap_or_default();
    let started = std::time::Instant::now();
    while last != 11 && out.len() < limit {
        if k == 0 {
            let logits = model
                .decode_batch(&[DecodeStep {
                    session,
                    token: TokenId::new(last),
                }])
                .expect("decode");
            last = argmax_row(&backend, logits.row(0).expect("row"));
            out.push(last);
        } else {
            let step = model
                .spec_mut()
                .expect("a bound head")
                .speculate(session, TokenId::new(last))
                .expect("speculate");
            accepted += step.accepted;
            drafted += step.drafted;
            for t in step.produced {
                out.push(t.get());
                last = t.get();
                if last == 11 {
                    break;
                }
            }
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    if let Some(region) = region {
        #[expect(clippy::print_stderr, reason = "the profile is the run's report")]
        {
            eprintln!("{}", region.finish().expect("the region closes").table(25));
        }
    }
    let after = model.expert_stats().unwrap_or_default();
    #[expect(
        clippy::print_stderr,
        reason = "the stream's numbers are the run's report"
    )]
    {
        eprintln!(
            "k={k}: {} fetches, hit {:.3}, read {:.2} s ({} MiB), drain {:.2} s, {} guessed, {} used",
            after.fetches - before.fetches,
            (after.hits - before.hits) as f64 / (after.wanted - before.wanted).max(1) as f64,
            after.read_seconds - before.read_seconds,
            (after.read_bytes - before.read_bytes) >> 20,
            after.drain_seconds - before.drain_seconds,
            after.prefetch_reads - before.prefetch_reads,
            after.prefetched - before.prefetched,
        );
    }
    (out, seconds, accepted, drafted)
}

#[test]
#[ignore = "requires a Metal device, the speech decoder export and the reference captures"]
#[expect(clippy::print_stderr, reason = "the rates are the test's report")]
fn speculative_greedy_equals_plain_greedy() {
    let Some(path) = decoder() else { return };
    let k: u32 = std::env::var("SHISO_TEST_SPEC_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    for clip in ["short_en", "meeting_60s"] {
        let refs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../test-vectors/references/nemotron_transcribe/{clip}/verbatim"
        ));
        let Some(cap) = capture(&refs) else {
            eprintln!("SKIP: no reference capture at {}", refs.display());
            return;
        };
        let (plain, plain_s, _, _) = transcribe_ids(&path, &cap, 0, 400);
        let (spec, spec_s, accepted, drafted) = transcribe_ids(&path, &cap, k, 400);
        eprintln!(
            "{clip}: {} tokens; plain {:.2} tok/s, speculative k={k} {:.2} tok/s; accepted {accepted} of {drafted}",
            plain.len(),
            plain.len() as f64 / plain_s,
            spec.len() as f64 / spec_s,
        );
        assert_eq!(spec, plain, "{clip}: speculation changed the greedy answer");
    }
}

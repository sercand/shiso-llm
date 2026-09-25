//! The MLX checkpoint of the sparse family, loaded and run on this machine's
//! Metal device.
//!
//! Run with the checkpoint directory named:
//!
//! ```text
//! SHISO_TEST_MODEL_QWEN4EXP_MLX=/path/to/Qwen3.8-Flash-Next-MLX-4bit \
//!   cargo test -p shiso-models --release --test qwen4exp_mlx_metal -- --ignored --nocapture
//! ```
//!
//! A skipped case says so on stdout; it passes as well, so read the output.
//!
//! The first case doubles as the stack's speed check on this machine, and a
//! few variables shape it: `SHISO_TEST_DECODE_TOKENS` (15) tokens decoded
//! after the prompt, `SHISO_TEST_PROMPT_REPEAT` repeats of a sentence in
//! front of it to time a long prefill, `SHISO_TEST_PREFILL_ROWS` (512) the
//! widest prefill pass, and `SHISO_TEST_RESERVE_GB` (6) the memory left
//! beside the expert pool - a wider margin keeps the machine out of swap,
//! which is what makes two runs comparable.
#![cfg(target_os = "macos")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "token ids and fixture sizes are far inside every width they are cast to, a \
              rate in a log line needs no exact mantissa, and a device run reads best as one \
              sequence"
)]

use std::path::PathBuf;
use std::sync::Arc;

use shiso_core::config::HfConfig;
use shiso_core::{Backend, Model, TokenId};
use shiso_engine::sparse::capture;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_models::qwen4exp::Qwen4ExpModel;

/// The checkpoint directory under test, or `None` when this box has none.
///
/// The MLX conversion under `SHISO_TEST_MODEL_QWEN4EXP_MLX`, or the reference
/// FP8 export under `SHISO_TEST_MODEL_QWEN4EXP_FP8` when that is set instead.
fn model_dir() -> Option<PathBuf> {
    let path = std::env::var("SHISO_TEST_MODEL_QWEN4EXP_MLX")
        .or_else(|_| std::env::var("SHISO_TEST_MODEL_QWEN4EXP_FP8"))
        .ok()
        .map(PathBuf::from)?;
    path.join("config.json").is_file().then_some(path)
}

/// Loads whichever container `dir` holds.
fn load(backend: &Arc<dyn Backend>, dir: &std::path::Path) -> Qwen4ExpModel {
    let hf = HfConfig::from_file(&dir.join("config.json")).expect("config.json parses");
    let opts = metal_options(backend);
    let reserve = std::env::var("SHISO_TEST_RESERVE_GB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(6)
        << 30;
    if shiso_models::qwen4exp::mlx::MlxCheckpoint::is_reference_export(dir) {
        Qwen4ExpModel::load_reference(backend, dir, &hf, opts, None, reserve)
    } else {
        Qwen4ExpModel::load_mlx(backend, dir, &hf, opts, None, reserve)
    }
    .expect("the checkpoint loads")
}

#[expect(
    clippy::print_stdout,
    reason = "the skip report and the transcript are a test's output"
)]
fn say(line: &str) {
    println!("{line}");
}

/// The options a Metal device runs this stack under.
fn metal_options(backend: &Arc<dyn Backend>) -> SparseOptions {
    let mut opts = SparseOptions::new();
    opts.tensor_core = false;
    opts.tiled_attention = false;
    opts.chunk.available = false;
    opts.entries = capture::capture(backend.as_ref());
    opts.prompt_matmul = capture::optional_paths(backend.as_ref()).prompt_matmul;
    opts.prefill_rows = std::env::var("SHISO_TEST_PREFILL_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    opts.engine.ctx = 4096;
    opts.engine.kv_pool_pages = 32;
    opts
}

/// One logits row, read back.
fn read_row(backend: &Arc<dyn Backend>, row: shiso_core::DeviceView<'_>) -> Vec<f32> {
    let mut bytes = vec![0u8; usize::try_from(row.len()).expect("a row fits usize")];
    backend
        .copy_d2h(row, &mut bytes)
        .expect("the row reads back");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// The index of the largest logit.
fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

#[test]
#[ignore = "needs the MLX checkpoint and a Metal device"]
fn the_checkpoint_answers_a_short_prompt_on_metal() {
    let Some(dir) = model_dir() else {
        say("SKIP: no checkpoint named by SHISO_TEST_MODEL_QWEN4EXP_MLX");
        return;
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("SHISO_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_test_writer()
        .try_init();
    let metal = shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let backend: Arc<dyn Backend> = metal;
    let started = std::time::Instant::now();
    let mut model = load(&backend, &dir);
    say(&format!(
        "loaded in {:.1}s",
        started.elapsed().as_secs_f64()
    ));

    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("the tokenizer loads");
    let long = std::env::var("SHISO_TEST_PROMPT_REPEAT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|n| {
            "The history of Paris spans more than two thousand years, from a Celtic settlement on the Seine to a Roman town, a medieval capital and a modern metropolis. "
                .repeat(n)
                + "The capital of France is"
        });
    let prompt = long.as_deref().unwrap_or("The capital of France is");
    let ids: Vec<TokenId> = tokenizer
        .encode(prompt, false)
        .expect("the prompt encodes")
        .get_ids()
        .iter()
        .map(|&id| TokenId::new(id))
        .collect();
    if ids.len() < 64 {
        say(&format!(
            "prompt ids {:?}",
            ids.iter().map(|t| t.get()).collect::<Vec<_>>()
        ));
    }

    let session = model.open_session().expect("a session opens");
    let started = std::time::Instant::now();
    let logits = model
        .prefill_last(session, &ids)
        .expect("the prompt prefills");
    let row = read_row(&backend, logits.row(0).expect("one logits row"));
    say(&format!(
        "prefill {} tokens {:.2}s ({:.1} tok/s)",
        ids.len(),
        started.elapsed().as_secs_f64(),
        ids.len() as f64 / started.elapsed().as_secs_f64()
    ));
    if let Some(stats) = model.stack().expert_stream_stats() {
        say(&format!("after prefill: {stats:?}"));
    }
    let finite = row.iter().all(|v| v.is_finite());
    let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (id, v) in top.iter().take(8) {
        let text = tokenizer.decode(&[*id as u32], false).unwrap_or_default();
        say(&format!("  top {id:>7} {v:>9.4} {text:?}"));
    }
    assert!(finite, "every logit is finite");

    let mut next = TokenId::new(argmax(&row) as u32);
    let mut out = vec![next.get()];
    let started = std::time::Instant::now();
    let decode_n: usize = std::env::var("SHISO_TEST_DECODE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    for _ in 0..decode_n {
        let step = [shiso_core::model::DecodeStep {
            session,
            token: next,
        }];
        let logits = model.decode_batch(&step).expect("a decode step runs");
        let row = read_row(&backend, logits.row(0).expect("one row"));
        next = TokenId::new(argmax(&row) as u32);
        out.push(next.get());
    }
    let secs = started.elapsed().as_secs_f64();
    say(&format!(
        "decoded {decode_n} in {secs:.2}s ({:.2} tok/s): {:?}",
        decode_n as f64 / secs,
        tokenizer.decode(&out, false).unwrap_or_default()
    ));
    if let Some(stats) = model.stack().expert_stream_stats() {
        say(&format!("expert stream: {stats:?}"));
    }
    // The same prompt and continuation again, teacher-forced: every expert
    // it routes to is already in a slot, so this is the decode rate the
    // device and the host overhead allow, with no disk in it.
    // Three times over: one replay on this device swings with its clocks.
    let mut session = session;
    for _ in 0..3 {
        model.close_session(session).expect("the session closes");
        session = model.open_session().expect("a second session opens");
        model
            .prefill_last(session, &ids)
            .expect("the prompt prefills again");
        let started = std::time::Instant::now();
        for &token in &out[..out.len() - 1] {
            let step = [shiso_core::model::DecodeStep {
                session,
                token: TokenId::new(token),
            }];
            model.decode_batch(&step).expect("a decode step runs");
        }
        let secs = started.elapsed().as_secs_f64();
        say(&format!(
            "warm replay: {:.2} tok/s",
            (out.len() - 1) as f64 / secs
        ));
    }
    if let Some(stats) = model.stack().expert_stream_stats() {
        say(&format!("expert stream after replay: {stats:?}"));
    }
}

/// The affine projection over one real stream-mix contraction, from the
/// checkpoint's own bytes, against a host restatement of the same bytes -
/// once from uploaded copies and once from the loader's in-place import.
#[test]
#[ignore = "needs the MLX checkpoint and a Metal device"]
fn the_affine_projection_reads_real_weights_as_the_host_does() {
    use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};
    use shiso_models::qwen4exp::mlx::MlxCheckpoint;
    let Some(dir) = model_dir() else {
        say("SKIP: no checkpoint named by SHISO_TEST_MODEL_QWEN4EXP_MLX");
        return;
    };
    let metal = shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let backend: Arc<dyn Backend> = metal;
    let ck = MlxCheckpoint::open(&dir).expect("the checkpoint opens");
    let base = "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down";
    let (out_dim, in_dim, rows) = (320usize, 10240usize, 5usize);
    let shapes = [
        ("weight", "U32", vec![out_dim as u64, in_dim as u64 / 8]),
        ("scales", "BF16", vec![out_dim as u64, in_dim as u64 / 64]),
        ("biases", "BF16", vec![out_dim as u64, in_dim as u64 / 64]),
    ];
    let host: Vec<&[u8]> = shapes
        .iter()
        .map(|(part, dtype, shape)| {
            ck.tensor_bytes(&format!("{base}.{part}"), dtype, shape)
                .expect("the array is there")
        })
        .collect();
    let bf = |b: &[u8], i: usize| {
        f32::from_bits(u32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])) << 16)
    };
    let x: Vec<f32> = (0..rows * in_dim)
        .map(|i| ((i * 7919 % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let mut want = vec![0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0f64;
            for k in 0..in_dim {
                let idx = o * in_dim + k;
                let word = u32::from_le_bytes(
                    host[0][idx / 8 * 4..idx / 8 * 4 + 4]
                        .try_into()
                        .expect("four bytes"),
                );
                let q = ((word >> (4 * (idx % 8))) & 0xF) as f32;
                let g = o * (in_dim / 64) + k / 64;
                acc +=
                    f64::from(q * bf(host[1], g) + bf(host[2], g)) * f64::from(x[r * in_dim + k]);
            }
            want[r * out_dim + o] = acc as f32;
        }
    }
    for (i, (part, dtype, shape)) in shapes.iter().enumerate() {
        let b = ck
            .import_tensor(&backend, &format!("{base}.{part}"), dtype, shape)
            .expect("import");
        let mut back = vec![0u8; b.len() as usize];
        backend.copy_d2h(b.view(), &mut back).expect("read back");
        let first = back.iter().zip(host[i]).position(|(a, b)| a != b);
        say(&format!(
            "{part}: {} bytes, first difference at {first:?}; host ptr {:p}",
            back.len(),
            host[i].as_ptr()
        ));
    }
    let xs: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xbuf = shiso_core::buffer::alloc(&backend, xs.len() as u64).expect("alloc");
    backend.copy_h2d(xbuf.view(), &xs).expect("upload");
    for (label, bufs) in [
        (
            "uploaded",
            host.iter()
                .map(|bytes| {
                    let b = shiso_core::buffer::alloc(&backend, bytes.len() as u64).expect("alloc");
                    backend.copy_h2d(b.view(), bytes).expect("upload");
                    b
                })
                .collect::<Vec<_>>(),
        ),
        (
            "imported",
            shapes
                .iter()
                .map(|(part, dtype, shape)| {
                    ck.import_tensor(&backend, &format!("{base}.{part}"), dtype, shape)
                        .expect("import")
                })
                .collect::<Vec<_>>(),
        ),
    ] {
        let out = shiso_core::buffer::alloc(&backend, (rows * out_dim * 4) as u64).expect("alloc");
        let args = [
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(bufs[0].view()),
            KernelArg::Buffer(bufs[1].view()),
            KernelArg::Buffer(bufs[2].view()),
            KernelArg::Buffer(xbuf.view()),
            KernelArg::U32(in_dim as u32),
            KernelArg::U32(out_dim as u32),
            KernelArg::U32(rows as u32),
        ];
        backend
            .run(
                KernelRef::new("q4e_matvec", "q4e_affine_mm_kernel"),
                &LaunchArgs::new(Extent3::new(40, 1, 1), Extent3::linear(256), &args),
            )
            .expect("launch");
        let got = read_row(&backend, out.view());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        say(&format!(
            "{label}: worst {worst} got[0..4] {:?} want[0..4] {:?}",
            &got[..4],
            &want[..4]
        ));
        assert!(worst < 1e-2, "{label}");
    }
}

/// One reference case: a prompt prefix and the reference's top logits at its
/// frontier.
struct RefCase {
    id: String,
    tokens: usize,
    prompt: PathBuf,
    digest: Option<String>,
    top: Vec<(usize, f32)>,
}

/// FNV-1a 64 over the ids' little-endian words, the reference's token digest.
fn tokens_digest(ids: &[u32]) -> String {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in ids.iter().flat_map(|id| id.to_le_bytes()) {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    format!("{h:016x}")
}

/// The reference file's text cases, with each case's own digest.
fn reference_cases(path: &std::path::Path) -> Vec<RefCase> {
    let text = std::fs::read_to_string(path).expect("the reference reads");
    let base = path.parent().expect("the reference has a directory");
    let mut digests = std::collections::HashMap::new();
    let mut cases: Vec<RefCase> = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            ["#", key, value] if key.starts_with("tokens_digest_") => {
                digests.insert(
                    key.trim_start_matches("tokens_digest_").to_owned(),
                    (*value).to_owned(),
                );
            }
            ["case", id, "text", _ctx, n, prompt, _k] => cases.push(RefCase {
                id: (*id).to_owned(),
                tokens: n.parse().expect("a token count"),
                prompt: base.join(prompt),
                digest: None,
                top: Vec::new(),
            }),
            ["top", _rank, id, value] => {
                if let Some(case) = cases.last_mut() {
                    case.top
                        .push((id.parse().expect("an id"), value.parse().expect("a logit")));
                }
            }
            _ => {}
        }
    }
    for case in &mut cases {
        case.digest = digests.get(&case.id).cloned();
    }
    cases
}

/// The reference's short cases, replayed on Metal.
///
/// The reference is the FP8 checkpoint through its own library (W8A16); this
/// is the 4-bit container through this stack's kernels, so the numbers are
/// not expected to agree to the digit. What is asserted is what the two
/// quantizations cannot explain away: the reference's top token is ours
/// wherever the reference is confident of it (a margin of a logit or more),
/// and our top token sits in the reference's top 16 everywhere.
#[test]
#[ignore = "needs the MLX checkpoint, the reference vectors and a Metal device"]
fn the_short_reference_cases_agree_with_the_fp8_reference() {
    let Some(dir) = model_dir() else {
        say("SKIP: no checkpoint named by SHISO_TEST_MODEL_QWEN4EXP_MLX");
        return;
    };
    let vectors = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/q4e-fp8-short.vec");
    if !vectors.is_file() {
        say("SKIP: test-vectors/references/q4e-fp8-short.vec is not on this machine");
        return;
    }
    let max_tokens: usize = std::env::var("SHISO_TEST_PARITY_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let metal = shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let backend: Arc<dyn Backend> = metal;
    let mut model = load(&backend, &dir);
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("the tokenizer loads");
    let mut failures = Vec::new();
    for case in reference_cases(&vectors) {
        if case.tokens > max_tokens {
            continue;
        }
        let text = std::fs::read_to_string(&case.prompt).expect("the prompt reads");
        let all: Vec<u32> = tokenizer
            .encode(text, false)
            .expect("the prompt encodes")
            .get_ids()
            .to_vec();
        let ids = &all[..case.tokens];
        let digest = tokens_digest(ids);
        assert_eq!(
            Some(&digest),
            case.digest.as_ref(),
            "{}: the tokenization differs",
            case.id
        );
        let session = model.open_session().expect("a session opens");
        let tokens: Vec<TokenId> = ids.iter().map(|&id| TokenId::new(id)).collect();
        let tap_dir = std::env::var("SHISO_TEST_TAP_DIR").ok();
        if tap_dir.is_some() {
            model.stack_mut().arm_residual_tap().expect("the tap arms");
        }
        let logits = model
            .prefill_last(session, &tokens)
            .expect("the prefix prefills");
        let row = read_row(&backend, logits.row(0).expect("one row"));
        if let Some(dir) = tap_dir {
            let tap = model
                .stack_mut()
                .take_residual_tap()
                .expect("the tap was armed");
            for record in tap.records() {
                let name = match record.point {
                    shiso_engine::sparse::tap::TapPoint::Embed => "embed".to_owned(),
                    shiso_engine::sparse::tap::TapPoint::Mixer(l) => format!("mixer{}", l.get()),
                    shiso_engine::sparse::tap::TapPoint::Layer(l) => format!("layer{}", l.get()),
                };
                let bytes: Vec<u8> = record.values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(
                    PathBuf::from(&dir).join(format!("{}.{name}.f32", case.id)),
                    bytes,
                )
                .expect("the tap writes");
            }
        }
        model.close_session(session).expect("the session closes");
        if let Ok(dir) = std::env::var("SHISO_TEST_DUMP_LOGITS") {
            let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(PathBuf::from(dir).join(format!("{}.f32", case.id)), bytes)
                .expect("the logits dump writes");
        }
        let ours = argmax(&row);
        let (ref_top, ref_v) = case.top[0];
        let margin = ref_v - case.top[1].1;
        let top16: Vec<usize> = case.top.iter().take(16).map(|(id, _)| *id).collect();
        let deltas: Vec<f32> = case
            .top
            .iter()
            .take(16)
            .map(|&(id, v)| row[id] - v)
            .collect();
        let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
        let spread = deltas.iter().map(|d| (d - mean).abs()).fold(0f32, f32::max);
        say(&format!(
            "{:<24} ours {:>7} {:?} ref {:>7} {:?} margin {margin:.2} | top16 logit shift {mean:+.3}, spread {spread:.3}",
            case.id,
            ours,
            tokenizer.decode(&[ours as u32], false).unwrap_or_default(),
            ref_top,
            tokenizer
                .decode(&[ref_top as u32], false)
                .unwrap_or_default(),
        ));
        if (margin >= 1.0 && ours != ref_top) || !top16.contains(&ours) {
            failures.push(case.id.clone());
        }
    }
    assert!(failures.is_empty(), "disagreeing cases: {failures:?}");
}

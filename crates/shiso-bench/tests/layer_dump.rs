//! The per-layer residual dump on a real sparse checkpoint.
//!
//! Two claims. The tap copies one row at every boundary of the frontier's pass
//! (the broadcast, and both halves of every layer) in pass order, and the
//! files [`LayerDump`] writes from it are the layout the reference writes. And
//! arming it changes nothing: the tapped pass runs without device graphs, and
//! the logits it produces must equal, to the bit, those of an untapped pass
//! whose single-row islands ran from a captured graph. A tap that moved the
//! answer would bisect a different computation from the one being debugged.
//!
//! Needs a CUDA device and the checkpoint, and holds the device for a model
//! load, so run it under the GPU lock:
//!
//! ```text
//! SHISO_CUDA_COMPILE=1 SHISO_TEST_MODEL_QWEN4EXP=/path/to/checkpoint.gguf \
//!   ~/work/qwen-3.8-flash/gpu_lock.sh \
//!   cargo test --release -p shiso-bench --test layer_dump -- --ignored --nocapture
//! ```
#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_bench::layers::{LayerDump, file_name, frontier_layers, pass_points};
use shiso_bench::rows::{IndexEntry, read_npy_header};
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::Model;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::EngineOptions;
use shiso_engine::sparse::capture;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_engine::sparse::tap::TapPoint;
use shiso_modelfile::GgufFile;
use shiso_models::qwen4exp::{Qwen4ExpConfig, Qwen4ExpModel};

/// Context the model is loaded with; the prompt is far inside it.
const CTX: u32 = 1024;

/// Tokens in the synthetic prompt: two prefill chunks' worth at a small chunk,
/// so the tapped pass is the last of several.
const PROMPT: u32 = 40;

/// Prefill chunk the prompt is fed in.
const CHUNK: usize = 16;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// A path named by `var`, when it names a file.
fn file_from(var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(var).ok()?);
    path.is_file().then_some(path)
}

/// `count` ids from a fixed pseudo-random walk over ordinary text ids, clear of
/// the special tokens at the top of the table.
fn prompt(count: u32) -> Vec<TokenId> {
    let mut state = 0x2545_f491_u32;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            TokenId::new(1_000 + (state >> 8) % 60_000)
        })
        .collect()
}

/// Loads the checkpoint with the configuration its own metadata declares.
fn load(backend: &Arc<dyn Backend>, path: &Path) -> Qwen4ExpModel {
    let file = GgufFile::open(path).expect("the checkpoint opens");
    let config = Qwen4ExpConfig::from_metadata(file.metadata()).expect("the metadata agrees");
    drop(file);
    let mut engine = EngineOptions::default();
    engine.ctx = CTX;
    let entries = capture::capture(backend.as_ref());
    let opts = SparseOptions {
        engine,
        co_resident_blocks: capture::co_resident_blocks(backend.as_ref(), entries),
        entries,
        ..SparseOptions::default()
    };
    Qwen4ExpModel::load_with(backend, path, None, None, &config, opts)
        .expect("the checkpoint loads")
}

/// The frontier row of `tokens` along the ordinary, untapped route.
fn plain_row(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    tokens: &[TokenId],
) -> Vec<f32> {
    let session = model.open_session().expect("a session opens");
    let (head, tail) = tokens.split_at(tokens.len() - 1);
    for piece in head.chunks(CHUNK) {
        model.prefill_chunk(session, piece).expect("a chunk");
    }
    let logits = model.prefill_last(session, tail).expect("the last token");
    let view = logits.row(0).expect("the row exists");
    let mut bytes = vec![0u8; usize::try_from(view.len()).expect("a row fits a host index")];
    backend
        .copy_d2h(view, &mut bytes)
        .expect("the row reads back");
    model.close_session(session).expect("the session closes");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// Whether two rows are the same bits.
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
#[ignore = "needs a CUDA device and a checkpoint of the sparse family"]
fn the_tap_copies_every_boundary_and_leaves_the_answer_unchanged() {
    let Some(path) = file_from("SHISO_TEST_MODEL_QWEN4EXP") else {
        skip("no checkpoint named by SHISO_TEST_MODEL_QWEN4EXP");
        return;
    };
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let mut model = load(&backend, &path);
    let n_layers = model.stack().shape().n_layers();
    let streams = model.stack().shape().streams as usize;
    let hidden = model.stack().shape().hidden as usize;
    let tokens = prompt(PROMPT);

    // Twice untapped first. Each run's single-row islands go through a device
    // graph (a new session is a new capture slot, so each captures and then
    // launches what it captured), which is the route the tapped run is held
    // to, and the pair shows that route is deterministic to begin with.
    let first = plain_row(&backend, &mut model, &tokens);
    let replayed = plain_row(&backend, &mut model, &tokens);
    assert!(same_bits(&first, &replayed), "two untapped runs disagree");

    let (row, tap) =
        frontier_layers(backend.as_ref(), &mut model, &tokens, CHUNK).expect("a tapped run");
    assert!(
        same_bits(&row, &replayed),
        "arming the tap moved the frontier's logits"
    );
    assert!(
        model.stack_mut().take_residual_tap().is_none(),
        "the tap is disarmed after the run"
    );
    let after = plain_row(&backend, &mut model, &tokens);
    assert!(
        same_bits(&after, &replayed),
        "a run after the tap disagrees"
    );

    let points: Vec<TapPoint> = tap.records().iter().map(|r| r.point).collect();
    assert_eq!(points, pass_points(n_layers));
    for record in tap.records() {
        assert_eq!(record.values.len(), streams * hidden);
        assert!(
            record.values.iter().all(|v| v.is_finite()),
            "{:?} holds a non-finite value",
            record.point
        );
    }
    // The broadcast tiles one embedding across the streams, and every island
    // adds something to them.
    let embed = &tap.records()[0].values;
    for stream in embed.chunks(hidden).skip(1) {
        assert!(
            same_bits(stream, &embed[..hidden]),
            "the broadcast is not a tiling"
        );
    }
    for pair in tap.records().windows(2) {
        assert!(
            !same_bits(&pair[0].values, &pair[1].values),
            "{:?} left the residual as {:?} found it",
            pair[1].point,
            pair[0].point
        );
    }

    let dir = std::env::temp_dir().join(format!("shiso-layer-dump-{}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    let mut dump = LayerDump::create(&dir).expect("a fresh directory");
    let entry = IndexEntry {
        prompt_file: "synthetic".to_owned(),
        frontier: PROMPT - 1,
        prompt_tokens: PROMPT,
    };
    dump.write("synthetic", &tap, n_layers, entry)
        .expect("the case writes");
    dump.finish().expect("the index writes");
    for point in pass_points(n_layers) {
        let bytes = std::fs::read(dir.join("synthetic").join(file_name(point, n_layers)))
            .expect("every boundary has a file");
        let header = read_npy_header(&bytes).expect("an npy header");
        assert_eq!(header.descr, "<f4");
        assert_eq!(header.shape, vec![streams, hidden]);
    }
    std::fs::remove_dir_all(&dir).expect("cleanup");
    model.teardown().expect("the model tears down");
}

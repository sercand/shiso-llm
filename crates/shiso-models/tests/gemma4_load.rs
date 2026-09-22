//! The E4B container, loaded onto a real device and asked for one logits row.
//!
//! This is the first case in the workspace that runs a Gemma 4 checkpoint at
//! all, so what it claims is deliberately narrow: the container opened, every
//! role bound, every payload reached the device, the per-layer table was opened
//! over the tensor that holds it, and one prefill produced a row of the right
//! width whose argmax names a token this vocabulary has. What the model *says*
//! is `gemma4_decode.rs`'s question.
//!
//! The case needs the container and a Metal device, so it is `#[ignore]`d and
//! skips with a printed note when either is missing. The path is local and
//! overridable by `SHISO_TEST_MODEL_GEMMA4`, the same variable the binding test
//! reads.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::Model;
use shiso_core::{DType, HfConfig};
use shiso_engine::EngineOptions;
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::gemma4::{Gemma4Config, Gemma4Model, PleOptions};

/// Layer slots the checkpoint carries.
const N_LAYERS: u32 = 42;
/// Residual-stream width.
const HIDDEN: u32 = 2560;
/// Vocabulary size, which is also the width of one logits row.
const VOCAB: u32 = 262_144;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Reports what a run produced, for the eye reading the log.
#[expect(clippy::print_stdout, reason = "the timing is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

/// The local container, overridable, or `None` when it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4").ok()?);
    path.is_file().then_some(path)
}

/// The E4B geometry, read from the shipped document.
///
/// The document carries `text_config`, so the path handed in is never opened -
/// which makes this profile the *document's* statement of the geometry rather
/// than the container's, and loading the container against it is therefore a
/// check that the two agree.
fn profile() -> Gemma4Config {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/gemma4-config.json");
    let document =
        HfConfig::from_file(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    Gemma4Config::from_hf(&document, Path::new("/nonexistent")).expect("the e4b geometry")
}

/// A context small enough that the cache is not what this case measures.
fn options() -> EngineOptions {
    let mut opts = EngineOptions::default();
    opts.ctx = 1024;
    opts.kv_pool_pages = 64;
    opts
}

#[test]
#[ignore = "needs the local checkpoint and a Metal device"]
fn the_container_loads_and_answers_one_logits_row() {
    let Some(path) = checkpoint() else {
        skip("no local Gemma 4 container");
        return;
    };
    if !metal_available() {
        skip("no Metal device");
        return;
    }
    let config = profile();
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("a Metal device");

    let started = std::time::Instant::now();
    let mut model = Gemma4Model::load(&backend, &path, &config, options(), PleOptions::default())
        .expect("the container loads");
    report(&format!("load: {:.3}s", started.elapsed().as_secs_f64()));

    let info = *model.info();
    assert_eq!(info.n_layers, N_LAYERS);
    assert_eq!(info.hidden, HIDDEN);
    assert_eq!(info.vocab, VOCAB);
    assert!(
        info.context_capacity > 0,
        "a loaded model has room for at least one token"
    );
    // The bulk of this container is `Q4_0`, and the loaded model still says so:
    // this is the assertion that nothing quietly widened a weight on its way to
    // the device.
    assert_eq!(model.ffn_gate_dtype(), DType::Q4_0);

    let session = model.open_session().expect("a session opens");
    let prompt = [TokenId::new(2), TokenId::new(106), TokenId::new(1645)];
    let logits = model
        .prefill_last(session, &prompt)
        .expect("the prompt runs");
    assert_eq!(logits.rows(), 1);
    assert_eq!(logits.vocab(), VOCAB);

    let row = logits.row(0).expect("the prompt's row");
    let mut host = vec![0u8; usize::try_from(VOCAB).expect("a vocabulary fits a host index") * 4];
    backend
        .copy_d2h(row, &mut host)
        .expect("the prompt's row reads back");
    let scores: Vec<f32> = host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(
        scores.len(),
        usize::try_from(VOCAB).expect("a vocabulary fits")
    );
    assert!(
        scores.iter().all(|s| s.is_finite()),
        "a logits row with a non-finite entry means a kernel read a buffer nobody filled"
    );
    let best = shiso_engine::argmax(&scores);
    assert!(
        best.get() < VOCAB,
        "the argmax names a token this vocabulary has: {}",
        best.get()
    );
    report(&format!("argmax of the first row: {}", best.get()));

    model.close_session(session).expect("the session closes");
    model.teardown().expect("the model tears down");
}

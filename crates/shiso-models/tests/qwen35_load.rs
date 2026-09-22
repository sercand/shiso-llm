//! The two containers a checkpoint of this family arrives in, each loaded onto a
//! real device.
//!
//! [`Qwen35Model::load`] is the only place the engine touches a file, and it is
//! the place where the two containers stop being interchangeable: a single-file
//! checkpoint's weights are already in a format the kernels read, so the device
//! wraps its mapping in place, while a directory checkpoint's weights are packed
//! and have to be decoded into buffers the loader owns. Both must end in the same
//! state - a bound model whose every tensor is resident - and that is what these
//! cases assert.
//!
//! # Why this stops at the load
//!
//! Nothing here decodes a token. A load that succeeded is the whole claim: the
//! header parsed, every role bound, every dimension agreed with the profile, and
//! every payload reached the device. What a kernel then computes is `decode.rs`'s
//! question, and it is asked against one container because the weights that
//! arrive are the same weights either way.
//!
//! Every case needs a checkpoint and a Metal device, so every case is
//! `#[ignore]`d and skips with a printed note when either is missing. The paths
//! are local and overridable by `SHISO_TEST_MODEL` and `SHISO_TEST_MODEL_OPTIQ`,
//! the same two variables the rest of this crate's model tests read.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::model::Model;
use shiso_core::{DType, HfConfig};
use shiso_engine::EngineOptions;
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model};

/// Layer slots the checkpoint carries.
const N_LAYERS: u32 = 32;
/// Residual-stream width.
const HIDDEN: u32 = 2560;
/// Vocabulary size.
const VOCAB: u32 = 248_320;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// The single-file checkpoint, when this machine has one.
fn gguf_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL").ok()?);
    path.is_file().then_some(path)
}

/// The directory checkpoint, when this machine has one.
fn directory_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_OPTIQ").ok()?);
    path.is_dir().then_some(path)
}

/// The shape profile both containers are loaded against.
///
/// One profile for both on purpose: the two files describe the same model, so a
/// profile that only one of them binds against would mean the containers
/// disagree about the architecture rather than about its spelling.
fn profile() -> Qwen35Config {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-config.json");
    Qwen35Config::from_hf(&HfConfig::from_file(&path).expect("the recorded config parses"))
        .expect("the recorded config is this family's")
}

/// A context small enough that the cache is not what these cases measure.
fn options() -> EngineOptions {
    let mut opts = EngineOptions::default();
    opts.ctx = 1024;
    opts.kv_pool_pages = 32;
    opts
}

/// Asserts the loaded model describes the checkpoint both containers hold.
fn check_geometry(model: &Qwen35Model) {
    let info = model.info();
    assert_eq!(info.n_layers, N_LAYERS);
    assert_eq!(info.hidden, HIDDEN);
    assert_eq!(info.vocab, VOCAB);
    assert!(
        info.context_capacity > 0,
        "a loaded model has room for at least one token"
    );
}

#[test]
#[ignore = "needs the local checkpoint directory and a Metal device"]
fn a_directory_checkpoint_loads_and_puts_its_weights_on_the_device() {
    let Some(path) = directory_path() else {
        skip("no local checkpoint directory");
        return;
    };
    if !metal_available() {
        skip("no Metal device");
        return;
    }
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("a Metal device");
    let mut model = Qwen35Model::load(&backend, &path, &profile(), options())
        .expect("the directory checkpoint loads");

    check_geometry(&model);
    // The packed weights decode to half precision, which is the one format the
    // directory loader produces and therefore the one thing the loaded model can
    // still say about where it came from.
    assert_eq!(model.ffn_gate_dtype(), DType::F16);

    let session = model.open_session().expect("a session opens");
    model.close_session(session).expect("the session closes");
    model.teardown().expect("the model tears down");
}

#[test]
#[ignore = "needs the local checkpoint and a Metal device"]
fn a_single_file_checkpoint_still_loads_and_puts_its_weights_on_the_device() {
    let Some(path) = gguf_path() else {
        skip("no local checkpoint");
        return;
    };
    if !metal_available() {
        skip("no Metal device");
        return;
    }
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("a Metal device");
    let mut model = Qwen35Model::load(&backend, &path, &profile(), options())
        .expect("the single-file checkpoint loads");

    check_geometry(&model);
    // Still the format the file stores, not a widened copy of it: this is the
    // assertion that the zero-copy path did not quietly become a decode.
    assert_eq!(model.ffn_gate_dtype(), DType::Q4_K);

    let session = model.open_session().expect("a session opens");
    model.close_session(session).expect("the session closes");
    model.teardown().expect("the model tears down");
}

//! The speech decoder's GGUF export on the Metal device, its routed experts
//! streamed from disk.
//!
//! Needs `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE` naming the decoder `.gguf`:
//! `cargo test --release -p shiso-models --test nemotron_h_gguf_metal -- --ignored`.
#![cfg(target_os = "macos")]

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

fn load(path: &std::path::Path) -> NemotronHModel {
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let file = GgufFile::open(path).expect("the export parses");
    let config = NemotronHConfig::from_gguf(&file).expect("the export's geometry reads");
    drop(file);
    let mut opts = EngineOptions::default();
    opts.ctx = 8192;
    NemotronHModel::load_gguf(
        &backend,
        path,
        &config,
        opts,
        GgufResidency::Streamed { slots: SLOTS },
    )
    .expect("the export loads with its experts streamed")
}

#[test]
#[ignore = "requires a Metal device and the speech decoder export"]
fn loads_and_reports_info() {
    let Some(path) = decoder() else { return };
    let model = load(&path);
    let info = model.info();
    assert_eq!(info.vocab, 131_072);
    assert_eq!(info.hidden, 2688);
    // Each routed layer joins the mixing layer before it: 52 - 23 slots.
    assert_eq!(info.n_layers, 29);
    assert!(model.expert_stats().is_some(), "the banks are streamed");
}

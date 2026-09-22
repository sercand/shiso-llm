//! Tests loading Gemma 4 from an MLX safetensors directory.
#![expect(
    clippy::print_stdout,
    clippy::unreadable_literal,
    reason = "test timings and constants"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::backend::Backend;
use shiso_core::config::HfConfig;
use shiso_core::ids::TokenId;
use shiso_core::model::Model;
use shiso_engine::EngineOptions;
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::gemma4::{Gemma4Config, Gemma4Model, PleOptions};

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

fn config() -> Gemma4Config {
    let path = vectors().join("loader/gemma4-config.json");
    let document = HfConfig::from_file(&path).expect("loader/gemma4-config.json");
    Gemma4Config::from_hf(&document, Path::new("/nonexistent")).expect("config")
}

fn mlx_dir() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4_MLX").ok()?);
    path.is_dir().then_some(path)
}

#[test]
#[ignore = "needs local Gemma 4 MLX checkpoint and a Metal device"]
fn gemma4_mlx_loads_and_prefills() {
    let Some(dir) = mlx_dir() else {
        println!("SKIP: no local Gemma 4 MLX directory");
        return;
    };
    if !metal_available() {
        println!("SKIP: no Metal device");
        return;
    }

    let cfg = config();
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("Metal device");

    let start1 = Instant::now();
    let mut model = Gemma4Model::load(
        &backend,
        &dir,
        &cfg,
        EngineOptions::default(),
        PleOptions::default(),
    )
    .expect("load model from MLX directory");
    let dur1 = start1.elapsed();
    println!("first load time: {dur1:?}");

    assert_eq!(model.info().n_layers, 42);
    assert_eq!(model.info().hidden, 2560);
    assert_eq!(model.info().vocab, 262144);

    // Verify sidecar exists
    let sidecar = dir.join("shiso-ple-affine4.bin");
    assert!(sidecar.is_file(), "sidecar binary must exist");

    // Prefill 3 tokens: <bos> (2), <turn|> (106), 1645
    let row_bytes = usize::try_from(model.info().logits_row_bytes()).expect("fits usize");
    let session = model.open_session().expect("session opens");
    let tokens = [TokenId::new(2), TokenId::new(106), TokenId::new(1645)];
    let logits = model
        .prefill_last(session, &tokens)
        .expect("prefill succeeds");
    assert_eq!(logits.rows(), 1);
    assert_eq!(logits.vocab(), 262144);

    let mut row = vec![0u8; row_bytes];
    backend
        .copy_d2h(logits.row(0).expect("first row"), &mut row)
        .expect("readback");
    let f32_row = bytemuck::cast_slice::<u8, f32>(&row);
    assert_eq!(f32_row.len(), 262144);
    assert!(f32_row.iter().all(|x| x.is_finite()));

    model.close_session(session).expect("session closes");
    model.teardown().expect("teardown");
    drop(model);

    // Second load is faster because sidecar already exists
    let start2 = Instant::now();
    let model2 = Gemma4Model::load(
        &backend,
        &dir,
        &cfg,
        EngineOptions::default(),
        PleOptions::default(),
    )
    .expect("second load succeeds");
    let dur2 = start2.elapsed();
    println!("second load time: {dur2:?}");
    drop(model2);
}

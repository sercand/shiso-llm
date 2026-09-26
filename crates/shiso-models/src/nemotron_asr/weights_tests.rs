//! Every export loads onto the device, two exports of the same weights agree,
//! and a store that is missing a tensor is refused by name.
//!
//! Needs `SHISO_TEST_MODEL_ASR` and a GPU device (Metal or CUDA).

use std::path::PathBuf;
use std::time::Instant;

use shiso_core::DeviceBuffer;

use super::AsrWeights;
use crate::audio_ops::Ops;
use crate::nemotron_asr::config::{AsrConfig, Flavour, Located, locate};

#[expect(
    clippy::print_stdout,
    reason = "a skipped case says why, and the load report is the output"
)]
fn say(line: &str) {
    println!("{line}");
}

fn exports() -> Option<Vec<(Flavour, PathBuf)>> {
    let root = PathBuf::from(std::env::var_os("SHISO_TEST_MODEL_ASR")?);
    Some(vec![
        (Flavour::Hf, root.join("nvidia")),
        (Flavour::Mlx, root.join("mlx")),
        (
            Flavour::GgufAsr,
            root.join("nvidia/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"),
        ),
        (
            Flavour::GgufParakeet,
            root.join("gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"),
        ),
    ])
}

fn ops() -> Ops {
    Ops::new(crate::audio_ops::testing::device().expect("a device for the device case"))
}

fn read(ops: &Ops, buf: &DeviceBuffer, from: usize, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * 4];
    let view = buf
        .slice(
            u64::try_from(from * 4).expect("fits"),
            u64::try_from(len * 4).expect("fits"),
        )
        .expect("in range");
    ops.backend()
        .copy_d2h(view, &mut bytes)
        .expect("reads back");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load(ops: &Ops, path: &std::path::Path) -> (Located, AsrConfig, AsrWeights) {
    let located = locate(path).expect("readable").expect("this family");
    let store = located.store().expect("opens");
    let config = AsrConfig::read(&located, &store).expect("reads");
    let start = Instant::now();
    let weights = AsrWeights::load(ops, &located, &config, &store).expect("loads");
    say(&format!(
        "{:?}: loaded in {:.2}s",
        located.flavour,
        start.elapsed().as_secs_f64()
    ));
    (located, config, weights)
}

/// Worst difference relative to the reference row's largest magnitude.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

#[test]
#[ignore = "requires a GPU device"]
fn every_export_loads_and_two_exports_agree() {
    let Some(exports) = exports() else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let ops = ops();
    let mut hf: Option<AsrWeights> = None;
    for (flavour, path) in exports {
        if !path.exists() {
            say(&format!("SKIP {flavour:?}: absent"));
            continue;
        }
        let (_, config, weights) = load(&ops, &path);
        assert_eq!(weights.layers.len(), config.layers);
        if let Some(reference) = &hf {
            for row in [0usize, 5000] {
                let (a, b) = (
                    read(&ops, &weights.embed, row * 640, 640),
                    read(&ops, &reference.embed, row * 640, 640),
                );
                let d = rel(&a, &b);
                say(&format!("{flavour:?}: embed row {row} vs hf {d:.2e}"));
                assert!(d <= 2e-2, "{flavour:?}: embed row {row} differs by {d}");
            }
            let (a, b) = (
                read(&ops, &weights.layers[0].rel, 0, 83 * 1024),
                read(&ops, &reference.layers[0].rel, 0, 83 * 1024),
            );
            let d = rel(&a, &b);
            say(&format!("{flavour:?}: layer 0 rel table vs hf {d:.2e}"));
            assert!(d <= 5e-2, "{flavour:?}: rel table differs by {d}");
        } else {
            assert_eq!(flavour, Flavour::Hf, "the reference export loads first");
            let blank = read(&ops, &weights.embed, 13_087 * 640, 640);
            assert!(
                blank.iter().all(|v| *v == 0.0),
                "the blank embedding row is zero"
            );
            hf = Some(weights);
        }
    }
}

#[test]
#[ignore = "requires a GPU device"]
fn a_store_missing_a_tensor_is_refused_by_name() {
    let Some(exports) = exports() else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let (_, hf_path) = &exports[0];
    let ops = ops();
    let located = locate(hf_path).expect("readable").expect("this family");
    let store = located.store().expect("opens");
    let config = AsrConfig::read(&located, &store).expect("reads");
    // The same file read through another export's name table: every role is
    // "missing", and the first one asked for must be named.
    let wrong = located.with_flavour(Flavour::GgufParakeet);
    let err = AsrWeights::load(&ops, &wrong, &config, &store).expect_err("refused");
    let text = err.to_string();
    assert!(text.contains("missing tensor enc."), "{text}");
}

#[test]
#[ignore = "requires a GPU device"]
fn a_store_missing_one_mid_stack_tensor_is_refused_by_name_and_flavour() {
    let Some(exports) = exports() else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    // One block's closing norm gone: the stated depth still says 24 blocks, so
    // the load must fail on that tensor rather than build a shorter model.
    let hidden = "encoder.layers.3.norm_out.weight";
    let (_, mlx) = &exports[1];
    if !mlx.exists() {
        return say(&format!("SKIP: {} is absent", mlx.display()));
    }
    let ops = ops();
    let located = locate(mlx)
        .expect("the path is readable")
        .expect("this family");
    let store = located.store().expect("opens").hiding(hidden);
    let err = AsrConfig::read(&located, &store)
        .and_then(|config| AsrWeights::load(&ops, &located, &config, &store).map(|_| ()))
        .expect_err("refused");
    let text = err.to_string();
    assert!(text.contains(hidden), "{text}");
    assert!(text.contains("Mlx"), "{text}");
}

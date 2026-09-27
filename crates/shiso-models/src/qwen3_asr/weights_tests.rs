//! Every export loads onto the device with the geometry its config states,
//! the computed position table is the one the GGUF bakes in, and a store
//! missing a tensor is refused by name.
//!
//! Needs `SHISO_TEST_MODEL_QWEN3ASR`, `SHISO_TEST_MODEL_R2T2` and a GPU device
//! (Metal or CUDA). Run in release: the HF exports widen a few hundred
//! megabytes of small parameters on the host.

use std::time::Instant;

use super::{QwenAsrWeights, sinusoids};
use crate::audio_ops::Ops;
use crate::audio_ops::load::download_f32;
use crate::audio_ops::store::dequant;
use crate::qwen3_asr::config::tests::{exports, say};
use crate::qwen3_asr::config::{Flavour, Qwen3AsrConfig, TOKENS_PER_CHUNK, locate};

fn ops() -> Option<Ops> {
    crate::audio_ops::testing::device().map(Ops::new)
}

#[test]
#[ignore = "requires a GPU device and the real checkpoints"]
fn every_export_loads() {
    let Some(ops) = ops() else {
        return;
    };
    let mut loaded = 0;
    for export in exports() {
        if !export.path.exists() {
            say(&format!(
                "SKIP {}: {} is absent",
                export.name,
                export.path.display()
            ));
            continue;
        }
        let located = locate(&export.path, None)
            .expect("readable")
            .expect("this family");
        let stores = located.stores().expect("opens");
        let config = Qwen3AsrConfig::read(&located, &stores).expect("reads");
        let start = Instant::now();
        let weights = QwenAsrWeights::load(&ops, &located, &config, &stores).expect("loads");
        let secs = start.elapsed().as_secs_f64();
        let (tower, text) = (&weights.tower, &weights.text);
        assert_eq!(tower.layers.len(), 24, "{}", export.name);
        assert_eq!(text.layers.len(), 28, "{}", export.name);
        assert_eq!(
            (tower.conv_out.in_dim(), tower.conv_out.out_dim()),
            (7680, 1024),
            "{}",
            export.name
        );
        assert!(tower.conv_out.bias.is_none(), "{}", export.name);
        assert_eq!(
            (tower.proj2.in_dim(), tower.proj2.out_dim()),
            (1024, 2048),
            "{}",
            export.name
        );
        let l0 = &tower.layers[0];
        assert_eq!((l0.fc1.in_dim(), l0.fc1.out_dim()), (1024, 4096));
        assert!(
            l0.q.bias.is_some() && l0.o.bias.is_some(),
            "{}",
            export.name
        );
        let d0 = &text.layers[0];
        assert_eq!((d0.q.in_dim(), d0.q.out_dim()), (2048, 2048));
        assert_eq!((d0.k.in_dim(), d0.k.out_dim()), (2048, 1024));
        assert_eq!((d0.down.in_dim(), d0.down.out_dim()), (6144, 2048));
        assert!(d0.q.bias.is_none(), "{}", export.name);
        assert_eq!(
            (text.embed.in_dim(), text.embed.out_dim()),
            (2048, 151_936),
            "{}",
            export.name
        );
        let positions = download_f32(&ops, tower.positions.view(), TOKENS_PER_CHUNK * 1024)
            .expect("reads back");
        assert_eq!(
            positions,
            sinusoids(TOKENS_PER_CHUNK, 1024),
            "{}",
            export.name
        );
        let q_norm = download_f32(&ops, d0.q_norm.view(), 128).expect("reads back");
        assert!(q_norm.iter().all(|v| v.is_finite()), "{}", export.name);
        say(&format!(
            "{}: {:?} loaded in {secs:.2}s",
            export.name, located.flavour
        ));
        loaded += 1;
    }
    say(&format!("loaded {loaded} exports"));
}

#[test]
#[ignore = "reads the real mmproj named by SHISO_TEST_MODEL_QWEN3ASR"]
fn the_computed_position_table_is_the_one_the_gguf_bakes_in() {
    let Some(export) = exports()
        .into_iter()
        .find(|e| e.flavour == Flavour::Gguf && e.path.exists())
    else {
        return say("SKIP: no GGUF export is on this machine");
    };
    let located = locate(&export.path, None)
        .expect("readable")
        .expect("this family");
    let stores = located.stores().expect("opens");
    let baked = dequant(
        &stores
            .tower()
            .require("a.position_embd.weight")
            .expect("baked"),
    )
    .expect("f32");
    let ours = sinusoids(TOKENS_PER_CHUNK, 1024);
    let worst = ours
        .iter()
        .zip(&baked)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    say(&format!("position table vs the baked one: {worst:.2e}"));
    assert!(worst <= 1e-6, "the tables differ by {worst}");
}

#[test]
#[ignore = "requires a GPU device and the real checkpoints"]
fn a_store_missing_a_tensor_is_refused_by_name_and_flavour() {
    let Some(ops) = ops() else {
        return;
    };
    let Some(export) = exports()
        .into_iter()
        .find(|e| e.flavour == Flavour::Mlx && e.path.exists())
    else {
        return say("SKIP: the MLX export is not on this machine");
    };
    // One mid-stack tensor gone: the stated depth still says 28 layers, so the
    // load must fail on that tensor rather than build a shorter model.
    let hidden = "model.layers.17.self_attn.k_norm.weight";
    let located = locate(&export.path, None)
        .expect("readable")
        .expect("this family");
    let stores = located.stores().expect("opens").hiding(hidden);
    let config = Qwen3AsrConfig::read(&located, &stores).expect("reads");
    let err = QwenAsrWeights::load(&ops, &located, &config, &stores).expect_err("refused");
    let text = err.to_string();
    assert!(text.contains(hidden) && text.contains("Mlx"), "{text}");
}

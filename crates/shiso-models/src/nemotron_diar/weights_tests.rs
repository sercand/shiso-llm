//! Every export loads, the upsampling conv reads the same from both layouts,
//! and a defect in an export is refused by name.

use std::sync::Arc;

use shiso_audio::AudioError;
use shiso_core::Backend;

use super::{DiarWeights, upsample_matrix};
use crate::audio_ops::Ops;
use crate::audio_ops::load::linear_rows;
use crate::nemotron_diar::config::{DiarConfig, locate};

#[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
fn say(line: &str) {
    eprintln!("{line}");
}

fn backend() -> Option<Arc<dyn Backend>> {
    crate::audio_ops::testing::device()
}

#[test]
fn both_upsample_layouts_give_the_same_matrix() {
    // out 2, in 2, taps 3: value = 100·o + 10·i + t.
    let (out, inp, taps) = (2, 2, 3);
    let mut in_taps = Vec::new();
    for o in 0..out {
        for i in 0..inp {
            for t in 0..taps {
                in_taps.push(f32::from(
                    u8::try_from(100 * o + 10 * i + t).expect("small"),
                ));
            }
        }
    }
    let mut taps_in = Vec::new();
    for o in 0..out {
        for t in 0..taps {
            for i in 0..inp {
                taps_in.push(f32::from(
                    u8::try_from(100 * o + 10 * i + t).expect("small"),
                ));
            }
        }
    }
    let a = upsample_matrix(&in_taps, out, inp, taps, false);
    let b = upsample_matrix(&taps_in, out, inp, taps, true);
    assert_eq!(a, b);
    // Row 1, tap 2, input 1: 100 + 10 + 2.
    assert!((a[(3 + 2) * inp + 1] - 112.0).abs() < f32::EPSILON);
}

#[test]
fn every_export_loads() {
    let Some(exports) = crate::nemotron_diar::config::tests::exports() else {
        return say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
    };
    let Some(backend) = backend() else { return };
    let ops = Ops::new(backend);
    for (_, path) in exports {
        if !path.exists() {
            say(&format!("SKIP: {} is absent", path.display()));
            continue;
        }
        let located = locate(&path).expect("reads").expect("this family");
        let store = located.store().expect("opens");
        let config = DiarConfig::read(&located, &store).expect("reads");
        let weights = DiarWeights::load(&ops, &located, &config, &store)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(weights.layers.len(), 31);
        assert_eq!(weights.silence.len(), 512);
    }
}

#[test]
fn a_missing_layer_tensor_is_refused_by_name() {
    let Some(exports) = crate::nemotron_diar::config::tests::exports() else {
        return say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
    };
    let Some(backend) = backend() else { return };
    let ops = Ops::new(backend);
    let (_, path) = &exports[1];
    if !path.exists() {
        return say(&format!("SKIP: {} is absent", path.display()));
    }
    let located = locate(path).expect("reads").expect("this family");
    let store = located.store().expect("opens");
    let config = DiarConfig::read(&located, &store).expect("reads");
    let hidden = "encoder.layers.30.norm1.weight";
    let store = store.hiding(hidden);
    let error = DiarWeights::load(&ops, &located, &config, &store).expect_err("a tensor is gone");
    assert!(
        matches!(&error, AudioError::Checkpoint { what } if what.contains(hidden) && what.contains("Mlx")),
        "{error}"
    );
}

#[test]
fn a_fused_matrix_of_the_wrong_height_is_refused() {
    let Some(exports) = crate::nemotron_diar::config::tests::exports() else {
        return say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
    };
    let Some(backend) = backend() else { return };
    let ops = Ops::new(backend);
    let (_, path) = &exports[1];
    if !path.exists() {
        return say(&format!("SKIP: {} is absent", path.display()));
    }
    let store = locate(path)
        .expect("reads")
        .expect("this family")
        .store()
        .expect("opens");
    let name = "encoder.layers.0.attn.w_qkv.weight";
    let error =
        linear_rows(&ops, &store, name, 0..512, 1535, 512).expect_err("1536 rows, not 1535");
    assert!(error.to_string().contains(name), "{error}");
    let error = linear_rows(&ops, &store, name, 1024..1600, 1536, 512).expect_err("past the end");
    assert!(error.to_string().contains("not inside"), "{error}");
}

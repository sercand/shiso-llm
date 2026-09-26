//! `audio_ops::load` builds the same linear from every container the speech
//! recognizer ships in, and the device product agrees with the container's
//! own values dequantised on the host.
//!
//! Needs `SHISO_TEST_MODEL_ASR` (the checkpoint directory holding `gguf/`,
//! `mlx/` and `nvidia/`). The `Q4_K` case is checked for shape and finiteness
//! only: the host has no `Q4_K` dequantiser, and the engine's projection that
//! runs it is covered by its own suites.
#![cfg(target_os = "macos")]
#![expect(
    clippy::cast_precision_loss,
    reason = "the fixture values are small literal-sized signals"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer};
use shiso_models::audio_ops::Ops;
use shiso_models::audio_ops::load::{LinearSource, linear, upload_f32};
use shiso_models::audio_ops::store::{TensorStore, dequant, dequant_affine};

#[expect(
    clippy::print_stdout,
    reason = "a skipped case says why, and the measured error is the report"
)]
fn say(line: &str) {
    println!("{line}");
}

fn root() -> Option<PathBuf> {
    std::env::var_os("SHISO_TEST_MODEL_ASR").map(PathBuf::from)
}

fn signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7919 % 2003) as f32 / 1001.5) - 1.0)
        .collect()
}

fn read(backend: &Arc<dyn Backend>, buf: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; usize::try_from(buf.len()).expect("fits")];
    backend
        .copy_d2h(buf.view(), &mut bytes)
        .expect("reads back");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One container's layer-0 query projection.
struct Case {
    what: &'static str,
    file: &'static str,
    safetensors: bool,
    weight: &'static str,
    affine: bool,
    host_oracle: bool,
}

const CASES: [Case; 4] = [
    Case {
        what: "hf f32",
        file: "nvidia/model.safetensors",
        safetensors: true,
        weight: "encoder.layers.0.self_attn.q_proj.weight",
        affine: false,
        host_oracle: true,
    },
    Case {
        what: "mlx affine8",
        file: "mlx/model.safetensors",
        safetensors: true,
        weight: "encoder.layers.0.self_attn.linear_q.weight",
        affine: true,
        host_oracle: true,
    },
    Case {
        what: "asr q8_0",
        file: "nvidia/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        safetensors: false,
        weight: "encoder.layers.0.self_attn.linear_q.weight",
        affine: false,
        host_oracle: true,
    },
    Case {
        what: "parakeet q4_k",
        file: "gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf",
        safetensors: false,
        weight: "enc.blocks.0.attn.linear_q.weight",
        affine: false,
        host_oracle: false,
    },
];

fn open(path: &Path, safetensors: bool) -> TensorStore {
    if safetensors {
        TensorStore::open_safetensors(path)
    } else {
        TensorStore::open_gguf(path)
    }
    .expect("the store opens")
}

#[test]
#[ignore = "requires a Metal device"]
fn a_linear_from_every_container_matches_its_host_values() {
    let Some(root) = root() else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let metal = shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let backend: Arc<dyn Backend> = metal;
    let ops = Ops::new(Arc::clone(&backend));
    let (rows, dim) = (5usize, 1024usize);
    let x = signal(rows * dim);
    for case in &CASES {
        let path = root.join(case.file);
        if !path.exists() {
            say(&format!("SKIP {}: {} is absent", case.what, path.display()));
            continue;
        }
        let store = open(&path, case.safetensors);
        let src = LinearSource {
            weight: case.weight,
            bias: None,
            affine_group: case.affine.then_some(64),
        };
        let lin = linear(&ops, &store, &src, dim, dim).expect("the linear builds");
        let (xb, yb) = (
            upload_f32(&ops, &x).expect("uploads"),
            ops.scratch(rows * dim).expect("allocates"),
        );
        lin.forward(&ops, xb.view(), yb.view(), rows).expect("runs");
        backend.sync().expect("drains");
        let got = read(&backend, &yb);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{}: non-finite output",
            case.what
        );
        if !case.host_oracle {
            say(&format!("{}: shape and finiteness only", case.what));
            continue;
        }
        let w = if case.affine {
            let stem = case.weight.trim_end_matches(".weight");
            dequant_affine(
                &store.require(case.weight).expect("words"),
                &store.require(&format!("{stem}.scales")).expect("scales"),
                &store.require(&format!("{stem}.biases")).expect("biases"),
                8,
                64,
            )
            .expect("dequantises")
        } else {
            dequant(&store.require(case.weight).expect("present")).expect("dequantises")
        };
        let mut worst = 0.0f32;
        for r in 0..rows {
            for c in 0..dim {
                let want: f32 = (0..dim).map(|i| x[r * dim + i] * w[c * dim + i]).sum();
                worst = worst.max((got[r * dim + c] - want).abs() / want.abs().max(1.0));
            }
        }
        say(&format!("{}: worst relative error {worst:.2e}", case.what));
        let tol = 1e-4;
        assert!(worst <= tol, "{}: worst {worst} over {tol}", case.what);
    }
}

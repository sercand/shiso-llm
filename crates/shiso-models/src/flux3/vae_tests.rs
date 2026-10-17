//! The autoencoder against the reference implementation's own module.
//!
//! `scripts/reference/flux3_vae_reference.py` runs the reference's
//! `AutoEncoder` at 32 base channels (the real one has 128) with seeded
//! weights in float32 and writes the weights, a picture, its tokens, a latent
//! and the decoded picture under `test-vectors/references/flux3/vae-tiny/`.
//! The reference's own loader refuses any key the module does not have, so
//! the state-dict keys it records are exactly the names a real
//! `ae.safetensors` carries; [`expected_names`] rebuilds that list from the
//! architecture and is checked against both the recorded keys and, when one
//! is at hand, the real file's header.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the comparison's rms is an f64 sum narrowed to the f32 the device computes in"
)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::Backend;
use shiso_vision::generator::ImageGenError;

use super::vae::Vae;
use crate::diffusion::ops::Ops;
use crate::diffusion::weights::StStore;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/flux3/vae-tiny")
}

#[expect(
    clippy::print_stderr,
    reason = "a skipped reference case must say why on stderr"
)]
fn skip(reason: &str) {
    eprintln!("SKIP flux3 vae reference case: {reason}");
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[expect(
    clippy::print_stderr,
    reason = "the device cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the vae device case");
        return None;
    }
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    Some(backend)
}

/// Asserts `|got - want| <= tol * max(|want|, rms(want))` element by element.
fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let rms =
        (want.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / want.len() as f64).sqrt() as f32;
    let mut worst = 0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let scale = w.abs().max(rms);
        let error = (g - w).abs();
        worst = worst.max(error / scale);
        assert!(
            error <= tol * scale,
            "{what}[{i}]: device {g} vs reference {w} (bound {})",
            tol * scale
        );
    }
    #[expect(
        clippy::print_stderr,
        reason = "the measured agreement is the case's report"
    )]
    {
        eprintln!("{what}: worst error {worst:.3e} of max(|w|, rms)");
    }
}

/// The tensor names of the reference's `AutoEncoder` with four levels of
/// widths `(1, 2, 4, 4)` x base, two residual blocks per encoder level and
/// three per decoder level, without `bn.num_batches_tracked`.
fn expected_names() -> BTreeSet<String> {
    let wb = |base: String| [format!("{base}.weight"), format!("{base}.bias")];
    let mut out: Vec<String> = Vec::new();
    let resnet = |base: &str, shortcut: bool, out: &mut Vec<String>| {
        for part in ["norm1", "conv1", "norm2", "conv2"] {
            out.extend(wb(format!("{base}.{part}")));
        }
        if shortcut {
            out.extend(wb(format!("{base}.nin_shortcut")));
        }
    };
    let mid = |base: &str, out: &mut Vec<String>| {
        resnet(&format!("{base}.block_1"), false, out);
        resnet(&format!("{base}.block_2"), false, out);
        for part in ["norm", "q", "k", "v", "proj_out"] {
            out.extend(wb(format!("{base}.attn_1.{part}")));
        }
    };
    for part in ["conv_in", "norm_out", "conv_out", "quant_conv"] {
        out.extend(wb(format!("encoder.{part}")));
    }
    // Encoder blocks change width at the first block of levels 1 and 2.
    for level in 0..4 {
        for block in 0..2 {
            resnet(
                &format!("encoder.down.{level}.block.{block}"),
                block == 0 && (level == 1 || level == 2),
                &mut out,
            );
        }
        if level != 3 {
            out.extend(wb(format!("encoder.down.{level}.downsample.conv")));
        }
    }
    mid("encoder.mid", &mut out);
    for part in ["post_quant_conv", "conv_in", "norm_out", "conv_out"] {
        out.extend(wb(format!("decoder.{part}")));
    }
    mid("decoder.mid", &mut out);
    // Decoder blocks change width at the first block of levels 1 and 0.
    for level in 0..4 {
        for block in 0..3 {
            resnet(
                &format!("decoder.up.{level}.block.{block}"),
                block == 0 && (level == 1 || level == 0),
                &mut out,
            );
        }
        if level != 0 {
            out.extend(wb(format!("decoder.up.{level}.upsample.conv")));
        }
    }
    out.extend(["bn.running_mean".to_owned(), "bn.running_var".to_owned()]);
    out.into_iter().collect()
}

#[test]
fn the_expected_names_are_the_reference_modules_state_dict() {
    let meta = fixture_dir().join("meta.json");
    let Ok(text) = std::fs::read_to_string(&meta) else {
        skip(
            "test-vectors/references/flux3/vae-tiny is absent; run scripts/reference/flux3_vae_reference.py",
        );
        return;
    };
    let meta: serde_json::Value = serde_json::from_str(&text).expect("meta.json parses");
    let recorded: BTreeSet<String> = meta["state_keys"]
        .as_array()
        .expect("state_keys")
        .iter()
        .map(|k| k.as_str().expect("a key").to_owned())
        .filter(|k| k != "bn.num_batches_tracked")
        .collect();
    let expected = expected_names();
    let missing: Vec<_> = recorded.difference(&expected).collect();
    let extra: Vec<_> = expected.difference(&recorded).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "the reference module has {missing:?} we do not list and we list {extra:?} it lacks"
    );
}

#[test]
fn vae_tensors_match_ae_safetensors() {
    // The reference's loader refuses missing and unexpected keys, so a real
    // `ae.safetensors` carries exactly the module's state-dict names.
    let Some(path) = std::env::var_os("SHISO_TEST_MODEL_FLUX3_VAE").map(PathBuf::from) else {
        skip("SHISO_TEST_MODEL_FLUX3_VAE is not set, so no real ae.safetensors header was read");
        return;
    };
    let path = if path.is_dir() {
        path.join("ae.safetensors")
    } else {
        path
    };
    let store = StStore::open(&path).expect("the header opens");
    let have: BTreeSet<String> = store
        .names()
        .map(str::to_owned)
        .filter(|n| n != "bn.num_batches_tracked")
        .collect();
    let want = expected_names();
    let missing: Vec<_> = want.difference(&have).collect();
    let extra: Vec<_> = have.difference(&want).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{}: missing {missing:?}, unexpected {extra:?}",
        path.display()
    );
    // And the shapes of the real widths: 128 base channels, 32 latent
    // channels, the 128 packed batch-norm features.
    let shape = |name: &str| store.require(name).expect("present").shape.clone();
    assert_eq!(shape("encoder.conv_in.weight"), [128, 3, 3, 3]);
    assert_eq!(shape("encoder.quant_conv.weight"), [64, 64, 1, 1]);
    assert_eq!(shape("decoder.post_quant_conv.weight"), [32, 32, 1, 1]);
    assert_eq!(shape("decoder.conv_out.weight"), [3, 128, 3, 3]);
    assert_eq!(shape("bn.running_mean"), [128]);
}

fn load_fixture(ops: &Ops) -> Option<Vae> {
    let weights = fixture_dir().join("weights.safetensors");
    if !weights.is_file() {
        skip(
            "test-vectors/references/flux3/vae-tiny is absent; run scripts/reference/flux3_vae_reference.py",
        );
        return None;
    }
    Some(Vae::load(&weights, ops).expect("the reference module's own tensor names load"))
}

#[test]
#[ignore = "requires a Metal device"]
fn encode_matches_the_reference_autoencoder() {
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let Some(vae) = load_fixture(&ops) else {
        return;
    };
    let dir = fixture_dir();
    let tokens = vae
        .encode(&ops, &read_f32(&dir.join("image.f32")), 48, 64)
        .expect("encodes");
    close(
        &tokens,
        &read_f32(&dir.join("tokens.f32")),
        1e-3,
        "encode tokens",
    );
}

#[test]
#[ignore = "requires a Metal device"]
fn decode_matches_the_reference_autoencoder() {
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let Some(vae) = load_fixture(&ops) else {
        return;
    };
    let dir = fixture_dir();
    let raw = vae
        .decode_raw(&ops, &read_f32(&dir.join("latent.f32")), 48, 64)
        .expect("decodes");
    let want = read_f32(&dir.join("decoded.f32"));
    // The fixture spans well past [-1, 1], so the unclamped field is what
    // can show a wrong norm, shortcut or upsample order.
    assert!(
        want.iter().any(|v| v.abs() > 1.2),
        "the fixture must exercise the unclamped range"
    );
    close(&raw, &want, 1e-3, "decode field");
    let mapped = vae
        .decode(&ops, &read_f32(&dir.join("latent.f32")), 48, 64)
        .expect("decodes");
    for (m, w) in mapped.iter().zip(&want) {
        assert!((m - (w.clamp(-1.0, 1.0) + 1.0) * 127.5).abs() < 0.05);
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn off_grid_pictures_and_short_token_runs_are_refused() {
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let Some(vae) = load_fixture(&ops) else {
        return;
    };
    assert!(matches!(
        vae.encode(&ops, &vec![0.0; 3 * 40 * 48], 48, 40),
        Err(ImageGenError::Backend(_))
    ));
    assert!(matches!(
        vae.decode_raw(&ops, &vec![0.0; 128], 48, 64),
        Err(ImageGenError::Backend(_))
    ));
}

#[test]
#[ignore = "requires a Metal device"]
fn a_file_missing_a_decoder_tensor_is_refused_naming_it() {
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let weights = fixture_dir().join("weights.safetensors");
    if !weights.is_file() {
        skip("the vae-tiny fixture is absent");
        return;
    }
    // Rewrite the file without `decoder.up.2.block.1.conv1.weight`.
    let bytes = std::fs::read(&weights).expect("reads");
    let n =
        usize::try_from(u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"))).expect("fits");
    let mut header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n]).expect("header");
    header.remove("decoder.up.2.block.1.conv1.weight");
    let text = serde_json::Value::Object(header).to_string();
    let mut file = (text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(text.as_bytes());
    file.extend_from_slice(&bytes[8 + n..]);
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ae.safetensors");
    std::fs::write(&path, file).expect("writes");
    match Vae::load(&path, &ops) {
        Err(ImageGenError::Checkpoint(message)) => {
            assert!(
                message.contains("decoder.up.2.block.1.conv1.weight"),
                "{message}"
            );
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

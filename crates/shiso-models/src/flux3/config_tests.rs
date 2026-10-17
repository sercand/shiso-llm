//! Locator and inspector refusals, against checkpoints this file forges.
//!
//! The forged `DiT` is tiny (hidden 256, two heads, one block per stack) but
//! has the real tensor names and the real relations between shapes, with the
//! text conditioning at its true 20480 width, so every refusal runs on any
//! machine without a checkpoint.

use std::path::{Path, PathBuf};

use shiso_vision::generator::ImageGenError;

use super::config::{Flux3Params, inspect, locate_and_inspect};
use crate::diffusion::gguf_forge::{ForgedTensor, write_gguf};
use crate::diffusion::weights::StStore;
use crate::image_registry::{DEFAULT_IMAGE_RESERVE_BYTES, ImageModelDesc, ImageResidency};

const H: u64 = 256;

/// What a forged checkpoint gets wrong.
pub(super) struct Forge {
    /// A tensor to leave out.
    pub(super) omit: Option<&'static str>,
    /// The width of `txt_in.weight`'s input.
    pub(super) txt_in: u64,
    /// The width of the QK norm scales.
    pub(super) head_dim: u64,
    /// Blocks in the `image_cond` stack (the others have one).
    pub(super) cond_depth: u64,
    /// `general.architecture` of the GGUF.
    pub(super) arch: &'static str,
}

impl Default for Forge {
    fn default() -> Self {
        Self {
            omit: None,
            txt_in: 20_480,
            head_dim: 128,
            cond_depth: 1,
            arch: "flux",
        }
    }
}

/// Every tensor of the forged `DiT` as `(name, shape in out-then-in order)`.
pub(super) fn layout(forge: &Forge) -> Vec<(String, Vec<u64>)> {
    let mlp = 3 * H;
    let mut out: Vec<(String, Vec<u64>)> = Vec::new();
    let block = |prefix: String, out: &mut Vec<(String, Vec<u64>)>| {
        for (name, shape) in [
            ("q_proj.weight", vec![H, H]),
            ("k_proj.weight", vec![H, H]),
            ("v_proj.weight", vec![H, H]),
            ("attn_out.weight", vec![H, H]),
            ("mlp_in.weight", vec![2 * mlp, H]),
            ("mlp_out.weight", vec![H, mlp]),
            ("norm.query_norm.scale", vec![forge.head_dim]),
            ("norm.key_norm.scale", vec![forge.head_dim]),
        ] {
            out.push((format!("{prefix}.{name}"), shape));
        }
    };
    block("content_mode_blocks.image.0".into(), &mut out);
    for i in 0..forge.cond_depth {
        block(format!("content_mode_blocks.image_cond.{i}"), &mut out);
    }
    block("txt_mode_blocks.0".into(), &mut out);
    block("single_blocks.0".into(), &mut out);
    for (name, shape) in [
        ("ref_time_in.in_layer.weight", vec![H, 256]),
        ("ref_time_in.out_layer.weight", vec![H, H]),
        ("time_in.in_layer.weight", vec![H, 256]),
        ("time_in.out_layer.weight", vec![H, H]),
        ("txt_in.weight", vec![H, forge.txt_in]),
        ("emb_in.image.weight", vec![H, 128]),
        ("emb_in.image_cond.weight", vec![H, 128]),
        ("early_stream_modulations.image.lin.weight", vec![3 * H, H]),
        (
            "early_stream_modulations.image_cond.lin.weight",
            vec![3 * H, H],
        ),
        ("early_stream_modulations.txt.lin.weight", vec![3 * H, H]),
        ("single_stream_modulations.image.lin.weight", vec![3 * H, H]),
        (
            "single_stream_modulations.image_cond.lin.weight",
            vec![3 * H, H],
        ),
        ("single_stream_modulations.txt.lin.weight", vec![3 * H, H]),
        (
            "final_layer.image.adaLN_modulation.1.weight",
            vec![2 * H, H],
        ),
        ("final_layer.image.linear.weight", vec![128, H]),
    ] {
        out.push((name.to_string(), shape));
    }
    out.retain(|(name, _)| Some(name.as_str()) != forge.omit);
    out
}

/// Whether a forged tensor is an F32 norm scale (else BF16).
pub(super) fn is_f32(name: &str) -> bool {
    name.rsplit('.').next() == Some("scale")
}

fn bytes_of(name: &str, shape: &[u64]) -> Vec<u8> {
    let elems: u64 = shape.iter().product();
    vec![0u8; usize::try_from(elems).expect("tiny") * if is_f32(name) { 4 } else { 2 }]
}

fn write_safetensors(path: &Path, forge: &Forge) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    let mut payload = Vec::new();
    for (name, shape) in layout(forge) {
        let bytes = bytes_of(&name, &shape);
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": if is_f32(&name) { "F32" } else { "BF16" },
                "shape": shape,
                "data_offsets": [offset, offset + bytes.len()],
            }),
        );
        offset += bytes.len();
        payload.extend_from_slice(&bytes);
    }
    let text = serde_json::Value::Object(header).to_string();
    let mut file = Vec::new();
    file.extend_from_slice(&(text.len() as u64).to_le_bytes());
    file.extend_from_slice(text.as_bytes());
    file.extend_from_slice(&payload);
    std::fs::write(path, file).expect("write safetensors");
}

fn write_forged_gguf(path: &Path, forge: &Forge) {
    let tensors = layout(forge);
    let payloads: Vec<Vec<u8>> = tensors.iter().map(|(n, s)| bytes_of(n, s)).collect();
    let nes: Vec<Vec<u64>> = tensors
        .iter()
        .map(|(_, s)| s.iter().rev().copied().collect())
        .collect();
    let forged: Vec<ForgedTensor<'_>> = tensors
        .iter()
        .zip(&payloads)
        .zip(&nes)
        .map(|(((name, _), bytes), ne)| ForgedTensor {
            name,
            ne,
            type_id: if is_f32(name) { 0 } else { 30 },
            bytes,
        })
        .collect();
    write_gguf(path, &[("general.architecture", forge.arch)], &forged);
}

fn desc_for(
    model: &Path,
    text: Option<PathBuf>,
    vae: Option<PathBuf>,
    tok: Option<PathBuf>,
) -> ImageModelDesc {
    ImageModelDesc {
        model_path: model.to_path_buf(),
        text_encoder_path: text,
        vae_path: vae,
        tokenizer_path: tok,
        assets_dir: None,
        residency: ImageResidency::Auto,
        reserve_bytes: DEFAULT_IMAGE_RESERVE_BYTES,
        draft_steps: 0,
        final_reuse: None,
    }
}

/// A directory with a forged `DiT` and stand-in companions; returns the model
/// path and a descriptor that resolves every companion.
fn checkpoint(forge: &Forge, gguf: bool) -> (tempfile::TempDir, PathBuf, ImageModelDesc) {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = dir
        .path()
        .join(if gguf { "dit.gguf" } else { "dit.safetensors" });
    if gguf {
        write_forged_gguf(&model, forge);
    } else {
        write_safetensors(&model, forge);
    }
    let text = dir.path().join("text_encoder");
    std::fs::create_dir(&text).expect("mkdir");
    std::fs::write(text.join("model.safetensors"), b"x").expect("text weights");
    std::fs::write(text.join("config.json"), b"{}").expect("text config");
    std::fs::write(dir.path().join("ae.safetensors"), b"x").expect("vae");
    std::fs::create_dir(dir.path().join("tokenizer")).expect("mkdir");
    let desc = desc_for(&model, None, None, None);
    (dir, model, desc)
}

fn refusal(result: Result<impl std::fmt::Debug, ImageGenError>, needles: &[&str]) {
    match result {
        Err(ImageGenError::Checkpoint(message)) => {
            for needle in needles {
                assert!(
                    message.contains(needle),
                    "refusal lacks {needle:?}: {message}"
                );
            }
        }
        other => panic!("expected a checkpoint refusal naming {needles:?}, got {other:?}"),
    }
}

#[test]
fn params_read_from_shapes() {
    let (_dir, _model, desc) = checkpoint(&Forge::default(), false);
    let located = locate_and_inspect(&desc).expect("the forged checkpoint is valid");
    assert_eq!(
        located.params,
        Flux3Params {
            hidden: 256,
            heads: 2,
            depth_mode: 1,
            depth_single: 1,
            mlp_hidden: 768,
            txt_in: 20_480,
            img_in: 128,
        }
    );
    assert!(located.vae_path.ends_with("ae.safetensors"));
    assert!(located.tokenizer_path.ends_with("tokenizer"));
    assert!(
        located
            .text_encoder_config
            .ends_with("text_encoder/config.json")
    );
    assert_eq!(located.text_encoder_files.len(), 1);
}

#[test]
fn params_read_from_a_gguf_too() {
    let (_dir, _model, desc) = checkpoint(&Forge::default(), true);
    let located = locate_and_inspect(&desc).expect("gguf with architecture flux");
    assert_eq!(located.params.hidden, 256);
    assert_eq!(located.params.heads, 2);
}

#[test]
fn the_real_depths_are_counted_from_block_names() {
    // Five per-mode blocks and 28 single blocks: only names and shapes
    // matter to the count, so tiny tensors stand in.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("dit.safetensors");
    write_safetensors(&path, &Forge::default());
    let store = StStore::open(&path).expect("opens");
    assert_eq!(store.count_blocks("content_mode_blocks.image."), Some(1));
    assert_eq!(
        store.count_blocks("content_mode_blocks.image_cond."),
        Some(1)
    );
    assert_eq!(store.count_blocks("txt_mode_blocks."), Some(1));
    let names = [
        "single_blocks.0.q_proj.weight",
        "single_blocks.27.q_proj.weight",
        "single_blocks.3.q_proj.weight",
    ];
    assert_eq!(
        crate::diffusion::weights::count_blocks_internal(names.iter().copied(), "single_blocks."),
        Some(28)
    );
}

#[test]
fn base_checkpoint_without_ref_time_in_is_refused() {
    let forge = Forge {
        omit: Some("ref_time_in.in_layer.weight"),
        ..Forge::default()
    };
    let (_dir, _model, desc) = checkpoint(&forge, false);
    refusal(
        locate_and_inspect(&desc),
        &["ref_time_in.in_layer.weight", "Base"],
    );
}

#[test]
fn gguf_architecture_must_be_flux() {
    let forge = Forge {
        arch: "llama",
        ..Forge::default()
    };
    let (_dir, _model, desc) = checkpoint(&forge, true);
    refusal(
        locate_and_inspect(&desc),
        &["general.architecture", "llama", "flux"],
    );
}

#[test]
fn text_width_must_be_20480() {
    let forge = Forge {
        txt_in: 4096,
        ..Forge::default()
    };
    let (_dir, _model, desc) = checkpoint(&forge, false);
    refusal(locate_and_inspect(&desc), &["txt_in.weight", "20480"]);
}

#[test]
fn a_missing_block_tensor_is_named() {
    let forge = Forge {
        omit: Some("single_blocks.0.v_proj.weight"),
        ..Forge::default()
    };
    let (_dir, _model, desc) = checkpoint(&forge, false);
    refusal(
        locate_and_inspect(&desc),
        &["single_blocks.0.v_proj.weight"],
    );
}

#[test]
fn heads_that_are_not_128_wide_are_refused() {
    let forge = Forge {
        head_dim: 64,
        ..Forge::default()
    };
    let (_dir, _model, desc) = checkpoint(&forge, false);
    refusal(
        locate_and_inspect(&desc),
        &["content_mode_blocks.image.0.norm.query_norm.scale", "128"],
    );
}

#[test]
fn per_mode_stacks_of_different_depth_are_refused() {
    let forge = Forge {
        cond_depth: 2,
        ..Forge::default()
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("dit.safetensors");
    write_safetensors(&path, &forge);
    let store = StStore::open(&path).expect("opens");
    refusal(inspect(&store), &["image_cond", "disagree"]);
}

#[test]
fn draft_steps_and_final_reuse_are_refused_but_swap_is_not() {
    let (_dir, _model, mut desc) = checkpoint(&Forge::default(), false);
    desc.residency = ImageResidency::Swap;
    locate_and_inspect(&desc).expect("flux3 drives swap residency itself");
    desc.draft_steps = 2;
    refusal(locate_and_inspect(&desc), &["--image-draft-steps", "flux3"]);
    desc.draft_steps = 0;
    desc.final_reuse = Some((1, 3));
    refusal(locate_and_inspect(&desc), &["--image-final-reuse", "flux3"]);
}

#[test]
fn a_missing_companion_names_the_flag() {
    let (dir, _model, desc) = checkpoint(&Forge::default(), false);
    std::fs::remove_file(dir.path().join("ae.safetensors")).expect("rm");
    refusal(
        locate_and_inspect(&desc),
        &["ae.safetensors", "--image-vae"],
    );
}

#[test]
fn an_unsupported_gguf_quant_is_refused_naming_type_and_tensor() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("dit.gguf");
    let bytes = vec![0u8; 136];
    write_gguf(
        &path,
        &[("general.architecture", "flux")],
        &[ForgedTensor {
            name: "single_blocks.0.q_proj.weight",
            ne: &[256, 1],
            type_id: 23,
            bytes: &bytes,
        }],
    );
    let desc = desc_for(&path, None, None, None);
    refusal(
        locate_and_inspect(&desc),
        &["iq4_xs", "single_blocks.0.q_proj.weight"],
    );
}

/// The real export, when `SHISO_TEST_MODEL_FLUX3` names it: the shapes the
/// shipped file declares must satisfy every relation the inspector checks.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "a skipped checkpoint test must say why on stderr"
)]
fn the_real_export_inspects_as_3072_wide_with_24_heads() {
    let Some(path) = std::env::var_os("SHISO_TEST_MODEL_FLUX3").map(PathBuf::from) else {
        eprintln!("SKIP the_real_export_inspects: SHISO_TEST_MODEL_FLUX3 is unset");
        return;
    };
    let store = StStore::open(&path).expect("the export opens");
    let params = inspect(&store).expect("the export satisfies every shape relation");
    assert_eq!(
        params,
        Flux3Params {
            hidden: 3072,
            heads: 24,
            depth_mode: 5,
            depth_single: 28,
            mlp_hidden: 9216,
            txt_in: 20_480,
            img_in: 128,
        }
    );
}

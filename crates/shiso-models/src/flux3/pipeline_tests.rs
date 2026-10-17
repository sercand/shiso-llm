//! The pipeline end to end on a tiny model this file forges: a 256-wide
//! transformer with the family's real tensor names and its true 20480-wide
//! text input, a text encoder 2560 wide (eight taps of it make that 20480)
//! with 32 layers and a toy vocabulary, and a 32-channel autoencoder with
//! the reference's tensor names. The weights are seeded noise, so the
//! pictures mean nothing; what is checked is plumbing - shapes, determinism,
//! crops, refusals - on a device, the same way a real checkpoint would run.
#![expect(
    clippy::cast_precision_loss,
    reason = "the forge turns small counts into weight scales"
)]

use std::path::Path;
use std::sync::Arc;

use image::{ImageFormat, Rgb, RgbImage};
use shiso_core::Backend;
use shiso_core::ids::Seed;
use shiso_vision::generator::{
    ImageGenError, ImageGenerationJob, ImageGenerator, ImageOutputFormat, InputReferenceImage,
};

use super::config_tests::{Forge, is_f32, layout};
use super::text_encoder_tests::{toy_tokenizer, write_decoder_sized};
use crate::image_registry::{
    DEFAULT_IMAGE_RESERVE_BYTES, ImageModelDesc, ImageResidency, open_image_generator,
};

/// A deterministic stream in `[-1, 1]`.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<u64>, Vec<u8>, &str)]) {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, shape, bytes, dtype) in tensors {
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [payload.len(), payload.len() + bytes.len()],
            }),
        );
        payload.extend_from_slice(bytes);
    }
    let text = serde_json::Value::Object(header).to_string();
    let mut file = (text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(text.as_bytes());
    file.extend_from_slice(&payload);
    std::fs::write(path, file).expect("writes safetensors");
}

/// Seeded random weights for the tiny transformer: BF16 linears at a scale
/// that keeps activations bounded, F32 norm scales near one.
fn forge_dit(path: &Path) {
    let mut noise = Noise(11);
    let tensors: Vec<_> = layout(&Forge::default())
        .into_iter()
        .map(|(name, shape)| {
            let count: u64 = shape.iter().product();
            let fan_in = *shape.last().expect("a shape") as f32;
            let scale = 0.5 / fan_in.sqrt();
            let mut bytes = Vec::new();
            for _ in 0..count {
                if is_f32(&name) {
                    bytes.extend_from_slice(&(1.0 + 0.1 * noise.next()).to_le_bytes());
                } else {
                    bytes.extend_from_slice(
                        &half::bf16::from_f32(noise.next() * scale)
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
            }
            let dtype = if is_f32(&name) { "F32" } else { "BF16" };
            (name, shape, bytes, dtype)
        })
        .collect();
    write_safetensors(path, &tensors);
}

/// The tiny autoencoder's tensors: base width 32, levels `(1, 2, 4, 4)`.
fn vae_layout() -> Vec<(String, Vec<u64>)> {
    const BASE: u64 = 32;
    let widths = [BASE, 2 * BASE, 4 * BASE, 4 * BASE];
    let mut out: Vec<(String, Vec<u64>)> = Vec::new();
    let conv = |name: String, o: u64, i: u64, k: u64, out: &mut Vec<(String, Vec<u64>)>| {
        out.push((format!("{name}.weight"), vec![o, i, k, k]));
        out.push((format!("{name}.bias"), vec![o]));
    };
    let norm = |name: String, c: u64, out: &mut Vec<(String, Vec<u64>)>| {
        out.push((format!("{name}.weight"), vec![c]));
        out.push((format!("{name}.bias"), vec![c]));
    };
    let resnet = |base: String, i: u64, o: u64, out: &mut Vec<(String, Vec<u64>)>| {
        norm(format!("{base}.norm1"), i, out);
        conv(format!("{base}.conv1"), o, i, 3, out);
        norm(format!("{base}.norm2"), o, out);
        conv(format!("{base}.conv2"), o, o, 3, out);
        if i != o {
            conv(format!("{base}.nin_shortcut"), o, i, 1, out);
        }
    };
    let mid = |base: &str, c: u64, out: &mut Vec<(String, Vec<u64>)>| {
        resnet(format!("{base}.block_1"), c, c, out);
        resnet(format!("{base}.block_2"), c, c, out);
        norm(format!("{base}.attn_1.norm"), c, out);
        for part in ["q", "k", "v", "proj_out"] {
            conv(format!("{base}.attn_1.{part}"), c, c, 1, out);
        }
    };
    conv("encoder.conv_in".into(), BASE, 3, 3, &mut out);
    let mut block_in = BASE;
    for (level, &w) in widths.iter().enumerate() {
        for block in 0..2 {
            resnet(
                format!("encoder.down.{level}.block.{block}"),
                block_in,
                w,
                &mut out,
            );
            block_in = w;
        }
        if level != 3 {
            conv(
                format!("encoder.down.{level}.downsample.conv"),
                w,
                w,
                3,
                &mut out,
            );
        }
    }
    mid("encoder.mid", block_in, &mut out);
    norm("encoder.norm_out".into(), block_in, &mut out);
    conv("encoder.quant_conv".into(), 64, 64, 1, &mut out);
    conv("encoder.conv_out".into(), 64, block_in, 3, &mut out);
    conv("decoder.post_quant_conv".into(), 32, 32, 1, &mut out);
    conv("decoder.conv_in".into(), widths[3], 32, 3, &mut out);
    mid("decoder.mid", widths[3], &mut out);
    let mut block_in = widths[3];
    for level in (0..4).rev() {
        for block in 0..3 {
            resnet(
                format!("decoder.up.{level}.block.{block}"),
                block_in,
                widths[level],
                &mut out,
            );
            block_in = widths[level];
        }
        if level != 0 {
            conv(
                format!("decoder.up.{level}.upsample.conv"),
                block_in,
                block_in,
                3,
                &mut out,
            );
        }
    }
    norm("decoder.norm_out".into(), block_in, &mut out);
    conv("decoder.conv_out".into(), 3, block_in, 3, &mut out);
    out.push(("bn.running_mean".into(), vec![128]));
    out.push(("bn.running_var".into(), vec![128]));
    out
}

fn forge_vae(path: &Path) {
    let mut noise = Noise(5);
    let tensors: Vec<_> = vae_layout()
        .into_iter()
        .map(|(name, shape)| {
            let count: u64 = shape.iter().product();
            let fan_in: u64 = shape.iter().skip(1).product::<u64>().max(1);
            let mut bytes = Vec::new();
            for _ in 0..count {
                let v = if name.starts_with("bn.running_var") {
                    1.0 + 0.2 * noise.next()
                } else if name.starts_with("bn.") {
                    0.2 * noise.next()
                } else if shape.len() == 1 && name.ends_with(".weight") {
                    1.0 + 0.1 * noise.next()
                } else if shape.len() == 1 {
                    0.05 * noise.next()
                } else {
                    noise.next() * 1.0 / (fan_in as f32).sqrt()
                };
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            (name, shape, bytes, "F32")
        })
        .collect();
    write_safetensors(path, &tensors);
}

/// A checkpoint directory of the tiny model and the descriptor that opens it.
pub(super) struct Tiny {
    /// Keeps the directory alive.
    pub _dir: tempfile::TempDir,
    /// What opens it.
    pub desc: ImageModelDesc,
}

pub(super) fn forge_tiny(residency: ImageResidency) -> Tiny {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    forge_dit(&root.join("dit.safetensors"));
    forge_vae(&root.join("ae.safetensors"));
    let text = root.join("text_encoder");
    std::fs::create_dir(&text).expect("mkdir");
    write_decoder_sized(&text, "model.language_model.", 32, 2560);
    let tokenizer = root.join("tokenizer");
    std::fs::create_dir(&tokenizer).expect("mkdir");
    toy_tokenizer(&tokenizer, Some("<|endoftext|>"));
    let desc = ImageModelDesc {
        model_path: root.join("dit.safetensors"),
        text_encoder_path: None,
        vae_path: None,
        tokenizer_path: None,
        assets_dir: None,
        residency,
        reserve_bytes: DEFAULT_IMAGE_RESERVE_BYTES,
        draft_steps: 0,
        final_reuse: None,
    };
    Tiny { _dir: dir, desc }
}

#[expect(
    clippy::print_stderr,
    reason = "the device cases are opt-in, so the reason they did not run belongs on stderr"
)]
pub(super) fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the flux3 pipeline case");
        return None;
    }
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    Some(backend)
}

pub(super) fn job(width: u32, height: u32, seed: u64) -> ImageGenerationJob {
    ImageGenerationJob {
        prompt: "word word word".to_owned(),
        width,
        height,
        steps: 2,
        seed: Seed::new(seed),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    }
}

fn png(w: u32, h: u32, salt: u32) -> InputReferenceImage {
    let img = RgbImage::from_fn(w, h, |x, y| {
        let v =
            |c: u32| u8::try_from((7 * x + 13 * y + 29 * c + x * y + salt) % 256).expect("byte");
        Rgb([v(0), v(1), v(2)])
    });
    let mut bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut bytes), ImageFormat::Png)
        .expect("encodes");
    InputReferenceImage { bytes }
}

fn dimensions(bytes: &[u8]) -> (u32, u32) {
    let img = image::load_from_memory(bytes).expect("the output decodes");
    (img.width(), img.height())
}

fn open(tiny: &Tiny, backend: &Arc<dyn Backend>) -> Arc<dyn ImageGenerator> {
    open_image_generator(&tiny.desc, backend).expect("the tiny model opens through the registry")
}

#[test]
fn the_forged_autoencoder_has_the_reference_tensor_names() {
    use std::collections::BTreeSet;
    let have: BTreeSet<String> = vae_layout().into_iter().map(|(n, _)| n).collect();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ae.safetensors");
    forge_vae(&path);
    let store = crate::diffusion::weights::StStore::open(&path).expect("opens");
    let on_disk: BTreeSet<String> = store.names().map(str::to_owned).collect();
    assert_eq!(have, on_disk);
    // 251 reference keys less `bn.num_batches_tracked`.
    assert_eq!(have.len(), 250);
}

#[test]
#[ignore = "requires a Metal device"]
fn same_seed_same_bytes() {
    let Some(backend) = device() else { return };
    let tiny = forge_tiny(ImageResidency::Resident);
    let model = open(&tiny, &backend);
    assert_eq!(model.model_id(), "flux-3-klein");
    let limits = model.limits();
    assert_eq!(
        (limits.size_multiple, limits.min_side, limits.max_side),
        (16, 64, 4096)
    );
    assert_eq!(
        (
            limits.default_size,
            limits.default_steps,
            limits.fixed_steps
        ),
        ((1024, 1024), 16, None)
    );
    let a = model.generate(&job(64, 96, 3), None).expect("generates");
    let b = model.generate(&job(64, 96, 3), None).expect("generates");
    let c = model.generate(&job(64, 96, 4), None).expect("generates");
    assert_eq!((a.width, a.height), (64, 96));
    assert_eq!(dimensions(&a.bytes), (64, 96));
    let pixels = image::load_from_memory(&a.bytes)
        .expect("decodes")
        .to_rgb8();
    let distinct: std::collections::BTreeSet<u8> = pixels.as_raw().iter().copied().collect();
    assert!(
        distinct.len() > 16,
        "a flat or saturated picture has {} levels",
        distinct.len()
    );
    assert_eq!(a.bytes, b.bytes, "the same seed must give the same bytes");
    assert_ne!(a.bytes, c.bytes, "another seed must give another picture");
    assert!(a.prompt_tokens > 0 && a.completion_tokens == 4 * 6 * 2);
}

#[test]
#[ignore = "requires a Metal device"]
fn a_single_reference_returns_at_its_own_size_when_the_canvas_was_asked_for() {
    let Some(backend) = device() else { return };
    let tiny = forge_tiny(ImageResidency::Resident);
    let model = open(&tiny, &backend);
    // 100x40 sits on a 112x64 canvas with the picture centred in it.
    let mut edit = job(112, 64, 9);
    edit.input_references = vec![png(100, 40, 0)];
    let at_canvas = model.generate(&edit, None).expect("edits");
    assert_eq!(
        (at_canvas.width, at_canvas.height),
        (100, 40),
        "the fitted box is cut back out of the canvas"
    );
    assert_eq!(dimensions(&at_canvas.bytes), (100, 40));

    // A job of any other size keeps its size.
    let mut other = job(64, 64, 9);
    other.input_references = vec![png(100, 40, 0)];
    assert_eq!(
        dimensions(&model.generate(&other, None).expect("edits").bytes),
        (64, 64)
    );

    // The reference changes the picture, and a different reference changes it again.
    let plain = model.generate(&job(112, 64, 9), None).expect("generates");
    let mut other_ref = job(112, 64, 9);
    other_ref.input_references = vec![png(100, 40, 77)];
    let other_ref = model.generate(&other_ref, None).expect("edits");
    assert_ne!(at_canvas.bytes, other_ref.bytes);
    assert_ne!(plain.bytes, at_canvas.bytes);
}

#[test]
#[ignore = "requires a Metal device"]
fn several_references_each_get_their_own_ids_and_the_output_keeps_the_job_size() {
    let Some(backend) = device() else { return };
    let tiny = forge_tiny(ImageResidency::Resident);
    let model = open(&tiny, &backend);
    let mut edit = job(64, 64, 2);
    edit.input_references = vec![png(48, 32, 0), png(64, 64, 5), png(33, 47, 9)];
    let out = model.generate(&edit, None).expect("edits");
    assert_eq!(dimensions(&out.bytes), (64, 64));
    // The order of the references is part of the request.
    edit.input_references.reverse();
    let reversed = model.generate(&edit, None).expect("edits");
    assert_ne!(out.bytes, reversed.bytes);
}

#[test]
#[ignore = "requires a Metal device"]
fn reference_refusals_name_the_reference() {
    let Some(backend) = device() else { return };
    let tiny = forge_tiny(ImageResidency::Resident);
    let model = open(&tiny, &backend);
    let mut many = job(64, 64, 1);
    many.input_references = (0..11).map(|i| png(32, 32, i)).collect();
    assert!(matches!(
        model.generate(&many, None),
        Err(ImageGenError::ReferenceImage(m)) if m.contains("11")
    ));
    let mut small = job(64, 64, 1);
    small.input_references = vec![png(8, 40, 0)];
    assert!(matches!(
        model.generate(&small, None),
        Err(ImageGenError::ReferenceImage(m)) if m.contains("8x40")
    ));
    let mut junk = job(64, 64, 1);
    junk.input_references = vec![InputReferenceImage {
        bytes: vec![1, 2, 3],
    }];
    assert!(matches!(
        model.generate(&junk, None),
        Err(ImageGenError::ReferenceImage(_))
    ));
    // Off-grid and out-of-range sizes are the job's own validation.
    assert!(matches!(
        model.generate(&job(70, 64, 1), None),
        Err(ImageGenError::InvalidDimensions { .. })
    ));
}

#[test]
#[ignore = "requires a Metal device"]
fn progress_ticks_once_per_step_without_a_partial_image() {
    let Some(backend) = device() else { return };
    let tiny = forge_tiny(ImageResidency::Resident);
    let model = open(&tiny, &backend);
    let seen = std::cell::RefCell::new(Vec::new());
    model
        .generate(
            &job(64, 64, 1),
            Some(&|event| seen.borrow_mut().push(event)),
        )
        .expect("generates");
    let seen = seen.into_inner();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].step, 0);
    assert_eq!(seen[1].step, 1);
    assert!(
        seen.iter()
            .all(|e| e.total_steps == 2 && e.partial_image.is_none())
    );
}

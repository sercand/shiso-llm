//! End-to-end: the Qwen-Image-2.1 Turbo install loads on Metal through the
//! registry and a real 512x512 RGBA image comes out.
//!
//! A flat field encodes to a perfectly valid PNG, so the bytes are decoded
//! and judged: exact size, four channels, real variance. The same seed must
//! reproduce the bytes exactly and another seed must not.
//!
//! Requires a Metal device and `SHISO_TEST_MODEL_QWEN_IMAGE21` (the
//! `turbo-int8` directory holding `image/` and `image_assets/`); without
//! either it says so on stderr and returns. `SHISO_TEST_IMAGE_OUT`, when
//! set, is where the first image is written for a human to look at.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::ids::Seed;
use shiso_metal::MetalBackend;
use shiso_models::image_registry::{DEFAULT_IMAGE_RESERVE_BYTES, ImageModelDesc, ImageResidency};
use shiso_vision::generator::{ImageGenerationJob, ImageGenerator, ImageOutputFormat};

/// A photograph's pixel standard deviation is tens; a flat field's is ~0.
const MIN_PIXEL_STDDEV: f64 = 10.0;

const PROMPT: &str = "a red fox standing in fresh snow, golden hour, photograph";

#[expect(
    clippy::cast_precision_loss,
    reason = "a 512x512x4 sample count is exact in f64"
)]
fn stddev(samples: &[u8]) -> f64 {
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    (samples
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

#[expect(
    clippy::print_stderr,
    reason = "timings and pixel statistics are the case's report"
)]
fn generate(generator: &Arc<dyn ImageGenerator>, seed: u64, label: &str) -> Vec<u8> {
    let job = ImageGenerationJob {
        prompt: PROMPT.to_string(),
        width: 512,
        height: 512,
        steps: 8,
        seed: Seed::new(seed),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    };
    let began = Instant::now();
    let last = std::cell::Cell::new(began);
    let report = |event: shiso_vision::generator::ImageStepEvent| {
        let now = Instant::now();
        eprintln!(
            "{label}: step {}/{} +{:.0} ms",
            event.step + 1,
            event.total_steps,
            (now - last.get()).as_secs_f64() * 1e3
        );
        assert!(event.partial_image.is_none(), "no partial previews");
        last.set(now);
    };
    let image = generator
        .generate(&job, Some(&report))
        .unwrap_or_else(|e| panic!("{label}: generation failed: {e}"));
    eprintln!(
        "{label}: generated in {:.2} s ({} prompt tokens)",
        began.elapsed().as_secs_f64(),
        image.prompt_tokens
    );
    assert_eq!(image.media_type, "image/png");
    assert_eq!((image.width, image.height), (512, 512));
    image.bytes
}

#[expect(
    clippy::print_stderr,
    reason = "a machine without the model skips with its reason on stderr"
)]
#[test]
#[ignore = "needs SHISO_TEST_MODEL_QWEN_IMAGE21 and a Metal GPU"]
fn qwen_image21_turbo_generates_a_real_rgba_image() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!("SKIP: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset");
        return;
    };
    if !shiso_metal::metal_available() {
        eprintln!("SKIP: no Metal device on this machine");
        return;
    }
    let root = PathBuf::from(root);
    let backend: Arc<dyn shiso_core::Backend> = MetalBackend::new().expect("Metal opens");
    let image = root.join("image");
    let desc = ImageModelDesc {
        model_path: image.join("diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"),
        text_encoder_path: Some(image.join("text_encoders")),
        vae_path: Some(image.join("vae")),
        tokenizer_path: None,
        assets_dir: Some(root.join("image_assets")),
        residency: ImageResidency::Auto,
        reserve_bytes: DEFAULT_IMAGE_RESERVE_BYTES,
    };
    let began = Instant::now();
    let generator = shiso_models::image_registry::open_image_generator(&desc, &backend)
        .expect("the Turbo install loads");
    eprintln!("load: {:.2} s", began.elapsed().as_secs_f64());
    assert_eq!(generator.model_id(), "qwen-image-2.1");
    assert_eq!(generator.limits().fixed_steps, Some(8));

    let first = generate(&generator, 42, "seed 42");
    let decoded = image::load_from_memory(&first).expect("the PNG decodes");
    assert_eq!((decoded.width(), decoded.height()), (512, 512));
    assert_eq!(
        decoded.color(),
        image::ColorType::Rgba8,
        "the image keeps its alpha channel"
    );
    let rgba = decoded.to_rgba8();
    let rgb: Vec<u8> = rgba
        .as_raw()
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    let sd = stddev(&rgb);
    eprintln!("seed 42: colour stddev {sd:.1}");
    assert!(sd > MIN_PIXEL_STDDEV, "a flat field: stddev {sd:.2}");
    if let Some(out) = std::env::var_os("SHISO_TEST_IMAGE_OUT") {
        std::fs::write(Path::new(&out), &first).expect("the image is written");
        eprintln!("wrote {}", Path::new(&out).display());
    }

    let again = generate(&generator, 42, "seed 42 again");
    assert_eq!(first, again, "the same seed reproduces the bytes");
    let other = generate(&generator, 7, "seed 7");
    assert_ne!(first, other, "another seed gives another image");
}

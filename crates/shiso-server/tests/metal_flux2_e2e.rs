//! End-to-end smoke: a real FLUX.2-klein checkpoint loads on Metal and a
//! real image comes out.
//!
//! The header-only checks a pipeline can also pass while emitting garbage
//! (a flat field encodes to a perfectly valid PNG), so this test reads the
//! encoded bytes back: IHDR/SOF0 must say 512x512, the decoded pixels must
//! have real variance, and the same seed must reproduce byte-identical
//! output. A denoising loop that silently collapses to a constant latent
//! fails here even though every container-level assert would pass.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::ids::Seed;
use shiso_metal::MetalBackend;
use shiso_models::image_registry::ImageModelDesc;
use shiso_vision::generator::{GeneratedImage, ImageGenerationJob, ImageOutputFormat};

/// Decoded-pixel standard deviation a real photograph clears by orders of
/// magnitude and a flat (or all-NaN-clamped-to-black) field never does.
const MIN_PIXEL_STDDEV: f64 = 1.0;

/// Reads width and height from a PNG IHDR chunk.
fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
    assert_eq!(
        &bytes[..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
        "output is not a PNG"
    );
    assert_eq!(&bytes[12..16], b"IHDR", "PNG has no IHDR at the front");
    let width = u32::from_be_bytes(
        bytes[16..20]
            .try_into()
            .expect("IHDR width field is four bytes"),
    );
    let height = u32::from_be_bytes(
        bytes[20..24]
            .try_into()
            .expect("IHDR height field is four bytes"),
    );
    (width, height)
}

/// Reads width and height from the first JPEG SOF marker (baseline or
/// progressive), walking the segment chain.
fn jpeg_dimensions(bytes: &[u8]) -> (u32, u32) {
    assert_eq!(&bytes[..2], &[0xFF, 0xD8], "output is not a JPEG");
    let mut i = 2;
    while i + 9 < bytes.len() {
        assert_eq!(bytes[i], 0xFF, "JPEG marker desync at byte {i}");
        let marker = bytes[i + 1];
        // Standalone markers (RSTn, SOI, EOI) carry no length field.
        if matches!(marker, 0xD0..=0xD9 | 0x01) {
            i += 2;
            continue;
        }
        let is_sof = matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC); // DHT, JPG, DAC
        let len = usize::from(u16::from_be_bytes(
            bytes[i + 2..i + 4]
                .try_into()
                .expect("segment length field is two bytes"),
        ));
        if is_sof {
            let height = u32::from(u16::from_be_bytes(
                bytes[i + 5..i + 7]
                    .try_into()
                    .expect("SOF height field is two bytes"),
            ));
            let width = u32::from(u16::from_be_bytes(
                bytes[i + 7..i + 9]
                    .try_into()
                    .expect("SOF width field is two bytes"),
            ));
            return (width, height);
        }
        i += 2 + len;
    }
    panic!("JPEG carries no start-of-frame segment");
}

/// Decodes the encoded bytes and asserts they are a real image, not a flat
/// field: exact requested dimensions, visible variance across the samples,
/// and enough distinct values that a constant (or NaN-collapsed) output
/// cannot pass.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel counts are bounded by the 2048x2048 generation contract, exact in f64"
)]
#[expect(
    clippy::print_stderr,
    reason = "the smoke test reports pixel statistics and timings on stderr under --nocapture"
)]
fn assert_real_image(bytes: &[u8], width: u32, height: u32, label: &str) {
    let image = image::load_from_memory(bytes)
        .unwrap_or_else(|e| panic!("{label}: encoded bytes do not decode: {e}"));
    assert_eq!(image.width(), width, "{label}: decoded width");
    assert_eq!(image.height(), height, "{label}: decoded height");
    let rgb = image.to_rgb8();
    let pixels = rgb.as_raw();
    let n = pixels.len() as f64;
    let mean = pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let variance = pixels
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    let stddev = variance.sqrt();
    let mut hist = [false; 256];
    for &v in pixels {
        hist[usize::from(v)] = true;
    }
    let distinct = hist.iter().filter(|&&hit| hit).count();
    eprintln!(
        "{label}: {width}x{height} pixels mean {mean:.1} stddev {stddev:.1} distinct values {distinct}"
    );
    assert!(
        stddev > MIN_PIXEL_STDDEV,
        "{label}: pixel stddev {stddev:.3} is below {MIN_PIXEL_STDDEV}: the pipeline \
         emitted a flat field, not a denoised image"
    );
    assert!(
        distinct > 16,
        "{label}: only {distinct} distinct pixel values: the pipeline emitted a \
         near-constant field (a NaN-collapsed or saturated output)"
    );
}

/// One full load + generate against `model_dir`, reporting timings on stderr.
#[expect(
    clippy::print_stderr,
    reason = "the smoke test reports load and per-step timings on stderr under --nocapture"
)]
fn generate_once(
    backend: &Arc<dyn shiso_core::Backend>,
    model_dir: &Path,
    assets_dir: Option<PathBuf>,
    format: ImageOutputFormat,
    label: &str,
) -> GeneratedImage {
    let desc = ImageModelDesc {
        model_path: model_dir.to_path_buf(),
        text_encoder_path: None,
        vae_path: None,
        tokenizer_path: None,
        assets_dir,
    };
    let load_start = Instant::now();
    let generator = shiso_models::image_registry::open_image_generator(&desc, backend)
        .unwrap_or_else(|e| panic!("{label}: checkpoint load failed: {e}"));
    eprintln!(
        "{label}: load took {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    let job = ImageGenerationJob {
        prompt: "a lovely red panda wearing an astronaut helmet floating in space".to_string(),
        width: 512,
        height: 512,
        steps: 4,
        seed: Seed::new(42),
        output_format: format,
        output_compression: None,
        input_references: Vec::new(),
    };

    // The progress callback only reports wall-clock between steps; it does
    // not touch generation state, so a run with and without it must produce
    // identical bytes (and the determinism half below relies on that).
    let gen_start = Instant::now();
    let last_step = RefCell::new(gen_start);
    let report = |event: shiso_vision::generator::ImageStepEvent| {
        let now = Instant::now();
        let delta = (now - *last_step.borrow()).as_secs_f64();
        last_step.replace(now);
        eprintln!(
            "{label}: step {}/{} done {:.1}s after generation start (+{delta:.1}s)",
            event.step + 1,
            event.total_steps,
            (now - gen_start).as_secs_f64()
        );
    };
    let result = generator
        .generate(&job, Some(&report))
        .unwrap_or_else(|e| panic!("{label}: generation failed: {e}"));
    eprintln!(
        "{label}: generation took {:.1}s",
        gen_start.elapsed().as_secs_f64()
    );
    result
}

#[expect(
    clippy::print_stderr,
    reason = "a machine with missing test models skips with explanation on stderr"
)]
#[tokio::test]
#[ignore = "needs SHISO_TEST_MODEL_FLUX2 or SHISO_TEST_MODEL_FLUX2_FP8 and Metal GPU"]
async fn test_flux2_klein_metal_generation() {
    if !shiso_metal::metal_available() {
        eprintln!("SKIP: no Metal device available on this machine");
        return;
    }

    let bf16_env = std::env::var("SHISO_TEST_MODEL_FLUX2").ok();
    let fp8_env = std::env::var("SHISO_TEST_MODEL_FLUX2_FP8").ok();

    if bf16_env.is_none() && fp8_env.is_none() {
        eprintln!("SKIP: neither SHISO_TEST_MODEL_FLUX2 nor SHISO_TEST_MODEL_FLUX2_FP8 is set");
        return;
    }

    let backend: Arc<dyn shiso_core::Backend> = MetalBackend::new().expect("Metal backend opens");

    // The host fallback is numerically the same arithmetic but orders of
    // magnitude slower; if the module table does not carry the diffusion
    // entries, this test would silently benchmark the fallback instead of
    // the kernels it exists to exercise.
    let entries = backend.module_entries("diffusion");
    assert!(
        entries
            .iter()
            .any(|name| name.contains("kernel_dit_gemm_bf16")),
        "the Metal backend does not expose the diffusion kernel module; the host \
         fallback would engage and the test would not exercise the kernels"
    );

    if let Some(path_str) = bf16_env.as_ref() {
        let model_bf16 = Path::new(path_str);
        if model_bf16.exists() {
            let result = generate_once(&backend, model_bf16, None, ImageOutputFormat::Png, "bf16");
            assert_eq!(&result.bytes[..4], &[0x89, b'P', b'N', b'G']);
            assert_eq!(result.media_type, "image/png");
            assert_eq!(
                png_dimensions(&result.bytes),
                (512, 512),
                "bf16: PNG IHDR disagrees with the requested size"
            );
            assert_real_image(&result.bytes, 512, 512, "bf16");

            // Same seed, same bytes: the Box-Muller stream and the kernel
            // sequence are both fixed, so a second run that differs means
            // uninitialized memory or a race is in the loop.
            let repeat = generate_once(
                &backend,
                model_bf16,
                None,
                ImageOutputFormat::Png,
                "bf16-repeat",
            );
            assert_eq!(
                result.bytes, repeat.bytes,
                "bf16: two generations with Seed(42) produced different bytes"
            );
            eprintln!("bf16: repeat run byte-identical (determinism ok)");
        } else {
            eprintln!("SKIP: SHISO_TEST_MODEL_FLUX2 directory does not exist: {path_str}");
        }
    }

    if let Some(path_str) = fp8_env.as_ref() {
        let model_fp8 = Path::new(path_str);
        if model_fp8.exists() {
            let assets_dir = bf16_env.as_ref().map(PathBuf::from);
            let desc = ImageModelDesc {
                model_path: model_fp8.to_path_buf(),
                text_encoder_path: None,
                vae_path: None,
                tokenizer_path: None,
                assets_dir,
            };
            let load_start = Instant::now();
            let generator = shiso_models::image_registry::open_image_generator(&desc, &backend)
                .expect("4B FP8 Model loads on Metal");
            eprintln!("fp8: load took {:.1}s", load_start.elapsed().as_secs_f64());
            assert_eq!(generator.model_id(), "flux-2-klein-4b-fp8");

            let job = ImageGenerationJob {
                prompt: "a lovely red panda wearing an astronaut helmet floating in space"
                    .to_string(),
                width: 512,
                height: 512,
                steps: 4,
                seed: Seed::new(42),
                output_format: ImageOutputFormat::Jpeg,
                output_compression: None,
                input_references: Vec::new(),
            };
            let gen_start = Instant::now();
            let result = generator
                .generate(&job, None)
                .expect("FP8 image generation succeeds");
            eprintln!(
                "fp8: generation took {:.1}s",
                gen_start.elapsed().as_secs_f64()
            );
            assert_eq!(&result.bytes[..2], &[0xFF, 0xD8]); // JPEG SOI marker
            assert_eq!(result.media_type, "image/jpeg");
            assert_eq!(
                jpeg_dimensions(&result.bytes),
                (512, 512),
                "fp8: JPEG SOF disagrees with the requested size"
            );
            assert_real_image(&result.bytes, 512, 512, "fp8");
        } else {
            eprintln!("SKIP: SHISO_TEST_MODEL_FLUX2_FP8 directory does not exist: {path_str}");
        }
    }
}

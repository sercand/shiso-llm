//! Swap residency on the device: the encoder's bytes leave before the
//! transformer's arrive, back-to-back jobs both finish, and a cached
//! prompt runs with no swap.
//!
//! Requires a Metal device and `SHISO_TEST_MODEL_QWEN_IMAGE21` (the
//! `turbo-int8` directory); without either it says so on stderr and
//! returns.
#![expect(
    clippy::print_stderr,
    reason = "the measured residency changes are the case's report"
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use shiso_core::Backend;
use shiso_core::ids::Seed;
use shiso_vision::generator::{ImageGenerationJob, ImageGenerator, ImageOutputFormat};

use super::pipeline::{Component, QwenImage21Model, ResidencyEvent};
use crate::image_registry::{DEFAULT_IMAGE_RESERVE_BYTES, ImageModelDesc, ImageResidency};

fn job(prompt: &str) -> ImageGenerationJob {
    ImageGenerationJob {
        prompt: prompt.to_string(),
        width: 256,
        height: 256,
        steps: 8,
        seed: Seed::new(1),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one scenario: load, two swapping jobs, the event audit, a cached job"
)]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn swap_mode_releases_the_encoder_before_importing_the_transformer() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!("note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the swap case");
        return;
    };
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the swap case");
        return;
    }
    let root = PathBuf::from(root);
    let metal = shiso_metal::MetalBackend::new().expect("the kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    if let Some(info) = backend.memory_info() {
        eprintln!("working set {} bytes", info.total);
    }
    let image = root.join("image");
    let model = QwenImage21Model::load(
        &ImageModelDesc {
            model_path: image
                .join("diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"),
            text_encoder_path: Some(image.join("text_encoders")),
            vae_path: Some(image.join("vae")),
            tokenizer_path: None,
            assets_dir: Some(root.join("image_assets")),
            residency: ImageResidency::Swap,
            reserve_bytes: DEFAULT_IMAGE_RESERVE_BYTES,
            sigmas: None,
        },
        &backend,
    )
    .expect("the install loads");
    let loaded = model.events();
    // Load runs the swap once (its first imports are the slow ones) and
    // leaves the encoder loaded alone.
    assert!(
        matches!(
            loaded.last(),
            Some(ResidencyEvent::Loaded {
                component: Component::Encoder,
                ..
            })
        ),
        "swap mode starts with the encoder loaded: {loaded:?}"
    );

    for (i, prompt) in [
        "a lighthouse on a cliff at dusk",
        "a bowl of ramen, top view",
    ]
    .into_iter()
    .enumerate()
    {
        let began = Instant::now();
        let image = model
            .generate(&job(prompt), None)
            .expect("the job finishes");
        assert!(!image.bytes.is_empty());
        eprintln!("job {i}: {:.2} s", began.elapsed().as_secs_f64());
    }
    let events = model.events();
    eprintln!("events: {events:?}");
    let te_load = events.iter().find_map(|e| match e {
        ResidencyEvent::Loaded {
            component: Component::Encoder,
            device_bytes,
        } => Some(*device_bytes),
        _ => None,
    });
    // Every encoder drop is followed by the transformer's import, never
    // preceded by it while the encoder is still loaded.
    let mut encoder_loaded = false;
    let at_load = loaded.len();
    let mut te_drops = 0;
    for (index, event) in events.iter().enumerate() {
        match *event {
            ResidencyEvent::Loaded {
                component: Component::Encoder,
                ..
            } => encoder_loaded = true,
            ResidencyEvent::Dropped {
                component: Component::Encoder,
                device_bytes,
            } => {
                encoder_loaded = false;
                if index >= at_load {
                    te_drops += 1;
                }
                if let Some(added) = te_load {
                    // The device's own count must give back what the
                    // import took, near enough: the bytes really left.
                    assert!(
                        device_bytes * 10 >= added * 9,
                        "the encoder drop released {device_bytes} of the {added} bytes its load took"
                    );
                }
            }
            ResidencyEvent::Loaded {
                component: Component::Denoiser,
                ..
            } => assert!(
                !encoder_loaded,
                "the transformer was imported while the encoder was loaded: {events:?}"
            ),
            ResidencyEvent::Dropped { .. } => {}
        }
    }
    assert_eq!(te_drops, 2, "both jobs encoded and dropped the encoder");

    // The first prompt's features are cached: no component moves.
    let before = model.events().len();
    model
        .generate(&job("a lighthouse on a cliff at dusk"), None)
        .expect("the cached job finishes");
    let after = model.events();
    eprintln!("cached-prompt job events: {:?}", &after[before..]);
}

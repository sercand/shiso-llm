//! Swap residency on the device: the encoder's bytes leave before the
//! transformer's arrive, back-to-back jobs both finish, a cached prompt
//! runs with no swap, and no path - a cached prompt after a resident job,
//! a failed encode - imports the transformer beside the encoder.
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

/// The Turbo install in forced swap mode, or `None` (with the reason on
/// stderr) when the device or the model is missing.
fn open_swap() -> Option<QwenImage21Model> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!("note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the swap case");
        return None;
    };
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the swap case");
        return None;
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
    Some(model)
}

/// Walks the log: the transformer is never imported while the encoder is
/// loaded, and each encoder drop gives back what its load took. Returns
/// the encoder drops from `from` on.
fn audit(events: &[ResidencyEvent], from: usize) -> usize {
    let te_load = events.iter().find_map(|e| match e {
        ResidencyEvent::Loaded {
            component: Component::Encoder,
            device_bytes,
        } => Some(*device_bytes),
        _ => None,
    });
    let mut encoder_loaded = false;
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
                if index >= from {
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
    te_drops
}

#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn swap_mode_releases_the_encoder_before_importing_the_transformer() {
    let Some(model) = open_swap() else {
        return;
    };
    let at_load = model.events().len();
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
    assert_eq!(
        audit(&events, at_load),
        2,
        "both jobs encoded and dropped the encoder"
    );

    // The first prompt's features are cached and the transformer was kept:
    // no component moves.
    let before = model.events().len();
    model
        .generate(&job("a lighthouse on a cliff at dusk"), None)
        .expect("the cached job finishes");
    let after = model.events();
    assert_eq!(
        after.len(),
        before,
        "a cached prompt moved components: {:?}",
        &after[before..]
    );
}

#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn no_path_imports_the_transformer_beside_the_encoder() {
    let Some(model) = open_swap() else {
        return;
    };
    let cached = "a lighthouse on a cliff at dusk";
    model
        .generate(&job(cached), None)
        .expect("the first job finishes");

    // (b) An encode that fails after the encoder is imported, then a job
    // whose prompt is cached: the failed job must not leave the encoder
    // where the cached job's transformer import lands.
    let error = model
        .failing_encode("a red fox in snow")
        .expect_err("a vision placeholder id is refused");
    eprintln!("failed encode: {error}");
    let events = model.events();
    assert!(
        matches!(
            events.last(),
            Some(ResidencyEvent::Dropped {
                component: Component::Encoder,
                ..
            })
        ),
        "the failed encode left the encoder loaded: {events:?}"
    );
    model
        .generate(&job(cached), None)
        .expect("the cached job after a failure finishes");
    audit(&model.events(), 0);

    // (a) The encoder loaded and the transformer not, as a resident job on
    // a larger host leaves a model that is then re-planned to swap, then a
    // cached prompt: the encoder must go before the transformer comes.
    model.stage_encoder_loaded().expect("the encoder loads");
    let before = model.events().len();
    model
        .generate(&job(cached), None)
        .expect("the cached job after a resident state finishes");
    let events = model.events();
    eprintln!("after the staged resident state: {:?}", &events[before..]);
    audit(&events, 0);
    assert!(
        matches!(
            events[before..],
            [
                ResidencyEvent::Dropped {
                    component: Component::Encoder,
                    ..
                },
                ResidencyEvent::Loaded {
                    component: Component::Denoiser,
                    ..
                }
            ]
        ),
        "a cached prompt from the resident state must drop the encoder, then import the \
         transformer: {:?}",
        &events[before..]
    );
}

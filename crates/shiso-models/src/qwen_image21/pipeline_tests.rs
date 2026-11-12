//! Swap residency on the device: the encoder's bytes leave before the
//! transformer's arrive, back-to-back jobs both finish, a cached prompt
//! runs with no swap, and no path - a cached prompt after a resident job,
//! a failed encode - imports the transformer beside the encoder. An edit
//! adds its two picture readers (the vision tower and the VAE encoder),
//! which must also be gone before the transformer arrives.
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
use shiso_vision::generator::{
    ImageGenError, ImageGenerationJob, ImageGenerator, ImageOutputFormat, InputReferenceImage,
};

use super::pipeline::{Component, QwenImage21Model, ResidencyEvent};
use crate::diffusion::weights::safetensors_files;
use crate::image_registry::{DEFAULT_IMAGE_RESERVE_BYTES, ImageModelDesc, ImageResidency};

/// A 256x256 job at the model's own step count (a distilled checkpoint
/// runs one schedule).
fn job_for(model: &QwenImage21Model, prompt: &str) -> ImageGenerationJob {
    ImageGenerationJob {
        prompt: prompt.to_string(),
        width: 256,
        height: 256,
        steps: model.limits().default_steps,
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
            model_path: safetensors_files(&image.join("diffusion_models"))
                .expect("a transformer file")
                .remove(0),
            text_encoder_path: Some(image.join("text_encoders")),
            vae_path: Some(image.join("vae")),
            tokenizer_path: None,
            assets_dir: Some(root.join("image_assets")),
            residency: ImageResidency::Swap,
            reserve_bytes: DEFAULT_IMAGE_RESERVE_BYTES,
            draft_steps: 0,
            final_reuse: None,
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
    let mut readers_loaded = 0usize;
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
                !encoder_loaded && readers_loaded == 0,
                "the transformer was imported while the encoder or an edit's picture reader \
                 was loaded: {events:?}"
            ),
            ResidencyEvent::Loaded {
                component: Component::Tower | Component::VaeEncoder,
                ..
            } => readers_loaded += 1,
            ResidencyEvent::Dropped {
                component: Component::Tower | Component::VaeEncoder,
                ..
            } => readers_loaded -= 1,
            ResidencyEvent::Dropped {
                component: Component::Denoiser,
                ..
            } => {}
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
            .generate(&job_for(&model, prompt), None)
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
        .generate(&job_for(&model, "a lighthouse on a cliff at dusk"), None)
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
        .generate(&job_for(&model, cached), None)
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
        .generate(&job_for(&model, cached), None)
        .expect("the cached job after a failure finishes");
    audit(&model.events(), 0);

    // (c) Both loaded, as a job planned resident leaves them, then a
    // cached prompt planned to swap: the transformer is already there, so
    // nothing is imported - and the encoder must still go, or the job
    // denoises beside the component its plan counted as dropped.
    model.stage_both_loaded().expect("both load");
    let before = model.events().len();
    model
        .generate(&job_for(&model, cached), None)
        .expect("the cached job from the resident state finishes");
    let events = model.events();
    eprintln!("after both were loaded: {:?}", &events[before..]);
    assert!(
        matches!(
            events[before..],
            [ResidencyEvent::Dropped {
                component: Component::Encoder,
                ..
            }]
        ),
        "a swap-planned job must drop the encoder even when the transformer is already \
         loaded: {:?}",
        &events[before..]
    );

    // (a) The encoder loaded and the transformer not, as a resident job on
    // a larger host leaves a model that is then re-planned to swap, then a
    // cached prompt: the encoder must go before the transformer comes.
    model.stage_encoder_loaded().expect("the encoder loads");
    let before = model.events().len();
    model
        .generate(&job_for(&model, cached), None)
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

/// A 256x256 opaque gradient with a darker square, as PNG bytes.
fn reference_png() -> InputReferenceImage {
    let picture = image::RgbaImage::from_fn(256, 256, |x, y| {
        let inside = (96..160).contains(&x) && (96..160).contains(&y);
        let v = |c: u32| u8::try_from((x + 2 * y + 40 * c) % 256).expect("a byte");
        if inside {
            image::Rgba([30, 30, 60, 255])
        } else {
            image::Rgba([v(0), v(1), v(2), 255])
        }
    });
    let mut bytes = Vec::new();
    picture
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("encodes");
    InputReferenceImage { bytes }
}

fn edit_job(model: &QwenImage21Model, prompt: &str, references: usize) -> ImageGenerationJob {
    ImageGenerationJob {
        input_references: vec![reference_png(); references],
        ..job_for(model, prompt)
    }
}

fn loads(events: &[ResidencyEvent], component: Component) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, ResidencyEvent::Loaded { component: c, .. } if *c == component))
        .count()
}

/// An edit in swap mode: the VAE encoder and the vision tower are loaded
/// for the encode and gone before the transformer is imported; the same
/// edit again needs no encoder at all; the same words without the picture
/// are another conditioning and are encoded; too many references are
/// refused before anything moves.
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn an_edit_drops_its_picture_readers_before_the_transformer() {
    let Some(model) = open_swap() else {
        return;
    };
    assert_eq!(model.max_references(), 10);
    let prompt = "make the square bright red";
    let at_load = model.events().len();
    let began = Instant::now();
    let image = model
        .generate(&edit_job(&model, prompt, 1), None)
        .expect("the edit finishes");
    eprintln!("edit: {:.2} s", began.elapsed().as_secs_f64());
    assert_eq!((image.width, image.height), (256, 256));
    let decoded = image::load_from_memory(&image.bytes).expect("the edit is a picture");
    assert_eq!((decoded.width(), decoded.height()), (256, 256));
    let events = model.events();
    let job_events = &events[at_load..];
    eprintln!("edit events: {job_events:?}");
    audit(&events, 0);
    assert_eq!(loads(job_events, Component::VaeEncoder), 1);
    assert_eq!(loads(job_events, Component::Tower), 1);
    // Each picture reader's drop is measured on the device and gives back
    // what its load took: the bytes really left before the transformer.
    for reader in [Component::VaeEncoder, Component::Tower] {
        let bytes = |loaded: bool| {
            job_events
                .iter()
                .find_map(|e| match *e {
                    ResidencyEvent::Loaded {
                        component,
                        device_bytes,
                    } if loaded && component == reader => Some(device_bytes),
                    ResidencyEvent::Dropped {
                        component,
                        device_bytes,
                    } if !loaded && component == reader => Some(device_bytes),
                    _ => None,
                })
                .expect("the reader was loaded and dropped")
        };
        let (took, gave) = (bytes(true), bytes(false));
        eprintln!("{reader:?}: load took {took} bytes, drop released {gave}");
        assert!(
            took > 0 && gave * 10 >= took * 9,
            "{reader:?}: the drop released {gave} of the {took} bytes its load took"
        );
    }
    assert_eq!(loads(job_events, Component::Denoiser), 1);

    // The same edit again: its conditioning is cached, no component moves.
    let before = model.events().len();
    model
        .generate(&edit_job(&model, prompt, 1), None)
        .expect("the cached edit finishes");
    let after = model.events();
    assert_eq!(
        after.len(),
        before,
        "a cached edit moved components: {:?}",
        &after[before..]
    );

    // The same words over no picture are a text job: encoded on its own,
    // never answered from the edit's features, and with no picture reader.
    let before = model.events().len();
    model
        .generate(&job_for(&model, prompt), None)
        .expect("the text job finishes");
    let events = model.events();
    let text_events = &events[before..];
    assert_eq!(loads(text_events, Component::Encoder), 1, "{text_events:?}");
    assert_eq!(loads(text_events, Component::Tower), 0);
    assert_eq!(loads(text_events, Component::VaeEncoder), 0);
    audit(&events, 0);

    // Eleven references: refused by count, with nothing loaded or dropped.
    let before = model.events().len();
    match model.generate(&edit_job(&model, prompt, 11), None) {
        Err(ImageGenError::ReferenceImage(message)) => {
            assert!(message.contains("11 reference images"), "{message}");
        }
        other => panic!("expected the count refusal, got {other:?}"),
    }
    assert_eq!(model.events().len(), before);
}

/// More references than the host's memory holds are refused by the planner
/// with the need, the budget and the count, before anything is loaded.
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn an_edit_over_the_memory_budget_is_refused() {
    let Some(model) = open_swap() else {
        return;
    };
    let before = model.events().len();
    // Ten 1024x1024-conditioned references: about 22 GB of prefix cache.
    match model.generate(&edit_job(&model, "combine them", 10), None) {
        Err(ImageGenError::ReferenceImage(message)) => {
            eprintln!("refusal: {message}");
            assert!(message.contains("10 reference images"), "{message}");
            assert!(message.contains("MiB"), "{message}");
        }
        other => panic!("expected a budget refusal, got {other:?}"),
    }
    assert_eq!(model.events().len(), before);
}

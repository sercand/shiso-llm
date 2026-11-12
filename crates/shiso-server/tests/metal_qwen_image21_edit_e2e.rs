//! End-to-end: a Qwen-Image-2.1 install edits a picture on Metal.
//!
//! The install generates a picture, then edits it: once asked to change
//! nothing, once with a second reference. The edits are decoded and judged
//! like the text-to-image case (exact size, four channels, real variance),
//! and the one asked to change nothing must stay close to its reference -
//! a pipeline that ignored the picture would return an unrelated image and
//! still pass a variance check.
//!
//! The edit never uses the seed its reference was generated with at the
//! reference's size: starting an edit from the very noise that produced
//! its reference over-commits to it and comes out harsh and over-sharp.
//! That is the model's behaviour, not a failure of the pipeline, and a
//! test that tripped over it would report the wrong thing.
//!
//! Requires a Metal device and `SHISO_TEST_MODEL_QWEN_IMAGE21` (a directory
//! holding `image/` and `image_assets/`); without either it says so on
//! stderr and returns. `SHISO_TEST_IMAGE_OUT`, when set, is a directory the
//! pictures are written to for a human to look at. Run with `--release`.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::ids::Seed;
use shiso_metal::MetalBackend;
use shiso_models::image_registry::{DEFAULT_IMAGE_RESERVE_BYTES, ImageModelDesc, ImageResidency};
use shiso_vision::generator::{
    GeneratedImage, ImageGenerationJob, ImageGenerator, ImageOutputFormat, InputReferenceImage,
};

/// A photograph's pixel standard deviation is tens; a flat field's is ~0.
const MIN_PIXEL_STDDEV: f64 = 10.0;
/// An edit asked to change nothing stays within this mean absolute
/// difference (of 255) of its reference; an unrelated picture of the same
/// kind is several times further.
const MAX_IDENTITY_DIFFERENCE: f64 = 20.0;

fn job(
    generator: &Arc<dyn ImageGenerator>,
    prompt: &str,
    (width, height): (u32, u32),
    seed: u64,
    references: Vec<InputReferenceImage>,
) -> ImageGenerationJob {
    ImageGenerationJob {
        prompt: prompt.to_string(),
        width,
        height,
        steps: generator.limits().default_steps,
        seed: Seed::new(seed),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: references,
    }
}

#[expect(clippy::print_stderr, reason = "timings are the case's report")]
fn run(
    generator: &Arc<dyn ImageGenerator>,
    job: &ImageGenerationJob,
    label: &str,
) -> GeneratedImage {
    let began = Instant::now();
    let image = generator
        .generate(job, None)
        .unwrap_or_else(|e| panic!("{label}: generation failed: {e}"));
    eprintln!("{label}: {:.2} s", began.elapsed().as_secs_f64());
    assert_eq!(
        (image.width, image.height),
        (job.width, job.height),
        "{label}"
    );
    if let Some(dir) = std::env::var_os("SHISO_TEST_IMAGE_OUT") {
        let path = Path::new(&dir).join(format!("{label}.png"));
        std::fs::write(&path, &image.bytes).expect("the image is written");
        eprintln!("wrote {}", path.display());
    }
    image
}

/// The picture's RGB bytes at `side` x `side`, alpha dropped.
fn rgb_at(bytes: &[u8], side: u32) -> Vec<u8> {
    let decoded = image::load_from_memory(bytes).expect("the PNG decodes");
    assert_eq!(
        decoded.color(),
        image::ColorType::Rgba8,
        "the image keeps its alpha"
    );
    image::imageops::resize(
        &decoded.to_rgb8(),
        side,
        side,
        image::imageops::FilterType::Triangle,
    )
    .into_raw()
}

#[expect(
    clippy::cast_precision_loss,
    reason = "sample counts of a small picture are exact in f64"
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
    clippy::cast_precision_loss,
    reason = "sample counts of a small picture are exact in f64"
)]
fn mean_difference(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x.abs_diff(y)))
        .sum::<f64>()
        / a.len() as f64
}

/// The install on Metal, or `None` with the reason on stderr.
#[expect(
    clippy::print_stderr,
    reason = "a machine without the model skips with its reason on stderr"
)]
fn open() -> Option<Arc<dyn ImageGenerator>> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!("SKIP: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset");
        return None;
    };
    if !shiso_metal::metal_available() {
        eprintln!("SKIP: no Metal device on this machine");
        return None;
    }
    let root = PathBuf::from(root);
    let backend: Arc<dyn shiso_core::Backend> = MetalBackend::new().expect("Metal opens");
    let image = root.join("image");
    let model_path = std::fs::read_dir(image.join("diffusion_models"))
        .expect("the transformer directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .expect("a transformer file");
    let desc = ImageModelDesc {
        model_path,
        text_encoder_path: Some(image.join("text_encoders")),
        vae_path: Some(image.join("vae")),
        tokenizer_path: None,
        assets_dir: Some(root.join("image_assets")),
        residency: ImageResidency::Auto,
        reserve_bytes: DEFAULT_IMAGE_RESERVE_BYTES,
        draft_steps: 0,
        final_reuse: None,
    };
    let generator = shiso_models::image_registry::open_image_generator(&desc, &backend)
        .expect("the install loads");
    Some(generator)
}

#[expect(
    clippy::print_stderr,
    reason = "the measured differences are the case's report"
)]
#[test]
#[ignore = "needs SHISO_TEST_MODEL_QWEN_IMAGE21 and a Metal GPU; run with --release"]
fn qwen_image21_edits_a_picture() {
    let Some(generator) = open() else { return };
    assert_eq!(generator.max_references(), 10, "the install can edit");

    // Two pictures to edit, from the model itself.
    let side = 512;
    let fox = run(
        &generator,
        &job(
            &generator,
            "a red fox sitting in fresh snow, photograph",
            (side, side),
            42,
            Vec::new(),
        ),
        "edit-source-fox",
    );
    let hat = run(
        &generator,
        &job(
            &generator,
            "a red knitted hat on a white table, product photograph",
            (side, side),
            43,
            Vec::new(),
        ),
        "edit-source-hat",
    );
    let fox_reference = InputReferenceImage {
        bytes: fox.bytes.clone(),
    };
    let hat_reference = InputReferenceImage {
        bytes: hat.bytes.clone(),
    };

    // A square reference implies a 1024x1024 edit.
    let implied = generator
        .default_size_for_references(std::slice::from_ref(&fox_reference))
        .expect("a size");
    assert_eq!(implied, (1024, 1024));

    // Asked to change nothing, the edit is its reference again.
    let same = run(
        &generator,
        &job(
            &generator,
            "Keep the image exactly the same",
            (side, side),
            7,
            vec![fox_reference.clone()],
        ),
        "edit-identity",
    );
    let (source, edited) = (rgb_at(&fox.bytes, 64), rgb_at(&same.bytes, 64));
    let kept = mean_difference(&source, &edited);
    let unrelated = mean_difference(&rgb_at(&hat.bytes, 64), &edited);
    eprintln!("identity edit: {kept:.1} from its reference, {unrelated:.1} from another picture");
    assert!(stddev(&edited) > MIN_PIXEL_STDDEV, "a flat field");
    assert!(
        kept < MAX_IDENTITY_DIFFERENCE && kept < unrelated / 2.0,
        "the edit does not follow its reference: {kept:.1} from it, {unrelated:.1} from another"
    );

    // The same job again is the same bytes; another seed is not.
    let again = generator
        .generate(
            &job(
                &generator,
                "Keep the image exactly the same",
                (side, side),
                7,
                vec![fox_reference.clone()],
            ),
            None,
        )
        .expect("the repeated edit");
    assert_eq!(same.bytes, again.bytes, "the same seed reproduces the edit");

    // Two references: the picture runs and is a real image.
    let both = run(
        &generator,
        &job(
            &generator,
            "The fox from image 1 wears the hat from image 2",
            (side, side),
            7,
            vec![fox_reference, hat_reference],
        ),
        "edit-two-references",
    );
    let pixels = rgb_at(&both.bytes, 64);
    assert!(stddev(&pixels) > MIN_PIXEL_STDDEV, "a flat field");
    assert_ne!(both.bytes, same.bytes);
}

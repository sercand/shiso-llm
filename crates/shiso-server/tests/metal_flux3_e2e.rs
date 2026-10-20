//! End-to-end: a real FLUX.3 Klein checkpoint loads on Metal and a real
//! picture comes out, text to image and edited from a reference.
//!
//! Every model path is read from the environment, and a case without them
//! skips with its reason on stderr - which looks like a pass, so check
//! stderr before reporting one:
//!
//! - `SHISO_TEST_MODEL_FLUX3`: the transformer (`.gguf` or `.safetensors`);
//! - `SHISO_TEST_MODEL_FLUX3_TEXT`: the text encoder (a directory of shards
//!   with its `config.json`, or one file with `config.json` beside it);
//! - `SHISO_TEST_MODEL_FLUX3_VAE`: `ae.safetensors`;
//! - `SHISO_TEST_TOKENIZER_FLUX3` (optional): the tokenizer directory, else
//!   the locator's usual places;
//! - `SHISO_TEST_FLUX3_RESIDENCY` (optional): `auto` (default), `resident` or
//!   `swap`, and `SHISO_TEST_FLUX3_RESERVE_GB` (optional, default the
//!   server's): the same knobs `--image-residency` and `--image-reserve-gb`
//!   set, so a host that holds everything can still run the swap;
//! - `RUST_LOG` (optional): `shiso_models=info` prints the residency events
//!   and phase timings on stderr;
//! - `SHISO_TEST_OUTPUT_DIR` (optional): where the cases save the PNGs they
//!   decoded, for a person to look at.
//!
//! A flat field encodes to a perfectly valid PNG, so the cases read the
//! encoded bytes back: the decoded size must be exactly the requested one and
//! the pixels must have real variance. That grades "a denoised picture, not a
//! blank", not "the right picture"; look at the saved files for that.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use image::{ImageFormat, Rgb, RgbImage};
use shiso_core::Backend;
use shiso_core::ids::Seed;
use shiso_metal::MetalBackend;
use shiso_models::image_registry::{
    DEFAULT_IMAGE_RESERVE_BYTES, ImageModelDesc, ImageResidency, open_image_generator,
};
use shiso_vision::generator::{
    GeneratedImage, ImageGenerationJob, ImageGenerator, ImageOutputFormat, InputReferenceImage,
};

/// Decoded-pixel standard deviation a real picture clears by a wide margin
/// and a flat (or NaN-clamped-to-black) field never does.
const MIN_PIXEL_STDDEV: f64 = 8.0;

/// The model paths, or why the case cannot run.
struct Models {
    model: PathBuf,
    text: PathBuf,
    vae: PathBuf,
    tokenizer: Option<PathBuf>,
}

#[expect(
    clippy::print_stderr,
    reason = "a machine without the test models skips with its reason on stderr"
)]
fn models() -> Option<Models> {
    let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
    let (Some(model), Some(text), Some(vae)) = (
        var("SHISO_TEST_MODEL_FLUX3"),
        var("SHISO_TEST_MODEL_FLUX3_TEXT"),
        var("SHISO_TEST_MODEL_FLUX3_VAE"),
    ) else {
        eprintln!(
            "SKIP: SHISO_TEST_MODEL_FLUX3, SHISO_TEST_MODEL_FLUX3_TEXT and \
             SHISO_TEST_MODEL_FLUX3_VAE must all be set; the end-to-end case NOT RUN"
        );
        return None;
    };
    for (name, path) in [
        ("transformer", &model),
        ("text encoder", &text),
        ("VAE", &vae),
    ] {
        if !path.exists() {
            eprintln!(
                "SKIP: the {name} path {} does not exist; the end-to-end case NOT RUN",
                path.display()
            );
            return None;
        }
    }
    Some(Models {
        model,
        text,
        vae,
        tokenizer: var("SHISO_TEST_TOKENIZER_FLUX3"),
    })
}

#[expect(
    clippy::print_stderr,
    reason = "a machine without a Metal device skips with its reason on stderr"
)]
fn backend() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("SKIP: no Metal device available on this machine; the end-to-end case NOT RUN");
        return None;
    }
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("Metal backend opens");
    Some(backend)
}

/// `SHISO_TEST_FLUX3_RESIDENCY`; an unknown word is a mistake, not a default.
fn residency_knob() -> ImageResidency {
    match std::env::var("SHISO_TEST_FLUX3_RESIDENCY").as_deref() {
        Err(_) | Ok("auto") => ImageResidency::Auto,
        Ok("resident") => ImageResidency::Resident,
        Ok("swap") => ImageResidency::Swap,
        Ok(other) => panic!("SHISO_TEST_FLUX3_RESIDENCY={other}: expected auto, resident or swap"),
    }
}

/// `SHISO_TEST_FLUX3_RESERVE_GB` in bytes, else the server's default.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a test knob in GiB, checked non-negative; the product is far below 2^53"
)]
fn reserve_knob() -> u64 {
    match std::env::var("SHISO_TEST_FLUX3_RESERVE_GB") {
        Err(_) => DEFAULT_IMAGE_RESERVE_BYTES,
        Ok(text) => {
            let gb: f64 = text
                .parse()
                .unwrap_or_else(|_| panic!("SHISO_TEST_FLUX3_RESERVE_GB={text} is not a number"));
            assert!(gb >= 0.0 && gb.is_finite(), "reserve must be >= 0 GiB");
            (gb * (1u64 << 30) as f64) as u64
        }
    }
}

#[expect(
    clippy::print_stderr,
    reason = "the case reports its load time on stderr under --nocapture"
)]
fn open(models: &Models, backend: &Arc<dyn Backend>) -> Arc<dyn ImageGenerator> {
    let desc = ImageModelDesc {
        model_path: models.model.clone(),
        text_encoder_path: Some(models.text.clone()),
        vae_path: Some(models.vae.clone()),
        tokenizer_path: models.tokenizer.clone(),
        assets_dir: None,
        residency: residency_knob(),
        reserve_bytes: reserve_knob(),
        draft_steps: 0,
        final_reuse: None,
    };
    if std::env::var_os("RUST_LOG").is_some() {
        // The residency events and phase timings are `tracing` records.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .try_init();
    }
    let began = Instant::now();
    let generator = open_image_generator(&desc, backend)
        .unwrap_or_else(|e| panic!("the FLUX.3 checkpoint did not load: {e}"));
    eprintln!("load took {:.1}s", began.elapsed().as_secs_f64());
    generator
}

/// Reads width and height from a PNG IHDR chunk.
fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
    assert_eq!(
        &bytes[..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
        "output is not a PNG"
    );
    assert_eq!(&bytes[12..16], b"IHDR", "PNG has no IHDR at the front");
    let field = |at: usize| {
        u32::from_be_bytes(
            bytes[at..at + 4]
                .try_into()
                .expect("IHDR fields are four bytes"),
        )
    };
    (field(16), field(20))
}

/// Decodes the encoded bytes, checks the size and that the pixels vary, and
/// saves the PNG under `SHISO_TEST_OUTPUT_DIR` when it is set.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel counts are bounded by the 4096x4096 generation contract, exact in f64"
)]
#[expect(
    clippy::print_stderr,
    reason = "the case reports pixel statistics and the saved path on stderr under --nocapture"
)]
fn assert_real_image(result: &GeneratedImage, width: u32, height: u32, label: &str) {
    assert_eq!(result.media_type, "image/png", "{label}: media type");
    assert_eq!(
        png_dimensions(&result.bytes),
        (width, height),
        "{label}: PNG IHDR disagrees with the requested size"
    );
    assert_eq!((result.width, result.height), (width, height), "{label}");
    let decoded = image::load_from_memory(&result.bytes)
        .unwrap_or_else(|e| panic!("{label}: encoded bytes do not decode: {e}"));
    assert_eq!(
        (decoded.width(), decoded.height()),
        (width, height),
        "{label}"
    );
    let rgb = decoded.to_rgb8();
    let pixels = rgb.as_raw();
    let n = pixels.len() as f64;
    let mean = pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let stddev = (pixels
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    eprintln!("{label}: {width}x{height} pixels mean {mean:.1} stddev {stddev:.1}");
    if let Some(dir) = std::env::var_os("SHISO_TEST_OUTPUT_DIR").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("creates the output directory");
        let path = dir.join(format!("{label}.png"));
        std::fs::write(&path, &result.bytes).expect("saves the picture");
        eprintln!("{label}: saved {}", path.display());
    }
    assert!(
        stddev > MIN_PIXEL_STDDEV,
        "{label}: pixel stddev {stddev:.3} is not above {MIN_PIXEL_STDDEV}: the pipeline emitted \
         a flat field, not a denoised picture"
    );
}

fn job(
    prompt: &str,
    width: u32,
    height: u32,
    references: Vec<InputReferenceImage>,
) -> ImageGenerationJob {
    ImageGenerationJob {
        prompt: prompt.to_owned(),
        width,
        height,
        steps: 16,
        seed: Seed::new(42),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: references,
    }
}

/// The highest device allocation the backend reported while it ran,
/// sampled every 50 ms, for the swap-versus-resident comparison.
struct DevicePeak {
    stop: Arc<std::sync::atomic::AtomicBool>,
    handle: std::thread::JoinHandle<u64>,
}

impl DevicePeak {
    fn start(backend: &Arc<dyn Backend>) -> Self {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (flag, backend) = (Arc::clone(&stop), Arc::clone(backend));
        let handle = std::thread::spawn(move || {
            let mut peak = 0;
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(info) = backend.memory_info() {
                    peak = peak.max(info.total.saturating_sub(info.free));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            peak
        });
        Self { stop, handle }
    }

    fn finish(self) -> u64 {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.handle.join().expect("the sampler does not panic")
    }
}

#[expect(
    clippy::cast_precision_loss,
    clippy::print_stderr,
    reason = "the case reports its generation time and device peak on stderr under --nocapture"
)]
fn run(
    backend: &Arc<dyn Backend>,
    generator: &Arc<dyn ImageGenerator>,
    job: &ImageGenerationJob,
    label: &str,
) -> GeneratedImage {
    let began = Instant::now();
    let peak = DevicePeak::start(backend);
    let last = std::cell::RefCell::new(began);
    let report = |event: shiso_vision::generator::ImageStepEvent| {
        let now = Instant::now();
        let step = (now - *last.borrow()).as_secs_f64();
        last.replace(now);
        eprintln!(
            "{label}: step {}/{} done (+{step:.1}s)",
            event.step + 1,
            event.total_steps
        );
    };
    let result = generator
        .generate(job, Some(&report))
        .unwrap_or_else(|e| panic!("{label}: generation failed: {e}"));
    eprintln!(
        "{label}: generation took {:.1}s",
        began.elapsed().as_secs_f64()
    );
    eprintln!(
        "{label}: peak device allocation {:.0} MB",
        peak.finish() as f64 / 1e6
    );
    result
}

#[test]
#[ignore = "needs the FLUX.3 files (SHISO_TEST_MODEL_FLUX3 and friends) and a Metal GPU; run with --release"]
fn flux3_generates_1024_png_for_prompt() {
    let (Some(backend), Some(models)) = (backend(), models()) else {
        return;
    };
    let generator = open(&models, &backend);
    assert_eq!(generator.model_id(), "flux-3-klein");
    let result = run(
        &backend,
        &generator,
        &job(
            "a red panda wearing an astronaut helmet, floating in space, photograph",
            1024,
            1024,
            Vec::new(),
        ),
        "flux3-t2i-1024",
    );
    assert_real_image(&result, 1024, 1024, "flux3-t2i-1024");
}

/// A deterministic picture with structure (a gradient, a disc and stripes),
/// 1000x750: a size off the 16-pixel grid, so the edit pads it onto a
/// 1008x752 canvas and the output must be cut back to the reference's own size.
fn reference_picture() -> InputReferenceImage {
    let (w, h) = (1000u32, 750u32);
    let img = RgbImage::from_fn(w, h, |x, y| {
        let (dx, dy) = (f64::from(x) - 500.0, f64::from(y) - 375.0);
        let in_disc = dx * dx + dy * dy < 150.0 * 150.0;
        let stripe = (x / 40) % 2 == 0;
        let channel = |base: u32, disc: u32| -> u8 {
            let v = if in_disc {
                disc
            } else {
                base + if stripe { 30 } else { 0 }
            };
            u8::try_from(v.min(255)).expect("clamped to a byte")
        };
        Rgb([
            channel(x * 160 / w, 230),
            channel(y * 160 / h, 60),
            channel(120, 40),
        ])
    });
    let mut bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut bytes), ImageFormat::Png)
        .expect("encodes the reference");
    InputReferenceImage { bytes }
}

#[test]
#[ignore = "needs the FLUX.3 files (SHISO_TEST_MODEL_FLUX3 and friends) and a Metal GPU; run with --release"]
fn flux3_edit_with_reference_keeps_size() {
    let (Some(backend), Some(models)) = (backend(), models()) else {
        return;
    };
    let generator = open(&models, &backend);
    let reference = reference_picture();
    // An omitted size becomes the reference's canvas; the picture then comes
    // back at the reference's own 1000x750.
    let canvas = shiso_models::flux3::default_size_for_references(std::slice::from_ref(&reference))
        .expect("the reference prepares");
    assert_eq!(canvas, (1008, 752));
    let result = run(
        &backend,
        &generator,
        &job(
            "the same scene as an oil painting, thick brush strokes",
            canvas.0,
            canvas.1,
            vec![reference],
        ),
        "flux3-edit-1000x750",
    );
    assert_real_image(&result, 1000, 750, "flux3-edit-1000x750");
}

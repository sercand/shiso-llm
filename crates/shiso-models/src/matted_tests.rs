//! The background post-processing step over a fake generator and a fake
//! remover: which alpha each background answers with, when the remover runs,
//! and that a failure fails the job instead of answering an opaque picture.

use std::sync::{Arc, Mutex};

use image::ImageEncoder as _;
use shiso_core::ids::Seed;
use shiso_vision::generator::{
    Background, GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits,
    ImageOutputFormat, ImageStepEvent,
};
use shiso_vision::matte::BackgroundRemover;

use super::matted::MattedGenerator;

const SIDE: u32 = 4;
const PIXELS: usize = (SIDE * SIDE) as usize;

/// A purple background (the colour the model leaves under its transparent
/// pixels) around an opaque red centre: the left half at alpha 0.
fn cut_out() -> Vec<u8> {
    let mut out = Vec::with_capacity(4 * PIXELS);
    for i in 0..PIXELS {
        if i % 4 < 2 {
            out.extend_from_slice(&[138, 55, 178, 0]);
        } else {
            out.extend_from_slice(&[200, 10, 10, 255]);
        }
    }
    out
}

/// An opaque picture with a few levels of noise below 255.
fn near_opaque() -> Vec<u8> {
    (0..PIXELS)
        .flat_map(|i| [40, 80, 120, 250 + u8::try_from(i % 6).expect("small")])
        .collect()
}

fn png(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(rgba, SIDE, SIDE, image::ExtendedColorType::Rgba8)
        .expect("encode");
    out
}

#[derive(Debug)]
struct FakeGen {
    rgba: Vec<u8>,
    asked: Mutex<Vec<ImageOutputFormat>>,
}

impl FakeGen {
    fn new(rgba: Vec<u8>) -> Arc<Self> {
        Arc::new(Self {
            rgba,
            asked: Mutex::new(Vec::new()),
        })
    }
}

impl ImageGenerator for FakeGen {
    fn model_id(&self) -> &'static str {
        "fake"
    }
    fn limits(&self) -> ImageLimits {
        ImageLimits {
            size_multiple: 1,
            min_side: 1,
            max_side: 64,
            default_size: (SIDE, SIDE),
            default_steps: 1,
            fixed_steps: None,
        }
    }
    fn generate(
        &self,
        job: &ImageGenerationJob,
        _progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        self.asked.lock().expect("lock").push(job.output_format);
        Ok(GeneratedImage {
            bytes: png(&self.rgba),
            media_type: "image/png",
            width: SIDE,
            height: SIDE,
            prompt_tokens: 3,
            image_input_tokens: 0,
            completion_tokens: 16,
            transparent: false,
        })
    }
}

#[derive(Debug)]
struct FakeRemover {
    fail: bool,
    seen: Mutex<Option<Vec<u8>>>,
}

impl FakeRemover {
    fn new(fail: bool) -> Arc<Self> {
        Arc::new(Self {
            fail,
            seen: Mutex::new(None),
        })
    }
}

/// The fake matte: the top half kept, the bottom half cut away.
fn fake_matte() -> Vec<u8> {
    (0..PIXELS)
        .map(|i| if i < PIXELS / 2 { 255 } else { 0 })
        .collect()
}

impl BackgroundRemover for FakeRemover {
    fn alpha(&self, rgb: &image::RgbImage) -> Result<Vec<u8>, String> {
        *self.seen.lock().expect("lock") = Some(rgb.as_raw().clone());
        if self.fail {
            return Err("the device went away".to_owned());
        }
        Ok(fake_matte())
    }
}

fn job(background: Background, output_format: ImageOutputFormat) -> ImageGenerationJob {
    ImageGenerationJob {
        prompt: "x".to_owned(),
        width: SIDE,
        height: SIDE,
        steps: 1,
        seed: Seed::new(1),
        output_format,
        output_compression: None,
        input_references: Vec::new(),
        background,
    }
}

fn run(
    source: Vec<u8>,
    remover: Option<Arc<FakeRemover>>,
    background: Background,
    format: ImageOutputFormat,
) -> Result<GeneratedImage, ImageGenError> {
    let remover: Option<Arc<dyn BackgroundRemover>> = match remover {
        Some(r) => Some(r),
        None => None,
    };
    let generator = MattedGenerator::new(FakeGen::new(source), remover);
    generator.generate(&job(background, format), None)
}

fn decode(image: &GeneratedImage) -> image::RgbaImage {
    image::load_from_memory(&image.bytes)
        .expect("decodes")
        .to_rgba8()
}

fn alphas(image: &GeneratedImage) -> Vec<u8> {
    decode(image).pixels().map(|p| p[3]).collect()
}

#[test]
fn auto_keeps_a_cut_out() {
    let out = run(cut_out(), None, Background::Auto, ImageOutputFormat::Png).expect("runs");
    let source: Vec<u8> = cut_out().chunks(4).map(|p| p[3]).collect();
    assert_eq!(alphas(&out), source);
    assert!(out.transparent);
    assert_eq!(out.media_type, "image/png");
}

#[test]
fn auto_clamps_near_opaque() {
    let out = run(
        near_opaque(),
        None,
        Background::Auto,
        ImageOutputFormat::Png,
    )
    .expect("runs");
    assert!(alphas(&out).iter().all(|&a| a == 255));
    assert!(!out.transparent);
}

#[test]
fn auto_jpeg_composites() {
    let out = run(cut_out(), None, Background::Auto, ImageOutputFormat::Jpeg).expect("runs");
    assert_eq!(out.media_type, "image/jpeg");
    assert!(!out.transparent);
    // The transparent half is white, not the purple underneath it.
    let pixel = *decode(&out).get_pixel(0, 0);
    assert!(
        pixel[0] > 240 && pixel[1] > 240 && pixel[2] > 240,
        "{pixel:?}"
    );
}

#[test]
fn opaque_composites_over_white() {
    let out = run(cut_out(), None, Background::Opaque, ImageOutputFormat::Png).expect("runs");
    let picture = decode(&out);
    assert!(picture.pixels().all(|p| p[3] == 255));
    assert_eq!(picture.get_pixel(0, 0).0, [255, 255, 255, 255]);
    assert_eq!(picture.get_pixel(3, 0).0, [200, 10, 10, 255]);
    assert!(!out.transparent);
}

#[test]
fn transparent_keeps_native_cut_out() {
    let remover = FakeRemover::new(false);
    let out = run(
        cut_out(),
        Some(Arc::clone(&remover)),
        Background::Transparent,
        ImageOutputFormat::Png,
    )
    .expect("runs");
    assert!(remover.seen.lock().expect("lock").is_none(), "remover ran");
    let source: Vec<u8> = cut_out().chunks(4).map(|p| p[3]).collect();
    assert_eq!(alphas(&out), source);
    assert!(out.transparent);
}

#[test]
fn transparent_mattes_an_opaque_picture() {
    let remover = FakeRemover::new(false);
    let out = run(
        near_opaque(),
        Some(Arc::clone(&remover)),
        Background::Transparent,
        ImageOutputFormat::Png,
    )
    .expect("runs");
    assert_eq!(alphas(&out), fake_matte());
    assert!(out.transparent);
    assert_eq!(out.media_type, "image/png");
    // The remover saw the picture composited over white, not raw colour.
    let seen = remover.seen.lock().expect("lock").clone().expect("ran");
    let expected: Vec<u8> = near_opaque()
        .chunks(4)
        .flat_map(|p| {
            let a = u32::from(p[3]);
            let over = |c: u8| {
                u8::try_from((u32::from(c) * a + 255 * (255 - a) + 127) / 255).expect("byte")
            };
            [over(p[0]), over(p[1]), over(p[2])]
        })
        .collect();
    assert_eq!(seen, expected);
}

#[test]
fn transparent_without_remover_is_refused() {
    let error = run(
        near_opaque(),
        None,
        Background::Transparent,
        ImageOutputFormat::Png,
    )
    .expect_err("refused");
    assert!(
        matches!(error, ImageGenError::NoBackgroundRemover),
        "{error:?}"
    );
    assert!(error.to_string().contains("--image-matte"), "{error}");
}

#[test]
fn remover_failure_fails_the_job() {
    let error = run(
        near_opaque(),
        Some(FakeRemover::new(true)),
        Background::Transparent,
        ImageOutputFormat::Png,
    )
    .expect_err("fails");
    assert!(
        matches!(error, ImageGenError::BackgroundRemoval(_)),
        "{error:?}"
    );
}

#[test]
fn inner_is_always_asked_for_png() {
    let inner = FakeGen::new(cut_out());
    let shared = Arc::clone(&inner);
    let shared: Arc<dyn ImageGenerator> = shared;
    let generator = MattedGenerator::new(shared, None);
    generator
        .generate(&job(Background::Opaque, ImageOutputFormat::Jpeg), None)
        .expect("runs");
    generator
        .generate(&job(Background::Auto, ImageOutputFormat::Jpeg), None)
        .expect("runs");
    assert_eq!(
        *inner.asked.lock().expect("lock"),
        vec![ImageOutputFormat::Png, ImageOutputFormat::Png]
    );
    assert!(!generator.has_remover());
}

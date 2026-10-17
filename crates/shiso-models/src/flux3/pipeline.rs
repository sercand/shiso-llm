//! The FLUX.3 Klein pipeline: a prompt (and optional reference pictures) in,
//! encoded image bytes out.
//!
//! One [`Flux3KleinModel`] holds the three components - the Qwen3-VL text
//! encoder, the diffusion transformer and the autoencoder - on one backend,
//! with the tokenizer and chat template that turn text into ids.
//! [`ImageGenerator::generate`] is four stages, each a method so a device
//! test can run one without the rest: [`Flux3KleinModel::encode_prompt`]
//! (padded, masked text conditioning), [`Flux3KleinModel::prepare_references`]
//! (VAE-encoded reference tokens), [`Flux3KleinModel::sample`] (the flow-map
//! Euler walk over `[image, image_cond, text]`) and
//! [`Flux3KleinModel::render`] (decode, crop, encode).
//!
//! # Sizes and crops
//!
//! The job's size is final: the server fills the model's default into every
//! request, so the pipeline cannot tell an omitted size from a requested one.
//! The reference implementation crops a single reference's output back to
//! the fitted box when the caller named no size; here that is the case where
//! the job's size equals the reference's canvas (what an omitted size
//! becomes once [`default_size`](super::reference::default_size) has run), so
//! an edit generated at the canvas size comes back at the reference's own
//! size, which need not sit on the 16-pixel grid.
//!
//! # Progress
//!
//! A progress event goes out after every step without a partial image: a
//! preview is a whole VAE decode, which per step would cost more than the
//! step.

use std::sync::Arc;
use std::time::Instant;

use shiso_core::Backend;
use shiso_core::ids::Seed;
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits,
    ImageOutputFormat, ImageStepEvent,
};

use super::config::Flux3Located;
use super::pack::{Pack, image_ids, reference_ids, text_ids};
use super::reference::{
    check_reference_count, check_reference_size, decode_reference, prepare_multi_reference,
    prepare_single_reference,
};
use super::sampler::{initial_noise, round_bf16, run_euler};
use super::scheduler::{ALPHA, DEFAULT_STEPS, schedule};
use super::text_encoder::{EncodedText, encode, load as load_text_encoder};
use super::transformer::{DiT, DiTInput};
use super::vae::{PIXELS_PER_TOKEN, Vae};
use crate::diffusion::encode::{Channels, encode_image};
use crate::diffusion::ops::Ops;
use crate::diffusion::prompt::{PromptTokenizer, load_chat_template};
use crate::diffusion::qwen3_text::TextEncoder;
use crate::diffusion::weights::StStore;
use crate::image_registry::{ImageModelDesc, ImageResidency};

/// Channels of one image token.
const IMG_CHANNELS: usize = 128;
/// Image sides are multiples of this.
const SIZE_MULTIPLE: u32 = 16;
/// The smallest side served.
const MIN_SIDE: u32 = 64;
/// The largest side served.
const MAX_SIDE: u32 = 4096;
/// The size a request that names none gets.
const DEFAULT_SIZE: (u32, u32) = (1024, 1024);

/// The FLUX.3 Klein generator: text encoder, transformer and autoencoder on
/// one device, driven one job at a time.
#[derive(Debug)]
pub(crate) struct Flux3KleinModel {
    model_id: String,
    pub(super) ops: Ops,
    pub(super) text: TextEncoder,
    pub(super) dit: DiT,
    pub(super) vae: Vae,
    /// The vocabulary; caller text spelling a control token is encoded as
    /// plain text so it cannot forge a marker.
    tokenizer: PromptTokenizer,
    chat_template: String,
}

/// The reference pictures of a job, encoded.
#[derive(Debug, Default)]
pub(crate) struct ReferenceTokens {
    /// Row-major `(rows, 128)` bf16-rounded latent tokens of every reference,
    /// in order.
    pub tokens: Vec<f32>,
    /// Rotary ids of those rows, four per row.
    pub ids: Vec<[f32; 4]>,
    /// Rows of `tokens`.
    pub rows: usize,
}

/// A job's pictures, prepared: the reference tokens and what the output is
/// cropped to.
#[derive(Debug)]
pub(crate) struct PreparedReferences {
    /// The encoded references (empty without any).
    pub tokens: ReferenceTokens,
    /// The canvas of a single reference, which a job of that size is
    /// generated at and cropped from.
    pub canvas: Option<(u32, u32)>,
    /// `[left, top, right, bottom]` of the fitted picture inside the canvas.
    pub crop: Option<[u32; 4]>,
}

/// What one sampling run needs.
pub(crate) struct SampleDesc<'a> {
    /// Row-major `(text_rows, 20480)` text conditioning.
    pub text: &'a [f32],
    /// Rows of `text`.
    pub text_rows: usize,
    /// The latent grid `(rows, cols)` of the image.
    pub grid: (usize, usize),
    /// The references, empty for text-to-image.
    pub references: &'a ReferenceTokens,
    /// The noise seed.
    pub seed: Seed,
    /// Euler steps.
    pub steps: u32,
}

impl Flux3KleinModel {
    /// Opens every component `located` names and uploads it to `backend`.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] for components that disagree or a
    /// residency this build does not run, and whatever a component's load
    /// refuses.
    pub(crate) fn load(
        located: &Flux3Located,
        desc: &ImageModelDesc,
        backend: &Arc<dyn Backend>,
    ) -> Result<Self, ImageGenError> {
        if desc.residency == ImageResidency::Swap {
            return Err(ImageGenError::Checkpoint(
                "--image-residency swap is not wired for the flux3 family yet".to_owned(),
            ));
        }
        let ops = Ops::new(backend)?;
        let text = load_text_encoder(
            &located.text_encoder_files,
            &located.text_encoder_config,
            &ops,
        )?;
        if text.context_width() != located.params.txt_in {
            return Err(ImageGenError::Checkpoint(format!(
                "the text encoder's taps are {} wide but the transformer reads {}",
                text.context_width(),
                located.params.txt_in
            )));
        }
        let store = StStore::open(&located.weight_file)?;
        let dit = DiT::load(&store, &ops, &located.params)?;
        let vae = Vae::load(&located.vae_path, &ops)?;
        let tokenizer_file = if located.tokenizer_path.is_dir() {
            located.tokenizer_path.join("tokenizer.json")
        } else {
            located.tokenizer_path.clone()
        };
        let tokenizer = PromptTokenizer::open(&tokenizer_file)?;
        let chat_template = load_chat_template(tokenizer_file.parent())?;
        Ok(Self {
            model_id: "flux-3-klein".to_owned(),
            ops,
            text,
            dit,
            vae,
            tokenizer,
            chat_template,
        })
    }

    /// Encodes `prompt` to padded text conditioning.
    ///
    /// # Errors
    /// The refusals of [`encode`].
    pub(crate) fn encode_prompt(&self, prompt: &str) -> Result<EncodedText, ImageGenError> {
        encode(
            &self.text,
            &self.tokenizer,
            &self.chat_template,
            &self.ops,
            prompt,
        )
    }

    /// Decodes, prepares and VAE-encodes the job's references.
    ///
    /// # Errors
    /// [`ImageGenError::ReferenceImage`] for too many references, one that
    /// does not decode or is under 16 pixels on a side, and the device's own
    /// failures.
    pub(crate) fn prepare_references(
        &self,
        job: &ImageGenerationJob,
    ) -> Result<PreparedReferences, ImageGenError> {
        let count = job.input_references.len();
        check_reference_count(count)?;
        let mut decoded = Vec::with_capacity(count);
        for (index, reference) in job.input_references.iter().enumerate() {
            let picture = decode_reference(&reference.bytes, index)?;
            check_reference_size(&picture, index)?;
            decoded.push(picture);
        }
        let (pictures, canvas, crop) = match decoded.as_slice() {
            [] => (Vec::new(), None, None),
            [one] => {
                let (canvas_picture, crop) = prepare_single_reference(one);
                let size = canvas_picture.dimensions();
                (vec![canvas_picture], Some(size), crop)
            }
            many => (
                many.iter().map(prepare_multi_reference).collect(),
                None,
                None,
            ),
        };
        let mut tokens = ReferenceTokens::default();
        for (index, picture) in pictures.iter().enumerate() {
            let (w, h) = (picture.width() as usize, picture.height() as usize);
            let planar = planar_unit_range(picture);
            let latent = self.vae.encode(&self.ops, &planar, w, h)?;
            tokens.tokens.extend(latent.into_iter().map(round_bf16));
            let ids = reference_ids(h / PIXELS_PER_TOKEN, w / PIXELS_PER_TOKEN, index);
            tokens.rows += ids.len();
            tokens.ids.extend(ids);
        }
        Ok(PreparedReferences {
            tokens,
            canvas,
            crop,
        })
    }

    /// Runs the flow-map Euler walk and returns the final image tokens,
    /// row-major `(rows * cols, 128)` f32.
    ///
    /// # Errors
    /// The device's failures, and a progress callback's.
    pub(crate) fn sample(
        &self,
        desc: &SampleDesc<'_>,
        progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<Vec<f32>, ImageGenError> {
        let ops = &self.ops;
        let (rows, cols) = desc.grid;
        let image_rows = rows * cols;
        let cond_rows = desc.references.rows;
        let pack = Pack::single(image_rows, cond_rows, desc.text_rows);
        tracing::info!(
            image_rows,
            cond_rows,
            text_rows = pack.text.len(),
            steps = desc.steps,
            "flux3 sampling started"
        );
        let count = image_rows * IMG_CHANNELS;
        let noise = initial_noise(desc.seed, count);
        let latents = ops.upload_f32("flux3 latents", &noise)?;
        let text = ops.upload_scratch("flux3 text", desc.text)?;
        let cond = if cond_rows > 0 {
            Some(ops.upload_scratch("flux3 references", &desc.references.tokens)?)
        } else {
            None
        };
        let flat = |ids: &[[f32; 4]]| ids.iter().flatten().copied().collect::<Vec<f32>>();
        let image_ids = ops.upload_f32("flux3 image ids", &flat(&image_ids(rows, cols)))?;
        let cond_ids = if cond_rows > 0 {
            Some(ops.upload_f32("flux3 reference ids", &flat(&desc.references.ids))?)
        } else {
            None
        };
        let text_ids = ops.upload_f32("flux3 text ids", &flat(&text_ids(desc.text_rows)))?;

        let times = schedule(desc.steps, ALPHA);
        let began = Instant::now();
        let step_began = std::cell::Cell::new(Instant::now());
        run_euler(
            ops,
            &latents,
            count,
            &times,
            &mut |_, step, state| {
                step_began.set(Instant::now());
                self.dit.forward(
                    ops,
                    &DiTInput {
                        image: state,
                        image_cond: cond.as_ref().map(|c| c.view()),
                        text: text.view(),
                        image_rows,
                        cond_rows,
                        text_rows: desc.text_rows,
                        t: step.t,
                        t_ref: step.t_ref,
                        image_ids: image_ids.view(),
                        cond_ids: cond_ids.as_ref().map(shiso_core::DeviceBuffer::view),
                        text_ids: text_ids.view(),
                    },
                )
            },
            &mut |index, step| {
                // Launches only queue; draining here charges the step's
                // device work to the step that issued it.
                ops.backend()
                    .sync()
                    .map_err(|e| ImageGenError::Backend(format!("flux3 step sync: {e}")))?;
                tracing::info!(
                    step = index + 1,
                    steps = desc.steps,
                    t = step.t,
                    seconds = step_began.get().elapsed().as_secs_f64(),
                    "flux3 denoising step"
                );
                if let Some(report) = progress {
                    report(ImageStepEvent {
                        step: u32::try_from(index).unwrap_or(u32::MAX),
                        total_steps: desc.steps,
                        partial_image: None,
                    });
                }
                Ok(())
            },
        )?;
        tracing::info!(
            seconds = began.elapsed().as_secs_f64(),
            steps = desc.steps,
            "flux3 denoising finished"
        );
        ops.download("flux3 final latents", latents.view(), count)
    }

    /// Decodes the image tokens, crops, and encodes the picture.
    ///
    /// # Errors
    /// The decoder's refusals and [`ImageGenError::Encoding`].
    pub(crate) fn render(
        &self,
        latents: &[f32],
        size: (u32, u32),
        crop: Option<[u32; 4]>,
        format: ImageOutputFormat,
        compression: Option<u32>,
    ) -> Result<(Vec<u8>, (u32, u32)), ImageGenError> {
        let (w, h) = (size.0 as usize, size.1 as usize);
        let rgb = self.vae.decode(&self.ops, latents, w, h)?;
        let (pixels, out) = pack_rgb8(&rgb, size, crop)?;
        let bytes = encode_image(&pixels, out.0, out.1, Channels::Rgb, format, compression)?;
        Ok((bytes, out))
    }
}

/// A picture as planar `(3, h, w)` f32 in `[-1, 1]`, the VAE's input range.
fn planar_unit_range(picture: &image::RgbImage) -> Vec<f32> {
    let (w, h) = (picture.width() as usize, picture.height() as usize);
    let mut out = vec![0f32; 3 * w * h];
    for (x, y, pixel) in picture.enumerate_pixels() {
        for (c, &v) in pixel.0.iter().enumerate() {
            out[(c * h + y as usize) * w + x as usize] = f32::from(v) / 127.5 - 1.0;
        }
    }
    out
}

/// Packs a planar `(3, h, w)` f32 field in `[0, 255]` to interleaved RGB
/// bytes, cropped to `crop` (`[left, top, right, bottom]`) when given.
///
/// The cast truncates, as the reference's does: its `to(torch.uint8)` does
/// not round, and a rounding here would put half the samples one level off.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "samples are clamped to [0, 255] first and the cast truncates on purpose"
)]
fn pack_rgb8(
    rgb: &[f32],
    size: (u32, u32),
    crop: Option<[u32; 4]>,
) -> Result<(Vec<u8>, (u32, u32)), ImageGenError> {
    let (w, h) = (size.0 as usize, size.1 as usize);
    if rgb.len() != 3 * w * h {
        return Err(ImageGenError::Encoding(format!(
            "expected {} colour samples, got {}",
            3 * w * h,
            rgb.len()
        )));
    }
    let [left, top, right, bottom] = crop.unwrap_or([0, 0, size.0, size.1]);
    if right > size.0 || bottom > size.1 || left >= right || top >= bottom {
        return Err(ImageGenError::Encoding(format!(
            "crop box {left},{top},{right},{bottom} does not fit a {w}x{h} picture"
        )));
    }
    let (left, top, right, bottom) = (left as usize, top as usize, right as usize, bottom as usize);
    let mut pixels = Vec::with_capacity((right - left) * (bottom - top) * 3);
    for y in top..bottom {
        for x in left..right {
            for c in 0..3 {
                pixels.push(rgb[(c * h + y) * w + x].clamp(0.0, 255.0) as u8);
            }
        }
    }
    Ok((
        pixels,
        (
            u32::try_from(right - left).unwrap_or(u32::MAX),
            u32::try_from(bottom - top).unwrap_or(u32::MAX),
        ),
    ))
}

impl ImageGenerator for Flux3KleinModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn limits(&self) -> ImageLimits {
        ImageLimits {
            size_multiple: SIZE_MULTIPLE,
            min_side: MIN_SIDE,
            max_side: MAX_SIDE,
            default_size: DEFAULT_SIZE,
            default_steps: DEFAULT_STEPS,
            fixed_steps: None,
        }
    }

    fn generate(
        &self,
        job: &ImageGenerationJob,
        progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        job.validate(&self.limits())?;
        let began = Instant::now();
        // Declared first so it drops last, after every working buffer of
        // this generation is back in the pool.
        let _trim = TrimPool(&self.ops);
        let references = self.prepare_references(job)?;
        let encoded = self.encode_prompt(&job.prompt)?;
        let latents = self.sample(
            &SampleDesc {
                text: &encoded.tokens,
                text_rows: encoded.len,
                grid: (
                    job.height as usize / PIXELS_PER_TOKEN,
                    job.width as usize / PIXELS_PER_TOKEN,
                ),
                references: &references.tokens,
                seed: job.seed,
                steps: job.steps,
            },
            progress,
        )?;
        // A single reference's output returns at the fitted size when the
        // job asked for the canvas it was padded to.
        let crop = references
            .crop
            .filter(|_| references.canvas == Some((job.width, job.height)));
        let phase = Instant::now();
        let (bytes, (width, height)) = self.render(
            &latents,
            (job.width, job.height),
            crop,
            job.output_format,
            job.output_compression,
        )?;
        tracing::info!(
            seconds = phase.elapsed().as_secs_f64(),
            total_seconds = began.elapsed().as_secs_f64(),
            bytes = bytes.len(),
            "flux3 image decoded and encoded"
        );
        Ok(GeneratedImage {
            bytes,
            media_type: job.output_format.mime_type(),
            width,
            height,
            prompt_tokens: u32::try_from(encoded.real).unwrap_or(u32::MAX),
            completion_tokens: u32::try_from(
                (job.height as usize / PIXELS_PER_TOKEN)
                    * (job.width as usize / PIXELS_PER_TOKEN)
                    * job.steps as usize,
            )
            .unwrap_or(u32::MAX),
        })
    }
}

/// Empties the working-buffer pool when a generation ends, however it ends.
struct TrimPool<'o>(&'o Ops);

impl Drop for TrimPool<'_> {
    fn drop(&mut self) {
        self.0.trim_pool();
    }
}

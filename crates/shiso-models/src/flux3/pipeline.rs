//! The FLUX.3 Klein pipeline: a prompt (and optional reference pictures) in,
//! encoded image bytes out.
//!
//! One [`Flux3KleinModel`] holds the three components - the Qwen3-VL text
//! encoder, the diffusion transformer and the autoencoder - on one backend,
//! with the tokenizer and chat template that turn text into ids.
//! [`ImageGenerator::generate`] is four stages, each a method so a device
//! test can run one without the rest: [`Flux3KleinModel::tokenize_prompt`] and the encoder
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
//! becomes once [`default_size_for_references`](super::default_size_for_references) has run), so
//! an edit generated at the canvas size comes back at the reference's own
//! size, which need not sit on the 16-pixel grid.
//!
//! # Residency
//!
//! The text encoder (about 8 GB in BF16, 4 GB as INT8) and the transformer
//! (4 GB as the Q4 export) do not both fit a 16 GB-class working set beside
//! a 1024x1024 job, so such a host swaps, as the Qwen-Image pipeline does:
//!
//! 1. The transformer is dropped if loaded, the encoder is loaded (its pages
//!    read ahead in large chunks first), the prompt encoded, and the encoder
//!    dropped *before* the transformer is imported. While it encodes, as
//!    much of the transformer as fits beside it is read into the page cache
//!    on a background thread.
//! 2. The rest is read ahead, the transformer imported, the steps run.
//! 3. The transformer stays loaded if it fits beside the decode; otherwise
//!    it is dropped before the decode, and the encoder's pages are read
//!    ahead behind the decode for the next job when they fit.
//!
//! The autoencoder is small and always loaded. Every transformer import
//! goes through one function that, in swap mode, drops the encoder first, so
//! the two are never loaded together whatever path led there (including an
//! encode that failed). [`JobSizer`] decides from byte counts alone, per
//! job, and refuses one no residency fits.
//!
//! # Progress
//!
//! A progress event goes out after every step without a partial image: a
//! preview is a whole VAE decode, which per step would cost more than the
//! step.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use shiso_core::Backend;
use shiso_core::ids::Seed;
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits,
    ImageOutputFormat, ImageStepEvent,
};

use super::config::{Flux3Located, Flux3Params};
use super::pack::{Pack, image_ids, reference_ids, text_ids};
use super::reference::{
    check_reference_count, check_reference_size, decode_reference, prepare_multi_reference,
    prepare_single_reference,
};
use super::sampler::{initial_noise, round_bf16, run_euler};
use super::scheduler::{ALPHA, DEFAULT_STEPS, schedule};
use super::sizing::{JobShape, JobSizer};
use super::text_encoder::{
    EncodedText, PaddedPrompt, TAP_LAYERS, detect_prefix, encode_padded, load as load_text_encoder,
    tokenize,
};
use super::transformer::{DiT, DiTInput};
use super::vae::{PIXELS_PER_TOKEN, Vae};
use crate::diffusion::encode::{Channels, encode_image};
use crate::diffusion::ops::Ops;
use crate::diffusion::prompt::{PromptTokenizer, load_chat_template};
use crate::diffusion::qwen3_text::TextEncoder;
use crate::diffusion::residency::{
    FileSpan, Footprint, Prefetch, Residency, ResidencyMode, split_at, total_bytes,
};
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

/// A read-ahead of a component's first spans already running, and the
/// spans it left for later.
type ReadAhead = (Prefetch, Vec<FileSpan>);

/// A pipeline component, for the residency log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Component {
    /// The text encoder.
    Encoder,
    /// The diffusion transformer.
    Denoiser,
}

/// One residency change, in the order it happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidencyEvent {
    /// The component was imported; `device_bytes` is the device's own
    /// allocation growth across the import.
    Loaded {
        /// Which one.
        component: Component,
        /// Device bytes the import added.
        device_bytes: i64,
    },
    /// The component was dropped; `device_bytes` is the allocation shrink.
    Dropped {
        /// Which one.
        component: Component,
        /// Device bytes the drop released.
        device_bytes: i64,
    },
}

/// The text encoder's files and the facts the planner needs about it.
#[derive(Debug)]
struct EncoderFiles {
    weights: Vec<PathBuf>,
    config: PathBuf,
    /// The byte ranges an import reads: the embedding and the layers through
    /// the last tap.
    spans: Vec<FileSpan>,
}

/// What a job changes between jobs.
#[derive(Debug)]
struct State {
    encoder: Option<TextEncoder>,
    dit: Option<DiT>,
    /// The encoder's read-ahead started behind the last decode.
    encoder_prefetch: Option<Prefetch>,
    events: Vec<ResidencyEvent>,
}

/// Text rows the load-time residency decision is sized for.
const PLAN_TEXT_ROWS: usize = 512;

/// The FLUX.3 Klein generator: text encoder, transformer and autoencoder on
/// one device, driven one job at a time.
#[derive(Debug)]
pub(crate) struct Flux3KleinModel {
    model_id: String,
    ops: Ops,
    params: Flux3Params,
    dit_path: PathBuf,
    dit_spans: Vec<FileSpan>,
    encoder_files: EncoderFiles,
    sizer: JobSizer,
    vae: Vae,
    /// The vocabulary; caller text spelling a control token is encoded as
    /// plain text so it cannot forge a marker.
    tokenizer: PromptTokenizer,
    chat_template: String,
    state: Mutex<State>,
}

/// Device bytes in use now, when the backend says.
fn device_used(backend: &Arc<dyn Backend>) -> Option<u64> {
    backend
        .memory_info()
        .map(|info| info.total.saturating_sub(info.free))
}

/// `after - before` as signed bytes.
fn delta(before: Option<u64>, after: Option<u64>) -> i64 {
    match (before, after) {
        (Some(b), Some(a)) => {
            i64::try_from(a).unwrap_or(i64::MAX) - i64::try_from(b).unwrap_or(i64::MAX)
        }
        _ => 0,
    }
}

fn ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[expect(clippy::cast_precision_loss, reason = "byte counts logged in MB")]
fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}

/// The text encoder's file facts from headers alone: the spans an import
/// reads, the residual width and the MLP width.
fn read_encoder_facts(files: &[PathBuf]) -> Result<(Vec<FileSpan>, usize, usize), ImageGenError> {
    let store = StStore::open_files(files)?;
    let prefix = detect_prefix(&store, files)?;
    let last_tap = TAP_LAYERS[TAP_LAYERS.len() - 1];
    let layers = format!("{prefix}layers.");
    let embed = format!("{prefix}embed_tokens.");
    let spans = store.spans(|name| {
        name.starts_with(&embed)
            || name.strip_prefix(&layers).is_some_and(|rest| {
                rest.split('.')
                    .next()
                    .and_then(|n| n.parse::<usize>().ok())
                    .is_some_and(|n| n <= last_tap)
            })
    });
    let axis = |name: String, axis: usize| -> Result<usize, ImageGenError> {
        store
            .require(&name)?
            .shape
            .get(axis)
            .and_then(|&d| usize::try_from(d).ok())
            .ok_or_else(|| ImageGenError::Checkpoint(format!("'{name}' has no axis {axis}")))
    };
    let hidden = axis(format!("{prefix}embed_tokens.weight"), 1)?;
    let inter = axis(format!("{prefix}layers.0.mlp.gate_proj.weight"), 0)?;
    Ok((spans, hidden, inter))
}

impl Flux3KleinModel {
    /// Opens the components `located` names: sizes them from headers,
    /// loads the autoencoder, and loads the rest when the residency plan for
    /// a default job keeps them together.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] for components that disagree, and
    /// [`ImageGenError::TooLargeForMemory`] when not even a default job fits
    /// the device; whatever a component's load refuses.
    pub(crate) fn load(
        located: &Flux3Located,
        desc: &ImageModelDesc,
        backend: &Arc<dyn Backend>,
    ) -> Result<Self, ImageGenError> {
        let began = Instant::now();
        let ops = Ops::new(backend)?;
        let (encoder_spans, encoder_hidden, encoder_inter) =
            read_encoder_facts(&located.text_encoder_files)?;
        if 8 * encoder_hidden != located.params.txt_in {
            return Err(ImageGenError::Checkpoint(format!(
                "the text encoder's taps are {} wide but the transformer reads {}",
                8 * encoder_hidden,
                located.params.txt_in
            )));
        }
        let dit_spans = StStore::open(&located.weight_file)?.spans(|_| true);
        let vae = Vae::load(&located.vae_path, &ops)?;
        let footprint = Footprint {
            encoder: total_bytes(&encoder_spans),
            denoiser: total_bytes(&dit_spans),
            decoder: Vae::weight_bytes(&located.vae_path)?,
        };
        let working_set = backend.memory_info().map_or(u64::MAX, |info| info.total);
        let mode = match desc.residency {
            ImageResidency::Auto => ResidencyMode::Auto,
            ImageResidency::Resident => ResidencyMode::Resident,
            ImageResidency::Swap => ResidencyMode::Swap,
        };
        let sizer = JobSizer {
            mode,
            working_set,
            reserve: desc.reserve_bytes,
            footprint,
            encoder_hidden,
            encoder_inter,
            hidden: located.params.hidden,
            mlp_hidden: located.params.mlp_hidden,
            txt_in: located.params.txt_in,
            vae_channels: vae.full_resolution_channels(),
        };
        tracing::info!(
            working_set_mb = mb(working_set),
            reserve_mb = mb(desc.reserve_bytes),
            encoder_mb = mb(footprint.encoder),
            denoiser_mb = mb(footprint.denoiser),
            decoder_mb = mb(footprint.decoder),
            ?mode,
            "flux3 residency inputs"
        );
        let tokenizer_file = if located.tokenizer_path.is_dir() {
            located.tokenizer_path.join("tokenizer.json")
        } else {
            located.tokenizer_path.clone()
        };
        let tokenizer = PromptTokenizer::open(&tokenizer_file)?;
        let chat_template = load_chat_template(tokenizer_file.parent())?;
        let model = Self {
            model_id: "flux-3-klein".to_owned(),
            ops,
            params: located.params.clone(),
            dit_path: located.weight_file.clone(),
            dit_spans,
            encoder_files: EncoderFiles {
                weights: located.text_encoder_files.clone(),
                config: located.text_encoder_config.clone(),
                spans: encoder_spans,
            },
            sizer,
            vae,
            tokenizer,
            chat_template,
            state: Mutex::new(State {
                encoder: None,
                dit: None,
                encoder_prefetch: None,
                events: Vec::new(),
            }),
        };
        // A default job decides what stays loaded: both components when they
        // fit together, else nothing yet (the first job swaps them in; the
        // cost of that first import is not measured here, so no warm-up runs).
        let residency = model.sizer.plan(&JobShape {
            width: DEFAULT_SIZE.0 as usize,
            height: DEFAULT_SIZE.1 as usize,
            text_rows: PLAN_TEXT_ROWS,
            cond_rows: 0,
        })?;
        tracing::info!(?residency, "flux3 load-time residency");
        if residency == Residency::Resident {
            let mut state = model.state.lock().unwrap_or_else(PoisonError::into_inner);
            model.ensure_encoder(&mut state)?;
            model.ensure_denoiser(&mut state, None, false)?;
        }
        tracing::info!(ms = ms(began), "flux3 model loaded");
        Ok(model)
    }

    /// Loads the text encoder if it is not, reading its pages ahead first.
    fn ensure_encoder(&self, state: &mut State) -> Result<(), ImageGenError> {
        if state.encoder.is_some() {
            return Ok(());
        }
        let phase = Instant::now();
        let prefetch = match state.encoder_prefetch.take() {
            Some(running) => running,
            None => Prefetch::spawn("encoder", self.encoder_files.spans.clone())?,
        };
        let read = prefetch.wait()?;
        tracing::info!(
            ms = ms(phase),
            read_mb = mb(read.bytes),
            read_mb_per_s = read.mb_per_s(),
            "flux3 phase te_prefetch"
        );
        let phase = Instant::now();
        let before = device_used(self.ops.backend());
        let encoder = load_text_encoder(
            &self.encoder_files.weights,
            &self.encoder_files.config,
            &self.ops,
        )?;
        if encoder.context_width() != self.params.txt_in {
            return Err(ImageGenError::Checkpoint(format!(
                "the text encoder's taps are {} wide but the transformer reads {}",
                encoder.context_width(),
                self.params.txt_in
            )));
        }
        let device_bytes = delta(before, device_used(self.ops.backend()));
        tracing::info!(ms = ms(phase), device_bytes, "flux3 phase te_load");
        state.encoder = Some(encoder);
        state.events.push(ResidencyEvent::Loaded {
            component: Component::Encoder,
            device_bytes,
        });
        Ok(())
    }

    /// Loads the transformer if it is not. `ahead` is a read-ahead of its
    /// first spans already running; the rest is read here.
    ///
    /// This is the only place the transformer is imported, and with `swap`
    /// it drops the encoder first - whatever path led here - so a swapping
    /// host never holds both. An encoder read-ahead left from an earlier
    /// decode is cancelled: this import may evict its pages.
    fn ensure_denoiser(
        &self,
        state: &mut State,
        ahead: Option<ReadAhead>,
        swap: bool,
    ) -> Result<(), ImageGenError> {
        state.encoder_prefetch = None;
        if state.dit.is_some() {
            return Ok(());
        }
        if swap {
            self.release(state, Component::Encoder);
        }
        let phase = Instant::now();
        let (head, rest) = match ahead {
            Some((head, rest)) => (Some(head), rest),
            None => (None, self.dit_spans.clone()),
        };
        let tail = Prefetch::spawn("denoiser", rest)?;
        let mut bytes = 0;
        if let Some(head) = head {
            bytes += head.wait()?.bytes;
        }
        let read = tail.wait()?;
        bytes += read.bytes;
        tracing::info!(
            ms = ms(phase),
            read_mb = mb(bytes),
            tail_mb_per_s = read.mb_per_s(),
            "flux3 phase dit_prefetch"
        );
        let phase = Instant::now();
        let before = device_used(self.ops.backend());
        let dit = DiT::load(&StStore::open(&self.dit_path)?, &self.ops, &self.params)?;
        let device_bytes = delta(before, device_used(self.ops.backend()));
        tracing::info!(ms = ms(phase), device_bytes, "flux3 phase dit_load");
        state.dit = Some(dit);
        state.events.push(ResidencyEvent::Loaded {
            component: Component::Denoiser,
            device_bytes,
        });
        Ok(())
    }

    /// Drops `component` if loaded and gives its pooled scratch back.
    fn release(&self, state: &mut State, component: Component) {
        let phase = Instant::now();
        let before = device_used(self.ops.backend());
        let dropped = match component {
            Component::Encoder => state.encoder.take().is_some(),
            Component::Denoiser => state.dit.take().is_some(),
        };
        if !dropped {
            return;
        }
        self.ops.trim_pool();
        let device_bytes = delta(device_used(self.ops.backend()), before);
        tracing::info!(ms = ms(phase), device_bytes, ?component, "flux3 phase drop");
        state.events.push(ResidencyEvent::Dropped {
            component,
            device_bytes,
        });
    }

    /// The prompt's conditioning, from the loaded encoder (loading it, and
    /// in swap mode dropping the transformer first and the encoder after),
    /// and the read-ahead of the transformer started during the encode.
    fn encode_text(
        &self,
        state: &mut State,
        padded: &PaddedPrompt,
        job: &JobShape,
        residency: Residency,
    ) -> Result<(EncodedText, Option<ReadAhead>), ImageGenError> {
        let swap = matches!(residency, Residency::Swap { .. });
        if swap {
            self.release(state, Component::Denoiser);
        }
        self.ensure_encoder(state)?;
        // While the encoder runs, read ahead as much of the transformer as
        // fits beside the encoder's phase; the rest waits for the drop.
        let mut ahead = None;
        if swap && state.dit.is_none() {
            let peaks = self.sizer.peaks(job);
            let room = self
                .sizer
                .working_set
                .saturating_sub(self.sizer.reserve)
                .saturating_sub(
                    self.sizer.footprint.encoder + self.sizer.footprint.decoder + peaks.encode,
                );
            let (head, rest) = split_at(&self.dit_spans, room);
            if !head.is_empty() {
                tracing::info!(
                    overlap_mb = mb(total_bytes(&head)),
                    "flux3 dit read-ahead during encode"
                );
                ahead = Some((Prefetch::spawn("denoiser-overlap", head)?, rest));
            }
        }
        let phase = Instant::now();
        let encoded = state
            .encoder
            .as_ref()
            .ok_or_else(|| ImageGenError::Backend("the text encoder is not loaded".to_owned()))
            .and_then(|enc| encode_padded(enc, &self.ops, padded));
        tracing::info!(
            ms = ms(phase),
            rows = padded.ids.len(),
            ok = encoded.is_ok(),
            "flux3 phase encode"
        );
        // Dropped whether the encode worked or not: a failed job must not
        // leave the encoder where the next job's transformer goes.
        if swap {
            self.release(state, Component::Encoder);
        }
        Ok((encoded?, ahead))
    }

    /// What happens between the last step and the decode: a swapping host
    /// that cannot keep the transformer beside the decode drops it, and
    /// reads the next job's encoder pages ahead behind the decode when they
    /// fit beside the decoder and the decode's working set - otherwise the
    /// read would evict pages the decode is using. Returns whether that
    /// read-ahead is left for after the decode.
    fn after_denoise(
        &self,
        state: &mut State,
        residency: Residency,
        shape: &JobShape,
    ) -> Result<bool, ImageGenError> {
        let Residency::Swap {
            keep_denoiser: false,
        } = residency
        else {
            return Ok(false);
        };
        self.release(state, Component::Denoiser);
        let decode_peak = self.sizer.peaks(shape).decode;
        let budget = self.sizer.working_set.saturating_sub(self.sizer.reserve);
        if self.sizer.footprint.encoder + self.sizer.footprint.decoder + decode_peak <= budget {
            state.encoder_prefetch = Some(Prefetch::spawn(
                "encoder",
                self.encoder_files.spans.clone(),
            )?);
            Ok(false)
        } else {
            Ok(true)
        }
    }

    /// Tokenizes and pads `prompt`.
    ///
    /// # Errors
    /// The refusals of [`tokenize`].
    pub(crate) fn tokenize_prompt(&self, prompt: &str) -> Result<PaddedPrompt, ImageGenError> {
        tokenize(&self.tokenizer, &self.chat_template, prompt)
    }

    /// The residency changes so far, oldest first.
    #[cfg(test)]
    pub(crate) fn events(&self) -> Vec<ResidencyEvent> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .events
            .clone()
    }

    /// The sizing the residency plan reads, for a test to shrink the
    /// device's working set.
    #[cfg(test)]
    pub(crate) fn sizer_mut(&mut self) -> &mut JobSizer {
        &mut self.sizer
    }
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
        dit: &DiT,
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
                dit.forward(
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
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        // Declared after the state guard so it drops first, with the guard
        // still held, after every working buffer of this generation is back
        // in the pool.
        let _trim = TrimPool(&self.ops);
        let references = self.prepare_references(job)?;
        let padded = self.tokenize_prompt(&job.prompt)?;
        let shape = JobShape {
            width: job.width as usize,
            height: job.height as usize,
            text_rows: padded.ids.len(),
            cond_rows: references.tokens.rows,
        };
        let residency = self.sizer.plan(&shape)?;
        tracing::info!(
            width = job.width,
            height = job.height,
            steps = job.steps,
            text_rows = shape.text_rows,
            cond_rows = shape.cond_rows,
            ?residency,
            "flux3 generation started"
        );
        let (encoded, ahead) = self.encode_text(&mut state, &padded, &shape, residency)?;
        let swap = matches!(residency, Residency::Swap { .. });
        self.ensure_denoiser(&mut state, ahead, swap)?;
        let latents = {
            let dit = state.dit.as_ref().ok_or_else(|| {
                ImageGenError::Backend("the transformer is not loaded".to_owned())
            })?;
            self.sample(
                dit,
                &SampleDesc {
                    text: &encoded.tokens,
                    text_rows: encoded.len,
                    grid: (
                        shape.height / PIXELS_PER_TOKEN,
                        shape.width / PIXELS_PER_TOKEN,
                    ),
                    references: &references.tokens,
                    seed: job.seed,
                    steps: job.steps,
                },
                progress,
            )?
        };
        let read_after_decode = self.after_denoise(&mut state, residency, &shape)?;
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
        // The read-ahead that could not run beside the decode runs after it,
        // once the decode's working buffers are released.
        if read_after_decode
            && state.encoder.is_none()
            && self.sizer.footprint.encoder + self.sizer.footprint.decoder
                <= self.sizer.working_set.saturating_sub(self.sizer.reserve)
        {
            self.ops.trim_pool();
            state.encoder_prefetch = Some(Prefetch::spawn(
                "encoder",
                self.encoder_files.spans.clone(),
            )?);
        }
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
                (shape.height / PIXELS_PER_TOKEN)
                    * (shape.width / PIXELS_PER_TOKEN)
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

//! The Qwen-Image-2.1 pipeline: a prompt in, encoded RGBA image bytes out.
//!
//! [`QwenImage21Model`] composes the family's parts - the templated prompt
//! ([`super::prompt`]), the shared Qwen3-VL text encoder read to its last
//! layer, the transformer's once-per-job text prefix and per-step target
//! pass ([`super::transformer`]), the sigma ladder ([`super::scheduler`])
//! and the RGBA decoder ([`super::vae`]) - and decides per job which of the
//! three stay on the device ([`crate::diffusion::residency`]).
//!
//! # Residency
//!
//! The INT8 text encoder's layers (6.95 GB) and the transformer (7.26 GB)
//! do not fit a 16 GB-class working set together, so such a host swaps:
//!
//! 1. The encoder is loaded (its pages read ahead in large sequential
//!    chunks first), the prompt encoded, and the encoder dropped. While it
//!    encodes, as much of the transformer as fits beside it is read into
//!    the page cache on a background thread.
//! 2. The rest of the transformer is read ahead, it is imported, and the
//!    prefix and the steps run.
//! 3. If the transformer fits beside the decode it stays loaded after the
//!    job - a later job whose prompt features are cached never needs the
//!    encoder and runs with no swap at all. Otherwise it is dropped before
//!    the decode and, when the encoder's pages fit beside the decoder and
//!    the decode's working set, they are read ahead behind the decode for
//!    the next job. A transformer import cancels any such read-ahead (its
//!    pages may be evicted by the import, so a later job reads them again).
//!
//! Every transformer import goes through one function, which in swap mode
//! drops the encoder first - also after a cached prompt or a failed encode -
//! so the two are never loaded together. Dropped components keep their
//! pages in the page cache (the shard's `release_cached` is not called):
//! the next swap-in of the same file is then partly a cache hit.
//!
//! The decoder (1 GB widened) is loaded once and kept. Every read-ahead only
//! warms the page cache: nothing is imported to the device while another
//! component's weights would have to share the working set with it.
//!
//! # Progress
//!
//! A progress event goes out after every step without a partial image: a
//! preview is a whole decode (2.2 s at 512x512), which per step would cost
//! more than a quarter of the image again.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use shiso_core::Backend;
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits, ImageStepEvent,
};

use super::config::QwenImageConfig;
use super::prompt::QwenImagePrompt;
use super::scheduler::{ScheduleSource, check_nodes, euler_dt, resolve};
use super::transformer::{BlockReuse, DitDesc, QwenDit, StepOpts};
use super::vae::{LatentGrid, QwenVae, QwenVaeDesc, SPATIAL_FACTOR};
use crate::diffusion::draft::upsample2x_noisy;
use crate::diffusion::encode::{Channels, encode_image};
use crate::diffusion::noise::gaussian;
use crate::diffusion::ops::Ops;
use crate::diffusion::qwen3_text::{TapSet, TextEncoder, TextEncoderDesc};
use crate::diffusion::residency::{
    FileSpan, Footprint, JobPeaks, PlanDesc, Prefetch, Residency, ResidencyMode, plan, split_at,
    total_bytes,
};
use crate::diffusion::weights::{StStore, safetensors_files};
use crate::image_registry::{ImageModelDesc, ImageResidency};

/// The text encoder's tensor prefix in the shipped file.
const TE_PREFIX: &str = "model.";
/// Prompts whose encoder features are kept, most recent first.
const FEATURE_CACHE: usize = 4;
/// Image sides are multiples of this.
const SIZE_MULTIPLE: u32 = 32;
/// The smallest side served.
const MIN_SIDE: u32 = 256;
/// The largest side served.
const MAX_SIDE: u32 = 2048;
/// The size a request that names none gets.
const DEFAULT_SIZE: (u32, u32) = (512, 512);
/// The base model's default step count.
const BASE_STEPS: u32 = 40;
/// Encoder rows the planner sizes the load-time decision for: the cap
/// plus the system turn.
const PLAN_TOKENS: usize = super::prompt::MAX_PROMPT_TOKENS + super::prompt::DROP_IDX;

/// Where the sigma ladder's nodes come from; both are shifted by the
/// scheduler config.
#[derive(Debug, Clone)]
enum Schedule {
    /// `model_index.json`'s `sample_sigmas`.
    Nodes(Vec<f32>),
    /// The base model's linear nodes.
    Linear,
}

impl Schedule {
    fn source(&self) -> ScheduleSource<'_> {
        match self {
            Self::Nodes(n) => ScheduleSource::Nodes(n),
            Self::Linear => ScheduleSource::Linear,
        }
    }

    fn fixed_steps(&self) -> Option<u32> {
        match self {
            Self::Nodes(v) => Some(u32::try_from(v.len()).unwrap_or(u32::MAX)),
            Self::Linear => None,
        }
    }
}

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
    /// The byte ranges an import reads: the layers and the embedding's
    /// row scales, never the vision tower, the head or (INT8) the table.
    spans: Vec<FileSpan>,
    hidden: usize,
    intermediate: usize,
}

/// What the worker thread changes between jobs.
#[derive(Debug)]
struct State {
    encoder: Option<TextEncoder>,
    denoiser: Option<QwenDit>,
    decoder: QwenVae,
    /// The encoder's read-ahead started behind the last decode.
    encoder_prefetch: Option<Prefetch>,
    /// `(kept ids, features)`, most recent first.
    features: Vec<(Vec<u32>, Vec<f32>)>,
    events: Vec<ResidencyEvent>,
}

/// The Qwen-Image-2.1 generator.
#[derive(Debug)]
pub(crate) struct QwenImage21Model {
    model_id: String,
    ops: Ops,
    config: QwenImageConfig,
    schedule: Schedule,
    prompt: QwenImagePrompt,
    dit_path: PathBuf,
    dit_spans: Vec<FileSpan>,
    encoder_files: EncoderFiles,
    mode: ResidencyMode,
    working_set: u64,
    reserve: u64,
    footprint: Footprint,
    /// Leading steps of every job run on the half-size grid
    /// ([`crate::diffusion::draft`]).
    draft_steps: usize,
    /// Blocks the last step replays from the step before
    /// ([`super::transformer::BlockReuse`]).
    final_reuse: Option<std::ops::Range<usize>>,
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

/// How much of the draft's upsampled state the hand-over keeps
/// ([`upsample2x_noisy`]'s `keep`): 1/4 keeps all of it, the most the
/// nearest-neighbour covariance allows. Measured on a 6-step distilled
/// checkpoint at 1024x1024 (4 draft steps): 1/8 and 1/16 start the
/// full-size steps at a higher sigma with no visible gain in detail.
const DRAFT_KEEP: f32 = 0.25;

fn ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[expect(clippy::cast_precision_loss, reason = "byte counts logged in MB")]
fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}

/// The install's documents: `assets_dir` when given, else the first
/// directory holding `model_index.json` - the model's own directory, an
/// `image_assets/` beside it, or the same two levels up.
fn locate_assets(desc: &ImageModelDesc) -> Result<PathBuf, ImageGenError> {
    if let Some(dir) = &desc.assets_dir {
        return Ok(dir.clone());
    }
    let home = if desc.model_path.is_file() {
        desc.model_path.parent().unwrap_or(&desc.model_path)
    } else {
        &desc.model_path
    };
    let mut tried = Vec::new();
    for dir in home.ancestors().take(4) {
        for candidate in [dir.to_path_buf(), dir.join("image_assets")] {
            if candidate.join("model_index.json").is_file() {
                return Ok(candidate);
            }
            tried.push(candidate);
        }
    }
    Err(ImageGenError::Checkpoint(format!(
        "no model_index.json found for {} (looked in {}); name the install's documents \
         with --image-assets-dir",
        desc.model_path.display(),
        tried
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

/// A companion's weights: `explicit` when given, else the first of
/// `candidates` that holds a safetensors file.
fn locate_weights(
    explicit: Option<&Path>,
    candidates: &[PathBuf],
    what: &str,
    flag: &str,
) -> Result<PathBuf, ImageGenError> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    for dir in candidates {
        if dir.is_dir() && safetensors_files(dir).is_ok_and(|files| !files.is_empty()) {
            return Ok(dir.clone());
        }
    }
    Err(ImageGenError::Checkpoint(format!(
        "no {what} weights found (looked in {}); name them with {flag}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

impl QwenImage21Model {
    /// Locates the install, reads its documents, loads the decoder, and
    /// loads whatever the residency plan for a default job keeps loaded.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] for a missing document or weight file,
    /// malformed `sample_sigmas`, or components that disagree; whatever a
    /// component load refuses.
    #[expect(
        clippy::too_many_lines,
        reason = "one install: locate, read, size, then load in plan order, straight-line"
    )]
    pub(crate) fn load(
        desc: &ImageModelDesc,
        backend: &Arc<dyn Backend>,
    ) -> Result<Self, ImageGenError> {
        let began = Instant::now();
        let ops = Ops::new(backend)?;
        let assets = locate_assets(desc)?;
        let config = QwenImageConfig::read(&assets)?;
        // The checkpoint's documents are its whole schedule: shipped nodes
        // are checked here so a malformed index fails the load, not the
        // first request.
        let schedule = match &config.sample_sigmas {
            Some(nodes) => {
                check_nodes(nodes)?;
                Schedule::Nodes(nodes.clone())
            }
            None => Schedule::Linear,
        };
        if let Some(fixed) = schedule.fixed_steps()
            && desc.draft_steps >= fixed
        {
            return Err(ImageGenError::Checkpoint(format!(
                "--image-draft-steps {} leaves none of this model's fixed {fixed} steps at full size",
                desc.draft_steps
            )));
        }
        if let Some((start, end)) = desc.final_reuse
            && (start >= end || end as usize > config.dit.num_layers)
        {
            return Err(ImageGenError::Checkpoint(format!(
                "--image-final-reuse {start}:{end} is not a span of this model's {} blocks",
                config.dit.num_layers
            )));
        }
        if let Some(fixed) = schedule.fixed_steps()
            && desc.final_reuse.is_some()
            && fixed.saturating_sub(desc.draft_steps) < 2
        {
            return Err(ImageGenError::Checkpoint(format!(
                "--image-final-reuse needs the last two of this model's fixed {fixed} steps at \
                 full size, and --image-draft-steps {} leaves fewer",
                desc.draft_steps
            )));
        }
        let model_id = "qwen-image-2.1".to_string();
        let prompt = QwenImagePrompt::load(
            desc.tokenizer_path
                .as_deref()
                .unwrap_or(&assets.join("processor")),
        )?;

        let home = if desc.model_path.is_file() {
            desc.model_path.parent().unwrap_or(&desc.model_path)
        } else {
            &desc.model_path
        };
        let up = home.parent().unwrap_or(home);
        let te_path = locate_weights(
            desc.text_encoder_path.as_deref(),
            &[up.join("text_encoders"), assets.join("text_encoder")],
            "text encoder",
            "--image-text-encoder",
        )?;
        let vae_path = locate_weights(
            desc.vae_path.as_deref(),
            &[up.join("vae"), assets.join("vae")],
            "VAE",
            "--image-vae",
        )?;

        // Sizes from headers alone: nothing is read but the tensor tables.
        let te_weights = safetensors_files(&te_path)?;
        let (te_spans, te_hidden, te_intermediate) = {
            let store = StStore::open_files(&te_weights)?;
            let embed = format!("{TE_PREFIX}embed_tokens.weight");
            // An INT8 table is gathered row by row from the mapping, so only
            // a BF16 one is an import the read-ahead should cover.
            let embed_imported = store
                .get(&embed)
                .is_some_and(|e| e.dtype != shiso_modelfile::st_header::StDType::I8);
            let layers = format!("{TE_PREFIX}layers.");
            let spans = store.spans(|name| {
                name.starts_with(&layers)
                    || (name.starts_with(&format!("{TE_PREFIX}embed_tokens."))
                        && (name != embed || embed_imported))
            });
            let shape = |name: &str, axis: usize| -> Result<usize, ImageGenError> {
                let entry = store.require(name)?;
                entry
                    .shape
                    .get(axis)
                    .and_then(|&d| usize::try_from(d).ok())
                    .ok_or_else(|| {
                        ImageGenError::Checkpoint(format!("'{name}' has no axis {axis}"))
                    })
            };
            let hidden = shape(&embed, 1)?;
            let intermediate = shape(&format!("{layers}0.mlp.gate_proj.weight"), 0)?;
            (spans, hidden, intermediate)
        };
        if te_hidden != config.dit.context_in_dim {
            return Err(ImageGenError::Checkpoint(format!(
                "the text encoder's rows are {te_hidden} wide but the transformer reads {}",
                config.dit.context_in_dim
            )));
        }
        let dit_spans = StStore::open(&desc.model_path)?.spans(|_| true);
        let decoder_bytes = {
            let store = StStore::open(&vae_path)?;
            store
                .names()
                .filter(|n| n.starts_with("decoder.") || n.starts_with("conv2."))
                .filter_map(|n| store.get(n))
                .map(|e| e.elements() * 4)
                .sum()
        };
        let footprint = Footprint {
            encoder: total_bytes(&te_spans),
            denoiser: total_bytes(&dit_spans),
            decoder: decoder_bytes,
        };
        let working_set = backend.memory_info().map_or(u64::MAX, |info| info.total);
        let mode = match desc.residency {
            ImageResidency::Auto => ResidencyMode::Auto,
            ImageResidency::Resident => ResidencyMode::Resident,
            ImageResidency::Swap => ResidencyMode::Swap,
        };
        tracing::info!(
            working_set_mb = mb(working_set),
            reserve_mb = mb(desc.reserve_bytes),
            encoder_mb = mb(footprint.encoder),
            denoiser_mb = mb(footprint.denoiser),
            decoder_mb = mb(footprint.decoder),
            ?mode,
            "qwen image residency inputs"
        );

        let phase = Instant::now();
        let decoder = QwenVae::load(
            &QwenVaeDesc {
                weights: &vae_path,
                config: &assets.join("vae/config.json"),
            },
            &ops,
        )?;
        tracing::info!(ms = ms(phase), "qwen image phase vae_load");

        let model = Self {
            model_id,
            ops,
            config,
            schedule,
            prompt,
            dit_path: desc.model_path.clone(),
            dit_spans,
            encoder_files: EncoderFiles {
                weights: te_weights,
                config: assets.join("text_encoder/config.json"),
                spans: te_spans,
                hidden: te_hidden,
                intermediate: te_intermediate,
            },
            mode,
            working_set,
            reserve: desc.reserve_bytes,
            footprint,
            draft_steps: desc.draft_steps as usize,
            final_reuse: desc
                .final_reuse
                .map(|(start, end)| start as usize..end as usize),
            state: Mutex::new(State {
                encoder: None,
                denoiser: None,
                decoder,
                encoder_prefetch: None,
                features: Vec::new(),
                events: Vec::new(),
            }),
        };

        // Load what a default job starts with: the encoder always (every new
        // prompt starts there), the transformer too when resident. A swapping
        // host first runs the swap once (transformer, encoder, transformer,
        // each dropped): on the 16 GB M4 the first one or two no-copy imports
        // of a component in a process cost 9-12 s however warm its pages are,
        // and a re-import in the steady alternation 1-2.5 s. That is a
        // server-start cost; left in place it lands on the first request.
        let grid = LatentGrid {
            h: DEFAULT_SIZE.1 as usize / SPATIAL_FACTOR,
            w: DEFAULT_SIZE.0 as usize / SPATIAL_FACTOR,
        };
        {
            let mut state = model.state.lock().unwrap_or_else(PoisonError::into_inner);
            let residency = model.plan_job(&state, grid, PLAN_TOKENS)?;
            tracing::info!(?residency, "qwen image load-time residency");
            if matches!(residency, Residency::Swap { .. }) {
                model.ensure_denoiser(&mut state, None, true)?;
                model.release(&mut state, Component::Denoiser);
                model.ensure_encoder(&mut state)?;
                model.release(&mut state, Component::Encoder);
                model.ensure_denoiser(&mut state, None, true)?;
                model.release(&mut state, Component::Denoiser);
            }
            model.ensure_encoder(&mut state)?;
            if residency == Residency::Resident {
                model.ensure_denoiser(&mut state, None, false)?;
            }
        }
        tracing::info!(
            ms = ms(began),
            model_id = %model.model_id,
            "qwen image model loaded"
        );
        Ok(model)
    }

    /// The job's activation peaks per phase.
    fn peaks(&self, state: &State, grid: LatentGrid, tokens: usize) -> JobPeaks {
        let (enc_h, enc_i) = (self.encoder_files.hidden, self.encoder_files.intermediate);
        let dit = &self.config.dit;
        let (h, mlp) = (dit.hidden(), dit.mlp_hidden());
        let kept = tokens.saturating_sub(super::prompt::DROP_IDX);
        let t = grid.cells();
        let row = 7 * h + 2 * mlp + dit.in_channels + dit.out_channels;
        let as_u64 = |v: usize| u64::try_from(v).unwrap_or(u64::MAX);
        JobPeaks {
            encode: as_u64(tokens * (8 * enc_h + 3 * enc_i) * 4),
            denoise: as_u64(2 * dit.num_layers * kept * h * 4 + (t + kept) * row * 4),
            decode: state.decoder.plan().working_set_bytes(grid),
        }
    }

    fn plan_desc(&self, state: &State, grid: LatentGrid, tokens: usize) -> PlanDesc {
        PlanDesc {
            mode: self.mode,
            working_set: self.working_set,
            reserve: self.reserve,
            footprint: self.footprint,
            peaks: self.peaks(state, grid, tokens),
        }
    }

    /// The job's residency, or a refusal naming the largest square side
    /// that fits.
    fn plan_job(
        &self,
        state: &State,
        grid: LatentGrid,
        tokens: usize,
    ) -> Result<Residency, ImageGenError> {
        plan(&self.plan_desc(state, grid, tokens)).map_err(|short| {
            let largest = (MIN_SIDE..=MAX_SIDE)
                .rev()
                .step_by(SIZE_MULTIPLE as usize)
                .find(|&side| {
                    let n = side as usize / SPATIAL_FACTOR;
                    plan(&self.plan_desc(state, LatentGrid { h: n, w: n }, tokens)).is_ok()
                })
                .unwrap_or(0);
            let (h, w) = grid.image_size();
            ImageGenError::TooLargeForMemory {
                width: u32::try_from(w).unwrap_or(u32::MAX),
                height: u32::try_from(h).unwrap_or(u32::MAX),
                needed_mib: short.needed.div_ceil(1 << 20),
                budget_mib: short.budget >> 20,
                largest_side: largest,
            }
        })
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
            "qwen image phase te_prefetch"
        );
        let phase = Instant::now();
        let before = device_used(self.ops.backend());
        let encoder = TextEncoder::load(
            &TextEncoderDesc {
                weights: &self.encoder_files.weights,
                config: &self.encoder_files.config,
                prefix: TE_PREFIX,
                taps: TapSet::LastPreNorm,
            },
            &self.ops,
        )?;
        if encoder.context_width() != self.config.dit.context_in_dim {
            return Err(ImageGenError::Checkpoint(format!(
                "the text encoder's features are {} wide but the transformer reads {}",
                encoder.context_width(),
                self.config.dit.context_in_dim
            )));
        }
        let device_bytes = delta(before, device_used(self.ops.backend()));
        tracing::info!(ms = ms(phase), device_bytes, "qwen image phase te_load");
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
    /// it drops the encoder first - whatever path led here (a cached prompt
    /// after a resident job, an encode that failed) - so a swapping host
    /// never holds both. An encoder read-ahead left from an earlier decode
    /// is cancelled: this import may evict its pages, and a later job must
    /// read them again rather than trust a stale result.
    fn ensure_denoiser(
        &self,
        state: &mut State,
        ahead: Option<ReadAhead>,
        swap: bool,
    ) -> Result<(), ImageGenError> {
        state.encoder_prefetch = None;
        if state.denoiser.is_some() {
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
            "qwen image phase dit_prefetch"
        );
        let phase = Instant::now();
        let before = device_used(self.ops.backend());
        let denoiser = QwenDit::load(
            &DitDesc {
                weights: &self.dit_path,
                config: &self.config.dit,
            },
            &self.ops,
        )?;
        let device_bytes = delta(before, device_used(self.ops.backend()));
        tracing::info!(ms = ms(phase), device_bytes, "qwen image phase dit_load");
        state.denoiser = Some(denoiser);
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
            Component::Denoiser => state.denoiser.take().is_some(),
        };
        if !dropped {
            return;
        }
        self.ops.trim_pool();
        let device_bytes = delta(device_used(self.ops.backend()), before);
        let label = match component {
            Component::Encoder => "te_drop",
            Component::Denoiser => "dit_drop",
        };
        tracing::info!(
            ms = ms(phase),
            device_bytes,
            phase = label,
            "qwen image phase drop"
        );
        state.events.push(ResidencyEvent::Dropped {
            component,
            device_bytes,
        });
    }

    /// The prompt's transformer-side features `(kept, context)`: from the
    /// cache, or encoded (loading the encoder, and in swap mode dropping
    /// the transformer first and the encoder after).
    fn features(
        &self,
        state: &mut State,
        ids: &super::prompt::PromptIds,
        residency: Residency,
    ) -> Result<(Vec<f32>, Option<ReadAhead>), ImageGenError> {
        let kept = ids.kept().to_vec();
        if let Some(at) = state.features.iter().position(|(k, _)| *k == kept) {
            let entry = state.features.remove(at);
            let features = entry.1.clone();
            state.features.insert(0, entry);
            tracing::info!(tokens = kept.len(), "qwen image prompt features cached");
            return Ok((features, None));
        }
        let swap = matches!(residency, Residency::Swap { .. });
        if swap {
            self.release(state, Component::Denoiser);
        }
        self.ensure_encoder(state)?;
        // While the encoder runs, read ahead as much of the transformer as
        // fits beside the encoder's phase; the rest waits for the drop.
        let mut ahead = None;
        if swap && state.denoiser.is_none() {
            let desc = self.plan_desc(state, LatentGrid { h: 1, w: 1 }, ids.ids.len());
            let room = desc
                .working_set
                .saturating_sub(desc.reserve)
                .saturating_sub(
                    self.footprint.encoder + self.footprint.decoder + desc.peaks.encode,
                );
            let (head, rest) = split_at(&self.dit_spans, room);
            if !head.is_empty() {
                tracing::info!(
                    overlap_mb = mb(total_bytes(&head)),
                    "qwen image dit read-ahead during encode"
                );
                ahead = Some((Prefetch::spawn("denoiser-overlap", head)?, rest));
            }
        }
        let phase = Instant::now();
        let encoded = self.encode(state, ids);
        tracing::info!(
            ms = ms(phase),
            tokens = ids.ids.len(),
            ok = encoded.is_ok(),
            "qwen image phase encode"
        );
        // Dropped whether the encode worked or not: a failed job must not
        // leave the encoder where the next job's transformer goes.
        if swap {
            self.release(state, Component::Encoder);
        }
        let features = encoded?;
        state.features.insert(0, (kept, features.clone()));
        state.features.truncate(FEATURE_CACHE);
        Ok((features, ahead))
    }

    /// The prompt's kept feature rows from the loaded encoder.
    fn encode(
        &self,
        state: &State,
        ids: &super::prompt::PromptIds,
    ) -> Result<Vec<f32>, ImageGenError> {
        let encoder = state
            .encoder
            .as_ref()
            .ok_or_else(|| ImageGenError::Backend("the text encoder is not loaded".to_string()))?;
        let all = encoder.prefill(&self.ops, &ids.ids)?;
        let width = encoder.context_width();
        Ok(all
            .get(ids.drop * width..)
            .ok_or_else(|| ImageGenError::Backend("the encoder returned too few rows".to_string()))?
            .to_vec())
    }

    /// Puts the model where a resident job on a larger host leaves a
    /// swapping one: the encoder loaded, the transformer not. Both are
    /// never loaded together here either.
    #[cfg(test)]
    pub(crate) fn stage_encoder_loaded(&self) -> Result<(), ImageGenError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.release(&mut state, Component::Denoiser);
        self.ensure_encoder(&mut state)
    }

    /// Runs a swap-mode encode of `prompt` with a vision placeholder id
    /// appended, which the encoder refuses after it is loaded: a real
    /// failure between the encoder's import and its drop.
    #[cfg(test)]
    pub(crate) fn failing_encode(&self, prompt: &str) -> Result<(), ImageGenError> {
        const IMAGE_PAD: u32 = 151_655;
        let mut ids = self.prompt.tokenize(prompt)?;
        ids.ids.push(IMAGE_PAD);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.features(
            &mut state,
            &ids,
            Residency::Swap {
                keep_denoiser: true,
            },
        )
        .map(|_| ())
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
}

impl ImageGenerator for QwenImage21Model {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn limits(&self) -> ImageLimits {
        let fixed = self.schedule.fixed_steps();
        ImageLimits {
            size_multiple: SIZE_MULTIPLE,
            min_side: MIN_SIDE,
            max_side: MAX_SIDE,
            default_size: DEFAULT_SIZE,
            default_steps: fixed.unwrap_or(BASE_STEPS),
            fixed_steps: fixed,
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one generation: plan, encode, prefix, the Euler loop, decode and encode, straight-line by construction"
    )]
    fn generate(
        &self,
        job: &ImageGenerationJob,
        progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        job.validate(&self.limits())?;
        if !job.input_references.is_empty() {
            return Err(ImageGenError::ReferenceImage(
                "reference images are not accepted: this model generates from text only \
                 (editing is not supported)"
                    .to_string(),
            ));
        }
        let began = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let _trim = TrimPool(&self.ops);
        let ids = self.prompt.tokenize(&job.prompt)?;
        let grid = LatentGrid {
            h: job.height as usize / SPATIAL_FACTOR,
            w: job.width as usize / SPATIAL_FACTOR,
        };
        let residency = self.plan_job(&state, grid, ids.ids.len())?;
        let steps = job.steps as usize;
        let mut sigmas = resolve(
            self.schedule.source(),
            &self.config.shift,
            steps,
            grid.cells(),
        )?;
        let draft = self.draft_steps;
        if draft >= steps && draft > 0 {
            return Err(ImageGenError::DraftTakesEveryStep {
                draft: u32::try_from(draft).unwrap_or(u32::MAX),
                steps: job.steps,
            });
        }
        // The draft grid is half the job's along each side (sizes sit on a
        // 32-pixel grid, so the latent grid is even), with its own ladder:
        // the dynamic shift follows the token count the steps run at.
        let draft_grid = LatentGrid {
            h: grid.h / 2,
            w: grid.w / 2,
        };
        // The last step replays a span of blocks the one before recorded,
        // when both run at full size.
        let reuse_span = self.final_reuse.clone().filter(|_| steps >= draft + 2);
        if self.final_reuse.is_some() && reuse_span.is_none() {
            return Err(ImageGenError::DraftTakesEveryStep {
                draft: u32::try_from(draft + 1).unwrap_or(u32::MAX),
                steps: job.steps,
            });
        }
        let draft_sigmas = if draft > 0 {
            resolve(
                self.schedule.source(),
                &self.config.shift,
                steps,
                draft_grid.cells(),
            )?
        } else {
            Vec::new()
        };
        tracing::info!(
            width = job.width,
            height = job.height,
            steps,
            tokens = ids.ids.len(),
            cells = grid.cells(),
            ?residency,
            "qwen image generation started"
        );

        let (features, ahead) = self.features(&mut state, &ids, residency)?;
        let swap = matches!(residency, Residency::Swap { .. });
        self.ensure_denoiser(&mut state, ahead, swap)?;
        let dit = state
            .denoiser
            .as_ref()
            .ok_or_else(|| ImageGenError::Backend("the transformer is not loaded".to_string()))?;

        let ops = &self.ops;
        let sync = || {
            ops.backend()
                .sync()
                .map_err(|e| ImageGenError::Backend(format!("qwen image sync: {e}")))
        };
        let phase = Instant::now();
        let kept = ids.ids.len() - ids.drop;
        let text = ops.upload_scratch("qwen text features", &features)?;
        let mut kv = dit.prefix(ops, text.view(), kept)?;
        drop(text);
        if draft > 0 {
            dit.prepare_steps(ops, &mut kv, &draft_sigmas[..draft], draft_grid)?;
        } else {
            dit.prepare_steps(ops, &mut kv, &sigmas[..steps], grid)?;
        }
        sync()?;
        tracing::info!(
            ms = ms(phase),
            tokens = kv.tokens(),
            "qwen image phase prefix"
        );

        let channels = self.config.dit.in_channels;
        let count = grid.cells() * channels;
        // One noise stream per job: the draft's starting noise first, then
        // the full-size noise its clean estimate is re-noised with, so a
        // seed fixes both.
        let draft_count = if draft > 0 {
            draft_grid.cells() * channels
        } else {
            0
        };
        let noise = gaussian(job.seed, draft_count + count);
        let (draft_noise, full_noise) = noise.split_at(draft_count);
        let mut latents = ops.upload_f32(
            "qwen latents",
            if draft > 0 { draft_noise } else { full_noise },
        )?;
        let velocity =
            ops.alloc_owned_f32("qwen velocity", grid.cells() * self.config.dit.out_channels)?;
        let residual = match reuse_span {
            Some(_) => Some(ops.alloc_owned_f32(
                "qwen reuse residual",
                grid.cells() * self.config.dit.hidden(),
            )?),
            None => None,
        };
        let denoise = Instant::now();
        for i in 0..steps {
            let phase = Instant::now();
            let drafting = i < draft;
            let (at, ladder, n) = if drafting {
                (draft_grid, &draft_sigmas, draft_count)
            } else {
                (grid, &sigmas, count)
            };
            let sigma = ladder[i];
            let reuse = match (&reuse_span, &residual) {
                (Some(span), Some(r)) if i + 2 == steps => Some(BlockReuse::Record {
                    span: span.clone(),
                    residual: r.view(),
                }),
                (Some(span), Some(r)) if i + 1 == steps => Some(BlockReuse::Replay {
                    span: span.clone(),
                    residual: r.view(),
                }),
                _ => None,
            };
            dit.velocity_with(
                ops,
                &kv,
                (latents.view(), at, sigma),
                velocity.view(),
                StepOpts { reuse },
            )?;
            let dt = euler_dt(ladder, i)?;
            ops.axpy(
                "qwen euler",
                latents.view(),
                latents.view(),
                velocity.view(),
                n,
                dt,
            )?;
            if i + 1 == draft {
                let state = ops.download("qwen draft latents", latents.view(), draft_count)?;
                let mut up = full_noise.to_vec();
                let handed = upsample2x_noisy(
                    &state,
                    (draft_grid.h, draft_grid.w, channels),
                    draft_sigmas[draft],
                    DRAFT_KEEP,
                    &mut up,
                );
                latents = ops.upload_f32("qwen latents", &up)?;
                // The full-size steps start where the hand-over put the
                // state, then walk the job's own ladder.
                sigmas[draft] = handed;
                dit.prepare_steps(ops, &mut kv, &sigmas[draft..steps], grid)?;
            }
            sync()?;
            tracing::info!(
                step = i + 1,
                steps,
                sigma,
                draft = drafting,
                ms = ms(phase),
                "qwen image phase step"
            );
            if let Some(report) = progress {
                report(ImageStepEvent {
                    step: u32::try_from(i).unwrap_or(u32::MAX),
                    total_steps: job.steps,
                    partial_image: None,
                });
            }
        }
        tracing::info!(ms = ms(denoise), steps, "qwen image denoising finished");
        drop(kv);
        drop(velocity);

        let mut read_after_decode = false;
        if let Residency::Swap {
            keep_denoiser: false,
        } = residency
        {
            self.release(&mut state, Component::Denoiser);
            // The transformer's room is free: the next job's encoder pages
            // read in behind this decode, but only when they fit beside the
            // decoder and the decode's working set - otherwise the read
            // would evict pages the decode is using.
            let decode_peak = self.peaks(&state, grid, ids.ids.len()).decode;
            let budget = self.working_set.saturating_sub(self.reserve);
            if self.footprint.encoder + self.footprint.decoder + decode_peak <= budget {
                state.encoder_prefetch = Some(Prefetch::spawn(
                    "encoder",
                    self.encoder_files.spans.clone(),
                )?);
            } else {
                tracing::info!(
                    decode_peak_mb = mb(decode_peak),
                    "qwen image encoder read-ahead deferred: it does not fit beside the decode"
                );
                read_after_decode = true;
            }
        }
        let loaded = match residency {
            Residency::Resident => {
                self.footprint.encoder + self.footprint.denoiser + self.footprint.decoder
            }
            Residency::Swap {
                keep_denoiser: true,
            } => self.footprint.denoiser + self.footprint.decoder,
            Residency::Swap {
                keep_denoiser: false,
            } => self.footprint.decoder,
        };
        let budget = self
            .working_set
            .saturating_sub(self.reserve)
            .saturating_sub(loaded);
        let phase = Instant::now();
        let rgba = state.decoder.decode(ops, latents.view(), grid, budget)?;
        drop(latents);
        tracing::info!(ms = ms(phase), "qwen image phase decode");
        // The read-ahead that could not run beside the decode runs after
        // it, once the decode's working buffers are released: a next job
        // that arrives after it finished finds the encoder's pages cached
        // (on an M4, the next job's ~3 s encoder read-ahead then takes 0).
        if read_after_decode
            && state.encoder.is_none()
            && self.footprint.encoder + self.footprint.decoder
                <= self.working_set.saturating_sub(self.reserve)
        {
            self.ops.trim_pool();
            state.encoder_prefetch = Some(Prefetch::spawn(
                "encoder",
                self.encoder_files.spans.clone(),
            )?);
        }

        let phase = Instant::now();
        let bytes = encode_image(
            &rgba,
            job.width,
            job.height,
            Channels::Rgba,
            job.output_format,
            job.output_compression,
        )?;
        tracing::info!(
            ms = ms(phase),
            total_ms = ms(began),
            bytes = bytes.len(),
            "qwen image phase png"
        );
        Ok(GeneratedImage {
            bytes,
            media_type: job.output_format.mime_type(),
            width: job.width,
            height: job.height,
            prompt_tokens: u32::try_from(kept).unwrap_or(u32::MAX),
            completion_tokens: u32::try_from(grid.cells() * steps).unwrap_or(u32::MAX),
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

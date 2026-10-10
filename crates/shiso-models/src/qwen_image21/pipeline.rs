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
use super::scheduler::{ScheduleSource, euler_dt, resolve};
use super::transformer::{DitDesc, QwenDit};
use super::vae::{LatentGrid, QwenVae, QwenVaeDesc, SPATIAL_FACTOR};
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

/// Where the sigma ladder comes from.
#[derive(Debug, Clone)]
enum Schedule {
    /// `model_index.json`'s `sample_sigmas`, run verbatim.
    Fixed(Vec<f32>),
    /// Raw nodes from the operator, shifted.
    Nodes(Vec<f32>),
    /// The base model's linear ladder, shifted.
    Linear,
}

impl Schedule {
    fn source(&self) -> ScheduleSource<'_> {
        match self {
            Self::Fixed(s) => ScheduleSource::Fixed(s),
            Self::Nodes(n) => ScheduleSource::Nodes(n),
            Self::Linear => ScheduleSource::Linear,
        }
    }

    fn fixed_steps(&self) -> Option<u32> {
        match self {
            Self::Fixed(v) | Self::Nodes(v) => Some(u32::try_from(v.len()).unwrap_or(u32::MAX)),
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

/// The operator's nodes, checked: they replace a schedule the checkpoint
/// does not carry, start at 1 (pure noise) and fall strictly within (0, 1].
fn check_nodes(nodes: &[f32], index_sigmas: bool) -> Result<(), ImageGenError> {
    if index_sigmas {
        return Err(ImageGenError::Checkpoint(
            "--image-sigmas names a schedule but model_index.json carries sample_sigmas; \
             two schedules for one checkpoint is ambiguous"
                .to_string(),
        ));
    }
    let falling = nodes.windows(2).all(|w| w[1] < w[0]);
    let in_range = nodes.iter().all(|&v| v.is_finite() && v > 0.0 && v <= 1.0);
    if nodes.first() != Some(&1.0) || !falling || !in_range {
        return Err(ImageGenError::Checkpoint(format!(
            "--image-sigmas {nodes:?} must start at 1.0 and fall strictly within (0, 1]"
        )));
    }
    Ok(())
}

impl QwenImage21Model {
    /// Locates the install, reads its documents, loads the decoder, and
    /// loads whatever the residency plan for a default job keeps loaded.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] for a missing document or weight file,
    /// a schedule conflict, or components that disagree; whatever a
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
        let schedule = match (&desc.sigmas, &config.sample_sigmas) {
            (Some(nodes), index) => {
                check_nodes(nodes, index.is_some())?;
                Schedule::Nodes(nodes.clone())
            }
            (None, Some(sigmas)) => Schedule::Fixed(sigmas.clone()),
            (None, None) => Schedule::Linear,
        };
        let model_id = match &schedule {
            Schedule::Fixed(_) => "qwen-image-2.1-turbo".to_string(),
            Schedule::Nodes(n) => format!("qwen-image-2.1-{}step", n.len()),
            Schedule::Linear => "qwen-image-2.1".to_string(),
        };
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
        let sigmas = resolve(
            self.schedule.source(),
            &self.config.shift,
            steps,
            grid.cells(),
        )?;
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
        dit.prepare_steps(ops, &mut kv, &sigmas[..steps], grid)?;
        sync()?;
        tracing::info!(
            ms = ms(phase),
            tokens = kv.tokens(),
            "qwen image phase prefix"
        );

        let count = grid.cells() * self.config.dit.in_channels;
        let latents = ops.upload_f32("qwen latents", &gaussian(job.seed, count))?;
        let velocity =
            ops.alloc_owned_f32("qwen velocity", grid.cells() * self.config.dit.out_channels)?;
        let denoise = Instant::now();
        for i in 0..steps {
            let phase = Instant::now();
            let sigma = sigmas[i];
            dit.velocity(ops, &kv, latents.view(), grid, sigma, velocity.view())?;
            ops.axpy(
                "qwen euler",
                latents.view(),
                latents.view(),
                velocity.view(),
                count,
                euler_dt(&sigmas, i)?,
            )?;
            sync()?;
            tracing::info!(
                step = i + 1,
                steps,
                sigma,
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
                    "qwen image encoder read-ahead skipped: it does not fit beside the decode"
                );
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

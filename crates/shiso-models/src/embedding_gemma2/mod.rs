//! The `EmbeddingGemma` 2 family: a small bidirectional encoder that places
//! text, pictures, video frames and audio in one vector space.
//!
//! The text side is its own stack - every layer attends both ways, sliding
//! layers over a symmetric window, the per-layer signal made from the input
//! rows alone with no token table of its own - and lives in [`encoder`]. The
//! media side is Gemma 4's: the same vision and audio towers, read from the
//! same multimodal projector layout, project straight into this encoder's
//! residual width. So this module owns the parts that are this family's and
//! no one else's: how a sequence is laid out, and how a pooled row becomes a
//! vector.
//!
//! # The sequence
//!
//! ```text
//! <bos> text... <|image> <|image|>*n <image|> ... <|audio> <|audio|>*n <audio|> ... <eos>
//! ```
//!
//! A video is its sampled frames, each laid out as a picture whose soft
//! tokens are `<|video|>`, back to back. Every position - the markers, the
//! brackets, the soft tokens - is in the mean: the reference pools over the
//! whole attention mask, prompt included.

mod config;
mod encoder;
mod weights;

#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::Arc;

use shiso_audio::AudioTower;
use shiso_core::{Backend, TokenId};
use shiso_modelfile::GgufFile;
use shiso_vision::{PreprocessSpec, ResizeRule, VisionItem, VisionTower};

pub use config::{EncoderConfig, GGUF_ARCH, LayerDims, Markers};
pub use encoder::Encoder;
pub use weights::EncoderWeights;

use crate::audio_ops::Ops;
use crate::audio_ops::load::download_f32;
use crate::audio_ops::store::TensorStore;
use crate::catalog::FamilyFacts;
use crate::embedder::{
    EmbedError, EmbedFrontend, EmbedPart, EmbedUsage, Embedder, EmbedderDesc, EmbedderInfo,
    Embedding, VideoSpec,
};
use crate::gemma4::{Gemma4AudioTower, Gemma4Tower, holds_audio_tower};

/// Family facts for `EmbeddingGemma` 2.
pub const FACTS: FamilyFacts = FamilyFacts {
    name: "embedding_gemma2",
};

/// The most positions one input may occupy.
///
/// The model card's context window. The container's `context_length` is the
/// rotary table's reach (262,144), not what the model was trained to pool
/// over, so it is not the number to hold an input to.
pub const MAX_TOKENS: usize = 8192;

/// The widths the model was trained to be truncated to, widest first, from
/// the model card's Matryoshka table.
const TRUNCATIONS: [usize; 4] = [768, 512, 256, 128];

/// Soft tokens one video frame costs: the video processor's
/// `max_soft_tokens`, half a picture's.
const VIDEO_FRAME_TOKENS: u32 = 140;
/// Video frames sampled per second of a clip, the video processor's `fps`.
const VIDEO_FPS: f64 = 1.0;
/// The most frames one clip contributes, the video processor's `max_frames`.
const VIDEO_MAX_FRAMES: usize = 32;

/// The task prompts the checkpoint's `config_sentence_transformers.json`
/// ships (`google/embeddinggemma-2`, revision `914f7f8`), for a deployment
/// that has only the GGUF: the container does not carry them, and a query
/// embedded without its prompt is a query from a different distribution.
const PROMPTS: [(&str, &str); 20] = [
    ("BitextMining", "task: search result | query: "),
    ("Classification", "task: classification | query: "),
    ("Clustering", "task: clustering | query: "),
    ("CodeRetrieval", "task: code retrieval | query: "),
    ("Document", "title: none | text: "),
    ("FactChecking", "task: fact checking | query: "),
    ("InstructionRetrieval", "task: code retrieval | query: "),
    ("MultilabelClassification", "task: classification | query: "),
    ("PairClassification", "task: sentence similarity | query: "),
    ("QuestionAnswering", "task: question answering | query: "),
    ("Reranking", "task: search result | query: "),
    ("Retrieval", "task: search result | query: "),
    ("Retrieval-document", "title: none | text: "),
    ("Retrieval-query", "task: search result | query: "),
    ("STS", "task: sentence similarity | query: "),
    ("SearchQuery", "task: search result | query: "),
    ("SentenceSimilarity", "task: sentence similarity | query: "),
    ("Summarization", "task: sentence similarity | query: "),
    ("document", "title: none | text: "),
    ("query", "task: search result | query: "),
];

/// Whether `path` is a container this family reads: a GGUF file declaring
/// [`GGUF_ARCH`].
#[must_use]
pub fn describes(path: &Path) -> bool {
    path.is_file()
        && GgufFile::open(path).is_ok_and(|file| file.metadata().architecture() == Some(GGUF_ARCH))
}

/// One position of a laid-out input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// A token whose row is read from the table.
    Token(TokenId),
    /// The next row of the encoded media, in order.
    Media,
}

/// A loaded `EmbeddingGemma` 2.
#[derive(Debug)]
pub struct EmbeddingGemma2 {
    info: EmbedderInfo,
    encoder: Encoder,
    vision: Option<Gemma4Tower>,
    audio: Option<Arc<dyn AudioTower>>,
    video: VideoSpec,
}

impl EmbeddingGemma2 {
    /// Opens the text model and, when given, the projector's towers.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Checkpoint`] for a container this family does not read
    /// or a tower whose width is not the encoder's; the device or tower error
    /// for a load failure.
    pub fn open(desc: &EmbedderDesc<'_>, backend: &Arc<dyn Backend>) -> Result<Self, EmbedError> {
        let store = TensorStore::open_gguf(desc.model).map_err(EmbedError::Audio)?;
        let file = store.gguf().ok_or_else(|| EmbedError::Checkpoint {
            what: format!("{} is not a GGUF file", desc.model.display()),
        })?;
        let config = EncoderConfig::from_gguf(file)?;
        let model_id = file
            .metadata()
            .get_str("general.name")
            .map(str::to_owned)
            .or_else(|| {
                desc.model
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "embedding".to_owned());
        let ops = Ops::new(Arc::clone(backend));
        let weights = EncoderWeights::load(&ops, &store, &config)?;
        let (vision, audio) = match desc.mmproj {
            Some(path) => open_towers(backend, path, config.hidden)?,
            None => (None, None),
        };
        let prompts = match desc.sidecar {
            Some(dir) => sidecar_prompts(dir)?,
            None => None,
        }
        .unwrap_or_else(|| {
            PROMPTS
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect()
        });
        if config.out_dim != TRUNCATIONS[0] {
            return Err(EmbedError::Checkpoint {
                what: format!(
                    "the output width is {}, not the {} this family's truncations start from",
                    config.out_dim, TRUNCATIONS[0]
                ),
            });
        }
        let info = EmbedderInfo {
            model_id,
            dim: config.out_dim,
            truncations: TRUNCATIONS.to_vec(),
            max_tokens: MAX_TOKENS,
            prompts,
        };
        tracing::info!(
            layers = config.n_layers(),
            hidden = config.hidden,
            vision = vision.is_some(),
            audio = audio.is_some(),
            "embedding model resident"
        );
        Ok(Self {
            info,
            encoder: Encoder::new(ops, config, weights),
            vision,
            audio,
            video: video_spec(),
        })
    }

    /// The marker ids this model lays sequences out with.
    #[must_use]
    pub const fn markers(&self) -> &Markers {
        &self.encoder.config().markers
    }

    /// One input's rows as the encoder reads them, and its positions by
    /// modality.
    fn rows(&self, parts: &[EmbedPart]) -> Result<(Vec<f32>, EmbedUsage), EmbedError> {
        if parts.is_empty() {
            return Err(EmbedError::Input {
                what: "an empty input".to_owned(),
            });
        }
        let (slots, usage) = self.layout(parts)?;
        let media = self.media_rows(parts)?;
        let h = self.encoder.config().hidden;
        // The reference scales a token row by `sqrt(hidden)` as it reads it,
        // and splices a media row in as the tower projected it.
        let scale = encoder::f32_of(h).sqrt();
        let mut input = vec![0f32; slots.len() * h];
        let mut next = media.chunks_exact(h);
        for (row, slot) in input.chunks_exact_mut(h).zip(&slots) {
            match slot {
                Slot::Token(id) => {
                    let id = usize::try_from(id.get()).unwrap_or(usize::MAX);
                    self.encoder.weights().tokens.row(id, row)?;
                    for v in row.iter_mut() {
                        *v *= scale;
                    }
                }
                Slot::Media => {
                    let src = next.next().ok_or_else(|| EmbedError::Input {
                        what: "fewer encoded media rows than soft tokens".to_owned(),
                    })?;
                    row.copy_from_slice(src);
                }
            }
        }
        Ok((input, usage))
    }

    /// One pass over packed inputs, their vectors appended to `out`.
    fn run_pack(
        &mut self,
        rows: &[f32],
        lengths: &[usize],
        usages: &mut Vec<EmbedUsage>,
        out: &mut Vec<Embedding>,
    ) -> Result<(), EmbedError> {
        let pooled = self.encoder.pooled(rows, lengths)?;
        for (pooled, usage) in pooled.iter().zip(usages.drain(..)) {
            out.push(Embedding {
                vector: self.encoder.project(pooled),
                usage,
            });
        }
        Ok(())
    }

    /// Lays `parts` out as positions, and counts them by modality.
    fn layout(&self, parts: &[EmbedPart]) -> Result<(Vec<Slot>, EmbedUsage), EmbedError> {
        let m = self.markers();
        let mut slots = vec![Slot::Token(m.bos)];
        let mut usage = EmbedUsage::default();
        let soft = |slots: &mut Vec<Slot>, open, close, count: usize| {
            slots.push(Slot::Token(open));
            slots.extend(std::iter::repeat_n(Slot::Media, count));
            slots.push(Slot::Token(close));
        };
        for part in parts {
            match part {
                EmbedPart::Text(ids) => slots.extend(ids.iter().map(|&t| Slot::Token(t))),
                EmbedPart::Image(item) => {
                    let n = self.vision_tokens(item)?;
                    usage.image = usage.image.saturating_add(count(n));
                    soft(&mut slots, m.image_open, m.image_close, n);
                }
                EmbedPart::Video(frames) => {
                    if frames.is_empty() {
                        return Err(EmbedError::Input {
                            what: "a video with no frames".to_owned(),
                        });
                    }
                    for frame in frames {
                        let n = self.vision_tokens(frame)?;
                        usage.video = usage.video.saturating_add(count(n));
                        soft(&mut slots, m.image_open, m.image_close, n);
                    }
                }
                EmbedPart::Audio(clip) => {
                    if self.audio.is_none() {
                        return Err(EmbedError::NoTower { modality: "audio" });
                    }
                    usage.audio = usage.audio.saturating_add(count(clip.tokens()));
                    soft(&mut slots, m.audio_open, m.audio_close, clip.tokens());
                }
            }
        }
        slots.push(Slot::Token(m.eos));
        let total = slots.len();
        if total > MAX_TOKENS {
            return Err(EmbedError::TooLong {
                tokens: total,
                max: MAX_TOKENS,
            });
        }
        usage.text = count(total)
            .saturating_sub(usage.image)
            .saturating_sub(usage.video)
            .saturating_sub(usage.audio);
        Ok((slots, usage))
    }

    /// Soft tokens `item` will encode to.
    fn vision_tokens(&self, item: &VisionItem) -> Result<usize, EmbedError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or(EmbedError::NoTower { modality: "vision" })?;
        Ok(item.pad_count(tower.preprocess().geometry.spatial_merge_size()))
    }

    /// Every media row the input splices in, in reading order, on the host.
    fn media_rows(&self, parts: &[EmbedPart]) -> Result<Vec<f32>, EmbedError> {
        let backend = self.encoder.ops().backend();
        let h = self.encoder.config().hidden;
        let mut rows = Vec::new();
        for part in parts {
            let encoded = match part {
                EmbedPart::Text(_) => continue,
                EmbedPart::Image(item) => self.encode_vision(std::slice::from_ref(item))?,
                EmbedPart::Video(frames) => self.encode_vision(frames)?,
                EmbedPart::Audio(clip) => {
                    let tower = self
                        .audio
                        .as_ref()
                        .ok_or(EmbedError::NoTower { modality: "audio" })?;
                    let out = tower.encode(backend, std::slice::from_ref(clip))?;
                    (out, clip.tokens())
                }
            };
            let (out, n) = encoded;
            rows.extend(download_f32(self.encoder.ops(), out.rows.view(), n * h)?);
        }
        Ok(rows)
    }

    /// One picture's or one clip's frames through the vision tower, and the
    /// rows it produced.
    fn encode_vision(
        &self,
        items: &[VisionItem],
    ) -> Result<(shiso_vision::EncodedRows, usize), EmbedError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or(EmbedError::NoTower { modality: "vision" })?;
        let out = tower.encode(self.encoder.ops().backend(), items)?;
        let n = usize::try_from(out.pad_total()).unwrap_or(usize::MAX);
        let want = items
            .iter()
            .map(|i| self.vision_tokens(i))
            .sum::<Result<usize, _>>()?;
        // The layout reserved `want` positions before the tower ran; a tower
        // that answered with another count would shift every row after it.
        if n != want {
            return Err(EmbedError::Core(shiso_core::CoreError::ShapeMismatch {
                field: "encoded media rows",
                expected: u64::try_from(want).unwrap_or(u64::MAX),
                got: u64::try_from(n).unwrap_or(u64::MAX),
            }));
        }
        Ok((out, n))
    }
}

impl Embedder for EmbeddingGemma2 {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }

    fn frontend(&self) -> EmbedFrontend {
        EmbedFrontend {
            image: self.vision.as_ref().map(|t| t.preprocess().clone()),
            video: self.vision.as_ref().map(|_| self.video.clone()),
            audio: self.audio.clone(),
        }
    }

    fn embed(&mut self, parts: &[EmbedPart]) -> Result<Embedding, EmbedError> {
        let (rows, usage) = self.rows(parts)?;
        let len = rows.len() / self.encoder.config().hidden;
        let mut out = Vec::with_capacity(1);
        self.run_pack(&rows, &[len], &mut vec![usage], &mut out)?;
        out.pop().ok_or_else(|| EmbedError::Input {
            what: "an empty input".to_owned(),
        })
    }

    fn embed_batch(&mut self, inputs: &[Vec<EmbedPart>]) -> Result<Vec<Embedding>, EmbedError> {
        let h = self.encoder.config().hidden;
        let mut out = Vec::with_capacity(inputs.len());
        let mut pack: Vec<f32> = Vec::new();
        let mut lengths: Vec<usize> = Vec::new();
        let mut usages: Vec<EmbedUsage> = Vec::new();
        let began = std::time::Instant::now();
        for parts in inputs {
            let (rows, usage) = self.rows(parts)?;
            let len = rows.len() / h;
            // A pack is closed before it would pass the longest single input
            // the model reads: the scratch is sized for that, and a longer
            // pass would grow it for every later request.
            if lengths.iter().sum::<usize>() + len > MAX_TOKENS {
                self.run_pack(&pack, &lengths, &mut usages, &mut out)?;
                pack.clear();
                lengths.clear();
            }
            pack.extend(rows);
            lengths.push(len);
            usages.push(usage);
        }
        if !lengths.is_empty() {
            self.run_pack(&pack, &lengths, &mut usages, &mut out)?;
        }
        tracing::debug!(
            inputs = inputs.len(),
            ms = began.elapsed().as_secs_f64() * 1e3,
            "embedded a batch"
        );
        Ok(out)
    }
}

/// The vision and audio towers a projector carries, either absent.
type Towers = (Option<Gemma4Tower>, Option<Arc<dyn AudioTower>>);

/// The towers the projector at `path` carries, each checked against the
/// encoder's width.
fn open_towers(
    backend: &Arc<dyn Backend>,
    path: &Path,
    hidden: usize,
) -> Result<Towers, EmbedError> {
    let file = GgufFile::open(path).map_err(|e| EmbedError::Checkpoint {
        what: format!("{}: {e}", path.display()),
    })?;
    let has_vision = file.metadata().get_bool("clip.has_vision_encoder") == Some(true);
    drop(file);
    let vision = if has_vision {
        let tower = Gemma4Tower::open(backend, path)?;
        if usize::try_from(tower.out_hidden()).ok() != Some(hidden) {
            return Err(EmbedError::Checkpoint {
                what: format!(
                    "the vision tower projects to {}, not the encoder's {hidden}",
                    tower.out_hidden()
                ),
            });
        }
        Some(tower)
    } else {
        None
    };
    let audio = if holds_audio_tower(path)? {
        let tower = Gemma4AudioTower::open(backend, path)?;
        if tower.out_hidden() != hidden {
            return Err(EmbedError::Checkpoint {
                what: format!(
                    "the audio tower projects to {}, not the encoder's {hidden}",
                    tower.out_hidden()
                ),
            });
        }
        let tower: Arc<dyn AudioTower> = Arc::new(tower);
        Some(tower)
    } else {
        None
    };
    if vision.is_none() && audio.is_none() {
        return Err(EmbedError::Checkpoint {
            what: format!(
                "{} carries neither a vision nor an audio tower",
                path.display()
            ),
        });
    }
    Ok((vision, audio))
}

/// The frame preprocess: a picture's, at a frame's soft-token budget.
fn video_spec() -> VideoSpec {
    let mut frames = PreprocessSpec::gemma4();
    let pool = frames.geometry.spatial_merge_size();
    let pool = u32::try_from(pool * pool).unwrap_or(u32::MAX);
    frames.resize = ResizeRule::PatchBudget {
        max_patches: VIDEO_FRAME_TOKENS.saturating_mul(pool),
    };
    VideoSpec {
        frames,
        fps: VIDEO_FPS,
        max_frames: VIDEO_MAX_FRAMES,
    }
}

/// The task prompts of a sentence-transformers configuration in `dir`, or
/// `None` when the directory has none.
fn sidecar_prompts(dir: &Path) -> Result<Option<Vec<(String, String)>>, EmbedError> {
    let path = dir.join("config_sentence_transformers.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(EmbedError::Checkpoint {
                what: format!("{}: {e}", path.display()),
            });
        }
    };
    let doc: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| EmbedError::Checkpoint {
            what: format!("{}: {e}", path.display()),
        })?;
    let Some(prompts) = doc.get("prompts").and_then(serde_json::Value::as_object) else {
        return Ok(None);
    };
    prompts
        .iter()
        .map(|(name, prompt)| {
            prompt
                .as_str()
                .map(|p| (name.clone(), p.to_owned()))
                .ok_or_else(|| EmbedError::Checkpoint {
                    what: format!("{}: prompt {name} is not a string", path.display()),
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// A count as the usage counters hold it.
fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

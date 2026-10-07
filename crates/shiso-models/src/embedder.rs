//! Embedding models, as everything above them sees one.
//!
//! An embedding model answers one question per input: where in its vector
//! space does this text, picture, clip or mixture of them land. Nothing about
//! that is a conversation, so it is not a [`shiso_core::Model`]: there is no
//! session, no cache that outlives the call and no logits row. What the server
//! hands over is a list of parts in reading order and what it gets back is one
//! pooled vector.
//!
//! The split of work between this contract and the server is the split between
//! what is a fact about a checkpoint and what is a fact about the wire:
//!
//! - **The family** owns the token layout - which ids open and close a
//!   sequence, which bracket a picture or a clip, how many soft tokens a frame
//!   costs - because a sequence laid out any other way is one the model was
//!   never trained on and still answers fluently.
//! - **The server** owns tokenizing the text, decoding the media into the
//!   items the family's [`EmbedFrontend`] describes, and everything a client
//!   chose: the task prompt, the truncated width, the encoding.
//!
//! [`matryoshka`] is here rather than in the server because it is the one
//! post-processing step every family with nested dimensions shares, and the
//! tests that hold a family to its reference need it too.

use std::path::Path;
use std::sync::Arc;

use shiso_audio::{AudioClip, AudioError, AudioTower};
use shiso_core::{Backend, CoreError, TokenId};
use shiso_vision::{PreprocessSpec, VisionError, VisionItem};

/// One span of an embedding input, in reading order.
#[derive(Debug, Clone)]
pub enum EmbedPart {
    /// Text the server already tokenized, with no special tokens: the family
    /// adds its own sequence markers.
    Text(Vec<TokenId>),
    /// One preprocessed picture.
    Image(VisionItem),
    /// A clip's sampled frames, each preprocessed as a picture under
    /// [`VideoSpec::frames`].
    Video(Vec<VisionItem>),
    /// One framed audio clip.
    Audio(AudioClip),
}

/// How a family wants a video turned into frames.
#[derive(Debug, Clone)]
pub struct VideoSpec {
    /// The preprocess each sampled frame runs through.
    pub frames: PreprocessSpec,
    /// Frames sampled per second of the clip.
    pub fps: f64,
    /// The most frames one clip may contribute; a longer clip is resampled
    /// uniformly down to this many rather than cut short, so the end of a clip
    /// is represented as well as its start.
    pub max_frames: usize,
}

impl VideoSpec {
    /// The frames of a clip of `frames` frames at `native_fps` this family
    /// reads, by index, in order.
    ///
    /// The reference's sampler, step for step: one frame every
    /// `native_fps / fps` frames by truncation, as many as the clip's duration
    /// at `fps` (at least one), and past [`VideoSpec::max_frames`] that list
    /// cut down to an evenly spaced subset including both ends. A decoder that
    /// already resampled to `fps` reports `native_fps == fps`, and every frame
    /// is kept.
    #[must_use]
    pub fn sample(&self, frames: usize, native_fps: f64) -> Vec<usize> {
        if frames == 0 {
            return Vec::new();
        }
        let step = native_fps / self.fps;
        let duration = f64_of(frames) / native_fps;
        let wanted = usize_of((duration * self.fps).floor()).max(1);
        let picked: Vec<usize> = (0..wanted)
            .map(|i| usize_of((f64_of(i) * step).floor()).min(frames - 1))
            .collect();
        if picked.len() <= self.max_frames || self.max_frames == 0 {
            return picked;
        }
        // numpy's `linspace(0, n - 1, k).astype(int)`.
        let last = f64_of(picked.len() - 1);
        let k = self.max_frames;
        (0..k)
            .map(|j| {
                let at = if k == 1 {
                    0.0
                } else {
                    last * f64_of(j) / f64_of(k - 1)
                };
                picked[usize_of(at.floor()).min(picked.len() - 1)]
            })
            .collect()
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "frame counts stay far below 2^52, where f64 is exact"
)]
fn f64_of(n: usize) -> f64 {
    n as f64
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "callers pass a floor of a non-negative frame position"
)]
fn usize_of(x: f64) -> usize {
    if x.is_finite() && x > 0.0 {
        x as usize
    } else {
        0
    }
}

/// What the server needs to turn a request's media into [`EmbedPart`]s,
/// taken once at boot.
///
/// A value rather than methods on [`Embedder`] because the embedder lives on
/// its own worker thread and the decoding runs on the request's: a picture is
/// resized while the device is busy with someone else's input, not after.
#[derive(Debug, Clone, Default)]
pub struct EmbedFrontend {
    /// The picture preprocess, when the checkpoint carries a vision tower.
    pub image: Option<PreprocessSpec>,
    /// The video sampling and frame preprocess; `None` without a vision tower.
    pub video: Option<VideoSpec>,
    /// The audio tower, for framing a clip into mel rows and counting its
    /// tokens. Encoding happens on the worker; this is only ever read.
    pub audio: Option<Arc<dyn AudioTower>>,
}

/// Tokens one input occupied, by what they carried.
///
/// The text count includes the family's own markers - the sequence's
/// opening and closing ids and the brackets around each media run - because
/// those are positions the model read and the context budget paid for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbedUsage {
    /// Text tokens and markers.
    pub text: u32,
    /// Soft tokens of pictures.
    pub image: u32,
    /// Soft tokens of video frames.
    pub video: u32,
    /// Soft tokens of audio clips.
    pub audio: u32,
}

impl EmbedUsage {
    /// Every position the input occupied.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.text
            .saturating_add(self.image)
            .saturating_add(self.video)
            .saturating_add(self.audio)
    }
}

/// One input's place in the embedding space.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    /// The pooled vector at the model's native width, NOT normalized: what a
    /// client asked for - a width, unit length - is applied by [`matryoshka`].
    pub vector: Vec<f32>,
    /// The positions the input occupied.
    pub usage: EmbedUsage,
}

/// The facts about a loaded embedding model that do not change while it runs.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedderInfo {
    /// The name the checkpoint answers to when the operator names none.
    pub model_id: String,
    /// Width of [`Embedding::vector`].
    pub dim: usize,
    /// The widths the model was trained to be truncated to, widest first and
    /// including [`EmbedderInfo::dim`]. A width outside this list is refused
    /// rather than served: a prefix the model was not trained to make
    /// meaningful still compares, and ranks badly without saying so.
    pub truncations: Vec<usize>,
    /// The most positions one input may occupy.
    pub max_tokens: usize,
    /// The task prompts the checkpoint was trained with, by name, as its
    /// sentence-transformers configuration spells them. Prepended to text.
    pub prompts: Vec<(String, String)>,
}

/// Why an input could not be embedded, or a checkpoint could not be opened.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The checkpoint is not one this family reads, or is malformed.
    #[error("embedding checkpoint: {what}")]
    Checkpoint {
        /// What was wrong, naming the file or tensor.
        what: String,
    },
    /// The input itself cannot be embedded: a caller's mistake.
    #[error("{what}")]
    Input {
        /// What was wrong with it.
        what: String,
    },
    /// The input occupies more positions than the model reads.
    #[error("the input is {tokens} tokens; this model reads at most {max}")]
    TooLong {
        /// Positions the input would occupy.
        tokens: usize,
        /// The model's limit.
        max: usize,
    },
    /// The input carries a modality this checkpoint has no tower for.
    #[error("this embedding model was loaded without a {modality} encoder")]
    NoTower {
        /// The modality, as the wire spells it.
        modality: &'static str,
    },
    /// A device or engine failure.
    #[error(transparent)]
    Core(#[from] CoreError),
    /// The vision tower refused or failed.
    #[error(transparent)]
    Vision(#[from] VisionError),
    /// The audio tower refused or failed.
    #[error(transparent)]
    Audio(#[from] AudioError),
}

impl EmbedError {
    /// Whether the failure is the caller's input rather than the server.
    #[must_use]
    pub const fn is_input(&self) -> bool {
        matches!(
            self,
            Self::Input { .. } | Self::TooLong { .. } | Self::NoTower { .. }
        )
    }
}

/// A loaded embedding model.
///
/// `Send` because it is built on the boot thread and run on its worker;
/// `&mut self` on [`Embedder::embed`] because the worker is the only caller
/// and a model may keep scratch it reuses between inputs.
pub trait Embedder: Send + std::fmt::Debug {
    /// What the model is.
    fn info(&self) -> &EmbedderInfo;

    /// What the server needs to decode media for this model.
    fn frontend(&self) -> EmbedFrontend;

    /// Embeds one input.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Input`], [`EmbedError::TooLong`] or
    /// [`EmbedError::NoTower`] for an input this model cannot read, and the
    /// device or tower error for a failure while running it.
    fn embed(&mut self, parts: &[EmbedPart]) -> Result<Embedding, EmbedError>;

    /// Embeds several inputs, answering in their order.
    ///
    /// Each vector is the one [`Embedder::embed`] gives its input alone; a
    /// family that can run several inputs in one pass overrides this to do
    /// so, and the default runs them one after another.
    ///
    /// # Errors
    ///
    /// The first input's refusal or the device's failure, as
    /// [`Embedder::embed`] reports it.
    fn embed_batch(&mut self, inputs: &[Vec<EmbedPart>]) -> Result<Vec<Embedding>, EmbedError> {
        inputs.iter().map(|parts| self.embed(parts)).collect()
    }
}

/// Where an embedding checkpoint is.
#[derive(Debug, Clone, Copy)]
pub struct EmbedderDesc<'a> {
    /// The text model: a GGUF file.
    pub model: &'a Path,
    /// The multimodal projector holding the vision and audio towers, when the
    /// operator gave one. Without it the model embeds text alone.
    pub mmproj: Option<&'a Path>,
    /// A directory that may hold the checkpoint's sentence-transformers
    /// configuration, whose task prompts then replace the family's defaults.
    pub sidecar: Option<&'a Path>,
}

/// Whether `model` is a checkpoint some embedding family reads.
///
/// Asked before a server decides what kind of model it is booting, so it
/// answers from the container's own declaration and opens nothing else.
#[must_use]
pub fn is_embedder(model: &Path) -> bool {
    crate::embedding_gemma2::describes(model)
}

/// Opens the embedding model `desc` names on `backend`.
///
/// # Errors
///
/// [`EmbedError::Checkpoint`] when no embedding family reads the checkpoint
/// or it is malformed, and the device or tower error for a load failure.
pub fn open_embedder(
    desc: &EmbedderDesc<'_>,
    backend: &Arc<dyn Backend>,
) -> Result<Box<dyn Embedder>, EmbedError> {
    if crate::embedding_gemma2::describes(desc.model) {
        return Ok(Box::new(crate::embedding_gemma2::EmbeddingGemma2::open(
            desc, backend,
        )?));
    }
    Err(EmbedError::Checkpoint {
        what: format!(
            "{} is not a checkpoint any embedding family reads",
            desc.model.display()
        ),
    })
}

/// `vector` cut to its first `dim` values and scaled to unit length.
///
/// The cut comes first: a nested-dimension model is trained so that every
/// prefix it lists is a usable vector on its own, and the prefix of a
/// normalized vector is not itself normalized. A zero vector stays zero
/// rather than becoming NaN.
///
/// # Errors
///
/// [`EmbedError::Input`] when `dim` is zero or wider than `vector`.
pub fn matryoshka(vector: &[f32], dim: usize) -> Result<Vec<f32>, EmbedError> {
    let prefix = vector
        .get(..dim)
        .filter(|p| !p.is_empty())
        .ok_or_else(|| EmbedError::Input {
            what: format!(
                "dimensions {dim} is outside 1..={} for this model",
                vector.len()
            ),
        })?;
    let norm = prefix
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    // The same floor sentence-transformers' Normalize passes to
    // `torch.nn.functional.normalize`, so a near-zero vector is scaled the
    // way the reference scales it rather than blown up.
    let inv = 1.0 / norm.max(1e-12);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a unit-length component is well inside f32"
    )]
    Ok(prefix
        .iter()
        .map(|v| (f64::from(*v) * inv) as f32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matryoshka_cuts_then_normalizes() {
        let full = [3.0, 4.0, 12.0];
        let cut = matryoshka(&full, 2).expect("a prefix of the vector");
        assert!((cut[0] - 0.6).abs() < 1e-7 && (cut[1] - 0.8).abs() < 1e-7);
        let whole = matryoshka(&full, 3).expect("the whole vector");
        assert!((whole[2] - 12.0 / 13.0).abs() < 1e-7);
    }

    #[test]
    fn matryoshka_refuses_widths_it_cannot_cut() {
        assert!(matryoshka(&[1.0, 2.0], 0).is_err());
        assert!(matryoshka(&[1.0, 2.0], 3).is_err());
    }

    fn spec(fps: f64, max_frames: usize) -> VideoSpec {
        VideoSpec {
            frames: PreprocessSpec::gemma4(),
            fps,
            max_frames,
        }
    }

    #[test]
    fn video_sampling_is_the_reference_samplers() {
        // The reference fixture: 32 frames at 8 fps, 1 fps -> 0, 8, 16, 24.
        assert_eq!(spec(1.0, 32).sample(32, 8.0), vec![0, 8, 16, 24]);
        // Already resampled by the decoder: every frame.
        assert_eq!(spec(1.0, 32).sample(5, 1.0), vec![0, 1, 2, 3, 4]);
        // Shorter than one sample period: one frame.
        assert_eq!(spec(1.0, 32).sample(3, 30.0), vec![0]);
        // 40 one-second samples cut to 4: linspace(0, 39, 4) = 0, 13, 26, 39.
        assert_eq!(spec(1.0, 4).sample(40, 1.0), vec![0, 13, 26, 39]);
        assert!(spec(1.0, 4).sample(0, 1.0).is_empty());
    }

    #[test]
    fn matryoshka_keeps_a_zero_vector_finite() {
        let zero = matryoshka(&[0.0; 4], 4).expect("a zero vector");
        assert!(zero.iter().all(|v| *v == 0.0));
    }
}

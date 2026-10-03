//! Speech recognizer of one audio-LLM architecture: a windowed transformer
//! audio tower over 1-second conv-stem chunks, whose rows replace the audio
//! placeholders of a chat prompt, and a dense decoder with a tied head that
//! writes `language X<asr_text>transcript`.
//!
//! Two checkpoints share the architecture - the base model and a fine-tune
//! with a retrained decoder - and every container they ship in (HF, MLX, a
//! GGUF text file with its mmproj) reads through one name table and one
//! config reader ([`config`]). Each constant says where its value comes
//! from: the checkpoint's own configs, or the reference implementation's
//! processor, encoder and parser, measured against it.
//!
//! The model has no streaming encoder: a clip is transcribed in chunks of at
//! most 35 s (the `stream` module), each one prompt and one greedy decode, and a
//! token's time is interpolated inside its chunk. Times are therefore
//! chunk-accurate, not word-accurate. A whole clip known up front can instead
//! go through the decoder several chunks at a time ([`batch`]), with the same
//! ids, as an offline job (the `offline` module). That holds only while every
//! decoder weight is one the audio set's own kernels read, whose rows sum
//! alone at any row count; an export with any other weight keeps the single
//! path ([`Qwen3Asr::with_offline_batch`]).

pub mod batch;
pub mod config;
pub mod context;
pub mod decoder;
pub mod encoder;
mod names;
mod offline;
pub mod prompt;
mod stream;
pub mod weights;

use crate::catalog::FamilyFacts;

/// Family facts for Qwen3 ASR.
pub const FACTS: FamilyFacts = FamilyFacts { name: "qwen3_asr" };

use std::path::Path;
use std::sync::{Arc, Mutex};

use shiso_audio::chunks::Span;
use shiso_audio::{
    AudioError, ContextEncoder, Language, LogScale, MelSpec, OfflineJob, PieceVocab,
    RecognizerStream, SpeechRecognizer, StreamOptions,
};
use shiso_core::Backend;

use crate::audio_ops::Ops;
use config::{Qwen3AsrConfig, language_name, languages, locate, read_merges, read_vocab};
use weights::{QwenAsrWeights, TextWeights};

/// What every stream of one loaded checkpoint shares.
#[derive(Debug)]
pub(crate) struct Model {
    pub(crate) ops: Ops,
    pub(crate) config: Qwen3AsrConfig,
    pub(crate) weights: QwenAsrWeights,
    pub(crate) vocab: Arc<PieceVocab>,
    /// Device memory of streams that have closed, for the next to take
    /// (see `stream::Memory`), at most `stream::POOL_STREAMS` of them.
    pub(crate) spare: Mutex<Vec<stream::Memory>>,
    /// Device memory of the last batch, for the next (see
    /// `batch::give_back`), at most `batch::POOL_BATCHES` of them.
    pub(crate) batch_spare: Mutex<Option<batch::BatchMemory>>,
}

/// A loaded checkpoint of this architecture.
#[derive(Debug)]
pub struct Qwen3Asr {
    model: Arc<Model>,
    mel: MelSpec,
    /// Spans an offline step transcribes, as configured; 0 is no offline
    /// path.
    offline_batch: usize,
    /// The first decoder weight whose rows do not sum alone, by name: with
    /// one, a batch could give other ids than the single path, so the model
    /// has no offline path whatever the batch.
    outside: Option<String>,
    /// The encoder the system turn's context is tokenized with; `None` when
    /// the export carries no BPE merges, so a request that sends a prompt or
    /// keywords is refused rather than tokenized wrong (see
    /// [`config::read_merges`]).
    context: Option<Arc<context::QwenContextEncoder>>,
}

/// The decoder's linears by their role, embedding (the tied head) first.
fn text_linears(text: &TextWeights) -> Vec<(String, &crate::audio_ops::Linear)> {
    let mut out = vec![("embed".to_owned(), &text.embed)];
    for (i, layer) in text.layers.iter().enumerate() {
        for (role, linear) in [
            ("q", &layer.q),
            ("k", &layer.k),
            ("v", &layer.v),
            ("o", &layer.o),
            ("gate", &layer.gate),
            ("up", &layer.up),
            ("down", &layer.down),
        ] {
            out.push((format!("layers.{i}.{role}"), linear));
        }
    }
    out
}

/// The name of the first of `linears` (name, whether its rows sum alone)
/// whose rows do not, or `None` when every one does.
pub(crate) fn first_outside(linears: impl IntoIterator<Item = (String, bool)>) -> Option<String> {
    linears
        .into_iter()
        .find_map(|(name, alone)| (!alone).then_some(name))
}

/// The spans an offline step takes: `configured`, or none when a weight
/// `outside` the audio set would let a batch change the ids.
pub(crate) fn offline_batch(configured: usize, outside: Option<&str>) -> usize {
    if outside.is_some() { 0 } else { configured }
}

/// The language `options` force, by the model's name for it.
// None: the language is left to the model (`Language::Auto`).
fn forced_of(options: &StreamOptions) -> Result<Option<&'static str>, AudioError> {
    match &options.language {
        Language::Auto => Ok(None),
        Language::Tag(tag) => language_name(tag)
            .map(Some)
            .ok_or_else(|| AudioError::UnknownLanguage { tag: tag.clone() }),
    }
}

/// The context `options` carry, as the raw ids the prompt takes.
fn context_of(options: &StreamOptions) -> Arc<[u32]> {
    options.context.ids().iter().map(|id| id.get()).collect()
}

/// The keywords the context of `options` was made from, as word sequences.
fn keywords_of(options: &StreamOptions) -> Arc<[Vec<String>]> {
    options.context.keywords().into()
}

/// Opens the checkpoint at `path` on `backend`; `mmproj` is `--asr-mmproj`,
/// the tower file of a text GGUF. An HF or MLX export with no BPE merges
/// (see [`config::read_merges`]) still loads, logging a warning: it just
/// takes no context, and a request that sends a prompt or keywords is
/// refused rather than silently dropped.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming what is broken in an export of this
/// family (see [`config::locate`]); a device error.
// None: the path is another family's, and the registry tries the next one.
pub fn open(
    path: &Path,
    mmproj: Option<&Path>,
    backend: &Arc<dyn Backend>,
) -> Result<Option<Qwen3Asr>, AudioError> {
    let Some(located) = locate(path, mmproj)? else {
        return Ok(None);
    };
    let stores = located.stores()?;
    let config = Qwen3AsrConfig::read(&located, &stores)?;
    let ops = Ops::new(Arc::clone(backend));
    let weights = QwenAsrWeights::load(&ops, &located, &config, &stores)?;
    let vocab = Arc::new(read_vocab(&located, &stores)?);
    // The ordinary pieces only, the same ids `read_vocab` gave them.
    let pieces: Vec<String> = (0..config::FIRST_SPECIAL)
        .map(|id| vocab.piece(id).map(str::to_owned))
        .collect::<Option<_>>()
        .ok_or_else(|| AudioError::Checkpoint {
            what: "the vocabulary is shorter than its specials' first id".into(),
        })?;
    let context = if let Some(merges) = read_merges(&located, &stores)? {
        Some(Arc::new(context::QwenContextEncoder::new(
            &pieces, &merges,
        )?))
    } else {
        tracing::warn!(
            "this export carries no BPE merges (no merges.txt or tokenizer.json beside it); \
             a request's prompt or keywords will be refused rather than tokenized"
        );
        None
    };
    // The worker frames the whole clip once; the floor-and-rescale belongs to
    // each chunk's own rows (the reference extracts every chunk's features on
    // their own), so the frames it computes stop at the log and
    // the stream finishes each chunk.
    let mut mel = config::mel_spec();
    mel.log_scale = LogScale::Log10;
    let outside = first_outside(
        text_linears(&weights.text)
            .into_iter()
            .map(|(name, linear)| (name, linear.rows_sum_alone())),
    );
    if let Some(name) = &outside {
        tracing::info!(
            weight = %name,
            "a decoder weight is outside the audio kernels, whose rows sum alone at any \
             row count; this export has no batched offline path and transcribes one chunk \
             at a time"
        );
    }
    Ok(Some(Qwen3Asr {
        offline_batch: 0,
        outside,
        context,
        model: Arc::new(Model {
            ops,
            config,
            weights,
            vocab,
            spare: Mutex::new(Vec::new()),
            batch_spare: Mutex::new(None),
        }),
        mel,
    }))
}

impl Qwen3Asr {
    /// Transcribes offline jobs `batch` spans a step; 0, the default, is no
    /// offline path. An export with a decoder weight outside the audio set
    /// keeps the single path whatever the batch, since a batch of it could
    /// give other ids.
    #[must_use]
    pub fn with_offline_batch(mut self, batch: usize) -> Self {
        self.offline_batch = batch;
        self
    }
}

impl SpeechRecognizer for Qwen3Asr {
    fn mel_spec(&self) -> &MelSpec {
        &self.mel
    }

    fn vocab(&self) -> &Arc<PieceVocab> {
        &self.model.vocab
    }

    /// One mel frame: a token's frame is a mel frame interpolated inside its
    /// chunk.
    fn frame_secs(&self) -> f64 {
        stream::MEL_FRAME_SECS
    }

    fn languages(&self) -> Vec<String> {
        languages()
            .iter()
            .map(|(code, _)| (*code).to_owned())
            .collect()
    }

    /// None: the tower sees each chunk whole.
    fn lookaheads(&self) -> &[usize] {
        &[]
    }

    fn chunks_at_cuts(&self) -> bool {
        true
    }

    fn is_special(&self, id: u32) -> bool {
        config::is_special(id)
    }

    fn context_encoder(&self) -> Option<Arc<dyn ContextEncoder>> {
        let concrete = Arc::clone(self.context.as_ref()?);
        let encoder: Arc<dyn ContextEncoder> = concrete;
        Some(encoder)
    }

    fn open_stream(
        &self,
        options: &StreamOptions,
    ) -> Result<Box<dyn RecognizerStream>, AudioError> {
        Ok(Box::new(stream::QwenStream::new(
            Arc::clone(&self.model),
            forced_of(options)?,
            context_of(options),
            keywords_of(options),
            options.cuts.as_deref(),
        )?))
    }

    fn batches_offline(&self) -> usize {
        offline_batch(self.offline_batch, self.outside.as_deref())
    }

    fn open_offline(
        &self,
        options: &StreamOptions,
        spans: Vec<Span>,
    ) -> Result<Box<dyn OfflineJob>, AudioError> {
        let batch = self.batches_offline();
        if batch == 0 {
            return Err(AudioError::BadSpec {
                reason: match &self.outside {
                    Some(name) => format!(
                        "decoder weight {name} is outside the audio kernels; \
                         stream the clip instead"
                    ),
                    None => "no offline batch is configured; stream the clip instead".into(),
                },
            });
        }
        Ok(Box::new(offline::QwenOffline::new(
            Arc::clone(&self.model),
            forced_of(options)?,
            context_of(options),
            keywords_of(options),
            spans,
            batch,
        )?))
    }
}

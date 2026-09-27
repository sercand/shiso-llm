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
//! ids.

pub mod batch;
pub mod config;
pub mod decoder;
pub mod encoder;
mod names;
pub mod prompt;
mod stream;
pub mod weights;

use std::path::Path;
use std::sync::{Arc, Mutex};

use shiso_audio::{
    AudioError, Language, LogScale, MelSpec, PieceVocab, RecognizerStream, SpeechRecognizer,
    StreamOptions,
};
use shiso_core::Backend;

use crate::audio_ops::Ops;
use config::{Qwen3AsrConfig, language_name, languages, locate, read_vocab};
use weights::QwenAsrWeights;

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
}

/// Opens the checkpoint at `path` on `backend`; `mmproj` is `--asr-mmproj`,
/// the tower file of a text GGUF.
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
    // The worker frames the whole clip once; the floor-and-rescale belongs to
    // each chunk's own rows (the reference extracts every chunk's features on
    // their own), so the frames it computes stop at the log and
    // the stream finishes each chunk.
    let mut mel = config::mel_spec();
    mel.log_scale = LogScale::Log10;
    Ok(Some(Qwen3Asr {
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

    fn open_stream(
        &self,
        options: &StreamOptions,
    ) -> Result<Box<dyn RecognizerStream>, AudioError> {
        let forced = match &options.language {
            Language::Auto => None,
            Language::Tag(tag) => Some(
                language_name(tag)
                    .ok_or_else(|| AudioError::UnknownLanguage { tag: tag.clone() })?,
            ),
        };
        Ok(Box::new(stream::QwenStream::new(
            Arc::clone(&self.model),
            forced,
            options.cuts.as_deref(),
        )?))
    }
}

//! Nemotron streaming transducer: a cache-aware conformer encoder with a
//! language prompt, an LSTM prediction network and a joint network, decoded
//! greedily.
//!
//! Every export this checkpoint ships in reads through one name table and one
//! config reader ([`config`]); the device path is the same for all of them.

pub mod config;
pub mod encoder;
mod names;
mod stream;
pub mod transducer;
pub mod weights;

use std::path::Path;
use std::sync::Arc;

use shiso_audio::{
    AudioError, Filterbank, Framing, LogScale, MelScale, MelSpec, PieceVocab, Preemphasis,
    RecognizerStream, Spectrum, SpeechRecognizer, StreamOptions, Window,
};
use shiso_core::Backend;

use crate::audio_ops::Ops;
use config::{AsrConfig, read_filterbank, read_vocab};
use encoder::EncoderState;
use transducer::{DecoderState, Transducer};
use weights::AsrWeights;

/// The frontend constants every export of this checkpoint states alike: a
/// 25 ms symmetric Hann window every 10 ms over a 512-point FFT of 16 kHz
/// audio, 0.97 pre-emphasis, a power spectrum and a 2^-24 log guard.
const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 512;
const WIN: usize = 400;
const HOP: usize = 160;

/// What every stream of one loaded checkpoint shares.
#[derive(Debug)]
pub(crate) struct Model {
    pub(crate) ops: Ops,
    pub(crate) config: AsrConfig,
    pub(crate) weights: AsrWeights,
}

/// A loaded Nemotron transducer.
#[derive(Debug)]
pub struct NemotronAsr {
    model: Arc<Model>,
    mel: MelSpec,
    vocab: Arc<PieceVocab>,
    special: Vec<bool>,
}

/// Opens the checkpoint at `path` on `backend`.
///
/// `None` is not a refusal: the path is another family's, and the registry
/// tries the next one.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming what is broken in an export of this
/// family; a device error.
pub fn open(path: &Path, backend: &Arc<dyn Backend>) -> Result<Option<NemotronAsr>, AudioError> {
    let Some(located) = config::locate(path)? else {
        return Ok(None);
    };
    let store = located.store()?;
    let config = AsrConfig::read(&located, &store)?;
    let ops = Ops::new(Arc::clone(backend));
    let weights = AsrWeights::load(&ops, &located, &config, &store)?;
    let vocab = Arc::new(read_vocab(&located, &store)?);
    let filterbank = match read_filterbank(&store, config.mels, N_FFT / 2 + 1)? {
        Some(matrix) => Filterbank::Given(matrix),
        None => Filterbank::Computed {
            scale: MelScale::Slaney,
            slaney_norm: true,
            f_min: 0.0,
            f_max: 8_000.0,
        },
    };
    let mel = MelSpec {
        sample_rate: SAMPLE_RATE,
        n_fft: N_FFT,
        win_length: WIN,
        hop_length: HOP,
        n_mels: config.mels,
        window: Window::HannSymmetric,
        framing: Framing::Centred,
        preemphasis: Preemphasis::Signal(0.97),
        spectrum: Spectrum::Power,
        filterbank,
        log_scale: LogScale::Natural {
            guard: 2f32.powi(-24),
        },
    };
    let special = (0..config.vocab)
        .map(|id| {
            let id = u32::try_from(id).unwrap_or(u32::MAX);
            id == config.blank
                || matches!(vocab.piece(id), None | Some("<unk>" | "<pad>" | "<blank>"))
        })
        .collect();
    Ok(Some(NemotronAsr {
        model: Arc::new(Model {
            ops,
            config,
            weights,
        }),
        mel,
        vocab,
        special,
    }))
}

impl SpeechRecognizer for NemotronAsr {
    fn mel_spec(&self) -> &MelSpec {
        &self.mel
    }

    fn vocab(&self) -> &Arc<PieceVocab> {
        &self.vocab
    }

    fn frame_secs(&self) -> f64 {
        self.model.config.frame_secs()
    }

    fn languages(&self) -> Vec<String> {
        self.model.config.dictionary.keys().cloned().collect()
    }

    fn lookaheads(&self) -> &[usize] {
        &self.model.config.lookaheads
    }

    fn is_special(&self, id: u32) -> bool {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.special.get(i))
            .copied()
            .unwrap_or(true)
    }

    fn open_stream(
        &self,
        options: &StreamOptions,
    ) -> Result<Box<dyn RecognizerStream>, AudioError> {
        let m = &self.model;
        let encoder = EncoderState::new(&m.ops, &m.config, options.lookahead)?;
        let decoder = DecoderState::new(
            &Transducer::new(&m.ops, &m.weights, &m.config),
            &options.language,
        )?;
        Ok(Box::new(stream::AsrStream::new(
            Arc::clone(&self.model),
            encoder,
            decoder,
            options.lookahead,
        )))
    }
}

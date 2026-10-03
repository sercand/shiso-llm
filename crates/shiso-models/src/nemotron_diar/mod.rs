//! Nemotron speaker diarizer: stacked mel features into a bidirectional `RoPE`
//! transformer, a subpixel-upsampling head that scores eight speakers every
//! 10 ms, and an arrival-order speaker cache that carries speaker identity
//! from one chunk to the next.
//!
//! Every export this checkpoint ships in reads through one name table and one
//! config reader ([`config`]); the device path is the same for all of them.

pub mod cache;
pub mod config;
pub mod encoder;
mod names;
pub mod weights;

use crate::catalog::FamilyFacts;

/// Family facts for Nemotron Diarization.
pub const FACTS: FamilyFacts = FamilyFacts {
    name: "nemotron_diar",
};

use std::path::Path;
use std::sync::Arc;

use shiso_audio::{
    AudioError, DiarizationPass, Diarizer, Filterbank, Framing, LogScale, MelFrames, MelScale,
    MelSpec, Preemphasis, Spectrum, Window, transcript::SpeakerActivity,
};
use shiso_core::Backend;

use crate::audio_ops::Ops;
use crate::nemotron_asr::config::read_filterbank;
use cache::OfflinePass;
use config::{DiarConfig, OFFLINE};
use encoder::Encoder;
use weights::DiarWeights;

/// The frontend constants every export of this checkpoint states alike: the
/// streaming recognizer's, a 25 ms symmetric Hann window every 10 ms over a
/// 512-point FFT of 16 kHz audio, 0.97 pre-emphasis, a power spectrum and a
/// 2^-24 log guard.
const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 512;
const WIN: usize = 400;
const HOP: usize = 160;

/// What every pass of one loaded checkpoint shares.
#[derive(Debug)]
struct Model {
    ops: Ops,
    config: DiarConfig,
    weights: DiarWeights,
}

/// A loaded Nemotron diarizer.
#[derive(Debug)]
pub struct NemotronDiarizer {
    model: Arc<Model>,
    mel: MelSpec,
}

/// One clip's pass, holding the checkpoint it runs on.
struct Pass {
    model: Arc<Model>,
    pass: OfflinePass,
}

impl DiarizationPass for Pass {
    // `None`: the pass has more steps to run; it is not a refusal.
    fn step(&mut self) -> Result<Option<SpeakerActivity>, AudioError> {
        let m = &self.model;
        let encoder = Encoder::new(&m.ops, &m.config, &m.weights);
        let Some(probs) = self.pass.step(&encoder, &m.weights.silence)? else {
            return Ok(None);
        };
        SpeakerActivity::new(FRAME_SECS, m.config.speakers, probs).map(Some)
    }
}

/// Seconds per activity frame: one mel hop.
const FRAME_SECS: f64 = 0.01;

/// Opens the checkpoint at `path` on `backend`.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming what is broken in an export of this
/// family; a device error.
// `None` is not a refusal: the path is another family's, and the registry
// tries the next one.
pub fn open(
    path: &Path,
    backend: &Arc<dyn Backend>,
) -> Result<Option<NemotronDiarizer>, AudioError> {
    let Some(located) = config::locate(path)? else {
        return Ok(None);
    };
    located.check_offline(&OFFLINE)?;
    let store = located.store()?;
    let config = DiarConfig::read(&located, &store)?;
    let ops = Ops::new(Arc::clone(backend));
    let weights = DiarWeights::load(&ops, &located, &config, &store)?;
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
    Ok(Some(NemotronDiarizer {
        model: Arc::new(Model {
            ops,
            config,
            weights,
        }),
        mel,
    }))
}

impl Diarizer for NemotronDiarizer {
    fn mel_spec(&self) -> &MelSpec {
        &self.mel
    }

    fn speakers(&self) -> usize {
        self.model.config.speakers
    }

    fn frame_secs(&self) -> f64 {
        FRAME_SECS
    }

    fn begin(&self, mel: MelFrames) -> Result<Box<dyn DiarizationPass>, AudioError> {
        let m = &self.model;
        let encoder = Encoder::new(&m.ops, &m.config, &m.weights);
        Ok(Box::new(Pass {
            model: Arc::clone(&self.model),
            pass: OfflinePass::new(&encoder, mel)?,
        }))
    }
}

#[cfg(test)]
mod tests;

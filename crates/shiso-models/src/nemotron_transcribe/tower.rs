//! The encoder as the decoder's audio tower.

use std::path::Path;
use std::sync::Arc;

use shiso_audio::{AudioClip, AudioError, AudioTower, MelSpec};
use shiso_core::Backend;
use shiso_core::media::VisionGrid;
use shiso_vision::EncodedRows;

use super::config::PeeConfig;
use super::encoder::encode;
use super::frontend::{audio_tokens, mel_spec};
use super::weights::PeeWeights;
use crate::audio_ops::Ops;
use crate::audio_ops::store::TensorStore;

/// The longest recording the encoder takes, seconds.
const MAX_SECONDS: f64 = 3600.0;

/// The encoder export, loaded, answering the tower contract.
#[derive(Debug)]
pub struct NemotronTranscribeTower {
    config: PeeConfig,
    weights: PeeWeights,
    mel: MelSpec,
    bytes: u64,
}

impl NemotronTranscribeTower {
    /// Opens the encoder export at `path` on `backend`.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] for an export of another architecture or
    /// a missing or misshapen tensor; a device error.
    pub fn open(backend: &Arc<dyn Backend>, path: &Path) -> Result<Self, AudioError> {
        let store = TensorStore::open_gguf(path)?;
        let gguf = store.gguf().ok_or_else(|| AudioError::Checkpoint {
            what: format!("{} is not a GGUF export", path.display()),
        })?;
        let config = PeeConfig::from_gguf(gguf)?;
        let ops = Ops::new(Arc::clone(backend));
        let weights = PeeWeights::load(&ops, &store, &config)?;
        let mel = mel_spec(&config, &weights.filterbank);
        let bytes = std::fs::metadata(path).map_or(0, |m| m.len());
        Ok(Self {
            config,
            weights,
            mel,
            bytes,
        })
    }

    /// The export's numbers.
    #[must_use]
    pub const fn config(&self) -> &PeeConfig {
        &self.config
    }
}

impl AudioTower for NemotronTranscribeTower {
    fn mel_spec(&self) -> &MelSpec {
        &self.mel
    }

    fn tokens_for(&self, valid_mel: usize) -> Option<usize> {
        tokens_for(valid_mel)
    }

    fn max_seconds(&self) -> f64 {
        MAX_SECONDS
    }

    fn out_hidden(&self) -> usize {
        self.config.fusion_dim
    }

    fn device_bytes(&self) -> u64 {
        self.bytes
    }

    fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        clips: &[AudioClip],
    ) -> Result<EncodedRows, AudioError> {
        let ops = Ops::new(Arc::clone(backend));
        let width = self.config.fusion_dim;
        let mut grids = Vec::with_capacity(clips.len());
        let mut parts = Vec::with_capacity(clips.len());
        for clip in clips {
            let e = encode(&ops, &self.weights, &self.config, clip.mel())?;
            grids.push(VisionGrid::new(
                1,
                1,
                u32::try_from(e.frames).map_err(|_| AudioError::BadSpec {
                    reason: "too many audio rows".into(),
                })?,
            ));
            parts.push(e);
        }
        let total: usize = parts.iter().map(|e| e.frames).sum();
        let rows = ops.scratch(total * width)?;
        let mut at = 0u64;
        for e in &parts {
            let bytes = (e.frames * width * 4) as u64;
            backend.copy_d2d(rows.slice(at, bytes)?, e.rows.view())?;
            at += bytes;
        }
        Ok(EncodedRows { rows, grids })
    }
}

/// Encoder rows for `valid_mel` mel rows, or `None` for a clip too short to
/// make one.
fn tokens_for(valid_mel: usize) -> Option<usize> {
    (valid_mel > 0).then(|| audio_tokens(valid_mel))
}

/// A stand-in with the tower's limits and no weights, for the limit tests.
#[cfg(test)]
pub(crate) fn limits_for_tests() -> impl Fn(MelSpec) -> Limits {
    |mel| Limits { mel }
}

/// The tower's limits and nothing else.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct Limits {
    mel: MelSpec,
}

#[cfg(test)]
impl AudioTower for Limits {
    fn mel_spec(&self) -> &MelSpec {
        &self.mel
    }

    fn tokens_for(&self, valid_mel: usize) -> Option<usize> {
        tokens_for(valid_mel)
    }

    fn max_seconds(&self) -> f64 {
        MAX_SECONDS
    }

    fn out_hidden(&self) -> usize {
        2688
    }

    fn device_bytes(&self) -> u64 {
        0
    }

    fn encode(&self, _: &Arc<dyn Backend>, _: &[AudioClip]) -> Result<EncodedRows, AudioError> {
        Err(AudioError::BadSpec {
            reason: "a stand-in with no weights".into(),
        })
    }
}

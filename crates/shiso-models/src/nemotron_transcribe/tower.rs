//! The encoder as the decoder's audio tower.
//!
//! By default the tower holds no weights between requests: they are loaded
//! when a clip is encoded and dropped when it is done. The 1.7 GB they take
//! on the device is what lets the decoder's routed-expert pool be that much
//! larger on a machine whose working set the pool otherwise fills, and the
//! export is read back from the page cache or the disk in well under a
//! second. A server taking many short requests opens it resident instead.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use shiso_audio::{AudioClip, AudioError, AudioTower, MelSpec};
use shiso_core::Backend;
use shiso_core::media::VisionGrid;
use shiso_vision::EncodedRows;

use super::config::PeeConfig;
use super::encoder::encode;
use super::frontend::{audio_tokens, mel_spec};
use super::weights::PeeWeights;
use crate::audio_ops::Ops;
use crate::audio_ops::store::{TensorStore, dequant};

/// The longest recording the encoder takes, seconds.
const MAX_SECONDS: f64 = 3600.0;

/// The encoder export, loaded, answering the tower contract.
#[derive(Debug)]
pub struct NemotronTranscribeTower {
    path: PathBuf,
    config: PeeConfig,
    /// The weights, when the tower was opened resident; `None` between the
    /// calls of a tower that loads them for each encode.
    weights: Mutex<Option<PeeWeights>>,
    resident: bool,
    mel: MelSpec,
    bytes: u64,
}

impl NemotronTranscribeTower {
    /// Opens the encoder export at `path` on `backend`, its weights resident
    /// when `resident`, otherwise loaded for each encode and dropped after.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] for an export of another architecture or
    /// a missing or misshapen tensor; a device error.
    pub fn open(
        backend: &Arc<dyn Backend>,
        path: &Path,
        resident: bool,
    ) -> Result<Self, AudioError> {
        let store = TensorStore::open_gguf(path)?;
        let gguf = store.gguf().ok_or_else(|| AudioError::Checkpoint {
            what: format!("{} is not a GGUF export", path.display()),
        })?;
        let config = PeeConfig::from_gguf(gguf)?;
        let filterbank = dequant(&store.require("preprocessor.fb")?)?;
        let mel = mel_spec(&config, &filterbank);
        let weights = if resident {
            Some(PeeWeights::load(
                &Ops::new(Arc::clone(backend)),
                &store,
                &config,
            )?)
        } else {
            None
        };
        let bytes = std::fs::metadata(path).map_or(0, |m| m.len());
        Ok(Self {
            path: path.to_path_buf(),
            config,
            weights: Mutex::new(weights),
            resident,
            mel,
            bytes,
        })
    }

    /// Whether the weights are on the device now.
    #[must_use]
    pub fn is_resident(&self) -> bool {
        self.held().is_ok_and(|held| held.is_some())
    }

    /// The weights' slot; a poisoned lock is a refusal, not a panic.
    fn held(&self) -> Result<MutexGuard<'_, Option<PeeWeights>>, AudioError> {
        self.weights.lock().map_err(|_| AudioError::BadSpec {
            reason: "the speech encoder's weights were left mid-load by a failed request".into(),
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
        let mut held = self.held()?;
        if held.is_none() {
            let store = TensorStore::open_gguf(&self.path)?;
            *held = Some(PeeWeights::load(&ops, &store, &self.config)?);
        }
        let weights = held.as_ref().ok_or(AudioError::Empty)?;
        let width = self.config.fusion_dim;
        let mut grids = Vec::with_capacity(clips.len());
        let mut parts = Vec::with_capacity(clips.len());
        for clip in clips {
            let e = encode(&ops, weights, &self.config, clip.mel())?;
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
        if !self.resident {
            // The rows are copied out; the weights go before the decoder's
            // prompt pass needs the room.
            backend.sync()?;
            *held = None;
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

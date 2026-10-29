//! The encoder as the decoder's audio tower.
//!
//! By default the tower holds no weights between requests: they are loaded
//! when a clip is encoded and dropped when it is done. The 1.7 GB they take
//! on the device is what lets the decoder's routed-expert pool be that much
//! larger on a machine whose working set the pool otherwise fills, and the
//! export is read back from the page cache or the disk in well under a
//! second. A server taking many short requests opens it resident instead.
//!
//! The tower is also the decoder's word timer. The timestamp head reads the
//! speech branch's rows, which only exist while a clip is encoded, so a clip
//! whose words will be timed is registered first, and its encode keeps the
//! head's output and the speaker activity until the request is done with
//! them.

use std::hash::Hasher as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use shiso_audio::{
    AudioClip, AudioError, AudioTower, MelSpec, TimingKey, TimingTicket, WordTimer, WordTiming,
};
use shiso_core::Backend;
use shiso_core::media::VisionGrid;
use shiso_vision::EncodedRows;

use super::align::{Activity, AlignDesc, align, targets};
use super::config::PeeConfig;
use super::ctc::{ClassRows, CtcEvidence, evidence};
use super::encoder::{Encoded, encode};
use super::frontend::{audio_tokens, mel_spec};
use super::spm::CtcTokenizer;
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
    /// The timestamp head's tokenizer.
    tokenizer: CtcTokenizer,
    /// Clips registered for word timing, and what their encodes kept.
    timings: Arc<Mutex<Timings>>,
}

/// What an encode keeps for timing one clip's words.
#[derive(Debug)]
struct Kept {
    head: CtcEvidence,
    /// `[frames][speakers]`, 80 ms.
    activity: Vec<f32>,
    /// `[mel frames][speakers]`, 10 ms.
    activity10: Vec<f32>,
}

/// One registered clip.
#[derive(Debug)]
struct Timing {
    key: TimingKey,
    fingerprint: u64,
    kept: Option<Arc<Kept>>,
}

/// The registrations, oldest first.
#[derive(Debug, Default)]
struct Timings {
    next: u64,
    entries: Vec<Timing>,
}

/// A digest of what the tower is handed for `clip`, to recognise it at
/// encode.
fn fingerprint(clip: &AudioClip) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_usize(clip.tokens());
    h.write(bytemuck::cast_slice(clip.mel().as_slice()));
    h.finish()
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
        let spm = gguf
            .metadata()
            .get_str("ctc.tokenizer.spm_model")
            .ok_or_else(|| AudioError::Checkpoint {
                what: "the speech encoder export has no ctc.tokenizer.spm_model".into(),
            })?;
        let tokenizer = CtcTokenizer::from_base64(spm)?;
        if tokenizer.len() + 1 != config.ctc_vocab {
            return Err(AudioError::Checkpoint {
                what: format!(
                    "the timestamp tokenizer has {} pieces; its {}-class head needs one fewer \
                     than its classes",
                    tokenizer.len(),
                    config.ctc_vocab
                ),
            });
        }
        let bytes = std::fs::metadata(path).map_or(0, |m| m.len());
        Ok(Self {
            path: path.to_path_buf(),
            config,
            weights: Mutex::new(weights),
            resident,
            mel,
            bytes,
            tokenizer,
            timings: Arc::default(),
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

    /// The registrations; a poisoned lock is a refusal, not a panic.
    fn timings(&self) -> Result<MutexGuard<'_, Timings>, AudioError> {
        self.timings.lock().map_err(|_| AudioError::BadSpec {
            reason: "the word timings were left mid-update by a failed request".into(),
        })
    }

    /// Keeps what timing needs from `e`, the encode of `clip`, when a
    /// registration is waiting for it.
    fn keep_for_timing(
        &self,
        ops: &Ops,
        weights: &PeeWeights,
        clip: &AudioClip,
        e: &Encoded,
    ) -> Result<(), AudioError> {
        if !self.timings()?.entries.iter().any(|t| t.kept.is_none()) {
            return Ok(());
        }
        let print = fingerprint(clip);
        if !self
            .timings()?
            .entries
            .iter()
            .any(|t| t.kept.is_none() && t.fingerprint == print)
        {
            return Ok(());
        }
        let kept = Arc::new(Kept {
            head: evidence(ops, &weights.ctc, &self.config, &e.speech)?,
            activity: e.activity.clone(),
            activity10: e.activity10.clone(),
        });
        if let Some(t) = self
            .timings()?
            .entries
            .iter_mut()
            .find(|t| t.kept.is_none() && t.fingerprint == print)
        {
            t.kept = Some(kept);
        }
        Ok(())
    }
}

/// An event or style tag: the reference times it like a word, but the
/// transcript the caller reads has none.
fn is_tag(word: &str) -> bool {
    (word.starts_with("<ev:") || word.starts_with("<sty:")) && word.ends_with('>')
}

impl WordTimer for NemotronTranscribeTower {
    fn want(&self, clip: &AudioClip) -> TimingTicket {
        let fingerprint = fingerprint(clip);
        let mut timings = self.timings.lock().unwrap_or_else(|poisoned| {
            // The registrations are plain data; one left behind by a failed
            // request is still a valid list.
            poisoned.into_inner()
        });
        let key = TimingKey(timings.next);
        timings.next += 1;
        timings.entries.push(Timing {
            key,
            fingerprint,
            kept: None,
        });
        let registry = Arc::downgrade(&self.timings);
        TimingTicket::new(key, move |key| {
            if let Some(registry) = registry.upgrade() {
                let mut timings = registry
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                timings.entries.retain(|t| t.key != key);
            }
        })
    }

    fn words(
        &self,
        ticket: &TimingTicket,
        transcript: &str,
        seconds: f64,
    ) -> Result<WordTiming, AudioError> {
        let kept = self
            .timings()?
            .entries
            .iter()
            .find(|t| t.key == ticket.key())
            .and_then(|t| t.kept.clone());
        let Some(kept) = kept else {
            return Ok(WordTiming::Untimed {
                reason: "the clip was not encoded after it was registered".into(),
            });
        };
        let untimed = |e: super::align::AlignError| WordTiming::Untimed {
            reason: e.to_string(),
        };
        let t = match targets(&self.tokenizer, transcript, kept.head.frames()) {
            Ok(t) => t,
            Err(e) => return Ok(untimed(e)),
        };
        let store = TensorStore::open_gguf(&self.path)?;
        let classes = ClassRows::read(&store, &t.columns, self.config.ctc.d_model)?;
        let log_probs = kept.head.log_probs(&classes);
        let words = align(
            &AlignDesc {
                targets: &t,
                log_probs: &log_probs,
                frames: kept.head.frames(),
                activity: Some(Activity {
                    rows80: &kept.activity,
                    rows10: &kept.activity10,
                    speakers: self.config.speakers,
                }),
                duration: seconds,
            },
            self.config.ctc_upsample,
        );
        Ok(match words {
            Ok(words) => {
                WordTiming::Timed(words.into_iter().filter(|w| !is_tag(&w.text)).collect())
            }
            Err(e) => untimed(e),
        })
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

    fn word_timer(&self) -> Option<&dyn WordTimer> {
        Some(self)
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
            self.keep_for_timing(&ops, weights, clip, &e)?;
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

#[cfg(test)]
#[path = "tower_tests.rs"]
mod tests;

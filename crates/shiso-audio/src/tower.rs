//! The contract an audio tower meets for a language model that reads audio.
//!
//! A tower turns a clip into soft-token rows the language model splices at its
//! audio placeholder run, the same way a vision tower's rows land at an image's.
//! The token count is fixed before the tower runs - the prompt's placeholder
//! run is sized from it - so [`AudioClip`] carries both the mel rows and the
//! count, bound together by [`clip_for`], and a tower must return exactly that
//! many rows.

use std::fmt;
use std::sync::Arc;

use shiso_core::Backend;
use shiso_vision::EncodedRows;

use crate::error::AudioError;
use crate::mel::{MelFrames, MelFrontend, MelSpec};
use crate::pcm::Pcm;

/// One clip ready for a tower: its valid mel rows and its soft-token count.
#[derive(Clone, PartialEq)]
pub struct AudioClip {
    mel: MelFrames,
    tokens: usize,
}

impl AudioClip {
    /// Binds mel rows to the token count the tower will produce for them.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] for a clip of no tokens.
    pub fn new(mel: MelFrames, tokens: usize) -> Result<Self, AudioError> {
        if tokens == 0 {
            return Err(AudioError::BadSpec {
                reason: "an audio clip of no soft tokens".into(),
            });
        }
        Ok(Self { mel, tokens })
    }

    /// The mel rows, every one of them valid.
    #[must_use]
    pub fn mel(&self) -> &MelFrames {
        &self.mel
    }

    /// Soft tokens the clip occupies in the prompt.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

/// Summarised rather than derived: thirty seconds of mel rows is a few hundred
/// thousand floats, and a derived `Debug` puts all of them in any log line or
/// test failure that names a clip.
impl fmt::Debug for AudioClip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioClip")
            .field("mel_rows", &self.mel.valid())
            .field("n_mels", &self.mel.n_mels())
            .field("tokens", &self.tokens)
            .finish()
    }
}

/// A language model's audio tower.
pub trait AudioTower: Send + Sync + fmt::Debug {
    /// The frontend clips must be framed with.
    fn mel_spec(&self) -> &MelSpec;

    /// Soft tokens for `valid_mel` mel rows, or `None` past what the tower
    /// accepts.
    fn tokens_for(&self, valid_mel: usize) -> Option<usize>;

    /// The longest clip the tower accepts, in seconds.
    fn max_seconds(&self) -> f64;

    /// Width of each output row.
    fn out_hidden(&self) -> usize;

    /// Device memory the tower's weights occupy.
    fn device_bytes(&self) -> u64;

    /// Encodes `clips` into one block of rows, in order, `clip.tokens()` rows
    /// each.
    ///
    /// # Errors
    ///
    /// A device error, or an [`AudioError`] naming what the clip violates.
    fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        clips: &[AudioClip],
    ) -> Result<EncodedRows, AudioError>;
}

/// Frames `pcm` for `tower` and counts its tokens.
///
/// # Errors
///
/// [`AudioError::TooLong`] past the tower's limit, [`AudioError::Empty`] for a
/// clip too short to make one token, or [`AudioError::BadSpec`] for a
/// frontend the tower's spec does not build.
pub fn clip_for(tower: &dyn AudioTower, pcm: &Pcm) -> Result<AudioClip, AudioError> {
    let seconds = pcm.duration_secs();
    let max = tower.max_seconds();
    let too_long = || AudioError::TooLong {
        seconds,
        max: crate::num::u32_ceil(max),
    };
    if seconds > max {
        return Err(too_long());
    }
    let frontend = MelFrontend::new(tower.mel_spec().clone())?;
    let all = frontend.compute(pcm.samples());
    let valid = all.valid();
    if valid == 0 {
        return Err(AudioError::Empty);
    }
    let tokens = tower.tokens_for(valid).ok_or_else(too_long)?;
    if tokens == 0 {
        return Err(AudioError::Empty);
    }
    let n = all.n_mels();
    let mel = MelFrames::from_parts(all.as_slice()[..valid * n].to_vec(), n, valid, valid);
    AudioClip::new(mel, tokens)
}

#[cfg(test)]
#[path = "tower_tests.rs"]
mod tests;

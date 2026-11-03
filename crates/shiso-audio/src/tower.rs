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

    /// The tower's word timer, for a tower that can time a transcript's
    /// words against a clip it encoded. `None` by default.
    fn word_timer(&self) -> Option<&dyn WordTimer> {
        None
    }
}

/// Times a transcript's words against the clip it was written from.
///
/// The words are only known once the language model has written them, long
/// after the clip was encoded, so the tower has to keep what timing needs
/// from that encode. It keeps it only for clips registered with
/// [`WordTimer::want`] before they are encoded, and only while their ticket
/// lives.
pub trait WordTimer: Send + Sync + fmt::Debug {
    /// Registers `clip`: the tower's next encode of it keeps what timing its
    /// words needs, until the ticket is dropped. The clip must reach the
    /// tower's encode: rows reused from an earlier request's encode leave
    /// nothing to time against.
    fn want(&self, clip: &AudioClip) -> TimingTicket;

    /// The words of `transcript`, timed against the clip `ticket` registered,
    /// which is `seconds` long.
    ///
    /// # Errors
    ///
    /// A device or checkpoint error. A transcript the timer cannot align is
    /// not an error: it is [`WordTiming::Untimed`] with the reason.
    fn words(
        &self,
        ticket: &TimingTicket,
        transcript: &str,
        seconds: f64,
    ) -> Result<WordTiming, AudioError>;
}

/// What a [`WordTimer`] made of a transcript.
#[derive(Debug, Clone, PartialEq)]
pub enum WordTiming {
    /// Its words, in order of start.
    Timed(Vec<crate::transcript::Word>),
    /// No times: the transcript could not be aligned, or the clip was not
    /// encoded after it was registered.
    Untimed {
        /// Why, for the log.
        reason: String,
    },
}

/// Which registration a [`TimingTicket`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimingKey(pub u64);

/// A clip registered with a [`WordTimer`]; dropping it releases what the
/// tower keeps for the clip.
pub struct TimingTicket {
    key: TimingKey,
    release: Option<Box<dyn FnOnce(TimingKey) + Send + Sync>>,
}

impl TimingTicket {
    /// A ticket for `key` that calls `release` with it when dropped.
    pub fn new(key: TimingKey, release: impl FnOnce(TimingKey) + Send + Sync + 'static) -> Self {
        Self {
            key,
            release: Some(Box::new(release)),
        }
    }

    /// The registration.
    #[must_use]
    pub const fn key(&self) -> TimingKey {
        self.key
    }
}

impl Drop for TimingTicket {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release(self.key);
        }
    }
}

impl fmt::Debug for TimingTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TimingTicket").field(&self.key).finish()
    }
}

/// Frames `pcm` for `tower` and counts its tokens.
///
/// # Errors
///
/// [`AudioError::TooLong`] past the tower's limit, [`AudioError::Empty`] for no
/// samples or a clip too short to make one token, or [`AudioError::BadSpec`] for a
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
    // A frontend that pads before framing makes a row of padding out of no
    // samples at all; that row is not audio.
    if pcm.samples().is_empty() {
        return Err(AudioError::Empty);
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

//! The contract a speaker diarizer meets and the server drives.
//!
//! One call answers one whole clip. A diarizer that runs in chunks keeps its
//! chunking inside: the speaker indices it reports are only comparable across
//! chunks when one pass assigns them, and the server needs activity over the
//! whole clip before it can say which speaker a word belongs to.

use std::fmt;

use crate::error::AudioError;
use crate::mel::{MelFrames, MelSpec};
use crate::transcript::SpeakerActivity;

/// A loaded speaker diarizer.
///
/// It holds the device it was loaded on, as a recognizer does: the weights
/// live there, so a pass elsewhere is not something to ask for.
pub trait Diarizer: Send + Sync + fmt::Debug {
    /// The frontend a clip must be framed with.
    fn mel_spec(&self) -> &MelSpec;

    /// Speakers the model tells apart; the width of every activity row.
    fn speakers(&self) -> usize;

    /// Seconds per activity frame.
    fn frame_secs(&self) -> f64;

    /// Speaker activity over every valid mel row of one whole clip.
    ///
    /// Speakers are indexed in the order the model first hears them, and the
    /// result has one frame per mel row.
    ///
    /// # Errors
    ///
    /// [`AudioError::Empty`] for a clip too short to make one frame,
    /// [`AudioError::BadSpec`] for frames of the wrong width, or a device
    /// error.
    fn diarize(&self, mel: &MelFrames) -> Result<SpeakerActivity, AudioError> {
        let mut pass = self.begin(mel.clone())?;
        loop {
            if let Some(activity) = pass.step()? {
                return Ok(activity);
            }
        }
    }

    /// Starts a pass over one whole clip that the caller advances a step at a
    /// time, so a thread serving several jobs can interleave a long clip's
    /// diarization with the others' work.
    ///
    /// # Errors
    ///
    /// As [`Diarizer::diarize`], for what can be refused before the first
    /// step.
    fn begin(&self, mel: MelFrames) -> Result<Box<dyn DiarizationPass>, AudioError>;
}

/// One clip's diarization in progress.
pub trait DiarizationPass: Send {
    /// Runs one step: `None` while there is more, the whole clip's activity
    /// after the last.
    ///
    /// # Errors
    ///
    /// A device error. The pass is not to be stepped again after one.
    fn step(&mut self) -> Result<Option<SpeakerActivity>, AudioError>;
}

#[cfg(test)]
#[path = "diarizer_tests.rs"]
mod tests;

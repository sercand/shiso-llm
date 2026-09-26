//! Mono PCM, the currency between the decoder and every frontend.

use crate::error::AudioError;
use crate::num::f64_of;

/// The rate every speech model here consumes.
pub const MODEL_RATE: u32 = 16_000;

/// One channel of `f32` samples in `[-1, 1]` at a known rate.
///
/// Mono by construction: every decoder downmixes before building one, so no
/// frontend has to ask how many channels it was handed.
#[derive(Debug, Clone, PartialEq)]
pub struct Pcm {
    samples: Vec<f32>,
    rate: u32,
}

impl Pcm {
    /// Binds samples to their rate.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadRate`] for a zero rate.
    pub fn new(samples: Vec<f32>, rate: u32) -> Result<Self, AudioError> {
        if rate == 0 {
            return Err(AudioError::BadRate { rate });
        }
        Ok(Self { samples, rate })
    }

    /// The samples.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Samples per second.
    #[must_use]
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Length in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        f64_of(self.samples.len()) / f64::from(self.rate)
    }

    /// The samples, by value.
    #[must_use]
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }
}

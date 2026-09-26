//! The real-input FFT of one frame, sized once per frontend.
//!
//! Planned once by the real-input FFT dependency, which picks the fastest
//! decomposition it has for the size and the machine and exploits the real
//! input, so a transform is about half a complex one. Accumulation is in `f64`
//! so the frontend's own error stays well below what the `f32` reference
//! extractors carry.

use std::fmt;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::error::AudioError;

/// A transform of one fixed power-of-two size.
pub(crate) struct Fft {
    n: usize,
    plan: Arc<dyn RealToComplex<f64>>,
}

impl fmt::Debug for Fft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fft")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

/// The buffers one transform works in, reused frame after frame.
#[derive(Debug, Clone)]
pub(crate) struct FftWork {
    /// The frame; the transform overwrites it.
    pub(crate) frame: Vec<f64>,
    /// Bins `0..=n/2`.
    pub(crate) bins: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl Fft {
    /// Plans a transform of `n` points.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] unless `n` is a power of two of at least 2.
    pub(crate) fn new(n: usize) -> Result<Self, AudioError> {
        if n < 2 || !n.is_power_of_two() {
            return Err(AudioError::BadSpec {
                reason: format!("n_fft {n} is not a power of two >= 2"),
            });
        }
        let plan = RealFftPlanner::<f64>::new().plan_fft_forward(n);
        Ok(Self { n, plan })
    }

    /// Points per transform.
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    /// Buffers sized for this transform.
    pub(crate) fn work(&self) -> FftWork {
        FftWork {
            frame: self.plan.make_input_vec(),
            bins: self.plan.make_output_vec(),
            scratch: self.plan.make_scratch_vec(),
        }
    }

    /// The DFT of `work.frame` into `work.bins` (bins `0..=n/2`). The frame is
    /// consumed.
    pub(crate) fn spectrum(&self, work: &mut FftWork) {
        self.plan
            .process_with_scratch(&mut work.frame, &mut work.bins, &mut work.scratch)
            .expect("the buffers come from this plan, so their lengths are its own");
    }
}

#[cfg(test)]
#[path = "fft_tests.rs"]
mod tests;

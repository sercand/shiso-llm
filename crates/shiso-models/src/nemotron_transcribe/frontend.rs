//! The mel frontend, the whole-recording normalization and the window plan.
//!
//! The frames are the reference's own: pre-emphasis 0.97 over the signal
//! rounded up to 4000 samples, `n_fft / 2` zeros on the left, a periodic Hann
//! window at the start of each 512-sample frame, the power spectrum through
//! the export's own filterbank and `ln(x + 2^-24)`; `1 + L / hop` frames.
//! The speech branch reads them normalized per mel bin over the whole
//! recording; the speaker branch reads them raw.
//!
//! A recording longer than one window is encoded in 40-second windows with
//! 4 seconds of context on each side, each on its own, and only each window's
//! core rows are kept.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "row counts are far inside f64's mantissa, and the normalized value is narrowed to the f32 the encoder reads"
)]

use shiso_audio::{
    Filterbank, Framing, LogScale, MelFrames, MelSpec, Preemphasis, Spectrum, Window,
};

use super::config::PeeConfig;

/// Samples the signal is rounded up to before it is framed.
const PAD_TO: usize = 4000;

/// The guard added before the log.
const LOG_GUARD: f32 = 1.0 / 16_777_216.0;

/// Added to a bin's standard deviation before it divides.
const STD_EPS: f64 = 1e-5;

/// The frontend `cfg` and the export's `filterbank` describe.
#[must_use]
pub fn mel_spec(cfg: &PeeConfig, filterbank: &[f32]) -> MelSpec {
    MelSpec {
        sample_rate: cfg.sample_rate,
        n_fft: cfg.n_fft,
        win_length: cfg.win_length,
        hop_length: cfg.hop_length,
        n_mels: cfg.mels,
        window: Window::HannPeriodic,
        framing: Framing::FftLeftPadded {
            pad_to_multiple: PAD_TO,
        },
        preemphasis: Preemphasis::Signal(0.97),
        spectrum: Spectrum::Power,
        filterbank: Filterbank::Given(filterbank.to_vec()),
        log_scale: LogScale::Natural { guard: LOG_GUARD },
    }
}

/// Encoder frames `mel_frames` mel rows stack to.
#[must_use]
pub const fn audio_tokens(mel_frames: usize) -> usize {
    mel_frames.div_ceil(8)
}

/// `mel`'s valid rows, each bin shifted by its mean and divided by its
/// unbiased standard deviation (plus 1e-5) over every valid row.
#[must_use]
pub fn normalize(mel: &MelFrames) -> Vec<f32> {
    let bins = mel.n_mels();
    let rows = mel.valid();
    let data = &mel.as_slice()[..rows * bins];
    let mut mean = vec![0.0f64; bins];
    for row in data.chunks_exact(bins) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += f64::from(*v);
        }
    }
    for m in &mut mean {
        *m /= rows.max(1) as f64;
    }
    let mut var = vec![0.0f64; bins];
    for row in data.chunks_exact(bins) {
        for ((s, m), v) in var.iter_mut().zip(&mean).zip(row) {
            let d = f64::from(*v) - m;
            *s += d * d;
        }
    }
    let inv: Vec<f64> = var
        .iter()
        .map(|s| 1.0 / ((s / (rows.saturating_sub(1).max(1)) as f64).sqrt() + STD_EPS))
        .collect();
    data.chunks_exact(bins)
        .flat_map(|row| {
            row.iter()
                .zip(&mean)
                .zip(&inv)
                .map(|((v, m), i)| ((f64::from(*v) - m) * i) as f32)
        })
        .collect()
}

/// One window of the encoder's plan, in mel frames unless named otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkSpan {
    /// First frame of the core.
    pub start: usize,
    /// One past the core's last frame.
    pub end: usize,
    /// First frame the window encodes.
    pub enc_start: usize,
    /// One past the last frame it encodes.
    pub enc_end: usize,
    /// Encoder rows of left context dropped from its output.
    pub left_drop: usize,
    /// Encoder rows it keeps.
    pub core: usize,
}

impl ChunkSpan {
    /// Mel frames the window encodes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.enc_end - self.enc_start
    }

    /// Whether it encodes nothing.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.enc_end == self.enc_start
    }

    /// Mel frames of right context past the core.
    #[must_use]
    pub const fn right(&self) -> usize {
        self.enc_end - self.end
    }
}

/// The windows `mel_frames` mel rows are encoded in: one when they fit,
/// otherwise cores of `window_core` encoder rows with `window_context` rows of
/// context on each side.
#[must_use]
pub fn chunk_plan(mel_frames: usize, cfg: &PeeConfig) -> Vec<ChunkSpan> {
    let core = cfg.window_core * cfg.stack;
    let context = cfg.window_context * cfg.stack;
    if mel_frames <= core {
        return vec![ChunkSpan {
            start: 0,
            end: mel_frames,
            enc_start: 0,
            enc_end: mel_frames,
            left_drop: 0,
            core: audio_tokens(mel_frames),
        }];
    }
    (0..mel_frames.div_ceil(core))
        .map(|ci| {
            let start = ci * core;
            let end = (start + core).min(mel_frames);
            let enc_start = start.saturating_sub(context);
            let enc_end = (end + context).min(mel_frames);
            ChunkSpan {
                start,
                end,
                enc_start,
                enc_end,
                left_drop: (start - enc_start) / cfg.stack,
                core: audio_tokens(end) - audio_tokens(start),
            }
        })
        .collect()
}

/// Rows `first .. first + len` of `rows` (`[_][bins]`) stacked `stack` at a
/// time frame-major, the last group zero-padded: `[ceil(len/stack)][stack·bins]`.
#[must_use]
pub fn stack_rows(rows: &[f32], bins: usize, first: usize, len: usize, stack: usize) -> Vec<f32> {
    let mut out = rows[first * bins..(first + len) * bins].to_vec();
    out.resize(len.div_ceil(stack) * stack * bins, 0.0);
    out
}

#[cfg(test)]
#[path = "frontend_tests.rs"]
mod tests;

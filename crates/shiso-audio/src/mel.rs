//! Log-mel frontends, every constant explicit.
//!
//! Three frontends are in use, and they share almost nothing but the idea:
//!
//! | | centred | reflect | left-padded |
//! |---|---|---|---|
//! | framing | `n_fft/2` zeros both sides, `n_fft`-sample frames | `n_fft/2` mirrored both sides, `n_fft`-sample frames | `win/2` zeros on the left, `win`-sample frames |
//! | window | symmetric Hann, centred in `n_fft` | periodic Hann, centred in `n_fft` | periodic Hann at the frame start |
//! | preemphasis | over the whole signal | none | none |
//! | spectrum | power | power | magnitude |
//! | filterbank | Slaney scale, area-normalised | Slaney scale, area-normalised | HTK scale, unnormalised |
//! | log scale | natural, per frame | [`LogScale::Log10MaxFloor`], over the whole clip | natural, per frame |
//!
//! So nothing here has a default. A family states each field from its own
//! checkpoint, and a mismatch is a wrong transcript, not an error. That is why
//! the reference fixtures pin each style separately.

use crate::error::AudioError;
use crate::fft::{Fft, FftWork};
use crate::num::{f32_of, f64_of, i64_of, usize_of};

/// The analysis window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// `0.5 − 0.5·cos(2πn/(N−1))`.
    HannSymmetric,
    /// `0.5 − 0.5·cos(2πn/N)`.
    HannPeriodic,
}

/// How the signal is cut into frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// `n_fft/2` zeros on each side; frame `t` spans `n_fft` samples from
    /// `t·hop`; the window sits centred in the frame. `1 + L/hop` frames, of
    /// which `L/hop` are valid.
    Centred,
    /// `n_fft/2` samples on each side, mirrored by reflection (numpy's
    /// `mode="reflect"`: the edge sample is never repeated) instead of
    /// zeroed; otherwise identical to [`Self::Centred`], including its frame
    /// counts. Forbids pre-emphasis, which no frontend here combines with it.
    CentredReflect,
    /// The signal is rounded up to `pad_to_multiple` samples with zeros, then
    /// `win/2` zeros go on the left only. Frame `t` spans `win` samples from
    /// `t·hop`, zero-filled to `n_fft`. It is valid while its last sample lies
    /// inside the real signal.
    LeftPadded {
        /// Round the signal length up to a multiple of this (1 for none).
        pad_to_multiple: usize,
    },
}

/// Pre-emphasis applied before framing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Preemphasis {
    /// None.
    None,
    /// `y[0] = x[0]`, `y[i] = x[i] − c·x[i−1]`, over the real samples only.
    Signal(f32),
}

/// What each FFT bin contributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spectrum {
    /// `|X|²`.
    Power,
    /// `|X|`.
    Magnitude,
}

/// The mel frequency scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    /// Linear below 1 kHz, logarithmic above.
    Slaney,
    /// `2595·log10(1 + f/700)`.
    Htk,
}

/// Where the filterbank comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum Filterbank {
    /// Triangles computed from a scale and range.
    Computed {
        /// The scale the triangle edges are spaced on.
        scale: MelScale,
        /// Divide each triangle by its width in Hz (unit area).
        slaney_norm: bool,
        /// Lowest edge, Hz.
        f_min: f32,
        /// Highest edge, Hz.
        f_max: f32,
    },
    /// A matrix the checkpoint ships, row-major `[n_mels][n_fft/2 + 1]`.
    Given(Vec<f32>),
}

/// How a mel energy becomes the log value the encoder reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogScale {
    /// `ln(x + guard)`, computed per frame as each row is filled.
    Natural {
        /// Added before the log so a silent bin does not diverge.
        guard: f32,
    },
    /// `log10(max(x, 1e-10))`, then `max(v, chunk_max - 8.0)`, then
    /// `(v + 4.0) / 4.0`, where `chunk_max` is the maximum over every valid
    /// row of the whole computed [`MelFrames`]. The floor and rescale need
    /// every frame's value first, so this is applied once in
    /// [`MelFrontend::compute`] after all frames exist, not per frame; the
    /// incremental `MelStream` frontend has no "after all frames exist" and
    /// refuses this variant with [`AudioError::BadSpec`].
    Log10MaxFloor,
    /// `log10(max(x, 1e-10))` per frame: the first stage of
    /// [`LogScale::Log10MaxFloor`] alone. A recognizer that transcribes a clip
    /// in chunks and floors each chunk on its own maximum computes the whole
    /// clip this way once, then finishes each chunk's rows with
    /// [`log10_max_floor`]: the floor depends on which rows share it, so it
    /// cannot be undone once applied over the whole clip.
    Log10,
}

/// Every constant of one log-mel frontend.
#[derive(Debug, Clone, PartialEq)]
pub struct MelSpec {
    /// Input rate the spec is written for.
    pub sample_rate: u32,
    /// FFT size, a power of two.
    pub n_fft: usize,
    /// Window length, at most `n_fft`.
    pub win_length: usize,
    /// Samples between frames.
    pub hop_length: usize,
    /// Mel bins per frame.
    pub n_mels: usize,
    /// Analysis window.
    pub window: Window,
    /// Framing rule.
    pub framing: Framing,
    /// Pre-emphasis.
    pub preemphasis: Preemphasis,
    /// Power or magnitude.
    pub spectrum: Spectrum,
    /// Filterbank source.
    pub filterbank: Filterbank,
    /// How a mel energy becomes a log value.
    pub log_scale: LogScale,
}

/// Mel frames, row-major `[total][n_mels]`; rows at or past `valid` are zero.
#[derive(Debug, Clone, PartialEq)]
pub struct MelFrames {
    data: Vec<f32>,
    n_mels: usize,
    total: usize,
    valid: usize,
}

impl MelFrames {
    /// Rows the caller already holds, every one valid - a reference
    /// extractor's output, say.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] when `data` is not whole rows of `n_mels`.
    pub fn from_rows(data: Vec<f32>, n_mels: usize) -> Result<Self, AudioError> {
        if n_mels == 0 || !data.len().is_multiple_of(n_mels) {
            return Err(bad(format!(
                "{} values are not whole rows of {n_mels}",
                data.len()
            )));
        }
        let rows = data.len() / n_mels;
        Ok(Self::from_parts(data, n_mels, rows, rows))
    }

    pub(crate) fn from_parts(data: Vec<f32>, n_mels: usize, total: usize, valid: usize) -> Self {
        debug_assert_eq!(data.len(), n_mels * total);
        Self {
            data,
            n_mels,
            total,
            valid,
        }
    }

    /// Bins per row.
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Rows, including the zeroed tail.
    #[must_use]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Rows that describe real audio.
    #[must_use]
    pub fn valid(&self) -> usize {
        self.valid
    }

    /// All rows.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Row `t`.
    #[must_use]
    pub fn row(&self, t: usize) -> &[f32] {
        &self.data[t * self.n_mels..(t + 1) * self.n_mels]
    }
}

/// A frontend built from one [`MelSpec`].
#[derive(Debug)]
pub struct MelFrontend {
    spec: MelSpec,
    window: Vec<f64>,
    bank: Vec<f32>,
    /// Each filter's nonzero bins, `[first, last)`: a triangle covers a few
    /// bins of the 257, so the sum over the rest adds exact zeros and is
    /// skipped without changing a bit of the result.
    support: Vec<(usize, usize)>,
    fft: Fft,
}

/// One frame's working buffers, reused frame after frame.
#[derive(Debug, Clone)]
pub(crate) struct FrameWork {
    fft: FftWork,
    power: Vec<f64>,
}

/// Frames per thread below which splitting the work costs more than it saves.
const FRAMES_PER_THREAD: usize = 512;

fn hz_to_mel(f: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + f / 700.0).log10(),
        MelScale::Slaney => {
            if f < 1000.0 {
                3.0 * f / 200.0
            } else {
                15.0 + (f / 1000.0).ln() / (6.4f64.ln() / 27.0)
            }
        }
    }
}

fn mel_to_hz(m: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10f64.powf(m / 2595.0) - 1.0),
        MelScale::Slaney => {
            if m < 15.0 {
                200.0 * m / 3.0
            } else {
                1000.0 * ((6.4f64.ln() / 27.0) * (m - 15.0)).exp()
            }
        }
    }
}

fn bad(reason: impl Into<String>) -> AudioError {
    AudioError::BadSpec {
        reason: reason.into(),
    }
}

/// Mirrors a virtual sample position `s` (any integer, inside `0..n` or not)
/// into `0..n`, matching numpy's `mode="reflect"`: the edge sample at `0` or
/// `n - 1` is never repeated, so a period is `2*(n - 1)`. `n` must be at
/// least 2.
fn reflect_index(s: i64, n: i64) -> i64 {
    let period = 2 * (n - 1);
    let m = s.rem_euclid(period);
    if m < n { m } else { period - m }
}

/// Fills the `left` reflect-padding samples on each side of `out`, which
/// already holds `samples` verbatim at `out[left..left + samples.len()]`.
fn reflect_pad(out: &mut [f64], samples: &[f32], left: usize) {
    let n = samples.len();
    if n == 0 {
        return;
    }
    let start = left + n;
    let right = out.len().saturating_sub(start);
    if n == 1 {
        // A single sample reflects onto itself from every offset.
        let value = f64::from(samples[0]);
        out[..left].fill(value);
        out[start..start + right].fill(value);
        return;
    }
    let nn = i64_of(n);
    for (i, slot) in out[..left].iter_mut().enumerate() {
        let s = i64_of(i) - i64_of(left);
        *slot = f64::from(samples[usize_of(reflect_index(s, nn))]);
    }
    for (j, slot) in out[start..start + right].iter_mut().enumerate() {
        let s = nn + i64_of(j);
        *slot = f64::from(samples[usize_of(reflect_index(s, nn))]);
    }
}

/// The floor-and-rescale [`LogScale::Log10MaxFloor`] finishes with, applied
/// once over the whole computed frames: `v = max(stage-one value,
/// chunk_max - 8)`, then `(v + 4) / 4`. `data` holds a stage-one
/// `log10(max(x, 1e-10))` value in every row up to `valid`, written by
/// [`MelFrontend::frame_row_with`]; rows at or past `valid` are the caller's
/// zeroed tail and are left alone.
fn apply_max_floor(data: &mut [f32], mels: usize, valid: usize) {
    let head = &mut data[..valid * mels];
    let Some(chunk_max) = head.iter().copied().reduce(f32::max) else {
        return;
    };
    let floor = f64::from(chunk_max) - 8.0;
    for slot in head.iter_mut() {
        let v = f64::from(*slot).max(floor);
        *slot = f32_of((v + 4.0) / 4.0);
    }
}

/// Frames computed with [`LogScale::Log10`], finished as
/// [`LogScale::Log10MaxFloor`] would finish exactly these frames: the floor at
/// their own maximum less 8, then `(v + 4) / 4`. Computing a clip with
/// [`LogScale::Log10`] and finishing it here gives the same values as
/// computing it with [`LogScale::Log10MaxFloor`].
#[must_use]
pub fn log10_max_floor(mel: &MelFrames) -> MelFrames {
    let mut data = mel.as_slice().to_vec();
    apply_max_floor(&mut data, mel.n_mels(), mel.valid());
    MelFrames::from_parts(data, mel.n_mels(), mel.total(), mel.valid())
}

/// The analysis window, placed in an `n_fft`-long vector as the framing asks.
fn window(spec: &MelSpec) -> Vec<f64> {
    let len = spec.win_length;
    let offset = match spec.framing {
        Framing::Centred | Framing::CentredReflect => (spec.n_fft - len) / 2,
        Framing::LeftPadded { .. } => 0,
    };
    let denom = match spec.window {
        Window::HannSymmetric => f64_of(len.saturating_sub(1).max(1)),
        Window::HannPeriodic => f64_of(len),
    };
    let mut window = vec![0.0; spec.n_fft];
    for (i, slot) in window[offset..offset + len].iter_mut().enumerate() {
        *slot = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * f64_of(i) / denom).cos();
    }
    window
}

/// The filterbank a spec asks for, row-major `[n_mels][bins]`.
fn filterbank(spec: &MelSpec, bins: usize) -> Result<Vec<f32>, AudioError> {
    let (scale, slaney_norm, f_min, f_max) = match &spec.filterbank {
        Filterbank::Given(matrix) => {
            if matrix.len() != spec.n_mels * bins {
                return Err(bad(format!(
                    "a given filterbank holds {} values, not {}",
                    matrix.len(),
                    spec.n_mels * bins
                )));
            }
            return Ok(matrix.clone());
        }
        Filterbank::Computed {
            scale,
            slaney_norm,
            f_min,
            f_max,
        } => (*scale, *slaney_norm, f64::from(*f_min), f64::from(*f_max)),
    };
    let nyquist = f64::from(spec.sample_rate) / 2.0;
    if !(f_min >= 0.0 && f_min < f_max && f_max <= nyquist) {
        return Err(bad(format!(
            "mel range {f_min}..{f_max} Hz does not fit 0..{nyquist}"
        )));
    }
    let (mel_lo, mel_hi) = (hz_to_mel(f_min, scale), hz_to_mel(f_max, scale));
    let edges: Vec<f64> = (0..spec.n_mels + 2)
        .map(|i| {
            mel_to_hz(
                mel_lo + (mel_hi - mel_lo) * f64_of(i) / f64_of(spec.n_mels + 1),
                scale,
            )
        })
        .collect();
    let mut bank = vec![0.0f32; spec.n_mels * bins];
    for (row, tri) in bank.chunks_exact_mut(bins).zip(edges.windows(3)) {
        let (lower, centre, upper) = (tri[0], tri[1], tri[2]);
        let norm = if slaney_norm {
            2.0 / (upper - lower)
        } else {
            1.0
        };
        for (bin, slot) in row.iter_mut().enumerate() {
            let freq = f64_of(bin) * f64::from(spec.sample_rate) / f64_of(spec.n_fft);
            let weight = ((freq - lower) / (centre - lower))
                .min((upper - freq) / (upper - centre))
                .max(0.0);
            *slot = f32_of(weight * norm);
        }
    }
    Ok(bank)
}

impl MelFrontend {
    /// Builds a frontend, refusing a spec whose fields contradict each other.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] naming the failed constraint.
    pub fn new(spec: MelSpec) -> Result<Self, AudioError> {
        let fft = Fft::new(spec.n_fft)?;
        let bins = spec.n_fft / 2 + 1;
        if spec.win_length == 0 || spec.win_length > spec.n_fft {
            return Err(bad(format!(
                "win_length {} must be in 1..={}",
                spec.win_length, spec.n_fft
            )));
        }
        if spec.hop_length == 0 || spec.n_mels == 0 {
            return Err(bad("hop_length and n_mels must be positive"));
        }
        if let LogScale::Natural { guard } = spec.log_scale
            && (guard.is_nan() || guard <= 0.0)
        {
            return Err(bad("a natural log_scale guard must be positive"));
        }
        if let Framing::LeftPadded { pad_to_multiple } = spec.framing {
            if pad_to_multiple == 0 {
                return Err(bad("pad_to_multiple must be at least 1"));
            }
            if spec.preemphasis != Preemphasis::None {
                return Err(bad(
                    "left-padded framing with pre-emphasis is not a frontend any model here uses",
                ));
            }
        }
        if spec.framing == Framing::CentredReflect && spec.preemphasis != Preemphasis::None {
            return Err(bad(
                "reflect framing with pre-emphasis is not a frontend any model here uses",
            ));
        }
        let window = window(&spec);
        let bank = filterbank(&spec, bins)?;
        let support = bank
            .chunks_exact(bins)
            .map(|row| {
                let first = row.iter().position(|w| *w != 0.0).unwrap_or(0);
                let last = row.iter().rposition(|w| *w != 0.0).map_or(first, |l| l + 1);
                (first, last)
            })
            .collect();
        Ok(Self {
            spec,
            window,
            bank,
            support,
            fft,
        })
    }

    /// The spec this frontend was built from.
    #[must_use]
    pub fn spec(&self) -> &MelSpec {
        &self.spec
    }

    #[cfg(test)]
    pub(crate) fn bank_row(&self, m: usize) -> &[f32] {
        let bins = self.spec.n_fft / 2 + 1;
        &self.bank[m * bins..(m + 1) * bins]
    }

    /// Valid frames for a signal of `samples` samples.
    #[must_use]
    pub fn valid_frames(&self, samples: usize) -> usize {
        let hop = self.spec.hop_length;
        match self.spec.framing {
            Framing::Centred | Framing::CentredReflect => samples / hop,
            Framing::LeftPadded { .. } => {
                // t·hop + win < win/2 + L  <=>  t·hop < L − (win − win/2)
                let reach = self.spec.win_length - self.spec.win_length / 2;
                if samples <= reach {
                    0
                } else {
                    (samples - reach - 1) / hop + 1
                }
                .min(self.total_frames(samples))
            }
        }
    }

    fn total_frames(&self, samples: usize) -> usize {
        let hop = self.spec.hop_length;
        match self.spec.framing {
            Framing::Centred | Framing::CentredReflect => 1 + samples / hop,
            Framing::LeftPadded { pad_to_multiple } => {
                let padded =
                    samples.div_ceil(pad_to_multiple) * pad_to_multiple + self.spec.win_length / 2;
                let span = self.spec.win_length + 1;
                if padded < span {
                    0
                } else {
                    (padded - span) / hop + 1
                }
            }
        }
    }

    /// The pre-emphasised, padded signal the frames are cut from.
    pub(crate) fn padded(&self, samples: &[f32]) -> Vec<f64> {
        let (left, len) = match self.spec.framing {
            Framing::Centred | Framing::CentredReflect => {
                (self.spec.n_fft / 2, samples.len() + self.spec.n_fft)
            }
            Framing::LeftPadded { pad_to_multiple } => {
                let left = self.spec.win_length / 2;
                (
                    left,
                    left + samples.len().div_ceil(pad_to_multiple) * pad_to_multiple,
                )
            }
        };
        let mut out = vec![0.0; len.max(left)];
        let mut prev = 0.0f64;
        for (i, &x) in samples.iter().enumerate() {
            let x = f64::from(x);
            out[left + i] = match self.spec.preemphasis {
                Preemphasis::None => x,
                Preemphasis::Signal(c) => {
                    if i == 0 {
                        x
                    } else {
                        x - f64::from(c) * prev
                    }
                }
            };
            prev = x;
        }
        // Reflect framing forbids pre-emphasis (checked in `new`), so `out`
        // still holds the real samples verbatim; mirror them into the
        // padding on each side instead of leaving it zero.
        if self.spec.framing == Framing::CentredReflect {
            reflect_pad(&mut out, samples, left);
        }
        out
    }

    /// Buffers for [`Self::frame_row_with`].
    pub(crate) fn work(&self) -> FrameWork {
        FrameWork {
            fft: self.fft.work(),
            power: vec![0.0; self.fft.len() / 2 + 1],
        }
    }

    /// Writes the log-mel row of the frame starting at `start` in `padded`.
    pub(crate) fn frame_row(&self, padded: &[f64], start: usize, out: &mut [f32]) {
        let mut work = self.work();
        self.frame_row_with(&mut work, padded, start, out);
    }

    /// [`Self::frame_row`] in buffers the caller keeps between frames.
    pub(crate) fn frame_row_with(
        &self,
        work: &mut FrameWork,
        padded: &[f64],
        start: usize,
        out: &mut [f32],
    ) {
        let n = self.fft.len();
        let span = match self.spec.framing {
            Framing::Centred | Framing::CentredReflect => n,
            Framing::LeftPadded { .. } => self.spec.win_length,
        };
        work.fft.frame.fill(0.0);
        for (i, (slot, w)) in work
            .fft
            .frame
            .iter_mut()
            .zip(&self.window)
            .take(span)
            .enumerate()
        {
            *slot = padded.get(start + i).copied().unwrap_or(0.0) * w;
        }
        self.fft.spectrum(&mut work.fft);
        for (p, c) in work.power.iter_mut().zip(&work.fft.bins) {
            let e = c.re * c.re + c.im * c.im;
            *p = match self.spec.spectrum {
                Spectrum::Power => e,
                Spectrum::Magnitude => e.sqrt(),
            };
        }
        let bins = n / 2 + 1;
        for (m, slot) in out.iter_mut().enumerate() {
            let (first, last) = self.support[m];
            let row = &self.bank[m * bins + first..m * bins + last];
            let e: f64 = row
                .iter()
                .zip(&work.power[first..last])
                .map(|(w, s)| f64::from(*w) * s)
                .sum();
            *slot = match self.spec.log_scale {
                LogScale::Natural { guard } => f32_of((e + f64::from(guard)).ln()),
                // Stage one only; the floor and rescale need every frame's
                // value first, so `MelFrontend::compute` finishes the job.
                LogScale::Log10MaxFloor | LogScale::Log10 => f32_of(e.max(1e-10).log10()),
            };
        }
    }

    /// Log-mel frames of a whole signal.
    #[must_use]
    pub fn compute(&self, samples: &[f32]) -> MelFrames {
        let total = self.total_frames(samples.len());
        let valid = self.valid_frames(samples.len());
        let padded = self.padded(samples);
        let mels = self.spec.n_mels;
        let mut data = vec![0.0f32; total * mels];
        // Frames are independent, so a long clip is cut into runs framed on
        // as many threads as the machine offers; each thread keeps its own
        // buffers and writes its own rows.
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(valid.div_ceil(FRAMES_PER_THREAD))
            .max(1);
        let per = valid.div_ceil(threads).max(1);
        let hop = self.spec.hop_length;
        std::thread::scope(|scope| {
            for (i, rows) in data[..valid * mels].chunks_mut(per * mels).enumerate() {
                let padded = &padded;
                scope.spawn(move || {
                    let mut work = self.work();
                    for (j, out) in rows.chunks_exact_mut(mels).enumerate() {
                        self.frame_row_with(&mut work, padded, (i * per + j) * hop, out);
                    }
                });
            }
        });
        if matches!(self.spec.log_scale, LogScale::Log10MaxFloor) {
            apply_max_floor(&mut data, mels, valid);
        }
        MelFrames::from_parts(data, mels, total, valid)
    }
}

#[cfg(test)]
#[path = "mel_tests.rs"]
pub(crate) mod tests;

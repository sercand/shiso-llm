//! The speech encoder's numbers, read from its GGUF export's metadata.
//!
//! The export holds three transformers of one block - the speech branch, the
//! speaker branch and the timestamp head - which differ only in width, depth,
//! how much of each head turns and whether queries and keys are normed, so
//! one [`BlockDims`] describes each.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "window and hop lengths are a few hundred samples, positive, and rounded on purpose"
)]

use shiso_audio::AudioError;
use shiso_modelfile::GgufFile;

/// The architecture name the encoder export declares.
pub const GGUF_ARCH: &str = "pee";

/// One rotary transformer's geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDims {
    /// Residual width.
    pub d_model: usize,
    /// Attention heads.
    pub heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Layers.
    pub layers: usize,
    /// Feed-forward inner width.
    pub ff: usize,
    /// Leading dimensions of each head that turn.
    pub rot_dims: usize,
    /// Whether queries and keys are layer-normed per head before turning.
    pub qk_norm: bool,
}

/// Every number the encoder runs by.
#[derive(Debug, Clone, PartialEq)]
pub struct PeeConfig {
    /// The speech branch.
    pub asr: BlockDims,
    /// The speaker branch.
    pub diar: BlockDims,
    /// The timestamp head's transformer.
    pub ctc: BlockDims,
    /// Mel bins a frame holds.
    pub mels: usize,
    /// Mel frames stacked into one encoder frame.
    pub stack: usize,
    /// Speakers the speaker branch scores.
    pub speakers: usize,
    /// The speaker head's hidden width.
    pub head_hidden: usize,
    /// The decoder width the fused rows are projected to.
    pub fusion_dim: usize,
    /// Activity above which a speaker counts as speaking.
    pub threshold: f32,
    /// The timestamp head's vocabulary, its blank last.
    pub ctc_vocab: usize,
    /// Timestamp frames per encoder frame.
    pub ctc_upsample: usize,
    /// What the timestamp transformer's branch is scaled by before its
    /// residual add.
    pub ctc_residual: f32,
    /// Encoder frames one window emits.
    pub window_core: usize,
    /// Encoder frames of context on each side of a window.
    pub window_context: usize,
    /// Input rate.
    pub sample_rate: u32,
    /// FFT size.
    pub n_fft: usize,
    /// Analysis window length, samples.
    pub win_length: usize,
    /// Samples between frames.
    pub hop_length: usize,
}

fn missing(key: &str) -> AudioError {
    AudioError::Checkpoint {
        what: format!("the speech encoder export has no {key}"),
    }
}

impl PeeConfig {
    /// Reads the numbers out of `file`'s metadata.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming another architecture, an absent key,
    /// or a geometry this build does not run (a head that does not divide its
    /// width, a rotation wider than a head, a stack other than 8, or a zero
    /// count where the plan, the frontend or the aligner divides by it).
    pub fn from_gguf(file: &GgufFile) -> Result<Self, AudioError> {
        let meta = file.metadata();
        if meta.architecture() != Some(GGUF_ARCH) {
            return Err(AudioError::Checkpoint {
                what: format!(
                    "the speech encoder export declares {:?}, not {GGUF_ARCH:?}",
                    meta.architecture().unwrap_or_default()
                ),
            });
        }
        let u = |key: &str| -> Result<usize, AudioError> {
            meta.get_u32(key)
                .map(|v| v as usize)
                .ok_or_else(|| missing(key))
        };
        let f = |key: &str| -> Result<f32, AudioError> {
            meta.get_f32(key).ok_or_else(|| missing(key))
        };
        // A count the window plan, the frontend or the aligner divides by or
        // steps over: zero is refused here rather than found as a division
        // by zero (or an empty window) halfway through a request.
        let n = |key: &str| -> Result<usize, AudioError> {
            match u(key)? {
                0 => Err(AudioError::Checkpoint {
                    what: format!("the speech encoder export sets {key} to 0"),
                }),
                v => Ok(v),
            }
        };
        let b = |key: &str| -> Result<bool, AudioError> {
            meta.get_bool(key).ok_or_else(|| missing(key))
        };
        let block = |p: &str| -> Result<BlockDims, AudioError> {
            let d_model = u(&format!("{p}.d_model"))?;
            let heads = u(&format!("{p}.n_heads"))?;
            let rot_dims = u(&format!("{p}.rotary_dims"))?;
            if heads == 0 || !d_model.is_multiple_of(heads) {
                return Err(AudioError::Checkpoint {
                    what: format!("{p}: {heads} heads do not divide a width of {d_model}"),
                });
            }
            let head_dim = d_model / heads;
            if rot_dims == 0 || rot_dims > head_dim || !rot_dims.is_multiple_of(2) {
                return Err(AudioError::Checkpoint {
                    what: format!("{p}: {rot_dims} rotary dims in a {head_dim}-wide head"),
                });
            }
            Ok(BlockDims {
                d_model,
                heads,
                head_dim,
                layers: u(&format!("{p}.n_layers"))?,
                ff: u(&format!("{p}.ff_hidden"))?,
                rot_dims,
                qk_norm: b(&format!("{p}.qk_norm"))?,
            })
        };
        let stack = u("pee.asr.subsampling_factor")?;
        if stack != 8 || u("pee.diar.subsampling_factor")? != stack {
            return Err(AudioError::Checkpoint {
                what: format!("a stack of {stack} mel frames; this build stacks 8"),
            });
        }
        let sample_rate = n("pee.preprocessor.sample_rate")?;
        let seconds = |key: &str| -> Result<usize, AudioError> {
            Ok((f(key)? * sample_rate as f32).round() as usize)
        };
        let win_length = seconds("pee.preprocessor.window_size")?;
        let hop_length = seconds("pee.preprocessor.window_stride")?;
        if win_length == 0 || hop_length == 0 {
            return Err(AudioError::Checkpoint {
                what: format!(
                    "the speech encoder export's analysis window ({win_length}) or hop \
                     ({hop_length}) rounds to 0 samples"
                ),
            });
        }
        Ok(Self {
            asr: block("pee.asr")?,
            diar: block("pee.diar")?,
            ctc: block("ctc")?,
            mels: n("pee.preprocessor.features")?,
            stack,
            speakers: n("pee.diar.num_speakers")?,
            head_hidden: u("pee.diar.tf_d_model")?,
            fusion_dim: u("pee.fusion.proj_dim")?,
            threshold: f("pee.fusion.speaker_activity_threshold")?,
            ctc_vocab: n("ctc.vocab_size")?,
            ctc_upsample: n("ctc.upsampling_factor")?,
            ctc_residual: f("ctc.residual_scale")?,
            window_core: n("pee.online.inference_length")?,
            window_context: n("pee.online.chunk_left_context")?,
            sample_rate: u32::try_from(sample_rate).map_err(|_| missing("a sample rate"))?,
            n_fft: n("pee.preprocessor.n_fft")?,
            win_length,
            hop_length,
        })
    }

    /// Refuses a decoder whose residual width is not what the fused rows are
    /// projected to.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming both widths.
    pub fn check_decoder(&self, hidden: usize) -> Result<(), AudioError> {
        if self.fusion_dim == hidden {
            return Ok(());
        }
        Err(AudioError::Checkpoint {
            what: format!(
                "the speech encoder projects to {} but the decoder is {hidden} wide",
                self.fusion_dim
            ),
        })
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
pub(crate) mod tests;

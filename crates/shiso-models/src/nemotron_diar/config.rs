//! Which export this is, and the numbers it runs with.
//!
//! The four exports agree on the architecture and disagree on where they
//! write it, so the depths and head counts are read from each export's own
//! keys, and the widths from the tensors every export shares. The offline
//! chunking constants are the family's: two exports state them (the reference
//! framework's config and the MLX config) and those are checked against the
//! family's here; the two GGUF exports carry only the training-time streaming
//! values, which the reference says are not for inference, and those are
//! never read.

use std::path::{Path, PathBuf};

use serde_json::Value;
use shiso_audio::AudioError;
use shiso_modelfile::{GgufFile, MetaValue};

use super::names::{Names, Stem};
use crate::audio_ops::store::{Stored, TensorStore};

/// Which export a checkpoint path is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavour {
    /// The reference framework's own safetensors, in f32.
    Hf,
    /// The MLX export, in bf16.
    Mlx,
    /// The audio-framework GGUF, bf16 or `q8_0`, which embeds its source
    /// config.
    GgufAudioFramework,
    /// The GGUF whose architecture key is `sortformer`.
    GgufSortformer,
}

/// A checkpoint path recognised as this family.
#[derive(Debug)]
pub struct Located {
    /// The export.
    pub flavour: Flavour,
    /// The file holding the tensors.
    pub weights: PathBuf,
    /// The export's config document: its own `config.json`, or the one an
    /// audio-framework GGUF embeds. `None` for the `sortformer` GGUF, whose
    /// facts are metadata keys.
    config: Option<Value>,
}

/// The numbers one checkpoint runs with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiarConfig {
    /// Encoder width.
    pub d_model: usize,
    /// Transformer layers.
    pub layers: usize,
    /// Attention heads.
    pub heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Feed-forward width.
    pub ff: usize,
    /// Mel bins.
    pub mels: usize,
    /// Mel frames stacked into one encoder frame; also the head's upsampling.
    pub stack: usize,
    /// Speakers told apart.
    pub speakers: usize,
    /// Width of the head.
    pub head_hidden: usize,
}

/// The reference's offline mode, in encoder frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfflineParams {
    /// Frames emitted per step.
    pub chunk: usize,
    /// Frames of future context each step sees and does not emit.
    pub lookahead: usize,
    /// Frames the FIFO holds between steps.
    pub fifo: usize,
    /// Frames moved from the FIFO to the speaker cache at once.
    pub update_period: usize,
    /// Frames the speaker cache holds.
    pub cache: usize,
    /// Silence frames kept per speaker in the cache.
    pub silence_per_speaker: usize,
    /// The probability floor inside the frame score.
    pub score_threshold: f32,
    /// Score added to frames that just left the FIFO.
    pub latest_boost: f32,
    /// Share of a speaker's budget boosted twice.
    pub strong_rate: f32,
    /// Share of a speaker's budget boosted once.
    pub weak_rate: f32,
    /// Share of a speaker's budget below which overlapping frames are kept.
    pub min_pos_rate: f32,
}

/// The family's offline constants, as the reference runs them.
pub const OFFLINE: OfflineParams = OfflineParams {
    chunk: 340,
    lookahead: 40,
    fifo: 40,
    update_period: 300,
    cache: 264,
    silence_per_speaker: 1,
    score_threshold: 0.25,
    latest_boost: 0.05,
    strong_rate: 0.75,
    weak_rate: 1.5,
    min_pos_rate: 0.5,
};

fn checkpoint(what: impl Into<String>) -> AudioError {
    AudioError::Checkpoint { what: what.into() }
}

fn read_json(path: &Path) -> Result<Value, AudioError> {
    let text = std::fs::read_to_string(path).map_err(|source| AudioError::Io {
        what: "reading a checkpoint config",
        source,
    })?;
    serde_json::from_str(&text).map_err(|e| checkpoint(format!("{}: {e}", path.display())))
}

/// The document an audio-framework GGUF embeds under `name`.
fn embedded_json(file: &GgufFile, name: &str) -> Result<Value, AudioError> {
    let meta = file.metadata();
    let names = meta
        .get_array("audiocpp.embedded_files.names")
        .ok_or_else(|| checkpoint("the GGUF lists no embedded files"))?
        .strings()
        .map_err(|e| checkpoint(format!("the embedded file names: {e}")))?;
    let index = names
        .iter()
        .position(|n| &**n == name)
        .ok_or_else(|| checkpoint(format!("the GGUF embeds no {name}")))?;
    let offsets = meta
        .get_array("audiocpp.embedded_files.offsets")
        .ok_or_else(|| checkpoint("the GGUF has no embedded file offsets"))?;
    let at = |i: usize| -> Result<usize, AudioError> {
        let value = offsets
            .get(i)
            .map_err(|e| checkpoint(format!("the embedded file offsets: {e}")))?;
        meta_u64(&value)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| checkpoint("an embedded file offset is not a count"))
    };
    let (start, end) = (at(index)?, at(index + 1)?);
    let data = meta
        .get_array("audiocpp.embedded_files.data")
        .ok_or_else(|| checkpoint("the GGUF has no embedded file data"))?;
    let bytes: Vec<u8> = data
        .iter()
        .skip(start)
        .take(end.saturating_sub(start))
        .map(|v| match v {
            Ok(MetaValue::U8(b)) => Ok(b),
            Ok(MetaValue::I8(b)) => Ok(b.cast_unsigned()),
            _ => Err(checkpoint("the embedded file data is not bytes")),
        })
        .collect::<Result<_, _>>()?;
    serde_json::from_slice(&bytes).map_err(|e| checkpoint(format!("the embedded {name}: {e}")))
}

fn meta_u64(value: &MetaValue) -> Option<u64> {
    match value {
        MetaValue::U8(v) => Some(u64::from(*v)),
        MetaValue::U16(v) => Some(u64::from(*v)),
        MetaValue::U32(v) => Some(u64::from(*v)),
        MetaValue::U64(v) => Some(*v),
        MetaValue::I32(v) => u64::try_from(*v).ok(),
        MetaValue::I64(v) => u64::try_from(*v).ok(),
        _ => None,
    }
}

/// Recognises `path` (a `.gguf`, or an export directory) as this family.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming what is broken in an export of this family.
// `None` is not a refusal: the path is some other family's, and the registry
// tries the next one.
pub fn locate(path: &Path) -> Result<Option<Located>, AudioError> {
    if path.is_file() {
        let Ok(file) = GgufFile::open(path) else {
            return Ok(None);
        };
        let meta = file.metadata();
        let (flavour, config) = match meta.architecture() {
            Some("sortformer") => (Flavour::GgufSortformer, None),
            Some("audiocpp")
                if meta.get_str("audiocpp.model_spec.family") == Some("nemotron_3_diar") =>
            {
                (
                    Flavour::GgufAudioFramework,
                    Some(embedded_json(&file, "config.json")?),
                )
            }
            _ => return Ok(None),
        };
        return Ok(Some(Located {
            flavour,
            weights: path.to_path_buf(),
            config,
        }));
    }
    let config_path = path.join("config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let config = read_json(&config_path)?;
    let flavour = match config.get("model_type").and_then(Value::as_str) {
        Some("nemotron3_diarization") => Flavour::Hf,
        Some("nemotron_diarization") => Flavour::Mlx,
        _ => return Ok(None),
    };
    Ok(Some(Located {
        flavour,
        weights: path.join("model.safetensors"),
        config: Some(config),
    }))
}

impl Located {
    /// Opens the tensor file.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] when it cannot be read.
    pub fn store(&self) -> Result<TensorStore, AudioError> {
        match self.flavour {
            Flavour::Hf | Flavour::Mlx => TensorStore::open_safetensors(&self.weights),
            Flavour::GgufAudioFramework | Flavour::GgufSortformer => {
                TensorStore::open_gguf(&self.weights)
            }
        }
    }

    /// The name table for this export.
    #[must_use]
    pub(crate) fn names(&self) -> Names {
        Names::new(self.flavour)
    }

    fn json_usize(&self, pointer: &str) -> Result<usize, AudioError> {
        self.config
            .as_ref()
            .and_then(|c| c.pointer(pointer))
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| {
                checkpoint(format!(
                    "{:?} config has no count at {pointer}",
                    self.flavour
                ))
            })
    }

    /// Refuses an export whose stated offline constants are not the family's.
    ///
    /// An export that states none is not refused: the constants are the
    /// reference's offline mode, not a property any export can change.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming the key that disagrees.
    pub fn check_offline(&self, params: &OfflineParams) -> Result<(), AudioError> {
        let keys: &[(&str, usize)] = match self.flavour {
            Flavour::Hf => &[
                ("/chunk_length", params.chunk),
                ("/chunk_right_context", params.lookahead),
                ("/fifo_length", params.fifo),
                ("/speaker_cache_update_period", params.update_period),
            ],
            Flavour::Mlx => &[
                ("/modules_config/chunk_len", params.chunk),
                ("/modules_config/chunk_right_context", params.lookahead),
                ("/modules_config/fifo_len", params.fifo),
                (
                    "/modules_config/spkcache_update_period",
                    params.update_period,
                ),
                ("/modules_config/spkcache_len", params.cache),
            ],
            Flavour::GgufAudioFramework | Flavour::GgufSortformer => &[],
        };
        for (pointer, want) in keys {
            let Some(stated) = self.config.as_ref().and_then(|c| c.pointer(pointer)) else {
                continue;
            };
            if stated.as_u64() != u64::try_from(*want).ok() {
                return Err(checkpoint(format!(
                    "{:?} config states {pointer} = {stated}, but the offline mode runs {want}",
                    self.flavour
                )));
            }
        }
        Ok(())
    }
}

fn gguf_usize(store: &TensorStore, key: &str) -> Result<usize, AudioError> {
    store
        .gguf()
        .and_then(|g| g.metadata().get_u32(key))
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| checkpoint(format!("the GGUF has no {key}")))
}

/// Length of axis `axis` of a stored tensor, in the reference's order.
fn axis(t: &Stored<'_>, axis: usize) -> Result<usize, AudioError> {
    t.shape
        .get(axis)
        .and_then(|d| usize::try_from(*d).ok())
        .ok_or_else(|| checkpoint(format!("{} has no axis {axis}", t.name)))
}

impl DiarConfig {
    /// Reads the numbers from the export's keys and tensors.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming the key or tensor that is missing or
    /// inconsistent.
    pub fn read(located: &Located, store: &TensorStore) -> Result<Self, AudioError> {
        let (layers, heads, speakers) = match located.flavour {
            Flavour::Hf => (
                located.json_usize("/audio_config/num_hidden_layers")?,
                located.json_usize("/audio_config/num_attention_heads")?,
                located.json_usize("/head_config/num_speakers")?,
            ),
            Flavour::Mlx => (
                located.json_usize("/encoder_config/n_layers")?,
                located.json_usize("/encoder_config/n_heads")?,
                located.json_usize("/num_speakers")?,
            ),
            Flavour::GgufAudioFramework => (
                located.json_usize("/encoder/num_hidden_layers")?,
                located.json_usize("/encoder/num_attention_heads")?,
                located.json_usize("/num_speakers")?,
            ),
            Flavour::GgufSortformer => (
                gguf_usize(store, "sortformer.encoder.n_layers")?,
                gguf_usize(store, "sortformer.encoder.n_heads")?,
                gguf_usize(store, "sortformer.num_speakers")?,
            ),
        };
        let names = located.names();
        // The stated depth must be the file's: the weights require every
        // stated layer, and a layer past it would otherwise go unread and
        // run a shallower encoder than the one exported.
        if layers == 0 || store.get(&names.weight(Stem::Norm1(layers))).is_some() {
            return Err(checkpoint(format!(
                "{:?} config states {layers} layers, but the file holds a layer {layers}",
                located.flavour
            )));
        }
        let d_model = store
            .require(&names.weight(Stem::InNorm))?
            .elements()
            .try_into()
            .map_err(|_| checkpoint("the input norm is too wide"))?;
        let fc1 = store.require(&names.weight(Stem::Fc1(0)))?;
        let ff = axis(&fc1, 0)?;
        let proj = store.require(&names.weight(Stem::Proj))?;
        let stacked = axis(&proj, 1)?;
        let enc_proj = store.require(&names.weight(Stem::EncProj))?;
        let head_hidden = axis(&enc_proj, 0)?;
        // Every export stacks eight frames; the head upsamples by the same
        // factor, which its output width checks below.
        let stack = 8;
        let upsample = store.require(&names.weight(Stem::Upsample))?;
        let up_out = axis(&upsample, 0)?;
        if heads == 0
            || d_model % heads != 0
            || stacked % stack != 0
            || up_out != stack * head_hidden
        {
            return Err(checkpoint(format!(
                "{:?}: {heads} heads over {d_model}, a {stacked}-wide stack and a {up_out}-wide \
                 upsample do not make this architecture",
                located.flavour
            )));
        }
        Ok(Self {
            d_model,
            layers,
            heads,
            head_dim: d_model / heads,
            ff,
            mels: stacked / stack,
            stack,
            speakers,
            head_hidden,
        })
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
pub(super) mod tests;

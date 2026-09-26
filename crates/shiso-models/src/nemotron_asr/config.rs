//! What checkpoint this is, and the numbers it runs with.
//!
//! The four exports of this checkpoint disagree on where they write each fact.
//! So the architecture's dimensions are read from the tensors themselves,
//! which every export agrees on by construction; only the facts a shape cannot
//! carry - the attention window, the lookaheads, the symbol cap, the blank id,
//! the language dictionary, the mel count - come from each export's own keys.
//! Two exports omit one such fact each; for those the value the other exports
//! state is used, and says so where it is written.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use shiso_audio::{AudioError, Language, PieceVocab};
use shiso_modelfile::{GgufFile, MetaValue};

use super::names::{Names, Stem};
use crate::audio_ops::store::{Stored, StoredType, TensorStore, dequant};

/// Which export a checkpoint path is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavour {
    /// The reference framework's own safetensors, in f32.
    Hf,
    /// The MLX export: 8-bit affine linears beside bf16 everything else.
    Mlx,
    /// The GGUF whose architecture key is `asr`.
    GgufAsr,
    /// The GGUF whose architecture key is `parakeet`.
    GgufParakeet,
}

/// A checkpoint path recognised as this family.
#[derive(Debug)]
pub struct Located {
    /// The export.
    pub flavour: Flavour,
    /// The file holding the tensors.
    pub weights: PathBuf,
    /// The directory the export lives in.
    pub dir: PathBuf,
    config: Option<Value>,
    processor: Option<Value>,
}

/// The numbers one checkpoint runs with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrConfig {
    /// Encoder width.
    pub d_model: usize,
    /// Conformer blocks.
    pub layers: usize,
    /// Attention heads.
    pub heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Feed-forward width.
    pub ff: usize,
    /// Depthwise conv width in the conv module.
    pub conv_kernel: usize,
    /// Subsampling channels.
    pub sub_channels: usize,
    /// Mel bins.
    pub mels: usize,
    /// Output classes, blank included.
    pub vocab: usize,
    /// The blank id.
    pub blank: u32,
    /// Prediction network width.
    pub pred_hidden: usize,
    /// Prediction network LSTM layers.
    pub pred_layers: usize,
    /// Joint network width.
    pub joint: usize,
    /// Language prompts.
    pub prompts: usize,
    /// Prompt MLP width.
    pub prompt_hidden: usize,
    /// Past encoder frames a query may attend to.
    pub att_left: usize,
    /// Lookaheads the encoder was trained for, ascending.
    pub lookaheads: Vec<usize>,
    /// Symbols the decoder may emit on one frame before it must advance.
    pub max_symbols: usize,
    /// The prompt id meaning "detect the language".
    pub auto_prompt: usize,
    /// Locale tag to prompt id.
    pub dictionary: BTreeMap<String, usize>,
}

/// Lookaheads of this checkpoint, for the one export (`asr`) that does not
/// list them; the other three agree on exactly these.
const FAMILY_LOOKAHEADS: [usize; 4] = [0, 3, 6, 13];
/// Symbols per frame, for the one export (`parakeet`) that does not state it;
/// the other three agree on exactly this.
const FAMILY_MAX_SYMBOLS: usize = 10;

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

/// Recognises `path` (a `.gguf`, or an export directory) as this family.
///
/// `None` is not a refusal: it means the path is some other family's, and the
/// registry tries the next one. A path that is this family's but broken is an
/// error naming the defect.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming what is broken in an export of this family.
pub fn locate(path: &Path) -> Result<Option<Located>, AudioError> {
    if path.is_file() {
        let Ok(file) = GgufFile::open(path) else {
            return Ok(None);
        };
        let meta = file.metadata();
        let flavour = match (
            meta.architecture(),
            meta.get_str("asr.head_type"),
            meta.get_str("stt.parakeet.head_kind"),
        ) {
            (Some("asr"), Some("rnnt"), _) => Flavour::GgufAsr,
            (Some("parakeet"), _, Some("rnnt")) => Flavour::GgufParakeet,
            _ => return Ok(None),
        };
        let dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        return Ok(Some(Located {
            flavour,
            weights: path.to_path_buf(),
            dir,
            config: None,
            processor: None,
        }));
    }
    let config_path = path.join("config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let config = read_json(&config_path)?;
    let flavour = match config.get("model_type").and_then(Value::as_str) {
        Some("nemotron3_5_asr") => Flavour::Hf,
        Some("nemotron_asr") if config.pointer("/prompt/num_prompts").is_some() => Flavour::Mlx,
        _ => return Ok(None),
    };
    let processor = match flavour {
        Flavour::Hf => Some(read_json(&path.join("processor_config.json"))?),
        _ => None,
    };
    Ok(Some(Located {
        flavour,
        weights: path.join("model.safetensors"),
        dir: path.to_path_buf(),
        config: Some(config),
        processor,
    }))
}

impl Located {
    /// The same paths read as another export, for a test that needs a name
    /// table that does not match the file.
    #[cfg(test)]
    pub(crate) fn with_flavour(&self, flavour: Flavour) -> Self {
        Self {
            flavour,
            weights: self.weights.clone(),
            dir: self.dir.clone(),
            config: self.config.clone(),
            processor: self.processor.clone(),
        }
    }

    /// Elements per affine group of an export that quantises its linears that
    /// way, as its config states; `None` for an export that does not.
    #[must_use]
    pub fn affine_group(&self) -> Option<usize> {
        if self.flavour != Flavour::Mlx {
            return None;
        }
        self.config
            .as_ref()
            .and_then(|c| c.pointer("/quantization/group_size"))
            .and_then(Value::as_u64)
            .and_then(|g| usize::try_from(g).ok())
    }

    /// Opens the tensor file.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] when it cannot be read.
    pub fn store(&self) -> Result<TensorStore, AudioError> {
        if self.flavour == Flavour::Mlx && self.affine_group().is_none() {
            return Err(checkpoint(
                "the MLX export's config states no quantization group_size",
            ));
        }
        match self.flavour {
            Flavour::Hf | Flavour::Mlx => TensorStore::open_safetensors(&self.weights),
            Flavour::GgufAsr | Flavour::GgufParakeet => TensorStore::open_gguf(&self.weights),
        }
    }

    fn json(&self, pointer: &str) -> Result<&Value, AudioError> {
        let (doc, which) = match (pointer.strip_prefix("/processor"), self.flavour) {
            (Some(_), _) => (self.processor.as_ref(), "processor_config.json"),
            (None, _) => (self.config.as_ref(), "config.json"),
        };
        let inner = pointer.strip_prefix("/processor").unwrap_or(pointer);
        doc.and_then(|d| d.pointer(inner))
            .ok_or_else(|| checkpoint(format!("{which} has no {inner}")))
    }

    fn json_usize(&self, pointer: &str) -> Result<usize, AudioError> {
        self.json(pointer)?
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| checkpoint(format!("{pointer} is not a count")))
    }
}

fn meta_u64(value: &MetaValue) -> Option<u64> {
    match value {
        MetaValue::U8(v) => Some(u64::from(*v)),
        MetaValue::U16(v) => Some(u64::from(*v)),
        MetaValue::U32(v) => Some(u64::from(*v)),
        MetaValue::U64(v) => Some(*v),
        MetaValue::I32(v) => u64::try_from(*v).ok(),
        _ => None,
    }
}

fn gguf_usize(store: &TensorStore, key: &str) -> Result<usize, AudioError> {
    store
        .gguf()
        .and_then(|g| g.metadata().get_u32(key))
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| checkpoint(format!("the GGUF has no {key}")))
}

fn gguf_usizes(store: &TensorStore, key: &str) -> Result<Vec<usize>, AudioError> {
    let array = store
        .gguf()
        .and_then(|g| g.metadata().get_array(key))
        .ok_or_else(|| checkpoint(format!("the GGUF has no {key}")))?;
    array
        .iter()
        .map(|v| {
            v.ok()
                .as_ref()
                .and_then(meta_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| checkpoint(format!("{key} holds a non-count")))
        })
        .collect()
}

fn gguf_strings(store: &TensorStore, key: &str) -> Result<Vec<String>, AudioError> {
    store
        .gguf()
        .and_then(|g| g.metadata().get_array(key))
        .ok_or_else(|| checkpoint(format!("the GGUF has no {key}")))?
        .strings()
        .map(|v| v.into_iter().map(String::from).collect())
        .map_err(|e| checkpoint(format!("{key}: {e}")))
}

/// Rows of a stored matrix: its first axis.
fn rows(t: &Stored<'_>) -> usize {
    t.shape
        .first()
        .and_then(|d| usize::try_from(*d).ok())
        .unwrap_or(0)
}

/// Columns of a stored linear, reading packed words through their scales.
fn cols(store: &TensorStore, t: &Stored<'_>, group: usize) -> Result<usize, AudioError> {
    if t.ty == StoredType::U32 {
        let stem = t.name.strip_suffix(".weight").unwrap_or(t.name);
        let scales = store.require(&format!("{stem}.scales"))?;
        let groups = scales.shape.get(1).copied().unwrap_or(0);
        return usize::try_from(groups * u64::try_from(group).unwrap_or(0))
            .map_err(|_| checkpoint(format!("{} is too wide", t.name)));
    }
    t.shape
        .get(1)
        .and_then(|d| usize::try_from(*d).ok())
        .ok_or_else(|| checkpoint(format!("{} is not a matrix", t.name)))
}

fn dictionary_from_json(value: &Value) -> Result<BTreeMap<String, usize>, AudioError> {
    value
        .as_object()
        .ok_or_else(|| checkpoint("the prompt dictionary is not an object"))?
        .iter()
        .map(|(k, v)| {
            v.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .map(|v| (k.clone(), v))
                .ok_or_else(|| checkpoint(format!("prompt {k} is not an id")))
        })
        .collect()
}

/// The conformer and LSTM depths the export states. The tensors are then
/// required to exist for every one of them, so an export missing a block's
/// tensor is refused naming it rather than read as a shallower model.
fn stated_depths(located: &Located, store: &TensorStore) -> Result<(usize, usize), AudioError> {
    Ok(match located.flavour {
        Flavour::Hf => (
            located.json_usize("/encoder_config/num_hidden_layers")?,
            located.json_usize("/num_decoder_layers")?,
        ),
        Flavour::Mlx => (
            located.json_usize("/encoder/n_layers")?,
            located.json_usize("/decoder/pred_rnn_layers")?,
        ),
        Flavour::GgufAsr => (
            gguf_usize(store, "asr.encoder.n_layers")?,
            gguf_usize(store, "asr.rnnt.pred_num_layers")?,
        ),
        Flavour::GgufParakeet => (
            gguf_usize(store, "stt.parakeet.encoder.n_layers")?,
            gguf_usize(store, "stt.parakeet.predictor.n_layers")?,
        ),
    })
}

impl AsrConfig {
    /// Reads the numbers of the checkpoint at `located`.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming the missing key or tensor, or a
    /// dimension the kernels are not written for.
    #[expect(
        clippy::too_many_lines,
        reason = "one block per fact and per export is the reading"
    )]
    pub fn read(located: &Located, store: &TensorStore) -> Result<Self, AudioError> {
        let names = Names::new(located.flavour);
        let w = |role| store.require(&names.weight(role));
        let conv0 = w(Stem::Conv0)?;
        let sub_channels = rows(&conv0);
        let sub_out = w(Stem::SubOut)?;
        let d_model = rows(&sub_out);
        let (layers, pred_layers) = stated_depths(located, store)?;
        let bias_u = store.require(&names.pos_biases(0).0)?;
        let (heads, head_dim) = match bias_u.shape.as_slice() {
            [h, d] => (
                usize::try_from(*h).unwrap_or(0),
                usize::try_from(*d).unwrap_or(0),
            ),
            other => {
                return Err(checkpoint(format!(
                    "{} has shape {other:?}, not [heads][head_dim]",
                    bias_u.name
                )));
            }
        };
        let ff = rows(&w(Stem::Ff(0, 1, 1))?);
        let dw = w(Stem::Dw(0))?;
        let conv_kernel = dw
            .shape
            .iter()
            .copied()
            .filter(|d| *d != 1)
            .nth(1)
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(0);
        let vocab = rows(&w(Stem::JointOut)?);
        let dec_proj = w(Stem::DecProj)?;
        let pred_hidden = rows(&dec_proj);
        let joint = rows(&w(Stem::EncProj)?);
        let prompt_in = w(Stem::Prompt1)?;
        let prompt_hidden = rows(&prompt_in);
        let prompts = cols(store, &prompt_in, located.affine_group().unwrap_or(1))?
            .checked_sub(d_model)
            .ok_or_else(|| checkpoint("the prompt linear is narrower than the encoder"))?;

        let (att_left, lookaheads, max_symbols, blank, mels, auto_prompt, dictionary) =
            match located.flavour {
                Flavour::Hf => {
                    let dict = dictionary_from_json(located.json("/processor/prompt_dictionary")?)?;
                    let auto = *dict
                        .get("auto")
                        .ok_or_else(|| checkpoint("the prompt dictionary has no auto entry"))?;
                    let lookaheads = located
                        .json("/encoder_config/supported_num_lookahead_tokens")?
                        .as_array()
                        .ok_or_else(|| checkpoint("supported_num_lookahead_tokens is not a list"))?
                        .iter()
                        .map(|v| {
                            v.as_u64()
                                .and_then(|v| usize::try_from(v).ok())
                                .ok_or_else(|| checkpoint("a lookahead is not a count"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (
                        located
                            .json_usize("/encoder_config/sliding_window")?
                            .checked_sub(1)
                            .ok_or_else(|| checkpoint("sliding_window is 0"))?,
                        lookaheads,
                        located.json_usize("/max_symbols_per_step")?,
                        u32::try_from(located.json_usize("/blank_token_id")?)
                            .map_err(|_| checkpoint("blank id"))?,
                        located.json_usize("/processor/feature_extractor/feature_size")?,
                        auto,
                        dict,
                    )
                }
                Flavour::Mlx => {
                    let dict = dictionary_from_json(located.json("/prompt/prompt_dictionary")?)?;
                    let auto = *dict
                        .get("auto")
                        .ok_or_else(|| checkpoint("the prompt dictionary has no auto entry"))?;
                    let contexts = located
                        .json("/encoder/att_context_size")?
                        .as_array()
                        .ok_or_else(|| checkpoint("att_context_size is not a list"))?;
                    let pair = |v: &Value, i: usize| {
                        v.get(i)
                            .and_then(Value::as_u64)
                            .and_then(|x| usize::try_from(x).ok())
                    };
                    let left = contexts
                        .first()
                        .and_then(|v| pair(v, 0))
                        .ok_or_else(|| checkpoint("att_context_size is empty"))?;
                    let lookaheads = contexts
                        .iter()
                        .map(|v| {
                            pair(v, 1).ok_or_else(|| {
                                checkpoint("an att_context_size entry is not a pair")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (
                        left,
                        lookaheads,
                        located.json_usize("/max_symbols")?,
                        u32::try_from(located.json_usize("/decoder/vocab_size")?)
                            .map_err(|_| checkpoint("blank id"))?,
                        located.json_usize("/preprocessor/features")?,
                        auto,
                        dict,
                    )
                }
                Flavour::GgufAsr => {
                    let mut dict = BTreeMap::new();
                    for entry in gguf_strings(store, "asr.rnnt.prompt_dictionary")? {
                        let (tag, id) = entry.rsplit_once(':').ok_or_else(|| {
                            checkpoint(format!("prompt entry {entry:?} has no id"))
                        })?;
                        dict.insert(
                            tag.to_owned(),
                            id.parse()
                                .map_err(|_| checkpoint(format!("prompt entry {entry:?}")))?,
                        );
                    }
                    let auto = *dict
                        .get("auto")
                        .ok_or_else(|| checkpoint("the prompt dictionary has no auto entry"))?;
                    (
                        gguf_usize(store, "asr.encoder.train_left_ctx")?,
                        FAMILY_LOOKAHEADS.to_vec(),
                        gguf_usize(store, "asr.rnnt.max_symbols_per_step")?,
                        u32::try_from(gguf_usize(store, "asr.rnnt.blank_id")?)
                            .map_err(|_| checkpoint("blank id"))?,
                        gguf_usize(store, "asr.encoder.feat_in")?,
                        auto,
                        dict,
                    )
                }
                Flavour::GgufParakeet => {
                    let tags = gguf_strings(store, "stt.parakeet.prompt.dictionary.locales")?;
                    let ids = gguf_usizes(store, "stt.parakeet.prompt.dictionary.indices")?;
                    if tags.len() != ids.len() {
                        return Err(checkpoint(
                            "the prompt dictionary's locales and indices differ in length",
                        ));
                    }
                    let choices =
                        gguf_usizes(store, "stt.parakeet.encoder.att_context_size_choices")?;
                    (
                        gguf_usize(store, "stt.parakeet.encoder.att_context_left")?,
                        choices.iter().skip(1).step_by(2).copied().collect(),
                        FAMILY_MAX_SYMBOLS,
                        u32::try_from(gguf_usize(store, "tokenizer.ggml.blank_token_id")?)
                            .map_err(|_| checkpoint("blank id"))?,
                        gguf_usize(store, "stt.frontend.num_mels")?,
                        gguf_usize(store, "stt.parakeet.prompt.auto_id")?,
                        tags.into_iter().zip(ids).collect(),
                    )
                }
            };
        let mut lookaheads = lookaheads;
        lookaheads.sort_unstable();
        lookaheads.dedup();
        let config = Self {
            d_model,
            layers,
            heads,
            head_dim,
            ff,
            conv_kernel,
            sub_channels,
            mels,
            vocab,
            blank,
            pred_hidden,
            pred_layers,
            joint,
            prompts,
            prompt_hidden,
            att_left,
            lookaheads,
            max_symbols,
            auto_prompt,
            dictionary,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), AudioError> {
        let fail = |what: &str| {
            Err(checkpoint(format!(
                "{what} is not something the kernels are written for"
            )))
        };
        if self.heads * self.head_dim != self.d_model {
            return fail("heads x head_dim != d_model");
        }
        if self.head_dim > 128 || !self.head_dim.is_multiple_of(2) {
            return fail("a head wider than 128 or odd");
        }
        if self.conv_kernel == 0 || self.conv_kernel > 16 {
            return fail("a conv kernel past 16 taps");
        }
        if usize::try_from(self.blank).map_or(true, |b| b + 1 != self.vocab) {
            return fail("a blank that is not the last class");
        }
        if self.layers == 0 || self.pred_layers == 0 || self.lookaheads.is_empty() {
            return fail("an empty stack");
        }
        Ok(())
    }

    /// The prompt id `language` selects.
    ///
    /// # Errors
    ///
    /// [`AudioError::UnknownLanguage`] for a tag the dictionary lacks.
    pub fn prompt_id(&self, language: &Language) -> Result<usize, AudioError> {
        match language {
            Language::Auto => Ok(self.auto_prompt),
            Language::Tag(tag) => self
                .dictionary
                .get(tag.as_str())
                .copied()
                .ok_or_else(|| AudioError::UnknownLanguage { tag: tag.clone() }),
        }
    }

    /// Seconds per encoder frame: the 10 ms hop times the 8x subsampling.
    #[must_use]
    pub fn frame_secs(&self) -> f64 {
        0.08
    }
}

/// The checkpoint's vocabulary, blank excluded.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] or [`AudioError::Vocab`] for a vocabulary that
/// is missing or not dense.
pub fn read_vocab(located: &Located, store: &TensorStore) -> Result<PieceVocab, AudioError> {
    match located.flavour {
        Flavour::Hf => {
            let path = located.dir.join("tokenizer.json");
            let text = std::fs::read_to_string(&path).map_err(|source| AudioError::Io {
                what: "reading tokenizer.json",
                source,
            })?;
            PieceVocab::from_tokenizer_json(&text)
        }
        Flavour::Mlx => {
            let list = located
                .json("/vocabulary")?
                .as_array()
                .ok_or_else(|| checkpoint("vocabulary is not a list"))?;
            let pieces = list
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| checkpoint("a vocabulary entry is not a string"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            PieceVocab::from_pieces(pieces)
        }
        Flavour::GgufAsr => PieceVocab::from_pieces(gguf_strings(store, "asr.tokenizer.vocab")?),
        Flavour::GgufParakeet => {
            PieceVocab::from_pieces(gguf_strings(store, "tokenizer.ggml.tokens")?)
        }
    }
}

/// The checkpoint's own mel filterbank, `[mels][n_fft/2 + 1]`, when it ships one.
///
/// `None` is not a refusal: it means the export carries no filterbank and the
/// frontend computes the standard one.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] for a filterbank of the wrong size.
pub fn read_filterbank(
    store: &TensorStore,
    mels: usize,
    bins: usize,
) -> Result<Option<Vec<f32>>, AudioError> {
    let Some(t) = store.get("preprocessor.fb") else {
        return Ok(None);
    };
    let values = dequant(&t)?;
    if values.len() != mels * bins {
        return Err(checkpoint(format!(
            "preprocessor.fb holds {} values, not {mels} x {bins}",
            values.len()
        )));
    }
    Ok(Some(values))
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;

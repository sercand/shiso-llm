//! The array-framework checkpoint directory: header, shards, and on-demand
//! decode.
//!
//! This is the crate's second container. Where [`crate::reader`] describes a
//! single self-contained file whose payloads the kernels read in place, a
//! checkpoint directory is several files plus a JSON config, and its weights
//! are in a packed format no kernel in this project reads. So the two
//! containers cannot behave the same way, and pretending otherwise would put
//! the difference somewhere worse:
//!
//! - **Describing is still eager.** Opening the directory parses every shard
//!   header, translates every tensor name, derives every output shape, and
//!   checks that the result is a complete model. Nothing is decoded.
//! - **Decoding is lazy and unbuffered.** A weight is unpacked when someone
//!   asks for it, straight from the mapping into a fresh buffer. The file does
//!   not accumulate decoded tensors on the way past, because a decoded model is
//!   several times larger than the packed one and would not fit anywhere
//!   useful.
//! - **The difference is visible in the signature.** [`SafetensorsFile::get`]
//!   has to park its result somewhere with a stable address, so it fills a
//!   per-tensor slot and never frees it - that is the API for tests and small
//!   reads. A loader streaming a whole model uses
//!   [`SafetensorsFile::tensor_bytes`], which hands back an owned buffer the
//!   caller drops as soon as it has been uploaded.
//!
//! # What this loader will not open
//!
//! A checkpoint directory here is multimodal: beside the language model sit a
//! vision tower and a speculative-decoding head, in their own shards, with
//! their own names. This loader follows the shard index and opens only the
//! shards the language-model tensors actually live in. The sibling files are
//! not skipped by name-matching after the fact - they are never mapped at all,
//! which is the difference between "we ignore 667 MB" and "we never read it".
//!
//! # Where the quantization parameters come from
//!
//! Bits per weight is *derived from the packed shape*: a row of `p` words and
//! `g` groups of `group_size` weights can only mean `p * 32 / (g * group_size)`
//! bits. The config's per-tensor table is then checked against that derivation
//! and a disagreement stops the load. The sidecar metadata file some exporters
//! write is not read at all: it is a summary, it can be stale, and a stale
//! bit-width decodes to a matrix that is wrong everywhere.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;

use half::{bf16, f16};
use serde::Deserialize;
use shiso_core::DType;

use crate::mlx_quant::{MlxAffineLayout, dequant_affine};
use crate::name_map::{GdnHeads, HeadAxis, HeadPermute, IdentityNames, NameMap, Role, ValueOp};
use crate::reader::{GgufError, GgufResult, TensorDesc, TensorShape, TensorView};
use crate::source::TensorSource;
use crate::st_header::{RawTensor, Shard, StDType, missing_file};

/// Trailing component of a packed weight's scale array.
pub(crate) const SCALES_SUFFIX: &str = ".scales";
/// Trailing component of a packed weight's bias array.
pub(crate) const BIASES_SUFFIX: &str = ".biases";
/// Trailing component of a weight array, packed or not.
pub(crate) const WEIGHT_SUFFIX: &str = ".weight";

/// A packed affine weight as stored: words, scales, biases, and the layout that decodes them.
#[derive(Debug, Clone, Copy)]
pub struct AffineTriple<'a> {
    /// Packed storage words.
    pub words: &'a [u8],
    /// Group scales.
    pub scales: &'a [u8],
    /// Group biases.
    pub biases: &'a [u8],
    /// Layout describing how to unpack the triple.
    pub layout: MlxAffineLayout,
    /// The half-precision format the scales and biases are stored in.
    pub scale_format: ScaleFormat,
}

/// The half-precision format a packed affine weight's scales and biases are
/// stored in.
///
/// Both are two bytes a value and nothing in the bytes says which, so it is
/// carried beside them: a kernel that read one as the other would decode
/// numbers right to three digits and wrong in the fourth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleFormat {
    /// Brain floats, what MLX writes by default.
    Bf16,
    /// IEEE half precision, what a re-encoding of `f16`-scaled blocks keeps.
    F16,
}

/// A packed affine weight in the binding layer's order: borrowed from the
/// mapping when no value-head reordering applies, reordered copies otherwise.
#[derive(Debug, Clone)]
pub struct PackedAffine<'a> {
    /// Packed storage words.
    pub words: Cow<'a, [u8]>,
    /// Group scales.
    pub scales: Cow<'a, [u8]>,
    /// Group biases.
    pub biases: Cow<'a, [u8]>,
    /// Layout describing how to unpack the triple.
    pub layout: MlxAffineLayout,
    /// The half-precision format the scales and biases are stored in.
    pub scale_format: ScaleFormat,
}

/// Architecture this container reports, matching the name the binding binds.
const ARCH: &str = "qwen35";
/// Architecture sub-key holding the layer count.
const BLOCK_COUNT_KEY: &str = "block_count";
/// Architecture sub-key holding the attention-layer stride.
const INTERVAL_KEY: &str = "full_attention_interval";
/// Quantization mode this loader implements.
const AFFINE_MODE: &str = "affine";
/// Producer tag the shard headers must carry.
const HEADER_FORMAT: &str = "mlx";
/// Config file naming the architecture and the quantization parameters.
const CONFIG_FILE: &str = "config.json";
/// Shard every directory has, and the only one a single-file export writes.
const PRIMARY_SHARD: &str = "model.safetensors";
/// Optional map from tensor name to shard, authoritative when present.
const SHARD_INDEX_FILE: &str = "model.safetensors.index.json";
/// Bits in one packed storage word.
const WORD_BITS: u64 = 32;

/// The parts of the checkpoint config this loader reads.
///
/// The document is a multimodal wrapper, so every language-model dimension
/// lives one level down under `text_config`. Reading the wrapper's own
/// `model_type` would report the multimodal family and miss a text half from a
/// different generation.
#[derive(Debug, Default, Deserialize)]
struct CheckpointConfig {
    /// Top-level model type when present.
    #[serde(default)]
    model_type: Option<String>,
    /// Language-model half of the config, when present.
    #[serde(default)]
    text_config: Option<TextConfig>,
    /// Quantization parameters under `quantization`.
    #[serde(default)]
    quantization: Option<QuantConfig>,
    /// Quantization parameters under `quantization_config`.
    #[serde(default)]
    quantization_config: Option<QuantConfig>,
}

impl CheckpointConfig {
    fn quant_config(&self) -> QuantConfig {
        self.quantization_config
            .clone()
            .or_else(|| self.quantization.clone())
            .unwrap_or_default()
    }
}

/// The language-model half of the config.
#[derive(Debug, Deserialize)]
struct TextConfig {
    /// Architecture of the text half.
    model_type: String,
    /// Number of layer slots.
    num_hidden_layers: u32,
    /// Every `interval`-th layer slot holds attention instead of recurrence.
    #[serde(default)]
    full_attention_interval: u32,
    /// Value heads in a recurrent block.
    #[serde(default)]
    linear_num_value_heads: usize,
    /// Query and key heads in a recurrent block.
    #[serde(default)]
    linear_num_key_heads: usize,
    /// Channels in one recurrent value head.
    #[serde(default)]
    linear_value_head_dim: usize,
    /// Channels in one recurrent query or key head.
    #[serde(default)]
    linear_key_head_dim: usize,
}

impl TextConfig {
    /// Head structure the value-head reordering is derived from.
    fn gdn_heads(&self) -> GdnHeads {
        GdnHeads {
            value_heads: self.linear_num_value_heads,
            key_heads: self.linear_num_key_heads,
            value_head_dim: self.linear_value_head_dim,
            key_head_dim: self.linear_key_head_dim,
        }
    }
}

/// Quantization parameters: the defaults, then the per-tensor overrides.
#[derive(Debug, Clone, Deserialize)]
struct QuantConfig {
    /// Weights sharing one scale and bias, unless a tensor overrides it.
    #[serde(default = "default_group_size")]
    group_size: u32,
    /// Bits per weight, unless a tensor overrides it.
    #[serde(default = "default_bits")]
    bits: u32,
    /// Quantization scheme; only the affine one is implemented.
    #[serde(default = "default_mode")]
    mode: String,
    /// One entry per quantized tensor, keyed by its full checkpoint name.
    #[serde(flatten)]
    per_tensor: HashMap<String, serde_json::Value>,
}

fn default_group_size() -> u32 {
    64
}

fn default_bits() -> u32 {
    4
}

fn default_mode() -> String {
    AFFINE_MODE.to_string()
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            group_size: default_group_size(),
            bits: default_bits(),
            mode: default_mode(),
            per_tensor: HashMap::new(),
        }
    }
}

impl QuantConfig {
    /// Parameters the config claims for one tensor, falling back to the
    /// defaults when the tensor has no entry of its own.
    fn for_tensor(&self, name: &str) -> TensorQuant {
        let direct = self.per_tensor.get(name);
        let fallback = if direct.is_none() {
            self.per_tensor
                .iter()
                .find(|(k, _)| {
                    k.ends_with(name)
                        || name.ends_with(k.as_str())
                        || k.strip_prefix("language_model.").is_some_and(|s| s == name)
                        || name
                            .strip_prefix("language_model.")
                            .is_some_and(|s| s == k.as_str())
                })
                .map(|(_, v)| v)
        } else {
            direct
        };

        fallback
            .cloned()
            .and_then(|value| serde_json::from_value::<TensorQuant>(value).ok())
            .unwrap_or(TensorQuant {
                bits: self.bits,
                group_size: self.group_size,
            })
    }
}

/// One tensor's quantization parameters.
#[derive(Debug, Clone, Copy, Deserialize)]
struct TensorQuant {
    /// Bits per weight the config claims.
    bits: u32,
    /// Weights per scale and bias.
    group_size: u32,
}

/// The shard index, when the directory has one.
#[derive(Debug, Deserialize)]
struct ShardIndex {
    /// Tensor name to the shard file holding it, relative to the directory.
    weight_map: HashMap<String, String>,
}

/// Where one source array lives and what it holds.
#[derive(Debug, Clone, Copy)]
struct Source {
    /// Index into the open shard list.
    shard: usize,
    /// Payload start, relative to that shard's payload base.
    start: u64,
    /// Payload end, relative to that shard's payload base.
    end: u64,
    /// Storage format of the array.
    dtype: StDType,
}

/// How a translated tensor is produced from its source arrays.
#[derive(Debug)]
enum Plan {
    /// Widen one array to `f32`, applying the role's value transform.
    Widen {
        /// The single source array.
        src: Source,
        /// Transform applied after widening.
        op: ValueOp,
    },
    /// Decode a packed affine triple to `f16`.
    Affine {
        /// Packed `u32` words.
        weight: Source,
        /// Per-group scales.
        scales: Source,
        /// Per-group biases.
        biases: Source,
        /// Geometry the three arrays agree on.
        layout: MlxAffineLayout,
    },
}

/// A decoded tensor, kept in the typed form it was decoded into.
///
/// Storing `Box<[f32]>` and `Box<[f16]>` rather than `Box<[u8]>` is what lets
/// [`SafetensorsFile::get`] hand out a [`TensorView`] whose payload is aligned
/// for the typed accessors; a byte buffer would be correctly sized and
/// arbitrarily aligned.
#[derive(Debug)]
enum Payload {
    /// Widened norm or control vector.
    F32(Box<[f32]>),
    /// Decoded weight matrix.
    F16(Box<[f16]>),
}

impl Payload {
    /// The payload as raw bytes, without copying.
    fn bytes(&self) -> &[u8] {
        match self {
            Self::F32(values) => bytemuck::cast_slice(values),
            Self::F16(values) => bytemuck::cast_slice(values),
        }
    }
}

/// One translated tensor: what it is, how to produce it, and where the copy
/// handed out by [`SafetensorsFile::get`] lives once it has been produced.
#[derive(Debug)]
struct Entry {
    /// The tensor as the binding layer sees it.
    desc: TensorDesc,
    /// How to produce its bytes.
    plan: Plan,
    /// Value-head reordering applied after the plan runs, `None` for the
    /// tensors the recurrent block's head order does not reach.
    permute: Option<HeadPermute>,
    /// The axis that reordering runs along, kept so a packed weight can be
    /// reordered in its packed form (see [`SafetensorsFile::affine_packed`]).
    axis: Option<HeadAxis>,
    /// Decoded copy, filled at most once and then kept alive for the file's
    /// lifetime, because a borrowed view has to point at a stable address.
    slot: OnceLock<Payload>,
}

/// An open checkpoint directory.
///
/// # Mutating the files
///
/// The shards stay mapped for as long as this value lives. Replacing,
/// truncating or rewriting any of them in the meantime is undefined behaviour,
/// so treat the directory as immutable while it is open.
#[derive(Debug)]
pub struct SafetensorsFile {
    /// Directory the checkpoint was opened from.
    dir: PathBuf,
    /// Mapped shards, in the order they were opened.
    shards: Vec<Shard>,
    /// Translated tensors, in checkpoint order.
    entries: Vec<Entry>,
    /// Name to index into `entries`.
    index: HashMap<Box<str>, usize>,
    /// Layer-slot count from the config.
    n_layers: u32,
    /// Attention-layer stride from the config.
    full_attention_interval: u32,
    /// Architecture this container reports.
    arch: String,
    /// Quantization configuration.
    quant: QuantConfig,
    /// Name mapping used for translation.
    names: Arc<dyn NameMap>,
    /// Head structure the value-head reorderings are derived from.
    heads: GdnHeads,
}

impl SafetensorsFile {
    /// Opens the checkpoint directory at `dir`, translating tensor names with `names`.
    ///
    /// Every shard holding a translated tensor is mapped and parsed, every
    /// tensor name is translated, and the result is checked for completeness
    /// against whatever `names` expects - so a directory that opens is one the
    /// binding layer can bind and the kernels can run.
    ///
    /// # Mutating the files
    ///
    /// Do not point this at a directory something else is writing, and do not
    /// replace or truncate a shard while the returned value is alive. The
    /// mappings are windows onto those files and the borrow checker cannot see
    /// the other writer.
    ///
    /// # Errors
    ///
    /// [`GgufError::Io`] when the directory is missing a required file or a
    /// header does not parse as JSON; [`GgufError::ArchMismatch`] when the
    /// quantization mode is not affine or a shard was written by another producer;
    /// [`GgufError::Truncated`] or [`GgufError::OutOfBounds`] for a header or
    /// payload span that does not fit its file; [`GgufError::LayoutMismatch`] for
    /// an unknown storage format or a packed triple whose shapes disagree;
    /// [`GgufError::MissingTensor`] when a packed weight has no scales or biases;
    /// and [`GgufError::ProfileMismatch`] when a derived bit width contradicts
    /// the config or the translated tensor count is not the one `names` expects.
    pub fn open_with(dir: impl AsRef<Path>, names: Arc<dyn NameMap>) -> GgufResult<Self> {
        let dir = dir.as_ref();
        let config = read_config(dir)?;
        let quant = config.quant_config();
        if quant.mode != AFFINE_MODE {
            return Err(GgufError::ArchMismatch {
                want: AFFINE_MODE,
                got: Box::from(quant.mode.as_str()),
            });
        }

        let mut shards = Vec::new();
        let mut arrays: HashMap<String, (usize, RawTensor)> = HashMap::new();
        for name in shard_files(dir, names.as_ref())? {
            let path = dir.join(&name);
            let (shard, tensors) = Shard::open(&path, HEADER_FORMAT)?;
            let slot = shards.len();
            shards.push(shard);
            for (tensor, raw) in tensors {
                arrays.insert(tensor, (slot, raw));
            }
        }

        let heads = config
            .text_config
            .as_ref()
            .map_or_else(GdnHeads::default, TextConfig::gdn_heads);
        let entries = translate(&arrays, &quant, heads, names.as_ref())?;

        let n_layers = config
            .text_config
            .as_ref()
            .map_or(0, |tc| tc.num_hidden_layers);
        let full_attention_interval = config
            .text_config
            .as_ref()
            .map_or(0, |tc| tc.full_attention_interval);
        // Completeness, for a family that can state it: every tensor the
        // binder looks up being present says nothing about a tensor it should
        // have looked up and did not, and a checkpoint short a layer binds
        // cleanly and answers wrongly.
        let optional = entries
            .iter()
            .filter(|entry| names.optional_tensors().contains(&&*entry.desc.name))
            .count();
        if let Some(want) = names.expected_tensors(n_layers, full_attention_interval)
            && entries.len() - optional != want
        {
            return Err(GgufError::ProfileMismatch {
                key: Box::from("tensor_count"),
                want: u64::try_from(want).unwrap_or(u64::MAX),
                got: u64::try_from(entries.len() - optional).unwrap_or(u64::MAX),
            });
        }

        let index = entries
            .iter()
            .enumerate()
            .map(|(at, entry)| (entry.desc.name.clone(), at))
            .collect();
        let arch = resolve_arch(&config);

        tracing::debug!(
            dir = %dir.display(),
            shards = shards.len(),
            tensors = entries.len(),
            layers = n_layers,
            arch = %arch,
            "opened checkpoint directory"
        );
        Ok(Self {
            dir: dir.to_path_buf(),
            shards,
            entries,
            index,
            n_layers,
            full_attention_interval,
            arch,
            quant,
            names,
            heads,
        })
    }

    /// Opens the checkpoint directory at `dir` with the default identity name map.
    ///
    /// # Errors
    ///
    /// Whatever [`SafetensorsFile::open_with`] reports.
    pub fn open(dir: impl AsRef<Path>) -> GgufResult<Self> {
        Self::open_with(dir, Arc::new(IdentityNames))
    }

    /// The directory this checkpoint was opened from.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }
    /// Name mapping used for translation.
    #[must_use]
    pub fn names(&self) -> &dyn NameMap {
        self.names.as_ref()
    }

    /// Every translated tensor, in checkpoint order.
    #[must_use]
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &TensorDesc> {
        self.entries.iter().map(|entry| &entry.desc)
    }

    /// Decoded bytes for `name`.
    ///
    /// Borrowed when the checkpoint already stores the tensor in its output
    /// form, owned when it had to be widened or unpacked. The distinction is in
    /// the return type on purpose: a caller uploading weights to a device needs
    /// to know whether the bytes it is looking at are a file mapping it can
    /// wrap, or a heap buffer it has to copy and then drop.
    ///
    /// This never fills the per-tensor slot [`SafetensorsFile::get`] uses, so a
    /// loop over a whole model peaks at one decoded tensor rather than all of
    /// them.
    ///
    /// # Errors
    ///
    /// [`GgufError::MissingTensor`] when no tensor has that name, plus the
    /// decode errors of [`crate::mlx_quant::dequant_affine`] and
    /// [`GgufError::OutOfBounds`] for a payload span outside its shard.
    pub fn tensor_bytes(&self, name: &str) -> GgufResult<Cow<'_, [u8]>> {
        let entry = self.entry(name)?;
        if let Plan::Widen { src, op } = &entry.plan
            && *op == ValueOp::Keep
            && entry.permute.is_none()
            && src.dtype == StDType::F32
        {
            return Ok(Cow::Borrowed(self.bytes(src)?));
        }
        // The one copy this crate makes: the decode produces typed values, and
        // an owned byte buffer is what a device upload wants. Copying is
        // cheaper than keeping the decoded model resident, which is the only
        // alternative.
        Ok(Cow::Owned(self.decode(entry)?.bytes().to_vec()))
    }
    /// Returns the packed affine triple (words, scales, biases, layout) for `name`.
    ///
    /// # Errors
    ///
    /// Returns shard I/O or bounds errors; returns `Ok(None)` when `name` has no
    /// `.scales`/`.biases` siblings (a plain tensor).
    pub fn affine_triple(&self, name: &str) -> GgufResult<Option<AffineTriple<'_>>> {
        let entry = self.find_entry(name)?;
        match &entry.plan {
            Plan::Affine {
                weight,
                scales,
                biases,
                layout,
            } => Ok(Some(AffineTriple {
                words: self.bytes(weight)?,
                scales: self.bytes(scales)?,
                biases: self.bytes(biases)?,
                layout: *layout,
                scale_format: scale_format(scales.dtype),
            })),
            Plan::Widen { .. } => Ok(None),
        }
    }

    /// The packed affine triple for `name` in the binding layer's order: the
    /// value-head reordering its role names applied to the packed words, scales
    /// and biases, so the weight never has to be decoded to be reordered.
    ///
    /// A reordering along the rows moves whole packed rows - a row's words and
    /// its groups' scales and biases travel together. One along the columns
    /// moves whole value heads, which is possible packed only because a head's
    /// channels are a whole number of groups: `value_head_dim` of 128 is two
    /// groups of 64, sixteen 4-bit words or thirty-two 8-bit ones. A geometry
    /// where that fails is refused rather than decoded.
    ///
    /// A tensor with no reordering is borrowed straight from the mapping, as
    /// [`SafetensorsFile::affine_triple`] returns it.
    ///
    /// # Errors
    ///
    /// Shard I/O or bounds errors, and [`GgufError::LayoutMismatch`] when a
    /// column reordering does not fall on group and word boundaries.
    // None: `name` is a plain tensor with no `.scales`/`.biases` siblings.
    pub fn affine_packed(&self, name: &str) -> GgufResult<Option<PackedAffine<'_>>> {
        let entry = self.find_entry(name)?;
        let Some(triple) = self.affine_triple(name)? else {
            return Ok(None);
        };
        let Some(axis) = entry.axis else {
            return Ok(Some(PackedAffine {
                words: Cow::Borrowed(triple.words),
                scales: Cow::Borrowed(triple.scales),
                biases: Cow::Borrowed(triple.biases),
                layout: triple.layout,
                scale_format: triple.scale_format,
            }));
        };
        let layout = triple.layout;
        let bits = layout.bits as usize;
        let rows = layout.out_rows;
        let words_per_row = layout.in_dim * bits / 32;
        let groups_per_row = layout.in_dim / layout.group_size;
        // The head structure, restated in the units each array counts in. A
        // row reordering needs nothing restated: its block is a whole row of
        // whatever the array holds. A column one counts a head's channels in
        // words for the packed array and in groups for the other two.
        let (word_heads, group_heads) = if axis == HeadAxis::ValueColumns {
            let dim = self.heads.value_head_dim;
            if !(dim * bits).is_multiple_of(32) || !dim.is_multiple_of(layout.group_size) {
                return Err(GgufError::LayoutMismatch {
                    tensor: name.into(),
                    want: Box::from("value heads a whole number of words and groups wide"),
                    got: Box::from(format!(
                        "{dim} channels at {bits} bits in groups of {}",
                        layout.group_size
                    )),
                });
            }
            let mut words = self.heads;
            words.value_head_dim = dim * bits / 32;
            let mut groups = self.heads;
            groups.value_head_dim = dim / layout.group_size;
            (words, groups)
        } else {
            (self.heads, self.heads)
        };
        // Parsed rather than cast: an MLX payload need not sit at an aligned offset.
        let mut words: Vec<u32> = le_words(triple.words, u32::from_le_bytes);
        HeadPermute::new(axis, word_heads, rows, words_per_row, name)?.apply(&mut words, name)?;
        let permute_groups = HeadPermute::new(axis, group_heads, rows, groups_per_row, name)?;
        let mut scales: Vec<u16> = le_words(triple.scales, u16::from_le_bytes);
        permute_groups.apply(&mut scales, name)?;
        let mut biases: Vec<u16> = le_words(triple.biases, u16::from_le_bytes);
        permute_groups.apply(&mut biases, name)?;
        Ok(Some(PackedAffine {
            words: Cow::Owned(words.iter().flat_map(|w| w.to_le_bytes()).collect()),
            scales: Cow::Owned(scales.iter().flat_map(|w| w.to_le_bytes()).collect()),
            biases: Cow::Owned(biases.iter().flat_map(|w| w.to_le_bytes()).collect()),
            layout,
            scale_format: triple.scale_format,
        }))
    }

    /// Bits per weight for `name` according to the checkpoint's quantization configuration.
    #[must_use]
    pub fn bits_for(&self, name: &str) -> u32 {
        if let Ok(entry) = self.find_entry(name)
            && let Plan::Affine { layout, .. } = &entry.plan
        {
            return layout.bits;
        }
        self.quant.for_tensor(name).bits
    }

    fn find_entry(&self, name: &str) -> GgufResult<&Entry> {
        if let Ok(entry) = self.entry(name) {
            return Ok(entry);
        }
        if let Some(stripped) = name.strip_suffix(WEIGHT_SUFFIX)
            && let Ok(entry) = self.entry(stripped)
        {
            return Ok(entry);
        }
        let with_weight = format!("{name}{WEIGHT_SUFFIX}");
        if let Ok(entry) = self.entry(&with_weight) {
            return Ok(entry);
        }
        Err(GgufError::MissingTensor { name: name.into() })
    }

    /// Looks an entry up, or reports the name that was missing.
    fn entry(&self, name: &str) -> GgufResult<&Entry> {
        self.index
            .get(name)
            .and_then(|&at| self.entries.get(at))
            .ok_or_else(|| GgufError::MissingTensor { name: name.into() })
    }

    /// Source bytes for one array.
    fn bytes(&self, src: &Source) -> GgufResult<&[u8]> {
        let shard = self
            .shards
            .get(src.shard)
            .ok_or_else(|| GgufError::OutOfBounds {
                tensor: Box::from(self.dir.to_string_lossy().as_ref()),
            })?;
        shard.payload(src.start, src.end)
    }

    /// Produces one tensor's values.
    ///
    /// The value-head reordering runs last, on the decoded values rather than
    /// on the packed ones: a packed row is a run of sub-byte fields with its
    /// own scales beside it, and permuting *that* would have to move three
    /// arrays in step. Reordering after the unpack costs one pass over a
    /// tensor that has just been written anyway.
    fn decode(&self, entry: &Entry) -> GgufResult<Payload> {
        match &entry.plan {
            Plan::Widen { src, op } => {
                let mut values = widen(self.bytes(src)?, src.dtype, &entry.desc.name)?;
                if *op == ValueOp::NegExp {
                    for value in &mut values {
                        *value = -value.exp();
                    }
                }
                if let Some(permute) = &entry.permute {
                    permute.apply(&mut values, &entry.desc.name)?;
                }
                Ok(Payload::F32(values.into_boxed_slice()))
            }
            Plan::Affine {
                weight,
                scales,
                biases,
                layout,
            } => {
                if scales.dtype != StDType::Bf16 {
                    return Err(GgufError::LayoutMismatch {
                        tensor: entry.desc.name.clone(),
                        want: Box::from("BF16 scales, the only ones the host decode reads"),
                        got: Box::from("F16 scales, which are read packed only"),
                    });
                }
                let mut values = Vec::new();
                dequant_affine(
                    self.bytes(weight)?,
                    self.bytes(scales)?,
                    self.bytes(biases)?,
                    *layout,
                    &mut values,
                )?;
                if let Some(permute) = &entry.permute {
                    permute.apply(&mut values, &entry.desc.name)?;
                }
                Ok(Payload::F16(values.into_boxed_slice()))
            }
        }
    }
}

impl TensorSource for SafetensorsFile {
    fn architecture(&self) -> Option<&str> {
        Some(&self.arch)
    }

    fn arch_u32(&self, suffix: &str) -> Option<u32> {
        match suffix {
            BLOCK_COUNT_KEY => Some(self.n_layers),
            INTERVAL_KEY => Some(self.full_attention_interval),
            _ => None,
        }
    }

    fn find(&self, name: &str) -> Option<&TensorDesc> {
        self.entry(name).ok().map(|entry| &entry.desc)
    }

    fn get(&self, name: &str) -> Option<TensorView<'_>> {
        let entry = self.entry(name).ok()?;
        if entry.slot.get().is_none() {
            match self.decode(entry) {
                // A racing caller may have filled the slot first; either copy
                // is the same bytes, so the loser of the race simply drops its
                // own and reads the winner's.
                Ok(payload) => drop(entry.slot.set(payload)),
                Err(err) => {
                    // The trait has no error channel, and widening it would put
                    // a decode failure in the signature of every container.
                    // Reporting `None` lets the binding layer name the tensor.
                    tracing::error!(tensor = name, error = %err, "decoding a tensor failed");
                    return None;
                }
            }
        }
        let payload = entry.slot.get()?;
        Some(TensorView {
            name: &entry.desc.name,
            dtype: entry.desc.dtype,
            data: payload.bytes(),
            shape: entry.desc.shape,
        })
    }
}

/// Widens one source array to `f32`.
fn widen(bytes: &[u8], dtype: StDType, tensor: &str) -> GgufResult<Vec<f32>> {
    match dtype {
        StDType::Bf16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)).to_f32())
            .collect()),
        StDType::F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f16::from_bits(u16::from_le_bytes(*pair)).to_f32())
            .collect()),
        StDType::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|quad| f32::from_le_bytes(*quad))
            .collect()),
        // The three formats this container never widens. `U32` is a packed
        // affine row, which has no meaning without the scales and biases
        // beside it; `U8` and `F8_E4M3` belong to the modelopt container and
        // reach this one only through a header that claimed to be MLX's.
        StDType::U32
        | StDType::U8
        | StDType::F8E4M3
        | StDType::I64
        | StDType::I16
        | StDType::I32 => Err(GgufError::LayoutMismatch {
            tensor: tensor.into(),
            want: Box::from("a float array"),
            got: Box::from("packed bytes with no scales"),
        }),
    }
}

/// Reads and parses the checkpoint config.
fn read_config(dir: &Path) -> GgufResult<CheckpointConfig> {
    let path = dir.join(CONFIG_FILE);
    let text = std::fs::read_to_string(&path).map_err(|err| missing_file(&path, &err))?;
    serde_json::from_str(&text).map_err(|err| {
        GgufError::Io(io::Error::other(format!(
            "{}: config is not a checkpoint config: {err}",
            path.display()
        )))
    })
}

/// Names the shards that hold language-model tensors, relative to `dir`.
///
/// When a shard index is present it is authoritative, including for a
/// single-shard export: following it is what keeps the sibling towers' files
/// out of the mapping list. Without one, the directory is a single-file export
/// and the primary shard is the whole model.
fn shard_files(dir: &Path, names: &dyn NameMap) -> GgufResult<Vec<String>> {
    let index_path = dir.join(SHARD_INDEX_FILE);
    if !index_path.is_file() {
        let primary = dir.join(PRIMARY_SHARD);
        if !primary.is_file() {
            return Err(missing_file(
                &primary,
                &io::Error::from(io::ErrorKind::NotFound),
            ));
        }
        return Ok(vec![PRIMARY_SHARD.to_string()]);
    }

    let text =
        std::fs::read_to_string(&index_path).map_err(|err| missing_file(&index_path, &err))?;
    let index: ShardIndex = serde_json::from_str(&text).map_err(|err| {
        GgufError::Io(io::Error::other(format!(
            "{}: shard index does not parse: {err}",
            index_path.display()
        )))
    })?;

    let mut files: Vec<String> = Vec::new();
    for (tensor, shard) in &index.weight_map {
        let matches = names.map_role(tensor).is_some()
            || tensor
                .strip_suffix(WEIGHT_SUFFIX)
                .and_then(|b| names.map_role(b))
                .is_some()
            || tensor
                .strip_suffix(SCALES_SUFFIX)
                .and_then(|b| names.map_role(b))
                .is_some()
            || tensor
                .strip_suffix(BIASES_SUFFIX)
                .and_then(|b| names.map_role(b))
                .is_some();
        if matches && !files.contains(shard) {
            files.push(shard.clone());
        }
    }
    if files.is_empty() {
        return Err(GgufError::MissingTensor {
            name: Box::from("no shards matched name map"),
        });
    }
    files.sort();
    Ok(files)
}

fn resolve_arch(config: &CheckpointConfig) -> String {
    // The document's own type first, then its text subtree's: a wrapper type
    // that names no family this reader knows (a rotated export names its
    // storage scheme there) still carries the family in the subtree.
    let known = |mt: &str| {
        if mt.starts_with("qwen3") {
            Some("qwen35".to_string())
        } else if mt.starts_with("gemma4") {
            Some("gemma4".to_string())
        } else {
            None
        }
    };
    let top = config.model_type.as_deref();
    let text = config.text_config.as_ref().map(|tc| tc.model_type.as_str());
    top.and_then(known)
        .or_else(|| text.and_then(known))
        .or_else(|| top.or(text).map(str::to_string))
        .unwrap_or_else(|| ARCH.to_string())
}

/// Groups the raw arrays into tensors and translates each one.
///
/// A packed weight is three arrays that share a base name; everything else is
/// one array whose name already ends in its own role. Grouping on the trailing
/// component is what lets both spellings go through the same name table.
fn translate(
    arrays: &HashMap<String, (usize, RawTensor)>,
    quant: &QuantConfig,
    heads: GdnHeads,
    names: &dyn NameMap,
) -> GgufResult<Vec<Entry>> {
    let mut entries = Vec::new();
    for (name, (shard, raw)) in arrays {
        if name.ends_with(SCALES_SUFFIX) || name.ends_with(BIASES_SUFFIX) {
            continue;
        }
        let dtype = raw.parsed_dtype(name)?;
        let packed_base = name
            .strip_suffix(WEIGHT_SUFFIX)
            .filter(|_| dtype == StDType::U32);
        let role_key = packed_base.unwrap_or(name.as_str());
        let Some(role) = names.map_role(role_key) else {
            continue;
        };
        let entry = match packed_base {
            Some(base) => packed_entry(base, *shard, raw, arrays, quant, &role, heads)?,
            None => widened_entry(name, *shard, dtype, raw, &role, heads)?,
        };
        entries.push(entry);
    }
    entries.sort_by(|left, right| left.desc.name.cmp(&right.desc.name));
    Ok(entries)
}

/// Rewrites a checkpoint shape into the order the binding layer checks.
pub(crate) fn gguf_shape(checkpoint_dims: &[u64]) -> GgufResult<TensorShape> {
    if checkpoint_dims.is_empty() {
        return TensorShape::new(&[1]);
    }
    let mut dims = checkpoint_dims;
    while dims.len() > 1 && dims.last() == Some(&1) {
        dims = &dims[..dims.len() - 1];
    }
    if dims.is_empty() {
        return TensorShape::new(&[1]);
    }
    let reversed: Vec<u64> = dims.iter().rev().copied().collect();
    TensorShape::new(&reversed)
}

/// Builds the entry for an unpacked tensor, which is widened to `f32`.
///
/// The reordering is resolved against the *checkpoint* shape rather than the
/// reversed one the binding layer sees, because the checkpoint shape is the
/// memory order: its leading axis is the slowest-varying one, which is the axis
/// the value-head index runs along for every tensor that carries one.
fn widened_entry(
    name: &str,
    shard: usize,
    dtype: StDType,
    raw: &RawTensor,
    role: &Role,
    heads: GdnHeads,
) -> GgufResult<Entry> {
    let shape = gguf_shape(&raw.shape)?;
    let elements = shape.elements()?;
    let bytes = payload_bytes(elements, DType::F32).ok_or_else(|| GgufError::TensorTooLarge {
        tensor: name.into(),
    })?;
    let (rows, cols) = leading_and_rest(name, &raw.shape)?;
    let permute = role
        .heads
        .map(|axis| HeadPermute::new(axis, heads, rows, cols, name))
        .transpose()?;
    Ok(Entry {
        desc: TensorDesc {
            name: role.gguf.as_str().into(),
            dtype: DType::F32,
            shape,
            abs_offset: raw.data_offsets[0],
            bytes,
        },
        plan: Plan::Widen {
            src: Source {
                shard,
                start: raw.data_offsets[0],
                end: raw.data_offsets[1],
                dtype,
            },
            op: role.op,
        },
        permute,
        axis: role.heads,
        slot: OnceLock::new(),
    })
}

/// Splits a checkpoint shape into its leading axis and the values under it.
///
/// Row-major memory makes this the only split a reordering needs: the leading
/// axis indexes whole contiguous runs, and everything below it is one run's
/// length. Trailing unit axes are part of that length and cost nothing, so
/// unlike [`gguf_shape`] this does not strip them.
///
/// # Errors
///
/// [`GgufError::BadRank`] for a rank-zero shape, and
/// [`GgufError::ElementCountOverflow`] when the trailing axes multiply past
/// `usize`.
fn leading_and_rest(name: &str, dims: &[u64]) -> GgufResult<(usize, usize)> {
    if dims.is_empty() {
        return Ok((1, 1));
    }
    let (&rows, rest) = dims.split_first().ok_or(GgufError::BadRank { ndim: 0 })?;
    let mut cols: usize = 1;
    for &dim in rest {
        cols = usize::try_from(dim)
            .ok()
            .and_then(|dim| cols.checked_mul(dim))
            .ok_or(GgufError::ElementCountOverflow)?;
    }
    let rows = usize::try_from(rows).map_err(|_| GgufError::TensorTooLarge {
        tensor: name.into(),
    })?;
    Ok((rows, cols))
}

/// Builds the entry for a packed weight, which is decoded to `f16`.
///
/// The bit width is derived from the packed and scale shapes and only then
/// compared with the config: the shapes cannot be stale, and a config that
/// disagrees with them is a checkpoint whose two halves were written at
/// different times.
fn packed_entry(
    base: &str,
    shard: usize,
    raw: &RawTensor,
    arrays: &HashMap<String, (usize, RawTensor)>,
    quant: &QuantConfig,
    role: &Role,
    heads: GdnHeads,
) -> GgufResult<Entry> {
    let scales = companion(base, SCALES_SUFFIX, arrays)?;
    let biases = companion(base, BIASES_SUFFIX, arrays)?;
    let claimed = quant.for_tensor(base);

    let (out_rows, packed_words) = two_dims(base, &raw.shape)?;
    let (scale_rows, groups) = two_dims(base, &scales.1.shape)?;
    if scale_rows != out_rows || scales.1.shape != biases.1.shape {
        return Err(GgufError::LayoutMismatch {
            tensor: base.into(),
            want: Box::from("scales and biases shaped like the packed rows"),
            got: Box::from(format!("{:?} and {:?}", scales.1.shape, biases.1.shape)),
        });
    }
    let group_size = u64::from(claimed.group_size);
    let in_dim = groups
        .checked_mul(group_size)
        .ok_or_else(|| GgufError::TensorTooLarge {
            tensor: base.into(),
        })?;
    let packed_bits =
        packed_words
            .checked_mul(WORD_BITS)
            .ok_or_else(|| GgufError::TensorTooLarge {
                tensor: base.into(),
            })?;
    if in_dim == 0 || !packed_bits.is_multiple_of(in_dim) {
        return Err(GgufError::LayoutMismatch {
            tensor: base.into(),
            want: Box::from("a packed row that is a whole number of bits per weight"),
            got: Box::from(format!("{packed_words} words over {in_dim} weights")),
        });
    }
    let bits = packed_bits / in_dim;
    if u64::from(claimed.bits) != bits {
        return Err(GgufError::ProfileMismatch {
            key: Box::from(base),
            want: u64::from(claimed.bits),
            got: bits,
        });
    }
    let bits = u32::try_from(bits).map_err(|_| GgufError::LayoutMismatch {
        tensor: base.into(),
        want: Box::from("4 or 8 bits per weight"),
        got: Box::from(format!("{bits} bits")),
    })?;

    let shape = TensorShape::new(&[in_dim, out_rows])?;
    let elements = shape.elements()?;
    let bytes = payload_bytes(elements, DType::F16).ok_or_else(|| GgufError::TensorTooLarge {
        tensor: base.into(),
    })?;
    let layout = MlxAffineLayout::new(
        usize::try_from(out_rows).unwrap_or(usize::MAX),
        usize::try_from(in_dim).unwrap_or(usize::MAX),
        usize::try_from(group_size).unwrap_or(usize::MAX),
        bits,
    );
    // Against the *unpacked* geometry: the reordering runs after the decode,
    // where a row is `in_dim` values rather than `packed_words` words.
    let permute = role
        .heads
        .map(|axis| {
            let (rows, cols) = leading_and_rest(base, &[out_rows, in_dim])?;
            HeadPermute::new(axis, heads, rows, cols, base)
        })
        .transpose()?;
    // One format for both arrays: a triple reports one `ScaleFormat`, so a
    // file that stored its scales in one half format and its biases in the
    // other would have every group's bias decoded as the wrong one.
    let (scales, biases) = (half_source(base, scales)?, half_source(base, biases)?);
    if scales.dtype != biases.dtype {
        return Err(GgufError::LayoutMismatch {
            tensor: base.into(),
            want: Box::from("scales and biases in the same half format"),
            got: Box::from(format!(
                "{:?} scales, {:?} biases",
                scales.dtype, biases.dtype
            )),
        });
    }
    Ok(Entry {
        desc: TensorDesc {
            name: role.gguf.as_str().into(),
            dtype: DType::F16,
            shape,
            abs_offset: raw.data_offsets[0],
            bytes,
        },
        plan: Plan::Affine {
            weight: Source {
                shard,
                start: raw.data_offsets[0],
                end: raw.data_offsets[1],
                dtype: StDType::U32,
            },
            scales,
            biases,
            layout,
        },
        permute,
        axis: role.heads,
        slot: OnceLock::new(),
    })
}

/// Little-endian values of `N` bytes each, read from `bytes` at any alignment.
fn le_words<T, const N: usize>(bytes: &[u8], read: fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .as_chunks::<N>()
        .0
        .iter()
        .map(|chunk| read(*chunk))
        .collect()
}

/// Finds one of a packed weight's companion arrays.
fn companion<'a>(
    base: &str,
    suffix: &str,
    arrays: &'a HashMap<String, (usize, RawTensor)>,
) -> GgufResult<(usize, &'a RawTensor)> {
    let name = format!("{base}{suffix}");
    arrays
        .get(&name)
        .map(|(shard, raw)| (*shard, raw))
        .ok_or_else(|| GgufError::MissingTensor {
            name: name.into_boxed_str(),
        })
}

/// Turns a companion array into a source, insisting it is `bf16` or `f16`.
///
/// Which one is recorded rather than assumed: the two are the same width, and
/// a reader that took one for the other would decode numbers that look right
/// to three digits and are wrong in the fourth. The host decode reads `bf16`
/// only and refuses the other; the packed path hands the format on.
fn half_source(base: &str, array: (usize, &RawTensor)) -> GgufResult<Source> {
    let (shard, raw) = array;
    let dtype = StDType::parse(&raw.dtype)
        .filter(|&dtype| matches!(dtype, StDType::Bf16 | StDType::F16))
        .ok_or_else(|| GgufError::LayoutMismatch {
            tensor: base.into(),
            want: Box::from("BF16 or F16 scales and biases"),
            got: Box::from(raw.dtype.as_str()),
        })?;
    Ok(Source {
        shard,
        start: raw.data_offsets[0],
        end: raw.data_offsets[1],
        dtype,
    })
}

/// The scale format a [`half_source`] recorded.
fn scale_format(dtype: StDType) -> ScaleFormat {
    if dtype == StDType::F16 {
        ScaleFormat::F16
    } else {
        ScaleFormat::Bf16
    }
}

/// Reads a rank-2 shape, which is the only rank a packed triple can have.
fn two_dims(tensor: &str, dims: &[u64]) -> GgufResult<(u64, u64)> {
    match dims {
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(GgufError::LayoutMismatch {
            tensor: tensor.into(),
            want: Box::from("a two-dimensional array"),
            got: Box::from(format!("{dims:?}")),
        }),
    }
}

/// Bytes an unquantized tensor of `elements` values occupies.
fn payload_bytes(elements: u64, dtype: DType) -> Option<u64> {
    elements.checked_mul(u64::from(dtype.block_bytes().get()))
}

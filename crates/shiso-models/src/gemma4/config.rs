//! The two documents this family is described by, turned into one geometry.
//!
//! Every dimension is checked against the value the kernels were written for. A
//! checkpoint that claimed a different head width would otherwise produce wrong
//! logits with no diagnostic at all - and "wrong logits, no diagnostic" is the
//! failure mode this whole file exists to convert into a named field and two
//! numbers.
//!
//! # Why there is a [`Geometry`] in the middle
//!
//! The family has two sources of truth: the `text_config` subtree of a published
//! `config.json`, and the `gemma4.*` metadata of the container. Either can be
//! the only one present. Writing a reader for each, each with its own checks and
//! its own shape construction, would have produced two stacks that are the same
//! today and diverge on the first edit - and the divergence would be silent,
//! because both would still pass their own assertions. So each reader's whole
//! job is to fill in the same small struct, and the checking and the shape
//! construction happen once, below, on the struct.
//!
//! # Why the shared slots are resolved rather than tabulated
//!
//! Eighteen of the forty-two slots do not write key/value runs; they read an
//! earlier slot's. Which earlier slot is not a number in either document - it
//! follows from the rule that a slot reads the last owning slot of its own kind,
//! which lands sliding slots on 22 and full slots on 23 for E4B. Hard-coding
//! those two would be a table that agrees with the rule by coincidence, so the
//! rule is what is written and the test asserts where it lands.

use std::path::Path;

use serde::Deserialize;
use shiso_core::config::HfConfig;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, TokenId};
use shiso_engine::shape::{
    Activation, AttnLayer, AttnScale, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
    PerLayerShape,
};
use shiso_modelfile::{GgufFile, GgufMetadata, MetaValue};

/// The `model_type` spellings this family answers to.
///
/// Two, because the wrapper document and the text-only document are both real
/// published spellings and neither is a mistake to be corrected in the other's
/// favour.
pub const ARCHES: [&str; 2] = ["gemma4", "gemma4_text"];

/// The GGUF architecture string.
pub const GGUF_ARCH: &str = "gemma4";

/// The `model_type` of the text subtree, when the document has one.
const TEXT_ARCH: &str = "gemma4_text";

/// The only feed-forward activation this family's kernels implement.
const ACTIVATION: &str = "gelu_pytorch_tanh";

/// The logit cap every checkpoint of this family carries.
const SOFTCAP: f32 = 30.0;

/// Rotary base of a windowed slot.
const SLIDING_THETA: f32 = 1.0e4;

/// Rotary base of a full-causal slot.
const GLOBAL_THETA: f32 = 1.0e6;

/// Epsilon every normalization in the stack adds.
const RMS_EPS: f32 = 1.0e-6;

/// Slots between full-causal slots: the last of every six is full-causal, which
/// puts them at 5, 11, 17, 23, 29, 35 and 41 over forty-two slots.
const FULL_ATTENTION_INTERVAL: usize = 6;

/// The `layer_types` spelling of a windowed slot.
const SLIDING: &str = "sliding_attention";

/// The `layer_types` spelling of a full-causal slot.
const FULL: &str = "full_attention";

/// The end-of-turn ids an E4B turn can stop on: `<turn|>`, `<eos>` and
/// `<|tool_response>`.
///
/// `<turn|>` leads, and the order is load-bearing rather than cosmetic:
/// [`shiso_core::StopTokens::primary`] is the one id a caller with room for
/// exactly one records, and what ends an *assistant turn* is `<turn|>`. `<eos>`
/// ends a document, which is not the same event and is not what a server
/// resuming a stored turn should be told the turn ended on. The document lists
/// the three in its own order and the container names only one of them, so
/// neither source can state this; it is stated here, and both readers adopt it.
///
/// The container's own `tokenizer.ggml.eos_token_id` is checked for membership
/// rather than against a position, because which of the three a container
/// happens to name is the container's business and the family's order is not
/// its to set.
const STOP_TOKENS: [u32; 3] = [106, 1, 50];

/// Layer slots an E4B stack holds.
///
/// Named rather than repeated because the per-layer sidecar's row stride is
/// derived from it: [`check`] refuses a checkpoint that disagrees, and
/// `crate::gemma4::ple_repack` builds its geometry from the same number, so the
/// file the repack writes cannot describe a stack this config would refuse.
pub(crate) const GEMMA4_LAYERS: u32 = 42;
/// Rows the per-layer embedding table holds, one per vocabulary entry.
pub(crate) const GEMMA4_PLE_VOCAB: u64 = 262_144;
/// Weights one layer slot of a per-layer row carries.
pub(crate) const GEMMA4_PLE_DIM: u32 = 256;

/// Rotary settings of one attention kind, as the text subtree writes them.
#[derive(Debug, Deserialize)]
struct RopeKind {
    rope_theta: f64,
    /// Absent means the whole head rotates, which is what a config that omits
    /// the key means; the width it implies is then checked like any other.
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
}

/// The two rotary blocks, one per attention kind.
#[derive(Debug, Deserialize)]
struct RopeParameters {
    full_attention: RopeKind,
    sliding_attention: RopeKind,
}

/// The subset of a Gemma 4 text config that determines model geometry.
///
/// Everything else in the file is training bookkeeping a forward pass never
/// reads, so it is not modeled: an unrelated key changing name must not break
/// loading.
#[derive(Debug, Deserialize)]
struct Gemma4TextConfig {
    model_type: String,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    global_head_dim: u32,
    vocab_size: u32,
    vocab_size_per_layer_input: u32,
    hidden_size_per_layer_input: u32,
    num_kv_shared_layers: u32,
    sliding_window: u32,
    rms_norm_eps: f64,
    hidden_activation: String,
    final_logit_softcapping: Option<f64>,
    layer_types: Vec<String>,
    /// The subtree's own stop token, used only when the wrapper above it does
    /// not list one.
    #[serde(default)]
    eos_token_id: Option<u32>,
    rope_parameters: RopeParameters,
}

/// Everything either reader has to produce, before any of it is checked.
///
/// One struct rather than two readers' worth of local variables, because it is
/// what makes "both readers build the same stack" a property of the code rather
/// than of two people keeping two functions in step.
#[derive(Debug)]
struct Geometry {
    /// Layer slots, all of which hold attention in this family.
    n_layers: u32,
    hidden: u32,
    intermediate: u32,
    vocab: u32,
    q_heads: u32,
    kv_heads: u32,
    /// Width of a windowed slot's head.
    head_dim: u32,
    /// Width of a full-causal slot's head, which is wider.
    global_head_dim: u32,
    /// Dimensions a full-causal slot rotates, out of `global_head_dim`.
    global_n_rot: u32,
    /// Positions a windowed slot sees.
    window: u32,
    /// Trailing slots that read an earlier slot's key/value runs.
    kv_shared: u32,
    /// Width of one per-layer embedding row.
    ple_dim: u32,
    /// Rows the per-layer embedding table holds.
    ple_vocab: u32,
    rms_eps: f32,
    /// Rotary base of a windowed slot.
    sliding_theta: f32,
    /// Rotary base of a full-causal slot.
    global_theta: f32,
    /// The logit cap.
    softcap: f32,
    /// One entry per slot: `true` where the slot is windowed.
    sliding: Vec<bool>,
}

/// What this family's checkpoint says, once it has been checked.
///
/// Three things, and they are together because a checkpoint that produced one
/// of them without the others would be half-loaded. The geometry runs the stack;
/// the stop tokens are what a scheduler compares against; the per-layer table's
/// row count is what the streamed reader for that table is opened with, and it
/// is a vocabulary of its own rather than the stack's.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Config {
    shape: HybridShape,
    stop_tokens: Vec<TokenId>,
    ple_vocab: u32,
}

impl Gemma4Config {
    /// Whether a parsed config document names this family.
    ///
    /// Separate from [`Gemma4Config::from_hf`] so a caller can refuse an
    /// unservable checkpoint before it pays for a device or opens a weights
    /// file.
    #[must_use]
    pub fn describes(config: &HfConfig) -> bool {
        ARCHES.contains(&config.model_type())
    }

    /// Recognises the document, then reads the geometry out of its `text_config`
    /// when it has one and out of the container at `checkpoint` when it does not.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the document describes another family -
    /// the one failure a caller reads as "not mine" - [`CoreError::ConfigMissing`]
    /// for an absent key, [`CoreError::ShapeMismatch`] when a dimension
    /// disagrees with the value the kernels were written for,
    /// [`CoreError::InvalidValue`] for an activation or a cap this build cannot
    /// run, and [`CoreError::Json`] when the subtree does not deserialize.
    pub fn from_hf(config: &HfConfig, checkpoint: &Path) -> CoreResult<Self> {
        if !Self::describes(config) {
            return Err(CoreError::UnknownArch {
                model_type: config.model_type().to_owned(),
            });
        }
        let Ok(subtree) = config.section("text_config") else {
            return Self::from_gguf(&open(checkpoint)?);
        };
        let cfg: Gemma4TextConfig = serde_json::from_value(subtree.clone())?;
        if cfg.model_type != TEXT_ARCH {
            return Err(CoreError::UnknownArch {
                model_type: cfg.model_type,
            });
        }
        if cfg.hidden_activation != ACTIVATION {
            return Err(CoreError::InvalidValue {
                key: "hidden_activation",
                value: cfg.hidden_activation,
            });
        }
        let softcap = cfg
            .final_logit_softcapping
            .ok_or_else(|| missing("text_config.final_logit_softcapping"))?;
        let rope = &cfg.rope_parameters;
        // Every one of these is a JSON number, which is an f64, consumed as the
        // f32 the kernels take: the narrowing rounds a scale factor rather than
        // dropping a usable digit.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "config scale factors are read as f64 and consumed as f32"
        )]
        let geometry = Geometry {
            n_layers: cfg.num_hidden_layers,
            hidden: cfg.hidden_size,
            intermediate: cfg.intermediate_size,
            vocab: cfg.vocab_size,
            q_heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            global_head_dim: cfg.global_head_dim,
            global_n_rot: rotated_width(
                cfg.global_head_dim,
                rope.full_attention.partial_rotary_factor.unwrap_or(1.0),
            ),
            window: cfg.sliding_window,
            kv_shared: cfg.num_kv_shared_layers,
            ple_dim: cfg.hidden_size_per_layer_input,
            ple_vocab: cfg.vocab_size_per_layer_input,
            rms_eps: cfg.rms_norm_eps as f32,
            sliding_theta: rope.sliding_attention.rope_theta as f32,
            global_theta: rope.full_attention.rope_theta as f32,
            softcap: softcap as f32,
            sliding: layer_kinds(&cfg.layer_types)?,
        };
        Self::build(&geometry, stop_tokens(config, cfg.eos_token_id)?)
    }

    /// Reads the geometry out of a container's own `gemma4.*` metadata.
    ///
    /// The container is the source rather than the document wherever a
    /// deployment has no sidecar, which is every quantized download. Two numbers
    /// are not stated as metadata and are derived instead: the vocabulary, which
    /// is the length of the tokenizer's token array, and the rotated width of a
    /// full-causal head, which is twice the number of finite entries in
    /// `rope_freqs.weight` - the file stores `1.0` for a rotated pair and `1e30`
    /// for the rest, so counting the finite ones counts the pairs.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the metadata names another architecture,
    /// [`CoreError::ConfigMissing`] for an absent key,
    /// [`CoreError::ShapeMismatch`] naming the first dimension that disagrees,
    /// [`CoreError::InvalidValue`] for a malformed pattern entry, and
    /// [`CoreError::Backend`] when the rotary tensor cannot be read.
    pub fn from_gguf(file: &GgufFile) -> CoreResult<Self> {
        let meta = file.metadata();
        if meta.architecture() != Some(GGUF_ARCH) {
            return Err(CoreError::UnknownArch {
                model_type: meta.architecture().unwrap_or_default().to_owned(),
            });
        }
        let tokens = meta
            .get_array("tokenizer.ggml.tokens")
            .ok_or_else(|| missing("tokenizer.ggml.tokens"))?;
        let vocab = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        let geometry = Geometry {
            n_layers: arch_u32(meta, "block_count")?,
            hidden: arch_u32(meta, "embedding_length")?,
            intermediate: arch_u32(meta, "feed_forward_length")?,
            vocab,
            q_heads: arch_u32(meta, "attention.head_count")?,
            kv_heads: arch_u32(meta, "attention.head_count_kv")?,
            head_dim: arch_u32(meta, "attention.key_length_swa")?,
            global_head_dim: arch_u32(meta, "attention.key_length")?,
            global_n_rot: rotated_width_from_freqs(file)?,
            window: arch_u32(meta, "attention.sliding_window")?,
            kv_shared: arch_u32(meta, "attention.shared_kv_layers")?,
            ple_dim: arch_u32(meta, "embedding_length_per_layer_input")?,
            // The container has no separate row count for the per-layer table.
            // It has never differed from the vocabulary in this family, and the
            // document's own `vocab_size_per_layer_input` is checked against the
            // same constant, so the two readers still have to land on one number.
            ple_vocab: vocab,
            rms_eps: arch_f32(meta, "attention.layer_norm_rms_epsilon")?,
            sliding_theta: arch_f32(meta, "rope.freq_base_swa")?,
            global_theta: arch_f32(meta, "rope.freq_base")?,
            softcap: arch_f32(meta, "final_logit_softcapping")?,
            sliding: sliding_pattern(meta)?,
        };
        let eos = meta
            .get_u32("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| missing("tokenizer.ggml.eos_token_id"))?;
        if !STOP_TOKENS.contains(&eos) {
            return Err(CoreError::InvalidValue {
                key: "tokenizer.ggml.eos_token_id",
                value: eos.to_string(),
            });
        }
        Self::build(&geometry, STOP_TOKENS.map(TokenId::new).to_vec())
    }

    /// The geometry the stack runs.
    #[must_use]
    pub const fn shape(&self) -> &HybridShape {
        &self.shape
    }

    /// The ids that end a turn.
    #[must_use]
    pub fn stop_tokens(&self) -> &[TokenId] {
        &self.stop_tokens
    }

    /// Rows in the per-layer embedding table.
    #[must_use]
    pub const fn ple_vocab(&self) -> u32 {
        self.ple_vocab
    }

    /// Checks a filled-in geometry and turns it into the stack it describes.
    fn build(geometry: &Geometry, stop_tokens: Vec<TokenId>) -> CoreResult<Self> {
        check(geometry)?;
        let shape = HybridShape {
            layers: vec![LayerKind::Attn; geometry.sliding.len()],
            attn: attn_layers(geometry)?,
            hidden: geometry.hidden,
            vocab: geometry.vocab,
            intermediate: geometry.intermediate,
            // No slot of this family holds a recurrent block. The convolution
            // width is 1 rather than 0 because it is a divisor wherever the
            // engine reads it, and a shape nothing consumes is still a shape
            // something could divide by.
            gdn: GdnShape::new(0, 0, 0, 1),
            attn_output_gate: false,
            rms_eps: geometry.rms_eps,
            activation: Activation::GeGluTanh,
            norms: NormScheme::PrePost,
            embed_scale: Some(embed_scale(geometry.hidden)),
            final_softcap: Some(geometry.softcap),
            layer_scalar: true,
            ple: Some(PerLayerShape::new(geometry.ple_dim, geometry.ple_vocab)),
            media_positions: MediaPositions::Linear,
            v_norm: true,
            moe: None,
        };
        shape.validate()?;
        Ok(Self {
            shape,
            stop_tokens,
            ple_vocab: geometry.ple_vocab,
        })
    }
}

/// Checks every dimension against the value the kernels were written for.
///
/// E4B only. The other sizes in the family have the same layout and different
/// numbers, and a stack built from numbers the kernels were not written for
/// produces confident nonsense rather than an error, so an unchecked dimension
/// is refused by name instead of adopted.
fn check(geometry: &Geometry) -> CoreResult<()> {
    let expect = HfConfig::expect;
    expect(
        "num_hidden_layers",
        u64::from(geometry.n_layers),
        u64::from(GEMMA4_LAYERS),
    )?;
    expect("hidden_size", u64::from(geometry.hidden), 2560)?;
    expect("intermediate_size", u64::from(geometry.intermediate), 10240)?;
    expect("num_attention_heads", u64::from(geometry.q_heads), 8)?;
    expect("num_key_value_heads", u64::from(geometry.kv_heads), 2)?;
    expect("head_dim", u64::from(geometry.head_dim), 256)?;
    expect("global_head_dim", u64::from(geometry.global_head_dim), 512)?;
    expect("vocab_size", u64::from(geometry.vocab), 262_144)?;
    expect(
        "vocab_size_per_layer_input",
        u64::from(geometry.ple_vocab),
        GEMMA4_PLE_VOCAB,
    )?;
    expect(
        "hidden_size_per_layer_input",
        u64::from(geometry.ple_dim),
        u64::from(GEMMA4_PLE_DIM),
    )?;
    expect("num_kv_shared_layers", u64::from(geometry.kv_shared), 18)?;
    expect("sliding_window", u64::from(geometry.window), 512)?;
    expect(
        "rotated_global_head_dim",
        u64::from(geometry.global_n_rot),
        128,
    )?;
    // The per-slot list is the only statement of which slots are windowed, and
    // the slot count is stated separately: a list that is short would leave the
    // trailing slots unbuilt with every individual dimension still correct.
    expect(
        "layer_types",
        u64::try_from(geometry.sliding.len()).unwrap_or(u64::MAX),
        u64::from(geometry.n_layers),
    )?;
    // The list's LENGTH being right says nothing about which slots it windows,
    // and a schedule that windowed the wrong ones would build a stack whose
    // every dimension is correct and whose attention reaches the wrong
    // positions. The pattern is the checked fact, not the count of entries.
    let off_pattern = geometry
        .sliding
        .iter()
        .enumerate()
        .filter(|(slot, sliding)| **sliding == (slot + 1).is_multiple_of(FULL_ATTENTION_INTERVAL))
        .count();
    expect(
        "layer_types.schedule_mismatch_count",
        u64::try_from(off_pattern).unwrap_or(u64::MAX),
        0,
    )?;
    // The four scale factors. Nothing above reaches them, so a checkpoint with
    // every integer of E4B and another rotary base would have built silently.
    expect_f32("final_logit_softcapping", geometry.softcap, SOFTCAP)?;
    expect_f32(
        "rope_parameters.sliding_attention.rope_theta",
        geometry.sliding_theta,
        SLIDING_THETA,
    )?;
    expect_f32(
        "rope_parameters.full_attention.rope_theta",
        geometry.global_theta,
        GLOBAL_THETA,
    )?;
    expect_f32("rms_norm_eps", geometry.rms_eps, RMS_EPS)
}

/// Checks one scale factor against the value the kernels were written for.
///
/// The tolerance is relative rather than absolute because these four span ten
/// orders of magnitude, from an epsilon of `1e-6` to a rotary base of `1e6`, and
/// one absolute allowance cannot be both a rounding margin at a million and
/// anything other than exact equality at a millionth. A millionth of the
/// expected value is roughly eight `f32` steps: wide enough for a `f64` config
/// constant narrowed to `f32`, or the same number written by two different
/// tools, and far too narrow for a checkpoint that means a different number.
///
/// [`CoreError::ShapeMismatch`] carries `u64`s, so these report as
/// [`CoreError::InvalidValue`] with both numbers in the message.
fn expect_f32(field: &'static str, got: f32, want: f32) -> CoreResult<()> {
    if (got - want).abs() <= want.abs() * 1e-6 {
        Ok(())
    } else {
        Err(CoreError::InvalidValue {
            key: field,
            value: format!("{got}, expected {want}"),
        })
    }
}

/// One attention descriptor per slot, with the shared slots pointed at owners.
fn attn_layers(geometry: &Geometry) -> CoreResult<Vec<AttnLayer>> {
    let owning =
        usize::try_from(geometry.n_layers.saturating_sub(geometry.kv_shared)).unwrap_or(usize::MAX);
    geometry
        .sliding
        .iter()
        .enumerate()
        .map(|(slot, sliding)| {
            let layer = if *sliding {
                AttnLayer::new(
                    geometry.q_heads,
                    geometry.kv_heads,
                    geometry.head_dim,
                    geometry.head_dim,
                    geometry.head_dim,
                    geometry.sliding_theta,
                )
                .with_window(geometry.window)
            } else {
                AttnLayer::new(
                    geometry.q_heads,
                    geometry.kv_heads,
                    geometry.global_head_dim,
                    geometry.global_n_rot,
                    geometry.global_head_dim,
                    geometry.global_theta,
                )
            }
            .with_scale(AttnScale::One);
            if slot < owning {
                return Ok(layer);
            }
            Ok(layer.shared_from(owner_of(&geometry.sliding, *sliding, owning)?))
        })
        .collect()
}

/// The slot a sharer reads: the last owning slot of its own kind.
///
/// Both kinds are present among the owners of every real checkpoint, and the
/// sharer's head width is its kind's, so reading the wrong kind's runs would be
/// an out-of-stride read inside a buffer whose length is right. There is no
/// fallback for a schedule where a kind has no owner at all - that is a
/// checkpoint this build cannot run, and it is refused here rather than by an
/// arithmetic error later.
fn owner_of(sliding: &[bool], kind: bool, owning: usize) -> CoreResult<LayerIdx> {
    sliding
        .iter()
        .take(owning)
        .rposition(|slot| *slot == kind)
        .and_then(|slot| u32::try_from(slot).ok())
        .map(LayerIdx::new)
        .ok_or_else(|| CoreError::InvalidValue {
            key: "num_kv_shared_layers",
            value: format!(
                "no {} slot below {owning} owns key/value runs for the sharers above it",
                if kind { SLIDING } else { FULL }
            ),
        })
}

/// The document's per-slot list as one flag per slot: `true` where it is
/// windowed.
///
/// An unrecognised spelling names its own slot, because that is the only thing
/// a caller can act on.
fn layer_kinds(types: &[String]) -> CoreResult<Vec<bool>> {
    types
        .iter()
        .enumerate()
        .map(|(index, kind)| match kind.as_str() {
            SLIDING => Ok(true),
            FULL => Ok(false),
            _ => Err(CoreError::InvalidValue {
                key: "text_config.layer_types",
                value: format!("entry {index} is {kind:?}"),
            }),
        })
        .collect()
}

/// The container's per-slot list, which is the same statement as a bool array.
fn sliding_pattern(meta: &GgufMetadata) -> CoreResult<Vec<bool>> {
    let key = "gemma4.attention.sliding_window_pattern";
    let array = meta.get_array(key).ok_or_else(|| missing(key))?;
    array
        .iter()
        .enumerate()
        .map(|(index, entry)| match entry {
            Ok(MetaValue::Bool(sliding)) => Ok(sliding),
            _ => Err(CoreError::InvalidValue {
                key: "gemma4.attention.sliding_window_pattern",
                value: format!("entry {index} is not a boolean"),
            }),
        })
        .collect()
}

/// Twice the number of finite entries in the rotary frequency tensor.
///
/// The tensor holds one entry per frequency of the widest head, `1.0` where the
/// pair is rotated and `1e30` where it is not, so the finite entries are the
/// rotated pairs and twice their count is the rotated width.
fn rotated_width_from_freqs(file: &GgufFile) -> CoreResult<u32> {
    let name = "rope_freqs.weight";
    let read = |source| CoreError::Backend {
        backend: GGUF_ARCH,
        op: "read the rotary frequency tensor",
        source: Box::new(source),
    };
    let view = file.require(name).map_err(read)?;
    let freqs = view.as_f32().map_err(read)?;
    // The threshold separates a real inverse frequency, which is at most 1, from
    // the sentinel the file stores for an unrotated pair, which is 1e30.
    let rotated = freqs.iter().filter(|freq| **freq < 1e10).count();
    Ok(u32::try_from(rotated).unwrap_or(u32::MAX).saturating_mul(2))
}

/// Width of the rotated part of a head.
///
/// The fraction applies to the head, and the width is a count of PAIRS doubled
/// rather than the product itself, because the rotation walks pairs and a factor
/// that implied an odd number of dimensions would leave one of them unpaired.
/// It is rounded rather than truncated so a value that lands at 127.99999
/// through float representation cannot quietly rotate one pair fewer than the
/// checkpoint.
fn rotated_width(head_dim: u32, factor: f64) -> u32 {
    // The product is a count of head dimensions: non-negative, and small for any
    // head a real checkpoint has, so the narrowing cast cannot truncate.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a rounded pair count of a head is a small integer"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "the product of two non-negative config values"
    )]
    let pairs = (f64::from(head_dim) * factor / 2.0).round() as u32;
    pairs.saturating_mul(2)
}

/// The multiplier on the gathered token rows, which is `sqrt(hidden)`.
#[expect(
    clippy::cast_precision_loss,
    reason = "a residual width is a small integer, and the scale it implies is a float"
)]
fn embed_scale(hidden: u32) -> f32 {
    (hidden as f32).sqrt()
}

/// The ids that end a turn, from the top of the document.
///
/// They live above `text_config` rather than in it, as a list or as a single id,
/// and a checkpoint whose wrapper omits them falls back to the subtree's own.
///
/// The document states which ids end a turn; it does not state which of them a
/// caller with room for one should record. So what comes back is the document's
/// membership in [`STOP_TOKENS`] order - the family's order, so that the two
/// readers agree on a leading id and not merely on a set - with anything the
/// document names that the family does not declare kept behind it, in the order
/// the document wrote it.
fn stop_tokens(config: &HfConfig, fallback: Option<u32>) -> CoreResult<Vec<TokenId>> {
    let key = "eos_token_id";
    let Ok(value) = config.section(key) else {
        return fallback
            .map(|id| vec![TokenId::new(id)])
            .ok_or_else(|| missing(key));
    };
    let malformed = || CoreError::InvalidValue {
        key: "eos_token_id",
        value: value.to_string(),
    };
    let ids = match value.as_array() {
        Some(entries) => entries
            .iter()
            .map(serde_json::Value::as_u64)
            .collect::<Option<Vec<u64>>>(),
        None => value.as_u64().map(|id| vec![id]),
    }
    .ok_or_else(malformed)?;
    let ids = ids
        .into_iter()
        .map(|id| u32::try_from(id).map_err(|_| malformed()))
        .collect::<CoreResult<Vec<u32>>>()?;
    let mut ordered: Vec<TokenId> = STOP_TOKENS
        .iter()
        .filter(|declared| ids.contains(declared))
        .map(|declared| TokenId::new(*declared))
        .collect();
    ordered.extend(
        ids.iter()
            .filter(|id| !STOP_TOKENS.contains(id))
            .map(|id| TokenId::new(*id)),
    );
    Ok(ordered)
}

/// Opens the container a document without a text subtree defers to.
fn open(checkpoint: &Path) -> CoreResult<GgufFile> {
    GgufFile::open_shared(checkpoint).map_err(|source| CoreError::Backend {
        backend: GGUF_ARCH,
        op: "read the checkpoint's metadata",
        source: Box::new(source),
    })
}

/// One architecture-prefixed count, or the key that is absent.
fn arch_u32(meta: &GgufMetadata, suffix: &str) -> CoreResult<u32> {
    meta.arch_u32(suffix)
        .ok_or_else(|| missing(&format!("{GGUF_ARCH}.{suffix}")))
}

/// One architecture-prefixed scale factor, or the key that is absent.
fn arch_f32(meta: &GgufMetadata, suffix: &str) -> CoreResult<f32> {
    meta.arch_f32(suffix)
        .ok_or_else(|| missing(&format!("{GGUF_ARCH}.{suffix}")))
}

/// The absent-key error, named.
fn missing(key: &str) -> CoreError {
    CoreError::ConfigMissing {
        key: key.to_owned(),
    }
}

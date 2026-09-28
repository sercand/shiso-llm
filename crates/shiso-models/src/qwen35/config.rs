//! The config document this family is described by, turned into geometry.
//!
//! Every dimension is checked against the value the kernels were written for. A
//! checkpoint that claims a different head width would otherwise produce wrong
//! logits with no diagnostic at all - and "wrong logits, no diagnostic" is the
//! failure mode this whole file exists to convert into a named field and two
//! numbers.
//!
//! # Why the asserts are here and not in the shared crate
//!
//! They were in the shared crate, which meant the bottom of the stack carried a
//! table of one checkpoint family's dimensions and a match arm on its name. The
//! document reader stays down there because every config spells a model type and
//! an architecture list the same way; what a *named* architecture's subtree
//! means is this crate's business, and it reaches that subtree through
//! [`HfConfig::section`].

use serde::Deserialize;
use shiso_core::config::HfConfig;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_engine::shape::{
    Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
};

/// The `model_type` of the text subtree this crate can read.
const TEXT_ARCH: &str = "qwen3_5_text";

/// The `model_type` values whose document carries that subtree.
const ARCHES: [&str; 2] = ["qwen3_5", "qwen3_5_text"];

/// `<|im_end|>`, the id every assistant turn of this family ends on.
///
/// It is a constant here for the same reason the vocabulary size is checked
/// against a constant below: this family ships one vocabulary, and [`check`]
/// refuses a checkpoint that claims a different one. The config document does
/// not carry this id - `text_config.eos_token_id` is `<|endoftext|>`, which is
/// where a *base* model stops - so a reader that trusted the document alone
/// would decode the end of every chat turn as text and keep generating. The
/// GGUF export of the same checkpoint declares this id and not that one, under
/// `tokenizer.ggml.eos_token_id`.
const IM_END: u32 = 248_046;

/// Rotary settings, as the text subtree writes them.
#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f64,
}

/// The frequency base a config means when it omits one.
fn default_rope_theta() -> f64 {
    1.0e7
}

/// The rotated fraction a config means when it omits one: the whole head.
fn default_partial_rotary() -> f64 {
    1.0
}

/// The subset of a Qwen3.5 text config that determines model geometry.
///
/// Everything else in the file is attention bookkeeping and training leftovers
/// a forward pass never reads, so it is not modeled: an unrelated key changing
/// name must not break loading.
#[derive(Debug, Deserialize)]
struct Qwen35TextConfig {
    model_type: String,
    num_hidden_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    attn_output_gate: bool,
    full_attention_interval: u32,
    rms_norm_eps: f64,
    eos_token_id: u32,
    linear_num_value_heads: u32,
    linear_num_key_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    layer_types: Vec<String>,
    rope_parameters: RopeParameters,
}

/// What this family's config says, once it has been checked.
///
/// Three things, and they are together because a checkpoint that produced one
/// of them without the others would be half-loaded. The geometry runs the stack;
/// the end-of-turn token is what a scheduler compares against; the attention
/// stride is what the tensor binding cross-checks the file's own metadata with.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Qwen35Config {
    shape: HybridShape,
    eos_token: TokenId,
    full_attention_interval: u32,
}

impl Qwen35Config {
    /// The three facts, given directly.
    ///
    /// The reading path is [`Qwen35Config::from_hf`]; this exists for a caller
    /// that already has the geometry - a container that carries its dimensions as
    /// metadata rather than as a config document, or a test that needs a stack
    /// small enough to fit in a fixture.
    #[must_use]
    pub const fn new(shape: HybridShape, eos_token: TokenId, full_attention_interval: u32) -> Self {
        Self {
            shape,
            eos_token,
            full_attention_interval,
        }
    }

    /// Reads the geometry out of a parsed config document.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the document does not describe this
    /// family, [`CoreError::ConfigMissing`] when the text subtree or a key in it
    /// is absent, [`CoreError::ShapeMismatch`] when a dimension disagrees with
    /// the value the kernels were written for, and [`CoreError::Json`] when the
    /// subtree does not deserialize.
    pub fn from_hf(config: &HfConfig) -> CoreResult<Self> {
        if !ARCHES.contains(&config.model_type()) {
            return Err(CoreError::UnknownArch {
                model_type: config.model_type().to_string(),
            });
        }
        let cfg: Qwen35TextConfig = serde_json::from_value(config.section("text_config")?.clone())?;
        if cfg.model_type != TEXT_ARCH {
            return Err(CoreError::UnknownArch {
                model_type: cfg.model_type,
            });
        }
        check(&cfg)?;
        let n_rot = rotated_width(cfg.head_dim, cfg.rope_parameters.partial_rotary_factor);
        let layers = layer_kinds(&cfg.layer_types, cfg.full_attention_interval)?;
        let attn_slots = layers
            .iter()
            .filter(|kind| **kind == LayerKind::Attn)
            .count();
        // Both arrive as f64 because that is what a JSON number is, and are used
        // as f32 because that is what the kernels consume: the narrowing rounds a
        // scale factor rather than dropping a usable digit.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "config scale factors are read as f64 and consumed as f32"
        )]
        let (rms_eps, rope_theta) = (
            cfg.rms_norm_eps as f32,
            cfg.rope_parameters.rope_theta as f32,
        );
        Ok(Self {
            shape: HybridShape {
                layers,
                hidden: cfg.hidden_size,
                vocab: cfg.vocab_size,
                intermediate: cfg.intermediate_size,
                // Every attention slot of this family has the same widths, the
                // same rotary base, no window and its own cache, so the list is
                // one descriptor repeated: the per-slot shape is what a family
                // whose slots differ needs, and stating it per slot here costs a
                // `vec!` and removes the question of which slot a width belongs
                // to. The ladder is the rotated width itself, which is what the
                // kernels have always derived their frequencies over.
                attn: vec![
                    AttnLayer::new(
                        cfg.num_attention_heads,
                        cfg.num_key_value_heads,
                        cfg.head_dim,
                        n_rot,
                        n_rot,
                        rope_theta,
                    );
                    attn_slots
                ],
                gdn: GdnShape::new(
                    cfg.linear_num_value_heads,
                    cfg.linear_num_key_heads,
                    cfg.linear_key_head_dim,
                    cfg.linear_conv_kernel_dim,
                ),
                attn_output_gate: cfg.attn_output_gate,
                rms_eps,
                // The family switches, all of them the value that describes what
                // this family has always done: one pre-norm pair per slot, a
                // SwiGLU feed-forward, unscaled embeddings, uncapped logits, no
                // layer scalar, no per-layer embedding table, three-stream media
                // positions and an unnormalized value projection.
                activation: Activation::SwiGlu,
                norms: NormScheme::PreOnly,
                embed_scale: None,
                final_softcap: None,
                layer_scalar: false,
                ple: None,
                media_positions: MediaPositions::Mrope3D,
                v_norm: false,
                moe: None,
                ssm: None,
                no_ffn: Vec::new(),
            },
            eos_token: TokenId::new(cfg.eos_token_id),
            full_attention_interval: cfg.full_attention_interval,
        })
    }

    /// Reads the geometry out of a config file.
    ///
    /// # Errors
    ///
    /// Whatever [`HfConfig::from_file`] and [`Qwen35Config::from_hf`] report.
    pub fn from_file(path: &std::path::Path) -> CoreResult<Self> {
        Self::from_hf(&HfConfig::from_file(path)?)
    }

    /// The geometry the stack runs.
    #[must_use]
    pub const fn shape(&self) -> &HybridShape {
        &self.shape
    }

    /// The end-of-sequence token the config document names.
    ///
    /// This is `<|endoftext|>`, not the end of a chat turn. Use
    /// [`Qwen35Config::stop_tokens`] to decide when a turn is over.
    #[must_use]
    pub const fn eos_token(&self) -> TokenId {
        self.eos_token
    }

    /// The ids that end a turn, most specific first.
    ///
    /// Two of them, and the order is what a caller with room for one should
    /// record: `<|im_end|>` is how a chat turn actually ends, and the
    /// document's own `eos_token_id` rides behind it because a checkpoint that
    /// emits `<|endoftext|>` has also stopped. A document that already names
    /// `<|im_end|>` gets it once.
    #[must_use]
    pub fn stop_tokens(&self) -> Vec<TokenId> {
        let im_end = TokenId::new(IM_END);
        if self.eos_token == im_end {
            vec![im_end]
        } else {
            vec![im_end, self.eos_token]
        }
    }

    /// Slots between attention blocks, which the tensor binding cross-checks the
    /// checkpoint's own metadata against.
    #[must_use]
    pub const fn full_attention_interval(&self) -> u32 {
        self.full_attention_interval
    }
}

/// Checks the dimensions the kernels fix, and the relations the rest must keep.
///
/// Two kinds of number live in a config. Some are compiled into the kernels:
/// the attention head width, the GDN head widths, the conv window, the rotated
/// width and the output gate, and the vocabulary that [`IM_END`] belongs to. A
/// checkpoint that differs in one of those would run and be wrong, so each is
/// held to its one value. The rest - depth, widths, head counts - are launch
/// arguments, and hold for any checkpoint of this family whose counts divide
/// the way grouped attention and grouped value heads need.
fn check(cfg: &Qwen35TextConfig) -> CoreResult<()> {
    let expect = HfConfig::expect;
    expect("vocab_size", u64::from(cfg.vocab_size), 248_320)?;
    expect("head_dim", u64::from(cfg.head_dim), 256)?;
    expect(
        "linear_key_head_dim",
        u64::from(cfg.linear_key_head_dim),
        128,
    )?;
    expect(
        "linear_value_head_dim",
        u64::from(cfg.linear_value_head_dim),
        128,
    )?;
    expect(
        "linear_conv_kernel_dim",
        u64::from(cfg.linear_conv_kernel_dim),
        4,
    )?;
    expect(
        "rotated_head_dim",
        u64::from(rotated_width(
            cfg.head_dim,
            cfg.rope_parameters.partial_rotary_factor,
        )),
        64,
    )?;
    expect_bool("attn_output_gate", cfg.attn_output_gate, true)?;
    // The launch arguments: nonzero, and widths the affine matmul tiles take
    // (inputs in whole groups of 64, outputs in whole tiles of 8).
    for (field, value) in [
        ("num_hidden_layers", cfg.num_hidden_layers),
        ("full_attention_interval", cfg.full_attention_interval),
        ("num_attention_heads", cfg.num_attention_heads),
        ("num_key_value_heads", cfg.num_key_value_heads),
        ("linear_num_key_heads", cfg.linear_num_key_heads),
        ("linear_num_value_heads", cfg.linear_num_value_heads),
    ] {
        if value == 0 {
            return Err(CoreError::ShapeMismatch {
                field,
                expected: 1,
                got: 0,
            });
        }
    }
    for (field, value) in [
        ("hidden_size", cfg.hidden_size),
        ("intermediate_size", cfg.intermediate_size),
    ] {
        if value == 0 || !value.is_multiple_of(64) {
            return Err(CoreError::ShapeMismatch {
                field,
                expected: u64::from(value.div_ceil(64).max(1) * 64),
                got: u64::from(value),
            });
        }
    }
    // The depth is stated twice - as a count and as the per-layer list - and a
    // checkpoint whose two statements disagree is one of them edited.
    let listed = u32::try_from(cfg.layer_types.len()).unwrap_or(u32::MAX);
    if cfg.num_hidden_layers != listed {
        return Err(CoreError::ShapeMismatch {
            field: "num_hidden_layers",
            expected: u64::from(listed),
            got: u64::from(cfg.num_hidden_layers),
        });
    }
    divides(
        "num_attention_heads",
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
    )?;
    divides(
        "linear_num_value_heads",
        cfg.linear_num_value_heads,
        cfg.linear_num_key_heads,
    )
}

/// `whole` must be a multiple of `part`: query heads share key/value heads, and
/// value heads share key heads, in whole groups.
fn divides(field: &'static str, whole: u32, part: u32) -> CoreResult<()> {
    if whole.is_multiple_of(part) {
        Ok(())
    } else {
        Err(CoreError::ShapeMismatch {
            field,
            expected: u64::from(whole.div_ceil(part) * part),
            got: u64::from(whole),
        })
    }
}

/// Width of the rotated part of a head.
///
/// The fraction applies to the head, not the residual stream, and it is rounded
/// rather than truncated so a value that lands at 63.99999 through float
/// representation cannot quietly rotate one dimension fewer than the checkpoint.
fn rotated_width(head_dim: u32, factor: f64) -> u32 {
    // The product is a count of head dimensions: non-negative, and small for
    // any head a real checkpoint has, so the narrowing cast cannot truncate.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a rounded dimension count of a head is a small integer"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "the product of two non-negative config values"
    )]
    let width = (f64::from(head_dim) * factor).round() as u32;
    width
}

/// Checks a boolean dimension with the same error shape as the numeric checks,
/// so one config audit reports every deviation the same way.
fn expect_bool(field: &'static str, got: bool, want: bool) -> CoreResult<()> {
    if got == want {
        Ok(())
    } else {
        Err(CoreError::ShapeMismatch {
            field,
            expected: u64::from(want),
            got: u64::from(got),
        })
    }
}

/// Turns the config's per-layer vocabulary into the layer list, cross-checked
/// against the attention interval.
///
/// The config states the same sequence twice, once as a per-layer list and once
/// as an interval, and a hand-edited file where the two disagree would put
/// attention blocks in the wrong slots while every individual dimension still
/// looked correct. An unrecognised spelling names its own slot, because that is
/// the only thing a caller can act on; a disagreement with the interval is
/// reported as a count of disagreeing slots, since the field name carried by
/// [`CoreError::ShapeMismatch`] is static and cannot hold an index.
fn layer_kinds(types: &[String], interval: u32) -> CoreResult<Vec<LayerKind>> {
    let kinds = types
        .iter()
        .enumerate()
        .map(|(index, kind)| match kind.as_str() {
            "linear_attention" => Ok(LayerKind::Gdn),
            "full_attention" => Ok(LayerKind::Attn),
            _ => Err(CoreError::ConfigMissing {
                key: format!("text_config.layer_types[{index}]"),
            }),
        })
        .collect::<CoreResult<Vec<LayerKind>>>()?;
    let mismatched = kinds
        .iter()
        .enumerate()
        .filter(|(index, claimed)| {
            let slot = u64::try_from(*index).unwrap_or(u64::MAX);
            let rule = if (slot + 1).is_multiple_of(u64::from(interval)) {
                LayerKind::Attn
            } else {
                LayerKind::Gdn
            };
            **claimed != rule
        })
        .count();
    HfConfig::expect(
        "layer_types.interval_mismatch_count",
        u64::try_from(mismatched).unwrap_or(u64::MAX),
        0,
    )?;
    Ok(kinds)
}

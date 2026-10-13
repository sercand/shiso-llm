//! The two config documents this family is described by, turned into geometry.
//!
//! Two rather than one, and they are read together because a checkpoint that
//! produced one without the other would be half-described: `config.json` says
//! what the stack IS, and `hf_quant_config.json` says what its weights are
//! stored AS. Neither is derivable from the other, every tensor the loader
//! binds needs both, and a reader that took only the first would decode
//! `W4A16_NVFP4` nibbles as bf16 and get finite, fluent, wrong numbers.
//!
//! # Why the asserts are here and not in the shared crate
//!
//! The same reason [`crate::qwen35::Qwen35Config`] gives: every config spells a
//! model type and an architecture list the same way, so the document reader
//! stays in [`shiso_core::config`]; what a *named* architecture's subtree means
//! is this crate's business.
//!
//! # What is checked, and what that buys
//!
//! Every dimension below is compared against the number the kernels were
//! written for, including three that are not widths at all - the rotary
//! sections, whether they interleave, and the attention interval. The rotary
//! pair is checked because `kernels/cuda/qwen35/q35_common.cuh` deals its
//! frequency pairs out by `i % 3`, which IS the triple `[11, 11, 10]`
//! interleaved and is not a parameter. A checkpoint that declared any other
//! triple would rotate every head against the wrong axis and produce a fluent
//! answer to a different prompt.
//!
//! # A Splash package states the same geometry twice
//!
//! The pre-quantized `*-Splash` container carries the shape in two documents
//! that were written by two different programs:
//!
//! - `manifest.json` - [`super::splash::Manifest`] - describes **this package
//!   as converted**: the widths the packer actually wrote, the padding it
//!   padded to, the roll of 40 layer kinds.
//! - `tokenizer/config.json` is, despite its path, the whole upstream Hugging
//!   Face model config of the checkpoint the package was converted from. There
//!   is no `config.json` at the package root.
//!
//! The manifest is authoritative for anything it states, because it describes
//! the bytes on disk. The upstream config is authoritative for everything the
//! manifest does not carry - the rotary parameters, the norm epsilon, the
//! attention head counts, the recurrent head widths, the end-of-sequence id.
//! Where the two overlap they must agree, and a disagreement is refused by name
//! with both numbers: it is the only cheap thing that catches a package whose
//! converter and whose upstream have drifted apart. [`Qwen35MoeConfig::from_splash`]
//! is that reading, and the refusals are [`agree`].

use std::path::Path;

use serde::Deserialize;
use shiso_core::config::HfConfig;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_engine::shape::{
    Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, MoeShape, NormScheme,
};

use super::quant::QuantConfig;
use super::splash::{self, Manifest};

/// The `model_type` of the text subtree this module can read.
const TEXT_ARCH: &str = "qwen3_5_moe_text";

/// The `model_type` values whose document carries that subtree.
const ARCHES: [&str; 2] = ["qwen3_5_moe", "qwen3_5_moe_text"];

/// `<|im_end|>`, the id every assistant turn of this family ends on.
///
/// The same id the dense family stops on, and carried here as its own constant
/// rather than imported from there: the two families ship the same vocabulary
/// today, and a constant shared across module boundaries would make that a
/// requirement rather than an observation. [`check`] refuses a checkpoint whose
/// vocabulary is a different size, which is what keeps the two in step.
///
/// The document does not carry this id - `text_config.eos_token_id` is
/// `<|endoftext|>`, which is where a *base* model stops - so a reader that
/// trusted the document alone would decode the end of every chat turn as text
/// and keep generating.
const IM_END: u32 = 248_046;

/// The rotary sections the recurrent family's kernels deal pairs out by.
///
/// Not a parameter anywhere below the host: `q35_common.cuh` picks a pair's
/// axis with `i % 3`, so this triple is what that arithmetic means at 32 pairs
/// and the only triple those entry points implement.
const MROPE_SECTION: [u32; 3] = [11, 11, 10];

/// Rotary settings, as the text subtree writes them.
#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f64,
    #[serde(default)]
    mrope_section: Vec<u32>,
    #[serde(default)]
    mrope_interleaved: bool,
}

/// The frequency base a config means when it omits one.
fn default_rope_theta() -> f64 {
    1.0e7
}

/// The rotated fraction a config means when it omits one: the whole head.
fn default_partial_rotary() -> f64 {
    1.0
}

/// The subset of a Qwen3.5 mixture-of-experts text config that determines model geometry.
///
/// Everything else in the file is attention bookkeeping, routing-loss
/// coefficients and the draft head's own depth, none of which a forward pass in
/// this build reads. An unrelated key changing name must not break loading.
#[derive(Debug, Deserialize)]
struct Qwen35MoeTextConfig {
    model_type: String,
    num_hidden_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    attn_output_gate: bool,
    full_attention_interval: u32,
    rms_norm_eps: f64,
    eos_token_id: u32,
    num_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    shared_expert_intermediate_size: u32,
    linear_num_value_heads: u32,
    linear_num_key_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    layer_types: Vec<String>,
    rope_parameters: RopeParameters,
}

/// What this family's two config documents say, once they have been checked.
///
/// Four things, and they are together for the reason the fields of
/// [`crate::qwen35::Qwen35Config`] are: a checkpoint that produced some of them
/// without the rest would be half-loaded. The geometry runs the stack; the
/// end-of-turn token is what a scheduler compares against; the attention stride
/// is what the tensor binding cross-checks the file's own metadata with; and
/// the quantization is what every tensor's decode is chosen by.
///
/// # Why the quantization document is an `Option`
///
/// Because one container of this family does not have one. A modelopt export
/// states its storage per *target* in `hf_quant_config.json`; a Splash package
/// states it per *section* in `layout.json` and carries no such document at
/// all. An empty [`QuantConfig`] would answer [`QuantConfig::algo_for`] with
/// [`super::QuantAlgo::None`] for every tensor in the package - "stored at its
/// declared element type" - which is a false answer rather than a missing one,
/// and the only consumer, [`super::check_export`], would then pass an export
/// whose weights it never checked. `None` makes that a refusal by name.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Qwen35MoeConfig {
    shape: HybridShape,
    eos_token: TokenId,
    full_attention_interval: u32,
    quant: Option<QuantConfig>,
}

impl Qwen35MoeConfig {
    /// The four facts, given directly.
    ///
    /// The reading path is [`Qwen35MoeConfig::from_files`]; this exists for a
    /// caller that already has the geometry - a test that needs a stack small
    /// enough to fit in a fixture, or a container that carries its dimensions
    /// as metadata rather than as a document.
    #[must_use]
    pub const fn new(
        shape: HybridShape,
        eos_token: TokenId,
        full_attention_interval: u32,
        quant: Option<QuantConfig>,
    ) -> Self {
        Self {
            shape,
            eos_token,
            full_attention_interval,
            quant,
        }
    }

    /// Whether a parsed config document names this family.
    ///
    /// Separate from [`Qwen35MoeConfig::from_hf`] for the reason
    /// [`crate::gemma4::Gemma4Config::describes`] is separate from its own
    /// reader: a caller can refuse an unservable checkpoint before it pays for
    /// a device or opens a weights file. It matters more here than there,
    /// because this family's reader needs a second document - the quantization
    /// one - that lives beside the weights, so "is this mine" cannot be asked
    /// by calling the reader and catching its refusal without first reading a
    /// file that a document for another family may not have beside it.
    ///
    /// The dense hybrid's `model_type` values are a disjoint set from these
    /// two, so no ordering between the two readers can make one answer for the
    /// other's document.
    #[must_use]
    pub fn describes(config: &HfConfig) -> bool {
        ARCHES.contains(&config.model_type())
    }

    /// Reads the geometry out of a parsed config document and a parsed
    /// quantization document.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the document does not describe this
    /// family, [`CoreError::ConfigMissing`] when the text subtree or a key in it
    /// is absent, [`CoreError::ShapeMismatch`] when a dimension disagrees with
    /// the value the kernels were written for, and [`CoreError::Json`] when the
    /// subtree does not deserialize.
    pub fn from_hf(config: &HfConfig, quant: QuantConfig) -> CoreResult<Self> {
        if !Self::describes(config) {
            return Err(CoreError::UnknownArch {
                model_type: config.model_type().to_string(),
            });
        }
        Self::from_text(config.section("text_config")?, Some(quant))
    }

    /// Reads the geometry out of a config document that has no quantization
    /// document beside it.
    ///
    /// The same widths and the same refusals as [`Qwen35MoeConfig::from_hf`];
    /// what it does not have is an answer to "how is this tensor stored",
    /// because the containers that take this path - an MLX checkpoint, a Splash
    /// package - state that per tensor rather than in a second document.
    ///
    /// `model_type` at the top of the document is checked when it is there and
    /// its absence is not a refusal: a document handed over as a bare
    /// `text_config` has no top level to state it, and the subtree's own
    /// `model_type` is checked either way.
    ///
    /// # Errors
    ///
    /// Whatever [`Qwen35MoeConfig::from_hf`] reports.
    pub fn from_json(document: &serde_json::Value) -> CoreResult<Self> {
        if let Some(model_type) = document
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            && !ARCHES.contains(&model_type)
        {
            return Err(CoreError::UnknownArch {
                model_type: model_type.to_string(),
            });
        }
        Self::from_text(document.get("text_config").unwrap_or(document), None)
    }

    /// Reads a Splash package's geometry: the manifest's account of it, checked
    /// against the upstream checkpoint's own config.
    ///
    /// `hf` is the document at `tokenizer/config.json`, which despite its path
    /// is the whole upstream Hugging Face model config; see the module docs.
    /// Every width the two documents both state is compared, and the geometry
    /// this returns is therefore both of their accounts at once.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the manifest names an architecture this
    /// module does not read, [`CoreError::ShapeMismatch`] naming the manifest's
    /// field and carrying both numbers when a width disagrees,
    /// [`CoreError::InvalidValue`] naming the slot when the two layer rolls
    /// disagree, and whatever [`Qwen35MoeConfig::from_json`] reports.
    pub fn from_splash(manifest: &Manifest, hf: &serde_json::Value) -> CoreResult<Self> {
        let config = Self::from_json(hf)?;
        agree(manifest, &config)?;
        Ok(config)
    }

    /// Reads a Splash package's two documents off disk.
    ///
    /// The paths are written here rather than left to each caller because one
    /// of them is a surprise: the model config lives at `tokenizer/config.json`
    /// and the package root has no `config.json`, so a caller guessing the
    /// obvious path gets a missing file rather than a wrong answer only because
    /// nothing is there to read.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming the path when either document is
    /// absent, [`CoreError::Json`] when either does not parse - which for a
    /// manifest written to another schema names `schema_version` and both
    /// numbers - and whatever [`Qwen35MoeConfig::from_splash`] reports.
    pub fn from_splash_dir(root: &Path) -> CoreResult<Self> {
        let manifest = Manifest::from_json(&read_document(&root.join("manifest.json"))?)?;
        let hf: serde_json::Value =
            serde_json::from_str(&read_document(&root.join("tokenizer").join("config.json"))?)?;
        Self::from_splash(&manifest, &hf)
    }

    /// Builds the geometry out of the text subtree, once the architecture has
    /// been recognised.
    fn from_text(text: &serde_json::Value, quant: Option<QuantConfig>) -> CoreResult<Self> {
        let cfg: Qwen35MoeTextConfig = serde_json::from_value(text.clone())?;
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
        let shape = HybridShape {
            // Every attention slot of this family has the same widths, the same
            // rotary base, no window and its own cache, exactly as in the dense
            // family: the list is one descriptor repeated.
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
            layers,
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            // The SHARED expert's width, because the shared expert is this
            // slot's dense feed-forward. The routed experts' width is the
            // router's own field below, and `check` has already refused a
            // document where the two disagree in a way this build cannot run.
            intermediate: cfg.shared_expert_intermediate_size,
            gdn: GdnShape::new(
                cfg.linear_num_value_heads,
                cfg.linear_num_key_heads,
                cfg.linear_key_head_dim,
                cfg.linear_conv_kernel_dim,
            ),
            attn_output_gate: cfg.attn_output_gate,
            rms_eps,
            // The family switches, all of them the value the dense hybrid takes:
            // one pre-norm pair per slot, a SwiGLU gate, unscaled embeddings,
            // uncapped logits, no layer scalar, no per-layer embedding table,
            // three-stream media positions and an unnormalized value projection.
            activation: Activation::SwiGlu,
            norms: NormScheme::PreOnly,
            embed_scale: None,
            final_softcap: None,
            layer_scalar: false,
            ple: None,
            media_positions: MediaPositions::Mrope3D,
            v_norm: false,
            // The one field the dense family leaves `None`, and the whole
            // reason this module exists.
            moe: Some(MoeShape::new(
                cfg.num_experts,
                cfg.num_experts_per_tok,
                // One always-on expert. The config states its width rather than
                // its count, so the count is read off the presence of the width
                // - which is the only thing the document says about it.
                1,
                cfg.moe_intermediate_size,
            )),
            ssm: None,
            short_conv: None,
            no_ffn: Vec::new(),
        };
        shape.validate()?;
        Ok(Self {
            shape,
            eos_token: TokenId::new(cfg.eos_token_id),
            full_attention_interval: cfg.full_attention_interval,
            quant,
        })
    }

    /// Reads both documents out of a checkpoint directory.
    ///
    /// The quantization document is required rather than optional: every
    /// checkpoint of this family that exists is quantized, and a directory
    /// without one is a directory whose tensors this build has no decode for.
    /// Refusing by name here beats discovering it at the first expert.
    ///
    /// # Errors
    ///
    /// Whatever [`HfConfig::from_file`], [`QuantConfig::from_file`] and
    /// [`Qwen35MoeConfig::from_hf`] report.
    pub fn from_dir(dir: &Path) -> CoreResult<Self> {
        Self::from_files(&dir.join("config.json"), &dir.join("hf_quant_config.json"))
    }

    /// Reads the geometry out of the two documents, named individually.
    ///
    /// # Errors
    ///
    /// Whatever [`HfConfig::from_file`], [`QuantConfig::from_file`] and
    /// [`Qwen35MoeConfig::from_hf`] report.
    pub fn from_files(config: &Path, quant: &Path) -> CoreResult<Self> {
        let quant = QuantConfig::from_file(quant)?;
        Self::from_hf(&HfConfig::from_file(config)?, quant)
    }

    /// The geometry the stack runs.
    #[must_use]
    pub const fn shape(&self) -> &HybridShape {
        &self.shape
    }

    /// The router, which this family always has.
    ///
    /// [`HybridShape::moe`] is an `Option` because a dense family shares that
    /// shape; a checkpoint that reached this type has one, and
    /// [`Qwen35MoeConfig::from_hf`] is the only thing that builds one.
    #[must_use]
    pub fn moe(&self) -> MoeShape {
        self.shape.moe.unwrap_or(MoeShape::new(0, 0, 0, 0))
    }

    /// What every tensor's decode is chosen by, for the container that states
    /// it in a document of its own.
    ///
    /// `None` for a container that does not have one - a Splash package states
    /// the storage of each section in `layout.json` - and the caller that needs
    /// the document refuses by name rather than reading an empty map as "none
    /// of this is quantized".
    #[must_use]
    pub const fn quant(&self) -> Option<&QuantConfig> {
        self.quant.as_ref()
    }

    /// The end-of-sequence token the config document names.
    ///
    /// This is `<|endoftext|>`, not the end of a chat turn. Use
    /// [`Qwen35MoeConfig::stop_tokens`] to decide when a turn is over.
    #[must_use]
    pub const fn eos_token(&self) -> TokenId {
        self.eos_token
    }

    /// The ids that end a turn, most specific first.
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

/// Reads one of a Splash package's two documents, naming the path when it is
/// not there.
fn read_document(path: &Path) -> CoreResult<String> {
    std::fs::read_to_string(path).map_err(|_| CoreError::ConfigMissing {
        key: path.display().to_string(),
    })
}

/// Checks a Splash manifest's account of the model against the upstream
/// config's, field by field.
///
/// Every number here is stated by both documents. `expected` is always the
/// manifest's, because the manifest describes the bytes the packer wrote and is
/// authoritative for anything it states; `got` is always the upstream
/// checkpoint's. Neither is preferred silently: they have to be the same
/// number, and the point of the comparison is the package whose converter and
/// whose upstream drifted apart between the two.
///
/// # Errors
///
/// [`CoreError::UnknownArch`] when the manifest names another family,
/// [`CoreError::ShapeMismatch`] naming the manifest's field with both numbers,
/// and [`CoreError::InvalidValue`] naming the slot whose layer kind disagrees.
fn agree(manifest: &Manifest, config: &Qwen35MoeConfig) -> CoreResult<()> {
    let target = &manifest.target;
    if !ARCHES.contains(&target.architecture.as_str()) {
        return Err(CoreError::UnknownArch {
            model_type: target.architecture.clone(),
        });
    }
    let shape = config.shape();
    let moe = config.moe();
    let expect = HfConfig::expect;
    // `HfConfig::expect` takes the required value second, so each pair below
    // reads (upstream, manifest) and reports (expected: manifest, got:
    // upstream).
    expect(
        "manifest.target.hidden_size",
        u64::from(shape.hidden),
        u64::from(target.hidden_size),
    )?;
    expect(
        "manifest.target.layers",
        u64::from(shape.n_layers()),
        u64::from(target.layers),
    )?;
    expect(
        "manifest.target.experts",
        u64::from(moe.experts),
        u64::from(target.experts),
    )?;
    expect(
        "manifest.target.experts_per_token",
        u64::from(moe.used),
        u64::from(target.experts_per_token),
    )?;
    expect(
        "manifest.target.moe_intermediate_size",
        u64::from(moe.intermediate),
        u64::from(target.moe_intermediate_size),
    )?;
    // Two more that are easy to miss because the two documents spell them
    // differently. The shared expert's width is the slot's dense feed-forward
    // width, which is `HybridShape::intermediate` and not the router's; the
    // vocabulary is what the output head was packed to.
    expect(
        "manifest.target.shared_expert_intermediate_size",
        u64::from(shape.intermediate),
        u64::from(target.shared_expert_intermediate_size),
    )?;
    expect(
        "manifest.target.vocabulary_size",
        u64::from(shape.vocab),
        u64::from(target.vocabulary_size),
    )?;
    agree_on_layers(&target.layer_types, &shape.layers)
}

/// Checks the two layer rolls slot by slot.
///
/// A count that matches with a slot that does not is the failure worth naming:
/// every width still checks out, and the stack runs a recurrent block where the
/// package holds an attention one's weights.
fn agree_on_layers(manifest: &[splash::LayerKind], upstream: &[LayerKind]) -> CoreResult<()> {
    HfConfig::expect(
        "manifest.target.layer_types.len",
        u64::try_from(upstream.len()).unwrap_or(u64::MAX),
        u64::try_from(manifest.len()).unwrap_or(u64::MAX),
    )?;
    for (slot, (packed, stated)) in manifest.iter().zip(upstream).enumerate() {
        let packed = LayerKind::from(*packed);
        if packed != *stated {
            return Err(CoreError::InvalidValue {
                key: "manifest.target.layer_types",
                value: format!(
                    "slot {slot} is {packed:?} in the manifest and {stated:?} in the upstream config"
                ),
            });
        }
    }
    Ok(())
}

/// Checks every dimension against the value the kernels were written for.
fn check(cfg: &Qwen35MoeTextConfig) -> CoreResult<()> {
    let expect = HfConfig::expect;
    expect("num_hidden_layers", u64::from(cfg.num_hidden_layers), 40)?;
    expect("hidden_size", u64::from(cfg.hidden_size), 2048)?;
    expect("vocab_size", u64::from(cfg.vocab_size), 248_320)?;
    expect(
        "num_attention_heads",
        u64::from(cfg.num_attention_heads),
        16,
    )?;
    expect("num_key_value_heads", u64::from(cfg.num_key_value_heads), 2)?;
    expect("head_dim", u64::from(cfg.head_dim), 256)?;
    expect("num_experts", u64::from(cfg.num_experts), 256)?;
    expect("num_experts_per_tok", u64::from(cfg.num_experts_per_tok), 8)?;
    expect(
        "moe_intermediate_size",
        u64::from(cfg.moe_intermediate_size),
        512,
    )?;
    // Not a constant for its own sake: the shared expert runs through the same
    // scratch rows as a dense feed-forward, sized from
    // `HybridShape::intermediate`, while the routed columns are sized from the
    // router's own width. Two widths that differ are runnable, but nothing in
    // this build has been measured at them, so the checkpoint that exists is
    // the one that is accepted.
    expect(
        "shared_expert_intermediate_size",
        u64::from(cfg.shared_expert_intermediate_size),
        512,
    )?;
    expect(
        "full_attention_interval",
        u64::from(cfg.full_attention_interval),
        4,
    )?;
    expect(
        "linear_num_value_heads",
        u64::from(cfg.linear_num_value_heads),
        32,
    )?;
    expect(
        "linear_num_key_heads",
        u64::from(cfg.linear_num_key_heads),
        16,
    )?;
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
    check_mrope(&cfg.rope_parameters)?;
    expect_bool("attn_output_gate", cfg.attn_output_gate, true)
}

/// Checks the rotary sections against what the entry points implement.
///
/// An omitted `mrope_section` is accepted as the interleaved triple rather than
/// defaulted to something else: a document of this family that leaves the block
/// out means the layout its kernels already deal out, and inventing a different
/// default here would rotate every head against the wrong axis with no
/// diagnostic at all.
fn check_mrope(rope: &RopeParameters) -> CoreResult<()> {
    if !rope.mrope_section.is_empty() && rope.mrope_section != MROPE_SECTION {
        return Err(CoreError::InvalidValue {
            key: "rope_parameters.mrope_section",
            value: format!(
                "{:?}, where the entry points deal pairs out as {MROPE_SECTION:?}",
                rope.mrope_section
            ),
        });
    }
    if rope.mrope_section.is_empty() {
        return Ok(());
    }
    expect_bool(
        "rope_parameters.mrope_interleaved",
        rope.mrope_interleaved,
        true,
    )
}

/// Width of the rotated part of a head.
///
/// The fraction applies to the head, not the residual stream, and it is rounded
/// rather than truncated so a value that lands at 63.99999 through float
/// representation cannot quietly rotate one dimension fewer than the checkpoint.
fn rotated_width(head_dim: u32, factor: f64) -> u32 {
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
/// looked correct.
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

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;

//! Tensor names to layer roles.
//!
//! A model file is a flat bag of named tensors; a forward pass needs them
//! grouped by layer and by role. This module performs that mapping once, up
//! front, and validates every dimension while it is at it. Doing the whole
//! model at open time rather than lazily during generation is what turns a
//! misnamed or misshaped weight into a load error with the tensor's name in it,
//! instead of a wrong number in the fortieth token.
//!
//! The mapping is driven by [`TensorSource`], not by a file format: whatever
//! container a checkpoint comes from, once it can answer `find` and `get` by
//! name and report its architecture, [`bind_qwen35`] can bind it. Nothing in
//! this module mentions a mapping or a magic number.
//!
//! # Why this is not in the reader crate
//!
//! It was. Role names are facts about one checkpoint family, and a reader that
//! knew them could not open a checkpoint from another family without being
//! edited. The reader keeps the trait; this crate, which knows what a
//! `ssm_conv1d` is, keeps the names.
//!
//! # Two ways to hold a bound model
//!
//! The role tree is generic over what it stores for each weight, and there are
//! exactly two useful choices:
//!
//! - [`ModelBinding`] stores [`TensorView`]s, which borrow the file. This is
//!   the cheap one: binding a whole model copies no weights. It is also the
//!   restrictive one, because the binding cannot outlive the file and cannot be
//!   held by a struct that also owns it.
//! - [`DescBinding`] stores [`TensorDesc`]s, which own their names and borrow
//!   nothing. A loader that streams weights to a device wants this: it walks
//!   the tree, asks the container for each tensor's bytes in turn, uploads
//!   them, and keeps the tree afterwards for the dimensions and formats the
//!   kernels dispatch on - none of which need the payload.
//!
//! Both go through the same role names and the same layout checks, so a
//! checkpoint that binds one way binds the other way identically. That is the
//! reason the tree is generic rather than duplicated: two copies of the layout
//! checks would drift, and the drift would be silent.

use shiso_core::{DType, LayerIdx};
use shiso_engine::shape::{HybridShape, LayerKind};
use shiso_modelfile::reader::{GgufError, GgufResult, TensorDesc, TensorShape, TensorView};
use shiso_modelfile::source::TensorSource;

use crate::qwen35::config::Qwen35Config;

/// Architecture name this binding accepts.
const ARCH: &str = "qwen35";
/// The decision checkpoint built on this backbone: the same blocks under its
/// own name, with a schema head beside them that this binding never reads.
pub(crate) const CLEF_ARCH: &str = "clef";
/// Metadata key holding the layer count.
const BLOCK_COUNT_KEY: &str = "block_count";
/// Metadata key holding the attention layer stride.
const INTERVAL_KEY: &str = "full_attention_interval";

/// Formats a weight matrix may be stored in.
///
/// A single quantized file mixes all of these: the embedding and the `down`
/// projections are usually wider than the rest, and the small vectors are kept
/// in `f32`. Rejecting a file because it uses one format everywhere would make
/// the loader useless on real checkpoints.
pub const WEIGHT_TYPES: &[DType] = &[
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::Q8_0,
    DType::Q4_0,
    DType::Q4_K,
    DType::Q5_K,
    DType::Q6_K,
    DType::Pq2_0,
    DType::Ptq1_0,
];

/// Formats a normalization or control vector may be stored in.
///
/// These are read directly by the kernels as floats, so a quantized vector is a
/// layout error rather than a slower path.
pub const VECTOR_TYPES: &[DType] = &[DType::F32];

/// What the layout checks need from whatever a bound weight is stored as.
///
/// Private on purpose: it exists so the checks below can be written once for
/// both handle types, not so a caller can add a third. Every method is a field
/// read, so being generic over this costs nothing at runtime.
trait RoleWeight {
    /// Storage format the container declares.
    fn dtype(&self) -> DType;
    /// Dimensions, in format order.
    fn shape(&self) -> TensorShape;
}

impl RoleWeight for TensorView<'_> {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> TensorShape {
        self.shape
    }
}

impl RoleWeight for TensorDesc {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> TensorShape {
        self.shape
    }
}

/// Recurrent block weights.
#[derive(Debug)]
pub struct GdnRoles<W> {
    /// Combined query, key and value projection.
    pub qkv: W,
    /// Output gate projection.
    pub gate: W,
    /// Short causal convolution applied to the projections.
    pub conv1d: W,
    /// Per-head recurrence coefficient source.
    pub a: W,
    /// Per-head, per-token mixing weight.
    pub alpha: W,
    /// Per-head, per-token update weight.
    pub beta: W,
    /// Per-head time-step bias.
    pub dt_bias: W,
    /// Per-head normalization weights.
    pub norm: W,
    /// Output projection.
    pub out: W,
}

/// Attention block weights.
#[derive(Debug)]
pub struct AttnRoles<W> {
    /// Query projection, which carries the output gate when the shape says so.
    pub q: W,
    /// Key projection.
    pub k: W,
    /// Value projection.
    pub v: W,
    /// Query-head normalization weights.
    pub q_norm: W,
    /// Key-head normalization weights.
    pub k_norm: W,
    /// Output projection.
    pub output: W,
}

/// Feed-forward block weights.
#[derive(Debug)]
pub struct FfnRoles<W> {
    /// Gating projection.
    pub gate: W,
    /// Up projection.
    pub up: W,
    /// Down projection back to the residual width.
    pub down: W,
}

/// Everything one layer slot needs.
#[derive(Debug)]
pub struct LayerRoles<W> {
    /// Which block this slot holds.
    pub kind: LayerKind,
    /// Norm before the block's main projection.
    pub input_norm: W,
    /// Norm after the block, before the feed-forward.
    ///
    /// Named for the role the file names it by - qwen35 spells it
    /// `blk.N.post_attention_norm`, and the same role is `blk.N.ffn_norm`
    /// elsewhere. It is the feed-forward's PRE-norm, and it is what the engine
    /// holds as `LayerWeights::ffn_norm`.
    pub post_norm: W,
    /// Feed-forward weights.
    pub ffn: FfnRoles<W>,
    /// Recurrent weights, present exactly when `kind` is [`LayerKind::Gdn`].
    pub gdn: Option<GdnRoles<W>>,
    /// Attention weights, present exactly when `kind` is [`LayerKind::Attn`].
    pub attn: Option<AttnRoles<W>>,
}

/// Everything a forward pass needs, as one tree of roles.
#[derive(Debug)]
pub struct ModelRoles<W> {
    /// Token embedding.
    pub token_embd: W,
    /// Norm before the output projection.
    pub output_norm: W,
    /// Output projection, `None` when the model ties it to `token_embd`, which
    /// is the common case: a quantized file stores the embedding once and
    /// reuses it, because duplicating a vocabulary-wide matrix would dominate
    /// the file size.
    pub lm_head: Option<W>,
    /// One entry per layer slot, index 0 first.
    pub layers: Vec<LayerRoles<W>>,
}

/// Recurrent block weights, borrowed from an open model.
pub type GdnWeights<'f> = GdnRoles<TensorView<'f>>;
/// Attention block weights, borrowed from an open model.
pub type AttnWeights<'f> = AttnRoles<TensorView<'f>>;
/// Feed-forward block weights, borrowed from an open model.
pub type FfnWeights<'f> = FfnRoles<TensorView<'f>>;
/// One layer slot, borrowed from an open model.
pub type BoundLayer<'f> = LayerRoles<TensorView<'f>>;

/// Everything a forward pass needs, borrowed from an open model.
///
/// The borrow is the point: these views are windows onto the model mapping, so
/// binding costs no weight copies and cannot outlive the file.
pub type ModelBinding<'f> = ModelRoles<TensorView<'f>>;

/// Everything a forward pass needs, as owned descriptors.
///
/// Nothing here borrows the file, so a loader can bind once, stream every
/// weight to a device, and keep the tree for the dimensions and formats the
/// kernel dispatch reads. No payload survives, which is deliberate: a decoded
/// weight that is already on the device has no business staying resident on the
/// host.
pub type DescBinding = ModelRoles<TensorDesc>;

/// Dimensions the qwen35 role names are checked against, derived once from the
/// shape profile.
#[derive(Debug)]
struct Dims {
    hidden: u64,
    intermediate: u64,
    vocab: u64,
    gdn_head: u64,
    gdn_qkv_out: u64,
    gdn_gate_out: u64,
    gdn_out: u64,
    v_heads: u64,
    conv_kernel: u64,
    attn_q_out: u64,
    attn_kv_out: u64,
    attn_head: u64,
    attn_out: u64,
}

impl Dims {
    /// Derives every expected tensor width from the shape profile.
    ///
    /// The recurrent block's combined projection is twice the query/key width
    /// plus the value width, because the projection carries its own gate; the
    /// attention query projection doubles the same way when the config says the
    /// output is gated. Getting these two right matters more than it looks: the
    /// numbers are what separate a correct weight read from a plausible one.
    ///
    /// # Errors
    ///
    /// [`GgufError::LayoutMismatch`] when the profile states no attention slot
    /// at all, or states slots of differing geometry.
    fn new(shape: &HybridShape) -> GgufResult<Self> {
        let gdn = shape.gdn;
        // Every attention slot of this family has the same widths, so the first
        // one describes all of them - and that is a claim about the profile, not
        // a hope, so it is checked rather than assumed. A family whose slots
        // differ needs a set of expected widths per slot; measuring its later
        // slots by the first one's widths would reject correct tensors, and on a
        // file whose names happen to line up it would accept wrong ones.
        let Some(gqa) = shape.attn.first().copied() else {
            return Err(GgufError::LayoutMismatch {
                tensor: Box::from("shape profile"),
                want: Box::from("at least one attention slot"),
                got: Box::from("no attention slots"),
            });
        };
        if let Some(rank) = shape.attn.iter().position(|slot| *slot != gqa) {
            return Err(GgufError::LayoutMismatch {
                tensor: Box::from("shape profile"),
                want: Box::from("attention slots of one geometry"),
                got: Box::from(format!(
                    "attention slots of differing geometry, first at slot {rank}"
                )),
            });
        }
        let v_width = u64::from(gdn.v_heads) * u64::from(gdn.head_dim);
        let qk_width = u64::from(gdn.qk_heads) * u64::from(gdn.head_dim);
        let q_factor: u64 = if shape.attn_output_gate { 2 } else { 1 };
        Ok(Self {
            hidden: u64::from(shape.hidden),
            intermediate: u64::from(shape.intermediate),
            vocab: u64::from(shape.vocab),
            gdn_head: u64::from(gdn.head_dim),
            gdn_qkv_out: 2 * qk_width + v_width,
            gdn_gate_out: v_width,
            gdn_out: v_width,
            v_heads: u64::from(gdn.v_heads),
            conv_kernel: u64::from(gdn.conv_kernel),
            attn_q_out: q_factor * u64::from(gqa.q_heads) * u64::from(gqa.head_dim),
            attn_kv_out: u64::from(gqa.kv_heads) * u64::from(gqa.head_dim),
            attn_head: u64::from(gqa.head_dim),
            attn_out: u64::from(gqa.q_heads) * u64::from(gqa.head_dim),
        })
    }
}

/// Binds every tensor a qwen35 forward pass needs.
///
/// The whole model is validated here: role names, dimensions and storage
/// formats. A file that binds is a file the kernels can read without further
/// checks, which is why this function looks thorough - it is the last place
/// where a name typo can still be reported as a name.
///
/// # Errors
///
/// [`GgufError::ArchMismatch`] if the file is not a qwen35 model or `profile`
/// describes a different family; [`GgufError::ProfileMismatch`] if a metadata
/// key contradicts the profile; [`GgufError::MissingTensor`] if a required role
/// is absent; [`GgufError::LayoutMismatch`] if a tensor has the wrong rank,
/// dimensions or format.
pub fn bind_qwen35<'f>(
    src: &'f dyn TensorSource,
    config: &Qwen35Config,
) -> GgufResult<ModelBinding<'f>> {
    bind_roles(src, config, &|name| src.get(name))
}

/// Binds every tensor a qwen35 forward pass needs, as owned descriptors.
///
/// Same role names, same layout checks and the same errors as
/// [`bind_qwen35`] - the only difference is that the tree holds
/// [`TensorDesc`]s, so it borrows nothing and can be stored beside the file it
/// came from. That is what a weight-streaming loader needs: the payloads are
/// fetched one at a time and dropped, while the tree stays for the dimensions
/// and formats the kernels dispatch on.
///
/// # Errors
///
/// Identical to [`bind_qwen35`].
pub fn bind_qwen35_desc(src: &dyn TensorSource, config: &Qwen35Config) -> GgufResult<DescBinding> {
    bind_roles(src, config, &|name| src.find(name).cloned())
}

/// The shared body of the two binders.
///
/// `fetch` is the only thing that differs between them, and it is a closure
/// rather than another trait method so that a container does not have to know
/// that two flavours of binding exist.
fn bind_roles<W, F>(
    src: &dyn TensorSource,
    config: &Qwen35Config,
    fetch: &F,
) -> GgufResult<ModelRoles<W>>
where
    W: RoleWeight,
    F: Fn(&str) -> Option<W>,
{
    let shape = config.shape();
    let declared = src.architecture();
    if declared != Some(ARCH) && declared != Some(CLEF_ARCH) {
        return Err(GgufError::ArchMismatch {
            want: ARCH,
            got: Box::from(declared.unwrap_or_default()),
        });
    }

    let n_layers = shape.n_layers();
    match src.arch_u32(BLOCK_COUNT_KEY) {
        Some(got) if got == n_layers => {}
        Some(got) => {
            return Err(GgufError::ProfileMismatch {
                key: Box::from(key(BLOCK_COUNT_KEY)),
                want: u64::from(n_layers),
                got: u64::from(got),
            });
        }
        None => {
            return Err(GgufError::LayoutMismatch {
                tensor: Box::from(key(BLOCK_COUNT_KEY)),
                want: Box::from(n_layers.to_string()),
                got: Box::from("absent"),
            });
        }
    }
    check_interval(src, config)?;

    let dims = Dims::new(shape)?;

    let token_embd = tensor(
        fetch,
        "token_embd.weight",
        &[dims.hidden, dims.vocab],
        WEIGHT_TYPES,
    )?;
    let output_norm = tensor(fetch, "output_norm.weight", &[dims.hidden], VECTOR_TYPES)?;
    let lm_head = match src.find("output.weight") {
        Some(_) => Some(tensor(
            fetch,
            "output.weight",
            &[dims.hidden, dims.vocab],
            WEIGHT_TYPES,
        )?),
        None => None,
    };

    let mut layers = Vec::with_capacity(usize::try_from(n_layers).unwrap_or(0));
    for il in 0..n_layers {
        let idx = LayerIdx::new(il);
        let Some(kind) = shape.layer_kind(idx) else {
            return Err(GgufError::ProfileMismatch {
                key: Box::from(key(BLOCK_COUNT_KEY)),
                want: u64::from(n_layers),
                got: u64::try_from(shape.layers.len()).unwrap_or(u64::MAX),
            });
        };
        layers.push(bind_layer(fetch, il, kind, &dims)?);
    }

    Ok(ModelRoles {
        token_embd,
        output_norm,
        lm_head,
        layers,
    })
}

/// Cross-checks the file's attention stride against the profile's per-layer
/// kind list.
///
/// The two are independent statements about the same model - one written by the
/// checkpoint converter, one read from the architecture config - and a forward
/// pass that disagrees with itself produces plausible garbage, so the
/// disagreement has to stop the load.
fn check_interval(src: &dyn TensorSource, config: &Qwen35Config) -> GgufResult<()> {
    let shape = config.shape();
    let Some(interval) = src.arch_u32(INTERVAL_KEY) else {
        return Ok(());
    };
    if interval == 0 {
        return Ok(());
    }
    for il in 0..shape.n_layers() {
        let Some(kind) = shape.layer_kind(LayerIdx::new(il)) else {
            return Err(GgufError::ProfileMismatch {
                key: Box::from(key(BLOCK_COUNT_KEY)),
                want: u64::from(shape.n_layers()),
                got: u64::try_from(shape.layers.len()).unwrap_or(u64::MAX),
            });
        };
        let expected = if (il + 1).is_multiple_of(interval) {
            LayerKind::Attn
        } else {
            LayerKind::Gdn
        };
        if kind != expected {
            return Err(GgufError::ProfileMismatch {
                key: Box::from(key(INTERVAL_KEY)),
                want: u64::from(config.full_attention_interval()),
                got: u64::from(interval),
            });
        }
    }
    Ok(())
}

/// Binds one layer slot: the two norms and the feed-forward are common to both
/// block kinds, and exactly one of the two role groups is present.
fn bind_layer<W, F>(fetch: &F, il: u32, kind: LayerKind, dims: &Dims) -> GgufResult<LayerRoles<W>>
where
    W: RoleWeight,
    F: Fn(&str) -> Option<W>,
{
    let input_norm = tensor(
        fetch,
        &name(il, "attn_norm.weight"),
        &[dims.hidden],
        VECTOR_TYPES,
    )?;
    let post_norm = tensor(
        fetch,
        &name(il, "post_attention_norm.weight"),
        &[dims.hidden],
        VECTOR_TYPES,
    )?;
    let ffn = FfnRoles {
        gate: tensor(
            fetch,
            &name(il, "ffn_gate.weight"),
            &[dims.hidden, dims.intermediate],
            WEIGHT_TYPES,
        )?,
        up: tensor(
            fetch,
            &name(il, "ffn_up.weight"),
            &[dims.hidden, dims.intermediate],
            WEIGHT_TYPES,
        )?,
        down: tensor(
            fetch,
            &name(il, "ffn_down.weight"),
            &[dims.intermediate, dims.hidden],
            WEIGHT_TYPES,
        )?,
    };
    let (gdn, attn) = match kind {
        LayerKind::Gdn => (Some(bind_gdn(fetch, il, dims)?), None),
        LayerKind::Attn => (None, Some(bind_attn(fetch, il, dims)?)),
        // This family's shape holds neither; its config never states one.
        LayerKind::Ssm => {
            return Err(GgufError::LayoutMismatch {
                tensor: format!("blk.{il}").into(),
                want: Box::from("a recurrent or attention slot"),
                got: Box::from(format!("{kind:?}")),
            });
        }
    };
    Ok(LayerRoles {
        kind,
        input_norm,
        post_norm,
        ffn,
        gdn,
        attn,
    })
}

/// Binds the nine recurrent-block tensors.
fn bind_gdn<W, F>(fetch: &F, il: u32, dims: &Dims) -> GgufResult<GdnRoles<W>>
where
    W: RoleWeight,
    F: Fn(&str) -> Option<W>,
{
    Ok(GdnRoles {
        qkv: tensor(
            fetch,
            &name(il, "attn_qkv.weight"),
            &[dims.hidden, dims.gdn_qkv_out],
            WEIGHT_TYPES,
        )?,
        gate: tensor(
            fetch,
            &name(il, "attn_gate.weight"),
            &[dims.hidden, dims.gdn_gate_out],
            WEIGHT_TYPES,
        )?,
        conv1d: tensor(
            fetch,
            &name(il, "ssm_conv1d.weight"),
            &[dims.conv_kernel, dims.gdn_qkv_out],
            VECTOR_TYPES,
        )?,
        a: tensor(fetch, &name(il, "ssm_a"), &[dims.v_heads], VECTOR_TYPES)?,
        alpha: tensor(
            fetch,
            &name(il, "ssm_alpha.weight"),
            &[dims.hidden, dims.v_heads],
            WEIGHT_TYPES,
        )?,
        beta: tensor(
            fetch,
            &name(il, "ssm_beta.weight"),
            &[dims.hidden, dims.v_heads],
            WEIGHT_TYPES,
        )?,
        dt_bias: tensor(
            fetch,
            &name(il, "ssm_dt.bias"),
            &[dims.v_heads],
            VECTOR_TYPES,
        )?,
        norm: tensor(
            fetch,
            &name(il, "ssm_norm.weight"),
            &[dims.gdn_head],
            VECTOR_TYPES,
        )?,
        out: tensor(
            fetch,
            &name(il, "ssm_out.weight"),
            &[dims.gdn_out, dims.hidden],
            WEIGHT_TYPES,
        )?,
    })
}

/// Binds the six attention-block tensors.
fn bind_attn<W, F>(fetch: &F, il: u32, dims: &Dims) -> GgufResult<AttnRoles<W>>
where
    W: RoleWeight,
    F: Fn(&str) -> Option<W>,
{
    Ok(AttnRoles {
        q: tensor(
            fetch,
            &name(il, "attn_q.weight"),
            &[dims.hidden, dims.attn_q_out],
            WEIGHT_TYPES,
        )?,
        k: tensor(
            fetch,
            &name(il, "attn_k.weight"),
            &[dims.hidden, dims.attn_kv_out],
            WEIGHT_TYPES,
        )?,
        v: tensor(
            fetch,
            &name(il, "attn_v.weight"),
            &[dims.hidden, dims.attn_kv_out],
            WEIGHT_TYPES,
        )?,
        q_norm: tensor(
            fetch,
            &name(il, "attn_q_norm.weight"),
            &[dims.attn_head],
            VECTOR_TYPES,
        )?,
        k_norm: tensor(
            fetch,
            &name(il, "attn_k_norm.weight"),
            &[dims.attn_head],
            VECTOR_TYPES,
        )?,
        output: tensor(
            fetch,
            &name(il, "attn_output.weight"),
            &[dims.attn_out, dims.hidden],
            WEIGHT_TYPES,
        )?,
    })
}

/// Builds a layer-scoped tensor name.
fn name(il: u32, suffix: &str) -> String {
    format!("blk.{il}.{suffix}")
}

/// Fetches a tensor and checks its dimensions and format.
fn tensor<W, F>(fetch: &F, name: &str, dims: &[u64], allow: &[DType]) -> GgufResult<W>
where
    W: RoleWeight,
    F: Fn(&str) -> Option<W>,
{
    let weight = fetch(name).ok_or_else(|| GgufError::MissingTensor { name: name.into() })?;
    let shape = weight.shape();
    if shape.dims() != dims {
        return Err(GgufError::LayoutMismatch {
            tensor: name.into(),
            want: Box::from(format!("{dims:?}")),
            got: Box::from(format!("{:?}", shape.dims())),
        });
    }
    if !allow.contains(&weight.dtype()) {
        return Err(GgufError::LayoutMismatch {
            tensor: name.into(),
            want: Box::from(format_type_set(allow)),
            got: Box::from(weight.dtype().type_name()),
        });
    }
    Ok(weight)
}

/// Renders an accepted format set for an error message.
fn format_type_set(allow: &[DType]) -> String {
    let names: Vec<&str> = allow.iter().map(|dtype| dtype.type_name()).collect();
    format!("one of {}", names.join(", "))
}

/// Prefixes a metadata key with the architecture.
fn key(suffix: &str) -> String {
    format!("{ARCH}.{suffix}")
}

#[cfg(test)]
mod tests {
    use super::Dims;
    use shiso_engine::shape::{
        Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
    };
    use shiso_modelfile::reader::GgufError;

    /// A qwen35-shaped profile over `attn` attention slots and nothing else.
    ///
    /// The widths the expected tensor sizes are derived from are the only thing
    /// these cases read, so the rest is this family's own settings verbatim.
    fn shape(attn: Vec<AttnLayer>) -> HybridShape {
        HybridShape {
            layers: vec![LayerKind::Attn; attn.len()],
            attn,
            hidden: 2048,
            vocab: 128,
            intermediate: 4096,
            gdn: GdnShape::new(4, 2, 64, 4),
            attn_output_gate: false,
            rms_eps: 1e-6,
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
        }
    }

    /// Eight slots of one geometry is what this family states, and the widths
    /// come off the slot every one of them is a copy of.
    #[test]
    fn slots_of_one_geometry_bind() {
        let dims = Dims::new(&shape(vec![AttnLayer::new(16, 2, 128, 128, 128, 1e7); 8]))
            .expect("eight identical slots describe one set of widths");
        assert_eq!(dims.attn_q_out, 16 * 128);
        assert_eq!(dims.attn_kv_out, 2 * 128);
        assert_eq!(dims.attn_head, 128);
        assert_eq!(dims.attn_out, 16 * 128);
    }

    /// A profile whose slots differ has no single set of expected widths, and
    /// measuring the later slots by the first one's would report a correct
    /// tensor as misshaped - or, where the numbers happen to agree, read the
    /// wrong weight without a word.
    #[test]
    fn slots_of_differing_geometry_are_refused() {
        let attn = vec![
            AttnLayer::new(16, 2, 128, 128, 128, 1e7),
            AttnLayer::new(16, 2, 512, 128, 128, 1e7),
        ];
        match Dims::new(&shape(attn)) {
            Err(GgufError::LayoutMismatch { got, .. }) => assert!(
                got.contains("differing geometry"),
                "the refusal says what disagreed: {got}"
            ),
            other => panic!("expected a geometry refusal, got {other:?}"),
        }
    }

    /// And a profile with no attention slot at all, which used to derive every
    /// attention width from a zero-filled descriptor and then check real tensors
    /// against it.
    #[test]
    fn a_profile_with_no_attention_slot_is_refused() {
        match Dims::new(&shape(Vec::new())) {
            Err(GgufError::LayoutMismatch { got, .. }) => assert!(
                got.contains("no attention slots"),
                "the refusal says what was missing: {got}"
            ),
            other => panic!("expected an empty-profile refusal, got {other:?}"),
        }
    }
}

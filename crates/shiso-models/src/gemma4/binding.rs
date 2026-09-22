//! Gemma 4 tensor names to layer roles.
//!
//! A container is a flat bag of named tensors; a forward pass needs them
//! grouped by slot and by role, with every width already checked. That checking
//! happens here, once, at open time, which is what turns a misnamed or misshaped
//! weight into a load error carrying the tensor's name instead of a wrong number
//! in the fortieth token.
//!
//! Like the other families' bindings, this one is driven by [`TensorSource`]
//! rather than by a file format: anything that can report its architecture and
//! answer `find` by name can be bound.
//!
//! # What is specific to this family
//!
//! Two things, and both are the reason this is not the qwen35 binding with the
//! names swapped.
//!
//! The first is the per-layer embedding path. Every slot carries an `inp_gate`
//! and a `proj` of its own, and above them the model carries one table,
//! `per_layer_token_embd`, holding every slot's row for every token. That table
//! is the largest tensor in the file and no forward pass ever needs more than
//! forty-two rows of it at a time, so it is bound as a descriptor and left in
//! the container to be streamed. Nothing here reads a payload.
//!
//! The second is that eighteen of the forty-two slots write no key/value runs;
//! they read an earlier slot's. The geometry says which, and the file agrees by
//! simply not shipping those slots' `attn_k`, `attn_v` and `attn_k_norm`. So a
//! sharing slot's key and value roles are [`Option`]s - and a file that ships
//! them anyway is refused rather than bound, because binding it would leave a
//! real tensor unread while every dimension still checked out.

use shiso_core::{DType, LayerIdx};
use shiso_engine::shape::{HybridShape, KvSource};
use shiso_modelfile::reader::{GgufError, GgufResult, TensorDesc};
use shiso_modelfile::source::TensorSource;

use crate::gemma4::config::{GGUF_ARCH, Gemma4Config};

/// Metadata key holding the slot count.
const BLOCK_COUNT_KEY: &str = "block_count";
/// Metadata key holding how many trailing slots read an earlier slot's runs.
const SHARED_KV_KEY: &str = "attention.shared_kv_layers";
/// Metadata key holding the width of one per-layer embedding row.
const PLE_DIM_KEY: &str = "embedding_length_per_layer_input";
/// The untied output projection, which this family never carries.
const LM_HEAD: &str = "output.weight";

/// Formats a weight matrix may be stored in.
///
/// A quantized container of this family mixes them: the bulk of the model is
/// `Q4_0`, the two vocabulary-wide tables are `Q6_K`, and the per-layer model
/// projection is `F16`. Insisting on one format everywhere would refuse every
/// real checkpoint.
///
/// `BF16` is deliberately absent: no container this family ships carries it
/// today, and it is added the day one does rather than admitted ahead of a
/// checkpoint that would exercise it.
pub const WEIGHT_TYPES: &[DType] = &[
    DType::F32,
    DType::F16,
    DType::Q4_0,
    DType::Q4_K,
    DType::Q6_K,
    DType::Q8_0,
];

/// Formats a normalization or control vector may be stored in.
///
/// The kernels read these directly as floats, so a quantized vector is a layout
/// error rather than a slower path.
pub const VECTOR_TYPES: &[DType] = &[DType::F32];

/// Attention weights of one slot.
///
/// The key, value and key-norm roles are absent exactly when the slot reads an
/// earlier slot's key/value runs. That is a fact about the geometry, so it is in
/// the type rather than in a comment: a consumer cannot forget to ask.
#[derive(Debug)]
pub struct Gemma4AttnRoles<W> {
    /// Query projection.
    pub q: W,
    /// Key projection, present only when this slot owns its runs.
    pub k: Option<W>,
    /// Value projection, present only when this slot owns its runs.
    pub v: Option<W>,
    /// Output projection back to the residual width.
    pub output: W,
    /// Query-head normalization weights.
    pub q_norm: W,
    /// Key-head normalization weights, present only when this slot owns its
    /// runs.
    pub k_norm: Option<W>,
}

/// Feed-forward weights of one slot.
#[derive(Debug)]
pub struct Gemma4FfnRoles<W> {
    /// Gating projection.
    pub gate: W,
    /// Up projection.
    pub up: W,
    /// Down projection back to the residual width.
    pub down: W,
}

/// The per-layer embedding weights of one slot.
///
/// The slot's row of the model-wide table arrives already projected; these three
/// are what turn it into the contribution the residual stream takes.
#[derive(Debug)]
pub struct Gemma4PleRoles<W> {
    /// Gate over the residual stream, spelled `blk.N.inp_gate` in the file.
    pub gate: W,
    /// Projection from the per-layer width back to the residual width, spelled
    /// `blk.N.proj`.
    pub proj: W,
    /// Norm applied after that projection, spelled `blk.N.post_norm`.
    pub post_norm: W,
}

/// Everything one slot needs.
///
/// Four norms rather than two: this family normalizes on both sides of the
/// attention block and on both sides of the feed-forward, and the file names all
/// four differently, so the field names say which side they are on rather than
/// which spelling they came from.
#[derive(Debug)]
pub struct Gemma4LayerRoles<W> {
    /// Norm before attention, spelled `blk.N.attn_norm`.
    pub input_norm: W,
    /// Norm after attention, spelled `blk.N.post_attention_norm`.
    pub post_attn_norm: W,
    /// Norm before the feed-forward, spelled `blk.N.ffn_norm`.
    pub pre_ffn_norm: W,
    /// Norm after the feed-forward, spelled `blk.N.post_ffw_norm`.
    pub post_ffn_norm: W,
    /// The scalar this slot's output is multiplied by, spelled
    /// `blk.N.layer_output_scale`. One element wide.
    pub layer_scale: W,
    /// Attention weights.
    pub attn: Gemma4AttnRoles<W>,
    /// Feed-forward weights.
    pub ffn: Gemma4FfnRoles<W>,
    /// Per-layer embedding weights.
    pub ple: Gemma4PleRoles<W>,
}

/// Everything a Gemma 4 forward pass needs, as one tree of roles.
#[derive(Debug)]
pub struct Gemma4Roles<W> {
    /// Token embedding, which is also the output projection: this family ties
    /// them, and a container carrying an untied one is refused.
    pub token_embd: W,
    /// Norm before the output projection.
    pub output_norm: W,
    /// Projection from the concatenated per-layer rows into the width the slots
    /// read, spelled `per_layer_model_proj`.
    pub ple_model_proj: W,
    /// Norm on that projection's output, spelled `per_layer_proj_norm`.
    pub ple_proj_norm: W,
    /// The per-layer embedding table, spelled `per_layer_token_embd`.
    ///
    /// Bound as a descriptor and left where it is. It holds one row per slot per
    /// token and is the largest tensor in the file, while a forward pass needs
    /// only the rows of the tokens in front of it - so it is streamed from the
    /// container rather than imported to the device.
    pub ple_table: W,
    /// One entry per slot, index 0 first.
    pub layers: Vec<Gemma4LayerRoles<W>>,
}

/// Everything a Gemma 4 forward pass needs, as owned descriptors.
///
/// Nothing here borrows the container, so a loader can bind once, stream every
/// weight to a device, and keep the tree afterwards for the dimensions and
/// formats the kernels dispatch on - none of which need the payload.
pub type Gemma4DescBinding = Gemma4Roles<TensorDesc>;

/// Model-wide widths the role names are checked against.
#[derive(Debug)]
struct Dims {
    /// Residual-stream width.
    hidden: u64,
    /// Feed-forward inner width.
    intermediate: u64,
    /// Vocabulary, which is the token embedding's row count.
    vocab: u64,
    /// Width of one slot's per-layer embedding row.
    ///
    /// Held at the profile's own width rather than widened, because this one is
    /// checked against a metadata value as well as used as a dimension.
    /// Narrowing it back for that comparison would need a fallback, and a
    /// fallback here would invent the very number it then reported as expected.
    ple_dim: u32,
    /// Every slot's per-layer row concatenated: `n_layers * ple_dim`.
    ple_rows: u64,
    /// Rows of the per-layer table, which is a vocabulary of its own.
    ple_vocab: u32,
}

impl Dims {
    /// Derives the model-wide widths from the shape profile.
    ///
    /// Both per-layer widths come off [`HybridShape::ple`], which already
    /// carries the pair. [`Gemma4Config`] states the table's row count a second
    /// time, and reading it from there would leave two paths to one number with
    /// nothing checking that they agree.
    ///
    /// # Errors
    ///
    /// [`GgufError::LayoutMismatch`] when the profile states no per-layer
    /// embedding table, which this family cannot run without.
    fn new(shape: &HybridShape) -> GgufResult<Self> {
        let Some(ple) = shape.ple else {
            return Err(GgufError::LayoutMismatch {
                tensor: Box::from("shape profile"),
                want: Box::from("a per-layer embedding table"),
                got: Box::from("none"),
            });
        };
        Ok(Self {
            hidden: u64::from(shape.hidden),
            intermediate: u64::from(shape.intermediate),
            vocab: u64::from(shape.vocab),
            ple_dim: ple.dim,
            ple_rows: u64::from(shape.n_layers()) * u64::from(ple.dim),
            ple_vocab: ple.vocab,
        })
    }
}

/// Widths of one slot, which differ between the two attention kinds.
///
/// A windowed slot's head is 256 wide and a full-causal one's is 512, so the
/// expected sizes cannot be derived once for the whole model the way a
/// single-geometry family's can.
#[derive(Debug)]
struct SlotDims {
    /// Width of one head.
    head: u64,
    /// Query projection width.
    q_out: u64,
    /// Key or value projection width.
    kv_out: u64,
    /// Whether this slot writes its own key/value runs.
    owns_kv: bool,
}

/// Binds every tensor a Gemma 4 forward pass needs, as owned descriptors.
///
/// The whole model is validated here: role names, dimensions and storage
/// formats, plus the two statements the container makes about itself that the
/// geometry can contradict - its slot count and how many of those slots share
/// key/value runs. A container that binds is one the kernels can read without
/// further checks.
///
/// # Errors
///
/// [`GgufError::ArchMismatch`] if the container is not a Gemma 4 model;
/// [`GgufError::ProfileMismatch`] if a metadata key contradicts the geometry;
/// [`GgufError::MissingTensor`] naming the first required role that is absent;
/// [`GgufError::LayoutMismatch`] for a tensor of the wrong rank, dimensions or
/// format, for an untied output projection, and for a sharing slot that carries
/// key/value weights of its own.
pub fn bind_gemma4_desc(
    src: &dyn TensorSource,
    config: &Gemma4Config,
) -> GgufResult<Gemma4DescBinding> {
    let shape = config.shape();
    let declared = src.architecture();
    if declared != Some(GGUF_ARCH) {
        return Err(GgufError::ArchMismatch {
            want: GGUF_ARCH,
            got: Box::from(declared.unwrap_or_default()),
        });
    }

    let n_layers = shape.n_layers();
    let dims = Dims::new(shape)?;
    expect_meta(src, BLOCK_COUNT_KEY, n_layers)?;
    expect_meta(src, SHARED_KV_KEY, shared_kv_slots(shape))?;
    expect_meta(src, PLE_DIM_KEY, dims.ple_dim)?;

    // The embedding is the output projection in this family. A container that
    // ships a separate one describes a model these kernels do not run, and
    // binding it would silently ignore a vocabulary-wide tensor.
    if src.find(LM_HEAD).is_some() {
        return Err(GgufError::LayoutMismatch {
            tensor: Box::from(LM_HEAD),
            want: Box::from("absent: this family ties the output projection to the embedding"),
            got: Box::from("present"),
        });
    }

    let token_embd = tensor(
        src,
        "token_embd.weight",
        &[dims.hidden, dims.vocab],
        WEIGHT_TYPES,
    )?;
    let output_norm = tensor(src, "output_norm.weight", &[dims.hidden], VECTOR_TYPES)?;
    let ple_model_proj = tensor(
        src,
        "per_layer_model_proj.weight",
        &[dims.hidden, dims.ple_rows],
        WEIGHT_TYPES,
    )?;
    let ple_proj_norm = tensor(
        src,
        "per_layer_proj_norm.weight",
        &[u64::from(dims.ple_dim)],
        VECTOR_TYPES,
    )?;
    let ple_table = tensor(
        src,
        "per_layer_token_embd.weight",
        &[dims.ple_rows, u64::from(dims.ple_vocab)],
        WEIGHT_TYPES,
    )?;

    let mut layers = Vec::with_capacity(shape.layers.len());
    for il in 0..n_layers {
        let slot = slot_dims(shape, il)?;
        layers.push(bind_layer(src, il, &dims, &slot)?);
    }

    Ok(Gemma4Roles {
        token_embd,
        output_norm,
        ple_model_proj,
        ple_proj_norm,
        ple_table,
        layers,
    })
}

/// How many attention slots read an earlier slot's key/value runs.
///
/// Counted off the geometry rather than taken from the metadata, because the
/// metadata is the statement being checked.
fn shared_kv_slots(shape: &HybridShape) -> u32 {
    let shared = shape
        .attn
        .iter()
        .filter(|slot| slot.kv != KvSource::Own)
        .count();
    u32::try_from(shared).unwrap_or(u32::MAX)
}

/// The widths of one slot, from the geometry's descriptor for it.
fn slot_dims(shape: &HybridShape, il: u32) -> GgufResult<SlotDims> {
    let Some(slot) = shape.attn_layer(LayerIdx::new(il)) else {
        return Err(GgufError::LayoutMismatch {
            tensor: Box::from(format!("blk.{il}")),
            want: Box::from("an attention slot"),
            got: Box::from("a slot the profile states no attention geometry for"),
        });
    };
    let head = u64::from(slot.head_dim);
    Ok(SlotDims {
        head,
        q_out: u64::from(slot.q_heads) * head,
        kv_out: u64::from(slot.kv_heads) * head,
        owns_kv: slot.kv == KvSource::Own,
    })
}

/// Binds one slot: four norms, a scalar, and the three role groups.
fn bind_layer(
    src: &dyn TensorSource,
    il: u32,
    dims: &Dims,
    slot: &SlotDims,
) -> GgufResult<Gemma4LayerRoles<TensorDesc>> {
    let norm = |suffix: &str| tensor(src, &name(il, suffix), &[dims.hidden], VECTOR_TYPES);
    Ok(Gemma4LayerRoles {
        input_norm: norm("attn_norm.weight")?,
        post_attn_norm: norm("post_attention_norm.weight")?,
        pre_ffn_norm: norm("ffn_norm.weight")?,
        post_ffn_norm: norm("post_ffw_norm.weight")?,
        layer_scale: tensor(
            src,
            &name(il, "layer_output_scale.weight"),
            &[1],
            VECTOR_TYPES,
        )?,
        attn: bind_attn(src, il, dims, slot)?,
        ffn: bind_ffn(src, il, dims)?,
        ple: bind_ple(src, il, dims)?,
    })
}

/// Binds a slot's attention weights, key and value included only when the slot
/// owns its runs.
///
/// A sharing slot is not merely allowed to omit them: it is required to. The
/// alternative - binding whatever is there and letting the geometry decide what
/// to read - would accept a container whose schedule disagrees with the config
/// and quietly never read one of its tensors.
fn bind_attn(
    src: &dyn TensorSource,
    il: u32,
    dims: &Dims,
    slot: &SlotDims,
) -> GgufResult<Gemma4AttnRoles<TensorDesc>> {
    let q = tensor(
        src,
        &name(il, "attn_q.weight"),
        &[dims.hidden, slot.q_out],
        WEIGHT_TYPES,
    )?;
    let output = tensor(
        src,
        &name(il, "attn_output.weight"),
        &[slot.q_out, dims.hidden],
        WEIGHT_TYPES,
    )?;
    let q_norm = tensor(
        src,
        &name(il, "attn_q_norm.weight"),
        &[slot.head],
        VECTOR_TYPES,
    )?;
    let (k, v, k_norm) = if slot.owns_kv {
        let kv = &[dims.hidden, slot.kv_out];
        (
            Some(tensor(src, &name(il, "attn_k.weight"), kv, WEIGHT_TYPES)?),
            Some(tensor(src, &name(il, "attn_v.weight"), kv, WEIGHT_TYPES)?),
            Some(tensor(
                src,
                &name(il, "attn_k_norm.weight"),
                &[slot.head],
                VECTOR_TYPES,
            )?),
        )
    } else {
        for suffix in ["attn_k.weight", "attn_v.weight", "attn_k_norm.weight"] {
            refuse_present(src, &name(il, suffix))?;
        }
        (None, None, None)
    };
    Ok(Gemma4AttnRoles {
        q,
        k,
        v,
        output,
        q_norm,
        k_norm,
    })
}

/// Binds a slot's three feed-forward projections.
fn bind_ffn(
    src: &dyn TensorSource,
    il: u32,
    dims: &Dims,
) -> GgufResult<Gemma4FfnRoles<TensorDesc>> {
    let wide = &[dims.hidden, dims.intermediate];
    Ok(Gemma4FfnRoles {
        gate: tensor(src, &name(il, "ffn_gate.weight"), wide, WEIGHT_TYPES)?,
        up: tensor(src, &name(il, "ffn_up.weight"), wide, WEIGHT_TYPES)?,
        down: tensor(
            src,
            &name(il, "ffn_down.weight"),
            &[dims.intermediate, dims.hidden],
            WEIGHT_TYPES,
        )?,
    })
}

/// Binds a slot's three per-layer embedding tensors.
fn bind_ple(
    src: &dyn TensorSource,
    il: u32,
    dims: &Dims,
) -> GgufResult<Gemma4PleRoles<TensorDesc>> {
    Ok(Gemma4PleRoles {
        gate: tensor(
            src,
            &name(il, "inp_gate.weight"),
            &[dims.hidden, u64::from(dims.ple_dim)],
            WEIGHT_TYPES,
        )?,
        proj: tensor(
            src,
            &name(il, "proj.weight"),
            &[u64::from(dims.ple_dim), dims.hidden],
            WEIGHT_TYPES,
        )?,
        post_norm: tensor(
            src,
            &name(il, "post_norm.weight"),
            &[dims.hidden],
            VECTOR_TYPES,
        )?,
    })
}

/// Builds a slot-scoped tensor name.
fn name(il: u32, suffix: &str) -> String {
    format!("blk.{il}.{suffix}")
}

/// Fetches a descriptor and checks its dimensions and format.
fn tensor(
    src: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    allow: &[DType],
) -> GgufResult<TensorDesc> {
    let weight = src
        .find(name)
        .cloned()
        .ok_or_else(|| GgufError::MissingTensor { name: name.into() })?;
    if weight.shape.dims() != dims {
        return Err(GgufError::LayoutMismatch {
            tensor: name.into(),
            want: Box::from(format!("{dims:?}")),
            got: Box::from(format!("{:?}", weight.shape.dims())),
        });
    }
    if !allow.contains(&weight.dtype) {
        return Err(GgufError::LayoutMismatch {
            tensor: name.into(),
            want: Box::from(format_type_set(allow)),
            got: Box::from(weight.dtype.type_name()),
        });
    }
    Ok(weight)
}

/// Refuses a tensor a sharing slot must not carry.
fn refuse_present(src: &dyn TensorSource, name: &str) -> GgufResult<()> {
    if src.find(name).is_some() {
        return Err(GgufError::LayoutMismatch {
            tensor: name.into(),
            want: Box::from("absent: a shared slot reads an earlier slot's key/value runs"),
            got: Box::from("present"),
        });
    }
    Ok(())
}

/// Checks one architecture-prefixed count against the geometry.
fn expect_meta(src: &dyn TensorSource, suffix: &str, want: u32) -> GgufResult<()> {
    match src.arch_u32(suffix) {
        Some(got) if got == want => Ok(()),
        Some(got) => Err(GgufError::ProfileMismatch {
            key: Box::from(key(suffix)),
            want: u64::from(want),
            got: u64::from(got),
        }),
        None => Err(GgufError::LayoutMismatch {
            tensor: Box::from(key(suffix)),
            want: Box::from(want.to_string()),
            got: Box::from("absent"),
        }),
    }
}

/// Renders an accepted format set for an error message.
fn format_type_set(allow: &[DType]) -> String {
    let names: Vec<&str> = allow.iter().map(|dtype| dtype.type_name()).collect();
    format!("one of {}", names.join(", "))
}

/// Prefixes a metadata key with the architecture.
fn key(suffix: &str) -> String {
    format!("{GGUF_ARCH}.{suffix}")
}

//! The draft head this family's sidecar has been naming all along.
//!
//! A speculative step needs a second model cheap enough to run several times
//! per target step and close enough to agree with it most of the time. This
//! family ships one in the sidecar and nowhere else: the GGUF export carries
//! 426 tensors and not one of them matches `mtp`, so a GGUF target reads its
//! head from the same sidecar directory its `config.json`, `tokenizer.json`
//! and `chat_template.jinja` already come from.
//!
//! # What the document states, and what it only describes
//!
//! Five keys mention the head. Three carry information and are read here:
//!
//! - `mtp_file` - the container, relative to the sidecar directory;
//! - `mtp_num_hidden_layers` - blocks in the head, which is one for every head
//!   this tree has seen;
//! - `mtplx_mtp_quantization` - the codec its packed tensors are written at.
//!
//! Two are descriptive and are deliberately ignored. `mtp_tensor_count` is a
//! census of the container, which the container itself states and which
//! [`MtpWeights::bind`] checks against the tensors it actually found;
//! `mtp_policy` is a producer's label (`optiq-int4-prequantized-gs64`) that
//! restates `mtplx_mtp_quantization` as prose. Reading either would be a second
//! statement of a fact this reader already has from a better source, and a
//! second statement is a second thing that can disagree.
//!
//! # A named head that is absent is an error, and an unnamed one is not
//!
//! [`read_manifest`] answers `None` when the document names no `mtp_file`,
//! because most checkpoints ship no head and a build without one decodes one
//! token a step exactly as it always did. It *refuses* when the document names
//! a file the directory does not hold, and the refusal carries the path it
//! looked at: the alternative is a server started with `--spec-k 3` that
//! silently falls back to plain decode, answers correctly, and gives the
//! operator no way to tell which of the two they got.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use half::f16;
use serde::Deserialize;
use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, alloc};
use shiso_core::config::HfConfig;
use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::attn::Attn;
use shiso_engine::draft::DraftParts;
use shiso_engine::ffn::Ffn;
use shiso_engine::projection::Weight;
use shiso_modelfile::mlx_quant::MlxAffineLayout;
use shiso_modelfile::st_header::{Shard, StDType};

/// Bits, group size and mode the head's tensors are packed at.
///
/// A type of its own rather than three fields on the manifest, because the
/// three only mean anything together: a group size read beside the wrong bit
/// width addresses a scale row that exists, and every dequantization after it
/// succeeds and is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineQuant {
    /// Bits per packed weight.
    pub bits: u32,
    /// Weights one scale and one bias cover.
    pub group_size: u32,
    /// Whether the container's words are already packed, rather than being
    /// full-precision rows a loader is expected to quantize.
    ///
    /// Every head this tree has seen states `true`. A `false` would mean a
    /// container of BF16 rows under an affine label, which this reader binds
    /// as it finds and does not quantize.
    pub prequantized: bool,
}

impl AffineQuant {
    /// The codec `mtp_policy` spells for this family: `int4`, group 64,
    /// already packed.
    ///
    /// Used when the document names a head without describing its codec. That
    /// combination has not been seen in the wild; it is answered rather than
    /// refused because the producer's own label states the same three numbers,
    /// and because a head that binds at the wrong codec is caught by
    /// [`MtpWeights::bind`]'s layout check rather than by a guess here.
    #[must_use]
    pub const fn family_default() -> Self {
        Self {
            bits: 4,
            group_size: 64,
            prequantized: true,
        }
    }
}

impl Default for AffineQuant {
    fn default() -> Self {
        Self::family_default()
    }
}

/// What the sidecar's document says about the draft head.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller reads all three, and a test
/// builds one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpManifest {
    /// Path to the head's container, resolved against the sidecar directory.
    pub file: PathBuf,
    /// Layers in the head. One, for every head this tree has seen.
    pub layers: u32,
    /// Bits, group size and mode the head's tensors are packed at.
    pub quant: AffineQuant,
}

/// The `mtplx_mtp_quantization` subtree, as the document writes it.
///
/// `mode` and `policy` are in the document and are not modeled: the first is
/// `affine` on every head this tree reads and is asserted by the reader that
/// unpacks a triple, and the second is the producer's label.
#[derive(Debug, Deserialize)]
struct RawQuant {
    #[serde(default = "default_bits")]
    bits: u32,
    #[serde(default = "default_group_size")]
    group_size: u32,
    #[serde(default)]
    prequantized: bool,
}

/// Bits a codec subtree means when it omits them.
const fn default_bits() -> u32 {
    AffineQuant::family_default().bits
}

/// The group a codec subtree means when it omits one.
const fn default_group_size() -> u32 {
    AffineQuant::family_default().group_size
}

/// Blocks a head has when `mtp_num_hidden_layers` is absent.
///
/// One. A head is a single block of the target's own architecture in front of
/// two normalizations and a projection, and a document that omits the count is
/// describing that shape rather than a deeper one.
const DEFAULT_LAYERS: u32 = 1;

/// Reads what the sidecar at `sidecar` says about its draft head.
///
/// `None` when the document names no `mtp_file`, which is not an error: most
/// checkpoints ship no head.
///
/// # Errors
///
/// [`CoreError::Io`] when there is no `config.json` to read,
/// [`CoreError::Json`] when it does not parse, and
/// [`CoreError::ConfigMissing`] naming the resolved path when the document
/// names a head the directory does not hold.
pub fn read_manifest(sidecar: &Path) -> CoreResult<Option<MtpManifest>> {
    let config = HfConfig::from_file(&sidecar.join("config.json"))?;
    read_manifest_from(sidecar, &config)
}

/// As [`read_manifest`], against a document that is already parsed.
///
/// Separate because a loader that has opened the sidecar's config to build the
/// target's geometry should not parse the same 55 KB again to find out whether
/// it ships a head.
///
/// `None` is not a refusal: most checkpoints of this family ship no head at
/// all, and a container that declares none is a container that speculation is
/// simply off for.
///
/// # Errors
///
/// As [`read_manifest`], less the two the parse itself reports.
pub fn read_manifest_from(sidecar: &Path, config: &HfConfig) -> CoreResult<Option<MtpManifest>> {
    let Ok(named) = config.section("mtp_file") else {
        return Ok(None);
    };
    let relative = named.as_str().ok_or_else(|| CoreError::InvalidValue {
        key: "mtp_file",
        value: named.to_string(),
    })?;
    let file = sidecar.join(relative);
    if !file.is_file() {
        return Err(CoreError::ConfigMissing {
            key: format!(
                "mtp_file names {relative}, and {} is not a file",
                file.display()
            ),
        });
    }
    let quant = match config.section("mtplx_mtp_quantization") {
        Ok(subtree) => {
            let raw: RawQuant = serde_json::from_value(subtree.clone())?;
            AffineQuant {
                bits: raw.bits,
                group_size: raw.group_size,
                prequantized: raw.prequantized,
            }
        }
        Err(_) => AffineQuant::family_default(),
    };
    let layers = config
        .section("mtp_num_hidden_layers")
        .ok()
        .and_then(serde_json::Value::as_u64)
        .and_then(|count| u32::try_from(count).ok())
        .unwrap_or(DEFAULT_LAYERS);
    tracing::debug!(
        file = %file.display(),
        layers,
        bits = quant.bits,
        group_size = quant.group_size,
        "the sidecar names a draft head"
    );
    Ok(Some(MtpManifest {
        file,
        layers,
        quant,
    }))
}

/// One tensor of the head, as the container stores it.
///
/// A descriptor rather than a slice, so that [`MtpWeights`] can own the mapping
/// and hand out bytes for a tensor without every field of the binding being a
/// borrow of the same value.
#[derive(Debug, Clone)]
pub struct HeadTensor {
    /// The container's own name for it.
    name: Box<str>,
    /// Storage format, as the header spells it.
    dtype: StDType,
    /// Dimensions, slowest-varying axis first.
    shape: Vec<u64>,
    /// Payload span, relative to the container's payload base.
    span: [u64; 2],
}

impl HeadTensor {
    /// The container's own name for this tensor.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Dimensions, slowest-varying axis first.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Storage format, as the header spells it.
    #[must_use]
    pub const fn dtype(&self) -> StDType {
        self.dtype
    }

    /// One axis of the shape, or the refusal naming the tensor.
    fn axis(&self, at: usize) -> CoreResult<u64> {
        self.shape
            .get(at)
            .copied()
            .ok_or_else(|| CoreError::ShapeMismatch {
                field: "draft head tensor rank",
                expected: u64::try_from(at).unwrap_or(u64::MAX).saturating_add(1),
                got: u64::try_from(self.shape.len()).unwrap_or(u64::MAX),
            })
    }
}

/// One affine-packed matrix of the head, with the three arrays that decode it.
///
/// The three travel together because none of them means anything alone, and
/// the layout travels with them because it is derived from all three: a triple
/// whose scales row count disagrees with its packed word count is a group size
/// read from the wrong place, and every weight it produces after that is
/// finite, plausible and wrong.
#[derive(Debug, Clone)]
pub struct PackedMatrix {
    /// The packed words, `[out_rows, in_dim * bits / 32]`.
    words: HeadTensor,
    /// One scale per group, `[out_rows, in_dim / group_size]`.
    scales: HeadTensor,
    /// One bias per group, the same shape as the scales.
    biases: HeadTensor,
    /// What the three decode under.
    layout: MlxAffineLayout,
}

impl PackedMatrix {
    /// The container's name for the packed words.
    #[must_use]
    pub fn name(&self) -> &str {
        self.words.name()
    }

    /// The shape of the packed words, which is not the logical matrix's.
    #[must_use]
    pub fn packed_shape(&self) -> &[u64] {
        self.words.shape()
    }

    /// What the three arrays decode under.
    #[must_use]
    pub const fn layout(&self) -> MlxAffineLayout {
        self.layout
    }

    /// The three descriptors, in the order
    /// [`shiso_modelfile::mlx_quant::dequant_affine`] takes them.
    #[must_use]
    pub const fn arrays(&self) -> [&HeadTensor; 3] {
        [&self.words, &self.scales, &self.biases]
    }
}

/// The head's single attention block.
#[derive(Debug)]
pub struct MtpAttn {
    /// The query projection, doubled by this family's folded output gate.
    pub q_proj: PackedMatrix,
    /// The key projection.
    pub k_proj: PackedMatrix,
    /// The value projection.
    pub v_proj: PackedMatrix,
    /// The output projection.
    pub o_proj: PackedMatrix,
    /// Per-head query normalization, one weight of `head_dim`.
    pub q_norm: HeadTensor,
    /// Per-head key normalization, one weight of `head_dim`.
    pub k_norm: HeadTensor,
}

/// The head's single feed-forward.
#[derive(Debug)]
pub struct MtpMlp {
    /// The gate half of the doubled input projection.
    pub gate_proj: PackedMatrix,
    /// The value half of the doubled input projection.
    pub up_proj: PackedMatrix,
    /// The projection back to the residual width.
    pub down_proj: PackedMatrix,
}

/// The head's single block: two normalizations, an attention and a
/// feed-forward.
#[derive(Debug)]
pub struct MtpLayer {
    /// Normalization in front of the attention.
    pub input_layernorm: HeadTensor,
    /// Normalization in front of the feed-forward.
    pub post_attention_layernorm: HeadTensor,
    /// The attention block.
    pub self_attn: MtpAttn,
    /// The feed-forward.
    pub mlp: MtpMlp,
}

/// Every tensor the draft head ships, bound against its container.
///
/// The mapping is owned here and the fields are descriptors into it, so a
/// binding is one value a loader can carry to the device rather than a graph of
/// borrows of a file somebody else has to keep open.
///
/// # What is not here
///
/// The vocabulary table and the output projection. The head shares both with
/// the target - `tie_word_embeddings` is true for this family and the head
/// ships neither - so a binding that carried them would be carrying the
/// target's, under the head's names, at the same shapes. Every launch would
/// succeed.
#[derive(Debug)]
pub struct MtpWeights {
    /// The container, mapped.
    shard: Shard,
    /// `mtp.fc.weight`: the pair-to-layer projection, `[hidden, 2 * hidden]`.
    ///
    /// BF16 and not packed, which is the container's own choice: it is the one
    /// matrix of the head that reads a concatenation, and the producer left it
    /// unquantized.
    pub fc: HeadTensor,
    /// `mtp.pre_fc_norm_embedding.weight`: the norm over the token embedding.
    pub pre_fc_norm_embedding: HeadTensor,
    /// `mtp.pre_fc_norm_hidden.weight`: the norm over the target hidden state.
    pub pre_fc_norm_hidden: HeadTensor,
    /// `mtp.norm.weight`: the norm in front of the shared vocabulary head.
    pub norm: HeadTensor,
    /// The head's single block.
    pub layer: MtpLayer,
    /// The codec the manifest stated, which every triple here was read under.
    pub quant: AffineQuant,
}

impl MtpWeights {
    /// Binds every tensor of the head the manifest names.
    ///
    /// Opened through [`Shard::open_untagged`], because `optiq/mtp.safetensors`
    /// carries no `__metadata__` object at all and the tagged opener refuses a
    /// header without a producer tag. What says whose rules the file was
    /// written under is the document that named it.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the container cannot be mapped or its
    /// header does not parse, [`CoreError::ConfigMissing`] naming the first
    /// tensor the container does not hold, and [`CoreError::ShapeMismatch`]
    /// when a triple's three arrays do not agree about its group size or the
    /// container holds a tensor count other than the 29 this head is.
    pub fn bind(manifest: &MtpManifest) -> CoreResult<Self> {
        let (shard, raw) = Shard::open_untagged(&manifest.file).map_err(read_failure)?;
        let mut table = HashMap::with_capacity(raw.len());
        for (name, tensor) in raw {
            let dtype = tensor.parsed_dtype(&name).map_err(read_failure)?;
            table.insert(
                name.clone(),
                HeadTensor {
                    name: name.into_boxed_str(),
                    dtype,
                    shape: tensor.shape,
                    span: tensor.data_offsets,
                },
            );
        }
        let found = u32::try_from(table.len()).unwrap_or(u32::MAX);
        if found != HEAD_TENSORS {
            return Err(CoreError::ShapeMismatch {
                field: "draft head tensor count",
                expected: u64::from(HEAD_TENSORS),
                got: u64::from(found),
            });
        }
        let quant = manifest.quant;
        let bound = Self {
            shard,
            fc: take(&mut table, "mtp.fc.weight")?,
            pre_fc_norm_embedding: take(&mut table, "mtp.pre_fc_norm_embedding.weight")?,
            pre_fc_norm_hidden: take(&mut table, "mtp.pre_fc_norm_hidden.weight")?,
            norm: take(&mut table, "mtp.norm.weight")?,
            layer: MtpLayer {
                input_layernorm: take(&mut table, "mtp.layers.0.input_layernorm.weight")?,
                post_attention_layernorm: take(
                    &mut table,
                    "mtp.layers.0.post_attention_layernorm.weight",
                )?,
                self_attn: MtpAttn {
                    q_proj: triple(&mut table, "mtp.layers.0.self_attn.q_proj", quant)?,
                    k_proj: triple(&mut table, "mtp.layers.0.self_attn.k_proj", quant)?,
                    v_proj: triple(&mut table, "mtp.layers.0.self_attn.v_proj", quant)?,
                    o_proj: triple(&mut table, "mtp.layers.0.self_attn.o_proj", quant)?,
                    q_norm: take(&mut table, "mtp.layers.0.self_attn.q_norm.weight")?,
                    k_norm: take(&mut table, "mtp.layers.0.self_attn.k_norm.weight")?,
                },
                mlp: MtpMlp {
                    gate_proj: triple(&mut table, "mtp.layers.0.mlp.gate_proj", quant)?,
                    up_proj: triple(&mut table, "mtp.layers.0.mlp.up_proj", quant)?,
                    down_proj: triple(&mut table, "mtp.layers.0.mlp.down_proj", quant)?,
                },
            },
            quant,
        };
        // Every tensor the container held was claimed by a field. The count
        // above says the container holds 29; this says the 29 it holds are the
        // 29 this head is, rather than 29 of somebody else's under names that
        // happen to be absent here.
        if !table.is_empty() {
            let mut left: Vec<&str> = table.keys().map(String::as_str).collect();
            left.sort_unstable();
            return Err(CoreError::InvalidValue {
                key: "draft head tensors nothing binds",
                value: left.join(", "),
            });
        }
        tracing::info!(
            file = %bound.shard.path().display(),
            tensors = HEAD_TENSORS,
            bits = quant.bits,
            group_size = quant.group_size,
            "draft head bound"
        );
        Ok(bound)
    }

    /// Tensors the head ships, which is the census its own document states.
    #[must_use]
    pub const fn tensor_count(&self) -> u32 {
        HEAD_TENSORS
    }

    /// The head's seven packed matrices, in binding order.
    #[must_use]
    pub fn packed(&self) -> [&PackedMatrix; 7] {
        [
            &self.layer.self_attn.q_proj,
            &self.layer.self_attn.k_proj,
            &self.layer.self_attn.v_proj,
            &self.layer.self_attn.o_proj,
            &self.layer.mlp.gate_proj,
            &self.layer.mlp.up_proj,
            &self.layer.mlp.down_proj,
        ]
    }

    /// The container bytes of one bound tensor.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the span is not inside the mapping, which a
    /// header that parsed cannot produce and which is checked anyway because
    /// the alternative is a slice into somebody else's memory.
    pub fn bytes(&self, tensor: &HeadTensor) -> CoreResult<&[u8]> {
        self.shard
            .payload(tensor.span[0], tensor.span[1])
            .map_err(read_failure)
    }
}

/// Tensors a head of this family ships, which its own document also states.
const HEAD_TENSORS: u32 = 29;

/// Words one 32-bit packed word holds at `bits` per weight.
fn weights_per_word(bits: u32) -> CoreResult<u64> {
    match bits {
        4 | 8 => Ok(u64::from(32 / bits)),
        other => Err(CoreError::InvalidValue {
            key: "draft head quantization bits",
            value: format!("{other}, and only 4 and 8 are packed formats"),
        }),
    }
}

/// Claims one tensor by name, or refuses naming it.
fn take(table: &mut HashMap<String, HeadTensor>, name: &str) -> CoreResult<HeadTensor> {
    table.remove(name).ok_or_else(|| CoreError::ConfigMissing {
        key: name.to_owned(),
    })
}

/// Claims one affine triple by its shared prefix and derives its layout.
///
/// The layout is *derived and then checked* rather than taken from the
/// manifest: the manifest states the codec the producer wrote the head at, and
/// the container states how many words and how many groups each matrix
/// actually has. Those two agreeing is the whole of "this triple decodes", and
/// where they disagree the container wins the shape and the disagreement is the
/// refusal.
fn triple(
    table: &mut HashMap<String, HeadTensor>,
    prefix: &str,
    quant: AffineQuant,
) -> CoreResult<PackedMatrix> {
    let words = take(table, &format!("{prefix}.weight"))?;
    let scales = take(table, &format!("{prefix}.scales"))?;
    let biases = take(table, &format!("{prefix}.biases"))?;
    let out_rows = words.axis(0)?;
    let packed_cols = words.axis(1)?;
    let in_dim = packed_cols.saturating_mul(weights_per_word(quant.bits)?);
    let groups = scales.axis(1)?;
    // The group size the container implies, against the one the document
    // states. A scales row that does not divide the input width evenly is the
    // failure this refuses: it addresses a scale that exists, for a group that
    // is not the one the weights were written in.
    if groups == 0 || in_dim % groups != 0 {
        return Err(CoreError::ShapeMismatch {
            field: "draft head group count",
            expected: in_dim,
            got: groups,
        });
    }
    let implied = in_dim / groups;
    if implied != u64::from(quant.group_size) {
        return Err(CoreError::ShapeMismatch {
            field: "draft head group size",
            expected: u64::from(quant.group_size),
            got: implied,
        });
    }
    if scales.shape() != biases.shape() {
        return Err(CoreError::ShapeMismatch {
            field: "draft head bias rows",
            expected: groups,
            got: biases.axis(1).unwrap_or(0),
        });
    }
    if scales.axis(0)? != out_rows {
        return Err(CoreError::ShapeMismatch {
            field: "draft head scale rows",
            expected: out_rows,
            got: scales.axis(0)?,
        });
    }
    let layout = MlxAffineLayout::new(
        usize::try_from(out_rows).unwrap_or(usize::MAX),
        usize::try_from(in_dim).unwrap_or(usize::MAX),
        usize::try_from(quant.group_size).unwrap_or(usize::MAX),
        quant.bits,
    );
    Ok(PackedMatrix {
        words,
        scales,
        biases,
        layout,
    })
}

/// Presents a container failure in the core vocabulary.
fn read_failure(source: shiso_modelfile::GgufError) -> CoreError {
    CoreError::Backend {
        backend: "qwen35-mtp",
        op: "load",
        source: Box::new(source),
    }
}

/// Makes a bound head resident, casting what no CUDA kernel reads.
///
/// # The two casts, and why they are casts rather than kernels
///
/// **There is no BF16 matvec on CUDA in this tree.** A grep of `kernels/cuda/`
/// for a `bf16` entry returns nothing; the only BF16 entries are two Metal
/// expert stages. The head's `mtp.fc` is BF16, and the six norm vectors are
/// BF16 where every norm the engine's kernels read is `f32`.
///
/// So `fc` is cast to `f16` at load and the norms to `f32`. The `f16` choice is
/// deliberate and it is the same ruling Ornith's head takes: a draft head is
/// graded by its **acceptance rate** and not by its logits, so the half-ulp a
/// BF16-to-F16 round trip costs on a weight is paid in a guess the target
/// occasionally rejects, never in an answer. Cost if that is wrong: a
/// measurably lower acceptance rate than the same head run in BF16, which is
/// visible in Task 3.6's numbers and nowhere else - the target's logits are
/// untouched either way.
///
/// The norms go to `f32` rather than `f16` because their kernels take `f32` and
/// a norm weight is `hidden` floats: 10 KB against the head's 86.7 MB.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when a tensor is in a format this cast does not
/// cover, [`CoreError::ShapeMismatch`] when a width does not fit a launch, and
/// whatever the device reports for an allocation or a transfer.
pub fn resident(
    head: &MtpWeights,
    backend: &Arc<dyn Backend>,
    hidden: u32,
) -> CoreResult<DraftParts> {
    let attn = Attn {
        q: packed(head, backend, &head.layer.self_attn.q_proj)?,
        k: Some(packed(head, backend, &head.layer.self_attn.k_proj)?),
        v: Some(packed(head, backend, &head.layer.self_attn.v_proj)?),
        output: packed(head, backend, &head.layer.self_attn.o_proj)?,
        q_norm: norm(head, backend, &head.layer.self_attn.q_norm)?,
        k_norm: Some(norm(head, backend, &head.layer.self_attn.k_norm)?),
    };
    let ffn = Ffn {
        gate: packed(head, backend, &head.layer.mlp.gate_proj)?,
        up: packed(head, backend, &head.layer.mlp.up_proj)?,
        down: packed(head, backend, &head.layer.mlp.down_proj)?,
    };
    let pair = u64::from(hidden).saturating_mul(2);
    let rows = head.fc.axis(0)?;
    let cols = head.fc.axis(1)?;
    if rows != u64::from(hidden) || cols != pair {
        return Err(CoreError::ShapeMismatch {
            field: "draft head fc shape",
            expected: pair,
            got: cols,
        });
    }
    Ok(DraftParts {
        fc: half_matrix(head, backend, &head.fc)?,
        norm_embed: norm(head, backend, &head.pre_fc_norm_embedding)?,
        norm_hidden: norm(head, backend, &head.pre_fc_norm_hidden)?,
        input_norm: norm(head, backend, &head.layer.input_layernorm)?,
        ffn_norm: norm(head, backend, &head.layer.post_attention_layernorm)?,
        output_norm: norm(head, backend, &head.norm)?,
        attn,
        ffn,
        // A DENSE head: this family's block is one feed-forward and its sidecar
        // ships no routed stacks. The routed arm belongs to `qwen35moe`, whose
        // head carries 256 experts beside a shared one.
        moe: None,
    })
}

/// Imports one affine triple exactly as the container wrote it.
fn packed(
    head: &MtpWeights,
    backend: &Arc<dyn Backend>,
    matrix: &PackedMatrix,
) -> CoreResult<Weight> {
    let dtype = match matrix.layout().bits {
        4 => DType::MlxAffine4Gs64,
        8 => DType::MlxAffine8Gs64,
        // `MtpWeights::bind` already refused any other width, through
        // `weights_per_word`. This arm is what keeps that true after an edit
        // to either of them rather than a `_ => unreachable!()`.
        _ => {
            return Err(CoreError::Unsupported {
                what: "a draft head packed at a bit width with no affine matvec",
                backend: "qwen35-mtp",
            });
        }
    };
    let [words, scales, biases] = matrix.arrays();
    Weight::import_affine_slices(
        backend,
        dtype,
        extent(matrix.layout().in_dim)?,
        extent(matrix.layout().out_rows)?,
        head.bytes(words)?,
        head.bytes(scales)?,
        head.bytes(biases)?,
    )
}

/// Imports one BF16 matrix as `f16`.
fn half_matrix(
    head: &MtpWeights,
    backend: &Arc<dyn Backend>,
    tensor: &HeadTensor,
) -> CoreResult<Weight> {
    if tensor.dtype() != StDType::Bf16 {
        return Err(CoreError::Unsupported {
            what: "a draft head matrix stored as anything but bfloat16",
            backend: "qwen35-mtp",
        });
    }
    let rows = u32::try_from(tensor.axis(0)?).unwrap_or(u32::MAX);
    let cols = u32::try_from(tensor.axis(1)?).unwrap_or(u32::MAX);
    let halves: Vec<u8> = head
        .bytes(tensor)?
        .chunks_exact(2)
        .flat_map(|word| f16::from_f32(bf16_of(word)).to_le_bytes().into_iter())
        .collect();
    let buffer = alloc(backend, u64::try_from(halves.len()).unwrap_or(0))?;
    backend.copy_h2d(buffer.view(), &halves)?;
    Ok(Weight::from_buffer(DType::F16, cols, rows, buffer))
}

/// Imports one BF16 norm vector as `f32`.
fn norm(
    head: &MtpWeights,
    backend: &Arc<dyn Backend>,
    tensor: &HeadTensor,
) -> CoreResult<DeviceBuffer> {
    if tensor.dtype() != StDType::Bf16 {
        return Err(CoreError::Unsupported {
            what: "a draft head norm stored as anything but bfloat16",
            backend: "qwen35-mtp",
        });
    }
    let floats: Vec<u8> = head
        .bytes(tensor)?
        .chunks_exact(2)
        .flat_map(|word| bf16_of(word).to_le_bytes().into_iter())
        .collect();
    let buffer = alloc(backend, u64::try_from(floats.len()).unwrap_or(0))?;
    backend.copy_h2d(buffer.view(), &floats)?;
    Ok(buffer)
}

/// The `f32` a little-endian BF16 word stands for.
///
/// Exact in both directions: BF16 is the top sixteen bits of an `f32`, so the
/// widening is a shift and no rounding happens at all.
fn bf16_of(word: &[u8]) -> f32 {
    let bits = u32::from(u16::from_le_bytes([
        word.first().copied().unwrap_or(0),
        word.get(1).copied().unwrap_or(0),
    ])) << 16;
    f32::from_bits(bits)
}

/// Narrows one container dimension to the width a launch counts in.
fn extent(dim: usize) -> CoreResult<u32> {
    u32::try_from(dim).map_err(|_| CoreError::ShapeMismatch {
        field: "draft head tensor extent",
        expected: u64::from(u32::MAX),
        got: u64::try_from(dim).unwrap_or(u64::MAX),
    })
}

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;

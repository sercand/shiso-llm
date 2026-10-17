//! The `modelopt` mixed-precision export of this family, and the import every
//! Qwen3.5 family shares for it.
//!
//! `modelopt` writes one directory layout for the dense hybrid and for the
//! routed one: `model.language_model.*` tensors under the multimodal wrapper,
//! each stored plain `BF16`, FP8 E4M3 with one `f32`, or NVFP4 under three
//! levels of scale, with `hf_quant_config.json` beside the shards saying which.
//! [`ModeloptFile`] reads the scheme off the bytes; everything here turns a
//! described tensor into a device weight. The routed family's loader
//! (`crate::qwen35moe`) calls the shared half for its recurrent and attention
//! slots, and this module's own `load` adds the dense feed-forward around it.
//!
//! # The norms are centred in this export, and nowhere else
//!
//! The reference applies every norm of this family except the recurrent
//! block's gated one as `x * (1 + w)`, and stores `w`. The MLX and GGUF
//! converters fold the `1` in, which is why the dense family's other
//! containers - and the engine's own norm, which multiplies by the stored
//! value - never see it. This export is unconverted: the 27B checkpoint's
//! input norms average -0.04 and its gated recurrent norm 0.87. Reading the
//! former as stored scales every residual read by about 0.04 instead of 1, and
//! that is `centred_norm`.

use std::path::Path;
use std::sync::Arc;

use half::{bf16, f16};
use shiso_core::backend::Backend;
use shiso_core::buffer::DeviceBuffer;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::LayerIdx;
use shiso_core::{DType, alloc};
use shiso_engine::attn::Attn;
use shiso_engine::ffn::Ffn;
use shiso_engine::gdn::Gdn;
use shiso_engine::layers::{Block, LayerWeights, Weights};
use shiso_engine::projection::Weight;
use shiso_engine::shape::{HybridShape, LayerKind};
use shiso_modelfile::modelopt::{ModeloptFile, ModeloptTensor, Stored};
use shiso_modelfile::st_header::StDType;
use shiso_modelfile::{GdnHeads, HeadAxis, HeadPermute};

use crate::qwen35moe::QuantConfig;
use crate::qwen35moe::binding::scheme_of;

/// Prefix every language-model tensor carries under the multimodal wrapper.
pub const TEXT_PREFIX: &str = "model.language_model.";

/// The output projection, which sits outside the wrapper.
pub const LM_HEAD: &str = "lm_head";

/// The prefixes a language-model load asks the container for.
///
/// The vision tower is described by its own loader and the draft head by
/// nothing, so neither is in this list and neither costs a description.
pub const TEXT_ROOTS: [&str; 2] = [TEXT_PREFIX, LM_HEAD];

/// The document beside the shards that marks a directory as this export.
const QUANT_DOCUMENT: &str = "hf_quant_config.json";

/// Whether `path` is a `modelopt` export directory.
///
/// The quantization document is the test rather than the shard headers: it is
/// what `modelopt` writes and what an MLX or plain export of the same family
/// does not, and it is what `load` needs before it reads a weight.
#[must_use]
pub fn is_modelopt_dir(path: &Path) -> bool {
    path.is_dir() && path.join(QUANT_DOCUMENT).is_file()
}

/// The recurrent block's head structure, which the value-head reordering is
/// resolved against.
pub(crate) fn gdn_heads(shape: &HybridShape) -> GdnHeads {
    GdnHeads::new(
        shape.gdn.v_heads as usize,
        shape.gdn.qk_heads as usize,
        shape.gdn.head_dim as usize,
        shape.gdn.head_dim as usize,
    )
}

/// The format the vocabulary table is widened to.
///
/// `f16` rather than `f32`: `shiso_engine::layers::EmbedKernel` has a gather
/// for both, the table is 248,320 rows of 2,048, and the wider one would cost
/// a gigabyte of device memory for precision a gather does not use - it reads
/// a row and hands it on.
pub(crate) const EMBED_DTYPE: DType = DType::F16;

/// Wraps a container failure as an engine one.
pub(crate) fn open_failure(err: &shiso_modelfile::GgufError) -> CoreError {
    CoreError::InvalidValue {
        key: "checkpoint",
        value: err.to_string(),
    }
}

/// The tensor called `name`, or the refusal naming it.
pub(crate) fn tensor<'a>(file: &'a ModeloptFile, name: &str) -> CoreResult<&'a ModeloptTensor> {
    file.get(name).ok_or_else(|| CoreError::InvalidValue {
        key: "missing tensor",
        value: name.to_string(),
    })
}

/// A tensor's bytes, or the refusal.
pub(crate) fn bytes<'a>(file: &'a ModeloptFile, t: &ModeloptTensor) -> CoreResult<&'a [u8]> {
    file.weight_bytes(t).map_err(|err| open_failure(&err))
}

/// Widens a `BF16` array to `f32` values.
///
/// A copy, and it has to be: `bf16` is the top sixteen bits of an `f32`, so
/// the widening is a shift, but nothing in this build dispatches a `BF16`
/// projection and a kernel reading these bytes as `f32` would read two values
/// as one.
///
/// Values rather than bytes because every caller either reorders them before
/// the upload or turns them into bytes itself, and a byte-returning twin would
/// be a second place for the widening to be written.
pub(crate) fn to_f32_values(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)).to_f32())
        .collect()
}

/// Widens a `BF16` array to `f16` bytes.
///
/// Both are sixteen bits and neither is the other: `bf16` trades seven mantissa
/// bits for three exponent bits. The conversion goes through `f32`, which is
/// exact in that direction, and rounds once on the way down.
pub(crate) fn to_f16(raw: &[u8]) -> Vec<u8> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .flat_map(|pair| {
            f16::from_f32(bf16::from_bits(u16::from_le_bytes(*pair)).to_f32())
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

/// A `BF16` vector - a norm, a bias - as an `f32` device buffer.
pub(crate) fn vector(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    name: &str,
) -> CoreResult<DeviceBuffer> {
    reordered_vector(backend, file, name, None)
}

/// The same, reordering the value heads first for a vector that carries them.
pub(crate) fn reordered_vector(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    name: &str,
    permute: Option<HeadPermute>,
) -> CoreResult<DeviceBuffer> {
    let t = tensor(file, name)?;
    let mut values = to_f32_values(bytes(file, t)?);
    if let Some(permute) = permute {
        permute
            .apply(&mut values, name)
            .map_err(|err| open_failure(&err))?;
    }
    let host: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload(backend, &host)
}

/// The recurrent decay driver, negated and exponentiated on the way in.
///
/// `A_log` is stored as the logarithm and the kernel wants `-exp(A_log)`. Doing
/// it here rather than in the kernel is what the dense family does through its
/// loader's `ValueOp::NegExp`, and it is the same arithmetic: one `exp` per
/// value head, once per load, against one per launch.
pub(crate) fn neg_exp(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    name: &str,
    permute: Option<HeadPermute>,
) -> CoreResult<DeviceBuffer> {
    let t = tensor(file, name)?;
    let mut values: Vec<f32> = to_f32_values(bytes(file, t)?)
        .iter()
        .map(|v| -v.exp())
        .collect();
    if let Some(permute) = permute {
        permute
            .apply(&mut values, name)
            .map_err(|err| open_failure(&err))?;
    }
    let host: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload(backend, &host)
}

/// Allocates a device buffer and fills it.
pub(crate) fn upload(backend: &Arc<dyn Backend>, host: &[u8]) -> CoreResult<DeviceBuffer> {
    let buffer = alloc(backend, u64::try_from(host.len()).unwrap_or(0))?;
    backend.copy_h2d(buffer.view(), host)?;
    Ok(buffer)
}

/// One projection, imported under whichever scheme its bytes are stored in.
///
/// The scheme comes from the container, which read it off the header. A plain
/// tensor is widened to `f32`; an FP8 one keeps its byte per weight and carries
/// its two scalars; an NVFP4 one carries its group array and its per-tensor
/// `f32`.
pub(crate) fn weight(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    name: &str,
) -> CoreResult<Weight> {
    reordered_weight(backend, file, name, None)
}

/// The same, reordering the value heads first for a tensor that carries them.
///
/// The permutation is applied to whatever the container stores - the raw FP8
/// bytes, or the widened `f32` - because it moves whole heads and a head is a
/// whole number of elements either way.
pub(crate) fn reordered_weight(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    name: &str,
    permute: Option<HeadPermute>,
) -> CoreResult<Weight> {
    let t = tensor(file, name)?;
    let rows = u32::try_from(t.rows()).unwrap_or(u32::MAX);
    let cols = u32::try_from(t.cols()).unwrap_or(u32::MAX);
    match t.stored() {
        Stored::Plain { .. } => {
            let mut values = to_f32_values(bytes(file, t)?);
            if let Some(permute) = permute {
                permute
                    .apply(&mut values, name)
                    .map_err(|err| open_failure(&err))?;
            }
            let host: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let len = host.len();
            Weight::import(backend, DType::F32, cols, rows, Arc::new(host), 0..len)
        }
        Stored::Fp8 {
            weight_scale,
            input_scale,
        } => {
            let mut host = bytes(file, t)?.to_vec();
            if let Some(permute) = permute {
                permute
                    .apply(&mut host, name)
                    .map_err(|err| open_failure(&err))?;
            }
            let len = host.len();
            Weight::import_fp8(
                backend,
                cols,
                rows,
                Arc::new(host),
                0..len,
                weight_scale,
                input_scale,
            )
        }
        Stored::Nvfp4 { global } => {
            // No tensor of this family carries both a value-head axis and this
            // codec - the reordering is the recurrent block's and the recurrent
            // block is FP8 and BF16 - and permuting one would have to move the
            // packed nibbles and the group scales in step at two different
            // granularities. Refused rather than half-implemented.
            if permute.is_some() {
                return Err(CoreError::Unsupported {
                    what: "a value-head reordering over a packed nibble weight",
                    backend: "qwen35",
                });
            }
            let parts = file
                .nvfp4_parts(t)
                .map_err(|err| open_failure(&err))?
                .ok_or_else(|| CoreError::InvalidValue {
                    key: "nvfp4 tensor",
                    value: format!("{name} lost its scale array between describe and import"),
                })?;
            let packed = parts.weights.len();
            let mut host = parts.weights.to_vec();
            host.extend_from_slice(parts.scales);
            let total = host.len();
            Weight::import_nvfp4(
                backend,
                cols,
                rows,
                Arc::new(host),
                0..packed,
                packed..total,
                global,
            )
        }
        // No export of this family stores MXFP8; the dense stack has no tier
        // for it, and widening it here would be a serving path nobody asked for.
        Stored::Mxfp8 => Err(CoreError::Unsupported {
            what: "an MXFP8 weight in the dense family's modelopt export",
            backend: "qwen35",
        }),
    }
}

/// The vocabulary table, widened to `f16`.
pub(crate) fn embedding(backend: &Arc<dyn Backend>, file: &ModeloptFile) -> CoreResult<Weight> {
    let t = tensor(file, &format!("{TEXT_PREFIX}embed_tokens"))?;
    let host = to_f16(bytes(file, t)?);
    let len = host.len();
    Weight::import(
        backend,
        EMBED_DTYPE,
        u32::try_from(t.cols()).unwrap_or(u32::MAX),
        u32::try_from(t.rows()).unwrap_or(u32::MAX),
        Arc::new(host),
        0..len,
    )
}

/// The value-head reordering one tensor needs, resolved against its own shape.
///
/// Not an `Option`. Every caller below has already decided that its tensor
/// carries the axis, so "no reordering" is not one of this function's answers:
/// it is the `None` that the one tensor without an axis passes to
/// [`reordered_weight`] directly. A `Result<Option<..>>` here would have made
/// every call site read as though the question were still open.
pub(crate) fn axis_for(
    file: &ModeloptFile,
    heads: GdnHeads,
    name: &str,
    axis: HeadAxis,
) -> CoreResult<HeadPermute> {
    let t = tensor(file, name)?;
    HeadPermute::new(
        axis,
        heads,
        usize::try_from(t.rows()).unwrap_or(usize::MAX),
        usize::try_from(t.cols()).unwrap_or(usize::MAX),
        name,
    )
    .map_err(|err| open_failure(&err))
}

/// One recurrent slot's weights, with its value heads reordered.
///
/// **Eight of the nine carry that axis**, and not the same one: see this
/// module's own note. The list below is the same list
/// `crates/shiso-models/src/qwen35/st_names.rs` carries for the sibling
/// container. It is written out here rather than shared because the two reach
/// their tensors through different name tables; what IS shared is
/// `HeadPermute`, which is the part that could be got wrong quietly.
pub(crate) fn import_gdn(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    slot: usize,
    heads: GdnHeads,
) -> CoreResult<Gdn> {
    let at = |role: &str| format!("{TEXT_PREFIX}layers.{slot}.linear_attn.{role}");
    let (qkv, gate, alpha) = (at("in_proj_qkv"), at("in_proj_z"), at("in_proj_a"));
    let (beta, out, conv) = (at("in_proj_b"), at("out_proj"), at("conv1d"));
    let (a_log, dt) = (at("A_log"), at("dt_bias"));
    Ok(Gdn {
        // The fused projection's leading `[q | k]` run is not indexed by value
        // head; only the trailing `v` run is.
        qkv: reordered_weight(
            backend,
            file,
            &qkv,
            Some(axis_for(file, heads, &qkv, HeadAxis::QkvTailRows)?),
        )?,
        gate: reordered_weight(
            backend,
            file,
            &gate,
            Some(axis_for(file, heads, &gate, HeadAxis::ValueRows)?),
        )?,
        alpha: reordered_weight(
            backend,
            file,
            &alpha,
            Some(axis_for(file, heads, &alpha, HeadAxis::PerHeadRows)?),
        )?,
        beta: reordered_weight(
            backend,
            file,
            &beta,
            Some(axis_for(file, heads, &beta, HeadAxis::PerHeadRows)?),
        )?,
        // The output projection reads the block's value channels, so its axis
        // is its COLUMNS rather than its rows.
        out: reordered_weight(
            backend,
            file,
            &out,
            Some(axis_for(file, heads, &out, HeadAxis::ValueColumns)?),
        )?,
        conv1d: reordered_vector(
            backend,
            file,
            &conv,
            Some(axis_for(file, heads, &conv, HeadAxis::QkvTailRows)?),
        )?,
        a: neg_exp(
            backend,
            file,
            &a_log,
            Some(axis_for(file, heads, &a_log, HeadAxis::PerHeadRows)?),
        )?,
        dt_bias: reordered_vector(
            backend,
            file,
            &dt,
            Some(axis_for(file, heads, &dt, HeadAxis::PerHeadRows)?),
        )?,
        // The only one of the nine with no value-head axis: it is indexed by
        // head WIDTH, and every head shares it.
        norm: vector(backend, file, &at("norm"))?,
    })
}

/// One attention slot's weights.
///
/// The two head norms are centred; see [`centred_norm`].
pub(crate) fn import_attn(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    slot: usize,
) -> CoreResult<Attn> {
    let at = |role: &str| format!("{TEXT_PREFIX}layers.{slot}.self_attn.{role}");
    // Every attention slot of this family owns its cache, so all three of the
    // key half's `Option`s are filled.
    Ok(Attn {
        q: weight(backend, file, &at("q_proj"))?,
        k: Some(weight(backend, file, &at("k_proj"))?),
        v: Some(weight(backend, file, &at("v_proj"))?),
        output: weight(backend, file, &at("o_proj"))?,
        q_norm: Some(centred_norm(backend, file, &at("q_norm"))?),
        k_norm: Some(centred_norm(backend, file, &at("k_norm"))?),
    })
}

/// A norm the reference applies as `1 + w`, widened to `f32` with the `1`
/// folded in.
pub(crate) fn centred_norm(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    name: &str,
) -> CoreResult<DeviceBuffer> {
    let t = tensor(file, name)?;
    let host: Vec<u8> = to_f32_values(bytes(file, t)?)
        .iter()
        .flat_map(|v| (1.0 + v).to_le_bytes())
        .collect();
    upload(backend, &host)
}

/// What one dense tensor is expected to look like.
#[derive(Debug, Clone, Copy)]
enum Expect {
    /// A projection of `rows` outputs over `cols` inputs.
    Matrix { rows: u64, cols: u64 },
    /// A vector, or a tensor the kernels read as a flat run of this many values.
    Values(u64),
}

/// Every tensor the dense forward pass reads, with the shape it must have.
///
/// In the order the pass reads them, so a load walks each shard's mapping
/// roughly front to back.
fn dense_expected(shape: &HybridShape) -> CoreResult<Vec<(String, Expect)>> {
    let gdn = shape.gdn;
    let gqa = shape.attn.first().copied().ok_or(CoreError::InvalidValue {
        key: "shape profile",
        value: "no attention slot to measure the attention projections by".to_owned(),
    })?;
    let hidden = u64::from(shape.hidden);
    let inter = u64::from(shape.intermediate);
    let vocab = u64::from(shape.vocab);
    let v_width = u64::from(gdn.v_heads) * u64::from(gdn.head_dim);
    let qk_width = u64::from(gdn.qk_heads) * u64::from(gdn.head_dim);
    let qkv_out = 2 * qk_width + v_width;
    let v_heads = u64::from(gdn.v_heads);
    let q_factor: u64 = if shape.attn_output_gate { 2 } else { 1 };
    let query_width = u64::from(gqa.q_heads) * u64::from(gqa.head_dim);
    let kv_width = u64::from(gqa.kv_heads) * u64::from(gqa.head_dim);
    let attn_head = u64::from(gqa.head_dim);
    let matrix = |rows, cols| Expect::Matrix { rows, cols };

    let mut want = vec![(format!("{TEXT_PREFIX}embed_tokens"), matrix(vocab, hidden))];
    for (slot, kind) in shape.layers.iter().enumerate() {
        let at = format!("{TEXT_PREFIX}layers.{slot}");
        want.push((format!("{at}.input_layernorm"), Expect::Values(hidden)));
        want.push((
            format!("{at}.post_attention_layernorm"),
            Expect::Values(hidden),
        ));
        match kind {
            LayerKind::Gdn => {
                let gdn_at = |role: &str| format!("{at}.linear_attn.{role}");
                want.extend([
                    (gdn_at("in_proj_qkv"), matrix(qkv_out, hidden)),
                    (gdn_at("in_proj_z"), matrix(v_width, hidden)),
                    (gdn_at("in_proj_a"), matrix(v_heads, hidden)),
                    (gdn_at("in_proj_b"), matrix(v_heads, hidden)),
                    (gdn_at("out_proj"), matrix(hidden, v_width)),
                    (
                        gdn_at("conv1d"),
                        Expect::Values(qkv_out * u64::from(gdn.conv_kernel)),
                    ),
                    (gdn_at("A_log"), Expect::Values(v_heads)),
                    (gdn_at("dt_bias"), Expect::Values(v_heads)),
                    (gdn_at("norm"), Expect::Values(u64::from(gdn.head_dim))),
                ]);
            }
            LayerKind::Attn => {
                let attn_at = |role: &str| format!("{at}.self_attn.{role}");
                want.extend([
                    (attn_at("q_proj"), matrix(q_factor * query_width, hidden)),
                    (attn_at("k_proj"), matrix(kv_width, hidden)),
                    (attn_at("v_proj"), matrix(kv_width, hidden)),
                    (attn_at("o_proj"), matrix(hidden, query_width)),
                    (attn_at("q_norm"), Expect::Values(attn_head)),
                    (attn_at("k_norm"), Expect::Values(attn_head)),
                ]);
            }
            // This family's shape holds neither; its config never states one.
            LayerKind::Ssm | LayerKind::ShortConv => {
                return Err(CoreError::Unsupported {
                    what: "a state-space or short-convolution slot in this family",
                    backend: "qwen35",
                });
            }
        }
        want.extend([
            (format!("{at}.mlp.gate_proj"), matrix(inter, hidden)),
            (format!("{at}.mlp.up_proj"), matrix(inter, hidden)),
            (format!("{at}.mlp.down_proj"), matrix(hidden, inter)),
        ]);
    }
    want.push((format!("{TEXT_PREFIX}norm"), Expect::Values(hidden)));
    want.push((LM_HEAD.to_owned(), matrix(vocab, hidden)));
    Ok(want)
}

/// Checks an open dense export against the geometry and its own quantization
/// document, before a weight is read.
///
/// Names in both directions, every shape, and every storage scheme against the
/// one the document declares - the same three checks the routed family's
/// `check_export` makes, for the same reason: each is a wrong answer rather
/// than a crash when it goes unchecked.
fn check_dense_export(
    file: &ModeloptFile,
    quant: &QuantConfig,
    shape: &HybridShape,
) -> CoreResult<()> {
    let want = dense_expected(shape)?;
    for (name, expect) in &want {
        let t = tensor(file, name)?;
        let fits = match *expect {
            Expect::Matrix { rows, cols } => t.rows() == rows && t.cols() == cols,
            Expect::Values(count) => t.elements() == count,
        };
        if !fits {
            return Err(CoreError::InvalidValue {
                key: "tensor shape",
                value: format!(
                    "{name} is {:?} and this geometry needs {expect:?}",
                    t.declared()
                ),
            });
        }
        let stored = t.stored();
        let claimed = scheme_of(quant.algo_for(name));
        if claimed != stored.kind() {
            return Err(CoreError::InvalidValue {
                key: "storage scheme",
                value: format!("{name} is stored {} and declared {claimed}", stored.kind()),
            });
        }
        // Every plain tensor is widened from `BF16` below, and a `F32` or
        // `F16` one read that way would be two values read as one.
        if let Stored::Plain { dtype } = stored
            && dtype != StDType::Bf16
        {
            return Err(CoreError::InvalidValue {
                key: "storage scheme",
                value: format!("{name} is plain {dtype:?}, and this loader widens BF16 only"),
            });
        }
    }
    let found = file.tensors().len();
    if found != want.len() {
        return Err(CoreError::ShapeMismatch {
            field: "tensor_count",
            expected: u64::try_from(want.len()).unwrap_or(u64::MAX),
            got: u64::try_from(found).unwrap_or(u64::MAX),
        });
    }
    Ok(())
}

/// Opens a dense `modelopt` export and moves every tensor to the device.
///
/// Returns the weights and the format the layer-zero feed-forward gate was
/// imported as, which is what the checkpoint tier's width is read from.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] and [`CoreError::InvalidValue`] for a
/// quantization document that is absent or names a codec with no decoder;
/// [`CoreError::InvalidValue`] naming the first tensor that is missing,
/// misshaped or stored under a scheme its document does not claim;
/// [`CoreError::ShapeMismatch`] when the export carries tensors the pass does
/// not read; and [`CoreError::Backend`] when a device transfer fails.
pub(crate) fn load(
    backend: &Arc<dyn Backend>,
    path: &Path,
    shape: &HybridShape,
) -> CoreResult<(Weights, DType)> {
    let quant = QuantConfig::from_file(&path.join(QUANT_DOCUMENT))?;
    let file = ModeloptFile::open(path, &TEXT_ROOTS).map_err(|err| open_failure(&err))?;
    check_dense_export(&file, &quant, shape)?;
    let weights = import_dense(backend, &file, shape)?;
    let ffn_gate = weights
        .layers
        .first()
        .and_then(|layer| layer.ffn.as_ref())
        .and_then(|ffn| ffn.gate.as_ref())
        .map_or(EMBED_DTYPE, |gate| gate.dtype);
    Ok((weights, ffn_gate))
}

/// Moves every dense tensor to the device, in the order the pass reads them.
fn import_dense(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    shape: &HybridShape,
) -> CoreResult<Weights> {
    let heads = gdn_heads(shape);
    let mut gdn_rank = 0u32;
    let mut attn_rank = 0u32;
    let mut layers = Vec::with_capacity(shape.layers.len());
    for (slot, kind) in shape.layers.iter().enumerate() {
        let il = LayerIdx::new(u32::try_from(slot).unwrap_or(u32::MAX));
        let kv_rank = shape.kv_owner_rank(il);
        let (block, ordinal) = match kind {
            LayerKind::Gdn => {
                let rank = gdn_rank;
                gdn_rank += 1;
                (Block::Gdn(import_gdn(backend, file, slot, heads)?), rank)
            }
            LayerKind::Attn => {
                let rank = attn_rank;
                attn_rank += 1;
                let attn = import_attn(backend, file, slot)?;
                (Block::Attn(attn), rank)
            }
            // This family's shape holds neither; its config never states one.
            LayerKind::Ssm | LayerKind::ShortConv => {
                return Err(CoreError::Unsupported {
                    what: "a state-space or short-convolution slot in this family",
                    backend: "qwen35",
                });
            }
        };
        let at = |role: &str| format!("{TEXT_PREFIX}layers.{slot}.{role}");
        layers.push(LayerWeights {
            input_norm: centred_norm(backend, file, &at("input_layernorm"))?,
            // The pre-only scheme: nothing normalizes a block's output and the
            // residual sum carries no scalar.
            post_attn_norm: None,
            ffn_norm: Some(centred_norm(
                backend,
                file,
                &at("post_attention_layernorm"),
            )?),
            post_ffn_norm: None,
            ffn: Some(Ffn {
                gate: Some(weight(backend, file, &at("mlp.gate_proj"))?),
                up: weight(backend, file, &at("mlp.up_proj"))?,
                down: weight(backend, file, &at("mlp.down_proj"))?,
            }),
            // A dense feed-forward: this family's slots have no router.
            moe: None,
            block,
            ordinal,
            kv_rank,
            layer_scalar: 1.0,
            ple: None,
        });
        tracing::debug!(slot, kind = ?kind, "imported a layer slot");
    }
    Ok(Weights {
        token_embd: embedding(backend, file)?,
        output_norm: centred_norm(backend, file, &format!("{TEXT_PREFIX}norm"))?,
        lm_head: Some(weight(backend, file, LM_HEAD)?),
        ple_model_proj: None,
        ple_proj_norm: None,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::{Expect, LM_HEAD, TEXT_PREFIX, dense_expected};
    use shiso_engine::shape::{
        Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
    };

    /// Three recurrent slots and one attention slot, at small widths.
    fn shape() -> HybridShape {
        HybridShape {
            layers: vec![
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Attn,
            ],
            attn: vec![AttnLayer::new(4, 2, 64, 64, 64, 1e7)],
            hidden: 256,
            vocab: 128,
            intermediate: 512,
            gdn: GdnShape::new(6, 2, 32, 4),
            attn_output_gate: true,
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
            short_conv: None,
            no_ffn: Vec::new(),
        }
    }

    /// The count is what makes the export check two-directional, so it has to
    /// be the per-slot sums: two norms and three feed-forward projections per
    /// slot, nine recurrent tensors or six attention ones, and three outside.
    #[test]
    fn the_dense_list_counts_every_slot_s_roles() {
        let want = dense_expected(&shape()).expect("one attention geometry");
        assert_eq!(want.len(), 4 * 5 + 3 * 9 + 6 + 3);
        assert_eq!(
            want.first().map(|(n, _)| n.as_str()),
            Some("model.language_model.embed_tokens")
        );
        assert_eq!(want.last().map(|(n, _)| n.as_str()), Some(LM_HEAD));
    }

    /// The widths that separate a right read from a plausible one: the fused
    /// recurrent projection carries `[q | k | v]`, and the gated query
    /// projection is twice the heads it reads.
    #[test]
    fn the_fused_and_gated_projections_have_their_doubled_widths() {
        let want = dense_expected(&shape()).expect("one attention geometry");
        let find = |name: String| {
            want.iter()
                .find(|(n, _)| *n == name)
                .map(|(_, e)| *e)
                .expect("named")
        };
        match find(format!("{TEXT_PREFIX}layers.0.linear_attn.in_proj_qkv")) {
            Expect::Matrix { rows, cols } => assert_eq!((rows, cols), (2 * 2 * 32 + 6 * 32, 256)),
            Expect::Values(_) => panic!("a projection is a matrix"),
        }
        match find(format!("{TEXT_PREFIX}layers.3.self_attn.q_proj")) {
            Expect::Matrix { rows, cols } => assert_eq!((rows, cols), (2 * 4 * 64, 256)),
            Expect::Values(_) => panic!("a projection is a matrix"),
        }
    }
}

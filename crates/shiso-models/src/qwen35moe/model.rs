//! The model type: a checkpoint of this family, loaded and answering the shared
//! contract.
//!
//! Everything here is identity. Which directory the weights came from, what its
//! tensors are called, which token ends a turn, how each of them is stored -
//! those are facts about this family, and they are the only facts this module
//! holds. The stack that runs them is [`shiso_engine::HybridStack`], the same
//! one the dense hybrid runs on, and the blocks are dispatched by
//! [`crate::qwen35::Qwen35Kernels`] rather than by a copy of it.
//!
//! # Three storage schemes in one load, and none of them is a branch downstream
//!
//! A tensor of this checkpoint is plain `BF16`, FP8 E4M3 with one `f32`, or
//! NVFP4 under three levels of scale, and which it is comes from the *bytes*
//! rather than from a table here - [`shiso_modelfile::modelopt::ModeloptFile`] read
//! it off the header, and `check_export` has already confirmed the bytes and
//! `hf_quant_config.json` agree about every one of the 31,333. So the import
//! below matches on what the container says and never asks the config again.
//!
//! **`BF16` is widened on the way in**, because no projection kernel in this
//! build reads it: the norms and the two recurrent drivers become `f32`, and
//! the vocabulary table becomes `f16` because that is what the row gather
//! reads and because doubling a 248,320-row table would cost a gigabyte to no
//! purpose.
//!
//! # The recurrent block's value heads are reordered on the way in
//!
//! `transformers` builds this block's query and key heads with
//! `repeat_interleave`, so checkpoint value head `k * group + m` reads key
//! head `k`. Every kernel in this tree reads key head `h % key_heads`. The two
//! disagree for any model with more value heads than key heads - this one has
//! 32 over 16 - and they disagree by a **permutation**, which leaves every
//! magnitude, every norm and every histogram exactly where it was.
//!
//! Eight tensors per recurrent slot carry that axis, and the axis each carries
//! is not the same: the fused projection and the convolution have it on their
//! trailing `v` run, the gate on all its rows, the two drivers one row per
//! head, and the output projection on its COLUMNS. `shiso_modelfile::HeadPermute`
//! resolves all four against a tensor's own shape, and this module applies it
//! to the stored bytes for the FP8 tensors and to the widened values for the
//! `BF16` ones - which is the same operation, because permuting rows of a
//! matrix whose elements are one byte each is permuting bytes.
//!
//! The sibling loader has done this since the family arrived
//! (`crates/shiso-models/src/qwen35/st_names.rs` carries the same eight
//! axes). **This one did not, and the model loaded, ran, stayed finite through
//! all forty slots with a residual growing smoothly from 1.04e-2 to 2.08e-1,
//! and answered fluent nonsense.**
//!
//! # The expert stacks are stacked here
//!
//! The checkpoint stores 256 separate tensors per projection per layer, each
//! with its own scale array and its own `f32`. The entries address an expert by
//! a stride, so the 768 tensors of a layer become three slabs - and they are
//! built by copying each expert straight from the mapping into its offset in a
//! device buffer, rather than by concatenating 134 MB on the host first.

use std::path::Path;
use std::sync::Arc;

use half::{bf16, f16};
use shiso_core::backend::Backend;
use shiso_core::buffer::DeviceBuffer;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, SessionId, TokenId};
use shiso_core::media::{MediaModel, VisionPrefill};
use shiso_core::model::{DecodeStep, Logits, Model, ModelInfo, StopTokens};
use shiso_core::{DType, alloc};
use shiso_engine::attn::Attn;
use shiso_engine::draft::DraftHead;
use shiso_engine::ffn::Ffn;
use shiso_engine::gdn::{Gdn, GdnGeometry};
use shiso_engine::layers::{Block, LayerWeights, Weights};
use shiso_engine::moe::{ActDtype, ExpertSlab, Moe};
use shiso_engine::projection::Weight;
use shiso_engine::shape::{HybridShape, LayerKind};
use shiso_engine::{EngineOptions, HybridStack, MAX_PREFILL_CHUNK};
use shiso_modelfile::modelopt::{ModeloptFile, ModeloptTensor, Stored};
use shiso_modelfile::{GdnHeads, HeadAxis, HeadPermute};

use crate::qwen35::Qwen35Kernels;
use crate::qwen35moe::binding::{TEXT_PREFIX, TEXT_ROOTS, check_export};
use crate::qwen35moe::config::Qwen35MoeConfig;

/// The recurrent block's head structure, which the value-head reordering is
/// resolved against.
fn gdn_heads(shape: &HybridShape) -> GdnHeads {
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
const EMBED_DTYPE: DType = DType::F16;

/// A loaded checkpoint of this family.
#[derive(Debug)]
pub struct Qwen35MoeModel {
    stack: HybridStack,
    info: ModelInfo,
}

impl Qwen35MoeModel {
    /// The stack this model forwards to.
    pub(crate) const fn stack(&self) -> &HybridStack {
        &self.stack
    }

    /// The stack this model forwards to, mutably.
    pub(crate) const fn stack_mut(&mut self) -> &mut HybridStack {
        &mut self.stack
    }

    /// Opens a checkpoint directory and puts it on the device.
    ///
    /// The export is checked against the geometry *and* against its own
    /// quantization document before a byte is imported: 31,333 names in both
    /// directions and every one of their storage schemes. A checkpoint short a
    /// single expert binds cleanly under any per-tensor lookup and answers
    /// wrongly, so it is refused here by that expert's name.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] naming the first tensor that is missing or
    /// stored under a scheme its own document does not claim,
    /// [`CoreError::ShapeMismatch`] for a width the kernels were not written
    /// for, and [`CoreError::Backend`] when a device allocation or transfer
    /// fails.
    pub fn load(
        backend: &Arc<dyn Backend>,
        path: &Path,
        config: &Qwen35MoeConfig,
        opts: EngineOptions,
    ) -> CoreResult<Self> {
        let file = ModeloptFile::open(path, &TEXT_ROOTS).map_err(|err| open_failure(&err))?;
        check_export(&file, config)?;
        let shape = config.shape();
        let weights = import_weights(backend, &file, shape)?;
        // The file is dropped once every weight is resident. Unlike the dense
        // family's load nothing here holds the mapping afterwards: each import
        // copies through the host, because the packed layouts are not what a
        // kernel reads in place.
        drop(file);

        let kernels = Qwen35Kernels::choose(backend.as_ref(), &GdnGeometry::new(&shape.gdn)?)?;
        let stack = HybridStack::new(
            backend,
            shape.clone(),
            weights,
            EMBED_DTYPE,
            Arc::new(kernels),
            // No per-layer embedding table: this family has none.
            None,
            opts,
        )?;
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: shape.vocab,
                hidden: shape.hidden,
                n_layers: shape.n_layers(),
                stop_tokens: StopTokens::new(&config.stop_tokens())?,
            },
            stack,
        })
    }
}

impl Qwen35MoeModel {
    /// Makes this family resident from a GGUF container.
    ///
    /// The sibling of [`Qwen35MoeModel::load`], which reads a `modelopt`
    /// safetensors export. The geometry comes from the file's own metadata
    /// rather than from a document beside it, so this takes no config: a GGUF
    /// that declares this architecture carries every width, and one that does
    /// not is refused by name.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the file declares another architecture,
    /// [`CoreError::ConfigMissing`] naming the key or tensor when one is
    /// absent, and whatever the device reports.
    pub fn load_gguf(
        backend: &Arc<dyn Backend>,
        path: &Path,
        opts: EngineOptions,
    ) -> CoreResult<Self> {
        Self::load_gguf_with_draft(backend, path, opts, None)
    }

    /// The same, with a multi-token-prediction head bound beside it.
    ///
    /// `head` is a DIRECTORY of the head's own - the shape this family's head
    /// is published in, and the shape the dense family already reads through
    /// `mtp_file`. It is ignored unless `opts.spec` asks for a speculative
    /// depth, so a build that binds one and then decodes plainly pays nothing
    /// for it.
    ///
    /// **The head has to be the target's own.** These weights predict the
    /// hidden states of the model they were split from, so a head from the base
    /// model over a finetune of it drafts fluently and is rejected. Nothing
    /// here can check that - the files do not say what they were trained
    /// against - and the acceptance rate is what reveals a mispairing.
    ///
    /// # Errors
    ///
    /// Whatever [`Qwen35MoeModel::load_gguf`] reports, plus
    /// [`CoreError::ConfigMissing`] naming the first tensor the head's
    /// container does not hold and [`CoreError::ShapeMismatch`] when one of its
    /// widths is not the target's.
    pub fn load_gguf_with_draft(
        backend: &Arc<dyn Backend>,
        path: &Path,
        opts: EngineOptions,
        head: Option<&Path>,
    ) -> CoreResult<Self> {
        let (file, config) = crate::qwen35moe::gguf::open_gguf(path)?;
        let shape = config.shape();
        // The table's OWN format, not this module's constant. The `modelopt`
        // path widens a `BF16` table to `f16` on the way in and so knows what
        // the gather will read; a GGUF stores it in whatever the quantizer
        // chose - `Q8_0` in `UD-Q4_K_XL` - and a gather dispatched for `f16`
        // over `Q8_0` rows would read 34-byte blocks as 16 halves and hand the
        // stack a plausible embedding of the wrong token.
        let embed = crate::qwen35moe::gguf::embed_dtype(&file)?;
        let weights = crate::qwen35moe::gguf::import_weights_gguf(backend, &file, shape)?;
        // Held until every weight is resident and dropped here. Unlike the
        // `modelopt` path this one imports most tensors IN PLACE out of the
        // mapping, so dropping it earlier would unmap bytes the device is
        // still reading.
        drop(file);

        let kernels = Qwen35Kernels::choose(backend.as_ref(), &GdnGeometry::new(&shape.gdn)?)?;
        let mut stack = HybridStack::new(
            backend,
            shape.clone(),
            weights,
            embed,
            Arc::new(kernels),
            None,
            opts,
        )?;
        if let Some(dir) = head.filter(|_| opts.spec.is_speculative()) {
            let bound = crate::qwen35moe::mtp::MtpHead::open(dir)?;
            let parts = bound.parts(backend, shape)?;
            // Sized for the widest pass the head ever runs, which is the prompt
            // chunk it is warmed over rather than the chain it drafts: a chain
            // is one row, and a head whose scratch was sized by the chain would
            // refuse to warm at all.
            let resident = DraftHead::new(
                backend,
                parts,
                stack.geometry(),
                opts.kv_pool_pages,
                MAX_PREFILL_CHUNK,
            )?;
            stack.bind_draft(resident)?;
            tracing::info!(dir = %dir.display(), "draft head bound");
        }
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: shape.vocab,
                hidden: shape.hidden,
                n_layers: shape.n_layers(),
                stop_tokens: StopTokens::new(&config.stop_tokens())?,
            },
            stack,
        })
    }
}

/// Wraps a container failure as an engine one.
fn open_failure(err: &shiso_modelfile::GgufError) -> CoreError {
    CoreError::InvalidValue {
        key: "checkpoint",
        value: err.to_string(),
    }
}

/// The tensor called `name`, or the refusal naming it.
fn tensor<'a>(file: &'a ModeloptFile, name: &str) -> CoreResult<&'a ModeloptTensor> {
    file.get(name).ok_or_else(|| CoreError::InvalidValue {
        key: "missing tensor",
        value: name.to_string(),
    })
}

/// A tensor's bytes, or the refusal.
fn bytes<'a>(file: &'a ModeloptFile, t: &ModeloptTensor) -> CoreResult<&'a [u8]> {
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
fn to_f32_values(raw: &[u8]) -> Vec<f32> {
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
fn to_f16(raw: &[u8]) -> Vec<u8> {
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
fn vector(backend: &Arc<dyn Backend>, file: &ModeloptFile, name: &str) -> CoreResult<DeviceBuffer> {
    reordered_vector(backend, file, name, None)
}

/// The same, reordering the value heads first for a vector that carries them.
fn reordered_vector(
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
fn neg_exp(
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
fn upload(backend: &Arc<dyn Backend>, host: &[u8]) -> CoreResult<DeviceBuffer> {
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
fn weight(backend: &Arc<dyn Backend>, file: &ModeloptFile, name: &str) -> CoreResult<Weight> {
    reordered_weight(backend, file, name, None)
}

/// The same, reordering the value heads first for a tensor that carries them.
///
/// The permutation is applied to whatever the container stores - the raw FP8
/// bytes, or the widened `f32` - because it moves whole heads and a head is a
/// whole number of elements either way.
fn reordered_weight(
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
                    backend: "qwen35moe",
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
    }
}

/// The vocabulary table, widened to `f16`.
fn embedding(backend: &Arc<dyn Backend>, file: &ModeloptFile) -> CoreResult<Weight> {
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

/// One projection of every expert in a layer, stacked into three buffers.
///
/// Each expert is copied straight from the mapping into its own offset. The
/// alternative - concatenating on the host first - would build a 134 MB `Vec`
/// per slab for bytes the device is about to hold anyway.
///
/// **The globals are an array**: every expert has its own `weight_scale_2` and
/// they differ, which is why the entries index one per expert rather than
/// taking a scalar.
fn expert_slab(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    slot: usize,
    role: &str,
    experts: u32,
) -> CoreResult<ExpertSlab> {
    let first = tensor(file, &expert_name(slot, 0, role))?;
    let packed = file
        .nvfp4_parts(first)
        .map_err(|err| open_failure(&err))?
        .ok_or_else(|| CoreError::InvalidValue {
            key: "expert storage",
            value: format!("slot {slot}'s {role} experts are not nvfp4"),
        })?;
    let weight_stride = u64::try_from(packed.weights.len()).unwrap_or(0);
    let scale_stride = u64::try_from(packed.scales.len()).unwrap_or(0);
    let count = u64::from(experts);
    let weights = alloc(backend, weight_stride.saturating_mul(count))?;
    let scales = alloc(backend, scale_stride.saturating_mul(count))?;
    let mut globals = Vec::with_capacity(experts as usize * 4);

    for expert in 0..experts {
        let name = expert_name(slot, expert, role);
        let t = tensor(file, &name)?;
        let parts = file
            .nvfp4_parts(t)
            .map_err(|err| open_failure(&err))?
            .ok_or_else(|| CoreError::InvalidValue {
                key: "expert storage",
                value: format!("{name} is not nvfp4"),
            })?;
        // Every expert of a projection has the same shape, and a slab built
        // from experts that do not would have each one at the wrong offset.
        if u64::try_from(parts.weights.len()).unwrap_or(0) != weight_stride
            || u64::try_from(parts.scales.len()).unwrap_or(0) != scale_stride
        {
            return Err(CoreError::ShapeMismatch {
                field: "expert slab stride",
                expected: weight_stride,
                got: u64::try_from(parts.weights.len()).unwrap_or(0),
            });
        }
        let at = u64::from(expert);
        backend.copy_h2d(
            weights.slice(at.saturating_mul(weight_stride), weight_stride)?,
            parts.weights,
        )?;
        backend.copy_h2d(
            scales.slice(at.saturating_mul(scale_stride), scale_stride)?,
            parts.scales,
        )?;
        globals.extend_from_slice(&parts.layout.global.to_le_bytes());
    }
    Ok(ExpertSlab::Nvfp4 {
        weights,
        scales,
        globals: upload(backend, &globals)?,
    })
}

/// One routed expert's tensor name.
fn expert_name(slot: usize, expert: u32, role: &str) -> String {
    format!("{TEXT_PREFIX}layers.{slot}.mlp.experts.{expert}.{role}")
}

/// One slot's routed feed-forward.
fn import_moe(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    slot: usize,
    experts: u32,
) -> CoreResult<Moe> {
    let at = |role: &str| format!("{TEXT_PREFIX}layers.{slot}.mlp.{role}");
    // The codec is not stated here: `Moe::new` derives it from the three slabs
    // and refuses if they disagree, so a checkpoint whose down projection was
    // exported in another format is a load error rather than a launch that
    // reads biases as per-expert scales.
    Moe::new(
        weight(backend, file, &at("gate"))?,
        expert_slab(backend, file, slot, "gate_proj", experts)?,
        expert_slab(backend, file, slot, "up_proj", experts)?,
        expert_slab(backend, file, slot, "down_proj", experts)?,
        weight(backend, file, &at("shared_expert_gate"))?,
        ActDtype::F32,
    )
}

/// The value-head reordering one tensor needs, resolved against its own shape.
///
/// Not an `Option`. Every caller below has already decided that its tensor
/// carries the axis, so "no reordering" is not one of this function's answers:
/// it is the `None` that the one tensor without an axis passes to
/// [`reordered_weight`] directly. A `Result<Option<..>>` here would have made
/// every call site read as though the question were still open.
fn axis_for(
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
fn import_gdn(
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
fn import_attn(backend: &Arc<dyn Backend>, file: &ModeloptFile, slot: usize) -> CoreResult<Attn> {
    let at = |role: &str| format!("{TEXT_PREFIX}layers.{slot}.self_attn.{role}");
    // Every attention slot of this family owns its cache, so all three of the
    // key half's `Option`s are filled.
    Ok(Attn {
        q: weight(backend, file, &at("q_proj"))?,
        k: Some(weight(backend, file, &at("k_proj"))?),
        v: Some(weight(backend, file, &at("v_proj"))?),
        output: weight(backend, file, &at("o_proj"))?,
        q_norm: vector(backend, file, &at("q_norm"))?,
        k_norm: Some(vector(backend, file, &at("k_norm"))?),
    })
}

/// Moves every tensor to the device, in the order the pass reads them.
fn import_weights(
    backend: &Arc<dyn Backend>,
    file: &ModeloptFile,
    shape: &HybridShape,
) -> CoreResult<Weights> {
    let experts = shape.moe.map_or(0, |moe| moe.experts);
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
                (Block::Attn(import_attn(backend, file, slot)?), rank)
            }
        };
        let moe = import_moe(backend, file, slot, experts)?;
        layers.push(LayerWeights {
            input_norm: vector(
                backend,
                file,
                &format!("{TEXT_PREFIX}layers.{slot}.input_layernorm"),
            )?,
            // The pre-only scheme, as the dense hybrid has: nothing normalizes
            // a block's output and the residual sum carries no scalar.
            post_attn_norm: None,
            ffn_norm: vector(
                backend,
                file,
                &format!("{TEXT_PREFIX}layers.{slot}.post_attention_layernorm"),
            )?,
            post_ffn_norm: None,
            // The SHARED expert takes the dense feed-forward's own field,
            // because that is what it is; the routed half is beside it.
            ffn: Ffn {
                gate: weight(
                    backend,
                    file,
                    &format!("{TEXT_PREFIX}layers.{slot}.mlp.shared_expert.gate_proj"),
                )?,
                up: weight(
                    backend,
                    file,
                    &format!("{TEXT_PREFIX}layers.{slot}.mlp.shared_expert.up_proj"),
                )?,
                down: weight(
                    backend,
                    file,
                    &format!("{TEXT_PREFIX}layers.{slot}.mlp.shared_expert.down_proj"),
                )?,
            },
            moe: Some(moe),
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
        output_norm: vector(backend, file, &format!("{TEXT_PREFIX}norm"))?,
        // This family ships its own output projection, and it is NVFP4 - the
        // one target outside the feed-forward that is.
        lm_head: Some(weight(backend, file, "lm_head")?),
        ple_model_proj: None,
        ple_proj_norm: None,
        layers,
    })
}

impl Model for Qwen35MoeModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        self.stack_mut().open_session()
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        self.stack_mut().close_session(session)
    }

    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.stack_mut().prefill_chunk(session, tokens)
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        self.stack_mut().prefill_last(session, tokens)
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        self.stack_mut().decode_batch(steps)
    }

    /// The media path this family has, answered to a caller holding the
    /// contract rather than the type.
    ///
    /// The `MediaModel` impl below has existed since Phase 8 Part C; nothing
    /// asked for it through a trait object until a server did, and the default
    /// this overrides answers `None`. A server that opens a tower checks
    /// exactly this before it binds a port - so leaving the default in place
    /// meant `--vision` over this family refused by name, which is the good
    /// half of the failure, and meant no route existed for the rows at all,
    /// which is the whole of it.
    fn spec_mut(&mut self) -> Option<&mut dyn shiso_core::model::Speculating> {
        // The head is what makes a step speculative, and a build without one
        // answers no here rather than one verify pass later.
        self.stack.draft_resident().then_some(self)
    }

    fn media_mut(&mut self) -> Option<&mut dyn MediaModel> {
        Some(self)
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.stack_mut().teardown()
    }
}

/// The media path, which this family has because its stack carries one.
///
/// One method and no chunked sibling, for the reason the contract states and
/// [`crate::qwen35::Qwen35Model`] repeats: the stack's walk over a prompt's
/// placeholder runs is built per call, so a prompt handed over in pieces would
/// restart the running multimodal position and the encoder row the splice has
/// reached. The stack still splits the prompt into device-sized chunks
/// underneath, sharing one walk across them.
///
/// **The tower this family's rows come from is `crate::qwen35::Qwen35Tower`**,
/// unchanged. The two families' towers are the same tower - 27 blocks of width
/// 1152 projecting to 2048 - and this checkpoint writes it under the same
/// `model.visual` prefix; all that differs is that it leaves it interleaved
/// through the language model's own shards rather than in a shard of its own,
/// which is a container arm in that family's loader and nothing here.
impl shiso_core::model::Speculating for Qwen35MoeModel {
    fn speculate(
        &mut self,
        session: SessionId,
        first: TokenId,
    ) -> CoreResult<shiso_core::model::Speculated> {
        let eos = self.info.stop_tokens.primary();
        self.stack.speculate(session, first, eos)
    }
}

impl MediaModel for Qwen35MoeModel {
    fn prefill_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>> {
        self.stack_mut().prefill_last_media(session, tokens, media)
    }
}

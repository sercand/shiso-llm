//! The model type: a checkpoint of this family, loaded and answering the shared
//! contract.
//!
//! Everything here is identity. Which file the weights came from, what its
//! tensors are called, which token ends a turn, which formats a kernel in this
//! build can read - those are facts about this family, and they are the only
//! facts this crate holds. The stack that runs them is
//! [`shiso_engine::HybridStack`], and every method below forwards to it.
//!
//! # One constructor, and what it removes
//!
//! [`Qwen35Model::load`] either returns a model whose weights are on the device
//! or it returns an error. There is no `new` plus `load` pair and no state that
//! means "not loaded yet", so no caller downstream has to ask.
//!
//! Weights are imported rather than copied: the file mapping goes to the backend,
//! which wraps it where the device can read host memory in place and copies where
//! it cannot. This crate never branches on which of the two happened.
//!
//! # Two containers, one load
//!
//! A checkpoint of this family ships either as a single self-describing file or
//! as a directory of shards whose weights are packed in a format no kernel here
//! reads. [`shiso_modelfile::ModelFile`] opens both and answers both through
//! [`shiso_modelfile::ModelFile::tensor_host`], which hands back shareable host bytes
//! and a range - the mapping and a window into it for the first, a decoded
//! buffer and the whole of it for the second. That is the entire difference, and
//! it is spent before it reaches this file: nothing below asks which container
//! it got.
//!
//! # Why the binding here is the descriptor one
//!
//! [`bind_qwen35_desc`] rather than its borrowing sibling, and that is not a
//! style choice. A borrowed binding holds a payload per role, and a directory
//! checkpoint has to decode a payload to produce one - so binding that way would
//! unpack the whole model into the open file and keep it there, several
//! gigabytes that the device copies are about to duplicate. The descriptor
//! binding carries names, formats and dimensions only; each payload is fetched
//! when its turn comes and dropped once it is resident.

use std::path::Path;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, import_host};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, SessionId, TokenId};
use shiso_core::media::{MediaModel, VisionPrefill};
use shiso_core::model::{
    DecisionModel, DecisionQuestion, DecodeStep, Logits, Model, ModelInfo, StopTokens,
};
use shiso_engine::attn::Attn;
use shiso_engine::draft::DraftHead;
use shiso_engine::ffn::Ffn;
use shiso_engine::gdn::{Gdn, GdnGeometry};
use shiso_engine::layers::{Block, LayerWeights, Weights};
use shiso_engine::projection::Weight;
use shiso_engine::rotation::{
    HeadOrder, InputRotation, RotationDesc, RotationDirection, RotationScratch,
};
use shiso_engine::shape::{HybridShape, LayerKind};
use shiso_engine::{EngineOptions, HybridStack, MAX_PREFILL_CHUNK};
use shiso_modelfile::{ModelFile, ScaleFormat, TensorDesc};

use crate::qwen35::binding::{DescBinding, LayerRoles, bind_qwen35_desc};
use crate::qwen35::config::Qwen35Config;
use crate::qwen35::kernels::Qwen35Kernels;
use crate::qwen35::mtp::{MtpManifest, MtpWeights};
use crate::qwen35::rotation::{MLX_FILE as MLX_ROTATION, Rotated, RotationPlan};

/// The formats a bound tensor may be stored as.
///
/// A tensor outside this list ends the load rather than selecting a slower path:
/// a model that quietly ran a dequantized copy of one tensor would produce logits
/// nobody can attribute to a kernel.
///
/// The nine sub-4-bit formats at the end are the mixed set of the Saluki export,
/// accepted as a set: that file pairs a gate in one codec with an up projection
/// in another, so a list holding some of them would refuse the file anyway.
const SUPPORTED: &[DType] = &[
    DType::F32,
    DType::F16,
    DType::Q8_0,
    DType::Q4_K,
    DType::Q5_K,
    DType::Q6_K,
    DType::Pq2_0,
    DType::Ptq1_0,
    DType::Q2_K,
    DType::IQ2_XXS,
    DType::IQ2_XS,
    DType::IQ2_S,
    DType::IQ3_XXS,
    DType::IQ3_S,
    DType::IQ1_S,
    DType::IQ1_M,
    DType::IQ4_XS,
];

/// A loaded checkpoint of this family.
#[derive(Debug)]
pub struct Qwen35Model {
    stack: HybridStack,
    info: ModelInfo,
    /// Storage format of the layer-zero feed-forward gate.
    ///
    /// Kept because it is the one thing a loaded model knows about how its weights
    /// were quantized that survives the load: the binding is dropped once the tensors
    /// are resident, and the checkpoint tier has to be able to say what width the
    /// rows it stores were produced at long afterwards.
    ffn_gate: DType,
    /// The joint schema head of a Clef checkpoint: the same backbone with a
    /// decision head beside it. `None` for every other checkpoint.
    decision: Option<crate::clef::ClefHead>,
}

impl Qwen35Model {
    /// Storage format of the layer-zero feed-forward gate.
    #[must_use]
    pub const fn ffn_gate_dtype(&self) -> DType {
        self.ffn_gate
    }

    /// The stack this model forwards to.
    #[must_use]
    pub(crate) const fn stack(&self) -> &HybridStack {
        &self.stack
    }

    /// The stack this model forwards to, mutably.
    pub(crate) const fn stack_mut(&mut self) -> &mut HybridStack {
        &mut self.stack
    }
}

impl Qwen35Model {
    /// Opens a checkpoint and puts it on the device.
    ///
    /// `path` is either a single checkpoint file or a checkpoint directory;
    /// [`ModelFile::open`] decides which from the path and neither this function
    /// nor anything it calls asks again.
    ///
    /// Weights are imported rather than copied: the host bytes are handed to the
    /// backend, which wraps them where the device can read host memory in place
    /// and copies where it cannot. The engine never branches on which of the two
    /// happened. For a single file those bytes are the mapping itself, so a
    /// quantized checkpoint of several gigabytes is served from the page cache;
    /// for a directory they are one decoded buffer per tensor, which is what the
    /// packed formats cost.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for a profile this type does not serve, for a
    /// bound tensor whose format no kernel reads, and for a layer list that
    /// disagrees with the profile; [`CoreError::Io`] and the binding's own failures
    /// for a file that cannot be read or does not carry the roles; and
    /// [`CoreError::Backend`] when a device allocation or transfer fails.
    pub fn load(
        backend: &Arc<dyn Backend>,
        path: &Path,
        config: &Qwen35Config,
        opts: EngineOptions,
    ) -> CoreResult<Self> {
        Self::load_with_draft(backend, path, config, opts, None)
    }

    /// As [`Qwen35Model::load`], with the draft head the sidecar names.
    ///
    /// `head` is `None` for a plain build, which is every build until an
    /// operator asks for a speculative depth. A head is only bound when the
    /// options carry one: a resident head on a stack whose sessions keep no
    /// rollback records would draft, verify, and have nothing to wind back to,
    /// and [`HybridStack::bind_draft`] refuses exactly that.
    ///
    /// # Errors
    ///
    /// As [`Qwen35Model::load`], plus whatever the head's binding, its casts
    /// and its residency report.
    pub fn load_with_draft(
        backend: &Arc<dyn Backend>,
        path: &Path,
        config: &Qwen35Config,
        opts: EngineOptions,
        head: Option<&MtpManifest>,
    ) -> CoreResult<Self> {
        // A `modelopt` export is a third container, with its own reader and
        // its own per-tensor storage schemes; everything after the weights are
        // resident is the same as for the other two.
        if crate::qwen35::modelopt::is_modelopt_dir(path) {
            let (weights, ffn_gate) = crate::qwen35::modelopt::load(backend, path, config.shape())?;
            let embed = crate::qwen35::modelopt::EMBED_DTYPE;
            return Self::assemble(backend, config, opts, head, (weights, embed, ffn_gate));
        }
        // A rotated directory keeps its output projection's columns where the
        // file put them; see `Qwen35RotatedNames`.
        let names: Arc<dyn shiso_modelfile::NameMap> = if path.join(MLX_ROTATION).is_file() {
            Arc::new(crate::qwen35::st_names::Qwen35RotatedNames)
        } else {
            Arc::new(crate::qwen35::Qwen35Names)
        };
        let file = ModelFile::open_with(path, names).map_err(open_failure)?;
        let binding = bind_qwen35_desc(&file, config).map_err(open_failure)?;
        validate_layers(config.shape(), &binding)?;
        let embed = embedding_dtype(&file, &binding.token_embd)?;
        let ffn_gate = match binding.layers.first() {
            Some(layer) => stored_dtype(&file, &layer.ffn.gate)?,
            None => embed,
        };
        let rotation = Rotation::for_file(backend, &file, path, config.shape())?;
        let weights = import_weights(backend, &file, &binding, config.shape(), rotation.as_ref())?;
        // A Clef checkpoint is this backbone under its own architecture name,
        // with the schema head the binding never reads beside it.
        let decision = if shiso_modelfile::source::TensorSource::architecture(&file)
            == Some(crate::qwen35::binding::CLEF_ARCH)
        {
            Some(crate::clef::ClefHead::load(backend, path)?)
        } else {
            None
        };
        // The file is dropped here on purpose: every weight that needed its mapping
        // holds its own handle on it, so the bytes stay mapped for exactly as long
        // as some device buffer names them and no longer. The same sentence covers
        // a directory checkpoint's decoded buffers: each is pinned by the device
        // buffer that named it, which is what makes a wrapped one safe to read, and
        // nothing is pinned twice because the file kept no copy of its own.
        drop(binding);
        drop(file);
        let mut model = Self::assemble(backend, config, opts, head, (weights, embed, ffn_gate))?;
        model.decision = decision;
        Ok(model)
    }

    /// Builds the stack over weights that are already resident, and binds the
    /// draft head when the options ask for one.
    ///
    /// `resident` is the weights, the format the vocabulary table was imported
    /// as, and the format of the layer-zero feed-forward gate - the three
    /// things a container's import decides and the stack needs.
    fn assemble(
        backend: &Arc<dyn Backend>,
        config: &Qwen35Config,
        opts: EngineOptions,
        head: Option<&MtpManifest>,
        resident: (Weights, DType, DType),
    ) -> CoreResult<Self> {
        let (weights, embed, ffn_gate) = resident;
        // Measured before the stack is announced resident, because a device
        // that will run no recurrent shape at all cannot run this model and
        // should say so at load rather than at the first prompt.
        let kernels =
            Qwen35Kernels::choose(backend.as_ref(), &GdnGeometry::new(&config.shape().gdn)?)?;
        let mut stack = HybridStack::new(
            backend,
            config.shape().clone(),
            weights,
            embed,
            Arc::new(kernels),
            // No per-layer embedding table: this family has none, and the
            // shape says so, so the loop never dispatches the branch.
            None,
            opts,
        )?;
        if let Some(manifest) = head.filter(|_| opts.spec.is_speculative()) {
            let bound = MtpWeights::bind(manifest)?;
            let parts = crate::qwen35::mtp::resident(&bound, backend, config.shape().hidden)?;
            // Sized for the widest pass the head ever runs, which is a prompt
            // chunk it is warmed over and not the chain it drafts: a chain is
            // one row, and a head whose scratch was sized by the chain would
            // refuse to warm at all.
            let resident_head = DraftHead::new(
                backend,
                parts,
                stack.geometry(),
                stack.pool_pages(),
                MAX_PREFILL_CHUNK,
            )?;
            stack.bind_draft(resident_head)?;
        }
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: config.shape().vocab,
                hidden: config.shape().hidden,
                n_layers: config.shape().n_layers(),
                stop_tokens: StopTokens::new(&config.stop_tokens())?,
            },
            stack,
            ffn_gate,
            decision: None,
        })
    }
}

impl Qwen35Model {
    /// Reads the block drafter in `dir` and binds it to this model's stack.
    ///
    /// # Errors
    ///
    /// Whatever [`crate::qwen35::block_draft::load`] reports for the directory,
    /// and [`CoreError::Unsupported`] from the stack for a model loaded without
    /// a speculative depth.
    pub fn bind_block_drafter(&mut self, backend: &Arc<dyn Backend>, dir: &Path) -> CoreResult<()> {
        let geometry = self.stack.geometry().clone();
        let parts = crate::qwen35::block_draft::load(
            backend,
            dir,
            (self.info.hidden, self.info.n_layers, self.info.vocab),
        )?;
        let drafter = shiso_engine::block_draft::BlockDrafter::new(
            backend,
            parts,
            (&geometry, self.info.n_layers),
            self.stack.pool_pages(),
            MAX_PREFILL_CHUNK,
        )?;
        self.stack.bind_block_drafter(drafter)
    }
}

/// Checks the bound layer list against the config's own.
///
/// The binding already checked the roles a slot carries; what is checked here is
/// that the slot kinds arrive in the order the config says they do, because the
/// recurrent ordinals and the cache layer ranks are both derived from that order.
fn validate_layers(shape: &HybridShape, binding: &DescBinding) -> CoreResult<()> {
    let want = u64::from(shape.n_layers());
    let got = u64::try_from(binding.layers.len()).unwrap_or(u64::MAX);
    if want != got {
        return Err(CoreError::ShapeMismatch {
            field: "layer count",
            expected: want,
            got,
        });
    }
    for (index, layer) in binding.layers.iter().enumerate() {
        let slot = u32::try_from(index).unwrap_or(u32::MAX);
        let expected = shape
            .layer_kind(LayerIdx::new(slot))
            .ok_or(CoreError::ShapeMismatch {
                field: "layer kind",
                expected: want,
                got: u64::from(slot),
            })?;
        if expected != layer.kind {
            return Err(CoreError::InvalidValue {
                key: "layer kind",
                value: format!(
                    "slot {slot} is {:?} but the profile says {expected:?}",
                    layer.kind
                ),
            });
        }
    }
    Ok(())
}

/// Moves every bound tensor to the device.
fn import_weights(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    binding: &DescBinding,
    shape: &HybridShape,
    rot: Option<&Rotation>,
) -> CoreResult<Weights> {
    let mut gdn_rank = 0;
    let mut attn_rank = 0;
    let mut layers = Vec::with_capacity(binding.layers.len());
    for (slot, layer) in binding.layers.iter().enumerate() {
        // The run rank comes from the shape rather than from the counter beside
        // it: for this family the two are the same number at every slot, and
        // they are the same number because every slot owns its cache. A family
        // whose slots share one does not have that coincidence, and the shape is
        // the only thing that knows.
        let il = LayerIdx::new(u32::try_from(slot).unwrap_or(u32::MAX));
        let kv_rank = shape.kv_owner_rank(il);
        let (block, ordinal) = match layer.kind {
            LayerKind::Gdn => {
                let rank = gdn_rank;
                gdn_rank += 1;
                (Block::Gdn(import_gdn(backend, file, layer, rot)?), rank)
            }
            LayerKind::Attn => {
                let rank = attn_rank;
                attn_rank += 1;
                (Block::Attn(import_attn(backend, file, layer, rot)?), rank)
            }
            // This family's shape holds neither; its config never states one.
            LayerKind::Ssm => {
                return Err(CoreError::Unsupported {
                    what: "a state-space slot in this family",
                    backend: "qwen35",
                });
            }
        };
        layers.push(LayerWeights {
            input_norm: vector(backend, file, &layer.input_norm)?,
            // This family normalizes only what a block reads, carries no
            // per-slot scalar, and closes a block out through the unscaled
            // fused norm. All three are read from the shape rather than from
            // these values being absent, so the launches here are the ones this
            // family has always dispatched.
            post_attn_norm: None,
            ffn_norm: Some(vector(backend, file, &layer.post_norm)?),
            post_ffn_norm: None,
            ffn: Some(Ffn {
                gate: Some(weight(backend, file, rot, &layer.ffn.gate)?),
                up: weight(backend, file, rot, &layer.ffn.up)?,
                down: weight(backend, file, rot, &layer.ffn.down)?,
            }),
            // A dense feed-forward: this family's slots have no router.
            moe: None,
            block,
            ordinal,
            kv_rank,
            layer_scalar: 1.0,
            ple: None,
        });
    }
    Ok(Weights {
        token_embd: embedding(backend, file, rot, &binding.token_embd)?,
        output_norm: vector(backend, file, &binding.output_norm)?,
        lm_head: binding
            .lm_head
            .as_ref()
            .map(|view| weight(backend, file, rot, view))
            .transpose()?,
        ple_model_proj: None,
        ple_proj_norm: None,
        layers,
    })
}

/// Moves one recurrent slot's weights to the device.
fn import_gdn(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    layer: &LayerRoles<TensorDesc>,
    rot: Option<&Rotation>,
) -> CoreResult<Gdn> {
    let bound = layer.gdn.as_ref().ok_or(CoreError::InvalidValue {
        key: "layer kind",
        value: "a recurrent slot bound without recurrent weights".to_owned(),
    })?;
    Ok(Gdn {
        qkv: weight(backend, file, rot, &bound.qkv)?,
        gate: weight(backend, file, rot, &bound.gate)?,
        alpha: weight(backend, file, rot, &bound.alpha)?,
        beta: weight(backend, file, rot, &bound.beta)?,
        out: weight(backend, file, rot, &bound.out)?,
        conv1d: vector(backend, file, &bound.conv1d)?,
        a: vector(backend, file, &bound.a)?,
        dt_bias: vector(backend, file, &bound.dt_bias)?,
        norm: vector(backend, file, &bound.norm)?,
    })
}

/// Moves one attention slot's weights to the device.
fn import_attn(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    layer: &LayerRoles<TensorDesc>,
    rot: Option<&Rotation>,
) -> CoreResult<Attn> {
    let bound = layer.attn.as_ref().ok_or(CoreError::InvalidValue {
        key: "layer kind",
        value: "an attention slot bound without attention weights".to_owned(),
    })?;
    // Every attention slot of this family owns its cache, so all three of the
    // key half's `Option`s are filled: a slot that reads another's would carry
    // none of them, and there is no such slot here.
    Ok(Attn {
        q: weight(backend, file, rot, &bound.q)?,
        k: Some(weight(backend, file, rot, &bound.k)?),
        v: Some(weight(backend, file, rot, &bound.v)?),
        output: weight(backend, file, rot, &bound.output)?,
        q_norm: Some(vector(backend, file, &bound.q_norm)?),
        k_norm: Some(vector(backend, file, &bound.k_norm)?),
    })
}

/// Imports one projection matrix, refusing a format this family does not ship.
///
/// Two checks, and they are not the same one twice. This list is the family's own:
/// these are the formats a checkpoint of this family is expected to store, and
/// anything else means the file is not what it claims. The engine then refuses,
/// separately, a format for which no matvec exists in this build. A file can pass
/// the first and fail the second - that is a build that cannot run a legitimate
/// checkpoint, which is a different problem and deserves a different message.
fn weight(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    rot: Option<&Rotation>,
    desc: &TensorDesc,
) -> CoreResult<Weight> {
    let stored = stored_weight(backend, file, desc)?;
    match rot {
        Some(rotation) => rotation.attach(backend, &desc.name, stored),
        None => refuse_unrotated(&desc.name, stored),
    }
}

/// [`weight`] before any rotation is attached.
fn stored_weight(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    desc: &TensorDesc,
) -> CoreResult<Weight> {
    if desc.dtype == DType::BF16 && matches!(file, ModelFile::Gguf(_)) {
        return widened_bf16(backend, file, desc);
    }
    if !SUPPORTED.contains(&desc.dtype) {
        return Err(CoreError::Unsupported {
            what: "a weight stored in a format this family's loader does not accept",
            backend: "qwen35",
        });
    }
    let dims = desc.shape.dims();
    let in_dim = extent(dims.first().copied().unwrap_or(0))?;
    let out_dim = extent(dims.get(1).copied().unwrap_or(1))?;
    // An MLX export's projections stay packed: the affine matmul reads them as
    // stored, and decoding a 27B export would cost four times its size in
    // device memory. Everything else in the directory is decoded as before.
    if let ModelFile::Safetensors(st) = file
        && let Some(packed) = st.affine_packed(&desc.name).map_err(open_failure)?
    {
        let dtype = affine_dtype(packed.layout.bits, packed.layout.group_size)?;
        // The 2-bit kernels read `f16` scales and the 4- and 8-bit ones brain
        // floats; the two are the same width, so the pairing is checked here.
        let want = if dtype == DType::MlxAffine2Gs128 {
            ScaleFormat::F16
        } else {
            ScaleFormat::Bf16
        };
        if packed.scale_format != want {
            return Err(CoreError::Unsupported {
                what: "an MLX affine weight whose scale format its matmul does not read",
                backend: "qwen35",
            });
        }
        return Weight::import_affine_slices(
            backend,
            dtype,
            in_dim,
            out_dim,
            &packed.words,
            &packed.scales,
            &packed.biases,
        );
    }
    let (host, range) = file.tensor_host(desc).map_err(open_failure)?;
    Weight::import(backend, desc.dtype, in_dim, out_dim, host, range)
}

/// Imports the token embedding: packed when a row-gather kernel reads its packed
/// format (4-bit affine), decoded otherwise - an 8-bit table is read by nothing
/// but the decoded path.
fn embedding(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    rot: Option<&Rotation>,
    desc: &TensorDesc,
) -> CoreResult<Weight> {
    if embedding_dtype(file, desc)? == desc.dtype && !desc.dtype.is_ternary() {
        let dims = desc.shape.dims();
        let in_dim = extent(dims.first().copied().unwrap_or(0))?;
        let out_dim = extent(dims.get(1).copied().unwrap_or(1))?;
        let (host, range) = file.tensor_host(desc).map_err(open_failure)?;
        let table = Weight::import(backend, desc.dtype, in_dim, out_dim, host, range)?;
        return match rot {
            Some(rotation) => rotation.attach(backend, &desc.name, table),
            None => Ok(table),
        };
    }
    weight(backend, file, rot, desc)
}

/// The format the token embedding is imported as; see [`embedding`].
fn embedding_dtype(file: &ModelFile, desc: &TensorDesc) -> CoreResult<DType> {
    match stored_dtype(file, desc)? {
        packed @ (DType::MlxAffine4Gs64 | DType::MlxAffine2Gs128) => Ok(packed),
        _ => Ok(desc.dtype),
    }
}

/// The packed format of an MLX affine weight at `bits` in groups of `group`.
fn affine_dtype(bits: u32, group: usize) -> CoreResult<DType> {
    match (bits, group) {
        (4, 64) => Ok(DType::MlxAffine4Gs64),
        (8, 64) => Ok(DType::MlxAffine8Gs64),
        (2, 128) => Ok(DType::MlxAffine2Gs128),
        _ => Err(CoreError::Unsupported {
            what: "an MLX affine weight at a bit width and group size with no matmul",
            backend: "qwen35",
        }),
    }
}

/// The format `desc` is imported as: its packed affine format when the file
/// keeps it packed, the binding's own otherwise.
fn stored_dtype(file: &ModelFile, desc: &TensorDesc) -> CoreResult<DType> {
    if let ModelFile::Safetensors(st) = file
        && let Some(triple) = st.affine_triple(&desc.name).map_err(open_failure)?
    {
        return affine_dtype(triple.layout.bits, triple.layout.group_size);
    }
    Ok(desc.dtype)
}

/// Imports a GGUF's `bfloat16` matrix widened to `f32`.
///
/// No projection reads brain floats, and the only ones a file of this family
/// stores are the recurrent block's two narrow gate projections - 48 rows a
/// layer. Widening is exact, costs a few megabytes, and keeps the projection
/// off a format no kernel was written for.
fn widened_bf16(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    desc: &TensorDesc,
) -> CoreResult<Weight> {
    let dims = desc.shape.dims();
    let in_dim = extent(dims.first().copied().unwrap_or(0))?;
    let out_dim = extent(dims.get(1).copied().unwrap_or(1))?;
    let bytes = file.tensor_bytes(&desc.name).map_err(open_failure)?;
    let wide: Vec<u8> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|h| half::bf16::from_le_bytes(*h).to_f32().to_le_bytes())
        .collect();
    let len = wide.len();
    Weight::import(backend, DType::F32, in_dim, out_dim, Arc::new(wide), 0..len)
}

/// Refuses a ternary or 2-bit weight from a checkpoint that states no
/// rotation.
///
/// Those formats exist for rotated exports, and one read in the plain basis
/// loads without complaint and generates nonsense. Checked per weight as it is
/// imported, not on the vocabulary table alone: an export can keep its table
/// at eight bits or half precision and still store every projection ternary.
/// [`Rotation::attach`] is the same check for a file that does state one.
fn refuse_unrotated(name: &str, weight: Weight) -> CoreResult<Weight> {
    if !weight.dtype.is_ternary() && weight.dtype != DType::MlxAffine2Gs128 {
        return Ok(weight);
    }
    Err(CoreError::InvalidValue {
        key: "prism.hadamard",
        value: format!(
            "{name} is stored ternary or 2-bit, but the checkpoint carries no rotation \
             metadata; it is in a basis this loader cannot recover, so it is refused \
             rather than generating nonsense"
        ),
    })
}

/// A checkpoint's rotation, with the scratch its projections share.
#[derive(Debug)]
struct Rotation {
    plan: RotationPlan,
    scratch: Arc<RotationScratch>,
    /// The recurrent output projection's head permutation, when the value
    /// heads repeat the key heads.
    value_heads: Option<HeadOrder>,
}

impl Rotation {
    /// The rotation `file` states, or `None` for an ordinary checkpoint.
    fn for_file(
        backend: &Arc<dyn Backend>,
        file: &ModelFile,
        path: &Path,
        shape: &HybridShape,
    ) -> CoreResult<Option<Self>> {
        let plan = match file {
            ModelFile::Gguf(gguf) => RotationPlan::from_gguf(gguf.metadata())?,
            ModelFile::Safetensors(_) if path.join(MLX_ROTATION).is_file() => {
                Some(RotationPlan::from_mlx_dir(path)?)
            }
            _ => None,
        };
        let Some(plan) = plan else {
            return Ok(None);
        };
        let gdn = &shape.gdn;
        let value_heads = plan.value_heads(gdn.head_dim, gdn.qk_heads, gdn.v_heads)?;
        let scratch = RotationScratch::new(backend, plan.widest(), MAX_PREFILL_CHUNK)?;
        Ok(Some(Self {
            plan,
            scratch,
            value_heads,
        }))
    }

    /// Attaches the rotation the plan states for `name` to `weight`.
    ///
    /// A ternary or 2-bit weight the plan does not list is refused: those
    /// formats exist for rotated exports, and one left in the plain basis
    /// means this reader has misread the file.
    fn attach(&self, backend: &Arc<dyn Backend>, name: &str, weight: Weight) -> CoreResult<Weight> {
        let Some(role) = self.plan.role(name) else {
            if weight.dtype.is_ternary() || weight.dtype == DType::MlxAffine2Gs128 {
                return Err(CoreError::InvalidValue {
                    key: "prism.hadamard.weight_names",
                    value: format!("{name} is ternary but not listed as rotated"),
                });
            }
            return Ok(weight);
        };
        let direction = match role {
            Rotated::Forward => RotationDirection::Forward,
            Rotated::Inverse => RotationDirection::Inverse,
        };
        let forward = direction == RotationDirection::Forward;
        let heads = if forward && name.ends_with(".ssm_out.weight") {
            self.value_heads
        } else {
            None
        };
        let rotation = InputRotation::new(
            backend,
            RotationDesc {
                block: self.plan.block(),
                direction,
                signs: self.plan.signs(weight.in_dim)?,
                heads,
                // An inverse rotation carries the scratch too: a table with no
                // separate output head is also read as the head, and that
                // projection rotates its input forward (`shiso_engine::rotation`).
                scratch: Some(Arc::clone(&self.scratch)),
            },
        )?;
        weight.with_rotation(rotation)
    }
}

/// Imports one weight vector, which every format stores as plain `f32`.
fn vector(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    desc: &TensorDesc,
) -> CoreResult<DeviceBuffer> {
    if desc.dtype != DType::F32 {
        return Err(CoreError::Unsupported {
            what: "a norm or bias vector stored as anything but single precision",
            backend: "qwen35",
        });
    }
    let (host, range) = file.tensor_host(desc).map_err(open_failure)?;
    import_host(backend, host, range)
}

/// Narrows one file dimension to the width a launch counts in.
fn extent(dim: u64) -> CoreResult<u32> {
    u32::try_from(dim).map_err(|_| CoreError::ShapeMismatch {
        field: "tensor extent",
        expected: u64::from(u32::MAX),
        got: dim,
    })
}

/// Presents a checkpoint-file failure in the core vocabulary.
fn open_failure(source: shiso_modelfile::GgufError) -> CoreError {
    CoreError::Backend {
        backend: "qwen35",
        op: "load",
        source: Box::new(source),
    }
}

impl Model for Qwen35Model {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        self.stack.open_session()
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        self.stack.close_session(session)
    }

    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.stack.prefill_chunk(session, tokens)
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        self.stack.prefill_last(session, tokens)
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        self.stack.decode_batch(steps)
    }

    fn media_mut(&mut self) -> Option<&mut dyn MediaModel> {
        Some(self)
    }

    fn decision_mut(&mut self) -> Option<&mut dyn DecisionModel> {
        self.decision.is_some().then_some(self)
    }

    fn spec_mut(&mut self) -> Option<&mut dyn shiso_core::model::Speculating> {
        // The head is what makes a step speculative, and a build without one
        // answers no here rather than one verify pass later.
        self.stack.draft_resident().then_some(self)
    }

    /// The stack takes the verify's greedy picks itself, so the list goes to
    /// it; the scheduler masks every row it draws from on its own.
    fn suppress_tokens(&mut self, ids: &[TokenId]) -> CoreResult<()> {
        self.stack.suppress_tokens(ids)
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.stack.teardown()
    }
}

/// The schema decision path, which a Clef checkpoint has because its head
/// reads this stack's final rows.
impl DecisionModel for Qwen35Model {
    fn capture_hidden(&mut self, session: SessionId) -> CoreResult<()> {
        self.stack.capture_hidden(session)
    }

    fn decide(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
        questions: &[DecisionQuestion],
    ) -> CoreResult<Vec<Vec<f32>>> {
        let rows = self.stack.take_hidden(session)?;
        let head = self.decision.as_ref().ok_or(CoreError::Unsupported {
            what: "a schema decision on a checkpoint without a decision head",
            backend: "qwen35",
        })?;
        head.score(&rows, prompt, questions)
    }
}

/// The media path, which this family has because its stack carries one.
///
/// One method and no chunked sibling, for the reason the contract states: the
/// stack's walk over a prompt's placeholder runs is built per call, so a prompt
/// handed over in pieces would restart the running multimodal position and the
/// encoder row the splice has reached. The stack still splits the prompt into
/// device-sized chunks underneath, sharing one walk across them, which is what
/// makes the single call cost no more than the chunked one would have.
impl MediaModel for Qwen35Model {
    fn prefill_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>> {
        self.stack.prefill_last_media(session, tokens, media)
    }
}

impl shiso_core::model::Speculating for Qwen35Model {
    fn speculate(
        &mut self,
        session: SessionId,
        first: TokenId,
    ) -> CoreResult<shiso_core::model::Speculated> {
        let eos = self.info.stop_tokens.primary();
        self.stack.speculate(session, first, eos)
    }

    // None: the stack drafts with a head rather than a block drafter, whose
    // greedy step is the only one it has.
    fn verify_block(
        &mut self,
        session: SessionId,
        first: TokenId,
        hint: shiso_core::model::DraftHint,
        proposal: &[TokenId],
    ) -> CoreResult<Option<shiso_core::model::VerifiedBlock>> {
        if !self.stack.block_resident() {
            return Ok(None);
        }
        let eos = self.info.stop_tokens.primary();
        self.stack
            .draft_and_verify(session, first, eos, hint, proposal)
            .map(Some)
    }

    fn settle_block(
        &mut self,
        session: SessionId,
        block: shiso_core::model::VerifiedBlock,
        accepted: u32,
    ) -> CoreResult<()> {
        let _ = block;
        self.stack.settle_verified(session, accepted)
    }
}

#[cfg(test)]
mod tests {
    use super::SUPPORTED;
    use shiso_core::DType;

    /// The nine mixed i-quant formats of the Saluki export, accepted together
    /// or not at all: a loader that takes the gate's `iq1_s` but refuses the
    /// up projection's `iq1_m` cannot load the file at all, so a partial list
    /// is no list.
    #[test]
    fn the_loader_accepts_the_mixed_i_quant_formats() {
        for dtype in [
            DType::Q2_K,
            DType::IQ2_XXS,
            DType::IQ2_XS,
            DType::IQ2_S,
            DType::IQ3_XXS,
            DType::IQ3_S,
            DType::IQ1_S,
            DType::IQ1_M,
            DType::IQ4_XS,
        ] {
            assert!(
                SUPPORTED.contains(&dtype),
                "the family must accept a weight stored in {}",
                dtype.type_name()
            );
        }
    }
}

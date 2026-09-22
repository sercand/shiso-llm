//! The model type: a Gemma 4 checkpoint, loaded and answering the shared
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
//! [`Gemma4Model::load`] either returns a model whose weights are on the device
//! or it returns an error. There is no `new` plus `load` pair and no state that
//! means "not loaded yet", so no caller downstream has to ask.
//!
//! Weights are imported rather than copied: the file mapping goes to the
//! backend, which wraps it where the device can read host memory in place and
//! copies where it cannot. This crate never branches on which of the two
//! happened.
//!
//! # The one tensor that is not imported
//!
//! `per_layer_token_embd` holds one `q6_K` block per layer slot for each of the
//! 262,144 token ids - 8,820 bytes a row, 2.3 GB in total, and the largest
//! tensor in the file by a wide margin. A forward pass needs the rows of the
//! tokens in front of it and nothing else, so the default holding streams them
//! off the disk through [`shiso_ple::PleStream`] and the device never sees the
//! rest.
//!
//! [`PleOptions`] can ask for the whole table instead. That is not a tuning
//! knob with an obvious better setting: streaming trades a disk read per chunk
//! for 2.3 GB of device memory, and which of the two a deployment can spare is
//! not something this crate knows. What it does guarantee is that the two are
//! the same numbers - the resident table and the streamed rows are the same
//! `q6_K` blocks, decoded by the same prepare kernel, which is asserted against
//! the device in `gemma4_decode.rs`.
//!
//! # Why the binding here is the descriptor one
//!
//! [`bind_gemma4_desc`] carries names, formats and dimensions only. A borrowed
//! binding would hold a payload per role, which for the per-layer table would
//! mean naming 2.3 GB of mapping that is then never read through that name -
//! and for this family it would also mean the table could not be handed to a
//! reader that wants the file offset rather than the bytes.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use shiso_core::buffer::{DeviceBuffer, alloc, import_host};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, SessionId, TokenId};
use shiso_core::media::{MediaModel, VisionPrefill};
use shiso_core::model::{DecodeStep, Logits, Model, ModelInfo, StopTokens};
use shiso_core::{Backend, DType};
use shiso_engine::attn::Attn;
use shiso_engine::dense::DenseKernels;
use shiso_engine::ffn::Ffn;
use shiso_engine::layers::{Block, LayerWeights, PerLayerWeights, PleTable, Weights};
use shiso_engine::projection::Weight;
use shiso_engine::shape::HybridShape;
use shiso_engine::{EngineOptions, HybridStack};
use shiso_modelfile::safetensors::SafetensorsFile;
use shiso_modelfile::{ModelFile, TensorDesc};
use shiso_ple::{PleConfig, PleStats, PleStream};

use crate::gemma4::binding::{
    Gemma4DescBinding, Gemma4LayerRoles, VECTOR_TYPES, WEIGHT_TYPES, bind_gemma4_desc,
};
use crate::gemma4::binding_mlx::{Gemma4Names, bind_gemma4_mlx_desc};
use crate::gemma4::config::Gemma4Config;
use crate::gemma4::ple_repack;

/// Bytes one row of the per-layer table occupies: one block of the table's own
/// format per layer slot.
///
/// Derived from the format and the slot count rather than written down, for the
/// reason the engine derives the same number the same way in
/// [`shiso_engine::layers::PerLayerGeometry::row_bytes`]: the two are the reader's
/// stride and the prepare kernel's, and a literal here would be a third statement
/// of a fact the other two compute. It is still *checked* against the tensor's
/// own `bytes / rows`, because the streaming reader turns a token id into a file
/// offset by multiplying it: a stride one block out reads every row after the
/// first from the middle of its predecessor, and every value that comes back is
/// finite.
fn ple_row_bytes(dtype: DType, n_layers: u32) -> u64 {
    u64::from(n_layers).saturating_mul(u64::from(dtype.block_bytes().get()))
}

/// Elements of one row of the per-layer table: one block's span per layer slot.
fn ple_row_elems(dtype: DType, n_layers: u32) -> u32 {
    n_layers.saturating_mul(dtype.block_elems().get())
}

/// How the per-layer table is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleOptions {
    /// Row-cache bytes for the streamed table; 0 disables the cache.
    pub cache_bytes: u64,
    /// Upload the whole table instead of streaming it.
    pub resident: bool,
}

impl Default for PleOptions {
    fn default() -> Self {
        Self {
            cache_bytes: 512 << 20,
            resident: false,
        }
    }
}

/// A loaded Gemma 4 checkpoint.
#[derive(Debug)]
pub struct Gemma4Model {
    stack: HybridStack,
    info: ModelInfo,
    /// Storage format of the layer-zero feed-forward gate.
    ///
    /// Kept because it is the one thing a loaded model knows about how its
    /// weights were quantized that survives the load: the binding is dropped
    /// once the tensors are resident, and the checkpoint tier has to be able to
    /// say what width the rows it stores were produced at long afterwards.
    ffn_gate: DType,
}

impl Gemma4Model {
    /// Storage format of the layer-zero feed-forward gate.
    #[must_use]
    pub const fn ffn_gate_dtype(&self) -> DType {
        self.ffn_gate
    }

    /// What the per-layer reader has done since this model was loaded.
    ///
    /// `None` when nothing could have read a row: the table is held resident, or
    /// the model has been torn down. The distinction matters here more than
    /// anywhere else in this family - a streamed table that reports zero reads
    /// is a cache that never missed, and a resident one reports nothing at all
    /// because no reader exists to count.
    #[must_use]
    pub fn ple_stats(&self) -> Option<PleStats> {
        self.stack.ple_stats()
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

    /// Opens a checkpoint and puts it on the device.
    ///
    /// `path` names the container the weights come from. Every tensor but the
    /// per-layer table is imported: the host bytes are handed to the backend,
    /// which wraps them where the device can read host memory in place and
    /// copies where it cannot. The per-layer table is held the way `ple` asks -
    /// streamed off the file by token id, which is the default, or imported like
    /// any other weight.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for a bound tensor whose format this family
    /// does not ship, for a per-layer table that is not `q6_K`, and for a
    /// streamed table in a container that is not a single mapped file;
    /// [`CoreError::ShapeMismatch`] when the table's row stride is not the one
    /// the reader and the prepare kernel both assume; [`CoreError::Io`] when the
    /// container cannot be opened a second time for the reader; the binding's
    /// own failures for a file that does not carry the roles;
    /// [`CoreError::Unsupported`] for a container this family cannot read; and
    /// [`CoreError::Backend`] when a device allocation or transfer fails.
    pub fn load(
        backend: &Arc<dyn Backend>,
        path: &Path,
        config: &Gemma4Config,
        opts: EngineOptions,
        ple: PleOptions,
    ) -> CoreResult<Self> {
        let file = ModelFile::open_with(path, Arc::new(Gemma4Names)).map_err(open_failure)?;
        let (embed, ffn_gate, table, weights) = match &file {
            ModelFile::Gguf(gguf) => {
                let binding = bind_gemma4_desc(gguf, config).map_err(open_failure)?;
                let embed = binding.token_embd.dtype;
                let ffn_gate = binding
                    .layers
                    .first()
                    .map_or(embed, |layer| layer.ffn.gate.dtype);
                let table = open_table(backend, &file, path, config, &binding.ple_table, ple)?;
                let weights = import_weights(backend, &file, &binding, config.shape())?;
                (embed, ffn_gate, table, weights)
            }
            ModelFile::Safetensors(st) => {
                let binding = bind_gemma4_mlx_desc(st, config).map_err(open_failure)?;
                let embed = DType::MlxAffine4Gs64;
                let ffn_gate = DType::MlxAffine8Gs64;
                let table = open_table_mlx(backend, st, path, config, ple)?;
                let weights = import_weights_mlx(backend, st, &binding, config.shape())?;
                (embed, ffn_gate, table, weights)
            }
            _ => {
                return Err(CoreError::Unsupported {
                    what: "gemma 4 reads a gguf file or an mlx directory",
                    backend: "gemma4",
                });
            }
        };
        // The file is dropped here on purpose: every weight that needed its
        // mapping holds its own handle on it, so the bytes stay mapped for
        // exactly as long as some device buffer names them and no longer. The
        // streamed table is the exception that proves it - it holds a file
        // descriptor of its own rather than a share of this mapping, because a
        // reader that seeks is not a buffer that is wrapped.
        drop(file);

        let stack = HybridStack::new(
            backend,
            config.shape().clone(),
            weights,
            embed,
            Arc::new(DenseKernels),
            Some(table),
            opts,
        )?;
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: config.shape().vocab,
                hidden: config.shape().hidden,
                n_layers: config.shape().n_layers(),
                stop_tokens: StopTokens::new(config.stop_tokens())?,
            },
            stack,
            ffn_gate,
        })
    }
}

/// Holds the per-layer table the way the options ask, checking its stride first.
///
/// The stride is the one number both halves of the per-layer path derive
/// independently: the reader multiplies a token id by it to find a row, and the
/// prepare kernel strides its staging by it to find a slot's block. They are
/// checked against each other here - and against the tensor's own size, which is
/// the container's third statement of the same fact - because a mismatch
/// produces rows that are finite, wrong, and attributable to nothing.
fn open_table(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    path: &Path,
    config: &Gemma4Config,
    desc: &TensorDesc,
    ple: PleOptions,
) -> CoreResult<PleTable> {
    if desc.dtype != DType::Q6_K {
        return Err(CoreError::Unsupported {
            what: "a per-layer embedding table in a format no prepare kernel decodes",
            backend: "gemma4",
        });
    }
    let n_layers = config.shape().n_layers();
    let expected = ple_row_bytes(desc.dtype, n_layers);
    let rows = u64::from(config.ple_vocab());
    let stride = desc.bytes.checked_div(rows).unwrap_or(0);
    if stride * rows != desc.bytes || stride != expected {
        return Err(CoreError::ShapeMismatch {
            field: "per-layer table row bytes",
            expected,
            got: stride,
        });
    }
    if ple.resident {
        let (host, range) = file.tensor_host(desc).map_err(open_failure)?;
        let table = Weight::import(
            backend,
            desc.dtype,
            ple_row_elems(desc.dtype, n_layers),
            config.ple_vocab(),
            host,
            range,
        )?;
        return Ok(PleTable::Resident { table });
    }
    // A sharded container has no single file the reader could seek in, and the
    // offsets on its descriptors are the shard's rather than a mapping's. Rather
    // than stream from the wrong place, this refuses and names the holding that
    // does work.
    if !file.is_mmap_backed() {
        return Err(CoreError::Unsupported {
            what: "streaming the per-layer table out of a container that is not a single mapped \
                   file; load it resident instead",
            backend: "gemma4",
        });
    }
    let shard = File::open(path).map_err(CoreError::Io)?;
    let row_bytes = u32::try_from(stride).map_err(|_| CoreError::ShapeMismatch {
        field: "per-layer table row bytes",
        expected,
        got: stride,
    })?;
    let stream = PleStream::open_direct(
        shard,
        desc.abs_offset,
        rows,
        row_bytes,
        ple.cache_bytes,
        &PleConfig::DEFAULT,
    )?;
    Ok(PleTable::Streamed {
        stream: Arc::new(stream),
        row_dtype: DType::Q6_K,
    })
}

/// Holds the per-layer table for an MLX checkpoint.
fn open_table_mlx(
    backend: &Arc<dyn Backend>,
    st: &SafetensorsFile,
    path: &Path,
    config: &Gemma4Config,
    ple: PleOptions,
) -> CoreResult<PleTable> {
    let triple = st
        .affine_triple("per_layer_token_embd.weight")
        .map_err(open_failure)?
        .ok_or_else(|| CoreError::InvalidValue {
            key: "per_layer_token_embd",
            value: "missing affine triple for per_layer_token_embd".into(),
        })?;

    let n_layers = config.shape().n_layers();
    let rows = u64::from(config.ple_vocab());
    let row_bytes = ple_repack::ROW_BYTES;

    let sidecar_path = ple_repack::ensure_affine_sidecar(path, triple)?;

    if ple.resident {
        let mut file = File::open(&sidecar_path).map_err(CoreError::Io)?;
        file.seek(SeekFrom::Start(ple_repack::HEADER_BYTES as u64))
            .map_err(CoreError::Io)?;
        let total_bytes = rows * u64::from(row_bytes);
        let buf = alloc(backend, total_bytes)?;
        let chunk_size = 64 * 1024 * 1024usize;
        let mut chunk = vec![0u8; chunk_size];
        let mut offset = 0u64;
        while offset < total_bytes {
            let to_read =
                usize::try_from((total_bytes - offset).min(chunk_size as u64)).expect("fits usize");
            file.read_exact(&mut chunk[..to_read])
                .map_err(CoreError::Io)?;
            let view = buf.view().slice(offset, to_read as u64)?;
            backend.copy_h2d(view, &chunk[..to_read])?;
            offset += to_read as u64;
        }
        let table = Weight::from_buffer(
            DType::MlxAffine4Gs64,
            256 * n_layers,
            config.ple_vocab(),
            buf,
        );
        return Ok(PleTable::Resident { table });
    }
    let shard = File::open(&sidecar_path).map_err(CoreError::Io)?;
    let stream = PleStream::open_direct(
        shard,
        ple_repack::HEADER_BYTES as u64,
        rows,
        row_bytes,
        ple.cache_bytes,
        &PleConfig::DEFAULT,
    )
    .map_err(|e| CoreError::Backend {
        backend: "ple",
        op: "open_direct",
        source: Box::new(e),
    })?;

    Ok(PleTable::Streamed {
        stream: Arc::new(stream),
        row_dtype: DType::MlxAffine4Gs64,
    })
}

/// Moves every bound tensor from an MLX checkpoint to the device.
fn import_weights_mlx(
    backend: &Arc<dyn Backend>,
    st: &SafetensorsFile,
    binding: &Gemma4DescBinding,
    shape: &HybridShape,
) -> CoreResult<Weights> {
    let mut layers = Vec::with_capacity(binding.layers.len());
    for (slot, layer) in binding.layers.iter().enumerate() {
        let ordinal = u32::try_from(slot).map_err(|_| CoreError::ShapeMismatch {
            field: "layer slot index",
            expected: u64::from(u32::MAX),
            got: u64::try_from(slot).unwrap_or(u64::MAX),
        })?;
        let kv_rank = shape.kv_owner_rank(LayerIdx::new(ordinal));
        layers.push(LayerWeights {
            input_norm: vector_mlx(backend, st, &layer.input_norm)?,
            post_attn_norm: Some(vector_mlx(backend, st, &layer.post_attn_norm)?),
            ffn_norm: vector_mlx(backend, st, &layer.pre_ffn_norm)?,
            post_ffn_norm: Some(vector_mlx(backend, st, &layer.post_ffn_norm)?),
            ffn: Ffn {
                gate: weight_mlx(backend, st, &layer.ffn.gate)?,
                up: weight_mlx(backend, st, &layer.ffn.up)?,
                down: weight_mlx(backend, st, &layer.ffn.down)?,
            },
            block: Block::Attn(import_attn_mlx(backend, st, layer)?),
            // A dense feed-forward: this family's slots have no router.
            moe: None,
            ordinal,
            kv_rank,
            layer_scalar: scalar_mlx(st, &layer.layer_scale)?,
            ple: Some(PerLayerWeights {
                gate: weight_mlx(backend, st, &layer.ple.gate)?,
                proj: weight_mlx(backend, st, &layer.ple.proj)?,
                post_norm: vector_mlx(backend, st, &layer.ple.post_norm)?,
            }),
        });
    }
    Ok(Weights {
        token_embd: weight_mlx(backend, st, &binding.token_embd)?,
        output_norm: vector_mlx(backend, st, &binding.output_norm)?,
        lm_head: None,
        ple_model_proj: Some(weight_mlx(backend, st, &binding.ple_model_proj)?),
        ple_proj_norm: Some(vector_mlx(backend, st, &binding.ple_proj_norm)?),
        layers,
    })
}

fn import_attn_mlx(
    backend: &Arc<dyn Backend>,
    st: &SafetensorsFile,
    layer: &Gemma4LayerRoles<TensorDesc>,
) -> CoreResult<Attn> {
    let attn = &layer.attn;
    Ok(Attn {
        q: weight_mlx(backend, st, &attn.q)?,
        k: attn
            .k
            .as_ref()
            .map(|desc| weight_mlx(backend, st, desc))
            .transpose()?,
        v: attn
            .v
            .as_ref()
            .map(|desc| weight_mlx(backend, st, desc))
            .transpose()?,
        output: weight_mlx(backend, st, &attn.output)?,
        q_norm: vector_mlx(backend, st, &attn.q_norm)?,
        k_norm: attn
            .k_norm
            .as_ref()
            .map(|desc| vector_mlx(backend, st, desc))
            .transpose()?,
    })
}

fn weight_mlx(
    backend: &Arc<dyn Backend>,
    st: &SafetensorsFile,
    desc: &TensorDesc,
) -> CoreResult<Weight> {
    let dims = desc.shape.dims();
    let in_dim = extent(dims.first().copied().unwrap_or(0))?;
    let out_dim = extent(dims.get(1).copied().unwrap_or(1))?;

    if let Some(triple) = st.affine_triple(&desc.name).map_err(open_failure)? {
        let bits = st.bits_for(&desc.name);
        let dtype = match bits {
            4 => DType::MlxAffine4Gs64,
            8 => DType::MlxAffine8Gs64,
            _ => {
                return Err(CoreError::Unsupported {
                    what: "unsupported affine bit width",
                    backend: "gemma4",
                });
            }
        };
        Weight::import_affine_slices(
            backend,
            dtype,
            in_dim,
            out_dim,
            triple.words,
            triple.scales,
            triple.biases,
        )
    } else {
        let bytes = st.tensor_bytes(&desc.name).map_err(open_failure)?;
        let buf = alloc(backend, u64::try_from(bytes.len()).unwrap_or(0))?;
        backend.copy_h2d(buf.view(), &bytes)?;
        Ok(Weight::from_buffer(DType::F32, in_dim, out_dim, buf))
    }
}

fn vector_mlx(
    backend: &Arc<dyn Backend>,
    st: &SafetensorsFile,
    desc: &TensorDesc,
) -> CoreResult<DeviceBuffer> {
    let bytes = st.tensor_bytes(&desc.name).map_err(open_failure)?;
    let buf = alloc(backend, u64::try_from(bytes.len()).unwrap_or(0))?;
    backend.copy_h2d(buf.view(), &bytes)?;
    Ok(buf)
}

fn scalar_mlx(st: &SafetensorsFile, desc: &TensorDesc) -> CoreResult<f32> {
    let bytes = st.tensor_bytes(&desc.name).map_err(open_failure)?;
    if bytes.len() < 4 {
        return Err(CoreError::ShapeMismatch {
            field: "scalar bytes",
            expected: 4,
            got: bytes.len() as u64,
        });
    }
    Ok(f32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes")))
}

/// Moves every bound tensor but the per-layer table to the device.
fn import_weights(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    binding: &Gemma4DescBinding,
    shape: &HybridShape,
) -> CoreResult<Weights> {
    let mut layers = Vec::with_capacity(binding.layers.len());
    for (slot, layer) in binding.layers.iter().enumerate() {
        // Named rather than clamped: a slot index that does not fit is a layer
        // list no geometry in this family describes, and saturating it would
        // hand every slot past the limit the same ordinal and the same cache
        // rank - forty-two slots reading one slot's runs, with nothing said.
        let ordinal = u32::try_from(slot).map_err(|_| CoreError::ShapeMismatch {
            field: "layer slot index",
            expected: u64::from(u32::MAX),
            got: u64::try_from(slot).unwrap_or(u64::MAX),
        })?;
        // Every slot of this family holds attention, so the ordinal and the slot
        // index are the same number. The cache rank is NOT: eighteen slots read
        // an earlier slot's runs, so from the first of them onward a slot's own
        // index and the rank of the runs it reads part company. The shape is the
        // only thing that knows which.
        let kv_rank = shape.kv_owner_rank(LayerIdx::new(ordinal));
        layers.push(LayerWeights {
            input_norm: vector(backend, file, &layer.input_norm)?,
            post_attn_norm: Some(vector(backend, file, &layer.post_attn_norm)?),
            // The file's `blk.N.ffn_norm` is the norm the feed-forward READS,
            // which is what the engine's `ffn_norm` is. The two spellings agree
            // here and do not in every family, which is why the engine field is
            // named after the role rather than after a file.
            ffn_norm: vector(backend, file, &layer.pre_ffn_norm)?,
            post_ffn_norm: Some(vector(backend, file, &layer.post_ffn_norm)?),
            ffn: Ffn {
                gate: weight(backend, file, &layer.ffn.gate)?,
                up: weight(backend, file, &layer.ffn.up)?,
                down: weight(backend, file, &layer.ffn.down)?,
            },
            block: Block::Attn(import_attn(backend, file, layer)?),
            // A dense feed-forward: this family's slots have no router.
            moe: None,
            ordinal,
            kv_rank,
            layer_scalar: scalar(file, &layer.layer_scale)?,
            ple: Some(PerLayerWeights {
                gate: weight(backend, file, &layer.ple.gate)?,
                proj: weight(backend, file, &layer.ple.proj)?,
                post_norm: vector(backend, file, &layer.ple.post_norm)?,
            }),
        });
    }
    Ok(Weights {
        token_embd: weight(backend, file, &binding.token_embd)?,
        output_norm: vector(backend, file, &binding.output_norm)?,
        // This family ties the output projection to the embedding, and the
        // binding refused a container carrying one of its own.
        lm_head: None,
        ple_model_proj: Some(weight(backend, file, &binding.ple_model_proj)?),
        ple_proj_norm: Some(vector(backend, file, &binding.ple_proj_norm)?),
        layers,
    })
}

/// Moves one slot's attention weights to the device.
///
/// The three key-half roles arrive together or not at all: a slot that reads an
/// earlier slot's runs projects no key and no value and normalizes no key, and
/// the binding already refused a container that shipped them anyway. Passing the
/// `Option`s straight through is therefore not a shrug - it is the geometry,
/// carried by the type.
fn import_attn(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    layer: &Gemma4LayerRoles<TensorDesc>,
) -> CoreResult<Attn> {
    let attn = &layer.attn;
    Ok(Attn {
        q: weight(backend, file, &attn.q)?,
        k: attn
            .k
            .as_ref()
            .map(|desc| weight(backend, file, desc))
            .transpose()?,
        v: attn
            .v
            .as_ref()
            .map(|desc| weight(backend, file, desc))
            .transpose()?,
        output: weight(backend, file, &attn.output)?,
        q_norm: vector(backend, file, &attn.q_norm)?,
        k_norm: attn
            .k_norm
            .as_ref()
            .map(|desc| vector(backend, file, desc))
            .transpose()?,
    })
}

/// Imports one projection matrix, refusing a format this family does not ship.
///
/// Two checks, and they are not the same one twice. [`WEIGHT_TYPES`] is the
/// family's own: these are the formats a checkpoint of this family is expected
/// to store, and anything else means the file is not what it claims. The engine
/// then refuses, separately, a format for which no matvec exists in this build.
fn weight(backend: &Arc<dyn Backend>, file: &ModelFile, desc: &TensorDesc) -> CoreResult<Weight> {
    if !WEIGHT_TYPES.contains(&desc.dtype) {
        return Err(CoreError::Unsupported {
            what: "a weight stored in a format this family's loader does not accept",
            backend: "gemma4",
        });
    }
    let dims = desc.shape.dims();
    let in_dim = extent(dims.first().copied().unwrap_or(0))?;
    let out_dim = extent(dims.get(1).copied().unwrap_or(1))?;
    let (host, range) = file.tensor_host(desc).map_err(open_failure)?;
    Weight::import(backend, desc.dtype, in_dim, out_dim, host, range)
}

/// Imports one weight vector, which every format stores as plain `f32`.
fn vector(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    desc: &TensorDesc,
) -> CoreResult<DeviceBuffer> {
    if !VECTOR_TYPES.contains(&desc.dtype) {
        return Err(CoreError::Unsupported {
            what: "a norm or control vector stored as anything but single precision",
            backend: "gemma4",
        });
    }
    let (host, range) = file.tensor_host(desc).map_err(open_failure)?;
    import_host(backend, host, range)
}

/// Reads a one-element tensor as the host float it is.
///
/// The per-slot output scalar is multiplied into the residual sum by the norm
/// kernel, which takes it as a launch argument rather than as a buffer: it is
/// one number per slot for the whole run, and a device allocation per slot to
/// hold four bytes would be forty-two allocations the kernel would then have to
/// dereference.
fn scalar(file: &ModelFile, desc: &TensorDesc) -> CoreResult<f32> {
    if !VECTOR_TYPES.contains(&desc.dtype) {
        return Err(CoreError::Unsupported {
            what: "a per-layer output scalar stored as anything but single precision",
            backend: "gemma4",
        });
    }
    let bytes = file.tensor_bytes(&desc.name).map_err(open_failure)?;
    let head: [u8; 4] =
        bytes
            .get(..4)
            .and_then(|head| head.try_into().ok())
            .ok_or(CoreError::ShapeMismatch {
                field: "per-layer output scalar bytes",
                expected: 4,
                got: desc.bytes,
            })?;
    Ok(f32::from_le_bytes(head))
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
        backend: "gemma4",
        op: "load",
        source: Box::new(source),
    }
}

impl Model for Gemma4Model {
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

    // The media path is bound: this family ships its tower in
    // `crate::gemma4::Gemma4Tower`, and the stack splices its rows. The audio
    // tower beside it in the same container stays unbound; that is a different
    // trait and a different phase.
    fn media_mut(&mut self) -> Option<&mut dyn MediaModel> {
        Some(self)
    }

    fn teardown(&mut self) -> CoreResult<()> {
        // The reader's counters die with the reader, so this is the last moment
        // anything can say what the streamed table cost: how many rows a step
        // asked for, how many of them the cache already had, and how long the
        // reads that were left took. A run that wants the number reads it here
        // rather than instrumenting the decode loop, which would time the
        // instrument.
        if let Some(stats) = self.stack.ple_stats() {
            tracing::info!(
                lookups = stats.lookups,
                hits = stats.hits,
                misses = stats.misses,
                reads = stats.reads,
                read_bytes = stats.read_bytes,
                evictions = stats.evictions,
                read_seconds = stats.read_seconds,
                "per-layer reader"
            );
        }
        self.stack.teardown()
    }
}

/// The media path, which this family has because its stack carries one and
/// its tower now exists: the same single-call contract the Qwen family's
/// implementation states, for the same reason.
impl MediaModel for Gemma4Model {
    fn prefill_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>> {
        self.stack.prefill_last_media(session, tokens, media)
    }
}

#[cfg(test)]
mod tests {
    use super::{PleOptions, ple_row_bytes, ple_row_elems};
    use shiso_core::DType;

    /// Layer slots the E4B stack holds, which is what the two widths scale with.
    const E4B_LAYERS: u32 = 42;

    #[test]
    fn the_default_holding_streams_the_table_behind_a_half_gigabyte_of_cache() {
        let opts = PleOptions::default();
        assert!(!opts.resident);
        assert_eq!(opts.cache_bytes, 512 << 20);
    }

    #[test]
    fn one_row_is_one_block_of_the_tables_format_per_layer_slot() {
        // The two numbers the reader and the prepare kernel each derive on their
        // own, stated here as the format's block numbers times the slot count so
        // that a widths change shows up as one edit rather than as two figures
        // that have to be kept in step.
        let dtype = DType::Q6_K;
        let block_bytes = u64::from(dtype.block_bytes().get());
        let block_elems = dtype.block_elems().get();
        let bytes = ple_row_bytes(dtype, E4B_LAYERS);
        let elems = ple_row_elems(dtype, E4B_LAYERS);
        assert_eq!(bytes, u64::from(E4B_LAYERS) * block_bytes);
        assert_eq!(elems, E4B_LAYERS * block_elems);

        // The relation the two have to keep whatever the numbers are: the row's
        // bytes are exactly what its elements occupy in the format the table is
        // stored in. If that ever parted, the staging the kernel strides would
        // not be the buffer the reader filled.
        assert_eq!(bytes, u64::from(elems / block_elems) * block_bytes);
        // One block per slot, so a stack one slot shorter is one block shorter -
        // which is the statement a flat constant could not make.
        assert_eq!(
            bytes - ple_row_bytes(dtype, E4B_LAYERS - 1),
            block_bytes,
            "the row does not scale one block per slot"
        );
        assert_eq!(elems - ple_row_elems(dtype, E4B_LAYERS - 1), block_elems);
    }
}

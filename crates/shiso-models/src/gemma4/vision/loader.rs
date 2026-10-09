//! Reading a Gemma 4 vision tower out of its mmproj container.
//!
//! The file is a multimodal projector GGUF: one container holding a SigLIP-style
//! vision tower under `v.*`, its linear projection into the language model under
//! `mm.*`, and — for this checkpoint — an audio tower beside it under `a.*` and
//! `mm.a.*`. Only the vision half is read here; the audio tensors are left in
//! the file, which is why there is no tensor-count contract: the loader asks for
//! each tensor it needs by name and refuses the file the moment one is missing.
//!
//! # The two widths the kernels read
//!
//! Activations are f32 end to end; the linear weights stay brain-float and the
//! GEMM widens them on read, exactly as the Qwen tower does. That fixes what
//! each stored format means: a BF16 tensor is uploaded as its bits, and an F32
//! tensor is rounded to nearest-even and uploaded as the result. The shipped
//! file stores the linears BF16 and the norms, the patch convolution and the
//! position table F32, so both paths run on the real checkpoint — the patch
//! projection is F32 in the file and BF16 on the device, and the rounding is
//! part of what the encoder's host reference reproduces.
//!
//! A quantized projector stores its linears as `Q8_0` blocks instead. Those are
//! uploaded exactly as stored and multiplied in that form - the GEMM reads the
//! blocks - so a `Q8_0` file costs the device half what a BF16 one does; its
//! norms, patch convolution and position table are F32 like the shipped file's.
//!
//! # The one tensor that is rearranged, not converted
//!
//! `v.patch_embd.weight` is a convolution kernel — `[x][y][channel][out]` with
//! the channel SLOWEST within an output row — while the preprocessor lays patch
//! rows out with the channel FASTEST (`y*48 + x*3 + c`). The GEMM this tower
//! uses reads weight rows in the preprocessor's order, so the row is transposed
//! at load. The element count does not change, which is exactly why a shape
//! check alone could never catch it: a skipped transpose loads cleanly and
//! answers confidently about a rearranged picture.

use std::path::Path;
use std::sync::Arc;

use shiso_core::{Backend, DType, DeviceBuffer, alloc};

use crate::audio_ops::GemmType;
use shiso_modelfile::safetensors::SafetensorsFile;
use shiso_modelfile::{GgufFile, NameMap, Role, TensorSource};
use shiso_vision::{VisionError, open_gguf};

/// Formats this loader moves to a device for anything but a linear's weight:
/// the norms, the patch convolution and the position table are read as
/// floats, and a block-quantized one would have to be decoded on the host.
const UPLOADABLE: &[DType] = &[DType::F32, DType::BF16];

/// Formats a linear's weight may be stored in: the two float widths, and
/// `Q8_0`, which a quantized projector stores its linears in and which the
/// tower's GEMM multiplies as it is, so it is uploaded unconverted.
const LINEAR_UPLOADABLE: &[DType] = &[DType::F32, DType::BF16, DType::Q8_0];

/// The architecture string a Gemma 4 mmproj declares.
const ARCH: &str = "clip";
/// The projector type that says the vision half is Gemma 4's.
const PROJECTOR: &str = "gemma4v";

/// Base of the rotation's frequency ladder.
///
/// Not in the container: no `clip.vision.rope_theta` key exists, and the
/// reference builds the ladder from its own constant. It is the number the
/// sidecar's `config.json` states for the vision tower, so it is a family fact
/// rather than a guess, and it is named here for the same reason the epsilon
/// default is.
pub(super) const ROPE_BASE: f32 = 100.0;

/// What the two-axis rotation's pairing needs from a head width: the head
/// splits in half, and each half splits again into a pair and its partner.
const ROTATION_VEC: usize = 4;

/// The tower's widths and the two scalars its kernels are driven with.
///
/// Every number is read from the container's own metadata; nothing here has a
/// default, because a tower that loads on a guessed width is a tower whose
/// every later shape check agrees with the guess.
#[derive(Debug, Clone, Copy)]
pub(super) struct Gemma4Dims {
    /// Transformer blocks.
    pub(super) depth: usize,
    /// Weights in one residual row.
    pub(super) hidden: usize,
    /// Attention heads.
    pub(super) heads: usize,
    /// Weights in one head: `hidden / heads`, and the width the per-head
    /// normalization and the rotation split.
    pub(super) head_dim: usize,
    /// Weights in one feed-forward row.
    pub(super) intermediate: usize,
    /// Width of the projected rows the language model consumes.
    pub(super) out_hidden: usize,
    /// Values in one patch row: `3 x patch x patch`, a still's single frame.
    pub(super) patch_dim: usize,
    /// Rows in each of the two position tables.
    pub(super) pos_rows: usize,
    /// Side of the patch the preprocessor cuts.
    pub(super) patch: usize,
    /// Epsilon inside every normalization's square root.
    pub(super) eps: f32,
    /// Base of the rotation's frequency ladder.
    pub(super) rope_base: f32,
}

/// The four scalars the reference wraps around one linear projection.
///
/// The container stores them per tensor — `attn_q.input_min` and its three
/// siblings — and a tensor without them is clamped at the float bounds, which
/// is the identity. The reference reaches the same answer by looking the names
/// up and defaulting to `±FLT_MAX`; so does this.
#[derive(Debug, Clone, Copy)]
pub(super) struct Clamps {
    /// Lower bound applied to the projection's INPUT.
    pub(super) input_min: f32,
    /// Upper bound applied to the projection's INPUT.
    pub(super) input_max: f32,
    /// Lower bound applied to the projection's OUTPUT.
    pub(super) output_min: f32,
    /// Upper bound applied to the projection's OUTPUT.
    pub(super) output_max: f32,
}

impl Clamps {
    /// Whether the INPUT bound cannot move a value. The encoder skips the
    /// input clamp when it is the identity, which is the common case: the
    #[expect(clippy::float_cmp, reason = "identity clamps are exact float extremes")]
    #[must_use]
    pub(super) const fn input_is_identity(&self) -> bool {
        self.input_min == f32::MIN && self.input_max == f32::MAX
    }

    #[expect(clippy::float_cmp, reason = "identity clamps are exact float extremes")]
    #[must_use]
    pub(super) const fn output_is_identity(&self) -> bool {
        self.output_min == f32::MIN && self.output_max == f32::MAX
    }
}

/// Clamps that leave values unchanged.
pub(super) const NO_CLAMPS: Clamps = Clamps {
    input_min: f32::MIN,
    input_max: f32::MAX,
    output_min: f32::MIN,
    output_max: f32::MAX,
};

/// One linear projection: its weight rows and the clamps around it.
#[derive(Debug)]
pub(super) struct Gemma4Linear {
    /// The weight, `[out][in]` row-major — the order the GEMM reads — in
    /// [`Gemma4Linear::dtype`]'s storage.
    pub(super) weight: DeviceBuffer,
    /// How the weight is stored: BF16, or `Q8_0` blocks kept as the file
    /// holds them, which the GEMM reads directly.
    pub(super) dtype: GemmType,
    /// The four scalars, read on the host.
    pub(super) clamps: Clamps,
}

/// One transformer block, in the order the encoder walks it.
#[derive(Debug)]
pub(super) struct Gemma4Block {
    /// Pre-attention normalization, f32 `[hidden]`.
    pub(super) ln1: DeviceBuffer,
    /// The query projection.
    pub(super) attn_q: Gemma4Linear,
    /// The key projection.
    pub(super) attn_k: Gemma4Linear,
    /// The value projection.
    pub(super) attn_v: Gemma4Linear,
    /// Per-head weight of the query's normalization, f32 `[head_dim]`.
    pub(super) q_norm: DeviceBuffer,
    /// Per-head weight of the key's normalization, f32 `[head_dim]`.
    pub(super) k_norm: DeviceBuffer,
    /// The attention output projection.
    pub(super) attn_out: Gemma4Linear,
    /// Post-attention normalization, applied to the projection's output before
    /// the residual add, f32 `[hidden]`.
    pub(super) attn_post_norm: DeviceBuffer,
    /// Pre-feed-forward normalization, f32 `[hidden]`.
    pub(super) ln2: DeviceBuffer,
    /// The gate half of the feed-forward.
    pub(super) ffn_gate: Gemma4Linear,
    /// The up half of the feed-forward.
    pub(super) ffn_up: Gemma4Linear,
    /// The down half of the feed-forward.
    pub(super) ffn_down: Gemma4Linear,
    /// Post-feed-forward normalization, f32 `[hidden]`.
    pub(super) ffn_post_norm: DeviceBuffer,
}

/// A Gemma 4 vision tower, on the device.
#[derive(Debug)]
pub(super) struct Gemma4VisionWeights {
    pub(super) dims: Gemma4Dims,
    /// The patch convolution, transposed to the preprocessor's row order and
    /// narrowed to BF16: `[hidden][patch_dim]`.
    pub(super) patch: DeviceBuffer,
    /// The position table's column half, host-resident f32
    /// `[pos_rows][hidden]`. The encoder sums the two tables' rows on the host
    /// and uploads one buffer; the note in the encoder's header says why.
    pub(super) position_x: Vec<f32>,
    /// The position table's row half.
    pub(super) position_y: Vec<f32>,
    /// The transformer blocks.
    pub(super) blocks: Vec<Gemma4Block>,
    /// The final projection into the language model, `[out_hidden][hidden]`.
    pub(super) projection: Gemma4Linear,
    /// F32 ones, `[hidden]`: the weight of the final unweighted normalization,
    /// which is the only way that kernel expresses "no weight".
    pub(super) ones: DeviceBuffer,
}

impl Gemma4VisionWeights {
    /// The widths the encoder runs on.
    #[must_use]
    pub(super) const fn dims(&self) -> Gemma4Dims {
        self.dims
    }

    /// Device bytes the tower occupies.
    #[must_use]
    pub(super) fn device_bytes(&self) -> u64 {
        let linear = |l: &Gemma4Linear| l.weight.len();
        let block = |b: &Gemma4Block| {
            b.ln1.len()
                + b.ln2.len()
                + b.q_norm.len()
                + b.k_norm.len()
                + b.attn_post_norm.len()
                + b.ffn_post_norm.len()
                + linear(&b.attn_q)
                + linear(&b.attn_k)
                + linear(&b.attn_v)
                + linear(&b.attn_out)
                + linear(&b.ffn_gate)
                + linear(&b.ffn_up)
                + linear(&b.ffn_down)
        };
        self.patch.len()
            + self.ones.len()
            + linear(&self.projection)
            + self.blocks.iter().map(block).sum::<u64>()
    }
}

/// Loads the Gemma 4 vision tower onto `backend`: the mmproj GGUF at `path`, or
/// the tower shards of the checkpoint directory `path` names.
pub(super) fn load_gemma4_weights(
    backend: &Arc<dyn Backend>,
    path: &Path,
) -> Result<Gemma4VisionWeights, VisionError> {
    if path.is_dir() {
        return load_gemma4_weights_mlx(backend, path);
    }
    let file = open_gguf(path)?;
    check_projector(&file, path)?;
    let mut dims = read_dims(&file)?;

    let patch_embd = load_patch(backend, &file, &dims)?;
    let (position_x, position_y) = load_positions(&file, &mut dims)?;
    let mut blocks = Vec::with_capacity(dims.depth);
    for i in 0..dims.depth {
        blocks.push(load_block(backend, &file, &dims, i)?);
    }
    let projection = load_linear(
        backend,
        &file,
        "mm.input_projection",
        dims.out_hidden,
        dims.hidden,
    )?;
    let ones = upload(
        backend,
        "pool norm weight (ones)",
        DType::F32,
        &f32_bytes(&vec![1.0f32; dims.hidden]),
    )?;
    Ok(Gemma4VisionWeights {
        dims,
        patch: patch_embd,
        position_x,
        position_y,
        blocks,
        projection,
        ones,
    })
}

/// Refuses a file that is not a Gemma 4 vision projector, by name.
fn check_projector(file: &GgufFile, path: &Path) -> Result<(), VisionError> {
    let meta = file.metadata();
    if meta.architecture() != Some(ARCH) {
        return Err(VisionError::Unsupported {
            what: "this mmproj is not a clip-architecture vision projector",
        });
    }
    let key = format!("{ARCH}.vision.projector_type");
    if meta.get_str(&key) != Some(PROJECTOR) {
        return Err(VisionError::Unsupported {
            what: "this clip mmproj's vision tower is not a gemma4v projector",
        });
    }
    let _ = path;
    Ok(())
}

/// Derives the tower's widths from its own metadata.
fn read_dims(file: &GgufFile) -> Result<Gemma4Dims, VisionError> {
    let meta = file.metadata();
    let read = |suffix: &str| -> Result<usize, VisionError> {
        let key = format!("{ARCH}.vision.{suffix}");
        let value = meta
            .get_u32(&key)
            .ok_or_else(|| VisionError::TowerMetadata { key: key.clone() })?;
        usize::try_from(value).map_err(|_| VisionError::TowerMetadata { key })
    };
    let hidden = read("embedding_length")?;
    if hidden == 0 || hidden % 4 != 0 {
        // The generic normalization walks float4s.
        return Err(VisionError::TowerNotDivisible {
            field: "embedding_length",
            value: hidden,
            divisor_field: "the normalization's float4 stride",
            divisor: 4,
        });
    }
    let heads = read("attention.head_count")?;
    let head_dim = if hidden % heads == 0 {
        hidden / heads
    } else {
        return Err(VisionError::TowerNotDivisible {
            field: "embedding_length",
            value: hidden,
            divisor_field: "attention.head_count",
            divisor: heads,
        });
    };
    if head_dim == 0 || head_dim % ROTATION_VEC != 0 {
        return Err(VisionError::TowerNotDivisible {
            field: "head_dim",
            value: head_dim,
            divisor_field: "the rotation's two halves and their pairs",
            divisor: ROTATION_VEC,
        });
    }
    let patch = read("patch_size")?;
    let eps = meta
        .get_f32(&format!("{ARCH}.vision.attention.layer_norm_epsilon"))
        .ok_or_else(|| VisionError::TowerMetadata {
            key: format!("{ARCH}.vision.attention.layer_norm_epsilon"),
        })?;
    Ok(Gemma4Dims {
        depth: read("block_count")?,
        hidden,
        heads,
        head_dim,
        intermediate: read("feed_forward_length")?,
        out_hidden: read("projection_dim")?,
        patch_dim: 3 * patch * patch,
        // The table's own extent is a fact of the tensor, not of the metadata;
        // `load_positions` fills this in when it has checked the shape.
        pos_rows: 0,
        patch,
        eps,
        rope_base: ROPE_BASE,
    })
}

/// Narrows a config scalar, which the document writes at double precision and
/// every kernel here reads at single.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the tower's epsilon and rope base are f32 everywhere downstream"
)]
fn narrow(value: f64) -> f32 {
    value as f32
}

/// The tower's geometry as the checkpoint directory's `config.json` states it.
///
/// The same numbers [`read_dims`] takes from the mmproj's metadata, from the
/// other container's own document. Writing them down here instead would make
/// this loader agree with one checkpoint and disagree silently with the next:
/// a tower loaded at the wrong depth or the wrong head count produces rows of
/// the right shape and the wrong content, which nothing downstream can catch.
fn read_dims_mlx(dir: &Path) -> Result<Gemma4Dims, VisionError> {
    let path = dir.join("config.json");
    let missing = |key: &str| VisionError::TowerMetadata { key: key.into() };
    let text = std::fs::read_to_string(&path).map_err(|_| missing("config.json"))?;
    let document: serde_json::Value =
        serde_json::from_str(&text).map_err(|_| missing("config.json"))?;
    let vision = document
        .get("vision_config")
        .ok_or_else(|| missing("vision_config"))?;
    let read = |key: &str| -> Result<usize, VisionError> {
        vision
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| missing(&format!("vision_config.{key}")))
    };

    let hidden = read("hidden_size")?;
    if hidden == 0 || hidden % 4 != 0 {
        // The generic normalization walks float4s.
        return Err(VisionError::TowerNotDivisible {
            field: "hidden_size",
            value: hidden,
            divisor_field: "the normalization's float4 stride",
            divisor: 4,
        });
    }
    let heads = read("num_attention_heads")?;
    if heads == 0 || hidden % heads != 0 {
        return Err(VisionError::TowerNotDivisible {
            field: "hidden_size",
            value: hidden,
            divisor_field: "num_attention_heads",
            divisor: heads,
        });
    }
    let head_dim = hidden / heads;
    if head_dim == 0 || head_dim % ROTATION_VEC != 0 {
        return Err(VisionError::TowerNotDivisible {
            field: "head_dim",
            value: head_dim,
            divisor_field: "the rotation's two halves and their pairs",
            divisor: ROTATION_VEC,
        });
    }
    let patch_side = read("patch_size")?;
    // The projected width is the language model's, not the tower's, so it comes
    // from the other half of the same document.
    let out_hidden = document
        .get("text_config")
        .and_then(|text| text.get("hidden_size"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| missing("text_config.hidden_size"))?;
    let eps = vision
        .get("rms_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| missing("vision_config.rms_norm_eps"))?;
    let rope_base = vision
        .get("rope_parameters")
        .and_then(|rope| rope.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| missing("vision_config.rope_parameters.rope_theta"))?;

    Ok(Gemma4Dims {
        depth: read("num_hidden_layers")?,
        hidden,
        heads,
        head_dim,
        intermediate: read("intermediate_size")?,
        out_hidden,
        patch_dim: 3 * patch_side * patch_side,
        pos_rows: read("position_embedding_size")?,
        patch: patch_side,
        eps: narrow(eps),
        rope_base: narrow(rope_base),
    })
}

/// One tensor's payload, with its name, format and shape all checked.
fn tensor<'a>(
    file: &'a GgufFile,
    name: &str,
    dims: &[usize],
) -> Result<shiso_modelfile::TensorView<'a>, VisionError> {
    tensor_in(file, name, dims, UPLOADABLE)
}

/// [`tensor`], accepting the formats in `allowed`.
fn tensor_in<'a>(
    file: &'a GgufFile,
    name: &str,
    dims: &[usize],
    allowed: &[DType],
) -> Result<shiso_modelfile::TensorView<'a>, VisionError> {
    let view = file.get(name).ok_or_else(|| VisionError::MissingTensor {
        name: name.to_owned(),
    })?;
    if !allowed.contains(&view.dtype) {
        return Err(VisionError::TensorDType {
            name: name.to_owned(),
            dtype: view.dtype,
        });
    }
    let declared: Vec<u64> = view.shape.dims().to_vec();
    let expected: Vec<u64> = dims
        .iter()
        .map(|d| u64::try_from(*d).unwrap_or(u64::MAX))
        .collect();
    // The container writes the fastest-varying axis first, so a tensor the
    // tower reads as `[slowest..fastest]` is declared in reverse. Callers
    // pass the DECLARED order; the comparison is then exact.
    if declared != expected {
        return Err(VisionError::TensorShape {
            name: name.to_owned(),
            declared,
            expected,
        });
    }
    Ok(view)
}

/// The four clamp scalars beside one linear, or the identity pair.
fn read_clamps(file: &GgufFile, name: &str) -> Result<Clamps, VisionError> {
    let scalar = |suffix: &str| -> Result<f32, VisionError> {
        let full = format!("{name}.{suffix}");
        let Some(view) = file.get(&full) else {
            return Ok(match suffix {
                "input_max" | "output_max" => f32::MAX,
                _ => f32::MIN,
            });
        };
        let bytes = view.data;
        if bytes.len() < 4 || view.dtype != DType::F32 {
            return Err(VisionError::TensorDType {
                name: full,
                dtype: view.dtype,
            });
        }
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    };
    Ok(Clamps {
        input_min: scalar("input_min")?,
        input_max: scalar("input_max")?,
        output_min: scalar("output_min")?,
        output_max: scalar("output_max")?,
    })
}

/// Reads one linear projection: its weight rows - BF16, or `Q8_0` blocks as
/// stored - and its four scalars.
///
/// The scalars hang off the projection's own name and not off its weight's -
/// `v.blk.0.attn_q.input_min`, beside `v.blk.0.attn_q.weight` - so `prefix` is
/// what [`read_clamps`] is given. Passing the weight's name instead looks up
/// four tensors no container has, finds nothing, and takes the absent-scalar
/// path, which is the identity: the tower then loads, runs, and answers without
/// the clipping the checkpoint asks for.
fn load_linear(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    prefix: &str,
    out: usize,
    inp: usize,
) -> Result<Gemma4Linear, VisionError> {
    let name = format!("{prefix}.weight");
    let view = tensor_in(file, &name, &[inp, out], LINEAR_UPLOADABLE)?;
    if view.dtype == DType::Q8_0 {
        // A row is a whole number of 32-element blocks or it is not a row
        // the GEMM can address; the shape check above has the element count.
        if !inp.is_multiple_of(32) || view.data.len() != out * inp / 32 * 34 {
            return Err(VisionError::TensorDType {
                name,
                dtype: view.dtype,
            });
        }
        return Ok(Gemma4Linear {
            weight: upload(backend, &name, DType::Q8_0, view.data)?,
            dtype: GemmType::Q8_0,
            clamps: read_clamps(file, prefix)?,
        });
    }
    let bytes = widen_or_narrow(view.data, view.dtype, DType::BF16, &name)?;
    Ok(Gemma4Linear {
        weight: upload(backend, &name, DType::BF16, &bytes)?,
        dtype: GemmType::Bf16,
        clamps: read_clamps(file, prefix)?,
    })
}

/// Reads one normalization weight as f32: the norms are F32 in the shipped
/// file, and the kernels that consume them read f32, so a BF16 norm is widened
/// rather than the kernels being narrowed.
fn load_norm(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    name: &str,
    width: usize,
) -> Result<DeviceBuffer, VisionError> {
    let view = tensor(file, name, &[width])?;
    let bytes = widen_or_narrow(view.data, view.dtype, DType::F32, name)?;
    upload(backend, name, DType::F32, &bytes)
}

/// One block, by its index.
fn load_block(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    dims: &Gemma4Dims,
    i: usize,
) -> Result<Gemma4Block, VisionError> {
    let at = |suffix: &str| format!("v.blk.{i}.{suffix}");
    let h = dims.hidden;
    let hd = dims.head_dim;
    Ok(Gemma4Block {
        ln1: load_norm(backend, file, &at("ln1.weight"), h)?,
        attn_q: load_linear(backend, file, &at("attn_q"), h, h)?,
        attn_k: load_linear(backend, file, &at("attn_k"), h, h)?,
        attn_v: load_linear(backend, file, &at("attn_v"), h, h)?,
        q_norm: load_norm(backend, file, &at("attn_q_norm.weight"), hd)?,
        k_norm: load_norm(backend, file, &at("attn_k_norm.weight"), hd)?,
        attn_out: load_linear(backend, file, &at("attn_out"), h, h)?,
        attn_post_norm: load_norm(backend, file, &at("attn_post_norm.weight"), h)?,
        ln2: load_norm(backend, file, &at("ln2.weight"), h)?,
        ffn_gate: load_linear(backend, file, &at("ffn_gate"), dims.intermediate, h)?,
        ffn_up: load_linear(backend, file, &at("ffn_up"), dims.intermediate, h)?,
        ffn_down: load_linear(backend, file, &at("ffn_down"), h, dims.intermediate)?,
        ffn_post_norm: load_norm(backend, file, &at("ffn_post_norm.weight"), h)?,
    })
}

/// The patch convolution, transposed from the checkpoint's channel-slowest row
/// to the preprocessor's channel-fastest one and narrowed to BF16.
///
/// The stored row of output channel `o` is indexed `c*patch^2 + y*patch + x`;
/// the row the GEMM consumes is indexed `y*3*patch + x*3 + c`. Same elements,
/// same count, different order — the transpose is the whole of this function.
fn load_patch(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    dims: &Gemma4Dims,
) -> Result<DeviceBuffer, VisionError> {
    let name = "v.patch_embd.weight";
    let p = dims.patch;
    let view = tensor(file, name, &[p, p, 3, dims.hidden])?;
    let src = read_f32_or_bf16(view.data, view.dtype)?;
    let mut packed = vec![0u16; dims.hidden * dims.patch_dim];
    for o in 0..dims.hidden {
        for c in 0..3usize {
            for y in 0..p {
                for x in 0..p {
                    let from = o * dims.patch_dim + c * p * p + y * p + x;
                    let to = o * dims.patch_dim + y * p * 3 + x * 3 + c;
                    packed[to] = bf16_bits(src[from]);
                }
            }
        }
    }
    let bytes: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
    upload(backend, name, DType::BF16, &bytes)
}

/// The position table, kept on the host as its two halves.
///
/// The reference gathers rows of the two tables and adds them; this tower
/// builds the same `[n][hidden]` sum on the host per item and adds it once.
/// The table is `[hidden][pos_rows][2]` in the file's order — 63 MB of f32 for
/// the shipped checkpoint — and the two halves are exactly the reference's
/// `tbl_x` and `tbl_y`.
fn load_positions(
    file: &GgufFile,
    dims: &mut Gemma4Dims,
) -> Result<(Vec<f32>, Vec<f32>), VisionError> {
    let name = "v.position_embd.weight";
    let view = file.get(name).ok_or_else(|| VisionError::MissingTensor {
        name: name.to_owned(),
    })?;
    if !UPLOADABLE.contains(&view.dtype) {
        return Err(VisionError::TensorDType {
            name: name.to_owned(),
            dtype: view.dtype,
        });
    }
    let declared = view.shape.dims();
    let (Some(&hidden), Some(&rows), Some(&tables)) =
        (declared.first(), declared.get(1), declared.get(2))
    else {
        return Err(VisionError::TensorShape {
            name: name.to_owned(),
            declared: declared.to_vec(),
            expected: vec![u64::try_from(dims.hidden).unwrap_or(u64::MAX), 0, 2],
        });
    };
    if hidden != u64::try_from(dims.hidden).unwrap_or(u64::MAX) || tables != 2 {
        return Err(VisionError::TensorShape {
            name: name.to_owned(),
            declared: declared.to_vec(),
            expected: vec![u64::try_from(dims.hidden).unwrap_or(u64::MAX), 0, 2],
        });
    }
    let rows = usize::try_from(rows).unwrap_or(usize::MAX);
    let values = read_f32_or_bf16(view.data, view.dtype)?;
    let plane = rows * dims.hidden;
    if values.len() != 2 * plane {
        return Err(VisionError::TensorShape {
            name: name.to_owned(),
            declared: declared.to_vec(),
            expected: vec![
                u64::try_from(dims.hidden).unwrap_or(u64::MAX),
                u64::try_from(rows).unwrap_or(u64::MAX),
                2,
            ],
        });
    }
    dims.pos_rows = rows;
    Ok((values[..plane].to_vec(), values[plane..].to_vec()))
}

/// Moves one checked payload to the device.
fn upload(
    backend: &Arc<dyn Backend>,
    name: &str,
    dtype: DType,
    bytes: &[u8],
) -> Result<DeviceBuffer, VisionError> {
    let fail = |source| VisionError::Upload {
        name: name.to_owned(),
        source,
    };
    let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let buffer = alloc(backend, len).map_err(fail)?;
    backend.copy_h2d(buffer.view(), bytes).map_err(fail)?;
    let _ = dtype;
    Ok(buffer)
}

/// Re-expresses a payload in another width: the two conversions this loader
/// performs, written once. BF16 to BF16 and F32 to F32 are copies.
fn widen_or_narrow(
    data: &[u8],
    from: DType,
    to: DType,
    name: &str,
) -> Result<Vec<u8>, VisionError> {
    let wrong = || VisionError::TensorDType {
        name: name.to_owned(),
        dtype: from,
    };
    match (from, to) {
        (DType::BF16, DType::BF16) | (DType::F32, DType::F32) => Ok(data.to_vec()),
        (DType::BF16, DType::F32) => {
            let mut out = Vec::with_capacity(data.len() * 2);
            for word in data.as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(*word);
                out.extend_from_slice(&f32::from_bits(u32::from(bits) << 16).to_le_bytes());
            }
            Ok(out)
        }
        (DType::F32, DType::BF16) => {
            let mut out = Vec::with_capacity(data.len() / 2);
            for word in data.as_chunks::<4>().0 {
                let value = f32::from_le_bytes(*word);
                out.extend_from_slice(&bf16_bits(value).to_le_bytes());
            }
            Ok(out)
        }
        _ => Err(wrong()),
    }
}

/// Reads a payload as f32 values, widening BF16 if that is how it is stored.
fn read_f32_or_bf16(data: &[u8], dtype: DType) -> Result<Vec<f32>, VisionError> {
    match dtype {
        DType::F32 => Ok(data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect()),
        DType::BF16 => Ok(data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| f32::from_bits(u32::from(u16::from_le_bytes(*w)) << 16))
            .collect()),
        _ => Err(VisionError::TensorDType {
            name: String::new(),
            dtype,
        }),
    }
}

/// The 16 bits an f32 keeps under round-to-nearest-even.
///
/// The GEMM widens by shifting, which is exact; this is the one place a value
/// is actually narrowed, and the encoder's host reference calls the same
/// function so the device and the reference see one number rather than two
/// roundings of one.
#[must_use]
pub(super) fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    // The rounding increment is the discarded half-way tie broken toward an
    // even kept bit: 0x7FFF plus the low kept bit.
    let increment = 0x7FFFu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(increment) >> 16) as u16
}

/// f32 values as little-endian bytes.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Name mapping for Gemma 4 vision tensors in MLX safetensors checkpoints.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Gemma4VisionNames;

impl NameMap for Gemma4VisionNames {
    fn map_role(&self, name: &str) -> Option<Role> {
        match name {
            "vision_tower.patch_embedder.input_proj"
            | "vision_tower.patch_embedder.input_proj.weight" => {
                return Some(Role::plain("v.patch_embd.weight"));
            }
            "vision_tower.patch_embedder.position_embedding_table" => {
                return Some(Role::plain("v.position_embd.weight"));
            }
            "embed_vision.embedding_projection" | "embed_vision.embedding_projection.weight" => {
                return Some(Role::plain("mm.input_projection.weight"));
            }
            _ => {}
        }

        let rest = name.strip_prefix("vision_tower.encoder.layers.")?;
        let (index, role) = rest.split_once('.')?;
        let layer: u32 = index.parse().ok()?;

        let suffix = match role {
            "input_layernorm.weight" => "ln1.weight",
            "post_attention_layernorm.weight" => "attn_post_norm.weight",
            "pre_feedforward_layernorm.weight" => "ln2.weight",
            "post_feedforward_layernorm.weight" => "ffn_post_norm.weight",

            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",

            "self_attn.q_proj.linear" | "self_attn.q_proj.linear.weight" => "attn_q.weight",
            "self_attn.k_proj.linear" | "self_attn.k_proj.linear.weight" => "attn_k.weight",
            "self_attn.v_proj.linear" | "self_attn.v_proj.linear.weight" => "attn_v.weight",
            "self_attn.o_proj.linear" | "self_attn.o_proj.linear.weight" => "attn_out.weight",

            "self_attn.q_proj.input_min" => "attn_q.input_min",
            "self_attn.q_proj.input_max" => "attn_q.input_max",
            "self_attn.q_proj.output_min" => "attn_q.output_min",
            "self_attn.q_proj.output_max" => "attn_q.output_max",

            "self_attn.k_proj.input_min" => "attn_k.input_min",
            "self_attn.k_proj.input_max" => "attn_k.input_max",
            "self_attn.k_proj.output_min" => "attn_k.output_min",
            "self_attn.k_proj.output_max" => "attn_k.output_max",

            "self_attn.v_proj.input_min" => "attn_v.input_min",
            "self_attn.v_proj.input_max" => "attn_v.input_max",
            "self_attn.v_proj.output_min" => "attn_v.output_min",
            "self_attn.v_proj.output_max" => "attn_v.output_max",

            "self_attn.o_proj.input_min" => "attn_out.input_min",
            "self_attn.o_proj.input_max" => "attn_out.input_max",
            "self_attn.o_proj.output_min" => "attn_out.output_min",
            "self_attn.o_proj.output_max" => "attn_out.output_max",

            "mlp.gate_proj.linear" | "mlp.gate_proj.linear.weight" => "ffn_gate.weight",
            "mlp.up_proj.linear" | "mlp.up_proj.linear.weight" => "ffn_up.weight",
            "mlp.down_proj.linear" | "mlp.down_proj.linear.weight" => "ffn_down.weight",

            "mlp.gate_proj.input_min" => "ffn_gate.input_min",
            "mlp.gate_proj.input_max" => "ffn_gate.input_max",
            "mlp.gate_proj.output_min" => "ffn_gate.output_min",
            "mlp.gate_proj.output_max" => "ffn_gate.output_max",

            "mlp.up_proj.input_min" => "ffn_up.input_min",
            "mlp.up_proj.input_max" => "ffn_up.input_max",
            "mlp.up_proj.output_min" => "ffn_up.output_min",
            "mlp.up_proj.output_max" => "ffn_up.output_max",

            "mlp.down_proj.input_min" => "ffn_down.input_min",
            "mlp.down_proj.input_max" => "ffn_down.input_max",
            "mlp.down_proj.output_min" => "ffn_down.output_min",
            "mlp.down_proj.output_max" => "ffn_down.output_max",

            _ => return None,
        };

        Some(Role::plain(format!("v.blk.{layer}.{suffix}")))
    }
}

fn load_gemma4_weights_mlx(
    backend: &Arc<dyn Backend>,
    dir: &Path,
) -> Result<Gemma4VisionWeights, VisionError> {
    let st = SafetensorsFile::open_with(dir, Arc::new(Gemma4VisionNames)).map_err(|_| {
        VisionError::Unsupported {
            what: "failed to open vision safetensors directory",
        }
    })?;
    let dims = read_dims_mlx(dir)?;

    let patch_embd = upload(
        backend,
        "v.patch_embd.weight",
        DType::BF16,
        &narrowed(&st, "v.patch_embd.weight", dims.patch_dim * dims.hidden)?,
    )?;

    // Two tables of `pos_rows` rows each, x then y, in one tensor.
    let pos = values(
        &st,
        "v.position_embd.weight",
        2 * dims.pos_rows * dims.hidden,
    )?;
    let (position_x, position_y) = pos.split_at(dims.pos_rows * dims.hidden);
    let (position_x, position_y) = (position_x.to_vec(), position_y.to_vec());

    let mut blocks = Vec::with_capacity(dims.depth);
    for i in 0..dims.depth {
        blocks.push(load_block_mlx(backend, &st, &dims, i)?);
    }

    let projection = Gemma4Linear {
        dtype: GemmType::Bf16,
        weight: upload(
            backend,
            "mm.input_projection.weight",
            DType::BF16,
            &narrowed(
                &st,
                "mm.input_projection.weight",
                dims.out_hidden * dims.hidden,
            )?,
        )?,
        // The export ships no clamp scalars beside the projector, so it is the
        // plain linear the reference falls back to when they are absent.
        clamps: NO_CLAMPS,
    };

    let ones = upload(
        backend,
        "pool norm weight (ones)",
        DType::F32,
        &f32_bytes(&vec![1.0f32; dims.hidden]),
    )?;

    Ok(Gemma4VisionWeights {
        dims,
        patch: patch_embd,
        position_x,
        position_y,
        blocks,
        projection,
        ones,
    })
}

fn load_block_mlx(
    backend: &Arc<dyn Backend>,
    st: &SafetensorsFile,
    dims: &Gemma4Dims,
    il: usize,
) -> Result<Gemma4Block, VisionError> {
    let p = format!("v.blk.{il}");
    Ok(Gemma4Block {
        ln1: load_norm_mlx(backend, st, &format!("{p}.ln1.weight"), dims.hidden)?,
        attn_q: load_linear_mlx(
            backend,
            st,
            &format!("{p}.attn_q"),
            dims.hidden,
            dims.hidden,
        )?,
        attn_k: load_linear_mlx(
            backend,
            st,
            &format!("{p}.attn_k"),
            dims.hidden,
            dims.hidden,
        )?,
        attn_v: load_linear_mlx(
            backend,
            st,
            &format!("{p}.attn_v"),
            dims.hidden,
            dims.hidden,
        )?,
        attn_out: load_linear_mlx(
            backend,
            st,
            &format!("{p}.attn_out"),
            dims.hidden,
            dims.hidden,
        )?,
        q_norm: load_norm_mlx(
            backend,
            st,
            &format!("{p}.attn_q_norm.weight"),
            dims.head_dim,
        )?,
        k_norm: load_norm_mlx(
            backend,
            st,
            &format!("{p}.attn_k_norm.weight"),
            dims.head_dim,
        )?,
        attn_post_norm: load_norm_mlx(
            backend,
            st,
            &format!("{p}.attn_post_norm.weight"),
            dims.hidden,
        )?,
        ln2: load_norm_mlx(backend, st, &format!("{p}.ln2.weight"), dims.hidden)?,
        ffn_gate: load_linear_mlx(
            backend,
            st,
            &format!("{p}.ffn_gate"),
            dims.intermediate,
            dims.hidden,
        )?,
        ffn_up: load_linear_mlx(
            backend,
            st,
            &format!("{p}.ffn_up"),
            dims.intermediate,
            dims.hidden,
        )?,
        ffn_down: load_linear_mlx(
            backend,
            st,
            &format!("{p}.ffn_down"),
            dims.hidden,
            dims.intermediate,
        )?,
        ffn_post_norm: load_norm_mlx(
            backend,
            st,
            &format!("{p}.ffn_post_norm.weight"),
            dims.hidden,
        )?,
    })
}

/// One tensor's values, decoded and counted.
///
/// The count is the check: the shapes here come from a config document, the
/// bytes from a shard, and a tensor of the wrong extent read against the wrong
/// geometry is a tower that loads and sees something else.
///
/// The loader decodes an unpacked tensor to `f32` and a packed one to `f16`,
/// and says which in the descriptor - so the element width comes from there
/// rather than from a guess at the byte count. The vision tower reads one
/// packed tensor, the projector, and everything else unpacked.
fn values(st: &SafetensorsFile, name: &str, want: usize) -> Result<Vec<f32>, VisionError> {
    let missing = || VisionError::MissingTensor { name: name.into() };
    let dtype = st.find(name).ok_or_else(missing)?.dtype;
    let bytes = st.tensor_bytes(name).map_err(|_| missing())?;
    let out = match dtype {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect(),
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| f32::from(half::f16::from_le_bytes(*word)))
            .collect(),
        other => {
            return Err(VisionError::Unsupported {
                what: unsupported_decode(other),
            });
        }
    };
    let out: Vec<f32> = out;
    if out.len() != want {
        return Err(VisionError::TowerTensorExtent {
            name: name.into(),
            want,
            got: out.len(),
        });
    }
    Ok(out)
}

/// Names a decoded element type this tower cannot read.
const fn unsupported_decode(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "the checkpoint loader handed the tower a bf16 tensor it does not decode",
        _ => "the checkpoint loader handed the tower an element type it does not decode",
    }
}

/// The same values, narrowed to the `bf16` the tower's kernels read.
fn narrowed(st: &SafetensorsFile, name: &str, want: usize) -> Result<Vec<u8>, VisionError> {
    let values = values(st, name, want)?;
    let mut packed = vec![0u8; values.len() * 2];
    for (slot, value) in packed.as_chunks_mut::<2>().0.iter_mut().zip(&values) {
        *slot = bf16_bits(*value).to_le_bytes();
    }
    Ok(packed)
}

fn load_norm_mlx(
    backend: &Arc<dyn Backend>,
    st: &SafetensorsFile,
    name: &str,
    width: usize,
) -> Result<DeviceBuffer, VisionError> {
    let values = values(st, name, width)?;
    upload(backend, name, DType::F32, &f32_bytes(&values))
}

fn load_linear_mlx(
    backend: &Arc<dyn Backend>,
    st: &SafetensorsFile,
    prefix: &str,
    out: usize,
    inp: usize,
) -> Result<Gemma4Linear, VisionError> {
    let name = format!("{prefix}.weight");
    let weight = upload(
        backend,
        &name,
        DType::BF16,
        &narrowed(st, &name, out * inp)?,
    )?;
    Ok(Gemma4Linear {
        dtype: GemmType::Bf16,
        weight,
        clamps: read_clamps_mlx(st, prefix)?,
    })
}

fn read_clamps_mlx(st: &SafetensorsFile, prefix: &str) -> Result<Clamps, VisionError> {
    let scalar = |suffix: &str| -> Result<f32, VisionError> {
        let full = format!("{prefix}.{suffix}");
        let Ok(bytes) = st.tensor_bytes(&full) else {
            return Ok(match suffix {
                "input_max" | "output_max" => f32::MAX,
                _ => f32::MIN,
            });
        };
        if bytes.len() < 4 {
            return Ok(match suffix {
                "input_max" | "output_max" => f32::MAX,
                _ => f32::MIN,
            });
        }
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    };
    Ok(Clamps {
        input_min: scalar("input_min")?,
        input_max: scalar("input_max")?,
        output_min: scalar("output_min")?,
        output_max: scalar("output_max")?,
    })
}

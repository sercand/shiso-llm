//! Reading an LFM2 vision tower out of its mmproj container.
//!
//! The file is a multimodal projector GGUF: the `SigLIP2` tower under `v.*` and
//! the family's two-layer projector under `mm.*`. Every width comes from the
//! file - the metadata for the tower's geometry, the tensors' own shapes for
//! the position grid and the projector - and the loader asks for each tensor
//! by name and refuses the file the moment one is missing or misshapen.
//!
//! # Every weight is uploaded as the file stores it
//!
//! The tower runs on the shared encoder GEMM, which reads single, half,
//! brain-float and `Q8_0` weights directly and widens on read. So nothing here
//! converts a weight: the shipped quantized projector stores most linears as
//! `Q8_0`, the feed-forward's down projection as IEEE half (its input width,
//! 4304, is not a whole number of 32-value blocks, so the converter could not
//! quantize it), and every bias, norm, the patch projection and the position
//! grid as f32 - and that is what reaches the device. A BF16 export loads
//! through the same path.
//!
//! # The attention's heads are padded, and that is exact
//!
//! The shipped head is 72 wide and the shared tiled attention takes heads in
//! whole 32-value slices, so the attention's four projections are re-laid at
//! load to give every head a 96-wide slot: the query, key and value weights
//! gain zero output rows (and zero bias) after each head's 72, and the output
//! projection gains zero input columns at the same places. A zero query or
//! key value adds nothing to a score, a zero value column produces a zero
//! output column, and a zero weight column reads it as nothing, so the tower
//! computes the same numbers - measured, the one-row attention it replaces
//! was about 70% of an encode. Padding a row is exact in every format,
//! `Q8_0` included: a zero row is zero blocks. Padding the output
//! projection's columns cuts across its `Q8_0` blocks (72 is not a multiple
//! of 32), so a `Q8_0` output projection is dequantized and stored as IEEE
//! half: the block values `scale * q` are rounded to half's 11 bits, a
//! relative error near 2^-12, against the 2^-8 the 8-bit quantization itself
//! already costs.
//!
//! # The one tensor that is rearranged, not converted
//!
//! `v.patch_embd.weight` is stored as a convolution kernel, `[out][c][y][x]`,
//! while the preprocessor lays a patch row out channel-fastest,
//! `y * 3 * patch + x * 3 + c` - which is also the order of the reference's
//! linear patch projection, of which the stored kernel is a reshaped copy.
//! The row is transposed back at load. The element count does not change,
//! which is exactly why a shape check alone could never catch a skipped
//! transpose: the tower would load cleanly and answer confidently about a
//! picture whose pixels were paired with the wrong weights.

use std::path::Path;
use std::sync::Arc;

use shiso_core::{Backend, DType, DeviceBuffer, alloc};
use shiso_modelfile::{GgufFile, TensorView};
use shiso_vision::{VisionError, open_gguf};

use super::{Lfm2Dims, PROJECTOR};
use crate::audio_ops::GemmType;

/// The architecture string a vision projector declares.
const ARCH: &str = "clip";

/// Formats a linear's weight may be stored in, all read by the GEMM as they
/// are.
const LINEAR_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16, DType::Q8_0];

/// Formats the vectors and the position grid may be stored in. They are read
/// as f32, so a narrower one is widened on the host.
const FLOAT_DTYPES: &[DType] = &[DType::F32, DType::BF16, DType::F16];

/// The slice the tiled attention takes a head in; every head slot is a whole
/// number of these.
const HEAD_SLICE: usize = 32;

/// Values one `Q8_0` block covers, and the bytes it takes.
const Q8_0_BLOCK: usize = 32;
const Q8_0_BYTES: usize = 34;

/// One biased linear projection, `[out][in]` row-major as the GEMM reads it.
#[derive(Debug)]
pub(super) struct Lfm2Linear {
    /// The weight, in [`Lfm2Linear::dtype`]'s storage.
    pub(super) weight: DeviceBuffer,
    /// How the weight is stored.
    pub(super) dtype: GemmType,
    /// The bias, f32 `[out]`.
    pub(super) bias: DeviceBuffer,
}

/// A layer norm's two vectors, f32 `[hidden]`.
#[derive(Debug)]
pub(super) struct Lfm2Norm {
    /// The scale.
    pub(super) weight: DeviceBuffer,
    /// The offset.
    pub(super) bias: DeviceBuffer,
}

/// One transformer block, in the order the encoder walks it.
#[derive(Debug)]
pub(super) struct Lfm2Block {
    /// Pre-attention layer norm.
    pub(super) ln1: Lfm2Norm,
    /// The query projection.
    pub(super) q: Lfm2Linear,
    /// The key projection.
    pub(super) k: Lfm2Linear,
    /// The value projection.
    pub(super) v: Lfm2Linear,
    /// The attention output projection.
    pub(super) out: Lfm2Linear,
    /// Pre-feed-forward layer norm.
    pub(super) ln2: Lfm2Norm,
    /// The feed-forward's first layer.
    pub(super) up: Lfm2Linear,
    /// The feed-forward's second layer.
    pub(super) down: Lfm2Linear,
}

/// An LFM2 vision tower and its projector, on the device.
#[derive(Debug)]
pub(super) struct Lfm2VisionWeights {
    /// The widths the encoder runs on.
    pub(super) dims: Lfm2Dims,
    /// The patch projection, transposed to the preprocessor's row order:
    /// `[hidden][patch_dim]`, f32.
    pub(super) patch: Lfm2Linear,
    /// The learned position grid, host-resident f32 `[side * side][hidden]`,
    /// row `y * side + x` for grid cell `(y, x)`. Resampled per picture on the
    /// host; see [`super::positions`].
    pub(super) positions: Vec<f32>,
    /// The transformer blocks.
    pub(super) blocks: Vec<Lfm2Block>,
    /// The final layer norm.
    pub(super) post_ln: Lfm2Norm,
    /// The projector's first layer, `merged -> proj_hidden`.
    pub(super) mm1: Lfm2Linear,
    /// The projector's second layer, `proj_hidden -> out_hidden`.
    pub(super) mm2: Lfm2Linear,
    /// Width of the projector's hidden layer.
    pub(super) proj_hidden: usize,
}

impl Lfm2VisionWeights {
    /// Device bytes the tower occupies.
    pub(super) fn device_bytes(&self) -> u64 {
        let linear = |l: &Lfm2Linear| l.weight.len() + l.bias.len();
        let norm = |n: &Lfm2Norm| n.weight.len() + n.bias.len();
        let block = |b: &Lfm2Block| {
            norm(&b.ln1)
                + norm(&b.ln2)
                + linear(&b.q)
                + linear(&b.k)
                + linear(&b.v)
                + linear(&b.out)
                + linear(&b.up)
                + linear(&b.down)
        };
        linear(&self.patch)
            + norm(&self.post_ln)
            + linear(&self.mm1)
            + linear(&self.mm2)
            + self.blocks.iter().map(block).sum::<u64>()
    }
}

/// Loads the tower in the mmproj at `path` onto `backend`.
pub(super) fn load_lfm2_weights(
    backend: &Arc<dyn Backend>,
    path: &Path,
) -> Result<Lfm2VisionWeights, VisionError> {
    let file = open_gguf(path)?;
    check_projector(&file)?;
    let mut dims = read_dims(&file)?;
    let positions = load_positions(&file, &mut dims)?;
    let loader = Loader {
        backend,
        file: &file,
    };
    let patch_embed = loader.patch(&dims)?;
    let mut blocks = Vec::with_capacity(dims.depth);
    for i in 0..dims.depth {
        blocks.push(loader.block(&dims, i)?);
    }
    let post_ln = loader.norm("v.post_ln", dims.hidden)?;
    // The projector's widths are its tensors' own: the first layer's input is
    // the unshuffled width the encoder produces, checked here, and its output
    // is whatever the export chose.
    let mm1_view = file
        .get("mm.1.weight")
        .ok_or_else(|| VisionError::MissingTensor {
            name: "mm.1.weight".to_owned(),
        })?;
    let proj_hidden = declared_axis(&mm1_view, 1, "mm.1.weight")?;
    let mm1 = loader.linear("mm.1", dims.merged(), proj_hidden)?;
    let mm2 = loader.linear("mm.2", proj_hidden, dims.out_hidden)?;
    Ok(Lfm2VisionWeights {
        dims,
        patch: patch_embed,
        positions,
        blocks,
        post_ln,
        mm1,
        mm2,
        proj_hidden,
    })
}

/// Refuses a file that is not this family's vision projector, by name.
fn check_projector(file: &GgufFile) -> Result<(), VisionError> {
    let meta = file.metadata();
    if meta.architecture() != Some(ARCH) {
        return Err(VisionError::Unsupported {
            what: "this mmproj is not a clip-architecture vision projector",
        });
    }
    // The key has been written both with and without the `vision.` infix by
    // the converters that produce these files; either spelling names the same
    // fact.
    let projector = meta
        .get_str(&format!("{ARCH}.projector_type"))
        .or_else(|| meta.get_str(&format!("{ARCH}.vision.projector_type")));
    if projector != Some(PROJECTOR) {
        return Err(VisionError::Unsupported {
            what: "this clip mmproj's vision tower is not an lfm2 projector",
        });
    }
    if meta.get_bool(&format!("{ARCH}.has_vision_encoder")) == Some(false) {
        return Err(VisionError::Unsupported {
            what: "this lfm2 mmproj carries no vision encoder",
        });
    }
    Ok(())
}

/// Derives the tower's widths from its own metadata.
fn read_dims(file: &GgufFile) -> Result<Lfm2Dims, VisionError> {
    let meta = file.metadata();
    let read = |suffix: &str| -> Result<usize, VisionError> {
        let key = format!("{ARCH}.vision.{suffix}");
        let value = meta
            .get_u32(&key)
            .ok_or_else(|| VisionError::TowerMetadata { key: key.clone() })?;
        let value = usize::try_from(value).map_err(|_| VisionError::TowerMetadata { key })?;
        if value == 0 {
            return Err(VisionError::TowerFieldZero {
                field: "a vision tower width",
            });
        }
        Ok(value)
    };
    let hidden = read("embedding_length")?;
    let heads = read("attention.head_count")?;
    if !hidden.is_multiple_of(heads) {
        return Err(VisionError::TowerNotDivisible {
            field: "embedding_length",
            value: hidden,
            divisor_field: "attention.head_count",
            divisor: heads,
        });
    }
    let patch = read("patch_size")?;
    let merge = read("projector.scale_factor")?;
    let eps_key = format!("{ARCH}.vision.attention.layer_norm_epsilon");
    let eps = meta
        .get_f32(&eps_key)
        .ok_or(VisionError::TowerMetadata { key: eps_key })?;
    Ok(Lfm2Dims {
        depth: read("block_count")?,
        hidden,
        heads,
        head_dim: hidden / heads,
        head_slot: (hidden / heads).next_multiple_of(HEAD_SLICE),
        intermediate: read("feed_forward_length")?,
        out_hidden: read("projection_dim")?,
        patch,
        patch_dim: 3 * patch * patch,
        merge,
        // A fact of the position tensor rather than of the metadata;
        // `load_positions` fills it in once it has checked the shape.
        pos_side: 0,
        eps,
    })
}

/// The learned position grid, kept on the host as f32 rows.
fn load_positions(file: &GgufFile, dims: &mut Lfm2Dims) -> Result<Vec<f32>, VisionError> {
    let name = "v.position_embd.weight";
    let view = file.get(name).ok_or_else(|| VisionError::MissingTensor {
        name: name.to_owned(),
    })?;
    let width = declared_axis(&view, 0, name)?;
    let rows = declared_axis(&view, 1, name)?;
    if width != dims.hidden || view.shape.dims().len() != 2 {
        return Err(VisionError::TensorShape {
            name: name.to_owned(),
            declared: view.shape.dims().to_vec(),
            expected: vec![wide(dims.hidden), wide(rows)],
        });
    }
    let side = rows.isqrt();
    if side * side != rows {
        return Err(VisionError::PositionGridNotSquare { count: rows });
    }
    dims.pos_side = side;
    floats(&view, name)
}

/// The checkpoint and the device, for the uploads below.
struct Loader<'a> {
    backend: &'a Arc<dyn Backend>,
    file: &'a GgufFile,
}

impl Loader<'_> {
    /// One block, by its index.
    fn block(&self, dims: &Lfm2Dims, i: usize) -> Result<Lfm2Block, VisionError> {
        let at = |suffix: &str| format!("v.blk.{i}.{suffix}");
        let h = dims.hidden;
        Ok(Lfm2Block {
            ln1: self.norm(&at("ln1"), h)?,
            q: self.head_rows(&at("attn_q"), dims)?,
            k: self.head_rows(&at("attn_k"), dims)?,
            v: self.head_rows(&at("attn_v"), dims)?,
            out: self.head_columns(&at("attn_out"), dims)?,
            ln2: self.norm(&at("ln2"), h)?,
            up: self.linear(&at("ffn_up"), h, dims.intermediate)?,
            down: self.linear(&at("ffn_down"), dims.intermediate, h)?,
        })
    }

    /// One biased linear, `inp -> out`, its weight uploaded as stored.
    fn linear(&self, prefix: &str, inp: usize, out: usize) -> Result<Lfm2Linear, VisionError> {
        let name = format!("{prefix}.weight");
        let view = self.checked(&name, &[inp, out], LINEAR_DTYPES)?;
        let dtype = gemm_type(&view, &name, inp, out)?;
        let weight = upload(self.backend, &name, view.data)?;
        let bias = self.vector(&format!("{prefix}.bias"), out)?;
        Ok(Lfm2Linear {
            weight,
            dtype,
            bias,
        })
    }

    /// A query, key or value projection, `hidden -> heads x head_slot`: each
    /// head's rows as stored, then zero rows up to its slot.
    fn head_rows(&self, prefix: &str, dims: &Lfm2Dims) -> Result<Lfm2Linear, VisionError> {
        let name = format!("{prefix}.weight");
        let (h, hd, slot) = (dims.hidden, dims.head_dim, dims.head_slot);
        let view = self.checked(&name, &[h, h], LINEAR_DTYPES)?;
        let dtype = gemm_type(&view, &name, h, h)?;
        let row = view.data.len() / h;
        let mut data = vec![0u8; dims.heads * slot * row];
        for head in 0..dims.heads {
            let src = &view.data[head * hd * row..][..hd * row];
            data[head * slot * row..][..hd * row].copy_from_slice(src);
        }
        let bias = self.floats_checked(&format!("{prefix}.bias"), h)?;
        let mut padded = vec![0.0f32; dims.heads * slot];
        for head in 0..dims.heads {
            padded[head * slot..][..hd].copy_from_slice(&bias[head * hd..][..hd]);
        }
        Ok(Lfm2Linear {
            weight: upload(self.backend, &name, &data)?,
            dtype,
            bias: upload(self.backend, &name, &f32_bytes(&padded))?,
        })
    }

    /// The attention output projection, `heads x head_slot -> hidden`: each
    /// row's columns per head as stored, then zero columns up to the slot.
    ///
    /// The columns of a `Q8_0` row straddle its blocks, so that format is
    /// dequantized and stored as half; the others keep their width and their
    /// values exactly.
    fn head_columns(&self, prefix: &str, dims: &Lfm2Dims) -> Result<Lfm2Linear, VisionError> {
        let name = format!("{prefix}.weight");
        let (h, hd, slot) = (dims.hidden, dims.head_dim, dims.head_slot);
        let view = self.checked(&name, &[h, h], LINEAR_DTYPES)?;
        let stored = gemm_type(&view, &name, h, h)?;
        let width = dims.heads * slot;
        let mut rows = vec![0.0f32; h * width];
        let row_bytes = view.data.len() / h;
        for o in 0..h {
            let values = row_floats(&view.data[o * row_bytes..][..row_bytes], stored, h);
            for head in 0..dims.heads {
                rows[o * width + head * slot..][..hd].copy_from_slice(&values[head * hd..][..hd]);
            }
        }
        let (dtype, bytes) = match stored {
            GemmType::F32 => (GemmType::F32, f32_bytes(&rows)),
            GemmType::Bf16 => (
                GemmType::Bf16,
                // Exact: every value here was a brain float widened by shifting.
                rows.iter()
                    .flat_map(|v| u16::try_from(v.to_bits() >> 16).unwrap_or(0).to_le_bytes())
                    .collect(),
            ),
            GemmType::F16 | GemmType::Q8_0 => (
                GemmType::F16,
                rows.iter()
                    .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                    .collect(),
            ),
        };
        Ok(Lfm2Linear {
            weight: upload(self.backend, &name, &bytes)?,
            dtype,
            bias: self.vector(&format!("{prefix}.bias"), h)?,
        })
    }

    /// A vector's values on the host, its shape and format checked.
    fn floats_checked(&self, name: &str, width: usize) -> Result<Vec<f32>, VisionError> {
        floats(&self.checked(name, &[width], FLOAT_DTYPES)?, name)
    }

    /// A layer norm's scale and offset.
    fn norm(&self, prefix: &str, width: usize) -> Result<Lfm2Norm, VisionError> {
        Ok(Lfm2Norm {
            weight: self.vector(&format!("{prefix}.weight"), width)?,
            bias: self.vector(&format!("{prefix}.bias"), width)?,
        })
    }

    /// A vector, as f32 on the device.
    fn vector(&self, name: &str, width: usize) -> Result<DeviceBuffer, VisionError> {
        let view = self.checked(name, &[width], FLOAT_DTYPES)?;
        upload(self.backend, name, &f32_bytes(&floats(&view, name)?))
    }

    /// The patch projection, transposed from the stored channel-slowest
    /// kernel to the preprocessor's channel-fastest row.
    ///
    /// The stored row of output channel `o` is indexed `c * p^2 + y * p + x`;
    /// the row the GEMM consumes is indexed `y * 3p + x * 3 + c`. Same
    /// elements, same count, different order - the transpose is the point.
    fn patch(&self, dims: &Lfm2Dims) -> Result<Lfm2Linear, VisionError> {
        let name = "v.patch_embd.weight";
        let p = dims.patch;
        let view = self.checked(name, &[p, p, 3, dims.hidden], FLOAT_DTYPES)?;
        let src = floats(&view, name)?;
        let mut packed = vec![0.0f32; dims.hidden * dims.patch_dim];
        for o in 0..dims.hidden {
            let row = o * dims.patch_dim;
            for c in 0..3usize {
                for y in 0..p {
                    for x in 0..p {
                        packed[row + y * p * 3 + x * 3 + c] = src[row + c * p * p + y * p + x];
                    }
                }
            }
        }
        Ok(Lfm2Linear {
            weight: upload(self.backend, name, &f32_bytes(&packed))?,
            dtype: GemmType::F32,
            bias: self.vector("v.patch_embd.bias", dims.hidden)?,
        })
    }

    /// One tensor, with its format and its declared shape checked.
    ///
    /// The container writes the fastest-varying axis first, so `dims` is in
    /// that order: a linear `[out][in]` is declared `[in, out]`.
    fn checked(
        &self,
        name: &str,
        dims: &[usize],
        allowed: &[DType],
    ) -> Result<TensorView<'_>, VisionError> {
        let view = self
            .file
            .get(name)
            .ok_or_else(|| VisionError::MissingTensor {
                name: name.to_owned(),
            })?;
        if !allowed.contains(&view.dtype) {
            return Err(VisionError::TensorDType {
                name: name.to_owned(),
                dtype: view.dtype,
            });
        }
        let expected: Vec<u64> = dims.iter().map(|d| wide(*d)).collect();
        if view.shape.dims() != expected.as_slice() {
            return Err(VisionError::TensorShape {
                name: name.to_owned(),
                declared: view.shape.dims().to_vec(),
                expected,
            });
        }
        Ok(view)
    }
}

/// How the GEMM reads a checked linear weight `inp -> out`.
fn gemm_type(
    view: &TensorView<'_>,
    name: &str,
    inp: usize,
    out: usize,
) -> Result<GemmType, VisionError> {
    Ok(match view.dtype {
        DType::F32 => GemmType::F32,
        DType::F16 => GemmType::F16,
        DType::BF16 => GemmType::Bf16,
        _ => {
            // A row is a whole number of blocks or it is not a row the GEMM
            // can address; the shape check has the element count.
            if !inp.is_multiple_of(Q8_0_BLOCK)
                || view.data.len() != out * inp / Q8_0_BLOCK * Q8_0_BYTES
            {
                return Err(VisionError::TensorDType {
                    name: name.to_owned(),
                    dtype: view.dtype,
                });
            }
            GemmType::Q8_0
        }
    })
}

/// One stored weight row of `width` values, as f32.
fn row_floats(row: &[u8], dtype: GemmType, width: usize) -> Vec<f32> {
    match dtype {
        GemmType::F32 => row
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect(),
        GemmType::F16 => row
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| half::f16::from_bits(u16::from_le_bytes(*w)).to_f32())
            .collect(),
        GemmType::Bf16 => row
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| f32::from_bits(u32::from(u16::from_le_bytes(*w)) << 16))
            .collect(),
        GemmType::Q8_0 => {
            let mut out = Vec::with_capacity(width);
            for block in row.chunks_exact(Q8_0_BYTES) {
                let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                out.extend(
                    block[2..]
                        .iter()
                        .map(|&q| scale * f32::from(i8::from_le_bytes([q]))),
                );
            }
            out
        }
    }
}

/// One declared axis of a tensor, as a count.
fn declared_axis(view: &TensorView<'_>, axis: usize, name: &str) -> Result<usize, VisionError> {
    view.shape
        .dims()
        .get(axis)
        .and_then(|&d| usize::try_from(d).ok())
        .filter(|&d| d > 0)
        .ok_or_else(|| VisionError::TensorShape {
            name: name.to_owned(),
            declared: view.shape.dims().to_vec(),
            expected: Vec::new(),
        })
}

/// A float payload as f32 values, widening the narrower widths.
fn floats(view: &TensorView<'_>, name: &str) -> Result<Vec<f32>, VisionError> {
    let data = view.data;
    Ok(match view.dtype {
        DType::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect(),
        DType::BF16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| f32::from_bits(u32::from(u16::from_le_bytes(*w)) << 16))
            .collect(),
        DType::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| half::f16::from_bits(u16::from_le_bytes(*w)).to_f32())
            .collect(),
        other => {
            return Err(VisionError::TensorDType {
                name: name.to_owned(),
                dtype: other,
            });
        }
    })
}

/// Moves one payload to the device.
fn upload(
    backend: &Arc<dyn Backend>,
    name: &str,
    bytes: &[u8],
) -> Result<DeviceBuffer, VisionError> {
    let fail = |source| VisionError::Upload {
        name: name.to_owned(),
        source,
    };
    let buffer = alloc(backend, wide(bytes.len())).map_err(fail)?;
    backend.copy_h2d(buffer.view(), bytes).map_err(fail)?;
    Ok(buffer)
}

/// f32 values as little-endian bytes.
pub(super) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A count as the container's width.
fn wide(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

//! Device weights from a [`TensorStore`], in whatever encoding the file chose.
//!
//! A family names a tensor and the shape it expects; these helpers pick the
//! multiply that suits the storage and refuse a tensor whose shape or encoding
//! is not what was asked for, naming it. Large matrices are imported from the
//! mapping as they are; only small parameters (norms, conv kernels, biases)
//! are dequantised on the host.

use shiso_audio::AudioError;
use shiso_core::{DType, DeviceBuffer};
use shiso_engine::projection::{AffineRanges, Weight};

use super::store::{Stored, StoredType, TensorStore, dequant};
use super::{GemmType, Linear, LinearWeight, Ops};

/// Where one linear's tensors are.
#[derive(Debug, Clone, Copy)]
pub struct LinearSource<'a> {
    /// The weight tensor. For an affine-quantised weight, its `.scales` and
    /// `.biases` siblings share its stem.
    pub weight: &'a str,
    /// The layer's own bias, when it has one.
    pub bias: Option<&'a str>,
    /// Elements per affine group, for a weight stored as packed words.
    pub affine_group: Option<usize>,
}

fn checkpoint(what: String) -> AudioError {
    AudioError::Checkpoint { what }
}

/// The shape with unit axes dropped, so a pointwise conv's `[out, in, 1]` and
/// `[out, 1, in]` both read as the matrix they are.
fn squeezed(t: &Stored<'_>) -> Vec<u64> {
    t.shape.iter().copied().filter(|d| *d != 1).collect()
}

fn as_u32(value: usize, what: &str) -> Result<u32, AudioError> {
    u32::try_from(value).map_err(|_| checkpoint(format!("{what} = {value} does not fit a u32")))
}

/// A weight stored as packed affine words, on the engine's affine projection.
fn affine_weight(
    ops: &Ops,
    store: &TensorStore,
    src: &LinearSource<'_>,
    t: &Stored<'_>,
    (out, inp): (usize, usize),
) -> Result<LinearWeight, AudioError> {
    let backend = ops.backend();
    let group = src.affine_group.ok_or_else(|| {
        checkpoint(format!(
            "{} is packed words but no affine group was given",
            t.name
        ))
    })?;
    let (rows, words) = match t.shape.as_slice() {
        [r, w] => (*r, *w),
        other => {
            return Err(checkpoint(format!(
                "{} has shape {other:?}, not [out][words]",
                t.name
            )));
        }
    };
    let bits = usize::try_from(words * 32).unwrap_or(usize::MAX) / inp.max(1);
    let dtype = match (bits, group) {
        (4, 64) => DType::MlxAffine4Gs64,
        (8, 64) => DType::MlxAffine8Gs64,
        (4, 32) => DType::MlxAffine4Gs32,
        _ => {
            return Err(checkpoint(format!(
                "{} packs {bits}-bit weights in groups of {group}, which no projection reads",
                t.name
            )));
        }
    };
    if usize::try_from(rows).ok() != Some(out)
        || words * 32 != u64::try_from(inp * bits).unwrap_or(0)
    {
        return Err(checkpoint(format!(
            "{} has shape {:?}, not [{out}][{inp}] packed",
            t.name, t.shape
        )));
    }
    let stem = src.weight.strip_suffix(".weight").unwrap_or(src.weight);
    let scales = store.require(&format!("{stem}.scales"))?;
    let biases = store.require(&format!("{stem}.biases"))?;
    // The projection indexes scales and biases from the geometry alone, so a
    // pair that does not match it would be read past its end or misread.
    let want = vec![
        u64::try_from(out).unwrap_or(0),
        u64::try_from(inp / group).unwrap_or(0),
    ];
    for side in [&scales, &biases] {
        if side.ty != StoredType::Bf16 || side.shape != want || !inp.is_multiple_of(group) {
            return Err(checkpoint(format!(
                "{} is {:?} {:?}; a group of {group} over [{out}][{inp}] needs bf16 {want:?}",
                side.name, side.ty, side.shape
            )));
        }
    }
    if scales.shard != t.shard || biases.shard != t.shard {
        return Err(checkpoint(format!(
            "{}'s scales and biases are in another file",
            t.name
        )));
    }
    let ranges = AffineRanges {
        words: t.range.clone(),
        scales: scales.range.clone(),
        biases: biases.range.clone(),
    };
    let w = Weight::import_affine(
        backend,
        dtype,
        as_u32(inp, "in")?,
        as_u32(out, "out")?,
        t.host.clone(),
        ranges,
    )?;
    Ok(LinearWeight::Projected(w))
}

/// A weight stored as whole elements or GGUF blocks, on the GEMM that suits it.
fn dense_weight(
    ops: &Ops,
    t: &Stored<'_>,
    (out, inp): (usize, usize),
) -> Result<LinearWeight, AudioError> {
    let backend = ops.backend();
    let want = [
        u64::try_from(out).unwrap_or(0),
        u64::try_from(inp).unwrap_or(0),
    ];
    if squeezed(t) != want {
        return Err(checkpoint(format!(
            "{} has shape {:?}, not [{out}][{inp}]",
            t.name, t.shape
        )));
    }
    let gemm = |dtype| -> Result<LinearWeight, AudioError> {
        let buffer = shiso_core::import_host(backend, t.host.clone(), t.range.clone())?;
        Ok(LinearWeight::Gemm {
            dtype,
            buffer,
            in_dim: inp,
            out_dim: out,
        })
    };
    Ok(match t.ty {
        StoredType::F32 => gemm(GemmType::F32)?,
        StoredType::Bf16 => gemm(GemmType::Bf16)?,
        StoredType::F16 => gemm(GemmType::F16)?,
        StoredType::Q8_0 => gemm(GemmType::Q8_0)?,
        StoredType::Q4K => {
            let w = Weight::import(
                backend,
                DType::Q4_K,
                as_u32(inp, "in")?,
                as_u32(out, "out")?,
                t.host.clone(),
                t.range.clone(),
            )?;
            LinearWeight::Projected(w)
        }
        ref other => {
            return Err(checkpoint(format!(
                "{} is stored as {other:?}, which no multiply here reads",
                t.name
            )));
        }
    })
}

/// Builds `y = x * W^T (+ b)` for a weight stored `[out][inp]`.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming the tensor for a missing tensor, a shape
/// other than `[out][inp]`, an affine layout that is not 4 or 8 bits, or an
/// encoding no multiply here reads; a device error while importing.
pub fn linear(
    ops: &Ops,
    store: &TensorStore,
    src: &LinearSource<'_>,
    out: usize,
    inp: usize,
) -> Result<Linear, AudioError> {
    let t = store.require(src.weight)?;
    let weight = if t.ty == StoredType::U32 {
        affine_weight(ops, store, src, &t, (out, inp))?
    } else {
        dense_weight(ops, &t, (out, inp))?
    };
    let bias = match src.bias {
        Some(name) => Some(f32_param(ops, store, name, out)?.0),
        None => None,
    };
    Ok(Linear { weight, bias })
}

/// Bytes one row of `inp` elements occupies in a dense encoding, or `None`
/// for an encoding a row cannot be cut from.
fn row_bytes(ty: &StoredType, inp: usize) -> Option<usize> {
    match ty {
        StoredType::F32 => Some(inp * 4),
        StoredType::Bf16 | StoredType::F16 => Some(inp * 2),
        StoredType::Q8_0 if inp.is_multiple_of(32) => Some(inp / 32 * 34),
        _ => None,
    }
}

/// Builds a linear over rows `rows` of a dense `[total_out][inp]` weight, for
/// one projection of a fused matrix.
///
/// The rows are imported in place from the mapping: a row of a dense encoding
/// is contiguous whatever the container, so a slice of rows is one byte range.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming the tensor for a missing tensor, a shape
/// other than `[total_out][inp]`, a row range past its end, or an encoding
/// whose rows cannot be cut (packed words, super-blocks); a device error while
/// importing.
pub fn linear_rows(
    ops: &Ops,
    store: &TensorStore,
    weight: &str,
    rows: std::ops::Range<usize>,
    total_out: usize,
    inp: usize,
) -> Result<Linear, AudioError> {
    let t = store.require(weight)?;
    let want = [
        u64::try_from(total_out).unwrap_or(0),
        u64::try_from(inp).unwrap_or(0),
    ];
    if squeezed(&t) != want {
        return Err(checkpoint(format!(
            "{weight} has shape {:?}, not [{total_out}][{inp}]",
            t.shape
        )));
    }
    if rows.start >= rows.end || rows.end > total_out {
        return Err(checkpoint(format!(
            "rows {rows:?} of {weight} are not inside its {total_out}"
        )));
    }
    let row = row_bytes(&t.ty, inp).ok_or_else(|| {
        checkpoint(format!(
            "{weight} is stored as {:?}, which rows cannot be cut from",
            t.ty
        ))
    })?;
    let dtype = match t.ty {
        StoredType::F32 => GemmType::F32,
        StoredType::Bf16 => GemmType::Bf16,
        StoredType::F16 => GemmType::F16,
        _ => GemmType::Q8_0,
    };
    let range = t.range.start + rows.start * row..t.range.start + rows.end * row;
    let buffer = shiso_core::import_host(ops.backend(), t.host.clone(), range)?;
    Ok(Linear {
        weight: LinearWeight::Gemm {
            dtype,
            buffer,
            in_dim: inp,
            out_dim: rows.end - rows.start,
        },
        bias: None,
    })
}

/// A small parameter as f32 on the device, with its host values.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming the tensor when it is missing, holds a
/// different element count, or is stored in an encoding the host cannot
/// dequantise; a device error while uploading.
pub fn f32_param(
    ops: &Ops,
    store: &TensorStore,
    name: &str,
    elements: usize,
) -> Result<(DeviceBuffer, Vec<f32>), AudioError> {
    let values = dequant(&store.require(name)?)?;
    if values.len() != elements {
        return Err(checkpoint(format!(
            "{name} holds {} values, not {elements}",
            values.len()
        )));
    }
    Ok((upload_f32(ops, &values)?, values))
}

/// Uploads `values` as a new f32 device buffer.
///
/// # Errors
///
/// A device error while allocating or copying.
pub fn upload_f32(ops: &Ops, values: &[f32]) -> Result<DeviceBuffer, AudioError> {
    let buffer = ops.scratch(values.len())?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    ops.backend().copy_h2d(
        buffer.slice(0, u64::try_from(bytes.len()).unwrap_or(u64::MAX))?,
        &bytes,
    )?;
    Ok(buffer)
}

/// Reads the first `floats` f32 values of `view` back to the host.
///
/// # Errors
///
/// A device error, or a view shorter than `floats` values.
pub fn download_f32(
    ops: &Ops,
    view: shiso_core::DeviceView<'_>,
    floats: usize,
) -> Result<Vec<f32>, AudioError> {
    let mut bytes = vec![0u8; floats * 4];
    ops.backend().copy_d2h(
        view.slice(0, u64::try_from(bytes.len()).unwrap_or(u64::MAX))?,
        &mut bytes,
    )?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// A conv weight as f32 in the reference layout `[out][in][k...]`.
///
/// `channels_last` is for exports that store `[out][k...][in]` (the MLX
/// convention), which are permuted here; every other export already matches.
/// A tensor whose element count differs from `shape`'s is refused by name.
///
/// # Errors
///
/// [`AudioError::Checkpoint`] naming the tensor for a missing tensor, a
/// different element count, or an encoding the host cannot dequantise.
pub fn conv_weight(
    store: &TensorStore,
    name: &str,
    shape: &[usize],
    channels_last: bool,
) -> Result<Vec<f32>, AudioError> {
    let values = dequant(&store.require(name)?)?;
    let want: usize = shape.iter().product();
    if values.len() != want {
        return Err(checkpoint(format!(
            "{name} holds {} values, not {shape:?}",
            values.len()
        )));
    }
    let [out, inp, kernel @ ..] = shape else {
        return Err(checkpoint(format!(
            "{name}: a conv weight needs [out][in][k...]"
        )));
    };
    if !channels_last || *inp == 1 {
        // With one input channel the two layouts are the same bytes.
        return Ok(values);
    }
    let taps: usize = kernel.iter().product();
    let mut permuted = vec![0.0f32; want];
    for o in 0..*out {
        for t in 0..taps {
            for i in 0..*inp {
                permuted[(o * inp + i) * taps + t] = values[(o * taps + t) * inp + i];
            }
        }
    }
    Ok(permuted)
}

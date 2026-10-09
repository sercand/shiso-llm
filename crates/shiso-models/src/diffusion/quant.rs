//! The per-row INT8 weight format and its `ConvRot` activation rotation.
//!
//! An image export in this format stores each quantized linear `<name>` as
//! three tensors: `<name>.weight`, signed byte codes `q [out, in]`;
//! `<name>.weight_scale`, one `f32` per weight row (`[out, 1]`, or one scalar
//! for the whole tensor); and `<name>.comfy_quant`, a `U8` JSON document naming
//! the format. The dequantized weight is `q[o, i] * s[o]`, and nothing here ever
//! computes it into memory: the codes reach the device as the bytes the file
//! wrote and the scale rides the GEMM epilogue.
//!
//! With `convrot` on, the export also rotated the weight's input axis: for
//! each 256-wide column group `g`, `W_rot[:, g] = W[:, g] . H` with
//! `H = kron(H4, H4, H4, H4) / 16`, `H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],
//! [-1,1,1,1]]`. `H` is symmetric and orthogonal (`H . H = I`), so rotating each
//! activation row the same way, `x_rot[g] = x[g] . H`, gives
//! `x_rot . W_rot^T = x . W^T` exactly in exact arithmetic. The rotation spreads
//! an outlier channel across its group before the weight was quantized, which
//! is why the rounding it saved is not undone by running unrotated.
//!
//! The layout is read, never guessed: a byte-code weight without its format
//! document or its scale is refused, as is any format, group size or shape
//! this module does not implement.

use serde::Deserialize;
use shiso_modelfile::st_header::StDType;
use shiso_vision::generator::ImageGenError;

use super::weights::StStore;

/// The only rotation group this build implements: the width `H` acts on.
pub(crate) const ROT_GROUP: usize = 256;

/// The rotation a linear's input takes before its GEMM.
///
/// One variant because one group size exists; the type makes "rotated, and by
/// what" a fact a [`super::lin::Lin`] carries rather than a loose integer a
/// call site could pass as zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RotGroup {
    /// `H = kron^4(H4) / 16` over each contiguous 256-wide group.
    Hadamard256,
}

impl RotGroup {
    /// Elements per group.
    pub(crate) const fn width(self) -> usize {
        match self {
            Self::Hadamard256 => ROT_GROUP,
        }
    }
}

/// The `<name>.comfy_quant` document, as the export writes it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantDoc {
    format: String,
    #[serde(default)]
    convrot: bool,
    #[serde(default)]
    convrot_groupsize: Option<u64>,
}

/// The per-row INT8 layout of one linear, read and checked from its tensors.
#[derive(Debug)]
pub(crate) struct Int8Layout {
    /// The input rotation, when the export applied one.
    pub rotate: Option<RotGroup>,
    /// One scale per weight row, `out` long (a scalar scale broadcast).
    pub row_scale: Vec<f32>,
}

fn refuse(message: String) -> ImageGenError {
    ImageGenError::Checkpoint(message)
}

/// Parses a `comfy_quant` document for the tensor `weight_name`, answering the
/// rotation it asks for. `None` is an answer, not a refusal: a plain per-row
/// INT8 weight (`convrot` absent or false) runs the same GEMM unrotated.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] naming the tensor when the bytes are not the
/// document, the format is not `int8_tensorwise`, or `convrot` is on with a
/// group size other than 256 (or none).
pub(crate) fn parse_quant_spec(
    weight_name: &str,
    bytes: &[u8],
) -> Result<Option<RotGroup>, ImageGenError> {
    let doc: QuantDoc = serde_json::from_slice(bytes).map_err(|e| {
        refuse(format!(
            "'{weight_name}': its comfy_quant document does not parse: {e}"
        ))
    })?;
    if doc.format != "int8_tensorwise" {
        return Err(refuse(format!(
            "'{weight_name}': quantization format '{}' is not implemented (only int8_tensorwise is)",
            doc.format
        )));
    }
    if !doc.convrot {
        return Ok(None);
    }
    match doc.convrot_groupsize {
        Some(256) => Ok(Some(RotGroup::Hadamard256)),
        Some(other) => Err(refuse(format!(
            "'{weight_name}': convrot group size {other} is not implemented (only 256 is)"
        ))),
        None => Err(refuse(format!(
            "'{weight_name}': convrot is on but the document names no convrot_groupsize"
        ))),
    }
}

fn dim(weight_name: &str, shape: &[u64], axis: usize) -> Result<usize, ImageGenError> {
    let value = shape
        .get(axis)
        .ok_or_else(|| refuse(format!("'{weight_name}' is not a matrix (shape {shape:?})")))?;
    usize::try_from(*value).map_err(|_| {
        refuse(format!(
            "'{weight_name}' dimension {value} does not fit usize"
        ))
    })
}

/// Reads and checks the per-row INT8 layout of the linear `name` (whose
/// `<name>.weight` the caller found stored as `I8`).
///
/// # Errors
/// [`ImageGenError::Checkpoint`] naming the tensor when the weight is not a
/// matrix, its `comfy_quant` or `weight_scale` companion is missing, the
/// document is refused by [`parse_quant_spec`], the scale is not `F32` shaped
/// `[out, 1]`, `[out]`, `[1]` or `[]`, or a rotated weight's input width is not
/// a multiple of the group.
pub(crate) fn read_int8_layout(store: &StStore, name: &str) -> Result<Int8Layout, ImageGenError> {
    let weight_name = format!("{name}.weight");
    let entry = store.require(&weight_name)?;
    if entry.dtype != StDType::I8 {
        return Err(refuse(format!(
            "'{weight_name}' is {:?}, not the I8 codes of a per-row INT8 linear",
            entry.dtype
        )));
    }
    if entry.shape.len() != 2 {
        return Err(refuse(format!(
            "'{weight_name}' is not a matrix (shape {:?})",
            entry.shape
        )));
    }
    let out_dim = dim(&weight_name, &entry.shape, 0)?;
    let in_dim = dim(&weight_name, &entry.shape, 1)?;

    let quant_name = format!("{name}.comfy_quant");
    if !store.contains(&quant_name) {
        return Err(refuse(format!(
            "'{weight_name}' stores I8 codes but '{quant_name}' is missing; the layout is \
             not guessed"
        )));
    }
    let rotate = parse_quant_spec(&weight_name, store.bytes(&quant_name)?)?;
    if let Some(group) = rotate
        && !in_dim.is_multiple_of(group.width())
    {
        return Err(refuse(format!(
            "'{weight_name}': input width {in_dim} is not a multiple of the convrot group {}",
            group.width()
        )));
    }

    let scale_name = format!("{name}.weight_scale");
    let scale_entry = store.get(&scale_name).ok_or_else(|| {
        refuse(format!(
            "'{weight_name}' stores I8 codes but its per-row scale '{scale_name}' is missing"
        ))
    })?;
    if scale_entry.dtype != StDType::F32 {
        return Err(refuse(format!(
            "'{scale_name}' is {:?}; a per-row INT8 scale is F32",
            scale_entry.dtype
        )));
    }
    let out_u64 = entry.shape[0];
    let row_scale = match scale_entry.shape.as_slice() {
        [rows, 1] | [rows] if *rows == out_u64 => store.f32_vec(&scale_name)?,
        [] | [1] => vec![store.f32_scalar(&scale_name)?; out_dim],
        other => {
            return Err(refuse(format!(
                "'{scale_name}' has shape {other:?}; a per-row INT8 scale is [{out_dim}, 1] \
                 or a scalar"
            )));
        }
    };
    Ok(Int8Layout { rotate, row_scale })
}

/// `v . H` for one 256-wide group, in place: four radix-4 butterflies, one
/// per base-4 digit of the index, then the exact `1/16`. The same association
/// the device kernel makes, so the two agree to the last bit wherever the
/// device does not contract a multiply-add.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "reached through `unrotate_row_f32`, whose first production caller is the INT8 text encoder"
    )
)]
fn rotate_group(v: &mut [f32]) {
    for stride in [1usize, 4, 16, 64] {
        for hi in 0..ROT_GROUP / (4 * stride) {
            for lo in 0..stride {
                let b = hi * 4 * stride + lo;
                let x = [v[b], v[b + stride], v[b + 2 * stride], v[b + 3 * stride]];
                let s = (x[0] + x[1]) + (x[2] + x[3]);
                v[b] = s - 2.0 * x[3];
                v[b + stride] = s - 2.0 * x[2];
                v[b + 2 * stride] = s - 2.0 * x[1];
                v[b + 3 * stride] = s - 2.0 * x[0];
            }
        }
    }
    for x in v.iter_mut() {
        *x *= 0.0625;
    }
}

/// Rotates every `group`-wide run of `row` by `H`, in place.
///
/// Because `H` is its own inverse this is both directions: it rotates an
/// activation row into the weight's frame, and it recovers an embedding row
/// stored in that frame, `row = (q_row * s) . H` - which is what a text
/// encoder's `ConvRot` `embed_tokens` needs before the row enters the stack.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] when `group` is not the implemented 256 or
/// the row is not a whole number of groups.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the INT8 text encoder's embedding gather is its first production caller"
    )
)]
pub(crate) fn unrotate_row_f32(row: &mut [f32], group: usize) -> Result<(), ImageGenError> {
    if group != ROT_GROUP {
        return Err(refuse(format!(
            "convrot group size {group} is not implemented (only {ROT_GROUP} is)"
        )));
    }
    if !row.len().is_multiple_of(group) {
        return Err(refuse(format!(
            "a {}-wide row is not a whole number of {group}-wide convrot groups",
            row.len()
        )));
    }
    for chunk in row.chunks_exact_mut(group) {
        rotate_group(chunk);
    }
    Ok(())
}

//! A linear layer in its stored native format.
//!
//! [`Lin`] pairs a weight buffer with the [`WeightKind`] that selects its GEMM
//! entry point. A new weight dtype is a new [`WeightKind`] variant plus one arm
//! in [`Lin::load`]; callers that only forward `kind` to the GEMM calls stay
//! unchanged.

use shiso_core::DeviceBuffer;
use shiso_modelfile::st_header::StDType;
use shiso_vision::generator::ImageGenError;

use super::ops::{LinWeight, Ops, WeightKind};
use super::quant::{RotGroup, read_int8_layout};
use super::weights::{BlockFormat, StStore};

/// A linear layer's weight in its stored native format, with its bias widened
/// to f32 and its FP8 scale (or per-row INT8 scales) read at load.
///
/// The weight buffer is an import of the shard mapping's byte range: BF16,
/// `F8_E4M3`, F32 and I8 weights all reach the device as the bytes the
/// checkpoint wrote, and the GEMM entry point is chosen from the header dtype. Only the
/// bias - a vector of at most a few tens of thousands of elements - is
/// widened, so the norm/bias rule and the zero-dequant rule for matrices
/// stay separate.
#[derive(Debug)]
pub(crate) struct Lin {
    /// Weight rows (`out_dim`, `in_dim`) in their stored format.
    pub w: DeviceBuffer,
    /// Stored format of `w`.
    pub kind: WeightKind,
    /// Epilogue multiplier (the FP8 tensor's `weight_scale`, else 1.0).
    pub scale: f32,
    /// One f32 scale per weight row, for [`WeightKind::Int8Row`] only.
    pub row_scale: Option<DeviceBuffer>,
    /// The rotation the export stored the weight's input axis under; the
    /// GEMM entry rotates the input to match (see [`super::quant`]).
    pub rotate: Option<RotGroup>,
    /// Bias as f32, if the checkpoint carries one.
    pub bias: Option<DeviceBuffer>,
    /// Rows of the weight matrix.
    pub out_dim: usize,
    /// Columns of the weight matrix.
    pub in_dim: usize,
}

impl Lin {
    /// Loads `{name}.weight` (and `{name}.bias` when present) onto the ops'
    /// backend, refusing a dtype no GEMM entry point reads.
    pub(crate) fn load(store: &StStore, ops: &Ops, name: &str) -> Result<Self, ImageGenError> {
        let weight_name = format!("{name}.weight");
        let entry = store.require(&weight_name)?;
        let kind = match (entry.block, entry.dtype) {
            (Some(BlockFormat::Q2K), _) => WeightKind::Q2K,
            (Some(BlockFormat::Q3K), _) => WeightKind::Q3K,
            (Some(BlockFormat::Q4K), _) => WeightKind::Q4K,
            (Some(BlockFormat::Q5K), _) => WeightKind::Q5K,
            (Some(BlockFormat::Q6K), _) => WeightKind::Q6K,
            (Some(BlockFormat::Q8_0), _) => WeightKind::Q8_0,
            (None, dtype) => Self::plain_kind(&weight_name, dtype)?,
        };
        let scale = if kind == WeightKind::Fp8 {
            store.weight_scale(&weight_name)?
        } else {
            1.0
        };
        // The codes' layout comes from the export's own document and scale
        // tensor, checked before a byte reaches the device.
        let (row_scale, rotate) = if kind == WeightKind::Int8Row {
            let layout = read_int8_layout(store, name)?;
            (
                Some(ops.upload_f32("load row scale", &layout.row_scale)?),
                layout.rotate,
            )
        } else {
            (None, None)
        };
        let w = store.import(ops.backend(), &weight_name)?;
        let shape = &entry.shape;
        let (out_dim, in_dim) = (
            usize::try_from(*shape.first().ok_or_else(|| {
                ImageGenError::Checkpoint(format!("'{weight_name}' has empty shape"))
            })?)
            .unwrap_or(usize::MAX),
            usize::try_from(*shape.get(1).ok_or_else(|| {
                ImageGenError::Checkpoint(format!("'{weight_name}' is not a matrix"))
            })?)
            .unwrap_or(usize::MAX),
        );
        if let Some(format) = entry.block {
            // The GGUF reader sized the span from the declared shape; this
            // pins the per-row geometry the kernels walk, so a row that is
            // not whole blocks is refused naming the tensor, not read into
            // its neighbour.
            let row_bytes = in_dim
                .is_multiple_of(format.block_elems())
                .then(|| format.row_bytes(in_dim));
            let stored = store.bytes(&weight_name)?.len();
            if row_bytes.and_then(|row| row.checked_mul(out_dim)) != Some(stored) {
                return Err(ImageGenError::Checkpoint(format!(
                    "'{weight_name}' is {format:?} with {in_dim} weights per row; a row must \
                     be whole {}-weight blocks of {} bytes and the tensor {stored} bytes \
                     long",
                    format.block_elems(),
                    format.block_bytes()
                )));
            }
        }
        let bias_name = format!("{name}.bias");
        let bias = if store.contains(&bias_name) {
            let values = store.f32_vec(&bias_name)?;
            Some(ops.upload_f32("load bias", &values)?)
        } else {
            None
        };
        Ok(Self {
            w,
            kind,
            scale,
            row_scale,
            rotate,
            bias,
            out_dim,
            in_dim,
        })
    }

    /// The GEMM kind of a weight stored as a plain safetensors dtype.
    fn plain_kind(weight_name: &str, dtype: StDType) -> Result<WeightKind, ImageGenError> {
        match dtype {
            StDType::Bf16 => Ok(WeightKind::Bf16),
            StDType::F8E4M3 => Ok(WeightKind::Fp8),
            StDType::F32 => Ok(WeightKind::F32),
            StDType::I8 => Ok(WeightKind::Int8Row),
            other => Err(ImageGenError::Checkpoint(format!(
                "'{weight_name}' has dtype {other:?}, which no diffusion GEMM entry reads"
            ))),
        }
    }

    /// The weight side of a GEMM against this linear: every fact the
    /// loader read, as the one value the GEMM entries take, so a call site
    /// cannot forget the rotation or the row scales.
    pub(crate) fn weight(&self) -> LinWeight<'_> {
        LinWeight::of(self)
    }
}

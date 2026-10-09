//! A linear layer in its stored native format.
//!
//! [`Lin`] pairs a weight buffer with the [`WeightKind`] that selects its GEMM
//! entry point. A new weight dtype is a new [`WeightKind`] variant plus one arm
//! in [`Lin::load`]; callers that only forward `kind` to the GEMM calls stay
//! unchanged.

use shiso_core::DeviceBuffer;
use shiso_modelfile::st_header::StDType;
use shiso_vision::generator::ImageGenError;

use super::ops::{Ops, WeightKind};
use super::weights::StStore;

/// A linear layer's weight in its stored native format, with its bias widened
/// to f32 and its FP8 scale read at load.
///
/// The weight buffer is an import of the shard mapping's byte range: BF16,
/// `F8_E4M3` and F32 weights all reach the device as the bytes the checkpoint
/// wrote, and the GEMM entry point is chosen from the header dtype. Only the
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
        let kind = match entry.dtype {
            StDType::Bf16 => WeightKind::Bf16,
            StDType::F8E4M3 => WeightKind::Fp8,
            StDType::F32 => WeightKind::F32,
            other => {
                return Err(ImageGenError::Checkpoint(format!(
                    "'{weight_name}' has dtype {other:?}, which no diffusion GEMM entry reads"
                )));
            }
        };
        let scale = if kind == WeightKind::Fp8 {
            store.weight_scale(&weight_name)?
        } else {
            1.0
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
            bias,
            out_dim,
            in_dim,
        })
    }
}

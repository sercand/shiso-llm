//! One `y = x * W^T (+ b)` over every weight type an audio checkpoint carries.
//!
//! Most quantised weights go through the engine's projection, which already
//! knows every block format. Four types go through the audio GEMM instead:
//! the engine refuses bf16, runs f32 one row at a time, names f16 kernels no
//! backend exports at two to eight rows, and stages `Q8_0` through half
//! precision from nine rows up - about 1e-2 of relative error where the audio
//! GEMM, dequantising in f32, keeps 1e-5. An encoder therefore never has to ask
//! what a weight is stored as.

use shiso_core::{CoreError, CoreResult, DeviceBuffer, DeviceView};
use shiso_engine::projection::{Weight, projection};

use super::fused::MatvecPart;
use super::{BiasRowsDesc, Ops, SiluMulDesc};

/// Rows per projected call. The engine's library GEMM route only engages on
/// views sized for its prefill chunk, so a longer sequence is cut into slices
/// of this many rows; on a backend without that route the slicing costs one
/// extra launch per 512 rows and nothing else.
const PROJECTED_ROWS: usize = 512;

/// The storage of a weight the audio GEMM reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmType {
    /// IEEE single.
    F32,
    /// Brain float.
    Bf16,
    /// IEEE half.
    F16,
    /// GGUF `Q8_0`: per 32 elements of a row, an f16 scale and 32 signed bytes.
    Q8_0,
}

impl GemmType {
    /// The matrix-vector kernel for this weight type.
    pub(crate) fn gemv_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_gemv_f32w",
            Self::Bf16 => "kernel_audio_gemv_bf16w",
            Self::F16 => "kernel_audio_gemv_f16w",
            Self::Q8_0 => "kernel_audio_gemv_q8_0w",
        }
    }

    /// The matrix-vector-by-runs kernel for this weight type.
    pub(crate) fn matvec_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_matvec_f32w",
            Self::Bf16 => "kernel_audio_matvec_bf16w",
            Self::F16 => "kernel_audio_matvec_f16w",
            Self::Q8_0 => "kernel_audio_matvec_q8_0w",
        }
    }

    /// The three-product matrix-vector-by-runs kernel for this weight type.
    pub(crate) fn matvec3_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_matvec3_f32w",
            Self::Bf16 => "kernel_audio_matvec3_bf16w",
            Self::F16 => "kernel_audio_matvec3_f16w",
            Self::Q8_0 => "kernel_audio_matvec3_q8_0w",
        }
    }

    /// The gated two-product matrix-vector-by-runs kernel for this type.
    pub(crate) fn matvec_silu_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_matvec_silu_f32w",
            Self::Bf16 => "kernel_audio_matvec_silu_bf16w",
            Self::F16 => "kernel_audio_matvec_silu_f16w",
            Self::Q8_0 => "kernel_audio_matvec_silu_q8_0w",
        }
    }

    pub(crate) fn entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_gemm_f32w",
            Self::Bf16 => "kernel_audio_gemm_bf16w",
            Self::F16 => "kernel_audio_gemm_f16w",
            Self::Q8_0 => "kernel_audio_gemm_q8_0w",
        }
    }

    /// The 64-row tiled GEMM for this weight type.
    pub(crate) fn entry64(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_gemm64_f32w",
            Self::Bf16 => "kernel_audio_gemm64_bf16w",
            Self::F16 => "kernel_audio_gemm64_f16w",
            Self::Q8_0 => "kernel_audio_gemm64_q8_0w",
        }
    }

    /// The embedding-row fetch for a table in this storage.
    pub(crate) fn embed_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_embed_token_f32w",
            Self::Bf16 => "kernel_audio_embed_token_bf16w",
            Self::F16 => "kernel_audio_embed_token_f16w",
            Self::Q8_0 => "kernel_audio_embed_token_q8_0w",
        }
    }

    /// Bytes of one `in_dim`-wide row, or `None` when `in_dim` is not a whole
    /// number of blocks.
    pub(crate) fn row_bytes(self, in_dim: usize) -> Option<usize> {
        match self {
            Self::F32 => Some(in_dim * 4),
            Self::Bf16 | Self::F16 => Some(in_dim * 2),
            Self::Q8_0 => in_dim.is_multiple_of(32).then_some(in_dim / 32 * 34),
        }
    }
}

/// Where a linear's weight lives and how it is multiplied.
#[derive(Debug)]
pub enum LinearWeight {
    /// Quantised or otherwise block-formatted, dispatched by the engine.
    Projected(Weight),
    /// Row-major `[out_dim][in_dim]` in `dtype`'s storage, on the audio GEMM.
    Gemm {
        /// Storage.
        dtype: GemmType,
        /// The rows.
        buffer: DeviceBuffer,
        /// Row length: the input width.
        in_dim: usize,
        /// Rows: the output width.
        out_dim: usize,
    },
}

/// A linear layer: a weight and an optional f32 bias.
#[derive(Debug)]
pub struct Linear {
    /// The weight.
    pub weight: LinearWeight,
    /// `[out_dim]`, when the layer has one.
    pub bias: Option<DeviceBuffer>,
}

impl Linear {
    /// Input width.
    #[must_use]
    pub fn in_dim(&self) -> usize {
        match &self.weight {
            LinearWeight::Projected(w) => w.in_dim as usize,
            LinearWeight::Gemm { in_dim, .. } => *in_dim,
        }
    }

    /// Output width.
    #[must_use]
    pub fn out_dim(&self) -> usize {
        match &self.weight {
            LinearWeight::Projected(w) => w.out_dim as usize,
            LinearWeight::Gemm { out_dim, .. } => *out_dim,
        }
    }

    /// `y[rows][out] = x[rows][in] * W^T (+ b)`.
    ///
    /// The audio GEMM's weights take its matrix-vector kernel at a few rows;
    /// [`Self::step`] is the faster one-row form a decoder step takes.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a view or weight shorter than the
    /// shape needs, or the launch's error.
    pub fn forward(
        &self,
        ops: &Ops,
        x: DeviceView<'_>,
        y: DeviceView<'_>,
        rows: usize,
    ) -> CoreResult<()> {
        let (k, n) = (self.in_dim(), self.out_dim());
        let bias = self.bias.as_ref().map(DeviceBuffer::view);
        match &self.weight {
            LinearWeight::Gemm { dtype, buffer, .. } => {
                let row = dtype.row_bytes(k).ok_or(CoreError::InvalidValue {
                    key: "linear.in_dim",
                    value: k.to_string(),
                })?;
                let want = u64::try_from(n * row).unwrap_or(u64::MAX);
                if buffer.len() < want {
                    return Err(CoreError::ShapeMismatch {
                        field: "linear.weight",
                        expected: want,
                        got: buffer.len(),
                    });
                }
                ops.gemm(*dtype, y, x, buffer.view(), bias, (rows, n, k))
            }
            LinearWeight::Projected(weight) => {
                let mut start = 0;
                while start < rows {
                    let take = PROJECTED_ROWS.min(rows - start);
                    let xs = x.slice(bytes(start * k), bytes(take * k))?;
                    let ys = y.slice(bytes(start * n), bytes(take * n))?;
                    let tok = u32::try_from(take).map_err(|_| CoreError::InvalidValue {
                        key: "linear.rows",
                        value: take.to_string(),
                    })?;
                    projection(&**ops.backend(), weight, tok, xs, ys)?;
                    if let Some(b) = bias {
                        ops.bias_rows(&BiasRowsDesc {
                            y: ys,
                            bias: b,
                            rows: take,
                            cols: n,
                        })?;
                    }
                    start += take;
                }
                Ok(())
            }
        }
    }
}

impl Linear {
    /// [`Self::forward`] for a decoder step's row or few: the audio GEMM's
    /// weights go to the matrix-vector-by-runs kernel, whose summation order
    /// differs from the matrix-vector kernel [`Self::forward`] takes at a few
    /// rows, so a model uses one or the other for each product. Other weights,
    /// and shapes that kernel does not take, go through [`Self::forward`].
    ///
    /// # Errors
    ///
    /// As [`Self::forward`].
    pub fn step(
        &self,
        ops: &Ops,
        x: DeviceView<'_>,
        y: DeviceView<'_>,
        rows: usize,
    ) -> CoreResult<()> {
        if let LinearWeight::Gemm { dtype, buffer, .. } = &self.weight {
            let (k, n) = (self.in_dim(), self.out_dim());
            let row = dtype.row_bytes(k).ok_or(CoreError::InvalidValue {
                key: "linear.in_dim",
                value: k.to_string(),
            })?;
            let want = u64::try_from(n * row).unwrap_or(u64::MAX);
            if buffer.len() < want {
                return Err(CoreError::ShapeMismatch {
                    field: "linear.weight",
                    expected: want,
                    got: buffer.len(),
                });
            }
            let bias = self.bias.as_ref().map(DeviceBuffer::view);
            if ops.matvec(*dtype, y, x, buffer.view(), bias, (rows, n, k))? {
                return Ok(());
            }
        }
        self.forward(ops, x, y, rows)
    }
}

/// A weight the matrix-vector-by-runs kernels read, with its type, rows,
/// depth and columns, or `None` for a weight they do not.
fn runs_weight(linear: &Linear) -> Option<(GemmType, &DeviceBuffer, usize, usize)> {
    match &linear.weight {
        LinearWeight::Gemm {
            dtype,
            buffer,
            in_dim,
            out_dim,
        } if linear.bias.is_none() => {
            let want = u64::try_from(out_dim * dtype.row_bytes(*in_dim)?).ok()?;
            (buffer.len() >= want).then_some((*dtype, buffer, *in_dim, *out_dim))
        }
        _ => None,
    }
}

/// A decoder step's three projections of one row of `x` - query, key and
/// value - as [`Linear::step`] forms each, in one launch when all three are
/// bias-free audio GEMM weights of one type and depth, else one after
/// another.
///
/// # Errors
///
/// As [`Linear::forward`].
pub fn step3(
    ops: &Ops,
    linears: [&Linear; 3],
    x: DeviceView<'_>,
    ys: [DeviceView<'_>; 3],
) -> CoreResult<()> {
    let found = linears.map(runs_weight);
    if let [Some(a), Some(b), Some(c)] = found
        && a.0 == b.0
        && b.0 == c.0
        && a.2 == b.2
        && b.2 == c.2
    {
        let parts = [(a, ys[0]), (b, ys[1]), (c, ys[2])].map(|((_, w, _, n), out)| MatvecPart {
            out,
            weight: w.view(),
            cols: n,
        });
        if ops.matvec3(a.0, parts, x, a.2)? {
            return Ok(());
        }
    }
    for (linear, y) in linears.into_iter().zip(ys) {
        linear.step(ops, x, y, 1)?;
    }
    Ok(())
}

/// A SiLU-gated MLP's projections of `rows` rows of `x` and their product,
/// `out = silu(x * gate^T) * (x * up^T)`, as [`Linear::step`] and
/// [`Ops::silu_mul`] form them: in one launch when both are bias-free audio
/// GEMM weights of one type and shape, else as those three launches with
/// `spare` holding the value projection.
///
/// # Errors
///
/// As [`Linear::forward`].
pub fn step_silu(
    ops: &Ops,
    [gate, up]: [&Linear; 2],
    x: DeviceView<'_>,
    out: DeviceView<'_>,
    spare: DeviceView<'_>,
    rows: usize,
) -> CoreResult<()> {
    if let (Some(g), Some(u)) = (runs_weight(gate), runs_weight(up))
        && g.0 == u.0
        && (g.2, g.3) == (u.2, u.3)
        && ops.matvec_silu(g.0, out, x, [g.1.view(), u.1.view()], (rows, g.3, g.2))?
    {
        return Ok(());
    }
    gate.step(ops, x, out, rows)?;
    up.step(ops, x, spare, rows)?;
    ops.silu_mul(&SiluMulDesc {
        out,
        gate: out,
        up: spare,
        count: rows * gate.out_dim(),
    })
}

/// Byte length of `floats` f32 values.
fn bytes(floats: usize) -> u64 {
    u64::try_from(floats).unwrap_or(u64::MAX).saturating_mul(4)
}

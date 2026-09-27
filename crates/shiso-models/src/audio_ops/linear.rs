//! One `y = x * W^T (+ b)` over every weight type an audio checkpoint carries.
//!
//! Most quantised weights go through the engine's projection, which already
//! knows every block format. Four types go through the audio GEMM instead:
//! the engine refuses bf16, runs f32 one row at a time, names f16 kernels no
//! backend exports at two to eight rows, and stages `Q8_0` through half
//! precision from nine rows up - about 1e-2 of relative error where the audio
//! GEMM, dequantising in f32, keeps 1e-5. An encoder therefore never has to ask
//! what a weight is stored as.

use shiso_core::{CoreError, CoreResult, DType, DeviceBuffer, DeviceView};
use shiso_engine::projection::{Weight, WeightAux, projection};

use super::fused::MatvecPart;
use super::runs::{AFFINE_GROUP, Affine8, WeightView};
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

    /// The matrix-vector-by-runs kernel over blocks of rows.
    pub(crate) fn matvec_rows_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_matvec_rows_f32w",
            Self::Bf16 => "kernel_audio_matvec_rows_bf16w",
            Self::F16 => "kernel_audio_matvec_rows_f16w",
            Self::Q8_0 => "kernel_audio_matvec_rows_q8_0w",
        }
    }

    /// The gated two-product kernel over blocks of rows.
    pub(crate) fn matvec_silu_rows_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_matvec_silu_rows_f32w",
            Self::Bf16 => "kernel_audio_matvec_silu_rows_bf16w",
            Self::F16 => "kernel_audio_matvec_silu_rows_f16w",
            Self::Q8_0 => "kernel_audio_matvec_silu_rows_q8_0w",
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

    /// The batched embedding-row fetch for a table in this storage: one row
    /// per sequence's state.
    pub(crate) fn embed_batch_entry(self) -> &'static str {
        match self {
            Self::F32 => "kernel_audio_embed_token_batch_f32w",
            Self::Bf16 => "kernel_audio_embed_token_batch_bf16w",
            Self::F16 => "kernel_audio_embed_token_batch_f16w",
            Self::Q8_0 => "kernel_audio_embed_token_batch_q8_0w",
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
    /// The weight as the audio set's own kernels read it: the audio GEMM's
    /// rows (checked against the shape), or an 8-bit affine weight with its
    /// scales and biases; `None` for any other projected weight.
    fn own(&self) -> CoreResult<Option<WeightView<'_>>> {
        let (k, n) = (self.in_dim(), self.out_dim());
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
                Ok(Some(WeightView::Dense(*dtype, buffer.view())))
            }
            LinearWeight::Projected(w) => match (w.dtype, w.aux().and_then(WeightAux::affine)) {
                (DType::MlxAffine8Gs64, Some(aux)) if k.is_multiple_of(AFFINE_GROUP) => {
                    let (words, groups) = (n * k, n * k / AFFINE_GROUP);
                    let q = w.view();
                    if q.len() < bytes(words / 4)
                        || aux.scales.len() < u64::try_from(groups * 2).unwrap_or(u64::MAX)
                        || aux.biases.len() < u64::try_from(groups * 2).unwrap_or(u64::MAX)
                    {
                        return Err(CoreError::ShapeMismatch {
                            field: "linear.affine",
                            expected: bytes(words / 4),
                            got: q.len(),
                        });
                    }
                    Ok(Some(WeightView::Affine8(Affine8 {
                        q,
                        scales: aux.scales.view(),
                        biases: aux.biases.view(),
                    })))
                }
                _ => Ok(None),
            },
        }
    }

    /// Whether every row of a product over this weight sums exactly as it
    /// would alone, at any row count: true for the weights the audio set's
    /// own GEMM and matrix-vector-by-runs kernels read (the audio GEMM's, and
    /// 8-bit affine ones, at a depth those kernels take); false for any
    /// other projected weight, whose engine kernels pick their tiling and
    /// split from the row count.
    #[must_use]
    pub fn rows_sum_alone(&self) -> bool {
        matches!(self.own(), Ok(Some(w)) if w.runs_fit(self.in_dim()))
    }

    /// [`Self::forward`] for a decoder step's row or few: the audio GEMM's
    /// weights and 8-bit affine ones go to the matrix-vector-by-runs kernel,
    /// whose summation order differs from the matrix-vector kernel
    /// [`Self::forward`] takes at a few rows, so a model uses one or the
    /// other for each product. Other weights, and shapes that kernel does not
    /// take, go through [`Self::forward`].
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
        if let Some(w) = self.own()? {
            let bias = self.bias.as_ref().map(DeviceBuffer::view);
            if ops.matvec(w, y, x, bias, (rows, self.out_dim(), self.in_dim()))? {
                return Ok(());
            }
        }
        self.forward(ops, x, y, rows)
    }

    /// [`Self::forward`], but an 8-bit affine weight goes to the audio set's
    /// own tiled GEMM - a prefill's rows at the rate the dense weights get,
    /// where the engine's affine projection runs at a fifth of it - and
    /// forms each weight as `q * scale + bias` in f32. Every other weight is
    /// exactly [`Self::forward`].
    ///
    /// # Errors
    ///
    /// As [`Self::forward`].
    pub fn forward_own(
        &self,
        ops: &Ops,
        x: DeviceView<'_>,
        y: DeviceView<'_>,
        rows: usize,
    ) -> CoreResult<()> {
        if let (LinearWeight::Projected(_), Some(w)) = (&self.weight, self.own()?) {
            let bias = self.bias.as_ref().map(DeviceBuffer::view);
            return ops.gemm_of(w, y, x, bias, (rows, self.out_dim(), self.in_dim()));
        }
        self.forward(ops, x, y, rows)
    }
}

/// A bias-free weight the matrix-vector-by-runs kernels read, with its depth
/// and columns, or `None`.
fn runs_weight(linear: &Linear) -> Option<(WeightView<'_>, usize, usize)> {
    if linear.bias.is_some() {
        return None;
    }
    let w = linear.own().ok()??;
    Some((w, linear.in_dim(), linear.out_dim()))
}

/// A decoder step's three projections of one row of `x` - query, key and
/// value - as [`Linear::step`] forms each, in one launch when all three are
/// bias-free weights of one kind and depth that the one-row kernels read,
/// else one after another.
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
        && a.1 == b.1
        && b.1 == c.1
    {
        let parts = [(a, ys[0]), (b, ys[1]), (c, ys[2])].map(|((w, _, n), out)| MatvecPart {
            out,
            weight: w,
            cols: n,
        });
        if ops.matvec3(&parts, x, a.1)? {
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
/// [`Ops::silu_mul`] form them: in one launch when both are bias-free
/// weights of one kind and shape that the one-row kernels read, else as
/// those three launches with `spare` holding the value projection.
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
        && (g.1, g.2) == (u.1, u.2)
        && ops.matvec_silu(out, x, [g.0, u.0], (rows, g.2, g.1))?
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

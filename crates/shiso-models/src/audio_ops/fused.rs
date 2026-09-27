//! A causal decoder step's fused launches: each does the work of a chain of
//! the plain launches in one, forming every value exactly as that chain does,
//! so a decoder takes whichever it likes and gets the same numbers. A step
//! of one row costs a few microseconds of launch per kernel whatever it does,
//! and a dense decoder's step is some twenty small kernels a layer.

use shiso_core::{CoreResult, DeviceView, Extent3, KernelArg};

use super::args::{ArgmaxSplitArgs, GemmArgs, Matvec3Args, NormArgs, QkRopeArgs};
use super::{
    CachePlanes, DECODE_STATE_WORDS, GemmType, LINEAR_BLOCK, MATVEC_BLOCK, MATVEC_COLS, MATVEC_RUN,
    NORM_SMEM, Ops, dim, limit, need, need_cache,
};

/// Columns one threadgroup of the split argmax scans.
const ARGMAX_SPAN: usize = 2048;

/// `x += add`, then `out = rms_norm(x) (* weight)`: a residual add and the
/// RMS norm after it, as [`Ops::axpy`] at alpha 1 and [`Ops::rms_norm`] form
/// them.
#[derive(Debug, Clone, Copy)]
pub struct AddRmsNormDesc<'a> {
    /// Destination, `[rows][dim]`; may be `add`, not `x`.
    pub out: DeviceView<'a>,
    /// The residual stream, `[rows][dim]`, added to in place.
    pub x: DeviceView<'a>,
    /// What is added, `[rows][dim]`.
    pub add: DeviceView<'a>,
    /// Per-dimension scale.
    pub weight: Option<DeviceView<'a>>,
    /// Rows.
    pub rows: usize,
    /// Row width.
    pub dim: usize,
    /// Added to the mean square.
    pub eps: f32,
}

/// A decoder step's per-head query and key RMS norms, their half-split
/// rotation at the state's position and the key and value append, as
/// [`Ops::rms_norm`], [`Ops::rope_half_at_state`] (offset 0) and
/// [`Ops::kv_append`] form them.
#[derive(Debug, Clone, Copy)]
pub struct QkRopeAppendDesc<'a> {
    /// Queries, `[rows][heads * head_dim]`, normalised and rotated in place.
    pub q: DeviceView<'a>,
    /// Keys, `[rows][kv_heads * head_dim]`; read only.
    pub k: DeviceView<'a>,
    /// Values, as the keys.
    pub v: DeviceView<'a>,
    /// The query norm's weight, `[head_dim]`.
    pub q_norm: DeviceView<'a>,
    /// The key norm's weight, `[head_dim]`.
    pub k_norm: DeviceView<'a>,
    /// The layer's cache; the rows land at the state's position.
    pub cache: CachePlanes<'a>,
    /// Query heads, a multiple of the cache's kv heads.
    pub heads: usize,
    /// Rows.
    pub rows: usize,
    /// Added to each head's mean square.
    pub eps: f32,
    /// Frequency base.
    pub theta: f32,
    /// The decoder state.
    pub state: DeviceView<'a>,
}

/// The first index of the largest value of one wide row, in two launches over
/// a caller-kept scratch of [`argmax_wide_parts`] words; the same answer as
/// [`Ops::argmax_rows`] gives.
#[derive(Debug, Clone, Copy)]
pub struct ArgmaxWideDesc<'a> {
    /// One `u32`.
    pub out: DeviceView<'a>,
    /// The row, `[cols]`.
    pub input: DeviceView<'a>,
    /// Scratch, at least [`argmax_wide_parts`]`(cols)` `u32` words.
    pub parts: DeviceView<'a>,
    /// Columns.
    pub cols: usize,
}

/// `u32` words of scratch [`ArgmaxWideDesc`] needs for a row of `cols`.
#[must_use]
pub const fn argmax_wide_parts(cols: usize) -> usize {
    2 * cols.div_ceil(ARGMAX_SPAN)
}

/// One of the three products [`Ops::matvec3`] takes: a destination, its
/// weight and its column count.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatvecPart<'a> {
    pub(crate) out: DeviceView<'a>,
    pub(crate) weight: DeviceView<'a>,
    pub(crate) cols: usize,
}

/// Whether the matrix-vector-by-runs kernels take a product of depth `k`
/// with these views.
fn runs_fit(k: usize, views: &[DeviceView<'_>]) -> bool {
    k.is_multiple_of(MATVEC_RUN) && views.iter().all(|v| v.offset().is_multiple_of(16))
}

impl Ops {
    /// [`AddRmsNormDesc`]: the residual add and the RMS norm after it.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::ShapeMismatch`] for a short view,
    /// [`shiso_core::CoreError::InvalidValue`] for a zero width, or the
    /// launch's error.
    pub fn add_rms_norm(&self, d: &AddRmsNormDesc<'_>) -> CoreResult<()> {
        limit(d.dim > 0, "add_rms_norm.dim", d.dim)?;
        let count = d.rows * d.dim;
        need(d.out, count, "add_rms_norm.out")?;
        need(d.x, count, "add_rms_norm.x")?;
        need(d.add, count, "add_rms_norm.add")?;
        if let Some(w) = d.weight {
            need(w, d.dim, "add_rms_norm.weight")?;
        }
        let args = NormArgs {
            rows: dim(d.rows, "add_rms_norm.rows")?,
            dim: dim(d.dim, "add_rms_norm.dim")?,
            eps: d.eps,
            flags: u32::from(d.weight.is_some()),
        };
        self.launch(
            "kernel_audio_add_rms_norm_f32",
            Extent3::linear(dim(d.rows.max(1), "add_rms_norm.rows")?),
            LINEAR_BLOCK,
            NORM_SMEM,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.x),
                KernelArg::Buffer(d.add),
                KernelArg::Buffer(d.weight.unwrap_or(d.x)),
            ],
        )
    }

    /// [`QkRopeAppendDesc`]: the query and key norms, the rotation and the
    /// append of a decoder step.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::InvalidValue`] for an odd or over-wide head,
    /// or heads that are not a multiple of the kv heads;
    /// [`shiso_core::CoreError::ShapeMismatch`] for a short view; or the
    /// launch's error.
    pub fn qk_norm_rope_append(&self, d: &QkRopeAppendDesc<'_>) -> CoreResult<()> {
        let c = &d.cache;
        need_cache(c, d.heads, "qk_norm_rope_append.heads")?;
        limit(
            c.head_dim.is_multiple_of(2),
            "qk_norm_rope_append.head_dim",
            c.head_dim,
        )?;
        need(d.q, d.rows * d.heads * c.head_dim, "qk_norm_rope_append.q")?;
        need(
            d.k,
            d.rows * c.kv_heads * c.head_dim,
            "qk_norm_rope_append.k",
        )?;
        need(
            d.v,
            d.rows * c.kv_heads * c.head_dim,
            "qk_norm_rope_append.v",
        )?;
        need(d.q_norm, c.head_dim, "qk_norm_rope_append.q_norm")?;
        need(d.k_norm, c.head_dim, "qk_norm_rope_append.k_norm")?;
        need(d.state, DECODE_STATE_WORDS, "qk_norm_rope_append.state")?;
        let args = QkRopeArgs {
            rows: dim(d.rows, "qk_norm_rope_append.rows")?,
            heads: dim(d.heads, "qk_norm_rope_append.heads")?,
            kv_heads: dim(c.kv_heads, "qk_norm_rope_append.kv_heads")?,
            head_dim: dim(c.head_dim, "qk_norm_rope_append.head_dim")?,
            cap: dim(c.cap, "qk_norm_rope_append.cap")?,
            eps: d.eps,
            theta: d.theta,
        };
        self.launch(
            "kernel_audio_qk_norm_rope_append_f32",
            Extent3::new(
                dim(d.heads + c.kv_heads, "qk_norm_rope_append.grid")?,
                dim(d.rows.max(1), "qk_norm_rope_append.grid")?,
                1,
            ),
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.q),
                KernelArg::Buffer(d.k),
                KernelArg::Buffer(d.v),
                KernelArg::Buffer(d.q_norm),
                KernelArg::Buffer(d.k_norm),
                KernelArg::Buffer(c.k),
                KernelArg::Buffer(c.v),
                KernelArg::Buffer(d.state),
            ],
        )
    }

    /// [`ArgmaxWideDesc`]: one wide row's argmax in two launches.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::ShapeMismatch`] for a short view, or the
    /// launch's error.
    pub fn argmax_wide(&self, d: &ArgmaxWideDesc<'_>) -> CoreResult<()> {
        need(d.out, 1, "argmax_wide.out")?;
        need(d.input, d.cols, "argmax_wide.input")?;
        need(d.parts, argmax_wide_parts(d.cols), "argmax_wide.parts")?;
        let parts = d.cols.div_ceil(ARGMAX_SPAN).max(1);
        let args = ArgmaxSplitArgs {
            cols: dim(d.cols, "argmax_wide.cols")?,
            span: dim(ARGMAX_SPAN, "argmax_wide.span")?,
            parts: dim(parts, "argmax_wide.parts")?,
        };
        let bytes = bytemuck::bytes_of(&args);
        self.launch(
            "kernel_audio_argmax_split_f32",
            Extent3::linear(args.parts),
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytes),
                KernelArg::Buffer(d.parts),
                KernelArg::Buffer(d.input),
            ],
        )?;
        self.launch(
            "kernel_audio_argmax_merge_f32",
            Extent3::linear(1),
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytes),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.parts),
            ],
        )
    }

    /// Three matrix-vector-by-runs products of one row of `input` (depth
    /// `k`) in one launch, each column summed as [`Ops::matvec`] sums it; no
    /// bias. Returns `false`, launching nothing, where [`Ops::matvec`] would.
    pub(crate) fn matvec3(
        &self,
        dtype: GemmType,
        parts: [MatvecPart<'_>; 3],
        input: DeviceView<'_>,
        k: usize,
    ) -> CoreResult<bool> {
        if !runs_fit(
            k,
            &[input, parts[0].weight, parts[1].weight, parts[2].weight],
        ) {
            return Ok(false);
        }
        need(input, k, "matvec3.x")?;
        let mut groups = 0;
        for p in &parts {
            need(p.out, p.cols, "matvec3.y")?;
            groups += p.cols.div_ceil(MATVEC_COLS);
        }
        let args = Matvec3Args {
            n0: dim(parts[0].cols, "matvec3.n0")?,
            n1: dim(parts[1].cols, "matvec3.n1")?,
            n2: dim(parts[2].cols, "matvec3.n2")?,
            k: dim(k, "matvec3.k")?,
        };
        self.launch(
            dtype.matvec3_entry(),
            Extent3::linear(dim(groups.max(1), "matvec3.grid")?),
            MATVEC_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(parts[0].out),
                KernelArg::Buffer(parts[1].out),
                KernelArg::Buffer(parts[2].out),
                KernelArg::Buffer(input),
                KernelArg::Buffer(parts[0].weight),
                KernelArg::Buffer(parts[1].weight),
                KernelArg::Buffer(parts[2].weight),
            ],
        )?;
        Ok(true)
    }

    /// `out = silu(x * Wg^T) * (x * Wu^T)` over `rows` rows in one launch,
    /// each product summed as [`Ops::matvec`] sums it and the product formed
    /// as [`Ops::silu_mul`] forms it. Returns `false`, launching nothing,
    /// where [`Ops::matvec`] would.
    pub(crate) fn matvec_silu(
        &self,
        dtype: GemmType,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        [gate, up]: [DeviceView<'_>; 2],
        (rows, cols, depth): (usize, usize, usize),
    ) -> CoreResult<bool> {
        if !runs_fit(depth, &[input, gate, up]) {
            return Ok(false);
        }
        need(out, rows * cols, "matvec_silu.y")?;
        need(input, rows * depth, "matvec_silu.x")?;
        let args = GemmArgs {
            m: dim(rows, "matvec_silu.m")?,
            n: dim(cols, "matvec_silu.n")?,
            k: dim(depth, "matvec_silu.k")?,
            flags: 0,
            k_chunk: dim(depth, "matvec_silu.k")?,
            splits: 1,
        };
        self.launch(
            dtype.matvec_silu_entry(),
            Extent3::new(
                dim(cols.div_ceil(MATVEC_COLS).max(1), "matvec_silu.grid")?,
                dim(rows.max(1), "matvec_silu.grid")?,
                1,
            ),
            MATVEC_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(input),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(up),
            ],
        )?;
        Ok(true)
    }
}

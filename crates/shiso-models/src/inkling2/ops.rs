//! Device kernel argument blocks and launch wrappers for the `inkling2` module.

#![expect(
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    reason = "thin kernel dispatch wrappers whose errors are always backend launch failures"
)]

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::audio_ops::{GemmType, Ops};
use shiso_core::{
    Backend, CoreError, CoreResult, DeviceView, Extent3, KernelArg, KernelRef, LaunchArgs,
};

/// Kernel registry module name shared by Metal and CUDA.
pub const MODULE: &str = "inkling2";

/// Arguments for `kernel_inkling2_embed_rmsnorm_bf16` and `kernel_inkling2_rmsnorm_bf16`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NormArgs {
    /// Batch / sequence rows.
    pub rows: u32,
    /// Hidden width.
    pub hidden: u32,
    /// RMSNorm epsilon.
    pub eps: f32,
}

/// Arguments for `kernel_inkling2_linear_bf16` and `kernel_inkling2_vit_linear_bias_bf16`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LinearArgs {
    /// Input rows.
    pub rows: u32,
    /// Input width.
    pub in_dim: u32,
    /// Output width.
    pub out_dim: u32,
}

/// Arguments for `kernel_inkling2_qkvr_gate_sink_bf16`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct QkvrGateSinkArgs {
    /// Input rows.
    pub rows: u32,
    /// Input width.
    pub in_dim: u32,
    /// Query and output-gate width (`n_head * head_dim`).
    pub q_dim: u32,
    /// Key and value width (`n_head_kv * head_dim`).
    pub kv_dim: u32,
    /// Relative position projection width (`n_head * d_rel`).
    pub r_dim: u32,
    /// Attention head count (per-query sink width).
    pub n_head: u32,
}

/// Device buffers for `kernel_inkling2_qkvr_gate_sink_bf16`.
#[derive(Debug)]
pub struct QkvrGateSinkViews<'a> {
    /// Query / R / Gate / Sink input `[rows, in_dim]`.
    pub x_q: DeviceView<'a>,
    /// Key input `[rows, in_dim]`.
    pub x_k: DeviceView<'a>,
    /// Value input `[rows, in_dim]`.
    pub x_v: DeviceView<'a>,
    /// Query weight `[q_dim, in_dim]` (BF16).
    pub wq: DeviceView<'a>,
    /// Key weight `[kv_dim, in_dim]` (BF16).
    pub wk: DeviceView<'a>,
    /// Value weight `[kv_dim, in_dim]` (BF16).
    pub wv: DeviceView<'a>,
    /// Relative weight `[r_dim, in_dim]` (BF16).
    pub wr: DeviceView<'a>,
    /// Output gate weight `[q_dim, in_dim]` (BF16).
    pub wgate: DeviceView<'a>,
    /// Per-query sink weight `[n_head, in_dim]` (BF16).
    pub wsink: DeviceView<'a>,
    /// Query output `[rows, q_dim]`.
    pub out_q: DeviceView<'a>,
    /// Key output `[rows, kv_dim]`.
    pub out_k: DeviceView<'a>,
    /// Value output `[rows, kv_dim]`.
    pub out_v: DeviceView<'a>,
    /// Relative output `[rows, r_dim]`.
    pub out_r: DeviceView<'a>,
    /// Output gate `[rows, q_dim]`.
    pub out_gate: DeviceView<'a>,
    /// Per-query sink output `[rows, n_head]`.
    pub out_sink: DeviceView<'a>,
}

/// Arguments for `kernel_inkling2_lm_head_softcap`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LmHeadArgs {
    /// Input rows.
    pub rows: u32,
    /// Hidden width.
    pub hidden: u32,
    /// Unpadded vocabulary size.
    pub vocab: u32,
    /// Softcap constant.
    pub softcap: f32,
}

/// Arguments for `kernel_inkling2_attn_res`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AttnResArgs {
    /// Batch rows.
    pub rows: u32,
    /// Hidden width.
    pub hidden: u32,
    /// Number of completed blocks in `blocks`.
    pub n_blocks: u32,
    /// Number of queries (1, 2, or 3).
    pub n_queries: u32,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Non-zero when `addend` should be added to `partial`.
    pub has_addend: u32,
}

/// Arguments for `kernel_inkling2_loop_control`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LoopCtrlArgs {
    /// Batch rows.
    pub rows: u32,
    /// Hidden width.
    pub hidden: u32,
    /// Controller bottleneck rank.
    pub rank: u32,
    /// RMSNorm epsilon.
    pub eps: f32,
}

/// Arguments for `kernel_inkling2_short_conv_static`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SconvStaticArgs {
    /// Sequence rows.
    pub rows: u32,
    /// Channel count.
    pub channels: u32,
    /// Convolution kernel width.
    pub kernel_size: u32,
    /// Output scale (`1 / sqrt(k + 1)`).
    pub scale: f32,
    /// Non-zero for a pass whose rows are not all kept: the lags are left as
    /// found and the pass's inputs are written behind them for
    /// [`sconv_commit`].
    pub defer_state: u32,
}

/// Arguments for `kernel_inkling2_short_conv_dynamic`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SconvDynamicArgs {
    /// Sequence rows.
    pub rows: u32,
    /// KV head count.
    pub kv_heads: u32,
    /// Head width.
    pub head_dim: u32,
    /// Convolution kernel width.
    pub kernel_size: u32,
    /// Output scale (`1 / sqrt(2*k + 1)`).
    pub scale: f32,
    /// As [`SconvStaticArgs::defer_state`].
    pub defer_state: u32,
}

/// Arguments for `kernel_inkling2_sconv_commit`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SconvCommitArgs {
    /// Channels across every layer's four states: `layers * 2 * (kv_dim + hidden)`.
    pub total: u32,
    /// Width of the K and V states.
    pub kv_dim: u32,
    /// Width of the post-attention and post-MLP states.
    pub hidden: u32,
    /// Lag rows a state carries (`k - 1`).
    pub lags: u32,
    /// Rows a deferring pass may write behind the lags.
    pub hist_rows: u32,
    /// Rows of the deferring pass that were kept.
    pub keep: u32,
}
const _: () = assert!(size_of::<SconvStaticArgs>() == 20);
const _: () = assert!(size_of::<SconvDynamicArgs>() == 24);
const _: () = assert!(size_of::<SconvCommitArgs>() == 24);

/// Arguments for `kernel_inkling2_qkvr_prep_store`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct QkvrPrepArgs {
    /// Sequence rows.
    pub rows: u32,
    /// Query heads.
    pub n_head: u32,
    /// KV heads.
    pub n_head_kv: u32,
    /// Head width.
    pub head_dim: u32,
    /// Relative position width per head.
    pub d_rel: u32,
    /// Q/K L2 clip threshold.
    pub qk_clip: f32,
    /// Relative state L2 clip threshold.
    pub rel_clip: f32,
    /// RMSNorm epsilon for V.
    pub eps: f32,
    /// Non-zero for full-attention layers (enables log scaling).
    pub is_full: u32,
    /// Log-scaling floor.
    pub log_n_floor: f32,
    /// Log-scaling alpha.
    pub log_alpha: f32,
    /// Position of row 0.
    pub pos0: u32,
    /// KV cache capacity in slots.
    pub kv_cap: u32,
    /// Non-zero when KV cache is a ring of `kv_cap`.
    pub is_swa: u32,
}

/// Arguments for `kernel_inkling2_rel_attn`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RelAttnArgs {
    /// Sequence rows.
    pub rows: u32,
    /// Query heads.
    pub n_head: u32,
    /// KV heads.
    pub n_head_kv: u32,
    /// Head width.
    pub head_dim: u32,
    /// Relative position extent.
    pub rel_extent: u32,
    /// Position of row 0.
    pub pos0: u32,
    /// KV cache capacity in slots.
    pub kv_cap: u32,
    /// Sliding window size.
    pub window_size: u32,
    /// Non-zero for sliding-window attention.
    pub is_swa: u32,
    /// Attention scale (`1 / sqrt(head_dim)`).
    pub scale: f32,
}

/// Arguments for `kernel_inkling2_route_topk`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RouteArgs {
    /// Sequence rows.
    pub rows: u32,
    /// Total routed experts.
    pub n_experts: u32,
    /// Experts selected per token.
    pub top_k: u32,
    /// Scaling factor after normalization.
    pub route_scale: f32,
}

/// Arguments for NVFP4 and MXFP8 MoE W13 kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MoeW13Args {
    /// Sequence rows.
    pub rows: u32,
    /// Input width.
    pub in_dim: u32,
    /// Per-expert intermediate width.
    pub inter_dim: u32,
    /// Active experts per token.
    pub top_k: u32,
    /// Gate softcap.
    pub gate_cap: f32,
    /// Up softcap.
    pub up_cap: f32,
}

/// Arguments for NVFP4 and MXFP8 MoE W2 kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MoeW2Args {
    /// Sequence rows.
    pub rows: u32,
    /// Per-expert intermediate width.
    pub inter_dim: u32,
    /// Output width.
    pub out_dim: u32,
    /// Active experts per token.
    pub top_k: u32,
}

/// Arguments for elementwise unary/add kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AddArgs {
    /// Element count.
    pub count: u32,
}

/// Fewest rows a BF16 projection takes the tiled GEMM at.
///
/// Below it the per-row kernels stream each weight row once for up to eight
/// rows; above it the 64-row simdgroup-matrix tile of the shared GEMM set
/// multiplies. A prefill pass of 256 rows spent 18.5 ms a layer in the fused
/// attention projections on the per-row kernel (M1 Max).
pub const GEMM_MIN_ROWS: u32 = 33;

/// `out[rows, out_dim] = x[rows, in_dim] @ w[out_dim, in_dim]^T` over BF16
/// weights: [`linear_bf16`] for a few rows, the tiled GEMM for a long pass.
///
/// The two sum in different orders, so a row's bits differ between a pass of
/// fewer than [`GEMM_MIN_ROWS`] rows and a longer one, and not otherwise.
///
/// # Errors
///
/// Whatever the launch reports.
pub fn linear_bf16_rows(
    gemm: &Ops,
    args: LinearArgs,
    x: DeviceView<'_>,
    w: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    if args.rows >= GEMM_MIN_ROWS && args.in_dim.is_multiple_of(16) {
        return gemm.gemm(
            GemmType::Bf16,
            out,
            x,
            w,
            None,
            (
                args.rows as usize,
                args.out_dim as usize,
                args.in_dim as usize,
            ),
        );
    }
    linear_bf16(gemm.backend().as_ref(), args, x, w, out)
}

/// The relative-position bias projection `[rows * n_head, d_rel] ->
/// [rows * n_head, rel_extent]`: [`rel_bias_bf16`] for a few rows, the tiled
/// GEMM for a long pass.
///
/// # Errors
///
/// Whatever the launch reports.
pub fn rel_bias_rows(
    gemm: &Ops,
    token_rows: u32,
    args: LinearArgs,
    r_in: DeviceView<'_>,
    w: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    if token_rows >= GEMM_MIN_ROWS {
        return linear_bf16_rows(gemm, args, r_in, w, out);
    }
    rel_bias_bf16(gemm.backend().as_ref(), args, r_in, w, out)
}

/// The six attention input projections: one fused launch for a few rows
/// ([`qkvr_gate_sink_bf16`]), six tiled GEMMs for a long pass.
///
/// # Errors
///
/// Whatever the launches report.
pub fn qkvr_gate_sink_rows(
    gemm: &Ops,
    args: QkvrGateSinkArgs,
    v: &QkvrGateSinkViews<'_>,
) -> CoreResult<()> {
    if args.rows < GEMM_MIN_ROWS || !args.in_dim.is_multiple_of(16) {
        return qkvr_gate_sink_bf16(gemm.backend().as_ref(), args, v);
    }
    let lin = |out_dim: u32| LinearArgs {
        rows: args.rows,
        in_dim: args.in_dim,
        out_dim,
    };
    linear_bf16_rows(gemm, lin(args.q_dim), v.x_q, v.wq, v.out_q)?;
    linear_bf16_rows(gemm, lin(args.kv_dim), v.x_k, v.wk, v.out_k)?;
    linear_bf16_rows(gemm, lin(args.kv_dim), v.x_v, v.wv, v.out_v)?;
    linear_bf16_rows(gemm, lin(args.r_dim), v.x_q, v.wr, v.out_r)?;
    linear_bf16_rows(gemm, lin(args.q_dim), v.x_q, v.wgate, v.out_gate)?;
    linear_bf16_rows(gemm, lin(args.n_head), v.x_q, v.wsink, v.out_sink)
}

/// Launches fused BF16 embedding lookup + RMSNorm.
pub fn embed_rmsnorm_bf16(
    backend: &dyn Backend,
    args: NormArgs,
    tokens: DeviceView<'_>,
    embed: DeviceView<'_>,
    norm_w: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_embed_rmsnorm_bf16"),
        &LaunchArgs::new(
            Extent3::linear(args.rows),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(tokens),
                KernelArg::Buffer(embed),
                KernelArg::Buffer(norm_w),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches RMSNorm with BF16 weights.
pub fn rmsnorm_bf16(
    backend: &dyn Backend,
    args: NormArgs,
    inp: DeviceView<'_>,
    w: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_rmsnorm_bf16"),
        &LaunchArgs::new(
            Extent3::linear(args.rows),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(inp),
                KernelArg::Buffer(w),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches BF16 linear projection (`F32 x @ BF16 w^T -> F32 out`).
pub fn linear_bf16(
    backend: &dyn Backend,
    args: LinearArgs,
    x: DeviceView<'_>,
    w: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_linear_bf16"),
        &LaunchArgs::new(
            Extent3::new(args.out_dim.div_ceil(4), args.rows.div_ceil(8), 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches the fused Q+K+V+R+Gate+Sink BF16 mega-projection.
pub fn qkvr_gate_sink_bf16(
    backend: &dyn Backend,
    args: QkvrGateSinkArgs,
    views: &QkvrGateSinkViews<'_>,
) -> CoreResult<()> {
    let total_out = args.q_dim + args.kv_dim + args.kv_dim + args.r_dim + args.q_dim + args.n_head;
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_qkvr_gate_sink_bf16"),
        &LaunchArgs::new(
            Extent3::new(total_out.div_ceil(4), args.rows.div_ceil(8), 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(views.x_q),
                KernelArg::Buffer(views.x_k),
                KernelArg::Buffer(views.x_v),
                KernelArg::Buffer(views.wq),
                KernelArg::Buffer(views.wk),
                KernelArg::Buffer(views.wv),
                KernelArg::Buffer(views.wr),
                KernelArg::Buffer(views.wgate),
                KernelArg::Buffer(views.wsink),
                KernelArg::Buffer(views.out_q),
                KernelArg::Buffer(views.out_k),
                KernelArg::Buffer(views.out_v),
                KernelArg::Buffer(views.out_r),
                KernelArg::Buffer(views.out_gate),
                KernelArg::Buffer(views.out_sink),
            ],
        ),
    )
}

/// Launches BF16 LM head projection with logit softcapping.
pub fn lm_head_softcap(
    backend: &dyn Backend,
    args: LmHeadArgs,
    x: DeviceView<'_>,
    w: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_lm_head_softcap"),
        &LaunchArgs::new(
            Extent3::new(args.vocab.div_ceil(4), args.rows.div_ceil(8), 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Device buffers for [`attn_res`].
#[derive(Debug)]
pub struct AttnResViews<'a> {
    /// Completed blocks `[n_blocks, rows, hidden]`.
    pub blocks: DeviceView<'a>,
    /// The current partial stream `[rows, hidden]`.
    pub partial: DeviceView<'a>,
    /// Added to `partial` when [`AttnResArgs::has_addend`] is set.
    pub addend: DeviceView<'a>,
    /// Queries `[n_queries, hidden]`.
    pub queries: DeviceView<'a>,
    /// Scratch for the sources' sums: [`attn_res_stats_bytes`] bytes.
    pub stats: DeviceView<'a>,
    /// Mixed outputs `[n_queries, rows, hidden]`.
    pub out: DeviceView<'a>,
}

/// Most sources (completed blocks and the partial stream) one mix reads.
pub const ATTN_RES_MAX_SOURCES: u32 = 16;
/// Slices of the width a source's sums are taken over.
const ATTN_RES_SLICES: u32 = 4;
/// Sums per slice: the squares, and the dot with each of up to four queries.
const ATTN_RES_STATS: u32 = 5;

/// Bytes of [`AttnResViews::stats`] a pass of `rows` rows needs.
#[must_use]
pub const fn attn_res_stats_bytes(rows: u32) -> u64 {
    rows as u64 * ATTN_RES_MAX_SOURCES as u64 * ATTN_RES_SLICES as u64 * ATTN_RES_STATS as u64 * 4
}

/// Launches block attention residual mixing: the sources' sums, then the mix.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] for more sources or queries than the kernels
/// carry, and whatever the device reports.
pub fn attn_res(backend: &dyn Backend, args: AttnResArgs, v: &AttnResViews<'_>) -> CoreResult<()> {
    if args.n_blocks + 1 > ATTN_RES_MAX_SOURCES || args.n_queries > 4 {
        return Err(CoreError::InvalidValue {
            key: "attn_res",
            value: format!(
                "{} blocks and {} queries exceed the kernels' {} sources and 4 queries",
                args.n_blocks, args.n_queries, ATTN_RES_MAX_SOURCES
            ),
        });
    }
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_attn_res_stats"),
        &LaunchArgs::new(
            Extent3::new((args.n_blocks + 1) * ATTN_RES_SLICES, args.rows, 1),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(v.blocks),
                KernelArg::Buffer(v.partial),
                KernelArg::Buffer(v.addend),
                KernelArg::Buffer(v.queries),
                KernelArg::Buffer(v.stats),
            ],
        ),
    )?;
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_attn_res_mix"),
        &LaunchArgs::new(
            Extent3::new(args.hidden.div_ceil(256), args.rows, 1),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(v.blocks),
                KernelArg::Buffer(v.partial),
                KernelArg::Buffer(v.addend),
                KernelArg::Buffer(v.stats),
                KernelArg::Buffer(v.out),
            ],
        ),
    )
}

/// Buffers for [`loop_control`].
#[derive(Debug)]
pub struct LoopCtrlViews<'a> {
    /// Current stream (`[rows, H]`).
    pub cur: DeviceView<'a>,
    /// Cross-pass message (`[rows, H]`).
    pub msg: DeviceView<'a>,
    /// Down projection (`[rank, 2*H]`).
    pub down_w: DeviceView<'a>,
    /// Gain projection (`[H, rank]`).
    pub gain_w: DeviceView<'a>,
    /// Forget projection (`[H, rank]`).
    pub forget_w: DeviceView<'a>,
    /// Scratch bottleneck buffer (`[rows, rank]`).
    pub z_scratch: DeviceView<'a>,
    /// Output stream (`[rows, H]`).
    pub out: DeviceView<'a>,
}

/// Launches parallel two-stage rank-64 cross-pass loop message controller.
pub fn loop_control(
    backend: &dyn Backend,
    args: LoopCtrlArgs,
    v: &LoopCtrlViews<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_loop_ctrl_down"),
        &LaunchArgs::new(
            Extent3::new(args.rank.div_ceil(4), args.rows, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(v.cur),
                KernelArg::Buffer(v.msg),
                KernelArg::Buffer(v.down_w),
                KernelArg::Buffer(v.z_scratch),
            ],
        ),
    )?;
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_loop_ctrl_apply"),
        &LaunchArgs::new(
            Extent3::new(args.hidden.div_ceil(4), args.rows, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(v.cur),
                KernelArg::Buffer(v.msg),
                KernelArg::Buffer(v.z_scratch),
                KernelArg::Buffer(v.gain_w),
                KernelArg::Buffer(v.forget_w),
                KernelArg::Buffer(v.out),
            ],
        ),
    )
}

/// Launches static short causal convolution.
pub fn short_conv_static(
    backend: &dyn Backend,
    args: SconvStaticArgs,
    x: DeviceView<'_>,
    w: DeviceView<'_>,
    state: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_short_conv_static"),
        &LaunchArgs::new(
            Extent3::linear(args.channels.div_ceil(256)),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w),
                KernelArg::Buffer(state),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches dynamic short causal convolution (full-attention K/V).
pub fn short_conv_dynamic(
    backend: &dyn Backend,
    args: SconvDynamicArgs,
    x: DeviceView<'_>,
    dyn_w: DeviceView<'_>,
    static_w: DeviceView<'_>,
    state: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    let channels = args.kv_heads * args.head_dim;
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_short_conv_dynamic"),
        &LaunchArgs::new(
            Extent3::linear(channels.div_ceil(256)),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(dyn_w),
                KernelArg::Buffer(static_w),
                KernelArg::Buffer(state),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Moves every short-convolution state of a session to where a pass that kept
/// `args.keep` of its rows leaves it, after that pass ran deferring.
///
/// # Errors
///
/// Whatever the device reports.
pub fn sconv_commit(
    backend: &dyn Backend,
    args: SconvCommitArgs,
    state: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_sconv_commit"),
        &LaunchArgs::new(
            Extent3::linear(args.total.div_ceil(256)),
            Extent3::linear(256),
            &[KernelArg::Bytes(bytes_of(&args)), KernelArg::Buffer(state)],
        ),
    )
}

/// Buffers for [`qkvr_prep_store`].
#[derive(Debug)]
pub struct QkvrPrepViews<'a> {
    /// Query buffer (in-place).
    pub q: DeviceView<'a>,
    /// Key buffer.
    pub k: DeviceView<'a>,
    /// Value buffer.
    pub v: DeviceView<'a>,
    /// Relative state buffer (in-place).
    pub r: DeviceView<'a>,
    /// Q L2-clip scale.
    pub q_scale: DeviceView<'a>,
    /// Relative L2-clip scale.
    pub r_scale: DeviceView<'a>,
    /// Key cache (F16).
    pub k_cache: DeviceView<'a>,
    /// Value cache (F16).
    pub v_cache: DeviceView<'a>,
}

/// Launches QKVR L2-clip / V-RMSNorm / log-scale / F16 KV cache store.
pub fn qkvr_prep_store(
    backend: &dyn Backend,
    args: QkvrPrepArgs,
    v: &QkvrPrepViews<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_qkvr_prep_store"),
        &LaunchArgs::new(
            Extent3::new(args.n_head + args.n_head_kv, args.rows, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(v.q),
                KernelArg::Buffer(v.k),
                KernelArg::Buffer(v.v),
                KernelArg::Buffer(v.r),
                KernelArg::Buffer(v.q_scale),
                KernelArg::Buffer(v.r_scale),
                KernelArg::Buffer(v.k_cache),
                KernelArg::Buffer(v.v_cache),
            ],
        ),
    )
}

/// Buffers for [`rel_attn`].
#[derive(Debug)]
pub struct RelAttnViews<'a> {
    /// Query buffer.
    pub q: DeviceView<'a>,
    /// Precomputed relative bias `[rows, n_head, rel_extent]`.
    pub rel_bias: DeviceView<'a>,
    /// Per-query attention sink logits `[rows, n_head]`.
    pub sinks: DeviceView<'a>,
    /// Output gate logits `[rows, n_head, head_dim]`.
    pub gate: DeviceView<'a>,
    /// Key cache (F16).
    pub k_cache: DeviceView<'a>,
    /// Value cache (F16).
    pub v_cache: DeviceView<'a>,
    /// Output buffer `[rows, n_head, head_dim]`.
    pub out: DeviceView<'a>,
    /// A short pass's per-run maximum and sum; see [`rel_attn_split_bytes`].
    pub split_ml: DeviceView<'a>,
    /// A short pass's per-run partial outputs.
    pub split_out: DeviceView<'a>,
}

/// Launches fast small-`in_dim` (`d_rel = 8`) relative-bias BF16 projection.
pub fn rel_bias_bf16(
    backend: &dyn Backend,
    args: LinearArgs,
    r_in: DeviceView<'_>,
    w: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_rel_bias_bf16"),
        &LaunchArgs::new(
            Extent3::new(args.out_dim.div_ceil(128), args.rows, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(r_in),
                KernelArg::Buffer(w),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Fewest rows a pass takes the attention tile at: half of its 64 query rows.
pub const REL_ATTN_TILE_MIN_ROWS: u32 = 32;

/// Launches relative-bias + per-query sink + output-gated attention.
///
/// Three kernels compute it, chosen by shape alone: a pass of
/// [`REL_ATTN_TILE_MIN_ROWS`] rows or more multiplies on simdgroup matrices,
/// 64 query rows a threadgroup; a shorter pass of 128-wide heads cuts each
/// row's keys into runs, a simdgroup each, and merges them; any other head
/// width walks its keys in one simdgroup. A decode step and a verify pass are
/// both short passes, so a position is summed the same way in either.
///
/// # Errors
///
/// Whatever the device reports.
pub fn rel_attn(backend: &dyn Backend, args: RelAttnArgs, v: &RelAttnViews<'_>) -> CoreResult<()> {
    let binds = [
        KernelArg::Bytes(bytes_of(&args)),
        KernelArg::Buffer(v.q),
        KernelArg::Buffer(v.rel_bias),
        KernelArg::Buffer(v.sinks),
        KernelArg::Buffer(v.gate),
        KernelArg::Buffer(v.k_cache),
        KernelArg::Buffer(v.v_cache),
        KernelArg::Buffer(v.out),
    ];
    if args.head_dim != 128 {
        return backend.run(
            KernelRef::new(MODULE, "kernel_inkling2_rel_attn"),
            &LaunchArgs::new(
                Extent3::new(args.n_head, args.rows.div_ceil(4), 1),
                Extent3::linear(128),
                &binds,
            ),
        );
    }
    if args.rows >= REL_ATTN_TILE_MIN_ROWS {
        return backend.run(
            KernelRef::new(MODULE, "kernel_inkling2_rel_attn_tile"),
            &LaunchArgs::new(
                Extent3::new(args.rows.div_ceil(64), args.n_head, 1),
                Extent3::linear(256),
                &binds,
            ),
        );
    }
    // The pass's last row has the most keys, a full layer's the whole prefix.
    let last = args.pos0 + args.rows - 1;
    let keys = if args.is_swa != 0 {
        (last + 1).min(args.window_size)
    } else {
        last + 1
    };
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_rel_attn_splitk"),
        &LaunchArgs::new(
            Extent3::new(args.n_head, rel_attn_splits(keys) / 4, args.rows),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(v.q),
                KernelArg::Buffer(v.rel_bias),
                KernelArg::Buffer(v.sinks),
                KernelArg::Buffer(v.k_cache),
                KernelArg::Buffer(v.v_cache),
                KernelArg::Buffer(v.split_ml),
                KernelArg::Buffer(v.split_out),
            ],
        ),
    )?;
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_rel_attn_reduce"),
        &LaunchArgs::new(
            Extent3::new(args.n_head, args.rows, 1),
            Extent3::linear(32),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(v.split_ml),
                KernelArg::Buffer(v.split_out),
                KernelArg::Buffer(v.gate),
                KernelArg::Buffer(v.out),
            ],
        ),
    )
}

/// Most runs a short pass's row is cut into, and so what a row of the split
/// scratch holds (`INK2_ATTN_MAX_SPLITS`).
pub const REL_ATTN_MAX_SPLITS: u32 = 64;

/// Runs for a row attending to `keys` keys (`ink2_attn_splits`): about 128
/// keys a run, a multiple of four. A function of the key count alone, so a
/// position splits the same way in a decode step and in a verify pass.
#[must_use]
pub fn rel_attn_splits(keys: u32) -> u32 {
    keys.div_ceil(128)
        .next_multiple_of(4)
        .clamp(4, REL_ATTN_MAX_SPLITS)
}

/// Bytes of [`RelAttnViews::split_ml`] and [`RelAttnViews::split_out`].
#[must_use]
pub const fn rel_attn_split_bytes(n_head: u32, head_dim: u32) -> (u64, u64) {
    let runs = REL_ATTN_TILE_MIN_ROWS as u64 * n_head as u64 * REL_ATTN_MAX_SPLITS as u64;
    (runs * 2 * 4, runs * head_dim as u64 * 4)
}

/// Launches parallel top-k sigmoid router with correction bias.
pub fn route_topk(
    backend: &dyn Backend,
    args: RouteArgs,
    logits: DeviceView<'_>,
    bias: DeviceView<'_>,
    out_ids: DeviceView<'_>,
    out_weights: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_route_topk"),
        &LaunchArgs::new(
            Extent3::linear(args.rows),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(logits),
                KernelArg::Buffer(bias),
                KernelArg::Buffer(out_ids),
                KernelArg::Buffer(out_weights),
            ],
        ),
    )
}

/// Buffers for NVFP4 routed experts.
#[derive(Debug)]
pub struct Nvfp4Views<'a> {
    /// Packed E2M1 weights.
    pub weight: DeviceView<'a>,
    /// Per-16 E4M3 group scales.
    pub scale: DeviceView<'a>,
    /// Per-expert F32 amax.
    pub amax: DeviceView<'a>,
    /// Scalar F32 outer scale denominator.
    pub denom: DeviceView<'a>,
}

/// Launches NVFP4 MoE W13 + interleaved softcapped SwiGLU.
pub fn moe_nvfp4_w13_swiglu(
    backend: &dyn Backend,
    args: MoeW13Args,
    x: DeviceView<'_>,
    w: &Nvfp4Views<'_>,
    expert_ids: DeviceView<'_>,
    expert_weights: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_nvfp4_w13_swiglu"),
        &LaunchArgs::new(
            Extent3::new(args.inter_dim.div_ceil(8), args.rows * args.top_k, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(w.amax),
                KernelArg::Buffer(w.denom),
                KernelArg::Buffer(expert_ids),
                KernelArg::Buffer(expert_weights),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches NVFP4 MoE W2 + expert accumulation.
pub fn moe_nvfp4_w2_cols(
    backend: &dyn Backend,
    args: MoeW2Args,
    act: DeviceView<'_>,
    w: &Nvfp4Views<'_>,
    expert_ids: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_nvfp4_w2_cols"),
        &LaunchArgs::new(
            Extent3::new(args.out_dim.div_ceil(16), args.rows * args.top_k, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(act),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(w.amax),
                KernelArg::Buffer(w.denom),
                KernelArg::Buffer(expert_ids),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Buffers for MXFP8 routed experts.
#[derive(Debug)]
pub struct Mxfp8Views<'a> {
    /// E4M3 weights.
    pub weight: DeviceView<'a>,
    /// Per-32 UE8M0 scales.
    pub scale: DeviceView<'a>,
}

/// Launches MXFP8 MoE W13 + interleaved softcapped SwiGLU.
pub fn moe_mxfp8_w13_swiglu(
    backend: &dyn Backend,
    args: MoeW13Args,
    x: DeviceView<'_>,
    w: &Mxfp8Views<'_>,
    expert_ids: DeviceView<'_>,
    expert_weights: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_mxfp8_w13_swiglu"),
        &LaunchArgs::new(
            Extent3::new(args.inter_dim.div_ceil(8), args.rows * args.top_k, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(expert_ids),
                KernelArg::Buffer(expert_weights),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches MXFP8 MoE W2 + expert accumulation.
pub fn moe_mxfp8_w2_cols(
    backend: &dyn Backend,
    args: MoeW2Args,
    act: DeviceView<'_>,
    w: &Mxfp8Views<'_>,
    expert_ids: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_mxfp8_w2_cols"),
        &LaunchArgs::new(
            Extent3::new(args.out_dim.div_ceil(16), args.rows * args.top_k, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(act),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(expert_ids),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Arguments for `kernel_inkling2_moe_plan`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MoePlanArgs {
    /// Batch rows.
    pub rows: u32,
    /// Experts per token.
    pub top_k: u32,
    /// Total routed experts (at most 256).
    pub n_experts: u32,
    /// Most routed columns one work item carries.
    pub batch: u32,
}

impl MoePlanArgs {
    /// The most work items a pass of this shape can plan: every expert's
    /// partial batch, and the full ones the columns can fill. The batched
    /// entries' grid is sized for it.
    #[must_use]
    pub const fn max_items(&self) -> u32 {
        let total = self.rows * self.top_k;
        let bound = self.n_experts + total / self.batch;
        if total < bound { total } else { bound }
    }
}

/// Plans a multi-row pass's routed experts into work items of up to
/// `args.batch` columns sharing one expert (`kernel_inkling2_moe_plan`).
///
/// # Errors
///
/// Whatever the device reports.
pub fn moe_plan(
    backend: &dyn Backend,
    args: MoePlanArgs,
    expert_ids: DeviceView<'_>,
    cols: DeviceView<'_>,
    items: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_plan"),
        &LaunchArgs::new(
            Extent3::linear(1),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(expert_ids),
                KernelArg::Buffer(cols),
                KernelArg::Buffer(items),
            ],
        ),
    )
}

/// The plan the batched expert entries walk, as [`moe_plan`] wrote it.
#[derive(Debug)]
pub struct MoeGroupViews<'a> {
    /// Per-expert routed columns `[n_experts, rows]` (`u32`), ascending.
    pub cols: DeviceView<'a>,
    /// Work items `[1 + 2 * max_items]` (`u32`): the count, then
    /// `(expert, first | count << 24)` each.
    pub items: DeviceView<'a>,
    /// [`MoePlanArgs::max_items`] of the pass: the grid's height.
    pub max_items: u32,
}

/// Launches grouped NVFP4 MoE W13 + softcapped SwiGLU (8-row weight reuse per expert).
pub fn moe_nvfp4_w13_grouped(
    backend: &dyn Backend,
    args: MoeW13Args,
    x: DeviceView<'_>,
    w: &Nvfp4Views<'_>,
    expert_weights: DeviceView<'_>,
    grp: &MoeGroupViews<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_nvfp4_w13_grouped"),
        &LaunchArgs::new(
            Extent3::new(args.inter_dim.div_ceil(4), grp.max_items, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(w.amax),
                KernelArg::Buffer(w.denom),
                KernelArg::Buffer(expert_weights),
                KernelArg::Buffer(grp.cols),
                KernelArg::Buffer(grp.items),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches grouped NVFP4 MoE W2 (writes per-slot outputs for `moe_reduce_topk`).
pub fn moe_nvfp4_w2_grouped(
    backend: &dyn Backend,
    args: MoeW2Args,
    act: DeviceView<'_>,
    w: &Nvfp4Views<'_>,
    grp: &MoeGroupViews<'_>,
    slots_out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_nvfp4_w2_grouped"),
        &LaunchArgs::new(
            Extent3::new(args.out_dim.div_ceil(8), grp.max_items, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(act),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(w.amax),
                KernelArg::Buffer(w.denom),
                KernelArg::Buffer(grp.cols),
                KernelArg::Buffer(grp.items),
                KernelArg::Buffer(slots_out),
            ],
        ),
    )
}

/// Launches grouped MXFP8 MoE W13 + softcapped SwiGLU (8-row weight reuse per expert).
pub fn moe_mxfp8_w13_grouped(
    backend: &dyn Backend,
    args: MoeW13Args,
    x: DeviceView<'_>,
    w: &Mxfp8Views<'_>,
    expert_weights: DeviceView<'_>,
    grp: &MoeGroupViews<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_mxfp8_w13_grouped"),
        &LaunchArgs::new(
            Extent3::new(args.inter_dim.div_ceil(4), grp.max_items, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(expert_weights),
                KernelArg::Buffer(grp.cols),
                KernelArg::Buffer(grp.items),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches grouped MXFP8 MoE W2 (writes per-slot outputs for `moe_reduce_topk`).
pub fn moe_mxfp8_w2_grouped(
    backend: &dyn Backend,
    args: MoeW2Args,
    act: DeviceView<'_>,
    w: &Mxfp8Views<'_>,
    grp: &MoeGroupViews<'_>,
    slots_out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_mxfp8_w2_grouped"),
        &LaunchArgs::new(
            Extent3::new(args.out_dim.div_ceil(8), grp.max_items, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(act),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(grp.cols),
                KernelArg::Buffer(grp.items),
                KernelArg::Buffer(slots_out),
            ],
        ),
    )
}

/// Columns a tiled work item carries: the tile's width.
pub const MOE_TILE_BATCH: u32 = 64;

/// Launches `kernel_inkling2_moe_nvfp4_w13_tile`: [`moe_nvfp4_w13_grouped`] as a
/// simdgroup-matrix GEMM over work items of up to [`MOE_TILE_BATCH`] columns.
///
/// # Errors
///
/// Whatever the device reports.
pub fn moe_nvfp4_w13_tile(
    backend: &dyn Backend,
    args: MoeW13Args,
    x: DeviceView<'_>,
    w: &Nvfp4Views<'_>,
    expert_weights: DeviceView<'_>,
    grp: &MoeGroupViews<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_nvfp4_w13_tile"),
        &LaunchArgs::new(
            Extent3::new((2 * args.inter_dim).div_ceil(64), grp.max_items, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(w.amax),
                KernelArg::Buffer(w.denom),
                KernelArg::Buffer(expert_weights),
                KernelArg::Buffer(grp.cols),
                KernelArg::Buffer(grp.items),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches `kernel_inkling2_moe_nvfp4_w2_tile`: [`moe_nvfp4_w2_grouped`] as a
/// simdgroup-matrix GEMM over work items of up to [`MOE_TILE_BATCH`] columns.
///
/// # Errors
///
/// Whatever the device reports.
pub fn moe_nvfp4_w2_tile(
    backend: &dyn Backend,
    args: MoeW2Args,
    act: DeviceView<'_>,
    w: &Nvfp4Views<'_>,
    grp: &MoeGroupViews<'_>,
    slots_out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_nvfp4_w2_tile"),
        &LaunchArgs::new(
            Extent3::new((args.out_dim).div_ceil(64), grp.max_items, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(act),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(w.amax),
                KernelArg::Buffer(w.denom),
                KernelArg::Buffer(grp.cols),
                KernelArg::Buffer(grp.items),
                KernelArg::Buffer(slots_out),
            ],
        ),
    )
}

/// Launches `kernel_inkling2_moe_mxfp8_w13_tile`: [`moe_mxfp8_w13_grouped`] as a
/// simdgroup-matrix GEMM over work items of up to [`MOE_TILE_BATCH`] columns.
///
/// # Errors
///
/// Whatever the device reports.
pub fn moe_mxfp8_w13_tile(
    backend: &dyn Backend,
    args: MoeW13Args,
    x: DeviceView<'_>,
    w: &Mxfp8Views<'_>,
    expert_weights: DeviceView<'_>,
    grp: &MoeGroupViews<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_mxfp8_w13_tile"),
        &LaunchArgs::new(
            Extent3::new((2 * args.inter_dim).div_ceil(64), grp.max_items, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(expert_weights),
                KernelArg::Buffer(grp.cols),
                KernelArg::Buffer(grp.items),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches `kernel_inkling2_moe_mxfp8_w2_tile`: [`moe_mxfp8_w2_grouped`] as a
/// simdgroup-matrix GEMM over work items of up to [`MOE_TILE_BATCH`] columns.
///
/// # Errors
///
/// Whatever the device reports.
pub fn moe_mxfp8_w2_tile(
    backend: &dyn Backend,
    args: MoeW2Args,
    act: DeviceView<'_>,
    w: &Mxfp8Views<'_>,
    grp: &MoeGroupViews<'_>,
    slots_out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_mxfp8_w2_tile"),
        &LaunchArgs::new(
            Extent3::new((args.out_dim).div_ceil(64), grp.max_items, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(act),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(grp.cols),
                KernelArg::Buffer(grp.items),
                KernelArg::Buffer(slots_out),
            ],
        ),
    )
}

/// Sums per-slot grouped W2 outputs `[rows, top_k, out_dim]` into `[rows, out_dim]`.
pub fn moe_reduce_topk(
    backend: &dyn Backend,
    args: MoeW2Args,
    slots_in: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    let total = args.rows * args.out_dim;
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_reduce_topk"),
        &LaunchArgs::new(
            Extent3::linear(total.div_ceil(256)),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(slots_in),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches elementwise `out = a + b`.
pub fn add2_f32(
    backend: &dyn Backend,
    count: u32,
    a: DeviceView<'_>,
    b: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    let args = AddArgs { count };
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_add2_f32"),
        &LaunchArgs::new(
            Extent3::linear(count.div_ceil(256)),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(a),
                KernelArg::Buffer(b),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

/// Launches elementwise `out = a + b + c`.
pub fn add3_f32(
    backend: &dyn Backend,
    count: u32,
    a: DeviceView<'_>,
    b: DeviceView<'_>,
    c: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    let args = AddArgs { count };
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_add3_f32"),
        &LaunchArgs::new(
            Extent3::linear(count.div_ceil(256)),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(a),
                KernelArg::Buffer(b),
                KernelArg::Buffer(c),
                KernelArg::Buffer(out),
            ],
        ),
    )
}

//! Device kernel argument blocks and launch wrappers for the `inkling2` module.

#![expect(
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    reason = "thin kernel dispatch wrappers whose errors are always backend launch failures"
)]

use bytemuck::{Pod, Zeroable, bytes_of};
use shiso_core::{Backend, CoreResult, DeviceView, Extent3, KernelArg, KernelRef, LaunchArgs};

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
}

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
            Extent3::new(args.vocab.div_ceil(4), args.rows, 1),
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

/// Launches block attention residual mixing.
pub fn attn_res(
    backend: &dyn Backend,
    args: AttnResArgs,
    blocks: DeviceView<'_>,
    partial: DeviceView<'_>,
    addend: DeviceView<'_>,
    queries: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_attn_res"),
        &LaunchArgs::new(
            Extent3::linear(args.rows),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(blocks),
                KernelArg::Buffer(partial),
                KernelArg::Buffer(addend),
                KernelArg::Buffer(queries),
                KernelArg::Buffer(out),
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

/// Launches relative-bias + per-query sink + output-gated attention (zero-barrier).
pub fn rel_attn(backend: &dyn Backend, args: RelAttnArgs, v: &RelAttnViews<'_>) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_rel_attn"),
        &LaunchArgs::new(
            Extent3::new(args.n_head, args.rows.div_ceil(4), 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(v.q),
                KernelArg::Buffer(v.rel_bias),
                KernelArg::Buffer(v.sinks),
                KernelArg::Buffer(v.gate),
                KernelArg::Buffer(v.k_cache),
                KernelArg::Buffer(v.v_cache),
                KernelArg::Buffer(v.out),
            ],
        ),
    )
}

/// Arguments for split-K Flash-Decoding `rel_attn` (long-context decode).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RelAttnSplitKArgs {
    /// Query heads.
    pub n_head: u32,
    /// KV heads.
    pub n_head_kv: u32,
    /// Head width (128).
    pub head_dim: u32,
    /// Relative position extent.
    pub rel_extent: u32,
    /// Query position.
    pub pos: u32,
    /// Number of KV splits (`16`).
    pub n_splits: u32,
    /// Attention scale (`1 / sqrt(head_dim)`).
    pub scale: f32,
}

/// Launches split-K Flash-Decoding stage 1 + stage 2 for long-context decode.
pub fn rel_attn_splitk(
    backend: &dyn Backend,
    args: RelAttnSplitKArgs,
    v: &RelAttnViews<'_>,
    split_ml: DeviceView<'_>,
    split_out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_rel_attn_splitk"),
        &LaunchArgs::new(
            Extent3::new(args.n_head, args.n_splits.div_ceil(4), 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(v.q),
                KernelArg::Buffer(v.rel_bias),
                KernelArg::Buffer(v.sinks),
                KernelArg::Buffer(v.k_cache),
                KernelArg::Buffer(v.v_cache),
                KernelArg::Buffer(split_ml),
                KernelArg::Buffer(split_out),
            ],
        ),
    )?;
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_rel_attn_reduce"),
        &LaunchArgs::new(
            Extent3::linear(args.n_head),
            Extent3::linear(32),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(split_ml),
                KernelArg::Buffer(split_out),
                KernelArg::Buffer(v.gate),
                KernelArg::Buffer(v.out),
            ],
        ),
    )
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
            Extent3::new(args.inter_dim.div_ceil(4), args.rows * args.top_k, 1),
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
pub fn moe_nvfp4_w2_accum(
    backend: &dyn Backend,
    args: MoeW2Args,
    act: DeviceView<'_>,
    w: &Nvfp4Views<'_>,
    expert_ids: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_nvfp4_w2_accum"),
        &LaunchArgs::new(
            Extent3::new(args.out_dim.div_ceil(4), args.rows, 1),
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
            Extent3::new(args.inter_dim.div_ceil(4), args.rows * args.top_k, 1),
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
pub fn moe_mxfp8_w2_accum(
    backend: &dyn Backend,
    args: MoeW2Args,
    act: DeviceView<'_>,
    w: &Mxfp8Views<'_>,
    expert_ids: DeviceView<'_>,
    out: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_mxfp8_w2_accum"),
        &LaunchArgs::new(
            Extent3::new(args.out_dim.div_ceil(4), args.rows, 1),
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

/// Arguments for `kernel_inkling2_moe_bucket`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MoeBucketArgs {
    /// Batch rows.
    pub rows: u32,
    /// Experts per token.
    pub top_k: u32,
    /// Total routed experts.
    pub n_experts: u32,
}

/// Buckets routed token-expert slots by expert id on GPU for grouped prefill.
pub fn moe_bucket(
    backend: &dyn Backend,
    args: MoeBucketArgs,
    expert_ids: DeviceView<'_>,
    expert_counts: DeviceView<'_>,
    expert_cols: DeviceView<'_>,
) -> CoreResult<()> {
    backend.run(
        KernelRef::new(MODULE, "kernel_inkling2_moe_bucket"),
        &LaunchArgs::new(
            Extent3::linear(args.n_experts),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(expert_ids),
                KernelArg::Buffer(expert_counts),
                KernelArg::Buffer(expert_cols),
            ],
        ),
    )
}

/// Grouping buffers for grouped MoE prefill kernels.
#[derive(Debug)]
pub struct MoeGroupViews<'a> {
    /// Per-expert token count `[n_experts]` (`u32`).
    pub counts: DeviceView<'a>,
    /// Per-expert routed slot indices `[n_experts, rows]` (`u32`).
    pub cols: DeviceView<'a>,
    /// Total number of routed experts.
    pub n_experts: u32,
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
            Extent3::new(args.inter_dim.div_ceil(4), grp.n_experts, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(w.amax),
                KernelArg::Buffer(w.denom),
                KernelArg::Buffer(expert_weights),
                KernelArg::Buffer(grp.counts),
                KernelArg::Buffer(grp.cols),
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
            Extent3::new(args.out_dim.div_ceil(4), grp.n_experts, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(act),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(w.amax),
                KernelArg::Buffer(w.denom),
                KernelArg::Buffer(grp.counts),
                KernelArg::Buffer(grp.cols),
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
            Extent3::new(args.inter_dim.div_ceil(4), grp.n_experts, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(expert_weights),
                KernelArg::Buffer(grp.counts),
                KernelArg::Buffer(grp.cols),
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
            Extent3::new(args.out_dim.div_ceil(4), grp.n_experts, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(act),
                KernelArg::Buffer(w.weight),
                KernelArg::Buffer(w.scale),
                KernelArg::Buffer(grp.counts),
                KernelArg::Buffer(grp.cols),
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

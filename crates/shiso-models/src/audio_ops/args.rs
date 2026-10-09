//! Argument blocks of the `audio` kernel set, field for field as
//! `crates/shiso-metal/metal/audio/audio_common.h` declares them.
//!
//! Each block is passed as the launch's first argument, and its size is pinned
//! beside it: a field added on one side only would shift every field after it
//! and the kernel would read a plausible, wrong geometry.

use bytemuck::{Pod, Zeroable};

/// `audio_args_unary`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct UnaryArgs {
    pub(crate) count: u32,
    pub(crate) op: u32,
    pub(crate) a: f32,
    pub(crate) b: f32,
}
const _: () = assert!(size_of::<UnaryArgs>() == 16);

/// `audio_args_axpy`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct AxpyArgs {
    pub(crate) count: u32,
    pub(crate) alpha: f32,
}
const _: () = assert!(size_of::<AxpyArgs>() == 8);

/// `audio_args_bias_rows`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct BiasRowsArgs {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
}
const _: () = assert!(size_of::<BiasRowsArgs>() == 8);

/// `audio_args_argmax`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ArgmaxArgs {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
}
const _: () = assert!(size_of::<ArgmaxArgs>() == 8);

/// `audio_args_glu`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GluArgs {
    pub(crate) rows: u32,
    pub(crate) channels: u32,
}
const _: () = assert!(size_of::<GluArgs>() == 8);

/// `audio_args_silu_mul`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct SiluMulArgs {
    pub(crate) count: u32,
}
const _: () = assert!(size_of::<SiluMulArgs>() == 4);

/// `audio_args_gelu_mul`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GeluMulArgs {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) up_stride: u32,
    pub(crate) up_offset: u32,
}
const _: () = assert!(size_of::<GeluMulArgs>() == 16);

/// `audio_args_norm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct NormArgs {
    pub(crate) rows: u32,
    pub(crate) dim: u32,
    pub(crate) eps: f32,
    pub(crate) flags: u32,
}
const _: () = assert!(size_of::<NormArgs>() == 16);

/// `audio_args_gemm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GemmArgs {
    pub(crate) m: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
    pub(crate) flags: u32,
    /// k per split, a multiple of the k tile; all of k with one split.
    pub(crate) k_chunk: u32,
    /// Splits of k the grid's z axis runs; the partial planes' count.
    pub(crate) splits: u32,
}
const _: () = assert!(size_of::<GemmArgs>() == 24);

/// `audio_args_conv2d`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct Conv2dArgs {
    pub(crate) t_in: u32,
    pub(crate) f_in: u32,
    pub(crate) c_in: u32,
    pub(crate) t_out: u32,
    pub(crate) f_out: u32,
    pub(crate) c_out: u32,
    pub(crate) pad_t: u32,
    pub(crate) pad_f: u32,
    pub(crate) depthwise: u32,
    pub(crate) has_bias: u32,
    pub(crate) relu: u32,
    pub(crate) in_valid: u32,
    pub(crate) out_valid: u32,
    pub(crate) batch: u32,
}
const _: () = assert!(size_of::<Conv2dArgs>() == 56);

/// `audio_args_dwconv1d`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct DwConv1dArgs {
    pub(crate) rows: u32,
    pub(crate) channels: u32,
    pub(crate) taps: u32,
}
const _: () = assert!(size_of::<DwConv1dArgs>() == 12);

/// `audio_args_unfold`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct UnfoldArgs {
    pub(crate) rows: u32,
    pub(crate) channels: u32,
    pub(crate) taps: u32,
    pub(crate) pad: u32,
}
const _: () = assert!(size_of::<UnfoldArgs>() == 16);

/// `audio_args_transpose`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct TransposeArgs {
    pub(crate) batch: u32,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
}
const _: () = assert!(size_of::<TransposeArgs>() == 12);

/// `audio_args_window_attn`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct WindowAttnArgs {
    pub(crate) n_q: u32,
    pub(crate) n_k: u32,
    pub(crate) heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) q_abs0: u32,
    pub(crate) k_abs0: u32,
    pub(crate) k_end: u32,
    pub(crate) chunk: u32,
    pub(crate) left_chunks: u32,
    pub(crate) d_min: i32,
    pub(crate) n_d: u32,
    pub(crate) scale: f32,
    pub(crate) ac_scale: f32,
    pub(crate) softcap: f32,
    pub(crate) has_bias_uv: u32,
    pub(crate) has_q_dim_scale: u32,
}
const _: () = assert!(size_of::<WindowAttnArgs>() == 64);

/// `audio_args_rope`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct RopeArgs {
    pub(crate) rows: u32,
    pub(crate) heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) pos0: u32,
    pub(crate) theta: f32,
}
const _: () = assert!(size_of::<RopeArgs>() == 20);

/// `audio_args_lstm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct LstmArgs {
    pub(crate) hidden: u32,
}
const _: () = assert!(size_of::<LstmArgs>() == 4);

/// `audio_args_greedy_window`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GreedyWindowArgs {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) avail: u32,
}
const _: () = assert!(size_of::<GreedyWindowArgs>() == 12);

/// `audio_args_greedy_step`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GreedyStepArgs {
    pub(crate) rows: u32,
    pub(crate) avail: u32,
    pub(crate) blank: u32,
    pub(crate) max_symbols: u32,
    pub(crate) slot: u32,
}
const _: () = assert!(size_of::<GreedyStepArgs>() == 20);

/// `audio_args_embed_row`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct EmbedRowArgs {
    pub(crate) width: u32,
    pub(crate) vocab: u32,
}
const _: () = assert!(size_of::<EmbedRowArgs>() == 8);

/// `audio_args_keep`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct KeepArgs {
    pub(crate) count: u32,
}
const _: () = assert!(size_of::<KeepArgs>() == 4);

/// `audio_args_cache_attn`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct CacheAttnArgs {
    pub(crate) n_q: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) pos0: u32,
    pub(crate) cap: u32,
    pub(crate) scale: f32,
    pub(crate) splits: u32,
}
const _: () = assert!(size_of::<CacheAttnArgs>() == 32);

/// `audio_args_kv_append`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct KvAppendArgs {
    pub(crate) rows: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) cap: u32,
}
const _: () = assert!(size_of::<KvAppendArgs>() == 16);

/// `audio_args_decode_step`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct DecodeStepArgs {
    pub(crate) eos0: u32,
    pub(crate) eos1: u32,
    pub(crate) max_tokens: u32,
    pub(crate) advance: u32,
    pub(crate) slot: u32,
}
const _: () = assert!(size_of::<DecodeStepArgs>() == 20);

/// `audio_args_embed_token`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct EmbedTokenArgs {
    pub(crate) width: u32,
    pub(crate) vocab: u32,
    pub(crate) group: u32,
}
const _: () = assert!(size_of::<EmbedTokenArgs>() == 12);

/// `audio_args_matvec3`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct Matvec3Args {
    pub(crate) n0: u32,
    pub(crate) n1: u32,
    pub(crate) n2: u32,
    pub(crate) k: u32,
}
const _: () = assert!(size_of::<Matvec3Args>() == 16);

/// `audio_args_qk_rope`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct QkRopeArgs {
    pub(crate) rows: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) cap: u32,
    pub(crate) eps: f32,
    pub(crate) theta: f32,
}
const _: () = assert!(size_of::<QkRopeArgs>() == 28);

/// `audio_args_argmax_split`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ArgmaxSplitArgs {
    pub(crate) cols: u32,
    pub(crate) span: u32,
    pub(crate) parts: u32,
}
const _: () = assert!(size_of::<ArgmaxSplitArgs>() == 12);

/// `audio_args_batch_attn`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct BatchAttnArgs {
    pub(crate) seqs: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) scale: f32,
    pub(crate) splits: u32,
}
const _: () = assert!(size_of::<BatchAttnArgs>() == 24);

/// `audio_args_qk_rope_seq`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct QkRopeSeqArgs {
    pub(crate) rows: u32,
    pub(crate) seqs: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) eps: f32,
    pub(crate) theta: f32,
}
const _: () = assert!(size_of::<QkRopeSeqArgs>() == 28);

/// `audio_args_decode_step_batch`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct DecodeStepBatchArgs {
    pub(crate) eos0: u32,
    pub(crate) eos1: u32,
    pub(crate) max_tokens: u32,
    pub(crate) advance: u32,
    pub(crate) slot: u32,
    pub(crate) seqs: u32,
}
const _: () = assert!(size_of::<DecodeStepBatchArgs>() == 24);

/// `audio_args_span_attn`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct SpanAttnArgs {
    pub(crate) n: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) left: u32,
    pub(crate) right: u32,
    pub(crate) scale: f32,
    pub(crate) row_groups: u32,
    pub(crate) segmented: u32,
}
const _: () = assert!(size_of::<SpanAttnArgs>() == 36);

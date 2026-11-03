//! `#[repr(C)]` mirrors of the kernels' argument blocks.
//!
//! The engine and model crates own the originals; this crate may not depend on
//! them, so each block is copied field for field and its size pinned. The size
//! pins are the only automatic guard, and a same-size reorder or retype passes
//! them: any edit to `crates/shiso-engine/src/launch.rs` or
//! `crates/shiso-models/src/audio_ops/args.rs` must be mirrored here by hand.

use bytemuck::{Pod, Zeroable};

/// Mirror of `GetRowsArgs` in `crates/shiso-engine/src/launch.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GetRowsArgs {
    /// Elements per vocabulary row.
    pub(crate) n_embd: i32,
    /// Rows an id may name.
    pub(crate) n_vocab: i32,
    /// Tokens covered.
    pub(crate) n_tokens: i32,
    /// Layout padding.
    pub(crate) pad0: i32,
    /// Bytes per source row.
    pub(crate) src_row_bytes: u64,
    /// Bytes per output row.
    pub(crate) dst_row_bytes: u64,
    /// Bytes between ids.
    pub(crate) token_stride: u64,
}
const _: () = assert!(size_of::<GetRowsArgs>() == 40);

/// Mirror of `NormArgs` in `crates/shiso-engine/src/launch.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct NormArgs {
    /// Elements per activation row.
    pub(crate) ne00: i32,
    /// Elements per weight row.
    pub(crate) ne00_t: i32,
    /// Bytes between activation rows.
    pub(crate) nb1: u64,
    /// Activation plane stride.
    pub(crate) nb2: u64,
    /// Activation third-axis stride.
    pub(crate) nb3: u64,
    /// Epsilon.
    pub(crate) eps: f32,
    /// Weight extent, first axis.
    pub(crate) nef1: [i32; 3],
    /// Weight extent, second axis.
    pub(crate) nef2: [i32; 3],
    /// Weight extent, third axis.
    pub(crate) nef3: [i32; 3],
    /// Weight strides, first axis.
    pub(crate) nbf1: [u64; 3],
    /// Weight strides, second axis.
    pub(crate) nbf2: [u64; 3],
    /// Weight strides, third axis.
    pub(crate) nbf3: [u64; 3],
}
const _: () = assert!(size_of::<NormArgs>() == 144);

/// Mirror of `GluArgs` in `crates/shiso-engine/src/launch.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GluArgs {
    /// Elements the activation covers.
    pub(crate) ne00: i32,
    /// Layout padding.
    pub(crate) pad0: [u8; 4],
    /// Bytes between gate rows.
    pub(crate) nb01: u64,
    /// Up row width.
    pub(crate) ne10: i32,
    /// Layout padding.
    pub(crate) pad1: [u8; 4],
    /// Bytes between up rows.
    pub(crate) nb11: u64,
    /// Elements the output covers.
    pub(crate) ne0: i32,
    /// Layout padding.
    pub(crate) pad2: [u8; 4],
    /// Bytes between output rows.
    pub(crate) nb1: u64,
    /// Gate element offset.
    pub(crate) i00: i32,
    /// Up element offset.
    pub(crate) i10: i32,
    /// Product multiplier.
    pub(crate) alpha: f32,
    /// Symmetric clamp, zero for none.
    pub(crate) limit: f32,
}
const _: () = assert!(size_of::<GluArgs>() == 64);

/// Mirror of `MulMvArgs` in `crates/shiso-engine/src/launch.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct MulMvArgs {
    /// Elements per weight row.
    pub(crate) ne00: i32,
    /// Weight rows.
    pub(crate) ne01: i32,
    /// Weight planes.
    pub(crate) ne02: i32,
    /// Layout padding.
    pub(crate) pad0: [u8; 4],
    /// Bytes per weight element.
    pub(crate) nb00: u64,
    /// Bytes per weight row.
    pub(crate) nb01: u64,
    /// Weight plane stride.
    pub(crate) nb02: u64,
    /// Weight third-axis stride.
    pub(crate) nb03: u64,
    /// Elements per activation row.
    pub(crate) ne10: i32,
    /// Activation rows.
    pub(crate) ne11: i32,
    /// Activation planes.
    pub(crate) ne12: i32,
    /// Layout padding.
    pub(crate) pad1: [u8; 4],
    /// Bytes per activation element.
    pub(crate) nb10: u64,
    /// Bytes per activation row.
    pub(crate) nb11: u64,
    /// Activation plane stride.
    pub(crate) nb12: u64,
    /// Activation third-axis stride.
    pub(crate) nb13: u64,
    /// Elements per output row.
    pub(crate) ne0: i32,
    /// Output rows.
    pub(crate) ne1: i32,
    /// Output rows per threadgroup.
    pub(crate) nr0: i32,
    /// Broadcast ratio, second axis.
    pub(crate) r2: i16,
    /// Broadcast ratio, third axis.
    pub(crate) r3: i16,
}
const _: () = assert!(size_of::<MulMvArgs>() == 112);

/// Mirror of `MulMvExtArgs` in `crates/shiso-engine/src/launch.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct MulMvExtArgs {
    /// Elements per weight row.
    pub(crate) ne00: i32,
    /// Weight rows.
    pub(crate) ne01: i32,
    /// Weight planes.
    pub(crate) ne02: i32,
    /// Layout padding.
    pub(crate) pad0: i32,
    /// Bytes per weight element.
    pub(crate) nb00: u64,
    /// Bytes per weight row.
    pub(crate) nb01: u64,
    /// Weight plane stride.
    pub(crate) nb02: u64,
    /// Weight third-axis stride.
    pub(crate) nb03: u64,
    /// Elements per activation row.
    pub(crate) ne10: i32,
    /// Activation rows.
    pub(crate) ne11: i32,
    /// Activation planes.
    pub(crate) ne12: i32,
    /// Layout padding.
    pub(crate) pad1: i32,
    /// Bytes per activation element.
    pub(crate) nb10: u64,
    /// Bytes per activation row.
    pub(crate) nb11: u64,
    /// Activation plane stride.
    pub(crate) nb12: u64,
    /// Activation third-axis stride.
    pub(crate) nb13: u64,
    /// Elements per output row.
    pub(crate) ne0: i32,
    /// Output rows.
    pub(crate) ne1: i32,
    /// Broadcast ratio, second axis.
    pub(crate) r2: i16,
    /// Broadcast ratio, third axis.
    pub(crate) r3: i16,
    /// Trailing padding.
    pub(crate) pad2: [u8; 4],
}
const _: () = assert!(size_of::<MulMvExtArgs>() == 112);

/// Mirror of `MulMmArgs` in `crates/shiso-engine/src/launch.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct MulMmArgs {
    /// Elements per weight row.
    pub(crate) ne00: i32,
    /// Weight planes.
    pub(crate) ne02: i32,
    /// Bytes per weight row.
    pub(crate) nb01: u64,
    /// Weight plane stride.
    pub(crate) nb02: u64,
    /// Weight third-axis stride.
    pub(crate) nb03: u64,
    /// Activation planes.
    pub(crate) ne12: i32,
    /// Layout padding.
    pub(crate) pad0: i32,
    /// Bytes per activation element.
    pub(crate) nb10: u64,
    /// Bytes per activation row.
    pub(crate) nb11: u64,
    /// Activation plane stride.
    pub(crate) nb12: u64,
    /// Activation third-axis stride.
    pub(crate) nb13: u64,
    /// Elements per output row.
    pub(crate) ne0: i32,
    /// Output rows.
    pub(crate) ne1: i32,
    /// Broadcast ratio, second axis.
    pub(crate) r2: i16,
    /// Broadcast ratio, third axis.
    pub(crate) r3: i16,
    /// Trailing padding.
    pub(crate) pad2: [i8; 4],
}
const _: () = assert!(size_of::<MulMmArgs>() == 88);

/// Mirror of `DenseQNormRopeArgs` in `crates/shiso-engine/src/launch.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct DenseQNormRopeArgs {
    /// Head width.
    pub(crate) head_dim: u32,
    /// Query heads.
    pub(crate) n_head: u32,
    /// Rotated dims.
    pub(crate) n_rot: u32,
    /// Dims the frequency ladder spans.
    pub(crate) freq_dims: u32,
    /// Tokens.
    pub(crate) n_tok: u32,
    /// Rope base.
    pub(crate) rope_base: f32,
    /// Norm epsilon.
    pub(crate) eps: f32,
}
const _: () = assert!(size_of::<DenseQNormRopeArgs>() == 28);

/// Mirror of `DenseKvNormRopeArgs` in `crates/shiso-engine/src/launch.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct DenseKvNormRopeArgs {
    /// Head width.
    pub(crate) head_dim: u32,
    /// KV heads.
    pub(crate) n_head_kv: u32,
    /// Rotated dims.
    pub(crate) n_rot: u32,
    /// Dims the frequency ladder spans.
    pub(crate) freq_dims: u32,
    /// Tokens.
    pub(crate) n_tok: u32,
    /// Rope base.
    pub(crate) rope_base: f32,
    /// Norm epsilon.
    pub(crate) eps: f32,
    /// Tokens per page.
    pub(crate) page_tokens: u32,
    /// Whether V is normalised.
    pub(crate) v_norm: u32,
    /// Elements between pages.
    pub(crate) page_stride: u32,
}
const _: () = assert!(size_of::<DenseKvNormRopeArgs>() == 40);

/// Mirror of `DenseAttnArgs` in `crates/shiso-engine/src/launch.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct DenseAttnArgs {
    /// Head width.
    pub(crate) head_dim: u32,
    /// Query heads.
    pub(crate) n_head: u32,
    /// KV heads.
    pub(crate) n_head_kv: u32,
    /// Tokens.
    pub(crate) n_tok: u32,
    /// Tokens per page.
    pub(crate) page_tokens: u32,
    /// Sliding window.
    pub(crate) window: u32,
    /// Score scale.
    pub(crate) scale: f32,
    /// Elements between pages.
    pub(crate) page_stride: u32,
}
const _: () = assert!(size_of::<DenseAttnArgs>() == 32);

/// Mirror of `GemmArgs` in `crates/shiso-models/src/audio_ops/args.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct AudioGemmArgs {
    /// Rows.
    pub(crate) m: u32,
    /// Columns.
    pub(crate) n: u32,
    /// Reduction length.
    pub(crate) k: u32,
    /// Flags.
    pub(crate) flags: u32,
    /// k per split.
    pub(crate) k_chunk: u32,
    /// Splits of k.
    pub(crate) splits: u32,
}
const _: () = assert!(size_of::<AudioGemmArgs>() == 24);

/// Mirror of `NormArgs` in `crates/shiso-models/src/audio_ops/args.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct AudioNormArgs {
    /// Rows.
    pub(crate) rows: u32,
    /// Row width.
    pub(crate) dim: u32,
    /// Epsilon.
    pub(crate) eps: f32,
    /// Flags.
    pub(crate) flags: u32,
}
const _: () = assert!(size_of::<AudioNormArgs>() == 16);

/// Mirror of `WindowAttnArgs` in `crates/shiso-models/src/audio_ops/args.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct WindowAttnArgs {
    /// Query rows.
    pub(crate) n_q: u32,
    /// Key rows.
    pub(crate) n_k: u32,
    /// Heads.
    pub(crate) heads: u32,
    /// Head width.
    pub(crate) head_dim: u32,
    /// Absolute index of the first query.
    pub(crate) q_abs0: u32,
    /// Absolute index of the first key.
    pub(crate) k_abs0: u32,
    /// Absolute end of the keys.
    pub(crate) k_end: u32,
    /// Chunk size.
    pub(crate) chunk: u32,
    /// Left chunks visible.
    pub(crate) left_chunks: u32,
    /// Smallest relative distance.
    pub(crate) d_min: i32,
    /// Relative distances.
    pub(crate) n_d: u32,
    /// Score scale.
    pub(crate) scale: f32,
    /// Content-score scale.
    pub(crate) ac_scale: f32,
    /// Score soft cap.
    pub(crate) softcap: f32,
    /// Whether `bias_u`/`bias_v` is present.
    pub(crate) has_bias_uv: u32,
    /// Whether per-dim query scale is present.
    pub(crate) has_q_dim_scale: u32,
}
const _: () = assert!(size_of::<WindowAttnArgs>() == 64);

/// Mirror of `UnaryArgs` in `crates/shiso-models/src/audio_ops/args.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct UnaryArgs {
    /// Elements.
    pub(crate) count: u32,
    /// Operation.
    pub(crate) op: u32,
    /// First constant.
    pub(crate) a: f32,
    /// Second constant.
    pub(crate) b: f32,
}
const _: () = assert!(size_of::<UnaryArgs>() == 16);

/// Mirror of `AxpyArgs` in `crates/shiso-models/src/audio_ops/args.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct AxpyArgs {
    /// Elements.
    pub(crate) count: u32,
    /// Multiplier.
    pub(crate) alpha: f32,
}
const _: () = assert!(size_of::<AxpyArgs>() == 8);

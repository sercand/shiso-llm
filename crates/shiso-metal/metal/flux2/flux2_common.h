// Shared argument blocks and helpers for the flux2-klein kernel set.
//
// This file is concatenated ahead of every other flux2 entry, so the kernels
// may not redeclare a block or a helper. The field order of every struct is
// the binding contract the driver mirrors
// (`crates/shiso-models/src/flux2/ops.rs` marshals `#[repr(C)]` Pods with
// `bytemuck::bytes_of`); a drift is a wrong number under load, so the sizes
// are asserted here as well as on the host.
//
// The arithmetic of every kernel is the driver's host twin, entry for entry:
// same index math, same section offsets, same accumulation order where the
// numbers depend on it. The host twins are the fallback for a device without
// this set, and the two must answer the same image.

// Square tile of the GEMM kernels in every axis: 16x16 threads is one
// threadgroup of 256 and two 1 KiB staging tiles, the shape the driver
// launches (`GEMM_TILE` in ops.rs).
#define FLUX2_GEMM_TILE 16u

// Threads per elementwise threadgroup, as the driver launches it
// (`BLOCK` in ops.rs). Every reduction here is sized for exactly this.
#define FLUX2_BLOCK 64u

// Partial slots per lane in the attention kernels: one per 64 head-dim
// values a lane strides over, so this bounds head_dim at 512 - far past the
// family's 128-wide heads, and a head wider than the bound would silently
// drop its tail, so the kernels refuse it instead.
#define FLUX2_ATTN_SLOTS 8u

/// One brain-float, widened.
///
/// The 16 bits are loaded as an integer and shifted into the high half of an
/// f32 rather than being cast through a native type: exact for every finite
/// value, and the same widening the driver's host twin does on the fallback
/// path, so the two read identical numbers.
static inline float flux2_bf16(device const ushort *p) {
    return as_type<float>((uint)(*p) << 16);
}

/// One E4M3 byte as a float: 1 sign, 4 exponent (bias 7), 3 mantissa bits,
/// no infinities, the all-ones pattern NaN. The exact inverse of the
/// checkpoint's quantizer, and the same decode the driver's host twin runs.
static inline float flux2_e4m3(uint b) {
    const uint e = (b >> 3u) & 0xFu;
    const uint m = b & 7u;
    float v;
    if (e == 0u) {
        v = float(m) * 0.001953125f;  // m * 2^-9
    } else if (e == 15u && m == 7u) {
        v = NAN;
    } else {
        v = (1.0f + float(m) * 0.125f) * exp2(float(e) - 7.0f);
    }
    return (b & 0x80u) ? -v : v;
}

/// SiLU, in the host twin's own form (`x / (1 + exp(-x))`).
static inline float flux2_silu(float x) {
    return x / (1.0f + exp(-x));
}

/// Inclusive sum over one simdgroup, left-folded in decreasing-offset order
/// so lane 0 ends up holding the total. Spelled out instead of using the
/// builtin because the shading language leaves that summation order
/// unspecified, and the host twin's fold is the answer being matched.
static inline float flux2_warp_sum(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

/// Block-wide reduction over the driver's 64-thread block (two simdgroups).
/// Shape: warp reduce, lane 0 publishes to scratch[warp], barrier, warp 0
/// re-sums, thread 0 broadcasts through the last slot. `scratch` needs three
/// floats and is passed in because an inlined helper has no threadgroup array
/// of its own to keep.
static inline float flux2_block_sum(float v, threadgroup float *scratch, uint tid) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = flux2_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = FLUX2_BLOCK / 32u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = flux2_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[n_warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[n_warp];
}

/// Whether a bias binding is the driver's absent-bias stand-in.
///
/// The driver binds a zero-filled 4-byte buffer wherever a linear or a
/// convolution carries no bias (`Ops::dummy`), and the argument block has no
/// room for a presence flag, so an exactly +0.0 first word is the absence
/// signal - the same reported convention the q4e hyper-connection port uses.
/// A real bias whose first element is exactly +0.0 would be misread as
/// absent; the host twin has no such ambiguity, and the deviation is
/// reported with this port.
static inline bool flux2_bias_absent(device const float *bias) {
    return as_type<uint>(bias[0]) == 0u;
}

// --- argument blocks --------------------------------------------------------
//
// One block per kernel family, passed as `constant T & args [[buffer(0)]]`.
// Field order and width are the driver's `#[repr(C)]` Pods, byte for byte.

/// A linear in its stored width: `out[m,n] = in[m,k] . weight[n,k] * scale + bias[n]`.
struct flux2_args_gemm {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

/// One elementwise pass over `count` values.
struct flux2_args_unary {
    uint32_t count;
};

/// SiLU gate over a fused row: the gate is the FIRST half
/// (`out[r,i] = silu(in[r,i]) * in[r,inner+i]`), read from a strided buffer
/// whose view may start mid-row, and written packed.
struct flux2_args_swiglu {
    uint32_t rows;
    uint32_t inner;
    uint32_t stride;
};

/// AdaLN modulation: `out = x * (1 + scale) + shift`, the per-dimension
/// vectors of `period` elements broadcast over the rows.
struct flux2_args_modulate {
    uint32_t count;
    uint32_t period;
};

/// Gated residual add: `out = a + gate * b`, `gate` broadcast over `period`.
struct flux2_args_add_gated {
    uint32_t count;
    uint32_t period;
};

/// Euler latent update: `out = x + dt * v`.
struct flux2_args_axpy {
    uint32_t count;
    float dt;
};

/// Row normalization (`rmsnorm`, `layer_norm`) and, with `eps` unused, the
/// (rows, dim) transpose.
struct flux2_args_norm {
    uint32_t rows;
    uint32_t dim;
    float eps;
};

/// Per-head RMS normalization with a per-dimension scale over the q and k
/// head sections of fused qkv rows, in place.
struct flux2_args_qk_norm {
    uint32_t tokens;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t row_stride;
    float eps;
};

/// Row-wise concatenation: `out[r] = [a[r], b[r]]`.
struct flux2_args_concat {
    uint32_t rows;
    uint32_t cols_a;
    uint32_t cols_b;
};

/// Joint-sequence attention over fused or packed rows. Head `h` of a row
/// starts at `row * stride + offset + h * head_dim` for each operand's own
/// offset; the cached rows carry the same layout against `cache_stride`.
struct flux2_args_sdpa {
    uint32_t txt_tokens;
    uint32_t img_tokens;
    uint32_t cache_tokens;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t txt_stride;
    uint32_t img_stride;
    uint32_t cache_stride;
    uint32_t q_offset;
    uint32_t k_offset;
    uint32_t v_offset;
};

/// Causal grouped-query attention over packed rows (text encoder prefill).
struct flux2_args_gqa {
    uint32_t tokens;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
};

/// 4-axis interleaved RoPE on the q and k head sections of strided rows, in
/// place, one launch for both.
struct flux2_args_rope_4d {
    uint32_t tokens;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t freqs_per_axis;
    uint32_t row_stride;
    uint32_t q_offset;
    uint32_t k_offset;
};

/// Rotate-half 1-axis RoPE on one packed (tokens, heads, head_dim) tensor.
struct flux2_args_rope_half {
    uint32_t tokens;
    uint32_t heads;
    uint32_t head_dim;
};

/// One NCHW f32 convolution.
struct flux2_args_conv2d {
    uint32_t batch;
    uint32_t in_c;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t out_c;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t k_h;
    uint32_t k_w;
    uint32_t stride;
    uint32_t pad;
};

/// Group normalization; the affine buffers hold `num_groups *
/// channels_per_group` elements, indexed by channel.
struct flux2_args_group_norm {
    uint32_t batch;
    uint32_t num_groups;
    uint32_t channels_per_group;
    uint32_t spatial;
    float eps;
};

/// Nearest-neighbour 2x upsampling of an NCHW field.
struct flux2_args_upsample {
    uint32_t batch;
    uint32_t channels;
    uint32_t h;
    uint32_t w;
};

/// Gather of a BF16 embedding table, widened to f32 on write.
struct flux2_args_embed {
    uint32_t tokens;
    uint32_t hidden;
};

// The blocks and the driver's Pods are edited together; the host asserts the
// same widths, and so does this side so a header-only drift fails the compile.
static_assert(sizeof(flux2_args_gemm) == 12, "GemmArgs");
static_assert(sizeof(flux2_args_unary) == 4, "UnaryArgs");
static_assert(sizeof(flux2_args_swiglu) == 12, "SwigluArgs");
static_assert(sizeof(flux2_args_modulate) == 8, "ModulateArgs");
static_assert(sizeof(flux2_args_add_gated) == 8, "AddGatedArgs");
static_assert(sizeof(flux2_args_axpy) == 8, "AxpyArgs");
static_assert(sizeof(flux2_args_norm) == 12, "NormArgs");
static_assert(sizeof(flux2_args_qk_norm) == 20, "QkNormArgs");
static_assert(sizeof(flux2_args_concat) == 12, "ConcatArgs");
static_assert(sizeof(flux2_args_sdpa) == 44, "SdpaArgs");
static_assert(sizeof(flux2_args_gqa) == 16, "GqaArgs");
static_assert(sizeof(flux2_args_rope_4d) == 28, "Rope4dArgs");
static_assert(sizeof(flux2_args_rope_half) == 12, "RopeHalfArgs");
static_assert(sizeof(flux2_args_conv2d) == 44, "Conv2dArgs");
static_assert(sizeof(flux2_args_group_norm) == 20, "GroupNormArgs");
static_assert(sizeof(flux2_args_upsample) == 16, "UpsampleArgs");
static_assert(sizeof(flux2_args_embed) == 8, "EmbedArgs");

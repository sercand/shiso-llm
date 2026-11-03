// Shared argument blocks and helpers for the diffusion DiT kernel set.
//
// This file is concatenated ahead of every other dit entry, so the kernels
// may not redeclare a block or a helper. The field order of every struct is
// the binding contract the driver mirrors
// (`crates/shiso-models/src/diffusion/ops.rs` marshals `#[repr(C)]` Pods with
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
#define DIT_GEMM_TILE 16u

// The simdgroup-matrix GEMM tile (`kernel_dit_gemm64_*` and the implicit-
// GEMM convolution): a 64x64 output tile per threadgroup of 128 threads, four
// simdgroups each owning a 32x32 quarter, k staged 16 at a time. The driver
// sends only products whose reduction length is a multiple of the k slice
// here and everything else to the 16x16 kernels.
#define DIT_G64_TILE 64u
#define DIT_G64_K 16u
#define DIT_G64_THREADS 128u

// The fused attention tile (`kernel_dit_flash_sdpa_f32`): 64 query rows per
// threadgroup of eight simdgroups (DIT_FA_RB blocks of eight rows each),
// keys streamed 32 at a time, heads up to DIT_FA_MAX_D wide in multiples of
// eight. The driver sends wider or ragged heads to `kernel_dit_sdpa_f32`.
//
// Measured on an M4 (10-core GPU) at a 1024x1024 Qwen-Image-2.1 step (4096
// rows over 22 cached, 32 heads of 128), with the head width fixed at 128:
// 98 ms per layer at 8 simdgroups x 8 rows, 118 ms at 4, 106 ms at 16;
// two row blocks per simdgroup (16 rows) ran 112-117 ms and 16-key blocks
// were no better. The previous 32-row, runtime-width tile took 205 ms.
#define DIT_FA_RB 1u
#define DIT_FA_SG 8u
#define DIT_FA_QROWS (8u * DIT_FA_RB * DIT_FA_SG)
#define DIT_FA_KEYS 32u
#define DIT_FA_MAX_D 128u
#define DIT_FA_THREADS (32u * DIT_FA_SG)

#define DIT_UNROLL _Pragma("clang loop unroll(full)")

// Threads per elementwise threadgroup, as the driver launches it
// (`BLOCK` in ops.rs). Every reduction here is sized for exactly this.
#define DIT_BLOCK 64u

// Partial slots per lane in the attention kernels: one per 64 head-dim
// values a lane strides over, so this bounds head_dim at 512. The family's
// heads are 128 wide, a quarter of the bound, and nothing here guards it:
// a head wider than 512 would silently drop its tail, so the driver's
// model config is what keeps head_dim inside the bound.
#define DIT_ATTN_SLOTS 8u

/// One brain-float, widened.
///
/// The 16 bits are loaded as an integer and shifted into the high half of an
/// f32 rather than being cast through a native type: exact for every finite
/// value, and the same widening the driver's host twin does on the fallback
/// path, so the two read identical numbers.
static inline float dit_bf16(device const ushort *p) {
    return as_type<float>((uint)(*p) << 16);
}

/// One E4M3 byte as a float: 1 sign, 4 exponent (bias 7), 3 mantissa bits,
/// no infinities, the all-ones pattern NaN. The exact inverse of the
/// checkpoint's quantizer, and the same decode the driver's host twin runs.
static inline float dit_e4m3(uint b) {
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

/// [`dit_e4m3`] without branches, for the GEMM tile's staging loads:
/// normal values are rebuilt as f32 bits directly (exponent rebias 7 -> 127,
/// mantissa shifted up), subnormals as `m * 2^-9`, the all-ones pattern NaN.
/// Every byte decodes to the same float the branching form returns.
static inline float dit_e4m3_bits(uint b) {
    const uint e = (b >> 3u) & 0xFu;
    const uint m = b & 7u;
    float v = (e == 0u) ? float(m) * 0.001953125f
                        : as_type<float>(((e + 120u) << 23) | (m << 20));
    v = (e == 15u && m == 7u) ? NAN : v;
    return (b & 0x80u) ? -v : v;
}

/// SiLU, in the host twin's own form (`x / (1 + exp(-x))`).
static inline float dit_silu(float x) {
    return x / (1.0f + exp(-x));
}

/// Inclusive sum over one simdgroup, left-folded in decreasing-offset order
/// so lane 0 ends up holding the total. Spelled out instead of using the
/// builtin because the shading language leaves that summation order
/// unspecified: the point is a fixed association, so the kernel answers the
/// same on every run and device. The butterfly is not the host twin's
/// sequential fold - a lane-parallel reduction cannot reproduce that
/// association - so the two are deterministic, not bit-identical, and the
/// rounding difference stays far inside the 1e-4 parity tolerance.
static inline float dit_warp_sum(float v) {
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
static inline float dit_block_sum(float v, threadgroup float *scratch, uint tid) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = dit_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = DIT_BLOCK / 32u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = dit_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[n_warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[n_warp];
}

/// [`dit_block_sum`]'s shape with `max` in place of `+`.
static inline float dit_block_max(float v, threadgroup float *scratch, uint tid) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = simd_max(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = DIT_BLOCK / 32u;
    v = (tid < n_warp) ? scratch[tid] : -INFINITY;
    if (warp == 0u) {
        v = simd_max(v);
    }
    if (tid == 0u) {
        scratch[n_warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[n_warp];
}

// --- argument blocks --------------------------------------------------------
//
// One block per kernel family, passed as `constant T & args [[buffer(0)]]`.
// Field order and width are the driver's `#[repr(C)]` Pods, byte for byte.

/// A linear in its stored width: `out[m,n] = in[m,k] . weight[n,k] * scale + bias[n]`.
///
/// `has_bias` is the bias presence contract, shared with conv2d: the
/// driver binds a dedicated 4-byte zero stand-in (`Ops::dummy`) wherever a
/// linear or convolution carries no bias, and MSL has no device-side query
/// for a bound buffer's length, so the flag - never the bias values -
/// decides whether the binding is read. `has_bias == 0` means the kernel
/// must not read `bias` at all; `has_bias == 1` means it is a real vector.
struct dit_args_gemm {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t has_bias;
};

/// A linear fused with what surrounds it in the transformer blocks:
/// `out[m,n] += gate[n] * (x[m,:] . weight[n,:] * scale + bias[n])`, with
/// the input row `x = [a (cols_a) | silu(g) * u (inner)]` formed while
/// staging - `g` at column `gu_offset` of a `gu_stride`-wide row of `gu`,
/// `u` `inner` columns after it. Either part may be empty; `cols_a` and
/// `inner` are multiples of four and their sum a multiple of 16.
struct dit_args_gemm_fused {
    uint32_t m;
    uint32_t n;
    uint32_t cols_a;
    uint32_t inner;
    uint32_t gu_stride;
    uint32_t gu_offset;
    uint32_t has_bias;
};

/// The rotation every ConvRot linear's input takes before its INT8 GEMM:
/// `out[r, g*256 .. g*256+256] = x[r, g*256 .. +256] . H` for each 256-wide
/// group `g`, `H = kron^4(H4) / 16`. The input row is read the way the fused
/// linear reads it (`[a (cols_a) | silu(g) * u (inner)]`, `inner == 0` for a
/// plain row), so a block's output linear rotates its swiglu input without a
/// separate pass; the output is packed, `cols_a + inner` wide, and never
/// aliases the input.
struct dit_args_hadamard {
    uint32_t rows;
    uint32_t cols_a;
    uint32_t inner;
    uint32_t gu_stride;
    uint32_t gu_offset;
};

/// One elementwise pass over `count` values.
struct dit_args_unary {
    uint32_t count;
};

/// SiLU gate over a fused row: the gate is the FIRST half
/// (`out[r,i] = silu(in[r,i]) * in[r,inner+i]`), read from a strided buffer
/// whose view may start mid-row, and written packed.
struct dit_args_swiglu {
    uint32_t rows;
    uint32_t inner;
    uint32_t stride;
};

/// AdaLN modulation: `out = x * (1 + scale) + shift`, the per-dimension
/// vectors of `period` elements broadcast over the rows.
struct dit_args_modulate {
    uint32_t count;
    uint32_t period;
};

/// Gated residual add: `out = a + gate * b`, `gate` broadcast over `period`.
struct dit_args_add_gated {
    uint32_t count;
    uint32_t period;
};

/// Euler latent update: `out = x + dt * v`.
struct dit_args_axpy {
    uint32_t count;
    float dt;
};

/// Row normalization (`rmsnorm`, `layer_norm`) and, with `eps` unused, the
/// (rows, dim) transpose.
struct dit_args_norm {
    uint32_t rows;
    uint32_t dim;
    float eps;
};

/// Row softmax of `scale * x`, in place, over `rows` rows of `cols`.
struct dit_args_softmax {
    uint32_t rows;
    uint32_t cols;
    float scale;
};

/// Per-head RMS normalization with a per-dimension scale over the q and k
/// head sections of fused qkv rows, in place.
struct dit_args_qk_norm {
    uint32_t tokens;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t row_stride;
    float eps;
};

/// Per-head RMS normalization with a per-dimension scale, then interleaved
/// RoPE from a per-token table, over separate packed q and k buffers, in
/// place. `cos`/`sin` hold `head_dim / 2` angles per token: pair `i` of a
/// head, `(x[2i], x[2i+1])`, turns by the angle in column `i`.
struct dit_args_qk_norm_rope {
    uint32_t tokens;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t q_stride;
    uint32_t k_stride;
    float eps;
};

/// Row-wise concatenation: `out[r] = [a[r], b[r]]`.
struct dit_args_concat {
    uint32_t rows;
    uint32_t cols_a;
    uint32_t cols_b;
};

/// Joint-sequence attention over fused or packed rows. Head `h` of a row
/// starts at `row * stride + offset + h * head_dim` for each operand's own
/// offset; the cached rows carry the same layout against `cache_stride`.
struct dit_args_sdpa {
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
/// Keys at or past `valid_keys` are masked for every row (`tokens` masks
/// nothing): the padded prefill's pad rows still attend every real key.
struct dit_args_gqa {
    uint32_t tokens;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t valid_keys;
};

/// 4-axis interleaved RoPE on the q and k head sections of strided rows, in
/// place, one launch for both.
struct dit_args_rope_4d {
    uint32_t tokens;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t freqs_per_axis;
    uint32_t row_stride;
    uint32_t q_offset;
    uint32_t k_offset;
};

/// Rotate-half 1-axis RoPE on one packed (tokens, heads, head_dim) tensor.
/// Rows from `pinned_from` on turn by position `pinned_pos` instead of their
/// own index (`pinned_from == tokens` pins nothing).
struct dit_args_rope_half {
    uint32_t tokens;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t pinned_from;
    uint32_t pinned_pos;
};

/// One NCHW f32 convolution. `has_bias` is the presence contract from
/// `dit_args_gemm`.
struct dit_args_conv2d {
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
    uint32_t has_bias;
};

/// Group normalization; the affine buffers hold `num_groups *
/// channels_per_group` elements, indexed by channel.
struct dit_args_group_norm {
    uint32_t batch;
    uint32_t num_groups;
    uint32_t channels_per_group;
    uint32_t spatial;
    float eps;
    /// Slices each group is split into, one threadgroup apiece.
    uint32_t chunks;
};

/// Nearest-neighbour 2x upsampling of an NCHW field.
struct dit_args_upsample {
    uint32_t batch;
    uint32_t channels;
    uint32_t h;
    uint32_t w;
};

/// Gather of a BF16 embedding table, widened to f32 on write.
struct dit_args_embed {
    uint32_t tokens;
    uint32_t hidden;
};

/// Per-pixel inverse norm over the channel axis of an NCHW field (the
/// channel RMS norm of the residual VAE decoder): `inv[s] = scale /
/// max(||x[:, s]||_2, eps)`, so `x[c, s] * inv[s] * gamma[c]` is the
/// reference's `normalize(x, dim=channels) * sqrt(C) * gamma`.
struct dit_args_channel_rms {
    uint32_t channels;
    uint32_t spatial;
    float scale;
    float eps;
};

/// A same-padded, stride-1 square convolution of the residual VAE decoder
/// with its input transform fused into the im2col gather (channel-RMS
/// norm, optionally SiLU; or a nearest 2x upsample) and an optional
/// accumulate epilogue (`out += conv`, the residual add). `in_h`/`in_w` are
/// the stored field's; for the upsampling entry the convolved field is
/// twice that, and `out_h`/`out_w` say so.
struct dit_args_conv_vae {
    uint32_t in_c;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t out_c;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t k;
    uint32_t pad;
    uint32_t has_bias;
    uint32_t accumulate;
    uint32_t silu;
    uint32_t reserved;
};

/// The residual VAE decoder's parameter-free upsampling shortcut, added
/// into a field at twice the source's size: `out[c, y, x] += in[(c * factor
/// + offset + (y & 1) * 2 + (x & 1)) / repeats, y / 2, x / 2]`. `factor` is
/// the temporal times the 2x2 spatial factor, `offset` selects the last
/// temporal slot (a single frame keeps only it), `repeats` is the channel
/// duplication `out_c * factor / in_c`.
struct dit_args_dup_up {
    uint32_t out_c;
    uint32_t h;
    uint32_t w;
    uint32_t factor;
    uint32_t offset;
    uint32_t repeats;
};

// The blocks and the driver's Pods are edited together; the host asserts the
// same widths, and so does this side so a header-only drift fails the compile.
static_assert(sizeof(dit_args_gemm) == 16, "GemmArgs");
static_assert(sizeof(dit_args_unary) == 4, "UnaryArgs");
static_assert(sizeof(dit_args_gemm_fused) == 28, "GemmFusedArgs");
static_assert(sizeof(dit_args_hadamard) == 20, "HadamardArgs");
static_assert(sizeof(dit_args_swiglu) == 12, "SwigluArgs");
static_assert(sizeof(dit_args_modulate) == 8, "ModulateArgs");
static_assert(sizeof(dit_args_add_gated) == 8, "AddGatedArgs");
static_assert(sizeof(dit_args_axpy) == 8, "AxpyArgs");
static_assert(sizeof(dit_args_norm) == 12, "NormArgs");
static_assert(sizeof(dit_args_softmax) == 12, "SoftmaxArgs");
static_assert(sizeof(dit_args_qk_norm) == 20, "QkNormArgs");
static_assert(sizeof(dit_args_qk_norm_rope) == 24, "QkNormRopeArgs");
static_assert(sizeof(dit_args_concat) == 12, "ConcatArgs");
static_assert(sizeof(dit_args_sdpa) == 44, "SdpaArgs");
static_assert(sizeof(dit_args_gqa) == 20, "GqaArgs");
static_assert(sizeof(dit_args_rope_4d) == 28, "Rope4dArgs");
static_assert(sizeof(dit_args_rope_half) == 20, "RopeHalfArgs");
static_assert(sizeof(dit_args_conv2d) == 48, "Conv2dArgs");
static_assert(sizeof(dit_args_group_norm) == 24, "GroupNormArgs");
static_assert(sizeof(dit_args_upsample) == 16, "UpsampleArgs");
static_assert(sizeof(dit_args_embed) == 8, "EmbedArgs");
static_assert(sizeof(dit_args_channel_rms) == 16, "ChannelRmsArgs");
static_assert(sizeof(dit_args_conv_vae) == 48, "ConvVaeArgs");
static_assert(sizeof(dit_args_dup_up) == 24, "DupUpArgs");

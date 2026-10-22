// Shared widths, reduction shapes and argument blocks for the vision-tower
// kernel set.  This file is concatenated ahead of every other vision entry, so
// it may not include anything and the files after it may not need to declare a
// helper.
//
// There is no upstream shading-language source for this tower: the set is
// written against two independent specs of the same arithmetic, and the naive
// f32 oracle beside it is the only thing that separates an ordering mistake
// from a model that answers confidently about a picture it never saw.  So the
// reductions below are spelled out rather than taken from the builtin, whose
// summation order the shading language leaves unspecified, and every kernel
// accumulates in f32 no matter what width it read from.

// Square tile the blocked GEMM walks, in both output axes and along the
// reduction axis.  16x16 threads is one threadgroup of 256 and two 1 KiB
// staging tiles, which fits every device this runs on with room to spare.
#define VIT_GEMM_TILE 16u

// The simdgroup-matrix GEMM's tile: 64 rows by 64 columns a threadgroup of
// four simdgroups, k staged 16 deep, each staged row padded by four floats
// against bank conflicts. See `kernel_vit_gemm64_bias`.
#define VIT_GEMM64_TILE 64u
#define VIT_GEMM64_K 16u
#define VIT_GEMM64_THREADS 128u

// Full unroll of a loop whose trip count is a compile-time constant. The
// simdgroup-matrix fragments only stay in registers when every loop that
// indexes them is unrolled; left as loops they spill and the tile runs several
// times slower.
#define VIT_UNROLL _Pragma("clang loop unroll(full)")

// The simdgroup-matrix attention: eight simdgroups of eight query rows a
// threadgroup, keys streamed 32 at a time. See `kernel_vit_flash_attn_d*`.
#define VIT_FLASH_SG 8u
#define VIT_FLASH_QROWS (8u * VIT_FLASH_SG)
#define VIT_FLASH_KEYS 32u
#define VIT_FLASH_THREADS (32u * VIT_FLASH_SG)

// Per-lane accumulator slots in the attention kernel: one per 32 head-dim
// values a lane strides over, so it bounds head_dim at 256.  Written as a fixed
// array because a lane's accumulator has to live in registers for the online
// softmax to be worth anything.
#define VIT_ATTN_SLOTS 8u

// 32 simdgroup partials plus the broadcast slot; see vit_block_sum.
#define VIT_BLOCK_SUM_SLOTS 33u

/// One brain-float, widened.
///
/// The 16 bits are loaded as an integer and shifted into the high half of an
/// f32 rather than being cast through a native type.  That is exact for every
/// finite value, it preserves NaN and infinity payloads, and it does not depend
/// on which shading-language version the device happens to accept.
static inline float vit_bf16(device const ushort *p) {
    return as_type<float>((uint) (*p) << 16);
}

// One IEEE half, widened. Exact: every half is a single-precision value.
static inline float vit_f16(device const ushort *p) {
    return float(as_type<half>(*p));
}

// How a tower's weights are stored, as a type the weight-reading kernels are
// templated on: one body per kernel, instantiated once per storage width.
// Each entry keeps its brain-float name and gains an `_f16` twin.
// `load4` widens four consecutive values; it reads through a packed vector
// because a weight row is only element-aligned in general.
struct vit_w_bf16 {
    static inline float load(device const ushort *p) { return vit_bf16(p); }
    static inline float4 load4(device const ushort *p) {
        const uint4 bits = uint4(ushort4(*((device const packed_ushort4 *) p)));
        return as_type<float4>(bits << 16);
    }
};
struct vit_w_f16 {
    static inline float load(device const ushort *p) { return vit_f16(p); }
    static inline float4 load4(device const ushort *p) {
        return float4(as_type<half4>(ushort4(*((device const packed_ushort4 *) p))));
    }
};

/// Inclusive sum over one simdgroup, left-folded in decreasing-offset order so
/// lane 0 ends up holding the total.  Spelled out instead of using the builtin
/// because the shading language leaves that summation order unspecified.
static inline float vit_warp_sum(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

/// Sum over one simdgroup, butterfly-folded so every lane ends up holding the
/// total.  The attention kernel needs the total in every lane, and a broadcast
/// after a one-sided fold would be a second ordering to keep in step.
static inline float vit_warp_sum_all(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += simd_shuffle_xor(v, off);
    }
    return v;
}

/// Block-wide reduction over at most 32 simdgroups.
///
/// Shape: warp reduce, lane 0 publishes to scratch[warp], barrier, warp 0
/// re-sums the lanes below n_warp, thread 0 broadcasts through the last slot.
/// `scratch` needs VIT_BLOCK_SUM_SLOTS floats and is passed in because an
/// inlined helper has no threadgroup array of its own to keep.
static inline float vit_block_sum(
        float v,
        threadgroup float *scratch,
        uint tid,
        uint nthreads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = vit_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = vit_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[VIT_BLOCK_SUM_SLOTS - 1u] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[VIT_BLOCK_SUM_SLOTS - 1u];
}

/// Input past which the tanh form of GELU has saturated.
///
/// `tanh` is evaluated as `(e^2x - 1) / (e^2x + 1)`, so an argument past about
/// 44 overflows single precision on both sides of that fraction and the result
/// is `inf / inf`, which is NaN. The cubic inside GELU reaches 44 at an input of
/// only about 10.3 -- a value a feed-forward row of this tower hits several
/// dozen times per image -- so without a guard the tower produces a handful of
/// NaNs in its first block, the layer normalization that follows spreads each
/// one over its whole row, and twenty-two blocks later every row is NaN and the
/// answer is the argmax of an all-NaN logits row.
///
/// Ten is below that threshold and is where the function has already become its
/// own limit to the last bit of an f32: `tanh` of the cubic at 10 is 1.0f
/// exactly, so GELU is `v` exactly above it and `0.0f` exactly below `-10`.
/// Selecting those limits therefore changes no representable result.
#define VIT_GELU_LINEAR 10.0f

/// The tanh approximation of GELU, written in the reference's own form.
///
/// Not interchangeable with the exact form below: the two differ by about 1e-4
/// per value, and the blocks apply this one two dozen times in a row on the
/// residual stream, so substituting the other is a drift the tower carries all
/// the way to the language model.
///
/// The saturation guard is a select on the INPUT applied after the expression,
/// not a clamp inside it. That placement is load-bearing: wrapping the tanh's
/// argument in a `clamp` inhibits the contraction the compiler otherwise
/// performs across the cubic, which moves every in-range result by an ulp and
/// moves a whole-tower comparison by about 2e-5 -- past the oracle's tolerance,
/// for a change that is numerically a no-op. Leaving the expression untouched
/// and discarding its result where it cannot be evaluated keeps the arithmetic
/// bit-for-bit what it was.
static inline float vit_gelu_tanh(float v) {
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    const float g = 0.5f * v * (1.0f + tanh(c * (v + 0.044715f * v * v * v)));
    return (v >= VIT_GELU_LINEAR) ? v : ((v <= -VIT_GELU_LINEAR) ? 0.0f : g);
}

/// The error function.
///
/// The shading language has none, so this is the standard rational-times-
/// Gaussian form (Abramowitz and Stegun 7.1.26), whose error is bounded by
/// 1.5e-7 everywhere -- about one unit in the last place of an f32 near 1, and
/// three orders of magnitude tighter than the 1e-4 that separates the two GELUs
/// below. Written here rather than at the call site so the host oracle has one
/// formula to transcribe rather than an inlined expression to re-derive.
static inline float vit_erf(float x) {
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    const float a = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float poly = t * (0.254829592f
            + t * (-0.284496736f
            + t * (1.421413741f
            + t * (-1.453152027f
            + t * 1.061405429f))));
    return sign * (1.0f - poly * exp(-a * a));
}

/// The exact GELU, through the error function.
///
/// The merger uses this one and the blocks use the approximation above. That is
/// not an inconsistency to tidy up: the two layers declare different
/// activations, the merger runs once per image, and the exact form costs
/// nothing at that rate.
static inline float vit_gelu_erf(float v) {
    return 0.5f * v * (1.0f + vit_erf(v * 0.7071067811865476f));
}

// --- argument blocks -------------------------------------------------------
//
// One block per kernel family, passed as `constant T & args [[buffer(0)]]`.
// Field order is the binding contract the host mirrors, so it is spelled out
// here once and never re-derived inside a kernel.

/// Blocked GEMM with a bias epilogue: `out[m, n] = in[m, k] . weight[n, k] + bias[n]`.
struct vit_args_gemm {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

/// Mean/variance normalization of one row, with a scale and an offset.
struct vit_args_layer_norm {
    uint32_t rows;
    uint32_t dim;
    float eps;
};

/// One elementwise pass over `count` values.
struct vit_args_unary {
    uint32_t count;
};

/// `x[i] += add[i % period]`.
///
/// The period exists for exactly one caller: a clip's temporal groups all sit
/// on the same patch grid, so one grid's worth of resampled position vectors is
/// added to each group in turn rather than being replicated on the host. Every
/// other caller passes `period == count`, which is the plain elementwise add.
struct vit_args_add {
    uint32_t count;
    uint32_t period;
};

/// Split the fused projection into three planes and rotate two of them.
struct vit_args_qkv_rope {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t grid_w;
    /// Patch rows in one temporal group, `grid_h * grid_w`. A row's spatial
    /// position is its offset inside its own group, so a clip's later frames
    /// repeat the first frame's grid rather than continuing past it.
    uint32_t plane;
    float rope_base;
};

/// Bidirectional attention over one temporal group of patch rows.
struct vit_args_attention {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    /// Patch rows in one temporal group; also the span a query attends over.
    uint32_t plane;
    /// Multiplicative factor on every query before the softmax. The kernel used
    /// to derive it as `rsqrt(head_dim)`, which is the right number for a tower
    /// that scales by the inverse root of the head width and the wrong one for a
    /// tower whose reference holds the score unscaled. It is an argument now so
    /// the two towers can share this kernel; each passes what its own reference
    /// computes.
    float scale;
};

/// Fold a `merge` x `merge` block of patch rows into one row.
struct vit_args_merge {
    uint32_t merged_rows;
    uint32_t hidden;
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t merge;
};

/// Overwrite the language model's embedding rows at the placeholder positions
/// with the rows this tower produced.
struct vit_args_splice {
    /// Token rows in the chunk, which is the destination row count.
    uint32_t rows;
    /// Width of one residual-stream row.
    uint32_t hidden;
    /// Rows in the encoder block, so a source index out of range is refused
    /// rather than read.
    uint32_t src_rows;
};

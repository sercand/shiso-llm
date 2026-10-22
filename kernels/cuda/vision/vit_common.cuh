/* Shared widths, reduction shapes, activations and argument blocks for the
 * parametric vision-tower set.
 *
 * The CUDA half of `kernels/metal/vision/vit_common.h`, written
 * against the same contract: nothing here assumes a hidden width, a depth, a
 * head count or a patch geometry, because the two towers this tree loads differ
 * in all four and a literal would be an encoder running the wrong arithmetic on
 * the other checkpoint while reporting success.
 *
 * Nothing in this file is copied. It is the Metal helpers' fold order
 * transcribed onto warp shuffles and the argument structs in the field order
 * the two encoders pack, so `q4e_ptx_oracle.sh --mode=fidelity` reports the unit
 * that includes it as generated.
 *
 * # Why the reductions are spelled out
 *
 * `metal/vision/KERNELS.md`: reductions only go through
 * `vit_warp_sum`/`vit_warp_sum_all`/`vit_block_sum`. There is no upstream
 * shading-language source for this tower and the naive f32 oracle beside it is
 * the whole gate, so the kernels are written against a fixed summation order
 * rather than a mathematically equal one, and this half has to fold the same
 * way for the same oracle to grade it.
 */
#ifndef SHISO_VIT_COMMON_CUH
#define SHISO_VIT_COMMON_CUH

#include <cuda_fp16.h>
#include <stdint.h>

/* Square tile the blocked GEMM walks, in both output axes and along the
 * reduction axis. 16x16 threads is one block of 256 and two 1 KiB staging
 * tiles. */
#define VIT_GEMM_TILE 16u

/* Per-lane accumulator slots in the attention kernel: one per 32 head-dim
 * values a lane strides over, so it bounds head_dim at 256. A fixed array
 * because a lane's accumulator has to live in registers for the online softmax
 * to be worth anything. */
#define VIT_ATTN_SLOTS 8u

/* 32 warp partials plus the broadcast slot; see `vit_block_sum`. */
#define VIT_BLOCK_SUM_SLOTS 33u

/* Every lane of a warp, which is the participation mask every shuffle below
 * takes: each one sits at a point the whole warp reaches. */
#define VIT_FULL_WARP 0xffffffffu

/* Threads the three flat kernels are launched with.
 *
 * Hard-coded rather than read off `blockDim`, exactly as the Metal twins hard-
 * code it, so the two halves index identically. It is the one block width those
 * three entries may be launched at, and both encoders and every case launch
 * them at it. */
#define VIT_LINEAR_BLOCK 256u

/* Dynamic shared memory, declared once with an alignment the float view below
 * satisfies. The normalization and the attention are the two entries that ask
 * for it; the driver hands over a 16-byte aligned base. */
extern __shared__ __align__(16) char vit_smem[];

/* One brain-float, widened.
 *
 * The 16 bits are loaded as an integer and shifted into the high half of an f32
 * rather than being cast through a native type. That is exact for every finite
 * value, it preserves NaN and infinity payloads, and it matches the Metal
 * twin's widening bit for bit. */
__device__ __forceinline__ float vit_bf16(const unsigned short *p) {
    return __uint_as_float((unsigned) (*p) << 16);
}

/* One IEEE half, widened. Exact: every half is a single-precision value. */
__device__ __forceinline__ float vit_f16(const unsigned short *p) {
    return __half2float(__ushort_as_half(*p));
}

/* How a tower's weights are stored, as a type the weight-reading kernels are
 * templated on: one body per kernel, instantiated once per storage width, so
 * the two widths cannot drift apart in anything but the load. Each entry point
 * keeps its brain-float name and gains an `_f16` twin. */
struct vit_w_bf16 {
    static __device__ __forceinline__ float load(const unsigned short *p) { return vit_bf16(p); }
};
struct vit_w_f16 {
    static __device__ __forceinline__ float load(const unsigned short *p) { return vit_f16(p); }
};

/* Inclusive sum over one warp, left-folded in decreasing-offset order so lane 0
 * ends up holding the total. Spelled out rather than reduced with a library
 * call for the reason the Metal twin is: a fixed summation order is part of the
 * contract. */
__device__ __forceinline__ float vit_warp_sum(float v) {
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
        v += __shfl_down_sync(VIT_FULL_WARP, v, off);
    }
    return v;
}

/* Sum over one warp, butterfly-folded so every lane ends up holding the total.
 * The attention needs the total in every lane, and a broadcast after a one-sided
 * fold would be a second ordering to keep in step. */
__device__ __forceinline__ float vit_warp_sum_all(float v) {
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
        v += __shfl_xor_sync(VIT_FULL_WARP, v, off);
    }
    return v;
}

/* Block-wide reduction over at most 32 warps.
 *
 * The Metal helper's shape exactly: warp reduce, lane 0 publishes to
 * `scratch[warp]`, barrier, warp 0 re-sums the lanes below `n_warp`, thread 0
 * broadcasts through the last slot. `scratch` needs VIT_BLOCK_SUM_SLOTS floats.
 * Every thread of the block must reach this call: it contains two barriers. */
__device__ __forceinline__ float vit_block_sum(
        float v, float *scratch, unsigned tid, unsigned nthreads) {
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    v = vit_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    __syncthreads();
    const unsigned n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = vit_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[VIT_BLOCK_SUM_SLOTS - 1u] = v;
    }
    __syncthreads();
    return scratch[VIT_BLOCK_SUM_SLOTS - 1u];
}

/* Input past which the tanh form of GELU has saturated.
 *
 * `tanh` is evaluated as `(e^2x - 1) / (e^2x + 1)`, so an argument past about 44
 * overflows single precision on both sides of that fraction and the result is
 * `inf / inf`, which is NaN. The cubic inside GELU reaches 44 at an input of
 * only about 10.3 -- a value a feed-forward row of this tower hits several dozen
 * times per image -- so without a guard the tower produces a handful of NaNs in
 * its first block, the normalization that follows spreads each one over its
 * whole row, and twenty-two blocks later every row is NaN.
 *
 * Ten is below that threshold and is where the function has already become its
 * own limit to the last bit of an f32, so selecting those limits changes no
 * representable result. */
#define VIT_GELU_LINEAR 10.0f

/* The tanh approximation of the Gaussian error linear unit, in the reference's
 * own form: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
 *
 * `tanhf` is the accurate entry, which is what `precise::tanh` is on the Metal
 * side and why this directory's manifest drops `--use_fast_math`.
 *
 * The saturation guard is a select on the INPUT applied after the expression,
 * not a clamp inside it, and that placement is load-bearing on this side too:
 * wrapping the argument in a clamp inhibits the contraction the compiler
 * performs across the cubic, which moves every in-range result by an ulp for a
 * change that is numerically a no-op. */
__device__ __forceinline__ float vit_gelu_tanh(float v) {
    const float c = 0.7978845608028654f;  /* sqrt(2/pi) */
    const float g = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    return (v >= VIT_GELU_LINEAR) ? v : ((v <= -VIT_GELU_LINEAR) ? 0.0f : g);
}

/* The error function, in the Metal twin's series rather than `erff`.
 *
 * The shading language has none, so the Metal set carries the standard rational-
 * times-Gaussian form (Abramowitz and Stegun 7.1.26), whose error is bounded by
 * 1.5e-7 everywhere. This side has `erff` and could call it; it does not,
 * because the shared oracle in `shiso-kerneltest` transcribes this series and a
 * case that grades both backends should be grading the same function on both.
 * The difference either way is three orders of magnitude inside the tolerance
 * and four inside the 1e-4 that separates the two GELUs, which is what the case
 * actually exists to catch. */
__device__ __forceinline__ float vit_erf(float x) {
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    const float a = fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float poly = t * (0.254829592f
            + t * (-0.284496736f
            + t * (1.421413741f
            + t * (-1.453152027f
            + t * 1.061405429f))));
    return sign * (1.0f - poly * expf(-a * a));
}

/* The exact GELU, through the error function.
 *
 * The merger uses this one and the blocks use the approximation above. The two
 * differ by about 1e-4 per value, the choice belongs to the layer, and a branch
 * that picked between them could be got wrong at one call site without anything
 * failing loudly -- which is why they are two entry points. */
__device__ __forceinline__ float vit_gelu_erf(float v) {
    return 0.5f * v * (1.0f + vit_erf(v * 0.7071067811865476f));
}

/* --- argument blocks ------------------------------------------------------
 *
 * Field order is the binding contract the two encoders and the engine mirror,
 * and each block carries a static assertion on its size: a launch whose
 * argument block is one field out reads a stride as an extent and computes a
 * product that is confidently wrong. */

/* Blocked GEMM with a bias epilogue:
 * `out[m, n] = in[m, k] . weight[n, k] + bias[n]`. */
struct vit_args_gemm {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};
static_assert(sizeof(struct vit_args_gemm) == 12, "vit_args_gemm is 12 bytes");

/* Mean/variance normalization of one row, with a scale and an offset. */
struct vit_args_layer_norm {
    uint32_t rows;
    uint32_t dim;
    float eps;
};
static_assert(sizeof(struct vit_args_layer_norm) == 12, "vit_args_layer_norm is 12 bytes");

/* One elementwise pass over `count` values. */
struct vit_args_unary {
    uint32_t count;
};
static_assert(sizeof(struct vit_args_unary) == 4, "vit_args_unary is 4 bytes");

/* `x[i] += add[i % period]`.
 *
 * The period exists for exactly one caller: a clip's temporal groups all sit on
 * the same patch grid, so one grid's worth of resampled position vectors is
 * added to each group in turn rather than being replicated on the host. Every
 * other caller passes `period == count`, which is the plain elementwise add. */
struct vit_args_add {
    uint32_t count;
    uint32_t period;
};
static_assert(sizeof(struct vit_args_add) == 8, "vit_args_add is 8 bytes");

/* Split the fused projection into three planes and rotate two of them. */
struct vit_args_qkv_rope {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t grid_w;
    /* Patch rows in one temporal group, `grid_h * grid_w`. A row's spatial
     * position is its offset inside its own group, so a clip's later frames
     * repeat the first frame's grid rather than continuing past it. */
    uint32_t plane;
    float rope_base;
};
static_assert(sizeof(struct vit_args_qkv_rope) == 24, "vit_args_qkv_rope is 24 bytes");

/* Bidirectional attention over one temporal group of patch rows. */
struct vit_args_attention {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    /* Patch rows in one temporal group; also the span a query attends over. */
    uint32_t plane;
    /* Multiplicative factor on every query before the softmax. It is an
     * argument rather than `rsqrt(head_dim)` so the two towers can share this
     * kernel: the Qwen one passes its head's inverse root width and Gemma 4
     * passes one. */
    float scale;
};
static_assert(sizeof(struct vit_args_attention) == 20, "vit_args_attention is 20 bytes");

/* Fold a `merge` x `merge` block of patch rows into one row. */
struct vit_args_merge {
    uint32_t merged_rows;
    uint32_t hidden;
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t merge;
};
static_assert(sizeof(struct vit_args_merge) == 20, "vit_args_merge is 20 bytes");

/* Overwrite the language model's embedding rows at the placeholder positions
 * with the rows this tower produced. */
struct vit_args_splice {
    /* Token rows in the chunk, which is the destination row count. */
    uint32_t rows;
    /* Width of one residual-stream row. */
    uint32_t hidden;
    /* Rows in the encoder block, so a source index out of range is refused
     * rather than read. */
    uint32_t src_rows;
};
static_assert(sizeof(struct vit_args_splice) == 12, "vit_args_splice is 12 bytes");

#endif /* SHISO_VIT_COMMON_CUH */

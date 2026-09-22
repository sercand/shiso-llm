/* Shared reduction shapes, activations and argument blocks for the dense set.
 *
 * The CUDA half of `crates/shiso-metal/metal/dense/dense_common.h`, written
 * against the same contract: "dense" is the parametric family, so nothing here
 * assumes a hidden width, a head count or a head dimension, and every dimension
 * arrives in an argument block rather than in a constant.
 *
 * Nothing in this file is a byte copy. The reductions are the Metal helpers'
 * fold order transcribed onto warp shuffles, the argument structs are the field
 * order `crates/shiso-engine/src/launch.rs` packs, and the `q6_K` block is a
 * file layout rather than anybody's source.
 *
 * # Why the reductions are spelled out
 *
 * `KERNELS.md`: reductions only go through `dense_warp_sum`/`dense_block_sum`,
 * because a golden that pins a decode step pins the order its partial sums were
 * added in. The same obligation holds here, and the same shape: a
 * decreasing-offset left fold, lane 0 publishing to scratch, warp 0 re-summing,
 * thread 0 broadcasting through the last slot.
 */
#ifndef SHISO_DENSE_COMMON_CUH
#define SHISO_DENSE_COMMON_CUH

#include <cuda_fp16.h>
#include <stdint.h>

/* 32 warp partials plus the broadcast slot; see `dense_block_sum`. */
#define DENSE_BLOCK_SUM_SLOTS 33u
/* Warps per attention threadgroup (256 threads). */
#define DENSE_ATTN_WARPS 8u
/* Widest head this set stages in shared memory. */
#define DENSE_ATTN_HD 512u
/* Accumulator slots one lane holds: DENSE_ATTN_HD / 32. */
#define DENSE_ATTN_LANE_SLOTS 16u
/* Queries per prefill threadgroup: two per warp. */
#define DENSE_ATTN_QT 16u
/* Keys per prefill tile, one per lane of the low half of a warp. */
#define DENSE_ATTN_KT 16u
/* Weights in one per-layer embedding row, which is one q6_K block exactly. */
#define DENSE_PLE_DIM 256u
/* Elements one q6_K dequantize call produces. */
#define DENSE_PLE_GROUP 16u
/* Every lane of a warp, which is the participation mask every shuffle below
 * takes: each one sits at a point the whole warp reaches. */
#define DENSE_FULL_WARP 0xffffffffu

/* Dynamic shared memory, declared once with an alignment the float and half
 * views below both satisfy. Every kernel that asks for threadgroup memory
 * reinterprets this one array; the driver hands over a 16-byte aligned base. */
extern __shared__ __align__(16) char dense_smem[];

/* Inclusive sum over one warp, left-folded in decreasing-offset order so lane 0
 * ends up holding the total. Spelled out rather than reduced with a library
 * call for the reason the Metal twin is: a fixed summation order is part of the
 * contract. */
__device__ __forceinline__ float dense_warp_sum(float v) {
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
        v += __shfl_down_sync(DENSE_FULL_WARP, v, off);
    }
    return v;
}

/* Block-wide reduction over at most 32 warps.
 *
 * The Metal helper's shape exactly: warp reduce, lane 0 publishes to
 * `scratch[warp]`, barrier, warp 0 re-sums the lanes below `n_warp`, thread 0
 * broadcasts through the last slot. `scratch` needs DENSE_BLOCK_SUM_SLOTS
 * floats and is passed in because this is inlined into a kernel that owns the
 * shared allocation. Every thread of the block must reach this call: it
 * contains two barriers. */
__device__ __forceinline__ float dense_block_sum(
        float v, float *scratch, unsigned tid, unsigned nthreads) {
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    v = dense_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    __syncthreads();
    const unsigned n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = dense_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[DENSE_BLOCK_SUM_SLOTS - 1u] = v;
    }
    __syncthreads();
    return scratch[DENSE_BLOCK_SUM_SLOTS - 1u];
}

/* Fused multiply-add, wrapped once so an inner product reads as one decision
 * and no call site can pick up a non-fused multiply by accident. */
__device__ __forceinline__ float dense_fma(float a, float b, float c) {
    return fmaf(a, b, c);
}

/* The tanh approximation of the Gaussian error linear unit:
 * 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
 *
 * `tanhf` is the accurate entry, which is what `precise::tanh` is on the Metal
 * side and why this directory's manifest drops `--use_fast_math`. The cubic
 * grows fast: a gate of about 10.7 already puts the argument past 44, where a
 * rational form in exp(2y) hands back inf/inf, and one NaN element poisons the
 * whole projection below it. */
__device__ __forceinline__ float dense_gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

/* Widens one bf16 value by placing its 16 bits into the high half of an f32. */
__device__ __forceinline__ float dense_bf16(const unsigned short *p) {
    return __uint_as_float((unsigned) (*p) << 16);
}

/* Physical row of a logical KV position, through the session page table.
 *
 * `pages[logical page]` is the physical page holding that span, so a position
 * becomes `pages[p / page_tokens] * page_tokens + p % page_tokens`. A null
 * table means the caller's rows are already physical. */
__device__ __forceinline__ uint64_t dense_kv_row(
        const int32_t *pages, uint64_t position, uint32_t page_tokens) {
    if (pages == nullptr) {
        return position;
    }
    return (uint64_t) pages[position / page_tokens] * page_tokens + (position % page_tokens);
}

/* Row `t`'s own page table inside a packed block of them.
 *
 * A decode step may carry one row per SESSION, and two sessions share no page:
 * a launch that read one table for every row would answer row one out of row
 * zero's conversation, fluently and at the right shape. So the table a row
 * reads is selected here, by the row index the grid already carries.
 *
 * `stride` is entries between one row's table and the next, and `0` is the
 * shared case rather than a sentinel - `pages + t * 0` is `pages` for every
 * row, which is exactly what a prefill chunk and a single-session step want.
 * A null table stays null, because that is the "no paging, position IS the
 * row" convention `dense_kv_row` answers for. */
__device__ __forceinline__ const int32_t *dense_row_pages(
        const int32_t *pages, unsigned t, uint32_t stride) {
    return (pages == nullptr) ? pages : pages + (uint64_t) t * stride;
}

/* Rotary over the leading `n_rot` dimensions of a block-resident head, with the
 * frequency ladder derived over `freq_dims`.
 *
 * Pair `i` with `i + freq_dims / 2` for `i < n_rot / 2`, at the angle
 * `pos * base^(-2i / freq_dims)`. THE TWO WIDTHS ARE SEPARATE ARGUMENTS: at
 * head_dim 512, n_rot 128 and freq_dims 512 the pairs are (i, i + 256) for
 * i < 64, and a kernel that read the pair stride off `n_rot` would pair
 * (i, i + 64) instead -- every head still normalized, every head still rotated,
 * every head wrong. Setting freq_dims = n_rot recovers the neox rotation.
 *
 * PRECONDITION: freq_dims >= n_rot. Below that the partner index lands back
 * inside the loop's own range and two lanes race over one dimension. Nothing
 * here checks it -- a kernel cannot refuse -- so the engine validates the pair
 * on the host.
 *
 * `powf`, `sinf` and `cosf` are the accurate library entries, as the Metal twin
 * uses the accurate builtins: the ladder spans six decades of base and the
 * reduced-precision intrinsics lose the far end of it. That is enforced by this
 * directory's manifest dropping `--use_fast_math` rather than by a spelling
 * here, because under that flag nvcc rewrites all three calls and no source
 * spelling opts back out. */
__device__ __forceinline__ void dense_rope_pairs(
        float *v, unsigned n_rot, unsigned freq_dims, float base, float pos,
        unsigned lane, unsigned nlanes) {
    const unsigned half_width = freq_dims / 2u;
    const unsigned n_pair = n_rot / 2u;
    for (unsigned i = lane; i < n_pair; i += nlanes) {
        const float theta = pos * powf(base, -2.0f * (float) i / (float) freq_dims);
        const float s = sinf(theta);
        const float c = cosf(theta);
        const float x0 = v[i];
        const float x1 = v[i + half_width];
        v[i] = x0 * c - x1 * s;
        v[i + half_width] = x0 * s + x1 * c;
    }
}

/* --- argument blocks ------------------------------------------------------
 *
 * Field order is the binding contract `crates/shiso-engine/src/launch.rs`
 * mirrors, and each block carries a static assertion on its size: a launch
 * whose argument block is one field out reads a stride as an extent and
 * computes a product that is confidently wrong. */

/* `x[i] *= s` over the first `n` elements. */
struct dense_args_scale {
    uint32_t n;
    float s;
};
static_assert(sizeof(struct dense_args_scale) == 8, "ScaleArgs is 8 bytes");

/* `x[i] = cap * tanh(x[i] / cap)` over the first `n` elements. */
struct dense_args_softcap {
    uint32_t n;
    float cap;
};
static_assert(sizeof(struct dense_args_softcap) == 8, "SoftcapArgs is 8 bytes");

/* Per-head RMSNorm plus the parametric rotation of the query projection. */
struct dense_args_q_norm_rope {
    uint32_t head_dim;
    uint32_t n_head;
    /* Leading dimensions the rotation touches. */
    uint32_t n_rot;
    /* Width the frequency ladder is derived over; NOT `n_rot` in general. */
    uint32_t freq_dims;
    uint32_t n_tok;
    float rope_base;
    float eps;
};
static_assert(sizeof(struct dense_args_q_norm_rope) == 28, "DenseQNormRopeArgs is 28 bytes");

/* Key norm + rotation and the value store, both into the f16 cache. */
struct dense_args_kv_norm_rope {
    uint32_t head_dim;
    uint32_t n_head_kv;
    uint32_t n_rot;
    uint32_t freq_dims;
    uint32_t n_tok;
    float rope_base;
    float eps;
    uint32_t page_tokens;
    /* Non-zero when the value is normalized before it is cached, by the same
     * mean square the key uses and with NO weight. */
    uint32_t v_norm;
    /* Entries between one row's page table and the next, or 0 when every row
     * of this launch reads the table the pointer names. See
     * `dense_row_pages`. */
    uint32_t page_stride;
};
static_assert(sizeof(struct dense_args_kv_norm_rope) == 40, "DenseKvNormRopeArgs is 40 bytes");

/* Windowed, bounded and scaled GQA over the paged f16 cache. */
struct dense_args_attn {
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_tok;
    uint32_t page_tokens;
    /* Positions a query may look back over; 0 is no window at all. */
    uint32_t window;
    /* What every reduced score is multiplied by. */
    float scale;
    /* Entries between one row's page table and the next, or 0 when every row
     * of this launch reads the table the pointer names. See
     * `dense_row_pages`. */
    uint32_t page_stride;
};
static_assert(sizeof(struct dense_args_attn) == 32, "DenseAttnArgs is 32 bytes");

/* The per-layer embedding mix, one threadgroup per (token, layer slot). */
struct dense_args_ple {
    uint32_t n_tok;
    uint32_t n_layers;
    /* Weights in one slot's row. Must be DENSE_PLE_DIM. */
    uint32_t dim;
    /* Bytes ONE TOKEN's whole row occupies: one slot's block per layer. */
    uint32_t row_bytes;
    /* Non-zero when `rows` is the whole table and `ids` picks the row. */
    uint32_t resident;
    float eps;
    /* Folded into the projection BEFORE the sum of squares. */
    float proj_scale;
    float embed_scale;
    float mix_scale;
};
static_assert(sizeof(struct dense_args_ple) == 36, "DensePleArgs is 36 bytes");

/* The gate multiply at one layer slot. */
struct dense_args_ple_gate {
    uint32_t n_tok;
    uint32_t n_layers;
    uint32_t dim;
    uint32_t layer;
};
static_assert(sizeof(struct dense_args_ple_gate) == 16, "DensePleGateArgs is 16 bytes");

/* Packed affine matvec and GEMM. */
struct dense_args_affine {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tok;
    uint32_t bits;
};
static_assert(sizeof(struct dense_args_affine) == 16, "DenseAffineArgs is 16 bytes");

/* Affine embedding gather. */
struct dense_args_embed_affine {
    uint32_t hidden;
    uint32_t n_tok;
};
static_assert(sizeof(struct dense_args_embed_affine) == 8, "DenseEmbedAffineArgs is 8 bytes");

/* Per-head RMSNorm of the three attention planes and the two-axis rotation of
 * two of them, for Gemma 4's vision tower. */
struct dense_args_vit_rope2d {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    /* Columns in the patch grid: a row's column is `row % grid_w` and its row
     * is `row / grid_w`, both taken within the item (a still is one plane). */
    uint32_t grid_w;
    float eps;
    float rope_base;
};
static_assert(sizeof(struct dense_args_vit_rope2d) == 24, "DenseVitRope2dArgs is 24 bytes");

/* A `3 x 3` average pool over the patch grid, scaled by a constant. */
struct dense_args_vit_pool {
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t hidden;
    float scale;
};
static_assert(sizeof(struct dense_args_vit_pool) == 16, "DenseVitPoolArgs is 16 bytes");

/* `out[i] = clamp(in[i], min, max)` over the first `count` elements.
 *
 * The clamped linear projection the reference wraps every vision weight in: the
 * input is clamped before the multiply and the output after it, and both are
 * this one elementwise kernel run over a different buffer. */
struct dense_args_clamp {
    uint32_t count;
    float min;
    float max;
};
static_assert(sizeof(struct dense_args_clamp) == 12, "ClampArgs is 12 bytes");

/* `shs_metal_args_glu` (`metal/generic/glu.metal`), mirroring `GluArgs`.
 *
 * The explicit pads are the four bytes the C layout leaves before each 64-bit
 * stride, spelled out on the Rust side so `bytemuck::Pod` can hold the struct
 * and spelled out here so the two declarations cannot drift. */
struct dense_args_glu {
    int32_t  ne00;
    int32_t  pad0;
    uint64_t nb01;
    int32_t  ne10;
    int32_t  pad1;
    uint64_t nb11;
    int32_t  ne0;
    int32_t  pad2;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
};
static_assert(sizeof(struct dense_args_glu) == 64, "GluArgs is 64 bytes");

/* `shs_metal_args_norm` (`metal/generic/norm.metal`), mirroring `NormArgs`. */
struct dense_args_norm {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
};
static_assert(sizeof(struct dense_args_norm) == 144, "NormArgs is 144 bytes");

/* --- the q6_K block the per-layer table is packed in ----------------------
 *
 * Low nibbles, the two high bits per weight, one signed scale per 16-element
 * subblock and an fp16 block scale. 210 bytes, matching the loader's
 * `BlockQ6K` and the Metal `shs_dense_block_q6_K`. The per-layer prepare steps
 * a pointer to THIS struct rather than a byte count, so the 210-byte slot
 * stride comes from the layout and cannot drift from it. */
#define DENSE_QK_K 256

struct shs_dense_block_q6_K {
    unsigned char ql[DENSE_QK_K / 2];
    unsigned char qh[DENSE_QK_K / 4];
    signed char   scales[DENSE_QK_K / 16];
    __half        d;
};
static_assert(sizeof(struct shs_dense_block_q6_K) == 210, "a q6_K block is 210 bytes");

/* Elements `16*il .. 16*il+15` of a q6_K block, in element order.
 *
 * The arithmetic of `dequantize_dense_q6_K` in
 * `metal/qwen35/gemv_ext_q5k_q6k.metal`: `il` walks sixteen 16-element groups,
 * `n128` picks the block's 128-element half, the quarter inside that half picks
 * which 32 `ql` bytes and which nibble, and `qh` supplies the top two bits from
 * its own two-bit field. The `-32` bias is folded before the multiply, which is
 * what makes q6_K symmetric. */
__device__ __forceinline__ void dequantize_dense_q6_K(
        const struct shs_dense_block_q6_K *xb, int il, float *reg) {
    const int n128 = il / 8;
    const int il128 = il & 7;
    const int quarter = il128 / 2;
    const int l0 = (il128 & 1) * 16;
    const unsigned ql_base = (unsigned) n128 * 64u;
    const unsigned qh_base = (unsigned) n128 * 32u;
    const unsigned sc_base = (unsigned) n128 * 8u + (unsigned) quarter * 2u
                           + (unsigned) (il128 & 1);
    const float d = __half2float(xb->d) * (float) ((int) xb->scales[sc_base]);

    for (int i = 0; i < 16; ++i) {
        const unsigned l = (unsigned) l0 + (unsigned) i;
        unsigned v;
        if (quarter == 0) {
            v = ((unsigned) xb->ql[ql_base + l] & 0x0Fu)
              | ((((unsigned) xb->qh[qh_base + l] >> 0u) & 3u) << 4u);
        } else if (quarter == 1) {
            v = ((unsigned) xb->ql[ql_base + 32u + l] & 0x0Fu)
              | ((((unsigned) xb->qh[qh_base + l] >> 2u) & 3u) << 4u);
        } else if (quarter == 2) {
            v = ((unsigned) xb->ql[ql_base + l] >> 4u)
              | ((((unsigned) xb->qh[qh_base + l] >> 4u) & 3u) << 4u);
        } else {
            v = ((unsigned) xb->ql[ql_base + 32u + l] >> 4u)
              | ((((unsigned) xb->qh[qh_base + l] >> 6u) & 3u) << 4u);
        }
        reg[i] = d * (float) ((int) v - 32);
    }
}

#endif /* SHISO_DENSE_COMMON_CUH */

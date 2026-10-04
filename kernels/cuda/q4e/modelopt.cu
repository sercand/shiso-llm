/* The `modelopt` codecs of this family's NVIDIA export: the routed stages over
 * an NVFP4 expert bank (the target's) and a block-scaled FP8 one (the draft
 * head's), and the decode of FP8 n-gram rows.
 *
 * Not an upstream unit. Every other file in this directory is a byte copy of
 * the source the family's kernels came from, and these entries are this
 * repository's own - which is why they are a module of their own
 * (`q4e_modelopt`) rather than lines added to `moe.cu` or `ple.cu`. Their
 * argument lists are the Metal set's `q4e/fp8.metal` ones, with the NVFP4
 * bank's two extra arrays where the FP8 bank has its block scales, so the
 * host plans one shape of stage for every codec.
 *
 * # Every decode here survives a flushed subnormal
 *
 * This module builds with the tree's `--use_fast_math`, which flushes
 * subnormals to zero. An `e4m3` scale byte with a zero exponent field IS a
 * subnormal (`m * 2^-9`), and this export's group scales reach them; a decode
 * that formed it as a subnormal `f32` and rescaled would read zero. So the
 * zero-exponent case is computed as a normal `f32` constant times the
 * mantissa, and every other case is assembled directly at its own exponent.
 *
 * # NVFP4, as the rest of the tree reads it
 *
 * `w[r][k] = e2m1(nibble) * (e4m3(scale[r][k / 16]) * global)`, the low nibble
 * of byte `k / 2` holding the even weight. The group scale and the global are
 * multiplied together first, once a group, and a group's sixteen products are
 * summed before that scale is applied - `kernels/cuda/qwen35moe`'s order.
 */

#include <stdint.h>

#define Q4E_NVFP4_GROUP 16u
#define Q4E_NVFP4_ROWS 8u

/* `e4m3` to `f32`, exact and safe under flush-to-zero. */
__device__ __forceinline__ float q4e_mo_e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 0x0Fu;
    const uint32_t m = b & 0x07u;
    float v;
    if ((b & 0x7Fu) == 0x7Fu) {
        v = __int_as_float(0x7fffffff);
    } else if (e == 0u) {
        v = (float) m * 0.001953125f; /* m * 2^-9 */
    } else {
        v = __int_as_float((int) (((e + 120u) << 23) | (m << 20)));
    }
    return (b & 0x80u) ? -v : v;
}

/* `e2m1` to `f32`, exact: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6. */
__device__ __forceinline__ float q4e_mo_e2m1(uint32_t c) {
    const uint32_t e = (c >> 1) & 3u;
    const uint32_t m = c & 1u;
    const float v = e ? __int_as_float((int) (((e + 126u) << 23) | (m << 22)))
                      : (m ? 0.5f : 0.0f);
    return (c & 8u) ? -v : v;
}

__device__ __forceinline__ float q4e_mo_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

/* Eight packed weights against eight activations, from zero, in order. */
__device__ __forceinline__ float q4e_mo_word_dot(uint32_t w, float4 a, float4 b) {
    float s = 0.0f;
    s = fmaf(q4e_mo_e2m1(w & 0xFu), a.x, s);
    s = fmaf(q4e_mo_e2m1((w >> 4) & 0xFu), a.y, s);
    s = fmaf(q4e_mo_e2m1((w >> 8) & 0xFu), a.z, s);
    s = fmaf(q4e_mo_e2m1((w >> 12) & 0xFu), a.w, s);
    s = fmaf(q4e_mo_e2m1((w >> 16) & 0xFu), b.x, s);
    s = fmaf(q4e_mo_e2m1((w >> 20) & 0xFu), b.y, s);
    s = fmaf(q4e_mo_e2m1((w >> 24) & 0xFu), b.z, s);
    s = fmaf(q4e_mo_e2m1((w >> 28) & 0xFu), b.w, s);
    return s;
}

/* One lane's share of the dot of one NVFP4 row with `x`: lane `l` takes
 * groups `l`, `l + 32`, ... The widths this family stores (2560 and 640) are
 * whole groups, and a row starts at a multiple of eight bytes, so a group is
 * one eight-byte load and four sixteen-byte ones. */
__device__ __forceinline__ float q4e_mo_nvfp4_row(
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ s, float global,
        const float *__restrict__ x, uint32_t in_dim, uint32_t lane) {
    float acc = 0.0f;
    const uint32_t groups = in_dim / Q4E_NVFP4_GROUP;
    for (uint32_t g = lane; g < groups; g += 32u) {
        const uint2 word = ((const uint2 *) w)[g];
        const float scale = q4e_mo_e4m3(s[g]) * global;
        const float4 *xg = (const float4 *) (x + (uint64_t) g * Q4E_NVFP4_GROUP);
        const float part = q4e_mo_word_dot(word.x, xg[0], xg[1])
                         + q4e_mo_word_dot(word.y, xg[2], xg[3]);
        acc = fmaf(part, scale, acc);
    }
    return acc;
}

/* Gate and up for one active expert's columns: eight output rows a block, one
 * per warp, the expert's span of the sorted column list swept so its rows stay
 * in cache. `map[e]` is the slot expert `e`'s matrices sit at; a column's
 * activation row is its token's, `col / n_used`.
 *
 * Buffers: out_gate, out_up, gate_w, gate_s, gate_g, up_w, up_s, up_g, x,
 * sorted, bounds, active, map, in_dim, out_dim, n_expert, n_used. Grid
 * `(ceil(out_dim / 8), min(columns, n_expert))`, 256 threads. */
extern "C" __global__ void q4e_nvfp4_experts_gate_up_kernel(
        float *out_gate, float *out_up,
        const uint8_t *gate_w, const uint8_t *gate_s, const float *gate_g,
        const uint8_t *up_w, const uint8_t *up_s, const float *up_g,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    if ((int) blockIdx.y >= active[n_expert]) {
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t o = blockIdx.x * Q4E_NVFP4_ROWS + (threadIdx.x >> 5);
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const uint64_t slot = map[expert];
    const uint64_t row = slot * out_dim + o;
    const uint64_t packed = in_dim / 2u;
    const uint64_t groups = in_dim / Q4E_NVFP4_GROUP;
    const uint8_t *gw = gate_w + row * packed;
    const uint8_t *uw = up_w + row * packed;
    const uint8_t *gs = gate_s + row * groups;
    const uint8_t *us = up_s + row * groups;
    const float gg = gate_g[slot];
    const float ug = up_g[slot];
    for (int c = c0; c < c1; ++c) {
        const uint32_t col = (uint32_t) sorted[c];
        const float *xr = x + (uint64_t) (col / n_used) * in_dim;
        const float g = q4e_mo_warp_sum(q4e_mo_nvfp4_row(gw, gs, gg, xr, in_dim, lane));
        const float u = q4e_mo_warp_sum(q4e_mo_nvfp4_row(uw, us, ug, xr, in_dim, lane));
        if (lane == 0u) {
            out_gate[(uint64_t) col * out_dim + o] = g;
            out_up[(uint64_t) col * out_dim + o] = u;
        }
    }
}

/* The routed down projection over an NVFP4 bank: each column reads its own row
 * of the activated intermediate and writes `[column, out_dim]`.
 *
 * Buffers: out, down_w, down_s, down_g, mid, sorted, bounds, active, map,
 * in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8), min(columns,
 * n_expert))`, 256 threads. */
extern "C" __global__ void q4e_nvfp4_experts_down_kernel(
        float *out, const uint8_t *down_w, const uint8_t *down_s, const float *down_g,
        const float *mid, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    (void) n_used;
    if ((int) blockIdx.y >= active[n_expert]) {
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t o = blockIdx.x * Q4E_NVFP4_ROWS + (threadIdx.x >> 5);
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const uint64_t slot = map[expert];
    const uint64_t row = slot * out_dim + o;
    const uint8_t *dw = down_w + row * (in_dim / 2u);
    const uint8_t *ds = down_s + row * (in_dim / Q4E_NVFP4_GROUP);
    const float dg = down_g[slot];
    for (int c = c0; c < c1; ++c) {
        const uint32_t col = (uint32_t) sorted[c];
        const float v = q4e_mo_warp_sum(
                q4e_mo_nvfp4_row(dw, ds, dg, mid + (uint64_t) col * in_dim, in_dim, lane));
        if (lane == 0u) {
            out[(uint64_t) col * out_dim + o] = v;
        }
    }
}

/* ---------------------------------------------------------------------------
 * The routed stages of a prompt pass, on tensor cores.
 *
 * A prompt routes tens of columns to every expert, and the warp-per-row
 * kernels above re-decode an expert's weights and run a scalar dot for every
 * one of them: measured on a 2,719-token prompt, 86% of the pass. These tile
 * an expert's product `Y[col][o] = W[o] . X[col]` as 64 rows by 64 columns,
 * 32 contraction elements a step, through `mma.sync` m16n8k16 with half
 * inputs and single-precision accumulation.
 *
 * The weights decode to halves EXACTLY: an `e2m1` value times an `e4m3` group
 * scale has at most six significant bits and lands between 2^-10 and 2688,
 * inside half precision's normal range. The per-expert global scale is NOT
 * folded in - at around 1e-4 it would push products into the subnormals - and
 * is applied to the finished single-precision sum instead. The activations are
 * rounded to halves, as the dense half-precision prompt tier rounds them.
 *
 * A column's sum is a fixed sequence of 16-wide steps in contraction order,
 * whatever else shares its tile, so a prompt's rows do not depend on how many
 * columns its chunk routed to the same expert.
 * ------------------------------------------------------------------------ */

#include <cuda_fp16.h>

#define Q4E_MO_BM 64u
#define Q4E_MO_BN 64u
#define Q4E_MO_BK 32u
/* Halves of padding per shared row: keeps the fragment loads of the eight
 * row groups of a warp on distinct banks. */
#define Q4E_MO_PAD 8u
#define Q4E_MO_LD (Q4E_MO_BK + Q4E_MO_PAD)
#define Q4E_MO_THREADS 128u

__device__ __forceinline__ void q4e_mo_mma(float *c, const uint32_t *a, const uint32_t *b) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* One expert's 64-row tile of `W`, every 64-column tile of its columns. `x`
 * is read at row `col / div` - the token for gate and up, the column itself
 * for down - and `out` written `[col, out_dim]`. Rows of `W` start at
 * `row0`; the tile covers rows `row0 .. row0 + 64`. */
__device__ __forceinline__ void q4e_mo_nvfp4_gemm_tile(
        float *__restrict__ out, const uint8_t *__restrict__ w,
        const uint8_t *__restrict__ s, float global, const float *__restrict__ x,
        const int *__restrict__ sorted, int c0, int c1, uint32_t div, uint32_t row0,
        uint32_t in_dim, uint32_t out_dim) {
    __shared__ __align__(16) __half a_tile[Q4E_MO_BM][Q4E_MO_LD];
    __shared__ __align__(16) __half b_tile[Q4E_MO_BN][Q4E_MO_LD];
    __shared__ int cols[Q4E_MO_BN];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t gid = lane >> 2;
    const uint32_t tig = lane & 3u;
    const uint64_t packed = in_dim / 2u;
    const uint64_t groups = in_dim / Q4E_NVFP4_GROUP;

    for (int n_base = c0; n_base < c1; n_base += (int) Q4E_MO_BN) {
        __syncthreads();
        if (tid < Q4E_MO_BN) {
            const int c = n_base + (int) tid;
            cols[tid] = c < c1 ? sorted[c] : -1;
        }
        float acc[Q4E_MO_BN / 8u][4];
#pragma unroll
        for (uint32_t i = 0; i < Q4E_MO_BN / 8u; ++i) {
            acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
        }
        for (uint32_t k0 = 0; k0 < in_dim; k0 += Q4E_MO_BK) {
            __syncthreads();
            /* A: 64 rows x 2 groups, one group a thread, decoded to halves. */
            {
                const uint32_t r = tid >> 1;
                const uint32_t g = (k0 / Q4E_NVFP4_GROUP) + (tid & 1u);
                const uint64_t row = row0 + r;
                __half *dst = &a_tile[r][(tid & 1u) * Q4E_NVFP4_GROUP];
                if (row < out_dim) {
                    const uint2 word = *(const uint2 *) (w + row * packed + (uint64_t) g * 8u);
                    const float scale = q4e_mo_e4m3(s[row * groups + g]);
#pragma unroll
                    for (uint32_t j = 0; j < 8u; ++j) {
                        dst[j] = __float2half_rn(q4e_mo_e2m1((word.x >> (4u * j)) & 0xFu) * scale);
                        dst[8u + j] =
                                __float2half_rn(q4e_mo_e2m1((word.y >> (4u * j)) & 0xFu) * scale);
                    }
                } else {
#pragma unroll
                    for (uint32_t j = 0; j < 16u; ++j) {
                        dst[j] = __float2half_rn(0.0f);
                    }
                }
            }
            /* B: 64 columns x 32 elements, eight a thread pass, as halves. */
#pragma unroll
            for (uint32_t pass = 0; pass < (Q4E_MO_BN * Q4E_MO_BK / 8u) / Q4E_MO_THREADS; ++pass) {
                const uint32_t i = pass * Q4E_MO_THREADS + tid;
                const uint32_t n = i / (Q4E_MO_BK / 8u);
                const uint32_t q = i % (Q4E_MO_BK / 8u);
                __half *dst = &b_tile[n][q * 8u];
                const int col = cols[n];
                if (col >= 0) {
                    const float4 *src = (const float4 *) (x + (uint64_t) ((uint32_t) col / div) * in_dim
                                                          + k0 + q * 8u);
                    const float4 v0 = src[0];
                    const float4 v1 = src[1];
                    dst[0] = __float2half_rn(v0.x);
                    dst[1] = __float2half_rn(v0.y);
                    dst[2] = __float2half_rn(v0.z);
                    dst[3] = __float2half_rn(v0.w);
                    dst[4] = __float2half_rn(v1.x);
                    dst[5] = __float2half_rn(v1.y);
                    dst[6] = __float2half_rn(v1.z);
                    dst[7] = __float2half_rn(v1.w);
                } else {
#pragma unroll
                    for (uint32_t j = 0; j < 8u; ++j) {
                        dst[j] = __float2half_rn(0.0f);
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (uint32_t kk = 0; kk < Q4E_MO_BK; kk += 16u) {
                const uint32_t r = warp * 16u + gid;
                uint32_t a[4];
                a[0] = *(const uint32_t *) &a_tile[r][kk + tig * 2u];
                a[1] = *(const uint32_t *) &a_tile[r + 8u][kk + tig * 2u];
                a[2] = *(const uint32_t *) &a_tile[r][kk + 8u + tig * 2u];
                a[3] = *(const uint32_t *) &a_tile[r + 8u][kk + 8u + tig * 2u];
#pragma unroll
                for (uint32_t nt = 0; nt < Q4E_MO_BN / 8u; ++nt) {
                    uint32_t b[2];
                    b[0] = *(const uint32_t *) &b_tile[nt * 8u + gid][kk + tig * 2u];
                    b[1] = *(const uint32_t *) &b_tile[nt * 8u + gid][kk + 8u + tig * 2u];
                    q4e_mo_mma(acc[nt], a, b);
                }
            }
        }
        /* c0, c1 at (row gid, columns 2 tig, +1); c2, c3 eight rows down. */
#pragma unroll
        for (uint32_t nt = 0; nt < Q4E_MO_BN / 8u; ++nt) {
#pragma unroll
            for (uint32_t h = 0; h < 4u; ++h) {
                const uint32_t n = nt * 8u + tig * 2u + (h & 1u);
                const uint32_t o = row0 + warp * 16u + gid + (h >> 1) * 8u;
                const int col = cols[n];
                if (col >= 0 && o < out_dim) {
                    out[(uint64_t) col * out_dim + o] = acc[nt][h] * global;
                }
            }
        }
    }
}

/* Gate and up of a prompt pass over an NVFP4 bank: the gate's row tiles, then
 * the up's, on the grid's first axis; one active expert a block on its second.
 *
 * Buffers: out_gate, out_up, gate_w, gate_s, gate_g, up_w, up_s, up_g, x,
 * sorted, bounds, active, map, in_dim, out_dim, n_expert, n_used. Grid
 * `(2 * ceil(out_dim / 64), min(columns, n_expert))`, 128 threads. */
extern "C" __global__ void __launch_bounds__(Q4E_MO_THREADS) q4e_nvfp4_experts_gemm_gate_up_kernel(
        float *out_gate, float *out_up,
        const uint8_t *gate_w, const uint8_t *gate_s, const float *gate_g,
        const uint8_t *up_w, const uint8_t *up_s, const float *up_g,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    if ((int) blockIdx.y >= active[n_expert]) {
        return;
    }
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    const uint32_t tiles = (out_dim + Q4E_MO_BM - 1u) / Q4E_MO_BM;
    const bool up = blockIdx.x >= tiles;
    const uint32_t row0 = (blockIdx.x - (up ? tiles : 0u)) * Q4E_MO_BM;
    const uint64_t slot = map[expert];
    const uint64_t first = slot * out_dim;
    const uint8_t *w = (up ? up_w : gate_w) + first * (in_dim / 2u);
    const uint8_t *s = (up ? up_s : gate_s) + first * (in_dim / Q4E_NVFP4_GROUP);
    const float global = (up ? up_g : gate_g)[slot];
    q4e_mo_nvfp4_gemm_tile(up ? out_up : out_gate, w, s, global, x, sorted, c0, c1, n_used,
                           row0, in_dim, out_dim);
}

/* The routed down projection of a prompt pass over an NVFP4 bank.
 *
 * Buffers: out, down_w, down_s, down_g, mid, sorted, bounds, active, map,
 * in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 64), min(columns,
 * n_expert))`, 128 threads. */
extern "C" __global__ void __launch_bounds__(Q4E_MO_THREADS) q4e_nvfp4_experts_gemm_down_kernel(
        float *out, const uint8_t *down_w, const uint8_t *down_s, const float *down_g,
        const float *mid, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    (void) n_used;
    if ((int) blockIdx.y >= active[n_expert]) {
        return;
    }
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    const uint64_t slot = map[expert];
    const uint64_t first = slot * out_dim;
    q4e_mo_nvfp4_gemm_tile(out, down_w + first * (in_dim / 2u),
                           down_s + first * (in_dim / Q4E_NVFP4_GROUP), down_g[slot], mid,
                           sorted, c0, c1, 1u, blockIdx.x * Q4E_MO_BM, in_dim, out_dim);
}

/* A brain float widened exactly. */
__device__ __forceinline__ float q4e_mo_bf16(uint16_t bits) {
    return __int_as_float((int) ((uint32_t) bits << 16));
}

#define Q4E_FP8_BLOCK 128u

/* One lane's share of the dot of one block-scaled FP8 row with `x`: lane `l`
 * reads bytes `4l .. 4l + 3` of every 128-wide stretch, so a warp consumes one
 * scale block's row segment per step - the Metal set's `q4e_fp8_row`. */
__device__ __forceinline__ float q4e_mo_fp8_row(
        const uint8_t *__restrict__ w, const uint16_t *__restrict__ scales,
        const float *__restrict__ x, uint32_t in_dim, uint32_t lane) {
    float acc = 0.0f;
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += Q4E_FP8_BLOCK) {
        const float s = q4e_mo_bf16(scales[k0 / Q4E_FP8_BLOCK]);
        const uint32_t k = k0 + lane * 4u;
        if (k + 3u < in_dim) {
            const uint32_t q = *(const uint32_t *) (w + k);
            const float4 xv = *(const float4 *) (x + k);
            const float part = q4e_mo_e4m3(q & 0xFFu) * xv.x
                             + q4e_mo_e4m3((q >> 8) & 0xFFu) * xv.y
                             + q4e_mo_e4m3((q >> 16) & 0xFFu) * xv.z
                             + q4e_mo_e4m3(q >> 24) * xv.w;
            acc += s * part;
        }
    }
    return acc;
}

/* Gate and up over a block-scaled FP8 bank - the draft head's routed experts.
 *
 * Buffers: out_gate, out_up, gate_w, gate_s, up_w, up_s, x, sorted, bounds,
 * active, map, in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8),
 * min(columns, n_expert))`, 256 threads, a warp per output row. */
extern "C" __global__ void q4e_fp8_experts_gate_up_kernel(
        float *out_gate, float *out_up, const uint8_t *gate_w, const uint16_t *gate_s,
        const uint8_t *up_w, const uint16_t *up_s, const float *x, const int *sorted,
        const int *bounds, const int *active, const uint32_t *map, uint32_t in_dim,
        uint32_t out_dim, uint32_t n_expert, uint32_t n_used) {
    if ((int) blockIdx.y >= active[n_expert]) {
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t o = blockIdx.x * Q4E_NVFP4_ROWS + (threadIdx.x >> 5);
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const uint64_t slot = map[expert];
    const uint32_t blocks = (in_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint32_t block_rows = (out_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint64_t row = slot * out_dim + o;
    const uint64_t srow = (slot * block_rows + o / Q4E_FP8_BLOCK) * blocks;
    const uint8_t *gw = gate_w + row * in_dim;
    const uint8_t *uw = up_w + row * in_dim;
    for (int c = c0; c < c1; ++c) {
        const uint32_t col = (uint32_t) sorted[c];
        const float *xr = x + (uint64_t) (col / n_used) * in_dim;
        const float g = q4e_mo_warp_sum(q4e_mo_fp8_row(gw, gate_s + srow, xr, in_dim, lane));
        const float u = q4e_mo_warp_sum(q4e_mo_fp8_row(uw, up_s + srow, xr, in_dim, lane));
        if (lane == 0u) {
            out_gate[(uint64_t) col * out_dim + o] = g;
            out_up[(uint64_t) col * out_dim + o] = u;
        }
    }
}

/* The routed down projection over a block-scaled FP8 bank.
 *
 * Buffers: out, down_w, down_s, mid, sorted, bounds, active, map, in_dim,
 * out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8), min(columns,
 * n_expert))`, 256 threads. */
extern "C" __global__ void q4e_fp8_experts_down_kernel(
        float *out, const uint8_t *down_w, const uint16_t *down_s, const float *mid,
        const int *sorted, const int *bounds, const int *active, const uint32_t *map,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_expert, uint32_t n_used) {
    (void) n_used;
    if ((int) blockIdx.y >= active[n_expert]) {
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t o = blockIdx.x * Q4E_NVFP4_ROWS + (threadIdx.x >> 5);
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const uint64_t slot = map[expert];
    const uint32_t blocks = (in_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint32_t block_rows = (out_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint8_t *dw = down_w + (slot * out_dim + o) * in_dim;
    const uint64_t srow = (slot * block_rows + o / Q4E_FP8_BLOCK) * blocks;
    for (int c = c0; c < c1; ++c) {
        const uint32_t col = (uint32_t) sorted[c];
        const float v = q4e_mo_warp_sum(
                q4e_mo_fp8_row(dw, down_s + srow, mid + (uint64_t) col * in_dim, in_dim, lane));
        if (lane == 0u) {
            out[(uint64_t) col * out_dim + o] = v;
        }
    }
}

/* Decodes gathered FP8 n-gram rows: `head_dim` E4M3 bytes a row, one scale for
 * the whole table. Row `r` lands at token `r / n_heads`, head slot
 * `r % n_heads`, head-slowest - `q4e_ple_dequant_kernel`'s placement.
 *
 * Buffers: out, rows, head_dim, n_heads, row_bytes, n_rows, scale (one float
 * on the device). Grid `n_rows`, a block of `head_dim` threads. */
extern "C" __global__ void q4e_ple_dequant_fp8_kernel(
        float *out, const uint8_t *rows, uint32_t head_dim, uint32_t n_heads,
        uint32_t row_bytes, uint32_t n_rows, const float *scale) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }
    const uint8_t *src = rows + (uint64_t) r * row_bytes;
    const uint32_t t = r / n_heads;
    const uint32_t h = r % n_heads;
    float *dst = out + (uint64_t) t * n_heads * head_dim + (uint64_t) h * head_dim;
    const float s = scale[0];
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        dst[i] = q4e_mo_e4m3(src[i]) * s;
    }
}

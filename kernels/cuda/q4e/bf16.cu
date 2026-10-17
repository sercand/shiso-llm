/* Dense projections over a brain-float weight, read as stored: row-major
 * `[out, in]` bf16, activations `[rows, in]` f32, output `[rows, out]` f32.
 *
 * Not an upstream unit, for the reason `modelopt.cu` is not. The brain-float
 * kernel in `matvec.cu` (`q4e_matmul_bf16_rows_kernel`) is the ported one and
 * stays as it is; this unit is what a device with tensor cores takes instead,
 * so that a checkpoint's brain-float matrices - the stream mixes, the routers,
 * the n-gram projections, the output head - are served in their own format
 * rather than widened to half precision at load.
 *
 * Seven entries:
 *
 * - `q4e_bf16_gemv_kernel`, one activation row (a decode step);
 * - `q4e_bf16_rows_kernel`, two to eight (a speculative verify);
 * - `q4e_bf16_rows_wide_kernel`, nine to thirty-two (a batched verify);
 * - `q4e_bf16_gemm_kernel`, a prompt chunk of any width, on tensor cores, its
 *   activations rounded to brain floats;
 * - `q4e_bf16_gemm_split_kernel`, the same with each activation carried as
 *   two brain floats (`hi + lo`), for the narrow projections whose sums must
 *   not see the rounding (the router above all);
 * - `q4e_bf16_gemm_pair_kernel`, the plain prompt kernel over two weights of
 *   one input stacked in one grid;
 * - `q4e_bf16_gemm_collapse_kernel`, the stream mix's expansion with the
 *   collapse of the gated streams in its epilogue.
 *
 * # The step kernels sum a row one way
 *
 * Argument list `out, w, x, in_dim, out_dim, splits, rows`. An output row's
 * contraction is cut into `splits` (1, 2, 4 or 8) slices, one warp each,
 * inside one block. Lane `l` owns the eight contiguous weights
 * `[i + 8 l, i + 8 l + 8)` of each 256-wide step `i` of its slice, one fused
 * multiply-add per weight per activation row in contraction order; then the
 * decreasing-offset shuffle fold, then the slices added in slice order. How
 * many output rows a warp owns (`R`) and how many activation rows share the
 * launch (`NT`) change which loads are shared, never the order of any
 * accumulator's adds - so a verify row is the single-row step's row, bit for
 * bit. The slice count is a function of the shape alone (the engine's
 * `bf16_split`), and folding inside the block leaves no second launch.
 *
 * Measured on a GB10 (kernel bench, wall clock, one row): 320 x 10240
 * 30.2 us (218 GB/s; 33.5 with the split across blocks and a second launch
 * folding it), 512 x 2560 14.2 us, 10240 x 2560 220 us (238 GB/s), 248320 x
 * 2560 5.30 ms (240 GB/s). The twelve-bit packed half kernel that served
 * these weights while they were re-encoded at load reads a quarter fewer
 * bytes and ran at 187, 130 and 230 GB/s on the first three.
 *
 * # The prompt kernels hold their rows to every pass width
 *
 * A 128 x 128 tile of `out[col][o]`, 32 contraction elements a step, through
 * `mma.sync` m16n8k16 with brain-float inputs and single-precision
 * accumulation. The weight tile arrives by `cp.async` straight into its
 * swizzled shared-memory layout, three steps deep; the activations arrive raw
 * (f32) beside it and the thread that copied a piece rounds it into a
 * swizzled brain-float tile. Each output's sum reads only its own activation
 * row, in an order fixed by the shape, so a prompt's rows do not depend on
 * where the prompt was cut or what shared its chunk.
 *
 * Measured on a GB10 (kernel bench, wall clock) at 2048 rows: 10240 x 2560
 * 2.19 ms (47 TFLOP/s), 12288 x 2560 2.67 ms (48), 2560 x 2560 0.61 ms (44);
 * the ported kernel's sixteen-row passes took 27.1 ms at 320 x 10240, where
 * this takes 0.46 ms (bound by reading the activations, 200 GB/s).
 */

#include <stdint.h>
#include <cuda_bf16.h>

#include "shiso_dev_inc/common.cuh"

/* ---------------------------------------------------------------------------
 * The step kernels.
 * ------------------------------------------------------------------------ */

/* Warps a step-kernel block runs. */
#define Q4E_BF_WARPS 8u
/* Contraction elements a warp covers per step: eight a lane. */
#define Q4E_BF_STEP 256u

/* Eight brain floats (sixteen bytes) as eight floats. */
__device__ __forceinline__ void q4e_bf_widen8(uint4 raw, float *v) {
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        v[2 * j] = __uint_as_float(w[j] << 16);
        v[2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
    }
}

/* One block's share of a step pass: `R` output rows a warp, `S` warps an
 * output row (the slices of its contraction), `NT` activation rows.
 *
 * Warp `w` takes slice `w % S` of row group `w / S`: the `R` rows from
 * `(blockIdx.x * (WARPS / S) + w / S) * R`. Its lanes walk the slice
 * `[s chunk, (s + 1) chunk)` 256 elements a step, lane `l` the eight from
 * `8 l`; several steps' weights are loaded before the first is consumed,
 * which keeps the loads in flight without touching the order of the adds. Each slice is folded across its lanes by the
 * decreasing-offset shuffle, and the `S` slices of an output are added in
 * slice order through shared memory. Rows past `out_dim` read the last row
 * and store nothing. */
template <int NT, int R>
__device__ __forceinline__ void q4e_bf_step(
        float *__restrict__ out, const __nv_bfloat16 *__restrict__ w, const float *__restrict__ x,
        uint32_t in_dim, uint32_t out_dim, uint32_t splits, float *__restrict__ part,
        float *__restrict__ out_b = nullptr, const __nv_bfloat16 *__restrict__ w_b = nullptr,
        uint32_t out_dim_b = 0u) {
    /* A pair (`out_dim_b != 0`) stacks `w_b`'s rows under `w`'s, as the
     * prompt GEMM's pair does: each row's sum is the same either way. */
    const uint32_t out_a = out_dim;
    const uint32_t out_all = out_dim + out_dim_b;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t slice = warp % splits;
    const uint32_t row0 = (blockIdx.x * (Q4E_BF_WARPS / splits) + warp / splits) * (uint32_t) R;
    /* A slice is whole eight-element loads, so a lane's load never straddles
     * two slices. */
    const uint32_t chunk = ((in_dim + splits - 1u) / splits + 7u) & ~7u;
    const uint32_t i0 = min(slice * chunk, in_dim);
    const uint32_t i1 = min(i0 + chunk, in_dim);
    const __nv_bfloat16 *wr[R];
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const uint32_t row = min(row0 + (uint32_t) r, out_all - 1u);
        wr[r] = row < out_a ? w + (uint64_t) row * in_dim : w_b + (uint64_t) (row - out_a) * in_dim;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            acc[r][t] = 0.0f;
        }
    }
    /* `D` steps' weights are loaded before the first of them is consumed:
     * the slices are short (five steps at the stream mix down), so a lane
     * needs every load of its slice in flight at once, not one ahead. */
    constexpr int D = R == 1 ? 4 : (R == 2 ? 2 : 1);
    for (uint32_t i = i0 + lane * 8u; i < i1; i += D * Q4E_BF_STEP) {
        uint4 raw[D][R];
#pragma unroll
        for (int d = 0; d < D; d++) {
            const uint32_t k = i + (uint32_t) d * Q4E_BF_STEP;
#pragma unroll
            for (int r = 0; r < R; r++) {
                raw[d][r] = k < i1 ? __ldg(reinterpret_cast<const uint4 *>(wr[r] + k))
                                   : make_uint4(0u, 0u, 0u, 0u);
            }
        }
#pragma unroll
        for (int d = 0; d < D; d++) {
            const uint32_t k = i + (uint32_t) d * Q4E_BF_STEP;
            if (k >= i1) {
                break;
            }
            float wv[R][8];
#pragma unroll
            for (int r = 0; r < R; r++) {
                q4e_bf_widen8(raw[d][r], wv[r]);
            }
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const float4 *xp = reinterpret_cast<const float4 *>(x + (uint64_t) t * in_dim + k);
                const float4 xa = __ldg(xp);
                const float4 xb = __ldg(xp + 1);
                const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int r = 0; r < R; r++) {
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        acc[r][t] = fmaf(wv[r][j], xv[j], acc[r][t]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float v = warp_sum_f32(acc[r][t]);
            if (lane == 0u) {
                part[(warp * R + r) * NT + t] = v;
            }
        }
    }
    __syncthreads();
    /* The first warp of each row group adds its group's slices, in order. */
    if (slice != 0u) {
        return;
    }
    for (uint32_t e = lane; e < (uint32_t) (R * NT); e += 32u) {
        const uint32_t r = e / NT;
        const uint32_t t = e % NT;
        float v = part[(warp * R + r) * NT + t];
        for (uint32_t s = 1u; s < splits; s++) {
            v += part[((warp + s) * R + r) * NT + t];
        }
        if (row0 + r < out_a) {
            out[(uint64_t) t * out_a + row0 + r] = v;
        } else if (row0 + r < out_all) {
            out_b[(uint64_t) t * out_dim_b + (row0 + r - out_a)] = v;
        }
    }
}

/* One activation row, one output row a warp. Grid
 * `ceil(out_dim / (8 / splits))`, 256 threads. */
extern "C" __global__ void __launch_bounds__(Q4E_BF_WARPS * 32u) q4e_bf16_gemv_kernel(
        float *out, const __nv_bfloat16 *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t splits, uint32_t rows) {
    __shared__ float part[Q4E_BF_WARPS];
    if (rows == 1u) {
        q4e_bf_step<1, 1>(out, w, x, in_dim, out_dim, splits, part);
    }
}

/* Two to eight activation rows, two output rows a warp. Grid
 * `ceil(out_dim / (16 / splits))`, 256 threads; any other row count writes
 * nothing. */
extern "C" __global__ void __launch_bounds__(Q4E_BF_WARPS * 32u) q4e_bf16_rows_kernel(
        float *out, const __nv_bfloat16 *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t splits, uint32_t rows) {
    __shared__ float part[Q4E_BF_WARPS * 2u * 8u];
    switch (rows) {
    case 2u: q4e_bf_step<2, 2>(out, w, x, in_dim, out_dim, splits, part); break;
    case 3u: q4e_bf_step<3, 2>(out, w, x, in_dim, out_dim, splits, part); break;
    case 4u: q4e_bf_step<4, 2>(out, w, x, in_dim, out_dim, splits, part); break;
    case 5u: q4e_bf_step<5, 2>(out, w, x, in_dim, out_dim, splits, part); break;
    case 6u: q4e_bf_step<6, 2>(out, w, x, in_dim, out_dim, splits, part); break;
    case 7u: q4e_bf_step<7, 2>(out, w, x, in_dim, out_dim, splits, part); break;
    case 8u: q4e_bf_step<8, 2>(out, w, x, in_dim, out_dim, splits, part); break;
    default: break;
    }
}

/* The step kernels over two weights of one input stacked in one grid:
 * `w_a`'s `out_dim_a` rows into `out_a`, then `w_b`'s into `out_b`, each row
 * the bits its own launch writes when both share the slice count (the
 * stream mix's contraction and injection logits do: eight). One activation
 * row, or (`q4e_bf16_rows_pair_kernel`) two to eight. Arguments
 * `out_a, w_a, out_dim_a, out_b, w_b, out_dim_b, x, in_dim, splits, rows`;
 * grids as the single-weight kernels' over `out_dim_a + out_dim_b` rows. */
extern "C" __global__ void __launch_bounds__(Q4E_BF_WARPS * 32u) q4e_bf16_gemv_pair_kernel(
        float *out_a, const __nv_bfloat16 *w_a, uint32_t out_dim_a,
        float *out_b, const __nv_bfloat16 *w_b, uint32_t out_dim_b,
        const float *x, uint32_t in_dim, uint32_t splits, uint32_t rows) {
    __shared__ float part[Q4E_BF_WARPS];
    if (rows == 1u) {
        q4e_bf_step<1, 1>(out_a, w_a, x, in_dim, out_dim_a, splits, part, out_b, w_b, out_dim_b);
    }
}

#define Q4E_BF_PAIR(n)                                                                     \
    case n##u:                                                                             \
        q4e_bf_step<n, 2>(out_a, w_a, x, in_dim, out_dim_a, splits, part, out_b, w_b,     \
                          out_dim_b);                                                      \
        break;
extern "C" __global__ void __launch_bounds__(Q4E_BF_WARPS * 32u) q4e_bf16_rows_pair_kernel(
        float *out_a, const __nv_bfloat16 *w_a, uint32_t out_dim_a,
        float *out_b, const __nv_bfloat16 *w_b, uint32_t out_dim_b,
        const float *x, uint32_t in_dim, uint32_t splits, uint32_t rows) {
    __shared__ float part[Q4E_BF_WARPS * 2u * 8u];
    switch (rows) {
    Q4E_BF_PAIR(2) Q4E_BF_PAIR(3) Q4E_BF_PAIR(4) Q4E_BF_PAIR(5)
    Q4E_BF_PAIR(6) Q4E_BF_PAIR(7) Q4E_BF_PAIR(8)
    default: break;
    }
}

/* ---------------------------------------------------------------------------
 * The stream mix's activation, expansion and collapse as one step launch.
 * ------------------------------------------------------------------------ */

/* The low-rank width the fused step is written for, the streams, and the
 * block-input elements one block covers (eight warps of four weight rows,
 * two warps a stream). */
#define Q4E_BF_MIX_RANK 320u
#define Q4E_BF_MIX_HC 4u
#define Q4E_BF_MIX_E 8u

/* `out[t][e] = (1 / n_hc) sum_h xn[t][h][e] * sigmoid(up[t][h][e])` with
 * `up[t] = w_up silu(x[t] / n_hc)`, for one to eight activation rows - the
 * activation (`q4e_scale_silu_kernel`'s expression, once a block into shared
 * memory), the expansion and the collapse (`q4e_hc_collapse_kernel`'s
 * expressions, in its stream order) of a decode step or verify in one launch.
 *
 * The expansion's rows are only 320 wide (640 bytes), which the general step
 * kernel reads a lane eight weights at a time 256 at a time - a second step
 * of eight live lanes, and one 640-byte read in flight a warp. Here a warp
 * owns four consecutive rows, 2560 contiguous bytes, as 160 sixteen-byte
 * chunks: lane `l` loads chunks `l, l + 32, ..., l + 128`, all five at once.
 * Chunk `c` is row `c / 40`, weights `8 (c % 40) ..`; its eight products are
 * a fused multiply-add chain from zero, added to the lane's sum for that row,
 * chunks in order; each row's lane sums fold by the decreasing-offset
 * shuffle. That order is a function of the shape alone and the same at every
 * row count, so a verify row is the single step's row, bit for bit.
 *
 * Warp `w` covers stream `w / 2`, elements `e0 + 4 (w % 2) ..+ 4`, so the
 * block holds every stream of its eight elements and folds the collapse
 * itself. Buffers: out `[rows, n_embd]`, w_up `[n_hc * n_embd, 320]` bf16,
 * x `[rows, 320]` (the low-rank rows before the activation), xn
 * `[rows, n_hc * n_embd]`. Grid `n_embd / 8`, 256 threads. */
template <int NT>
__device__ __forceinline__ void q4e_bf_mix_step(
        float *__restrict__ out, const __nv_bfloat16 *__restrict__ w_up,
        const float *__restrict__ x, const float *__restrict__ xn, float inv_hc,
        uint32_t n_embd, float *__restrict__ xs, float *__restrict__ gate) {
    constexpr uint32_t K = Q4E_BF_MIX_RANK;
    constexpr uint32_t CPR = K / 8u;              /* chunks a row: 40 */
    constexpr uint32_t PER = 4u * CPR / 32u;      /* chunks a lane: 5 */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t e0 = blockIdx.x * Q4E_BF_MIX_E;
    const uint32_t h = warp >> 1;
    const uint32_t el0 = (warp & 1u) * 4u;
    const __nv_bfloat16 *wr = w_up + ((uint64_t) h * n_embd + e0 + el0) * K;
    uint4 raw[PER];
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) {
        raw[j] = __ldg(reinterpret_cast<const uint4 *>(wr) + lane + 32u * j);
    }
    for (uint32_t i = threadIdx.x; i < NT * K; i += Q4E_BF_WARPS * 32u) {
        const float v = x[i] * inv_hc;
        xs[i] = v / (1.0f + __expf(-v));
    }
    __syncthreads();
    float acc[4][NT];
#pragma unroll
    for (uint32_t r = 0; r < 4u; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            acc[r][t] = 0.0f;
        }
    }
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) {
        const uint32_t c = lane + 32u * j;
        const uint32_t row = c / CPR;
        const uint32_t k = (c % CPR) * 8u;
        /* Chunks `32 j ..` span at most two rows. */
        const uint32_t lo = (32u * j) / CPR;
        float wv[8];
        q4e_bf_widen8(raw[j], wv);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float4 xa = *reinterpret_cast<const float4 *>(xs + t * K + k);
            const float4 xb = *reinterpret_cast<const float4 *>(xs + t * K + k + 4u);
            const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            float p = 0.0f;
#pragma unroll
            for (int q = 0; q < 8; q++) {
                p = fmaf(wv[q], xv[q], p);
            }
#pragma unroll
            for (uint32_t r = 0; r < 4u; r++) {
                if (r == lo || r == lo + 1u) {
                    acc[r][t] = row == r ? acc[r][t] + p : acc[r][t];
                }
            }
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < 4u; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float u = warp_sum_f32(acc[r][t]);
            if (lane == 0u) {
                gate[(h * Q4E_BF_MIX_E + el0 + r) * NT + t] = 1.0f / (1.0f + __expf(-u));
            }
        }
    }
    __syncthreads();
    /* The collapse kernel's fold: the streams in order. */
    if (threadIdx.x < Q4E_BF_MIX_E * (uint32_t) NT) {
        const uint32_t el = threadIdx.x % Q4E_BF_MIX_E;
        const uint32_t t = threadIdx.x / Q4E_BF_MIX_E;
        const float inv = 1.0f / (float) Q4E_BF_MIX_HC;
        const float *xn_t = xn + (uint64_t) t * Q4E_BF_MIX_HC * n_embd + e0 + el;
        float sum = 0.0f;
#pragma unroll
        for (uint32_t hh = 0; hh < Q4E_BF_MIX_HC; hh++) {
            sum += xn_t[(uint64_t) hh * n_embd] * gate[(hh * Q4E_BF_MIX_E + el) * NT + t];
        }
        out[(uint64_t) t * n_embd + e0 + el] = sum * inv;
    }
}

#define Q4E_BF_MIX(n) \
    case n##u: q4e_bf_mix_step<n>(out, w_up, x, xn, inv_hc, n_embd, xs, gate); break;
/* Arguments `out, w_up, x, xn, inv_hc, in_dim, n_embd, n_hc, rows`; `in_dim`
 * 320, `n_hc` 4 and `rows` 1 to 8, any other row count writes nothing. */
extern "C" __global__ void __launch_bounds__(Q4E_BF_WARPS * 32u) q4e_bf16_mix_step_kernel(
        float *out, const __nv_bfloat16 *w_up, const float *x, const float *xn, float inv_hc,
        uint32_t in_dim, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    (void) in_dim;
    (void) n_hc;
    __shared__ __align__(16) float xs[8u * Q4E_BF_MIX_RANK];
    __shared__ float gate[Q4E_BF_MIX_HC * Q4E_BF_MIX_E * 8u];
    switch (rows) {
    Q4E_BF_MIX(1) Q4E_BF_MIX(2) Q4E_BF_MIX(3) Q4E_BF_MIX(4)
    Q4E_BF_MIX(5) Q4E_BF_MIX(6) Q4E_BF_MIX(7) Q4E_BF_MIX(8)
    default: break;
    }
}

/* Nine to thirty-two activation rows, four output rows a warp: past eight
 * rows the activations each warp re-reads for its rows outweigh the weight,
 * and four rows a warp halve them. Grid `ceil(out_dim / (32 / splits))`, 256
 * threads; any other row count writes nothing. */
#define Q4E_BF_WIDE(n) case n##u: q4e_bf_step<n, 4>(out, w, x, in_dim, out_dim, splits, part); break;
extern "C" __global__ void __launch_bounds__(Q4E_BF_WARPS * 32u, 1) q4e_bf16_rows_wide_kernel(
        float *out, const __nv_bfloat16 *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t splits, uint32_t rows) {
    __shared__ float part[Q4E_BF_WARPS * 4u * 32u];
    switch (rows) {
    Q4E_BF_WIDE(9) Q4E_BF_WIDE(10) Q4E_BF_WIDE(11) Q4E_BF_WIDE(12)
    Q4E_BF_WIDE(13) Q4E_BF_WIDE(14) Q4E_BF_WIDE(15) Q4E_BF_WIDE(16)
    Q4E_BF_WIDE(17) Q4E_BF_WIDE(18) Q4E_BF_WIDE(19) Q4E_BF_WIDE(20)
    Q4E_BF_WIDE(21) Q4E_BF_WIDE(22) Q4E_BF_WIDE(23) Q4E_BF_WIDE(24)
    Q4E_BF_WIDE(25) Q4E_BF_WIDE(26) Q4E_BF_WIDE(27) Q4E_BF_WIDE(28)
    Q4E_BF_WIDE(29) Q4E_BF_WIDE(30) Q4E_BF_WIDE(31) Q4E_BF_WIDE(32)
    default: break;
    }
}
#undef Q4E_BF_WIDE

/* ---------------------------------------------------------------------------
 * The prompt GEMM.
 * ------------------------------------------------------------------------ */

/* A 128 x 128 tile of `out[col][o]`, sixteen warps of 32 x 32. */
#define Q4E_BF_BM 128u
#define Q4E_BF_BN 128u
#define Q4E_BF_BK 32u
#define Q4E_BF_GEMM_THREADS 512u
/* Output tiles a raster group spans; see the kernel. */
#define Q4E_BF_GROUP_TILES 16u
#define Q4E_BF_W_STEP (Q4E_BF_BM * Q4E_BF_BK * 2u)  /* weight bytes a step */
#define Q4E_BF_X_RAW (Q4E_BF_BN * Q4E_BF_BK * 4u)   /* activation bytes a step */
#define Q4E_BF_X_TILE (Q4E_BF_BN * Q4E_BF_BK * 2u)  /* bytes of one rounded tile */
/* Contraction steps in flight: three for the plain kernel; two for the split
 * one, whose rounded tiles are twice as many. */
#define Q4E_BF_STAGES(SPLIT) ((SPLIT) ? 2u : 3u)
/* Dynamic shared memory: the ring of weight and raw activation steps, and two
 * rounded activation tiles (four for the split kernel: hi and lo). */
#define Q4E_BF_GEMM_SMEM(SPLIT) \
    (Q4E_BF_STAGES(SPLIT) * (Q4E_BF_W_STEP + Q4E_BF_X_RAW) + ((SPLIT) ? 4u : 2u) * Q4E_BF_X_TILE)

__device__ __forceinline__ void q4e_bf_cp16(void *smem, const void *gmem) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void q4e_bf_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void q4e_bf_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void q4e_bf_mma(float *c, const uint32_t *a, const uint32_t *b) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void q4e_bf_ldsm4(uint32_t *r, const void *p) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

/* Two activations rounded to a pair of brain floats, the first low. */
__device__ __forceinline__ uint32_t q4e_bf_pack(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}

/* What rounding left of two activations, as a pair of brain floats. */
__device__ __forceinline__ uint32_t q4e_bf_rest(float a, float b, uint32_t hi) {
    const float ha = __uint_as_float(hi << 16);
    const float hb = __uint_as_float(hi & 0xFFFF0000u);
    return q4e_bf_pack(__fsub_rn(a, ha), __fsub_rn(b, hb));
}

/* Byte offset of sixteen-byte chunk `c` (0..3) of row `r` in a tile of
 * 64-byte rows, swizzled so the eight rows one ldmatrix phase reads land on
 * eight distinct bank groups without padding. */
__device__ __forceinline__ uint32_t q4e_bf_off(uint32_t r, uint32_t c) {
    return r * 64u + ((c ^ ((r >> 1) & 3u)) << 4);
}

/* `out[col][o] = sum_k w[o][k] x[col][k]` for a prompt chunk of `rows`
 * columns.
 *
 * Each step covers 32 contraction elements. The weight's step arrives by
 * `cp.async` directly into its swizzled ring slot, from which the warps read
 * it with ldmatrix; the activations' step arrives raw, and the thread that
 * copied a piece rounds it into a swizzled brain-float tile (two, hi and lo,
 * for `SPLIT`), so only its own wait covers the copy and one barrier a step
 * publishes everything. Each warp accumulates its 32 x 32 block with two
 * m16n8k16 steps (four for `SPLIT`: the hi product, then the lo one) - the
 * same instructions on the same operands in the same order whichever columns
 * share the tile.
 *
 * Tiles run in raster groups of sixteen output tiles, every column tile of a
 * group before the next group, so the blocks resident at once share a few
 * weight tiles and a few activation tiles.
 *
 * Buffers: out `[rows, out_dim]` f32, w `[out_dim, in_dim]` bf16, x
 * `[rows, in_dim]` f32. Grid `(ceil(out_dim / 128), ceil(rows / 128))`, 512
 * threads, `Q4E_BF_GEMM_SMEM(SPLIT)` bytes of dynamic shared memory. `in_dim`
 * a multiple of 32, `w` and `x` sixteen-byte aligned. */
template <bool SPLIT, bool NORM = false>
__device__ __forceinline__ void q4e_bf_gemm(
        float *__restrict__ out, const __nv_bfloat16 *__restrict__ w, const float *__restrict__ x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows,
        float *__restrict__ out_b = nullptr, const __nv_bfloat16 *__restrict__ w_b = nullptr,
        uint32_t out_dim_b = 0u, const float *__restrict__ scale = nullptr,
        const float *__restrict__ nw = nullptr, uint32_t n_embd = 0u) {
    /* With `NORM`, `x` is the residual streams `[rows, n_hc * n_embd]` and
     * each activation is normed as it is read - `x * scale[col][h] * nw[k]`,
     * the expression `q4e_hc_combine_norm_kernel` stores - before the
     * rounding: the operands, and so the sums, of the plain kernel over the
     * stored normed copy. */
    /* A pair (`out_dim_b != 0`) stacks a second weight's rows under the
     * first's - rows `out_dim ..` of the grid are `w_b`'s, written to `out_b`
     * at its own stride - so two projections of one input share the
     * activation reads. Each output's sum is the same instructions on the same
     * operands either way. */
    const uint32_t out_a = out_dim;
    const uint32_t out_all = out_dim + out_dim_b;
    constexpr uint32_t STAGES = Q4E_BF_STAGES(SPLIT);
    constexpr uint32_t TILES = SPLIT ? 2u : 1u;
    extern __shared__ __align__(16) uint8_t q4e_bf_dyn[];
    uint8_t *w_ring = q4e_bf_dyn;                              /* [STAGES][128][32] bf16 */
    uint8_t *x_raw = w_ring + STAGES * Q4E_BF_W_STEP;          /* [STAGES][128][32] f32 */
    uint8_t *x16 = x_raw + STAGES * Q4E_BF_X_RAW;              /* [2][TILES] bf16 tiles */

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t gid = lane >> 2;
    const uint32_t tig = lane & 3u;
    const uint32_t wm = warp & 3u; /* 32 output rows */
    const uint32_t wn = warp >> 2; /* 32 columns */

    const uint32_t out_tiles = gridDim.x;
    const uint32_t col_tiles = gridDim.y;
    const uint32_t lin = blockIdx.y * out_tiles + blockIdx.x;
    const uint32_t per_group = Q4E_BF_GROUP_TILES * col_tiles;
    const uint32_t first = (lin / per_group) * Q4E_BF_GROUP_TILES;
    const uint32_t gsize = min(out_tiles - first, Q4E_BF_GROUP_TILES);
    const uint32_t in_group = lin % per_group;
    const uint32_t row0 = (first + in_group % gsize) * Q4E_BF_BM;
    const uint32_t col0 = (in_group / gsize) * Q4E_BF_BN;
    const uint32_t nk = in_dim / Q4E_BF_BK;

    /* Weights: every thread owns sixteen bytes (eight weights) of row
     * `tid / 4`, chunk `tid % 4`. */
    const uint32_t ar = tid >> 2;
    const uint32_t ac = tid & 3u;
    const bool a_live = row0 + ar < out_all;
    const uint32_t a_row = min(row0 + ar, out_all - 1u);
    const __nv_bfloat16 *a_src = (a_row < out_a ? w + (uint64_t) a_row * in_dim
                                                : w_b + (uint64_t) (a_row - out_a) * in_dim)
                               + ac * 8u;
    const uint32_t a_dst = q4e_bf_off(ar, ac);
    /* Activations: every thread owns four of column `tid / 8` and four of
     * column `tid / 8 + 64`. */
    const uint32_t bn = tid >> 3;
    const uint32_t bq = tid & 7u;
    const bool b_live0 = col0 + bn < rows;
    const bool b_live1 = col0 + bn + 64u < rows;
    const float *b_src0 = x + (uint64_t) min(col0 + bn, rows - 1u) * in_dim + bq * 4u;
    const float *b_src1 = x + (uint64_t) min(col0 + bn + 64u, rows - 1u) * in_dim + bq * 4u;

    /* Rows and columns past the matrix stage zeros once, in every slot. */
#pragma unroll
    for (uint32_t s = 0; s < STAGES; s++) {
        if (!a_live) {
            *reinterpret_cast<uint4 *>(w_ring + s * Q4E_BF_W_STEP + a_dst) = make_uint4(0u, 0u, 0u, 0u);
        }
        if (!b_live0) {
            *reinterpret_cast<float4 *>(x_raw + s * Q4E_BF_X_RAW + tid * 16u) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (!b_live1) {
            *reinterpret_cast<float4 *>(x_raw + s * Q4E_BF_X_RAW + (tid + Q4E_BF_GEMM_THREADS) * 16u) =
                    make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    auto issue = [&](uint32_t kt) {
        const uint32_t s = kt % STAGES;
        const uint32_t k0 = kt * Q4E_BF_BK;
        if (a_live) {
            q4e_bf_cp16(w_ring + s * Q4E_BF_W_STEP + a_dst, a_src + k0);
        }
        if (b_live0) {
            q4e_bf_cp16(x_raw + s * Q4E_BF_X_RAW + tid * 16u, b_src0 + k0);
        }
        if (b_live1) {
            q4e_bf_cp16(x_raw + s * Q4E_BF_X_RAW + (tid + Q4E_BF_GEMM_THREADS) * 16u, b_src1 + k0);
        }
    };

    float acc[2][4][4];
#pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        }
    }

#pragma unroll
    for (uint32_t s = 0; s + 1u < STAGES; s++) {
        if (s < nk) {
            issue(s);
        }
        q4e_bf_cp_commit();
    }
    for (uint32_t kt = 0; kt < nk; kt++) {
        const uint32_t s = kt % STAGES;
        const uint32_t ab = kt & 1u;
        q4e_bf_cp_wait<STAGES - 2u>();
        /* This thread's activations of step `kt`, rounded into tile `ab`.
         * Its last readers were step `kt - 2`'s, which every warp finished
         * before the previous barrier. */
        {
            uint8_t *t = x16 + ab * TILES * Q4E_BF_X_TILE;
            float4 v0 = *reinterpret_cast<const float4 *>(x_raw + s * Q4E_BF_X_RAW + tid * 16u);
            float4 v1 = *reinterpret_cast<const float4 *>(
                    x_raw + s * Q4E_BF_X_RAW + (tid + Q4E_BF_GEMM_THREADS) * 16u);
            if (NORM) {
                const uint32_t k = kt * Q4E_BF_BK + bq * 4u;
                const uint32_t n_hc = in_dim / n_embd;
                const uint32_t h = k / n_embd;
                const float4 wk = *reinterpret_cast<const float4 *>(nw + k);
                const float s0 = scale[(uint64_t) min(col0 + bn, rows - 1u) * n_hc + h];
                const float s1 = scale[(uint64_t) min(col0 + bn + 64u, rows - 1u) * n_hc + h];
                v0 = make_float4(v0.x * s0 * wk.x, v0.y * s0 * wk.y, v0.z * s0 * wk.z, v0.w * s0 * wk.w);
                v1 = make_float4(v1.x * s1 * wk.x, v1.y * s1 * wk.y, v1.z * s1 * wk.z, v1.w * s1 * wk.w);
            }
            /* Four brain floats are half a chunk: chunk `bq / 2`, its
             * `bq % 2` half. */
            const uint32_t o0 = q4e_bf_off(bn, bq >> 1) + (bq & 1u) * 8u;
            const uint32_t o1 = q4e_bf_off(bn + 64u, bq >> 1) + (bq & 1u) * 8u;
            const uint2 h0 = make_uint2(q4e_bf_pack(v0.x, v0.y), q4e_bf_pack(v0.z, v0.w));
            const uint2 h1 = make_uint2(q4e_bf_pack(v1.x, v1.y), q4e_bf_pack(v1.z, v1.w));
            *reinterpret_cast<uint2 *>(t + o0) = h0;
            *reinterpret_cast<uint2 *>(t + o1) = h1;
            if (SPLIT) {
                uint8_t *l = t + Q4E_BF_X_TILE;
                *reinterpret_cast<uint2 *>(l + o0) =
                        make_uint2(q4e_bf_rest(v0.x, v0.y, h0.x), q4e_bf_rest(v0.z, v0.w, h0.y));
                *reinterpret_cast<uint2 *>(l + o1) =
                        make_uint2(q4e_bf_rest(v1.x, v1.y, h1.x), q4e_bf_rest(v1.z, v1.w, h1.y));
            }
        }
        __syncthreads();
        /* The slot step `kt + STAGES - 1` lands in held step `kt - 1`, whose
         * weights every warp finished multiplying before this barrier. */
        if (kt + STAGES - 1u < nk) {
            issue(kt + STAGES - 1u);
        }
        q4e_bf_cp_commit();

        const uint8_t *as = w_ring + s * Q4E_BF_W_STEP;
        const uint8_t *bs = x16 + ab * TILES * Q4E_BF_X_TILE;
#pragma unroll
        for (uint32_t kq = 0; kq < 2u; kq++) {
            uint32_t af[2][4];
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; mt++) {
                const uint32_t r = wm * 32u + mt * 16u + (lane & 15u);
                q4e_bf_ldsm4(af[mt], as + q4e_bf_off(r, kq * 2u + (lane >> 4)));
            }
#pragma unroll
            for (uint32_t part = 0; part < TILES; part++) {
                uint32_t bf[4][2];
#pragma unroll
                for (uint32_t np = 0; np < 2u; np++) {
                    uint32_t rr[4];
                    const uint32_t n = wn * 32u + np * 16u + (lane & 7u) + (lane >> 4) * 8u;
                    q4e_bf_ldsm4(rr, bs + part * Q4E_BF_X_TILE + q4e_bf_off(n, kq * 2u + ((lane >> 3) & 1u)));
                    bf[2 * np][0] = rr[0];
                    bf[2 * np][1] = rr[1];
                    bf[2 * np + 1][0] = rr[2];
                    bf[2 * np + 1][1] = rr[3];
                }
#pragma unroll
                for (uint32_t mt = 0; mt < 2u; mt++) {
#pragma unroll
                    for (uint32_t nt = 0; nt < 4u; nt++) {
                        q4e_bf_mma(acc[mt][nt], af[mt], bf[nt]);
                    }
                }
            }
        }
    }
    /* c0, c1 at (row gid, columns 2 tig, +1); c2, c3 eight rows down. */
#pragma unroll
    for (uint32_t mt = 0; mt < 2u; mt++) {
#pragma unroll
        for (uint32_t nt = 0; nt < 4u; nt++) {
#pragma unroll
            for (uint32_t h = 0; h < 4u; h++) {
                const uint32_t col = col0 + wn * 32u + nt * 8u + tig * 2u + (h & 1u);
                const uint32_t o = row0 + wm * 32u + mt * 16u + gid + (h >> 1) * 8u;
                if (col < rows && o < out_a) {
                    out[(uint64_t) col * out_a + o] = acc[mt][nt][h];
                } else if (col < rows && o < out_all) {
                    out_b[(uint64_t) col * out_dim_b + (o - out_a)] = acc[mt][nt][h];
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(Q4E_BF_GEMM_THREADS, 1) q4e_bf16_gemm_kernel(
        float *out, const __nv_bfloat16 *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    q4e_bf_gemm<false>(out, w, x, in_dim, out_dim, rows);
}

/* Two projections of one input, `w_a` `[out_dim_a, in_dim]` into `out_a`
 * and `w_b` `[out_dim_b, in_dim]` into `out_b`, as one launch whose rows are
 * `w_a`'s then `w_b`'s: each output is the bits `q4e_bf16_gemm_kernel` writes
 * for its own weight. The stream mix's contraction (320 rows) and its
 * injection logits (4) fit the three row tiles the contraction alone takes,
 * so the injection's whole read of the normed streams is saved. Grid
 * `(ceil((out_dim_a + out_dim_b) / 128), ceil(rows / 128))`, otherwise as
 * the plain kernel. */
extern "C" __global__ void __launch_bounds__(Q4E_BF_GEMM_THREADS, 1) q4e_bf16_gemm_pair_kernel(
        float *out_a, const __nv_bfloat16 *w_a, uint32_t out_dim_a,
        float *out_b, const __nv_bfloat16 *w_b, uint32_t out_dim_b,
        const float *x, uint32_t in_dim, uint32_t rows) {
    q4e_bf_gemm<false>(out_a, w_a, x, in_dim, out_dim_a, rows, out_b, w_b, out_dim_b);
}

/* The pair over the residual streams, each activation normed as it is read
 * (`x * scale[row][h] * nw[k]`, `scale` `[rows, in_dim / n_embd]` as
 * `q4e_hc_combine_scale_kernel` leaves it, `nw` the mix's norm weight
 * `[in_dim]`): the pair kernel's bits over the normed copy that kernel no
 * longer writes. Grid as the pair kernel's. */
extern "C" __global__ void __launch_bounds__(Q4E_BF_GEMM_THREADS, 1) q4e_bf16_gemm_pair_norm_kernel(
        float *out_a, const __nv_bfloat16 *w_a, uint32_t out_dim_a,
        float *out_b, const __nv_bfloat16 *w_b, uint32_t out_dim_b,
        const float *res, const float *scale, const float *nw, uint32_t n_embd,
        uint32_t in_dim, uint32_t rows) {
    q4e_bf_gemm<false, true>(out_a, w_a, res, in_dim, out_dim_a, rows, out_b, w_b, out_dim_b,
                             scale, nw, n_embd);
}

extern "C" __global__ void __launch_bounds__(Q4E_BF_GEMM_THREADS, 1) q4e_bf16_gemm_split_kernel(
        float *out, const __nv_bfloat16 *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    q4e_bf_gemm<true>(out, w, x, in_dim, out_dim, rows);
}

/* ---------------------------------------------------------------------------
 * The stream mix's expansion and collapse, fused.
 * ------------------------------------------------------------------------ */

/* Streams the fused collapse is written for: four 32-row stream slices make
 * one 128-row tile. */
#define Q4E_BF_COLLAPSE_HC 4u
/* Block-input elements one tile covers: 32 of each stream. */
#define Q4E_BF_COLLAPSE_E (Q4E_BF_BM / Q4E_BF_COLLAPSE_HC)

/* `out[t][e] = (1 / n_hc) sum_h xn[t][h][e] * sigmoid(up[t][h][e])`, where
 * `up[t] = w_up x[t]` is `q4e_bf16_gemm_kernel`'s expansion of the low-rank
 * row `x[t]` - the expansion and `q4e_hc_collapse_kernel<float>` in one
 * launch, so the expanded streams (`[rows, n_hc * n_embd]` f32, the widest
 * activation of a prompt chunk) are never written or read back.
 *
 * The tile's 128 weight rows are elements `e0 .. e0 + 31` of each of the four
 * streams (rows `h * n_embd + e`), so warp row `wm` is stream `wm`; every
 * output of the expansion is the plain kernel's: the same weight row and
 * activation row through the same m16n8k16 steps in the same order. Each
 * warp then turns its sums into the gates `1 / (1 + __expf(-up))` in shared
 * memory, and after one barrier each thread folds one (row, element) over the
 * streams in stream order with the collapse kernel's expressions, from the
 * normed streams in global memory - the collapse kernel's bits.
 *
 * Buffers: out `[rows, n_embd]` f32, w_up `[n_hc * n_embd, in_dim]` bf16, x
 * `[rows, in_dim]` f32 (the activated low-rank rows), xn `[rows, n_hc *
 * n_embd]` f32. Grid `(n_embd / 32, ceil(rows / 128))`, 512 threads,
 * `Q4E_BF_GEMM_SMEM(false)` bytes of dynamic shared memory. `n_hc` 4,
 * `n_embd` a multiple of 32, `in_dim` a multiple of 32.
 *
 * Measured on a GB10 (kernel bench, wall clock) at 4096 rows: 1.27-1.34 ms,
 * against 2.96-3.33 ms for the expansion and the collapse kernel. Tried and
 * dropped: issuing the normed streams' loads before the expansion, into
 * registers (118 of them): 1.50-1.56 ms. */
template <bool NORM>
__device__ __forceinline__ void q4e_bf_gemm_collapse(
        float *out, const __nv_bfloat16 *w_up, const float *x, const float *xn,
        const float *scale, const float *nw,
        uint32_t in_dim, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    constexpr uint32_t STAGES = Q4E_BF_STAGES(false);
    extern __shared__ __align__(16) uint8_t q4e_bf_cdyn[];
    uint8_t *w_ring = q4e_bf_cdyn;                              /* [STAGES][128][32] bf16 */
    uint8_t *x_raw = w_ring + STAGES * Q4E_BF_W_STEP;          /* [STAGES][128][32] f32 */
    uint8_t *x16 = x_raw + STAGES * Q4E_BF_X_RAW;              /* [2] bf16 tiles */

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t gid = lane >> 2;
    const uint32_t tig = lane & 3u;
    const uint32_t wm = warp & 3u; /* stream `wm`, 32 elements */
    const uint32_t wn = warp >> 2; /* 32 columns */

    /* The plain kernel's raster grouping, over element tiles. */
    const uint32_t out_tiles = gridDim.x;
    const uint32_t col_tiles = gridDim.y;
    const uint32_t lin = blockIdx.y * out_tiles + blockIdx.x;
    const uint32_t per_group = Q4E_BF_GROUP_TILES * col_tiles;
    const uint32_t first = (lin / per_group) * Q4E_BF_GROUP_TILES;
    const uint32_t gsize = min(out_tiles - first, Q4E_BF_GROUP_TILES);
    const uint32_t in_group = lin % per_group;
    const uint32_t e0 = (first + in_group % gsize) * Q4E_BF_COLLAPSE_E;
    const uint32_t col0 = (in_group / gsize) * Q4E_BF_BN;
    const uint32_t nk = in_dim / Q4E_BF_BK;
    (void) n_hc;

    /* Weights: every thread owns sixteen bytes of tile row `tid / 4`, which
     * is element `e0 + r % 32` of stream `r / 32`. */
    const uint32_t ar = tid >> 2;
    const uint32_t ac = tid & 3u;
    const uint32_t a_row = (ar / Q4E_BF_COLLAPSE_E) * n_embd + e0 + ar % Q4E_BF_COLLAPSE_E;
    const __nv_bfloat16 *a_src = w_up + (uint64_t) a_row * in_dim + ac * 8u;
    const uint32_t a_dst = q4e_bf_off(ar, ac);
    const uint32_t bn = tid >> 3;
    const uint32_t bq = tid & 7u;
    const bool b_live0 = col0 + bn < rows;
    const bool b_live1 = col0 + bn + 64u < rows;
    const float *b_src0 = x + (uint64_t) min(col0 + bn, rows - 1u) * in_dim + bq * 4u;
    const float *b_src1 = x + (uint64_t) min(col0 + bn + 64u, rows - 1u) * in_dim + bq * 4u;

#pragma unroll
    for (uint32_t s = 0; s < STAGES; s++) {
        if (!b_live0) {
            *reinterpret_cast<float4 *>(x_raw + s * Q4E_BF_X_RAW + tid * 16u) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (!b_live1) {
            *reinterpret_cast<float4 *>(x_raw + s * Q4E_BF_X_RAW + (tid + Q4E_BF_GEMM_THREADS) * 16u) =
                    make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    auto issue = [&](uint32_t kt) {
        const uint32_t s = kt % STAGES;
        const uint32_t k0 = kt * Q4E_BF_BK;
        q4e_bf_cp16(w_ring + s * Q4E_BF_W_STEP + a_dst, a_src + k0);
        if (b_live0) {
            q4e_bf_cp16(x_raw + s * Q4E_BF_X_RAW + tid * 16u, b_src0 + k0);
        }
        if (b_live1) {
            q4e_bf_cp16(x_raw + s * Q4E_BF_X_RAW + (tid + Q4E_BF_GEMM_THREADS) * 16u, b_src1 + k0);
        }
    };

    float acc[2][4][4];
#pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        }
    }
#pragma unroll
    for (uint32_t s = 0; s + 1u < STAGES; s++) {
        if (s < nk) {
            issue(s);
        }
        q4e_bf_cp_commit();
    }
    for (uint32_t kt = 0; kt < nk; kt++) {
        const uint32_t s = kt % STAGES;
        const uint32_t ab = kt & 1u;
        q4e_bf_cp_wait<STAGES - 2u>();
        {
            uint8_t *t = x16 + ab * Q4E_BF_X_TILE;
            const float4 v0 = *reinterpret_cast<const float4 *>(x_raw + s * Q4E_BF_X_RAW + tid * 16u);
            const float4 v1 = *reinterpret_cast<const float4 *>(
                    x_raw + s * Q4E_BF_X_RAW + (tid + Q4E_BF_GEMM_THREADS) * 16u);
            const uint32_t o0 = q4e_bf_off(bn, bq >> 1) + (bq & 1u) * 8u;
            const uint32_t o1 = q4e_bf_off(bn + 64u, bq >> 1) + (bq & 1u) * 8u;
            *reinterpret_cast<uint2 *>(t + o0) =
                    make_uint2(q4e_bf_pack(v0.x, v0.y), q4e_bf_pack(v0.z, v0.w));
            *reinterpret_cast<uint2 *>(t + o1) =
                    make_uint2(q4e_bf_pack(v1.x, v1.y), q4e_bf_pack(v1.z, v1.w));
        }
        __syncthreads();
        if (kt + STAGES - 1u < nk) {
            issue(kt + STAGES - 1u);
        }
        q4e_bf_cp_commit();

        const uint8_t *as = w_ring + s * Q4E_BF_W_STEP;
        const uint8_t *bs = x16 + ab * Q4E_BF_X_TILE;
#pragma unroll
        for (uint32_t kq = 0; kq < 2u; kq++) {
            uint32_t af[2][4];
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; mt++) {
                const uint32_t r = wm * 32u + mt * 16u + (lane & 15u);
                q4e_bf_ldsm4(af[mt], as + q4e_bf_off(r, kq * 2u + (lane >> 4)));
            }
            uint32_t bf[4][2];
#pragma unroll
            for (uint32_t np = 0; np < 2u; np++) {
                uint32_t rr[4];
                const uint32_t n = wn * 32u + np * 16u + (lane & 7u) + (lane >> 4) * 8u;
                q4e_bf_ldsm4(rr, bs + q4e_bf_off(n, kq * 2u + ((lane >> 3) & 1u)));
                bf[2 * np][0] = rr[0];
                bf[2 * np][1] = rr[1];
                bf[2 * np + 1][0] = rr[2];
                bf[2 * np + 1][1] = rr[3];
            }
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; mt++) {
#pragma unroll
                for (uint32_t nt = 0; nt < 4u; nt++) {
                    q4e_bf_mma(acc[mt][nt], af[mt], bf[nt]);
                }
            }
        }
    }
    /* Every warp is past its last read of the ring before it holds gates. */
    q4e_bf_cp_wait<0>();
    __syncthreads();
    /* The gates, `[stream][column][element]` f32: 4 x 128 x 32. */
    float *gate = reinterpret_cast<float *>(q4e_bf_cdyn);
#pragma unroll
    for (uint32_t mt = 0; mt < 2u; mt++) {
#pragma unroll
        for (uint32_t nt = 0; nt < 4u; nt++) {
#pragma unroll
            for (uint32_t h = 0; h < 4u; h++) {
                const uint32_t col = wn * 32u + nt * 8u + tig * 2u + (h & 1u);
                const uint32_t e = mt * 16u + gid + (h >> 1) * 8u;
                const float u = acc[mt][nt][h];
                gate[(wm * Q4E_BF_BN + col) * Q4E_BF_COLLAPSE_E + e] = 1.0f / (1.0f + __expf(-u));
            }
        }
    }
    __syncthreads();
    /* The collapse: thread `tid` folds element `tid % 32` of columns
     * `tid / 32, + 16, ...`, the streams in order. */
    const float inv = 1.0f / (float) Q4E_BF_COLLAPSE_HC;
    const uint32_t e = tid & (Q4E_BF_COLLAPSE_E - 1u);
    for (uint32_t c = tid / Q4E_BF_COLLAPSE_E; c < Q4E_BF_BN; c += Q4E_BF_GEMM_THREADS / Q4E_BF_COLLAPSE_E) {
        const uint32_t t = col0 + c;
        if (t >= rows) {
            break;
        }
        const float *xn_t = xn + (uint64_t) t * Q4E_BF_COLLAPSE_HC * n_embd + e0 + e;
        float v[Q4E_BF_COLLAPSE_HC];
#pragma unroll
        for (uint32_t h = 0; h < Q4E_BF_COLLAPSE_HC; h++) {
            v[h] = xn_t[(uint64_t) h * n_embd];
        }
        if (NORM) {
            /* The residual normed as it is read: the combine's expression. */
#pragma unroll
            for (uint32_t h = 0; h < Q4E_BF_COLLAPSE_HC; h++) {
                const float sc = scale[(uint64_t) t * Q4E_BF_COLLAPSE_HC + h];
                v[h] = v[h] * sc * nw[(uint64_t) h * n_embd + e0 + e];
            }
        }
        float sum = 0.0f;
#pragma unroll
        for (uint32_t h = 0; h < Q4E_BF_COLLAPSE_HC; h++) {
            sum += v[h] * gate[(h * Q4E_BF_BN + c) * Q4E_BF_COLLAPSE_E + e];
        }
        out[(uint64_t) t * n_embd + e0 + e] = sum * inv;
    }
}

extern "C" __global__ void __launch_bounds__(Q4E_BF_GEMM_THREADS, 1) q4e_bf16_gemm_collapse_kernel(
        float *out, const __nv_bfloat16 *w_up, const float *x, const float *xn,
        uint32_t in_dim, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    q4e_bf_gemm_collapse<false>(out, w_up, x, xn, nullptr, nullptr, in_dim, n_embd, n_hc, rows);
}

/* The fused expansion and collapse over the residual streams, each normed
 * as it is read (`res * scale[row][h] * nw[h * n_embd + e]`): the collapse
 * kernel's bits over the normed copy `q4e_hc_combine_scale_kernel` no longer
 * writes. Grid as `q4e_bf16_gemm_collapse_kernel`'s. */
extern "C" __global__ void __launch_bounds__(Q4E_BF_GEMM_THREADS, 1) q4e_bf16_gemm_collapse_norm_kernel(
        float *out, const __nv_bfloat16 *w_up, const float *x, const float *res,
        const float *scale, const float *nw,
        uint32_t in_dim, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    q4e_bf_gemm_collapse<true>(out, w_up, x, res, scale, nw, in_dim, n_embd, n_hc, rows);
}

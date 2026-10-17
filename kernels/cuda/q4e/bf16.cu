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
 * Five entries:
 *
 * - `q4e_bf16_gemv_kernel`, one activation row (a decode step);
 * - `q4e_bf16_rows_kernel`, two to eight (a speculative verify);
 * - `q4e_bf16_rows_wide_kernel`, nine to thirty-two (a batched verify);
 * - `q4e_bf16_gemm_kernel`, a prompt chunk of any width, on tensor cores, its
 *   activations rounded to brain floats;
 * - `q4e_bf16_gemm_split_kernel`, the same with each activation carried as
 *   two brain floats (`hi + lo`), for the narrow projections whose sums must
 *   not see the rounding (the router above all).
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
        uint32_t in_dim, uint32_t out_dim, uint32_t splits, float *__restrict__ part) {
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
        wr[r] = w + (uint64_t) min(row0 + (uint32_t) r, out_dim - 1u) * in_dim;
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
        if (row0 + r < out_dim) {
            out[(uint64_t) t * out_dim + row0 + r] = v;
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
template <bool SPLIT>
__device__ __forceinline__ void q4e_bf_gemm(
        float *__restrict__ out, const __nv_bfloat16 *__restrict__ w, const float *__restrict__ x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
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
    const bool a_live = row0 + ar < out_dim;
    const __nv_bfloat16 *a_src = w + (uint64_t) min(row0 + ar, out_dim - 1u) * in_dim + ac * 8u;
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
            const float4 v0 = *reinterpret_cast<const float4 *>(x_raw + s * Q4E_BF_X_RAW + tid * 16u);
            const float4 v1 = *reinterpret_cast<const float4 *>(
                    x_raw + s * Q4E_BF_X_RAW + (tid + Q4E_BF_GEMM_THREADS) * 16u);
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
                if (col < rows && o < out_dim) {
                    out[(uint64_t) col * out_dim + o] = acc[mt][nt][h];
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

extern "C" __global__ void __launch_bounds__(Q4E_BF_GEMM_THREADS, 1) q4e_bf16_gemm_split_kernel(
        float *out, const __nv_bfloat16 *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    q4e_bf_gemm<true>(out, w, x, in_dim, out_dim, rows);
}

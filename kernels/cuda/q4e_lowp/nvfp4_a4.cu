/* The routed expert stages of a prompt pass over an NVFP4 bank with NVFP4
 * activations (W4A4), on the block-scaled FP4 tensor-core instruction.
 *
 * Not an upstream unit. Written for this repository.
 *
 * The weight-only prompt stages (`q4e/modelopt.cu`) decode every weight to a
 * half and multiply half-rounded activations on the f16 instruction. Here the
 * activations are quantized to the weights' own format first, so both operands
 * go into `mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`
 * straight from shared memory, as stored: sixteen `e2m1` codes under one
 * `ue4m3` scale on either side, four scales a 64-wide step. Nothing is decoded,
 * and the instruction does four times the f16 one's work per issue.
 *
 * Five entries:
 *
 * - `q4e_nvfp4_a4_quant_kernel` (`f32` rows) and
 *   `q4e_nvfp4_a4_quant_bf16_kernel` (brain-float rows) - activations to
 *   NVFP4 under a global scale per row: dynamic (`448 * 6 / amax(row)`), or
 *   static, read from a table the caller indexes per row;
 * - `q4e_nvfp4_a4_gate_up_kernel` - gate and up over the routed columns with
 *   the SwiGLU fused into the epilogue: it writes `silu(g) * u` per column as
 *   brain floats, which the quantizer then reads as the down stage's input;
 * - `q4e_nvfp4_a4_down_kernel` - the down projection over those columns, one
 *   brain-float row per column;
 * - `q4e_nvfp4_a4_combine_kernel` - every token's routed rows, weighted and
 *   summed in slot order into one `f32` row.
 *
 * Brain floats between the stages, not `f32`, because past a few thousand
 * columns the stages are bound by those rows rather than by the weights: at
 * an 8,192-token chunk the down stage's `f32` output alone is 839 MB written
 * and read again by the combine, 3.6 ms each way at the GB10's bandwidth. A
 * brain float's rounding (2^-9 relative) is far below the four-bit
 * activations' own.
 *
 * # The quantized activation layout
 *
 * One buffer, three arrays back to back for `rows` rows of width `k`:
 * codes `[rows][k / 2]` (the even element in the low nibble, as the weights
 * store them), `e4m3` block scales `[rows][k / 16]`, and the row's global
 * scale `gs` as an `f32` `[rows]`. A value is `e2m1 * e4m3(scale) / gs`; a
 * weight is `e2m1 * e4m3(scale) * global`, so a product summed by the
 * instruction is scaled once, in the epilogue, by `global / gs`.
 *
 * # Rows do not depend on the pass
 *
 * A row's quantization reads that row alone, and a column's sum is the fixed
 * sequence of 64-wide instruction steps in contraction order, whatever shares
 * its tile, so a prompt's rows come out the same at every chunk width.
 *
 * The entries exist only in a build for a family-specific architecture
 * (`sm_120a`, `sm_121a`: `SHISO_CUDA_HAVE_MXF4`); anywhere else the unit is
 * empty, the host finds no entry, and the weight-only stages stay the only
 * prompt path.
 */

#include "lowp.cuh"

#if defined(SHISO_CUDA_HAVE_MXF4)

/* Elements one NVFP4 block scale covers. */
#define LP_A4_GROUP 16u

/* Byte offsets of the three arrays of a quantized activation buffer. */
__device__ __forceinline__ uint64_t lp_a4_scales_at(uint32_t rows, uint32_t k) {
    return (uint64_t) rows * (k / 2u);
}

__device__ __forceinline__ uint64_t lp_a4_gs_at(uint32_t rows, uint32_t k) {
    return lp_a4_scales_at(rows, k) + (uint64_t) rows * (k / LP_A4_GROUP);
}

/* One pass of a row's quantization: 128 consecutive elements, four a lane,
 * so every load is one coalesced line. A block of sixteen is four
 * neighbouring lanes, which agree on its amax by two shuffles and on its
 * scale by computing it from the same bits; the first lane of the four
 * gathers their sixteen codes and stores them. */
__device__ __forceinline__ void lp_a4_quant_pass(uint8_t *codes, uint8_t *scales, const float *v,
                                                 float gs, uint32_t p, uint32_t lane) {
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
    uint32_t sbyte;
    const uint32_t nib = lp_fp4_codes4(v, a, gs, sbyte);
    const uint32_t lo = nib | (__shfl_down_sync(0xffffffffu, nib, 1) << 16);
    const uint32_t hi = __shfl_down_sync(0xffffffffu, lo, 2);
    if ((lane & 3u) == 0u) {
        const uint32_t g = p * 8u + (lane >> 2);
        *(uint2 *) (codes + (uint64_t) g * 8u) = make_uint2(lo, hi);
        scales[g] = (uint8_t) sbyte;
    }
}

/* The row's global scale: static, or `448 * 6 / amax` over the lanes'
 * partial maxima. */
__device__ __forceinline__ float lp_a4_global(float amax, const float *static_gs,
                                              const int *static_idx, uint32_t row) {
    if (static_gs != nullptr) {
        return static_gs[static_idx != nullptr ? static_idx[row] : 0];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    return amax > 0.0f ? __fdiv_rn(2688.0f, amax) : 0.0f;
}

/* One row of `x` to NVFP4, by one warp, holding up to `P` passes of it in
 * registers between the amax and the quantization - a second read of an
 * 84 MB chunk of `f32` rows would come from DRAM, not from the cache - and
 * reading twice past that. */
template <uint32_t P, typename T>
__device__ __forceinline__ void lp_a4_quant_row(uint8_t *q, const T *x, const float *static_gs,
                                                const int *static_idx, uint32_t rows, uint32_t k,
                                                uint32_t row, uint32_t lane) {
    const T *xr = x + (uint64_t) row * k;
    const uint32_t passes = k / 128u;
    uint8_t *codes = q + (uint64_t) row * (k / 2u);
    uint8_t *scales = q + lp_a4_scales_at(rows, k) + (uint64_t) row * (k / LP_A4_GROUP);
    float gs;
    if (passes <= P) {
        float held[P][4];
        float amax = 0.0f;
#pragma unroll
        for (uint32_t p = 0; p < P; ++p) {
            if (p < passes) {
                lp_load4(xr + p * 128u + lane * 4u, held[p]);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(held[p][0]), fabsf(held[p][1])),
                                         fmaxf(fabsf(held[p][2]), fabsf(held[p][3]))));
            }
        }
        gs = lp_a4_global(amax, static_gs, static_idx, row);
#pragma unroll
        for (uint32_t p = 0; p < P; ++p) {
            if (p < passes) {
                lp_a4_quant_pass(codes, scales, held[p], gs, p, lane);
            }
        }
    } else {
        float amax = 0.0f;
        if (static_gs == nullptr) {
            for (uint32_t p = 0; p < passes; ++p) {
                float v[4];
                lp_load4(xr + p * 128u + lane * 4u, v);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
            }
        }
        gs = lp_a4_global(amax, static_gs, static_idx, row);
        for (uint32_t p = 0; p < passes; ++p) {
            float v[4];
            lp_load4(xr + p * 128u + lane * 4u, v);
            lp_a4_quant_pass(codes, scales, v, gs, p, lane);
        }
    }
    if (lane == 0u) {
        ((float *) (q + lp_a4_gs_at(rows, k)))[row] = gs;
    }
}

/* Passes of 128 a quantizer warp holds in registers: the family's hidden
 * width (2,560) whole. */
#define LP_A4_HELD 20u

/* `rows` rows of `x [rows, k]` to NVFP4, a warp a row.
 *
 * The global scale of row `r` is `static_gs[static_idx ? static_idx[r] : 0]`
 * when `static_gs` is given, and otherwise `448 * 6 / amax(row)` - the
 * largest block of the row then takes the largest scale - or zero for a row
 * of zeros, whose codes and scales are then all zero.
 *
 * Buffers: q (the layout above), x (`f32`), static_gs (or null), static_idx
 * (or null), rows, k. Grid `ceil(rows / 4)`, 128 threads; `k` a multiple of
 * 128. */
extern "C" __global__ void __launch_bounds__(128) q4e_nvfp4_a4_quant_kernel(
        uint8_t *q, const float *x, const float *static_gs, const int *static_idx,
        uint32_t rows, uint32_t k) {
    const uint32_t row = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (row < rows) {
        lp_a4_quant_row<LP_A4_HELD>(q, x, static_gs, static_idx, rows, k, row, threadIdx.x & 31u);
    }
}

/* The same over brain-float rows (`x` as `uint16_t` bit patterns). */
extern "C" __global__ void __launch_bounds__(128) q4e_nvfp4_a4_quant_bf16_kernel(
        uint8_t *q, const uint16_t *x, const float *static_gs, const int *static_idx,
        uint32_t rows, uint32_t k) {
    const uint32_t row = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (row < rows) {
        lp_a4_quant_row<LP_A4_HELD>(q, x, static_gs, static_idx, rows, k, row, threadIdx.x & 31u);
    }
}

/* ---------------------------------------------------------------------------
 * The GEMM.
 *
 * A block owns 128 weight rows - for gate and up the same 64 rows of both, so
 * one staged activation tile feeds both products and the SwiGLU has both
 * halves in one thread's registers - and walks its expert's columns 64 at a
 * time, the column tiles of one expert split over `LP_A4_SPLIT` neighbouring
 * blocks as the weight-only stages split them, so a heavy expert is not one
 * long block and its weight tile is still read from DRAM about once.
 *
 * Both operands arrive by `cp.async`, `LP_A4_SK` contraction elements a
 * stage - a whole 128-byte line of every weight row - `LP_A4_STAGES` stages
 * deep. Eight warps each own 32 weight rows (two m16 tiles: gate and up of
 * the same rows, or two neighbouring tiles of down) by 32 columns, and take
 * both operands with `ldmatrix`: the code bytes of a row are the
 * instruction's fragment bytes in order, so a weight tile is used exactly as
 * it sits in memory. A row's or a column's scales for the whole stage are one
 * 16-byte shared load, a 32-bit register per 64-wide step.
 *
 * GB10, the kernel bench's Zipf routing over 512 experts, median of three,
 * against the weight-only stages (gate-and-up there plus its separate
 * SwiGLU): gate+up 2,048 tokens 5.95 + 0.72 -> 4.64 ms, 8,192 tokens
 * 12.37 + 2.78 -> 6.07 ms; down 3.69 -> 2.83 ms and 8.74 -> 5.13 ms. At
 * 2,048 tokens both are at about 210 GB/s of weights and rows - the bound
 * is the bank, read once, not the arithmetic; at 8,192 the brain-float rows
 * are a third of the bytes. The first version - 128-wide stages three deep,
 * `f32` rows, 64 columns a tile - took 5.47 / 6.91 ms (gate+up, 2,048 /
 * 8,192); line-wide stages with brain-float rows 4.59 / 6.15, and 128
 * columns a tile 4.64 / 5.96.
 * ------------------------------------------------------------------------ */

#ifndef LP_A4_SK
#define LP_A4_SK 256u
#endif
#ifndef LP_A4_STAGES
#define LP_A4_STAGES 2u
#endif
#define LP_A4_BW 128u
#ifndef LP_A4_BN
#define LP_A4_BN 128u
#endif
/* Columns a warp owns, and its m16n8 tiles and ldmatrix pairs across them. */
#define LP_A4_WN (LP_A4_BN / 2u)
#define LP_A4_NT (LP_A4_WN / 8u)
#define LP_A4_NP (LP_A4_WN / 16u)
#define LP_A4_THREADS 256u
#define LP_A4_SPLIT 4u
/* 16-byte chunks a row's codes take in one stage, and 64-wide steps a stage. */
#define LP_A4_CPR (LP_A4_SK / 32u)
#define LP_A4_STEPS (LP_A4_SK / 64u)

/* One stage: weight codes, weight scales, activation codes, activation scales. */
#define LP_A4_W_BYTES (LP_A4_BW * LP_A4_SK / 2u)
#define LP_A4_WS_BYTES (LP_A4_BW * LP_A4_SK / LP_A4_GROUP)
#define LP_A4_X_BYTES (LP_A4_BN * LP_A4_SK / 2u)
#define LP_A4_XS_BYTES (LP_A4_BN * LP_A4_SK / LP_A4_GROUP)
#define LP_A4_STAGE (LP_A4_W_BYTES + LP_A4_WS_BYTES + LP_A4_X_BYTES + LP_A4_XS_BYTES)
/* Floats a staged output column takes: 128 rows plus 4 against bank conflicts. */
#define LP_A4_OLD (LP_A4_BW + 4u)
#define LP_A4_RING (LP_A4_STAGES * LP_A4_STAGE)
/* Dynamic shared memory a block takes: the ring (the output tile is staged
 * over it), the column list and two scale factors a column. */
#define LP_A4_SMEM (LP_A4_RING + LP_A4_BN * 4u * 3u)

static_assert(LP_A4_BN * LP_A4_OLD * 4u <= LP_A4_RING, "the output tile fits over the ring");
static_assert(LP_A4_SK == 128u || LP_A4_SK == 256u, "a stage is half a line or a line");

/* Byte offset of chunk `c` of row `r` in a stage's code tile. */
__device__ __forceinline__ uint32_t lp_a4_off(uint32_t r, uint32_t c) {
    return LP_A4_SK == 256u ? lp_off128(r, c) : lp_off64(r, c);
}

/* A row's (or a column's) scale register for every step of the stage. */
__device__ __forceinline__ void lp_a4_scales(const uint8_t *p, uint32_t *s) {
    if (LP_A4_STEPS == 4u) {
        const uint4 v = *(const uint4 *) p;
        s[0] = v.x;
        s[1] = v.y;
        s[2] = v.z;
        s[3] = v.w;
    } else {
        const uint2 v = *(const uint2 *) p;
        s[0] = v.x;
        s[1] = v.y;
    }
}

__device__ __forceinline__ void lp_mma_fp4(float *c, const uint32_t *a, const uint32_t *b,
                                           uint32_t sa, uint32_t sb) {
    asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1."
            "e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, "
            "{0, 0}, %11, {0, 0};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}

/* One block: weight rows `row0 ..` of gate (smem rows 0..63) and of up
 * (64..127), or rows `row0 .. row0 + 127` of down. Every `LP_A4_SPLIT`th
 * 64-column tile of the expert's columns `c0 .. c1`, from tile `z`.
 * Activation row of column `col` is `col / div`; the output is `[col][o]`
 * brain floats. */
template <bool GU>
__device__ __forceinline__ void lp_a4_block(
        const uint8_t *w0, const uint8_t *s0, float g0, const uint8_t *w1, const uint8_t *s1,
        float g1, uint16_t *out, const uint8_t *xq, uint32_t rows,
        const int *__restrict__ sorted, int c0, int c1, uint32_t div, uint32_t in_dim,
        uint32_t out_dim, uint32_t row0, uint32_t z) {
    extern __shared__ __align__(128) uint8_t lp_a4_dyn[];
    uint8_t *ring = lp_a4_dyn;
    int *cols = (int *) (lp_a4_dyn + LP_A4_RING);
    float *alpha_a = (float *) (cols + LP_A4_BN);
    float *alpha_b = alpha_a + LP_A4_BN;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t gid = lane >> 2;
    const uint32_t tig = lane & 3u;
    const uint32_t wm = warp & 3u;
    const uint32_t wn = warp >> 2;
    const uint32_t nk = (in_dim + LP_A4_SK - 1u) / LP_A4_SK;
    const uint64_t w_stride = in_dim / 2u;
    const uint64_t s_stride = in_dim / LP_A4_GROUP;

    const uint8_t *x_codes = xq;
    const uint8_t *x_scales = xq + lp_a4_scales_at(rows, in_dim);
    const float *x_gs = (const float *) (xq + lp_a4_gs_at(rows, in_dim));

    /* This thread's weight code chunks: chunk `tid % CPR` of smem rows
     * `tid / CPR + (256 / CPR) i`. For gate and up the first half of those
     * rows is gate's and the second up's, at the same matrix row. */
    constexpr uint32_t W_PER = LP_A4_BW * LP_A4_CPR / LP_A4_THREADS;
    constexpr uint32_t W_ROWS = LP_A4_THREADS / LP_A4_CPR;
    const uint32_t wc = tid % LP_A4_CPR;
    const uint32_t wr = tid / LP_A4_CPR;
    const uint8_t *w_src[W_PER];
    bool w_ok[W_PER];
#pragma unroll
    for (uint32_t i = 0; i < W_PER; ++i) {
        const uint32_t r = wr + W_ROWS * i;
        const bool second = GU && r >= LP_A4_BW / 2u;
        const uint32_t grow = row0 + (second ? r - LP_A4_BW / 2u : r);
        w_ok[i] = grow < out_dim;
        w_src[i] = (second ? w1 : w0) + (uint64_t) (w_ok[i] ? grow : 0u) * w_stride + wc * 16u;
    }
    /* Weight scales: every thread one 8-byte half of smem row `tid / 2`'s
     * stage (two for a 256-wide stage: 16 bytes a row). */
    constexpr uint32_t WS_HALVES = LP_A4_SK / 128u; /* 8-byte pieces a row a stage */
    const uint32_t ws_r = WS_HALVES == 2u ? tid >> 1 : tid & (LP_A4_BW - 1u);
    const uint32_t ws_h = WS_HALVES == 2u ? tid & 1u : 0u;
    const bool ws_on = WS_HALVES == 2u || tid < LP_A4_BW;
    const uint8_t *ws_src;
    bool ws_row_ok;
    {
        const bool second = GU && ws_r >= LP_A4_BW / 2u;
        const uint32_t grow = row0 + (second ? ws_r - LP_A4_BW / 2u : ws_r);
        ws_row_ok = grow < out_dim;
        ws_src = (second ? s1 : s0) + (uint64_t) (ws_row_ok ? grow : 0u) * s_stride + ws_h * 8u;
    }

    /* Smem row of this warp's m16 tile `mt`. */
    auto tile_row = [&](uint32_t mt) -> uint32_t {
        return GU ? mt * 64u + wm * 16u : wm * 32u + mt * 16u;
    };

    for (int n_base = c0 + (int) (z * LP_A4_BN); n_base < c1;
         n_base += (int) (LP_A4_SPLIT * LP_A4_BN)) {
        __syncthreads(); /* the previous tile is done with every buffer */
        if (tid < LP_A4_BN) {
            const int c = n_base + (int) tid;
            const int col = c < c1 ? sorted[c] : -1;
            cols[tid] = col;
            float aa = 0.0f, ab = 0.0f;
            if (col >= 0) {
                const float gs = x_gs[(uint32_t) col / div];
                if (gs != 0.0f) {
                    aa = __fdiv_rn(g0, gs);
                    ab = __fdiv_rn(g1, gs);
                }
            }
            alpha_a[tid] = aa;
            alpha_b[tid] = ab;
        }
        __syncthreads();
        const int n_valid = min((int) LP_A4_BN, c1 - n_base);

        /* Activation codes: chunk `tid % CPR` of columns `tid / CPR + 32 i`;
         * scales: 8-byte halves as the weights take them. */
        constexpr uint32_t X_PER = LP_A4_BN * LP_A4_CPR / LP_A4_THREADS;
        const uint8_t *x_src[X_PER];
        bool x_ok[X_PER];
#pragma unroll
        for (uint32_t i = 0; i < X_PER; ++i) {
            const int col = cols[wr + W_ROWS * i];
            x_ok[i] = col >= 0;
            x_src[i] = x_codes + (uint64_t) (x_ok[i] ? (uint32_t) col / div : 0u) * w_stride + wc * 16u;
        }
        const uint32_t xs_c = WS_HALVES == 2u ? tid >> 1 : tid;
        const bool xs_on = xs_c < LP_A4_BN;
        bool xs_row_ok = false;
        const uint8_t *xs_src = x_scales;
        if (xs_on) {
            const int col = cols[xs_c];
            xs_row_ok = col >= 0;
            xs_src = x_scales + (uint64_t) (xs_row_ok ? (uint32_t) col / div : 0u) * s_stride + ws_h * 8u;
        }

        auto issue = [&](uint32_t kt, uint32_t slot) {
            uint8_t *sw = ring + slot * LP_A4_STAGE;
            uint8_t *sws = sw + LP_A4_W_BYTES;
            uint8_t *sx = sws + LP_A4_WS_BYTES;
            uint8_t *sxs = sx + LP_A4_X_BYTES;
            const uint32_t k0 = kt * LP_A4_SK;
            const bool c_in = k0 + wc * 32u < in_dim; /* this chunk inside the contraction */
            const uint32_t kb = c_in ? k0 / 2u : 0u;
#pragma unroll
            for (uint32_t i = 0; i < W_PER; ++i) {
                lp_cp16(sw + lp_a4_off(wr + W_ROWS * i, wc), w_src[i] + kb, w_ok[i] && c_in);
            }
#pragma unroll
            for (uint32_t i = 0; i < X_PER; ++i) {
                lp_cp16(sx + lp_a4_off(wr + W_ROWS * i, wc), x_src[i] + kb, x_ok[i] && c_in);
            }
            const bool s_in = k0 + ws_h * 128u < in_dim; /* this 8-byte half inside */
            const uint32_t sb = s_in ? k0 / LP_A4_GROUP : 0u;
            if (ws_on) {
                lp_cp8(sws + ws_r * (LP_A4_SK / LP_A4_GROUP) + ws_h * 8u, ws_src + sb,
                       ws_row_ok && s_in);
            }
            if (xs_on) {
                lp_cp8(sxs + xs_c * (LP_A4_SK / LP_A4_GROUP) + ws_h * 8u, xs_src + sb,
                       xs_row_ok && s_in);
            }
        };

        float acc[2][LP_A4_NT][4];
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
#pragma unroll
            for (uint32_t j = 0; j < LP_A4_NT; ++j) {
                acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
            }
        }

#pragma unroll
        for (uint32_t s = 0; s + 1u < LP_A4_STAGES; ++s) {
            if (s < nk) {
                issue(s, s);
            }
            lp_commit();
        }
        for (uint32_t kt = 0; kt < nk; ++kt) {
            lp_wait<LP_A4_STAGES - 2u>();
            __syncthreads();
            /* Into the slot step `kt - 1` read, which every warp has finished
             * with: they all passed the barrier. */
            const uint32_t nxt = kt + LP_A4_STAGES - 1u;
            if (nxt < nk) {
                issue(nxt, nxt % LP_A4_STAGES);
            }
            lp_commit();

            const uint8_t *st = ring + (kt % LP_A4_STAGES) * LP_A4_STAGE;
            const uint32_t sw = lp_smem(st);
            const uint8_t *sws = st + LP_A4_W_BYTES;
            const uint32_t sx = lp_smem(st + LP_A4_W_BYTES + LP_A4_WS_BYTES);
            const uint8_t *sxs = st + LP_A4_W_BYTES + LP_A4_WS_BYTES + LP_A4_X_BYTES;
            uint32_t sa[2][LP_A4_STEPS];
            uint32_t sb[LP_A4_NT][LP_A4_STEPS];
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; ++mt) {
                const uint32_t sr = tile_row(mt) + (lane >> 2) + (lane & 1u) * 8u;
                lp_a4_scales(sws + sr * (LP_A4_SK / LP_A4_GROUP), sa[mt]);
            }
#pragma unroll
            for (uint32_t nt = 0; nt < LP_A4_NT; ++nt) {
                const uint32_t sc = wn * LP_A4_WN + nt * 8u + (lane >> 2);
                lp_a4_scales(sxs + sc * (LP_A4_SK / LP_A4_GROUP), sb[nt]);
            }
            const uint32_t k0 = kt * LP_A4_SK;
#pragma unroll
            for (uint32_t j = 0; j < LP_A4_STEPS; ++j) {
                if (k0 + j * 64u >= in_dim) {
                    break;
                }
                uint32_t a[2][4];
#pragma unroll
                for (uint32_t mt = 0; mt < 2u; ++mt) {
                    lp_ldsm4(a[mt], sw + lp_a4_off(tile_row(mt) + (lane & 15u), 2u * j + (lane >> 4)));
                }
#pragma unroll
                for (uint32_t np = 0; np < LP_A4_NP; ++np) {
                    if ((int) (wn * LP_A4_WN + np * 16u) < n_valid) {
                        uint32_t b[4];
                        const uint32_t n = wn * LP_A4_WN + np * 16u + (lane & 7u) + ((lane >> 4) << 3);
                        lp_ldsm4(b, sx + lp_a4_off(n, 2u * j + ((lane >> 3) & 1u)));
#pragma unroll
                        for (uint32_t h = 0; h < 2u; ++h) {
#pragma unroll
                            for (uint32_t mt = 0; mt < 2u; ++mt) {
                                lp_mma_fp4(acc[mt][2u * np + h], a[mt], b + 2u * h, sa[mt][j],
                                           sb[2u * np + h][j]);
                            }
                        }
                    }
                }
            }
        }

        /* The output tile through shared memory, so each column's rows leave
         * as whole lines. c0, c1 sit at (row gid, columns 2 tig, +1); c2, c3
         * eight rows down. */
        lp_wait<0>();
        __syncthreads(); /* every warp is done with the ring */
        float *os = (float *) ring;
#pragma unroll
        for (uint32_t nt = 0; nt < LP_A4_NT; ++nt) {
            if ((int) (wn * LP_A4_WN + nt * 8u) < n_valid) {
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t n = wn * LP_A4_WN + nt * 8u + tig * 2u + (e & 1u);
                    const uint32_t dr = gid + (e >> 1) * 8u;
                    if (GU) {
                        const float g = acc[0][nt][e] * alpha_a[n];
                        const float u = acc[1][nt][e] * alpha_b[n];
                        os[n * LP_A4_OLD + wm * 16u + dr] = (g / (1.0f + __expf(-g))) * u;
                    } else {
#pragma unroll
                        for (uint32_t mt = 0; mt < 2u; ++mt) {
                            os[n * LP_A4_OLD + tile_row(mt) + dr] = acc[mt][nt][e] * alpha_a[n];
                        }
                    }
                }
            }
        }
        __syncthreads();
        {
            /* A column's rows as brain floats, eight (16 bytes) a thread. */
            const uint32_t width = GU ? LP_A4_BW / 2u : LP_A4_BW;
            const uint32_t per_col = width / 8u;
            const uint32_t q = tid % per_col;
            const uint32_t o = row0 + q * 8u;
            for (int n = (int) (tid / per_col); n < n_valid; n += (int) (LP_A4_THREADS / per_col)) {
                const int col = cols[n];
                const float *src = &os[n * LP_A4_OLD + q * 8u];
                uint32_t h[4];
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    h[e] = lp_to_bf16(src[2u * e]) | (lp_to_bf16(src[2u * e + 1u]) << 16);
                }
                uint16_t *dst = out + (uint64_t) col * out_dim + o;
                if ((out_dim & 7u) == 0u && o + 7u < out_dim) {
                    *(uint4 *) dst = make_uint4(h[0], h[1], h[2], h[3]);
                } else {
#pragma unroll
                    for (uint32_t e = 0; e < 8u; ++e) {
                        if (o + e < out_dim) {
                            dst[e] = (uint16_t) (h[e / 2u] >> (16u * (e & 1u)));
                        }
                    }
                }
            }
        }
    }
}

/* Gate and up of a prompt pass, W4A4, with the SwiGLU fused:
 * `mid[col][r] = bf16(silu(g) * u)`, `g` and `u` each the instruction's sum
 * times its matrix's global over the column's activation global. A block
 * covers the same 64 rows of both, row tile `blockIdx.x / LP_A4_SPLIT`,
 * column tiles from `blockIdx.x % LP_A4_SPLIT`; one active expert a block on
 * the grid's second axis.
 *
 * Buffers: mid (brain floats `[columns, out_dim]`), gate_w, gate_s, gate_g,
 * up_w, up_s, up_g, xq (the token rows, quantized), sorted, bounds, active,
 * map, in_dim, out_dim, n_expert, n_used, rows (the tokens `xq` holds). Grid
 * `(ceil(out_dim / 64) * 4, min(columns, n_expert))`, 256 threads,
 * `LP_A4_SMEM` bytes of dynamic shared memory; `in_dim` a multiple of 128. */
extern "C" __global__ void __launch_bounds__(LP_A4_THREADS) q4e_nvfp4_a4_gate_up_kernel(
        uint16_t *mid, const uint8_t *gate_w, const uint8_t *gate_s, const float *gate_g,
        const uint8_t *up_w, const uint8_t *up_s, const float *up_g, const uint8_t *xq,
        const int *sorted, const int *bounds, const int *active, const uint32_t *map,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_expert, uint32_t n_used, uint32_t rows) {
    const int n_active = active[n_expert];
    if ((int) blockIdx.y >= n_active) {
        return;
    }
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t z = blockIdx.x % LP_A4_SPLIT;
    if (c1 - c0 <= (int) (z * LP_A4_BN)) {
        return;
    }
    const uint32_t row0 = (blockIdx.x / LP_A4_SPLIT) * (LP_A4_BW / 2u);
    const uint64_t slot = map[expert];
    const uint64_t first = slot * out_dim;
    lp_a4_block<true>(gate_w + first * (in_dim / 2u), gate_s + first * (in_dim / LP_A4_GROUP),
                      gate_g[slot], up_w + first * (in_dim / 2u),
                      up_s + first * (in_dim / LP_A4_GROUP), up_g[slot], mid, xq, rows, sorted,
                      c0, c1, n_used, in_dim, out_dim, row0, z);
}

/* The routed down projection of a prompt pass, W4A4: 128 rows a block.
 *
 * Buffers: out (brain floats `[columns, out_dim]`), down_w, down_s, down_g,
 * midq (the routed columns, quantized), sorted, bounds, active, map, in_dim,
 * out_dim, n_expert, n_used, rows (the columns `midq` holds). Grid
 * `(ceil(out_dim / 128) * 4, min(columns, n_expert))`, 256 threads,
 * `LP_A4_SMEM` bytes of dynamic shared memory; `in_dim` a multiple of 128. */
extern "C" __global__ void __launch_bounds__(LP_A4_THREADS) q4e_nvfp4_a4_down_kernel(
        uint16_t *out, const uint8_t *down_w, const uint8_t *down_s, const float *down_g,
        const uint8_t *midq, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used, uint32_t rows) {
    (void) n_used;
    const int n_active = active[n_expert];
    if ((int) blockIdx.y >= n_active) {
        return;
    }
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t z = blockIdx.x % LP_A4_SPLIT;
    if (c1 - c0 <= (int) (z * LP_A4_BN)) {
        return;
    }
    const uint32_t row0 = (blockIdx.x / LP_A4_SPLIT) * LP_A4_BW;
    const uint64_t slot = map[expert];
    const uint64_t first = slot * out_dim;
    const uint8_t *w = down_w + first * (in_dim / 2u);
    const uint8_t *s = down_s + first * (in_dim / LP_A4_GROUP);
    const float g = down_g[slot];
    lp_a4_block<false>(w, s, g, w, s, g, out, midq, rows, sorted, c0, c1, 1u, in_dim, out_dim,
                       row0, z);
}

/* Every token's routed rows, weighted and summed in slot order:
 * `out[t][e] = sum_s down[t * n_used + s][e] * weights[t * n_used + s]`, the
 * routed rows brain floats and the sum `f32` from zero - the weight-only
 * stage's combine (`q4e_moe_combine_kernel`) over this stage's rows.
 *
 * Buffers: out `[tokens, n_embd]` f32, down (brain floats), weights, n_embd,
 * n_used. Grid `(ceil(n_embd / 512), tokens)`, 256 threads; `n_embd` even. */
extern "C" __global__ void __launch_bounds__(256) q4e_nvfp4_a4_combine_kernel(
        float *out, const uint16_t *down, const float *weights, uint32_t n_embd,
        uint32_t n_used) {
    const uint32_t t = blockIdx.y;
    const uint32_t e = (blockIdx.x * 256u + threadIdx.x) * 2u;
    if (e >= n_embd) {
        return;
    }
    float a0 = 0.0f, a1 = 0.0f;
    for (uint32_t s = 0; s < n_used; ++s) {
        const uint64_t col = (uint64_t) t * n_used + s;
        const uint32_t pair = *(const uint32_t *) (down + col * n_embd + e);
        const float w = weights[col];
        a0 = __fadd_rn(a0, __fmul_rn(lp_bf16(pair & 0xFFFFu), w));
        a1 = __fadd_rn(a1, __fmul_rn(lp_bf16(pair >> 16), w));
    }
    *(float2 *) (out + (uint64_t) t * n_embd + e) = make_float2(a0, a1);
}

#endif /* SHISO_CUDA_HAVE_MXF4 */

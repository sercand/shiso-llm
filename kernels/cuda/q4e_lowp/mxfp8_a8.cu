/* Dense projections of a prompt pass over an MXFP8 weight with MXFP8
 * activations (W8A8), on the block-scaled FP8 tensor-core instruction.
 *
 * Not an upstream unit. Written for this repository.
 *
 * The weight-only prompt kernel (`q4e/mxfp8.cu`, `q4e_mxfp8_gemm_kernel`)
 * widens every code to a half, rounds the activations to halves, and scales
 * each 32-wide group's product into the running sum in single precision. Here
 * the activations are quantized to the weight's own format first - `e4m3`
 * codes under one `ue8m0` power of two per 32 along a row - so both operands
 * go into `mma.sync ... kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`
 * straight from shared memory, and the group scales are applied by the
 * instruction itself: no conversion in the loop and no per-group epilogue.
 *
 * Two entries:
 *
 * - `q4e_mxfp8_a8_quant_kernel` - `f32` rows to MXFP8, per row per 32:
 *   `e = clamp(ceil(log2(amax / 448)), -127, 127)` (0 for an all-zero group),
 *   `code = e4m3_rn(x / 2^e)`, scale byte `e + 127`;
 * - `q4e_mxfp8_a8_gemm_kernel` - `out[row][o] = sum_k w[o][k] x[row][k]`.
 *
 * # The quantized activation layout
 *
 * Codes `[rows][k]`, then the scale bytes `[rows][k / 32]`.
 *
 * # Rows do not depend on the pass
 *
 * A row's quantization reads that row alone and its sums are the fixed
 * sequence of 32-wide instruction steps in contraction order, whatever
 * shares the tile.
 *
 * Present only in a build for a family-specific architecture
 * (`SHISO_CUDA_HAVE_MXF4`); see `nvfp4_a4.cu`.
 */

#include "lowp.cuh"

#if defined(SHISO_CUDA_HAVE_MXF4)

#define LP_A8_GROUP 32u

/* The exact `ceil(log2(r))` of a positive finite `f32`. */
__device__ __forceinline__ int lp_ceil_log2(float r) {
    const uint32_t bits = __float_as_uint(r);
    const uint32_t ex = (bits >> 23) & 0xFFu;
    const uint32_t mant = bits & 0x7FFFFFu;
    if (ex == 0u) {
        const int msb = 31 - __clz((int) mant);
        return msb - 149 + ((mant & (mant - 1u)) != 0u ? 1 : 0);
    }
    return (int) ex - 127 + (mant != 0u ? 1 : 0);
}

/* `x [rows, k]` to MXFP8, eight threads a 32-wide group: four consecutive
 * values a thread, so a warp's loads are whole lines; the eight agree on the
 * group's amax by three shuffles and compute its exponent from the same bits.
 *
 * Buffers: q (the layout above), x, rows, k. Grid `ceil(rows * k / 1024)`,
 * 256 threads; `k` a multiple of 32. */
extern "C" __global__ void __launch_bounds__(256) q4e_mxfp8_a8_quant_kernel(
        uint8_t *q, const float *x, uint32_t rows, uint32_t k) {
    const uint64_t total = (uint64_t) rows * k / 4u;
    const uint64_t item = (uint64_t) blockIdx.x * 256u + threadIdx.x;
    const bool live = item < total;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (live) {
        lp_load4(x + item * 4u, v);
    }
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    if (!live) {
        return;
    }
    int e = 0;
    if (amax > 0.0f) {
        e = min(max(lp_ceil_log2(__fdiv_rn(amax, 448.0f)), -127), 127);
    }
    /* `2^-e` is a normal float for every `e` a finite amax reaches (at most
     * 120), and multiplying by it is the division by `2^e`, exactly. */
    const float inv = __uint_as_float((uint32_t) (127 - e) << 23);
    const uint32_t w = lp_to_e4m3(__fmul_rn(v[0], inv)) | (lp_to_e4m3(__fmul_rn(v[1], inv)) << 8)
                     | (lp_to_e4m3(__fmul_rn(v[2], inv)) << 16)
                     | (lp_to_e4m3(__fmul_rn(v[3], inv)) << 24);
    *(uint32_t *) (q + item * 4u) = w;
    if ((threadIdx.x & 7u) == 0u) {
        q[(uint64_t) rows * k + item / 8u] = (uint8_t) (e + 127);
    }
}

/* ---------------------------------------------------------------------------
 * The GEMM.
 *
 * A 128 x 128 tile of `out[row][o]`: the activation rows are the
 * instruction's M side and the weight rows its N side, so a thread's two
 * neighbouring accumulators are two neighbouring outputs of one row and leave
 * as one 8-byte store. Eight warps, each 64 rows by 32 outputs.
 *
 * Codes arrive by `cp.async`, 64 contraction elements a stage (a 64-byte
 * half line of every row), `LP_A8_STAGES` deep, two stages an iteration. Scales arrive 16 groups (512
 * elements, eight stages) at a time per row, double-buffered, issued with the
 * first stage they serve - so the contraction width must be a multiple of
 * 512, which every MXFP8 projection of this family's export is, and the
 * planner keeps any other on the weight-only kernel. A scale register holds
 * four consecutive groups and the instruction's byte selector picks the one
 * a 32-wide step needs.
 *
 * Tiles run in raster groups of sixteen output tiles, every row tile of a
 * group before the next, as the weight-only kernel runs them.
 *
 * GB10, 2,048 rows, median of three, against the weight-only kernel (the
 * quantizer's own time added): 10240 x 2560 2.68 -> 1.12 + 0.06 ms (40 -> 96
 * TFLOP/s), 12288 x 2560 3.22 -> 1.34 + 0.06, 6144 x 2560 1.61 -> 0.68 +
 * 0.06, 2560 x 6144 1.66 -> 0.60 + 0.25.
 * ------------------------------------------------------------------------ */

#define LP_A8_BM 128u
#define LP_A8_BN 128u
#define LP_A8_SK 64u
#define LP_A8_THREADS 256u
#define LP_A8_STAGES 4u
#define LP_A8_GROUP_TILES 16u
/* Stages one scale chunk serves, and the bytes of a row's chunk. */
#define LP_A8_SCALE_STAGES 8u
#define LP_A8_SCALE_CHUNK (LP_A8_SCALE_STAGES * LP_A8_SK / LP_A8_GROUP)

#define LP_A8_TILE (LP_A8_BM * LP_A8_SK) /* code bytes of one operand a stage */
#define LP_A8_STAGE (2u * LP_A8_TILE)
#define LP_A8_RING (LP_A8_STAGES * LP_A8_STAGE)
#define LP_A8_SCALES (2u * (LP_A8_BM + LP_A8_BN) * LP_A8_SCALE_CHUNK)
/* Dynamic shared memory a block takes. */
#define LP_A8_SMEM (LP_A8_RING + LP_A8_SCALES)

template <int SEL>
__device__ __forceinline__ void lp_mma_fp8(float *c, const uint32_t *a, const uint32_t *b,
                                           uint32_t sa, uint32_t sb) {
    asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3."
            "e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, "
            "{%12, 0}, %11, {%12, 0};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb),
              "n"(SEL));
}

/* Buffers: out `[rows, out_dim]` f32, codes, scales (the weight), xq (the
 * activations, quantized), in_dim, out_dim, rows. Grid `(ceil(out_dim / 128),
 * ceil(rows / 128))`, 256 threads, `LP_A8_SMEM` bytes of dynamic shared
 * memory; `in_dim` a multiple of 512. */
extern "C" __global__ void __launch_bounds__(LP_A8_THREADS, 1) q4e_mxfp8_a8_gemm_kernel(
        float *out, const uint8_t *codes, const uint8_t *scales, const uint8_t *xq,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    extern __shared__ __align__(128) uint8_t lp_a8_dyn[];
    uint8_t *ring = lp_a8_dyn;
    uint8_t *sring = lp_a8_dyn + LP_A8_RING; /* [2][BM + BN][chunk] */

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t gid = lane >> 2;
    const uint32_t tig = lane & 3u;
    const uint32_t wm = warp & 1u; /* 64 activation rows */
    const uint32_t wn = warp >> 1; /* 32 outputs */

    const uint32_t out_tiles = gridDim.x;
    const uint32_t row_tiles = gridDim.y;
    const uint32_t lin = blockIdx.y * out_tiles + blockIdx.x;
    const uint32_t per_group = LP_A8_GROUP_TILES * row_tiles;
    const uint32_t first = (lin / per_group) * LP_A8_GROUP_TILES;
    const uint32_t gsize = min(out_tiles - first, LP_A8_GROUP_TILES);
    const uint32_t in_group = lin % per_group;
    const uint32_t o0 = (first + in_group % gsize) * LP_A8_BN;
    const uint32_t r0 = (in_group / gsize) * LP_A8_BM;
    const uint32_t nk = in_dim / LP_A8_SK;
    const uint32_t nsg = in_dim / LP_A8_GROUP;

    const uint8_t *x_scales = xq + (uint64_t) rows * in_dim;

    /* Codes: every thread owns chunk `tid & 3` of activation row `tid / 4`
     * and `tid / 4 + 64`, and the same of weight rows. */
    const uint32_t cc = tid & 3u;
    const uint32_t cr = tid >> 2;
    const uint8_t *a_src[2];
    const uint8_t *b_src[2];
    bool a_ok[2], b_ok[2];
#pragma unroll
    for (uint32_t i = 0; i < 2u; ++i) {
        const uint32_t r = r0 + cr + 64u * i;
        const uint32_t o = o0 + cr + 64u * i;
        a_ok[i] = r < rows;
        b_ok[i] = o < out_dim;
        a_src[i] = xq + (uint64_t) (a_ok[i] ? r : 0u) * in_dim + cc * 16u;
        b_src[i] = codes + (uint64_t) (b_ok[i] ? o : 0u) * in_dim + cc * 16u;
    }
    /* Scales: thread `t` owns the 16-byte chunk of smem row `t` - activation
     * rows first, then weight rows. A row past the matrix scales by one. */
    const bool s_act = tid < LP_A8_BM;
    const uint32_t s_r = s_act ? r0 + tid : o0 + (tid - LP_A8_BM);
    const bool s_ok = s_act ? s_r < rows : s_r < out_dim;
    const uint8_t *s_src = s_act ? x_scales + (uint64_t) (s_ok ? s_r : 0u) * nsg
                                 : scales + (uint64_t) (s_ok ? s_r : 0u) * nsg;
    if (!s_ok) {
#pragma unroll
        for (uint32_t b = 0; b < 2u; ++b) {
            *(uint4 *) (sring + (b * (LP_A8_BM + LP_A8_BN) + tid) * LP_A8_SCALE_CHUNK) =
                    make_uint4(0x7F7F7F7Fu, 0x7F7F7F7Fu, 0x7F7F7F7Fu, 0x7F7F7F7Fu);
        }
    }

    auto issue = [&](uint32_t kt) {
        uint8_t *st = ring + (kt % LP_A8_STAGES) * LP_A8_STAGE;
        const uint32_t kb = kt * LP_A8_SK;
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t r = cr + 64u * i;
            lp_cp16(st + lp_off64(r, cc), a_src[i] + kb, a_ok[i]);
            lp_cp16(st + LP_A8_TILE + lp_off64(r, cc), b_src[i] + kb, b_ok[i]);
        }
        if (kt % LP_A8_SCALE_STAGES == 0u && s_ok) {
            const uint32_t chunk = kt / LP_A8_SCALE_STAGES;
            lp_cp16(sring + ((chunk & 1u) * (LP_A8_BM + LP_A8_BN) + tid) * LP_A8_SCALE_CHUNK,
                    s_src + chunk * LP_A8_SCALE_CHUNK, true);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        }
    }

#pragma unroll
    for (uint32_t s = 0; s + 1u < LP_A8_STAGES; ++s) {
        if (s < nk) {
            issue(s);
        }
        lp_commit();
    }
    /* Two stages an iteration, so the scale byte a step selects - which
     * alternates between the register's low and high pair - is a constant. */
    for (uint32_t kk = 0; kk < nk; kk += 2u) {
#pragma unroll
    for (uint32_t hh = 0; hh < 2u; ++hh) {
        const uint32_t kt = kk + hh;
        lp_wait<LP_A8_STAGES - 2u>();
        __syncthreads();
        /* The slot stage `kt - 1` read, which every warp has passed; and a
         * scale chunk goes into the buffer the chunk before last used, whose
         * last stage every warp finished well before this barrier. */
        if (kt + LP_A8_STAGES - 1u < nk) {
            issue(kt + LP_A8_STAGES - 1u);
        }
        lp_commit();

        const uint8_t *st = ring + (kt % LP_A8_STAGES) * LP_A8_STAGE;
        const uint32_t sa_base = lp_smem(st);
        const uint32_t sb_base = lp_smem(st + LP_A8_TILE);
        const uint32_t chunk = kt / LP_A8_SCALE_STAGES;
        const uint8_t *ss = sring + (chunk & 1u) * (LP_A8_BM + LP_A8_BN) * LP_A8_SCALE_CHUNK;
        /* The register of four groups this stage's two fall in: its low pair
         * on an even stage, its high pair on an odd one. */
        const uint32_t word = ((kt % LP_A8_SCALE_STAGES) * 2u) / 4u * 4u;
        const uint32_t sel0 = 2u * hh;

        uint32_t sa[4], sb[4];
#pragma unroll
        for (uint32_t mt = 0; mt < 4u; ++mt) {
            const uint32_t r = wm * 64u + mt * 16u + (lane >> 2) + (lane & 1u) * 8u;
            sa[mt] = *(const uint32_t *) (ss + r * LP_A8_SCALE_CHUNK + word);
        }
#pragma unroll
        for (uint32_t nt = 0; nt < 4u; ++nt) {
            const uint32_t n = wn * 32u + nt * 8u + (lane >> 2);
            sb[nt] = *(const uint32_t *) (ss + (LP_A8_BM + n) * LP_A8_SCALE_CHUNK + word);
        }
#pragma unroll
        for (uint32_t j = 0; j < LP_A8_SK / LP_A8_GROUP; ++j) {
            uint32_t a[4][4];
            uint32_t b[4][2];
#pragma unroll
            for (uint32_t mt = 0; mt < 4u; ++mt) {
                const uint32_t r = wm * 64u + mt * 16u + (lane & 15u);
                lp_ldsm4(a[mt], sa_base + lp_off64(r, 2u * j + (lane >> 4)));
            }
#pragma unroll
            for (uint32_t np = 0; np < 2u; ++np) {
                uint32_t t[4];
                const uint32_t n = wn * 32u + np * 16u + (lane & 7u) + ((lane >> 4) << 3);
                lp_ldsm4(t, sb_base + lp_off64(n, 2u * j + ((lane >> 3) & 1u)));
                b[2u * np][0] = t[0];
                b[2u * np][1] = t[1];
                b[2u * np + 1u][0] = t[2];
                b[2u * np + 1u][1] = t[3];
            }
#pragma unroll
            for (uint32_t mt = 0; mt < 4u; ++mt) {
#pragma unroll
                for (uint32_t nt = 0; nt < 4u; ++nt) {
                    if (sel0 + j == 0u) {
                        lp_mma_fp8<0>(acc[mt][nt], a[mt], b[nt], sa[mt], sb[nt]);
                    } else if (sel0 + j == 1u) {
                        lp_mma_fp8<1>(acc[mt][nt], a[mt], b[nt], sa[mt], sb[nt]);
                    } else if (sel0 + j == 2u) {
                        lp_mma_fp8<2>(acc[mt][nt], a[mt], b[nt], sa[mt], sb[nt]);
                    } else {
                        lp_mma_fp8<3>(acc[mt][nt], a[mt], b[nt], sa[mt], sb[nt]);
                    }
                }
            }
        }
    }
    }
    /* c0, c1 at (row gid, outputs 2 tig, +1); c2, c3 eight rows down. */
#pragma unroll
    for (uint32_t mt = 0; mt < 4u; ++mt) {
#pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t r = r0 + wm * 64u + mt * 16u + gid + h * 8u;
            if (r < rows) {
                float *dst = out + (uint64_t) r * out_dim;
#pragma unroll
                for (uint32_t nt = 0; nt < 4u; ++nt) {
                    const uint32_t o = o0 + wn * 32u + nt * 8u + tig * 2u;
                    const float v0 = acc[mt][nt][2u * h];
                    const float v1 = acc[mt][nt][2u * h + 1u];
                    if ((out_dim & 1u) == 0u && o + 1u < out_dim) {
                        *(float2 *) (dst + o) = make_float2(v0, v1);
                    } else {
                        if (o < out_dim) {
                            dst[o] = v0;
                        }
                        if (o + 1u < out_dim) {
                            dst[o + 1u] = v1;
                        }
                    }
                }
            }
        }
    }
}

#endif /* SHISO_CUDA_HAVE_MXF4 */

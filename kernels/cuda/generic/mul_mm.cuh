/* The prefill GEMM, once per weight codec.
 *
 * Nothing here is copied. What is preserved from the Metal twin in
 * `crates/shiso-metal/metal/qwen35/gemm_q4k.metal` is the contract: the entry
 * name, the argument block, the buffer order, the 64-by-32 output tile, the
 * 32-wide k tile, the 128-thread block, the 8192 bytes of threadgroup memory
 * and the guarded epilogue. It also preserves where the arithmetic rounds -
 * both staged tiles are `half` and the accumulator is `float`, exactly as the
 * Metal instantiation's `S0`/`S1`/`simdgroup_float8x8` triple makes them - so
 * the two devices narrow the same values at the same points.
 *
 * # What is NOT preserved, said plainly
 *
 * The order the products are summed in. Metal accumulates through
 * `simdgroup_multiply_accumulate` over 8x8 fragments, whose internal order is
 * the device's rather than the kernel's; this reads the two staged tiles
 * directly and sums along k in index order. Both are float sums of the same
 * half products and neither is the oracle's order either, which is why the
 * shared case grades this at `TOL_QUANT` rather than by equality - and why the
 * fixture it grades with is exact in binary, so that for that fixture every
 * order agrees to the last bit.
 *
 * # The two function constants are gone, and that is the whole point
 *
 * `FC_mul_mm_bc_inp` is false and `FC_mul_mm_bc_out` is true at every launch
 * the engine makes - `launch::mm_consts` is one fixed table - so this compiles
 * the one variant: no input-tail broadcast, and the output tile always guarded.
 * `mm_consts_for(by_name)` hands this backend an empty constant table for the
 * same reason. A row width that is not a whole number of 32 would need the
 * input tail the host never asks for, and no quantized codec here can have one:
 * `q4_0` and `q8_0` pack 32 elements to a block and `q4_K` packs 256, so a row
 * of any of the three is a whole number of 32-wide k tiles by construction.
 */
#ifndef SHISO_MUL_MM_CUH
#define SHISO_MUL_MM_CUH

#include "generic.cuh"

/* Output rows one tile covers. */
#define SHISO_MM_NR0 64
/* Tokens one tile covers. */
#define SHISO_MM_NR1 32
/* Input elements one pass of the k loop covers: one q4_0 block exactly, and the
 * tile width every other codec stages into as well. */
#define SHISO_MM_NK SHISO_QK4_0
/* Threads per block, mirroring `shiso_engine::launch::GEMM_BLOCK`. */
#define SHISO_MM_BLOCK 128

/* Stages sixteen weights of row `w_row`'s `kb`-th k tile, `w_half` picking
 * which sixteen. One codec per specialization; everything else about the GEMM
 * is the same product read the same way.
 *
 * The staged value is `half` for every codec, because that is what the Metal
 * template's `S0` is and where it rounds is part of the contract. */
template <typename codec>
__device__ __forceinline__ void shiso_mm_stage_weights(
        const struct shiso_args_mul_mm &args, const char *__restrict__ src0,
        uint64_t offset0, int r0, int w_row, int w_half, int kb, __half *sa);

/* q4_0: four four-element dequantize calls, so each block is read by two
 * threads rather than dequantized sixteen times over. */
template <>
__device__ __forceinline__ void shiso_mm_stage_weights<struct shiso_block_q4_0>(
        const struct shiso_args_mul_mm &args, const char *__restrict__ src0,
        uint64_t offset0, int r0, int w_row, int w_half, int kb, __half *sa) {
    const struct shiso_block_q4_0 *x =
        (const struct shiso_block_q4_0 *) (src0 + args.nb01 * (uint64_t) (r0 + w_row)
                                           + offset0)
        + kb;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float4 chunk;
        shiso_dequantize_q4_0_t4(x, 4 * w_half + c, &chunk);
        __half *slot = sa + w_row * SHISO_MM_NK + 16 * w_half + 4 * c;
        slot[0] = __float2half(chunk.x);
        slot[1] = __float2half(chunk.y);
        slot[2] = __float2half(chunk.z);
        slot[3] = __float2half(chunk.w);
    }
}

/* q4_K: one sixteen-element group of one 256-weight super-block.
 *
 * Which group is the whole of the mapping, and it is the Metal twin's. There,
 * `nl` is 16, `il` starts at `tiitg % 2` and advances by two per k tile until
 * it wraps, and the block pointer advances on the wrap - so k tile `kb` and
 * half `w_half` read element `kb * 32 + 16 * w_half` onward, which is super-block
 * `(kb * 32 + 16 * w_half) / 256` and group `((kb * 32 + 16 * w_half) % 256) / 16`.
 * Spelled as that arithmetic rather than as a walking pointer because there is
 * no loop here to walk one in, and because the two have to agree exactly: a
 * group index one out is every weight of the tile read from the wrong scale
 * pair, which is plausible and wrong. */
template <>
__device__ __forceinline__ void shiso_mm_stage_weights<struct shiso_block_q4_K>(
        const struct shiso_args_mul_mm &args, const char *__restrict__ src0,
        uint64_t offset0, int r0, int w_row, int w_half, int kb, __half *sa) {
    const int e0 = kb * SHISO_MM_NK + 16 * w_half;
    const struct shiso_block_q4_K *x =
        (const struct shiso_block_q4_K *) (src0 + args.nb01 * (uint64_t) (r0 + w_row)
                                           + offset0)
        + e0 / SHISO_QK_K;
    float tmp[16];
    shiso_dequantize_q4_K_16(x, (e0 % SHISO_QK_K) / 16, tmp);
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        sa[w_row * SHISO_MM_NK + 16 * w_half + c] = __float2half(tmp[c]);
    }
}

/* q8_0: one half of one 32-weight block, which is the `nl` of 2 the Metal
 * instantiation takes. The k tile and the block are both 32 elements wide, so
 * tile `kb` is block `kb` exactly and `w_half` is the dequantizer's `il`. */
template <>
__device__ __forceinline__ void shiso_mm_stage_weights<struct shiso_block_q8_0>(
        const struct shiso_args_mul_mm &args, const char *__restrict__ src0,
        uint64_t offset0, int r0, int w_row, int w_half, int kb, __half *sa) {
    const struct shiso_block_q8_0 *x =
        (const struct shiso_block_q8_0 *) (src0 + args.nb01 * (uint64_t) (r0 + w_row)
                                           + offset0)
        + kb;
    float tmp[16];
    shiso_dequantize_q8_0_16(x, w_half, tmp);
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        sa[w_row * SHISO_MM_NK + 16 * w_half + c] = __float2half(tmp[c]);
    }
}

/* f16: the row is already half, so staging is a copy. `nb01` is the row stride
 * in bytes and carries the element width with it. */
template <>
__device__ __forceinline__ void shiso_mm_stage_weights<__half>(
        const struct shiso_args_mul_mm &args, const char *__restrict__ src0,
        uint64_t offset0, int r0, int w_row, int w_half, int kb, __half *sa) {
    const __half *x = (const __half *) (src0 + args.nb01 * (uint64_t) (r0 + w_row) + offset0)
                    + (uint64_t) kb * SHISO_MM_NK + 16 * w_half;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        sa[w_row * SHISO_MM_NK + 16 * w_half + c] = x[c];
    }
}

template <typename codec>
__device__ __forceinline__ void shiso_mul_mm_f32_impl(
        const struct shiso_args_mul_mm args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    /* `shiso_engine::launch::GEMM_SMEM_BYTES` is 8192, which is the larger of
     * the two claims on this allocation: the two staged tiles need 6144 and the
     * guarded epilogue needs the whole of it as floats. */
    extern __shared__ char shiso_mm_shmem[];
    __half *sa = (__half *) shiso_mm_shmem;                       /* [NR0][NK] */
    __half *sb = (__half *) (shiso_mm_shmem + SHISO_MM_NR0 * SHISO_MM_NK * 2);

    const int tiitg = (int) threadIdx.x;
    const int im = (int) blockIdx.z;
    const int r0 = (int) blockIdx.y * SHISO_MM_NR0;
    const int r1 = (int) blockIdx.x * SHISO_MM_NR1;

    const int nr0 = (args.ne0 - r0 < SHISO_MM_NR0) ? (args.ne0 - r0) : SHISO_MM_NR0;
    const int nr1 = (args.ne1 - r1 < SHISO_MM_NR1) ? (args.ne1 - r1) : SHISO_MM_NR1;

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    const uint64_t offset0 = (uint64_t) (i12 / args.r2) * args.nb02
                           + (uint64_t) (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = (uint64_t) i13 * args.nb13 + (uint64_t) i12 * args.nb12;

    /* Weight staging: one thread per half of a row, four four-element chunks
     * each, so all 128 threads load and each q4_0 block is read by two of them
     * rather than dequantized twice over by sixteen. */
    const int w_row = tiitg / 2;
    const int w_half = tiitg % 2;
    /* Activation staging: one thread per quarter of a token's k tile. */
    const int a_tok = tiitg / 4;
    const int a_quarter = tiitg % 4;

    /* Output assignment: 2048 values over 128 threads is sixteen each, taken as
     * one output row by sixteen consecutive tokens, which is what makes the
     * inner loop one staged weight against sixteen staged activations. */
    const int o_row = tiitg % SHISO_MM_NR0;
    const int o_tok0 = (tiitg / SHISO_MM_NR0) * 16;

    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc[j] = 0.0f;
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += SHISO_MM_NK) {
        const int kb = loop_k / SHISO_MM_NK;

        __syncthreads();

        /* A row past the output extent stages zeros rather than a duplicate of
         * the last valid row. The epilogue discards those columns either way,
         * and zeros are the version that reads nothing past the allocation. */
        if (w_row < nr0) {
            shiso_mm_stage_weights<codec>(args, src0, offset0, r0, w_row, w_half, kb, sa);
        } else {
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                sa[w_row * SHISO_MM_NK + 16 * w_half + c] = __float2half(0.0f);
            }
        }

        if (a_tok < nr1) {
            const float *y = (const float *) (src1 + offset1
                                              + args.nb11 * (uint64_t) (r1 + a_tok)
                                              + args.nb10 * (uint64_t) (loop_k + 8 * a_quarter));
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                sb[a_tok * SHISO_MM_NK + 8 * a_quarter + c] = __float2half(y[c]);
            }
        } else {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                sb[a_tok * SHISO_MM_NK + 8 * a_quarter + c] = __float2half(0.0f);
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < SHISO_MM_NK; ++k) {
            const float wv = __half2float(sa[o_row * SHISO_MM_NK + k]);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                acc[j] += wv * __half2float(sb[(o_tok0 + j) * SHISO_MM_NK + k]);
            }
        }
    }

    if (r0 + o_row >= args.ne0) {
        return;
    }
    float *c = (float *) dst + (uint64_t) im * (uint64_t) args.ne1 * (uint64_t) args.ne0
             + (uint64_t) (r0 + o_row);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int tok = o_tok0 + j;
        if (tok < nr1) {
            c[(uint64_t) (r1 + tok) * (uint64_t) args.ne0] = acc[j];
        }
    }
}

/* One `.visible .entry` per codec. The names are the host's: `launch::mm_name`
 * writes them as `&'static str` literals and this backend has to answer to the
 * spelling it is handed. */
#define SHISO_MM(SUFFIX, CODEC)                                                         \
    extern "C" __global__ __launch_bounds__(SHISO_MM_BLOCK) void                        \
    kernel_mul_mm_##SUFFIX##_f32(const struct shiso_args_mul_mm args,                   \
                                 const char *__restrict__ src0,                         \
                                 const char *__restrict__ src1,                         \
                                 char *__restrict__ dst) {                              \
        shiso_mul_mm_f32_impl<CODEC>(args, src0, src1, dst);                            \
    }

#endif /* SHISO_MUL_MM_CUH */

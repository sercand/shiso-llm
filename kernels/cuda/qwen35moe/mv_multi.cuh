/* The FP8 and NVFP4 projections at a few tokens: one weight read, several rows.
 *
 * `kernel_q35moe_mv_{fp8,nvfp4}_f32` put the token on the grid's y axis, so a
 * pass of `n` rows reads every weight `n` times. That is the right shape at one
 * row and the wrong one at a speculative verify, which runs a draft block of
 * eight rows through every projection: measured on the dense 27B export, a
 * verify step cost 380 ms against 94 ms for a one-row decode step, and
 * speculation ran slower than not speculating at all.
 *
 * These entries read each weight once per `T` tokens and keep `T` partial sums
 * per lane. **Each token's sum is folded in exactly the order the one-token
 * entry folds it** - the same lane owns the same groups (or words), the same
 * `part` is formed and fused into the accumulator the same way, and the same
 * warp and block folds close it out. A row's answer is therefore the same bits
 * whether it ran alone or beside seven others, which is what keeps a verify
 * pass's greedy pick equal to the decode step it replaces.
 *
 * `T` is a template argument and the host names the instantiation; a launch
 * covers `ceil(n_tok / T)` token blocks on the grid's y axis, and a block's
 * trailing tokens past `n_tok` are neither read nor written.
 */
#ifndef SHISO_Q35MOE_MV_MULTI_CUH
#define SHISO_Q35MOE_MV_MULTI_CUH

#include "q35moe_common.cuh"

/* `shiso_e4m3`, branch-free and to the bit: the seven magnitude bits shifted
 * into an `f32`'s exponent and mantissa fields read as a value 2^120 too small
 * - subnormals included, since this unit does not flush - and the multiply by
 * 2^120 is exact. The all-ones magnitude is the format's NaN, which the shift
 * would read as 480, so it is selected out exactly as the reference decoder
 * returns it. Measured: the branchy decoder was most of the FP8 entries' issue
 * slots at eight tokens. */
__device__ __forceinline__ float shiso_e4m3_exact_fast(unsigned byte) {
    const unsigned mag = byte & 0x7Fu;
    const float v = __int_as_float((int) (((byte & 0x80u) << 24) | (mag << 20))) * 0x1p120f;
    return (mag == 0x7Fu) ? __int_as_float(0x7fffffff) : v;
}

/* `shiso_e2m1_fast`, by the same construction: three magnitude bits into the
 * exponent and mantissa fields, sign to the top, and an exact 2^126. */
__device__ __forceinline__ float shiso_e2m1_shift(unsigned code) {
    return __int_as_float((int) (((code & 8u) << 28) | ((code & 7u) << 22))) * 0x1p126f;
}

/* The eight-weight chain `shiso_nvfp4_word_dot` folds, over already-decoded
 * weights. Same fused multiply-adds in the same order from the same zero, so
 * the same bits. */
__device__ __forceinline__ float shiso_q35moe_dot8(const float *w, float4 a, float4 b) {
    float s = 0.0f;
    s = fmaf(w[0], a.x, s);
    s = fmaf(w[1], a.y, s);
    s = fmaf(w[2], a.z, s);
    s = fmaf(w[3], a.w, s);
    s = fmaf(w[4], b.x, s);
    s = fmaf(w[5], b.y, s);
    s = fmaf(w[6], b.z, s);
    s = fmaf(w[7], b.w, s);
    return s;
}

/* `T` tokens of the NVFP4 matvec: grid `(ceil(out_dim / (W * R)), ceil(n_tok /
 * T))` for a block of `W` warps, `R` rows per warp.
 *
 * A lane owns the groups the one-token entry gives it - `lane`, `lane + 32`, and
 * so on - so one turn of the group loop is one tile of 32 groups across the
 * warp, and every warp of the block is on the same tile at the same turn. The
 * block stages that tile's activations for all `T` tokens in shared memory
 * once and every warp reads them from there: without it each warp streamed
 * the whole activation block out of L2 on its own, which at an `in_dim` of
 * 17408 was 700 MB of L2 traffic to read a 50 MB weight and ran at 79 GB/s.
 * Per group, a lane decodes each of its rows' sixteen weights once and folds
 * every token into every row.
 *
 * The tile is laid out `[t][quarter][lane]` in `float4`s, so a warp's sixteen-
 * byte loads are lane-contiguous and conflict-free.
 *
 * The narrow path (a width that is not a whole number of groups, or a
 * misaligned operand) calls the one-token dot per row and token: the same
 * arithmetic by construction, and no export here takes it. */
template <int T, int R>
__device__ __forceinline__ void shiso_q35moe_mv_nvfp4_multi(
        const struct shiso_args_q35moe_nvfp4 args,
        const unsigned char *__restrict__ weights,
        const unsigned char *__restrict__ scales,
        const float *__restrict__ src,
        float *__restrict__ dst) {
    __shared__ float4 tile[T][4][SHISO_Q35MOE_WARP];

    const unsigned lane = threadIdx.x & 31u;
    const int row0 = (int) (blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u)) * R;
    const int tok0 = (int) blockIdx.y * T;
    if (tok0 >= args.n_tok) {
        return;
    }
    const int live = min(T, args.n_tok - tok0);
    /* Rows this warp owns; zero for a warp wholly past the matrix, which still
     * stages its share of every tile so the block's barriers line up. */
    const int rows = max(0, min(R, args.out_dim - row0));
    const int64_t packed_width = (int64_t) args.in_dim / SHISO_NVFP4_PER_BYTE;
    const int64_t scale_width = (int64_t) args.in_dim / SHISO_NVFP4_GROUP;
    const unsigned char *packed = weights + (int64_t) min(row0, args.out_dim - 1) * packed_width;
    const unsigned char *rs = scales + (int64_t) min(row0, args.out_dim - 1) * scale_width;
    const float *x0 = src + (int64_t) tok0 * (int64_t) args.in_dim;

    const bool wide = (args.in_dim % SHISO_NVFP4_GROUP) == 0
                   && ((uintptr_t) weights & 7u) == 0u
                   && ((uintptr_t) src & 15u) == 0u
                   && (packed_width % 8) == 0;
    if (!wide) {
        for (int r = 0; r < rows; ++r) {
            for (int t = 0; t < live; ++t) {
                const float sum = shiso_q35moe_nvfp4_row_dot(
                        packed + (int64_t) r * packed_width, rs + (int64_t) r * scale_width,
                        args.global_scale, x0 + (int64_t) t * args.in_dim, args.in_dim,
                        lane);
                if (lane == 0u) {
                    dst[(int64_t) (tok0 + t) * (int64_t) args.out_dim + row0 + r] = sum;
                }
            }
        }
        return;
    }

    float acc[R][T];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    const int groups = args.in_dim / SHISO_NVFP4_GROUP;
    const int slots = T * 4 * SHISO_Q35MOE_WARP;
    /* The next tile's weight words and scale bytes are loaded before this
     * tile is computed, so the DRAM wait of one turn overlaps the arithmetic
     * of the previous one instead of following the barrier it sits behind. */
    uint2 next_word[R];
    unsigned next_scale[R];
    const auto fetch = [&](int g) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            /* A warp's trailing rows past `out_dim` read the last real row
             * and are never written, so no load leaves the matrix. */
            const int rr = min(r, max(rows, 1) - 1);
            if (g < groups && rows > 0) {
                next_word[r] = ((const uint2 *) (packed + (int64_t) rr * packed_width))[g];
                next_scale[r] = rs[(int64_t) rr * scale_width + g];
            }
        }
    };
    fetch((int) lane);
    for (int base = 0; base < groups; base += SHISO_Q35MOE_WARP) {
        /* Stage: slot `(t, quarter, l)` holds elements `[4 * quarter, +4)` of
         * group `base + l` for token `t`. */
        __syncthreads();
        for (int i = (int) threadIdx.x; i < slots; i += (int) blockDim.x) {
            const int t = i / (4 * SHISO_Q35MOE_WARP);
            const int quarter = (i / SHISO_Q35MOE_WARP) % 4;
            const int l = i % SHISO_Q35MOE_WARP;
            const int g = base + l;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (t < live && g < groups) {
                v = ((const float4 *) (x0 + (int64_t) t * args.in_dim
                                       + (int64_t) g * SHISO_NVFP4_GROUP))[quarter];
            }
            tile[t][quarter][l] = v;
        }
        uint2 word[R];
        unsigned scale_byte[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            word[r] = next_word[r];
            scale_byte[r] = next_scale[r];
        }
        fetch(base + SHISO_Q35MOE_WARP + (int) lane);
        __syncthreads();

        const int g = base + (int) lane;
        if (g >= groups || rows == 0) {
            continue;
        }
        float w[R][SHISO_NVFP4_GROUP];
        float scale[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            scale[r] = shiso_e4m3_exact_fast(scale_byte[r]) * args.global_scale;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                w[r][j] = shiso_e2m1_shift(word[r].x >> (4u * (unsigned) j));
                w[r][8 + j] = shiso_e2m1_shift(word[r].y >> (4u * (unsigned) j));
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            if (t < live) {
                const float4 a = tile[t][0][lane];
                const float4 b = tile[t][1][lane];
                const float4 c = tile[t][2][lane];
                const float4 d = tile[t][3][lane];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float part = shiso_q35moe_dot8(w[r], a, b)
                                     + shiso_q35moe_dot8(w[r] + 8, c, d);
                    acc[r][t] = fmaf(part, scale[r], acc[r][t]);
                }
            }
        }
    }
    if (rows == 0) {
        return;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            /* Every lane folds every slot, live or not, so the shuffle never
             * waits on a lane that skipped it. */
            const float sum = shiso_q35moe_warp_sum(acc[r][t]);
            if (lane == 0u && t < live && r < rows) {
                dst[(int64_t) (tok0 + t) * (int64_t) args.out_dim + row0 + r] = sum;
            }
        }
    }
}

/* `T` tokens of the FP8 matvec: grid `(ceil(out_dim / R), ceil(n_tok / T))`,
 * block 256, `R` rows per block. A thread walks the same words the one-token
 * entry gives it, decodes each of its rows' word once, and loads each token's
 * four activations once for all `R` rows; the byte fallback and the block fold
 * are the one-token entry's, per row and token. */
template <int T, int R>
__device__ __forceinline__ void shiso_q35moe_mv_fp8_multi(
        const struct shiso_args_q35moe_fp8 args,
        const unsigned char *__restrict__ weights,
        const float *__restrict__ src,
        float *__restrict__ dst) {
    __shared__ float partials[R][T][SHISO_Q35MOE_WARPS];

    const int row0 = (int) blockIdx.x * R;
    const int tok0 = (int) blockIdx.y * T;
    if (row0 >= args.out_dim || tok0 >= args.n_tok) {
        return;
    }
    const int live = min(T, args.n_tok - tok0);
    const int rows = min(R, args.out_dim - row0);
    const float *x0 = src + (int64_t) tok0 * (int64_t) args.in_dim;
    const unsigned char *wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        wrow[r] = weights + (int64_t) (row0 + min(r, rows - 1)) * (int64_t) args.in_dim;
    }

    float acc[R][T];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    const bool vec = ((uintptr_t) src & 15u) == 0u;
    const int quads = ((args.in_dim & 3) == 0) ? args.in_dim / 4 : 0;
    for (int q = (int) threadIdx.x; q < quads; q += SHISO_Q35MOE_BLOCK) {
        float decoded[R][4];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned word = ((const unsigned *) wrow[r])[q];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                decoded[r][j] = shiso_e4m3_exact_fast(word >> (8u * (unsigned) j));
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            if (t < live) {
                const float *xq = x0 + (int64_t) t * args.in_dim + q * 4;
                float xv[4];
                if (vec) {
                    const float4 v = *(const float4 *) xq;
                    xv[0] = v.x;
                    xv[1] = v.y;
                    xv[2] = v.z;
                    xv[3] = v.w;
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        xv[j] = xq[j];
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[r][t] = fmaf(decoded[r][j], xv[j], acc[r][t]);
                    }
                }
            }
        }
    }
    for (int k = quads * 4 + (int) threadIdx.x; k < args.in_dim; k += SHISO_Q35MOE_BLOCK) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float decoded = shiso_e4m3_exact_fast(wrow[r][k]);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                if (t < live) {
                    acc[r][t] = fmaf(decoded, x0[(int64_t) t * args.in_dim + k], acc[r][t]);
                }
            }
        }
    }

    /* `shiso_q35moe_block_sum`, once per slot and in its order: each warp
     * folds by shuffle, lane zero parks the warp's sum, and warp zero folds
     * the eight parked sums by the same shuffle ladder. */
    const unsigned lane = threadIdx.x % SHISO_Q35MOE_WARP;
    const unsigned warp = threadIdx.x / SHISO_Q35MOE_WARP;
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float folded = shiso_q35moe_warp_sum(acc[r][t]);
            if (lane == 0u) {
                partials[r][t][warp] = folded;
            }
        }
    }
    __syncthreads();
    if (warp != 0u) {
        return;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float total = (lane < SHISO_Q35MOE_WARPS) ? partials[r][t][lane] : 0.0f;
#pragma unroll
            for (int offset = SHISO_Q35MOE_WARPS / 2; offset > 0; offset >>= 1) {
                total += __shfl_xor_sync(0xffffffffu, total, offset, SHISO_Q35MOE_WARP);
            }
            if (lane == 0u && t < live && r < rows) {
                dst[(int64_t) (tok0 + t) * (int64_t) args.out_dim + row0 + r] =
                        total * args.weight_scale;
            }
        }
    }
}

#define SHISO_Q35MOE_MV_MULTI(t, r)                                            \
    extern "C" __global__ void kernel_q35moe_mv_nvfp4_f32_t##t##_r##r(         \
            const struct shiso_args_q35moe_nvfp4 args,                         \
            const unsigned char *__restrict__ weights,                         \
            const unsigned char *__restrict__ scales,                          \
            const float *__restrict__ src,                                     \
            float *__restrict__ dst) {                                         \
        shiso_q35moe_mv_nvfp4_multi<t, r>(args, weights, scales, src, dst);    \
    }                                                                          \
    extern "C" __global__ void kernel_q35moe_mv_fp8_f32_t##t##_r##r(           \
            const struct shiso_args_q35moe_fp8 args,                           \
            const unsigned char *__restrict__ weights,                         \
            const float *__restrict__ src,                                     \
            float *__restrict__ dst) {                                         \
        shiso_q35moe_mv_fp8_multi<t, r>(args, weights, src, dst);              \
    }

/* Four rows a warp for NVFP4 and two a block for FP8 are the measured best at
 * eight tokens on the dense 27B export's shapes (`[17408, 5120]`,
 * `[5120, 17408]`, `[10240, 5120]`): NVFP4 163-185 GB/s, FP8 194-219 GB/s,
 * against 210-265 GB/s for the one-token entries reading the same bytes. The
 * entry pairs share a macro, so each instantiation exists for both codecs. */
SHISO_Q35MOE_MV_MULTI(2, 2)
SHISO_Q35MOE_MV_MULTI(2, 4)
SHISO_Q35MOE_MV_MULTI(4, 2)
SHISO_Q35MOE_MV_MULTI(4, 4)
SHISO_Q35MOE_MV_MULTI(8, 2)
SHISO_Q35MOE_MV_MULTI(8, 4)

/* The whole weight decoded into an `f32` plane, `out_dim` rows of `in_dim`,
 * for the vendor multiply a prefill chunk takes.
 *
 * Grid `(out_dim, 1, 1)`, block 256, one row per block. Each value is decoded
 * exactly as the matvec decodes it before its product - the FP8 scale applied
 * to the weight here rather than to the finished sum, which is the one place
 * the two routes round differently, and the reason a prefill chunk and a decode
 * step are not bit-comparable on any codec that takes the vendor route. */
extern "C" __global__ void kernel_q35moe_dequant_plane_fp8_f32(
        const struct shiso_args_q35moe_fp8 args,
        const unsigned char *__restrict__ weights,
        float *__restrict__ dst) {
    const int64_t row = (int64_t) blockIdx.x;
    if (row >= args.out_dim) {
        return;
    }
    const unsigned char *w = weights + row * (int64_t) args.in_dim;
    float *out = dst + row * (int64_t) args.in_dim;
    for (int k = (int) threadIdx.x; k < args.in_dim; k += (int) blockDim.x) {
        out[k] = shiso_e4m3(w[k]) * args.weight_scale;
    }
}

extern "C" __global__ void kernel_q35moe_dequant_plane_nvfp4_f32(
        const struct shiso_args_q35moe_nvfp4 args,
        const unsigned char *__restrict__ weights,
        const unsigned char *__restrict__ scales,
        float *__restrict__ dst) {
    const int64_t row = (int64_t) blockIdx.x;
    if (row >= args.out_dim) {
        return;
    }
    const unsigned char *packed = weights + row * (int64_t) (args.in_dim / SHISO_NVFP4_PER_BYTE);
    const unsigned char *rs = scales + row * (int64_t) (args.in_dim / SHISO_NVFP4_GROUP);
    float *out = dst + row * (int64_t) args.in_dim;
    for (int k = (int) threadIdx.x; k < args.in_dim; k += (int) blockDim.x) {
        out[k] = shiso_nvfp4_weight(packed, rs, k, args.global_scale);
    }
}

#endif /* SHISO_Q35MOE_MV_MULTI_CUH */

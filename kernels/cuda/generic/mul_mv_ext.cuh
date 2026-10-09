/* The extended matvec: fifteen entry points per weight codec, for what is one
 * entry point and two function constants on Metal.
 *
 * Nothing here is copied. The decomposition is the Metal twin's in
 * `crates/shiso-metal/metal/generic/mul_mv_ext.metal` - the same lane split,
 * the same four-chunks-per-thread inner loop, the same reduction order and the
 * same guarded store - because the host names these kernels itself and hands
 * them the same argument block and the same grid. What changes is that the two
 * constants become template parameters and each instantiation gets its own
 * `.visible .entry`, which is what a module with one symbol per instantiation
 * requires.
 *
 * # The two widths are separate arguments and the names carry both
 *
 * `r1ptg` is how many token rows one threadgroup takes and `nxpsg` is how many
 * lanes split the input. The engine picks them independently
 * (`ext_r1ptg(n_tok)` and `ext_nxpsg(in_dim, n_tok)`), so fifteen pairs are
 * reachable and all fifteen are instantiated. A pair that is merely absent is
 * invisible at any single width: the model loads, answers at one prompt length
 * and refuses at another.
 *
 * # One body per walk, one codec parameter
 *
 * The Metal side is two templates over a block struct and a dequantizer: the
 * four-element walk (`kernel_mul_mv_ext_q4_f32_disp`) and the whole-sub-block
 * walk (`kernel_mul_mv_ext_kq_f32_disp`). These are the same two templates
 * with the same two, spelled as traits structs because a CUDA entry point
 * cannot be a template instantiation and the macros at the bottom have to
 * name a codec. The q4_0 instantiations are graded at every one of the
 * fifteen pairs by `crates/shiso-cuda/tests/dense_q4_0.rs`, which is what
 * keeps a change made for the second codec from moving the first, and the
 * nine `kq` codecs at every pair by `crates/shiso-cuda/tests/dense_iq.rs`.
 *
 * # What the loop assumes about the row width
 *
 * Each iteration reads `chpt` chunks spaced `nxpsg` chunks apart while the loop
 * condition tests only the first of them, so the row has to be a whole number
 * of `4 * chpt * nxpsg` elements. That is exactly the rule `ext_nxpsg` enforces
 * on the host - 16 lanes only for a width divisible by 256, 8 only for 128 -
 * and it is stated here because a kernel launched at a width that breaks it
 * reads into the next row and says nothing.
 */
#ifndef SHISO_MUL_MV_EXT_CUH
#define SHISO_MUL_MV_EXT_CUH

#include "generic.cuh"

/* Simdgroups per threadgroup, mirroring `shiso_engine::launch::EXT_NSG`. The
 * block is exactly this many warps wide and the row arithmetic below depends on
 * it: a wider block would run lanes no output row accounts for. */
#define SHISO_EXT_NSG 2
/* Chunks one thread takes per turn of the outer loop. */
#define SHISO_EXT_CHPT 4

/* The q4_0 codec: an 18-byte block of 32 weights, read four at a time. */
struct shiso_mv_codec_q4_0 {
    typedef struct shiso_block_q4_0 block_t;
    static constexpr int chunks_per_block = SHISO_Q4_0_CHUNKS;
    static __device__ __forceinline__ void t4(const block_t *xb, int il, float4 *reg) {
        shiso_dequantize_q4_0_t4(xb, il, reg);
    }
};

/* The q6_K codec: a 210-byte block of 256 weights, read four at a time. Used by
 * the output head, whose weight is the vocabulary table a Gemma 4 GGUF packs at
 * q6_K and ties the head to. */
struct shiso_mv_codec_q6_K {
    typedef struct shiso_block_q6_K block_t;
    static constexpr int chunks_per_block = SHISO_Q6_K_CHUNKS;
    static __device__ __forceinline__ void t4(const block_t *xb, int il, float4 *reg) {
        shiso_dequantize_q6_K_t4(xb, il, reg);
    }
};

/* The q4_K codec: a 144-byte block of 256 weights, read four at a time. The
 * bulk of every qwen35 linear layer - 131 tensors of a `Q4_K_M` container - and
 * the only way any of them is read below two tokens, since
 * `crates/shiso-engine/src/projection.rs` takes the GEMM at two and above and
 * this family at one. */
struct shiso_mv_codec_q4_K {
    typedef struct shiso_block_q4_K block_t;
    static constexpr int chunks_per_block = SHISO_Q4_K_CHUNKS;
    static __device__ __forceinline__ void t4(const block_t *xb, int il, float4 *reg) {
        shiso_dequantize_q4_K_t4(xb, il, reg);
    }
};

/* The q5_K codec: a 176-byte block of 256 weights, read four at a time. Used
 * by the qwen35 container's 48 q5_K tensors, which the generic set has no other
 * path to. */
struct shiso_mv_codec_q5_K {
    typedef struct shiso_block_q5_K block_t;
    static constexpr int chunks_per_block = SHISO_Q5_K_CHUNKS;
    static __device__ __forceinline__ void t4(const block_t *xb, int il, float4 *reg) {
        shiso_dequantize_q5_K_t4(xb, il, reg);
    }
};

/* The nine mixed i-quant codecs and the `q2_K` one beside them.
 *
 * These do not carry a `chunks_per_block` and a four-element unpack like the
 * four above: all nine blocks are 256 weights and none of them unpacks to a
 * four-element chunk without re-deriving a group's scale and signs nine
 * times over, so they take the `kq` walk below - one whole 32-weight
 * sub-block a lane a turn - exactly as they do on Metal. The `deq32` member
 * is `shiso_deq32_<format>` from `generic.cuh`, which writes the sub-block's
 * 32 dequantized weights as eight `float4` and the pair `(a, b)` such that
 * the sub-block's contribution is `a * sum(q * y) + b * sum(y)`; for the
 * i-quants that pair is `(1, 0)` because the weights are already fully
 * dequantized. */
struct shiso_mv_codec_q2_K {
    typedef struct shiso_block_q2_K block_t;
    static __device__ __forceinline__ void deq32(
            const block_t *xb, int s, float4 *qv, float2 *ab) {
        shiso_deq32_q2_K(xb, s, qv, ab);
    }
};

struct shiso_mv_codec_iq2_xxs {
    typedef struct shiso_block_iq2_xxs block_t;
    static __device__ __forceinline__ void deq32(
            const block_t *xb, int s, float4 *qv, float2 *ab) {
        shiso_deq32_iq2_xxs(xb, s, qv, ab);
    }
};

struct shiso_mv_codec_iq2_xs {
    typedef struct shiso_block_iq2_xs block_t;
    static __device__ __forceinline__ void deq32(
            const block_t *xb, int s, float4 *qv, float2 *ab) {
        shiso_deq32_iq2_xs(xb, s, qv, ab);
    }
};

struct shiso_mv_codec_iq2_s {
    typedef struct shiso_block_iq2_s block_t;
    static __device__ __forceinline__ void deq32(
            const block_t *xb, int s, float4 *qv, float2 *ab) {
        shiso_deq32_iq2_s(xb, s, qv, ab);
    }
};

struct shiso_mv_codec_iq3_xxs {
    typedef struct shiso_block_iq3_xxs block_t;
    static __device__ __forceinline__ void deq32(
            const block_t *xb, int s, float4 *qv, float2 *ab) {
        shiso_deq32_iq3_xxs(xb, s, qv, ab);
    }
};

struct shiso_mv_codec_iq3_s {
    typedef struct shiso_block_iq3_s block_t;
    static __device__ __forceinline__ void deq32(
            const block_t *xb, int s, float4 *qv, float2 *ab) {
        shiso_deq32_iq3_s(xb, s, qv, ab);
    }
};

struct shiso_mv_codec_iq1_s {
    typedef struct shiso_block_iq1_s block_t;
    static __device__ __forceinline__ void deq32(
            const block_t *xb, int s, float4 *qv, float2 *ab) {
        shiso_deq32_iq1_s(xb, s, qv, ab);
    }
};

struct shiso_mv_codec_iq1_m {
    typedef struct shiso_block_iq1_m block_t;
    static __device__ __forceinline__ void deq32(
            const block_t *xb, int s, float4 *qv, float2 *ab) {
        shiso_deq32_iq1_m(xb, s, qv, ab);
    }
};

struct shiso_mv_codec_iq4_xs {
    typedef struct shiso_block_iq4_xs block_t;
    static __device__ __forceinline__ void deq32(
            const block_t *xb, int s, float4 *qv, float2 *ab) {
        shiso_deq32_iq4_xs(xb, s, qv, ab);
    }
};


template <typename codec, int r1ptg, int nxpsg>
__device__ __forceinline__ void shiso_mul_mv_ext_f32_impl(
        const struct shiso_args_mul_mv_ext args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    constexpr int chpb  = codec::chunks_per_block;
    constexpr int chpt  = SHISO_EXT_CHPT;
    constexpr int nypsg = 32 / nxpsg;

    const int tiisg = (int) threadIdx.x;
    const int sgitg = (int) threadIdx.y;

    const int tx = tiisg % nxpsg;
    const int ty = tiisg / nxpsg;

    const int i01 = (int) blockIdx.x * (nypsg * SHISO_EXT_NSG) + nypsg * sgitg + ty;
    const int i11 = (int) blockIdx.y * r1ptg;
    const int i1m = (int) blockIdx.z;

    const int i12 = i1m % args.ne12;
    const int i13 = i1m / args.ne12;

    const uint64_t offset0 = (uint64_t) i01 * args.nb01
                           + (uint64_t) (i12 / args.r2) * args.nb02
                           + (uint64_t) (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = (uint64_t) i11 * args.nb11
                           + (uint64_t) i12 * args.nb12
                           + (uint64_t) i13 * args.nb13;

    /* A row past the output extent still runs, reading row zero, because the
     * reduction below is a whole-warp operation and a lane that returned early
     * would take its partial sums out of it. Only the store is guarded. */
    typedef typename codec::block_t block_t;
    const block_t *xq = (i01 < args.ne01)
                            ? (const block_t *) (src0 + offset0) + tx / chpb
                            : (const block_t *) src0;

    const float4 *y4[r1ptg];
#pragma unroll
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11)
                      ? (const float4 *) (src1 + offset1 + (uint64_t) ir1 * args.nb11) + tx
                      : (const float4 *) src1;
    }

    float sumf[r1ptg];
#pragma unroll
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        sumf[ir1] = 0.0f;
    }

    int cch = tx % chpb;

    for (int ich = tx; 4 * ich < args.ne00; ich += chpt * nxpsg) {
        float4 lx[chpt];

#pragma unroll
        for (int ch = 0; ch < chpt; ++ch) {
            codec::t4(xq, cch, &lx[ch]);
            cch += nxpsg;
            if (cch >= chpb) {
                xq += cch / chpb;
                cch %= chpb;
            }
        }

#pragma unroll
        for (int ch = 0; ch < chpt; ++ch) {
#pragma unroll
            for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
                const float4 yv = y4[ir1][ch * nxpsg];
                sumf[ir1] += lx[ch].x * yv.x + lx[ch].y * yv.y + lx[ch].z * yv.z
                           + lx[ch].w * yv.w;
            }
        }

#pragma unroll
        for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt * nxpsg;
        }
    }

    /* Only the lanes of one row, and in the order the Metal twin folds them:
     * a golden that pins a decode step pins the order its partial sums were
     * added in. The width argument keeps every shuffle inside its own row's
     * lane group, which the guards below would already achieve - it is there so
     * the segmentation is stated rather than inferred. */
#pragma unroll
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 16, nxpsg);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 8, nxpsg);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 4, nxpsg);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 2, nxpsg);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 1, nxpsg);
        }
    }

    if (tx != 0 || i01 >= args.ne01) {
        return;
    }
#pragma unroll
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (i11 + ir1 < args.ne11) {
            float *dst_f32 = (float *) dst
                           + (uint64_t) i1m * (uint64_t) args.ne0 * (uint64_t) args.ne1
                           + (uint64_t) (i11 + ir1) * (uint64_t) args.ne0;
            dst_f32[i01] = sumf[ir1];
        }
    }
}

/* The `kq` walk: one whole 32-weight sub-block a lane a turn.
 *
 * The same grid, the same lane split, the same argument block and the same
 * reduction as `shiso_mul_mv_ext_f32_impl` above, so the host cannot tell
 * the two apart beyond the entry name. What differs is the unit a lane
 * takes. The four-element walk above unpacks four weights per turn, and for
 * the mixed i-quants that means re-deriving a group's grid indices, signs
 * and scale nine times over its 32 weights; the Metal twin measured the same
 * shape for the K-quants and moved to the whole-sub-block walk for it. Here
 * a lane takes sub-block `u`, `deq32` unpacks its 32 weights once, and the
 * fold is `a * sum(q * y) + b * sum(y)` in the order the Metal twin folds
 * them - `((qy.x + qy.y) + (qy.z + qy.w))` after eight sequential
 * component-wise adds - because a golden that pins a decode step pins the
 * order its partial sums were added in.
 *
 * Every block this family reads is 256 weights, eight sub-blocks of 32, so
 * the walk needs no `chunks_per_block`: sub-block `u` lives at block `u / 8`
 * and field `u % 8`, and the row's sub-block count is `ne00 / 32` - a whole
 * number for every width `ext_nxpsg` accepts. */
template <typename codec, int r1ptg, int nxpsg>
__device__ __forceinline__ void shiso_mul_mv_ext_kq_f32_impl(
        const struct shiso_args_mul_mv_ext args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    constexpr int nypsg = 32 / nxpsg;

    const int tiisg = (int) threadIdx.x;
    const int sgitg = (int) threadIdx.y;

    const int tx = tiisg % nxpsg;
    const int ty = tiisg / nxpsg;

    const int i01 = (int) blockIdx.x * (nypsg * SHISO_EXT_NSG) + nypsg * sgitg + ty;
    const int i11 = (int) blockIdx.y * r1ptg;
    const int i1m = (int) blockIdx.z;

    const int i12 = i1m % args.ne12;
    const int i13 = i1m / args.ne12;

    const uint64_t offset0 = (uint64_t) i01 * args.nb01
                           + (uint64_t) (i12 / args.r2) * args.nb02
                           + (uint64_t) (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = (uint64_t) i11 * args.nb11
                           + (uint64_t) i12 * args.nb12
                           + (uint64_t) i13 * args.nb13;

    /* A row past the output extent still runs, reading row zero, because the
     * reduction below is a whole-warp operation and a lane that returned
     * early would take its partial sums out of it. Only the store is
     * guarded. Unlike the four-element walk, the lane's first sub-block is
     * `tx` rather than a chunk offset into the block, so `xq` starts at the
     * row itself. */
    typedef typename codec::block_t block_t;
    const block_t *xq = (i01 < args.ne01)
                            ? (const block_t *) (src0 + offset0)
                            : (const block_t *) src0;

    const float4 *y4[r1ptg];
#pragma unroll
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11)
                      ? (const float4 *) (src1 + offset1 + (uint64_t) ir1 * args.nb11)
                      : (const float4 *) src1;
    }

    float sumf[r1ptg];
#pragma unroll
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        sumf[ir1] = 0.0f;
    }

    const int nsub = args.ne00 / 32;

    for (int u = tx; u < nsub; u += nxpsg) {
        float4 qv[8];
        float2 ab;
        codec::deq32(xq + u / 8, u % 8, qv, &ab);
        const float *qf = (const float *) qv;

#pragma unroll
        for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
            const float4 *y = y4[ir1] + u * 8;
            float qy[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            float sy[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float4 yk = y[k];
                qy[0] += qf[4 * k + 0] * yk.x;
                qy[1] += qf[4 * k + 1] * yk.y;
                qy[2] += qf[4 * k + 2] * yk.z;
                qy[3] += qf[4 * k + 3] * yk.w;
                sy[0] += yk.x;
                sy[1] += yk.y;
                sy[2] += yk.z;
                sy[3] += yk.w;
            }
            sumf[ir1] += ab.x * ((qy[0] + qy[1]) + (qy[2] + qy[3]))
                       + ab.y * ((sy[0] + sy[1]) + (sy[2] + sy[3]));
        }
    }

    /* Only the lanes of one row, and in the order the Metal twin folds
     * them, exactly as `shiso_mul_mv_ext_f32_impl` states. */
#pragma unroll
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 16, nxpsg);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 8, nxpsg);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 4, nxpsg);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 2, nxpsg);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += __shfl_down_sync(0xffffffffu, sumf[ir1], 1, nxpsg);
        }
    }

    if (tx != 0 || i01 >= args.ne01) {
        return;
    }
#pragma unroll
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (i11 + ir1 < args.ne11) {
            float *dst_f32 = (float *) dst
                           + (uint64_t) i1m * (uint64_t) args.ne0 * (uint64_t) args.ne1
                           + (uint64_t) (i11 + ir1) * (uint64_t) args.ne0;
            dst_f32[i01] = sumf[ir1];
        }
    }
}

/* One `.visible .entry` per pair. The names are the host's - the engine writes
 * them as `&'static str` literals in `launch::EXT_Q4_0_BY_NAME`,
 * `EXT_Q4_K_BY_NAME`, `EXT_Q5_K_BY_NAME` and `EXT_Q6_K_BY_NAME` - so the
 * spelling here is not ours to choose. */
#define SHISO_MV_EXT_Q4_0(R, NX)                                                        \
    extern "C" __global__ void kernel_mul_mv_ext_q4_0_f32_r1_##R##_nx##NX(              \
            const struct shiso_args_mul_mv_ext args, const char *__restrict__ src0,     \
            const char *__restrict__ src1, char *__restrict__ dst) {                    \
        shiso_mul_mv_ext_f32_impl<shiso_mv_codec_q4_0, R, NX>(args, src0, src1, dst);   \
    }

#define SHISO_MV_EXT_Q6_K(R, NX)                                                        \
    extern "C" __global__ void kernel_mul_mv_ext_q6_K_f32_r1_##R##_nx##NX(              \
            const struct shiso_args_mul_mv_ext args, const char *__restrict__ src0,     \
            const char *__restrict__ src1, char *__restrict__ dst) {                    \
        shiso_mul_mv_ext_f32_impl<shiso_mv_codec_q6_K, R, NX>(args, src0, src1, dst);   \
    }

#define SHISO_MV_EXT_Q4_K(R, NX)                                                        \
    extern "C" __global__ void kernel_mul_mv_ext_q4_K_f32_r1_##R##_nx##NX(              \
            const struct shiso_args_mul_mv_ext args, const char *__restrict__ src0,     \
            const char *__restrict__ src1, char *__restrict__ dst) {                    \
        shiso_mul_mv_ext_f32_impl<shiso_mv_codec_q4_K, R, NX>(args, src0, src1, dst);   \
    }

#define SHISO_MV_EXT_Q5_K(R, NX)                                                        \
    extern "C" __global__ void kernel_mul_mv_ext_q5_K_f32_r1_##R##_nx##NX(              \
            const struct shiso_args_mul_mv_ext args, const char *__restrict__ src0,     \
            const char *__restrict__ src1, char *__restrict__ dst) {                    \
        shiso_mul_mv_ext_f32_impl<shiso_mv_codec_q5_K, R, NX>(args, src0, src1, dst);   \
    }

/* The `kq` walk's macro, one per format and pair. The format tag is the
 * first argument and appears in the entry name verbatim - `q2_K` keeps the
 * capital `K` the Metal twin was named with - and selects the codec struct
 * `shiso_mv_codec_##FMT` beside it, so a name in `launch::EXT_Q2_K_BY_NAME`
 * and the instantiation that answers it are spelled from one token. */
#define SHISO_MV_EXT_KQ(FMT, R, NX)                                                     \
    extern "C" __global__ void kernel_mul_mv_ext_##FMT##_f32_r1_##R##_nx##NX(           \
            const struct shiso_args_mul_mv_ext args, const char *__restrict__ src0,     \
            const char *__restrict__ src1, char *__restrict__ dst) {                    \
        shiso_mul_mv_ext_kq_f32_impl<shiso_mv_codec_##FMT, R, NX>(                      \
                args, src0, src1, dst);                                                 \
    }

#endif /* SHISO_MUL_MV_EXT_CUH */

/* The tower's projections over weights the checkpoint stored quantized:
 * MXFP8 (E4M3 codes, one power-of-two `ue8m0` scale per 32 weights) and NVFP4
 * (E2M1 codes two a byte, one E4M3 scale per 16 weights, one f32 scale per
 * tensor), each read as stored.
 *
 * Weight-only: the activation side is the split operand the BLAS path reads
 * (`kernel_vit_to_bf16` and its fused producers), two brain-float planes whose
 * sum is the f32 row to about sixteen bits. A tile of codes is decoded in
 * shared memory to the brain floats it stands for - exactly, for both formats:
 * an E4M3 code times a power of two, and an E2M1 code times an E4M3 scale, are
 * products of at most six significant bits - and multiplied against both planes
 * on the matrix units, each plane into its own f32 accumulator, the two added
 * once at the end. That is the arithmetic of the two BLAS calls this replaces
 * over the weights decoded once at load, with one difference: an NVFP4 tensor's
 * f32 scale multiplies the finished sum here, where the load-time decode folded
 * it into every weight and rounded the product to a brain float.
 *
 * Nothing here is copied. The pipeline is `q4e_mxfp8_gemm_kernel`'s
 * (`kernels/cuda/q4e/mxfp8.cu`): raw codes by `cp.async` three steps deep, the
 * copying thread decoding its own piece into a swizzled tile so its own wait
 * covers the copy, one barrier a step. Its activation side is simpler: the
 * planes are brain floats already, so they are copied straight into the ring
 * and read from it by `ldmatrix`.
 */
#ifndef SHISO_VIT_QUANT_CUH
#define SHISO_VIT_QUANT_CUH

#include <cuda_bf16.h>

#include "vit_common.cuh"

/* Two tilings of `out[row][o]`, each warp 32 outputs by 32 patch rows, one
 * 32-wide contraction step at a time:
 *   wide   128 outputs x 128 patch rows, sixteen warps (512 threads);
 *   narrow  64 outputs x  64 patch rows, four warps (128 threads),
 * the narrow one for a projection whose wide grid would leave the device's
 * multiprocessors idle - a small picture's 196 patch rows into 1152 outputs
 * is eighteen wide tiles on 48 SMs. The host picks by the grid alone. */
#define VIT_Q_BK 32u
/* Steps in flight behind the one being multiplied. */
#define VIT_Q_STAGES 3u
/* Output tiles a raster group spans: the blocks resident at once share a few
 * weight tiles and a few activation tiles. */
#define VIT_Q_GROUP_TILES 16u
/* Dynamic shared memory of a `BM x BN` tile: the ring (raw codes - 32 MXFP8
 * codes a row, or 32 NVFP4 codes in half the bytes - the high plane, the low
 * plane) and two decoded weight tiles. The wide tile's is above the 48 KiB
 * default, which the backend raises for the entry on its first launch. */
#define VIT_Q_SMEM_OF(BM, BN) \
    (VIT_Q_STAGES * ((BM) * VIT_Q_BK + 2u * (BN) * VIT_Q_BK * 2u) + 2u * (BM) * VIT_Q_BK * 2u)
#define VIT_Q_SMEM VIT_Q_SMEM_OF(128u, 128u)
#define VIT_Q_SMEM_NARROW VIT_Q_SMEM_OF(64u, 64u)

/* `vit_args_qmm`: the projection's extents and an NVFP4 tensor's own scale
 * (one for MXFP8). */
struct vit_args_qmm {
    uint32_t m;     /* patch rows */
    uint32_t n;     /* outputs, the weight's rows */
    uint32_t k;     /* contraction, the weight's columns */
    float global;   /* multiplies the finished sum */
};
static_assert(sizeof(struct vit_args_qmm) == 16, "vit_args_qmm is 16 bytes");

/* The two formats, as the type the kernel body is templated on. */
struct vit_q_mxfp8 {
    static constexpr bool NVFP4 = false;
};
struct vit_q_nvfp4 {
    static constexpr bool NVFP4 = true;
};

/* `cp.async` of `BYTES` with a source size: `valid` false copies nothing and
 * writes zeros, which is how a row or a contraction step past the matrix
 * stages - every slot, every time, since a slot is reused. */
__device__ __forceinline__ void vit_q_cp16(void *smem, const void *gmem, bool valid) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    const uint32_t n = valid ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void vit_q_cp8(void *smem, const void *gmem, bool valid) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    const uint32_t n = valid ? 8u : 0u;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void vit_q_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void vit_q_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void vit_q_ldsm4(uint32_t *r, const void *p) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

/* `d += a * b` over one 16x8x16 brain-float fragment, f32 accumulate. */
__device__ __forceinline__ void vit_q_mma(float *d, const uint32_t *a, const uint32_t *b) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* Byte offset of sixteen-byte chunk `c` (0..3) of row `r` in a tile of 64-byte
 * rows, swizzled so the eight rows one `ldmatrix` phase reads land on eight
 * distinct bank groups without padding. */
__device__ __forceinline__ uint32_t vit_q_off(uint32_t r, uint32_t c) {
    return r * 64u + ((c ^ ((r >> 1) & 3u)) << 4);
}

/* Two floats as two brain floats, the first low. Every value reaching this is
 * a brain float already, so the rounding never moves a bit. */
__device__ __forceinline__ uint32_t vit_q_pack(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}

/* An E4M3 byte's value times 2^-120: its seven magnitude bits placed under
 * the top of an f32's exponent field. Exact for normals and subnormals alike -
 * both formats' subnormal steps line up at that offset - and the two NaN codes
 * come out as the finite 480 * 2^-120 instead (a checkpoint holds none). */
__device__ __forceinline__ float vit_q_e4m3_lo(uint32_t byte) {
    return __uint_as_float(((byte & 0x80u) << 24) | ((byte & 0x7Fu) << 20));
}

/* A `ue8m0` scale byte: `2^(byte - 127)`, byte 0 being the one subnormal. */
__device__ __forceinline__ float vit_q_ue8m0(uint32_t byte) {
    return byte == 0u ? __uint_as_float(1u << 22) : __uint_as_float(byte << 23);
}

/* An E2M1 nibble's value times 2^-126: its three magnitude bits placed at the
 * bottom of an f32's exponent field, exactly as `vit_q_e4m3_lo` does. */
__device__ __forceinline__ float vit_q_e2m1_lo(uint32_t nib) {
    return __uint_as_float(((nib & 0x8u) << 28) | ((nib & 0x7u) << 22));
}

/* Sixteen MXFP8 codes under one scale, as sixteen brain floats. The code is
 * restored to its value before the scale multiplies it, so no product
 * underflows on the way. */
__device__ __forceinline__ void vit_q_mx16(uint4 q, float scale, uint4 &h0, uint4 &h1) {
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[j] = vit_q_e4m3_lo((w[i] >> (8 * j)) & 0xFFu) * 0x1p120f * scale;
        }
        o[2 * i] = vit_q_pack(v[0], v[1]);
        o[2 * i + 1] = vit_q_pack(v[2], v[3]);
    }
    h0 = make_uint4(o[0], o[1], o[2], o[3]);
    h1 = make_uint4(o[4], o[5], o[6], o[7]);
}

/* Sixteen NVFP4 codes (low nibble first) under one E4M3 scale, as sixteen
 * brain floats: `e2m1 * e4m3`, the tensor's f32 scale left for the sum. */
__device__ __forceinline__ void vit_q_fp16(uint2 q, float scale, uint4 &h0, uint4 &h1) {
    const uint32_t w[2] = {q.x, q.y};
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float a = vit_q_e2m1_lo((w[i] >> (8 * j)) & 0xFu) * 0x1p126f * scale;
            const float b = vit_q_e2m1_lo((w[i] >> (8 * j + 4)) & 0xFu) * 0x1p126f * scale;
            o[4 * i + j] = vit_q_pack(a, b);
        }
    }
    h0 = make_uint4(o[0], o[1], o[2], o[3]);
    h1 = make_uint4(o[4], o[5], o[6], o[7]);
}

/* `out[row][o] = global * sum_k w[o][k] (hi[row][k] + lo[row][k])`.
 *
 * `planes` is the split operand: `m * k` brain floats of the nearest values,
 * then `m * k` of what they left over. `codes` and `scales` are the weight as
 * the checkpoint stores it, `[n][k]` and `[n][k / 32]` bytes for MXFP8,
 * `[n][k / 2]` and `[n][k / 16]` for NVFP4.
 *
 * `k` a multiple of 32 for MXFP8 and of 16 for NVFP4: a step past the end of
 * a ragged NVFP4 row (the tower's second feed-forward layer contracts over
 * 4304 = 134.5 steps) stages zero codes and zero activations, which add
 * nothing. An NVFP4 row of `k / 2` bytes need only be eight-byte aligned,
 * which `k % 16 == 0` gives; the planes' rows need sixteen, which `k % 8 == 0`
 * gives.
 *
 * Grid `(ceil(n / BM), ceil(m / BN))`, `BM * BN / 32` threads,
 * `VIT_Q_SMEM_OF(BM, BN)` bytes of dynamic shared memory. */
template <typename F, uint32_t BM, uint32_t BN>
__device__ __forceinline__ void vit_qmm_body(
        const struct vit_args_qmm args,
        float *__restrict__ out,
        const uint16_t *__restrict__ planes,
        const uint8_t *__restrict__ codes,
        const uint8_t *__restrict__ scales) {
    extern __shared__ __align__(16) uint8_t vit_q_dyn[];
    constexpr uint32_t RAW = BM * VIT_Q_BK;       /* raw code bytes a step */
    constexpr uint32_t A_TILE = BM * VIT_Q_BK * 2u; /* one decoded weight tile */
    constexpr uint32_t B_TILE = BN * VIT_Q_BK * 2u; /* one plane's tile */
    constexpr uint32_t WARPS_M = BM / 32u;
    constexpr uint32_t THREADS = BM * BN / 32u;
    uint8_t *a_raw = vit_q_dyn;                    /* [STAGES][RAW] */
    uint8_t *b_hi = a_raw + VIT_Q_STAGES * RAW;    /* [STAGES] tiles */
    uint8_t *b_lo = b_hi + VIT_Q_STAGES * B_TILE;  /* [STAGES] tiles */
    uint8_t *a16 = b_lo + VIT_Q_STAGES * B_TILE;   /* [2] tiles */

    const uint32_t m = args.m;
    const uint32_t n = args.n;
    const uint32_t k = args.k;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t gid = lane >> 2;
    const uint32_t tig = lane & 3u;
    const uint32_t wm = warp % WARPS_M; /* 32 outputs */
    const uint32_t wn = warp / WARPS_M; /* 32 patch rows */

    const uint32_t out_tiles = gridDim.x;
    const uint32_t row_tiles = gridDim.y;
    const uint32_t lin = blockIdx.y * out_tiles + blockIdx.x;
    const uint32_t per_group = VIT_Q_GROUP_TILES * row_tiles;
    const uint32_t first = (lin / per_group) * VIT_Q_GROUP_TILES;
    const uint32_t gsize = min(out_tiles - first, VIT_Q_GROUP_TILES);
    const uint32_t in_group = lin % per_group;
    const uint32_t o0 = (first + in_group % gsize) * BM;
    const uint32_t r0 = (in_group / gsize) * BN;
    const uint32_t nk = (k + VIT_Q_BK - 1u) / VIT_Q_BK;

    /* Codes: the first `2 BM` threads own sixteen weights of row `tid / 2`,
     * the `tid % 2` half of a step, and the scale that covers them. */
    static_assert(THREADS >= 2u * BM, "a thread for every half row of codes");
    const bool a_on = tid < 2u * BM;
    const uint32_t ar = (tid >> 1) & (BM - 1u);
    const uint32_t ah = tid & 1u;
    const bool a_live = a_on && o0 + ar < n;
    const uint32_t a_row = min(o0 + ar, n - 1u);
    const uint32_t groups = F::NVFP4 ? k / 16u : k / 32u;
    const uint8_t *a_src = codes + (uint64_t) a_row * (F::NVFP4 ? k / 2u : k) + ah * (F::NVFP4 ? 8u : 16u);
    const uint8_t *s_src = scales + (uint64_t) a_row * groups + (F::NVFP4 ? ah : 0u);
    /* Activations: piece `p` is chunk `p % 4` of patch row `p / 4`, in both
     * planes; a thread owns pieces `tid`, `tid + THREADS`, ... */
    constexpr uint32_t B_PIECES = 4u * BN / THREADS;
    static_assert(B_PIECES * THREADS == 4u * BN, "whole activation pieces a thread");
    const uint64_t plane = (uint64_t) m * k;

    /* Whether this thread's sixteen weights of step `kt` are inside the row. */
    auto a_in = [&](uint32_t kt) {
        return a_live && kt * VIT_Q_BK + ah * 16u < k;
    };
    auto issue = [&](uint32_t kt) {
        const uint32_t s = kt % VIT_Q_STAGES;
        const uint32_t k0 = kt * VIT_Q_BK;
        if (a_on) {
            const bool in = a_in(kt);
            if (F::NVFP4) {
                vit_q_cp8(a_raw + s * RAW + tid * 8u, a_src + (in ? k0 / 2u : 0u), in);
            } else {
                vit_q_cp16(a_raw + s * RAW + tid * 16u, a_src + (in ? k0 : 0u), in);
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < B_PIECES; i++) {
            const uint32_t p = tid + i * THREADS;
            const uint32_t bn = p >> 2;
            const uint32_t bc = p & 3u;
            const bool in = r0 + bn < m && k0 + bc * 8u < k;
            const uint16_t *src =
                    planes + (uint64_t) min(r0 + bn, m - 1u) * k + bc * 8u + (in ? k0 : 0u);
            vit_q_cp16(b_hi + s * B_TILE + vit_q_off(bn, bc), src, in);
            vit_q_cp16(b_lo + s * B_TILE + vit_q_off(bn, bc), src + plane, in);
        }
    };
    /* The scale byte of step `kt`, or zero past the row: it multiplies zero
     * codes there. */
    auto scale_of = [&](uint32_t kt) -> uint32_t {
        if (!a_in(kt)) {
            return 0u;
        }
        return __ldg(s_src + (F::NVFP4 ? 2u * kt : kt));
    };
    /* Two steps of scales ride ahead in registers. */
    uint32_t sb0 = a_on && nk > 0u ? scale_of(0u) : 0u;
    uint32_t sb1 = a_on && nk > 1u ? scale_of(1u) : 0u;

    float acc_hi[2][4][4];
    float acc_lo[2][4][4];
#pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
#pragma unroll
            for (uint32_t h = 0; h < 4u; h++) {
                acc_hi[i][j][h] = 0.0f;
                acc_lo[i][j][h] = 0.0f;
            }
        }
    }

#pragma unroll
    for (uint32_t s = 0; s + 1u < VIT_Q_STAGES; s++) {
        if (s < nk) {
            issue(s);
        }
        vit_q_commit();
    }
    for (uint32_t kt = 0; kt < nk; kt++) {
        const uint32_t s = kt % VIT_Q_STAGES;
        const uint32_t ab = kt & 1u;
        vit_q_wait<VIT_Q_STAGES - 2u>();
        /* This thread's codes of step `kt`, decoded into tile `ab`. Its last
         * readers were step `kt - 2`'s, which every warp finished before the
         * previous barrier. */
        if (a_on) {
            uint4 h0, h1;
            if (F::NVFP4) {
                const uint2 q = *reinterpret_cast<const uint2 *>(a_raw + s * RAW + tid * 8u);
                vit_q_fp16(q, vit_q_e4m3_lo(sb0) * 0x1p120f, h0, h1);
            } else {
                const uint4 q = *reinterpret_cast<const uint4 *>(a_raw + s * RAW + tid * 16u);
                vit_q_mx16(q, vit_q_ue8m0(sb0), h0, h1);
            }
            uint8_t *t = a16 + ab * A_TILE;
            *reinterpret_cast<uint4 *>(t + vit_q_off(ar, 2u * ah)) = h0;
            *reinterpret_cast<uint4 *>(t + vit_q_off(ar, 2u * ah + 1u)) = h1;
            sb0 = sb1;
            sb1 = kt + 2u < nk ? scale_of(kt + 2u) : 0u;
        }
        __syncthreads();
        /* The slot step `kt + 2` lands in held step `kt - 1`, which every warp
         * finished multiplying before this barrier. */
        if (kt + VIT_Q_STAGES - 1u < nk) {
            issue(kt + VIT_Q_STAGES - 1u);
        }
        vit_q_commit();

        const uint8_t *as = a16 + ab * A_TILE;
        const uint8_t *bh = b_hi + s * B_TILE;
        const uint8_t *bl = b_lo + s * B_TILE;
#pragma unroll
        for (uint32_t kq = 0; kq < 2u; kq++) {
            uint32_t af[2][4];
            uint32_t bfh[4][2];
            uint32_t bfl[4][2];
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; mt++) {
                const uint32_t r = wm * 32u + mt * 16u + (lane & 15u);
                vit_q_ldsm4(af[mt], as + vit_q_off(r, kq * 2u + (lane >> 4)));
            }
#pragma unroll
            for (uint32_t np = 0; np < 2u; np++) {
                const uint32_t r = wn * 32u + np * 16u + (lane & 7u) + (lane >> 4) * 8u;
                const uint32_t off = vit_q_off(r, kq * 2u + ((lane >> 3) & 1u));
                uint32_t rr[4];
                vit_q_ldsm4(rr, bh + off);
                bfh[2 * np][0] = rr[0];
                bfh[2 * np][1] = rr[1];
                bfh[2 * np + 1][0] = rr[2];
                bfh[2 * np + 1][1] = rr[3];
                vit_q_ldsm4(rr, bl + off);
                bfl[2 * np][0] = rr[0];
                bfl[2 * np][1] = rr[1];
                bfl[2 * np + 1][0] = rr[2];
                bfl[2 * np + 1][1] = rr[3];
            }
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; mt++) {
#pragma unroll
                for (uint32_t nt = 0; nt < 4u; nt++) {
                    vit_q_mma(acc_hi[mt][nt], af[mt], bfh[nt]);
                    vit_q_mma(acc_lo[mt][nt], af[mt], bfl[nt]);
                }
            }
        }
    }
    /* c0, c1 at (output gid, patch rows 2 tig, +1); c2, c3 eight outputs on. */
    const float global = args.global;
#pragma unroll
    for (uint32_t mt = 0; mt < 2u; mt++) {
#pragma unroll
        for (uint32_t nt = 0; nt < 4u; nt++) {
#pragma unroll
            for (uint32_t h = 0; h < 4u; h++) {
                const uint32_t row = r0 + wn * 32u + nt * 8u + tig * 2u + (h & 1u);
                const uint32_t o = o0 + wm * 32u + mt * 16u + gid + (h >> 1) * 8u;
                if (row < m && o < n) {
                    out[(uint64_t) row * n + o] = (acc_hi[mt][nt][h] + acc_lo[mt][nt][h]) * global;
                }
            }
        }
    }
}

/* The four entries: each format at the wide tile and at the narrow one. */
#define VIT_QMM_ENTRY(NAME, FMT, BM, BN, MIN_BLOCKS)                                         \
    extern "C" __global__ void __launch_bounds__((BM) * (BN) / 32u, MIN_BLOCKS) NAME(        \
            const struct vit_args_qmm args, float *__restrict__ out,                         \
            const uint16_t *__restrict__ planes, const uint8_t *__restrict__ codes,          \
            const uint8_t *__restrict__ scales) {                                            \
        vit_qmm_body<FMT, BM, BN>(args, out, planes, codes, scales);                         \
    }
VIT_QMM_ENTRY(kernel_vit_qmm_mxfp8, vit_q_mxfp8, 128u, 128u, 1)
VIT_QMM_ENTRY(kernel_vit_qmm_nvfp4, vit_q_nvfp4, 128u, 128u, 1)
VIT_QMM_ENTRY(kernel_vit_qmm_mxfp8_narrow, vit_q_mxfp8, 64u, 64u, 2)
VIT_QMM_ENTRY(kernel_vit_qmm_nvfp4_narrow, vit_q_nvfp4, 64u, 64u, 2)

#endif /* SHISO_VIT_QUANT_CUH */

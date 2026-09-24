#pragma once

// shiso's decode-shaped dense trellis projection.  Not from exllamav3; it
// reuses exllamav3's trellis decoder, Hadamard helpers and PTX wrappers, and
// the weight-stationary block's A staging (exl3_moe_ws.cuh).
//
// The cooperative exl3_gemm_kernel runs one block an SM whatever the shape,
// so a projection whose output is narrow has a handful of warps streaming
// all of it: the recurrent output projection (6144 -> 2560) reaches 166 GB/s
// and the shared expert's down projection (640 -> 2560) 76 GB/s, on a GB10
// whose weights stream at ~230.  Here a warp owns one n16 column group over
// one K slice, a block is WARPS of those, and the grid is (N / 16 / WARPS,
// ksplit): enough warps in flight for every shape of this checkpoint, sized
// by the host from the shape alone.
//
// Two launches:
//   exl3_dgemv_had   A_had = fp16(Had128(A * suh)), the cooperative kernel's
//                    own input phase (had_fh_r_128_inner), one warp per
//                    (row, 128 columns of K);
//   exl3_dgemv       the slices, each into its own fp32 partial row; the
//                    last block to finish a 128-column output tile (an atomic
//                    ticket per tile) sums the tile's slices in slice order,
//                    applies the output Hadamard and svh, and stores C.  The
//                    sum's order is fixed by the slice index, never by which
//                    block finished first, so a replay is bit-identical.
//
// Per row the arithmetic depends on the shape and ksplit only -- not on how
// many rows share the pass -- so a token decodes to the same bits alone, in a
// verify pass or in a batch.

#include <stdint.h>
#include <cuda_fp16.h>

#include "util.cuh"
#include "ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"

#define DG_KT 64

// Shared memory holds only the trellis tiles: the activation is read straight
// into MMA fragments from A_frag (below), which every block reads and L1/L2
// serve.  Staging a zero-padded 16-row A tile per stage, as the MoE block does,
// took 40% of the smem and capped the SM at eight warps (measured: 166 GB/s on
// the recurrent input projection; this layout lifts the cap).
template <int bits, int STAGES, int WARPS, int MT>
struct DgSmem {
    static constexpr int tile_words = 256 * bits / 16;
    static constexpr int b_stage = (DG_KT / 16) * WARPS * tile_words;   // uint16
    static constexpr int total = STAGES * b_stage * 2;
};

// A_frag: the transformed activation in m16n8k16 A-fragment order.  For m16
// tile mt and k16 block kb, lane l's eight halves (one uint4) are the four
// registers ldmatrix.x4 would give it: rows r = l / 4 and r + 8, columns
// c = 2 (l % 4) and c + 8 of the tile, two halves each.  Rows past size_m are
// written as zeros up to the tile boundary.
__device__ __forceinline__ size_t dg_frag_index(int r, int k, int kb_total) {
    const int mt = r >> 4, rr = r & 15, kb = k >> 4, kq = k & 15;
    const int lane = (rr & 7) * 4 + ((kq & 7) >> 1);
    const int reg = (rr >> 3) + 2 * (kq >> 3);
    return (((size_t)mt * kb_total + kb) * 32 + lane) * 8 + reg * 2 + (kq & 1);
}

// had_fh_r_128_inner's arithmetic (pre-scale, Hadamard, 1/sqrt(128), round),
// stored in fragment order.  One warp per (row, 128 columns of K), over
// size_m rounded up to the tile.
__global__ void __launch_bounds__(256)
exl3_dgemv_had(const float *__restrict__ A, half *__restrict__ A_frag, const half *__restrict__ suh,
               int size_m, int size_k) {
    const int chunks = size_k / 128;
    const int rows = (size_m + 15) & ~15;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (w >= rows * chunks) return;
    const int t = threadIdx.x & 31;
    const int r = w / chunks, c = w - r * chunks;
    half2 lo, hi;
    if (r < size_m) {
        float4 v = ((const float4 *)(A + (size_t)r * size_k + c * 128))[t];
        const half4 scales = ((const half4 *)(suh + c * 128))[t];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
        const float s0 = v.x + v.y, d0 = v.x - v.y, s1 = v.z + v.w, d1 = v.z - v.w;
        v.x = s0 + s1; v.y = d0 + d1; v.z = s0 - s1; v.w = d0 - d1;
        shuffle_had_f2x32(v.x, v.y, t);
        shuffle_had_f2x32(v.z, v.w, t);
        const float rs = 0.088388347648f;
        v.x *= rs; v.y *= rs; v.z *= rs; v.w *= rs;
        lo = __floats2half2_rn(v.x, v.y);
        hi = __floats2half2_rn(v.z, v.w);
    } else {
        lo = hi = __floats2half2_rn(0.0f, 0.0f);
    }
    const int k = c * 128 + t * 4;
    const int kb_total = size_k / 16;
    *(half2 *)(A_frag + dg_frag_index(r, k, kb_total)) = lo;
    *(half2 *)(A_frag + dg_frag_index(r, k + 2, kb_total)) = hi;
}

// Rows [m0, m0 + 16 MT) of one K slice: warp w's two n8 fragments of columns
// [n0 + 16w, n0 + 16w + 16) per tile.
template <int bits, int cb, int STAGES, int WARPS, int MT>
__device__ __forceinline__ void dg_pass(const half *__restrict__ A_frag, int mt0, int kb_total,
                                        int k_lo, int kslice,
                                        const uint16_t *__restrict__ B, int size_n, int n0,
                                        FragC (*c)[2], uint8_t *smem) {
    typedef DgSmem<bits, STAGES, WARPS, MT> S;
    constexpr int TW = S::tile_words;
    constexpr int THREADS = WARPS * 32;
    uint16_t *sh_b = (uint16_t *)smem;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int blocks_n = size_n / 16;
    const int n_stages = kslice / DG_KT;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) { c[mt][0] = {}; c[mt][1] = {}; }

    auto load_stage = [&](int s, int buf) {
        const int k0 = k_lo + s * DG_KT;
        uint16_t *sb = sh_b + buf * S::b_stage;
        constexpr int per_tile = TW * 2 / 16;
        constexpr int pieces = (DG_KT / 16) * WARPS * per_tile;
        for (int p = t; p < pieces; p += THREADS) {
            const int tile = p / per_tile, piece = p - tile * per_tile;
            const int kb = tile / WARPS, nbi = tile % WARPS;
            const size_t src = ((size_t)(k0 / 16 + kb) * blocks_n + n0 / 16 + nbi) * TW;
            cp_async((int4 *)sb + p, (const int4 *)(B + src) + piece);
        }
        cp_async_fence();
    };
    const uint4 *af = (const uint4 *)A_frag;
    auto load_a = [&](int s, uint4 (*a)[DG_KT / 16]) {
        const int kb0 = (k_lo + s * DG_KT) / 16;
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int kb = 0; kb < DG_KT / 16; kb++)
                a[mt][kb] = __ldg(af + ((size_t)(mt0 + mt) * kb_total + kb0 + kb) * 32 + lane);
    };
    auto compute_stage = [&](int buf, uint4 (*a)[DG_KT / 16]) {
        const uint16_t *sb = sh_b + buf * S::b_stage;
#pragma unroll
        for (int kb = 0; kb < DG_KT / 16; kb++) {
            FragB fb[2];
            dq_dispatch<bits, cb>((const uint32_t *)(sb + (kb * WARPS + warp) * TW), lane << 3, fb[0], fb[1]);
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                const FragA &fa = *reinterpret_cast<const FragA *>(&a[mt][kb]);
                ptx_mma_m16n8k16(fa, fb[0], c[mt][0]);
                ptx_mma_m16n8k16(fa, fb[1], c[mt][1]);
            }
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < n_stages) load_stage(s, s); else cp_async_fence();
    }
    uint4 a_cur[MT][DG_KT / 16], a_nxt[MT][DG_KT / 16];
    if (n_stages > 0) load_a(0, a_cur);
    for (int s = 0; s < n_stages; s++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int nxt = s + STAGES - 1;
        if (nxt < n_stages) load_stage(nxt, nxt % STAGES); else cp_async_fence();
        if (s + 1 < n_stages) load_a(s + 1, a_nxt);
        compute_stage(s % STAGES, a_cur);
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int kb = 0; kb < DG_KT / 16; kb++) a_cur[mt][kb] = a_nxt[mt][kb];
    }
    cp_async_wait<0>();
    __syncthreads();
}

// grid (size_n / (16 * WARPS), ksplit).  part holds ksplit * size_m * size_n
// floats; tickets[size_n / 128] start at zero and are left at zero.
template <int bits, int cb, int STAGES, int WARPS, int MT>
__global__ void __launch_bounds__(WARPS * 32)
exl3_dgemv(const half *__restrict__ A_frag, const uint16_t *__restrict__ B, float *__restrict__ C,
           float *__restrict__ part, int *__restrict__ tickets, const half *__restrict__ svh,
           int size_m, int size_k, int size_n, int ksplit) {
    extern __shared__ __align__(16) uint8_t dg_smem[];
    __shared__ int s_last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n0 = blockIdx.x * 16 * WARPS;
    // Slices are whole pipeline stages and may differ by one stage, so the
    // host can pick a slice count that divides the grid evenly over the SMs
    // rather than one that divides K.
    const int ks = blockIdx.y;
    const int stages = size_k / DG_KT;
    const int s_lo = ks * stages / ksplit, s_hi = (ks + 1) * stages / ksplit;
    const int k_lo = s_lo * DG_KT;
    const int kslice = (s_hi - s_lo) * DG_KT;
    const int kb_total = size_k / 16;
    float *my_part = part + (size_t)ks * size_m * size_n;
    for (int m0 = 0; m0 < size_m; m0 += MT * 16) {
        const int rows = min(MT * 16, size_m - m0);
        FragC c[MT][2];
        dg_pass<bits, cb, STAGES, WARPS, MT>(A_frag, m0 / 16, kb_total, k_lo, kslice, B, size_n, n0, c, dg_smem);
        const int r0 = lane >> 2, col = n0 + warp * 16 + (lane & 3) * 2;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
#pragma unroll
            for (int f = 0; f < 2; f++) {
                const int ra = mt * 16 + r0, rb = ra + 8;
                if (ra < rows)
                    __stcg((float2 *)(my_part + (size_t)(m0 + ra) * size_n + col + f * 8),
                           make_float2(c[mt][f].elems[0], c[mt][f].elems[1]));
                if (rb < rows)
                    __stcg((float2 *)(my_part + (size_t)(m0 + rb) * size_n + col + f * 8),
                           make_float2(c[mt][f].elems[2], c[mt][f].elems[3]));
            }
        }
    }

    // The last of the tile's blocks (every slice of every column group in it)
    // finishes the tile.
    constexpr int groups_per_tile = 128 / (16 * WARPS);
    const int tile = n0 / 128;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int prev = atomicAdd(&tickets[tile], 1);
        s_last = prev == groups_per_tile * ksplit - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int c0 = tile * 128;
    for (int r = warp; r < size_m; r += WARPS) {
        float4 v = __ldcg((const float4 *)(part + (size_t)r * size_n + c0) + lane);
        for (int j = 1; j < ksplit; j++) {
            const float4 p = __ldcg((const float4 *)(part + ((size_t)j * size_m + r) * size_n + c0) + lane);
            v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
        }
        const float s0 = v.x + v.y, d0 = v.x - v.y, s1 = v.z + v.w, d1 = v.z - v.w;
        v.x = s0 + s1; v.y = d0 + d1; v.z = s0 - s1; v.w = d0 - d1;
        shuffle_had_f2x32(v.x, v.y, lane);
        shuffle_had_f2x32(v.z, v.w, lane);
        const float rs = 0.088388347648f;
        v.x *= rs; v.y *= rs; v.z *= rs; v.w *= rs;
        const half4 sc = ((const half4 *)(svh + c0))[lane];
        v.x *= __low2float(sc.x); v.y *= __high2float(sc.x);
        v.z *= __low2float(sc.y); v.w *= __high2float(sc.y);
        ((float4 *)(C + (size_t)r * size_n + c0))[lane] = v;
    }
    if (threadIdx.x == 0) tickets[tile] = 0;
}

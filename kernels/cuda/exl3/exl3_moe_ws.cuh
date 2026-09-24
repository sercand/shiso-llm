#pragma once

// shiso's weight-stationary routed-expert prefill for EXL3.  Not from exllamav3;
// it reuses exllamav3's trellis decoder, Hadamard helpers and PTX wrappers.
//
// exl3_moe_kernel streams every expert's weights once per 16 rows of that
// expert's tokens: at a 2048-token chunk an expert holds ~40 rows, so its
// tiles are read and decoded three times, and the cooperative groups pay a
// grid barrier between every phase.  Here a thread block owns one (expert,
// 128-column output tile) and walks K once, decoding each B tile into MMA
// fragments a single time and multiplying it against all of the expert's rows
// in 64-row passes.  No inter-block barriers, no lock chains: the launch is an
// ordinary grid of n_expert * (n / 128) blocks the hardware balances, and a
// block whose expert has no rows exits at once.
//
// Three launches per layer:
//   exl3_moe_ws_prep    x[token] -> Had128(x * suh_g[e]), Had128(x * suh_u[e])
//                       as fp16 rows in expert-sorted order (the routing map's
//                       order, so an expert's rows are contiguous);
//   exl3_moe_ws_gateup  per (expert, 128 of the 640 columns): both GEMMs, then
//                       the epilogue exl3_moe_kernel runs between its GEMMs
//                       (output Hadamard, svh, SiLU gate, suh_d, input
//                       Hadamard) into the fp16 down input;
//   exl3_moe_ws_down    per (expert, 128 of the 2560 columns): the down GEMM
//                       with the output Hadamard and svh_d, stored to the
//                       assignment's own slot_out row -- the per-slot layout
//                       the combine kernel already sums, so a chunk's result
//                       does not depend on which block finished first.
//
// Numerics follow exl3_moe_kernel's fp32 staging path operation for operation
// (fp16 A and B, fp32 accumulation, fp32 Hadamards, fp16 down input); only the
// MMA accumulation order differs.
//
// Measured and not kept (GB10, 2048-token chunk, 9.4 ms/layer as shipped):
// 32-row passes with two blocks per SM (12.6 ms), 32-wide K stages (10.6),
// splitting an expert's passes over several blocks (10.3 at 4), and a variant
// that Hadamard-transformed x inside the gate/up kernel instead of staging
// a_g/a_u (10.9, 218 registers).  The kernel is neither occupancy- nor
// bandwidth-bound at this point; its remaining gap to the 5.4 ms weight floor
// is issue/latency inside the k loop.

#include <stdint.h>
#include <cuda_fp16.h>

#include "util.cuh"
#include "ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"

#define WS_THREADS   256
#ifndef WS_ROWS
#define WS_ROWS      64           // rows per pass: WS_MT m16 tiles
#endif
#define WS_MT        (WS_ROWS / 16)
#ifndef WS_KT
#define WS_KT        64           // K per pipeline stage: four k16 blocks
#endif
#define WS_A_COLS    (WS_KT / 8)  // int4 columns per staged A row
/* XOR swizzle of the int4 column so eight consecutive rows of an ldmatrix hit
 * eight bank groups: rows are 128 B apart at KT = 64 (any row bit works), 64 B
 * apart at KT = 32 (rows 2j and 2j+1 already differ, so use the next bits). */
#define WS_SWZ(R)    (WS_A_COLS == 8 ? ((R) & 7) : (((R) >> 1) & (WS_A_COLS - 1)))

// ---- 128-point Hadamard helpers with explicit scale pointers ---------------
// hadamard_inner.cuh's variants index their scale vector with blockIdx.y;
// these take the 128-element slice directly.  One warp, four values per lane.
__device__ __forceinline__ void ws_had4(float4 &v, int lane) {
    const float s0 = v.x + v.y, d0 = v.x - v.y, s1 = v.z + v.w, d1 = v.z - v.w;
    v.x = s0 + s1; v.y = d0 + d1; v.z = s0 - s1; v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, lane);
    shuffle_had_f2x32(v.z, v.w, lane);
    const float r = 0.088388347648f;
    v.x *= r; v.y *= r; v.z *= r; v.w *= r;
}
__device__ __forceinline__ void ws_scale4(float4 &v, const half *sc, int lane) {
    const half4 s = ((const half4 *)sc)[lane];
    v.x *= __low2float(s.x); v.y *= __high2float(s.x);
    v.z *= __low2float(s.y); v.w *= __high2float(s.y);
}

// ---- prep --------------------------------------------------------------------
// Row i of a_g / a_u is sorted assignment i: token slot_sorted[i] / n_used
// through expert e where bounds[e] <= i < bounds[e+1].  One warp per (row,
// 128-column chunk); gate and up read the same x.
__global__ void __launch_bounds__(WS_THREADS)
exl3_moe_ws_prep(const float *__restrict__ x, half *__restrict__ a_g, half *__restrict__ a_u,
                 const half *__restrict__ suh_g, const half *__restrict__ suh_u,
                 const int32_t *__restrict__ bounds, const int32_t *__restrict__ slot_sorted,
                 int n_rows, int n_expert, int n_used, int k) {
    const int lane = threadIdx.x & 31;
    const int chunks = k / 128;
    const int task = blockIdx.x * (WS_THREADS / 32) + (threadIdx.x >> 5);
    if (task >= n_rows * chunks) return;
    const int row = task / chunks, c = task - row * chunks;
    int lo = 0, hi = n_expert;               // bounds[lo] <= row < bounds[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (bounds[mid] <= row) lo = mid; else hi = mid;
    }
    const int e = lo;
    const int token = slot_sorted[row] / n_used;
    const float4 xv = ((const float4 *)(x + (size_t)token * k + c * 128))[lane];
    float4 g = xv, u = xv;
    ws_scale4(g, suh_g + (size_t)e * k + c * 128, lane);
    ws_scale4(u, suh_u + (size_t)e * k + c * 128, lane);
    ws_had4(g, lane);
    ws_had4(u, lane);
    half4 og(__floats2half2_rn(g.x, g.y), __floats2half2_rn(g.z, g.w));
    half4 ou(__floats2half2_rn(u.x, u.y), __floats2half2_rn(u.z, u.w));
    ((half4 *)(a_g + (size_t)row * k + c * 128))[lane] = og;
    ((half4 *)(a_u + (size_t)row * k + c * 128))[lane] = ou;
}

// ---- prep, by token (shiso) ----------------------------------------------------
// The same rows as exl3_moe_ws_prep, produced the other way round: one warp per
// (token, 128-column chunk) reads the token's input once and writes it to every
// expert row it was routed to, through the sort's inverse.  The per-row form
// reads each token's input once per expert - ten times - which at an
// 8192-row chunk is 839 MB of the 1.7 GB the launch moved.
__global__ void __launch_bounds__(WS_THREADS)
exl3_moe_ws_invert(int32_t *__restrict__ pos, const int32_t *__restrict__ slot_sorted, int n_rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n_rows) pos[slot_sorted[r]] = r;
}

__global__ void __launch_bounds__(WS_THREADS)
exl3_moe_ws_prep_tok(const float *__restrict__ x, half *__restrict__ a_g, half *__restrict__ a_u,
                     const half *__restrict__ suh_g, const half *__restrict__ suh_u,
                     const int32_t *__restrict__ ids, const int32_t *__restrict__ pos,
                     int n_tok, int n_used, int k) {
    const int lane = threadIdx.x & 31;
    const int chunks = k / 128;
    const int task = blockIdx.x * (WS_THREADS / 32) + (threadIdx.x >> 5);
    if (task >= n_tok * chunks) return;
    const int t = task / chunks, c = task - t * chunks;
    const float4 xv = ((const float4 *)(x + (size_t)t * k + c * 128))[lane];
    for (int s = 0; s < n_used; s++) {
        const int slot = t * n_used + s;
        const int e = ids[slot];
        const int row = pos[slot];
        float4 g = xv, u = xv;
        ws_scale4(g, suh_g + (size_t)e * k + c * 128, lane);
        ws_scale4(u, suh_u + (size_t)e * k + c * 128, lane);
        ws_had4(g, lane);
        ws_had4(u, lane);
        ((half4 *)(a_g + (size_t)row * k + c * 128))[lane] = half4(__floats2half2_rn(g.x, g.y), __floats2half2_rn(g.z, g.w));
        ((half4 *)(a_u + (size_t)row * k + c * 128))[lane] = half4(__floats2half2_rn(u.x, u.y), __floats2half2_rn(u.z, u.w));
    }
}

// ---- GEMM core -----------------------------------------------------------------
// C_i[m][n0 + j] = sum_k A_i[m][k] B_i[k][n0 + j] for i in {0, 1} (TWO) or {0}.
// Warp w owns columns [n0 + 16w, n0 + 16w + 16): two n8 fragments per k16
// block, decoded once and applied to the pass's four m16 tiles.  Shared memory
// per stage: each A 64 x 64 halfs (8 KB), XOR-swizzled so both the cp.async
// fill and ldmatrix are bank-conflict free; each B four k16 x eight n16 tiles.
template <int bits, bool TWO, int STAGES>
struct WsSmem {
    static constexpr int tile_words = 256 * bits / 16;               // uint16 per 16x16 tile
    static constexpr int a_stage = WS_ROWS * WS_KT;                  // halfs, per matrix
    static constexpr int b_stage = (WS_KT / 16) * 8 * tile_words;    // uint16, per matrix
    static constexpr int n_mat = TWO ? 2 : 1;
    static constexpr int stage_bytes = n_mat * (a_stage * 2 + b_stage * 2);
    static constexpr int bytes = STAGES * stage_bytes;
    static constexpr int epilogue_bytes = 2 * 32 * 128 * 4;          // two [32][128] fp32 tiles
    static constexpr int total = bytes > epilogue_bytes ? bytes : epilogue_bytes;
};

template <int bits, int cb, bool TWO, int STAGES>
__device__ __forceinline__ void ws_gemm_pass(
        const half *__restrict__ A0, const half *__restrict__ A1, int m_rows,
        const uint16_t *__restrict__ B0, const uint16_t *__restrict__ B1,
        int k, int n, int n0,
        FragC (*c0)[2], FragC (*c1)[2], uint8_t *smem) {
    typedef WsSmem<bits, TWO, STAGES> S;
    constexpr int TW = S::tile_words;
    half *sh_a0 = (half *)smem;
    half *sh_a1 = sh_a0 + STAGES * S::a_stage;
    uint16_t *sh_b0 = (uint16_t *)(sh_a1 + (TWO ? STAGES * S::a_stage : 0));
    uint16_t *sh_b1 = sh_b0 + STAGES * S::b_stage;

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int blocks_n = n / 16;
    const int n_stages = k / WS_KT;

#pragma unroll
    for (int mt = 0; mt < WS_MT; mt++)
#pragma unroll
        for (int f = 0; f < 2; f++) { c0[mt][f] = {}; if (TWO) c1[mt][f] = {}; }

    auto load_a = [&](const half *A, half *sh, int k0) {
#pragma unroll
        for (int i = 0; i < WS_ROWS * WS_A_COLS / WS_THREADS; i++) {
            const int idx = i * WS_THREADS + t;
            const int m = idx / WS_A_COLS, kc = idx % WS_A_COLS;
            int4 *dst = (int4 *)sh + m * WS_A_COLS + (kc ^ WS_SWZ(m));
            if (m < m_rows) cp_async(dst, (const int4 *)(A + (size_t)m * k + k0) + kc);
            else *dst = make_int4(0, 0, 0, 0);
        }
    };
    auto load_b = [&](const uint16_t *B, uint16_t *sh, int k0) {
        constexpr int per_tile = TW * 2 / 16;                    // 16-byte pieces per tile
        constexpr int pieces = (WS_KT / 16) * 8 * per_tile;
        for (int p = t; p < pieces; p += WS_THREADS) {
            const int tile = p / per_tile, piece = p - tile * per_tile;
            const int kb = tile >> 3, nbi = tile & 7;
            const size_t src = ((size_t)(k0 / 16 + kb) * blocks_n + n0 / 16 + nbi) * TW;
            cp_async((int4 *)sh + p, (const int4 *)(B + src) + piece);
        }
    };
    auto load_stage = [&](int s, int buf) {
        const int k0 = s * WS_KT;
        load_a(A0, sh_a0 + buf * S::a_stage, k0);
        load_b(B0, sh_b0 + buf * S::b_stage, k0);
        if (TWO) {
            load_a(A1, sh_a1 + buf * S::a_stage, k0);
            load_b(B1, sh_b1 + buf * S::b_stage, k0);
        }
        cp_async_fence();
    };

    auto compute_stage = [&](int buf) {
        const half *a0 = sh_a0 + buf * S::a_stage, *a1 = sh_a1 + buf * S::a_stage;
        const uint16_t *b0 = sh_b0 + buf * S::b_stage, *b1 = sh_b1 + buf * S::b_stage;
        const int r = (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
        for (int kb = 0; kb < WS_KT / 16; kb++) {
            FragB fb0[2], fb1[2];
            dq_dispatch<bits, cb>((const uint32_t *)(b0 + (kb * 8 + warp) * TW), lane << 3, fb0[0], fb0[1]);
            if (TWO) dq_dispatch<bits, cb>((const uint32_t *)(b1 + (kb * 8 + warp) * TW), lane << 3, fb1[0], fb1[1]);
            const int c = kb * 2 + (lane >> 4);
#pragma unroll
            for (int mt = 0; mt < WS_MT; mt++) {
                const int R = mt * 16 + r;
                const int off = R * WS_A_COLS + (c ^ WS_SWZ(R));
                FragA fa;
                ldsm4(fa, (const int4 *)a0 + off);
                ptx_mma_m16n8k16(fa, fb0[0], c0[mt][0]);
                ptx_mma_m16n8k16(fa, fb0[1], c0[mt][1]);
                if (TWO) {
                    ldsm4(fa, (const int4 *)a1 + off);
                    ptx_mma_m16n8k16(fa, fb1[0], c1[mt][0]);
                    ptx_mma_m16n8k16(fa, fb1[1], c1[mt][1]);
                }
            }
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < n_stages) load_stage(s, s); else cp_async_fence();
    }
    for (int s = 0; s < n_stages; s++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int nxt = s + STAGES - 1;
        if (nxt < n_stages) load_stage(nxt, nxt % STAGES); else cp_async_fence();
        compute_stage(s % STAGES);
    }
    cp_async_wait<0>();
    __syncthreads();
}

// One warp's accumulators for one m16 tile into a [rows][128] fp32 tile in
// shared memory (this warp's 16 columns).
__device__ __forceinline__ void ws_store_tile(float *sh, const FragC *c, int lane, int warp) {
    const int r0 = lane >> 2, col = warp * 16 + (lane & 3) * 2;
#pragma unroll
    for (int f = 0; f < 2; f++) {
        float *p0 = sh + r0 * 128 + col + f * 8;
        float *p1 = sh + (r0 + 8) * 128 + col + f * 8;
        p0[0] = c[f].elems[0]; p0[1] = c[f].elems[1];
        p1[0] = c[f].elems[2]; p1[1] = c[f].elems[3];
    }
}

// ---- gate + up ---------------------------------------------------------------
// grid.x = n_expert * (inter / 128), grid.y = how many blocks share an
// expert's 64-row passes (a hot expert with hundreds of rows would otherwise
// be one block's serial tail).  act[row][inter] receives the fp16 down input
// for sorted row `row`.
template <int bits, int cb, int STAGES>
__global__ void __launch_bounds__(WS_THREADS)
exl3_moe_ws_gateup(const half *__restrict__ a_g, const half *__restrict__ a_u,
                   half *__restrict__ act,
                   const uint16_t *__restrict__ gate_trellis, const half *__restrict__ gate_svh,
                   const uint16_t *__restrict__ up_trellis, const half *__restrict__ up_svh,
                   const half *__restrict__ down_suh,
                   const int32_t *__restrict__ bounds,
                   int hidden, int inter) {
    extern __shared__ __align__(16) uint8_t ws_smem[];
    const int tiles_n = inter / 128;
    const int e = blockIdx.x / tiles_n, n0 = (blockIdx.x - e * tiles_n) * 128;
    const int r0 = bounds[e], m = bounds[e + 1] - r0;
    if (m <= 0) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t tstride = (size_t)hidden * inter * bits / 16;
    const uint16_t *Bg = gate_trellis + (size_t)e * tstride;
    const uint16_t *Bu = up_trellis + (size_t)e * tstride;
    const half *svh_g = gate_svh + (size_t)e * inter + n0;
    const half *svh_u = up_svh + (size_t)e * inter + n0;
    const half *suh_d = down_suh + (size_t)e * inter + n0;
    float *sh_g = (float *)ws_smem;                 // [32][128], after the k loop
    float *sh_u = sh_g + 32 * 128;

    for (int p0 = (int)blockIdx.y * WS_ROWS; p0 < m; p0 += WS_ROWS * (int)gridDim.y) {
        const int rows = min(WS_ROWS, m - p0);
        FragC cg[WS_MT][2], cu[WS_MT][2];
        ws_gemm_pass<bits, cb, true, STAGES>(a_g + (size_t)(r0 + p0) * hidden, a_u + (size_t)(r0 + p0) * hidden,
                                              rows, Bg, Bu, hidden, inter, n0, cg, cu, ws_smem);
#pragma unroll
        for (int h = 0; h < WS_MT / 2; h++) {
            if (h * 32 >= rows) break;
            __syncthreads();
            ws_store_tile(sh_g, cg[h * 2], lane, warp);
            ws_store_tile(sh_g + 16 * 128, cg[h * 2 + 1], lane, warp);
            ws_store_tile(sh_u, cu[h * 2], lane, warp);
            ws_store_tile(sh_u + 16 * 128, cu[h * 2 + 1], lane, warp);
            __syncthreads();
            for (int rr = warp; rr < 32; rr += WS_THREADS / 32) {
                const int row = p0 + h * 32 + rr;
                if (row >= m) break;
                float4 g = ((const float4 *)(sh_g + rr * 128))[lane];
                float4 u = ((const float4 *)(sh_u + rr * 128))[lane];
                ws_had4(g, lane); ws_had4(u, lane);
                ws_scale4(g, svh_g, lane); ws_scale4(u, svh_u, lane);
                g.x = g.x / (1.0f + __expf(-g.x)) * u.x;
                g.y = g.y / (1.0f + __expf(-g.y)) * u.y;
                g.z = g.z / (1.0f + __expf(-g.z)) * u.z;
                g.w = g.w / (1.0f + __expf(-g.w)) * u.w;
                ws_scale4(g, suh_d, lane);
                ws_had4(g, lane);
                half4 o(__floats2half2_rn(g.x, g.y), __floats2half2_rn(g.z, g.w));
                ((half4 *)(act + (size_t)(r0 + row) * inter + n0))[lane] = o;
            }
        }
    }
}

// ---- down ------------------------------------------------------------------------
// grid.x = n_expert * (hidden / 128).  slot_out[slot][hidden] for the
// assignment slot_sorted[row] gets its unweighted expert output, as floats
// (exl3_moe_ws_down) or as halves (exl3_moe_ws_down_h).  The half form is
// shiso's: at an 8192-row chunk the slot outputs are 839 MB a layer, written
// here and read back whole by the combine, and halving them halves both.
__device__ __forceinline__ void ws_store_out(float *dst, const float4 &o) { *(float4 *)dst = o; }
__device__ __forceinline__ void ws_store_out(half *dst, const float4 &o) {
    *(half4 *)dst = half4(__floats2half2_rn(o.x, o.y), __floats2half2_rn(o.z, o.w));
}

template <int bits, int cb, int STAGES, typename OUT>
__device__ __forceinline__ void ws_down_body(const half *__restrict__ act, OUT *__restrict__ slot_out,
                 const uint16_t *__restrict__ down_trellis, const half *__restrict__ down_svh,
                 const int32_t *__restrict__ bounds, const int32_t *__restrict__ slot_sorted,
                 int hidden, int inter) {
    extern __shared__ __align__(16) uint8_t ws_smem[];
    const int tiles_n = hidden / 128;
    const int e = blockIdx.x / tiles_n, n0 = (blockIdx.x - e * tiles_n) * 128;
    const int r0 = bounds[e], m = bounds[e + 1] - r0;
    if (m <= 0) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t tstride = (size_t)hidden * inter * bits / 16;
    const uint16_t *Bd = down_trellis + (size_t)e * tstride;
    const half *svh_d = down_svh + (size_t)e * hidden + n0;
    float *sh_o = (float *)ws_smem;                 // [32][128]

    for (int p0 = (int)blockIdx.y * WS_ROWS; p0 < m; p0 += WS_ROWS * (int)gridDim.y) {
        const int rows = min(WS_ROWS, m - p0);
        FragC cd[WS_MT][2], unused[WS_MT][2];
        ws_gemm_pass<bits, cb, false, STAGES>(act + (size_t)(r0 + p0) * inter, NULL, rows, Bd, NULL,
                                               inter, hidden, n0, cd, unused, ws_smem);
#pragma unroll
        for (int h = 0; h < WS_MT / 2; h++) {
            if (h * 32 >= rows) break;
            __syncthreads();
            ws_store_tile(sh_o, cd[h * 2], lane, warp);
            ws_store_tile(sh_o + 16 * 128, cd[h * 2 + 1], lane, warp);
            __syncthreads();
            for (int rr = warp; rr < 32; rr += WS_THREADS / 32) {
                const int row = p0 + h * 32 + rr;
                if (row >= m) break;
                float4 o = ((const float4 *)(sh_o + rr * 128))[lane];
                ws_had4(o, lane);
                ws_scale4(o, svh_d, lane);
                const int slot = slot_sorted[r0 + row];
                ws_store_out(slot_out + (size_t)slot * hidden + n0 + lane * 4, o);
            }
        }
    }
}

template <int bits, int cb, int STAGES>
__global__ void __launch_bounds__(WS_THREADS)
exl3_moe_ws_down(const half *__restrict__ act, float *__restrict__ slot_out,
                 const uint16_t *__restrict__ down_trellis, const half *__restrict__ down_svh,
                 const int32_t *__restrict__ bounds, const int32_t *__restrict__ slot_sorted,
                 int hidden, int inter) {
    ws_down_body<bits, cb, STAGES>(act, slot_out, down_trellis, down_svh, bounds, slot_sorted, hidden, inter);
}

template <int bits, int cb, int STAGES>
__global__ void __launch_bounds__(WS_THREADS)
exl3_moe_ws_down_h(const half *__restrict__ act, half *__restrict__ slot_out,
                   const uint16_t *__restrict__ down_trellis, const half *__restrict__ down_svh,
                   const int32_t *__restrict__ bounds, const int32_t *__restrict__ slot_sorted,
                   int hidden, int inter) {
    ws_down_body<bits, cb, STAGES>(act, slot_out, down_trellis, down_svh, bounds, slot_sorted, hidden, inter);
}

// ---- combine -----------------------------------------------------------------
// out[t] = sum_s weights[t][s] * slot_out[t][s] + sigmoid(logit[t]) * shared[t],
// over half slot outputs, in slot order, so a chunk's result does not depend
// on which expert finished first.  The shared expert's gated add rides along:
// it was a second pass over the same rows.  Grid (ceil(n_embd / 4 / 256),
// n_tok), 256 threads, four columns a thread.
__global__ void __launch_bounds__(256)
exl3_moe_ws_combine(float *__restrict__ out, const half *__restrict__ slot_out,
                    const float *__restrict__ weights, const float *__restrict__ shared,
                    const float *__restrict__ logit, int n_embd, int n_used) {
    const int t = blockIdx.y;
    const int e = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (e >= n_embd) return;
    const float g = 1.0f / (1.0f + __expf(-logit[t]));
    const float4 sh = *(const float4 *)(shared + (size_t)t * n_embd + e);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < n_used; s++) {
        const size_t col = (size_t)t * n_used + s;
        const float w = weights[col];
        const half4 d = *(const half4 *)(slot_out + col * n_embd + e);
        acc.x += __low2float(d.x) * w;
        acc.y += __high2float(d.x) * w;
        acc.z += __low2float(d.y) * w;
        acc.w += __high2float(d.y) * w;
    }
    acc.x += sh.x * g; acc.y += sh.y * g; acc.z += sh.z * g; acc.w += sh.w * g;
    *(float4 *)(out + (size_t)t * n_embd + e) = acc;
}

// ---- decode shape (shiso) -------------------------------------------------------
// The weight-stationary launches above size their grid by the expert count, so a
// decode step of one token runs 2,560 gate/up blocks of which fifty have rows,
// and those fifty land on forty-eight SMs: two waves, the second two blocks
// wide, at one block an SM because the 64-row staging takes 72 KB.  Measured
// on a GB10: 115 us gate/up and 70 us down a layer, 55% of the bandwidth the
// 24.6 MB they stream costs.
//
// These are the same GEMMs cut finer.  A block is WARPS warps, one n16 column
// group each, for one *active* expert (the sort's active list), over 16-row
// passes; the 128-column Hadamard epilogues move out into their own launches,
// which read the raw single-precision sums back.  Every column's sum is the
// same sequence of m16n8k16 MMAs over the same decoded fragments and the same
// fp16 A rows, in the same k order, as exl3_moe_ws_gateup / _down's, and the
// epilogues are the same operations on the same values -- so a pass through
// these is bit-identical to one through the weight-stationary launches, and
// the choice between them is a matter of speed alone.
#ifndef WSD_KT
#define WSD_KT 64
#endif
#define WSD_A_COLS (WSD_KT / 8)

template <int bits, bool TWO, int STAGES, int WARPS>
struct WsdSmem {
    static constexpr int tile_words = 256 * bits / 16;
    static constexpr int a_stage = 16 * WSD_KT;                       // halfs, per matrix
    static constexpr int b_stage = (WSD_KT / 16) * WARPS * tile_words;  // uint16, per matrix
    static constexpr int n_mat = TWO ? 2 : 1;
    static constexpr int stage_bytes = n_mat * (a_stage * 2 + b_stage * 2);
    static constexpr int total = STAGES * stage_bytes;
};

// One 16-row pass: warp w's two n8 fragments of columns [n0 + 16w, n0 + 16w + 16).
template <int bits, int cb, bool TWO, int STAGES, int WARPS>
__device__ __forceinline__ void wsd_gemm_pass(
        const half *__restrict__ A0, const half *__restrict__ A1, int m_rows,
        const uint16_t *__restrict__ B0, const uint16_t *__restrict__ B1,
        int k, int n, int n0, FragC *c0, FragC *c1, uint8_t *smem) {
    typedef WsdSmem<bits, TWO, STAGES, WARPS> S;
    constexpr int TW = S::tile_words;
    constexpr int THREADS = WARPS * 32;
    half *sh_a0 = (half *)smem;
    half *sh_a1 = sh_a0 + STAGES * S::a_stage;
    uint16_t *sh_b0 = (uint16_t *)(sh_a1 + (TWO ? STAGES * S::a_stage : 0));
    uint16_t *sh_b1 = sh_b0 + STAGES * S::b_stage;

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int blocks_n = n / 16;
    const int n_stages = k / WSD_KT;
#pragma unroll
    for (int f = 0; f < 2; f++) { c0[f] = {}; if (TWO) c1[f] = {}; }

    auto load_a = [&](const half *A, half *sh, int k0) {
        for (int idx = t; idx < 16 * WSD_A_COLS; idx += THREADS) {
            const int m = idx / WSD_A_COLS, kc = idx % WSD_A_COLS;
            int4 *dst = (int4 *)sh + m * WSD_A_COLS + (kc ^ (m & 7));
            if (m < m_rows) cp_async(dst, (const int4 *)(A + (size_t)m * k + k0) + kc);
            else *dst = make_int4(0, 0, 0, 0);
        }
    };
    auto load_b = [&](const uint16_t *B, uint16_t *sh, int k0) {
        constexpr int per_tile = TW * 2 / 16;
        constexpr int pieces = (WSD_KT / 16) * WARPS * per_tile;
        for (int p = t; p < pieces; p += THREADS) {
            const int tile = p / per_tile, piece = p - tile * per_tile;
            const int kb = tile / WARPS, nbi = tile % WARPS;
            const size_t src = ((size_t)(k0 / 16 + kb) * blocks_n + n0 / 16 + nbi) * TW;
            cp_async((int4 *)sh + p, (const int4 *)(B + src) + piece);
        }
    };
    auto load_stage = [&](int s, int buf) {
        const int k0 = s * WSD_KT;
        load_a(A0, sh_a0 + buf * S::a_stage, k0);
        load_b(B0, sh_b0 + buf * S::b_stage, k0);
        if (TWO) {
            load_a(A1, sh_a1 + buf * S::a_stage, k0);
            load_b(B1, sh_b1 + buf * S::b_stage, k0);
        }
        cp_async_fence();
    };
    auto compute_stage = [&](int buf) {
        const half *a0 = sh_a0 + buf * S::a_stage, *a1 = sh_a1 + buf * S::a_stage;
        const uint16_t *b0 = sh_b0 + buf * S::b_stage, *b1 = sh_b1 + buf * S::b_stage;
        const int r = (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
        for (int kb = 0; kb < WSD_KT / 16; kb++) {
            FragB fb0[2], fb1[2];
            dq_dispatch<bits, cb>((const uint32_t *)(b0 + (kb * WARPS + warp) * TW), lane << 3, fb0[0], fb0[1]);
            if (TWO) dq_dispatch<bits, cb>((const uint32_t *)(b1 + (kb * WARPS + warp) * TW), lane << 3, fb1[0], fb1[1]);
            const int c = kb * 2 + (lane >> 4);
            const int off = r * WSD_A_COLS + (c ^ (r & 7));
            FragA fa;
            ldsm4(fa, (const int4 *)a0 + off);
            ptx_mma_m16n8k16(fa, fb0[0], c0[0]);
            ptx_mma_m16n8k16(fa, fb0[1], c0[1]);
            if (TWO) {
                ldsm4(fa, (const int4 *)a1 + off);
                ptx_mma_m16n8k16(fa, fb1[0], c1[0]);
                ptx_mma_m16n8k16(fa, fb1[1], c1[1]);
            }
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < n_stages) load_stage(s, s); else cp_async_fence();
    }
    for (int s = 0; s < n_stages; s++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int nxt = s + STAGES - 1;
        if (nxt < n_stages) load_stage(nxt, nxt % STAGES); else cp_async_fence();
        compute_stage(s % STAGES);
    }
    cp_async_wait<0>();
    __syncthreads();
}

// A warp's fragments of rows [0, m) into dst[row][ld], its 16 columns at col0.
__device__ __forceinline__ void wsd_store_raw(float *dst, int ld, int m, int col0, const FragC *c, int lane) {
    const int r0 = lane >> 2, col = col0 + (lane & 3) * 2;
#pragma unroll
    for (int f = 0; f < 2; f++) {
        if (r0 < m) *(float2 *)(dst + (size_t)r0 * ld + col + f * 8) = make_float2(c[f].elems[0], c[f].elems[1]);
        if (r0 + 8 < m) *(float2 *)(dst + (size_t)(r0 + 8) * ld + col + f * 8) = make_float2(c[f].elems[2], c[f].elems[3]);
    }
}

// exl3_moe_ws_prep_tok with the slot loop spread over warps: one warp per
// (token, slot, 128-column chunk).  At one token the by-token form is twenty
// warps walking ten slots each, 12 us of a decode step's layer; each slot's
// row is the same arithmetic either way.
__global__ void __launch_bounds__(WS_THREADS)
exl3_moe_wsd_prep(const float *__restrict__ x, half *__restrict__ a_g, half *__restrict__ a_u,
                  const half *__restrict__ suh_g, const half *__restrict__ suh_u,
                  const int32_t *__restrict__ ids, const int32_t *__restrict__ pos,
                  int n_tok, int n_used, int k) {
    const int lane = threadIdx.x & 31;
    const int chunks = k / 128;
    const int task = blockIdx.x * (WS_THREADS / 32) + (threadIdx.x >> 5);
    if (task >= n_tok * n_used * chunks) return;
    const int slot = task / chunks, c = task - slot * chunks;
    const int t = slot / n_used;
    const float4 xv = ((const float4 *)(x + (size_t)t * k + c * 128))[lane];
    const int e = ids[slot];
    const int row = pos[slot];
    float4 g = xv, u = xv;
    ws_scale4(g, suh_g + (size_t)e * k + c * 128, lane);
    ws_scale4(u, suh_u + (size_t)e * k + c * 128, lane);
    ws_had4(g, lane);
    ws_had4(u, lane);
    ((half4 *)(a_g + (size_t)row * k + c * 128))[lane] = half4(__floats2half2_rn(g.x, g.y), __floats2half2_rn(g.z, g.w));
    ((half4 *)(a_u + (size_t)row * k + c * 128))[lane] = half4(__floats2half2_rn(u.x, u.y), __floats2half2_rn(u.z, u.w));
}

// grid.x = min(n_rows, n_expert) * (inter / (16 * WARPS)): block b is active
// expert b / tiles_n.  raw_g / raw_u[row][inter] get the sorted rows' sums.
template <int bits, int cb, int STAGES, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
exl3_moe_wsd_gateup(const half *__restrict__ a_g, const half *__restrict__ a_u,
                    float *__restrict__ raw_g, float *__restrict__ raw_u,
                    const uint16_t *__restrict__ gate_trellis, const uint16_t *__restrict__ up_trellis,
                    const int32_t *__restrict__ bounds, const int32_t *__restrict__ active,
                    int n_expert, int hidden, int inter) {
    extern __shared__ __align__(16) uint8_t ws_smem[];
    const int tiles_n = inter / (16 * WARPS);
    const int j = blockIdx.x / tiles_n;
    if (j >= active[n_expert]) return;
    const int e = active[j], n0 = (blockIdx.x - j * tiles_n) * 16 * WARPS;
    const int r0 = bounds[e], m = bounds[e + 1] - r0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t tstride = (size_t)hidden * inter * bits / 16;
    const uint16_t *Bg = gate_trellis + (size_t)e * tstride;
    const uint16_t *Bu = up_trellis + (size_t)e * tstride;
    for (int p0 = 0; p0 < m; p0 += 16) {
        const int rows = min(16, m - p0);
        FragC cg[2], cu[2];
        wsd_gemm_pass<bits, cb, true, STAGES, WARPS>(a_g + (size_t)(r0 + p0) * hidden, a_u + (size_t)(r0 + p0) * hidden,
                                                     rows, Bg, Bu, hidden, inter, n0, cg, cu, ws_smem);
        wsd_store_raw(raw_g + (size_t)(r0 + p0) * inter, inter, rows, n0 + warp * 16, cg, lane);
        wsd_store_raw(raw_u + (size_t)(r0 + p0) * inter, inter, rows, n0 + warp * 16, cu, lane);
    }
}

// The gate/up epilogue of exl3_moe_ws_gateup, over the raw sums: one warp per
// (sorted row, 128 columns).
__global__ void __launch_bounds__(256)
exl3_moe_wsd_act(const float *__restrict__ raw_g, const float *__restrict__ raw_u, half *__restrict__ act,
                 const half *__restrict__ gate_svh, const half *__restrict__ up_svh,
                 const half *__restrict__ down_suh, const int32_t *__restrict__ ids,
                 const int32_t *__restrict__ slot_sorted, const int32_t *__restrict__ bounds,
                 int n_expert, int n_rows, int inter) {
    const int lane = threadIdx.x & 31;
    const int chunks = inter / 128;
    const int task = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (task >= n_rows * chunks) return;
    const int row = task / chunks, c = task - row * chunks;
    if (row >= bounds[n_expert]) return;
    const int e = ids[slot_sorted[row]];
    const int n0 = c * 128;
    float4 g = ((const float4 *)(raw_g + (size_t)row * inter + n0))[lane];
    float4 u = ((const float4 *)(raw_u + (size_t)row * inter + n0))[lane];
    ws_had4(g, lane); ws_had4(u, lane);
    ws_scale4(g, gate_svh + (size_t)e * inter + n0, lane); ws_scale4(u, up_svh + (size_t)e * inter + n0, lane);
    g.x = g.x / (1.0f + __expf(-g.x)) * u.x;
    g.y = g.y / (1.0f + __expf(-g.y)) * u.y;
    g.z = g.z / (1.0f + __expf(-g.z)) * u.z;
    g.w = g.w / (1.0f + __expf(-g.w)) * u.w;
    ws_scale4(g, down_suh + (size_t)e * inter + n0, lane);
    ws_had4(g, lane);
    half4 o(__floats2half2_rn(g.x, g.y), __floats2half2_rn(g.z, g.w));
    ((half4 *)(act + (size_t)row * inter + n0))[lane] = o;
}

// grid.x = min(n_rows, n_expert) * (hidden / (16 * WARPS)); raw_d[row][hidden].
template <int bits, int cb, int STAGES, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
exl3_moe_wsd_down(const half *__restrict__ act, float *__restrict__ raw_d,
                  const uint16_t *__restrict__ down_trellis,
                  const int32_t *__restrict__ bounds, const int32_t *__restrict__ active,
                  int n_expert, int hidden, int inter) {
    extern __shared__ __align__(16) uint8_t ws_smem[];
    const int tiles_n = hidden / (16 * WARPS);
    const int j = blockIdx.x / tiles_n;
    if (j >= active[n_expert]) return;
    const int e = active[j], n0 = (blockIdx.x - j * tiles_n) * 16 * WARPS;
    const int r0 = bounds[e], m = bounds[e + 1] - r0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t tstride = (size_t)hidden * inter * bits / 16;
    const uint16_t *Bd = down_trellis + (size_t)e * tstride;
    for (int p0 = 0; p0 < m; p0 += 16) {
        const int rows = min(16, m - p0);
        FragC cd[2], unused[2];
        wsd_gemm_pass<bits, cb, false, STAGES, WARPS>(act + (size_t)(r0 + p0) * inter, NULL, rows, Bd, NULL,
                                                      inter, hidden, n0, cd, unused, ws_smem);
        wsd_store_raw(raw_d + (size_t)(r0 + p0) * hidden, hidden, rows, n0 + warp * 16, cd, lane);
    }
}

// exl3_moe_ws_down_h's epilogue and exl3_moe_ws_combine in one.  A block is
// one (token, 128 columns); warp s transforms slot s's row -- the Hadamard,
// svh, and the rounding to the half the weight-stationary down stores -- into
// shared memory, and warp 0 then sums the slots in slot order with the
// combine's own expression, lane l on columns 4l..4l+3 as there.  n_used is
// at most WSD_MAX_USED.  Grid (hidden / 128, n_tok), 32 * n_used threads.
#define WSD_MAX_USED 16
__global__ void __launch_bounds__(32 * WSD_MAX_USED)
exl3_moe_wsd_combine(float *__restrict__ out, const float *__restrict__ raw_d,
                     const half *__restrict__ down_svh, const int32_t *__restrict__ ids,
                     const int32_t *__restrict__ pos, const float *__restrict__ weights,
                     const float *__restrict__ shared, const float *__restrict__ logit,
                     int n_embd, int n_used) {
    __shared__ half4 sh_d[WSD_MAX_USED][32];
    const int lane = threadIdx.x & 31, s = threadIdx.x >> 5;
    const int t = blockIdx.y;
    const int n0 = blockIdx.x * 128;
    if (s < n_used) {
        const size_t col = (size_t)t * n_used + s;
        const int e = ids[col];
        float4 o = ((const float4 *)(raw_d + (size_t)pos[col] * n_embd + n0))[lane];
        ws_had4(o, lane);
        ws_scale4(o, down_svh + (size_t)e * n_embd + n0, lane);
        sh_d[s][lane] = half4(__floats2half2_rn(o.x, o.y), __floats2half2_rn(o.z, o.w));
    }
    __syncthreads();
    if (s != 0) return;
    const int e0 = n0 + lane * 4;
    const float g = 1.0f / (1.0f + __expf(-logit[t]));
    const float4 sh = *(const float4 *)(shared + (size_t)t * n_embd + e0);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = 0; j < n_used; j++) {
        const float w = weights[(size_t)t * n_used + j];
        const half4 d = sh_d[j][lane];
        acc.x += __low2float(d.x) * w;
        acc.y += __high2float(d.x) * w;
        acc.z += __low2float(d.y) * w;
        acc.w += __high2float(d.y) * w;
    }
    acc.x += sh.x * g; acc.y += sh.y * g; acc.z += sh.z * g; acc.w += sh.w * g;
    *(float4 *)(out + (size_t)t * n_embd + e0) = acc;
}

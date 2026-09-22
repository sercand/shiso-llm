#pragma once

// Vendored from exllamav3 (quant/reconstruct.cu, MIT, Copyright (c) 2025 Turboderp):
// the reconstruct_had_kernel alone, with the torch wrapper left behind and the
// C++17-removed `register` dropped.  See VENDOR.md.
//
// shiso uses it for the prefill path of the dense EXL3 tensors: past a few
// hundred rows it is cheaper to expand a tensor once to fp16 and run cuBLAS
// than to stream the trellis once per 16 rows (exllamav3's own prefill design).

#include <cuda_fp16.h>
#include <stdint.h>

#include "util.cuh"
#include "ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"

// Fused reconstruct + both-side Hadamard: emits W = diag(suh) . H128 . W_hat . H128 . diag(svh)
// (per 128x128 tile, 1/sqrt(128) per side), i.e. the ORIGINAL-basis weights, so the hgemm
// prefill path can run on the raw input with no standalone input/output had_r_128 launches.
//
// 128x128 tile lives in shared memory with an XOR swizzle on the half4 chunk index
// (chunk ^= (row >> 2) & 31) making all phases bank-conflict-free except the 16-lane dequant
// scatter; both transforms reuse the natural-order Sylvester butterfly (symmetric, so
// column-then-row order is H W H).
#define RH_THREADS 256

template <int K, int cb>
__global__ __launch_bounds__(RH_THREADS)
void reconstruct_had_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 256 * K / 16;
    constexpr float r_scale = 0.08838834764831845f;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int kb = blockIdx.y;
    int nb = blockIdx.x;
    int n = nb * 8;
    int row_len = gridDim.x * 128;

    __shared__ uint32_t s_packed[8][8][packed_size / 2];
    __shared__ half2 stile[128 * 64];

    auto tix = [&] (int R, int q, int p)
    {
        return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
    };

    constexpr int j_int4 = packed_size / 8;
    for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS)
    {
        int j = u / (8 * j_int4);
        int r = u % (8 * j_int4);
        const uint16_t* gp = g_packed +
            ((size_t) ((kb * 8 + j) * packed_blocks_n + packed_n_offset + n)) * packed_size;
        ((int4*) s_packed[j])[r] = ((const int4*) gp)[r];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (RH_THREADS / 32); ++jj)
    {
        int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
        int wn = warp_id % 8;
        FragB frag[2];
        dq_dispatch<K, cb>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = j * 16 + (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    const half2 rs2 = __float2half2_rn(r_scale);
    constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
    #pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq)
    {
        int q = warp_id * CHUNKS_PW + qq;
        int qs = q ^ lane_id;
        // a[i]/b[i]: 4 columns x row (4 * lane + i), all four rows share swizzle == lane
        half2 a[4], b[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v = *((const half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            a[i] = v.x;
            b[i] = v.y;
        }
        // butterfly over the 4 rows, two columns at a time, fp16 with pre-applied scale
        #pragma unroll
        for (int x = 0; x < 2; ++x)
        {
            half2* v = x == 0 ? a : b;
            half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
            half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
            v[0] = __hmul2(__hadd2(s0, s1), rs2);
            v[1] = __hmul2(__hadd2(d0, d1), rs2);
            v[2] = __hmul2(__hsub2(s0, s1), rs2);
            v[3] = __hmul2(__hsub2(d0, d1), rs2);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = shuffle_had_h2x32(v[i], lane_id);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v;
            v.x = a[i];
            v.y = b[i];
            *((half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    // Row transform fused with the store: after the lane butterfly, lane l holds the
    // FINAL columns 4l..4l+3 of row R. Apply the sign scales in registers and write the
    // coalesced 256-byte row directly.
    constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
    const half4 sv4 = ((const half4*) svh)[nb * 32 + lane_id];
    #pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr)
    {
        int R = warp_id * ROWS_PW + rr;
        int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        half2 v01 = stile[base];
        half2 v23 = stile[base + 1];
        float v0 = __low2float(v01), v1 = __high2float(v01);
        float v2 = __low2float(v23), v3 = __high2float(v23);
        float s0 = v0 + v1, d0 = v0 - v1;
        float s1 = v2 + v3, d1 = v2 - v3;
        half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
        h01 = shuffle_had_h2x32(h01, lane_id);
        h23 = shuffle_had_h2x32(h23, lane_id);
        half2 su2 = __half2half2(suh[kb * 128 + R]);
        half4 v;
        v.x = __hmul2(__hmul2(h01, su2), sv4.x);
        v.y = __hmul2(__hmul2(h23, su2), sv4.y);
        *((half4*) (g_unpacked + (size_t) (kb * 128 + R) * row_len + nb * 128 + lane_id * 4)) = v;
    }
}


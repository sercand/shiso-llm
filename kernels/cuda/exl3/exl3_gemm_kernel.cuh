#pragma once

// Vendored from exllamav3 (quant/exl3_gemm_kernel.cuh, MIT, Copyright (c) 2025 Turboderp)
// with shiso modifications, see VENDOR.md:
//   - the activation matrix A is fp32 (shiso keeps every activation in fp32); the
//     input Hadamard phase converts it to the fp16 A_had the MMA loop reads;
//   - exl3_mgemm addresses the per-expert matrices as one stacked tensor
//     (base + expert * stride for tiles, suh and svh) selected by int32 ids,
//     instead of device pointer tables, and drops the expert-range filtering,
//     weighted reduction and per-matrix width lists shiso does not use.

#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_devctx.cuh"

// C[m][n] = (Had128(A * suh) @ B) then Had128 * svh per 128-wide output block.
// Cooperative launch: the input transform needs a grid-wide barrier before the
// matmul, and every 16-row slice of A is one pass over the whole grid.
template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_kernel(EXL3_GEMM_ARGS)
{
    auto grid = cg::this_grid();

    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_fh_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
    }

    int size_m_ = size_m;
    const half* A_ = A_had;
    void* C_ = C;

    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner
        <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
        (A_, B, C_, MIN(size_m_, 16), size_k, size_n, locks, svh);

        A_ += 16 * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
        else                  C_ = (void*) (((half*) C_) + 16 * size_n);
        size_m_ -= 16;

        grid.sync();
    }
}

// Expert fan-out: slot j runs A (or A[j] when bszm_in > 1) through matrix
// ids[j] of a stacked expert tensor into C[j].  blockIdx.z picks which slots a
// block group serves; the groups synchronize among themselves only.  Slots at
// or past `split` are the second tensor's (see EXL3_MGEMM_ARGS): slot j there
// is row j - split of the first tensor's ids, input and output, so C and C2
// come out in the same [split][m][n] layout.
template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_mgemm_kernel(EXL3_MGEMM_ARGS)
{
    int bszm = MAX(bszm_in, bszm_out);
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    for (int i = 0; i < bszm; i += gridDim.z)
    {
        int j = i + blockIdx.z;
        int jj = j;                 // slot within its tensor
        const uint16_t* B = nullptr;
        const half* suh = nullptr;
        const half* svh = nullptr;
        void* C_out = C;
        if (j < bszm)
        {
            const uint16_t* Bb = B_base;
            const half* suhb = suh_base;
            const half* svhb = svh_base;
            if (B_base2 && j >= split)
            {
                jj = j - split;
                Bb = B_base2; suhb = suh_base2; svhb = svh_base2; C_out = C2;
            }
            int mat_index = B_indices ? (int) B_indices[jj] : jj;
            if (mat_index >= 0)
            {
                B = Bb + (size_t) mat_index * B_stride;
                suh = suhb + (size_t) mat_index * size_k;
                svh = svhb + (size_t) mat_index * size_n;
            }
        }

        // Input scales and Hadamard into this slot's A_had slab

        if (B)
        {
            int total_warps = size_m * size_k / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const float* A_ = bszm_in == 1 ? A : A + (size_t) jj * size_m * size_k;
            half* A_had_ = A_had + (size_t) j * size_m * size_k;

            for(; this_warp < total_warps; this_warp += warps_grid)
                had_fh_r_128_inner<true, false>
                (
                    A_ + this_warp * 128,
                    A_had_ + this_warp * 128,
                    suh + (this_warp * 128) % size_k,
                    0.088388347648f  // 1/sqrt(128)
                );
        }

        group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);

        // Matmul, 16 rows of A at a time

        int size_m_ = size_m;
        half* A_ = A_had + (size_t) j * size_m * size_k;
        void* C_;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_out) + (size_t) jj * size_m * size_n);
        else                  C_ = (void*) (((half*) C_out) + (size_t) jj * size_m * size_n);
        void* C_base = C_;

        while (size_m_ > 0)
        {
            if (B)
            {
                int lock_offs = blockIdx.z * size_n / 128;

                exl3_gemm_kernel_inner
                <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false>
                (A_, B, C_, MIN(size_m_, 16), size_k, size_n, locks + lock_offs, nullptr);
            }

            A_ += 16 * size_k;
            if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
            else                  C_ = (void*) (((half*) C_) + 16 * size_n);
            size_m_ -= 16;

            group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
        }

        // Output Hadamard and scales

        if (B)
        {
            int total_warps = size_m * size_n / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            C_ = C_base;

            for(; this_warp < total_warps; this_warp += warps_grid)
            {
                if constexpr (c_fp32)
                    had_ff_r_128_inner<false, true>
                    (
                        ((const float*) C_) + this_warp * 128,
                        ((float*) C_) + this_warp * 128,
                        svh + (this_warp * 128) % size_n,
                        0.088388347648f
                    );
                else
                    had_hf_r_128_inner<false, true>
                    (
                        ((const half*) C_) + this_warp * 128,
                        ((half*) C_) + this_warp * 128,
                        svh + (this_warp * 128) % size_n,
                        0.088388347648f
                    );
            }
        }
    }
}

#pragma once

// Vendored from exllamav3 (quant/exl3_kernel_map.cuh, MIT, Copyright (c) 2025 Turboderp);
// kernel argument lists and tile shapes only, see VENDOR.md for the shiso changes.


#define EXL3_GEMM_T_ARGS \
    const int bits, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

#define EXL3_GEMM_ARGS \
    const float* __restrict__ A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh

// shiso: stacked expert tensors -- matrix q is B_base + q * B_stride (uint16
// units), suh_base + q * size_k, svh_base + q * size_n -- selected per slot by
// int32 ids.  Two stacked tensors of the same shape can share one launch:
// slots [0, split) address the first and write C, slots [split, bszm) address
// the second with ids[slot - split], the same input row, and write C2 (the
// routed gate and up projections read the same activation).  With
// B_base2 == nullptr every slot uses the first tensor.  See VENDOR.md.
#define EXL3_MGEMM_ARGS \
    const float* __restrict__  A, \
    const uint16_t* __restrict__ B_base, \
    const size_t B_stride, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh_base, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh_base, \
    const int32_t* __restrict__ B_indices, \
    const int bszm_in, \
    const int bszm_out, \
    const uint16_t* __restrict__ B_base2, \
    const half* __restrict__ suh_base2, \
    const half* __restrict__ svh_base2, \
    void* __restrict__ C2, \
    const int split

typedef void (*fp_exl3_gemm_kernel) (EXL3_GEMM_ARGS);
typedef void (*fp_exl3_mgemm_kernel) (EXL3_MGEMM_ARGS);

#define EXL3_GEMM_SHAPE_1     16,     16,    128,     6,     5
#define EXL3_GEMM_SHAPE_2     16,     32,    128,     4,     3
#define EXL3_GEMM_SHAPE_3     16,     32,    256,     4,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     4,     3

#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512
#define EXL3_GEMM_BLOCKDIM  0, 256, 512, 512, 256

#define EXL3_GEMM_NUM_SHAPES 4

// Shape 1 not currently used anywhere
#define EXL3_GEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    exl3_gemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_4>

#define EXL3_MGEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, \
    exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    exl3_mgemm_kernel<_bits, _c_fp32, cb, EXL3_GEMM_SHAPE_4>

#define EXL3_GEMM_BASE_THREADS 256

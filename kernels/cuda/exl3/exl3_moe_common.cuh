#pragma once

// Vendored from exllamav3 (quant/exl3_moe_common.cuh, MIT, Copyright (c) 2025 Turboderp)
// with the shiso argument list, see VENDOR.md and exl3_moe_kernel.cuh.

#include <cuda_fp16.h>
#include <stdint.h>

#define MOE_SMS_PER_EXPERT 8       // default/minimum group width, also sets max concurrency (buffer count)
#define MOE_MAX_SMS_PER_EXPERT 32  // widest expert group when few experts are active
#define MOE_TILESIZE_K 32
#define MOE_TILESIZE_M 16
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3

#ifndef EXL3_GEMM_BASE_THREADS
#define EXL3_GEMM_BASE_THREADS 256
#endif

#ifndef SMEM_MAX
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6
#endif

// The three projections are stacked expert tensors: expert e's tiles are
// `trellis + e * hidden * intermediate * bits / 16`, its scale vectors
// `suh + e * k` and `svh + e * n` with (k, n) = (hidden, intermediate) for
// gate and up and (intermediate, hidden) for down.  The routing comes as the
// assignments sorted by expert -- expert e owns slot_sorted[bounds[e] ..
// bounds[e + 1]) -- where a slot is token * num_experts_per_tok + rank, and
// slot_out[slot] receives that assignment's unweighted down projection: the
// routing weights are applied by the caller's combine, in a fixed order, so a
// chunk's output does not depend on which expert finished first.  Every
// group's staging buffers hold num_tokens rows, the bound on any expert's
// share: temp_state is the fp16 gathered input for gate and up (2 x hidden
// halfs a row) and afterwards the fp32 down output (hidden floats a row, the
// same bytes), temp_intermediate the fp32 gate and up outputs (2 x
// intermediate floats a row), temp_act the fp16 down input.
#define EXL3_MOE_KERNEL_ARGS                    \
    const float* __restrict__ hidden_state,     \
    half* __restrict__ temp_state,              \
    float* __restrict__ temp_intermediate,      \
    half* __restrict__ temp_act,                \
    float* __restrict__ slot_out,               \
                                                \
    const uint16_t* __restrict__ gate_trellis,  \
    const half* __restrict__ gate_suh,          \
    const half* __restrict__ gate_svh,          \
    const uint16_t* __restrict__ up_trellis,    \
    const half* __restrict__ up_suh,            \
    const half* __restrict__ up_svh,            \
    const uint16_t* __restrict__ down_trellis,  \
    const half* __restrict__ down_suh,          \
    const half* __restrict__ down_svh,          \
                                                \
    const int32_t* __restrict__ expert_bounds,  \
    const int32_t* __restrict__ slot_sorted,    \
                                                \
    const int num_tokens,                       \
    const int hidden_dim,                       \
    const int intermediate_dim,                 \
    const int num_experts,                      \
    const int num_experts_per_tok,              \
                                                \
    int* __restrict__ locks

typedef void (*fp_exl3_moe_kernel)(EXL3_MOE_KERNEL_ARGS);

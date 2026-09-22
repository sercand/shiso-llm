/* MLX native packed affine matrix-vector (qmv), matrix-matrix (qmm) and the two
 * row gathers that read the same layout.
 *
 * Group size 64, affine layout: weights in little-endian packed u32 words,
 * scales and biases as bf16 arrays. These have no upstream CUDA twin; the
 * arithmetic comes from `crates/shiso-modelfile/src/mlx_quant.rs` (LSB-first u32
 * unpack, `w*scale + bias`) and from the Metal bodies in
 * `metal/dense/dense_affine.metal` and `dense_affine_rows.metal`, both already
 * in this tree, so nothing here is a byte copy and there is no new upstream to
 * attribute.
 *
 * Factorization:
 *   sum_i (q_i * scale + bias) * x_i = scale * sum_i (q_i * x_i) + bias * sum_i x_i
 * evaluated per group of 64 elements, accumulated across the row, then warp
 * reduced.
 *
 * A row is walked as `in_dim / 64` groups, which is exact because
 * `shiso_engine::projection::affine_geometry` refuses an `in_dim` that is not a
 * whole number of groups. That refusal is where the tail is handled and it is
 * backend-neutral: a width the kernel would drop the last group of never
 * reaches a launch. Phase 5 found the opposite arrangement - a kernel quietly
 * dropping the tail - and it produced a projection wrong in its last columns
 * and nowhere else.
 */
#ifndef SHISO_DENSE_AFFINE_CUH
#define SHISO_DENSE_AFFINE_CUH

#include "dense_common.cuh"

/* Activation columns one matrix threadgroup carries.
 *
 * Eight, measured, and mirrored from `shiso_engine::projection::
 * AFFINE_GEMM_TOKENS`: a 64-token prefill of Gemma 4 E4B on the mlx container
 * runs at 16 tok/s with ONE column a threadgroup and 40 with eight, and sixteen
 * measures the same 40 - past eight the packed row is no longer what the
 * prefill is waiting for. Eight accumulators a thread also stay in registers.
 * The chunk is not required to be a multiple of it. */
#define DENSE_AFFINE_TOKENS 8u

/* 4-bit affine matrix-vector multiplication (single token). */
extern "C" __global__ void kernel_dense_affine_qmv_4_f32(
        const uint32_t *__restrict__ w,
        const unsigned short *__restrict__ scales,
        const unsigned short *__restrict__ biases,
        const float *__restrict__ x,
        float *__restrict__ y,
        const struct dense_args_affine args) {
    const unsigned row = blockIdx.x;
    if (row >= args.out_dim) {
        return;
    }
    const unsigned lane = threadIdx.x;
    const unsigned words_per_row = args.in_dim / 8u;
    const unsigned groups_per_row = args.in_dim / 64u;
    const unsigned word_in_group = lane / 4u;
    const unsigned shift0 = ((2u * lane) % 8u) * 4u;
    const unsigned shift1 = ((2u * lane + 1u) % 8u) * 4u;

    const uint32_t *row_w = w + (uint64_t) row * words_per_row;
    const unsigned short *row_s = scales + (uint64_t) row * groups_per_row;
    const unsigned short *row_b = biases + (uint64_t) row * groups_per_row;

    float row_acc = 0.0f;
    for (unsigned g = 0u; g < groups_per_row; ++g) {
        const uint32_t word = row_w[g * 8u + word_in_group];
        const float q0 = (float) ((word >> shift0) & 0x0fu);
        const float q1 = (float) ((word >> shift1) & 0x0fu);
        const float x0 = x[g * 64u + 2u * lane];
        const float x1 = x[g * 64u + 2u * lane + 1u];

        const float g_sum_qx = q0 * x0 + q1 * x1;
        const float g_sum_x = x0 + x1;
        row_acc += dense_bf16(row_s + g) * g_sum_qx + dense_bf16(row_b + g) * g_sum_x;
    }

    row_acc = dense_warp_sum(row_acc);
    if (lane == 0u) {
        y[row] = row_acc;
    }
}

/* 8-bit affine matrix-vector multiplication (single token). */
extern "C" __global__ void kernel_dense_affine_qmv_8_f32(
        const uint32_t *__restrict__ w,
        const unsigned short *__restrict__ scales,
        const unsigned short *__restrict__ biases,
        const float *__restrict__ x,
        float *__restrict__ y,
        const struct dense_args_affine args) {
    const unsigned row = blockIdx.x;
    if (row >= args.out_dim) {
        return;
    }
    const unsigned lane = threadIdx.x;
    const unsigned words_per_row = args.in_dim / 4u;
    const unsigned groups_per_row = args.in_dim / 64u;
    const unsigned word_in_group = lane / 2u;
    const unsigned shift0 = (lane & 1u) ? 16u : 0u;
    const unsigned shift1 = (lane & 1u) ? 24u : 8u;

    const uint32_t *row_w = w + (uint64_t) row * words_per_row;
    const unsigned short *row_s = scales + (uint64_t) row * groups_per_row;
    const unsigned short *row_b = biases + (uint64_t) row * groups_per_row;

    float row_acc = 0.0f;
    for (unsigned g = 0u; g < groups_per_row; ++g) {
        const uint32_t word = row_w[g * 16u + word_in_group];
        const float q0 = (float) ((word >> shift0) & 0xffu);
        const float q1 = (float) ((word >> shift1) & 0xffu);
        const float x0 = x[g * 64u + 2u * lane];
        const float x1 = x[g * 64u + 2u * lane + 1u];

        const float g_sum_qx = q0 * x0 + q1 * x1;
        const float g_sum_x = x0 + x1;
        row_acc += dense_bf16(row_s + g) * g_sum_qx + dense_bf16(row_b + g) * g_sum_x;
    }

    row_acc = dense_warp_sum(row_acc);
    if (lane == 0u) {
        y[row] = row_acc;
    }
}

/* 4-bit affine matrix-matrix multiplication: one row, EIGHT activation columns.
 *
 * Every loop over the band is a compile-time trip count with no early exit, so
 * `acc` stays in registers. A `break` on the runtime span would make the bound
 * dynamic, the array would spill to local memory - which is device memory here
 * - and each accumulate would cost a round trip, which is worse than the weight
 * traffic this kernel exists to avoid. The tail band reads its LAST column
 * repeatedly instead, and discards those lanes when it writes. */
extern "C" __global__ void kernel_dense_affine_qmm_4_f32(
        const uint32_t *__restrict__ w,
        const unsigned short *__restrict__ scales,
        const unsigned short *__restrict__ biases,
        const float *__restrict__ x,
        float *__restrict__ y,
        const struct dense_args_affine args) {
    const unsigned row = blockIdx.x;
    const unsigned tok0 = blockIdx.y * DENSE_AFFINE_TOKENS;
    if (row >= args.out_dim || tok0 >= args.n_tok) {
        return;
    }
    const unsigned span = min(DENSE_AFFINE_TOKENS, args.n_tok - tok0);
    const unsigned lane = threadIdx.x;
    const unsigned words_per_row = args.in_dim / 8u;
    const unsigned groups_per_row = args.in_dim / 64u;
    const unsigned word_in_group = lane / 4u;
    const unsigned shift0 = ((2u * lane) % 8u) * 4u;
    const unsigned shift1 = ((2u * lane + 1u) % 8u) * 4u;

    const uint32_t *row_w = w + (uint64_t) row * words_per_row;
    const unsigned short *row_s = scales + (uint64_t) row * groups_per_row;
    const unsigned short *row_b = biases + (uint64_t) row * groups_per_row;

    float acc[DENSE_AFFINE_TOKENS];
#pragma unroll
    for (unsigned j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
        acc[j] = 0.0f;
    }
    const unsigned last = args.n_tok - 1u;

    for (unsigned g = 0u; g < groups_per_row; ++g) {
        const uint32_t word = row_w[g * 8u + word_in_group];
        const float q0 = (float) ((word >> shift0) & 0x0fu);
        const float q1 = (float) ((word >> shift1) & 0x0fu);
        const float s = dense_bf16(row_s + g);
        const float b = dense_bf16(row_b + g);
        const unsigned at = g * 64u + 2u * lane;
#pragma unroll
        for (unsigned j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
            const float *tok_x = x + (uint64_t) min(tok0 + j, last) * args.in_dim;
            const float x0 = tok_x[at];
            const float x1 = tok_x[at + 1u];
            acc[j] += s * (q0 * x0 + q1 * x1) + b * (x0 + x1);
        }
    }

#pragma unroll
    for (unsigned j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
        const float total = dense_warp_sum(acc[j]);
        if (lane == 0u && j < span) {
            y[(uint64_t) (tok0 + j) * args.out_dim + row] = total;
        }
    }
}

/* 8-bit affine matrix-matrix multiplication: one row, EIGHT activation columns.
 *
 * The band arrangement and the tail handling are the 4-bit kernel's; see the
 * note there. */
extern "C" __global__ void kernel_dense_affine_qmm_8_f32(
        const uint32_t *__restrict__ w,
        const unsigned short *__restrict__ scales,
        const unsigned short *__restrict__ biases,
        const float *__restrict__ x,
        float *__restrict__ y,
        const struct dense_args_affine args) {
    const unsigned row = blockIdx.x;
    const unsigned tok0 = blockIdx.y * DENSE_AFFINE_TOKENS;
    if (row >= args.out_dim || tok0 >= args.n_tok) {
        return;
    }
    const unsigned span = min(DENSE_AFFINE_TOKENS, args.n_tok - tok0);
    const unsigned lane = threadIdx.x;
    const unsigned words_per_row = args.in_dim / 4u;
    const unsigned groups_per_row = args.in_dim / 64u;
    const unsigned word_in_group = lane / 2u;
    const unsigned shift0 = (lane & 1u) ? 16u : 0u;
    const unsigned shift1 = (lane & 1u) ? 24u : 8u;

    const uint32_t *row_w = w + (uint64_t) row * words_per_row;
    const unsigned short *row_s = scales + (uint64_t) row * groups_per_row;
    const unsigned short *row_b = biases + (uint64_t) row * groups_per_row;

    float acc[DENSE_AFFINE_TOKENS];
#pragma unroll
    for (unsigned j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
        acc[j] = 0.0f;
    }
    const unsigned last = args.n_tok - 1u;

    for (unsigned g = 0u; g < groups_per_row; ++g) {
        const uint32_t word = row_w[g * 16u + word_in_group];
        const float q0 = (float) ((word >> shift0) & 0xffu);
        const float q1 = (float) ((word >> shift1) & 0xffu);
        const float s = dense_bf16(row_s + g);
        const float b = dense_bf16(row_b + g);
        const unsigned at = g * 64u + 2u * lane;
#pragma unroll
        for (unsigned j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
            const float *tok_x = x + (uint64_t) min(tok0 + j, last) * args.in_dim;
            const float x0 = tok_x[at];
            const float x1 = tok_x[at + 1u];
            acc[j] += s * (q0 * x0 + q1 * x1) + b * (x0 + x1);
        }
    }

#pragma unroll
    for (unsigned j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
        const float total = dense_warp_sum(acc[j]);
        if (lane == 0u && j < span) {
            y[(uint64_t) (tok0 + j) * args.out_dim + row] = total;
        }
    }
}

/* Gathers and dequantizes token rows from a 4-bit affine embedding table.
 *
 * One thread per (column, token), so the grid is `(ceil(hidden/64), n_tok)` at
 * a block of 64 and the tail block's guard is the only thing between a hidden
 * width that is not a whole number of blocks and a write past the row. */
extern "C" __global__ void kernel_dense_embed_affine4_f32(
        const struct dense_args_embed_affine args,
        float *__restrict__ out,
        const uint32_t *__restrict__ words,
        const unsigned short *__restrict__ scales,
        const unsigned short *__restrict__ biases,
        const int32_t *__restrict__ ids) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned tok = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= args.hidden || tok >= args.n_tok) {
        return;
    }

    const uint64_t row = (uint64_t) ids[tok];
    const unsigned words_per_row = args.hidden / 8u;
    const unsigned groups_per_row = args.hidden / 64u;

    const unsigned g = col / 64u;
    const unsigned in_group = col % 64u;
    const unsigned word_idx = in_group / 8u;
    const unsigned shift = (in_group % 8u) * 4u;

    const uint32_t word = words[row * (uint64_t) words_per_row + g * 8u + word_idx];
    const float q = (float) ((word >> shift) & 0x0fu);
    const float scale = dense_bf16(scales + row * (uint64_t) groups_per_row + g);
    const float bias = dense_bf16(biases + row * (uint64_t) groups_per_row + g);

    out[(uint64_t) tok * args.hidden + col] = q * scale + bias;
}

#endif /* SHISO_DENSE_AFFINE_CUH */

// MLX native packed affine matrix-vector (qmv) and matrix-matrix (qmm) kernels.
//
// Ported from MLX `quantized.h` (Apache-2.0). Group size 64, affine layout:
// weights in little-endian packed u32 words, scales and biases as bf16 arrays.
//
// Factorization: sum_i (q_i * scale + bias) * x_i = scale * sum_i (q_i * x_i) + bias * sum_i x_i.
// Evaluated per group of 64 elements, accumulated across the row, then warp-reduced.
//
// The matvec entries do one token. The matrix entries do DENSE_AFFINE_TOKENS of
// them per threadgroup, which is the whole reason they are separate kernels: a
// packed row is expensive to fetch and cheap to reuse, so unpacking it once and
// walking several activation columns with it divides the weight traffic of a
// prefill chunk by that many. One token per threadgroup - which is what a matvec
// launched n_tok times is - reads the entire matrix once per token, and at this
// model's size that is the whole cost of a prefill.

/// Activation columns one matrix threadgroup carries.
///
/// Eight, measured: a 64-token prefill of Gemma 4 E4B on the mlx container runs
/// at 16 tok/s with one column a threadgroup and 40 with eight, and sixteen
/// measures the same 40 - past eight the packed row is no longer what the
/// prefill is waiting for. Eight accumulators a thread also stay in registers,
/// which sixteen need not on every device. The chunk is not required to be a
/// multiple of it.
#define DENSE_AFFINE_TOKENS 8u

#include <metal_stdlib>
using namespace metal;

/// 4-bit affine matrix-vector multiplication (single token).
kernel void kernel_dense_affine_qmv_4_f32(
    device const uint32_t * w [[buffer(0)]],
    device const ushort * scales [[buffer(1)]],
    device const ushort * biases [[buffer(2)]],
    device const float * x [[buffer(3)]],
    device float * y [[buffer(4)]],
    constant dense_args_affine & args [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const uint row = tg_pos.x;
    if (row >= args.out_dim) {
        return;
    }
    const uint lane = simd_lid;
    const uint words_per_row = args.in_dim / 8u;
    const uint groups_per_row = args.in_dim / 64u;
    const uint word_in_group = lane / 4u;
    const uint shift0 = ((2u * lane) % 8u) * 4u;
    const uint shift1 = ((2u * lane + 1u) % 8u) * 4u;

    device const uint32_t * row_w = w + (uint64_t) row * words_per_row;
    device const ushort * row_s = scales + (uint64_t) row * groups_per_row;
    device const ushort * row_b = biases + (uint64_t) row * groups_per_row;

    float row_acc = 0.0f;
    for (uint g = 0u; g < groups_per_row; ++g) {
        const uint32_t word = row_w[g * 8u + word_in_group];
        const float q0 = (float) ((word >> shift0) & 0x0fu);
        const float q1 = (float) ((word >> shift1) & 0x0fu);
        const float x0 = x[g * 64u + 2u * lane];
        const float x1 = x[g * 64u + 2u * lane + 1u];

        const float g_sum_qx = q0 * x0 + q1 * x1;
        const float g_sum_x = x0 + x1;
        const float s = dense_bf16(row_s + g);
        const float b = dense_bf16(row_b + g);
        row_acc += s * g_sum_qx + b * g_sum_x;
    }

    row_acc = dense_warp_sum(row_acc);
    if (lane == 0u) {
        y[row] = row_acc;
    }
}

/// 8-bit affine matrix-vector multiplication (single token).
kernel void kernel_dense_affine_qmv_8_f32(
    device const uint32_t * w [[buffer(0)]],
    device const ushort * scales [[buffer(1)]],
    device const ushort * biases [[buffer(2)]],
    device const float * x [[buffer(3)]],
    device float * y [[buffer(4)]],
    constant dense_args_affine & args [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const uint row = tg_pos.x;
    if (row >= args.out_dim) {
        return;
    }
    const uint lane = simd_lid;
    const uint words_per_row = args.in_dim / 4u;
    const uint groups_per_row = args.in_dim / 64u;
    const uint word_in_group = lane / 2u;
    const uint shift0 = (lane & 1u) ? 16u : 0u;
    const uint shift1 = (lane & 1u) ? 24u : 8u;

    device const uint32_t * row_w = w + (uint64_t) row * words_per_row;
    device const ushort * row_s = scales + (uint64_t) row * groups_per_row;
    device const ushort * row_b = biases + (uint64_t) row * groups_per_row;

    float row_acc = 0.0f;
    for (uint g = 0u; g < groups_per_row; ++g) {
        const uint32_t word = row_w[g * 16u + word_in_group];
        const float q0 = (float) ((word >> shift0) & 0xffu);
        const float q1 = (float) ((word >> shift1) & 0xffu);
        const float x0 = x[g * 64u + 2u * lane];
        const float x1 = x[g * 64u + 2u * lane + 1u];

        const float g_sum_qx = q0 * x0 + q1 * x1;
        const float g_sum_x = x0 + x1;
        const float s = dense_bf16(row_s + g);
        const float b = dense_bf16(row_b + g);
        row_acc += s * g_sum_qx + b * g_sum_x;
    }

    row_acc = dense_warp_sum(row_acc);
    if (lane == 0u) {
        y[row] = row_acc;
    }
}

/// 4-bit affine matrix-matrix multiplication: one row, several activation columns.
kernel void kernel_dense_affine_qmm_4_f32(
    device const uint32_t * w [[buffer(0)]],
    device const ushort * scales [[buffer(1)]],
    device const ushort * biases [[buffer(2)]],
    device const float * x [[buffer(3)]],
    device float * y [[buffer(4)]],
    constant dense_args_affine & args [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const uint row = tg_pos.x;
    const uint tok0 = tg_pos.y * DENSE_AFFINE_TOKENS;
    if (row >= args.out_dim || tok0 >= args.n_tok) {
        return;
    }
    const uint span = min(DENSE_AFFINE_TOKENS, args.n_tok - tok0);
    const uint lane = simd_lid;
    const uint words_per_row = args.in_dim / 8u;
    const uint groups_per_row = args.in_dim / 64u;
    const uint word_in_group = lane / 4u;
    const uint shift0 = ((2u * lane) % 8u) * 4u;
    const uint shift1 = ((2u * lane + 1u) % 8u) * 4u;

    device const uint32_t * row_w = w + (uint64_t) row * words_per_row;
    device const ushort * row_s = scales + (uint64_t) row * groups_per_row;
    device const ushort * row_b = biases + (uint64_t) row * groups_per_row;

    // Every loop over the band is a compile-time trip count with no early exit,
    // so `acc` stays in registers. A `break` on the runtime span would make the
    // bound dynamic, the array would spill to thread-local memory - which is
    // device memory on this hardware - and each accumulate would cost a round
    // trip, which is worse than the weight traffic this kernel exists to avoid.
    // The tail band reads its last column repeatedly instead, and discards those
    // lanes when it writes.
    float acc[DENSE_AFFINE_TOKENS];
#pragma clang loop unroll(full)
    for (uint j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
        acc[j] = 0.0f;
    }
    const uint last = args.n_tok - 1u;

    for (uint g = 0u; g < groups_per_row; ++g) {
        const uint32_t word = row_w[g * 8u + word_in_group];
        const float q0 = (float) ((word >> shift0) & 0x0fu);
        const float q1 = (float) ((word >> shift1) & 0x0fu);
        const float s = dense_bf16(row_s + g);
        const float b = dense_bf16(row_b + g);
        const uint at = g * 64u + 2u * lane;
#pragma clang loop unroll(full)
        for (uint j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
            device const float * tok_x = x + (uint64_t) min(tok0 + j, last) * args.in_dim;
            const float x0 = tok_x[at];
            const float x1 = tok_x[at + 1u];
            acc[j] += s * (q0 * x0 + q1 * x1) + b * (x0 + x1);
        }
    }

#pragma clang loop unroll(full)
    for (uint j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
        const float total = dense_warp_sum(acc[j]);
        if (lane == 0u && j < span) {
            y[(uint64_t) (tok0 + j) * args.out_dim + row] = total;
        }
    }
}

/// 8-bit affine matrix-matrix multiplication: one row, several activation columns.
kernel void kernel_dense_affine_qmm_8_f32(
    device const uint32_t * w [[buffer(0)]],
    device const ushort * scales [[buffer(1)]],
    device const ushort * biases [[buffer(2)]],
    device const float * x [[buffer(3)]],
    device float * y [[buffer(4)]],
    constant dense_args_affine & args [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const uint row = tg_pos.x;
    const uint tok0 = tg_pos.y * DENSE_AFFINE_TOKENS;
    if (row >= args.out_dim || tok0 >= args.n_tok) {
        return;
    }
    const uint span = min(DENSE_AFFINE_TOKENS, args.n_tok - tok0);
    const uint lane = simd_lid;
    const uint words_per_row = args.in_dim / 4u;
    const uint groups_per_row = args.in_dim / 64u;
    const uint word_in_group = lane / 2u;
    const uint shift0 = (lane & 1u) ? 16u : 0u;
    const uint shift1 = (lane & 1u) ? 24u : 8u;

    device const uint32_t * row_w = w + (uint64_t) row * words_per_row;
    device const ushort * row_s = scales + (uint64_t) row * groups_per_row;
    device const ushort * row_b = biases + (uint64_t) row * groups_per_row;

    // Every loop over the band is a compile-time trip count with no early exit,
    // so `acc` stays in registers. A `break` on the runtime span would make the
    // bound dynamic, the array would spill to thread-local memory - which is
    // device memory on this hardware - and each accumulate would cost a round
    // trip, which is worse than the weight traffic this kernel exists to avoid.
    // The tail band reads its last column repeatedly instead, and discards those
    // lanes when it writes.
    float acc[DENSE_AFFINE_TOKENS];
#pragma clang loop unroll(full)
    for (uint j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
        acc[j] = 0.0f;
    }
    const uint last = args.n_tok - 1u;

    for (uint g = 0u; g < groups_per_row; ++g) {
        const uint32_t word = row_w[g * 16u + word_in_group];
        const float q0 = (float) ((word >> shift0) & 0xffu);
        const float q1 = (float) ((word >> shift1) & 0xffu);
        const float s = dense_bf16(row_s + g);
        const float b = dense_bf16(row_b + g);
        const uint at = g * 64u + 2u * lane;
#pragma clang loop unroll(full)
        for (uint j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
            device const float * tok_x = x + (uint64_t) min(tok0 + j, last) * args.in_dim;
            const float x0 = tok_x[at];
            const float x1 = tok_x[at + 1u];
            acc[j] += s * (q0 * x0 + q1 * x1) + b * (x0 + x1);
        }
    }

#pragma clang loop unroll(full)
    for (uint j = 0u; j < DENSE_AFFINE_TOKENS; ++j) {
        const float total = dense_warp_sum(acc[j]);
        if (lane == 0u && j < span) {
            y[(uint64_t) (tok0 + j) * args.out_dim + row] = total;
        }
    }
}

/* The routed expert stages over MLX native packed affine weights.
 *
 * The third pair in this directory, after `experts.cuh` (NVFP4, the `modelopt`
 * export) and `experts_kblock.cuh` (the `ggml` super-blocks, the GGUF export).
 * This one exists for a container that is neither: **the multi-token-prediction
 * head**, which this family publishes as a directory of its own -
 * `Qwen3.6-35B-A3B-MTP-4bit`, split from `Qwen/Qwen3.6-35B-A3B` for `mlx-vlm`'s
 * speculative decoding - and which is quantized at 4-bit affine, group 64.
 *
 * `shiso_engine::moe::expert_entry` has named these two entries since
 * `ExpertCodec::Q4Affine` was added, and `shiso_kerneltest::qwen35moe::q4affine`
 * grades them. **On this backend neither existed**: the Metal module builds
 * them, the CUDA one never did, and nothing had noticed because no CUDA
 * container had carried affine EXPERTS - the dense family's affine weights are
 * all plain projections, which `cuda/dense`'s `qmv` entries serve. The head is
 * the first routed block in this tree stored this way.
 *
 * # The arithmetic is the dense entry's, factored the same way
 *
 *   sum_i (q_i * scale + bias) * x_i
 *     = scale * sum_i (q_i * x_i) + bias * sum_i x_i
 *
 * evaluated per group of 64. That is `kernel_dense_affine_qmv_4_f32` in
 * `cuda/dense/dense_affine.cuh` line for line, including the lane decomposition
 * - two weights a lane a group, `word_in_group = lane / 4`, the two nibble
 * shifts - and including the order the two terms are added in. It is written
 * out here rather than included because a module is a separately compiled image
 * on this backend and `cuda/qwen35moe` does not include `cuda/dense`; what
 * holds the two together is that they are the same six lines over the same
 * layout, and that layout is `crates/shiso-modelfile/src/mlx_quant.rs`, which is
 * graded against MLX.
 *
 * # The slab strides are derived, not passed
 *
 * An expert's words are `out_dim * in_dim / 8`, its scales and biases
 * `out_dim * in_dim / 64` each. A caller cannot state a stride that disagrees
 * with the widths, which is the same rule `shiso_args_q35moe_experts` already
 * enforces for the other two codecs.
 */
#ifndef SHISO_Q35MOE_EXPERTS_AFFINE_CUH
#define SHISO_Q35MOE_EXPERTS_AFFINE_CUH

#include "q35moe_common.cuh"

/* Weights one affine group covers. */
#define SHISO_Q35MOE_AFFINE_GROUP 64u
/* Four-bit weights one packed 32-bit word holds. */
#define SHISO_Q35MOE_AFFINE_PER_WORD 8u

/* Widens one bf16 value by placing its 16 bits into the high half of an f32.
 *
 * `dense_bf16`'s body; see the header for why it is spelled twice rather than
 * included. */
__device__ __forceinline__ float shiso_q35moe_bf16(const unsigned short *p) {
    return __uint_as_float((unsigned) (*p) << 16);
}

/* One expert's row of an affine slab, dotted against one activation row.
 *
 * Returns a value meaningful in every lane. `in_dim` is a whole number of
 * groups by `shiso_engine::projection::affine_geometry`'s refusal, which is
 * where this format's tail is handled - a width the walk below would drop the
 * last group of never reaches a launch.
 */
__device__ __forceinline__ float shiso_q35moe_affine_row_dot(
        const unsigned *__restrict__ words,
        const unsigned short *__restrict__ scales,
        const unsigned short *__restrict__ biases,
        const float *__restrict__ x,
        int in_dim,
        unsigned lane) {
    const unsigned groups = (unsigned) in_dim / SHISO_Q35MOE_AFFINE_GROUP;
    const unsigned word_in_group = lane / 4u;
    const unsigned shift0 = ((2u * lane) % SHISO_Q35MOE_AFFINE_PER_WORD) * 4u;
    const unsigned shift1 = ((2u * lane + 1u) % SHISO_Q35MOE_AFFINE_PER_WORD) * 4u;

    float acc = 0.0f;
    for (unsigned g = 0u; g < groups; ++g) {
        const unsigned word = words[g * 8u + word_in_group];
        const float q0 = (float) ((word >> shift0) & 0x0Fu);
        const float q1 = (float) ((word >> shift1) & 0x0Fu);
        const float x0 = x[g * SHISO_Q35MOE_AFFINE_GROUP + 2u * lane];
        const float x1 = x[g * SHISO_Q35MOE_AFFINE_GROUP + 2u * lane + 1u];
        acc += shiso_q35moe_bf16(scales + g) * (q0 * x0 + q1 * x1)
             + shiso_q35moe_bf16(biases + g) * (x0 + x1);
    }
    return shiso_q35moe_warp_sum(acc);
}

/* The staged gate and up projections over affine experts, in one launch.
 *
 * Grid `(ceil(out_dim / 4), min(columns, n_expert), 1)`, block `(128, 1, 1)`,
 * and the same column sweep the other two pairs in this directory document.
 */
extern "C" __global__ void kernel_q35moe_experts_gate_up_q4affine_f32(
        const struct shiso_args_q35moe_experts args,
        const unsigned *__restrict__ gate_w,
        const unsigned short *__restrict__ gate_s,
        const unsigned short *__restrict__ gate_b,
        const unsigned *__restrict__ up_w,
        const unsigned short *__restrict__ up_s,
        const unsigned short *__restrict__ up_b,
        const float *__restrict__ x,
        const int32_t *__restrict__ sorted_cols,
        const int32_t *__restrict__ bounds,
        const int32_t *__restrict__ active,
        float *__restrict__ gate_out,
        float *__restrict__ up_out) {
    if ((int32_t) blockIdx.y >= active[args.n_expert]) {
        return;
    }
    const int expert = (int) active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    const unsigned lane = threadIdx.x & 31u;
    const int row = (int) (blockIdx.x * SHISO_Q35MOE_EXPERT_WARPS + (threadIdx.x >> 5u));
    if (row >= args.out_dim) {
        return;
    }
    const int64_t words_row = (int64_t) args.in_dim / (int64_t) SHISO_Q35MOE_AFFINE_PER_WORD;
    const int64_t group_row = (int64_t) args.in_dim / (int64_t) SHISO_Q35MOE_AFFINE_GROUP;
    const int64_t w_base = ((int64_t) expert * args.out_dim + row) * words_row;
    const int64_t g_base = ((int64_t) expert * args.out_dim + row) * group_row;

    for (int c = c0; c < c1; ++c) {
        const int col = (int) sorted_cols[c];
        /* Column `col` is `token * n_used + slot`, and every slot of a token
         * reads the same activation row. */
        const float *xrow = x + (int64_t) (col / args.n_used) * args.in_dim;
        const float g = shiso_q35moe_affine_row_dot(
                gate_w + w_base, gate_s + g_base, gate_b + g_base, xrow, args.in_dim, lane);
        const float u = shiso_q35moe_affine_row_dot(
                up_w + w_base, up_s + g_base, up_b + g_base, xrow, args.in_dim, lane);
        if (lane == 0u) {
            gate_out[(int64_t) col * args.out_dim + row] = g;
            up_out[(int64_t) col * args.out_dim + row] = u;
        }
    }
}

/* The routed down projection over affine experts.
 *
 * Unlike the gate and up stage, each column reads its OWN input row: the
 * activation here is what that (token, slot) pair's own experts produced.
 */
extern "C" __global__ void kernel_q35moe_experts_down_q4affine_f32(
        const struct shiso_args_q35moe_experts args,
        const unsigned *__restrict__ down_w,
        const unsigned short *__restrict__ down_s,
        const unsigned short *__restrict__ down_b,
        const float *__restrict__ mid,
        const int32_t *__restrict__ sorted_cols,
        const int32_t *__restrict__ bounds,
        const int32_t *__restrict__ active,
        float *__restrict__ dst) {
    if ((int32_t) blockIdx.y >= active[args.n_expert]) {
        return;
    }
    const int expert = (int) active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    const unsigned lane = threadIdx.x & 31u;
    const int row = (int) (blockIdx.x * SHISO_Q35MOE_EXPERT_WARPS + (threadIdx.x >> 5u));
    if (row >= args.out_dim) {
        return;
    }
    const int64_t words_row = (int64_t) args.in_dim / (int64_t) SHISO_Q35MOE_AFFINE_PER_WORD;
    const int64_t group_row = (int64_t) args.in_dim / (int64_t) SHISO_Q35MOE_AFFINE_GROUP;
    const int64_t w_base = ((int64_t) expert * args.out_dim + row) * words_row;
    const int64_t g_base = ((int64_t) expert * args.out_dim + row) * group_row;

    for (int c = c0; c < c1; ++c) {
        const int col = (int) sorted_cols[c];
        const float *xrow = mid + (int64_t) col * args.in_dim;
        const float v = shiso_q35moe_affine_row_dot(
                down_w + w_base, down_s + g_base, down_b + g_base, xrow, args.in_dim, lane);
        if (lane == 0u) {
            dst[(int64_t) col * args.out_dim + row] = v;
        }
    }
}

#endif /* SHISO_Q35MOE_EXPERTS_AFFINE_CUH */

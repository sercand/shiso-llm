/* The routed expert stage this family needs and the sparse one could not
 * serve.
 *
 * `crates/shiso-engine/src/sparse/moe.rs` has a complete routed feed-forward
 * and eight extracted, fidelity-gated entries behind it. Six of them are
 * general - the router, the three sort passes, the activation, the combine and
 * the shared expert's gated add - and this family dispatches those unchanged.
 * The two that are not general are the expert stages themselves: the fused one
 * is a trellis cooperative launch and the staged one is templated on
 * `{Q5_1, Q8_0}`. Neither reads NVFP4, which is what this checkpoint's 30,720
 * expert tensors are, and that is the gap this file fills.
 *
 * # One tensor per expert, not a stride into a slab
 *
 * The sparse family addresses an expert by `W + expert * expert_bytes` into
 * one stacked slab. This checkpoint stores 256 separate tensors per layer per
 * projection, each with its own scale array and its own per-tensor `f32`. The
 * loader stacks them - the tensors are contiguous and the upload is ours to
 * order - so the entries below take the same stride form, with two additions
 * the sparse family's slab does not have: a second slab of `e4m3` group scales
 * at its own stride, and an array of one `f32` per expert.
 *
 * **The per-expert `f32` is an array and not a scalar**, which is the whole
 * difference from the projection entries beside this file. Every expert of a
 * layer has its own, they differ, and a kernel that read expert zero's for all
 * 256 would be wrong by whatever the spread is - a plausible residual, and the
 * next normalization very nearly removes the part of it that is uniform.
 *
 * # Grouped by expert, always, and the seam that avoids
 *
 * Both entries walk the expert-sorted column map the sort stage builds:
 * expert `e` owns `sorted[bounds[e] .. bounds[e + 1]]`. There is no second,
 * ungrouped path. At a prefill chunk the grouping is what keeps an expert's
 * weight row in L1 across the sixteen-odd columns routed to it; at one token
 * each expert owns exactly one column and the same code is the per-column form
 * with no grouping left to do. One path rather than two means no threshold at
 * which a model starts answering differently, which is the defect the FP8
 * cross-check case beside this exists to catch.
 *
 * `blockIdx.y` indexes the ACTIVE experts rather than all of them, through the
 * list the scan pass leaves in `active`: at one token eight experts receive a
 * column and 248 empty blocks per launch would be most of the launch. The host
 * bounds the extent by the column count, because a chunk cannot activate more
 * experts than it has columns.
 */
#ifndef SHISO_Q35MOE_EXPERTS_CUH
#define SHISO_Q35MOE_EXPERTS_CUH

#include "q35moe_common.cuh"

/* Warps per block of the expert entries; each owns one output row. */
#define SHISO_Q35MOE_EXPERT_WARPS 4
/* Threads per block of the expert entries. */
#define SHISO_Q35MOE_EXPERT_BLOCK (SHISO_Q35MOE_WARP * SHISO_Q35MOE_EXPERT_WARPS)

/* The four widths a routed expert stage is launched over.
 *
 * Sixteen bytes, one struct by value, and the PTX says so - see
 * `shiso_args_q35moe_fp8` for why that is the only spelling the packer
 * supports. Everything else a stage needs is derived here rather than passed:
 * an expert's weight slab stride is `out_dim * in_dim / 2`, its scale slab
 * stride is `out_dim * in_dim / 16`, and a caller cannot state one that
 * disagrees with the widths.
 */
struct shiso_args_q35moe_experts {
    /* Elements in one weight row, counted in weights. */
    int32_t in_dim;
    /* Weight rows of one expert. */
    int32_t out_dim;
    /* Routed experts the layer holds. */
    int32_t n_expert;
    /* Experts one token is routed to. */
    int32_t n_used;
};

/* One expert's row of an NVFP4 slab, dotted against one activation row.
 *
 * `shiso_q35moe_nvfp4_row_dot` is the whole of it, and that is the point: the
 * projection entries beside this file call the same function over the same
 * decomposition, so a change to how this format is walked cannot land in one
 * of the two and not the other. That has happened once already in this
 * directory - the projection entry kept a per-byte walk for a phase after the
 * expert stage stopped using one - and the two then read the same checkpoint
 * at 68 and 65 GB/s for different reasons.
 *
 * A lane takes whole SIXTEEN-WEIGHT GROUPS. This took one four-byte word per
 * lane per turn until 2026-09-22, and one weight per lane before that; the
 * table in the shared helper has what each step cost.
 */
__device__ __forceinline__ float shiso_q35moe_expert_dot(
        const unsigned char *__restrict__ packed,
        const unsigned char *__restrict__ scales,
        float global_scale,
        const float *__restrict__ x,
        int in_dim,
        unsigned lane) {
    return shiso_q35moe_nvfp4_row_dot(packed, scales, global_scale, x, in_dim, lane);
}

/* The staged gate and up projections over NVFP4 experts, in one launch.
 *
 * Grid `(ceil(out_dim / 4), min(columns, n_expert), 1)`, block `(128, 1, 1)`.
 *
 * One launch rather than two because the two share everything but the matrix:
 * the same activation row, the same column list, the same expert. Two launches
 * would read the activation row twice and walk the sorted map twice, and the
 * activation row is the operand that is not in cache.
 *
 * `gate_out` and `up_out` are `[columns, out_dim]` row-major, which is what
 * the activation stage and then the down stage read.
 */
extern "C" __global__ void kernel_q35moe_experts_gate_up_nvfp4_f32(
        const struct shiso_args_q35moe_experts args,
        const unsigned char *__restrict__ gate_w,
        const unsigned char *__restrict__ gate_s,
        const float *__restrict__ gate_g,
        const unsigned char *__restrict__ up_w,
        const unsigned char *__restrict__ up_s,
        const float *__restrict__ up_g,
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

    const int64_t packed_row = (int64_t) args.in_dim / SHISO_NVFP4_PER_BYTE;
    const int64_t scale_row = (int64_t) args.in_dim / SHISO_NVFP4_GROUP;
    const int64_t weight_base = (int64_t) expert * args.out_dim * packed_row + row * packed_row;
    const int64_t scale_base = (int64_t) expert * args.out_dim * scale_row + row * scale_row;
    const float gate_scale = gate_g[expert];
    const float up_scale = up_g[expert];

    /* One sweep over this expert's columns. The two weight rows stay in cache
     * across the sweep, which is the whole reason the columns were sorted. */
    for (int c = c0; c < c1; ++c) {
        const int col = (int) sorted_cols[c];
        /* Column `col` is `token * n_used + slot`, and every slot of a token
         * reads the same activation row. */
        const float *xrow = x + (int64_t) (col / args.n_used) * args.in_dim;
        const float g = shiso_q35moe_expert_dot(
                gate_w + weight_base, gate_s + scale_base, gate_scale, xrow, args.in_dim, lane);
        const float u = shiso_q35moe_expert_dot(
                up_w + weight_base, up_s + scale_base, up_scale, xrow, args.in_dim, lane);
        if (lane == 0u) {
            gate_out[(int64_t) col * args.out_dim + row] = g;
            up_out[(int64_t) col * args.out_dim + row] = u;
        }
    }
}

/* The routed down projection over NVFP4 experts.
 *
 * Grid `(ceil(out_dim / 4), min(columns, n_expert), 1)`, block `(128, 1, 1)`.
 *
 * `dst` is `[columns, out_dim]` row-major, which is column-major over
 * (token, slot) - and is exactly what `q4e_moe_combine_kernel` reads, at
 * `down[col * n_embd + e]`. Writing it the other way round would give the
 * combine a transposed block that is the right size.
 *
 * Unlike the gate and up stage, each column reads its OWN input row: the
 * activation here is what that (token, slot) pair's own experts produced.
 */
extern "C" __global__ void kernel_q35moe_experts_down_nvfp4_f32(
        const struct shiso_args_q35moe_experts args,
        const unsigned char *__restrict__ down_w,
        const unsigned char *__restrict__ down_s,
        const float *__restrict__ down_g,
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

    const int64_t packed_row = (int64_t) args.in_dim / SHISO_NVFP4_PER_BYTE;
    const int64_t scale_row = (int64_t) args.in_dim / SHISO_NVFP4_GROUP;
    const int64_t weight_base = (int64_t) expert * args.out_dim * packed_row + row * packed_row;
    const int64_t scale_base = (int64_t) expert * args.out_dim * scale_row + row * scale_row;
    const float scale = down_g[expert];

    for (int c = c0; c < c1; ++c) {
        const int col = (int) sorted_cols[c];
        const float *xrow = mid + (int64_t) col * args.in_dim;
        const float v = shiso_q35moe_expert_dot(
                down_w + weight_base, down_s + scale_base, scale, xrow, args.in_dim, lane);
        if (lane == 0u) {
            dst[(int64_t) col * args.out_dim + row] = v;
        }
    }
}

#endif /* SHISO_Q35MOE_EXPERTS_CUH */

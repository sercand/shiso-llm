/* The routed expert stages over `ggml` super-block codecs.
 *
 * The sibling of `experts.cuh`, which reads this family's `modelopt` NVFP4
 * export. This one reads the `GGUF` export of the same architecture, and it
 * exists because that container stores the routed experts in **two codecs at
 * once**: `unsloth`'s `Qwen3.6-35B-A3B-UD-Q4_K_XL` keeps most experts'
 * gate and up projections in `Q4_K` and most down projections in `Q5_K`. A
 * routed block of that file therefore cannot be served by one entry per stage
 * pair, which is why `ExpertCodec` became a per-STAGE property rather than a
 * per-block one.
 *
 * And "most" is the operative word: a "UD" container is DYNAMIC, so the codec
 * varies by LAYER as well as by stage. This one holds `ffn_down_exps` in q5_K
 * for thirty-eight slots and q6_K for three, and `ffn_gate_exps`/`ffn_up_exps`
 * in q4_K for forty and q5_K for one. Every super-block codec this tree decodes
 * is instantiated below for that reason rather than on demand.
 *
 * # Why the checkpoint is worth a second pair of entries at all
 *
 * The NVFP4 export this family was ported against is a **bad quantization**:
 * its body collapses onto one CJK token, which `scripts/qwen35moe_mtp_reference.py`
 * establishes in `transformers` with no part of this tree involved, and which
 * llama.cpp's fluent output from the GGUF of the same base model is the control
 * for. So these entries are not an alternative container for a working model;
 * for now they are the only container of this architecture that answers.
 *
 * # The walk, and why it is the coalesced one
 *
 * A lane takes whole FOUR-ELEMENT CHUNKS strided by the warp, which is the
 * decomposition `shiso_dequant_plane_t4` already hands over: chunk `c` of a row
 * lives in block `c / per_block` at index `c % per_block`, and the four
 * activations it multiplies are `((const float4 *) x)[c]`. Lanes are therefore
 * adjacent in `c`, so one `float4` load per lane per turn is 32 lanes times 16
 * bytes of CONTIGUOUS activation - four whole cache lines, perfectly coalesced,
 * with none of the 64-byte lane stride the NVFP4 walk beside this has to accept
 * because its group is sixteen weights wide.
 *
 * Per four weights a lane issues one weight-block read (in cache across the
 * sixteen chunks of a block), one `float4`, and four fused multiply-adds.
 *
 * # One template, two codecs, and no third spelling of the arithmetic
 *
 * The decode is `shiso_dequant_plane_t4<codec>`, which is `generic.cuh`'s own
 * `shiso_dequantize_q4_K_t4` and `shiso_dequantize_q5_K_t4` - the functions the
 * plain matvec and the prefill GEMM read these formats through, and the ones
 * `shiso_kerneltest::qwen35::gemv` grades against an oracle dot product. A
 * routed stage that spelled the element order again would be a second place for
 * it to be wrong, and the two would disagree only on weights nobody looked at.
 *
 * **This directory now includes `cuda/generic`**, which its `KERNEL.toml` said
 * it did not. That comment described a file set with no reason to; this one has
 * one, and instantiating a shared codec template is the opposite of copying it.
 */
#ifndef SHISO_Q35MOE_EXPERTS_KBLOCK_CUH
#define SHISO_Q35MOE_EXPERTS_KBLOCK_CUH

#include "../generic/dequant_plane.cuh"
#include "q35moe_common.cuh"

/* One expert's row of a super-block slab, dotted against one activation row.
 *
 * `blocks` is the row's first block and `in_dim` its width in WEIGHTS; the
 * block count is derived, because a caller that stated one disagreeing with the
 * width would read a row of the right length out of the wrong place.
 *
 * Returns a value meaningful in every lane.
 */
template <typename codec>
__device__ __forceinline__ float shiso_q35moe_kblock_row_dot(
        const codec *__restrict__ blocks,
        const float *__restrict__ x,
        int in_dim,
        unsigned lane) {
    constexpr int per_block = shiso_dequant_plane_chunks<codec>();
    const int chunks = in_dim / 4;
    float acc = 0.0f;
    for (int c = (int) lane; c < chunks; c += SHISO_Q35MOE_WARP) {
        float4 w;
        shiso_dequant_plane_t4<codec>(blocks + c / per_block, c % per_block, &w);
        const float4 a = ((const float4 *) x)[c];
        acc = fmaf(w.x, a.x, acc);
        acc = fmaf(w.y, a.y, acc);
        acc = fmaf(w.z, a.z, acc);
        acc = fmaf(w.w, a.w, acc);
    }
    /* The tail is `in_dim % 4` weights, which this architecture's widths - 2048
     * and 512 - cannot produce, and which is here so that a future width that
     * breaks it is slow rather than wrong. */
    for (int k = chunks * 4 + (int) lane; k < in_dim; k += SHISO_Q35MOE_WARP) {
        float4 w;
        const int c = k / 4;
        shiso_dequant_plane_t4<codec>(blocks + c / per_block, c % per_block, &w);
        const float parts[4] = {w.x, w.y, w.z, w.w};
        acc = fmaf(parts[k & 3], x[k], acc);
    }
    return shiso_q35moe_warp_sum(acc);
}

/* Blocks one row of `in_dim` weights occupies in this codec. */
template <typename codec>
__device__ __forceinline__ int shiso_q35moe_kblock_row(int in_dim) {
    return in_dim / (4 * shiso_dequant_plane_chunks<codec>());
}

/* The staged gate and up projections over super-block experts, in one launch.
 *
 * Grid `(ceil(out_dim / 4), min(columns, n_expert), 1)`, block `(128, 1, 1)`,
 * and the same column sweep `experts.cuh` documents: one launch rather than two
 * because the two share the activation row, the sorted map and the expert, and
 * the activation row is the operand that is not in cache.
 *
 * Gate and up are ONE codec here. They are in the container this serves, and a
 * pair that were not would need two entries rather than a wider one, because
 * the codec is a template argument and not a value.
 */
template <typename codec>
__device__ __forceinline__ void shiso_q35moe_experts_gate_up_kblock(
        const struct shiso_args_q35moe_experts args,
        const codec *__restrict__ gate_w,
        const codec *__restrict__ up_w,
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
    const int64_t row_blocks = shiso_q35moe_kblock_row<codec>(args.in_dim);
    const int64_t base = ((int64_t) expert * args.out_dim + row) * row_blocks;
    for (int c = c0; c < c1; ++c) {
        const int col = (int) sorted_cols[c];
        /* Column `col` is `token * n_used + slot`, and every slot of a token
         * reads the same activation row. */
        const float *xrow = x + (int64_t) (col / args.n_used) * args.in_dim;
        const float g = shiso_q35moe_kblock_row_dot<codec>(gate_w + base, xrow, args.in_dim, lane);
        const float u = shiso_q35moe_kblock_row_dot<codec>(up_w + base, xrow, args.in_dim, lane);
        if (lane == 0u) {
            gate_out[(int64_t) col * args.out_dim + row] = g;
            up_out[(int64_t) col * args.out_dim + row] = u;
        }
    }
}

/* The routed down projection over super-block experts.
 *
 * Unlike the gate and up stage, each column reads its OWN input row: the
 * activation here is what that (token, slot) pair's own experts produced.
 */
template <typename codec>
__device__ __forceinline__ void shiso_q35moe_experts_down_kblock(
        const struct shiso_args_q35moe_experts args,
        const codec *__restrict__ down_w,
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
    const int64_t row_blocks = shiso_q35moe_kblock_row<codec>(args.in_dim);
    const int64_t base = ((int64_t) expert * args.out_dim + row) * row_blocks;
    for (int c = c0; c < c1; ++c) {
        const int col = (int) sorted_cols[c];
        const float *xrow = mid + (int64_t) col * args.in_dim;
        const float v = shiso_q35moe_kblock_row_dot<codec>(down_w + base, xrow, args.in_dim, lane);
        if (lane == 0u) {
            dst[(int64_t) col * args.out_dim + row] = v;
        }
    }
}

/* The entries themselves. C linkage and one name per (stage, codec), because
 * `shiso_engine::moe::expert_entry` resolves a name rather than a mangling -
 * see `crates/shiso-engine/src/sparse/capture.rs` for what reading a mangled
 * template name costs when it is subtly wrong. */
#define SHISO_Q35MOE_KBLOCK_ENTRIES(SUFFIX, CODEC)                                      \
    extern "C" __global__ void kernel_q35moe_experts_gate_up_##SUFFIX##_f32(             \
            const struct shiso_args_q35moe_experts args,                                \
            const unsigned char *__restrict__ gate_w,                                   \
            const unsigned char *__restrict__ up_w,                                     \
            const float *__restrict__ x,                                                \
            const int32_t *__restrict__ sorted_cols,                                    \
            const int32_t *__restrict__ bounds,                                          \
            const int32_t *__restrict__ active,                                          \
            float *__restrict__ gate_out,                                                \
            float *__restrict__ up_out) {                                                \
        shiso_q35moe_experts_gate_up_kblock<struct CODEC>(                               \
                args, (const struct CODEC *) gate_w, (const struct CODEC *) up_w, x,     \
                sorted_cols, bounds, active, gate_out, up_out);                          \
    }                                                                                    \
    extern "C" __global__ void kernel_q35moe_experts_down_##SUFFIX##_f32(                \
            const struct shiso_args_q35moe_experts args,                                 \
            const unsigned char *__restrict__ down_w,                                    \
            const float *__restrict__ mid,                                               \
            const int32_t *__restrict__ sorted_cols,                                     \
            const int32_t *__restrict__ bounds,                                          \
            const int32_t *__restrict__ active,                                          \
            float *__restrict__ dst) {                                                   \
        shiso_q35moe_experts_down_kblock<struct CODEC>(                                  \
                args, (const struct CODEC *) down_w, mid, sorted_cols, bounds, active,   \
                dst);                                                                    \
    }

SHISO_Q35MOE_KBLOCK_ENTRIES(q4k, shiso_block_q4_K)
SHISO_Q35MOE_KBLOCK_ENTRIES(q5k, shiso_block_q5_K)
SHISO_Q35MOE_KBLOCK_ENTRIES(q6k, shiso_block_q6_K)
SHISO_Q35MOE_KBLOCK_ENTRIES(q8_0, shiso_block_q8_0)

#endif /* SHISO_Q35MOE_EXPERTS_KBLOCK_CUH */

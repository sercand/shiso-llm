/* Shared geometry, reduction shapes and argument blocks for the qwen35 set.
 *
 * The CUDA half of `crates/shiso-metal/metal/qwen35/q35_common.h`, written
 * against the same contract. Unlike `dense`, this family is NOT parametric:
 * the delta rule's state row, its chunk width and its unit split are compiled
 * in, exactly as they are in both the Metal set and the CUDA source below.
 *
 * # Where the arithmetic comes from, and what this file is
 *
 * Nothing here is a byte copy, so no unit in this directory carries a
 * provenance header and `scripts/q4e_ptx_oracle.sh --mode=fidelity` reports
 * the module as generated. What it IS is a transcription, and the thing
 * transcribed is in this tree twice over:
 *
 *   - `kernels/cuda/q4e/q4e.cuh` carries `q4e_block_sum` and `q4e_rope_mrope`,
 *     extracted verbatim from the reference implementation its own provenance
 *     header names, and held byte-for-byte by the fidelity gate. The two
 *     reductions below are their fold order, and
 *     `q35_rope_mrope_interleaved` is `q4e_rope_mrope` with the per-axis pair
 *     budget removed - see that function's own note.
 *   - `crates/shiso-metal/metal/qwen35/q35_common.h` is the Metal port of the
 *     same helpers, and is what produced the committed goldens.
 *
 * Where the two disagree the Metal one wins, because the oracles in
 * `crates/shiso-kerneltest/src/qwen35/` grade against it and the goldens were
 * captured through it.
 *
 * # Why the reductions are spelled out
 *
 * `KERNELS.md`: reductions only go through `q35_warp_sum`/`q35_block_sum`,
 * because a golden that pins a decode step pins the order its partial sums
 * were added in. Same shape on both sides: a decreasing-offset left fold, lane
 * 0 publishing to scratch, warp 0 re-summing, thread 0 broadcasting through
 * the last slot.
 */
#ifndef SHISO_Q35_COMMON_CUH
#define SHISO_Q35_COMMON_CUH

#include <cuda_fp16.h>
#include <stdint.h>

/* GDN state row, `Q35_GDN_D`. Both delta-rule kernels build their register
 * layout from it, so it is a constant rather than an argument on either side. */
#define Q35_GDN_D 128u
/* Tokens per chunked run, `Q35_GDN_CHUNK`. */
#define Q35_GDN_CHUNK 64u
/* Query rows staged at a time in the chunk kernel's output phase. */
#define Q35_GDN_QTILE 32u
/* Lanes per value column in the chunk kernel: 128 * this is its block. */
#define Q35_GDN_CSPLIT 8u
/* Lanes per state column in the recurrent kernel: 128 * this is its block. */
#define Q35_GDN_SPLIT 4u
/* Queries per prefill threadgroup. */
#define Q35_ATTN_QT 16u
/* Keys per prefill tile. */
#define Q35_ATTN_KT 32u
/* Warps per attention block (256 threads). */
#define Q35_ATTN_WARPS 8u
/* Accumulator slots one lane holds: Q35_ATTN_HD / 32. */
#define Q35_ATTN_LANE_SLOTS 8u
/* Widest attention head this set stages. */
#define Q35_ATTN_HD 256u
/* conv_kernel 4 minus the current token. */
#define Q35_CONV_HIST 3u
/* 32 warp partials plus the broadcast slot; see `q35_block_sum`. */
#define Q35_BLOCK_SUM_SLOTS 33u
/* One quarter of a state row: Q35_GDN_D / 4. */
#define Q35_GDN_QUARTER 32u
/* Recurrent scratch: block-sum slots plus the published k and q rows. */
#define Q35_GDN_REC_SLOTS (Q35_BLOCK_SUM_SLOTS + 2u * Q35_GDN_D)
/* Every lane of a warp, which is the participation mask every shuffle here
 * takes: each one sits at a point the whole warp reaches. */
#define Q35_FULL_WARP 0xffffffffu

/* Dynamic shared memory, declared once with an alignment every view below
 * satisfies. Each kernel reinterprets this one array; the driver hands over a
 * 16-byte aligned base. */
extern __shared__ __align__(16) char q35_smem[];

/* Inclusive sum over one warp, left-folded in decreasing-offset order so lane 0
 * ends up holding the total. Spelled out rather than reduced with a library
 * call for the reason the Metal twin is: a fixed summation order is part of
 * the contract, and `q4e_block_sum` folds the same way. */
__device__ __forceinline__ float q35_warp_sum(float v) {
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
        v += __shfl_down_sync(Q35_FULL_WARP, v, off);
    }
    return v;
}

/* Block-wide reduction over at most 32 warps.
 *
 * `scratch` needs Q35_BLOCK_SUM_SLOTS floats and is passed in because this is
 * inlined into a kernel that owns the shared allocation. Every thread of the
 * block must reach this call: it contains two barriers. */
__device__ __forceinline__ float q35_block_sum(
        float v, float *scratch, unsigned tid, unsigned nthreads) {
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    v = q35_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    __syncthreads();
    const unsigned n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = q35_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[Q35_BLOCK_SUM_SLOTS - 1u] = v;
    }
    __syncthreads();
    return scratch[Q35_BLOCK_SUM_SLOTS - 1u];
}

/* The canonical combination of four unit-chain partials: the quarters pair up
 * before they cross. Floating-point addition is commutative but not
 * associative, so the pairing is the whole content of this function. */
__device__ __forceinline__ float q35_canonical4(float q0, float q1, float q2, float q3) {
    return (q0 + q1) + (q2 + q3);
}

/* Fused multiply-add, wrapped once so an inner product reads as one decision
 * and no call site can pick up a non-fused multiply by accident. `q4e/gdn.cu`
 * spells the same thing `fmaf`. */
__device__ __forceinline__ float q35_fma(float a, float b, float c) {
    return fmaf(a, b, c);
}

/* The three activations, and why each spells its exponential the way it does.
 *
 * This directory's manifest drops `--use_fast_math`, so `expf` here is the
 * ACCURATE library entry and `__expf` is the reduced-precision intrinsic. That
 * is a per-call choice rather than a directory-wide one, which is the whole
 * reason the flag had to go: under it there is no spelling that opts back out.
 *
 * All three take `__expf`, because both references do. `q4e/gdn.cu` writes
 * `__expf` in the conv's silu and in both gates, and the Metal twins sit below
 * `q35_common.h`'s `math_mode(safe)` region - which stops at `q35_fma`
 * deliberately - so their `exp` is the library's fast one too. The accurate
 * entry here would be a third answer, not a better one. The rope helpers above
 * are the opposite case and take the accurate `powf`/`sinf`/`cosf`. */

/* x / (1 + exp(-x)), written as the source writes it rather than as exp2. */
__device__ __forceinline__ float q35_silu(float a) {
    return a / (1.0f + __expf(-a));
}

__device__ __forceinline__ float q35_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

/* softplus with the overflow guard the decay gate relies on: above 20 the
 * exp() term is far below the ulp of x, and below it the naive form overflows.
 * `q4e_gdn_gates_kernel` guards at the same 20 and spells the pair
 * `log1pf(__expf(x))`; the Metal twin hand-rolls a log1p because MSL has none,
 * and its own note says the target is "within an ulp of the libm function the
 * port is written against", which is this one. */
__device__ __forceinline__ float q35_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(__expf(x));
}

/* Physical row of a logical KV position, through the session page table.
 *
 * `pages[logical page]` is the physical page holding that span, so a position
 * becomes `pages[p / page_tokens] * page_tokens + p % page_tokens`. A null
 * table means the caller's rows are already physical, which exists so a
 * single-page fixture can launch without one.
 *
 * NOT `q4e_kv_row`: that helper reads its page width off the compiled-in
 * `SHS_Q4E_PAGE_TOKENS`, and this family's width is a field of the argument
 * block. Feeding one where the other belongs is the shape of defect
 * `HANDOFF.md` opens with. */
__device__ __forceinline__ uint64_t q35_kv_row(
        const int32_t *pages, uint64_t position, uint32_t page_tokens) {
    if (pages == nullptr) {
        return position;
    }
    return (uint64_t) pages[position / page_tokens] * page_tokens + (position % page_tokens);
}

/* Row `t`'s own page table inside a packed block of them.
 *
 * A decode step may carry one row per SESSION, and two sessions share no page:
 * a launch that read one table for every row would answer row one out of row
 * zero's conversation, fluently and at the right shape. So the table a row
 * reads is selected here, by the row index the grid already carries.
 *
 * `stride` is entries between one row's table and the next, and `0` is the
 * shared case rather than a sentinel - `pages + t * 0` is `pages` for every
 * row, which is exactly what a prefill chunk and a single-session step want.
 * A null table stays null, because that is the "no paging, position IS the
 * row" convention `q35_kv_row` answers for. */
__device__ __forceinline__ const int32_t *q35_row_pages(
        const int32_t *pages, unsigned t, uint32_t stride) {
    return (pages == nullptr) ? pages : pages + (uint64_t) t * stride;
}

/* NeoX rope over the leading `n_rot` dimensions of a block-resident row: pair
 * (i, i + n_rot/2), everything past n_rot passes through untouched.
 *
 * `powf`, `sinf` and `cosf` are the accurate library entries, as
 * `q4e_rope_mrope` uses the accurate `powf`/`sincosf` and the Metal twin the
 * accurate builtins: the ladder spans six decades of base and the
 * reduced-precision intrinsics lose the far end of it. That is enforced by
 * this directory's manifest dropping `--use_fast_math` rather than by a
 * spelling here, because under that flag nvcc rewrites all three calls and no
 * source spelling opts back out. */
__device__ __forceinline__ void q35_rope_neox(
        float *v, unsigned n_rot, float base, float pos,
        unsigned lane, unsigned nlanes) {
    const unsigned n_half = n_rot / 2u;
    for (unsigned i = lane; i < n_half; i += nlanes) {
        const float theta = pos * powf(base, -2.0f * (float) i / (float) n_rot);
        const float s = sinf(theta);
        const float c = cosf(theta);
        const float x0 = v[i];
        const float x1 = v[i + n_half];
        v[i] = x0 * c - x1 * s;
        v[i + n_half] = x0 * s + x1 * c;
    }
}

/* Interleaved multimodal rope over the leading `n_rot` dimensions.
 *
 * Same pairing and same frequency ladder as `q35_rope_neox`; the one
 * difference is where the position comes from. Pair `i` belongs to section
 * `i % 3`, and the sections are the temporal, height and width streams in that
 * order.
 *
 * `q4e_rope_mrope` deals the pairs the same way and then adds a per-axis
 * budget: once an axis has taken `sec_*` pairs the rest fall back to the
 * temporal stream. This family's argument block carries no such field -
 * `KERNELS.md`: "The two `_mrope` entry points reuse those two blocks
 * unchanged: three position streams are three more buffers, not more fields" -
 * so the budget is absent here rather than passed as a large number. Dealing
 * and budget are the same function when every budget is at least `n_rot / 6`,
 * which is what this checkpoint sets them to.
 *
 * Three agreeing streams reproduce `q35_rope_neox` exactly, which is a
 * property to reason with and NOT one this file leans on: a chunk with no
 * media is dispatched to the one-dimensional entry points instead. */
__device__ __forceinline__ void q35_rope_mrope_interleaved(
        float *v, unsigned n_rot, float base,
        float pos_t, float pos_h, float pos_w,
        unsigned lane, unsigned nlanes) {
    const unsigned n_half = n_rot / 2u;
    for (unsigned i = lane; i < n_half; i += nlanes) {
        const unsigned section = i % 3u;
        const float pos = (section == 0u) ? pos_t : ((section == 1u) ? pos_h : pos_w);
        const float theta = pos * powf(base, -2.0f * (float) i / (float) n_rot);
        const float s = sinf(theta);
        const float c = cosf(theta);
        const float x0 = v[i];
        const float x1 = v[i + n_half];
        v[i] = x0 * c - x1 * s;
        v[i + n_half] = x0 * s + x1 * c;
    }
}

/* --- argument blocks ------------------------------------------------------
 *
 * Field order is the binding contract `crates/shiso-engine/src/launch.rs`
 * mirrors, and each block carries a static assertion on its size, which the
 * Rust side restates. A launch whose argument block is one field out reads a
 * stride as an extent and computes a product that is confidently wrong; that
 * pair of assertions is the whole of what keeps the two declarations from
 * drifting. */

/* Causal depthwise conv over the q|k|v channels plus silu. */
struct q35_args_gdn_conv {
    uint32_t channels;
    uint32_t conv_kernel;
    uint32_t n_tok;
};
static_assert(sizeof(struct q35_args_gdn_conv) == 12, "ConvArgs is 12 bytes");

/* In-place L2 normalization of one family of heads inside a strided row. */
struct q35_args_gdn_l2norm {
    uint32_t head_dim;
    uint32_t n_heads;
    uint32_t stride;
    uint32_t offset;
};
static_assert(sizeof(struct q35_args_gdn_l2norm) == 16, "L2NormArgs is 16 bytes");

/* decay = softplus(alpha_proj + dt_bias) * a, beta = sigmoid(beta_proj). */
struct q35_args_gdn_gates {
    uint32_t n_head;
    uint32_t n_tok;
};
static_assert(sizeof(struct q35_args_gdn_gates) == 8, "GateArgs is 8 bytes");

/* Per-head RMSNorm of the delta-rule output, then the sigmoid output gate. */
struct q35_args_gdn_out_gate {
    uint32_t head_dim;
    uint32_t n_head;
    float eps;
};
static_assert(sizeof(struct q35_args_gdn_out_gate) == 12, "OutGateArgs is 12 bytes");

/* Sequential gated delta rule, one block per value head. The chunked form
 * shares this block field for field. */
struct q35_args_gdn_recurrent {
    uint32_t head_dim;
    uint32_t n_head_k;
    uint32_t n_head_v;
    uint32_t k_offset;
    uint32_t v_offset;
    uint32_t stride;
    uint32_t n_tok;
};
static_assert(sizeof(struct q35_args_gdn_recurrent) == 28, "RecurrentArgs is 28 bytes");

/* Per-head RMSNorm and partial rope of the q projection, plus the raw
 * output-gate slice that rides beside each query in the same row. */
struct q35_args_q_norm_rope {
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_rot;
    uint32_t n_tok;
    float rope_base;
    float eps;
};
static_assert(sizeof(struct q35_args_q_norm_rope) == 24, "QNormRopeArgs is 24 bytes");

/* Key norm + rope into the f16 cache, with the value copied beside it. */
struct q35_args_k_norm_rope {
    uint32_t head_dim;
    uint32_t n_head_kv;
    uint32_t n_rot;
    uint32_t n_tok;
    float rope_base;
    float eps;
    /* Rows per KV page: a logical position becomes a physical row only through
     * the session page table. See `q35_kv_row`. */
    uint32_t page_tokens;
    /* Entries between one row's page table and the next, or 0 when every row
     * of this launch reads the table the pointer names. See `q35_row_pages`. */
    uint32_t page_stride;
};
static_assert(sizeof(struct q35_args_k_norm_rope) == 32, "KNormRopeArgs is 32 bytes");

/* Causal GQA over a contiguous f16 cache, one block per (head, token). The
 * prefill entry shares this block field for field. */
struct q35_args_attn_decode {
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_tok;
    uint32_t page_tokens;
    /* Entries between one row's page table and the next, or 0 when every row
     * of this launch reads the table the pointer names. See `q35_row_pages`. */
    uint32_t page_stride;
};
static_assert(sizeof(struct q35_args_attn_decode) == 24, "AttnArgs is 24 bytes");

/* out = attn * sigmoid(gate), the gate half of the q projection. */
struct q35_args_attn_gate {
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_tok;
};
static_assert(sizeof(struct q35_args_attn_gate) == 12, "AttnGateArgs is 12 bytes");

#endif /* SHISO_Q35_COMMON_CUH */

/* Cross-family device helpers shared by every qwen4exp translation unit.
 *
 * Each of these is called from more than one kernel family, which
 * is why it lives here rather than in a single unit: q4e_block_sum from seven
 * families, q4e_block_argmax from the router and the sampler, q4e_rope_mrope
 * from five sites, q4e_kv_row from every kernel that touches the KV cache.
 *
 * Family-local helpers deliberately stay in their own unit.
 */

#ifndef SHISO_Q4E_CUH
#define SHISO_Q4E_CUH

#include "shiso_dev_inc/common.cuh"
#include "shiso_dev_inc/tensor.cuh"
#include "q4e_page.cuh"


__device__ static float q4e_block_sum(float v) {
    __shared__ float s[32];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    v = warp_sum_f32(v);
    if (lane == 0u) s[warp] = v;
    __syncthreads();
    const uint32_t n_warp = (blockDim.x + 31u) >> 5u;
    v = (threadIdx.x < n_warp) ? s[threadIdx.x] : 0.0f;
    if (warp == 0u) v = warp_sum_f32(v);
    __shared__ float total;
    if (threadIdx.x == 0u) total = v;
    __syncthreads();
    return total;
}

__device__ static float q4e_block_max(float v) {
    __shared__ float s[32];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    v = warp_max_f32(v);
    if (lane == 0u) s[warp] = v;
    __syncthreads();
    const uint32_t n_warp = (blockDim.x + 31u) >> 5u;
    v = (threadIdx.x < n_warp) ? s[threadIdx.x] : -INFINITY;
    if (warp == 0u) v = warp_max_f32(v);
    __shared__ float total;
    if (threadIdx.x == 0u) total = v;
    __syncthreads();
    return total;
}

/* Grouped RMSNorm: the mean square is taken over one n_embd stream, but the
 * affine vector spans all hc * n_embd.  Normalizing over the full width
 * instead is a common and silent mistake. */
__device__ __forceinline__ static void q4e_store(float *p, float v) { *p = v; }
__device__ __forceinline__ static void q4e_store(__half *p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ static float q4e_load(const float *p) { return *p; }
__device__ __forceinline__ static float q4e_load(const __half *p) { return __half2float(*p); }

/* Block-wide argmax, ties resolved toward the lower index.  The tie rule is
 * load bearing: it is what a stable descending argsort does, and the router
 * has to agree with the reference on which expert a tie picks. */
__device__ static void q4e_block_argmax(float v, int32_t i, float *out_v, int32_t *out_i) {
    __shared__ float s_v[32];
    __shared__ int32_t s_i[32];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    for (uint32_t d = 16u; d > 0u; d >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, d);
        const int32_t oi = __shfl_down_sync(0xffffffffu, i, d);
        if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) { v = ov; i = oi; }
    }
    if (lane == 0u) { s_v[warp] = v; s_i[warp] = i; }
    __syncthreads();
    const uint32_t n_warp = (blockDim.x + 31u) >> 5u;
    if (warp == 0u) {
        v = (lane < n_warp) ? s_v[lane] : -INFINITY;
        i = (lane < n_warp) ? s_i[lane] : -1;
        for (uint32_t d = 16u; d > 0u; d >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, d);
            const int32_t oi = __shfl_down_sync(0xffffffffu, i, d);
            if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) { v = ov; i = oi; }
        }
        if (lane == 0u) { s_v[0] = v; s_i[0] = i; }
    }
    __syncthreads();
    *out_v = s_v[0];
    *out_i = s_i[0];
}

__device__ static void q4e_rope_neox(
        float *v, uint32_t n_rot, float base, uint32_t pos, uint32_t lane, uint32_t nlanes) {
    const uint32_t half = n_rot / 2u;
    for (uint32_t i = lane; i < half; i += nlanes) {
        /* Accurate powf/sincosf, not the fast-math intrinsics: the angle feeds
         * every attention layer, and __powf's error at base 1e7 is a
         * systematic rotation offset rather than noise that averages out.
         * RoPE is far off the critical path, so the precision is free. */
        const float theta = (float)pos * powf(base, -(float)(2u * i) / (float)n_rot);
        float s, c;
        sincosf(theta, &s, &c);
        const float x0 = v[i];
        const float x1 = v[i + half];
        v[i]        = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

/* Interleaved multimodal RoPE.
 *
 * mrope_section splits the n_rot/2 frequency pairs across three position axes
 * -- time, height, width -- interleaved rather than chunked, so every axis
 * keeps a spread of frequencies instead of a contiguous band: pair i belongs
 * to axis i % 3.  Qwen3.8-Flash-Next uses [11, 11, 10] over 32 pairs, which
 * tiles exactly, so the budget check below never fires for this checkpoint; it
 * is here because the rule is general and a future section triple need not
 * tile.  Anything past its section's budget falls back to the time axis, which
 * is what the reference does.
 *
 * With pos_t == pos_h == pos_w this is bit-identical to q4e_rope_neox -- same
 * expression, same iteration order, same casts -- which is what makes a
 * text-only prompt a no-op.  q4e_rope_neox is kept beside it as the reference
 * that property is stated against; the gate that actually checks it is an
 * end-to-end SHS_QWEN4EXP_TRACE diff of a text prompt against the pre-mRoPE
 * build, which exercises all four rope sites in place rather than in a
 * harness. */
__device__ static void q4e_rope_mrope(
        float *v, uint32_t n_rot, float base,
        int32_t pos_t, int32_t pos_h, int32_t pos_w,
        uint32_t sec_t, uint32_t sec_h, uint32_t sec_w,
        uint32_t lane, uint32_t nlanes) {
    const uint32_t half = n_rot / 2u;
    for (uint32_t i = lane; i < half; i += nlanes) {
        uint32_t axis = i % 3u;
        const uint32_t used = i / 3u;   /* pairs this axis has already taken */
        const uint32_t budget = (axis == 0u) ? sec_t : ((axis == 1u) ? sec_h : sec_w);
        if (used >= budget) axis = 0u;
        const int32_t p = (axis == 0u) ? pos_t : ((axis == 1u) ? pos_h : pos_w);
        /* Accurate powf/sincosf for the same reason as q4e_rope_neox. */
        const float theta = (float)(uint32_t)p * powf(base, -(float)(2u * i) / (float)n_rot);
        float s, c;
        sincosf(theta, &s, &c);
        const float x0 = v[i];
        const float x1 = v[i + half];
        v[i]        = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

/* Physical cache row of logical position p, through this sequence's page
 * table.  Every KV-touching kernel
 * below goes through this; a tile that is page-aligned and no wider than a
 * page translates its first position and walks the rest contiguously, which
 * is what keeps the translation to one lookup per tile. */
__device__ __forceinline__ static uint32_t q4e_kv_row(const int32_t *pages, uint32_t p) {
    return (uint32_t)pages[p >> SHS_Q4E_PAGE_SHIFT] * SHS_Q4E_PAGE_TOKENS +
           (p & SHS_Q4E_PAGE_MASK);
}

#endif /* SHISO_Q4E_CUH */

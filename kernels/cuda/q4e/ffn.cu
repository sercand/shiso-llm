/* The routed block's step launch: everything a decode step or a verify pass of
 * at most eight rows computes from the block input before the routed experts.
 *
 * Not an upstream unit, for the reason `modelopt.cu` is not.
 *
 * # One launch for what reads the block input
 *
 * `q4e_ffn_route_kernel` (one row) and `q4e_ffn_route_rows_kernel` (two to
 * eight) compute, from the block input `x` `[rows, in_dim]`:
 *
 * - the router's logits `[rows, n_expert]` - the bits `q4e_bf16_gemv_kernel`
 *   and `q4e_bf16_rows_kernel` write from the brain-float router
 *   `[n_expert, in_dim]` at `splits` slices a row;
 * - the shared expert's gate logit `[rows]` - the same kernels' bits for the
 *   brain-float gate `[1, in_dim]` at `gate_splits` slices;
 * - the routed selection and its sort - ids, weights, bounds, the active
 *   list, the sorted columns and their inverse - the bits
 *   `q4e_moe_route_sort_tiny_kernel` (moe.cu) writes from those logits;
 * - the shared expert: its MXFP8 gate and up, activated - the bits
 *   `q4e_swiglu_kernel` writes from what the group launches
 *   (`q4e_mxfp8_gemv_group_narrow_kernel`, `q4e_mxfp8_rows_group_kernel`)
 *   store - and its MXFP8 down projection, the bits `q4e_mxfp8_gemv_kernel`
 *   and `q4e_mxfp8_rows_kernel` write.
 *
 * At a decode step those were seven launches: the router (33 us in the model
 * at 79 GB/s) with the shared expert's gate and up beside it on a branch
 * (35 us), the one-block selection behind the router (11 us), the activation,
 * the shared down projection (20 us, beside the routed experts), and the gate
 * logit after the join (2 us). Each was a short grid with a ramp and a tail,
 * and the selection was a serial chain with the memory system idle.
 *
 * Here they are work items of one grid of 256-thread blocks, taken in order
 * from a counter (see `q4e_fr_main`): the shared gate and up first, then the
 * router's rows and the gate's row, then the shared down projection. The
 * router item that finishes last - a ticket taken after its stores, fenced -
 * runs the selection over the logits the others left in the cache, while the
 * shared down items stream; those wait for the gate-and-up items by a count.
 * Nothing is ordered by arrival except which block does the selection and
 * when a down item starts: every value is computed as before, from complete
 * inputs.
 *
 * GB10, kernel bench over 24 layers' weights (none in the cache), the
 * seven-launch chain with the shared expert forked as the model runs it
 * against this launch: one row 51.0 -> 39.3 us, two 59.8 -> 39.5, five
 * 65.1 -> 46.8, eight 74.1 -> 54.2, every output bit for bit. (At one and
 * two rows the engine reserves shared memory it does not use, to hold the
 * launch to two blocks a multiprocessor; see `moe::FUSED_ROUTE_SMEM`.)
 *
 * Each logit's arithmetic is `q4e_bf_step`'s (bf16.cu) at one output row a
 * warp: lane `l` owns the eight weights `[i + 8 l, +8)` of each 256-wide step
 * of its slice, one fused multiply-add per weight per activation row in
 * contraction order, the decreasing-offset fold, then the slices added in
 * slice order. Each MXFP8 row is `q4e_mx_rows`' (mxfp8.cuh): lane `l` owns
 * element pairs `l, l + 32, ...`, then the exact fold. How many rows share the
 * launch moves no add in either, so a verify row is the decode step's row bit
 * for bit. The selection is `q4e_moe_route_sort_tiny_kernel`'s lines,
 * unchanged but for reading the logits from the second-level cache (the first
 * is not coherent across blocks).
 */

#include <stdint.h>
#include <cuda_bf16.h>

#include "shiso_dev_inc/common.cuh"
#include "mxfp8.cuh"

/* Warps a block of the route launch runs, and the contraction elements a
 * warp covers a step (eight a lane). */
#define Q4E_FR_WARPS 8u
#define Q4E_FR_STEP 256u
/* The widest pass the route launch takes. */
#define Q4E_FR_ROWS 8u
/* The selection's limits (`q4e_moe_route_sort_tiny_kernel`'s). */
#define Q4E_FR_EXPERTS 512u
#define Q4E_FR_BLOCK 256u /* the block q4e_moe_route_kernel runs at */

/* Eight brain floats (sixteen bytes) as eight floats. */
__device__ __forceinline__ void q4e_fr_widen8(uint4 raw, float *v) {
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        v[2 * j] = __uint_as_float(w[j] << 16);
        v[2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
    }
}

/* One warp's slice `[i0, i1)` of one brain-float row against `NT` activation
 * rows, folded: `q4e_bf_step`'s lane order and fold, so the value lane 0
 * returns is that kernel's partial for this slice. Four steps' loads are
 * issued before the first is consumed. */
template <int NT>
__device__ __forceinline__ void q4e_fr_slice(
        const __nv_bfloat16 *__restrict__ wr, const float *__restrict__ x, uint32_t in_dim,
        uint32_t i0, uint32_t i1, uint32_t lane, float *acc) {
#pragma unroll
    for (int t = 0; t < NT; t++) {
        acc[t] = 0.0f;
    }
    constexpr int D = 4;
    for (uint32_t i = i0 + lane * 8u; i < i1; i += D * Q4E_FR_STEP) {
        uint4 raw[D];
#pragma unroll
        for (int d = 0; d < D; d++) {
            const uint32_t k = i + (uint32_t) d * Q4E_FR_STEP;
            raw[d] = k < i1 ? __ldg(reinterpret_cast<const uint4 *>(wr + k))
                            : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int d = 0; d < D; d++) {
            const uint32_t k = i + (uint32_t) d * Q4E_FR_STEP;
            if (k >= i1) {
                break;
            }
            float wv[8];
            q4e_fr_widen8(raw[d], wv);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const float4 *xp = reinterpret_cast<const float4 *>(x + (uint64_t) t * in_dim + k);
                const float4 xa = __ldg(xp);
                const float4 xb = __ldg(xp + 1);
                const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[t] = fmaf(wv[j], xv[j], acc[t]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        acc[t] = warp_sum_f32(acc[t]);
    }
}

/* One block's rows of a brain-float matrix `[out_dim, in_dim]` at `splits`
 * slices a row: warp `w` takes slice `w % splits` of row
 * `blockIdx * (8 / splits) + w / splits`, and the first warp of each row adds
 * the slices in order. `part` holds `8 * NT` floats. */
template <int NT>
__device__ __forceinline__ void q4e_fr_rows(
        float *__restrict__ out, const __nv_bfloat16 *__restrict__ w,
        const float *__restrict__ x, uint32_t in_dim, uint32_t out_dim, uint32_t splits,
        uint32_t block, float *__restrict__ part) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t slice = warp % splits;
    const uint32_t row = block * (Q4E_FR_WARPS / splits) + warp / splits;
    const uint32_t chunk = ((in_dim + splits - 1u) / splits + 7u) & ~7u;
    const uint32_t i0 = min(slice * chunk, in_dim);
    const uint32_t i1 = min(i0 + chunk, in_dim);
    float acc[NT];
    q4e_fr_slice<NT>(w + (uint64_t) min(row, out_dim - 1u) * in_dim, x, in_dim, i0, i1, lane,
                     acc);
    if (lane == 0u) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            part[warp * NT + t] = acc[t];
        }
    }
    __syncthreads();
    if (slice != 0u || row >= out_dim) {
        return;
    }
    for (uint32_t t = lane; t < (uint32_t) NT; t += 32u) {
        float v = part[warp * NT + t];
        for (uint32_t s = 1u; s < splits; s++) {
            v += part[(warp + s) * NT + t];
        }
        out[(uint64_t) t * out_dim + row] = v;
    }
}

/* The orderable unsigned key of a float (`q4e_route_key`, moe.cu). */
__device__ __forceinline__ uint32_t q4e_fr_key(float v) {
    if (v != v) return 0u;
    const uint32_t b = v == 0.0f ? 0u : __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

/* The selection's shared memory (`q4e_moe_route_sort_tiny_kernel`'s). */
struct q4e_fr_pick {
    int32_t ids[Q4E_FR_ROWS * 16u];
    int32_t counts[Q4E_FR_EXPERTS];
    int32_t bounds[Q4E_FR_EXPERTS];
    uint32_t warp[Q4E_FR_WARPS];
};

/* `q4e_moe_route_sort_tiny_kernel`'s body, line for line, but for the logits
 * read through the second-level cache and its shared arrays passed in.
 * 256 threads. */
__device__ __forceinline__ void q4e_fr_select(
        int32_t *ids, float *weights, const float *logits,
        int32_t *bounds, int32_t *active, int32_t *sorted, int32_t *pos,
        uint32_t n_tok, uint32_t n_expert, uint32_t n_used, q4e_fr_pick &sm) {
    int32_t *s_ids = sm.ids;
    int32_t *s_counts = sm.counts;
    int32_t *s_bounds = sm.bounds;
    uint32_t *s_warp = sm.warp;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    constexpr uint32_t PER = Q4E_FR_EXPERTS / 32u;

    for (uint32_t e = threadIdx.x; e < Q4E_FR_EXPERTS; e += blockDim.x) s_counts[e] = 0;

    if (warp < n_tok) {
        const float *row = logits + (uint64_t)warp * n_expert;
        float v[PER];
        float m = -INFINITY;
#pragma unroll
        for (uint32_t j = 0; j < PER; j++) {
            const uint32_t e = lane + 32u * j;
            v[j] = e < n_expert ? __ldcg(row + e) : -INFINITY;
            if (e < n_expert) m = fmaxf(m, v[j]);
        }
        m = warp_max_f32(m);
        const float s_max = __shfl_sync(0xffffffffu, m, 0);
        float warp_total = 0.0f;
#pragma unroll
        for (uint32_t vw = 0; vw < Q4E_FR_BLOCK / 32u; vw++) {
            float sum = 0.0f;
#pragma unroll
            for (uint32_t k = 0; k < PER / (Q4E_FR_BLOCK / 32u); k++) {
                const uint32_t j = vw + (Q4E_FR_BLOCK / 32u) * k;
                if (lane + 32u * j < n_expert) sum += __expf(v[j] - s_max);
            }
            sum = warp_sum_f32(sum);
            const float s_w = __shfl_sync(0xffffffffu, sum, 0);
            if (lane == vw) warp_total = s_w;
        }
        warp_total = warp_sum_f32(warp_total);
        const float inv_sum = 1.0f / __shfl_sync(0xffffffffu, warp_total, 0);

        float my_w = 0.0f;
        int32_t my_i = -1;
#pragma unroll 1
        for (uint32_t s = 0; s < 16u; s++) {
            if (s >= n_used) break;
            float bv[PER];
            uint32_t bj[PER];
#pragma unroll
            for (uint32_t j = 0; j < PER; j++) {
                bv[j] = (lane + 32u * j < n_expert) ? v[j] : -INFINITY;
                bj[j] = j;
            }
#define Q4E_FR_TREE_LEVEL(W)                                                       \
    _Pragma("unroll") for (uint32_t j = 0; j < (W); j++) {                         \
        const bool take = bv[j + (W)] > bv[j]                                      \
                       || (bv[j + (W)] == bv[j] && bj[j + (W)] < bj[j]);           \
        bv[j] = take ? bv[j + (W)] : bv[j];                                        \
        bj[j] = take ? bj[j + (W)] : bj[j];                                        \
    }
            static_assert(PER == 16u, "the tree below is four levels");
            Q4E_FR_TREE_LEVEL(8u)
            Q4E_FR_TREE_LEVEL(4u)
            Q4E_FR_TREE_LEVEL(2u)
            Q4E_FR_TREE_LEVEL(1u)
#undef Q4E_FR_TREE_LEVEL
            const uint32_t key = q4e_fr_key(bv[0]);
            const uint32_t top_key = __reduce_max_sync(0xffffffffu, key);
            const uint32_t mine = lane + 32u * bj[0];
            const uint32_t top_e = __reduce_min_sync(0xffffffffu, key == top_key ? mine : 0xffffffffu);
            const float top = __shfl_sync(0xffffffffu, bv[0], top_e & 31u);
            if (lane == s) {
                my_w = top;
                my_i = (int32_t)top_e;
            }
#pragma unroll
            for (uint32_t j = 0; j < PER; j++) {
                v[j] = top_e == lane + 32u * j ? -INFINITY : v[j];
            }
        }
        const float e_w = lane < n_used ? __expf(my_w - s_max) * inv_sum : 0.0f;
        float wsum = 0.0f;
#pragma unroll
        for (uint32_t s = 0; s < 16u; s++) {
            if (s >= n_used) break;
            wsum += __shfl_sync(0xffffffffu, e_w, s);
        }
        wsum = fmaxf(wsum, 1e-20f);
        if (lane < n_used) {
            ids[(uint64_t)warp * n_used + lane] = my_i;
            weights[(uint64_t)warp * n_used + lane] = e_w / wsum;
            s_ids[warp * n_used + lane] = my_i;
        }
    }
    __syncthreads();

    const uint32_t n_rows = n_tok * n_used;
    for (uint32_t i = threadIdx.x; i < n_rows; i += blockDim.x) {
        const int32_t e = s_ids[i];
        if (e >= 0 && (uint32_t)e < n_expert) atomicAdd(&s_counts[e], 1);
    }
    __syncthreads();
    const uint32_t e0 = 2u * threadIdx.x;
    const int32_t c0 = e0 < n_expert ? s_counts[e0] : 0;
    const int32_t c1 = e0 + 1u < n_expert ? s_counts[e0 + 1u] : 0;
    const uint32_t mine = ((uint32_t)(c0 + c1) << 16) | (uint32_t)((c0 > 0) + (c1 > 0));
    uint32_t incl = mine;
#pragma unroll
    for (uint32_t d = 1; d < 32u; d <<= 1) {
        const uint32_t o = __shfl_up_sync(0xffffffffu, incl, d);
        if (lane >= d) incl += o;
    }
    if (lane == 31u) s_warp[warp] = incl;
    __syncthreads();
    uint32_t base = 0;
    for (uint32_t w = 0; w < warp; w++) base += s_warp[w];
    const uint32_t excl = base + incl - mine;
    const int32_t at_rows = (int32_t)(excl >> 16);
    int32_t at_live = (int32_t)(excl & 0xFFFFu);
    if (e0 < n_expert) {
        bounds[e0] = at_rows;
        s_bounds[e0] = at_rows;
        if (c0 > 0) active[at_live++] = (int32_t)e0;
    }
    if (e0 + 1u < n_expert) {
        bounds[e0 + 1u] = at_rows + c0;
        s_bounds[e0 + 1u] = at_rows + c0;
        if (c1 > 0) active[at_live] = (int32_t)(e0 + 1u);
    }
    if (threadIdx.x == blockDim.x - 1u) {
        const uint32_t total = base + incl;
        bounds[n_expert] = (int32_t)(total >> 16);
        active[n_expert] = (int32_t)(total & 0xFFFFu);
    }
    __syncthreads();
    for (uint32_t i = warp; i < n_rows; i += blockDim.x / 32u) {
        const int32_t e = s_ids[i];
        int32_t rank = 0;
        for (uint32_t j0 = 0; j0 < i; j0 += 32u) {
            const uint32_t j = j0 + lane;
            rank += __popc(__ballot_sync(0xffffffffu, j < i && s_ids[j] == e));
        }
        if (lane == 0u && e >= 0 && (uint32_t)e < n_expert) {
            const int32_t at = s_bounds[e] + rank;
            sorted[at] = (int32_t)i;
            pos[i] = at;
        }
    }
}

/* `silu(gate) * up`, exactly as `q4e_swiglu_kernel` (moe.cu, built with the
 * same flags) computes it from the two stored rows. */
__device__ __forceinline__ float q4e_fr_swiglu(float g, float u) {
    return (g / (1.0f + __expf(-g))) * u;
}

/* Chunks of a row a warp holds in flight at once: five cover the 2560-wide
 * block input, so a row is one round trip to memory rather than five. */
#define Q4E_FR_MX_DEPTH 5u
/* Scale bytes a warp stages per row: the widest contraction the route launch
 * takes is 32 x 160 = 5120. */
#define Q4E_FR_MX_SCALES 160u
/* Shared memory a warp stages a row pair in. */
#define Q4E_FR_MX_STAGE (2u * Q4E_MX_CHUNK + 2u * Q4E_FR_MX_SCALES)

/* Row `row` of the shared expert's MXFP8 gate and up against `NT` activation
 * rows, activated: each row's sum is `q4e_mx_rows_pf`'s (mxfp8.cu) - lane `l`
 * owns element pairs `l, l + 32, ...` of the row, one exact fused
 * multiply-add per weight per activation row in contraction order, then the
 * exact decreasing-offset fold - which is also `q4e_mx_rows_xs`'s at two to
 * eight rows, so `mid` holds the bits `q4e_swiglu_kernel` writes from what
 * the group launches store.
 *
 * What differs is only when the bytes are asked for: every code of up to
 * `Q4E_FR_MX_DEPTH` chunks of both rows and the rows' scale bytes are loaded
 * before the first multiply (the group kernel holds one chunk ahead, which
 * left the shared expert a chain of five round trips a row). `stage` is this
 * warp's `2 * 512` code bytes and `2 * Q4E_FR_MX_SCALES` scale bytes of
 * shared memory. */
template <int NT>
__device__ __forceinline__ void q4e_fr_mx_pair(
        float *__restrict__ mid, const uint8_t *__restrict__ g_codes,
        const uint8_t *__restrict__ g_scales, const uint8_t *__restrict__ u_codes,
        const uint8_t *__restrict__ u_scales, const float *__restrict__ x, uint32_t in_dim,
        uint32_t inter, uint32_t row, uint32_t lane, uint8_t *__restrict__ stage) {
    const uint32_t ng = in_dim / Q4E_MX_GROUP;
    const uint8_t *wr[2] = {g_codes + (uint64_t) row * in_dim, u_codes + (uint64_t) row * in_dim};
    uint8_t *sstage = stage + 2u * Q4E_MX_CHUNK;
    float acc[2][NT];
#pragma unroll
    for (int r = 0; r < 2; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            acc[r][t] = 0.0f;
        }
    }
    /* Both rows' scale bytes, sixteen a lane-load: a row's scales start on a
     * sixteen-byte boundary because the contraction is whole 512-wide
     * chunks. */
    const uint32_t sq = ng / 16u;
    uint4 sv = make_uint4(0u, 0u, 0u, 0u);
    if (lane < 2u * sq) {
        const uint8_t *srow = (lane < sq ? g_scales : u_scales) + (uint64_t) row * ng;
        sv = __ldg(reinterpret_cast<const uint4 *>(srow + 16u * (lane % sq)));
    }
    const uint32_t hi = lane >> 4;
    bool scales_staged = false;
    for (uint32_t base0 = 0; base0 < in_dim; base0 += Q4E_FR_MX_DEPTH * Q4E_MX_CHUNK) {
        uint4 buf[Q4E_FR_MX_DEPTH][2];
#pragma unroll
        for (uint32_t d = 0; d < Q4E_FR_MX_DEPTH; d++) {
            const uint32_t off = base0 + d * Q4E_MX_CHUNK + 16u * lane;
#pragma unroll
            for (int r = 0; r < 2; r++) {
                buf[d][r] = off < in_dim ? __ldg(reinterpret_cast<const uint4 *>(wr[r] + off))
                                         : make_uint4(0u, 0u, 0u, 0u);
            }
        }
        if (!scales_staged) {
            if (lane < 2u * sq) {
                const uint32_t r = lane / sq;
                *reinterpret_cast<uint4 *>(sstage + r * Q4E_FR_MX_SCALES + 16u * (lane % sq)) = sv;
            }
            scales_staged = true;
        }
#pragma unroll
        for (uint32_t d = 0; d < Q4E_FR_MX_DEPTH; d++) {
            const uint32_t base = base0 + d * Q4E_MX_CHUNK;
            if (base >= in_dim) {
                break;
            }
            __syncwarp();
#pragma unroll
            for (int r = 0; r < 2; r++) {
                *reinterpret_cast<uint4 *>(stage + r * Q4E_MX_CHUNK + 16u * lane) = buf[d][r];
            }
            __syncwarp();
            /* This chunk's sixteen scale bytes a row, groups `base / 32 + 4 q`
             * to `+ 3` in word `q`, as `q4e_mx_rows_pf` assembles them. */
            uint32_t sw[2][4];
#pragma unroll
            for (int r = 0; r < 2; r++) {
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    sw[r][q] = *reinterpret_cast<const uint32_t *>(
                            sstage + r * Q4E_FR_MX_SCALES + base / Q4E_MX_GROUP + 4u * q);
                }
            }
            /* `q4e_mx_rows`' fast weights (mxfp8.cuh) unless this chunk holds
             * a NaN code or a scale of 135 or more: bit for bit the same
             * weights either way, a dozen integer operations a weight fewer
             * on the fast side. */
            uint32_t odd = 0u;
#pragma unroll
            for (int r = 0; r < 2; r++) {
                odd |= q4e_mx_nan4(buf[d][r].x) | q4e_mx_nan4(buf[d][r].y)
                     | q4e_mx_nan4(buf[d][r].z) | q4e_mx_nan4(buf[d][r].w)
                     | q4e_mx_big4(sw[r][0]) | q4e_mx_big4(sw[r][1])
                     | q4e_mx_big4(sw[r][2]) | q4e_mx_big4(sw[r][3]);
            }
            if (!__any_sync(0xffffffffu, odd != 0u)) {
#pragma unroll
                for (uint32_t j = 0; j < Q4E_MX_CHUNK / 64u; j++) {
                    const uint32_t e = base + 2u * lane + 64u * j;
                    float2 xv[NT];
#pragma unroll
                    for (int t = 0; t < NT; t++) {
                        xv[t] = __ldg(reinterpret_cast<const float2 *>(x + (uint64_t) t * in_dim + e));
                    }
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        const uint32_t p = *reinterpret_cast<const uint16_t *>(
                                stage + r * Q4E_MX_CHUNK + 2u * lane + 64u * j);
                        const float2 w = q4e_mx_pair_k(
                                p, q4e_mx_k(sw[r][j >> 1] + 0x78787878u, 2u * (j & 1u) + hi));
#pragma unroll
                        for (int t = 0; t < NT; t++) {
                            acc[r][t] = q4e_mx_fma(w.x, xv[t].x, acc[r][t]);
                            acc[r][t] = q4e_mx_fma(w.y, xv[t].y, acc[r][t]);
                        }
                    }
                }
                continue;
            }
#pragma unroll
            for (uint32_t j = 0; j < Q4E_MX_CHUNK / 64u; j++) {
                const uint32_t e = base + 2u * lane + 64u * j;
                float2 xv[NT];
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    xv[t] = __ldg(reinterpret_cast<const float2 *>(x + (uint64_t) t * in_dim + e));
                }
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint32_t p = *reinterpret_cast<const uint16_t *>(
                            stage + r * Q4E_MX_CHUNK + 2u * lane + 64u * j);
                    const float s =
                            q4e_mx_scale((sw[r][j >> 1] >> (16u * (j & 1u) + 8u * hi)) & 0xFFu);
                    const float2 v = q4e_mx_pair(p);
                    const float w0 = q4e_mx_mul(v.x, s);
                    const float w1 = q4e_mx_mul(v.y, s);
#pragma unroll
                    for (int t = 0; t < NT; t++) {
                        acc[r][t] = q4e_mx_fma(w0, xv[t].x, acc[r][t]);
                        acc[r][t] = q4e_mx_fma(w1, xv[t].y, acc[r][t]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float g = q4e_mx_warp_sum(acc[0][t]);
        const float u = q4e_mx_warp_sum(acc[1][t]);
        if (lane == 0u) {
            mid[(uint64_t) t * inter + row] = q4e_fr_swiglu(g, u);
        }
    }
}

/* The words of `sync` the launch owns: the item counter, the selection's
 * ticket, the shared gate-and-up items done, and the blocks finished. */
#define Q4E_FR_SYNC_ITEM 0u
#define Q4E_FR_SYNC_TICKET 1u
#define Q4E_FR_SYNC_SHARED 2u
#define Q4E_FR_SYNC_DONE 3u

__device__ __forceinline__ uint32_t q4e_fr_ld_acquire(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

/* Everything one launch writes and reads. */
struct q4e_fr_args {
    float *logits;
    float *gate_logit;
    float *mid;
    float *shared_out;
    int32_t *ids;
    float *weights;
    int32_t *bounds;
    int32_t *active;
    int32_t *sorted;
    int32_t *pos;
    uint32_t *sync;
    const __nv_bfloat16 *router;
    const __nv_bfloat16 *gate;
    const uint8_t *g_codes;
    const uint8_t *g_scales;
    const uint8_t *u_codes;
    const uint8_t *u_scales;
    const uint8_t *d_codes;
    const uint8_t *d_scales;
    const float *x;
    uint32_t in_dim;
    uint32_t n_expert;
    uint32_t n_used;
    uint32_t inter;
    uint32_t splits;
    uint32_t gate_splits;
};

/* A block's shared memory, declared once in each entry: a template that
 * declared it would get a copy per row count. The staged weights and the
 * selection's arrays are never live in one block at once. */
struct q4e_fr_smem {
    float part[Q4E_FR_WARPS * Q4E_FR_ROWS];
    union {
        __align__(16) uint8_t stage[Q4E_FR_WARPS * Q4E_FR_MX_STAGE];
        q4e_fr_pick pick;
    } u;
    uint32_t item;
    uint32_t last;
};

/* The whole launch at `NT` activation rows. Each block takes the next item:
 *
 * - `[0, SB)`: eight rows of the shared expert's gate and up, activated
 *   (`q4e_fr_mx_pair`), counted done in `sync[2]`;
 * - `[SB, SB + RB)`: two router rows at four slices (`q4e_fr_rows`), then
 *   the gate logit's item; each takes a ticket, and the last ticket runs
 *   the selection;
 * - the rest: sixteen rows of the shared expert's down projection
 *   (`q4e_mx_rows`, two rows a warp - `q4e_mxfp8_gemv_kernel`'s body), once
 *   every gate-and-up item is done.
 *
 * Items rather than block indices so a down item only ever waits for items
 * blocks already running have taken: nothing can wait on a block the device
 * has not scheduled. The gate-and-up items come first because each is the
 * longest (a 5 KB row pair, then its arithmetic); the router's stream beside
 * them, and the down items stream while the selection runs. */
template <int NT>
__device__ __forceinline__ void q4e_fr_main(const q4e_fr_args &a, uint32_t hidden_out,
                                            q4e_fr_smem &sm) {
    float *part = sm.part;
    uint8_t *stage = sm.u.stage;
    uint32_t &s_item = sm.item;
    uint32_t &s_last = sm.last;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t shared_items = (a.inter + Q4E_FR_WARPS - 1u) / Q4E_FR_WARPS;
    const uint32_t router_items = a.n_expert * a.splits / Q4E_FR_WARPS;
    /* Item ranges: the router's (and the gate logit's), the shared gate and
     * up's, then the shared down's. */
    const uint32_t gu_lo = 0u, rt_lo = shared_items;
    const uint32_t dn_lo = shared_items + router_items + 1u;
    if (threadIdx.x == 0u) {
        s_item = atomicAdd(a.sync + Q4E_FR_SYNC_ITEM, 1u);
    }
    __syncthreads();
    const uint32_t item = s_item;
    if (item >= gu_lo && item < gu_lo + shared_items) {
        const uint32_t row = (item - gu_lo) * Q4E_FR_WARPS + warp;
        if (row < a.inter) {
            q4e_fr_mx_pair<NT>(a.mid, a.g_codes, a.g_scales, a.u_codes, a.u_scales, a.x,
                               a.in_dim, a.inter, row, lane, stage + warp * Q4E_FR_MX_STAGE);
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            __threadfence();
            atomicAdd(a.sync + Q4E_FR_SYNC_SHARED, 1u);
        }
    } else if (item >= rt_lo && item <= rt_lo + router_items) {
        const uint32_t b = item - rt_lo;
        if (b < router_items) {
            q4e_fr_rows<NT>(a.logits, a.router, a.x, a.in_dim, a.n_expert, a.splits, b, part);
        } else {
            q4e_fr_rows<NT>(a.gate_logit, a.gate, a.x, a.in_dim, 1u, a.gate_splits, 0u, part);
        }
        /* Every store of this block is made before its ticket is taken, and
         * fenced so the block that takes the last one sees them. */
        __syncthreads();
        if (threadIdx.x == 0u) {
            __threadfence();
            const uint32_t t = atomicAdd(a.sync + Q4E_FR_SYNC_TICKET, 1u);
            s_last = t == router_items;
        }
        __syncthreads();
        if (s_last) {
            __threadfence();
            q4e_fr_select(a.ids, a.weights, a.logits, a.bounds, a.active, a.sorted, a.pos,
                          (uint32_t) NT, a.n_expert, a.n_used, sm.u.pick);
        }
    } else {
        if (threadIdx.x == 0u) {
            while (q4e_fr_ld_acquire(a.sync + Q4E_FR_SYNC_SHARED) < shared_items) {
                __nanosleep(128);
            }
            __threadfence();
        }
        __syncthreads();
        const uint32_t row0 = ((item - dn_lo) * Q4E_FR_WARPS + warp) * 2u;
        if (row0 < hidden_out) {
            q4e_mx_rows<NT, 2, 1>(a.shared_out, a.d_codes, a.d_scales, a.mid, a.inter,
                                  hidden_out, row0, lane, stage + warp * Q4E_FR_MX_STAGE);
        }
    }
    /* The last block to finish puts the words back for the next launch:
     * every other block is done with them by then. */
    __syncthreads();
    if (threadIdx.x == 0u) {
        __threadfence();
        if (atomicAdd(a.sync + Q4E_FR_SYNC_DONE, 1u) == gridDim.x - 1u) {
            a.sync[Q4E_FR_SYNC_ITEM] = 0u;
            a.sync[Q4E_FR_SYNC_TICKET] = 0u;
            a.sync[Q4E_FR_SYNC_SHARED] = 0u;
            a.sync[Q4E_FR_SYNC_DONE] = 0u;
            __threadfence();
        }
    }
}

#define Q4E_FR_PACK                                                                        \
    const q4e_fr_args a = {logits, gate_logit, mid, shared_out, ids, weights, bounds,      \
                           active, sorted, pos, sync, router, gate, g_codes, g_scales,     \
                           u_codes, u_scales, d_codes, d_scales, x, in_dim, n_expert,      \
                           n_used, inter, splits, gate_splits}

/* Buffers: logits `[rows, n_expert]`, gate_logit `[rows]`, mid
 * `[rows, inter]` (the shared expert's activated rows), shared_out
 * `[rows, in_dim]`, ids and weights `[rows, n_used]`, bounds and active
 * `[n_expert + 1]`, sorted and pos `[rows * n_used]`, sync `[4]` `u32` (zero
 * before the first launch; the last block puts it back), router
 * `[n_expert, in_dim]` bf16, gate `[1, in_dim]` bf16, the shared expert's
 * MXFP8 gate and up codes `[inter, in_dim]` and scales `[inter, in_dim / 32]`
 * and down codes `[in_dim, inter]` and scales `[in_dim, inter / 32]`, x
 * `[rows, in_dim]`.
 *
 * Grid `ceil(inter / 8) + n_expert * splits / 8 + 1 + ceil(in_dim / 16)`,
 * 256 threads. `splits` and `gate_splits` divide eight; `in_dim` a multiple
 * of 512 and at most 5120; `inter` a multiple of 32; `n_expert` a multiple
 * of 32 and at most 512; `n_used` at most 16; every code array sixteen-byte
 * aligned.
 *
 * One activation row (`rows` must be 1): the decode step's entry, compiled
 * apart from the verify's so its registers are its own - one entry over every
 * row count is held to the widest one's, which halved the blocks an SM
 * keeps (capping it at 64 registers to fit a fourth block spilled and
 * measured slower). */
extern "C" __global__ void __launch_bounds__(Q4E_FR_WARPS * 32u) q4e_ffn_route_kernel(
        float *logits, float *gate_logit, float *mid, float *shared_out, int32_t *ids,
        float *weights, int32_t *bounds, int32_t *active, int32_t *sorted, int32_t *pos,
        uint32_t *sync, const __nv_bfloat16 *router, const __nv_bfloat16 *gate,
        const uint8_t *g_codes, const uint8_t *g_scales, const uint8_t *u_codes,
        const uint8_t *u_scales, const uint8_t *d_codes, const uint8_t *d_scales,
        const float *x, uint32_t in_dim, uint32_t n_expert, uint32_t n_used, uint32_t inter,
        uint32_t splits, uint32_t gate_splits, uint32_t rows) {
    Q4E_FR_PACK;
    __shared__ q4e_fr_smem sm;
    if (rows == 1u) {
        q4e_fr_main<1>(a, in_dim, sm);
    }
}

/* Two to eight activation rows: the verify's entry; any other count writes
 * nothing. Arguments and grid as `q4e_ffn_route_kernel`'s. */
extern "C" __global__ void __launch_bounds__(Q4E_FR_WARPS * 32u) q4e_ffn_route_rows_kernel(
        float *logits, float *gate_logit, float *mid, float *shared_out, int32_t *ids,
        float *weights, int32_t *bounds, int32_t *active, int32_t *sorted, int32_t *pos,
        uint32_t *sync, const __nv_bfloat16 *router, const __nv_bfloat16 *gate,
        const uint8_t *g_codes, const uint8_t *g_scales, const uint8_t *u_codes,
        const uint8_t *u_scales, const uint8_t *d_codes, const uint8_t *d_scales,
        const float *x, uint32_t in_dim, uint32_t n_expert, uint32_t n_used, uint32_t inter,
        uint32_t splits, uint32_t gate_splits, uint32_t rows) {
    Q4E_FR_PACK;
    __shared__ q4e_fr_smem sm;
    switch (rows) {
    case 2u: q4e_fr_main<2>(a, in_dim, sm); break;
    case 3u: q4e_fr_main<3>(a, in_dim, sm); break;
    case 4u: q4e_fr_main<4>(a, in_dim, sm); break;
    case 5u: q4e_fr_main<5>(a, in_dim, sm); break;
    case 6u: q4e_fr_main<6>(a, in_dim, sm); break;
    case 7u: q4e_fr_main<7>(a, in_dim, sm); break;
    case 8u: q4e_fr_main<8>(a, in_dim, sm); break;
    default: break;
    }
}

/* ---------------------------------------------------------------------------
 * The routed block's close: the weighted sum of the routed columns and the
 * gated shared expert in one pass.
 * ------------------------------------------------------------------------ */

/* `out[t] = sum_s weights[t][s] down[t n_used + s] + sigmoid(logit[t])
 * shared[t]`: `q4e_moe_combine_kernel` then `q4e_shared_add_kernel` (moe.cu,
 * built with the same flags), expression for expression - the columns added
 * in slot order from zero, then the shared term added onto that sum - in one
 * launch instead of two, with the sum never stored in between.
 *
 * Buffers: out `[rows, n_embd]`, down `[rows * n_used, n_embd]`, weights
 * `[rows, n_used]`, shared `[rows, n_embd]`, logit `[rows]`. Grid
 * `(ceil(n_embd / 256), rows)`, 256 threads. */
extern "C" __global__ void q4e_ffn_combine_kernel(
        float *out, const float *down, const float *weights, const float *shared,
        const float *logit, uint32_t n_embd, uint32_t n_used) {
    const uint32_t t = blockIdx.y;
    const float g = 1.0f / (1.0f + __expf(-logit[t]));
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n_embd;
         e += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_used; s++) {
            const uint64_t col = (uint64_t)t * n_used + s;
            acc += down[col * n_embd + e] * weights[col];
        }
        acc += shared[(uint64_t)t * n_embd + e] * g;
        out[(uint64_t)t * n_embd + e] = acc;
    }
}

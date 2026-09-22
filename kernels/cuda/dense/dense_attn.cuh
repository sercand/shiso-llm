/* Windowed, bounded and scaled GQA over the paged f16 cache.
 *
 * The span is the whole of what makes this set parametric. A causal kernel
 * derives what a query may see from the query's own position: everything at or
 * before it. Here the low end is a sliding window (`pos + 1 - window`, or 0
 * when the layer has none) and the high end is a per-token bound the caller
 * supplies, which for a text token is its own position and for a bidirectional
 * prefix reaches PAST it. Neither end is derivable from the other, and a kernel
 * that walked `0..=pos` would be right on exactly the layers that have no
 * window.
 *
 * The score scale is an argument for the same reason: a layer of this family
 * passes 1.0 where a qwen35 layer passes `1 / sqrt(head_dim)`, and it is
 * applied to the REDUCED score rather than folded into the query, so the number
 * in the argument block is the number the softmax sees.
 *
 * The shared-memory figures below are compile-time, from `DENSE_ATTN_HD` (512)
 * and never from `head_dim`: 16448 = 8*512*4 + 2*8*4 for decode, 16384 =
 * 512*16*2 for prefill. Sizing them off `head_dim` would pass every case at
 * Gemma's width and overrun at another.
 */
#ifndef SHISO_DENSE_ATTN_CUH
#define SHISO_DENSE_ATTN_CUH

#include "dense_common.cuh"

/* The first key position a query at `pos` may read under `window`.
 *
 * `0` is no window at all. The comparison is written as `pos + 1 > window`
 * rather than `pos >= window` so the subtraction never underflows an unsigned
 * position. */
__device__ __forceinline__ unsigned dense_attn_lo(unsigned pos, unsigned window) {
    return (window != 0u && pos + 1u > window) ? (pos + 1u - window) : 0u;
}

/* The decode softmax: one threadgroup per (head, token), positions spread over
 * the block's warps.
 *
 * One warp owns whole positions and reduces each score with a shuffle
 * butterfly, so the position loop never needs a barrier. The softmax is online
 * (running max + running denominator) and the per-warp partials are merged at
 * the end by re-referencing every partial against the global max, which is what
 * makes the merge order-independent. */
extern "C" __global__ void kernel_dense_attn_decode(
        const struct dense_args_attn args,
        float *__restrict__ out,
        const float *__restrict__ q,
        const __half *__restrict__ k_cache,
        const __half *__restrict__ v_cache,
        const int32_t *__restrict__ pos,
        const int32_t *__restrict__ bound,
        const int32_t *__restrict__ pages) {
    const unsigned head_dim = args.head_dim;
    const unsigned t = blockIdx.y;
    const unsigned h = blockIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5u;
    /* Query heads share key/value heads in whole groups: an integer divide, not
     * a modulo. */
    const unsigned hkv = h / (args.n_head / args.n_head_kv);
    const unsigned slots = (head_dim + 31u) / 32u;

    /* s_v[DENSE_ATTN_WARPS][DENSE_ATTN_HD], then s_m and s_l below it. */
    float *s_v = (float *) dense_smem;
    float *s_m = s_v + DENSE_ATTN_WARPS * DENSE_ATTN_HD;
    float *s_l = s_m + DENSE_ATTN_WARPS;

    const uint64_t qbase = ((uint64_t) t * args.n_head + h) * head_dim;
    /* Both arrays are indexed by a compile-time bound with a runtime guard
     * inside, rather than by a runtime trip count: a dynamically indexed local
     * array lives in local memory, which on this device is device memory, and
     * every accumulate would then cost a round trip. */
    float qv[DENSE_ATTN_LANE_SLOTS];
    float acc[DENSE_ATTN_LANE_SLOTS];
#pragma unroll
    for (unsigned j = 0u; j < DENSE_ATTN_LANE_SLOTS; ++j) {
        const unsigned i = lane + 32u * j;
        qv[j] = (j < slots && i < head_dim) ? q[qbase + i] : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    /* The span, in logical positions. The rows under it are translated one at a
     * time: a walk over positions crosses a page boundary wherever the session
     * happened to start each page. */
    const unsigned p_lo = dense_attn_lo((uint32_t) pos[t], args.window);
    const unsigned p_hi = (uint32_t) bound[t];
    /* This ROW's page table, not the launch's. A batched decode step carries
     * one row per session and two sessions share no page, so a table selected
     * once for the whole launch would answer every row past the first out of
     * row zero's conversation. */
    const int32_t *row_pages = dense_row_pages(pages, t, args.page_stride);
    for (unsigned p = p_lo + warp; p <= p_hi; p += DENSE_ATTN_WARPS) {
        const uint64_t kb =
            ((uint64_t) dense_kv_row(row_pages, p, args.page_tokens) * args.n_head_kv + hkv)
            * head_dim;
        float dot = 0.0f;
#pragma unroll
        for (unsigned j = 0u; j < DENSE_ATTN_LANE_SLOTS; ++j) {
            const unsigned i = lane + 32u * j;
            if (j < slots && i < head_dim) {
                dot = dense_fma(qv[j], __half2float(k_cache[kb + i]), dot);
            }
        }
        /* Butterfly, so every lane ends up holding the whole score. */
        for (unsigned off = 16u; off > 0u; off >>= 1u) {
            dot += __shfl_xor_sync(DENSE_FULL_WARP, dot, off);
        }
        dot *= args.scale;

        const float m_new = fmaxf(m, dot);
        const float corr = expf(m - m_new);
        const float wgt = expf(dot - m_new);
#pragma unroll
        for (unsigned j = 0u; j < DENSE_ATTN_LANE_SLOTS; ++j) {
            const unsigned i = lane + 32u * j;
            const float v =
                (j < slots && i < head_dim) ? __half2float(v_cache[kb + i]) : 0.0f;
            acc[j] = acc[j] * corr + wgt * v;
        }
        l = l * corr + wgt;
        m = m_new;
    }

#pragma unroll
    for (unsigned j = 0u; j < DENSE_ATTN_LANE_SLOTS; ++j) {
        const unsigned i = lane + 32u * j;
        if (j < slots && i < head_dim) {
            s_v[warp * DENSE_ATTN_HD + i] = acc[j];
        }
    }
    if (lane == 0u) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    __syncthreads();

    float gm = -INFINITY;
    for (unsigned w = 0u; w < DENSE_ATTN_WARPS; ++w) {
        gm = fmaxf(gm, s_m[w]);
    }
    /* An empty span -- a bound below the window's low end -- would otherwise
     * reference every partial against -INFINITY and write a row of NaN into the
     * residual stream. Zero is what an attention over no keys contributes. */
    if (gm == -INFINITY) {
        for (unsigned i = threadIdx.x; i < head_dim; i += 256u) {
            out[qbase + i] = 0.0f;
        }
        return;
    }
    float den = 0.0f;
    for (unsigned w = 0u; w < DENSE_ATTN_WARPS; ++w) {
        den += s_l[w] * expf(s_m[w] - gm);
    }
    const float inv = 1.0f / den;
    for (unsigned i = threadIdx.x; i < head_dim; i += 256u) {
        float num = 0.0f;
        for (unsigned w = 0u; w < DENSE_ATTN_WARPS; ++w) {
            num += s_v[w * DENSE_ATTN_HD + i] * expf(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

/* The prefill softmax: DENSE_ATTN_QT queries per threadgroup, DENSE_ATTN_KT
 * keys per tile, one key per lane.
 *
 * Two shapes carry this kernel. The first is that lane `l` scores key `p0 + l`
 * entirely on its own -- there is no per-key reduction -- so the tile's softmax
 * needs a max butterfly over the whole warp instead. The second is that the key
 * weights of a tile are warp-private: the broadcast is a shuffle from the lane
 * that computed the weight, which needs no barrier at all. That also keeps
 * every barrier in this kernel outside the query loop, where a short final tile
 * would otherwise leave some warps unable to reach one.
 *
 * The tile's key range is the union of its queries' spans -- the lowest window
 * start against the highest bound -- and the per-query mask inside the tile is
 * that query's own span. A tile whose queries disagree therefore stages keys
 * some of them cannot see, which is a wasted load rather than a wrong answer.
 *
 * Shared memory is the transposed keys alone: 16 KiB at head_dim 512. The
 * values are read straight from the cache because that copy would be a
 * bandwidth cache rather than arithmetic, and the device holds the same f16
 * values in the same order. */
/* One table for the whole launch, and `args.page_stride` is not read here.
 * This entry is reached only by a chunk of DENSE_ATTN_QT tokens or more, and a
 * chunk is one session's own consecutive positions by construction - a decode
 * batch is at most MAX_DECODE_BATCH rows and takes the decode entry above. The
 * host refuses a per-row table on this path rather than letting a tile
 * silently read row t0's. */
extern "C" __global__ void kernel_dense_attn_prefill(
        const struct dense_args_attn args,
        float *__restrict__ out,
        const float *__restrict__ q,
        const __half *__restrict__ k_cache,
        const __half *__restrict__ v_cache,
        const int32_t *__restrict__ pos,
        const int32_t *__restrict__ bound,
        const int32_t *__restrict__ pages) {
    const unsigned head_dim = args.head_dim;
    const unsigned h = blockIdx.x;
    const unsigned t0 = blockIdx.y * DENSE_ATTN_QT;
    if (t0 >= args.n_tok) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    const unsigned hkv = h / (args.n_head / args.n_head_kv);
    /* Unguarded below, and therefore a precondition: this entry requires
     * `head_dim % 32 == 0`. The engine validates it on the host. */
    const unsigned slots = head_dim / 32u;

    __half *s_k = (__half *) dense_smem;
    const unsigned n_q =
        (args.n_tok - t0 < DENSE_ATTN_QT) ? (args.n_tok - t0) : DENSE_ATTN_QT;

    /* The union of the tile's spans: the lowest window start any query of this
     * tile has, against the highest bound any of them carries. */
    unsigned tile_lo = 0xFFFFFFFFu;
    unsigned tile_hi = 0u;
    for (unsigned i = 0u; i < n_q; ++i) {
        const unsigned lo = dense_attn_lo((uint32_t) pos[t0 + i], args.window);
        const unsigned hi = (uint32_t) bound[t0 + i];
        tile_lo = min(tile_lo, lo);
        tile_hi = max(tile_hi, hi);
    }

    /* Two queries per warp, each with its own running max, running denominator
     * and head_dim/32 accumulator slots. */
    float acc[2][DENSE_ATTN_LANE_SLOTS];
    float m[2], l[2];
#pragma unroll
    for (unsigned s = 0u; s < 2u; ++s) {
        m[s] = -INFINITY;
        l[s] = 0.0f;
#pragma unroll
        for (unsigned j = 0u; j < DENSE_ATTN_LANE_SLOTS; ++j) {
            acc[s][j] = 0.0f;
        }
    }

    for (unsigned p0 = tile_lo; p0 <= tile_hi; p0 += DENSE_ATTN_KT) {
        const unsigned n_k =
            (tile_hi + 1u - p0 < DENSE_ATTN_KT) ? (tile_hi + 1u - p0) : DENSE_ATTN_KT;
        for (unsigned idx = tid; idx < n_k * head_dim; idx += blockDim.x) {
            const unsigned k = idx / head_dim;
            const unsigned d = idx - k * head_dim;
            const uint64_t row = dense_kv_row(pages, p0 + k, args.page_tokens);
            const uint64_t src =
                ((uint64_t) row * args.n_head_kv + hkv) * head_dim + d;
            s_k[(size_t) d * DENSE_ATTN_KT + k] = k_cache[src];
        }
        __syncthreads();

#pragma unroll
        for (unsigned s = 0u; s < 2u; ++s) {
            const unsigned qi = warp * 2u + s;
            if (qi >= n_q) {
                break;
            }
            const unsigned q_lo = dense_attn_lo((uint32_t) pos[t0 + qi], args.window);
            const unsigned q_hi = (uint32_t) bound[t0 + qi];
            const uint64_t qbase = ((uint64_t) (t0 + qi) * args.n_head + h) * head_dim;

            float dot = 0.0f;
            if (lane < n_k && p0 + lane >= q_lo && p0 + lane <= q_hi) {
                for (unsigned d = 0u; d < head_dim; ++d) {
                    dot = dense_fma(q[qbase + d],
                                    __half2float(s_k[(size_t) d * DENSE_ATTN_KT + lane]),
                                    dot);
                }
                dot *= args.scale;
            } else {
                dot = -INFINITY;
            }

            float tile_max = dot;
            for (unsigned off = 16u; off > 0u; off >>= 1u) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(DENSE_FULL_WARP, tile_max, off));
            }
            if (tile_max == -INFINITY) {
                continue; /* every key of this tile is masked for this query */
            }

            const float m_new = fmaxf(m[s], tile_max);
            const float corr = expf(m[s] - m_new);
            const float wgt = expf(dot - m_new); /* 0 wherever the key was masked */
            float wsum = wgt;
            for (unsigned off = 16u; off > 0u; off >>= 1u) {
                wsum += __shfl_xor_sync(DENSE_FULL_WARP, wsum, off);
            }
            l[s] = l[s] * corr + wsum;
            m[s] = m_new;

#pragma unroll
            for (unsigned j = 0u; j < DENSE_ATTN_LANE_SLOTS; ++j) {
                if (j >= slots) {
                    break;
                }
                const unsigned d = lane + 32u * j;
                float a = acc[s][j] * corr;
                for (unsigned k = 0u; k < n_k; ++k) {
                    const uint64_t vrow = dense_kv_row(pages, p0 + k, args.page_tokens);
                    const uint64_t vb =
                        ((uint64_t) vrow * args.n_head_kv + hkv) * head_dim + d;
                    a = dense_fma(__shfl_sync(DENSE_FULL_WARP, wgt, (int) k),
                                  __half2float(v_cache[vb]), a);
                }
                acc[s][j] = a;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned s = 0u; s < 2u; ++s) {
        const unsigned qi = warp * 2u + s;
        if (qi >= n_q) {
            break;
        }
        /* An empty span contributes nothing rather than a row of NaN; see the
         * decode kernel's guard. */
        const float inv = (l[s] > 0.0f) ? (1.0f / l[s]) : 0.0f;
        const uint64_t base = ((uint64_t) (t0 + qi) * args.n_head + h) * head_dim;
        /* Unguarded, and therefore the `head_dim % 32 == 0` precondition above:
         * `slots` is head_dim / 32 and the 32 lanes cover the row between them,
         * so a head that is not a whole number of lanes would leave its tail
         * unwritten AND scatter part of it past the row. */
#pragma unroll
        for (unsigned j = 0u; j < DENSE_ATTN_LANE_SLOTS; ++j) {
            if (j >= slots) {
                break;
            }
            out[base + lane + 32u * j] = acc[s][j] * inv;
        }
    }
}

#endif /* SHISO_DENSE_ATTN_CUH */

/* Gated attention: the causal GQA pair over the paged f16 cache, and the gate
 * its output takes before the output projection.
 *
 * The CUDA twins of `kernels/metal/qwen35/gated_attn.metal` and
 * `gated_attn_prefill.metal`, over the laws `kernels/cuda/q4e/qsa.cu` carries
 * as `q4e_qsa_attention_kernel`, `q4e_qsa_attention_tiled_kernel` and
 * `q4e_qsa_gate_kernel`.
 *
 * `hkv = h / (n_head / n_head_kv)` -- an integer DIVIDE. The delta rule's key
 * head is `h % n_head_k`, a modulo, and the two families sit ten lines apart in
 * `q4e/qsa.cu` and `q4e/gdn.cu`. Swapping one for the other leaves every head
 * plausible and wrong.
 *
 * WHAT THIS DIRECTORY'S MANIFEST BUYS HERE: with `--use_fast_math` gone, `expf`
 * is the accurate entry and `__expf` the intrinsic, per call. The online
 * softmax takes `__expf` throughout, because both references do -- `q4e/qsa.cu`
 * writes `__expf` and the Metal bodies sit under the library's fast default.
 *
 * WHAT IS STAGED, AND WHY IT IS NOT THE CUDA ORIGINAL'S ANSWER. The tiled law
 * stages the transposed keys, the values and the query tile; the Metal port
 * stages the keys alone, 16 KiB at head_dim 256, because the other two would
 * need 50 KiB against a 32768 B threadgroup limit. That limit does not exist
 * here -- a `sm_121` block may opt into 101376 B, measured -- so this file
 * could take the original's shape, and does not. Two reasons, both about what
 * is being graded rather than about what is fastest: the Metal arrangement is
 * what produced the committed goldens and what
 * `crates/shiso-kerneltest/src/qwen35/attn.rs` grades, and its own note is that
 * the two extra copies are bandwidth caches rather than arithmetic - the same
 * f16 values in the same order, so the score loop's serial fold over the head
 * and the accumulator's fold over a tile's keys are unchanged either way.
 * Staging them is therefore a measurement for a perf task, not a correctness
 * one, and it can be made later without moving a number.
 *
 * Geometry, per `KERNELS.md`: decode grid `(n_head, n_tok)`, block 256, 8256 B;
 * prefill grid `(n_head, ceil(n_tok / 16))`, block 256, 16384 B; the gate flat
 * at `ceil(n_tok * n_head * head_dim / 256)`, block 256, no shared memory.
 */
#ifndef SHISO_Q35_ATTN_CUH
#define SHISO_Q35_ATTN_CUH

#include "q35_common.cuh"

/* Causal GQA over the resident cache with an online softmax, one block per
 * (query head, token).
 *
 * Positions are spread over the block's WARPS, not over its threads: a warp
 * owns whole positions and reduces its score with a butterfly, so nothing in
 * the position loop needs a barrier. The per-warp (max, denominator,
 * accumulator) triples are merged at the end by re-referencing every partial
 * against the global max, which is what makes the merge order-independent. */
extern "C" __global__ void kernel_q35_attn_decode(
        const struct q35_args_attn_decode args,
        float *__restrict__ out,
        const float *__restrict__ q,
        const __half *__restrict__ k_cache,
        const __half *__restrict__ v_cache,
        const int32_t *__restrict__ pos,
        const int32_t *__restrict__ pages) {
    const unsigned head_dim = args.head_dim;
    const unsigned t = blockIdx.y;
    const unsigned h = blockIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5u;
    const unsigned hkv = h / (args.n_head / args.n_head_kv);
    const float scale = rsqrtf((float) head_dim);
    const unsigned slots = (head_dim + 31u) / 32u;

    /* s_v[Q35_ATTN_WARPS][Q35_ATTN_HD], then s_m and s_l below it. The stride
     * is the family's widest head rather than `head_dim`, because that is what
     * the 8256 B the host asks for is computed from. */
    float *s_v = (float *) q35_smem;
    float *s_m = s_v + Q35_ATTN_WARPS * Q35_ATTN_HD;
    float *s_l = s_m + Q35_ATTN_WARPS;

    const uint64_t qbase = ((uint64_t) t * args.n_head + h) * (uint64_t) head_dim;
    float qv[Q35_ATTN_LANE_SLOTS];
    float acc[Q35_ATTN_LANE_SLOTS];
    for (unsigned j = 0u; j < slots; j++) {
        const unsigned i = lane + 32u * j;
        qv[j] = (i < head_dim) ? q[qbase + i] * scale : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    /* The causal bound is the position of the token itself, read off `pos`
     * rather than off a token count. The rows under it are translated one at a
     * time: a walk over positions crosses a page boundary wherever the session
     * happened to start each page. */
    const unsigned last = (uint32_t) pos[t];
    /* This ROW's page table, not the launch's. A batched decode step carries
     * one row per session and two sessions share no page, so a table selected
     * once for the whole launch would answer every row past the first out of
     * row zero's conversation. */
    const int32_t *row_pages = q35_row_pages(pages, t, args.page_stride);
    for (unsigned p = warp; p <= last; p += Q35_ATTN_WARPS) {
        const uint64_t kb =
            (q35_kv_row(row_pages, p, args.page_tokens) * args.n_head_kv + hkv)
            * (uint64_t) head_dim;
        float dot = 0.0f;
        for (unsigned j = 0u; j < slots; j++) {
            const unsigned i = lane + 32u * j;
            if (i < head_dim) {
                dot = q35_fma(qv[j], __half2float(k_cache[kb + i]), dot);
            }
        }
        /* Butterfly, so every lane ends up holding the whole score. */
        for (unsigned off = 16u; off > 0u; off >>= 1u) {
            dot += __shfl_xor_sync(Q35_FULL_WARP, dot, off);
        }

        const float m_new = fmaxf(m, dot);
        const float corr = __expf(m - m_new);
        const float w = __expf(dot - m_new);
        for (unsigned j = 0u; j < slots; j++) {
            const unsigned i = lane + 32u * j;
            const float v = (i < head_dim) ? __half2float(v_cache[kb + i]) : 0.0f;
            acc[j] = acc[j] * corr + w * v;
        }
        l = l * corr + w;
        m = m_new;
    }

    for (unsigned j = 0u; j < slots; j++) {
        const unsigned i = lane + 32u * j;
        if (i < head_dim) {
            s_v[warp * Q35_ATTN_HD + i] = acc[j];
        }
    }
    if (lane == 0u) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    __syncthreads();

    float gm = -INFINITY;
    for (unsigned w = 0u; w < Q35_ATTN_WARPS; w++) {
        gm = fmaxf(gm, s_m[w]);
    }
    float den = 0.0f;
    for (unsigned w = 0u; w < Q35_ATTN_WARPS; w++) {
        den += s_l[w] * __expf(s_m[w] - gm);
    }
    const float inv = 1.0f / den;
    for (unsigned i = threadIdx.x; i < head_dim; i += 256u) {
        float num = 0.0f;
        for (unsigned w = 0u; w < Q35_ATTN_WARPS; w++) {
            num += s_v[w * Q35_ATTN_HD + i] * __expf(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

/* Tiled causal GQA for prefill: Q35_ATTN_QT queries per block, Q35_ATTN_KT keys
 * per tile, one key per lane.
 *
 * Two shapes carry this kernel and both come from the tiled law. Lane `l`
 * scores key `p0 + l` entirely on its own -- there is no per-key reduction --
 * so the tile's softmax needs a max butterfly over the whole warp instead. And
 * the key weights of a tile are warp-private: the law keeps them in a per-warp
 * row of shared memory ordered by a warp barrier, and here, as on the Metal
 * side, the same broadcast is a shuffle from the lane that computed the weight,
 * which needs no barrier. That keeps every barrier outside the query loop,
 * where a short final tile would otherwise leave some warps unable to reach
 * one. A barrier added there changes nothing numerically and costs
 * throughput. */
extern "C" __global__ void kernel_q35_attn_prefill(
        const struct q35_args_attn_decode args,
        float *__restrict__ out,
        const float *__restrict__ q,
        const __half *__restrict__ k_cache,
        const __half *__restrict__ v_cache,
        const int32_t *__restrict__ pos,
        const int32_t *__restrict__ pages) {
    const unsigned head_dim = args.head_dim;
    const unsigned h = blockIdx.x;
    const unsigned t0 = blockIdx.y * Q35_ATTN_QT;
    if (t0 >= args.n_tok) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    const unsigned hkv = h / (args.n_head / args.n_head_kv);
    const unsigned slots = head_dim / 32u;
    const float scale = rsqrtf((float) head_dim);

    __half *s_k = (__half *) q35_smem;
    /* One table for the whole launch, and `args.page_stride` is not read here.
     * This entry is reached only by a chunk of Q35_ATTN_QT tokens or more, and
     * a chunk is one session's own consecutive positions by construction - a
     * decode batch is at most MAX_DECODE_BATCH rows and takes the decode entry
     * above. The host refuses a per-row table on this path rather than letting
     * a tile silently read row t0's. */
    const unsigned n_q =
        (args.n_tok - t0 < Q35_ATTN_QT) ? (args.n_tok - t0) : Q35_ATTN_QT;

    /* The furthest key any query of this tile may attend to: the bound is the
     * position of the last query, and the rows under it are translated through
     * the session page table as they are staged. */
    unsigned last = 0u;
    for (unsigned i = 0u; i < n_q; i++) {
        const unsigned p = (uint32_t) pos[t0 + i];
        if (p > last) {
            last = p;
        }
    }

    /* Two queries per warp, each with its own running max, running denominator
     * and head_dim/32 accumulator slots. */
    float acc[2][Q35_ATTN_LANE_SLOTS];
    float m[2], l[2];
    for (unsigned s = 0u; s < 2u; s++) {
        m[s] = -INFINITY;
        l[s] = 0.0f;
        for (unsigned j = 0u; j < slots; j++) {
            acc[s][j] = 0.0f;
        }
    }

    for (unsigned p0 = 0u; p0 <= last; p0 += Q35_ATTN_KT) {
        const unsigned n_k =
            (last + 1u - p0 < Q35_ATTN_KT) ? (last + 1u - p0) : Q35_ATTN_KT;
        for (unsigned idx = tid; idx < n_k * head_dim; idx += blockDim.x) {
            const unsigned k = idx / head_dim;
            const unsigned d = idx - k * head_dim;
            const uint64_t row = q35_kv_row(pages, p0 + k, args.page_tokens);
            const uint64_t src =
                (row * args.n_head_kv + hkv) * (uint64_t) head_dim + d;
            s_k[(size_t) d * Q35_ATTN_KT + k] = k_cache[src];
        }
        __syncthreads();

        for (unsigned s = 0u; s < 2u; s++) {
            const unsigned qi = warp * 2u + s;
            if (qi >= n_q) {
                break;
            }
            const unsigned qpos = (uint32_t) pos[t0 + qi];
            const uint64_t qbase =
                ((uint64_t) (t0 + qi) * args.n_head + h) * (uint64_t) head_dim;

            float dot = 0.0f;
            if (lane < n_k && p0 + lane <= qpos) {
                for (unsigned d = 0u; d < head_dim; d++) {
                    dot = q35_fma(q[qbase + d] * scale,
                                  __half2float(s_k[(size_t) d * Q35_ATTN_KT + lane]), dot);
                }
            } else {
                dot = -INFINITY;
            }

            float tile_max = dot;
            for (unsigned off = 16u; off > 0u; off >>= 1u) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(Q35_FULL_WARP, tile_max, off));
            }
            if (tile_max == -INFINITY) {
                continue; /* every key of this tile is masked for this query */
            }

            const float m_new = fmaxf(m[s], tile_max);
            const float corr = __expf(m[s] - m_new);
            const float w = __expf(dot - m_new); /* 0 wherever the key was masked */
            float wsum = w;
            for (unsigned off = 16u; off > 0u; off >>= 1u) {
                wsum += __shfl_xor_sync(Q35_FULL_WARP, wsum, off);
            }
            l[s] = l[s] * corr + wsum;
            m[s] = m_new;

            for (unsigned j = 0u; j < slots; j++) {
                const unsigned d = lane + 32u * j;
                float a = acc[s][j] * corr;
                for (unsigned k = 0u; k < n_k; k++) {
                    const uint64_t vrow = q35_kv_row(pages, p0 + k, args.page_tokens);
                    const uint64_t vb =
                        (vrow * args.n_head_kv + hkv) * (uint64_t) head_dim + d;
                    a = q35_fma(__shfl_sync(Q35_FULL_WARP, w, (int) k),
                                __half2float(v_cache[vb]), a);
                }
                acc[s][j] = a;
            }
        }
        __syncthreads();
    }

    for (unsigned s = 0u; s < 2u; s++) {
        const unsigned qi = warp * 2u + s;
        if (qi >= n_q) {
            break;
        }
        const float inv = 1.0f / l[s];
        const uint64_t base =
            ((uint64_t) (t0 + qi) * args.n_head + h) * (uint64_t) head_dim;
        for (unsigned j = 0u; j < slots; j++) {
            out[base + lane + 32u * j] = acc[s][j] * inv;
        }
    }
}

/* The attention output gate: x *= sigmoid(gate). Applied after attention and
 * before the output projection, so it never sees a normalized or scaled value.
 * The gate is the untouched second half of each query row, which
 * `kernel_q35_q_norm_rope` copied out raw. */
extern "C" __global__ void kernel_q35_attn_gate_mul(
        const struct q35_args_attn_gate args,
        float *__restrict__ x,
        const float *__restrict__ gate) {
    const uint64_t i = (uint64_t) blockIdx.x * 256u + threadIdx.x;
    if (i >= (uint64_t) args.n_tok * args.n_head * args.head_dim) {
        return;
    }
    x[i] *= q35_sigmoid(gate[i]);
}

#endif /* SHISO_Q35_ATTN_CUH */

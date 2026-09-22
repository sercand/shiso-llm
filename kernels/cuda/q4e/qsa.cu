/* Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels).
 * No line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 */

#include "q4e.cuh"

/* ---------------------------------------------------------------------------
 * QSA attention layers (every SHS_N_FULL_ATTN_INTERVAL-th layer).
 *
 * The query projection emits a query and an output gate interleaved per head:
 * head h owns [q(head_dim) | gate(head_dim)] inside a 2 * head_dim * n_head
 * row.  Splitting the flat row down the middle instead is wrong and is the
 * single easiest mistake to make here.
 *
 * RoPE covers only the first n_rot of head_dim in NeoX pairing (i with
 * i + n_rot/2); the rest passes through.  The checkpoint's interleaved mRoPE
 * degenerates to this for text, where all three position axes carry the same
 * value.
 * ------------------------------------------------------------------------ */

/* Per-head RMSNorm then partial RoPE, reading the query out of its strided
 * slot and writing it contiguous. */
extern "C" __global__ void q4e_qsa_q_norm_rope_kernel(
        float *q_out, float *gate_out, const float *qkv, const float *w,
        uint32_t head_dim, uint32_t n_head, uint32_t n_rot, float rope_base,
        const int32_t *mrope, uint32_t sec_t, uint32_t sec_h, uint32_t sec_w,
        float eps) {
    const uint32_t t = blockIdx.y;
    const uint32_t h = blockIdx.x;
    const uint64_t src = (uint64_t)t * n_head * 2u * head_dim + (uint64_t)h * 2u * head_dim;
    const uint64_t dst = ((uint64_t)t * n_head + h) * head_dim;

    extern __shared__ float q4e_q_smem[];
    float *s_q = q4e_q_smem;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = qkv[src + i];
        s_q[i] = v;
        sum += v * v;
        gate_out[dst + i] = qkv[src + head_dim + i];
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) s_q[i] *= scale * w[i];
    __syncthreads();
    q4e_rope_mrope(s_q, n_rot, rope_base,
                   mrope[3u * t], mrope[3u * t + 1u], mrope[3u * t + 2u],
                   sec_t, sec_h, sec_w, threadIdx.x, blockDim.x);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) q_out[dst + i] = s_q[i];
}

/* Key norm + RoPE straight into the f16 KV cache, and the value beside it.
 * The cache is f16 because it is read in full on every decode step; f32 would
 * double that traffic for no accuracy that survives the softmax. */
extern "C" __global__ void q4e_qsa_store_kv_kernel(
        __half *k_cache, __half *v_cache, const float *k, const float *v,
        const float *kw, uint32_t head_dim, uint32_t n_head_kv, uint32_t n_rot,
        float rope_base, const int32_t *pos, const int32_t *mrope,
        uint32_t sec_t, uint32_t sec_h, uint32_t sec_w, const int32_t *pages,
        uint32_t pool_slots, float eps) {
    const uint32_t t = blockIdx.y;
    const uint32_t h = blockIdx.x;
    const uint64_t src = ((uint64_t)t * n_head_kv + h) * head_dim;

    extern __shared__ float q4e_k_smem[];
    float *s_k = q4e_k_smem;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float x = k[src + i];
        s_k[i] = x;
        sum += x * x;
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) s_k[i] *= scale * kw[i];
    __syncthreads();
    /* This is where the decoupling is visible in one line: the angle comes from
     * mrope, the cache slot below still comes from pos.  They are the same
     * number for text and diverge inside an image. */
    q4e_rope_mrope(s_k, n_rot, rope_base,
                   mrope[3u * t], mrope[3u * t + 1u], mrope[3u * t + 2u],
                   sec_t, sec_h, sec_w, threadIdx.x, blockDim.x);
    __syncthreads();

    /* The only write into the cache, and it is always at the frontier: the
     * page holding pos[t] is the last one the sequence owns. */
    const uint64_t slot = ((uint64_t)q4e_kv_row(pages, (uint32_t)pos[t]) * n_head_kv + h) * head_dim;
    if (slot + head_dim > (uint64_t)pool_slots * n_head_kv * head_dim) return;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        k_cache[slot + i] = __float2half(s_k[i]);
        v_cache[slot + i] = __float2half(v[src + i]);
    }
}

/* Causal GQA over the resident cache with online softmax.  One block per
 * (query head, token); the 24 query heads share 2 key/value heads.
 *
 * Positions are spread over the block's warps, not over its threads: a warp
 * owns whole positions and reduces its dot product with shuffles, so nothing
 * in the position loop needs __syncthreads.  Walking positions one at a time
 * with a block-wide reduction each -- the obvious shape -- cost 1.35 us per
 * position and made attention 9% of a decode step at a 400-token context,
 * for 830 KiB of cache traffic.
 *
 * Each lane keeps head_dim / 32 accumulator slots in registers; the warps'
 * partial (max, denominator, accumulator) triples are merged at the end. */
#define Q4E_ATTN_LANE_SLOTS 8u          /* head_dim <= 32 * this */
#define Q4E_ATTN_WARPS      8u

extern "C" __global__ void q4e_qsa_attention_kernel(
        float *out, const __half *k_cache, const __half *v_cache, const float *q,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *pages, uint32_t n_tok) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t hkv = h / (n_head / n_head_kv);
    const uint32_t last = (uint32_t)pos[t];
    const float scale = rsqrtf((float)head_dim);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slots = (head_dim + 31u) / 32u;
    const uint64_t qbase = ((uint64_t)t * n_head + h) * head_dim;

    float qv[Q4E_ATTN_LANE_SLOTS];
    float acc[Q4E_ATTN_LANE_SLOTS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        qv[j] = (i < head_dim) ? q[qbase + i] * scale : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    /* A warp walks positions Q4E_ATTN_WARPS apart, so this one pays a page
     * lookup per position; it is the A/B fallback path, not the hot one. */
    for (uint32_t p = warp; p <= last; p += Q4E_ATTN_WARPS) {
        const uint64_t kb = ((uint64_t)q4e_kv_row(pages, p) * n_head_kv + hkv) * head_dim;
        float dot = 0.0f;
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            if (i < head_dim) dot += qv[j] * __half2float(k_cache[kb + i]);
        }
        /* Butterfly, so every lane ends up with the score. */
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);

        const float m_new = fmaxf(m, dot);
        const float corr = __expf(m - m_new);
        const float w = __expf(dot - m_new);
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            const float v = (i < head_dim) ? __half2float(v_cache[kb + i]) : 0.0f;
            acc[j] = acc[j] * corr + w * v;
        }
        l = l * corr + w;
        m = m_new;
    }

    /* Merge the per-warp partials.  s_acc is warp-major so the final pass
     * reads one contiguous row per warp. */
    extern __shared__ float q4e_attn_smem[];
    float *s_acc = q4e_attn_smem;                       /* warps * head_dim */
    __shared__ float s_m[Q4E_ATTN_WARPS];
    __shared__ float s_l[Q4E_ATTN_WARPS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        if (i < head_dim) s_acc[(uint64_t)warp * head_dim + i] = acc[j];
    }
    if (lane == 0u) { s_m[warp] = m; s_l[warp] = l; }
    __syncthreads();

    float gm = -INFINITY;
    for (uint32_t w = 0; w < Q4E_ATTN_WARPS; w++) gm = fmaxf(gm, s_m[w]);
    float den = 0.0f;
    for (uint32_t w = 0; w < Q4E_ATTN_WARPS; w++) den += s_l[w] * __expf(s_m[w] - gm);
    const float inv = 1.0f / den;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float num = 0.0f;
        for (uint32_t w = 0; w < Q4E_ATTN_WARPS; w++) {
            num += s_acc[(uint64_t)w * head_dim + i] * __expf(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

/* Batched-decode variants of the KV store and the per-query attention.  Each
 * query row (blockIdx.y = t) belongs to a different sequence, so `pages`
 * becomes a per-row array of page tables; pos, q, k, v and out are already
 * row-major over t.  The bodies are the single-token scalar kernels with only
 * that indirection changed -- the tiled prefill attention is not used here
 * because it stages several queries under one shared page table. */
extern "C" __global__ void q4e_qsa_store_kv_batch_kernel(
        __half *k_cache, __half *v_cache, const float *k, const float *v,
        const float *kw, uint32_t head_dim, uint32_t n_head_kv, uint32_t n_rot,
        float rope_base, const int32_t *pos, const int32_t *mrope,
        uint32_t sec_t, uint32_t sec_h, uint32_t sec_w,
        const int32_t *const *pages_arr, uint32_t pool_slots, float eps) {
    const uint32_t t = blockIdx.y;
    const uint32_t h = blockIdx.x;
    const int32_t *pages = pages_arr[t];
    const uint64_t src = ((uint64_t)t * n_head_kv + h) * head_dim;

    extern __shared__ float q4e_k_smem[];
    float *s_k = q4e_k_smem;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float x = k[src + i];
        s_k[i] = x;
        sum += x * x;
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) s_k[i] *= scale * kw[i];
    __syncthreads();
    q4e_rope_mrope(s_k, n_rot, rope_base,
                   mrope[3u * t], mrope[3u * t + 1u], mrope[3u * t + 2u],
                   sec_t, sec_h, sec_w, threadIdx.x, blockDim.x);
    __syncthreads();
    const uint64_t slot = ((uint64_t)q4e_kv_row(pages, (uint32_t)pos[t]) * n_head_kv + h) * head_dim;
    if (slot + head_dim > (uint64_t)pool_slots * n_head_kv * head_dim) return;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        k_cache[slot + i] = __float2half(s_k[i]);
        v_cache[slot + i] = __float2half(v[src + i]);
    }
}

extern "C" __global__ void q4e_qsa_attention_batch_kernel(
        float *out, const __half *k_cache, const __half *v_cache, const float *q,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *const *pages_arr, uint32_t n_tok) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const int32_t *pages = pages_arr[t];
    const uint32_t hkv = h / (n_head / n_head_kv);
    const uint32_t last = (uint32_t)pos[t];
    const float scale = rsqrtf((float)head_dim);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slots = (head_dim + 31u) / 32u;
    const uint64_t qbase = ((uint64_t)t * n_head + h) * head_dim;

    float qv[Q4E_ATTN_LANE_SLOTS];
    float acc[Q4E_ATTN_LANE_SLOTS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        qv[j] = (i < head_dim) ? q[qbase + i] * scale : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    for (uint32_t p = warp; p <= last; p += Q4E_ATTN_WARPS) {
        const uint64_t kb = ((uint64_t)q4e_kv_row(pages, p) * n_head_kv + hkv) * head_dim;
        float dot = 0.0f;
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            if (i < head_dim) dot += qv[j] * __half2float(k_cache[kb + i]);
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);

        const float m_new = fmaxf(m, dot);
        const float corr = __expf(m - m_new);
        const float w = __expf(dot - m_new);
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            const float v = (i < head_dim) ? __half2float(v_cache[kb + i]) : 0.0f;
            acc[j] = acc[j] * corr + w * v;
        }
        l = l * corr + w;
        m = m_new;
    }

    extern __shared__ float q4e_attn_smem[];
    float *s_acc = q4e_attn_smem;
    __shared__ float s_m[Q4E_ATTN_WARPS];
    __shared__ float s_l[Q4E_ATTN_WARPS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        if (i < head_dim) s_acc[(uint64_t)warp * head_dim + i] = acc[j];
    }
    if (lane == 0u) { s_m[warp] = m; s_l[warp] = l; }
    __syncthreads();

    float gm = -INFINITY;
    for (uint32_t w = 0; w < Q4E_ATTN_WARPS; w++) gm = fmaxf(gm, s_m[w]);
    float den = 0.0f;
    for (uint32_t w = 0; w < Q4E_ATTN_WARPS; w++) den += s_l[w] * __expf(s_m[w] - gm);
    const float inv = 1.0f / den;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float num = 0.0f;
        for (uint32_t w = 0; w < Q4E_ATTN_WARPS; w++) {
            num += s_acc[(uint64_t)w * head_dim + i] * __expf(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

/* Tiled causal GQA, for prefill.
 *
 * The decode kernel above gives one block per (head, query) and streams the
 * whole key cache past it, so a chunk of T queries reads the cache T times.
 * At 1024-token chunks that was 40% of prefill.  Here a block owns Q4E_ATTN_QT
 * queries of one head and stages each key tile in shared memory, so the cache
 * is read once per query tile instead of once per query.
 *
 * The score for key `l` lives entirely in lane `l`, which is what removes the
 * per-key warp reduction the decode kernel pays: one max-reduction per tile of
 * 32 keys rather than one per key.  s_k is stored dim-major so those reads hit
 * 32 different banks; s_v stays key-major because the accumulation walks dims. */
#define Q4E_ATTN_QT 16u          /* queries per block */
#define Q4E_ATTN_KT 32u          /* keys per tile: one per lane */
#define Q4E_ATTN_TILE_WARPS 8u

/* Key tiles start at multiples of Q4E_ATTN_KT and a page is a whole number of
 * them, so a tile never straddles a page and one translation covers it. */
static_assert(SHS_Q4E_PAGE_TOKENS % Q4E_ATTN_KT == 0u,
              "a key tile must not straddle a KV page");

extern "C" __global__ void q4e_qsa_attention_tiled_kernel(
        float *out, const __half *k_cache, const __half *v_cache, const float *q,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *pages, uint32_t n_tok) {
    const uint32_t h = blockIdx.x;
    const uint32_t t0 = blockIdx.y * Q4E_ATTN_QT;
    if (t0 >= n_tok) return;
    const uint32_t hkv = h / (n_head / n_head_kv);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slots = head_dim / 32u;          /* dims per lane */
    const float scale = rsqrtf((float)head_dim);

    extern __shared__ char q4e_tile_smem[];
    __half *s_k = (__half *)q4e_tile_smem;                        /* [dim][KT] */
    __half *s_v = s_k + (size_t)head_dim * Q4E_ATTN_KT;           /* [KT][dim] */
    float  *s_q = (float *)(s_v + (size_t)head_dim * Q4E_ATTN_KT);/* [QT][dim] */
    float  *s_w = s_q + (size_t)Q4E_ATTN_QT * head_dim;           /* [warps][KT] */

    /* This block's queries, and the furthest key any of them may attend to. */
    const uint32_t n_q = (n_tok - t0 < Q4E_ATTN_QT) ? (n_tok - t0) : Q4E_ATTN_QT;
    uint32_t last = 0;
    for (uint32_t i = 0; i < n_q; i++) {
        const uint32_t p = (uint32_t)pos[t0 + i];
        if (p > last) last = p;
    }
    for (uint32_t idx = threadIdx.x; idx < n_q * head_dim; idx += blockDim.x) {
        const uint32_t qi = idx / head_dim, d = idx % head_dim;
        s_q[qi * head_dim + d] = q[((uint64_t)(t0 + qi) * n_head + h) * head_dim + d] * scale;
    }

    /* Two queries per warp; each lane keeps head_dim / 32 accumulator slots
     * for each of them. */
    float acc[2][Q4E_ATTN_LANE_SLOTS];
    float m[2], l[2];
    for (int s = 0; s < 2; s++) {
        m[s] = -INFINITY;
        l[s] = 0.0f;
        for (uint32_t j = 0; j < slots; j++) acc[s][j] = 0.0f;
    }
    __syncthreads();

    for (uint32_t p0 = 0; p0 <= last; p0 += Q4E_ATTN_KT) {
        const uint32_t n_k = (last + 1u - p0 < Q4E_ATTN_KT) ? (last + 1u - p0) : Q4E_ATTN_KT;
        const uint32_t krow = q4e_kv_row(pages, p0);     /* one lookup per tile */
        for (uint32_t idx = threadIdx.x; idx < n_k * head_dim; idx += blockDim.x) {
            const uint32_t k = idx / head_dim, d = idx % head_dim;
            const uint64_t src = ((uint64_t)(krow + k) * n_head_kv + hkv) * head_dim + d;
            s_k[(size_t)d * Q4E_ATTN_KT + k] = k_cache[src];
            s_v[(size_t)k * head_dim + d] = v_cache[src];
        }
        __syncthreads();

        for (int s = 0; s < 2; s++) {
            const uint32_t qi = warp * 2u + (uint32_t)s;
            if (qi >= n_q) break;
            const uint32_t qpos = (uint32_t)pos[t0 + qi];
            const float *qrow = s_q + (size_t)qi * head_dim;

            /* Lane l scores key p0 + l against this query, alone. */
            float dot = 0.0f;
            if (lane < n_k && p0 + lane <= qpos) {
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot = fmaf(qrow[d], __half2float(s_k[(size_t)d * Q4E_ATTN_KT + lane]), dot);
                }
            } else {
                dot = -INFINITY;
            }

            float tile_max = dot;
            for (int off = 16; off > 0; off >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));
            }
            if (tile_max == -INFINITY) continue;          /* fully masked tile */
            const float m_new = fmaxf(m[s], tile_max);
            const float corr = __expf(m[s] - m_new);
            const float w = __expf(dot - m_new);          /* 0 where masked */
            s_w[(size_t)warp * Q4E_ATTN_KT + lane] = w;

            float wsum = w;
            for (int off = 16; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
            l[s] = l[s] * corr + wsum;
            m[s] = m_new;

            __syncwarp();
            for (uint32_t j = 0; j < slots; j++) {
                const uint32_t d = lane + 32u * j;
                float a = acc[s][j] * corr;
                for (uint32_t k = 0; k < n_k; k++) {
                    a = fmaf(s_w[(size_t)warp * Q4E_ATTN_KT + k],
                             __half2float(s_v[(size_t)k * head_dim + d]), a);
                }
                acc[s][j] = a;
            }
            __syncwarp();
        }
        __syncthreads();
    }

    for (int s = 0; s < 2; s++) {
        const uint32_t qi = warp * 2u + (uint32_t)s;
        if (qi >= n_q) break;
        const float inv = 1.0f / l[s];
        const uint64_t base = ((uint64_t)(t0 + qi) * n_head + h) * head_dim;
        for (uint32_t j = 0; j < slots; j++) {
            out[base + lane + 32u * j] = acc[s][j] * inv;
        }
    }
}

/* out = attn * sigmoid(gate), then the caller projects with attn_output. */
extern "C" __global__ void q4e_qsa_gate_kernel(float *x, const float *gate, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + __expf(-gate[i]));
}

// Tiled causal GQA attention for prefill: Q35_ATTN_QT queries per threadgroup,
// Q35_ATTN_KT keys per tile, one key per lane.
//
// Two shapes carry this kernel and both come from the tiled CUDA original.  The
// first is that lane l scores key p0 + l entirely on its own -- there is no
// per-key reduction -- so the tile's softmax needs a max butterfly over the
// whole simdgroup instead.  The second is that the key weights of a tile are
// warp-private: CUDA keeps them in a per-warp row of shared memory and orders it
// with a warp barrier, and here the same register broadcast is a shuffle from the
// lane that computed the weight, which needs no barrier at all.  That also keeps
// every barrier in this kernel outside the query loop, where a short final tile
// would otherwise leave some simdgroups unable to reach one.
//
// Threadgroup memory is the transposed keys alone: 16 KiB at head_dim 256.  The
// CUDA original also stages the values and the queries, which would need 50 KiB
// in total -- above this device's 32 KiB per-threadgroup limit.  Both of those
// copies are bandwidth caches rather than arithmetic: the device holds the same
// f16 values in the same order, so reading them directly leaves the score loop's
// serial fold over the head dimension, and the accumulator's fold over the keys
// of a tile, exactly as they are.
// One table for the whole launch, and `args.page_stride` is not read here.
// This entry is reached only by a chunk of Q35_ATTN_QT tokens or more, and a
// chunk is one session's own consecutive positions by construction - a decode
// batch is at most MAX_DECODE_BATCH rows and takes the decode entry instead.
// The host refuses a per-row table on this path rather than letting a tile
// silently read row t0's.
kernel void kernel_q35_attn_prefill(
        constant q35_args_attn_prefill & args [[buffer(0)]],
        device float *out [[buffer(1)]],
        device const float *q [[buffer(2)]],
        device const half *k_cache [[buffer(3)]],
        device const half *v_cache [[buffer(4)]],
        device const int32_t *pos [[buffer(5)]],
        device const int32_t *pages [[buffer(6)]],
        threadgroup char *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint h = tgpig.x;
    const uint t0 = tgpig.y * Q35_ATTN_QT;
    if (t0 >= args.n_tok) {
        return;
    }
    const uint tid = tpitg.x;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    // 16 query heads share 4 key/value heads: an integer divide, not a modulo.
    const uint hkv = h / (args.n_head / args.n_head_kv);
    const uint slots = head_dim / 32u;
    const float scale = rsqrt((float) head_dim);

    threadgroup half *s_k = (threadgroup half *) tbuf;
    const uint n_q = (args.n_tok - t0 < Q35_ATTN_QT) ? (args.n_tok - t0) : Q35_ATTN_QT;

    // The furthest key any query of this tile may attend to: the bound is the
    // position of the last query, and the rows under it are translated through
    // the session page table as they are staged.
    uint last = 0u;
    for (uint i = 0u; i < n_q; i++) {
        const uint p = (uint32_t) pos[t0 + i];
        if (p > last) {
            last = p;
        }
    }

    // Two queries per simdgroup, each with its own running max, running
    // denominator and head_dim/32 accumulator slots.
    float acc[2][Q35_ATTN_LANE_SLOTS];
    float m[2], l[2];
    for (uint s = 0u; s < 2u; s++) {
        m[s] = -INFINITY;
        l[s] = 0.0f;
        for (uint j = 0u; j < slots; j++) {
            acc[s][j] = 0.0f;
        }
    }

    for (uint p0 = 0u; p0 <= last; p0 += Q35_ATTN_KT) {
        const uint n_k = (last + 1u - p0 < Q35_ATTN_KT) ? (last + 1u - p0) : Q35_ATTN_KT;
        for (uint idx = tid; idx < n_k * head_dim; idx += ntg.x) {
            const uint k = idx / head_dim;
            const uint d = idx - k * head_dim;
            const uint64_t row = q35_kv_row(pages, p0 + k, args.page_tokens);
            const uint64_t src = ((uint64_t) row * args.n_head_kv + hkv) * head_dim + d;
            s_k[(size_t) d * Q35_ATTN_KT + k] = k_cache[src];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = 0u; s < 2u; s++) {
            const uint qi = warp * 2u + s;
            if (qi >= n_q) {
                break;
            }
            const uint qpos = (uint32_t) pos[t0 + qi];
            const uint64_t qbase = ((uint64_t) (t0 + qi) * args.n_head + h) * head_dim;

            float dot = 0.0f;
            if (lane < n_k && p0 + lane <= qpos) {
                for (uint d = 0u; d < head_dim; d++) {
                    dot = q35_fma(q[qbase + d] * scale, (float) s_k[(size_t) d * Q35_ATTN_KT + lane], dot);
                }
            } else {
                dot = -INFINITY;
            }

            float tile_max = dot;
            for (uint off = 16u; off > 0u; off >>= 1u) {
                tile_max = max(tile_max, simd_shuffle_xor(tile_max, off));
            }
            if (tile_max == -INFINITY) {
                continue;  // every key of this tile is masked for this query
            }

            const float m_new = max(m[s], tile_max);
            const float corr = exp(m[s] - m_new);
            const float w = exp(dot - m_new);  // 0 wherever the key was masked
            float wsum = w;
            for (uint off = 16u; off > 0u; off >>= 1u) {
                wsum += simd_shuffle_xor(wsum, off);
            }
            l[s] = l[s] * corr + wsum;
            m[s] = m_new;

            for (uint j = 0u; j < slots; j++) {
                const uint d = lane + 32u * j;
                float a = acc[s][j] * corr;
                for (uint k = 0u; k < n_k; k++) {
                    const uint64_t vrow = q35_kv_row(pages, p0 + k, args.page_tokens);
                    const uint64_t vb = ((uint64_t) vrow * args.n_head_kv + hkv) * head_dim + d;
                    a = q35_fma(simd_shuffle(w, (int) k), (float) v_cache[vb], a);
                }
                acc[s][j] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint s = 0u; s < 2u; s++) {
        const uint qi = warp * 2u + s;
        if (qi >= n_q) {
            break;
        }
        const float inv = 1.0f / l[s];
        const uint64_t base = ((uint64_t) (t0 + qi) * args.n_head + h) * head_dim;
        for (uint j = 0u; j < slots; j++) {
            out[base + lane + 32u * j] = acc[s][j] * inv;
        }
    }
}

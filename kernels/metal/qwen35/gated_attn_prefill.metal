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

// The prefill softmax on simdgroup matrices, for head_dim 256.
//
// The kernel above scores one key per lane with a scalar dot over the whole
// head, re-reading the query from device memory for every key, and blends the
// values one key at a time with the page-table translation inside that loop.
// Measured on an M4 over Qwen3.5-4B (16 query heads of 256, 2048-token
// prompt): 104 ms a layer-chunk, 44% of the whole prefill, at ~0.33 TFLOP/s.
//
// Here a threadgroup takes Q35_FA_QR query rows of one head, eight a
// simdgroup, held as 8x8 f32 fragments with the score scale folded in (it is
// 1/16 at this width, so the fold is exact). Keys and values are staged
// Q35_FA_KB at a time as the f16 the cache already holds, each position
// translated through the page table once, and multiplied as half fragments
// against the f32 ones; scores, probabilities, the online softmax state and the
// value accumulators are f32. A lane holds two columns of one row of every
// fragment, so the four lanes of a row fold the running max and denominator
// with two shuffles, and the accumulators are rescaled lane by lane.
//
// Nothing is narrowed that the scalar kernel kept wide. A first version held
// the query and the probabilities as half: it passed the kernel case, but a
// prompt fed in chunks whose tails fall to the f32 decode kernel then reordered
// the top of the logit row (up to 0.14 apart against the 512-wide route), which
// the chunk-route parity case refuses.
//
// Causality is per row: a key is live for a row when its position is at or
// before the row's own. The first block holds position 0, which every row may
// see, so a row's running max is finite from the first block on and a later
// block that is wholly masked for it rescales by exp(0) and adds zeros.
//
// grid `(n_head, ceil(n_tok / Q35_FA_QR))`, block 128, static threadgroup
// memory. One page table for the launch, as for the kernel above.
#define Q35_FA_SG 4u
#define Q35_FA_QR (8u * Q35_FA_SG)
#define Q35_FA_KB 16u
#define Q35_FA_HD 256u
// Staged row stride in halves: eight past the head so the transposed key
// loads of one fragment do not land on one bank.
#define Q35_FA_LD (Q35_FA_HD + 8u)
#define Q35_UNROLL _Pragma("clang loop unroll(full)")

kernel void kernel_q35_attn_prefill_mma(
        constant q35_args_attn_prefill & args [[buffer(0)]],
        device float *out [[buffer(1)]],
        device const float *q [[buffer(2)]],
        device const half *k_cache [[buffer(3)]],
        device const half *v_cache [[buffer(4)]],
        device const int32_t *pos [[buffer(5)]],
        device const int32_t *pages [[buffer(6)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint HD = Q35_FA_HD;
    constexpr uint ND = HD / 8u;
    constexpr uint KB = Q35_FA_KB;
    constexpr uint NB = KB / 8u;
    constexpr uint LD = Q35_FA_LD;
    constexpr uint NT = 32u * Q35_FA_SG;
    threadgroup half sk[KB * LD];
    threadgroup half sv[KB * LD];

    const uint h = tgpig.x;
    const uint t0 = tgpig.y * Q35_FA_QR;
    if (t0 >= args.n_tok) {
        return;
    }
    const uint hkv = h / (args.n_head / args.n_head_kv);
    const float scale = rsqrt((float) HD);
    const uint n_q = min(Q35_FA_QR, args.n_tok - t0);

    // The furthest key any row of this tile may attend to.
    uint last = 0u;
    for (uint i = 0u; i < n_q; i++) {
        last = max(last, (uint) (uint32_t) pos[t0 + i]);
    }

    // The lane's place in every 8x8 fragment: columns `fn, fn + 1` of row `fr`.
    const ushort fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    const ushort fr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    const uint row = t0 + sg * 8u + fr;
    const bool live = row < args.n_tok;
    // A row past the chunk computes against the first row's position and is
    // never written; it only has to stay finite.
    const uint qpos = (uint32_t) pos[live ? row : t0];

    simdgroup_float8x8 qf[ND];
    simdgroup_float8x8 of[ND];
    device const float *qp = q + ((uint64_t) (live ? row : t0) * args.n_head + h) * HD;
    Q35_UNROLL for (uint i = 0u; i < ND; ++i) {
        qf[i].thread_elements()[0] = qp[8u * i + fn] * scale;
        qf[i].thread_elements()[1] = qp[8u * i + fn + 1u] * scale;
        of[i] = simdgroup_float8x8(0.0f);
    }

    float run_max = -INFINITY;
    float run_sum = 0.0f;
    constexpr uint HD4 = HD / 4u;
    constexpr uint SL = KB * HD4 / NT;
    for (uint p0 = 0u; p0 <= last; p0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        Q35_UNROLL for (uint t = 0u; t < SL; ++t) {
            const uint e = tid + t * NT;
            const uint j = e / HD4;
            const uint d = (e - j * HD4) * 4u;
            const uint s = p0 + j;
            half4 kv4 = half4(0.0h);
            half4 vv4 = half4(0.0h);
            if (s <= last) {
                const uint64_t src =
                    ((uint64_t) q35_kv_row(pages, s, args.page_tokens) * args.n_head_kv + hkv) * HD + d;
                kv4 = half4(*((device const packed_half4 *) (k_cache + src)));
                vv4 = half4(*((device const packed_half4 *) (v_cache + src)));
            }
            *((threadgroup half4 *) (sk + j * LD + d)) = kv4;
            *((threadgroup half4 *) (sv + j * LD + d)) = vv4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores: S (8 x KB) = Q (8 x HD) . K^T, one fragment per 8 keys.
        simdgroup_float8x8 sf[NB];
        Q35_UNROLL for (uint b = 0u; b < NB; ++b) {
            sf[b] = simdgroup_float8x8(0.0f);
        }
        Q35_UNROLL for (uint i = 0u; i < ND; ++i) {
            Q35_UNROLL for (uint b = 0u; b < NB; ++b) {
                simdgroup_half8x8 kt;
                // kt(r, c) = K[8b + c][8i + r]: the transposed load.
                simdgroup_load(kt, sk + 8u * b * LD + 8u * i, LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sf[b], qf[i], kt, sf[b]);
            }
        }

        float block_max = -INFINITY;
        Q35_UNROLL for (uint b = 0u; b < NB; ++b) {
            Q35_UNROLL for (uint c = 0u; c < 2u; ++c) {
                const bool real = p0 + 8u * b + fn + c <= qpos;
                const float score = real ? sf[b].thread_elements()[c] : -INFINITY;
                sf[b].thread_elements()[c] = score;
                block_max = max(block_max, score);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        const float new_max = max(run_max, block_max);
        const float alpha = exp(run_max - new_max);
        float block_sum = 0.0f;
        simdgroup_float8x8 pf[NB];
        Q35_UNROLL for (uint b = 0u; b < NB; ++b) {
            Q35_UNROLL for (uint c = 0u; c < 2u; ++c) {
                const float p = exp(sf[b].thread_elements()[c] - new_max);
                pf[b].thread_elements()[c] = p;
                block_sum += p;
            }
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 8);
        run_sum = run_sum * alpha + block_sum;
        run_max = new_max;

        Q35_UNROLL for (uint i = 0u; i < ND; ++i) {
            of[i].thread_elements()[0] *= alpha;
            of[i].thread_elements()[1] *= alpha;
            Q35_UNROLL for (uint b = 0u; b < NB; ++b) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, sv + 8u * b * LD + 8u * i, LD);
                simdgroup_multiply_accumulate(of[i], pf[b], vf, of[i]);
            }
        }
    }

    if (!live) {
        return;
    }
    const float inv = 1.0f / run_sum;
    device float *op = out + ((uint64_t) row * args.n_head + h) * HD;
    Q35_UNROLL for (uint i = 0u; i < ND; ++i) {
        op[8u * i + fn] = of[i].thread_elements()[0] * inv;
        op[8u * i + fn + 1u] = of[i].thread_elements()[1] * inv;
    }
}

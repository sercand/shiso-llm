// A routed feed-forward over row-major affine expert stacks, with a biased
// sigmoid router and a squared-rectifier activation, and the attention store
// for a slot that neither normalizes nor rotates its keys.
//
// The expert stages share the column conventions of the qwen35moe set, because
// the sort and the combine between them are that set's kernels: a routed column
// is `token * n_used + slot`, the second projection reads the column's own row,
// and both write `[column, out_dim]` row-major. What differs is the storage:
// these read the container's own layout - expert `e`'s row `r` is
// `in_dim / 8` packed words at `(e * out_dim + r) * in_dim / 8`, and its
// scales and biases `in_dim / 64` bf16 values at `(e * out_dim + r) * in_dim /
// 64` - so a width that is not a multiple of a storage tile needs no padding.

#include <metal_stdlib>
using namespace metal;

/// Output rows one simdgroup of an expert stage covers: the input chunk a lane
/// loads is read once for all of them.
constant constexpr uint DENSE_EXPERT_ROWS = 4u;

/// Routed columns the fused router and sort holds; the host routes a wider
/// chunk through the separate sort.
constant constexpr uint DENSE_ROUTE_SORT_COLUMNS = 256u;

/// Simdgroups the router's threadgroup may hold; its argmax publishes one
/// candidate per simdgroup.
constant constexpr uint DENSE_ROUTE_MAX_SIMDGROUPS = 32u;

/// The better of two candidates: the higher score, the lower id on a tie.
static inline bool dense_route_better(float v, uint i, float w, uint j) {
    return v > w || (v == w && i < j);
}

/// One token's biased sigmoid choice, by the whole threadgroup: `score` and
/// `choice` are staged, each pick is a threadgroup argmax, and the chosen ids
/// and weights land at `ids[base..]` and `weights[base..]`.
///
/// Every thread of the threadgroup must reach it: it holds barriers.
static inline void dense_route_token(
        constant dense_args_moe_route & args,
        device int * ids,
        device float * weights,
        device const float * logits,
        device const float * bias,
        threadgroup float * score,
        threadgroup float * choice,
        threadgroup float * best_v,
        threadgroup uint * best_i,
        threadgroup int * tids,
        uint t,
        uint tid,
        uint nt,
        uint lane,
        uint sg,
        uint n_sg) {
    const uint e_n = args.experts;
    for (uint e = tid; e < e_n; e += nt) {
        const float s = 1.0f / (1.0f + exp(-logits[(uint64_t) t * e_n + e]));
        score[e] = s;
        choice[e] = s + bias[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint base = t * args.used;
    float total = 0.0f;
    for (uint k = 0; k < args.used; ++k) {
        float v = -INFINITY;
        uint i = 0xffffffffu;
        for (uint e = tid; e < e_n; e += nt) {
            if (dense_route_better(choice[e], e, v, i)) {
                v = choice[e];
                i = e;
            }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            const float w = simd_shuffle_down(v, off);
            const uint j = simd_shuffle_down(i, off);
            if (dense_route_better(w, j, v, i)) {
                v = w;
                i = j;
            }
        }
        if (lane == 0u) {
            best_v[sg] = v;
            best_i[sg] = i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bv = best_v[0];
        uint bi = best_i[0];
        for (uint g = 1; g < n_sg; ++g) {
            if (dense_route_better(best_v[g], best_i[g], bv, bi)) {
                bv = best_v[g];
                bi = best_i[g];
            }
        }
        total += score[bi];
        if (tid == 0) {
            choice[bi] = -INFINITY;
            ids[base + k] = int(bi);
            weights[base + k] = score[bi];
            if (tids != nullptr) {
                tids[base + k] = int(bi);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const float denom = (args.norm != 0u && args.used > 1u) ? total + 1e-20f : 1.0f;
        for (uint k = 0; k < args.used; ++k) {
            weights[base + k] = weights[base + k] / denom * args.scale;
        }
    }
}

/// Scores every expert with a sigmoid, chooses the `used` best by score plus a
/// per-expert bias, and weights each chosen expert by its UNBIASED score.
///
/// The bias only steers the choice; the weights are the sigmoids themselves,
/// renormalized to sum to one when `norm` is set and then multiplied by
/// `scale`. Ties go to the lower expert id. One threadgroup a token; each pick
/// is a threadgroup argmax, so a pick costs two barriers rather than a pass of
/// one thread over every expert.
///
/// Threadgroup memory: `2 * experts` floats.
///
/// Buffers: args, ids `[n_tok * used]` int32, weights `[n_tok * used]`,
/// logits `[n_tok, experts]`, bias `[experts]`.
kernel void kernel_dense_moe_route_sigmoid_f32(
        constant dense_args_moe_route & args [[buffer(0)]],
        device int * ids [[buffer(1)]],
        device float * weights [[buffer(2)]],
        device const float * logits [[buffer(3)]],
        device const float * bias [[buffer(4)]],
        threadgroup float * tbuf [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint n_sg [[simdgroups_per_threadgroup]]) {
    threadgroup float best_v[DENSE_ROUTE_MAX_SIMDGROUPS];
    threadgroup uint best_i[DENSE_ROUTE_MAX_SIMDGROUPS];
    dense_route_token(args, ids, weights, logits, bias, tbuf, tbuf + args.experts,
                      best_v, best_i, nullptr, tgpig.x, tpitg.x, ntg.x, lane, sg, n_sg);
}

/// The router and the counting sort together, for a chunk small enough that
/// one threadgroup routes every token: the sorted column map, the expert
/// bounds and the active list the expert stages read, in one launch instead of
/// six.
///
/// The sort is stable - an expert's columns in column order - where the
/// atomic scatter's is not; nothing downstream depends on the order within an
/// expert, because the combine addresses columns by index.
///
/// Threadgroup memory: `3 * experts` words. The chosen ids are kept in a
/// threadgroup array of [`DENSE_ROUTE_SORT_COLUMNS`] as well as written out, so
/// the count and the scatter read them at threadgroup latency.
///
/// Buffers: args, ids, weights, logits, bias, sorted `[columns]`, bounds
/// `[experts + 1]`, active `[experts + 1]` (the count at `[experts]`).
kernel void kernel_dense_moe_route_sort_sigmoid_f32(
        constant dense_args_moe_route & args [[buffer(0)]],
        device int * ids [[buffer(1)]],
        device float * weights [[buffer(2)]],
        device const float * logits [[buffer(3)]],
        device const float * bias [[buffer(4)]],
        device int * sorted [[buffer(5)]],
        device int * bounds [[buffer(6)]],
        device int * active [[buffer(7)]],
        constant uint & n_tok [[buffer(8)]],
        threadgroup float * tbuf [[threadgroup(0)]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint n_sg [[simdgroups_per_threadgroup]]) {
    threadgroup float best_v[DENSE_ROUTE_MAX_SIMDGROUPS];
    threadgroup uint best_i[DENSE_ROUTE_MAX_SIMDGROUPS];
    threadgroup int tids[DENSE_ROUTE_SORT_COLUMNS];
    const uint e_n = args.experts;
    const uint tid = tpitg.x;
    for (uint t = 0; t < n_tok; ++t) {
        dense_route_token(args, ids, weights, logits, bias, tbuf, tbuf + e_n,
                          best_v, best_i, tids, t, tid, ntg.x, lane, sg, n_sg);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) {
        return;
    }
    threadgroup int * count = (threadgroup int *) (tbuf + 2u * e_n);
    const uint columns = n_tok * args.used;
    for (uint e = 0; e < e_n; ++e) {
        count[e] = 0;
    }
    for (uint c = 0; c < columns; ++c) {
        count[tids[c]] += 1;
    }
    int at = 0;
    int n_active = 0;
    for (uint e = 0; e < e_n; ++e) {
        bounds[e] = at;
        if (count[e] > 0) {
            active[n_active] = int(e);
            n_active += 1;
        }
        const int next = at + count[e];
        count[e] = at;
        at = next;
    }
    bounds[e_n] = at;
    active[e_n] = n_active;
    for (uint c = 0; c < columns; ++c) {
        const int e = tids[c];
        sorted[count[e]] = int(c);
        count[e] += 1;
    }
}

/// `out[t] = shared[t] + sum_k weights[t * used + k] * down[t * used + k]`:
/// the routed columns recombined into one contribution per token, with the
/// shared expert's row added in the same pass.
///
/// Grid: `(hidden / 256, n_tok)`.
///
/// Buffers: args, out `[n_tok, hidden]`, shared `[n_tok, hidden]`, down
/// `[columns, hidden]`, weights `[columns]`.
kernel void kernel_dense_moe_combine_add_f32(
        constant dense_args_moe_route & args [[buffer(0)]],
        device float * out [[buffer(1)]],
        device const float * shared [[buffer(2)]],
        device const float * down [[buffer(3)]],
        device const float * weights [[buffer(4)]],
        constant uint & hidden [[buffer(5)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint e = tgpig.x * ntg.x + tpitg.x;
    const uint t = tgpig.y;
    if (e >= hidden) {
        return;
    }
    float acc = 0.0f;
    for (uint k = 0; k < args.used; ++k) {
        const uint col = t * args.used + k;
        acc += weights[col] * down[(uint64_t) col * hidden + e];
    }
    out[(uint64_t) t * hidden + e] = shared[(uint64_t) t * hidden + e] + acc;
}

/// One routed stage over row-major 4-bit affine stacks.
///
/// `grid.y` indexes the ACTIVE experts through the list the sort's scan left
/// behind, `grid.x` picks `DENSE_EXPERT_ROWS * simdgroups` output rows, and the
/// threadgroup sweeps that expert's span of the sorted column map. A lane takes
/// four packed words at a time - 32 contiguous inputs, half of one group - and
/// dots them against every row the simdgroup owns, so the input is read once
/// a simdgroup and each row's words arrive as one 16-byte load.
///
/// Buffers: args, words, scales, biases, input, sorted, bounds, active, out.
kernel void kernel_dense_experts_affine4_f32(
        constant dense_args_experts & args [[buffer(0)]],
        device const uint32_t * w [[buffer(1)]],
        device const ushort * sc [[buffer(2)]],
        device const ushort * bi [[buffer(3)]],
        device const float * x [[buffer(4)]],
        device const int * sorted [[buffer(5)]],
        device const int * bounds [[buffer(6)]],
        device const int * active [[buffer(7)]],
        device float * out [[buffer(8)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint n_sg [[simdgroups_per_threadgroup]]) {
    constexpr uint R = DENSE_EXPERT_ROWS;
    if (int(tgpig.y) >= active[args.n_expert]) {
        return;
    }
    const uint expert = uint(active[tgpig.y]);
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1u];
    const uint in_dim = args.in_dim;
    const uint out_dim = args.out_dim;
    const uint chunks = in_dim / 32u;
    const uint groups = in_dim / 64u;
    const uint row0 = (tgpig.x * n_sg + sg) * R;
    if (row0 >= out_dim) {
        return;
    }
    device const uint4 * wr[R];
    device const ushort * sr[R];
    device const ushort * br[R];
    for (uint r = 0; r < R; ++r) {
        const uint64_t at = (uint64_t) expert * out_dim + min(row0 + r, out_dim - 1u);
        wr[r] = (device const uint4 *) (w + at * (in_dim / 8u));
        sr[r] = sc + at * groups;
        br[r] = bi + at * groups;
    }
    for (int c = c0; c < c1; ++c) {
        const uint col = uint(sorted[c]);
        const uint xr = args.by_token != 0u ? col / args.n_used : col;
        device const float4 * xrow = (device const float4 *) (x + (uint64_t) xr * in_dim);
        float acc[R] = {};
        for (uint i = lane; i < chunks; i += 32u) {
            float4 xv[8];
            float sx = 0.0f;
            for (uint q = 0; q < 8u; ++q) {
                xv[q] = xrow[i * 8u + q];
                sx += (xv[q].x + xv[q].y) + (xv[q].z + xv[q].w);
            }
            const uint g = i / 2u;
            for (uint r = 0; r < R; ++r) {
                const uint4 word = wr[r][i];
                float qx = 0.0f;
                for (uint q = 0; q < 4u; ++q) {
                    const uint wd = word[q];
                    const float4 lo = xv[2u * q];
                    const float4 hi = xv[2u * q + 1u];
                    qx += float(wd & 0xfu) * lo.x + float((wd >> 4u) & 0xfu) * lo.y
                        + float((wd >> 8u) & 0xfu) * lo.z + float((wd >> 12u) & 0xfu) * lo.w
                        + float((wd >> 16u) & 0xfu) * hi.x + float((wd >> 20u) & 0xfu) * hi.y
                        + float((wd >> 24u) & 0xfu) * hi.z + float(wd >> 28u) * hi.w;
                }
                acc[r] += dense_bf16(sr[r] + g) * qx + dense_bf16(br[r] + g) * sx;
            }
        }
        for (uint r = 0; r < R; ++r) {
            const float v = simd_sum(acc[r]);
            if (lane == 0u && row0 + r < out_dim) {
                out[(uint64_t) col * out_dim + row0 + r] =
                        args.act != 0u ? max(v, 0.0f) * max(v, 0.0f) : v;
            }
        }
    }
}

/// One routed stage over row-major IQ4_NL stacks, optionally through a slot
/// map.
///
/// The same grid and column walk as the affine stage. A row is `in_dim / 32`
/// blocks of 18 bytes - an f16 scale, then sixteen bytes whose low nibbles are
/// the block's first sixteen values and whose high nibbles are its last
/// sixteen, each an index into the `IQ4NL` table - so expert `e`'s row `r`
/// starts at block `(slot * out_dim + r) * in_dim / 32`, where `slot` is `e`
/// itself or, for a streamed bank, `map[layer_base + e]`. A lane takes whole
/// blocks, strided by the simdgroup width, and dots each against every row the
/// simdgroup owns, so a width of 84 blocks (2688 inputs) needs no padding.
///
/// Buffers: args, blocks, map, input, sorted, bounds, active, out.
kernel void kernel_dense_experts_iq4nl_f32(
        constant dense_args_experts & args [[buffer(0)]],
        device const uchar * w [[buffer(1)]],
        device const int * map [[buffer(2)]],
        device const float * x [[buffer(3)]],
        device const int * sorted [[buffer(4)]],
        device const int * bounds [[buffer(5)]],
        device const int * active [[buffer(6)]],
        device float * out [[buffer(7)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint n_sg [[simdgroups_per_threadgroup]]) {
    constexpr uint R = DENSE_EXPERT_ROWS;
    constexpr uint BLOCK_BYTES = 18u;
    if (int(tgpig.y) >= active[args.n_expert]) {
        return;
    }
    const uint expert = uint(active[tgpig.y]);
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1u];
    const uint in_dim = args.in_dim;
    const uint out_dim = args.out_dim;
    const uint blocks = in_dim / 32u;
    const uint row0 = (tgpig.x * n_sg + sg) * R;
    if (row0 >= out_dim) {
        return;
    }
    const uint slot = args.mapped != 0u ? uint(map[args.layer_base + expert]) : expert;
    device const uchar * wr[R];
    for (uint r = 0; r < R; ++r) {
        const uint64_t at = (uint64_t) slot * out_dim + min(row0 + r, out_dim - 1u);
        wr[r] = w + at * blocks * BLOCK_BYTES;
    }
    for (int c = c0; c < c1; ++c) {
        const uint col = uint(sorted[c]);
        const uint xr = args.by_token != 0u ? col / args.n_used : col;
        device const float4 * xrow = (device const float4 *) (x + (uint64_t) xr * in_dim);
        float acc[R] = {};
        for (uint b = lane; b < blocks; b += 32u) {
            float4 xv[8];
            for (uint q = 0; q < 8u; ++q) {
                xv[q] = xrow[b * 8u + q];
            }
            for (uint r = 0; r < R; ++r) {
                device const uchar * blk = wr[r] + b * BLOCK_BYTES;
                const float d = float(*((device const half *) blk));
                device const uchar * qs = blk + 2u;
                float s = 0.0f;
                for (uint q = 0; q < 4u; ++q) {
                    const uchar b0 = qs[4u * q];
                    const uchar b1 = qs[4u * q + 1u];
                    const uchar b2 = qs[4u * q + 2u];
                    const uchar b3 = qs[4u * q + 3u];
                    const float4 lo = xv[q];
                    const float4 hi = xv[q + 4u];
                    s += float(IQ4NL[b0 & 0xfu]) * lo.x + float(IQ4NL[b1 & 0xfu]) * lo.y
                        + float(IQ4NL[b2 & 0xfu]) * lo.z + float(IQ4NL[b3 & 0xfu]) * lo.w
                        + float(IQ4NL[b0 >> 4u]) * hi.x + float(IQ4NL[b1 >> 4u]) * hi.y
                        + float(IQ4NL[b2 >> 4u]) * hi.z + float(IQ4NL[b3 >> 4u]) * hi.w;
                }
                acc[r] += d * s;
            }
        }
        for (uint r = 0; r < R; ++r) {
            const float v = simd_sum(acc[r]);
            if (lane == 0u && row0 + r < out_dim) {
                out[(uint64_t) col * out_dim + row0 + r] =
                        args.act != 0u ? max(v, 0.0f) * max(v, 0.0f) : v;
            }
        }
    }
}

/// `out[i] = max(in[i], 0)^2` over `n` floats.
kernel void kernel_dense_relu2_f32(
        constant dense_args_flat & args [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device float * dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) {
        return;
    }
    const float v = max(src[gid], 0.0f);
    dst[gid] = v * v;
}

/// Writes keys and values into the f16 cache row the page table names, as
/// they came out of their projections: no norm and no rotation.
///
/// The key/value half of the norm-and-rope store for a slot that has neither;
/// it takes the same argument block and reads only its widths and its page
/// fields. Grid: `(n_head_kv, n_tok)`.
///
/// Buffers: args, k_cache, v_cache, k_in, v_in, pos, pages.
kernel void kernel_dense_kv_store_f32(
        constant dense_args_kv_norm_rope & args [[buffer(0)]],
        device half * k_cache [[buffer(1)]],
        device half * v_cache [[buffer(2)]],
        device const float * k_in [[buffer(3)]],
        device const float * v_in [[buffer(4)]],
        device const int32_t * pos [[buffer(5)]],
        device const int32_t * pages [[buffer(6)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    const uint64_t row = dense_kv_row(dense_row_pages(pages, t, args.page_stride),
                                      (uint32_t) pos[t], args.page_tokens);
    const uint64_t src = ((uint64_t) t * args.n_head_kv + h) * head_dim;
    const uint64_t slot = ((uint64_t) row * args.n_head_kv + h) * head_dim;
    for (uint i = tpitg.x; i < head_dim; i += ntg.x) {
        k_cache[slot + i] = half(k_in[src + i]);
        v_cache[slot + i] = half(v_in[src + i]);
    }
}

// q4e FP8 entries: the sparse stack's kernels over a checkpoint whose routed
// experts are FP8 E4M3 with one brain-float scale per 128 x 128 block, whose
// n-gram rows are FP8 E4M3 under one scale for the whole table, and whose
// vocabulary table is brain float.
//
// An E4M3 byte is decoded through a 256-entry table each threadgroup builds in
// threadgroup memory before it reads a weight: one value per thread of a
// 256-thread group, exact, and cheaper than the bit arithmetic per weight.
//
// `q4e_common.h` is concatenated ahead of this file. No fp math pragma.

/// Side of one scale block of a block-scaled weight.
constant constexpr uint Q4E_FP8_BLOCK = 128u;
/// Output rows one expert threadgroup covers, one per simdgroup.
constant constexpr uint Q4E_FP8_ROWS = 8u;

/// One E4M3 byte as a float: 1 sign, 4 exponent (bias 7), 3 mantissa bits,
/// no infinities, the all-ones pattern NaN.
static inline float q4e_e4m3(uint b) {
    const uint e = (b >> 3u) & 0xFu;
    const uint m = b & 7u;
    float v;
    if (e == 0u) {
        v = float(m) * 0.001953125f;  // m * 2^-9
    } else if (e == 15u && m == 7u) {
        v = NAN;
    } else {
        v = (1.0f + float(m) * 0.125f) * exp2(float(e) - 7.0f);
    }
    return (b & 0x80u) ? -v : v;
}

/// Fills the decode table; every thread of a 256-thread group writes one entry.
static inline void q4e_e4m3_table(threadgroup float *lut, uint tid) {
    lut[tid] = q4e_e4m3(tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// One lane's share of the dot of one block-scaled FP8 row with `x`:
/// lane `l` reads bytes `4l, 4l + 1, ...` of every 128-wide stretch, so a
/// simdgroup consumes one scale block's row segment per step.
static inline float q4e_fp8_row(device const uchar *w, device const ushort *scales,
                                device const float *x, uint in_dim,
                                threadgroup const float *lut, uint lane) {
    float acc = 0.0f;
    for (uint k0 = 0; k0 < in_dim; k0 += Q4E_FP8_BLOCK) {
        const float s = q4e_bf16(scales[k0 / Q4E_FP8_BLOCK]);
        const uint k = k0 + lane * 4u;
        if (k + 3u < in_dim) {
            const uchar4 q = *((device const uchar4 *)(w + k));
            const float4 xv = *((device const float4 *)(x + k));
            const float part = lut[q.x] * xv.x + lut[q.y] * xv.y + lut[q.z] * xv.z +
                               lut[q.w] * xv.w;
            acc += s * part;
        }
    }
    return acc;
}

/// Gate and up for one active expert's columns: eight output rows a
/// threadgroup, one per simdgroup, the expert's span of the sorted column list
/// swept so its rows stay in cache. `map[e]` is the slot expert `e`'s matrices
/// sit at; a column's activation row is its token's, `col / n_used`.
///
/// Buffers: out_gate, out_up, gate_w, gate_s, up_w, up_s, x, sorted, bounds,
/// active, map, in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8),
/// min(columns, n_expert))`, 256 threads.
kernel void q4e_fp8_experts_gate_up_kernel(
        device float *out_gate [[buffer(0)]],
        device float *out_up [[buffer(1)]],
        device const uchar *gate_w [[buffer(2)]],
        device const ushort *gate_s [[buffer(3)]],
        device const uchar *up_w [[buffer(4)]],
        device const ushort *up_s [[buffer(5)]],
        device const float *x [[buffer(6)]],
        device const int *sorted [[buffer(7)]],
        device const int *bounds [[buffer(8)]],
        device const int *active [[buffer(9)]],
        device const uint *map [[buffer(10)]],
        constant uint &in_dim [[buffer(11)]],
        constant uint &out_dim [[buffer(12)]],
        constant uint &n_expert [[buffer(13)]],
        constant uint &n_used [[buffer(14)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lut[256];
    q4e_e4m3_table(lut, tid);
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint o = group.x * Q4E_FP8_ROWS + simd;
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const ulong slot = map[expert];
    const uint blocks = (in_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint block_rows = (out_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const ulong row = slot * out_dim + o;
    const ulong srow = (slot * block_rows + o / Q4E_FP8_BLOCK) * blocks;
    device const uchar *gw = gate_w + row * in_dim;
    device const uchar *uw = up_w + row * in_dim;
    for (int c = c0; c < c1; ++c) {
        const uint col = uint(sorted[c]);
        device const float *xr = x + ulong(col / n_used) * in_dim;
        const float g = q4e_warp_sum(q4e_fp8_row(gw, gate_s + srow, xr, in_dim, lut, lane));
        const float u = q4e_warp_sum(q4e_fp8_row(uw, up_s + srow, xr, in_dim, lut, lane));
        if (lane == 0u) {
            out_gate[ulong(col) * out_dim + o] = g;
            out_up[ulong(col) * out_dim + o] = u;
        }
    }
}

/// The routed down projection over a block-scaled FP8 bank: each column reads
/// its own row of the activated intermediate and writes `[column, out_dim]`.
///
/// Buffers: out, down_w, down_s, mid, sorted, bounds, active, map, in_dim,
/// out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8), min(columns,
/// n_expert))`, 256 threads.
kernel void q4e_fp8_experts_down_kernel(
        device float *out [[buffer(0)]],
        device const uchar *down_w [[buffer(1)]],
        device const ushort *down_s [[buffer(2)]],
        device const float *mid [[buffer(3)]],
        device const int *sorted [[buffer(4)]],
        device const int *bounds [[buffer(5)]],
        device const int *active [[buffer(6)]],
        device const uint *map [[buffer(7)]],
        constant uint &in_dim [[buffer(8)]],
        constant uint &out_dim [[buffer(9)]],
        constant uint &n_expert [[buffer(10)]],
        constant uint &n_used [[buffer(11)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lut[256];
    q4e_e4m3_table(lut, tid);
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint o = group.x * Q4E_FP8_ROWS + simd;
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const ulong slot = map[expert];
    const uint blocks = (in_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint block_rows = (out_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    device const uchar *dw = down_w + (slot * out_dim + o) * in_dim;
    const ulong srow = (slot * block_rows + o / Q4E_FP8_BLOCK) * blocks;
    for (int c = c0; c < c1; ++c) {
        const uint col = uint(sorted[c]);
        const float v = q4e_warp_sum(
            q4e_fp8_row(dw, down_s + srow, mid + ulong(col) * in_dim, in_dim, lut, lane));
        if (lane == 0u) {
            out[ulong(col) * out_dim + o] = v;
        }
    }
    (void)n_used;
}

/// Decodes gathered FP8 n-gram rows: `head_dim` E4M3 bytes a row, one scale
/// for the whole table. Row `r` lands at token `r / n_heads`, head slot
/// `r % n_heads`, head-slowest - `q4e_ple_dequant_kernel`'s placement.
///
/// Buffers: out, rows, head_dim, n_heads, row_bytes, n_rows, scale (one float
/// on the device). Grid
/// `n_rows`, `head_dim` threads.
kernel void q4e_ple_dequant_fp8_kernel(
        device float *out [[buffer(0)]],
        device const uchar *rows [[buffer(1)]],
        constant uint &head_dim [[buffer(2)]],
        constant uint &n_heads [[buffer(3)]],
        constant uint &row_bytes [[buffer(4)]],
        constant uint &n_rows [[buffer(5)]],
        device const float *scale [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 tid [[thread_position_in_threadgroup]],
        uint3 tpg [[threads_per_threadgroup]]) {
    const uint r = group.x;
    if (r >= n_rows) {
        return;
    }
    device const uchar *src = rows + ulong(r) * row_bytes;
    const uint t = r / n_heads;
    const uint h = r % n_heads;
    device float *dst = out + ulong(t) * n_heads * head_dim + ulong(h) * head_dim;
    for (uint i = tid.x; i < head_dim; i += tpg.x) {
        dst[i] = q4e_e4m3(uint(src[i])) * scale[0];
    }
}

/// `embed_tokens_bf16_kernel`: one brain-float vocabulary row per token,
/// widened exactly; an id outside the vocabulary writes zeros.
///
/// Buffers: out, weights, tokens, n_tokens, n_embd, n_vocab. Grid
/// `ceil(n_tokens * n_embd / 256)`, 256 threads.
kernel void embed_tokens_bf16_kernel(
        device float *out [[buffer(0)]],
        device const ushort *weights [[buffer(1)]],
        device const int *tokens [[buffer(2)]],
        constant uint &n_tokens [[buffer(3)]],
        constant uint &n_embd [[buffer(4)]],
        constant uint &n_vocab [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong n = ulong(n_tokens) * n_embd;
    if (ulong(gid) >= n) {
        return;
    }
    const uint row = gid / n_embd;
    const uint col = gid - row * n_embd;
    const int token = tokens[row];
    if (token < 0 || uint(token) >= n_vocab) {
        out[gid] = 0.0f;
        return;
    }
    out[gid] = q4e_bf16(weights[ulong(uint(token)) * n_embd + col]);
}

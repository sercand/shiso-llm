// q4e affine entries: the sparse stack's kernels over MLX packed affine
// weights, which have no CUDA twin.
//
// The layout is MLX's, read in place from the checkpoint: a matrix of
// `out_dim` rows is three arrays - `out_dim x in_dim / 8` little-endian `u32`
// words of eight 4-bit codes each (lowest nibble first), and `out_dim x
// in_dim / G` brain-float scales and biases - and element `k` of row `r` is
// `code * scale[r][k / G] + bias[r][k / G]`. A routed expert bank stacks
// `E` such matrices per array.
//
// Every projection sums with the factorisation MLX itself uses,
// `scale * sum(q * x) + bias * sum(x)` per group, then folds the group sums
// across the row in word order inside each lane and across lanes with the
// decreasing-offset fold. One lane reads one whole word per step, so a
// simdgroup consumes 128 contiguous bytes of a row at a time.
//
// `q4e_common.h` is concatenated ahead of this file. No fp math pragma.

/// Codes per affine group of the projections here (MLX `group_size`).
constant constexpr uint Q4E_AFFINE_GROUP = 64u;
/// Output rows one projection threadgroup covers, one per simdgroup.
constant constexpr uint Q4E_AFFINE_ROWS = 8u;
/// Activation rows one multi-row projection threadgroup carries.
constant constexpr uint Q4E_AFFINE_TOKENS = 8u;

/// The dot of one packed row with `T` activation rows, one lane's share.
///
/// `acc[t]` receives lane-partial sums; the caller folds them. The words a
/// lane reads are `lane, lane + 32, ...`, so the order a row is summed in
/// depends on the shape alone and never on how many rows share the pass.
template <uint T>
static inline void q4e_affine_row(device const uint *w, device const ushort *s,
                                  device const ushort *b, device const float *x,
                                  uint in_dim, uint rows_here, uint group,
                                  uint lane, thread float (&acc)[T]) {
    const uint words = in_dim / 8u;
    const uint per_group = group / 8u;
    for (uint t = 0; t < T; ++t) {
        acc[t] = 0.0f;
    }
    for (uint wi = lane; wi < words; wi += 32u) {
        const uint word = w[wi];
        const uint g = wi / per_group;
        const float scale = q4e_bf16(s[g]);
        const float bias = q4e_bf16(b[g]);
        float q[8];
        for (uint j = 0; j < 8u; ++j) {
            q[j] = float((word >> (4u * j)) & 0xFu);
        }
        const uint k = wi * 8u;
        for (uint t = 0; t < T; ++t) {
            if (t < rows_here) {
                device const float4 *xv = (device const float4 *)(x + ulong(t) * in_dim + k);
                const float4 x0 = xv[0];
                const float4 x1 = xv[1];
                const float qx = q[0] * x0.x + q[1] * x0.y + q[2] * x0.z + q[3] * x0.w +
                                 q[4] * x1.x + q[5] * x1.y + q[6] * x1.z + q[7] * x1.w;
                const float sx = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w);
                acc[t] += scale * qx + bias * sx;
            }
        }
    }
}

/// `out[r][o] = sum_k W[o][k] x[r][k]` over `rows` activation rows.
///
/// Grid `(ceil(out_dim / 8), ceil(rows / T))`, 256 threads: simdgroup `i` owns
/// output row `8 * group.x + i` for the `T` activation rows starting at
/// `T * group.y`.
///
/// Buffers: out, w, scales, biases, x, in_dim, out_dim, rows.
template <uint T>
static inline void q4e_affine_project(device float *out, device const uint *w,
                                      device const ushort *scales, device const ushort *biases,
                                      device const float *x, uint in_dim, uint out_dim,
                                      uint rows, uint3 group, uint lane, uint simd) {
    const uint o = group.x * Q4E_AFFINE_ROWS + simd;
    const uint r0 = group.y * T;
    if (o >= out_dim || r0 >= rows) {
        return;
    }
    const uint rows_here = min(T, rows - r0);
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    float acc[T];
    q4e_affine_row<T>(w + ulong(o) * (in_dim / 8u), scales + ulong(o) * groups,
                      biases + ulong(o) * groups, x + ulong(r0) * in_dim, in_dim,
                      rows_here, Q4E_AFFINE_GROUP, lane, acc);
    for (uint t = 0; t < T; ++t) {
        const float v = q4e_warp_sum(acc[t]);
        if (lane == 0u && t < rows_here) {
            out[ulong(r0 + t) * out_dim + o] = v;
        }
    }
}

/// One activation row: a decode step's projections.
kernel void q4e_affine_mv_kernel(
        device float *out [[buffer(0)]],
        device const uint *w [[buffer(1)]],
        device const ushort *scales [[buffer(2)]],
        device const ushort *biases [[buffer(3)]],
        device const float *x [[buffer(4)]],
        constant uint &in_dim [[buffer(5)]],
        constant uint &out_dim [[buffer(6)]],
        constant uint &rows [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    q4e_affine_project<1>(out, w, scales, biases, x, in_dim, out_dim, rows, group, lane, simd);
}

/// Several activation rows: a prefill chunk's projections, reading each
/// packed word once for `Q4E_AFFINE_TOKENS` rows.
kernel void q4e_affine_mm_kernel(
        device float *out [[buffer(0)]],
        device const uint *w [[buffer(1)]],
        device const ushort *scales [[buffer(2)]],
        device const ushort *biases [[buffer(3)]],
        device const float *x [[buffer(4)]],
        constant uint &in_dim [[buffer(5)]],
        constant uint &out_dim [[buffer(6)]],
        constant uint &rows [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    q4e_affine_project<Q4E_AFFINE_TOKENS>(out, w, scales, biases, x, in_dim, out_dim, rows,
                                          group, lane, simd);
}

/// The row gather that opens a pass, over an affine vocabulary table:
/// `embed_tokens_hc_kernel`'s contract with the table dequantized in place.
///
/// Buffers: out, tokens, w, scales, biases, n_vocab, n_tokens, n_embd, n_hc.
/// Grid `ceil(n_tokens * n_hc * n_embd / 256)`, 256 threads.
kernel void q4e_affine_embed_hc_kernel(
        device float *out [[buffer(0)]],
        device const int *tokens [[buffer(1)]],
        device const uint *w [[buffer(2)]],
        device const ushort *scales [[buffer(3)]],
        device const ushort *biases [[buffer(4)]],
        constant uint &n_vocab [[buffer(5)]],
        constant uint &n_tokens [[buffer(6)]],
        constant uint &n_embd [[buffer(7)]],
        constant uint &n_hc [[buffer(8)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong n = ulong(n_tokens) * n_hc * n_embd;
    if (ulong(gid) >= n) {
        return;
    }
    const uint d = gid % n_embd;
    const uint t = (gid / n_embd) / n_hc;
    const int tok_i = tokens[t];
    uint tok = tok_i < 0 ? 0u : uint(tok_i);
    if (tok >= n_vocab) {
        tok = 0u;
    }
    const uint groups = n_embd / Q4E_AFFINE_GROUP;
    const uint word = w[ulong(tok) * (n_embd / 8u) + d / 8u];
    const float q = float((word >> (4u * (d % 8u))) & 0xFu);
    const ulong p = ulong(tok) * groups + d / Q4E_AFFINE_GROUP;
    out[gid] = q * q4e_bf16(scales[p]) + q4e_bf16(biases[p]);
}

/// Gate and up for one active expert's columns, eight output rows a
/// threadgroup, one per simdgroup; the expert's span of the sorted column list
/// is swept so its rows stay in cache across the columns routed to it.
///
/// `map[e]` is the slot expert `e`'s matrices sit at inside the three banks:
/// the identity for a bank that is wholly resident, and the slot the stream
/// placed it in for one that is streamed. A column's activation row is its
/// token's: `col / n_used`.
///
/// Buffers: out_gate, out_up, gate_w, gate_s, gate_b, up_w, up_s, up_b, x,
/// sorted, bounds, active, map, in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8),
/// min(columns, n_expert))`, 256 threads.
kernel void q4e_affine_experts_gate_up_kernel(
        device float *out_gate [[buffer(0)]],
        device float *out_up [[buffer(1)]],
        device const uint *gate_w [[buffer(2)]],
        device const ushort *gate_s [[buffer(3)]],
        device const ushort *gate_b [[buffer(4)]],
        device const uint *up_w [[buffer(5)]],
        device const ushort *up_s [[buffer(6)]],
        device const ushort *up_b [[buffer(7)]],
        device const float *x [[buffer(8)]],
        device const int *sorted [[buffer(9)]],
        device const int *bounds [[buffer(10)]],
        device const int *active [[buffer(11)]],
        device const uint *map [[buffer(12)]],
        constant uint &in_dim [[buffer(13)]],
        constant uint &out_dim [[buffer(14)]],
        constant uint &n_expert [[buffer(15)]],
        constant uint &n_used [[buffer(16)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint o = group.x * Q4E_AFFINE_ROWS + simd;
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    const ulong slot = map[expert];
    const ulong row = slot * out_dim + o;
    device const uint *gw = gate_w + row * (in_dim / 8u);
    device const ushort *gs = gate_s + row * groups;
    device const ushort *gb = gate_b + row * groups;
    device const uint *uw = up_w + row * (in_dim / 8u);
    device const ushort *us = up_s + row * groups;
    device const ushort *ub = up_b + row * groups;
    for (int c = c0; c < c1; ++c) {
        const uint col = uint(sorted[c]);
        device const float *xr = x + ulong(col / n_used) * in_dim;
        float g[1];
        float u[1];
        q4e_affine_row<1>(gw, gs, gb, xr, in_dim, 1u, Q4E_AFFINE_GROUP, lane, g);
        q4e_affine_row<1>(uw, us, ub, xr, in_dim, 1u, Q4E_AFFINE_GROUP, lane, u);
        const float gv = q4e_warp_sum(g[0]);
        const float uv = q4e_warp_sum(u[0]);
        if (lane == 0u) {
            out_gate[ulong(col) * out_dim + o] = gv;
            out_up[ulong(col) * out_dim + o] = uv;
        }
    }
}

/// The routed down projection: each column reads its OWN row of the
/// activated intermediate and writes `[column, out_dim]` row-major, which is
/// what `q4e_moe_combine_kernel` reads.
///
/// Buffers: out, down_w, down_s, down_b, mid, sorted, bounds, active, map,
/// in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8), min(columns, n_expert))`, 256 threads.
kernel void q4e_affine_experts_down_kernel(
        device float *out [[buffer(0)]],
        device const uint *down_w [[buffer(1)]],
        device const ushort *down_s [[buffer(2)]],
        device const ushort *down_b [[buffer(3)]],
        device const float *mid [[buffer(4)]],
        device const int *sorted [[buffer(5)]],
        device const int *bounds [[buffer(6)]],
        device const int *active [[buffer(7)]],
        device const uint *map [[buffer(8)]],
        constant uint &in_dim [[buffer(9)]],
        constant uint &out_dim [[buffer(10)]],
        constant uint &n_expert [[buffer(11)]],
        constant uint &n_used [[buffer(12)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint o = group.x * Q4E_AFFINE_ROWS + simd;
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    const ulong row = ulong(map[expert]) * out_dim + o;
    device const uint *dw = down_w + row * (in_dim / 8u);
    device const ushort *ds = down_s + row * groups;
    device const ushort *db = down_b + row * groups;
    for (int c = c0; c < c1; ++c) {
        const uint col = uint(sorted[c]);
        float d[1];
        q4e_affine_row<1>(dw, ds, db, mid + ulong(col) * in_dim, in_dim, 1u, Q4E_AFFINE_GROUP,
                          lane, d);
        const float v = q4e_warp_sum(d[0]);
        if (lane == 0u) {
            out[ulong(col) * out_dim + o] = v;
        }
    }
}

/// Decodes gathered n-gram rows stored as MLX affine 4-bit, group 32.
///
/// A gathered row is the three arrays' slices of one table row, concatenated
/// by the reader: `head_dim / 2` bytes of codes, then `head_dim / 32`
/// brain-float scales, then as many biases. Row `r` belongs to token
/// `r / n_heads` and lands at head slot `r % n_heads`, head-slowest inside the
/// token's `n_heads * head_dim` wide result - `q4e_ple_dequant_kernel`'s
/// placement.
///
/// Buffers: out, rows, head_dim, n_heads, row_bytes, n_rows. Grid `n_rows`,
/// one thread per element (the host launches `head_dim` threads).
kernel void q4e_ple_dequant_affine_kernel(
        device float *out [[buffer(0)]],
        device const uchar *rows [[buffer(1)]],
        constant uint &head_dim [[buffer(2)]],
        constant uint &n_heads [[buffer(3)]],
        constant uint &row_bytes [[buffer(4)]],
        constant uint &n_rows [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 tid [[thread_position_in_threadgroup]],
        uint3 tpg [[threads_per_threadgroup]]) {
    const uint r = group.x;
    if (r >= n_rows) {
        return;
    }
    device const uchar *src = rows + ulong(r) * row_bytes;
    device const ushort *params = (device const ushort *)(src + head_dim / 2u);
    const uint groups = head_dim / 32u;
    const uint t = r / n_heads;
    const uint h = r % n_heads;
    device float *dst = out + ulong(t) * n_heads * head_dim + ulong(h) * head_dim;
    for (uint i = tid.x; i < head_dim; i += tpg.x) {
        const uchar byte = src[i / 2u];
        const float q = float((i & 1u) ? (byte >> 4u) : (byte & 0xFu));
        const uint g = i / 32u;
        dst[i] = q * q4e_bf16(params[g]) + q4e_bf16(params[groups + g]);
    }
}

// The per-layer embedding pair: one row of a table too large to hold, turned
// into the vector each layer slot multiplies its gate by.
//
// Gemma 4 E4B carries one q6_K block per layer slot per token id, so a token's
// whole row is `n_layers` blocks laid end to end.  The table is streamed off
// the file a row at a time and staged in front of the chunk, or held resident
// for a caller that can afford it; the only difference the kernel sees is
// whether it indexes `rows` by the token id or by the token's own position in
// the chunk.
//
// The q6_K decoder is `dequantize_dense_q6_K` from
// metal/qwen35/gemv_ext_q5k_q6k.metal, which this library compiles ahead of the
// dense set.  It is REUSED rather than copied: a second reading of the same
// packing would be a second thing to get wrong, and both would produce finite
// weights.

/// Normalizes the model's per-layer projection, dequantizes the token's row and
/// mixes the two.
///
/// One threadgroup per (layer slot, token), dispatched as `(n_layers, n_tok)`.
/// Per slot:
///
///   p_i   = proj[t, l*dim + i] * proj_scale
///   p_i  <- p_i * rsqrt(sum_i(p_i^2) / dim + eps) * norm_w[i]
///   e_i   = dequant(rows + row*row_bytes + l*sizeof(block))[i] * embed_scale
///   out_i = (p_i + e_i) * mix_scale
///
/// where `row` is `ids[t]` for a resident table and `t` for a streamed one.
///
/// The sum of squares goes through `dense_block_sum` rather than the bare
/// builtin, for the reason every reduction in this set does: the builtin's
/// summation order is unspecified, and a golden that pins a decode step pins the
/// order its partials were added in.
///
/// The decoded row is staged in threadgroup memory rather than dequantized per
/// element: one `dequantize_dense_q6_K` call produces sixteen weights, so a
/// per-element decode would do sixteen times the unpacking work and throw
/// fifteen sixteenths of it away.  `tbuf` therefore holds the block sum's slots
/// first and the staged row after them.
kernel void kernel_dense_ple_prepare_q6_K(
        constant dense_args_ple & args,
        device       float   * out,
        device const float   * proj,
        device const float   * norm_w,
        device const char    * rows,
        device const int32_t * ids,
        threadgroup float * tbuf [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg_v [[threads_per_threadgroup]]) {
    const uint l = tgpig.x;
    const uint t = tgpig.y;
    if (l >= args.n_layers || t >= args.n_tok) return;
    const uint tpitg_x = (uint) tpitg.x;
    const uint ntg_x = (uint) ntg_v.x;

    const uint dim = args.dim;
    threadgroup float * emb = tbuf + DENSE_BLOCK_SUM_SLOTS;

    // The token's row, then this slot's block inside it.  The block pointer is
    // stepped by `l` rather than by a byte count, so the stride is the struct's
    // own 210 bytes and cannot drift from the layout the decoder reads.
    const uint64_t row = (args.resident != 0u) ? (uint64_t) ids[t] : (uint64_t) t;
    device const shs_dense_block_q6_K * blk =
        (device const shs_dense_block_q6_K *) (rows + row * (uint64_t) args.row_bytes) + l;

    for (uint g = tpitg_x; g < dim / DENSE_PLE_GROUP; g += ntg_x) {
        float4x4 reg;
        dequantize_dense_q6_K(blk, (short) g, reg);
        for (uint j = 0u; j < DENSE_PLE_GROUP; ++j) {
            emb[g * DENSE_PLE_GROUP + j] = reg[j / 4u][j % 4u];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float * p = proj + ((uint64_t) t * args.n_layers + l) * dim;
    device       float * dst = out + ((uint64_t) t * args.n_layers + l) * dim;

    float acc = 0.0f;
    for (uint i = tpitg_x; i < dim; i += ntg_x) {
        const float v = p[i] * args.proj_scale;
        acc = dense_fma(v, v, acc);
    }
    const float total = dense_block_sum(acc, tbuf, tpitg_x, ntg_x);
    const float inv = rsqrt(total / (float) dim + args.eps);

    for (uint i = tpitg_x; i < dim; i += ntg_x) {
        const float pv = p[i] * args.proj_scale * inv * norm_w[i];
        const float ev = emb[i] * args.embed_scale;
        dst[i] = (pv + ev) * args.mix_scale;
    }
}

/// `gate[t, i] = dense_gelu_tanh(gate[t, i]) * ple[t, layer, i]`, in place.
///
/// Flat over the `n_tok * dim` gate: the slot is a field of the argument block
/// rather than a grid dimension, because one layer of the loop runs one of these
/// and the gate it multiplies is the only row in flight.
///
/// In place because the gate is this launch's only reader as well as its only
/// writer, and a destination of its own would be a buffer nothing else looks at.
kernel void kernel_dense_ple_gate_mul(
        constant dense_args_ple_gate & args,
        device       float * gate,
        device const float * ple,
        uint i [[thread_position_in_grid]]) {
    const uint n = args.n_tok * args.dim;
    if (i >= n) return;
    const uint t = i / args.dim;
    const uint d = i - t * args.dim;
    const uint64_t at = ((uint64_t) t * args.n_layers + args.layer) * args.dim + d;
    gate[i] = dense_gelu_tanh(gate[i]) * ple[at];
}

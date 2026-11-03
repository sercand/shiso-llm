// The vision tower's dense kernels: the fused attention-plane normalization and
// two-axis rotation, the pooler that folds a patch block into one token, and the
// elementwise clamp every clamped linear projection is built from.
//
// These live in the `dense` set rather than the `vision` set because they are
// written against argument blocks and not against one checkpoint's widths, and
// because the tower they serve reuses the `dense` set's reduction helpers and
// its neox rotation. The Qwen ViT has its own `vision`-set kernels; this is the
// Gemma 4 tower's, and the two differ in exactly the ways a second family
// would: a rotation over two spatial axes instead of the patch's grid position,
// a pooling fold instead of a spatial merge, and a clamp around every
// projection.
//
// There is no upstream shading-language source for this tower either. It is
// written against the reference graph in
// third_party/llama.cpp/tools/mtmd/models/gemma4v.cpp and the naive f32 oracle
// beside it in crates/shiso-metal/tests/dense_vision.rs, which is the only thing
// that separates an ordering mistake from a tower that answers confidently about
// a picture it never saw.

/// Per-head RMSNorm of the query, key and value planes and the two-axis rotation
/// of the query and the key, in place on the three planes.
///
/// One threadgroup per (head, patch row). Each plane's head is staged in
/// threadgroup memory because the rotation pairs a channel with one half a
/// quarter-head away, which no single lane owns.
///
/// The rotation is the reference's two `ggml_rope_ext` calls, each over half the
/// head: the leading half (`[0, head_dim/2)`) rotates by the patch's column and
/// the trailing half (`[head_dim/2, head_dim)`) by its row, both pairing `(j,
/// j + head_dim/4)` at the angle `pos * base^(-2j / (head_dim/2))`. A checkpoint
/// that calls this a "partial" rotation means how the ladder is built, not that
/// any channel is left alone: every channel rotates, half by one axis and half
/// by the other.
///
/// The value plane is normalized with no weight and never rotated -- the same
/// mean square the key takes, which is the whole of the difference between the
/// two, as in `kernel_dense_kv_norm_rope_store`.
kernel void kernel_dense_vit_qkv_norm_rope2d(
        constant dense_args_vit_rope2d & args [[buffer(0)]],
        device float *q                    [[buffer(1)]],
        device float *k                    [[buffer(2)]],
        device float *v                    [[buffer(3)]],
        device const float *q_norm_w       [[buffer(4)]],
        device const float *k_norm_w       [[buffer(5)]],
        threadgroup float *tbuf            [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint n = tgpig.y;
    const uint h = tgpig.x;
    if (n >= args.rows) {
        return;
    }
    const uint hd = args.head_dim;
    const uint dim = args.heads * hd;
    const uint tid = tpitg.x;
    const uint nt = ntg.x;
    const uint64_t base = (uint64_t) n * dim + (uint64_t) h * hd;

    // A still is one plane, so the patch's grid position is its row within the
    // whole item: column is the fast axis and row the slow one.
    const float col = (float) (n % args.grid_w);
    const float row = (float) (n / args.grid_w);

    threadgroup float *scratch = tbuf;
    threadgroup float *s_q = scratch + DENSE_BLOCK_SUM_SLOTS;
    threadgroup float *s_k = s_q + hd;
    threadgroup float *s_v = s_k + hd;

    float qsum = 0.0f;
    float ksum = 0.0f;
    float vsum = 0.0f;
    for (uint i = tid; i < hd; i += nt) {
        const float qv = q[base + i];
        const float kv = k[base + i];
        const float vv = v[base + i];
        s_q[i] = qv;
        s_k[i] = kv;
        s_v[i] = vv;
        qsum += qv * qv;
        ksum += kv * kv;
        vsum += vv * vv;
    }
    // The weight multiplies the normalized value directly: this tower stores the
    // weight itself, not a deviation from one.
    const float qinv = rsqrt(dense_block_sum(qsum, scratch, tid, nt) / (float) hd + args.eps);
    const float kinv = rsqrt(dense_block_sum(ksum, scratch, tid, nt) / (float) hd + args.eps);
    for (uint i = tid; i < hd; i += nt) {
        s_q[i] *= qinv * q_norm_w[i];
        s_k[i] *= kinv * k_norm_w[i];
    }
    // The value carries no weight: it is divided by its own root mean square and
    // nothing else.
    const float vinv = rsqrt(dense_block_sum(vsum, scratch, tid, nt) / (float) hd + args.eps);
    for (uint i = tid; i < hd; i += nt) {
        s_v[i] *= vinv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Two neox rotations over the two halves of the head, each over a ladder
    // built for half the head's width. The offset slice for the trailing half is
    // a pointer into the same staged row; the helper only ever touches the
    // `n_rot` dimensions it is handed.
    const uint half_hd = hd / 2u;
    dense_rope_pairs(s_q, half_hd, half_hd, args.rope_base, col, tid, nt);
    dense_rope_pairs(s_q + half_hd, half_hd, half_hd, args.rope_base, row, tid, nt);
    dense_rope_pairs(s_k, half_hd, half_hd, args.rope_base, col, tid, nt);
    dense_rope_pairs(s_k + half_hd, half_hd, half_hd, args.rope_base, row, tid, nt);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < hd; i += nt) {
        q[base + i] = s_q[i];
        k[base + i] = s_k[i];
        v[base + i] = s_v[i];
    }
}

/// A `3 x 3` average pool over the patch grid, scaled by `args.scale`.
///
/// One output row folds a `3 x 3` block of patch rows into one row of the same
/// width, taking the mean of the nine and then multiplying by the scale. The
/// reference reaches the same number by `ggml_pool_2d(AVG, 3, 3, ...)` followed
/// by `ggml_scale(sqrt(n_embd))`; the sum is accumulated in the same row-major
/// order (outer row, inner column) so the division by nine and the scale land on
/// the same f32 the reference produces.
///
/// Both grid extents are whole multiples of three by the resize rule, so the
/// nine-source index never runs off the grid and no padding is needed.
kernel void kernel_dense_vit_avgpool3_scale(
        constant dense_args_vit_pool & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *in               [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    const uint hidden = args.hidden;
    const uint out_w = args.grid_w / 3u;
    const uint total = (args.grid_h / 3u) * out_w * hidden;
    if (i >= total) {
        return;
    }
    const uint orow = i / hidden;
    const uint c = i % hidden;
    const uint oy = orow / out_w;
    const uint ox = orow % out_w;
    float sum = 0.0f;
    for (uint dy = 0u; dy < 3u; dy++) {
        for (uint dx = 0u; dx < 3u; dx++) {
            const uint sy = oy * 3u + dy;
            const uint sx = ox * 3u + dx;
            sum += in[((uint64_t) sy * args.grid_w + sx) * hidden + c];
        }
    }
    out[i] = (sum / 9.0f) * args.scale;
}

/// `out[i] = clamp(in[i], args.min, args.max)` over `args.count` elements.
///
/// Elementwise and out-of-place, so the same buffer may be given as both `out`
/// and `in` to clamp a projection's result in place: each element is read once
/// and written once by the same thread, so an aliasing pair cannot race.
kernel void kernel_dense_clamp_f32(
        constant dense_args_clamp & args [[buffer(0)]],
        device float *out                 [[buffer(1)]],
        device const float *in            [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    out[i] = x < args.min ? args.min : (x > args.max ? args.max : x);
}

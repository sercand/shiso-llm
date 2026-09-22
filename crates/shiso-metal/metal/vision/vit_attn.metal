// The tower's attention: split the fused projection, rotate it in two spatial
// axes, then attend over every patch of the same frame.
//
// Two things here are easy to get subtly wrong and impossible to see
// afterwards, so both are spelled out rather than inferred at the call site.
//
// The rotation is two-dimensional. The head's channels split in half and pair
// (i, i + head_dim/2), which is the same pairing the language model's blocks
// use; what differs is the angle. The first quarter of the channels rotate by
// the patch's ROW in the grid and the second quarter by its COLUMN, both over
// the same frequency ladder built for a width of head_dim/2. A checkpoint that
// describes this as a "partial" rotation is describing how those frequencies
// are built, not a subset of channels being left alone: every channel rotates.
//
// The attention is bidirectional and bounded by the frame. There is no causal
// mask -- a patch sees every other patch -- but a clip's frames are separate
// spans, so a query never attends across a temporal group boundary.

/// Splits the fused projection into three planes and rotates the first two.
///
/// The bias is already in `qkv`: it came out of the GEMM's epilogue, so this
/// kernel only reshapes and rotates. Values are read from the fused row rather
/// than written out and read back, which is what lets the rotation run without
/// a device-memory barrier inside the threadgroup.
kernel void kernel_vit_qkv_rope(
        constant vit_args_qkv_rope & args [[buffer(0)]],
        device float *q                   [[buffer(1)]],
        device float *k                   [[buffer(2)]],
        device float *v                   [[buffer(3)]],
        device const float *qkv           [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint n = tgpig.y;
    const uint h = tgpig.x;
    if (n >= args.rows) {
        return;
    }
    const uint nthreads = ntg.x;
    const uint hd = args.head_dim;
    const uint dim = args.heads * hd;
    const uint half_hd = hd / 2u;
    // Channels per spatial axis: the row axis owns the first quarter of the
    // head's channels and the column axis the second, and each is paired with
    // its partner half a head away.
    const uint per_axis = half_hd / 2u;

    // A row's spatial position is its offset inside its own temporal group, so
    // a clip's later frames repeat the first frame's grid.
    const uint local = n % args.plane;
    const float row = (float) (local / args.grid_w);
    const float col = (float) (local % args.grid_w);

    device const float *src = qkv + (uint64_t) n * 3u * dim;
    const uint off = h * hd;
    device float *qd = q + (uint64_t) n * dim + off;
    device float *kd = k + (uint64_t) n * dim + off;
    device float *vd = v + (uint64_t) n * dim + off;

    for (uint i = tpitg.x; i < hd; i += nthreads) {
        vd[i] = src[2u * dim + off + i];
    }
    for (uint i = tpitg.x; i < half_hd; i += nthreads) {
        const uint freq = (i < per_axis) ? i : (i - per_axis);
        const float pos = (i < per_axis) ? row : col;
        const float inv = pow(args.rope_base, -2.0f * (float) freq / (float) half_hd);
        const float theta = pos * inv;
        const float s = sin(theta);
        const float c = cos(theta);
        const float q0 = src[off + i];
        const float q1 = src[off + i + half_hd];
        qd[i] = q0 * c - q1 * s;
        qd[i + half_hd] = q1 * c + q0 * s;
        const float k0 = src[dim + off + i];
        const float k1 = src[dim + off + i + half_hd];
        kd[i] = k0 * c - k1 * s;
        kd[i + half_hd] = k1 * c + k0 * s;
    }
}

/// Bidirectional attention over one temporal group, one threadgroup per
/// (patch row, head).
///
/// Simdgroups own whole key positions and stride over the group, each keeping
/// its own running softmax maximum, denominator and value accumulator; the
/// combine at the end rescales every simdgroup's partial onto the largest
/// maximum. That shape is what keeps the exponentials bounded without a second
/// pass over the keys, and it is why a lane's accumulator has to be a register
/// array rather than threadgroup memory: it is touched once per key.
///
/// `smem` holds, in order, the query row (head_dim floats), one accumulator per
/// simdgroup (simdgroups * head_dim), then the per-simdgroup maximum and
/// denominator (one each). The host sizes it from the same formula.
kernel void kernel_vit_attention(
        constant vit_args_attention & args [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *q              [[buffer(2)]],
        device const float *k              [[buffer(3)]],
        device const float *v              [[buffer(4)]],
        threadgroup float *smem            [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint n = tgpig.y;
    const uint h = tgpig.x;
    const uint hd = args.head_dim;
    // A head wider than the per-lane accumulator can hold would be silently
    // truncated, which is the one failure this tower cannot detect downstream.
    if (n >= args.rows || hd > VIT_ATTN_SLOTS * 32u) {
        return;
    }
    const uint dim = args.heads * hd;
    const uint off = h * hd;
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    const uint warps = nthreads >> 5u;
    const float scale = args.scale;

    threadgroup float *s_q = smem;
    threadgroup float *s_acc = s_q + hd;
    threadgroup float *s_max = s_acc + (uint64_t) warps * hd;
    threadgroup float *s_den = s_max + warps;

    device const float *qd = q + (uint64_t) n * dim + off;
    for (uint i = tid; i < hd; i += nthreads) {
        s_q[i] = qd[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = (n / args.plane) * args.plane;
    const uint end = base + args.plane;

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float acc[VIT_ATTN_SLOTS];
    for (uint s = 0u; s < VIT_ATTN_SLOTS; s++) {
        acc[s] = 0.0f;
    }

    for (uint p = base + warp; p < end; p += warps) {
        device const float *kd = k + (uint64_t) p * dim + off;
        float dot = 0.0f;
        for (uint d = lane; d < hd; d += 32u) {
            dot += s_q[d] * kd[d];
        }
        dot = vit_warp_sum_all(dot);
        const float score = dot * scale;
        const float new_max = max(run_max, score);
        const float rescale = exp(run_max - new_max);
        const float weight = exp(score - new_max);
        run_den = run_den * rescale + weight;
        run_max = new_max;
        device const float *vd = v + (uint64_t) p * dim + off;
        uint slot = 0u;
        for (uint d = lane; d < hd; d += 32u, slot++) {
            acc[slot] = acc[slot] * rescale + weight * vd[d];
        }
    }

    {
        uint slot = 0u;
        for (uint d = lane; d < hd; d += 32u, slot++) {
            s_acc[(uint64_t) warp * hd + d] = acc[slot];
        }
        if (lane == 0u) {
            s_max[warp] = run_max;
            s_den[warp] = run_den;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < hd; d += nthreads) {
        float m = -INFINITY;
        for (uint w = 0u; w < warps; w++) {
            m = max(m, s_max[w]);
        }
        float den = 0.0f;
        float num = 0.0f;
        for (uint w = 0u; w < warps; w++) {
            const float r = exp(s_max[w] - m);
            den += s_den[w] * r;
            num += s_acc[(uint64_t) w * hd + d] * r;
        }
        out[(uint64_t) n * dim + off + d] = (den > 0.0f) ? (num / den) : 0.0f;
    }
}

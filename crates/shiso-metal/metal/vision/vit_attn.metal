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

/// The same attention on simdgroup matrices, for a head `HD` wide: one pass
/// over each temporal group's keys with an online softmax, everything in f32.
///
/// A threadgroup takes 64 query rows of one head - eight per simdgroup, held
/// as 8x8 fragments - inside one temporal group, and streams that group's keys
/// 32 at a time through threadgroup memory, the next block loading into
/// registers while the current one is multiplied. A lane holds two columns of
/// one row of each score fragment; the four lanes of a row (lane bits 0 and 3)
/// fold the running maximum and denominator with two shuffles, and the value
/// accumulators, laid out the same way, are rescaled lane by lane.
///
/// Queries, keys, values and probabilities stay f32 - nothing is narrowed on
/// the way to a fragment - so a score is the scalar kernel's dot product
/// summed in another order, and the softmax is the same max-subtracted form,
/// accumulated block by block. Keys past the group's end are masked to -inf,
/// so a group that is not a whole number of blocks needs no special case.
/// Measured on an M4 over a 640x480 picture (1200 patch rows, 16 heads of 72):
/// 58.8 ms a block for the per-row kernel above, 2.6 ms for this one.
///
/// grid `(groups * ceil(plane / 64), heads)`, block 256, no dynamic
/// threadgroup memory.
#define VIT_FLASH_PARAMS \
        constant vit_args_attention & args [[buffer(0)]], \
        device float *out                  [[buffer(1)]], \
        device const float *q              [[buffer(2)]], \
        device const float *k              [[buffer(3)]], \
        device const float *v              [[buffer(4)]], \
        uint3 tgpig[[threadgroup_position_in_grid]], \
        ushort tid [[thread_index_in_threadgroup]], \
        ushort lane [[thread_index_in_simdgroup]], \
        ushort sg [[simdgroup_index_in_threadgroup]]

template <uint HD>
kernel void kernel_vit_flash_attn(VIT_FLASH_PARAMS) {
    const uint QR = VIT_FLASH_QROWS;
    const uint KB = VIT_FLASH_KEYS;
    const uint NB = KB / 8u;
    const uint NT = VIT_FLASH_THREADS;
    const uint ND = HD / 8u;
    // Each staged row padded by four floats so the transposed key loads do
    // not land eight rows on one bank.
    const uint LDK = HD + 4u;
    threadgroup float kv[2u * VIT_FLASH_KEYS * (HD + 4u)];

    const uint h = tgpig.y;
    const uint tiles = (args.plane + QR - 1u) / QR;
    const uint base = (tgpig.x / tiles) * args.plane;
    const uint end = min(base + args.plane, args.rows);
    const uint q0 = base + (tgpig.x % tiles) * QR;
    if (h >= args.heads || base >= args.rows) {
        return;
    }
    const uint dim = args.heads * HD;
    const uint off = h * HD;
    const float scale = args.scale;

    // The lane's place in every 8x8 fragment: columns `fn, fn + 1` of row
    // `fr`; the four lanes sharing a row differ in lane bits 0 and 3.
    const ushort fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    const ushort fr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    const uint qrow = q0 + sg * 8u + fr;
    const bool live = qrow < end;

    // The query fragments, read straight into the lane's own elements (a
    // fragment's element layout is the same for every element type).
    simdgroup_float8x8 qf[ND];
    simdgroup_float8x8 of[ND];
    device const float *qp = q + (uint64_t) (live ? qrow : base) * dim + off;
    VIT_UNROLL for (uint i = 0; i < ND; ++i) {
        qf[i].thread_elements()[0] = live ? qp[8u * i + fn] : 0.0f;
        qf[i].thread_elements()[1] = live ? qp[8u * i + fn + 1u] : 0.0f;
        of[i] = simdgroup_float8x8(0.0f);
    }

    float run_max = -INFINITY;
    float run_sum = 0.0f;
    threadgroup float *sk = kv;
    threadgroup float *sv = kv + KB * LDK;
    // Four consecutive head values per load; a key past the group's end
    // stages as zero and is masked below.
    const uint HD4 = HD / 4u;
    const uint SL = (KB * HD4 + NT - 1u) / NT;
    float4 rk[SL];
    float4 rv[SL];
    auto fetch = [&](uint s0) {
        VIT_UNROLL for (uint t = 0; t < SL; ++t) {
            const uint e = tid + t * NT;
            const uint j = e / HD4;
            const uint d = (e - j * HD4) * 4u;
            const uint s = s0 + j;
            rk[t] = float4(0.0f);
            rv[t] = float4(0.0f);
            if (e < KB * HD4 && s < end) {
                rk[t] = float4(*((device const packed_float4 *) (k + (uint64_t) s * dim + off + d)));
                rv[t] = float4(*((device const packed_float4 *) (v + (uint64_t) s * dim + off + d)));
            }
        }
    };
    fetch(base);
    for (uint s0 = base; s0 < end; s0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        VIT_UNROLL for (uint t = 0; t < SL; ++t) {
            const uint e = tid + t * NT;
            if (e < KB * HD4) {
                const uint j = e / HD4;
                const uint d = (e - j * HD4) * 4u;
                *((threadgroup float4 *) (sk + j * LDK + d)) = rk[t];
                *((threadgroup float4 *) (sv + j * LDK + d)) = rv[t];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s0 + KB < end) {
            fetch(s0 + KB);
        }

        // Scores: S (8 x KB) = Q (8 x HD) . K^T, one fragment per 8 keys.
        simdgroup_float8x8 sf[NB];
        VIT_UNROLL for (uint b = 0; b < NB; ++b) {
            sf[b] = simdgroup_float8x8(0.0f);
        }
        VIT_UNROLL for (uint i = 0; i < ND; ++i) {
            VIT_UNROLL for (uint b = 0; b < NB; ++b) {
                simdgroup_float8x8 kt;
                // kt(r, c) = K[8b + c][8i + r]: the transposed load.
                simdgroup_load(kt, sk + 8u * b * LDK + 8u * i, LDK, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sf[b], qf[i], kt, sf[b]);
            }
        }

        float block_max = -INFINITY;
        VIT_UNROLL for (uint b = 0; b < NB; ++b) {
            VIT_UNROLL for (uint c = 0; c < 2u; ++c) {
                const float score = sf[b].thread_elements()[c] * scale;
                const bool real = s0 + 8u * b + fn + c < end;
                sf[b].thread_elements()[c] = real ? score : -INFINITY;
                block_max = max(block_max, sf[b].thread_elements()[c]);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        const float new_max = max(run_max, block_max);
        const float alpha = exp(run_max - new_max);
        float block_sum = 0.0f;
        VIT_UNROLL for (uint b = 0; b < NB; ++b) {
            VIT_UNROLL for (uint c = 0; c < 2u; ++c) {
                const float p = exp(sf[b].thread_elements()[c] - new_max);
                sf[b].thread_elements()[c] = p;
                block_sum += p;
            }
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 8);
        run_sum = run_sum * alpha + block_sum;
        run_max = new_max;

        VIT_UNROLL for (uint i = 0; i < ND; ++i) {
            of[i].thread_elements()[0] *= alpha;
            of[i].thread_elements()[1] *= alpha;
            VIT_UNROLL for (uint b = 0; b < NB; ++b) {
                simdgroup_float8x8 vf;
                simdgroup_load(vf, sv + 8u * b * LDK + 8u * i, LDK);
                simdgroup_multiply_accumulate(of[i], sf[b], vf, of[i]);
            }
        }
    }

    // Normalize by the denominator and write each lane's two elements of
    // every fragment straight to its row.
    if (!live) {
        return;
    }
    const float inv = 1.0f / run_sum;
    device float *op = out + (uint64_t) qrow * dim + off;
    VIT_UNROLL for (uint i = 0; i < ND; ++i) {
        op[8u * i + fn] = of[i].thread_elements()[0] * inv;
        op[8u * i + fn + 1u] = of[i].thread_elements()[1] * inv;
    }
}

// One instantiation per head width a tower ships with: 64 and 80 beside the
// 72 of the Qwen 3.5 towers. A wider head would need more than the 32 KiB of
// threadgroup memory two f32 key blocks of 32 rows hold.
template [[host_name("kernel_vit_flash_attn_d64")]]
kernel void kernel_vit_flash_attn<64u>(VIT_FLASH_PARAMS);
template [[host_name("kernel_vit_flash_attn_d72")]]
kernel void kernel_vit_flash_attn<72u>(VIT_FLASH_PARAMS);
template [[host_name("kernel_vit_flash_attn_d80")]]
kernel void kernel_vit_flash_attn<80u>(VIT_FLASH_PARAMS);

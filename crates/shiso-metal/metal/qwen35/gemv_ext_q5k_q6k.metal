// Q5_K and Q6_K coverage for the small-batch matvec and the quantized gather.
//
// The generic set stops at q4_K: it has no q5_K or q6_K block struct, no
// dequantizer for either, and therefore no dispatch instantiation for either.
// The qwen35 checkpoint keeps its expert and up projections in q5_K and its
// output projection in q6_K, so both formats need a path before an engine can
// read those tensors at all.
//
// This file is the last entry of the translation unit, so everything the
// generic set declares is already visible and nothing here re-declares it: the
// q4_K scale/min unpack (`shs_dense_q4_K_scale_min`), the argument blocks, the
// `kernel_mul_mv_ext_q4_f32_disp` template and its `mul_mv_ext_q4_f32_t`
// typedef all come from `metal/generic/mul_mv_ext.metal`, and `QK_K` from
// `metal/generic/00_common.metal`.
//
// The dequantizers are the upstream `dequantize_q5_K` / `dequantize_q6_K`
// shapes with the `shs_dense_` prefix this set uses. Neither the element order
// nor the fold order is adjusted: `il` walks 16-value groups in both, so
// `reg[i / 4][i % 4]` is element `16*il + i` exactly as in the q4_K helper
// above, and each value is built as `dl * q - ml` (q5_K) or `d * (q - 32)`
// (q6_K) with the same left-to-right association. That is the contract, not a
// style choice: a reordered fold changes the emitted bits.

/// `q5_K`: fp16 scale and minimum, six 6-bit scale/min pairs, the fifth-bit
// plane (one bit per weight) and the low nibbles. 176 bytes, matching the
// loader's `BlockQ5K`.
struct shs_dense_block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[QK_K / 8];
    uchar qs[QK_K / 2];
};

// `q6_K`: low nibbles, the two high bits per weight, one signed scale per
// 16-element subblock and an fp16 block scale. 210 bytes, matching the
// loader's `BlockQ6K`.
struct shs_dense_block_q6_K {
    uchar ql[QK_K / 2];
    uchar qh[QK_K / 4];
    char scales[QK_K / 16];
    half d;
};

// Element `16*il + i` of a q5_K block: `il` walks 16-value groups, group
// `group = il / 2` supplies the scale/min pair and the nibble half, and the
// odd `il` reads the upper half of that group's 32 bytes. The fifth bit lives
// in its own plane, addressed by the position inside the 32-value group and
// bit-positioned by the group, so the two nibbles of one byte sit 32 elements
// apart rather than adjacent.
template <typename type4x4>
void dequantize_dense_q5_K(device const shs_dense_block_q5_K *xb, short il, thread type4x4 &reg) {
    const short group = il / 2;
    const short l0 = (il & 1) * 16;
    const uchar2 sm = shs_dense_q4_K_scale_min(group, 0, xb->scales);
    const float dl = (float)xb->d * (float)sm.x;
    const float ml = (float)xb->dmin * (float)sm.y;
    device const uchar *q = xb->qs + (group / 2) * 32 + l0;
    device const uchar *qh = xb->qh + l0;
    const uint shift = (uint)(group & 1) * 4u;
    const uchar hmask = (uchar)(1u << (uint)group);

    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        uint v = ((uint)q[i] >> shift) & 0x0Fu;
        v += (qh[i] & hmask) ? 16u : 0u;
        reg_f[i / 4][i % 4] = dl * (float)v - ml;
    }
    reg = (type4x4)reg_f;
}

// The same 16-value group shape for q6_K: two 128-element halves, each walked
// as four 32-element quarters, with the two high bits interleaved four per byte
// and one signed scale per 16 elements. The -32 bias is folded before the
// multiply, as upstream folds it.
template <typename type4x4>
void dequantize_dense_q6_K(device const shs_dense_block_q6_K *xb, short il, thread type4x4 &reg) {
    const short n128 = il / 8;
    const short il128 = il & 7;
    const short quarter = il128 / 2;
    const short l0 = (il128 & 1) * 16;
    const uint ql_base = (uint)n128 * 64u;
    const uint qh_base = (uint)n128 * 32u;
    const uint sc_base = (uint)n128 * 8u + (uint)quarter * 2u + (uint)(il128 & 1);
    const float d = (float)xb->d * (float)((int)xb->scales[sc_base]);

    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        const uint l = (uint)l0 + (uint)i;
        uint v;
        if (quarter == 0) {
            v = ((uint)xb->ql[ql_base + l] & 0x0Fu) |
                ((((uint)xb->qh[qh_base + l] >> 0u) & 3u) << 4u);
        } else if (quarter == 1) {
            v = ((uint)xb->ql[ql_base + 32u + l] & 0x0Fu) |
                ((((uint)xb->qh[qh_base + l] >> 2u) & 3u) << 4u);
        } else if (quarter == 2) {
            v = ((uint)xb->ql[ql_base + l] >> 4u) |
                ((((uint)xb->qh[qh_base + l] >> 4u) & 3u) << 4u);
        } else {
            v = ((uint)xb->ql[ql_base + 32u + l] >> 4u) |
                ((((uint)xb->qh[qh_base + l] >> 6u) & 3u) << 4u);
        }
        reg_f[i / 4][i % 4] = d * (float)((int)v - 32);
    }
    reg = (type4x4)reg_f;
}

// Sub-block `s` (32 weights) as integer codes for the K-quant matvec in
// `metal/generic/mul_mv_ext.metal`, with the pair that folds the scales in
// after the dot. The codes are the helpers' own above, read straight out of
// the packed block.
void deq32_q5_K(device const shs_dense_block_q5_K *xb, short s, thread float4 *qv, thread float2 &ab) {
    const uchar2 sm = shs_dense_q4_K_scale_min(s, 0, xb->scales);
    ab = float2((float)xb->d * (float)sm.x, -((float)xb->dmin * (float)sm.y));
    device const uchar4 *q = (device const uchar4 *)(xb->qs + (s / 2) * 32);
    device const uchar4 *h = (device const uchar4 *)(xb->qh);
    const uchar sh = (s & 1) ? 4 : 0;
    const uchar hmask = (uchar)(1u << (uint)s);
    for (short k = 0; k < 8; ++k) {
        const float4 lo = float4((q[k] >> sh) & uchar(0x0F));
        qv[k] = lo + select(float4(0.0f), float4(16.0f), (h[k] & hmask) != uchar(0));
    }
}

// q6_K has no minimum and two signed scales per 32 weights, one per 16, so
// the scales ride on the codes and the pair is `(d, 0)`. A q6_K block is 210
// bytes, so a block after the first is only two-byte aligned and the bytes are
// read through packed vectors, which carry no alignment of their own.
void deq32_q6_K(device const shs_dense_block_q6_K *xb, short s, thread float4 *qv, thread float2 &ab) {
    const uint n128 = (uint)s >> 2u;
    const uint quarter = (uint)s & 3u;
    device const packed_uchar4 *ql = (device const packed_uchar4 *)(xb->ql + n128 * 64u + (quarter & 1u) * 32u);
    device const packed_uchar4 *qh = (device const packed_uchar4 *)(xb->qh + n128 * 32u);
    const uchar ql_shift = (uchar)((quarter >> 1u) * 4u);
    const uchar qh_shift = (uchar)(quarter * 2u);
    const float sc0 = (float)((int)xb->scales[n128 * 8u + quarter * 2u]);
    const float sc1 = (float)((int)xb->scales[n128 * 8u + quarter * 2u + 1u]);
    ab = float2((float)xb->d, 0.0f);
    for (short k = 0; k < 8; ++k) {
        const uchar4 lo = (uchar4(ql[k]) >> ql_shift) & uchar(0x0F);
        const uchar4 hi = (uchar4(qh[k]) >> qh_shift) & uchar(3);
        const float4 v = float4(lo | (hi << uchar(4))) - 32.0f;
        qv[k] = v * (k < 4 ? sc0 : sc1);
    }
}

// The same K-quant dispatch the q4_K rows use, with its row-blocking constants
// from the same function constants (indexes 600 and 601).
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_dense_block_q5_K, deq32_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_dense_block_q5_K, deq32_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_dense_block_q5_K, deq32_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_dense_block_q5_K, deq32_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_dense_block_q5_K, deq32_q5_K>;

template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_dense_block_q6_K, deq32_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_dense_block_q6_K, deq32_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_dense_block_q6_K, deq32_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_dense_block_q6_K, deq32_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_dense_block_q6_K, deq32_q6_K>;

// One threadgroup per 256-element block of one row, exactly the shape of
// `kernel_get_rows_q4_K_f32`, so the gather shares its argument block and its
// launch geometry. The scale is signed here and there is no minimum: q6_K is
// symmetric around -32, folded before the multiply.
kernel void kernel_get_rows_q6_K_f32(
        constant shs_metal_args_get_rows_q8_0 & args,
        device const char    * src0,
        device const char    * src1,
        device       char    * dst,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        ushort                 tiitg[[thread_index_in_threadgroup]],
        ushort3                ntg [[threads_per_threadgroup]]) {
    const int32_t block = (int32_t)tgpig.x;
    const int32_t tok_i = (int32_t)tgpig.y;
    if (tok_i >= args.n_tokens) return;

    const int32_t token =
        ((const device int32_t *)(src1 + (uint64_t)tok_i * args.token_stride))[0];
    if (token < 0 || token >= args.n_vocab) return;

    const device shs_dense_block_q6_K *row =
        (const device shs_dense_block_q6_K *)(src0 + (uint64_t)token * args.src_row_bytes);
    const device shs_dense_block_q6_K *qb = row + block;
    device float *out =
        (device float *)(dst + (uint64_t)tok_i * args.dst_row_bytes);

    const int32_t i0 = block * QK_K;
    for (int32_t i = (int32_t)tiitg; i < QK_K; i += (int32_t)ntg.x) {
        const int32_t idx = i0 + i;
        if (idx < args.n_embd) {
            const uint n128 = (uint)i >> 7u;
            const uint r = (uint)i & 127u;
            const uint quarter = r >> 5u;
            const uint l = r & 31u;
            const uint ql_base = n128 * 64u;
            const uint qh_base = n128 * 32u;
            uint q;
            if (quarter == 0u) {
                q = (qb->ql[ql_base + l] & 0x0Fu) |
                    (((qb->qh[qh_base + l] >> 0u) & 3u) << 4u);
            } else if (quarter == 1u) {
                q = (qb->ql[ql_base + 32u + l] & 0x0Fu) |
                    (((qb->qh[qh_base + l] >> 2u) & 3u) << 4u);
            } else if (quarter == 2u) {
                q = (qb->ql[ql_base + l] >> 4u) |
                    (((qb->qh[qh_base + l] >> 4u) & 3u) << 4u);
            } else {
                q = (qb->ql[ql_base + 32u + l] >> 4u) |
                    (((qb->qh[qh_base + l] >> 6u) & 3u) << 4u);
            }
            const uint sc_base = n128 * 8u + quarter * 2u + (l >> 4u);
            out[idx] = (float)qb->d * (float)((int)qb->scales[sc_base]) *
                       (float)((int)q - 32);
        }
    }
}

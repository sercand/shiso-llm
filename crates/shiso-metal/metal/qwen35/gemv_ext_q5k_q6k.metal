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

// The float4 view the matvec consumes, identical in shape to
// `dequantize_dense_q4_K_t4`: chunk `il` is elements `[4*il, 4*il+4)`, taken as
// row `il & 3` of the 16-value group `il / 4`.
template <typename type4>
void dequantize_dense_q5_K_t4(device const shs_dense_block_q5_K *xb, short il, thread type4 &reg) {
    float4x4 tmp;
    dequantize_dense_q5_K(xb, il / 4, tmp);
    const short row = il & 3;
    for (int i = 0; i < 4; i++) {
        reg[i] = tmp[row][i];
    }
}

template <typename type4>
void dequantize_dense_q6_K_t4(device const shs_dense_block_q6_K *xb, short il, thread type4 &reg) {
    float4x4 tmp;
    dequantize_dense_q6_K(xb, il / 4, tmp);
    const short row = il & 3;
    for (int i = 0; i < 4; i++) {
        reg[i] = tmp[row][i];
    }
}

// `epb` is elements per block and the dispatch template divides it by four to
// get chunks per block, which is the same 256 the q4_K lines pass: both formats
// carry 256 weights per block, so the chunking the matvec walks is unchanged.
// The row-blocking constants come from the same function constants as the q4_K
// path (indexes 600 and 601), because this is the same dispatch template.
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_1")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<1, shs_dense_block_q5_K, 256, dequantize_dense_q5_K_t4>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, shs_dense_block_q5_K, 256, dequantize_dense_q5_K_t4>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, shs_dense_block_q5_K, 256, dequantize_dense_q5_K_t4>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, shs_dense_block_q5_K, 256, dequantize_dense_q5_K_t4>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, shs_dense_block_q5_K, 256, dequantize_dense_q5_K_t4>;

template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_1")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<1, shs_dense_block_q6_K, 256, dequantize_dense_q6_K_t4>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, shs_dense_block_q6_K, 256, dequantize_dense_q6_K_t4>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, shs_dense_block_q6_K, 256, dequantize_dense_q6_K_t4>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, shs_dense_block_q6_K, 256, dequantize_dense_q6_K_t4>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, shs_dense_block_q6_K, 256, dequantize_dense_q6_K_t4>;

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

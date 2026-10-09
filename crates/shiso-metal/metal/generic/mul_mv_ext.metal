struct shs_metal_args_mul_mv_ext {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

struct shs_dense_block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

static inline uchar2 shs_dense_q4_K_scale_min(int j, int k, device const uchar *q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0x0f) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}



template <typename type4x4>
void dequantize_dense_q4_K(device const shs_dense_block_q4_K *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = shs_dense_q4_K_scale_min(is, il / 2, xb->scales);
    const float d = il < 2 ? (float)xb->d : (float)xb->d * (1.0f / 16.0f);
    const float min = (float)xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;

    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        reg_f[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
    reg = (type4x4)reg_f;
}

// Sub-block `s` (32 weights) of a q4_K block as integer codes, with the
// sub-block's scale and negated minimum: the sub-block's dot is
// `ab.x * sum(q * y) + ab.y * sum(y)`. Sub-blocks 2k and 2k+1 share the same 32
// bytes, low nibble then high.
void deq32_q4_K(device const shs_dense_block_q4_K *xb, short s, thread float4 *qv, thread float2 &ab) {
    const uchar2 sc = shs_dense_q4_K_scale_min(s, 0, xb->scales);
    ab = float2((float)xb->d * sc[0], -((float)xb->dmin * sc[1]));
    device const uchar4 *q = (device const uchar4 *)(xb->qs + (s / 2) * 32);
    const uchar sh = (s & 1) ? 4 : 0;
    for (short k = 0; k < 8; ++k) {
        qv[k] = float4((q[k] >> sh) & uchar(0x0F));
    }
}

// ggml q4_0: one half scale and 32 four-bit weights biased by -8, the low
// nibble of byte j carrying element j and the high nibble element j + 16.
//
// It sits beside its two dequantizers rather than in the prelude next to
// block_q8_0, because the dequantizers are the only readers of it as a struct:
// the library is one translation unit, so gemm_q4k.metal instantiates
// kernel_mul_mm over this same declaration. get_rows.metal holds a second
// declaration of the same 18 bytes under its own name and walks it byte by
// byte, which is why nothing here has to be shared with that gather.
struct block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2];
};

// The two dequantizers below read the block as uint16_t pairs, which is what
// lets one masked multiply produce two elements: mask0 takes the nibble of the
// low byte and mask1 the same nibble of the high byte, and the 1/256 in d2
// undoes that byte's shift. il selects the nibble half, so il == 0 is elements
// 0..15 and il == 1 is elements 16..31.
template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 *xb, short il, thread type4x4 &reg) {
    device const uint16_t *qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

// The float4 view of the same unpack: il runs 0..7 over the eight float4 chunks
// of a block, its high bit choosing the nibble half and its low two bits the
// pair of uint16_t within that half.
template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 *xb, short il, thread type4 &reg) {
    device const uint16_t *qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}

// Small-batch mat-vec kernel used for 2..8 prompt tokens.
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant shs_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread

    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// Small-batch prompt matvec for 2..5 tokens. It bridges decode-style matvec and
// full matmul when prefill chunks are too small to amortize matrix tiles.
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant shs_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0, 32, dequantize_q8_0_t4>;

template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_1")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<1, block_q4_0, 32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_0, 32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_0, 32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_0, 32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_0, 32, dequantize_q4_0_t4>;

// The K-quant small-batch matvec: the q4_K, q5_K and q6_K rows of the family.
//
// Same grid, same lane split, same argument block as the dispatch above, so
// the host side cannot tell the two apart. What differs is the unit a lane
// takes. The float4 walk above unpacks four weights per turn, and for a
// K-quant that means re-deriving the sub-block's scale and minimum for every
// four weights: at one token that arithmetic, not memory, was the bound, and
// these formats ran at 30-70 GB/s (q6_K at 6) where q4_0 reaches 195 on the
// same M-series part. Here a lane takes a whole 32-weight sub-block, unpacks
// the integer codes once, and folds the scale and minimum in after the dot:
// `dl * sum(q * y) - ml * sum(y)`. The weights are read in their packed form
// and never widened to a float matrix anywhere.
//
// `deq32` writes the sub-block's 32 codes as eight float4 and the pair
// `(a, b)` such that the sub-block's contribution is `a * sum(q * y) +
// b * sum(y)`.
template<short r1ptg, typename q_t, void (*deq32)(device const q_t *, short, thread float4 *, thread float2 &)>
kernel void kernel_mul_mv_ext_kq_f32_disp(
        constant shs_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    // Every K-quant block is 256 weights, eight sub-blocks of 32.
    const int nsub = args.ne00/32;

    for (int u = tx; u < nsub; u += nxpsg) {
        float4 qv[8];
        float2 ab;
        deq32(xq + u/8, (short)(u%8), qv, ab);

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            device const float4 * y = y4[ir1] + u*8;
            float4 qy = 0.0f;
            float4 sy = 0.0f;
#pragma unroll(8)
            for (short k = 0; k < 8; ++k) {
                const float4 yk = y[k];
                qy += qv[k]*yk;
                sy += yk;
            }
            sumf[ir1] += ab.x*((qy[0] + qy[1]) + (qy[2] + qy[3])) + ab.y*((sy[0] + sy[1]) + (sy[2] + sy[3]));
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

typedef decltype(kernel_mul_mv_ext_kq_f32_disp<1, shs_dense_block_q4_K, deq32_q4_K>) mul_mv_ext_kq_f32_t;

template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_dense_block_q4_K, deq32_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_dense_block_q4_K, deq32_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_dense_block_q4_K, deq32_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_dense_block_q4_K, deq32_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_dense_block_q4_K, deq32_q4_K>;

// The mixed i-quant formats and the 2-bit K-quant, on the same kq walk.
//
// All nine blocks are 256 weights, so the lane split above needs no change:
// lane `s` of a block owns weights `32*s .. 32*s+32`, which for the i-quants
// is exactly one of their eight 32-weight groups (and for q2_K two of its
// sixteen 16-weight subblocks). The dequantizers below are the device twins of
// the host reference decoders in `shiso_modelfile::iq_quant` — same block
// bytes in, same f32 values out, in the same multiplication order — and the
// codebooks they read are the `constant` arrays of `iq_tables.metal`.
//
// The i-quants are non-linear: a weight is a codebook byte times a sign, not
// an affine image of a small integer code, so there is no `(a, b)` pair to
// fold the dot into. Each dequantizer therefore writes the fully dequantized
// weights and hands back `ab = (1, 0)`; the template's `b * sum(y)` leg then
// contributes nothing and `a * sum(q * y)` is the plain dot of the weights.
// q2_K is affine per 16-weight subblock but the two subblocks of a 32-weight
// lane carry different scale/min pairs, so it takes the same route.

// The grid byte `j` of a packed entry, as a float (the 2- and 3-bit grids).
static inline float shs_iq_byte(ulong grid, short j) {
    return (float)((grid >> (8 * j)) & 0xFF);
}

// The same for the ternary grid, whose bytes are signed.
static inline float shs_iq_sbyte(ulong grid, short j) {
    return (float)(char)((grid >> (8 * j)) & 0xFF);
}

// The sign lane `j` of a sign byte: the 2- and 3-bit formats negate the grid
// byte where the byte's bit `j` is set.
static inline float shs_iq_sign(uchar signs, short j) {
    return (signs & (uchar)(1u << j)) ? -1.0f : 1.0f;
}

struct shs_block_q2_K {
    uchar scales[16];
    uchar qs[64];
    half d;
    half dmin;
};

// q2_K stores its weights plane by plane, not in subblock order: the 16
// weights of subblock `16*(8*plane + 2*j + h)` sit at byte `32*plane + 16*h`
// of `qs` in bit field `2*j`, and take scale byte `8*plane + 2*j + h` as
// `d * (sc & 0xF) * q - dmin * (sc >> 4)`. A 32-weight lane is one `j` of one
// plane, so it reads 32 bytes at a fixed shift and two scale bytes.
void deq32_q2_K(device const shs_block_q2_K *xb, short s, thread float4 *qv, thread float2 &ab) {
    const short plane = s >> 2;
    const short shift = 2 * (s & 3);
    device const uchar * q = xb->qs + plane * 32;
    device const uchar * sc = xb->scales + plane * 8 + 2 * (s & 3);
    for (short h = 0; h < 2; ++h) {
        const float dl = (float)xb->d * (float)(sc[h] & 0xF);
        const float ml = (float)xb->dmin * (float)(sc[h] >> 4);
        for (short l = 0; l < 16; ++l) {
            qv[h * 4 + l / 4][l % 4] = dl * (float)((q[h * 16 + l] >> shift) & 3) - ml;
        }
    }
    ab = float2(1.0f, 0.0f);
}

struct shs_block_iq2_xxs {
    half d;
    ushort qs[32];
};

// Two index words then two sign-and-scale words per 32-weight group, as four
// u16s; the group's scale is the top nibble of the second word. The index
// words are read as u16 pairs because a 66-byte block only ever lands on a
// 2-byte boundary.
void deq32_iq2_xxs(device const shs_block_iq2_xxs *xb, short s, thread float4 *qv, thread float2 &ab) {
    const uint aux0 = (uint)xb->qs[4 * s + 0] | ((uint)xb->qs[4 * s + 1] << 16);
    const uint aux1 = (uint)xb->qs[4 * s + 2] | ((uint)xb->qs[4 * s + 3] << 16);
    const float db = (float)xb->d * (0.5f + (float)((aux1 >> 28) & 0xF)) * 0.25f;
    for (short l = 0; l < 4; ++l) {
        const ulong grid = IQ2XXS_GRID[(aux0 >> (8 * l)) & 0xFF];
        const uchar signs = KSIGNS_IQ2XS[(aux1 >> (7 * l)) & 0x7F];
        for (short j = 0; j < 8; ++j) {
            qv[2 * l + j / 4][j % 4] = db * shs_iq_byte(grid, j) * shs_iq_sign(signs, j);
        }
    }
    ab = float2(1.0f, 0.0f);
}

struct shs_block_iq2_xs {
    half d;
    ushort qs[32];
    uchar scales[8];
};

// Nine index bits into the grid, seven sign bits above them, two 4-bit scales
// a byte for the group's two 16-weight halves.
void deq32_iq2_xs(device const shs_block_iq2_xs *xb, short s, thread float4 *qv, thread float2 &ab) {
    const float d = (float)xb->d;
    const uchar sc = xb->scales[s];
    const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
    for (short l = 0; l < 4; ++l) {
        const ushort q = xb->qs[4 * s + l];
        const ulong grid = IQ2XS_GRID[q & 511];
        const uchar signs = KSIGNS_IQ2XS[q >> 9];
        const float db = (l < 2) ? db0 : db1;
        for (short j = 0; j < 8; ++j) {
            qv[2 * l + j / 4][j % 4] = db * shs_iq_byte(grid, j) * shs_iq_sign(signs, j);
        }
    }
    ab = float2(1.0f, 0.0f);
}

struct shs_block_iq2_s {
    half d;
    uchar qs[64];
    uchar qh[8];
    uchar scales[8];
};

// The grid indices occupy the first half of `qs` and the per-weight signs the
// second; two high index bits per quad come from `qh`.
void deq32_iq2_s(device const shs_block_iq2_s *xb, short s, thread float4 *qv, thread float2 &ab) {
    const float d = (float)xb->d;
    const uchar sc = xb->scales[s];
    const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
    const uint qh = xb->qh[s];
    for (short l = 0; l < 4; ++l) {
        const uint idx = (uint)xb->qs[4 * s + l] | ((qh << (8 - 2 * l)) & 0x300);
        const ulong grid = IQ2S_GRID[idx];
        const uchar signs = xb->qs[QK_K / 8 + 4 * s + l];
        const float db = (l < 2) ? db0 : db1;
        for (short j = 0; j < 8; ++j) {
            qv[2 * l + j / 4][j % 4] = db * shs_iq_byte(grid, j) * shs_iq_sign(signs, j);
        }
    }
    ab = float2(1.0f, 0.0f);
}

struct shs_block_iq3_xxs {
    half d;
    uchar qs[96];
};

// Sixty-four grid-index bytes, then one sign-and-scale word per group after
// them. Each index covers four weights, so a quad reads two grid entries and
// splits its eight weights between them.
void deq32_iq3_xxs(device const shs_block_iq3_xxs *xb, short s, thread float4 *qv, thread float2 &ab) {
    device const uchar * a = xb->qs + QK_K / 4 + 4 * s;
    const uint aux = (uint)a[0] | ((uint)a[1] << 8) | ((uint)a[2] << 16) | ((uint)a[3] << 24);
    const float db = (float)xb->d * (0.5f + (float)((aux >> 28) & 0xF)) * 0.5f;
    for (short l = 0; l < 4; ++l) {
        const uchar signs = KSIGNS_IQ2XS[(aux >> (7 * l)) & 0x7F];
        const uint g1 = IQ3XXS_GRID[xb->qs[8 * s + 2 * l]];
        const uint g2 = IQ3XXS_GRID[xb->qs[8 * s + 2 * l + 1]];
        for (short j = 0; j < 4; ++j) {
            qv[2 * l + j / 4][j % 4] = db * (float)((g1 >> (8 * j)) & 0xFF) * shs_iq_sign(signs, j);
            qv[2 * l + (j + 4) / 4][(j + 4) % 4] = db * (float)((g2 >> (8 * j)) & 0xFF) * shs_iq_sign(signs, j + 4);
        }
    }
    ab = float2(1.0f, 0.0f);
}

struct shs_block_iq3_s {
    half d;
    uchar qs[64];
    uchar qh[8];
    uchar signs[32];
    uchar scales[4];
};

// Odd 4-bit scales (`1 + 2 * nibble`), one scale byte per two groups; the
// high index bit of each of a group's eight indices is one bit of `qh`, and
// the signs are one byte per quad.
void deq32_iq3_s(device const shs_block_iq3_s *xb, short s, thread float4 *qv, thread float2 &ab) {
    const float d = (float)xb->d;
    const uchar sc = xb->scales[s / 2];
    const float db = d * (1.0f + 2.0f * (float)((s & 1) ? (sc >> 4) : (sc & 0xF)));
    const uint qh = xb->qh[s];
    for (short l = 0; l < 4; ++l) {
        const uint g1 = IQ3S_GRID[(uint)xb->qs[8 * s + 2 * l] | ((qh << (8 - 2 * l)) & 256)];
        const uint g2 = IQ3S_GRID[(uint)xb->qs[8 * s + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
        const uchar signs = xb->signs[4 * s + l];
        for (short j = 0; j < 4; ++j) {
            qv[2 * l + j / 4][j % 4] = db * (float)((g1 >> (8 * j)) & 0xFF) * shs_iq_sign(signs, j);
            qv[2 * l + (j + 4) / 4][(j + 4) % 4] = db * (float)((g2 >> (8 * j)) & 0xFF) * shs_iq_sign(signs, j + 4);
        }
    }
    ab = float2(1.0f, 0.0f);
}

struct shs_block_iq1_s {
    half d;
    uchar qs[32];
    ushort qh[8];
};

// The group word carries the high three index bits of each quad (three bits
// a quad from bit 0 up), a 3-bit scale multiplier at bits 12..14, and the
// sign of the grid shift at bit 15: a weight is `dl * (grid + delta)`, and
// skipping the delta is wrong, not merely imprecise.
void deq32_iq1_s(device const shs_block_iq1_s *xb, short s, thread float4 *qv, thread float2 &ab) {
    const ushort qh = xb->qh[s];
    const float dl = (float)xb->d * (float)(2 * ((qh >> 12) & 7) + 1);
    const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
    for (short l = 0; l < 4; ++l) {
        const uint idx = (uint)xb->qs[4 * s + l] | (uint(((qh >> (3 * l)) & 7)) << 8);
        const ulong grid = IQ1S_GRID[idx];
        for (short j = 0; j < 8; ++j) {
            qv[2 * l + j / 4][j % 4] = dl * (shs_iq_sbyte(grid, j) + delta);
        }
    }
    ab = float2(1.0f, 0.0f);
}

struct shs_block_iq1_m {
    uchar qs[32];
    uchar qh[16];
    uchar scales[8];
};

// The block f16 is not stored: it is the four top nibbles of the odd scale
// bytes, least significant first. Each 16-weight subblock then carries its
// own 3-bit scale multiplier, and each quad its own grid-shift sign, so the
// lane assembles two scales and four deltas for its 32 weights.
void deq32_iq1_m(device const shs_block_iq1_m *xb, short s, thread float4 *qv, thread float2 &ab) {
    const uint w0 = (uint)xb->scales[0] | ((uint)xb->scales[1] << 8);
    const uint w1 = (uint)xb->scales[2] | ((uint)xb->scales[3] << 8);
    const uint w2 = (uint)xb->scales[4] | ((uint)xb->scales[5] << 8);
    const uint w3 = (uint)xb->scales[6] | ((uint)xb->scales[7] << 8);
    const ushort dbits = (ushort)((w0 >> 12) | ((w1 >> 8) & 0x00F0) | ((w2 >> 4) & 0x0F00) | (w3 & 0xF000));
    const float d = (float)as_type<half>(dbits);
    const uint words[4] = { w0, w1, w2, w3 };
    const uint word = words[s / 2];
    const short shift = 6 * (s % 2);
    const float dl0 = d * (float)(2 * ((word >> shift) & 7) + 1);
    const float dl1 = d * (float)(2 * ((word >> (shift + 3)) & 7) + 1);
    for (short m = 0; m < 4; ++m) {
        const uchar qhb = xb->qh[2 * s + m / 2];
        const uint idx = (uint)xb->qs[4 * s + m] | ((((uint)qhb) << ((m & 1) ? 4 : 8)) & 0x700);
        const float delta = (qhb & ((m & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
        const float dl = (m < 2) ? dl0 : dl1;
        const ulong grid = IQ1S_GRID[idx];
        for (short j = 0; j < 8; ++j) {
            qv[2 * m + j / 4][j % 4] = dl * (shs_iq_sbyte(grid, j) + delta);
        }
    }
    ab = float2(1.0f, 0.0f);
}

struct shs_block_iq4_xs {
    half d;
    ushort scales_h;
    uchar scales_l[4];
    uchar qs[128];
};

// A 6-bit subblock scale split across two planes, offset so 32 means no
// scaling; the weights are nibble indices into the non-linear codebook, low
// nibble first for the group's first 16 weights.
void deq32_iq4_xs(device const shs_block_iq4_xs *xb, short s, thread float4 *qv, thread float2 &ab) {
    const uint ls = ((uint)(xb->scales_l[s / 2] >> (4 * (s % 2))) & 0xF)
                  | ((((uint)xb->scales_h) >> (2 * s)) & 3) << 4;
    const float dl = (float)xb->d * ((float)ls - 32.0f);
    for (short j = 0; j < 16; ++j) {
        const uchar q = xb->qs[16 * s + j];
        qv[j / 4][j % 4] = dl * (float)IQ4NL[q & 0xF];
        qv[4 + j / 4][j % 4] = dl * (float)IQ4NL[q >> 4];
    }
    ab = float2(1.0f, 0.0f);
}

template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_block_q2_K, deq32_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_block_q2_K, deq32_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_block_q2_K, deq32_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_block_q2_K, deq32_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_block_q2_K, deq32_q2_K>;

template [[host_name("kernel_mul_mv_ext_iq2_xxs_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_block_iq2_xxs, deq32_iq2_xxs>;
template [[host_name("kernel_mul_mv_ext_iq2_xxs_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_block_iq2_xxs, deq32_iq2_xxs>;
template [[host_name("kernel_mul_mv_ext_iq2_xxs_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_block_iq2_xxs, deq32_iq2_xxs>;
template [[host_name("kernel_mul_mv_ext_iq2_xxs_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_block_iq2_xxs, deq32_iq2_xxs>;
template [[host_name("kernel_mul_mv_ext_iq2_xxs_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_block_iq2_xxs, deq32_iq2_xxs>;

template [[host_name("kernel_mul_mv_ext_iq2_xs_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_block_iq2_xs, deq32_iq2_xs>;
template [[host_name("kernel_mul_mv_ext_iq2_xs_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_block_iq2_xs, deq32_iq2_xs>;
template [[host_name("kernel_mul_mv_ext_iq2_xs_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_block_iq2_xs, deq32_iq2_xs>;
template [[host_name("kernel_mul_mv_ext_iq2_xs_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_block_iq2_xs, deq32_iq2_xs>;
template [[host_name("kernel_mul_mv_ext_iq2_xs_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_block_iq2_xs, deq32_iq2_xs>;

template [[host_name("kernel_mul_mv_ext_iq2_s_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_block_iq2_s, deq32_iq2_s>;
template [[host_name("kernel_mul_mv_ext_iq2_s_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_block_iq2_s, deq32_iq2_s>;
template [[host_name("kernel_mul_mv_ext_iq2_s_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_block_iq2_s, deq32_iq2_s>;
template [[host_name("kernel_mul_mv_ext_iq2_s_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_block_iq2_s, deq32_iq2_s>;
template [[host_name("kernel_mul_mv_ext_iq2_s_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_block_iq2_s, deq32_iq2_s>;

template [[host_name("kernel_mul_mv_ext_iq3_xxs_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_block_iq3_xxs, deq32_iq3_xxs>;
template [[host_name("kernel_mul_mv_ext_iq3_xxs_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_block_iq3_xxs, deq32_iq3_xxs>;
template [[host_name("kernel_mul_mv_ext_iq3_xxs_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_block_iq3_xxs, deq32_iq3_xxs>;
template [[host_name("kernel_mul_mv_ext_iq3_xxs_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_block_iq3_xxs, deq32_iq3_xxs>;
template [[host_name("kernel_mul_mv_ext_iq3_xxs_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_block_iq3_xxs, deq32_iq3_xxs>;

template [[host_name("kernel_mul_mv_ext_iq3_s_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_block_iq3_s, deq32_iq3_s>;
template [[host_name("kernel_mul_mv_ext_iq3_s_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_block_iq3_s, deq32_iq3_s>;
template [[host_name("kernel_mul_mv_ext_iq3_s_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_block_iq3_s, deq32_iq3_s>;
template [[host_name("kernel_mul_mv_ext_iq3_s_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_block_iq3_s, deq32_iq3_s>;
template [[host_name("kernel_mul_mv_ext_iq3_s_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_block_iq3_s, deq32_iq3_s>;

template [[host_name("kernel_mul_mv_ext_iq1_s_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_block_iq1_s, deq32_iq1_s>;
template [[host_name("kernel_mul_mv_ext_iq1_s_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_block_iq1_s, deq32_iq1_s>;
template [[host_name("kernel_mul_mv_ext_iq1_s_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_block_iq1_s, deq32_iq1_s>;
template [[host_name("kernel_mul_mv_ext_iq1_s_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_block_iq1_s, deq32_iq1_s>;
template [[host_name("kernel_mul_mv_ext_iq1_s_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_block_iq1_s, deq32_iq1_s>;

template [[host_name("kernel_mul_mv_ext_iq1_m_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_block_iq1_m, deq32_iq1_m>;
template [[host_name("kernel_mul_mv_ext_iq1_m_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_block_iq1_m, deq32_iq1_m>;
template [[host_name("kernel_mul_mv_ext_iq1_m_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_block_iq1_m, deq32_iq1_m>;
template [[host_name("kernel_mul_mv_ext_iq1_m_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_block_iq1_m, deq32_iq1_m>;
template [[host_name("kernel_mul_mv_ext_iq1_m_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_block_iq1_m, deq32_iq1_m>;

template [[host_name("kernel_mul_mv_ext_iq4_xs_f32_r1_1")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<1, shs_block_iq4_xs, deq32_iq4_xs>;
template [[host_name("kernel_mul_mv_ext_iq4_xs_f32_r1_2")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<2, shs_block_iq4_xs, deq32_iq4_xs>;
template [[host_name("kernel_mul_mv_ext_iq4_xs_f32_r1_3")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<3, shs_block_iq4_xs, deq32_iq4_xs>;
template [[host_name("kernel_mul_mv_ext_iq4_xs_f32_r1_4")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<4, shs_block_iq4_xs, deq32_iq4_xs>;
template [[host_name("kernel_mul_mv_ext_iq4_xs_f32_r1_5")]] kernel mul_mv_ext_kq_f32_t kernel_mul_mv_ext_kq_f32_disp<5, shs_block_iq4_xs, deq32_iq4_xs>;

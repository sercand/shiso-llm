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

template <typename type4>
void dequantize_dense_q4_K_t4(device const shs_dense_block_q4_K *xb, short il, thread type4 &reg) {
    float4x4 tmp;
    dequantize_dense_q4_K(xb, il / 4, tmp);
    const short row = il & 3;
    for (int i = 0; i < 4; i++) {
        reg[i] = tmp[row][i];
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

template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_1")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<1, shs_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, shs_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, shs_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, shs_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, shs_dense_block_q4_K, 256, dequantize_dense_q4_K_t4>;

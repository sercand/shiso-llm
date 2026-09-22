// Metal matvec kernels used by generation.

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

struct shs_metal_args_mul_mv {
    int ne00;
    int ne01;
    int ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int ne10;
    int ne11;
    int ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int ne0;
    int ne1;
    int nr0;
    short r2;
    short r3;
};


template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// Decode-time Q8_0 matrix-vector multiply. shiso uses this for Q8_0 dense
// projections such as shared experts and output-side small matvecs.
[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant shs_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant shs_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// Decode Q-A/KV pair. Both projections consume the same activation row but
// have independent weight ranges and output extents. Keep the standalone Q8_0
// lane/block traversal and two-stage reduction verbatim for each bank; only
// the activation load and threadgroup scheduling are shared.
[[host_name("kernel_mul_mv_q8_0_f32_pair")]]
kernel void kernel_mul_mv_q8_0_f32_pair(
        constant shs_metal_args_mul_mv & args0,
        constant shs_metal_args_mul_mv & args1,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = args0.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const bool active_a = r0 < args0.ne01;
    const bool active_b = r0 < args1.ne01;

    device const float *y = (device const float *)src1;
    device const block_q8_0 *ax_a[NR0];
    device const block_q8_0 *ax_b[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int out_row = r0 + row;
        ax_a[row] = active_a && out_row < args0.ne01
            ? (device const block_q8_0 *)(src0_a + (uint64_t)out_row * args0.nb01)
            : (device const block_q8_0 *)src0_a;
        ax_b[row] = active_b && out_row < args1.ne01
            ? (device const block_q8_0 *)(src0_b + (uint64_t)out_row * args1.nb01)
            : (device const block_q8_0 *)src0_b;
    }

    float suma[NR0] = { 0.f };
    float sumb[NR0] = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const int out_row = r0 + row;
            if (active_a && out_row < args0.ne01) {
                device const int8_t *qs = ax_a[row][ib].qs + il * NQ;
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[i];
                }
                suma[row] += sumq * ax_a[row][ib].d;
            }
            if (active_b && out_row < args1.ne01) {
                device const int8_t *qs = ax_b[row][ib].qs + il * NQ;
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[i];
                }
                sumb[row] += sumq * ax_b[row][ib].d;
            }
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shared = (threadgroup float *)shmem;
    threadgroup float *sha[NR0];
    threadgroup float *shb[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sha[row] = shared + NW * row;
        shb[row] = shared + NW * (NR0 + row);
        if (sgitg == 0) {
            sha[row][tiisg] = 0.0f;
            if (active_b) shb[row][tiisg] = 0.0f;
        }
        suma[row] = simd_sum(suma[row]);
        if (active_b) sumb[row] = simd_sum(sumb[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sha[row][sgitg] = suma[row];
            if (active_b) shb[row][sgitg] = sumb[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out_a = (device float *)dst_a;
    device float *out_b = (device float *)dst_b;
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const float total_a = simd_sum(sha[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const int out_row = r0 + row;
            if (active_a && out_row < args0.ne01) out_a[out_row] = total_a;
        }
        if (active_b) {
            const float total_b = simd_sum(shb[row][tiisg]);
            if (tiisg == 0 && sgitg == 0) {
                const int out_row = r0 + row;
                if (out_row < args1.ne01) out_b[out_row] = total_b;
            }
        }
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

/*
 * Bit-identical twin of dequantize_q8_0 for the MPP staging loop: same
 * half(float(qs[i]) * d) per element, but the 16 consecutive int8 lanes are
 * fetched as eight aligned 16-bit loads instead of sixteen byte loads.
 * xb->qs is always 2-byte aligned (block_q8_0.d is a half).
 */
void dequantize_q8_0_pairs(device const block_q8_0 *xb, short il, thread half4x4 & reg) {
    device const ushort *qs16 = (device const ushort *)(xb->qs + 16*il);
    const float d = xb->d;

    float4x4 reg_f;

    FOR_UNROLL (short i = 0; i < 8; i++) {
        const ushort u = qs16[i];
        reg_f[i/2][(i%2)*2 + 0] = ((float)(int8_t)(u & 0xFF)) * d;
        reg_f[i/2][(i%2)*2 + 1] = ((float)(int8_t)(u >> 8)) * d;
    }

    reg = (half4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    for (int i = 0; i < 4; i++) {
        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
    }
}

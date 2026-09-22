struct shs_metal_args_cpy {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Typed copy/conversion between graph tensors. shiso uses this for layout
// materialization and F32/F16 conversions at graph boundaries such as KV/cache
// packing and compressor pooling.
template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant shs_metal_args_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;
// Host-visible copy/conversion variants used by the graph.
template [[host_name("kernel_cpy_f32_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<float, float>;
template [[host_name("kernel_cpy_f32_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<float, half>;
template [[host_name("kernel_cpy_f16_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<half, float>;
template [[host_name("kernel_cpy_f16_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<half, half>;

// Contiguous 1D conversions avoid the generic tensor-index reconstruction
// above. Packed vector types retain scalar alignment, so tensor views whose
// offsets are float/half aligned do not need additional 16/8-byte alignment.
// The final vector is converted element-by-element when n is not divisible by
// four, preserving the generic kernel's bounds and conversion semantics.
kernel void kernel_cpy_contig_f32_f16_4(
        constant uint & n,
        device const packed_float4 * src,
        device       packed_half4  * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        const float4 value = float4(src[gid]);
        dst[gid] = packed_half4(half4(value));
        return;
    }

    device const float * src_scalar = (device const float *)src;
    device       half  * dst_scalar = (device       half  *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = half(src_scalar[i + lane]);
    }
}

kernel void kernel_cpy_contig_f16_f32_4(
        constant uint & n,
        device const packed_half4  * src,
        device       packed_float4 * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        const half4 value = half4(src[gid]);
        dst[gid] = packed_float4(float4(value));
        return;
    }

    device const half  * src_scalar = (device const half  *)src;
    device       float * dst_scalar = (device       float *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = float(src_scalar[i + lane]);
    }
}

// Bitwise F16 transport for cache staging. Use ushort rather than half so NaN
// payloads and every other binary16 encoding pass through unchanged.
kernel void kernel_cpy_contig_f16_f16_bits_4(
        constant uint & n,
        device const packed_ushort4 * src,
        device       packed_ushort4 * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        dst[gid] = src[gid];
        return;
    }

    device const ushort * src_scalar = (device const ushort *)src;
    device       ushort * dst_scalar = (device       ushort *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = src_scalar[i + lane];
    }
}

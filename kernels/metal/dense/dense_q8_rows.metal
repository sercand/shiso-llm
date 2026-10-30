// A `q8_0` projection over one to eight activation rows - a decode step, a
// speculative verify pass, a small decode batch - reading every weight block
// once for all of them. One row takes it too, so a verified row and the
// serial step it checks are summed by the same code in the same order.
//
// The generic one-row kernel re-reads the whole matrix for each row and the
// extended family's row blocking reached about 36 GB/s on `q8_0` (measured on
// an M4: the 131072-row vocabulary projection over four rows took 10.4 ms
// against 3.7 ms for one). Here four lanes share a block, eight values each,
// so a simdgroup sweeps eight blocks a step; each simdgroup owns two output
// rows and every lane holds its eight activations of every row in registers.
// A row's sum runs in the same order whatever `R` is, so a row's answer does
// not depend on how many rows share the launch.

// Compiled safe: under the library's fast-math default the compiler may
// reassociate the per-row sums differently in each row-count instantiation,
// and a verified row would then not sum as the one-row step it checks. The
// build default is restored at the end of the file.
#pragma METAL fp math_mode(safe)

/// One `q8_0` block as stored: the struct tells the compiler the quants are
/// two-byte aligned, which a byte pointer into the row does not.
struct dense_q8_block {
    half d;
    char qs[32];
};

/// `dense_args_q8_rows`.
struct dense_args_q8_rows {
    uint32_t in_dim;
    uint32_t out_dim;
};

/// Output rows one threadgroup owns.
constant constexpr uint DENSE_Q8_ROWS_NR0 = 2u;

/// Simdgroups that split a row's blocks between them.
constant constexpr uint DENSE_Q8_ROWS_NSG = 4u;

/// `y[r][o] = sum_k W[o][k] x[r][k]` for `R` activation rows.
///
/// A threadgroup owns two output rows; its four simdgroups take every fourth
/// eight-block step of them, four lanes to a block, and their partials meet
/// in threadgroup memory, added in simdgroup order. Every order involved is a
/// function of the shape alone.
///
/// Buffers: args, w (`[out_dim][in_dim/32]` `q8_0` blocks), x `[R][in_dim]`,
/// y `[R][out_dim]`. Grid `ceil(out_dim / 2)`, 128 threads.
template <short R>
kernel void kernel_dense_q8_0_rows_f32(
        constant dense_args_q8_rows & args [[buffer(0)]],
        device const uchar * w [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        uint tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NR0 = DENSE_Q8_ROWS_NR0;
    constexpr uint NSG = DENSE_Q8_ROWS_NSG;
    constexpr uint BLOCK_BYTES = 34u;
    threadgroup float partial[NSG * NR0 * 8];
    const uint blocks = args.in_dim / 32u;
    const uint row0 = tg * NR0;
    const uint ix = lane / 4u;
    const uint il = lane % 4u;
    device const dense_q8_block * wr[NR0];
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < NR0; ++r) {
        wr[r] = (device const dense_q8_block *) (w
                + (uint64_t) min(row0 + r, args.out_dim - 1u) * blocks * BLOCK_BYTES);
    }
    float acc[NR0][R];
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < NR0; ++r) {
        _Pragma("clang loop unroll(full)")
        for (short t = 0; t < R; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    for (uint b = sg * 8u + ix; b < blocks; b += NSG * 8u) {
        float xl[R][8];
        _Pragma("clang loop unroll(full)")
        for (short t = 0; t < R; ++t) {
            device const float4 * xp =
                    (device const float4 *) (x + (uint64_t) t * args.in_dim + b * 32u + il * 8u);
            const float4 a = xp[0];
            const float4 c = xp[1];
            xl[t][0] = a.x; xl[t][1] = a.y; xl[t][2] = a.z; xl[t][3] = a.w;
            xl[t][4] = c.x; xl[t][5] = c.y; xl[t][6] = c.z; xl[t][7] = c.w;
        }
        _Pragma("clang loop unroll(full)")
        for (uint r = 0; r < NR0; ++r) {
            device const dense_q8_block * blk = wr[r] + b;
            const float d = float(blk->d);
            float q[8];
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 8; ++i) {
                q[i] = float(blk->qs[il * 8u + i]);
            }
            _Pragma("clang loop unroll(full)")
            for (short t = 0; t < R; ++t) {
                float s = 0.0f;
                _Pragma("clang loop unroll(full)")
                for (short i = 0; i < 8; ++i) {
                    s += q[i] * xl[t][i];
                }
                acc[r][t] += s * d;
            }
        }
    }
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < NR0; ++r) {
        _Pragma("clang loop unroll(full)")
        for (short t = 0; t < R; ++t) {
            const float v = dense_warp_sum(acc[r][t]);
            if (lane == 0) {
                partial[(sg * NR0 + r) * 8u + uint(t)] = v;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NR0 * uint(R)) {
        const uint r = tid / uint(R);
        const uint t = tid % uint(R);
        float v = 0.0f;
        for (uint g = 0; g < NSG; ++g) {
            v += partial[(g * NR0 + r) * 8u + t];
        }
        if (row0 + r < args.out_dim) {
            y[(uint64_t) t * args.out_dim + row0 + r] = v;
        }
    }
}

template [[host_name("kernel_dense_q8_0_rows_f32_r1")]] kernel void kernel_dense_q8_0_rows_f32<1>(constant dense_args_q8_rows &, device const uchar *, device const float *, device float *, uint, ushort, ushort, ushort);
template [[host_name("kernel_dense_q8_0_rows_f32_r2")]] kernel void kernel_dense_q8_0_rows_f32<2>(constant dense_args_q8_rows &, device const uchar *, device const float *, device float *, uint, ushort, ushort, ushort);
template [[host_name("kernel_dense_q8_0_rows_f32_r3")]] kernel void kernel_dense_q8_0_rows_f32<3>(constant dense_args_q8_rows &, device const uchar *, device const float *, device float *, uint, ushort, ushort, ushort);
template [[host_name("kernel_dense_q8_0_rows_f32_r4")]] kernel void kernel_dense_q8_0_rows_f32<4>(constant dense_args_q8_rows &, device const uchar *, device const float *, device float *, uint, ushort, ushort, ushort);
template [[host_name("kernel_dense_q8_0_rows_f32_r5")]] kernel void kernel_dense_q8_0_rows_f32<5>(constant dense_args_q8_rows &, device const uchar *, device const float *, device float *, uint, ushort, ushort, ushort);
template [[host_name("kernel_dense_q8_0_rows_f32_r6")]] kernel void kernel_dense_q8_0_rows_f32<6>(constant dense_args_q8_rows &, device const uchar *, device const float *, device float *, uint, ushort, ushort, ushort);
template [[host_name("kernel_dense_q8_0_rows_f32_r7")]] kernel void kernel_dense_q8_0_rows_f32<7>(constant dense_args_q8_rows &, device const uchar *, device const float *, device float *, uint, ushort, ushort, ushort);
template [[host_name("kernel_dense_q8_0_rows_f32_r8")]] kernel void kernel_dense_q8_0_rows_f32<8>(constant dense_args_q8_rows &, device const uchar *, device const float *, device float *, uint, ushort, ushort, ushort);

#pragma METAL fp math_mode(fast)

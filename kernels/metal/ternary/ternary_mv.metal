// Matrix-vector products over ternary and 2-bit affine weights, one
// activation row a threadgroup (the engine sends two rows and up to the
// matmuls, so only `T = 1` is instantiated).
//
// Three storage formats, one arithmetic. Every weight is `scale * c + bias`
// with one scale and one bias per 128 consecutive inputs of a row:
//
// - PQ2_0: 34-byte blocks `[f16 scale][32 bytes of 2-bit codes]`, bias = -scale;
// - MLX 2-bit: `u32` words of sixteen 2-bit codes, `f16` scale and bias arrays;
// - PTQ1_0: 28-byte blocks `[24 bytes of 5-trit bytes][2 bytes of 4-trit
//   bytes][f16 scale]`, bias = -scale; see `shiso_modelfile::ternary` for
//   which element each trit is.
//
// Each group sums as `scale * sum(c x) + bias * sum(x)`. The 2-bit bodies of
// PQ2_0 and MLX 2-bit are the same code reading the same codes in the same
// order, so given a file and its lossless re-encoding they produce the same
// bits.
//
// Geometry, after `q4e_affine_mv_kernel`: a threadgroup is `64 * splits`
// threads; simdgroup `i` owns output rows `8 * group.x + 4 * (i % 2) ..+ 4` over
// the `(i / 2)`-th of `splits` interleaved slices of the contraction; a sweep
// of one simdgroup covers four 128-input groups, eight lanes a group. The
// slices meet in threadgroup memory and are added in slice order.

/// Inputs one scale covers.
constant constexpr uint TERNARY_GROUP = 128u;
/// Output rows one threadgroup covers.
constant constexpr uint TERNARY_ROWS = 8u;
/// Output rows one simdgroup carries.
constant constexpr uint TERNARY_SG_ROWS = 4u;
/// Inputs a simdgroup sweeps at once: four groups.
constant constexpr uint TERNARY_SWEEP = 4u * TERNARY_GROUP;
/// Largest `splits` a launch may carry (1024 threads).
constant constexpr uint TERNARY_MAX_SPLITS = 16u;

/// Storage format tags.
constant constexpr uint TERNARY_PQ2 = 0u;
constant constexpr uint TERNARY_MLX2 = 1u;
constant constexpr uint TERNARY_PTQ1 = 2u;

/// Sixteen activations at `x`, prescaled so code `j` of a 16-bit half-word can
/// be masked out in place: element `8h + j` is divided by `4^j`, which is
/// exact. Returns their plain sum.
static inline float ternary_load_x16(device const float *x, thread float (&xt)[16]) {
    device const float4 *xv = (device const float4 *) x;
    float sum = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
        const float4 v = xv[i];
        sum += (v.x + v.y) + (v.z + v.w);
        const uint j = (4u * i) % 8u;
        xt[4u * i + 0u] = v.x * exp2(-2.0f * float(j + 0u));
        xt[4u * i + 1u] = v.y * exp2(-2.0f * float(j + 1u));
        xt[4u * i + 2u] = v.z * exp2(-2.0f * float(j + 2u));
        xt[4u * i + 3u] = v.w * exp2(-2.0f * float(j + 3u));
    }
    return sum;
}

/// The dot of sixteen 2-bit codes (two half-words, low first) with
/// activations from `ternary_load_x16`.
static inline float ternary_qdot16(ushort lo, ushort hi, thread const float (&xt)[16]) {
    float acc = 0.0f;
    for (uint j = 0; j < 8u; ++j) {
        const ushort mask = ushort(3u << (2u * j));
        acc += xt[j] * float(lo & mask) + xt[8u + j] * float(hi & mask);
    }
    return acc;
}

/// Trit `n` of a packed byte: its base-3 digits, most significant first.
static inline float ternary_trit(uint byte, uint n) {
    const uint pow3[5] = {1u, 3u, 9u, 27u, 81u};
    return float((((byte * pow3[n]) & 255u) * 3u) >> 8u);
}

/// One lane's share of a PTQ1_0 block against one activation row: slot `j`
/// (0..8) of the block owns stage-one bytes `2j, 2j + 1`, stage-two byte `j`
/// and, for `j < 2`, tail byte `j`. Returns `sum(c x)`, with `sum(x)` of the
/// same elements in `xsum`.
static inline float ternary_ptq1_slot(device const uchar *blk, device const float *x, uint j,
                                      thread float &xsum) {
    float acc = 0.0f;
    float sum = 0.0f;
    const uint b0 = blk[2u * j];
    const uint b1 = blk[2u * j + 1u];
    for (uint n = 0; n < 5u; ++n) {
        const float2 v = *(device const float2 *) (x + 16u * n + 2u * j);
        acc += v.x * ternary_trit(b0, n) + v.y * ternary_trit(b1, n);
        sum += v.x + v.y;
    }
    const uint b2 = blk[16u + j];
    for (uint n = 0; n < 5u; ++n) {
        const float v = x[80u + 8u * n + j];
        acc += v * ternary_trit(b2, n);
        sum += v;
    }
    if (j < 2u) {
        const uint b3 = blk[24u + j];
        for (uint n = 0; n < 4u; ++n) {
            const float v = x[120u + 2u * n + j];
            acc += v * ternary_trit(b3, n);
            sum += v;
        }
    }
    xsum = sum;
    return acc;
}

/// `R` rows against `T` activation rows, one lane's share; see the file
/// comment for which inputs a lane reads. `w` points at row 0 of the `R` in the
/// format's own addressing, `s` and `b` likewise (MLX only).
template <uint FMT, uint R, uint T>
static inline void ternary_rows(device const uchar *w, device const half *s,
                                device const half *b, device const float *x, uint in_dim,
                                uint rows_out, uint rows_here, uint lane, uint split,
                                uint splits, thread float (&acc)[R][T]) {
    const uint groups = in_dim / TERNARY_GROUP;
    const ulong row_bytes = FMT == TERNARY_PQ2    ? ulong(groups) * 34u
                            : FMT == TERNARY_PTQ1 ? ulong(groups) * 28u
                                                  : ulong(in_dim) / 4u;
    for (uint r = 0; r < R; ++r) {
        for (uint t = 0; t < T; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    const uint slot = lane % 8u;
    for (uint k0 = split * TERNARY_SWEEP; k0 < in_dim; k0 += TERNARY_SWEEP * splits) {
        const uint g = k0 / TERNARY_GROUP + lane / 8u;
        if (g >= groups) {
            continue;
        }
        float sc[R];
        float bi[R];
        ushort lo[R];
        ushort hi[R];
        device const uchar *blk[R];
        for (uint r = 0; r < R; ++r) {
            if (r >= rows_out) {
                continue;
            }
            device const uchar *row = w + ulong(r) * row_bytes;
            if (FMT == TERNARY_PQ2) {
                device const ushort *p = (device const ushort *) (row + ulong(g) * 34u);
                sc[r] = float(as_type<half>(p[0]));
                bi[r] = -sc[r];
                lo[r] = p[1u + 2u * slot];
                hi[r] = p[2u + 2u * slot];
            } else if (FMT == TERNARY_MLX2) {
                device const ushort *p = (device const ushort *) (row + ulong(g) * 32u);
                sc[r] = float(s[ulong(r) * groups + g]);
                bi[r] = float(b[ulong(r) * groups + g]);
                lo[r] = p[2u * slot];
                hi[r] = p[2u * slot + 1u];
            } else {
                blk[r] = row + ulong(g) * 28u;
                sc[r] = float(as_type<half>(*(device const ushort *) (blk[r] + 26u)));
                bi[r] = -sc[r];
            }
        }
        for (uint t = 0; t < T; ++t) {
            if (t >= rows_here) {
                continue;
            }
            device const float *xg = x + ulong(t) * in_dim + ulong(g) * TERNARY_GROUP;
            if (FMT == TERNARY_PTQ1) {
                for (uint r = 0; r < R; ++r) {
                    if (r < rows_out) {
                        float sum;
                        const float dot = ternary_ptq1_slot(blk[r], xg, slot, sum);
                        acc[r][t] += sc[r] * dot + bi[r] * sum;
                    }
                }
            } else {
                float xt[16];
                const float sum = ternary_load_x16(xg + 16u * slot, xt);
                for (uint r = 0; r < R; ++r) {
                    if (r < rows_out) {
                        acc[r][t] += sc[r] * ternary_qdot16(lo[r], hi[r], xt) + bi[r] * sum;
                    }
                }
            }
        }
    }
}

/// The whole projection for one threadgroup; see the file comment.
template <uint FMT, uint T>
static inline void ternary_project(constant ternary_args_mv &args, device const uchar *w,
                                   device const half *scales, device const half *biases,
                                   device const float *x, device float *y, uint2 group,
                                   uint lane, uint simd, threadgroup float *part) {
    const uint in_dim = args.in_dim;
    const uint out_dim = args.out_dim;
    const uint r0 = group.y * T;
    if (r0 >= args.n_tok) {
        return;
    }
    const uint splits = args.splits;
    const uint split = simd / 2u;
    const uint sub = (simd % 2u) * TERNARY_SG_ROWS;
    const uint rows_here = min(T, args.n_tok - r0);
    const uint groups = in_dim / TERNARY_GROUP;
    const uint o = group.x * TERNARY_ROWS + sub;
    const uint rows_out = o < out_dim ? min(TERNARY_SG_ROWS, out_dim - o) : 0u;
    const ulong row_bytes = FMT == TERNARY_PQ2    ? ulong(groups) * 34u
                            : FMT == TERNARY_PTQ1 ? ulong(groups) * 28u
                                                  : ulong(in_dim) / 4u;
    float acc[TERNARY_SG_ROWS][T];
    ternary_rows<FMT, TERNARY_SG_ROWS, T>(w + ulong(o) * row_bytes,
                                          scales + ulong(o) * groups,
                                          biases + ulong(o) * groups, x + ulong(r0) * in_dim,
                                          in_dim, rows_out, rows_here, lane, split, splits, acc);
    for (uint r = 0; r < TERNARY_SG_ROWS; ++r) {
        for (uint t = 0; t < T; ++t) {
            const float v = simd_sum(acc[r][t]);
            if (splits == 1u) {
                if (lane == 0u && t < rows_here && r < rows_out) {
                    y[ulong(r0 + t) * out_dim + o + r] = v;
                }
            } else if (lane == 0u) {
                part[(split * TERNARY_ROWS + sub + r) * T + t] = v;
            }
        }
    }
    if (splits == 1u) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tid = simd * 32u + lane;
    if (tid < TERNARY_ROWS * T) {
        const uint r = tid / T;
        const uint t = tid % T;
        const uint oo = group.x * TERNARY_ROWS + r;
        if (oo < out_dim && t < rows_here) {
            float v = 0.0f;
            for (uint p = 0; p < splits; ++p) {
                v += part[(p * TERNARY_ROWS + r) * T + t];
            }
            y[ulong(r0 + t) * out_dim + oo] = v;
        }
    }
}

template <uint FMT, uint T>
kernel void kernel_ternary_mv(
        constant ternary_args_mv & args [[buffer(0)]],
        device const uchar * w          [[buffer(1)]],
        device const half  * scales     [[buffer(2)]],
        device const half  * biases     [[buffer(3)]],
        device const float * x          [[buffer(4)]],
        device       float * y          [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint  lane  [[thread_index_in_simdgroup]],
        uint  simd  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[TERNARY_MAX_SPLITS * TERNARY_ROWS * T];
    ternary_project<FMT, T>(args, w, scales, biases, x, y, group, lane, simd, part);
}

#define TERNARY_MV(name, fmt, t)                                                                  \
    template [[host_name("kernel_ternary_mv_" #name "_f32_t" #t)]] kernel void                   \
    kernel_ternary_mv<fmt, t>(constant ternary_args_mv &, device const uchar *,                  \
                              device const half *, device const half *, device const float *,    \
                              device float *, uint2, uint, uint);

TERNARY_MV(pq2_0, TERNARY_PQ2, 1)
TERNARY_MV(mlx2, TERNARY_MLX2, 1)
TERNARY_MV(ptq1_0, TERNARY_PTQ1, 1)

/// One weight row per token id, decoded to f32: the embedding gather.
///
/// Grid `(ceil(in_dim / 128), n_tok)`, 128 threads: thread `i` of group `x`
/// writes element `128 x + i` of the row `ids[y]` names.
template <uint FMT>
kernel void kernel_ternary_get_rows(
        constant ternary_args_rows & args [[buffer(0)]],
        device const uchar * w            [[buffer(1)]],
        device const half  * scales       [[buffer(2)]],
        device const half  * biases       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * y            [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint2 tid2  [[thread_position_in_threadgroup]]) {
    const uint i = tid2.x;
    const uint g = group.x;
    const uint groups = args.in_dim / TERNARY_GROUP;
    const ulong id = ulong(uint(ids[group.y]));
    float scale;
    float bias;
    uint code;
    if (FMT == TERNARY_PQ2) {
        device const uchar *blk = w + (id * groups + g) * 34u;
        scale = float(as_type<half>(*(device const ushort *) blk));
        bias = -scale;
        code = (blk[2u + i / 4u] >> (2u * (i % 4u))) & 3u;
    } else if (FMT == TERNARY_MLX2) {
        device const uchar *blk = w + (id * groups + g) * 32u;
        scale = float(scales[id * groups + g]);
        bias = float(biases[id * groups + g]);
        code = (blk[i / 4u] >> (2u * (i % 4u))) & 3u;
    } else {
        device const uchar *blk = w + (id * groups + g) * 28u;
        scale = float(as_type<half>(*(device const ushort *) (blk + 26u)));
        bias = -scale;
        if (i < 80u) {
            code = uint(ternary_trit(blk[i % 16u], i / 16u));
        } else if (i < 120u) {
            code = uint(ternary_trit(blk[16u + (i - 80u) % 8u], (i - 80u) / 8u));
        } else {
            code = uint(ternary_trit(blk[24u + (i - 120u) % 2u], (i - 120u) / 2u));
        }
    }
    y[ulong(group.y) * args.in_dim + ulong(g) * TERNARY_GROUP + i] = scale * float(code) + bias;
}

template [[host_name("kernel_ternary_get_rows_pq2_0_f32")]] kernel void
kernel_ternary_get_rows<TERNARY_PQ2>(constant ternary_args_rows &, device const uchar *,
                                     device const half *, device const half *,
                                     device const int *, device float *, uint2, uint2);
template [[host_name("kernel_ternary_get_rows_mlx2_f32")]] kernel void
kernel_ternary_get_rows<TERNARY_MLX2>(constant ternary_args_rows &, device const uchar *,
                                      device const half *, device const half *,
                                      device const int *, device float *, uint2, uint2);
template [[host_name("kernel_ternary_get_rows_ptq1_0_f32")]] kernel void
kernel_ternary_get_rows<TERNARY_PTQ1>(constant ternary_args_rows &, device const uchar *,
                                      device const half *, device const half *,
                                      device const int *, device float *, uint2, uint2);

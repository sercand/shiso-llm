// Matrix-matrix products over ternary and 2-bit affine weights, for passes of
// two rows and up: a verify window, a prompt chunk.
//
// The matvec in `ternary_mv.metal` spends about three operations a weight a
// token (mask, convert, multiply-add), so its cost is linear in the rows a
// pass carries: eight rows cost eight one-row passes. Here each weight is
// decoded once a threadgroup, into threadgroup memory, and the products run
// on the simdgroup matrix units, eight tokens to an instruction.
//
// A threadgroup is four simdgroups and covers 32 output rows by `TN` tokens.
// Per 64-input step it stages the decoded weights `W[32][64]` - exact in f32:
// `scale * code + bias` with an f16 scale, a code in 0..3 and the bias minus
// the scale - and the activations `X[TN][64]`; simdgroup `i` then multiplies
// its eight rows against every 8-token block, accumulating in f32. Rows and
// tokens past the pass are staged as zeros and never stored.
//
// Buffers are the matvec's, in its order; the argument block is
// `ternary_args_mv` (its `splits` is ignored).

/// Output rows one threadgroup covers.
constant constexpr uint TERNARY_MM_ROWS = 32u;
/// Inputs staged per step.
constant constexpr uint TERNARY_MM_K = 64u;

/// The code of input `e` (0..127) of a `PTQ1_0` block.
static inline uint ternary_ptq1_code(device const uchar *blk, uint e) {
    if (e < 80u) {
        return uint(ternary_trit(blk[e % 16u], e / 16u));
    }
    if (e < 120u) {
        return uint(ternary_trit(blk[16u + (e - 80u) % 8u], (e - 80u) / 8u));
    }
    return uint(ternary_trit(blk[24u + (e - 120u) % 2u], (e - 120u) / 2u));
}

template <uint FMT, uint TN>
kernel void kernel_ternary_mm(
        constant ternary_args_mv & args [[buffer(0)]],
        device const uchar * w          [[buffer(1)]],
        device const half  * scales     [[buffer(2)]],
        device const half  * biases     [[buffer(3)]],
        device const float * x          [[buffer(4)]],
        device       float * y          [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]],
        uint  simd  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float wt[TERNARY_MM_ROWS * TERNARY_MM_K];
    threadgroup float xt[TN * TERNARY_MM_K];

    const uint in_dim = args.in_dim;
    const uint out_dim = args.out_dim;
    const uint groups = in_dim / TERNARY_GROUP;
    const ulong row_bytes = FMT == TERNARY_PQ2    ? ulong(groups) * 34u
                            : FMT == TERNARY_PTQ1 ? ulong(groups) * 28u
                                                  : ulong(in_dim) / 4u;
    const uint o0 = group.x * TERNARY_MM_ROWS;
    const uint t0 = group.y * TN;

    simdgroup_matrix<float, 8, 8> acc[TN / 8u];
    for (uint j = 0; j < TN / 8u; ++j) {
        acc[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // Thread `tid` of 128 decodes sixteen consecutive inputs of one row.
    const uint wr = tid / 4u;
    const uint wk = (tid % 4u) * 16u;
    const uint row = o0 + wr;

    for (uint k0 = 0; k0 < in_dim; k0 += TERNARY_MM_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint k = k0 + wk;
            const uint g = k / TERNARY_GROUP;
            const uint e0 = k % TERNARY_GROUP;
            threadgroup float *dst = wt + wr * TERNARY_MM_K + wk;
            if (row >= out_dim) {
                for (uint i = 0; i < 16u; ++i) {
                    dst[i] = 0.0f;
                }
            } else if (FMT == TERNARY_PTQ1) {
                device const uchar *blk = w + ulong(row) * row_bytes + ulong(g) * 28u;
                const float sc = float(as_type<half>(*(device const ushort *) (blk + 26u)));
                for (uint i = 0; i < 16u; ++i) {
                    dst[i] = sc * float(ternary_ptq1_code(blk, e0 + i)) - sc;
                }
            } else {
                float sc;
                float bi;
                uint word;
                if (FMT == TERNARY_PQ2) {
                    device const ushort *p =
                            (device const ushort *) (w + ulong(row) * row_bytes + ulong(g) * 34u);
                    sc = float(as_type<half>(p[0]));
                    bi = -sc;
                    word = uint(p[1u + e0 / 8u]) | (uint(p[2u + e0 / 8u]) << 16u);
                } else {
                    device const uint *p =
                            (device const uint *) (w + ulong(row) * row_bytes + ulong(g) * 32u);
                    sc = float(scales[ulong(row) * groups + g]);
                    bi = float(biases[ulong(row) * groups + g]);
                    word = p[e0 / 16u];
                }
                for (uint i = 0; i < 16u; ++i) {
                    dst[i] = sc * float((word >> (2u * i)) & 3u) + bi;
                }
            }
        }
        for (uint i = tid; i < TN * (TERNARY_MM_K / 4u); i += 128u) {
            const uint t = i / (TERNARY_MM_K / 4u);
            const uint q = i % (TERNARY_MM_K / 4u);
            float4 v = float4(0.0f);
            if (t0 + t < args.n_tok) {
                v = *(device const float4 *) (x + ulong(t0 + t) * in_dim + k0 + 4u * q);
            }
            *(threadgroup float4 *) (xt + t * TERNARY_MM_K + 4u * q) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TERNARY_MM_K; kk += 8u) {
            simdgroup_matrix<float, 8, 8> a;
            simdgroup_load(a, wt + simd * 8u * TERNARY_MM_K + kk, TERNARY_MM_K);
            for (uint j = 0; j < TN / 8u; ++j) {
                simdgroup_matrix<float, 8, 8> b;
                // B[k][t] = X[t][k]: the staged rows read transposed.
                simdgroup_load(b, xt + j * 8u * TERNARY_MM_K + kk, TERNARY_MM_K, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
            }
        }
    }

    // acc[j][r][t] is row `o0 + 8 simd + r`, token `t0 + 8 j + t`. Staged
    // through the weight tile, which is free now, so a partial tile stores
    // only what exists.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *out = wt;
    for (uint j = 0; j < TN / 8u; ++j) {
        simdgroup_store(acc[j], out + simd * 8u * TN + 8u * j, TN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < TERNARY_MM_ROWS * TN; i += 128u) {
        const uint r = i / TN;
        const uint t = i % TN;
        if (o0 + r < out_dim && t0 + t < args.n_tok) {
            y[ulong(t0 + t) * out_dim + o0 + r] = out[r * TN + t];
        }
    }
}

#define TERNARY_MM(name, fmt, tn)                                                                 \
    template [[host_name("kernel_ternary_mm_" #name "_f32_n" #tn)]] kernel void                  \
    kernel_ternary_mm<fmt, tn>(constant ternary_args_mv &, device const uchar *,                 \
                               device const half *, device const half *, device const float *,   \
                               device float *, uint2, uint, uint);

TERNARY_MM(pq2_0, TERNARY_PQ2, 8)
TERNARY_MM(pq2_0, TERNARY_PQ2, 32)
TERNARY_MM(mlx2, TERNARY_MLX2, 8)
TERNARY_MM(mlx2, TERNARY_MLX2, 32)
TERNARY_MM(ptq1_0, TERNARY_PTQ1, 8)
TERNARY_MM(ptq1_0, TERNARY_PTQ1, 32)

// ---------------------------------------------------------------------------
// Two to eight rows: the matrix units fed from registers
// ---------------------------------------------------------------------------
//
// A verify window is too narrow for the staged tile above to pay for its
// threadgroup traffic, and too wide for the matvec, which spends three
// operations a weight a token. Here a weight is decoded straight into the
// lane that holds it in an 8x8 fragment - two inputs of one row - and one
// multiply-accumulate then serves all eight tokens.
//
// Fragment layout (8x8, f32): lane `l` holds elements (fm, fn) and
// (fm, fn + 1), fm = (q & 4) + ((l / 2) % 4), fn = (q & 2) * 2 + (l % 2) * 2,
// q = l / 4. A is W (rows by inputs), B is X transposed (inputs by tokens).
//
// A threadgroup is four simdgroups over the same 32 rows; simdgroup `s` takes
// the 128-input groups `g` with `g % 4 == s` and the four partial tiles meet
// in threadgroup memory, added in simdgroup order.

/// Row tiles (of eight) one simdgroup of the register kernel carries.
constant constexpr uint TERNARY_MMV_TILES = 4u;

template <uint FMT>
kernel void kernel_ternary_mmv(
        constant ternary_args_mv & args [[buffer(0)]],
        device const uchar * w          [[buffer(1)]],
        device const half  * scales     [[buffer(2)]],
        device const half  * biases     [[buffer(3)]],
        device const float * x          [[buffer(4)]],
        device       float * y          [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint  lane  [[thread_index_in_simdgroup]],
        uint  simd  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4u * TERNARY_MMV_TILES * 64u];

    const uint in_dim = args.in_dim;
    const uint out_dim = args.out_dim;
    const uint n_tok = args.n_tok;
    const uint groups = in_dim / TERNARY_GROUP;
    const ulong row_bytes = FMT == TERNARY_PQ2    ? ulong(groups) * 34u
                            : FMT == TERNARY_PTQ1 ? ulong(groups) * 28u
                                                  : ulong(in_dim) / 4u;
    const uint q = lane / 4u;
    const uint fm = (q & 4u) + ((lane / 2u) % 4u);
    const uint fn = (q & 2u) * 2u + (lane % 2u) * 2u;
    const uint o0 = group.x * (8u * TERNARY_MMV_TILES);

    simdgroup_matrix<float, 8, 8> acc[TERNARY_MMV_TILES];
    for (uint rt = 0; rt < TERNARY_MMV_TILES; ++rt) {
        acc[rt] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // B's column fn and fn + 1 are tokens; past the pass they read as zero.
    const bool t0_live = fn < n_tok;
    const bool t1_live = fn + 1u < n_tok;
    device const float *x0 = x + ulong(min(fn, n_tok - 1u)) * in_dim;
    device const float *x1 = x + ulong(min(fn + 1u, n_tok - 1u)) * in_dim;

    for (uint g = simd; g < groups; g += 4u) {
        float sc[TERNARY_MMV_TILES];
        float bi[TERNARY_MMV_TILES];
        device const uchar *blk[TERNARY_MMV_TILES];
        for (uint rt = 0; rt < TERNARY_MMV_TILES; ++rt) {
            const uint row = min(o0 + 8u * rt + fm, out_dim - 1u);
            device const uchar *r = w + ulong(row) * row_bytes;
            if (FMT == TERNARY_PQ2) {
                blk[rt] = r + ulong(g) * 34u;
                sc[rt] = float(as_type<half>(*(device const ushort *) blk[rt]));
                bi[rt] = -sc[rt];
            } else if (FMT == TERNARY_PTQ1) {
                blk[rt] = r + ulong(g) * 28u;
                sc[rt] = float(as_type<half>(*(device const ushort *) (blk[rt] + 26u)));
                bi[rt] = -sc[rt];
            } else {
                blk[rt] = r + ulong(g) * 32u;
                sc[rt] = float(scales[ulong(row) * groups + g]);
                bi[rt] = float(biases[ulong(row) * groups + g]);
            }
        }
        const uint kg = g * TERNARY_GROUP;
        for (uint ks = 0; ks < TERNARY_GROUP; ks += 8u) {
            // B[k][t]: lane holds inputs fm of tokens fn and fn + 1.
            simdgroup_matrix<float, 8, 8> b;
            b.thread_elements()[0] = t0_live ? x0[kg + ks + fm] : 0.0f;
            b.thread_elements()[1] = t1_live ? x1[kg + ks + fm] : 0.0f;
            const uint e = ks + fn;
            for (uint rt = 0; rt < TERNARY_MMV_TILES; ++rt) {
                uint c0;
                uint c1;
                if (FMT == TERNARY_PTQ1) {
                    c0 = ternary_ptq1_code(blk[rt], e);
                    c1 = ternary_ptq1_code(blk[rt], e + 1u);
                } else {
                    const uint base = FMT == TERNARY_PQ2 ? 2u : 0u;
                    const uint byte = blk[rt][base + e / 4u];
                    c0 = (byte >> (2u * (e % 4u))) & 3u;
                    c1 = (byte >> (2u * (e % 4u) + 2u)) & 3u;
                }
                simdgroup_matrix<float, 8, 8> a;
                a.thread_elements()[0] = fma(sc[rt], float(c0), bi[rt]);
                a.thread_elements()[1] = fma(sc[rt], float(c1), bi[rt]);
                simdgroup_multiply_accumulate(acc[rt], a, b, acc[rt]);
            }
        }
    }

    for (uint rt = 0; rt < TERNARY_MMV_TILES; ++rt) {
        simdgroup_store(acc[rt], part + (simd * TERNARY_MMV_TILES + rt) * 64u, 8u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tid = simd * 32u + lane;
    for (uint i = tid; i < TERNARY_MMV_TILES * 64u; i += 128u) {
        const uint rt = i / 64u;
        const uint r = (i % 64u) / 8u;
        const uint t = i % 8u;
        const uint row = o0 + 8u * rt + r;
        if (row < out_dim && t < n_tok) {
            float v = 0.0f;
            for (uint s = 0; s < 4u; ++s) {
                v += part[(s * TERNARY_MMV_TILES + rt) * 64u + r * 8u + t];
            }
            y[ulong(t) * out_dim + row] = v;
        }
    }
}

template [[host_name("kernel_ternary_mmv_pq2_0_f32")]] kernel void
kernel_ternary_mmv<TERNARY_PQ2>(constant ternary_args_mv &, device const uchar *,
                                device const half *, device const half *, device const float *,
                                device float *, uint2, uint, uint);
template [[host_name("kernel_ternary_mmv_mlx2_f32")]] kernel void
kernel_ternary_mmv<TERNARY_MLX2>(constant ternary_args_mv &, device const uchar *,
                                 device const half *, device const half *, device const float *,
                                 device float *, uint2, uint, uint);
template [[host_name("kernel_ternary_mmv_ptq1_0_f32")]] kernel void
kernel_ternary_mmv<TERNARY_PTQ1>(constant ternary_args_mv &, device const uchar *,
                                 device const half *, device const half *, device const float *,
                                 device float *, uint2, uint, uint);

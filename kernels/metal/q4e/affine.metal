// q4e affine entries: the sparse stack's kernels over MLX packed affine
// weights, which have no CUDA twin.
//
// The layout is MLX's, read in place from the checkpoint: a matrix of
// `out_dim` rows is three arrays - `out_dim x in_dim / 8` little-endian `u32`
// words of eight 4-bit codes each (lowest nibble first), and `out_dim x
// in_dim / G` brain-float scales and biases - and element `k` of row `r` is
// `code * scale[r][k / G] + bias[r][k / G]`. A routed expert bank stacks
// `E` such matrices per array.
//
// Every projection sums with the factorisation MLX itself uses,
// `scale * sum(q * x) + bias * sum(x)` per group, then folds the group sums
// across the row in word order inside each lane and across lanes with the
// decreasing-offset fold. One lane reads two words (sixteen codes) per step
// and holds the matching activations in registers across four rows, so a
// simdgroup consumes 256 contiguous bytes of each of four rows at a time.
//
// `q4e_common.h` is concatenated ahead of this file. No fp math pragma.

/// Codes per affine group of the projections here (MLX `group_size`).
constant constexpr uint Q4E_AFFINE_GROUP = 64u;
/// Output rows one projection threadgroup covers.
constant constexpr uint Q4E_AFFINE_ROWS = 8u;
/// Output rows one simdgroup carries; a 64-thread threadgroup is two of them.
constant constexpr uint Q4E_AFFINE_SG_ROWS = 4u;
/// Codes one lane reads per sweep: two words, one 8-byte load.
constant constexpr uint Q4E_AFFINE_LANE_CODES = 16u;
/// Codes a simdgroup sweeps at once.
constant constexpr uint Q4E_AFFINE_SWEEP = 32u * Q4E_AFFINE_LANE_CODES;
/// Activation rows one multi-row projection threadgroup carries.
constant constexpr uint Q4E_AFFINE_TOKENS = 8u;

/// Sixteen activations at `x`, prescaled so a code can be masked out of its
/// 16-bit quarter of a word without a shift: element `4i + j` is divided by
/// `16^j`, which is exact. Returns their plain sum.
static inline float q4e_affine_load_x(device const float *x, thread float (&xt)[16]) {
    device const float4 *xv = (device const float4 *)x;
    float sum = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
        const float4 v = xv[i];
        sum += (v.x + v.y) + (v.z + v.w);
        xt[4u * i + 0u] = v.x;
        xt[4u * i + 1u] = v.y * (1.0f / 16.0f);
        xt[4u * i + 2u] = v.z * (1.0f / 256.0f);
        xt[4u * i + 3u] = v.w * (1.0f / 4096.0f);
    }
    return sum;
}

/// The dot of sixteen codes (two words) with prescaled activations.
static inline float q4e_affine_qdot(uint2 w, thread const float (&xt)[16]) {
    const ushort q[4] = {ushort(w.x & 0xFFFFu), ushort(w.x >> 16u), ushort(w.y & 0xFFFFu),
                         ushort(w.y >> 16u)};
    float acc = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
        acc += xt[4u * i + 0u] * float(q[i] & 0x000Fu) + xt[4u * i + 1u] * float(q[i] & 0x00F0u) +
               xt[4u * i + 2u] * float(q[i] & 0x0F00u) + xt[4u * i + 3u] * float(q[i] & 0xF000u);
    }
    return acc;
}

/// `R` packed rows against `T` activation rows, one lane's share.
///
/// Lane `l` of K-split `p` (of `splits`) reads codes
/// `[16 (32 p + l) + 512 splits s, ... + 16)` of every row on sweep `s`,
/// holds those activations in registers across the `R` rows, and adds
/// `scale * sum(q x) + bias * sum(x)` per sweep. The order a row is summed in
/// depends on the shape and the split alone - never on `R`, `T` or how many
/// rows share the pass. `acc[r][t]` receives lane partials; the caller folds
/// them. `w`, `s`, `b` point at row 0 of the `R`; rows past `rows_out` and
/// activation rows past `rows_here` are skipped.
template <uint R, uint T>
static inline void q4e_affine_rows(device const uint *w, device const ushort *s,
                                   device const ushort *b, device const float *x,
                                   uint in_dim, uint rows_out, uint rows_here, uint lane,
                                   uint split, uint splits, thread float (&acc)[R][T]) {
    const uint words = in_dim / 8u;
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    for (uint r = 0; r < R; ++r) {
        for (uint t = 0; t < T; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    for (uint k = (split * 32u + lane) * Q4E_AFFINE_LANE_CODES; k < in_dim;
         k += Q4E_AFFINE_SWEEP * splits) {
        uint2 wv[R];
        float sc[R];
        float bi[R];
        const uint g = k / Q4E_AFFINE_GROUP;
        for (uint r = 0; r < R; ++r) {
            if (r < rows_out) {
                wv[r] = *(device const uint2 *)(w + ulong(r) * words + k / 8u);
                sc[r] = q4e_bf16(s[ulong(r) * groups + g]);
                bi[r] = q4e_bf16(b[ulong(r) * groups + g]);
            }
        }
        for (uint t = 0; t < T; ++t) {
            if (t < rows_here) {
                float xt[16];
                const float sum = q4e_affine_load_x(x + ulong(t) * in_dim + k, xt);
                for (uint r = 0; r < R; ++r) {
                    if (r < rows_out) {
                        acc[r][t] += sc[r] * q4e_affine_qdot(wv[r], xt) + bi[r] * sum;
                    }
                }
            }
        }
    }
}

/// `out[r][o] = sum_k W[o][k] x[r][k]` over `rows` activation rows.
///
/// Grid `(ceil(out_dim / 8), ceil(rows / T))`, `64 S` threads: simdgroup `i`
/// owns output rows `8 * group.x + 4 (i % 2) ..+ 4` for the `T` activation
/// rows starting at `T * group.y`, over the `(i / 2)`-th of `S` interleaved
/// slices of the contraction. The `S` partials meet in threadgroup memory and
/// are added in slice order. `S` is the host's choice from the shape alone
/// (`affine_splits`), so every pass width sums a row the same way.
///
/// Buffers: out, w, scales, biases, x, in_dim, out_dim, rows.
template <uint T>
static inline void q4e_affine_project(device float *out, device const uint *w,
                                      device const ushort *scales, device const ushort *biases,
                                      device const float *x, uint in_dim, uint out_dim,
                                      uint rows, uint3 group, uint lane, uint simd, uint n_simd,
                                      threadgroup float *part) {
    const uint r0 = group.y * T;
    if (r0 >= rows) {
        return;
    }
    const uint splits = max(1u, n_simd / 2u);
    const uint split = simd / 2u;
    const uint sub = (simd % 2u) * Q4E_AFFINE_SG_ROWS;
    const uint rows_here = min(T, rows - r0);
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    const uint o = group.x * Q4E_AFFINE_ROWS + sub;
    const bool live = o < out_dim && split < splits;
    const uint rows_out = live ? min(Q4E_AFFINE_SG_ROWS, out_dim - o) : 0u;
    float acc[Q4E_AFFINE_SG_ROWS][T];
    q4e_affine_rows<Q4E_AFFINE_SG_ROWS, T>(
        w + ulong(o) * (in_dim / 8u), scales + ulong(o) * groups, biases + ulong(o) * groups,
        x + ulong(r0) * in_dim, in_dim, rows_out, rows_here, lane, split, splits, acc);
    for (uint r = 0; r < Q4E_AFFINE_SG_ROWS; ++r) {
        for (uint t = 0; t < T; ++t) {
            const float v = q4e_warp_sum(acc[r][t]);
            if (splits == 1u) {
                if (lane == 0u && t < rows_here && r < rows_out) {
                    out[ulong(r0 + t) * out_dim + o + r] = v;
                }
            } else if (lane == 0u) {
                part[(split * Q4E_AFFINE_ROWS + sub + r) * T + t] = v;
            }
        }
    }
    if (splits == 1u) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tid = simd * 32u + lane;
    if (tid < Q4E_AFFINE_ROWS * T) {
        const uint r = tid / T;
        const uint t = tid % T;
        const uint oo = group.x * Q4E_AFFINE_ROWS + r;
        if (oo < out_dim && t < rows_here) {
            float v = 0.0f;
            for (uint p = 0; p < splits; ++p) {
                v += part[(p * Q4E_AFFINE_ROWS + r) * T + t];
            }
            out[ulong(r0 + t) * out_dim + oo] = v;
        }
    }
}

/// K-splits one projection threadgroup may carry (1024 threads).
constant constexpr uint Q4E_AFFINE_MAX_SPLITS = 16u;

/// One activation row: a decode step's projections.
kernel void q4e_affine_mv_kernel(
        device float *out [[buffer(0)]],
        device const uint *w [[buffer(1)]],
        device const ushort *scales [[buffer(2)]],
        device const ushort *biases [[buffer(3)]],
        device const float *x [[buffer(4)]],
        constant uint &in_dim [[buffer(5)]],
        constant uint &out_dim [[buffer(6)]],
        constant uint &rows [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint n_simd [[simdgroups_per_threadgroup]]) {
    threadgroup float part[Q4E_AFFINE_MAX_SPLITS * Q4E_AFFINE_ROWS];
    q4e_affine_project<1>(out, w, scales, biases, x, in_dim, out_dim, rows, group, lane, simd,
                          n_simd, part);
}

/// Several activation rows: a prefill chunk's projections, reading each
/// packed word once for `Q4E_AFFINE_TOKENS` rows.
kernel void q4e_affine_mm_kernel(
        device float *out [[buffer(0)]],
        device const uint *w [[buffer(1)]],
        device const ushort *scales [[buffer(2)]],
        device const ushort *biases [[buffer(3)]],
        device const float *x [[buffer(4)]],
        constant uint &in_dim [[buffer(5)]],
        constant uint &out_dim [[buffer(6)]],
        constant uint &rows [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint n_simd [[simdgroups_per_threadgroup]]) {
    threadgroup float part[Q4E_AFFINE_MAX_SPLITS * Q4E_AFFINE_ROWS * Q4E_AFFINE_TOKENS];
    q4e_affine_project<Q4E_AFFINE_TOKENS>(out, w, scales, biases, x, in_dim, out_dim, rows,
                                          group, lane, simd, n_simd, part);
}

/// Activation rows, output rows and contraction codes one tile of the prompt
/// projection covers.
constant constexpr uint Q4E_GEMM_BM = 32u;
constant constexpr uint Q4E_GEMM_BN = 32u;
constant constexpr uint Q4E_GEMM_BK = 32u;
/// Row pitch of a staged tile, padded off the bank stride.
constant constexpr uint Q4E_GEMM_LD = Q4E_GEMM_BK + 4u;

/// One staged 32-code slice of a `BM x BN` tile into the accumulators: the
/// simdgroup at `(sm, sn)` multiplies its 16 x 16 quarter out of 8 x 8
/// fragments, the slice's four fragment steps in order.
template <uint BN, uint NW>
static inline void q4e_gemm_mma(threadgroup const float *xs, threadgroup const float *ws, uint sm,
                                uint sn, thread simdgroup_float8x8 (&acc)[NW][2][2]) {
    for (uint kk = 0; kk < Q4E_GEMM_BK; kk += 8u) {
        simdgroup_float8x8 a[2];
        for (uint i = 0; i < 2u; ++i) {
            simdgroup_load(a[i], xs + (sm + 8u * i) * Q4E_GEMM_LD + kk, Q4E_GEMM_LD);
        }
        for (uint n = 0; n < NW; ++n) {
            simdgroup_float8x8 b[2];
            for (uint j = 0; j < 2u; ++j) {
                simdgroup_load(b[j], ws + (n * BN + sn + 8u * j) * Q4E_GEMM_LD + kk, Q4E_GEMM_LD,
                               ulong2(0, 0), true);
            }
            for (uint i = 0; i < 2u; ++i) {
                for (uint j = 0; j < 2u; ++j) {
                    simdgroup_multiply_accumulate(acc[n][i][j], a[i], b[j], acc[n][i][j]);
                }
            }
        }
    }
}

/// Where thread `tid` stages from in a `BM x BN` tile: activation row
/// `q4e_gemm_xr<BM>(tid)` and weight row `q4e_gemm_wr<BN>(tid)`.
template <uint BM>
static inline uint q4e_gemm_xr(uint tid) {
    return tid / (Q4E_GEMM_BK / (BM * Q4E_GEMM_BK / 128u));
}
template <uint BN>
static inline uint q4e_gemm_wr(uint tid) {
    return tid / (Q4E_GEMM_BK / (BN * Q4E_GEMM_BK / 128u));
}

/// One `BM x BN` output tile of `NW` projections that share their
/// activation rows, accumulated over the whole contraction by four
/// simdgroups of 16 x 16 each: `BM` 32 and `BN` 32 for a dense chunk, `BM` 16
/// and `BN` 64 for an expert's few columns. Thread `tid` stages activation row
/// `q4e_gemm_xr<BM>(tid)` from `xrow` (zeros where `x_live` is false) and
/// weight row `q4e_gemm_wr<BN>(tid)` of each projection from `wrow`, `srow`,
/// `brow` (zeros where `w_live` is false). `xs` holds one staged activation
/// tile, `ws` `NW` weight tiles. The next slice's words and activations are
/// read before this slice's multiplies, so they are in flight while the
/// matrix units work.
template <uint BM, uint BN, uint NW>
static inline void q4e_affine_gemm_tile(device const float *xrow, bool x_live,
                                        thread device const uint *(&wrow)[NW],
                                        thread device const ushort *(&srow)[NW],
                                        thread device const ushort *(&brow)[NW], bool w_live,
                                        uint in_dim, threadgroup float *xs, threadgroup float *ws,
                                        uint tid, uint simd,
                                        thread simdgroup_float8x8 (&acc)[NW][2][2]) {
    constexpr uint XPT = BM * Q4E_GEMM_BK / 128u;
    constexpr uint WPT = BN * Q4E_GEMM_BK / 128u;
    constexpr uint WWORDS = WPT / 8u;
    const uint sm = (simd / (BN / 16u)) * 16u;
    const uint sn = (simd % (BN / 16u)) * 16u;
    const uint xr = q4e_gemm_xr<BM>(tid);
    const uint xc = (tid % (Q4E_GEMM_BK / XPT)) * XPT;
    const uint wr = q4e_gemm_wr<BN>(tid);
    const uint wc = (tid % (Q4E_GEMM_BK / WPT)) * WPT;
    for (uint n = 0; n < NW; ++n) {
        for (uint i = 0; i < 2u; ++i) {
            for (uint j = 0; j < 2u; ++j) {
                acc[n][i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
    }
    float4 xv[XPT / 4u];
    uint word[NW][WWORDS];
    float scale[NW];
    float bias[NW];
    for (uint v = 0; v < XPT / 4u; ++v) {
        xv[v] = x_live ? *(device const float4 *)(xrow + xc + 4u * v) : float4(0.0f);
    }
    for (uint n = 0; n < NW; ++n) {
        for (uint q = 0; q < WWORDS; ++q) {
            word[n][q] = w_live ? wrow[n][wc / 8u + q] : 0u;
        }
        scale[n] = w_live ? q4e_bf16(srow[n][wc / Q4E_AFFINE_GROUP]) : 0.0f;
        bias[n] = w_live ? q4e_bf16(brow[n][wc / Q4E_AFFINE_GROUP]) : 0.0f;
    }
    for (uint k0 = 0; k0 < in_dim; k0 += Q4E_GEMM_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint v = 0; v < XPT / 4u; ++v) {
            *(threadgroup float4 *)(xs + xr * Q4E_GEMM_LD + xc + 4u * v) = xv[v];
        }
        for (uint n = 0; n < NW; ++n) {
            threadgroup float *wd = ws + (n * BN + wr) * Q4E_GEMM_LD + wc;
            for (uint q = 0; q < WWORDS; ++q) {
                float wv[8];
                for (uint j = 0; j < 8u; ++j) {
                    wv[j] = float((word[n][q] >> (4u * j)) & 0xFu) * scale[n] + bias[n];
                }
                *(threadgroup float4 *)(wd + 8u * q) = float4(wv[0], wv[1], wv[2], wv[3]);
                *(threadgroup float4 *)(wd + 8u * q + 4u) = float4(wv[4], wv[5], wv[6], wv[7]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint k1 = k0 + Q4E_GEMM_BK;
        if (k1 < in_dim) {
            for (uint v = 0; v < XPT / 4u; ++v) {
                if (x_live) {
                    xv[v] = *(device const float4 *)(xrow + k1 + xc + 4u * v);
                }
            }
            if (w_live) {
                const uint g = (k1 + wc) / Q4E_AFFINE_GROUP;
                for (uint n = 0; n < NW; ++n) {
                    for (uint q = 0; q < WWORDS; ++q) {
                        word[n][q] = wrow[n][(k1 + wc) / 8u + q];
                    }
                    scale[n] = q4e_bf16(srow[n][g]);
                    bias[n] = q4e_bf16(brow[n][g]);
                }
            }
        }
        q4e_gemm_mma<BN, NW>(xs, ws, sm, sn, acc);
    }
}

/// Stores projection `n`'s `BM x BN` tile into `cs` (pitch `BN + 4`) for the
/// caller to scatter.
template <uint BM, uint BN, uint NW>
static inline void q4e_affine_gemm_stage(thread simdgroup_float8x8 (&acc)[NW][2][2], uint n,
                                         threadgroup float *cs, uint simd) {
    const uint sm = (simd / (BN / 16u)) * 16u;
    const uint sn = (simd % (BN / 16u)) * 16u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 2u; ++i) {
        for (uint j = 0; j < 2u; ++j) {
            simdgroup_store(acc[n][i][j], cs + (sm + 8u * i) * (BN + 4u) + sn + 8u * j, BN + 4u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// Columns and weight rows one tile of the routed prompt kernels covers.
constant constexpr uint Q4E_GEMM_EXPERT_COLS = 16u;
constant constexpr uint Q4E_GEMM_EXPERT_ROWS = 64u;

/// `out[r][o] = sum_k W[o][k] x[r][k]` for a prompt chunk, on the matrix units.
///
/// A 32 x 32 tile of the output a threadgroup of four simdgroups: each step
/// stages 32 activation rows and 32 weight rows of a 32-code slice of the
/// contraction in threadgroup memory - the weight dequantized there as
/// `q * scale + bias` - and each simdgroup multiplies its 16 x 16 quarter out
/// of 8 x 8 fragments. A row's sum runs over the slices in order and over the
/// fragments of a slice in order, so it depends on the shape alone and never
/// on which rows share its tile: every chunk width sums a row the same way,
/// which is the promise a prompt pass makes. It is not the one-row kernels'
/// order, which is why only a prompt pass is planned onto it.
///
/// Buffers: out, w, scales, biases, x, in_dim, out_dim, rows. `in_dim` a
/// multiple of 64. Grid `(ceil(out_dim / 32), ceil(rows / 32))`, 128 threads.
kernel void q4e_affine_gemm_kernel(
        device float *out [[buffer(0)]],
        device const uint *w [[buffer(1)]],
        device const ushort *scales [[buffer(2)]],
        device const ushort *biases [[buffer(3)]],
        device const float *x [[buffer(4)]],
        constant uint &in_dim [[buffer(5)]],
        constant uint &out_dim [[buffer(6)]],
        constant uint &rows [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[Q4E_GEMM_BM * Q4E_GEMM_LD];
    threadgroup float ws[Q4E_GEMM_BN * Q4E_GEMM_LD];
    const uint n0 = group.x * Q4E_GEMM_BN;
    const uint m0 = group.y * Q4E_GEMM_BM;
    const uint xr = q4e_gemm_xr<Q4E_GEMM_BM>(tid);
    const uint wr = q4e_gemm_wr<Q4E_GEMM_BN>(tid);
    const bool x_live = m0 + xr < rows;
    const bool w_live = n0 + wr < out_dim;
    const ulong row = w_live ? n0 + wr : 0u;
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    device const uint *wrow[1] = {w + row * (in_dim / 8u)};
    device const ushort *srow[1] = {scales + row * groups};
    device const ushort *brow[1] = {biases + row * groups};
    simdgroup_float8x8 acc[1][2][2];
    q4e_affine_gemm_tile<Q4E_GEMM_BM, Q4E_GEMM_BN, 1>(
        x + ulong(x_live ? m0 + xr : 0u) * in_dim, x_live, wrow, srow, brow, w_live, in_dim, xs,
        ws, tid, simd, acc);
    // The tile goes out through threadgroup memory, where the rows and
    // columns past the ends are dropped.
    q4e_affine_gemm_stage<Q4E_GEMM_BM, Q4E_GEMM_BN, 1>(acc, 0u, xs, simd);
    for (uint e = tid; e < Q4E_GEMM_BM * Q4E_GEMM_BN; e += 128u) {
        const uint r = e / Q4E_GEMM_BN;
        const uint c = e % Q4E_GEMM_BN;
        if (m0 + r < rows && n0 + c < out_dim) {
            out[ulong(m0 + r) * out_dim + n0 + c] = xs[r * (Q4E_GEMM_BN + 4u) + c];
        }
    }
}

/// `out[r][o] = sum_k W[o][k] x[r][k]` for a prompt chunk over a
/// brain-float weight, on the matrix units: `q4e_affine_gemm_kernel` with the
/// weight widened instead of dequantized, the same tiles and the same order,
/// so a row's sum is the shape's alone. The router, the recurrent gates and the
/// stream mixes' injections are this format; one launch covers a chunk they
/// otherwise take sixteen rows a launch.
///
/// Buffers: out, w (bf16 bits), x, in_dim (a multiple of 32), out_dim, rows.
/// Grid `(ceil(out_dim / 32), ceil(rows / 32))`, 128 threads.
kernel void q4e_bf16_gemm_kernel(
        device float *out [[buffer(0)]],
        device const ushort *w [[buffer(1)]],
        device const float *x [[buffer(2)]],
        constant uint &in_dim [[buffer(3)]],
        constant uint &out_dim [[buffer(4)]],
        constant uint &rows [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = Q4E_GEMM_BM;
    constexpr uint BN = Q4E_GEMM_BN;
    threadgroup float xs[BM * Q4E_GEMM_LD];
    threadgroup float ws[BN * Q4E_GEMM_LD];
    const uint n0 = group.x * BN;
    const uint m0 = group.y * BM;
    const uint sm = (simd / 2u) * 16u;
    const uint sn = (simd % 2u) * 16u;
    // Eight activations and eight weights a thread a slice, both of row
    // `tid / 4`.
    const uint tr = tid / 4u;
    const uint tc = (tid % 4u) * 8u;
    const bool x_live = m0 + tr < rows;
    const bool w_live = n0 + tr < out_dim;
    device const float *xr = x + ulong(x_live ? m0 + tr : 0u) * in_dim + tc;
    device const ushort *wr = w + ulong(w_live ? n0 + tr : 0u) * in_dim + tc;
    simdgroup_float8x8 acc[1][2][2];
    for (uint i = 0; i < 2u; ++i) {
        for (uint j = 0; j < 2u; ++j) {
            acc[0][i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    float4 xa = x_live ? *(device const float4 *)xr : float4(0.0f);
    float4 xb = x_live ? *(device const float4 *)(xr + 4u) : float4(0.0f);
    ushort4 wa = w_live ? *(device const ushort4 *)wr : ushort4(0);
    ushort4 wb = w_live ? *(device const ushort4 *)(wr + 4u) : ushort4(0);
    for (uint k0 = 0; k0 < in_dim; k0 += Q4E_GEMM_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *xd = xs + tr * Q4E_GEMM_LD + tc;
        *(threadgroup float4 *)xd = xa;
        *(threadgroup float4 *)(xd + 4u) = xb;
        threadgroup float *wd = ws + tr * Q4E_GEMM_LD + tc;
        *(threadgroup float4 *)wd =
            float4(q4e_bf16(wa.x), q4e_bf16(wa.y), q4e_bf16(wa.z), q4e_bf16(wa.w));
        *(threadgroup float4 *)(wd + 4u) =
            float4(q4e_bf16(wb.x), q4e_bf16(wb.y), q4e_bf16(wb.z), q4e_bf16(wb.w));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint k1 = k0 + Q4E_GEMM_BK;
        if (k1 < in_dim) {
            if (x_live) {
                xa = *(device const float4 *)(xr + k1);
                xb = *(device const float4 *)(xr + k1 + 4u);
            }
            if (w_live) {
                wa = *(device const ushort4 *)(wr + k1);
                wb = *(device const ushort4 *)(wr + k1 + 4u);
            }
        }
        q4e_gemm_mma<BN, 1>(xs, ws, sm, sn, acc);
    }
    q4e_affine_gemm_stage<BM, BN, 1>(acc, 0u, xs, simd);
    for (uint e = tid; e < BM * BN; e += 128u) {
        const uint r = e / BN;
        const uint c = e % BN;
        if (m0 + r < rows && n0 + c < out_dim) {
            out[ulong(m0 + r) * out_dim + n0 + c] = xs[r * (BN + 4u) + c];
        }
    }
}

/// Gate and up for a prompt chunk's routed columns, on the matrix units:
/// `q4e_affine_experts_gate_up_kernel`'s contract, with an expert's columns
/// taken 32 at a time as the rows of one tile, so its weights are staged once
/// a tile rather than once a column. A column's result depends on the shape
/// alone, never on which columns share its tile: the prompt pass's promise.
///
/// Sixteen columns and 64 weight rows a tile: an expert of a chunk gets a
/// dozen columns or so, and a 32-column tile would spend two thirds of its
/// multiplies on zeros.
///
/// Buffers as `q4e_affine_experts_gate_up_kernel`. Grid `(2 ceil(out_dim /
/// 64), min(columns, n_expert))`, 128 threads: the gate's tiles, then the
/// up's.
kernel void q4e_affine_experts_gemm_gate_up_kernel(
        device float *out_gate [[buffer(0)]],
        device float *out_up [[buffer(1)]],
        device const uint *gate_w [[buffer(2)]],
        device const ushort *gate_s [[buffer(3)]],
        device const ushort *gate_b [[buffer(4)]],
        device const uint *up_w [[buffer(5)]],
        device const ushort *up_s [[buffer(6)]],
        device const ushort *up_b [[buffer(7)]],
        device const float *x [[buffer(8)]],
        device const int *sorted [[buffer(9)]],
        device const int *bounds [[buffer(10)]],
        device const int *active [[buffer(11)]],
        device const uint *map [[buffer(12)]],
        constant uint &in_dim [[buffer(13)]],
        constant uint &out_dim [[buffer(14)]],
        constant uint &n_expert [[buffer(15)]],
        constant uint &n_used [[buffer(16)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = Q4E_GEMM_EXPERT_COLS;
    constexpr uint BN = Q4E_GEMM_EXPERT_ROWS;
    threadgroup float xs[BM * Q4E_GEMM_LD];
    threadgroup float ws[BN * Q4E_GEMM_LD];
    // The output tile is staged where the weight tile was, once it is done.
    threadgroup float *cs = ws;
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    // The first half of the grid's width is the gate's tiles, the second the
    // up's: one weight tile a threadgroup keeps two of them on a core.
    const uint tiles = (out_dim + BN - 1u) / BN;
    const bool is_up = group.x >= tiles;
    const uint n0 = (is_up ? group.x - tiles : group.x) * BN;
    const uint xr = q4e_gemm_xr<BM>(tid);
    const uint wr = q4e_gemm_wr<BN>(tid);
    const bool w_live = n0 + wr < out_dim;
    const ulong row = ulong(map[expert]) * out_dim + (w_live ? n0 + wr : 0u);
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    device const uint *wrow[1] = {(is_up ? up_w : gate_w) + row * (in_dim / 8u)};
    device const ushort *srow[1] = {(is_up ? up_s : gate_s) + row * groups};
    device const ushort *brow[1] = {(is_up ? up_b : gate_b) + row * groups};
    device float *dst = is_up ? out_up : out_gate;
    for (int t0 = c0; t0 < c1; t0 += int(BM)) {
        const bool x_live = t0 + int(xr) < c1;
        const uint col = x_live ? uint(sorted[t0 + int(xr)]) : 0u;
        simdgroup_float8x8 acc[1][2][2];
        q4e_affine_gemm_tile<BM, BN, 1>(x + ulong(col / n_used) * in_dim, x_live, wrow, srow,
                                        brow, w_live, in_dim, xs, ws, tid, simd, acc);
        q4e_affine_gemm_stage<BM, BN, 1>(acc, 0u, cs, simd);
        for (uint e = tid; e < BM * BN; e += 128u) {
            const uint r = e / BN;
            const uint c = e % BN;
            if (t0 + int(r) < c1 && n0 + c < out_dim) {
                dst[ulong(sorted[t0 + int(r)]) * out_dim + n0 + c] = cs[r * (BN + 4u) + c];
            }
        }
    }
}

/// The routed down projection for a prompt chunk, on the matrix units:
/// `q4e_affine_experts_down_kernel`'s contract, tiled as
/// `q4e_affine_experts_gemm_gate_up_kernel` is.
///
/// Buffers as `q4e_affine_experts_down_kernel`. Grid `(ceil(out_dim / 64),
/// min(columns, n_expert))`, 128 threads.
kernel void q4e_affine_experts_gemm_down_kernel(
        device float *out [[buffer(0)]],
        device const uint *down_w [[buffer(1)]],
        device const ushort *down_s [[buffer(2)]],
        device const ushort *down_b [[buffer(3)]],
        device const float *mid [[buffer(4)]],
        device const int *sorted [[buffer(5)]],
        device const int *bounds [[buffer(6)]],
        device const int *active [[buffer(7)]],
        device const uint *map [[buffer(8)]],
        constant uint &in_dim [[buffer(9)]],
        constant uint &out_dim [[buffer(10)]],
        constant uint &n_expert [[buffer(11)]],
        constant uint &n_used [[buffer(12)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = Q4E_GEMM_EXPERT_COLS;
    constexpr uint BN = Q4E_GEMM_EXPERT_ROWS;
    threadgroup float xs[BM * Q4E_GEMM_LD];
    threadgroup float ws[BN * Q4E_GEMM_LD];
    threadgroup float *cs = ws;
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint n0 = group.x * BN;
    const uint xr = q4e_gemm_xr<BM>(tid);
    const uint wr = q4e_gemm_wr<BN>(tid);
    const bool w_live = n0 + wr < out_dim;
    const ulong row = ulong(map[expert]) * out_dim + (w_live ? n0 + wr : 0u);
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    device const uint *wrow[1] = {down_w + row * (in_dim / 8u)};
    device const ushort *srow[1] = {down_s + row * groups};
    device const ushort *brow[1] = {down_b + row * groups};
    for (int t0 = c0; t0 < c1; t0 += int(BM)) {
        const bool x_live = t0 + int(xr) < c1;
        const uint col = x_live ? uint(sorted[t0 + int(xr)]) : 0u;
        simdgroup_float8x8 acc[1][2][2];
        q4e_affine_gemm_tile<BM, BN, 1>(mid + ulong(col) * in_dim, x_live, wrow, srow, brow,
                                        w_live, in_dim, xs, ws, tid, simd, acc);
        q4e_affine_gemm_stage<BM, BN, 1>(acc, 0u, cs, simd);
        for (uint e = tid; e < BM * BN; e += 128u) {
            const uint r = e / BN;
            const uint c = e % BN;
            if (t0 + int(r) < c1 && n0 + c < out_dim) {
                out[ulong(sorted[t0 + int(r)]) * out_dim + n0 + c] = cs[r * (BN + 4u) + c];
            }
        }
    }
}

/// The row gather that opens a pass, over an affine vocabulary table:
/// `embed_tokens_hc_kernel`'s contract with the table dequantized in place.
///
/// Buffers: out, tokens, w, scales, biases, n_vocab, n_tokens, n_embd, n_hc.
/// Grid `ceil(n_tokens * n_hc * n_embd / 256)`, 256 threads.
kernel void q4e_affine_embed_hc_kernel(
        device float *out [[buffer(0)]],
        device const int *tokens [[buffer(1)]],
        device const uint *w [[buffer(2)]],
        device const ushort *scales [[buffer(3)]],
        device const ushort *biases [[buffer(4)]],
        constant uint &n_vocab [[buffer(5)]],
        constant uint &n_tokens [[buffer(6)]],
        constant uint &n_embd [[buffer(7)]],
        constant uint &n_hc [[buffer(8)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong n = ulong(n_tokens) * n_hc * n_embd;
    if (ulong(gid) >= n) {
        return;
    }
    const uint d = gid % n_embd;
    const uint t = (gid / n_embd) / n_hc;
    const int tok_i = tokens[t];
    uint tok = tok_i < 0 ? 0u : uint(tok_i);
    if (tok >= n_vocab) {
        tok = 0u;
    }
    const uint groups = n_embd / Q4E_AFFINE_GROUP;
    const uint word = w[ulong(tok) * (n_embd / 8u) + d / 8u];
    const float q = float((word >> (4u * (d % 8u))) & 0xFu);
    const ulong p = ulong(tok) * groups + d / Q4E_AFFINE_GROUP;
    out[gid] = q * q4e_bf16(scales[p]) + q4e_bf16(biases[p]);
}

/// Gate and up for one active expert's columns, eight output rows a
/// threadgroup, four per simdgroup; the expert's span of the sorted column list
/// is swept so its rows stay in cache across the columns routed to it.
///
/// `map[e]` is the slot expert `e`'s matrices sit at inside the three banks:
/// the identity for a bank that is wholly resident, and the slot the stream
/// placed it in for one that is streamed. A column's activation row is its
/// token's: `col / n_used`.
///
/// Buffers: out_gate, out_up, gate_w, gate_s, gate_b, up_w, up_s, up_b, x,
/// sorted, bounds, active, map, in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8),
/// min(columns, n_expert))`, 64 threads.
kernel void q4e_affine_experts_gate_up_kernel(
        device float *out_gate [[buffer(0)]],
        device float *out_up [[buffer(1)]],
        device const uint *gate_w [[buffer(2)]],
        device const ushort *gate_s [[buffer(3)]],
        device const ushort *gate_b [[buffer(4)]],
        device const uint *up_w [[buffer(5)]],
        device const ushort *up_s [[buffer(6)]],
        device const ushort *up_b [[buffer(7)]],
        device const float *x [[buffer(8)]],
        device const int *sorted [[buffer(9)]],
        device const int *bounds [[buffer(10)]],
        device const int *active [[buffer(11)]],
        device const uint *map [[buffer(12)]],
        constant uint &in_dim [[buffer(13)]],
        constant uint &out_dim [[buffer(14)]],
        constant uint &n_expert [[buffer(15)]],
        constant uint &n_used [[buffer(16)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint n_simd [[simdgroups_per_threadgroup]]) {
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    const ulong slot = map[expert];
    for (uint sub = simd * Q4E_AFFINE_SG_ROWS; sub < Q4E_AFFINE_ROWS;
         sub += n_simd * Q4E_AFFINE_SG_ROWS) {
        const uint o = group.x * Q4E_AFFINE_ROWS + sub;
        if (o >= out_dim) {
            return;
        }
        const uint rows_out = min(Q4E_AFFINE_SG_ROWS, out_dim - o);
        const ulong row = slot * out_dim + o;
        device const uint *gw = gate_w + row * (in_dim / 8u);
        device const ushort *gs = gate_s + row * groups;
        device const ushort *gb = gate_b + row * groups;
        device const uint *uw = up_w + row * (in_dim / 8u);
        device const ushort *us = up_s + row * groups;
        device const ushort *ub = up_b + row * groups;
        for (int c = c0; c < c1; ++c) {
            const uint col = uint(sorted[c]);
            device const float *xr = x + ulong(col / n_used) * in_dim;
            float g[Q4E_AFFINE_SG_ROWS][1];
            float u[Q4E_AFFINE_SG_ROWS][1];
            q4e_affine_rows<Q4E_AFFINE_SG_ROWS, 1>(gw, gs, gb, xr, in_dim, rows_out, 1u, lane, 0u,
                                                   1u, g);
            q4e_affine_rows<Q4E_AFFINE_SG_ROWS, 1>(uw, us, ub, xr, in_dim, rows_out, 1u, lane, 0u,
                                                   1u, u);
            for (uint r = 0; r < Q4E_AFFINE_SG_ROWS; ++r) {
                const float gv = q4e_warp_sum(g[r][0]);
                const float uv = q4e_warp_sum(u[r][0]);
                if (lane == 0u && r < rows_out) {
                    out_gate[ulong(col) * out_dim + o + r] = gv;
                    out_up[ulong(col) * out_dim + o + r] = uv;
                }
            }
        }
    }
}

/// Lanes one row of the narrow routed down projection is split across.
constant constexpr uint Q4E_DOWN_SPLIT_LANES = 8u;
/// Sixteen-code chunks one of those lanes may own: contractions up to 1024.
constant constexpr uint Q4E_DOWN_SPLIT_CHUNKS = 8u;
/// Output rows one threadgroup of the split down projection covers.
constant constexpr uint Q4E_DOWN_SPLIT_ROWS = 16u;

/// The row split of the routed down projection at `C` chunks a lane.
template <uint C>
static inline void q4e_affine_down_split(
        device float *out, device const uint *down_w, device const ushort *down_s,
        device const ushort *down_b, device const float *mid, device const int *sorted,
        device const uint *map, int expert, int c0, int c1, uint in_dim, uint out_dim,
        uint3 group, uint lane, uint simd, uint n_simd) {
    constexpr uint per_simd = 32u / Q4E_DOWN_SPLIT_LANES;
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    const uint words = in_dim / 8u;
    const uint part = lane % Q4E_DOWN_SPLIT_LANES;
    for (uint sub = simd * per_simd; sub < Q4E_DOWN_SPLIT_ROWS; sub += n_simd * per_simd) {
        const uint o = group.x * Q4E_DOWN_SPLIT_ROWS + sub + lane / Q4E_DOWN_SPLIT_LANES;
        const bool live = o < out_dim;
        const ulong row = ulong(map[expert]) * out_dim + (live ? o : 0u);
        device const uint *dw = down_w + row * words;
        device const ushort *ds = down_s + row * groups;
        device const ushort *db = down_b + row * groups;
        uint2 wv[C];
        float sc[C];
        float bi[C];
        for (uint i = 0; i < C; ++i) {
            const uint k = (part * C + i) * Q4E_AFFINE_LANE_CODES;
            wv[i] = *(device const uint2 *)(dw + k / 8u);
            sc[i] = q4e_bf16(ds[k / Q4E_AFFINE_GROUP]);
            bi[i] = q4e_bf16(db[k / Q4E_AFFINE_GROUP]);
        }
        for (int c = c0; c < c1; ++c) {
            const uint col = uint(sorted[c]);
            device const float *xr = mid + ulong(col) * in_dim;
            float acc = 0.0f;
            for (uint i = 0; i < C; ++i) {
                const uint k = (part * C + i) * Q4E_AFFINE_LANE_CODES;
                float xt[16];
                const float sum = q4e_affine_load_x(xr + k, xt);
                acc += sc[i] * q4e_affine_qdot(wv[i], xt) + bi[i] * sum;
            }
            for (ushort off = Q4E_DOWN_SPLIT_LANES / 2u; off > 0; off >>= 1) {
                acc += simd_shuffle_down(acc, off);
            }
            if (part == 0u && live) {
                out[ulong(col) * out_dim + o] = acc;
            }
        }
    }
}

/// The routed down projection: each column reads its OWN row of the
/// activated intermediate and writes `[column, out_dim]` row-major, which is
/// what `q4e_moe_combine_kernel` reads.
///
/// The contraction is the intermediate width, 640 here: a quarter over one
/// 512-code sweep, so the sweep layout of the dense projections leaves a
/// lane 32 bytes in flight and three quarters of the simdgroup idle on its
/// second sweep. A contraction that is a multiple of 128 up to 1024 is split
/// by rows instead: eight lanes a row, four rows a simdgroup at once, each lane
/// owning `in_dim / 128` contiguous chunks of sixteen codes, loaded before any
/// is used; the eight partials fold with a decreasing-offset shuffle. Sixteen
/// rows a threadgroup (`q4e_down_split` in the host). Any other width takes the
/// dense projections' layout at eight rows a threadgroup. Either way the order
/// a row is summed in depends on the shape alone.
///
/// Buffers: out, down_w, down_s, down_b, mid, sorted, bounds, active, map,
/// in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / rows),
/// min(columns, n_expert))`, 64 threads.
kernel void q4e_affine_experts_down_kernel(
        device float *out [[buffer(0)]],
        device const uint *down_w [[buffer(1)]],
        device const ushort *down_s [[buffer(2)]],
        device const ushort *down_b [[buffer(3)]],
        device const float *mid [[buffer(4)]],
        device const int *sorted [[buffer(5)]],
        device const int *bounds [[buffer(6)]],
        device const int *active [[buffer(7)]],
        device const uint *map [[buffer(8)]],
        constant uint &in_dim [[buffer(9)]],
        constant uint &out_dim [[buffer(10)]],
        constant uint &n_expert [[buffer(11)]],
        constant uint &n_used [[buffer(12)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint n_simd [[simdgroups_per_threadgroup]]) {
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    const uint groups = in_dim / Q4E_AFFINE_GROUP;
    const uint words = in_dim / 8u;
    const uint span = Q4E_DOWN_SPLIT_LANES * Q4E_AFFINE_LANE_CODES;
    if (in_dim % span == 0u && in_dim <= span * Q4E_DOWN_SPLIT_CHUNKS) {
        const device float *m = mid;
#define Q4E_DOWN_SPLIT_CASE(C)                                                               \
    case C:                                                                                   \
        q4e_affine_down_split<C>(out, down_w, down_s, down_b, m, sorted, map, expert, c0, c1, \
                                 in_dim, out_dim, group, lane, simd, n_simd);                 \
        return;
        switch (in_dim / span) {
            Q4E_DOWN_SPLIT_CASE(1)
            Q4E_DOWN_SPLIT_CASE(2)
            Q4E_DOWN_SPLIT_CASE(3)
            Q4E_DOWN_SPLIT_CASE(4)
            Q4E_DOWN_SPLIT_CASE(5)
            Q4E_DOWN_SPLIT_CASE(6)
            Q4E_DOWN_SPLIT_CASE(7)
            Q4E_DOWN_SPLIT_CASE(8)
            default:
                return;
        }
#undef Q4E_DOWN_SPLIT_CASE
    }
    for (uint sub = simd * Q4E_AFFINE_SG_ROWS; sub < Q4E_AFFINE_ROWS;
         sub += n_simd * Q4E_AFFINE_SG_ROWS) {
        const uint o = group.x * Q4E_AFFINE_ROWS + sub;
        if (o >= out_dim) {
            return;
        }
        const uint rows_out = min(Q4E_AFFINE_SG_ROWS, out_dim - o);
        const ulong row = ulong(map[expert]) * out_dim + o;
        device const uint *dw = down_w + row * words;
        device const ushort *ds = down_s + row * groups;
        device const ushort *db = down_b + row * groups;
        for (int c = c0; c < c1; ++c) {
            const uint col = uint(sorted[c]);
            float d[Q4E_AFFINE_SG_ROWS][1];
            q4e_affine_rows<Q4E_AFFINE_SG_ROWS, 1>(dw, ds, db, mid + ulong(col) * in_dim, in_dim,
                                                   rows_out, 1u, lane, 0u, 1u, d);
            for (uint r = 0; r < Q4E_AFFINE_SG_ROWS; ++r) {
                const float v = q4e_warp_sum(d[r][0]);
                if (lane == 0u && r < rows_out) {
                    out[ulong(col) * out_dim + o + r] = v;
                }
            }
        }
    }
}

/// Decodes gathered n-gram rows stored as MLX affine 4-bit, group 32.
///
/// A gathered row is the three arrays' slices of one table row, concatenated
/// by the reader: `head_dim / 2` bytes of codes, then `head_dim / 32`
/// brain-float scales, then as many biases. Row `r` belongs to token
/// `r / n_heads` and lands at head slot `r % n_heads`, head-slowest inside the
/// token's `n_heads * head_dim` wide result - `q4e_ple_dequant_kernel`'s
/// placement.
///
/// Buffers: out, rows, head_dim, n_heads, row_bytes, n_rows. Grid `n_rows`,
/// one thread per element (the host launches `head_dim` threads).
kernel void q4e_ple_dequant_affine_kernel(
        device float *out [[buffer(0)]],
        device const uchar *rows [[buffer(1)]],
        constant uint &head_dim [[buffer(2)]],
        constant uint &n_heads [[buffer(3)]],
        constant uint &row_bytes [[buffer(4)]],
        constant uint &n_rows [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 tid [[thread_position_in_threadgroup]],
        uint3 tpg [[threads_per_threadgroup]]) {
    const uint r = group.x;
    if (r >= n_rows) {
        return;
    }
    device const uchar *src = rows + ulong(r) * row_bytes;
    device const ushort *params = (device const ushort *)(src + head_dim / 2u);
    const uint groups = head_dim / 32u;
    const uint t = r / n_heads;
    const uint h = r % n_heads;
    device float *dst = out + ulong(t) * n_heads * head_dim + ulong(h) * head_dim;
    for (uint i = tid.x; i < head_dim; i += tpg.x) {
        const uchar byte = src[i / 2u];
        const float q = float((i & 1u) ? (byte >> 4u) : (byte & 0xFu));
        const uint g = i / 32u;
        dst[i] = q * q4e_bf16(params[g]) + q4e_bf16(params[groups + g]);
    }
}

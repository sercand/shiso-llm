/* The `audio` module: the audio encoders' kernel set.
 *
 * Nothing here is copied. The arithmetic is the Metal twin's in
 * `crates/shiso-metal/metal/audio`, the launch contract is that directory's
 * KERNELS.md, and the entry names are the host's `&'static str` literals in
 * `crates/shiso-models/src/audio_ops`, so every entry carries C linkage.
 *
 * Launch geometry is the contract's: the element-wise entries take a flat
 * index over 256-thread blocks, the norms one block per row, the GEMM a 32x64
 * output tile per 128-thread block with 12288 bytes of dynamic shared memory
 * (the 64-row GEMM 64x64 per 256 threads with 20480),
 * the window and causal attention one warp per (head, block of eight
 * queries), and decode attention a warp per (kv head, split) with four to a
 * block, closed by a 256-thread merge per head.
 */

#include "audio_common.cuh"

/* ---- element-wise --------------------------------------------------------- */

extern "C" __global__ void kernel_audio_unary_f32(
        const struct audio_args_unary args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    float y;
    switch (args.op) {
        case AUDIO_OP_RELU:     y = fmaxf(x, 0.0f); break;
        case AUDIO_OP_SILU:     y = x * audio_sigmoid(x); break;
        case AUDIO_OP_SIGMOID:  y = audio_sigmoid(x); break;
        case AUDIO_OP_GELU_ERF: y = 0.5f * x * (1.0f + audio_erf(x * 0.70710678118654752f)); break;
        case AUDIO_OP_TANH:     y = tanhf(x); break;
        case AUDIO_OP_CLAMP:    y = fminf(fmaxf(x, args.a), args.b); break;
        case AUDIO_OP_SCALE:    y = x * args.a; break;
        default:                y = x; break;
    }
    out[i] = y;
}

extern "C" __global__ void kernel_audio_axpy_f32(
        const struct audio_args_axpy args, float *y, const float *x) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    y[i] = y[i] + args.alpha * x[i];
}

extern "C" __global__ void kernel_audio_glu_f32(
        const struct audio_args_glu args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.channels) {
        return;
    }
    const uint r = i / args.channels;
    const uint c = i - r * args.channels;
    const float *row = in + (uint64_t) r * 2u * args.channels;
    out[i] = row[c] * audio_sigmoid(row[args.channels + c]);
}

extern "C" __global__ void kernel_audio_silu_mul_f32(
        const struct audio_args_silu_mul args, float *out, const float *gate, const float *up) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    const float g = gate[i];
    out[i] = g * audio_sigmoid(g) * up[i];
}

extern "C" __global__ void kernel_audio_bias_rows_f32(
        const struct audio_args_bias_rows args, float *y, const float *b) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.cols) {
        return;
    }
    y[i] = y[i] + b[i % args.cols];
}

/* The larger of two (value, index) candidates; a tie keeps the lower index. */
static __device__ __forceinline__ void audio_argmax_pick(float &v, uint &i, float ov, uint oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

/* Per row, the first index of the largest value, one block of
 * AUDIO_ARGMAX_BLOCK threads per row. A NaN never wins and a row with nothing
 * above negative infinity answers 0. */
extern "C" __global__ void kernel_audio_argmax_rows_f32(
        const struct audio_args_argmax args, uint *out, const float *in) {
    __shared__ float best_v[AUDIO_ARGMAX_BLOCK / 32u];
    __shared__ uint best_i[AUDIO_ARGMAX_BLOCK / 32u];
    const uint r = blockIdx.x;
    if (r >= args.rows) {
        return;
    }
    const float *row = in + (uint64_t) r * args.cols;
    float v = -INFINITY;
    uint i = 0xffffffffu;
    for (uint c = threadIdx.x; c < args.cols; c += AUDIO_ARGMAX_BLOCK) {
        if (row[c] > v) {
            v = row[c];
            i = c;
        }
    }
    for (uint off = 16u; off > 0u; off >>= 1u) {
        audio_argmax_pick(v, i, __shfl_xor_sync(0xffffffffu, v, off),
                          __shfl_xor_sync(0xffffffffu, i, off));
    }
    if ((threadIdx.x & 31u) == 0u) {
        best_v[threadIdx.x >> 5u] = v;
        best_i[threadIdx.x >> 5u] = i;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (uint s = 1; s < AUDIO_ARGMAX_BLOCK / 32u; ++s) {
            audio_argmax_pick(v, i, best_v[s], best_i[s]);
        }
        out[r] = i == 0xffffffffu ? 0u : i;
    }
}

/* ---- norms ---------------------------------------------------------------- */

/* Two-pass about the row's first value, as the Metal twin: a row with a large
 * mean and a small variance keeps its precision. Safe in place. */
extern "C" __global__ void kernel_audio_layer_norm_f32(
        const struct audio_args_norm args, float *out, const float *in,
        const float *weight, const float *bias) {
    extern __shared__ float scratch[];
    const uint r = blockIdx.x;
    if (r >= args.rows) {
        return;
    }
    const uint nt = blockDim.x;
    const uint tid = threadIdx.x;
    const float *src = in + (uint64_t) r * args.dim;
    float *dst = out + (uint64_t) r * args.dim;
    const float pivot = src[0];
    float sum = 0.0f;
    for (uint i = tid; i < args.dim; i += nt) {
        sum += src[i] - pivot;
    }
    const float dmean = audio_block_sum(sum, scratch, tid, nt) / (float) args.dim;
    float var = 0.0f;
    for (uint i = tid; i < args.dim; i += nt) {
        const float d = (src[i] - pivot) - dmean;
        var += d * d;
    }
    const float rstd = rsqrtf(audio_block_sum(var, scratch, tid, nt) / (float) args.dim + args.eps);
    const bool has_bias = (args.flags & AUDIO_NORM_HAS_BIAS) != 0u;
    for (uint i = tid; i < args.dim; i += nt) {
        const float y = ((src[i] - pivot) - dmean) * rstd * weight[i];
        dst[i] = has_bias ? y + bias[i] : y;
    }
}

extern "C" __global__ void kernel_audio_rms_norm_f32(
        const struct audio_args_norm args, float *out, const float *in, const float *weight) {
    extern __shared__ float scratch[];
    const uint r = blockIdx.x;
    if (r >= args.rows) {
        return;
    }
    const uint nt = blockDim.x;
    const uint tid = threadIdx.x;
    const float *src = in + (uint64_t) r * args.dim;
    float *dst = out + (uint64_t) r * args.dim;
    float sq = 0.0f;
    for (uint i = tid; i < args.dim; i += nt) {
        sq += src[i] * src[i];
    }
    const float rms = rsqrtf(audio_block_sum(sq, scratch, tid, nt) / (float) args.dim + args.eps);
    const bool has_weight = (args.flags & AUDIO_NORM_HAS_WEIGHT) != 0u;
    for (uint i = tid; i < args.dim; i += nt) {
        const float y = src[i] * rms;
        dst[i] = has_weight ? y * weight[i] : y;
    }
}

/* ---- the GEMM ------------------------------------------------------------- */

struct audio_w_f32 {
    static __device__ __forceinline__ float load(const void *w, uint64_t i) {
        return ((const float *) w)[i];
    }
};
struct audio_w_bf16 {
    static __device__ __forceinline__ float load(const void *w, uint64_t i) {
        return audio_bf16((const unsigned short *) w + i);
    }
};
struct audio_w_f16 {
    static __device__ __forceinline__ float load(const void *w, uint64_t i) {
        return __half2float(((const __half *) w)[i]);
    }
};
/* GGUF Q8_0: 34-byte blocks of an f16 scale then 32 signed bytes along each
 * row, dequantised in f32. The scale is read as two bytes so the load makes no
 * assumption about the base the buffer was imported at; blocks are 2-byte but
 * not 4-byte aligned relative to it. */
struct audio_w_q8_0 {
    static __device__ __forceinline__ float load(const void *w, uint64_t i) {
        const unsigned char *block = (const unsigned char *) w + (i >> 5u) * 34u;
        const unsigned short bits = (unsigned short) (block[0] | (block[1] << 8));
        const float d = __half2float(__ushort_as_half(bits));
        return d * (float) ((const signed char *) block)[2u + (i & 31u)];
    }
};

/* y = x * W^T (+ b): a 32-row by 64-column output tile per block of 128
 * threads, both operands staged through shared memory with zero fill past
 * every edge; each thread accumulates sixteen outputs of one row in f32. With
 * more than one split, block z covers its slice of k and writes a partial
 * tile into plane z of `y`, which the split-sum pass adds up. */
/* BM rows by 64 columns per block of 4 * BM threads, each thread one row by
 * 16 columns: the 32-row tile on 128 threads, and the 64-row tile the host
 * picks for more than 32 rows on 256. */
template <typename W, uint BM>
static __device__ __forceinline__ void audio_gemm(
        const struct audio_args_gemm args, float *y, const float *x, const void *w,
        const float *bias) {
    extern __shared__ float smem[];
    const uint NT = 4u * BM;
    const uint BN = AUDIO_GEMM_TILE_N;
    const uint BK = AUDIO_GEMM_TILE_K;
    float *sa = smem;           /* [BM rows][32 k] */
    float *sb = smem + BM * BK; /* [64 cols][32 k] */
    const uint tid = threadIdx.x;
    const uint n0 = blockIdx.x * BN;
    const uint m0 = blockIdx.y * BM;
    const uint k_begin = blockIdx.z * args.k_chunk;
    const uint k_end = min(args.k, k_begin + args.k_chunk);
    float *out = y + (uint64_t) blockIdx.z * args.m * args.n;
    const uint row = tid >> 2u;         /* 0..BM-1 */
    const uint col0 = (tid & 3u) * 16u; /* 0, 16, 32, 48 */
    float acc[16];
    for (uint j = 0; j < 16u; ++j) {
        acc[j] = 0.0f;
    }
    for (uint k0 = k_begin; k0 < k_end; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += NT) {
            const uint rr = e / BK;
            const uint kk = e - rr * BK;
            const uint gk = k0 + kk;
            const uint gm = m0 + rr;
            sa[e] = (gm < args.m && gk < k_end) ? x[(uint64_t) gm * args.k + gk] : 0.0f;
        }
        for (uint e = tid; e < BN * BK; e += NT) {
            const uint rr = e / BK;
            const uint kk = e - rr * BK;
            const uint gk = k0 + kk;
            const uint gn = n0 + rr;
            sb[e] = (gn < args.n && gk < k_end) ? W::load(w, (uint64_t) gn * args.k + gk) : 0.0f;
        }
        __syncthreads();
        for (uint kk = 0; kk < BK; ++kk) {
            const float a = sa[row * BK + kk];
            for (uint j = 0; j < 16u; ++j) {
                acc[j] += a * sb[(col0 + j) * BK + kk];
            }
        }
        __syncthreads();
    }
    /* A partial tile carries no bias: the split-sum pass adds it once. */
    const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u && args.splits <= 1u;
    const uint gm = m0 + row;
    if (gm >= args.m) {
        return;
    }
    for (uint j = 0; j < 16u; ++j) {
        const uint gn = n0 + col0 + j;
        if (gn < args.n) {
            out[(uint64_t) gm * args.n + gn] = acc[j] + (has_bias ? bias[gn] : 0.0f);
        }
    }
}

/* y = x * W^T (+ b) for a handful of rows: each warp owns 4 output columns of
 * up to 8 rows and strides its columns' weights along k, so a block of four
 * warps covers 16 columns and the grid is `n / 16` by `m / 8`. */
template <typename W>
static __device__ __forceinline__ void audio_gemv(
        const struct audio_args_gemm args, float *y, const float *x, const void *w,
        const float *bias) {
    const uint MT = AUDIO_GEMV_ROWS;
    const uint NC = AUDIO_GEMV_COLS;
    const uint lane = threadIdx.x & 31u;
    const uint warp = threadIdx.x >> 5u;
    const uint n0 = (blockIdx.x * 4u + warp) * NC;
    const uint m0 = blockIdx.y * MT;
    if (n0 >= args.n || m0 >= args.m) {
        return;
    }
    const uint rows = min(MT, args.m - m0);
    const uint cols = min(NC, args.n - n0);
    float acc[AUDIO_GEMV_ROWS][AUDIO_GEMV_COLS];
    for (uint r = 0; r < MT; ++r) {
        for (uint c = 0; c < NC; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    for (uint kk = lane; kk < args.k; kk += 32u) {
        float wv[AUDIO_GEMV_COLS];
        for (uint c = 0; c < NC; ++c) {
            wv[c] = c < cols ? W::load(w, (uint64_t) (n0 + c) * args.k + kk) : 0.0f;
        }
        for (uint r = 0; r < rows; ++r) {
            const float xv = x[(uint64_t) (m0 + r) * args.k + kk];
            for (uint c = 0; c < NC; ++c) {
                acc[r][c] += xv * wv[c];
            }
        }
    }
    const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u;
    for (uint r = 0; r < rows; ++r) {
        for (uint c = 0; c < cols; ++c) {
            const float total = audio_warp_sum(acc[r][c]);
            if (lane == 0u) {
                y[(uint64_t) (m0 + r) * args.n + n0 + c] = total + (has_bias ? bias[n0 + c] : 0.0f);
            }
        }
    }
}

#define AUDIO_GEMM_ENTRIES(SUFFIX, W) \
    extern "C" __global__ void kernel_audio_gemm_##SUFFIX( \
            const struct audio_args_gemm args, float *y, const float *x, const void *w, const float *bias) { \
        audio_gemm<W, AUDIO_GEMM_TILE_M>(args, y, x, w, bias); \
    } \
    extern "C" __global__ void kernel_audio_gemm64_##SUFFIX( \
            const struct audio_args_gemm args, float *y, const float *x, const void *w, const float *bias) { \
        audio_gemm<W, AUDIO_GEMM64_TILE_M>(args, y, x, w, bias); \
    } \
    extern "C" __global__ void kernel_audio_gemv_##SUFFIX( \
            const struct audio_args_gemm args, float *y, const float *x, const void *w, const float *bias) { \
        audio_gemv<W>(args, y, x, w, bias); \
    }

AUDIO_GEMM_ENTRIES(f32w, audio_w_f32)
AUDIO_GEMM_ENTRIES(bf16w, audio_w_bf16)
AUDIO_GEMM_ENTRIES(f16w, audio_w_f16)
AUDIO_GEMM_ENTRIES(q8_0w, audio_w_q8_0)

/* y = x * W^T (+ b) one row at a time, the Metal twin's order: each lane dots
 * runs of eight consecutive weights (a Q8_0 quarter block, scaled once),
 * `(w0.x0) + (w1.x1)` over two four-wide halves; a lane sums its runs in k
 * order, a warp sum closes each warp, and the two warps' sums are added in
 * warp order. The grid is `ceil(n / 4)` by `m`, 64 threads. */
template <typename W>
static __device__ __forceinline__ float audio_run8(const void *w, uint64_t row0, uint j,
                                                   const float *xr) {
    const uint64_t at = row0 + (uint64_t) j * 8u;
    const float *xp = xr + (uint64_t) j * 8u;
    float lo = 0.0f;
    float hi = 0.0f;
    for (uint e = 0; e < 4u; ++e) {
        lo += W::load(w, at + e) * xp[e];
        hi += W::load(w, at + 4u + e) * xp[4u + e];
    }
    return lo + hi;
}
template <>
__device__ __forceinline__ float audio_run8<audio_w_q8_0>(const void *w, uint64_t row0, uint j,
                                                          const float *xr) {
    const uint64_t at = row0 + (uint64_t) j * 8u;
    const unsigned char *block = (const unsigned char *) w + (at >> 5u) * 34u;
    const unsigned short bits = (unsigned short) (block[0] | (block[1] << 8));
    const float d = __half2float(__ushort_as_half(bits));
    const signed char *qs = (const signed char *) block + 2u + (at & 31u);
    const float *xp = xr + (uint64_t) j * 8u;
    float lo = 0.0f;
    float hi = 0.0f;
    for (uint e = 0; e < 4u; ++e) {
        lo += (float) qs[e] * xp[e];
        hi += (float) qs[4u + e] * xp[4u + e];
    }
    return (lo + hi) * d;
}

template <typename W>
static __device__ __forceinline__ void audio_matvec(
        const struct audio_args_gemm args, float *y, const float *x, const void *w,
        const float *bias) {
    const uint NR = AUDIO_MATVEC_COLS;
    const uint NSG = AUDIO_MATVEC_SG;
    __shared__ float part[AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    const uint lane = threadIdx.x & 31u;
    const uint warp = threadIdx.x >> 5u;
    const uint n0 = blockIdx.x * NR;
    const uint row = blockIdx.y;
    const uint runs = args.k >> 3u;
    const float *xr = x + (uint64_t) row * args.k;
    float acc[AUDIO_MATVEC_COLS];
    for (uint c = 0; c < NR; ++c) {
        acc[c] = 0.0f;
    }
    for (uint j = warp * 32u + lane; j < runs; j += NSG * 32u) {
        for (uint c = 0; c < NR; ++c) {
            const uint64_t col = min(n0 + c, args.n - 1u);
            acc[c] += audio_run8<W>(w, col * args.k, j, xr);
        }
    }
    for (uint c = 0; c < NR; ++c) {
        acc[c] = audio_warp_sum(acc[c]);
    }
    if (lane == 0u) {
        for (uint c = 0; c < NR; ++c) {
            part[c * NSG + warp] = acc[c];
        }
    }
    __syncthreads();
    if (warp == 0u && lane < NR && n0 + lane < args.n) {
        float total = 0.0f;
        for (uint s = 0; s < NSG; ++s) {
            total += part[lane * NSG + s];
        }
        const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u;
        y[(uint64_t) row * args.n + n0 + lane] = total + (has_bias ? bias[n0 + lane] : 0.0f);
    }
}

#define AUDIO_MATVEC_ENTRY(SUFFIX, W) \
    extern "C" __global__ void kernel_audio_matvec_##SUFFIX( \
            const struct audio_args_gemm args, float *y, const float *x, const void *w, const float *bias) { \
        audio_matvec<W>(args, y, x, w, bias); \
    }

AUDIO_MATVEC_ENTRY(f32w, audio_w_f32)
AUDIO_MATVEC_ENTRY(bf16w, audio_w_bf16)
AUDIO_MATVEC_ENTRY(f16w, audio_w_f16)
AUDIO_MATVEC_ENTRY(q8_0w, audio_w_q8_0)

/* y[i] = sum_z parts[z][i] (+ b[i % n]): closes a split-k GEMM, summing the
 * planes in order so the result does not depend on scheduling. */
extern "C" __global__ void kernel_audio_split_sum_f32(
        const struct audio_args_gemm args, float *y, const float *parts, const float *bias) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint count = args.m * args.n;
    if (i >= count) {
        return;
    }
    float total = 0.0f;
    for (uint z = 0; z < args.splits; ++z) {
        total += parts[(uint64_t) z * count + i];
    }
    const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u;
    y[i] = total + (has_bias ? bias[i % args.n] : 0.0f);
}

/* ---- convolutions and layout moves --------------------------------------- */

extern "C" __global__ void kernel_audio_conv2d_s2_f32(
        const struct audio_args_conv2d args, float *out, const float *in, const float *w,
        const float *bias) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint plane = args.t_out * args.f_out * args.c_out;
    if (i >= args.batch * plane) {
        return;
    }
    // Planes are independent images: each one's padding is zeros, never the
    // neighbouring plane's rows.
    const uint b = i / plane;
    const uint j = i - b * plane;
    const uint co = j % args.c_out;
    const uint f = (j / args.c_out) % args.f_out;
    const uint t = j / (args.c_out * args.f_out);
    const float *in_b = in + (uint64_t) b * args.t_in * args.f_in * args.c_in;
    if (t >= args.out_valid) {
        out[i] = 0.0f;
        return;
    }
    const uint rows = min(args.t_in, args.in_valid);
    const uint per = args.depthwise != 0u ? 1u : args.c_in;
    float acc = args.has_bias != 0u ? bias[co] : 0.0f;
    for (uint kt = 0; kt < 3u; ++kt) {
        const int ti = (int) (2u * t + kt) - (int) args.pad_t;
        if (ti < 0 || (uint) ti >= rows) {
            continue;
        }
        for (uint kf = 0; kf < 3u; ++kf) {
            const int fi = (int) (2u * f + kf) - (int) args.pad_f;
            if (fi < 0 || (uint) fi >= args.f_in) {
                continue;
            }
            const float *px = in_b + ((uint64_t) ti * args.f_in + (uint) fi) * args.c_in;
            for (uint g = 0; g < per; ++g) {
                const uint ci = args.depthwise != 0u ? co : g;
                acc += w[((co * per + g) * 3u + kt) * 3u + kf] * px[ci];
            }
        }
    }
    out[i] = args.relu != 0u ? fmaxf(acc, 0.0f) : acc;
}

// The patches of a stride-2 3x3 conv2d, so a GEMM can take the conv:
// `out[b][t][f][(kt*3 + kf)*c_in + ci] = in[b][2t+kt-pad_t][2f+kf-pad_f][ci]`,
// zero outside the plane, past `in_valid` and on output rows past
// `out_valid`. Reads the conv's argument block; `c_out`, `depthwise`,
// `has_bias` and `relu` are unread.
extern "C" __global__ void kernel_audio_im2col_s2_f32(
        const struct audio_args_conv2d args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint taps = 9u * args.c_in;
    const uint plane = args.t_out * args.f_out * taps;
    if (i >= args.batch * plane) {
        return;
    }
    const uint b = i / plane;
    const uint j = i - b * plane;
    const uint ci = j % args.c_in;
    const uint tap = (j / args.c_in) % 9u;
    const uint f = (j / taps) % args.f_out;
    const uint t = j / (taps * args.f_out);
    const int ti = (int) (2u * t + tap / 3u) - (int) args.pad_t;
    const int fi = (int) (2u * f + tap % 3u) - (int) args.pad_f;
    const uint rows = min(args.t_in, args.in_valid);
    float v = 0.0f;
    if (t < args.out_valid && ti >= 0 && (uint) ti < rows && fi >= 0 && (uint) fi < args.f_in) {
        v = in[(((uint64_t) b * args.t_in + (uint) ti) * args.f_in + (uint) fi) * args.c_in + ci];
    }
    out[i] = v;
}

extern "C" __global__ void kernel_audio_dwconv1d_f32(
        const struct audio_args_dwconv1d args, float *out, const float *in, const float *w) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.channels) {
        return;
    }
    const uint c = i % args.channels;
    const uint t = i / args.channels;
    float acc = 0.0f;
    for (uint j = 0; j < args.taps; ++j) {
        acc += w[c * args.taps + j] * in[(uint64_t) (t + j) * args.channels + c];
    }
    out[i] = acc;
}

extern "C" __global__ void kernel_audio_unfold_f32(
        const struct audio_args_unfold args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint width = args.taps * args.channels;
    if (i >= args.rows * width) {
        return;
    }
    const uint t = i / width;
    const uint j = (i - t * width) / args.channels;
    const uint c = i % args.channels;
    const int src = (int) (t + j) - (int) args.pad;
    out[i] = (src < 0 || (uint) src >= args.rows) ? 0.0f : in[(uint64_t) src * args.channels + c];
}

extern "C" __global__ void kernel_audio_transpose_f32(
        const struct audio_args_transpose args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint plane = args.rows * args.cols;
    if (i >= args.batch * plane) {
        return;
    }
    const uint b = i / plane;
    const uint r = (i - b * plane) / args.cols;
    const uint c = i % args.cols;
    out[(uint64_t) b * plane + (uint64_t) c * args.rows + r] = in[i];
}

/* ---- attention and rotation ---------------------------------------------- */

/* The relative-position half of the attention logits, once per (query, head,
 * distance): scores[i][h][t] = (q'[i] + bias_v) . rel[t] over head h's dims,
 * with q' = q * q_dim_scale. A warp takes AUDIO_ATTN_QB queries by
 * AUDIO_REL_ROWS distance rows; lanes stride the head's dims. */
extern "C" __global__ void kernel_audio_rel_scores_f32(
        const struct audio_args_window_attn args, float *scores, const float *q,
        const float *bias_v, const float *q_dim_scale, const float *rel) {
    const uint QB = AUDIO_ATTN_QB;
    const uint TR = AUDIO_REL_ROWS;
    const uint t0 = blockIdx.x * TR;
    const uint h = blockIdx.y;
    const uint i0 = blockIdx.z * QB;
    const uint lane = threadIdx.x;
    if (t0 >= args.n_d || h >= args.heads || i0 >= args.n_q) {
        return;
    }
    const uint hd = args.head_dim;
    const uint w = args.heads * hd;
    const uint hoff = h * hd;
    const uint nq = min(QB, args.n_q - i0);
    const uint nt = min(TR, args.n_d - t0);
    float acc[AUDIO_ATTN_QB][AUDIO_REL_ROWS];
    for (uint r = 0; r < QB; ++r) {
        for (uint t = 0; t < TR; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    for (uint e = lane; e < hd; e += 32u) {
        const float scale = args.has_q_dim_scale != 0u ? q_dim_scale[e] : 1.0f;
        const float bv = args.has_bias_uv != 0u ? bias_v[hoff + e] : 0.0f;
        float rv[AUDIO_REL_ROWS];
        for (uint t = 0; t < TR; ++t) {
            rv[t] = t < nt ? rel[(uint64_t) (t0 + t) * w + hoff + e] : 0.0f;
        }
        for (uint r = 0; r < QB; ++r) {
            if (r < nq) {
                const float qv = q[(uint64_t) (i0 + r) * w + hoff + e] * scale + bv;
                for (uint t = 0; t < TR; ++t) {
                    acc[r][t] += qv * rv[t];
                }
            }
        }
    }
    for (uint r = 0; r < QB; ++r) {
        for (uint t = 0; t < TR; ++t) {
            const float total = audio_warp_sum_all(acc[r][t]);
            if (lane == 0u && r < nq && t < nt) {
                scores[((uint64_t) (i0 + r) * args.heads + h) * args.n_d + t0 + t] = total;
            }
        }
    }
}

/* Attention for a block of AUDIO_ATTN_QB queries of one head per warp. The
 * block's queries (content bias and per-dim scale applied) sit in shared
 * memory; keys stream through 32 at a time, one per lane, with the relative
 * term read from the pre-pass's scores, the running max and sum moving once
 * per 32 keys, and each lane owning a slice of the head's dims for the value
 * sum. Visibility is decided per (query, key), so a block may straddle
 * chunks. */
extern "C" __global__ void kernel_audio_window_attn_f32(
        const struct audio_args_window_attn args, float *out, const float *q, const float *k,
        const float *v, const float *bias_u, const float *q_dim_scale, const float *rel_scores) {
    extern __shared__ float qa[];
    const uint QB = AUDIO_ATTN_QB;
    const uint h = blockIdx.x;
    const uint i0 = blockIdx.y * QB;
    const uint lane = threadIdx.x;
    if (h >= args.heads || i0 >= args.n_q) {
        return;
    }
    const uint hd = args.head_dim;
    const uint w = args.heads * hd;
    const uint hoff = h * hd;
    const uint nq = min(QB, args.n_q - i0);
    const bool has_rel = args.n_d > 0u;
    for (uint e = lane; e < QB * hd; e += 32u) {
        const uint r = e / hd;
        const uint d = e - r * hd;
        float qv = 0.0f;
        if (r < nq) {
            qv = q[(uint64_t) (i0 + r) * w + hoff + d];
            if (args.has_q_dim_scale != 0u) {
                qv *= q_dim_scale[d];
            }
            if (args.has_bias_uv != 0u) {
                qv += bias_u[hoff + d];
            }
        }
        qa[e] = qv;
    }
    __syncwarp();

    const uint qfirst = args.q_abs0 + i0;
    const uint qlast = qfirst + nq - 1u;
    const uint c_first = qfirst / args.chunk;
    const uint c_last = qlast / args.chunk;
    const uint lo_chunk = c_first > args.left_chunks ? c_first - args.left_chunks : 0u;
    const uint lo_abs = max(lo_chunk * args.chunk, args.k_abs0);
    const uint hi_abs = min(min((c_last + 1u) * args.chunk, args.k_end), args.k_abs0 + args.n_k);

    float m[AUDIO_ATTN_QB];
    float l[AUDIO_ATTN_QB];
    float acc[AUDIO_ATTN_QB][AUDIO_ATTN_SLOTS];
    for (uint r = 0; r < QB; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    for (uint base = lo_abs; base < hi_abs; base += 32u) {
        const uint ka = base + lane;
        const bool key_in = ka < hi_abs;
        const uint j = ka - args.k_abs0;
        const uint kc = ka / args.chunk;
        float sc[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            sc[r] = 0.0f;
        }
        if (key_in) {
            const float *krow = k + (uint64_t) j * w + hoff;
            for (uint d = 0; d < hd; ++d) {
                const float kd = krow[d];
                for (uint r = 0; r < QB; ++r) {
                    sc[r] += qa[r * hd + d] * kd;
                }
            }
        }
        for (uint r = 0; r < QB; ++r) {
            const uint qabs = qfirst + r;
            const uint qc = qabs / args.chunk;
            const bool visible = key_in && r < nq && kc <= qc && qc - kc <= args.left_chunks;
            float s_r = -INFINITY;
            if (visible) {
                float bd = 0.0f;
                if (has_rel) {
                    const int dr = (int) qabs - (int) ka - args.d_min;
                    if (dr >= 0 && (uint) dr < args.n_d) {
                        bd = rel_scores[((uint64_t) (i0 + r) * args.heads + h) * args.n_d + (uint) dr];
                    }
                }
                s_r = args.scale * (args.ac_scale * sc[r] + bd);
                if (args.softcap > 0.0f) {
                    s_r = args.softcap * tanhf(s_r / args.softcap);
                }
            }
            sc[r] = s_r;
        }
        float p[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            const float block_max = audio_warp_max_all(sc[r]);
            const float m_new = fmaxf(m[r], block_max);
            const float corr = m_new == -INFINITY ? 1.0f : expf(m[r] - m_new);
            p[r] = sc[r] == -INFINITY ? 0.0f : expf(sc[r] - m_new);
            l[r] = l[r] * corr + audio_warp_sum_all(p[r]);
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                acc[r][s] *= corr;
            }
            m[r] = m_new;
        }
        const uint keys_here = min(32u, hi_abs - base);
        for (uint jj = 0; jj < keys_here; ++jj) {
            const uint jv = base + jj - args.k_abs0;
            const float *vrow = v + (uint64_t) jv * w + hoff;
            float vv[AUDIO_ATTN_SLOTS];
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                const uint e = lane + 32u * s;
                vv[s] = e < hd ? vrow[e] : 0.0f;
            }
            for (uint r = 0; r < QB; ++r) {
                const float pj = __shfl_sync(0xffffffffu, p[r], jj);
                for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                    acc[r][s] += pj * vv[s];
                }
            }
        }
    }
    for (uint r = 0; r < nq; ++r) {
        const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            const uint e = lane + 32u * s;
            if (e < hd) {
                out[(uint64_t) (i0 + r) * w + hoff + e] = acc[r][s] * inv;
            }
        }
    }
}

/* Half-split rotary embedding, in place: for m < hd/2, (x[m], x[m+hd/2])
 * rotates by (pos0 + row) * theta^(-2m/hd). */
extern "C" __global__ void kernel_audio_rope_half_f32(
        const struct audio_args_rope args, float *x) {
    const uint half_dim = args.head_dim / 2u;
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.heads * half_dim) {
        return;
    }
    const uint m = i % half_dim;
    const uint rh = i / half_dim;
    const uint row = rh / args.heads;
    float *p = x + (uint64_t) rh * args.head_dim;
    const float inv_freq = powf(args.theta, -2.0f * (float) m / (float) args.head_dim);
    const float ang = (float) (args.pos0 + row) * inv_freq;
    float s;
    float c;
    sincosf(ang, &s, &c);
    const float a = p[m];
    const float b = p[m + half_dim];
    p[m] = a * c - b * s;
    p[m + half_dim] = b * c + a * s;
}

/* ---- the recurrent cell --------------------------------------------------- */

extern "C" __global__ void kernel_audio_lstm_cell_f32(
        const struct audio_args_lstm args, float *h_out, float *c, const float *gates) {
    const uint u = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint H = args.hidden;
    if (u >= H) {
        return;
    }
    const float gi = audio_sigmoid(gates[u]);
    const float gf = audio_sigmoid(gates[H + u]);
    const float gg = tanhf(gates[2u * H + u]);
    const float go = audio_sigmoid(gates[3u * H + u]);
    const float cn = gf * c[u] + gi * gg;
    c[u] = cn;
    h_out[u] = go * tanhf(cn);
}

/* ---- greedy transducer search on the device ------------------------------ */

/* out[r][c] = relu(pending[t + r][c] + dec[c]) for the window's rows, 0 for
 * rows at or past avail; t is read from the search state. */
extern "C" __global__ void kernel_audio_greedy_window_f32(
        const struct audio_args_greedy_window args, float *out, const float *pending,
        const float *dec, const uint *state) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.cols) {
        return;
    }
    const uint r = i / args.cols;
    const uint c = i - r * args.cols;
    const uint src = state[AUDIO_GREEDY_T] + r;
    out[i] = src < args.avail ? fmaxf(pending[(uint64_t) src * args.cols + c] + dec[c], 0.0f) : 0.0f;
}

/* One decision of greedy search over the window's argmax ids, on one thread,
 * as the Metal twin. */
extern "C" __global__ void kernel_audio_greedy_step_u32(
        const struct audio_args_greedy_step args, uint *state, uint *out, const uint *ids) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    uint t = state[AUDIO_GREEDY_T];
    uint sym = state[AUDIO_GREEDY_SYM];
    out[2u * args.slot] = AUDIO_GREEDY_NONE;
    out[2u * args.slot + 1u] = 0u;
    state[AUDIO_GREEDY_EMIT] = 0u;
    if (t >= args.avail) {
        state[AUDIO_GREEDY_DONE] = 1u;
        return;
    }
    const uint m = min(args.rows, args.avail - t);
    uint j = m;
    for (uint r = 0; r < m; ++r) {
        if (ids[r] != args.blank) {
            j = r;
            break;
        }
    }
    if (j == m) {
        t += m;
        sym = 0u;
    } else {
        if (j > 0u) {
            t += j;
            sym = 0u;
        }
        const uint k = ids[j];
        out[2u * args.slot] = k;
        out[2u * args.slot + 1u] = t;
        state[AUDIO_GREEDY_TOKEN] = k;
        state[AUDIO_GREEDY_EMIT] = 1u;
        sym += 1u;
        if (sym >= args.max_symbols) {
            t += 1u;
            sym = 0u;
        }
    }
    state[AUDIO_GREEDY_T] = t;
    state[AUDIO_GREEDY_SYM] = sym;
    state[AUDIO_GREEDY_DONE] = t >= args.avail ? 1u : 0u;
}

/* out = table[token], the state's last token's row; zeros past the table. */
extern "C" __global__ void kernel_audio_embed_row_f32(
        const struct audio_args_embed_row args, float *out, const float *table, const uint *state) {
    const uint u = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (u >= args.width) {
        return;
    }
    const uint k = state[AUDIO_GREEDY_TOKEN];
    out[u] = k < args.vocab ? table[(uint64_t) k * args.width + u] : 0.0f;
}

/* The cell above, keeping the new state only under the emit flag; x_out
 * always takes h'. */
extern "C" __global__ void kernel_audio_lstm_cell_gated_f32(
        const struct audio_args_lstm args, float *x_out, float *h, float *c, const float *gates,
        const uint *state) {
    const uint u = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint H = args.hidden;
    if (u >= H) {
        return;
    }
    const float gi = audio_sigmoid(gates[u]);
    const float gf = audio_sigmoid(gates[H + u]);
    const float gg = tanhf(gates[2u * H + u]);
    const float go = audio_sigmoid(gates[3u * H + u]);
    const float cn = gf * c[u] + gi * gg;
    const float hn = go * tanhf(cn);
    x_out[u] = hn;
    if (state[AUDIO_GREEDY_EMIT] != 0u) {
        c[u] = cn;
        h[u] = hn;
    }
}

/* dst = src when the last step emitted, else nothing. */
extern "C" __global__ void kernel_audio_keep_f32(
        const struct audio_args_keep args, float *dst, const float *src, const uint *state) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count || state[AUDIO_GREEDY_EMIT] == 0u) {
        return;
    }
    dst[i] = src[i];
}

/* ---- a causal decoder over a device KV cache ----------------------------- */

/* The half-split rotation at the decoder state's position: row `row` turns by
 * (state[POS] + pos0 + row) * theta^(-2m/hd). */
extern "C" __global__ void kernel_audio_rope_half_state_f32(
        const struct audio_args_rope args, float *x, const uint *state) {
    const uint half_dim = args.head_dim / 2u;
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.heads * half_dim) {
        return;
    }
    const uint m = i % half_dim;
    const uint rh = i / half_dim;
    const uint row = rh / args.heads;
    float *p = x + (uint64_t) rh * args.head_dim;
    const float inv_freq = powf(args.theta, -2.0f * (float) m / (float) args.head_dim);
    const float ang = (float) (state[AUDIO_DECODE_POS] + args.pos0 + row) * inv_freq;
    float s;
    float c;
    sincosf(ang, &s, &c);
    const float a = p[m];
    const float b = p[m + half_dim];
    p[m] = a * c - b * s;
    p[m + half_dim] = b * c + a * s;
}

/* rows new k and v rows, [rows][kv_heads * hd], into one layer's cache planes
 * [kv_heads][cap][hd] from the state's position on; a row at or past cap is
 * dropped, and the position is left to the step kernel. */
extern "C" __global__ void kernel_audio_kv_append_f32(
        const struct audio_args_kv_append args, float *k_cache, float *v_cache, const float *k,
        const float *v, const uint *state) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint hd = args.head_dim;
    if (i >= args.rows * args.kv_heads * hd) {
        return;
    }
    const uint e = i % hd;
    const uint rh = i / hd;
    const uint h = rh % args.kv_heads;
    const uint r = rh / args.kv_heads;
    const uint pos = state[AUDIO_DECODE_POS] + r;
    if (pos >= args.cap) {
        return;
    }
    const uint64_t dst = ((uint64_t) h * args.cap + pos) * hd + e;
    k_cache[dst] = k[i];
    v_cache[dst] = v[i];
}

/* Causal grouped-query attention of a block of AUDIO_ATTN_QB queries of one
 * head over a layer's cache planes, one warp per block, as the Metal twin:
 * query i at pos0 + i sees rows 0..=pos0 + i of kv head h / (heads /
 * kv_heads), and no row past the block's last query is read. */
extern "C" __global__ void kernel_audio_causal_attn_f32(
        const struct audio_args_cache_attn args, float *out, const float *q,
        const float *k_cache, const float *v_cache) {
    extern __shared__ float qa[];
    const uint QB = AUDIO_ATTN_QB;
    const uint h = blockIdx.x;
    const uint i0 = blockIdx.y * QB;
    const uint lane = threadIdx.x;
    if (h >= args.heads || i0 >= args.n_q) {
        return;
    }
    const uint hd = args.head_dim;
    const uint w = args.heads * hd;
    const uint hoff = h * hd;
    const uint kvh = h / (args.heads / args.kv_heads);
    const float *kp = k_cache + (uint64_t) kvh * args.cap * hd;
    const float *vp = v_cache + (uint64_t) kvh * args.cap * hd;
    const uint nq = min(QB, args.n_q - i0);
    for (uint e = lane; e < QB * hd; e += 32u) {
        const uint r = e / hd;
        const uint d = e - r * hd;
        qa[e] = r < nq ? q[(uint64_t) (i0 + r) * w + hoff + d] : 0.0f;
    }
    __syncwarp();

    const uint qfirst = args.pos0 + i0;
    const uint hi = min(qfirst + nq, args.cap);

    float m[AUDIO_ATTN_QB];
    float l[AUDIO_ATTN_QB];
    float acc[AUDIO_ATTN_QB][AUDIO_ATTN_SLOTS];
    for (uint r = 0; r < QB; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    for (uint base = 0; base < hi; base += 32u) {
        const uint ka = base + lane;
        const bool key_in = ka < hi;
        float sc[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            sc[r] = 0.0f;
        }
        if (key_in) {
            const float *krow = kp + (uint64_t) ka * hd;
            for (uint d = 0; d < hd; ++d) {
                const float kd = krow[d];
                for (uint r = 0; r < QB; ++r) {
                    sc[r] += qa[r * hd + d] * kd;
                }
            }
        }
        for (uint r = 0; r < QB; ++r) {
            const bool visible = key_in && r < nq && ka <= qfirst + r;
            sc[r] = visible ? args.scale * sc[r] : -INFINITY;
        }
        float p[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            const float block_max = audio_warp_max_all(sc[r]);
            const float m_new = fmaxf(m[r], block_max);
            const float corr = m_new == -INFINITY ? 1.0f : expf(m[r] - m_new);
            p[r] = sc[r] == -INFINITY ? 0.0f : expf(sc[r] - m_new);
            l[r] = l[r] * corr + audio_warp_sum_all(p[r]);
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                acc[r][s] *= corr;
            }
            m[r] = m_new;
        }
        const uint keys_here = min(32u, hi - base);
        for (uint jj = 0; jj < keys_here; ++jj) {
            const float *vrow = vp + (uint64_t) (base + jj) * hd;
            float vv[AUDIO_ATTN_SLOTS];
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                const uint e = lane + 32u * s;
                vv[s] = e < hd ? vrow[e] : 0.0f;
            }
            for (uint r = 0; r < QB; ++r) {
                const float pj = __shfl_sync(0xffffffffu, p[r], jj);
                for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                    acc[r][s] += pj * vv[s];
                }
            }
        }
    }
    for (uint r = 0; r < nq; ++r) {
        const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            const uint e = lane + 32u * s;
            if (e < hd) {
                out[(uint64_t) (i0 + r) * w + hoff + e] = acc[r][s] * inv;
            }
        }
    }
}

/* The one query of a decode step, per head, over cache rows 0..=state[POS] of
 * its kv head, split over the rows, as the Metal twin: one warp per (kv head,
 * split, tile of AUDIO_DECODE_GT query heads), four splits to a 128-thread
 * block, grid (kv_heads, ceil(splits / 4), ceil(group / GT)). Lanes own four
 * consecutive dims; rows go AUDIO_DECODE_KB at a time. Each (head, split)
 * leaves its unnormalised accumulator at partials[(h * splits + split) * hd]
 * and its (max, sum) pair at heads * splits * hd + 2 (h * splits + split).
 * The exponentials are the accurate ones; the Metal twin takes the fast ones
 * for speed, within 3 ulp of 1 on every weight. */
extern "C" __global__ void kernel_audio_decode_attn_f32(
        const struct audio_args_cache_attn args, float *partials, const float *q,
        const float *k_cache, const float *v_cache, const uint *state) {
    const uint KB = AUDIO_DECODE_KB;
    const uint GT = AUDIO_DECODE_GT;
    const uint lane = threadIdx.x & 31u;
    const uint sg = threadIdx.x >> 5u;
    const uint kvh = blockIdx.x;
    const uint split = blockIdx.y * AUDIO_DECODE_SPLITS_PER_TG + sg;
    const uint group = args.heads / args.kv_heads;
    const uint g0 = blockIdx.z * GT;
    if (kvh >= args.kv_heads || split >= args.splits || g0 >= group) {
        return;
    }
    const uint ng = min(GT, group - g0);
    const uint h0 = kvh * group + g0;
    const uint hd = args.head_dim;
    const uint n4 = hd / 4u;
    const bool owns = lane < n4;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 *kp = (const float4 *) (k_cache + (uint64_t) kvh * args.cap * hd);
    const float4 *vp = (const float4 *) (v_cache + (uint64_t) kvh * args.cap * hd);
    float4 qv[AUDIO_DECODE_GT];
    float m[AUDIO_DECODE_GT];
    float l[AUDIO_DECODE_GT];
    float4 acc[AUDIO_DECODE_GT];
    for (uint g = 0; g < GT; ++g) {
        qv[g] = (g < ng && owns) ? ((const float4 *) (q + (uint64_t) (h0 + g) * hd))[lane] : zero;
        m[g] = -INFINITY;
        l[g] = 0.0f;
        acc[g] = zero;
    }
    const uint len = min(state[AUDIO_DECODE_POS] + 1u, args.cap);
    const uint per = ((len + args.splits - 1u) / args.splits + KB - 1u) / KB * KB;
    const uint start = split * per;
    const uint end = min(start + per, len);
    for (uint base = start; base < end; base += KB) {
        const uint n = min(KB, end - base);
        float4 kv[AUDIO_DECODE_KB];
        float4 vv[AUDIO_DECODE_KB];
        for (uint b = 0; b < KB; ++b) {
            const bool in = b < n && owns;
            kv[b] = in ? kp[(uint64_t) (base + b) * n4 + lane] : zero;
            vv[b] = in ? vp[(uint64_t) (base + b) * n4 + lane] : zero;
        }
        for (uint g = 0; g < GT; ++g) {
            float p[AUDIO_DECODE_KB];
            float mb = -INFINITY;
            for (uint b = 0; b < KB; ++b) {
                const float dx = qv[g].x * kv[b].x;
                const float dy = qv[g].y * kv[b].y;
                const float dz = qv[g].z * kv[b].z;
                const float dw = qv[g].w * kv[b].w;
                const float s = args.scale * audio_warp_sum_all((dx + dy) + (dz + dw));
                p[b] = b < n ? s : -INFINITY;
                mb = fmaxf(mb, p[b]);
            }
            const float m_new = fmaxf(m[g], mb);
            const float corr = m[g] == -INFINITY ? 0.0f : expf(m[g] - m_new);
            float sum = 0.0f;
            float4 a = make_float4(acc[g].x * corr, acc[g].y * corr, acc[g].z * corr, acc[g].w * corr);
            for (uint b = 0; b < KB; ++b) {
                const float pb = b < n ? expf(p[b] - m_new) : 0.0f;
                sum += pb;
                a.x += pb * vv[b].x;
                a.y += pb * vv[b].y;
                a.z += pb * vv[b].z;
                a.w += pb * vv[b].w;
            }
            l[g] = l[g] * corr + sum;
            acc[g] = a;
            m[g] = m_new;
        }
    }
    const uint64_t ml0 = (uint64_t) args.heads * args.splits * hd;
    for (uint g = 0; g < ng; ++g) {
        const uint64_t slot = (uint64_t) (h0 + g) * args.splits + split;
        if (owns) {
            ((float4 *) (partials + slot * hd))[lane] = acc[g];
        }
        if (lane == 0u) {
            partials[ml0 + 2u * slot] = m[g];
            partials[ml0 + 2u * slot + 1u] = l[g];
        }
    }
}

/* Closes decode attention, one block of AUDIO_DECODE_MERGE_THREADS per head,
 * as the Metal twin: the splits' weights formed once into shared memory and
 * summed by the block reduction, warp g summing the accumulators of splits
 * g, g + 8, ... in order, and the eight partial sums added in warp order. */
extern "C" __global__ void kernel_audio_decode_attn_merge_f32(
        const struct audio_args_cache_attn args, float *out, const float *partials) {
    __shared__ float wts[AUDIO_DECODE_MAX_SPLITS];
    __shared__ float scratch[AUDIO_BLOCK_SUM_SLOTS];
    __shared__ float4 part[AUDIO_DECODE_MERGE_SG][32];
    const uint NT = AUDIO_DECODE_MERGE_THREADS;
    const uint tid = threadIdx.x;
    const uint lane = tid & 31u;
    const uint sg = tid >> 5u;
    const uint h = blockIdx.x;
    if (h >= args.heads) {
        return;
    }
    const uint hd = args.head_dim;
    const uint n4 = hd / 4u;
    const bool owns = lane < n4;
    const uint splits = min(args.splits, AUDIO_DECODE_MAX_SPLITS);
    const float *ml = partials + (uint64_t) args.heads * args.splits * hd
            + 2u * (uint64_t) h * args.splits;
    const float4 *accs = (const float4 *) (partials + (uint64_t) h * args.splits * hd);
    float mx = -INFINITY;
    for (uint s = tid; s < splits; s += NT) {
        mx = fmaxf(mx, ml[2u * s]);
    }
    mx = audio_warp_max_all(mx);
    if (lane == 0u) {
        scratch[sg] = mx;
    }
    __syncthreads();
    mx = -INFINITY;
    for (uint g = 0; g < AUDIO_DECODE_MERGE_SG; ++g) {
        mx = fmaxf(mx, scratch[g]);
    }
    __syncthreads();
    float t = 0.0f;
    for (uint s = tid; s < splits; s += NT) {
        const float ms = ml[2u * s];
        const float ws = ms == -INFINITY ? 0.0f : expf(ms - mx);
        wts[s] = ws;
        t += ml[2u * s + 1u] * ws;
    }
    const float total = audio_block_sum(t, scratch, tid, NT);
    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (owns) {
        for (uint s = sg; s < splits; s += AUDIO_DECODE_MERGE_SG) {
            const float4 v = accs[(uint64_t) s * n4 + lane];
            const float w = wts[s];
            a.x += v.x * w;
            a.y += v.y * w;
            a.z += v.z * w;
            a.w += v.w * w;
        }
    }
    part[sg][lane] = a;
    __syncthreads();
    if (sg == 0u && owns) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint g = 0; g < AUDIO_DECODE_MERGE_SG; ++g) {
            sum.x += part[g][lane].x;
            sum.y += part[g][lane].y;
            sum.z += part[g][lane].z;
            sum.w += part[g][lane].w;
        }
        const float inv = total > 0.0f ? 1.0f / total : 0.0f;
        ((float4 *) (out + (uint64_t) h * hd))[lane] =
                make_float4(sum.x * inv, sum.y * inv, sum.z * inv, sum.w * inv);
    }
}

/* One decision of a causal decoder, on one thread, as the Metal twin: the
 * position moves by `advance`, an end id stops without emitting, any other id
 * is emitted to out[slot] and stops at the max_tokens-th, and a done state
 * writes NONE, lowers the emit flag and changes nothing else. */
extern "C" __global__ void kernel_audio_decode_step_u32(
        const struct audio_args_decode_step args, uint *state, uint *out, const uint *ids) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    out[args.slot] = AUDIO_DECODE_NONE;
    state[AUDIO_DECODE_EMIT] = 0u;
    if (state[AUDIO_DECODE_DONE] != 0u) {
        return;
    }
    const uint count = state[AUDIO_DECODE_COUNT];
    if (count >= args.max_tokens) {
        state[AUDIO_DECODE_DONE] = 1u;
        state[AUDIO_DECODE_STOP] = AUDIO_DECODE_STOP_CAP;
        return;
    }
    state[AUDIO_DECODE_POS] += args.advance;
    const uint k = ids[0];
    if (k == args.eos0 || k == args.eos1) {
        state[AUDIO_DECODE_DONE] = 1u;
        state[AUDIO_DECODE_STOP] = AUDIO_DECODE_STOP_EOS;
        return;
    }
    out[args.slot] = k;
    state[AUDIO_DECODE_TOKEN] = k;
    state[AUDIO_DECODE_EMIT] = 1u;
    state[AUDIO_DECODE_COUNT] = count + 1u;
    if (count + 1u >= args.max_tokens) {
        state[AUDIO_DECODE_DONE] = 1u;
        state[AUDIO_DECODE_STOP] = AUDIO_DECODE_STOP_CAP;
    }
}

/* out = table[token] dequantised, the state's last token's row of a
 * [vocab][width] table read through the GEMM's weight readers; zeros past the
 * table. */
template <typename W>
static __device__ __forceinline__ void audio_embed_token(
        const struct audio_args_embed_token args, float *out, const void *table,
        const uint *state) {
    const uint u = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (u >= args.width) {
        return;
    }
    const uint k = state[AUDIO_DECODE_TOKEN];
    out[u] = k < args.vocab ? W::load(table, (uint64_t) k * args.width + u) : 0.0f;
}

#define AUDIO_EMBED_ENTRY(SUFFIX, W) \
    extern "C" __global__ void kernel_audio_embed_token_##SUFFIX( \
            const struct audio_args_embed_token args, float *out, const void *table, \
            const uint *state) { \
        audio_embed_token<W>(args, out, table, state); \
    }

AUDIO_EMBED_ENTRY(f32w, audio_w_f32)
AUDIO_EMBED_ENTRY(bf16w, audio_w_bf16)
AUDIO_EMBED_ENTRY(f16w, audio_w_f16)
AUDIO_EMBED_ENTRY(q8_0w, audio_w_q8_0)

/* The row above from an 8-bit affine table: per group of a row a bf16 scale
 * and bias, the bytes packed four to a little-endian u32, each value one fused
 * q * scale + bias. */
extern "C" __global__ void kernel_audio_embed_token_affine8w(
        const struct audio_args_embed_token args, float *out, const uint *words,
        const unsigned short *scales, const unsigned short *biases, const uint *state) {
    const uint u = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (u >= args.width) {
        return;
    }
    const uint k = state[AUDIO_DECODE_TOKEN];
    if (k >= args.vocab) {
        out[u] = 0.0f;
        return;
    }
    const uint64_t i = (uint64_t) k * args.width + u;
    const uint64_t g = (uint64_t) k * (args.width / args.group) + u / args.group;
    const float qv = (float) ((words[i >> 2u] >> (8u * (uint) (i & 3u))) & 0xffu);
    out[u] = fmaf(qv, audio_bf16(scales + g), audio_bf16(biases + g));
}

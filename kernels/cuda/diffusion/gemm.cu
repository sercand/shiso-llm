/* Linear-layer entries of the `diffusion` module: the scalar 16x16 GEMM, the
 * 64x64-tile GEMM (plain, k-quant, INT8, and fused with the block's swiglu
 * input and gated residual), and the ConvRot rotation.
 *
 * Include-only piece of `diffusion.cu`; ported from the Metal twin. The 64x64
 * tile is plain FMAs here (the twin uses simdgroup matrices): each of the 128
 * threads owns 32 outputs, k is staged 16 at a time, and every output is the
 * k-ascending fold of its row and column from zero, like the scalar kernel.
 * It is the same arithmetic to rounding, not a reimplementation of the
 * algorithm, so the shared kerneltest cases grade it unchanged. Tensor cores
 * are the obvious later speedup; nothing outside this file would change.
 *
 * Weights stay in the checkpoint's stored width and are widened on read; the
 * epilogue scale is a scalar (`float weight_scale`, every kind but INT8) or a
 * per-row vector (`const float* row_scale`, INT8 only), exactly as `Ops`
 * binds it (`scale_binding`).
 */

/* --- stored-width decoders -------------------------------------------------- */

static __device__ __forceinline__ float dit_weight_value(float w) { return w; }
static __device__ __forceinline__ float dit_weight_value(u16 w) {
    return __uint_as_float(((u32)w) << 16);
}
static __device__ __forceinline__ float dit_weight_value(u8 w) { return dit_e4m3_bits((u32)w); }
static __device__ __forceinline__ float dit_weight_value(i8 w) { return (float)w; }

/* The epilogue multiplier: one value for the tensor, or one per weight row
 * (== output column). */
struct dit_scale_tensor {
    float s;
    __device__ float at(u32) const { return s; }
};
struct dit_scale_rows {
    const float* s;
    __device__ float at(u32 col) const { return s[col]; }
};

/* --- the scalar 16x16 GEMM -------------------------------------------------- */

/* Block (16,16); grid (ceil(n/16), ceil(m/16)). `k` is walked strictly in
 * increasing order, the same association as the host twin's `for k in 0..K`. */
template <typename W, typename SC>
static __device__ __forceinline__ void dit_gemm_body(
        const dit_args_gemm& args, float* out, const float* in, const W* weight,
        const SC scale, const float* bias) {
    __shared__ float a_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    __shared__ float b_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    const u32 tx = threadIdx.x;
    const u32 ty = threadIdx.y;
    const u32 row = blockIdx.y * DIT_GEMM_TILE + ty;
    const u32 col = blockIdx.x * DIT_GEMM_TILE + tx;
    /* The weight row this thread stages is not the one it accumulates: the tile
     * is filled row-per-ty so k runs along x (contiguous), then read column-wise. */
    const u32 stage_n = blockIdx.x * DIT_GEMM_TILE + ty;
    const bool has_bias = args.has_bias != 0u;
    float acc = 0.0f;
    for (u32 k0 = 0u; k0 < args.k; k0 += DIT_GEMM_TILE) {
        const u32 k = k0 + tx;
        const bool have_k = k < args.k;
        a_tile[ty][tx] = (row < args.m && have_k) ? in[(u64)row * args.k + k] : 0.0f;
        b_tile[ty][tx] = (stage_n < args.n && have_k)
                ? dit_weight_value(weight[(u64)stage_n * args.k + k]) : 0.0f;
        __syncthreads();
        for (u32 kk = 0u; kk < DIT_GEMM_TILE; kk++) {
            acc += a_tile[ty][kk] * b_tile[tx][kk];
        }
        __syncthreads();
    }
    if (row < args.m && col < args.n) {
        out[(u64)row * args.n + col] = acc * scale.at(col) + (has_bias ? bias[col] : 0.0f);
    }
}

extern "C" __global__ void kernel_dit_gemm_f32(
        const dit_args_gemm args, float* out, const float* in, const float* weight,
        float weight_scale, const float* bias) {
    dit_gemm_body(args, out, in, weight, dit_scale_tensor{weight_scale}, bias);
}

extern "C" __global__ void kernel_dit_gemm_bf16(
        const dit_args_gemm args, float* out, const float* in, const u16* weight,
        float weight_scale, const float* bias) {
    dit_gemm_body(args, out, in, weight, dit_scale_tensor{weight_scale}, bias);
}

extern "C" __global__ void kernel_dit_gemm_fp8(
        const dit_args_gemm args, float* out, const float* in, const u8* weight,
        float weight_scale, const float* bias) {
    dit_gemm_body(args, out, in, weight, dit_scale_tensor{weight_scale}, bias);
}

/* INT8: binding 4 is the n-long per-row scale VECTOR, not a scalar. */
extern "C" __global__ void kernel_dit_gemm_i8(
        const dit_args_gemm args, float* out, const float* in, const i8* weight,
        const float* row_scale, const float* bias) {
    dit_gemm_body(args, out, in, weight, dit_scale_rows{row_scale}, bias);
}

/* --- operand readers for the 64x64 tile ------------------------------------- */

/* Each reader gives four consecutive k of one row as a float4, read one scalar
 * at a time: a view starts on any 4-byte boundary, so vector loads would fault.
 * `kAlongM` readers instead gather four consecutive ROWS of one column
 * (`load4m`), for the channel-first convolutions. */

/* Dense activation rows: element (r, k) of x[m][k]. */
struct dit_rows_f32 {
    static constexpr bool kAlongM = false;
    const float* x;
    u32 k;
    __device__ float4 load4(u32 r, u32 kk) const {
        const float* p = x + (u64)r * k + kk;
        return make_float4(p[0], p[1], p[2], p[3]);
    }
};

/* Weight rows (n, k) in their stored width. */
struct dit_wrows_f32 {
    const float* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const float* p = w + (u64)n * k + kk;
        return make_float4(p[0], p[1], p[2], p[3]);
    }
};
struct dit_wrows_bf16 {
    const u16* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const u16* p = w + (u64)n * k + kk;
        return make_float4(dit_bf16(p), dit_bf16(p + 1), dit_bf16(p + 2), dit_bf16(p + 3));
    }
};
/* Signed INT8 codes widened exactly; the per-row scale rides the epilogue. */
struct dit_wrows_i8 {
    const i8* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const i8* p = w + (u64)n * k + kk;
        return make_float4((float)p[0], (float)p[1], (float)p[2], (float)p[3]);
    }
};
struct dit_wrows_fp8 {
    const u8* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const u8* p = w + (u64)n * k + kk;
        return make_float4(dit_e4m3_bits(p[0]), dit_e4m3_bits(p[1]),
                           dit_e4m3_bits(p[2]), dit_e4m3_bits(p[3]));
    }
};

/* The input row of a block's output linear without materializing it: columns
 * [0, cols_a) are a's row, the rest silu(g) * u of the fused gate/up row. Both
 * widths are multiples of four, so a float4 never straddles the two parts. */
struct dit_rows_swiglu {
    static constexpr bool kAlongM = false;
    const float* a;
    const float* gu;
    u32 cols_a, inner, gu_stride, gu_offset;
    __device__ float4 load4(u32 r, u32 kk) const {
        if (kk < cols_a) {
            const float* p = a + (u64)r * cols_a + kk;
            return make_float4(p[0], p[1], p[2], p[3]);
        }
        const float* g = gu + (u64)r * gu_stride + gu_offset + (kk - cols_a);
        return make_float4(dit_silu(g[0]) * g[inner], dit_silu(g[1]) * g[inner + 1],
                           dit_silu(g[2]) * g[inner + 2], dit_silu(g[3]) * g[inner + 3]);
    }
};

/* Epilogues. `store(r, c, v)` gets one accumulator; each output element is
 * stored by exactly one thread, so the in-place variants are race-free.
 * `kRowsFast` makes adjacent threads own adjacent ROWS (for NCHW stores, where
 * a pixel row is the contiguous axis). */
template <typename SC>
struct dit_store_rows {
    static constexpr bool kRowsFast = false;
    float* out;
    u32 n;
    SC scale;
    const float* bias;
    bool has_bias;
    __device__ void store(u32 r, u32 c, float v) const {
        out[(u64)r * n + c] = v * scale.at(c) + (has_bias ? bias[c] : 0.0f);
    }
};

template <typename SC>
struct dit_store_gated {
    static constexpr bool kRowsFast = false;
    float* out;
    u32 n;
    SC scale;
    const float* bias;
    bool has_bias;
    const float* gate;
    __device__ void store(u32 r, u32 c, float v) const {
        const u64 i = (u64)r * n + c;
        out[i] += gate[c] * (v * scale.at(c) + (has_bias ? bias[c] : 0.0f));
    }
};

/* --- the 64x64 tile --------------------------------------------------------- */

/* Static shared floats: [64][16+4] activations and [16][64+4] weights. */
#define DIT_G64_SMEM (DIT_G64_TILE * (DIT_G64_K + 4u) + DIT_G64_K * (DIT_G64_TILE + 4u))

/* C[M,N] = A[M,K] . W[N,K]^T over a 64x64 output tile; block 128 threads, grid
 * (ceil(N/64), ceil(M/64)). `K` must be a multiple of 16 (the host checks).
 *
 * Thread map: p = tid % 16, q = tid / 16 (0..7). Default: rows q + 8i (i < 8),
 * columns p + 16j (j < 4). Rows-fast epilogues: rows p + 16i (i < 4), columns
 * q + 8j (j < 8), so a warp's stores land on consecutive pixels. */
template <typename AR, typename WR, typename EP>
static __device__ __forceinline__ void dit_gemm64_tile(
        u32 M, u32 N, u32 K, const AR& a, const WR& w, const EP& ep, float* smem) {
    constexpr u32 BM = DIT_G64_TILE, BN = DIT_G64_TILE, BK = DIT_G64_K, NT = DIT_G64_THREADS;
    constexpr u32 LDA = BK + 4u, LDB = BN + 4u;
    constexpr u32 SLOTS = BM * BK / 4u / NT; /* float4 slots per thread per operand */
    constexpr bool RF = EP::kRowsFast;
    constexpr u32 NI = RF ? 4u : 8u;
    constexpr u32 NJ = RF ? 8u : 4u;
    float* sa = smem;            /* [BM rows][BK k] */
    float* sb = smem + BM * LDA; /* [BK k][BN cols] */
    const u32 n0 = blockIdx.x * BN;
    const u32 m0 = blockIdx.y * BM;
    const u32 tid = threadIdx.x;
    const u32 p = tid % 16u;
    const u32 q = tid / 16u;
    float acc[NI][NJ];
#pragma unroll
    for (u32 i = 0; i < NI; ++i) {
#pragma unroll
        for (u32 j = 0; j < NJ; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    for (u32 k0 = 0u; k0 < K; k0 += BK) {
        /* Stage: slot e is row e/4, k (e%4)*4 of each operand. */
#pragma unroll
        for (u32 s = 0; s < SLOTS; ++s) {
            const u32 e = tid + s * NT;
            const u32 rr = e / (BK / 4u);
            const u32 kk = (e - rr * (BK / 4u)) * 4u;
            if constexpr (AR::kAlongM) {
                /* Slot e is column e/(BM/4), rows (e%(BM/4))*4 + 0..3. */
                const u32 ka = e / (BM / 4u);
                const u32 ma = (e - ka * (BM / 4u)) * 4u;
                const float4 av = a.load4m(m0 + ma, k0 + ka, M);
                sa[(ma + 0u) * LDA + ka] = av.x;
                sa[(ma + 1u) * LDA + ka] = av.y;
                sa[(ma + 2u) * LDA + ka] = av.z;
                sa[(ma + 3u) * LDA + ka] = av.w;
            } else {
                const float4 av = (m0 + rr < M) ? a.load4(m0 + rr, k0 + kk)
                                                : make_float4(0.f, 0.f, 0.f, 0.f);
                sa[rr * LDA + kk + 0u] = av.x;
                sa[rr * LDA + kk + 1u] = av.y;
                sa[rr * LDA + kk + 2u] = av.z;
                sa[rr * LDA + kk + 3u] = av.w;
            }
            const float4 bv = (n0 + rr < N) ? w.load4(n0 + rr, k0 + kk)
                                            : make_float4(0.f, 0.f, 0.f, 0.f);
            sb[(kk + 0u) * LDB + rr] = bv.x;
            sb[(kk + 1u) * LDB + rr] = bv.y;
            sb[(kk + 2u) * LDB + rr] = bv.z;
            sb[(kk + 3u) * LDB + rr] = bv.w;
        }
        __syncthreads();
#pragma unroll
        for (u32 kk = 0; kk < BK; ++kk) {
            float af[NI];
            float bf[NJ];
#pragma unroll
            for (u32 i = 0; i < NI; ++i) {
                af[i] = sa[(RF ? (p + 16u * i) : (q + 8u * i)) * LDA + kk];
            }
#pragma unroll
            for (u32 j = 0; j < NJ; ++j) {
                bf[j] = sb[kk * LDB + (RF ? (q + 8u * j) : (p + 16u * j))];
            }
#pragma unroll
            for (u32 i = 0; i < NI; ++i) {
#pragma unroll
                for (u32 j = 0; j < NJ; ++j) {
                    acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (u32 i = 0; i < NI; ++i) {
#pragma unroll
        for (u32 j = 0; j < NJ; ++j) {
            const u32 gm = m0 + (RF ? (p + 16u * i) : (q + 8u * i));
            const u32 gn = n0 + (RF ? (q + 8u * j) : (p + 16u * j));
            if (gm < M && gn < N) {
                ep.store(gm, gn, acc[i][j]);
            }
        }
    }
}

/* --- gemm64 entries --------------------------------------------------------- */

#define DIT_GEMM64_PARAMS(WT) \
    const dit_args_gemm args, float* out, const float* in, const WT* weight, \
    float weight_scale, const float* bias

#define DIT_GEMM64_BODY(WR) \
    __shared__ float smem[DIT_G64_SMEM]; \
    dit_gemm64_tile(args.m, args.n, args.k, dit_rows_f32{in, args.k}, WR{weight, args.k}, \
        dit_store_rows<dit_scale_tensor>{out, args.n, dit_scale_tensor{weight_scale}, bias, \
                                         args.has_bias != 0u}, \
        smem);

extern "C" __global__ void kernel_dit_gemm64_f32(DIT_GEMM64_PARAMS(float)) {
    DIT_GEMM64_BODY(dit_wrows_f32)
}
extern "C" __global__ void kernel_dit_gemm64_bf16(DIT_GEMM64_PARAMS(u16)) {
    DIT_GEMM64_BODY(dit_wrows_bf16)
}
extern "C" __global__ void kernel_dit_gemm64_fp8(DIT_GEMM64_PARAMS(u8)) {
    DIT_GEMM64_BODY(dit_wrows_fp8)
}
/* Block-quant weights (k % 256 == 0, k % 32 == 0 for Q8_0); binding 4 is the
 * unit scale. */
extern "C" __global__ void kernel_dit_gemm64_q2k(DIT_GEMM64_PARAMS(u8)) {
    DIT_GEMM64_BODY(dit_wrows_q2k)
}
extern "C" __global__ void kernel_dit_gemm64_q3k(DIT_GEMM64_PARAMS(u8)) {
    DIT_GEMM64_BODY(dit_wrows_q3k)
}
extern "C" __global__ void kernel_dit_gemm64_q4k(DIT_GEMM64_PARAMS(u8)) {
    DIT_GEMM64_BODY(dit_wrows_q4k)
}
extern "C" __global__ void kernel_dit_gemm64_q5k(DIT_GEMM64_PARAMS(u8)) {
    DIT_GEMM64_BODY(dit_wrows_q5k)
}
extern "C" __global__ void kernel_dit_gemm64_q6k(DIT_GEMM64_PARAMS(u8)) {
    DIT_GEMM64_BODY(dit_wrows_q6k)
}
extern "C" __global__ void kernel_dit_gemm64_q8_0(DIT_GEMM64_PARAMS(u8)) {
    DIT_GEMM64_BODY(dit_wrows_q8_0)
}

/* Per-row INT8: binding 4 is the n-long scale vector. */
extern "C" __global__ void kernel_dit_gemm64_i8(
        const dit_args_gemm args, float* out, const float* in, const i8* weight,
        const float* row_scale, const float* bias) {
    __shared__ float smem[DIT_G64_SMEM];
    dit_gemm64_tile(args.m, args.n, args.k, dit_rows_f32{in, args.k},
        dit_wrows_i8{weight, args.k},
        dit_store_rows<dit_scale_rows>{out, args.n, dit_scale_rows{row_scale}, bias,
                                       args.has_bias != 0u},
        smem);
}

/* --- fused linear: swiglu/concat input, gated-residual output --------------- */

/* out += gate * linear([a | swiglu(gu)]). Binding order: args, out, a, gu,
 * weight, scale, bias, gate. */
#define DIT_GEMM_FUSED_PARAMS(WT) \
    const dit_args_gemm_fused args, float* out, const float* a, const float* gu, \
    const WT* weight, float weight_scale, const float* bias, const float* gate

#define DIT_GEMM_FUSED_BODY(WR) \
    __shared__ float smem[DIT_G64_SMEM]; \
    const u32 k = args.cols_a + args.inner; \
    dit_gemm64_tile(args.m, args.n, k, \
        dit_rows_swiglu{a, gu, args.cols_a, args.inner, args.gu_stride, args.gu_offset}, \
        WR{weight, k}, \
        dit_store_gated<dit_scale_tensor>{out, args.n, dit_scale_tensor{weight_scale}, bias, \
                                          args.has_bias != 0u, gate}, \
        smem);

extern "C" __global__ void kernel_dit_gemm64_fused_f32(DIT_GEMM_FUSED_PARAMS(float)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_f32)
}
extern "C" __global__ void kernel_dit_gemm64_fused_bf16(DIT_GEMM_FUSED_PARAMS(u16)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_bf16)
}
extern "C" __global__ void kernel_dit_gemm64_fused_fp8(DIT_GEMM_FUSED_PARAMS(u8)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_fp8)
}
extern "C" __global__ void kernel_dit_gemm64_fused_q2k(DIT_GEMM_FUSED_PARAMS(u8)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_q2k)
}
extern "C" __global__ void kernel_dit_gemm64_fused_q3k(DIT_GEMM_FUSED_PARAMS(u8)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_q3k)
}
extern "C" __global__ void kernel_dit_gemm64_fused_q4k(DIT_GEMM_FUSED_PARAMS(u8)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_q4k)
}
extern "C" __global__ void kernel_dit_gemm64_fused_q5k(DIT_GEMM_FUSED_PARAMS(u8)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_q5k)
}
extern "C" __global__ void kernel_dit_gemm64_fused_q6k(DIT_GEMM_FUSED_PARAMS(u8)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_q6k)
}
extern "C" __global__ void kernel_dit_gemm64_fused_q8_0(DIT_GEMM_FUSED_PARAMS(u8)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_q8_0)
}

/* Fused per-row INT8; binding 5 is the n-long row-scale vector. A ConvRot
 * weight's input must already be rotated (the host rotates `[a | swiglu(gu)]`
 * into a packed scratch row with kernel_dit_hadamard256_f32 and binds it as
 * `a`). */
extern "C" __global__ void kernel_dit_gemm64_fused_i8(
        const dit_args_gemm_fused args, float* out, const float* a, const float* gu,
        const i8* weight, const float* row_scale, const float* bias, const float* gate) {
    __shared__ float smem[DIT_G64_SMEM];
    const u32 k = args.cols_a + args.inner;
    dit_gemm64_tile(args.m, args.n, k,
        dit_rows_swiglu{a, gu, args.cols_a, args.inner, args.gu_stride, args.gu_offset},
        dit_wrows_i8{weight, k},
        dit_store_gated<dit_scale_rows>{out, args.n, dit_scale_rows{row_scale}, bias,
                                        args.has_bias != 0u, gate},
        smem);
}

/* --- ConvRot activation rotation -------------------------------------------- */

/* x . H for each 256-wide group of a row, H = kron(H4,H4,H4,H4)/16 with
 * H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]. Four radix-4 butterflies
 * (one per base-4 digit), each y_j = S - 2*v_(3-j), S the sum of the four. */

static __device__ __forceinline__ float dit_rot_input(
        const dit_args_hadamard& args, const float* a, const float* gu, u32 r, u32 kk) {
    if (kk < args.cols_a) {
        return a[(u64)r * args.cols_a + kk];
    }
    const float* g = gu + (u64)r * args.gu_stride + args.gu_offset + (kk - args.cols_a);
    return dit_silu(g[0]) * g[args.inner];
}

static __device__ __forceinline__ float4 dit_h4(float4 v) {
    const float s = (v.x + v.y) + (v.z + v.w);
    return make_float4(s - 2.0f * v.w, s - 2.0f * v.z, s - 2.0f * v.y, s - 2.0f * v.x);
}

/* out[r,:] = rot([a | swiglu(gu)][r,:]), packed cols_a+inner wide (a multiple
 * of 256). Grid (width/256, rows) x 64: one block per (row, group), four
 * elements per thread per stage. */
extern "C" __global__ void kernel_dit_hadamard256_f32(
        const dit_args_hadamard args, float* out, const float* a, const float* gu) {
    __shared__ float v[256];
    const u32 tid = threadIdx.x;
    const u32 r = blockIdx.y;
    const u32 base = blockIdx.x * 256u;
    const u32 width = args.cols_a + args.inner;
    /* Digit 0: elements 4t..4t+3 are this thread's own. */
    float4 x = make_float4(dit_rot_input(args, a, gu, r, base + 4u * tid),
                           dit_rot_input(args, a, gu, r, base + 4u * tid + 1u),
                           dit_rot_input(args, a, gu, r, base + 4u * tid + 2u),
                           dit_rot_input(args, a, gu, r, base + 4u * tid + 3u));
    x = dit_h4(x);
    v[4u * tid] = x.x;
    v[4u * tid + 1u] = x.y;
    v[4u * tid + 2u] = x.z;
    v[4u * tid + 3u] = x.w;
    /* Digits 1 and 2 (strides 4 and 16) through shared memory. */
    for (u32 stride = 4u; stride <= 16u; stride *= 4u) {
        __syncthreads();
        const u32 lo = tid % stride;
        const u32 b = (tid - lo) * 4u + lo;
        const float4 y = dit_h4(make_float4(v[b], v[b + stride], v[b + 2u * stride],
                                            v[b + 3u * stride]));
        __syncthreads();
        v[b] = y.x;
        v[b + stride] = y.y;
        v[b + 2u * stride] = y.z;
        v[b + 3u * stride] = y.w;
    }
    __syncthreads();
    /* Digit 3 (stride 64): thread t owns t, t+64, t+128, t+192. */
    float4 y = dit_h4(make_float4(v[tid], v[tid + 64u], v[tid + 128u], v[tid + 192u]));
    y = make_float4(y.x * 0.0625f, y.y * 0.0625f, y.z * 0.0625f, y.w * 0.0625f);
    float* o = out + (u64)r * width + base;
    o[tid] = y.x;
    o[tid + 64u] = y.y;
    o[tid + 128u] = y.z;
    o[tid + 192u] = y.w;
}

// y = x * W^T (+ b) for f32 activations and f32, bf16, f16 or Q8_0 weights.
//
// A 32-row by 64-column output tile per threadgroup of 128 threads: four
// simdgroups, each owning all 32 rows of a 16-column strip as a 4x2 grid of
// 8x8 accumulators, so every staged weight element feeds 32 rows. Both
// operands are staged through threadgroup memory in f32 - 32x32 of the
// activations and 64x32 of the weights, 12 KiB - with zero fill past every
// edge, so ragged shapes need no special case in the multiply, and the
// epilogue writes only in-bounds elements. Staging loads four consecutive k
// at a time whenever the row length allows it, which is what keeps the
// weights streaming at the memory's rate rather than one scalar per lane.

struct audio_w_f32 {
    static float load(device const void *w, uint64_t i) { return ((device const float *) w)[i]; }
    static float4 load4(device const void *w, uint64_t i) {
        return *((device const float4 *) ((device const float *) w + i));
    }
};
struct audio_w_bf16 {
    static float load(device const void *w, uint64_t i) { return audio_bf16(((device const ushort *) w) + i); }
    static float4 load4(device const void *w, uint64_t i) {
        const ushort4 b = *((device const ushort4 *) ((device const ushort *) w + i));
        return float4(as_type<float>((uint) b.x << 16), as_type<float>((uint) b.y << 16),
                      as_type<float>((uint) b.z << 16), as_type<float>((uint) b.w << 16));
    }
};
struct audio_w_f16 {
    static float load(device const void *w, uint64_t i) { return (float) ((device const half *) w)[i]; }
    static float4 load4(device const void *w, uint64_t i) {
        return float4(*((device const half4 *) ((device const half *) w + i)));
    }
};
// GGUF Q8_0: 34-byte blocks of an f16 scale then 32 signed bytes, laid along
// each row. Rows are whole blocks, so element i of the flattened [n][k]
// matrix sits in block i / 32 at position i % 32; four consecutive elements
// from a multiple of four never straddle a block. Dequantising here in f32,
// rather than staging the product in half precision, is the reason this
// weight type is on this GEMM at all.
struct audio_w_q8_0 {
    static float load(device const void *w, uint64_t i) {
        device const uchar *block = (device const uchar *) w + (i >> 5u) * 34u;
        const float d = (float) *((device const half *) block);
        return d * (float) ((device const char *) block)[2u + (i & 31u)];
    }
    static float4 load4(device const void *w, uint64_t i) {
        device const uchar *block = (device const uchar *) w + (i >> 5u) * 34u;
        const float d = (float) *((device const half *) block);
        device const packed_char4 *q = (device const packed_char4 *) (block + 2u + (i & 31u));
        return d * float4(char4(*q));
    }
};

// The weight a tiled GEMM stages, as one object: a dense type's rows, or an
// 8-bit affine weight's bytes with its per-group scales and biases.
template <typename W>
struct audio_dense {
    device const void *w;
    float load(uint64_t i) const { return W::load(w, i); }
    float4 load4(uint64_t i) const { return W::load4(w, i); }
};
// MLX 8-bit affine, groups of 64: the bytes four to a little-endian u32 along
// each row, and per group a bf16 scale and bias; element i of the flattened
// [n][k] matrix (k a multiple of 64, so no group crosses a row) is the fused
// `q * scale + bias` of group i / 64, as the embedding-row fetch forms it.
struct audio_affine8 {
    device const uint *q;
    device const ushort *scales;
    device const ushort *biases;
    float load(uint64_t i) const {
        const float qv = (float) ((q[i >> 2u] >> (8u * (uint) (i & 3u))) & 0xffu);
        return fma(qv, audio_bf16(scales + (i >> 6u)), audio_bf16(biases + (i >> 6u)));
    }
    float4 load4(uint64_t i) const {
        const uint word = q[i >> 2u];
        const float s = audio_bf16(scales + (i >> 6u));
        const float b = audio_bf16(biases + (i >> 6u));
        return float4(fma((float) (word & 0xffu), s, b), fma((float) ((word >> 8u) & 0xffu), s, b),
                      fma((float) ((word >> 16u) & 0xffu), s, b), fma((float) (word >> 24u), s, b));
    }
};

template <typename R>
static inline void audio_gemm_tile(
        constant audio_args_gemm & args,
        device float *y,
        device const float *x,
        const R w,
        device const float *bias,
        threadgroup float *smem,
        uint3 tgpig,
        ushort tid,
        ushort sg) {
    const uint BM = AUDIO_GEMM_TILE_M;
    const uint BN = AUDIO_GEMM_TILE_N;
    const uint BK = AUDIO_GEMM_TILE_K;
    threadgroup float *sa = smem;           // [32 rows][32 k]
    threadgroup float *sb = smem + BM * BK; // [64 cols][32 k]
    const uint n0 = tgpig.x * BN;
    const uint m0 = tgpig.y * BM;
    const uint c0 = sg * 16u;
    const bool vec = (args.k & 3u) == 0u;
    // Split k: slice z covers [k_begin, k_end) and writes a partial tile into
    // its own plane of `y`, which the split-sum pass then adds up. With one
    // split the slice is all of k and the plane is `y` itself.
    const uint k_begin = tgpig.z * args.k_chunk;
    const uint k_end = min(args.k, k_begin + args.k_chunk);
    device float *out = y + (uint64_t) tgpig.z * args.m * args.n;

    simdgroup_float8x8 acc[4][2];
    AUDIO_UNROLL for (uint i = 0; i < 4u; ++i) {
        AUDIO_UNROLL for (uint j = 0; j < 2u; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    for (uint k0 = k_begin; k0 < k_end; k0 += BK) {
        // 32x32 activations: 256 float4 slots over 128 threads, two each.
        for (uint e = tid; e < BM * BK / 4u; e += 128u) {
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            const uint gm = m0 + rr;
            const uint gk = k0 + kk;
            float4 v = float4(0.0f);
            if (gm < args.m) {
                device const float *row = x + (uint64_t) gm * args.k;
                if (vec && gk + 3u < k_end) {
                    v = *((device const float4 *) (row + gk));
                } else {
                    for (uint q = 0; q < 4u; ++q) {
                        v[q] = gk + q < k_end ? row[gk + q] : 0.0f;
                    }
                }
            }
            *((threadgroup float4 *) (sa + rr * BK + kk)) = v;
        }
        // 64x32 weights: 512 float4 slots, four each.
        for (uint e = tid; e < BN * BK / 4u; e += 128u) {
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            const uint gn = n0 + rr;
            const uint gk = k0 + kk;
            float4 v = float4(0.0f);
            if (gn < args.n) {
                const uint64_t base = (uint64_t) gn * args.k;
                if (vec && gk + 3u < k_end) {
                    v = w.load4(base + gk);
                } else {
                    for (uint q = 0; q < 4u; ++q) {
                        v[q] = gk + q < k_end ? w.load(base + gk + q) : 0.0f;
                    }
                }
            }
            *((threadgroup float4 *) (sb + rr * BK + kk)) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        AUDIO_UNROLL for (uint kk = 0; kk < BK; kk += 8u) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[2];
            AUDIO_UNROLL for (uint i = 0; i < 4u; ++i) {
                simdgroup_load(a[i], sa + (8u * i) * BK + kk, BK);
            }
            AUDIO_UNROLL for (uint j = 0; j < 2u; ++j) {
                // Transposed load: b[j](r, c) = W[c0 + 8j + c][k0 + kk + r].
                simdgroup_load(b[j], sb + (c0 + 8u * j) * BK + kk, BK, ulong2(0, 0), true);
            }
            AUDIO_UNROLL for (uint i = 0; i < 4u; ++i) {
                AUDIO_UNROLL for (uint j = 0; j < 2u; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The staged tiles are done with; the 32x64 result fits in their place.
    threadgroup float *sc = smem;
    AUDIO_UNROLL for (uint i = 0; i < 4u; ++i) {
        AUDIO_UNROLL for (uint j = 0; j < 2u; ++j) {
            simdgroup_store(acc[i][j], sc + (8u * i) * BN + c0 + 8u * j, BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A partial tile carries no bias: the split-sum pass adds it once.
    const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u && args.splits <= 1u;
    for (uint e = tid; e < BM * BN; e += 128u) {
        const uint rr = e / BN;
        const uint cc = e - rr * BN;
        const uint gm = m0 + rr;
        const uint gn = n0 + cc;
        if (gm < args.m && gn < args.n) {
            out[(uint64_t) gm * args.n + gn] = sc[e] + (has_bias ? bias[gn] : 0.0f);
        }
    }
}

// The same product for many rows: a 64x64 tile per threadgroup of four
// simdgroups, each owning a 32x32 quarter as a 4x4 grid of 8x8 accumulators,
// so each staged operand feeds four times the multiplies it does in the
// 32-row kernel. k is staged 16 at a time: the activations as `[row][k]`, the
// weights transposed to `[k][col]` so every fragment load is a plain one, each
// staged row padded by four floats against bank conflicts - 9.25 KiB, small
// enough for several threadgroups a core. The next slice's global reads are
// issued after the current slice's multiplies, which keeps them out of the
// registers the accumulators need.
//
// It takes whole 16-k slices of all of k - `k % 16 == 0`, one split, rows
// float4-aligned - and the host sends any other product to the 32-row
// kernel. Carrying a split's offsets or a ragged slice's checks cost it 8 to
// 14% (M1 Max): measured over a 969-row decoder layer (Q8_0), 22.9 ms
// against 27.7 for the 8-simdgroup 32x16 layout it replaced, whose 20 KiB of
// staging held fewer threadgroups; 128-row, 128-column, double-buffered,
// 8-deep and prefetching variants were all slower. It is slower than the
// 32-row kernel for 32 rows or fewer, which is why both exist.
//
// Every output element is still the multiply-accumulates of its row and
// column over 8-k steps in k order from zero, as in every tile shape here and
// in the 32-row kernel without a split, so the layout changes speed and never
// a value.
template <typename R>
static inline void audio_gemm64_tile(
        constant audio_args_gemm & args,
        device float *y,
        device const float *x,
        const R w,
        device const float *bias,
        threadgroup float *smem,
        uint3 tgpig,
        ushort tid,
        ushort lane,
        ushort sg) {
    const uint BM = AUDIO_GEMM64_TILE_M;
    const uint BN = AUDIO_GEMM_TILE_N;
    const uint BK = AUDIO_GEMM64_TILE_K;
    const uint NT = AUDIO_GEMM64_THREADS;
    const uint LDA = BK + 4u;
    const uint LDB = BN + 4u;
    threadgroup float *sa = smem;            // [BM rows][BK k]
    threadgroup float *sb = smem + BM * LDA; // [BK k][BN cols]
    const uint n0 = tgpig.x * BN;
    const uint m0 = tgpig.y * BM;
    // Four simdgroups in a 2x2 grid, each owning 32 rows by 32 columns.
    const uint r0 = (sg >> 1u) * 32u;
    const uint c0 = (sg & 1u) * 32u;
    // Each thread's two float4 slots of the activations and of the weights:
    // slot e of a 64x16 operand is row e / 4, k (e % 4) * 4.
    const uint SLOTS = BM * BK / 4u / NT;
    float4 ra[SLOTS];
    float4 rb[SLOTS];
    // One slice's global reads; rows and columns past the edge read as zero.
    auto fetch = [&](uint k0) {
        AUDIO_UNROLL for (uint q = 0; q < SLOTS; ++q) {
            const uint e = tid + q * NT;
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            const uint gm = m0 + rr;
            ra[q] = gm < args.m ? *((device const float4 *) (x + (uint64_t) gm * args.k + k0 + kk))
                                : float4(0.0f);
        }
        AUDIO_UNROLL for (uint q = 0; q < SLOTS; ++q) {
            const uint e = tid + q * NT;
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            const uint gn = n0 + rr;
            rb[q] = gn < args.n ? w.load4((uint64_t) gn * args.k + k0 + kk) : float4(0.0f);
        }
    };

    simdgroup_float8x8 acc[4][4];
    AUDIO_UNROLL for (uint i = 0; i < 4u; ++i) {
        AUDIO_UNROLL for (uint j = 0; j < 4u; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    fetch(0);
    for (uint k0 = 0; k0 < args.k; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        AUDIO_UNROLL for (uint q = 0; q < SLOTS; ++q) {
            const uint e = tid + q * NT;
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            *((threadgroup float4 *) (sa + rr * LDA + kk)) = ra[q];
            sb[(kk + 0u) * LDB + rr] = rb[q].x;
            sb[(kk + 1u) * LDB + rr] = rb[q].y;
            sb[(kk + 2u) * LDB + rr] = rb[q].z;
            sb[(kk + 3u) * LDB + rr] = rb[q].w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        AUDIO_UNROLL for (uint kk = 0; kk < BK; kk += 8u) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[4];
            AUDIO_UNROLL for (uint i = 0; i < 4u; ++i) {
                simdgroup_load(a[i], sa + (r0 + 8u * i) * LDA + kk, LDA);
            }
            AUDIO_UNROLL for (uint j = 0; j < 4u; ++j) {
                // b[j](r, c) = W[c0 + 8j + c][k0 + kk + r].
                simdgroup_load(b[j], sb + kk * LDB + c0 + 8u * j, LDB);
            }
            AUDIO_UNROLL for (uint i = 0; i < 4u; ++i) {
                AUDIO_UNROLL for (uint j = 0; j < 4u; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        if (k0 + BK < args.k) {
            fetch(k0 + BK);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The staged operands are done with; each simdgroup writes its fragments
    // through its own 8x8 slot at the start of the tile.
    threadgroup float *slot = smem + sg * 64u;
    const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u;
    AUDIO_UNROLL for (uint i = 0; i < 4u; ++i) {
        AUDIO_UNROLL for (uint j = 0; j < 4u; ++j) {
            simdgroup_store(acc[i][j], slot, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                const uint gm = m0 + r0 + 8u * i + (e >> 3u);
                const uint gn = n0 + c0 + 8u * j + (e & 7u);
                if (gm < args.m && gn < args.n) {
                    y[(uint64_t) gm * args.n + gn] = slot[e] + (has_bias ? bias[gn] : 0.0f);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

/// `y[i] = sum_z parts[z][i] (+ b[i % n])`: closes a split-k GEMM, summing
/// the planes in order so the result does not depend on scheduling.
kernel void kernel_audio_split_sum_f32(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *parts       [[buffer(2)]],
        device const float *bias        [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
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

template <typename W>
kernel void kernel_audio_gemm(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const void *w            [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        threadgroup float *smem         [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    audio_gemm_tile(args, y, x, audio_dense<W>{w}, bias, smem, tgpig, tid, sg);
}

template <typename W>
kernel void kernel_audio_gemm64(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const void *w            [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        threadgroup float *smem         [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    audio_gemm64_tile(args, y, x, audio_dense<W>{w}, bias, smem, tgpig, tid, lane, sg);
}

/// The tiled GEMMs over an 8-bit affine weight: the bytes where a dense
/// weight goes, then its scales and biases.
kernel void kernel_audio_gemm_affine8w(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const uint *q            [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        device const ushort *scales     [[buffer(5)]],
        device const ushort *biases     [[buffer(6)]],
        threadgroup float *smem         [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    audio_gemm_tile(args, y, x, audio_affine8{q, scales, biases}, bias, smem, tgpig, tid, sg);
}

kernel void kernel_audio_gemm64_affine8w(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const uint *q            [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        device const ushort *scales     [[buffer(5)]],
        device const ushort *biases     [[buffer(6)]],
        threadgroup float *smem         [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    audio_gemm64_tile(args, y, x, audio_affine8{q, scales, biases}, bias, smem, tgpig, tid, lane, sg);
}

typedef decltype(kernel_audio_gemm<audio_w_f32>) audio_gemm_t;
template [[host_name("kernel_audio_gemm_f32w")]]  kernel audio_gemm_t kernel_audio_gemm<audio_w_f32>;
template [[host_name("kernel_audio_gemm_bf16w")]] kernel audio_gemm_t kernel_audio_gemm<audio_w_bf16>;
template [[host_name("kernel_audio_gemm_f16w")]]  kernel audio_gemm_t kernel_audio_gemm<audio_w_f16>;
template [[host_name("kernel_audio_gemm_q8_0w")]] kernel audio_gemm_t kernel_audio_gemm<audio_w_q8_0>;

typedef decltype(kernel_audio_gemm64<audio_w_f32>) audio_gemm64_t;
template [[host_name("kernel_audio_gemm64_f32w")]]  kernel audio_gemm64_t kernel_audio_gemm64<audio_w_f32>;
template [[host_name("kernel_audio_gemm64_bf16w")]] kernel audio_gemm64_t kernel_audio_gemm64<audio_w_bf16>;
template [[host_name("kernel_audio_gemm64_f16w")]]  kernel audio_gemm64_t kernel_audio_gemm64<audio_w_f16>;
template [[host_name("kernel_audio_gemm64_q8_0w")]] kernel audio_gemm64_t kernel_audio_gemm64<audio_w_q8_0>;

// y = x * W^T (+ b) for a handful of rows: the matrix-vector shape. The tiled
// GEMM above makes one threadgroup per 64 output columns, which for a few rows
// is too few threadgroups to keep the memory busy - a 1024-column projection
// is sixteen of them. Here each simdgroup owns 4 output columns of up to 8
// rows and streams those columns' weights along k four at a time across its
// lanes, so a threadgroup of four simdgroups covers 16 columns and the grid
// is `n / 16` by `m / 8`. Each lane keeps its 8x4 partial sums in registers;
// one simdgroup reduction per output closes them.
template <typename W>
kernel void kernel_audio_gemv(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const void *w            [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    const uint MT = AUDIO_GEMV_ROWS;
    const uint NC = AUDIO_GEMV_COLS;
    const uint n0 = (tgpig.x * 4u + sg) * NC;
    const uint m0 = tgpig.y * MT;
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
    const bool vec = (args.k & 3u) == 0u;
    if (vec) {
        for (uint k4 = lane * 4u; k4 < args.k; k4 += 128u) {
            float4 wv[AUDIO_GEMV_COLS];
            for (uint c = 0; c < NC; ++c) {
                wv[c] = c < cols ? W::load4(w, (uint64_t) (n0 + c) * args.k + k4) : float4(0.0f);
            }
            for (uint r = 0; r < rows; ++r) {
                const float4 xv = *((device const float4 *) (x + (uint64_t) (m0 + r) * args.k + k4));
                for (uint c = 0; c < NC; ++c) {
                    acc[r][c] += dot(xv, wv[c]);
                }
            }
        }
    } else {
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
    }
    const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u;
    for (uint r = 0; r < rows; ++r) {
        for (uint c = 0; c < cols; ++c) {
            const float total = simd_sum(acc[r][c]);
            if (lane == 0u) {
                y[(uint64_t) (m0 + r) * args.n + n0 + c] = total + (has_bias ? bias[n0 + c] : 0.0f);
            }
        }
    }
}

typedef decltype(kernel_audio_gemv<audio_w_f32>) audio_gemv_t;
template [[host_name("kernel_audio_gemv_f32w")]]  kernel audio_gemv_t kernel_audio_gemv<audio_w_f32>;
template [[host_name("kernel_audio_gemv_bf16w")]] kernel audio_gemv_t kernel_audio_gemv<audio_w_bf16>;
template [[host_name("kernel_audio_gemv_f16w")]]  kernel audio_gemv_t kernel_audio_gemv<audio_w_f16>;
template [[host_name("kernel_audio_gemv_q8_0w")]] kernel audio_gemv_t kernel_audio_gemv<audio_w_q8_0>;

// y = x * W^T (+ b) for one row at a time, the shape of a decoder step. The
// matrix-vector kernel above keeps a lane on four consecutive weights of each
// of four columns, which for Q8_0 is a 4-byte read per column per step and
// streamed at about 150 GB/s on M1 Max. Here each lane dots a run of eight
// consecutive weights - a Q8_0 quarter block, or 16 bytes of bf16 - so a
// simdgroup reads 256 weights of a column per step, and two simdgroups share
// each column's k range; measured 330-410 GB/s on the same shapes, and the
// 151936-column head at about 355 GB/s against 175. The summation order
// differs from the matrix-vector kernel's, which is why both exist: a lane
// sums its runs in k order, `simd_sum` closes each simdgroup, and the two
// simdgroups' sums are added in simdgroup order.
// One run's eight weights as the multi-row kernels hold them, read once and
// dotted with each row of a block: the weights as floats and, for a block
// format, the run's scale (and an affine group's bias). `apply8(load8(..))`
// is each type's `dot8`, term for term.
struct audio_run8_vals {
    float4 w0;
    float4 w1;
    float d;
    float b;
};
static inline float audio_apply8_plain(const thread audio_run8_vals & v, float4 x0, float4 x1) {
    return dot(v.w0, x0) + dot(v.w1, x1);
}

struct audio_run8_f32 {
    static uint64_t row_bytes(uint k) { return (uint64_t) k * 4u; }
    static audio_run8_vals load8(device const uchar *row, uint j) {
        device const float4 *p = (device const float4 *) (row + (uint64_t) j * 32u);
        return audio_run8_vals{p[0], p[1], 0.0f, 0.0f};
    }
    static float apply8(const thread audio_run8_vals & v, float4 x0, float4 x1) {
        return audio_apply8_plain(v, x0, x1);
    }
    static float dot8(device const uchar *row, uint j, float4 x0, float4 x1) {
        device const float4 *p = (device const float4 *) (row + (uint64_t) j * 32u);
        return dot(p[0], x0) + dot(p[1], x1);
    }
};
struct audio_run8_bf16 {
    static uint64_t row_bytes(uint k) { return (uint64_t) k * 2u; }
    static audio_run8_vals load8(device const uchar *row, uint j) {
        const uint4 b = *((device const uint4 *) (row + (uint64_t) j * 16u));
        const float4 w0 = float4(as_type<float>(b.x << 16), as_type<float>(b.x & 0xffff0000u),
                                 as_type<float>(b.y << 16), as_type<float>(b.y & 0xffff0000u));
        const float4 w1 = float4(as_type<float>(b.z << 16), as_type<float>(b.z & 0xffff0000u),
                                 as_type<float>(b.w << 16), as_type<float>(b.w & 0xffff0000u));
        return audio_run8_vals{w0, w1, 0.0f, 0.0f};
    }
    static float apply8(const thread audio_run8_vals & v, float4 x0, float4 x1) {
        return audio_apply8_plain(v, x0, x1);
    }
    static float dot8(device const uchar *row, uint j, float4 x0, float4 x1) {
        const uint4 b = *((device const uint4 *) (row + (uint64_t) j * 16u));
        const float4 w0 = float4(as_type<float>(b.x << 16), as_type<float>(b.x & 0xffff0000u),
                                 as_type<float>(b.y << 16), as_type<float>(b.y & 0xffff0000u));
        const float4 w1 = float4(as_type<float>(b.z << 16), as_type<float>(b.z & 0xffff0000u),
                                 as_type<float>(b.w << 16), as_type<float>(b.w & 0xffff0000u));
        return dot(w0, x0) + dot(w1, x1);
    }
};
struct audio_run8_f16 {
    static uint64_t row_bytes(uint k) { return (uint64_t) k * 2u; }
    static audio_run8_vals load8(device const uchar *row, uint j) {
        device const half4 *p = (device const half4 *) (row + (uint64_t) j * 16u);
        return audio_run8_vals{float4(p[0]), float4(p[1]), 0.0f, 0.0f};
    }
    static float apply8(const thread audio_run8_vals & v, float4 x0, float4 x1) {
        return audio_apply8_plain(v, x0, x1);
    }
    static float dot8(device const uchar *row, uint j, float4 x0, float4 x1) {
        device const half4 *p = (device const half4 *) (row + (uint64_t) j * 16u);
        return dot(float4(p[0]), x0) + dot(float4(p[1]), x1);
    }
};
// A run is a quarter of a 34-byte block. Blocks are only 2-byte aligned, so
// the eight quants are read as four halfwords; the run's dot is scaled once.
struct audio_run8_q8_0 {
    static uint64_t row_bytes(uint k) { return (uint64_t) (k >> 5u) * 34u; }
    static audio_run8_vals load8(device const uchar *row, uint j) {
        device const uchar *block = row + (uint64_t) (j >> 2u) * 34u;
        const float d = (float) *((device const half *) block);
        device const ushort *qs = (device const ushort *) (block + 2u + (j & 3u) * 8u);
        const float4 q0 = float4(as_type<char4>((uint) qs[0] | ((uint) qs[1] << 16u)));
        const float4 q1 = float4(as_type<char4>((uint) qs[2] | ((uint) qs[3] << 16u)));
        return audio_run8_vals{q0, q1, d, 0.0f};
    }
    static float apply8(const thread audio_run8_vals & v, float4 x0, float4 x1) {
        return (dot(v.w0, x0) + dot(v.w1, x1)) * v.d;
    }
    static float dot8(device const uchar *row, uint j, float4 x0, float4 x1) {
        device const uchar *block = row + (uint64_t) (j >> 2u) * 34u;
        const float d = (float) *((device const half *) block);
        device const ushort *qs = (device const ushort *) (block + 2u + (j & 3u) * 8u);
        const float4 q0 = float4(as_type<char4>((uint) qs[0] | ((uint) qs[1] << 16u)));
        const float4 q1 = float4(as_type<char4>((uint) qs[2] | ((uint) qs[3] << 16u)));
        return (dot(q0, x0) + dot(q1, x1)) * d;
    }
};

// A matrix-vector-by-runs weight as one object: `row(c)` is column c's
// reader, whose `dot8(j, x0, x1)` dots run j. A dense type's row is a byte
// pointer.
template <typename W>
struct audio_runs {
    device const void *w;
    uint k;
    typedef device const uchar *row_t;
    row_t row(uint c) const { return (device const uchar *) w + (uint64_t) c * W::row_bytes(k); }
    float dot8(row_t r, uint j, float4 x0, float4 x1) const { return W::dot8(r, j, x0, x1); }
    audio_run8_vals load8(row_t r, uint j) const { return W::load8(r, j); }
    float apply8(const thread audio_run8_vals & v, float4 x0, float4 x1) const {
        return W::apply8(v, x0, x1);
    }
};
// MLX 8-bit affine, groups of 64: a run is two words of one group, and its
// dot is `scale * (q . x) + bias * sum(x)`. A row is only the column's first
// group: its words are 16 groups' worth of u32 further on, and keeping one
// index a row rather than three pointers keeps the gated pair's eight rows
// in registers (three pointers a row measured 2x slower there).
struct audio_runs_affine8 {
    device const uint *q;
    device const ushort *scales;
    device const ushort *biases;
    uint k;
    typedef uint row_t;
    row_t row(uint c) const { return c * (k >> 6u); }
    float dot8(row_t g0, uint j, float4 x0, float4 x1) const {
        const uint g = g0 + (j >> 3u);
        const uint2 w = *((device const uint2 *) (q + (uint64_t) g * 16u + 2u * (j & 7u)));
        const float4 q0 = float4(as_type<uchar4>(w.x));
        const float4 q1 = float4(as_type<uchar4>(w.y));
        const float sc = audio_bf16(scales + g);
        const float bi = audio_bf16(biases + g);
        const float4 one = float4(1.0f);
        return sc * (dot(q0, x0) + dot(q1, x1)) + bi * (dot(x0, one) + dot(x1, one));
    }
    audio_run8_vals load8(row_t g0, uint j) const {
        const uint g = g0 + (j >> 3u);
        const uint2 w = *((device const uint2 *) (q + (uint64_t) g * 16u + 2u * (j & 7u)));
        return audio_run8_vals{float4(as_type<uchar4>(w.x)), float4(as_type<uchar4>(w.y)),
                               audio_bf16(scales + g), audio_bf16(biases + g)};
    }
    float apply8(const thread audio_run8_vals & v, float4 x0, float4 x1) const {
        const float4 one = float4(1.0f);
        return v.d * (dot(v.w0, x0) + dot(v.w1, x1)) + v.b * (dot(x0, one) + dot(x1, one));
    }
};

/// One matrix-vector-by-runs tile: columns `[n0, n0 + 4)` of `w` against
/// the row `xr`. The sum of each column lands in `part` (the two simdgroups'
/// halves, `[col][sg]`), and after the barrier lane `c` of simdgroup 0 adds
/// them into the column's total, which is what it returns; every other thread
/// returns 0. A column past the edge reads the last one and is never written,
/// so the loop needs no branch.
template <typename R>
static inline float audio_matvec_tile(const R w, uint n, uint k, uint n0,
                                      device const float *xr, threadgroup float *part,
                                      ushort lane, ushort sg) {
    const uint NR = AUDIO_MATVEC_COLS;
    const uint NSG = AUDIO_MATVEC_SG;
    const uint runs = k >> 3u;
    typename R::row_t wr[AUDIO_MATVEC_COLS];
    for (uint c = 0; c < NR; ++c) {
        wr[c] = w.row(min(n0 + c, n - 1u));
    }
    float acc[AUDIO_MATVEC_COLS];
    for (uint c = 0; c < NR; ++c) {
        acc[c] = 0.0f;
    }
    for (uint j = sg * 32u + lane; j < runs; j += NSG * 32u) {
        device const float4 *xp = (device const float4 *) (xr + j * 8u);
        const float4 x0 = xp[0];
        const float4 x1 = xp[1];
        for (uint c = 0; c < NR; ++c) {
            acc[c] += w.dot8(wr[c], j, x0, x1);
        }
    }
    for (uint c = 0; c < NR; ++c) {
        acc[c] = simd_sum(acc[c]);
    }
    if (lane == 0u) {
        for (uint c = 0; c < NR; ++c) {
            part[c * NSG + sg] = acc[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (sg == 0u && lane < NR) {
        for (uint s = 0; s < NSG; ++s) {
            total += part[lane * NSG + s];
        }
    }
    return total;
}

template <typename R>
static inline void audio_matvec_body(constant audio_args_gemm & args, device float *y,
                                     device const float *x, const R w, device const float *bias,
                                     threadgroup float *part, uint3 tgpig, ushort lane, ushort sg) {
    const uint n0 = tgpig.x * AUDIO_MATVEC_COLS;
    const uint row = tgpig.y;
    const float total = audio_matvec_tile(w, args.n, args.k, n0, x + (uint64_t) row * args.k,
                                          part, lane, sg);
    if (sg == 0u && lane < AUDIO_MATVEC_COLS && n0 + lane < args.n) {
        const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u;
        y[(uint64_t) row * args.n + n0 + lane] = total + (has_bias ? bias[n0 + lane] : 0.0f);
    }
}

template <typename W>
kernel void kernel_audio_matvec(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const void *w            [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec_body(args, y, x, audio_runs<W>{w, args.k}, bias, part, tgpig, lane, sg);
}

/// The one-product kernel over an 8-bit affine weight: the bytes where a
/// dense weight goes, then its scales and biases.
kernel void kernel_audio_matvec_affine8w(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const uint *q            [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        device const ushort *scales     [[buffer(5)]],
        device const ushort *biases     [[buffer(6)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec_body(args, y, x, audio_runs_affine8{q, scales, biases, args.k}, bias, part, tgpig,
                      lane, sg);
}

/// Three matrix-vector-by-runs products of one row in one launch - a
/// decoder step's query, key and value projections: threadgroups
/// `[0, t0)` take `w0`'s columns, `[t0, t0 + t1)` `w1`'s and the rest `w2`'s,
/// `t_i = ceil(n_i / 4)`, each column summed exactly as the one-product
/// kernel sums it. No bias.
template <typename R>
static inline void audio_matvec3_body(constant audio_args_matvec3 & args, device float *y0,
                                      device float *y1, device float *y2, device const float *x,
                                      const R w0, const R w1, const R w2,
                                      threadgroup float *part, uint3 tgpig, ushort lane,
                                      ushort sg) {
    const uint t0 = (args.n0 + AUDIO_MATVEC_COLS - 1u) / AUDIO_MATVEC_COLS;
    const uint t1 = (args.n1 + AUDIO_MATVEC_COLS - 1u) / AUDIO_MATVEC_COLS;
    uint t = tgpig.x;
    device float *y = y0;
    R w = w0;
    uint n = args.n0;
    if (t >= t0 + t1) {
        t -= t0 + t1;
        y = y2;
        w = w2;
        n = args.n2;
    } else if (t >= t0) {
        t -= t0;
        y = y1;
        w = w1;
        n = args.n1;
    }
    const uint n0 = t * AUDIO_MATVEC_COLS;
    const float total = audio_matvec_tile(w, n, args.k, n0, x, part, lane, sg);
    if (sg == 0u && lane < AUDIO_MATVEC_COLS && n0 + lane < n) {
        y[n0 + lane] = total + 0.0f;
    }
}

template <typename W>
kernel void kernel_audio_matvec3(
        constant audio_args_matvec3 & args [[buffer(0)]],
        device float *y0                   [[buffer(1)]],
        device float *y1                   [[buffer(2)]],
        device float *y2                   [[buffer(3)]],
        device const float *x              [[buffer(4)]],
        device const void *w0              [[buffer(5)]],
        device const void *w1              [[buffer(6)]],
        device const void *w2              [[buffer(7)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec3_body(args, y0, y1, y2, x, audio_runs<W>{w0, args.k}, audio_runs<W>{w1, args.k},
                       audio_runs<W>{w2, args.k}, part, tgpig, lane, sg);
}

/// The three-product kernel over 8-bit affine weights: each weight's bytes,
/// then each one's scales, then each one's biases.
kernel void kernel_audio_matvec3_affine8w(
        constant audio_args_matvec3 & args [[buffer(0)]],
        device float *y0                   [[buffer(1)]],
        device float *y1                   [[buffer(2)]],
        device float *y2                   [[buffer(3)]],
        device const float *x              [[buffer(4)]],
        device const uint *q0              [[buffer(5)]],
        device const uint *q1              [[buffer(6)]],
        device const uint *q2              [[buffer(7)]],
        device const ushort *s0            [[buffer(8)]],
        device const ushort *s1            [[buffer(9)]],
        device const ushort *s2            [[buffer(10)]],
        device const ushort *b0            [[buffer(11)]],
        device const ushort *b1            [[buffer(12)]],
        device const ushort *b2            [[buffer(13)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec3_body(args, y0, y1, y2, x, audio_runs_affine8{q0, s0, b0, args.k},
                       audio_runs_affine8{q1, s1, b1, args.k},
                       audio_runs_affine8{q2, s2, b2, args.k}, part, tgpig, lane, sg);
}

/// A SiLU-gated MLP's two projections of one row and their product in one
/// launch: `y[c] = g * sigmoid(g) * u` with `g = x . wg[c]` and
/// `u = x . wu[c]`, each column summed exactly as the one-product kernel
/// sums it (the two share a loop, so both weights stream at once) and the
/// product formed as the SiLU-product kernel forms it. No bias.
template <typename R>
static inline void audio_matvec_silu_body(constant audio_args_gemm & args, device float *y,
                                          device const float *x, const R wg, const R wu,
                                          threadgroup float *part, uint3 tgpig, ushort lane,
                                          ushort sg) {
    const uint NR = AUDIO_MATVEC_COLS;
    const uint NSG = AUDIO_MATVEC_SG;
    const uint n0 = tgpig.x * NR;
    const uint row = tgpig.y;
    device const float *xr = x + (uint64_t) row * args.k;
    const uint runs = args.k >> 3u;
    typename R::row_t gr[AUDIO_MATVEC_COLS];
    typename R::row_t ur[AUDIO_MATVEC_COLS];
    for (uint c = 0; c < NR; ++c) {
        const uint col = min(n0 + c, args.n - 1u);
        gr[c] = wg.row(col);
        ur[c] = wu.row(col);
    }
    float ag[AUDIO_MATVEC_COLS];
    float au[AUDIO_MATVEC_COLS];
    for (uint c = 0; c < NR; ++c) {
        ag[c] = 0.0f;
        au[c] = 0.0f;
    }
    for (uint j = sg * 32u + lane; j < runs; j += NSG * 32u) {
        device const float4 *xp = (device const float4 *) (xr + j * 8u);
        const float4 x0 = xp[0];
        const float4 x1 = xp[1];
        for (uint c = 0; c < NR; ++c) {
            ag[c] += wg.dot8(gr[c], j, x0, x1);
            au[c] += wu.dot8(ur[c], j, x0, x1);
        }
    }
    for (uint c = 0; c < NR; ++c) {
        ag[c] = simd_sum(ag[c]);
        au[c] = simd_sum(au[c]);
    }
    if (lane == 0u) {
        for (uint c = 0; c < NR; ++c) {
            part[c * NSG + sg] = ag[c];
            part[(NR + c) * NSG + sg] = au[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < NR && n0 + lane < args.n) {
        float g = 0.0f;
        float u = 0.0f;
        for (uint s = 0; s < NSG; ++s) {
            g += part[lane * NSG + s];
            u += part[(NR + lane) * NSG + s];
        }
        g = g + 0.0f;
        u = u + 0.0f;
        y[(uint64_t) row * args.n + n0 + lane] = g * audio_sigmoid(g) * u;
    }
}

template <typename W>
kernel void kernel_audio_matvec_silu(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const void *wg           [[buffer(3)]],
        device const void *wu           [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2u * AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec_silu_body(args, y, x, audio_runs<W>{wg, args.k}, audio_runs<W>{wu, args.k},
                           part, tgpig, lane, sg);
}

/// The gated pair over 8-bit affine weights: both weights' bytes, then their
/// scales, then their biases.
kernel void kernel_audio_matvec_silu_affine8w(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const uint *qg           [[buffer(3)]],
        device const uint *qu           [[buffer(4)]],
        device const ushort *sg_        [[buffer(5)]],
        device const ushort *su         [[buffer(6)]],
        device const ushort *bg         [[buffer(7)]],
        device const ushort *bu         [[buffer(8)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2u * AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec_silu_body(args, y, x, audio_runs_affine8{qg, sg_, bg, args.k},
                           audio_runs_affine8{qu, su, bu, args.k}, part, tgpig, lane, sg);
}

/// The one-row tile over a block of up to `AUDIO_MATVEC_ROWS` rows of `x`
/// (`nrows` of them, `k` apart) at once: every weight run is read and
/// converted once (`load8`) and dotted with each row (`apply8`), and each row's column sum is formed exactly as
/// `audio_matvec_tile` forms it on that row alone - the same runs per lane in
/// the same order, the same simdgroup sum, the two halves added in order -
/// so a batch of decoder rows costs one pass over the weights and keeps
/// every row's value. Lane `c < 4` of simdgroup 0 gets row `r`'s column `c`
/// in `total[r]`.
template <typename R>
static inline void audio_matvec_rows_tile(const R w, uint n, uint k, uint n0,
                                          device const float *x, uint nrows,
                                          threadgroup float *part, ushort lane, ushort sg,
                                          thread float *total) {
    const uint NR = AUDIO_MATVEC_COLS;
    const uint NSG = AUDIO_MATVEC_SG;
    const uint RB = AUDIO_MATVEC_ROWS;
    const uint runs = k >> 3u;
    typename R::row_t wr[AUDIO_MATVEC_COLS];
    for (uint c = 0; c < NR; ++c) {
        wr[c] = w.row(min(n0 + c, n - 1u));
    }
    float acc[AUDIO_MATVEC_ROWS][AUDIO_MATVEC_COLS];
    for (uint r = 0; r < RB; ++r) {
        for (uint c = 0; c < NR; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    for (uint j = sg * 32u + lane; j < runs; j += NSG * 32u) {
        audio_run8_vals v[AUDIO_MATVEC_COLS];
        for (uint c = 0; c < NR; ++c) {
            v[c] = w.load8(wr[c], j);
        }
        for (uint r = 0; r < RB; ++r) {
            if (r < nrows) {
                device const float4 *xp = (device const float4 *) (x + (uint64_t) r * k + j * 8u);
                const float4 x0 = xp[0];
                const float4 x1 = xp[1];
                for (uint c = 0; c < NR; ++c) {
                    acc[r][c] += w.apply8(v[c], x0, x1);
                }
            }
        }
    }
    for (uint r = 0; r < RB; ++r) {
        for (uint c = 0; c < NR; ++c) {
            acc[r][c] = simd_sum(acc[r][c]);
        }
    }
    if (lane == 0u) {
        for (uint r = 0; r < RB; ++r) {
            for (uint c = 0; c < NR; ++c) {
                part[(r * NR + c) * NSG + sg] = acc[r][c];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < RB; ++r) {
        float t = 0.0f;
        if (sg == 0u && lane < NR) {
            for (uint s = 0; s < NSG; ++s) {
                t += part[(r * NR + lane) * NSG + s];
            }
        }
        total[r] = t;
    }
}

/// `kernel_audio_matvec`'s product over rows `[8 y, 8 y + 8)` of the input
/// in one threadgroup: grid `(ceil(n / 4), ceil(m / 8))`.
template <typename R>
static inline void audio_matvec_rows_body(constant audio_args_gemm & args, device float *y,
                                          device const float *x, const R w,
                                          device const float *bias, threadgroup float *part,
                                          uint3 tgpig, ushort lane, ushort sg) {
    const uint n0 = tgpig.x * AUDIO_MATVEC_COLS;
    const uint row0 = tgpig.y * AUDIO_MATVEC_ROWS;
    const uint nrows = min(AUDIO_MATVEC_ROWS, args.m - row0);
    float total[AUDIO_MATVEC_ROWS];
    audio_matvec_rows_tile(w, args.n, args.k, n0, x + (uint64_t) row0 * args.k, nrows, part,
                           lane, sg, total);
    if (sg == 0u && lane < AUDIO_MATVEC_COLS && n0 + lane < args.n) {
        const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u;
        for (uint r = 0; r < nrows; ++r) {
            y[(uint64_t) (row0 + r) * args.n + n0 + lane] =
                    total[r] + (has_bias ? bias[n0 + lane] : 0.0f);
        }
    }
}

template <typename W>
kernel void kernel_audio_matvec_rows(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const void *w            [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[AUDIO_MATVEC_ROWS * AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec_rows_body(args, y, x, audio_runs<W>{w, args.k}, bias, part, tgpig, lane, sg);
}

kernel void kernel_audio_matvec_rows_affine8w(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const uint *q            [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        device const ushort *scales     [[buffer(5)]],
        device const ushort *biases     [[buffer(6)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[AUDIO_MATVEC_ROWS * AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec_rows_body(args, y, x, audio_runs_affine8{q, scales, biases, args.k}, bias, part,
                           tgpig, lane, sg);
}

/// `kernel_audio_matvec_silu`'s gated pair over rows `[8 y, 8 y + 8)` of the
/// input in one threadgroup, each product the multi-row tile's, the product
/// formed as the one-row kernel forms it.
template <typename R>
static inline void audio_matvec_silu_rows_body(constant audio_args_gemm & args, device float *y,
                                               device const float *x, const R wg, const R wu,
                                               threadgroup float *part, uint3 tgpig,
                                               ushort lane, ushort sg) {
    const uint n0 = tgpig.x * AUDIO_MATVEC_COLS;
    const uint row0 = tgpig.y * AUDIO_MATVEC_ROWS;
    const uint nrows = min(AUDIO_MATVEC_ROWS, args.m - row0);
    device const float *xr = x + (uint64_t) row0 * args.k;
    const uint span = AUDIO_MATVEC_ROWS * AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG;
    float g[AUDIO_MATVEC_ROWS];
    float u[AUDIO_MATVEC_ROWS];
    audio_matvec_rows_tile(wg, args.n, args.k, n0, xr, nrows, part, lane, sg, g);
    audio_matvec_rows_tile(wu, args.n, args.k, n0, xr, nrows, part + span, lane, sg, u);
    if (sg == 0u && lane < AUDIO_MATVEC_COLS && n0 + lane < args.n) {
        for (uint r = 0; r < nrows; ++r) {
            const float gr = g[r] + 0.0f;
            const float ur = u[r] + 0.0f;
            y[(uint64_t) (row0 + r) * args.n + n0 + lane] = gr * audio_sigmoid(gr) * ur;
        }
    }
}

template <typename W>
kernel void kernel_audio_matvec_silu_rows(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const void *wg           [[buffer(3)]],
        device const void *wu           [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2u * AUDIO_MATVEC_ROWS * AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec_silu_rows_body(args, y, x, audio_runs<W>{wg, args.k}, audio_runs<W>{wu, args.k},
                                part, tgpig, lane, sg);
}

kernel void kernel_audio_matvec_silu_rows_affine8w(
        constant audio_args_gemm & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const uint *qg           [[buffer(3)]],
        device const uint *qu           [[buffer(4)]],
        device const ushort *sg_        [[buffer(5)]],
        device const ushort *su         [[buffer(6)]],
        device const ushort *bg         [[buffer(7)]],
        device const ushort *bu         [[buffer(8)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2u * AUDIO_MATVEC_ROWS * AUDIO_MATVEC_COLS * AUDIO_MATVEC_SG];
    audio_matvec_silu_rows_body(args, y, x, audio_runs_affine8{qg, sg_, bg, args.k},
                                audio_runs_affine8{qu, su, bu, args.k}, part, tgpig, lane, sg);
}

typedef decltype(kernel_audio_matvec<audio_run8_f32>) audio_matvec_t;
template [[host_name("kernel_audio_matvec_f32w")]]  kernel audio_matvec_t kernel_audio_matvec<audio_run8_f32>;
template [[host_name("kernel_audio_matvec_bf16w")]] kernel audio_matvec_t kernel_audio_matvec<audio_run8_bf16>;
template [[host_name("kernel_audio_matvec_f16w")]]  kernel audio_matvec_t kernel_audio_matvec<audio_run8_f16>;
template [[host_name("kernel_audio_matvec_q8_0w")]] kernel audio_matvec_t kernel_audio_matvec<audio_run8_q8_0>;

typedef decltype(kernel_audio_matvec3<audio_run8_f32>) audio_matvec3_t;
template [[host_name("kernel_audio_matvec3_f32w")]]  kernel audio_matvec3_t kernel_audio_matvec3<audio_run8_f32>;
template [[host_name("kernel_audio_matvec3_bf16w")]] kernel audio_matvec3_t kernel_audio_matvec3<audio_run8_bf16>;
template [[host_name("kernel_audio_matvec3_f16w")]]  kernel audio_matvec3_t kernel_audio_matvec3<audio_run8_f16>;
template [[host_name("kernel_audio_matvec3_q8_0w")]] kernel audio_matvec3_t kernel_audio_matvec3<audio_run8_q8_0>;

typedef decltype(kernel_audio_matvec_silu<audio_run8_f32>) audio_matvec_silu_t;
template [[host_name("kernel_audio_matvec_silu_f32w")]]  kernel audio_matvec_silu_t kernel_audio_matvec_silu<audio_run8_f32>;
template [[host_name("kernel_audio_matvec_silu_bf16w")]] kernel audio_matvec_silu_t kernel_audio_matvec_silu<audio_run8_bf16>;
template [[host_name("kernel_audio_matvec_silu_f16w")]]  kernel audio_matvec_silu_t kernel_audio_matvec_silu<audio_run8_f16>;
template [[host_name("kernel_audio_matvec_silu_q8_0w")]] kernel audio_matvec_silu_t kernel_audio_matvec_silu<audio_run8_q8_0>;

typedef decltype(kernel_audio_matvec_rows<audio_run8_f32>) audio_matvec_rows_t;
template [[host_name("kernel_audio_matvec_rows_f32w")]]  kernel audio_matvec_rows_t kernel_audio_matvec_rows<audio_run8_f32>;
template [[host_name("kernel_audio_matvec_rows_bf16w")]] kernel audio_matvec_rows_t kernel_audio_matvec_rows<audio_run8_bf16>;
template [[host_name("kernel_audio_matvec_rows_f16w")]]  kernel audio_matvec_rows_t kernel_audio_matvec_rows<audio_run8_f16>;
template [[host_name("kernel_audio_matvec_rows_q8_0w")]] kernel audio_matvec_rows_t kernel_audio_matvec_rows<audio_run8_q8_0>;

typedef decltype(kernel_audio_matvec_silu_rows<audio_run8_f32>) audio_matvec_silu_rows_t;
template [[host_name("kernel_audio_matvec_silu_rows_f32w")]]  kernel audio_matvec_silu_rows_t kernel_audio_matvec_silu_rows<audio_run8_f32>;
template [[host_name("kernel_audio_matvec_silu_rows_bf16w")]] kernel audio_matvec_silu_rows_t kernel_audio_matvec_silu_rows<audio_run8_bf16>;
template [[host_name("kernel_audio_matvec_silu_rows_f16w")]]  kernel audio_matvec_silu_rows_t kernel_audio_matvec_silu_rows<audio_run8_f16>;
template [[host_name("kernel_audio_matvec_silu_rows_q8_0w")]] kernel audio_matvec_silu_rows_t kernel_audio_matvec_silu_rows<audio_run8_q8_0>;

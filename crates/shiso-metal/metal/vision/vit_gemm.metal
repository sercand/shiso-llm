// The one matrix multiply the vision tower needs.
//
// Every projection in the tower is the same shape of operation: f32 activation
// rows times a bf16 weight held transposed, accumulating in f32, with a bf16
// bias broadcast over the output columns. The patch projection, the fused qkv,
// the attention output, both feed-forward matrices and both merger layers all
// go through this entry point; there is no second GEMM to keep in step with it.
//
// Weights stay in the width the checkpoint stores them and are widened on read.
// The activations are f32 because that is what the residual stream this tower
// feeds is, end to end: nothing here narrows a value that later has to be
// added to an f32 hidden state.

/// `out[m, n] = sum_k in[m, k] * weight[n, k] + bias[n]`.
///
/// The weight is indexed as `[n][k]` -- output width slowest -- which is how
/// the checkpoint writes a linear layer, so no transpose pass runs anywhere in
/// this tower.
///
/// # Why a staged tile and not a thread per output
///
/// A thread that owned one output column would read `weight[n * k + ...]` with
/// neighbouring threads a whole row apart, which is the worst access this
/// device has. Staging a VIT_GEMM_TILE square of each operand makes both loads
/// contiguous along `k` and reuses each one across the tile.
///
/// # The summation order is the contract
///
/// `k` is walked strictly in increasing order: the tile loop advances by
/// VIT_GEMM_TILE and the inner loop walks the tile front to back, so the sum is
/// the same association a plain `for k in 0..K` loop makes. That is deliberate
/// -- the oracle this kernel is held to is exactly that loop -- and a blocking
/// change that reorders `k` is a change to the numbers, not just to the speed.
template <typename W>
static inline void vit_gemm_bias_body(
        constant vit_args_gemm & args,
        device float *out,
        device const float *in,
        device const ushort *weight,
        device const ushort *bias,
        threadgroup float (*a_tile)[VIT_GEMM_TILE],
        threadgroup float (*b_tile)[VIT_GEMM_TILE],
        uint3 tgpig,
        ushort3 tpitg) {
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;
    const uint row = tgpig.y * VIT_GEMM_TILE + ty;
    const uint col = tgpig.x * VIT_GEMM_TILE + tx;
    // The weight row this thread stages, which is not the one it accumulates:
    // the tile is filled row-per-thread-y so the `k` index runs along x and the
    // load stays contiguous, then read back column-wise in the inner loop.
    const uint stage_n = tgpig.x * VIT_GEMM_TILE + ty;

    float acc = 0.0f;
    for (uint k0 = 0u; k0 < args.k; k0 += VIT_GEMM_TILE) {
        const uint k = k0 + tx;
        const bool have_k = k < args.k;
        a_tile[ty][tx] = (row < args.m && have_k)
                ? in[(uint64_t) row * args.k + k]
                : 0.0f;
        b_tile[ty][tx] = (stage_n < args.n && have_k)
                ? W::load(weight + (uint64_t) stage_n * args.k + k)
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < VIT_GEMM_TILE; kk++) {
            acc += a_tile[ty][kk] * b_tile[tx][kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < args.m && col < args.n) {
        out[(uint64_t) row * args.n + col] = acc + W::load(bias + col);
    }
}

// The brain-float tower's entry and the half-precision tower's twin. The
// tiles are declared here because threadgroup memory belongs to a kernel.
kernel void kernel_vit_gemm_bias(
        constant vit_args_gemm & args [[buffer(0)]],
        device float *out             [[buffer(1)]],
        device const float *in        [[buffer(2)]],
        device const ushort *weight   [[buffer(3)]],
        device const ushort *bias     [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];
    threadgroup float b_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];
    vit_gemm_bias_body<vit_w_bf16>(args, out, in, weight, bias, a_tile, b_tile, tgpig, tpitg);
}

kernel void kernel_vit_gemm_bias_f16(
        constant vit_args_gemm & args [[buffer(0)]],
        device float *out             [[buffer(1)]],
        device const float *in        [[buffer(2)]],
        device const ushort *weight   [[buffer(3)]],
        device const ushort *bias     [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];
    threadgroup float b_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];
    vit_gemm_bias_body<vit_w_f16>(args, out, in, weight, bias, a_tile, b_tile, tgpig, tpitg);
}

/// The same product on simdgroup matrices: `out[m, n] = sum_k in[m, k] *
/// weight[n, k] + bias[n]`, for `k % 16 == 0` and 16-byte-aligned input rows.
///
/// A threadgroup of four simdgroups takes a 64x64 output tile, each simdgroup
/// a 32x32 quarter as a 4x4 grid of 8x8 f32 accumulators. k is staged 16 at a
/// time: the activations as `[row][k]`, the weights widened on the way in and
/// transposed to `[k][col]`, so every fragment load is a plain one. The next
/// slice's global reads are issued after the current slice's multiplies.
///
/// The operands stay f32 - the activations as they are, the weights widened
/// exactly - so every product is the one the blocked kernel above forms; what
/// changes is only where the partial sums round, eight k at a time in k order
/// inside the hardware's 8x8 step. Measured on an M4 over a 640x480 picture
/// (1200 patch rows, 27 blocks of width 1152): the blocked kernel ran the
/// four projections of a block in 93 ms, this tile in 12 ms (3.1 TFLOPS on
/// the fused qkv).
template <typename W>
static inline void vit_gemm64_body(
        constant vit_args_gemm & args,
        device float *out,
        device const float *in,
        device const ushort *weight,
        device const ushort *bias,
        threadgroup float *smem,
        uint3 tgpig,
        ushort tid,
        ushort lane,
        ushort sg) {
    const uint BM = VIT_GEMM64_TILE;
    const uint BN = VIT_GEMM64_TILE;
    const uint BK = VIT_GEMM64_K;
    const uint NT = VIT_GEMM64_THREADS;
    const uint LDA = BK + 4u;
    const uint LDB = BN + 4u;
    threadgroup float *sa = smem;            // [BM rows][BK k]
    threadgroup float *sb = smem + BM * LDA; // [BK k][BN cols]
    const uint n0 = tgpig.x * BN;
    const uint m0 = tgpig.y * BM;
    // Four simdgroups in a 2x2 grid, each owning 32 rows by 32 columns.
    const uint r0 = (sg >> 1u) * 32u;
    const uint c0 = (sg & 1u) * 32u;
    // Each thread's float4 slots of both operands: slot e of a 64x16 operand
    // is row e / 4, k (e % 4) * 4.
    const uint SLOTS = BM * BK / 4u / NT;
    float4 ra[SLOTS];
    float4 rb[SLOTS];
    // One slice's global reads; rows and columns past the edge read as zero.
    auto fetch = [&](uint k0) {
        VIT_UNROLL for (uint q = 0; q < SLOTS; ++q) {
            const uint e = tid + q * NT;
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            const uint gm = m0 + rr;
            ra[q] = gm < args.m ? *((device const float4 *) (in + (uint64_t) gm * args.k + k0 + kk))
                                : float4(0.0f);
            const uint gn = n0 + rr;
            rb[q] = gn < args.n ? W::load4(weight + (uint64_t) gn * args.k + k0 + kk)
                                : float4(0.0f);
        }
    };

    simdgroup_float8x8 acc[4][4];
    VIT_UNROLL for (uint i = 0; i < 4u; ++i) {
        VIT_UNROLL for (uint j = 0; j < 4u; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    fetch(0);
    for (uint k0 = 0; k0 < args.k; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        VIT_UNROLL for (uint q = 0; q < SLOTS; ++q) {
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
        VIT_UNROLL for (uint kk = 0; kk < BK; kk += 8u) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[4];
            VIT_UNROLL for (uint i = 0; i < 4u; ++i) {
                simdgroup_load(a[i], sa + (r0 + 8u * i) * LDA + kk, LDA);
            }
            VIT_UNROLL for (uint j = 0; j < 4u; ++j) {
                // b[j](r, c) = weight[c0 + 8j + c][k0 + kk + r].
                simdgroup_load(b[j], sb + kk * LDB + c0 + 8u * j, LDB);
            }
            VIT_UNROLL for (uint i = 0; i < 4u; ++i) {
                VIT_UNROLL for (uint j = 0; j < 4u; ++j) {
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
    // through its own 8x8 slot at the start of the tile, adding the bias.
    threadgroup float *slot = smem + sg * 64u;
    VIT_UNROLL for (uint i = 0; i < 4u; ++i) {
        VIT_UNROLL for (uint j = 0; j < 4u; ++j) {
            simdgroup_store(acc[i][j], slot, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                const uint gm = m0 + r0 + 8u * i + (e >> 3u);
                const uint gn = n0 + c0 + 8u * j + (e & 7u);
                if (gm < args.m && gn < args.n) {
                    out[(uint64_t) gm * args.n + gn] = slot[e] + W::load(bias + gn);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Both operands' staging: 64 x (16 + 4) activations and 16 x (64 + 4)
// weights, 9472 bytes.
#define VIT_GEMM64_SMEM_FLOATS (VIT_GEMM64_TILE * (VIT_GEMM64_K + 4u) + VIT_GEMM64_K * (VIT_GEMM64_TILE + 4u))

kernel void kernel_vit_gemm64_bias(
        constant vit_args_gemm & args [[buffer(0)]],
        device float *out             [[buffer(1)]],
        device const float *in        [[buffer(2)]],
        device const ushort *weight   [[buffer(3)]],
        device const ushort *bias     [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[VIT_GEMM64_SMEM_FLOATS];
    vit_gemm64_body<vit_w_bf16>(args, out, in, weight, bias, smem, tgpig, tid, lane, sg);
}

kernel void kernel_vit_gemm64_bias_f16(
        constant vit_args_gemm & args [[buffer(0)]],
        device float *out             [[buffer(1)]],
        device const float *in        [[buffer(2)]],
        device const ushort *weight   [[buffer(3)]],
        device const ushort *bias     [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[VIT_GEMM64_SMEM_FLOATS];
    vit_gemm64_body<vit_w_f16>(args, out, in, weight, bias, smem, tgpig, tid, lane, sg);
}

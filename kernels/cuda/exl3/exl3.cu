/* Host side of the vendored EXL3 kernels: kernel instances for the bit widths
 * this model uses, the tile-shape heuristic, the cooperative launch geometry,
 * and the per-stream lock/scratch buffers.  Torch-free port of the launch
 * logic in exllamav3's quant/exl3_gemm.cu and exl3_kernel_map.cu. */
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "exl3.h"
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_gemm_kernel.cuh"
#include "exl3_reconstruct.cuh"
#include "exl3_moe_kernel.cuh"
#include "exl3_moe_ws.cuh"

/* Only the mul1 codebook (cb 2) and fp32 outputs are instantiated: the
 * qwen4exp checkpoint is mul1 throughout, K = 4 for routed experts and the
 * indexer, 5 for the MTP input projections, 6 for every other dense tensor. */
#define SHS_EXL3_MIN_BITS 4
#define SHS_EXL3_MAX_BITS 6

static fp_exl3_gemm_kernel g_gemm[SHS_EXL3_MAX_BITS - SHS_EXL3_MIN_BITS + 1][EXL3_GEMM_NUM_SHAPES + 1] = {
    { EXL3_GEMM_KERNEL_INSTANCES(4, true, 2) },
    { EXL3_GEMM_KERNEL_INSTANCES(5, true, 2) },
    { EXL3_GEMM_KERNEL_INSTANCES(6, true, 2) },
};
static fp_exl3_mgemm_kernel g_mgemm[SHS_EXL3_MAX_BITS - SHS_EXL3_MIN_BITS + 1][EXL3_GEMM_NUM_SHAPES + 1] = {
    { EXL3_MGEMM_KERNEL_INSTANCES(4, true, 2) },
    { EXL3_MGEMM_KERNEL_INSTANCES(5, true, 2) },
    { EXL3_MGEMM_KERNEL_INSTANCES(6, true, 2) },
};
typedef void (*fp_exl3_reconstruct_kernel)(half *, const uint16_t *, const half *, const half *, int, int);
static fp_exl3_reconstruct_kernel g_reconstruct[SHS_EXL3_MAX_BITS - SHS_EXL3_MIN_BITS + 1] = {
    reconstruct_had_kernel<4, 2>, reconstruct_had_kernel<5, 2>, reconstruct_had_kernel<6, 2>,
};
/* The routed-expert block: the N = 128 tile, since the intermediate width
 * (640) is not a multiple of 256. */
static fp_exl3_moe_kernel g_moe[SHS_EXL3_MAX_BITS - SHS_EXL3_MIN_BITS + 1] = {
    exl3_moe_kernel<4, 128, 2>, exl3_moe_kernel<5, 128, 2>, exl3_moe_kernel<6, 128, 2>,
};
#define SHS_EXL3_MOE_BLOCK_DIM (EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
static const int g_tile_k[] = { EXL3_GEMM_TILESIZE_K };
static const int g_tile_n[] = { EXL3_GEMM_TILESIZE_N };
static const int g_block_dim[] = { EXL3_GEMM_BLOCKDIM };

/* Per-stream state.  The lock buffer is shared by every launch on a stream
 * (the kernels leave it zeroed), so two streams must not share one; the
 * scratch holds the fp16 Hadamard-transformed activations. */
#define SHS_EXL3_MAX_STREAMS 8
struct exl3_stream_ctx {
    cudaStream_t stream;
    int         *locks;
    half        *had;
    uint64_t     had_halfs;
    uint8_t     *moe;          /* fused MoE staging (prefill only, never captured) */
    uint64_t     moe_bytes;
    uint8_t     *ws;           /* weight-stationary MoE staging (prefill only) */
    uint64_t     ws_bytes;
};
static exl3_stream_ctx g_ctx[SHS_EXL3_MAX_STREAMS];
static int g_n_ctx = 0;
static int g_num_sms = 0;
static int g_max_blocks = 0;   /* co-resident blocks: one per SM at 90 KB of shared memory */

static bool exl3_fail(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return false;
    fprintf(stderr, "shiso: exl3 %s: %s\n", what, cudaGetErrorString(err));
    (void)cudaGetLastError();
    return true;
}

static bool exl3_device_init(void) {
    if (g_num_sms) return true;
    int dev = 0;
    if (exl3_fail(cudaGetDevice(&dev), "device query")) return false;
    if (exl3_fail(cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev), "SM count")) return false;
    /* Every kernel instance needs the 90 KB dynamic shared memory opt-in
     * before its first launch; a capture would reject the attribute call. */
    int occ = 0;
    for (int b = 0; b <= SHS_EXL3_MAX_BITS - SHS_EXL3_MIN_BITS; b++) {
        for (int s = 1; s <= EXL3_GEMM_NUM_SHAPES; s++) {
            if (exl3_fail(cudaFuncSetAttribute((const void *)g_gemm[b][s],
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX),
                          "gemm shared memory opt-in")) return false;
            if (exl3_fail(cudaFuncSetAttribute((const void *)g_mgemm[b][s],
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX),
                          "mgemm shared memory opt-in")) return false;
            int o = 0;
            if (exl3_fail(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                  &o, g_gemm[b][s], g_block_dim[s], SMEM_MAX), "occupancy")) return false;
            occ = (occ == 0 || o < occ) ? o : occ;
        }
    }
    for (int b = 0; b <= SHS_EXL3_MAX_BITS - SHS_EXL3_MIN_BITS; b++) {
        if (exl3_fail(cudaFuncSetAttribute((const void *)g_moe[b],
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX),
                      "moe shared memory opt-in")) return false;
        int o = 0;
        if (exl3_fail(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                              &o, g_moe[b], SHS_EXL3_MOE_BLOCK_DIM, SMEM_MAX), "moe occupancy")) return false;
        occ = (occ == 0 || o < occ) ? o : occ;
    }
    if (occ < 1) occ = 1;
    g_max_blocks = g_num_sms * occ;
    return true;
}

static exl3_stream_ctx *exl3_ctx(cudaStream_t stream, uint64_t had_halfs) {
    if (!exl3_device_init()) return NULL;
    exl3_stream_ctx *c = NULL;
    for (int i = 0; i < g_n_ctx; i++) if (g_ctx[i].stream == stream) c = &g_ctx[i];
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(stream, &cap);
    const bool capturing = cap != cudaStreamCaptureStatusNone;
    if (!c) {
        if (g_n_ctx == SHS_EXL3_MAX_STREAMS || capturing) {
            fprintf(stderr, "shiso: exl3: no lock buffer for this stream (%s)\n",
                    capturing ? "cannot allocate during graph capture" : "too many streams");
            return NULL;
        }
        c = &g_ctx[g_n_ctx];
        memset(c, 0, sizeof(*c));
        c->stream = stream;
        const size_t lock_bytes = (MAX_TILES_C + 2 * MAX_BARRIERS + MOE_SCHED_INTS) * sizeof(int);
        if (exl3_fail(cudaMalloc((void **)&c->locks, lock_bytes), "lock buffer") ||
            exl3_fail(cudaMemset(c->locks, 0, lock_bytes), "lock buffer clear")) return NULL;
        g_n_ctx++;
    }
    if (c->had_halfs < had_halfs) {
        if (capturing) {
            fprintf(stderr, "shiso: exl3: activation scratch too small during graph capture "
                            "(%llu < %llu halfs); reserve it first\n",
                    (unsigned long long)c->had_halfs, (unsigned long long)had_halfs);
            return NULL;
        }
        /* The old scratch is kept, not freed: a decode graph captured earlier
         * on this stream still names it, and would replay into freed memory.
         * A few tens of MB over a session; the buffers are small. */
        uint64_t want = had_halfs < (8u << 20) ? (8u << 20) : had_halfs;   /* 16 MiB floor */
        half *had = NULL;
        if (exl3_fail(cudaMalloc((void **)&had, want * sizeof(half)), "activation scratch")) return NULL;
        c->had = had;
        c->had_halfs = want;
    }
    return c;
}

extern "C" int shs_exl3_reserve(cudaStream_t stream, uint64_t had_halfs) {
    return exl3_ctx(stream, had_halfs) != NULL;
}

extern "C" int shs_exl3_prepare_stream(cudaStream_t stream) {
    if (g_n_ctx == 0) return 1;
    uint64_t halfs = 0;
    for (int i = 0; i < g_n_ctx; i++) if (g_ctx[i].had_halfs > halfs) halfs = g_ctx[i].had_halfs;
    return exl3_ctx(stream, halfs) != NULL;
}

/* exllamav3's shape heuristic for Blackwell (select_gemm_shape, CC_BLACKWELL
 * branch), with the shape's tile constraints checked. */
static int exl3_select_shape(int size_m, int size_k, int size_n, int K, bool multi,
                             int bszm_in, int bszm_out) {
    (void)size_m;
    const bool mod_256 = (size_n % 256 == 0);
    const bool mod_512 = (size_n % 512 == 0);
    const int k_eff = size_k * bszm_in;
    const int n_eff = size_n * bszm_out;
    int shape;
    if ((K == 4 || K == 2) && !multi && k_eff <= 2048) shape = 1;
    else if (K >= 7) {
        if (mod_256 && n_eff <= 8192) shape = k_eff > 32768 ? 3 : 2;
        else if (mod_512 && n_eff > 32768) shape = 4;
        else shape = 2;
    }
    else if (mod_256 && n_eff <= 4096) shape = (k_eff > 8192 && K >= 3) ? 3 : 2;
    else if (mod_512 && n_eff > 16384) shape = 4;
    else if (mod_256) shape = 3;
    else shape = 2;
    for (int s = shape; s >= 1; s--) {
        if (size_k % g_tile_k[s] == 0 && size_n % g_tile_n[s] == 0) return s;
    }
    for (int s = 1; s <= EXL3_GEMM_NUM_SHAPES; s++) {
        if (size_k % g_tile_k[s] == 0 && size_n % g_tile_n[s] == 0) return s;
    }
    return 0;
}

static bool exl3_check_shape(uint32_t k, uint32_t n, uint32_t bits, const char *what) {
    if (bits < SHS_EXL3_MIN_BITS || bits > SHS_EXL3_MAX_BITS || k == 0 || n == 0 ||
        (k % 16u) != 0u || (n % 128u) != 0u) {
        fprintf(stderr, "shiso: exl3 %s: unsupported shape k=%u n=%u K=%u\n", what, k, n, bits);
        return false;
    }
    return true;
}

extern "C" int shs_exl3_gemm(const float *x, const void *tiles, const void *suh, const void *svh,
                             float *y, uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                             cudaStream_t stream) {
    if (!x || !tiles || !suh || !svh || !y || m == 0) return 0;
    if (!exl3_check_shape(k, n, bits, "gemm")) return 0;
    exl3_stream_ctx *c = exl3_ctx(stream, (uint64_t)m * k);
    if (!c) return 0;

    const int shape = exl3_select_shape((int)m, (int)k, (int)n, (int)bits, false, 1, 1);
    if (!shape) return 0;
    fp_exl3_gemm_kernel kernel = g_gemm[bits - SHS_EXL3_MIN_BITS][shape];
    /* Blocks sharing an output tile column hand their partial sums down a
     * lock chain, so past about four blocks per column the chain costs more
     * than the extra streams earn: the indexer's 2560 x 128 K = 4 tensor ran
     * three times slower on 48 blocks than on 8, the 2560 x 640 shared
     * expert 15% slower than on 20.  Measured on GB10 (48 SMs, one block
     * each); the wide tensors still take every SM. */
    const int tiles_k = (int)(k / g_tile_k[shape]), tiles_n = (int)(n / g_tile_n[shape]);
    int num_blocks = 4 * tiles_n;
    if (num_blocks < 8) num_blocks = 8;
    if (num_blocks > g_max_blocks) num_blocks = g_max_blocks;
    if (num_blocks > tiles_k * tiles_n) num_blocks = tiles_k * tiles_n;

    const half *suh_h = (const half *)suh;
    const half *svh_h = (const half *)svh;
    const uint16_t *B = (const uint16_t *)tiles;
    half *had = c->had;
    int *locks = c->locks;
    int size_m = (int)m, size_k = (int)k, size_n = (int)n;
    void *args[] = { (void *)&x, (void *)&B, (void *)&y, (void *)&size_m, (void *)&size_k,
                     (void *)&size_n, (void *)&locks, (void *)&suh_h, (void *)&had, (void *)&svh_h };
    return !exl3_fail(cudaLaunchCooperativeKernel((const void *)kernel, dim3(num_blocks), dim3(g_block_dim[shape]),
                                                  args, SMEM_MAX, stream), "gemm launch");
}

/* `n_slots` slots over the first tensor, then the same slots again over the
 * second when it is given (tiles2 != NULL): 2 * n_slots launch slots. */
static int exl3_mgemm_launch(const float *x, int x_per_slot,
                             const void *tiles, const void *suh, const void *svh, float *y,
                             const void *tiles2, const void *suh2, const void *svh2, float *y2,
                             const int32_t *ids, uint32_t n_slots,
                             uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                             cudaStream_t stream) {
    if (!x || !tiles || !suh || !svh || !y || m == 0 || n_slots == 0) return 0;
    if (tiles2 && (!suh2 || !svh2 || !y2)) return 0;
    if (!exl3_check_shape(k, n, bits, "mgemm")) return 0;
    const uint32_t total = tiles2 ? 2u * n_slots : n_slots;
    /* One transformed slab per slot: an undersized scratch is silent
     * out-of-bounds corruption (exllamav3's comment: "found the hard way"). */
    exl3_stream_ctx *c = exl3_ctx(stream, (uint64_t)total * m * k);
    if (!c) return 0;

    const int bszm_in = x_per_slot ? (int)total : 1;
    const int bszm_out = (int)total;
    const int shape = exl3_select_shape((int)m, (int)k, (int)n, (int)bits, true, bszm_in, bszm_out);
    if (!shape) return 0;
    fp_exl3_mgemm_kernel kernel = g_mgemm[bits - SHS_EXL3_MIN_BITS][shape];

    /* Geometry: every slot gets its own block group when the co-resident
     * budget allows, so a decode launch streams all its experts at once
     * instead of in rounds.  On GB10 (48 blocks) the ten routed experts of
     * a token ran 23% faster on 10 groups of 4 than on exllamav3's 6 groups
     * of 8, and the two shared-expert projections as 2 groups of 24 take
     * half the time of two dense launches. */
    const int n_tiles = (int)(k / g_tile_k[shape]) * (int)(n / g_tile_n[shape]);
    int concurrency = (int)total;
    if (concurrency > g_max_blocks) concurrency = g_max_blocks;
    if (concurrency > MAX_BARRIERS) concurrency = MAX_BARRIERS;
    int per_group = g_max_blocks / concurrency;
    if (per_group > n_tiles) per_group = n_tiles;
    if (per_group < 1) per_group = 1;

    const size_t stride = (size_t)k * n * bits / 8u / sizeof(uint16_t);
    const uint16_t *B = (const uint16_t *)tiles, *B2 = (const uint16_t *)tiles2;
    const half *suh_h = (const half *)suh, *suh2_h = (const half *)suh2;
    const half *svh_h = (const half *)svh, *svh2_h = (const half *)svh2;
    half *had = c->had;
    int *locks = c->locks;
    int size_m = (int)m, size_k = (int)k, size_n = (int)n, split = (int)n_slots;
    void *args[] = { (void *)&x, (void *)&B, (void *)&stride, (void *)&y, (void *)&size_m,
                     (void *)&size_k, (void *)&size_n, (void *)&locks, (void *)&suh_h, (void *)&had,
                     (void *)&svh_h, (void *)&ids, (void *)&bszm_in, (void *)&bszm_out,
                     (void *)&B2, (void *)&suh2_h, (void *)&svh2_h, (void *)&y2, (void *)&split };
    return !exl3_fail(cudaLaunchCooperativeKernel((const void *)kernel, dim3(per_group, 1, concurrency),
                                                  dim3(g_block_dim[shape]), args, SMEM_MAX, stream),
                      "mgemm launch");
}

extern "C" int shs_exl3_mgemm(const float *x, int x_per_slot,
                              const void *tiles, const void *suh, const void *svh,
                              const int32_t *ids, uint32_t n_slots,
                              float *y, uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                              cudaStream_t stream) {
    return exl3_mgemm_launch(x, x_per_slot, tiles, suh, svh, y, NULL, NULL, NULL, NULL,
                             ids, n_slots, m, k, n, bits, stream);
}

extern "C" int shs_exl3_mgemm_pair(const float *x, int x_per_slot,
                                   const void *tiles, const void *suh, const void *svh, float *y,
                                   const void *tiles2, const void *suh2, const void *svh2, float *y2,
                                   const int32_t *ids, uint32_t n_slots,
                                   uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                                   cudaStream_t stream) {
    if (!tiles2) return 0;
    return exl3_mgemm_launch(x, x_per_slot, tiles, suh, svh, y, tiles2, suh2, svh2, y2,
                             ids, n_slots, m, k, n, bits, stream);
}

extern "C" int shs_exl3_reconstruct(const void *tiles, const void *suh, const void *svh,
                                    uint32_t k, uint32_t n, uint32_t bits, void *w_out,
                                    cudaStream_t stream) {
    if (!tiles || !suh || !svh || !w_out) return 0;
    if (!exl3_check_shape(k, n, bits, "reconstruct") || (k % 128u) != 0u) {
        fprintf(stderr, "shiso: exl3 reconstruct: k=%u is not a multiple of 128\n", k);
        return 0;
    }
    /* One block per 128x128 tile; the tile row index is n/16 packed blocks
     * long, the same layout the GEMMs read. */
    g_reconstruct[bits - SHS_EXL3_MIN_BITS]<<<dim3(n / 128u, k / 128u), RH_THREADS, 0, stream>>>(
            (half *)w_out, (const uint16_t *)tiles, (const half *)suh, (const half *)svh, (int)(n / 16u), 0);
    return !exl3_fail(cudaGetLastError(), "reconstruct launch");
}

/* Geometry as exllamav3's exl3_moe: groups of MOE_SMS_PER_EXPERT blocks, one
 * expert at a time each, every block of the grid co-resident for the group
 * barriers.  With 256 experts and ten per token nearly every expert has rows
 * in a real chunk, so the group count is never the limit. */
extern "C" int shs_exl3_moe(const float *x, float *slot_out, uint32_t n_tok, uint32_t hidden, uint32_t inter,
                            const void *gate_tiles, const void *gate_suh, const void *gate_svh,
                            const void *up_tiles, const void *up_suh, const void *up_svh,
                            const void *down_tiles, const void *down_suh, const void *down_svh,
                            uint32_t bits, const int32_t *expert_bounds, const int32_t *slot_sorted,
                            uint32_t n_expert, uint32_t n_used, cudaStream_t stream) {
    if (!x || !slot_out || !gate_tiles || !gate_suh || !gate_svh || !up_tiles || !up_suh || !up_svh ||
        !down_tiles || !down_suh || !down_svh || !expert_bounds || !slot_sorted ||
        n_tok == 0 || n_expert == 0 || n_used == 0) return 0;
    if (!exl3_check_shape(hidden, inter, bits, "moe") || (hidden % 128u) != 0u) {
        fprintf(stderr, "shiso: exl3 moe: hidden=%u is not a multiple of 128\n", hidden);
        return 0;
    }
    exl3_stream_ctx *c = exl3_ctx(stream, 0);
    if (!c) return 0;

    int num_groups = g_num_sms / MOE_SMS_PER_EXPERT;
    if (num_groups > MOE_MAX_GROUPS) num_groups = MOE_MAX_GROUPS;
    if (num_groups > (int)n_expert) num_groups = (int)n_expert;
    if (num_groups < 1) num_groups = 1;
    int group_size = g_num_sms / num_groups;
    if (group_size > MOE_MAX_SMS_PER_EXPERT) group_size = MOE_MAX_SMS_PER_EXPERT;
    if (group_size * num_groups > g_max_blocks) group_size = g_max_blocks / num_groups;
    if (group_size < 1) group_size = 1;

    /* Per group and token: gathered gate and up inputs (fp16, later the fp32
     * down output), gate and up outputs (fp32), the down input (fp16). */
    const uint64_t row_bytes = 2u * hidden * sizeof(half) + 2u * inter * sizeof(float) + inter * sizeof(half);
    const uint64_t moe_bytes = (uint64_t)num_groups * n_tok * row_bytes;
    if (c->moe_bytes < moe_bytes) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) {
            fprintf(stderr, "shiso: exl3 moe: cannot allocate staging during graph capture\n");
            return 0;
        }
        if (exl3_fail(cudaStreamSynchronize(stream), "moe staging drain")) return 0;
        if (c->moe) (void)cudaFree(c->moe);
        c->moe = NULL;
        c->moe_bytes = 0;
        if (exl3_fail(cudaMalloc((void **)&c->moe, moe_bytes), "moe staging")) return 0;
        c->moe_bytes = moe_bytes;
    }
    half *state = (half *)c->moe;
    float *inter_gu = (float *)(state + (uint64_t)num_groups * n_tok * 2u * hidden);
    half *act = (half *)(inter_gu + (uint64_t)num_groups * n_tok * 2u * inter);

    const uint16_t *gt = (const uint16_t *)gate_tiles, *ut = (const uint16_t *)up_tiles,
                   *dt = (const uint16_t *)down_tiles;
    const half *gs = (const half *)gate_suh, *gv = (const half *)gate_svh,
               *us = (const half *)up_suh, *uv = (const half *)up_svh,
               *ds = (const half *)down_suh, *dv = (const half *)down_svh;
    int *locks = c->locks;
    int i_tok = (int)n_tok, i_hidden = (int)hidden, i_inter = (int)inter,
        i_expert = (int)n_expert, i_used = (int)n_used;
    void *args[] = { (void *)&x, (void *)&state, (void *)&inter_gu, (void *)&act, (void *)&slot_out, (void *)&gt, (void *)&gs, (void *)&gv, (void *)&ut, (void *)&us,
                     (void *)&uv, (void *)&dt, (void *)&ds, (void *)&dv, (void *)&expert_bounds,
                     (void *)&slot_sorted, (void *)&i_tok, (void *)&i_hidden,
                     (void *)&i_inter, (void *)&i_expert, (void *)&i_used, (void *)&locks };
    return !exl3_fail(cudaLaunchCooperativeKernel((const void *)g_moe[bits - SHS_EXL3_MIN_BITS],
                                                  dim3(group_size, 1, num_groups),
                                                  dim3(SHS_EXL3_MOE_BLOCK_DIM), args, SMEM_MAX, stream),
                      "moe launch");
}

/* Weight-stationary routed block; see exl3_moe_ws.cuh.  The staging holds the
 * two fp16 gate/up inputs and the fp16 down input for every assignment row. */
#define SHS_WS_STAGES 3
template <int bits>
static bool exl3_moe_ws_optin(void) {
    static bool done = false;
    if (done) return true;
    if (exl3_fail(cudaFuncSetAttribute((const void *)exl3_moe_ws_gateup<bits, 2, SHS_WS_STAGES>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       WsSmem<bits, true, SHS_WS_STAGES>::total), "moe ws gateup smem") ||
        exl3_fail(cudaFuncSetAttribute((const void *)exl3_moe_ws_down<bits, 2, SHS_WS_STAGES>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       WsSmem<bits, false, SHS_WS_STAGES>::total), "moe ws down smem"))
        return false;
    done = true;
    return true;
}

template <int bits>
static int exl3_moe_ws_launch(const float *x, float *slot_out, uint32_t n_tok, uint32_t hidden, uint32_t inter,
                              const uint16_t *gt, const half *gs, const half *gv,
                              const uint16_t *ut, const half *us, const half *uv,
                              const uint16_t *dt, const half *ds, const half *dv,
                              const int32_t *bounds, const int32_t *slot_sorted,
                              uint32_t n_expert, uint32_t n_used, cudaStream_t stream, exl3_stream_ctx *c) {
    if (!exl3_moe_ws_optin<bits>()) return 0;
    const uint32_t n_rows = n_tok * n_used;
    const uint64_t need = (uint64_t)n_rows * (2u * hidden + inter) * sizeof(half);
    if (c->ws_bytes < need) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) {
            fprintf(stderr, "shiso: exl3 moe ws: cannot allocate staging during graph capture\n");
            return 0;
        }
        if (exl3_fail(cudaStreamSynchronize(stream), "moe ws staging drain")) return 0;
        if (c->ws) (void)cudaFree(c->ws);
        c->ws = NULL;
        c->ws_bytes = 0;
        if (exl3_fail(cudaMalloc((void **)&c->ws, need), "moe ws staging")) return 0;
        c->ws_bytes = need;
    }
    half *a_g = (half *)c->ws;
    half *a_u = a_g + (size_t)n_rows * hidden;
    half *act = a_u + (size_t)n_rows * hidden;

    const int prep_tasks = (int)n_rows * (int)(hidden / 128u);
    exl3_moe_ws_prep<<<(prep_tasks + WS_THREADS / 32 - 1) / (WS_THREADS / 32), WS_THREADS, 0, stream>>>(
            x, a_g, a_u, gs, us, bounds, slot_sorted, (int)n_rows, (int)n_expert, (int)n_used, (int)hidden);
    if (exl3_fail(cudaGetLastError(), "moe ws prep")) return 0;
    /* One block per (expert, tile): a second dimension would share an
     * expert's passes between blocks, which measured slower (see the header). */
    const dim3 ggrid(n_expert * (inter / 128u), 1);
    exl3_moe_ws_gateup<bits, 2, SHS_WS_STAGES><<<ggrid, WS_THREADS, WsSmem<bits, true, SHS_WS_STAGES>::total, stream>>>(
            a_g, a_u, act, gt, gv, ut, uv, ds, bounds, (int)hidden, (int)inter);
    if (exl3_fail(cudaGetLastError(), "moe ws gate/up")) return 0;
    const dim3 dgrid(n_expert * (hidden / 128u), 1);
    exl3_moe_ws_down<bits, 2, SHS_WS_STAGES><<<dgrid, WS_THREADS, WsSmem<bits, false, SHS_WS_STAGES>::total, stream>>>(
            act, slot_out, dt, dv, bounds, slot_sorted, (int)hidden, (int)inter);
    return !exl3_fail(cudaGetLastError(), "moe ws down");
}

extern "C" int shs_exl3_moe_ws(const float *x, float *slot_out, uint32_t n_tok, uint32_t hidden, uint32_t inter,
                               const void *gate_tiles, const void *gate_suh, const void *gate_svh,
                               const void *up_tiles, const void *up_suh, const void *up_svh,
                               const void *down_tiles, const void *down_suh, const void *down_svh,
                               uint32_t bits, const int32_t *expert_bounds, const int32_t *slot_sorted,
                               uint32_t n_expert, uint32_t n_used, cudaStream_t stream) {
    if (!x || !slot_out || !gate_tiles || !gate_suh || !gate_svh || !up_tiles || !up_suh || !up_svh ||
        !down_tiles || !down_suh || !down_svh || !expert_bounds || !slot_sorted ||
        n_tok == 0 || n_expert == 0 || n_used == 0) return 0;
    /* K in 64-wide stages, N in 128-wide tiles, on both projections' shapes. */
    if (!exl3_check_shape(hidden, inter, bits, "moe ws") || (hidden % 128u) != 0u || (inter % 64u) != 0u) {
        fprintf(stderr, "shiso: exl3 moe ws: unsupported shape hidden=%u inter=%u\n", hidden, inter);
        return 0;
    }
    exl3_stream_ctx *c = exl3_ctx(stream, 0);
    if (!c) return 0;
    const uint16_t *gt = (const uint16_t *)gate_tiles, *ut = (const uint16_t *)up_tiles,
                   *dt = (const uint16_t *)down_tiles;
    const half *gs = (const half *)gate_suh, *gv = (const half *)gate_svh,
               *us = (const half *)up_suh, *uv = (const half *)up_svh,
               *ds = (const half *)down_suh, *dv = (const half *)down_svh;
    switch (bits) {
    case 4: return exl3_moe_ws_launch<4>(x, slot_out, n_tok, hidden, inter, gt, gs, gv, ut, us, uv, dt, ds, dv,
                                         expert_bounds, slot_sorted, n_expert, n_used, stream, c);
    case 5: return exl3_moe_ws_launch<5>(x, slot_out, n_tok, hidden, inter, gt, gs, gv, ut, us, uv, dt, ds, dv,
                                         expert_bounds, slot_sorted, n_expert, n_used, stream, c);
    case 6: return exl3_moe_ws_launch<6>(x, slot_out, n_tok, hidden, inter, gt, gs, gv, ut, us, uv, dt, ds, dv,
                                         expert_bounds, slot_sorted, n_expert, n_used, stream, c);
    default: return 0;
    }
}

/* The half-output down projection has no host launcher here - the engine
 * launches it by name - so its instances are made explicitly. */
template __global__ void exl3_moe_ws_down_h<4, 2, SHS_WS_STAGES>(
        const half *, half *, const uint16_t *, const half *, const int32_t *, const int32_t *, int, int);
template __global__ void exl3_moe_ws_down_h<5, 2, SHS_WS_STAGES>(
        const half *, half *, const uint16_t *, const half *, const int32_t *, const int32_t *, int, int);
template __global__ void exl3_moe_ws_down_h<6, 2, SHS_WS_STAGES>(
        const half *, half *, const uint16_t *, const half *, const int32_t *, const int32_t *, int, int);

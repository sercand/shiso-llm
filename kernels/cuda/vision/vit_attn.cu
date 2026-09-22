/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the
 * kernels, which were `static` in a header written to be included into one
 * translation unit and now have to be reachable from a module registry).  No
 * line of kernel arithmetic was edited, reordered or reformatted; a change to
 * any of it is a defect, not a cleanup.
 *
 * The attention family: the fused projection split with the two-dimensional
 * rotation, and the bidirectional softmax that consumes its three planes. They
 * share the head geometry and the plane layout, which is the seam this unit is
 * cut along.
 */

#include "vision.cuh"

/* Split the fused [q | k | v] projection into three planes and rotate q and k.
 *
 * The ViT's RoPE is 2D: the head's 72 dims split in half NeoX-style, pairing
 * channel i with i+36; the first 18 pairs rotate by the patch's ROW index and
 * the last 18 by its COLUMN index, both at base 1e4.  So the whole head
 * rotates -- "partial_rotary_factor 0.5" in the reference describes how the 36
 * cos/sin values are built before being used on both axes, not a partial
 * rotation. */
extern "C" __global__ void qwen3vl_qkv_rope_kernel(
        float *q, float *k, float *v, const float *qkv, const uint16_t *bias,
        uint32_t rows, uint32_t grid_w) {
    const uint32_t n = blockIdx.y;
    const uint32_t h = blockIdx.x;
    if (n >= rows) return;
    const uint32_t dim = SHS_QWEN3VL_VISION_DIM;
    const uint32_t hd = SHS_QWEN3VL_VISION_HEAD_DIM;
    const uint32_t half = hd / 2u;                 /* 36 */
    uint32_t row, col;
    qwen3vl_patch_rc(n, grid_w, &row, &col);

    const float *src = qkv + (size_t)n * 3u * dim;
    const uint32_t off = h * hd;
    float *qd = q + (size_t)n * dim + off;
    float *kd = k + (size_t)n * dim + off;
    float *vd = v + (size_t)n * dim + off;

    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) {
        qd[i] = src[off + i] + qwen3vl_bf16(bias + off + i);
        kd[i] = src[dim + off + i] + qwen3vl_bf16(bias + dim + off + i);
        vd[i] = src[2u * dim + off + i] + qwen3vl_bf16(bias + 2u * dim + off + i);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
        /* Pair i uses frequency index i % 18; the row axis owns the first 18
         * pairs and the column axis the second 18. */
        const uint32_t fi = (i < half / 2u) ? i : (i - half / 2u);
        const uint32_t pos = (i < half / 2u) ? row : col;
        const float inv = powf(SHS_QWEN3VL_VISION_ROPE_BASE,
                               -(float)(2u * fi) / (float)SHS_QWEN3VL_VISION_ROT);
        float s, c;
        sincosf((float)pos * inv, &s, &c);
        const float q0 = qd[i], q1 = qd[i + half];
        qd[i] = q0 * c - q1 * s;
        qd[i + half] = q1 * c + q0 * s;
        const float k0 = kd[i], k1 = kd[i + half];
        kd[i] = k0 * c - k1 * s;
        kd[i + half] = k1 * c + k0 * s;
    }
}

/* Full bidirectional attention over the whole grid, online softmax.
 *
 * One block per (row, head); warps own whole key positions and reduce their
 * dot product with shuffles.  head_dim is 72, which is not a multiple of the
 * warp width, so a lane strides by 32 and the low 8 lanes carry a third
 * element -- correct for any head_dim, just slightly ragged for this one.
 * Attention never crosses an image, and grid_t is 1 for a still image, so the
 * key range is simply every row. */
extern "C" __global__ void qwen3vl_attention_kernel(
        float *out, const float *q, const float *k, const float *v,
        uint32_t rows) {
    const uint32_t n = blockIdx.y;
    const uint32_t h = blockIdx.x;
    if (n >= rows) return;
    const uint32_t dim = SHS_QWEN3VL_VISION_DIM;
    const uint32_t hd = SHS_QWEN3VL_VISION_HEAD_DIM;
    const uint32_t off = h * hd;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t warps = blockDim.x >> 5;
    const float scale = rsqrtf((float)hd);

    extern __shared__ float qwen3vl_attn_smem[];
    float *s_q = qwen3vl_attn_smem;                  /* hd */
    float *s_acc = s_q + hd;                         /* warps * hd */
    float *s_max = s_acc + (size_t)warps * hd;       /* warps */
    float *s_den = s_max + warps;                    /* warps */

    const float *qd = q + (size_t)n * dim + off;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) s_q[i] = qd[i];
    __syncthreads();

    float run_max = -INFINITY, run_den = 0.0f;
    float acc[3] = {0.0f, 0.0f, 0.0f};   /* ceil(72/32) slots per lane */
    for (uint32_t p = warp; p < rows; p += warps) {
        const float *kd = k + (size_t)p * dim + off;
        float dot = 0.0f;
        for (uint32_t d = lane; d < hd; d += 32u) dot += s_q[d] * kd[d];
        for (uint32_t m = 16u; m; m >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, m);
        const float score = dot * scale;
        const float new_max = fmaxf(run_max, score);
        const float rescale = __expf(run_max - new_max);
        const float weight = __expf(score - new_max);
        run_den = run_den * rescale + weight;
        run_max = new_max;
        const float *vd = v + (size_t)p * dim + off;
        uint32_t slot = 0;
        for (uint32_t d = lane; d < hd; d += 32u, slot++) {
            acc[slot] = acc[slot] * rescale + weight * vd[d];
        }
    }
    /* Publish each warp's partial softmax, then combine them. */
    {
        uint32_t slot = 0;
        for (uint32_t d = lane; d < hd; d += 32u, slot++) {
            s_acc[(size_t)warp * hd + d] = acc[slot];
        }
        if (lane == 0) { s_max[warp] = run_max; s_den[warp] = run_den; }
    }
    __syncthreads();
    if (threadIdx.x < hd) {
        float m = -INFINITY;
        for (uint32_t w = 0; w < warps; w++) m = fmaxf(m, s_max[w]);
        float den = 0.0f, num = 0.0f;
        for (uint32_t w = 0; w < warps; w++) {
            const float r = __expf(s_max[w] - m);
            den += s_den[w] * r;
            num += s_acc[(size_t)w * hd + threadIdx.x] * r;
        }
        out[(size_t)n * dim + off + threadIdx.x] = den > 0.0f ? num / den : 0.0f;
    }
}

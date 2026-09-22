/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the
 * kernels, which were `static` in a header written to be included into one
 * translation unit and now have to be reachable from a module registry).  No
 * line of kernel arithmetic was edited, reordered or reformatted; a change to
 * any of it is a defect, not a cleanup.
 *
 * The learned position embedding, on its own.  It is the only kernel in the
 * tower that reads the position table, the only one that resamples anything,
 * and the only one whose arithmetic depends on the table being a square grid.
 */

#include "vision.cuh"

/* Add the learned position embedding, bilinearly resampled from its fixed
 * 48x48 grid onto this image's grid.
 *
 * The reference interpolates with linspace(0, 47, n) -- align_corners style --
 * and a single point when the extent is 1, then applies the same merge-block
 * reorder as the patch sequence, which is why the (row, col) for output row n
 * comes from qwen3vl_patch_rc. */
extern "C" __global__ void qwen3vl_pos_embed_kernel(
        float *x, const uint16_t *table, uint32_t rows,
        uint32_t grid_h, uint32_t grid_w) {
    const uint32_t n = blockIdx.x;
    if (n >= rows) return;
    uint32_t row, col;
    qwen3vl_patch_rc(n, grid_w, &row, &col);

    const uint32_t side = SHS_QWEN3VL_VISION_POS_SIDE;
    const float hs = (grid_h > 1u) ? (float)(side - 1u) / (float)(grid_h - 1u) : 0.0f;
    const float ws = (grid_w > 1u) ? (float)(side - 1u) / (float)(grid_w - 1u) : 0.0f;
    const float hy = (float)row * hs, wx = (float)col * ws;
    const uint32_t h0 = (uint32_t)hy, w0 = (uint32_t)wx;
    const uint32_t h1 = (h0 + 1u < side) ? h0 + 1u : side - 1u;
    const uint32_t w1 = (w0 + 1u < side) ? w0 + 1u : side - 1u;
    const float dh = hy - (float)h0, dw = wx - (float)w0;
    const float c11 = dh * dw;
    const float c10 = dh - c11;          /* dh * (1 - dw) */
    const float c01 = dw - c11;          /* (1 - dh) * dw */
    const float c00 = 1.0f - dh - c01;   /* (1 - dh) * (1 - dw) */

    const uint32_t dim = SHS_QWEN3VL_VISION_DIM;
    const uint16_t *e00 = table + (size_t)(h0 * side + w0) * dim;
    const uint16_t *e01 = table + (size_t)(h0 * side + w1) * dim;
    const uint16_t *e10 = table + (size_t)(h1 * side + w0) * dim;
    const uint16_t *e11 = table + (size_t)(h1 * side + w1) * dim;
    float *dst = x + (size_t)n * dim;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] += c00 * qwen3vl_bf16(e00 + i) + c01 * qwen3vl_bf16(e01 + i) +
                  c10 * qwen3vl_bf16(e10 + i) + c11 * qwen3vl_bf16(e11 + i);
    }
}

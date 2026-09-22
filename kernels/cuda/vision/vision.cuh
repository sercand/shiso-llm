/* Cross-family device helpers and geometry shared by every qwen3vl vision
 * translation unit.
 *
 *
 * The weight-pointer typedefs that sit between the geometry and the helpers
 * upstream are host descriptors for the encode driver and are named by no
 * kernel, so they are not carried here: this tree holds device code only.
 *
 * q4e.cuh supplies q4e_block_sum, which the layer normalization calls.  That is
 * how the source reached it too -- the vision header is included after the
 * qwen4exp one in a single translation unit upstream -- so the dependency is
 * the source's, not the split's.
 */

#ifndef SHISO_QWEN3VL_VISION_CUH
#define SHISO_QWEN3VL_VISION_CUH

#include "q4e.cuh"

#define SHS_QWEN3VL_VISION_LAYERS    27u
#define SHS_QWEN3VL_VISION_DIM       1152u
#define SHS_QWEN3VL_VISION_HEADS     16u
#define SHS_QWEN3VL_VISION_HEAD_DIM  72u    /* 1152 / 16 */
#define SHS_QWEN3VL_VISION_ROT       36u    /* head_dim * partial_rotary 0.5 */
#define SHS_QWEN3VL_VISION_FF        4304u
#define SHS_QWEN3VL_VISION_OUT       2560u  /* == the LLM's hidden size */
#define SHS_QWEN3VL_VISION_PATCH_DIM 1536u  /* 3 * 2 * 16 * 16 */
#define SHS_QWEN3VL_VISION_MERGED    4608u  /* 1152 * spatial_merge^2 */
#define SHS_QWEN3VL_VISION_POS_SIDE  48u    /* sqrt(num_position_embeddings) */
#define SHS_QWEN3VL_VISION_EPS       1e-6f
#define SHS_QWEN3VL_VISION_ROPE_BASE 10000.0f

#ifndef SHS_QWEN3VL_VISION_STREAM
#define SHS_QWEN3VL_VISION_STREAM 0
#endif

__device__ __forceinline__ static float qwen3vl_bf16(const uint16_t *p) {
    return __uint_as_float((uint32_t)(*p) << 16);
}

/* Patch n of the flattened sequence sits at grid (row, col).
 *
 * The sequence is in merge-block order: blocks row-major over the
 * (grid_h/2, grid_w/2) block grid, and within a block its 2x2 patches
 * row-major.  shs_image_preprocess_qwen3vl emits patches in exactly this
 * order, the ViT's 2D RoPE indexes positions with it, and it is what makes the
 * merger's "four consecutive rows are one 2x2 block" reshape correct.  All
 * three have to agree, so the mapping lives in one place. */
__device__ __forceinline__ static void qwen3vl_patch_rc(
        uint32_t n, uint32_t grid_w, uint32_t *row, uint32_t *col) {
    const uint32_t bw = grid_w / 2u;
    const uint32_t block = n / 4u, local = n & 3u;
    *row = (block / bw) * 2u + (local >> 1);
    *col = (block % bw) * 2u + (local & 1u);
}

#endif /* SHISO_QWEN3VL_VISION_CUH */

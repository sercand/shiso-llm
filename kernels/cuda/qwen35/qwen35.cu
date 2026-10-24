/* The `qwen35` module: the entries the engine asks for under that name.
 *
 * The module name is not a choice. `crates/shiso-engine/src/projection.rs`
 * dispatches the prefill GEMM as `KernelRef::specialized(Q35, ...)`, and on this
 * backend a module name is the key of a separately compiled image rather than a
 * label - so an entry the engine asks for under `qwen35` has to be in an image
 * called `qwen35`, whatever else that name promises. The unit used to live in
 * `cuda/generic`, because the two share a q4_0 dequantizer and an
 * argument-block header; it has its own directory now, because the family's own
 * kernels are arriving here and a Gemma 4 pass has no business reaching into a
 * directory named for another family. `KERNEL.toml` still includes
 * `cuda/generic`, for the three templates below and nothing else.
 *
 * What this module carries, in two groups. The Qwen 3.5 dense hybrid's own
 * kernels - the rotary and norm pair with its two multimodal siblings, the
 * gated attention family, the gated delta network - are in the `.cuh` pieces
 * this unit includes, each beside a note on which law in `cuda/q4e` it is the
 * twin of. And the entries that belong to no family, which the engine asks for
 * under this name on behalf of whatever container is loaded:
 *
 *   - the three prefill GEMMs, because `launch::mm_name` hands every codec's
 *     tiled product to this module whatever family it belongs to;
 *   - the q6_K vocabulary gather, because `crates/shiso-engine/src/layers.rs`
 *     maps `DType::Q6_K` here - and a Gemma 4 GGUF needs it, since the QAT
 *     export packs the vocabulary table at q6_K whatever its linear layers are
 *     packed at;
 *   - the q6_K extended matvec, because `launch::ext_module` maps that codec
 *     here too, and the output head is the vocabulary table tied back;
 *   - the q5_K extended matvec, because `ext_module` maps that codec here as
 *     well and a `Q4_K_M` qwen35 container keeps 48 tensors at it.
 */

#include "mul_mm.cuh"
#include "dequant_plane.cuh"
#include "mul_mv_ext.cuh"
#include "get_rows_q6_K.cuh"

/* The family's own entries. One `.cuh` per heading of
 * `kernels/metal/qwen35/KERNELS.md`. */
#include "q35_norm_rope.cuh"
#include "q35_attn.cuh"
#include "q35_gdn.cuh"

/* The tiled products. The f16 one is the per-layer model projection, which is
 * the only unquantized weight in either Gemma 4 container; the q4_K one is the
 * bulk of every qwen35 linear layer; the q8_0 one is that family's two gate
 * projections, `ssm_alpha` and `ssm_beta`, which a `Q4_K_M` container keeps at
 * that codec - 48 tensors, one pair per recurrent layer. Every codec
 * `launch::mm_name` answers for has an entry here, because a name that table
 * hands out and this module does not export is a prefill that refuses at its
 * first chunk. */
SHISO_MM(q4_0, struct shiso_block_q4_0)
SHISO_MM(f16, __half)
SHISO_MM(q4_K, struct shiso_block_q4_K)
SHISO_MM(q8_0, struct shiso_block_q8_0)
/* About half of a `Q4_K_M` export's down projections. */
SHISO_MM(q6_K, struct shiso_block_q6_K)

/* The staging half of the vendor-GEMM route, one entry per codec above.
 *
 * Deliberately the same five and in the same order: `launch::mm_name` and
 * `launch::dequant_plane_name` are read together by
 * `projection::mm_projection`, which asks the vendor library first and falls
 * back to the tile, so a codec with a tile and no stager would silently never
 * take the faster path and one with a stager and no tile would have no
 * fallback at all. */
SHISO_DEQUANT_PLANE(q4_0, struct shiso_block_q4_0)
SHISO_DEQUANT_PLANE(f16, struct shiso_block_f16)
SHISO_DEQUANT_PLANE(q4_K, struct shiso_block_q4_K)
SHISO_DEQUANT_PLANE(q8_0, struct shiso_block_q8_0)
SHISO_DEQUANT_PLANE(q6_K, struct shiso_block_q6_K)

/* Fifteen pairs of the q6_K matvec, for the same reason the generic module
 * carries fifteen of the q4_0 one: `ext_r1ptg` and `ext_nxpsg` pick the row
 * blocking and the lane split independently, so every pair is reachable and a
 * pair that is merely absent is invisible at any single prompt length. */
SHISO_MV_EXT_Q6_K(1, 4)
SHISO_MV_EXT_Q6_K(1, 8)
SHISO_MV_EXT_Q6_K(1, 16)
SHISO_MV_EXT_Q6_K(2, 4)
SHISO_MV_EXT_Q6_K(2, 8)
SHISO_MV_EXT_Q6_K(2, 16)
SHISO_MV_EXT_Q6_K(3, 4)
SHISO_MV_EXT_Q6_K(3, 8)
SHISO_MV_EXT_Q6_K(3, 16)
SHISO_MV_EXT_Q6_K(4, 4)
SHISO_MV_EXT_Q6_K(4, 8)
SHISO_MV_EXT_Q6_K(4, 16)
SHISO_MV_EXT_Q6_K(5, 4)
SHISO_MV_EXT_Q6_K(5, 8)
SHISO_MV_EXT_Q6_K(5, 16)

/* Fifteen pairs of the q5_K matvec, for the same reason: the qwen35 container
 * keeps 48 tensors at q5_K, and `launch::ext_module` sends that codec here. */
SHISO_MV_EXT_Q5_K(1, 4)
SHISO_MV_EXT_Q5_K(1, 8)
SHISO_MV_EXT_Q5_K(1, 16)
SHISO_MV_EXT_Q5_K(2, 4)
SHISO_MV_EXT_Q5_K(2, 8)
SHISO_MV_EXT_Q5_K(2, 16)
SHISO_MV_EXT_Q5_K(3, 4)
SHISO_MV_EXT_Q5_K(3, 8)
SHISO_MV_EXT_Q5_K(3, 16)
SHISO_MV_EXT_Q5_K(4, 4)
SHISO_MV_EXT_Q5_K(4, 8)
SHISO_MV_EXT_Q5_K(4, 16)
SHISO_MV_EXT_Q5_K(5, 4)
SHISO_MV_EXT_Q5_K(5, 8)
SHISO_MV_EXT_Q5_K(5, 16)

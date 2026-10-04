/* The `qwen35moe` module: the two codecs a `Qwen3_5MoeForConditionalGeneration`
 * checkpoint stores its weights in, and the routed feed-forward that reads
 * them.
 *
 * ONE translation unit, because on this backend a module name is the key of a
 * separately compiled image rather than a label, and
 * `crates/shiso-kernels/build_plan.rs` refuses two directories claiming one
 * module name. Everything the engine dispatches under this family's module
 * name has to land in an image of that name.
 *
 * This directory exists rather than more entries in `cuda/qwen35` because the
 * two families share their recurrent and attention KERNELS and nothing else:
 * `crates/shiso-models/src/qwen35moe/` dispatches this family's blocks through
 * `Qwen35Kernels`, and what it cannot get there is a codec the dense container
 * has never carried.
 *
 * Nothing here is a byte copy, so no file carries a provenance header and
 * `scripts/q4e_ptx_oracle.sh --mode=fidelity` reports them as generated. The
 * arithmetic is `crates/shiso-modelfile/src/fp8.rs` and
 * `crates/shiso-modelfile/src/nvfp4.rs`, both of which are in this tree and both of
 * which are graded bit-for-bit against torch 2.14.
 *
 * `experts_kblock.cuh` is the exception to "nothing here is a byte copy" only
 * in the sense that it copies nothing either: it INSTANTIATES `cuda/generic`'s
 * codec templates rather than respelling them, so the `ggml` super-block
 * arithmetic this family reads is the same arithmetic the plain matvec and the
 * prefill GEMM read, graded by the same oracle.
 */

#include "q35moe_common.cuh"
#include "mv_fp8.cuh"
#include "mv_nvfp4.cuh"
#include "mv_multi.cuh"
#include "experts.cuh"
#include "experts_kblock.cuh"
#include "experts_affine.cuh"

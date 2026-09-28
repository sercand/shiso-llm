/* The `dense` module: Gemma 4's parametric attention set and its helpers.
 *
 * ONE translation unit, because a module name is the registry key for a
 * separately compiled image and the driver resolves entry names inside one
 * image only. The engine dispatches this whole set under `launch::DENSE`
 * ("dense"), so every entry it names has to land here; the pieces below are
 * split by family for reading, not for linking.
 *
 * Nothing in this directory is a byte copy of anything, so no file here carries
 * a provenance header and `scripts/q4e_ptx_oracle.sh --mode=fidelity` reports
 * it as generated. The arithmetic is the Metal twins' in
 * `crates/shiso-metal/metal/dense`, which is what produced the committed
 * goldens, and the argument blocks are the ones
 * `crates/shiso-engine/src/launch.rs` packs.
 *
 * The entry names are the host's. They are `&'static str` literals at the call
 * sites in `crates/shiso-engine/src/dense`, so the spelling here is not ours to
 * choose and every entry carries C linkage for it.
 */

#include "dense_misc.cuh"
#include "dense_norm_rope.cuh"
#include "dense_attn.cuh"
#include "dense_ple.cuh"
#include "dense_affine.cuh"
#include "dense_exact_affine.cuh"
#include "dense_block_draft.cuh"
#include "dense_vision.cuh"

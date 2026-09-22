/* The `vision` module: the parametric vision-tower set both towers dispatch.
 *
 * ONE translation unit, because a module name is the registry key for a
 * separately compiled image and the driver resolves entry names inside one image
 * only. `crates/shiso-models/src/qwen35/vision/encoder.rs`,
 * `crates/shiso-models/src/gemma4/vision/encoder.rs` and
 * `crates/shiso-engine/src/vision/mod.rs` all dispatch under the module name
 * `vision`, so every entry they name has to land here; the pieces below are
 * split by family for reading, not for linking.
 *
 * Nothing in this unit is a byte copy of anything, so it carries no provenance
 * header and `scripts/q4e_ptx_oracle.sh --mode=fidelity` reports it as
 * generated. The arithmetic is the Metal twins' in
 * `crates/shiso-metal/metal/vision`, and the argument blocks are the ones the
 * two encoders and the engine pack.
 *
 * This is NOT the `qwen3vl_*` set in the same directory. That one is a verbatim
 * extraction of a different decomposition of the same tower, held byte-for-byte
 * by the fidelity gate, dispatched by nothing, and refuted entry by entry in
 * `KERNEL_NAMES.md`.
 *
 * The entry names are the host's. They are `&'static str` literals at the call
 * sites in `shiso-models` and `shiso-engine`, so the spelling here is not ours
 * to choose and every entry carries C linkage for it.
 */

#include "vit_linear.cuh"
#include "vit_tower.cuh"
#include "vit_seam.cuh"

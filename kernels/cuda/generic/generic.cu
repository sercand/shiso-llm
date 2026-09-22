/* The `generic` module: the q4_0 extended matvec at every lane split and row
 * blocking the engine can select, the plain f16 matvec a decode falls back
 * to, the plain q8_0 matvec that is the only decode path that codec has, the
 * gated SiLU a qwen35 feed-forward closes on, the two fused RMS norms a dense
 * slot passes through, the residual add, and the q4_K extended matvec every
 * qwen35 linear layer is read through at one token.
 *
 * One translation unit per module, because a module name is the registry key
 * for a separately compiled image and the driver resolves entry names inside
 * one image only. The thirty-six entries below are the whole of this module.
 *
 * Nothing in this directory is a byte copy of anything, so no file here carries
 * a provenance header and `scripts/q4e_ptx_oracle.sh --mode=fidelity` reports
 * them as generated. The arithmetic is derived from the Metal twins in
 * `crates/shiso-metal/metal/generic` and from the argument blocks in
 * `crates/shiso-engine/src/launch.rs`, both of which are in this tree.
 */

#include "mul_mv_ext.cuh"
#include "mul_mv_dense.cuh"
#include "mul_mv_q8_0.cuh"
#include "glu.cuh"
#include "norm.cuh"
#include "bin.cuh"
#include "get_rows_f16.cuh"
#include "get_rows_block.cuh"

/* Fifteen pairs: five row blockings by three lane splits. `ext_r1ptg` and
 * `ext_nxpsg` choose the two independently, so every pair is reachable and
 * every pair is here. */
SHISO_MV_EXT_Q4_0(1, 4)
SHISO_MV_EXT_Q4_0(1, 8)
SHISO_MV_EXT_Q4_0(1, 16)
SHISO_MV_EXT_Q4_0(2, 4)
SHISO_MV_EXT_Q4_0(2, 8)
SHISO_MV_EXT_Q4_0(2, 16)
SHISO_MV_EXT_Q4_0(3, 4)
SHISO_MV_EXT_Q4_0(3, 8)
SHISO_MV_EXT_Q4_0(3, 16)
SHISO_MV_EXT_Q4_0(4, 4)
SHISO_MV_EXT_Q4_0(4, 8)
SHISO_MV_EXT_Q4_0(4, 16)
SHISO_MV_EXT_Q4_0(5, 4)
SHISO_MV_EXT_Q4_0(5, 8)
SHISO_MV_EXT_Q4_0(5, 16)

/* Fifteen more pairs of the same template at the codec the bulk of a qwen35
 * `Q4_K_M` container is packed at. They are HERE rather than in `cuda/qwen35`
 * because `launch::ext_module` maps `Q4_K` to `launch::GENERIC` - the host
 * names the module and this backend answers where it is asked, whatever family
 * the weight belongs to. Without them `ext_name_for` returned `None` on a
 * backend that specializes by name, the projection fell through to
 * `plain_projection`, and a decode refused with "`Q4_K` has no plain matvec at
 * this width" - which named the family that could not serve it rather than the
 * one that should have. */
SHISO_MV_EXT_Q4_K(1, 4)
SHISO_MV_EXT_Q4_K(1, 8)
SHISO_MV_EXT_Q4_K(1, 16)
SHISO_MV_EXT_Q4_K(2, 4)
SHISO_MV_EXT_Q4_K(2, 8)
SHISO_MV_EXT_Q4_K(2, 16)
SHISO_MV_EXT_Q4_K(3, 4)
SHISO_MV_EXT_Q4_K(3, 8)
SHISO_MV_EXT_Q4_K(3, 16)
SHISO_MV_EXT_Q4_K(4, 4)
SHISO_MV_EXT_Q4_K(4, 8)
SHISO_MV_EXT_Q4_K(4, 16)
SHISO_MV_EXT_Q4_K(5, 4)
SHISO_MV_EXT_Q4_K(5, 8)
SHISO_MV_EXT_Q4_K(5, 16)

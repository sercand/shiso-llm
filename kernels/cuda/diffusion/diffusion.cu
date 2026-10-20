/* The `diffusion` module: every device operation the image-generation drivers
 * (`crates/shiso-models/src/diffusion/ops.rs`, `ops/vae.rs`) launch.
 *
 * One translation unit per module, because a module name is the registry key
 * of a separately compiled image and `Ops::new` resolves every entry of
 * `REQUIRED_KERNELS` inside the one image named `diffusion`. The pieces below
 * are include-only (KERNEL.toml lists only this file). The entries are the
 * Metal twin's (`crates/shiso-metal/metal/diffusion/diffusion.metal`), under
 * the same names; `KERNEL_NAMES.md` is the table, and the host-only test
 * `required_kernels_all_present_on_cuda` checks it against `REQUIRED_KERNELS`.
 */

#include "common.cuh"
#include "elementwise.cu"
#include "kquant.cuh"
#include "gemm.cu"
#include "norm.cu"
#include "attention.cu"
#include "rope.cu"
#include "conv.cu"

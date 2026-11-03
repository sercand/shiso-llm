/* The `ternary` module: one translation unit, because a module name is the key
 * of a separately compiled image and the engine dispatches this whole set under
 * `launch::TERNARY` ("ternary"). The arithmetic is the Metal twins' in
 * `kernels/metal/ternary`; every entry carries C linkage because the
 * names are the host's `&'static str` literals. */

#include "rotate.cuh"
#include "ternary_mv.cuh"
#include "ternary_mm.cuh"

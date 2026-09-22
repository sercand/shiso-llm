/* GENERATED mirror of the paged-KV geometry.
 *
 * Rust side: `shiso_core::page` (crates/shiso-core/src/page.rs, constants
 * PAGE_TOKENS / PAGE_SHIFT / PAGE_MASK / MROPE_BACK).  The layout is
 * transcribed from the reference implementation VENDOR provenance names.
 *
 * The macros carry this project's `SHS_` prefix, as every kernel here does.
 * The copied kernels that `#include` this file spell them the same way, and
 * `scripts/q4e_ptx_oracle.sh --mode=fidelity` maps the prefix back to the
 * upstream spelling before it diffs, so the byte-fidelity proof is unaffected.
 *
 * The four values must stay byte-equal to the Rust constants: a host page table
 * and a device kv_row() that disagreed would hand a kernel the bytes of another
 * token and nothing would report it.  Regenerate from the Rust side rather than
 * editing here.
 */
#ifndef SHISO_Q4E_PAGE_CUH
#define SHISO_Q4E_PAGE_CUH

#define SHS_Q4E_PAGE_TOKENS 256u
#define SHS_Q4E_PAGE_SHIFT  8u
#define SHS_Q4E_PAGE_MASK   (SHS_Q4E_PAGE_TOKENS - 1u)

/* Rows the mRoPE table carries in FRONT of a chunk's first token: the pooled
 * key kernel derives its position by rounding DOWN to the compression
 * boundary, so it can name a position up to r-1 slots before the chunk starts
 * and those rows have to exist.  4 covers r = 4 with one row to spare. */
#define SHS_Q4E_MROPE_BACK  4u

#endif /* SHISO_Q4E_PAGE_CUH */

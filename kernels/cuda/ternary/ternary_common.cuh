/* The ternary set's argument blocks, mirrored from
 * `crates/shiso-metal/metal/ternary/ternary_common.h` and packed by
 * `crates/shiso-engine/src/launch.rs`. */
#ifndef SHISO_TERNARY_COMMON_CUH
#define SHISO_TERNARY_COMMON_CUH

#include <stdint.h>

/* `kernel_rotate_wht_f32`. `rep > 1` reads `nk * rep` heads of `hd` in tiled
 * order (head `m * nk + k`) as grouped (head `k * rep + m`). */
struct ternary_args_rotate {
    uint32_t n_tok;
    uint32_t width;
    uint32_t block;
    uint32_t inverse;
    uint32_t nk;
    uint32_t rep;
    uint32_t hd;
    uint32_t pad;
};
static_assert(sizeof(struct ternary_args_rotate) == 32, "RotateArgs is 32 bytes");

/* `kernel_ternary_mv_*`: `y[t][o] = sum_k W[o][k] x[t][k]`; a block is
 * `64 * splits` threads over `splits` interleaved contraction slices. */
struct ternary_args_mv {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tok;
    uint32_t splits;
};
static_assert(sizeof(struct ternary_args_mv) == 16, "MvArgs is 16 bytes");

/* `kernel_ternary_get_rows_*`. */
struct ternary_args_rows {
    uint32_t in_dim;
    uint32_t n_tok;
};
static_assert(sizeof(struct ternary_args_rows) == 8, "RowsArgs is 8 bytes");

#endif

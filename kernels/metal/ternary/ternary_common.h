// The ternary kernel set's argument blocks.
//
// A rotated checkpoint stores each matrix with a blockwise Walsh-Hadamard
// rotation folded into its input axis, and its weights as ternary codes under
// one f16 scale per 128 inputs. This set is the two halves of reading one: the
// rotation the activations have to pass through first, and the matmuls over
// the codes. Every entry is f32 in, f32 out.


/// `kernel_rotate_wht_f32`.
///
/// `rep > 1` reads the input through a head permutation: the activation
/// arrives with `nk * rep` heads of `hd` in tiled order (head `m * nk + k`)
/// and is rotated as if it were in grouped order (head `k * rep + m`).
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

/// `kernel_ternary_mv_*` and `kernel_ternary_mm_*`: `y[t][o] = sum_k W[o][k] x[t][k]`.
///
/// `splits` is the number of interleaved slices of the contraction one
/// threadgroup carries (a threadgroup is `64 * splits` threads); the host picks
/// it from the weight's shape alone.
struct ternary_args_mv {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tok;
    uint32_t splits;
};

/// `kernel_ternary_get_rows_*`: one weight row per token id, decoded to f32.
struct ternary_args_rows {
    uint32_t in_dim;
    uint32_t n_tok;
};

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

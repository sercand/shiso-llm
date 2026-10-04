# ternary entry-point names

The lookup table a loader resolves a kernel through. There is one module,
`ternary`, and one translation unit in it (`ternary.cu`), for the same reason as
`kernels/cuda/dense/KERNEL_NAMES.md` states: the engine dispatches the whole set
under `launch::TERNARY`.

Written from the sources, not yet pasted from a `--mode=ptx` listing: this set
was authored on a machine without nvcc. Re-paste from
`bash scripts/q4e_ptx_oracle.sh --mode=ptx` on the first CUDA build.

## Module `ternary`

Argument order is the Metal twins' buffer order (`ternary_common.h`).

```
kernel_rotate_wht_f32
```

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
kernel_ternary_mv_pq2_0_f32_t1
kernel_ternary_mv_mlx2_f32_t1
kernel_ternary_mv_ptq1_0_f32_t1
kernel_ternary_get_rows_pq2_0_f32
kernel_ternary_get_rows_mlx2_f32
kernel_ternary_get_rows_ptq1_0_f32
kernel_ternary_mm_pq2_0_f32_n8
kernel_ternary_mm_pq2_0_f32_n32
kernel_ternary_mmv_pq2_0_f32
kernel_ternary_mm_mlx2_f32_n8
kernel_ternary_mm_mlx2_f32_n32
kernel_ternary_mmv_mlx2_f32
kernel_ternary_mm_ptq1_0_f32_n8
kernel_ternary_mm_ptq1_0_f32_n32
kernel_ternary_mmv_ptq1_0_f32
```

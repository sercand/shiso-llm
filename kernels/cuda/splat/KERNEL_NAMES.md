# splat entry-point names

The lookup table a loader resolves a kernel through. The module is loaded by
its registry key (`splat`); a kernel inside it is fetched by the exact name
recorded here. Every entry carries C linkage and is spelled out in full in the
source, so each name below is the name the driver answers to, verbatim. These
are the names `REQUIRED_KERNELS` in `crates/shiso-models/src/idol/splat_ops.rs`
lists and the names the Metal twin (`kernels/metal/splat/splat.metal`)
registers.

**Status: written on a machine with no CUDA toolchain; these names are read
from the source, not pasted from compiled PTX.** The first CUDA build should
replace this with the output of `grep -o '\.visible \.entry [^(]*' splat.ptx`.

## Module `splat` - 9 entries

```
kernel_splat_attn_f32
kernel_splat_deform_im2col_f32
kernel_splat_gelu_erf_f32
kernel_splat_merge2x2_f32
kernel_splat_relu_f32
kernel_splat_resize_bilinear_f32
kernel_splat_rotary_tables_f32
kernel_splat_sigmoid_f32
kernel_splat_window_attn_f32
```

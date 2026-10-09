//! The family-neutral diffusion toolkit shared by every image family.
//!
//! What is true of running a `DiT` image pipeline on a device and of nothing
//! family-specific lives here: the kernel driver ([`ops`]) with its GEMMs
//! (BF16, FP8 with a per-tensor scale, F32, plain and fused), flash attention,
//! implicit-GEMM convolution, the norms and the scratch pool; the safetensors
//! store and linear layer ([`weights`], [`lin`]) that bind a checkpoint's
//! stored dtype to a GEMM entry point without dequantizing it; and the
//! flow-match Euler sigma ladder ([`euler`]). A family keeps its own config,
//! tensor names, positional encoding and time-shift fit, and composes these.
//!
//! New weight dtypes extend [`ops::WeightKind`] and [`lin::Lin::load`]; the
//! device kernels they launch are registered in
//! `crates/shiso-metal/metal/diffusion/`.

pub(crate) mod euler;
pub(crate) mod lin;
pub(crate) mod ops;
pub(crate) mod weights;

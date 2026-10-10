//! The family-neutral diffusion toolkit shared by every image family.
//!
//! What is true of running a `DiT` image pipeline on a device and of nothing
//! family-specific lives here: the kernel driver ([`ops`]) with its GEMMs
//! (BF16, FP8 with a per-tensor scale, F32, per-row INT8 with its `ConvRot`
//! input rotation ([`quant`]), plain and fused), flash attention,
//! implicit-GEMM convolution, the norms and the scratch pool; the safetensors
//! store and linear layer ([`weights`], [`lin`]) that bind a checkpoint's
//! stored dtype to a GEMM entry point without dequantizing it; and the
//! flow-match Euler sigma ladder ([`euler`]) and the seeded starting noise
//! ([`noise`]); the PNG/JPEG encoder ([`encode`]) takes
//! interleaved RGB or RGBA bytes; the Qwen3-style prefill text encoder
//! ([`qwen3_text`]) and the injection-safe prompt tokenizer ([`prompt`]) the
//! families condition through; and the residency planner and page-cache
//! prefetch ([`residency`]) that let a small host swap components. A family keeps its own config,
//! tensor names, positional encoding and time-shift fit, and composes these.
//!
//! New weight dtypes extend [`ops::WeightKind`] and [`lin::Lin::load`]; the
//! device kernels they launch are registered in
//! `crates/shiso-metal/metal/diffusion/`.

pub(crate) mod encode;
pub(crate) mod euler;
pub(crate) mod lin;
pub(crate) mod noise;
pub(crate) mod ops;
pub(crate) mod prompt;
pub(crate) mod quant;
pub(crate) mod qwen3_text;
pub(crate) mod residency;
pub(crate) mod weights;

#[cfg(test)]
mod quant_tests;
#[cfg(test)]
mod residency_tests;

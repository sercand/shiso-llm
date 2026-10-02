//! FLUX.2 rectified-flow image generation: config, scheduler, transformer, VAE.
//!
//! The family reads its architecture from the checkpoint's own tensors -
//! widths from linear shapes, depth by counting block names, precision from
//! tensor dtypes - and runs every linear in its stored format: BF16 rows
//! through the BF16 GEMM, `F8_E4M3` rows through the FP8 GEMM with the
//! tensor's `weight_scale` in the epilogue, F32 rows through the F32 GEMM.
//! Nothing is dequantized into a duplicate buffer.
//!
//! [`config`] locates and inspects a checkpoint, [`pipeline`] drives one
//! generation, and [`scheduler`], [`transformer`], [`text_encoder`] and
//! [`vae`] are the parts it composes. [`ops`] is the family's device-kernel
//! surface with a host twin for every entry, so builds without the flux2
//! Metal kernels still run (slowly) on any backend.

pub mod config;
pub(crate) mod kv_cache;
pub(crate) mod ops;
pub(crate) mod pipeline;
pub(crate) mod rope;
pub(crate) mod scheduler;
pub(crate) mod text_encoder;
pub(crate) mod transformer;
pub(crate) mod vae;
pub(crate) mod weights;

use crate::catalog::{FamilyFacts, FamilyFeatures};

/// Family facts for Flux2 image generation.
pub const FACTS: FamilyFacts = FamilyFacts {
    name: "flux2",
    accepted_roles: &["image"],
    tasks: &["image_generation"],
    features: FamilyFeatures {
        tools: false,
        reasoning_efforts: &[],
        default_reasoning_effort: None,
        structured_output: false,
    },
};

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod scheduler_tests;

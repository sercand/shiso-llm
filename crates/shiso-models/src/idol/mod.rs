//! The 3D splat family: preprocessing, then the encoders and decoders.
//!
//! Plan 1 of the family lands the host-side preprocessing stage alone: the
//! square crop, the erosion and the composite that turn a photograph into the
//! 1024-square RGB image the encoders read. The device stages arrive with
//! their own plans, which is why this module has no backend in it yet.

pub mod conditioner;
pub mod conditioning;
pub mod config;
pub(crate) mod decoders;
pub(crate) mod denoiser;
pub mod dinov3;
pub mod draws;
pub mod gaussian;
pub mod generator;
pub mod latent;
pub mod model;
pub(crate) mod nn;
pub mod octree;
pub mod ply;
pub mod preprocess;
pub mod sampler;

#[cfg(test)]
mod device_tests;

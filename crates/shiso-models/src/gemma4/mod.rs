//! The Gemma 4 family: what its two descriptions mean, and the geometry they
//! have to agree on.
//!
//! This family is unusual in being described twice. A published checkpoint ships
//! a `config.json` whose `text_config` carries every dimension; the quantized
//! container carries the same dimensions again under its own `gemma4.*` metadata
//! prefix, and a deployment may have either one without the other. So the family
//! has two readers rather than one, and the thing that makes that safe is that
//! both reduce their source to the same [`shiso_engine::shape::HybridShape`] -
//! asserted by a test against the real container, because two readers that each
//! pass their own checks and disagree by one head width would hand the same
//! weights two different stacks and nothing downstream could tell.
//!
//! [`Gemma4Config`], the tensor-role binding beside it, [`Gemma4Model`] and the
//! checkpoint identity are all here, for the same reason the other families keep
//! theirs together: they are the facts that are true of this checkpoint family
//! and of nothing else.

mod binding;
pub mod binding_mlx;
mod ckpt;
mod config;
mod model;
pub mod ple_repack;
mod vision;

pub use binding::{
    Gemma4AttnRoles, Gemma4DescBinding, Gemma4FfnRoles, Gemma4LayerRoles, Gemma4PleRoles,
    Gemma4Roles, VECTOR_TYPES, WEIGHT_TYPES, bind_gemma4_desc,
};
pub use binding_mlx::{Gemma4Names, bind_gemma4_mlx_desc};
pub use ckpt::{GEMMA4_MODEL_ID, quant_bits_of};
pub use config::{ARCHES, GGUF_ARCH, Gemma4Config};
pub use model::{Gemma4Model, PleOptions};
pub use vision::Gemma4Tower;

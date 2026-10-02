//! shiso-models - every checkpoint family this build can serve, one module each.
//!
//! A family module holds the three things that are true of its checkpoints
//! and of nothing else: the config reader that recognises-or-refuses a
//! document, the tensor-role binding that names its weights, and the model
//! type that ties both to the engine. Everything the families share - the
//! residual stream, the paged cache, the sessions, the sampler - is in the
//! engine, because none of it is a fact about one family.
//!
//! One crate rather than one per family: the families are edited together
//! (a new engine seam breaks all of them at once, by design), the checkpoint
//! tags they hand the shared cache directory have to be distinct across the
//! whole set, and the dispatch list above them wants one dependency, not a
//! growing row of them.
#![forbid(unsafe_code)]

pub mod audio_ops;
pub mod audio_registry;
pub mod flux2;
pub mod gemma4;
pub mod image_registry;
pub mod nemotron_asr;
pub mod nemotron_diar;
pub mod nemotron_h;
pub mod qwen35;
pub mod qwen35moe;
pub mod qwen3_asr;
pub mod qwen4exp;

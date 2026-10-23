//! Decision models run in-process: the arithmetic that turns scored label
//! rows into answers, shared by the server's decision routes and by embedders
//! that have no HTTP in between.
//!
//! The arithmetic and scoring modules name no model family; only the loader
//! knows which checkpoints it can open. The crate depends on no server, router
//! or HTTP crate. [`Decider`] opens a decision checkpoint and answers requests on a
//! worker thread of its own; [`score`] is the seam it implements, and the one
//! other runners (the server's scheduler) use the pure steps of.

#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!("the `metal` and `cuda` features are mutually exclusive");

pub mod decider;
pub mod error;
pub mod literal;
mod load;
pub mod math;
pub mod pull;
pub mod schema;
pub mod score;
mod worker;

#[cfg(test)]
mod tests;

pub use decider::{DEFAULT_CTX, Decider, OpenDesc};
pub use pull::{InstalledModel, ensure_model};

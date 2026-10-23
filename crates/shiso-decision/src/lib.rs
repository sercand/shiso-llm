//! Decision models run in-process: ask a model a multiple-choice question
//! about some text and get the probability of each answer back, with no HTTP
//! in between. The server's `/v1/decisions` route and embedding applications
//! share the arithmetic here.
//!
//! # What it serves
//!
//! Models that carry a **decision readout** or a **decision head**, which is
//! what [`Decider::open`] checks for. Scoring the labels of a plain chat model
//! needs the server's chat-template prompt renderer, so that stays in the
//! server. The arithmetic and scoring modules name no model family; only the
//! loader knows which checkpoints it can open. The crate depends on no
//! server, router or HTTP-server crate (the model download speaks HTTP as a
//! client only).
//!
//! [`score`] is the seam [`Decider`] implements, and the one other runners
//! (the server's scheduler) use the pure steps of; [`literal`] is the encoder
//! the decider and the server's vocabulary share. Both are seams for those
//! two callers, not API to build on: embedders use [`Decider`] and
//! [`ensure_model`].
//!
//! # Features
//!
//! Pick the backend with the `metal` (Apple) or `cuda` (NVIDIA) feature; they
//! are mutually exclusive and neither is on by default. A build with neither
//! compiles and tests, but [`ensure_model`] and [`Decider::open`] then fail
//! with [`DecideError::NoDevice`] before downloading or loading anything.
//!
//! # Threading
//!
//! A [`Decider`] owns the model on a worker thread of its own and answers
//! requests one at a time, in arrival order; share it between threads freely.
//! [`Decider::decide_blocking`] blocks the caller, [`Decider::decide`] is an
//! async future that needs no particular executor. Dropping a `Decider`
//! joins the worker, which can block for as long as one pass in progress, so
//! drop it from a thread that may block (not on an async executor's thread)
//! and expect any request still queued to fail with [`DecideError::Engine`].
//! [`ensure_model`] also blocks, on a runtime of its own, and refuses to run
//! inside an async executor.
//!
//! # Example
//!
//! ```ignore
//! use std::path::Path;
//! use shiso_decision::schema::{Choice, DecisionRequest, Question};
//! use shiso_decision::{Decider, OpenDesc, ensure_model};
//!
//! # fn main() -> Result<(), shiso_decision::error::DecideError> {
//! // Download (or find) the model; `false` from the callback cancels.
//! let model = ensure_model("d1-3b", Path::new("/var/lib/app/models"), &mut |done, total| {
//!     eprintln!("{done}/{total} bytes");
//!     true
//! })?;
//! let decider = Decider::open(&OpenDesc::new(&model.dir))?;
//! let choice = |value: &str| Choice { value: value.to_owned(), description: None };
//! let answers = decider.decide_blocking(&DecisionRequest {
//!     input: "cwd: /repo\ntask: unknown\ncmd: ls -la".to_owned(),
//!     questions: vec![Question::Choice {
//!         name: "decision".to_owned(),
//!         instructions: "Is this command safe to run without asking?".to_owned(),
//!         choices: vec![choice("approve"), choice("ask"), choice("deny")],
//!     }],
//!     temperature: None,
//! })?;
//! println!("{} {:?}", answers[0].choice, answers[0].probabilities);
//! # Ok(())
//! # }
//! ```
//!
//! [`DecideError`]: error::DecideError
//! [`DecideError::NoDevice`]: error::DecideError::NoDevice
//! [`DecideError::Engine`]: error::DecideError::Engine

#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!("the `metal` and `cuda` features are mutually exclusive");

pub mod decider;
pub mod error;
#[doc(hidden)]
pub mod literal;
mod load;
pub mod math;
pub mod pull;
pub mod schema;
#[doc(hidden)]
pub mod score;
mod worker;

#[cfg(test)]
mod tests;

pub use decider::{DEFAULT_CTX, Decider, OpenDesc};
pub use pull::{InstalledModel, ensure_model};

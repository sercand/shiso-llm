//! The decision-scoring core: [`decide`] answers a `shiso_ir::decision`
//! request from the model an [`AppState`](crate::state::AppState) serves.
//!
//! A decision question is answered from the next-token distribution at the
//! answer position of a single prefill: the question is rendered with the
//! model's chat template, read once, and the labels (letters, digits, or
//! `yes`/`no`) are scored from that row. Nothing is generated. A decision
//! checkpoint instead lays the questions out itself and is read through its
//! readout rows or its head.
//!
//! # What lives where
//!
//! - [`core`]: the question model's prompt wording (prompt-format version 1),
//!   the validators every surface shares, preparation, the scoring fan-out
//!   and [`decide`] itself.
//! - [`head`] and [`readout`]: the two ways a decision checkpoint is read.
//! - [`math`] and [`score`]: the pure arithmetic - probabilities, confidences,
//!   and folding a scored row into an answer - which name no model family and
//!   touch no device.
//! - [`literal`]: encoding prompt text in which some byte ranges are caller
//!   text, so a caller's words never become a marker of the prompt.
//!
//! Parsing a wire body into a request and spelling an answer back out are the
//! surface's: `/v1/decisions` and `/v1/systemone` live in `shiso-server`, and an
//! embedding application builds the IR directly.

pub mod core;
pub(crate) mod head;
pub mod literal;
pub mod math;
pub(crate) mod readout;
pub mod score;

pub use core::{decide, scored_answer};

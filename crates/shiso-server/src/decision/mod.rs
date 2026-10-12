//! The decision-scoring API family: `POST /v1/decisions` and `POST /v1/systemone`.
//!
//! A decision question is answered from the next-token distribution at the
//! answer position of a single prefill: the question is rendered with the
//! server's chat template, read once, and the labels (letters, digits, or
//! `yes`/`no`) are scored from that row. Nothing is generated.
//!
//! # The two surfaces share one core
//!
//! `decisions` and `systemone` are two wire readings of the same engine — the
//! shared parts (question model, prompt-format version 1 wording, validation,
//! label checks, scoring fan-out) live in `core`, the probability and
//! confidence formulas in `math`. Each surface owns its own serde structs,
//! like every other route in this crate owns its own wire format: `/v1/decisions`
//! speaks the official `OpenAI` Decisions API (public beta) shape, `/v1/systemone`
//! the System One shape with its `images` extension, and the two differ exactly
//! where their wires do - a `predicate` question and a `noul` criterion are the
//! same `Kind::YesNo` underneath.

pub(crate) mod core;
pub(crate) mod decisions;
pub(crate) mod decisions_encode;
pub(crate) mod decisions_wire;
pub(crate) mod head;
pub(crate) mod math;
pub(crate) mod readout;
pub(crate) mod systemone;
pub(crate) mod systemone_encode;
pub(crate) mod systemone_wire;

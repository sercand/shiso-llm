//! shiso-bench - the measured lane and the graded one.
//!
//! Two jobs share this crate because they share a loader, a device and a
//! readback buffer, and nothing else:
//!
//! - the **perf** side records what a checkpoint costs on a device and grades
//!   what was recorded. Measuring and grading are two commands, and only the
//!   first of them ever touches a device.
//! - the **parity** binary and the modules under it record and grade what a
//!   checkpoint *answers*. They assert; they time nothing.
//!
//! The separation is deliberate and is the crate's one rule: no parity path
//! records a duration and no perf path records a logit. A parity number that
//! moved because a machine was busy would be unfalsifiable, and a throughput
//! number that moved because a fixture changed would be meaningless.
//!
//! # What parity can mean here, and what it cannot
//!
//! A fixture captured on the same machine that replays it is a **regression
//! lock**: it proves that today's build answers what yesterday's build answered.
//! It proves nothing about whether either answer is right. The only artifact in
//! this tree that can prove correctness is one captured from an independent
//! implementation of the same model, and that capture needs hardware this
//! machine does not have.
//!
//! So the harness is built to extract as much as a single-implementation machine
//! can honestly give:
//!
//! 1. the recorded slice is the top of the *distribution*, not the argmax, so a
//!    build that still picks the same winner for the wrong reason fails
//!    ([`compare`]);
//! 2. the comparator refuses a slice that is not a slice of a real distribution
//!    at all - non-finite, collapsed, or out of rank order - so a catastrophic
//!    failure cannot pass by producing garbage that happens to match;
//! 3. the engine is compared against itself along two routes through the same
//!    weights, which needs no reference at all ([`replay`]).
//!
//! # What a perf number can mean here, and what it cannot
//!
//! The same discipline, applied to timings. A throughput figure taken on one
//! machine says what that machine did on that day, so it is recorded with the
//! protocol that produced it, the agreement between its repetitions, and a
//! digest of the sources it was taken against ([`measure`], [`baseline`]).
//! Grading happens somewhere else entirely: [`baseline`] does arithmetic on
//! committed JSON, needs no device, and writes nothing. That split is the only
//! reason a five-per-cent threshold is worth anything - a threshold applied to
//! a fresh sample from whatever machine is running the gate would be noise
//! wearing a gate's clothes.
#![forbid(unsafe_code)]

pub mod baseline;
pub mod chunking;
pub mod compare;
pub mod decode;
pub mod kv;
pub mod measure;
pub mod ple;
pub mod prefill;
pub mod replay;
pub mod run;
pub mod vecfile;

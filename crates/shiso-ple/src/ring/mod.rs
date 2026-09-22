//! The ring reader seam: one type surface, two implementations, chosen by target.
//!
//! Everything above this module - the cache, the fetch passes, the stream
//! itself - is ordinary portable Rust that names [`uring::Ring`]. Which body
//! that name reaches is decided here, once, by `cfg(target_os)`:
//!
//! * on Linux it is a real `io_uring`, built on the kernel's submission and
//!   completion queues;
//! * everywhere else it is the stub arm, where the constructor answers
//!   [`crate::PleError::BackendRefused`] and the type itself is uninhabited, so
//!   the batch path cannot be reached even in principle.
//!
//! # Why a module swap and not a trait
//!
//! A trait would need an associated type for the ring handle just so the stub
//! arm had something to name, and it would put a dynamic dispatch on the one
//! call that matters. A module swap costs nothing at runtime, keeps both arms
//! honest by making them answer to the same call sites, and means the stub arm
//! *is* the type-check harness: if a signature drifts on one side, the other
//! stops compiling.
//!
//! # Why the refusal is loud
//!
//! A target without `io_uring` and a Linux build whose kernel refuses to give one
//! are the same situation, and in both a caller who asked for the ring has to
//! hear about it. Downgrading silently would let a measurement that means to
//! compare the two readers compare one of them with itself.

// One module name, two files, chosen by target. Both files declare the same
// type and the same three methods; neither declares a nested module, so the
// `#[path]` below is the whole of the mechanism.
#[cfg(target_os = "linux")]
#[path = "uring.rs"]
mod arm;
#[cfg(not(target_os = "linux"))]
#[path = "stub.rs"]
mod arm;

/// The ring reader for this target.
///
/// Call sites name `ring::uring::Ring` on every target; only the body behind it
/// changes.
pub(crate) mod uring {
    pub(crate) use super::arm::Ring;
}

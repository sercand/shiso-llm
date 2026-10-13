//! The per-platform decoder sessions.
//!
//! Each arm exports the same two items - `Session` with `open`, `decode` and
//! `flush`, and `probe` - so [`crate::HwDecoder`] has one body. The arm for a
//! platform with no OS decoder answers [`crate::HwError::Unavailable`], which is
//! what makes the crate build everywhere.

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub(crate) use macos::{Session, probe};

// Also compiled for tests on other Unix hosts, so its driver-free pieces are
// unit-tested on machines without an NVIDIA GPU. Unix only: the bindings'
// layout asserts assume an LP64 `unsigned long`, which Windows does not have.
#[cfg(any(target_os = "linux", all(test, unix)))]
mod linux;
#[cfg(target_os = "linux")]
pub(crate) use linux::{Session, probe};

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod other;
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) use other::{Session, probe};

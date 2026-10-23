//! The CPU device's kernel table.
//!
//! One row per `(module, entry)`: the group code and how the dispatcher must
//! treat its arguments. Later kernels add their rows here; a launch that names
//! an entry without a row is `Unsupported`, never a silent no-op.

mod args;

use crate::dispatch::Table;

/// Every kernel the CPU device runs. Empty until the first kernels land.
pub(crate) static TABLE: &Table = &[];

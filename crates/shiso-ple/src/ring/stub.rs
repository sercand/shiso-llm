//! The ring reader on a target that has no `io_uring`.
//!
//! Every entry point answers a typed refusal rather than panicking, and the
//! type has no inhabitant, so "the ring ran on macOS" is not a state this crate
//! can be in - it is a state it cannot express.

use crate::handle::{Counters, RowReader};
use crate::{PleError, PleResult};

/// A ring that cannot exist here.
///
/// The private field is uninhabited, which is what makes that structural rather
/// than a matter of nobody calling the constructor: there is no value of this
/// type, so [`Ring::read_rows`] has no reachable body to write.
#[derive(Debug)]
pub(crate) struct Ring(Absent);

/// The type with no values.
#[derive(Debug)]
enum Absent {}

impl Ring {
    /// Refuses, naming the thing that is missing.
    ///
    /// # Errors
    ///
    /// Always [`PleError::BackendRefused`].
    pub(crate) const fn open(_reader: &RowReader, _qd: u32) -> PleResult<Self> {
        Err(PleError::BackendRefused(
            "io_uring: this target has no io_uring, so the ring reader does not exist in this build",
        ))
    }

    /// Unreachable: there is no `Ring` on this target to call it on.
    pub(crate) fn read_rows(
        &mut self,
        _reader: &RowReader,
        _rows: &[u64],
        _staging: &mut [u8],
        _stats: &Counters,
    ) -> PleResult<()> {
        match self.0 {}
    }

    /// Unreachable, for the same reason.
    pub(crate) fn is_poisoned(&self) -> bool {
        match self.0 {}
    }
}

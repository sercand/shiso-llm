//! Device scratch a model keeps from one call to the next.
//!
//! A fresh device buffer is expensive to first touch: on an M1 Max a new
//! 111 MB buffer took 5.6 ms to write once and 9.2 ms to zero-fill and
//! drain, against 1.0 ms to rewrite one already touched - some 50 us a
//! megabyte of page faults. A tower's conv stem makes some 200 MB of patches
//! and planes per group of chunks and a decoder tens of megabytes of
//! activations per transcription, so a model that keeps them pays that once.

use shiso_core::{CoreResult, DeviceBuffer};

use super::Ops;

/// Grow-only numbered buffers, each at least as long as the last call asked
/// and holding whatever that call left in it: a slot is for a buffer every
/// element of which is written before it is read.
#[derive(Debug, Default)]
pub struct Workspace {
    slots: Vec<Option<DeviceBuffer>>,
}

impl Workspace {
    /// An empty workspace; the first reservation allocates.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Makes slot `i` of `sizes` hold at least `sizes[i]` f32 values,
    /// replacing (and dropping) a slot that is shorter. A replaced slot's
    /// contents are gone; one kept holds the last call's.
    ///
    /// # Errors
    ///
    /// Allocation failures from the backend.
    pub fn reserve(&mut self, ops: &Ops, sizes: &[usize]) -> CoreResult<()> {
        if self.slots.len() < sizes.len() {
            self.slots.resize_with(sizes.len(), || None);
        }
        for (slot, &floats) in self.slots.iter_mut().zip(sizes) {
            let want = u64::try_from(floats.max(1))
                .unwrap_or(u64::MAX)
                .saturating_mul(4);
            if slot.as_ref().is_none_or(|b| b.len() < want) {
                *slot = None;
                *slot = Some(ops.scratch_uninit(floats)?);
            }
        }
        Ok(())
    }

    /// Slot `i`, reserved by the last [`Self::reserve`].
    ///
    /// # Panics
    ///
    /// When slot `i` was never reserved: a caller reserves every slot it
    /// takes first, so this is a bug in the caller.
    #[must_use]
    pub fn get(&self, i: usize) -> &DeviceBuffer {
        self.slots
            .get(i)
            .and_then(Option::as_ref)
            .expect("a workspace slot is reserved before it is taken")
    }
}

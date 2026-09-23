//! The per-row pointer arrays a batched decode dereferences, and who fills
//! them.
//!
//! Six kernels in this set carry per-sequence state through a launch that
//! serves a whole batch: the recurrent convolution and recurrence, the n-gram
//! convolution, the key/value store, the dense attention and the indexer's key
//! store. Each of them takes, in place of the one buffer its scalar sibling
//! takes, a *device array* of one pointer per row - `float * const *states`, or
//! `const int32_t *const *pages_arr`. The array is what makes one launch serve
//! rows that belong to different sequences.
//!
//! # Why this is a step of the pass and not an operand
//!
//! An [`Operand`] names a buffer, and the operand vocabulary can name each of
//! these arrays. What it cannot name is the act of *filling* one: the contents
//! are addresses, a [`DeviceView`] is deliberately not an address, and the only
//! thing in the process that can turn one into the other is the backend. So the
//! pack is a host-side step between writing the inputs and issuing the pass,
//! exactly like the input rows beside it, and for the same reason: nothing
//! fails when it is skipped. A launch handed an unfilled array dereferences
//! whatever the allocator left at those addresses, and on a poisoned allocation
//! that is a fault, while on a real device it is another tenant's memory.
//!
//! [`Operand`]: crate::sparse::bind::Operand
//!
//! # The layout, and why the stride is the batch's own width
//!
//! One array per kind of per-session state a batched launch addresses, laid out
//! back to back inside [`Buf::RowPointers`], each holding one slot per row of
//! *this* batch. The stride is the batch's own row count rather than the widest
//! one the block can take, which is the opposite of the choice every other
//! buffer here makes - and it is safe for one reason only: a batched island is
//! never captured, so there is no baked-in address for a narrower batch to
//! move. See [`crate::sparse::island::IslandContext`].

use shiso_core::backend::{Backend, DEVICE_POINTER_BYTES};
use shiso_core::buffer::DeviceView;
use shiso_core::error::{CoreError, CoreResult};

use crate::shape::LayerKind;
use crate::sparse::scratch::{Buf, Scratch};
use crate::sparse::shape::SparseShape;
use crate::sparse::stack::SparseSession;

/// Which per-row pointer array a slot of the pack holds.
///
/// Deliberately NOT `#[non_exhaustive]`: the block families build one to name
/// the array their launch reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PtrArray {
    /// One page table per row, which every cache-addressing kernel reads.
    Pages,
    /// One n-gram convolution window per row.
    NgramWindows,
    /// One recurrent slot's convolution window per row.
    RecurrentConv(u32),
    /// One recurrent slot's state matrix per row.
    RecurrentState(u32),
    /// One recurrent slot's convolution rollback records per row.
    RecurrentConvCkpt(u32),
    /// One recurrent slot's state rollback records per row.
    RecurrentStateCkpt(u32),
}

/// Where each array sits inside the pack.
///
/// A value rather than three loose numbers because the three are what decide
/// every offset, and two of them read from different places - the geometry says
/// how many recurrent slots and whether there is an n-gram block, the batch says
/// how many rows. An offset computed from a mismatched pair is an array that
/// overlaps its neighbour, which is one row's state pointer read as another's.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Rows this batch carries, which is the stride of every array.
    pub rows: u32,
    /// Recurrent slots the stack holds state for.
    pub slots: u32,
    /// Whether the stack folds in an n-gram block, and therefore whether the
    /// window array exists at all.
    pub ngram: bool,
    /// Whether the rows carry speculative rollback records, and therefore
    /// whether the two record arrays per recurrent slot exist. They come after
    /// every other array, so a pack without them lays the rest out exactly as
    /// one with them does.
    pub rollback: bool,
}

impl Layout {
    /// The layout a geometry implies at `rows` rows.
    ///
    /// The *sizing* half, and the only one that can answer before a session
    /// exists: [`Buf::RowPointers`] is allocated when the stack is built.
    /// Addressing goes through [`Layout::of_rows`], which asks the sessions
    /// themselves. The geometry cannot say whether a session will be opened
    /// for speculation, so this sizes for the record arrays as well.
    #[must_use]
    pub fn of(shape: &SparseShape, rows: u32) -> Self {
        Self {
            rows,
            slots: shape.count(LayerKind::Gdn),
            ngram: shape.ple.is_some(),
            rollback: true,
        }
    }

    /// The layout a batch of live sessions is packed and read at.
    ///
    /// Derived from the sessions rather than from the geometry because that is
    /// what both ends can see: the writer holds them and so does the binder
    /// that resolves an array operand, while neither holds a shape. Two
    /// derivations of one layout would be two chances for the writer and the
    /// reader to disagree about where an array starts, and an array read one
    /// slot over is one row's state pointer answered for another's.
    ///
    /// It agrees with [`Layout::of`] by construction - a session allocates one
    /// state pair per recurrent slot the geometry names and an n-gram window
    /// exactly when the geometry folds one in - and that agreement is what
    /// makes the buffer sized from the shape big enough for the pack written
    /// from the sessions.
    #[must_use]
    pub fn of_rows(rows: &[&SparseSession]) -> Self {
        let first = rows.first();
        Self {
            rows: u32::try_from(rows.len()).unwrap_or(u32::MAX),
            slots: first.map_or(0, |row| {
                u32::try_from(row.recurrent_slots()).unwrap_or(u32::MAX)
            }),
            ngram: first.is_some_and(|row| row.ngram_window().is_some()),
            // Every row or none: a record array with a hole in it would be a
            // launch writing one row's records through a pointer nobody packed.
            rollback: !rows.is_empty() && rows.iter().all(|row| row.recurrent_rollback(0).is_ok()),
        }
    }

    /// Arrays the pack holds.
    #[must_use]
    pub const fn arrays(&self) -> u32 {
        let per_slot = if self.rollback { 4 } else { 2 };
        1u32.saturating_add(self.ngram as u32)
            .saturating_add(self.slots.saturating_mul(per_slot))
    }

    /// Bytes the whole pack occupies.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        (self.arrays() as u64)
            .saturating_mul(self.rows as u64)
            .saturating_mul(DEVICE_POINTER_BYTES)
    }

    /// Which array this is, counting from the front of the pack.
    ///
    /// `None` is an answer rather than a failure: it means this stack has no
    /// such array - an n-gram window on a stack that folds in no n-gram block,
    /// or a recurrent slot past the last one - and the caller turns that into a
    /// refusal naming the state, which is more use than an index.
    #[must_use]
    pub const fn ordinal(&self, array: PtrArray) -> Option<u32> {
        let after_pages = 1u32.saturating_add(self.ngram as u32);
        let after_state = after_pages.saturating_add(self.slots.saturating_mul(2));
        match array {
            PtrArray::Pages => Some(0),
            PtrArray::NgramWindows => {
                if self.ngram {
                    Some(1)
                } else {
                    None
                }
            }
            PtrArray::RecurrentConv(slot) => {
                if slot < self.slots {
                    Some(after_pages.saturating_add(slot.saturating_mul(2)))
                } else {
                    None
                }
            }
            PtrArray::RecurrentState(slot) => {
                if slot < self.slots {
                    Some(
                        after_pages
                            .saturating_add(slot.saturating_mul(2))
                            .saturating_add(1),
                    )
                } else {
                    None
                }
            }
            PtrArray::RecurrentConvCkpt(slot) => {
                if self.rollback && slot < self.slots {
                    Some(after_state.saturating_add(slot.saturating_mul(2)))
                } else {
                    None
                }
            }
            PtrArray::RecurrentStateCkpt(slot) => {
                if self.rollback && slot < self.slots {
                    Some(
                        after_state
                            .saturating_add(slot.saturating_mul(2))
                            .saturating_add(1),
                    )
                } else {
                    None
                }
            }
        }
    }

    /// The byte range one array occupies inside the pack.
    ///
    /// `None` is an answer rather than a failure, for the reason
    /// [`Layout::ordinal`] gives.
    #[must_use]
    pub const fn span(&self, array: PtrArray) -> Option<(u64, u64)> {
        let Some(ordinal) = self.ordinal(array) else {
            return None;
        };
        let width = (self.rows as u64).saturating_mul(DEVICE_POINTER_BYTES);
        Some(((ordinal as u64).saturating_mul(width), width))
    }
}

/// Arrays a stack of this geometry packs at its widest batch.
///
/// The sizing half of [`Layout`], separate because [`Buf::RowPointers`] is
/// allocated once for the widest batch the block can take while a pack is
/// written at the width of the batch in front of it.
#[must_use]
pub fn arrays(shape: &SparseShape) -> u32 {
    Layout::of(shape, 0).arrays()
}

/// Fills every per-row pointer array for one batch, in one transfer.
///
/// One call rather than one per array: the arrays are contiguous and in
/// ordinal order, so the whole pack is one list of views, and a device that
/// wrote them one array at a time would take a hundred transfers a step on the
/// shipping geometry.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the device cannot hand out addresses at all,
/// or when a row's session holds none of the state an array names;
/// [`CoreError::ShapeMismatch`] when the rows disagree about how many recurrent
/// slots they carry, which would mean two of them came from different stacks;
/// and whatever the backend reports for the transfer.
pub fn pack(
    backend: &dyn Backend,
    scratch: &Scratch,
    shape: &SparseShape,
    rows: &[&SparseSession],
) -> CoreResult<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let layout = Layout::of_rows(rows);
    // The buffer was sized from the geometry and is written from the sessions.
    // They agree by construction; this is where that stops being an argument,
    // because a pack wider than the buffer is a write past the end of the
    // scratch rather than a failed launch.
    let sized = Layout::of(shape, layout.rows);
    if sized.arrays() < layout.arrays() {
        return Err(CoreError::ShapeMismatch {
            field: "per-row pointer arrays",
            expected: u64::from(sized.arrays()),
            got: u64::from(layout.arrays()),
        });
    }
    let pointers = backend.as_pointers().ok_or(CoreError::Unsupported {
        what: "a batched decode on a device that cannot write the addresses of its own \
               allocations, which is what the per-row pointer arrays hold",
        backend: "engine",
    })?;
    let slots = usize::try_from(layout.slots).unwrap_or(usize::MAX);
    for row in rows {
        if row.recurrent_slots() != slots {
            return Err(CoreError::ShapeMismatch {
                field: "recurrent slots per row",
                expected: layout.slots.into(),
                got: row.recurrent_slots() as u64,
            });
        }
    }
    let missing = |what: &'static str| CoreError::Unsupported {
        what,
        backend: "engine",
    };

    let mut views: Vec<DeviceView<'_>> =
        Vec::with_capacity(usize::try_from(layout.arrays()).unwrap_or(0) * rows.len());
    for row in rows {
        views.push(row.pages());
    }
    if layout.ngram {
        for row in rows {
            views.push(row.ngram_window().ok_or_else(|| {
                missing(
                    "the n-gram convolution window of a session whose stack folds in an n-gram \
                     block",
                )
            })?);
        }
    }
    for slot in 0..layout.slots {
        for row in rows {
            views.push(row.recurrent(slot)?.0);
        }
        for row in rows {
            views.push(row.recurrent(slot)?.1);
        }
    }
    if layout.rollback {
        for slot in 0..layout.slots {
            for row in rows {
                views.push(row.recurrent_rollback(slot)?.0);
            }
            for row in rows {
                views.push(row.recurrent_rollback(slot)?.1);
            }
        }
    }

    let dst = scratch.view(Buf::RowPointers)?.slice(0, layout.bytes())?;
    pointers.pack_pointers(dst, &views)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYOUT: Layout = Layout {
        rows: 4,
        slots: 3,
        ngram: true,
        rollback: true,
    };

    #[test]
    fn every_array_has_its_own_span_and_no_two_of_them_overlap() {
        // The failure this rules out is the one the batched kernels exist to
        // fix wearing a new hat: two arrays at one offset is one slot's state
        // pointer read as another's, and every launch still succeeds.
        let mut spans = Vec::new();
        spans.push(LAYOUT.span(PtrArray::Pages).expect("every pack has pages"));
        spans.push(
            LAYOUT
                .span(PtrArray::NgramWindows)
                .expect("this stack folds in an n-gram block"),
        );
        for slot in 0..LAYOUT.slots {
            spans.push(
                LAYOUT
                    .span(PtrArray::RecurrentConv(slot))
                    .expect("the slot exists"),
            );
            spans.push(
                LAYOUT
                    .span(PtrArray::RecurrentState(slot))
                    .expect("the slot exists"),
            );
            spans.push(
                LAYOUT
                    .span(PtrArray::RecurrentConvCkpt(slot))
                    .expect("the slot has records"),
            );
            spans.push(
                LAYOUT
                    .span(PtrArray::RecurrentStateCkpt(slot))
                    .expect("the slot has records"),
            );
        }
        assert_eq!(spans.len(), LAYOUT.arrays() as usize);
        spans.sort_unstable();
        for pair in spans.windows(2) {
            assert_eq!(
                pair[0].0 + pair[0].1,
                pair[1].0,
                "two arrays leave a gap or overlap: {spans:?}"
            );
        }
        let (last, width) = spans[spans.len() - 1];
        assert_eq!(last + width, LAYOUT.bytes());
    }

    #[test]
    fn a_stack_with_no_n_gram_block_has_no_window_array_and_the_rest_close_up() {
        let without = Layout {
            ngram: false,
            ..LAYOUT
        };
        assert_eq!(without.ordinal(PtrArray::NgramWindows), None);
        assert_eq!(without.arrays(), LAYOUT.arrays() - 1);
        // The recurrent arrays start one slot earlier rather than leaving a
        // hole where the window array would have been.
        assert_eq!(without.ordinal(PtrArray::RecurrentConv(0)), Some(1));
        assert_eq!(LAYOUT.ordinal(PtrArray::RecurrentConv(0)), Some(2));
    }

    #[test]
    fn a_recurrent_slot_past_the_last_one_has_no_span_rather_than_a_wrapped_offset() {
        assert_eq!(LAYOUT.ordinal(PtrArray::RecurrentConv(3)), None);
        assert_eq!(LAYOUT.ordinal(PtrArray::RecurrentState(3)), None);
        assert!(LAYOUT.ordinal(PtrArray::RecurrentState(2)).is_some());
    }

    #[test]
    fn a_pack_without_records_keeps_every_other_array_where_it_was() {
        // The record arrays come last, so whether a batch's sessions were
        // opened for speculation moves nothing a plain batched launch reads.
        let plain = Layout {
            rollback: false,
            ..LAYOUT
        };
        assert_eq!(plain.ordinal(PtrArray::RecurrentStateCkpt(0)), None);
        assert_eq!(plain.arrays(), LAYOUT.arrays() - 2 * LAYOUT.slots);
        for slot in 0..LAYOUT.slots {
            assert_eq!(
                plain.span(PtrArray::RecurrentState(slot)),
                LAYOUT.span(PtrArray::RecurrentState(slot))
            );
        }
    }

    #[test]
    fn the_stride_is_the_batchs_own_width() {
        let narrow = Layout { rows: 2, ..LAYOUT };
        assert_eq!(
            narrow.span(PtrArray::RecurrentConv(0)),
            Some((2 * 2 * DEVICE_POINTER_BYTES, 2 * DEVICE_POINTER_BYTES))
        );
        assert_eq!(
            LAYOUT.span(PtrArray::RecurrentConv(0)),
            Some((2 * 4 * DEVICE_POINTER_BYTES, 4 * DEVICE_POINTER_BYTES))
        );
    }
}

//! Host allocations behind the CPU device's buffer handles.
//!
//! A slab is a `Vec<Chunk>` rather than a `Vec<u8>`: a byte vector is only
//! promised one-byte alignment, so reinterpreting it as `f32` could fail on an
//! allocator that hands back an odd address. A 16-byte-aligned chunk makes
//! every slab base 16-aligned, which also backs each allocation with a whole
//! number of 16-byte chunks for free. Alignment of a *view* then depends only
//! on its offset, which is the caller's to get right and is refused loudly
//! when it is not.
//!
//! The chunks are storage only: a slab remembers the byte length that was
//! requested, reports it as the buffer's size, and bounds-checks every view
//! against it, so the padding of the last chunk is never reachable.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use shiso_core::{CoreError, CoreResult, DeviceView, RawBuffer};

/// Bytes per [`Chunk`], and so the granularity of a slab's storage (not of the
/// size it reports).
const CHUNK_BYTES: usize = 16;

/// Sixteen bytes with sixteen-byte alignment; the unit slabs are made of.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(16))]
pub(crate) struct Chunk([u8; CHUNK_BYTES]);

/// One allocation: chunk storage and the byte length that was asked for.
#[derive(Debug)]
struct Slab {
    chunks: Vec<Chunk>,
    /// Requested size; views are checked against this, not the storage.
    bytes: usize,
}

/// Every live allocation of one CPU device, by handle.
#[derive(Debug)]
pub(crate) struct Slabs {
    /// The next handle to hand out. Never reused, so a stale view of a freed
    /// buffer is refused instead of silently reading a newer allocation.
    next: NonZeroU64,
    map: HashMap<NonZeroU64, Slab>,
}

impl Slabs {
    /// An empty set of slabs.
    pub(crate) fn new() -> Self {
        Self {
            next: NonZeroU64::MIN,
            map: HashMap::new(),
        }
    }

    /// Allocates `bytes` of zeroed memory, backed by a whole number of chunks.
    ///
    /// The handle's size is the requested one, not the rounded one: a whole-
    /// buffer view must be exactly as long as the bytes a caller uploads, since
    /// `copy_h2d` and `copy_d2h` refuse any other length.
    pub(crate) fn alloc(&mut self, bytes: u64) -> CoreResult<RawBuffer> {
        let chunks = usize::try_from(bytes.div_ceil(CHUNK_BYTES as u64)).map_err(|_| {
            CoreError::OutOfBounds {
                offset: 0,
                len: bytes,
                buffer_bytes: 0,
            }
        })?;
        // Zeroed by the allocator rather than written: the OS maps zero pages
        // on first touch, so a large KV pool costs address space at boot, not
        // resident memory (a written `Vec` committed the whole pool, and on an
        // overcommitting Linux host passed the reservation and was killed
        // later). A request the host cannot satisfy is an error the caller can
        // see, not the abort `vec!` would turn it into.
        let slab: Vec<Chunk> =
            bytemuck::allocation::try_zeroed_vec(chunks).map_err(|()| CoreError::Backend {
                backend: "cpu",
                op: "alloc_raw",
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    format!("the host could not allocate {bytes} bytes"),
                )),
            })?;
        let id = self.next;
        // A u64 handle counter cannot wrap in any process lifetime; the
        // fallback keeps the arithmetic total without an `unwrap`.
        self.next = id.checked_add(1).unwrap_or(id);
        let len = usize::try_from(bytes).map_err(|_| CoreError::OutOfBounds {
            offset: 0,
            len: bytes,
            buffer_bytes: 0,
        })?;
        self.map.insert(
            id,
            Slab {
                chunks: slab,
                bytes: len,
            },
        );
        Ok(RawBuffer::new(id, bytes))
    }

    /// Drops an allocation. Returns whether the handle was live.
    pub(crate) fn free(&mut self, id: NonZeroU64) -> bool {
        self.map.remove(&id).is_some()
    }

    /// The bytes behind `view`.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when the range leaves its allocation, and
    /// [`CoreError::InvalidValue`] for a handle that is not live.
    pub(crate) fn bytes(&self, view: DeviceView<'_>) -> CoreResult<&[u8]> {
        let slab = self.map.get(&view.id()).ok_or_else(|| stale(view))?;
        let all: &[u8] = bytemuck::cast_slice(&slab.chunks);
        let range = span(view, slab.bytes)?;
        Ok(&all[range])
    }

    /// The bytes behind `view`, writable.
    ///
    /// # Errors
    ///
    /// What [`Slabs::bytes`] reports.
    pub(crate) fn bytes_mut(&mut self, view: DeviceView<'_>) -> CoreResult<&mut [u8]> {
        let slab = self.map.get_mut(&view.id()).ok_or_else(|| stale(view))?;
        let range = span(view, slab.bytes)?;
        let all: &mut [u8] = bytemuck::cast_slice_mut(&mut slab.chunks);
        Ok(&mut all[range])
    }

    /// Copies `src` over `dst`, which must be the same length; the two may
    /// share an allocation and overlap, as `memmove` does.
    ///
    /// # Errors
    ///
    /// What [`Slabs::bytes`] reports for either view.
    pub(crate) fn copy(&mut self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()> {
        // Both ranges are proven before anything moves, so the removal below
        // cannot lose a slab to an early return.
        let from = span(src, self.slab_bytes(src)?)?;
        let to = span(dst, self.slab_bytes(dst)?)?;
        if dst.id() == src.id() {
            let slab = self.map.get_mut(&src.id()).ok_or_else(|| stale(src))?;
            let all: &mut [u8] = bytemuck::cast_slice_mut(&mut slab.chunks);
            all.copy_within(from, to.start);
            return Ok(());
        }
        // The destination slab is taken out of the map for the copy so the
        // source can be borrowed beside it, without staging through a `Vec`.
        let mut dst_slab = self.map.remove(&dst.id()).ok_or_else(|| stale(dst))?;
        if let Some(src_slab) = self.map.get(&src.id()) {
            let source: &[u8] = bytemuck::cast_slice(&src_slab.chunks);
            let target: &mut [u8] = bytemuck::cast_slice_mut(&mut dst_slab.chunks);
            target[to].copy_from_slice(&source[from]);
        }
        self.map.insert(dst.id(), dst_slab);
        Ok(())
    }

    /// Size in bytes of the allocation `view` names.
    fn slab_bytes(&self, view: DeviceView<'_>) -> CoreResult<usize> {
        let slab = self.map.get(&view.id()).ok_or_else(|| stale(view))?;
        Ok(slab.bytes)
    }
}

/// The error for a view of a handle that is not live.
fn stale(view: DeviceView<'_>) -> CoreError {
    CoreError::InvalidValue {
        key: "buffer handle",
        value: view.id().get().to_string(),
    }
}

/// `view`'s byte range inside an allocation of `bytes`.
fn span(view: DeviceView<'_>, bytes: usize) -> CoreResult<Range<usize>> {
    let out_of_bounds = || CoreError::OutOfBounds {
        offset: view.offset(),
        len: view.len(),
        buffer_bytes: bytes as u64,
    };
    let end = view
        .offset()
        .checked_add(view.len())
        .filter(|&end| end <= bytes as u64)
        .ok_or_else(out_of_bounds)?;
    let start = usize::try_from(view.offset()).map_err(|_| out_of_bounds())?;
    let end = usize::try_from(end).map_err(|_| out_of_bounds())?;
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_allocation_the_host_cannot_hold_is_an_error_not_an_abort() {
        // Past what any layout can describe, so the refusal does not depend on
        // how much memory the test host has.
        let refused = Slabs::new().alloc(u64::MAX / 2);
        assert!(refused.is_err(), "{refused:?}");
    }

    #[test]
    fn an_allocation_reads_back_as_zeros() {
        let mut slabs = Slabs::new();
        let raw = slabs.alloc(4096 + 3).expect("a small allocation");
        let slab = slabs.map.get(&raw.id()).expect("live");
        assert_eq!(slab.bytes, 4096 + 3);
        assert!(
            bytemuck::cast_slice::<Chunk, u8>(&slab.chunks)
                .iter()
                .all(|&b| b == 0)
        );
    }
}

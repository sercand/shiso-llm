//! Host allocations behind the CPU device's buffer handles.
//!
//! A slab is a `Vec<Chunk>` rather than a `Vec<u8>`: a byte vector is only
//! promised one-byte alignment, so reinterpreting it as `f32` could fail on an
//! allocator that hands back an odd address. A 16-byte-aligned chunk makes
//! every slab base 16-aligned, which also backs each allocation with a whole
//! number of 16-byte chunks for free. Alignment of a *view* then depends only on its offset,
//! which is the caller's to get right and is refused loudly when it is not.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use shiso_core::{CoreError, CoreResult, DeviceView, RawBuffer};

/// Bytes per [`Chunk`], and so the rounding of every allocation.
const CHUNK_BYTES: usize = 16;

/// Sixteen bytes with sixteen-byte alignment; the unit slabs are made of.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(16))]
pub(crate) struct Chunk([u8; CHUNK_BYTES]);

/// Every live allocation of one CPU device, by handle.
#[derive(Debug)]
pub(crate) struct Slabs {
    /// The next handle to hand out. Never reused, so a stale view of a freed
    /// buffer is refused instead of silently reading a newer allocation.
    next: NonZeroU64,
    map: HashMap<NonZeroU64, Vec<Chunk>>,
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
        let mut slab: Vec<Chunk> = Vec::new();
        // A request the host cannot satisfy is an error the caller can see,
        // not the abort `vec!` would turn it into.
        slab.try_reserve_exact(chunks)
            .map_err(|source| CoreError::Backend {
                backend: "cpu",
                op: "alloc_raw",
                source: Box::new(source),
            })?;
        slab.resize(chunks, Chunk::zeroed());
        let id = self.next;
        // A u64 handle counter cannot wrap in any process lifetime; the
        // fallback keeps the arithmetic total without an `unwrap`.
        self.next = id.checked_add(1).unwrap_or(id);
        self.map.insert(id, slab);
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
        let all: &[u8] = bytemuck::cast_slice(slab);
        let range = span(view, all.len())?;
        Ok(&all[range])
    }

    /// The bytes behind `view`, writable.
    ///
    /// # Errors
    ///
    /// What [`Slabs::bytes`] reports.
    pub(crate) fn bytes_mut(&mut self, view: DeviceView<'_>) -> CoreResult<&mut [u8]> {
        let slab = self.map.get_mut(&view.id()).ok_or_else(|| stale(view))?;
        let all: &mut [u8] = bytemuck::cast_slice_mut(slab);
        let range = span(view, all.len())?;
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
            let all: &mut [u8] = bytemuck::cast_slice_mut(slab);
            all.copy_within(from, to.start);
            return Ok(());
        }
        // The destination slab is taken out of the map for the copy so the
        // source can be borrowed beside it, without staging through a `Vec`.
        let mut dst_slab = self.map.remove(&dst.id()).ok_or_else(|| stale(dst))?;
        if let Some(src_slab) = self.map.get(&src.id()) {
            let source: &[u8] = bytemuck::cast_slice(src_slab);
            let target: &mut [u8] = bytemuck::cast_slice_mut(&mut dst_slab);
            target[to].copy_from_slice(&source[from]);
        }
        self.map.insert(dst.id(), dst_slab);
        Ok(())
    }

    /// Size in bytes of the allocation `view` names.
    fn slab_bytes(&self, view: DeviceView<'_>) -> CoreResult<usize> {
        let slab = self.map.get(&view.id()).ok_or_else(|| stale(view))?;
        Ok(slab.len() * CHUNK_BYTES)
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

//! Moving a model's weights onto the device at the speed the host can read them.
//!
//! A weight import is a copy out of a file mapping. Handed to the driver as it
//! stands, that copy is *pageable*: the driver cannot DMA out of memory the
//! kernel may move, so it walks the source through a bounce buffer of its own,
//! faulting the mapping in one page at a time on the calling thread. Measured
//! on a GB10 over a mapping already in the page cache, that path moves
//! 0.9 GB/s, and a 62 GB checkpoint took four to six minutes to load.
//!
//! This module is the other path. A handful of page-locked slots, each with a
//! stream of its own: a slot is filled from the mapping by several threads at
//! once - the page faults are the cost, and they parallelise - and its copy is
//! queued without a wait, so the next slot fills while the device drains this
//! one. The same measurement moves 16 to 35 GB/s, and a cold read is bound by
//! the disk (5.2 GB/s here) rather than by the copy.
//!
//! The slots are kept between imports, because a checkpoint is a thousand of
//! them and pinning memory is itself slow, and released by
//! [`shiso_core::Backend::finish_imports`] once the load is over: they are a
//! quarter of a gigabyte of host memory nothing reads afterwards.

use std::thread;

use shiso_core::CoreResult;

use crate::device::OwnedStream;
use crate::sys::nvidia;
use crate::sys::{DevicePtr, PinnedPtr, SysResult, host_error};

/// Imports shorter than this take the plain synchronous copy.
///
/// Below a mebibyte the copy is a few page faults and one transfer either way,
/// and the norms, biases and routers a checkpoint is mostly made of by count
/// are all far below it.
pub(crate) const STAGED_MIN: usize = 1 << 20;

/// Bytes one slot holds.
const SLOT_BYTES: usize = 64 << 20;

/// Slots in flight. Four keeps the device busy while one fills, measured.
const SLOTS: usize = 4;

/// Page-locked host bytes the staging holds while it is open.
pub(crate) const STAGING_BYTES: u64 = (SLOTS * SLOT_BYTES) as u64;

/// Threads that fill one slot.
///
/// The copy is page-fault bound, not bandwidth bound, so it scales with
/// threads until the faults contend on the mapping; eight is where the
/// measurement stopped improving.
const COPY_THREADS: usize = 8;

/// The smallest piece one fill thread is given, so a short tail is not split
/// into slivers that cost more to spawn than to copy.
const MIN_PIECE: usize = 1 << 20;

/// One page-locked slot and the stream its copies run on.
#[derive(Debug)]
struct Slot {
    host: PinnedPtr,
    stream: OwnedStream,
}

impl Slot {
    /// Pins one slot and opens its stream.
    fn open() -> SysResult<Self> {
        let host = nvidia::mem_host_alloc(SLOT_BYTES as u64)?;
        match OwnedStream::open() {
            Ok(stream) => Ok(Self { host, stream }),
            Err(error) => {
                // SAFETY: the allocation was made just above, nothing has
                // queued a copy out of it, and it is freed exactly once here.
                if let Err(free) = unsafe { nvidia::mem_host_free(host) } {
                    tracing::error!(error = %free, "pinned slot release failed");
                }
                Err(error)
            }
        }
    }

    /// Waits for every copy queued out of this slot.
    fn wait(&self) -> SysResult<()> {
        // SAFETY: the stream is owned by this slot and live for as long as it is.
        unsafe { nvidia::stream_synchronize(self.stream.get()) }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        // The pinned memory may still be the source of a queued copy, and
        // freeing it under one is a device read of freed host memory.
        if let Err(error) = self.wait()
            && !error.is_teardown_noop()
        {
            tracing::error!(error = %error, "pinned slot wait failed");
        }
        // SAFETY: the wait above drained the only stream that reads this
        // allocation, which was made by `Slot::open` and is freed only here.
        if let Err(error) = unsafe { nvidia::mem_host_free(self.host) }
            && !error.is_teardown_noop()
        {
            tracing::error!(error = %error, "pinned slot release failed");
        }
    }
}

/// Where one load's import time went, logged when the staging is released.
#[derive(Debug, Default)]
pub(crate) struct ImportTally {
    /// Imports, bytes and seconds through the staging.
    pub(crate) staged: (u64, u64, f64),
    /// Imports, bytes and seconds through the plain copy.
    pub(crate) plain: (u64, u64, f64),
    /// Seconds spent reserving device memory for imports.
    pub(crate) alloc_seconds: f64,
}

impl ImportTally {
    /// Adds one import to a `(count, bytes, seconds)` bucket.
    pub(crate) fn add(bucket: &mut (u64, u64, f64), bytes: usize, seconds: f64) {
        bucket.0 += 1;
        bucket.1 += bytes as u64;
        bucket.2 += seconds;
    }

    /// Logs the tally, if anything was imported.
    #[expect(
        clippy::cast_precision_loss,
        reason = "gigabytes for a log line; a mantissa's worth of bytes is not the point"
    )]
    pub(crate) fn report(&self) {
        if self.staged.0 + self.plain.0 == 0 {
            return;
        }
        let gb = |bytes: u64| bytes as f64 / 1e9;
        tracing::info!(
            staged = self.staged.0,
            staged_gb = gb(self.staged.1),
            staged_s = self.staged.2,
            plain = self.plain.0,
            plain_gb = gb(self.plain.1),
            plain_s = self.plain.2,
            alloc_s = self.alloc_seconds,
            "weight imports finished"
        );
    }
}

/// The slots one backend imports through.
#[derive(Debug)]
pub(crate) struct Staging {
    slots: Vec<Slot>,
}

impl Staging {
    /// Pins every slot.
    ///
    /// # Errors
    ///
    /// Whatever the driver reports for an allocation or a stream it refused.
    pub(crate) fn open() -> CoreResult<Self> {
        let slots = (0..SLOTS)
            .map(|_| Slot::open())
            .collect::<SysResult<Vec<_>>>()?;
        Ok(Self { slots })
    }

    /// Copies `src` to `dst`, returning once all of it has landed.
    ///
    /// The wait at the end is what lets the caller hand the buffer to the
    /// work stream: these copies run on the slots' own streams, which nothing
    /// else orders against.
    ///
    /// # Errors
    ///
    /// Whatever the driver reports for a queued copy or a wait. Every slot is
    /// drained before an error is returned, so the destination may be freed.
    pub(crate) fn upload(&mut self, dst: DevicePtr, src: &[u8]) -> CoreResult<()> {
        // Readahead for the whole tensor, issued once: the fill threads then
        // mostly find their pages already on their way in from the disk.
        nvidia::advise_willneed(src);
        let queued = self.queue(dst, src);
        let drained = self.slots.iter().try_for_each(Slot::wait);
        queued?;
        drained?;
        Ok(())
    }

    /// Fills and queues every chunk of `src`, reusing a slot only once its
    /// previous copy is done.
    fn queue(&mut self, dst: DevicePtr, src: &[u8]) -> CoreResult<()> {
        for (k, chunk) in src.chunks(SLOT_BYTES).enumerate() {
            let slot = self
                .slots
                .get(k % self.slots.len())
                .ok_or_else(|| host_error("import_raw", "the staging has no slots"))?;
            slot.wait()?;
            let base = slot.host.raw() as *mut u8;
            // SAFETY: the slot's allocation is `SLOT_BYTES` long and `chunk` is
            // at most that; the wait above proved no queued copy still reads
            // it, and this method holds the staging by `&mut`, so this borrow
            // is the only reference to that memory while it lives.
            let host = unsafe { std::slice::from_raw_parts_mut(base, chunk.len()) };
            fill(host, chunk);
            let at = dst
                .offset((k * SLOT_BYTES) as u64)
                .ok_or_else(|| host_error("import_raw", "the destination address overflowed"))?;
            // SAFETY: the caller's allocation holds `src.len()` bytes from
            // `dst`, so this chunk's range lies inside it; the slot holds the
            // chunk, and `upload` drains the stream before either can be freed
            // or the slot refilled.
            unsafe { nvidia::memcpy_htod_pinned(at, slot.host, chunk.len(), slot.stream.get()) }?;
        }
        Ok(())
    }
}

/// Copies `src` into `dst` on several threads at once.
fn fill(dst: &mut [u8], src: &[u8]) {
    let piece = src.len().div_ceil(COPY_THREADS).max(MIN_PIECE);
    if piece >= src.len() {
        dst.copy_from_slice(src);
        return;
    }
    thread::scope(|scope| {
        for (to, from) in dst.chunks_mut(piece).zip(src.chunks(piece)) {
            scope.spawn(move || to.copy_from_slice(from));
        }
    });
}

//! The device context, its streams, and the table that owns every allocation.
//!
//! # One context, two streams
//!
//! The backend retains the device's *primary* context rather than creating one
//! of its own, so that anything else in the process that touches the same
//! device shares it instead of fighting over a second one. The ordinal is fixed
//! at the first open and a later open with a different ordinal is refused: two
//! contexts on two devices would silently make one backend's buffers
//! unreachable from the other's launches, and refusing is how that turns into a
//! message instead of a wrong answer.
//!
//! Two streams exist. Ordinary work runs on the work stream, which is what
//! gives the [`shiso_core::Backend`] trait its promise that a copy issued before a
//! launch is ordered before it. Graph capture runs on its own stream, because a
//! capture swallows everything submitted to its stream and the work stream must
//! stay usable for the transfers a capture is not allowed to contain.
//!
//! # The buffer table
//!
//! A caller never sees a device address. It sees a [`RawBuffer`] whose id is a
//! key into this table, and the table is the only thing that can turn that key
//! plus an offset back into an address — after proving the range fits. A stale
//! id is then a lookup miss rather than a read of whatever now lives at that
//! address, which is the single largest difference between this and a raw
//! handle API.

use std::collections::HashMap;
use std::num::NonZeroU64;

use shiso_core::MemTag;

use parking_lot::Mutex;
use shiso_core::{CoreError, CoreResult, DeviceView, RawBuffer};

use crate::CudaBackend;
use crate::sys::nvidia;
use crate::sys::{
    CuContext, CuDevice, CuEvent, CuStream, DevicePtr, SysError, SysResult, host_error,
};

/// The ordinal this process bound to, once anything has bound one.
///
/// Process-scoped rather than per-backend because the resource it guards is
/// process-scoped: the primary context belongs to the device, not to whoever
/// asked for it first.
static BOUND_ORDINAL: Mutex<Option<u32>> = Mutex::new(None);

/// Decides whether `ordinal` may be opened, given what is already bound.
///
/// Split out from the opening itself so the rule can be tested without a
/// driver: it is a rule about this process, not about any device.
fn bind_ordinal(bound: &mut Option<u32>, ordinal: u32) -> CoreResult<()> {
    match *bound {
        None => {
            *bound = Some(ordinal);
            Ok(())
        }
        Some(already) if already == ordinal => Ok(()),
        Some(already) => Err(CoreError::Unsupported {
            what: "a second CUDA device in one process",
            backend: crate::sys::BACKEND,
        })
        .inspect_err(|_| {
            tracing::error!(
                bound = already,
                requested = ordinal,
                "refusing to open a second CUDA device ordinal",
            );
        }),
    }
}

/// A stream that closes itself.
#[derive(Debug)]
pub(crate) struct OwnedStream(CuStream);

impl OwnedStream {
    /// Opens a stream.
    pub(crate) fn open() -> SysResult<Self> {
        nvidia::stream_create().map(Self)
    }

    /// The handle, for a call that is about to use it.
    pub(crate) const fn get(&self) -> CuStream {
        self.0
    }
}

impl Drop for OwnedStream {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the handle and it has not been
        // destroyed, because destruction happens here and only here.
        if let Err(err) = unsafe { nvidia::stream_destroy(self.0) }
            && !err.is_teardown_noop()
        {
            tracing::error!(error = %err, "stream destroy failed");
        }
    }
}

/// One event, closed on drop.
#[derive(Debug)]
pub(crate) struct OwnedEvent(CuEvent);

impl OwnedEvent {
    /// Opens an event.
    pub(crate) fn open() -> SysResult<Self> {
        nvidia::event_create().map(Self)
    }

    /// The handle, for a call that is about to use it.
    pub(crate) const fn get(&self) -> CuEvent {
        self.0
    }
}

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the handle and it has not been
        // destroyed, because destruction happens here and only here.
        if let Err(err) = unsafe { nvidia::event_destroy(self.0) }
            && !err.is_teardown_noop()
        {
            tracing::error!(error = %err, "event destroy failed");
        }
    }
}

/// One retain of a device's primary context, released on drop.
#[derive(Debug)]
struct OwnedContext {
    device: CuDevice,
    ctx: CuContext,
}

impl OwnedContext {
    /// Retains the primary context of `device`.
    fn retain(device: CuDevice) -> SysResult<Self> {
        let ctx = nvidia::primary_ctx_retain(device)?;
        Ok(Self { device, ctx })
    }
}

impl Drop for OwnedContext {
    fn drop(&mut self) {
        // SAFETY: this value owns exactly one retain, taken in `retain` and
        // released only here.
        if let Err(err) = unsafe { nvidia::primary_ctx_release(self.device) }
            && !err.is_teardown_noop()
        {
            tracing::error!(error = %err, "primary context release failed");
        }
    }
}

/// The device, its context and its streams.
#[derive(Debug)]
pub(crate) struct DeviceCtx {
    /// Dropped last, after both streams: a stream outliving its context is a
    /// use-after-free, and field order is what enforces that.
    work: OwnedStream,
    capture: OwnedStream,
    /// The branch streams a fork runs on, one beside each of the two above.
    side_work: OwnedStream,
    side_capture: OwnedStream,
    /// Where the main stream was at the fork, and the branch at its end.
    fork_event: OwnedEvent,
    join_event: OwnedEvent,
    ctx: OwnedContext,
    multiprocessors: u32,
    /// Whether the device shares its memory with the host; see
    /// [`memory_info`].
    pub(crate) integrated: bool,
}

impl DeviceCtx {
    /// Initialises the driver, binds `ordinal` and opens both streams.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when this process already bound a different
    /// ordinal, and [`CoreError::Backend`] for any driver failure — including
    /// the one every driverless host gets, where the initialisation itself is
    /// refused.
    pub(crate) fn open(ordinal: u32) -> CoreResult<Self> {
        // `cuInit` panics inside cudarc when libcuda is absent, so the library
        // is probed first and a driverless host gets an `Err`.
        if !nvidia::library_present() {
            return Err(SysError::no_driver("libcuda").into());
        }
        nvidia::init()?;
        bind_ordinal(&mut BOUND_ORDINAL.lock(), ordinal)?;
        let device = nvidia::device_get(ordinal)?;
        let ctx = OwnedContext::retain(device)?;
        // SAFETY: the retain above is live for as long as `ctx` is, and `ctx`
        // outlives this call.
        unsafe { nvidia::ctx_set_current(ctx.ctx) }?;
        let multiprocessors = nvidia::multiprocessor_count(device)?;
        let integrated = nvidia::integrated(device)?;
        let work = OwnedStream::open()?;
        let capture = OwnedStream::open()?;
        let side_work = OwnedStream::open()?;
        let side_capture = OwnedStream::open()?;
        let fork_event = OwnedEvent::open()?;
        let join_event = OwnedEvent::open()?;
        tracing::debug!(ordinal, multiprocessors, integrated, "opened CUDA device");
        Ok(Self {
            work,
            capture,
            side_work,
            side_capture,
            fork_event,
            join_event,
            ctx,
            multiprocessors,
            integrated,
        })
    }

    /// Makes this backend's context current on the calling thread.
    ///
    /// A context is a per-thread binding, and a backend is shared across every
    /// thread that holds its `Arc`, so each entry point binds before it does
    /// anything. The call is a driver-local assignment, not a device round
    /// trip, which is what makes paying it per entry point cheaper than the
    /// class of bug it removes: a launch from an unbound thread does not fail
    /// politely, it fails as an invalid context somewhere later.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the context cannot be made current.
    pub(crate) fn bind(&self) -> CoreResult<()> {
        // SAFETY: `self.ctx` holds a live retain for as long as `self` exists,
        // and `self` is borrowed for this call.
        unsafe { nvidia::ctx_set_current(self.ctx.ctx) }?;
        Ok(())
    }

    /// The stream ordinary work runs on.
    pub(crate) const fn work_stream(&self) -> CuStream {
        self.work.get()
    }

    /// The stream a graph capture runs on.
    pub(crate) const fn capture_stream(&self) -> CuStream {
        self.capture.get()
    }

    /// The branch stream beside the work stream.
    pub(crate) const fn side_work_stream(&self) -> CuStream {
        self.side_work.get()
    }

    /// The branch stream beside the capture stream.
    pub(crate) const fn side_capture_stream(&self) -> CuStream {
        self.side_capture.get()
    }

    /// The event a fork records on the main stream.
    pub(crate) const fn fork_event(&self) -> CuEvent {
        self.fork_event.get()
    }

    /// The event the end of a branch records on its stream.
    pub(crate) const fn join_event(&self) -> CuEvent {
        self.join_event.get()
    }

    /// Multiprocessors on this device, for sizing a cooperative grid.
    pub(crate) const fn multiprocessors(&self) -> u32 {
        self.multiprocessors
    }
}

/// One live allocation.
#[derive(Debug, Clone, Copy)]
struct Entry {
    ptr: DevicePtr,
    /// Addressable bytes. This is what the caller asked for, which may be less
    /// than what was reserved; see [`BufferTable::insert`].
    bytes: u64,
}

/// The table of live allocations, keyed by the id a [`RawBuffer`] carries.
#[derive(Debug)]
pub(crate) struct BufferTable {
    /// Ids are handed out from here and never reused while the table lives, so
    /// a stale id stays a miss rather than becoming someone else's buffer.
    next_id: NonZeroU64,
    live: HashMap<NonZeroU64, Entry>,
}

impl Default for BufferTable {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferTable {
    /// An empty table.
    pub(crate) fn new() -> Self {
        Self {
            next_id: NonZeroU64::MIN,
            live: HashMap::new(),
        }
    }

    /// Registers an allocation and returns the handle that names it.
    ///
    /// `bytes` is the addressable length the caller asked for, which may be
    /// smaller than what the driver actually reserved: a zero-byte request is
    /// still given a real address, because a zero-length view has to resolve to
    /// *something* and an address of zero is the one value a kernel must never
    /// receive.
    pub(crate) fn insert(&mut self, ptr: DevicePtr, bytes: u64) -> RawBuffer {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).unwrap_or(NonZeroU64::MIN);
        self.live.insert(id, Entry { ptr, bytes });
        RawBuffer::new(id, bytes)
    }

    /// Removes an allocation, yielding its address if it was live.
    pub(crate) fn remove(&mut self, id: NonZeroU64) -> Option<DevicePtr> {
        self.live.remove(&id).map(|entry| entry.ptr)
    }

    /// Turns a borrowed range into the address its first byte lives at.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the id names no live allocation, and
    /// [`CoreError::OutOfBounds`] when the range runs past the end of one. The
    /// range is never clamped: a kernel that wrote part of what it was asked to
    /// is worse than one that did not run.
    pub(crate) fn resolve(&self, view: DeviceView<'_>) -> CoreResult<DevicePtr> {
        let entry = self.live.get(&view.id()).ok_or_else(|| {
            host_error(
                "buffer",
                format!("handle {} is not a live allocation", view.id()),
            )
        })?;
        let fits = view
            .offset()
            .checked_add(view.len())
            .is_some_and(|end| end <= entry.bytes);
        if !fits {
            return Err(CoreError::OutOfBounds {
                offset: view.offset(),
                len: view.len(),
                buffer_bytes: entry.bytes,
            });
        }
        entry.ptr.offset(view.offset()).ok_or_else(|| {
            host_error(
                "buffer",
                "the view's offset would carry the address past the end of the address space",
            )
        })
    }

    /// Number of live allocations.
    pub(crate) fn len(&self) -> usize {
        self.live.len()
    }

    /// Empties the table, yielding every address that was still live.
    pub(crate) fn drain(&mut self) -> Vec<DevicePtr> {
        self.live.drain().map(|(_, entry)| entry.ptr).collect()
    }
}

/// The device's free and total memory, from its own context.
///
/// # Errors
///
/// [`CoreError::Backend`] when the context cannot be made current or the
/// driver refuses the query.
pub(crate) fn memory_info(backend: &CudaBackend) -> CoreResult<shiso_core::MemoryInfo> {
    backend.device.bind()?;
    let (free, total) = nvidia::mem_get_info()?;
    // On a device that shares memory with the host, the driver counts the
    // page cache as taken; measured on a GB10 right after a load, it reported
    // 113 of 122 GiB in use while 33 GiB of that was file cache the kernel
    // reclaims on demand. The kernel's own estimate of what an allocation can
    // still get is the better answer there, and the larger of the two is kept
    // so a host that cannot say never makes the figure worse.
    let free = if backend.device.integrated {
        free.max(host_available().unwrap_or(0).min(total))
    } else {
        free
    };
    Ok(shiso_core::MemoryInfo { free, total })
}

/// The kernel's estimate of memory available to a new allocation without
/// swapping (`MemAvailable`), in bytes; `None` where it cannot be read.
fn host_available() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

/// Reserves device memory and records it in the table.
///
/// # Errors
///
/// [`CoreError::Backend`] when a capture is running — see below — or when the
/// device refuses the reservation.
pub(crate) fn alloc(backend: &CudaBackend, bytes: u64) -> CoreResult<RawBuffer> {
    backend.device.bind()?;
    let mut state = backend.state.lock();
    // A graph bakes the addresses it was captured with into itself and nothing
    // updates them afterwards. Allocating mid-capture is how a buffer that the
    // captured graph will name ends up at an address the *next* run does not
    // have, so it is refused outright rather than left to produce a graph that
    // replays into freed memory.
    if state.graphs.is_capturing() {
        return Err(host_error(
            "alloc_raw",
            "device memory cannot be allocated while a graph capture is running",
        ));
    }
    // A zero-byte request still gets a real address; see `BufferTable::insert`.
    let ptr = nvidia::mem_alloc(bytes.max(1))?;
    Ok(state.buffers.insert(ptr, bytes))
}

/// Bytes one write of the zero run covers.
///
/// A `static` rather than a `const` so the run is one array in the binary
/// instead of a copy in every function that names it, and a mebibyte so that
/// clearing a slab of tens of them is tens of transfers rather than thousands.
static ZERO_RUN: [u8; 1 << 20] = [0; 1 << 20];

/// Writes zeros over a whole allocation.
///
/// The driver has a fill primitive and this build has no binding for one, so
/// the zeros are written through the ordinary host copy. That is why the run is
/// a mebibyte: the cost is the number of transfers, and a reservation happens
/// once or twice in a process rather than once a pass.
///
/// # Errors
///
/// Whatever [`copy_h2d`] reports.
pub(crate) fn clear(backend: &CudaBackend, raw: RawBuffer) -> CoreResult<()> {
    let run = u64::try_from(ZERO_RUN.len()).unwrap_or(u64::MAX);
    let mut at = 0;
    while at < raw.bytes() {
        let len = raw.bytes().saturating_sub(at).min(run);
        let width = usize::try_from(len).unwrap_or(ZERO_RUN.len());
        copy_h2d(
            backend,
            DeviceView::from_raw(raw, at, len),
            &ZERO_RUN[..width],
        )?;
        at = at.saturating_add(len);
    }
    Ok(())
}

/// Releases an allocation.
///
/// An id the table does not know is not an error. This runs from the owning
/// buffer's `Drop`, which cannot retry and cannot report, and the only way to
/// reach it twice is a handle that was already released — in which case the
/// memory is already gone and there is nothing left to do.
///
/// # Errors
///
/// [`CoreError::Backend`] when the driver refuses the free.
pub(crate) fn free(backend: &CudaBackend, raw: RawBuffer) -> CoreResult<()> {
    let Some(ptr) = backend.state.lock().buffers.remove(raw.id()) else {
        tracing::debug!(id = raw.id().get(), "CUDA handle was already freed");
        return Ok(());
    };
    backend.device.bind()?;
    // SAFETY: the address came out of the table, which holds each address
    // exactly once and has just given up its only copy, so this is the first
    // and last free of that allocation.
    unsafe { nvidia::mem_free(ptr) }?;
    Ok(())
}

/// Copies host bytes into a device range, waiting for the copy.
///
/// # Errors
///
/// [`CoreError::OutOfBounds`] when the lengths disagree or the range runs past
/// the allocation, and [`CoreError::Backend`] when the transfer fails.
pub(crate) fn copy_h2d(backend: &CudaBackend, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
    if src.len() as u64 != dst.len() {
        return Err(CoreError::OutOfBounds {
            offset: dst.offset(),
            len: src.len() as u64,
            buffer_bytes: dst.len(),
        });
    }
    backend.device.bind()?;
    let state = backend.state.lock();
    let ptr = state.buffers.resolve(dst)?;
    // SAFETY: `resolve` proved the range lies inside a live allocation and the
    // length check above proved `src` is exactly that long. The lock is still
    // held, so nothing can free the allocation under the copy.
    unsafe { nvidia::memcpy_htod(ptr, src, backend.device.work_stream()) }?;
    Ok(())
}

/// Allocates device memory and fills it with `host[range]`.
///
/// Large ranges go through [`crate::upload::Staging`]; see that module for why
/// the plain copy is not good enough for a checkpoint. The staging is opened on
/// the first large import and kept until
/// [`shiso_core::Backend::finish_imports`].
///
/// # Errors
///
/// [`CoreError::OutOfBounds`] when `range` runs past `host`, and whatever the
/// allocation or the copy reports. A failed copy frees the allocation.
pub(crate) fn import(
    backend: &CudaBackend,
    host: &[u8],
    range: std::ops::Range<usize>,
) -> CoreResult<RawBuffer> {
    let src = host.get(range.clone()).ok_or(CoreError::OutOfBounds {
        offset: range.start as u64,
        len: range.end.saturating_sub(range.start) as u64,
        buffer_bytes: host.len() as u64,
    })?;
    let len = src.len() as u64;
    let started = std::time::Instant::now();
    let raw = alloc(backend, len)?;
    let allocated = started.elapsed().as_secs_f64();
    let view = DeviceView::from_raw(raw, 0, len);
    let staged = src.len() >= crate::upload::STAGED_MIN;
    let copied = if staged {
        staged_copy(backend, view, src)
    } else {
        copy_h2d(backend, view, src)
    };
    {
        let mut tally = backend.imports.lock();
        tally.alloc_seconds += allocated;
        let seconds = started.elapsed().as_secs_f64() - allocated;
        let bucket = if staged {
            &mut tally.staged
        } else {
            &mut tally.plain
        };
        crate::upload::ImportTally::add(bucket, src.len(), seconds);
    }
    if let Err(error) = copied {
        if let Err(free_error) = free(backend, raw) {
            tracing::error!(error = %free_error, "freeing a failed import");
        }
        return Err(error);
    }
    Ok(raw)
}

/// Copies `src` into a freshly allocated range through the staging slots.
fn staged_copy(backend: &CudaBackend, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
    backend.device.bind()?;
    // Resolved and released before the staging lock is taken: the address
    // cannot move, and the allocation belongs to the import that made it until
    // it returns, so nothing can free it under the copy.
    let ptr = backend.state.lock().buffers.resolve(dst)?;
    let mut staging = backend.staging.lock();
    if staging.is_none() {
        *staging = Some(crate::upload::Staging::open()?);
        backend
            .ledger
            .record(MemTag::UploadStaging, crate::upload::STAGING_BYTES);
    }
    staging
        .as_mut()
        .expect("the staging was opened just above")
        .upload(ptr, src)
}

/// Writes zeros over a device range.
///
/// On whichever stream work is going to right now, which is the point: this
/// runs inside a captured island, and a fill sent to the work stream while a
/// capture is open executes once and is never recorded - so every replay of
/// that island skips it. That is not a failure anywhere. It is the histogram
/// this fill exists to zero, accumulating from the second decoded token on.
///
/// The two host-touching transfers beside this one stay on the work stream
/// because they drain it, and a drain is not something a capture can hold.
///
/// # Errors
///
/// [`CoreError::Backend`] when the fill fails, and whatever the buffer table
/// reports for a range that is not inside a live allocation.
pub(crate) fn fill_zero(backend: &CudaBackend, dst: DeviceView<'_>) -> CoreResult<()> {
    if dst.is_empty() {
        return Ok(());
    }
    backend.device.bind()?;
    let state = backend.state.lock();
    let ptr = state.buffers.resolve(dst)?;
    let len = usize::try_from(dst.len()).unwrap_or(usize::MAX);
    let stream = state.graphs.stream(&backend.device);
    // SAFETY: `resolve` proved the range lies inside a live allocation, and the
    // lock is still held so nothing can free it under the fill.
    unsafe { nvidia::memset_zero(ptr, len, stream) }?;
    Ok(())
}

/// Writes the address of each range in `srcs` into `dst`, one eight-byte slot
/// each.
///
/// The whole point of the kernels this feeds is that they dereference the slots
/// themselves: one page table per row of a batch, one recurrent state per row.
/// So the array has to hold real addresses rather than handles, and that is why
/// this lives here and not above - the buffer table is the only thing in the
/// process that can turn a view into an address, and nothing that calls this
/// ever sees one.
///
/// The resolve and the transfer happen under one lock. Taking the table lock
/// twice would leave a window in which an allocation this call had already
/// resolved could be freed, and what would then reach the device is an address
/// that is no longer anybody's.
///
/// # Errors
///
/// [`CoreError::OutOfBounds`] when `dst` is shorter than the slots `srcs`
/// needs, and [`CoreError::Backend`] when a range names no live allocation or
/// the transfer fails.
pub(crate) fn pack_pointers(
    backend: &CudaBackend,
    dst: DeviceView<'_>,
    srcs: &[DeviceView<'_>],
) -> CoreResult<()> {
    let want = (srcs.len() as u64).saturating_mul(shiso_core::DEVICE_POINTER_BYTES);
    if want > dst.len() {
        return Err(CoreError::OutOfBounds {
            offset: dst.offset(),
            len: want,
            buffer_bytes: dst.len(),
        });
    }
    backend.device.bind()?;
    let state = backend.state.lock();
    let mut bytes = Vec::with_capacity(srcs.len() * 8);
    for src in srcs {
        bytes.extend_from_slice(&state.buffers.resolve(*src)?.raw().to_ne_bytes());
    }
    let ptr = state.buffers.resolve(dst.slice(0, want)?)?;
    // SAFETY: `resolve` proved the destination range lies inside a live
    // allocation and is exactly `want` bytes, which is the length of `bytes`.
    // The lock is still held, so nothing can free the allocation under the copy.
    unsafe { nvidia::memcpy_htod(ptr, &bytes, backend.device.work_stream()) }?;
    Ok(())
}

/// Reads a device range back into host bytes, waiting for the copy.
///
/// # Errors
///
/// [`CoreError::OutOfBounds`] when the lengths disagree or the range runs past
/// the allocation, and [`CoreError::Backend`] when the transfer fails.
pub(crate) fn copy_d2h(
    backend: &CudaBackend,
    src: DeviceView<'_>,
    dst: &mut [u8],
) -> CoreResult<()> {
    if dst.len() as u64 != src.len() {
        return Err(CoreError::OutOfBounds {
            offset: src.offset(),
            len: dst.len() as u64,
            buffer_bytes: src.len(),
        });
    }
    backend.device.bind()?;
    let state = backend.state.lock();
    let ptr = state.buffers.resolve(src)?;
    // SAFETY: `resolve` proved the range lies inside a live allocation and the
    // length check above proved `dst` is exactly that long. The lock is still
    // held, so nothing can free the allocation under the copy.
    unsafe { nvidia::memcpy_dtoh(dst, ptr, backend.device.work_stream()) }?;
    Ok(())
}

/// Copies one device range into another.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] when the two ranges differ in length,
/// [`CoreError::OutOfBounds`] when either runs past its allocation, and
/// [`CoreError::Backend`] when the copy cannot be queued.
pub(crate) fn copy_d2d(
    backend: &CudaBackend,
    dst: DeviceView<'_>,
    src: DeviceView<'_>,
) -> CoreResult<()> {
    if dst.len() != src.len() {
        return Err(CoreError::ShapeMismatch {
            field: "copy_d2d len",
            expected: src.len(),
            got: dst.len(),
        });
    }
    backend.device.bind()?;
    let state = backend.state.lock();
    let src_ptr = state.buffers.resolve(src)?;
    let dst_ptr = state.buffers.resolve(dst)?;
    let stream = state.graphs.stream(&backend.device);
    // SAFETY: both ranges were proved to lie inside live allocations and are
    // equal in length; the lock is held for the duration, so neither can be
    // freed under the copy.
    unsafe { nvidia::memcpy_dtod(dst_ptr, src_ptr, src.len(), stream) }?;
    Ok(())
}

/// Blocks until every operation queued on the work stream has finished.
///
/// # Errors
///
/// [`CoreError::Backend`] when the stream reports a failure while draining.
/// A fault raised by a kernel launched earlier surfaces here, because a launch
/// is asynchronous and reports nothing itself.
pub(crate) fn sync(backend: &CudaBackend) -> CoreResult<()> {
    backend.device.bind()?;
    // SAFETY: the work stream is owned by `backend.device` and lives as long as
    // the borrow.
    unsafe { nvidia::stream_synchronize(backend.device.work_stream()) }?;
    Ok(())
}

/// Frees every allocation the table still holds, at teardown.
///
/// Reaching this with a non-empty table means a buffer outlived the backend,
/// which the ownership rules make impossible through the safe API — every
/// `DeviceBuffer` holds an `Arc` to the backend that made it. It is still freed
/// and still logged, because a leak that is only visible as growing device
/// memory is the worst kind.
pub(crate) fn release_all(table: &mut BufferTable) {
    let leaked = table.drain();
    if leaked.is_empty() {
        return;
    }
    tracing::error!(
        count = leaked.len(),
        "CUDA backend closed with live allocations; freeing them",
    );
    for ptr in leaked {
        // SAFETY: each address came out of the table, which held it exactly
        // once and has now given up its only copy.
        if let Err(err) = unsafe { nvidia::mem_free(ptr) }
            && !err.is_teardown_noop()
        {
            tracing::error!(error = %err, "device free failed during teardown");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use shiso_core::{Backend, CoreError};

    use super::{BufferTable, bind_ordinal};
    use crate::sys::DevicePtr;
    use crate::testing::TableBackend;

    #[test]
    fn a_view_resolves_to_its_allocation_base_plus_its_own_offset() {
        let backend = TableBackend::new();
        let handle: Arc<dyn Backend> = backend.clone();
        let buffer = shiso_core::alloc(&handle, 256).expect("allocation");
        let base = TableBackend::first_address();
        let whole = buffer.view();
        let tail = buffer.slice(64, 32).expect("sub-range");

        let table = backend.table();
        assert_eq!(
            table.resolve(whole).expect("resolve").raw(),
            base,
            "the whole-buffer view starts at the allocation's own address",
        );
        assert_eq!(
            table.resolve(tail).expect("resolve").raw(),
            base + 64,
            "a sub-range starts that many bytes further in",
        );
    }

    #[test]
    fn a_range_past_the_end_of_an_allocation_is_out_of_bounds_not_a_clamp() {
        // A view is always in bounds of the buffer it came from, so the way to
        // reach the check is the way it happens in practice: a handle that
        // outlived the allocation it named and now points at a shorter one.
        // Both tables hand out their first id, so the view names the short
        // entry.
        let backend = TableBackend::new();
        let handle: Arc<dyn Backend> = backend.clone();
        let buffer = shiso_core::alloc(&handle, 64).expect("allocation");
        let long = buffer.view();
        let mut shorter = BufferTable::new();
        let short = shorter.insert(DevicePtr::from_raw(0x2_0000), 16);
        assert_eq!(short.id(), long.id(), "the two tables agree on the id");

        match shorter.resolve(long) {
            Err(CoreError::OutOfBounds {
                offset,
                len,
                buffer_bytes,
            }) => {
                assert_eq!((offset, len, buffer_bytes), (0, 64, 16));
            }
            other => panic!("expected an out-of-bounds error, got {other:?}"),
        }
    }

    #[test]
    fn an_id_the_table_never_issued_is_a_backend_error_not_a_wrong_address() {
        let backend = TableBackend::new();
        let handle: Arc<dyn Backend> = backend.clone();
        let buffer = shiso_core::alloc(&handle, 32).expect("allocation");
        let view = buffer.view();
        let empty = BufferTable::new();
        match empty.resolve(view) {
            Err(CoreError::Backend { backend, op, .. }) => {
                assert_eq!(backend, "cuda");
                assert_eq!(op, "buffer");
            }
            other => panic!("expected a backend error, got {other:?}"),
        }
    }

    #[test]
    fn a_freed_id_is_never_handed_out_again() {
        let mut table = BufferTable::new();
        let first = table.insert(DevicePtr::from_raw(0x3_0000), 8);
        table.remove(first.id());
        let second = table.insert(DevicePtr::from_raw(0x4_0000), 8);
        assert_ne!(
            first.id(),
            second.id(),
            "reusing an id would turn a stale handle into someone else's buffer",
        );
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn removing_an_id_twice_reports_that_it_was_already_gone() {
        let mut table = BufferTable::new();
        let raw = table.insert(DevicePtr::from_raw(0x5_0000), 8);
        assert!(table.remove(raw.id()).is_some());
        assert!(
            table.remove(raw.id()).is_none(),
            "a second release must not produce a second address to free",
        );
    }

    #[test]
    fn the_first_ordinal_wins_and_a_second_one_is_refused() {
        let mut bound = None;
        bind_ordinal(&mut bound, 0).expect("first bind");
        assert_eq!(bound, Some(0));
        bind_ordinal(&mut bound, 0).expect("rebinding the same ordinal is fine");
        match bind_ordinal(&mut bound, 1) {
            Err(CoreError::Unsupported { backend, .. }) => assert_eq!(backend, "cuda"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert_eq!(bound, Some(0), "a refused bind must not move the binding");
    }

    #[test]
    fn a_zero_length_view_still_resolves_to_a_real_address() {
        let mut table = BufferTable::new();
        let raw = table.insert(DevicePtr::from_raw(0x6_0000), 0);
        assert_eq!(raw.bytes(), 0);
        assert_eq!(table.len(), 1);
        assert_eq!(
            table.remove(raw.id()).map(DevicePtr::raw),
            Some(0x6_0000),
            "a zero-byte request is still backed by an address a kernel can take",
        );
    }

    #[test]
    fn ids_start_at_one_so_no_handle_is_ever_a_null_shaped_value() {
        let mut table = BufferTable::new();
        let raw = table.insert(DevicePtr::from_raw(0x7_0000), 4);
        assert_eq!(raw.id(), NonZeroU64::MIN);
    }
}

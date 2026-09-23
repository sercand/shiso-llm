//! Device, buffer table and command batching.
//!
//! One [`MetalBackend`] owns the system device, one command queue, the compiled
//! kernel library and the buffer table. Everything mutable sits behind one
//! mutex, which is what makes the `Backend` promise "calls are internally
//! serialised and ordered on one implicit queue" literally true rather than a
//! claim about Metal.
//!
//! # Batching
//!
//! A forward pass is hundreds of dispatches, and committing a command buffer per
//! dispatch would measure submission overhead instead of kernels. So launches
//! accumulate into one open command buffer with one open compute encoder, and
//! exactly three things flush: [`Backend::sync`] and the two host-copy paths,
//! which must not touch memory the GPU is still writing.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions,
};
use parking_lot::{Mutex, MutexGuard};
use shiso_core::{
    Backend, CoreError, CoreResult, DeviceView, HostBytes, KernelRef, LaunchArgs, RawBuffer,
};

use crate::compile::compile_library;
use crate::pipeline::{PipelineCache, PipelineState};

/// A driver-level failure message.
///
/// `CoreError::Backend` wants a real `source`, so the `NSError` text is carried
/// as a typed value instead of being formatted into the error's own message.
#[derive(Debug)]
pub(crate) struct MetalError {
    message: String,
}

impl MetalError {
    /// Wraps a driver message.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Builds the `Backend` wrapper for one failing Metal operation.
    pub(crate) fn backend(op: &'static str, message: impl Into<String>) -> CoreError {
        CoreError::Backend {
            backend: "metal",
            op,
            source: Box::new(Self::new(message)),
        }
    }
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MetalError {}

impl From<String> for MetalError {
    fn from(message: String) -> Self {
        Self { message }
    }
}

/// One live device allocation.
///
/// `base` is the leading slack of an imported wrap: the wrap has to start on a
/// page boundary, so the logical start of the imported range sits `base` bytes
/// into the wrap. Owned allocations use `0`.
pub(crate) struct BufferEntry {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Leading slack of a page-aligned wrap; `0` for owned allocations.
    pub(crate) base: u64,
    /// Addressable bytes of the logical range.
    pub(crate) bytes: u64,
}

/// Driver handles have no `Debug`, so an entry prints the range it owns.
impl fmt::Debug for BufferEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferEntry")
            .field("base", &self.base)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

/// The command buffer currently being filled, if any.
#[derive(Default)]
enum Batch {
    /// Nothing is open; the next launch starts a new command buffer.
    #[default]
    Idle,
    /// A command buffer is open and may still receive encoders.
    Open {
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        encoder: Encoder,
    },
}

/// The encoder attached to the open command buffer.
#[derive(Default)]
enum Encoder {
    /// Every encoder on this command buffer has been closed.
    #[default]
    None,
    /// The compute encoder is open and accepting dispatches.
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    /// A blit encoder is open; it is closed before the next dispatch.
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
}

pub(crate) struct State {
    /// Handles are issued from here. `0` is not a valid `RawBuffer` id, so the
    /// counter starts at one and a freed id is never reused while visible.
    next_id: NonZeroU64,
    buffers: HashMap<NonZeroU64, BufferEntry>,
    batch: Batch,
    pipelines: PipelineCache,
    /// Command buffers handed to the queue, which is what the batching test
    /// counts: hundreds of dispatches must cost a handful of commits.
    commits: usize,
}

/// Driver handles have no `Debug`, so the state reports what it can.
impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("State")
            .field("next_id", &self.next_id)
            .field("buffers", &self.buffers.len())
            .field("pipelines", &self.pipelines.len())
            .field("commits", &self.commits)
            .finish_non_exhaustive()
    }
}

impl State {
    /// Registers `buffer` and returns the handle that names it.
    fn insert(
        &mut self,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        base: u64,
        bytes: u64,
    ) -> RawBuffer {
        let id = self.next_id;
        self.next_id = NonZeroU64::new(id.get() + 1).unwrap_or(id);
        self.buffers.insert(
            id,
            BufferEntry {
                buffer,
                base,
                bytes,
            },
        );
        RawBuffer::new(id, bytes)
    }

    /// Resolves a view to its entry and proves the range fits inside it.
    pub(crate) fn view(&self, view: DeviceView<'_>) -> CoreResult<&BufferEntry> {
        let entry = self.buffers.get(&view.id()).ok_or_else(|| {
            MetalError::backend(
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
        Ok(entry)
    }
}

/// The Metal compute backend.
///
/// One instance per process is the intended shape: it owns the device, the
/// queue and the compiled kernel library, and the buffer table plus the pipeline
/// cache only pay off when they are shared.
pub struct MetalBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    inner: Mutex<State>,
}

/// Probes for a Metal device without building a backend.
///
/// GPU tests skip when this is `false`, so the default test gate never needs a
/// GPU.
#[must_use]
pub fn metal_available() -> bool {
    MTLCreateSystemDefaultDevice().is_some()
}

impl MetalBackend {
    /// Opens the default device, opens a command queue and compiles the kernel
    /// library.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when there is no Metal device, the device refuses a
    /// command queue, or the kernel library does not compile; the compiler's own
    /// diagnostic is the error source in the latter case.
    pub fn new() -> CoreResult<Arc<Self>> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| MetalError::backend("new", "no default Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| MetalError::backend("new", "the device refused a command queue"))?;
        let library = compile_library(&device)?;
        tracing::debug!(
            device = %device.name(),
            max_buffer_bytes = device.maxBufferLength(),
            "opened Metal device",
        );
        Ok(Arc::new(Self {
            device,
            queue,
            library,
            inner: Mutex::new(State {
                next_id: NonZeroU64::MIN,
                buffers: HashMap::default(),
                batch: Batch::default(),
                pipelines: PipelineCache::default(),
                commits: 0,
            }),
        }))
    }

    /// Number of live device allocations.
    #[must_use]
    pub fn live_buffers(&self) -> usize {
        self.inner.lock().buffers.len()
    }

    /// Command buffers handed to the queue since the backend opened.
    ///
    /// A batching backend should commit a handful of these for a whole forward
    /// pass; a per-dispatch regression shows up here immediately.
    #[must_use]
    pub fn commits(&self) -> usize {
        self.inner.lock().commits
    }

    /// Number of pipelines the cache currently holds.
    #[must_use]
    pub fn pipelines(&self) -> usize {
        self.inner.lock().pipelines.len()
    }

    /// Resolves the pipeline for `kernel` and caches it, without dispatching.
    ///
    /// An engine calls this while it loads a model, so an entry point the library
    /// does not declare, or a device that refuses the pipeline, fails at load time
    /// rather than on the first token. The pipeline lands under the same key a
    /// launch uses, so preparing a kernel costs nothing later.
    ///
    /// A kernel whose entry point reads specialization constants has to be passed
    /// with its full `consts` table, exactly as it would be when launched.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the entry point is absent from the compiled
    /// library or the device refuses the pipeline.
    pub fn prepare(&self, kernel: KernelRef) -> CoreResult<()> {
        let mut state = self.inner.lock();
        self.resolve(&mut state, kernel).map(|_| ())
    }

    /// The one lock that serialises every call, as `Backend` requires.
    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.inner.lock()
    }

    /// The most threadgroup memory one dispatch may request on this device.
    ///
    /// `run::dispatch` checks a launch's `smem_bytes` against this before
    /// calling `setThreadgroupMemoryLength_atIndex`, which does not validate
    /// its argument itself: a length past this limit is undefined behaviour,
    /// not a reported error, so the check has to happen on this side.
    pub(crate) fn max_threadgroup_memory(&self) -> u64 {
        u64::try_from(self.device.maxThreadgroupMemoryLength()).unwrap_or(u64::MAX)
    }

    /// The pipeline for `kernel`, resolving and caching it on a miss.
    ///
    /// The specialization table is built from `kernel.consts` here, so the
    /// constants a pipeline is built with are always the ones in its key.
    ///
    /// A kernel that reads `function_constant`s is only complete once they are
    /// supplied, and Metal will not report it politely: building a pipeline from a
    /// function whose required constants are unset trips an internal assertion that
    /// aborts the process instead of returning an error. A `KernelRef` naming such
    /// a kernel therefore has to carry its whole `consts` table, which is also why
    /// a specialization is part of the cache key rather than a separate opt-in.
    /// Resolves and caches `kernel`, or returns the cached pipeline.
    pub(crate) fn resolve(
        &self,
        state: &mut State,
        kernel: KernelRef,
    ) -> CoreResult<PipelineState> {
        if let Some(pipeline) = state.pipelines.get(kernel) {
            return Ok(pipeline);
        }
        let constants = crate::run::constant_values(kernel.consts)?;
        state
            .pipelines
            .insert(&self.device, &self.library, kernel, constants.as_deref())
    }

    /// Ends the open encoder, commits the open command buffer and waits.
    ///
    /// A no-op when nothing is open, so every flush point can call it
    /// unconditionally.
    pub(crate) fn flush(state: &mut State, op: &'static str) -> CoreResult<()> {
        let Batch::Open {
            command_buffer,
            encoder,
        } = std::mem::take(&mut state.batch)
        else {
            return Ok(());
        };
        match encoder {
            Encoder::Compute(encoder) => encoder.endEncoding(),
            Encoder::Blit(encoder) => encoder.endEncoding(),
            Encoder::None => {}
        }
        state.commits += 1;
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let status = command_buffer.status();
        if status != MTLCommandBufferStatus::Completed {
            let detail = command_buffer.error().map_or_else(
                || format!("{status:?}"),
                |err| format!("{status:?}: {}", err.localizedDescription()),
            );
            return Err(MetalError::backend(op, detail));
        }
        Ok(())
    }

    /// The open command buffer, opening one when the slot is empty.
    fn command_buffer(
        &self,
        state: &mut State,
    ) -> CoreResult<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if matches!(state.batch, Batch::Idle) {
            let command_buffer = self.queue.commandBuffer().ok_or_else(|| {
                MetalError::backend("run", "the command queue refused a command buffer")
            })?;
            state.batch = Batch::Open {
                command_buffer,
                encoder: Encoder::None,
            };
        }
        match &state.batch {
            Batch::Open { command_buffer, .. } => Ok(command_buffer.clone()),
            Batch::Idle => Err(MetalError::backend("run", "no command buffer is open")),
        }
    }

    /// The open compute encoder, opening a command buffer or replacing a blit
    /// encoder as needed.
    pub(crate) fn compute_encoder(
        &self,
        state: &mut State,
    ) -> CoreResult<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        let command_buffer = self.command_buffer(state)?;
        let Batch::Open { encoder, .. } = &mut state.batch else {
            return Err(MetalError::backend("run", "no command buffer is open"));
        };
        if let Encoder::Blit(blit) = encoder {
            blit.endEncoding();
            *encoder = Encoder::None;
        }
        if let Encoder::Compute(compute) = encoder {
            return Ok(compute.clone());
        }
        let compute = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MetalError::backend("run", "the command buffer refused a compute encoder")
        })?;
        *encoder = Encoder::Compute(compute.clone());
        Ok(compute)
    }

    /// The open blit encoder, closing the compute encoder first.
    ///
    /// A blit cannot share an encoder slot with a compute encoder, so this is
    /// the one place a dispatch encoder is closed early. It stays on the same
    /// command buffer, which is what keeps the copy inside the same batch.
    pub(crate) fn blit_encoder(
        &self,
        state: &mut State,
    ) -> CoreResult<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>> {
        let command_buffer = self.command_buffer(state)?;
        let Batch::Open { encoder, .. } = &mut state.batch else {
            return Err(MetalError::backend("copy_d2d", "no command buffer is open"));
        };
        if let Encoder::Compute(compute) = encoder {
            compute.endEncoding();
            *encoder = Encoder::None;
        }
        if let Encoder::Blit(blit) = encoder {
            return Ok(blit.clone());
        }
        let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
            MetalError::backend("copy_d2d", "the command buffer refused a blit encoder")
        })?;
        *encoder = Encoder::Blit(blit.clone());
        Ok(blit)
    }

    /// The host address of one buffer's logical start.
    fn contents(entry: &BufferEntry) -> *mut u8 {
        let slack = usize::try_from(entry.base).unwrap_or(usize::MAX);
        // SAFETY: `contents` is the base of a live allocation and `base` is at
        // most its length, so the sum stays inside the same object. The entry
        // owns the `Retained` that keeps those bytes mapped, and the borrow of
        // `entry` cannot outlive it.
        unsafe { entry.buffer.contents().as_ptr().cast::<u8>().add(slack) }
    }
}

impl Backend for MetalBackend {
    fn backend_name(&self) -> &'static str {
        "metal"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let length = usize::try_from(bytes).map_err(|_| {
            MetalError::backend("alloc_raw", format!("{bytes} bytes is unaddressable here"))
        })?;
        let buffer = self
            .device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                MetalError::backend(
                    "alloc_raw",
                    format!("the device refused a {bytes} byte allocation"),
                )
            })?;
        Ok(self.inner.lock().insert(buffer, 0, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        // Dropping the entry drops the last `Retained` reference, which is the
        // whole free. An absent id means the handle was already released, which
        // is idempotent rather than a failure: `Drop` cannot retry anyway.
        if self.inner.lock().buffers.remove(&raw.id()).is_none() {
            tracing::debug!(id = raw.id().get(), "Metal handle was already freed");
        }
        Ok(())
    }

    /// Wraps `host[range]` without copying when the range can be wrapped, and
    /// allocates plus copies when it cannot.
    ///
    /// The wrap has to start on a page boundary, so the logical start of the
    /// range sits `slack` bytes into the wrap and every access adds that slack
    /// back. The wrap is clamped to the mapping, so no byte outside `host` is
    /// ever addressable, and the `Arc` that [`shiso_core::import_host`] keeps
    /// inside the returned `DeviceBuffer` is what keeps the pages mapped.
    fn imports_copy(&self) -> bool {
        false
    }

    fn import_raw(&self, host: &dyn HostBytes, range: Range<usize>) -> CoreResult<RawBuffer> {
        let all: &[u8] = host.as_ref();
        if range.end > all.len() {
            return Err(CoreError::OutOfBounds {
                offset: range.start as u64,
                len: range.end.saturating_sub(range.start) as u64,
                buffer_bytes: all.len() as u64,
            });
        }
        let len = range.end - range.start;
        let page = page_size();
        let base = all.as_ptr() as usize;
        let start = base + range.start;
        let aligned = start & !(page - 1);
        let slack = start - aligned;
        let want = (slack + len + page - 1) & !(page - 1);
        let inside = aligned + want <= base + all.len();
        let fits = self.device.maxBufferLength();
        if inside && want <= fits {
            let pointer = ptr::NonNull::new(aligned as *mut c_void).ok_or_else(|| {
                MetalError::backend("import_raw", "the mapping is at a null address")
            })?;
            // SAFETY: `aligned` and `want` stay inside the mapping `host` owns,
            // and the `Arc` that `import_host` stores inside the returned
            // `DeviceBuffer` holds those pages mapped for exactly as long as
            // this handle can be dispatched against.
            let wrapped = unsafe {
                self.device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        pointer,
                        want,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
            };
            if let Some(buffer) = wrapped {
                let mut state = self.inner.lock();
                return Ok(state.insert(buffer, slack as u64, len as u64));
            }
            tracing::debug!(
                offset = range.start,
                len,
                "Metal refused the no-copy wrap; copying instead",
            );
        }
        self.copy_import(host, range)
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        let mut state = self.inner.lock();
        Self::flush(&mut state, "copy_h2d")?;
        if src.len() as u64 != dst.len() {
            return Err(CoreError::OutOfBounds {
                offset: dst.offset(),
                len: src.len() as u64,
                buffer_bytes: dst.len(),
            });
        }
        let entry = state.view(dst)?;
        let offset = usize::try_from(dst.offset()).unwrap_or(0);
        // SAFETY: `view` proved the range fits and the length check above
        // proved `src` is no longer than the view.
        unsafe {
            let target = Self::contents(entry).add(offset);
            ptr::copy_nonoverlapping(src.as_ptr(), target, src.len());
        }
        Ok(())
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        if dst.len() as u64 != src.len() {
            return Err(CoreError::OutOfBounds {
                offset: src.offset(),
                len: dst.len() as u64,
                buffer_bytes: src.len(),
            });
        }
        let mut state = self.inner.lock();
        Self::flush(&mut state, "copy_d2h")?;
        let entry = state.view(src)?;
        let offset = usize::try_from(src.offset()).unwrap_or(0);
        // SAFETY: bounds proven by `view` and by the length check above.
        unsafe {
            let source = Self::contents(entry).add(offset);
            ptr::copy_nonoverlapping(source, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_d2d(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()> {
        if dst.len() != src.len() {
            return Err(CoreError::ShapeMismatch {
                field: "copy_d2d len",
                expected: src.len(),
                got: dst.len(),
            });
        }
        let mut state = self.inner.lock();
        let src_entry = state.view(src)?;
        let dst_entry = state.view(dst)?;
        let (src_buffer, dst_buffer) = (src_entry.buffer.clone(), dst_entry.buffer.clone());
        let (src_base, dst_base) = (src_entry.base + src.offset(), dst_entry.base + dst.offset());
        let size = usize::try_from(src.len()).unwrap_or(usize::MAX);
        let src_offset = usize::try_from(src_base).unwrap_or(usize::MAX);
        let dst_offset = usize::try_from(dst_base).unwrap_or(usize::MAX);
        let blit = self.blit_encoder(&mut state)?;
        // SAFETY: both ranges were bounds-checked against their own entries and
        // the two lengths are equal.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buffer,
                src_offset,
                &dst_buffer,
                dst_offset,
                size,
            );
        }
        Ok(())
    }

    fn run(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()> {
        crate::run::dispatch(self, kernel, args)
    }

    fn max_block_threads(&self, kernel: KernelRef) -> Option<u32> {
        let mut state = self.inner.lock();
        // A kernel that will not build has no limit to report. The resolve
        // error is logged rather than swallowed: a variant dropping out of
        // the candidate set for a compile reason is worth seeing.
        let pipeline = match self.resolve(&mut state, kernel) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                tracing::debug!(
                    entry = kernel.entry,
                    error = %error.chain(),
                    "no block limit: the pipeline did not resolve",
                );
                return None;
            }
        };
        u32::try_from(pipeline.maxTotalThreadsPerThreadgroup()).ok()
    }

    fn sync(&self) -> CoreResult<()> {
        let mut state = self.inner.lock();
        Self::flush(&mut state, "sync")
    }
}

impl fmt::Debug for MetalBackend {
    /// Driver handles have no `Debug`, so the identity is the device name and
    /// the contents of the tables.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.inner.lock();
        f.debug_struct("MetalBackend")
            .field("device", &self.device.name())
            .field("live_buffers", &state.buffers.len())
            .field("pipelines", &state.pipelines.len())
            .field("commits", &state.commits)
            // The driver handles themselves have no Debug, so they are
            // named by what they hold rather than printed.
            .finish_non_exhaustive()
    }
}

impl Drop for MetalBackend {
    /// Drains the open command buffer so nothing is in flight behind it.
    ///
    /// A `Drop` cannot report failure, so it is logged: an unfinished dispatch
    /// against freed memory is exactly the kind of bug a trace should show.
    fn drop(&mut self) {
        let mut state = self.inner.lock();
        if let Err(err) = Self::flush(&mut state, "sync") {
            tracing::error!(error = %err, "Metal flush on drop failed");
        }
    }
}

impl MetalBackend {
    /// Allocates and fills `host[range]` the long way round.
    ///
    /// This is the fallback for a range Metal will not wrap: unaligned ends past
    /// the mapping, or a range wider than `maxBufferLength`. It cannot reuse
    /// [`Backend::copy_h2d`] because that takes a `DeviceView`, and a view can
    /// only be borrowed from a `DeviceBuffer` the caller owns.
    fn copy_import(&self, host: &dyn HostBytes, range: Range<usize>) -> CoreResult<RawBuffer> {
        let src = &host.as_ref()[range];
        let buffer = self
            .device
            .newBufferWithLength_options(src.len(), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                MetalError::backend(
                    "import_raw",
                    format!("the device refused a {} byte copy", src.len()),
                )
            })?;
        let mut state = self.inner.lock();
        let raw = state.insert(buffer, 0, src.len() as u64);
        let entry = state
            .buffers
            .get(&raw.id())
            .ok_or_else(|| MetalError::backend("import_raw", "the fresh allocation vanished"))?;
        // SAFETY: the allocation is `src.len()` bytes, `src` is exactly that
        // long, and no other thread has the handle yet.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), Self::contents(entry), src.len());
        }
        Ok(raw)
    }
}

/// Bytes in one host page, which is the alignment a no-copy wrap needs.
fn page_size() -> usize {
    // `sysconf` cannot fail for `_SC_PAGESIZE`; the guard only keeps a
    // hypothetical zero from turning the alignment arithmetic below into a
    // division by zero.
    // SAFETY: `sysconf` with `_SC_PAGESIZE` only reads a value and cannot fail;
    // libc marks it unsafe only because `sysconf` is not guaranteed
    // async-signal-safe in general.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page).unwrap_or(4096).max(1)
}

// SAFETY: `MetalBackend` is `Send` and `Sync` because of what the driver
// guarantees per type plus what the mutex guarantees about the rest.
//
// * `MTLDevice`, `MTLCommandQueue` and `MTLLibrary` are documented thread-safe,
//   and objc2-metal already declares their protocol objects `Send + Sync`.
// * `MTLBuffer` and `MTLComputePipelineState` are documented thread-safe for
//   use from any thread; objc2-metal does not mark the buffer protocol objects
//   `Send`, so the auto implementation stops at the buffer table and this impl
//   is what replaces it.
// * `MTLCommandBuffer` and the command encoders are *not* documented as safe to
//   share, and they are the only state here that is not immutable. They live in
//   `State`, which is reachable only through `inner`, and every path that
//   touches an encoder takes the lock first. Nothing hands an encoder handle
//   out, so no handle can outlive the lock that produced it.
unsafe impl Send for MetalBackend {}
// SAFETY: as above; every path into the mutable state goes through
// `self.inner.lock()`, so there is no data race on the encoder handles, and the
// remaining members are thread-safe by the driver's own documentation.
unsafe impl Sync for MetalBackend {}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MetalBackend>();
};

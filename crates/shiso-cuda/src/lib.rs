//! shiso-cuda — the CUDA compute backend.
//!
//! One device is opened, every kernel module this build embeds is loaded at
//! open time, and every launch goes through [`shiso_core::Backend`], so an
//! engine holds `Arc<dyn Backend>` and never names this crate. The three
//! optional capabilities — graph capture, matrix multiplication and cooperative
//! launch — are reached through `as_graph()`, `as_blas()` and `as_coop()`, for
//! the same reason: nothing above this crate should have to know which device
//! it is running on in order to ask.
//!
//! # It compiles everywhere; it runs on one target
//!
//! The driver bindings are a Linux-only dependency. Everywhere else this crate
//! builds against a stub arm where every driver entry point returns a typed
//! error, so [`CudaBackend::new`] on a developer machine fails with
//! `CoreError::Backend { backend: "cuda", op: "cuInit", .. }` rather than
//! failing to link or panicking. That is not a courtesy: it is what lets the
//! parts of this backend that are *not* driver calls — the buffer table, the
//! argument packer, the module registry's fail-closed rule, the graph cache's
//! state machine, the multiply validator — be built and tested on a machine
//! with no GPU, which is where they are written.
//!
//! What cannot be tested there is anything that needs a driver to answer. Those
//! tests are marked `#[ignore]` and gated to Linux, and nothing in this crate
//! claims a result it did not execute.
//!
//! # Weights cost a copy
//!
//! [`shiso_core::Backend::import_raw`] is left at its default, which allocates
//! and copies. This device can be made to read host memory in place, but only
//! for a mapping registered with the driver, and that registration has costs
//! and failure modes of its own. So every weight tensor costs one copy at load
//! time. That is a real line in a memory budget and it is stated here rather
//! than hidden behind a fast path that only sometimes applies.
//!
//! # Launch geometry
//!
//! `LaunchArgs::grid` is a block count and `LaunchArgs::block` is threads per
//! block, passed to the device exactly as the caller computed them. Nothing
//! here divides, rounds up, or otherwise reinterprets a grid.
//!
//! # The one place this backend and its sibling differ
//!
//! `KernelRef::module` is a lookup key here. The Metal backend compiles every
//! kernel into one flat library, so its module field is metadata; here each
//! module is a separately compiled image with its own symbol table, and both
//! halves of a `KernelRef` are needed to name a function. The other side of
//! that coin is `KernelRef::consts`: there is no way to give a compiled entry
//! point a constant at launch time on this device, so a non-empty constant
//! table is [`shiso_core::CoreError::Unsupported`] rather than a field this
//! backend quietly ignores. Template variants are separate entry points in the
//! generated registry, with their own names.
#![expect(
    unsafe_code,
    reason = "every driver entry point is foreign, and the launch path builds its argument array at runtime"
)]

mod blas;
mod device;
mod graph;
mod launch;
mod ptxmod;
mod sys;
#[cfg(test)]
mod testing;

use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use shiso_core::{
    Backend, BlasBackend, CoopLaunch, CoreResult, DeviceView, Extent3, GemmDesc, GraphBackend,
    GraphCapture, GraphKey, KernelRef, LaunchArgs, PointerPack, RawBuffer,
};

use crate::blas::Blas;
pub use crate::blas::{BlasTier, CudaGemm, GemmBatch, Op, Operand};
use crate::device::{BufferTable, DeviceCtx};
use crate::graph::GraphCache;
use crate::launch::LaunchMode;
use crate::ptxmod::ModuleRegistry;

/// How a backend is opened.
///
/// A struct rather than a list of positional flags: `new(0, true, false)` reads
/// the same whichever way round the two flags go, and the next knob would make
/// it worse. Build one with [`CudaOptions::default`] and the `with_*` methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CudaOptions {
    /// Whether the backend offers graph capture at all.
    ///
    /// When this is off, `as_graph()` answers `None`. That is deliberately not
    /// an error variant: a caller that asks for a capability and gets `None`
    /// cannot forget to handle the refusal, because there is nothing to forget.
    pub decode_graphs: bool,
    /// Whether single-precision matrix-multiply operands may be rounded on
    /// their way into the multiplier.
    ///
    /// This is a quality decision, not a tuning knob, so it is an argument
    /// rather than an environment variable: whoever chose the quality mode is
    /// the one who should be setting it.
    pub tf32: bool,
}

impl Default for CudaOptions {
    /// Graphs on, reduced-precision accumulation on.
    ///
    /// Both match what the reference implementation this port follows does by
    /// default, which is what the reference outputs were captured with.
    fn default() -> Self {
        Self {
            decode_graphs: true,
            tf32: true,
        }
    }
}

impl CudaOptions {
    /// Chooses whether graph capture is offered.
    #[must_use]
    pub const fn with_decode_graphs(mut self, on: bool) -> Self {
        self.decode_graphs = on;
        self
    }

    /// Chooses whether single-precision operands may be rounded.
    #[must_use]
    pub const fn with_tf32(mut self, on: bool) -> Self {
        self.tf32 = on;
        self
    }
}

/// Everything one lock protects.
///
/// The buffer table and the graph cache share a lock because they share an
/// invariant: a captured graph names addresses the table hands out, and the two
/// must not move relative to each other. Everything else — the module registry
/// and the matrix library — carries its own lock, and is always taken *after*
/// this one, which is the whole of the lock ordering.
#[derive(Debug, Default)]
pub(crate) struct State {
    pub(crate) buffers: BufferTable,
    pub(crate) graphs: GraphCache,
    /// The slab the cooperative kernels share, and how large it is.
    ///
    /// Held here rather than handed out as an ordinary allocation because its
    /// address has to outlive a captured graph: every block of a cooperative
    /// launch addresses this one slab, and a capture bakes the address in.
    pub(crate) scratch: Option<(RawBuffer, u64)>,
    /// The slab the matrix tier widens a quantized operand into, and how large
    /// it is.
    ///
    /// Separate from `scratch` above and not a second use of it: that slab's
    /// head is the lock chain a cooperative grid hands partial sums down, and
    /// a prefill chunk writing a dequantized weight over it would not be a
    /// wrong answer, it would be the next cooperative launch waiting on a
    /// barrier nobody releases.
    pub(crate) blas_staging: Option<(RawBuffer, u64)>,
}

/// The CUDA compute backend.
///
/// One instance per process is the intended shape: it owns the device context,
/// both streams, every loaded kernel module and the buffer table, and all of
/// those only pay off when they are shared. A second instance on a *different*
/// device ordinal is refused outright — see [`CudaBackend::new`].
pub struct CudaBackend {
    options: CudaOptions,
    state: Mutex<State>,
    modules: ModuleRegistry,
    blas: Blas,
    /// Declared last so it drops last: every other member holds handles that
    /// are only valid inside this context.
    device: DeviceCtx,
}

/// Whether this build can reach a CUDA driver at all.
///
/// Tests that need a device skip when this is `false`, so the default gate
/// never needs a GPU. It is always `false` on a target built against the stub
/// arm, and it is `false` on a target that has the bindings but no driver
/// installed.
#[must_use]
pub fn cuda_available() -> bool {
    sys::nvidia::init().is_ok()
}

impl CudaBackend {
    /// Opens device `ordinal` and loads every kernel module this build embeds.
    ///
    /// Modules are loaded here rather than on first use, because a kernel set
    /// that does not load is a deployment error and should surface while a
    /// model is loading rather than part-way through answering a request.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Unsupported`] when this process already opened
    /// a different ordinal — two contexts on two devices would make one
    /// backend's buffers silently unreachable from the other's launches — and
    /// [`shiso_core::CoreError::Backend`] when the driver cannot be
    /// initialised, the device refuses a context, a stream or a handle, or a
    /// kernel module is empty or unloadable in this build. On a target with no
    /// driver, the first of those is what every call gets.
    pub fn new(ordinal: u32, options: CudaOptions) -> CoreResult<Arc<Self>> {
        let device = DeviceCtx::open(ordinal)?;
        let blas = Blas::open(options.tf32)?;
        blas.bind_stream(device.work_stream())?;
        let modules = ModuleRegistry::load_all()?;
        tracing::debug!(
            ordinal,
            decode_graphs = options.decode_graphs,
            tf32 = options.tf32,
            modules = modules.len(),
            "opened CUDA backend",
        );
        Ok(Arc::new(Self {
            options,
            state: Mutex::new(State::default()),
            modules,
            blas,
            device,
        }))
    }

    /// The options this backend was opened with.
    #[must_use]
    pub const fn options(&self) -> CudaOptions {
        self.options
    }

    /// Chooses whether single-precision multiply operands may be rounded.
    ///
    /// Takes effect on the next multiply; it does not re-run anything.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] when the library refuses the mode.
    pub fn set_tf32(&self, on: bool) -> CoreResult<()> {
        self.blas.set_tf32(on)
    }

    /// Number of live device allocations.
    #[must_use]
    pub fn live_buffers(&self) -> usize {
        self.state.lock().buffers.len()
    }

    /// Number of captured graphs held.
    #[must_use]
    pub fn captured_graphs(&self) -> usize {
        self.state.lock().graphs.len()
    }

    /// Number of kernel modules this build loaded.
    #[must_use]
    pub fn loaded_modules(&self) -> usize {
        self.modules.len()
    }

    /// Entry-point resolutions served from the handle cache rather than the
    /// driver.
    ///
    /// A forward pass launches the same few dozen entry points over and over,
    /// so after the first pass this should be growing by one per launch. It
    /// growing slower than that means something is handing out `KernelRef`s
    /// that differ where they should not, which costs a driver round trip per
    /// launch and is otherwise invisible.
    #[must_use]
    pub fn entry_cache_hits(&self) -> usize {
        self.modules.cache_hits()
    }

    /// Resolves an entry point and caches its handle, without launching it.
    ///
    /// An engine calls this while it loads a model, so a kernel this build does
    /// not embed fails at load time rather than on the first token.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Unsupported`] for a specialized reference, and
    /// [`shiso_core::CoreError::Backend`] when the module or the entry point is
    /// absent from this build.
    pub fn prepare(&self, kernel: KernelRef) -> CoreResult<()> {
        launch::reject_function_constants(kernel)?;
        self.device.bind()?;
        self.modules.function(kernel).map(|_| ())
    }

    /// Runs a matrix multiply this backend can express and the shared trait
    /// cannot: a strided batch, or the heuristic tier.
    ///
    /// The shared [`GemmDesc`] says how each operand is read but describes an
    /// unbatched multiply on the default tier, which is what
    /// [`BlasBackend::gemm`] runs. A caller that needs more than that reaches
    /// it here, through the concrete type.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] for a descriptor that describes
    /// nothing runnable or a library failure,
    /// [`shiso_core::CoreError::OutOfBounds`] for an operand shorter than the
    /// shape it declares, and [`shiso_core::CoreError::Unsupported`] for a
    /// batched multiply on the heuristic tier.
    pub fn gemm_ex(&self, desc: &CudaGemm<'_>) -> CoreResult<()> {
        blas::gemm(self, desc)
    }
}

impl Backend for CudaBackend {
    fn backend_name(&self) -> &'static str {
        sys::BACKEND
    }

    fn resolve_entry(&self, module: &'static str, logical: &str) -> Option<KernelRef> {
        ModuleRegistry::resolve(module, logical)
    }

    fn module_entries(&self, module: &'static str) -> Vec<String> {
        ModuleRegistry::entries(module)
    }

    /// Yes, and it is the same "yes" that `launch::reject_function_constants`
    /// is the other half of. A module here holds one symbol per instantiation,
    /// there is no launch-time constant, and the refusal is deliberate - so a
    /// caller that would have specialized has to ask for the variant by name
    /// instead, and this is how it finds that out before it builds a
    /// `KernelRef` that cannot be launched.
    fn specializes_by_name(&self) -> bool {
        true
    }

    /// The driver's own limit for the COMPILED entry point, which is its
    /// `__launch_bounds__` where it has one.
    ///
    /// The qwen35 GDN set is nine entry points that differ only in the lane
    /// count their index arithmetic assumes, and `shiso_engine::tune` picks
    /// among them by asking this. Answering `None` for all nine - which is what
    /// the default did before this - is read as "the backend has no opinion",
    /// so the reference shape stands and every case that loops over the viable
    /// ones silently finds none.
    fn max_block_threads(&self, kernel: KernelRef) -> Option<u32> {
        self.modules.max_block_threads(kernel)
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        device::alloc(self, bytes)
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        device::free(self, raw)
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        device::copy_h2d(self, dst, src)
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        device::copy_d2h(self, src, dst)
    }

    fn copy_d2d(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()> {
        device::copy_d2d(self, dst, src)
    }

    fn fill_zero(&self, dst: DeviceView<'_>) -> CoreResult<()> {
        device::fill_zero(self, dst)
    }

    fn run(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()> {
        launch::dispatch(self, kernel, args, LaunchMode::Ordinary)
    }

    fn sync(&self) -> CoreResult<()> {
        device::sync(self)
    }

    fn as_graph(&self) -> Option<&dyn GraphBackend> {
        self.options.decode_graphs.then_some(self)
    }

    fn as_blas(&self) -> Option<&dyn BlasBackend> {
        Some(self)
    }

    fn as_coop(&self) -> Option<&dyn CoopLaunch> {
        Some(self)
    }

    fn as_pointers(&self) -> Option<&dyn PointerPack> {
        Some(self)
    }
}

impl PointerPack for CudaBackend {
    fn pack_pointers(&self, dst: DeviceView<'_>, srcs: &[DeviceView<'_>]) -> CoreResult<()> {
        device::pack_pointers(self, dst, srcs)
    }
}

impl GraphBackend for CudaBackend {
    fn capture(&self, key: GraphKey) -> CoreResult<GraphCapture<'_>> {
        graph::capture(self, key)
    }

    fn finish_capture(&self, key: GraphKey) -> CoreResult<()> {
        graph::finish_capture(self, key)
    }

    fn abort_capture(&self, key: GraphKey) -> CoreResult<()> {
        graph::abort_capture(self, key)
    }

    fn replay(&self, key: GraphKey) -> CoreResult<bool> {
        graph::replay(self, key)
    }

    fn invalidate_all(&self) {
        graph::invalidate_all(self);
    }
}

impl BlasBackend for CudaBackend {
    fn gemm(&self, desc: &GemmDesc<'_>) -> CoreResult<()> {
        blas::gemm_core(self, desc)
    }

    // `None` is not a refusal here either; see `blas::staging`.
    fn staging(&self, bytes: u64) -> CoreResult<Option<DeviceView<'_>>> {
        blas::staging(self, bytes)
    }
}

impl CoopLaunch for CudaBackend {
    fn reserve_scratch(&self, bytes: u64) -> CoreResult<()> {
        self.device.bind()?;
        // The lock is taken in a block of its own, and that is not tidiness.
        // Every helper below takes the same lock, and it is not reentrant: a
        // reservation that held it across the allocation would not be slow, it
        // would never return.
        {
            let state = self.state.lock();
            // Growing mid-capture would move an address the capture has already
            // recorded, which is the same reason an ordinary allocation is
            // refused there - except that here the caller can legitimately ask
            // for a size it already has, so that case stays an answer rather
            // than a refusal.
            if let Some((_, held)) = state.scratch
                && held >= bytes
            {
                return Ok(());
            }
            if state.graphs.is_capturing() {
                return Err(shiso_core::CoreError::Unsupported {
                    what: "growing cooperative scratch while a graph capture is running",
                    backend: "cuda",
                });
            }
        }
        let raw = device::alloc(self, bytes.max(1))?;
        // A fresh allocation is whatever the last tenant of that address left
        // behind. The lock chain the cooperative kernels hand partial sums down
        // sits at the head of this slab, and every one of them *leaves* it
        // zeroed rather than initializing it - so the first launch after a
        // reservation reads whatever was there, and a nonzero chain is a grid
        // that waits on a barrier nobody will release.
        device::clear(self, raw)?;
        let spare = {
            let mut state = self.state.lock();
            match state.scratch {
                // Another caller grew the slab while this one was allocating.
                // Its slab is at least as large, so it stays and this one is
                // the spare.
                Some((_, held)) if held >= bytes => Some(raw),
                // The old slab goes only once the new one exists, so a failed
                // growth leaves the caller with what it had rather than with
                // nothing.
                _ => state.scratch.replace((raw, bytes)).map(|(old, _)| old),
            }
        };
        if let Some(spare) = spare {
            device::free(self, spare)?;
        }
        Ok(())
    }

    fn scratch(&self) -> Option<DeviceView<'_>> {
        let state = self.state.lock();
        let (raw, bytes) = state.scratch?;
        Some(DeviceView::from_raw(raw, 0, bytes))
    }

    fn max_co_resident_blocks(
        &self,
        kernel: KernelRef,
        block: Extent3,
        smem: u32,
    ) -> CoreResult<u32> {
        launch::max_co_resident_blocks(self, kernel, block, smem)
    }

    fn run_cooperative(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()> {
        launch::dispatch(self, kernel, args, LaunchMode::Cooperative)
    }
}

impl fmt::Debug for CudaBackend {
    /// Driver handles print as nothing useful, so the identity is what the
    /// backend holds.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.lock();
        f.debug_struct("CudaBackend")
            .field("options", &self.options)
            .field("live_buffers", &state.buffers.len())
            .field("captured_graphs", &state.graphs.len())
            .field("modules", &self.modules.len())
            .finish_non_exhaustive()
    }
}

impl Drop for CudaBackend {
    /// Drains the work stream and frees anything the table still holds.
    ///
    /// A `Drop` cannot report failure, so both are logged. Draining first
    /// matters: freeing memory a queued kernel is still writing to is exactly
    /// the kind of fault that surfaces somewhere else entirely.
    fn drop(&mut self) {
        if let Err(err) = device::sync(self) {
            tracing::error!(error = %err, "CUDA drain on drop failed");
        }
        // The matrix tier's staging slab is an ordinary allocation that no
        // owner ever returns, so it is released here by name. Left to
        // `release_all` it would still be freed, but as a *live* allocation at
        // teardown - which is logged as an error, and an error every clean run
        // prints is one nobody reads.
        let staging = self.state.lock().blas_staging.take();
        if let Some((raw, _)) = staging
            && let Err(err) = device::free(self, raw)
        {
            tracing::error!(error = %err, "matrix staging slab free failed");
        }
        device::release_all(&mut self.state.lock().buffers);
    }
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaBackend>();
};

/// The entry-name resolution rule, exposed so it can be tested and so a caller
/// can ask before it builds a plan around a kernel.
///
/// A thin face over the module registry: the registry itself is private because
/// it owns driver handles, and this is the one question about it a caller outside
/// the crate has any business asking.
#[derive(Debug)]
pub struct ModuleEntries;

impl ModuleEntries {
    /// The names `module` exposes, read out of its own text.
    #[must_use]
    pub fn names(module: &str) -> Vec<String> {
        ModuleRegistry::entries(module)
    }

    /// Resolves `logical` against those names, or `None` when it is absent or
    /// ambiguous. See [`ModuleEntries::pick`] for what ambiguous means here.
    #[must_use]
    pub fn resolve(module: &'static str, logical: &str) -> Option<KernelRef> {
        ModuleRegistry::resolve(module, logical)
    }

    /// The choosing rule over a list of names, without a module.
    #[must_use]
    pub fn pick<'a>(names: &'a [String], logical: &str) -> Option<&'a str> {
        ModuleRegistry::pick(names, logical)
    }
}

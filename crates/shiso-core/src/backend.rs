//! The device abstraction.
//!
//! One trait describes everything a compute device must do: allocate, transfer,
//! launch. It is object-safe on purpose — engines hold `Arc<dyn Backend>` and
//! hand `&dyn Backend` down the call stack, so a model never knows which device
//! it is running on and a backend can be swapped without touching a call site.
//!
//! Transfers and launches are ordered: calls on one backend are ordered on a
//! single implicit stream, so a copy issued before a launch is ordered before
//! it. A device that cannot promise that ordering must not implement the trait.

use std::fmt;
use std::ops::Range;

use crate::buffer::{DeviceView, HostBytes, RawBuffer};

use crate::error::CoreResult;

/// A three-dimensional extent: a grid or a block, always three-dimensional.
///
/// A one-dimensional launch uses [`Extent3::linear`] rather than three
/// positional arguments, so a one-dimensional grid cannot accidentally be
/// passed as `(1, 1, x)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent3 {
    /// Fastest-varying extent.
    pub x: u32,
    /// Second dimension, `1` for a linear launch.
    pub y: u32,
    /// Third dimension, `1` for a linear launch.
    pub z: u32,
}

impl Extent3 {
    /// A one-dimensional launch of `x` units.
    #[must_use]
    pub const fn linear(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// An explicit three-dimensional launch.
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total number of units, for sizing scratch space.
    #[must_use]
    pub const fn volume(self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

impl From<u32> for Extent3 {
    fn from(x: u32) -> Self {
        Self::linear(x)
    }
}

impl From<(u32, u32, u32)> for Extent3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self { x, y, z }
    }
}

/// A specialization constant baked into a pipeline: a Metal function constant
/// or a CUDA template variant.
///
/// Specialization travels *inside* the [`KernelRef`], so nothing parses a name
/// suffix at launch time and a misspelled variant is a missing table entry at
/// compile time rather than a launch failure at request time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FnConst {
    /// Index of the constant in the kernel's constant table.
    pub index: u16,
    /// Its value.
    pub value: FnConstValue,
}

/// Value of a specialization constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FnConstValue {
    /// A toggle such as "apply the output gate".
    Bool(bool),
    /// Signed tile dimension.
    I16(i16),
    /// Unsigned tile dimension.
    U16(u16),
    /// Count or stride baked into the kernel.
    U32(u32),
}

/// Names one kernel entry point together with its specialization.
///
/// Values come from a generated registry or a backend's own const table, never
/// from a runtime-formatted string: a kernel that does not exist is then a
/// compile error in the table, not a failed launch under load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelRef {
    /// Module or library the entry point lives in.
    pub module: &'static str,
    /// Entry point name.
    pub entry: &'static str,
    /// Specialization constants baked into the pipeline.
    pub consts: &'static [FnConst],
}

impl KernelRef {
    /// An unspecialized entry point.
    #[must_use]
    pub const fn new(module: &'static str, entry: &'static str) -> Self {
        Self {
            module,
            entry,
            consts: &[],
        }
    }

    /// An entry point with a specialization table attached.
    #[must_use]
    pub const fn specialized(
        module: &'static str,
        entry: &'static str,
        consts: &'static [FnConst],
    ) -> Self {
        Self {
            module,
            entry,
            consts,
        }
    }
}

/// One kernel parameter.
///
/// Arguments are passed in kernel declaration order: ordinal *i* here is
/// parameter *i* for a CUDA launch and binding index *i* for a Metal encoder.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum KernelArg<'a> {
    /// Unsigned 32-bit scalar.
    U32(u32),
    /// Signed 32-bit scalar.
    I32(i32),
    /// Unsigned 64-bit scalar, typically a byte stride.
    U64(u64),
    /// Single-precision scalar.
    F32(f32),
    /// A range of device memory.
    Buffer(DeviceView<'a>),
    /// A by-value struct argument. Built with `bytemuck::bytes_of` at the call
    /// site so the layout comes from the struct definition, never from a
    /// hand-packed byte array.
    Bytes(&'a [u8]),
}

/// A kernel launch: grid, block, dynamic shared memory and arguments.
#[derive(Debug, Clone, Copy)]
pub struct LaunchArgs<'a> {
    /// Number of blocks.
    pub grid: Extent3,
    /// Threads per block.
    pub block: Extent3,
    /// Dynamic shared (threadgroup) memory in bytes, placed at argument index
    /// zero when nonzero.
    pub smem_bytes: u32,
    /// Parameters in declaration order.
    pub args: &'a [KernelArg<'a>],
}

impl<'a> LaunchArgs<'a> {
    /// A launch with no dynamic shared memory.
    #[must_use]
    pub const fn new(grid: Extent3, block: Extent3, args: &'a [KernelArg<'a>]) -> Self {
        Self {
            grid,
            block,
            smem_bytes: 0,
            args,
        }
    }
}

/// A compute device.
///
/// Object-safe, `Send + Sync`, and every method takes `&self`: an engine holds
/// `Arc<dyn Backend>` and hands it to whichever component needs it. Calls are
/// internally serialised and ordered on one implicit stream, so a backend that
/// cannot promise that ordering must not implement this trait.
///
/// Allocation is exposed as `alloc_raw`/`free_raw` rather than a safe
/// constructor: allocation ownership lives in
/// [`crate::buffer::DeviceBuffer`], which calls these through its `Drop`.
pub trait Backend: Send + Sync + fmt::Debug {
    /// Short stable name used in error messages and traces.
    fn backend_name(&self) -> &'static str;

    /// Allocates `bytes` of device memory.
    #[doc(hidden)]
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the device refuses the allocation. A
    /// successful call always returns a `RawBuffer` whose `bytes` is the size
    /// actually reserved, which may exceed `bytes`.
    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer>;

    /// Releases a handle obtained from [`Backend::alloc_raw`].
    #[doc(hidden)]
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the device refuses the free. The handle is
    /// still considered dead afterwards, so the handle is never retried.
    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()>;

    /// Wraps host bytes without copying where the device supports it, otherwise
    /// allocates and copies.
    ///
    /// The default allocates and copies. A backend whose device can read host
    /// memory in place overrides this to wrap the mapping instead.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::OutOfBounds`] when `range` runs past the host bytes, and
    /// whatever [`Backend::alloc_raw`] reports for the fallback copy.
    fn import_raw(&self, host: &dyn HostBytes, range: Range<usize>) -> CoreResult<RawBuffer> {
        let len = range.end.saturating_sub(range.start);
        let raw = self.alloc_raw(len as u64)?;
        let view = DeviceView::from_raw(raw, 0, len as u64);
        self.copy_h2d(view, &host.as_ref()[range])?;
        Ok(raw)
    }

    /// Whether [`Backend::import_raw`] copies the host bytes, so nothing reads
    /// them once it returns.
    ///
    /// A loader uses this to hand a file's pages back to the kernel after each
    /// import: on a device that copies, keeping them cached costs the device's
    /// own allocations a reclaim, and on one that wraps the mapping in place
    /// dropping them would make the device read the file from disk. The
    /// default copies.
    fn imports_copy(&self) -> bool {
        true
    }

    /// Says the load is over: no more [`Backend::import_raw`] calls are coming.
    ///
    /// A backend that keeps host resources to make a burst of imports fast -
    /// page-locked staging, say - releases them here. Another import after this
    /// is still correct; it only pays to set that up again. The default has
    /// nothing to release.
    fn finish_imports(&self) {}

    /// The device's memory as its driver reports it now, or `None` when this
    /// backend cannot say.
    ///
    /// What sizing a cache to a fraction of the device reads, after the
    /// weights are resident. `None` is not zero: a caller that asked to fit a
    /// fraction refuses to start on a device that cannot answer, rather than
    /// guessing a size. The default cannot answer.
    fn memory_info(&self) -> Option<MemoryInfo> {
        None
    }

    /// Resolves an entry point whose real name is only known once its module is
    /// loaded, returning `None` when this device has no such kernel.
    ///
    /// Almost every kernel in this project is named by a constant, which is why
    /// [`KernelRef`] holds `&'static str`: a kernel that does not exist is then a
    /// compile error in a table rather than a failed launch under load. One family
    /// breaks that. Template kernels get names their compiler mangles from the
    /// argument list, and the only place those names exist is inside the module
    /// the toolchain produced - they cannot be written down ahead of time without
    /// reimplementing the mangling, and a *nearly* right mangled name is worse than
    /// a wrong one, because it can silently match a different instantiation.
    ///
    /// So a backend that loads such a module reads its entry list and answers here.
    /// `logical` is how the caller names the kernel it wants; what comes back is a
    /// `KernelRef` the caller can launch, interned by the backend for the process's
    /// life - which is exactly as long as the loaded module lives.
    ///
    /// The default answers `None`: a backend whose entry names are all constants
    /// has nothing to resolve, and a caller that gets `None` should report the
    /// missing kernel rather than fall back to a guess.
    fn resolve_entry(&self, _module: &'static str, _logical: &str) -> Option<KernelRef> {
        None
    }

    /// Every entry point `module` exposes, as the built module itself spells
    /// them.
    ///
    /// The companion to [`Backend::resolve_entry`], for the case that method
    /// cannot serve: a template with several instantiations has several mangled
    /// names that all contain the template's own, and telling them apart needs
    /// the argument list each name encodes. Resolving by name is then a choice
    /// no backend can make, because the meaning of those arguments belongs to
    /// the caller's kernel table and not to the device.
    ///
    /// So the names come out whole and the caller decodes them. That is the
    /// safe direction: a decoder reads the arguments a symbol actually carries
    /// and compares them with the ones it wanted, where a *mangler* would have
    /// to reproduce the compiler's encoding and could produce a name that
    /// matches a different instantiation - which launches happily and computes
    /// the wrong thing.
    ///
    /// The default is empty, which is every backend whose entry names are all
    /// constants and every build that compiled no kernels.
    fn module_entries(&self, _module: &'static str) -> Vec<String> {
        Vec::new()
    }

    /// Whether this device needs a specialization spelled into the entry name.
    ///
    /// Metal bakes a function constant into a pipeline when the pipeline is
    /// created, so one entry point serves every specialization of it and the
    /// caller passes the constants beside the name. A CUDA module has one
    /// symbol per instantiation and no way at all to give a compiled entry a
    /// constant at launch time; `crates/shiso-cuda/src/launch.rs` refuses the
    /// attempt rather than launching the wrong variant silently, because
    /// ignoring the field would compute a different product and say nothing.
    /// A caller on such a device has to ask for the variant by name.
    ///
    /// This is a question about the device, not about a kernel, so it is one
    /// method rather than a per-kernel table: the answer is the same for every
    /// entry point a backend owns.
    ///
    /// The default is `false` - a backend that can specialize at launch says
    /// nothing here, and every caller that never specializes can ignore this
    /// entirely.
    fn specializes_by_name(&self) -> bool {
        false
    }

    /// Threads per threadgroup this device will accept for `kernel`.
    ///
    /// `None` is "cannot say", which covers both a backend that does not
    /// describe launch limits and a kernel this device would refuse to build.
    /// The two are not distinguished because the caller's response to them
    /// differs by *how many* came back `None`, not by which reason produced
    /// one: every variant `None` means the backend has no opinion and the
    /// reference shape stands, while some `None` among numbers means those
    /// particular variants are not viable.
    ///
    /// # Cost
    ///
    /// The answer is a property of the *built* pipeline, so a first call for a
    /// given kernel compiles one. That is deliberate rather than incidental:
    /// the caller that asks this question is about to launch the kernel, so
    /// the compile is work brought forward, not work added.
    fn max_block_threads(&self, _kernel: KernelRef) -> Option<u32> {
        None
    }

    /// Copies host bytes into a device range.
    ///
    /// `src.len()` must equal `dst.len()`: a short or long source is
    /// [`crate::CoreError::OutOfBounds`] rather than a partial transfer, because
    /// a half-written tensor is worse than a failed one.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::OutOfBounds`] when `src` is longer than the destination
    /// view, and [`crate::CoreError::Backend`] when the transfer fails.
    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()>;

    /// Reads a device range back to host bytes.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::OutOfBounds`] when `dst` is shorter than `src`, and
    /// [`crate::CoreError::Backend`] when the copy fails.
    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()>;

    /// Writes zeros over a device range.
    ///
    /// A step of its own rather than a side effect of whoever allocated the
    /// buffer, because the buffers that need it are reused every layer: a
    /// histogram accumulated with `atomicAdd` is only a histogram if it starts
    /// at zero, and a cursor read back as an offset is only an offset if it
    /// does. Zeroing at allocation would be right exactly once.
    ///
    /// The default copies a run of zeros from the host, which is correct
    /// everywhere and is what a backend without a fill primitive gets. A
    /// backend whose device has one overrides this - and should, because the
    /// default's host round trip cannot be recorded into a graph.
    ///
    /// # Errors
    ///
    /// Whatever [`Backend::copy_h2d`] reports.
    fn fill_zero(&self, dst: DeviceView<'_>) -> CoreResult<()> {
        /// One run of zeros, reused rather than allocated per call.
        const ZEROS: [u8; 4096] = [0; 4096];
        let mut at = 0u64;
        while at < dst.len() {
            let len = (dst.len() - at).min(ZEROS.len() as u64);
            let width = usize::try_from(len).unwrap_or(ZEROS.len());
            self.copy_h2d(dst.slice(at, len)?, &ZEROS[..width])?;
            at += len;
        }
        Ok(())
    }

    /// Device-to-device copy. Requires `dst.len() == src.len()`.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::ShapeMismatch`] with `field: "copy_d2d len"` when the
    /// two views differ in length, and [`crate::CoreError::Backend`] when the
    /// copy fails.
    fn copy_d2d(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()>;

    /// Launches a kernel.
    ///
    /// `args` are in kernel declaration order. `smem_bytes > 0` reserves dynamic
    /// shared (threadgroup) memory, which the backends bind at argument index
    /// zero after the explicit arguments.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the launch fails, and
    /// [`crate::CoreError::Unsupported`] for a kernel this build did not register.
    fn run(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()>;

    /// Blocks until the device is idle.
    ///
    /// Also the point at which a batching backend flushes whatever it has
    /// accumulated, so a caller that needs results visible to another queue
    /// calls this rather than a backend-specific flush.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the device reports a failure while draining
    /// its queue.
    fn sync(&self) -> CoreResult<()>;

    /// Graph capture support, absent by default.
    ///
    /// Capabilities are accessors rather than error variants, so a backend that
    /// cannot do something is `None` and a caller cannot forget to check: there
    /// is nothing to forget to check.
    fn as_graph(&self) -> Option<&dyn GraphBackend> {
        None
    }

    /// BLAS support, absent by default.
    fn as_blas(&self) -> Option<&dyn BlasBackend> {
        None
    }

    /// Cooperative launch support, absent by default.
    fn as_coop(&self) -> Option<&dyn CoopLaunch> {
        None
    }

    /// Device-pointer packing, absent by default.
    fn as_pointers(&self) -> Option<&dyn PointerPack> {
        None
    }
}

/// Device memory at one moment, in bytes.
///
/// `free` is what the driver would still grant, which on a device that shares
/// memory with its host counts everything else the machine holds too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Bytes the driver would still grant.
    pub free: u64,
    /// Bytes the device has.
    pub total: u64,
}

/// Bytes one device pointer occupies in a packed array.
///
/// Every device this workspace targets addresses memory with sixty-four bits,
/// and the kernels that read a packed array index it as an array of pointers,
/// so the stride is the pointer width rather than a layout choice a caller
/// could make differently.
pub const DEVICE_POINTER_BYTES: u64 = 8;

/// A device that can write the addresses of its own allocations into its own
/// memory.
///
/// A capability rather than part of [`Backend`] for the reason the other three
/// are: a device that cannot hand out addresses answers `None` and a caller
/// cannot forget to check.
///
/// # Why this is a backend's job and not a caller's
///
/// A [`DeviceView`] is three numbers - a handle, an offset and a length - and
/// none of them is an address. That is deliberate: it is what makes a view
/// impossible to forge and what lets a plan be built on a machine with no
/// device. Some kernels, though, take an *array of pointers* - one page table
/// per row of a batch, one recurrent state per row - and something has to turn
/// views into the addresses that array holds. Doing it here keeps the
/// translation in the one place that already owns it, so no caller ever holds a
/// raw device address.
pub trait PointerPack: Send + Sync + fmt::Debug {
    /// Writes the address of each range in `srcs` into `dst`, in order, one
    /// [`DEVICE_POINTER_BYTES`]-wide slot each.
    ///
    /// `dst` must be at least `srcs.len() * DEVICE_POINTER_BYTES` long. A
    /// shorter destination is an error rather than a partial write: an array of
    /// pointers with a hole in it is a kernel dereferencing whatever was at that
    /// address before.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::OutOfBounds`] when `dst` is shorter than the slots
    /// `srcs` needs, and [`crate::CoreError::Backend`] when a range names no
    /// live allocation or the write fails.
    fn pack_pointers(&self, dst: DeviceView<'_>, srcs: &[DeviceView<'_>]) -> CoreResult<()>;
}
/// Identifies one captured device-graph replay slot.
///
/// `Copy + Eq + Hash` so a scheduler can key a cache of prepared graphs by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphKey(u64);

impl GraphKey {
    /// Builds a key from a slot number.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// The slot number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Guard returned while a graph is being captured.
///
/// Dropping it without committing aborts the capture, so an early `?` inside a
/// captured region can never leave the device in capture mode. That is the whole
/// reason capture is expressed as a guard rather than a begin/end pair.
#[derive(Debug)]
pub struct GraphCapture<'a> {
    backend: &'a dyn GraphBackend,
    key: GraphKey,
    committed: bool,
}

impl GraphCapture<'_> {
    /// The slot being filled.
    #[must_use]
    pub const fn key(&self) -> GraphKey {
        self.key
    }

    /// Finishes capture and registers the graph for replay.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the capture cannot be instantiated or
    /// uploaded. The guard stays live on failure, so the device still leaves
    /// capture mode on drop.
    pub fn finish(mut self) -> CoreResult<()> {
        self.committed = true;
        self.backend.finish_capture(self.key)
    }
}

impl Drop for GraphCapture<'_> {
    /// Aborts an uncommitted capture.
    ///
    /// A `Drop` cannot report failure, so an abort error is logged rather than
    /// returned; leaving the device mid-capture would corrupt every later launch.
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Err(err) = self.backend.abort_capture(self.key) {
            tracing::error!(
                key = self.key.get(),
                error = %err,
                "graph capture abort failed",
            );
        }
    }
}

/// Device-graph capture and replay.
///
/// This is a capability, not part of [`Backend`]: a device that records command
/// streams incrementally does not implement it, and `as_graph()` returning `None`
/// says so without an error variant every caller must remember to check.
pub trait GraphBackend: Send + Sync + fmt::Debug {
    /// Begins capturing into `key`. The returned guard commits or aborts.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the device cannot enter capture mode.
    fn capture(&self, key: GraphKey) -> CoreResult<GraphCapture<'_>>;

    /// Finishes a capture started by [`GraphBackend::capture`] and registers it
    /// for replay.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the capture cannot be instantiated or
    /// uploaded.
    fn finish_capture(&self, key: GraphKey) -> CoreResult<()>;

    /// Discards a capture in progress, returning the device to normal execution.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the device cannot leave capture mode.
    /// The [`GraphCapture`] guard logs such a failure rather than swallowing it,
    /// because a device left mid-capture breaks every later launch.
    fn abort_capture(&self, key: GraphKey) -> CoreResult<()>;

    /// Replays a captured graph. `Ok(false)` means no graph is registered.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the graph failed to run.
    fn replay(&self, key: GraphKey) -> CoreResult<bool>;

    /// Drops every captured graph, for example after a device reset.
    fn invalidate_all(&self);
}

impl<'a> GraphCapture<'a> {
    /// Wraps a capture the device has already entered.
    ///
    /// Only a [`GraphBackend`] implementation calls this, immediately after the
    /// device has actually entered capture mode.
    #[must_use]
    pub fn started(backend: &'a dyn GraphBackend, key: GraphKey) -> Self {
        Self {
            backend,
            key,
            committed: false,
        }
    }
}

/// Element types a BLAS call can be asked to multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GemmDType {
    /// Both operands and the accumulator are single precision.
    F32,
    /// Single precision that is never rounded on its way into the multiplier,
    /// whatever reduced-precision mode the backend was opened with.
    ///
    /// For the projections whose answer a rounding can flip - a router choosing
    /// experts from logits a hair apart is the one this exists for.
    F32Strict,
    /// Half-precision operands accumulated in single precision.
    F16,
    /// Half-precision operands accumulated in single precision and written
    /// back as halves, for a destination whose every reader takes halves.
    F16ToF16,
    /// Brain-float operands accumulated in single precision.
    BF16,
}

/// How a multiply reads one of its operands.
///
/// A named pair rather than a `bool`, because at a call site `true` says
/// nothing about which of the two it means, and the two produce different
/// numbers from the same three buffers.
///
/// Not `#[non_exhaustive]`: a caller outside this crate builds one for every
/// multiply it describes, and there is no third way to read a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GemmOp {
    /// Read as stored.
    #[default]
    AsStored,
    /// Read transposed.
    Transposed,
}

/// One matrix multiply, as a single argument block.
///
/// Row strides travel with their matrix, so transposing `lda` and `ldb` at a
/// call site does not compile: the fields are named, and the operand views carry
/// their own lengths.
#[derive(Debug, Clone, Copy)]
///
/// Not `#[non_exhaustive]`, and the reason is the same everywhere it is missing
/// in this workspace: that attribute exists so a library can add a field without
/// breaking callers it cannot recompile. Every caller here is in this repository
/// and recompiles with it, and this is a type other crates have to BUILD - so all
/// the attribute would buy is that they cannot. A new field should break every
/// site that now has to supply it.
pub struct GemmDesc<'a> {
    /// How the left operand is read.
    ///
    /// A projection whose weight is stored one output row at a time is read
    /// transposed, and the transposition is not a stride the caller could have
    /// folded into `lda`: it changes which of the two dimensions the leading
    /// dimension counts.
    pub op_a: GemmOp,
    /// How the right operand is read.
    pub op_b: GemmOp,
    /// Rows of the product.
    pub m: u32,
    /// Columns of the product.
    pub n: u32,
    /// Shared inner dimension.
    pub k: u32,
    /// Leading dimension of `a`.
    pub lda: u32,
    /// Leading dimension of `b`.
    pub ldb: u32,
    /// Leading dimension of `c`.
    pub ldc: u32,
    /// Left operand.
    pub a: DeviceView<'a>,
    /// Right operand.
    pub b: DeviceView<'a>,
    /// Accumulator, read-modify-written when `beta` is nonzero.
    pub c: DeviceView<'a>,
    /// Scale applied to the product.
    pub alpha: f32,
    /// Scale applied to the existing accumulator.
    pub beta: f32,
    /// Element type of the operands.
    pub dtype: GemmDType,
}

/// A BLAS-capable device.
pub trait BlasBackend: Send + Sync + fmt::Debug {
    /// Multiplies and accumulates.
    ///
    /// One description struct rather than twelve positional arguments: a
    /// transposed stride has to be a deliberate field assignment, not a
    /// plausible-looking integer in the wrong slot.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Unsupported`] for a dtype pair this device has no path for,
    /// [`crate::CoreError::Backend`] when the call fails, and
    /// [`crate::CoreError::OutOfBounds`] when a view is shorter than the descriptor
    /// claims.
    fn gemm(&self, desc: &GemmDesc<'_>) -> CoreResult<()>;

    /// Device scratch at least `bytes` long, for an operand this tier cannot
    /// read in the form the caller holds it.
    ///
    /// A vendor matrix library has no quantized operand type, so a caller with
    /// a block-quantized weight has to widen it into somewhere first. That
    /// somewhere cannot be an ordinary allocation taken per call - a hundred
    /// megabytes of `cudaMalloc` and `cudaFree` measured 4.9 ms on this
    /// device's driver, which at one projection per tensor is most of a
    /// prefill chunk - and it cannot be the cooperative slab, whose head is a
    /// live lock chain for a cooperative grid. So it is the tier's own, grown
    /// on demand and released with the device.
    ///
    /// The default is `None`, which is not a failure: it says this device has
    /// no staging tier, and a caller that needs one keeps whatever path it
    /// would have taken anyway. Every caller must handle `None`, because a
    /// device that ran out of memory answers the same way a device that never
    /// had the tier does.
    ///
    /// The returned view is valid until the next call, which is why it is
    /// borrowed from `&self` and not handed out as a buffer: two callers
    /// staging at once would each see the other's operand.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the device refuses the allocation.
    fn staging(&self, _bytes: u64) -> CoreResult<Option<DeviceView<'_>>> {
        Ok(None)
    }
}

/// A device that can run cooperative kernels, where every block must be
/// resident at once because they synchronize against each other.
pub trait CoopLaunch: Send + Sync + fmt::Debug {
    /// Largest grid for `kernel` at this block size and shared-memory budget
    /// that still fits the device.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the device cannot answer for this kernel.
    fn max_co_resident_blocks(
        &self,
        kernel: KernelRef,
        block: Extent3,
        smem: u32,
    ) -> CoreResult<u32>;

    /// Reserves the per-stream scratch a cooperative kernel needs, growing it if
    /// what is already reserved is too small.
    ///
    /// Separate from an ordinary allocation because of when it has to happen: the
    /// kernels that need it synchronize across their whole grid, so every block
    /// addresses one shared slab, and the slab has to exist and keep its address
    /// for the life of a captured graph. A caller that allocated per launch would
    /// hand a captured graph an address that is free by the time it replays.
    ///
    /// Reserving less than is already held is a no-op rather than a shrink, for the
    /// same reason.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when the device cannot reserve that much, and
    /// [`crate::CoreError::Unsupported`] when a reservation is attempted while a
    /// graph capture is open - the growth would move an address the capture has
    /// already baked in.
    fn reserve_scratch(&self, bytes: u64) -> CoreResult<()>;

    /// The scratch reserved by [`CoopLaunch::reserve_scratch`], or `None` when
    /// nothing has been reserved yet.
    fn scratch(&self) -> Option<DeviceView<'_>>;

    /// Launches with the whole grid resident.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Unsupported`] when the device cannot launch cooperatively,
    /// [`crate::CoreError::Backend`] when the launch itself fails.
    fn run_cooperative(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> CoreResult<()>;
}

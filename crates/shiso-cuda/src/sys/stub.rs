//! The driver surface on a target that has no driver.
//!
//! Every entry point here has exactly the signature its Linux counterpart has
//! and answers [`SysError::no_driver`]. That is what makes this file a
//! type-check harness rather than dead weight: the whole backend — the buffer
//! table, the argument packer, the registry, the graph cache, the BLAS tier —
//! compiles and unit-tests against these bodies on a developer machine, and a
//! signature that drifts on the Linux side stops compiling here.
//!
//! No entry point panics and none is `unimplemented!()`. A caller on this
//! target gets a typed [`shiso_core::CoreError::Backend`] naming the driver
//! entry point it tried to reach, at exactly the place it would have got a
//! driver error, which is the only behaviour that lets the layers above be
//! exercised at all.

use std::ffi::{CStr, c_void};

use shiso_core::Extent3;

use super::{
    CuBlas, CuBlasLt, CuContext, CuDevice, CuEvent, CuFunction, CuGraph, CuGraphExec, CuModule,
    CuStream, DevicePtr, GemmCall, GemmStride, PinnedPtr, SysError, SysResult,
};

/// Whether the driver library can be found: never, on this target.
pub(crate) fn library_present() -> bool {
    false
}

/// Whether libcublasLt can be found: never, on this target.
pub(crate) fn blas_lt_library_present() -> bool {
    false
}

/// Initialises the driver.
///
/// # Errors
///
/// Always, on this target: there is no driver to initialise.
pub(crate) fn init() -> SysResult<()> {
    Err(SysError::no_driver("cuInit"))
}

/// Resolves a device by ordinal.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn device_get(_ordinal: u32) -> SysResult<CuDevice> {
    Err(SysError::no_driver("cuDeviceGet"))
}

/// Retains the device's primary context.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn primary_ctx_retain(_device: CuDevice) -> SysResult<CuContext> {
    Err(SysError::no_driver("cuDevicePrimaryCtxRetain"))
}

/// Releases one retain of the device's primary context.
///
/// # Safety
///
/// `device` must name a context this process previously retained and has not
/// yet released that retain of.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn primary_ctx_release(_device: CuDevice) -> SysResult<()> {
    Err(SysError::no_driver("cuDevicePrimaryCtxRelease"))
}

/// Binds a context to the calling thread.
///
/// # Safety
///
/// `ctx` must be a live context that has not been released.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn ctx_set_current(_ctx: CuContext) -> SysResult<()> {
    Err(SysError::no_driver("cuCtxSetCurrent"))
}

/// Number of multiprocessors on the device.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn multiprocessor_count(_device: CuDevice) -> SysResult<u32> {
    Err(SysError::no_driver("cuDeviceGetAttribute"))
}

/// Whether the device shares its memory with the host.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn integrated(_device: CuDevice) -> SysResult<bool> {
    Err(SysError::no_driver("cuDeviceGetAttribute"))
}

/// Opens a stream.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn stream_create() -> SysResult<CuStream> {
    Err(SysError::no_driver("cuStreamCreate"))
}

/// Closes a stream.
///
/// # Safety
///
/// `stream` must be live and must not be destroyed twice.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn stream_destroy(_stream: CuStream) -> SysResult<()> {
    Err(SysError::no_driver("cuStreamDestroy"))
}

/// Blocks until every operation queued on `stream` has finished.
///
/// # Safety
///
/// `stream` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn stream_synchronize(_stream: CuStream) -> SysResult<()> {
    Err(SysError::no_driver("cuStreamSynchronize"))
}

/// Opens an event that records ordering only.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn event_create() -> SysResult<CuEvent> {
    Err(SysError::no_driver("cuEventCreate"))
}

/// Closes an event.
///
/// # Safety
///
/// `event` must be live and must not be destroyed twice.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn event_destroy(_event: CuEvent) -> SysResult<()> {
    Err(SysError::no_driver("cuEventDestroy"))
}

/// Marks the point a stream has reached.
///
/// # Safety
///
/// `event` and `stream` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn event_record(_event: CuEvent, _stream: CuStream) -> SysResult<()> {
    Err(SysError::no_driver("cuEventRecord"))
}

/// Makes a stream wait for an event.
///
/// # Safety
///
/// `stream` and `event` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn stream_wait_event(_stream: CuStream, _event: CuEvent) -> SysResult<()> {
    Err(SysError::no_driver("cuStreamWaitEvent"))
}

/// Reserves device memory.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn mem_alloc(_bytes: u64) -> SysResult<DevicePtr> {
    Err(SysError::no_driver("cuMemAlloc"))
}

/// The current context's free and total device memory, in bytes.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn mem_get_info() -> SysResult<(u64, u64)> {
    Err(SysError::no_driver("cuMemGetInfo"))
}

/// Releases device memory.
///
/// # Safety
///
/// `ptr` must be the exact address [`mem_alloc`] returned, and must not already
/// have been freed.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn mem_free(_ptr: DevicePtr) -> SysResult<()> {
    Err(SysError::no_driver("cuMemFree"))
}

/// Copies host bytes into device memory and waits for the copy.
///
/// # Safety
///
/// `dst .. dst + src.len()` must lie inside one live allocation.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn memcpy_htod(_dst: DevicePtr, _src: &[u8], _stream: CuStream) -> SysResult<()> {
    Err(SysError::no_driver("cuMemcpyHtoDAsync"))
}

/// Asks the kernel to start reading `bytes` in. A no-op on this target.
pub(crate) const fn advise_willneed(_bytes: &[u8]) {}

/// Reserves page-locked host memory.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn mem_host_alloc(_bytes: u64) -> SysResult<PinnedPtr> {
    Err(SysError::no_driver("cuMemAllocHost"))
}

/// Releases page-locked host memory.
///
/// # Safety
///
/// `ptr` must be the exact address [`mem_host_alloc`] returned, must not
/// already have been freed, and no queued copy may still be reading it.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn mem_host_free(_ptr: PinnedPtr) -> SysResult<()> {
    Err(SysError::no_driver("cuMemFreeHost"))
}

/// Queues a copy of page-locked host memory into device memory.
///
/// # Safety
///
/// Both ranges must lie inside live allocations, and neither may be freed or
/// written until `stream` has been synchronised.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn memcpy_htod_pinned(
    _dst: DevicePtr,
    _src: PinnedPtr,
    _len: usize,
    _stream: CuStream,
) -> SysResult<()> {
    Err(SysError::no_driver("cuMemcpyHtoDAsync"))
}

/// Copies device memory into host bytes and waits for the copy.
///
/// # Safety
///
/// `src .. src + dst.len()` must lie inside one live allocation.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn memcpy_dtoh(
    _dst: &mut [u8],
    _src: DevicePtr,
    _stream: CuStream,
) -> SysResult<()> {
    Err(SysError::no_driver("cuMemcpyDtoHAsync"))
}

/// Queues a device-to-device copy on `stream`.
///
/// # Safety
///
/// Both ranges must lie inside live allocations and must not overlap.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn memcpy_dtod(
    _dst: DevicePtr,
    _src: DevicePtr,
    _bytes: u64,
    _stream: CuStream,
) -> SysResult<()> {
    Err(SysError::no_driver("cuMemcpyDtoDAsync"))
}

/// Queues a zero fill of `len` bytes at `dst` on `stream`.
///
/// # Safety
///
/// `dst .. dst + len` must lie inside one live allocation.
///
/// # Errors
///
/// Always [`SysError::Host`]: there is no driver on this target.
pub(crate) unsafe fn memset_zero(_dst: DevicePtr, _len: usize, _stream: CuStream) -> SysResult<()> {
    Err(SysError::no_driver("cuMemsetD8Async"))
}

/// Loads a kernel module from an in-memory image.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn module_load(_image: &[u8]) -> SysResult<CuModule> {
    Err(SysError::no_driver("cuModuleLoadData"))
}

/// Unloads a kernel module.
///
/// # Safety
///
/// `module` must be live and must not be unloaded twice.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn module_unload(_module: CuModule) -> SysResult<()> {
    Err(SysError::no_driver("cuModuleUnload"))
}

/// Resolves one entry point inside a loaded module.
///
/// # Safety
///
/// `module` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn module_get_function(
    _module: CuModule,
    _entry: &CStr,
) -> SysResult<CuFunction> {
    Err(SysError::no_driver("cuModuleGetFunction"))
}

/// Raises the dynamic shared-memory ceiling for one entry point.
///
/// # Safety
///
/// `func` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn func_set_max_dynamic_smem(_func: CuFunction, _bytes: u32) -> SysResult<()> {
    Err(SysError::no_driver("cuFuncSetAttribute"))
}

/// Threads per block this entry point may be launched with.
///
/// # Safety
///
/// `func` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn func_max_threads_per_block(_func: CuFunction) -> SysResult<u32> {
    Err(SysError::no_driver("cuFuncGetAttribute"))
}

/// Blocks of `func` that stay resident on one multiprocessor.
///
/// # Safety
///
/// `func` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn occupancy_max_active_blocks(
    _func: CuFunction,
    _block_threads: u32,
    _smem: u32,
) -> SysResult<u32> {
    Err(SysError::no_driver(
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    ))
}

/// Queues an ordinary launch on `stream`.
///
/// # Safety
///
/// `params` must hold exactly one pointer per declared kernel parameter, in
/// order, each addressing a value of that parameter's type; `func` and `stream`
/// must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn launch_kernel(
    _func: CuFunction,
    _grid: Extent3,
    _block: Extent3,
    _smem: u32,
    _stream: CuStream,
    _params: &mut [*mut c_void],
) -> SysResult<()> {
    Err(SysError::no_driver("cuLaunchKernel"))
}

/// Queues a launch whose whole grid must be resident at once.
///
/// # Safety
///
/// As [`launch_kernel`], and the grid must additionally fit the device's
/// co-residency limit for this entry point.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn launch_cooperative(
    _func: CuFunction,
    _grid: Extent3,
    _block: Extent3,
    _smem: u32,
    _stream: CuStream,
    _params: &mut [*mut c_void],
) -> SysResult<()> {
    Err(SysError::no_driver("cuLaunchCooperativeKernel"))
}

/// Puts `stream` into capture mode.
///
/// # Safety
///
/// `stream` must be live and must not already be capturing.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn stream_begin_capture(_stream: CuStream) -> SysResult<()> {
    Err(SysError::no_driver("cuStreamBeginCapture"))
}

/// Takes `stream` out of capture mode and yields what it captured.
///
/// # Safety
///
/// `stream` must be the stream a capture was begun on.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn stream_end_capture(_stream: CuStream) -> SysResult<CuGraph> {
    Err(SysError::no_driver("cuStreamEndCapture"))
}

/// Turns a captured graph into a replayable one.
///
/// # Safety
///
/// `graph` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn graph_instantiate(_graph: CuGraph) -> SysResult<CuGraphExec> {
    Err(SysError::no_driver("cuGraphInstantiateWithFlags"))
}

/// Releases a captured graph.
///
/// # Safety
///
/// `graph` must be live and must not be destroyed twice.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn graph_destroy(_graph: CuGraph) -> SysResult<()> {
    Err(SysError::no_driver("cuGraphDestroy"))
}

/// Releases an instantiated graph.
///
/// # Safety
///
/// `exec` must be live and must not be destroyed twice.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn graph_exec_destroy(_exec: CuGraphExec) -> SysResult<()> {
    Err(SysError::no_driver("cuGraphExecDestroy"))
}

/// Replays an instantiated graph on `stream`.
///
/// # Safety
///
/// `exec` and `stream` must be live, and every buffer the graph names must
/// still be allocated at the address it was captured with.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn graph_launch(_exec: CuGraphExec, _stream: CuStream) -> SysResult<()> {
    Err(SysError::no_driver("cuGraphLaunch"))
}

/// Opens a classic math-library handle.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn blas_create() -> SysResult<CuBlas> {
    Err(SysError::no_driver("cublasCreate"))
}

/// Closes a classic math-library handle.
///
/// # Safety
///
/// `handle` must be live and must not be destroyed twice.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn blas_destroy(_handle: CuBlas) -> SysResult<()> {
    Err(SysError::no_driver("cublasDestroy"))
}

/// Binds the stream every later call on `handle` runs on.
///
/// # Safety
///
/// `handle` and `stream` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn blas_set_stream(_handle: CuBlas, _stream: CuStream) -> SysResult<()> {
    Err(SysError::no_driver("cublasSetStream"))
}

/// Chooses between reduced-precision and full-precision accumulation.
///
/// # Safety
///
/// `handle` must be live.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn blas_set_math_mode(_handle: CuBlas, _tf32: bool) -> SysResult<()> {
    Err(SysError::no_driver("cublasSetMathMode"))
}

/// Queues one matrix multiply.
///
/// # Safety
///
/// Every address in `call` must lie inside a live allocation large enough for
/// the dimensions and strides the call declares.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn blas_gemm(_handle: CuBlas, _call: &GemmCall) -> SysResult<()> {
    Err(SysError::no_driver("cublasGemmEx"))
}

/// Queues a batch of matrix multiplies at fixed strides.
///
/// # Safety
///
/// As [`blas_gemm`], with each allocation additionally covering
/// `stride * (count - 1)` further elements.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn blas_gemm_strided(
    _handle: CuBlas,
    _call: &GemmCall,
    _stride: &GemmStride,
) -> SysResult<()> {
    Err(SysError::no_driver("cublasGemmStridedBatchedEx"))
}

/// Opens a heuristic-driven math-library handle.
///
/// # Errors
///
/// Always, on this target.
pub(crate) fn blas_lt_create() -> SysResult<CuBlasLt> {
    Err(SysError::no_driver("cublasLtCreate"))
}

/// Closes a heuristic-driven math-library handle.
///
/// # Safety
///
/// `handle` must be live and must not be destroyed twice.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn blas_lt_destroy(_handle: CuBlasLt) -> SysResult<()> {
    Err(SysError::no_driver("cublasLtDestroy"))
}

/// Queues one matrix multiply through the heuristic path.
///
/// # Safety
///
/// As [`blas_gemm`], and `workspace .. workspace + workspace_bytes` must be a
/// live allocation this call may overwrite.
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn blas_lt_matmul(
    _handle: CuBlasLt,
    _call: &GemmCall,
    _workspace: DevicePtr,
    _workspace_bytes: u64,
    _stream: CuStream,
) -> SysResult<()> {
    Err(SysError::no_driver("cublasLtMatmul"))
}

/// Queues one matrix multiply through the heuristic path with the algorithm
/// pinned per shape, whatever the column count.
///
/// # Safety
///
/// As [`blas_gemm`].
///
/// # Errors
///
/// Always, on this target.
pub(crate) unsafe fn blas_lt_matmul_pinned(
    _handle: CuBlasLt,
    _call: &GemmCall,
    _stream: CuStream,
) -> SysResult<()> {
    Err(SysError::no_driver("cublasLtMatmul"))
}

//! The driver surface on a target that has one.
//!
//! Thin wrappers over generated bindings, one per driver entry point, and
//! nothing else: no caching, no policy, no bookkeeping. Everything that decides
//! *when* to call one of these lives a layer up, where it is portable Rust that
//! a machine without a GPU can still compile and test. What is left here is the
//! part that cannot be tested anywhere but on a real device, and keeping it
//! mechanical is what makes that acceptable.
//!
//! # Why raw bindings and not the binding crate's safe layer
//!
//! The launch path builds its argument array at runtime, one slot per kernel
//! parameter, from a list whose length and element types are only known then.
//! The safe layer's launch builder wants the parameter list at compile time, so
//! the only way to launch a kernel whose signature the host learned from a
//! registry is the raw entry point with a `*mut *mut c_void`.
//!
//! # Status handling
//!
//! Every call funnels through [`check`], which turns a non-success status into
//! a [`SysError::Driver`] carrying the driver's own symbolic name and message.
//! Those two strings come from the driver, not from a table here, so a status
//! this build has never heard of still prints as something a reader can search
//! for.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_int, c_void};

use cudarc::cublas::sys as blas;
use cudarc::cublaslt::sys as lt;
use cudarc::driver::sys as cu;
use parking_lot::Mutex;
use shiso_core::Extent3;

use super::{
    CuBlas, CuBlasLt, CuContext, CuDevice, CuEvent, CuFunction, CuGraph, CuGraphExec, CuModule,
    CuStream, DevicePtr, GemmCall, GemmStride, MathType, Op, PinnedPtr, SysError, SysResult,
};

/// Turns a driver status into a result, naming the failing entry point.
fn check(op: &'static str, status: cu::CUresult) -> SysResult<()> {
    if status == cu::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    Err(SysError::Driver {
        op,
        code: status as i32,
        name: status_text(status, true),
        text: status_text(status, false),
    })
}

/// Asks the driver for its own name or message for a status.
///
/// Falls back to a placeholder rather than failing: this runs on the error
/// path, and an error about an error helps nobody.
fn status_text(status: cu::CUresult, symbolic: bool) -> String {
    let mut ptr: *const std::ffi::c_char = std::ptr::null();
    // SAFETY: the driver writes a pointer to one of its own static strings into
    // `ptr` and reads nothing else; `ptr` is a live local for the whole call.
    let got = unsafe {
        if symbolic {
            cu::cuGetErrorName(status, &raw mut ptr)
        } else {
            cu::cuGetErrorString(status, &raw mut ptr)
        }
    };
    if got != cu::CUresult::CUDA_SUCCESS || ptr.is_null() {
        return "<no description>".to_owned();
    }
    // SAFETY: the driver returned success, so `ptr` addresses a NUL-terminated
    // static string owned by the driver and valid for the process lifetime.
    let text = unsafe { CStr::from_ptr(ptr) };
    text.to_string_lossy().into_owned()
}

/// Turns a math-library status into a result.
fn check_blas(op: &'static str, status: blas::cublasStatus_t) -> SysResult<()> {
    if status == blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Ok(());
    }
    Err(SysError::Math {
        op,
        code: status as i32,
    })
}

/// Turns a heuristic-library status into a result.
fn check_lt(op: &'static str, status: lt::cublasStatus_t) -> SysResult<()> {
    if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Ok(());
    }
    Err(SysError::Math {
        op,
        code: status as i32,
    })
}

/// A count or dimension in the shape the driver takes it.
fn as_int(op: &'static str, value: u64) -> SysResult<c_int> {
    c_int::try_from(value).map_err(|_| SysError::Host {
        op,
        what: "a dimension does not fit the driver's signed 32-bit argument",
    })
}

/// A byte count in the shape the driver takes it.
fn as_size(op: &'static str, value: u64) -> SysResult<usize> {
    usize::try_from(value).map_err(|_| SysError::Host {
        op,
        what: "a byte count does not fit a host word",
    })
}

/// Initialises the driver.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver library is absent or refuses to
/// initialise.
pub(crate) fn init() -> SysResult<()> {
    // SAFETY: no arguments and no pointers; the flag is reserved and must be 0.
    check("cuInit", unsafe { cu::cuInit(0) })
}

/// Resolves a device by ordinal.
///
/// # Errors
///
/// [`SysError::Driver`] when no device carries that ordinal.
pub(crate) fn device_get(ordinal: u32) -> SysResult<CuDevice> {
    let ordinal = as_int("cuDeviceGet", u64::from(ordinal))?;
    let mut device: cu::CUdevice = 0;
    // SAFETY: `device` is a live local of exactly the type the driver writes.
    check("cuDeviceGet", unsafe {
        cu::cuDeviceGet(&raw mut device, ordinal)
    })?;
    Ok(CuDevice::from_raw(device))
}

/// Retains the device's primary context.
///
/// # Errors
///
/// [`SysError::Driver`] when the device refuses a context.
pub(crate) fn primary_ctx_retain(device: CuDevice) -> SysResult<CuContext> {
    let mut ctx: cu::CUcontext = std::ptr::null_mut();
    // SAFETY: `ctx` is a live local of exactly the type the driver writes, and
    // `device` came from `cuDeviceGet`.
    check("cuDevicePrimaryCtxRetain", unsafe {
        cu::cuDevicePrimaryCtxRetain(&raw mut ctx, device.raw())
    })?;
    Ok(CuContext::from_raw(ctx as usize))
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
/// [`SysError::Driver`] when the driver refuses the release.
pub(crate) unsafe fn primary_ctx_release(device: CuDevice) -> SysResult<()> {
    // SAFETY: the caller guarantees the outstanding retain.
    check("cuDevicePrimaryCtxRelease", unsafe {
        cu::cuDevicePrimaryCtxRelease_v2(device.raw())
    })
}

/// Binds a context to the calling thread.
///
/// # Safety
///
/// `ctx` must be a live context that has not been released.
///
/// # Errors
///
/// [`SysError::Driver`] when the context cannot be made current.
pub(crate) unsafe fn ctx_set_current(ctx: CuContext) -> SysResult<()> {
    // SAFETY: the caller guarantees `ctx` is live; the driver only records it
    // as this thread's current context.
    check("cuCtxSetCurrent", unsafe {
        cu::cuCtxSetCurrent(ctx.raw() as cu::CUcontext)
    })
}

/// Number of multiprocessors on the device.
///
/// Used to size a cooperative grid and to report in a trace. It is never used
/// to pick a kernel or an algorithm: device properties are reported
/// inconsistently enough across drivers and emulation layers that branching
/// numerics on them is how a build starts producing different answers on
/// different machines.
///
/// # Errors
///
/// [`SysError::Driver`] when the attribute cannot be read.
pub(crate) fn multiprocessor_count(device: CuDevice) -> SysResult<u32> {
    let mut count: c_int = 0;
    // SAFETY: `count` is a live local of the type the driver writes.
    check("cuDeviceGetAttribute", unsafe {
        cu::cuDeviceGetAttribute(
            &raw mut count,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device.raw(),
        )
    })?;
    u32::try_from(count).map_err(|_| SysError::Host {
        op: "cuDeviceGetAttribute",
        what: "the device reported a negative multiprocessor count",
    })
}

/// Whether the device shares its memory with the host.
///
/// Read for one purpose: on such a device the driver's free figure leaves out
/// the host's reclaimable page cache, which the kernel hands back on demand,
/// so sizing a cache from it alone undercounts what is really available.
///
/// # Errors
///
/// [`SysError::Driver`] when the attribute cannot be read.
pub(crate) fn integrated(device: CuDevice) -> SysResult<bool> {
    let mut flag: c_int = 0;
    // SAFETY: `flag` is a live local of the type the driver writes.
    check("cuDeviceGetAttribute", unsafe {
        cu::cuDeviceGetAttribute(
            &raw mut flag,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED,
            device.raw(),
        )
    })?;
    Ok(flag != 0)
}

/// Opens a stream.
///
/// # Errors
///
/// [`SysError::Driver`] when the device refuses a stream.
pub(crate) fn stream_create() -> SysResult<CuStream> {
    let mut stream: cu::CUstream = std::ptr::null_mut();
    // SAFETY: `stream` is a live local of the type the driver writes; flag 0 is
    // the default blocking stream behaviour.
    check("cuStreamCreate", unsafe {
        cu::cuStreamCreate(&raw mut stream, 0)
    })?;
    Ok(CuStream::from_raw(stream as usize))
}

/// Closes a stream.
///
/// # Safety
///
/// `stream` must be live and must not be destroyed twice.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver refuses the destroy.
pub(crate) unsafe fn stream_destroy(stream: CuStream) -> SysResult<()> {
    // SAFETY: the caller guarantees the handle is live and unreleased.
    check("cuStreamDestroy", unsafe {
        cu::cuStreamDestroy_v2(stream.raw() as cu::CUstream)
    })
}

/// Blocks until every operation queued on `stream` has finished.
///
/// # Safety
///
/// `stream` must be live.
///
/// # Errors
///
/// [`SysError::Driver`] when the stream reports a failure while draining. That
/// includes a fault raised by a kernel launched earlier: an asynchronous launch
/// reports here, not at the launch.
pub(crate) unsafe fn stream_synchronize(stream: CuStream) -> SysResult<()> {
    // SAFETY: the caller guarantees the handle is live.
    check("cuStreamSynchronize", unsafe {
        cu::cuStreamSynchronize(stream.raw() as cu::CUstream)
    })
}

/// Opens an event that records ordering only, not time.
///
/// # Errors
///
/// [`SysError::Driver`] when the device refuses an event.
pub(crate) fn event_create() -> SysResult<CuEvent> {
    let mut event: cu::CUevent = std::ptr::null_mut();
    // SAFETY: `event` is a live local of the type the driver writes.
    check("cuEventCreate", unsafe {
        cu::cuEventCreate(
            &raw mut event,
            cu::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
        )
    })?;
    Ok(CuEvent::from_raw(event as usize))
}

/// Closes an event.
///
/// # Safety
///
/// `event` must be live and must not be destroyed twice.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver refuses the destroy.
pub(crate) unsafe fn event_destroy(event: CuEvent) -> SysResult<()> {
    // SAFETY: the caller guarantees the handle is live and unreleased.
    check("cuEventDestroy", unsafe {
        cu::cuEventDestroy_v2(event.raw() as cu::CUevent)
    })
}

/// Marks the point `stream` has reached. Inside a capture this is a node the
/// graph orders later nodes against rather than a host-visible time.
///
/// # Safety
///
/// `event` and `stream` must be live.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver refuses the record.
pub(crate) unsafe fn event_record(event: CuEvent, stream: CuStream) -> SysResult<()> {
    // SAFETY: the caller guarantees both handles are live.
    check("cuEventRecord", unsafe {
        cu::cuEventRecord(event.raw() as cu::CUevent, stream.raw() as cu::CUstream)
    })
}

/// Makes everything queued on `stream` from now on wait for `event`. Waiting
/// on an event recorded in a capturing stream is what pulls `stream` into
/// that capture as a parallel branch.
///
/// # Safety
///
/// `stream` and `event` must be live.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver refuses the wait.
pub(crate) unsafe fn stream_wait_event(stream: CuStream, event: CuEvent) -> SysResult<()> {
    // SAFETY: the caller guarantees both handles are live.
    check("cuStreamWaitEvent", unsafe {
        cu::cuStreamWaitEvent(stream.raw() as cu::CUstream, event.raw() as cu::CUevent, 0)
    })
}

/// Reserves device memory.
///
/// # Errors
///
/// [`SysError::Driver`] when the device cannot reserve `bytes`.
pub(crate) fn mem_alloc(bytes: u64) -> SysResult<DevicePtr> {
    let size = as_size("cuMemAlloc", bytes)?;
    let mut ptr: cu::CUdeviceptr = 0;
    // SAFETY: `ptr` is a live local of the type the driver writes.
    check("cuMemAlloc", unsafe {
        cu::cuMemAlloc_v2(&raw mut ptr, size)
    })?;
    Ok(DevicePtr::from_raw(ptr))
}

/// The current context's free and total device memory, in bytes.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver refuses the query, which it does with
/// no current context.
pub(crate) fn mem_get_info() -> SysResult<(u64, u64)> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: both are live locals of the type the driver writes.
    check("cuMemGetInfo", unsafe {
        cu::cuMemGetInfo_v2(&raw mut free, &raw mut total)
    })?;
    Ok((free as u64, total as u64))
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
/// [`SysError::Driver`] when the driver refuses the free.
pub(crate) unsafe fn mem_free(ptr: DevicePtr) -> SysResult<()> {
    // SAFETY: the caller guarantees the address is a live allocation of its own.
    check("cuMemFree", unsafe { cu::cuMemFree_v2(ptr.raw()) })
}

/// Copies host bytes into device memory and waits for the copy.
///
/// The wait is not optional. `src` is borrowed for the duration of this call
/// and nothing keeps it alive afterwards, so returning before the device has
/// read it would leave the driver reading a dead stack frame. The two
/// host-touching transfers are exactly the points where this backend drains its
/// stream, which is the same discipline its sibling backend follows.
///
/// # Safety
///
/// `dst .. dst + src.len()` must lie inside one live allocation.
///
/// # Errors
///
/// [`SysError::Driver`] when the copy or the wait fails.
pub(crate) unsafe fn memcpy_htod(dst: DevicePtr, src: &[u8], stream: CuStream) -> SysResult<()> {
    let stream = stream.raw() as cu::CUstream;
    // SAFETY: the caller guarantees the destination range; `src` is a live
    // borrow for the whole call and the synchronise below keeps it live until
    // the device has finished reading it.
    check("cuMemcpyHtoDAsync", unsafe {
        cu::cuMemcpyHtoDAsync_v2(dst.raw(), src.as_ptr().cast::<c_void>(), src.len(), stream)
    })?;
    // SAFETY: the stream was just used successfully, so it is live.
    check("cuStreamSynchronize", unsafe {
        cu::cuStreamSynchronize(stream)
    })
}

/// Reserves page-locked host memory.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver cannot lock `bytes` of host memory.
pub(crate) fn mem_host_alloc(bytes: u64) -> SysResult<PinnedPtr> {
    let size = as_size("cuMemAllocHost", bytes)?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    // SAFETY: `ptr` is a live local of the type the driver writes.
    check("cuMemAllocHost", unsafe {
        cu::cuMemAllocHost_v2(&raw mut ptr, size)
    })?;
    Ok(PinnedPtr::from_raw(ptr as usize))
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
/// [`SysError::Driver`] when the driver refuses the free.
pub(crate) unsafe fn mem_host_free(ptr: PinnedPtr) -> SysResult<()> {
    // SAFETY: the caller guarantees the address is a live allocation of its own
    // that nothing is still reading.
    check("cuMemFreeHost", unsafe {
        cu::cuMemFreeHost(ptr.raw() as *mut c_void)
    })
}

/// Queues a copy of `len` bytes of page-locked host memory into device memory.
///
/// No wait, unlike [`memcpy_htod`]: the source is memory the caller owns and
/// keeps alive until it has synchronised `stream`, which is the whole reason
/// pinned memory is used for it.
///
/// # Safety
///
/// `dst .. dst + len` must lie inside one live allocation, `src .. src + len`
/// inside one live pinned allocation, and neither may be freed or written until
/// `stream` has been synchronised.
///
/// # Errors
///
/// [`SysError::Driver`] when the copy cannot be queued.
pub(crate) unsafe fn memcpy_htod_pinned(
    dst: DevicePtr,
    src: PinnedPtr,
    len: usize,
    stream: CuStream,
) -> SysResult<()> {
    // SAFETY: the caller guarantees both ranges and their lifetimes.
    check("cuMemcpyHtoDAsync", unsafe {
        cu::cuMemcpyHtoDAsync_v2(
            dst.raw(),
            src.raw() as *const c_void,
            len,
            stream.raw() as cu::CUstream,
        )
    })
}

/// Asks the kernel to start reading `bytes` in, if they are a file mapping.
///
/// Advice, not a request that can fail: a range that is not a mapping, or a
/// kernel that ignores the hint, costs the copy that follows its readahead and
/// nothing else, so the status is not reported.
pub(crate) fn advise_willneed(bytes: &[u8]) {
    const PAGE: usize = 4096;
    let start = bytes.as_ptr() as usize;
    let aligned = start & !(PAGE - 1);
    let len = bytes.len().saturating_add(start - aligned);
    // SAFETY: `MADV_WILLNEED` neither reads nor writes the range and never
    // changes its contents; the range starts at the page holding `bytes` and
    // covers only pages that hold part of it, all of which are mapped for as
    // long as the borrow lives.
    let _ = unsafe { libc::madvise(aligned as *mut c_void, len, libc::MADV_WILLNEED) };
}

/// Writes `len` zero bytes at `dst`, on the stream.
///
/// A device-side fill rather than a copy of zeros from the host: it takes no
/// host buffer, it does not drain the stream, and - the reason it exists at
/// all - it can be recorded into a graph capture, which a host round trip
/// cannot.
///
/// # Safety
///
/// `dst .. dst + len` must lie inside one live allocation.
///
/// # Errors
///
/// [`SysError::Driver`] when the fill fails.
pub(crate) unsafe fn memset_zero(dst: DevicePtr, len: usize, stream: CuStream) -> SysResult<()> {
    // SAFETY: the caller guarantees the destination range, and the stream was
    // created by this backend and lives as long as it does.
    check("cuMemsetD8Async", unsafe {
        cu::cuMemsetD8Async(dst.raw(), 0, len, stream.raw() as cu::CUstream)
    })
}

/// Copies device memory into host bytes and waits for the copy.
///
/// # Safety
///
/// `src .. src + dst.len()` must lie inside one live allocation.
///
/// # Errors
///
/// [`SysError::Driver`] when the copy or the wait fails.
pub(crate) unsafe fn memcpy_dtoh(
    dst: &mut [u8],
    src: DevicePtr,
    stream: CuStream,
) -> SysResult<()> {
    let stream = stream.raw() as cu::CUstream;
    let len = dst.len();
    // SAFETY: the caller guarantees the source range; `dst` is a live exclusive
    // borrow for the whole call and the synchronise below keeps it live until
    // the device has finished writing it.
    check("cuMemcpyDtoHAsync", unsafe {
        cu::cuMemcpyDtoHAsync_v2(dst.as_mut_ptr().cast::<c_void>(), src.raw(), len, stream)
    })?;
    // SAFETY: the stream was just used successfully, so it is live.
    check("cuStreamSynchronize", unsafe {
        cu::cuStreamSynchronize(stream)
    })
}

/// Queues a device-to-device copy on `stream`.
///
/// No wait: both ends are device memory the caller owns, so the copy is ordered
/// against the launches around it by the stream and nothing on the host has to
/// observe it.
///
/// # Safety
///
/// Both ranges must lie inside live allocations and must not overlap.
///
/// # Errors
///
/// [`SysError::Driver`] when the copy cannot be queued.
pub(crate) unsafe fn memcpy_dtod(
    dst: DevicePtr,
    src: DevicePtr,
    bytes: u64,
    stream: CuStream,
) -> SysResult<()> {
    let size = as_size("cuMemcpyDtoDAsync", bytes)?;
    // SAFETY: the caller guarantees both ranges and that they do not overlap.
    check("cuMemcpyDtoDAsync", unsafe {
        cu::cuMemcpyDtoDAsync_v2(dst.raw(), src.raw(), size, stream.raw() as cu::CUstream)
    })
}

/// Loads a kernel module from an in-memory image.
///
/// Two image shapes reach here. A compiled object starts with the ELF magic and
/// is handed to the driver as it stands. Anything else is assembly text, which
/// the driver reads as a C string, so it is copied into a NUL-terminated buffer
/// first — passing a slice that merely happens to be followed by a zero byte
/// would work until the day it was not.
///
/// # Errors
///
/// [`SysError::Host`] when the image is empty or holds an interior NUL, and
/// [`SysError::Driver`] when the driver refuses it — which is where an image
/// built for a different architecture surfaces.
pub(crate) fn module_load(image: &[u8]) -> SysResult<CuModule> {
    const OP: &str = "cuModuleLoadData";
    if image.is_empty() {
        return Err(SysError::Host {
            op: "cuModuleLoadData",
            what: "the kernel image is empty",
        });
    }
    let mut module: cu::CUmodule = std::ptr::null_mut();
    let status = if image.starts_with(b"\x7fELF") {
        // SAFETY: `image` is a live borrow for the whole call and the driver
        // copies whatever it needs out of it before returning.
        unsafe { cu::cuModuleLoadData(&raw mut module, image.as_ptr().cast::<c_void>()) }
    } else {
        let text = image.strip_suffix(b"\0").unwrap_or(image);
        let text = CString::new(text).map_err(|_| SysError::Host {
            op: "cuModuleLoadData",
            what: "the kernel image is text with an interior NUL byte",
        })?;
        // SAFETY: `text` is a live NUL-terminated buffer for the whole call.
        unsafe { cu::cuModuleLoadData(&raw mut module, text.as_ptr().cast::<c_void>()) }
    };
    check(OP, status)?;
    Ok(CuModule::from_raw(module as usize))
}

/// Unloads a kernel module.
///
/// # Safety
///
/// `module` must be live and must not be unloaded twice.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver refuses the unload.
pub(crate) unsafe fn module_unload(module: CuModule) -> SysResult<()> {
    // SAFETY: the caller guarantees the handle is live and unreleased.
    check("cuModuleUnload", unsafe {
        cu::cuModuleUnload(module.raw() as cu::CUmodule)
    })
}

/// Resolves one entry point inside a loaded module.
///
/// # Safety
///
/// `module` must be live.
///
/// # Errors
///
/// [`SysError::Driver`] when the module declares no such entry point.
pub(crate) unsafe fn module_get_function(module: CuModule, entry: &CStr) -> SysResult<CuFunction> {
    let mut func: cu::CUfunction = std::ptr::null_mut();
    // SAFETY: the caller guarantees the module is live; `entry` is a live
    // NUL-terminated borrow for the whole call.
    check("cuModuleGetFunction", unsafe {
        cu::cuModuleGetFunction(
            &raw mut func,
            module.raw() as cu::CUmodule,
            entry.as_ptr().cast(),
        )
    })?;
    Ok(CuFunction::from_raw(func as usize))
}

/// Raises the dynamic shared-memory ceiling for one entry point.
///
/// # Safety
///
/// `func` must be live.
///
/// # Errors
///
/// [`SysError::Driver`] when the device cannot give that entry point that much
/// shared memory.
pub(crate) unsafe fn func_set_max_dynamic_smem(func: CuFunction, bytes: u32) -> SysResult<()> {
    let bytes = as_int("cuFuncSetAttribute", u64::from(bytes))?;
    // SAFETY: the caller guarantees the handle is live.
    check("cuFuncSetAttribute", unsafe {
        cu::cuFuncSetAttribute(
            func.raw() as cu::CUfunction,
            cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes,
        )
    })
}

/// Threads per block this entry point may be launched with.
///
/// The driver's own answer, which is the `__launch_bounds__` the unit was
/// compiled with when there is one and the device maximum when there is not -
/// and which is therefore a property of the COMPILED entry rather than of the
/// device. That is what the caller wants: a variant compiled for 1024 lanes on
/// a part that allows 1024 is viable, and the same variant on a part that
/// allows 512 is not, and neither question can be answered from the device
/// limit alone.
///
/// # Safety
///
/// `func` must be live.
///
/// # Errors
///
/// [`SysError::Driver`] when the device cannot answer for this entry point.
pub(crate) unsafe fn func_max_threads_per_block(func: CuFunction) -> SysResult<u32> {
    const OP: &str = "cuFuncGetAttribute";
    let mut value: c_int = 0;
    // SAFETY: the caller guarantees the handle is live; `value` is a live local
    // of the type the driver writes.
    check(OP, unsafe {
        cu::cuFuncGetAttribute(
            &raw mut value,
            cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            func.raw() as cu::CUfunction,
        )
    })?;
    u32::try_from(value).map_err(|_| SysError::Host {
        op: OP,
        what: "the device reported a negative thread limit",
    })
}

/// Blocks of `func` that stay resident on one multiprocessor.
///
/// # Safety
///
/// `func` must be live.
///
/// # Errors
///
/// [`SysError::Driver`] when the device cannot answer for this entry point.
pub(crate) unsafe fn occupancy_max_active_blocks(
    func: CuFunction,
    block_threads: u32,
    smem: u32,
) -> SysResult<u32> {
    const OP: &str = "cuOccupancyMaxActiveBlocksPerMultiprocessor";
    let threads = as_int(OP, u64::from(block_threads))?;
    let smem = as_size(OP, u64::from(smem))?;
    let mut blocks: c_int = 0;
    // SAFETY: the caller guarantees the handle is live; `blocks` is a live
    // local of the type the driver writes.
    check(OP, unsafe {
        cu::cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &raw mut blocks,
            func.raw() as cu::CUfunction,
            threads,
            smem,
        )
    })?;
    u32::try_from(blocks).map_err(|_| SysError::Host {
        op: OP,
        what: "the device reported a negative occupancy",
    })
}

/// Queues an ordinary launch on `stream`.
///
/// Grid, block and shared-memory size go through untouched. Nothing here
/// divides, rounds up or reinterprets a dimension: the geometry a caller
/// computed is the geometry the device runs, because a backend that silently
/// adjusted one would make every kernel's index arithmetic a guess.
///
/// # Safety
///
/// `params` must hold exactly one pointer per declared kernel parameter, in
/// order, each addressing a live value of that parameter's type and size;
/// `func` and `stream` must be live. Every device address reachable from those
/// values must name memory the kernel is allowed to touch.
///
/// # Errors
///
/// [`SysError::Driver`] when the launch is refused. A fault *inside* the kernel
/// is reported by the next synchronise, not here.
pub(crate) unsafe fn launch_kernel(
    func: CuFunction,
    grid: Extent3,
    block: Extent3,
    smem: u32,
    stream: CuStream,
    params: &mut [*mut c_void],
) -> SysResult<()> {
    // SAFETY: the caller guarantees the parameter array matches the entry
    // point's signature and that both handles are live; `params` is a live
    // exclusive borrow for the whole call and the driver copies the argument
    // values out of it before returning.
    check("cuLaunchKernel", unsafe {
        cu::cuLaunchKernel(
            func.raw() as cu::CUfunction,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            smem,
            stream.raw() as cu::CUstream,
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
}

/// Queues a launch whose whole grid must be resident at once.
///
/// # Safety
///
/// As [`launch_kernel`], and the grid must additionally fit the device's
/// co-residency limit for this entry point — a grid that does not is refused
/// rather than run, but a grid that fits and whose blocks then wait on each
/// other incorrectly will hang the device.
///
/// # Errors
///
/// [`SysError::Driver`] when the launch is refused, which includes a grid the
/// device cannot make co-resident.
pub(crate) unsafe fn launch_cooperative(
    func: CuFunction,
    grid: Extent3,
    block: Extent3,
    smem: u32,
    stream: CuStream,
    params: &mut [*mut c_void],
) -> SysResult<()> {
    // SAFETY: as `launch_kernel`, plus the caller's co-residency guarantee.
    check("cuLaunchCooperativeKernel", unsafe {
        cu::cuLaunchCooperativeKernel(
            func.raw() as cu::CUfunction,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            smem,
            stream.raw() as cu::CUstream,
            params.as_mut_ptr(),
        )
    })
}

/// Puts `stream` into capture mode.
///
/// The mode is the strict one: any work submitted to another stream while this
/// capture is running fails the capture instead of being silently left out of
/// the graph. This backend has exactly one capture stream and no cross-stream
/// work, so the strict mode costs nothing and turns a whole class of "the graph
/// replayed but did less than the eager pass" bug into an immediate error.
///
/// # Safety
///
/// `stream` must be live and must not already be capturing.
///
/// # Errors
///
/// [`SysError::Driver`] when the stream cannot enter capture mode.
pub(crate) unsafe fn stream_begin_capture(stream: CuStream) -> SysResult<()> {
    // SAFETY: the caller guarantees the stream is live and not already
    // capturing.
    check("cuStreamBeginCapture", unsafe {
        cu::cuStreamBeginCapture_v2(
            stream.raw() as cu::CUstream,
            cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
        )
    })
}

/// Takes `stream` out of capture mode and yields what it captured.
///
/// # Safety
///
/// `stream` must be the stream a capture was begun on.
///
/// # Errors
///
/// [`SysError::Driver`] when the capture was invalidated — which is how the
/// strict capture mode reports illegal work — and [`SysError::Host`] when the
/// driver reports success but hands back nothing.
pub(crate) unsafe fn stream_end_capture(stream: CuStream) -> SysResult<CuGraph> {
    let mut graph: cu::CUgraph = std::ptr::null_mut();
    // SAFETY: the caller guarantees the stream is the capturing one; `graph` is
    // a live local of the type the driver writes.
    check("cuStreamEndCapture", unsafe {
        cu::cuStreamEndCapture(stream.raw() as cu::CUstream, &raw mut graph)
    })?;
    if graph.is_null() {
        return Err(SysError::Host {
            op: "cuStreamEndCapture",
            what: "the capture produced no graph",
        });
    }
    Ok(CuGraph::from_raw(graph as usize))
}

/// Turns a captured graph into a replayable one.
///
/// # Safety
///
/// `graph` must be live.
///
/// # Errors
///
/// [`SysError::Driver`] when the graph cannot be instantiated.
pub(crate) unsafe fn graph_instantiate(graph: CuGraph) -> SysResult<CuGraphExec> {
    let mut exec: cu::CUgraphExec = std::ptr::null_mut();
    // SAFETY: the caller guarantees the graph is live; `exec` is a live local
    // of the type the driver writes. No flags: this backend never auto-frees
    // or uploads a graph at instantiation.
    check("cuGraphInstantiateWithFlags", unsafe {
        cu::cuGraphInstantiateWithFlags(&raw mut exec, graph.raw() as cu::CUgraph, 0)
    })?;
    Ok(CuGraphExec::from_raw(exec as usize))
}

/// Releases a captured graph.
///
/// Safe to call as soon as the graph has been instantiated: the instantiated
/// copy does not borrow from it.
///
/// # Safety
///
/// `graph` must be live and must not be destroyed twice.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver refuses the destroy.
pub(crate) unsafe fn graph_destroy(graph: CuGraph) -> SysResult<()> {
    // SAFETY: the caller guarantees the handle is live and unreleased.
    check("cuGraphDestroy", unsafe {
        cu::cuGraphDestroy(graph.raw() as cu::CUgraph)
    })
}

/// Releases an instantiated graph.
///
/// # Safety
///
/// `exec` must be live and must not be destroyed twice.
///
/// # Errors
///
/// [`SysError::Driver`] when the driver refuses the destroy.
pub(crate) unsafe fn graph_exec_destroy(exec: CuGraphExec) -> SysResult<()> {
    // SAFETY: the caller guarantees the handle is live and unreleased.
    check("cuGraphExecDestroy", unsafe {
        cu::cuGraphExecDestroy(exec.raw() as cu::CUgraphExec)
    })
}

/// Replays an instantiated graph on `stream`.
///
/// # Safety
///
/// `exec` and `stream` must be live, and every buffer the graph names must
/// still be allocated at the address it was captured with. A graph bakes its
/// addresses in at capture: nothing updates them afterwards, so a replay after
/// a reallocation reads whatever now lives there.
///
/// # Errors
///
/// [`SysError::Driver`] when the replay cannot be queued.
pub(crate) unsafe fn graph_launch(exec: CuGraphExec, stream: CuStream) -> SysResult<()> {
    // SAFETY: the caller guarantees both handles are live and that the graph's
    // baked-in addresses are still valid.
    check("cuGraphLaunch", unsafe {
        cu::cuGraphLaunch(exec.raw() as cu::CUgraphExec, stream.raw() as cu::CUstream)
    })
}

/// Opens a classic math-library handle.
///
/// # Errors
///
/// [`SysError::Math`] when the library refuses a handle.
pub(crate) fn blas_create() -> SysResult<CuBlas> {
    let mut handle: blas::cublasHandle_t = std::ptr::null_mut();
    // SAFETY: `handle` is a live local of the type the library writes.
    check_blas("cublasCreate", unsafe {
        blas::cublasCreate_v2(&raw mut handle)
    })?;
    Ok(CuBlas::from_raw(handle as usize))
}

/// Closes a classic math-library handle.
///
/// # Safety
///
/// `handle` must be live and must not be destroyed twice.
///
/// # Errors
///
/// [`SysError::Math`] when the library refuses the destroy.
pub(crate) unsafe fn blas_destroy(handle: CuBlas) -> SysResult<()> {
    // SAFETY: the caller guarantees the handle is live and unreleased.
    check_blas("cublasDestroy", unsafe {
        blas::cublasDestroy_v2(handle.raw() as blas::cublasHandle_t)
    })
}

/// Binds the stream every later call on `handle` runs on.
///
/// This is what lets a matrix multiply be part of a captured graph: the library
/// records its work on whatever stream it is bound to, so a capture that did
/// not rebind it first would either miss the multiply or fail outright.
///
/// # Safety
///
/// `handle` and `stream` must be live.
///
/// # Errors
///
/// [`SysError::Math`] when the library refuses the binding.
pub(crate) unsafe fn blas_set_stream(handle: CuBlas, stream: CuStream) -> SysResult<()> {
    // SAFETY: the caller guarantees both handles are live.
    check_blas("cublasSetStream", unsafe {
        blas::cublasSetStream_v2(
            handle.raw() as blas::cublasHandle_t,
            stream.raw() as blas::cudaStream_t,
        )
    })
}

/// Chooses between reduced-precision and full-precision accumulation.
///
/// # Safety
///
/// `handle` must be live.
///
/// # Errors
///
/// [`SysError::Math`] when the library refuses the mode.
pub(crate) unsafe fn blas_set_math_mode(handle: CuBlas, tf32: bool) -> SysResult<()> {
    let mode = if tf32 {
        blas::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH
    } else {
        blas::cublasMath_t::CUBLAS_DEFAULT_MATH
    };
    // SAFETY: the caller guarantees the handle is live.
    check_blas("cublasSetMathMode", unsafe {
        blas::cublasSetMathMode(handle.raw() as blas::cublasHandle_t, mode)
    })
}

/// The library's spelling of an operand's transposition.
const fn blas_op(op: Op) -> blas::cublasOperation_t {
    match op {
        Op::N => blas::cublasOperation_t::CUBLAS_OP_N,
        Op::T => blas::cublasOperation_t::CUBLAS_OP_T,
    }
}

/// The library's spelling of an operand's element type.
const fn blas_dtype(dtype: MathType) -> blas::cudaDataType {
    match dtype {
        MathType::F32 => blas::cudaDataType::CUDA_R_32F,
        MathType::F16 => blas::cudaDataType::CUDA_R_16F,
        MathType::Bf16 => blas::cudaDataType::CUDA_R_16BF,
    }
}

/// The element type the *accumulator* is written back as.
///
/// Always single precision, for the same reason [`blas_compute`] always is: the
/// operand type says how the inputs are stored and the compute type says what
/// the multiply accumulates in, and the destination of every multiply this
/// engine issues is a single-precision activation. Handing the library the
/// operand type here instead is not a refusal - it is a successful call that
/// writes halves into a float buffer, so a chunk wide enough to reach this tier
/// comes back as noise and a narrow one, which never reaches it, does not.
///
/// The one exception is a call that asked for a half-precision destination
/// outright, which the engine makes only where the buffer it writes is read as
/// halves.
const fn blas_out_dtype(out_half: bool) -> blas::cudaDataType {
    if out_half {
        blas::cudaDataType::CUDA_R_16F
    } else {
        blas::cudaDataType::CUDA_R_32F
    }
}

/// The accumulator this call asks for.
///
/// Always single precision. `tf32` only chooses whether single-precision
/// *operands* may be rounded on their way into the multiplier; it never changes
/// what the accumulator is, and half and brain-float operands accumulate in
/// single precision either way.
const fn blas_compute(dtype: MathType, tf32: bool) -> blas::cublasComputeType_t {
    match dtype {
        MathType::F32 if tf32 => blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
        MathType::F32 | MathType::F16 | MathType::Bf16 => {
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F
        }
    }
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
/// [`SysError::Math`] when the library refuses the call.
pub(crate) unsafe fn blas_gemm(handle: CuBlas, call: &GemmCall) -> SysResult<()> {
    let alpha = call.alpha;
    let beta = call.beta;
    // SAFETY: the caller guarantees every device range; `alpha` and `beta` are
    // live locals the library reads during the call, and the handle is live.
    check_blas("cublasGemmEx", unsafe {
        blas::cublasGemmEx(
            handle.raw() as blas::cublasHandle_t,
            blas_op(call.op_a),
            blas_op(call.op_b),
            call.m,
            call.n,
            call.k,
            (&raw const alpha).cast::<c_void>(),
            call.a.raw() as *const c_void,
            blas_dtype(call.dtype),
            call.lda,
            call.b.raw() as *const c_void,
            blas_dtype(call.dtype),
            call.ldb,
            (&raw const beta).cast::<c_void>(),
            call.c.raw() as *mut c_void,
            blas_out_dtype(call.out_half),
            call.ldc,
            blas_compute(call.dtype, call.tf32),
            blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    })
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
/// [`SysError::Math`] when the library refuses the call.
pub(crate) unsafe fn blas_gemm_strided(
    handle: CuBlas,
    call: &GemmCall,
    stride: &GemmStride,
) -> SysResult<()> {
    let alpha = call.alpha;
    let beta = call.beta;
    // SAFETY: as `blas_gemm`, and the caller additionally guarantees every
    // batch element stays inside its allocation.
    check_blas("cublasGemmStridedBatchedEx", unsafe {
        blas::cublasGemmStridedBatchedEx(
            handle.raw() as blas::cublasHandle_t,
            blas_op(call.op_a),
            blas_op(call.op_b),
            call.m,
            call.n,
            call.k,
            (&raw const alpha).cast::<c_void>(),
            call.a.raw() as *const c_void,
            blas_dtype(call.dtype),
            call.lda,
            stride.a,
            call.b.raw() as *const c_void,
            blas_dtype(call.dtype),
            call.ldb,
            stride.b,
            (&raw const beta).cast::<c_void>(),
            call.c.raw() as *mut c_void,
            blas_out_dtype(call.out_half),
            call.ldc,
            stride.c,
            stride.count,
            blas_compute(call.dtype, call.tf32),
            blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    })
}

/// Opens a heuristic-driven math-library handle.
///
/// # Errors
///
/// [`SysError::Math`] when the library refuses a handle.
pub(crate) fn blas_lt_create() -> SysResult<CuBlasLt> {
    let mut handle: lt::cublasLtHandle_t = std::ptr::null_mut();
    // SAFETY: `handle` is a live local of the type the library writes.
    check_lt("cublasLtCreate", unsafe {
        lt::cublasLtCreate(&raw mut handle)
    })?;
    Ok(CuBlasLt::from_raw(handle as usize))
}

/// Closes a heuristic-driven math-library handle.
///
/// # Safety
///
/// `handle` must be live and must not be destroyed twice.
///
/// # Errors
///
/// [`SysError::Math`] when the library refuses the destroy.
pub(crate) unsafe fn blas_lt_destroy(handle: CuBlasLt) -> SysResult<()> {
    algo_cache().lock().clear();
    pin_cache().lock().clear();
    // SAFETY: the caller guarantees the handle is live and unreleased.
    check_lt("cublasLtDestroy", unsafe {
        lt::cublasLtDestroy(handle.raw() as lt::cublasLtHandle_t)
    })
}

/// What makes two heuristic queries the same question.
///
/// The library's own answer depends on nothing else: same shapes, same
/// transpositions, same element type, same workspace budget means the same
/// best algorithm. Caching on exactly that is what removes a heuristic query
/// from every decode step without ever returning an algorithm chosen for a
/// different problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AlgoKey {
    op_a: bool,
    op_b: bool,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    dtype: u32,
    tf32: bool,
    workspace: u64,
}

/// Largest number of distinct problem shapes whose chosen algorithm is kept.
///
/// A decode loop asks about a handful of shapes forever; a prefill loop asks
/// about a few more. The bound exists so that a workload that keeps inventing
/// shapes cannot grow this without limit, and it is enforced by clearing rather
/// than evicting one entry, because the next few queries repopulate it and a
/// least-recently-used list would cost more than the query it saves.
const ALGO_CACHE_CAP: usize = 256;

/// The process-wide algorithm cache.
type AlgoCache = Mutex<HashMap<AlgoKey, lt::cublasLtMatmulAlgo_t>>;

/// The cache, created on first use.
fn algo_cache() -> &'static AlgoCache {
    static CACHE: std::sync::OnceLock<AlgoCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Owns one heuristic-path descriptor for the length of a call.
///
/// The library's descriptors are created and destroyed explicitly, and a `?`
/// between the two would leak one on every failure. Each of these releases its
/// own on drop, so the failure paths need no cleanup at all.
struct LtDesc(lt::cublasLtMatmulDesc_t);

impl Drop for LtDesc {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `cublasLtMatmulDescCreate` and this
        // is the only owner, so it is live and has not been destroyed.
        let status = unsafe { lt::cublasLtMatmulDescDestroy(self.0) };
        if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            tracing::error!(status = status as i32, "matmul descriptor destroy failed");
        }
    }
}

/// Owns one heuristic-path matrix layout for the length of a call.
struct LtLayout(lt::cublasLtMatrixLayout_t);

impl Drop for LtLayout {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `cublasLtMatrixLayoutCreate` and
        // this is the only owner, so it is live and has not been destroyed.
        let status = unsafe { lt::cublasLtMatrixLayoutDestroy(self.0) };
        if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            tracing::error!(status = status as i32, "matrix layout destroy failed");
        }
    }
}

/// Owns one heuristic-path preference block for the length of a call.
struct LtPreference(lt::cublasLtMatmulPreference_t);

impl Drop for LtPreference {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `cublasLtMatmulPreferenceCreate`
        // and this is the only owner, so it is live and has not been destroyed.
        let status = unsafe { lt::cublasLtMatmulPreferenceDestroy(self.0) };
        if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            tracing::error!(status = status as i32, "matmul preference destroy failed");
        }
    }
}

/// The heuristic library's spelling of an operand's element type.
const fn lt_dtype(dtype: MathType) -> lt::cudaDataType {
    match dtype {
        MathType::F32 => lt::cudaDataType::CUDA_R_32F,
        MathType::F16 => lt::cudaDataType::CUDA_R_16F,
        MathType::Bf16 => lt::cudaDataType::CUDA_R_16BF,
    }
}

/// The accumulator the heuristic path asks for; see [`blas_compute`].
const fn lt_compute(dtype: MathType, tf32: bool) -> lt::cublasComputeType_t {
    match dtype {
        MathType::F32 if tf32 => lt::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
        MathType::F32 | MathType::F16 | MathType::Bf16 => {
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F
        }
    }
}

/// Builds the operation descriptor, both layouts and the accumulator layout.
///
/// Split out of [`blas_lt_matmul`] so the guards above have one obvious scope
/// and the call itself stays readable.
fn lt_setup(call: &GemmCall) -> SysResult<(LtDesc, LtLayout, LtLayout, LtLayout)> {
    let scale = lt::cudaDataType::CUDA_R_32F;
    let mut desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
    // SAFETY: `desc` is a live local of the type the library writes.
    check_lt("cublasLtMatmulDescCreate", unsafe {
        lt::cublasLtMatmulDescCreate(&raw mut desc, lt_compute(call.dtype, call.tf32), scale)
    })?;
    let desc = LtDesc(desc);

    for (attr, op) in [
        (
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            call.op_a,
        ),
        (
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            call.op_b,
        ),
    ] {
        let value: i32 = match op {
            Op::N => 0,
            Op::T => 1,
        };
        // SAFETY: `desc.0` is live and owned here; `value` is a live local of
        // exactly the width declared, which the library copies during the call.
        check_lt("cublasLtMatmulDescSetAttribute", unsafe {
            lt::cublasLtMatmulDescSetAttribute(
                desc.0,
                attr,
                (&raw const value).cast::<c_void>(),
                size_of::<i32>(),
            )
        })?;
    }

    // Column-major layouts, so the stored extents of an operand are its
    // pre-transposition shape: the transposition flags above are what tell the
    // library how to read them.
    let (a_rows, a_cols) = match call.op_a {
        Op::N => (call.m, call.k),
        Op::T => (call.k, call.m),
    };
    let (b_rows, b_cols) = match call.op_b {
        Op::N => (call.k, call.n),
        Op::T => (call.n, call.k),
    };
    let a = lt_layout(call.dtype, a_rows, a_cols, call.lda)?;
    let b = lt_layout(call.dtype, b_rows, b_cols, call.ldb)?;
    // Single precision unless the call asked for halves: see `blas_out_dtype`.
    let out = if call.out_half {
        MathType::F16
    } else {
        MathType::F32
    };
    let c = lt_layout(out, call.m, call.n, call.ldc)?;
    Ok((desc, a, b, c))
}

/// One matrix layout of the given element type, extents and leading dimension.
fn lt_layout(dtype: MathType, rows: i32, cols: i32, ld: i32) -> SysResult<LtLayout> {
    let rows = u64::try_from(rows).map_err(|_| SysError::Host {
        op: "cublasLtMatrixLayoutCreate",
        what: "a matrix extent is negative",
    })?;
    let cols = u64::try_from(cols).map_err(|_| SysError::Host {
        op: "cublasLtMatrixLayoutCreate",
        what: "a matrix extent is negative",
    })?;
    let mut layout: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
    // SAFETY: `layout` is a live local of the type the library writes.
    check_lt("cublasLtMatrixLayoutCreate", unsafe {
        lt::cublasLtMatrixLayoutCreate(&raw mut layout, lt_dtype(dtype), rows, cols, i64::from(ld))
    })?;
    Ok(LtLayout(layout))
}

/// Queues one matrix multiply through the heuristic path.
///
/// The library is asked which algorithm to use once per problem shape and the
/// answer is remembered; a decode loop runs the same handful of shapes for the
/// life of the process, so re-asking every step would spend measurable time
/// re-deriving an answer that cannot have changed.
///
/// # Safety
///
/// As [`blas_gemm`], and `workspace .. workspace + workspace_bytes` must be a
/// live allocation this call may overwrite.
///
/// # Errors
///
/// [`SysError::Math`] when the library refuses any step, and
/// [`SysError::Host`] when it reports no usable algorithm for this shape.
pub(crate) unsafe fn blas_lt_matmul(
    handle: CuBlasLt,
    call: &GemmCall,
    workspace: DevicePtr,
    workspace_bytes: u64,
    stream: CuStream,
) -> SysResult<()> {
    let lt_handle = handle.raw() as lt::cublasLtHandle_t;
    let (desc, a, b, c) = lt_setup(call)?;
    let key = AlgoKey {
        op_a: matches!(call.op_a, Op::T),
        op_b: matches!(call.op_b, Op::T),
        m: call.m,
        n: call.n,
        k: call.k,
        lda: call.lda,
        ldb: call.ldb,
        ldc: call.ldc,
        dtype: lt_dtype(call.dtype) as u32,
        tf32: call.tf32,
        workspace: workspace_bytes,
    };

    let algo = if let Some(algo) = algo_cache().lock().get(&key).copied() {
        algo
    } else {
        // SAFETY: every handle passed here is live and owned by this call.
        let algo = unsafe { lt_heuristic(lt_handle, &desc, &a, &b, &c, workspace_bytes) }?;
        let mut cache = algo_cache().lock();
        if cache.len() >= ALGO_CACHE_CAP {
            tracing::debug!(
                cap = ALGO_CACHE_CAP,
                "matmul algorithm cache full; clearing"
            );
            cache.clear();
        }
        cache.insert(key, algo);
        algo
    };

    let alpha = call.alpha;
    let beta = call.beta;
    let workspace_size = as_size("cublasLtMatmul", workspace_bytes)?;
    // SAFETY: the caller guarantees every device range and the workspace;
    // `alpha`, `beta` and `algo` are live locals the library reads during the
    // call, and the descriptors live until this function returns.
    check_lt("cublasLtMatmul", unsafe {
        lt::cublasLtMatmul(
            lt_handle,
            desc.0,
            (&raw const alpha).cast::<c_void>(),
            call.a.raw() as *const c_void,
            a.0,
            call.b.raw() as *const c_void,
            b.0,
            (&raw const beta).cast::<c_void>(),
            call.c.raw() as *const c_void,
            c.0,
            call.c.raw() as *mut c_void,
            c.0,
            &raw const algo,
            workspace.raw() as *mut c_void,
            workspace_size,
            stream.raw() as lt::cudaStream_t,
        )
    })
}

/// The column count a pinned algorithm is chosen at.
///
/// Any fixed number keeps the promise - what matters is that the choice never
/// sees the caller's own column count - and this one is a whole prefill chunk,
/// the shape the pinned tier spends nearly all of its time on.
const PIN_REFERENCE_COLUMNS: i32 = 4_096;

/// What makes two pinned multiplies the same question: [`AlgoKey`] without the
/// column count or a workspace, and with the output precision, which the
/// ordinary key reads off the element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PinKey {
    op_a: Op,
    op_b: Op,
    m: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    dtype: u32,
    tf32: bool,
    out_half: bool,
}

/// The process-wide pinned-algorithm table. Unbounded, because it holds one
/// entry per projection shape and a model has a few dozen.
type PinCache = Mutex<HashMap<PinKey, lt::cublasLtMatmulAlgo_t>>;

/// The table, created on first use.
fn pin_cache() -> &'static PinCache {
    static CACHE: std::sync::OnceLock<PinCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Sets one attribute of an algorithm's configuration.
fn algo_set(
    algo: &mut lt::cublasLtMatmulAlgo_t,
    attr: lt::cublasLtMatmulAlgoConfigAttributes_t,
    value: u32,
) -> SysResult<()> {
    // SAFETY: `algo` is a live, initialised algorithm the caller owns, and
    // `value` a live local of exactly the width declared, copied by the call.
    check_lt("cublasLtMatmulAlgoConfigSetAttribute", unsafe {
        lt::cublasLtMatmulAlgoConfigSetAttribute(
            &raw mut *algo,
            attr,
            (&raw const value).cast::<c_void>(),
            size_of::<u32>(),
        )
    })
}

/// The algorithm every multiply of `call`'s shape takes, whatever its column
/// count: the library's choice at [`PIN_REFERENCE_COLUMNS`] columns with no
/// workspace, then held to one slice of the contraction.
///
/// # Safety
///
/// `handle` must be live.
unsafe fn pinned_algo(
    handle: lt::cublasLtHandle_t,
    call: &GemmCall,
) -> SysResult<lt::cublasLtMatmulAlgo_t> {
    let key = PinKey {
        op_a: call.op_a,
        op_b: call.op_b,
        m: call.m,
        k: call.k,
        lda: call.lda,
        ldb: call.ldb,
        ldc: call.ldc,
        dtype: lt_dtype(call.dtype) as u32,
        tf32: call.tf32,
        out_half: call.out_half,
    };
    if let Some(algo) = pin_cache().lock().get(&key).copied() {
        return Ok(algo);
    }
    // The reference problem is described, never run, so a transposed right
    // operand's leading dimension only has to be wide enough to describe it.
    let reference = GemmCall {
        n: PIN_REFERENCE_COLUMNS,
        ldb: match call.op_b {
            Op::N => call.ldb,
            Op::T => call.ldb.max(PIN_REFERENCE_COLUMNS),
        },
        ..*call
    };
    let (desc, a, b, c) = lt_setup(&reference)?;
    // No workspace: a split contraction reduces through one, and an algorithm
    // that cannot run without it is not one this tier may take.
    // SAFETY: every handle is live for the call.
    let mut algo = unsafe { lt_heuristic(handle, &desc, &a, &b, &c, 0) }?;
    algo_set(
        &mut algo,
        lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        1,
    )?;
    algo_set(
        &mut algo,
        lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as u32,
    )?;
    pin_cache().lock().insert(key, algo);
    Ok(algo)
}

/// Queues one matrix multiply through the heuristic path with its algorithm
/// pinned per shape, so a column sums the same way at every column count.
///
/// # Safety
///
/// As [`blas_gemm`].
///
/// # Errors
///
/// [`SysError::Math`] when the library refuses any step - including when the
/// pinned algorithm cannot run at this column count, which is a refusal rather
/// than a quiet change of algorithm - and [`SysError::Host`] when it offers
/// none for the reference shape.
pub(crate) unsafe fn blas_lt_matmul_pinned(
    handle: CuBlasLt,
    call: &GemmCall,
    stream: CuStream,
) -> SysResult<()> {
    let lt_handle = handle.raw() as lt::cublasLtHandle_t;
    // SAFETY: the caller guarantees the handle is live.
    let algo = unsafe { pinned_algo(lt_handle, call) }?;
    let (desc, a, b, c) = lt_setup(call)?;
    let mut fit = std::mem::MaybeUninit::<lt::cublasLtMatmulHeuristicResult_t>::zeroed();
    // SAFETY: every handle is live; `algo` and `fit` are live locals of the
    // types the library reads and writes.
    check_lt("cublasLtMatmulAlgoCheck", unsafe {
        lt::cublasLtMatmulAlgoCheck(
            lt_handle,
            desc.0,
            a.0,
            b.0,
            c.0,
            c.0,
            &raw const algo,
            fit.as_mut_ptr(),
        )
    })?;
    // SAFETY: the check succeeded, so the library filled the result.
    let fit = unsafe { fit.assume_init() };
    if fit.workspaceSize != 0 {
        return Err(SysError::Host {
            op: "cublasLtMatmulAlgoCheck",
            what: "the pinned algorithm asks for a workspace at this column count",
        });
    }
    let alpha = call.alpha;
    let beta = call.beta;
    // SAFETY: the caller guarantees every device range; `alpha`, `beta` and
    // `algo` are live locals the library reads during the call, and the
    // descriptors live until this function returns.
    check_lt("cublasLtMatmul", unsafe {
        lt::cublasLtMatmul(
            lt_handle,
            desc.0,
            (&raw const alpha).cast::<c_void>(),
            call.a.raw() as *const c_void,
            a.0,
            call.b.raw() as *const c_void,
            b.0,
            (&raw const beta).cast::<c_void>(),
            call.c.raw() as *const c_void,
            c.0,
            call.c.raw() as *mut c_void,
            c.0,
            &raw const algo,
            std::ptr::null_mut(),
            0,
            stream.raw() as lt::cudaStream_t,
        )
    })
}

/// Asks the library which algorithm fits this problem best.
///
/// # Safety
///
/// Every handle must be live for the duration of the call.
unsafe fn lt_heuristic(
    handle: lt::cublasLtHandle_t,
    desc: &LtDesc,
    a: &LtLayout,
    b: &LtLayout,
    c: &LtLayout,
    workspace_bytes: u64,
) -> SysResult<lt::cublasLtMatmulAlgo_t> {
    let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
    // SAFETY: `pref` is a live local of the type the library writes.
    check_lt("cublasLtMatmulPreferenceCreate", unsafe {
        lt::cublasLtMatmulPreferenceCreate(&raw mut pref)
    })?;
    let pref = LtPreference(pref);
    let budget = workspace_bytes;
    // SAFETY: `pref.0` is live and owned here; `budget` is a live local of
    // exactly the width declared, which the library copies during the call.
    check_lt("cublasLtMatmulPreferenceSetAttribute", unsafe {
        lt::cublasLtMatmulPreferenceSetAttribute(
            pref.0,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&raw const budget).cast::<c_void>(),
            size_of::<u64>(),
        )
    })?;

    let mut result = std::mem::MaybeUninit::<lt::cublasLtMatmulHeuristicResult_t>::zeroed();
    let mut found: c_int = 0;
    // SAFETY: every handle is live; `result` and `found` are live locals of the
    // types the library writes, and exactly one result is requested.
    check_lt("cublasLtMatmulAlgoGetHeuristic", unsafe {
        lt::cublasLtMatmulAlgoGetHeuristic(
            handle,
            desc.0,
            a.0,
            b.0,
            c.0,
            c.0,
            pref.0,
            1,
            result.as_mut_ptr(),
            &raw mut found,
        )
    })?;
    if found < 1 {
        return Err(SysError::Host {
            op: "cublasLtMatmulAlgoGetHeuristic",
            what: "the library found no algorithm for this problem shape",
        });
    }
    // SAFETY: the library reported at least one result, so it initialised the
    // first element of the array it was given.
    let result = unsafe { result.assume_init() };
    Ok(result.algo)
}

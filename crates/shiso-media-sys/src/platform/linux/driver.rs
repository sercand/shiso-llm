//! The two driver libraries, loaded once at run time, and the primary context.
//!
//! Nothing links against `libcuda` or `libnvcuvid`: a machine without the NVIDIA
//! driver gets [`HwError::Unavailable`] from the first `open`, the same binary
//! runs on both, and the macOS no-libcuda check is unaffected. The libraries are
//! loaded on first use into a process-wide static and never unloaded, which is
//! what keeps the function pointers copied out of them valid for the life of the
//! process.

use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

use libloading::Library;

use super::ffi::{
    CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CuContext, CuCtxPopCurrent, CuCtxPushCurrent,
    CuDevice, CuDeviceGet, CuDevicePrimaryCtxRelease, CuDevicePrimaryCtxRetain, CuDevicePtr,
    CuInit, CuMemcpy2d, CudaMemcpy2d, CuvidCreateDecoder, CuvidCreateVideoParser, CuvidDecodeCaps,
    CuvidDecodePicture, CuvidDestroyDecoder, CuvidDestroyVideoParser, CuvidGetDecoderCaps,
    CuvidMapVideoFrame64, CuvidParseVideoData, CuvidUnmapVideoFrame64,
};
use super::surface::check;
use crate::{HwCodec, HwError};

/// The driver entry points this crate calls, and the libraries they live in.
#[derive(Debug)]
pub(super) struct Driver {
    pub(super) cu_init: CuInit,
    pub(super) cu_device_get: CuDeviceGet,
    pub(super) cu_primary_ctx_retain: CuDevicePrimaryCtxRetain,
    pub(super) cu_primary_ctx_release: CuDevicePrimaryCtxRelease,
    pub(super) cu_ctx_push: CuCtxPushCurrent,
    pub(super) cu_ctx_pop: CuCtxPopCurrent,
    pub(super) cu_memcpy_2d: CuMemcpy2d,
    pub(super) get_decoder_caps: CuvidGetDecoderCaps,
    pub(super) create_decoder: CuvidCreateDecoder,
    pub(super) destroy_decoder: CuvidDestroyDecoder,
    pub(super) decode_picture: CuvidDecodePicture,
    pub(super) map_frame: CuvidMapVideoFrame64,
    pub(super) unmap_frame: CuvidUnmapVideoFrame64,
    pub(super) create_parser: CuvidCreateVideoParser,
    pub(super) parse_data: CuvidParseVideoData,
    pub(super) destroy_parser: CuvidDestroyVideoParser,
    /// Held so the pointers above stay valid; never dropped (see the module doc).
    _cuda: Library,
    _nvcuvid: Library,
}

/// The process's driver, loaded and initialised on first use. The outcome,
/// failure included, is cached: a machine without the driver does not retry the
/// `dlopen` on every request.
static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();

/// The loaded driver.
///
/// # Errors
///
/// [`HwError::Unavailable`] when either library or any entry point is missing,
/// or `cuInit` fails.
pub(super) fn driver() -> Result<&'static Driver, HwError> {
    DRIVER
        .get_or_init(Driver::load)
        .as_ref()
        .map_err(|reason| HwError::Unavailable {
            reason: reason.clone(),
        })
}

/// Opens a shared library by its versioned soname, the name the driver package
/// installs (the unversioned name is a development symlink).
fn open(name: &str) -> Result<Library, String> {
    // SAFETY: loading runs the library's initialisers. These are NVIDIA's driver
    // libraries, loaded by the same name every CUDA program loads them by, and
    // their initialisers have no preconditions on the caller.
    unsafe { Library::new(name) }.map_err(|err| format!("cannot load {name}: {err}"))
}

/// Copies an entry point out of `lib`.
///
/// # Safety
///
/// `T` must be the function pointer type the library's header declares for
/// `name`, and the pointer must not be called after `lib` is dropped.
unsafe fn symbol<T: Copy>(lib: &Library, name: &str) -> Result<T, String> {
    // SAFETY: the caller states `T` is the symbol's declared type.
    unsafe { lib.get::<T>(name) }
        .map(|sym| *sym)
        .map_err(|err| format!("the NVIDIA driver has no {name}: {err}"))
}

impl Driver {
    fn load() -> Result<Self, String> {
        let cuda = open("libcuda.so.1")?;
        let nvcuvid = open("libnvcuvid.so.1")?;
        // SAFETY: every type below is the ffi module's transcription of the
        // header declaration for that exact name (the `_v2` names are the ones
        // the header's macros select for 64-bit builds), and the libraries are
        // moved into the returned value, which lives in a static that is never
        // dropped, so no pointer outlives its library.
        let driver = unsafe {
            Self {
                cu_init: symbol(&cuda, "cuInit")?,
                cu_device_get: symbol(&cuda, "cuDeviceGet")?,
                cu_primary_ctx_retain: symbol(&cuda, "cuDevicePrimaryCtxRetain")?,
                cu_primary_ctx_release: symbol(&cuda, "cuDevicePrimaryCtxRelease_v2")?,
                cu_ctx_push: symbol(&cuda, "cuCtxPushCurrent_v2")?,
                cu_ctx_pop: symbol(&cuda, "cuCtxPopCurrent_v2")?,
                cu_memcpy_2d: symbol(&cuda, "cuMemcpy2D_v2")?,
                get_decoder_caps: symbol(&nvcuvid, "cuvidGetDecoderCaps")?,
                create_decoder: symbol(&nvcuvid, "cuvidCreateDecoder")?,
                destroy_decoder: symbol(&nvcuvid, "cuvidDestroyDecoder")?,
                decode_picture: symbol(&nvcuvid, "cuvidDecodePicture")?,
                map_frame: symbol(&nvcuvid, "cuvidMapVideoFrame64")?,
                unmap_frame: symbol(&nvcuvid, "cuvidUnmapVideoFrame64")?,
                create_parser: symbol(&nvcuvid, "cuvidCreateVideoParser")?,
                parse_data: symbol(&nvcuvid, "cuvidParseVideoData")?,
                destroy_parser: symbol(&nvcuvid, "cuvidDestroyVideoParser")?,
                _cuda: cuda,
                _nvcuvid: nvcuvid,
            }
        };
        // SAFETY: cuInit takes a flags word that must be 0 and is safe to call
        // more than once per process (shiso-cuda may already have called it).
        let rc = unsafe { (driver.cu_init)(0) };
        check("cuInit", rc, None).map_err(|err| err.to_string())?;
        Ok(driver)
    }

    /// Asks the decoder what it can do for `caps`' codec, chroma format and bit
    /// depth, filling the rest of `caps`. The primary context must be current on
    /// this thread (a [`Current`] guard, or inside a parser callback).
    ///
    /// # Errors
    ///
    /// The mapped driver error.
    pub(super) fn decoder_caps(
        &self,
        codec: HwCodec,
        caps: &mut CuvidDecodeCaps,
    ) -> Result<(), HwError> {
        // SAFETY: `caps` is a valid, exclusively borrowed CUVIDDECODECAPS for the
        // call.
        let rc = unsafe { (self.get_decoder_caps)(caps) };
        check("cuvidGetDecoderCaps", rc, Some(codec))
    }

    /// Copies `rows` rows of `row_bytes` bytes, `pitch` apart on the device from
    /// `src`, into `dst` packed. The primary context must be current on this
    /// thread.
    ///
    /// # Errors
    ///
    /// [`HwError::Failed`] when `dst` is too small or the copy fails.
    pub(super) fn copy_to_host(
        &self,
        src: CuDevicePtr,
        pitch: usize,
        row_bytes: usize,
        rows: usize,
        dst: &mut [u8],
    ) -> Result<(), HwError> {
        let needed = row_bytes.checked_mul(rows);
        if needed != Some(dst.len()) || row_bytes > pitch {
            return Err(HwError::Failed {
                detail: format!(
                    "a {rows}x{row_bytes} copy at pitch {pitch} into {} bytes",
                    dst.len()
                ),
            });
        }
        let copy = CudaMemcpy2d {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: CU_MEMORYTYPE_DEVICE,
            src_host: ptr::null(),
            src_device: src,
            src_array: ptr::null_mut(),
            src_pitch: pitch,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: CU_MEMORYTYPE_HOST,
            dst_host: dst.as_mut_ptr().cast::<c_void>(),
            dst_device: 0,
            dst_array: ptr::null_mut(),
            dst_pitch: row_bytes,
            width_in_bytes: row_bytes,
            height: rows,
        };
        // SAFETY: the destination is `dst`, exclusively borrowed and exactly
        // `rows * row_bytes` long (checked above), written at pitch `row_bytes`;
        // the source is a mapped surface the caller holds mapped for the call.
        // The non-async copy returns once the host buffer is written.
        let rc = unsafe { (self.cu_memcpy_2d)(&raw const copy) };
        check("cuMemcpy2D_v2", rc, None)
    }
}

/// A retained reference to device 0's primary context.
///
/// Sharing the primary context (rather than creating one) means the decoder and
/// the compute backend live in one context on the device, which is what the
/// backend uses too; a second context would cost its own memory reservation and
/// time-slice against the first.
#[derive(Debug)]
pub(super) struct PrimaryContext {
    driver: &'static Driver,
    device: CuDevice,
    ctx: CuContext,
}

impl PrimaryContext {
    /// Retains device 0's primary context.
    ///
    /// # Errors
    ///
    /// [`HwError::Unavailable`] when there is no device; otherwise the mapped
    /// driver error.
    pub(super) fn retain(driver: &'static Driver) -> Result<Self, HwError> {
        let mut device: CuDevice = 0;
        // SAFETY: `device` is a valid out-pointer for the call.
        let rc = unsafe { (driver.cu_device_get)(&raw mut device, 0) };
        check("cuDeviceGet", rc, None)?;
        let mut ctx: CuContext = ptr::null_mut();
        // SAFETY: `ctx` is a valid out-pointer and `device` came from cuDeviceGet.
        let rc = unsafe { (driver.cu_primary_ctx_retain)(&raw mut ctx, device) };
        check("cuDevicePrimaryCtxRetain", rc, None)?;
        if ctx.is_null() {
            return Err(HwError::Failed {
                detail: "cuDevicePrimaryCtxRetain returned a null context".into(),
            });
        }
        Ok(Self {
            driver,
            device,
            ctx,
        })
    }

    /// Makes the context current on this thread until the guard drops.
    ///
    /// # Errors
    ///
    /// The mapped driver error.
    pub(super) fn push(&self) -> Result<Current<'_>, HwError> {
        // SAFETY: `self.ctx` is a retained primary context, valid until `self`
        // drops, which the returned guard's borrow prevents.
        let rc = unsafe { (self.driver.cu_ctx_push)(self.ctx) };
        check("cuCtxPushCurrent_v2", rc, None)?;
        Ok(Current { owner: self })
    }

    /// The loaded driver.
    pub(super) fn driver(&self) -> &'static Driver {
        self.driver
    }
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        // SAFETY: this value holds exactly one retain on `self.device`'s primary
        // context, taken in `retain`, and this is its only release.
        let rc = unsafe { (self.driver.cu_primary_ctx_release)(self.device) };
        if rc != super::ffi::CuResult::SUCCESS {
            tracing::warn!(code = rc.0, "cuDevicePrimaryCtxRelease_v2 failed");
        }
    }
}

/// Proof that the primary context is current on this thread; pops it on drop.
#[derive(Debug)]
pub(super) struct Current<'a> {
    owner: &'a PrimaryContext,
}

impl Drop for Current<'_> {
    fn drop(&mut self) {
        let mut popped: CuContext = ptr::null_mut();
        // SAFETY: this guard pushed the context on this thread (it is not Send,
        // holding a reference to a non-Sync value), so the pop takes exactly
        // that push off the stack.
        let rc = unsafe { (self.owner.driver.cu_ctx_pop)(&raw mut popped) };
        if rc != super::ffi::CuResult::SUCCESS {
            tracing::warn!(code = rc.0, "cuCtxPopCurrent_v2 failed");
        }
    }
}

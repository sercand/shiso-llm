//! The device the audio model tests run on: Metal on macOS, CUDA device 0 on
//! Linux. One helper, so a test says what it checks rather than which vendor
//! it checks it on, and the same case runs on both machines.

use std::sync::Arc;

use shiso_core::Backend;

/// The device, or `None` having said on stderr why there is none.
#[expect(
    clippy::print_stderr,
    reason = "a skipped device case says why on stderr"
)]
pub(crate) fn device() -> Option<Arc<dyn Backend>> {
    #[cfg(target_os = "macos")]
    {
        if !shiso_metal::metal_available() {
            eprintln!("SKIP: no Metal device");
            return None;
        }
        let metal: Arc<dyn Backend> =
            shiso_metal::MetalBackend::new().expect("the Metal library compiles");
        Some(metal)
    }
    #[cfg(target_os = "linux")]
    {
        match shiso_cuda::CudaBackend::new(0, shiso_cuda::CudaOptions::default()) {
            Ok(cuda) => {
                let cuda: Arc<dyn Backend> = cuda;
                Some(cuda)
            }
            Err(error) => {
                eprintln!("SKIP: no CUDA device: {error}");
                None
            }
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        eprintln!("SKIP: no device backend for this platform");
        None
    }
}

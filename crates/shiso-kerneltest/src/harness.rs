//! The device wrapper every shared case is written against.
//!
//! This is `crates/shiso-metal/tests/q35_harness.rs` with the one Metal-shaped
//! thing in it removed. Everything a case does goes through
//! [`shiso_core::Backend`] already, so the only thing that had to change is how
//! the device is opened: the backend opens it and hands it here, rather than
//! this file naming a backend and opening one itself.
//!
//! `f16` fixtures are written as explicit bit patterns and read back through
//! [`crate::oracle::f16_to_f32`], so nothing in the fixture path re-implements a
//! float conversion.

use std::any::Any;
use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer, KernelRef, LaunchArgs, alloc};

/// An open device, plus whatever concrete handle keeps its kernels loaded.
#[derive(Debug)]
pub struct Gpu {
    device: Arc<dyn Backend>,
    // Read by nothing. Dropping it would release the library or module set the
    // entry points were resolved out of, so the field is load-bearing even
    // though nothing reads it, and it is `Any` because this crate must not name
    // either backend: the dependency runs from the backends to here.
    #[expect(dead_code, reason = "a keep-alive; dropping it unloads the kernels")]
    keepalive: Box<dyn Any + Send + Sync>,
}

impl Gpu {
    /// Wraps a device a backend has already opened.
    ///
    /// `keepalive` is whatever concrete handle owns the compiled kernels -
    /// Metal's `MetalBackend`, CUDA's `CudaBackend`. It is never read. It is
    /// held because dropping it unloads the library or the module set the entry
    /// points were resolved out of, and a launch after that fails for a reason
    /// that has nothing to do with the kernel under test.
    #[must_use]
    pub fn new(device: Arc<dyn Backend>, keepalive: Box<dyn Any + Send + Sync>) -> Self {
        Self { device, keepalive }
    }

    /// The device this case set is running against, for a case that needs the
    /// trait object rather than the helpers below.
    #[must_use]
    pub fn device(&self) -> &Arc<dyn Backend> {
        &self.device
    }

    /// What the backend calls itself, for an assertion message that has to say
    /// which device produced the number.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        self.device.backend_name()
    }

    /// Allocates and fills a device buffer.
    ///
    /// # Panics
    ///
    /// When the allocation or the copy is refused. A case cannot continue past
    /// either, and the panic is the failure.
    #[must_use]
    pub fn upload(&self, bytes: &[u8]) -> DeviceBuffer {
        let buffer = alloc(&self.device, bytes.len() as u64).expect("device allocation");
        self.device
            .copy_h2d(buffer.view(), bytes)
            .expect("host-to-device copy");
        buffer
    }

    /// Reads a whole device buffer back.
    ///
    /// # Panics
    ///
    /// When the copy is refused.
    #[must_use]
    pub fn read(&self, buffer: &DeviceBuffer) -> Vec<u8> {
        let mut out = vec![0u8; usize::try_from(buffer.len()).expect("buffer length fits usize")];
        self.device
            .copy_d2h(buffer.view(), &mut out)
            .expect("device-to-host copy");
        out
    }

    /// Reads a whole device buffer as `f32` values.
    ///
    /// # Panics
    ///
    /// When the copy is refused.
    #[must_use]
    pub fn read_f32(&self, buffer: &DeviceBuffer) -> Vec<f32> {
        f32_values(&self.read(buffer))
    }
}

/// Launches one kernel, naming the geometry if the driver refuses it.
///
/// # Panics
///
/// When the backend refuses the launch. The message carries the entry name, the
/// whole geometry and the backend's own error chain, because a refusal here is
/// as often a shape this device will not run as it is a kernel that is absent.
pub fn run(gpu: &Gpu, kernel: KernelRef, launch: &LaunchArgs<'_>) {
    gpu.device.run(kernel, launch).unwrap_or_else(|err| {
        panic!(
            "{} grid {:?} block {:?} smem {}: {}",
            kernel.entry,
            launch.grid,
            launch.block,
            launch.smem_bytes,
            err.chain(),
        )
    });
}

/// Waits for the queue, which is what makes a read-back meaningful.
///
/// # Panics
///
/// When the queue does not drain.
pub fn drain(gpu: &Gpu) {
    gpu.device.sync().expect("the queue drains");
}

/// `f32` values as little-endian bytes.
#[must_use]
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `f32` values from little-endian bytes, without assuming any alignment.
#[must_use]
pub fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_bits(u32::from_le_bytes(*w)))
        .collect()
}

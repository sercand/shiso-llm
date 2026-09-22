//! Shared GPU harness for the q35 (qwen35) kernel tests.
//!
//! The q35 test targets are separate binaries, so the device wrapper, the launch
//! helper and the byte conversions live here instead of being copied into each
//! file. Same shape as `tests/metal_kernels.rs`: a case passes when the device
//! agrees with the naive CPU oracle over the same buffers, so a failure points at
//! the kernel or at its launch geometry rather than at a second implementation.
//!
//! `f16` fixtures are written as explicit bit patterns and read back through
//! `oracle::f16_to_f32`, so nothing in the fixture path re-implements a float
//! conversion.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer, KernelRef, LaunchArgs, alloc};
use shiso_metal::MetalBackend;

/// An open device plus the concrete handle, kept so the library stays alive.
pub(crate) struct Gpu {
    // Read by nothing, but dropping it would release the library the pipelines
    // were compiled into, so the field is load-bearing even though nothing
    // reads it.
    #[expect(
        dead_code,
        reason = "the library handle is only a keep-alive; dropping it unloads the library"
    )]
    pub metal: Arc<MetalBackend>,
    pub device: Arc<dyn Backend>,
}

impl Gpu {
    #[expect(
        clippy::print_stderr,
        reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
    )]
    pub(crate) fn open() -> Option<Self> {
        if !shiso_metal::metal_available() {
            eprintln!("note: no Metal device on this machine, skipping the GPU cases");
            return None;
        }
        let metal = MetalBackend::new().expect("the embedded kernel library compiles");
        let device: Arc<dyn Backend> = metal.clone();
        Some(Self { metal, device })
    }

    /// Allocates and fills a device buffer.
    pub(crate) fn upload(&self, bytes: &[u8]) -> DeviceBuffer {
        let buffer = alloc(&self.device, bytes.len() as u64).expect("device allocation");
        self.device
            .copy_h2d(buffer.view(), bytes)
            .expect("host-to-device copy");
        buffer
    }

    /// Reads a whole device buffer back.
    pub(crate) fn read(&self, buffer: &DeviceBuffer) -> Vec<u8> {
        let mut out = vec![0u8; usize::try_from(buffer.len()).expect("buffer length fits usize")];
        self.device
            .copy_d2h(buffer.view(), &mut out)
            .expect("device-to-host copy");
        out
    }

    /// Reads a whole device buffer as `f32` values.
    pub(crate) fn read_f32(&self, buffer: &DeviceBuffer) -> Vec<f32> {
        f32_values(&self.read(buffer))
    }
}

/// Launches one kernel, naming the geometry if the driver refuses it.
pub(crate) fn run(gpu: &Gpu, kernel: KernelRef, launch: &LaunchArgs<'_>) {
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
pub(crate) fn drain(gpu: &Gpu) {
    gpu.device.sync().expect("the queue drains");
}

/// `f32` values as little-endian bytes.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `f32` values from little-endian bytes, without assuming any alignment.
pub(crate) fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|w| f32::from_bits(u32::from_le_bytes([w[0], w[1], w[2], w[3]])))
        .collect()
}

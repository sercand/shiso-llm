//! The arm this crate presents where Metal does not exist.
//!
//! A `Device::Metal` is a runtime choice a caller can make on any host - a
//! parity fixture names its device in the file, and the file travels between
//! machines. So the choice has to be *representable* everywhere and refused
//! where it cannot be honoured, exactly as the sibling CUDA backend refuses on a
//! machine with no driver. Gating the caller instead would make a fixture that
//! names the wrong device a compile error on one host and a typed error on the
//! other.
//!
//! [`MetalBackend`] is uninhabited here: [`MetalBackend::new`] only ever
//! returns the refusal, and the [`Backend`] implementation exists so the type
//! still coerces to `Arc<dyn Backend>` at the call site. Every method is
//! `match *self {}` - unreachable because no value can be built, and there is no
//! second implementation to drift from the real one.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use shiso_core::{
    Backend, CoreError, CoreResult, DeviceView, HostBytes, KernelRef, LaunchArgs, RawBuffer,
};

/// Probes for a Metal device without building a backend.
///
/// Always `false` here: this target has no Metal runtime to ask.
#[must_use]
pub const fn metal_available() -> bool {
    false
}

/// The Metal backend, on a target that has none.
#[derive(Debug)]
pub enum MetalBackend {}

impl MetalBackend {
    /// Refuses, because this build has no Metal runtime.
    ///
    /// # Errors
    ///
    /// Always [`CoreError::Backend`]; the `metal` feature is off, or the target
    /// has no Objective-C runtime to reach a device through.
    pub fn new() -> CoreResult<Arc<Self>> {
        Err(CoreError::Backend {
            backend: "metal",
            op: "new",
            source: "this build has no Metal runtime".into(),
        })
    }
}

impl Backend for MetalBackend {
    fn backend_name(&self) -> &'static str {
        match *self {}
    }

    fn alloc_raw(&self, _bytes: u64) -> CoreResult<RawBuffer> {
        match *self {}
    }

    fn free_raw(&self, _raw: RawBuffer) -> CoreResult<()> {
        match *self {}
    }

    fn import_raw(&self, _host: &dyn HostBytes, _range: Range<usize>) -> CoreResult<RawBuffer> {
        match *self {}
    }

    fn copy_h2d(&self, _dst: DeviceView<'_>, _src: &[u8]) -> CoreResult<()> {
        match *self {}
    }

    fn copy_d2h(&self, _src: DeviceView<'_>, _dst: &mut [u8]) -> CoreResult<()> {
        match *self {}
    }

    fn copy_d2d(&self, _dst: DeviceView<'_>, _src: DeviceView<'_>) -> CoreResult<()> {
        match *self {}
    }

    fn run(&self, _kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
        match *self {}
    }

    fn sync(&self) -> CoreResult<()> {
        match *self {}
    }
}

impl fmt::Display for MetalBackend {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {}
    }
}

//! Kernel launch.
//!
//! Geometry is CUDA semantics and is passed straight through: `grid` is the
//! threadgroup count, `block` is the threads per threadgroup, and nothing here
//! divides, rounds up or otherwise reinterprets either. Specialization comes
//! from `KernelRef::consts`, so a launch never formats or parses a name.
//!
//! There is no `commit` in this module. Launches accumulate into the open
//! command buffer the backend keeps open; only `sync` and the host-copy paths
//! flush.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDataType, MTLFunctionConstantValues,
    MTLSize,
};
use shiso_core::{CoreResult, Extent3, FnConst, FnConstValue, KernelArg, KernelRef, LaunchArgs};

use crate::device::MetalBackend;
use crate::device::MetalError;

/// Appends one dispatch for `kernel` to the backend's open command buffer.
///
/// This is the body of [`shiso_core::Backend::run`]; it lives here so the
/// device module stays about devices and this file owns the encoder calls.
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] when the pipeline cannot be resolved, the
/// threadgroup is larger than the device allows, an argument names a buffer that
/// is no longer live, or the command buffer refuses an encoder.
pub(crate) fn dispatch(
    backend: &MetalBackend,
    kernel: KernelRef,
    launch: &LaunchArgs<'_>,
) -> CoreResult<()> {
    let mut state = backend.lock();
    let resolved = backend.resolve(&mut state, kernel)?;
    let pipeline = resolved.pipeline;
    // A kernel with no dynamic threadgroup binding sized its staging itself;
    // the CUDA-derived request is not its to honour. See `PipelineState`.
    let smem = if resolved.dynamic_threadgroup {
        launch.smem_bytes
    } else {
        0
    };
    let threads = u64::from(launch.block.x)
        .saturating_mul(u64::from(launch.block.y))
        .saturating_mul(u64::from(launch.block.z));
    let limit = u64::try_from(pipeline.maxTotalThreadsPerThreadgroup()).unwrap_or(u64::MAX);
    if threads == 0 || threads > limit {
        return Err(MetalError::backend(
            "run",
            format!(
                "{threads} threads per threadgroup is outside 1..={limit} for {:?}",
                kernel.entry
            ),
        ));
    }
    let smem_bytes = u64::from(smem);
    let smem_limit = backend.max_threadgroup_memory();
    if smem_bytes > smem_limit {
        return Err(MetalError::backend(
            "run",
            format!(
                "{smem_bytes} bytes of threadgroup memory is outside 0..={smem_limit} for {:?}",
                kernel.entry
            ),
        ));
    }

    let encoder = backend.compute_encoder(&mut state)?;
    encoder.setComputePipelineState(&pipeline);

    for (index, arg) in launch.args.iter().enumerate() {
        let at = NSUInteger::try_from(index).map_err(|_| {
            MetalError::backend("run", format!("argument index {index} is out of range"))
        })?;
        match arg {
            KernelArg::Buffer(view) => {
                let entry = state.view(*view)?;
                let offset = usize::try_from(entry.base + view.offset()).unwrap_or(usize::MAX);
                // SAFETY: `view` proved the range fits inside this entry, so
                // `base + offset` lands inside a buffer the table retains for at
                // least `view.len()` bytes. `at` is a parameter ordinal of the
                // kernel being launched, and Metal only records the binding.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&entry.buffer), offset, at);
                }
            }
            KernelArg::U32(value) => set_scalar(&encoder, at, *value),
            KernelArg::I32(value) => set_scalar(&encoder, at, *value),
            KernelArg::U64(value) => set_scalar(&encoder, at, *value),
            KernelArg::F32(value) => set_scalar(&encoder, at, *value),
            KernelArg::Bytes(bytes) => {
                if !bytes.is_empty() {
                    // SAFETY: the slice is a live by-value argument at the call
                    // site and Metal copies the bytes during the send.
                    unsafe {
                        encoder.setBytes_length_atIndex(
                            NonNull::from(*bytes).cast::<c_void>(),
                            bytes.len(),
                            at,
                        );
                    }
                }
            }
            // A newly added core argument must be bound here rather than
            // skipped, or the kernel would launch with a stale binding.
            _ => {
                return Err(MetalError::backend(
                    "run",
                    format!("kernel argument at index {index} is unsupported by Metal"),
                ));
            }
        }
    }

    if smem > 0 {
        // SAFETY: threadgroup memory lives in its own binding space, so index
        // zero cannot collide with the parameter bindings above. The length
        // was checked above against `max_threadgroup_memory`, which is the
        // one validation Metal itself performs neither here nor at pipeline
        // creation -- an over-limit length passed to this call is undefined
        // behaviour, not a reported error, so the bound above is load-bearing.
        unsafe {
            encoder
                .setThreadgroupMemoryLength_atIndex(usize::try_from(smem).unwrap_or(usize::MAX), 0);
        }
    }

    encoder.dispatchThreadgroups_threadsPerThreadgroup(size(launch.grid), size(launch.block));
    Ok(())
}

/// Binds one by-value scalar argument.
fn set_scalar<T: Copy>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    at: NSUInteger,
    value: T,
) {
    // SAFETY: `value` is a live local of exactly `size_of::<T>()` bytes and
    // Metal copies them during the message send rather than keeping the
    // pointer.
    unsafe {
        encoder.setBytes_length_atIndex(NonNull::from(&value).cast::<c_void>(), size_of::<T>(), at);
    }
}

/// `Extent3` in the shape Metal wants, with no arithmetic.
fn size(extent: Extent3) -> MTLSize {
    MTLSize {
        width: usize::try_from(u64::from(extent.x)).unwrap_or(usize::MAX),
        height: usize::try_from(extent.y).unwrap_or(usize::MAX),
        depth: usize::try_from(extent.z).unwrap_or(usize::MAX),
    }
}

/// Builds the specialization table for `consts`, or `None` when it is empty.
pub(crate) fn constant_values(
    consts: &[FnConst],
) -> CoreResult<Option<Retained<MTLFunctionConstantValues>>> {
    if consts.is_empty() {
        return Ok(None);
    }
    let values = MTLFunctionConstantValues::new();
    for c in consts {
        let at = NSUInteger::from(c.index);
        match c.value {
            FnConstValue::Bool(value) => set(&values, MTLDataType::Bool, &value, at),
            FnConstValue::I16(value) => set(&values, MTLDataType::Short, &value, at),
            FnConstValue::U16(value) => set(&values, MTLDataType::UShort, &value, at),
            FnConstValue::U32(value) => set(&values, MTLDataType::UInt, &value, at),
            // A new core variant must be mapped here rather than silently
            // dropped, or a specialized kernel would launch with defaults.
            _ => {
                return Err(MetalError::backend(
                    "run",
                    format!("function constant type {:?} is unmapped", c.value),
                ));
            }
        }
    }
    Ok(Some(values))
}

/// One `setConstantValue:type:atIndex:` send.
fn set<T: Copy>(
    values: &MTLFunctionConstantValues,
    data_type: MTLDataType,
    value: &T,
    at: NSUInteger,
) {
    // SAFETY: `value` points at a live local of exactly the width `data_type`
    // names, which the caller's match guarantees, and Metal copies it during
    // the send.
    unsafe {
        values.setConstantValue_type_atIndex(NonNull::from(value).cast::<c_void>(), data_type, at);
    }
}

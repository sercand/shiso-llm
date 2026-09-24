//! The driver seam: one function surface, two implementations, chosen by target.
//!
//! Everything above this module — the device, the module registry, the launch
//! path, the BLAS tier and the graph cache — is ordinary portable Rust that
//! calls `sys::nvidia::*`. Which body those calls reach is decided here, once,
//! by `cfg(target_os)`:
//!
//! * on Linux, [`nvidia`] is the real thing, built on generated driver bindings;
//! * everywhere else it is the stub arm, where every entry point answers
//!   [`SysError::Host`].
//!
//! # Why a module swap and not a trait
//!
//! A trait would put a vtable dispatch on the launch path, which is the one
//! path in this crate that runs hundreds of times per token, and it would force
//! every handle type to be an associated type just so the stub arm could name
//! something. Const generics cannot swap function bodies per target without
//! monomorphising the whole orchestration layer. A module swap costs nothing at
//! runtime, keeps both arms honest by making them share this file's type
//! definitions, and means the stub arm *is* the type-check harness: if a
//! signature drifts on one side, the other side stops compiling.
//!
//! # Handles
//!
//! Driver handles are opaque tokens. Rust never dereferences one, so they are
//! carried as integers rather than raw pointers: that keeps them `Send + Sync`
//! without an unsafe impl, keeps every field private so a handle cannot be
//! forged, and makes the only way to obtain one a call that actually asked the
//! driver for it. The owning wrappers in the modules above are what release
//! them.

use std::fmt;

use shiso_core::CoreError;
use thiserror::Error;

/// The name this backend reports in every error it raises.
pub(crate) const BACKEND: &str = "cuda";

/// Dynamic shared memory an entry point gets without asking, in bytes.
///
/// Every device this backend targets gives a kernel this much on request and
/// refuses more until the entry point has opted in, once, before its first
/// launch. The opt-in itself lives in the registry; this constant is the line
/// it checks against, and it is a property of the hardware contract rather
/// than of either target arm, so it is defined here where both can see it.
pub(crate) const DEFAULT_SMEM_LIMIT: u32 = 48 * 1024;

// One module name, two files, chosen by target. Both files declare the same
// function surface; neither declares a nested module, so `#[path]` here is the
// whole of the mechanism.
#[cfg(target_os = "linux")]
#[path = "nvidia.rs"]
mod arm;
#[cfg(not(target_os = "linux"))]
#[path = "stub.rs"]
mod arm;

/// The driver surface for this target.
///
/// Call sites name `sys::nvidia::…` on every target; only the body behind it
/// changes.
pub(crate) mod nvidia {
    pub(crate) use super::arm::*;
}

/// Result of one driver call.
pub(crate) type SysResult<T> = Result<T, SysError>;

/// A failure reported by the driver, or by the absence of one.
///
/// This is the crate's internal error currency. It never escapes: every public
/// entry point converts it into [`CoreError::Backend`], so a caller sees one
/// structured error type and reaches the driver's own message through
/// `source()`.
#[derive(Debug, Error)]
pub(crate) enum SysError {
    /// A driver entry point returned a non-success status.
    #[cfg_attr(
        all(not(target_os = "linux"), not(test)),
        expect(
            dead_code,
            reason = "only the driver arm reports a driver status; the variant exists so both arms share one error type"
        )
    )]
    #[error("{op} failed: {name} ({code}): {text}")]
    Driver {
        /// The driver entry point that failed, used as `CoreError::Backend::op`.
        op: &'static str,
        /// The numeric status the driver returned.
        code: i32,
        /// The driver's symbolic name for that status.
        name: String,
        /// The driver's human-readable text for that status.
        text: String,
    },
    /// A math-library entry point returned a non-success status.
    ///
    /// Kept apart from [`SysError::Driver`] because the two use different
    /// status vocabularies and only the driver can turn a code into a name.
    #[cfg_attr(
        all(not(target_os = "linux"), not(test)),
        expect(
            dead_code,
            reason = "only the driver arm reports a math-library status; the variant exists so both arms share one error type"
        )
    )]
    #[error("{op} failed with status {code}")]
    Math {
        /// The entry point that failed.
        op: &'static str,
        /// The numeric status the library returned.
        code: i32,
    },
    /// The call never reached a driver.
    ///
    /// Either this target has no driver at all, or the arguments could not be
    /// expressed in the shapes the driver takes.
    #[error("{op}: {what}")]
    Host {
        /// The entry point that refused.
        op: &'static str,
        /// Why it refused.
        what: &'static str,
    },
}

impl SysError {
    /// The operation name, for [`CoreError::Backend::op`].
    pub(crate) const fn op(&self) -> &'static str {
        match self {
            Self::Driver { op, .. } | Self::Math { op, .. } | Self::Host { op, .. } => op,
        }
    }

    /// The message every entry point on a target without a driver answers with.
    #[cfg_attr(
        all(target_os = "linux", not(test)),
        expect(
            dead_code,
            reason = "only the stub arm has an absent driver to report; the constructor lives here so both arms share one error type"
        )
    )]
    pub(crate) const fn no_driver(op: &'static str) -> Self {
        Self::Host {
            op,
            what: "this build has no CUDA driver: the backend was compiled against its stub arm",
        }
    }

    /// Whether this failure means "the thing you are releasing is already gone".
    ///
    /// Teardown races are normal: the process may be exiting, the context may
    /// already have been torn down by a previous release, or the driver may
    /// have deinitialised itself first. Releasing a resource in that state is
    /// not a failure — the resource is gone, which is what the caller wanted —
    /// so a `Drop` that hits one of these logs nothing and moves on, while any
    /// other status is still reported.
    pub(crate) const fn is_teardown_noop(&self) -> bool {
        match self {
            // deinitialised, invalid context, context already destroyed
            Self::Driver { code, .. } => matches!(*code, 4 | 201 | 709),
            Self::Math { .. } | Self::Host { .. } => false,
        }
    }
}

impl From<SysError> for CoreError {
    /// Wraps a driver failure as the one structured error this crate exports.
    ///
    /// The operation name travels as a field rather than being formatted into
    /// a message, and the driver error stays reachable as `source()`, so
    /// `{:#}` chains it and a caller can still branch on the cause.
    fn from(err: SysError) -> Self {
        Self::Backend {
            backend: BACKEND,
            op: err.op(),
            source: Box::new(err),
        }
    }
}

/// Builds a [`CoreError::Backend`] for a failure this crate detected itself.
///
/// Used where the driver was never asked — a bad argument, an empty kernel
/// image, a capture that is already running — so that those failures arrive in
/// exactly the same shape as the driver's own.
pub(crate) fn host_error(op: &'static str, message: impl Into<String>) -> CoreError {
    CoreError::Backend {
        backend: BACKEND,
        op,
        source: Box::new(HostFailure {
            message: message.into(),
        }),
    }
}

/// A failure raised by this crate rather than by the driver.
#[derive(Debug)]
pub(crate) struct HostFailure {
    message: String,
}

impl fmt::Display for HostFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HostFailure {}

macro_rules! opaque_handle {
    ($(#[$meta:meta])* $name:ident, $repr:ty) => {
        $(#[$meta])*
        ///
        /// An opaque driver token. The field is private and the only way to
        /// obtain one is a driver call that produced it, so it cannot be forged
        /// and it carries no meaning this crate is allowed to interpret.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub(crate) struct $name($repr);

        impl $name {
            /// Wraps a token the driver just handed back.
            ///
            /// Only the target arm of this module calls this, immediately after
            /// the call that produced the value.
            #[cfg_attr(
                all(not(target_os = "linux"), not(test)),
                expect(
                    dead_code,
                    reason = "the stub arm never receives a handle from a driver, but the constructor has to exist for both arms to share one type"
                )
            )]
            pub(crate) const fn from_raw(raw: $repr) -> Self {
                Self(raw)
            }

            /// The token, on its way back into a driver call.
            #[cfg_attr(
                all(not(target_os = "linux"), not(test)),
                expect(
                    dead_code,
                    reason = "the stub arm never passes a handle to a driver, but the accessor has to exist for both arms to share one type"
                )
            )]
            pub(crate) const fn raw(self) -> $repr {
                self.0
            }
        }
    };
}

opaque_handle!(
    /// A physical device, as the driver numbers them.
    CuDevice,
    i32
);
opaque_handle!(
    /// The device's primary context.
    CuContext,
    usize
);
opaque_handle!(
    /// A stream: an ordered queue of device work.
    CuStream,
    usize
);
opaque_handle!(
    /// An event: a marker in one stream that another can wait on.
    CuEvent,
    usize
);
opaque_handle!(
    /// A loaded kernel module.
    CuModule,
    usize
);
opaque_handle!(
    /// One entry point inside a loaded module.
    CuFunction,
    usize
);
opaque_handle!(
    /// A captured, not-yet-instantiated graph.
    CuGraph,
    usize
);
opaque_handle!(
    /// An instantiated graph, ready to replay.
    CuGraphExec,
    usize
);
opaque_handle!(
    /// A classic math-library handle.
    CuBlas,
    usize
);
opaque_handle!(
    /// A heuristic-driven math-library handle.
    CuBlasLt,
    usize
);

opaque_handle!(
    /// Page-locked host memory the driver allocated.
    ///
    /// The device reads it by DMA without the driver first copying it into a
    /// bounce buffer of its own, which is what makes an asynchronous copy out
    /// of it both possible and fast.
    PinnedPtr,
    usize
);

/// An address in device memory.
///
/// Distinct from a host pointer on purpose: the two are the same width and are
/// never interchangeable, and the type is what stops one being passed where the
/// other belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct DevicePtr(u64);

impl DevicePtr {
    /// Wraps an address the allocator just returned.
    #[cfg_attr(
        all(not(target_os = "linux"), not(test)),
        expect(
            dead_code,
            reason = "only the driver arm has an allocator to take an address from"
        )
    )]
    pub(crate) const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The address, on its way back into a driver call.
    pub(crate) const fn raw(self) -> u64 {
        self.0
    }

    /// The address `bytes` further in.
    ///
    /// Returns `None` on overflow rather than wrapping: an address that wrapped
    /// past the end of the address space is not an address, and a kernel handed
    /// one would corrupt memory it was never given.
    pub(crate) const fn offset(self, bytes: u64) -> Option<Self> {
        match self.0.checked_add(bytes) {
            Some(sum) => Some(Self(sum)),
            None => None,
        }
    }
}

/// Which operand a matrix multiply reads transposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Op {
    /// Read as stored.
    N,
    /// Read transposed.
    T,
}

/// Element type of a matrix multiply's operands.
///
/// The accumulator is single precision in every case, which is the arithmetic
/// the reference numbers for this port were captured with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MathType {
    /// Single-precision operands.
    F32,
    /// Half-precision operands.
    F16,
    /// Brain-float operands.
    Bf16,
}

/// One resolved matrix multiply, with every view already turned into an
/// address and every dimension already proven to fit the driver's `int`.
///
/// This is the form the driver takes. The borrowed, bounds-checkable form a
/// caller writes is `CudaGemm`, and turning one into the other is the whole of
/// the validation step.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(
        dead_code,
        reason = "the stub arm neither produces nor consumes driver values; the definitions exist so both arms share one type surface"
    )
)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct GemmCall {
    /// Transposition of the left operand.
    pub(crate) op_a: Op,
    /// Transposition of the right operand.
    pub(crate) op_b: Op,
    /// Rows of the product.
    pub(crate) m: i32,
    /// Columns of the product.
    pub(crate) n: i32,
    /// Shared inner dimension.
    pub(crate) k: i32,
    /// Left operand.
    pub(crate) a: DevicePtr,
    /// Leading dimension of the left operand.
    pub(crate) lda: i32,
    /// Right operand.
    pub(crate) b: DevicePtr,
    /// Leading dimension of the right operand.
    pub(crate) ldb: i32,
    /// Accumulator.
    pub(crate) c: DevicePtr,
    /// Leading dimension of the accumulator.
    pub(crate) ldc: i32,
    /// Scale applied to the product.
    pub(crate) alpha: f32,
    /// Scale applied to the existing accumulator.
    pub(crate) beta: f32,
    /// Operand element type.
    pub(crate) dtype: MathType,
    /// Whether reduced-precision accumulation of single-precision operands is
    /// allowed for this call.
    pub(crate) tf32: bool,
    /// Whether the product is written back in half precision rather than
    /// single. Accumulation is single precision either way.
    pub(crate) out_half: bool,
}

/// The extra arguments a strided-batched multiply needs.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(
        dead_code,
        reason = "the stub arm neither produces nor consumes driver values; the definitions exist so both arms share one type surface"
    )
)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct GemmStride {
    /// Number of multiplies in the batch.
    pub(crate) count: i32,
    /// Element stride between consecutive left operands.
    pub(crate) a: i64,
    /// Element stride between consecutive right operands.
    pub(crate) b: i64,
    /// Element stride between consecutive accumulators.
    pub(crate) c: i64,
}

#[cfg(test)]
mod tests {
    use super::{DevicePtr, SysError};
    use shiso_core::CoreError;

    #[test]
    fn a_driver_failure_becomes_a_backend_error_naming_the_operation() {
        let err = SysError::Driver {
            op: "cuModuleLoadData",
            code: 209,
            name: "CUDA_ERROR_NO_BINARY_FOR_GPU".to_owned(),
            text: "no kernel image is available".to_owned(),
        };
        let core: CoreError = err.into();
        match core {
            CoreError::Backend {
                backend,
                op,
                source,
            } => {
                assert_eq!(backend, "cuda");
                assert_eq!(op, "cuModuleLoadData");
                let text = source.to_string();
                assert!(text.contains("CUDA_ERROR_NO_BINARY_FOR_GPU"), "{text}");
                assert!(text.contains("209"), "{text}");
            }
            other => panic!("expected a backend error, got {other:?}"),
        }
    }

    #[test]
    fn the_absence_of_a_driver_is_a_backend_error_not_a_panic() {
        let core: CoreError = SysError::no_driver("cuInit").into();
        match core {
            CoreError::Backend { backend, op, .. } => {
                assert_eq!(backend, "cuda");
                assert_eq!(op, "cuInit");
            }
            other => panic!("expected a backend error, got {other:?}"),
        }
    }

    /// Both arms share these definitions, and only one arm ever builds one. The
    /// round trip here is what keeps the shared surface honest on a target that
    /// has no driver to build one from: a field that stopped being carried, or
    /// a handle that stopped round-tripping, fails here rather than on the
    /// Linux box.
    #[test]
    fn every_shared_driver_value_round_trips_through_its_own_representation() {
        use super::{
            CuBlas, CuBlasLt, CuContext, CuDevice, CuFunction, CuGraph, CuGraphExec, CuModule,
            CuStream, GemmCall, GemmStride, MathType, Op,
        };

        assert_eq!(CuDevice::from_raw(3).raw(), 3);
        assert_eq!(CuContext::from_raw(4).raw(), 4);
        assert_eq!(CuStream::from_raw(5).raw(), 5);
        assert_eq!(CuModule::from_raw(6).raw(), 6);
        assert_eq!(CuFunction::from_raw(7).raw(), 7);
        assert_eq!(CuGraph::from_raw(8).raw(), 8);
        assert_eq!(CuGraphExec::from_raw(9).raw(), 9);
        assert_eq!(CuBlas::from_raw(10).raw(), 10);
        assert_eq!(CuBlasLt::from_raw(11).raw(), 11);
        assert_eq!(DevicePtr::from_raw(12).raw(), 12);

        let call = GemmCall {
            op_a: Op::N,
            op_b: Op::T,
            m: 1,
            n: 2,
            k: 3,
            a: DevicePtr::from_raw(0x100),
            lda: 4,
            b: DevicePtr::from_raw(0x200),
            ldb: 5,
            c: DevicePtr::from_raw(0x300),
            ldc: 6,
            alpha: 1.0,
            beta: 0.0,
            dtype: MathType::Bf16,
            tf32: true,
            out_half: false,
        };
        assert_eq!((call.op_a, call.op_b), (Op::N, Op::T));
        assert_eq!((call.m, call.n, call.k), (1, 2, 3));
        assert_eq!((call.lda, call.ldb, call.ldc), (4, 5, 6));
        assert_eq!(call.a.raw(), 0x100);
        assert_eq!(call.b.raw(), 0x200);
        assert_eq!(call.c.raw(), 0x300);
        assert!(call.alpha > call.beta);
        assert_eq!(call.dtype, MathType::Bf16);
        assert!(call.tf32);

        let stride = GemmStride {
            count: 4,
            a: 16,
            b: 32,
            c: 64,
        };
        assert_eq!(
            (stride.count, stride.a, stride.b, stride.c),
            (4, 16, 32, 64)
        );
    }

    #[test]
    fn a_math_library_failure_becomes_a_backend_error_naming_the_operation() {
        let core: CoreError = SysError::Math {
            op: "cublasGemmEx",
            code: 7,
        }
        .into();
        match core {
            CoreError::Backend { op, source, .. } => {
                assert_eq!(op, "cublasGemmEx");
                assert!(source.to_string().contains('7'));
            }
            other => panic!("expected a backend error, got {other:?}"),
        }
    }

    /// The stub arm has to be *total*: a single entry point that panicked, or
    /// that was accidentally left out, would turn "this machine has no GPU"
    /// into a crash somewhere in the middle of a request. Calling every one of
    /// them and insisting on a typed error is the only way to know the whole
    /// surface is there.
    ///
    /// Gated to targets without a driver, because on a target that has one
    /// these calls would hand the driver handles that name nothing. Split in
    /// three only because one function covering thirty entry points is a
    /// function nobody reads.
    #[cfg(not(target_os = "linux"))]
    mod stub_surface {
        use super::super::{
            CuBlas, CuBlasLt, CuContext, CuDevice, CuFunction, CuGraph, CuGraphExec, CuModule,
            CuStream, DevicePtr, GemmCall, GemmStride, MathType, Op, SysResult, nvidia,
        };
        use shiso_core::{CoreError, Extent3};

        /// Asserts one entry point answered a backend error naming `op`.
        fn refused<T: std::fmt::Debug>(op: &str, result: SysResult<T>) {
            let err = result.expect_err("a target without a driver cannot succeed");
            let core: CoreError = err.into();
            match core {
                CoreError::Backend {
                    backend, op: got, ..
                } => {
                    assert_eq!(backend, "cuda");
                    assert_eq!(
                        got, op,
                        "the error has to name the entry point it came from"
                    );
                }
                other => panic!("expected a backend error from {op}, got {other:?}"),
            }
        }

        /// A multiply description whose every field is a value, not a handle.
        fn call() -> GemmCall {
            let ptr = DevicePtr::from_raw(0x1000);
            GemmCall {
                op_a: Op::N,
                op_b: Op::N,
                m: 1,
                n: 1,
                k: 1,
                a: ptr,
                lda: 1,
                b: ptr,
                ldb: 1,
                c: ptr,
                ldc: 1,
                alpha: 1.0,
                beta: 0.0,
                dtype: MathType::F32,
                tf32: false,
                out_half: false,
            }
        }

        #[test]
        fn nothing_that_opens_a_resource_succeeds_without_a_driver() {
            refused("cuInit", nvidia::init());
            refused("cuDeviceGet", nvidia::device_get(0));
            refused(
                "cuDevicePrimaryCtxRetain",
                nvidia::primary_ctx_retain(CuDevice::from_raw(0)),
            );
            refused(
                "cuDeviceGetAttribute",
                nvidia::multiprocessor_count(CuDevice::from_raw(0)),
            );
            refused("cuStreamCreate", nvidia::stream_create());
            refused("cuMemAlloc", nvidia::mem_alloc(16));
            refused("cuModuleLoadData", nvidia::module_load(b"//ptx"));
            refused("cublasCreate", nvidia::blas_create());
            refused("cublasLtCreate", nvidia::blas_lt_create());
        }

        #[test]
        fn nothing_that_moves_memory_or_launches_work_succeeds_without_a_driver() {
            let stream = CuStream::from_raw(1);
            let module = CuModule::from_raw(1);
            let func = CuFunction::from_raw(1);
            let ptr = DevicePtr::from_raw(0x1000);
            let extent = Extent3::linear(1);
            let mut params: [*mut std::ffi::c_void; 0] = [];
            let mut host = [0_u8; 4];
            let source = [0_u8; 4];

            // SAFETY: on this target every one of these is the stub body, which
            // returns before it looks at an argument. The handles name nothing
            // and are never dereferenced, here or there.
            unsafe {
                refused(
                    "cuCtxSetCurrent",
                    nvidia::ctx_set_current(CuContext::from_raw(1)),
                );
                refused(
                    "cuDevicePrimaryCtxRelease",
                    nvidia::primary_ctx_release(CuDevice::from_raw(0)),
                );
                refused("cuStreamDestroy", nvidia::stream_destroy(stream));
                refused("cuStreamSynchronize", nvidia::stream_synchronize(stream));
                refused("cuMemFree", nvidia::mem_free(ptr));
                refused(
                    "cuMemcpyHtoDAsync",
                    nvidia::memcpy_htod(ptr, &source, stream),
                );
                refused(
                    "cuMemcpyDtoHAsync",
                    nvidia::memcpy_dtoh(&mut host, ptr, stream),
                );
                refused(
                    "cuMemcpyDtoDAsync",
                    nvidia::memcpy_dtod(ptr, ptr, 4, stream),
                );
                refused("cuModuleUnload", nvidia::module_unload(module));
                refused(
                    "cuModuleGetFunction",
                    nvidia::module_get_function(module, c"entry"),
                );
                refused(
                    "cuFuncSetAttribute",
                    nvidia::func_set_max_dynamic_smem(func, 65_536),
                );
                refused(
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                    nvidia::occupancy_max_active_blocks(func, 128, 0),
                );
                refused(
                    "cuLaunchKernel",
                    nvidia::launch_kernel(func, extent, extent, 0, stream, &mut params),
                );
                refused(
                    "cuLaunchCooperativeKernel",
                    nvidia::launch_cooperative(func, extent, extent, 0, stream, &mut params),
                );
            }
        }

        #[test]
        fn no_graph_or_multiply_entry_point_succeeds_without_a_driver() {
            let stream = CuStream::from_raw(1);
            let graph = CuGraph::from_raw(1);
            let exec = CuGraphExec::from_raw(1);
            let handle = CuBlas::from_raw(1);
            let lt = CuBlasLt::from_raw(1);
            let ptr = DevicePtr::from_raw(0x1000);
            let call = call();
            let stride = GemmStride {
                count: 1,
                a: 0,
                b: 0,
                c: 0,
            };

            // SAFETY: as above — these are the stub bodies on this target and
            // they return before looking at a handle.
            unsafe {
                refused("cuStreamBeginCapture", nvidia::stream_begin_capture(stream));
                refused("cuStreamEndCapture", nvidia::stream_end_capture(stream));
                refused(
                    "cuGraphInstantiateWithFlags",
                    nvidia::graph_instantiate(graph),
                );
                refused("cuGraphDestroy", nvidia::graph_destroy(graph));
                refused("cuGraphExecDestroy", nvidia::graph_exec_destroy(exec));
                refused("cuGraphLaunch", nvidia::graph_launch(exec, stream));
                refused("cublasDestroy", nvidia::blas_destroy(handle));
                refused("cublasSetStream", nvidia::blas_set_stream(handle, stream));
                refused(
                    "cublasSetMathMode",
                    nvidia::blas_set_math_mode(handle, true),
                );
                refused("cublasGemmEx", nvidia::blas_gemm(handle, &call));
                refused(
                    "cublasGemmStridedBatchedEx",
                    nvidia::blas_gemm_strided(handle, &call, &stride),
                );
                refused("cublasLtDestroy", nvidia::blas_lt_destroy(lt));
                refused(
                    "cublasLtMatmul",
                    nvidia::blas_lt_matmul(lt, &call, ptr, 1024, stream),
                );
            }
        }
    }

    #[test]
    fn a_device_address_that_would_wrap_is_refused_rather_than_wrapped() {
        let top = DevicePtr::from_raw(u64::MAX - 7);
        assert_eq!(top.offset(7).map(DevicePtr::raw), Some(u64::MAX));
        assert_eq!(top.offset(8), None);
    }
}

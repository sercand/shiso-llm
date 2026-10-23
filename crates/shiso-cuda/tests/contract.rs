//! What this crate promises to the code above it, checked from outside it.
//!
//! These run on every target. On a host with no driver — which is every
//! developer machine in this project — the promise being checked is that the
//! absence shows up as a typed error naming the driver entry point that could
//! not be reached, at the moment a backend is opened, rather than as a panic, a
//! link failure, or a backend that half-works.
//!
//! The rest is shape: that the public descriptor types can actually be built by
//! a caller in another crate. That is not a tautology. A descriptor marked
//! `#[non_exhaustive]` with no constructor cannot be built outside the crate
//! that defines it at all, and the failure mode is that the whole matrix-
//! multiply surface is unreachable from the one crate that needs it. Building
//! one here is what keeps that from being discovered later.

use std::sync::Arc;

use shiso_core::{
    Backend, CoreError, CoreResult, DeviceView, GemmDType, KernelRef, LaunchArgs, RawBuffer,
};
use shiso_cuda::{BlasTier, CudaBackend, CudaGemm, CudaOptions, GemmBatch, Op, Operand};

/// A backend that only hands out buffer handles, so a test in this crate can
/// hold a real device view without a device.
#[derive(Debug, Default)]
struct HandleBackend {
    next: std::sync::atomic::AtomicU64,
}

impl Backend for HandleBackend {
    fn backend_name(&self) -> &'static str {
        "test"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let id = self
            .next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .saturating_add(1);
        let id = std::num::NonZeroU64::new(id).ok_or(CoreError::Unsupported {
            what: "more handles",
            backend: "test",
        })?;
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, _raw: RawBuffer) -> CoreResult<()> {
        Ok(())
    }

    fn copy_h2d(&self, _dst: DeviceView<'_>, _src: &[u8]) -> CoreResult<()> {
        Ok(())
    }

    fn copy_d2h(&self, _src: DeviceView<'_>, _dst: &mut [u8]) -> CoreResult<()> {
        Ok(())
    }

    fn copy_d2d(&self, _dst: DeviceView<'_>, _src: DeviceView<'_>) -> CoreResult<()> {
        Ok(())
    }

    fn run(&self, _kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
        Ok(())
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}

#[test]
fn opening_a_backend_without_a_driver_names_the_entry_point_it_could_not_reach() {
    if shiso_cuda::cuda_available() {
        // This host has a driver, so the failure this test describes cannot
        // happen; the Linux suite is what covers the success path.
        return;
    }
    match CudaBackend::new(0, CudaOptions::default()) {
        Err(CoreError::Backend { backend, op, .. }) => {
            assert_eq!(backend, "cuda");
            assert_eq!(
                op, "libcuda",
                "the first thing a backend does is probe the driver library, and that is what has \
                 to be named when there is none",
            );
        }
        Ok(_) => panic!("a host with no driver must not produce a working backend"),
        Err(other) => panic!("expected a backend error, got {other:?}"),
    }
}

#[test]
fn the_availability_probe_agrees_with_what_opening_a_backend_does() {
    let available = shiso_cuda::cuda_available();
    let opened = CudaBackend::new(0, CudaOptions::default()).is_ok();
    assert!(
        available || !opened,
        "a backend must not open on a host the probe says has no driver",
    );
}

#[test]
fn the_default_options_are_the_ones_the_reference_numbers_were_captured_with() {
    let options = CudaOptions::default();
    assert!(options.decode_graphs);
    assert!(options.tf32);
}

#[test]
fn each_option_can_be_turned_off_without_naming_the_others() {
    let options = CudaOptions::default()
        .with_decode_graphs(false)
        .with_tf32(false);
    assert!(!options.decode_graphs);
    assert!(!options.tf32);
    assert!(
        CudaOptions::default().with_tf32(false).decode_graphs,
        "setting one knob must not disturb the other",
    );
}

#[test]
fn the_ordinary_library_path_is_what_a_caller_gets_without_asking() {
    assert_eq!(BlasTier::default(), BlasTier::Classic);
}

#[test]
fn a_caller_in_another_crate_can_build_every_multiply_this_backend_accepts() {
    let backend: Arc<dyn Backend> = Arc::new(HandleBackend::default());
    let a = shiso_core::alloc(&backend, 1024).expect("allocation");
    let b = shiso_core::alloc(&backend, 1024).expect("allocation");
    let c = shiso_core::alloc(&backend, 1024).expect("allocation");

    let plain = CudaGemm {
        op_a: Op::N,
        op_b: Op::T,
        m: 8,
        n: 8,
        k: 8,
        a: Operand {
            view: a.view(),
            ld: 8,
        },
        b: Operand {
            view: b.view(),
            ld: 8,
        },
        c: Operand {
            view: c.view(),
            ld: 8,
        },
        alpha: 1.0,
        beta: 0.0,
        dtype: GemmDType::F16,
        batch: None,
        tier: BlasTier::Classic,
    };
    assert_eq!(plain.op_b, Op::T);

    let batched = CudaGemm {
        batch: Some(GemmBatch {
            count: 4,
            stride_a: 64,
            stride_b: 64,
            stride_c: 64,
        }),
        tier: BlasTier::Lt,
        ..plain
    };
    assert_eq!(batched.batch.map(|b| b.count), Some(4));
    assert_eq!(batched.tier, BlasTier::Lt);
    // The descriptor prints, which is what a failing launch's trace needs.
    assert!(format!("{batched:?}").contains("CudaGemm"));
}

/// An entry name that only exists inside a built module resolves, and an
/// ambiguous one does not resolve to a guess.
///
/// On a machine with no toolchain every module is empty, so what this pins here
/// is the half that matters most: an unresolvable name comes back as `None`
/// rather than as something a caller can launch. The positive half - an exact
/// name and a uniquely-containing one both resolving - needs a built module and
/// is in the Linux runbook beside it.
#[test]
fn an_entry_that_cannot_be_resolved_is_refused_rather_than_guessed() {
    // `resolve_entry` is on the trait, so this is the call a caller makes.
    let refused = shiso_cuda::ModuleEntries::resolve("q4e_matvec", "q4e_matmul_f16_rows_kernel");
    assert!(
        refused.is_none(),
        "with no built module there is no entry to resolve, and a guess would be worse than nothing"
    );

    // And the rule itself, over a name list, which is testable without a device.
    let names = [
        "q4e_gdn_conv_kernel".to_owned(),
        "_Z26q4e_matmul_f16_rows_kernelILi1EffE".to_owned(),
        "_Z26q4e_matmul_f16_rows_kernelILi2EffE".to_owned(),
        "_Z20q4e_moe_down_kernelIfE".to_owned(),
    ];
    assert_eq!(
        shiso_cuda::ModuleEntries::pick(&names, "q4e_gdn_conv_kernel"),
        Some("q4e_gdn_conv_kernel"),
        "a kernel with C linkage is named by itself"
    );
    assert_eq!(
        shiso_cuda::ModuleEntries::pick(&names, "q4e_moe_down_kernel"),
        Some("_Z20q4e_moe_down_kernelIfE"),
        "one instantiation is the only thing it could be"
    );
    assert_eq!(
        shiso_cuda::ModuleEntries::pick(&names, "q4e_matmul_f16_rows_kernel"),
        None,
        "two instantiations need the argument list to choose between them, so neither is picked"
    );
}

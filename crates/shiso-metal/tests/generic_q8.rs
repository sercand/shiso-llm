//! Metal's run of the shared `q8_0` gather and extended matvec cases.
//!
//! The cases live in `shiso-kerneltest`, so this file is the backend's half
//! only: open a device, or say why the cases did not run.
//!
//! The resolve test is not ignore-gated, so a missing or misnamed entry point
//! fails the default gate. The cases are `#[ignore]`d: `cargo test -p shiso-metal
//! --features metal --test kernels generic_q8:: -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::generic::rows_q8::{self, EXT_SHAPES};
use shiso_metal::MetalBackend;

/// Opens the device, or says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Gpu> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the GPU cases");
        return None;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    let device: Arc<dyn Backend> = metal.clone();
    Some(Gpu::new(device, Box::new(metal)))
}

/// Both entries resolve to pipelines. The matvec is specialized, as the case
/// launches it: an unspecialized one has no simdgroup count to compile with.
#[test]
fn every_q8_entry_resolves() {
    if !shiso_metal::metal_available() {
        return;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    metal
        .prepare(KernelRef::new("generic", rows_q8::ENTRIES[0]))
        .expect("the q8_0 gather resolves");
    for entry in &rows_q8::ENTRIES[1..] {
        for (in_dim, nxpsg, consts) in EXT_SHAPES {
            metal
                .prepare(KernelRef::specialized("generic", entry, consts))
                .unwrap_or_else(|err| panic!("{entry} resolves at {in_dim}/nx{nxpsg}: {err}"));
        }
    }
}

/// One ignored wrapper per shared case.
macro_rules! device_cases {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            #[ignore = "requires a Metal device"]
            fn $name() {
                let Some(gpu) = open() else { return };
                rows_q8::$name(&gpu);
            }
        )*
    };
}

device_cases!(
    the_q8_0_gather_dequantizes_the_rows_the_ids_name,
    the_q8_0_ext_matvec_matches_the_oracle_dot_at_every_row_count,
);

//! Every q35 entry point in the Task 5 Produces list resolves as a pipeline.
//!
//! Task 5 (docs/superpowers/plans/tasks/task-05-qwen35-msl-kernels.md), Step 12.
//! This is the only q35 target that is not ignore-gated: it dispatches nothing,
//! so a missing or misnamed entry point fails the default cargo test gate for
//! the crate instead of hiding behind the GPU gate.
//!
//! The names are the Produces list verbatim.  The ten `kernel_mul_mv_ext_*` and
//! three `kernel_mul_mm_*` entries are listed with the specialization table they
//! are launched with, because Metal refuses an unspecialized pipeline for a
//! kernel that reads a function constant, and reports that as a missing entry.
#![cfg(all(feature = "metal", target_os = "macos"))]

use shiso_core::{FnConst, FnConstValue, KernelRef};
use shiso_metal::MetalBackend;

/// The matvec simdgroup count and threads per row (task-04 Step 3).  `q5_K` and
/// `q6_K` instantiate the same template as `q4_K`, so they take the same pair.
const FC_MUL_MV: u16 = 600;
const FC_MUL_MM: u16 = 700;
const EXT_SHAPES: [FnConst; 2] = [
    FnConst {
        index: FC_MUL_MV,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: FC_MUL_MV + 1,
        value: FnConstValue::I16(16),
    },
];

/// The prefill GEMM's two broadcast toggles (`gemm_q4k.metal`: `FC_MUL_MM` 700).
/// Both off is the plain C = A @ B pipeline, which is what T6 dispatches; the
/// host sets both as `MTLDataTypeBool`, so they take Bool rather than a short.
const GEMM_BROADCAST: [FnConst; 2] = [
    FnConst {
        index: FC_MUL_MM,
        value: FnConstValue::Bool(false),
    },
    FnConst {
        index: FC_MUL_MM + 1,
        value: FnConstValue::Bool(false),
    },
];

/// Entry points that resolve exactly as declared.
const ENTRIES: [&str; 21] = [
    "kernel_q35_gdn_conv",
    "kernel_q35_gdn_l2norm",
    "kernel_q35_gdn_gates",
    "kernel_q35_gdn_recurrent_s1",
    "kernel_q35_gdn_recurrent_s2",
    "kernel_q35_gdn_recurrent_s4",
    "kernel_q35_gdn_recurrent_hybrid",
    "kernel_q35_gdn_out_gate",
    "kernel_q35_gdn_chunk_c1",
    "kernel_q35_gdn_chunk_c2",
    "kernel_q35_gdn_chunk_c4",
    "kernel_q35_gdn_chunk_c8",
    "kernel_q35_gdn_chunk_hybrid",
    "kernel_q35_q_norm_rope",
    "kernel_q35_k_norm_rope_kv",
    "kernel_q35_q_norm_rope_mrope",
    "kernel_q35_k_norm_rope_kv_mrope",
    "kernel_q35_attn_decode",
    "kernel_q35_attn_prefill",
    "kernel_q35_attn_gate_mul",
    "kernel_get_rows_q6_K_f32",
];

/// The new matvec `r1_N` for `N` in 1..=5, both quant types.
const EXT_ENTRIES: [&str; 10] = [
    "kernel_mul_mv_ext_q5_K_f32_r1_1",
    "kernel_mul_mv_ext_q5_K_f32_r1_2",
    "kernel_mul_mv_ext_q5_K_f32_r1_3",
    "kernel_mul_mv_ext_q5_K_f32_r1_4",
    "kernel_mul_mv_ext_q5_K_f32_r1_5",
    "kernel_mul_mv_ext_q6_K_f32_r1_1",
    "kernel_mul_mv_ext_q6_K_f32_r1_2",
    "kernel_mul_mv_ext_q6_K_f32_r1_3",
    "kernel_mul_mv_ext_q6_K_f32_r1_4",
    "kernel_mul_mv_ext_q6_K_f32_r1_5",
];

/// The prefill matmul prefill matmul instantiations kept by `gemm_q4k.metal`.
const GEMM_ENTRIES: [&str; 4] = [
    "kernel_mul_mm_f16_f32",
    "kernel_mul_mm_q8_0_f32",
    "kernel_mul_mm_q4_K_f32",
    "kernel_mul_mm_q4_0_f32",
];

/// Every name in the Produces list resolves to a pipeline.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "an opt-in case that did not run belongs on stderr"
)]
fn produces_list_resolves() {
    if !shiso_metal::metal_available() {
        eprintln!("SKIP: no Metal device, so the Produces list was not checked");
        return;
    }
    let backend = MetalBackend::new().expect("the q35 library compiles");

    for entry in ENTRIES {
        let kernel = KernelRef::new("qwen35", entry);
        backend
            .prepare(kernel)
            .unwrap_or_else(|err| panic!("{entry} did not resolve: {err}"));
    }
    for entry in EXT_ENTRIES {
        let kernel = KernelRef::specialized("qwen35", entry, &EXT_SHAPES);
        backend
            .prepare(kernel)
            .unwrap_or_else(|err| panic!("{entry}#600:s2,601:s16 did not resolve: {err}"));
    }
    for entry in GEMM_ENTRIES {
        let kernel = KernelRef::specialized("qwen35", entry, &GEMM_BROADCAST);
        backend
            .prepare(kernel)
            .unwrap_or_else(|err| panic!("{entry}#700:f0,701:f0 did not resolve: {err}"));
    }
}

/// A device that can answer does, and a name it cannot build is `None`
/// rather than a zero a caller would read as "no threads allowed".
#[test]
#[expect(
    clippy::print_stderr,
    reason = "an opt-in case that did not run belongs on stderr"
)]
fn a_metal_device_reports_a_block_limit_per_kernel() {
    if !shiso_metal::metal_available() {
        eprintln!("SKIP: no Metal device, so no block limit was read");
        return;
    }
    let backend = MetalBackend::new().expect("the q35 library compiles");
    let device: &dyn shiso_core::Backend = backend.as_ref();

    // A real entry point answers with a usable number. The value is a
    // property of this device and is deliberately not pinned.
    let limit = device
        .max_block_threads(KernelRef::new("qwen35", "kernel_q35_gdn_conv"))
        .expect("a resolvable kernel has a limit");
    assert!(limit >= 1, "a resolvable kernel allows at least one thread");

    // A name the library does not carry cannot be built, so there is no
    // limit to report.
    assert_eq!(
        device.max_block_threads(KernelRef::new("qwen35", "kernel_that_is_not_real")),
        None,
    );
}

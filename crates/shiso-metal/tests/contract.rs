//! The parts of the `Backend` contract that are checkable without a GPU.
//!
//! These run in the default gate: they are type-level assertions plus the
//! specialization-key behaviour the pipeline cache is built on. Nothing here
//! opens a device, so it runs on the default gate wherever the Metal backend
//! is built (see the target gate below).
// The whole file is Metal-only: without the feature, or on a target where the
// Objective-C stack is absent, `MetalBackend` does not exist and this target is
// empty rather than failing to compile.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use shiso_core::{Backend, FnConst, FnConstValue, KernelRef};

/// One specialization table, built in a `const` context: a kernel's
/// specialization is static data, never a formatted or parsed string.
const MUL_MV_CONSTS: [FnConst; 2] = [
    FnConst {
        index: 600,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: 601,
        value: FnConstValue::I16(16),
    },
];

/// The same entry point with a different tile width.
const MUL_MV_WIDER: [FnConst; 2] = [
    FnConst {
        index: 600,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: 601,
        value: FnConstValue::I16(32),
    },
];

const ENTRY: &str = "kernel_mul_mv_ext_q4_K_f32_r1_1";

#[test]
fn equal_specialization_hashes_equal() {
    // Two independent `const` definitions of the same specialization: the cache
    // key must come from the entry point and the table, not from the identity of
    // the array behind it.
    const A: KernelRef = KernelRef::specialized("generic", ENTRY, &MUL_MV_CONSTS);
    const B: KernelRef = KernelRef::specialized("generic", ENTRY, &MUL_MV_CONSTS);

    assert_eq!(A, B);
    assert_eq!(hash(&A), hash(&B));
}

#[test]
fn differing_specialization_does_not_collide() {
    const A: KernelRef = KernelRef::specialized("generic", ENTRY, &MUL_MV_CONSTS);
    const B: KernelRef = KernelRef::specialized("generic", ENTRY, &MUL_MV_WIDER);

    assert_ne!(A, B, "a different tile width is a different kernel");
    assert_ne!(
        hash(&A),
        hash(&B),
        "the cache key collapsed two specializations",
    );
}

#[test]
fn differing_entry_point_does_not_collide() {
    const A: KernelRef = KernelRef::specialized("generic", ENTRY, &MUL_MV_CONSTS);
    const B: KernelRef = KernelRef::specialized("generic", "kernel_rms_norm_f32_4", &MUL_MV_CONSTS);

    assert_ne!(A, B);
    assert_ne!(hash(&A), hash(&B));
}

#[test]
fn unspecialized_entry_point_is_a_distinct_key() {
    const PLAIN: KernelRef = KernelRef::new("generic", ENTRY);
    const SPEC: KernelRef = KernelRef::specialized("generic", ENTRY, &MUL_MV_CONSTS);

    assert!(PLAIN.consts.is_empty());
    assert_ne!(hash(&PLAIN), hash(&SPEC));
}

/// The hash the pipeline cache would see, taken through the same trait so the
/// assertion is about `KernelRef`'s own `Hash` and not a test-local encoding.
fn hash(kernel: &KernelRef) -> u64 {
    let mut hasher = DefaultHasher::new();
    kernel.hash(&mut hasher);
    hasher.finish()
}

/// `Backend: Send + Sync + Debug` is what lets an engine share one device across
/// threads, and `MetalBackend` has to keep that promise through its driver
/// handles.
#[cfg(feature = "metal")]
#[test]
fn metal_backend_satisfies_the_trait_it_claims() {
    fn assert_backend<T: Backend>() {}
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_debug<T: std::fmt::Debug>() {}
    // Object safety: the engine only ever holds `Arc<dyn Backend>`, so the
    // coercion itself is the assertion. It is never called, because coercing a
    // value would need a GPU.
    fn coerce(backend: Arc<shiso_metal::MetalBackend>) -> Arc<dyn Backend> {
        backend
    }

    assert_backend::<shiso_metal::MetalBackend>();
    assert_send_sync::<shiso_metal::MetalBackend>();
    assert_debug::<shiso_metal::MetalBackend>();
    let _ = coerce;
}

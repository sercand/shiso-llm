//! The pool sizing of a device-fitted GGUF load, apart from any device.

use shiso_core::CoreError;

use super::{Banks, Fit, fit_slots};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

fn fit(total: u64, auto: bool) -> Fit {
    Fit {
        total,
        other: 3 * GIB,
        reserve: GIB,
        auto,
        banks: Banks {
            bytes: 16 * GIB,
            slot_bytes: 4 * MIB,
            experts: 128,
        },
    }
}

#[test]
fn auto_keeps_banks_resident_when_everything_fits() {
    assert_eq!(fit_slots(&fit(20 * GIB, true)).expect("fits"), None);
    // Fit never takes the resident case, and never asks for more slots than
    // the banks hold experts.
    assert_eq!(fit_slots(&fit(20 * GIB, false)).expect("fits"), Some(4096));
}

#[test]
fn a_pool_sized_to_the_device_counts_whole_slots() {
    // 4 GiB plus 6 MiB past the fixed weights: 1025 whole 4 MiB slots.
    let slots = fit_slots(&fit(8 * GIB + 6 * MIB, true)).expect("fits");
    assert_eq!(slots, Some(1025));
}

#[test]
fn a_device_short_of_one_layer_is_unsupported() {
    // 127 slots past the fixed weights, one short of a layer's 128 experts.
    let err = fit_slots(&fit(4 * GIB + 127 * 4 * MIB, true)).expect_err("too small");
    assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
    assert_eq!(
        fit_slots(&fit(4 * GIB + 128 * 4 * MIB, false)).expect("one layer fits"),
        Some(128)
    );
    // A device smaller than the fixed weights alone.
    let err = fit_slots(&fit(2 * GIB, false)).expect_err("too small");
    assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
}

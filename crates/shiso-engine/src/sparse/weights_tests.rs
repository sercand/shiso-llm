//! The group-scaled constructors' length checks and the parts they answer,
//! on a backend that only hands out sizes.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use shiso_core::DType;
use shiso_core::backend::{Backend, KernelRef, LaunchArgs};
use shiso_core::buffer::{DeviceBuffer, DeviceView, RawBuffer, alloc};
use shiso_core::error::{CoreError, CoreResult};

use super::{MXFP8_GROUP, Weight, WeightPart, mxfp8_bytes, nvfp4_bytes};

/// Allocations with ids and lengths and no bytes: every check under test is
/// about lengths, which a buffer carries itself.
#[derive(Debug, Default)]
struct Sizes {
    next: AtomicU64,
}

impl Backend for Sizes {
    fn backend_name(&self) -> &'static str {
        "sizes"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let id = NonZeroU64::new(id).expect("ids start at one");
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

fn backend() -> Arc<dyn Backend> {
    Arc::new(Sizes::default())
}

fn buf(backend: &Arc<dyn Backend>, bytes: u64) -> DeviceBuffer {
    alloc(backend, bytes).expect("a size-only allocation")
}

/// The field a refusal names.
fn field(err: &CoreError) -> &'static str {
    match err {
        CoreError::ShapeMismatch { field, .. } => field,
        other => panic!("a length refusal, not {other:?}"),
    }
}

#[test]
fn mxfp8_bytes_are_one_code_a_weight_and_one_scale_a_group() {
    assert_eq!(mxfp8_bytes(2_560, 10_240), Some((26_214_400, 819_200)));
    assert_eq!(mxfp8_bytes(MXFP8_GROUP, 1), Some((32, 1)));
    assert_eq!(mxfp8_bytes(2_570, 16), None, "not whole groups");
    assert_eq!(mxfp8_bytes(0, 16), None, "no contraction at all");
}

#[test]
fn an_mxfp8_weight_checks_both_arrays_and_answers_its_two_parts() {
    let be = backend();
    let (in_dim, out_dim) = (2_560, 48);
    let (codes, scales) = mxfp8_bytes(in_dim, out_dim).expect("whole groups");
    let w = Weight::mxfp8(in_dim, out_dim, [buf(&be, codes), buf(&be, scales)])
        .expect("both arrays are the right length");
    assert_eq!(w.dtype, DType::Mxfp8E4M3G32);
    assert_eq!(w.slots, 1);
    assert!(w.is_resident());
    assert_eq!(w.len(), codes + scales);
    assert_eq!(w.part(WeightPart::Whole).expect("codes").len(), codes);
    assert_eq!(
        w.part(WeightPart::GroupScales).expect("scales").len(),
        scales
    );
    for refused in [
        WeightPart::GlobalScale,
        WeightPart::BlockScales,
        WeightPart::Tiles,
    ] {
        let err = w.part(refused).expect_err("an MXFP8 weight has two parts");
        assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
    }

    let err = Weight::mxfp8(in_dim, out_dim, [buf(&be, codes - 1), buf(&be, scales)])
        .expect_err("a short code array");
    assert_eq!(field(&err), "mxfp8 codes bytes");
    let err = Weight::mxfp8(in_dim, out_dim, [buf(&be, codes), buf(&be, scales + 1)])
        .expect_err("a long scale array");
    assert_eq!(field(&err), "mxfp8 group scales bytes");
    let err = Weight::mxfp8(2_570, out_dim, [buf(&be, 2_570 * 48), buf(&be, 80 * 48)])
        .expect_err("a contraction that is not whole groups");
    assert_eq!(field(&err), "mxfp8 contraction width");
}

#[test]
fn an_uploaded_mxfp8_weight_checks_the_host_arrays_before_it_allocates() {
    let be = backend();
    let w = Weight::mxfp8_uploaded(&be, 64, 3, &[0u8; 192], &[127u8; 6])
        .expect("two groups a row, three rows");
    assert_eq!(w.len(), 198);
    let err =
        Weight::mxfp8_uploaded(&be, 64, 3, &[0u8; 192], &[127u8; 5]).expect_err("one scale short");
    assert_eq!(field(&err), "mxfp8 group scales bytes");
}

#[test]
fn a_resident_nvfp4_bank_holds_every_slot_back_to_back() {
    let be = backend();
    let (in_dim, out_dim, slots) = (2_560, 640, 4);
    let (codes, scales) = nvfp4_bytes(in_dim, out_dim);
    let n = u64::from(slots);
    let w = Weight::nvfp4(
        in_dim,
        out_dim,
        slots,
        [buf(&be, codes * n), buf(&be, scales * n), buf(&be, 4 * n)],
    )
    .expect("every array is slots times one matrix");
    assert_eq!(w.dtype, DType::Nvfp4Gs16);
    assert_eq!(w.slots, slots);
    assert!(w.is_resident());
    assert_eq!(w.part(WeightPart::Whole).expect("codes").len(), codes * n);
    assert_eq!(
        w.part(WeightPart::GroupScales).expect("scales").len(),
        scales * n
    );
    assert_eq!(
        w.part(WeightPart::GlobalScale).expect("globals").len(),
        4 * n
    );
    let err = w
        .part(WeightPart::AffineScales)
        .expect_err("an NVFP4 bank has three parts");
    assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");

    for (arrays, want) in [
        ([codes * n - 1, scales * n, 4 * n], "nvfp4 codes bytes"),
        (
            [codes * n, scales * (n - 1), 4 * n],
            "nvfp4 group scales bytes",
        ),
        ([codes * n, scales * n, 4], "nvfp4 global scales bytes"),
    ] {
        let err = Weight::nvfp4(in_dim, out_dim, slots, arrays.map(|b| buf(&be, b)))
            .expect_err("one array is the wrong length");
        assert_eq!(field(&err), want);
    }
    let err = Weight::nvfp4(in_dim, out_dim, 0, [0, 0, 0].map(|b| buf(&be, b)))
        .expect_err("a bank of nothing");
    assert_eq!(field(&err), "nvfp4 bank slots");
    let err = Weight::nvfp4(40, out_dim, 1, [0, 0, 0].map(|b| buf(&be, b)))
        .expect_err("40 is not whole groups of sixteen");
    assert_eq!(field(&err), "nvfp4 contraction width");
}

//! What the router hands the driver for the two `modelopt` codecs.
//!
//! Not a numerical check - the kernels are graded in
//! `crates/shiso-kerneltest/src/qwen35moe/`, against a host codec, on a device.
//! This is the other half, and it is the half this repository's own handoff
//! says fails silently: **eleven of twelve defects found bringing this engine
//! up on CUDA were the host handing a correct kernel the wrong argument.** So
//! the launch is recorded and asserted - which entry, which grid, which block,
//! and what the sixteen bytes of the argument block actually contain.
//!
//! The backend here allocates and records and does nothing else. It is the
//! smallest thing that can hold a real [`DeviceView`], which has no public
//! constructor on purpose.

use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::{CoreError, Extent3};

use crate::recorder::{Bytes, Launch, Recorder};

use super::{ModeloptArgs, Weight, WeightAux, modelopt_dispatchable};
use shiso_core::DType;

/// A backend, a source buffer and a destination buffer for one projection.
struct Fixture {
    backend: Arc<dyn Backend>,
    recorder: Arc<Recorder>,
    src: shiso_core::DeviceBuffer,
    dst: shiso_core::DeviceBuffer,
}

fn fixture(in_dim: u32, out_dim: u32, n_tok: u32) -> Fixture {
    let recorder = Recorder::new();
    let backend: Arc<dyn Backend> = recorder.clone();
    let src = shiso_core::alloc(
        &backend,
        u64::from(in_dim) * u64::from(n_tok) * super::F32_BYTES,
    )
    .expect("allocation");
    let dst = shiso_core::alloc(
        &backend,
        u64::from(out_dim) * u64::from(n_tok) * super::F32_BYTES,
    )
    .expect("allocation");
    Fixture {
        backend,
        recorder,
        src,
        dst,
    }
}

/// The argument block the recorder captured, decoded.
fn captured(launch: &Launch) -> ModeloptArgs {
    assert_eq!(
        launch.args.len(),
        16,
        "{}: the entry declares a sixteen-byte first parameter",
        launch.entry
    );
    *bytemuck::from_bytes::<ModeloptArgs>(&launch.args)
}

#[test]
fn an_fp8_projection_names_the_matvec_at_a_decode_batch_and_the_gemm_past_it() {
    // The grid and block are asserted beside the entry because they are not
    // the same shape for the two paths and the kernels cannot refuse a wrong
    // one: the matvec is one block per (row, token) at 256 threads, and the
    // GEMM is one block per 16x16 output tile at one thread each. A matvec
    // launched on the GEMM's grid would compute forty rows' worth of output
    // into three, and nothing would fail.
    //
    // The widths are the threshold's own: one row, the widest decode batch,
    // one past it, and a prefill chunk. `GEMM_MIN_TOKENS` moved from 2 to
    // `MAX_DECODE_BATCH + 1` when the GEMM was measured at 17 GB/s against the
    // matvec's 120, so what this pins is that a decode batch takes the matvec
    // and a chunk takes the GEMM - which is the whole content of the change.
    for (n_tok, entry, grid, block) in [
        (
            1,
            "kernel_q35moe_mv_fp8_f32",
            Extent3::new(40, 1, 1),
            Extent3::new(256, 1, 1),
        ),
        (
            crate::MAX_DECODE_BATCH,
            "kernel_q35moe_mv_fp8_f32",
            Extent3::new(40, crate::MAX_DECODE_BATCH, 1),
            Extent3::new(256, 1, 1),
        ),
        (
            crate::MAX_DECODE_BATCH + 1,
            "kernel_q35moe_mm_fp8_f32",
            Extent3::new(3, 1, 1),
            Extent3::new(16, 16, 1),
        ),
        (
            512,
            "kernel_q35moe_mm_fp8_f32",
            Extent3::new(3, 32, 1),
            Extent3::new(16, 16, 1),
        ),
    ] {
        let f = fixture(64, 40, n_tok);
        let weight = Weight::import_fp8(
            &f.backend,
            64,
            40,
            Arc::new(Bytes(vec![0x38; 64 * 40])),
            0..64 * 40,
            0.25,
            Some(2.0),
        )
        .expect("an fp8 weight");
        super::projection(
            f.backend.as_ref(),
            &weight,
            n_tok,
            f.src.view(),
            f.dst.view(),
        )
        .expect("a launch");
        let launches = f.recorder.launches();
        assert_eq!(launches.len(), 1);
        let launch = &launches[0];
        assert_eq!(launch.module, "qwen35moe");
        assert_eq!(launch.entry, entry, "at {n_tok} rows");
        assert_eq!(launch.grid, grid, "at {n_tok} rows");
        assert_eq!(launch.block, block, "at {n_tok} rows");
        // Three buffers: the weight, the activations, the output. NVFP4 takes
        // four, and handing one entry the other's operand list is exactly the
        // shape of defect this file exists for.
        assert_eq!(launch.buffers(), 3);
        let args = captured(launch);
        assert_eq!(args.in_dim, 64);
        assert_eq!(args.out_dim, 40);
        assert_eq!(args.n_tok, i32::try_from(n_tok).expect("a small count"));
        // The WEIGHT scale, not the activation one. The two are both in the
        // aux and only one of them belongs in this slot.
        assert!((args.scale - 0.25).abs() < f32::EPSILON, "{}", args.scale);
    }
}

#[test]
fn an_nvfp4_projection_carries_four_buffers_and_the_per_tensor_scale() {
    let n_tok = 1;
    let f = fixture(64, 40, n_tok);
    // 64 weights a row is 32 packed bytes and 4 group scales.
    let mut host = vec![0x11u8; 32 * 40];
    host.extend(std::iter::repeat_n(0x38u8, 4 * 40));
    let weight = Weight::import_nvfp4(
        &f.backend,
        64,
        40,
        Arc::new(Bytes(host)),
        0..32 * 40,
        32 * 40..32 * 40 + 4 * 40,
        3.178_914e-5,
    )
    .expect("an nvfp4 weight");
    super::projection(
        f.backend.as_ref(),
        &weight,
        n_tok,
        f.src.view(),
        f.dst.view(),
    )
    .expect("a launch");
    let launches = f.recorder.launches();
    assert_eq!(launches.len(), 1);
    let launch = &launches[0];
    assert_eq!(launch.entry, "kernel_q35moe_mv_nvfp4_f32");
    // One WARP per output row, four rows to a block, so forty rows are ten
    // blocks of 128 - and NOT the forty blocks of 256 the FP8 entry beside it
    // still takes. The two codecs take different matvec geometries because a
    // four-bit row is half the bytes of an eight-bit one: the block-per-row
    // shape left the NVFP4 entry paying a 256-wide fold across shared memory
    // for four bytes a thread, and it measured 47-100 GB/s against the FP8
    // entry's 211 on the same decode step.
    assert_eq!(launch.grid, Extent3::new(10, 1, 1));
    assert_eq!(launch.block, Extent3::new(128, 1, 1));
    // The group scale array is a buffer of its own, between the weight and the
    // activations. An entry handed three buffers would read the activations as
    // scales and whatever followed as activations.
    assert_eq!(launch.buffers(), 4);
    let args = captured(launch);
    // The LOGICAL width, not the packed one: the entry derives `in_dim / 2`
    // and `in_dim / 16` itself, and a caller that passed 32 here would read
    // half the row under a quarter of the scales.
    assert_eq!(args.in_dim, 64);
    assert!(
        (args.scale - 3.178_914e-5).abs() < 1e-10,
        "the per-tensor scale is the one that is easy to drop: {}",
        args.scale
    );
}

#[test]
fn a_scale_array_that_does_not_cover_the_rows_is_refused_at_import() {
    let f = fixture(64, 40, 1);
    let mut host = vec![0x11u8; 32 * 40];
    // One group short of what forty rows of four groups needs.
    host.extend(std::iter::repeat_n(0x38u8, 4 * 40 - 1));
    let err = Weight::import_nvfp4(
        &f.backend,
        64,
        40,
        Arc::new(Bytes(host)),
        0..32 * 40,
        32 * 40..32 * 40 + 4 * 40 - 1,
        1.0,
    )
    .expect_err("one group short");
    match err {
        CoreError::ShapeMismatch {
            field,
            expected,
            got,
        } => {
            assert_eq!(field, "nvfp4 group scale bytes");
            assert_eq!((expected, got), (160, 159));
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn a_scale_that_cannot_multiply_is_refused_at_import() {
    // A NaN scale poisons every weight of the tensor identically, which
    // produces a uniformly NaN residual that surfaces several layers later and
    // is attributed to whatever was running then.
    let f = fixture(64, 40, 1);
    for scale in [f32::NAN, f32::INFINITY] {
        Weight::import_fp8(
            &f.backend,
            64,
            40,
            Arc::new(Bytes(vec![0x38; 64 * 40])),
            0..64 * 40,
            scale,
            None,
        )
        .expect_err("a scale that cannot multiply");
    }
}

#[test]
fn a_width_that_is_not_a_whole_number_of_groups_is_refused_at_import() {
    let f = fixture(64, 40, 1);
    // NVFP4 groups sixteen weights, and the entries derive both the packed
    // width and the scale width from `in_dim`: a row of 40 would have its tail
    // read out of the row below.
    modelopt_dispatchable(DType::Nvfp4Gs16, 40).expect_err("40 is not a multiple of 16");
    modelopt_dispatchable(DType::Nvfp4Gs16, 64).expect("64 is four groups");
    modelopt_dispatchable(DType::Nvfp4Gs16, 0).expect_err("a width of nothing");
    // FP8 has no such constraint: one byte is one weight.
    modelopt_dispatchable(DType::Fp8E4M3, 41).expect("any width at all");

    let err = Weight::import_nvfp4(
        &f.backend,
        40,
        4,
        Arc::new(Bytes(vec![0u8; 80])),
        0..80,
        0..0,
        1.0,
    )
    .expect_err("40 is not a multiple of 16");
    assert!(matches!(err, CoreError::InvalidValue { .. }), "{err:?}");
}

#[test]
fn the_two_widths_a_dtype_states_are_the_buffers_they_name() {
    // `row_bytes` answers the length of the array it is asked about, which is
    // what made `block_bytes` 8 rather than 9 for NVFP4: the `e4m3` scale byte
    // that covers a group is in an array of its own.
    assert_eq!(DType::Fp8E4M3.block_elems().get(), 1);
    assert_eq!(DType::Fp8E4M3.block_bytes().get(), 1);
    assert_eq!(DType::Nvfp4Gs16.block_elems().get(), 16);
    assert_eq!(DType::Nvfp4Gs16.block_bytes().get(), 8);
    assert!(DType::Nvfp4Gs16.is_modelopt());
    assert!(DType::Fp8E4M3.is_modelopt());
    assert!(!DType::Nvfp4Gs16.is_affine());
    assert!(!DType::Q4_K.is_modelopt());
}

#[test]
fn a_modelopt_weight_with_the_wrong_aux_is_refused_rather_than_launched() {
    // Reachable only through a loader that built the weight with the wrong
    // constructor, and its answer would be the matrix divided by a number
    // around 3e-5 rather than a failure.
    let f = fixture(64, 40, 1);
    let weight = Weight::from_buffer(
        DType::Fp8E4M3,
        64,
        40,
        shiso_core::alloc(&f.backend, 64 * 40).expect("allocation"),
    );
    assert!(weight.aux().is_none());
    let err = super::projection(f.backend.as_ref(), &weight, 1, f.src.view(), f.dst.view())
        .expect_err("no scale to decode it by");
    match err {
        CoreError::InvalidValue { key, value } => {
            assert_eq!(key, "modelopt aux");
            assert!(value.contains("fp8_e4m3"), "{value}");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn the_affine_accessor_still_answers_for_an_affine_weight_and_not_for_these() {
    let f = fixture(64, 40, 1);
    let weight = Weight::import_fp8(
        &f.backend,
        64,
        40,
        Arc::new(Bytes(vec![0x38; 64 * 40])),
        0..64 * 40,
        1.0,
        None,
    )
    .expect("an fp8 weight");
    // The embedding gather asks this question, and an FP8 weight answering
    // "yes, here are your scales and biases" would hand it two buffers that
    // are not scales and not biases.
    assert!(weight.aux().and_then(WeightAux::affine).is_none());
}

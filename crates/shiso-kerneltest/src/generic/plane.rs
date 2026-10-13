//! The staging pass that widens a whole weight plane to f32 for the vendor
//! GEMM.
//!
//! `kernel_dequant_plane_<codec>_f32` is the first half of the prefill route
//! `crates/shiso-engine/src/projection.rs`'s `blas_projection` takes: cuBLAS
//! has no quantized operand type, so the weight is written out as f32 and the
//! library multiplies that. The four entries are the four codecs
//! `launch::dequant_plane_name` answers for (`q4_K`, `q4_0`, `q8_0` and
//! `f16`), and **none of them was graded anywhere in the tree** until this
//! file. That hole is what let a scrambled `f16` plane ship: every quantized
//! codec walks its blocks correctly and `f16`, which has no block, did not.
//!
//! # What this grades that the product cases do not
//!
//! The matvec and the tiled GEMM cases in `product.rs` and
//! `qwen35/gemv.rs` grade the *dequantizer* - the arithmetic that turns a
//! packed block into four floats. This grades the **walk**: which source
//! element the kernel reads for a given destination slot. A dequantizer that
//! is bit-perfect and a walk that is off by a stride produce a plane that is
//! elementwise made of the right numbers in the wrong order, which no dot
//! product against a reference weight can see once the wrong weight has been
//! summed with the right activation.
//!
//! So the comparison here is against the staged plane itself, slot by slot,
//! and the fixture is two things at once: wide enough that a row is more than
//! one four-element chunk (the walk is correct at chunk zero for every codec,
//! including the broken one) and taller than one row (so a row-stride error is
//! a failure rather than a coincidence).
//!
//! # Why the comparison is on exact bits
//!
//! Nothing here accumulates. Each destination slot is one dequantized weight
//! written straight through, so the device's value and the host's are the same
//! arithmetic on the same inputs and the only question is whether they name
//! the same input. The oracles are the tree's own, re-used rather than
//! rewritten: `qwen35::gemv::exact_q4_k` for `q4_K`,
//! `dense::q4_0::dequant_q4_0` for `q4_0`, `generic::product::q8_matrix` for
//! `q8_0` and `generic::product::to_f16_and_back` for `f16`. A second copy of
//! any of those formulas would make this a case that grades an oracle against
//! itself.
//!
//! There is no Metal driver for these entries, and that is not an omission:
//! the staging pass exists only to feed a vendor matrix library, `as_blas()`
//! is `None` on every Metal build, and there is no
//! `kernels/metal/**/dequant_plane` to resolve. The case names no backend
//! anyway - the day a Metal twin exists, it is graded by this file from its
//! first run.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "the kernel argument struct takes i32/u64 dimensions while the fixture is written in usize, and every fixture dimension here is a small literal"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef};
use shiso_modelfile::quant::BlockQ6K;

use crate::dense::q4_0::{dequant_q4_0, pack_q4_0};
use crate::generic::product::{QK_K, QK8_0, to_f16_and_back};
use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle;
use crate::qwen35::gemv::{dequant_q6_k, exact_q4_k, pack_q6_k, q4_k_bytes};

/// Every entry point this file's case launches, which is exactly the table
/// `shiso_engine::launch::dequant_plane_name` answers from. Written out so a
/// codec that gained a staging entry and no case is a diff in this list.
pub const ENTRIES: [&str; 5] = [
    "kernel_dequant_plane_q4_K_f32",
    "kernel_dequant_plane_q4_0_f32",
    "kernel_dequant_plane_q8_0_f32",
    "kernel_dequant_plane_f16_f32",
    "kernel_dequant_plane_q6_K_f32",
];

/// The module the engine dispatches the staging pass under,
/// `shiso_engine::launch::Q35`.
const MODULE: &str = "qwen35";

/// Threads one staging block runs, mirroring
/// `shiso_engine::launch::DEQUANT_BLOCK` and `SHISO_DEQUANT_BLOCK`.
///
/// The kernel's stride loop is written in terms of the constant rather than
/// `blockDim.x`, so a launch that disagreed would leave part of every row
/// unwritten.
const DEQUANT_BLOCK: u32 = 256;

/// Elements per row of the fixture.
///
/// Three things at once. It is a whole number of blocks at all three quantized
/// codecs (six `q4_K` blocks, forty-eight of the 32-element ones), so no codec
/// needs a tail. It is `384` four-element chunks, which is **more than one**
/// and more than `DEQUANT_BLOCK`, so each thread runs the stride loop twice
/// and a walk that is correct only at its first chunk fails here. And it is a
/// multiple of four, which is the guard `blas_projection` applies to `f16`.
const WIDTH: usize = 1536;

/// Rows of the fixture. More than one so a wrong row stride is a failure, and
/// odd so a row is not confused with its neighbour by a halving.
const ROWS: usize = 3;

/// The value the destination is filled with before the launch.
///
/// Not zero: a slot the kernel never wrote and a slot it wrote a zero into are
/// different defects, and a staged plane's unwritten slot in production is not
/// a zero either - it is whatever the previous tensor left in the slab.
const POISON: f32 = -8.5e30;

/// The staging kernel's argument block
/// (`kernels/cuda/generic/dequant_plane.cuh`), mirroring
/// `shiso_engine::launch::DequantPlaneArgs`.
///
/// No padding is spelled out because the declared layout has none: two `i32`
/// fill the first eight bytes and the `u64` stride lands on its own alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DequantPlaneArgs {
    ne00: i32,
    ne01: i32,
    nb01: u64,
}

const _: () = assert!(size_of::<DequantPlaneArgs>() == 16);

/// A fixture: the packed weight bytes, the plane they dequantize to, and the
/// packed row stride the three of them agree on.
type Fixture = (Vec<u8>, Vec<f32>, u64);

/// Stages one weight plane and reads the whole destination back.
///
/// `row_bytes` is the packed row stride, which for a block codec is a whole
/// number of blocks and for `f16` is two bytes an element - the same number
/// `launch::row_bytes` computes for the real launch.
fn stage(gpu: &Gpu, entry: &'static str, weight: &[u8], row_bytes: u64) -> Vec<f32> {
    let d_weight = gpu.upload(weight);
    let dst: DeviceBuffer = gpu.upload(&f32_bytes(&vec![POISON; ROWS * WIDTH]));
    let args = DequantPlaneArgs {
        ne00: WIDTH as i32,
        ne01: ROWS as i32,
        nb01: row_bytes,
    };
    run(
        gpu,
        KernelRef::new(MODULE, entry),
        &shiso_core::LaunchArgs {
            grid: Extent3::new(ROWS as u32, 1, 1),
            block: Extent3::linear(DEQUANT_BLOCK),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d_weight.view()),
                KernelArg::Buffer(dst.view()),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&dst)
}

/// Asserts the staged plane is the reference plane bit for bit, and says which
/// source element the kernel appears to have read when it is not.
///
/// The second half of the message is what turns a wall of mismatches into a
/// diagnosis: if the value at destination slot `i` is the reference value of
/// some other slot `j`, the walk read the wrong element rather than
/// dequantized the right one wrongly, and `j - i` is the stride error.
fn same_plane(got: &[f32], want: &[f32], codec: &str) {
    assert_eq!(got.len(), want.len(), "{codec}: staged plane length");
    let mut mismatches = 0usize;
    let mut first: Option<String> = None;
    for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
        // Zero is compared by value and not by sign: `f16` holds a signed zero
        // and every codec here can produce one, but nothing downstream of a
        // staged plane can tell the two apart once the GEMM has summed it.
        if g.to_bits() == w.to_bits() || (g == 0.0 && w == 0.0) {
            continue;
        }
        mismatches += 1;
        if first.is_none() {
            let row = index / WIDTH;
            let column = index % WIDTH;
            let source = want
                .iter()
                .position(|&v| v.to_bits() == g.to_bits())
                .map_or_else(
                    || "no element of the reference plane".to_owned(),
                    |at| {
                        format!(
                            "reference element {at} (row {}, column {})",
                            at / WIDTH,
                            at % WIDTH
                        )
                    },
                );
            first = Some(format!(
                "{codec}: staged [{row}][{column}] is {g:e} ({:#010x}), expected {w:e} ({:#010x}); the value read is {source}",
                g.to_bits(),
                w.to_bits(),
            ));
        }
    }
    assert!(
        mismatches == 0,
        "{}\n({mismatches} of {} slots differ)",
        first.unwrap_or_default(),
        got.len(),
    );
}

/// The `q4_K` fixture: `ROWS * (WIDTH / 256)` blocks, each with its own scale
/// pair and its own twelve packed scale/min bytes.
///
/// The scale bytes walk the whole `0..=255` range rather than staying under
/// 64, so the four groups whose scale is spliced out of two bytes' top bits
/// are exercised as well as the four that are not.
fn q4_k_plane() -> Fixture {
    let per_row = WIDTH / QK_K;
    let mut blocks = Vec::with_capacity(ROWS * per_row);
    for index in 0..ROWS * per_row {
        let mut values = [0u8; QK_K];
        for (e, slot) in values.iter_mut().enumerate() {
            *slot = ((e * 7 + index * 5) % 16) as u8;
        }
        let mut scales = [0u8; 12];
        for (j, slot) in scales.iter_mut().enumerate() {
            *slot = ((j * 37 + index * 91) % 256) as u8;
        }
        blocks.push(oracle::Q4K {
            d: half::f16::from_f32(0.0123 + index as f32 * 0.0007).to_bits(),
            dmin: half::f16::from_f32(index as f32 * 0.0003 - 0.0041).to_bits(),
            scales,
            qs: oracle::pack_q4_k_dense(&values),
        });
    }
    let mut want = vec![0.0f32; ROWS * WIDTH];
    for row in 0..ROWS {
        for b in 0..per_row {
            let block = &blocks[row * per_row + b];
            for e in 0..QK_K {
                want[row * WIDTH + b * QK_K + e] = exact_q4_k(block, e);
            }
        }
    }
    let row_bytes = (per_row * 144) as u64;
    (q4_k_bytes(&blocks), want, row_bytes)
}

/// The `q4_0` fixture: one packed row per output row, off the shared signal so
/// no two rows hold the same numbers.
fn q4_0_plane() -> Fixture {
    let mut packed = Vec::new();
    let mut want = Vec::with_capacity(ROWS * WIDTH);
    let mut row_bytes = 0u64;
    for row in 0..ROWS {
        let bytes = pack_q4_0(&oracle::signal(WIDTH, 0x4D07 ^ (row as u32 * 0x9E37)));
        row_bytes = bytes.len() as u64;
        want.extend(dequant_q4_0(&bytes));
        packed.extend(bytes);
    }
    (packed, want, row_bytes)
}

/// The `q8_0` fixture, straight off the matrix builder the product cases use.
fn q8_0_plane() -> Fixture {
    let (blocks, want) = crate::generic::product::q8_matrix(ROWS, WIDTH);
    let row_bytes = ((WIDTH / QK8_0) * 34) as u64;
    (bytemuck::cast_slice(&blocks).to_vec(), want, row_bytes)
}

/// The `q6_K` fixture: `ROWS * (WIDTH / 256)` blocks of 210 bytes, so every
/// odd block starts off a four-byte boundary, each with its own codes.
fn q6_k_plane() -> Fixture {
    let per_row = WIDTH / QK_K;
    let blocks: Vec<BlockQ6K> = (0..ROWS * per_row)
        .map(|index| {
            let mut values = [0u8; QK_K];
            for (e, slot) in values.iter_mut().enumerate() {
                *slot = ((e * 11 + index * 7) % 64) as u8;
            }
            pack_q6_k(&values)
        })
        .collect();
    let mut want = vec![0.0f32; ROWS * WIDTH];
    for (index, block) in blocks.iter().enumerate() {
        for e in 0..QK_K {
            want[index * QK_K + e] = dequant_q6_k(block, e);
        }
    }
    let bytes: Vec<u8> = blocks
        .iter()
        .flat_map(bytemuck::bytes_of)
        .copied()
        .collect();
    (bytes, want, (per_row * size_of::<BlockQ6K>()) as u64)
}

/// The `f16` fixture: the shared signal rounded to half and back, which is
/// what an unquantized weight actually holds.
fn f16_plane() -> Fixture {
    let (bytes, want) = to_f16_and_back(&oracle::signal(ROWS * WIDTH, 0xF167));
    (bytes, want, (WIDTH * 2) as u64)
}

/// The staged plane is the weight plane, element for element, at every codec
/// the vendor route can name.
///
/// Walked over all five rather than spot-checked at one, because the walk is
/// per-codec: `shiso_dequant_plane_body` steps its source pointer by
/// `shiso_dequant_plane_chunks<codec>()`, so a codec whose type is not a block
/// struct has its own pointer arithmetic and passing at the other four says
/// nothing about it. `f16` is the codec that is not a block, and it is the one
/// that was wrong.
///
/// # Panics
///
/// When any staged element differs from the reference plane, which is what the
/// case is for.
pub fn the_staged_plane_matches_the_weight_at_every_codec(gpu: &Gpu) {
    let cases: [(&str, &'static str, Fixture); 5] = [
        ("q4_K", ENTRIES[0], q4_k_plane()),
        ("q4_0", ENTRIES[1], q4_0_plane()),
        ("q8_0", ENTRIES[2], q8_0_plane()),
        ("f16", ENTRIES[3], f16_plane()),
        ("q6_K", ENTRIES[4], q6_k_plane()),
    ];
    for (codec, entry, (weight, want, row_bytes)) in cases {
        assert_eq!(
            weight.len(),
            row_bytes as usize * ROWS,
            "{codec}: the fixture is {ROWS} whole rows",
        );
        assert_eq!(want.len(), ROWS * WIDTH, "{codec}: the reference plane");
        let got = stage(gpu, entry, &weight, row_bytes);
        same_plane(&got, &want, codec);
    }
}

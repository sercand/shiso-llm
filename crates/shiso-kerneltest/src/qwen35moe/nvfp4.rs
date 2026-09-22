//! The NVFP4 projection on the device, graded against the host unpacker over
//! real tensor corners.
//!
//! Two entries and one law:
//!
//! ```text
//! dst[out + tok * out_dim] = sum_in w[out][in] * x[in + tok * in_dim]
//! w[r][k] = e2m1(nibble) * (e4m3(scale[r][k / 16]) * global)
//! ```
//!
//! # The fixtures are real corners, and that is the case's whole strength
//!
//! A synthetic fixture would carry a `weight_scale_2` of 1, and **a kernel
//! that dropped the per-tensor level entirely would pass every case built on
//! one**. These corners come out of the checkpoint with their own globals -
//! about 3e-5 and 1e-4 - so a kernel that drops the level is wrong by about
//! 31,458x on the first element of the first row and the case says so.
//!
//! Two tensors, because their group counts per row differ in the real file
//! (2,048 inputs in 128 groups against 512 in 32), so a scale-array stride
//! taken from the wrong axis passes on one and fails on the other. Neither row
//! count is a multiple of the GEMM's 16-wide tile.
//!
//! The expected values are the host unpacker's, not torch's. `nvfp4.rs` is
//! already graded bit for bit against torch, so a torch decode here too would
//! grade the device against a transcript rather than against a codec.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the kernel argument struct takes i32 dimensions while the fixtures are \
              written in usize, and every fixture dimension here is a small literal"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::nvfp4::{Nvfp4Layout, dequant_nvfp4};

mod fixtures {
    include!("nvfp4_fixtures.rs");
}

/// The module every entry in this file lives in.
const MODULE: &str = "qwen35moe";

/// Output rows one block of the matvec entry covers, which is
/// `SHISO_Q35MOE_MV_ROWS` in `kernels/cuda/qwen35moe/q35moe_common.cuh` and
/// `MODELOPT_MV_ROWS` in `crates/shiso-engine/src/projection.rs`.
///
/// **This case launched the entry at `(out_dim, 1, 1)` by 256 threads for a
/// phase after the entry became one warp per row, and passed.** It passed
/// because the wrong geometry is redundant rather than wrong here: with a
/// block per row and four rows to a block, block `b` covers rows `4b..4b+8`,
/// every row below `out_dim` is still written and every write is still the
/// right value for the row it lands on. So the case graded arithmetic
/// correctly while grading a launch nobody performs. It is spelled from the
/// same two constants the engine uses for that reason.
const MV_ROWS: u32 = 4;

/// Threads per block of the matvec entry, which is `SHISO_Q35MOE_MV_BLOCK`:
/// one warp per output row.
const MV_BLOCK: u32 = 32 * MV_ROWS;

/// Tile edge of the GEMM entry, which is `SHISO_Q35MOE_TILE`.
const TILE: u32 = 16;

/// What a dot product here is allowed to diverge from a sequential one by, as
/// a fraction of what the sum accumulated.
///
/// The same four `f32` ulps the FP8 cases use, and for the same reason - a
/// tree fold against a left-to-right one. Measured on this GB10 over these
/// corners: the numbers each case prints. It is a shared constant rather than
/// two because the two codecs differ in how a weight is produced and not at
/// all in how the products are summed, so a budget that differed between them
/// would be saying something about the codecs that is not true.
const BUDGET: f32 = 4.0 * f32::EPSILON;

/// The widths and the per-tensor scale an NVFP4 projection is launched over.
///
/// `in_dim` is the LOGICAL width: the packed row is `in_dim / 2` bytes and the
/// scale row is `in_dim / 16`, both derived inside the kernel rather than
/// passed, so a caller cannot state a packed width that disagrees with the
/// logical one. Sixteen bytes, and the PTX declares it as one
/// `.param .align 4 .b8 ..._param_0[16]`, so it is bound as bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Nvfp4Args {
    in_dim: i32,
    out_dim: i32,
    n_tok: i32,
    global_scale: f32,
}
const _: () = assert!(size_of::<Nvfp4Args>() == 16);

/// One reference product: the values, and what each of them accumulated.
///
/// The second half is what the divergence is normalized by; see
/// [`crate::qwen35moe::fp8`] for the argument, which is the same one.
struct Reference {
    values: Vec<f32>,
    magnitude: Vec<f32>,
}

/// Activations that are not all one magnitude, so a dot product cannot be
/// right by cancellation.
fn activations(n_tok: usize, in_dim: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(n_tok * in_dim);
    for tok in 0..n_tok {
        for col in 0..in_dim {
            let i = ((tok * in_dim + col) % 65_536) as u16;
            values.push(
                f32::from(i % 197).mul_add(0.013, -1.3) * if col % 4 == 0 { -1.0 } else { 1.0 },
            );
        }
    }
    values
}

/// The reference product, from the host unpacker and a plain loop.
fn reference(
    weights: &[u8],
    scales: &[u8],
    global: f32,
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    n_tok: usize,
) -> Reference {
    let mut decoded = Vec::new();
    dequant_nvfp4(
        Nvfp4Layout::new(out_dim, in_dim, global),
        weights,
        scales,
        &mut decoded,
    )
    .expect("the fixture's own geometry");
    let mut values = vec![0.0f32; n_tok * out_dim];
    let mut magnitude = vec![0.0f32; n_tok * out_dim];
    for tok in 0..n_tok {
        for row in 0..out_dim {
            let mut sum = 0.0f32;
            let mut carried = 0.0f32;
            for col in 0..in_dim {
                let term = decoded[row * in_dim + col] * x[tok * in_dim + col];
                sum += term;
                carried += term.abs();
            }
            values[tok * out_dim + row] = sum;
            magnitude[tok * out_dim + row] = carried;
        }
    }
    Reference { values, magnitude }
}

/// Compares and reports, naming the worst divergence and the budget.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
fn assert_close(got: &[f32], want: &Reference, budget: f32, label: &str) {
    assert_eq!(got.len(), want.values.len(), "{label}: element count");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (a, b)) in got.iter().zip(&want.values).enumerate() {
        let scale = want.magnitude[i].max(f32::MIN_POSITIVE);
        let relative = (a - b).abs() / scale;
        if relative > worst {
            worst = relative;
            worst_at = i;
        }
    }
    println!(
        "{label}: worst divergence {worst:e} of what the sum carried, at element {worst_at}, \
         budget {budget:e}"
    );
    assert!(
        worst <= budget,
        "{label}: element {worst_at} was {} against {} over a sum carrying {} \
         ({worst:e}, budget {budget:e})",
        got[worst_at],
        want.values[worst_at],
        want.magnitude[worst_at],
    );
}

/// One fixture corner, as the cases pass it around.
struct Corner {
    weights: &'static [u8],
    scales: &'static [u8],
    global: f32,
    rows: usize,
    cols: usize,
    name: &'static str,
}

/// The routed expert's gate projection: 2,048 inputs in 128 groups per row.
fn expert() -> Corner {
    Corner {
        weights: &fixtures::EXPERT_WEIGHTS,
        scales: &fixtures::EXPERT_SCALES,
        global: fixtures::EXPERT_GLOBAL,
        rows: fixtures::EXPERT_ROWS,
        cols: fixtures::EXPERT_COLS,
        name: "experts.0.gate_proj",
    }
}

/// The shared expert's down projection: 512 inputs in 32 groups per row, which
/// is the axis a scale stride taken from the wrong dimension gets wrong.
fn shared_down() -> Corner {
    Corner {
        weights: &fixtures::SHARED_DOWN_WEIGHTS,
        scales: &fixtures::SHARED_DOWN_SCALES,
        global: fixtures::SHARED_DOWN_GLOBAL,
        rows: fixtures::SHARED_DOWN_ROWS,
        cols: fixtures::SHARED_DOWN_COLS,
        name: "shared_expert.down_proj",
    }
}

/// The device's own decode of every four-bit code, against the host table.
///
/// The sibling of `every_e4m3_code_decodes_the_way_the_host_decodes_it`, and
/// it exists for a reason that case did not have: the kernels stopped reading
/// `SHISO_E2M1` and started computing the same value from the code's bits.
/// `__constant__` is broadcast storage and every lane of a matvec holds a
/// different code, so the table was serializing eight ways on every decode -
/// 41.4% of this family's warp stall cycles by `ncu`'s accounting. The
/// arithmetic that replaced it claims to be **bit-identical**, and a claim of
/// bit-identity over sixteen inputs is one a case can settle rather than
/// sample.
///
/// One row per code, sixteen weights to a row because a row of this format is
/// a whole number of groups: weight zero of row `r` is code `r` and the other
/// fifteen are zero, the activation is `1` in its first element and zero
/// after, and both scale levels are one - so `dst[r]` IS the decode.
///
/// # Panics
///
/// When the launch is refused or a code decodes differently.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
pub fn every_e2m1_code_decodes_the_way_the_host_table_does(gpu: &Gpu) {
    /// Codes the format defines, which is every four-bit pattern.
    const CODES: usize = 16;
    /// Weights in one NVFP4 group, and so the narrowest row this format has.
    const GROUP: usize = 16;
    /// The `e4m3` byte for exactly 1.0: exponent field 7, mantissa 0.
    const ONE_E4M3: u8 = 0x38;
    /// The host's own table, from `kernels/cuda/qwen35moe/q35moe_common.cuh`
    /// and from `crates/shiso-modelfile/src/nvfp4.rs` before it.
    const TABLE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    // Row `r`: the low nibble of its first byte is code `r`, everything else
    // is code 0. The low nibble is the earlier weight, which is the one
    // convention in this format the checkpoint cannot settle.
    let mut packed = vec![0u8; CODES * GROUP / 2];
    for (row, code) in (0..CODES).enumerate() {
        packed[row * GROUP / 2] = code as u8;
    }
    let scales = [ONE_E4M3; CODES];
    let mut x = vec![0.0f32; GROUP];
    x[0] = 1.0;

    let args = Nvfp4Args {
        in_dim: GROUP as i32,
        out_dim: CODES as i32,
        n_tok: 1,
        global_scale: 1.0,
    };
    let w = gpu.upload(&packed);
    let sc = gpu.upload(&scales);
    let src = gpu.upload(&f32_bytes(&x));
    let dst = gpu.upload(&f32_bytes(&[f32::from_bits(0xA5A5_A5A5); CODES]));
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_q35moe_mv_nvfp4_f32"),
        &LaunchArgs {
            grid: Extent3::new((CODES as u32).div_ceil(MV_ROWS), 1, 1),
            block: Extent3::new(MV_BLOCK, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(sc.view()),
                KernelArg::Buffer(src.view()),
                KernelArg::Buffer(dst.view()),
            ],
        },
    );
    drain(gpu);
    let got = gpu.read_f32(&dst);

    let mut mismatches = 0;
    for (code, value) in got.iter().enumerate().take(CODES) {
        let magnitude = TABLE[code & 0x7];
        let want = if code & 0x8 == 0 {
            magnitude
        } else {
            -magnitude
        };
        // By bits, with the same one exception `every_e4m3_code_...` carries
        // and for the same reason: code 8 decodes to negative zero and the
        // accumulator starts at positive zero, so no dot product can carry the
        // sign of a zero. The decode is right; a sum is what loses it.
        let differs = if *value == 0.0 && want == 0.0 {
            false
        } else {
            value.to_bits() != want.to_bits()
        };
        if differs {
            mismatches += 1;
            println!(
                "code {code:#x}: device {value} (0x{:08x}), host {want} (0x{:08x})",
                value.to_bits(),
                want.to_bits()
            );
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches} of {CODES} e2m1 codes decoded differently on {}",
        gpu.backend_name(),
    );
    println!(
        "all {CODES} e2m1 codes decode identically to the host table on {} (zero compared by \
         value, not by sign: a sum cannot carry it)",
        gpu.backend_name()
    );
}

/// Runs the matvec over one corner and returns what the device wrote.
fn matvec(gpu: &Gpu, corner: &Corner, x: &[f32], n_tok: usize) -> Vec<f32> {
    let args = Nvfp4Args {
        in_dim: corner.cols as i32,
        out_dim: corner.rows as i32,
        n_tok: n_tok as i32,
        global_scale: corner.global,
    };
    let w = gpu.upload(corner.weights);
    let s = gpu.upload(corner.scales);
    let src = gpu.upload(&f32_bytes(x));
    let dst = gpu.upload(&f32_bytes(&vec![
        f32::from_bits(0xA5A5_A5A5);
        n_tok * corner.rows
    ]));
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_q35moe_mv_nvfp4_f32"),
        &LaunchArgs {
            grid: Extent3::new((corner.rows as u32).div_ceil(MV_ROWS), n_tok as u32, 1),
            block: Extent3::new(MV_BLOCK, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(s.view()),
                KernelArg::Buffer(src.view()),
                KernelArg::Buffer(dst.view()),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&dst)
}

/// Runs the GEMM over one corner and returns what the device wrote.
fn gemm(gpu: &Gpu, corner: &Corner, x: &[f32], n_tok: usize) -> Vec<f32> {
    let args = Nvfp4Args {
        in_dim: corner.cols as i32,
        out_dim: corner.rows as i32,
        n_tok: n_tok as i32,
        global_scale: corner.global,
    };
    let w = gpu.upload(corner.weights);
    let s = gpu.upload(corner.scales);
    let src = gpu.upload(&f32_bytes(x));
    let dst = gpu.upload(&f32_bytes(&vec![
        f32::from_bits(0xA5A5_A5A5);
        n_tok * corner.rows
    ]));
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_q35moe_mm_nvfp4_f32"),
        &LaunchArgs {
            grid: Extent3::new(
                (corner.rows as u32).div_ceil(TILE),
                (n_tok as u32).div_ceil(TILE),
                1,
            ),
            block: Extent3::new(TILE, TILE, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(s.view()),
                KernelArg::Buffer(src.view()),
                KernelArg::Buffer(dst.view()),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&dst)
}

/// The matvec over both corners, against the host unpacker.
///
/// # Panics
///
/// When a launch is refused or a product diverges past the budget.
pub fn nvfp4_matvec_matches_the_host_unpacker(gpu: &Gpu) {
    for corner in [expert(), shared_down()] {
        const TOK: usize = 3;
        let x = activations(TOK, corner.cols);
        let want = reference(
            corner.weights,
            corner.scales,
            corner.global,
            &x,
            corner.rows,
            corner.cols,
            TOK,
        );
        let got = matvec(gpu, &corner, &x, TOK);
        assert_close(
            &got,
            &want,
            BUDGET,
            &format!("nvfp4 matvec, {}", corner.name),
        );
    }
}

/// The tiled GEMM over both corners, at row counts that are not whole tiles.
///
/// # Panics
///
/// When a launch is refused or a product diverges past the budget.
pub fn nvfp4_gemm_matches_the_host_unpacker(gpu: &Gpu) {
    for corner in [expert(), shared_down()] {
        // 7 tokens is not half a tile, and 40 and 17 rows are 2.5 and 1.06 of
        // one. An unguarded edge reads a row past the matrix and writes a row
        // past the output, and the destination is poisoned rather than zeroed
        // so the second of those is a failure rather than a value that was
        // already right.
        const TOK: usize = 7;
        let x = activations(TOK, corner.cols);
        let want = reference(
            corner.weights,
            corner.scales,
            corner.global,
            &x,
            corner.rows,
            corner.cols,
            TOK,
        );
        let got = gemm(gpu, &corner, &x, TOK);
        assert_close(&got, &want, BUDGET, &format!("nvfp4 gemm, {}", corner.name));
    }
}

/// A kernel that drops the per-tensor scale fails this case, and nothing else.
///
/// The level is the one that is easy to drop and costs about 31,458x when it
/// is dropped, which the next normalization very nearly hides. This asserts
/// that the fixtures can actually see that: run the same corner through the
/// host unpacker twice, once with its real global and once with a global of
/// one, and the two must be far apart. If they were not - if the corner's
/// global happened to be 1 - every other case in this file would pass a kernel
/// that ignored the argument, and this case is what says they do not.
///
/// # Panics
///
/// When a launch is refused, or when the fixture cannot tell the two apart.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
pub fn the_per_tensor_scale_is_visible_in_these_fixtures(gpu: &Gpu) {
    for corner in [expert(), shared_down()] {
        const TOK: usize = 1;
        let x = activations(TOK, corner.cols);
        let with = reference(
            corner.weights,
            corner.scales,
            corner.global,
            &x,
            corner.rows,
            corner.cols,
            TOK,
        );
        let without = reference(
            corner.weights,
            corner.scales,
            1.0,
            &x,
            corner.rows,
            corner.cols,
            TOK,
        );
        let ratio = 1.0 / corner.global;
        println!(
            "{}: global {:e}, so dropping it rescales by {ratio:.0}x",
            corner.name, corner.global
        );
        assert!(
            ratio > 1_000.0,
            "{}: a fixture whose global is {} cannot catch a kernel that ignores it",
            corner.name,
            corner.global,
        );
        // Not a tolerance: the two references must be different answers, not
        // nearly-equal ones.
        let first_with = with.values[0].abs();
        let first_without = without.values[0].abs();
        assert!(
            first_without > first_with * 1_000.0,
            "{}: {first_without} against {first_with} is not the factor the global is",
            corner.name,
        );

        // And the device agrees with the one that has it.
        let got = matvec(gpu, &corner, &x, TOK);
        assert_close(
            &got,
            &with,
            BUDGET,
            &format!("nvfp4 matvec with the global, {}", corner.name),
        );
    }
}

/// The two entries agree with each other on the same weights.
///
/// The host picks between them on the row count, so a seam here is a model
/// that answers differently depending on how the prompt was chunked.
///
/// # Panics
///
/// When a launch is refused or the two paths disagree.
pub fn the_matvec_and_the_gemm_agree_on_the_same_weights(gpu: &Gpu) {
    for corner in [expert(), shared_down()] {
        const TOK: usize = 5;
        let x = activations(TOK, corner.cols);
        let want = Reference {
            magnitude: reference(
                corner.weights,
                corner.scales,
                corner.global,
                &x,
                corner.rows,
                corner.cols,
                TOK,
            )
            .magnitude,
            values: matvec(gpu, &corner, &x, TOK),
        };
        let got = gemm(gpu, &corner, &x, TOK);
        assert_close(
            &got,
            &want,
            BUDGET,
            &format!("nvfp4 gemm against matvec, {}", corner.name),
        );
    }
}

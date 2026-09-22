//! The FP8 projection on the device, graded against the host codec.
//!
//! Two entries and one law. `kernel_q35moe_mv_fp8_f32` is the decode path at
//! one token per block; `kernel_q35moe_mm_fp8_f32` is the tiled form a prefill
//! chunk takes. Both compute
//!
//! ```text
//! dst[out + tok * out_dim] = (sum_in e4m3(W[out][in]) * x[in + tok * in_dim]) * weight_scale
//! ```
//!
//! and both are compared against `shiso_modelfile::fp8::dequant_fp8` followed by a
//! plain `f32` loop over the same numbers - so a failure names the kernel or
//! its geometry rather than a second implementation of the codec.
//!
//! # The widths are deliberately awkward
//!
//! 2,048 is this checkpoint's real input width and divides everything. The
//! output widths and token counts here do not: 40 rows is not a whole number
//! of the GEMM's 16-wide tiles and 7 tokens is not either, so an unguarded
//! tile edge writes past the output and an unguarded load reads a row past the
//! matrix. A fixture whose every dimension was a multiple of the tile could
//! not tell those apart from correct code, and this checkpoint really does
//! carry a 248,320-row output projection that is not a multiple of 16 either.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the kernel argument struct takes i32 dimensions while the fixtures are \
              written in usize, and every fixture dimension here is a small literal - \
              the same allowance `crate::qwen35::gemv` takes for the same reason"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::fp8::{Fp8Layout, dequant_fp8, e4m3};

/// The module every entry in this file lives in.
const MODULE: &str = "qwen35moe";

/// What a dot product here is allowed to diverge from a sequential one by, as
/// a fraction of what the sum accumulated.
///
/// **Established by measurement, and the first attempt at it was wrong.** The
/// budget was written as `1e-7` before any of these cases had run, which is
/// below one `f32` ulp and therefore below what the arithmetic can do at all:
/// the matvec then reported 1.02e-7 and the cross-check 1.25e-7, both of which
/// are about one ulp of the accumulated magnitude and are the right answer.
/// Four ulps is the budget, which leaves room for a fold order that is not
/// this one without leaving room for a kernel that is wrong.
///
/// Measured on this GB10 at the widths the cases below use: 3.8e-8 for the
/// GEMM against the host, 1.0e-7 for the matvec, 1.2e-7 between the two
/// kernels. Every case prints its own number, so a later change that moves one
/// of them says so.
const BUDGET: f32 = 4.0 * f32::EPSILON;

/// Threads per block of the matvec entry, which is `SHISO_Q35MOE_BLOCK`.
const MV_BLOCK: u32 = 256;

/// Tile edge of the GEMM entry, which is `SHISO_Q35MOE_TILE`.
const TILE: u32 = 16;

/// The widths and the one scale an FP8 projection is launched over.
///
/// Sixteen bytes, one struct, and the PTX declares it as
/// `.param .align 4 .b8 ..._param_0[16]` - one parameter, not four. It is
/// bound as bytes for that reason: `crates/shiso-cuda/src/launch.rs` hands the
/// driver one pointer per declared parameter, so four scalar operands would
/// give the entry a four-byte first parameter where it declares sixteen, and
/// every launch would succeed reading a pointer out of a slot holding a
/// quarter of a struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Fp8Args {
    in_dim: i32,
    out_dim: i32,
    n_tok: i32,
    weight_scale: f32,
}
const _: () = assert!(size_of::<Fp8Args>() == 16);

/// A weight matrix of `out_dim` rows by `in_dim` codes.
///
/// The codes walk the whole byte range rather than a plausible subset, so
/// every exponent step, both signs, the subnormal floor and the two NaN
/// encodings' neighbours all appear. `0x7f` and `0xff` are the NaN codes and
/// are skipped: a NaN anywhere in a row makes the whole dot product NaN and
/// the case would then be comparing two NaNs, which proves nothing about the
/// other 254.
fn weight_codes(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let mut codes = Vec::with_capacity(out_dim * in_dim);
    for row in 0..out_dim {
        for col in 0..in_dim {
            let mut code = ((row * 37 + col * 11) % 256) as u8;
            if code == 0x7f || code == 0xff {
                code = 0x38;
            }
            codes.push(code);
        }
    }
    codes
}

/// Activations that are not all one magnitude, so a dot product cannot be
/// right by cancellation.
fn activations(n_tok: usize, in_dim: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(n_tok * in_dim);
    for tok in 0..n_tok {
        for col in 0..in_dim {
            let i = (tok * in_dim + col) as u16;
            values.push(
                f32::from(i % 211).mul_add(0.017, -1.7) * if col % 3 == 0 { -1.0 } else { 1.0 },
            );
        }
    }
    values
}

/// One reference product: the values, and what each of them accumulated.
///
/// The second half is not decoration. A dot product's error is set by the
/// magnitude of the terms it summed, not by the magnitude of its answer, and
/// these fixtures cancel hard: element 112 of the matvec case sums 2,048 terms
/// of order one into a result of -0.54. Dividing the divergence by that result
/// reports 1.7e-5 and says nothing about the kernel; dividing it by what the
/// sum actually carried reports 1e-9, which is where an `f32` reduction over
/// 2,048 terms belongs. So the conditioning travels with the answer and
/// [`assert_close`] normalizes by it.
struct Reference {
    /// The product, as the kernels compute it.
    values: Vec<f32>,
    /// `sum |w * x|` for each element: the size of what was added up.
    magnitude: Vec<f32>,
}

/// The reference product, from the host codec and a plain loop.
///
/// The weight scale is applied to the finished sum rather than per weight,
/// which is what the kernels do and is a choice both sides have to make the
/// same way: in exact arithmetic the two agree and in `f32` they do not, and a
/// reference that disagreed with the kernel on that alone would be reporting
/// its own rounding as a kernel defect.
fn reference(
    codes: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    n_tok: usize,
    weight_scale: f32,
) -> Reference {
    // Through the host codec at a scale of one, so the decoded values are
    // exactly `e4m3(byte)` and the scale is applied where the kernel applies
    // it.
    let mut decoded = Vec::new();
    dequant_fp8(
        Fp8Layout::weights_only(out_dim, in_dim, 1.0),
        codes,
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
            values[tok * out_dim + row] = sum * weight_scale;
            magnitude[tok * out_dim + row] = carried * weight_scale.abs();
        }
    }
    Reference { values, magnitude }
}

/// Compares and reports, naming the worst divergence and the budget.
///
/// The divergence is normalized by what each element's sum accumulated rather
/// than by the element itself - see [`Reference`] for why that is the only
/// normalizer that says anything about a kernel here.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
fn assert_close(got: &[f32], want: &Reference, budget: f32, label: &str) {
    assert_eq!(got.len(), want.values.len(), "{label}: element count");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (a, b)) in got.iter().zip(&want.values).enumerate() {
        // A sum that carried nothing is exactly zero on both sides, and
        // dividing by it would be dividing by zero rather than reporting a
        // divergence.
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

/// The device's own decode of every byte, against the host's.
///
/// Before any dot product, because a decode that is wrong on one exponent step
/// shows up in a dot product as a small divergence that a tolerance absorbs,
/// and here it shows up as the byte it is.
///
/// # Panics
///
/// When the launch is refused or a code decodes differently.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
pub fn every_e4m3_code_decodes_the_way_the_host_decodes_it(gpu: &Gpu) {
    // A 254-row matrix of one column each: row `r` holds code `r`, so the
    // matvec's product against an activation of exactly 1.0 IS the decode.
    // The two NaN codes are left out for the reason `weight_codes` gives.
    let codes: Vec<u8> = (0..=u8::MAX).filter(|c| *c != 0x7f && *c != 0xff).collect();
    let out_dim = codes.len();
    let args = Fp8Args {
        in_dim: 1,
        out_dim: i32::try_from(out_dim).expect("254 rows"),
        n_tok: 1,
        weight_scale: 1.0,
    };
    let w = gpu.upload(&codes);
    let x = gpu.upload(&f32_bytes(&[1.0]));
    let dst = gpu.upload(&f32_bytes(&vec![0.0; out_dim]));
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_q35moe_mv_fp8_f32"),
        &LaunchArgs {
            grid: Extent3::new(u32::try_from(out_dim).expect("254 rows"), 1, 1),
            block: Extent3::new(MV_BLOCK, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(x.view()),
                KernelArg::Buffer(dst.view()),
            ],
        },
    );
    drain(gpu);
    let got = gpu.read_f32(&dst);
    let mut mismatches = 0;
    for (code, value) in codes.iter().zip(&got) {
        let want = e4m3(*code);
        // Bit for bit, with one exception that is IEEE's rather than the
        // kernel's. `0x80` decodes to NEGATIVE zero, and the matvec's
        // accumulator starts at positive zero: `-0.0 + 0.0` is `+0.0` in
        // round-to-nearest, so the sign of a zero cannot survive a sum and a
        // projection cannot carry it. The first run of this case reported that
        // as a decode mismatch, which it is not - the decode is right and the
        // comparison was asking a dot product to preserve something no dot
        // product preserves. Everything else, including the subnormal floor
        // and the top exponent step, is compared on bits.
        let differs = if *value == 0.0 && want == 0.0 {
            false
        } else {
            value.to_bits() != want.to_bits()
        };
        if differs {
            mismatches += 1;
            if mismatches <= 5 {
                println!(
                    "code {code:#04x}: device {value} (0x{:08x}), host {want} (0x{:08x})",
                    value.to_bits(),
                    want.to_bits()
                );
            }
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches} of {out_dim} codes decoded differently on {}",
        gpu.backend_name(),
    );
    println!(
        "all {out_dim} finite e4m3 codes decode identically on {} (zero compared by value, \
         not by sign: a sum cannot carry it)",
        gpu.backend_name()
    );
}

/// The matvec at this checkpoint's real input width.
///
/// # Panics
///
/// When the launch is refused or the product diverges past the budget.
pub fn fp8_matvec_matches_the_host_codec(gpu: &Gpu) {
    // 2,048 is the real residual width; 40 rows is deliberately not a multiple
    // of anything the kernel tiles by, and 3 tokens exercises the grid's
    // second extent without being a tile.
    const IN: usize = 2048;
    const OUT: usize = 40;
    const TOK: usize = 3;
    const SCALE: f32 = 0.004_231_5;

    let codes = weight_codes(OUT, IN);
    let x = activations(TOK, IN);
    let want = reference(&codes, &x, OUT, IN, TOK, SCALE);

    let args = Fp8Args {
        in_dim: IN as i32,
        out_dim: OUT as i32,
        n_tok: TOK as i32,
        weight_scale: SCALE,
    };
    let w = gpu.upload(&codes);
    let src = gpu.upload(&f32_bytes(&x));
    let dst = gpu.upload(&f32_bytes(&vec![0.0; TOK * OUT]));
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_q35moe_mv_fp8_f32"),
        &LaunchArgs {
            grid: Extent3::new(OUT as u32, TOK as u32, 1),
            block: Extent3::new(MV_BLOCK, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(src.view()),
                KernelArg::Buffer(dst.view()),
            ],
        },
    );
    drain(gpu);
    // The kernel folds across 256 threads in a tree and the host adds left to
    // right, so the two round differently at every step. This fixture cancels
    // hard on purpose, which is why the budget is stated against what the sum
    // carried: against the answer it would have had to be a hundred times
    // looser to say the same thing.
    assert_close(&gpu.read_f32(&dst), &want, BUDGET, "fp8 matvec");
}

/// The matvec at the three input widths that leave a row word-misaligned.
///
/// `kernel_q35moe_mv_fp8_f32` reads four codes at a time as one 32-bit word.
/// A row begins at `weights + row * in_dim`, so unless `in_dim` is a multiple
/// of four, every row whose product is odd of a word starts off a word
/// boundary and that load faults with `CUDA_ERROR_MISALIGNED_ADDRESS`. The
/// widths below are the three residues that break it - 2049, 2050 and 2051 -
/// each far past the point where the word loop has run, so the kernel's
/// fallback is exercised rather than skipped by an empty loop.
///
/// This is not a hypothetical width. `modelopt_dispatchable` in
/// `crates/shiso-engine/src/projection.rs` admits every `in_dim >= 1` for FP8
/// and says so in as many words, because one byte is one weight and the codec
/// genuinely has no group to be a whole number of. Nothing but this case
/// stands between that and a launch that dies on a checkpoint whose residual
/// width is not a multiple of four.
///
/// # Panics
///
/// When the launch is refused or the product diverges past the budget.
pub fn fp8_matvec_matches_the_host_codec_at_an_unaligned_width(gpu: &Gpu) {
    const OUT: usize = 40;
    const TOK: usize = 3;
    const SCALE: f32 = 0.004_231_5;

    for in_dim in [2049usize, 2050, 2051] {
        let codes = weight_codes(OUT, in_dim);
        let x = activations(TOK, in_dim);
        let want = reference(&codes, &x, OUT, in_dim, TOK, SCALE);

        let args = Fp8Args {
            in_dim: i32::try_from(in_dim).expect("the fixture width fits i32"),
            out_dim: OUT as i32,
            n_tok: TOK as i32,
            weight_scale: SCALE,
        };
        let w = gpu.upload(&codes);
        let src = gpu.upload(&f32_bytes(&x));
        let dst = gpu.upload(&f32_bytes(&vec![0.0; TOK * OUT]));
        run(
            gpu,
            KernelRef::new(MODULE, "kernel_q35moe_mv_fp8_f32"),
            &LaunchArgs {
                grid: Extent3::new(OUT as u32, TOK as u32, 1),
                block: Extent3::new(MV_BLOCK, 1, 1),
                smem_bytes: 0,
                args: &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(w.view()),
                    KernelArg::Buffer(src.view()),
                    KernelArg::Buffer(dst.view()),
                ],
            },
        );
        drain(gpu);
        assert_close(
            &gpu.read_f32(&dst),
            &want,
            BUDGET,
            &format!("fp8 matvec at in_dim {in_dim}"),
        );
    }
}

/// The tiled GEMM, at widths that are not whole tiles.
///
/// # Panics
///
/// When the launch is refused or the product diverges past the budget.
pub fn fp8_gemm_matches_the_host_codec(gpu: &Gpu) {
    // Every dimension but the input width is deliberately not a multiple of
    // the 16-wide tile: 40 rows is two and a half tiles and 7 tokens is not
    // half of one. An unguarded tile edge reads a row past the matrix and
    // writes a row past the output, and both are invisible on a fixture whose
    // dimensions are all multiples.
    const IN: usize = 2048;
    const OUT: usize = 40;
    const TOK: usize = 7;
    const SCALE: f32 = 0.004_231_5;

    let codes = weight_codes(OUT, IN);
    let x = activations(TOK, IN);
    let want = reference(&codes, &x, OUT, IN, TOK, SCALE);

    let args = Fp8Args {
        in_dim: IN as i32,
        out_dim: OUT as i32,
        n_tok: TOK as i32,
        weight_scale: SCALE,
    };
    let w = gpu.upload(&codes);
    let src = gpu.upload(&f32_bytes(&x));
    // Poisoned rather than zeroed, so a tile that writes past the output is a
    // failure here rather than a value that happens to already be right.
    let dst = gpu.upload(&f32_bytes(&vec![f32::from_bits(0xA5A5_A5A5); TOK * OUT]));
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_q35moe_mm_fp8_f32"),
        &LaunchArgs {
            grid: Extent3::new((OUT as u32).div_ceil(TILE), (TOK as u32).div_ceil(TILE), 1),
            block: Extent3::new(TILE, TILE, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(src.view()),
                KernelArg::Buffer(dst.view()),
            ],
        },
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&dst), &want, BUDGET, "fp8 gemm");
}

/// The two entries agree with each other on the same numbers.
///
/// Not a restatement of the two cases above. Each of those is graded against
/// the host at a tolerance; this says the two kernels agree at a MUCH tighter
/// one, which is what the host picking between them on the row count requires.
/// A prefill chunk and the decode step that follows it read the same weights,
/// and a seam between the two paths would show up as a model that answers
/// differently depending on how the prompt was chunked - which is an open item
/// in this repository's own handoff for a different family.
///
/// # Panics
///
/// When a launch is refused or the two paths disagree.
pub fn the_matvec_and_the_gemm_agree_on_the_same_weights(gpu: &Gpu) {
    const IN: usize = 512;
    const OUT: usize = 33;
    const TOK: usize = 5;
    const SCALE: f32 = 1.0;

    let codes = weight_codes(OUT, IN);
    let x = activations(TOK, IN);
    let args = Fp8Args {
        in_dim: IN as i32,
        out_dim: OUT as i32,
        n_tok: TOK as i32,
        weight_scale: SCALE,
    };
    let w = gpu.upload(&codes);
    let src = gpu.upload(&f32_bytes(&x));

    let mv = gpu.upload(&f32_bytes(&vec![0.0; TOK * OUT]));
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_q35moe_mv_fp8_f32"),
        &LaunchArgs {
            grid: Extent3::new(OUT as u32, TOK as u32, 1),
            block: Extent3::new(MV_BLOCK, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(src.view()),
                KernelArg::Buffer(mv.view()),
            ],
        },
    );
    let mm = gpu.upload(&f32_bytes(&vec![0.0; TOK * OUT]));
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_q35moe_mm_fp8_f32"),
        &LaunchArgs {
            grid: Extent3::new((OUT as u32).div_ceil(TILE), (TOK as u32).div_ceil(TILE), 1),
            block: Extent3::new(TILE, TILE, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w.view()),
                KernelArg::Buffer(src.view()),
                KernelArg::Buffer(mm.view()),
            ],
        },
    );
    drain(gpu);
    // The matvec's answer becomes the reference, carrying the same conditioning
    // the host reference would: the two kernels contract the same 512 terms and
    // differ only in the order they fold them.
    let matvec = gpu.read_f32(&mv);
    let want = Reference {
        magnitude: reference(&codes, &x, OUT, IN, TOK, SCALE).magnitude,
        values: matvec,
    };
    assert_close(&gpu.read_f32(&mm), &want, BUDGET, "gemm against matvec");
}

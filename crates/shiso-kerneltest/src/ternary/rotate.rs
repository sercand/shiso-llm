//! Cases for `kernel_rotate_wht_f32`, against a host transform written the
//! slow way: a dense Sylvester matrix product per block.
//!
//! The host side never shares a line with the kernel: the device runs a
//! butterfly, the oracle evaluates `H[i][j] = (-1)^popcount(i & j)` directly,
//! so an error in the butterfly's pairing cannot cancel against the same
//! error here.

use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

use super::fixtures::{Lcg, rel_err};
use crate::harness::{Gpu, drain, f32_bytes, run};

/// The one entry this module launches.
pub const ENTRY: &str = "kernel_rotate_wht_f32";

/// Mirrors `ternary_args_rotate`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RotateArgs {
    n_tok: u32,
    width: u32,
    block: u32,
    inverse: u32,
    nk: u32,
    rep: u32,
    hd: u32,
    pad: u32,
}

/// Threads one threadgroup runs.
const THREADS: u32 = 256;

/// Relative tolerance: a 1024-point transform sums 1024 products, so its
/// rounding is a few ulps of the row's largest value.
const TOL: f32 = 1e-5;

/// Value heads a permuted case reads as grouped: `(head_dim, key_heads, rep)`.
#[derive(Debug, Clone, Copy)]
struct Heads {
    hd: usize,
    nk: usize,
    rep: usize,
}

/// The host transform of one row of `width`.
fn host_rotate(
    row: &[f32],
    signs: &[f32],
    block: usize,
    inverse: bool,
    heads: Option<Heads>,
) -> Vec<f32> {
    let width = row.len();
    let permuted: Vec<f32> = (0..width)
        .map(|e| match heads {
            Some(Heads { hd, nk, rep }) => {
                let head = e / hd;
                let (k, m) = (head / rep, head % rep);
                row[(m * nk + k) * hd + e % hd]
            }
            None => row[e],
        })
        .collect();
    let scale = 1.0 / f64::from(u32::try_from(block).expect("block size")).sqrt();
    let mut out = vec![0.0_f32; width];
    for b in (0..width).step_by(block) {
        for i in 0..block {
            let mut acc = 0.0_f64;
            for j in 0..block {
                let x = f64::from(permuted[b + j])
                    * if inverse {
                        1.0
                    } else {
                        f64::from(signs[b + j])
                    };
                acc += if (i & j).count_ones() % 2 == 0 { x } else { -x };
            }
            let y = acc * scale;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the oracle rounds once, at the end"
            )]
            let y = y as f32;
            out[b + i] = if inverse { y * signs[b + i] } else { y };
        }
    }
    out
}

/// One case's shape.
#[derive(Debug, Clone, Copy)]
struct Case {
    width: usize,
    block: usize,
    n_tok: usize,
    inverse: bool,
    heads: Option<Heads>,
    in_place: bool,
}

/// Runs one case and checks it against the host transform.
fn check(gpu: &Gpu, case: Case, seed: u64) {
    let mut rng = Lcg::new(seed);
    let x = rng.floats(case.width * case.n_tok);
    let signs = rng.signs(case.width);
    let want: Vec<f32> = x
        .chunks_exact(case.width)
        .flat_map(|row| host_rotate(row, &signs, case.block, case.inverse, case.heads))
        .collect();

    let src = gpu.upload(&f32_bytes(&x));
    let sign_buf = gpu.upload(&f32_bytes(&signs));
    let out = gpu.upload(&f32_bytes(&vec![0.0; x.len()]));
    let dst = if case.in_place { &src } else { &out };
    let heads = case.heads.unwrap_or(Heads {
        hd: 1,
        nk: 1,
        rep: 1,
    });
    let as_u32 = |v: usize| u32::try_from(v).expect("fixture dimension");
    let args = RotateArgs {
        n_tok: as_u32(case.n_tok),
        width: as_u32(case.width),
        block: as_u32(case.block),
        inverse: u32::from(case.inverse),
        nk: as_u32(heads.nk),
        rep: as_u32(heads.rep),
        hd: as_u32(heads.hd),
        pad: 0,
    };
    run(
        gpu,
        KernelRef::new("ternary", ENTRY),
        &LaunchArgs {
            grid: Extent3::new(as_u32(case.width / case.block), as_u32(case.n_tok), 1),
            block: Extent3::linear(THREADS),
            smem_bytes: as_u32(case.block * 4),
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(src.view()),
                KernelArg::Buffer(sign_buf.view()),
                KernelArg::Buffer(dst.view()),
            ],
        },
    );
    drain(gpu);
    let got = gpu.read_f32(dst);
    let err = rel_err(&got, &want);
    assert!(err <= TOL, "{case:?}: relative error {err} above {TOL}");
}

/// Forward over one block, several blocks and the widths this family uses.
pub fn forward_matches_the_dense_transform(gpu: &Gpu) {
    for (seed, (width, n_tok)) in [(1024, 1), (5120, 3), (17408, 2), (512, 4)]
        .into_iter()
        .enumerate()
    {
        let block = if width == 512 { 512 } else { 1024 };
        let case = Case {
            width,
            block,
            n_tok,
            inverse: false,
            heads: None,
            in_place: false,
        };
        check(gpu, case, 11 + seed as u64);
    }
}

/// Inverse, out of place and in place.
pub fn inverse_matches_the_dense_transform(gpu: &Gpu) {
    for in_place in [false, true] {
        let case = Case {
            width: 5120,
            block: 1024,
            n_tok: 2,
            inverse: true,
            heads: None,
            in_place,
        };
        check(gpu, case, 21);
    }
}

/// The tiled-to-grouped head permutation a grouped `ssm_out` reads through.
pub fn permuted_heads_read_tiled_as_grouped(gpu: &Gpu) {
    let heads = Some(Heads {
        hd: 128,
        nk: 16,
        rep: 3,
    });
    let case = Case {
        width: 6144,
        block: 1024,
        n_tok: 3,
        inverse: false,
        heads,
        in_place: false,
    };
    check(gpu, case, 31);
}

/// Forward then inverse is the identity, on the device alone.
pub fn forward_then_inverse_is_the_identity(gpu: &Gpu) {
    let mut rng = Lcg::new(41);
    let (width, block) = (5120_u32, 1024_u32);
    let x = rng.floats(width as usize * 2);
    let signs = gpu.upload(&f32_bytes(&rng.signs(width as usize)));
    let rows = gpu.upload(&f32_bytes(&x));
    for inverse in [0, 1] {
        let args = RotateArgs {
            n_tok: 2,
            width,
            block,
            inverse,
            nk: 1,
            rep: 1,
            hd: 1,
            pad: 0,
        };
        run(
            gpu,
            KernelRef::new("ternary", ENTRY),
            &LaunchArgs {
                grid: Extent3::new(width / block, 2, 1),
                block: Extent3::linear(THREADS),
                smem_bytes: block * 4,
                args: &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(rows.view()),
                    KernelArg::Buffer(signs.view()),
                    KernelArg::Buffer(rows.view()),
                ],
            },
        );
    }
    drain(gpu);
    let err = rel_err(&gpu.read_f32(&rows), &x);
    assert!(err <= TOL, "round trip error {err}");
}

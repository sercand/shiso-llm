//! The twelve-bit repack of half-precision weights, expanded back: graded
//! against a host decode of the same planes.
//!
//! `q4e_p12_expand_kernel` restores a packed weight to the halves (or the
//! floats they are) the checkpoint held, for the prompt tier's multiply. It
//! has a path that decodes eight columns a thread with wide loads and stores
//! wherever the contraction is a multiple of eight, and a column-at-a-time path
//! otherwise. Both must give exactly the tensor the planes describe, so every
//! comparison here is bitwise.
//!
//! The layout is transcribed from `shiso_engine::sparse::pack12`, which this
//! crate may not depend on: mantissa bytes `[out][in]`, exponent nibbles
//! `[out][in/2]`, one base exponent a row, then - each region aligned to 16
//! bytes - the exception row pointers, columns and halves.
#![expect(
    clippy::cast_possible_truncation,
    reason = "kernel arguments are u32 while the fixtures index in usize; every size here is a \
              small model dimension"
)]

use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, run};

/// The module, mirroring `kernels::MOD_MATVEC`.
const MODULE: &str = "q4e_matvec";
/// Threads a row, mirroring the engine's launch.
const THREADS: u32 = 256;

/// Rounds `at` up to the 16-byte boundary the device layout uses.
const fn align16(at: usize) -> usize {
    (at + 15) & !15
}

/// A packed weight and the halves it decodes to.
struct Packed {
    bytes: Vec<u8>,
    halves: Vec<u16>,
}

/// A deterministic `out x in` packed weight: every code from zero (a zero
/// weight) to fifteen, both signs, bases across the half range, and a few
/// exceptions a row that the decode writes over the planes.
fn packed(in_dim: usize, out_dim: usize, seed: u32) -> Packed {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let cells = in_dim * out_dim;
    let mut mant = vec![0u8; cells];
    let mut expo = vec![0u8; cells / 2];
    // Non-negative bases: the device reads them as i8, and 0..=15 is the same
    // byte either way.
    let mut base = vec![0u8; out_dim];
    let mut halves = vec![0u16; cells];
    let mut ptr = vec![0u32; out_dim + 1];
    let (mut col, mut val) = (Vec::new(), Vec::new());
    for row in 0..out_dim {
        // Codes 1..=15 land on exponents base+1..=base+15, which must stay a
        // finite half's: 1..=30.
        base[row] = (next() % 16) as u8;
        for c in 0..in_dim {
            let at = row * in_dim + c;
            let byte = (next() & 0xff) as u8;
            let code = (next() % 16) as u8;
            mant[at] = byte;
            expo[at / 2] |= code << ((at & 1) * 4);
            halves[at] = if code == 0 {
                0
            } else {
                (u16::from(byte & 0x80) << 8)
                    | ((u16::from(base[row]) + u16::from(code)) << 10)
                    | (u16::from(byte & 0x7f) << 3)
            };
        }
        for _ in 0..3 {
            let c = next() as usize % in_dim;
            let h = (next() & 0x7bff) as u16;
            col.push(c as u32);
            val.push(h);
            halves[row * in_dim + c] = h;
        }
        ptr[row + 1] = col.len() as u32;
    }
    let mut bytes = mant;
    bytes.extend_from_slice(&expo);
    bytes.extend_from_slice(&base);
    bytes.resize(align16(bytes.len()), 0);
    bytes.extend(ptr.iter().flat_map(|p| p.to_le_bytes()));
    bytes.resize(align16(bytes.len()), 0);
    bytes.extend(col.iter().flat_map(|c| c.to_le_bytes()));
    bytes.extend(val.iter().flat_map(|v| v.to_le_bytes()));
    Packed { bytes, halves }
}

/// Expands `weight` on the device into `out_dim * in_dim` elements of
/// `width` bytes each, through the instance `entry`.
fn expand(
    gpu: &Gpu,
    entry: &'static str,
    weight: &Packed,
    in_dim: u32,
    out_dim: u32,
    width: usize,
) -> Vec<u8> {
    let w = gpu.upload(&weight.bytes);
    // Filled with a pattern no decode writes, so a column the kernel skipped
    // shows up as a mismatch rather than as a zero that happens to agree.
    let dst = gpu.upload(&vec![0xA5u8; in_dim as usize * out_dim as usize * width]);
    run(
        gpu,
        KernelRef::new(MODULE, entry),
        &LaunchArgs {
            grid: Extent3::new(out_dim, 1, 1),
            block: Extent3::new(THREADS, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(dst.view()),
                KernelArg::Buffer(w.view()),
                KernelArg::U32(in_dim),
                KernelArg::U32(out_dim),
            ],
        },
    );
    drain(gpu);
    gpu.read(&dst)
}

/// Asserts the device's halves are the oracle's, naming the first that is not.
fn same_halves(got: &[u8], want: &[u16], what: &str) {
    for (i, (bytes, w)) in got.chunks_exact(2).zip(want).enumerate() {
        let g = u16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(g, *w, "{what}: element {i}");
    }
}

/// Asserts the device's floats are the oracle's halves widened, bitwise.
fn same_floats(got: &[u8], want: &[u16], what: &str) {
    for (i, (bytes, w)) in got.chunks_exact(4).zip(want).enumerate() {
        let g = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let w = half::f16::from_bits(*w).to_f32();
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}");
    }
}

/// `q4e_p12_expand_kernel` restores the halves the planes and exceptions
/// describe, to halves and to floats, on the eight-column path (a contraction
/// of 2560, the hidden width) and the column-at-a-time path (one of 2570).
///
/// # Panics
///
/// When an element differs from the host decode.
pub fn the_expansion_restores_the_packed_halves(gpu: &Gpu) {
    for (in_dim, out_dim) in [(2_560_u32, 48_u32), (2_570, 33)] {
        let weight = packed(in_dim as usize, out_dim as usize, in_dim ^ out_dim);
        let what = format!("{out_dim}x{in_dim}");
        let halves = expand(
            gpu,
            "_Z21q4e_p12_expand_kernelI6__halfEvPT_PKhjj",
            &weight,
            in_dim,
            out_dim,
            2,
        );
        same_halves(&halves, &weight.halves, &format!("{what} halves"));
        let floats = expand(
            gpu,
            "_Z21q4e_p12_expand_kernelIfEvPT_PKhjj",
            &weight,
            in_dim,
            out_dim,
            4,
        );
        same_floats(&floats, &weight.halves, &format!("{what} floats"));
    }
}

//! Cases for the gated short convolution (`dense_short_conv.metal`).
//!
//! The kernel is held to a host rule within a float tolerance, run as two
//! chunks so the session window it rewrites is graded too: the second chunk
//! has to start exactly where the first left off, and the window has to hold
//! the gated product `b * x`, not the raw input.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "fixture dimensions are small positive integers"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 1] = ["kernel_dense_short_conv_f32"];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Args {
    n_tok: u32,
    channels: u32,
    taps: u32,
    pad: u32,
}

fn signal(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

fn close(have: f32, want: f32, tol: f32) -> bool {
    (have - want).abs() <= tol * (1.0 + want.abs())
}

/// The convolution over two chunks against the host rule, window included.
pub fn short_conv_carries_its_window_across_chunks(gpu: &Gpu) {
    // Not a multiple of the 256-thread block, so the bounds check runs.
    let (channels, taps) = (300usize, 3usize);
    let chunks = [4usize, 1, 3];
    let total: usize = chunks.iter().sum();
    let stride = 3 * channels;
    let proj = signal(total * stride, 31);
    let weight = signal(channels * taps, 32);
    let window0 = signal((taps - 1) * channels, 33);
    // Host: the gated inputs of the whole sequence, the starting window first.
    let mut seq = window0.clone();
    for t in 0..total {
        let row = &proj[t * stride..(t + 1) * stride];
        seq.extend((0..channels).map(|c| row[c] * row[2 * channels + c]));
    }
    let window_buf = gpu.upload(&f32_bytes(&window0));
    let w_buf = gpu.upload(&f32_bytes(&weight));
    let mut first = 0usize;
    for &n in &chunks {
        let p_buf = gpu.upload(&f32_bytes(&proj[first * stride..(first + n) * stride]));
        let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n * channels]));
        let args = Args {
            n_tok: n as u32,
            channels: channels as u32,
            taps: taps as u32,
            pad: 0,
        };
        run(
            gpu,
            KernelRef::new("dense", "kernel_dense_short_conv_f32"),
            &LaunchArgs {
                grid: Extent3::linear((channels as u32).div_ceil(256)),
                block: Extent3::linear(256),
                smem_bytes: 0,
                args: &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(p_buf.view()),
                    KernelArg::Buffer(window_buf.view()),
                    KernelArg::Buffer(w_buf.view()),
                    KernelArg::Buffer(out_buf.view()),
                ],
            },
        );
        drain(gpu);
        let got = gpu.read_f32(&out_buf);
        for t in 0..n {
            let row = &proj[(first + t) * stride..(first + t + 1) * stride];
            for c in 0..channels {
                let mut acc = 0.0f32;
                for j in 0..taps {
                    acc += weight[c * taps + j] * seq[(first + t + j) * channels + c];
                }
                let want = row[channels + c] * acc;
                let have = got[t * channels + c];
                assert!(
                    close(have, want, 1e-5),
                    "row {} channel {c}: {have} against {want}",
                    first + t
                );
            }
        }
        first += n;
    }
    let window = gpu.read_f32(&window_buf);
    for j in 0..taps - 1 {
        for c in 0..channels {
            let want = seq[(total + j) * channels + c];
            assert_eq!(
                window[j * channels + c].to_bits(),
                want.to_bits(),
                "window {j} channel {c}"
            );
        }
    }
}

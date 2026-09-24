//! A column-invariant multiply gives one column the same bits at every column
//! count.
//!
//! A projection's columns are a pass's rows, so this is the property that keeps
//! a prompt's answer independent of where its chunks were cut. The ordinary
//! entry point does not have it: it chooses a split of the contraction from the
//! whole shape, column count included, and on a GB10 every output narrower than
//! about a thousand rows summed a column differently at 1 to 4096 columns than
//! in an 8192-column call. The case keeps that contrast visible by asserting it
//! as well - a library update that made the ordinary path invariant too would
//! fail here, and should be read, not silenced.
//!
//! The shapes are this port's own projections: the recurrent gates (48 out),
//! the stream mixes (320 and 4 out of 10240, 10240 out of 320), the shared
//! expert, the attention key and value heads and the wide ones.
//!
//! GPU case: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --release --test
//! blas_pinned -- --ignored --nocapture`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::backend::{GemmDType, GemmDesc, GemmOp, GemmReduction};
use shiso_core::{Backend, DeviceBuffer, alloc};
use shiso_cuda::{CudaBackend, CudaOptions};

/// The widest call, whose leading columns every narrower call is held to.
const FULL: u32 = 8_192;

/// Column counts the narrower calls take: single rows, runs off every tile
/// width, and whole chunks.
const COLUMNS: [u32; 10] = [1, 7, 16, 33, 64, 128, 320, 576, 1_024, 4_096];

/// `(m, k)` of the projections the case multiplies.
const SHAPES: [(u32, u32); 12] = [
    (48, 2_560),
    (320, 10_240),
    (512, 2_560),
    (640, 2_560),
    (2_560, 640),
    (2_560, 6_144),
    (6_144, 2_560),
    (10_240, 2_560),
    (12_288, 2_560),
    (4, 10_240),
    (10_240, 320),
    (512, 128),
];

/// `n` small pseudo-random values, as halves or floats.
fn fill(n: u32, seed: u32, half: bool) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::new();
    for _ in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let high = u16::try_from(state >> 16).expect("sixteen bits");
        let value = (f32::from(high) / 65_536.0 - 0.5) * 0.2;
        if half {
            out.extend_from_slice(&half::f16::from_f32(value).to_le_bytes());
        } else {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// One shape's operands and the widest product's columns.
struct Case {
    weight: DeviceBuffer,
    input: DeviceBuffer,
    output: DeviceBuffer,
    outputs: u32,
    depth: u32,
    dtype: GemmDType,
}

impl Case {
    fn new(backend: &Arc<dyn Backend>, outputs: u32, depth: u32, dtype: GemmDType) -> Self {
        let half = matches!(dtype, GemmDType::F16 | GemmDType::F16ToF16);
        let width = if half { 2 } else { 4 };
        let weight = alloc(backend, u64::from(outputs * depth) * width).expect("the weight fits");
        let input =
            alloc(backend, u64::from(depth) * u64::from(FULL) * width).expect("the input fits");
        let output =
            alloc(backend, u64::from(outputs) * u64::from(FULL) * 4).expect("the output fits");
        backend
            .copy_h2d(weight.view(), &fill(outputs * depth, 7, half))
            .expect("the weight uploads");
        backend
            .copy_h2d(input.view(), &fill(depth * FULL, 11, half))
            .expect("the input uploads");
        Self {
            weight,
            input,
            output,
            outputs,
            depth,
            dtype,
        }
    }

    /// The product's first `n` columns, from a call of exactly `n` columns.
    fn columns(&self, backend: &Arc<dyn Backend>, n: u32, reduction: GemmReduction) -> Vec<u8> {
        backend
            .as_blas()
            .expect("the CUDA backend has a matrix tier")
            .gemm(&GemmDesc {
                op_a: GemmOp::Transposed,
                op_b: GemmOp::AsStored,
                m: self.outputs,
                n,
                k: self.depth,
                lda: self.depth,
                ldb: self.depth,
                ldc: self.outputs,
                a: self.weight.view(),
                b: self.input.view(),
                c: self.output.view(),
                alpha: 1.0,
                beta: 0.0,
                dtype: self.dtype,
                reduction,
            })
            .expect("the multiply runs");
        let out_width = if self.dtype == GemmDType::F16ToF16 {
            2
        } else {
            4
        };
        let bytes = u64::from(self.outputs) * u64::from(n) * out_width;
        let mut out = vec![0_u8; usize::try_from(bytes).expect("a host index")];
        backend
            .copy_d2h(
                self.output.view().slice(0, bytes).expect("in range"),
                &mut out,
            )
            .expect("the product reads back");
        out
    }

    /// The column counts at which a call of that many columns disagrees with
    /// the widest call's leading columns.
    fn moved(&self, backend: &Arc<dyn Backend>, reduction: GemmReduction) -> Vec<u32> {
        let widest = self.columns(backend, FULL, reduction);
        COLUMNS
            .into_iter()
            .filter(|&n| {
                let got = self.columns(backend, n, reduction);
                got[..] != widest[..got.len()]
            })
            .collect()
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_column_invariant_multiply_sums_a_column_one_way_at_every_width() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let mut ordinary_moved = 0;
    for dtype in [GemmDType::F16, GemmDType::F16ToF16, GemmDType::F32Strict] {
        for (m, k) in SHAPES {
            let case = Case::new(&backend, m, k, dtype);
            let moved = case.moved(&backend, GemmReduction::ColumnInvariant);
            assert!(
                moved.is_empty(),
                "{dtype:?} {m}x{k}: a column moved at {moved:?} columns"
            );
            if !case.moved(&backend, GemmReduction::Fastest).is_empty() {
                ordinary_moved += 1;
            }
        }
    }
    assert!(
        ordinary_moved > 0,
        "the ordinary entry point is column-invariant on every shape here now; \
         the pinned tier may no longer be needed - measure before removing it"
    );
}

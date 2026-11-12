//! The processor's share of a projection the GPU is also computing.
//!
//! Every buffer this backend allocates is shared storage: the same bytes the
//! GPU reads are mapped into the process. So while the GPU works through one
//! range of a large projection's columns, the system matrix library can work
//! through the rest in place, with no copy in either direction. Measured on
//! an M4 (10-core GPU, 4P+6E) over 30 s: the GPU alone sustains 3.39 TFLOPS
//! on a 4096-cube single-precision product, the two together 4.49 (GPU 3.16,
//! processor 1.32) - the processor's share costs the GPU about 7% of its own
//! rate to memory contention and returns four times that.
//!
//! The matrix library has no byte operand, so the codes are widened one piece
//! of the range at a time into a scratch that stays cache-sized; the weight is
//! never dequantized as a whole. The row scale, the bias and the gate are
//! applied to the product exactly as the GPU's epilogue applies them.

use std::ffi::{c_int, c_long, c_ulong};
use std::slice::from_raw_parts_mut;

use parking_lot::Mutex;
use shiso_core::{CoreError, CoreResult, DeviceView, HostGemm, HostGemmDesc};

use crate::device::{MetalBackend, MetalError};

/// `CblasRowMajor`.
const ROW_MAJOR: c_int = 101;
/// `CblasNoTrans`.
const NO_TRANS: c_int = 111;
/// `CblasTrans`.
const TRANS: c_int = 112;

/// Widened codes held at once, in elements: 16 MiB of `f32`, which keeps one
/// piece's weight inside the performance cores' shared cache while leaving
/// each product large enough that the library's threads stay busy.
const PIECE_ELEMENTS: usize = 4 << 20;

/// Threads the two passes around each product run on. Both are streaming
/// passes over tens of megabytes the matrix library's own threads sit out,
/// and one core's memory rate held them to a fifth of the host's time; four
/// is the performance-core count of the smallest machine this runs on, and
/// the passes are bandwidth-bound past that.
const HELPERS: usize = 4;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    /// `C = alpha * op(A) * op(B) + beta * C`, single precision.
    fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
    /// Signed bytes widened to single precision.
    fn vDSP_vflt8(a: *const i8, stride_a: c_long, c: *mut f32, stride_c: c_long, n: c_ulong);
}

/// The two scratch matrices one piece needs, kept between calls so a step's
/// few hundred projections do not each fault in fresh pages.
#[derive(Debug, Default)]
pub(crate) struct HostScratch {
    /// One piece of the weight, widened: `[piece, k]`.
    weight: Vec<f32>,
    /// That piece's unscaled product: `[m, piece]`.
    product: Vec<f32>,
}

/// The host-side scratch, behind its own lock: a multiply holds it for tens
/// of milliseconds and must not hold the device state with it.
pub(crate) type HostScratchLock = Mutex<HostScratch>;

/// A `usize` product of the shape, refused when it does not fit.
fn elements(a: usize, b: usize) -> CoreResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| MetalError::backend("host_gemm", "the shape overflows the address space"))
}

/// A BLAS dimension, refused when the library's `int` cannot hold it.
fn dim(value: usize) -> CoreResult<c_int> {
    c_int::try_from(value)
        .map_err(|_| MetalError::backend("host_gemm", "a dimension exceeds the matrix library's"))
}

impl MetalBackend {
    /// The host address of `view`, checked to hold `bytes` and to be aligned
    /// for `align`-byte elements.
    fn host_span(&self, view: DeviceView<'_>, bytes: usize, align: usize) -> CoreResult<usize> {
        if view.len() < bytes as u64 {
            return Err(CoreError::OutOfBounds {
                offset: view.offset(),
                len: bytes as u64,
                buffer_bytes: view.len(),
            });
        }
        let address = self.host_address(view)?;
        if !address.is_multiple_of(align) {
            return Err(MetalError::backend(
                "host_gemm",
                "an operand is not aligned for its element type",
            ));
        }
        Ok(address)
    }

    /// `count` floats starting at `address`, copied out.
    fn host_floats(address: usize, count: usize) -> Vec<f32> {
        // SAFETY: `host_span` proved `count * 4` bytes at `address` lie inside
        // a live, 4-aligned allocation the caller's view keeps alive, and the
        // caller of `gemm_i8_rows` has ordered the device past its writes.
        unsafe { std::slice::from_raw_parts(address as *const f32, count) }.to_vec()
    }
}

impl HostGemm for MetalBackend {
    fn gemm_i8_rows(&self, desc: &HostGemmDesc<'_>) -> CoreResult<()> {
        let (m, n, k) = (desc.m as usize, desc.n as usize, desc.k as usize);
        let (start, end) = (desc.col_start as usize, desc.col_end as usize);
        if start > end || end > n {
            return Err(CoreError::OutOfBounds {
                offset: start as u64,
                len: end.saturating_sub(start) as u64,
                buffer_bytes: n as u64,
            });
        }
        if start == end || m == 0 || k == 0 {
            return Ok(());
        }
        let vector = elements(n, 4)?;
        let input = self.host_span(desc.input, elements(elements(m, k)?, 4)?, 4)?;
        let weight = self.host_span(desc.weight, elements(n, k)?, 1)?;
        let out = self.host_span(desc.out, elements(elements(m, n)?, 4)?, 4)?;
        let scale = Self::host_floats(self.host_span(desc.row_scale, vector, 4)?, n);
        let bias = match desc.bias {
            Some(view) => Some(Self::host_floats(self.host_span(view, vector, 4)?, n)),
            None => None,
        };
        let gate = match desc.gate {
            Some(view) => Some(Self::host_floats(self.host_span(view, vector, 4)?, n)),
            None => None,
        };

        let piece = (PIECE_ELEMENTS / k).clamp(1, end - start);
        let mut scratch = self.host_scratch().lock();
        let HostScratch {
            weight: wide,
            product,
        } = &mut *scratch;
        wide.resize(elements(piece, k)?, 0.0);
        product.resize(elements(m, piece)?, 0.0);
        let (blas_m, blas_k) = (dim(m)?, dim(k)?);

        let mut first = start;
        while first < end {
            let cols = piece.min(end - first);
            widen(weight + first * k, &mut wide[..cols * k], k);
            // SAFETY: `input` addresses `m * k` live, 4-aligned floats the
            // device is not writing (the caller synced past its writes and
            // submits nothing that touches it); `wide` is `[cols, k]` and
            // `product` at least `[m, cols]`, both owned here.
            unsafe {
                cblas_sgemm(
                    ROW_MAJOR,
                    NO_TRANS,
                    TRANS,
                    blas_m,
                    dim(cols)?,
                    blas_k,
                    1.0,
                    input as *const f32,
                    blas_k,
                    wide.as_ptr(),
                    blas_k,
                    0.0,
                    product.as_mut_ptr(),
                    dim(cols)?,
                );
            }
            let finish = Finish {
                scale: &scale[first..first + cols],
                bias: bias.as_ref().map(|b| &b[first..first + cols]),
                gate: gate.as_ref().map(|g| &g[first..first + cols]),
            };
            let band = m.div_ceil(HELPERS).max(1);
            std::thread::scope(|scope| {
                for (index, rows) in product[..m * cols].chunks(band * cols).enumerate() {
                    let finish = &finish;
                    scope.spawn(move || {
                        for (r, row) in rows.chunks_exact(cols).enumerate() {
                            let at = out + ((index * band + r) * n + first) * 4;
                            // SAFETY: `out` addresses `m * n` live, 4-aligned
                            // floats, and this run is columns `first..first +
                            // cols` of one row, inside them. Each row belongs
                            // to one band and so to one thread; the device may
                            // be writing `out` meanwhile, but only columns
                            // outside this call's range. Nothing else reads
                            // or writes these bytes while the borrow lives.
                            let dst = unsafe { from_raw_parts_mut(at as *mut f32, cols) };
                            finish.row(dst, row);
                        }
                    });
                }
            });
            first += cols;
        }
        Ok(())
    }
}

/// Widens the signed bytes at `source` into `wide`, a whole number of
/// `k`-long rows, on [`HELPERS`] threads.
fn widen(source: usize, wide: &mut [f32], k: usize) {
    let band = (wide.len() / k).div_ceil(HELPERS).max(1) * k;
    std::thread::scope(|scope| {
        for (index, rows) in wide.chunks_mut(band).enumerate() {
            scope.spawn(move || {
                // SAFETY: the caller proved `wide.len()` live bytes at
                // `source`, and this band's bytes are the ones at the band's
                // own offset; `rows` is this thread's alone. Byte loads need
                // no alignment.
                unsafe {
                    vDSP_vflt8(
                        (source + index * band) as *const i8,
                        1,
                        rows.as_mut_ptr(),
                        1,
                        rows.len() as c_ulong,
                    );
                }
            });
        }
    });
}

/// What turns one piece's raw product into the projection's columns: the
/// GPU epilogue's own arithmetic, `v * scale + bias`, stored, or added
/// scaled by the gate.
struct Finish<'a> {
    scale: &'a [f32],
    bias: Option<&'a [f32]>,
    gate: Option<&'a [f32]>,
}

impl Finish<'_> {
    /// Finishes one row's run of columns from its raw `product`.
    fn row(&self, dst: &mut [f32], product: &[f32]) {
        epilogue(dst, product, self.scale, self.bias, self.gate);
    }
}

/// One row's run of columns, finished the way the GPU's epilogue finishes
/// it: `v * scale + bias`, stored, or added scaled by the gate.
fn epilogue(
    dst: &mut [f32],
    product: &[f32],
    scale: &[f32],
    bias: Option<&[f32]>,
    gate: Option<&[f32]>,
) {
    match (bias, gate) {
        (None, None) => {
            for ((d, &v), &s) in dst.iter_mut().zip(product).zip(scale) {
                *d = v * s;
            }
        }
        (Some(bias), None) => {
            for (((d, &v), &s), &b) in dst.iter_mut().zip(product).zip(scale).zip(bias) {
                *d = v * s + b;
            }
        }
        (None, Some(gate)) => {
            for (((d, &v), &s), &g) in dst.iter_mut().zip(product).zip(scale).zip(gate) {
                *d += g * (v * s);
            }
        }
        (Some(bias), Some(gate)) => {
            for ((((d, &v), &s), &b), &g) in
                dst.iter_mut().zip(product).zip(scale).zip(bias).zip(gate)
            {
                *d += g * (v * s + b);
            }
        }
    }
}

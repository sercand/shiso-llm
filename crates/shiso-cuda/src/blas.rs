//! Matrix multiplication, in two tiers.
//!
//! # Two tiers, and why the simple one is the default
//!
//! [`BlasTier::Classic`] is the vendor library's ordinary entry point. It is
//! the default, and that is a numerical decision rather than a conservative
//! one: the reference outputs this port is checked against were produced by
//! that entry point, and a different library path is entitled to reduce in a
//! different order. Two orders that are equally correct still differ in the low
//! bits, and the golden vectors pin the low bits.
//!
//! [`BlasTier::Lt`] is the heuristic path, which asks the library to choose an
//! algorithm for the exact problem shape and gives it a workspace to use. It is
//! opt-in per call, never a default, and it only earns its keep on shapes where
//! the ordinary path leaves the device idle.
//!
//! # Column-major, stated once
//!
//! The vendor library reads matrices column-major and there is no flag to
//! change that. A leading dimension is therefore a *column* stride, and the
//! stored extents of an operand are its shape before transposition — the
//! operation flags are what say how to read them. Every bound this module
//! checks follows from those two sentences, and they are checked here rather
//! than left to the device, because an under-sized operand does not fault: it
//! reads whatever is next in the allocation and returns numbers.

use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use shiso_core::{CoreError, CoreResult, DeviceView, GemmDType, GemmDesc, GemmOp};

use crate::CudaBackend;
use crate::device::BufferTable;
use crate::sys::nvidia;
use crate::sys::{
    CuBlas, CuBlasLt, CuStream, DevicePtr, GemmCall, GemmStride, MathType, Op as SysOp, host_error,
};

/// Bytes of scratch the heuristic tier is allowed to use.
///
/// The library picks an algorithm to fit whatever budget it is given, so this
/// is a ceiling rather than a reservation shape: too small a budget silently
/// narrows the choice of algorithms, which is exactly the thing the tier exists
/// to widen.
const LT_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;

/// Which operand a multiply reads transposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Op {
    /// Read as stored.
    N,
    /// Read transposed.
    T,
}

impl Op {
    /// The shared vocabulary's spelling of the same choice.
    const fn from_core(op: GemmOp) -> Self {
        match op {
            GemmOp::AsStored => Self::N,
            GemmOp::Transposed => Self::T,
        }
    }

    /// The internal spelling.
    const fn to_sys(self) -> SysOp {
        match self {
            Self::N => SysOp::N,
            Self::T => SysOp::T,
        }
    }
}

/// Which library path a multiply takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum BlasTier {
    /// The vendor library's ordinary entry point. The default, and the one the
    /// reference outputs were produced with.
    #[default]
    Classic,
    /// The heuristic entry point, which chooses an algorithm per problem shape
    /// and uses a workspace. Opt-in per call.
    Lt,
}

/// One operand: where it lives and how far apart its columns are.
///
/// The stride travels with the matrix it belongs to. Three `i64` strides in a
/// row as positional arguments is how a transposed pair of leading dimensions
/// compiles cleanly and returns wrong numbers that only a reference comparison
/// catches; a named field per matrix makes the same mistake a compile error.
#[derive(Debug, Clone, Copy)]
pub struct Operand<'a> {
    /// The matrix's bytes.
    pub view: DeviceView<'a>,
    /// Elements between the starts of consecutive columns.
    pub ld: i64,
}

/// The extra description a strided batch needs.
#[derive(Debug, Clone, Copy)]
pub struct GemmBatch {
    /// Number of multiplies in the batch.
    pub count: u32,
    /// Elements between consecutive left operands.
    pub stride_a: i64,
    /// Elements between consecutive right operands.
    pub stride_b: i64,
    /// Elements between consecutive accumulators.
    pub stride_c: i64,
}

/// One matrix multiply, as a single argument block.
///
/// Deliberately *not* `#[non_exhaustive]`: a caller outside this crate has to
/// be able to build one, and a non-exhaustive struct with no constructor cannot
/// be built at all. A new field here is a compile error at every call site,
/// which for a descriptor whose fields are all load-bearing is the right place
/// for it to surface.
#[derive(Debug, Clone, Copy)]
pub struct CudaGemm<'a> {
    /// How the left operand is read.
    pub op_a: Op,
    /// How the right operand is read.
    pub op_b: Op,
    /// Rows of the product.
    pub m: u32,
    /// Columns of the product.
    pub n: u32,
    /// Shared inner dimension.
    pub k: u32,
    /// Left operand.
    pub a: Operand<'a>,
    /// Right operand.
    pub b: Operand<'a>,
    /// Accumulator, read as well as written when `beta` is nonzero.
    pub c: Operand<'a>,
    /// Scale applied to the product.
    pub alpha: f32,
    /// Scale applied to the existing accumulator.
    pub beta: f32,
    /// Element type of all three matrices.
    pub dtype: GemmDType,
    /// `None` for one multiply; `Some` for a strided batch of them.
    pub batch: Option<GemmBatch>,
    /// Which library path to take.
    pub tier: BlasTier,
}

impl<'a> CudaGemm<'a> {
    /// The unbatched multiply the shared trait describes.
    ///
    /// The shared [`GemmDesc`] says how each operand is read but carries no
    /// batch and no tier choice, so this is what a caller reaching this backend
    /// through `as_blas()` gets. Anything richer goes through
    /// [`CudaBackend::gemm_ex`].
    fn from_core(desc: &GemmDesc<'a>) -> Self {
        Self {
            op_a: Op::from_core(desc.op_a),
            op_b: Op::from_core(desc.op_b),
            m: desc.m,
            n: desc.n,
            k: desc.k,
            a: Operand {
                view: desc.a,
                ld: i64::from(desc.lda),
            },
            b: Operand {
                view: desc.b,
                ld: i64::from(desc.ldb),
            },
            c: Operand {
                view: desc.c,
                ld: i64::from(desc.ldc),
            },
            alpha: desc.alpha,
            beta: desc.beta,
            dtype: desc.dtype,
            batch: None,
            tier: BlasTier::Classic,
        }
    }
}

/// Bytes one element of this type occupies.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for an element type added to the shared
/// vocabulary that this backend has no multiply path for. Guessing a width
/// would make every bound check below wrong by a factor.
fn element_bytes(dtype: GemmDType) -> CoreResult<u64> {
    match dtype {
        GemmDType::F32 | GemmDType::F32Strict => Ok(4),
        GemmDType::F16 | GemmDType::F16ToF16 | GemmDType::BF16 => Ok(2),
        _ => Err(unsupported_dtype()),
    }
}

/// The internal spelling of an element type.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for an element type this backend has no multiply
/// path for.
fn math_type(dtype: GemmDType) -> CoreResult<MathType> {
    match dtype {
        GemmDType::F32 | GemmDType::F32Strict => Ok(MathType::F32),
        GemmDType::F16 | GemmDType::F16ToF16 => Ok(MathType::F16),
        GemmDType::BF16 => Ok(MathType::Bf16),
        _ => Err(unsupported_dtype()),
    }
}

/// The refusal an unmapped element type gets.
const fn unsupported_dtype() -> CoreError {
    CoreError::Unsupported {
        what: "this matrix element type",
        backend: crate::sys::BACKEND,
    }
}

/// A dimension in the shape the library takes it.
fn dim(what: &'static str, value: u64) -> CoreResult<i32> {
    i32::try_from(value).map_err(|_| {
        host_error(
            "gemm",
            format!("{what} of {value} is larger than a multiply can be"),
        )
    })
}

/// The stored extents of one operand, before transposition.
///
/// Column-major: an untransposed left operand is `m` rows by `k` columns, and a
/// transposed one is stored `k` by `m`.
const fn stored_extents(op: Op, rows: u32, cols: u32) -> (u32, u32) {
    match op {
        Op::N => (rows, cols),
        Op::T => (cols, rows),
    }
}

/// Smallest number of elements a column-major matrix of this shape occupies.
///
/// `ld * (cols - 1) + rows`: the last column starts `ld * (cols - 1)` elements
/// in and is `rows` long. Using `ld * cols` instead would demand padding past
/// the last column that no caller has to supply.
fn min_elements(rows: u32, cols: u32, ld: i64) -> CoreResult<u64> {
    let ld =
        u64::try_from(ld).map_err(|_| host_error("gemm", "a leading dimension is negative"))?;
    if ld < u64::from(rows) {
        return Err(host_error(
            "gemm",
            format!("a leading dimension of {ld} is shorter than the {rows} rows it must span"),
        ));
    }
    let last = ld
        .checked_mul(u64::from(cols).saturating_sub(1))
        .ok_or_else(|| host_error("gemm", "a matrix's element count overflows"))?;
    last.checked_add(u64::from(rows))
        .ok_or_else(|| host_error("gemm", "a matrix's element count overflows"))
}

/// Proves one operand's view is long enough for the shape it claims.
fn check_operand(
    label: &'static str,
    operand: Operand<'_>,
    rows: u32,
    cols: u32,
    dtype: GemmDType,
    batch: Option<(u32, i64)>,
) -> CoreResult<()> {
    let mut elements = min_elements(rows, cols, operand.ld)?;
    if let Some((count, stride)) = batch {
        let stride = u64::try_from(stride)
            .map_err(|_| host_error("gemm", format!("the batch stride of {label} is negative")))?;
        let span = stride
            .checked_mul(u64::from(count).saturating_sub(1))
            .ok_or_else(|| host_error("gemm", "a batch's element count overflows"))?;
        elements = elements
            .checked_add(span)
            .ok_or_else(|| host_error("gemm", "a batch's element count overflows"))?;
    }
    let needed = elements
        .checked_mul(element_bytes(dtype)?)
        .ok_or_else(|| host_error("gemm", "a matrix's byte count overflows"))?;
    if operand.view.len() < needed {
        return Err(CoreError::OutOfBounds {
            offset: operand.view.offset(),
            len: needed,
            buffer_bytes: operand.view.len(),
        });
    }
    Ok(())
}

/// Proves a multiply describes something the device can actually run.
///
/// Every check here is one the device would not make: an operand shorter than
/// its declared shape does not fault, it reads the next thing in the allocation
/// and returns plausible numbers.
///
/// # Errors
///
/// [`CoreError::Backend`] for a dimension of zero, a negative or too-short
/// leading dimension, an empty batch, or an arithmetic overflow, and
/// [`CoreError::OutOfBounds`] for an operand whose view is shorter than its
/// declared shape.
pub(crate) fn validate(gemm: &CudaGemm<'_>) -> CoreResult<()> {
    for (label, value) in [("m", gemm.m), ("n", gemm.n), ("k", gemm.k)] {
        if value == 0 {
            return Err(host_error(
                "gemm",
                format!("{label} is zero, so the multiply would produce nothing"),
            ));
        }
    }
    if let Some(batch) = gemm.batch
        && batch.count == 0
    {
        return Err(host_error(
            "gemm",
            "a batch of zero multiplies produces nothing",
        ));
    }
    let batch_a = gemm.batch.map(|b| (b.count, b.stride_a));
    let batch_b = gemm.batch.map(|b| (b.count, b.stride_b));
    let batch_c = gemm.batch.map(|b| (b.count, b.stride_c));

    let (a_rows, a_cols) = stored_extents(gemm.op_a, gemm.m, gemm.k);
    let (b_rows, b_cols) = stored_extents(gemm.op_b, gemm.k, gemm.n);
    check_operand("a", gemm.a, a_rows, a_cols, gemm.dtype, batch_a)?;
    check_operand("b", gemm.b, b_rows, b_cols, gemm.dtype, batch_b)?;
    check_operand("c", gemm.c, gemm.m, gemm.n, gemm.dtype, batch_c)?;
    Ok(())
}

/// Turns a validated multiply into the addresses and integers the library takes.
fn resolve(
    table: &BufferTable,
    gemm: &CudaGemm<'_>,
    tf32: bool,
) -> CoreResult<(GemmCall, Option<GemmStride>)> {
    let call = GemmCall {
        op_a: gemm.op_a.to_sys(),
        op_b: gemm.op_b.to_sys(),
        m: dim("m", u64::from(gemm.m))?,
        n: dim("n", u64::from(gemm.n))?,
        k: dim("k", u64::from(gemm.k))?,
        a: table.resolve(gemm.a.view)?,
        lda: dim("lda", u64::try_from(gemm.a.ld).unwrap_or(u64::MAX))?,
        b: table.resolve(gemm.b.view)?,
        ldb: dim("ldb", u64::try_from(gemm.b.ld).unwrap_or(u64::MAX))?,
        c: table.resolve(gemm.c.view)?,
        ldc: dim("ldc", u64::try_from(gemm.c.ld).unwrap_or(u64::MAX))?,
        alpha: gemm.alpha,
        beta: gemm.beta,
        dtype: math_type(gemm.dtype)?,
        tf32: tf32 && gemm.dtype != GemmDType::F32Strict,
        out_half: gemm.dtype == GemmDType::F16ToF16,
    };
    let stride = gemm
        .batch
        .map(|batch| -> CoreResult<GemmStride> {
            Ok(GemmStride {
                count: dim("batch count", u64::from(batch.count))?,
                a: batch.stride_a,
                b: batch.stride_b,
                c: batch.stride_c,
            })
        })
        .transpose()?;
    Ok((call, stride))
}

/// The heuristic tier's handle and the scratch it was given.
#[derive(Debug)]
struct LtTier {
    handle: CuBlasLt,
    workspace: DevicePtr,
    workspace_bytes: u64,
}

impl Drop for LtTier {
    fn drop(&mut self) {
        // SAFETY: this value owns the only copy of both handles, and both are
        // released here and only here.
        if let Err(err) = unsafe { nvidia::mem_free(self.workspace) }
            && !err.is_teardown_noop()
        {
            tracing::error!(error = %err, "matmul workspace free failed");
        }
        // SAFETY: as above.
        if let Err(err) = unsafe { nvidia::blas_lt_destroy(self.handle) }
            && !err.is_teardown_noop()
        {
            tracing::error!(error = %err, "matmul library handle destroy failed");
        }
    }
}

/// The classic handle, the optional heuristic tier, and the accumulation mode.
#[derive(Debug)]
pub(crate) struct Blas {
    classic: CuBlas,
    /// Opened on the first call that asks for it, because the workspace it owns
    /// is sixty-four megabytes of device memory that a workload never using the
    /// tier should not be paying for.
    lt: Mutex<Option<LtTier>>,
    /// Whether single-precision operands may be rounded on their way into the
    /// multiplier. Atomic rather than behind the lock: every multiply reads it
    /// and nothing else in this struct is touched on that path.
    tf32: AtomicBool,
}

impl Blas {
    /// Opens the classic handle and sets the accumulation mode.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the library refuses a handle or the mode.
    pub(crate) fn open(tf32: bool) -> CoreResult<Self> {
        let classic = nvidia::blas_create()?;
        // SAFETY: the handle was just created and has not been destroyed.
        unsafe { nvidia::blas_set_math_mode(classic, tf32) }?;
        Ok(Self {
            classic,
            lt: Mutex::new(None),
            tf32: AtomicBool::new(tf32),
        })
    }

    /// Chooses whether single-precision operands may be rounded.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the library refuses the mode.
    pub(crate) fn set_tf32(&self, on: bool) -> CoreResult<()> {
        // SAFETY: the handle is owned by this value and lives as long as it.
        unsafe { nvidia::blas_set_math_mode(self.classic, on) }?;
        self.tf32.store(on, Ordering::Relaxed);
        Ok(())
    }

    /// Whether single-precision operands may currently be rounded.
    pub(crate) fn tf32(&self) -> bool {
        self.tf32.load(Ordering::Relaxed)
    }

    /// Binds the stream every later multiply runs on.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the library refuses the binding.
    pub(crate) fn bind_stream(&self, stream: CuStream) -> CoreResult<()> {
        // SAFETY: the handle is owned by this value; the stream is owned by the
        // device context, which outlives it.
        unsafe { nvidia::blas_set_stream(self.classic, stream) }?;
        Ok(())
    }

    /// The classic handle, for a call that is about to use it.
    const fn classic(&self) -> CuBlas {
        self.classic
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        // The heuristic tier owns its own handle and workspace and releases
        // them from its own `Drop`, which runs when this struct's fields are
        // dropped immediately after this body. Both happen while the device
        // context is still alive, because the context is declared last in the
        // backend and so outlives everything that holds a handle into it.
        // SAFETY: this value owns the only copy of the classic handle, released
        // here and only here.
        if let Err(err) = unsafe { nvidia::blas_destroy(self.classic) }
            && !err.is_teardown_noop()
        {
            tracing::error!(error = %err, "matrix library handle destroy failed");
        }
    }
}

/// Runs one multiply.
///
/// # Errors
///
/// [`CoreError::Backend`] for an invalid descriptor or a library failure,
/// [`CoreError::OutOfBounds`] for an operand shorter than its declared shape,
/// and whatever the buffer table reports for a range that is not live.
pub(crate) fn gemm(backend: &CudaBackend, gemm: &CudaGemm<'_>) -> CoreResult<()> {
    validate(gemm)?;
    backend.device.bind()?;
    let tf32 = backend.blas.tf32();

    if gemm.tier == BlasTier::Lt {
        if gemm.batch.is_some() {
            return Err(CoreError::Unsupported {
                what: "a strided batch on the heuristic matrix-multiply tier",
                backend: crate::sys::BACKEND,
            });
        }
        return lt_gemm(backend, gemm, tf32);
    }

    let state = backend.state.lock();
    let (call, stride) = resolve(&state.buffers, gemm, tf32)?;
    let handle = backend.blas.classic();
    match stride {
        // SAFETY: `validate` proved every operand covers the shape it declares
        // and `resolve` proved every range lies inside a live allocation. The
        // state lock is held, so nothing can free one under the call.
        None => unsafe { nvidia::blas_gemm(handle, &call) },
        // SAFETY: as above, and `validate` additionally covered every batch
        // element's stride.
        Some(stride) => unsafe { nvidia::blas_gemm_strided(handle, &call, &stride) },
    }?;
    Ok(())
}

/// Runs one multiply through the heuristic tier, opening it if needed.
fn lt_gemm(backend: &CudaBackend, gemm: &CudaGemm<'_>, tf32: bool) -> CoreResult<()> {
    let mut tier = backend.blas.lt.lock();
    if tier.is_none() {
        let handle = nvidia::blas_lt_create()?;
        let workspace = nvidia::mem_alloc(LT_WORKSPACE_BYTES).inspect_err(|_| {
            // SAFETY: the handle was just created and nothing else holds it.
            if let Err(err) = unsafe { nvidia::blas_lt_destroy(handle) } {
                tracing::error!(error = %err, "matmul library handle destroy failed");
            }
        })?;
        tracing::debug!(
            workspace_bytes = LT_WORKSPACE_BYTES,
            "opened the heuristic matrix-multiply tier",
        );
        *tier = Some(LtTier {
            handle,
            workspace,
            workspace_bytes: LT_WORKSPACE_BYTES,
        });
    }
    let opened = tier
        .as_ref()
        .ok_or_else(|| host_error("gemm", "the heuristic matrix-multiply tier failed to open"))?;

    let state = backend.state.lock();
    let (call, _) = resolve(&state.buffers, gemm, tf32)?;
    let stream = state.graphs.stream(&backend.device);
    // SAFETY: `validate` proved every operand covers the shape it declares and
    // `resolve` proved every range lies inside a live allocation; the state lock
    // is held, so nothing can free one under the call. The workspace is owned by
    // the tier, which is locked for the duration.
    unsafe {
        nvidia::blas_lt_matmul(
            opened.handle,
            &call,
            opened.workspace,
            opened.workspace_bytes,
            stream,
        )
    }?;
    Ok(())
}

/// Runs the plain multiply the shared trait describes.
///
/// # Errors
///
/// As [`gemm`].
pub(crate) fn gemm_core(backend: &CudaBackend, desc: &GemmDesc<'_>) -> CoreResult<()> {
    gemm(backend, &CudaGemm::from_core(desc))
}

/// The slab a caller widens a quantized operand into, grown to fit.
///
/// Grow-only and never shrunk, because the caller is a prefill chunk walking a
/// checkpoint's tensors: a slab sized for the widest weight serves every
/// narrower one, and a slab that shrank to each request would re-allocate at
/// every layer. It is released when the backend is.
///
/// Unlike the cooperative slab beside it this one is **not** cleared on
/// growth. Nothing reads it before the caller writes it - the staging kernel
/// covers every element of the plane the multiply then reads, and the multiply
/// reads no more than the plane - so clearing would be a full-width write of
/// zeroes over bytes that are about to be overwritten, which at a hundred
/// megabytes a tensor is the cost of the whole route.
///
/// `None` is not a refusal: it says this device produced no slab, and the
/// caller keeps whatever path it would have taken without one. Only a device
/// that refuses the allocation outright is an error.
///
/// # Errors
///
/// [`CoreError::Backend`] when the device refuses the allocation.
pub(crate) fn staging(backend: &CudaBackend, bytes: u64) -> CoreResult<Option<DeviceView<'_>>> {
    backend.device.bind()?;
    // The lock is dropped before the allocation for the reason
    // `CoopLaunch::reserve_scratch` states: `device::alloc` takes the same
    // lock and it is not reentrant.
    if let Some((raw, held)) = backend.state.lock().blas_staging
        && held >= bytes
    {
        return Ok(Some(DeviceView::from_raw(raw, 0, held)));
    }
    let raw = crate::device::alloc(backend, bytes.max(1))?;
    let spare = {
        let mut state = backend.state.lock();
        match state.blas_staging {
            // Another caller grew the slab while this one was allocating. Its
            // slab is at least as large, so it stays and this one is the spare.
            Some((_, held)) if held >= bytes => Some(raw),
            // The old slab goes only once the new one exists, so a failed
            // growth leaves the caller with what it had rather than nothing.
            _ => state.blas_staging.replace((raw, bytes)).map(|(old, _)| old),
        }
    };
    if let Some(spare) = spare {
        crate::device::free(backend, spare)?;
    }
    let Some((raw, held)) = backend.state.lock().blas_staging else {
        return Ok(None);
    };
    Ok(Some(DeviceView::from_raw(raw, 0, held)))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use shiso_core::{Backend, CoreError, DeviceBuffer, GemmDType};

    use super::{BlasTier, CudaGemm, GemmBatch, Op, Operand, min_elements, validate};
    use crate::testing::TableBackend;

    /// A buffer of `bytes` bytes, and the backend that owns it.
    fn buffers(sizes: &[u64]) -> (Arc<TableBackend>, Vec<DeviceBuffer>) {
        let backend = TableBackend::new();
        let handle: Arc<dyn Backend> = backend.clone();
        let buffers = sizes
            .iter()
            .map(|&bytes| shiso_core::alloc(&handle, bytes).expect("allocation"))
            .collect();
        (backend, buffers)
    }

    /// A 4x4 single-precision multiply over three buffers exactly large enough.
    fn square<'a>(a: &'a DeviceBuffer, b: &'a DeviceBuffer, c: &'a DeviceBuffer) -> CudaGemm<'a> {
        CudaGemm {
            op_a: Op::N,
            op_b: Op::N,
            m: 4,
            n: 4,
            k: 4,
            a: Operand {
                view: a.view(),
                ld: 4,
            },
            b: Operand {
                view: b.view(),
                ld: 4,
            },
            c: Operand {
                view: c.view(),
                ld: 4,
            },
            alpha: 1.0,
            beta: 0.0,
            dtype: GemmDType::F32,
            batch: None,
            tier: BlasTier::Classic,
        }
    }

    #[test]
    fn a_matrix_needs_room_for_its_last_column_and_not_a_byte_more() {
        // Four columns of stride 8, four rows: the last column starts at 24 and
        // is 4 long, so 28 elements is enough and 32 would be padding nobody
        // has to supply.
        assert_eq!(min_elements(4, 4, 8).expect("shape"), 28);
        assert_eq!(min_elements(4, 1, 8).expect("shape"), 4);
        assert_eq!(min_elements(1, 1, 1).expect("shape"), 1);
    }

    #[test]
    fn a_leading_dimension_shorter_than_the_rows_it_spans_is_refused() {
        match min_elements(8, 2, 4) {
            Err(CoreError::Backend { source, .. }) => {
                let text = source.to_string();
                assert!(text.contains("shorter than"), "{text}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_multiply_whose_operands_exactly_fit_is_accepted() {
        let (_backend, bufs) = buffers(&[64, 64, 64]);
        let gemm = square(&bufs[0], &bufs[1], &bufs[2]);
        validate(&gemm).expect("three 4x4 single-precision matrices need 64 bytes each");
    }

    #[test]
    fn an_operand_one_element_short_of_its_shape_is_out_of_bounds() {
        let (_backend, bufs) = buffers(&[60, 64, 64]);
        let gemm = square(&bufs[0], &bufs[1], &bufs[2]);
        match validate(&gemm) {
            Err(CoreError::OutOfBounds {
                len, buffer_bytes, ..
            }) => {
                assert_eq!((len, buffer_bytes), (64, 60));
            }
            other => panic!("expected an out-of-bounds error, got {other:?}"),
        }
    }

    #[test]
    fn a_transposed_operand_is_measured_against_its_stored_shape() {
        // With op_a = T the left operand is stored k-by-m, so a leading
        // dimension that spans m rows is the wrong one and a 2x8 problem needs
        // a different length than an 8x2 one.
        let (_backend, bufs) = buffers(&[64, 256, 64]);
        let mut gemm = square(&bufs[0], &bufs[1], &bufs[2]);
        gemm.m = 2;
        gemm.n = 2;
        gemm.k = 8;
        gemm.a.ld = 8;
        gemm.b.ld = 8;
        gemm.c.ld = 2;
        gemm.op_a = Op::T;
        // Stored 8 rows by 2 columns with ld 8: 8 * 1 + 8 = 16 elements, 64
        // bytes, which is exactly what the buffer holds.
        validate(&gemm).expect("a transposed left operand measured against its stored shape");

        gemm.op_a = Op::N;
        // Stored 2 rows by 8 columns with ld 8: 8 * 7 + 2 = 58 elements, 232
        // bytes, which the same buffer does not hold.
        assert!(
            validate(&gemm).is_err(),
            "the same buffer cannot serve both orientations, which is why the flag is checked",
        );
    }

    #[test]
    fn a_dimension_of_zero_is_refused_by_name() {
        let (_backend, bufs) = buffers(&[64, 64, 64]);
        for field in 0..3 {
            let mut gemm = square(&bufs[0], &bufs[1], &bufs[2]);
            let name = match field {
                0 => {
                    gemm.m = 0;
                    "m"
                }
                1 => {
                    gemm.n = 0;
                    "n"
                }
                _ => {
                    gemm.k = 0;
                    "k"
                }
            };
            match validate(&gemm) {
                Err(CoreError::Backend { source, .. }) => {
                    let text = source.to_string();
                    assert!(text.starts_with(name), "{text}");
                }
                other => panic!("expected a refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_batch_needs_room_for_every_element_it_declares() {
        let (_backend, bufs) = buffers(&[64 * 3, 64 * 3, 64 * 3]);
        let mut gemm = square(&bufs[0], &bufs[1], &bufs[2]);
        gemm.batch = Some(GemmBatch {
            count: 3,
            stride_a: 16,
            stride_b: 16,
            stride_c: 16,
        });
        validate(&gemm)
            .expect("three 4x4 matrices at a stride of sixteen elements fit in 192 bytes");

        gemm.batch = Some(GemmBatch {
            count: 4,
            stride_a: 16,
            stride_b: 16,
            stride_c: 16,
        });
        assert!(
            validate(&gemm).is_err(),
            "a fourth element has to be paid for in every operand",
        );
    }

    #[test]
    fn an_empty_batch_is_refused() {
        let (_backend, bufs) = buffers(&[64, 64, 64]);
        let mut gemm = square(&bufs[0], &bufs[1], &bufs[2]);
        gemm.batch = Some(GemmBatch {
            count: 0,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
        });
        assert!(validate(&gemm).is_err());
    }

    #[test]
    fn half_precision_operands_need_half_the_bytes() {
        let (_backend, bufs) = buffers(&[32, 32, 32]);
        let mut gemm = square(&bufs[0], &bufs[1], &bufs[2]);
        gemm.dtype = GemmDType::F16;
        validate(&gemm).expect("three 4x4 half-precision matrices need 32 bytes each");
        gemm.dtype = GemmDType::F32;
        assert!(
            validate(&gemm).is_err(),
            "the same buffers cannot hold the single-precision form",
        );
    }

    #[test]
    fn the_classic_tier_is_the_default() {
        assert_eq!(BlasTier::default(), BlasTier::Classic);
    }
}

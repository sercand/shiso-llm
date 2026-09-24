//! Weights that live on the device, and the projection that reads them.
//!
//! A projection is `dst[out + tok * out_dim] = sum_in weight[out, in] * src[in + tok * in_dim]`,
//! f32 on the activation side and the weight in whatever the file stored it in.
//! Two kernel families implement it, and this is the only file that chooses
//! between them:
//!
//! - the extended family blocks 1..=5 tokens per threadgroup and covers `q4_K`,
//!   `q5_K`, `q6_K`, `f16`, and `q8_0` at two tokens or more;
//! - the plain family takes what that one has no instantiation for, which is
//!   every `f32` projection and a single-token `q8_0` one;
//! - the `modelopt` pair - FP8 E4M3 and NVFP4 - has a matvec and a tiled GEMM
//!   of its own in the `qwen35moe` module and reaches neither family above,
//!   because their scales are arrays beside the weights rather than fields
//!   inside a block, so an entry of theirs takes a buffer those families have
//!   no slot for.
//!
//! Both write the same row-major `out_dim`-wide output, so nothing downstream has
//! to know which one ran.
//!
//! - the tiled GEMM takes a chunk of two rows or more when the format has one,
//!   which is `q4_K`, `q8_0` and `f16`.
//!
//! # And above the tile, the vendor matrix library
//!
//! On a backend that has one, a prefill chunk does not take the tile at all:
//! the weight is widened to f32 by `kernel_dequant_plane_*` and multiplied by
//! `as_blas()`, which measured 2.1x on this device's prefill. See
//! [`blas_projection`], which also says why that route is called at one fixed
//! token count whatever the chunk carries. It is an `Option` and not a
//! requirement: a backend whose `as_blas()` is `None` - every Metal build -
//! keeps the tile, and so does a caller whose activation and output buffers
//! are shorter than [`crate::MAX_PREFILL_CHUNK`] rows.
//!
//! # Why the GEMM is a correctness concern and not only a speed one
//!
//! [`ext_r1ptg`] maps any token count onto a row blocking the extended family has
//! an instantiation for, so the matvecs alone would answer every shape - but a
//! matvec re-reads the whole weight matrix once per row block, and a 64-row chunk
//! through the four-row blocking therefore reads a two-gigabyte checkpoint sixteen
//! times. That is not a tuning difference; it is the difference between a prompt
//! that prefills and one that appears to hang. The GEMM is the default for a chunk
//! and the matvec is the fallback for the formats and widths it does not cover.

use std::ops::Range;
use std::sync::Arc;

use shiso_core::backend::{Backend, FnConst, KernelArg, KernelRef};

use shiso_core::{
    CoreError, CoreResult, DType, DeviceBuffer, DeviceView, Extent3, GemmDType, GemmDesc, GemmOp,
    GemmReduction, HostBytes, alloc, import_host,
};

use crate::launch::{
    DEQUANT_BLOCK, DenseAffineArgs, DequantPlaneArgs, F32_BYTES, GEMM_BLOCK, GEMM_SMEM_BYTES,
    GENERIC, MulMmArgs, MulMvArgs, MulMvExtArgs, PlainMv, Q35, Q35MOE, dequant_plane_grid,
    dequant_plane_name, ext_block, ext_consts_static, ext_grid, ext_module, ext_name, ext_name_for,
    ext_nxpsg, ext_r1ptg, launch_failure, mm_consts_for, mm_grid, mm_name, plain_block,
    plain_consts_for, plain_mv, plain_mv_grid, run,
};

/// One projection dimension, as the kernels spell it.
///
/// Every argument block names a dimension `int32_t`, so a width that does not fit
/// is refused here rather than wrapped into a negative stride that a kernel would
/// then walk backwards.
fn extent(value: u32, field: &'static str) -> CoreResult<i32> {
    i32::try_from(value).map_err(|_| CoreError::ShapeMismatch {
        field,
        expected: u64::try_from(i32::MAX).unwrap_or(u64::MAX),
        got: u64::from(value),
    })
}

/// Bytes of one weight row, counted in whole blocks.
///
/// Both matvec families step a row in whole blocks, so a width that is not a whole
/// number of them would have its tail read out of the row below. Refusing the width
/// is better than a projection that answers from the neighbour.
fn row_bytes(dtype: DType, in_dim: u32) -> CoreResult<u64> {
    let elems = u64::from(dtype.block_elems().get());
    let bytes = u64::from(dtype.block_bytes().get());
    let width = u64::from(in_dim);
    if width == 0 || !width.is_multiple_of(elems) {
        return Err(CoreError::InvalidValue {
            key: "projection in_dim",
            value: format!("{width} is not a whole number of {elems}-element blocks"),
        });
    }
    (width / elems)
        .checked_mul(bytes)
        .ok_or(CoreError::ShapeMismatch {
            field: "weight row bytes",
            expected: u64::MAX,
            got: width,
        })
}

/// Bytes one `bf16` scale or bias word occupies.
const BF16_BYTES: u64 = 2;

/// Threads per block of the `qwen35moe` FP8 matvec entry, which is
/// `SHISO_Q35MOE_BLOCK` in `kernels/cuda/qwen35moe/q35moe_common.cuh`.
///
/// The FP8 entry alone. It gives one block to one output row and folds through
/// shared memory, and at this family's widths that is the right shape for it:
/// a byte per weight means a 2048-wide row is 2048 bytes, which is eight bytes
/// a thread and a fold worth paying for. The entry measures 211 GB/s of this
/// device's 239 GB/s peak on the decode step, so there is nothing in its
/// geometry to win.
const MODELOPT_MV_BLOCK: u32 = 256;

/// Output rows one block of the `qwen35moe` NVFP4 matvec covers, which is
/// `SHISO_Q35MOE_MV_ROWS` there.
///
/// The NVFP4 entry takes the other shape - one WARP per output row, four rows
/// to a block - because four bits a weight makes the same row 1024 bytes, and
/// a 256-thread block splitting 1024 bytes pays a 256-wide fold for four bytes
/// a thread. It measured 47-100 GB/s against the FP8 entry's 211 on the same
/// step before this split.
const MODELOPT_MV_ROWS: u32 = 4;

/// Threads per block of the `qwen35moe` NVFP4 matvec entry, which is
/// `SHISO_Q35MOE_MV_BLOCK` there: one warp per row.
const MODELOPT_NVFP4_MV_BLOCK: u32 = 32 * MODELOPT_MV_ROWS;

/// Tile edge of the `qwen35moe` GEMM entries, which is `SHISO_Q35MOE_TILE`
/// there. Both axes, and the block is one thread per output element.
const MODELOPT_TILE: u32 = 16;

/// Checks a packed affine weight's element type and geometry, returning the
/// number of quantization groups one row carries.
///
/// The group count is the whole check: every affine entry point walks a row as
/// `in_dim / group` groups and reads one scale and one bias for each, so an
/// `in_dim` that is not a whole number of groups leaves a tail the kernel never
/// reads and produces a projection that is wrong in its last columns and
/// nowhere else - the hardest kind of wrong to notice.
fn affine_geometry(dtype: DType, in_dim: u32, out_dim: u32) -> CoreResult<u64> {
    extent(in_dim, "projection in_dim")?;
    extent(out_dim, "projection out_dim")?;
    if !dtype.is_affine() {
        return Err(CoreError::InvalidValue {
            key: "dtype",
            value: format!("{dtype:?} is not affine"),
        });
    }
    let group = dtype.block_elems().get();
    if !in_dim.is_multiple_of(group) {
        return Err(CoreError::InvalidValue {
            key: "affine in_dim",
            value: format!("{in_dim} is not a whole number of {group}-element groups"),
        });
    }
    Ok(u64::from(in_dim / group))
}

/// Refuses an array whose length is not the one the geometry implies.
fn expect_len(field: &'static str, got: usize, want: u64) -> CoreResult<()> {
    let got = u64::try_from(got).unwrap_or(u64::MAX);
    if got == want {
        return Ok(());
    }
    Err(CoreError::ShapeMismatch {
        field,
        expected: want,
        got,
    })
}

/// Auxiliary buffers for packed affine weights (scales and biases).
#[derive(Debug)]
pub struct AffineAux {
    /// Per-group scales as BF16 words.
    pub scales: DeviceBuffer,
    /// Per-group biases as BF16 words.
    pub biases: DeviceBuffer,
}

/// What a weight carries beside its own bytes.
///
/// Three codecs in this engine store a matrix as more than one array, and they
/// do not agree on what the extra arrays are: an MLX affine weight has a scale
/// and a bias per group, an FP8 weight has one `f32` for the whole tensor and
/// possibly a second for the activations, and an NVFP4 weight has an `e4m3`
/// byte per group AND one `f32` above them. An enum rather than three
/// `Option`s on [`Weight`], because a weight is exactly one of these and three
/// `Option`s would let it be two.
#[derive(Debug)]
pub enum WeightAux {
    /// Per-group scales and biases, for an MLX affine weight.
    Affine(AffineAux),
    /// FP8 E4M3: one `f32` for the whole tensor.
    Fp8 {
        /// The one `f32` every weight of this tensor is multiplied by.
        weight_scale: f32,
        /// The scale the activations were calibrated at, when the export
        /// states one.
        ///
        /// Carried and not yet read. The device path decodes the weights and
        /// multiplies by `f32` activations, which is more precise than the
        /// checkpoint's own W8A8 intent and is what the oracles grade; this is
        /// what a later tensor-core path needs and it is kept so that path
        /// does not have to reopen the checkpoint to find it.
        input_scale: Option<f32>,
    },
    /// NVFP4: an `e4m3` byte per group of sixteen, and one `f32` above them.
    Nvfp4 {
        /// One `e4m3` byte per group of sixteen weights along a row.
        group_scales: DeviceBuffer,
        /// The whole tensor's scale, which the group scales are relative to.
        ///
        /// A field rather than something a caller folds in elsewhere: dropping
        /// it rescales a tensor by about 31,458x in this checkpoint, and the
        /// next normalization very nearly hides that.
        global: f32,
    },
}

impl WeightAux {
    /// The affine pair, or `None` for a weight of another codec.
    #[must_use]
    pub const fn affine(&self) -> Option<&AffineAux> {
        match self {
            Self::Affine(aux) => Some(aux),
            Self::Fp8 { .. } | Self::Nvfp4 { .. } => None,
        }
    }
}

/// Slices of words, scales and biases for importing a packed affine weight.
#[derive(Debug, Clone)]
pub struct AffineRanges {
    /// Byte range of packed u32 words.
    pub words: Range<usize>,
    /// Byte range of bf16 scales.
    pub scales: Range<usize>,
    /// Byte range of bf16 biases.
    pub biases: Range<usize>,
}

/// One projection weight, resident on the device.
///
/// The buffer is private on purpose: a kernel is never handed a buffer, only a
/// [`DeviceView`] that cannot outlive this, and that is the whole reason the type
/// exists. A weight that has been dropped cannot be read by a launch that was
/// queued before it went.
pub struct Weight {
    /// Element type the weight was stored as.
    pub dtype: DType,
    /// Elements in one weight row: the projection's input width.
    pub in_dim: u32,
    /// Weight rows: the projection's output width.
    pub out_dim: u32,
    buffer: DeviceBuffer,
    aux: Option<WeightAux>,
}

impl Weight {
    /// Wraps `host[range]` as a device weight, without copying where the backend can.
    ///
    /// This is the path a checkpoint loads by: the host bytes are the file's own
    /// mapping and a quantized model is gigabytes the device can read out of the
    /// page cache instead of through a copy. The range is still checked against the
    /// weight's own size rather than trusted, because the two come from different
    /// places - the shape from the binding, the extent from the file's directory -
    /// and a payload shorter than the declared shape should fail here rather than
    /// read off the end of a mapping.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when `range` is not exactly the weight's size,
    /// and whatever [`import_host`] reports.
    pub fn import(
        backend: &Arc<dyn Backend>,
        dtype: DType,
        in_dim: u32,
        out_dim: u32,
        host: Arc<dyn HostBytes>,
        range: Range<usize>,
    ) -> CoreResult<Self> {
        extent(in_dim, "projection in_dim")?;
        extent(out_dim, "projection out_dim")?;
        dispatchable(dtype, in_dim)?;
        let row = row_bytes(dtype, in_dim)?;
        let total = row
            .checked_mul(u64::from(out_dim))
            .ok_or(CoreError::ShapeMismatch {
                field: "weight bytes",
                expected: u64::MAX,
                got: row,
            })?;
        let given = u64::try_from(range.end.saturating_sub(range.start)).unwrap_or(u64::MAX);
        if given != total {
            return Err(CoreError::ShapeMismatch {
                field: "weight import bytes",
                expected: total,
                got: given,
            });
        }
        let buffer = import_host(backend, host, range)?;
        Ok(Self {
            dtype,
            in_dim,
            out_dim,
            buffer,
            aux: None,
        })
    }

    /// Imports a packed affine weight from three byte ranges.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when `dtype` is not affine or `in_dim` is not
    /// a whole number of quantization groups, [`CoreError::ShapeMismatch`]
    /// when dimensions do not fit, or backend errors from buffer allocation and copy.
    pub fn import_affine(
        backend: &Arc<dyn Backend>,
        dtype: DType,
        in_dim: u32,
        out_dim: u32,
        host: Arc<dyn HostBytes>,
        ranges: AffineRanges,
    ) -> CoreResult<Self> {
        affine_geometry(dtype, in_dim, out_dim)?;
        let buffer = import_host(backend, Arc::clone(&host), ranges.words)?;
        let scales = import_host(backend, Arc::clone(&host), ranges.scales)?;
        let biases = import_host(backend, host, ranges.biases)?;
        Ok(Self {
            dtype,
            in_dim,
            out_dim,
            buffer,
            aux: Some(WeightAux::Affine(AffineAux { scales, biases })),
        })
    }

    /// Imports a packed affine weight from three byte slices.
    ///
    /// The three lengths are checked against the geometry rather than trusted:
    /// the kernels index words, scales and biases from `in_dim` and `out_dim`
    /// alone, so a short array is read past its end and a long one hides a
    /// layout this loader has misread.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when `dtype` is not affine or `in_dim` is not
    /// a whole number of quantization groups, [`CoreError::ShapeMismatch`]
    /// when dimensions or array lengths do not fit, or backend errors from
    /// buffer allocation and copy.
    pub fn import_affine_slices(
        backend: &Arc<dyn Backend>,
        dtype: DType,
        in_dim: u32,
        out_dim: u32,
        words: &[u8],
        scales: &[u8],
        biases: &[u8],
    ) -> CoreResult<Self> {
        let groups = affine_geometry(dtype, in_dim, out_dim)?;
        let bits = u64::from(dtype.affine_bits().unwrap_or_default());
        let rows = u64::from(out_dim);
        expect_len(
            "affine words",
            words.len(),
            rows * u64::from(in_dim) * bits / 8,
        )?;
        expect_len("affine scales", scales.len(), rows * groups * BF16_BYTES)?;
        expect_len("affine biases", biases.len(), rows * groups * BF16_BYTES)?;
        let buffer = alloc(backend, u64::try_from(words.len()).unwrap_or(0))?;
        backend.copy_h2d(buffer.view(), words)?;
        let s_buf = alloc(backend, u64::try_from(scales.len()).unwrap_or(0))?;
        backend.copy_h2d(s_buf.view(), scales)?;
        let b_buf = alloc(backend, u64::try_from(biases.len()).unwrap_or(0))?;
        backend.copy_h2d(b_buf.view(), biases)?;
        Ok(Self {
            dtype,
            in_dim,
            out_dim,
            buffer,
            aux: Some(WeightAux::Affine(AffineAux {
                scales: s_buf,
                biases: b_buf,
            })),
        })
    }
    /// Imports an FP8 weight and the scale that decodes it.
    ///
    /// `input_scale` is carried rather than applied. A target whose
    /// activations were calibrated is a different launch from one whose were
    /// not, and neither is "an activation scale of one"; the device path here
    /// quantizes neither side, so this is the number a later tensor-core path
    /// will want and the reason it does not have to reopen the checkpoint.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when `weight_scale` is not finite,
    /// [`CoreError::ShapeMismatch`] when the range is not exactly one byte per
    /// weight, and whatever [`import_host`] reports.
    pub fn import_fp8(
        backend: &Arc<dyn Backend>,
        in_dim: u32,
        out_dim: u32,
        host: Arc<dyn HostBytes>,
        range: Range<usize>,
        weight_scale: f32,
        input_scale: Option<f32>,
    ) -> CoreResult<Self> {
        finite("fp8 weight_scale", weight_scale)?;
        if let Some(scale) = input_scale {
            finite("fp8 input_scale", scale)?;
        }
        let mut weight = Self::import(backend, DType::Fp8E4M3, in_dim, out_dim, host, range)?;
        weight.aux = Some(WeightAux::Fp8 {
            weight_scale,
            input_scale,
        });
        Ok(weight)
    }

    /// Imports an NVFP4 weight, its per-group scale array and its per-tensor
    /// scale.
    ///
    /// The scale array's length is checked against the geometry rather than
    /// trusted: the kernels index it from `in_dim` and `out_dim` alone, so an
    /// array one group short is read past its end on the last row and every
    /// earlier row reads the group below its own.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when `global` is not finite,
    /// [`CoreError::ShapeMismatch`] when either range is not the length the
    /// geometry implies, and whatever [`import_host`] reports.
    pub fn import_nvfp4(
        backend: &Arc<dyn Backend>,
        in_dim: u32,
        out_dim: u32,
        host: Arc<dyn HostBytes>,
        weights: Range<usize>,
        scales: Range<usize>,
        global: f32,
    ) -> CoreResult<Self> {
        finite("nvfp4 weight_scale_2", global)?;
        let mut weight = Self::import(
            backend,
            DType::Nvfp4Gs16,
            in_dim,
            out_dim,
            Arc::clone(&host),
            weights,
        )?;
        let groups = u64::from(in_dim) / u64::from(DType::Nvfp4Gs16.block_elems().get());
        let want = groups.saturating_mul(u64::from(out_dim));
        let given = u64::try_from(scales.end.saturating_sub(scales.start)).unwrap_or(u64::MAX);
        if given != want {
            return Err(CoreError::ShapeMismatch {
                field: "nvfp4 group scale bytes",
                expected: want,
                got: given,
            });
        }
        weight.aux = Some(WeightAux::Nvfp4 {
            group_scales: import_host(backend, host, scales)?,
            global,
        });
        Ok(weight)
    }

    /// Constructs a weight from an already-allocated device buffer.
    #[must_use]
    pub fn from_buffer(dtype: DType, in_dim: u32, out_dim: u32, buffer: DeviceBuffer) -> Self {
        Self {
            dtype,
            in_dim,
            out_dim,
            buffer,
            aux: None,
        }
    }

    /// Whatever this weight carries beside its own bytes.
    #[must_use]
    pub fn aux(&self) -> Option<&WeightAux> {
        self.aux.as_ref()
    }

    /// The whole weight, as a kernel argument.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_> {
        self.buffer.view()
    }

    /// Bytes of one weight row: [`row_bytes`] at this weight's own width.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the width is not a whole number of the
    /// format's blocks.
    pub fn row_bytes(&self) -> CoreResult<u64> {
        row_bytes(self.dtype, self.in_dim)
    }
}

impl std::fmt::Debug for Weight {
    /// Names the weight by its shape and format, and its allocation by size.
    ///
    /// Not derived, because a device allocation has no host-readable content: the
    /// derived form would print an opaque identifier that tells a reader nothing
    /// and hides the three numbers that actually distinguish one weight from
    /// another.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weight")
            .field("dtype", &self.dtype)
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .field("bytes", &self.buffer.len())
            .finish_non_exhaustive()
    }
}

/// Refuses a scale that cannot multiply.
///
/// A NaN or an infinity here is not a rounding problem: it poisons every
/// weight of the tensor identically, which produces a uniformly NaN residual
/// that the next normalization turns into a uniformly NaN row - and that IS
/// visible, eventually, several layers later and attributed to whatever was
/// running then.
fn finite(key: &'static str, value: f32) -> CoreResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(CoreError::InvalidValue {
            key,
            value: format!("{value}"),
        })
    }
}

/// Refuses a format no matvec family can launch at some row blocking.
///
/// Checked at import rather than at the launch, because a format with no kernel
/// is a property of the file: a model that discovered it on the first forward
/// pass would already have spent the load, and the caller would learn about its
/// checkpoint from a failure forty tokens into a generation.
///
/// Both families are consulted. A format the extended family skips at one row
/// count is fine as long as the plain family covers that width, which is exactly
/// the case the single-row `q8_0` path exists for.
///
/// # Errors
///
/// [`CoreError::Unsupported`] naming the gap.
fn dispatchable(dtype: DType, in_dim: u32) -> CoreResult<()> {
    if dtype.is_affine() {
        return Ok(());
    }
    if dtype.is_modelopt() {
        // Both have a matvec and a GEMM of their own in the `qwen35moe`
        // module, and neither is reachable through the extended or plain
        // families - so the loop below would refuse them for having no entry
        // in tables they were never going to be in.
        return modelopt_dispatchable(dtype, in_dim);
    }
    let plain = plain_mv(dtype, u64::from(in_dim)).is_some();
    for blocking in 1..=5 {
        if ext_name(dtype, blocking).is_none() && !plain {
            return Err(CoreError::Unsupported {
                what: "a weight whose format has no matvec at one of the row blockings",
                backend: "engine",
            });
        }
    }
    Ok(())
}

/// The extended matvec's argument block for one projection shape.
///
/// Weight and activations are both one plane deep, so the plane strides repeat the
/// row stride instead of describing a second batch axis: the tokens are the second
/// axis of the product, and that is the axis the grid spreads. `nb00` is the row
/// size, which is what the device fixture this family was verified against passes;
/// the kernel advances a row by `nb01` and never reads `nb00`.
fn ext_args(dtype: DType, in_dim: u32, out_dim: u32, n_tok: u32) -> CoreResult<MulMvExtArgs> {
    let ne00 = extent(in_dim, "projection in_dim")?;
    let ne01 = extent(out_dim, "projection out_dim")?;
    let rows = extent(n_tok, "projection n_tok")?;
    let row = row_bytes(dtype, in_dim)?;
    let planes = row.saturating_mul(u64::from(out_dim));
    let act_row = u64::from(in_dim).saturating_mul(F32_BYTES);
    let act_planes = act_row.saturating_mul(u64::from(n_tok));
    Ok(MulMvExtArgs {
        ne00,
        ne01,
        ne02: 1,
        pad0: 0,
        nb00: row,
        nb01: row,
        nb02: planes,
        nb03: planes,
        ne10: ne00,
        ne11: rows,
        ne12: 1,
        pad1: 0,
        nb10: F32_BYTES,
        nb11: act_row,
        nb12: act_planes,
        nb13: act_planes,
        ne0: ne01,
        ne1: rows,
        r2: 1,
        r3: 1,
        pad2: [0; 4],
    })
}

/// The plain matvec's argument block for one projection shape.
///
/// The plain family takes the same product as the extended one and reads the rows
/// per threadgroup as an argument (`nr0`) rather than as a template parameter, so
/// the blocking the caller chose travels inside the block.
fn plain_args(
    dtype: DType,
    in_dim: u32,
    out_dim: u32,
    n_tok: u32,
    mv: &PlainMv,
) -> CoreResult<MulMvArgs> {
    let ne00 = extent(in_dim, "projection in_dim")?;
    let ne01 = extent(out_dim, "projection out_dim")?;
    let rows = extent(n_tok, "projection n_tok")?;
    let nr0 = extent(mv.rows_per_threadgroup, "projection rows per threadgroup")?;
    let row = row_bytes(dtype, in_dim)?;
    let planes = row.saturating_mul(u64::from(out_dim));
    let act_row = u64::from(in_dim).saturating_mul(F32_BYTES);
    let act_planes = act_row.saturating_mul(u64::from(n_tok));
    Ok(MulMvArgs {
        ne00,
        ne01,
        ne02: 1,
        pad0: [0; 4],
        nb00: row,
        nb01: row,
        nb02: planes,
        nb03: planes,
        ne10: ne00,
        ne11: rows,
        ne12: 1,
        pad1: [0; 4],
        nb10: F32_BYTES,
        nb11: act_row,
        nb12: act_planes,
        nb13: act_planes,
        ne0: ne01,
        ne1: rows,
        nr0,
        r2: 1,
        r3: 1,
    })
}

/// Multiplies one resident weight by an activation block.
///
/// `src` is `n_tok` rows of `in_dim` f32 and `dst` is `n_tok` rows of `out_dim`
/// f32, both contiguous. Both lengths are checked before anything is launched: a
/// matvec that writes past its output is a wrong answer rather than a crash, and
/// the two operands are the only place the shapes can still be argued about.
///
/// # Errors
///
/// [`CoreError::OutOfBounds`] when either activation range is shorter than the
/// launch would write, [`CoreError::ShapeMismatch`] for a width the argument block
/// cannot carry, and [`CoreError::InvalidValue`] when the format has no matvec
/// instantiation at this token count at all.
pub fn projection(
    backend: &dyn Backend,
    weight: &Weight,
    n_tok: u32,
    src: DeviceView<'_>,
    dst: DeviceView<'_>,
) -> CoreResult<()> {
    let tokens = u64::from(n_tok);
    let need_src = u64::from(weight.in_dim)
        .saturating_mul(tokens)
        .saturating_mul(F32_BYTES);
    if src.len() < need_src {
        return Err(CoreError::OutOfBounds {
            offset: 0,
            len: need_src,
            buffer_bytes: src.len(),
        });
    }
    let need_dst = u64::from(weight.out_dim)
        .saturating_mul(tokens)
        .saturating_mul(F32_BYTES);
    if dst.len() < need_dst {
        return Err(CoreError::OutOfBounds {
            offset: 0,
            len: need_dst,
            buffer_bytes: dst.len(),
        });
    }
    // A chunk wide enough to tile goes through the GEMM when its format has one:
    if weight.dtype.is_affine() {
        return affine_projection(backend, weight, n_tok, src, dst);
    }
    if weight.dtype.is_modelopt() {
        return modelopt_projection(backend, weight, n_tok, src, dst);
    }
    // the matvec family re-reads the whole weight matrix once per row block, so a
    // 64-token prompt through the four-row blocking reads every weight sixteen
    // times over. The GEMM reads them once.
    if n_tok >= GEMM_MIN_TOKENS && mm_name(weight.dtype).is_some() {
        return mm_projection(backend, weight, n_tok, src, dst);
    }
    let r1ptg = ext_r1ptg(n_tok);
    let nxpsg = ext_nxpsg(u64::from(weight.in_dim), n_tok);
    // The lane split is a pipeline constant on one backend and part of the
    // entry name on the other, and which it is belongs to the device rather
    // than to this dispatch. Asking here keeps the two spellings from being a
    // `cfg`: a CUDA module holds one symbol per instantiation and refuses a
    // launch-time constant outright, so the name has to carry the split there.
    let by_name = backend.specializes_by_name();
    if let Some(entry) = ext_name_for(weight.dtype, r1ptg, nxpsg, by_name) {
        let consts: &'static [FnConst] = if by_name {
            &[]
        } else {
            ext_consts_static(nxpsg).ok_or(CoreError::InvalidValue {
                key: "projection lane split",
                value: format!("{nxpsg} is not a split the extended family is built with"),
            })?
        };
        let args = ext_args(weight.dtype, weight.in_dim, weight.out_dim, n_tok)?;
        return run(
            backend,
            KernelRef::specialized(ext_module(weight.dtype), entry, consts),
            ext_grid(weight.out_dim, n_tok, nxpsg, r1ptg),
            ext_block(),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(weight.view()),
                KernelArg::Buffer(src),
                KernelArg::Buffer(dst),
            ],
        );
    }
    plain_projection(backend, weight, n_tok, src, dst)
}

/// Rows from which the tiled GEMM is worth its tile staging.
///
/// **One past the widest decode batch, and that is a measured number rather than
/// a guess.** It was `2` - as low as the gate can go and still mean anything,
/// since at one row the product is a matvec and the GEMM would stage a 32-wide
/// tile to use one column of it - which meant that the moment a decode step
/// carried two rows, every projection left the matvec family and took the
/// GEMM. Measured on this device with `nsys`, over the dense checkpoint's 1.36
/// GB of `q4_K` projections (131 launches a step, one per tensor, the same
/// bytes whichever family runs them):
///
/// | rows | entry | ms/step | achieved |
/// | --- | --- | --- | --- |
/// | 1 | `kernel_mul_mv_ext_q4_K_f32_r1_1_nx16` | 11.38 | **119.6 GB/s** |
/// | 2 | `kernel_mul_mm_q4_K_f32` | 77.58 | **17.5 GB/s** |
/// | 4 | `kernel_mul_mm_q4_K_f32` | 78.20 | **17.4 GB/s** |
/// | 8 | `kernel_mul_mm_q4_K_f32` | 78.46 | **17.3 GB/s** |
///
/// The GEMM is 6.8x slower *per byte* than the matvec it replaces and flat in
/// the row count, so at decode widths it is a fixed 78 ms whatever the batch -
/// which made a batch of two slower than two sequential steps. The matvec
/// family is not: it blocks up to five rows per threadgroup
/// ([`ext_r1ptg`]), so two and four rows read the weights once and eight read
/// them twice.
///
/// The threshold therefore says **a decode batch takes the matvec and a chunk
/// wider than one takes the GEMM**. A prompt shorter than nine tokens is on the
/// matvec side of that line too, which is deliberate rather than tolerated:
/// [`ext_r1ptg`] blocks such a chunk at up to five rows per threadgroup, so it
/// reads the weight once exactly as the GEMM would, at six times the rate.
///
/// The crossover measured above in fact sits well past this line - the GEMM
/// only wins once the matvec would re-read the weights more than about 4.4
/// times over, around sixty rows - and the threshold is deliberately NOT put
/// there. Nine is the width this phase measured on both sides of; sixty is an
/// extrapolation, and a prefill chunk is [`crate::MAX_PREFILL_CHUNK`] rows
/// either way.
///
/// **This does not make the GEMM fast, and a later phase still owns that.** All
/// it does is stop a decode step paying for it. `docs/handoff/cuda-decode-perf.md`
/// carries the measurement and what it implies.
///
/// [`ext_r1ptg`]: crate::launch::ext_r1ptg
const GEMM_MIN_TOKENS: u32 = crate::MAX_DECODE_BATCH + 1;

/// Tokens one vendor multiply covers, whatever the chunk carries.
///
/// The number itself is measured; that it is *a* number rather than the
/// chunk's own width is the correctness requirement [`blas_projection`]
/// states. A chunk wider than this is covered by several multiplies of this
/// shape rather than one wider one, so every output element is still produced
/// by a multiply of the same shape and therefore in the same reduction order.
///
/// It has to divide [`crate::MAX_PREFILL_CHUNK`], because the prefill scratch
/// is sized for that many rows and a pad that did not divide it would need a
/// band past the end of the buffer. That is asserted below rather than
/// promised here: a pad somebody retunes to 192 would otherwise read past a
/// live allocation and return numbers.
///
/// Measured on a GB10 2026-09-22, dense prefill in slope tok/s at 64 / 512 /
/// 2048 tokens, against **291.37 / 296.65 / 278.26** for the tile this
/// replaces:
///
/// | pad | 64 | 512 | 2048 |
/// | --- | --- | --- | --- |
/// | 128 | 379.04 | 545.68 | 489.22 |
/// | **256** | **336.38** | **597.64** | **530.22** |
/// | 512 | 259.30 | 623.18 | 546.95 |
///
/// The staged plane is read once per band, so a narrow pad costs a long chunk
/// a second read of a hundred megabytes and a wide one costs a short chunk
/// arithmetic on columns that are not there. 512 is the fastest at the two
/// long lengths and the only one that is *slower than the tile* at 64, which
/// is a regression against a committed baseline; 256 is the widest pad that
/// beats the tile at every length measured, and that is why it is this.
const PREFILL_BLAS_PAD: u32 = 256;

/// The invariant [`PREFILL_BLAS_PAD`]'s own documentation states, enforced.
const _: () = assert!(
    crate::MAX_PREFILL_CHUNK.is_multiple_of(PREFILL_BLAS_PAD),
    "the vendor band width must divide the prefill chunk cap, or the last \
     band of a full chunk runs past the scratch the stack allocated"
);

/// The vendor-library half of [`projection`], or `false` when this device has
/// no vendor library to route to.
///
/// # What this replaces, and by how much
///
/// The hand-written tile in `kernels/cuda/generic/mul_mm.cuh` is a port of a
/// Metal threadgroup shape and reaches no tensor core on this device. Measured
/// 2026-09-22 over the dense checkpoint's 131 `q4_K` projections at a
/// 512-token chunk, it takes **1051 ms of a 1784 ms prefill pass** - 59% of it
/// which works out at 2.4 TFLOP/s. The vendor library on the same shapes
/// measures 12.5 TFLOP/s in plain single precision and 25.9 with the
/// reduced-mantissa tensor path this backend enables by default
/// (`CudaOptions::tf32`), which is the one that actually runs. End to end
/// that is prefill at 512 tokens going from 296.65 to 597.64 tok/s;
/// `docs/handoff/cuda-decode-perf.md` section 18 carries the tables.
///
/// # What this route does and does not narrow
///
/// It is **not** the more precise of the two. The tile stages both operands as
/// `f16`; this route hands the library f32 and the library, with TF32 on,
/// truncates both operands to a ten-bit mantissa - the same mantissa width
/// `f16` carries. The two therefore round their inputs to the same precision
/// and differ in exponent range (TF32 keeps f32's eight bits, `f16` has five)
/// and in the order they reduce. The disagreement `PREFILL_GEMM_BUDGET`
/// records is that, and it is not attributable to either side.
///
/// # Why there is a staging pass at all
///
/// The vendor library has no quantized operand type, so the weight is widened
/// into f32 by `kernel_dequant_plane_*` first. That costs its own memory
/// traffic - one read of the quantized plane and one write of a plane eight
/// times larger - and it is still the faster route by a factor of several,
/// because what it replaces is not a memory-bound kernel but a compute-bound
/// one running at a twelfth of the device's arithmetic.
///
/// The staged plane is written row-major, `out_dim` rows of `in_dim` f32,
/// which is how the weight is already stored. Read column-major - the only way
/// the vendor library reads anything - that is an `in_dim` by `out_dim` matrix
/// with a leading dimension of `in_dim`, so it is the **transposed** left
/// operand and no repacking is needed. The activations are `n_tok` rows of
/// `in_dim`, which column-major is `in_dim` by `n_tok` with the same leading
/// dimension; the output is `n_tok` rows of `out_dim`, which is `out_dim` by
/// `n_tok` with a leading dimension of `out_dim`. All three fall out of the
/// layouts this engine already uses.
///
/// # Why it returns a `bool` rather than refusing
///
/// Every reason this path is unavailable is a reason to run the tile instead,
/// not a reason to fail: a backend with no matrix tier at all (every Metal
/// build, where `as_blas()` is `None`), a codec with no staging entry, a
/// caller whose buffers are cut to the chunk, and a device that will not give
/// up the staging slab all answer the same way. A refusal here would turn a
/// device that has always prefilled into one that cannot.
fn blas_projection(
    backend: &dyn Backend,
    weight: &Weight,
    n_tok: u32,
    src: DeviceView<'_>,
    dst: DeviceView<'_>,
) -> CoreResult<bool> {
    let (Some(blas), Some(entry)) = (backend.as_blas(), dequant_plane_name(weight.dtype)) else {
        return Ok(false);
    };
    // Every call is made at ONE token count, whatever the chunk carries, and
    // that is a correctness requirement rather than a tuning choice. The
    // vendor library picks an algorithm per problem shape, and a narrow
    // problem gets one that splits the reduction across blocks - so the order
    // the products are summed in would depend on how many tokens are in the
    // chunk. This engine's contract is the opposite: `parity.rs`'s
    // `the_answer_does_not_depend_on_how_the_prompt_was_chunked` derives which
    // chunk widths must agree *to the bit* and asserts it, and calling the
    // library at the chunk's own width made 248,316 of 248,320 logits differ
    // between widths 512 and 384. Padding to the cap costs a short prompt some
    // arithmetic it does not need, and a chunk-dependent answer would cost
    // the property that gate exists to hold.
    let pad = PREFILL_BLAS_PAD;
    // A chunk wider than the cap is not something this engine produces - the
    // stack slices at the cap - but [`projection`] is public, and the bands
    // below would run past the scratch the cap sized.
    if n_tok > crate::MAX_PREFILL_CHUNK {
        return Ok(false);
    }
    // **The eligibility threshold is a constant, not a function of `n_tok`,
    // and that is the whole point of it.** It was `blocks * pad` rows, which
    // moved with the chunk: a caller handing in buffers cut to its own chunk
    // then took this route at 256 tokens and the tile at 255 and 257, so the
    // fallback reintroduced exactly the width-dependence the banding above
    // exists to remove. Asking for [`crate::MAX_PREFILL_CHUNK`] rows makes it
    // a question about the caller's buffers instead.
    //
    // `crate::layers::Scratch` is sized for the cap, so the hybrid stack
    // always passes. `crate::draft`'s head is the caller that does not: it
    // slices its pair buffer to `2 * hidden * n_tok` and runs at most
    // `MAX_PREFILL_CHUNK - 1` rows, so it now takes the tile at every width it
    // can reach. A caller that *did* reach the cap with a chunk-sized buffer
    // would be handing in a cap-sized buffer, which is the operand this route
    // wants - but its answers would then differ between a cap-width chunk and
    // a narrower one, so a caller for which that matters sizes its buffers to
    // the cap and not to the chunk.
    let eligible_src = u64::from(weight.in_dim)
        .saturating_mul(u64::from(crate::MAX_PREFILL_CHUNK))
        .saturating_mul(F32_BYTES);
    let eligible_dst = u64::from(weight.out_dim)
        .saturating_mul(u64::from(crate::MAX_PREFILL_CHUNK))
        .saturating_mul(F32_BYTES);
    if src.len() < eligible_src || dst.len() < eligible_dst {
        return Ok(false);
    }
    // The staging kernel walks a row in four-element chunks, so a width that
    // is not a whole number of them would leave the row's last one to three
    // elements holding whatever the previous tensor left in the slab. No
    // codec here can have one - `q4_0` and `q8_0` pack 32 elements to a block
    // and `q4_K` packs 256, and `row_bytes` above refuses a width that is not
    // a whole number of blocks - but `f16` has no block at all, so this is
    // the check that stands in for one.
    if !weight.in_dim.is_multiple_of(4) {
        return Ok(false);
    }
    let blocks = n_tok.div_ceil(pad).max(1);
    let plane = u64::from(weight.in_dim)
        .saturating_mul(u64::from(weight.out_dim))
        .saturating_mul(F32_BYTES);
    let Some(staged) = blas.staging(plane)? else {
        return Ok(false);
    };
    // The slab is grown rather than sized, so it is normally *longer* than
    // this plane. Narrowing it here is not tidiness: `blas.rs` checks every
    // operand against the shape it declares, and a view longer than its shape
    // would pass a check that is there to catch the operand that is too short.
    let staged = staged.slice(0, plane)?;
    let carried = u64::from(n_tok)
        .saturating_mul(u64::from(weight.in_dim))
        .saturating_mul(F32_BYTES);
    // Only the rows the bands actually read, which is `blocks * pad` and not
    // the whole cap: zeroing rows no multiply touches would be a megabyte of
    // writes per tensor that changes nothing.
    let banded_src = u64::from(weight.in_dim)
        .saturating_mul(u64::from(blocks).saturating_mul(u64::from(pad)))
        .saturating_mul(F32_BYTES);
    // The rows past the chunk are whatever the previous tensor left there.
    // Zeroed rather than ignored: the multiply reads them, and a leftover
    // infinity in a row nothing reads still produces a NaN in a row nothing
    // reads - until a later layer reduces across the padded region and it
    // stops being nothing.
    backend.fill_zero(src.slice(carried, banded_src.saturating_sub(carried))?)?;
    let args = DequantPlaneArgs {
        ne00: extent(weight.in_dim, "projection in_dim")?,
        ne01: extent(weight.out_dim, "projection out_dim")?,
        nb01: row_bytes(weight.dtype, weight.in_dim)?,
    };
    run(
        backend,
        KernelRef::new(Q35, entry),
        dequant_plane_grid(weight.out_dim),
        Extent3::linear(DEQUANT_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(weight.view()),
            KernelArg::Buffer(staged),
        ],
    )
    .map_err(|err| launch_failure(entry, err))?;
    // One multiply per band, all of them the same shape. The plane is staged
    // once and every band reads it, so a narrower pad trades a second read of
    // the staged weight for less padding arithmetic - which is the whole of
    // why [`PREFILL_BLAS_PAD`] is the number it is.
    let src_band = u64::from(weight.in_dim)
        .saturating_mul(u64::from(pad))
        .saturating_mul(F32_BYTES);
    let dst_band = u64::from(weight.out_dim)
        .saturating_mul(u64::from(pad))
        .saturating_mul(F32_BYTES);
    for band in 0..u64::from(blocks) {
        blas.gemm(&GemmDesc {
            op_a: GemmOp::Transposed,
            op_b: GemmOp::AsStored,
            m: weight.out_dim,
            n: pad,
            k: weight.in_dim,
            lda: weight.in_dim,
            ldb: weight.in_dim,
            ldc: weight.out_dim,
            a: staged,
            b: src.slice(band.saturating_mul(src_band), src_band)?,
            c: dst.slice(band.saturating_mul(dst_band), dst_band)?,
            // This projection replaces its destination rather than
            // accumulating into it: a nonzero second scale would add this
            // band's product onto whatever the previous tensor left behind.
            alpha: 1.0,
            beta: 0.0,
            dtype: GemmDType::F32,
            reduction: GemmReduction::Fastest,
        })?;
    }
    Ok(true)
}

/// The tiled-GEMM half of [`projection`]: the same product, read once.
///
/// The weight is the first operand and the activations the second, both one plane
/// deep, so the two plane strides repeat the whole-operand size and both broadcast
/// ratios are one. The output is `n_tok` rows of `out_dim` f32, which is the same
/// destination layout the matvec families write.
fn mm_projection(
    backend: &dyn Backend,
    weight: &Weight,
    n_tok: u32,
    src: DeviceView<'_>,
    dst: DeviceView<'_>,
) -> CoreResult<()> {
    if blas_projection(backend, weight, n_tok, src, dst)? {
        return Ok(());
    }
    let entry = mm_name(weight.dtype).ok_or(CoreError::InvalidValue {
        key: "projection dtype",
        value: format!("{:?} has no tiled product", weight.dtype),
    })?;
    let row = row_bytes(weight.dtype, weight.in_dim)?;
    let plane = row.saturating_mul(u64::from(weight.out_dim));
    let act_row = u64::from(weight.in_dim).saturating_mul(F32_BYTES);
    let act_plane = act_row.saturating_mul(u64::from(n_tok));
    let args = MulMmArgs {
        ne00: extent(weight.in_dim, "projection in_dim")?,
        ne02: 1,
        nb01: row,
        nb02: plane,
        nb03: plane,
        ne12: 1,
        pad0: 0,
        nb10: F32_BYTES,
        nb11: act_row,
        nb12: act_plane,
        nb13: act_plane,
        ne0: extent(weight.out_dim, "projection out_dim")?,
        ne1: extent(n_tok, "projection n_tok")?,
        r2: 1,
        r3: 1,
        pad2: [0; 4],
    };
    run(
        backend,
        KernelRef::specialized(Q35, entry, mm_consts_for(backend.specializes_by_name())),
        mm_grid(weight.out_dim, n_tok),
        Extent3::linear(GEMM_BLOCK),
        GEMM_SMEM_BYTES,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(weight.view()),
            KernelArg::Buffer(src),
            KernelArg::Buffer(dst),
        ],
    )
    .map_err(|err| launch_failure(entry, err))
}

/// The plain-matvec half of [`projection`]: the same product, for the formats and
/// widths the extended family has no instantiation at.
///
/// The geometry differs and the result does not. `x` divides the output rows by the
/// instantiation's own row count, `y` carries the token and `z` the batch, which is
/// one here: a decode step is one session's token, and a wider batch is a separate
/// launch rather than a taller grid.
fn plain_projection(
    backend: &dyn Backend,
    weight: &Weight,
    n_tok: u32,
    src: DeviceView<'_>,
    dst: DeviceView<'_>,
) -> CoreResult<()> {
    let mv = plain_mv(weight.dtype, u64::from(weight.in_dim)).ok_or(CoreError::InvalidValue {
        key: "projection dtype",
        value: format!("{:?} has no plain matvec at this width", weight.dtype),
    })?;
    let consts =
        plain_consts_for(mv.nsg, backend.specializes_by_name()).ok_or(CoreError::InvalidValue {
            key: "projection simdgroups",
            value: format!("{} is past the plain family's simdgroup cap", mv.nsg),
        })?;
    let args = plain_args(weight.dtype, weight.in_dim, weight.out_dim, n_tok, &mv)?;
    run(
        backend,
        KernelRef::specialized(GENERIC, mv.name, consts),
        plain_mv_grid(weight.out_dim, mv.rows_per_threadgroup, n_tok, 1),
        plain_block(mv.nsg),
        mv.smem_bytes,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(weight.view()),
            KernelArg::Buffer(src),
            KernelArg::Buffer(dst),
        ],
    )
}

/// Activation columns one threadgroup of the affine matrix entries carries.
///
/// `DENSE_AFFINE_TOKENS` in `metal/dense/dense_affine.metal`, which is where the
/// number is chosen; a launch that disagreed would leave a band of the chunk
/// unwritten.
pub const AFFINE_GEMM_TOKENS: u32 = 8;

/// Threads one affine threadgroup runs: one simdgroup, whose reduction the
/// entries close with.
const AFFINE_SIMD_WIDTH: u32 = 32;

/// The refusal for a packed affine format with no entry point at its width.
const fn unsupported_bits() -> CoreError {
    CoreError::Unsupported {
        what: "affine bit width",
        backend: "metal",
    }
}

/// The widths and the one scale a `qwen35moe` projection is launched over.
///
/// Sixteen bytes, one struct, and the entries declare it as one parameter -
/// `.param .align 4 .b8 ..._param_0[16]` in the generated PTX. It is bound as
/// bytes for that reason: `crates/shiso-cuda/src/launch.rs` hands the driver
/// one pointer per DECLARED parameter, so four scalar operands would give an
/// entry a four-byte first parameter where it declares sixteen and every
/// launch would succeed, reading a pointer out of a slot holding a quarter of
/// a struct. `HANDOFF.md` records that costing a session once.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModeloptArgs {
    /// Elements in one weight row, counted in weights and not in bytes.
    in_dim: i32,
    /// Weight rows.
    out_dim: i32,
    /// Activation rows this launch covers.
    n_tok: i32,
    /// The tensor's own scale: FP8's `weight_scale`, NVFP4's `weight_scale_2`.
    scale: f32,
}

/// Whether the two `modelopt` codecs can be launched at this width.
///
/// FP8 has no constraint at all - one byte is one weight - and NVFP4 needs a
/// whole number of sixteen-weight groups, because the entries derive the
/// packed width and the scale width from `in_dim` and a row that is not a
/// whole number of groups would have its tail read out of the row below.
fn modelopt_dispatchable(dtype: DType, in_dim: u32) -> CoreResult<()> {
    let group = dtype.block_elems().get();
    if in_dim == 0 || !in_dim.is_multiple_of(group) {
        return Err(CoreError::InvalidValue {
            key: "modelopt projection in_dim",
            value: format!(
                "{in_dim} is not a whole number of {group}-weight groups for {}",
                dtype.type_name()
            ),
        });
    }
    Ok(())
}

/// Runs an FP8 or NVFP4 projection, matvec or tiled, on the row count.
///
/// The same threshold every other codec here takes, and for the same reason: a
/// matvec re-reads the whole weight matrix once per token, so a prefill chunk
/// through it reads a 23.4 GB checkpoint once per row.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] when the weight is missing the scales its own
/// format needs, and [`CoreError::Backend`] naming the entry when a launch
/// fails.
fn modelopt_projection(
    backend: &dyn Backend,
    weight: &Weight,
    n_tok: u32,
    src: DeviceView<'_>,
    dst: DeviceView<'_>,
) -> CoreResult<()> {
    let tiled = n_tok >= GEMM_MIN_TOKENS;
    let mut args = ModeloptArgs {
        in_dim: extent(weight.in_dim, "projection in_dim")?,
        out_dim: extent(weight.out_dim, "projection out_dim")?,
        n_tok: extent(n_tok, "projection n_tok")?,
        scale: 1.0,
    };
    // The two codecs take DIFFERENT matvec geometries, so the grid cannot be
    // decided before the aux is read. See `MODELOPT_MV_BLOCK` for why: at four
    // bits a weight a row is half as many bytes, and the block-per-row shape
    // the FP8 entry wants leaves the NVFP4 one folding more than it loads. The
    // tiled route is one shape for both - it is a GEMM over decoded floats and
    // the codec is gone by the time the tile is staged.
    let nvfp4 = matches!(weight.aux.as_ref(), Some(WeightAux::Nvfp4 { .. }));
    let grid = if tiled {
        Extent3::new(
            weight.out_dim.div_ceil(MODELOPT_TILE),
            n_tok.div_ceil(MODELOPT_TILE),
            1,
        )
    } else if nvfp4 {
        Extent3::new(weight.out_dim.div_ceil(MODELOPT_MV_ROWS), n_tok, 1)
    } else {
        Extent3::new(weight.out_dim, n_tok, 1)
    };
    let block = if tiled {
        Extent3::new(MODELOPT_TILE, MODELOPT_TILE, 1)
    } else if nvfp4 {
        Extent3::linear(MODELOPT_NVFP4_MV_BLOCK)
    } else {
        Extent3::linear(MODELOPT_MV_BLOCK)
    };
    match weight.aux.as_ref() {
        Some(WeightAux::Fp8 { weight_scale, .. }) => {
            args.scale = *weight_scale;
            let entry = if tiled {
                "kernel_q35moe_mm_fp8_f32"
            } else {
                "kernel_q35moe_mv_fp8_f32"
            };
            run(
                backend,
                KernelRef::new(Q35MOE, entry),
                grid,
                block,
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(weight.view()),
                    KernelArg::Buffer(src),
                    KernelArg::Buffer(dst),
                ],
            )
        }
        Some(WeightAux::Nvfp4 {
            group_scales,
            global,
        }) => {
            args.scale = *global;
            let entry = if tiled {
                "kernel_q35moe_mm_nvfp4_f32"
            } else {
                "kernel_q35moe_mv_nvfp4_f32"
            };
            run(
                backend,
                KernelRef::new(Q35MOE, entry),
                grid,
                block,
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(weight.view()),
                    KernelArg::Buffer(group_scales.view()),
                    KernelArg::Buffer(src),
                    KernelArg::Buffer(dst),
                ],
            )
        }
        // A weight of one of these formats with no scales is a loader that
        // built it through the wrong constructor, and its answer would be the
        // matrix divided by a number around 3e-5. Refused rather than
        // defaulted.
        Some(WeightAux::Affine(_)) | None => Err(CoreError::InvalidValue {
            key: "modelopt aux",
            value: format!(
                "a {} weight carrying no scale of its own",
                weight.dtype.type_name()
            ),
        }),
    }
}

/// Multiplies a packed affine weight by an activation block.
fn affine_projection(
    backend: &dyn Backend,
    weight: &Weight,
    n_tok: u32,
    src: DeviceView<'_>,
    dst: DeviceView<'_>,
) -> CoreResult<()> {
    let aux = weight
        .aux
        .as_ref()
        .and_then(WeightAux::affine)
        .ok_or_else(|| CoreError::InvalidValue {
            key: "affine aux",
            value: "affine weight is missing auxiliary scales/biases".into(),
        })?;
    let bits = weight.dtype.affine_bits().ok_or_else(unsupported_bits)?;
    let entry = match (bits, n_tok) {
        (4, 1) => "kernel_dense_affine_qmv_4_f32",
        (8, 1) => "kernel_dense_affine_qmv_8_f32",
        (4, _) => "kernel_dense_affine_qmm_4_f32",
        (8, _) => "kernel_dense_affine_qmm_8_f32",
        _ => return Err(unsupported_bits()),
    };
    let args = DenseAffineArgs {
        in_dim: weight.in_dim,
        out_dim: weight.out_dim,
        n_tok,
        bits,
    };
    // One threadgroup a row for a single token; for a chunk, one threadgroup per
    // row per band of AFFINE_GEMM_TOKENS columns, because that is how many the
    // matrix entry carries with one fetch of the packed row.
    let grid = if n_tok == 1 {
        Extent3::linear(weight.out_dim)
    } else {
        Extent3::new(weight.out_dim, n_tok.div_ceil(AFFINE_GEMM_TOKENS), 1)
    };
    let block = Extent3::linear(AFFINE_SIMD_WIDTH);
    run(
        backend,
        KernelRef::new(crate::launch::DENSE, entry),
        grid,
        block,
        0,
        &[
            KernelArg::Buffer(weight.view()),
            KernelArg::Buffer(aux.scales.view()),
            KernelArg::Buffer(aux.biases.view()),
            KernelArg::Buffer(src),
            KernelArg::Buffer(dst),
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
        ],
    )
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;

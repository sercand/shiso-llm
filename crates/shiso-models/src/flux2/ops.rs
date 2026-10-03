//! The flux2 kernel driver: every device operation the pipeline issues.
//!
//! One funnel, like the vision encoder's `Run` and the audio family's `Ops`:
//! argument blocks are `#[repr(C)]` `Pod` structs marshalled with
//! `bytemuck::bytes_of`, so their layout comes from the struct definition and
//! is asserted against the kernel headers at compile time. Every launch names
//! the `flux2` module; a backend whose module table does not carry the flux2
//! entry points runs the same operations through the host fallback in this
//! file - slower by orders of magnitude, numerically the same arithmetic, and
//! the reason a build without the Metal kernels still produces an image
//! instead of an error.

#![expect(
    clippy::cast_possible_truncation,
    reason = "kernel argument dimensions are element counts bounded by the generation contract (images up to 2048x2048); they fit u32 with orders of magnitude to spare"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "dimensions converted for trig and grid math are small tensor extents, far below f32's exact-integer range"
)]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use half::bf16;
use shiso_core::{
    Backend, DeviceBuffer, DeviceView, Extent3, KernelArg, KernelRef, LaunchArgs, alloc,
};
use shiso_vision::generator::ImageGenError;

use super::weights::f8e4m3_to_f32;

/// The kernel module every flux2 entry point lives under.
pub(crate) const MODULE: &str = "flux2";

/// Threads per elementwise block.
const BLOCK: u32 = 64;
/// Tile edge of the GEMM kernels.
const GEMM_TILE: u32 = 16;

fn grid_for(total: u32, tile: u32) -> u32 {
    total.div_ceil(tile).max(1)
}

fn fail(step: &'static str) -> impl Fn(shiso_core::CoreError) -> ImageGenError {
    move |source| ImageGenError::Backend(format!("{step}: {source}"))
}

/// `DeviceView::slice` answers a core error; the family's drivers speak
/// [`ImageGenError`], so every sub-range goes through this instead.
pub(crate) trait SliceExt<'a> {
    /// Borrows `[offset, offset + length)` bytes, in family error terms.
    fn cut(&self, offset: u64, length: u64) -> Result<DeviceView<'a>, ImageGenError>;
}

impl<'a> SliceExt<'a> for DeviceView<'a> {
    fn cut(&self, offset: u64, length: u64) -> Result<DeviceView<'a>, ImageGenError> {
        self.slice(offset, length)
            .map_err(|e| ImageGenError::Backend(format!("slicing device range: {e}")))
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GemmArgs {
    m: u32,
    n: u32,
    k: u32,
    // 1 when the bias binding is a real vector, 0 when the dummy stands in.
    // MSL has no device-side query for a bound buffer's length, so absence
    // must be a flag: a value sentinel (bit-exact +0.0 first word) would
    // silently drop a real bias whose first element is exactly +0.0.
    has_bias: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UnaryArgs {
    count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SwigluArgs {
    rows: u32,
    inner: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ModulateArgs {
    count: u32,
    period: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AddGatedArgs {
    count: u32,
    period: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UpsampleArgs {
    batch: u32,
    channels: u32,
    h: u32,
    w: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct NormArgs {
    rows: u32,
    dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct QkNormArgs {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    /// Elements per fused qkv row; q and k are head sections of it.
    row_stride: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ConcatArgs {
    rows: u32,
    cols_a: u32,
    cols_b: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GroupNormArgs {
    batch: u32,
    num_groups: u32,
    channels_per_group: u32,
    spatial: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Rope4dArgs {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    freqs_per_axis: u32,
    /// Elements per token row: packed tensors pass `heads * head_dim`, a fused
    /// qkv row passes its full width, so `RoPE` runs in place on fused rows.
    row_stride: u32,
    /// Where the q and k head sections start inside a row (elements).
    q_offset: u32,
    k_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RopeHalfArgs {
    tokens: u32,
    heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SdpaArgs {
    txt_tokens: u32,
    img_tokens: u32,
    cache_tokens: u32,
    heads: u32,
    head_dim: u32,
    txt_stride: u32,
    img_stride: u32,
    /// Elements per cached row (a cached fused-qkv row is wider than a head).
    cache_stride: u32,
    /// Where the q/k/v head sections start inside a row (elements); zero for
    /// packed (non-fused) bindings.
    q_offset: u32,
    k_offset: u32,
    v_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GqaArgs {
    tokens: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Conv2dArgs {
    batch: u32,
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    k_h: u32,
    k_w: u32,
    stride: u32,
    pad: u32,
    // Presence flag, as in `GemmArgs`: the flag, not the bias values,
    // tells the kernel whether the binding is real.
    has_bias: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AxpyArgs {
    count: u32,
    dt: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct EmbedArgs {
    tokens: u32,
    hidden: u32,
}

// The argument blocks and the kernel headers are edited together; a drift is
// a wrong number under load, so it is a compile error here instead.
const _: () = assert!(size_of::<GemmArgs>() == 16);
const _: () = assert!(size_of::<UnaryArgs>() == 4);
const _: () = assert!(size_of::<SwigluArgs>() == 12);
const _: () = assert!(size_of::<ModulateArgs>() == 8);
const _: () = assert!(size_of::<AddGatedArgs>() == 8);
const _: () = assert!(size_of::<UpsampleArgs>() == 16);
const _: () = assert!(size_of::<NormArgs>() == 12);
const _: () = assert!(size_of::<QkNormArgs>() == 20);
const _: () = assert!(size_of::<ConcatArgs>() == 12);
const _: () = assert!(size_of::<GroupNormArgs>() == 20);
const _: () = assert!(size_of::<Rope4dArgs>() == 28);
const _: () = assert!(size_of::<RopeHalfArgs>() == 12);
const _: () = assert!(size_of::<SdpaArgs>() == 44);
const _: () = assert!(size_of::<GqaArgs>() == 16);
const _: () = assert!(size_of::<Conv2dArgs>() == 48);
const _: () = assert!(size_of::<AxpyArgs>() == 8);
const _: () = assert!(size_of::<EmbedArgs>() == 8);

/// The stored format of a linear weight, taken from the tensor header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WeightKind {
    /// Brain float, executed natively.
    Bf16,
    /// FP8 E4M3, executed natively with the per-tensor scale in the epilogue.
    Fp8,
    /// IEEE single, executed natively.
    F32,
}

/// One matrix multiply against a linear weight stored in its native format.
pub(crate) struct GemmCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub input: DeviceView<'a>,
    /// BF16, `F8_E4M3` or F32 weight rows, (n, k), exactly as the checkpoint stored them.
    pub weight: DeviceView<'a>,
    /// Multiplier applied to the accumulated dot product: the FP8 tensor's
    /// `weight_scale`, or 1.0 otherwise. It rides the epilogue, so the weight
    /// is never rescaled in memory.
    pub weight_scale: f32,
    pub bias: Option<DeviceView<'a>>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub kind: WeightKind,
}

/// Joint-sequence scaled dot-product attention (the transformer's blocks).
///
/// Fused-row contract (Task 3): every q/k/v binding is a buffer of token rows
/// `stride` elements wide, and `q_offset`/`k_offset`/`v_offset` give where the
/// q, k and v head sections start inside a row. The transformer passes one
/// fused qkv view three times with offsets `0`, `heads * head_dim` and
/// `2 * heads * head_dim`; the VAE passes separate packed tensors with all
/// offsets zero. Head `h` of a row starts at `row * stride + offset +
/// h * head_dim`. The cached rows carry the same layout as the live rows.
pub(crate) struct SdpaCall<'a> {
    pub step: &'static str,
    pub out_txt: DeviceView<'a>,
    pub out_img: DeviceView<'a>,
    pub q_txt: DeviceView<'a>,
    pub k_txt: DeviceView<'a>,
    pub v_txt: DeviceView<'a>,
    pub q_img: DeviceView<'a>,
    pub k_img: DeviceView<'a>,
    pub v_img: DeviceView<'a>,
    /// Where the q/k/v head sections start inside a row (elements).
    pub q_offset: usize,
    pub k_offset: usize,
    pub v_offset: usize,
    /// Cached reference-token keys/values, attended between the txt and img
    /// rows (the reference's `[txt, cached_ref, img]` key order);
    /// `cache_tokens == 0` binds the dummy. Softmax is order-invariant, so
    /// the host twin scores cache rows last without changing the result.
    pub cache_k: DeviceView<'a>,
    pub cache_v: DeviceView<'a>,
    /// Elements per cached row (a cached fused-qkv row is wider than a head).
    pub cache_stride: usize,
    pub txt_tokens: usize,
    pub img_tokens: usize,
    pub cache_tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// Elements per token row in the strided q/k/v buffers (a fused qkv row).
    pub txt_stride: usize,
    pub img_stride: usize,
}

/// Grouped-query causal attention (the text encoder's prefill).
pub(crate) struct GqaCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub q: DeviceView<'a>,
    pub k: DeviceView<'a>,
    pub v: DeviceView<'a>,
    pub tokens: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

/// One 2D convolution of the VAE decoder.
pub(crate) struct Conv2dCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub input: DeviceView<'a>,
    pub weight: DeviceView<'a>,
    pub bias: Option<DeviceView<'a>>,
    pub batch: usize,
    pub in_c: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_c: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub k_h: usize,
    pub k_w: usize,
    pub stride: usize,
    pub pad: usize,
}

/// Group normalization of the VAE decoder.
pub(crate) struct GroupNormCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub input: DeviceView<'a>,
    pub weight: DeviceView<'a>,
    pub bias: DeviceView<'a>,
    pub batch: usize,
    pub groups: usize,
    pub channels_per_group: usize,
    pub spatial: usize,
    pub eps: f32,
}

/// RMS normalization over rows of `dim`.
pub(crate) struct NormCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub input: DeviceView<'a>,
    pub weight: DeviceView<'a>,
    pub rows: usize,
    pub dim: usize,
    pub eps: f32,
}

/// Weightless `LayerNorm` over rows of `dim`: mean-subtracting, with no
/// affine parameters (the transformer blocks' `elementwise_affine=False`
/// norms, which are *not* the `RMSNorm` the text encoder uses).
pub(crate) struct LayerNormCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub input: DeviceView<'a>,
    pub rows: usize,
    pub dim: usize,
    pub eps: f32,
}

/// QK-norm: RMS normalization with a per-dimension scale over the q and k
/// head sections of fused qkv rows, applied in place before `RoPE`.
pub(crate) struct QkNormCall<'a> {
    pub step: &'static str,
    /// Fused qkv rows; q occupies `[0, heads * head_dim)`, k the next
    /// `heads * head_dim` elements of the row.
    pub qkv: DeviceView<'a>,
    pub q_scale: DeviceView<'a>,
    pub k_scale: DeviceView<'a>,
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub row_stride: usize,
    pub eps: f32,
}

/// Row-wise concatenation: `out[r] = [a[r], b[r]]` (the single block's
/// `linear2` input, the reference's `cat(attn, mlp_act)`).
pub(crate) struct ConcatCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub a: DeviceView<'a>,
    pub b: DeviceView<'a>,
    pub rows: usize,
    pub cols_a: usize,
    pub cols_b: usize,
}

/// `AdaLN` modulation: `out = x * (1 + scale) + shift`, with the per-dimension
/// `scale`/`shift` vectors of `period` elements broadcast over the rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModulateCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub x: DeviceView<'a>,
    pub scale: DeviceView<'a>,
    pub shift: DeviceView<'a>,
    pub count: usize,
    pub period: usize,
}

/// Gated residual add: `out = a + gate * b`, with `gate` a per-dimension
/// vector of `period` elements broadcast over the rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AddGatedCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub a: DeviceView<'a>,
    pub gate: DeviceView<'a>,
    pub b: DeviceView<'a>,
    pub count: usize,
    pub period: usize,
}

/// Nearest-neighbour 2x spatial upsampling of an NCHW f32 field.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UpsampleCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub input: DeviceView<'a>,
    pub batch: usize,
    pub channels: usize,
    pub h: usize,
    pub w: usize,
}

/// 4-axis interleaved `RoPE` applied to Q and K in one launch.
///
/// Fused-row contract (Task 3): the q and k bindings are token-row buffers
/// `row_stride` elements wide, and `q_offset`/`k_offset` give where the q and
/// k head sections start inside a row. The transformer binds the same fused
/// qkv buffer twice with offsets `0` and `heads * head_dim`, so one launch
/// rotates each section exactly once and the v section (which no offset
/// points at) is never touched; packed separate q/k tensors pass zero for
/// both. Head `h` of a row starts at `row * row_stride + offset +
/// h * head_dim`.
pub(crate) struct Rope4dCall<'a> {
    pub step: &'static str,
    pub q: DeviceView<'a>,
    pub k: DeviceView<'a>,
    /// (tokens, 4) axis positions as f32.
    pub positions: DeviceView<'a>,
    pub freqs: DeviceView<'a>,
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub freqs_per_axis: usize,
    /// Elements per token row of the q/k buffers.
    pub row_stride: usize,
    /// Where the q and k head sections start inside a row (elements).
    pub q_offset: usize,
    pub k_offset: usize,
}

/// Embedding gather: `tokens` rows of a BF16 table, widened to f32 on write.
pub(crate) struct EmbedCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub table: DeviceView<'a>,
    pub ids: DeviceView<'a>,
    pub tokens: usize,
    pub hidden: usize,
}

/// The driver every flux2 component shares.
#[derive(Debug)]
pub(crate) struct Ops {
    backend: Arc<dyn Backend>,
    /// Set when the backend's module table does not carry the flux2 entries.
    host_fallback: bool,
    /// A zero-filled 4-byte buffer bound wherever an optional bias is absent,
    /// so every launch of an entry point binds the same buffer count.
    dummy: DeviceBuffer,
}

impl Ops {
    /// Wraps `backend`, probing its module table once for the flux2 entries.
    pub(crate) fn new(backend: &Arc<dyn Backend>) -> Result<Self, ImageGenError> {
        let entries = backend.module_entries(MODULE);
        let host_fallback = !entries
            .iter()
            .any(|name| name.contains("kernel_flux2_gemm_bf16"));
        let dummy = alloc(backend, 4).map_err(|e| ImageGenError::Backend(e.to_string()))?;
        backend
            .fill_zero(dummy.view())
            .map_err(|e| ImageGenError::Backend(e.to_string()))?;
        Ok(Self {
            backend: Arc::clone(backend),
            host_fallback,
            dummy,
        })
    }

    /// The wrapped device, for direct copies and imports.
    pub(crate) fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// A zero-filled 4-byte view, bound wherever an optional operand is absent.
    pub(crate) fn dummy_view(&self) -> DeviceView<'_> {
        self.dummy.view()
    }

    /// Allocates an uninitialized f32 buffer of `count` elements.
    pub(crate) fn alloc_f32(
        &self,
        step: &'static str,
        count: usize,
    ) -> Result<DeviceBuffer, ImageGenError> {
        alloc(&self.backend, u64::try_from(count).unwrap_or(u64::MAX) * 4).map_err(fail(step))
    }

    /// Uploads f32 values.
    pub(crate) fn upload_f32(
        &self,
        step: &'static str,
        values: &[f32],
    ) -> Result<DeviceBuffer, ImageGenError> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = self.alloc_f32(step, values.len())?;
        self.backend
            .copy_h2d(buffer.view(), &bytes)
            .map_err(fail(step))?;
        Ok(buffer)
    }

    /// Uploads token ids.
    pub(crate) fn upload_u32(
        &self,
        step: &'static str,
        values: &[u32],
    ) -> Result<DeviceBuffer, ImageGenError> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = alloc(&self.backend, bytes.len() as u64).map_err(fail(step))?;
        self.backend
            .copy_h2d(buffer.view(), &bytes)
            .map_err(fail(step))?;
        Ok(buffer)
    }

    /// Reads back `count` f32 elements.
    pub(crate) fn download(
        &self,
        step: &'static str,
        view: DeviceView<'_>,
        count: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        let mut bytes = vec![0u8; count * 4];
        self.backend
            .copy_d2h(view, &mut bytes)
            .map_err(fail(step))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect())
    }

    /// Device-to-device contiguous copy.
    pub(crate) fn copy(
        &self,
        step: &'static str,
        dst: DeviceView<'_>,
        src: DeviceView<'_>,
    ) -> Result<(), ImageGenError> {
        self.backend.copy_d2d(dst, src).map_err(fail(step))
    }

    fn launch(
        &self,
        step: &'static str,
        entry: &'static str,
        grid: Extent3,
        block: Extent3,
        args: &[KernelArg<'_>],
    ) -> Result<(), ImageGenError> {
        self.backend
            .run(
                KernelRef::new(MODULE, entry),
                &LaunchArgs {
                    grid,
                    block,
                    smem_bytes: 0,
                    args,
                },
            )
            .map_err(fail(step))
    }

    fn elementwise(
        &self,
        step: &'static str,
        entry: &'static str,
        count: usize,
        buffers: &[DeviceView<'_>],
    ) -> Result<(), ImageGenError> {
        let count = count as u32;
        let args = UnaryArgs { count };
        let mut kernel_args: Vec<KernelArg<'_>> = vec![KernelArg::Bytes(bytemuck::bytes_of(&args))];
        kernel_args.extend(buffers.iter().copied().map(KernelArg::Buffer));
        self.launch(
            step,
            entry,
            Extent3::linear(grid_for(count, BLOCK)),
            Extent3::linear(BLOCK),
            &kernel_args,
        )
    }

    /// out[i] = silu(input[i]).
    pub(crate) fn silu(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        count: usize,
    ) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let x = self.download(step, input, count)?;
            let y: Vec<f32> = x.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
            return self.write_back(step, out, &y);
        }
        self.elementwise(step, "kernel_flux2_silu_f32", count, &[out, input])
    }

    /// `SwiGLU` over a fused row: out[r, i] = silu(in[r, i]) * in[r, inner + i].
    ///
    /// The reference's activation gates with the *first* half of the row
    /// (`SiLUActivation` chunks in two and applies `silu` to the first
    /// chunk), so the double-block MLP and the single-block `linear1` both
    /// emit (gate, value) in that order. The kernel reads one strided buffer
    /// (the single block's halves start mid-row) and writes a packed
    /// (rows, inner) result.
    pub(crate) fn swiglu(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        rows: usize,
        inner: usize,
        stride: usize,
    ) -> Result<(), ImageGenError> {
        if self.host_fallback {
            // Reads span `[(rows - 1) * stride, (rows - 1) * stride + 2 * inner)`
            // of the view; the single block's view starts mid-row, so a
            // `rows * stride` download would over-read past the buffer end.
            let need = if rows == 0 {
                0
            } else {
                (rows - 1) * stride + 2 * inner
            };
            let x = self.download(step, input, need)?;
            let mut y = Vec::with_capacity(rows * inner);
            for r in 0..rows {
                for i in 0..inner {
                    let g = x[r * stride + i];
                    let v = x[r * stride + inner + i];
                    y.push((g / (1.0 + (-g).exp())) * v);
                }
            }
            return self.write_back(step, out, &y);
        }
        let args = SwigluArgs {
            rows: rows as u32,
            inner: inner as u32,
            stride: stride as u32,
        };
        self.launch(
            step,
            "kernel_flux2_swiglu_f32",
            Extent3::new(inner as u32, rows as u32, 1),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(input),
            ],
        )
    }

    /// out[i] = silu(a[i]) * b[i] over two packed buffers (the text encoder's
    /// separate gate/up projections, where `a` is the gate: the same
    /// first-half gating the fused-row `swiglu` applies).
    #[expect(
        clippy::many_single_char_names,
        reason = "elementwise gate/up math reads clearer with a/b/x/g/y than with long names"
    )]
    pub(crate) fn swiglu2(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        a: DeviceView<'_>,
        b: DeviceView<'_>,
        count: usize,
    ) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let g = self.download(step, a, count)?;
            let u = self.download(step, b, count)?;
            let y: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(&s, &v)| (s / (1.0 + (-s).exp())) * v)
                .collect();
            return self.write_back(step, out, &y);
        }
        self.elementwise(step, "kernel_flux2_swiglu2_f32", count, &[out, a, b])
    }

    /// out = x * (1 + scale) + shift, with `scale`/`shift` a per-dimension
    /// vector of `period` elements broadcast over rows (the `AdaLN` modulation).
    pub(crate) fn modulate(&self, call: &ModulateCall<'_>) -> Result<(), ImageGenError> {
        let ModulateCall {
            step,
            out,
            x,
            scale,
            shift,
            count,
            period,
        } = *call;
        if self.host_fallback {
            let xv = self.download(step, x, count)?;
            let sc = self.download(step, scale, period)?;
            let sh = self.download(step, shift, period)?;
            let y: Vec<f32> = xv
                .iter()
                .enumerate()
                .map(|(i, &v)| v * (1.0 + sc[i % period]) + sh[i % period])
                .collect();
            return self.write_back(step, out, &y);
        }
        let args = ModulateArgs {
            count: count as u32,
            period: period as u32,
        };
        self.launch(
            step,
            "kernel_flux2_modulate_f32",
            Extent3::linear(grid_for(count as u32, BLOCK)),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(x),
                KernelArg::Buffer(scale),
                KernelArg::Buffer(shift),
            ],
        )
    }

    /// out = a + gate * b, with `gate` a per-dimension vector of `period`
    /// elements broadcast over rows (the gated residual add).
    pub(crate) fn add_gated(&self, call: &AddGatedCall<'_>) -> Result<(), ImageGenError> {
        let AddGatedCall {
            step,
            out,
            a,
            gate,
            b,
            count,
            period,
        } = *call;
        if self.host_fallback {
            let av = self.download(step, a, count)?;
            let g = self.download(step, gate, period)?;
            let bv = self.download(step, b, count)?;
            let y: Vec<f32> = av
                .iter()
                .enumerate()
                .map(|(i, &v)| v + g[i % period] * bv[i])
                .collect();
            return self.write_back(step, out, &y);
        }
        let args = AddGatedArgs {
            count: count as u32,
            period: period as u32,
        };
        self.launch(
            step,
            "kernel_flux2_add_gated_f32",
            Extent3::linear(grid_for(count as u32, BLOCK)),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(a),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(b),
            ],
        )
    }

    /// out = a + b, elementwise.
    pub(crate) fn add(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        a: DeviceView<'_>,
        b: DeviceView<'_>,
        count: usize,
    ) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let av = self.download(step, a, count)?;
            let bv = self.download(step, b, count)?;
            let y: Vec<f32> = av.iter().zip(&bv).map(|(&v, &w)| v + w).collect();
            return self.write_back(step, out, &y);
        }
        self.elementwise(step, "kernel_flux2_add_f32", count, &[out, a, b])
    }

    /// out = x + dt * v (the Euler latent update).
    pub(crate) fn axpy(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        x: DeviceView<'_>,
        v: DeviceView<'_>,
        count: usize,
        dt: f32,
    ) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let xv = self.download(step, x, count)?;
            let vv = self.download(step, v, count)?;
            let y: Vec<f32> = xv.iter().zip(&vv).map(|(&a, &b)| a + dt * b).collect();
            return self.write_back(step, out, &y);
        }
        let args = AxpyArgs {
            count: count as u32,
            dt,
        };
        self.launch(
            step,
            "kernel_flux2_axpy_f32",
            Extent3::linear(grid_for(count as u32, BLOCK)),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(x),
                KernelArg::Buffer(v),
            ],
        )
    }

    fn write_back(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        values: &[f32],
    ) -> Result<(), ImageGenError> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.backend.copy_h2d(out, &bytes).map_err(fail(step))
    }

    /// out[m,n] = input[m,k] @ weight[n,k]^T * `weight_scale` + bias[n].
    ///
    /// The weight is read in its stored format - BF16 or `F8_E4M3` bytes - and
    /// the scale multiplies the accumulator, never the weight in memory.
    pub(crate) fn gemm(&self, call: &GemmCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            return self.gemm_host(call);
        }
        let args = GemmArgs {
            m: call.m as u32,
            n: call.n as u32,
            k: call.k as u32,
            has_bias: u32::from(call.bias.is_some()),
        };
        let bias = call.bias.unwrap_or(self.dummy.view());
        let entry = match call.kind {
            WeightKind::Fp8 => "kernel_flux2_gemm_fp8",
            WeightKind::Bf16 => "kernel_flux2_gemm_bf16",
            WeightKind::F32 => "kernel_flux2_gemm_f32",
        };
        self.launch(
            call.step,
            entry,
            Extent3::new(
                grid_for(call.n as u32, GEMM_TILE),
                grid_for(call.m as u32, GEMM_TILE),
                1,
            ),
            Extent3::new(GEMM_TILE, GEMM_TILE, 1),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.input),
                KernelArg::Buffer(call.weight),
                KernelArg::F32(call.weight_scale),
                KernelArg::Buffer(bias),
            ],
        )
    }

    fn gemm_host(&self, call: &GemmCall<'_>) -> Result<(), ImageGenError> {
        let input = self.download(call.step, call.input, call.m * call.k)?;
        let weight_bytes = self.read_bytes(call.step, call.weight)?;
        let bias = match call.bias {
            Some(view) => self.download(call.step, view, call.n)?,
            None => vec![0.0; call.n],
        };
        let scale = call.weight_scale;
        let mut out = vec![0f32; call.m * call.n];
        for i in 0..call.m {
            for j in 0..call.n {
                let mut acc = 0f32;
                for k in 0..call.k {
                    let wi = j * call.k + k;
                    let w = match call.kind {
                        WeightKind::Fp8 => f8e4m3_to_f32(weight_bytes[wi]),
                        WeightKind::Bf16 => {
                            bf16::from_le_bytes([weight_bytes[wi * 2], weight_bytes[wi * 2 + 1]])
                                .to_f32()
                        }
                        WeightKind::F32 => f32::from_le_bytes([
                            weight_bytes[wi * 4],
                            weight_bytes[wi * 4 + 1],
                            weight_bytes[wi * 4 + 2],
                            weight_bytes[wi * 4 + 3],
                        ]),
                    };
                    acc += input[i * call.k + k] * w;
                }
                out[i * call.n + j] = acc * scale + bias[j];
            }
        }
        self.write_back(call.step, call.out, &out)
    }

    /// Raw bytes of a device buffer (for host fallbacks over native-format weights).
    pub(crate) fn read_bytes(
        &self,
        step: &'static str,
        view: DeviceView<'_>,
    ) -> Result<Vec<u8>, ImageGenError> {
        let mut bytes = vec![0u8; usize::try_from(view.len()).unwrap_or(usize::MAX)];
        self.backend
            .copy_d2h(view, &mut bytes)
            .map_err(fail(step))?;
        Ok(bytes)
    }

    /// `RMSNorm` each row: out = x * rsqrt(mean(x^2) + eps) * weight.
    pub(crate) fn rmsnorm(&self, call: &NormCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let x = self.download(call.step, call.input, call.rows * call.dim)?;
            let w = self.download(call.step, call.weight, call.dim)?;
            let mut out = vec![0f32; x.len()];
            for r in 0..call.rows {
                let row = &x[r * call.dim..(r + 1) * call.dim];
                let ms = row.iter().map(|v| v * v).sum::<f32>() / call.dim as f32;
                let inv = 1.0 / (ms + call.eps).sqrt();
                for (i, &v) in row.iter().enumerate() {
                    out[r * call.dim + i] = v * inv * w[i];
                }
            }
            return self.write_back(call.step, call.out, &out);
        }
        let args = NormArgs {
            rows: call.rows as u32,
            dim: call.dim as u32,
            eps: call.eps,
        };
        self.launch(
            call.step,
            "kernel_flux2_rmsnorm_f32",
            Extent3::linear(call.rows as u32),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.input),
                KernelArg::Buffer(call.weight),
            ],
        )
    }

    /// Weightless `LayerNorm` each row: out = (x - mean) / sqrt(var + eps).
    ///
    /// Mean-subtracting, unlike `rmsnorm`, and with no affine parameters:
    /// the transformer's block norms are `LayerNorm(affine=False)`, whose
    /// shift/scale come from the `AdaLN` modulation instead.
    pub(crate) fn layer_norm(&self, call: &LayerNormCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let x = self.download(call.step, call.input, call.rows * call.dim)?;
            let mut out = vec![0f32; x.len()];
            for r in 0..call.rows {
                let row = &x[r * call.dim..(r + 1) * call.dim];
                let mean = row.iter().sum::<f32>() / call.dim as f32;
                let var =
                    row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / call.dim as f32;
                let inv = 1.0 / (var + call.eps).sqrt();
                for (i, &v) in row.iter().enumerate() {
                    out[r * call.dim + i] = (v - mean) * inv;
                }
            }
            return self.write_back(call.step, call.out, &out);
        }
        let args = NormArgs {
            rows: call.rows as u32,
            dim: call.dim as u32,
            eps: call.eps,
        };
        self.launch(
            call.step,
            "kernel_flux2_layer_norm_f32",
            Extent3::linear(call.rows as u32),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.input),
            ],
        )
    }

    /// QK-norm the reference applies before `RoPE`: per-head RMS
    /// normalization with a per-dimension scale, in place on the q and k
    /// head sections of fused qkv rows (`x * rsqrt(mean(x^2) + eps) * scale`).
    pub(crate) fn qk_norm(&self, call: &QkNormCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let qs = self.download(call.step, call.q_scale, call.head_dim)?;
            let ks = self.download(call.step, call.k_scale, call.head_dim)?;
            let mut x = self.download(call.step, call.qkv, call.tokens * call.row_stride)?;
            let k_base = call.heads * call.head_dim;
            for t in 0..call.tokens {
                for h in 0..call.heads {
                    let q0 = t * call.row_stride + h * call.head_dim;
                    rms_scale_head(&mut x[q0..q0 + call.head_dim], &qs, call.eps);
                    let k0 = q0 + k_base;
                    rms_scale_head(&mut x[k0..k0 + call.head_dim], &ks, call.eps);
                }
            }
            return self.write_back(call.step, call.qkv, &x);
        }
        let args = QkNormArgs {
            tokens: call.tokens as u32,
            heads: call.heads as u32,
            head_dim: call.head_dim as u32,
            row_stride: call.row_stride as u32,
            eps: call.eps,
        };
        self.launch(
            call.step,
            "kernel_flux2_qk_norm_f32",
            Extent3::new(call.tokens as u32, call.heads as u32, 1),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.qkv),
                KernelArg::Buffer(call.q_scale),
                KernelArg::Buffer(call.k_scale),
            ],
        )
    }

    /// Row-wise concatenation: out[r] = [a[r], b[r]], both sides packed.
    pub(crate) fn concat_rows(&self, call: &ConcatCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let a = self.download(call.step, call.a, call.rows * call.cols_a)?;
            let b = self.download(call.step, call.b, call.rows * call.cols_b)?;
            let width = call.cols_a + call.cols_b;
            let mut out = vec![0f32; call.rows * width];
            for r in 0..call.rows {
                let (ra, rb) = (
                    &a[r * call.cols_a..(r + 1) * call.cols_a],
                    &b[r * call.cols_b..(r + 1) * call.cols_b],
                );
                out[r * width..r * width + call.cols_a].copy_from_slice(ra);
                out[r * width + call.cols_a..(r + 1) * width].copy_from_slice(rb);
            }
            return self.write_back(call.step, call.out, &out);
        }
        let args = ConcatArgs {
            rows: call.rows as u32,
            cols_a: call.cols_a as u32,
            cols_b: call.cols_b as u32,
        };
        self.launch(
            call.step,
            "kernel_flux2_concat_rows_f32",
            Extent3::linear(call.rows as u32),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.a),
                KernelArg::Buffer(call.b),
            ],
        )
    }

    /// Joint non-causal attention over [text; image] with optional cached
    /// reference keys/values, reading strided fused-qkv rows where asked.
    ///
    /// The launch layout (three q/k/v bindings per stream plus the cache, and
    /// `SdpaArgs` with the section offsets) is the Task-3 kernel contract: a
    /// kernel must read head `h` of a row at `row * stride + offset +
    /// h * head_dim` for each operand's own offset.
    pub(crate) fn sdpa(&self, call: &SdpaCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            return self.sdpa_host(call);
        }
        let args = SdpaArgs {
            txt_tokens: call.txt_tokens as u32,
            img_tokens: call.img_tokens as u32,
            cache_tokens: call.cache_tokens as u32,
            heads: call.heads as u32,
            head_dim: call.head_dim as u32,
            txt_stride: call.txt_stride as u32,
            img_stride: call.img_stride as u32,
            cache_stride: call.cache_stride as u32,
            q_offset: call.q_offset as u32,
            k_offset: call.k_offset as u32,
            v_offset: call.v_offset as u32,
        };
        let cache_k = if call.cache_tokens == 0 {
            self.dummy.view()
        } else {
            call.cache_k
        };
        let cache_v = if call.cache_tokens == 0 {
            self.dummy.view()
        } else {
            call.cache_v
        };
        self.launch(
            call.step,
            "kernel_flux2_sdpa_f32",
            Extent3::new(
                (call.txt_tokens + call.img_tokens) as u32,
                call.heads as u32,
                1,
            ),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out_txt),
                KernelArg::Buffer(call.out_img),
                KernelArg::Buffer(call.q_txt),
                KernelArg::Buffer(call.k_txt),
                KernelArg::Buffer(call.v_txt),
                KernelArg::Buffer(call.q_img),
                KernelArg::Buffer(call.k_img),
                KernelArg::Buffer(call.v_img),
                KernelArg::Buffer(cache_k),
                KernelArg::Buffer(cache_v),
            ],
        )
    }

    fn sdpa_host(&self, call: &SdpaCall<'_>) -> Result<(), ImageGenError> {
        let txt = call.txt_tokens;
        let img = call.img_tokens;
        let hd = call.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let q_txt = self.download(call.step, call.q_txt, txt * call.txt_stride)?;
        let k_txt = self.download(call.step, call.k_txt, txt * call.txt_stride)?;
        let v_txt = self.download(call.step, call.v_txt, txt * call.txt_stride)?;
        let q_img = self.download(call.step, call.q_img, img * call.img_stride)?;
        let k_img = self.download(call.step, call.k_img, img * call.img_stride)?;
        let v_img = self.download(call.step, call.v_img, img * call.img_stride)?;
        let (cache_k, cache_v) = if call.cache_tokens == 0 {
            (Vec::new(), Vec::new())
        } else {
            (
                self.download(
                    call.step,
                    call.cache_k,
                    call.cache_tokens * call.cache_stride,
                )?,
                self.download(
                    call.step,
                    call.cache_v,
                    call.cache_tokens * call.cache_stride,
                )?,
            )
        };
        // Row accessor: strided fused rows for txt/img, packed rows for cache.
        // Each operand reads its own head section at its binding's offset -
        // the transformer binds one fused buffer three times, and reading k
        // or v from the q offset would silently compute `softmax(q q^T) q`.
        // The outputs accumulate host-side and go back in one upload each:
        // a copy per (token, head) is millions of round-trips per generation.
        let width = call.heads * hd;
        let mut out_txt = vec![0f32; txt * width];
        let mut out_img = vec![0f32; img * width];
        for h in 0..call.heads {
            for t in 0..(txt + img) {
                let (qsrc, qrow) = if t < txt {
                    (&q_txt, t * call.txt_stride + call.q_offset + h * hd)
                } else {
                    (&q_img, (t - txt) * call.img_stride + call.q_offset + h * hd)
                };
                let mut scores = Vec::with_capacity(txt + img + call.cache_tokens);
                for s in 0..(txt + img) {
                    let (ksrc, krow) = if s < txt {
                        (&k_txt, s * call.txt_stride + call.k_offset + h * hd)
                    } else {
                        (&k_img, (s - txt) * call.img_stride + call.k_offset + h * hd)
                    };
                    let mut dot = 0f32;
                    for d in 0..hd {
                        dot += qsrc[qrow + d] * ksrc[krow + d];
                    }
                    scores.push(dot * scale);
                }
                for s in 0..call.cache_tokens {
                    let krow = s * call.cache_stride + call.k_offset + h * hd;
                    let mut dot = 0f32;
                    for d in 0..hd {
                        dot += qsrc[qrow + d] * cache_k[krow + d];
                    }
                    scores.push(dot * scale);
                }
                let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0f32;
                for v in &mut scores {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum;
                let mut acc = vec![0f32; hd];
                for (s, &p) in scores.iter().enumerate().take(txt + img) {
                    let (vsrc, vrow) = if s < txt {
                        (&v_txt, s * call.txt_stride + call.v_offset + h * hd)
                    } else {
                        (&v_img, (s - txt) * call.img_stride + call.v_offset + h * hd)
                    };
                    for d in 0..hd {
                        acc[d] += p * vsrc[vrow + d];
                    }
                }
                for (s, &p) in scores.iter().enumerate().skip(txt + img) {
                    let vrow = (s - (txt + img)) * call.cache_stride + call.v_offset + h * hd;
                    for d in 0..hd {
                        acc[d] += p * cache_v[vrow + d];
                    }
                }
                for v in &mut acc {
                    *v *= inv;
                }
                if t < txt {
                    out_txt[t * width + h * hd..t * width + h * hd + hd].copy_from_slice(&acc);
                } else {
                    let row = t - txt;
                    out_img[row * width + h * hd..row * width + h * hd + hd].copy_from_slice(&acc);
                }
            }
        }
        if txt > 0 {
            self.write_back(call.step, call.out_txt, &out_txt)?;
        }
        if img > 0 {
            self.write_back(call.step, call.out_img, &out_img)?;
        }
        Ok(())
    }

    /// Causal grouped-query attention (text encoder prefill).
    pub(crate) fn gqa(&self, call: &GqaCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            return self.gqa_host(call);
        }
        let args = GqaArgs {
            tokens: call.tokens as u32,
            q_heads: call.q_heads as u32,
            kv_heads: call.kv_heads as u32,
            head_dim: call.head_dim as u32,
        };
        self.launch(
            call.step,
            "kernel_flux2_gqa_f32",
            Extent3::new(call.tokens as u32, call.q_heads as u32, 1),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.q),
                KernelArg::Buffer(call.k),
                KernelArg::Buffer(call.v),
            ],
        )
    }

    fn gqa_host(&self, call: &GqaCall<'_>) -> Result<(), ImageGenError> {
        let hd = call.head_dim;
        let q = self.download(call.step, call.q, call.tokens * call.q_heads * hd)?;
        let k = self.download(call.step, call.k, call.tokens * call.kv_heads * hd)?;
        let v = self.download(call.step, call.v, call.tokens * call.kv_heads * hd)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let group = call.q_heads / call.kv_heads;
        let mut out = vec![0f32; call.tokens * call.q_heads * hd];
        for t in 0..call.tokens {
            for h in 0..call.q_heads {
                let kvh = h / group;
                let qrow = (t * call.q_heads + h) * hd;
                let mut scores: Vec<f32> = (0..=t)
                    .map(|s| {
                        let krow = (s * call.kv_heads + kvh) * hd;
                        let mut dot = 0f32;
                        for d in 0..hd {
                            dot += q[qrow + d] * k[krow + d];
                        }
                        dot * scale
                    })
                    .collect();
                let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0f32;
                for x in &mut scores {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                for (s, &p) in scores.iter().enumerate() {
                    let vrow = (s * call.kv_heads + kvh) * hd;
                    for d in 0..hd {
                        out[qrow + d] += p * v[vrow + d];
                    }
                }
                for d in 0..hd {
                    out[qrow + d] /= sum;
                }
            }
        }
        self.write_back(call.step, call.out, &out)
    }

    /// 4-axis interleaved `RoPE` on Q and K in one launch, in place, on rows of
    /// `row_stride` elements (packed or fused-qkv).
    pub(crate) fn rope_4d(&self, call: &Rope4dCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let freqs = self.download(call.step, call.freqs, call.freqs_per_axis * 4)?;
            let positions = self.download(call.step, call.positions, call.tokens * 4)?;
            // Rotate the q section from the q binding and the k section from
            // the k binding, each exactly once. The transformer binds the same
            // fused buffer to both with different offsets, so the second pass
            // re-reads the buffer the first wrote and rotates only the k
            // section - q is never rotated twice and v is never touched.
            for (tensor, offset) in [(call.q, call.q_offset), (call.k, call.k_offset)] {
                let mut x = self.download(call.step, tensor, call.tokens * call.row_stride)?;
                rope_4d_host(&mut x, &positions, &freqs, call, offset);
                self.write_back(call.step, tensor, &x)?;
            }
            return Ok(());
        }
        let args = Rope4dArgs {
            tokens: call.tokens as u32,
            heads: call.heads as u32,
            head_dim: call.head_dim as u32,
            freqs_per_axis: call.freqs_per_axis as u32,
            row_stride: call.row_stride as u32,
            q_offset: call.q_offset as u32,
            k_offset: call.k_offset as u32,
        };
        self.launch(
            call.step,
            "kernel_flux2_rope_4d",
            Extent3::new(call.tokens as u32, call.heads as u32, 1),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.q),
                KernelArg::Buffer(call.k),
                KernelArg::Buffer(call.positions),
                KernelArg::Buffer(call.freqs),
            ],
        )
    }

    /// rotate-half 1-axis `RoPE` on one (tokens, heads, `head_dim`) tensor, with
    /// the token index as the position.
    pub(crate) fn rope_half(
        &self,
        step: &'static str,
        tensor: DeviceView<'_>,
        freqs: DeviceView<'_>,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let f = self.download(step, freqs, head_dim / 2)?;
            let mut x = self.download(step, tensor, tokens * heads * head_dim)?;
            for t in 0..tokens {
                for h in 0..heads {
                    let row = (t * heads + h) * head_dim;
                    for i in 0..head_dim / 2 {
                        let angle = t as f32 * f[i];
                        let (cos, sin) = (angle.cos(), angle.sin());
                        let x0 = x[row + i];
                        let x1 = x[row + i + head_dim / 2];
                        x[row + i] = x0 * cos - x1 * sin;
                        x[row + i + head_dim / 2] = x1 * cos + x0 * sin;
                    }
                }
            }
            return self.write_back(step, tensor, &x);
        }
        let args = RopeHalfArgs {
            tokens: tokens as u32,
            heads: heads as u32,
            head_dim: head_dim as u32,
        };
        self.launch(
            step,
            "kernel_flux2_rope_half",
            Extent3::new(tokens as u32, heads as u32, 1),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(tensor),
                KernelArg::Buffer(freqs),
            ],
        )
    }

    /// 2D convolution, NCHW, f32 throughout (the VAE stores f32).
    pub(crate) fn conv2d(&self, call: &Conv2dCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            return self.conv2d_host(call);
        }
        let args = Conv2dArgs {
            batch: call.batch as u32,
            in_c: call.in_c as u32,
            in_h: call.in_h as u32,
            in_w: call.in_w as u32,
            out_c: call.out_c as u32,
            out_h: call.out_h as u32,
            out_w: call.out_w as u32,
            k_h: call.k_h as u32,
            k_w: call.k_w as u32,
            stride: call.stride as u32,
            pad: call.pad as u32,
            has_bias: u32::from(call.bias.is_some()),
        };
        let bias = call.bias.unwrap_or(self.dummy.view());
        self.launch(
            call.step,
            "kernel_flux2_conv2d_f32",
            Extent3::new(
                call.out_w as u32,
                call.out_c as u32,
                (call.batch * call.out_h) as u32,
            ),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.input),
                KernelArg::Buffer(call.weight),
                KernelArg::Buffer(bias),
            ],
        )
    }

    fn conv2d_host(&self, call: &Conv2dCall<'_>) -> Result<(), ImageGenError> {
        let x = self.download(
            call.step,
            call.input,
            call.batch * call.in_c * call.in_h * call.in_w,
        )?;
        let w = self.download(
            call.step,
            call.weight,
            call.out_c * call.in_c * call.k_h * call.k_w,
        )?;
        let b = match call.bias {
            Some(view) => self.download(call.step, view, call.out_c)?,
            None => vec![0.0; call.out_c],
        };
        let mut out = vec![0f32; call.batch * call.out_c * call.out_h * call.out_w];
        for n in 0..call.batch {
            for oc in 0..call.out_c {
                for oh in 0..call.out_h {
                    for ow in 0..call.out_w {
                        let mut acc = b[oc];
                        for ic in 0..call.in_c {
                            for kh in 0..call.k_h {
                                for kw in 0..call.k_w {
                                    let ih = oh * call.stride + kh;
                                    let iw = ow * call.stride + kw;
                                    if ih < call.pad
                                        || iw < call.pad
                                        || ih - call.pad >= call.in_h
                                        || iw - call.pad >= call.in_w
                                    {
                                        continue;
                                    }
                                    let xi = ((n * call.in_c + ic) * call.in_h + (ih - call.pad))
                                        * call.in_w
                                        + (iw - call.pad);
                                    let wi =
                                        ((oc * call.in_c + ic) * call.k_h + kh) * call.k_w + kw;
                                    acc += x[xi] * w[wi];
                                }
                            }
                        }
                        out[((n * call.out_c + oc) * call.out_h + oh) * call.out_w + ow] = acc;
                    }
                }
            }
        }
        self.write_back(call.step, call.out, &out)
    }

    /// Group normalization with affine weight and bias.
    pub(crate) fn group_norm(&self, call: &GroupNormCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let per_group = call.channels_per_group * call.spatial;
            let channels = call.groups * call.channels_per_group;
            let x = self.download(call.step, call.input, call.batch * call.groups * per_group)?;
            // The affine parameters are per channel: the buffers hold
            // `channels` elements, not one per group element.
            let w = self.download(call.step, call.weight, channels)?;
            let b = self.download(call.step, call.bias, channels)?;
            let mut out = vec![0f32; x.len()];
            for g in 0..(call.batch * call.groups) {
                let base = g * per_group;
                let mean: f32 = x[base..base + per_group].iter().sum::<f32>() / per_group as f32;
                let var: f32 = x[base..base + per_group]
                    .iter()
                    .map(|v| (v - mean) * (v - mean))
                    .sum::<f32>()
                    / per_group as f32;
                let inv = 1.0 / (var + call.eps).sqrt();
                // Element i of the group belongs to channel i / spatial
                // within the group; groups repeat per batch item.
                for i in 0..per_group {
                    let ch = (g % call.groups) * call.channels_per_group + i / call.spatial;
                    out[base + i] = (x[base + i] - mean) * inv * w[ch] + b[ch];
                }
            }
            return self.write_back(call.step, call.out, &out);
        }
        let args = GroupNormArgs {
            batch: call.batch as u32,
            num_groups: call.groups as u32,
            channels_per_group: call.channels_per_group as u32,
            spatial: call.spatial as u32,
            eps: call.eps,
        };
        self.launch(
            call.step,
            "kernel_flux2_group_norm_f32",
            Extent3::linear((call.batch * call.groups) as u32),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.input),
                KernelArg::Buffer(call.weight),
                KernelArg::Buffer(call.bias),
            ],
        )
    }

    /// Nearest-neighbour 2x spatial upsampling: (batch, channels, h, w) to
    /// (batch, channels, 2h, 2w), f32.
    pub(crate) fn upsample_nearest(&self, call: &UpsampleCall<'_>) -> Result<(), ImageGenError> {
        let UpsampleCall {
            step,
            out,
            input,
            batch,
            channels,
            h,
            w,
        } = *call;
        if self.host_fallback {
            let x = self.download(step, input, batch * channels * h * w)?;
            let mut y = vec![0f32; batch * channels * (2 * h) * (2 * w)];
            for n in 0..batch {
                for c in 0..channels {
                    for i in 0..h {
                        for j in 0..w {
                            let v = x[((n * channels + c) * h + i) * w + j];
                            for di in 0..2 {
                                for dj in 0..2 {
                                    y[((n * channels + c) * 2 * h + 2 * i + di) * 2 * w
                                        + 2 * j
                                        + dj] = v;
                                }
                            }
                        }
                    }
                }
            }
            return self.write_back(step, out, &y);
        }
        let args = UpsampleArgs {
            batch: batch as u32,
            channels: channels as u32,
            h: h as u32,
            w: w as u32,
        };
        self.launch(
            step,
            "kernel_flux2_upsample_nearest_f32",
            Extent3::new(w as u32, h as u32, (batch * channels) as u32),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(input),
            ],
        )
    }

    /// Transpose a (rows, cols) f32 matrix.
    pub(crate) fn transpose(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let x = self.download(step, input, rows * cols)?;
            let mut y = vec![0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    y[c * rows + r] = x[r * cols + c];
                }
            }
            return self.write_back(step, out, &y);
        }
        let args = NormArgs {
            rows: rows as u32,
            dim: cols as u32,
            eps: 0.0,
        };
        self.launch(
            step,
            "kernel_flux2_transpose_f32",
            Extent3::linear(grid_for((rows * cols) as u32, BLOCK)),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(input),
            ],
        )
    }

    /// Gathers `tokens` rows of a BF16 embedding table into f32 rows.
    pub(crate) fn embed(&self, call: &EmbedCall<'_>) -> Result<(), ImageGenError> {
        if self.host_fallback {
            let ids = self.read_bytes(call.step, call.ids)?;
            let table = self.read_bytes(call.step, call.table)?;
            let mut out = Vec::with_capacity(call.tokens * call.hidden);
            for t in 0..call.tokens {
                let id = u32::from_le_bytes([
                    ids[t * 4],
                    ids[t * 4 + 1],
                    ids[t * 4 + 2],
                    ids[t * 4 + 3],
                ]) as usize;
                for d in 0..call.hidden {
                    let i = (id * call.hidden + d) * 2;
                    out.push(bf16::from_le_bytes([table[i], table[i + 1]]).to_f32());
                }
            }
            return self.write_back(call.step, call.out, &out);
        }
        let args = EmbedArgs {
            tokens: call.tokens as u32,
            hidden: call.hidden as u32,
        };
        self.launch(
            call.step,
            "kernel_flux2_embed_bf16_f32",
            Extent3::new(call.tokens as u32, grid_for(call.hidden as u32, BLOCK), 1),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.table),
                KernelArg::Buffer(call.ids),
            ],
        )
    }
}

/// Host twin of one QK-norm head: `x * rsqrt(mean(x^2) + eps) * scale`, the
/// reference's `RMSNorm` evaluated entirely in f32 (the family's buffers are
/// f32, so the reference's cast-back-then-scale step is the same multiply).
fn rms_scale_head(row: &mut [f32], scale: &[f32], eps: f32) {
    let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for (v, s) in row.iter_mut().zip(scale) {
        *v *= inv * s;
    }
}

/// Host twin of `kernel_flux2_rope_4d`: interleaved pairs within each axis,
/// rotated in place on the `heads * head_dim` head section that starts
/// `section_offset` elements into each row (the q or k section of a fused
/// qkv row, or offset 0 of a packed tensor).
fn rope_4d_host(
    x: &mut [f32],
    positions: &[f32],
    freqs: &[f32],
    geom: &Rope4dCall<'_>,
    section_offset: usize,
) {
    let axes = 4;
    let axis_dim = geom.head_dim / axes;
    for t in 0..geom.tokens {
        for h in 0..geom.heads {
            let row = t * geom.row_stride + section_offset + h * geom.head_dim;
            for a in 0..axes {
                let pos = positions[t * 4 + a];
                for j in 0..geom.freqs_per_axis {
                    let angle = pos * freqs[a * geom.freqs_per_axis + j];
                    let (cos, sin) = (angle.cos(), angle.sin());
                    let i0 = row + a * axis_dim + j * 2;
                    let i1 = i0 + 1;
                    let (x0, x1) = (x[i0], x[i1]);
                    x[i0] = x0 * cos - x1 * sin;
                    x[i1] = x0 * sin + x1 * cos;
                }
            }
        }
    }
}

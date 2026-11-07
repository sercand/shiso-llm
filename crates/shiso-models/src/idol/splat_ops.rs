//! The `splat` kernel module's driver: the background remover's operations
//! the diffusion set does not carry.
//!
//! A module of its own rather than more `diffusion` entries, because the
//! diffusion driver's probe is what refuses a backend without the image
//! kernels: growing that list would refuse every image family on a backend
//! that has not ported these yet. [`SplatOps::new`] probes this module the
//! same way and refuses a backend missing any entry, naming them.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceView, Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_vision::generator::ImageGenError;

use super::nn::{AttnShape, Qkv, bytes};
use crate::diffusion::ops::{Ops, Scratch, SliceExt};

/// The module every entry here lives under.
pub(crate) const MODULE: &str = "splat";

/// Every entry this driver launches.
pub(crate) const REQUIRED_KERNELS: &[&str] = &[
    "kernel_splat_relu_f32",
    "kernel_splat_sigmoid_f32",
    "kernel_splat_gelu_erf_f32",
    "kernel_splat_resize_bilinear_f32",
    "kernel_splat_merge2x2_f32",
    "kernel_splat_window_attn_f32",
    "kernel_splat_deform_im2col_f32",
    "kernel_splat_attn_f32",
];

/// Threads per threadgroup, every entry.
const BLOCK: u32 = 64;
/// The largest window the attention kernel keeps a score row for.
const MAX_WINDOW_TOKENS: usize = 256;
/// The most threadgroups one launch puts on its second or third grid axis:
/// CUDA's limit there (Metal's is far larger, so a launch that fits CUDA
/// fits both).
const MAX_GRID_ROWS: usize = 65_535;

/// `count` rows from row `first`, cut into launches of at most [`MAX_GRID_ROWS`].
fn row_chunks(first: usize, count: usize) -> Vec<(usize, usize)> {
    (0..count)
        .step_by(MAX_GRID_ROWS)
        .map(|done| (first + done, MAX_GRID_ROWS.min(count - done)))
        .collect()
}

/// The fused attention's fixed head width, query rows per threadgroup and
/// threads per threadgroup (`SPLAT_ATTN_*` in the kernel files).
pub(crate) const ATTN_HEAD: usize = 64;
const ATTN_QROWS: usize = 64;
const ATTN_THREADS: u32 = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AttnArgs {
    queries: u32,
    keys: u32,
    heads: u32,
    width: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UnaryArgs {
    count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ResizeArgs {
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MergeArgs {
    h: u32,
    w: u32,
    channels: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WindowArgs {
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    window: u32,
    shift: u32,
    heads: u32,
    head_dim: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DeformArgs {
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    k_h: u32,
    k_w: u32,
    stride: u32,
    pad: u32,
    row0: u32,
    rows: u32,
}

const _: () = assert!(size_of::<AttnArgs>() == 20);
const _: () = assert!(size_of::<UnaryArgs>() == 4);
const _: () = assert!(size_of::<ResizeArgs>() == 20);
const _: () = assert!(size_of::<MergeArgs>() == 12);
const _: () = assert!(size_of::<WindowArgs>() == 36);
const _: () = assert!(size_of::<DeformArgs>() == 44);

/// One bilinear resampling of a `(channels, in_h, in_w)` field, corners
/// aligned.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResizeCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub input: DeviceView<'a>,
    pub channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_h: usize,
    pub out_w: usize,
}

/// One shifted-window attention over an `h x w` token grid (see the kernel).
#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowAttnCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub q: DeviceView<'a>,
    pub k: DeviceView<'a>,
    pub v: DeviceView<'a>,
    /// The key and value of a padded position: the projections' biases.
    pub k_pad: DeviceView<'a>,
    pub v_pad: DeviceView<'a>,
    /// The relative position bias, `((2 window - 1)^2, heads)`.
    pub table: DeviceView<'a>,
    pub h: usize,
    pub w: usize,
    pub window: usize,
    pub shift: usize,
    pub heads: usize,
    pub head_dim: usize,
}

/// One block of the modulated deformable convolution's columns.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeformCall<'a> {
    pub step: &'static str,
    /// `(rows, channels * k_h * k_w)`.
    pub cols: DeviceView<'a>,
    /// `(channels, in_h, in_w)`.
    pub input: DeviceView<'a>,
    /// `(2 * taps, out_h * out_w)`, `(y, x)` per tap.
    pub offset: DeviceView<'a>,
    /// `(taps, out_h * out_w)`.
    pub mask: DeviceView<'a>,
    pub channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    /// The first output pixel of the block, row-major.
    pub row0: usize,
    /// Output pixels in the block.
    pub rows: usize,
}

/// The `splat` module's driver, beside the diffusion driver it shares a
/// device with.
#[derive(Debug)]
pub(crate) struct SplatOps {
    ops: Arc<Ops>,
}

fn arg_u32(step: &'static str, value: usize) -> Result<u32, ImageGenError> {
    u32::try_from(value).map_err(|_| {
        ImageGenError::Backend(format!("{step}: {value} does not fit a launch argument"))
    })
}

/// `softmax(q k^T / 8) v` per 64-wide head in one launch of
/// `kernel_splat_attn_f32`: `q` is `(queries, heads * 64)`, `k` and `v` are
/// `(keys, heads * 64)`, row-major, and so is the answer.
///
/// A free function over [`Ops`] rather than a [`SplatOps`] method so the
/// family's transformers, which hold only the diffusion driver, reach it
/// through [`super::nn::attention`]; the generator probes the module once
/// when it opens, so a missing entry is refused there, not mid-request.
///
/// # Errors
///
/// [`ImageGenError::Backend`] for a head of another width, an empty side, a
/// count past the argument block or the grid, or a device failure.
#[expect(
    clippy::cast_precision_loss,
    reason = "a head width (64) is exact in f32"
)]
pub(crate) fn fused_attention<'o>(
    ops: &'o Ops,
    step: &'static str,
    qkv: Qkv<'_>,
    shape: AttnShape,
) -> Result<Scratch<'o>, ImageGenError> {
    let AttnShape {
        queries,
        keys,
        heads,
        head_dim,
    } = shape;
    if head_dim != ATTN_HEAD || queries == 0 || keys == 0 || heads == 0 {
        return Err(ImageGenError::Backend(format!(
            "{step}: the fused attention takes {ATTN_HEAD}-wide heads and at least one query, \
             key and head (got {queries} queries, {keys} keys, {heads} heads of {head_dim})"
        )));
    }
    let tiles = queries.div_ceil(ATTN_QROWS);
    if tiles > MAX_GRID_ROWS || heads > MAX_GRID_ROWS {
        return Err(ImageGenError::Backend(format!(
            "{step}: {queries} queries over {heads} heads is past one launch's grid"
        )));
    }
    let width = heads * head_dim;
    let out = ops.alloc_f32(step, queries * width)?;
    let args = AttnArgs {
        queries: arg_u32(step, queries)?,
        keys: arg_u32(step, keys)?,
        heads: arg_u32(step, heads)?,
        width: arg_u32(step, width)?,
        scale: 1.0 / (head_dim as f32).sqrt(),
    };
    // Every view must hold its rows: the kernel reads by index, not by view.
    for (name, view, rows) in [
        ("q", qkv.q, queries),
        ("k", qkv.k, keys),
        ("v", qkv.v, keys),
    ] {
        if view.len() < bytes(rows * width) {
            return Err(ImageGenError::Backend(format!(
                "{step}: {name} holds {} bytes, short of {rows} rows of {width}",
                view.len()
            )));
        }
    }
    ops.backend()
        .run(
            KernelRef::new(MODULE, "kernel_splat_attn_f32"),
            &LaunchArgs {
                grid: Extent3::new(arg_u32(step, tiles)?, args.heads, 1),
                block: Extent3::linear(ATTN_THREADS),
                smem_bytes: 0,
                args: &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(qkv.q),
                    KernelArg::Buffer(qkv.k),
                    KernelArg::Buffer(qkv.v),
                ],
            },
        )
        .map_err(|e| ImageGenError::Backend(format!("{step}: {e}")))?;
    Ok(out)
}

impl SplatOps {
    /// Wraps `ops`, probing its backend for every entry in
    /// [`REQUIRED_KERNELS`].
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Backend`] naming the backend and every missing entry.
    pub(crate) fn new(ops: Arc<Ops>) -> Result<Self, ImageGenError> {
        let backend = ops.backend();
        let missing: Vec<&str> = REQUIRED_KERNELS
            .iter()
            .copied()
            .filter(|entry| {
                backend
                    .max_block_threads(KernelRef::new(MODULE, entry))
                    .is_none()
            })
            .collect();
        if !missing.is_empty() {
            return Err(ImageGenError::Backend(format!(
                "the {} backend does not expose the splat kernel module: missing [{}]",
                backend.backend_name(),
                missing.join(", ")
            )));
        }
        Ok(Self { ops })
    }

    /// The diffusion driver this one sits beside.
    pub(crate) fn ops(&self) -> &Ops {
        &self.ops
    }

    fn launch(
        &self,
        step: &'static str,
        entry: &'static str,
        grid: Extent3,
        args: &[KernelArg<'_>],
    ) -> Result<(), ImageGenError> {
        self.ops
            .backend()
            .run(
                KernelRef::new(MODULE, entry),
                &LaunchArgs {
                    grid,
                    block: Extent3::linear(BLOCK),
                    smem_bytes: 0,
                    args,
                },
            )
            .map_err(|e| ImageGenError::Backend(format!("{step}: {e}")))
    }

    fn unary(
        &self,
        step: &'static str,
        entry: &'static str,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        count: usize,
    ) -> Result<(), ImageGenError> {
        let count = arg_u32(step, count)?;
        let args = UnaryArgs { count };
        self.launch(
            step,
            entry,
            Extent3::linear(count.div_ceil(BLOCK)),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(input),
            ],
        )
    }

    /// `out[i] = max(input[i], 0)`.
    pub(crate) fn relu(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        count: usize,
    ) -> Result<(), ImageGenError> {
        self.unary(step, "kernel_splat_relu_f32", out, input, count)
    }

    /// `out[i] = sigmoid(input[i])`.
    pub(crate) fn sigmoid(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        count: usize,
    ) -> Result<(), ImageGenError> {
        self.unary(step, "kernel_splat_sigmoid_f32", out, input, count)
    }

    /// `out[i] = gelu(input[i])`, the exact erf form.
    pub(crate) fn gelu_erf(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        count: usize,
    ) -> Result<(), ImageGenError> {
        self.unary(step, "kernel_splat_gelu_erf_f32", out, input, count)
    }

    /// Bilinear resampling with aligned corners.
    pub(crate) fn resize_bilinear(&self, call: &ResizeCall<'_>) -> Result<(), ImageGenError> {
        if call.in_h == 0 || call.in_w == 0 || call.out_h == 0 || call.out_w == 0 {
            return Err(ImageGenError::Backend(format!(
                "{}: an empty field does not resample",
                call.step
            )));
        }
        // Output rows and channels ride the grid's second and third axes.
        if call.out_h > MAX_GRID_ROWS || call.channels > MAX_GRID_ROWS {
            return Err(ImageGenError::Backend(format!(
                "{}: a {}-row, {}-channel field is past one launch's {MAX_GRID_ROWS} rows",
                call.step, call.out_h, call.channels
            )));
        }
        let args = ResizeArgs {
            channels: arg_u32(call.step, call.channels)?,
            in_h: arg_u32(call.step, call.in_h)?,
            in_w: arg_u32(call.step, call.in_w)?,
            out_h: arg_u32(call.step, call.out_h)?,
            out_w: arg_u32(call.step, call.out_w)?,
        };
        self.launch(
            call.step,
            "kernel_splat_resize_bilinear_f32",
            Extent3::new(args.out_w.div_ceil(BLOCK), args.out_h, args.channels),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.input),
            ],
        )
    }

    /// Swin's 2x2 patch merge of `(h * w, channels)` token rows.
    pub(crate) fn merge2x2(
        &self,
        step: &'static str,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        hw: (usize, usize),
        channels: usize,
    ) -> Result<(), ImageGenError> {
        let (h, w) = hw;
        if !h.is_multiple_of(2) || !w.is_multiple_of(2) {
            return Err(ImageGenError::Backend(format!(
                "{step}: a {h}x{w} grid does not merge 2x2"
            )));
        }
        let args = MergeArgs {
            h: arg_u32(step, h)?,
            w: arg_u32(step, w)?,
            channels: arg_u32(step, channels)?,
        };
        self.launch(
            step,
            "kernel_splat_merge2x2_f32",
            Extent3::new(
                (4 * args.channels).div_ceil(BLOCK),
                arg_u32(step, (h / 2) * (w / 2))?,
                1,
            ),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out),
                KernelArg::Buffer(input),
            ],
        )
    }

    /// Shifted-window self attention.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a head width (32) is exact in f32"
    )]
    pub(crate) fn window_attn(&self, call: &WindowAttnCall<'_>) -> Result<(), ImageGenError> {
        let ws = call.window;
        if ws == 0 || ws * ws > MAX_WINDOW_TOKENS || call.shift >= ws {
            return Err(ImageGenError::Backend(format!(
                "{}: a {ws}-wide window shifted by {} is not one the kernel holds",
                call.step, call.shift
            )));
        }
        let hp = call.h.div_ceil(ws) * ws;
        let wp = call.w.div_ceil(ws) * ws;
        let args = WindowArgs {
            h: arg_u32(call.step, call.h)?,
            w: arg_u32(call.step, call.w)?,
            hp: arg_u32(call.step, hp)?,
            wp: arg_u32(call.step, wp)?,
            window: arg_u32(call.step, ws)?,
            shift: arg_u32(call.step, call.shift)?,
            heads: arg_u32(call.step, call.heads)?,
            head_dim: arg_u32(call.step, call.head_dim)?,
            scale: 1.0 / (call.head_dim as f32).sqrt(),
        };
        self.launch(
            call.step,
            "kernel_splat_window_attn_f32",
            Extent3::new(arg_u32(call.step, (hp / ws) * (wp / ws))?, args.heads, 1),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.q),
                KernelArg::Buffer(call.k),
                KernelArg::Buffer(call.v),
                KernelArg::Buffer(call.k_pad),
                KernelArg::Buffer(call.v_pad),
                KernelArg::Buffer(call.table),
            ],
        )
    }

    /// One block of deformable columns, in launches of at most
    /// [`MAX_GRID_ROWS`] rows (the rows ride the grid's second axis).
    pub(crate) fn deform_im2col(&self, call: &DeformCall<'_>) -> Result<(), ImageGenError> {
        let width = call.channels * call.k * call.k;
        for (row0, rows) in row_chunks(call.row0, call.rows) {
            let args = DeformArgs {
                channels: arg_u32(call.step, call.channels)?,
                in_h: arg_u32(call.step, call.in_h)?,
                in_w: arg_u32(call.step, call.in_w)?,
                out_h: arg_u32(call.step, call.out_h)?,
                out_w: arg_u32(call.step, call.out_w)?,
                k_h: arg_u32(call.step, call.k)?,
                k_w: arg_u32(call.step, call.k)?,
                stride: arg_u32(call.step, call.stride)?,
                pad: arg_u32(call.step, call.pad)?,
                row0: arg_u32(call.step, row0)?,
                rows: arg_u32(call.step, rows)?,
            };
            // This launch's rows start `row0 - call.row0` rows into the
            // columns buffer.
            let skip = (row0 - call.row0) * width * 4;
            let cols = call.cols.cut(skip as u64, (rows * width * 4) as u64)?;
            self.launch(
                call.step,
                "kernel_splat_deform_im2col_f32",
                Extent3::new(arg_u32(call.step, width)?.div_ceil(BLOCK), args.rows, 1),
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(cols),
                    KernelArg::Buffer(call.input),
                    KernelArg::Buffer(call.offset),
                    KernelArg::Buffer(call.mask),
                ],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::REQUIRED_KERNELS;

    fn kernels() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
    }

    /// Every entry the driver launches is defined by both backends' sources
    /// and listed in the CUDA name table: a backend missing one refuses the
    /// whole model at load, so the gap is found here instead.
    #[test]
    fn a_launch_never_puts_more_rows_on_one_grid_axis_than_a_device_takes() {
        // A 256x256 field at a 64-wide deformable branch is 65536 rows in one
        // block; one CUDA grid axis takes at most 65535.
        let chunks = super::row_chunks(7, 65_536);
        assert!(chunks.iter().all(|&(_, rows)| rows <= super::MAX_GRID_ROWS));
        assert_eq!(chunks.first(), Some(&(7, super::MAX_GRID_ROWS)));
        assert_eq!(chunks.iter().map(|&(_, rows)| rows).sum::<usize>(), 65_536);
        let mut next = 7;
        for (row0, rows) in chunks {
            assert_eq!(row0, next, "the chunks tile the rows in order");
            next += rows;
        }
        assert_eq!(super::row_chunks(0, 10), vec![(0, 10)]);
    }

    #[test]
    fn both_backends_define_every_required_entry() {
        let metal = std::fs::read_to_string(kernels().join("metal/splat/splat.metal"))
            .expect("the Metal source");
        let cuda = std::fs::read_to_string(kernels().join("cuda/splat/splat.cu"))
            .expect("the CUDA source");
        let names = std::fs::read_to_string(kernels().join("cuda/splat/KERNEL_NAMES.md"))
            .expect("the CUDA name table");
        for entry in REQUIRED_KERNELS {
            assert!(
                metal.contains(&format!("kernel void {entry}(")),
                "Metal lacks {entry}"
            );
            assert!(
                cuda.contains(&format!("extern \"C\" __global__ void {entry}(")),
                "CUDA lacks {entry}"
            );
            assert!(
                names.lines().any(|l| l.trim() == *entry),
                "KERNEL_NAMES.md lacks {entry}"
            );
        }
    }
}

//! The residual VAE decoder's entries: the channel RMS norm's per-pixel
//! inverse norm, the two convolutions that fuse their input transform into
//! the implicit-GEMM gather (norm -> silu, nearest 2x), and the
//! parameter-free upsampling shortcut.
//!
//! The decoder these serve normalizes over the channel axis of each pixel
//! and always convolves the result, so the normalized field is never
//! written: pass 1 leaves one inverse norm per pixel and the convolution
//! applies it, with the gain and the activation, as it stages each tap.
//! At 512x512 the full-resolution 144- and 288-channel fields are the
//! decode's largest tensors; neither the normalized copy nor the upsampled
//! copy of one ever exists.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the residual VAE decoder is reached only from its device tests until the image pipeline that decodes with it lands"
    )
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceView, Extent3, KernelArg};
use shiso_vision::generator::ImageGenError;

use super::{BLOCK, G64_K, G64_THREADS, G64_TILE, Ops, grid_for};

/// The norm's floor on the per-pixel L2 norm (the reference normalizes with
/// the default `F.normalize` epsilon).
pub(crate) const CHANNEL_NORM_EPS: f32 = 1e-12;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ChannelRmsArgs {
    channels: u32,
    spatial: u32,
    scale: f32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ConvVaeArgs {
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    k: u32,
    pad: u32,
    // Presence flag, as in `GemmArgs`.
    has_bias: u32,
    accumulate: u32,
    silu: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DupUpArgs {
    out_c: u32,
    h: u32,
    w: u32,
    factor: u32,
    offset: u32,
    repeats: u32,
}

const _: () = assert!(size_of::<ChannelRmsArgs>() == 16);
const _: () = assert!(size_of::<ConvVaeArgs>() == 48);
const _: () = assert!(size_of::<DupUpArgs>() == 24);

/// Pass 1 of the channel RMS norm over an NCHW field of one image:
/// `inv[s] = sqrt(channels) / max(||input[:, s]||_2, eps)`.
pub(crate) struct ChannelRmsCall<'a> {
    pub step: &'static str,
    /// `spatial` floats.
    pub inv: DeviceView<'a>,
    /// `(channels, spatial)`.
    pub input: DeviceView<'a>,
    pub channels: usize,
    pub spatial: usize,
}

/// What a VAE convolution's gather does to its input field.
#[derive(Clone, Copy)]
pub(crate) enum VaeConvInput<'a> {
    /// Channel RMS norm (pass 1's `inv`, the per-channel `gamma`), then
    /// `SiLU` when `silu` is set.
    Rms {
        inv: DeviceView<'a>,
        gamma: DeviceView<'a>,
        silu: bool,
    },
    /// Nearest-neighbour 2x upsampling: the convolution runs over a field
    /// twice the stored one in each axis.
    Up2,
}

/// One same-padded, stride-1, square convolution of one NCHW image with
/// its input transform fused into the gather.
pub(crate) struct VaeConvCall<'a> {
    pub step: &'static str,
    /// `(out_c, h', w')`, where `(h', w')` is the stored field's size, or
    /// twice it for [`VaeConvInput::Up2`].
    pub out: DeviceView<'a>,
    /// The stored field, `(in_c, in_h, in_w)`.
    pub input: DeviceView<'a>,
    pub transform: VaeConvInput<'a>,
    /// `[out_c][in_c][k][k]` f32.
    pub weight: DeviceView<'a>,
    pub bias: Option<DeviceView<'a>>,
    pub in_c: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_c: usize,
    /// Odd kernel edge; the padding is `k / 2`.
    pub k: usize,
    /// `out += conv` (the residual add) instead of `out = conv`.
    pub accumulate: bool,
}

/// The up block's parameter-free shortcut added into the upsampled field:
/// `out (out_c, 2h, 2w) += dup_up(input (in_c, h, w))`.
pub(crate) struct DupUpCall<'a> {
    pub step: &'static str,
    pub out: DeviceView<'a>,
    pub input: DeviceView<'a>,
    pub in_c: usize,
    pub out_c: usize,
    pub h: usize,
    pub w: usize,
    /// 2 when the block also upsamples in time (of which a single frame
    /// keeps the last slot), 1 when it does not.
    pub factor_t: usize,
}

/// The shortcut's channel map for a single frame: `(factor, offset,
/// repeats)` such that output channel `c` at sub-pixel `(dy, dx)` reads
/// input channel `(c * factor + offset + 2 * dy + dx) / repeats`.
///
/// The reference repeats every input channel `repeats` times, views the
/// result as `(out_c, factor_t, 2, 2)` and interleaves the last three into
/// time, height and width; a single frame keeps only time slot
/// `factor_t - 1`, which is `offset = (factor_t - 1) * 4` into each
/// output channel's `factor` slots.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] when `out_c * factor` is not a multiple
/// of `in_c` (the reference refuses the same shape), or a factor is zero.
pub(crate) fn dup_up_map(
    in_c: usize,
    out_c: usize,
    factor_t: usize,
) -> Result<(usize, usize, usize), ImageGenError> {
    let factor = factor_t * 4;
    if in_c == 0 || factor_t == 0 || !(out_c * factor).is_multiple_of(in_c) {
        return Err(ImageGenError::Checkpoint(format!(
            "upsampling shortcut {in_c} -> {out_c} channels with temporal factor {factor_t}: \
             {out_c} x {factor} is not a whole multiple of {in_c}"
        )));
    }
    Ok((factor, (factor_t - 1) * 4, out_c * factor / in_c))
}

impl Ops {
    /// Channel RMS norm, pass 1 (see [`ChannelRmsCall`]).
    pub(crate) fn channel_rms_inv(&self, call: &ChannelRmsCall<'_>) -> Result<(), ImageGenError> {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a channel count (at most a few thousand) is exact in f32"
        )]
        let scale = (call.channels as f32).sqrt();
        let args = ChannelRmsArgs {
            channels: call.channels as u32,
            spatial: call.spatial as u32,
            scale,
            eps: CHANNEL_NORM_EPS,
        };
        self.launch(
            call.step,
            "kernel_dit_channel_rms_inv_f32",
            Extent3::linear(grid_for(call.spatial as u32, BLOCK)),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.inv),
                KernelArg::Buffer(call.input),
            ],
        )
    }

    /// One VAE convolution (see [`VaeConvCall`]). There is no scalar
    /// fallback: a tap count off the simdgroup tile's 16-wide k slice, or
    /// an even kernel, is refused.
    pub(crate) fn vae_conv(&self, call: &VaeConvCall<'_>) -> Result<(), ImageGenError> {
        let taps = call.in_c * call.k * call.k;
        if !taps.is_multiple_of(G64_K) || call.k.is_multiple_of(2) {
            return Err(ImageGenError::Backend(format!(
                "{}: a {}x{} convolution over {} channels has {taps} taps; the fused VAE \
                 convolution needs an odd kernel and a multiple of {G64_K} taps",
                call.step, call.k, call.k, call.in_c
            )));
        }
        let (out_h, out_w) = match call.transform {
            VaeConvInput::Rms { .. } => (call.in_h, call.in_w),
            VaeConvInput::Up2 => (2 * call.in_h, 2 * call.in_w),
        };
        let args = ConvVaeArgs {
            in_c: call.in_c as u32,
            in_h: call.in_h as u32,
            in_w: call.in_w as u32,
            out_c: call.out_c as u32,
            out_h: out_h as u32,
            out_w: out_w as u32,
            k: call.k as u32,
            pad: (call.k / 2) as u32,
            has_bias: u32::from(call.bias.is_some()),
            accumulate: u32::from(call.accumulate),
            silu: u32::from(matches!(
                call.transform,
                VaeConvInput::Rms { silu: true, .. }
            )),
            reserved: 0,
        };
        let bias = call.bias.unwrap_or(self.dummy.view());
        let grid = Extent3::new(
            grid_for(call.out_c as u32, G64_TILE),
            grid_for((out_h * out_w) as u32, G64_TILE),
            1,
        );
        let head = [
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(call.out),
            KernelArg::Buffer(call.input),
            KernelArg::Buffer(call.weight),
            KernelArg::Buffer(bias),
        ];
        match call.transform {
            VaeConvInput::Rms { inv, gamma, .. } => {
                let mut bound = head.to_vec();
                bound.extend([KernelArg::Buffer(inv), KernelArg::Buffer(gamma)]);
                self.launch(
                    call.step,
                    "kernel_dit_conv2d_rms_gemm_f32",
                    grid,
                    Extent3::linear(G64_THREADS),
                    &bound,
                )
            }
            VaeConvInput::Up2 => self.launch(
                call.step,
                "kernel_dit_conv2d_up2_gemm_f32",
                grid,
                Extent3::linear(G64_THREADS),
                &head,
            ),
        }
    }

    /// The up block's shortcut, added in place (see [`DupUpCall`]).
    pub(crate) fn dup_up_add(&self, call: &DupUpCall<'_>) -> Result<(), ImageGenError> {
        let (factor, offset, repeats) = dup_up_map(call.in_c, call.out_c, call.factor_t)?;
        let args = DupUpArgs {
            out_c: call.out_c as u32,
            h: call.h as u32,
            w: call.w as u32,
            factor: factor as u32,
            offset: offset as u32,
            repeats: repeats as u32,
        };
        self.launch(
            call.step,
            "kernel_dit_dup_up_add_f32",
            Extent3::new(
                grid_for((4 * call.h * call.w) as u32, BLOCK),
                call.out_c as u32,
                1,
            ),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(call.out),
                KernelArg::Buffer(call.input),
            ],
        )
    }
}

//! The Qwen-Image-2.1 RGBA VAE decoder.
//!
//! The transformer hands over its latent as packed tokens, `(h * w, 64)` in
//! row-major cell order. Decoding transposes them to a channel-first
//! `(64, h, w)` field on the device and runs the residual decoder entirely
//! in f32: `post_quant_conv` (1x1), `conv_in` (3x3, 64 -> 1152), the mid
//! block (resnet, single-head attention over every latent cell, resnet),
//! five up blocks of three resnets (the first four end in a nearest-2x
//! upsampler plus a parameter-free "dup-up" shortcut from the block's
//! input), and the head (channel RMS norm, `SiLU`, 3x3 conv to RGBA). The
//! decoded field is clamped to [-1, 1] and read back once, as RGBA8.
//!
//! The pipeline's latent denormalization (`z * latents_std + latents_mean`,
//! per channel) is folded into `post_quant_conv` at load: a 1x1 convolution
//! of an affine-per-channel input is the same convolution with its weight
//! columns scaled and its bias shifted (see [`fold_denorm`]), so the decode
//! starts from the tokens exactly as the transformer left them.
//!
//! Every norm in this decoder is a channel RMS norm: each pixel's channel
//! vector is L2-normalized, scaled by `sqrt(C)` and a per-channel gain - no
//! mean, no groups. The norm always feeds a convolution, so its apply step
//! (and the resnets' `SiLU`) rides that convolution's gather and the
//! normalized field is never written; the upsamplers' 2x interpolation rides
//! their convolution's gather the same way (`diffusion::ops::vae`). What is
//! left live at full resolution is the residual stream, one convolution's
//! output and the block's shortcut source.
//!
//! The checkpoint stores BF16 in the original layout (not the reference implementation's renamed one) and
//! names: `conv2` is `post_quant_conv`, `decoder.conv1` is `conv_in`,
//! `decoder.middle.{0,1,2}` the mid block, `decoder.upsamples.<i>.upsamples.<j>`
//! the up blocks' resnets (`residual.{0,2,3,6}` = norm1, conv1, norm2,
//! conv2, plus an optional 1x1 `shortcut`) and upsampler (`resample.1`),
//! `decoder.head.{0,2}` the output norm and convolution. Its kernels are
//! 5-D with a unit time axis (`[o, i, 1, k, k]`): a single frame through a
//! causal convolution whose time extent is one is a 2-D convolution. Every
//! tensor is widened to f32 at load (lossless for BF16; a quantized format
//! is refused by the widening read). The upsamplers' `time_conv` weights
//! act only on a second frame and are never read; the encoder is never
//! loaded.

use std::path::Path;

use serde::Deserialize;
use shiso_core::{DeviceBuffer, DeviceView};
use shiso_vision::generator::ImageGenError;

use crate::diffusion::ops::{
    ChannelRmsCall, Conv2dCall, DupUpCall, GemmCall, LinWeight, Ops, Scratch, SliceExt,
    VaeConvCall, VaeConvInput, dup_up_map,
};
use crate::diffusion::weights::StStore;

/// Pixels per latent cell along each axis: four 2x upsamplers.
pub(crate) const SPATIAL_FACTOR: usize = 16;

/// Attention scores formed at once, in f32 elements (256 MiB): a 1024x1024
/// image's 4096 latent cells score in one block, a 2048x2048 one's in four.
const ATTN_SCORE_BUDGET: usize = 1 << 26;

/// Where the decoder's weights and its config document live.
pub(crate) struct QwenVaeDesc<'a> {
    /// The VAE safetensors file (or a directory holding it).
    pub weights: &'a Path,
    /// The VAE's `config.json`: latent statistics and the temporal
    /// upsampling pattern, which no tensor carries.
    pub config: &'a Path,
}

/// The latent grid of one image, in cells (the image is
/// [`SPATIAL_FACTOR`] times it along each axis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LatentGrid {
    /// Cells per column.
    pub h: usize,
    /// Cells per row.
    pub w: usize,
}

impl LatentGrid {
    /// Latent cells, which is the token count.
    pub(crate) fn cells(self) -> usize {
        self.h * self.w
    }

    /// Image pixels per side, `(height, width)`.
    pub(crate) fn image_size(self) -> (usize, usize) {
        (self.h * SPATIAL_FACTOR, self.w * SPATIAL_FACTOR)
    }
}

/// A decoder stage the device tests read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VaeTap {
    /// `post_quant_conv` output, `(z_dim, h, w)` (denormalization included).
    PostQuant,
    /// `decoder.conv_in` output, `(c0, h, w)`.
    ConvIn,
    /// Mid-block output, `(c0, h, w)`.
    MidBlock,
}

/// A stage observer: called with the stage, its device field and its
/// element count, in issue order.
pub(crate) type Tap<'t> =
    dyn FnMut(VaeTap, DeviceView<'_>, usize) -> Result<(), ImageGenError> + 't;

#[derive(Debug, Deserialize)]
struct VaeConfig {
    z_dim: usize,
    is_residual: bool,
    temperal_downsample: Vec<bool>,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    scale_factor_spatial: Option<usize>,
    patch_size: Option<usize>,
}

/// One convolution, its shape read from the weight tensor.
#[derive(Debug)]
struct Conv {
    w: DeviceBuffer,
    bias: Option<DeviceBuffer>,
    out_c: usize,
    in_c: usize,
    k: usize,
}

/// One residual block: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2,
/// added to the input (or to its 1x1 projection when the width changes).
#[derive(Debug)]
struct Resnet {
    gamma1: DeviceBuffer,
    conv1: Conv,
    gamma2: DeviceBuffer,
    conv2: Conv,
    shortcut: Option<Conv>,
}

/// The mid block's single-head attention over every latent cell.
#[derive(Debug)]
struct Attention {
    gamma: DeviceBuffer,
    qkv: Conv,
    proj: Conv,
}

/// An up block's upsampler: nearest 2x, then a 3x3 convolution; the
/// block's dup-up shortcut has this temporal factor.
#[derive(Debug)]
struct Upsampler {
    conv: Conv,
    factor_t: usize,
}

#[derive(Debug)]
struct UpBlock {
    in_c: usize,
    resnets: Vec<Resnet>,
    upsampler: Option<Upsampler>,
}

/// The decoder's channel widths, for the working-set estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodePlan {
    /// Latent channels.
    pub z_dim: usize,
    /// Width of `conv_in` and the mid block.
    pub mid_c: usize,
    /// `(in_c, out_c, upsamples)` per up block.
    pub blocks: Vec<(usize, usize, bool)>,
    /// Image channels.
    pub out_c: usize,
}

impl DecodePlan {
    /// An upper bound on the bytes of working memory a decode of `grid`
    /// allocates through the scratch pool.
    ///
    /// Every buffer the decode asks the pool for is counted once, with no
    /// credit for reuse across sizes: a released buffer stays pooled (and,
    /// until the queue drains, referenced) for the rest of the decode, so
    /// pool residency - not the live set at any one moment - is what a
    /// decode costs. Per up block that is the residual stream and one
    /// convolution output at the block's width plus the upsampled output;
    /// the block's input is the previous stage's output, already counted.
    pub(crate) fn working_set_bytes(&self, grid: LatentGrid) -> u64 {
        let mut planes_x_spatial: u128 = 0;
        let mut spatial = grid.cells() as u128;
        let s0 = spatial;
        // The latent stage: tokens transposed, post_quant, conv_in, the mid
        // resnets' (in place) stream plus one conv output, and the
        // attention's qkv, the two transposed operands, the context in both
        // layouts and the projection.
        let (z, c0) = (self.z_dim as u128, self.mid_c as u128);
        planes_x_spatial += s0 * (2 * z + c0 * (1 + 1 + 3 + 4 + 1));
        // Score blocks, per-pixel inverse norms (two live at most per size).
        let score_rows = (ATTN_SCORE_BUDGET as u128 / s0.max(1)).clamp(1, s0.max(1));
        planes_x_spatial += score_rows * s0;
        for &(_, out_c, upsamples) in &self.blocks {
            let out_c = out_c as u128;
            planes_x_spatial += spatial * (2 * out_c + 2);
            if upsamples {
                spatial *= 4;
                planes_x_spatial += spatial * out_c;
            }
        }
        planes_x_spatial += spatial * (self.out_c as u128 + 1);
        u64::try_from(planes_x_spatial.saturating_mul(4)).unwrap_or(u64::MAX)
    }

    /// Refuses `grid` when its working set is over `budget_bytes`.
    ///
    /// # Errors
    /// [`ImageGenError::DecodeTooLarge`] naming the image size, the estimate
    /// and the budget.
    pub(crate) fn check_budget(
        &self,
        grid: LatentGrid,
        budget_bytes: u64,
    ) -> Result<(), ImageGenError> {
        let needed = self.working_set_bytes(grid);
        if needed <= budget_bytes {
            return Ok(());
        }
        let (height, width) = grid.image_size();
        Err(ImageGenError::DecodeTooLarge {
            width: u32::try_from(width).unwrap_or(u32::MAX),
            height: u32::try_from(height).unwrap_or(u32::MAX),
            needed_mib: needed.div_ceil(1 << 20),
            budget_mib: budget_bytes >> 20,
        })
    }
}

/// The field the decoder is carrying.
#[derive(Debug, Clone, Copy)]
struct Field {
    c: usize,
    h: usize,
    w: usize,
}

impl Field {
    fn spatial(self) -> usize {
        self.h * self.w
    }

    fn len(self) -> usize {
        self.c * self.h * self.w
    }
}

/// One norm -> silu -> conv: `input`'s per-pixel inverse norms are
/// computed, then the convolution applies them with `gamma` as it gathers,
/// writing `out` or (with `accumulate`) adding onto it.
#[derive(Clone, Copy)]
struct NormConv<'a> {
    conv: &'a Conv,
    gamma: &'a DeviceBuffer,
    input: &'a DeviceBuffer,
    out: DeviceView<'a>,
    field: Field,
    accumulate: bool,
    step: &'static str,
}

/// A resnet's input: consumed (the residual add lands in it) or kept for
/// the caller (the add lands in a fresh buffer).
enum Stream<'o, 'b> {
    Owned(Scratch<'o>),
    Kept(&'b DeviceBuffer),
}

impl Stream<'_, '_> {
    fn buffer(&self) -> &DeviceBuffer {
        match self {
            Stream::Owned(s) => s,
            Stream::Kept(b) => b,
        }
    }
}

/// The loaded decoder.
#[derive(Debug)]
pub(crate) struct QwenVae {
    z_dim: usize,
    post_quant: Conv,
    conv_in: Conv,
    mid_resnets: [Resnet; 2],
    mid_attn: Attention,
    up_blocks: Vec<UpBlock>,
    gamma_out: DeviceBuffer,
    conv_out: Conv,
    plan: DecodePlan,
}

/// Folds the per-channel latent denormalization `z = x * std + mean` into a
/// 1x1 convolution `y = W z + b` (`w` is `[out_c][in_c]`): `W' = W diag(std)`
/// and `b' = b + W mean`, so `y = W' x + b'`. The bias sums run in f64.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] when the statistics or the bias disagree
/// with the weight's shape.
pub(crate) fn fold_denorm(
    w: &mut [f32],
    b: &mut [f32],
    mean: &[f32],
    std: &[f32],
) -> Result<(), ImageGenError> {
    let in_c = mean.len();
    if std.len() != in_c || in_c == 0 || w.len() != b.len() * in_c {
        return Err(ImageGenError::Checkpoint(format!(
            "latent statistics ({} means, {} deviations) do not fit post_quant_conv \
             ({} weights, {} outputs)",
            mean.len(),
            std.len(),
            w.len(),
            b.len()
        )));
    }
    for (row, bias) in w.chunks_exact_mut(in_c).zip(b.iter_mut()) {
        let mut shift = f64::from(*bias);
        for ((wv, &m), &s) in row.iter_mut().zip(mean).zip(std) {
            shift += f64::from(*wv) * f64::from(m);
            *wv *= s;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the f64 sum is narrowed to the f32 the convolution runs in"
        )]
        {
            *bias = shift as f32;
        }
    }
    Ok(())
}

/// One decoded value in [-1, 1] as the reference's 8-bit sample:
/// `round_half_even(clamp(x / 2 + 0.5, 0, 1) * 255)`.
fn quantize(v: f32) -> u8 {
    let unit = (v.clamp(-1.0, 1.0) * 0.5 + 0.5).clamp(0.0, 1.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is in [0, 255] after the clamp"
    )]
    let byte = (unit * 255.0).round_ties_even() as u8;
    byte
}

/// A channel-first `(4, plane)` field in [-1, 1] as interleaved RGBA8
/// (straight alpha), the reference image processor's conversion.
///
/// # Errors
/// [`ImageGenError::Backend`] for a field of the wrong length or one
/// holding NaN (a numerics failure upstream; clamping it would hide it).
pub(crate) fn rgba8_from_field(field: &[f32], plane: usize) -> Result<Vec<u8>, ImageGenError> {
    if field.len() != 4 * plane {
        return Err(ImageGenError::Backend(format!(
            "decoded field holds {} values, expected 4 x {plane}",
            field.len()
        )));
    }
    if field.iter().any(|v| v.is_nan()) {
        return Err(ImageGenError::Backend(
            "VAE decode produced NaN in the decoded image field".to_owned(),
        ));
    }
    let mut out = vec![0u8; 4 * plane];
    for (c, channel) in field.chunks_exact(plane).enumerate() {
        for (p, &v) in channel.iter().enumerate() {
            out[p * 4 + c] = quantize(v);
        }
    }
    Ok(out)
}

/// Reads a convolution kernel `[o, i, k, k]` or `[o, i, 1, k, k]` (a
/// causal kernel with a unit time axis) and its bias, widened to f32.
fn load_conv(store: &StStore, ops: &Ops, name: &str) -> Result<Conv, ImageGenError> {
    let weight_name = format!("{name}.weight");
    let shape: Vec<usize> = store
        .require(&weight_name)?
        .shape
        .iter()
        .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
        .collect();
    let (out_c, in_c, k) = match shape.as_slice() {
        [o, i, kh, kw] | [o, i, 1, kh, kw] if kh == kw && kh % 2 == 1 => (*o, *i, *kh),
        _ => {
            return Err(ImageGenError::Checkpoint(format!(
                "'{weight_name}' has shape {shape:?}; the decoder takes square odd kernels \
                 [o, i, k, k] or [o, i, 1, k, k]"
            )));
        }
    };
    let bias_name = format!("{name}.bias");
    Ok(Conv {
        w: store.f32_widen(ops, "vae conv weight", &weight_name)?,
        bias: if store.contains(&bias_name) {
            Some(store.f32_widen(ops, "vae conv bias", &bias_name)?)
        } else {
            None
        },
        out_c,
        in_c,
        k,
    })
}

/// Reads a channel-norm gain of `channels` elements.
fn load_gamma(
    store: &StStore,
    ops: &Ops,
    name: &str,
    channels: usize,
) -> Result<DeviceBuffer, ImageGenError> {
    let entry = store.require(name)?;
    if usize::try_from(entry.elements()).ok() != Some(channels) {
        return Err(ImageGenError::Checkpoint(format!(
            "'{name}' holds {} gains for a {channels}-channel field",
            entry.elements()
        )));
    }
    store.f32_widen(ops, "vae norm gain", name)
}

fn load_resnet(store: &StStore, ops: &Ops, base: &str) -> Result<Resnet, ImageGenError> {
    let conv1 = load_conv(store, ops, &format!("{base}.residual.2"))?;
    let conv2 = load_conv(store, ops, &format!("{base}.residual.6"))?;
    let shortcut_name = format!("{base}.shortcut");
    let shortcut = if store.contains(&format!("{shortcut_name}.weight")) {
        Some(load_conv(store, ops, &shortcut_name)?)
    } else {
        None
    };
    let resnet = Resnet {
        gamma1: load_gamma(store, ops, &format!("{base}.residual.0.gamma"), conv1.in_c)?,
        gamma2: load_gamma(store, ops, &format!("{base}.residual.3.gamma"), conv1.out_c)?,
        conv1,
        conv2,
        shortcut,
    };
    let (in_c, out_c) = (resnet.conv1.in_c, resnet.conv1.out_c);
    let fits = resnet.conv2.in_c == out_c
        && resnet.conv2.out_c == out_c
        && match &resnet.shortcut {
            Some(s) => s.in_c == in_c && s.out_c == out_c && s.k == 1,
            None => in_c == out_c,
        };
    if !fits {
        return Err(ImageGenError::Checkpoint(format!(
            "resnet '{base}' convolutions do not chain ({in_c} -> {out_c})"
        )));
    }
    Ok(resnet)
}

impl QwenVae {
    /// Loads the decoder half of the VAE at `desc`.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] for a missing or misshapen tensor, a
    /// config this decoder does not implement (not residual, patchified, a
    /// spatial factor other than 16, an output other than RGBA), or latent statistics that do not fit;
    /// [`ImageGenError::Backend`] when an upload fails.
    #[expect(
        clippy::too_many_lines,
        reason = "one straight pass over the checkpoint's blocks, each check beside the tensor it guards"
    )]
    pub(crate) fn load(desc: &QwenVaeDesc<'_>, ops: &Ops) -> Result<Self, ImageGenError> {
        let config: VaeConfig = {
            let text = std::fs::read_to_string(desc.config).map_err(|e| {
                ImageGenError::Checkpoint(format!(
                    "vae config {}: {e}; the latent statistics are in no tensor",
                    desc.config.display()
                ))
            })?;
            serde_json::from_str(&text).map_err(|e| {
                ImageGenError::Checkpoint(format!("parsing {}: {e}", desc.config.display()))
            })?
        };
        if !config.is_residual {
            return Err(ImageGenError::Checkpoint(
                "vae config is not residual (is_residual: false); only the residual \
                 decoder with dup-up shortcuts is implemented"
                    .to_owned(),
            ));
        }
        if let Some(p) = config.patch_size {
            return Err(ImageGenError::Checkpoint(format!(
                "vae config sets patch_size {p}; a patchified decoder output is not implemented"
            )));
        }
        if config
            .scale_factor_spatial
            .is_some_and(|f| f != SPATIAL_FACTOR)
        {
            return Err(ImageGenError::Checkpoint(format!(
                "vae config's spatial factor {:?} is not the {SPATIAL_FACTOR} this decoder runs",
                config.scale_factor_spatial
            )));
        }

        let store = StStore::open(desc.weights)?;

        // post_quant_conv with the latent denormalization folded in.
        let pq_name = "conv2";
        let pq_entry = store.require(&format!("{pq_name}.weight"))?;
        let pq_shape: Vec<u64> = pq_entry.shape.clone();
        let z_dim = config.z_dim;
        if !(pq_shape.len() >= 2
            && pq_shape.iter().skip(2).all(|&d| d == 1)
            && usize::try_from(pq_shape[0]).ok() == Some(z_dim)
            && usize::try_from(pq_shape[1]).ok() == Some(z_dim))
        {
            return Err(ImageGenError::Checkpoint(format!(
                "post_quant_conv '{pq_name}.weight' has shape {pq_shape:?}, expected a 1x1 \
                 {z_dim} -> {z_dim} kernel"
            )));
        }
        let mut pq_w = store.f32_vec(&format!("{pq_name}.weight"))?;
        let mut pq_b = store.f32_vec(&format!("{pq_name}.bias"))?;
        fold_denorm(
            &mut pq_w,
            &mut pq_b,
            &config.latents_mean,
            &config.latents_std,
        )?;
        let post_quant = Conv {
            w: ops.upload_f32("vae post_quant weight", &pq_w)?,
            bias: Some(ops.upload_f32("vae post_quant bias", &pq_b)?),
            out_c: z_dim,
            in_c: z_dim,
            k: 1,
        };

        let conv_in = load_conv(&store, ops, "decoder.conv1")?;
        if conv_in.in_c != z_dim {
            return Err(ImageGenError::Checkpoint(format!(
                "decoder.conv1 reads {} channels, the latent has {z_dim}",
                conv_in.in_c
            )));
        }
        let mid_c = conv_in.out_c;
        let mid_resnets = [
            load_resnet(&store, ops, "decoder.middle.0")?,
            load_resnet(&store, ops, "decoder.middle.2")?,
        ];
        let mid_attn = Attention {
            gamma: load_gamma(&store, ops, "decoder.middle.1.norm.gamma", mid_c)?,
            qkv: load_conv(&store, ops, "decoder.middle.1.to_qkv")?,
            proj: load_conv(&store, ops, "decoder.middle.1.proj")?,
        };
        if mid_resnets
            .iter()
            .any(|r| r.conv1.in_c != mid_c || r.conv1.out_c != mid_c)
            || mid_attn.qkv.in_c != mid_c
            || mid_attn.qkv.out_c != 3 * mid_c
            || mid_attn.qkv.k != 1
            || mid_attn.proj.in_c != mid_c
            || mid_attn.proj.out_c != mid_c
            || mid_attn.proj.k != 1
        {
            return Err(ImageGenError::Checkpoint(format!(
                "decoder.middle does not keep the {mid_c}-channel width of decoder.conv1"
            )));
        }

        let depth = store.count_blocks("decoder.upsamples.").ok_or_else(|| {
            ImageGenError::Checkpoint("the VAE decoder has no up blocks".to_owned())
        })?;
        // The decoder upsamples in time where the encoder downsampled, in
        // reverse block order; only the blocks that upsample have a factor.
        if config.temperal_downsample.len() + 1 != depth {
            return Err(ImageGenError::Checkpoint(format!(
                "vae config lists {} temporal factors for {depth} up blocks (expected {})",
                config.temperal_downsample.len(),
                depth.saturating_sub(1)
            )));
        }
        let temporal_up: Vec<bool> = config.temperal_downsample.iter().rev().copied().collect();
        let mut up_blocks = Vec::with_capacity(depth);
        let mut width = mid_c;
        for (i, &temporal) in temporal_up
            .iter()
            .chain(std::iter::once(&false))
            .enumerate()
            .take(depth)
        {
            let base = format!("decoder.upsamples.{i}.upsamples");
            let entries = store
                .count_blocks(&format!("{base}."))
                .ok_or_else(|| ImageGenError::Checkpoint(format!("{base} is empty")))?;
            let mut resnets = Vec::with_capacity(entries);
            let mut upsampler = None;
            for j in 0..entries {
                let entry = format!("{base}.{j}");
                if store.contains(&format!("{entry}.resample.1.weight")) {
                    if j + 1 != entries {
                        return Err(ImageGenError::Checkpoint(format!(
                            "{entry} is an upsampler before the block's last entry"
                        )));
                    }
                    upsampler = Some(Upsampler {
                        conv: load_conv(&store, ops, &format!("{entry}.resample.1"))?,
                        factor_t: if temporal { 2 } else { 1 },
                    });
                } else {
                    resnets.push(load_resnet(&store, ops, &entry)?);
                }
            }
            let block_in = width;
            let mut c = width;
            for r in &resnets {
                if r.conv1.in_c != c {
                    return Err(ImageGenError::Checkpoint(format!(
                        "{base} resnet reads {} channels after a {c}-channel stage",
                        r.conv1.in_c
                    )));
                }
                c = r.conv1.out_c;
            }
            if resnets.is_empty() {
                return Err(ImageGenError::Checkpoint(format!("{base} has no resnets")));
            }
            if let Some(up) = &upsampler {
                if up.conv.in_c != c || up.conv.out_c != c {
                    return Err(ImageGenError::Checkpoint(format!(
                        "{base} upsampler maps {} -> {} channels on a {c}-channel field",
                        up.conv.in_c, up.conv.out_c
                    )));
                }
                // The shortcut's shape contract, checked now so a decode
                // never discovers it.
                dup_up_map(block_in, c, up.factor_t)?;
            }
            if (i + 1 == depth) == upsampler.is_some() {
                return Err(ImageGenError::Checkpoint(format!(
                    "{base}: every up block but the last upsamples, and the last does not"
                )));
            }
            width = c;
            up_blocks.push(UpBlock {
                in_c: block_in,
                resnets,
                upsampler,
            });
        }

        let conv_out = load_conv(&store, ops, "decoder.head.2")?;
        if conv_out.in_c != width {
            return Err(ImageGenError::Checkpoint(format!(
                "decoder.head.2 reads {} channels, the last up block emits {width}",
                conv_out.in_c
            )));
        }
        let gamma_out = load_gamma(&store, ops, "decoder.head.0.gamma", width)?;
        if conv_out.out_c != 4 {
            return Err(ImageGenError::Checkpoint(format!(
                "decoder.head.2 emits {} channels; the RGBA image takes 4",
                conv_out.out_c
            )));
        }

        let plan = DecodePlan {
            z_dim,
            mid_c,
            blocks: up_blocks
                .iter()
                .map(|b| {
                    let out = b.resnets.last().map_or(b.in_c, |r| r.conv1.out_c);
                    (b.in_c, out, b.upsampler.is_some())
                })
                .collect(),
            out_c: conv_out.out_c,
        };
        Ok(Self {
            z_dim,
            post_quant,
            conv_in,
            mid_resnets,
            mid_attn,
            up_blocks,
            gamma_out,
            conv_out,
            plan,
        })
    }

    /// The decoder's channel plan (for the working-set estimate).
    pub(crate) fn plan(&self) -> &DecodePlan {
        &self.plan
    }

    /// Image channels the decoder emits (4: RGBA).
    #[cfg(test)]
    pub(crate) fn out_channels(&self) -> usize {
        self.conv_out.out_c
    }

    /// Decodes the transformer's latent tokens `(grid.cells(), z_dim)` f32
    /// (pre-denormalization, row-major cells) into interleaved RGBA8, the
    /// image `16 * grid` pixels.
    ///
    /// # Errors
    /// [`ImageGenError::DecodeTooLarge`] when the decode's working set is
    /// over `budget_bytes` (checked before any device work);
    /// [`ImageGenError::Backend`] for a device failure or a NaN in the
    /// decoded field.
    pub(crate) fn decode(
        &self,
        ops: &Ops,
        latents: DeviceView<'_>,
        grid: LatentGrid,
        budget_bytes: u64,
    ) -> Result<Vec<u8>, ImageGenError> {
        let field = self.decode_field(ops, latents, grid, budget_bytes, None)?;
        let (h, w) = grid.image_size();
        rgba8_from_field(&field, h * w)
    }

    /// The decode up to the clamped channel-first `(4, H, W)` f32 field,
    /// with an optional observer of intermediate stages.
    ///
    /// # Errors
    /// As [`QwenVae::decode`].
    #[expect(
        clippy::too_many_lines,
        reason = "the decoder's stage graph, straight-line by construction"
    )]
    pub(crate) fn decode_field(
        &self,
        ops: &Ops,
        latents: DeviceView<'_>,
        grid: LatentGrid,
        budget_bytes: u64,
        mut tap: Option<&mut Tap<'_>>,
    ) -> Result<Vec<f32>, ImageGenError> {
        if grid.cells() == 0 {
            return Err(ImageGenError::Backend(format!(
                "VAE decode of an empty {}x{} latent grid",
                grid.w, grid.h
            )));
        }
        self.plan.check_budget(grid, budget_bytes)?;
        let cells = grid.cells();
        let needed = (cells * self.z_dim * 4) as u64;
        if latents.len() < needed {
            return Err(ImageGenError::Backend(format!(
                "latent tokens hold {} bytes, a {}x{} grid of {} channels needs {needed}",
                latents.len(),
                grid.w,
                grid.h,
                self.z_dim
            )));
        }
        let mut field = Field {
            c: self.z_dim,
            h: grid.h,
            w: grid.w,
        };

        // Tokens (cells, z) -> channel-first (z, cells), then the folded
        // post_quant_conv and conv_in.
        let z = ops.alloc_f32("vae unpack", field.len())?;
        ops.transpose("vae unpack", z.view(), latents, cells, self.z_dim)?;
        let pq = Self::conv(ops, &self.post_quant, z.view(), field, "vae post_quant")?;
        drop(z);
        if let Some(tap) = tap.as_mut() {
            tap(VaeTap::PostQuant, pq.view(), field.len())?;
        }
        let mut x = Self::conv(ops, &self.conv_in, pq.view(), field, "vae conv_in")?;
        drop(pq);
        field.c = self.conv_in.out_c;
        if let Some(tap) = tap.as_mut() {
            tap(VaeTap::ConvIn, x.view(), field.len())?;
        }

        // Mid block.
        x = Self::resnet(
            ops,
            &self.mid_resnets[0],
            Stream::Owned(x),
            &mut field,
            "vae mid 0",
        )?;
        x = self.attention(ops, x, field)?;
        x = Self::resnet(
            ops,
            &self.mid_resnets[1],
            Stream::Owned(x),
            &mut field,
            "vae mid 2",
        )?;
        if let Some(tap) = tap.as_mut() {
            tap(VaeTap::MidBlock, x.view(), field.len())?;
        }

        // Up blocks. A block that upsamples keeps its input for the dup-up
        // shortcut, so its first resnet must not add into it in place.
        for block in &self.up_blocks {
            let block_in = field;
            let (mut y, kept) = match &block.upsampler {
                Some(_) => (
                    Self::resnet(
                        ops,
                        &block.resnets[0],
                        Stream::Kept(&x),
                        &mut field,
                        "vae up",
                    )?,
                    Some(x),
                ),
                None => (
                    Self::resnet(
                        ops,
                        &block.resnets[0],
                        Stream::Owned(x),
                        &mut field,
                        "vae up",
                    )?,
                    None,
                ),
            };
            for resnet in &block.resnets[1..] {
                y = Self::resnet(ops, resnet, Stream::Owned(y), &mut field, "vae up")?;
            }
            x = match (&block.upsampler, kept) {
                (Some(up), Some(shortcut)) => {
                    let step = "vae upsample";
                    let out = ops.alloc_f32(step, up.conv.out_c * 4 * field.spatial())?;
                    ops.vae_conv(&VaeConvCall {
                        step,
                        out: out.view(),
                        input: y.view(),
                        transform: VaeConvInput::Up2,
                        weight: up.conv.w.view(),
                        bias: up.conv.bias.as_ref().map(DeviceBuffer::view),
                        in_c: field.c,
                        in_h: field.h,
                        in_w: field.w,
                        out_c: up.conv.out_c,
                        k: up.conv.k,
                        accumulate: false,
                    })?;
                    drop(y);
                    ops.dup_up_add(&DupUpCall {
                        step: "vae dup-up",
                        out: out.view(),
                        input: shortcut.view(),
                        in_c: block_in.c,
                        out_c: up.conv.out_c,
                        h: block_in.h,
                        w: block_in.w,
                        factor_t: up.factor_t,
                    })?;
                    drop(shortcut);
                    field = Field {
                        c: up.conv.out_c,
                        h: 2 * field.h,
                        w: 2 * field.w,
                    };
                    out
                }
                _ => y,
            };
        }

        // Head: channel norm -> silu -> conv_out, all in one convolution.
        let step = "vae head";
        let spatial = field.spatial();
        let inv = ops.alloc_f32(step, spatial)?;
        ops.channel_rms_inv(&ChannelRmsCall {
            step,
            inv: inv.view(),
            input: x.view(),
            channels: field.c,
            spatial,
        })?;
        let out_c = self.conv_out.out_c;
        let rgba = ops.alloc_f32(step, out_c * spatial)?;
        ops.vae_conv(&VaeConvCall {
            step,
            out: rgba.view(),
            input: x.view(),
            transform: VaeConvInput::Rms {
                inv: inv.view(),
                gamma: self.gamma_out.view(),
                silu: true,
            },
            weight: self.conv_out.w.view(),
            bias: self.conv_out.bias.as_ref().map(DeviceBuffer::view),
            in_c: field.c,
            in_h: field.h,
            in_w: field.w,
            out_c,
            k: self.conv_out.k,
            accumulate: false,
        })?;
        drop(x);
        let mut out = ops.download("vae readback", rgba.view(), out_c * spatial)?;
        // The reference clamps the decoded field to [-1, 1]; NaN survives the
        // clamp so the conversion can refuse it.
        for v in &mut out {
            if !v.is_nan() {
                *v = v.clamp(-1.0, 1.0);
            }
        }
        Ok(out)
    }

    /// A plain convolution (no input transform) over `field`, same padding.
    fn conv<'o>(
        ops: &'o Ops,
        conv: &Conv,
        input: DeviceView<'_>,
        field: Field,
        step: &'static str,
    ) -> Result<Scratch<'o>, ImageGenError> {
        let out = ops.alloc_f32(step, conv.out_c * field.spatial())?;
        ops.conv2d(&Conv2dCall {
            step,
            out: out.view(),
            input,
            weight: conv.w.view(),
            bias: conv.bias.as_ref().map(DeviceBuffer::view),
            batch: 1,
            in_c: conv.in_c,
            in_h: field.h,
            in_w: field.w,
            out_c: conv.out_c,
            out_h: field.h,
            out_w: field.w,
            k_h: conv.k,
            k_w: conv.k,
            stride: 1,
            pad: conv.k / 2,
        })?;
        Ok(out)
    }

    /// norm -> silu -> conv (see [`NormConv`]).
    fn norm_conv(ops: &Ops, call: &NormConv<'_>) -> Result<(), ImageGenError> {
        let NormConv {
            conv,
            gamma,
            input,
            out,
            field,
            accumulate,
            step,
        } = *call;
        let inv = ops.alloc_f32(step, field.spatial())?;
        ops.channel_rms_inv(&ChannelRmsCall {
            step,
            inv: inv.view(),
            input: input.view(),
            channels: field.c,
            spatial: field.spatial(),
        })?;
        ops.vae_conv(&VaeConvCall {
            step,
            out,
            input: input.view(),
            transform: VaeConvInput::Rms {
                inv: inv.view(),
                gamma: gamma.view(),
                silu: true,
            },
            weight: conv.w.view(),
            bias: conv.bias.as_ref().map(DeviceBuffer::view),
            in_c: field.c,
            in_h: field.h,
            in_w: field.w,
            out_c: conv.out_c,
            k: conv.k,
            accumulate,
        })
    }

    /// One residual block; `field` takes the block's output width.
    ///
    /// The residual stream is the 1x1 shortcut's output when the width
    /// changes, else the input itself (consumed) or a copy of it (kept);
    /// conv2 accumulates onto it in its epilogue.
    fn resnet<'o>(
        ops: &'o Ops,
        r: &Resnet,
        x: Stream<'o, '_>,
        field: &mut Field,
        step: &'static str,
    ) -> Result<Scratch<'o>, ImageGenError> {
        let out_field = Field {
            c: r.conv1.out_c,
            ..*field
        };
        let h1 = ops.alloc_f32(step, out_field.len())?;
        Self::norm_conv(
            ops,
            &NormConv {
                conv: &r.conv1,
                gamma: &r.gamma1,
                input: x.buffer(),
                out: h1.view(),
                field: *field,
                accumulate: false,
                step,
            },
        )?;
        let stream = match (&r.shortcut, x) {
            (Some(shortcut), x) => Self::conv(ops, shortcut, x.buffer().view(), *field, step)?,
            (None, Stream::Owned(x)) => x,
            (None, Stream::Kept(x)) => {
                let copy = ops.alloc_f32(step, field.len())?;
                ops.copy(step, copy.view(), x.view())?;
                copy
            }
        };
        Self::norm_conv(
            ops,
            &NormConv {
                conv: &r.conv2,
                gamma: &r.gamma2,
                input: &h1,
                out: stream.view(),
                field: out_field,
                accumulate: true,
                step,
            },
        )?;
        *field = out_field;
        Ok(stream)
    }

    /// The mid block's attention with its residual add, in place on `x`:
    /// channel norm -> 1x1 qkv -> softmax(q k^T / sqrt(c)) v over every
    /// latent cell (one head, `c` wide) -> 1x1 projection.
    #[expect(
        clippy::many_single_char_names,
        reason = "the attention is written in its formula's letters (x, q, k, v, c, s)"
    )]
    fn attention<'o>(
        &self,
        ops: &'o Ops,
        stream: Scratch<'o>,
        field: Field,
    ) -> Result<Scratch<'o>, ImageGenError> {
        let (x, c, s) = (stream, field.c, field.spatial());
        let attn = &self.mid_attn;
        let step = "vae attn";
        let inv = ops.alloc_f32(step, s)?;
        ops.channel_rms_inv(&ChannelRmsCall {
            step,
            inv: inv.view(),
            input: x.view(),
            channels: c,
            spatial: s,
        })?;
        // No activation between the norm and the projection.
        let qkv = ops.alloc_f32(step, 3 * c * s)?;
        ops.vae_conv(&VaeConvCall {
            step,
            out: qkv.view(),
            input: x.view(),
            transform: VaeConvInput::Rms {
                inv: inv.view(),
                gamma: attn.gamma.view(),
                silu: false,
            },
            weight: attn.qkv.w.view(),
            bias: attn.qkv.bias.as_ref().map(DeviceBuffer::view),
            in_c: c,
            in_h: field.h,
            in_w: field.w,
            out_c: 3 * c,
            k: 1,
            accumulate: false,
        })?;
        drop(inv);
        // The qkv field is channel-first: q is channels [0, c), k the next
        // c, v the last c (the reference chunks the channel axis in three).
        let section = (c * s * 4) as u64;
        let (q, k, v) = (
            qkv.view().cut(0, section)?,
            qkv.view().cut(section, section)?,
            qkv.view().cut(2 * section, section)?,
        );
        // The GEMM reads its right operand as (n, k) rows, so q and k go to
        // (cells, channels) and v is read as it stands, (channels, cells).
        let qt = ops.alloc_f32(step, s * c)?;
        let kt = ops.alloc_f32(step, s * c)?;
        ops.transpose(step, qt.view(), q, c, s)?;
        ops.transpose(step, kt.view(), k, c, s)?;
        let ctx_t = ops.alloc_f32(step, s * c)?;
        let block = (ATTN_SCORE_BUDGET / s).clamp(1, s);
        let scores = ops.alloc_f32(step, block * s)?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "a channel count (1152) is exact in f32"
        )]
        let scale = 1.0 / (c as f32).sqrt();
        let row_bytes = |n: usize| (n * c * 4) as u64;
        for r0 in (0..s).step_by(block) {
            let rows = block.min(s - r0);
            let slab = scores.view().cut(0, (rows * s * 4) as u64)?;
            f32_gemm(
                ops,
                "vae attn scores",
                slab,
                qt.view().cut(row_bytes(r0), row_bytes(rows))?,
                kt.view(),
                (rows, s, c),
            )?;
            ops.softmax_rows("vae attn softmax", slab, rows, s, scale)?;
            f32_gemm(
                ops,
                "vae attn values",
                ctx_t.view().cut(row_bytes(r0), row_bytes(rows))?,
                slab,
                v,
                (rows, c, s),
            )?;
        }
        drop((qt, kt, scores, qkv));
        let ctx = ops.alloc_f32(step, c * s)?;
        ops.transpose(step, ctx.view(), ctx_t.view(), s, c)?;
        drop(ctx_t);
        let proj = Self::conv(ops, &attn.proj, ctx.view(), field, step)?;
        ops.add(step, x.view(), x.view(), proj.view(), c * s)?;
        Ok(x)
    }
}

/// `out (m, n) = input (m, k) . weight (n, k)^T`, all f32: the attention's
/// two products. The one place this decoder reaches the linear GEMM.
fn f32_gemm(
    ops: &Ops,
    step: &'static str,
    out: DeviceView<'_>,
    input: DeviceView<'_>,
    weight: DeviceView<'_>,
    (m, n, k): (usize, usize, usize),
) -> Result<(), ImageGenError> {
    ops.gemm(&GemmCall {
        step,
        out,
        input,
        weight: LinWeight::f32(weight),
        m,
        n,
        k,
    })
}

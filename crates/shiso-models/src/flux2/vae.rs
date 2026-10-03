//! The FLUX.2 spatial VAE decoder.
//!
//! Decoding runs in three parts. First the latent sequence is unpacked on
//! the host: the transformer's patchified (tokens, 128) rows become a
//! (128, H/16, W/16) field, denormalized with the checkpoint's own
//! `bn.running_mean`/`bn.running_var`, then spatially unpatchified to
//! (32, H/8, W/8). Then the device runs the convolutional decoder -
//! `post_quant_conv`, `conv_in`, the mid block (resnet, single-head spatial
//! attention, resnet), the up blocks with their resnets and nearest
//! upsamplers, and the final norm/conv - entirely in f32. Finally the RGB
//! field is read back and clamped.
//!
//! The decoder runs f32 by plan decision, while the shipped checkpoints store
//! the whole VAE in `BF16`. Every VAE tensor is therefore imported through
//! the widening read: `BF16` to `F32` is lossless (the wider format holds
//! every `BF16` value exactly), so the zero-dequant constraint - which
//! protects *quantized* formats like `F8_E4M3` from being flattened into f32
//! and stays native in the `DiT` - is not engaged. A quantized VAE is refused
//! by the widening read rather than silently dequantized.
//!
//! Every channel width, kernel size and block count is read from the weight
//! shapes; the config document beside the weights supplies only what no
//! tensor carries - the group count and, when present, the batch-norm
//! epsilon.

use std::path::Path;

use serde::Deserialize;
use shiso_core::DeviceBuffer;
use shiso_vision::generator::ImageGenError;

use super::ops::{Conv2dCall, GroupNormCall, Ops};
use super::weights::StStore;

/// Group-norm epsilon the decoder uses (the reference's group-norm epsilon).
const GN_EPS: f32 = 1e-6;
/// Batch-norm epsilon for the latent denormalization, used when the
/// checkpoint's `vae/config.json` carries no `batch_norm_eps` (the reference
/// trains with this value).
const DEFAULT_BN_EPS: f32 = 1e-4;

/// One convolution with its shape read from the weight tensor.
#[derive(Debug)]
struct Conv {
    w: DeviceBuffer,
    bias: Option<DeviceBuffer>,
    out_c: usize,
    in_c: usize,
    k_h: usize,
    k_w: usize,
}

/// A residual normalization block: two (norm, silu, conv) stages and an
/// optional 1x1 shortcut.
#[derive(Debug)]
struct Resnet {
    norm1_w: DeviceBuffer,
    norm1_b: DeviceBuffer,
    conv1: Conv,
    norm2_w: DeviceBuffer,
    norm2_b: DeviceBuffer,
    conv2: Conv,
    shortcut: Option<Conv>,
}

/// The mid block's single-head spatial attention.
#[derive(Debug)]
struct AttnBlock {
    norm_w: DeviceBuffer,
    norm_b: DeviceBuffer,
    q: Conv,
    k: Conv,
    v: Conv,
    out: Conv,
}

#[derive(Debug)]
struct UpBlock {
    resnets: Vec<Resnet>,
    upsample: Option<Conv>,
}

/// The field the decoder is carrying: channel count and spatial size. The
/// convolutions it runs are same-padding, so only a shortcut or an upsampler
/// changes it.
#[derive(Debug, Clone, Copy)]
struct Field {
    channels: usize,
    h: usize,
    w: usize,
}

/// The loaded decoder.
#[derive(Debug)]
pub(crate) struct Vae {
    groups: usize,
    bn_mean: Vec<f32>,
    bn_std: Vec<f32>,
    post_quant: Conv,
    conv_in: Conv,
    mid_resnets: [Resnet; 2],
    mid_attn: AttnBlock,
    up_blocks: Vec<UpBlock>,
    norm_out_w: DeviceBuffer,
    norm_out_b: DeviceBuffer,
    conv_out: Conv,
}

#[derive(Debug, Deserialize)]
struct VaeConfig {
    norm_num_groups: Option<usize>,
    batch_norm_eps: Option<f32>,
}

/// Reads one convolution weight and its bias, widened to the f32 the decoder
/// runs in (see the module doc: `BF16` checkpoints widen losslessly).
#[expect(
    clippy::cast_possible_truncation,
    reason = "safetensors dims are element counts of conv kernels; they fit usize on any supported target"
)]
fn load_conv(store: &StStore, ops: &Ops, name: &str) -> Result<Conv, ImageGenError> {
    let weight_name = format!("{name}.weight");
    let entry = store.require(&weight_name)?;
    let shape = &entry.shape;
    let (out_c, in_c, k_h, k_w) = (
        *shape
            .first()
            .ok_or_else(|| ImageGenError::Checkpoint(format!("'{weight_name}' has empty shape")))?
            as usize,
        *shape.get(1).ok_or_else(|| {
            ImageGenError::Checkpoint(format!("'{weight_name}' is not a conv kernel"))
        })? as usize,
        shape.get(2).copied().unwrap_or(1) as usize,
        shape.get(3).copied().unwrap_or(1) as usize,
    );
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
        k_h,
        k_w,
    })
}

/// Reads a group-norm affine pair (both widened to the f32 the decoder runs
/// in), uploaded as vectors of `channels`.
fn load_norm(
    store: &StStore,
    ops: &Ops,
    name: &str,
) -> Result<(DeviceBuffer, DeviceBuffer), ImageGenError> {
    Ok((
        store.f32_widen(ops, "vae norm affine", &format!("{name}.weight"))?,
        store.f32_widen(ops, "vae norm affine", &format!("{name}.bias"))?,
    ))
}

fn load_resnet(store: &StStore, ops: &Ops, base: &str) -> Result<Resnet, ImageGenError> {
    let (norm1_w, norm1_b) = load_norm(store, ops, &format!("{base}norm1"))?;
    let (norm2_w, norm2_b) = load_norm(store, ops, &format!("{base}norm2"))?;
    Ok(Resnet {
        norm1_w,
        norm1_b,
        conv1: load_conv(store, ops, &format!("{base}conv1"))?,
        norm2_w,
        norm2_b,
        conv2: load_conv(store, ops, &format!("{base}conv2"))?,
        shortcut: if store.contains(&format!("{base}conv_shortcut.weight")) {
            Some(load_conv(store, ops, &format!("{base}conv_shortcut"))?)
        } else {
            None
        },
    })
}

impl Vae {
    /// Opens the VAE component at `path`.
    pub(crate) fn load(path: &Path, ops: &Ops) -> Result<Self, ImageGenError> {
        let store = StStore::open(path)?;
        // The config document is required, not optional: the group count is
        // carried by no tensor, and a default would reshape every group
        // norm in the decoder. The batch-norm epsilon may default (the
        // reference trains with it); the document itself may not.
        let config: VaeConfig =
            super::config::read_component_config(&path.join("config.json"), "vae")?;
        let groups = config.norm_num_groups.ok_or_else(|| {
            ImageGenError::Checkpoint(
                "vae/config.json carries no norm_num_groups: the group count is not \
                 derivable from tensor shapes and a default would reshape every group norm"
                    .to_string(),
            )
        })?;
        if groups == 0 {
            return Err(ImageGenError::Checkpoint(
                "vae/config.json declares norm_num_groups 0; no field splits into zero groups"
                    .to_string(),
            ));
        }

        let bn_mean = store.f32_vec("bn.running_mean")?;
        let bn_var = store.f32_vec("bn.running_var")?;
        if bn_mean.len() != bn_var.len() {
            return Err(ImageGenError::Checkpoint(
                "bn.running_mean and bn.running_var disagree on channel count".to_string(),
            ));
        }
        let bn_std: Vec<f32> = bn_var
            .iter()
            .map(|v| (v + config.batch_norm_eps.unwrap_or(DEFAULT_BN_EPS)).sqrt())
            .collect();

        let post_quant = load_conv(&store, ops, "post_quant_conv")?;
        let conv_in = load_conv(&store, ops, "decoder.conv_in")?;

        let mid_resnets = [
            load_resnet(&store, ops, "decoder.mid_block.resnets.0.")?,
            load_resnet(&store, ops, "decoder.mid_block.resnets.1.")?,
        ];
        let attn_base = "decoder.mid_block.attentions.0.";
        let norm_base = if store.contains(&format!("{attn_base}norm.weight")) {
            "norm"
        } else {
            "group_norm"
        };
        let (norm_w, norm_b) = {
            let n = format!("{attn_base}{norm_base}");
            (
                store.f32_widen(ops, "vae norm affine", &format!("{n}.weight"))?,
                store.f32_widen(ops, "vae norm affine", &format!("{n}.bias"))?,
            )
        };
        // The reference names the attention output projection `to_out.0`
        // (a Sequential); an export that flattened it may carry the upstream
        // `out` instead. Try the reference spelling first and fall back to
        // the upstream one; if neither is present the missing-tensor error
        // names `out`, the last spelling tried.
        let out_base = if store.contains(&format!("{attn_base}to_out.0.weight")) {
            "to_out.0"
        } else {
            "out"
        };
        let mid_attn = AttnBlock {
            norm_w,
            norm_b,
            q: load_conv(&store, ops, &format!("{attn_base}to_q"))?,
            k: load_conv(&store, ops, &format!("{attn_base}to_k"))?,
            v: load_conv(&store, ops, &format!("{attn_base}to_v"))?,
            out: load_conv(&store, ops, &format!("{attn_base}{out_base}"))?,
        };

        let depth = store
            .count_blocks("decoder.up_blocks.")
            .ok_or_else(|| ImageGenError::Checkpoint("VAE decoder has no up_blocks".to_string()))?;
        let mut up_blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let base = format!("decoder.up_blocks.{i}.");
            let resnets_depth = store.count_blocks(&format!("{base}resnets.")).unwrap_or(0);
            if resnets_depth == 0 {
                return Err(ImageGenError::Checkpoint(format!(
                    "{base}resnets has no blocks"
                )));
            }
            let mut resnets = Vec::with_capacity(resnets_depth);
            for j in 0..resnets_depth {
                resnets.push(load_resnet(&store, ops, &format!("{base}resnets.{j}."))?);
            }
            let upsample = if store.contains(&format!("{base}upsamplers.0.conv.weight")) {
                Some(load_conv(&store, ops, &format!("{base}upsamplers.0.conv"))?)
            } else {
                None
            };
            up_blocks.push(UpBlock { resnets, upsample });
        }

        let norm_out_w = store.f32_widen(ops, "vae norm affine", "decoder.conv_norm_out.weight")?;
        let norm_out_b = store.f32_widen(ops, "vae norm affine", "decoder.conv_norm_out.bias")?;
        let conv_out = load_conv(&store, ops, "decoder.conv_out")?;

        Ok(Self {
            groups,
            bn_mean,
            bn_std,
            post_quant,
            conv_in,
            mid_resnets,
            mid_attn,
            up_blocks,
            norm_out_w,
            norm_out_b,
            conv_out,
        })
    }

    /// Decodes a patchified latent sequence (tokens, `in_channels`) f32 into
    /// an RGB field (3, height, width) f32 in [0, 255].
    #[expect(
        clippy::too_many_lines,
        reason = "the decoder's stage graph, straight-line by construction"
    )]
    pub(crate) fn decode(
        &self,
        ops: &Ops,
        latents: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        let z = self.unpack_latents(latents, width, height)?;
        let mut x = ops.upload_f32("vae latent", &z)?;
        let mut field = Field {
            channels: self.post_quant.in_c,
            h: height / 8,
            w: width / 8,
        };

        // post_quant_conv and conv_in.
        let y = Self::conv(
            ops,
            &self.post_quant,
            x.view(),
            field.channels,
            &field,
            "vae pq",
        )?;
        field.channels = self.post_quant.out_c;
        drop(x);
        x = y;
        let y = Self::conv(
            ops,
            &self.conv_in,
            x.view(),
            field.channels,
            &field,
            "vae in",
        )?;
        field.channels = self.conv_in.out_c;
        drop(x);
        x = y;

        // Mid block.
        x = self.resnet(ops, &self.mid_resnets[0], &x, &mut field, "vae mid0")?;
        x = self.attention(ops, &self.mid_attn, &x, &field)?;
        x = self.resnet(ops, &self.mid_resnets[1], &x, &mut field, "vae mid1")?;

        // Up blocks; the last one has no upsampler.
        for (i, block) in self.up_blocks.iter().enumerate() {
            for (j, resnet) in block.resnets.iter().enumerate() {
                let step: &'static str = match (i, j) {
                    (0, 0) => "vae up 0.0",
                    (0, 1) => "vae up 0.1",
                    (0, 2) => "vae up 0.2",
                    (1, 0) => "vae up 1.0",
                    (1, 1) => "vae up 1.1",
                    (1, 2) => "vae up 1.2",
                    (2, 0) => "vae up 2.0",
                    (2, 1) => "vae up 2.1",
                    (2, 2) => "vae up 2.2",
                    (3, 0) => "vae up 3.0",
                    (3, 1) => "vae up 3.1",
                    (3, 2) => "vae up 3.2",
                    _ => "vae up n",
                };
                x = self.resnet(ops, resnet, &x, &mut field, step)?;
            }
            if let Some(up) = &block.upsample {
                // Reference `Upsample.forward`: nearest 2x interpolation
                // FIRST, then the 3x3 conv over the 2x field - convolving at
                // the low resolution and upscaling after would sample the
                // conv weights against the wrong field.
                let upsampled = ops.alloc_f32(
                    "vae upsample",
                    field.channels * (2 * field.h) * (2 * field.w),
                )?;
                ops.upsample_nearest(&super::ops::UpsampleCall {
                    step: "vae upsample",
                    out: upsampled.view(),
                    input: x.view(),
                    batch: 1,
                    channels: field.channels,
                    h: field.h,
                    w: field.w,
                })?;
                drop(x);
                field.h *= 2;
                field.w *= 2;
                let y = Self::conv(
                    ops,
                    up,
                    upsampled.view(),
                    field.channels,
                    &field,
                    "vae upsample conv",
                )?;
                drop(upsampled);
                field.channels = up.out_c;
                x = y;
            }
        }

        // Final norm, silu, conv_out.
        let spatial = field.h * field.w;
        let normed = ops.alloc_f32("vae out norm", field.channels * spatial)?;
        ops.group_norm(&GroupNormCall {
            step: "vae out norm",
            out: normed.view(),
            input: x.view(),
            weight: self.norm_out_w.view(),
            bias: self.norm_out_b.view(),
            batch: 1,
            groups: self.groups,
            channels_per_group: field.channels / self.groups,
            spatial,
            eps: GN_EPS,
        })?;
        ops.silu(
            "vae out silu",
            normed.view(),
            normed.view(),
            field.channels * spatial,
        )?;
        let rgb = Self::conv(
            ops,
            &self.conv_out,
            normed.view(),
            field.channels,
            &field,
            "vae out",
        )?;
        let out = ops.download("vae readback", rgb.view(), 3 * spatial)?;
        // The decoder was trained to emit an image in [-1, 1]; the reference
        // clamps there before the pipeline maps to [0, 255]. Doing the map
        // here keeps the encoder's [0, 255] contract on this function's doc
        // honest instead of letting a [-1, 1] field clamp to black.
        //
        // `f32::clamp` panics on a NaN operand, and a numerics failure
        // upstream shows up as exactly that - so one pass over the whole
        // field turns a non-finite decode into the typed backend error the
        // caller logs, rather than unwinding the image worker through a
        // panic the catch_unwind would only report as "the generator
        // panicked".
        if out.iter().any(|v| v.is_nan()) {
            return Err(ImageGenError::Backend(
                "VAE decode produced NaN in the decoded image field".to_owned(),
            ));
        }
        let out: Vec<f32> = out
            .iter()
            .map(|v| (v.clamp(-1.0, 1.0) + 1.0) * 127.5)
            .collect();
        Ok(out)
    }

    /// Host-side latent unpack: patchified (tokens, 128) rows to a
    /// denormalized, unpatchified (`z_channels`, H/8, W/8) field.
    ///
    /// Feature `c * 4 + p` of the token at latent cell (li, lj) is channel
    /// `c` at spatial position `(2*li + p/2, 2*lj + p%2)` - the inverse of
    /// the 2x2 patchification the transformer's input was built with. The
    /// flattened feature index is channel-major (`f = c * pi * pj + p`),
    /// pinned by the upstream pack in `AutoEncoder.encode`:
    /// `rearrange(mean, "... c (i pi) (j pj) -> ... (c pi pj) i j")` - the
    /// `(c pi pj)` group means the channel index varies slowest, exactly the
    /// `(c, p)` split below.
    fn unpack_latents(
        &self,
        latents: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        let (hl, wl) = (height / 16, width / 16);
        let zc = self.post_quant.in_c;
        if latents.len() != hl * wl * zc * 4 {
            return Err(ImageGenError::Backend(format!(
                "latent sequence is {} elements, expected {}",
                latents.len(),
                hl * wl * zc * 4
            )));
        }
        // The batch-norm statistics live in the PACKED space: the reference
        // `AutoEncoder` normalises with a BatchNorm2d over `prod(ps) *
        // z_channels` = 128 features and `inv_normalize` runs before the
        // unpack, so entry `f` of the stats belongs to packed feature `f`,
        // not to latent channel `f / 4`.
        if self.bn_mean.len() != zc * 4 {
            return Err(ImageGenError::Checkpoint(format!(
                "bn statistics cover {} features, expected {} ({} latent channels x 4 \
                 patch positions)",
                self.bn_mean.len(),
                zc * 4,
                zc
            )));
        }
        let mut z = vec![0f32; zc * height / 8 * width / 8];
        let oh = height / 8;
        let ow = width / 8;
        for token in 0..(hl * wl) {
            let (li, lj) = (token / wl, token % wl);
            for f in 0..(zc * 4) {
                let (c, p) = (f / 4, f % 4);
                let (pi, pj) = (p / 2, p % 2);
                let raw = latents[token * zc * 4 + f];
                z[(c * oh + 2 * li + pi) * ow + 2 * lj + pj] =
                    raw * self.bn_std[f] + self.bn_mean[f];
            }
        }
        Ok(z)
    }

    /// One convolution with its bias, returning the output buffer. The
    /// spatial size is the field's; same-padding convolutions preserve it.
    fn conv(
        ops: &Ops,
        conv: &Conv,
        input: shiso_core::DeviceView<'_>,
        in_c: usize,
        field: &Field,
        step: &'static str,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let pad = conv.k_h / 2;
        let out_h = (field.h + 2 * pad - conv.k_h) + 1;
        let out_w = (field.w + 2 * pad - conv.k_w) + 1;
        let out = ops.alloc_f32(step, conv.out_c * out_h * out_w)?;
        ops.conv2d(&Conv2dCall {
            step,
            out: out.view(),
            input,
            weight: conv.w.view(),
            bias: conv.bias.as_ref().map(DeviceBuffer::view),
            batch: 1,
            in_c,
            in_h: field.h,
            in_w: field.w,
            out_c: conv.out_c,
            out_h,
            out_w,
            k_h: conv.k_h,
            k_w: conv.k_w,
            stride: 1,
            pad,
        })?;
        Ok(out)
    }

    /// One residual normalization block; updates the field's channels when a
    /// shortcut changes them.
    fn resnet(
        &self,
        ops: &Ops,
        resnet: &Resnet,
        x: &DeviceBuffer,
        field: &mut Field,
        step: &'static str,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let spatial = field.h * field.w;
        let xn = ops.alloc_f32(step, field.channels * spatial)?;
        ops.group_norm(&GroupNormCall {
            step,
            out: xn.view(),
            input: x.view(),
            weight: resnet.norm1_w.view(),
            bias: resnet.norm1_b.view(),
            batch: 1,
            groups: self.groups,
            channels_per_group: field.channels / self.groups,
            spatial,
            eps: GN_EPS,
        })?;
        ops.silu(step, xn.view(), xn.view(), field.channels * spatial)?;
        let mut y = Self::conv(ops, &resnet.conv1, xn.view(), field.channels, field, step)?;
        ops.group_norm(&GroupNormCall {
            step,
            out: xn.view(),
            input: y.view(),
            weight: resnet.norm2_w.view(),
            bias: resnet.norm2_b.view(),
            batch: 1,
            groups: self.groups,
            channels_per_group: resnet.conv1.out_c / self.groups,
            spatial,
            eps: GN_EPS,
        })?;
        ops.silu(step, xn.view(), xn.view(), resnet.conv1.out_c * spatial)?;
        y = Self::conv(
            ops,
            &resnet.conv2,
            xn.view(),
            resnet.conv1.out_c,
            field,
            step,
        )?;
        let summed = if let Some(shortcut) = &resnet.shortcut {
            let s = Self::conv(ops, shortcut, x.view(), field.channels, field, step)?;
            let out = ops.alloc_f32(step, shortcut.out_c * spatial)?;
            ops.add(
                step,
                out.view(),
                s.view(),
                y.view(),
                shortcut.out_c * spatial,
            )?;
            field.channels = shortcut.out_c;
            out
        } else {
            let out = ops.alloc_f32(step, field.channels * spatial)?;
            ops.add(
                step,
                out.view(),
                x.view(),
                y.view(),
                field.channels * spatial,
            )?;
            out
        };
        Ok(summed)
    }

    /// Single-head spatial self-attention with a residual add: the group
    /// norm feeds the q/k/v projections directly, with no activation between.
    fn attention(
        &self,
        ops: &Ops,
        attn: &AttnBlock,
        x: &DeviceBuffer,
        field: &Field,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let channels = field.channels;
        let spatial = field.h * field.w;
        let xn = ops.alloc_f32("vae attn norm", channels * spatial)?;
        ops.group_norm(&GroupNormCall {
            step: "vae attn norm",
            out: xn.view(),
            input: x.view(),
            weight: attn.norm_w.view(),
            bias: attn.norm_b.view(),
            batch: 1,
            groups: self.groups,
            channels_per_group: channels / self.groups,
            spatial,
            eps: GN_EPS,
        })?;
        // No activation between the norm and the projections: the reference's
        // attention block reads the normed field straight into q/k/v, unlike
        // the resnet blocks, which do activate after each norm. A SiLU here
        // would bend every mid-block attention statistic away from the
        // trained field.
        let q = Self::conv(ops, &attn.q, xn.view(), channels, field, "vae attn q")?;
        let k = Self::conv(ops, &attn.k, xn.view(), channels, field, "vae attn k")?;
        let v = Self::conv(ops, &attn.v, xn.view(), channels, field, "vae attn v")?;
        // The attention kernel wants (spatial, channels) rows.
        let qt = ops.alloc_f32("vae attn qt", spatial * channels)?;
        let kt = ops.alloc_f32("vae attn kt", spatial * channels)?;
        let vt = ops.alloc_f32("vae attn vt", spatial * channels)?;
        ops.transpose("vae attn t", qt.view(), q.view(), channels, spatial)?;
        ops.transpose("vae attn t", kt.view(), k.view(), channels, spatial)?;
        ops.transpose("vae attn t", vt.view(), v.view(), channels, spatial)?;
        let at = ops.alloc_f32("vae attn ctx", spatial * channels)?;
        ops.sdpa(&super::ops::SdpaCall {
            step: "vae attention",
            out_txt: at.view(),
            out_img: at.view(),
            q_txt: qt.view(),
            k_txt: kt.view(),
            v_txt: vt.view(),
            q_img: qt.view(),
            k_img: kt.view(),
            v_img: vt.view(),
            // Separate packed q/k/v tensors, not fused rows: every head
            // section starts at column 0 of its own buffer.
            q_offset: 0,
            k_offset: 0,
            v_offset: 0,
            cache_k: ops.dummy_view(),
            cache_v: ops.dummy_view(),
            cache_stride: channels,
            txt_tokens: spatial,
            img_tokens: 0,
            cache_tokens: 0,
            heads: 1,
            head_dim: channels,
            txt_stride: channels,
            img_stride: channels,
        })?;
        let ctx = ops.alloc_f32("vae attn ctx ch", channels * spatial)?;
        ops.transpose("vae attn t", ctx.view(), at.view(), spatial, channels)?;
        let proj = Self::conv(ops, &attn.out, ctx.view(), channels, field, "vae attn proj")?;
        let out = ops.alloc_f32("vae attn residual", channels * spatial)?;
        ops.add(
            "vae attn residual",
            out.view(),
            x.view(),
            proj.view(),
            channels * spatial,
        )?;
        Ok(out)
    }
}

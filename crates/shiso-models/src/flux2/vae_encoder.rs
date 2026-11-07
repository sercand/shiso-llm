//! The FLUX.2 spatial VAE encoder: a picture to its latent posterior.
//!
//! The decoder's mirror image, built from the same parts (`vae`'s
//! convolutions, residual blocks and single-head mid attention): `conv_in`,
//! one down block per entry of `block_out_channels` with two residual blocks
//! each and a stride-2 downsampler on all but the last, the mid block,
//! `conv_norm_out` + `SiLU` + `conv_out`, then `quant_conv`. The result is the
//! posterior's moments, `2 * latent_channels` planes at an eighth of the
//! picture's size: the first half the mean, the second the log-variance.
//!
//! The downsampler is the one convolution the decoder has no twin for. The
//! reference pads the field by one column on the right and one row at the
//! bottom only, then runs a 3x3 kernel at stride 2 with no padding of its own.
//! The driver's convolution pads symmetrically, so the 3x3 kernel is loaded
//! into the bottom-right of a 4x4 one whose first row and column are zero:
//! with a symmetric pad of one, tap `(1 + a, 1 + b)` of the 4x4 kernel reads
//! exactly the pixel tap `(a, b)` of the 3x3 kernel reads in the reference,
//! the zero row and column absorb the left/top pad, and the right/bottom pad
//! is the reference's own extra row and column. A plain symmetric 3x3 would
//! shift every latent by half a pixel.
//!
//! The encoder runs f32, like the decoder, widening the `BF16` checkpoint
//! losslessly.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the splat conditioner is the encoder's first caller and lands next"
    )
)]

use std::path::Path;

use shiso_vision::generator::ImageGenError;

use super::vae::{
    Conv, DEFAULT_BN_EPS, Field, GN_EPS, Resnet, SpatialAttention, VaeConfig, attention, conv,
    load_attention, load_conv, load_norm, load_resnet, resnet,
};
use crate::diffusion::ops::{Conv2dCall, GroupNormCall, Ops};
use crate::diffusion::weights::StStore;
use shiso_core::DeviceBuffer;

/// One down block: its residual blocks and, except for the last block, the
/// stride-2 downsampler (stored as the zero-bordered 4x4 kernel the module
/// doc describes).
#[derive(Debug)]
struct DownBlock {
    resnets: Vec<Resnet>,
    downsample: Option<Conv>,
}

/// The loaded encoder, with the batch-norm statistics the latent's tokens are
/// normalised with.
#[derive(Debug)]
pub(crate) struct VaeEncoder {
    groups: usize,
    conv_in: Conv,
    down_blocks: Vec<DownBlock>,
    mid_resnets: [Resnet; 2],
    mid_attn: SpatialAttention,
    norm_out_w: DeviceBuffer,
    norm_out_b: DeviceBuffer,
    conv_out: Conv,
    quant: Conv,
    bn_mean: Vec<f32>,
    bn_var: Vec<f32>,
    bn_eps: f32,
}

/// The reference's clamp on the posterior's log-variance, applied before the
/// standard deviation is taken.
const LOGVAR_RANGE: (f32, f32) = (-30.0, 20.0);

/// Loads a downsampler's 3x3 stride-2 kernel as the zero-bordered 4x4 one.
#[expect(
    clippy::cast_possible_truncation,
    reason = "safetensors dims are element counts of conv kernels; they fit usize on any supported target"
)]
fn load_downsampler(store: &StStore, ops: &Ops, name: &str) -> Result<Conv, ImageGenError> {
    let weight_name = format!("{name}.weight");
    let shape = store.require(&weight_name)?.shape.clone();
    let [out_c, in_c, 3, 3] = shape.as_slice() else {
        return Err(ImageGenError::Checkpoint(format!(
            "'{weight_name}' is {shape:?}; a downsampler is a 3x3 kernel"
        )));
    };
    let (out_c, in_c) = (*out_c as usize, *in_c as usize);
    let taps = store.f32_vec(&weight_name)?;
    let mut wide = vec![0.0f32; out_c * in_c * 16];
    for plane in 0..out_c * in_c {
        for a in 0..3 {
            for b in 0..3 {
                wide[plane * 16 + (a + 1) * 4 + b + 1] = taps[plane * 9 + a * 3 + b];
            }
        }
    }
    Ok(Conv {
        w: ops.upload_f32("vae downsample weight", &wide)?,
        bias: Some(store.f32_widen(ops, "vae conv bias", &format!("{name}.bias"))?),
        out_c,
        in_c,
        k_h: 4,
        k_w: 4,
    })
}

impl VaeEncoder {
    /// Opens the encoder half of the VAE component at `path`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] when the config document, a tensor or a
    /// shape is not the one the encoder reads.
    pub(crate) fn load(path: &Path, ops: &Ops) -> Result<Self, ImageGenError> {
        let store = StStore::open(path)?;
        let config: VaeConfig =
            super::config::read_component_config(&path.join("config.json"), "vae")?;
        let groups = config.norm_num_groups.filter(|g| *g > 0).ok_or_else(|| {
            ImageGenError::Checkpoint(
                "vae/config.json carries no positive norm_num_groups: the group count is not \
                 derivable from tensor shapes"
                    .to_string(),
            )
        })?;
        let bn_mean = store.f32_vec("bn.running_mean")?;
        let bn_var = store.f32_vec("bn.running_var")?;
        if bn_mean.len() != bn_var.len() {
            return Err(ImageGenError::Checkpoint(
                "bn.running_mean and bn.running_var disagree on feature count".to_string(),
            ));
        }
        let depth = store.count_blocks("encoder.down_blocks.").ok_or_else(|| {
            ImageGenError::Checkpoint("VAE encoder has no down_blocks".to_string())
        })?;
        let mut down_blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let base = format!("encoder.down_blocks.{i}.");
            let count = store.count_blocks(&format!("{base}resnets.")).unwrap_or(0);
            if count == 0 {
                return Err(ImageGenError::Checkpoint(format!(
                    "{base}resnets has no blocks"
                )));
            }
            let resnets = (0..count)
                .map(|j| load_resnet(&store, ops, &format!("{base}resnets.{j}.")))
                .collect::<Result<Vec<_>, _>>()?;
            let downsample = if store.contains(&format!("{base}downsamplers.0.conv.weight")) {
                Some(load_downsampler(
                    &store,
                    ops,
                    &format!("{base}downsamplers.0.conv"),
                )?)
            } else {
                None
            };
            down_blocks.push(DownBlock {
                resnets,
                downsample,
            });
        }
        let (norm_out_w, norm_out_b) = load_norm(&store, ops, "encoder.conv_norm_out")?;
        Ok(Self {
            groups,
            conv_in: load_conv(&store, ops, "encoder.conv_in")?,
            down_blocks,
            mid_resnets: [
                load_resnet(&store, ops, "encoder.mid_block.resnets.0.")?,
                load_resnet(&store, ops, "encoder.mid_block.resnets.1.")?,
            ],
            mid_attn: load_attention(&store, ops, "encoder.mid_block.attentions.0.")?,
            norm_out_w,
            norm_out_b,
            conv_out: load_conv(&store, ops, "encoder.conv_out")?,
            quant: load_conv(&store, ops, "quant_conv")?,
            bn_mean,
            bn_var,
            bn_eps: config.batch_norm_eps.unwrap_or(DEFAULT_BN_EPS),
        })
    }

    /// The latent's channel count (half the moments' planes).
    pub(crate) fn latent_channels(&self) -> usize {
        self.quant.out_c / 2
    }

    /// The batch-norm statistics over the packed features, and their epsilon.
    pub(crate) fn batch_norm(&self) -> (&[f32], &[f32], f32) {
        (&self.bn_mean, &self.bn_var, self.bn_eps)
    }

    /// The posterior's moments for a `(3, height, width)` picture in `[0, 1]`:
    /// `(2 * latent_channels, height / 8, width / 8)`, mean planes first.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Backend`] for a picture whose size the encoder cannot
    /// halve three times, or a device failure.
    pub(crate) fn moments(
        &self,
        ops: &Ops,
        rgb: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        let halvings = self
            .down_blocks
            .iter()
            .filter(|b| b.downsample.is_some())
            .count();
        let unit = 1usize << halvings;
        if width == 0 || height == 0 || !width.is_multiple_of(unit) || !height.is_multiple_of(unit)
        {
            return Err(ImageGenError::Backend(format!(
                "a {width}x{height} picture is not a multiple of {unit} on both sides"
            )));
        }
        if rgb.len() != 3 * width * height {
            return Err(ImageGenError::Backend(format!(
                "the picture is {} values, a {width}x{height} RGB field is {}",
                rgb.len(),
                3 * width * height
            )));
        }
        // The reference encodes `image * 2 - 1`.
        let signed: Vec<f32> = rgb.iter().map(|v| v * 2.0 - 1.0).collect();
        let input = ops.upload_scratch("vae encode input", &signed)?;
        let mut field = Field {
            channels: 3,
            h: height,
            w: width,
        };
        let mut x = conv(ops, &self.conv_in, input.view(), 3, &field, "vae enc in")?;
        drop(input);
        field.channels = self.conv_in.out_c;

        for block in &self.down_blocks {
            for block_resnet in &block.resnets {
                x = resnet(
                    ops,
                    self.groups,
                    block_resnet,
                    &x,
                    &mut field,
                    "vae enc down",
                )?;
            }
            if let Some(down) = &block.downsample {
                x = Self::downsample(ops, down, &x, &mut field)?;
            }
        }

        x = resnet(
            ops,
            self.groups,
            &self.mid_resnets[0],
            &x,
            &mut field,
            "vae enc mid0",
        )?;
        x = attention(ops, self.groups, &self.mid_attn, &x, &field)?;
        x = resnet(
            ops,
            self.groups,
            &self.mid_resnets[1],
            &x,
            &mut field,
            "vae enc mid1",
        )?;

        let moments = self.head(ops, &x, &mut field)?;
        let spatial = field.h * field.w;
        let values = ops.download(
            "vae enc readback",
            moments.view(),
            self.quant.out_c * spatial,
        )?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(ImageGenError::Backend(
                "VAE encode produced a non-finite moment".to_owned(),
            ));
        }
        Ok(values)
    }

    /// `conv_norm_out`, `SiLU`, `conv_out` and `quant_conv`: the moments.
    fn head<'o>(
        &self,
        ops: &'o Ops,
        x: &DeviceBuffer,
        field: &mut Field,
    ) -> Result<crate::diffusion::ops::Scratch<'o>, ImageGenError> {
        let spatial = field.h * field.w;
        let normed = ops.alloc_f32("vae enc out norm", field.channels * spatial)?;
        ops.group_norm(&GroupNormCall {
            step: "vae enc out norm",
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
            "vae enc out silu",
            normed.view(),
            normed.view(),
            field.channels * spatial,
        )?;
        let out = conv(
            ops,
            &self.conv_out,
            normed.view(),
            field.channels,
            field,
            "vae enc out",
        )?;
        field.channels = self.conv_out.out_c;
        conv(
            ops,
            &self.quant,
            out.view(),
            field.channels,
            field,
            "vae quant",
        )
    }

    /// The stride-2 downsampler: the zero-bordered 4x4 kernel over a
    /// symmetric pad of one (see the module doc).
    fn downsample<'o>(
        ops: &'o Ops,
        down: &Conv,
        x: &DeviceBuffer,
        field: &mut Field,
    ) -> Result<crate::diffusion::ops::Scratch<'o>, ImageGenError> {
        let (out_h, out_w) = (field.h / 2, field.w / 2);
        let y = ops.alloc_f32("vae enc downsample", down.out_c * out_h * out_w)?;
        ops.conv2d(&Conv2dCall {
            step: "vae enc downsample",
            out: y.view(),
            input: x.view(),
            weight: down.w.view(),
            bias: down.bias.as_ref().map(DeviceBuffer::view),
            batch: 1,
            in_c: field.channels,
            in_h: field.h,
            in_w: field.w,
            out_c: down.out_c,
            out_h,
            out_w,
            k_h: 4,
            k_w: 4,
            stride: 2,
            pad: 1,
        })?;
        *field = Field {
            channels: down.out_c,
            h: out_h,
            w: out_w,
        };
        Ok(y)
    }

    /// A posterior draw: `mean + exp(logvar / 2) * noise`, with the
    /// reference's clamp on the log-variance. `noise` is standard normal, one
    /// value per latent element, `(latent_channels, height / 8, width / 8)`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Backend`] when the noise is not the latent's size.
    pub(crate) fn sample(moments: &[f32], noise: &[f32]) -> Result<Vec<f32>, ImageGenError> {
        let half = moments.len() / 2;
        if noise.len() != half || moments.len() != 2 * half {
            return Err(ImageGenError::Backend(format!(
                "{} noise values for a {half}-value latent",
                noise.len()
            )));
        }
        let (mean, logvar) = moments.split_at(half);
        Ok(mean
            .iter()
            .zip(logvar)
            .zip(noise)
            .map(|((m, lv), n)| m + (0.5 * lv.clamp(LOGVAR_RANGE.0, LOGVAR_RANGE.1)).exp() * n)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::VaeEncoder;

    #[test]
    fn the_sample_is_the_mean_plus_the_scaled_noise() {
        // Two latent values: log-variances 0 and ln 4 give standard deviations
        // 1 and 2.
        let moments = [1.0, -1.0, 0.0, 4.0f32.ln()];
        let sample = VaeEncoder::sample(&moments, &[0.5, 0.5]).expect("matching sizes");
        assert!((sample[0] - 1.5).abs() < 1e-6);
        assert!((sample[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn the_log_variance_is_clamped_before_the_exponential() {
        let moments = [0.0, 100.0];
        let sample = VaeEncoder::sample(&moments, &[1.0]).expect("matching sizes");
        assert!((sample[0] - 10.0f32.exp()).abs() / 10.0f32.exp() < 1e-6);
    }

    #[test]
    fn noise_of_the_wrong_size_is_refused() {
        assert!(VaeEncoder::sample(&[0.0, 0.0, 0.0, 0.0], &[0.0]).is_err());
    }
}

//! The Qwen-Image-2.1 VAE encoder: an RGBA picture in, the latent tokens
//! the transformer reads out.
//!
//! An edit's reference picture enters the transformer as the tokens the
//! VAE would decode back to it, so the encoder is the decoder's mirror
//! over the same building blocks ([`super::vae`]): `encoder.conv1`, five
//! down blocks of two resnets each, the mid block (resnet, attention,
//! resnet), then `encoder.head` and the 1x1 quant convolution `conv1`.
//! Every block but the last halves the field with a stride-two 3x3
//! convolution over a field zero-padded on its right and bottom edges only,
//! and adds a parameter-free shortcut of the block's input: its
//! `(time, 2, 2)` neighbourhoods folded into channels and averaged in
//! groups (`Ops::avg_down_add`). The picture is a single frame, so where a
//! block also folds time the frame sits behind a zero one and the
//! downsampler's `time_conv`, which acts only on a second frame, is never
//! read - as in the decoder.
//!
//! The quant convolution emits a distribution's mean and log-variance; the
//! reference takes the mode, which is the mean, and normalizes it with the
//! config's latent statistics. Both are folded into the convolution at
//! load: only the mean's rows are kept, scaled and shifted so the output is
//! already the normalized latent ([`fold_latent_norm`]).

use shiso_core::{DeviceBuffer, DeviceView};
use shiso_vision::generator::ImageGenError;

use super::vae::{
    Attention, Conv, ConvLayout, Field, LatentGrid, QwenVae, QwenVaeDesc, Resnet, SPATIAL_FACTOR,
    Stream, VaeConfig, load_conv, load_gamma, load_resnet,
};
use crate::diffusion::ops::{
    AvgDownCall, ChannelRmsCall, Conv2dCall, Ops, Scratch, VaeConvCall, VaeConvInput,
    avg_down_group,
};
use crate::diffusion::weights::StStore;

/// An encoder stage the device tests read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VaeEncodeTap {
    /// Down block `i`'s output, shortcut included.
    Down(usize),
    /// Mid-block output.
    Mid,
}

/// A stage observer: called with the stage, its device field and its
/// element count, in issue order.
pub(crate) type EncodeTap<'t> =
    dyn FnMut(VaeEncodeTap, DeviceView<'_>, usize) -> Result<(), ImageGenError> + 't;

/// A down block's downsampler: a stride-two 3x3 convolution; the block's
/// averaging shortcut has this temporal factor.
#[derive(Debug)]
struct Downsampler {
    conv: Conv,
    factor_t: usize,
}

#[derive(Debug)]
struct DownBlock {
    resnets: Vec<Resnet>,
    downsampler: Option<Downsampler>,
}

/// The encoder's channel widths, for the working-set estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodePlan {
    /// `(in_c, out_c, downsamples)` per down block.
    pub blocks: Vec<(usize, usize, bool)>,
    /// Width of the mid block.
    pub mid_c: usize,
    /// Latent channels.
    pub z_dim: usize,
}

impl EncodePlan {
    /// The plan of the encoder in `store`, from its tensor shapes alone:
    /// nothing is imported, so a host can size an encode before it decides
    /// to load one.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] when the file holds no encoder, or one
    /// whose blocks are not shaped as [`QwenVaeEncoder::load`] reads them.
    pub(crate) fn read(store: &StStore, z_dim: usize) -> Result<Self, ImageGenError> {
        let axis = |name: &str, i: usize| -> Result<usize, ImageGenError> {
            store
                .require(name)?
                .shape
                .get(i)
                .and_then(|&d| usize::try_from(d).ok())
                .ok_or_else(|| ImageGenError::Checkpoint(format!("'{name}' has no axis {i}")))
        };
        let depth = store
            .count_blocks("encoder.downsamples.")
            .ok_or_else(|| ImageGenError::Checkpoint("the VAE file holds no encoder".to_owned()))?;
        let mut blocks = Vec::with_capacity(depth);
        let mut width = axis("encoder.conv1.weight", 0)?;
        for i in 0..depth {
            let base = format!("encoder.downsamples.{i}.downsamples");
            let entries = store
                .count_blocks(&format!("{base}."))
                .ok_or_else(|| ImageGenError::Checkpoint(format!("{base} is empty")))?;
            let out_c = axis(&format!("{base}.0.residual.2.weight"), 0)?;
            let downsamples = store.contains(&format!("{base}.{}.resample.1.weight", entries - 1));
            blocks.push((width, out_c, downsamples));
            width = out_c;
        }
        Ok(Self {
            blocks,
            mid_c: width,
            z_dim,
        })
    }

    /// An upper bound on the bytes of working memory an encode to `grid`
    /// allocates through the scratch pool.
    ///
    /// As in the decoder's estimate, every buffer the encode asks the pool
    /// for is counted once with no credit for reuse across sizes. Per down
    /// block that is the block's kept input, two convolution outputs and a
    /// residual stream at the block's width, the per-pixel norms, and the
    /// downsampled output.
    pub(crate) fn working_set_bytes(&self, grid: LatentGrid) -> u64 {
        let mut spatial = (grid.cells() * SPATIAL_FACTOR * SPATIAL_FACTOR) as u128;
        let mut elements: u128 = spatial * FIRST_CONV_PLANES as u128;
        for &(in_c, out_c, downsamples) in &self.blocks {
            let (in_c, out_c) = (in_c as u128, out_c as u128);
            elements += spatial * (in_c + 3 * out_c + 2);
            if downsamples {
                spatial /= 4;
                elements += spatial * out_c;
            }
        }
        // The latent stage: the mid resnets' stream plus one convolution
        // output, the attention's qkv, transposed operands, context in both
        // layouts and projection, its score block, then the head and the
        // tokens in both layouts.
        let (s, c, z) = (spatial, self.mid_c as u128, self.z_dim as u128);
        let score_rows = (super::vae::ATTN_SCORE_BUDGET as u128 / s.max(1)).clamp(1, s.max(1));
        elements += s * (c * (1 + 1 + 3 + 4 + 1) + 2 + 2 * z + 2 * z) + score_rows * s;
        u64::try_from(elements.saturating_mul(4)).unwrap_or(u64::MAX)
    }
}

/// The loaded encoder.
#[derive(Debug)]
pub(crate) struct QwenVaeEncoder {
    z_dim: usize,
    conv_in: Conv,
    down_blocks: Vec<DownBlock>,
    mid_resnets: [Resnet; 2],
    mid_attn: Attention,
    gamma_out: DeviceBuffer,
    conv_out: Conv,
    /// The quant convolution's mean rows with the latent normalization
    /// folded in.
    quant: Conv,
}

/// Input planes the first convolution is given: the picture's four and
/// twelve of zeros.
///
/// The tiled convolution stages its reduction sixteen taps at a time, and a
/// 3x3 kernel over four channels is 36 taps, so the stored convolution ran
/// on the per-pixel kernel: 714 ms of a 1024x1024 encode on an M4, a third
/// of the whole encoder. Sixteen channels make it 144 taps, a shape the tile
/// takes; the added weights and planes are zero, so every added product is
/// an exact zero.
pub(crate) const FIRST_CONV_PLANES: usize = 16;

/// A stored `[out_c][in_c][taps]` convolution weight widened to `to` input
/// channels: each output keeps its own values in place and the channels
/// past `in_c` are zero.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] when `w` is not a whole number of taps per
/// `(out_c, in_c)` pair or `to` is narrower than `in_c`.
pub(crate) fn pad_in_channels(
    w: &[f32],
    out_c: usize,
    in_c: usize,
    to: usize,
) -> Result<Vec<f32>, ImageGenError> {
    let pairs = out_c * in_c;
    if pairs == 0 || !w.len().is_multiple_of(pairs) || to < in_c {
        return Err(ImageGenError::Checkpoint(format!(
            "a convolution weight of {} values cannot be widened from {out_c}x{in_c} to {to} \
             input channels",
            w.len()
        )));
    }
    let taps = w.len() / pairs;
    let mut padded = vec![0f32; out_c * to * taps];
    for (stored, wide) in w
        .chunks_exact(in_c * taps)
        .zip(padded.chunks_exact_mut(to * taps))
    {
        wide[..stored.len()].copy_from_slice(stored);
    }
    Ok(padded)
}

/// The picture's interleaved RGBA8 bytes as the channel-first `(4, plane)`
/// f32 field the encoder reads: `2 * v / 255 - 1`, as the reference
/// processor writes it.
///
/// # Errors
/// [`ImageGenError::Backend`] when `rgba` is not `4 * plane` bytes.
pub(crate) fn rgba_field(rgba: &[u8], plane: usize) -> Result<Vec<f32>, ImageGenError> {
    if rgba.len() != 4 * plane {
        return Err(ImageGenError::Backend(format!(
            "an RGBA picture of {plane} pixels holds {} bytes, expected {}",
            rgba.len(),
            4 * plane
        )));
    }
    let mut field = vec![0f32; 4 * plane];
    for (p, pixel) in rgba.chunks_exact(4).enumerate() {
        for (c, &v) in pixel.iter().enumerate() {
            field[c * plane + p] = 2.0 * (f32::from(v) / 255.0) - 1.0;
        }
    }
    Ok(field)
}

/// Folds "take the mean, then `(mean - latents_mean) / latents_std`" into
/// the quant convolution `y = W x + b` (`w` is `[out_c][in_c]`, its first
/// `mean.len()` outputs the distribution's mean): returns `W'` and `b'` of
/// those rows alone with `W' = diag(1 / std) W` and `b' = (b - mean) / std`.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] when the statistics do not fit the
/// convolution or a deviation is not positive.
pub(crate) fn fold_latent_norm(
    w: &[f32],
    b: &[f32],
    mean: &[f32],
    std: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), ImageGenError> {
    let z = mean.len();
    let fits = z > 0
        && std.len() == z
        && b.len() >= z
        && w.len().is_multiple_of(b.len())
        && std.iter().all(|&s| s > 0.0);
    if !fits {
        return Err(ImageGenError::Checkpoint(format!(
            "latent statistics ({} means, {} deviations) do not fit quant_conv ({} weights, \
             {} outputs), or a deviation is not positive",
            mean.len(),
            std.len(),
            w.len(),
            b.len()
        )));
    }
    let in_c = w.len() / b.len();
    let mut wf = Vec::with_capacity(z * in_c);
    let mut bf = Vec::with_capacity(z);
    for (o, row) in w.chunks_exact(in_c).take(z).enumerate() {
        wf.extend(row.iter().map(|&v| v / std[o]));
        bf.push((b[o] - mean[o]) / std[o]);
    }
    Ok((wf, bf))
}

impl QwenVaeEncoder {
    /// Loads the encoder half of the VAE at `desc`.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] for a missing or misshapen tensor, a
    /// config this VAE does not implement, an input other than RGBA, or
    /// latent statistics that do not fit; [`ImageGenError::Backend`] when
    /// an upload fails.
    #[expect(
        clippy::too_many_lines,
        reason = "one straight pass over the checkpoint's blocks, each check beside the tensor it guards"
    )]
    pub(crate) fn load(desc: &QwenVaeDesc<'_>, ops: &Ops) -> Result<Self, ImageGenError> {
        let config = VaeConfig::read(desc.config)?;
        let store = StStore::open(desc.weights)?;
        let z_dim = config.z_dim;
        let refuse = |message: String| Err(ImageGenError::Checkpoint(message));

        let conv_in = load_conv(&store, ops, "encoder.conv1", ConvLayout::Stored)?;
        if conv_in.in_c != 4 {
            return refuse(format!(
                "encoder.conv1 reads {} channels; the RGBA picture has 4",
                conv_in.in_c
            ));
        }
        let conv_in = Conv {
            w: ops.upload_f32(
                "vae conv weight",
                &pad_in_channels(
                    &store.f32_vec("encoder.conv1.weight")?,
                    conv_in.out_c,
                    conv_in.in_c,
                    FIRST_CONV_PLANES,
                )?,
            )?,
            in_c: FIRST_CONV_PLANES,
            ..conv_in
        };

        let depth = store.count_blocks("encoder.downsamples.").ok_or_else(|| {
            ImageGenError::Checkpoint("the VAE encoder has no down blocks".to_owned())
        })?;
        if config.temperal_downsample.len() + 1 != depth {
            return refuse(format!(
                "vae config lists {} temporal factors for {depth} down blocks (expected {})",
                config.temperal_downsample.len(),
                depth.saturating_sub(1)
            ));
        }
        let mut down_blocks = Vec::with_capacity(depth);
        let mut width = conv_in.out_c;
        for i in 0..depth {
            let base = format!("encoder.downsamples.{i}.downsamples");
            let entries = store
                .count_blocks(&format!("{base}."))
                .ok_or_else(|| ImageGenError::Checkpoint(format!("{base} is empty")))?;
            let mut resnets = Vec::with_capacity(entries);
            let mut downsampler = None;
            for j in 0..entries {
                let entry = format!("{base}.{j}");
                if store.contains(&format!("{entry}.resample.1.weight")) {
                    if j + 1 != entries {
                        return refuse(format!(
                            "{entry} is a downsampler before the block's last entry"
                        ));
                    }
                    let temporal = config.temperal_downsample.get(i).copied().unwrap_or(false);
                    downsampler = Some(Downsampler {
                        conv: load_conv(
                            &store,
                            ops,
                            &format!("{entry}.resample.1"),
                            ConvLayout::Stored,
                        )?,
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
                    return refuse(format!(
                        "{base} resnet reads {} channels after a {c}-channel stage",
                        r.conv1.in_c
                    ));
                }
                c = r.conv1.out_c;
            }
            if resnets.is_empty() {
                return refuse(format!("{base} has no resnets"));
            }
            if let Some(down) = &downsampler
                && (down.conv.in_c != c || down.conv.out_c != c || down.conv.k != 3)
            {
                return refuse(format!(
                    "{base} downsampler is a {}x{} kernel mapping {} -> {} channels on a \
                     {c}-channel field",
                    down.conv.k, down.conv.k, down.conv.in_c, down.conv.out_c
                ));
            }
            if (i + 1 == depth) == downsampler.is_some() {
                return refuse(format!(
                    "{base}: every down block but the last downsamples, and the last does not"
                ));
            }
            // The shortcut's shape contract, checked now so an encode never
            // discovers it.
            let (factor_s, factor_t) = downsampler.as_ref().map_or((1, 1), |d| (2, d.factor_t));
            avg_down_group(block_in, c, factor_s, factor_t)?;
            width = c;
            down_blocks.push(DownBlock {
                resnets,
                downsampler,
            });
        }
        let halvings = down_blocks
            .iter()
            .filter(|b| b.downsampler.is_some())
            .count();
        if 1usize << halvings != SPATIAL_FACTOR {
            return refuse(format!(
                "the VAE encoder halves the picture {halvings} times; latent cells are \
                 {SPATIAL_FACTOR} pixels wide"
            ));
        }

        let mid_resnets = [
            load_resnet(&store, ops, "encoder.middle.0")?,
            load_resnet(&store, ops, "encoder.middle.2")?,
        ];
        let mid_attn = Attention {
            gamma: load_gamma(&store, ops, "encoder.middle.1.norm.gamma", width)?,
            qkv: load_conv(&store, ops, "encoder.middle.1.to_qkv", ConvLayout::TapMajor)?,
            proj: load_conv(&store, ops, "encoder.middle.1.proj", ConvLayout::Stored)?,
        };
        if mid_resnets
            .iter()
            .any(|r| r.conv1.in_c != width || r.conv1.out_c != width)
            || mid_attn.qkv.in_c != width
            || mid_attn.qkv.out_c != 3 * width
            || mid_attn.qkv.k != 1
            || mid_attn.proj.in_c != width
            || mid_attn.proj.out_c != width
            || mid_attn.proj.k != 1
        {
            return refuse(format!(
                "encoder.middle does not keep the {width}-channel width of the last down block"
            ));
        }

        let conv_out = load_conv(&store, ops, "encoder.head.2", ConvLayout::TapMajor)?;
        let gamma_out = load_gamma(&store, ops, "encoder.head.0.gamma", width)?;
        if conv_out.in_c != width || conv_out.out_c != 2 * z_dim {
            return refuse(format!(
                "encoder.head.2 maps {} -> {} channels; the mid block emits {width} and the \
                 distribution takes {}",
                conv_out.in_c,
                conv_out.out_c,
                2 * z_dim
            ));
        }

        // quant_conv, cut to the mean's rows with the normalization folded in.
        let q_name = "conv1";
        let q_shape = store.require(&format!("{q_name}.weight"))?.shape.clone();
        let wide = u64::try_from(2 * z_dim).unwrap_or(u64::MAX);
        if !(q_shape.len() >= 2
            && q_shape.iter().skip(2).all(|&d| d == 1)
            && q_shape[0] == wide
            && q_shape[1] == wide)
        {
            return refuse(format!(
                "quant_conv '{q_name}.weight' has shape {q_shape:?}, expected a 1x1 \
                 {wide} -> {wide} kernel"
            ));
        }
        if config.latents_mean.len() != z_dim {
            return refuse(format!(
                "vae config carries {} latent means for {z_dim} latent channels",
                config.latents_mean.len()
            ));
        }
        let (q_w, q_b) = fold_latent_norm(
            &store.f32_vec(&format!("{q_name}.weight"))?,
            &store.f32_vec(&format!("{q_name}.bias"))?,
            &config.latents_mean,
            &config.latents_std,
        )?;
        let quant = Conv {
            w: ops.upload_f32("vae quant weight", &q_w)?,
            bias: Some(ops.upload_f32("vae quant bias", &q_b)?),
            out_c: z_dim,
            in_c: 2 * z_dim,
            k: 1,
        };

        Ok(Self {
            z_dim,
            conv_in,
            down_blocks,
            mid_resnets,
            mid_attn,
            gamma_out,
            conv_out,
            quant,
        })
    }

    /// The loaded encoder's channel plan, for the tests to hold against
    /// the one [`EncodePlan::read`] derives from the file's shapes.
    #[cfg(test)]
    pub(crate) fn plan(&self) -> EncodePlan {
        EncodePlan {
            blocks: self
                .down_blocks
                .iter()
                .map(|b| {
                    let out_c = b.resnets.last().map_or(0, |r| r.conv1.out_c);
                    (b.resnets[0].conv1.in_c, out_c, b.downsampler.is_some())
                })
                .collect(),
            mid_c: self.conv_out.in_c,
            z_dim: self.z_dim,
        }
    }

    /// Encodes an interleaved RGBA8 picture of `16 * grid` pixels into the
    /// transformer's latent tokens `(grid.cells(), z_dim)` f32, row-major
    /// cells, already normalized.
    ///
    /// # Errors
    /// [`ImageGenError::Backend`] for a picture that is not the grid's size
    /// or a device failure.
    pub(crate) fn encode<'o>(
        &self,
        ops: &'o Ops,
        rgba: &[u8],
        grid: LatentGrid,
    ) -> Result<Scratch<'o>, ImageGenError> {
        self.encode_with(ops, rgba, grid, None)
    }

    /// [`Self::encode`] with an optional observer of intermediate stages.
    ///
    /// # Errors
    /// As [`Self::encode`].
    #[expect(
        clippy::too_many_lines,
        reason = "the encoder's stage graph, straight-line by construction"
    )]
    pub(crate) fn encode_with<'o>(
        &self,
        ops: &'o Ops,
        rgba: &[u8],
        grid: LatentGrid,
        mut tap: Option<&mut EncodeTap<'_>>,
    ) -> Result<Scratch<'o>, ImageGenError> {
        if grid.cells() == 0 {
            return Err(ImageGenError::Backend(format!(
                "VAE encode to an empty {}x{} latent grid",
                grid.w, grid.h
            )));
        }
        let (height, width) = grid.image_size();
        let mut field = Field {
            c: FIRST_CONV_PLANES,
            h: height,
            w: width,
        };
        // The picture's four planes, then the zero planes the widened
        // first convolution reads beside them.
        let mut planes = rgba_field(rgba, field.spatial())?;
        planes.resize(field.len(), 0.0);
        let pixels = ops.upload_scratch("vae pixels", &planes)?;
        drop(planes);
        let mut x = QwenVae::conv(ops, &self.conv_in, pixels.view(), field, "vae enc conv_in")?;
        drop(pixels);
        field.c = self.conv_in.out_c;

        // Down blocks. Every block keeps its input for the averaging
        // shortcut, so its first resnet must not add into it in place.
        for (i, block) in self.down_blocks.iter().enumerate() {
            let block_in = field;
            let step = "vae down";
            let mut y =
                QwenVae::resnet(ops, &block.resnets[0], Stream::Kept(&x), &mut field, step)?;
            for resnet in &block.resnets[1..] {
                y = QwenVae::resnet(ops, resnet, Stream::Owned(y), &mut field, step)?;
            }
            let (out, factor_s, factor_t) = match &block.downsampler {
                Some(down) => {
                    let out_field = Field {
                        c: down.conv.out_c,
                        h: field.h / 2,
                        w: field.w / 2,
                    };
                    let out = ops.alloc_f32("vae downsample", out_field.len())?;
                    // No padding and a stride of two over an even field:
                    // the taps that fall off the right and bottom edges
                    // read zero, which is the reference's one-sided pad.
                    ops.conv2d(&Conv2dCall {
                        step: "vae downsample",
                        out: out.view(),
                        input: y.view(),
                        weight: down.conv.w.view(),
                        bias: down.conv.bias.as_ref().map(DeviceBuffer::view),
                        batch: 1,
                        in_c: field.c,
                        in_h: field.h,
                        in_w: field.w,
                        out_c: out_field.c,
                        out_h: out_field.h,
                        out_w: out_field.w,
                        k_h: down.conv.k,
                        k_w: down.conv.k,
                        stride: 2,
                        pad: 0,
                    })?;
                    drop(y);
                    field = out_field;
                    (out, 2, down.factor_t)
                }
                None => (y, 1, 1),
            };
            ops.avg_down_add(&AvgDownCall {
                step: "vae avg-down",
                out: out.view(),
                input: x.view(),
                in_c: block_in.c,
                out_c: field.c,
                h: field.h,
                w: field.w,
                factor_s,
                factor_t,
            })?;
            x = out;
            if let Some(tap) = tap.as_mut() {
                tap(VaeEncodeTap::Down(i), x.view(), field.len())?;
            }
        }

        // Mid block.
        x = QwenVae::resnet(
            ops,
            &self.mid_resnets[0],
            Stream::Owned(x),
            &mut field,
            "vae enc mid 0",
        )?;
        x = QwenVae::attention(ops, &self.mid_attn, x, field)?;
        x = QwenVae::resnet(
            ops,
            &self.mid_resnets[1],
            Stream::Owned(x),
            &mut field,
            "vae enc mid 2",
        )?;
        if let Some(tap) = tap.as_mut() {
            tap(VaeEncodeTap::Mid, x.view(), field.len())?;
        }

        // Head: channel norm -> silu -> conv_out in one convolution, then
        // the folded quant convolution and the transpose to tokens.
        let step = "vae enc head";
        let spatial = field.spatial();
        let inv = ops.alloc_f32(step, spatial)?;
        ops.channel_rms_inv(&ChannelRmsCall {
            step,
            inv: inv.view(),
            input: x.view(),
            channels: field.c,
            spatial,
        })?;
        let moments = ops.alloc_f32(step, self.conv_out.out_c * spatial)?;
        ops.vae_conv(&VaeConvCall {
            step,
            out: moments.view(),
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
            out_c: self.conv_out.out_c,
            k: self.conv_out.k,
            accumulate: false,
        })?;
        drop((x, inv));
        field.c = self.conv_out.out_c;
        let latent = QwenVae::conv(ops, &self.quant, moments.view(), field, "vae quant")?;
        drop(moments);
        let tokens = ops.alloc_f32("vae pack", spatial * self.z_dim)?;
        ops.transpose(
            "vae pack",
            tokens.view(),
            latent.view(),
            self.z_dim,
            spatial,
        )?;
        Ok(tokens)
    }
}

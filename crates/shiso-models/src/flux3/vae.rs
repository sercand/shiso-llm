//! The FLUX.3 autoencoder: the encoder that turns reference pictures into
//! latent tokens and the decoder that turns the denoised tokens back into
//! pixels.
//!
//! The checkpoint (`ae.safetensors`) names its modules the way the
//! reference implementation does (`encoder.down.N.block.M`,
//! `decoder.up.N.block.M`, `mid.block_1`, `mid.attn_1`, `nin_shortcut`,
//! `norm_out`), not the way the FLUX.2 export does (`up_blocks`,
//! `resnets`, `conv_shortcut`, a `config.json` beside it), so this is its own
//! loader: the FLUX.2 one cannot find a single decoder tensor in this file.
//! Every width and depth is read from the tensor shapes; the group count
//! (32), both epsilons (group norm 1e-6, latent batch norm 1e-4) and the
//! latent patch (2x2) are the reference's constants, which no tensor carries.
//!
//! Everything runs in f32; the BF16 the file may store widens exactly.
//!
//! A latent token is the 128 features `c * 4 + pi * 2 + pj` of cell
//! `(li, lj)`: channel `c` of the 32-channel field at `(2 li + pi, 2 lj +
//! pj)`, batch-normalised with the checkpoint's running statistics.
//! Encoding keeps the first 32 of the encoder's 64 output channels (the
//! mean; the second half is the variance, unused at inference).
//!
//! The encoder's downsampling pads one zero on the right and bottom and then
//! convolves with stride 2 and no padding. The convolution kernels skip taps
//! that fall outside the field, which is that zero pad, so the call asks for
//! `pad = 0`, `stride = 2` and half-size outputs explicitly.

#![expect(
    clippy::many_single_char_names,
    reason = "the stages are written in the formulas' own letters: x, h, w, c, q, k, v"
)]

use std::path::{Path, PathBuf};

use shiso_core::DeviceBuffer;
use shiso_vision::generator::ImageGenError;

use crate::diffusion::ops::{
    Conv2dCall, GemmCall, GroupNormCall, LinWeight, Ops, Scratch, SliceExt, UpsampleCall,
};
use crate::diffusion::weights::{StStore, safetensors_files};

/// Groups of every group norm.
const GROUPS: usize = 32;
/// Group-norm epsilon.
const GN_EPS: f32 = 1e-6;
/// Epsilon of the latent batch norm.
const BN_EPS: f32 = 1e-4;
/// Latent channels before the 2x2 patch.
const LATENT_CHANNELS: usize = 32;
/// Scores the mid-block attention forms at once, in f32 elements (1 GiB).
const ATTN_SCORE_BUDGET: usize = 1 << 28;
/// Pixels per token side: 8x for the VAE, 2x for the patch.
pub(crate) const PIXELS_PER_TOKEN: usize = 16;

fn refuse(message: String) -> ImageGenError {
    ImageGenError::Checkpoint(message)
}

/// One convolution with its shape read from the weight tensor.
#[derive(Debug)]
struct Conv {
    w: DeviceBuffer,
    bias: DeviceBuffer,
    out_c: usize,
    in_c: usize,
    k: usize,
}

/// A group-norm gain and offset.
#[derive(Debug)]
struct Norm {
    w: DeviceBuffer,
    b: DeviceBuffer,
}

/// A residual block: two (norm, silu, conv) stages and an optional 1x1
/// shortcut when the width changes.
#[derive(Debug)]
struct Resnet {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

/// The mid block's single-head spatial attention.
#[derive(Debug)]
struct Attention {
    norm: Norm,
    q: Conv,
    k: Conv,
    v: Conv,
    out: Conv,
}

/// One resolution level: its residual blocks and the sampler leaving it
/// (a stride-2 conv going down, a nearest-2x conv going up).
#[derive(Debug)]
struct Level {
    blocks: Vec<Resnet>,
    sampler: Option<Conv>,
}

#[derive(Debug)]
struct Mid {
    block_1: Resnet,
    attn_1: Attention,
    block_2: Resnet,
}

#[derive(Debug)]
struct Encoder {
    conv_in: Conv,
    levels: Vec<Level>,
    mid: Mid,
    norm_out: Norm,
    conv_out: Conv,
    quant_conv: Conv,
}

#[derive(Debug)]
struct Decoder {
    post_quant_conv: Conv,
    conv_in: Conv,
    mid: Mid,
    /// Indexed by level; the decoder runs the last one first.
    levels: Vec<Level>,
    norm_out: Norm,
    conv_out: Conv,
}

/// The loaded autoencoder.
#[derive(Debug)]
pub(crate) struct Vae {
    encoder: Encoder,
    decoder: Decoder,
    bn_mean: Vec<f32>,
    bn_std: Vec<f32>,
}

/// The field a stage is carrying: channels and spatial size.
#[derive(Debug, Clone, Copy)]
struct Field {
    c: usize,
    h: usize,
    w: usize,
}

impl Field {
    fn len(self) -> usize {
        self.c * self.h * self.w
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "tensor dimensions of convolution kernels fit usize on every supported target"
)]
fn load_conv(store: &StStore, ops: &Ops, name: &str) -> Result<Conv, ImageGenError> {
    let weight_name = format!("{name}.weight");
    let entry = store.require(&weight_name)?;
    let &[out_c, in_c, k_h, k_w] = entry.shape.as_slice() else {
        return Err(refuse(format!(
            "'{weight_name}' has shape {:?}, but a convolution kernel has four axes",
            entry.shape
        )));
    };
    if k_h != k_w {
        return Err(refuse(format!(
            "'{weight_name}' has a {k_h}x{k_w} kernel; only square kernels are read"
        )));
    }
    Ok(Conv {
        w: store.f32_widen(ops, "vae conv weight", &weight_name)?,
        bias: store.f32_widen(ops, "vae conv bias", &format!("{name}.bias"))?,
        out_c: out_c as usize,
        in_c: in_c as usize,
        k: k_h as usize,
    })
}

fn load_norm(store: &StStore, ops: &Ops, name: &str) -> Result<Norm, ImageGenError> {
    Ok(Norm {
        w: store.f32_widen(ops, "vae norm gain", &format!("{name}.weight"))?,
        b: store.f32_widen(ops, "vae norm offset", &format!("{name}.bias"))?,
    })
}

fn load_resnet(store: &StStore, ops: &Ops, base: &str) -> Result<Resnet, ImageGenError> {
    let shortcut = format!("{base}.nin_shortcut");
    Ok(Resnet {
        norm1: load_norm(store, ops, &format!("{base}.norm1"))?,
        conv1: load_conv(store, ops, &format!("{base}.conv1"))?,
        norm2: load_norm(store, ops, &format!("{base}.norm2"))?,
        conv2: load_conv(store, ops, &format!("{base}.conv2"))?,
        shortcut: if store.contains(&format!("{shortcut}.weight")) {
            Some(load_conv(store, ops, &shortcut)?)
        } else {
            None
        },
    })
}

fn load_mid(store: &StStore, ops: &Ops, base: &str) -> Result<Mid, ImageGenError> {
    let attn = format!("{base}.attn_1");
    Ok(Mid {
        block_1: load_resnet(store, ops, &format!("{base}.block_1"))?,
        attn_1: Attention {
            norm: load_norm(store, ops, &format!("{attn}.norm"))?,
            q: load_conv(store, ops, &format!("{attn}.q"))?,
            k: load_conv(store, ops, &format!("{attn}.k"))?,
            v: load_conv(store, ops, &format!("{attn}.v"))?,
            out: load_conv(store, ops, &format!("{attn}.proj_out"))?,
        },
        block_2: load_resnet(store, ops, &format!("{base}.block_2"))?,
    })
}

fn load_levels(
    store: &StStore,
    ops: &Ops,
    base: &str,
    sampler: &str,
) -> Result<Vec<Level>, ImageGenError> {
    let levels = store
        .count_blocks(&format!("{base}."))
        .ok_or_else(|| refuse(format!("no tensor named '{base}.<n>.*' in the VAE")))?;
    (0..levels)
        .map(|l| {
            let level = format!("{base}.{l}");
            let blocks = store
                .count_blocks(&format!("{level}.block."))
                .ok_or_else(|| refuse(format!("'{level}' has no residual blocks")))?;
            let sampler_name = format!("{level}.{sampler}.conv");
            Ok(Level {
                blocks: (0..blocks)
                    .map(|b| load_resnet(store, ops, &format!("{level}.block.{b}")))
                    .collect::<Result<_, _>>()?,
                sampler: if store.contains(&format!("{sampler_name}.weight")) {
                    Some(load_conv(store, ops, &sampler_name)?)
                } else {
                    None
                },
            })
        })
        .collect()
}

/// The VAE weights file: `path` itself, or the only `.safetensors` in the
/// directory it names.
fn weights_file(path: &Path) -> Result<PathBuf, ImageGenError> {
    if path.is_dir() {
        let files = safetensors_files(path)?;
        match files.as_slice() {
            [one] => Ok(one.clone()),
            [] => Err(refuse(format!(
                "no .safetensors VAE weights in {}",
                path.display()
            ))),
            _ => Err(refuse(format!(
                "{} holds several .safetensors files; name the VAE file itself",
                path.display()
            ))),
        }
    } else {
        Ok(path.to_path_buf())
    }
}

impl Vae {
    /// Opens the autoencoder at `path` (`ae.safetensors`, or a directory
    /// holding it) and uploads every weight as f32.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] naming the missing tensor or the
    /// disagreeing width.
    pub(crate) fn load(path: &Path, ops: &Ops) -> Result<Self, ImageGenError> {
        let file = weights_file(path)?;
        let store = StStore::open(&file)?;
        let encoder = Encoder {
            conv_in: load_conv(&store, ops, "encoder.conv_in")?,
            levels: load_levels(&store, ops, "encoder.down", "downsample")?,
            mid: load_mid(&store, ops, "encoder.mid")?,
            norm_out: load_norm(&store, ops, "encoder.norm_out")?,
            conv_out: load_conv(&store, ops, "encoder.conv_out")?,
            quant_conv: load_conv(&store, ops, "encoder.quant_conv")?,
        };
        let decoder = Decoder {
            post_quant_conv: load_conv(&store, ops, "decoder.post_quant_conv")?,
            conv_in: load_conv(&store, ops, "decoder.conv_in")?,
            mid: load_mid(&store, ops, "decoder.mid")?,
            levels: load_levels(&store, ops, "decoder.up", "upsample")?,
            norm_out: load_norm(&store, ops, "decoder.norm_out")?,
            conv_out: load_conv(&store, ops, "decoder.conv_out")?,
        };
        let bn_mean = store.f32_vec("bn.running_mean")?;
        let bn_var = store.f32_vec("bn.running_var")?;
        if bn_mean.len() != bn_var.len() || bn_mean.len() != LATENT_CHANNELS * 4 {
            return Err(refuse(format!(
                "bn.running_mean has {} entries and bn.running_var {}; the latent batch norm \
                 covers {} packed features",
                bn_mean.len(),
                bn_var.len(),
                LATENT_CHANNELS * 4
            )));
        }
        let vae = Self {
            encoder,
            decoder,
            bn_std: bn_var.iter().map(|v| (v + BN_EPS).sqrt()).collect(),
            bn_mean,
        };
        vae.check_widths()?;
        Ok(vae)
    }

    /// Cross-checks the widths the two halves hand each other, so a file of
    /// another autoencoder is refused at load, not at the first picture.
    fn check_widths(&self) -> Result<(), ImageGenError> {
        let (e, d) = (&self.encoder, &self.decoder);
        let want = |what: &str, got: usize, expected: usize| {
            if got == expected {
                Ok(())
            } else {
                Err(refuse(format!(
                    "the VAE's {what} is {got} wide, but this build reads {expected}"
                )))
            }
        };
        want("encoder.conv_in input", e.conv_in.in_c, 3)?;
        want(
            "encoder.quant_conv output",
            e.quant_conv.out_c,
            2 * LATENT_CHANNELS,
        )?;
        want(
            "decoder.post_quant_conv input",
            d.post_quant_conv.in_c,
            LATENT_CHANNELS,
        )?;
        want("decoder.conv_out output", d.conv_out.out_c, 3)?;
        want(
            "levels of the decoder against the encoder",
            d.levels.len(),
            e.levels.len(),
        )?;
        for conv in [&e.conv_in, &e.conv_out, &d.conv_in, &d.conv_out] {
            want("convolution kernel", conv.k, 3)?;
        }
        Ok(())
    }

    /// Pixel rows (and columns) one latent row (column) covers in the
    /// pictures this autoencoder handles: `2^(levels-1)` for the VAE, times
    /// the 2x2 patch.
    fn pixels_per_cell(&self) -> usize {
        (1usize << (self.encoder.levels.len() - 1)) * 2
    }

    /// Checks `width` x `height` are on this autoencoder's token grid.
    fn check_grid(&self, width: usize, height: usize) -> Result<(), ImageGenError> {
        let cell = self.pixels_per_cell();
        if width == 0 || height == 0 || !width.is_multiple_of(cell) || !height.is_multiple_of(cell)
        {
            return Err(ImageGenError::Backend(format!(
                "a {width}x{height} picture is off the VAE's {cell}-pixel token grid"
            )));
        }
        Ok(())
    }

    /// Encodes an RGB picture - planar `(3, height, width)` f32 in `[-1, 1]` -
    /// to `(height/16 * width/16, 128)` normalised latent tokens, row-major.
    ///
    /// # Errors
    /// [`ImageGenError::Backend`] for a picture off the 16-pixel grid and the
    /// device's own failures.
    pub(crate) fn encode(
        &self,
        ops: &Ops,
        rgb: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        self.check_grid(width, height)?;
        if rgb.len() != 3 * width * height {
            return Err(ImageGenError::Backend(format!(
                "a {width}x{height} picture has {} samples, not {}",
                rgb.len(),
                3 * width * height
            )));
        }
        let e = &self.encoder;
        let mut field = Field {
            c: 3,
            h: height,
            w: width,
        };
        let mut x: Scratch<'_> = ops.upload_scratch("vae picture", rgb)?;
        x = conv(ops, &e.conv_in, &x, &mut field)?;
        let last = e.levels.len() - 1;
        for (index, level) in e.levels.iter().enumerate() {
            for block in &level.blocks {
                x = resnet(ops, block, &x, &mut field)?;
            }
            if let (Some(down), true) = (&level.sampler, index != last) {
                x = downsample(ops, down, &x, &mut field)?;
            }
        }
        x = resnet(ops, &e.mid.block_1, &x, &mut field)?;
        x = attention(ops, &e.mid.attn_1, &x, field)?;
        x = resnet(ops, &e.mid.block_2, &x, &mut field)?;
        x = norm_silu(ops, &e.norm_out, &x, field)?;
        x = conv(ops, &e.conv_out, &x, &mut field)?;
        x = conv(ops, &e.quant_conv, &x, &mut field)?;
        let all = ops.download("vae encoded", x.view(), field.len())?;
        drop(x);
        // The first half of the channels is the mean.
        let (zc, plane) = (LATENT_CHANNELS, field.h * field.w);
        Ok(self.patchify(&all[..zc * plane], field.h, field.w))
    }

    /// `(32, 2 hl, 2 wl)` mean field to normalised `(hl * wl, 128)` tokens.
    fn patchify(&self, mean: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (hl, wl) = (h / 2, w / 2);
        let mut tokens = vec![0f32; hl * wl * LATENT_CHANNELS * 4];
        for li in 0..hl {
            for lj in 0..wl {
                let row = (li * wl + lj) * LATENT_CHANNELS * 4;
                for f in 0..LATENT_CHANNELS * 4 {
                    let (c, p) = (f / 4, f % 4);
                    let (pi, pj) = (p / 2, p % 2);
                    let raw = mean[(c * h + 2 * li + pi) * w + 2 * lj + pj];
                    tokens[row + f] = (raw - self.bn_mean[f]) / self.bn_std[f];
                }
            }
        }
        tokens
    }

    /// Decodes `(height/16 * width/16, 128)` latent tokens to a planar
    /// `(3, height, width)` f32 picture, unclamped (nominally `[-1, 1]`).
    ///
    /// # Errors
    /// [`ImageGenError::Backend`] for tokens that do not cover the picture or
    /// a picture off the 16-pixel grid, and the device's own failures.
    pub(crate) fn decode_raw(
        &self,
        ops: &Ops,
        tokens: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        self.check_grid(width, height)?;
        let cell = self.pixels_per_cell();
        let (hl, wl) = (height / cell, width / cell);
        if tokens.len() != hl * wl * LATENT_CHANNELS * 4 {
            return Err(ImageGenError::Backend(format!(
                "latent sequence is {} elements, expected {}",
                tokens.len(),
                hl * wl * LATENT_CHANNELS * 4
            )));
        }
        let d = &self.decoder;
        let z = self.unpatchify(tokens, hl, wl);
        let mut field = Field {
            c: LATENT_CHANNELS,
            h: 2 * hl,
            w: 2 * wl,
        };
        let mut x: Scratch<'_> = ops.upload_scratch("vae latent", &z)?;
        x = conv(ops, &d.post_quant_conv, &x, &mut field)?;
        x = conv(ops, &d.conv_in, &x, &mut field)?;
        x = resnet(ops, &d.mid.block_1, &x, &mut field)?;
        x = attention(ops, &d.mid.attn_1, &x, field)?;
        x = resnet(ops, &d.mid.block_2, &x, &mut field)?;
        // The deepest level runs first; every level but level 0 ends in an
        // upsampler: nearest 2x first, then the 3x3 conv at the new size.
        for (index, level) in d.levels.iter().enumerate().rev() {
            for block in &level.blocks {
                x = resnet(ops, block, &x, &mut field)?;
            }
            if let (Some(up), true) = (&level.sampler, index != 0) {
                x = upsample(ops, up, &x, &mut field)?;
            }
        }
        x = norm_silu(ops, &d.norm_out, &x, field)?;
        x = conv(ops, &d.conv_out, &x, &mut field)?;
        let out = ops.download("vae decoded", x.view(), field.len())?;
        if out.iter().any(|v| v.is_nan()) {
            return Err(ImageGenError::Backend(
                "VAE decode produced NaN in the decoded image field".to_owned(),
            ));
        }
        Ok(out)
    }

    /// Decodes tokens to planar `(3, height, width)` f32 in `[0, 255]`
    /// (clamped to the decoder's `[-1, 1]` first, then mapped).
    ///
    /// # Errors
    /// The refusals of [`Vae::decode_raw`].
    pub(crate) fn decode(
        &self,
        ops: &Ops,
        tokens: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        Ok(self
            .decode_raw(ops, tokens, width, height)?
            .iter()
            .map(|v| (v.clamp(-1.0, 1.0) + 1.0) * 127.5)
            .collect())
    }

    /// De-normalised, unpatchified `(32, 2 hl, 2 wl)` field of `(hl * wl,
    /// 128)` tokens: the inverse of [`Vae::patchify`].
    fn unpatchify(&self, tokens: &[f32], hl: usize, wl: usize) -> Vec<f32> {
        let (h, w) = (2 * hl, 2 * wl);
        let mut z = vec![0f32; LATENT_CHANNELS * h * w];
        for li in 0..hl {
            for lj in 0..wl {
                let row = (li * wl + lj) * LATENT_CHANNELS * 4;
                for f in 0..LATENT_CHANNELS * 4 {
                    let (c, p) = (f / 4, f % 4);
                    let (pi, pj) = (p / 2, p % 2);
                    z[(c * h + 2 * li + pi) * w + 2 * lj + pj] =
                        tokens[row + f] * self.bn_std[f] + self.bn_mean[f];
                }
            }
        }
        z
    }

    /// Bytes the weights occupy on the device (all f32).
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] when the file cannot be opened.
    pub(crate) fn weight_bytes(path: &Path) -> Result<u64, ImageGenError> {
        let store = StStore::open(&weights_file(path)?)?;
        Ok(store
            .names()
            .filter(|n| n.starts_with("encoder.") || n.starts_with("decoder."))
            .filter_map(|n| store.get(n))
            .map(|e| e.elements() * 4)
            .sum())
    }

    /// The widest field at full resolution, in channels: the largest
    /// activation either half carries.
    pub(crate) fn full_resolution_channels(&self) -> usize {
        let widest = |levels: &[Level]| {
            levels
                .first()
                .into_iter()
                .flat_map(|l| l.blocks.iter())
                .map(|b| b.conv1.out_c.max(b.conv1.in_c))
                .max()
                .unwrap_or(0)
        };
        widest(&self.decoder.levels).max(widest(&self.encoder.levels))
    }

    /// Estimated peak working bytes of encoding or decoding a picture of
    /// `pixels` pixels whose widest full-resolution field has `channels`:
    /// four such fields live at once (a residual block's input, two stages
    /// and its sum) plus the attention's score block, which is the smaller
    /// of the budget and the square of the mid-level's positions.
    pub(crate) fn working_set_bytes(channels: usize, pixels: usize) -> u64 {
        let field = (channels as u64) * (pixels as u64) * 4;
        let mid_positions = (pixels / 64).max(1) as u64;
        let scores = (mid_positions * mid_positions).min(ATTN_SCORE_BUDGET as u64) * 4;
        4 * field + scores
    }
}

/// A same-padding convolution; updates the field's channels.
fn conv<'o>(
    ops: &'o Ops,
    conv: &Conv,
    input: &DeviceBuffer,
    field: &mut Field,
) -> Result<Scratch<'o>, ImageGenError> {
    if conv.in_c != field.c {
        return Err(ImageGenError::Backend(format!(
            "a convolution reads {} channels but the field carries {}",
            conv.in_c, field.c
        )));
    }
    let out = ops.alloc_f32("vae conv", conv.out_c * field.h * field.w)?;
    ops.conv2d(&Conv2dCall {
        step: "vae conv",
        out: out.view(),
        input: input.view(),
        weight: conv.w.view(),
        bias: Some(conv.bias.view()),
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
    field.c = conv.out_c;
    Ok(out)
}

/// The encoder's downsample: one zero on the right and bottom, then a 3x3
/// stride-2 convolution without padding (see the module doc).
fn downsample<'o>(
    ops: &'o Ops,
    conv: &Conv,
    input: &DeviceBuffer,
    field: &mut Field,
) -> Result<Scratch<'o>, ImageGenError> {
    let (oh, ow) = (field.h / 2, field.w / 2);
    let out = ops.alloc_f32("vae downsample", conv.out_c * oh * ow)?;
    ops.conv2d(&Conv2dCall {
        step: "vae downsample",
        out: out.view(),
        input: input.view(),
        weight: conv.w.view(),
        bias: Some(conv.bias.view()),
        batch: 1,
        in_c: conv.in_c,
        in_h: field.h,
        in_w: field.w,
        out_c: conv.out_c,
        out_h: oh,
        out_w: ow,
        k_h: conv.k,
        k_w: conv.k,
        stride: 2,
        pad: 0,
    })?;
    *field = Field {
        c: conv.out_c,
        h: oh,
        w: ow,
    };
    Ok(out)
}

/// The decoder's upsample: nearest 2x first, then the 3x3 convolution at
/// the doubled size.
fn upsample<'o>(
    ops: &'o Ops,
    up: &Conv,
    input: &DeviceBuffer,
    field: &mut Field,
) -> Result<Scratch<'o>, ImageGenError> {
    let doubled = ops.alloc_f32("vae upsample", field.c * 4 * field.h * field.w)?;
    ops.upsample_nearest(&UpsampleCall {
        step: "vae upsample",
        out: doubled.view(),
        input: input.view(),
        batch: 1,
        channels: field.c,
        h: field.h,
        w: field.w,
    })?;
    field.h *= 2;
    field.w *= 2;
    conv(ops, up, &doubled, field)
}

fn group_norm(
    ops: &Ops,
    norm: &Norm,
    out: &DeviceBuffer,
    input: &DeviceBuffer,
    field: Field,
) -> Result<(), ImageGenError> {
    if !field.c.is_multiple_of(GROUPS) {
        return Err(ImageGenError::Backend(format!(
            "{} channels do not split into {GROUPS} groups",
            field.c
        )));
    }
    ops.group_norm(&GroupNormCall {
        step: "vae norm",
        out: out.view(),
        input: input.view(),
        weight: norm.w.view(),
        bias: norm.b.view(),
        batch: 1,
        groups: GROUPS,
        channels_per_group: field.c / GROUPS,
        spatial: field.h * field.w,
        eps: GN_EPS,
    })
}

/// `silu(group_norm(x))`.
fn norm_silu<'o>(
    ops: &'o Ops,
    norm: &Norm,
    input: &DeviceBuffer,
    field: Field,
) -> Result<Scratch<'o>, ImageGenError> {
    let out = ops.alloc_f32("vae norm", field.len())?;
    group_norm(ops, norm, &out, input, field)?;
    ops.silu("vae silu", out.view(), out.view(), field.len())?;
    Ok(out)
}

/// One residual block: `x + conv2(silu(norm2(conv1(silu(norm1(x))))))`, the
/// shortcut projecting `x` when the width changes.
fn resnet<'o>(
    ops: &'o Ops,
    block: &Resnet,
    x: &DeviceBuffer,
    field: &mut Field,
) -> Result<Scratch<'o>, ImageGenError> {
    let input = *field;
    let a = norm_silu(ops, &block.norm1, x, input)?;
    let mut mid = input;
    let b = conv(ops, &block.conv1, &a, &mut mid)?;
    drop(a);
    let c = norm_silu(ops, &block.norm2, &b, mid)?;
    drop(b);
    let mut out_field = mid;
    let d = conv(ops, &block.conv2, &c, &mut out_field)?;
    drop(c);
    let summed = ops.alloc_f32("vae resnet", out_field.len())?;
    if let Some(shortcut) = &block.shortcut {
        let mut sf = input;
        let s = conv(ops, shortcut, x, &mut sf)?;
        ops.add(
            "vae resnet",
            summed.view(),
            s.view(),
            d.view(),
            out_field.len(),
        )?;
    } else {
        ops.add(
            "vae resnet",
            summed.view(),
            x.view(),
            d.view(),
            out_field.len(),
        )?;
    }
    *field = out_field;
    Ok(summed)
}

/// Single-head spatial self-attention with a residual add. The norm feeds
/// q, k and v directly, with no activation between.
fn attention<'o>(
    ops: &'o Ops,
    attn: &Attention,
    x: &DeviceBuffer,
    field: Field,
) -> Result<Scratch<'o>, ImageGenError> {
    let channels = field.c;
    let spatial = field.h * field.w;
    let xn = ops.alloc_f32("vae attn norm", field.len())?;
    group_norm(ops, &attn.norm, &xn, x, field)?;
    let mut f = field;
    let q = conv(ops, &attn.q, &xn, &mut f)?;
    let k = conv(ops, &attn.k, &xn, &mut f)?;
    let v = conv(ops, &attn.v, &xn, &mut f)?;
    drop(xn);
    // softmax(q k^T / sqrt(c)) v as two GEMMs around a row softmax, a block
    // of query rows at a time: the scores of a 1024x1024 picture are 16384 x
    // 16384. The GEMM reads its right operand as (n, k) rows, so q and k go
    // to (spatial, channels) and v is read as it stands, (channels, spatial).
    let qt = ops.alloc_f32("vae attn qt", spatial * channels)?;
    let kt = ops.alloc_f32("vae attn kt", spatial * channels)?;
    ops.transpose("vae attn t", qt.view(), q.view(), channels, spatial)?;
    ops.transpose("vae attn t", kt.view(), k.view(), channels, spatial)?;
    drop(q);
    drop(k);
    let at = ops.alloc_f32("vae attn ctx", spatial * channels)?;
    let block = (ATTN_SCORE_BUDGET / spatial).clamp(1, spatial);
    let scores = ops.alloc_f32("vae attn scores", block * spatial)?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "a channel count is exact in f32"
    )]
    let scale = 1.0 / (channels as f32).sqrt();
    let row_bytes = |n: usize| (n * channels * 4) as u64;
    for r0 in (0..spatial).step_by(block) {
        let rows = block.min(spatial - r0);
        let slab = scores.view().cut(0, (rows * spatial * 4) as u64)?;
        ops.gemm(&GemmCall {
            step: "vae attn scores",
            out: slab,
            input: qt.view().cut(row_bytes(r0), row_bytes(rows))?,
            weight: LinWeight::f32(kt.view()),
            m: rows,
            n: spatial,
            k: channels,
        })?;
        ops.softmax_rows("vae attn softmax", slab, rows, spatial, scale)?;
        ops.gemm(&GemmCall {
            step: "vae attn values",
            out: at.view().cut(row_bytes(r0), row_bytes(rows))?,
            input: slab,
            weight: LinWeight::f32(v.view()),
            m: rows,
            n: channels,
            k: spatial,
        })?;
    }
    drop(scores);
    drop(qt);
    drop(kt);
    drop(v);
    let ctx = ops.alloc_f32("vae attn ctx ch", field.len())?;
    ops.transpose("vae attn t", ctx.view(), at.view(), spatial, channels)?;
    drop(at);
    let mut pf = field;
    let proj = conv(ops, &attn.out, &ctx, &mut pf)?;
    let out = ops.alloc_f32("vae attn residual", field.len())?;
    ops.add(
        "vae attn residual",
        out.view(),
        x.view(),
        proj.view(),
        field.len(),
    )?;
    Ok(out)
}

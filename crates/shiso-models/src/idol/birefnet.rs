//! The background remover: a picture with a background to its matte.
//!
//! A Swin-L backbone runs on the picture at 1024 and again at 512; each
//! level's two answers are concatenated, the coarsest level is joined by the
//! other three resampled to its size, and a squeeze block narrows it. A
//! four-level decoder then climbs back to 1024, re-reading the picture at
//! every level (cut into patches the level's size and stacked on channels),
//! gating each level by a predicted attention map, and answers one logit per
//! pixel. The decoder blocks are ASPP blocks whose branches are modulated
//! deformable convolutions.
//!
//! The picture is resampled to 1024 square on the way in and the matte back
//! to the picture's own size on the way out, both bilinear with aligned
//! corners, as the reference does: the matte is at the size it was handed,
//! whatever that was.
//!
//! Every `BatchNorm` follows a convolution and runs with its running
//! statistics, so it is folded into that convolution at load. The deformable
//! convolution's modulator is `2 * sigmoid(m)`, and the factor 2 is folded
//! into its weight. The weights ship in fp16 and run widened to f32.

use std::path::Path;
use std::sync::Arc;

use shiso_core::DeviceBuffer;
use shiso_vision::generator::ImageGenError;

use super::nn::{Affine, bytes, host_lin, layer_norm, linear};
use super::preprocess::Matting;
use super::splat_ops::{DeformCall, ResizeCall, SplatOps, WindowAttnCall};
use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{Conv2dCall, GemmCall, LinWeight, Ops, Scratch, SliceExt, WeightKind};
use crate::diffusion::weights::StStore;

/// The side the network reads.
const SIDE: usize = 1024;
/// `BatchNorm` and `LayerNorm` epsilon (both modules' defaults).
const NORM_EPS: f32 = 1e-5;
/// Swin's window.
const WINDOW: usize = 12;
/// The `ImageNet` affine the picture is normalised with.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Deformable columns formed at once, in f32 elements (128 MiB).
const COLUMN_BUDGET: usize = 1 << 25;

/// A convolution with any `BatchNorm` after it folded in.
#[derive(Debug)]
struct Conv {
    w: DeviceBuffer,
    bias: DeviceBuffer,
    out_c: usize,
    in_c: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

/// A `(channels, h, w)` field on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape {
    c: usize,
    h: usize,
    w: usize,
}

impl Shape {
    const fn plane(self) -> usize {
        self.h * self.w
    }
    const fn len(self) -> usize {
        self.c * self.h * self.w
    }
}

/// Reads `{name}.weight`/`.bias` and, when `bn` names one, folds that
/// `BatchNorm` in: `w' = w g / s`, `b' = (b - mean) g / s + beta` with
/// `s = sqrt(var + eps)`. `scale` multiplies the folded weight.
#[expect(
    clippy::cast_possible_truncation,
    reason = "conv kernel dims are small element counts"
)]
fn load_conv(
    store: &StStore,
    ops: &Ops,
    name: &str,
    bn: Option<&str>,
    (stride, pad, scale): (usize, usize, f32),
) -> Result<Conv, ImageGenError> {
    let wname = format!("{name}.weight");
    let shape = store.require(&wname)?.shape.clone();
    let [out_c, in_c, k_h, k_w] = shape.as_slice() else {
        return Err(ImageGenError::Checkpoint(format!(
            "{wname} is {shape:?}, not a conv kernel"
        )));
    };
    if k_h != k_w {
        return Err(ImageGenError::Checkpoint(format!(
            "{wname} is not a square kernel"
        )));
    }
    let (out_c, in_c, k) = (*out_c as usize, *in_c as usize, *k_h as usize);
    let mut w = store.f32_vec(&wname)?;
    let bname = format!("{name}.bias");
    let mut b = if store.contains(&bname) {
        store.f32_vec(&bname)?
    } else {
        vec![0.0; out_c]
    };
    if let Some(bn) = bn {
        let gamma = store.f32_vec(&format!("{bn}.weight"))?;
        let beta = store.f32_vec(&format!("{bn}.bias"))?;
        let mean = store.f32_vec(&format!("{bn}.running_mean"))?;
        let var = store.f32_vec(&format!("{bn}.running_var"))?;
        if [gamma.len(), beta.len(), mean.len(), var.len()] != [out_c; 4] {
            return Err(ImageGenError::Checkpoint(format!(
                "{bn} does not cover {out_c} channels"
            )));
        }
        let per_out = in_c * k * k;
        for o in 0..out_c {
            let g = gamma[o] / (var[o] + NORM_EPS).sqrt();
            for v in &mut w[o * per_out..(o + 1) * per_out] {
                *v *= g;
            }
            b[o] = (b[o] - mean[o]) * g + beta[o];
        }
    }
    if (scale - 1.0).abs() > 0.0 {
        for v in &mut w {
            *v *= scale;
        }
    }
    Ok(Conv {
        w: ops.upload_f32("birefnet conv", &w)?,
        bias: ops.upload_f32("birefnet conv", &b)?,
        out_c,
        in_c,
        k,
        stride,
        pad,
    })
}

/// One modulated deformable convolution with its `BatchNorm` folded in.
#[derive(Debug)]
struct Deform {
    offset: Conv,
    modulator: Conv,
    /// `(out, in * k * k)` rows, already scaled by the modulator's factor 2,
    /// with the folded norm's shift as the GEMM's bias.
    weight: Lin,
    k: usize,
    pad: usize,
}

#[derive(Debug)]
struct Aspp {
    branches: Vec<Deform>,
    pool: Conv,
    project: Conv,
}

#[derive(Debug)]
struct DecBlock {
    conv_in: Conv,
    aspp: Aspp,
    conv_out: Conv,
}

#[derive(Debug)]
struct SimpleConvs {
    conv1: Conv,
    conv_out: Conv,
}

#[derive(Debug)]
struct SwinBlock {
    norm1: Affine,
    q: Lin,
    k: Lin,
    v: Lin,
    k_bias: DeviceBuffer,
    v_bias: DeviceBuffer,
    table: DeviceBuffer,
    proj: Lin,
    norm2: Affine,
    fc1: Lin,
    fc2: Lin,
}

#[derive(Debug)]
struct SwinLayer {
    dim: usize,
    heads: usize,
    blocks: Vec<SwinBlock>,
    merge: Option<(Affine, Lin)>,
    norm: Affine,
}

/// The loaded remover.
#[derive(Debug)]
pub struct BiRefNet {
    ops: SplatOps,
    patch_embed: Conv,
    patch_norm: Affine,
    layers: Vec<SwinLayer>,
    squeeze: DecBlock,
    ipt: [SimpleConvs; 5],
    blocks: [DecBlock; 4],
    lateral: [Conv; 3],
    gdt: [(Conv, Conv); 3],
    conv_out: Conv,
}

fn load_deform(store: &StStore, ops: &Ops, base: &str, k: usize) -> Result<Deform, ImageGenError> {
    let pad = k / 2;
    let regular = load_conv(
        store,
        ops,
        &format!("{base}.atrous_conv.regular_conv"),
        Some(&format!("{base}.bn")),
        (1, pad, 2.0),
    )?;
    if regular.k != k {
        return Err(ImageGenError::Checkpoint(format!(
            "{base} is a {}x{} kernel, expected {k}x{k}",
            regular.k, regular.k
        )));
    }
    Ok(Deform {
        offset: load_conv(
            store,
            ops,
            &format!("{base}.atrous_conv.offset_conv"),
            None,
            (1, pad, 1.0),
        )?,
        modulator: load_conv(
            store,
            ops,
            &format!("{base}.atrous_conv.modulator_conv"),
            None,
            (1, pad, 1.0),
        )?,
        weight: Lin {
            w: regular.w,
            kind: WeightKind::F32,
            scale: 1.0,
            row_scale: None,
            rotate: None,
            bias: Some(regular.bias),
            out_dim: regular.out_c,
            in_dim: regular.in_c * k * k,
        },
        k,
        pad,
    })
}

fn load_dec_block(store: &StStore, ops: &Ops, base: &str) -> Result<DecBlock, ImageGenError> {
    let att = format!("{base}.dec_att");
    let mut branches = vec![load_deform(store, ops, &format!("{att}.aspp1"), 1)?];
    for (i, k) in [1usize, 3, 7].into_iter().enumerate() {
        branches.push(load_deform(
            store,
            ops,
            &format!("{att}.aspp_deforms.{i}"),
            k,
        )?);
    }
    Ok(DecBlock {
        conv_in: load_conv(
            store,
            ops,
            &format!("{base}.conv_in"),
            Some(&format!("{base}.bn_in")),
            (1, 1, 1.0),
        )?,
        aspp: Aspp {
            branches,
            pool: load_conv(
                store,
                ops,
                &format!("{att}.global_avg_pool.1"),
                Some(&format!("{att}.global_avg_pool.2")),
                (1, 0, 1.0),
            )?,
            project: load_conv(
                store,
                ops,
                &format!("{att}.conv1"),
                Some(&format!("{att}.bn1")),
                (1, 0, 1.0),
            )?,
        },
        conv_out: load_conv(
            store,
            ops,
            &format!("{base}.conv_out"),
            Some(&format!("{base}.bn_out")),
            (1, 1, 1.0),
        )?,
    })
}

fn load_swin_block(
    store: &StStore,
    ops: &Ops,
    base: &str,
    dim: usize,
) -> Result<SwinBlock, ImageGenError> {
    let w = store.f32_vec(&format!("{base}.attn.qkv.weight"))?;
    let b = store.f32_vec(&format!("{base}.attn.qkv.bias"))?;
    if w.len() != 3 * dim * dim || b.len() != 3 * dim {
        return Err(ImageGenError::Checkpoint(format!(
            "{base}.attn.qkv is not 3 x {dim} rows"
        )));
    }
    let part = |i: usize| {
        (
            &w[i * dim * dim..(i + 1) * dim * dim],
            &b[i * dim..(i + 1) * dim],
        )
    };
    let lin = |name: &str| -> Result<Lin, ImageGenError> {
        let w = store.f32_vec(&format!("{name}.weight"))?;
        let bias_name = format!("{name}.bias");
        let b = if store.contains(&bias_name) {
            Some(store.f32_vec(&bias_name)?)
        } else {
            None
        };
        let out_dim = store.require(&format!("{name}.weight"))?.shape[0];
        let out_dim = usize::try_from(out_dim).unwrap_or(0);
        host_lin(ops, &w, b.as_deref(), out_dim, w.len() / out_dim.max(1))
    };
    let (qw, qb) = part(0);
    let (kw, kb) = part(1);
    let (vw, vb) = part(2);
    Ok(SwinBlock {
        norm1: Affine::load(store, ops, &format!("{base}.norm1"))?,
        q: host_lin(ops, qw, Some(qb), dim, dim)?,
        k: host_lin(ops, kw, Some(kb), dim, dim)?,
        v: host_lin(ops, vw, Some(vb), dim, dim)?,
        k_bias: ops.upload_f32("swin pad key", kb)?,
        v_bias: ops.upload_f32("swin pad value", vb)?,
        table: store.f32_widen(
            ops,
            "swin bias table",
            &format!("{base}.attn.relative_position_bias_table"),
        )?,
        proj: lin(&format!("{base}.attn.proj"))?,
        norm2: Affine::load(store, ops, &format!("{base}.norm2"))?,
        fc1: lin(&format!("{base}.mlp.fc1"))?,
        fc2: lin(&format!("{base}.mlp.fc2"))?,
    })
}

impl BiRefNet {
    /// Opens the remover's weights at `path`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a tensor of another shape, and
    /// [`ImageGenError::Backend`] for a device without the `splat` kernels.
    pub(crate) fn load(path: &Path, ops: Arc<Ops>) -> Result<Self, ImageGenError> {
        let store = StStore::open_files(&[path.to_path_buf()])?;
        let o = &*ops;
        let dims = [192usize, 384, 768, 1536];
        let depths = [2usize, 2, 18, 2];
        let heads = [6usize, 12, 24, 48];
        let mut layers = Vec::with_capacity(4);
        for i in 0..4 {
            let base = format!("bb.layers.{i}");
            let blocks = (0..depths[i])
                .map(|j| load_swin_block(&store, o, &format!("{base}.blocks.{j}"), dims[i]))
                .collect::<Result<Vec<_>, _>>()?;
            let merge = if i < 3 {
                let w = store.f32_vec(&format!("{base}.downsample.reduction.weight"))?;
                Some((
                    Affine::load(&store, o, &format!("{base}.downsample.norm"))?,
                    host_lin(o, &w, None, 2 * dims[i], 4 * dims[i])?,
                ))
            } else {
                None
            };
            layers.push(SwinLayer {
                dim: dims[i],
                heads: heads[i],
                blocks,
                merge,
                norm: Affine::load(&store, o, &format!("bb.norm{i}"))?,
            });
        }
        let simple = |name: &str| -> Result<SimpleConvs, ImageGenError> {
            Ok(SimpleConvs {
                conv1: load_conv(
                    &store,
                    o,
                    &format!("decoder.{name}.conv1"),
                    None,
                    (1, 1, 1.0),
                )?,
                conv_out: load_conv(
                    &store,
                    o,
                    &format!("decoder.{name}.conv_out"),
                    None,
                    (1, 1, 1.0),
                )?,
            })
        };
        let gdt = |i: usize| -> Result<(Conv, Conv), ImageGenError> {
            Ok((
                load_conv(
                    &store,
                    o,
                    &format!("decoder.gdt_convs_{i}.0"),
                    Some(&format!("decoder.gdt_convs_{i}.1")),
                    (1, 1, 1.0),
                )?,
                load_conv(
                    &store,
                    o,
                    &format!("decoder.gdt_convs_attn_{i}.0"),
                    None,
                    (1, 0, 1.0),
                )?,
            ))
        };
        let lateral = |i: usize| {
            load_conv(
                &store,
                o,
                &format!("decoder.lateral_block{i}.conv"),
                None,
                (1, 0, 1.0),
            )
        };
        Ok(Self {
            patch_embed: load_conv(&store, o, "bb.patch_embed.proj", None, (4, 0, 1.0))?,
            patch_norm: Affine::load(&store, o, "bb.patch_embed.norm")?,
            layers,
            squeeze: load_dec_block(&store, o, "squeeze_module.0")?,
            ipt: [
                simple("ipt_blk1")?,
                simple("ipt_blk2")?,
                simple("ipt_blk3")?,
                simple("ipt_blk4")?,
                simple("ipt_blk5")?,
            ],
            blocks: [
                load_dec_block(&store, o, "decoder.decoder_block1")?,
                load_dec_block(&store, o, "decoder.decoder_block2")?,
                load_dec_block(&store, o, "decoder.decoder_block3")?,
                load_dec_block(&store, o, "decoder.decoder_block4")?,
            ],
            lateral: [lateral(2)?, lateral(3)?, lateral(4)?],
            gdt: [gdt(2)?, gdt(3)?, gdt(4)?],
            conv_out: load_conv(&store, o, "decoder.conv_out1.0", None, (1, 0, 1.0))?,
            ops: SplatOps::new(ops)?,
        })
    }

    fn ops(&self) -> &Ops {
        self.ops.ops()
    }

    /// The matte's probabilities for a `(3, h, w)` picture in `[0, 1]`,
    /// `(h, w)` in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// A device failure.
    pub(crate) fn probabilities(
        &self,
        rgb: &[f32],
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        let ops = self.ops();
        let input = ops.upload_scratch("birefnet input", rgb)?;
        let square = Shape {
            c: 3,
            h: SIDE,
            w: SIDE,
        };
        let x = self.resize(input.view(), Shape { c: 3, h, w }, square)?;
        drop(input);
        // The ImageNet affine, plane by plane, on the host: the decoder reads
        // the normalised square there anyway to cut it into patches.
        let mut normed = ops.download("birefnet input", x.view(), square.len())?;
        for (c, plane) in normed.chunks_exact_mut(square.plane()).enumerate() {
            for v in plane {
                *v = (*v - MEAN[c]) / STD[c];
            }
        }
        let x = ops.upload_scratch("birefnet input", &normed)?;
        let logits = self.forward(&x, &normed)?;
        self.ops
            .sigmoid("birefnet out", logits.view(), logits.view(), square.plane())?;
        let mask = self.resize(
            logits.view(),
            Shape { c: 1, ..square },
            Shape { c: 1, h, w },
        )?;
        let out = ops.download("birefnet readback", mask.view(), h * w)?;
        ops.trim_pool();
        if out.iter().any(|v| !v.is_finite()) {
            return Err(ImageGenError::Backend("the matte is not finite".to_owned()));
        }
        Ok(out)
    }

    /// The network on the normalised 1024 square (`x` on the device and
    /// `host` its host copy, which the decoder cuts into patches): one logit
    /// per pixel.
    fn forward<'o>(&'o self, x: &DeviceBuffer, host: &[f32]) -> Result<Scratch<'o>, ImageGenError> {
        let square = Shape {
            c: 3,
            h: SIDE,
            w: SIDE,
        };
        let full = self.backbone(x.view(), square)?;
        let half_shape = Shape {
            c: 3,
            h: SIDE / 2,
            w: SIDE / 2,
        };
        let half_input = self.resize(x.view(), square, half_shape)?;
        let half = self.backbone(half_input.view(), half_shape)?;
        drop(half_input);
        // Each level: its own answer, then the half-scale answer resampled up.
        let mut levels = Vec::with_capacity(4);
        for ((f, fs), (g, gs)) in full.into_iter().zip(half) {
            let up = self.resize(g.view(), gs, Shape { c: gs.c, ..fs })?;
            levels.push(self.concat(&[(f.view(), fs), (up.view(), Shape { c: gs.c, ..fs })])?);
        }
        let x4s = levels[3].1;
        let ups = (0..3)
            .map(|i| {
                let (buf, s) = &levels[i];
                let target = Shape { c: s.c, ..x4s };
                self.resize(buf.view(), *s, target).map(|b| (b, target))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (x4, x4_shape) = self.concat(&[
            (ups[0].0.view(), ups[0].1),
            (ups[1].0.view(), ups[1].1),
            (ups[2].0.view(), ups[2].1),
            (levels[3].0.view(), x4s),
        ])?;
        drop(ups);
        let (p4, p4s) = self.dec_block(&self.squeeze, x4.view(), x4_shape)?;
        drop(x4);
        self.decoder(host, &levels, (p4, p4s))
    }

    /// The decoder: four levels from the squeezed coarsest field up to 1024.
    fn decoder<'o>(
        &'o self,
        host: &[f32],
        levels: &[(Scratch<'o>, Shape)],
        start: (Scratch<'o>, Shape),
    ) -> Result<Scratch<'o>, ImageGenError> {
        let (mut p, mut ps) = start;
        // Level 4 (the coarsest) up to level 1: picture patches, the block,
        // the gate, then up a level plus the lateral of the level above.
        for level in (1..=4usize).rev() {
            let (patches, pshape) = self.patches(host, ps)?;
            let (conv_in, conv_in_s) = self.simple(&self.ipt[level], patches.view(), pshape)?;
            drop(patches);
            let (joined, js) = self.concat(&[(p.view(), ps), (conv_in.view(), conv_in_s)])?;
            drop((conv_in, p));
            let (out, os) = self.dec_block(&self.blocks[level - 1], joined.view(), js)?;
            drop(joined);
            if level == 1 {
                // The finest level is resampled to the picture's side, joined
                // by the picture itself, and read out.
                let side = Shape {
                    c: os.c,
                    h: SIDE,
                    w: SIDE,
                };
                let up = self.resize(out.view(), os, side)?;
                let (patches, pshape) = self.patches(host, side)?;
                let (img, imgs) = self.simple(&self.ipt[0], patches.view(), pshape)?;
                let (joined, js) = self.concat(&[(up.view(), side), (img.view(), imgs)])?;
                let (logits, _) = self.conv(&self.conv_out, joined.view(), js)?;
                return Ok(logits);
            }
            let (gdt_conv, gdt_head) = &self.gdt[level - 2];
            let (g, gs) = self.conv(gdt_conv, out.view(), os)?;
            self.ops
                .relu("birefnet gdt", g.view(), g.view(), gs.len())?;
            let (attn, _) = self.conv(gdt_head, g.view(), gs)?;
            self.ops
                .sigmoid("birefnet gdt", attn.view(), attn.view(), os.plane())?;
            let gated = self.ops().alloc_f32("birefnet gate", os.len())?;
            self.ops()
                .backend()
                .fill_zero(gated.view())
                .map_err(|e| ImageGenError::Backend(e.to_string()))?;
            // gated[c][p] = attn[p] * out[c][p]: the one-channel gate repeats
            // every plane.
            self.ops().add_gated(&crate::diffusion::ops::AddGatedCall {
                step: "birefnet gate",
                out: gated.view(),
                a: gated.view(),
                gate: attn.view(),
                b: out.view(),
                count: os.len(),
                period: os.plane(),
            })?;
            drop((g, attn, out));
            let (lat_in, lat_s) = &levels[level - 2];
            let up = self.resize(gated.view(), os, Shape { c: os.c, ..*lat_s })?;
            let (lat, _) = self.conv(&self.lateral[level - 2], lat_in.view(), *lat_s)?;
            let summed = self
                .ops()
                .alloc_f32("birefnet lateral", lat_s.c * lat_s.plane())?;
            self.ops().add(
                "birefnet lateral",
                summed.view(),
                up.view(),
                lat.view(),
                lat_s.c * lat_s.plane(),
            )?;
            p = summed;
            ps = Shape {
                c: lat_s.c,
                ..*lat_s
            };
        }
        Err(ImageGenError::Backend(
            "the decoder ended without a read-out".to_owned(),
        ))
    }

    /// The picture cut into patches the size of `target`'s grid, stacked on
    /// channels: `(3 * g * g, target.h, target.w)` with `g = 1024 / target.h`.
    fn patches(&self, host: &[f32], target: Shape) -> Result<(Scratch<'_>, Shape), ImageGenError> {
        let (hg, wg) = (SIDE / target.h, SIDE / target.w);
        let (h, w) = (target.h, target.w);
        let mut out = vec![0.0f32; 3 * hg * wg * h * w];
        for c in 0..3 {
            for gi in 0..hg {
                for gj in 0..wg {
                    let ch = (c * hg + gi) * wg + gj;
                    for y in 0..h {
                        let src = (c * SIDE + gi * h + y) * SIDE + gj * w;
                        out[(ch * h + y) * w..(ch * h + y + 1) * w]
                            .copy_from_slice(&host[src..src + w]);
                    }
                }
            }
        }
        Ok((
            self.ops().upload_scratch("birefnet patches", &out)?,
            Shape {
                c: 3 * hg * wg,
                h,
                w,
            },
        ))
    }

    /// Two convolutions, no activation between.
    fn simple(
        &self,
        convs: &SimpleConvs,
        x: shiso_core::DeviceView<'_>,
        s: Shape,
    ) -> Result<(Scratch<'_>, Shape), ImageGenError> {
        let (h, hs) = self.conv(&convs.conv1, x, s)?;
        self.conv(&convs.conv_out, h.view(), hs)
    }

    /// `conv_in` + `ReLU`, the ASPP block, `conv_out` (its norm folded, no
    /// activation after).
    fn dec_block(
        &self,
        block: &DecBlock,
        x: shiso_core::DeviceView<'_>,
        s: Shape,
    ) -> Result<(Scratch<'_>, Shape), ImageGenError> {
        let (h, hs) = self.conv(&block.conv_in, x, s)?;
        self.ops
            .relu("birefnet dec", h.view(), h.view(), hs.len())?;
        let (a, as_) = self.aspp(&block.aspp, h.view(), hs)?;
        drop(h);
        self.conv(&block.conv_out, a.view(), as_)
    }

    /// Four deformable branches and the pooled branch, joined and projected.
    fn aspp(
        &self,
        aspp: &Aspp,
        x: shiso_core::DeviceView<'_>,
        s: Shape,
    ) -> Result<(Scratch<'_>, Shape), ImageGenError> {
        let ops = self.ops();
        let mut parts = Vec::with_capacity(5);
        for branch in &aspp.branches {
            let (y, ys) = self.deform(branch, x, s)?;
            self.ops
                .relu("birefnet aspp", y.view(), y.view(), ys.len())?;
            parts.push((y, ys));
        }
        // The pooled branch: the channel means, a 1x1 conv and ReLU, then
        // "resampled" from one pixel, which is a broadcast.
        let ones = ops.upload_scratch("birefnet pool", &vec![1.0 / plane_f32(s); s.plane()])?;
        let pooled = ops.alloc_f32("birefnet pool", s.c)?;
        ops.gemm(&GemmCall {
            step: "birefnet pool",
            out: pooled.view(),
            input: x,
            weight: LinWeight::f32(ones.view()),
            m: s.c,
            n: 1,
            k: s.plane(),
        })?;
        let one = Shape { c: s.c, h: 1, w: 1 };
        let (p, ps) = self.conv(&aspp.pool, pooled.view(), one)?;
        self.ops
            .relu("birefnet pool", p.view(), p.view(), ps.len())?;
        let wide = Shape { c: ps.c, ..s };
        let broadcast = self.resize(p.view(), ps, wide)?;
        parts.push((broadcast, wide));
        let views: Vec<_> = parts.iter().map(|(b, s)| (b.view(), *s)).collect();
        let (joined, js) = self.concat(&views)?;
        drop(parts);
        let (y, ys) = self.conv(&aspp.project, joined.view(), js)?;
        self.ops
            .relu("birefnet aspp", y.view(), y.view(), ys.len())?;
        Ok((y, ys))
    }

    /// A modulated deformable convolution, as columns and a GEMM per block of
    /// output pixels.
    fn deform(
        &self,
        d: &Deform,
        x: shiso_core::DeviceView<'_>,
        s: Shape,
    ) -> Result<(Scratch<'_>, Shape), ImageGenError> {
        let ops = self.ops();
        let (offset, _) = self.conv(&d.offset, x, s)?;
        let (mask, _) = self.conv(&d.modulator, x, s)?;
        let taps = d.k * d.k;
        self.ops.sigmoid(
            "birefnet deform",
            mask.view(),
            mask.view(),
            taps * s.plane(),
        )?;
        let width = s.c * taps;
        if width != d.weight.in_dim {
            return Err(ImageGenError::Backend(format!(
                "a deformable conv over {} columns was handed {width}",
                d.weight.in_dim
            )));
        }
        let out_c = d.weight.out_dim;
        let block = (COLUMN_BUDGET / width).clamp(1, s.plane());
        let cols = ops.alloc_f32("birefnet deform cols", block * width)?;
        let rows_out = ops.alloc_f32("birefnet deform out", s.plane() * out_c)?;
        for row0 in (0..s.plane()).step_by(block) {
            let rows = block.min(s.plane() - row0);
            self.ops.deform_im2col(&DeformCall {
                step: "birefnet deform",
                cols: cols.view(),
                input: x,
                offset: offset.view(),
                mask: mask.view(),
                channels: s.c,
                in_h: s.h,
                in_w: s.w,
                out_h: s.h,
                out_w: s.w,
                k: d.k,
                stride: 1,
                pad: d.pad,
                row0,
                rows,
            })?;
            ops.gemm(&GemmCall {
                step: "birefnet deform",
                out: rows_out
                    .view()
                    .cut(bytes(row0 * out_c), bytes(rows * out_c))?,
                input: cols.view().cut(0, bytes(rows * width))?,
                weight: d.weight.weight(),
                m: rows,
                n: out_c,
                k: width,
            })?;
        }
        let out_s = Shape { c: out_c, ..s };
        let out = ops.alloc_f32("birefnet deform", out_s.len())?;
        ops.transpose(
            "birefnet deform",
            out.view(),
            rows_out.view(),
            s.plane(),
            out_c,
        )?;
        Ok((out, out_s))
    }

    /// A plain convolution with its bias.
    fn conv(
        &self,
        conv: &Conv,
        x: shiso_core::DeviceView<'_>,
        s: Shape,
    ) -> Result<(Scratch<'_>, Shape), ImageGenError> {
        if conv.in_c != s.c {
            return Err(ImageGenError::Backend(format!(
                "a {}-channel conv was handed {} channels",
                conv.in_c, s.c
            )));
        }
        let out_h = (s.h + 2 * conv.pad - conv.k) / conv.stride + 1;
        let out_w = (s.w + 2 * conv.pad - conv.k) / conv.stride + 1;
        let out_s = Shape {
            c: conv.out_c,
            h: out_h,
            w: out_w,
        };
        let out = self.ops().alloc_f32("birefnet conv", out_s.len())?;
        self.ops().conv2d(&Conv2dCall {
            step: "birefnet conv",
            out: out.view(),
            input: x,
            weight: conv.w.view(),
            bias: Some(conv.bias.view()),
            batch: 1,
            in_c: s.c,
            in_h: s.h,
            in_w: s.w,
            out_c: conv.out_c,
            out_h,
            out_w,
            k_h: conv.k,
            k_w: conv.k,
            stride: conv.stride,
            pad: conv.pad,
        })?;
        Ok((out, out_s))
    }

    /// Bilinear resampling with aligned corners.
    fn resize(
        &self,
        x: shiso_core::DeviceView<'_>,
        from: Shape,
        to: Shape,
    ) -> Result<Scratch<'_>, ImageGenError> {
        let out = self.ops().alloc_f32("birefnet resize", to.len())?;
        self.ops.resize_bilinear(&ResizeCall {
            step: "birefnet resize",
            out: out.view(),
            input: x,
            channels: from.c,
            in_h: from.h,
            in_w: from.w,
            out_h: to.h,
            out_w: to.w,
        })?;
        Ok(out)
    }

    /// Channel concatenation of same-sized fields.
    fn concat(
        &self,
        parts: &[(shiso_core::DeviceView<'_>, Shape)],
    ) -> Result<(Scratch<'_>, Shape), ImageGenError> {
        let first = parts
            .first()
            .ok_or_else(|| ImageGenError::Backend("an empty concatenation".to_owned()))?
            .1;
        let c: usize = parts.iter().map(|(_, s)| s.c).sum();
        let out_s = Shape { c, ..first };
        let out = self.ops().alloc_f32("birefnet concat", out_s.len())?;
        let mut at = 0usize;
        for (view, s) in parts {
            if (s.h, s.w) != (first.h, first.w) {
                return Err(ImageGenError::Backend(
                    "a concatenation of different sizes".to_owned(),
                ));
            }
            self.ops().copy(
                "birefnet concat",
                out.view().cut(bytes(at), bytes(s.len()))?,
                view.cut(0, bytes(s.len()))?,
            )?;
            at += s.len();
        }
        Ok((out, out_s))
    }

    /// The Swin backbone on a `(3, h, w)` field: its four levels, each as a
    /// normed `(channels, h, w)` field.
    fn backbone(
        &self,
        x: shiso_core::DeviceView<'_>,
        s: Shape,
    ) -> Result<Vec<(Scratch<'_>, Shape)>, ImageGenError> {
        let ops = self.ops();
        let (planes, ps) = self.conv(&self.patch_embed, x, s)?;
        let (mut h, mut w) = (ps.h, ps.w);
        let tokens = ops.alloc_f32("swin tokens", ps.len())?;
        ops.transpose(
            "swin tokens",
            tokens.view(),
            planes.view(),
            ps.c,
            ps.plane(),
        )?;
        drop(planes);
        let mut x = layer_norm(
            ops,
            "swin embed norm",
            tokens.view(),
            h * w,
            ps.c,
            NORM_EPS,
            Some(&self.patch_norm),
        )?;
        drop(tokens);
        let mut outs = Vec::with_capacity(4);
        for layer in &self.layers {
            let dim = layer.dim;
            for (j, block) in layer.blocks.iter().enumerate() {
                let shift = if j % 2 == 0 { 0 } else { WINDOW / 2 };
                self.swin_block(block, x.view(), (h, w), (dim, layer.heads), shift)?;
            }
            let normed = layer_norm(
                ops,
                "swin out norm",
                x.view(),
                h * w,
                dim,
                NORM_EPS,
                Some(&layer.norm),
            )?;
            let field = ops.alloc_f32("swin out", h * w * dim)?;
            ops.transpose("swin out", field.view(), normed.view(), h * w, dim)?;
            outs.push((field, Shape { c: dim, h, w }));
            if let Some((norm, reduction)) = &layer.merge {
                let merged = ops.alloc_f32("swin merge", h * w * dim)?;
                self.ops
                    .merge2x2("swin merge", merged.view(), x.view(), (h, w), dim)?;
                let (mh, mw) = (h / 2, w / 2);
                let mn = layer_norm(
                    ops,
                    "swin merge norm",
                    merged.view(),
                    mh * mw,
                    4 * dim,
                    NORM_EPS,
                    Some(norm),
                )?;
                x = linear(ops, "swin merge", reduction, mn.view(), mh * mw)?;
                (h, w) = (mh, mw);
            }
        }
        Ok(outs)
    }

    /// One Swin block over `(h * w, dim)` token rows, in place.
    #[expect(
        clippy::many_single_char_names,
        reason = "x, h, w, n, q, k and v are the block's stream, grid and projections"
    )]
    fn swin_block(
        &self,
        block: &SwinBlock,
        x: shiso_core::DeviceView<'_>,
        (h, w): (usize, usize),
        (dim, heads): (usize, usize),
        shift: usize,
    ) -> Result<(), ImageGenError> {
        let ops = self.ops();
        let tokens = h * w;
        let n = layer_norm(
            ops,
            "swin norm1",
            x,
            tokens,
            dim,
            NORM_EPS,
            Some(&block.norm1),
        )?;
        let q = linear(ops, "swin q", &block.q, n.view(), tokens)?;
        let k = linear(ops, "swin k", &block.k, n.view(), tokens)?;
        let v = linear(ops, "swin v", &block.v, n.view(), tokens)?;
        drop(n);
        let attended = ops.alloc_f32("swin attention", tokens * dim)?;
        self.ops.window_attn(&WindowAttnCall {
            step: "swin attention",
            out: attended.view(),
            q: q.view(),
            k: k.view(),
            v: v.view(),
            k_pad: block.k_bias.view(),
            v_pad: block.v_bias.view(),
            table: block.table.view(),
            h,
            w,
            window: WINDOW,
            shift,
            heads,
            head_dim: dim / heads,
        })?;
        drop((q, k, v));
        let projected = linear(ops, "swin proj", &block.proj, attended.view(), tokens)?;
        ops.add("swin residual", x, x, projected.view(), tokens * dim)?;
        drop((attended, projected));
        let n = layer_norm(
            ops,
            "swin norm2",
            x,
            tokens,
            dim,
            NORM_EPS,
            Some(&block.norm2),
        )?;
        let up = linear(ops, "swin mlp", &block.fc1, n.view(), tokens)?;
        self.ops
            .gelu_erf("swin mlp", up.view(), up.view(), tokens * block.fc1.out_dim)?;
        let down = linear(ops, "swin mlp", &block.fc2, up.view(), tokens)?;
        ops.add("swin residual", x, x, down.view(), tokens * dim)
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a plane of at most 1024 x 1024 pixels is exact in f32"
)]
fn plane_f32(s: Shape) -> f32 {
    s.plane() as f32
}

impl Matting for BiRefNet {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the matte is clamped to [0, 255] before the cast, which truncates as the reference's does"
    )]
    fn alpha(&self, rgb: &image::RgbImage) -> Result<Vec<u8>, String> {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let mut planar = vec![0.0f32; 3 * w * h];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            for c in 0..3 {
                planar[(c * h + y as usize) * w + x as usize] = f32::from(pixel[c]) / 255.0;
            }
        }
        let probs = self
            .probabilities(&planar, h, w)
            .map_err(|e| e.to_string())?;
        Ok(probs
            .iter()
            .map(|p| (p.clamp(0.0, 1.0) * 255.0) as u8)
            .collect())
    }
}

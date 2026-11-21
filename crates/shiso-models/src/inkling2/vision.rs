//! Inkling-2 native Vision Transformer (`model.visual.*`) and image preprocessing.
//!
//! Preprocessing matches the official `Inkling2ImageProcessor`:
//! - Alpha compositing onto an 8×8 gray checkerboard (204 / 153).
//! - Padding bottom/right to multiples of `40` (`native_patch_size * adapter_merge_size`)
//!   with `-1.0` on the 0–255 scale.
//! - Rescaling by `1/255` and normalizing with mean `0.5`, std `0.5` (`[-1, 1]`).
//! - Patchifying into `20×20×3` native patches (`1200` floats per patch), 4 patches per
//!   LM image token.

#![expect(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    reason = "vision encoder executes multi-stage tensor graph with bounded image dimensions"
)]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use shiso_core::{
    Backend, CoreResult, DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs, VisionGrid, alloc,
};
use shiso_modelfile::st_header::StDType;
use shiso_vision::{
    EncodedRows, MediaPlacement, PreprocessSpec, VisionError, VisionItem, VisionTower,
};

use super::config::VisionConfig;
use super::ops::{self, LinearArgs, MODULE};
use super::weights::SafetensorsDir;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VitNormArgs {
    rows: u32,
    hidden: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VitRopeArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    theta: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VitAttnArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VitUnaryArgs {
    count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VitLayerScaleArgs {
    rows: u32,
    hidden: u32,
}

#[derive(Debug)]
struct VitLayerWeights {
    ln1_w: DeviceBuffer,
    ln1_b: DeviceBuffer,
    q_w: DeviceBuffer,
    q_b: DeviceBuffer,
    k_w: DeviceBuffer,
    k_b: DeviceBuffer,
    v_w: DeviceBuffer,
    v_b: DeviceBuffer,
    out_w: DeviceBuffer,
    out_b: DeviceBuffer,
    ls1_gamma: DeviceBuffer,
    ln2_w: DeviceBuffer,
    ln2_b: DeviceBuffer,
    fc_w: DeviceBuffer,
    fc_b: DeviceBuffer,
    proj_w: DeviceBuffer,
    proj_b: DeviceBuffer,
    ls2_gamma: DeviceBuffer,
}

/// Device-resident Inkling-2 Vision Transformer + 2×2 spatial merger projector.
#[derive(Debug)]
pub struct Inkling2VisionTower {
    config: VisionConfig,
    preprocess: PreprocessSpec,
    device_bytes: u64,
    patch_embed_w: DeviceBuffer,
    /// Host copy of `position_embedding` (`[grid_size * grid_size, hidden]`, F32)
    /// for exact bilinear interpolation per image aspect ratio.
    pos_embed_host: Vec<f32>,
    /// Host copy of `class_embedding + cls_position_embedding` (`[hidden]`, F32).
    cls_embed_host: Vec<f32>,
    ln_pre_w: DeviceBuffer,
    ln_pre_b: DeviceBuffer,
    layers: Vec<VitLayerWeights>,
    proj_in_w: DeviceBuffer,
    proj_out_w: DeviceBuffer,
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

/// Bilinearly interpolates the `grid_size × grid_size` learned position embedding
/// onto a `(h_patches, w_patches)` grid (`align_corners=False`).
fn interpolate_pos_embed(
    table: &[f32],
    grid_size: usize,
    hidden: usize,
    h_patches: usize,
    w_patches: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; h_patches * w_patches * hidden];
    let axis_weights = |idx: usize, target: usize| -> Vec<(usize, f32)> {
        let center = (grid_size as f32 / target as f32) * (idx as f32 + 0.5);
        let mut ws = Vec::new();
        let mut sum = 0.0_f32;
        for g in 0..grid_size {
            let d = (g as f32 + 0.5 - center).abs();
            let w = (1.0 - d).max(0.0);
            if w > 0.0 {
                ws.push((g, w));
                sum += w;
            }
        }
        if sum > 0.0 {
            for (_, w) in &mut ws {
                *w /= sum;
            }
        }
        ws
    };
    for y in 0..h_patches {
        let wy = axis_weights(y, h_patches);
        for x in 0..w_patches {
            let wx = axis_weights(x, w_patches);
            let dst = &mut out[(y * w_patches + x) * hidden..][..hidden];
            for &(gy, yw) in &wy {
                for &(gx, xw) in &wx {
                    let w = yw * xw;
                    let src = &table[(gy * grid_size + gx) * hidden..][..hidden];
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d += w * s;
                    }
                }
            }
        }
    }
    out
}

/// Composites an RGBA image onto Inkling-2's 8×8 checkerboard (`204`/`153`).
#[must_use]
pub fn composite_rgba_checkerboard(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for r in 0..height {
        for c in 0..width {
            let bg: u32 = if ((r / 8) + (c / 8)) % 2 == 0 {
                204
            } else {
                153
            };
            let s = (r * width + c) * 4;
            let d = (r * width + c) * 3;
            let a = u32::from(rgba[s + 3]);
            for ch in 0..3 {
                let v = (u32::from(rgba[s + ch]) * a + bg * (255 - a) + 127) / 255;
                rgb[d + ch] = v as u8;
            }
        }
    }
    rgb
}

impl Inkling2VisionTower {
    /// Loads `model.visual.*` from an opened Inkling-2 checkpoint directory.
    ///
    /// # Errors
    ///
    /// Returns [`shiso_core::CoreError`] when vision tensors are missing or device allocation fails.
    pub fn load(
        backend: &Arc<dyn Backend>,
        dir: &SafetensorsDir,
        config: VisionConfig,
    ) -> CoreResult<Self> {
        let mut device_bytes = 0u64;
        let mut up = |name: &str| -> CoreResult<DeviceBuffer> {
            let buf = dir.upload(backend, name, StDType::Bf16)?;
            device_bytes += buf.len();
            Ok(buf)
        };

        let patch_embed_w = up("model.visual.backbone.embeddings.patch_embedding.weight")?;
        let (pos_bytes, _) = dir.tensor_bytes(
            "model.visual.backbone.embeddings.position_embedding",
            StDType::Bf16,
        )?;
        let pos_embed_host = bf16_bytes_to_f32(pos_bytes);
        let (cls_bytes, _) = dir.tensor_bytes(
            "model.visual.backbone.embeddings.class_embedding",
            StDType::Bf16,
        )?;
        let (cls_pos_bytes, _) = dir.tensor_bytes(
            "model.visual.backbone.embeddings.cls_position_embedding",
            StDType::Bf16,
        )?;
        let cls_embed_host: Vec<f32> = bf16_bytes_to_f32(cls_bytes)
            .into_iter()
            .zip(bf16_bytes_to_f32(cls_pos_bytes))
            .map(|(a, b)| a + b)
            .collect();

        let ln_pre_w = up("model.visual.backbone.ln_pre.weight")?;
        let ln_pre_b = up("model.visual.backbone.ln_pre.bias")?;
        let proj_in_w = up("model.visual.proj_in.weight")?;
        let proj_out_w = up("model.visual.proj_out.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.visual.backbone.layers.{i}");
            layers.push(VitLayerWeights {
                ln1_w: up(&format!("{p}.ln_1.weight"))?,
                ln1_b: up(&format!("{p}.ln_1.bias"))?,
                q_w: up(&format!("{p}.self_attn.q_proj.weight"))?,
                q_b: up(&format!("{p}.self_attn.q_proj.bias"))?,
                k_w: up(&format!("{p}.self_attn.k_proj.weight"))?,
                k_b: up(&format!("{p}.self_attn.k_proj.bias"))?,
                v_w: up(&format!("{p}.self_attn.v_proj.weight"))?,
                v_b: up(&format!("{p}.self_attn.v_proj.bias"))?,
                out_w: up(&format!("{p}.self_attn.out_proj.weight"))?,
                out_b: up(&format!("{p}.self_attn.out_proj.bias"))?,
                ls1_gamma: up(&format!("{p}.ls_1.gamma"))?,
                ln2_w: up(&format!("{p}.ln_2.weight"))?,
                ln2_b: up(&format!("{p}.ln_2.bias"))?,
                fc_w: up(&format!("{p}.mlp.c_fc.weight"))?,
                fc_b: up(&format!("{p}.mlp.c_fc.bias"))?,
                proj_w: up(&format!("{p}.mlp.c_proj.weight"))?,
                proj_b: up(&format!("{p}.mlp.c_proj.bias"))?,
                ls2_gamma: up(&format!("{p}.ls_2.gamma"))?,
            });
        }

        let preprocess = PreprocessSpec::inkling2();

        Ok(Self {
            config,
            preprocess,
            device_bytes,
            patch_embed_w,
            pos_embed_host,
            cls_embed_host,
            ln_pre_w,
            ln_pre_b,
            layers,
            proj_in_w,
            proj_out_w,
        })
    }

    /// Encodes one image's preprocessed patch rows (`[gh * merge * gw * merge, 1200]`)
    /// into `[gh * gw, text_hidden_size]` F32 embeddings on `backend`.
    ///
    /// # Errors
    ///
    /// Returns [`shiso_core::CoreError`] on device allocation or launch failure.
    pub fn encode_patches(
        &self,
        backend: &Arc<dyn Backend>,
        native_patches: &[f32],
        grid_h_tokens: u32,
        grid_w_tokens: u32,
    ) -> CoreResult<DeviceBuffer> {
        let c = &self.config;
        let m = c.adapter_merge_size;
        let h_patches = grid_h_tokens * m;
        let w_patches = grid_w_tokens * m;
        let n_patches = h_patches * w_patches;
        let seq = n_patches + 1; // prepend CLS token
        let width = c.hidden_size;
        let heads = c.num_attention_heads;
        let head_dim = width / heads;
        let mlp_dim = (width as f32 * c.mlp_ratio) as u32;
        let in_patch_dim = 3 * c.native_patch_size * c.native_patch_size;

        let d_patches = alloc(backend, (native_patches.len() * 4).max(4) as u64)?;
        backend.copy_h2d(d_patches.view(), cast_slice(native_patches))?;

        // Patch linear embedding for the n_patches rows -> hidden[1..seq]
        let d_hidden = alloc(backend, u64::from(seq) * u64::from(width) * 4)?;
        let patch_dst = d_hidden.view().slice(
            u64::from(width) * 4,
            u64::from(n_patches) * u64::from(width) * 4,
        )?;
        ops::linear_bf16(
            backend.as_ref(),
            LinearArgs {
                rows: n_patches,
                in_dim: in_patch_dim,
                out_dim: width,
            },
            d_patches.view(),
            self.patch_embed_w.view(),
            patch_dst,
        )?;

        // Add bilinear position embeddings to patches + upload CLS token at row 0
        let pos_interp = interpolate_pos_embed(
            &self.pos_embed_host,
            c.posemb_grid_size as usize,
            width as usize,
            h_patches as usize,
            w_patches as usize,
        );
        let d_pos = alloc(backend, (pos_interp.len() * 4).max(4) as u64)?;
        backend.copy_h2d(d_pos.view(), cast_slice(&pos_interp))?;
        ops::add2_f32(
            backend.as_ref(),
            n_patches * width,
            patch_dst,
            d_pos.view(),
            patch_dst,
        )?;
        backend.copy_h2d(
            d_hidden.view().slice(0, u64::from(width) * 4)?,
            cast_slice(&self.cls_embed_host),
        )?;

        // Build 2D positions: CLS = (0, 0), patches = 1-indexed (row+1, col+1)
        let mut pos_ids = vec![0u32; (seq * 2) as usize];
        for r in 0..h_patches {
            for col in 0..w_patches {
                let idx = (1 + r * w_patches + col) as usize;
                pos_ids[idx * 2] = r + 1;
                pos_ids[idx * 2 + 1] = col + 1;
            }
        }
        let d_pos_ids = alloc(backend, (pos_ids.len() * 4) as u64)?;
        backend.copy_h2d(d_pos_ids.view(), cast_slice(&pos_ids))?;

        let d_norm = alloc(backend, u64::from(seq) * u64::from(width) * 4)?;
        let d_q = alloc(backend, u64::from(seq) * u64::from(width) * 4)?;
        let d_k = alloc(backend, u64::from(seq) * u64::from(width) * 4)?;
        let d_v = alloc(backend, u64::from(seq) * u64::from(width) * 4)?;
        let d_attn = alloc(backend, u64::from(seq) * u64::from(width) * 4)?;
        let d_out = alloc(backend, u64::from(seq) * u64::from(width) * 4)?;
        let d_mlp = alloc(backend, u64::from(seq) * u64::from(mlp_dim) * 4)?;

        let ln_args = VitNormArgs {
            rows: seq,
            hidden: width,
            eps: c.layer_norm_eps,
        };
        let rope_args = VitRopeArgs {
            rows: seq,
            heads,
            head_dim,
            theta: 10_000.0,
        };
        let attn_args = VitAttnArgs {
            rows: seq,
            heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };
        let ls_args = VitLayerScaleArgs {
            rows: seq,
            hidden: width,
        };

        // ln_pre
        backend.run(
            KernelRef::new(MODULE, "kernel_inkling2_vit_layernorm_bf16"),
            &LaunchArgs::new(
                Extent3::linear(seq),
                Extent3::linear(256),
                &[
                    KernelArg::Bytes(bytes_of(&ln_args)),
                    KernelArg::Buffer(d_hidden.view()),
                    KernelArg::Buffer(self.ln_pre_w.view()),
                    KernelArg::Buffer(self.ln_pre_b.view()),
                    KernelArg::Buffer(d_hidden.view()),
                ],
            ),
        )?;

        for lw in &self.layers {
            // ln_1
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_layernorm_bf16"),
                &LaunchArgs::new(
                    Extent3::linear(seq),
                    Extent3::linear(256),
                    &[
                        KernelArg::Bytes(bytes_of(&ln_args)),
                        KernelArg::Buffer(d_hidden.view()),
                        KernelArg::Buffer(lw.ln1_w.view()),
                        KernelArg::Buffer(lw.ln1_b.view()),
                        KernelArg::Buffer(d_norm.view()),
                    ],
                ),
            )?;
            let lin = LinearArgs {
                rows: seq,
                in_dim: width,
                out_dim: width,
            };
            for (w, b, dst) in [
                (&lw.q_w, &lw.q_b, &d_q),
                (&lw.k_w, &lw.k_b, &d_k),
                (&lw.v_w, &lw.v_b, &d_v),
            ] {
                backend.run(
                    KernelRef::new(MODULE, "kernel_inkling2_vit_linear_bias_bf16"),
                    &LaunchArgs::new(
                        Extent3::new(width.div_ceil(4), seq, 1),
                        Extent3::linear(128),
                        &[
                            KernelArg::Bytes(bytes_of(&lin)),
                            KernelArg::Buffer(d_norm.view()),
                            KernelArg::Buffer(w.view()),
                            KernelArg::Buffer(b.view()),
                            KernelArg::Buffer(dst.view()),
                        ],
                    ),
                )?;
            }
            // 2D axial pairwise RoPE on Q and K
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_axial_rope"),
                &LaunchArgs::new(
                    Extent3::new(heads, seq, 1),
                    Extent3::linear(head_dim / 2),
                    &[
                        KernelArg::Bytes(bytes_of(&rope_args)),
                        KernelArg::Buffer(d_pos_ids.view()),
                        KernelArg::Buffer(d_q.view()),
                        KernelArg::Buffer(d_k.view()),
                    ],
                ),
            )?;
            // Bidirectional full attention
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_attn"),
                &LaunchArgs::new(
                    Extent3::new(heads, seq, 1),
                    Extent3::linear(head_dim),
                    &[
                        KernelArg::Bytes(bytes_of(&attn_args)),
                        KernelArg::Buffer(d_q.view()),
                        KernelArg::Buffer(d_k.view()),
                        KernelArg::Buffer(d_v.view()),
                        KernelArg::Buffer(d_attn.view()),
                    ],
                ),
            )?;
            // out_proj + ls_1 residual
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_linear_bias_bf16"),
                &LaunchArgs::new(
                    Extent3::new(width.div_ceil(4), seq, 1),
                    Extent3::linear(128),
                    &[
                        KernelArg::Bytes(bytes_of(&lin)),
                        KernelArg::Buffer(d_attn.view()),
                        KernelArg::Buffer(lw.out_w.view()),
                        KernelArg::Buffer(lw.out_b.view()),
                        KernelArg::Buffer(d_out.view()),
                    ],
                ),
            )?;
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_layerscale_add"),
                &LaunchArgs::new(
                    Extent3::linear((seq * width).div_ceil(256)),
                    Extent3::linear(256),
                    &[
                        KernelArg::Bytes(bytes_of(&ls_args)),
                        KernelArg::Buffer(d_hidden.view()),
                        KernelArg::Buffer(d_out.view()),
                        KernelArg::Buffer(lw.ls1_gamma.view()),
                        KernelArg::Buffer(d_hidden.view()),
                    ],
                ),
            )?;

            // ln_2 + MLP (fc -> GELU -> proj) + ls_2 residual
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_layernorm_bf16"),
                &LaunchArgs::new(
                    Extent3::linear(seq),
                    Extent3::linear(256),
                    &[
                        KernelArg::Bytes(bytes_of(&ln_args)),
                        KernelArg::Buffer(d_hidden.view()),
                        KernelArg::Buffer(lw.ln2_w.view()),
                        KernelArg::Buffer(lw.ln2_b.view()),
                        KernelArg::Buffer(d_norm.view()),
                    ],
                ),
            )?;
            let fc_args = LinearArgs {
                rows: seq,
                in_dim: width,
                out_dim: mlp_dim,
            };
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_linear_bias_bf16"),
                &LaunchArgs::new(
                    Extent3::new(mlp_dim.div_ceil(4), seq, 1),
                    Extent3::linear(128),
                    &[
                        KernelArg::Bytes(bytes_of(&fc_args)),
                        KernelArg::Buffer(d_norm.view()),
                        KernelArg::Buffer(lw.fc_w.view()),
                        KernelArg::Buffer(lw.fc_b.view()),
                        KernelArg::Buffer(d_mlp.view()),
                    ],
                ),
            )?;
            let gelu_args = VitUnaryArgs {
                count: seq * mlp_dim,
            };
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_gelu_erf"),
                &LaunchArgs::new(
                    Extent3::linear(gelu_args.count.div_ceil(256)),
                    Extent3::linear(256),
                    &[
                        KernelArg::Bytes(bytes_of(&gelu_args)),
                        KernelArg::Buffer(d_mlp.view()),
                    ],
                ),
            )?;
            let proj_args = LinearArgs {
                rows: seq,
                in_dim: mlp_dim,
                out_dim: width,
            };
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_linear_bias_bf16"),
                &LaunchArgs::new(
                    Extent3::new(width.div_ceil(4), seq, 1),
                    Extent3::linear(128),
                    &[
                        KernelArg::Bytes(bytes_of(&proj_args)),
                        KernelArg::Buffer(d_mlp.view()),
                        KernelArg::Buffer(lw.proj_w.view()),
                        KernelArg::Buffer(lw.proj_b.view()),
                        KernelArg::Buffer(d_out.view()),
                    ],
                ),
            )?;
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_vit_layerscale_add"),
                &LaunchArgs::new(
                    Extent3::linear((seq * width).div_ceil(256)),
                    Extent3::linear(256),
                    &[
                        KernelArg::Bytes(bytes_of(&ls_args)),
                        KernelArg::Buffer(d_hidden.view()),
                        KernelArg::Buffer(d_out.view()),
                        KernelArg::Buffer(lw.ls2_gamma.view()),
                        KernelArg::Buffer(d_hidden.view()),
                    ],
                ),
            )?;
        }

        // Spatial 2×2 merge (drop CLS at row 0): reorder [gh, m, gw, m, W] -> [gh*gw, m*m*W]
        let n_tokens = grid_h_tokens * grid_w_tokens;
        let merged_dim = m * m * width;
        let mut host_patches = vec![0u8; (u64::from(n_patches) * u64::from(width) * 4) as usize];
        backend.sync()?;
        backend.copy_d2h(patch_dst, &mut host_patches)?;
        let patch_f32: &[f32] = cast_slice(&host_patches);
        let mut merged = vec![0.0_f32; (n_tokens * merged_dim) as usize];
        let gh = grid_h_tokens as usize;
        let gw = grid_w_tokens as usize;
        let ms = m as usize;
        let ws = width as usize;
        for r in 0..gh {
            for col in 0..gw {
                let tok = r * gw + col;
                for mr in 0..ms {
                    for mc in 0..ms {
                        let src_patch = (r * ms + mr) * (gw * ms) + (col * ms + mc);
                        let dst_off = tok * (ms * ms * ws) + (mr * ms + mc) * ws;
                        merged[dst_off..dst_off + ws]
                            .copy_from_slice(&patch_f32[src_patch * ws..(src_patch + 1) * ws]);
                    }
                }
            }
        }

        let d_merged = alloc(backend, (merged.len() * 4).max(4) as u64)?;
        backend.copy_h2d(d_merged.view(), cast_slice(&merged))?;
        let d_proj_mid = alloc(
            backend,
            u64::from(n_tokens) * u64::from(c.projection_hidden_dim) * 4,
        )?;
        let d_proj_out = alloc(
            backend,
            u64::from(n_tokens) * u64::from(c.text_hidden_size) * 4,
        )?;

        ops::linear_bf16(
            backend.as_ref(),
            LinearArgs {
                rows: n_tokens,
                in_dim: merged_dim,
                out_dim: c.projection_hidden_dim,
            },
            d_merged.view(),
            self.proj_in_w.view(),
            d_proj_mid.view(),
        )?;
        let p_gelu = VitUnaryArgs {
            count: n_tokens * c.projection_hidden_dim,
        };
        backend.run(
            KernelRef::new(MODULE, "kernel_inkling2_vit_gelu_erf"),
            &LaunchArgs::new(
                Extent3::linear(p_gelu.count.div_ceil(256)),
                Extent3::linear(256),
                &[
                    KernelArg::Bytes(bytes_of(&p_gelu)),
                    KernelArg::Buffer(d_proj_mid.view()),
                ],
            ),
        )?;
        ops::linear_bf16(
            backend.as_ref(),
            LinearArgs {
                rows: n_tokens,
                in_dim: c.projection_hidden_dim,
                out_dim: c.text_hidden_size,
            },
            d_proj_mid.view(),
            self.proj_out_w.view(),
            d_proj_out.view(),
        )?;
        Ok(d_proj_out)
    }
}

impl VisionTower for Inkling2VisionTower {
    fn preprocess(&self) -> &PreprocessSpec {
        &self.preprocess
    }

    fn placement(&self) -> MediaPlacement {
        MediaPlacement::INKLING2
    }

    fn out_hidden(&self) -> u32 {
        self.config.text_hidden_size
    }

    fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        items: &[VisionItem],
    ) -> Result<EncodedRows, VisionError> {
        let mut parts = Vec::with_capacity(items.len());
        let mut grids = Vec::with_capacity(items.len());
        for item in items {
            let (_t, grid_h, grid_w) = item.grid();
            let gh = (grid_h as u32) / self.config.adapter_merge_size;
            let gw = (grid_w as u32) / self.config.adapter_merge_size;
            let encoded = self
                .encode_patches(backend, item.as_flat(), gh, gw)
                .map_err(|source| VisionError::Encode {
                    step: "inkling2 vision encode",
                    source,
                })?;
            parts.push(encoded);
            grids.push(VisionGrid::new(1, gh, gw));
        }
        EncodedRows::gather(
            backend,
            &parts,
            grids,
            u64::from(self.config.text_hidden_size) * 4,
        )
    }
}

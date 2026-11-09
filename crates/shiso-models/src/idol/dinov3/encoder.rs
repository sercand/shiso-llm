//! The tower's graph: patches in, one token per patch plus five out.
//!
//! A 16x16 stride-16 convolution embeds the patches; the `CLS` token and the
//! four registers are prepended; 32 pre-norm blocks follow, each
//! `x += lambda1 * attn(norm1(x))` and `x += lambda2 * mlp(norm2(x))` with a
//! gated `SiLU` MLP and the rotary turn on the patch rows only; then the
//! model's own affine norm and the conditioning's parameter-free one. The
//! output is the conditioning stream, `(5 + patches, hidden)`.
//!
//! The rotary turn uses the half-split pairing ([`Pairing::Halves`]) with the
//! tables of [`super::rope_tables`]; the five prefix rows get the identity
//! (cosine one, sine zero), which is how the reference leaves them unrotated.

use std::path::Path;

use shiso_core::DeviceBuffer;
use shiso_vision::generator::ImageGenError;

use super::{DinoConfig, normalize_pixels, rope_tables};
use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{Conv2dCall, Ops, SliceExt};
use crate::diffusion::weights::StStore;
use crate::idol::nn::{
    Affine, AttnShape, Pairing, Qkv, add_gated_in_place, attention, bytes, host_lin, layer_norm,
    linear, read_linear, rotate_in_place, turned,
};

/// Epsilon of the conditioning's parameter-free norm (`F.layer_norm`'s
/// default), applied after the model's own.
const COND_NORM_EPS: f32 = 1e-5;

/// The full-width rotary tables, `cos - 1` and `sin`.
#[derive(Debug, Clone, Copy)]
struct Tables<'a> {
    cos_m1: shiso_core::DeviceView<'a>,
    sin: shiso_core::DeviceView<'a>,
}

/// One pre-norm block.
#[derive(Debug)]
struct Block {
    norm1: Affine,
    q: Lin,
    q_turn: Lin,
    k: Lin,
    k_turn: Lin,
    v: Lin,
    o: Lin,
    scale1: DeviceBuffer,
    norm2: Affine,
    gate: Lin,
    up: Lin,
    down: Lin,
    scale2: DeviceBuffer,
}

/// The loaded tower.
#[derive(Debug)]
pub(crate) struct DinoV3 {
    config: DinoConfig,
    patch_w: DeviceBuffer,
    patch_b: DeviceBuffer,
    /// The `CLS` token and the registers, `(1 + registers, hidden)`.
    prefix: Vec<f32>,
    blocks: Vec<Block>,
    norm: Affine,
}

/// Loads a query or key projection with its turned twin.
fn load_turned(
    store: &StStore,
    ops: &Ops,
    name: &str,
    config: &DinoConfig,
) -> Result<(Lin, Lin), ImageGenError> {
    let (w, bias) = read_linear(store, name)?;
    let (tw, tb) = turned(
        &w,
        bias.as_deref(),
        config.heads,
        config.head_dim(),
        Pairing::Halves,
    );
    Ok((
        host_lin(ops, &w, bias.as_deref(), config.hidden, config.hidden)?,
        host_lin(ops, &tw, tb.as_deref(), config.hidden, config.hidden)?,
    ))
}

impl DinoV3 {
    /// Opens the tower at `dir` (its `model.safetensors` and `config.json`).
    ///
    /// The shape is [`DinoConfig::vit_h_plus`]; the config document is read
    /// and refused when it names another, because a head count read from the
    /// wrong document would reshape every attention without an error.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a missing tensor or another shape.
    pub(crate) fn load(dir: &Path, ops: &Ops) -> Result<Self, ImageGenError> {
        let config = DinoConfig::vit_h_plus();
        check_config(&dir.join("config.json"), &config)?;
        let store = StStore::open(dir)?;
        let mut prefix = store.f32_vec("embeddings.cls_token")?;
        prefix.extend(store.f32_vec("embeddings.register_tokens")?);
        if prefix.len() != (1 + config.registers) * config.hidden {
            return Err(ImageGenError::Checkpoint(format!(
                "the CLS and register tokens are {} values, expected {}",
                prefix.len(),
                (1 + config.registers) * config.hidden
            )));
        }
        let blocks = (0..config.layers)
            .map(|i| {
                let base = format!("layer.{i}.");
                let (q, q_turn) =
                    load_turned(&store, ops, &format!("{base}attention.q_proj"), &config)?;
                let (k, k_turn) =
                    load_turned(&store, ops, &format!("{base}attention.k_proj"), &config)?;
                Ok(Block {
                    norm1: Affine::load(&store, ops, &format!("{base}norm1"))?,
                    q,
                    q_turn,
                    k,
                    k_turn,
                    v: Lin::load(&store, ops, &format!("{base}attention.v_proj"))?,
                    o: Lin::load(&store, ops, &format!("{base}attention.o_proj"))?,
                    scale1: store.f32_widen(
                        ops,
                        "dino layer scale",
                        &format!("{base}layer_scale1.lambda1"),
                    )?,
                    norm2: Affine::load(&store, ops, &format!("{base}norm2"))?,
                    gate: Lin::load(&store, ops, &format!("{base}mlp.gate_proj"))?,
                    up: Lin::load(&store, ops, &format!("{base}mlp.up_proj"))?,
                    down: Lin::load(&store, ops, &format!("{base}mlp.down_proj"))?,
                    scale2: store.f32_widen(
                        ops,
                        "dino layer scale",
                        &format!("{base}layer_scale2.lambda1"),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, ImageGenError>>()?;
        Ok(Self {
            config,
            patch_w: store.f32_widen(
                ops,
                "dino patch embed",
                "embeddings.patch_embeddings.weight",
            )?,
            patch_b: store.f32_widen(
                ops,
                "dino patch embed",
                "embeddings.patch_embeddings.bias",
            )?,
            prefix,
            blocks,
            norm: Affine::load(&store, ops, "norm")?,
        })
    }

    /// The conditioning stream for a `(3, height, width)` picture in
    /// `[0, 1]`: `(1 + registers + patches, hidden)`, `CLS` first, then the
    /// registers, then the patches row by row.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Backend`] for a picture that is not a whole number of
    /// patches, or a device failure.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the layer-norm epsilon is 1e-5, exact enough in f32"
    )]
    pub(crate) fn tokens(
        &self,
        ops: &Ops,
        rgb: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        let c = &self.config;
        if width == 0
            || height == 0
            || !width.is_multiple_of(c.patch)
            || !height.is_multiple_of(c.patch)
        {
            return Err(ImageGenError::Backend(format!(
                "a {width}x{height} picture is not a whole number of {}-pixel patches",
                c.patch
            )));
        }
        if rgb.len() != 3 * width * height {
            return Err(ImageGenError::Backend(format!(
                "the picture is {} values, a {width}x{height} RGB field is {}",
                rgb.len(),
                3 * width * height
            )));
        }
        let (grid_h, grid_w) = (height / c.patch, width / c.patch);
        let patches = grid_h * grid_w;
        let prefix_rows = 1 + c.registers;
        let tokens = prefix_rows + patches;
        let hidden = c.hidden;
        let eps = c.layer_norm_eps as f32;

        let x = self.embed(ops, rgb, width, height)?;
        let (cos_m1, sin) = self.rope(ops, grid_h, grid_w)?;

        let count = tokens * hidden;
        let tables = Tables {
            cos_m1: cos_m1.view(),
            sin: sin.view(),
        };
        for block in &self.blocks {
            self.block(ops, block, x.view(), tables, tokens)?;
        }

        let normed = layer_norm(
            ops,
            "dino norm",
            x.view(),
            tokens,
            hidden,
            eps,
            Some(&self.norm),
        )?;
        let out = layer_norm(
            ops,
            "dino cond norm",
            normed.view(),
            tokens,
            hidden,
            COND_NORM_EPS,
            None,
        )?;
        let values = ops.download("dino readback", out.view(), count)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(ImageGenError::Backend(
                "the image tower produced a non-finite token".to_owned(),
            ));
        }
        Ok(values)
    }

    /// One pre-norm block over `tokens` rows of `x`, in place.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the layer-norm epsilon is 1e-5, exact enough in f32"
    )]
    fn block(
        &self,
        ops: &Ops,
        block: &Block,
        x: shiso_core::DeviceView<'_>,
        tables: Tables<'_>,
        tokens: usize,
    ) -> Result<(), ImageGenError> {
        let cfg = &self.config;
        let hidden = cfg.hidden;
        let eps = cfg.layer_norm_eps as f32;
        let count = tokens * hidden;
        let normed = layer_norm(
            ops,
            "dino norm1",
            x,
            tokens,
            hidden,
            eps,
            Some(&block.norm1),
        )?;
        let queries = linear(ops, "dino q", &block.q, normed.view(), tokens)?;
        let turned_q = linear(ops, "dino q", &block.q_turn, normed.view(), tokens)?;
        rotate_in_place(
            ops,
            "dino rope",
            queries.view(),
            turned_q.view(),
            tables.cos_m1,
            tables.sin,
            count,
        )?;
        drop(turned_q);
        let keys = linear(ops, "dino k", &block.k, normed.view(), tokens)?;
        let turned_k = linear(ops, "dino k", &block.k_turn, normed.view(), tokens)?;
        rotate_in_place(
            ops,
            "dino rope",
            keys.view(),
            turned_k.view(),
            tables.cos_m1,
            tables.sin,
            count,
        )?;
        drop(turned_k);
        let values = linear(ops, "dino v", &block.v, normed.view(), tokens)?;
        drop(normed);
        let attended = attention(
            ops,
            "dino attention",
            Qkv {
                q: queries.view(),
                k: keys.view(),
                v: values.view(),
            },
            AttnShape {
                queries: tokens,
                keys: tokens,
                heads: cfg.heads,
                head_dim: cfg.head_dim(),
            },
        )?;
        drop((queries, keys, values));
        let projected = linear(ops, "dino o", &block.o, attended.view(), tokens)?;
        add_gated_in_place(
            ops,
            "dino residual",
            x,
            block.scale1.view(),
            projected.view(),
            count,
            hidden,
        )?;
        drop((attended, projected));

        let normed = layer_norm(
            ops,
            "dino norm2",
            x,
            tokens,
            hidden,
            eps,
            Some(&block.norm2),
        )?;
        let gate = linear(ops, "dino gate", &block.gate, normed.view(), tokens)?;
        let up = linear(ops, "dino up", &block.up, normed.view(), tokens)?;
        drop(normed);
        ops.swiglu2(
            "dino swiglu",
            gate.view(),
            gate.view(),
            up.view(),
            tokens * cfg.intermediate,
        )?;
        drop(up);
        let down = linear(ops, "dino down", &block.down, gate.view(), tokens)?;
        add_gated_in_place(
            ops,
            "dino residual",
            x,
            block.scale2.view(),
            down.view(),
            count,
            hidden,
        )?;
        Ok(())
    }

    /// The token rows before the blocks: the prefix, then the patch
    /// embeddings in row-major patch order.
    fn embed<'o>(
        &self,
        ops: &'o Ops,
        rgb: &[f32],
        width: usize,
        height: usize,
    ) -> Result<crate::diffusion::ops::Scratch<'o>, ImageGenError> {
        let c = &self.config;
        let (grid_h, grid_w) = (height / c.patch, width / c.patch);
        let patches = grid_h * grid_w;
        let prefix_rows = 1 + c.registers;
        let pixels = ops.upload_scratch("dino pixels", &normalize_pixels(rgb))?;
        let planes = ops.alloc_f32("dino patch embed", c.hidden * patches)?;
        ops.conv2d(&Conv2dCall {
            step: "dino patch embed",
            out: planes.view(),
            input: pixels.view(),
            weight: self.patch_w.view(),
            bias: Some(self.patch_b.view()),
            batch: 1,
            in_c: 3,
            in_h: height,
            in_w: width,
            out_c: c.hidden,
            out_h: grid_h,
            out_w: grid_w,
            k_h: c.patch,
            k_w: c.patch,
            stride: c.patch,
            pad: 0,
        })?;
        let x = ops.alloc_f32("dino tokens", (prefix_rows + patches) * c.hidden)?;
        let prefix_bytes: Vec<u8> = self.prefix.iter().flat_map(|v| v.to_le_bytes()).collect();
        ops.backend()
            .copy_h2d(
                x.view().cut(0, bytes(prefix_rows * c.hidden))?,
                &prefix_bytes,
            )
            .map_err(|e| ImageGenError::Backend(format!("dino prefix: {e}")))?;
        ops.transpose(
            "dino patch rows",
            x.view()
                .cut(bytes(prefix_rows * c.hidden), bytes(patches * c.hidden))?,
            planes.view(),
            c.hidden,
            patches,
        )?;
        Ok(x)
    }

    /// The full-width `cos - 1` and `sin` tables: identity on the prefix rows,
    /// the patch tables tiled across the heads on the rest.
    fn rope(
        &self,
        ops: &Ops,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<(DeviceBuffer, DeviceBuffer), ImageGenError> {
        let c = &self.config;
        let head_dim = c.head_dim();
        let prefix_rows = 1 + c.registers;
        let (cos, sin) = rope_tables(c, grid_h, grid_w);
        let rows = prefix_rows + grid_h * grid_w;
        let mut cos_m1 = vec![0.0f32; rows * c.hidden];
        let mut sin_full = vec![0.0f32; rows * c.hidden];
        for patch in 0..grid_h * grid_w {
            let row = prefix_rows + patch;
            for head in 0..c.heads {
                for d in 0..head_dim {
                    let at = row * c.hidden + head * head_dim + d;
                    cos_m1[at] = cos[patch * head_dim + d] - 1.0;
                    sin_full[at] = sin[patch * head_dim + d];
                }
            }
        }
        Ok((
            ops.upload_f32("dino rope", &cos_m1)?,
            ops.upload_f32("dino rope", &sin_full)?,
        ))
    }
}

/// The fields of the tower's `config.json` that fix its shape.
#[derive(Debug, serde::Deserialize)]
struct DinoDocument {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_register_tokens: usize,
    patch_size: usize,
    intermediate_size: usize,
    hidden_act: String,
    use_gated_mlp: bool,
}

/// Refuses a config document that does not describe `config`.
fn check_config(path: &Path, config: &DinoConfig) -> Result<(), ImageGenError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ImageGenError::Checkpoint(format!("{}: {e}", path.display())))?;
    let doc: DinoDocument = serde_json::from_str(&text)
        .map_err(|e| ImageGenError::Checkpoint(format!("{}: {e}", path.display())))?;
    let found = (
        doc.hidden_size,
        doc.num_hidden_layers,
        doc.num_attention_heads,
        doc.num_register_tokens,
        doc.patch_size,
        doc.intermediate_size,
    );
    let want = (
        config.hidden,
        config.layers,
        config.heads,
        config.registers,
        config.patch,
        config.intermediate,
    );
    if found != want || doc.hidden_act != "silu" || !doc.use_gated_mlp {
        return Err(ImageGenError::Checkpoint(format!(
            "{} describes (hidden, layers, heads, registers, patch, intermediate) {found:?} with a \
             {} {} MLP; this tower is {want:?} with a gated silu MLP",
            path.display(),
            if doc.use_gated_mlp { "gated" } else { "plain" },
            doc.hidden_act
        )));
    }
    Ok(())
}

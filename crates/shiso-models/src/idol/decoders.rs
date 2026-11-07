//! The two decoders: where the anchors go, and what each anchor becomes.
//!
//! Both read the denoised latent (8192 tokens of 16) through cross attention
//! and embed 3D points with the same sinusoids. The octree decoder answers,
//! for every occupied cell at a level, eight logits for how the cell's points
//! split among its children; its four blocks are modulated by the level. The
//! Gaussian decoder answers 480 values per anchor (see
//! [`super::gaussian::Field`]); its sixteen blocks add self attention among
//! the anchors.

use std::path::Path;
use std::sync::Arc;

use shiso_core::DeviceBuffer;
use shiso_vision::generator::ImageGenError;

use super::config::IdolConfig;
use super::denoiser::{HostLinear, silu, sinusoids};
use super::nn::{
    Affine, AttnShape, HeadNorm, Qkv, add_gated_in_place, attention, bytes, host_lin, layer_norm,
    linear, point_embed_v2, read_linear,
};
use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{ModulateCall, Ops, Scratch, SliceExt};
use crate::diffusion::weights::StStore;

/// The block norms' epsilon.
const BLOCK_EPS: f32 = 1e-6;
/// The read-out norm's epsilon (`F.layer_norm`'s default).
const OUT_EPS: f32 = 1e-5;
/// The level embedding's longest period.
const LEVEL_PERIOD: f64 = 1024.0;

/// A cross attention onto the latent: queries from the points, keys and
/// values from the latent's tokens.
#[derive(Debug)]
struct Cross {
    q: Lin,
    k: Lin,
    v: Lin,
    q_norm: HeadNorm,
    k_norm: HeadNorm,
    out: Lin,
}

/// A self attention among the points.
#[derive(Debug)]
struct SelfAttn {
    q: Lin,
    k: Lin,
    v: Lin,
    q_norm: HeadNorm,
    k_norm: HeadNorm,
    out: Lin,
}

/// An MLP of two linears around a tanh GELU.
#[derive(Debug)]
struct Mlp {
    up: Lin,
    down: Lin,
}

/// The input side both decoders share: `input_layer(in_proj(p) + embed(p))`.
#[derive(Debug)]
struct PointInput {
    in_proj: Lin,
    input_layer: Lin,
    out_proj: Lin,
    width: usize,
}

/// The shape both decoders run at.
#[derive(Debug, Clone, Copy)]
struct Shape {
    width: usize,
    heads: usize,
    head_dim: usize,
}

fn split_rows(
    store: &StStore,
    ops: &Ops,
    name: &str,
    parts: usize,
    width: usize,
) -> Result<Vec<Lin>, ImageGenError> {
    let (w, b) = read_linear(store, name)?;
    let b = b.ok_or_else(|| ImageGenError::Checkpoint(format!("{name} has no bias")))?;
    let in_dim = w.len() / (parts * width);
    if w.len() != parts * width * in_dim || b.len() != parts * width {
        return Err(ImageGenError::Checkpoint(format!(
            "{name} is not {parts} x {width} rows"
        )));
    }
    (0..parts)
        .map(|i| {
            host_lin(
                ops,
                &w[i * width * in_dim..(i + 1) * width * in_dim],
                Some(&b[i * width..(i + 1) * width]),
                width,
                in_dim,
            )
        })
        .collect()
}

fn load_cross(
    store: &StStore,
    ops: &Ops,
    base: &str,
    shape: Shape,
) -> Result<Cross, ImageGenError> {
    let mut kv = split_rows(store, ops, &format!("{base}to_kv"), 2, shape.width)?;
    let v = kv
        .pop()
        .ok_or_else(|| ImageGenError::Checkpoint(format!("{base}to_kv")))?;
    let k = kv
        .pop()
        .ok_or_else(|| ImageGenError::Checkpoint(format!("{base}to_kv")))?;
    Ok(Cross {
        q: Lin::load(store, ops, &format!("{base}to_q"))?,
        k,
        v,
        q_norm: HeadNorm::load(
            store,
            ops,
            &format!("{base}q_rms_norm"),
            shape.heads,
            shape.head_dim,
        )?,
        k_norm: HeadNorm::load(
            store,
            ops,
            &format!("{base}k_rms_norm"),
            shape.heads,
            shape.head_dim,
        )?,
        out: Lin::load(store, ops, &format!("{base}to_out"))?,
    })
}

fn load_self(
    store: &StStore,
    ops: &Ops,
    base: &str,
    shape: Shape,
) -> Result<SelfAttn, ImageGenError> {
    let mut qkv = split_rows(store, ops, &format!("{base}to_qkv"), 3, shape.width)?;
    let missing = || ImageGenError::Checkpoint(format!("{base}to_qkv"));
    let v = qkv.pop().ok_or_else(missing)?;
    let k = qkv.pop().ok_or_else(missing)?;
    let q = qkv.pop().ok_or_else(missing)?;
    Ok(SelfAttn {
        q,
        k,
        v,
        q_norm: HeadNorm::load(
            store,
            ops,
            &format!("{base}q_rms_norm"),
            shape.heads,
            shape.head_dim,
        )?,
        k_norm: HeadNorm::load(
            store,
            ops,
            &format!("{base}k_rms_norm"),
            shape.heads,
            shape.head_dim,
        )?,
        out: Lin::load(store, ops, &format!("{base}to_out"))?,
    })
}

fn load_mlp(store: &StStore, ops: &Ops, base: &str) -> Result<Mlp, ImageGenError> {
    Ok(Mlp {
        up: Lin::load(store, ops, &format!("{base}mlp.mlp.0"))?,
        down: Lin::load(store, ops, &format!("{base}mlp.mlp.2"))?,
    })
}

impl PointInput {
    fn load(store: &StStore, ops: &Ops, width: usize) -> Result<Self, ImageGenError> {
        Ok(Self {
            in_proj: Lin::load(store, ops, "in_proj")?,
            input_layer: Lin::load(store, ops, "input_layer")?,
            out_proj: Lin::load(store, ops, "out_proj")?,
            width,
        })
    }

    /// The points' rows before the blocks.
    fn embed<'o>(&self, ops: &'o Ops, points: &[f32]) -> Result<Scratch<'o>, ImageGenError> {
        let n = points.len() / 3;
        let p = ops.upload_scratch("decoder points", points)?;
        let projected = linear(ops, "decoder in", &self.in_proj, p.view(), n)?;
        let embedded = ops.upload_scratch("decoder points", &point_embed_v2(points, self.width))?;
        ops.add(
            "decoder in",
            projected.view(),
            projected.view(),
            embedded.view(),
            n * self.width,
        )?;
        linear(ops, "decoder in", &self.input_layer, projected.view(), n)
    }

    /// The read-out: a parameter-free norm, then the head; downloaded.
    fn read_out(&self, ops: &Ops, x: &DeviceBuffer, n: usize) -> Result<Vec<f32>, ImageGenError> {
        let normed = layer_norm(ops, "decoder out", x.view(), n, self.width, OUT_EPS, None)?;
        let out = linear(ops, "decoder out", &self.out_proj, normed.view(), n)?;
        let values = ops.download("decoder readback", out.view(), n * self.out_proj.out_dim)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(ImageGenError::Backend(
                "a decoder produced a non-finite output".to_owned(),
            ));
        }
        Ok(values)
    }
}

/// `x += out(attention(q_norm(q), k_norm(k), v))` with `q` from `h` and `k`,
/// `v` from `context` (`keys` rows).
#[expect(
    clippy::many_single_char_names,
    reason = "q, k, v, o, h and x are the attention's own names"
)]
fn attend_into(
    ops: &Ops,
    x: shiso_core::DeviceView<'_>,
    gate: Option<shiso_core::DeviceView<'_>>,
    parts: (&Lin, &Lin, &Lin, &HeadNorm, &HeadNorm, &Lin),
    inputs: (shiso_core::DeviceView<'_>, shiso_core::DeviceView<'_>),
    rows: (usize, usize),
    shape: Shape,
) -> Result<(), ImageGenError> {
    let (lq, lk, lv, q_norm, k_norm, lo) = parts;
    let (h, context) = inputs;
    let (queries, keys) = rows;
    let q = linear(ops, "decoder q", lq, h, queries)?;
    q_norm.apply(ops, "decoder q norm", q.view(), queries)?;
    let k = linear(ops, "decoder k", lk, context, keys)?;
    k_norm.apply(ops, "decoder k norm", k.view(), keys)?;
    let v = linear(ops, "decoder v", lv, context, keys)?;
    let attended = attention(
        ops,
        "decoder attention",
        Qkv {
            q: q.view(),
            k: k.view(),
            v: v.view(),
        },
        AttnShape {
            queries,
            keys,
            heads: shape.heads,
            head_dim: shape.head_dim,
        },
    )?;
    drop((q, k, v));
    let o = linear(ops, "decoder attn out", lo, attended.view(), queries)?;
    let count = queries * shape.width;
    match gate {
        Some(gate) => add_gated_in_place(
            ops,
            "decoder residual",
            x,
            gate,
            o.view(),
            count,
            shape.width,
        ),
        None => ops.add("decoder residual", x, x, o.view(), count),
    }
}

/// `x += [gate *] mlp(h)`.
fn mlp_into(
    ops: &Ops,
    x: shiso_core::DeviceView<'_>,
    gate: Option<shiso_core::DeviceView<'_>>,
    mlp: &Mlp,
    h: shiso_core::DeviceView<'_>,
    rows: usize,
) -> Result<(), ImageGenError> {
    let up = linear(ops, "decoder mlp", &mlp.up, h, rows)?;
    ops.gelu_tanh("decoder mlp", up.view(), up.view(), rows * mlp.up.out_dim)?;
    let down = linear(ops, "decoder mlp", &mlp.down, up.view(), rows)?;
    let count = rows * mlp.down.out_dim;
    match gate {
        Some(gate) => add_gated_in_place(
            ops,
            "decoder residual",
            x,
            gate,
            down.view(),
            count,
            mlp.down.out_dim,
        ),
        None => ops.add("decoder residual", x, x, down.view(), count),
    }
}

/// One octree block: level-modulated cross attention and MLP.
#[derive(Debug)]
struct OctreeBlock {
    cross: Cross,
    mlp: Mlp,
}

/// The loaded octree decoder.
#[derive(Debug)]
pub(crate) struct OctreeNet {
    ops: Arc<Ops>,
    shape: Shape,
    input: PointInput,
    level_embed: [HostLinear; 2],
    modulation: HostLinear,
    blocks: Vec<OctreeBlock>,
    cond_channels: usize,
}

impl OctreeNet {
    /// Opens `dir/decoder.safetensors` under `dir/config.json`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a document or tensor of another shape.
    pub(crate) fn load(dir: &Path, ops: Arc<Ops>) -> Result<Self, ImageGenError> {
        let args = IdolConfig::read(dir)?.octree;
        let store = StStore::open_files(&[dir.join("decoder.safetensors")])?;
        let shape = Shape {
            width: args.model_channels,
            heads: args.num_heads,
            head_dim: args.model_channels / args.num_heads,
        };
        let blocks = (0..args.num_blocks)
            .map(|i| {
                let base = format!("blocks.{i}.");
                Ok(OctreeBlock {
                    cross: load_cross(&store, &ops, &format!("{base}cross_attn."), shape)?,
                    mlp: load_mlp(&store, &ops, &base)?,
                })
            })
            .collect::<Result<Vec<_>, ImageGenError>>()?;
        Ok(Self {
            input: PointInput::load(&store, &ops, shape.width)?,
            level_embed: [
                HostLinear::load(&store, "l_embedder.mlp.0")?,
                HostLinear::load(&store, "l_embedder.mlp.2")?,
            ],
            modulation: HostLinear::load(&store, "adaLN_modulation.1")?,
            blocks,
            cond_channels: args.cond_channels,
            shape,
            ops,
        })
    }

    /// The child logits, `(cells, 8)`, for cells whose centres are `centres`
    /// (`(cells, 3)` in `[0, 1]`) and whose children sit at `resolution`
    /// cells per side, under `latent`.
    ///
    /// # Errors
    ///
    /// A device failure, or a latent that is not whole tokens.
    pub(crate) fn logits(
        &self,
        centres: &[f32],
        resolution: f32,
        latent: &[f32],
    ) -> Result<Vec<f32>, ImageGenError> {
        let ops = &*self.ops;
        let Shape { width, .. } = self.shape;
        let cells = centres.len() / 3;
        let keys = latent.len() / self.cond_channels;
        if keys * self.cond_channels != latent.len() || cells * 3 != centres.len() {
            return Err(ImageGenError::Backend(
                "the octree decoder was handed a ragged latent or cell list".to_owned(),
            ));
        }
        let level = self.level_embed[1].apply(
            &self.level_embed[0]
                .apply(&sinusoids(resolution, LEVEL_PERIOD, true))
                .into_iter()
                .map(silu)
                .collect::<Vec<_>>(),
        );
        let modulation = self
            .modulation
            .apply(&level.into_iter().map(silu).collect::<Vec<_>>());
        let modulation = ops.upload_scratch("octree modulation", &modulation)?;
        let chunk = |i: usize| modulation.view().cut(bytes(i * width), bytes(width));
        let context = ops.upload_scratch("octree latent", latent)?;
        let x = self.input.embed(ops, centres)?;
        let count = cells * width;
        for block in &self.blocks {
            for (which, step) in [(0, "octree norm1"), (1, "octree norm2")] {
                let h = layer_norm(ops, step, x.view(), cells, width, BLOCK_EPS, None)?;
                ops.modulate(&ModulateCall {
                    step,
                    out: h.view(),
                    x: h.view(),
                    scale: chunk(3 * which + 1)?,
                    shift: chunk(3 * which)?,
                    count,
                    period: width,
                })?;
                let gate = Some(chunk(3 * which + 2)?);
                if which == 0 {
                    let c = &block.cross;
                    attend_into(
                        ops,
                        x.view(),
                        gate,
                        (&c.q, &c.k, &c.v, &c.q_norm, &c.k_norm, &c.out),
                        (h.view(), context.view()),
                        (cells, keys),
                        self.shape,
                    )?;
                } else {
                    mlp_into(ops, x.view(), gate, &block.mlp, h.view(), cells)?;
                }
            }
        }
        self.input.read_out(ops, &x, cells)
    }
}

/// One Gaussian decoder block.
#[derive(Debug)]
struct GaussianBlock {
    self_attn: SelfAttn,
    norm2: Affine,
    cross: Cross,
    mlp: Mlp,
}

/// The loaded Gaussian decoder.
#[derive(Debug)]
pub(crate) struct GaussianNet {
    ops: Arc<Ops>,
    shape: Shape,
    input: PointInput,
    blocks: Vec<GaussianBlock>,
    cond_channels: usize,
    /// The fixed per-Gaussian offset pattern, `(per_anchor, 3)`.
    pub(crate) perturbation: Vec<[f32; 3]>,
    /// The checkpoint's base offset scale.
    pub(crate) base_offset_scale: f32,
}

impl GaussianNet {
    /// Opens `dir/decoder_gs.safetensors` under `dir/config.json`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a document or tensor of another shape.
    pub(crate) fn load(dir: &Path, ops: Arc<Ops>) -> Result<Self, ImageGenError> {
        let args = IdolConfig::read(dir)?.gaussian;
        let store = StStore::open_files(&[dir.join("decoder_gs.safetensors")])?;
        let shape = Shape {
            width: args.model_channels,
            heads: args.num_heads,
            head_dim: args.model_channels / args.num_heads,
        };
        let blocks = (0..args.num_blocks)
            .map(|i| {
                let base = format!("blocks.{i}.");
                Ok(GaussianBlock {
                    self_attn: load_self(&store, &ops, &format!("{base}self_attn."), shape)?,
                    norm2: Affine::load(&store, &ops, &format!("{base}norm2"))?,
                    cross: load_cross(&store, &ops, &format!("{base}cross_attn."), shape)?,
                    mlp: load_mlp(&store, &ops, &base)?,
                })
            })
            .collect::<Result<Vec<_>, ImageGenError>>()?;
        let table = store.f32_vec("points_offset_perturbation")?;
        let per_anchor = args.representation_config.num_gaussians;
        if table.len() != per_anchor * 3 {
            return Err(ImageGenError::Checkpoint(format!(
                "points_offset_perturbation is {} values, expected {per_anchor} x 3",
                table.len()
            )));
        }
        Ok(Self {
            input: PointInput::load(&store, &ops, shape.width)?,
            blocks,
            cond_channels: args.cond_channels,
            perturbation: table.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            base_offset_scale: store.f32_scalar("base_offset_scale")?,
            shape,
            ops,
        })
    }

    /// The raw outputs, `(anchors, 480)`, for `anchors` (`(n, 3)` in
    /// `[0, 1]`) under `latent`.
    ///
    /// # Errors
    ///
    /// A device failure, or a latent that is not whole tokens.
    #[expect(
        clippy::many_single_char_names,
        reason = "x, h, n, s and c are the block's stream, norm, rows and two attentions"
    )]
    pub(crate) fn features(
        &self,
        anchors: &[f32],
        latent: &[f32],
    ) -> Result<Vec<f32>, ImageGenError> {
        let ops = &*self.ops;
        let Shape { width, .. } = self.shape;
        let n = anchors.len() / 3;
        let keys = latent.len() / self.cond_channels;
        if keys * self.cond_channels != latent.len() || n * 3 != anchors.len() {
            return Err(ImageGenError::Backend(
                "the Gaussian decoder was handed a ragged latent or anchor list".to_owned(),
            ));
        }
        let context = ops.upload_scratch("gaussian latent", latent)?;
        let x = self.input.embed(ops, anchors)?;
        for block in &self.blocks {
            let h = layer_norm(ops, "gaussian norm1", x.view(), n, width, BLOCK_EPS, None)?;
            let s = &block.self_attn;
            attend_into(
                ops,
                x.view(),
                None,
                (&s.q, &s.k, &s.v, &s.q_norm, &s.k_norm, &s.out),
                (h.view(), h.view()),
                (n, n),
                self.shape,
            )?;
            let h = layer_norm(
                ops,
                "gaussian norm2",
                x.view(),
                n,
                width,
                BLOCK_EPS,
                Some(&block.norm2),
            )?;
            let c = &block.cross;
            attend_into(
                ops,
                x.view(),
                None,
                (&c.q, &c.k, &c.v, &c.q_norm, &c.k_norm, &c.out),
                (h.view(), context.view()),
                (n, keys),
                self.shape,
            )?;
            let h = layer_norm(ops, "gaussian norm3", x.view(), n, width, BLOCK_EPS, None)?;
            mlp_into(ops, x.view(), None, &block.mlp, h.view(), n)?;
        }
        self.input.read_out(ops, &x, n)
    }
}

//! The Qwen-Image-2.1 diffusion transformer with its prefix key/value cache.
//!
//! Every block is one stream: the text tokens and the target image's latent
//! tokens run through the same weights, told apart only by their
//! modulation and their attention mask. A block is
//!
//! ```text
//! h  = LN(x) * (1 + scale1)                       (LN: no affine, eps 1e-6)
//! q, k, v = to_q(h), to_k(h), to_v(h)             (three linears)
//! q, k = RoPE(RMSNorm_head(q) * norm_q), RoPE(RMSNorm_head(k) * norm_k)
//! x += tanh(gate1) * to_out(attention(q, k, v))
//! h  = LN(x) * (1 + scale2)
//! x += tanh(gate2) * out(silu(gate(h)) * proj(h))
//! ```
//!
//! and the head is `proj_out(LN(x) * (1 + norm_out.linear(silu(temb))))`,
//! the velocity. The modulation is shared by every block (see
//! [`super::embed`]).
//!
//! Under `causal_condition` the text tokens are modulated at `t = 0`, attend
//! causally among themselves and never see the target; the target attends
//! to every text token and every target token. So nothing the text tokens
//! compute depends on the step, and a job splits into
//!
//! - [`QwenDit::prefix`], once: `txt_in`, then all blocks over the `P` text
//!   tokens with causal attention, keeping each layer's keys and values
//!   after `QK-norm` and `RoPE` ([`PrefixKv`]). The last block's attention
//!   output and MLP feed nothing and are skipped.
//! - [`QwenDit::velocity`], per step: `img_in`, then all blocks over the `T`
//!   target tokens only, each attending with the target's queries over the
//!   layer's cached prefix keys/values followed by the target's own. The
//!   cached rows are bound as the fused attention's extra keys in place, so
//!   no step copies the prefix.
//!
//! This is exact, not an approximation: the reference's full block-causal
//! pass and this split agree to 0.0 on its fixtures.
//!
//! Weights run in their stored format through [`Lin`]: the block linears
//! are per-row INT8 with a `ConvRot` input rotation (rotated by the GEMM
//! entry), everything else BF16. The fused MLP input `img_mlp.gate_up`
//! (gate rows first) runs as one product whose `[gate | value]` rows the
//! output linear's `SwiGLU` reader consumes directly (see
//! [`super::config`]).

use std::path::Path;

use shiso_core::{DeviceBuffer, DeviceView};
use shiso_vision::generator::ImageGenError;

use super::config::{DitConfig, DitNames, MlpNaming, SINUSOID_DIM};
use super::embed::{Embedders, TimeCond, sinusoid};
use super::rope::{angle_table, target_positions, text_positions};
use super::vae::LatentGrid;
use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{
    GemmCall, LayerNormCall, LinearResidualCall, ModulateCall, Ops, QkNormRopeCall, Scratch,
    SdpaCall, SliceExt, SwigluInput,
};
use crate::diffusion::weights::StStore;

/// Where the transformer's weights live and what shape they are.
pub(crate) struct DitDesc<'a> {
    /// The transformer safetensors file (or a directory holding one export).
    pub weights: &'a Path,
    /// The geometry from `transformer/config.json`.
    pub config: &'a DitConfig,
}

/// The block MLP's input projection, as the export stores it.
#[derive(Debug)]
enum MlpIn {
    /// `[gate | value]` rows from one matrix.
    Fused(Lin),
    /// Separate gate and value matrices.
    Split { gate: Lin, proj: Lin },
}

/// One transformer block's weights.
#[derive(Debug)]
struct Block {
    to_q: Lin,
    to_k: Lin,
    to_v: Lin,
    to_out: Lin,
    norm_q: DeviceBuffer,
    norm_k: DeviceBuffer,
    mlp_in: MlpIn,
    mlp_out: Lin,
}

/// Every layer's text keys and values after `QK-norm` and `RoPE`, for one
/// job's prompt, plus the target grid's rotary table once a step has run.
#[derive(Debug)]
pub(crate) struct PrefixKv {
    /// `(layers, tokens, hidden)` keys.
    k: DeviceBuffer,
    /// `(layers, tokens, hidden)` values.
    v: DeviceBuffer,
    tokens: usize,
    layers: usize,
    hidden: usize,
    /// The job's step inputs, uploaded once by [`QwenDit::prepare_steps`].
    steps: Option<StepTables>,
}

/// What every step of one job reads from the host, uploaded once before
/// the first: an upload drains the device queue (the host copy waits for
/// queued work), so doing it per step would hide a sync inside
/// [`QwenDit::velocity`].
#[derive(Debug)]
struct StepTables {
    /// The grid the rotary table was built for.
    grid: LatentGrid,
    /// The target's `[cos | sin]` table, `2 * (cells, head_dim / 2)`.
    rope: DeviceBuffer,
    /// The sigmas the sinusoid rows were built for, in row order.
    sigmas: Vec<f32>,
    /// One `SINUSOID_DIM` row per sigma.
    sinusoids: DeviceBuffer,
}

impl PrefixKv {
    /// Text tokens the prefix holds.
    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    fn layer<'a>(
        &self,
        buffer: &'a DeviceBuffer,
        layer: usize,
    ) -> Result<DeviceView<'a>, ImageGenError> {
        let span = (self.tokens * self.hidden * 4) as u64;
        buffer.view().cut(layer as u64 * span, span)
    }

    /// Layer `layer`'s keys, `(tokens, hidden)`.
    #[cfg(test)]
    pub(crate) fn keys(&self, layer: usize) -> Result<DeviceView<'_>, ImageGenError> {
        self.layer(&self.k, layer)
    }
}

/// How a block's attention reads and keeps keys and values.
#[derive(Clone, Copy)]
enum Attend<'a> {
    /// The text prefix: keys and values are written into these views and
    /// attended causally; `last` skips what only a next layer would read.
    Prefix {
        k: DeviceView<'a>,
        v: DeviceView<'a>,
        last: bool,
    },
    /// The target: keys and values are scratch, and the layer's cached
    /// prefix rows are attended first.
    Target {
        cache_k: DeviceView<'a>,
        cache_v: DeviceView<'a>,
        cache_tokens: usize,
    },
}

/// One block's inputs beyond its weights.
struct BlockRun<'a> {
    /// The stream, `(rows, hidden)`, updated in place.
    x: DeviceView<'a>,
    rows: usize,
    /// `[scale1 | gate1 | scale2 | gate2]`, gates through `tanh`.
    modulation: DeviceView<'a>,
    /// `(rows, head_dim / 2)` rotary cosines and sines.
    cos: DeviceView<'a>,
    sin: DeviceView<'a>,
    attend: Attend<'a>,
}

/// The loaded transformer.
#[derive(Debug)]
pub(crate) struct QwenDit {
    embed: Embedders,
    blocks: Vec<Block>,
    proj_out: Lin,
    /// The text tokens' modulation (`t = 0`), gates through `tanh`.
    mod_txt: DeviceBuffer,
    /// `hidden` zeros: the shift of a scale-only modulation.
    zeros: DeviceBuffer,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    mlp_hidden: usize,
    context: usize,
    in_channels: usize,
    out_channels: usize,
    axes: [usize; 3],
    eps: f32,
}

/// `f32` elements as bytes.
fn bytes(elements: usize) -> u64 {
    (elements * 4) as u64
}

impl QwenDit {
    /// Loads every block the config names.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] when the weights disagree with the
    /// config or a tensor is missing or of a dtype no GEMM reads.
    pub(crate) fn load(desc: &DitDesc<'_>, ops: &Ops) -> Result<Self, ImageGenError> {
        let all: Vec<usize> = (0..desc.config.num_layers).collect();
        Self::load_blocks(desc, ops, &all)
    }

    /// Loads the non-block weights and checkpoint blocks `which`, in that
    /// order, as the model's layers: the device tests run a one-layer model
    /// over block 0 or block 31 this way.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the config's eps (1e-6) is the f32 the norm kernels take"
    )]
    pub(crate) fn load_blocks(
        desc: &DitDesc<'_>,
        ops: &Ops,
        which: &[usize],
    ) -> Result<Self, ImageGenError> {
        let config = desc.config;
        if which.is_empty() || which.iter().any(|&i| i >= config.num_layers) {
            return Err(ImageGenError::Checkpoint(format!(
                "blocks {which:?} are not a non-empty selection of the {} layers",
                config.num_layers
            )));
        }
        let store = StStore::open(desc.weights)?;
        let names = DitNames::detect(&store)?;
        names.check(&store, config)?;
        let embed = Embedders::load(&store, ops, &names, config)?;
        let mut blocks = Vec::with_capacity(which.len());
        for &i in which {
            let lin = |part: &str| Lin::load(&store, ops, &names.block(i, part));
            let norm = |part: &str| -> Result<DeviceBuffer, ImageGenError> {
                let values = store.f32_vec(&format!("{}.weight", names.block(i, part)))?;
                ops.upload_f32("qwen qk norm", &values)
            };
            let mlp_in = match names.mlp {
                MlpNaming::Fused => MlpIn::Fused(lin("img_mlp.gate_up")?),
                MlpNaming::Split => MlpIn::Split {
                    gate: lin("img_mlp.gate_layer")?,
                    proj: lin("img_mlp.proj")?,
                },
            };
            blocks.push(Block {
                to_q: lin("attn.to_q")?,
                to_k: lin("attn.to_k")?,
                to_v: lin("attn.to_v")?,
                to_out: lin("attn.to_out.0")?,
                norm_q: norm("attn.norm_q")?,
                norm_k: norm("attn.norm_k")?,
                mlp_in,
                mlp_out: lin("img_mlp.out")?,
            });
        }
        let hidden = config.hidden();
        let mod_txt = embed.modulation_at(ops, 0.0)?;
        let zeros = ops.upload_f32("qwen zero shift", &vec![0.0; hidden])?;
        Ok(Self {
            proj_out: Lin::load(&store, ops, &names.top("proj_out"))?,
            embed,
            blocks,
            mod_txt,
            zeros,
            hidden,
            heads: config.num_attention_heads,
            head_dim: config.attention_head_dim,
            mlp_hidden: config.mlp_hidden(),
            context: config.context_in_dim,
            in_channels: config.in_channels,
            out_channels: config.out_channels,
            axes: config.rope_axes()?,
            eps: config.eps as f32,
        })
    }

    /// The non-block embedders, for the device tests.
    #[cfg(test)]
    pub(crate) fn embedders(&self) -> &Embedders {
        &self.embed
    }

    /// Runs the prompt's `p` text-feature rows `(p, context)` through
    /// `txt_in` and every block, keeping each layer's keys and values.
    ///
    /// # Errors
    /// [`ImageGenError::Backend`] for an empty prompt, a feature buffer
    /// shorter than `p` rows, or a failed launch.
    pub(crate) fn prefix(
        &self,
        ops: &Ops,
        text_feats: DeviceView<'_>,
        p: usize,
    ) -> Result<PrefixKv, ImageGenError> {
        self.prefix_run(ops, text_feats, p, false).map(|(kv, _)| kv)
    }

    /// [`QwenDit::prefix`] that also runs the last block's attention output
    /// and MLP and hands back the text stream, for the device tests to read
    /// what the production path skips.
    #[cfg(test)]
    pub(crate) fn prefix_with_stream<'o>(
        &self,
        ops: &'o Ops,
        text_feats: DeviceView<'_>,
        p: usize,
    ) -> Result<(PrefixKv, Scratch<'o>), ImageGenError> {
        self.prefix_run(ops, text_feats, p, true)
    }

    fn prefix_run<'o>(
        &self,
        ops: &'o Ops,
        text_feats: DeviceView<'_>,
        p: usize,
        whole_last: bool,
    ) -> Result<(PrefixKv, Scratch<'o>), ImageGenError> {
        if p == 0 || text_feats.len() < bytes(p * self.context) {
            return Err(ImageGenError::Backend(format!(
                "prefix: {p} text rows of {} features need {} bytes, the buffer holds {}",
                self.context,
                bytes(p * self.context),
                text_feats.len()
            )));
        }
        let h = self.hidden;
        let layers = self.blocks.len();
        let kv = PrefixKv {
            k: ops.alloc_owned_f32("qwen prefix k", layers * p * h)?,
            v: ops.alloc_owned_f32("qwen prefix v", layers * p * h)?,
            tokens: p,
            layers,
            hidden: h,
            steps: None,
        };
        let x = self.embed.txt_in(ops, text_feats, p)?;
        let (cos, sin) = angle_table(&text_positions(p), self.axes);
        let table = ops.upload_scratch("qwen prefix rope", &[cos, sin].concat())?;
        let half = p * self.head_dim / 2;
        for (l, block) in self.blocks.iter().enumerate() {
            self.block(
                ops,
                block,
                &BlockRun {
                    x: x.view(),
                    rows: p,
                    modulation: self.mod_txt.view(),
                    cos: table.view().cut(0, bytes(half))?,
                    sin: table.view().cut(bytes(half), bytes(half))?,
                    attend: Attend::Prefix {
                        k: kv.layer(&kv.k, l)?,
                        v: kv.layer(&kv.v, l)?,
                        last: !whole_last && l + 1 == layers,
                    },
                },
            )?;
        }
        Ok((kv, x))
    }

    /// Uploads what the job's steps read from the host - the target grid's
    /// rotary table and the timestep sinusoid of every sigma - once, so
    /// [`QwenDit::velocity`] issues no host transfer and no hidden sync.
    /// The two uploads each drain the queue, once per job.
    ///
    /// # Errors
    /// [`ImageGenError::Backend`] for an empty ladder or a failed upload.
    pub(crate) fn prepare_steps(
        &self,
        ops: &Ops,
        kv: &mut PrefixKv,
        sigmas: &[f32],
        grid: LatentGrid,
    ) -> Result<(), ImageGenError> {
        if sigmas.is_empty() || grid.cells() == 0 {
            return Err(ImageGenError::Backend(
                "prepare_steps: no sigmas or an empty grid".to_string(),
            ));
        }
        let (cos, sin) = angle_table(&target_positions(kv.tokens, grid), self.axes);
        let rope = ops.upload_f32("qwen target rope", &[cos, sin].concat())?;
        let rows: Vec<f32> = sigmas.iter().flat_map(|&s| sinusoid(s)).collect();
        let sinusoids = ops.upload_f32("qwen sinusoids", &rows)?;
        kv.steps = Some(StepTables {
            grid,
            rope,
            sigmas: sigmas.to_vec(),
            sinusoids,
        });
        Ok(())
    }

    /// The velocity of the target latents `(cells, in_channels)` at
    /// `sigma`, into `out` `(cells, out_channels)`, attending over `kv`.
    /// `sigma` and `grid` must be ones [`QwenDit::prepare_steps`] uploaded
    /// for `kv`; nothing is uploaded here.
    ///
    /// # Errors
    /// [`ImageGenError::Backend`] for a sigma outside [0, 1], a prefix from
    /// another model, a buffer too short for the grid, or a failed launch.
    pub(crate) fn velocity(
        &self,
        ops: &Ops,
        kv: &PrefixKv,
        latents: DeviceView<'_>,
        grid: LatentGrid,
        sigma: f32,
        out: DeviceView<'_>,
    ) -> Result<(), ImageGenError> {
        let t = grid.cells();
        if out.len() < bytes(t * self.out_channels) {
            return Err(ImageGenError::Backend(format!(
                "velocity: {t} tokens of {} channels do not fit a {}-byte output",
                self.out_channels,
                out.len()
            )));
        }
        let (x, cond) = self.target_stream(ops, kv, latents, grid, sigma)?;
        let normed = ops.alloc_f32("qwen head norm", t * self.hidden)?;
        self.norm_scale(ops, x.view(), normed.view(), cond.final_scale.view(), t)?;
        ops.gemm(&GemmCall {
            step: "qwen proj out",
            out,
            input: normed.view(),
            weight: self.proj_out.weight(),
            m: t,
            n: self.out_channels,
            k: self.hidden,
        })
    }

    /// The target stream after every block, with the step's conditioning:
    /// [`QwenDit::velocity`] up to the head.
    pub(crate) fn target_stream<'o>(
        &self,
        ops: &'o Ops,
        kv: &PrefixKv,
        latents: DeviceView<'_>,
        grid: LatentGrid,
        sigma: f32,
    ) -> Result<(Scratch<'o>, TimeCond<'o>), ImageGenError> {
        let t = grid.cells();
        let h = self.hidden;
        if !(0.0..=1.0).contains(&sigma) {
            return Err(ImageGenError::Backend(format!(
                "velocity: sigma {sigma} is outside [0, 1]"
            )));
        }
        if kv.layers != self.blocks.len() || kv.hidden != h {
            return Err(ImageGenError::Backend(format!(
                "velocity: the prefix holds {} layers of width {}, this model {} of {h}",
                kv.layers,
                kv.hidden,
                self.blocks.len()
            )));
        }
        if t == 0 || latents.len() < bytes(t * self.in_channels) {
            return Err(ImageGenError::Backend(format!(
                "velocity: a {}x{} grid of {} channels needs {} bytes of latents, got {}",
                grid.h,
                grid.w,
                self.in_channels,
                bytes(t * self.in_channels),
                latents.len()
            )));
        }
        let steps = kv
            .steps
            .as_ref()
            .filter(|s| s.grid == grid)
            .ok_or_else(|| {
                ImageGenError::Backend(format!(
                    "velocity: no step tables for a {}x{} grid; prepare_steps first",
                    grid.h, grid.w
                ))
            })?;
        let row = steps
            .sigmas
            .iter()
            .position(|s| s.to_bits() == sigma.to_bits())
            .ok_or_else(|| {
                ImageGenError::Backend(format!(
                    "velocity: sigma {sigma} is not one prepare_steps uploaded"
                ))
            })?;
        let table = steps.rope.view();
        let features = steps
            .sinusoids
            .view()
            .cut(bytes(row * SINUSOID_DIM), bytes(SINUSOID_DIM))?;
        let half = t * self.head_dim / 2;
        let cond = self.embed.time_cond_of(ops, features)?;
        self.embed.tanh_gates(ops, cond.modulation.view())?;
        let x = ops.alloc_f32("qwen target stream", t * h)?;
        self.embed.img_in(ops, latents, t, x.view())?;
        for (l, block) in self.blocks.iter().enumerate() {
            self.block(
                ops,
                block,
                &BlockRun {
                    x: x.view(),
                    rows: t,
                    modulation: cond.modulation.view(),
                    cos: table.cut(0, bytes(half))?,
                    sin: table.cut(bytes(half), bytes(half))?,
                    attend: Attend::Target {
                        cache_k: kv.layer(&kv.k, l)?,
                        cache_v: kv.layer(&kv.v, l)?,
                        cache_tokens: kv.tokens,
                    },
                },
            )?;
        }
        Ok((x, cond))
    }

    /// `out = LN(x) * (1 + scale)` over `rows` rows.
    fn norm_scale(
        &self,
        ops: &Ops,
        x: DeviceView<'_>,
        out: DeviceView<'_>,
        scale: DeviceView<'_>,
        rows: usize,
    ) -> Result<(), ImageGenError> {
        let h = self.hidden;
        ops.layer_norm(&LayerNormCall {
            step: "qwen norm",
            out,
            input: x,
            rows,
            dim: h,
            eps: self.eps,
        })?;
        ops.modulate(&ModulateCall {
            step: "qwen norm scale",
            out,
            x: out,
            scale,
            shift: self.zeros.view(),
            count: rows * h,
            period: h,
        })
    }

    /// `out = lin(input)` over `rows` rows.
    fn linear(
        ops: &Ops,
        step: &'static str,
        lin: &Lin,
        input: DeviceView<'_>,
        out: DeviceView<'_>,
        rows: usize,
    ) -> Result<(), ImageGenError> {
        ops.gemm(&GemmCall {
            step,
            out,
            input,
            weight: lin.weight(),
            m: rows,
            n: lin.out_dim,
            k: lin.in_dim,
        })
    }

    /// One block over `run.rows` rows of the stream (see the module docs).
    #[expect(
        clippy::too_many_lines,
        reason = "one block's attention and MLP branches, straight-line by construction"
    )]
    fn block(&self, ops: &Ops, w: &Block, run: &BlockRun<'_>) -> Result<(), ImageGenError> {
        let (h, m) = (self.hidden, run.rows);
        let quarter = |i: u64| run.modulation.cut(i * bytes(h), bytes(h));
        let normed = ops.alloc_f32("qwen block norm", m * h)?;
        self.norm_scale(ops, run.x, normed.view(), quarter(0)?, m)?;

        let queries = ops.alloc_f32("qwen q", m * h)?;
        Self::linear(ops, "qwen to_q", &w.to_q, normed.view(), queries.view(), m)?;
        let target_kv;
        let (keys, values) = match run.attend {
            Attend::Prefix { k, v, .. } => (k, v),
            Attend::Target { .. } => {
                target_kv = (
                    ops.alloc_f32("qwen k", m * h)?,
                    ops.alloc_f32("qwen v", m * h)?,
                );
                (target_kv.0.view(), target_kv.1.view())
            }
        };
        Self::linear(ops, "qwen to_k", &w.to_k, normed.view(), keys, m)?;
        Self::linear(ops, "qwen to_v", &w.to_v, normed.view(), values, m)?;
        ops.qk_norm_rope(&QkNormRopeCall {
            step: "qwen qk norm rope",
            q: queries.view(),
            k: keys,
            q_scale: w.norm_q.view(),
            k_scale: w.norm_k.view(),
            cos: run.cos,
            sin: run.sin,
            tokens: m,
            heads: self.heads,
            head_dim: self.head_dim,
            q_stride: h,
            k_stride: h,
            eps: self.eps,
        })?;

        let attn = ops.alloc_f32("qwen attention", m * h)?;
        let dummy = ops.dummy_view();
        let call = match run.attend {
            // The text rows alone, causally: the `txt` slot of the call.
            Attend::Prefix { .. } => SdpaCall {
                step: "qwen prefix attention",
                out_txt: attn.view(),
                out_img: dummy,
                q_txt: queries.view(),
                k_txt: keys,
                v_txt: values,
                q_img: dummy,
                k_img: dummy,
                v_img: dummy,
                q_offset: 0,
                k_offset: 0,
                v_offset: 0,
                cache_k: dummy,
                cache_v: dummy,
                cache_stride: h,
                txt_tokens: m,
                img_tokens: 0,
                cache_tokens: 0,
                heads: self.heads,
                head_dim: self.head_dim,
                txt_stride: h,
                img_stride: h,
                causal: true,
            },
            // The target rows as queries; their keys plus the cached
            // prefix rows (softmax is order-free, so the prefix sitting
            // after the live rows in the kernel's key order is the same
            // attention).
            Attend::Target {
                cache_k,
                cache_v,
                cache_tokens,
            } => SdpaCall {
                step: "qwen target attention",
                out_txt: dummy,
                out_img: attn.view(),
                q_txt: dummy,
                k_txt: dummy,
                v_txt: dummy,
                q_img: queries.view(),
                k_img: keys,
                v_img: values,
                q_offset: 0,
                k_offset: 0,
                v_offset: 0,
                cache_k,
                cache_v,
                cache_stride: h,
                txt_tokens: 0,
                img_tokens: m,
                cache_tokens,
                heads: self.heads,
                head_dim: self.head_dim,
                txt_stride: h,
                img_stride: h,
                causal: false,
            },
        };
        ops.sdpa(&call)?;
        if let Attend::Prefix { last: true, .. } = run.attend {
            // The last layer's text stream feeds nothing: its keys and
            // values are all the target reads.
            return Ok(());
        }

        ops.linear_residual(&LinearResidualCall {
            step: "qwen to_out",
            residual: run.x,
            gate: quarter(1)?,
            dense: Some(attn.view()),
            dense_cols: h,
            swiglu: None,
            weight: w.to_out.weight(),
            m,
            n: h,
        })?;
        drop(attn);
        drop(queries);

        self.norm_scale(ops, run.x, normed.view(), quarter(2)?, m)?;
        let mlp = self.mlp_hidden;
        match &w.mlp_in {
            MlpIn::Fused(gate_up) => {
                let gu = ops.alloc_f32("qwen gate_up", m * 2 * mlp)?;
                Self::linear(ops, "qwen gate_up", gate_up, normed.view(), gu.view(), m)?;
                ops.linear_residual(&LinearResidualCall {
                    step: "qwen mlp out",
                    residual: run.x,
                    gate: quarter(3)?,
                    dense: None,
                    dense_cols: 0,
                    swiglu: Some(SwigluInput {
                        rows: gu.view(),
                        stride: 2 * mlp,
                        offset: 0,
                        inner: mlp,
                    }),
                    weight: w.mlp_out.weight(),
                    m,
                    n: h,
                })
            }
            MlpIn::Split { gate, proj } => {
                let gated = ops.alloc_f32("qwen gate", m * mlp)?;
                let value = ops.alloc_f32("qwen proj", m * mlp)?;
                Self::linear(ops, "qwen gate", gate, normed.view(), gated.view(), m)?;
                Self::linear(ops, "qwen proj", proj, normed.view(), value.view(), m)?;
                ops.swiglu2(
                    "qwen swiglu",
                    gated.view(),
                    gated.view(),
                    value.view(),
                    m * mlp,
                )?;
                ops.linear_residual(&LinearResidualCall {
                    step: "qwen mlp out",
                    residual: run.x,
                    gate: quarter(3)?,
                    dense: Some(gated.view()),
                    dense_cols: mlp,
                    swiglu: None,
                    weight: w.mlp_out.weight(),
                    m,
                    n: h,
                })
            }
        }
    }
}

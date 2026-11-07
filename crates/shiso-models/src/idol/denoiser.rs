//! The denoiser: a rectified-flow transformer over the object latent, the
//! picture's conditioning and the camera.
//!
//! The latent (8192 tokens of 16) and the conditioning (4101 rows, the two
//! streams projected and summed) are each refined by two blocks of their own;
//! the camera (one token of 5) by a small MLP. Then all 12294 rows run through
//! 24 joint blocks, and the latent and camera rows are read out through a
//! final time-modulated norm. Every block attends with learned rotary
//! positions: a per-block MLP reads each token and predicts a 3D offset per
//! head, which three learned frequency tables turn into rotation angles.
//!
//! The time embedding, the shared modulation and the camera's two ends are a
//! single row each and run on the host; everything over token rows runs on
//! the device. The rotary angles are predicted on the device and turned into
//! cosine and sine tables on the host, because the driver has no cosine.
//!
//! The reference runs the blocks in fp16; this runs f32 throughout (see the
//! module doc of [`super::nn`]).

use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use shiso_core::DeviceBuffer;
use shiso_vision::generator::ImageGenError;

use super::conditioning::Conditioning;
use super::config::{DenoiserArgs, IdolConfig};
use super::generator::Prepared;
use super::nn::{
    Affine, AttnShape, HeadNorm, Pairing, Qkv, add_gated_in_place, attention, bytes, host_lin,
    layer_norm, linear, point_embed_v1, read_linear, rotate_in_place, turned,
};
use super::splat_ops::{RotaryCall, rotary_tables};
use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{ModulateCall, Ops, SliceExt};
use crate::diffusion::weights::StStore;

/// The block norms' epsilon.
const BLOCK_EPS: f32 = 1e-6;
/// The rotary MLP's norm epsilon and the read-out norm's (`LayerNorm` and
/// `F.layer_norm` defaults).
const DEFAULT_EPS: f32 = 1e-5;
/// The time embedding's sinusoid count.
const FREQ_EMBED: usize = 256;
/// The time embedding's longest period.
const MAX_PERIOD: f64 = 10_000.0;

/// A linear kept on the host, for single-row work.
#[derive(Debug)]
pub(crate) struct HostLinear {
    w: Vec<f32>,
    b: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl HostLinear {
    pub(crate) fn load(store: &StStore, name: &str) -> Result<Self, ImageGenError> {
        let (w, b) = read_linear(store, name)?;
        let out_dim = b.as_ref().map_or(0, Vec::len);
        let b = b.ok_or_else(|| ImageGenError::Checkpoint(format!("{name} has no bias")))?;
        let in_dim = w.len().checked_div(out_dim).unwrap_or(0);
        if in_dim == 0 || in_dim * out_dim != w.len() {
            return Err(ImageGenError::Checkpoint(format!(
                "{name}: a {} weight for a {out_dim}-wide bias",
                w.len()
            )));
        }
        Ok(Self {
            w,
            b,
            out_dim,
            in_dim,
        })
    }

    pub(crate) fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        (0..self.out_dim)
            .map(|r| {
                let row = &self.w[r * self.in_dim..(r + 1) * self.in_dim];
                self.b[r] + row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>()
            })
            .collect()
    }
}

/// Where a pass's time went, kept only while the `shiso::idol` debug log is
/// on.
///
/// Launches are queued, so a stage's wall time means nothing unless the
/// queue is drained on both sides of it. The drains cost a pass several
/// stalls, which is why they happen only when somebody reads the numbers.
#[derive(Debug, Default)]
struct PassTimes {
    enabled: bool,
    attention: Duration,
    rotary: Duration,
}

impl PassTimes {
    fn new() -> Self {
        Self {
            enabled: tracing::enabled!(target: "shiso::idol", tracing::Level::DEBUG),
            ..Self::default()
        }
    }

    /// Runs `stage`, adding its drained wall time to `slot` when enabled.
    fn time<T>(
        &mut self,
        ops: &Ops,
        slot: fn(&mut Self) -> &mut Duration,
        stage: impl FnOnce() -> Result<T, ImageGenError>,
    ) -> Result<T, ImageGenError> {
        if !self.enabled {
            return stage();
        }
        drain(ops)?;
        let start = Instant::now();
        let out = stage()?;
        drain(ops)?;
        *slot(self) += start.elapsed();
        Ok(out)
    }
}

fn drain(ops: &Ops) -> Result<(), ImageGenError> {
    ops.backend()
        .sync()
        .map_err(|e| ImageGenError::Backend(format!("denoiser timing drain: {e}")))
}

/// `x * sigmoid(x)`.
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The tanh-approximated GELU.
pub(crate) fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0f32 / PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

/// The sinusoidal embedding of one scalar: `[cos(t f), sin(t f)]` over 128
/// frequencies `exp(-ln(max_period) i / 128)`, with `t` scaled by `2 pi`
/// first when `two_pi`.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the frequencies are built in f64 and stored as the f32 the reference computes"
)]
pub(crate) fn sinusoids(t: f32, max_period: f64, two_pi: bool) -> Vec<f32> {
    let half = FREQ_EMBED / 2;
    let mut out = vec![0.0f32; FREQ_EMBED];
    for i in 0..half {
        let freq = ((-(max_period.ln()) as f32 * i as f32) / half as f32).exp();
        let mut arg = t * freq;
        if two_pi {
            arg = arg * 2.0 * PI;
        }
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

/// One learned rotary position predictor.
#[derive(Debug)]
struct Repo {
    norm: Affine,
    gate: Lin,
    content: Lin,
    project: Lin,
    /// How many pairs a head turns (32: ten for x, ten for y, twelve for z).
    pairs: usize,
    /// Each pair's frequency times pi, rounded once to f32, on the device.
    rates: DeviceBuffer,
    /// Which of the three predicted offsets each pair turns by, as `u32`.
    axis: DeviceBuffer,
}

/// How a block's norms are scaled.
#[derive(Debug)]
enum Norms {
    /// The time modulation plus this block's own `(6 * width)` shift table.
    Modulated(Vec<f32>),
    /// Plain affine norms (the conditioning's refiner).
    Affine(Box<(Affine, Affine)>),
}

/// One transformer block.
#[derive(Debug)]
struct Block {
    norms: Norms,
    q: Lin,
    q_turn: Lin,
    k: Lin,
    k_turn: Lin,
    v: Lin,
    q_norm: HeadNorm,
    k_norm: HeadNorm,
    out: Lin,
    up: Lin,
    down: Lin,
}

/// The loaded denoiser.
#[derive(Debug)]
pub(crate) struct IdolDenoiser {
    ops: Arc<Ops>,
    args: DenoiserArgs,
    head_dim: usize,
    t_embed: [HostLinear; 2],
    modulation: HostLinear,
    final_shift: Vec<f32>,
    input: Lin,
    cond1: Lin,
    cond2: Lin,
    position: DeviceBuffer,
    noise_repo: Vec<Repo>,
    context_repo: Vec<Repo>,
    repo: Vec<Repo>,
    noise_refiner: Vec<Block>,
    context_refiner: Vec<Block>,
    blocks: Vec<Block>,
    cam: [HostLinear; 2],
    out: Lin,
    cam_out: HostLinear,
}

/// The denoiser's answer: a velocity for the latent and one for the camera.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Velocity {
    /// `(q_token_length, in_channels)`.
    pub(crate) latent: Vec<f32>,
    /// `cam_channels` values.
    pub(crate) camera: Vec<f32>,
}

fn load_repo(store: &StStore, ops: &Ops, base: &str) -> Result<Repo, ImageGenError> {
    let mut rates = Vec::new();
    let mut axis = Vec::new();
    for a in 0u32..3 {
        let f = store.f32_vec(&format!("{base}freqs_{a}"))?;
        axis.extend(std::iter::repeat_n(a, f.len()));
        rates.extend(f.into_iter().map(|f| f * PI));
    }
    Ok(Repo {
        pairs: rates.len(),
        rates: ops.upload_f32("denoiser repo rates", &rates)?,
        axis: ops.upload_u32("denoiser repo axis", &axis)?,
        norm: Affine::load(store, ops, &format!("{base}norm"))?,
        gate: Lin::load(store, ops, &format!("{base}gate_map"))?,
        content: Lin::load(store, ops, &format!("{base}content_map"))?,
        project: Lin::load(store, ops, &format!("{base}final_map"))?,
    })
}

fn load_block(
    store: &StStore,
    ops: &Ops,
    base: &str,
    args: &DenoiserArgs,
    modulated: bool,
) -> Result<Block, ImageGenError> {
    let width = args.model_channels;
    let heads = args.num_heads;
    let head_dim = width / heads;
    let (w, b) = read_linear(store, &format!("{base}attn.qkv"))?;
    let b = b.ok_or_else(|| ImageGenError::Checkpoint(format!("{base}attn.qkv has no bias")))?;
    if w.len() != 3 * width * width || b.len() != 3 * width {
        return Err(ImageGenError::Checkpoint(format!(
            "{base}attn.qkv is not 3 x {width} rows of {width}"
        )));
    }
    let part = |i: usize| {
        (
            &w[i * width * width..(i + 1) * width * width],
            &b[i * width..(i + 1) * width],
        )
    };
    let (qw, qb) = part(0);
    let (kw, kb) = part(1);
    let (vw, vb) = part(2);
    let (qtw, qtb) = turned(qw, Some(qb), heads, head_dim, Pairing::Adjacent);
    let (ktw, ktb) = turned(kw, Some(kb), heads, head_dim, Pairing::Adjacent);
    let norms = if modulated {
        Norms::Modulated(store.f32_vec(&format!("{base}shift_table"))?)
    } else {
        Norms::Affine(Box::new((
            Affine::load(store, ops, &format!("{base}norm1"))?,
            Affine::load(store, ops, &format!("{base}norm2"))?,
        )))
    };
    Ok(Block {
        norms,
        q: host_lin(ops, qw, Some(qb), width, width)?,
        q_turn: host_lin(ops, &qtw, qtb.as_deref(), width, width)?,
        k: host_lin(ops, kw, Some(kb), width, width)?,
        k_turn: host_lin(ops, &ktw, ktb.as_deref(), width, width)?,
        v: host_lin(ops, vw, Some(vb), width, width)?,
        q_norm: HeadNorm::load(store, ops, &format!("{base}attn.q_norm"), heads, head_dim)?,
        k_norm: HeadNorm::load(store, ops, &format!("{base}attn.k_norm"), heads, head_dim)?,
        out: Lin::load(store, ops, &format!("{base}attn.out"))?,
        up: Lin::load(store, ops, &format!("{base}mlp.mlp.0"))?,
        down: Lin::load(store, ops, &format!("{base}mlp.mlp.2"))?,
    })
}

impl IdolDenoiser {
    /// Opens `dir/denoiser.safetensors` under `dir/config.json`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a document or tensor of another shape.
    pub(crate) fn load(dir: &Path, ops: Arc<Ops>) -> Result<Self, ImageGenError> {
        let args = IdolConfig::read(dir)?.denoiser;
        let store = StStore::open_files(&[dir.join("denoiser.safetensors")])?;
        let width = args.model_channels;
        let load_all = |prefix: &str, count: usize, modulated: bool| {
            (0..count)
                .map(|i| load_block(&store, &ops, &format!("{prefix}.{i}."), &args, modulated))
                .collect::<Result<Vec<_>, ImageGenError>>()
        };
        let repos = |prefix: &str, count: usize| {
            (0..count)
                .map(|i| load_repo(&store, &ops, &format!("{prefix}.{i}.")))
                .collect::<Result<Vec<_>, ImageGenError>>()
        };
        let pos = store.f32_vec("pos_pe")?;
        if pos.len() != args.q_token_length * 3 {
            return Err(ImageGenError::Checkpoint(format!(
                "pos_pe is {} values, expected {} x 3",
                pos.len(),
                args.q_token_length
            )));
        }
        let position = ops.upload_f32("denoiser positions", &point_embed_v1(&pos, width))?;
        Ok(Self {
            head_dim: width / args.num_heads,
            t_embed: [
                HostLinear::load(&store, "t_embedder.mlp.0")?,
                HostLinear::load(&store, "t_embedder.mlp.2")?,
            ],
            modulation: HostLinear::load(&store, "adaLN_modulation.1")?,
            final_shift: store.f32_vec("shift_table")?,
            input: Lin::load(&store, &ops, "input_layer")?,
            cond1: Lin::load(&store, &ops, "cond_embedder")?,
            cond2: Lin::load(&store, &ops, "cond_embedder2")?,
            position,
            noise_repo: repos("noise_repo_layers", args.num_refiner_blocks)?,
            context_repo: repos("context_repo_layers", args.num_refiner_blocks)?,
            repo: repos("repo_layers", args.num_blocks)?,
            noise_refiner: load_all("noise_refiner", args.num_refiner_blocks, true)?,
            context_refiner: load_all("context_refiner", args.num_refiner_blocks, false)?,
            blocks: load_all("blocks", args.num_blocks, true)?,
            cam: [
                HostLinear::load(&store, "cam_refiner.mlp.0")?,
                HostLinear::load(&store, "cam_refiner.mlp.2")?,
            ],
            out: Lin::load(&store, &ops, "out_layer")?,
            cam_out: HostLinear::load(&store, "cam_out_layer")?,
            args,
            ops,
        })
    }

    /// The conditioning stream for `cond`: both projections summed and run
    /// through the context refiner, read back to the host.
    ///
    /// The refiner sees neither the latent nor the timestep, so one request
    /// computes this once per conditioning and hands it to every
    /// [`IdolDenoiser::forward`].
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Backend`] for streams of the wrong width or a device
    /// failure.
    pub(crate) fn context(&self, cond: &Conditioning) -> Result<Prepared, ImageGenError> {
        let ops = &*self.ops;
        let a = &self.args;
        let (width, n_c) = (a.model_channels, cond.rows);
        if cond.dinov3.len() != n_c * a.cond_channels || cond.vae.len() != n_c * a.cond2_channels {
            return Err(ImageGenError::Backend(format!(
                "the denoiser reads {n_c} conditioning rows of {} and {} values, got {} and {}",
                a.cond_channels,
                a.cond2_channels,
                cond.dinov3.len(),
                cond.vae.len()
            )));
        }
        let mut times = PassTimes::new();
        let stream = ops.alloc_f32("denoiser cond", n_c * width)?;
        let f1 = ops.upload_scratch("denoiser cond", &cond.dinov3)?;
        let f2 = ops.upload_scratch("denoiser cond", &cond.vae)?;
        let c1 = linear(ops, "denoiser cond", &self.cond1, f1.view(), n_c)?;
        let c2 = linear(ops, "denoiser cond", &self.cond2, f2.view(), n_c)?;
        ops.add(
            "denoiser cond",
            stream.view(),
            c1.view(),
            c2.view(),
            n_c * width,
        )?;
        drop((f1, f2, c1, c2));
        for (repo, block) in self.context_repo.iter().zip(&self.context_refiner) {
            self.block(block, repo, stream.view(), n_c, None, &mut times)?;
        }
        Ok(Prepared {
            rows: n_c,
            stream: ops.download("denoiser cond readback", stream.view(), n_c * width)?,
        })
    }

    /// The velocity at `(latent, camera)` for the model timestep `timestep`
    /// (the sampler's `t` times 1000) under the conditioning stream
    /// `prepared` (from [`IdolDenoiser::context`]).
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Backend`] for inputs of the wrong shape or a device
    /// failure.
    #[expect(
        clippy::too_many_lines,
        reason = "the denoiser's stage graph, straight-line by construction"
    )]
    pub(crate) fn forward(
        &self,
        latent: &[f32],
        camera: &[f32],
        timestep: f32,
        prepared: &Prepared,
    ) -> Result<Velocity, ImageGenError> {
        let ops = &*self.ops;
        let started = Instant::now();
        let mut times = PassTimes::new();
        let a = &self.args;
        let width = a.model_channels;
        let (n_x, n_c) = (a.q_token_length, prepared.rows);
        let rows = n_x + n_c + 1;
        if prepared.stream.len() != n_c * width {
            return Err(ImageGenError::Backend(format!(
                "the denoiser reads {n_c} conditioning rows of {width}, got {} values",
                prepared.stream.len()
            )));
        }
        if latent.len() != n_x * a.in_channels || camera.len() != a.cam_channels {
            return Err(ImageGenError::Backend(format!(
                "the denoiser reads a {}x{} latent and a {}-value camera, got {} and {}",
                n_x,
                a.in_channels,
                a.cam_channels,
                latent.len(),
                camera.len()
            )));
        }
        let t_emb = self.t_embed[1].apply(
            &self.t_embed[0]
                .apply(&sinusoids(timestep, MAX_PERIOD, false))
                .into_iter()
                .map(silu)
                .collect::<Vec<_>>(),
        );
        let t_mod = self
            .modulation
            .apply(&t_emb.iter().copied().map(silu).collect::<Vec<_>>());

        let h = ops.alloc_f32("denoiser stream", rows * width)?;
        let stream =
            |first: usize, count: usize| h.view().cut(bytes(first * width), bytes(count * width));

        // The latent stream.
        let z = ops.upload_scratch("denoiser latent", latent)?;
        let h_x = linear(ops, "denoiser input", &self.input, z.view(), n_x)?;
        ops.add(
            "denoiser positions",
            stream(0, n_x)?,
            h_x.view(),
            self.position.view(),
            n_x * width,
        )?;
        drop((z, h_x));
        for (repo, block) in self.noise_repo.iter().zip(&self.noise_refiner) {
            self.block(block, repo, stream(0, n_x)?, n_x, Some(&t_mod), &mut times)?;
        }

        // The conditioning stream, refined once per request: copied in, as
        // the joint blocks below write over these rows.
        let cond_bytes: Vec<u8> = prepared
            .stream
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        ops.backend()
            .copy_h2d(stream(n_x, n_c)?, &cond_bytes)
            .map_err(|e| ImageGenError::Backend(format!("denoiser cond rows: {e}")))?;
        drop(cond_bytes);

        // The camera token.
        let cam_hidden: Vec<f32> = self.cam[0]
            .apply(camera)
            .into_iter()
            .map(gelu_tanh)
            .collect();
        let cam_row = self.cam[1].apply(&cam_hidden);
        let cam_bytes: Vec<u8> = cam_row.iter().flat_map(|v| v.to_le_bytes()).collect();
        ops.backend()
            .copy_h2d(stream(n_x + n_c, 1)?, &cam_bytes)
            .map_err(|e| ImageGenError::Backend(format!("denoiser camera row: {e}")))?;

        for (repo, block) in self.repo.iter().zip(&self.blocks) {
            self.block(block, repo, h.view(), rows, Some(&t_mod), &mut times)?;
        }

        // Read-out: a parameter-free norm, the final time modulation, the heads.
        let mut shift = self.final_shift[..width].to_vec();
        let mut scale = self.final_shift[width..2 * width].to_vec();
        for i in 0..width {
            shift[i] += t_emb[i];
            scale[i] += t_emb[i];
        }
        let normed = layer_norm(
            ops,
            "denoiser out norm",
            stream(0, n_x)?,
            n_x,
            width,
            DEFAULT_EPS,
            None,
        )?;
        let scale_buf = ops.upload_scratch("denoiser out", &scale)?;
        let shift_buf = ops.upload_scratch("denoiser out", &shift)?;
        ops.modulate(&ModulateCall {
            step: "denoiser out",
            out: normed.view(),
            x: normed.view(),
            scale: scale_buf.view(),
            shift: shift_buf.view(),
            count: n_x * width,
            period: width,
        })?;
        let v = linear(ops, "denoiser out", &self.out, normed.view(), n_x)?;
        let latent_v = ops.download("denoiser readback", v.view(), n_x * a.in_channels)?;
        let cam_h = ops.download("denoiser readback", stream(n_x + n_c, 1)?, width)?;
        let cam_v = self
            .cam_out
            .apply(&host_modulated_norm(&cam_h, &scale, &shift));
        if latent_v.iter().chain(&cam_v).any(|v| !v.is_finite()) {
            return Err(ImageGenError::Backend(
                "the denoiser produced a non-finite velocity".to_owned(),
            ));
        }
        tracing::debug!(
            target: "shiso::idol",
            pass_ms = started.elapsed().as_millis(),
            attention_ms = times.attention.as_millis(),
            rotary_ms = times.rotary.as_millis(),
            "denoiser pass"
        );
        Ok(Velocity {
            latent: latent_v,
            camera: cam_v,
        })
    }

    /// One block over `rows` rows of `x`, in place, with its rotary positions
    /// predicted by `repo` from the block's input.
    #[expect(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "one transformer block, straight-line; q, k, v, o and x are the attention's own names"
    )]
    fn block(
        &self,
        block: &Block,
        repo: &Repo,
        x: shiso_core::DeviceView<'_>,
        rows: usize,
        t_mod: Option<&[f32]>,
        times: &mut PassTimes,
    ) -> Result<(), ImageGenError> {
        let ops = &*self.ops;
        let width = self.args.model_channels;
        let count = rows * width;
        let (cos_m1, sin) = times.time(ops, |t| &mut t.rotary, || self.rotary(repo, x, rows))?;

        let modulation = match (&block.norms, t_mod) {
            (Norms::Modulated(table), Some(t_mod)) => {
                let values: Vec<f32> = t_mod.iter().zip(table).map(|(m, s)| m + s).collect();
                Some(ops.upload_scratch("denoiser modulation", &values)?)
            }
            (Norms::Affine(..), None) => None,
            _ => {
                return Err(ImageGenError::Backend(
                    "a modulated block without a time modulation, or the reverse".to_owned(),
                ));
            }
        };
        // shift, scale, gate for the attention, then for the MLP.
        let chunk = |i: usize| -> Result<shiso_core::DeviceView<'_>, ImageGenError> {
            modulation
                .as_ref()
                .ok_or_else(|| ImageGenError::Backend("no modulation".to_owned()))?
                .view()
                .cut(bytes(i * width), bytes(width))
        };
        let norm = |step: &'static str, which: usize| -> Result<_, ImageGenError> {
            match &block.norms {
                Norms::Affine(pair) => layer_norm(
                    ops,
                    step,
                    x,
                    rows,
                    width,
                    BLOCK_EPS,
                    Some(if which == 0 { &pair.0 } else { &pair.1 }),
                ),
                Norms::Modulated(_) => {
                    let h = layer_norm(ops, step, x, rows, width, BLOCK_EPS, None)?;
                    ops.modulate(&ModulateCall {
                        step,
                        out: h.view(),
                        x: h.view(),
                        scale: chunk(3 * which + 1)?,
                        shift: chunk(3 * which)?,
                        count,
                        period: width,
                    })?;
                    Ok(h)
                }
            }
        };

        let h = norm("denoiser norm1", 0)?;
        let project = |lin: &Lin, turn: &Lin, head_norm: &HeadNorm| -> Result<_, ImageGenError> {
            let y = linear(ops, "denoiser qk", lin, h.view(), rows)?;
            let t = linear(ops, "denoiser qk", turn, h.view(), rows)?;
            rotate_in_place(
                ops,
                "denoiser rope",
                y.view(),
                t.view(),
                cos_m1.view(),
                sin.view(),
                count,
            )?;
            head_norm.apply(ops, "denoiser qk norm", y.view(), rows)?;
            Ok(y)
        };
        let q = project(&block.q, &block.q_turn, &block.q_norm)?;
        let k = project(&block.k, &block.k_turn, &block.k_norm)?;
        let v = linear(ops, "denoiser v", &block.v, h.view(), rows)?;
        drop(h);
        let attended = times.time(
            ops,
            |t| &mut t.attention,
            || {
                attention(
                    ops,
                    "denoiser attention",
                    Qkv {
                        q: q.view(),
                        k: k.view(),
                        v: v.view(),
                    },
                    AttnShape {
                        queries: rows,
                        keys: rows,
                        heads: self.args.num_heads,
                        head_dim: self.head_dim,
                    },
                )
            },
        )?;
        drop((q, k, v, cos_m1, sin));
        let o = linear(ops, "denoiser attn out", &block.out, attended.view(), rows)?;
        drop(attended);
        if modulation.is_some() {
            add_gated_in_place(
                ops,
                "denoiser residual",
                x,
                chunk(2)?,
                o.view(),
                count,
                width,
            )?;
        } else {
            ops.add("denoiser residual", x, x, o.view(), count)?;
        }
        drop(o);

        let h = norm("denoiser norm2", 1)?;
        let up = linear(ops, "denoiser mlp", &block.up, h.view(), rows)?;
        drop(h);
        ops.gelu_tanh(
            "denoiser mlp",
            up.view(),
            up.view(),
            rows * block.up.out_dim,
        )?;
        let down = linear(ops, "denoiser mlp", &block.down, up.view(), rows)?;
        if modulation.is_some() {
            add_gated_in_place(
                ops,
                "denoiser residual",
                x,
                chunk(5)?,
                down.view(),
                count,
                width,
            )
        } else {
            ops.add("denoiser residual", x, x, down.view(), count)
        }
    }

    /// The rotary tables for `rows` rows of `x`: `cos - 1` and `sin`, each
    /// `(rows, width)`, every pair `(2j, 2j + 1)` of a head turning by the same
    /// angle.
    fn rotary<'o>(
        &'o self,
        repo: &Repo,
        x: shiso_core::DeviceView<'_>,
        rows: usize,
    ) -> Result<
        (
            crate::diffusion::ops::Scratch<'o>,
            crate::diffusion::ops::Scratch<'o>,
        ),
        ImageGenError,
    > {
        let ops = &*self.ops;
        let width = self.args.model_channels;
        let heads = self.args.num_heads;
        let pairs = self.head_dim / 2;
        if repo.pairs != pairs {
            return Err(ImageGenError::Checkpoint(format!(
                "the rotary tables carry {} frequencies for {pairs} pairs",
                repo.pairs
            )));
        }
        let hn = layer_norm(
            ops,
            "denoiser repo",
            x,
            rows,
            width,
            DEFAULT_EPS,
            Some(&repo.norm),
        )?;
        let gate = linear(ops, "denoiser repo", &repo.gate, hn.view(), rows)?;
        let content = linear(ops, "denoiser repo", &repo.content, hn.view(), rows)?;
        drop(hn);
        ops.swiglu2(
            "denoiser repo",
            gate.view(),
            gate.view(),
            content.view(),
            rows * repo.gate.out_dim,
        )?;
        let offsets = linear(ops, "denoiser repo", &repo.project, gate.view(), rows)?;
        drop(gate);
        let cos_m1 = ops.alloc_f32("denoiser rope", rows * width)?;
        let sin = ops.alloc_f32("denoiser rope", rows * width)?;
        rotary_tables(
            ops,
            &RotaryCall {
                step: "denoiser rope",
                cos_m1: cos_m1.view(),
                sin: sin.view(),
                offsets: offsets.view(),
                rates: repo.rates.view(),
                axis: repo.axis.view(),
                rows,
                heads,
                head_dim: self.head_dim,
            },
        )?;
        Ok((cos_m1, sin))
    }
}

/// The read-out's parameter-free norm and modulation on one host row.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "a row of 1024 values; the statistics are taken in f64 and stored as f32"
)]
fn host_modulated_norm(row: &[f32], scale: &[f32], shift: &[f32]) -> Vec<f32> {
    let n = row.len() as f64;
    let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let var = row
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / n;
    let inv = 1.0 / (var + f64::from(DEFAULT_EPS)).sqrt();
    row.iter()
        .zip(scale.iter().zip(shift))
        .map(|(v, (s, t))| ((f64::from(*v) - mean) * inv) as f32 * (1.0 + s) + t)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the expected frequency is built in f64 and compared in f32"
    )]
    fn the_time_embedding_starts_with_the_cosine_of_t() {
        // Frequency 0 is exp(0) = 1, so column 0 is cos(t) and column 128 sin(t).
        let e = sinusoids(0.5, MAX_PERIOD, false);
        assert!((e[0] - 0.5f32.cos()).abs() < 1e-6);
        assert!((e[128] - 0.5f32.sin()).abs() < 1e-6);
        // The last frequency is 10000^(-127/128).
        let last = (-(MAX_PERIOD.ln()) * 127.0 / 128.0).exp() as f32;
        assert!((e[127] - (0.5 * last).cos()).abs() < 1e-6);
    }

    #[test]
    fn the_level_embedding_scales_by_two_pi() {
        let e = sinusoids(0.25, 1024.0, true);
        assert!((e[128] - (0.25 * 2.0 * PI).sin()).abs() < 1e-6);
    }

    #[test]
    fn the_host_norm_centres_and_scales_before_modulating() {
        let out = host_modulated_norm(&[1.0, 3.0], &[0.0, 1.0], &[0.5, 0.0]);
        // (x - 2) / 1 = [-1, 1] (eps aside), then [-1 * 1 + 0.5, 1 * 2 + 0].
        assert!((out[0] + 0.5).abs() < 1e-4, "{out:?}");
        assert!((out[1] - 2.0).abs() < 1e-4, "{out:?}");
    }

    #[test]
    fn gelu_tanh_matches_known_points() {
        assert!(gelu_tanh(0.0).abs() < 1e-7);
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-5);
        assert!((gelu_tanh(-1.0) + 0.158_808).abs() < 1e-5);
    }
}

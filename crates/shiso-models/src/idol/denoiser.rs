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
//! the device. The rotary positions are predicted on the device, and one
//! launch turns a block's queries and keys by them and applies the per-head
//! norm ([`super::splat_ops::rope_norm`]).
//!
//! The reference runs the blocks in fp16; this runs f32 throughout (see the
//! module doc of [`super::nn`]).
//!
//! # What one pass leaves for the next
//!
//! Two things, both between passes of one request and neither changing an
//! answer the reference would give:
//!
//! - The latent stream after its own refiner ([`Refined`]): a step's guided
//!   and unguided passes refine the same latent at the same timestep, so the
//!   second copies the first's rows.
//! - The conditioning rows' keys and values in every joint block ([`Kept`]),
//!   when the sampler asks for them with [`Pass::Keep`]. A [`Pass::Held`]
//!   pass then runs the latent and the camera alone and attends to those
//!   keys beside its own: the approximation a request opts into with a
//!   refresh interval, described on
//!   [`super::generator::Pipeline::refresh_every`].

use std::f32::consts::PI;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use shiso_core::DeviceBuffer;
use shiso_vision::generator::ImageGenError;

use super::conditioning::Conditioning;
use super::config::{DenoiserArgs, IdolConfig};
use super::generator::{Pass, Prepared};
use super::nn::{
    Affine, AttnShape, HEAD_NORM_EPS, HeadNorm, HeavyKey, Qkv, add_gated_in_place, attention,
    bytes, host_lin, layer_norm, linear, point_embed_v1, read_linear,
};
use super::splat_ops::{ATTN_HEAD, RopeNormCall, rope_norm};
use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{GemmCall, ModulateCall, Ops, SliceExt};
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
    k: Lin,
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
    /// What the last pass left for the next one.
    carried: Mutex<Carried>,
}

/// What a pass leaves for the next one of the same request.
#[derive(Debug, Default)]
struct Carried {
    refined: Option<Refined>,
    kept: Option<Kept>,
}

/// The latent stream after the noise refiner, kept from one pass for the next.
///
/// The refiner reads the latent, its positions and the timestep and nothing
/// else, so a step's guided and unguided passes - the same latent at the same
/// timestep under two conditionings - refine it to the same rows. The second
/// pass copies them instead of running the two blocks again: the same
/// launches' output, so the same bits (a 25-step request's file is byte for
/// byte the one it was before). Measured on an M4 at 8192 latent rows: 0.3 s
/// of a step's 13.6 s.
#[derive(Debug)]
struct Refined {
    /// The latent the rows were refined from, compared bit for bit.
    latent: Vec<f32>,
    /// The timestep they were refined at.
    timestep: f32,
    /// The refined rows, `(q_token_length, model_channels)`.
    rows: DeviceBuffer,
}

impl Refined {
    /// Whether these rows are the refinement of `latent` at `timestep`.
    fn is_for(&self, latent: &[f32], timestep: f32) -> bool {
        self.timestep.to_bits() == timestep.to_bits()
            && self.latent.len() == latent.len()
            && self
                .latent
                .iter()
                .zip(latent)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

/// The conditioning rows' keys and values in every joint block, as the last
/// [`Pass::Keep`] computed them: turned, normed and ready to attend to.
///
/// 24 blocks of two `(rows, model_channels)` buffers: 806 MiB at the
/// checkpoint's 4101 rows, held only while a request that asked for it
/// integrates ([`IdolDenoiser::release`]).
#[derive(Debug)]
struct Kept {
    /// Conditioning rows in each buffer.
    rows: usize,
    /// One pair per joint block.
    blocks: Vec<KeptBlock>,
}

/// One joint block's kept conditioning keys and values.
#[derive(Debug)]
struct KeptBlock {
    k: DeviceBuffer,
    v: DeviceBuffer,
}

/// Where a block's attention finds the keys and values that are not its own
/// rows', and where it leaves the ones that are.
#[derive(Debug, Clone, Copy)]
enum Keys<'a> {
    /// The block attends to its own rows and keeps nothing.
    Own,
    /// As [`Keys::Own`], and rows `first .. first + kept.rows` of the keys
    /// and values are copied out.
    Keep {
        /// Where they go.
        into: &'a KeptBlock,
        /// The first conditioning row.
        first: usize,
        /// How many there are.
        rows: usize,
    },
    /// The block attends to its own rows and then to `rows` kept ones.
    Held {
        /// The kept keys and values.
        from: &'a KeptBlock,
        /// How many there are.
        rows: usize,
    },
}

/// The rows one block runs over.
#[derive(Debug, Clone, Copy)]
struct Span<'a> {
    /// Rows held on the device.
    rows: usize,
    /// The row that stands for several identical ones, when there is one.
    heavy: Option<HeavyKey>,
    /// The keys beyond the block's own rows.
    keys: Keys<'a>,
}

impl Span<'_> {
    /// `rows` rows, each standing for itself, attending to themselves.
    const fn plain(rows: usize) -> Self {
        Self {
            rows,
            heavy: None,
            keys: Keys::Own,
        }
    }
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
    if head_dim != ATTN_HEAD || head_dim * heads != width {
        return Err(ImageGenError::Checkpoint(format!(
            "the denoiser attends {ATTN_HEAD}-wide heads, the checkpoint has {heads} heads over \
             {width}"
        )));
    }
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
        k: host_lin(ops, kw, Some(kb), width, width)?,
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
            carried: Mutex::new(Carried::default()),
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
    /// A conditioning whose rows are all the same row - the guidance pass's
    /// zero conditioning - is refined and answered as that one row with
    /// [`Prepared::copies`] set to its length. Nothing in the network tells
    /// equal conditioning rows apart (they carry no position; their rotary
    /// angles are predicted from their content), so they stay equal through
    /// every block, and `n` equal keys weigh in a softmax as one key scoring
    /// `ln(n)` higher. The joint blocks then run 8194 rows instead of 12294:
    /// the attention at 0.44 of its cost and the linears at 0.67.
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
        // Held rows and what each stands for.
        let (held, copies) = if n_c > 1
            && all_rows_equal(&cond.dinov3, a.cond_channels)
            && all_rows_equal(&cond.vae, a.cond2_channels)
        {
            (1, n_c)
        } else {
            (n_c, 1)
        };
        let n_c = held;
        let mut times = PassTimes::new();
        let stream = ops.alloc_f32("denoiser cond", n_c * width)?;
        let f1 = ops.upload_scratch("denoiser cond", &cond.dinov3[..n_c * a.cond_channels])?;
        let f2 = ops.upload_scratch("denoiser cond", &cond.vae[..n_c * a.cond2_channels])?;
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
        // A lone held row attends only to itself, where its weight cancels.
        for (repo, block) in self.context_repo.iter().zip(&self.context_refiner) {
            self.block(
                block,
                repo,
                stream.view(),
                Span::plain(n_c),
                None,
                &mut times,
            )?;
        }
        Ok(Prepared {
            rows: n_c,
            stream: ops.download("denoiser cond readback", stream.view(), n_c * width)?,
            copies,
        })
    }

    /// Drops what the passes carried between them: the refined latent and
    /// the kept conditioning keys. A request's last pass is followed by this,
    /// so the next request's encoders find the memory free.
    pub(crate) fn release(&self) {
        if let Ok(mut carried) = self.carried.lock() {
            *carried = Carried::default();
        }
    }

    /// The velocity at `(latent, camera)` for the model timestep `timestep`
    /// (the sampler's `t` times 1000) under the conditioning stream
    /// `prepared` (from [`IdolDenoiser::context`]), run as `pass` says.
    ///
    /// A [`Pass::Held`] pass lays its rows out as the latent then the camera
    /// and attends to the kept conditioning keys after them. The softmax is
    /// a sum over the same keys in another order than a full pass's, so a
    /// held pass straight after the [`Pass::Keep`] it reads, at the same
    /// latent and timestep, answers that pass's velocity to rounding.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Backend`] for inputs of the wrong shape, a
    /// [`Pass::Keep`] of a collapsed conditioning, a [`Pass::Held`] with no
    /// kept pass of this conditioning's length before it, or a device
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
        pass: Pass,
    ) -> Result<Velocity, ImageGenError> {
        let ops = &*self.ops;
        let started = Instant::now();
        let mut times = PassTimes::new();
        let a = &self.args;
        let width = a.model_channels;
        // A held pass carries no conditioning rows of its own.
        let n_x = a.q_token_length;
        let n_c = match pass {
            Pass::Full | Pass::Keep => prepared.rows,
            Pass::Held => 0,
        };
        let rows = n_x + n_c + 1;
        let mut carried = self
            .carried
            .lock()
            .map_err(|_poison| ImageGenError::Backend("the denoiser's carried state".to_owned()))?;
        let Carried { refined, kept } = &mut *carried;
        match pass {
            Pass::Full => {}
            Pass::Keep => {
                if prepared.copies != 1 {
                    return Err(ImageGenError::Backend(format!(
                        "a conditioning held as {} row(s) standing for {} each has no rows to keep",
                        prepared.rows, prepared.copies
                    )));
                }
                if kept.as_ref().is_none_or(|k| k.rows != prepared.rows) {
                    *kept = None;
                    let buffer =
                        || ops.alloc_owned_f32("denoiser kept keys", prepared.rows * width);
                    *kept = Some(Kept {
                        rows: prepared.rows,
                        blocks: (0..self.blocks.len())
                            .map(|_| {
                                Ok(KeptBlock {
                                    k: buffer()?,
                                    v: buffer()?,
                                })
                            })
                            .collect::<Result<_, ImageGenError>>()?,
                    });
                }
            }
            Pass::Held => {
                if kept.as_ref().is_none_or(|k| k.rows != prepared.rows) {
                    return Err(ImageGenError::Backend(format!(
                        "a held pass over {} conditioning rows follows no pass that kept them",
                        prepared.rows
                    )));
                }
            }
        }
        let kept = kept.as_ref();
        // Held rows standing for several are attended as one weighted key
        // each; the refiner only ever answers one such row.
        let heavy = match (prepared.copies, n_c) {
            (1, _) => None,
            (copies, 1) if copies > 1 => Some(HeavyKey { key: n_x, copies }),
            (copies, _) => {
                return Err(ImageGenError::Backend(format!(
                    "the denoiser attends one repeated conditioning row, got {n_c} rows \
                     standing for {copies} each"
                )));
            }
        };
        if pass != Pass::Held && prepared.stream.len() != n_c * width {
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

        // The latent stream: refined here, or copied from the pass before
        // when that one refined this latent at this timestep (see `Refined`).
        if let Some(same) = refined.as_ref().filter(|r| r.is_for(latent, timestep)) {
            ops.copy("denoiser refined latent", stream(0, n_x)?, same.rows.view())?;
        } else {
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
                self.block(
                    block,
                    repo,
                    stream(0, n_x)?,
                    Span::plain(n_x),
                    Some(&t_mod),
                    &mut times,
                )?;
            }
            // One latent stream (32 MiB), reused by every later pass.
            let rows = match refined.take() {
                Some(last) => last.rows,
                None => ops.alloc_owned_f32("denoiser refined latent", n_x * width)?,
            };
            ops.copy("denoiser refined latent", rows.view(), stream(0, n_x)?)?;
            *refined = Some(Refined {
                latent: latent.to_vec(),
                timestep,
                rows,
            });
        }

        // The conditioning stream, refined once per request: copied in, as
        // the joint blocks below write over these rows.
        if n_c > 0 {
            let cond_bytes: Vec<u8> = prepared
                .stream
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            ops.backend()
                .copy_h2d(stream(n_x, n_c)?, &cond_bytes)
                .map_err(|e| ImageGenError::Backend(format!("denoiser cond rows: {e}")))?;
        }

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

        for (index, (repo, block)) in self.repo.iter().zip(&self.blocks).enumerate() {
            let keys = match (pass, kept) {
                (Pass::Keep, Some(kept)) => Keys::Keep {
                    into: &kept.blocks[index],
                    first: n_x,
                    rows: kept.rows,
                },
                (Pass::Held, Some(kept)) => Keys::Held {
                    from: &kept.blocks[index],
                    rows: kept.rows,
                },
                _ => Keys::Own,
            };
            self.block(
                block,
                repo,
                h.view(),
                Span { rows, heavy, keys },
                Some(&t_mod),
                &mut times,
            )?;
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

    /// One block over `span`'s rows of `x`, in place, with its rotary
    /// positions predicted by `repo` from the block's input.
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
        span: Span<'_>,
        t_mod: Option<&[f32]>,
        times: &mut PassTimes,
    ) -> Result<(), ImageGenError> {
        let Span { rows, heavy, keys } = span;
        let ops = &*self.ops;
        let width = self.args.model_channels;
        let count = rows * width;
        let offsets = times.time(ops, |t| &mut t.rotary, || self.offsets(repo, x, rows))?;

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
        let q = linear(ops, "denoiser qk", &block.q, h.view(), rows)?;
        // The keys and values this block attends to: its own rows', and
        // after them the kept ones when the pass holds any. The projections
        // write the head of each buffer; the kept rows are copied in behind.
        let key_rows = rows
            + match keys {
                Keys::Held { rows, .. } => rows,
                Keys::Own | Keys::Keep { .. } => 0,
            };
        let k = ops.alloc_f32("denoiser qk", key_rows * width)?;
        let v = ops.alloc_f32("denoiser v", key_rows * width)?;
        let own = bytes(count);
        for (step, lin, out) in [
            ("denoiser qk", &block.k, k.view().cut(0, own)?),
            ("denoiser v", &block.v, v.view().cut(0, own)?),
        ] {
            ops.gemm(&GemmCall {
                step,
                out,
                input: h.view(),
                weight: lin.weight(),
                m: rows,
                n: lin.out_dim,
                k: lin.in_dim,
            })?;
        }
        drop(h);
        times.time(
            ops,
            |t| &mut t.rotary,
            || {
                rope_norm(
                    ops,
                    &RopeNormCall {
                        step: "denoiser rope",
                        q: q.view(),
                        k: k.view().cut(0, own)?,
                        offsets: offsets.view(),
                        rates: repo.rates.view(),
                        axis: repo.axis.view(),
                        gamma_q: block.q_norm.gamma(),
                        gamma_k: block.k_norm.gamma(),
                        rows,
                        heads: self.args.num_heads,
                        head_dim: self.head_dim,
                        eps: HEAD_NORM_EPS,
                    },
                )
            },
        )?;
        drop(offsets);
        match keys {
            Keys::Own => {}
            Keys::Keep {
                into,
                first,
                rows: kept_rows,
            } => {
                let (at, len) = (bytes(first * width), bytes(kept_rows * width));
                ops.copy("denoiser kept keys", into.k.view(), k.view().cut(at, len)?)?;
                ops.copy("denoiser kept keys", into.v.view(), v.view().cut(at, len)?)?;
            }
            Keys::Held {
                from,
                rows: kept_rows,
            } => {
                let len = bytes(kept_rows * width);
                ops.copy("denoiser kept keys", k.view().cut(own, len)?, from.k.view())?;
                ops.copy("denoiser kept keys", v.view().cut(own, len)?, from.v.view())?;
            }
        }
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
                        keys: key_rows,
                        heads: self.args.num_heads,
                        head_dim: self.head_dim,
                        heavy,
                    },
                )
            },
        )?;
        drop((q, k, v));
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

    /// The rotary positions `repo` predicts for `rows` rows of `x`: three
    /// offsets per head, `(rows, heads, 3)`.
    fn offsets<'o>(
        &'o self,
        repo: &Repo,
        x: shiso_core::DeviceView<'_>,
        rows: usize,
    ) -> Result<crate::diffusion::ops::Scratch<'o>, ImageGenError> {
        let ops = &*self.ops;
        let width = self.args.model_channels;
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
        linear(ops, "denoiser repo", &repo.project, gate.view(), rows)
    }
}

/// Whether every `width`-wide row of `values` is bit for bit the first one.
fn all_rows_equal(values: &[f32], width: usize) -> bool {
    let mut rows = values.chunks_exact(width);
    let Some(first) = rows.next() else {
        return false;
    };
    let same = |a: &f32, b: &f32| a.to_bits() == b.to_bits();
    rows.all(|row| row.iter().zip(first).all(|(a, b)| same(a, b)))
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

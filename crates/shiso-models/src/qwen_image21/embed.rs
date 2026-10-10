//! The Qwen-Image-2.1 embedders: timestep, shared modulation, text and
//! latent input projections.
//!
//! - **Timestep.** The pipeline passes sigma in [0, 1]; the sinusoid takes
//!   `t = 1000 * sigma` against 128 frequencies `exp(-ln(10000) * j / 128)`
//!   and lays out the 128 cosines FIRST, then the 128 sines.
//!   `temb = linear_2(silu(linear_1(sinusoid)))`.
//! - **Modulation.** One `modulation.1` linear for the whole model (no
//!   per-block weights) maps `silu(temb)` to `[scale1 | gate1 | scale2 |
//!   gate2]`, 4096 each: attention then MLP, scale then gate, and no shift.
//!   A block normalizes as `LN(x) * (1 + scale)` and adds `tanh(gate) *
//!   branch`; the gates are passed through `tanh` once here, so the fused
//!   linears' gated-residual epilogue takes them as they are. The output
//!   norm's scale is `norm_out.linear(silu(temb))`, also scale-only.
//! - **Text.** `txt_in(x) = out_layer(gelu_tanh(in_layer(norm(x))))`, the
//!   norm zero-centred RMS: `x * rsqrt(mean(x^2) + eps) * (1 + w)`. The
//!   `1 + w` is formed once at load, so the device runs a plain RMS norm.
//! - **Latents.** `img_in` maps each 64-channel latent token to the width.
//!
//! Under `causal_condition` the text tokens are modulated at `t = 0`
//! whatever the step, so that modulation is a constant of the weights,
//! computed once at load by [`Embedders::modulation_at`].

use shiso_core::{DeviceBuffer, DeviceView};
use shiso_vision::generator::ImageGenError;

use super::config::{DitConfig, DitNames, SINUSOID_DIM};
use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{GemmCall, NormCall, Ops, Scratch, SliceExt};
use crate::diffusion::weights::StStore;

/// The pipeline's sigma-to-timestep factor, applied inside the sinusoid.
const TIME_FACTOR: f32 = 1000.0;

/// The sinusoid's longest period.
const MAX_PERIOD: f32 = 10_000.0;

/// The timestep sinusoid of `sigma`: cosines then sines, `SINUSOID_DIM`
/// values.
///
/// Every operation is in `f32`, in the order exponent, `exp`, times the
/// sigma, times 1000. The reference scales the sigma first (1000 sigma, then
/// times the frequency), so the two orders can differ by an `f32` rounding
/// in the angle; at `t = 1000` one ulp of the argument is 6e-5 of the sine.
/// Measured against the captured reference the sinusoid agrees to that
/// level, and the end-to-end image is unchanged, so the order stays as
/// written rather than being made more precise: a more exact angle would
/// move the features away from the ones the model ran on in the reference.
#[expect(
    clippy::cast_precision_loss,
    reason = "frequency indices are below 128, exact in f32"
)]
pub(crate) fn sinusoid(sigma: f32) -> Vec<f32> {
    let half = SINUSOID_DIM / 2;
    let neg_log = -MAX_PERIOD.ln();
    let args: Vec<f32> = (0..half)
        .map(|j| {
            let freq = (neg_log * j as f32 / half as f32).exp();
            sigma * freq * TIME_FACTOR
        })
        .collect();
    args.iter()
        .map(|a| a.cos())
        .chain(args.iter().map(|a| a.sin()))
        .collect()
}

/// The time conditioning of one sigma, on the device.
pub(crate) struct TimeCond<'o> {
    /// `temb`, `(hidden)`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the head reads the final scale derived from it; only the device tests read temb itself"
        )
    )]
    pub temb: Scratch<'o>,
    /// `[scale1 | gate1 | scale2 | gate2]`, `(4 * hidden)`; the gates are
    /// raw until [`Embedders::tanh_gates`] runs.
    pub modulation: Scratch<'o>,
    /// The output norm's scale, `(hidden)`.
    pub final_scale: Scratch<'o>,
}

/// Every non-block weight of the transformer.
#[derive(Debug)]
pub(crate) struct Embedders {
    time_1: Lin,
    time_2: Lin,
    modulation: Lin,
    norm_out: Lin,
    /// `1 + text_norm.weight`, f32.
    txt_norm: DeviceBuffer,
    txt_in: Lin,
    txt_out: Lin,
    img_in: Lin,
    hidden: usize,
    context: usize,
    in_channels: usize,
    eps: f32,
}

impl Embedders {
    /// Loads the non-block weights (shapes already checked by
    /// [`DitNames::check`]).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the config's eps (1e-6) is the f32 the norm kernels take"
    )]
    pub(crate) fn load(
        store: &StStore,
        ops: &Ops,
        names: &DitNames,
        config: &DitConfig,
    ) -> Result<Self, ImageGenError> {
        let lin = |module: &str| Lin::load(store, ops, &names.top(module));
        let norm: Vec<f32> = store
            .f32_vec(&format!("{}.weight", names.top("txt_in.text_norm")))?
            .iter()
            .map(|w| 1.0 + w)
            .collect();
        Ok(Self {
            time_1: lin("time_text_embed.timestep_embedder.linear_1")?,
            time_2: lin("time_text_embed.timestep_embedder.linear_2")?,
            modulation: lin("modulation.1")?,
            norm_out: lin("norm_out.linear")?,
            txt_norm: ops.upload_f32("qwen txt norm", &norm)?,
            txt_in: lin("txt_in.in_layer")?,
            txt_out: lin("txt_in.out_layer")?,
            img_in: lin("img_in")?,
            hidden: config.hidden(),
            context: config.context_in_dim,
            in_channels: config.in_channels,
            eps: config.eps as f32,
        })
    }

    /// A one-row product `out = lin(input)`.
    fn row(
        ops: &Ops,
        step: &'static str,
        lin: &Lin,
        input: DeviceView<'_>,
        out: DeviceView<'_>,
    ) -> Result<(), ImageGenError> {
        ops.gemm(&GemmCall {
            step,
            out,
            input,
            weight: lin.weight(),
            m: 1,
            n: lin.out_dim,
            k: lin.in_dim,
        })
    }

    /// `temb`, the raw modulation and the output scale at `sigma`.
    ///
    /// Uploads the sinusoid, and an upload drains the device queue first:
    /// for the load-time `t = 0` row and the tests. A step reads a row the
    /// job uploaded once instead ([`Embedders::time_cond_of`]).
    pub(crate) fn time_cond<'o>(
        &self,
        ops: &'o Ops,
        sigma: f32,
    ) -> Result<TimeCond<'o>, ImageGenError> {
        let features = ops.upload_scratch("qwen sinusoid", &sinusoid(sigma))?;
        self.time_cond_of(ops, features.view())
    }

    /// [`Embedders::time_cond`] from sinusoid features already on the
    /// device, `(SINUSOID_DIM)`: no host transfer.
    pub(crate) fn time_cond_of<'o>(
        &self,
        ops: &'o Ops,
        features: DeviceView<'_>,
    ) -> Result<TimeCond<'o>, ImageGenError> {
        let h = self.hidden;
        let hidden1 = ops.alloc_f32("qwen temb", h)?;
        Self::row(ops, "qwen temb 1", &self.time_1, features, hidden1.view())?;
        ops.silu("qwen temb silu", hidden1.view(), hidden1.view(), h)?;
        let temb = ops.alloc_f32("qwen temb", h)?;
        Self::row(
            ops,
            "qwen temb 2",
            &self.time_2,
            hidden1.view(),
            temb.view(),
        )?;
        // Both heads read silu(temb); temb itself stays for the tests.
        ops.silu("qwen temb silu", hidden1.view(), temb.view(), h)?;
        let modulation = ops.alloc_f32("qwen modulation", 4 * h)?;
        Self::row(
            ops,
            "qwen modulation",
            &self.modulation,
            hidden1.view(),
            modulation.view(),
        )?;
        let final_scale = ops.alloc_f32("qwen norm_out", h)?;
        Self::row(
            ops,
            "qwen norm_out",
            &self.norm_out,
            hidden1.view(),
            final_scale.view(),
        )?;
        Ok(TimeCond {
            temb,
            modulation,
            final_scale,
        })
    }

    /// `tanh` over the two gate quarters of a `[scale1 | gate1 | scale2 |
    /// gate2]` modulation, in place.
    pub(crate) fn tanh_gates(
        &self,
        ops: &Ops,
        modulation: DeviceView<'_>,
    ) -> Result<(), ImageGenError> {
        let quarter_bytes = (self.hidden * 4) as u64;
        for at in [1u64, 3] {
            let gate = modulation.cut(at * quarter_bytes, quarter_bytes)?;
            ops.tanh("qwen gate tanh", gate, gate, self.hidden)?;
        }
        Ok(())
    }

    /// The block modulation at `sigma` with its gates through `tanh`, in a
    /// buffer of its own: the text tokens' constant `t = 0` row.
    pub(crate) fn modulation_at(
        &self,
        ops: &Ops,
        sigma: f32,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let cond = self.time_cond(ops, sigma)?;
        self.tanh_gates(ops, cond.modulation.view())?;
        let owned = ops.alloc_owned_f32("qwen modulation t0", 4 * self.hidden)?;
        ops.copy("qwen modulation t0", owned.view(), cond.modulation.view())?;
        Ok(owned)
    }

    /// `txt_in` over `p` rows of text features, into a pooled stream.
    pub(crate) fn txt_in<'o>(
        &self,
        ops: &'o Ops,
        feats: DeviceView<'_>,
        p: usize,
    ) -> Result<Scratch<'o>, ImageGenError> {
        let h = self.hidden;
        let normed = ops.alloc_f32("qwen txt norm", p * self.context)?;
        ops.rmsnorm(&NormCall {
            step: "qwen txt norm",
            out: normed.view(),
            input: feats,
            weight: self.txt_norm.view(),
            rows: p,
            dim: self.context,
            eps: self.eps,
        })?;
        let inner = ops.alloc_f32("qwen txt in", p * h)?;
        ops.gemm(&GemmCall {
            step: "qwen txt in",
            out: inner.view(),
            input: normed.view(),
            weight: self.txt_in.weight(),
            m: p,
            n: h,
            k: self.context,
        })?;
        ops.gelu_tanh("qwen txt gelu", inner.view(), inner.view(), p * h)?;
        let out = ops.alloc_f32("qwen txt stream", p * h)?;
        ops.gemm(&GemmCall {
            step: "qwen txt out",
            out: out.view(),
            input: inner.view(),
            weight: self.txt_out.weight(),
            m: p,
            n: h,
            k: h,
        })?;
        Ok(out)
    }

    /// `img_in` over `t` latent tokens, into `out` `(t, hidden)`.
    pub(crate) fn img_in(
        &self,
        ops: &Ops,
        latents: DeviceView<'_>,
        t: usize,
        out: DeviceView<'_>,
    ) -> Result<(), ImageGenError> {
        ops.gemm(&GemmCall {
            step: "qwen img in",
            out,
            input: latents,
            weight: self.img_in.weight(),
            m: t,
            n: self.hidden,
            k: self.in_channels,
        })
    }
}

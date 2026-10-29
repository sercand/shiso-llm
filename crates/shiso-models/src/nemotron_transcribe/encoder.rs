//! One recording through the speech branch, the speaker branch and the
//! fusion: the rows the decoder reads in place of the audio placeholder.

#![expect(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    reason = "the names follow the model math (c, w, d, s); eight speakers is exact in f32"
)]

use shiso_audio::{AudioError, MelFrames};
use shiso_core::DeviceBuffer;

use super::config::PeeConfig;
use super::diar::speaker_activity;
use super::frontend::{chunk_plan, normalize, stack_rows};
use super::transformer::{add, layer_norm};
use super::weights::PeeWeights;
use crate::audio_ops::Ops;
use crate::audio_ops::load::{download_f32, upload_f32};

/// What one recording encodes to.
#[derive(Debug)]
pub struct Encoded {
    /// `[frames][fusion_dim]`, on the device: the decoder's audio rows.
    pub rows: DeviceBuffer,
    /// Encoder frames, 80 ms each.
    pub frames: usize,
    /// The speech branch's output `[frames][d_model]`, on the host: what the
    /// timestamp head reads.
    pub speech: Vec<f32>,
    /// Speaker activity `[frames][speakers]`.
    pub activity: Vec<f32>,
}

/// The speech branch over every window of `plan`, its core rows joined.
fn speech(
    ops: &Ops,
    w: &PeeWeights,
    c: &PeeConfig,
    normed: &[f32],
    plan: &[super::frontend::ChunkSpan],
) -> Result<Vec<f32>, AudioError> {
    let d = c.asr.d_model;
    let mut out = Vec::new();
    for win in plan {
        let rows = win.len().div_ceil(c.stack);
        let stacked = upload_f32(
            ops,
            &stack_rows(normed, c.mels, win.enc_start, win.len(), c.stack),
        )?;
        let x = ops.scratch(rows * d)?;
        w.asr_proj.forward(ops, stacked.view(), x.view(), rows)?;
        let y = w.asr.forward(ops, &x, rows)?;
        let all = download_f32(ops, y.view(), rows * d)?;
        out.extend_from_slice(&all[win.left_drop * d..(win.left_drop + win.core) * d]);
    }
    Ok(out)
}

/// The fused rows: `proj(LN(speech) + kernel · LN(activity > threshold))`.
fn fuse(
    ops: &Ops,
    w: &PeeWeights,
    c: &PeeConfig,
    speech: &[f32],
    activity: &[f32],
    frames: usize,
) -> Result<DeviceBuffer, AudioError> {
    let d = c.asr.d_model;
    let s = c.speakers;
    let (nw, nb) = &w.fusion.diar_norm;
    let mut speaker_rows = vec![0.0f32; frames * d];
    for (row, probs) in speaker_rows
        .chunks_exact_mut(d)
        .zip(activity.chunks_exact(s))
    {
        let bin: Vec<f32> = probs
            .iter()
            .map(|p| if *p > c.threshold { 1.0 } else { 0.0 })
            .collect();
        let mean = bin.iter().sum::<f32>() / s as f32;
        let var = bin.iter().map(|b| (b - mean) * (b - mean)).sum::<f32>() / s as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (k, b) in bin.iter().enumerate() {
            let normed = (b - mean) * inv * nw[k] + nb[k];
            for (o, kv) in row.iter_mut().zip(&w.fusion.kernel[k * d..(k + 1) * d]) {
                *o += kv * normed;
            }
        }
    }
    let x = upload_f32(ops, speech)?;
    let normed = ops.scratch(frames * d)?;
    layer_norm(ops, normed.view(), x.view(), &w.fusion.asr_norm, frames, d)?;
    let extra = upload_f32(ops, &speaker_rows)?;
    add(ops, normed.view(), extra.view(), frames * d)?;
    let rows = ops.scratch(frames * c.fusion_dim)?;
    w.fusion
        .proj
        .forward(ops, normed.view(), rows.view(), frames)?;
    Ok(rows)
}

/// Encodes one recording's raw log-mel.
///
/// # Errors
///
/// [`AudioError::Empty`] for a clip with no frames, or a device error.
pub fn encode(
    ops: &Ops,
    w: &PeeWeights,
    c: &PeeConfig,
    mel: &MelFrames,
) -> Result<Encoded, AudioError> {
    let t_mel = mel.valid();
    if t_mel == 0 {
        return Err(AudioError::Empty);
    }
    let plan = chunk_plan(t_mel, c);
    let normed = normalize(mel);
    let raw = &mel.as_slice()[..t_mel * c.mels];
    let speech = speech(ops, w, c, &normed, &plan)?;
    let frames = speech.len() / c.asr.d_model;
    let mut activity = Vec::with_capacity(frames * c.speakers);
    for (win, rows) in plan
        .iter()
        .zip(speaker_activity(ops, w, c, raw, t_mel, &plan)?)
    {
        activity.extend_from_slice(
            &rows[win.left_drop * c.speakers..(win.left_drop + win.core) * c.speakers],
        );
    }
    let rows = fuse(ops, w, c, &speech, &activity, frames)?;
    Ok(Encoded {
        rows,
        frames,
        speech,
        activity,
    })
}

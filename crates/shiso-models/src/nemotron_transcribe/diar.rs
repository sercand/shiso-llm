//! The speaker branch: eight speakers' activity every 80 ms, carried across
//! windows by a speaker cache.
//!
//! Each window's raw mel is stacked and projected into embeddings; the first
//! window is scored on its own, and every later one behind the speaker cache
//! and the FIFO of embeddings the windows before it left. A window's core
//! rows are the ones kept, and the state is updated the way the reference's
//! synchronous streaming update does it: the core joins the FIFO, an
//! overflowing FIFO spills into the cache, and a cache past 188 rows is
//! compressed to the most confidently single-speaker frames of each speaker,
//! grouped by speaker, each group closed by the learned silence embedding.
//!
//! Positions restart at zero over cache, FIFO and window together.

#![expect(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::neg_cmp_op_on_partial_ord,
    reason = "the names and the negated comparisons follow the reference's own code, whose `!(p > 0.5)` also disables a NaN; the casts are budgets of a few hundred rows"
)]

use shiso_audio::AudioError;
use shiso_core::DeviceBuffer;

use super::config::PeeConfig;
use super::frontend::{ChunkSpan, stack_rows};
use super::weights::PeeWeights;
use crate::audio_ops::load::{download_f32, upload_f32};
use crate::audio_ops::{Ops, UnaryDesc, UnaryOp, UnfoldDesc};
use crate::nemotron_diar::weights::UPSAMPLE_TAPS;

/// Rows the speaker cache holds.
const CACHE_LEN: usize = 188;
/// Rows the FIFO holds between windows.
const FIFO_LEN: usize = 40;
/// Rows moved from the FIFO to the cache at once.
const UPDATE_PERIOD: usize = 300;
/// Silence frames kept per speaker when the cache is compressed.
const SIL_PER_SPEAKER: usize = 1;
/// The probability floor inside a frame's score.
const SCORE_FLOOR: f32 = 0.25;
/// The log-domain boost of frames that just joined the cache.
const BOOST_LATEST: f32 = 0.05;
/// Shares of a speaker's budget boosted four- and two-fold.
const STRONG_RATE: f32 = 0.75;
const WEAK_RATE: f32 = 1.5;
/// Share of a speaker's budget past which overlapping frames are dropped.
const MIN_POS_RATE: f32 = 0.5;

/// The state the windows carry: embeddings and their activities.
#[derive(Debug, Default)]
struct State {
    cache: Vec<f32>,
    cache_preds: Vec<f32>,
    fifo: Vec<f32>,
    fifo_preds: Vec<f32>,
    compressed: bool,
}

/// Indices of the `k` largest of `v`, ties to the lower index.
fn topk(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.truncate(k.min(v.len()));
    idx
}

/// Keeps `CACHE_LEN` of the cache's `n` rows: the reference's compression,
/// scored in its product domain so the same frames are kept.
fn compress(state: &mut State, n: usize, d: usize, speakers: usize, silence: &[f32]) {
    let per = CACHE_LEN / speakers - SIL_PER_SPEAKER;
    let strong = (per as f32 * STRONG_RATE).floor() as usize;
    let weak = (per as f32 * WEAK_RATE).floor() as usize;
    let min_pos = (per as f32 * MIN_POS_RATE).floor() as usize;
    let latest = BOOST_LATEST.exp();
    let nt = n + SIL_PER_SPEAKER;
    let preds = &state.cache_preds;
    let mut scores = vec![vec![f32::INFINITY; nt]; speakers];
    for t in 0..n {
        let p = &preds[t * speakers..(t + 1) * speakers];
        for s in 0..speakers {
            let mut prod = 1.0f32;
            for (j, pj) in p.iter().enumerate() {
                prod *= if j == s {
                    1.0
                } else {
                    (1.0 - pj).max(SCORE_FLOOR)
                };
            }
            prod *= p[s].max(SCORE_FLOOR);
            scores[s][t] = prod * 2.0;
        }
    }
    for (s, row) in scores.iter_mut().enumerate() {
        let mut positive = 0usize;
        for t in 0..n {
            if !(preds[t * speakers + s] > 0.5) {
                row[t] = 0.0;
            }
            positive += usize::from(row[t] > 1.0);
        }
        if positive >= min_pos {
            for v in row.iter_mut().take(n) {
                if !(*v > 1.0) {
                    *v = 0.0;
                }
            }
        }
        for v in row.iter_mut().take(n).skip(CACHE_LEN) {
            *v *= latest;
        }
        for t in topk(&row[..n], strong) {
            row[t] *= 4.0;
        }
        for t in topk(&row[..n], weak) {
            row[t] *= 2.0;
        }
    }
    let flat: Vec<f32> = scores.into_iter().flatten().collect();
    let placeholder = flat.len();
    let mut pick: Vec<usize> = topk(&flat, CACHE_LEN)
        .into_iter()
        .map(|i| if flat[i] == 0.0 { placeholder } else { i })
        .collect();
    pick.sort_unstable();
    let mut emb = vec![0.0f32; CACHE_LEN * d];
    let mut out = vec![0.0f32; CACHE_LEN * speakers];
    for (i, &p) in pick.iter().enumerate() {
        let t = (p != placeholder).then_some(p % nt).filter(|t| *t < n);
        let e = &mut emb[i * d..(i + 1) * d];
        match t {
            None => e.copy_from_slice(silence),
            Some(t) => {
                e.copy_from_slice(&state.cache[t * d..(t + 1) * d]);
                out[i * speakers..(i + 1) * speakers]
                    .copy_from_slice(&state.cache_preds[t * speakers..(t + 1) * speakers]);
            }
        }
    }
    state.cache = emb;
    state.cache_preds = out;
}

/// The speaker head over `rows` transformer rows: activities at the mel rate,
/// `[rows·stack][speakers]`, on the host.
fn head(
    ops: &Ops,
    w: &PeeWeights,
    c: &PeeConfig,
    hidden: &DeviceBuffer,
    rows: usize,
) -> Result<Vec<f32>, AudioError> {
    let hh = c.head_hidden;
    let fine = rows * c.stack;
    let z = ops.scratch(rows * hh)?;
    let u = ops.scratch(rows * UPSAMPLE_TAPS * hh)?;
    let up = ops.scratch(fine * hh)?;
    let g = ops.scratch(fine * hh)?;
    let logits = ops.scratch(fine * c.speakers)?;
    let h = &w.head;
    h.enc_proj.forward(ops, hidden.view(), z.view(), rows)?;
    ops.unfold(&UnfoldDesc {
        out: u.view(),
        input: z.view(),
        rows,
        channels: hh,
        taps: UPSAMPLE_TAPS,
        pad: UPSAMPLE_TAPS / 2,
    })?;
    // `[rows][stack·head]` is `[rows·stack][head]` in the same bytes.
    h.upsample.forward(ops, u.view(), up.view(), rows)?;
    let relu = |x: &DeviceBuffer, count: usize| {
        ops.unary(&UnaryDesc {
            op: UnaryOp::Relu,
            out: x.view(),
            input: x.view(),
            count,
        })
    };
    relu(&up, fine * hh)?;
    h.dense.forward(ops, up.view(), g.view(), fine)?;
    relu(&g, fine * hh)?;
    h.out.forward(ops, g.view(), logits.view(), fine)?;
    let mut p = download_f32(ops, logits.view(), fine * c.speakers)?;
    for v in &mut p {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
    Ok(p)
}

/// Every window's speaker activity over its encoder rows, `[rows][speakers]`
/// for each window in `plan` (rows `left_drop .. left_drop + core` filled),
/// at 80 ms: each row the mean of its eight 10 ms frames that lie inside the
/// recording.
///
/// # Errors
///
/// A device error.
pub fn speaker_activity(
    ops: &Ops,
    w: &PeeWeights,
    c: &PeeConfig,
    raw_mel: &[f32],
    mel_frames: usize,
    plan: &[ChunkSpan],
) -> Result<Vec<Vec<f32>>, AudioError> {
    let d = c.diar.d_model;
    let n_spk = c.speakers;
    let sub = c.stack;
    let mut state = State::default();
    let mut out = Vec::with_capacity(plan.len());
    for (wi, win) in plan.iter().enumerate() {
        let tb = win.len().div_ceil(sub);
        let stacked = stack_rows(raw_mel, c.mels, win.enc_start, win.len(), sub);
        let x = upload_f32(ops, &stacked)?;
        let emb_dev = ops.scratch(tb * d)?;
        w.diar_proj.forward(ops, x.view(), emb_dev.view(), tb)?;
        let emb = download_f32(ops, emb_dev.view(), tb * d)?;
        let s_rows = state.cache.len() / d;
        let f_rows = state.fifo.len() / d;
        let l = s_rows + f_rows + tb;
        let mut input = Vec::with_capacity(l * d);
        input.extend_from_slice(&state.cache);
        input.extend_from_slice(&state.fifo);
        input.extend_from_slice(&emb);
        let hidden = w.diar.forward(ops, &upload_f32(ops, &input)?, l)?;
        let p10 = head(ops, w, c, &hidden, l)?;
        // 80 ms rows: the mean of each row's 10 ms frames inside the
        // recording (the cache and FIFO rows' all count).
        let mut p80 = vec![0.0f32; l * n_spk];
        for r in 0..l {
            let first = if r >= s_rows + f_rows {
                win.enc_start + (r - s_rows - f_rows) * sub
            } else {
                0
            };
            let live = (0..sub)
                .filter(|j| r < s_rows + f_rows || first + j < mel_frames)
                .collect::<Vec<_>>();
            let denom = (live.len() as f32).max(1e-6);
            for s in 0..n_spk {
                let sum: f32 = live.iter().map(|j| p10[(r * sub + j) * n_spk + s]).sum();
                p80[r * n_spk + s] = sum / denom;
            }
        }
        let lc = win.left_drop;
        let rc = win.right().div_ceil(sub);
        let core = tb.saturating_sub(lc + rc);
        let mut rows = vec![0.0f32; tb * n_spk];
        if core > 0 {
            for t in 0..win.core {
                let src = s_rows + f_rows + lc + t.min(core - 1);
                rows[(lc + t) * n_spk..(lc + t + 1) * n_spk]
                    .copy_from_slice(&p80[src * n_spk..(src + 1) * n_spk]);
            }
        }
        out.push(rows);
        if wi + 1 == plan.len() {
            break;
        }
        state.fifo_preds = p80[s_rows * n_spk..(s_rows + f_rows) * n_spk].to_vec();
        state.fifo.extend_from_slice(&emb[lc * d..(lc + core) * d]);
        let first = s_rows + f_rows + lc;
        state
            .fifo_preds
            .extend_from_slice(&p80[first * n_spk..(first + core) * n_spk]);
        if f_rows + core > FIFO_LEN {
            let pop =
                (UPDATE_PERIOD.max((core + f_rows).saturating_sub(FIFO_LEN))).min(f_rows + core);
            if !state.compressed {
                state.cache_preds = p80[..s_rows * n_spk].to_vec();
            }
            state.cache.extend(state.fifo.drain(..pop * d));
            let moved: Vec<f32> = state.fifo_preds.drain(..pop * n_spk).collect();
            state.cache_preds.extend(moved);
            let n = state.cache.len() / d;
            if n > CACHE_LEN {
                compress(&mut state, n, d, n_spk, &w.head.silence);
                state.compressed = true;
            }
        }
    }
    Ok(out)
}

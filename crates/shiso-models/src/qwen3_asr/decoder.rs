//! The dense decoder on the device: the prompt with the tower's rows in place
//! of its audio placeholders, one prefill, then greedy decoding until an end
//! id or the token cap (notes §3, §4).
//!
//! The prefill is one pass of GEMMs over every prompt row. Decoding then
//! follows the transducer's pattern: each step - the embedding row of the
//! state's last token, 28 layers on the one-row matrix-vector kernels with
//! attention over the cache, the tied head, the row argmax and the step
//! kernel - is queued without a read, up to 8 steps at a time, and the host
//! reads the emitted ids and the state once per batch. A step's layer is the
//! fused launches of `audio_ops` (the residual add inside the next norm, the
//! three projections in one launch, the query and key norms with the
//! rotation and the append, the gated pair with its product): nine launches
//! where the plain chain takes nineteen, each value formed as that chain
//! forms it. A step after the decoder stopped changes
//! nothing, so a batch that overshoots the end costs time and nothing else.
//!
//! Everything stays f32 - the residual stream, every norm, the attention and
//! the cache - whatever width the weights are stored in, because the
//! reference ran in f32 from the bf16 weights and the ids are held to it
//! exactly.
//!
//! The decoder state's position (`DECODE_POS`) is the one thing the host
//! writes between prefill and decode: the prefill appends its rows from a
//! state at position 0, and the host then writes the prefill's row count
//! before the first step, which advances by 0 over the prefill's own argmax.

use std::time::{Duration, Instant};

use shiso_audio::AudioError;
use shiso_core::{DType, DeviceBuffer, DeviceView};
use shiso_engine::projection::WeightAux;

use super::config::{AUDIO_PAD, STOP_TOKENS, TextConfig};
use super::encoder::Encoded;
use super::prompt;
use super::weights::{TextLayer, TextWeights};
use crate::audio_ops::{
    AddRmsNormDesc, ArgmaxWideDesc, AxpyDesc, CachePlanes, CausalAttnDesc, DECODE_ATTN_SPLITS,
    DECODE_COUNT, DECODE_DONE, DECODE_NO_TOKEN, DECODE_POS, DECODE_STATE_WORDS, DECODE_TOKEN,
    DecodeAttnDesc, DecodeStepDesc, EmbedTable, EmbedTokenDesc, KvAppendDesc, Linear, LinearWeight,
    NormDesc, Ops, QkRopeAppendDesc, RopeDesc, SiluMulDesc, Workspace, argmax_wide_parts,
    decode_attention_partials, step_silu, step3,
};

/// Tokens one transcription may emit: the reference's vLLM path's
/// `max_new_tokens` (notes §3, Generation). A chunk is at most 35 s, which
/// speech fills with a few hundred tokens; the cap only bounds a degenerate
/// repetition, which the parser then collapses.
pub const MAX_NEW_TOKENS: usize = 4096;
/// Steps queued per read-back. A pass here is some 8 ms and a read-back
/// some 0.2 ms, the other way round from the transducer's search, so the
/// batch is small: every step after the end in the last batch is a whole
/// wasted pass. On long-dialogue (214 tokens over 3 chunks, M1 Max) 32 ran
/// 253 passes, 16 ran 237, 8 ran 221 and 4 ran 217, and 8 and 4 took the
/// same time.
const BATCH: usize = 8;
/// Cache rows past the prompt a fresh cache is sized for: a transcript of a
/// full chunk is a few hundred tokens, so most chunks never grow it.
const HEADROOM: usize = 256;
/// The longest chunk the stream's plan cuts, in mel frames: 35 s, a 30 s
/// chunk and the 5 s its cut is searched for past the mark.
const LONGEST_CHUNK_FRAMES: usize = 3_500;
/// The capacity a fresh cache takes for the longest chunk, in any language:
/// the prompt and the headroom. A cache past it grew while a chunk decoded
/// far past any speech - toward the token cap, at some 320 KB a row - and
/// is not worth keeping once its stream closes.
pub(crate) const POOLED_CAPACITY: usize = prompt::FRAME_IDS
    + super::encoder::audio_tokens(LONGEST_CHUNK_FRAMES)
    + prompt::LONGEST_SUFFIX
    + HEADROOM;

fn bytes(floats: usize) -> u64 {
    u64::try_from(floats).unwrap_or(u64::MAX).saturating_mul(4)
}

/// Rows `[row, row + count)` of a buffer of `width`-float rows.
fn rows(
    buf: &DeviceBuffer,
    row: usize,
    count: usize,
    width: usize,
) -> Result<DeviceView<'_>, AudioError> {
    Ok(buf.slice(bytes(row * width), bytes(count * width))?)
}

fn words(host: &[u8]) -> Vec<u32> {
    host.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn word(value: usize, what: &str) -> Result<u32, AudioError> {
    u32::try_from(value).map_err(|_| AudioError::BadSpec {
        reason: format!("{what} {value} does not fit a u32"),
    })
}

fn count_of(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// `1 / sqrt(head_dim)`, the scale on every attention logit.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the kernel takes the scale as f32"
)]
fn attention_scale(head_dim: usize) -> f32 {
    (1.0 / crate::audio_ops::f64_of(head_dim).sqrt()) as f32
}

/// One stream's key and value cache: per layer two f32 planes of
/// `[kv_heads][capacity][head_dim]`, the layout the cache kernels read.
///
/// A transcription starts it empty and appends every prompt row and every
/// decoded token. The capacity only grows: to the prompt plus a headroom
/// before the prefill, then by doubling (a copy on the device of the rows
/// already filled) before any batch of steps that could pass it, because the
/// append drops a row at or past the capacity rather than failing.
#[derive(Debug)]
pub struct KvCache {
    /// `[k, v]` per layer; empty until the first transcription sizes it.
    planes: Vec<[DeviceBuffer; 2]>,
    capacity: usize,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Rows past the prompt the first sizing adds.
    headroom: usize,
    /// Times the planes were reallocated.
    grown: usize,
}

impl KvCache {
    /// An empty cache for a decoder of `config`'s geometry; the first
    /// transcription allocates it.
    #[must_use]
    pub fn new(config: &TextConfig) -> Self {
        Self {
            planes: Vec::new(),
            capacity: 0,
            layers: config.layers,
            kv_heads: config.kv_heads,
            head_dim: config.head_dim,
            headroom: HEADROOM,
            grown: 0,
        }
    }

    /// A cache of `capacity` rows with no headroom, so a transcription grows
    /// it to exactly its prompt and then again while it decodes.
    #[cfg(test)]
    pub(crate) fn tight(
        ops: &Ops,
        config: &TextConfig,
        capacity: usize,
    ) -> Result<Self, AudioError> {
        let mut cache = Self::new(config);
        cache.headroom = 0;
        cache.reserve(ops, 0, capacity)?;
        cache.grown = 0;
        Ok(cache)
    }

    /// A cache that says it holds `capacity` rows and holds none, for the
    /// stream pool's bookkeeping: nothing may transcribe with it.
    #[cfg(test)]
    pub(crate) fn claiming(config: &TextConfig, capacity: usize) -> Self {
        let mut cache = Self::new(config);
        cache.capacity = capacity;
        cache
    }

    /// Rows the cache holds without growing.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Times the planes were reallocated since the cache was made.
    #[cfg(test)]
    pub(crate) fn grown(&self) -> usize {
        self.grown
    }

    fn plane_floats(&self, capacity: usize) -> usize {
        self.kv_heads * capacity * self.head_dim
    }

    /// Makes room for `need` rows, keeping the first `filled`: to `need` or,
    /// while rows are filled, twice the capacity, whichever is more. An
    /// empty cache (before a prefill) takes exactly `need`, so a stream whose
    /// chunks grow only by the prompt stays at [`POOLED_CAPACITY`] or under.
    fn reserve(&mut self, ops: &Ops, filled: usize, need: usize) -> Result<(), AudioError> {
        if need <= self.capacity {
            return Ok(());
        }
        let capacity = if filled == 0 {
            need
        } else {
            need.max(2 * self.capacity)
        };
        let floats = self.plane_floats(capacity);
        let mut planes = Vec::with_capacity(self.layers);
        for _ in 0..self.layers {
            // Rows past the filled ones are never read before they are
            // written, so the new planes are not cleared.
            planes.push([
                shiso_core::alloc(ops.backend(), bytes(floats))?,
                shiso_core::alloc(ops.backend(), bytes(floats))?,
            ]);
        }
        if filled > 0 {
            let run = filled * self.head_dim;
            for (new, old) in planes.iter().zip(&self.planes) {
                for (to, from) in new.iter().zip(old) {
                    for h in 0..self.kv_heads {
                        ops.backend().copy_d2d(
                            to.slice(bytes(h * capacity * self.head_dim), bytes(run))?,
                            from.slice(bytes(h * self.capacity * self.head_dim), bytes(run))?,
                        )?;
                    }
                }
            }
        }
        self.planes = planes;
        self.capacity = capacity;
        self.grown += 1;
        Ok(())
    }

    fn layer(&self, layer: usize) -> CachePlanes<'_> {
        let [k, v] = &self.planes[layer];
        CachePlanes {
            k: k.view(),
            v: v.view(),
            kv_heads: self.kv_heads,
            cap: self.capacity,
            head_dim: self.head_dim,
        }
    }
}

/// Where one transcription's time went.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Timing {
    /// Prompt rows.
    pub(crate) prompt: usize,
    /// Embedding, prefill and the first argmax, to the host's handoff.
    pub(crate) prefill: Duration,
    /// Every batch of steps, read-backs included.
    pub(crate) decode: Duration,
    /// Decode passes run, including those after the end in the last batch.
    pub(crate) passes: usize,
    /// Tokens emitted.
    pub(crate) tokens: usize,
}

/// Activations of one transcription, sized for its prompt; a decode pass
/// uses the first row of each. The large ones are workspace slots, kept
/// between transcriptions; every element of those is written before it is
/// read.
struct Scratch<'w> {
    x: &'w DeviceBuffer,
    h: &'w DeviceBuffer,
    q: &'w DeviceBuffer,
    k: &'w DeviceBuffer,
    v: &'w DeviceBuffer,
    att: &'w DeviceBuffer,
    gate: &'w DeviceBuffer,
    up: &'w DeviceBuffer,
    logits: &'w DeviceBuffer,
    /// The head's argmax, one `u32`.
    ids: DeviceBuffer,
    /// The wide argmax's candidates.
    parts: DeviceBuffer,
    partials: DeviceBuffer,
    state: DeviceBuffer,
    /// One `u32` per step of a batch.
    out: DeviceBuffer,
}

/// Where the head's input row comes from.
#[derive(Clone, Copy)]
enum Pass {
    /// A prefill row of `s.x`, its layers' residuals all added.
    Prefill,
    /// A decode pass's row: `s.x` plus the last layer's residual in `s.h`.
    Decode,
}

/// The decoder over one loaded checkpoint.
#[derive(Debug)]
pub struct Decoder<'a> {
    ops: &'a Ops,
    weights: &'a TextWeights,
    config: &'a TextConfig,
}

impl<'a> Decoder<'a> {
    /// Binds the decoder to its weights.
    #[must_use]
    pub fn new(ops: &'a Ops, weights: &'a TextWeights, config: &'a TextConfig) -> Self {
        Self {
            ops,
            weights,
            config,
        }
    }

    /// The greedy transcript of `audio`'s rows, as ids: the prompt of
    /// [`prompt::ids`] with the rows in place of its placeholders, decoded
    /// until an end id (which is not returned) or [`MAX_NEW_TOKENS`]. With
    /// `forced` the prompt ends in `language {forced}<asr_text>` and the ids
    /// are the transcript alone; without it they start with the model's own
    /// `language X<asr_text>`.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] for rows of another width than the decoder's,
    /// no rows, or a forced language the model does not support;
    /// [`AudioError::Checkpoint`] for an embedding stored in an encoding no
    /// row fetch reads; a device error.
    pub fn transcribe(
        &self,
        cache: &mut KvCache,
        audio: &Encoded,
        forced: Option<&str>,
    ) -> Result<Vec<u32>, AudioError> {
        self.transcribe_with(cache, &mut Workspace::new(), audio, forced)
    }

    /// [`Self::transcribe`] in `ws`'s buffers, which a model keeps from one
    /// transcription to the next.
    ///
    /// # Errors
    ///
    /// As [`Self::transcribe`].
    pub fn transcribe_with(
        &self,
        cache: &mut KvCache,
        ws: &mut Workspace,
        audio: &Encoded,
        forced: Option<&str>,
    ) -> Result<Vec<u32>, AudioError> {
        let (tokens, t) = self.timed_with(cache, ws, audio, forced)?;
        tracing::debug!(
            prompt = t.prompt,
            prefill_ms = t.prefill.as_secs_f64() * 1e3,
            decode_ms = t.decode.as_secs_f64() * 1e3,
            passes = t.passes,
            tokens = t.tokens,
            "transcribed a chunk"
        );
        Ok(tokens)
    }

    /// [`Self::transcribe`], with where the time went.
    #[cfg(test)]
    pub(crate) fn timed(
        &self,
        cache: &mut KvCache,
        audio: &Encoded,
        forced: Option<&str>,
    ) -> Result<(Vec<u32>, Timing), AudioError> {
        self.timed_with(cache, &mut Workspace::new(), audio, forced)
    }

    /// [`Self::transcribe_with`], with where the time went.
    pub(crate) fn timed_with(
        &self,
        cache: &mut KvCache,
        ws: &mut Workspace,
        audio: &Encoded,
        forced: Option<&str>,
    ) -> Result<(Vec<u32>, Timing), AudioError> {
        let c = self.config;
        if audio.tokens == 0 {
            return Err(AudioError::BadSpec {
                reason: "no audio rows to transcribe".into(),
            });
        }
        if audio.buffer.len() < bytes(audio.tokens * c.hidden) {
            return Err(AudioError::BadSpec {
                reason: format!(
                    "{} audio rows in {} bytes, not rows of the decoder's {}",
                    audio.tokens,
                    audio.buffer.len(),
                    c.hidden
                ),
            });
        }
        let ids = prompt::ids(audio.tokens, forced)?;
        let n = ids.len();
        let start = Instant::now();
        let s = self.scratch(n, ws)?;
        cache.reserve(self.ops, 0, n + cache.headroom)?;
        self.embed_prompt(&s, &ids, audio)?;
        for (l, layer) in self.weights.layers.iter().enumerate() {
            self.layer(layer, cache.layer(l), &s, n)?;
        }
        self.head(&s, n - 1, Pass::Prefill)?;
        // The handoff: the prefill appended from position 0; the first step
        // decodes from `n`. The write waits for the prefill.
        let mut handoff = [0u32; DECODE_STATE_WORDS];
        handoff[DECODE_POS] = word(n, "a prompt of")?;
        let raw: Vec<u8> = handoff.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.ops.backend().copy_h2d(s.state.view(), &raw)?;
        let prefill = start.elapsed();
        let (tokens, passes) = self.decode(cache, &s, n)?;
        let timing = Timing {
            prompt: n,
            prefill,
            decode: start.elapsed().saturating_sub(prefill),
            passes,
            tokens: tokens.len(),
        };
        Ok((tokens, timing))
    }

    fn scratch<'w>(&self, n: usize, ws: &'w mut Workspace) -> Result<Scratch<'w>, AudioError> {
        let (c, ops) = (self.config, self.ops);
        let (qw, kw) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let (d, f) = (n * c.hidden, n * c.ffn);
        ws.reserve(ops, &[d, d, n * qw, n * kw, n * kw, n * qw, f, f, c.vocab])?;
        let ws = &*ws;
        Ok(Scratch {
            x: ws.get(0),
            h: ws.get(1),
            q: ws.get(2),
            k: ws.get(3),
            v: ws.get(4),
            att: ws.get(5),
            gate: ws.get(6),
            up: ws.get(7),
            logits: ws.get(8),
            ids: ops.scratch(1)?,
            parts: ops.scratch(argmax_wide_parts(c.vocab))?,
            partials: ops.scratch(decode_attention_partials(
                c.heads,
                c.head_dim,
                DECODE_ATTN_SPLITS,
            ))?,
            state: ops.scratch(DECODE_STATE_WORDS)?,
            out: ops.scratch(BATCH)?,
        })
    }

    /// The embedding as the row fetch reads it.
    fn table(&self) -> Result<EmbedTable<'_>, AudioError> {
        let embed: &Linear = &self.weights.embed;
        match &embed.weight {
            LinearWeight::Gemm { dtype, buffer, .. } => Ok(EmbedTable::Dense {
                dtype: *dtype,
                rows: buffer.view(),
            }),
            LinearWeight::Projected(w) => match (w.dtype, w.aux().and_then(WeightAux::affine)) {
                (DType::MlxAffine8Gs64, Some(aux)) => Ok(EmbedTable::Affine8 {
                    words: w.view(),
                    scales: aux.scales.view(),
                    biases: aux.biases.view(),
                    group: 64,
                }),
                (other, _) => Err(AudioError::Checkpoint {
                    what: format!("the embedding is stored as {other:?}, which no row fetch reads"),
                }),
            },
        }
    }

    /// The prompt's rows into `s.x`: each text id's embedding row, and the
    /// tower's rows over the placeholder run, in order and unscaled (notes
    /// §2.7).
    fn embed_prompt(
        &self,
        s: &Scratch<'_>,
        ids: &[u32],
        audio: &Encoded,
    ) -> Result<(), AudioError> {
        let (ops, c) = (self.ops, self.config);
        let first =
            ids.iter()
                .position(|&t| t == AUDIO_PAD)
                .ok_or_else(|| AudioError::BadSpec {
                    reason: "a prompt with no audio placeholder".into(),
                })?;
        ops.backend().copy_d2d(
            rows(s.x, first, audio.tokens, c.hidden)?,
            rows(&audio.buffer, 0, audio.tokens, c.hidden)?,
        )?;
        // The row fetch reads its id from a decoder state; each text row gets
        // a state of its own, uploaded at once.
        let text: Vec<(usize, u32)> = ids
            .iter()
            .enumerate()
            .filter(|&(i, _)| i < first || i >= first + audio.tokens)
            .map(|(i, &t)| (i, t))
            .collect();
        let mut raw = vec![0u8; text.len() * DECODE_STATE_WORDS * 4];
        for (j, &(_, t)) in text.iter().enumerate() {
            let at = (j * DECODE_STATE_WORDS + DECODE_TOKEN) * 4;
            raw[at..at + 4].copy_from_slice(&t.to_le_bytes());
        }
        let states = ops.scratch(text.len() * DECODE_STATE_WORDS)?;
        ops.backend().copy_h2d(states.view(), &raw)?;
        let table = self.table()?;
        for (j, &(i, _)) in text.iter().enumerate() {
            ops.embed_token(&EmbedTokenDesc {
                out: rows(s.x, i, 1, c.hidden)?,
                table,
                state: rows(&states, j, 1, DECODE_STATE_WORDS)?,
                width: c.hidden,
                vocab: c.vocab,
            })?;
        }
        Ok(())
    }

    fn rms(
        &self,
        out: DeviceView<'_>,
        input: DeviceView<'_>,
        weight: &DeviceBuffer,
        rows: usize,
        dim: usize,
    ) -> Result<(), AudioError> {
        Ok(self.ops.rms_norm(&NormDesc {
            out,
            input,
            weight: Some(weight.view()),
            bias: None,
            rows,
            dim,
            eps: self.config.rms_eps,
        })?)
    }

    fn residual(
        &self,
        x: &DeviceBuffer,
        add: &DeviceBuffer,
        count: usize,
    ) -> Result<(), AudioError> {
        Ok(self.ops.axpy(&AxpyDesc {
            y: x.view(),
            x: add.view(),
            alpha: 1.0,
            count,
        })?)
    }

    /// One prefill layer over the `n` prompt rows of `s.x`, in place (notes
    /// §4): every projection on the audio set's tiled GEMM (an 8-bit affine
    /// weight too), attention causal from position 0.
    fn layer(
        &self,
        layer: &TextLayer,
        cache: CachePlanes<'_>,
        s: &Scratch<'_>,
        n: usize,
    ) -> Result<(), AudioError> {
        let (ops, c) = (self.ops, self.config);
        let hd = c.head_dim;
        self.rms(s.h.view(), s.x.view(), &layer.attn_norm, n, c.hidden)?;
        layer.q.forward_own(ops, s.h.view(), s.q.view(), n)?;
        layer.k.forward_own(ops, s.h.view(), s.k.view(), n)?;
        layer.v.forward_own(ops, s.h.view(), s.v.view(), n)?;
        // The q and k norms are over each head's dims, before the rotation.
        self.rms(s.q.view(), s.q.view(), &layer.q_norm, n * c.heads, hd)?;
        self.rms(s.k.view(), s.k.view(), &layer.k_norm, n * c.kv_heads, hd)?;
        for (x, heads) in [(&s.q, c.heads), (&s.k, c.kv_heads)] {
            ops.rope_half(&RopeDesc {
                x: x.view(),
                rows: n,
                heads,
                head_dim: hd,
                pos0: 0,
                theta: c.rope_theta,
            })?;
        }
        ops.kv_append(&KvAppendDesc {
            cache,
            k: s.k.view(),
            v: s.v.view(),
            state: s.state.view(),
            rows: n,
        })?;
        ops.causal_attention(&CausalAttnDesc {
            out: s.att.view(),
            q: s.q.view(),
            cache,
            n_q: n,
            heads: c.heads,
            pos0: 0,
            scale: attention_scale(hd),
        })?;
        layer.o.forward_own(ops, s.att.view(), s.h.view(), n)?;
        self.residual(s.x, s.h, n * c.hidden)?;
        self.rms(s.h.view(), s.x.view(), &layer.ffn_norm, n, c.hidden)?;
        layer.gate.forward_own(ops, s.h.view(), s.gate.view(), n)?;
        layer.up.forward_own(ops, s.h.view(), s.up.view(), n)?;
        ops.silu_mul(&SiluMulDesc {
            out: s.gate.view(),
            gate: s.gate.view(),
            up: s.up.view(),
            count: n * c.ffn,
        })?;
        layer.down.forward_own(ops, s.gate.view(), s.h.view(), n)?;
        self.residual(s.x, s.h, n * c.hidden)
    }

    /// The final norm of row `row` of `s.x`, the tied head, and its argmax
    /// into `s.ids`; a tie keeps the lowest id, as the reference's argmax
    /// does. After a decode pass the last layer's residual is still in
    /// `s.h`, and is added first.
    fn head(&self, s: &Scratch<'_>, row: usize, pass: Pass) -> Result<(), AudioError> {
        let (ops, c) = (self.ops, self.config);
        let h = rows(s.h, 0, 1, c.hidden)?;
        match pass {
            Pass::Prefill => self.rms(
                h,
                rows(s.x, row, 1, c.hidden)?,
                &self.weights.norm,
                1,
                c.hidden,
            )?,
            Pass::Decode => self.add_rms(s, &self.weights.norm)?,
        }
        self.weights.embed.step(ops, h, s.logits.view(), 1)?;
        Ok(ops.argmax_wide(&ArgmaxWideDesc {
            out: s.ids.view(),
            input: s.logits.view(),
            parts: s.parts.view(),
            cols: c.vocab,
        })?)
    }

    /// `x += h`, then `h = rms_norm(x) * weight`, over a decode pass's row.
    fn add_rms(&self, s: &Scratch<'_>, weight: &DeviceBuffer) -> Result<(), AudioError> {
        let c = self.config;
        let h = rows(s.h, 0, 1, c.hidden)?;
        Ok(self.ops.add_rms_norm(&AddRmsNormDesc {
            out: h,
            x: rows(s.x, 0, 1, c.hidden)?,
            add: h,
            weight: Some(weight.view()),
            rows: 1,
            dim: c.hidden,
            eps: c.rms_eps,
        })?)
    }

    /// One layer of a decode pass over `s.x`'s row, in the fused launches.
    /// The layer's input is `x + h` - the residual the layer before left in
    /// `s.h` is added by this layer's first norm (the first layer's `h` is
    /// already normed) - and its own last residual is left in `s.h` the same
    /// way: the values are the ones [`Self::layer`] forms, in fewer launches.
    fn layer_step(
        &self,
        layer: &TextLayer,
        cache: CachePlanes<'_>,
        s: &Scratch<'_>,
        first: bool,
    ) -> Result<(), AudioError> {
        let (ops, c) = (self.ops, self.config);
        if first {
            self.rms(s.h.view(), s.x.view(), &layer.attn_norm, 1, c.hidden)?;
        } else {
            self.add_rms(s, &layer.attn_norm)?;
        }
        step3(
            ops,
            [&layer.q, &layer.k, &layer.v],
            s.h.view(),
            [s.q.view(), s.k.view(), s.v.view()],
        )?;
        ops.qk_norm_rope_append(&QkRopeAppendDesc {
            q: s.q.view(),
            k: s.k.view(),
            v: s.v.view(),
            q_norm: layer.q_norm.view(),
            k_norm: layer.k_norm.view(),
            cache,
            heads: c.heads,
            rows: 1,
            eps: c.rms_eps,
            theta: c.rope_theta,
            state: s.state.view(),
        })?;
        ops.decode_attention(&DecodeAttnDesc {
            out: s.att.view(),
            partials: s.partials.view(),
            splits: DECODE_ATTN_SPLITS,
            q: s.q.view(),
            cache,
            heads: c.heads,
            state: s.state.view(),
            scale: attention_scale(c.head_dim),
        })?;
        layer.o.step(ops, s.att.view(), s.h.view(), 1)?;
        self.add_rms(s, &layer.ffn_norm)?;
        step_silu(
            ops,
            [&layer.gate, &layer.up],
            s.h.view(),
            s.gate.view(),
            s.up.view(),
            1,
        )?;
        layer.down.step(ops, s.gate.view(), s.h.view(), 1)?;
        Ok(())
    }

    fn step(&self, s: &Scratch<'_>, advance: usize, slot: usize) -> Result<(), AudioError> {
        Ok(self.ops.decode_step(&DecodeStepDesc {
            state: s.state.view(),
            out: s.out.view(),
            ids: s.ids.view(),
            eos: STOP_TOKENS,
            max_tokens: MAX_NEW_TOKENS,
            advance,
            slot,
        })?)
    }

    /// One decode pass over the state's last token, ending in the head's
    /// argmax.
    fn pass(&self, cache: &KvCache, s: &Scratch<'_>) -> Result<(), AudioError> {
        let c = self.config;
        self.ops.embed_token(&EmbedTokenDesc {
            out: s.x.view(),
            table: self.table()?,
            state: s.state.view(),
            width: c.hidden,
            vocab: c.vocab,
        })?;
        for (l, layer) in self.weights.layers.iter().enumerate() {
            self.layer_step(layer, cache.layer(l), s, l == 0)?;
        }
        self.head(s, 0, Pass::Decode)
    }

    /// Greedy steps from the prefill's argmax until the state says done;
    /// returns the emitted ids and the passes run.
    fn decode(
        &self,
        cache: &mut KvCache,
        s: &Scratch<'_>,
        prompt: usize,
    ) -> Result<(Vec<u32>, usize), AudioError> {
        let backend = self.ops.backend();
        let mut tokens = Vec::new();
        let (mut pos, mut emitted, mut passes) = (prompt, 0usize, 0usize);
        let mut first = true;
        loop {
            // The first batch's first step takes the prefill's argmax, with no
            // pass before it. Past the cap a step emits nothing, so no batch
            // runs more passes than tokens remain.
            let room = MAX_NEW_TOKENS.saturating_sub(emitted);
            let run = if first { BATCH - 1 } else { BATCH }.min(room);
            // Each pass appends one row at the state's position, which only
            // the steps move; the append drops rows past the capacity, so the
            // cache grows before the batch is queued.
            cache.reserve(self.ops, pos, pos + run)?;
            let mut slot = 0;
            if first {
                self.step(s, 0, 0)?;
                slot = 1;
            }
            for _ in 0..run {
                self.pass(cache, s)?;
                self.step(s, 1, slot)?;
                slot += 1;
            }
            passes += run;
            let mut host = vec![0u8; 4 * slot];
            backend.copy_d2h(s.out.slice(0, bytes(slot))?, &mut host)?;
            tokens.extend(words(&host).into_iter().filter(|&t| t != DECODE_NO_TOKEN));
            let mut host = vec![0u8; 4 * DECODE_STATE_WORDS];
            backend.copy_d2h(s.state.view(), &mut host)?;
            let st = words(&host);
            pos = count_of(st[DECODE_POS]);
            let now = count_of(st[DECODE_COUNT]);
            if st[DECODE_DONE] != 0 {
                break;
            }
            if now == emitted && !first {
                return Err(AudioError::BadSpec {
                    reason: format!(
                        "a batch of {run} decode steps emitted nothing and did not stop"
                    ),
                });
            }
            emitted = now;
            first = false;
        }
        Ok((tokens, passes))
    }
}

#[cfg(test)]
#[path = "decoder_tests.rs"]
mod tests;

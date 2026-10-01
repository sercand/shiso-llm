//! The dense decoder on the device: the prompt with the tower's rows in place
//! of its audio placeholders, one prefill, then greedy decoding until an end
//! id or the token cap.
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
//!
//! A layer's per-row work - the norms, the projections, the residuals, the
//! gated MLP and the head - is written once here (`Decoder::prefill_layer`,
//! `Decoder::step_layer`, `Decoder::logits`) with the attention between
//! them left to the caller, so the batched decoder (`super::batch`) forms
//! every row exactly as this one does and differs only in how its sequences
//! attend.

use std::ops::Range;
use std::time::{Duration, Instant};

use shiso_audio::AudioError;
use shiso_core::{DType, DeviceBuffer, DeviceView};
use shiso_engine::projection::WeightAux;

use super::config::{AUDIO_PAD, STOP_TOKENS, TextConfig};
use super::encoder::Encoded;
use super::prompt::{self, Conditioning};
use super::weights::{TextLayer, TextWeights};
use crate::audio_ops::{
    AddRmsNormDesc, ArgmaxWideDesc, AxpyDesc, CachePlanes, CausalAttnDesc, DECODE_ATTN_SPLITS,
    DECODE_COUNT, DECODE_DONE, DECODE_NO_TOKEN, DECODE_POS, DECODE_STATE_WORDS, DECODE_TOKEN,
    DecodeAttnDesc, DecodeStepDesc, EmbedTable, EmbedTokenDesc, KvAppendDesc, Linear, LinearWeight,
    NormDesc, Ops, QkRopeAppendDesc, RopeDesc, SiluMulDesc, Workspace, argmax_wide_parts,
    decode_attention_partials, step_silu, step3,
};

/// Tokens one transcription may emit: the `max_new_tokens` the reference
/// serving path generates with. A chunk is at most 35 s, which
/// speech fills with a few hundred tokens; the cap only bounds a degenerate
/// repetition, which the parser then collapses.
pub const MAX_NEW_TOKENS: usize = 4096;
/// Steps queued per read-back. A pass here is some 8 ms and a read-back
/// some 0.2 ms, the other way round from the transducer's search, so the
/// batch is small: every step after the end in the last batch is a whole
/// wasted pass. On long-dialogue (214 tokens over 3 chunks, M1 Max) 32 ran
/// 253 passes, 16 ran 237, 8 ran 221 and 4 ran 217, and 8 and 4 took the
/// same time.
pub(super) const BATCH: usize = 8;
/// Cache rows past the prompt a fresh cache is sized for: a transcript of a
/// full chunk is a few hundred tokens, so most chunks never grow it.
pub(super) const HEADROOM: usize = 256;
/// The longest chunk the stream's plan cuts, in mel frames: 35 s, a 30 s
/// chunk and the 5 s its cut is searched for past the mark.
const LONGEST_CHUNK_FRAMES: usize = 3_500;
/// The capacity a fresh cache takes for the longest chunk, in any language:
/// the prompt with the longest context, and the headroom. A cache past it
/// grew while a chunk decoded far past any speech - toward the token cap, at
/// some 320 KB a row - and is not worth keeping once its stream closes.
pub(crate) const POOLED_CAPACITY: usize = prompt::FRAME_IDS
    + super::encoder::audio_tokens(LONGEST_CHUNK_FRAMES)
    + prompt::LONGEST_SUFFIX
    + prompt::MAX_CONTEXT_IDS
    + HEADROOM;

pub(super) fn bytes(floats: usize) -> u64 {
    u64::try_from(floats).unwrap_or(u64::MAX).saturating_mul(4)
}

/// Rows `[row, row + count)` of a buffer of `width`-float rows.
pub(super) fn rows(
    buf: &DeviceBuffer,
    row: usize,
    count: usize,
    width: usize,
) -> Result<DeviceView<'_>, AudioError> {
    Ok(buf.slice(bytes(row * width), bytes(count * width))?)
}

pub(super) fn words(host: &[u8]) -> Vec<u32> {
    host.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(super) fn word(value: usize, what: &str) -> Result<u32, AudioError> {
    u32::try_from(value).map_err(|_| AudioError::BadSpec {
        reason: format!("{what} {value} does not fit a u32"),
    })
}

pub(super) fn count_of(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// `1 / sqrt(head_dim)`, the scale on every attention logit.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the kernel takes the scale as f32"
)]
pub(super) fn attention_scale(head_dim: usize) -> f32 {
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
    /// The rows each reallocation copied, in order.
    #[cfg(test)]
    copied: Vec<usize>,
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
            #[cfg(test)]
            copied: Vec::new(),
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
        cache.copied.clear();
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

    /// The rows each reallocation since the cache was made copied.
    #[cfg(test)]
    pub(crate) fn copied(&self) -> &[usize] {
        &self.copied
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
        #[cfg(test)]
        self.copied.push(filled);
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

/// The per-row activations of a prefill's rows, which a decode pass reuses
/// for its row (or its batch's rows): workspace slots, kept between
/// transcriptions, every element of which is written before it is read.
pub(super) struct Acts<'w> {
    pub(super) x: &'w DeviceBuffer,
    pub(super) h: &'w DeviceBuffer,
    pub(super) q: &'w DeviceBuffer,
    pub(super) k: &'w DeviceBuffer,
    pub(super) v: &'w DeviceBuffer,
    pub(super) att: &'w DeviceBuffer,
    pub(super) gate: &'w DeviceBuffer,
    pub(super) up: &'w DeviceBuffer,
    /// The head's rows of logits.
    pub(super) logits: &'w DeviceBuffer,
}

/// What a layer's input row is, where the head takes it.
#[derive(Clone, Copy)]
pub(super) enum HeadInput<'v> {
    /// Rows whose layers' residuals are all added: a prefill's last rows.
    Rows(DeviceView<'v>),
    /// A decode pass's rows: `x` plus the last layer's residual in `h`.
    Residual,
}

/// Activations of one transcription, sized for its prompt; a decode pass
/// uses the first row of each.
struct Scratch<'w> {
    a: Acts<'w>,
    /// The head's argmax, one `u32`.
    ids: DeviceBuffer,
    /// The wide argmax's candidates.
    parts: DeviceBuffer,
    partials: DeviceBuffer,
    state: DeviceBuffer,
    /// One `u32` per step of a batch.
    out: DeviceBuffer,
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
    /// `cond.forced` the prompt ends in `language {forced}<asr_text>` and the
    /// ids are the transcript alone; without it they start with the model's
    /// own `language X<asr_text>`.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] for rows of another width than the decoder's,
    /// no rows, or a forced language the model does not support;
    /// [`AudioError::BadContext`] for more context ids than a prompt carries;
    /// [`AudioError::Checkpoint`] for an embedding stored in an encoding no
    /// row fetch reads; a device error.
    pub fn transcribe(
        &self,
        cache: &mut KvCache,
        audio: &Encoded,
        cond: &Conditioning<'_>,
    ) -> Result<Vec<u32>, AudioError> {
        self.transcribe_with(cache, &mut Workspace::new(), audio, cond)
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
        cond: &Conditioning<'_>,
    ) -> Result<Vec<u32>, AudioError> {
        let (tokens, t) = self.timed_with(cache, ws, audio, cond)?;
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
        cond: &Conditioning<'_>,
    ) -> Result<(Vec<u32>, Timing), AudioError> {
        self.timed_with(cache, &mut Workspace::new(), audio, cond)
    }

    /// [`Self::transcribe_with`], with where the time went.
    pub(crate) fn timed_with(
        &self,
        cache: &mut KvCache,
        ws: &mut Workspace,
        audio: &Encoded,
        cond: &Conditioning<'_>,
    ) -> Result<(Vec<u32>, Timing), AudioError> {
        self.check_audio(audio)?;
        let ids = prompt::ids(audio.tokens, cond)?;
        let n = ids.len();
        let start = Instant::now();
        let s = self.scratch(n, ws)?;
        cache.reserve(self.ops, 0, n + cache.headroom)?;
        self.embed_prompt(s.a.x, 0, &ids, audio)?;
        for (l, layer) in self.weights.layers.iter().enumerate() {
            self.layer(layer, cache.layer(l), &s, n)?;
        }
        let last = rows(s.a.x, n - 1, 1, self.config.hidden)?;
        self.head(&s, HeadInput::Rows(last))?;
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

    /// Refuses audio rows the decoder cannot read: none, or fewer bytes than
    /// rows of its width.
    pub(super) fn check_audio(&self, audio: &Encoded) -> Result<(), AudioError> {
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
        Ok(())
    }

    /// `ws`'s slots as the activations of `n` prompt rows, with logits for
    /// `heads` rows.
    pub(super) fn acts<'w>(
        &self,
        n: usize,
        heads: usize,
        ws: &'w mut Workspace,
    ) -> Result<Acts<'w>, AudioError> {
        let c = self.config;
        let (qw, kw) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let (d, f) = (n * c.hidden, n * c.ffn);
        ws.reserve(
            self.ops,
            &[d, d, n * qw, n * kw, n * kw, n * qw, f, f, heads * c.vocab],
        )?;
        let ws = &*ws;
        Ok(Acts {
            x: ws.get(0),
            h: ws.get(1),
            q: ws.get(2),
            k: ws.get(3),
            v: ws.get(4),
            att: ws.get(5),
            gate: ws.get(6),
            up: ws.get(7),
            logits: ws.get(8),
        })
    }

    fn scratch<'w>(&self, n: usize, ws: &'w mut Workspace) -> Result<Scratch<'w>, AudioError> {
        let (c, ops) = (self.config, self.ops);
        Ok(Scratch {
            a: self.acts(n, 1, ws)?,
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
    pub(super) fn table(&self) -> Result<EmbedTable<'_>, AudioError> {
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

    /// The prompt's rows into `x` from row `row0`: each text id's embedding
    /// row, and the tower's rows over the placeholder run, in order and
    /// unscaled, as the reference merges them.
    pub(super) fn embed_prompt(
        &self,
        x: &DeviceBuffer,
        row0: usize,
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
            rows(x, row0 + first, audio.tokens, c.hidden)?,
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
                out: rows(x, row0 + i, 1, c.hidden)?,
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

    /// `linear` over each run of rows of `x` into the same rows of `y`, one
    /// GEMM a run: the tiled GEMM's pick of kernel and of k splits depends on
    /// the rows it is given, so a run is rows that are summed alike.
    fn project(
        &self,
        linear: &Linear,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        runs: &[Range<usize>],
    ) -> Result<(), AudioError> {
        let (k, n) = (linear.in_dim(), linear.out_dim());
        for run in runs {
            linear.forward_own(
                self.ops,
                rows(x, run.start, run.len(), k)?,
                rows(y, run.start, run.len(), n)?,
                run.len(),
            )?;
        }
        Ok(())
    }

    /// One prefill layer over the `n` prompt rows of `a.x`, in place: every
    /// projection on the audio set's tiled GEMM (an 8-bit affine weight
    /// too), one GEMM per run of `runs`, which cover `[0, n)`. `attend`
    /// runs between the projections and the output projection: it takes the
    /// projected `a.q`, `a.k` and `a.v`, fills the cache and leaves the
    /// attention's rows in `a.att`.
    pub(super) fn prefill_layer(
        &self,
        layer: &TextLayer,
        a: &Acts<'_>,
        runs: &[Range<usize>],
        n: usize,
        attend: impl FnOnce() -> Result<(), AudioError>,
    ) -> Result<(), AudioError> {
        let c = self.config;
        self.rms(a.h.view(), a.x.view(), &layer.attn_norm, n, c.hidden)?;
        self.project(&layer.q, a.h, a.q, runs)?;
        self.project(&layer.k, a.h, a.k, runs)?;
        self.project(&layer.v, a.h, a.v, runs)?;
        attend()?;
        self.project(&layer.o, a.att, a.h, runs)?;
        self.residual(a.x, a.h, n * c.hidden)?;
        self.rms(a.h.view(), a.x.view(), &layer.ffn_norm, n, c.hidden)?;
        self.project(&layer.gate, a.h, a.gate, runs)?;
        self.project(&layer.up, a.h, a.up, runs)?;
        self.ops.silu_mul(&SiluMulDesc {
            out: a.gate.view(),
            gate: a.gate.view(),
            up: a.up.view(),
            count: n * c.ffn,
        })?;
        self.project(&layer.down, a.gate, a.h, runs)?;
        self.residual(a.x, a.h, n * c.hidden)
    }

    /// One prefill layer of this decoder's one sequence: attention causal
    /// from position 0 over its own cache.
    fn layer(
        &self,
        layer: &TextLayer,
        cache: CachePlanes<'_>,
        s: &Scratch<'_>,
        n: usize,
    ) -> Result<(), AudioError> {
        let (ops, c, a) = (self.ops, self.config, &s.a);
        let hd = c.head_dim;
        let whole = 0..n;
        self.prefill_layer(layer, a, std::slice::from_ref(&whole), n, || {
            // The q and k norms are over each head's dims, before the rotation.
            self.rms(a.q.view(), a.q.view(), &layer.q_norm, n * c.heads, hd)?;
            self.rms(a.k.view(), a.k.view(), &layer.k_norm, n * c.kv_heads, hd)?;
            for (x, heads) in [(&a.q, c.heads), (&a.k, c.kv_heads)] {
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
                k: a.k.view(),
                v: a.v.view(),
                state: s.state.view(),
                rows: n,
            })?;
            Ok(ops.causal_attention(&CausalAttnDesc {
                out: a.att.view(),
                q: a.q.view(),
                cache,
                n_q: n,
                heads: c.heads,
                pos0: 0,
                scale: attention_scale(hd),
            })?)
        })
    }

    /// The final norm of `rows` head inputs into `a.h`, and the tied head's
    /// logits of each into `a.logits`. After a decode pass the last layer's
    /// residual is still in `a.h`, and is added first.
    pub(super) fn logits(
        &self,
        a: &Acts<'_>,
        count: usize,
        input: HeadInput<'_>,
    ) -> Result<(), AudioError> {
        let c = self.config;
        let h = rows(a.h, 0, count, c.hidden)?;
        match input {
            HeadInput::Rows(x) => self.rms(h, x, &self.weights.norm, count, c.hidden)?,
            HeadInput::Residual => self.add_rms(a, &self.weights.norm, count)?,
        }
        Ok(self
            .weights
            .embed
            .step(self.ops, h, rows(a.logits, 0, count, c.vocab)?, count)?)
    }

    /// The head of one row and its argmax into `s.ids`; a tie keeps the
    /// lowest id, as the reference's argmax does.
    fn head(&self, s: &Scratch<'_>, input: HeadInput<'_>) -> Result<(), AudioError> {
        self.logits(&s.a, 1, input)?;
        Ok(self.ops.argmax_wide(&ArgmaxWideDesc {
            out: s.ids.view(),
            input: s.a.logits.view(),
            parts: s.parts.view(),
            cols: self.config.vocab,
        })?)
    }

    /// `x += h`, then `h = rms_norm(x) * weight`, over a decode pass's
    /// `count` rows.
    fn add_rms(&self, a: &Acts<'_>, weight: &DeviceBuffer, count: usize) -> Result<(), AudioError> {
        let c = self.config;
        let h = rows(a.h, 0, count, c.hidden)?;
        Ok(self.ops.add_rms_norm(&AddRmsNormDesc {
            out: h,
            x: rows(a.x, 0, count, c.hidden)?,
            add: h,
            weight: Some(weight.view()),
            rows: count,
            dim: c.hidden,
            eps: c.rms_eps,
        })?)
    }

    /// One layer of a decode pass over the first `count` rows of `a.x`, in
    /// the fused launches. The layer's input is `x + h` - the residual the
    /// layer before left in `a.h` is added by this layer's first norm (the
    /// first layer's `h` is already normed) - and its own last residual is
    /// left in `a.h` the same way: the values are the ones
    /// [`Self::prefill_layer`] forms, in fewer launches. `attend` runs
    /// between the projections and the output projection, as there.
    ///
    /// One row takes the three projections in one launch; more rows take
    /// them one after another, each row summed as the one launch sums it.
    pub(super) fn step_layer(
        &self,
        layer: &TextLayer,
        a: &Acts<'_>,
        count: usize,
        first: bool,
        attend: impl FnOnce() -> Result<(), AudioError>,
    ) -> Result<(), AudioError> {
        let (ops, c) = (self.ops, self.config);
        if first {
            self.rms(
                rows(a.h, 0, count, c.hidden)?,
                rows(a.x, 0, count, c.hidden)?,
                &layer.attn_norm,
                count,
                c.hidden,
            )?;
        } else {
            self.add_rms(a, &layer.attn_norm, count)?;
        }
        if count == 1 {
            step3(
                ops,
                [&layer.q, &layer.k, &layer.v],
                a.h.view(),
                [a.q.view(), a.k.view(), a.v.view()],
            )?;
        } else {
            for (linear, y) in [(&layer.q, a.q), (&layer.k, a.k), (&layer.v, a.v)] {
                linear.step(ops, a.h.view(), y.view(), count)?;
            }
        }
        attend()?;
        layer.o.step(ops, a.att.view(), a.h.view(), count)?;
        self.add_rms(a, &layer.ffn_norm, count)?;
        step_silu(
            ops,
            [&layer.gate, &layer.up],
            a.h.view(),
            a.gate.view(),
            a.up.view(),
            count,
        )?;
        layer.down.step(ops, a.gate.view(), a.h.view(), count)?;
        Ok(())
    }

    /// One layer of this decoder's decode pass: the one row's attention over
    /// its own cache.
    fn layer_step(
        &self,
        layer: &TextLayer,
        cache: CachePlanes<'_>,
        s: &Scratch<'_>,
        first: bool,
    ) -> Result<(), AudioError> {
        let (ops, c, a) = (self.ops, self.config, &s.a);
        self.step_layer(layer, a, 1, first, || {
            ops.qk_norm_rope_append(&QkRopeAppendDesc {
                q: a.q.view(),
                k: a.k.view(),
                v: a.v.view(),
                q_norm: layer.q_norm.view(),
                k_norm: layer.k_norm.view(),
                cache,
                heads: c.heads,
                rows: 1,
                eps: c.rms_eps,
                theta: c.rope_theta,
                state: s.state.view(),
            })?;
            Ok(ops.decode_attention(&DecodeAttnDesc {
                out: a.att.view(),
                partials: s.partials.view(),
                splits: DECODE_ATTN_SPLITS,
                q: a.q.view(),
                cache,
                heads: c.heads,
                state: s.state.view(),
                scale: attention_scale(c.head_dim),
            })?)
        })
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
            out: s.a.x.view(),
            table: self.table()?,
            state: s.state.view(),
            width: c.hidden,
            vocab: c.vocab,
        })?;
        for (l, layer) in self.weights.layers.iter().enumerate() {
            self.layer_step(layer, cache.layer(l), s, l == 0)?;
        }
        self.head(s, HeadInput::Residual)
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
pub(crate) mod tests;

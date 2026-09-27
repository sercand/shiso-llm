//! The decoder over a batch of chunks at once: one stacked prefill, then
//! decode steps that take every live sequence's row through each weight in
//! one launch.
//!
//! Each chunk's ids are the ones [`Decoder::transcribe`] gives it alone. A
//! row of a batched launch is formed exactly as the single launch forms it:
//!
//! - the layers' per-row work is the single decoder's own
//!   (`Decoder::prefill_layer`, `Decoder::step_layer`,
//!   `Decoder::logits`); only the attention between the projections
//!   differs, and the batched attention kernels run the single kernels'
//!   bodies over each sequence's rows and plane;
//! - the prefill's GEMMs take the stacked rows in runs whose rows are summed
//!   as they would be alone (`gemm_rows_sum_alone`): the prompts longer
//!   than the 32-row tile side by side, a shorter one on its own;
//! - a decode step's projections take the batch's rows on the multi-row
//!   matrix-vector kernels, which read each weight run once for up to four
//!   rows and sum each row as the one-row kernels do, and its head argmaxes
//!   each row with the single path's wide argmax;
//! - the decode attention keeps the single path's 32 splits: fewer measured
//!   faster at eight sequences, but a split is a partial sum, and its count
//!   changes the float.
//!
//! **The cache.** The sequences' planes share one key and one value buffer
//! per layer (see [`SeqTable`] for why), laid out back to back, each sized
//! for its prompt and `HEADROOM` rows as a fresh `KvCache` is. Before
//! each batch of steps, a sequence whose position could pass its plane
//! grows it the single path's way - to what it needs or twice its capacity -
//! from the positions the last read-back gave. Growing re-lays out the whole
//! set into new buffers with a copy on the device: a plane whose capacity is
//! unchanged moves in one copy, a grown one a copy per head. A plane cannot
//! grow in place without leaving holes, and growth is rare - a transcript
//! of a full chunk is a few hundred tokens, inside the headroom - so the
//! simpler layout wins.
//!
//! **Finished sequences.** A sequence that ends stops appending and emits
//! nothing, but its row keeps flowing through the weights until the batch
//! compacts: once half or more of the sequences a batch runs are done, the
//! next batch of steps runs only the live ones, their states copied into a
//! fresh block and their planes left where they are (only the table changes).
//! Compacting costs a few small copies and saves each finished row's share
//! of every launch: eight mixed chunks decoded in 4.0 s compacting against
//! 4.7 s keeping every row (M1 Max, HF bf16, `batched_step_and_prefill_times`).
//!
//! A pass over eight sequences measured 37 ms against 12 ms for one (the
//! single path's pass): 4.7 ms a sequence. The prefill is compute-bound and
//! stacking gains nothing on it; it is one pass for the batch's rows so the
//! decode can start for all of them at once.

use std::ops::Range;
use std::time::{Duration, Instant};

use shiso_audio::AudioError;
use shiso_core::DeviceBuffer;

use super::Model;
use super::config::{STOP_TOKENS, TextConfig};
use super::decoder::{
    Acts, BATCH, Decoder, HEADROOM, HeadInput, MAX_NEW_TOKENS, POOLED_CAPACITY, attention_scale,
    bytes, count_of, rows, word, words,
};
use super::encoder::Encoded;
use super::prompt;
use super::weights::TextWeights;
use crate::audio_ops::{
    ArgmaxWideDesc, BatchPlanes, CausalAttnSegDesc, DECODE_ATTN_SPLITS, DECODE_COUNT, DECODE_DONE,
    DECODE_NO_TOKEN, DECODE_POS, DECODE_STATE_WORDS, DecodeAttnBatchDesc, DecodeStepBatchDesc,
    EmbedTokenBatchDesc, Ops, QkRopeAppendBatchDesc, QkRopeAppendSegDesc, SeqSpan, SeqTable,
    Workspace, argmax_wide_parts, decode_attention_partials, gemm_rows_sum_alone,
};

/// One chunk of a batch: the tower's rows of its audio.
pub type ChunkInput<'e> = &'e Encoded;

/// Batch-sized memories the model keeps: one. An offline job runs one batch
/// at a time, and a second set would be over a gigabyte held for a job that
/// is not running.
pub(crate) const POOL_BATCHES: usize = 1;

/// Where one batch's time went.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BatchTiming {
    /// Stacked prompt rows.
    pub(crate) prompt: usize,
    /// Embedding, the stacked prefill and the first argmaxes.
    pub(crate) prefill: Duration,
    /// Every batch of steps, read-backs and growth included.
    pub(crate) decode: Duration,
    /// Batched decode passes run, each over every sequence still in the
    /// batch.
    pub(crate) passes: usize,
    /// Tokens emitted over all sequences.
    pub(crate) tokens: usize,
}

/// A batch's key and value caches: per layer one key and one value buffer
/// holding every sequence's `[kv_heads][cap][head_dim]` plane back to back.
#[derive(Debug)]
pub(crate) struct BatchCache {
    /// `[k, v]` per layer; empty until the first batch sizes them.
    planes: Vec<[DeviceBuffer; 2]>,
    /// Rows of `head_dim` floats each buffer holds.
    rows: usize,
    /// Each sequence's plane in the current batch.
    pub(crate) spans: Vec<SeqSpan>,
    /// The most sequences one batch laid out: what the pool judges the
    /// buffers' size by.
    widest: usize,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Rows past each prompt a plane is first sized for.
    headroom: usize,
    /// Fill every fresh buffer with NaN, so a test sees which rows a batch
    /// wrote.
    #[cfg(test)]
    sentinel: bool,
    /// Re-layouts to grow a plane.
    #[cfg(test)]
    pub(crate) grown: usize,
    /// Rows each sequence appended, at the end of its batch.
    #[cfg(test)]
    pub(crate) ends: Vec<usize>,
}

impl BatchCache {
    fn new(config: &TextConfig) -> Self {
        Self {
            planes: Vec::new(),
            rows: 0,
            spans: Vec::new(),
            widest: 0,
            layers: config.layers,
            kv_heads: config.kv_heads,
            head_dim: config.head_dim,
            headroom: HEADROOM,
            #[cfg(test)]
            sentinel: false,
            #[cfg(test)]
            grown: 0,
            #[cfg(test)]
            ends: Vec::new(),
        }
    }

    /// Planes of `caps` back to back, from row 0.
    fn stack(&self, caps: &[usize]) -> (Vec<SeqSpan>, usize) {
        let mut plane = 0;
        let spans = caps
            .iter()
            .map(|&cap| {
                let span = SeqSpan {
                    plane,
                    cap,
                    ..SeqSpan::default()
                };
                plane += self.kv_heads * cap;
                span
            })
            .collect();
        (spans, plane)
    }

    /// Fresh buffers of `rows` rows for every layer. Rows past the filled
    /// ones are never read before they are written, so they are not
    /// cleared.
    fn alloc(&self, ops: &Ops, rows: usize) -> Result<Vec<[DeviceBuffer; 2]>, AudioError> {
        let floats = rows * self.head_dim;
        let mut planes = Vec::with_capacity(self.layers);
        for _ in 0..self.layers {
            planes.push([ops.scratch_uninit(floats)?, ops.scratch_uninit(floats)?]);
        }
        #[cfg(test)]
        if self.sentinel {
            Self::fill_nan(ops, &planes)?;
        }
        Ok(planes)
    }

    #[cfg(test)]
    fn fill_nan(ops: &Ops, planes: &[[DeviceBuffer; 2]]) -> Result<(), AudioError> {
        for buffer in planes.iter().flatten() {
            let len = usize::try_from(buffer.len()).expect("a host-sized buffer");
            let nan: Vec<u8> = f32::NAN
                .to_le_bytes()
                .into_iter()
                .cycle()
                .take(len)
                .collect();
            ops.backend().copy_h2d(buffer.view(), &nan)?;
        }
        Ok(())
    }

    /// Lays out a fresh batch of `prompts` rows each: every plane its
    /// prompt and the headroom, in the buffers already held when they are
    /// long enough.
    fn lay_out(&mut self, ops: &Ops, prompts: &[usize]) -> Result<(), AudioError> {
        let caps: Vec<usize> = prompts.iter().map(|&n| n + self.headroom).collect();
        let (spans, total) = self.stack(&caps);
        if total > self.rows || self.planes.is_empty() {
            // The old buffers go first, so the two sets are never held at
            // once.
            self.planes.clear();
            self.planes = self.alloc(ops, total)?;
            self.rows = total;
        } else {
            #[cfg(test)]
            if self.sentinel {
                Self::fill_nan(ops, &self.planes)?;
            }
        }
        self.spans = spans;
        self.widest = self.widest.max(prompts.len());
        Ok(())
    }

    /// Grows each plane whose `need` passes its capacity, keeping each
    /// sequence's first `filled` rows: to `need` or twice the capacity,
    /// whichever is more. Returns whether anything moved, so the caller
    /// rebuilds its table.
    fn grow(&mut self, ops: &Ops, filled: &[usize], need: &[usize]) -> Result<bool, AudioError> {
        let caps: Vec<usize> = self
            .spans
            .iter()
            .zip(need)
            .map(|(s, &n)| if n > s.cap { n.max(2 * s.cap) } else { s.cap })
            .collect();
        if caps.iter().zip(&self.spans).all(|(&c, s)| c == s.cap) {
            return Ok(false);
        }
        let (spans, total) = self.stack(&caps);
        let planes = self.alloc(ops, total)?;
        let (hd, backend) = (self.head_dim, ops.backend());
        for (new, old) in planes.iter().zip(&self.planes) {
            for (to, from) in new.iter().zip(old) {
                for ((a, b), &rows) in self.spans.iter().zip(&spans).zip(filled) {
                    if rows == 0 {
                        continue;
                    }
                    if a.cap == b.cap {
                        let plane = self.kv_heads * a.cap * hd;
                        backend.copy_d2d(
                            to.slice(bytes(b.plane * hd), bytes(plane))?,
                            from.slice(bytes(a.plane * hd), bytes(plane))?,
                        )?;
                        continue;
                    }
                    for h in 0..self.kv_heads {
                        backend.copy_d2d(
                            to.slice(bytes((b.plane + h * b.cap) * hd), bytes(rows * hd))?,
                            from.slice(bytes((a.plane + h * a.cap) * hd), bytes(rows * hd))?,
                        )?;
                    }
                }
            }
        }
        self.planes = planes;
        self.rows = total;
        self.spans = spans;
        #[cfg(test)]
        {
            self.grown += 1;
        }
        Ok(true)
    }

    /// Layer `layer`'s buffers under `table`.
    fn layer<'a>(&'a self, layer: usize, table: &'a SeqTable) -> BatchPlanes<'a> {
        let [k, v] = &self.planes[layer];
        BatchPlanes {
            k: k.view(),
            v: v.view(),
            table,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
        }
    }

    /// The decode table of sequences `live`, in that order.
    fn table(&self, ops: &Ops, live: &[usize]) -> Result<SeqTable, AudioError> {
        let spans: Vec<SeqSpan> = live
            .iter()
            .map(|&i| SeqSpan {
                plane: self.spans[i].plane,
                cap: self.spans[i].cap,
                ..SeqSpan::default()
            })
            .collect();
        Ok(SeqTable::new(ops, &spans, self.kv_heads)?)
    }

    /// Sequence `seq`'s keys (`which` 0) or values (1) of layer `layer`, as
    /// `[kv_heads][cap][head_dim]`.
    #[cfg(test)]
    pub(crate) fn plane(&self, ops: &Ops, layer: usize, seq: usize, which: usize) -> Vec<f32> {
        let s = self.spans[seq];
        let hd = self.head_dim;
        crate::audio_ops::load::download_f32(
            ops,
            self.planes[layer][which]
                .slice(bytes(s.plane * hd), bytes(self.kv_heads * s.cap * hd))
                .expect("in range"),
            self.kv_heads * s.cap * hd,
        )
        .expect("reads")
    }
}

/// A batch's device memory: the shared caches and the activations, all
/// written before they are read. A model keeps one for the next batch
/// ([`give_back`]), because a fresh buffer's first touch costs some 50 us a
/// megabyte (M1 Max) and a batch's is over a gigabyte.
#[derive(Debug)]
pub struct BatchMemory {
    pub(crate) cache: BatchCache,
    ws: Workspace,
}

impl BatchMemory {
    /// Empty memory for a decoder of `config`'s geometry; the first batch
    /// allocates it.
    #[must_use]
    pub fn new(config: &TextConfig) -> Self {
        Self {
            cache: BatchCache::new(config),
            ws: Workspace::new(),
        }
    }

    /// Rows of `head_dim` floats each of the cache's buffers holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.cache.rows
    }

    /// Memory that fills every fresh cache buffer with NaN.
    #[cfg(test)]
    pub(crate) fn with_sentinel(mut self) -> Self {
        self.cache.sentinel = true;
        self
    }

    /// Memory whose planes start at `headroom` rows past their prompts.
    #[cfg(test)]
    pub(crate) fn with_headroom(mut self, headroom: usize) -> Self {
        self.cache.headroom = headroom;
        self
    }

    /// Memory that says it laid out `seqs` sequences in `rows` rows and
    /// holds none, for the pool's bookkeeping: nothing may transcribe with
    /// it.
    #[cfg(test)]
    pub(crate) fn claiming(config: &TextConfig, seqs: usize, rows: usize) -> Self {
        let mut memory = Self::new(config);
        memory.cache.rows = rows;
        memory.cache.widest = seqs;
        memory
    }
}

/// Returns a finished batch's `memory` to `pool`, or drops it. A set whose
/// planes grew past [`POOLED_CAPACITY`] each held a degenerate chunk's rows,
/// up to some 1.3 GB a sequence at the token cap, and would hold them for the
/// life of the process; the pool keeps [`POOL_BATCHES`].
pub(crate) fn give_back(pool: &mut Option<BatchMemory>, memory: BatchMemory) {
    let c = &memory.cache;
    if c.rows <= c.widest * c.kv_heads * POOLED_CAPACITY
        && usize::from(pool.is_some()) < POOL_BATCHES
    {
        *pool = Some(memory);
    }
}

/// The per-launch scratch of one batch, beside the activations.
struct Scratch {
    /// Each sequence's argmax, `[seqs]` `u32`.
    ids: DeviceBuffer,
    /// Each row's wide-argmax candidates, `[seqs][parts]`.
    parts: DeviceBuffer,
    partials: DeviceBuffer,
    /// `u32`, `[BATCH][seqs]`.
    out: DeviceBuffer,
}

/// The batched decoder over one loaded checkpoint.
#[derive(Debug)]
pub struct BatchTranscriber<'a> {
    decoder: Decoder<'a>,
    ops: &'a Ops,
    weights: &'a TextWeights,
    config: &'a TextConfig,
    /// Run only the live sequences once half a batch is done.
    compact: bool,
}

impl<'a> BatchTranscriber<'a> {
    /// Binds the batched decoder to its weights.
    #[must_use]
    pub fn new(ops: &'a Ops, weights: &'a TextWeights, config: &'a TextConfig) -> Self {
        Self {
            decoder: Decoder::new(ops, weights, config),
            ops,
            weights,
            config,
            compact: true,
        }
    }

    /// The transcriber that compacts (the default) or keeps finished rows in
    /// place, to measure the one against the other.
    #[cfg(test)]
    pub(crate) fn compacting(mut self, compact: bool) -> Self {
        self.compact = compact;
        self
    }

    /// Each chunk's greedy ids, in order: what [`Decoder::transcribe`] gives
    /// that chunk alone with the same `forced`.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] for no chunks, a chunk the single decoder
    /// refuses, or a forced language the model does not support;
    /// [`AudioError::Checkpoint`] for an embedding no row fetch reads; a
    /// device error.
    pub fn transcribe(
        &self,
        chunks: &[ChunkInput<'_>],
        forced: Option<&str>,
    ) -> Result<Vec<Vec<u32>>, AudioError> {
        self.transcribe_with(&mut BatchMemory::new(self.config), chunks, forced)
    }

    /// [`Self::transcribe`] in `memory`, which a model keeps from one batch
    /// to the next.
    ///
    /// # Errors
    ///
    /// As [`Self::transcribe`].
    pub fn transcribe_with(
        &self,
        memory: &mut BatchMemory,
        chunks: &[ChunkInput<'_>],
        forced: Option<&str>,
    ) -> Result<Vec<Vec<u32>>, AudioError> {
        let (ids, t) = self.timed_with(memory, chunks, forced)?;
        tracing::debug!(
            chunks = chunks.len(),
            prompt = t.prompt,
            prefill_ms = t.prefill.as_secs_f64() * 1e3,
            decode_ms = t.decode.as_secs_f64() * 1e3,
            passes = t.passes,
            tokens = t.tokens,
            "transcribed a batch"
        );
        Ok(ids)
    }

    /// [`Self::transcribe_with`], with where the time went.
    pub(crate) fn timed_with(
        &self,
        memory: &mut BatchMemory,
        chunks: &[ChunkInput<'_>],
        forced: Option<&str>,
    ) -> Result<(Vec<Vec<u32>>, BatchTiming), AudioError> {
        let (ops, c) = (self.ops, self.config);
        if chunks.is_empty() {
            return Err(AudioError::BadSpec {
                reason: "a batch of no chunks".into(),
            });
        }
        let mut prompts = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            self.decoder.check_audio(chunk)?;
            prompts.push(prompt::ids(chunk.tokens, forced)?);
        }
        let ns: Vec<usize> = prompts.iter().map(Vec::len).collect();
        let (b, total) = (ns.len(), ns.iter().sum::<usize>());
        let start = Instant::now();
        let BatchMemory { cache, ws } = memory;
        let a = self.decoder.acts(total, b, ws)?;
        let s = Scratch {
            ids: ops.scratch(b)?,
            parts: ops.scratch(b * argmax_wide_parts(c.vocab))?,
            partials: ops.scratch(decode_attention_partials(
                b * c.heads,
                c.head_dim,
                DECODE_ATTN_SPLITS,
            ))?,
            out: ops.scratch(BATCH * b)?,
        };
        cache.lay_out(ops, &ns)?;
        let mut row0 = Vec::with_capacity(b);
        let mut spans = cache.spans.clone();
        let mut at = 0;
        for (i, (span, &n)) in spans.iter_mut().zip(&ns).enumerate() {
            self.decoder.embed_prompt(a.x, at, &prompts[i], chunks[i])?;
            span.row0 = at;
            span.rows = n;
            row0.push(at);
            at += n;
        }
        let table = SeqTable::new(ops, &spans, c.kv_heads)?;
        self.prefill(cache, &a, &table, &ns)?;
        // Each sequence's last prompt row, gathered for the head.
        for (i, (&r, &n)) in row0.iter().zip(&ns).enumerate() {
            ops.backend().copy_d2d(
                rows(a.gate, i, 1, c.hidden)?,
                rows(a.x, r + n - 1, 1, c.hidden)?,
            )?;
        }
        self.head(&a, &s, b, HeadInput::Rows(rows(a.gate, 0, b, c.hidden)?))?;
        // The handoff, as the single path's: each state at its prompt's end.
        let mut raw = vec![0u8; b * DECODE_STATE_WORDS * 4];
        for (i, &n) in ns.iter().enumerate() {
            let at = (i * DECODE_STATE_WORDS + DECODE_POS) * 4;
            raw[at..at + 4].copy_from_slice(&word(n, "a prompt of")?.to_le_bytes());
        }
        let states = ops.scratch_uninit(b * DECODE_STATE_WORDS)?;
        ops.backend().copy_h2d(states.view(), &raw)?;
        let prefill = start.elapsed();
        let (ids, passes) = self.decode(cache, &a, &s, states, &ns)?;
        let timing = BatchTiming {
            prompt: total,
            prefill,
            decode: start.elapsed().saturating_sub(prefill),
            passes,
            tokens: ids.iter().map(Vec::len).sum(),
        };
        Ok((ids, timing))
    }

    /// The stacked prefill: every layer over all `ns` prompts' rows, each
    /// attending over its own plane.
    fn prefill(
        &self,
        cache: &BatchCache,
        a: &Acts<'_>,
        table: &SeqTable,
        ns: &[usize],
    ) -> Result<(), AudioError> {
        let (ops, c) = (self.ops, self.config);
        let total = ns.iter().sum();
        let runs = gemm_runs(ns);
        for (l, layer) in self.weights.layers.iter().enumerate() {
            let planes = cache.layer(l, table);
            self.decoder.prefill_layer(layer, a, &runs, total, || {
                ops.qk_norm_rope_append_seg(&QkRopeAppendSegDesc {
                    q: a.q.view(),
                    k: a.k.view(),
                    v: a.v.view(),
                    q_norm: layer.q_norm.view(),
                    k_norm: layer.k_norm.view(),
                    cache: planes,
                    heads: c.heads,
                    rows: total,
                    eps: c.rms_eps,
                    theta: c.rope_theta,
                })?;
                Ok(ops.causal_attention_seg(&CausalAttnSegDesc {
                    out: a.att.view(),
                    q: a.q.view(),
                    cache: planes,
                    heads: c.heads,
                    scale: attention_scale(c.head_dim),
                })?)
            })?;
        }
        Ok(())
    }

    /// The head of `count` rows and each row's argmax into `s.ids`, by the
    /// single path's wide argmax.
    fn head(
        &self,
        a: &Acts<'_>,
        s: &Scratch,
        count: usize,
        input: HeadInput<'_>,
    ) -> Result<(), AudioError> {
        let c = self.config;
        self.decoder.logits(a, count, input)?;
        let parts = argmax_wide_parts(c.vocab);
        for r in 0..count {
            self.ops.argmax_wide(&ArgmaxWideDesc {
                out: s.ids.slice(bytes(r), bytes(1))?,
                input: rows(a.logits, r, 1, c.vocab)?,
                parts: rows(&s.parts, r, 1, parts)?,
                cols: c.vocab,
            })?;
        }
        Ok(())
    }

    /// One decode pass of the `table`'s sequences, ending in each one's
    /// argmax.
    fn pass(
        &self,
        cache: &BatchCache,
        a: &Acts<'_>,
        s: &Scratch,
        table: &SeqTable,
        states: &DeviceBuffer,
    ) -> Result<(), AudioError> {
        let (ops, c) = (self.ops, self.config);
        let seqs = table.len();
        ops.embed_token_batch(&EmbedTokenBatchDesc {
            out: a.x.view(),
            table: self.decoder.table()?,
            states: states.view(),
            width: c.hidden,
            vocab: c.vocab,
            seqs,
        })?;
        for (l, layer) in self.weights.layers.iter().enumerate() {
            let planes = cache.layer(l, table);
            self.decoder.step_layer(layer, a, seqs, l == 0, || {
                ops.qk_norm_rope_append_batch(&QkRopeAppendBatchDesc {
                    q: a.q.view(),
                    k: a.k.view(),
                    v: a.v.view(),
                    q_norm: layer.q_norm.view(),
                    k_norm: layer.k_norm.view(),
                    cache: planes,
                    heads: c.heads,
                    eps: c.rms_eps,
                    theta: c.rope_theta,
                    states: states.view(),
                })?;
                Ok(ops.decode_attention_batch(&DecodeAttnBatchDesc {
                    out: a.att.view(),
                    partials: s.partials.view(),
                    splits: DECODE_ATTN_SPLITS,
                    q: a.q.view(),
                    cache: planes,
                    heads: c.heads,
                    states: states.view(),
                    scale: attention_scale(c.head_dim),
                })?)
            })?;
        }
        self.head(a, s, seqs, HeadInput::Residual)
    }

    fn step(
        &self,
        s: &Scratch,
        states: &DeviceBuffer,
        (seqs, advance, slot): (usize, usize, usize),
    ) -> Result<(), AudioError> {
        Ok(self.ops.decode_step_batch(&DecodeStepBatchDesc {
            states: states.view(),
            out: s.out.view(),
            ids: s.ids.view(),
            eos: STOP_TOKENS,
            max_tokens: MAX_NEW_TOKENS,
            advance,
            slot,
            seqs,
        })?)
    }

    /// Greedy steps from the prefill's argmaxes until every state says done;
    /// returns each sequence's emitted ids and the batched passes run.
    fn decode(
        &self,
        cache: &mut BatchCache,
        a: &Acts<'_>,
        s: &Scratch,
        mut states: DeviceBuffer,
        prompts: &[usize],
    ) -> Result<(Vec<Vec<u32>>, usize), AudioError> {
        let (ops, backend) = (self.ops, self.ops.backend());
        let b = prompts.len();
        let mut tokens = vec![Vec::new(); b];
        // Per sequence: its position, tokens emitted, and whether it is done.
        let mut pos = prompts.to_vec();
        let mut count = vec![0usize; b];
        let mut done = vec![false; b];
        // The sequences the states and the table run, in their order.
        let mut live: Vec<usize> = (0..b).collect();
        let mut table = cache.table(ops, &live)?;
        let (mut passes, mut first) = (0usize, true);
        loop {
            let seqs = live.len();
            // Past the cap a step emits nothing, so no batch runs more
            // passes than its longest-running sequence has tokens left.
            let room = live
                .iter()
                .map(|&i| MAX_NEW_TOKENS.saturating_sub(count[i]))
                .max()
                .unwrap_or(0);
            let run = if first { BATCH - 1 } else { BATCH }.min(room);
            // Each pass appends one row at a live state's position, which
            // only the steps move; the append drops rows past a plane, so
            // the planes grow before the batch is queued.
            let need: Vec<usize> = (0..b)
                .map(|i| if done[i] { 0 } else { pos[i] + run })
                .collect();
            if cache.grow(ops, &pos, &need)? {
                table = cache.table(ops, &live)?;
            }
            let mut slot = 0;
            if first {
                self.step(s, &states, (seqs, 0, 0))?;
                slot = 1;
            }
            for _ in 0..run {
                self.pass(cache, a, s, &table, &states)?;
                self.step(s, &states, (seqs, 1, slot))?;
                slot += 1;
            }
            passes += run;
            let mut host = vec![0u8; 4 * slot * seqs];
            backend.copy_d2h(s.out.slice(0, bytes(slot * seqs))?, &mut host)?;
            let out = words(&host);
            for t in 0..slot {
                for (j, &i) in live.iter().enumerate() {
                    let id = out[t * seqs + j];
                    if id != DECODE_NO_TOKEN {
                        tokens[i].push(id);
                    }
                }
            }
            let mut host = vec![0u8; 4 * DECODE_STATE_WORDS * seqs];
            backend.copy_d2h(
                states.slice(0, bytes(DECODE_STATE_WORDS * seqs))?,
                &mut host,
            )?;
            let st = words(&host);
            let mut moved = false;
            for (j, &i) in live.iter().enumerate() {
                let w = &st[j * DECODE_STATE_WORDS..(j + 1) * DECODE_STATE_WORDS];
                pos[i] = count_of(w[DECODE_POS]);
                let now = count_of(w[DECODE_COUNT]);
                moved |= now != count[i];
                count[i] = now;
                done[i] = w[DECODE_DONE] != 0;
            }
            if live.iter().all(|&i| done[i]) {
                break;
            }
            if !moved && !first {
                return Err(AudioError::BadSpec {
                    reason: format!(
                        "a batch of {run} decode steps over {seqs} sequences emitted nothing and did not stop"
                    ),
                });
            }
            first = false;
            let still: Vec<usize> = (0..seqs).filter(|&j| !done[live[j]]).collect();
            if self.compact && 2 * still.len() <= seqs {
                states = self.compacted(&states, &still)?;
                live = still.iter().map(|&j| live[j]).collect();
                table = cache.table(ops, &live)?;
            }
        }
        #[cfg(test)]
        {
            cache.ends = pos;
        }
        Ok((tokens, passes))
    }

    /// The states of the batch's sequences `keep`, in that order, in a
    /// fresh block: the planes stay where they are.
    fn compacted(&self, states: &DeviceBuffer, keep: &[usize]) -> Result<DeviceBuffer, AudioError> {
        let fresh = self.ops.scratch_uninit(keep.len() * DECODE_STATE_WORDS)?;
        for (to, &from) in keep.iter().enumerate() {
            self.ops.backend().copy_d2d(
                rows(&fresh, to, 1, DECODE_STATE_WORDS)?,
                rows(states, from, 1, DECODE_STATE_WORDS)?,
            )?;
        }
        Ok(fresh)
    }
}

/// The stacked prompts of `ns` rows each as runs of rows the GEMM sums
/// alike: prompts longer than the tile side by side, a shorter one alone.
fn gemm_runs(ns: &[usize]) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::new();
    let mut at = 0;
    let mut joinable = false;
    for &n in ns {
        let alone = gemm_rows_sum_alone(n);
        match runs.last_mut() {
            Some(last) if joinable && alone => last.end = at + n,
            _ => runs.push(at..at + n),
        }
        joinable = alone;
        at += n;
    }
    runs
}

impl Model {
    /// [`BatchTranscriber::transcribe`] in the model's pooled batch memory,
    /// returned to the pool afterwards under [`give_back`]'s rule.
    ///
    /// # Errors
    ///
    /// As [`BatchTranscriber::transcribe`].
    pub(crate) fn transcribe_batch(
        &self,
        chunks: &[ChunkInput<'_>],
        forced: Option<&str>,
    ) -> Result<Vec<Vec<u32>>, AudioError> {
        let pooled = self
            .batch_spare
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let mut memory = pooled.unwrap_or_else(|| BatchMemory::new(&self.config.text));
        let out = BatchTranscriber::new(&self.ops, &self.weights.text, &self.config.text)
            .transcribe_with(&mut memory, chunks, forced);
        give_back(
            &mut self
                .batch_spare
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            memory,
        );
        out
    }
}

#[cfg(test)]
mod runs {
    use super::gemm_runs;

    #[test]
    fn long_prompts_share_a_gemm_and_short_ones_run_alone() {
        assert_eq!(gemm_runs(&[405]), vec![0..405]);
        assert_eq!(gemm_runs(&[405, 159, 549]), vec![0..1113]);
        // 20 and 32 rows would be summed apart from how they are alone.
        assert_eq!(
            gemm_runs(&[405, 20, 159, 33, 32, 32]),
            [0..405, 405..425, 425..617, 617..649, 649..681]
        );
    }
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;

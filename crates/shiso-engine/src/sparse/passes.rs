//! Deciding what a pass is made of: the two entry points, and the one body
//! they share.
//!
//! It is a second `impl` block on [`SparseStack`], split out for the per-file
//! line cap exactly as [`crate::sparse::ckpt`] and [`crate::sparse::run`] are,
//! rather than a layer above it. [`crate::sparse::stack`] owns everything that
//! outlives a launch; this decides what one launch sequence is; and
//! [`crate::sparse::run`] puts a whole chunk in order around it.
//!
//! # One sequence's rows, and one row per sequence
//!
//! [`SparseStack::plan_chunk`] plans consecutive positions of one session -
//! a five-hundred row prompt or a single decode step, it is the same shape of
//! work. [`SparseStack::plan_batch`] plans one row per session, from sessions
//! that have nothing to do with each other, and that is a different pass in
//! three places:
//!
//! - **Per-session state.** Six kernels carry it through a launch: the
//!   recurrent convolution and recurrence, the n-gram convolution, the
//!   key/value store, the dense attention and the indexer's key store. Each has
//!   a batched sibling taking a device array of one pointer per row, which
//!   [`crate::sparse::rows`] packs. A batched pass dispatches those siblings; a
//!   pass that dispatched the scalar ones over one session's page table would
//!   read another sequence's context as this row's and decode it to fluent
//!   text.
//! - **The pooled indexer key pass.** It has no batched sibling: its grid is
//!   derived from the first and last position of the chunk, which for unrelated
//!   rows is not a range at all. So it is planned once per row, over that row's
//!   own page table, its own position slot and its own rotary lookback. One
//!   block per row, on a launch that is a memory move - the cost of the launch
//!   is the launch.
//! - **The selection.** There is no batched selected-block attention kernel in
//!   this set, and the batched attention entry is the dense one - the kernel
//!   set's own note beside it says so. So a batched pass runs the indexer's
//!   *storing* half, which keeps every row's key planes current for the scalar
//!   passes that follow, and takes the dense batched attention rather than a
//!   selection. That is more work than a scalar decode does at a long context,
//!   never less, and it is the one place where a row's answer depends on the
//!   width of the batch it was decoded in.
//!
//! # And the third thing: a batched island is never captured
//!
//! See [`IslandContext`]. A capture bakes in the addresses its launches were
//! encoded with, and for a batched island those are the page tables of the
//! exact sessions in the batch, in the exact row order. No field of an
//! [`IslandKey`] can name an ordered set of eight session handles, and hashing
//! one into the twenty-four bits the slot has would alias two different batches
//! onto one slot with nothing to notice it. So a batched island carries
//! [`IslandContext::Batch`], which has no slot at all.

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, SessionId, TokenId};

use crate::sparse::bind::{LayerRef, Operand, Role, WeightRef};
use crate::sparse::gather;
use crate::sparse::hyper::{self, ChainLink, MixRoles, NextNorm};
use crate::sparse::island::{self, Island, IslandContext, IslandKey, LayerChain};
use crate::sparse::kernels::{Elem, F32_BYTES};
pub use crate::sparse::matmul::PassRole;
use crate::sparse::matmul::{self, Bound, Device};
use crate::sparse::media;
use crate::sparse::moe::{self, RowOrigin};
use crate::sparse::plan::{Plan, Step};
use crate::sparse::ple::{self, NgramContext, PleGather};
use crate::sparse::qsa::{self, AttnContext, AttnSlot};
use crate::sparse::recurrent::{self, Recurrence};
use crate::sparse::scratch::Buf;
use crate::sparse::stack::{IslandSpan, MAX_DECODE_BATCH, Pass, SparseSession, SparseStack};
use crate::sparse::weights::BlockDesc;

impl SparseStack {
    /// Plans one pass over `tokens` for one session.
    ///
    /// `logit_rows` is how many of the chunk's final rows the vocabulary
    /// projection runs on. Zero means a non-final prefill chunk, which produces
    /// no logits and therefore does not run the collapsing mix at all - that
    /// projection is the single widest matmul in the model and running it on
    /// rows nobody reads is most of a prefill chunk's cost.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::ShapeMismatch`] for an empty
    /// or over-wide chunk, and whatever the block families report for a weight
    /// this build cannot project.
    pub fn plan_chunk(
        &self,
        session: SessionId,
        tokens: &[TokenId],
        logit_rows: u32,
    ) -> CoreResult<Pass> {
        self.plan_chunk_media(session, tokens, logit_rows, None, PassRole::Step)
    }

    /// Plans one prompt chunk: [`SparseStack::plan_chunk`] with every row
    /// computed as a pass of any other width would compute it.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::plan_chunk`].
    pub fn plan_prompt(
        &self,
        session: SessionId,
        tokens: &[TokenId],
        logit_rows: u32,
    ) -> CoreResult<Pass> {
        self.plan_chunk_media(
            session,
            tokens,
            logit_rows,
            None,
            PassRole::Prompt { snapshot: None },
        )
    }

    /// The same pass, with the splice that a chunk carrying a picture needs.
    ///
    /// `media_rows` is the height of the WHOLE request's encoder block, not
    /// this chunk's share of it: the splice map the chunk uploads indexes that
    /// block, and a prompt split across several passes splices out of the one
    /// block every time.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::plan_chunk`].
    pub fn plan_chunk_media(
        &self,
        session: SessionId,
        tokens: &[TokenId],
        logit_rows: u32,
        media_rows: Option<u32>,
        role: PassRole,
    ) -> CoreResult<Pass> {
        let live = self.session(session)?;
        let n_tok = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if n_tok == 0 || n_tok > self.opts.prefill_rows {
            return Err(CoreError::ShapeMismatch {
                field: "chunk rows",
                expected: u64::from(self.opts.prefill_rows),
                got: u64::from(n_tok),
            });
        }
        if logit_rows > n_tok {
            return Err(CoreError::ShapeMismatch {
                field: "logit rows",
                expected: u64::from(n_tok),
                got: u64::from(logit_rows),
            });
        }
        // One session's consecutive positions, whether that is a five-hundred
        // row prompt or a single decode step. What makes rows *unrelated* is
        // that they belong to different sessions, which is the other entry
        // point below.
        self.plan_rows(
            &[live],
            n_tok,
            logit_rows,
            RowOrigin::Sequential,
            media_rows,
            role,
        )
    }

    /// Plans one pass over one row per session, which is what a decode batch is.
    ///
    /// The rows are unrelated sequences rather than one sequence's consecutive
    /// positions, and this module's notes say what that changes.
    ///
    /// A batch of one row is planned as *sequential*, not as a one-row batch:
    /// what makes rows unrelated is that they belong to different sequences, and
    /// one row belongs to one. Planning it as unrelated would take the batched
    /// kernels where the scalar ones are exactly right, and would pack a
    /// one-slot pointer array to say what the buffer itself already says.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::plan_chunk`], plus [`CoreError::Unsupported`] for an
    /// empty batch or one wider than the logits block, and
    /// [`CoreError::InvalidValue`] when one session appears in two rows.
    pub fn plan_batch(&self, sessions: &[SessionId]) -> CoreResult<Pass> {
        let rows = u32::try_from(sessions.len()).unwrap_or(u32::MAX);
        if rows == 0 || rows > MAX_DECODE_BATCH {
            return Err(CoreError::Unsupported {
                what: "a decode batch wider than the logits block",
                backend: "engine",
            });
        }
        let live = sessions
            .iter()
            .map(|id| self.session(*id))
            .collect::<CoreResult<Vec<_>>>()?;
        let origin = if rows == 1 {
            RowOrigin::Sequential
        } else {
            crate::sparse::stack::distinct(sessions)?;
            RowOrigin::Unrelated
        };
        // A decode batch reads one token per session and embeds no picture: a
        // request's media is spliced during its prefill and is in the cache by
        // the time any of its rows decode.
        self.plan_rows(&live, rows, rows, origin, None, PassRole::Step)
    }

    /// Plans one segmented pass: each session's run of consecutive positions,
    /// in session order, all of them in one pass over the weights.
    ///
    /// `lens[i]` is how many rows session `i` runs. Every row produces logits,
    /// because a segmented pass is a batch's speculative verify and what comes
    /// back is every row's greedy pick.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::ShapeMismatch`] for runs
    /// that do not pair with the sessions or a pass wider than the logits
    /// block, [`CoreError::InvalidValue`] when one session runs twice, and
    /// whatever the block families report.
    pub fn plan_segments(&self, sessions: &[SessionId], lens: &[u32]) -> CoreResult<Pass> {
        if sessions.len() != lens.len() {
            return Err(CoreError::ShapeMismatch {
                field: "segmented pass runs",
                expected: sessions.len() as u64,
                got: lens.len() as u64,
            });
        }
        let runs = moe::Segments::new(lens)?;
        let rows = runs.rows();
        if rows > crate::sparse::stack::MAX_LOGIT_ROWS {
            return Err(CoreError::ShapeMismatch {
                field: "segmented pass rows",
                expected: u64::from(crate::sparse::stack::MAX_LOGIT_ROWS),
                got: u64::from(rows),
            });
        }
        crate::sparse::stack::distinct(sessions)?;
        let live = sessions
            .iter()
            .map(|id| self.session(*id))
            .collect::<CoreResult<Vec<_>>>()?;
        self.plan_rows(
            &live,
            rows,
            rows,
            RowOrigin::Segmented(runs),
            None,
            PassRole::Step,
        )
    }

    /// The pass itself.
    #[expect(
        clippy::too_many_lines,
        reason = "the pass is planned in the order it runs - gathers, islands, \
                  logits - and each stage reads what the one before it chose"
    )]
    fn plan_rows(
        &self,
        rows: &[&SparseSession],
        n_tok: u32,
        logit_rows: u32,
        origin: RowOrigin,
        media_rows: Option<u32>,
        role: PassRole,
    ) -> CoreResult<Pass> {
        // A torn-down stack has no scratch to plan against, and the plan would
        // describe launches over buffers that no longer exist.
        self.resident()?;
        let first = rows.first().ok_or(CoreError::ShapeMismatch {
            field: "pass rows",
            expected: 1,
            got: 0,
        })?;
        let batched = origin != RowOrigin::Sequential;
        // Only one sequence's prompt rows are promised to every pass width: a
        // batch's rows are steps by construction.
        let role = if batched { PassRole::Step } else { role };
        // One sequence's frontier decides whether its selection is live. A batch
        // has one frontier per row, and each row answers for itself: its
        // attention is planned as a single-row step of its own sequence, over
        // its own selection when its own context is past the budget.
        let selected = !batched
            && self
                .shape
                .indexer
                .sparse_at(first.written().saturating_add(n_tok));
        // A segmented pass's run `r` is session `r`'s, and covers `len(r)` rows
        // past its frontier; a batch's row `r` covers one.
        let run_rows = |row: u32| match origin {
            RowOrigin::Segmented(runs) => runs.len(row),
            RowOrigin::Sequential | RowOrigin::Unrelated => 1,
        };
        let batch_selected = if batched {
            rows.iter()
                .enumerate()
                .filter(|(row, live)| {
                    let row = u32::try_from(*row).unwrap_or(u32::MAX);
                    self.shape
                        .indexer
                        .sparse_at(live.written().saturating_add(run_rows(row)))
                })
                .fold(0u8, |bits, (row, _)| bits | (1u8 << (row % 8)))
        } else {
            0
        };
        // Rows at the front of a run whose own single-row step would still
        // attend densely: the row at `written + i` is past the budget exactly
        // when a step from that frontier is.
        let within = |written: u32, rows: u32| {
            (0..rows)
                .take_while(|&i| {
                    !self
                        .shape
                        .indexer
                        .sparse_at(written.saturating_add(i).saturating_add(1))
                })
                .count()
        };
        let dense_rows = if batched {
            0
        } else {
            u32::try_from(within(first.written(), n_tok)).unwrap_or(n_tok)
        };
        let mut run_dense_rows = [0u8; MAX_DECODE_BATCH as usize];
        if let RowOrigin::Segmented(runs) = origin {
            for ((run, live), slot) in rows.iter().enumerate().zip(&mut run_dense_rows) {
                let run = u32::try_from(run).unwrap_or(u32::MAX);
                *slot = u8::try_from(within(live.written(), runs.len(run))).unwrap_or(u8::MAX);
            }
        }
        let ctx = AttnContext {
            selected,
            tensor_core: self.opts.tensor_core,
            tiled: self.opts.tiled_attention,
            origin,
            batch_selected,
            dense_rows,
            run_dense_rows,
            role,
        };
        // A pass that crosses the budget is planned with its attention split
        // at the crossing, and the island key does not say where that is: a
        // capture of it replayed a step later would split rows that are all
        // past the budget by then. It happens once per sequence, so it is
        // simply never captured.
        let crosses = selected && dense_rows > 0;
        // A prompt chunk is never captured either: a capture may not grow the
        // scratch, so a captured pass falls back to the per-row tiers, and the
        // widest-pass tiers are the whole of what makes its rows invariant.
        let capturable = !crosses
            && role == PassRole::Step
            && if batched {
                self.opts.graphs.capture_batch
            } else {
                self.opts.graphs.capturable(n_tok)
            };
        let device = Device {
            capturing: capturable,
            role,
            ..self.device
        };
        let device = Device {
            half_block_input: self.stream_elem(n_tok, device) == Elem::F16,
            ..device
        };
        let context = if batched {
            IslandContext::Batch
        } else {
            IslandContext::Session(context_id(first.id()))
        };

        let mut plan = Plan::new();
        plan.append(gather::rows(
            &self.shape,
            self.desc.token_embd.dtype,
            gather::Bound::new(
                Operand::Scratch(Buf::Ids),
                Operand::Weight(WeightRef::model(Role::TokenEmbd)),
                Operand::Scratch(Buf::Embed),
            ),
            n_tok,
        )?);
        // Between the gather and the broadcast, and it has to be exactly here:
        // before it the placeholder rows are the embeddings of the placeholder
        // ids, and after the broadcast there are four copies of each to fix
        // instead of one.
        if media_rows.is_some() {
            plan.append(media::splice(n_tok));
        }
        plan.append(hyper::init(&self.shape, n_tok));

        let mut islands = Vec::with_capacity(self.desc.layers.len() * 2);
        for (index, layer) in self.desc.layers.iter().enumerate() {
            let il = LayerIdx::new(u32::try_from(index).unwrap_or(u32::MAX));
            let chain = island::chain(&self.shape, il);
            for half in Island::ORDER {
                let start = plan.len();
                let body = self.plan_island(
                    LayerRef::target(il.get()),
                    layer,
                    il,
                    half,
                    chain,
                    ctx,
                    origin,
                    n_tok,
                    device,
                )?;
                plan.append(body);
                islands.push(IslandSpan {
                    key: IslandKey {
                        layer: il,
                        island: half,
                        rows: n_tok,
                        selected,
                        context,
                    },
                    start,
                    len: plan.len().saturating_sub(start),
                });
            }
        }

        // A segmented pass is a verify, and a verify's output is every row's
        // greedy pick rather than the rows themselves.
        self.plan_logits(
            &mut plan,
            n_tok,
            logit_rows,
            origin == RowOrigin::Unrelated,
            device,
        )?;

        let gather = self.plan_gathers(rows, n_tok, origin)?;
        // The per-stream scratch every cooperative launch in the pass reads is
        // reserved once, in front of the first island, at the widest carve-up
        // any of them makes. It cannot be reserved where it is needed: growing
        // it moves an address, every cooperative launch in a layer sits inside
        // an island that may be captured, and a reservation attempted mid
        // capture is refused by the device. Reserving the maximum up front is
        // what makes the offsets the launches were planned with the offsets
        // they are issued with.
        let (plan, mut islands) = with_reservation(plan, islands);
        plan.validate()?;
        if batched && capturable {
            let slot = self.batch_slot(rows, &plan)?;
            for span in &mut islands {
                span.key.context = IslandContext::Batched(slot);
            }
        }
        Ok(Pass {
            media_rows,
            plan,
            islands,
            capturable,
            selected,
            rows: n_tok,
            origin,
            gather,
        })
    }

    /// The collapsing mix, the vocabulary projection and the greedy pick.
    ///
    /// Split out of the body above for the per-function line cap, and it splits
    /// cleanly: nothing here reads a session. `logit_rows` of zero is a
    /// non-final prefill chunk, which produces no logits and therefore does not
    /// run the collapsing mix at all - that projection is the single widest
    /// matmul in the model and running it on rows nobody reads is most of a
    /// prefill chunk's cost.
    fn plan_logits(
        &self,
        plan: &mut Plan,
        n_tok: u32,
        logit_rows: u32,
        batched: bool,
        device: Device,
    ) -> CoreResult<()> {
        if logit_rows > 0 {
            plan.append(hyper::collapse(
                &self.shape,
                &self.desc.output_mix,
                MixRoles::output(),
                n_tok,
                device,
                self.stream_elem(n_tok, device),
            )?);
            // The rows that get sampled are the chunk's *last* ones, so the
            // widest matmul in the model starts where they do rather than at
            // the front of the collapsed block. A projection from offset zero
            // would run on the chunk's first rows: the right count, the right
            // width, and the wrong tokens - which reads as a model that answers
            // fluently and ignores the end of its own prompt.
            let last = u64::from(n_tok.saturating_sub(logit_rows))
                .saturating_mul(u64::from(self.shape.hidden))
                .saturating_mul(F32_BYTES);
            matmul::project(
                plan,
                self.desc.output,
                Bound::new(
                    WeightRef::model(Role::Output),
                    Operand::ScratchFrom(Buf::Mixed, last),
                    Operand::Scratch(Buf::Logits),
                ),
                logit_rows,
                // The vocabulary projection runs on the logit rows alone, and a
                // prompt always asks for exactly one: its width never moves
                // with where the prompt was cut, so it keeps the tier a single
                // row is fastest on rather than expanding the widest weight in
                // the model for one row.
                Device {
                    role: PassRole::Step,
                    ..device
                },
            )?;
            // More than one logit row means a speculative verify or a decode
            // batch, and what comes back from a verify is the greedy pick of
            // every row rather than the rows themselves. A batch wants the rows,
            // so the pick is skipped there: its output is one slot per row
            // either way and reading it would cost a transfer nobody looks at.
            if logit_rows > 1 && !batched {
                plan.launch(
                    crate::sparse::kernels::Entry::plain(
                        crate::sparse::kernels::MOD_MISC,
                        "q4e_argmax_rows_kernel",
                    ),
                    crate::sparse::kernels::argmax_rows(logit_rows),
                    // `out, logits, n_vocab`
                    crate::sparse::bind::Args::of(&[
                        Operand::Scratch(Buf::Argmax),
                        Operand::Scratch(Buf::Logits),
                        Operand::U32(self.shape.vocab),
                    ]),
                );
            }
        }
        Ok(())
    }

    /// The n-gram gathers this pass will need, one per row.
    ///
    /// A batch's rows are different sequences, so each hashes its own trailing
    /// grams out of its own path: one gather covering `n_tok` consecutive
    /// positions would be the first row's grams answered for every row, which
    /// is a whole layer's contribution to the residual decoded out of another
    /// sequence's table rows.
    fn plan_gathers(
        &self,
        rows: &[&SparseSession],
        n_tok: u32,
        origin: RowOrigin,
    ) -> CoreResult<Vec<PleGather>> {
        let Some(ple) = self.shape.ple else {
            return Ok(Vec::new());
        };
        if let RowOrigin::Segmented(runs) = origin {
            // One gather per run, over that run's own positions: the rows land
            // in run order, which is the pass's row order.
            return rows
                .iter()
                .enumerate()
                .map(|(run, live)| {
                    let len = runs.len(u32::try_from(run).unwrap_or(u32::MAX));
                    let written = live.written();
                    PleGather::start(ple, written.saturating_add(len), written, len)
                })
                .collect();
        }
        if origin == RowOrigin::Sequential {
            let live = rows.first().ok_or(CoreError::ShapeMismatch {
                field: "pass rows",
                expected: 1,
                got: 0,
            })?;
            let written = live.written();
            return Ok(vec![PleGather::start(
                ple,
                written.saturating_add(n_tok),
                written,
                n_tok,
            )?]);
        }
        rows.iter()
            .map(|live| {
                let written = live.written();
                PleGather::start(ple, written.saturating_add(1), written, 1)
            })
            .collect()
    }

    /// The precision every stream mix of a pass of `n_tok` rows holds its
    /// normed residual and its expansion in.
    ///
    /// One answer for the whole pass, the draft head's mixes included, because
    /// every combine writes the normed residual the next mix reads and the two
    /// have to agree on its element type.
    pub(crate) fn stream_elem(&self, n_tok: u32, device: Device) -> Elem {
        let target = self
            .desc
            .layers
            .iter()
            .flat_map(|layer| [&layer.attn_mix, &layer.ffn_mix]);
        let head = self
            .desc
            .draft
            .iter()
            .flat_map(|head| [&head.layer.attn_mix, &head.layer.ffn_mix, &head.output_mix]);
        let mixes = target
            .chain(std::iter::once(&self.desc.output_mix))
            .chain(head);
        if matmul::half_streams(mixes, n_tok, device) {
            Elem::F16
        } else {
            Elem::F32
        }
    }

    /// One island's body.
    #[expect(
        clippy::too_many_arguments,
        reason = "every argument is a distinct decision the island body reads; \
                  a struct here would be assembled at the one call site and \
                  destructured immediately"
    )]
    pub(crate) fn plan_island(
        &self,
        slot: LayerRef,
        layer: &crate::sparse::weights::LayerDesc,
        il: LayerIdx,
        half: Island,
        chain: LayerChain,
        ctx: AttnContext,
        origin: RowOrigin,
        n_tok: u32,
        device: Device,
    ) -> CoreResult<Plan> {
        let mut plan = Plan::new();
        let elem = self.stream_elem(n_tok, device);
        let next = match half {
            // The token mixer's combine always feeds this layer's own
            // feed-forward mix, which is the one place in a pass where nothing
            // can come between them.
            Island::Mixer => NextNorm::Chained(WeightRef::layer(slot, Role::FfnMixNorm)),
            Island::Experts => match chain.experts_next {
                ChainLink::Broken => NextNorm::Broken,
                ChainLink::Chained => il
                    .next()
                    .filter(|next| next.get() < self.shape.n_layers())
                    .map_or(NextNorm::Broken, |next| {
                        NextNorm::Chained(WeightRef::layer(
                            LayerRef::target(next.get()),
                            Role::AttnMixNorm,
                        ))
                    }),
            },
        };
        match half {
            Island::Mixer => {
                // The n-gram block runs inside this island, ahead of its mix,
                // because it reads and rewrites the residual: a block that ran
                // after the mix would be mixing a residual it was about to
                // change.
                if let (Some(ple), Some(ngram)) = (self.shape.ple, layer.ngram)
                    && ple.layer == il
                {
                    plan.append(ple::block(
                        &self.shape,
                        ple,
                        &ngram,
                        slot,
                        NgramContext {
                            spec: self.opts.spec,
                            origin,
                        },
                        n_tok,
                        device,
                    )?);
                }
                plan.append(hyper::mix(
                    &self.shape,
                    &layer.attn_mix,
                    MixRoles::token_mixer(slot),
                    chain.mixer_input,
                    n_tok,
                    device,
                    elem,
                )?);
                match layer.block {
                    BlockDesc::Recurrent(desc) => plan.append(recurrent::block(
                        &self.shape,
                        &desc,
                        slot,
                        layer.ordinal,
                        Recurrence {
                            chunk: self.opts.chunk,
                            spec: self.opts.spec,
                            origin,
                        },
                        n_tok,
                        device,
                    )?),
                    BlockDesc::Attn(desc) => {
                        let attn = AttnSlot {
                            layer: slot,
                            rank: layer.ordinal,
                            // Rows, not pages: see `AttnSlot::pool_slots`.
                            pool_slots: self
                                .opts
                                .engine
                                .kv_pool_pages
                                .saturating_mul(shiso_core::page::PAGE_TOKENS),
                            ctx: self.opts.engine.ctx,
                        };
                        plan.append(qsa::block(&self.shape, &desc, attn, ctx, n_tok, device)?);
                    }
                }
            }
            Island::Experts => {
                plan.append(hyper::mix(
                    &self.shape,
                    &layer.ffn_mix,
                    MixRoles::experts(slot),
                    chain.experts_input,
                    n_tok,
                    device,
                    elem,
                )?);
                plan.append(moe::block(
                    &self.shape,
                    &layer.experts,
                    slot,
                    n_tok,
                    device,
                )?);
            }
        }
        plan.append(hyper::combine(&self.shape, next, n_tok, elem)?);
        Ok(plan)
    }
}

/// The execution-context number a session's captured islands are filed under.
///
/// A session handle is sixty-four bits and a key field is twenty-four, so this
/// is a narrowing - and it is a *checked* one in the key, which refuses rather
/// than masking. The alternative, hashing, would alias two live sessions onto
/// one slot with no way to notice.
/// Distinct batched passes the registry holds before it starts over.
const BATCH_SLOTS: usize = 1 << 12;

impl SparseStack {
    /// The context slot a capturable batched pass is filed under.
    ///
    /// Keyed by everything a replay depends on: the sessions in row order,
    /// whose addresses the launches bind, and every step of the plan, whose
    /// launches and operands the capture bakes in. A pass that differs in any
    /// of them - another selection mode, another run length - is a different
    /// slot and a capture of its own, never a replay of this one. Starting
    /// over drops every captured graph first, so a slot handed out again cannot
    /// name a graph of the pass it used to mean.
    pub(crate) fn batch_slot(&self, rows: &[&SparseSession], plan: &Plan) -> CoreResult<u32> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for row in rows {
            row.id().hash(&mut hasher);
        }
        plan.steps().hash(&mut hasher);
        let print = hasher.finish();
        let mut slots = self
            .batch_slots
            .lock()
            .map_err(|_| CoreError::Unsupported {
                what: "the batch registry after a panic while it was held",
                backend: "engine",
            })?;
        if let Some(slot) = slots.get(&print) {
            return Ok(*slot);
        }
        if slots.len() >= BATCH_SLOTS {
            if let Some(graphs) = self.backend.as_graph() {
                graphs.invalidate_all();
            }
            slots.clear();
        }
        let slot = island::BATCH_CONTEXT_BASE
            .saturating_add(u32::try_from(slots.len()).unwrap_or(u32::MAX));
        slots.insert(print, slot);
        Ok(slot)
    }
}

fn context_id(session: SessionId) -> u32 {
    u32::try_from(session.get().get()).unwrap_or(u32::MAX)
}

/// Puts the pass's one cooperative-scratch reservation in front of its first
/// island, and moves every island along by it.
///
/// A free function rather than a step the planner pushed as it went, because
/// the size is not known until the last launch has been planned: it is the
/// widest carve-up any of them makes, and a reservation that grew between two
/// launches would move the address the first of them was planned against.
pub(crate) fn with_reservation(body: Plan, islands: Vec<IslandSpan>) -> (Plan, Vec<IslandSpan>) {
    let bytes = body.coop_reservation();
    if bytes == 0 {
        return (body, islands);
    }
    let mut plan = Plan::new();
    plan.push(Step::Reserve { bytes });
    plan.append(body);
    let shifted = islands
        .into_iter()
        .map(|span| IslandSpan {
            start: span.start.saturating_add(1),
            ..span
        })
        .collect();
    (plan, shifted)
}

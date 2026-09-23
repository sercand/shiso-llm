//! The draft head's forward pass.
//!
//! A speculative step needs a second model that is cheap enough to run several
//! times per target step and close enough to be right most of the time. This
//! family's answer is a head that is **one layer of the same architecture at
//! the same widths** - the shipped head's block is tensor-for-tensor identical
//! in shape and format to a target attention layer - in front of which sit two
//! normalizations and two projections that turn a token and a target hidden
//! state into that layer's input.
//!
//! # What that identity buys, and what it costs
//!
//! It buys the whole layer planner: the head's block is planned by the same
//! code that plans a target attention slot, at the same widths, with the same
//! kernels. Nothing here re-derives a grid.
//!
//! It costs the type system a distinction it would otherwise not need. Layer
//! zero exists in both weight sets and its tensors are the same shape in both,
//! so a reference that resolved against the wrong one binds a weight of exactly
//! the right size and every launch succeeds. That is what
//! [`crate::sparse::bind::LayerRef`] is for, and it is why nothing in this
//! module names a layer by its index alone.
//!
//! # Where a draft starts
//!
//! Not from a gather alone. A target step leaves a hidden state; the head reads
//! the row for the last *committed* position together with the token at that
//! position plus one, and predicts the token after *that*. So its residual is
//! opened by [`crate::sparse::hyper::init_add`] over two terms rather than by
//! the broadcast a target pass opens with, and the hidden term is the reason
//! the head cannot run on the step that produced it: the row it reads is the
//! row that step wrote.
//!
//! # Two normalizations that are not the stream norm
//!
//! Both pre-norms are **one** group. The embedding is one group of the residual
//! width; the hidden state is one group of the *multi-stream* width - a single
//! RMS over all four streams together, against a weight that is four streams
//! wide. Taking the group count off the shape would normalize each stream
//! separately against a quarter of that weight: four normalizations of the
//! right width, the wrong numbers, and no launch that would notice.

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, SessionId, TokenId};

use crate::sparse::bind::{LayerRef, Operand, Role, WeightRef};
use crate::sparse::gather;
use crate::sparse::hyper::{self, MixRoles};
use crate::sparse::island::{self, Island};
use crate::sparse::matmul::{self, Bound, Device};
use crate::sparse::moe::RowOrigin;
use crate::sparse::mtp::{self, DraftSource, MAX_DRAFT, StepOutcome};
use crate::sparse::plan::Plan;
use crate::sparse::qsa::AttnContext;
use crate::sparse::scratch::Buf;
use crate::sparse::stack::{Pass, SparseStack};
use crate::sparse::weights::{LayerDesc, MixDesc, WeightDesc};
use shiso_core::model::Speculated;

/// Rows the widest draft pass of a decode step runs.
///
/// A chain step drafts one row. The pass that catches the head up after a
/// verify runs the accepted prefix plus the row the target just committed,
/// which is at most one more than the deepest draft. The pass that fills the
/// head's cache while a prompt is read is wider - a whole chunk - which is why
/// the buffers this module carves are sized by the chunk and not by this.
pub const ROWS: u32 = MAX_DRAFT + 1;

/// Whose rows a draft pass carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DraftRows {
    /// One session's consecutive positions; the pick is the last row's.
    Sequential,
    /// One row per session, each at its own position; every row is picked.
    /// The bits say which rows' own heads have outgrown the indexer budget.
    Batch(u8),
}

/// The widths one draft pass plans from.
///
/// Deliberately NOT `#[non_exhaustive]`: a loader builds one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftDesc {
    /// The projection of the normed token embedding.
    pub fc_embed: WeightDesc,
    /// The projection of the normed hidden state.
    pub fc_hidden: WeightDesc,
    /// The head's single layer.
    pub layer: LayerDesc,
    /// The mix that collapses the head's streams for the vocabulary head.
    pub output_mix: MixDesc,
    /// The narrowed vocabulary head's widths, when the build has one.
    pub vocabulary: Option<WeightDesc>,
}

impl SparseStack {
    /// The plan one draft pass runs.
    ///
    /// `hidden` names the multi-stream rows the head starts from: the target's
    /// own residual on the pass that follows a target step, and the head's
    /// previous output on a chain step. `frontier` is one past the highest
    /// position this pass writes, which is what decides whether the head's own
    /// attention has outgrown the indexer budget - the head's cache is its own
    /// and fills at its own rate, so the target's frontier is the wrong number
    /// here even though the two run over the same page table.
    ///
    /// # Errors
    ///
    /// Whatever the gather, the projections and the layer's own planners
    /// report, and [`shiso_core::error::CoreError::Unsupported`] when this
    /// stack carries no draft head.
    #[expect(
        clippy::too_many_lines,
        reason = "the head's pass is one sequence of stages and reads as one: \
                  the two terms it opens with are only meaningful beside each \
                  other, and a split between them would hide that they are \
                  added"
    )]
    pub(crate) fn plan_draft(
        &self,
        hidden: Operand,
        frontier: u32,
        n_tok: u32,
        pick: bool,
        device: Device,
        rows: DraftRows,
    ) -> CoreResult<Plan> {
        let head = self.desc.draft.as_ref().ok_or(CoreError::Unsupported {
            what: "a draft pass on a stack that carries no draft head",
            backend: "engine",
        })?;
        let shape = &self.shape;
        let mut plan = gather::rows(
            shape,
            self.desc.token_embd.dtype,
            gather::Bound::new(
                Operand::Scratch(Buf::Ids),
                Operand::Weight(WeightRef::model(Role::TokenEmbd)),
                Operand::Scratch(Buf::Embed),
            ),
            n_tok,
        )?;

        // The token term. One group of the residual width, into a buffer the
        // target pass uses for a block's output and this pass has spare.
        plan.append(hyper::norm_of(
            shape.hidden,
            1,
            shape.rms_eps,
            WeightRef::draft_model(Role::DraftNormEmbed),
            Operand::Scratch(Buf::Embed),
            Operand::Scratch(Buf::BlockOut),
            n_tok,
        ));
        matmul::project(
            &mut plan,
            head.fc_embed,
            Bound::new(
                WeightRef::draft_model(Role::DraftFcEmbed),
                Operand::Scratch(Buf::BlockOut),
                Operand::Scratch(Buf::Mixed),
            ),
            n_tok,
            device,
        )?;

        // The hidden term. One group of the *multi-stream* width, and then the
        // same matrix on every stream - which is one projection over
        // `n_tok * streams` rows rather than a per-stream launch, because the
        // streams are contiguous and the matrix does not depend on which one a
        // row came from.
        plan.append(hyper::norm_of(
            shape.stream_dim(),
            1,
            shape.rms_eps,
            WeightRef::draft_model(Role::DraftNormHidden),
            hidden,
            Operand::Scratch(Buf::DraftNormed),
            n_tok,
        ));
        matmul::project(
            &mut plan,
            head.fc_hidden,
            Bound::new(
                WeightRef::draft_model(Role::DraftFcHidden),
                Operand::Scratch(Buf::DraftNormed),
                Operand::Scratch(Buf::DraftHidden),
            ),
            n_tok.saturating_mul(shape.streams),
            device,
        )?;

        plan.append(hyper::init_add(
            shape,
            Operand::Scratch(Buf::Mixed),
            Operand::Scratch(Buf::DraftHidden),
            n_tok,
        ));

        // The head's block, through the target's own layer planner. The context
        // is the one a single-row decode takes: a draft never carries unrelated
        // rows, and its attention runs dense until its own cache holds more
        // complete blocks than the indexer budget.
        let (origin, batch_selected) = match rows {
            DraftRows::Sequential => (RowOrigin::Sequential, 0),
            DraftRows::Batch(bits) => (RowOrigin::Unrelated, bits),
        };
        let ctx = AttnContext {
            selected: origin == RowOrigin::Sequential && shape.indexer.sparse_at(frontier),
            tensor_core: self.opts.tensor_core,
            tiled: self.opts.tiled_attention,
            origin,
            batch_selected,
            ..AttnContext::default()
        };
        let chain = island::draft_chain();
        for half in Island::ORDER {
            plan.append(self.plan_island(
                LayerRef::draft(),
                &head.layer,
                LayerIdx::new(0),
                half,
                chain,
                ctx,
                origin,
                n_tok,
                device,
            )?);
        }

        // The head's own collapsing mix, then the *target's* vocabulary head on
        // the last row alone: the head shares the projection exactly as it
        // shares the embedding table, and a draft is one token.
        //
        // A pass that is only filling the head's cache asks for neither. The
        // projection is the single widest matmul in the model, and running it
        // for a token nobody reads is most of what such a pass would cost.
        if !pick {
            let (plan, _) = crate::sparse::passes::with_reservation(plan, Vec::new());
            plan.validate()?;
            return Ok(plan);
        }
        plan.append(hyper::collapse(
            shape,
            &head.output_mix,
            MixRoles::draft_output(),
            n_tok,
            device,
            self.stream_elem(n_tok, device),
        )?);
        // The rows the head collapsed are `n_tok` of them; one sequence's pick is
        // the last one, for the reason a chunk's logits are its last rows, and a
        // batch's is every row.
        let (last, picked) = match rows {
            DraftRows::Sequential => (
                u64::from(n_tok.saturating_sub(1))
                    .saturating_mul(u64::from(shape.hidden))
                    .saturating_mul(crate::sparse::kernels::F32_BYTES),
                1,
            ),
            DraftRows::Batch(_) => (0, n_tok),
        };
        // Through the narrowed head when there is one. A row it cannot reach is
        // a guess the target rejects, which costs a verify row that was paid
        // for either way - so this is a speed-for-acceptance trade and not a
        // correctness one, and the target's own pass still reads every row.
        let (vocabulary, width) = head.vocabulary.map_or(
            (WeightRef::model(Role::Output), self.desc.output),
            |narrow| (WeightRef::draft_model(Role::DraftVocabulary), narrow),
        );
        matmul::project(
            &mut plan,
            width,
            Bound::new(
                vocabulary,
                Operand::ScratchFrom(Buf::Mixed, last),
                Operand::Scratch(Buf::Logits),
            ),
            picked,
            device,
        )?;
        // A draft is greedy and the caller wants the token, not the row: four
        // bytes back instead of a megabyte.
        plan.launch(
            crate::sparse::kernels::Entry::plain(
                crate::sparse::kernels::MOD_MISC,
                "q4e_argmax_rows_kernel",
            ),
            crate::sparse::kernels::argmax_rows(picked),
            // `out, logits, n_vocab, n_rows`
            crate::sparse::bind::Args::of(&[
                Operand::Scratch(Buf::Argmax),
                Operand::Scratch(Buf::Logits),
                Operand::U32(head.vocabulary.map_or(shape.vocab, |narrow| narrow.out_dim)),
                Operand::U32(picked),
            ]),
        );
        // The head's cooperative launches read the same per-stream scratch the
        // target's do, and it is reserved the same way: once, in front of
        // everything, at the widest carve-up the pass makes. A draft that ran
        // before any target pass would otherwise be the first launch on the
        // stream and find nothing reserved at all.
        let (plan, _) = crate::sparse::passes::with_reservation(plan, Vec::new());
        plan.validate()?;
        Ok(plan)
    }
}

impl SparseStack {
    /// Runs one draft pass and returns the token it proposes.
    ///
    /// `tokens[i]` is the token that already sits at `positions[i]`, and what
    /// comes back is the head's greedy guess for the token after the last of
    /// them. `hidden` names the multi-stream rows the head starts from.
    ///
    /// The session is *not* advanced. A draft is a proposal: the target's
    /// verify pass is what commits, and a proposal the target rejects has to
    /// leave the session exactly as it found it.
    ///
    /// # Errors
    ///
    /// [`shiso_core::error::CoreError::ShapeMismatch`] for an empty pass or one
    /// wider than a prefill chunk, [`shiso_core::error::CoreError::UnknownSession`],
    /// [`shiso_core::error::CoreError::ContextOverflow`] when a drafted
    /// position reaches past the context, and whatever the planner, the device
    /// and the transfers report.
    /// `None` is not a refusal: it is what `pick: false` asked for. A pass that
    /// is only filling the head's cache runs no vocabulary projection, so there
    /// is no token to hand back - and returning a placeholder one would be a
    /// token a caller could draft with.
    pub fn draft(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        positions: &[u32],
        hidden: Operand,
        pick: bool,
    ) -> CoreResult<Option<TokenId>> {
        let n_tok = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if n_tok == 0 || n_tok > crate::sparse::stack::MAX_PREFILL_CHUNK {
            return Err(CoreError::ShapeMismatch {
                field: "draft rows",
                expected: u64::from(crate::sparse::stack::MAX_PREFILL_CHUNK),
                got: u64::from(n_tok),
            });
        }
        // One past the highest position the head will write, which is both what
        // the page table has to cover and what decides whether the head's own
        // attention has outgrown its budget.
        let reach = positions
            .iter()
            .copied()
            .max()
            .map_or(0, |at| at.saturating_add(1));
        if self.reserve_to(session, reach)?
            && let Some(graphs) = self.backend.as_graph()
        {
            graphs.invalidate_all();
        }
        self.write_draft_inputs(session, tokens, positions)?;

        let plan = self.plan_draft(
            hidden,
            reach,
            n_tok,
            pick,
            self.device(),
            DraftRows::Sequential,
        )?;
        let pass = Pass {
            // The draft head runs over rows the target already read, so it
            // splices nothing: the embeddings it starts from are the target's
            // own, picture rows included.
            media_rows: None,
            plan,
            // Not captured. A chain step is the same shape every time and is
            // worth capturing, but an island whose replay bound the *target's*
            // weights would be a draft that predicted from the target's first
            // layer - so the capture keys have to learn about the weight set
            // before this may.
            islands: Vec::new(),
            capturable: false,
            selected: false,
            rows: n_tok,
            origin: RowOrigin::Sequential,
            gather: Vec::new(),
        };
        self.execute(session, &pass)?;
        pick.then(|| self.read_draft_pick()).transpose()
    }
}

impl SparseStack {
    /// One draft pass over one row per session: `tokens[i]` at `positions[i]`
    /// of session `i`, the head reading `hidden`'s row `i`, and the head's pick
    /// for every row back.
    ///
    /// The batch form of [`SparseStack::draft`] with `pick` set: each row is
    /// the arithmetic a one-row draft of its session would do, over its own
    /// cache, so the guesses are the ones each session would have drafted
    /// alone - in one pass over the head's weights instead of one per session.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::draft`], for any row.
    pub(crate) fn draft_batch(
        &mut self,
        sessions: &[SessionId],
        tokens: &[TokenId],
        positions: &[u32],
        hidden: Operand,
    ) -> CoreResult<Vec<TokenId>> {
        let n = u32::try_from(sessions.len()).unwrap_or(u32::MAX);
        if n == 0 || n > crate::sparse::stack::MAX_DECODE_BATCH || tokens.len() != sessions.len() {
            return Err(CoreError::ShapeMismatch {
                field: "batched draft rows",
                expected: u64::from(crate::sparse::stack::MAX_DECODE_BATCH),
                got: u64::from(n),
            });
        }
        let mut republished = false;
        let mut bits = 0u8;
        let mut reach = 0u32;
        for (row, (session, at)) in sessions.iter().zip(positions).enumerate() {
            let past = at.saturating_add(1);
            republished |= self.reserve_to(*session, past)?;
            if self.shape.indexer.sparse_at(past) {
                bits |= 1u8 << (row % 8);
            }
            reach = reach.max(past);
        }
        if republished && let Some(graphs) = self.backend.as_graph() {
            graphs.invalidate_all();
        }
        let rows = tokens
            .iter()
            .zip(positions)
            .map(|(token, at)| crate::sparse::inputs::DecodeRow {
                token: *token,
                position: *at,
            })
            .collect::<Vec<_>>();
        self.write_placed_inputs(&rows)?;
        let plan = self.plan_draft(
            hidden,
            reach,
            n,
            true,
            self.device(),
            DraftRows::Batch(bits),
        )?;
        let pass = Pass {
            media_rows: None,
            plan,
            islands: Vec::new(),
            capturable: false,
            selected: false,
            rows: n,
            origin: RowOrigin::Unrelated,
            gather: Vec::new(),
        };
        self.pack_row_pointers(sessions)?;
        self.execute_rows(sessions, &pass)?;
        self.read_draft_picks(n)
    }

    /// The head's chains for several sessions at once, one batched draft pass
    /// a step: the first step opens from each session's staged row, gathered
    /// side by side, and every step after it from the head's own rows.
    fn chain_batch(
        &mut self,
        sessions: &[SessionId],
        firsts: &[TokenId],
        frontiers: &[u32],
        budget: u32,
    ) -> CoreResult<Vec<Vec<TokenId>>> {
        let width = u64::from(self.shape.stream_dim()) * crate::sparse::kernels::F32_BYTES;
        for (row, session) in sessions.iter().enumerate() {
            let stage = self
                .scratch_view(Buf::DraftStage)?
                .slice(row as u64 * width, width)?;
            let pending =
                self.session(*session)?
                    .pending_draft()
                    .ok_or(CoreError::Unsupported {
                        what: "a batched draft for a session that has staged no row",
                        backend: "engine",
                    })?;
            self.backend.copy_d2d(stage, pending)?;
        }
        let mut drafts = vec![Vec::with_capacity(budget as usize); sessions.len()];
        let mut tokens = firsts.to_vec();
        let mut positions = frontiers
            .iter()
            .map(|at| at.saturating_sub(1))
            .collect::<Vec<_>>();
        let mut hidden = Operand::Scratch(Buf::DraftStage);
        for _ in 0..budget {
            let picks = self.draft_batch(sessions, &tokens, &positions, hidden)?;
            for ((draft, token), pick) in drafts.iter_mut().zip(tokens.iter_mut()).zip(picks) {
                draft.push(pick);
                *token = pick;
            }
            for at in &mut positions {
                *at = at.saturating_add(1);
            }
            hidden = Operand::Scratch(Buf::Residual);
        }
        Ok(drafts)
    }

    /// Claims page-table entries up to `positions`, and says whether the table
    /// moved.
    ///
    /// [`SparseStack::reserve`] extends a session by a row count because a
    /// chunk's rows are the ones after its frontier. A draft's are not: its
    /// chain runs ahead of the frontier and the pass that catches it up starts
    /// from inside the accepted prefix, so what it can state is the position it
    /// reaches rather than how far past the frontier that is.
    fn reserve_to(&mut self, session: SessionId, positions: u32) -> CoreResult<bool> {
        let capacity = self.opts.engine.ctx;
        if positions > capacity {
            return Err(CoreError::ContextOverflow {
                session,
                cap: capacity,
            });
        }
        let live = self
            .sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let change = live.pages.ensure(positions)?;
        Ok(change == crate::sparse::paged::TableChange::Republished)
    }

    /// Writes the three inputs a draft pass reads, at positions of its own.
    fn write_draft_inputs(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        positions: &[u32],
    ) -> CoreResult<()> {
        let vocab = self.shape.vocab;
        let Self {
            backend,
            resident,
            sessions,
            staging,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        if !sessions.contains_key(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        let rows = &mut staging.inputs;
        rows.fill_at(tokens, positions, vocab)?;
        rows.upload_to(backend.as_ref(), &resident.scratch)
    }

    /// Reads back the one token the head picked.
    fn read_draft_pick(&self) -> CoreResult<TokenId> {
        self.read_draft_picks(1)?
            .first()
            .copied()
            .ok_or(CoreError::ShapeMismatch {
                field: "draft picks",
                expected: 1,
                got: 0,
            })
    }

    /// Reads back the tokens the head picked for the first `rows` rows.
    fn read_draft_picks(&self, rows: u32) -> CoreResult<Vec<TokenId>> {
        let view = self
            .scratch_view(Buf::Argmax)?
            .slice(0, u64::from(rows) * 4)?;
        let mut bytes = vec![0u8; usize::try_from(rows).unwrap_or(0) * 4];
        self.backend.copy_d2h(view, &mut bytes)?;
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| self.draft_token(i32::from_le_bytes(*word)))
            .collect()
    }

    /// The token one row of the draft head's argmax names.
    fn draft_token(&self, pick: i32) -> CoreResult<TokenId> {
        // Through the narrowed head's id table when there is one: the argmax
        // that comes back is a row of the *slice*, and returning it unmapped
        // would propose whatever token sits at that index of the whole
        // vocabulary - a valid id, a plausible word, and not the one the head
        // chose.
        let narrow = self
            .resident()?
            .weights
            .draft
            .as_ref()
            .and_then(|head| head.head.as_ref());
        let id = u32::try_from(pick).ok().and_then(|row| match narrow {
            Some(slice) => slice.id(row),
            None => (row < self.shape.vocab).then_some(row),
        });
        id.map(TokenId::new).ok_or(CoreError::InvalidValue {
            key: "draft pick",
            value: format!("row {pick} is not a row of the head this draft projected through"),
        })
    }
}

/// The tokens a speculative step drafted and what the target made of them.
impl SparseStack {
    /// One speculative decode step: draft, verify, settle.
    ///
    /// `first` is the token the caller sampled from the previous pass, which
    /// this step commits unconditionally at the session's frontier. The head
    /// chains guesses behind it, the target verifies all of them in one pass,
    /// and what comes back says how many were kept and which row of that pass
    /// carries the distribution to sample next.
    ///
    /// # What the session looks like afterwards
    ///
    /// Committed by exactly [`StepOutcome::committed`], with its recurrent
    /// state and n-gram window wound back to the last of those rows and a draft
    /// row staged for the next step. A rejected draft leaves nothing behind: the
    /// cache rows it wrote sit past the frontier, and the next pass overwrites
    /// them before anything reads them.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] on a stack with no draft head or a session
    /// not opened for speculation, [`CoreError::UnknownSession`], and whatever
    /// the draft, the verify pass and the rollback report.
    pub fn speculate(
        &mut self,
        session: SessionId,
        first: TokenId,
        eos: TokenId,
    ) -> CoreResult<Speculated> {
        let live = self.session(session)?;
        let (depth, frontier) = (live.spec_depth(), live.written());
        let budget = mtp::draft_budget(depth, frontier, self.opts.engine.ctx, MAX_DRAFT + 1);
        let drafts = self.chain(session, first, frontier, budget)?;

        // The verify pass: the committed token and every guess behind it, at
        // consecutive positions from the frontier. Its row `i` is the target's
        // own pick for the position `drafts[i]` proposed.
        let mut rows = Vec::with_capacity(drafts.len().saturating_add(1));
        rows.push(first);
        rows.extend_from_slice(&drafts);
        let logit_rows = u32::try_from(rows.len()).unwrap_or(1);
        self.run_rows(session, &rows, logit_rows)?;
        let picks = self.read_picks(logit_rows)?;

        let outcome = StepOutcome::settle(
            first,
            &drafts,
            picks.get(..drafts.len()).unwrap_or(&[]),
            eos,
            (!drafts.is_empty()).then_some(DraftSource::Head),
        )?;
        // The rewind before the commit, and both before the row is staged: the
        // staged row is the residual of the last *committed* position, and the
        // rewind is what makes the recurrent state agree with it.
        if let Some(rewind) = outcome.rollback {
            self.spec_rollback(session, rewind)?;
        }
        self.advance(session, &outcome.committed, false, Some(outcome.logits_row))?;
        self.stage_draft_row(session, outcome.logits_row)?;
        self.reflush(
            session,
            frontier,
            &outcome.committed,
            Operand::Scratch(Buf::Residual),
        )?;
        // What the caller has not seen: the guesses that were kept, and then the
        // target's own pick at the row the step settled on. `first` is left out
        // because it was already the caller's - handing it back would emit it
        // twice.
        let accepted = usize::try_from(outcome.logits_row).unwrap_or(0);
        let mut produced = outcome.committed.get(1..).unwrap_or(&[]).to_vec();
        produced.push(*picks.get(accepted).ok_or(CoreError::ShapeMismatch {
            field: "verify rows",
            expected: u64::from(outcome.logits_row).saturating_add(1),
            got: u64::try_from(picks.len()).unwrap_or(u64::MAX),
        })?);
        Ok(Speculated {
            produced,
            accepted: outcome.logits_row,
            drafted: u32::try_from(drafts.len()).unwrap_or(u32::MAX),
        })
    }

    /// Rewrites the head's cache for the rows this step committed, out of the
    /// target's own hidden states.
    ///
    /// The chain wrote those rows already, but it wrote them from the head's
    /// *own* states: the first guess reads the target's staged row, and every
    /// guess after it reads what the head itself produced. That is off the
    /// distribution the head was trained against, and it compounds - so a chain
    /// left in the cache makes the next chain worse, and the one after that
    /// worse again.
    ///
    /// Measured on a repetitive prompt without this, the head's acceptance was
    /// 16% at a chain of three and speculation was *slower* than plain decode
    /// at every depth tried.
    ///
    /// It costs one head pass over the accepted rows, which is a layer and no
    /// vocabulary projection. It has to run after the row is staged, because it
    /// overwrites the residual that row is copied from.
    fn reflush(
        &mut self,
        session: SessionId,
        frontier: u32,
        committed: &[TokenId],
        hidden: Operand,
    ) -> CoreResult<()> {
        // Row `i` of the verify pass is position `frontier + i`, and the head
        // at that position reads the token after it - which is `committed[i +
        // 1]`, still inside this step's own rows. The last committed position
        // has no such token yet: it is the one the next chain starts from, out
        // of the row just staged.
        let Some(tokens) = committed.get(1..) else {
            return Ok(());
        };
        if tokens.is_empty() {
            return Ok(());
        }
        let positions: Vec<u32> = (0..tokens.len())
            .map(|i| frontier.saturating_add(u32::try_from(i).unwrap_or(0)))
            .collect();
        self.draft(session, tokens, &positions, hidden, false)?;
        Ok(())
    }

    /// One speculative step for several sequences at once: each drafts its own
    /// guesses, and one segmented pass verifies every sequence's committed
    /// token and guesses together, reading the weights once for all of them.
    ///
    /// `steps[i]` is session `i`'s committed token, as [`SparseStack::speculate`]
    /// takes it, and what comes back is what `speculate` would have returned
    /// for each, in the same order. Each run of the verify is the arithmetic
    /// that session's own verify pass would do - its state, its cache, its
    /// selection - so a sequence decodes to the same text whether it is
    /// verified alone or beside others.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::speculate`], for any of the sessions, and
    /// [`CoreError::InvalidValue`] when one session appears twice.
    #[expect(
        clippy::too_many_lines,
        reason = "the step's phases run in one order over one set of borrowed \
                  plans and picks, and reading them as one sequence is what \
                  shows that every session settles before any reflushes"
    )]
    pub fn speculate_batch(
        &mut self,
        steps: &[(SessionId, TokenId)],
        eos: TokenId,
    ) -> CoreResult<Vec<Speculated>> {
        let started = std::time::Instant::now();
        let mut plans = Vec::with_capacity(steps.len());
        for &(session, first) in steps {
            let live = self.session(session)?;
            let (depth, frontier) = (live.spec_depth(), live.written());
            let budget = mtp::draft_budget(depth, frontier, self.opts.engine.ctx, MAX_DRAFT + 1);
            plans.push((session, first, frontier, budget));
        }
        // One batched draft pass a step when every session drafts as deep - the
        // usual case, since the depth is the server's and only the end of a
        // context shortens it - and each session's own chain otherwise.
        let uniform = plans.windows(2).all(|pair| pair[0].3 == pair[1].3);
        let chains = if uniform && plans.len() > 1 {
            let sessions = plans.iter().map(|plan| plan.0).collect::<Vec<_>>();
            let firsts = plans.iter().map(|plan| plan.1).collect::<Vec<_>>();
            let frontiers = plans.iter().map(|plan| plan.2).collect::<Vec<_>>();
            self.chain_batch(&sessions, &firsts, &frontiers, plans[0].3)?
        } else {
            let mut chains = Vec::with_capacity(plans.len());
            for &(session, first, frontier, budget) in &plans {
                chains.push(self.chain(session, first, frontier, budget)?);
            }
            chains
        };
        let mut runs: Vec<(SessionId, u32, Vec<TokenId>)> = Vec::with_capacity(steps.len());
        for (&(session, first, frontier, _), drafts) in plans.iter().zip(chains) {
            let mut rows = Vec::with_capacity(drafts.len().saturating_add(1));
            rows.push(first);
            rows.extend_from_slice(&drafts);
            runs.push((session, frontier, rows));
        }
        let borrowed = runs
            .iter()
            .map(|(session, _, rows)| (*session, rows.as_slice()))
            .collect::<Vec<_>>();
        let drafted_at = started.elapsed();
        self.run_segments(&borrowed)?;
        let total = runs.iter().map(|(_, _, rows)| rows.len()).sum::<usize>();
        let picks = self.read_picks(u32::try_from(total).unwrap_or(u32::MAX))?;
        let verified_at = started.elapsed();
        // Every catch-up pass below overwrites the residual, and each reads its
        // own session's verify rows: they are held first.
        let width = u64::from(self.shape.stream_dim()) * crate::sparse::kernels::F32_BYTES;
        {
            let hold = self
                .scratch_view(Buf::ResidualHold)?
                .slice(0, total as u64 * width)?;
            let residual = self
                .scratch_view(Buf::Residual)?
                .slice(0, total as u64 * width)?;
            self.backend.copy_d2d(hold, residual)?;
        }

        let mut settled = Vec::with_capacity(runs.len());
        let mut start = 0usize;
        for (session, frontier, rows) in &runs {
            let drafts = rows.get(1..).unwrap_or(&[]);
            let first = rows.first().copied().ok_or(CoreError::ShapeMismatch {
                field: "verify run rows",
                expected: 1,
                got: 0,
            })?;
            let own = picks.get(start..start + rows.len()).unwrap_or(&[]);
            let outcome = StepOutcome::settle(
                first,
                drafts,
                own.get(..drafts.len()).unwrap_or(&[]),
                eos,
                (!drafts.is_empty()).then_some(DraftSource::Head),
            )?;
            if let Some(rewind) = outcome.rollback {
                self.spec_rollback(*session, rewind)?;
            }
            let row = u32::try_from(start).unwrap_or(u32::MAX) + outcome.logits_row;
            self.advance(*session, &outcome.committed, false, Some(row))?;
            self.stage_draft_row(*session, row)?;
            settled.push((
                *session,
                *frontier,
                start,
                outcome,
                own.to_vec(),
                drafts.len(),
            ));
            start += rows.len();
        }

        let mut out = Vec::with_capacity(settled.len());
        for (session, frontier, start, outcome, own, drafted) in settled {
            let hidden = Operand::ScratchFrom(Buf::ResidualHold, start as u64 * width);
            self.reflush(session, frontier, &outcome.committed, hidden)?;
            let accepted = usize::try_from(outcome.logits_row).unwrap_or(0);
            let mut produced = outcome.committed.get(1..).unwrap_or(&[]).to_vec();
            produced.push(*own.get(accepted).ok_or(CoreError::ShapeMismatch {
                field: "verify rows",
                expected: u64::from(outcome.logits_row).saturating_add(1),
                got: own.len() as u64,
            })?);
            out.push(Speculated {
                produced,
                accepted: outcome.logits_row,
                drafted: u32::try_from(drafted).unwrap_or(u32::MAX),
            });
        }
        tracing::debug!(
            sessions = steps.len(),
            rows = total,
            draft_ms = drafted_at.as_secs_f64() * 1e3,
            verify_ms = verified_at.saturating_sub(drafted_at).as_secs_f64() * 1e3,
            settle_ms = started.elapsed().saturating_sub(verified_at).as_secs_f64() * 1e3,
            "a batched speculative step's phases"
        );
        Ok(out)
    }

    /// Whether a draft head is resident, which is what makes a step speculative.
    #[must_use]
    pub fn has_draft_head(&self) -> bool {
        self.desc.draft.is_some()
    }

    /// The head's chain of guesses behind `first`.
    ///
    /// The head at position `p` reads the hidden state for `p` and the token at
    /// `p + 1`. So the first guess is drafted at the frontier's last committed
    /// position with `first` as its token, and each one after that moves one
    /// position along and reads the head's own output.
    fn chain(
        &mut self,
        session: SessionId,
        first: TokenId,
        frontier: u32,
        budget: u32,
    ) -> CoreResult<Vec<TokenId>> {
        let mut drafts = Vec::with_capacity(budget as usize);
        let mut token = first;
        let mut at = frontier.saturating_sub(1);
        // The first draft reads the row the previous pass staged; every one
        // after it reads the head's own residual, which its previous pass left
        // in row zero.
        let mut hidden = Operand::DraftRow;
        for _ in 0..budget {
            let pick = self.draft(session, &[token], &[at], hidden, true)?.ok_or(
                CoreError::Unsupported {
                    what: "a draft pass that was asked for a token and produced none",
                    backend: "engine",
                },
            )?;
            drafts.push(pick);
            token = pick;
            at = at.saturating_add(1);
            hidden = Operand::Scratch(Buf::Residual);
        }
        Ok(drafts)
    }

    /// The greedy pick of every row of the pass that just ran.
    fn read_picks(&self, rows: u32) -> CoreResult<Vec<TokenId>> {
        let want = u64::from(rows).saturating_mul(4);
        let view = self.scratch_view(Buf::Argmax)?.slice(0, want)?;
        let mut bytes = vec![0u8; usize::try_from(want).unwrap_or(0)];
        self.backend.copy_d2h(view, &mut bytes)?;
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| {
                let pick = i32::from_le_bytes(*word);
                u32::try_from(pick)
                    .ok()
                    .filter(|row| *row < self.shape.vocab)
                    .map(TokenId::new)
                    .ok_or(CoreError::InvalidValue {
                        key: "verify pick",
                        value: format!(
                            "row {pick} is not a row of a {}-entry table",
                            self.shape.vocab
                        ),
                    })
            })
            .collect()
    }
}

impl SparseStack {
    /// Fills the head's own cache over a chunk of prompt the target has just
    /// read.
    ///
    /// Without this the head drafts against key and value rows nothing wrote.
    /// It still *answers* - its attention reads whatever was last at those
    /// addresses and its output is a token - so the only thing that shows is a
    /// draft the target rarely agrees with: a build that is slower for
    /// speculating and correct either way. Measured cold on a repetitive
    /// prompt, the head's acceptance was 16%.
    ///
    /// # Which rows, and in one pass
    ///
    /// The head at position `p` reads the target's hidden state for `p` and the
    /// token at `p + 1`. A chunk covering `[start, start + n)` can therefore
    /// answer for `[start, start + n - 1)` out of its own rows; its last
    /// position needs the token after it, which is the next chunk's first, and
    /// is warmed by [`SparseStack::warm_draft_boundary`] when that chunk
    /// arrives.
    ///
    /// One pass, not several. The rows this reads are the ones the chunk left
    /// in the residual, and this pass's own opening launch overwrites them - so
    /// a second pass over the same chunk would read rows the first replaced.
    ///
    /// # Errors
    ///
    /// Whatever [`SparseStack::draft`] reports.
    pub fn warm_draft(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        start: u32,
    ) -> CoreResult<()> {
        let rows = tokens.len().saturating_sub(1);
        let Some(span) = tokens.get(1..) else {
            return Ok(());
        };
        if !self.has_draft_head() || rows == 0 {
            return Ok(());
        }
        let positions: Vec<u32> = (0..rows)
            .map(|i| start.saturating_add(u32::try_from(i).unwrap_or(0)))
            .collect();
        self.draft(
            session,
            span,
            &positions,
            Operand::Scratch(Buf::Residual),
            false,
        )?;
        Ok(())
    }

    /// Fills the one row a chunk could not: its predecessor's last position.
    ///
    /// Called with the chunk's first token *before* the target reads the chunk,
    /// and that order is the whole of it. This pass writes the residual, and
    /// after the target's chunk has run the residual is what
    /// [`SparseStack::warm_draft`] reads; before it, the residual holds the
    /// previous chunk's rows, which nothing needs any more. Running it the
    /// other way round would cost this chunk's first row.
    ///
    /// It also has to come before the chunk is warmed rather than after, so
    /// that the rows that attend to this position see it written rather than
    /// see whatever the pool last held there.
    ///
    /// # Errors
    ///
    /// Whatever [`SparseStack::draft`] reports.
    pub fn warm_draft_boundary(
        &mut self,
        session: SessionId,
        first: TokenId,
        start: u32,
    ) -> CoreResult<()> {
        let Some(previous) = start.checked_sub(1) else {
            return Ok(());
        };
        if !self.has_draft_head() || self.session(session)?.pending_draft().is_none() {
            return Ok(());
        }
        self.draft(session, &[first], &[previous], Operand::DraftRow, false)?;
        Ok(())
    }
}

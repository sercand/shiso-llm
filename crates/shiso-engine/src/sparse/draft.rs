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
use crate::sparse::indexer::RingTap;
use crate::sparse::island::{self, Island};
use crate::sparse::matmul::{self, Bound, Device, NormedStreams};
use crate::sparse::moe::RowOrigin;
use crate::sparse::mtp::{self, DraftSource, MAX_DRAFT, StepOutcome};
use crate::sparse::plan::Plan;
use crate::sparse::qsa::AttnContext;
use crate::sparse::scratch::Buf;
use crate::sparse::stack::{Pass, SparseStack};
use crate::sparse::weights::{LayerDesc, MixDesc, WeightDesc};
use shiso_core::model::{KeptDrafts, Speculated, VerifiedBlock};

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
    /// One session's consecutive positions; the pick is the last row's. The
    /// tap, when there is one, copies the head's raw key ring into the
    /// snapshot buffer as it stands before one of those rows: see
    /// [`crate::sparse::indexer::RingTap`].
    Sequential(Option<RingTap>),
    /// One row per session, each at its own position; every row is picked.
    /// The bits say which rows' own heads have outgrown the indexer budget.
    Batch(u8),
}

/// Where a draft pass ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DraftEnd {
    /// The whole pass, through the collapsing mix, the vocabulary head and the
    /// argmax.
    Pick,
    /// The layer and no further: a pass that only fills the head's cache.
    Fill,
    /// A diagnostic stop: see [`DraftProbe`].
    Probe(DraftProbe),
}

/// The stage a diagnostic draft pass stops after, leaving that stage's
/// multi-stream rows in [`Buf::Residual`].
///
/// What it is for: holding the head's pass against an independent reference
/// stage by stage. The two buffers the opening terms are written to are never
/// reused later in the pass, so [`Buf::DraftNormed`] (the normed hidden term)
/// and [`Buf::DraftHidden`] (its projection, per stream) can be read after any
/// pass; the token term's projection, in [`Buf::Mixed`], only after a
/// [`DraftProbe::LayerInput`] stop, since the collapsing mix overwrites it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftProbe {
    /// After the two terms are added: the residual the head's layer opens on.
    LayerInput,
    /// After the attention half's combine, before the feed-forward half.
    ///
    /// The attention half writes the head's key and value rows for the probed
    /// position, the same rows a full pass at that position writes.
    PostAttention,
}

/// Whose answers decide what a speculative step keeps: see
/// [`SparseStack::speculate_judged`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Judge<'a> {
    /// The target's own greedy picks: the served step.
    Target,
    /// A given text: `text[i]` stands in for the target's answer at verify row
    /// `i`, which is the token at the position after the step's frontier plus
    /// `i`. The step commits the guesses that match it, then `text` at the
    /// first that does not, so a session walks the text whatever its guesses.
    Forced(&'a [TokenId]),
}

/// One speculative step as a harness inspects it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecTrace {
    /// The guesses the step verified, in order.
    pub drafts: Vec<TokenId>,
    /// The target's own greedy pick at every verify row: row `i` is its pick
    /// for the position `drafts[i]` proposed, the last row the one after them.
    /// These are the target's whatever the [`Judge`].
    pub picks: Vec<TokenId>,
    /// What the step returned.
    pub outcome: Speculated,
}

/// One session's verify pass, run and not yet settled.
///
/// Everything a settle needs to commit the rows that were kept and nothing
/// else: the token the step commits whatever happens, the guesses behind it,
/// and where in the pass its rows sit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedRound {
    /// The session the rows are this one's.
    session: SessionId,
    /// The token the step commits unconditionally, at `frontier`.
    first: TokenId,
    /// The session's frontier when the pass ran.
    frontier: u32,
    /// The guesses behind `first`, in order.
    drafts: Vec<TokenId>,
    /// Where the guesses came from, or `None` for a step that drafted nothing.
    source: Option<DraftSource>,
    /// The row of the pass `first` sits at: zero alone, the run's offset in a
    /// segmented pass.
    start: u32,
}

impl VerifiedRound {
    /// The tokens the verify pass runs: `first`, then the guesses.
    fn rows(&self) -> Vec<TokenId> {
        let mut rows = Vec::with_capacity(self.drafts.len().saturating_add(1));
        rows.push(self.first);
        rows.extend_from_slice(&self.drafts);
        rows
    }

    /// How many rows of the pass are this round's.
    fn row_count(&self) -> u32 {
        u32::try_from(self.drafts.len())
            .unwrap_or(u32::MAX)
            .saturating_add(1)
    }

    /// This round's rows of a per-row answer covering the whole pass.
    fn own<'a>(&self, answers: &'a [TokenId]) -> &'a [TokenId] {
        let from = self.start as usize;
        answers
            .get(from..from.saturating_add(self.row_count() as usize))
            .unwrap_or(&[])
    }

    /// The outcome of keeping the first `accepted` guesses, as the caller
    /// chose them.
    ///
    /// No end-of-turn check here, unlike [`StepOutcome::settle`]: a caller
    /// that chooses its own tokens stops at the end of its turn itself, and
    /// the count it hands back already stops there.
    fn kept(&self, accepted: u32) -> CoreResult<StepOutcome> {
        let drafted = u32::try_from(self.drafts.len()).unwrap_or(u32::MAX);
        if accepted > drafted {
            return Err(CoreError::ShapeMismatch {
                field: "kept drafts",
                expected: u64::from(drafted),
                got: u64::from(accepted),
            });
        }
        let mut committed = Vec::with_capacity(accepted as usize + 1);
        committed.push(self.first);
        committed.extend_from_slice(&self.drafts[..accepted as usize]);
        Ok(StepOutcome {
            committed,
            rollback: mtp::Rollback::new(accepted, drafted),
            logits_row: accepted,
            source: self.source,
        })
    }
}

/// The caller's proposal over a session's drafts: the drafts when they begin
/// with it, the proposal where they do not, cut to `budget` either way.
///
/// The proposal is structure the caller knows comes next, so a draft that
/// does not begin with it is wrong from its first disagreement; one that does
/// is the proposal and more, and is kept whole.
fn scaffolded(
    drafts: Vec<TokenId>,
    source: Option<DraftSource>,
    proposal: &[TokenId],
    budget: u32,
) -> (Vec<TokenId>, Option<DraftSource>) {
    let (mut drafts, source) = if proposal.is_empty() || drafts.starts_with(proposal) {
        (drafts, source)
    } else {
        (proposal.to_vec(), Some(DraftSource::Scaffold))
    };
    drafts.truncate(budget as usize);
    (drafts, source)
}

/// What a greedy step hands back: the guesses it kept, then the answer at the
/// row it settled on. `first` is left out because it was already the
/// caller's - handing it back would emit it twice.
fn produced(outcome: &StepOutcome, answers: &[TokenId]) -> CoreResult<Vec<TokenId>> {
    let accepted = usize::try_from(outcome.logits_row).unwrap_or(0);
    let mut produced = outcome.committed.get(1..).unwrap_or(&[]).to_vec();
    produced.push(*answers.get(accepted).ok_or(CoreError::ShapeMismatch {
        field: "verify rows",
        expected: u64::from(outcome.logits_row).saturating_add(1),
        got: u64::try_from(answers.len()).unwrap_or(u64::MAX),
    })?);
    Ok(produced)
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
        end: DraftEnd,
        device: Device,
        rows: DraftRows,
    ) -> CoreResult<Plan> {
        // The head's combines store the normed streams whatever the target
        // pass leaves: only the target's mixes are checked for the norming
        // reads (`SparseStack::normed_streams`).
        let device = Device {
            normed: NormedStreams::Stored,
            ..device
        };
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
        if end == DraftEnd::Probe(DraftProbe::LayerInput) {
            return finish(plan);
        }

        // The head's block, through the target's own layer planner. The context
        // is the one a single-row decode takes: a draft never carries unrelated
        // rows, and its attention runs dense until its own cache holds more
        // complete blocks than the indexer budget.
        let (origin, batch_selected, ring_tap) = match rows {
            DraftRows::Sequential(tap) => (RowOrigin::Sequential, 0, tap),
            DraftRows::Batch(bits) => (RowOrigin::Unrelated, u64::from(bits), None),
        };
        let ctx = AttnContext {
            selected: origin == RowOrigin::Sequential && shape.indexer.sparse_at(frontier),
            tensor_core: self.opts.tensor_core,
            tiled: self.opts.tiled_attention,
            origin,
            batch_selected,
            ring_tap,
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
                &ctx,
                &origin,
                n_tok,
                device,
            )?);
            if half == Island::Mixer && end == DraftEnd::Probe(DraftProbe::PostAttention) {
                return finish(plan);
            }
        }

        // The head's own collapsing mix, then the *target's* vocabulary head on
        // the last row alone: the head shares the projection exactly as it
        // shares the embedding table, and a draft is one token.
        //
        // A pass that is only filling the head's cache asks for neither. The
        // projection is the single widest matmul in the model, and running it
        // for a token nobody reads is most of what such a pass would cost.
        if end != DraftEnd::Pick {
            return finish(plan);
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
            DraftRows::Sequential(_) => (
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
        finish(plan)
    }
}

/// A draft plan made runnable: its scratch reserved and the whole validated.
///
/// The head's cooperative launches read the same per-stream scratch the
/// target's do, and it is reserved the same way: once, in front of everything,
/// at the widest carve-up the pass makes. A draft that ran before any target
/// pass would otherwise be the first launch on the stream and find nothing
/// reserved at all.
fn finish(plan: Plan) -> CoreResult<Plan> {
    let (plan, _) = crate::sparse::passes::with_reservation(plan, Vec::new());
    plan.validate()?;
    Ok(plan)
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
        let end = if pick { DraftEnd::Pick } else { DraftEnd::Fill };
        self.draft_tapped(session, tokens, positions, hidden, end, None)
    }

    /// Runs one draft pass at `position` that stops at `probe`, leaving that
    /// stage's rows in the residual: a diagnostic, for holding the head's pass
    /// against a reference stage by stage (see [`DraftProbe`]).
    ///
    /// `token` and `hidden` are what [`SparseStack::draft`] takes for a one-row
    /// pass. Like a draft, it advances nothing.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::draft`].
    pub fn draft_probe(
        &mut self,
        session: SessionId,
        token: TokenId,
        position: u32,
        hidden: Operand,
        probe: DraftProbe,
    ) -> CoreResult<()> {
        self.draft_tapped(
            session,
            &[token],
            &[position],
            hidden,
            DraftEnd::Probe(probe),
            None,
        )?;
        Ok(())
    }

    /// [`SparseStack::draft`], with the head's raw key ring also copied into
    /// the snapshot buffer by `tap`, ending where `end` says.
    ///
    /// `None` is not a refusal: it is what any `end` but [`DraftEnd::Pick`]
    /// asked for, as `pick: false` is for [`SparseStack::draft`].
    fn draft_tapped(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        positions: &[u32],
        hidden: Operand,
        end: DraftEnd,
        tap: Option<RingTap>,
    ) -> CoreResult<Option<TokenId>> {
        let n_tok = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if n_tok == 0 || n_tok > self.opts.prefill_rows {
            return Err(CoreError::ShapeMismatch {
                field: "draft rows",
                expected: u64::from(self.opts.prefill_rows),
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

        let pick = end == DraftEnd::Pick;
        let plan = self.plan_draft(
            hidden,
            reach,
            n_tok,
            end,
            self.device(),
            DraftRows::Sequential(tap),
        )?;
        // A chain step is the same shape every time and worth capturing, and
        // its key cannot alias a target island's: it is filed under the slot
        // the registry gives this exact pass - its session and every launch
        // and operand, the head's own weights among them - behind a layer
        // index no target layer has. Only when the verify passes are captured
        // too, and only the one-row chain step.
        let capture = n_tok == 1 && pick && self.opts.graphs.capture_verify;
        let islands = if capture {
            let slot = self.batch_slot(&[self.session(session)?], &plan)?;
            let start = plan
                .steps()
                .iter()
                .take_while(|step| matches!(step, crate::sparse::plan::Step::Reserve { .. }))
                .count();
            vec![crate::sparse::stack::IslandSpan {
                key: island::IslandKey {
                    layer: LayerIdx::new(DRAFT_ISLAND_LAYER),
                    island: Island::Mixer,
                    rows: 1,
                    selected: false,
                    context: island::IslandContext::Batched(slot),
                },
                start,
                len: plan.len().saturating_sub(start),
            }]
        } else {
            Vec::new()
        };
        let pass = Pass {
            // The draft head runs over rows the target already read, so it
            // splices nothing: the embeddings it starts from are the target's
            // own, picture rows included.
            media_rows: None,
            plan,
            islands,
            capturable: capture,
            selected: false,
            rows: n_tok,
            origin: RowOrigin::Sequential,
            gather: Vec::new(),
            ngram_island: None,
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
                // The head attends over its own cache only, which its prompt
                // warm-up fills at sequence positions, so it rotates at those
                // too: a shift here and none there would be the prefill-decode
                // mismatch the target's own shift exists to remove.
                rope_shift: 0,
            })
            .collect::<Vec<_>>();
        self.write_placed_inputs(&rows)?;
        let plan = self.plan_draft(
            hidden,
            reach,
            n,
            DraftEnd::Pick,
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
            ngram_island: None,
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
        self.make_room(session, positions)?;
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

/// One speculative step's drafts beside the target's own picks, at debug
/// level: the record that separates a head whose first guess is wrong (the
/// draft's input or first step) from one whose chain decays (its own rows), and
/// a step the prompt lookup drafted from one the head did.
fn trace_step(
    session: SessionId,
    source: Option<DraftSource>,
    drafts: &[TokenId],
    picks: &[TokenId],
    accepted: u32,
) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let ids = |tokens: &[TokenId]| {
            tokens
                .iter()
                .map(|token| token.get().to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let lookup = matches!(source, Some(DraftSource::Lookup(_)));
        tracing::debug!(
            %session,
            lookup,
            drafted = drafts.len(),
            accepted,
            drafts = %ids(drafts),
            target = %ids(picks.get(..drafts.len()).unwrap_or(picks)),
            "speculative step"
        );
    }
}

/// The layer index a captured draft chain step is filed under: past every
/// target layer a key can name, so the two can never share a slot.
const DRAFT_ISLAND_LAYER: u32 = 0xffff;

/// The prompt lookup beside the head.
impl SparseStack {
    /// The drafts a step verifies: the prompt lookup's proposal when it is
    /// longer than the head's chain and starts with all of it
    /// ([`mtp::lookup_beats_head`]), the head's chain otherwise.
    ///
    /// A guess either way - the verify pass decides what is kept - so the
    /// lookup changes how many tokens a step can commit and never which.
    fn lookup_or_head(
        &self,
        session: SessionId,
        first: TokenId,
        head: Vec<TokenId>,
        budget: u32,
        eos: TokenId,
    ) -> CoreResult<(Vec<TokenId>, Option<DraftSource>)> {
        let fallback = |head: Vec<TokenId>| {
            let source = (!head.is_empty()).then_some(DraftSource::Head);
            (head, source)
        };
        let reach = self.opts.lookup_drafts.min(budget);
        if reach == 0 {
            return Ok(fallback(head));
        }
        let live = self.session(session)?;
        let Some(spec) = live.spec.as_ref() else {
            return Ok(fallback(head));
        };
        let policy = mtp::NgramPolicy::new(reach, mtp::NGRAM_MIN_MATCH, eos);
        let proposal = mtp::propose(
            live.history(),
            first,
            spec.generated_from,
            &spec.lookup_fails,
            policy,
        );
        match proposal {
            Some(proposal) if mtp::lookup_beats_head(&proposal, &head) => {
                tracing::trace!(%session, len = proposal.len(), "prompt lookup drafts the step");
                let id = proposal.id();
                Ok((proposal.tokens().to_vec(), Some(DraftSource::Lookup(id))))
            }
            _ => Ok(fallback(head)),
        }
    }

    /// Writes a rejected lookup proposal into the session's failure ring.
    fn remember_lookup(&mut self, session: SessionId, outcome: &StepOutcome) -> CoreResult<()> {
        if let Some(id) = outcome.failed_lookup() {
            let live = self
                .sessions
                .get_mut(&session)
                .ok_or(CoreError::UnknownSession(session))?;
            if let Some(spec) = live.spec.as_mut() {
                spec.lookup_fails.remember(id);
            }
        }
        Ok(())
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
        Ok(self
            .speculate_judged(session, first, eos, Judge::Target)?
            .outcome)
    }

    /// [`SparseStack::speculate`], with what the step drafted and what the
    /// target answered handed back, and `judge` deciding what it keeps.
    ///
    /// With [`Judge::Target`] this is the served step exactly. With
    /// [`Judge::Forced`] the step keeps the guesses that match a given text
    /// instead of the target's picks, so a harness can walk a prompt it holds
    /// an independent reference for and still exercise every rejection,
    /// rewind and catch-up a served step runs: the verify pass, the rewind, the
    /// staged row and the head's catch-up are the served code over the rows the
    /// text commits.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::speculate`], and [`CoreError::ShapeMismatch`] when a
    /// forced text is shorter than the step's drafts.
    pub fn speculate_judged(
        &mut self,
        session: SessionId,
        first: TokenId,
        eos: TokenId,
        judge: Judge<'_>,
    ) -> CoreResult<SpecTrace> {
        let mut rounds = self.draft_rounds(&[(session, first, &[])], eos)?;
        let rows = self.verify_rounds(&mut rounds)?;
        let round = rounds.pop().ok_or(CoreError::ShapeMismatch {
            field: "verify rounds",
            expected: 1,
            got: 0,
        })?;
        // Row `i` of the verify pass is the target's own pick for the position
        // `drafts[i]` proposed.
        let picks = self.read_picks(rows)?;
        // The answers the step settles against: the target's own picks, or the
        // forced text's tokens at the same positions.
        let answers = match judge {
            Judge::Target => picks.as_slice(),
            Judge::Forced(text) => text.get(..rows as usize).ok_or(CoreError::ShapeMismatch {
                field: "forced verify answers",
                expected: u64::from(rows),
                got: text.len() as u64,
            })?,
        };

        let outcome = StepOutcome::settle(
            first,
            &round.drafts,
            answers.get(..round.drafts.len()).unwrap_or(&[]),
            eos,
            round.source,
        )?;
        trace_step(
            session,
            round.source,
            &round.drafts,
            &picks,
            outcome.logits_row,
        );
        self.settle_rounds(std::slice::from_ref(&round), std::slice::from_ref(&outcome))?;
        let produced = produced(&outcome, answers)?;
        let outcome = Speculated {
            produced,
            accepted: outcome.logits_row,
            drafted: u32::try_from(round.drafts.len()).unwrap_or(u32::MAX),
        };
        Ok(SpecTrace {
            drafts: round.drafts,
            picks,
            outcome,
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
    pub fn speculate_batch(
        &mut self,
        steps: &[(SessionId, TokenId)],
        eos: TokenId,
    ) -> CoreResult<Vec<Speculated>> {
        let started = std::time::Instant::now();
        let asked = steps
            .iter()
            .map(|&(session, first)| (session, first, &[][..]))
            .collect::<Vec<_>>();
        let mut rounds = self.draft_rounds(&asked, eos)?;
        let drafted_at = started.elapsed();
        let total = self.verify_rounds(&mut rounds)?;
        let picks = self.read_picks(total)?;
        let verified_at = started.elapsed();
        let mut outcomes = Vec::with_capacity(rounds.len());
        for round in &rounds {
            let own = round.own(&picks);
            let outcome = StepOutcome::settle(
                round.first,
                &round.drafts,
                own.get(..round.drafts.len()).unwrap_or(&[]),
                eos,
                round.source,
            )?;
            trace_step(
                round.session,
                round.source,
                &round.drafts,
                own,
                outcome.logits_row,
            );
            outcomes.push(outcome);
        }
        self.settle_rounds(&rounds, &outcomes)?;
        let mut out = Vec::with_capacity(rounds.len());
        for (round, outcome) in rounds.iter().zip(&outcomes) {
            out.push(Speculated {
                produced: produced(outcome, round.own(&picks))?,
                accepted: outcome.logits_row,
                drafted: u32::try_from(round.drafts.len()).unwrap_or(u32::MAX),
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

    /// The first half of a step whose tokens the caller chooses: each session
    /// drafts behind its `first`, one pass verifies them all, and what comes
    /// back is every session's drafts with the target's logits over them.
    /// Nothing is committed until [`SparseStack::settle_verified`].
    ///
    /// `steps[i]` is session `i`'s committed token and the caller's proposal,
    /// which replaces the session's drafts wherever the two disagree; see
    /// [`shiso_core::model::Speculating::verify_block`]. The drafts themselves
    /// are the head's greedy chain or the prompt lookup, whatever rule the
    /// caller will choose by: a draft only decides how much a step *can*
    /// keep, and the caller's own choice at every row decides what it does.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::speculate_batch`].
    pub fn verify_blocks(
        &mut self,
        steps: &[(SessionId, TokenId, &[TokenId])],
        eos: TokenId,
    ) -> CoreResult<Vec<VerifiedBlock>> {
        let mut rounds = self.draft_rounds(steps, eos)?;
        let total = self.verify_rounds(&mut rounds)?;
        let mut logits = self.read_logits(total)?;
        let vocab = self.shape.vocab as usize;
        let blocks = if let [round] = rounds.as_slice() {
            vec![VerifiedBlock {
                drafts: round.drafts.clone(),
                logits: std::mem::take(&mut logits),
                vocab,
            }]
        } else {
            rounds
                .iter()
                .map(|round| {
                    let from = round.start as usize * vocab;
                    let to = from + round.row_count() as usize * vocab;
                    VerifiedBlock {
                        drafts: round.drafts.clone(),
                        logits: logits.get(from..to).unwrap_or(&[]).to_vec(),
                        vocab,
                    }
                })
                .collect()
        };
        self.verified = rounds;
        Ok(blocks)
    }

    /// Commits, for every session [`SparseStack::verify_blocks`] verified, its
    /// `first` and the first `accepted` of its drafts, winding it back past the
    /// rest - the second half of the step, after the caller has chosen.
    ///
    /// `kept` names the sessions in the order they were verified.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when `kept` does not name the pending
    /// sessions in order (a settle with nothing pending included),
    /// [`CoreError::ShapeMismatch`] for more kept drafts than a session
    /// drafted, and whatever the rewind, the catch-up and the commit report.
    pub fn settle_verified(&mut self, kept: &[KeptDrafts]) -> CoreResult<()> {
        let rounds = std::mem::take(&mut self.verified);
        let matches = rounds.len() == kept.len()
            && rounds
                .iter()
                .zip(kept)
                .all(|(round, kept)| round.session == kept.session);
        if !matches {
            return Err(CoreError::Unsupported {
                what: "a settle that does not name the pending verify's sessions in order",
                backend: "engine",
            });
        }
        let outcomes = rounds
            .iter()
            .zip(kept)
            .map(|(round, kept)| round.kept(kept.accepted))
            .collect::<CoreResult<Vec<_>>>()?;
        self.settle_rounds(&rounds, &outcomes)
    }

    /// Every session's drafts behind its `first`: the head's chain - one
    /// batched chain when every session drafts as deep, which is the usual
    /// case, since the depth is the server's and only the end of a context
    /// shortens it - then the prompt lookup beside it, then the caller's
    /// proposal over both. Runs nothing through the target.
    ///
    /// A verify that was never settled is dropped here: a new round is drafted
    /// from the sessions as they stand, which is where that verify left them.
    fn draft_rounds(
        &mut self,
        steps: &[(SessionId, TokenId, &[TokenId])],
        eos: TokenId,
    ) -> CoreResult<Vec<VerifiedRound>> {
        self.verified.clear();
        let mut plans = Vec::with_capacity(steps.len());
        for &(session, first, _) in steps {
            let live = self.session(session)?;
            let (depth, frontier) = (live.spec_depth(), live.written());
            let budget = mtp::draft_budget(depth, frontier, self.opts.engine.ctx, MAX_DRAFT + 1);
            plans.push((
                session,
                first,
                frontier,
                budget.min(self.opts.head_depth()),
                budget,
            ));
        }
        let uniform = plans.windows(2).all(|pair| pair[0].3 == pair[1].3);
        let chains = if uniform && plans.len() > 1 {
            let sessions = plans.iter().map(|plan| plan.0).collect::<Vec<_>>();
            let firsts = plans.iter().map(|plan| plan.1).collect::<Vec<_>>();
            let frontiers = plans.iter().map(|plan| plan.2).collect::<Vec<_>>();
            self.chain_batch(&sessions, &firsts, &frontiers, plans[0].3)?
        } else {
            let mut chains = Vec::with_capacity(plans.len());
            for &(session, first, frontier, head, _) in &plans {
                chains.push(self.chain(session, first, frontier, head)?);
            }
            chains
        };
        let mut rounds = Vec::with_capacity(plans.len());
        for ((&(session, first, frontier, _, budget), head), &(_, _, proposal)) in
            plans.iter().zip(chains).zip(steps)
        {
            let (drafts, source) = self.lookup_or_head(session, first, head, budget, eos)?;
            let (drafts, source) = scaffolded(drafts, source, proposal, budget);
            rounds.push(VerifiedRound {
                session,
                first,
                frontier,
                drafts,
                source,
                start: 0,
            });
        }
        Ok(rounds)
    }

    /// The verify pass over every round: each session's `first` and drafts at
    /// consecutive positions from its frontier, one session alone or several
    /// in one segmented pass. Advances nothing, and files each round's first
    /// row of the pass in [`VerifiedRound::start`]. Returns the rows the pass
    /// ran.
    fn verify_rounds(&mut self, rounds: &mut [VerifiedRound]) -> CoreResult<u32> {
        if let [round] = rounds {
            let rows = round.rows();
            let count = round.row_count();
            self.run_rows(round.session, &rows, count)?;
            return Ok(count);
        }
        let rows = rounds.iter().map(VerifiedRound::rows).collect::<Vec<_>>();
        let borrowed = rounds
            .iter()
            .zip(&rows)
            .map(|(round, rows)| (round.session, rows.as_slice()))
            .collect::<Vec<_>>();
        self.run_segments(&borrowed)?;
        let mut start = 0u32;
        for round in rounds.iter_mut() {
            round.start = start;
            start = start.saturating_add(round.row_count());
        }
        Ok(start)
    }

    /// Settles every round by its outcome: the rewind, the staged draft row,
    /// the head's catch-up and the commit.
    ///
    /// The rewind first: the staged row is the residual of the last
    /// *committed* position, and the rewind is what makes the recurrent state
    /// agree with it. Then the row is staged, because the catch-up overwrites
    /// the residual it is copied from - which is why, over several sessions,
    /// every one stages before any catches up, each catch-up reading its own
    /// verify rows out of a copy held first. And the commit last, because a
    /// checkpoint it admits carries the head's raw key ring, and the head's
    /// rows up to the position before the new frontier are the catch-up's to
    /// write: committed before it, the blob would carry the chain's guesses
    /// for them. No catch-up writes the logits block, so the row a commit
    /// files is still the verify's.
    fn settle_rounds(
        &mut self,
        rounds: &[VerifiedRound],
        outcomes: &[StepOutcome],
    ) -> CoreResult<()> {
        let width = u64::from(self.shape.stream_dim()) * crate::sparse::kernels::F32_BYTES;
        let several = rounds.len() > 1;
        if several {
            let total = rounds
                .iter()
                .map(|round| u64::from(round.row_count()))
                .sum::<u64>();
            let hold = self
                .scratch_view(Buf::ResidualHold)?
                .slice(0, total * width)?;
            let residual = self.scratch_view(Buf::Residual)?.slice(0, total * width)?;
            self.backend.copy_d2d(hold, residual)?;
        }
        for (round, outcome) in rounds.iter().zip(outcomes) {
            self.remember_lookup(round.session, outcome)?;
            if let Some(rewind) = outcome.rollback {
                self.spec_rollback(round.session, rewind)?;
            }
            self.stage_draft_row(round.session, round.start + outcome.logits_row)?;
        }
        for (round, outcome) in rounds.iter().zip(outcomes) {
            let hidden = if several {
                Operand::ScratchFrom(Buf::ResidualHold, u64::from(round.start) * width)
            } else {
                Operand::Scratch(Buf::Residual)
            };
            self.reflush(round.session, round.frontier, &outcome.committed, hidden)?;
            self.advance(
                round.session,
                &outcome.committed,
                false,
                Some(round.start + outcome.logits_row),
            )?;
            self.mark_staged_current(round.session)?;
        }
        Ok(())
    }

    /// The front `rows` of the logits block, on the host.
    fn read_logits(&self, rows: u32) -> CoreResult<Vec<f32>> {
        let want = u64::from(rows)
            .saturating_mul(u64::from(self.shape.vocab))
            .saturating_mul(crate::sparse::kernels::F32_BYTES);
        let view = self.scratch_view(Buf::Logits)?.slice(0, want)?;
        let mut bytes = vec![0u8; usize::try_from(want).unwrap_or(0)];
        self.backend.copy_d2h(view, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }

    /// Whether a draft head is resident, which is what makes a step speculative.
    #[must_use]
    pub fn has_draft_head(&self) -> bool {
        self.desc.draft.is_some()
    }

    /// Marks the staged draft row as the one for the frontier `session` now
    /// stands on: what a step that staged its row before committing does once
    /// the commit has moved the frontier.
    fn mark_staged_current(&mut self, session: SessionId) -> CoreResult<()> {
        let live = self
            .sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        if let Some(spec) = live.spec.as_mut() {
            spec.staged_for = live.written;
        }
        Ok(())
    }

    /// The planes this stack's pages carry, and so the ranks its sessions keep
    /// a raw indexer key ring for: every attention slot's, and the draft
    /// head's when one is resident.
    #[must_use]
    pub fn page_planes(&self) -> crate::sparse::paged::PagePlanes {
        crate::sparse::paged::PagePlanes {
            indexer: true,
            draft: self.has_draft_head(),
            kv: self.opts.engine.kv_elem,
        }
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
        self.warm_draft_tapped(session, tokens, start, None)
    }

    /// [`SparseStack::warm_draft`] over a chunk a grid checkpoint is cut
    /// inside, at row `snapshot` of the target's pass over it.
    ///
    /// That checkpoint stands the head where it has written up to the position
    /// before the cut, and the raw keys of the block holding that position are
    /// in the head's ring only while this pass runs: it writes on past them.
    /// So the pass copies them into the snapshot buffer before that row of its
    /// own - row `snapshot - 1`, since the head's row `i` is the target's
    /// position `start + i`.
    pub(crate) fn warm_draft_tapped(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        start: u32,
        snapshot: Option<u32>,
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
        // The head's ring is the last rank's section of the blob.
        let offset = crate::sparse::paged::snapshot_ring_offset(&self.shape, self.opts.spec)
            .saturating_add(
                u64::from(self.shape.count(crate::shape::LayerKind::Attn))
                    .saturating_mul(crate::sparse::paged::ring_rank_bytes(&self.shape)),
            );
        let tap = snapshot
            .and_then(|row| row.checked_sub(1))
            .filter(|row| usize::try_from(*row).is_ok_and(|row| row < rows))
            .map(|row| RingTap { row, offset });
        self.draft_tapped(
            session,
            span,
            &positions,
            Operand::Scratch(Buf::Residual),
            DraftEnd::Fill,
            tap,
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

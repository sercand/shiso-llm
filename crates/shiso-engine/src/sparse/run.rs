//! Running a chunk: claim, plan, gather, write, issue, advance.
//!
//! [`crate::sparse::stack`] decides what a pass is made of and
//! [`crate::sparse::exec`] hands each step to the device. This is the file that
//! puts a whole chunk in order, and the order is the whole of it - every step
//! below is here because doing it at any other point produces a fluent wrong
//! answer rather than a failure.
//!
//! It is a second half of [`SparseStack`] split out for the per-file line cap,
//! exactly as [`crate::sparse::ckpt`] and [`crate::sparse::passes`] are, rather
//! than a layer above it.
//!
//! # The order, and what each position buys
//!
//! 1. **Claim the cache rows.** Before the pass, not after: a page the frontier
//!    has not claimed may be handed to another session at any moment, and a
//!    pass that wrote into one would be writing into somebody else's context.
//!    A claim that republished the page table also invalidates every captured
//!    graph, because a capture bakes in the table's address.
//! 2. **Plan.** The plan is also what starts the n-gram gather: the handle it
//!    carries is the one thing in a pass that has to be started early and
//!    awaited late.
//! 3. **Issue the gather.** The rows are a dozen cold reads a token off a table
//!    that is tens of gibibytes; they go onto a thread of their own here so the
//!    host work below overlaps them.
//! 4. **Write the inputs.** The token ids, the positions and the rotary table.
//!    A decode batch of more than one row adds the fifth input here, and it is
//!    an input in every sense the other three are: the per-row pointer arrays
//!    the batched kernels dereference. Nothing fails when they are missing -
//!    every launch of the pass reads whatever the allocator last left at those
//!    addresses and still succeeds - which is why they are written from one
//!    place, at one point in this order, like the rows above them.
//! 5. **Land the rows.** The await, and the upload into the staging the decode
//!    launch was planned against. It happens before the pass is issued rather
//!    than at the block, because a pass is handed to the device as one stream:
//!    the point where the launches could be split around the await is the point
//!    where an island stops being capturable.
//! 6. **Issue.** Every step, in order, islands replayed where the device has
//!    them.
//! 7. **Advance and commit.** After the pass, not before: a session that
//!    recorded the tokens first and then failed to run them would believe it
//!    held a prefix nothing computed.

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::{DecodeStep, Logits};

use crate::sparse::kernels::F32_BYTES;
use crate::sparse::media::MediaRun;
use crate::sparse::ple::{PleGather, PleRows};
use crate::sparse::scratch::Buf;
use crate::sparse::stack::{MAX_DECODE_BATCH, MAX_PREFILL_CHUNK, Pass, SparseStack};

impl SparseStack {
    /// Advances a session by a prompt chunk that produces no logits.
    ///
    /// A chunk wider than one pass is split here rather than refused: the split
    /// is invisible in the result, because the recurrent state, the n-gram
    /// window and the cache carry the prefix across an internal boundary
    /// exactly as across a caller's.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::ContextOverflow`] when the
    /// chunk reaches past the context, [`CoreError::ShapeMismatch`] for an
    /// empty chunk, and whatever the planner, the streaming tier, the device or
    /// the cache report.
    pub fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.advance_over(session, tokens, false, None)
    }

    /// Advances a session by its final prompt chunk and returns the row its
    /// last token produced.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::prefill_chunk`], plus [`CoreError::ShapeMismatch`] if
    /// the logits block is too short for one row, which would mean the scratch
    /// and the vocabulary disagree.
    pub fn prefill_last(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
    ) -> CoreResult<Logits<'_>> {
        self.advance_over(session, tokens, true, None)?;
        self.logits(1)
    }

    /// Advances a session by the WHOLE of a media-carrying prompt, splicing the
    /// tower's rows over the placeholder positions, and returns the row its
    /// last token produced.
    ///
    /// The whole prompt rather than a chunk, because the walk that produces the
    /// three axis streams and the splice map is built per call: a prompt handed
    /// over in pieces would restart the running multimodal position and the
    /// encoder row the splice has reached. The prompt is still split into
    /// device-sized chunks underneath, sharing one walk and one rotary prefix
    /// across them, which is what makes the single call cost no more than the
    /// chunked one would have.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::prefill_last`], plus [`CoreError::ShapeMismatch`] when
    /// the prompt's placeholder run and the encoder block's rows disagree - the
    /// count the items carry is the count the rendered prompt used, and nothing
    /// below re-expands a run, so the mismatch is the one media failure that
    /// otherwise stays fluent and answers about a rearranged picture.
    pub fn prefill_last_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &shiso_core::media::VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>> {
        // An empty item list is the text path, not a degenerate multimodal one:
        // a caller that builds the same request shape every time gets the old
        // launches back by handing over no items.
        let media = (!media.is_empty()).then_some(media);
        self.advance_over(session, tokens, true, media)?;
        self.logits(1)
    }

    /// Advances every member of a decode batch by one token.
    ///
    /// Row `i` of the result is `steps[i]`, and an empty slice is `Ok` with
    /// zero rows rather than an error.
    ///
    /// # One sequence's rows and several sequences' rows are two passes
    ///
    /// A batch of one is a chunk of one row and takes the ordinary path: the
    /// scalar kernels are exactly right for it, and taking the batched ones
    /// would pack a one-slot array to say what the buffer itself already says.
    /// A batch of more is one pass over the batched kernels, which is the whole
    /// of why it can be one pass at all - every kernel that carries per-sequence
    /// state reads it through an array this step packs.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] when any step names a handle this stack
    /// does not hold - checked for every member before any of them runs, so a
    /// batch that fails advances nobody - [`CoreError::Unsupported`] for a batch
    /// wider than the logits block or one this device cannot address per row,
    /// [`CoreError::InvalidValue`] when one session appears in two rows, and
    /// whatever a chunk reports.
    pub fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        let rows = u32::try_from(steps.len()).unwrap_or(u32::MAX);
        if rows > MAX_DECODE_BATCH {
            return Err(CoreError::Unsupported {
                what: "a decode batch wider than the logits block",
                backend: "engine",
            });
        }
        for step in steps {
            if !self.sessions.contains_key(&step.session) {
                return Err(CoreError::UnknownSession(step.session));
            }
        }
        if rows > 1 {
            self.run_batch(steps)?;
        } else {
            for step in steps {
                // A decode step writes a logits row and ends no prompt: the
                // row is what the sampler reads, not a boundary to resume from.
                self.run_chunk(step.session, &[step.token], 1, false, None)?;
            }
        }
        self.logits(rows)
    }

    /// One pass over one row per session, in this module's own order.
    ///
    /// The order is the chunk's with one step added, and the added step is the
    /// one that makes the pass a batch: the per-row pointer arrays are packed
    /// after the inputs are written and before anything is issued. Packing them
    /// after the pass would leave every batched launch dereferencing whatever
    /// the allocator last left at those addresses, and every launch would still
    /// succeed.
    ///
    /// The claim loop runs first and runs for every row, for the reason a chunk
    /// claims before it plans: a page the frontier has not claimed may be handed
    /// to another session at any moment. A claim that republished any row's page
    /// table invalidates every captured graph in the process, because a capture
    /// bakes in the table's address - the batch's own islands are not captured,
    /// but the scalar passes of the sessions in it are.
    fn run_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<()> {
        let mut plan_phase = Phases::start();
        let sessions = steps.iter().map(|step| step.session).collect::<Vec<_>>();
        let mut republished = false;
        for step in steps {
            republished |= self.reserve(step.session, 1)?;
        }
        if republished && let Some(graphs) = self.backend.as_graph() {
            graphs.invalidate_all();
        }
        let mut pass = self.plan_batch(&sessions)?;
        plan_phase.mark("plan");
        plan_phase.report(pass.plan.len());
        // One token per row, borrowed out of the step list so the pair a gather
        // is issued for is the pair the row was planned for.
        let tokens = steps
            .iter()
            .map(|step| std::slice::from_ref(&step.token))
            .collect::<Vec<_>>();
        let rows = sessions
            .iter()
            .copied()
            .zip(tokens.iter().copied())
            .collect::<Vec<_>>();
        let mut phase = Phases::start();
        let gathered = self.issue_gathers(&mut pass, &rows)?;
        phase.mark("issue");
        self.write_batch_inputs(steps)?;
        self.pack_row_pointers(&sessions)?;
        phase.mark("inputs");
        self.land_gathers(gathered)?;
        phase.mark("land");
        self.execute_rows(&sessions, &pass)?;
        phase.mark("execute");
        phase.report(pass.plan.len());
        // Row `i` of the logits block is step `i`, which is what makes the
        // frontier row a session records the row it can be resumed from.
        for (row, step) in steps.iter().enumerate() {
            let row = u32::try_from(row).unwrap_or(u32::MAX);
            self.advance(step.session, &[step.token], false, Some(row))?;
        }
        Ok(())
    }

    /// One segmented pass: every session's run of tokens at its own frontier,
    /// one pass over the weights, logits for every row. Nothing is advanced:
    /// this is a batch's verify, and how many of each run's rows a session
    /// keeps is decided from the picks afterwards.
    ///
    /// # Errors
    ///
    /// Whatever [`SparseStack::plan_segments`], the gathers, the inputs and the
    /// pass report.
    pub(crate) fn run_segments(&mut self, runs: &[(SessionId, &[TokenId])]) -> CoreResult<()> {
        let mut republished = false;
        for (session, tokens) in runs {
            let n = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
            republished |= self.reserve(*session, n)?;
        }
        if republished && let Some(graphs) = self.backend.as_graph() {
            graphs.invalidate_all();
        }
        let sessions = runs.iter().map(|(session, _)| *session).collect::<Vec<_>>();
        let lens = runs
            .iter()
            .map(|(_, tokens)| u32::try_from(tokens.len()).unwrap_or(u32::MAX))
            .collect::<Vec<_>>();
        let mut pass = self.plan_segments(&sessions, &lens)?;
        let gathered = self.issue_gathers(&mut pass, runs)?;
        self.write_segment_inputs(runs)?;
        // The recurrence and its convolution run every session's run in one
        // launch, through the per-row arrays: one pointer per run here.
        self.pack_row_pointers(&sessions)?;
        self.land_gathers(gathered)?;
        self.execute_rows(&sessions, &pass)?;
        Ok(())
    }

    /// Writes every per-row pointer array this pass's batched launches read.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::Unsupported`] on a device
    /// that cannot hand out the addresses of its own allocations, and whatever
    /// the transfer reports.
    pub(crate) fn pack_row_pointers(&self, sessions: &[SessionId]) -> CoreResult<()> {
        let resident = self.resident()?;
        let rows = sessions
            .iter()
            .map(|id| self.session(*id))
            .collect::<CoreResult<Vec<_>>>()?;
        crate::sparse::rows::pack(self.backend.as_ref(), &resident.scratch, &self.shape, &rows)
    }

    /// The front `rows` of the logits block.
    fn logits(&self, rows: u32) -> CoreResult<Logits<'_>> {
        let vocab = self.shape.vocab;
        let bytes = u64::from(rows)
            .saturating_mul(u64::from(vocab))
            .saturating_mul(F32_BYTES);
        Logits::new(
            self.scratch_view(Buf::Logits)?.slice(0, bytes)?,
            rows,
            vocab,
        )
    }

    /// Runs `tokens` in passes no wider than one chunk.
    ///
    /// `with_logits` applies to the last piece only: the pieces before it are
    /// ordinary prefill chunks, which is what makes the split invisible. The
    /// piece that carries the logits is also the one whose commit earns a
    /// checkpoint, because it is the position a later request would resume
    /// from.
    fn advance_over(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        with_logits: bool,
        media: Option<&shiso_core::media::VisionPrefill<'_>>,
    ) -> CoreResult<()> {
        if tokens.is_empty() {
            return Err(CoreError::ShapeMismatch {
                field: "prompt chunk",
                expected: 1,
                got: 0,
            });
        }
        let start = self.session(session)?.written();
        // This family states one position rule and every request it serves is
        // walked by it: a placeholder token's three axes are a grid coordinate.
        // The dense families read theirs off the config because two of them
        // disagree; there is one of this one.
        let positions = crate::shape::MediaPositions::Mrope3D;
        let mut run = media.map(|prefill| MediaRun::new(prefill, start, positions));
        let width = usize::try_from(MAX_PREFILL_CHUNK).unwrap_or(usize::MAX);
        let mut rest = tokens;
        while !rest.is_empty() {
            // A chunk may not end inside a placeholder run: the walk hands out
            // an item's encoder rows and its grid positions together, and a cut
            // through one would split an item's rows across two passes whose
            // maps were built from two different walks of the same item.
            let take = match run.as_ref() {
                Some(run) => run.walk.split_point(rest, width),
                None => rest.len().min(width),
            };
            let (head, tail) = rest.split_at(take);
            let logit_rows = u32::from(tail.is_empty() && with_logits);
            // The draft head's own cache is filled as the prompt is read, and
            // the three calls around this one are in the only order that works:
            // see `warm_draft_boundary`. A build with no head does none of it.
            let start = self.session(session)?.written();
            if let Some(first) = head.first() {
                self.warm_draft_boundary(session, *first, start)?;
            }
            self.run_chunk(session, head, logit_rows, logit_rows > 0, run.as_mut())?;
            // Staged before the warm-up reads the residual, because the warm-up
            // is what overwrites it.
            if self.has_draft_head() {
                let last = u32::try_from(head.len().saturating_sub(1)).unwrap_or(0);
                self.stage_draft_row(session, last)?;
            }
            self.warm_draft(session, head, start)?;
            // The rows the next piece needs are known the moment this one is
            // enqueued, and this one is about to spend its time on the device,
            // so reading them now costs nothing and the fetch that asks for
            // them hits. A prefetch that fails costs a later gather its hit and
            // nothing else, so it is logged rather than propagated.
            let next = &tail[..tail.len().min(width)];
            if !next.is_empty()
                && let Err(error) = self.prefetch_next(session, next)
            {
                tracing::debug!(%error, "the n-gram prefetch for the next chunk did not start");
            }
            rest = tail;
        }
        // The count the items carry is the count the rendered prompt used, and
        // nothing below re-expands a placeholder run. So the walk having handed
        // out every encoder row is the statement that the prompt and the tower
        // described the same request - the one mismatch that otherwise succeeds,
        // stays fluent and answers about a rearranged picture.
        if let Some(run) = run.as_ref() {
            let consumed = u64::from(run.walk.rows_consumed());
            let available = u64::from(run.prefill.row_count());
            if consumed != available {
                return Err(CoreError::ShapeMismatch {
                    field: "vision placeholder tokens",
                    expected: available,
                    got: consumed,
                });
            }
        }
        Ok(())
    }

    /// One pass over one chunk, in the order this module's notes give.
    ///
    /// `earned` is the chunk's own answer to whether it ends a prompt, which a
    /// logits row does not settle: a decode step writes one too.
    fn run_chunk(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        logit_rows: u32,
        earned: bool,
        media: Option<&mut MediaRun<'_>>,
    ) -> CoreResult<()> {
        self.run_rows_media(session, tokens, logit_rows, media)?;
        // `logit_rows - 1` is the row this chunk's last position produced, and
        // `None` for a chunk that produced none: a checkpoint admitted without
        // that row cannot be resumed from without re-running a step.
        self.advance(session, tokens, earned, logit_rows.checked_sub(1))
    }

    /// The pass itself, without committing anything.
    ///
    /// Split from [`SparseStack::run_chunk`] for the one caller that must not
    /// commit what it ran: a verify pass runs the drafts *and* the token behind
    /// them, and how many of those rows the session keeps is not known until
    /// its picks have been read back. Advancing here would put a session on a
    /// prefix the target did not agree with, and every later launch would
    /// succeed.
    pub(crate) fn run_rows(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        logit_rows: u32,
    ) -> CoreResult<()> {
        self.run_rows_media(session, tokens, logit_rows, None)
    }

    /// The same pass, splicing a request's encoder rows over its placeholder
    /// tokens when the chunk carries any.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::run_rows`], plus whatever the walk and the splice
    /// report.
    pub(crate) fn run_rows_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        logit_rows: u32,
        media: Option<&mut MediaRun<'_>>,
    ) -> CoreResult<()> {
        let n_tok = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if self.reserve(session, n_tok)?
            && let Some(graphs) = self.backend.as_graph()
        {
            // A captured island holds the page table it was captured against.
            // The pool has just handed this session a different one, so every
            // capture in this process now names pages that may belong to
            // somebody else.
            graphs.invalidate_all();
        }
        let mut phase = Phases::start();
        // The encoder block's height travels into the plan and its address into
        // the binder, and the two are read from the same run: a plan carrying a
        // splice that the binder cannot name is refused by `execute_rows_media`
        // rather than issued.
        let block = media.as_ref().map(|run| run.prefill);
        let mut pass = self.plan_chunk_media(
            session,
            tokens,
            logit_rows,
            block.map(|prefill| prefill.row_count()),
        )?;
        phase.mark("plan");
        let gathered = self.issue_gathers(&mut pass, &[(session, tokens)])?;
        phase.mark("issue");
        match media {
            Some(run) => self.write_inputs_media(session, tokens, run)?,
            None => self.write_inputs(session, tokens)?,
        }
        phase.mark("inputs");
        self.land_gathers(gathered)?;
        phase.mark("land");
        // The tap is taken out for the pass and put back after it, on the
        // failure path too, so an armed tap outlives a pass that failed and
        // the caller still disarms it. Unarmed, this is the plain call.
        match self.tap.take() {
            None => {
                self.execute_rows_media(&[session], &pass, block.map(|prefill| prefill.rows()))?;
            }
            Some(mut tap) => {
                let ran = self.execute_rows_tapped(
                    &[session],
                    &pass,
                    block.map(|prefill| prefill.rows()),
                    Some(&mut tap),
                );
                self.tap = Some(tap);
                ran?;
            }
        }
        phase.mark("execute");
        phase.report(pass.plan.len());
        Ok(())
    }

    /// Starts this pass's n-gram reads on threads of their own, one per row.
    ///
    /// The gathers are taken out of the pass rather than borrowed, so a pass
    /// that reaches [`SparseStack::execute`] still holding one is a pass whose
    /// rows nobody fetched - which that method refuses rather than decoding a
    /// layer out of the staging's previous contents.
    ///
    /// `rows` is one `(session, tokens)` pair per row of the pass, in row order,
    /// and the order is load-bearing: the decode launch reads the staging by
    /// `(row, hash head)`, so a gather landed out of order is one sequence's
    /// n-gram rows decoded as another's. An empty result means this geometry
    /// folds in no n-gram block.
    fn issue_gathers(
        &mut self,
        pass: &mut Pass,
        rows: &[(SessionId, &[TokenId])],
    ) -> CoreResult<Vec<PleRows>> {
        let gathers = std::mem::take(&mut pass.gather);
        if gathers.is_empty() {
            return Ok(Vec::new());
        }
        if gathers.len() != rows.len() {
            return Err(CoreError::ShapeMismatch {
                field: "n-gram gathers per row",
                expected: rows.len() as u64,
                got: gathers.len() as u64,
            });
        }
        let Self {
            resident,
            sessions,
            staging,
            ..
        } = self;
        let tier = resident
            .as_ref()
            .and_then(|resident| resident.ngram.as_ref())
            .ok_or(CoreError::Unsupported {
                what: "an n-gram gather on a stack that holds no streamed table",
                backend: "engine",
            })?;
        let mut issued = Vec::with_capacity(gathers.len());
        for (gather, (session, tokens)) in gathers.into_iter().zip(rows.iter()) {
            let live = sessions
                .get(session)
                .ok_or(CoreError::UnknownSession(*session))?;
            // The hash reads the tokens *behind* each position, so what it needs
            // is that row's own path by absolute position - the prefix its
            // session has already written, then the tokens about to run, which
            // are not in the path yet because the pass that computes them has
            // not run. One reused buffer rather than a fresh vector: this runs
            // once a token on the decode path, and the identifiers are hashed
            // out of it before the next row refills it.
            staging.path.clear();
            staging.path.extend_from_slice(live.history());
            staging.path.extend_from_slice(tokens);
            issued.push(gather.issue(tier.table().stream(), &staging.path)?);
        }
        Ok(issued)
    }

    /// Awaits every row's rows and uploads them into the staging the pass was
    /// planned against, in row order.
    ///
    /// One transfer rather than one per row: the staging is a single slab the
    /// decode launch reads from its front, and the rows of row `i` sit exactly
    /// where `i` rows of the same width put them.
    fn land_gathers(&self, gathered: Vec<PleRows>) -> CoreResult<()> {
        if gathered.is_empty() {
            return Ok(());
        }
        let mut bytes = Vec::new();
        for rows in gathered {
            let landed = rows.await_rows()?;
            if bytes.is_empty() {
                bytes = landed;
            } else {
                bytes.extend_from_slice(&landed);
            }
        }
        let tier = self
            .resident()?
            .ngram
            .as_ref()
            .ok_or(CoreError::Unsupported {
                what: "an n-gram upload on a stack that holds no streamed table",
                backend: "engine",
            })?;
        tier.upload(self.backend.as_ref(), &bytes)
    }

    /// Warms the table for the chunk after the one just enqueued.
    ///
    /// `tokens` is that next chunk, whose positions begin where the session now
    /// stands - which is why this is called after the chunk before it has been
    /// committed rather than before.
    fn prefetch_next(&self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        let Some(ple) = self.shape.ple else {
            return Ok(());
        };
        let Some(tier) = self.resident()?.ngram.as_ref() else {
            return Ok(());
        };
        let live = self.session(session)?;
        let n_tok = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        let frontier = live.written().saturating_add(n_tok);
        let gather = PleGather::start(ple, frontier, live.written(), n_tok)?;
        let mut path = Vec::with_capacity(live.history().len().saturating_add(tokens.len()));
        path.extend_from_slice(live.history());
        path.extend_from_slice(tokens);
        crate::sparse::ple::prefetch(tier.table().stream(), &gather, &path)
    }
}

/// Wall-clock between the phases of one decoded token.
///
/// A diagnostic switch, off unless `SHISO_PHASE_LOG=1`, and it measures rather
/// than changes: the work issued is identical and only an `Instant` is read
/// between the phases. It exists because "decode is a third of the reference's
/// rate" is a number with no address until the step is broken into the four
/// things it is made of - planning, starting the streamed gather, writing the
/// host inputs, and issuing the launches.
#[derive(Debug)]
struct Phases {
    at: Option<std::time::Instant>,
    marks: Vec<(&'static str, u128)>,
}

impl Phases {
    fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SHISO_PHASE_LOG").as_deref() == Ok("1"))
    }

    fn start() -> Self {
        Self {
            at: Self::on().then(std::time::Instant::now),
            marks: Vec::new(),
        }
    }

    fn mark(&mut self, what: &'static str) {
        if let Some(at) = self.at.as_mut() {
            let now = std::time::Instant::now();
            self.marks.push((what, now.duration_since(*at).as_micros()));
            *at = now;
        }
    }

    fn report(&self, steps: usize) {
        if self.at.is_some() {
            tracing::info!(steps, marks = ?self.marks, "decode phases, microseconds");
        }
    }
}

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

use crate::sparse::ckpt::Boundary;
use crate::sparse::kernels::{F32_BYTES, GDN_CHUNK};
use crate::sparse::media::MediaRun;
use crate::sparse::moe::RowOrigin;
use crate::sparse::passes::PassRole;
use crate::sparse::ple::{PleGather, PleRows};
use crate::sparse::scratch::Buf;
use crate::sparse::stack::{MAX_DECODE_BATCH, MAX_PROMPT_RUNS, Pass, SparseStack};

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
        self.advance_over(session, tokens, false, None)?;
        self.mark_prompt_end(session)
    }

    /// Records the session's frontier as where its next generated tokens
    /// begin, which is what the prompt lookup keys past.
    fn mark_prompt_end(&mut self, session: SessionId) -> CoreResult<()> {
        let live = self
            .sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let written = live.written;
        if let Some(spec) = live.spec.as_mut() {
            spec.generated_from = written;
        }
        Ok(())
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
        self.mark_prompt_end(session)?;
        self.logits(1)
    }

    /// Advances a session by `tokens` in one pass planned as a speculative
    /// verify plans its rows, and returns every row's logits.
    ///
    /// A verify's rows are held to the single decode step, not to a prompt
    /// chunk: they take the tiers their own row count reaches, which is what
    /// makes a verify's pick at every row the pick a plain step would make. This
    /// is that pass without the draft that normally supplies its rows, so a
    /// caller can put it beside the decode steps it stands in for; unlike a
    /// verify, it commits every row.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::prefill_chunk`], and [`CoreError::ShapeMismatch`] for
    /// more rows than the logits block holds.
    pub fn step_rows(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        let rows = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if rows == 0 || rows > crate::sparse::stack::MAX_LOGIT_ROWS {
            return Err(CoreError::ShapeMismatch {
                field: "step rows",
                expected: u64::from(crate::sparse::stack::MAX_LOGIT_ROWS),
                got: u64::from(rows),
            });
        }
        self.run_chunk(session, tokens, rows, false, None)?;
        self.logits(rows)
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
        self.mark_prompt_end(session)?;
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
        if rows > 1 && self.backend.as_pointers().is_none() {
            // The batched kernels read one pointer per row out of a device
            // array, and a device that cannot hand out its allocations'
            // addresses cannot fill one. Each row is then its own one-row
            // pass - the arithmetic its sequence would get decoding alone -
            // and its logits row is moved into place once the next pass has
            // no further use for row zero.
            let vocab_row = u64::from(self.shape.vocab) * 4;
            let mut staged = vec![0u8; usize::try_from(vocab_row).unwrap_or(usize::MAX)];
            let mut rows_out = Vec::with_capacity(steps.len());
            for step in steps {
                self.run_chunk(step.session, &[step.token], 1, false, None)?;
                let row = self.logits(1)?.row(0)?;
                self.backend.copy_d2h(row, &mut staged)?;
                rows_out.push(staged.clone());
            }
            let block = self.logits(rows)?;
            for (at, bytes) in rows_out.iter().enumerate() {
                let row = block.row(u32::try_from(at).unwrap_or(u32::MAX))?;
                self.backend.copy_h2d(row, bytes)?;
            }
        } else if rows > 1 {
            self.run_batch(steps)?;
        } else {
            for step in steps {
                // A decode step writes a logits row and ends no prompt: the
                // row is what the sampler reads, not a boundary to resume from.
                self.run_chunk(step.session, &[step.token], 1, false, None)?;
            }
        }
        // A session that may speculate keeps its staged row current through
        // plain steps too: the residual of the position just committed, which
        // row `i` of the pass still holds. Left stale, the row would name a
        // position the session has moved past, and the end of the turn could
        // not be checkpointed - the checkpoint carries the row.
        if self.has_draft_head() {
            for (row, step) in steps.iter().enumerate() {
                let speculates = self
                    .sessions
                    .get(&step.session)
                    .is_some_and(|live| live.spec_depth().is_speculative());
                if speculates {
                    let row = u32::try_from(row).unwrap_or(u32::MAX);
                    self.stage_draft_row(step.session, row)?;
                }
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

    /// Reads several scoring sessions' prompt tails in one prompt-role pass and
    /// returns one logits row per run, in run order: row `r` is run `r`'s last
    /// position.
    ///
    /// Every row is the row [`SparseStack::prefill_last`] would leave for that
    /// session and those tokens, and every session ends where that read would
    /// leave it - but for the draft head's cache, which this pass does not
    /// write. A solo read writes it twice: over the tail (which
    /// `warm_draft_below_boundary` skips for a scoring session's tail anyway),
    /// and, through `warm_draft_boundary`, at the position before the tail -
    /// the prefix's last position - from the staged row the prefix left. Both
    /// writes are harmless to skip here: a scoring session's tail is never
    /// drafted from, so nothing reads the head's cache at or past the
    /// prefix's last position, and the tail is never checkpointed (its prompt
    /// end is answered and dropped, and the refusals below keep every run
    /// clear of a boundary that would be), so no checkpoint carries the
    /// difference to a later session.
    ///
    /// # Errors
    ///
    /// Every refusal comes before any page is claimed, so a refused call leaves
    /// every session as it stood: [`CoreError::Unsupported`] for a session that
    /// is not scoring or carries media, a run inside which this session's own
    /// read would stop to take a checkpoint, more runs than
    /// [`MAX_PROMPT_RUNS`] or more rows than one prefill chunk, and a build
    /// whose chunked recurrence is off or unavailable (its solo reads walk the
    /// sequential kernel, which a segmented pass runs only through per-row
    /// arrays it does not pack; that one carries
    /// [`shiso_core::NO_BATCHED_PROMPT_PASS`], the others refuse the call
    /// only) - all of which a caller reads one session at a time instead;
    /// [`CoreError::ContextOverflow`]; [`CoreError::ShapeMismatch`] for no runs
    /// or an empty one; [`CoreError::InvalidValue`] for a session named twice;
    /// and whatever the pass reports.
    pub fn prefill_segments(&mut self, runs: &[(SessionId, &[TokenId])]) -> CoreResult<Logits<'_>> {
        self.check_prompt_runs(runs)?;
        let mut republished = false;
        for (session, tokens) in runs {
            republished |=
                self.reserve(*session, u32::try_from(tokens.len()).unwrap_or(u32::MAX))?;
        }
        if republished && let Some(graphs) = self.backend.as_graph() {
            graphs.invalidate_all();
        }
        let sessions = runs.iter().map(|(session, _)| *session).collect::<Vec<_>>();
        let lens = runs
            .iter()
            .map(|(_, tokens)| u32::try_from(tokens.len()).unwrap_or(u32::MAX))
            .collect::<Vec<_>>();
        let mut pass = self.plan_prompt_segments(&sessions, &lens)?;
        let gathered = self.issue_gathers(&mut pass, runs)?;
        self.write_segment_inputs(runs)?;
        // No per-row pointer arrays: every launch here that reads a session's
        // state names it by run.
        self.land_gathers(gathered)?;
        // The tap goes back on the failure path too, as in a solo chunk.
        match self.tap.take() {
            None => {
                self.execute_rows(&sessions, &pass)?;
            }
            Some(mut tap) => {
                let ran = self.execute_rows_tapped(&sessions, &pass, None, Some(&mut tap));
                self.tap = Some(tap);
                ran?;
            }
        }
        let RowOrigin::Segmented(segments) = pass.origin else {
            return Err(CoreError::Unsupported {
                what: "a batched prompt pass planned as something other than segments",
                backend: "engine",
            });
        };
        // Staged before any session advances, each from its own run's last row
        // of the residual: a solo read stages its pass's last row, and here
        // that row is `last_row(r)`, not `r`.
        if self.has_draft_head() {
            for (run, session) in (0..segments.count()).zip(&sessions) {
                self.stage_draft_row(*session, segments.last_row(run))?;
            }
        }
        for (run, (session, tokens)) in (0..segments.count()).zip(runs) {
            self.advance_at(*session, tokens, Boundary::PromptEnd, Some(run))?;
            if let Some(live) = self.sessions.get_mut(session)
                && let Some(spec) = live.spec.as_mut()
            {
                spec.staged_for = live.written;
            }
            self.mark_prompt_end(*session)?;
        }
        self.logits(segments.count())
    }

    /// The refusals of [`SparseStack::prefill_segments`], all of them before
    /// anything is claimed or launched.
    fn check_prompt_runs(&self, runs: &[(SessionId, &[TokenId])]) -> CoreResult<()> {
        if runs.is_empty() || runs.iter().any(|(_, tokens)| tokens.is_empty()) {
            return Err(CoreError::ShapeMismatch {
                field: "batched prompt runs",
                expected: 1,
                got: 0,
            });
        }
        // Each run's recurrence is the chunked one its solo read takes; with
        // that kernel off, the pass would fall to the verify's segmented walk,
        // which reads per-row pointer arrays this pass never packs.
        // A fact about the build, not this call, so it is the refusal that
        // says there is no such pass here at all.
        if self.opts.chunk.min_rows == 0 || !self.opts.chunk.available {
            return Err(CoreError::Unsupported {
                what: shiso_core::NO_BATCHED_PROMPT_PASS,
                backend: "engine",
            });
        }
        if runs.len() > MAX_PROMPT_RUNS as usize {
            return Err(CoreError::Unsupported {
                what: "a batched prompt pass of more runs than one pass carries",
                backend: "engine",
            });
        }
        let rows = runs.iter().map(|(_, tokens)| tokens.len()).sum::<usize>();
        if rows > usize::try_from(self.opts.prefill_rows).unwrap_or(usize::MAX) {
            return Err(CoreError::Unsupported {
                what: "a batched prompt pass wider than one prefill chunk",
                backend: "engine",
            });
        }
        let sessions = runs.iter().map(|(session, _)| *session).collect::<Vec<_>>();
        crate::sparse::stack::distinct(&sessions)?;
        for (session, tokens) in runs {
            let live = self.session(*session)?;
            if !live.scoring || live.media {
                return Err(CoreError::Unsupported {
                    what: "a batched prompt pass over a session that is not a text-only score",
                    backend: "engine",
                });
            }
            let next = u64::from(live.written())
                .saturating_add(u64::try_from(tokens.len()).unwrap_or(u64::MAX));
            if next > u64::from(self.opts.engine.ctx) {
                return Err(CoreError::ContextOverflow {
                    session: *session,
                    cap: self.opts.engine.ctx,
                });
            }
            if self.shared_stop(*session, tokens)?.is_some() {
                return Err(CoreError::Unsupported {
                    what: "a batched prompt run its session's own read would stop inside to \
                           take a checkpoint",
                    backend: "engine",
                });
            }
        }
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

    /// Warms the draft head's cache over `head`, which starts at `start`,
    /// below the session's [`SparseSession::warm_until`] only.
    ///
    /// A scoring session's tail past its shared boundary is never resumed and
    /// never drafted from, so the head's cache is warmed only over the part a
    /// later session can stand on.
    fn warm_draft_below_boundary(
        &mut self,
        session: SessionId,
        head: &[TokenId],
        start: u32,
    ) -> CoreResult<()> {
        let warm_until = self.session(session)?.warm_until;
        let warm_end = usize::try_from(warm_until.saturating_sub(start))
            .unwrap_or(usize::MAX)
            .min(head.len());
        if warm_end > 0 {
            self.warm_draft(session, &head[..warm_end], start)?;
        }
        Ok(())
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
        let (start, rope_shift) = {
            let live = self.session(session)?;
            (live.written(), live.rope_shift)
        };
        if media.is_some() {
            // Marked before anything runs, so a chunk that fails halfway still
            // leaves the session off disk unless a content key names it.
            if let Some(live) = self.sessions.get_mut(&session) {
                live.media = true;
            }
        }
        let mut run = media.map(|prefill| {
            MediaRun::new(
                prefill,
                start,
                rope_shift,
                crate::sparse::media::MEDIA_POSITIONS,
            )
        });
        let width = usize::try_from(self.opts.prefill_rows).unwrap_or(usize::MAX);
        let mut rest = tokens;
        while !rest.is_empty() {
            // A chunk may not end inside a placeholder run: the walk hands out
            // an item's encoder rows and its grid positions together, and a cut
            // through one would split an item's rows across two passes whose
            // maps were built from two different walks of the same item.
            let at = self.session(session)?.written();
            let mut take = match run.as_ref() {
                Some(run) => run.walk.split_point(rest, width),
                None => on_grid(at, rest.len().min(width), rest.len()),
            };
            // A chunk also stops at a boundary a later request is likely to
            // resume from - a message end, or where this prompt branches off a
            // path the cache holds - so the commit there can take a checkpoint.
            // One extra split and one snapshot per new boundary.
            let mut shared = false;
            if let Some(stop) = self.shared_stop(session, rest)?
                && stop <= take
                && (run.is_none() || self.stop_clears_media(session, at, stop)?)
            {
                take = stop;
                shared = true;
            }
            let (head, tail) = rest.split_at(take);
            let logit_rows = u32::from(tail.is_empty() && with_logits);
            // The draft head's own cache is filled as the prompt is read, and
            // the three calls around this one are in the only order that works:
            // see `warm_draft_boundary`. A build with no head does none of it.
            let start = self.session(session)?.written();
            if let Some(first) = head.first() {
                self.warm_draft_boundary(session, *first, start)?;
            }
            let boundary = if logit_rows > 0 {
                Boundary::PromptEnd
            } else if shared {
                Boundary::Shared
            } else {
                Boundary::Step
            };
            let snapshot = if logit_rows > 0 && run.is_none() {
                self.snapshot_row(session, start, head.len())?
            } else {
                None
            };
            self.run_rows_media(
                session,
                head,
                logit_rows,
                run.as_mut(),
                PassRole::Prompt { snapshot },
            )?;
            self.keep_rope_shift(session, run.as_ref());
            // A prompt that ends between two grid points is checkpointed at the
            // last one instead of at its end: a longer prompt continues from
            // there exactly as a cold prefill of it would, where a checkpoint at
            // the end would have it cut a run in two. The end itself is then an
            // ordinary span, and a repeat of this prompt re-reads the few
            // positions past the grid point.
            let boundary = match snapshot {
                Some(row) => {
                    let lead = usize::try_from(row).unwrap_or(head.len()).min(head.len());
                    if self.commit_snapshot(session, &head[..lead])? {
                        Boundary::Step
                    } else {
                        boundary
                    }
                }
                None => boundary,
            };
            // Staged after the pass and before the commit: the commit is where
            // a prompt's final chunk earns its checkpoint, and a speculating
            // session's checkpoint carries this row - staged any later, the
            // first chunk had none to carry and was never checkpointed, and a
            // later chunk carried the chunk before it's. And before the warm-up
            // reads the residual, because the warm-up is what overwrites it.
            if self.has_draft_head() {
                let last = u32::try_from(head.len().saturating_sub(1)).unwrap_or(0);
                self.stage_draft_row(session, last)?;
            }
            self.advance_at(session, head, boundary, logit_rows.checked_sub(1))?;
            // The row staged before the advance is the residual of the chunk's
            // last position, so it drafts from the frontier the advance just
            // reached.
            if let Some(live) = self.sessions.get_mut(&session)
                && let Some(spec) = live.spec.as_mut()
            {
                spec.staged_for = live.written;
            }
            self.warm_draft_below_boundary(session, head, start)?;
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
        run.as_ref().map_or(Ok(()), every_row_consumed)
    }

    /// Records where a media chunk's walk left the rotary stream: whatever
    /// reads this session next - the rest of the prompt as text, a decode
    /// step, a verify - rotates from there. Mid-item the number is transient,
    /// and nothing reads it there: a prompt never ends inside a run.
    fn keep_rope_shift(&mut self, session: SessionId, run: Option<&MediaRun<'_>>) {
        if let Some(run) = run
            && let Some(live) = self.sessions.get_mut(&session)
        {
            live.rope_shift = run.walk.shift();
        }
    }

    /// Whether a media prompt's chunk may stop `stop` tokens past `at`: only
    /// under a content key, which is the only way the checkpoint taken there
    /// can be stored, and only outside every picture, because the walk hands
    /// out an item's rows and positions in one pass.
    fn stop_clears_media(&self, session: SessionId, at: u32, stop: usize) -> CoreResult<bool> {
        let pos = at.saturating_add(u32::try_from(stop).unwrap_or(u32::MAX));
        Ok(self
            .session(session)?
            .key
            .as_ref()
            .is_some_and(|key| !key.spans.iter().any(|span| span.splits(pos))))
    }

    /// Tokens of `rest` to run before the next shared boundary, when one falls
    /// inside it: a message-boundary hint, or the point where this path stops
    /// matching the tree.
    ///
    /// A boundary closer than the tier's floor to the last checkpoint on this
    /// path is skipped - a checkpoint is worth the prefill between it and the
    /// one below it, and a few hundred tokens of that is not worth a snapshot -
    /// and so is everything on a stack with no disk tier, which has nowhere to
    /// resume a later request from. `None` is not a refusal: the chunk simply
    /// runs to its own width.
    fn shared_stop(&self, session: SessionId, rest: &[TokenId]) -> CoreResult<Option<usize>> {
        let Some(ckpt) = self.ckpt else {
            return Ok(None);
        };
        let live = self.session(session)?;
        let pos = live.written();
        let len = pos.saturating_add(u32::try_from(rest.len()).unwrap_or(u32::MAX));
        let path = live.cache_path_with(rest);
        let branch = self.resident()?.cache.branch_point(&path);
        // Every candidate is judged on its own: a branch point a few tokens into
        // a shared template opening is too close to be worth a snapshot, and it
        // must not hide the message boundary a thousand tokens later.
        let floor = ckpt.options.min_tokens.max(1);
        let lowest = pos.max(live.resumed_at).saturating_add(floor).max(floor);
        // A media score's own boundaries - where its picture starts and where
        // its sibling questions part - are the host's call, and its siblings
        // stand on exactly those, so they are taken wherever they fall. A text
        // score's keep the floor: its siblings read together in one batched
        // pass, which a stop inside a run would refuse.
        let media_score = live.scoring && live.media;
        let own_lowest = if media_score {
            pos.saturating_add(1)
        } else {
            lowest
        };
        let clears = |at: u32| {
            !live.media
                || live
                    .key
                    .as_ref()
                    .is_some_and(|key| !key.spans.iter().any(|span| span.splits(at)))
        };
        // Each is moved back onto the recurrence's grid first: a stop between
        // two grid points would cut a run, and every later pass of this prompt -
        // and of every prompt resumed from the checkpoint taken here - would sum
        // differently from a cold prefill of the same tokens. The resume re-reads
        // the few positions between the grid point and the boundary instead. In
        // a media prompt a boundary whose grid point lies inside a picture - the
        // end of a shared picture, where sibling questions part - is taken where
        // it is rather than dropped, because the alternative is every sibling
        // reading the picture again.
        let place = |stop: u32, lowest: u32| {
            let grid = stop - stop % GDN_CHUNK;
            let exact = live.media.then_some(stop);
            std::iter::once(grid)
                .chain(exact)
                .find(|at| *at > pos && *at < len && *at >= lowest && clears(*at))
        };
        let stop = live
            .hints
            .iter()
            .filter_map(|hint| place(*hint, own_lowest))
            .chain(place(branch, lowest))
            .min();
        Ok(stop.and_then(|stop| usize::try_from(stop - pos).ok()))
    }

    /// The row of a prompt's final pass at which its state is snapshotted, when
    /// it gets one: the pass starts on the grid, ends off it, and reaches a grid
    /// point in between - and a tier is open to keep the snapshot, and the
    /// chunked recurrence, the only one that writes one, is what the pass runs.
    /// `None` is not a refusal: the prompt's end is checkpointed as it is.
    fn snapshot_row(&self, session: SessionId, start: u32, rows: usize) -> CoreResult<Option<u32>> {
        let has_buffer = self
            .resident
            .as_ref()
            .is_some_and(|resident| resident.snapshot.is_some());
        // A scoring session's prompt end is never checkpointed, so neither
        // is the grid point that stands in for it.
        let live = self.session(session)?;
        if self.ckpt.is_none() || !has_buffer || live.media || live.scoring {
            return Ok(None);
        }
        let chunk = self.opts.chunk;
        if chunk.min_rows == 0 || !chunk.available {
            return Ok(None);
        }
        let end = start.saturating_add(u32::try_from(rows).unwrap_or(u32::MAX));
        let grid = GDN_CHUNK;
        let below = end - end % grid;
        Ok(
            (start.is_multiple_of(grid) && !end.is_multiple_of(grid) && below > start)
                .then(|| below - start),
        )
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
        self.run_rows_media(session, tokens, logit_rows, media, PassRole::Step)?;
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
        self.run_rows_media(session, tokens, logit_rows, None, PassRole::Step)
    }

    /// The same pass, splicing a request's encoder rows over its placeholder
    /// tokens when the chunk carries any. `role` says which reference pass its
    /// rows must reproduce: see [`PassRole`].
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
        role: PassRole,
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
            role,
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
            issued.push(gather.fetch(tier.table().stream(), &staging.path)?);
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

/// Whether a finished media prompt's walk handed out every encoder row.
///
/// The count the items carry is the count the rendered prompt used, and nothing
/// re-expands a placeholder run. So the walk having handed out every row is the
/// statement that the prompt and the tower described the same request - the one
/// mismatch that otherwise succeeds, stays fluent and answers about a rearranged
/// picture.
fn every_row_consumed(run: &MediaRun<'_>) -> CoreResult<()> {
    let consumed = u64::from(run.walk.rows_consumed());
    let available = u64::from(run.prefill.row_count());
    if consumed == available {
        Ok(())
    } else {
        Err(CoreError::ShapeMismatch {
            field: "vision placeholder tokens",
            expected: available,
            got: consumed,
        })
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

/// Tokens of a piece that starts at position `start`, wants `want` and has
/// `rest` to go: ending on the recurrence's run grid unless it ends the call.
///
/// The recurrent blocks' chunked walk counts its runs from the pass's first
/// row, so a pass that started between two grid points would cut every run of
/// its prompt in two - summed differently from a cold prefill of the same
/// tokens, which starts on the grid. Rounded down, so a piece never grows past
/// the pass width; a start off the grid - a resume that stood on a turn's end -
/// takes the short piece up to the next grid point and is on the grid after it.
fn on_grid(start: u32, want: usize, rest: usize) -> usize {
    if want >= rest {
        return rest;
    }
    let grid = u64::from(GDN_CHUNK);
    let end = u64::from(start).saturating_add(want as u64);
    let down = end - end % grid;
    let take = if down > u64::from(start) {
        down - u64::from(start)
    } else {
        (u64::from(start) / grid + 1) * grid - u64::from(start)
    };
    usize::try_from(take).unwrap_or(want).min(rest)
}

//! Several scores' tails in one prompt pass.
//!
//! # Who reads, and in what order
//!
//! The planner hands over a batch only when the first sequence with prompt
//! left is itself a score whose tail fits the quantum, so a generation at the
//! front is never batched past. Here the batch's head - that lead - is read
//! this tick whatever its siblings do: a lead its resume did not stand on the
//! shared prefix reads alone, and the batch waits a tick. Starving the lead
//! would starve everything admitted after it, generations included.
//!
//! # Waiting for a sibling's checkpoint
//!
//! Siblings are admitted together, and at the first quantum none of them has
//! a checkpoint to stand on. The first reads alone and writes it at its stand
//! point; the rest are put back to resume again. They ask again only once
//! some read has crossed a stand point since their last miss - the
//! [`ScorePass::shared_reads`] epoch - so a family waiting on a long first
//! read costs nothing per tick rather than a resume per sibling per tick.
//!
//! # A model without the pass
//!
//! A model that refuses the pass with [`NO_BATCHED_PROMPT_PASS`] has no such
//! pass - the trait's default, or a build that cannot run it - and the planner
//! stops offering batches to it. Any other refusal refuses that call only,
//! whether or not a pass was answered before, and the batch is offered again
//! on a later tick. Either way the refused tick reads only its lead, so a
//! refusal never widens a tick past a solo read.

use std::sync::Arc;
use std::time::Instant;

use shiso_core::{CoreError, NO_BATCHED_PROMPT_PASS, PromptRun};

use super::batching::{self, PrefillStep, SlotFacts};
use super::lifecycle::SeqSlot;
use super::{Coordinator, Quantum, finish_scoring};

/// What the batched prompt pass has shown on this build so far.
#[derive(Debug, Default)]
pub(super) struct ScorePass {
    /// Reads that crossed a score's stand point, each of which may have
    /// written the checkpoint its siblings wait on.
    pub(super) shared_reads: u64,
    /// Whether the model has answered a batched pass.
    answered: bool,
    /// Whether it said it has no such pass at all: the planner stops
    /// offering batches.
    refused: bool,
}

impl Coordinator {
    /// What the planner is told about every resident sequence: as they are,
    /// except that on a model without the batched pass no score is offered
    /// for one.
    pub(super) fn planned_facts(&self) -> Vec<SlotFacts> {
        let mut facts = self.slots.facts();
        if self.score_pass.refused {
            for fact in &mut facts {
                fact.score = None;
            }
        }
        facts
    }

    /// One sequence's quantum, read alone, noting a read that crossed a
    /// score's stand point: the checkpoint its siblings wait on may now exist.
    pub(super) fn read_alone(&mut self, step: PrefillStep) {
        let before = self.slots.slots().get(step.slot).and_then(|slot| {
            slot.score_stand_point()
                .map(|point| (point, slot.prompt_cursor()))
        });
        self.prefill_step(step);
        if let Some((point, from)) = before
            && from < point
            && self
                .slots
                .slots()
                .get(step.slot)
                .is_some_and(|slot| slot.prompt_cursor() >= point)
        {
            self.score_pass.shared_reads += 1;
        }
    }

    /// Several scores' last quanta in one prompt pass.
    ///
    /// Every member resumes first, as its solo quantum would, except one
    /// whose last resume missed with no shared read since. Members that then
    /// stand on their shared prefix are read together. A lead that does not
    /// stand reads alone instead, and so does the first member that missed
    /// when none stands. Every other member that missed is put back to
    /// resume again. One prompt read a tick either way.
    pub(super) fn prefill_batch_step(&mut self, steps: &[PrefillStep], quantum: usize) {
        let epoch = self.score_pass.shared_reads;
        for step in steps {
            if let Some(shared) = self
                .slots
                .get_mut(step.slot)
                .and_then(|slot| slot.take_score_resume_at(epoch))
            {
                self.resume_score(step.slot, shared);
            }
        }
        let (runs, missed) = self.score_runs(steps, quantum);
        let lead = steps.first().map(|step| step.slot);
        let lead_waits = lead.is_some_and(|at| {
            !runs.iter().any(|run| run.step.slot == at)
                && self
                    .slots
                    .slots()
                    .get(at)
                    .is_some_and(|slot| !slot.is_done() && !slot.prompt_done())
        });
        let leader = if lead_waits {
            lead
        } else if runs.is_empty() {
            missed.first().copied()
        } else {
            None
        };
        for at in &missed {
            if Some(*at) != leader
                && let Some(slot) = self.slots.get_mut(*at)
            {
                slot.rearm_score_resume(epoch);
            }
        }
        if let Some(at) = leader {
            self.read_leader(at, quantum);
        } else if let [only] = runs.as_slice() {
            self.read_alone(only.step);
        } else if runs.len() > 1 {
            self.run_score_batch(&runs);
        }
    }

    /// The members that stand on their shared prefix, as quanta of their
    /// whole remaining prompts within `quantum`, and the members that do not.
    fn score_runs(&self, steps: &[PrefillStep], quantum: usize) -> (Vec<Quantum>, Vec<usize>) {
        let mut runs: Vec<Quantum> = Vec::new();
        let mut missed: Vec<usize> = Vec::new();
        let mut rows = 0_usize;
        for step in steps {
            let Some(slot) = self.slots.slots().get(step.slot) else {
                continue;
            };
            if slot.is_done() || slot.prompt_done() {
                continue;
            }
            if !slot.score_stands_on_shared() {
                missed.push(step.slot);
                continue;
            }
            let (start, end) = slot.prompt_window(usize::MAX);
            let tokens = end - start;
            if runs.len() == batching::SCORE_BATCH_RUNS || rows.saturating_add(tokens) > quantum {
                continue;
            }
            rows += tokens;
            runs.push(Quantum {
                step: PrefillStep {
                    slot: step.slot,
                    tokens,
                    last: true,
                },
                session: slot.session,
                prompt: Arc::clone(&slot.prompt),
                window: start..end,
            });
        }
        (runs, missed)
    }

    /// The quantum one score reads alone this tick: its solo quantum, after
    /// the resume it still has pending, unless that resume already missed with
    /// no shared read since.
    fn read_leader(&mut self, at: usize, quantum: usize) {
        let epoch = self.score_pass.shared_reads;
        if let Some(shared) = self
            .slots
            .get_mut(at)
            .and_then(|slot| slot.settle_score_resume(epoch))
        {
            self.resume_score(at, shared);
        }
        let Some(fact) = self.slots.slots().get(at).map(SeqSlot::facts) else {
            return;
        };
        let tokens = batching::on_grid(fact.prompt_at, fact.prompt_remaining, quantum);
        self.read_alone(PrefillStep {
            slot: at,
            tokens,
            last: tokens == fact.prompt_remaining,
        });
    }

    /// One prompt pass over `runs`, each a score's whole remaining prompt,
    /// each answered from its own row. A refusal reads the first run alone
    /// and leaves the rest for later ticks; any other failure fails exactly
    /// these sequences, whose replies retirement then sends, and a run whose
    /// row cannot be read back fails alone while the others answer.
    fn run_score_batch(&mut self, runs: &[Quantum]) {
        let started = Instant::now();
        let decode_steps = self.decode_steps;
        let outcome = {
            let Self {
                model,
                backend,
                slots,
                row,
                scores,
                ..
            } = self;
            let pieces = runs
                .iter()
                .map(|run| PromptRun {
                    session: run.session,
                    tokens: &run.prompt[run.window.clone()],
                })
                .collect::<Vec<_>>();
            model
                .as_model_mut()
                .prefill_last_many(&pieces)
                .map(|logits| {
                    for (at, run) in (0_u32..).zip(runs) {
                        let Some(slot) = slots.get_mut(run.step.slot) else {
                            continue;
                        };
                        let read = logits
                            .row(at)
                            .and_then(|view| backend.copy_d2h(view, row.as_mut_slice()));
                        if let Err(error) = read {
                            slot.fail(error.to_string());
                            continue;
                        }
                        slot.advance_prefill(run.step.tokens);
                        slot.time_prefill(run.step.tokens, started.elapsed(), decode_steps);
                        finish_scoring(row, scores, slot);
                    }
                })
        };
        match outcome {
            Ok(()) => {
                let rows = runs.iter().map(|run| run.step.tokens).sum::<usize>();
                tracing::debug!(
                    runs = runs.len(),
                    rows,
                    "score tails read in one prompt pass"
                );
                self.score_pass.answered = true;
                self.stats.score_pass(runs.len() as u64);
                self.stats.prefilled(rows as u64);
            }
            Err(CoreError::Unsupported { what, .. }) => {
                if what == NO_BATCHED_PROMPT_PASS {
                    tracing::warn!(
                        what,
                        answered = self.score_pass.answered,
                        "the model has no batched prompt pass; score tails are read one at a time"
                    );
                    self.score_pass.refused = true;
                } else {
                    tracing::debug!(what, "a batched prompt pass was refused; reading alone");
                }
                if let Some(first) = runs.first() {
                    self.read_alone(first.step);
                }
            }
            Err(error) => {
                for run in runs {
                    self.fail_slot(run.step.slot, &error, "a batched prompt pass failed");
                }
            }
        }
    }
}

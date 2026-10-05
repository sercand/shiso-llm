//! Who runs this tick, and how much prompt gets read while they do.
//!
//! # The two things one tick may contain
//!
//! A tick is at most one batched decode step over every sequence that is ready
//! to decode, followed by at most one bounded prefill quantum for one sequence
//! that is still reading its prompt. That order is the whole fairness rule: a
//! sequence that is mid-answer gets its token before any prompt is read, so a
//! long prompt arriving cannot stall an answer that is already streaming, and a
//! sequence that is still prefilling can never starve a decoder for longer than
//! one quantum.
//!
//! # Why the prefill is bounded and the solo case is not
//!
//! A forward pass is a device call nothing interrupts. Running a whole
//! ten-thousand-token prompt as one call adds that whole call's latency to every
//! sequence decoding beside it, which is exactly the stutter batching is
//! supposed to remove. So while anything is decoding the prompt is read in
//! quanta. With nothing else running there is nobody to starve, and small chunks
//! buy only more launches, so the width goes up.
//!
//! # The one prompt that is not divided
//!
//! A sequence carrying media reads its whole prompt in one quantum. That is not
//! a fairness exemption anybody chose: a media prefill's position stream and its
//! splice cursor are running state that the model's contract rebuilds per call,
//! so a prompt split across two calls is a prompt whose second half is rotated
//! and spliced from the wrong place. The cost is bounded by what a picture costs
//! in tokens and is paid by whatever else is decoding at the time; the
//! alternative is a wrong answer nobody can see.
//!
//! # Flex sequences run only in standard traffic's absence
//!
//! A flex sequence is left out of a tick - no decode row, no prompt quantum -
//! whenever a standard sequence has work, and planned like any other when none
//! does. Leaving it out is the whole of pausing it: its session and its cache
//! stay resident, nothing is thrown away, and the first tick after the standard
//! work drains picks it up where it stood. The pause lands between two calls
//! because a tick is the only place a call is chosen, so a flex prompt that was
//! mid-read finishes the quantum it was in and no more.
//!
//! # Why over-admission is impossible rather than checked
//!
//! The slot table owns its sequences in a `Vec` with a maximum it will not push
//! past, so "more resident sessions than the configuration allows" is not a
//! state the type can hold. There is no counter to keep in step with a list, and
//! therefore no way for the two to disagree.

use crate::scheduler::config::{PREFILL_GRID, SchedulerConfig};
use crate::scheduler::lifecycle::SeqSlot;

/// What the planner needs to know about one resident sequence.
///
/// A summary rather than the sequence itself, so that the planning rule is a
/// pure function of a handful of numbers and can be tested without a model, a
/// session or a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotFacts {
    /// Whether this sequence is standing on a sampled token and wants a step.
    pub(crate) ready_to_decode: bool,
    /// Prompt tokens it has still not read. Zero once prefill is done.
    pub(crate) prompt_remaining: usize,
    /// The position of the next prompt token it reads, which is where its next
    /// quantum starts: past a resumed prefix, not necessarily on the grid.
    pub(crate) prompt_at: usize,
    /// Whether its prompt has to be read in a single quantum.
    ///
    /// True for a sequence carrying media. The contract a model's media path
    /// offers takes a whole prompt, because the position stream a placeholder
    /// run produces is not the linear one and neither is the encoder row the
    /// splice has reached: both are running state, and a prompt handed over in
    /// pieces would restart them at every piece. A request that pays for that is
    /// one whose answer is fluent and about a rearranged picture, so the split
    /// the planner would otherwise choose is not available rather than merely
    /// discouraged.
    pub(crate) indivisible_prompt: bool,
    /// Whether this is a flex sequence, which runs only when no standard one
    /// has work.
    pub(crate) flex: bool,
    /// What a score's next read covers. `None` for a generation, and for a
    /// score that is not reading its prompt.
    pub(crate) score: Option<ScoreFacts>,
}

/// What the planner knows about a score sequence's next read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScoreFacts {
    /// Prompt rows its next read covers if that read ends the prompt: what is
    /// left, or - before the resume its first quantum makes - what is left
    /// past the grid point below its shared prefix, where a sibling's
    /// checkpoint would stand it.
    pub(crate) tail: usize,
}

/// Runs one batched prompt pass carries at most.
pub(crate) const SCORE_BATCH_RUNS: usize = shiso_engine::sparse::MAX_PROMPT_RUNS as usize;

/// The widest prompt read a tick may make: narrow beside decoding sequences,
/// which it would otherwise stall, and wide with nobody to starve.
pub(crate) const fn quantum(decoding: bool, cfg: &SchedulerConfig) -> usize {
    if decoding {
        cfg.mixed_quantum
    } else {
        cfg.solo_chunk
    }
}

/// The score tails one pass reads: in admission order, each fitting what is
/// left of `quantum`, none paused.
fn score_batch(
    facts: &[SlotFacts],
    runs: &impl Fn(&SlotFacts) -> bool,
    quantum: usize,
) -> Vec<PrefillStep> {
    let mut batch = Vec::new();
    let mut rows = 0_usize;
    for (at, fact) in facts.iter().enumerate() {
        if batch.len() == SCORE_BATCH_RUNS {
            break;
        }
        let Some(score) = fact.score else { continue };
        let fits = score.tail > 0 && rows.saturating_add(score.tail) <= quantum;
        if fact.prompt_remaining == 0 || fact.indivisible_prompt || !runs(fact) || !fits {
            continue;
        }
        rows += score.tail;
        batch.push(PrefillStep {
            slot: at,
            tokens: score.tail,
            last: true,
        });
    }
    batch
}

impl SlotFacts {
    /// Whether this sequence wants anything from the device this tick.
    const fn has_work(&self) -> bool {
        self.ready_to_decode || self.prompt_remaining > 0
    }
}

/// One sequence's share of a tick's prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillStep {
    /// Which resident sequence reads.
    pub(crate) slot: usize,
    /// How many of its prompt tokens this quantum covers.
    pub(crate) tokens: usize,
    /// Whether this quantum finishes the prompt, which is the chunk that
    /// produces logits.
    pub(crate) last: bool,
}

/// What one tick will do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TickPlan {
    /// The resident sequences that get a decode step, in slot order.
    pub(crate) decode: Vec<usize>,
    /// The one sequence that reads prompt this tick, if any.
    pub(crate) prefill: Option<PrefillStep>,
    /// Several scores' last quanta in one prompt pass; when it is non-empty,
    /// `prefill` is `None`.
    pub(crate) prefill_batch: Vec<PrefillStep>,
    /// The flex sequences that had work and were left out for standard work,
    /// in slot order.
    pub(crate) paused: Vec<usize>,
}

impl TickPlan {
    /// Whether this tick would do nothing at all.
    ///
    /// Never true while something is paused: a flex sequence is only paused
    /// for a standard one with work, and that one is in the plan.
    pub(crate) const fn is_idle(&self) -> bool {
        self.decode.is_empty() && self.prefill.is_none() && self.prefill_batch.is_empty()
    }
}

/// Chooses what one tick does.
///
/// Pure, and deliberately so: the fairness rule is the scheduler's most
/// load-bearing claim and it should be assertable against a list of numbers
/// rather than against a running engine.
pub(crate) fn plan(facts: &[SlotFacts], cfg: &SchedulerConfig) -> TickPlan {
    let standard_busy = facts.iter().any(|fact| !fact.flex && fact.has_work());
    let runs = |fact: &SlotFacts| !(standard_busy && fact.flex);
    let paused: Vec<usize> = facts
        .iter()
        .enumerate()
        .filter_map(|(at, fact)| (!runs(fact) && fact.has_work()).then_some(at))
        .collect();
    // The decode set is chosen after the flex rows are left out, so the prefill
    // width below is chosen from it too: a paused row decodes nothing, and a
    // standard prompt read beside it has nobody to starve.
    let decode: Vec<usize> = facts
        .iter()
        .enumerate()
        .filter_map(|(at, fact)| (fact.ready_to_decode && runs(fact)).then_some(at))
        .collect();

    // The first sequence with prompt left, in admission order. Round-robin
    // across prefilling sequences would interleave two prompts into the same
    // number of quanta while finishing neither sooner, and the one that started
    // first is the one whose client has been waiting longest.
    let quantum = quantum(!decode.is_empty(), cfg);
    let lead = facts
        .iter()
        .enumerate()
        .find(|(_at, fact)| fact.prompt_remaining > 0 && runs(fact));
    // Score tails are read together only when the lead is itself a score whose
    // tail fits: a generation at the front reads alone, exactly as before, and
    // a batch is bounded by the same quantum a solo read would be.
    let batches = lead.is_some_and(|(_at, fact)| {
        !fact.indivisible_prompt
            && fact
                .score
                .is_some_and(|score| score.tail > 0 && score.tail <= quantum)
    });
    let prefill_batch = if batches {
        score_batch(facts, &runs, quantum)
    } else {
        Vec::new()
    };
    // The lead heads every batch - it is the first score that fits - so a
    // batch never reads a later prompt instead of the lead's.
    let headed = prefill_batch.first().map(|step| step.slot) == lead.map(|(at, _fact)| at);
    let prefill_batch = if prefill_batch.len() > 1 && headed {
        prefill_batch
    } else {
        Vec::new()
    };
    let prefill = if prefill_batch.is_empty() {
        lead.map(|(at, fact)| {
            let tokens = if fact.indivisible_prompt {
                fact.prompt_remaining
            } else {
                on_grid(fact.prompt_at, fact.prompt_remaining, quantum)
            };
            PrefillStep {
                slot: at,
                tokens,
                last: tokens == fact.prompt_remaining,
            }
        })
    } else {
        None
    };

    TickPlan {
        decode,
        prefill,
        prefill_batch,
        paused,
    }
}

/// Tokens one quantum reads from `at` with `remaining` to go: at most
/// `quantum`, ending on [`PREFILL_GRID`] unless it ends the prompt.
///
/// Rounded down to the grid, so a quantum never grows past its width - except
/// when the grid point below lies behind `at`, which only a quantum narrower
/// than the grid or a start off it reaches: then it runs to the grid point
/// above, which is at most one grid step, rather than cut a run.
pub(crate) fn on_grid(at: usize, remaining: usize, quantum: usize) -> usize {
    let want = remaining.min(quantum.max(1));
    if want == remaining {
        return want;
    }
    let end = at.saturating_add(want);
    let down = end - end % PREFILL_GRID;
    let take = if down > at {
        down - at
    } else {
        (at / PREFILL_GRID + 1) * PREFILL_GRID - at
    };
    take.min(remaining)
}

/// The resident sequences, bounded.
///
/// This is the residency policy in one type. A sequence occupies a slot from the
/// moment its session is opened to the one call that closes it, and there is no
/// path that takes a slot away from a live sequence to give it to a newer one:
/// the bound is enforced at admission, where a request simply waits, rather than
/// mid-turn, where it would mean throwing away work already paid for.
#[derive(Debug)]
pub(crate) struct SlotTable {
    slots: Vec<SeqSlot>,
    max: usize,
}

impl SlotTable {
    /// A table with room for `max` sequences.
    pub(crate) fn new(max: usize) -> Self {
        let max = max.max(1);
        Self {
            slots: Vec::with_capacity(max),
            max,
        }
    }

    /// How many sequences are resident.
    pub(crate) const fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether nothing is resident.
    pub(crate) const fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Whether another sequence would fit.
    pub(crate) const fn has_room(&self) -> bool {
        self.slots.len() < self.max
    }

    /// Takes a slot, or hands the sequence back when there is no room.
    ///
    /// Returning the argument rather than a `bool` is what makes the caller
    /// unable to drop a sequence it believed was admitted: there is no way to
    /// ignore the refusal and still have the value.
    pub(crate) fn admit(&mut self, slot: SeqSlot) -> Option<SeqSlot> {
        if !self.has_room() {
            return Some(slot);
        }
        self.slots.push(slot);
        None
    }

    /// The resident sequences.
    pub(crate) fn slots(&self) -> &[SeqSlot] {
        &self.slots
    }

    /// The resident sequences, mutably.
    pub(crate) fn slots_mut(&mut self) -> &mut [SeqSlot] {
        &mut self.slots
    }

    /// One resident sequence.
    pub(crate) fn get_mut(&mut self, at: usize) -> Option<&mut SeqSlot> {
        self.slots.get_mut(at)
    }

    /// What the planner needs to know about every resident sequence.
    pub(crate) fn facts(&self) -> Vec<SlotFacts> {
        self.slots.iter().map(SeqSlot::facts).collect()
    }

    /// Removes and returns every sequence that has stopped.
    ///
    /// Two-phase: the finished ones come out and the survivors stay, in order,
    /// so that a slot index taken before a retirement is never reused for a
    /// different sequence within the same tick.
    pub(crate) fn take_finished(&mut self) -> Vec<SeqSlot> {
        let mut finished = Vec::new();
        let mut survivors = Vec::with_capacity(self.slots.len());
        for slot in self.slots.drain(..) {
            if slot.is_done() {
                finished.push(slot);
            } else {
                survivors.push(slot);
            }
        }
        self.slots = survivors;
        finished
    }

    /// Removes and returns every sequence, for the shutdown path.
    pub(crate) fn drain(&mut self) -> Vec<SeqSlot> {
        std::mem::take(&mut self.slots)
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefillStep, ScoreFacts, SlotFacts, on_grid, plan};
    use crate::scheduler::config::{PREFILL_GRID, SchedulerConfig};

    fn cfg() -> SchedulerConfig {
        SchedulerConfig {
            mixed_quantum: 128,
            solo_chunk: 512,
            ..SchedulerConfig::new()
        }
    }

    fn decoding() -> SlotFacts {
        SlotFacts {
            ready_to_decode: true,
            prompt_remaining: 0,
            prompt_at: 0,
            indivisible_prompt: false,
            flex: false,
            score: None,
        }
    }

    fn prefilling(remaining: usize) -> SlotFacts {
        SlotFacts {
            ready_to_decode: false,
            prompt_remaining: remaining,
            prompt_at: 0,
            indivisible_prompt: false,
            flex: false,
            score: None,
        }
    }

    fn flex(fact: SlotFacts) -> SlotFacts {
        SlotFacts { flex: true, ..fact }
    }

    fn finished() -> SlotFacts {
        SlotFacts {
            ready_to_decode: false,
            prompt_remaining: 0,
            ..decoding()
        }
    }

    /// The flex rule: while a standard sequence has work, a flex one gets
    /// neither a decode row nor a prompt quantum, and says it was paused.
    #[test]
    fn a_flex_sequence_is_left_out_while_a_standard_one_decodes() {
        let facts = [flex(decoding()), decoding(), flex(prefilling(1000))];
        let plan = plan(&facts, &cfg());
        assert_eq!(plan.decode, vec![1]);
        assert_eq!(plan.prefill, None, "the flex prompt is not read either");
        assert_eq!(plan.paused, vec![0, 2]);
    }

    #[test]
    fn a_flex_sequence_is_left_out_while_a_standard_prompt_is_read() {
        let facts = [flex(decoding()), prefilling(1000)];
        let plan = plan(&facts, &cfg());
        assert!(plan.decode.is_empty());
        assert_eq!(
            plan.prefill,
            Some(PrefillStep {
                slot: 1,
                tokens: 512,
                last: false
            }),
            "a paused row decodes nothing, so the standard prompt has nobody to \
             starve and takes the solo width"
        );
        assert_eq!(plan.paused, vec![0]);
    }

    /// With no standard work - none resident, or only ones that have stopped -
    /// flex sequences are planned exactly like standard ones.
    #[test]
    fn flex_sequences_run_when_no_standard_one_has_work() {
        let facts = [flex(decoding()), finished(), flex(prefilling(1000))];
        let plan = plan(&facts, &cfg());
        assert_eq!(plan.decode, vec![0]);
        assert_eq!(
            plan.prefill,
            Some(PrefillStep {
                slot: 2,
                tokens: 128,
                last: false
            })
        );
        assert!(plan.paused.is_empty());
    }

    /// A tick with something paused is never idle, which is what keeps the
    /// idle path from ending a paused sequence as stuck.
    #[test]
    fn a_plan_that_pauses_something_is_not_idle() {
        let facts = [flex(decoding()), decoding()];
        let plan = plan(&facts, &cfg());
        assert!(!plan.paused.is_empty());
        assert!(!plan.is_idle());
    }

    /// A prompt carrying media is read whole, whatever the quantum would have
    /// been - including while something else is decoding, which is the case the
    /// mixed quantum exists for.
    fn prefilling_media(remaining: usize) -> SlotFacts {
        SlotFacts {
            indivisible_prompt: true,
            ..prefilling(remaining)
        }
    }

    #[test]
    fn a_media_prompt_is_read_in_one_quantum_however_wide_it_is() {
        let facts = [prefilling_media(1000)];
        let plan = plan(&facts, &cfg());
        assert_eq!(
            plan.prefill,
            Some(PrefillStep {
                slot: 0,
                tokens: 1000,
                last: true
            }),
            "the solo width does not apply to a prompt that cannot be split"
        );
    }

    #[test]
    fn a_media_prompt_is_still_read_whole_beside_a_decoding_sequence() {
        let facts = [decoding(), prefilling_media(1000)];
        let plan = plan(&facts, &cfg());
        assert_eq!(plan.decode, vec![0]);
        assert_eq!(
            plan.prefill,
            Some(PrefillStep {
                slot: 1,
                tokens: 1000,
                last: true
            }),
            "the mixed quantum does not apply either; the split would be wrong, not slow"
        );
    }

    /// The fairness rule: everything that can decode does, in the same tick as
    /// the prefill quantum rather than after it.
    #[test]
    fn every_decoding_sequence_gets_its_step_in_the_tick_that_also_reads_prompt() {
        let facts = [decoding(), prefilling(1000), decoding()];
        let plan = plan(&facts, &cfg());
        assert_eq!(plan.decode, vec![0, 2]);
        assert_eq!(
            plan.prefill,
            Some(PrefillStep {
                slot: 1,
                tokens: 128,
                last: false
            }),
            "a shared tick reads at most one mixed quantum"
        );
    }

    /// With nobody to starve, the chunk widens - and the numbers are the two
    /// configured ones, not a third derived by accident.
    #[test]
    fn a_prefill_with_nothing_decoding_takes_the_solo_width() {
        let facts = [prefilling(1000)];
        let plan = plan(&facts, &cfg());
        assert_eq!(
            plan.prefill,
            Some(PrefillStep {
                slot: 0,
                tokens: 512,
                last: false
            })
        );
        assert!(plan.decode.is_empty());
    }

    /// The quantum never overruns the prompt, and the chunk that reaches the end
    /// is marked - it is the one that produces logits, and getting that wrong
    /// means either a missing first token or a forward pass with nothing to read.
    #[test]
    fn the_last_quantum_is_exactly_the_remainder_and_says_so() {
        let facts = [prefilling(3)];
        let plan = plan(&facts, &cfg());
        assert_eq!(
            plan.prefill,
            Some(PrefillStep {
                slot: 0,
                tokens: 3,
                last: true
            })
        );
    }

    /// Only one sequence reads prompt per tick, so two long prompts cannot
    /// combine into one unbounded forward pass.
    #[test]
    fn two_prompts_do_not_share_a_tick() {
        let facts = [prefilling(500), prefilling(500)];
        let plan = plan(&facts, &cfg());
        assert_eq!(plan.prefill.map(|step| step.slot), Some(0));
    }

    #[test]
    fn a_table_of_finished_sequences_plans_nothing() {
        let facts = [SlotFacts {
            indivisible_prompt: false,
            ready_to_decode: false,
            prompt_remaining: 0,
            prompt_at: 0,
            flex: false,
            score: None,
        }];
        assert!(plan(&facts, &cfg()).is_idle());
        assert!(plan(&[], &cfg()).is_idle());
    }

    /// A prompt that resumed off the grid reaches it with its first quantum and
    /// stays on it: every later cut lands where a cold prefill of the same
    /// prompt could have cut too.
    #[test]
    fn a_quantum_after_an_off_grid_resume_ends_on_the_grid() {
        let facts = [SlotFacts {
            prompt_at: 1_000,
            ..prefilling(5_000)
        }];
        let step = plan(&facts, &cfg()).prefill.expect("a prefill");
        assert_eq!((1_000 + step.tokens) % PREFILL_GRID, 0);
        assert!(step.tokens <= 512, "rounded down, never past the width");
    }

    /// A width off the grid is rounded down to it, and one narrower than the
    /// grid runs to the next grid point instead of cutting a run.
    #[test]
    fn a_quantum_is_rounded_to_the_grid_and_never_below_one_step() {
        assert_eq!(on_grid(0, 10_000, 1_000), 960);
        assert_eq!(on_grid(0, 10_000, 8), PREFILL_GRID);
        assert_eq!(on_grid(60, 10_000, 8), 4);
        assert_eq!(
            on_grid(60, 3, 8),
            3,
            "the end of the prompt is never rounded"
        );
        assert_eq!(on_grid(64, 100, 512), 100);
    }

    fn scoring(tail: usize) -> SlotFacts {
        SlotFacts {
            score: Some(ScoreFacts { tail }),
            ..prefilling(tail)
        }
    }

    fn slots(steps: &[PrefillStep]) -> Vec<usize> {
        steps.iter().map(|step| step.slot).collect()
    }

    #[test]
    fn resident_score_tails_share_one_prompt_pass() {
        let facts = [scoring(100), scoring(90), scoring(120), scoring(60)];
        let plan = plan(&facts, &cfg());
        assert_eq!(plan.prefill, None);
        assert_eq!(slots(&plan.prefill_batch), vec![0, 1, 2, 3]);
        assert!(plan.prefill_batch.iter().all(|step| step.last));
        assert_eq!(
            plan.prefill_batch
                .iter()
                .map(|step| step.tokens)
                .collect::<Vec<_>>(),
            vec![100, 90, 120, 60]
        );
    }

    #[test]
    fn a_prompt_that_is_not_a_score_at_the_front_reads_alone() {
        let facts = [prefilling(300), scoring(10), scoring(10)];
        let plan = plan(&facts, &cfg());
        assert!(plan.prefill_batch.is_empty());
        assert_eq!(plan.prefill.map(|step| step.slot), Some(0));
    }

    #[test]
    fn beside_a_decoding_sequence_a_batch_stays_within_the_mixed_quantum() {
        let facts = [decoding(), scoring(50), scoring(50), scoring(50)];
        let plan = plan(&facts, &cfg());
        assert_eq!(plan.decode, vec![0]);
        assert_eq!(
            slots(&plan.prefill_batch),
            vec![1, 2],
            "a third would pass 128 rows"
        );
    }

    #[test]
    fn a_flex_score_never_joins_a_batch_while_standard_work_runs() {
        let facts = [scoring(10), flex(scoring(10)), scoring(10)];
        let plan = plan(&facts, &cfg());
        assert_eq!(slots(&plan.prefill_batch), vec![0, 2]);
        assert_eq!(plan.paused, vec![1]);
    }

    #[test]
    fn one_score_alone_reads_as_it_always_did() {
        let facts = [scoring(40), prefilling(300)];
        let plan = plan(&facts, &cfg());
        assert!(plan.prefill_batch.is_empty());
        assert_eq!(
            plan.prefill,
            Some(PrefillStep {
                slot: 0,
                tokens: 40,
                last: true
            })
        );
    }

    /// A lead whose whole prompt its siblings' checkpoint would cover has no
    /// tail to batch: it reads alone rather than be passed over.
    #[test]
    fn a_lead_with_no_tail_past_the_shared_point_reads_alone() {
        let facts = [scoring_from(0, 128), scoring(10), scoring(10)];
        let plan = plan(&facts, &cfg());
        assert!(plan.prefill_batch.is_empty());
        assert_eq!(plan.prefill.map(|step| step.slot), Some(0));
    }

    fn scoring_from(tail: usize, remaining: usize) -> SlotFacts {
        SlotFacts {
            score: Some(ScoreFacts { tail }),
            ..prefilling(remaining)
        }
    }

    #[test]
    fn a_score_whose_tail_outgrows_the_quantum_reads_alone_in_quanta() {
        let facts = [scoring(900), scoring(10)];
        let plan = plan(&facts, &cfg());
        assert!(plan.prefill_batch.is_empty());
        assert_eq!(
            plan.prefill,
            Some(PrefillStep {
                slot: 0,
                tokens: 512,
                last: false
            })
        );
    }
}

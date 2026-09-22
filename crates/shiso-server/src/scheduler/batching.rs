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
//! # Why over-admission is impossible rather than checked
//!
//! The slot table owns its sequences in a `Vec` with a maximum it will not push
//! past, so "more resident sessions than the configuration allows" is not a
//! state the type can hold. There is no counter to keep in step with a list, and
//! therefore no way for the two to disagree.

use crate::scheduler::config::SchedulerConfig;
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
}

impl TickPlan {
    /// Whether this tick would do nothing at all.
    pub(crate) const fn is_idle(&self) -> bool {
        self.decode.is_empty() && self.prefill.is_none()
    }
}

/// Chooses what one tick does.
///
/// Pure, and deliberately so: the fairness rule is the scheduler's most
/// load-bearing claim and it should be assertable against a list of numbers
/// rather than against a running engine.
pub(crate) fn plan(facts: &[SlotFacts], cfg: &SchedulerConfig) -> TickPlan {
    let decode: Vec<usize> = facts
        .iter()
        .enumerate()
        .filter_map(|(at, fact)| fact.ready_to_decode.then_some(at))
        .collect();

    // The first sequence with prompt left, in admission order. Round-robin
    // across prefilling sequences would interleave two prompts into the same
    // number of quanta while finishing neither sooner, and the one that started
    // first is the one whose client has been waiting longest.
    let prefill = facts
        .iter()
        .enumerate()
        .find(|(_at, fact)| fact.prompt_remaining > 0)
        .map(|(at, fact)| {
            let quantum = if decode.is_empty() {
                cfg.solo_chunk
            } else {
                cfg.mixed_quantum
            };
            let tokens = if fact.indivisible_prompt {
                fact.prompt_remaining
            } else {
                fact.prompt_remaining.min(quantum.max(1))
            };
            PrefillStep {
                slot: at,
                tokens,
                last: tokens == fact.prompt_remaining,
            }
        });

    TickPlan { decode, prefill }
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
    use super::{PrefillStep, SlotFacts, plan};
    use crate::scheduler::config::SchedulerConfig;

    fn cfg() -> SchedulerConfig {
        SchedulerConfig {
            mixed_quantum: 8,
            solo_chunk: 64,
            ..SchedulerConfig::new()
        }
    }

    fn decoding() -> SlotFacts {
        SlotFacts {
            ready_to_decode: true,
            prompt_remaining: 0,
            indivisible_prompt: false,
        }
    }

    fn prefilling(remaining: usize) -> SlotFacts {
        SlotFacts {
            ready_to_decode: false,
            prompt_remaining: remaining,
            indivisible_prompt: false,
        }
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
                tokens: 8,
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
                tokens: 64,
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
        }];
        assert!(plan(&facts, &cfg()).is_idle());
        assert!(plan(&[], &cfg()).is_idle());
    }
}

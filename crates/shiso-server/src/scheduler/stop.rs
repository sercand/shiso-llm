//! Stop sequences, enforced once, where the text is produced.
//!
//! # Why this is in the engine loop and not in an encoder
//!
//! A stop sequence is a promise about the text a caller receives: generation
//! ends when the sequence is produced, and the sequence itself is not in the
//! answer. A blocking surface could keep that promise on its own by trimming the
//! finished turn, but a streaming surface could not - by the time the whole
//! sequence exists, the first half of it has already been sent. Two
//! implementations would then disagree for exactly the inputs that matter, which
//! is the divergence the event shape exists to rule out. So there is one
//! implementation, it sits where the text is produced, and both encodings read
//! its output.
//!
//! # Why text is held back
//!
//! A sequence can straddle two tokens: with `"BC"` as a stop sequence, the
//! fragments `"AB"` and `"CD"` complete it even though neither contains it. The
//! only way to be sure a fragment is safe to send is to know it cannot become
//! the head of a sequence, so everything within one sequence-length of the end
//! is withheld until more text settles it. The tail is released by [`flush`] at
//! the end of a turn that ended some other way.
//!
//! [`flush`]: StopWatcher::flush

/// What one fragment turned into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopStep {
    /// No sequence has completed. The text is safe to send; it may be empty,
    /// which means everything that arrived is still being held back.
    Continue(String),
    /// A sequence completed. The text before it is safe to send, and the turn is
    /// over.
    Stopped {
        /// Everything produced before the sequence started.
        text: String,
        /// The sequence that matched, as the caller spelled it.
        sequence: String,
    },
}

/// Watches a growing answer for the caller's stop sequences.
#[derive(Debug, Default)]
pub struct StopWatcher {
    sequences: Vec<String>,
    pending: String,
    longest: usize,
}

impl StopWatcher {
    /// A watcher for `sequences`.
    ///
    /// An empty sequence is dropped rather than honoured: it matches at position
    /// zero of everything, so honouring one would end every turn before its
    /// first token and report that the caller asked for it.
    #[must_use]
    pub fn new(sequences: &[String]) -> Self {
        let sequences: Vec<String> = sequences
            .iter()
            .filter(|sequence| !sequence.is_empty())
            .cloned()
            .collect();
        let longest = sequences.iter().map(String::len).max().unwrap_or(0);
        Self {
            sequences,
            pending: String::new(),
            longest,
        }
    }

    /// Whether anything is being watched for.
    ///
    /// The caller checks this to skip the hold-back entirely: a request with no
    /// stop sequences should not pay a one-token delay for a promise it never
    /// made.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Takes the text a token produced.
    #[must_use]
    pub fn push(&mut self, fragment: &str) -> StopStep {
        if self.sequences.is_empty() {
            return StopStep::Continue(fragment.to_owned());
        }
        self.pending.push_str(fragment);

        // The earliest match wins. A later one would silently discard text the
        // caller asked to keep, because the answer ends at the first sequence
        // the model produced and not at whichever one this loop saw first.
        let hit = self
            .sequences
            .iter()
            .filter_map(|sequence| {
                self.pending
                    .find(sequence.as_str())
                    .map(|at| (at, sequence))
            })
            .min_by_key(|(at, _sequence)| *at);
        if let Some((at, sequence)) = hit {
            let sequence = sequence.clone();
            let text = self.pending.get(..at).unwrap_or_default().to_owned();
            self.pending.clear();
            return StopStep::Stopped { text, sequence };
        }

        StopStep::Continue(self.release())
    }

    /// Everything still held back, for a turn that ended some other way.
    #[must_use]
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Splits off the part of the buffer that cannot become a sequence head.
    ///
    /// The retained tail is one byte short of the longest sequence, snapped down
    /// to a character boundary: splitting a string in the middle of a multi-byte
    /// character is not something this type is allowed to do, and rounding the
    /// hold-back up is always safe while rounding it down is not.
    fn release(&mut self) -> String {
        let keep = self.longest.saturating_sub(1);
        if self.pending.len() <= keep {
            return String::new();
        }
        let mut at = self.pending.len() - keep;
        while at > 0 && !self.pending.is_char_boundary(at) {
            at -= 1;
        }
        let tail = self.pending.get(at..).unwrap_or_default().to_owned();
        let head = self.pending.get(..at).unwrap_or_default().to_owned();
        self.pending = tail;
        head
    }
}

#[cfg(test)]
mod tests {
    use super::{StopStep, StopWatcher};

    fn sequences(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    /// With nothing to watch for, every fragment passes through untouched and
    /// unbuffered: a request that named no stop sequence must not pay a delay.
    #[test]
    fn a_watcher_with_no_sequences_passes_every_fragment_straight_through() {
        let mut watcher = StopWatcher::new(&[]);
        assert!(watcher.is_empty());
        assert_eq!(
            watcher.push("hello"),
            StopStep::Continue("hello".to_owned())
        );
        assert_eq!(watcher.flush(), "");
    }

    /// The case a per-fragment scan gets wrong: neither fragment contains the
    /// sequence, and together they do.
    #[test]
    fn a_sequence_split_across_two_fragments_still_ends_the_turn() {
        let mut watcher = StopWatcher::new(&sequences(&["BC"]));
        assert_eq!(watcher.push("AB"), StopStep::Continue("A".to_owned()));
        assert_eq!(
            watcher.push("CD"),
            StopStep::Stopped {
                text: String::new(),
                sequence: "BC".to_owned(),
            }
        );
    }

    /// The text before the sequence is kept; the sequence itself is not part of
    /// the answer.
    #[test]
    fn the_answer_keeps_what_came_before_the_sequence_and_not_the_sequence() {
        let mut watcher = StopWatcher::new(&sequences(&["\n\nHuman:"]));
        let step = watcher.push("the answer is 4\n\nHuman: and then");
        assert_eq!(
            step,
            StopStep::Stopped {
                text: "the answer is 4".to_owned(),
                sequence: "\n\nHuman:".to_owned(),
            }
        );
    }

    /// Two sequences, and the one that appears first is the one that stopped it.
    #[test]
    fn the_earliest_sequence_in_the_text_is_the_one_that_stopped_it() {
        let mut watcher = StopWatcher::new(&sequences(&["END", "STOP"]));
        let step = watcher.push("aaa STOP bbb END ccc");
        assert_eq!(
            step,
            StopStep::Stopped {
                text: "aaa ".to_owned(),
                sequence: "STOP".to_owned(),
            }
        );
    }

    /// A turn that ends for another reason must still deliver the tail that was
    /// being held back, or the last few characters of every answer vanish.
    #[test]
    fn a_turn_that_ends_otherwise_still_delivers_the_held_back_tail() {
        let mut watcher = StopWatcher::new(&sequences(&["XYZ"]));
        let mut seen = String::new();
        for fragment in ["abc", "def"] {
            match watcher.push(fragment) {
                StopStep::Continue(text) => seen.push_str(&text),
                StopStep::Stopped { .. } => panic!("nothing matched"),
            }
        }
        seen.push_str(&watcher.flush());
        assert_eq!(seen, "abcdef");
    }

    /// The hold-back must never split a multi-byte character, because the two
    /// halves are not text and the encoder downstream would refuse them.
    #[test]
    fn the_hold_back_never_splits_a_multi_byte_character() {
        // A four-byte sequence forces three bytes of hold-back, which lands in
        // the middle of these two-byte characters unless the split is snapped.
        let mut watcher = StopWatcher::new(&sequences(&["----"]));
        let mut seen = String::new();
        for fragment in ["é", "é", "é", "é"] {
            match watcher.push(fragment) {
                StopStep::Continue(text) => seen.push_str(&text),
                StopStep::Stopped { .. } => panic!("nothing matched"),
            }
        }
        seen.push_str(&watcher.flush());
        assert_eq!(seen, "éééé");
    }

    /// An empty sequence matches everything at position zero, so honouring one
    /// would end every turn before its first token.
    #[test]
    fn an_empty_sequence_is_dropped_rather_than_matching_everything() {
        let mut watcher = StopWatcher::new(&sequences(&[""]));
        assert!(watcher.is_empty());
        assert_eq!(
            watcher.push("hello"),
            StopStep::Continue("hello".to_owned())
        );
    }
}

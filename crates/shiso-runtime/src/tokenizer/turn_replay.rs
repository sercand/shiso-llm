//! Finished turns, kept as the exact ids the model ran, so the next request in
//! a conversation re-enters the prefix cache instead of re-tokenizing it.
//!
//! # Why re-tokenizing is not enough
//!
//! A multi-turn client sends the whole conversation back, and the template
//! renders the previous assistant turn as text again. That text is what the
//! model generated, character for character, but encoding it does not always
//! give the ids the model generated: a model is free to emit a split its own
//! vocabulary would not produce (`>` then `The` where encoding the same text
//! gives `>The`). Measured on a tool-calling turn with a 7 KB HTML argument,
//! the first such split sat 2 000 tokens into the generation. From that id on
//! the prompt differs from every stored path, and a hybrid model resumes only
//! at a checkpoint, so the request fell back to the previous prompt's
//! checkpoint and prefilled the whole generated turn again.
//!
//! # What is kept, and why the match is on text
//!
//! Each record is the rendered prompt text followed by the decoded generation,
//! and the prompt ids followed by the generated ids. A new prompt whose text
//! starts with a record's text takes that record's ids for the span and encodes
//! only the rest. The match is on the whole rendered text, so a template that
//! renders the turn differently - reasoning dropped, a call re-serialized - is
//! no match and the prompt is encoded cold, as before: the ids are only ever
//! swapped for other ids of the same text.
//!
//! Only turns that ended on the end-of-turn marker are kept. A stop sequence
//! leaves generated ids past the text the caller received, and a cut budget can
//! end mid-character; neither decodes to the text the next prompt will carry.
//! Prompts with media are never kept or matched: their pad runs are the same
//! ids for different pictures.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use shiso_core::TokenId;

/// One finished turn: its text and the ids the model ran for it.
struct Turn {
    text: Box<str>,
    tokens: Arc<[TokenId]>,
}

/// The records, oldest first, and the ids they hold between them.
#[derive(Default)]
struct Inner {
    turns: VecDeque<Turn>,
    held: usize,
}

/// A span of a prompt that a finished turn already covers.
#[derive(Debug, Clone)]
pub struct Replayed {
    /// Bytes of the prompt text the ids stand for.
    pub text_len: usize,
    /// The ids the model ran for that text.
    pub tokens: Arc<[TokenId]>,
}

/// Recently finished turns, bounded by the ids they hold.
///
/// Written by the scheduler as turns retire and read by the renderer as
/// requests arrive; one short lock each way, never held across an encode.
pub struct TurnReplay {
    inner: Mutex<Inner>,
    budget: usize,
}

impl fmt::Debug for TurnReplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("TurnReplay")
            .field("turns", &inner.turns.len())
            .field("held", &inner.held)
            .field("budget", &self.budget)
            .finish()
    }
}

impl TurnReplay {
    /// An empty store holding at most `budget` ids across its records.
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            budget,
        }
    }

    /// Keeps a finished turn: `text` is what the next prompt would render for
    /// it and `tokens` the ids the model ran.
    ///
    /// A record whose text this one extends is dropped: the conversation has
    /// moved past it, and a prompt that extends the older text but not this
    /// one has already left the path both describe.
    pub fn record(&self, text: String, tokens: Arc<[TokenId]>) {
        if tokens.len() > self.budget || text.is_empty() {
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let before = inner.turns.len();
        inner.turns.retain(|turn| !text.starts_with(&*turn.text));
        if inner.turns.len() != before {
            inner.held = inner.turns.iter().map(|turn| turn.tokens.len()).sum();
        }
        inner.held += tokens.len();
        inner.turns.push_back(Turn {
            text: text.into_boxed_str(),
            tokens,
        });
        while inner.held > self.budget {
            let Some(oldest) = inner.turns.pop_front() else {
                break;
            };
            inner.held -= oldest.tokens.len();
        }
    }

    /// The longest record `text` starts with, which becomes the newest.
    ///
    /// `None` is a prompt no finished turn covers: a first turn, a template
    /// that renders history differently from what was generated, or a record
    /// already evicted.
    #[must_use]
    pub fn longest_prefix(&self, text: &str) -> Option<Replayed> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let (at, _) = inner
            .turns
            .iter()
            .enumerate()
            .filter(|(_, turn)| text.starts_with(&*turn.text))
            .max_by_key(|(_, turn)| turn.text.len())?;
        let turn = inner.turns.remove(at)?;
        let replayed = Replayed {
            text_len: turn.text.len(),
            tokens: Arc::clone(&turn.tokens),
        };
        inner.turns.push_back(turn);
        Some(replayed)
    }
}

#[cfg(test)]
mod tests {
    use super::TurnReplay;
    use shiso_core::TokenId;
    use std::sync::Arc;

    fn ids(raw: &[u32]) -> Arc<[TokenId]> {
        raw.iter().map(|&id| TokenId::new(id)).collect()
    }

    #[test]
    fn a_prompt_extending_a_turn_gets_its_ids() {
        let replay = TurnReplay::new(100);
        replay.record("<s>hi<a>yo<e>".to_owned(), ids(&[1, 2, 3, 4, 5]));
        let hit = replay
            .longest_prefix("<s>hi<a>yo<e>\n<u>more")
            .expect("the prompt extends the turn");
        assert_eq!(hit.text_len, "<s>hi<a>yo<e>".len());
        assert_eq!(&*hit.tokens, &*ids(&[1, 2, 3, 4, 5]));
    }

    #[test]
    fn a_prompt_that_renders_the_turn_differently_misses() {
        let replay = TurnReplay::new(100);
        replay.record("<s>hi<a><think>x</think>yo<e>".to_owned(), ids(&[1, 2, 3]));
        assert!(
            replay
                .longest_prefix("<s>hi<a><think></think>yo<e>")
                .is_none()
        );
    }

    #[test]
    fn the_longest_covering_turn_wins() {
        let replay = TurnReplay::new(100);
        replay.record("ab".to_owned(), ids(&[1]));
        replay.record("xy".to_owned(), ids(&[9]));
        replay.record("abcd".to_owned(), ids(&[1, 2]));
        let hit = replay.longest_prefix("abcdef").expect("covered");
        assert_eq!(hit.text_len, 4);
    }

    #[test]
    fn an_extended_turn_replaces_the_one_it_extends() {
        let replay = TurnReplay::new(100);
        replay.record("ab".to_owned(), ids(&[1]));
        replay.record("abcd".to_owned(), ids(&[1, 2]));
        assert!(replay.longest_prefix("abxx").is_none());
    }

    #[test]
    fn the_oldest_turn_goes_when_the_budget_is_spent() {
        let replay = TurnReplay::new(4);
        replay.record("one".to_owned(), ids(&[1, 1]));
        replay.record("two".to_owned(), ids(&[2, 2]));
        // Reading `one` makes it the newest, so `two` is what goes.
        assert!(replay.longest_prefix("one!").is_some());
        replay.record("six".to_owned(), ids(&[6, 6]));
        assert!(replay.longest_prefix("two!").is_none());
        assert!(replay.longest_prefix("one!").is_some());
        assert!(replay.longest_prefix("six!").is_some());
    }

    #[test]
    fn a_turn_larger_than_the_budget_is_not_kept() {
        let replay = TurnReplay::new(2);
        replay.record("big".to_owned(), ids(&[1, 2, 3]));
        assert!(replay.longest_prefix("big!").is_none());
    }
}

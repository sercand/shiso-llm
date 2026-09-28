//! What a caller says the audio is likely to contain, and the recognizer's
//! own ids for it.
//!
//! The text is the caller's and the ids are the family's: how a context is
//! worded for the model and how it is tokenized belong to the recognizer, so
//! the server holds an encoder the recognizer handed it and never tokenizes
//! on its own. The ids are made once per request, off the audio thread, and
//! shared by every chunk the request is cut into.

use std::fmt;
use std::sync::Arc;

use shiso_core::TokenId;

use crate::error::AudioError;

/// Words and text a request says its audio is likely to contain.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Context {
    /// Free text about the audio (the `prompt` field), trimmed; `None` when
    /// the request sent none.
    pub prompt: Option<String>,
    /// Terms the audio is likely to contain (the `keywords` field), each
    /// trimmed and non-empty, in the order sent.
    pub keywords: Vec<String>,
}

impl Context {
    /// Whether the request gave no context at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prompt.is_none() && self.keywords.is_empty()
    }
}

/// A context as the recognizer's token ids, ready for its prompt, with the
/// keywords it was made from as [`keyword_words`] sequences, so a transcript
/// can be checked for echoing them ([`is_echo`]). Cheap to clone: every chunk
/// of a job shares one allocation of each.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextIds {
    ids: Arc<[TokenId]>,
    keywords: Arc<[Vec<String>]>,
}

impl ContextIds {
    /// Wraps ids an encoder produced, from a context with no keywords: a
    /// transcript made with it is never judged an echo.
    #[must_use]
    pub fn new(ids: Vec<TokenId>) -> Self {
        Self::with_keywords(ids, Vec::new())
    }

    /// Wraps ids an encoder produced from a context whose keywords are
    /// `keywords`, each already split by [`keyword_words`].
    #[must_use]
    pub fn with_keywords(ids: Vec<TokenId>, keywords: Vec<Vec<String>>) -> Self {
        Self {
            ids: ids.into(),
            keywords: keywords.into(),
        }
    }

    /// The ids, in order.
    #[must_use]
    pub fn ids(&self) -> &[TokenId] {
        &self.ids
    }

    /// The keywords' word sequences, in the order the request sent them;
    /// empty for a context of a prompt alone.
    #[must_use]
    pub fn keywords(&self) -> &[Vec<String>] {
        &self.keywords
    }

    /// How many ids there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are none: the prompt's context slot stays empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// `text` as the words a keyword and a transcript are compared by:
/// lowercased, split on whitespace and punctuation.
///
/// The casing is Turkish-aware and then folds the dotless `ı` onto `i`:
/// generic lowercasing turns `İ` into `i` plus a combining dot, so
/// `İstanbul` would not meet `istanbul`, and a Latin keyword written in
/// capitals (`I`) must still meet the model's lowercase `i`. Folding costs
/// the `ı`/`i` distinction, which only this comparison sees.
#[must_use]
pub fn keyword_words(text: &str) -> Vec<String> {
    let mut lowered = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            'I' | 'İ' | 'ı' => lowered.push('i'),
            c if c.is_alphanumeric() => lowered.extend(c.to_lowercase()),
            _ => lowered.push(' '),
        }
    }
    lowered.split_whitespace().map(str::to_owned).collect()
}

/// The fewest words a span needs before its keyword share is judged: one or
/// two words that are a keyword are an answer, not an echo.
const ECHO_MIN_WORDS: usize = 3;
/// The fewest distinct keywords one run must hold to read as a list.
const ECHO_RUN_KEYWORDS: usize = 3;
/// The most other words between two keyword occurrences of one run.
const ECHO_RUN_GAP: usize = 1;

/// Whether `text` - one span's transcript - echoes the keyword list
/// `keywords` (each a [`keyword_words`] sequence) rather than transcribing
/// speech. Over the text's [`keyword_words`], keyword occurrences are
/// matched left to right, the longest keyword first at each word; the span
/// echoes when
///
/// - it has at least 3 words and keyword occurrences cover at least half of
///   them, or
/// - one run of occurrences, each at most 1 other word after the last, holds
///   at least 3 distinct keywords: a list, which a sentence rarely is.
///
/// Three distinct keywords anywhere in a span is not enough: a 30 s span of
/// technical speech names three of them without reading out the list.
///
/// No keywords, or a text with no words, never echoes.
///
/// Measured with these thresholds, the first tried (2026-09-28, Apple M4,
/// Qwen3-ASR 1.7B `Q8_0`, a private 58-minute meeting of 31 keywords, offline
/// with silence kept), the recognizer decoding each echoing span again with
/// no context: 7 of 318 spans were decoded again, and word error rate
/// against another system's export came back from 35.4% to 31.7%, the
/// no-keyword baseline's own. Insertions fell from 563 to 295 (282 without
/// keywords), keyword matches from 114 to 55 of 306 (39 without), most of
/// the 114 having been the echoed lists themselves, and segments reading as
/// a list from 11 (470 words) to 4 (149 words; 1 and 66 without keywords).
/// On a private 5-minute sample the labelled wording's keyword recall held
/// at 73.5% (44.1% without keywords) at 24.6% word error rate (25.6%), and
/// a prompt put before the keywords, which had echoed to 31.2%, fell to
/// 24.4%.
#[must_use]
pub fn is_echo(text: &str, keywords: &[Vec<String>]) -> bool {
    if keywords.iter().all(Vec::is_empty) {
        return false;
    }
    let words = keyword_words(text);
    let (mut covered, mut i) = (0, 0);
    // The distinct keywords of the current run, and where its last ended.
    let mut run: Vec<&[String]> = Vec::new();
    let mut run_end = 0;
    while i < words.len() {
        let found = keywords
            .iter()
            .filter(|k| !k.is_empty() && words[i..].starts_with(k))
            .max_by_key(|k| k.len());
        let Some(keyword) = found else {
            i += 1;
            continue;
        };
        if run.is_empty() || i - run_end > ECHO_RUN_GAP {
            run.clear();
        }
        if !run.contains(&keyword.as_slice()) {
            run.push(keyword);
        }
        if run.len() >= ECHO_RUN_KEYWORDS {
            return true;
        }
        covered += keyword.len();
        i += keyword.len();
        run_end = i;
    }
    words.len() >= ECHO_MIN_WORDS && 2 * covered >= words.len()
}

/// Turns a [`Context`] into the ids a recognizer's prompt carries.
pub trait ContextEncoder: Send + Sync + fmt::Debug {
    /// The context worded as the model reads it, then tokenized.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadContext`] for a context longer than
    /// [`Self::max_ids`] or one the tokenizer cannot encode.
    fn encode(&self, context: &Context) -> Result<ContextIds, AudioError>;

    /// The most ids a context may take in this recognizer's prompt.
    fn max_ids(&self) -> usize;
}

#[cfg(test)]
#[path = "context_tests.rs"]
mod tests;

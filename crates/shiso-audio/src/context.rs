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

/// A context as the recognizer's token ids, ready for its prompt. Cheap to
/// clone: every chunk of a job shares one allocation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextIds(Arc<[TokenId]>);

impl ContextIds {
    /// Wraps ids an encoder produced.
    #[must_use]
    pub fn new(ids: Vec<TokenId>) -> Self {
        Self(ids.into())
    }

    /// The ids, in order.
    #[must_use]
    pub fn ids(&self) -> &[TokenId] {
        &self.0
    }

    /// How many ids there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are none: the prompt's context slot stays empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
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

//! The context the system turn carries: how a request's keywords and prompt
//! are worded for the model, and the checkpoint's own byte-level BPE that
//! turns the wording into ids.
//!
//! The reference fills the system turn with free text, so the wording is
//! plain text rather than a template of its own: the prompt as sent, then
//! the keywords on one line, comma-separated. This is provisional until it
//! is measured against a held-out recording; a template that names the
//! keywords more explicitly (e.g. `"Likely to appear: ..."`) is worth trying
//! if the plain wording under-recognises them.
//!
//! Only the vocabulary's ordinary pieces enter the BPE, not its added
//! tokens: a keyword that spells `<|im_end|>` is encoded byte by byte and
//! cannot end the system turn early.

use std::str::FromStr as _;

use shiso_audio::{AudioError, Context, ContextEncoder, ContextIds};
use shiso_core::TokenId;

use super::prompt::MAX_CONTEXT_IDS;

/// Qwen2's pre-tokenizer split, as its `tokenizer.json` spells it.
const SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The encoder a Qwen3-ASR checkpoint's context is tokenized with.
#[derive(Debug)]
pub(crate) struct QwenContextEncoder {
    tokenizer: tokenizers::Tokenizer,
}

fn bad(reason: impl Into<String>) -> AudioError {
    AudioError::BadContext {
        reason: reason.into(),
    }
}

impl QwenContextEncoder {
    /// The BPE over `pieces` (the ordinary pieces, id = index, in the
    /// byte alphabet) and `merges` (`"left right"`, in rank order).
    ///
    /// # Errors
    ///
    /// [`AudioError::Vocab`] for pieces or merges the BPE refuses.
    pub(crate) fn new(pieces: &[String], merges: &[String]) -> Result<Self, AudioError> {
        let vocab: serde_json::Map<String, serde_json::Value> = pieces
            .iter()
            .enumerate()
            .map(|(id, piece)| (piece.clone(), serde_json::Value::from(id)))
            .collect();
        let document = serde_json::json!({
            "version": "1.0",
            "added_tokens": [],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": SPLIT},
                     "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": false,
                     "trim_offsets": false, "use_regex": false}
                ]
            },
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                        "trim_offsets": false, "use_regex": false},
            "model": {"type": "BPE", "dropout": null, "unk_token": null,
                      "continuing_subword_prefix": null, "end_of_word_suffix": null,
                      "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                      "vocab": vocab, "merges": merges}
        });
        let tokenizer = tokenizers::Tokenizer::from_str(&document.to_string()).map_err(|e| {
            AudioError::Vocab {
                reason: format!("the context tokenizer: {e}"),
            }
        })?;
        Ok(Self { tokenizer })
    }
}

/// The system turn's text for `context`: the prompt, then the keywords
/// joined by `, ` on the next line; either alone when the other is absent.
pub(crate) fn wording(context: &Context) -> String {
    let keywords = context.keywords.join(", ");
    match (&context.prompt, keywords.is_empty()) {
        (Some(prompt), false) => format!("{prompt}\n{keywords}"),
        (Some(prompt), true) => prompt.clone(),
        (None, _) => keywords,
    }
}

impl ContextEncoder for QwenContextEncoder {
    fn encode(&self, context: &Context) -> Result<ContextIds, AudioError> {
        let text = wording(context);
        let encoding = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| bad(format!("the context does not tokenize: {e}")))?;
        let ids = encoding.get_ids();
        if ids.len() > MAX_CONTEXT_IDS {
            return Err(bad(format!(
                "the context is {} tokens; this model's prompt carries at most {MAX_CONTEXT_IDS}",
                ids.len()
            )));
        }
        Ok(ContextIds::new(
            ids.iter().copied().map(TokenId::new).collect(),
        ))
    }

    fn max_ids(&self) -> usize {
        MAX_CONTEXT_IDS
    }
}

#[cfg(test)]
#[path = "context_tests.rs"]
mod tests;

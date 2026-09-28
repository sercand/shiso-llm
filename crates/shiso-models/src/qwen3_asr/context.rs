//! The context the system turn carries: how a request's keywords and prompt
//! are worded for the model, and the checkpoint's own byte-level BPE that
//! turns the wording into ids.
//!
//! The reference fills the system turn with free text, so the wording is
//! plain text rather than a template of its own: the prompt as sent, then
//! the keywords labelled `Keywords: ` and comma-joined on the next line.
//! [`wording`]'s doc comment carries the private-recording measurement below
//! that chose the label over the plain comma list.
//!
//! Only the vocabulary's ordinary pieces enter the BPE, not its added
//! tokens: a keyword that spells `<|im_end|>` is encoded byte by byte and
//! cannot end the system turn early.

use std::str::FromStr as _;

use shiso_audio::{AudioError, Context, ContextEncoder, ContextIds, keyword_words};
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
/// labelled `Keywords: ` and joined by `, ` on the next line; either alone
/// when the other is absent.
///
/// Measured on a private short recording (2026-09-28, an M-series machine)
/// against the plain comma list this replaced: with no context,
/// keyword recall was 44.1% at 25.6% word error rate. The plain `a, b, c`
/// wording raised recall to 64.7% at 25.1% WER. This label raised it further
/// to 73.5% at 24.6% WER with no insertions - the best recall among wordings
/// that stayed within the baseline's WER + 1.0 pp, and the fewest insertions
/// among those tied on recall. Newline-joining the keywords under no label
/// matched this recall (73.5%) but cost one insertion; prefixing a scenario
/// prompt onto the plain comma list reached 100% recall but pushed WER to
/// 31.2%, over the bound.
pub(crate) fn wording(context: &Context) -> String {
    let keywords = context.keywords.join(", ");
    let labelled = if keywords.is_empty() {
        String::new()
    } else {
        format!("Keywords: {keywords}")
    };
    match (&context.prompt, labelled.is_empty()) {
        (Some(prompt), false) => format!("{prompt}\n{labelled}"),
        (Some(prompt), true) => prompt.clone(),
        (None, _) => labelled,
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
            return Err(AudioError::TooManyContextTokens {
                tokens: ids.len(),
                max: MAX_CONTEXT_IDS,
            });
        }
        Ok(ContextIds::with_keywords(
            ids.iter().copied().map(TokenId::new).collect(),
            context.keywords.iter().map(|k| keyword_words(k)).collect(),
        ))
    }

    fn max_ids(&self) -> usize {
        MAX_CONTEXT_IDS
    }
}

#[cfg(test)]
#[path = "context_tests.rs"]
mod tests;

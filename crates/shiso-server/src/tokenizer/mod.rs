//! The vocabulary and the chat template: everything between a conversation and
//! the token ids the engine runs.
//!
//! Three concerns live here and they are deliberately separate types.
//! [`Vocabulary`] is the checkpoint's own encoder and decoder.
//! [`chat_render::ChatTemplate`] is the Jinja document that says how a
//! conversation becomes prompt text. [`message_preprocess`] is the step in
//! between: it turns the neutral IR into the value shape a template expects and
//! pulls the media out as it goes.
//!
//! # Why detokenization is incremental and not per token
//!
//! A byte-pair vocabulary does not decode one id at a time. A multi-byte
//! character can span two ids, and the leading-space convention means a piece's
//! rendering depends on what came before it, so `decode([id])` for each id in
//! turn produces text that is subtly and sometimes catastrophically different
//! from `decode(all_ids)`. [`Detokenizer`] therefore decodes the whole run each
//! step and emits only what is new, which costs a quadratic amount of decoding
//! for a linear amount of output and is worth it: the alternative is a stream
//! whose concatenation does not equal the blocking answer, which is exactly the
//! divergence the IR's accumulator exists to rule out.

pub mod chat_render;
pub mod jinja_helpers;
pub mod message_preprocess;
pub mod render_prompt;
pub mod turn_replay;

use std::path::Path;

use shiso_core::{PadTokens, TokenId};
use shiso_vision::MediaPlacement;

/// Why the vocabulary could not be used.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TokenizerError {
    /// The vocabulary file could not be read or parsed.
    #[error("the vocabulary at {path} could not be loaded: {message}")]
    Load {
        /// The file that was tried.
        path: String,
        /// What the vocabulary library reported.
        message: String,
    },
    /// Text could not be encoded.
    #[error("the prompt could not be encoded: {message}")]
    Encode {
        /// What the vocabulary library reported.
        message: String,
    },
    /// Ids could not be decoded.
    #[error("the generated ids could not be decoded: {message}")]
    Decode {
        /// What the vocabulary library reported.
        message: String,
    },
}

/// The checkpoint's vocabulary.
///
/// A newtype rather than a re-export, so the rest of the crate speaks
/// [`TokenId`] and never the bare `u32` the underlying library uses. That is
/// not ceremony: a prompt id, a pad marker id and a vocabulary *size* are all
/// `u32`, and the newtype is what stops two of them being swapped.
#[derive(Debug)]
pub struct Vocabulary {
    inner: tokenizers::Tokenizer,
    bos: Option<TokenId>,
}

impl Vocabulary {
    /// Loads a vocabulary from a `tokenizer.json`, and the opening token from
    /// the `tokenizer_config.json` beside it.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::Load`] when the file is absent or does not parse. An
    /// unreadable or absent `tokenizer_config.json` is NOT an error: it leaves
    /// the vocabulary with no opening token, which is what a checkpoint that
    /// declares none behaves like.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let inner =
            tokenizers::Tokenizer::from_file(path).map_err(|error| TokenizerError::Load {
                path: path.display().to_string(),
                message: error.to_string(),
            })?;
        let bos = path
            .parent()
            .map(|dir| dir.join("tokenizer_config.json"))
            .and_then(|config| bos_from_config(&config, &inner));
        Ok(Self { inner, bos })
    }

    /// The opening token this checkpoint puts in front of a prompt, if any.
    #[must_use]
    pub const fn bos(&self) -> Option<TokenId> {
        self.bos
    }

    /// Declares the opening token explicitly, for a vocabulary built in memory.
    #[must_use]
    pub const fn with_bos(mut self, bos: Option<TokenId>) -> Self {
        self.bos = bos;
        self
    }

    /// Wraps a vocabulary that is already in hand.
    ///
    /// The path-taking constructor above is what a server uses; this is what a
    /// test uses to build a small deterministic vocabulary without writing one
    /// to a file. Both end at the same struct, so a test that renders a prompt
    /// is exercising the same encoder a deployment does.
    #[must_use]
    pub const fn from_tokenizer(inner: tokenizers::Tokenizer) -> Self {
        Self { inner, bos: None }
    }

    /// Encodes prompt text.
    ///
    /// `add_special` is the template's decision, not this type's: a rendered
    /// chat template has already written every marker the model expects, and
    /// adding another set around it produces a prompt the model has never seen
    /// in training.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::Encode`] when the text cannot be encoded.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>, TokenizerError> {
        let encoded =
            self.inner
                .encode(text, add_special)
                .map_err(|error| TokenizerError::Encode {
                    message: error.to_string(),
                })?;
        Ok(encoded
            .get_ids()
            .iter()
            .copied()
            .map(TokenId::new)
            .collect())
    }

    /// Encodes a prompt the way the checkpoint expects to receive one: with its
    /// opening token in front, exactly once.
    ///
    /// # Why this is not the caller's decision
    ///
    /// Upstream llama.cpp adds the opening token in the tokenizer, gated on the
    /// container's `tokenizer.ggml.add_bos_token`, and every prompt route goes
    /// through it - `/completion`, `/v1/completions` and `/v1/chat/completions`
    /// alike. To keep a chat template that writes `<bos>` itself from producing
    /// two, it strips a leading one from the rendered text before tokenizing
    /// (`common/chat.cpp`, the `add_bos && string_starts_with` pair).
    ///
    /// This port had the arrangement the other way round - the template wrote
    /// the token and the encoder never added one - which is correct for chat and
    /// leaves a raw completion with no opening token at all. Gemma 4 answers
    /// `"The capital of France is"` with `" France is France is France is"` and
    /// `"<bos>The capital of France is"` with `" Paris."`, so the omission is
    /// not subtle once it matters.
    ///
    /// The de-duplication here is on the ids rather than on the text, which is
    /// the one deliberate difference from upstream: a rendered template whose
    /// `<bos>` has already tokenized to the opening id is left alone, and so is
    /// a raw prompt that happens to begin with the token spelled out. Upstream
    /// strips a text prefix and then adds unconditionally, so it emits two for
    /// that second case. Its own comment there reads `// TODO: improve this
    /// later`.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::Encode`] when the text cannot be encoded.
    pub fn encode_prompt(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        let mut tokens = self.encode(text, false)?;
        if let Some(bos) = self.bos
            && tokens.first() != Some(&bos)
        {
            tokens.insert(0, bos);
        }
        Ok(tokens)
    }

    /// Decodes a run of ids back to text.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::Decode`] when the ids cannot be decoded.
    pub fn decode(&self, tokens: &[TokenId]) -> Result<String, TokenizerError> {
        let ids: Vec<u32> = tokens.iter().map(|token| token.get()).collect();
        self.inner
            .decode(&ids, false)
            .map_err(|error| TokenizerError::Decode {
                message: error.to_string(),
            })
    }

    /// How many ids this vocabulary defines, added tokens included.
    ///
    /// The bound a caller-supplied token id is checked against. A surface that
    /// accepts raw ids has to refuse one the model has no embedding row for,
    /// and the alternative is a device-side out-of-bounds read reported as a
    /// backend failure.
    #[must_use]
    pub fn size(&self) -> u32 {
        u32::try_from(self.inner.get_vocab_size(true)).unwrap_or(u32::MAX)
    }

    /// The id of one marker, when this vocabulary declares it.
    ///
    /// `None` for a marker a text-only checkpoint has never heard of, which is
    /// why the pad expansion is a no-op rather than an error there.
    #[must_use]
    pub fn marker_id(&self, marker: &str) -> Option<TokenId> {
        self.inner.token_to_id(marker).map(TokenId::new)
    }

    /// The placeholder ids this family's placement resolves to here.
    ///
    /// Resolved through the vocabulary rather than written down as constants:
    /// the marker is a property of the family's template and the id is a
    /// property of this checkpoint's vocabulary, and a constant would be a
    /// second answer to the second question that can disagree with the first.
    #[must_use]
    pub fn media_ids(&self, placement: &MediaPlacement) -> PadTokens {
        PadTokens::new(
            self.marker_id(placement.image_marker),
            placement.video_marker.and_then(|m| self.marker_id(m)),
        )
    }

    /// The bracket ids this family writes around an image's run, when it writes
    /// one and this vocabulary declares both halves.
    ///
    /// All or nothing on purpose: half a bracket is a prompt shape the model
    /// was never trained on, and it would be indistinguishable in the output
    /// from one it was.
    #[must_use]
    pub fn wrap_ids(&self, placement: &MediaPlacement) -> Option<(TokenId, TokenId)> {
        let (open, close) = placement.wrap_image?;
        Some((self.marker_id(open)?, self.marker_id(close)?))
    }
}

/// The opening token a `tokenizer_config.json` declares, resolved against this
/// vocabulary.
///
/// Two keys decide it, in the order Hugging Face reads them and upstream
/// llama.cpp mirrors from `tokenizer.ggml.add_bos_token` and
/// `tokenizer.ggml.bos_token_id`:
///
/// - `add_bos_token: false` settles it - the checkpoint says not to, and no
///   `bos_token` beside it changes that;
/// - otherwise a `bos_token` naming a token this vocabulary declares is it.
///   The key is a bare string in some exports and a `{"content": ...}` object in
///   others, and both spellings mean the same thing.
///
/// A checkpoint that names no `bos_token` gets none, which is the answer for
/// the Qwen family: its `tokenizer_config.json` carries `eos_token` and no
/// opening token at all. Gemma 4's names `<bos>` and sets no `add_bos_token`,
/// which agrees with its container's `add_bos_token = true` and
/// `bos_token_id = 2`.
///
/// Returning `None` on an unreadable or unparseable file is deliberate. The
/// vocabulary itself came from `tokenizer.json`; this file is a sidecar, and a
/// deployment that points at a bare `tokenizer.json` should keep working rather
/// than fail to start over a prompt-shaping detail.
fn bos_from_config(path: &Path, inner: &tokenizers::Tokenizer) -> Option<TokenId> {
    let text = std::fs::read_to_string(path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    if config.get("add_bos_token") == Some(&serde_json::Value::Bool(false)) {
        return None;
    }
    let declared = config.get("bos_token")?;
    let name = declared
        .as_str()
        .or_else(|| declared.get("content")?.as_str())?;
    inner.token_to_id(name).map(TokenId::new)
}

/// Turns a growing run of ids into the text that is new since the last call.
///
/// # Why the whole run is decoded every time
///
/// See the module docs: a piece's rendering depends on its neighbours, so the
/// only way to be sure the stream's concatenation equals the blocking answer is
/// to decode the same run the blocking answer decodes. The state kept here is
/// how much of that answer has already been sent, not a running buffer of
/// separately decoded pieces.
///
/// # What a `None` vocabulary means
///
/// A detokenizer with no vocabulary emits no text and passes the ids through
/// untouched. That is the shape a test harness uses, and it is also what a
/// deployment gets if a checkpoint ever ships without a vocabulary: the ids
/// still reach the caller, which is strictly more than a failed request.
#[derive(Debug)]
pub struct Detokenizer<'a> {
    vocabulary: Option<&'a Vocabulary>,
    tokens: Vec<TokenId>,
    emitted: usize,
}

impl<'a> Detokenizer<'a> {
    /// A detokenizer over `vocabulary`, with nothing decoded yet.
    #[must_use]
    pub const fn new(vocabulary: Option<&'a Vocabulary>) -> Self {
        Self {
            vocabulary,
            tokens: Vec::new(),
            emitted: 0,
        }
    }

    /// Adds one id and returns the text it completed, which may be empty.
    ///
    /// An empty answer is normal and is not an error: a lead byte of a
    /// multi-byte character completes nothing, and the text arrives with the
    /// id that finishes it.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::Decode`] when the run so far cannot be decoded.
    pub fn push(&mut self, token: TokenId) -> Result<String, TokenizerError> {
        self.tokens.push(token);
        let Some(vocabulary) = self.vocabulary else {
            return Ok(String::new());
        };
        let whole = vocabulary.decode(&self.tokens)?;
        // A shrinking decode is possible in principle - a vocabulary is free to
        // render a pair differently from its parts - and `get` rather than an
        // index is what keeps that a missing fragment rather than a panic.
        let fresh = whole.get(self.emitted..).unwrap_or_default().to_owned();
        self.emitted = whole.len();
        Ok(fresh)
    }

    /// The ids seen so far.
    #[must_use]
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }
}

#[cfg(test)]
mod bos_tests {
    use super::{TokenId, Vocabulary, bos_from_config};
    use std::str::FromStr as _;

    /// A vocabulary declaring an opening token and one word.
    fn stub() -> tokenizers::Tokenizer {
        let document = serde_json::json!({
            "version": "1.0",
            "truncation": serde_json::Value::Null,
            "padding": serde_json::Value::Null,
            "added_tokens": [{
                "id": 2, "content": "<bos>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true,
            }],
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": serde_json::Value::Null,
            "decoder": serde_json::Value::Null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "word": 1, "<bos>": 2},
                "unk_token": "<unk>",
            },
        })
        .to_string();
        tokenizers::Tokenizer::from_str(&document).expect("the stub vocabulary loads")
    }

    /// Writes a `tokenizer_config.json` and reads the opening token back out of
    /// it, which is the whole of what the sidecar is consulted for.
    fn bos_for(config: &serde_json::Value) -> Option<TokenId> {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let path = dir.path().join("tokenizer_config.json");
        std::fs::write(&path, config.to_string()).expect("the config is written");
        bos_from_config(&path, &stub())
    }

    #[test]
    fn a_prompt_with_no_opening_token_is_given_one() {
        let vocabulary = Vocabulary::from_tokenizer(stub()).with_bos(Some(TokenId::new(2)));
        let tokens = vocabulary
            .encode_prompt("word")
            .expect("the prompt encodes");
        assert_eq!(
            tokens.first().copied().map(TokenId::get),
            Some(2),
            "the checkpoint's opening token goes in front of a raw prompt",
        );
        assert_eq!(tokens.len(), 2, "and nothing else is added: {tokens:?}");
    }

    #[test]
    fn a_prompt_that_already_opens_with_it_is_left_alone() {
        let vocabulary = Vocabulary::from_tokenizer(stub()).with_bos(Some(TokenId::new(2)));
        let tokens = vocabulary
            .encode_prompt("<bos>word")
            .expect("the prompt encodes");
        assert_eq!(
            tokens.iter().map(|t| t.get()).collect::<Vec<_>>(),
            vec![2, 1],
            "a rendered chat template has already written the token, and a second \
             one is a prompt shape the model was never trained on",
        );
    }

    #[test]
    fn a_checkpoint_that_declares_none_gets_none() {
        let vocabulary = Vocabulary::from_tokenizer(stub());
        let tokens = vocabulary
            .encode_prompt("word")
            .expect("the prompt encodes");
        assert_eq!(
            tokens.iter().map(|t| t.get()).collect::<Vec<_>>(),
            vec![1],
            "the Qwen family names no opening token and must not be given one",
        );
    }

    #[test]
    fn the_sidecar_is_read_the_way_hugging_face_writes_it() {
        assert_eq!(
            bos_for(&serde_json::json!({"bos_token": "<bos>"})).map(TokenId::get),
            Some(2),
            "the bare-string spelling",
        );
        assert_eq!(
            bos_for(&serde_json::json!({"bos_token": {"content": "<bos>"}})).map(TokenId::get),
            Some(2),
            "the object spelling means the same thing",
        );
        assert_eq!(
            bos_for(&serde_json::json!({"bos_token": "<bos>", "add_bos_token": false})),
            None,
            "an explicit refusal settles it, whatever bos_token says beside it",
        );
        assert_eq!(
            bos_for(&serde_json::json!({"eos_token": "</s>"})),
            None,
            "a checkpoint that names no opening token declares none",
        );
        assert_eq!(
            bos_for(&serde_json::json!({"bos_token": "<nope>"})),
            None,
            "a name this vocabulary does not declare resolves to no id at all",
        );
    }
}

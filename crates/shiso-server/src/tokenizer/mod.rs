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
pub mod literal_text;
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
    /// Which spellings of `inner` are markers, read once at load.
    markers: shiso_decision::literal::Markers,
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
        Ok(Self {
            markers: shiso_decision::literal::Markers::of(&inner),
            inner,
            bos,
        })
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
    pub fn from_tokenizer(inner: tokenizers::Tokenizer) -> Self {
        Self {
            markers: shiso_decision::literal::Markers::of(&inner),
            inner,
            bos: None,
        }
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

    /// Encodes a prompt as [`Self::encode_prompt`] does, except that the byte
    /// ranges in `literal` are read as ordinary text even where they spell an
    /// added token.
    ///
    /// The ranges are text a caller wrote that happens to spell a marker; see
    /// [`literal_text`]. The prompt is encoded once as usual, and each stretch
    /// between two added tokens the template wrote that holds a literal range
    /// is encoded again with added tokens read as text. Added tokens are split
    /// out before any other step of the encoder, so such a stretch is encoded
    /// on its own in the first pass as well, and re-encoding it alone gives the
    /// ids the same characters would get anywhere in a prompt.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::Encode`] when the text cannot be encoded.
    pub fn encode_prompt_with_literals(
        &self,
        text: &str,
        literal: &[std::ops::Range<usize>],
    ) -> Result<Vec<TokenId>, TokenizerError> {
        if literal.is_empty() {
            return self.encode_prompt(text);
        }
        let mut tokens = shiso_decision::literal::encode_with_literals(&self.inner, text, literal)
            .map_err(|error| TokenizerError::Encode {
                message: error.to_string(),
            })?;
        if let Some(bos) = self.bos
            && tokens.first() != Some(&bos)
        {
            tokens.insert(0, bos);
        }
        Ok(tokens)
    }

    /// Encodes one piece of a decision prompt: no opening token, and caller
    /// text in `literal` that spells a marker is read as text. The one
    /// implementation lives in `shiso-decision`, which the in-process decider
    /// shares.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::Encode`] when the text cannot be encoded.
    pub fn encode_piece(
        &self,
        text: &str,
        literal: &[std::ops::Range<usize>],
    ) -> Result<Vec<TokenId>, TokenizerError> {
        shiso_decision::literal::encode_piece(&self.inner, &self.markers, text, literal).map_err(
            |error| TokenizerError::Encode {
                message: error.to_string(),
            },
        )
    }

    /// Encodes `text` as [`Self::encode`] does, and says where in it the
    /// encoder last restarted: just past its last added token.
    ///
    /// Added tokens are split out before any other step of the encoder (see
    /// [`Self::encode_prompt_with_literals`]), so the stretch after the last
    /// one is normalized, pre-tokenized and merged on its own - its ids are
    /// the ids of that stretch encoded alone, whatever came before it. What a
    /// caller appends to `text` can only change the ids from that point on,
    /// unless the appended text completes an added token that started before
    /// it, which [`AddedSeam::holds_after`] rules out per appended string.
    ///
    /// `None` for the seam when the encoding holds no added token, and when
    /// this vocabulary declares any added token that strips the whitespace
    /// around it or matches only as a whole word: either lets a match depend
    /// on characters across the split, so the split is not a clean restart.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::Encode`] when the text cannot be encoded.
    pub fn encode_with_seam(
        &self,
        text: &str,
    ) -> Result<(Vec<TokenId>, Option<AddedSeam>), TokenizerError> {
        let encoded = self
            .inner
            .encode(text, false)
            .map_err(|error| TokenizerError::Encode {
                message: error.to_string(),
            })?;
        let tokens: Vec<TokenId> = encoded
            .get_ids()
            .iter()
            .copied()
            .map(TokenId::new)
            .collect();
        let added = self.inner.get_added_tokens_decoder();
        if added
            .values()
            .any(|token| token.lstrip || token.rstrip || token.single_word)
        {
            return Ok((tokens, None));
        }
        let seam = encoded
            .get_ids()
            .iter()
            .zip(encoded.get_offsets())
            .enumerate()
            .rev()
            .find(|(_, (id, _))| added.contains_key(id))
            .map(|(at, (_, offsets))| AddedSeam {
                text: offsets.1,
                tokens: at + 1,
                spellings: added
                    .values()
                    .map(|token| token.content.clone())
                    .filter(|content| !content.is_empty())
                    .collect(),
            });
        Ok((tokens, seam))
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

    /// The text of every added token this vocabulary marks special.
    ///
    /// What caller text is kept from encoding as a token; see
    /// [`literal_text`]. Special only, because those are the tokens the
    /// literal encoding reads as text: an added token that is not special is
    /// split out of any text, which is what the families declaring one want.
    #[must_use]
    pub fn special_spellings(&self) -> Vec<String> {
        self.inner
            .get_added_tokens_decoder()
            .into_values()
            .filter(|token| token.special)
            .map(|token| token.content)
            .collect()
    }

    /// The ids a model must never generate: every special added token except
    /// the ones that end a turn and the ones a generated tag is spelled with.
    ///
    /// A model quoting a control token - writing `<|im_start|>` into its own
    /// reasoning - tends to pick the token's single id rather than its text,
    /// and that id then is the control token: a turn opener in the middle of
    /// an answer, or a turn closer that stops it mid-thought. Taking those ids
    /// off the table leaves the model the text pieces, which is what a quote
    /// is. The stop ids stay, or no turn could end; so does any special whose
    /// text is part of a tag the output parser reads (`tags`), or the model
    /// could not open a reasoning channel or call a tool. Sorted, so a boot log
    /// and a test see the same order.
    #[must_use]
    pub fn generation_suppressed(
        &self,
        stop: shiso_core::StopTokens,
        tags: &[&str],
    ) -> Vec<TokenId> {
        let mut ids: Vec<TokenId> = self
            .inner
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, token)| token.special && !token.content.is_empty())
            .filter(|(_, token)| !tags.iter().any(|tag| tag.contains(token.content.as_str())))
            .map(|(id, _)| TokenId::new(id))
            .filter(|id| !stop.contains(*id))
            .collect();
        ids.sort_unstable_by_key(|id| id.get());
        ids
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
        .with_audio(placement.audio_marker.and_then(|m| self.marker_id(m)))
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

    /// The bracket ids around an audio clip's run, all or nothing as for images.
    #[must_use]
    pub fn audio_wrap_ids(&self, placement: &MediaPlacement) -> Option<(TokenId, TokenId)> {
        let (open, close) = placement.wrap_audio?;
        Some((self.marker_id(open)?, self.marker_id(close)?))
    }
}

/// Where an encoding last restarted: just past its last added token.
///
/// Returned by [`Vocabulary::encode_with_seam`]. Encoding `text[self.text..]`
/// followed by an appended string gives the ids of `text` followed by that
/// string from index [`AddedSeam::tokens`] on, whenever
/// [`AddedSeam::holds_after`] says the appended string cannot complete an added
/// token that began before the seam. That is what lets a caller check many
/// strings appended to one long text at the cost of the short stretch after
/// the seam each, instead of the whole text each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedSeam {
    /// The byte in the text just past the last added token.
    pub text: usize,
    /// The index in the ids just past it.
    pub tokens: usize,
    /// Every added token's spelling, for [`AddedSeam::holds_after`].
    spellings: Vec<String>,
}

impl AddedSeam {
    /// Whether the seam still splits `text` followed by `appended`.
    ///
    /// The one way appending can reach behind the seam is an added token
    /// spelled as some suffix of `text` that starts before the seam, followed
    /// by a prefix of `appended`: the encoder would match it across the seam.
    /// A match that starts at or after the seam lies inside the stretch the
    /// caller encodes, so it is seen there. `appended` must also be ASCII, so
    /// that normalizing it cannot compose it with the character before it.
    #[must_use]
    pub fn holds_after(&self, text: &str, appended: &str) -> bool {
        if !appended.is_ascii() {
            return false;
        }
        let after = text.len().saturating_sub(self.text);
        !self.spellings.iter().any(|spelling| {
            (1..=spelling.len().min(appended.len())).any(|tail| {
                let split = spelling.len() - tail;
                spelling.is_char_boundary(split)
                    && appended
                        .as_bytes()
                        .starts_with(&spelling.as_bytes()[split..])
                    && split > after
                    && text.ends_with(&spelling[..split])
            })
        })
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
        // An incomplete multi-byte sequence at the tail decodes to U+FFFD; hold
        // it back until subsequent tokens complete the character.
        if whole.ends_with('\u{FFFD}') {
            return Ok(String::new());
        }
        // A shrinking decode is possible in principle - a vocabulary is free to
        // render a pair differently from its parts - and `get` rather than an
        // index is what keeps that a missing fragment rather than a panic.
        let fresh = whole.get(self.emitted..).unwrap_or_default().to_owned();
        self.emitted = whole.len();
        Ok(fresh)
    }

    /// Delivers whatever text was held back because the tail ended with an
    /// incomplete multi-byte character.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::Decode`] when the run so far cannot be decoded.
    pub fn flush(&mut self) -> Result<String, TokenizerError> {
        let Some(vocabulary) = self.vocabulary else {
            return Ok(String::new());
        };
        let whole = vocabulary.decode(&self.tokens)?;
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

#[cfg(test)]
mod detokenizer_tests {
    use super::{Detokenizer, TokenId, Vocabulary};
    use std::str::FromStr as _;

    /// A vocabulary with byte-level pieces that split an emoji across two tokens.
    fn byte_level_stub() -> tokenizers::Tokenizer {
        let document = serde_json::json!({
            "version": "1.0",
            "truncation": serde_json::Value::Null,
            "padding": serde_json::Value::Null,
            "added_tokens": [],
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": serde_json::Value::Null,
            "post_processor": serde_json::Value::Null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            },
            "model": {
                "type": "BPE",
                "vocab": {
                    "ĠðŁĺ": 0,
                    "Ĭ": 1,
                    "hello": 2,
                },
                "merges": [],
            },
        })
        .to_string();
        tokenizers::Tokenizer::from_str(&document).expect("the stub vocabulary loads")
    }

    #[test]
    fn multi_byte_character_is_held_back_until_completed() {
        let tokenizer = byte_level_stub();
        let vocabulary = Vocabulary::from_tokenizer(tokenizer);
        let mut detok = Detokenizer::new(Some(&vocabulary));

        // Token 2 is "hello"
        let first = detok.push(TokenId::new(2)).expect("pushes hello");
        assert_eq!(first, "hello");

        // Token 0 is " " + first two bytes of emoji
        let second = detok
            .push(TokenId::new(0))
            .expect("pushes incomplete byte piece");
        assert_eq!(
            second, "",
            "an incomplete character at the tail produces no text"
        );

        // Token 1 completes the emoji into " 😊"
        let third = detok.push(TokenId::new(1)).expect("completes emoji");
        assert_eq!(
            third, " 😊",
            "the completed multi-byte character is delivered in full"
        );
    }

    #[test]
    fn incomplete_tail_is_delivered_on_flush() {
        let tokenizer = byte_level_stub();
        let vocabulary = Vocabulary::from_tokenizer(tokenizer);
        let mut detok = Detokenizer::new(Some(&vocabulary));

        let first = detok
            .push(TokenId::new(0))
            .expect("pushes incomplete byte piece");
        assert_eq!(first, "");

        let tail = detok.flush().expect("flushes tail");
        assert_eq!(tail, " \u{FFFD}", "flushing delivers any uncompleted tail");
    }

    /// The marker the template wrote stays one id; the copy inside the literal
    /// range is encoded as the characters it is made of.
    #[test]
    fn a_literal_range_is_encoded_as_text_and_the_rest_is_not() {
        use std::str::FromStr as _;
        let document = serde_json::json!({
            "version": "1.0",
            "truncation": serde_json::Value::Null,
            "padding": serde_json::Value::Null,
            "added_tokens": [{
                "id": 4, "content": "<m>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true,
            }],
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": {
                "type": "ByteLevel", "add_prefix_space": false,
                "trim_offsets": false, "use_regex": false
            },
            "post_processor": serde_json::Value::Null,
            "decoder": serde_json::Value::Null,
            "model": {
                "type": "BPE",
                "vocab": {"<": 0, "m": 1, ">": 2, "a": 3},
                "merges": [],
            },
        })
        .to_string();
        let tokenizer = tokenizers::Tokenizer::from_str(&document).expect("loads");
        let vocabulary = Vocabulary::from_tokenizer(tokenizer);
        let text = "<m>a<m>a";
        let ids = |tokens: Vec<TokenId>| tokens.iter().map(|t| t.get()).collect::<Vec<_>>();
        assert_eq!(
            ids(vocabulary.encode_prompt(text).expect("encodes")),
            vec![4, 3, 4, 3]
        );
        assert_eq!(
            ids(vocabulary
                .encode_prompt_with_literals(text, std::slice::from_ref(&(4..7)))
                .expect("encodes")),
            vec![4, 3, 0, 1, 2, 3]
        );
    }

    /// The stop ids stay pickable or no turn could end, a special spelled
    /// inside a parser tag stays or no channel could open, and an added token
    /// that is not special is the parser's business: none of those is
    /// suppressed. Every other special is.
    #[test]
    fn generation_suppresses_specials_but_stops_and_tags() {
        use std::str::FromStr as _;
        let added: Vec<serde_json::Value> = [
            (1, "<|im_start|>", true),
            (2, "<|im_end|>", true),
            (3, "<|image_pad|>", true),
            (4, "<|channel>", true),
            (5, "<think>", false),
        ]
        .into_iter()
        .map(|(id, content, special)| {
            serde_json::json!({
                "id": id, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": special,
            })
        })
        .collect();
        let document = serde_json::json!({
            "version": "1.0",
            "truncation": serde_json::Value::Null,
            "padding": serde_json::Value::Null,
            "added_tokens": added,
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": serde_json::Value::Null,
            "decoder": serde_json::Value::Null,
            "model": {"type": "WordLevel", "vocab": {
                "<unk>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<|image_pad|>": 3,
                "<|channel>": 4, "<think>": 5,
            }, "unk_token": "<unk>"},
        })
        .to_string();
        let vocabulary =
            Vocabulary::from_tokenizer(tokenizers::Tokenizer::from_str(&document).expect("loads"));
        let stop = shiso_core::StopTokens::new(&[TokenId::new(2)]).expect("one stop");
        let ids = vocabulary.generation_suppressed(stop, &["<|channel>thought\n", "<think>"]);
        assert_eq!(ids, vec![TokenId::new(1), TokenId::new(3)]);
    }

    /// The model-side half of the boot contract, mirrored from the engine
    /// stack's documented refusal: a suppression list longer than the buffer
    /// it is written into, or an id at or past the logits width, is refused
    /// rather than truncated or skipped - either would leave a control token
    /// pickable that the server said was not. The stub lets the boot helper
    /// be tested without a device; what the tests below pin is that the
    /// server propagates the refusal as a boot failure instead of sanitizing
    /// the list into something the model would take.
    #[derive(Debug)]
    struct BoundCheckingModel {
        info: shiso_core::ModelInfo,
    }

    impl BoundCheckingModel {
        /// A model whose logits rows are `vocab` wide, ending turns on id 0.
        fn with_width(vocab: u32) -> Self {
            Self {
                info: shiso_core::ModelInfo {
                    context_capacity: 8,
                    vocab,
                    hidden: 8,
                    n_layers: 1,
                    stop_tokens: shiso_core::StopTokens::one(TokenId::new(0)),
                },
            }
        }

        fn refused(what: &'static str) -> shiso_core::CoreError {
            shiso_core::CoreError::Unsupported {
                what,
                backend: "test",
            }
        }
    }

    impl shiso_core::Model for BoundCheckingModel {
        fn info(&self) -> &shiso_core::ModelInfo {
            &self.info
        }

        fn open_session(&mut self) -> shiso_core::CoreResult<shiso_core::SessionId> {
            Err(Self::refused("sessions on the suppression-bound stub"))
        }

        fn close_session(&mut self, _session: shiso_core::SessionId) -> shiso_core::CoreResult<()> {
            Err(Self::refused("sessions on the suppression-bound stub"))
        }

        fn prefill_chunk(
            &mut self,
            _session: shiso_core::SessionId,
            _tokens: &[TokenId],
        ) -> shiso_core::CoreResult<()> {
            Err(Self::refused("prefill on the suppression-bound stub"))
        }

        fn prefill_last(
            &mut self,
            _session: shiso_core::SessionId,
            _tokens: &[TokenId],
        ) -> shiso_core::CoreResult<shiso_core::Logits<'_>> {
            Err(Self::refused("prefill on the suppression-bound stub"))
        }

        fn decode_batch(
            &mut self,
            _steps: &[shiso_core::DecodeStep],
        ) -> shiso_core::CoreResult<shiso_core::Logits<'_>> {
            Err(Self::refused("decode on the suppression-bound stub"))
        }

        fn suppress_tokens(&mut self, ids: &[TokenId]) -> shiso_core::CoreResult<()> {
            let capacity = shiso_engine::sparse::scratch::SUPPRESS_CAPACITY;
            if u32::try_from(ids.len())
                .ok()
                .is_none_or(|count| count > capacity)
            {
                return Err(shiso_core::CoreError::InvalidValue {
                    key: "suppressed token count",
                    value: format!("{} (at most {capacity})", ids.len()),
                });
            }
            if let Some(bad) = ids.iter().find(|id| id.get() >= self.info.vocab) {
                return Err(shiso_core::CoreError::InvalidValue {
                    key: "suppressed token id",
                    value: format!("{} (vocabulary {})", bad.get(), self.info.vocab),
                });
            }
            Ok(())
        }

        fn teardown(&mut self) -> shiso_core::CoreResult<()> {
            Ok(())
        }
    }

    /// The no-parser scheme pair, which asks for no generated tags: the tests
    /// below are about the two bounds, not about tag carve-outs.
    const NO_PARSERS: (
        shiso_parsers::ReasoningScheme,
        shiso_parsers::ToolCallScheme,
    ) = (
        shiso_parsers::ReasoningScheme::None,
        shiso_parsers::ToolCallScheme::None,
    );

    /// A `WordLevel` vocabulary whose added tokens are exactly `specials`, so a
    /// test can place special ids wherever a checkpoint file could.
    fn vocabulary_with_specials(specials: &[(u32, &str)]) -> Vocabulary {
        let mut vocab = serde_json::Map::new();
        vocab.insert("<unk>".to_string(), serde_json::Value::from(0));
        let added: Vec<serde_json::Value> = specials
            .iter()
            .map(|(id, content)| {
                vocab.insert((*content).to_string(), serde_json::Value::from(*id));
                serde_json::json!({
                    "id": id, "content": content, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true,
                })
            })
            .collect();
        let document = serde_json::json!({
            "version": "1.0",
            "truncation": serde_json::Value::Null,
            "padding": serde_json::Value::Null,
            "added_tokens": added,
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": serde_json::Value::Null,
            "decoder": serde_json::Value::Null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
        })
        .to_string();
        Vocabulary::from_tokenizer(
            tokenizers::Tokenizer::from_str(&document).expect("the specials vocabulary loads"),
        )
    }

    /// A special whose id reaches past the logits width is a vocabulary that
    /// disagrees with the model, and boot must end over it. Dropping the id
    /// would boot a server that can still generate a control token the
    /// suppression list said was off the table - the exact failure the
    /// refusal exists to prevent.
    #[test]
    fn a_special_id_past_the_logits_width_refuses_boot() {
        let vocabulary = vocabulary_with_specials(&[(900, "<|far|>")]);
        let mut model = BoundCheckingModel::with_width(64);
        let error = crate::suppress_control_tokens(&mut model, Some(&vocabulary), NO_PARSERS)
            .expect_err("a special id past the logits width refuses boot");
        let message = format!("{error:#}");
        assert!(
            message.contains("suppressed token id") && message.contains("900 (vocabulary 64)"),
            "the boot failure names the offending id and the width it passed: {message}"
        );
        // The refusal is about the width, not about refusing everything: the
        // same model boots when the special fits inside it.
        let tame = vocabulary_with_specials(&[(2, "<|near|>")]);
        assert!(
            crate::suppress_control_tokens(&mut model, Some(&tame), NO_PARSERS).is_ok(),
            "a special inside the logits width still boots"
        );
    }

    /// More specials than the suppression buffer holds is the same failure
    /// from the other side: the engine refuses rather than truncate, and boot
    /// carries that refusal out rather than serving with a silently
    /// shortened list.
    #[test]
    fn more_specials_than_the_suppression_capacity_refuses_boot() {
        let capacity = usize::try_from(shiso_engine::sparse::scratch::SUPPRESS_CAPACITY)
            .expect("the suppression capacity fits any supported usize");
        let specials: Vec<(u32, String)> = (0..=capacity)
            .map(|i| {
                let id = u32::try_from(i + 100).expect("the special ids stay inside u32");
                (id, format!("<|s{i}>"))
            })
            .collect();
        let borrowed: Vec<(u32, &str)> = specials
            .iter()
            .map(|(id, content)| (*id, content.as_str()))
            .collect();
        let vocabulary = vocabulary_with_specials(&borrowed);
        // Wide enough that every id fits: the refusal can only be the count.
        let mut model = BoundCheckingModel::with_width(8_192);
        let error = crate::suppress_control_tokens(&mut model, Some(&vocabulary), NO_PARSERS)
            .expect_err("more specials than the suppression capacity refuses boot");
        let message = format!("{error:#}");
        assert!(
            message.contains("suppressed token count")
                && message.contains(&format!("{} (at most {capacity})", capacity + 1)),
            "the boot failure names the count and the capacity it passed: {message}"
        );
    }
}

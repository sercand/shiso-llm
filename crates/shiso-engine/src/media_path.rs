//! The content-keyed path a media session is filed under, shared by both
//! stacks.
//!
//! A picture is a run of one placeholder id, so two pictures of one grid are
//! the same token path, and a cache that filed a media prompt under its own
//! ids would serve one picture's rows for another's request. The host hands a
//! media session a [`MediaKey`] instead: the prompt with each run written in
//! ids derived from the picture's content, all above every vocabulary id. The
//! model still reads the prompt's own ids; only the cache reads the key.
//!
//! The rules for when such a session may be checkpointed, and where a resume
//! of it may stand, live here so the sparse and the hybrid stack cannot come
//! to disagree about them.

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_core::media::{KEY_ID_FLOOR, MediaKey, MediaSpan, PadTokens};

/// A media session's cache key, and its history as the cache reads it.
#[derive(Debug)]
pub(crate) struct SessionKey {
    /// The prompt's key path, as long as the prompt.
    pub(crate) prompt: Vec<TokenId>,
    /// The prompt's items, in prompt order.
    pub(crate) spans: Vec<MediaSpan>,
    /// The session's history with every position the prompt keyed written in
    /// its key id.
    ///
    /// Kept rather than derived because a commit reads it once per span and a
    /// decode step is one span: rebuilding it there would make every token
    /// cost a pass over the conversation.
    pub(crate) path: Vec<TokenId>,
}

impl SessionKey {
    /// The key `key` for a session about to read `prompt`, checked to be
    /// `prompt` with only its spans rewritten.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a key of another length, or one that
    /// differs from `prompt` at a position no span covers.
    pub(crate) fn for_prompt(prompt: &[TokenId], key: &MediaKey<'_>) -> CoreResult<Self> {
        let path = key.path();
        let differs = path.len() != prompt.len()
            || path
                .iter()
                .zip(prompt)
                .any(|(keyed, own)| keyed != own && keyed.get() < KEY_ID_FLOOR);
        if differs {
            return Err(CoreError::InvalidValue {
                key: "media cache key",
                value: format!(
                    "a {}-token key for a {}-token prompt that differs outside its spans",
                    path.len(),
                    prompt.len()
                ),
            });
        }
        Ok(Self {
            prompt: path.to_vec(),
            spans: key.spans().to_vec(),
            path: Vec::with_capacity(path.len()),
        })
    }

    /// Appends `tokens`, which sit at the positions from `at`, as the cache
    /// reads them.
    pub(crate) fn extend_keyed(&self, at: u32, tokens: &[TokenId], out: &mut Vec<TokenId>) {
        let at = usize::try_from(at).unwrap_or(usize::MAX);
        out.extend(tokens.iter().enumerate().map(|(i, token)| {
            self.prompt
                .get(at.saturating_add(i))
                .copied()
                .filter(|id| id.get() >= KEY_ID_FLOOR)
                .unwrap_or(*token)
        }));
    }

    /// Appends `tokens` at the positions from `at` to the kept path.
    pub(crate) fn push(&mut self, at: u32, tokens: &[TokenId]) {
        let mut path = std::mem::take(&mut self.path);
        self.extend_keyed(at, tokens, &mut path);
        self.path = path;
    }

    /// Whether `q` lies strictly inside one of the prompt's runs.
    #[must_use]
    pub(crate) fn splits(&self, q: u32) -> bool {
        self.spans.iter().any(|span| span.splits(q))
    }
}

/// Whether a session may be checkpointed at `q`: a text session anywhere, and
/// a media session only under a content key and never inside one of its runs.
#[must_use]
pub(crate) fn keyed_at(media: bool, key: Option<&SessionKey>, q: u32) -> bool {
    match key {
        None => !media,
        Some(key) => !key.splits(q),
    }
}

/// The path a resume of `prompt` looks up, and the pictures on it.
///
/// A keyed session matches under its key and reads under its own ids; the two
/// are the same length and differ only inside the spans, which
/// [`SessionKey::for_prompt`] checked.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] for a resume of a prompt other than the one the
/// key was set for.
pub(crate) fn resume_lookup(
    key: Option<&SessionKey>,
    prompt: &[TokenId],
) -> CoreResult<(Vec<TokenId>, Vec<MediaSpan>)> {
    match key {
        Some(key) if key.prompt.len() == prompt.len() => {
            Ok((key.prompt.clone(), key.spans.clone()))
        }
        Some(_) => Err(CoreError::InvalidValue {
            key: "media cache key",
            value: "a resume of a prompt other than the one the key was set for".into(),
        }),
        None => Ok((prompt.to_vec(), Vec::new())),
    }
}

/// The first `take` tokens of `rest`, cut short at the first picture that ends
/// inside them with more of the prompt after it.
///
/// A placeholder followed by anything else is a picture's last row: a prompt's
/// pictures are fenced by their template's own markers, so two never touch.
#[must_use]
pub(crate) fn picture_end_within(rest: &[TokenId], take: usize, pads: PadTokens) -> usize {
    let take = take.min(rest.len());
    (1..take)
        .find(|at| pads.matches(rest[at - 1]) && !pads.matches(rest[*at]))
        .unwrap_or(take)
}

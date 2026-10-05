//! A loaded model and the logits it produces.
//!
//! There is no `load` method on the trait. Each implementation has an inherent
//! constructor that returns an initialized value or an error, so a `dyn Model`
//! is always loaded and there is no half-built state for every caller to guard
//! against.

use std::num::NonZeroU32;

use crate::buffer::DeviceView;
use crate::error::{CoreError, CoreResult};
use crate::ids::{SessionId, TokenId};

/// The tokens that end a turn.
///
/// Up to four, because that is what any served family declares; a fifth is a
/// compile-time fact to widen this, not a runtime allocation. The width is
/// inline rather than a `Vec` for the reason the rest of this file is: a
/// scheduler holds one of these and reads it once per sampled token, and a heap
/// pointer on that path buys nothing.
///
/// A set rather than a single id because a family can end a turn several ways -
/// Gemma 4 states three - and a scheduler that knew only one would report a
/// finished turn as a truncated one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopTokens {
    /// The declared ids, in the order the family declared them, packed from the
    /// front. The first slot is always filled: both constructors refuse an empty
    /// set, so there is no `StopTokens` that ends no turn.
    ids: [Option<TokenId>; 4],
}

impl StopTokens {
    /// How many ids one family may declare.
    const WIDTH: usize = 4;

    /// The set of one, which is what every family with a single EOS declares.
    #[must_use]
    pub const fn one(id: TokenId) -> Self {
        Self {
            ids: [Some(id), None, None, None],
        }
    }

    /// The set a family declares, in its own order.
    ///
    /// A repeated id is taken once, because this is a set and a family that
    /// names the same id twice means it once - a document that lists its
    /// end-of-turn marker in two places is describing one way to end a turn, not
    /// two, and spending a slot on the second would make four declared ids look
    /// like five.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when `ids` is empty - a model that ends no
    /// turn is a config error rather than a model that never stops - or names
    /// more than four distinct ids, which is a reason to widen [`StopTokens`]
    /// rather than to drop an id that ends a turn.
    pub fn new(ids: &[TokenId]) -> CoreResult<Self> {
        let refused = || CoreError::InvalidValue {
            key: "stop_tokens",
            value: format!(
                "{} ids, which is not 1 to {} distinct",
                ids.len(),
                Self::WIDTH
            ),
        };
        let mut slots = [None; Self::WIDTH];
        let mut filled = 0;
        for id in ids {
            if slots.iter().flatten().any(|declared| declared == id) {
                continue;
            }
            let Some(slot) = slots.get_mut(filled) else {
                return Err(refused());
            };
            *slot = Some(*id);
            filled += 1;
        }
        if filled == 0 {
            return Err(refused());
        }
        Ok(Self { ids: slots })
    }

    /// Whether `id` is one of the ids that end a turn.
    #[must_use]
    pub fn contains(&self, id: TokenId) -> bool {
        self.ids.iter().flatten().any(|declared| *declared == id)
    }

    /// The first id, which is what a single-EOS caller records: the
    /// speculative decoding path, which passes one stop id to its verify pass
    /// because that is all a draft head's rejection sampler takes - a family's
    /// own `Speculating` implementation and `shiso-bench`'s `draft` bench both
    /// call this for it.
    ///
    /// A family that declares several puts the one that ends an assistant turn
    /// first, because that is the one a caller with room for exactly one wants.
    #[must_use]
    pub const fn primary(&self) -> TokenId {
        match self.ids[0] {
            Some(id) => id,
            // Unreachable: `one` fills this slot and `new` refuses an empty
            // slice. It is a value rather than a panic because a panic here
            // would be a `# Panics` clause on a method that cannot panic.
            None => TokenId::new(0),
        }
    }

    /// The declared ids, in the order the family declared them.
    pub fn iter(&self) -> impl Iterator<Item = TokenId> + '_ {
        self.ids.iter().flatten().copied()
    }
}

/// What a scheduler needs to know about a loaded model.
///
/// Four numbers and a stop set, and not one of them says which model this is.
/// That is the point: a scheduler sizes a readback buffer, checks a prompt
/// against a context, and recognizes the end of a turn, and none of those is a
/// reason for this crate to know that a checkpoint family exists.
///
/// The geometry a forward pass needs - head widths, layer kinds, rotary bases -
/// is not here and is not a smaller version of this. It belongs to whatever
/// runs the pass, and a model that needs to hand it to something else hands
/// over its own type rather than widening this one.
///
/// Deliberately NOT `#[non_exhaustive]`. That attribute lets a library add a
/// field without breaking its callers, and here the callers SHOULD break: a new
/// field is a fact about a model that no existing model crate is reporting, and
/// a compile error at each of them is the right way to find that out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInfo {
    /// Tokens one session may hold, prompt included.
    pub context_capacity: u32,
    /// Vocabulary size, which is the width of one logits row.
    pub vocab: u32,
    /// Residual-stream width.
    pub hidden: u32,
    /// Total layer slots.
    pub n_layers: u32,
    /// Every token that ends a turn.
    pub stop_tokens: StopTokens,
}

impl ModelInfo {
    /// Bytes of one logits row, which is what a caller reading a row allocates.
    #[must_use]
    pub const fn logits_row_bytes(&self) -> u64 {
        self.vocab as u64 * 4
    }
}

/// One decode step for one session: the token just accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeStep {
    /// Session to advance.
    pub session: SessionId,
    /// Token fed in for this step.
    pub token: TokenId,
}

/// One session's piece of a batched prompt pass: the tokens it reads from its
/// current frontier, which end its prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptRun<'a> {
    /// Session to advance.
    pub session: SessionId,
    /// The tokens it reads.
    pub tokens: &'a [TokenId],
}

/// A `[rows, vocab]` block of single-precision logits on the device.
///
/// Borrowed from the model that produced it, so the rows cannot be freed while
/// a sampler still holds a view of them.
#[derive(Debug, Clone, Copy)]
pub struct Logits<'a> {
    view: DeviceView<'a>,
    rows: u32,
    vocab: u32,
}

impl<'a> Logits<'a> {
    /// Wraps a device range as a logits block.
    ///
    /// The view must be exactly `rows * vocab * 4` bytes; a model that hands
    /// over a shorter range gets [`CoreError::ShapeMismatch`] here rather than an
    /// out-of-bounds row later.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when `view` is shorter than
    /// `rows * vocab * 4` bytes.
    pub fn new(view: DeviceView<'a>, rows: u32, vocab: u32) -> CoreResult<Self> {
        let expected = u64::from(rows) * u64::from(vocab) * 4;
        if view.len() < expected {
            return Err(CoreError::ShapeMismatch {
                field: "logits view",
                expected,
                got: view.len(),
            });
        }
        Ok(Self { view, rows, vocab })
    }

    /// Number of sequences in the block.
    #[must_use]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Logits per sequence, which is the model's vocabulary size.
    #[must_use]
    pub const fn vocab(&self) -> u32 {
        self.vocab
    }

    /// The whole block as one view, for a sampler that copies it out in bulk.
    #[must_use]
    pub const fn view(&self) -> DeviceView<'a> {
        self.view
    }

    /// Row `i`, which belongs to the `i`-th entry of the request that produced
    /// this block.
    ///
    /// `i >= rows` is [`CoreError::OutOfBounds`].
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when `i` is at or past [`Logits::rows`].
    pub fn row(&self, i: u32) -> CoreResult<DeviceView<'a>> {
        if i >= self.rows {
            return Err(CoreError::OutOfBounds {
                offset: u64::from(i) * u64::from(self.vocab) * 4,
                len: u64::from(self.vocab) * 4,
                buffer_bytes: u64::from(self.rows) * u64::from(self.vocab) * 4,
            });
        }
        let offset = u64::from(i) * u64::from(self.vocab) * 4;
        self.view.slice(offset, u64::from(self.vocab) * 4)
    }
}

/// A loaded model.
///
/// `Send` but deliberately not `Sync`: exactly one thread owns a model at a
/// time, which is what lets the session bookkeeping stay plain mutable state
/// instead of a lock. The scheduler thread holds a `Box<dyn Model>` and every
/// method is object-safe.
pub trait Model: Send + std::fmt::Debug {
    /// What a scheduler needs to know about this model.
    fn info(&self) -> &ModelInfo;

    /// Allocates a fresh, empty session.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the device state a session needs cannot be
    /// created.
    fn open_session(&mut self) -> CoreResult<SessionId>;

    /// Releases a session's state. A second call for the same id is
    /// [`CoreError::UnknownSession`], not a silent success.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] when `session` was never minted or has
    /// already been closed.
    fn close_session(&mut self, session: SessionId) -> CoreResult<()>;

    /// Advances a session by a non-final prompt chunk. Produces no logits,
    /// because none would be consumed.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] for a closed or unknown id,
    /// [`CoreError::ContextOverflow`] when the chunk pushes the session past
    /// [`ModelInfo::context_capacity`], and [`CoreError::Backend`] on a device
    /// failure.
    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()>;

    /// Advances a session by its final prompt chunk and returns the last-token
    /// row.
    ///
    /// A separate method rather than an `is_last: bool` with an optional return:
    /// the type then says whether logits exist.
    ///
    /// # Errors
    ///
    /// As [`Model::prefill_chunk`], plus [`CoreError::Backend`] on a device
    /// failure.
    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>>;

    /// Advances every member by one token. Row `i` of the result is `steps[i]`,
    /// and an empty slice is `Ok` with zero rows rather than an error.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] when any step names a closed or unknown
    /// session, and [`CoreError::Backend`] on a device failure.
    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>>;

    /// Reads several sessions' final prompt pieces in one pass, and returns
    /// one logits row per run in run order: row `r` is the row
    /// [`Model::prefill_last`] would return for `runs[r]`.
    ///
    /// A family that answers must leave every session exactly where
    /// `prefill_last` would. It must refuse with [`CoreError::Unsupported`],
    /// before reading anything, any call it cannot answer that way, so the
    /// caller can read the runs one at a time instead.
    ///
    /// # Errors
    ///
    /// The default is [`CoreError::Unsupported`] and reads nothing: most
    /// families have no such pass.
    fn prefill_last_many(&mut self, _runs: &[PromptRun<'_>]) -> CoreResult<Logits<'_>> {
        Err(CoreError::Unsupported {
            what: "a batched prompt pass",
            backend: "model",
        })
    }

    /// This model's media path, when it has one.
    ///
    /// `None` is a fact about the model rather than a refusal of a request: a
    /// family with no way to splice encoder rows into a prompt has nothing to
    /// answer here, and a caller holding `None` knows it before it has opened a
    /// session or read a prompt. The default is `None`, which is the truthful
    /// answer for every text-only family and keeps a new one from having to
    /// write a method that only says no.
    ///
    /// Deliberately not a pair of defaulted prefill methods that return
    /// "unsupported". Those would put the question at every call site and answer
    /// it one prefill too late.
    fn media_mut(&mut self) -> Option<&mut dyn crate::media::MediaModel> {
        None
    }

    /// The speculative path, when this build carries a draft head.
    ///
    /// `None` for every model that decodes one token a step, which is the
    /// truthful answer and keeps a family without a head from having to write a
    /// method that only says no.
    fn spec_mut(&mut self) -> Option<&mut dyn Speculating> {
        None
    }

    /// Keeps every token the model picks for itself off `ids`.
    ///
    /// A scheduler draws from logits rows it reads back, and masks those rows
    /// itself. What it cannot reach is a pick taken inside the model - the
    /// greedy picks of a speculative verify - so a model that takes picks of
    /// its own has to be told the list. The list replaces the last one, and is
    /// set before the model serves.
    ///
    /// The default is right for a model that takes no picks of its own and
    /// refuses for one that does, so a speculating family that forgot this
    /// method is a boot failure rather than a draft that lands on a token the
    /// server said could not be generated.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] from the default on a model with a
    /// speculative path, and whatever an implementation refuses: an id past
    /// the vocabulary, or a list longer than it can hold.
    fn suppress_tokens(&mut self, ids: &[TokenId]) -> CoreResult<()> {
        if ids.is_empty() || self.spec_mut().is_none() {
            return Ok(());
        }
        Err(CoreError::Unsupported {
            what: "suppressing tokens in a speculative path that does not implement it",
            backend: "model",
        })
    }

    /// Releases device resources, reporting failures.
    ///
    /// Implementations also free what they can from `Drop`, but `Drop` cannot
    /// return an error or impose ordering, so a caller that cares calls this.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when a device resource could not be released.
    /// Releasing continues past the first failure so as much as possible is
    /// reclaimed, and the first error is the one returned.
    fn teardown(&mut self) -> CoreResult<()>;
}

/// Sampling configuration for one request.
///
/// Parameters are separate from [`crate::Seed`] so one request's settings can be
/// shared by several sequences while each keeps its own PRNG stream.
#[derive(Debug, Clone, Copy, PartialEq)]
///
/// Not `#[non_exhaustive]`: this is a type the surfaces BUILD, field by field,
/// from what a caller asked for. Per the rule the workspace settled on, the
/// attribute belongs on what a crate hands out, not on what another crate has to
/// construct.
pub struct SamplingParams {
    /// Softmax divisor. Zero or below means argmax, with no division at all.
    pub temperature: f32,
    /// Size of the top-k stage, or `None` for no top-k stage. A sentinel value
    /// meaning "off" does not survive into this API.
    pub top_k: Option<NonZeroU32>,
    /// Cumulative-mass cutoff, as a fraction of the remaining mass.
    pub top_p: f32,
    /// Floor as a fraction of the highest surviving probability.
    pub min_p: f32,
}

impl SamplingParams {
    /// Neutral stage settings: temperature 1.0, no top-k, `top_p` 1.0,
    /// `min_p` 0.0.
    ///
    /// Every stage is a no-op, so a caller can layer one on top without first
    /// deciding what the untouched stages should do.
    ///
    /// **This is not greedy.** Temperature is 1.0, so [`SamplingParams::is_greedy`]
    /// is false and a sampler built from it draws. The name used to say otherwise
    /// and cost at least one reader a red test; greedy is `temperature: 0.0`.
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            min_p: 0.0,
        }
    }

    /// Whether sampling is a plain argmax.
    #[must_use]
    pub const fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::neutral()
    }
}

/// What one speculative step produced.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller reads all three, and the
/// acceptance pair is the number that says whether the head is earning the
/// width the verify pass pays for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Speculated {
    /// The tokens the caller has not seen yet, in order: the drafts the target
    /// agreed with, and then the target's own next pick.
    ///
    /// The token the caller passed in is *not* here. It was already the
    /// caller's, and a step that handed it back would emit it twice.
    pub produced: Vec<TokenId>,
    /// Guesses the target agreed with.
    pub accepted: u32,
    /// Guesses the head made.
    pub drafted: u32,
}

/// What a drafter may know about how the caller will choose tokens: the
/// sequence's sampling key and temperature.
///
/// A drafter that scores its candidates with the same noise the caller's keyed
/// draw will add proposes the token that draw is likely to land on, rather than
/// the most likely token. Only acceptance depends on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftHint {
    /// The key the caller's keyed draw uses.
    pub key: u64,
    /// The temperature it divides by; zero or below is a greedy caller.
    pub temperature: f32,
}

impl DraftHint {
    /// A greedy caller: no noise to follow.
    #[must_use]
    pub const fn greedy() -> Self {
        Self {
            key: 0,
            temperature: 0.0,
        }
    }
}

/// A drafted block and the target's logits over it, before anything is kept.
///
/// The first half of a two-phase speculative step. Row `i` of the logits is the
/// distribution the target puts on the token after `first, drafts[..i]`, so the
/// caller can choose each token exactly as it would have in one-token decoding -
/// with whatever sampling rule that position would have had - and keep the
/// drafts up to the first one it did not choose. Nothing is committed until
/// [`Speculating::settle_block`] is told how many were kept.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedBlock {
    /// The tokens drafted behind `first`, in order.
    pub drafts: Vec<TokenId>,
    /// `drafts.len() + 1` rows of `vocab` logits, back to back.
    pub logits: Vec<f32>,
    /// Entries in one row.
    pub vocab: usize,
}

impl VerifiedBlock {
    /// Row `i`'s logits, or an empty slice past the last row.
    #[must_use]
    pub fn row(&self, i: usize) -> &[f32] {
        self.logits
            .get(i.saturating_mul(self.vocab)..i.saturating_add(1).saturating_mul(self.vocab))
            .unwrap_or(&[])
    }

    /// Row `i`'s logits, writable, or an empty slice past the last row: what
    /// a sampler masks before it draws.
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        self.logits
            .get_mut(i.saturating_mul(self.vocab)..i.saturating_add(1).saturating_mul(self.vocab))
            .unwrap_or(&mut [])
    }
}

/// A model that can propose several tokens and verify them in one pass.
///
/// Separate from [`Model`] for the reason [`crate::media::MediaModel`] is:
/// most families cannot do this, and a defaulted method returning "unsupported"
/// would put the question at every call site and answer it one step too late.
///
/// # Greedy only
///
/// A step is accepted by comparing the draft against the target's *greedy*
/// pick. That is exactly right when the caller samples greedily and wrong
/// otherwise: with a temperature the caller might not have chosen the token the
/// comparison accepted on, and the committed prefix would be one the model
/// never sampled. A caller that samples must not call this.
pub trait Speculating: Model {
    /// Commits `first` and as many of the head's guesses behind it as the
    /// target agrees with.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::Unsupported`] when the
    /// session was not opened for speculation, and whatever the device reports.
    fn speculate(&mut self, session: SessionId, first: TokenId) -> CoreResult<Speculated>;

    /// One speculative step for several sessions, `steps[i]` being session
    /// `i`'s committed token, with what [`Speculating::speculate`] would have
    /// returned for each, in order.
    ///
    /// The default is exactly that: one step after another. A family whose
    /// verify can carry several sessions in one pass over its weights answers
    /// it in one, and has to leave every session where its own step would.
    ///
    /// # Errors
    ///
    /// As [`Speculating::speculate`] for any session.
    fn speculate_batch(&mut self, steps: &[(SessionId, TokenId)]) -> CoreResult<Vec<Speculated>> {
        steps
            .iter()
            .map(|(session, first)| self.speculate(*session, *first))
            .collect()
    }

    /// Drafts behind `first` and verifies the block, committing nothing: the
    /// first half of a step whose tokens the caller chooses itself. `hint` is
    /// how the caller will choose them, which a drafter may follow.
    ///
    /// `proposal` is what the caller already knows comes next - the fixed
    /// parts of a structure it is writing, say - and may be empty. A family
    /// verifies it in place of its own drafts wherever the two disagree. It
    /// changes which tokens are verified and never which are accepted: every
    /// row is still chosen by the caller's rule.
    ///
    /// This is what lets a sampled sequence speculate. Its caller draws each
    /// row's token by a rule that depends only on that row's logits and the
    /// token's position, keeps the drafts up to the first it did not draw, and
    /// hands that count to [`Speculating::settle_block`].
    ///
    /// The default is `None`: a family with only the greedy step speculates
    /// greedy sequences through [`Speculating::speculate`] and no others.
    ///
    /// # Errors
    ///
    /// As [`Speculating::speculate`].
    // None: this family has no two-phase step.
    fn verify_block(
        &mut self,
        session: SessionId,
        first: TokenId,
        hint: DraftHint,
        proposal: &[TokenId],
    ) -> CoreResult<Option<VerifiedBlock>> {
        let _ = (session, first, hint, proposal);
        Ok(None)
    }

    /// Commits `first` and the first `accepted` drafts of the block
    /// [`Speculating::verify_block`] returned for `session`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for a family with no two-phase step, or no
    /// block verified for `session`; [`CoreError::ShapeMismatch`] for more kept
    /// drafts than were drafted; and whatever the device reports.
    fn settle_block(
        &mut self,
        session: SessionId,
        block: VerifiedBlock,
        accepted: u32,
    ) -> CoreResult<()> {
        let _ = (session, block, accepted);
        Err(CoreError::Unsupported {
            what: "a two-phase speculative step on a family that has none",
            backend: "model",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::StopTokens;
    use crate::error::CoreError;
    use crate::ids::TokenId;

    /// The families that end a turn one way still have to answer both questions
    /// the set is asked: which id a single-EOS caller records, and whether a
    /// sampled token ended the turn.
    #[test]
    fn a_single_id_is_the_primary_and_the_only_member() {
        let stop = StopTokens::one(TokenId::new(248_044));
        assert_eq!(stop.primary(), TokenId::new(248_044));
        assert!(stop.contains(TokenId::new(248_044)));
        assert!(!stop.contains(TokenId::new(106)));
        assert_eq!(stop.iter().collect::<Vec<_>>(), vec![TokenId::new(248_044)]);
    }

    /// Gemma 4's three: `<eos>`, `<turn|>` and `<|tool_response>`.
    #[test]
    fn every_id_a_family_declares_ends_a_turn() {
        let ids = [TokenId::new(1), TokenId::new(106), TokenId::new(50)];
        let stop = StopTokens::new(&ids).expect("three ids fit");
        for id in ids {
            assert!(
                stop.contains(id),
                "{id:?} was declared and is not recognised"
            );
        }
        assert!(!stop.contains(TokenId::new(2)));
        assert_eq!(stop.primary(), TokenId::new(1));
        assert_eq!(stop.iter().collect::<Vec<_>>(), ids.to_vec());
    }

    /// A model that declares nothing would decode until its budget ran out, and
    /// the server would report every turn as truncated. That is a config error,
    /// not a default.
    #[test]
    fn a_set_with_no_ids_is_refused() {
        let error = StopTokens::new(&[]).expect_err("an empty set is not a model");
        assert!(matches!(error, CoreError::InvalidValue { key, .. } if key == "stop_tokens"));
    }

    /// A document that names its end-of-turn marker twice is describing one way
    /// to end a turn, not two.
    #[test]
    fn an_id_declared_twice_takes_one_slot() {
        let stop = StopTokens::new(&[TokenId::new(106), TokenId::new(106)])
            .expect("one distinct id is a set");
        assert_eq!(stop.iter().collect::<Vec<_>>(), vec![TokenId::new(106)]);
        assert_eq!(stop.primary(), TokenId::new(106));
        assert_eq!(stop, StopTokens::one(TokenId::new(106)));
    }

    /// Four is the width, and a fifth id is a reason to widen the type rather
    /// than a reason to drop an id that ends a turn.
    #[test]
    fn a_fifth_id_is_refused_rather_than_dropped() {
        let ids = [
            TokenId::new(1),
            TokenId::new(2),
            TokenId::new(3),
            TokenId::new(4),
            TokenId::new(5),
        ];
        assert!(StopTokens::new(&ids[..4]).is_ok());
        let error = StopTokens::new(&ids).expect_err("five do not fit");
        assert!(matches!(error, CoreError::InvalidValue { key, .. } if key == "stop_tokens"));
    }
}

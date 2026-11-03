//! Speculative decode: what a step proposes, what the target keeps, and what a
//! rejection has to put back.
//!
//! One decode step through this family can commit more than one token. A draft
//! head predicts the token *after* the next one, a chain of those gives `k`
//! guesses, the target verifies all of them in one `1 + k` row pass, and the
//! longest prefix whose argmax agrees is committed. Everything in this module is
//! the part of that which is arithmetic over tokens rather than a launch:
//! bounding `k`, proposing a continuation from the context, deciding how much of
//! a draft the target agreed with, and naming the byte ranges a rejection copies
//! back.
//!
//! # Why the decidable half is separate from the pass
//!
//! The same reason [`crate::sparse::plan`] exists. The numbers a draft pass
//! produces need the draft head's kernels to run; whether a rejected draft
//! leaves a session exactly as it found it does not - that is host arithmetic
//! over a logits row and a handful of byte ranges, and it is the half of
//! speculation that is *silently* wrong when it is wrong. A build that accepts
//! one token too many answers fluently from a state the target never computed,
//! which is indistinguishable from a working build until somebody compares two
//! runs.
//!
//! # Two drafters, and why one gates the other
//!
//! The head is one. The other is a prompt lookup: when the last few tokens
//! already occurred earlier in the context, whatever followed them there is a
//! free guess, and it is strongest exactly where a conversation echoes text back.
//! It never replaces the head's draft, only extends it, and only when the head's
//! first token agrees with it - two independent drafters agreeing is the evidence
//! that the earlier occurrence is the text being repeated. A disagreement costs
//! nothing, because the head's own chain is what an unaided step would have
//! drafted anyway.
//!
//! # This module is SHARED, and that constrains what may be added to it
//!
//! It is re-exported as `shiso_engine::mtp` and read by **both** stacks: the
//! sparse lane through `crate::sparse::*`, and the hybrid lane's
//! [`crate::draft`], whose header carries the ruling that put it here rather
//! than duplicating it. The reason that ruling is safe is this file's import
//! list: **nothing here may import anything but `shiso_core`.** No `Plan`, no
//! `Operand`, no `SparseStack`, no buffer.
//!
//! That is the whole guard. A `SparseStack`-shaped assumption cannot be added
//! to the decidable half without adding an import to this file, which is a
//! visible edit here rather than a silent behaviour change in a lane its
//! author was not thinking about. The 28 host tests in `mtp/tests.rs` run on
//! every `cargo test -p shiso-engine` and cover both lanes' use of it.
//!
//! # No environment knobs
//!
//! The reference reads its three lookup settings from the environment. Here they
//! are fields of [`NgramPolicy`], which a caller builds from its own command
//! line: a deployment that wants the lookup off asks for zero drafts, and a
//! process that was never told anything runs [`NgramPolicy::off`].

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;

/// Drafts one step may carry.
///
/// The verify pass is `1 + k` rows wide, so this is also what bounds the
/// per-position rollback state every recurrent slot has to keep: a deeper draft
/// is a wider slab on every session, which is why the ceiling is a constant here
/// rather than something a request can raise.
pub const MAX_DRAFT: u32 = 8;

/// Longest key the prompt lookup tries.
///
/// Past a handful of tokens a match is already certain enough that a longer key
/// only costs another pass over the history.
pub const NGRAM_MAX_MATCH: u32 = 6;

/// Shortest key the lookup will accept by default.
///
/// Five rather than three: a three-token key matches template scaffolding all
/// over a long conversation, and every firing measured at three in the reference
/// was such a false positive.
pub const NGRAM_MIN_MATCH: u32 = 5;

/// Proposals the failure ring remembers.
///
/// A ring rather than a set: the oldest entry is the least interesting one to
/// keep, and forgetting a failure costs exactly one more rejected draft.
const FAIL_RING: usize = 256;

/// How deep one session may speculate.
///
/// A newtype rather than a bare `u32` because the ceiling is not the caller's:
/// [`SpecDepth::new`] clamps to [`MAX_DRAFT`], so a value of this type is always
/// a depth the per-position rollback slabs were actually sized for. A plain
/// integer would let a request ask for twelve and get a rollback that restores a
/// position nothing recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct SpecDepth(u32);

impl SpecDepth {
    /// Plain decode: one row per step, no drafts, no rollback state.
    pub const NONE: Self = Self(0);

    /// The depth `asked` clamped to what this build can verify and roll back.
    #[must_use]
    pub const fn new(asked: u32) -> Self {
        Self(if asked > MAX_DRAFT { MAX_DRAFT } else { asked })
    }

    /// Drafts this depth allows.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether anything is drafted at all.
    #[must_use]
    pub const fn is_speculative(self) -> bool {
        self.0 != 0
    }

    /// Rows a verify pass at this depth runs, which is one more than the drafts.
    #[must_use]
    pub const fn verify_rows(self) -> u32 {
        self.0.saturating_add(1)
    }

    /// Per-position states a rollback at this depth may have to copy back.
    ///
    /// The same number as the drafts: the row the step commits is row zero of
    /// the verify pass and needs no restoring, so only the drafted rows behind it
    /// are recorded.
    #[must_use]
    pub const fn rollback_slots(self) -> u32 {
        self.0
    }
}

/// How many drafts one step may actually carry, which is the narrowest of four
/// ceilings.
///
/// A free function taking all four rather than a chain of `min` calls at the call
/// site, because three of the four are easy to forget and each one that is
/// forgotten fails somewhere else: past the context the verify pass writes cache
/// rows the session has no claim on, and past the caller's buffer the committed
/// tokens have nowhere to go.
///
/// `position` is where the step starts - the number of tokens the session has
/// already written - and `committed_cap` is how many tokens the caller can accept
/// back, the one it samples included.
#[must_use]
pub fn draft_budget(depth: SpecDepth, position: u32, ctx: u32, committed_cap: u32) -> u32 {
    // The committed token itself occupies a position and a slot, so both of the
    // remaining budgets are one less than the room left.
    let room_in_context = ctx.saturating_sub(position).saturating_sub(1);
    let room_in_output = committed_cap.saturating_sub(1);
    depth.get().min(room_in_context).min(room_in_output)
}

/// The settings the prompt lookup runs under.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one of these from its
/// own command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramPolicy {
    /// Tokens the lookup may propose. Zero turns it off.
    ///
    /// Off is the right default and not a placeholder. A hit turns a four-row
    /// verify into a nine-row one, which is worth it on text that echoes and is
    /// not worth it on text that does not, and the reference measured both.
    pub max_k: u32,
    /// Shortest key a match may be found by, clamped into
    /// `1..=NGRAM_MAX_MATCH` by [`NgramPolicy::new`].
    pub min_match: u32,
    /// The lowest id treated as a marker rather than as text.
    ///
    /// A continuation that walks into this range came from a turn boundary
    /// rather than from an echo, and proposing it would draft the start of a
    /// message the model has not decided to write.
    pub special_floor: TokenId,
}

impl NgramPolicy {
    /// The lookup, off.
    #[must_use]
    pub const fn off() -> Self {
        Self {
            max_k: 0,
            min_match: NGRAM_MIN_MATCH,
            special_floor: TokenId::new(u32::MAX),
        }
    }

    /// The lookup proposing at most `max_k` tokens, with every setting clamped
    /// to a range it can actually run in.
    ///
    /// `min_match` above [`NGRAM_MAX_MATCH`] would silently disable the lookup -
    /// no key that long is ever tried - so it is clamped rather than obeyed.
    #[must_use]
    pub const fn new(max_k: u32, min_match: u32, special_floor: TokenId) -> Self {
        let min_match = if min_match < 1 {
            1
        } else if min_match > NGRAM_MAX_MATCH {
            NGRAM_MAX_MATCH
        } else {
            min_match
        };
        Self {
            max_k: if max_k > MAX_DRAFT { MAX_DRAFT } else { max_k },
            min_match,
            special_floor,
        }
    }

    /// Whether the lookup runs at all.
    #[must_use]
    pub const fn is_on(self) -> bool {
        self.max_k != 0
    }
}

impl Default for NgramPolicy {
    fn default() -> Self {
        Self::off()
    }
}

/// What identifies one lookup proposal.
///
/// The position it was read from *and* a hash of what it proposes, because the
/// same continuation is reachable through keys of several lengths and a shorter
/// draft from the same place is a different bet. The position alone would
/// suppress both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProposalId {
    /// Where in the history the match ended.
    position: u32,
    /// A hash of the proposed tokens, in order.
    digest: u64,
}

impl ProposalId {
    /// The 64-bit FNV-1a offset basis.
    const BASIS: u64 = 14_695_981_039_346_656_037;
    /// The 64-bit FNV-1a prime.
    const PRIME: u64 = 1_099_511_628_211;

    /// The identity of the proposal `tokens`, read from `position`.
    #[must_use]
    pub fn new(position: u32, tokens: &[TokenId]) -> Self {
        let mut digest = Self::BASIS;
        for token in tokens {
            digest = (digest ^ u64::from(token.get())).wrapping_mul(Self::PRIME);
        }
        Self { position, digest }
    }
}

/// Proposals this generation's target has already rejected.
///
/// A rejection is remembered so the same match is not tried twice. A *partial*
/// acceptance counts as a rejection: the tokens after the divergence are the
/// wrong ones, and this draft from this position would be wrong the same way.
#[derive(Debug, Clone)]
pub struct FailRing {
    entries: Vec<ProposalId>,
    next: usize,
}

impl FailRing {
    /// An empty ring.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            next: 0,
        }
    }

    /// Records that the target disagreed with `id`.
    pub fn remember(&mut self, id: ProposalId) {
        if self.entries.len() < FAIL_RING {
            self.entries.push(id);
            self.next = self.entries.len() % FAIL_RING;
            return;
        }
        if let Some(slot) = self.entries.get_mut(self.next) {
            *slot = id;
        }
        self.next = self.next.saturating_add(1) % FAIL_RING;
    }

    /// Whether `id` has already been rejected.
    #[must_use]
    pub fn rejected(&self, id: ProposalId) -> bool {
        self.entries.contains(&id)
    }

    /// How many rejections are remembered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing has been rejected yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Forgets every rejection, which is what starting a new generation does.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next = 0;
    }
}

impl Default for FailRing {
    fn default() -> Self {
        Self::new()
    }
}

/// One continuation the lookup proposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proposal {
    /// The proposed tokens, in order.
    tokens: Vec<TokenId>,
    /// What identifies this proposal to the failure ring.
    id: ProposalId,
}

impl Proposal {
    /// The proposed tokens.
    #[must_use]
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    /// What identifies this proposal to the failure ring.
    #[must_use]
    pub const fn id(&self) -> ProposalId {
        self.id
    }

    /// How many tokens it proposes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether it proposes nothing, which [`propose`] never returns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Proposes a continuation of `history ++ [first]` by looking for it earlier in
/// the same text.
///
/// The longest key is tried first and the most recent earlier occurrence wins,
/// which is what makes the lookup follow the text being echoed rather than the
/// first time anything like it was said.
///
/// `generated_from` is where this generation's own tokens begin. A key is only
/// accepted when it lies entirely inside that range, because a prompt ends with
/// a turn header and a key reaching back into one matches every earlier turn
/// boundary in the context - which is where the reference's false positives came
/// from. The cost is that the lookup stays quiet for the first few tokens of a
/// generation, which is exactly when it has nothing to echo yet.
///
/// `None` means no key matched, or what matched was refused: it ran into the
/// marker range, or the failure ring has seen it fail already.
#[must_use]
pub fn propose(
    history: &[TokenId],
    first: TokenId,
    generated_from: u32,
    memory: &FailRing,
    policy: NgramPolicy,
) -> Option<Proposal> {
    if !policy.is_on() {
        return None;
    }
    // The sequence the key is taken from is the history with the token this step
    // is committing on the end: that token is part of the pattern being matched,
    // and a lookup that ignored it would propose the continuation of a different
    // phrase.
    let len = u32::try_from(history.len()).unwrap_or(u32::MAX);
    let seq_len = len.saturating_add(1);
    let at = |index: u32| -> Option<TokenId> {
        match index.cmp(&len) {
            std::cmp::Ordering::Less => history
                .get(usize::try_from(index).unwrap_or(usize::MAX))
                .copied(),
            std::cmp::Ordering::Equal => Some(first),
            std::cmp::Ordering::Greater => None,
        }
    };
    // Key lengths the lookup may try: the longest is the one whose key still lies
    // inside this generation and leaves a token before it to read from.
    let longest = NGRAM_MAX_MATCH
        .min(seq_len.saturating_sub(1))
        .min(seq_len.saturating_sub(generated_from));
    if longest < policy.min_match {
        return None;
    }
    // One pass over the history for every key length at once, rather than one
    // pass per length. A window's continuation depends only on where the window
    // ends, never on how long its key was, so keeping the newest proposal of
    // each longer key seen so far is the same answer as trying each length in
    // turn - and at a long context the history scan is the whole cost of a
    // step's lookup, on the scheduler thread, ahead of the verify launch.
    let longest_usize = usize::try_from(longest).ok()?;
    let min_usize = usize::try_from(policy.min_match).ok()?;
    // The key's tokens before `first`, which every window's own tokens before
    // its last are compared against, nearest first.
    let tail = history.get(history.len().checked_sub(longest_usize - 1)?..)?;
    let mut fallback = None;
    // Windows end at `end`, most recent first. The newest one that can still be
    // followed by a token ends at the last history position, and the oldest
    // whole window ends at `min_match - 1`.
    let mut hay = history;
    while let Some(end) = rfind(hay, first) {
        hay = &history[..end];
        if end + 1 < min_usize {
            break;
        }
        let matched = 1 + hay
            .iter()
            .rev()
            .zip(tail.iter().rev())
            .take_while(|(here, there)| here == there)
            .count();
        // Older windows only ever replace the fallback with a longer key.
        let beaten = fallback.as_ref().is_some_and(|&(best, _)| matched <= best);
        if matched < min_usize || beaten {
            continue;
        }
        let Some(proposal) = u32::try_from(end)
            .ok()
            .and_then(|end| read_forward(&at, end, seq_len, memory, policy))
        else {
            continue;
        };
        if matched >= longest_usize {
            return Some(proposal);
        }
        fallback = Some((matched, proposal));
    }
    fallback.map(|(_, proposal)| proposal)
}

/// The last position of `needle` in `hay`.
///
/// Compares a fixed-width block with no early exit before looking inside it,
/// which the compiler turns into vector compares: nearly every position is a
/// miss, so the scan is bounded by how fast it can rule a block out.
#[expect(
    clippy::needless_bitwise_bool,
    reason = "the eager `|` is what keeps the block compare branch-free and vectorised"
)]
fn rfind(hay: &[TokenId], needle: TokenId) -> Option<usize> {
    const BLOCK: usize = 32;
    let whole = hay.len() - hay.len() % BLOCK;
    if let Some(at) = hay[whole..].iter().rposition(|&token| token == needle) {
        return Some(whole + at);
    }
    hay[..whole]
        .chunks_exact(BLOCK)
        .rposition(|block| {
            block
                .iter()
                .fold(false, |hit, &token| hit | (token == needle))
        })
        .and_then(|block| {
            let start = block * BLOCK;
            hay[start..start + BLOCK]
                .iter()
                .rposition(|&token| token == needle)
                .map(|at| start + at)
        })
}

/// Reads the continuation after a matched window, refusing one that runs into
/// the marker range or one the ring has already seen fail.
fn read_forward(
    at: &impl Fn(u32) -> Option<TokenId>,
    end: u32,
    seq_len: u32,
    memory: &FailRing,
    policy: NgramPolicy,
) -> Option<Proposal> {
    let mut tokens = Vec::with_capacity(usize::try_from(policy.max_k).unwrap_or(0));
    let mut index = end.saturating_add(1);
    while u32::try_from(tokens.len()).unwrap_or(u32::MAX) < policy.max_k && index < seq_len {
        let token = at(index)?;
        if token >= policy.special_floor {
            // A proposal that reaches a marker is dropped whole rather than
            // truncated: the tokens before the marker are the tail of a message,
            // and continuing from them is a bet on the model ending its turn in
            // the same place.
            return None;
        }
        tokens.push(token);
        index = index.saturating_add(1);
    }
    if tokens.is_empty() {
        return None;
    }
    let id = ProposalId::new(end, &tokens);
    (!memory.rejected(id)).then_some(Proposal { tokens, id })
}

/// Which drafter a step's proposal came from.
///
/// Named rather than a `bool`, because the two are accounted for differently: a
/// lookup's rejections are remembered and a head's are not, and the two acceptance
/// rates are the numbers that say whether the lookup is earning its verify width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DraftSource {
    /// The draft head's own chain.
    Head,
    /// The prompt lookup, whose proposal extended the head's whole chain.
    Lookup(ProposalId),
    /// The caller's own proposal: structure it knew would come next. Not the
    /// drafter's guess, so it teaches the drafter's acceptance estimate
    /// nothing.
    Scaffold,
}

/// Whether a lookup proposal is worth taking in place of the head's chain.
///
/// Two conditions, and both are necessary. It has to be *longer* than what the
/// head drafted, because a proposal no longer than that is the same verify
/// width for a worse guess; and it has to agree with *every* guess the head
/// made, so that it only ever extends the head's chain. `head` is empty on a
/// step where the head drafted nothing, which is a step with no evidence at
/// all that the earlier occurrence is the text being echoed.
///
/// Agreement on the first guess alone was the rule once, and it let a lookup
/// that diverged from the head at its second guess replace guesses the head
/// gets right most of the time: on prose the head keeps its second guess in
/// nearly nine steps of ten that kept its first, and an echo that disagrees
/// with it there is more often a phrase that recurred than one being copied.
/// Where the two agree the verify rows they share are the head's own, so a
/// step that takes the lookup commits at least what the head alone would have.
#[must_use]
pub fn lookup_beats_head(proposal: &Proposal, head: &[TokenId]) -> bool {
    proposal.tokens.len() > head.len() && !head.is_empty() && proposal.tokens.starts_with(head)
}

/// How much of a draft the target agreed with.
///
/// `target` is the target's own pick for each drafted position: row `i` of the
/// verify pass is the distribution after the token `drafts[i]` proposed, so
/// `target[i]` is what the target would have written there. The accepted prefix
/// is the longest run of agreement, stopped at the turn's end - a draft may
/// continue past an end-of-turn token, and committing what follows would put
/// tokens after the end of the answer.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] when the target produced fewer picks than there
/// were drafts. Refusing is the point: a build that silently compared what it
/// had would accept a prefix on the strength of rows nothing computed.
pub fn accepted_prefix(drafts: &[TokenId], target: &[TokenId], eos: TokenId) -> CoreResult<u32> {
    if target.len() < drafts.len() {
        return Err(CoreError::ShapeMismatch {
            field: "verify rows",
            expected: u64::try_from(drafts.len()).unwrap_or(u64::MAX),
            got: u64::try_from(target.len()).unwrap_or(u64::MAX),
        });
    }
    let mut accepted = 0u32;
    for (draft, picked) in drafts.iter().zip(target) {
        if draft != picked {
            break;
        }
        accepted = accepted.saturating_add(1);
        if *draft == eos {
            break;
        }
    }
    Ok(accepted)
}

/// What one speculative step settled on.
///
/// The tokens to commit and how far the session has to be wound back, in one
/// value: a caller that got the two separately could commit the tokens and skip
/// the rewind, which is the failure that leaves a session standing on recurrent
/// state for positions it never committed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StepOutcome {
    /// The tokens this step commits, the one the caller sampled first.
    pub committed: Vec<TokenId>,
    /// The rewind the session owes, or `None` when every draft was accepted.
    pub rollback: Option<Rollback>,
    /// Which row of the verify pass carries the distribution the caller samples
    /// next, which is the number of drafts that were accepted.
    pub logits_row: u32,
    /// Where the proposal came from, or `None` on a step that drafted nothing.
    pub source: Option<DraftSource>,
}

impl StepOutcome {
    /// Settles one verified step.
    ///
    /// `first` is the token the caller sampled and is committed unconditionally;
    /// `drafts` are the guesses behind it and `target` the target's own picks for
    /// the same positions.
    ///
    /// # Errors
    ///
    /// Whatever [`accepted_prefix`] reports.
    pub fn settle(
        first: TokenId,
        drafts: &[TokenId],
        target: &[TokenId],
        eos: TokenId,
        source: Option<DraftSource>,
    ) -> CoreResult<Self> {
        let accepted = accepted_prefix(drafts, target, eos)?;
        let taken = usize::try_from(accepted).unwrap_or(0);
        let mut committed = Vec::with_capacity(taken.saturating_add(1));
        committed.push(first);
        committed.extend_from_slice(drafts.get(..taken).unwrap_or(&[]));
        let drafted = u32::try_from(drafts.len()).unwrap_or(u32::MAX);
        Ok(Self {
            committed,
            rollback: Rollback::new(accepted, drafted),
            logits_row: accepted,
            source,
        })
    }

    /// Whether the lookup's proposal should be written off after this step.
    ///
    /// A partial acceptance counts as a failure, for the reason [`FailRing`]
    /// gives: the tokens past the divergence are wrong, and the same draft from
    /// the same place would be wrong the same way.
    #[must_use]
    pub const fn failed_lookup(&self) -> Option<ProposalId> {
        match (self.rollback, self.source) {
            (Some(_), Some(DraftSource::Lookup(id))) => Some(id),
            _ => None,
        }
    }

    /// How many tokens this step commits.
    #[must_use]
    pub fn committed_len(&self) -> u32 {
        u32::try_from(self.committed.len()).unwrap_or(u32::MAX)
    }
}

/// A rewind the session owes after a draft the target did not agree with.
///
/// Constructed only when there is something to undo: [`Rollback::new`] answers
/// `None` for a step whose whole draft was accepted, so a value of this type
/// always names work, and a caller cannot hold one that means "nothing to do".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rollback {
    keep: u32,
}

impl Rollback {
    /// The rewind owed when `accepted` of `drafted` guesses were kept, or `None`
    /// when they all were.
    #[must_use]
    pub const fn new(accepted: u32, drafted: u32) -> Option<Self> {
        if accepted >= drafted {
            return None;
        }
        Some(Self { keep: accepted })
    }

    /// The per-position state to restore: the one recorded after row `keep` of
    /// the verify pass.
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.keep
    }

    /// Where in a per-position slab of `stride`-byte records this rewind's record
    /// begins.
    ///
    /// The whole of the addressing, in one place. A slab holds one record per
    /// drafted row in row order, so the record to copy back is at `keep *
    /// stride`, and the two call sites that need it - the recurrent slots and
    /// the n-gram window - would otherwise each compute it.
    #[must_use]
    pub fn offset(self, stride: u64) -> u64 {
        u64::from(self.keep).saturating_mul(stride)
    }
}

#[cfg(test)]
mod tests;

//! What speculation decides, checked without a device.
//!
//! Every case here is about a token count or a byte offset, which is the half of
//! speculative decoding that produces a wrong answer rather than an error when it
//! is wrong: accepting one draft too many commits a token the target never chose,
//! and skipping a rewind leaves a session standing on state for positions it
//! never committed. Both read as a working build.

use shiso_core::ids::TokenId;

use super::{
    DraftSource, FailRing, MAX_DRAFT, NGRAM_MAX_MATCH, NgramPolicy, ProposalId, Rollback,
    SpecDepth, StepOutcome, accepted_prefix, draft_budget, lookup_beats_head, propose,
};

/// Tokens, shortest way.
fn toks(ids: &[u32]) -> Vec<TokenId> {
    ids.iter().copied().map(TokenId::new).collect()
}

/// The marker floor the cases use, well above every text token they name.
const SPECIAL: TokenId = TokenId::new(1000);

/// A lookup that will propose up to `k` tokens on a three-token key.
fn lookup(k: u32) -> NgramPolicy {
    NgramPolicy::new(k, 3, SPECIAL)
}

// ---------------------------------------------------------------------------
// the depth and the budget
// ---------------------------------------------------------------------------

#[test]
fn a_depth_past_the_ceiling_is_clamped_rather_than_obeyed() {
    // The ceiling sizes the per-position rollback slabs on every session, so a
    // request that asked for twelve and got twelve would roll back from records
    // nothing wrote.
    assert_eq!(SpecDepth::new(12).get(), MAX_DRAFT);
    assert_eq!(SpecDepth::new(3).get(), 3);
    assert_eq!(SpecDepth::NONE.get(), 0);
    assert!(!SpecDepth::NONE.is_speculative());
    assert!(SpecDepth::new(1).is_speculative());
}

#[test]
fn a_verify_pass_is_one_row_wider_than_the_draft_and_rolls_back_one_narrower() {
    let depth = SpecDepth::new(4);
    assert_eq!(depth.verify_rows(), 5, "the committed row plus four drafts");
    assert_eq!(
        depth.rollback_slots(),
        4,
        "row zero is committed and needs no restoring"
    );
}

#[test]
fn the_budget_is_the_narrowest_of_the_context_and_the_callers_buffer() {
    let depth = SpecDepth::new(8);
    // Room to spare on both.
    assert_eq!(draft_budget(depth, 10, 1024, 64), 8);
    // Two positions left in the context: one for the committed token, one draft.
    assert_eq!(draft_budget(depth, 1022, 1024, 64), 1);
    // The context is exactly full for the committed token and nothing else.
    assert_eq!(draft_budget(depth, 1023, 1024, 64), 0);
    // The caller can take three tokens back, so two of them may be drafts.
    assert_eq!(draft_budget(depth, 10, 1024, 3), 2);
    // A caller with room for one token gets the plain path.
    assert_eq!(draft_budget(depth, 10, 1024, 1), 0);
}

#[test]
fn a_position_past_the_context_does_not_wrap_into_a_huge_budget() {
    // Saturating rather than checked on purpose: the caller refuses the step for
    // its own reasons, and this must not be the place that turns an overflow
    // into a draft that writes cache rows the session has no claim on.
    assert_eq!(draft_budget(SpecDepth::new(8), 4096, 1024, 64), 0);
}

// ---------------------------------------------------------------------------
// the accept rule
// ---------------------------------------------------------------------------

#[test]
fn the_accepted_prefix_stops_at_the_first_disagreement() {
    let drafts = toks(&[7, 8, 9, 10]);
    let target = toks(&[7, 8, 42, 10]);
    assert_eq!(
        accepted_prefix(&drafts, &target, SPECIAL).expect("equal lengths"),
        2,
        "the tokens after a divergence are the target's, not the draft's"
    );
}

#[test]
fn a_whole_draft_the_target_agreed_with_is_accepted_whole() {
    let drafts = toks(&[7, 8, 9]);
    assert_eq!(
        accepted_prefix(&drafts, &drafts, SPECIAL).expect("equal lengths"),
        3
    );
}

#[test]
fn acceptance_stops_at_the_end_of_the_turn_even_when_the_rest_agrees() {
    // A drafter may run past the end of an answer, and the target's argmax may
    // agree with it there. Committing those tokens would put text after the end
    // of the turn.
    let eos = TokenId::new(2);
    let drafts = toks(&[7, 2, 9]);
    let target = toks(&[7, 2, 9]);
    assert_eq!(
        accepted_prefix(&drafts, &target, eos).expect("equal lengths"),
        2,
        "the end-of-turn token is committed and nothing after it is"
    );
}

#[test]
fn fewer_verify_rows_than_drafts_is_refused_rather_than_compared_as_far_as_it_goes() {
    // The failure this refuses is silent: comparing what happened to be there
    // would accept a prefix on the strength of rows nothing computed.
    let drafts = toks(&[7, 8, 9]);
    let target = toks(&[7, 8]);
    let error = accepted_prefix(&drafts, &target, SPECIAL)
        .expect_err("a short verify block is not a shorter comparison");
    assert!(
        format!("{error}").contains("verify rows"),
        "the refusal names what was short: {error}"
    );
}

// ---------------------------------------------------------------------------
// the rewind
// ---------------------------------------------------------------------------

#[test]
fn a_fully_accepted_draft_owes_no_rewind_at_all() {
    // Not a rewind of zero rows: the type cannot represent "nothing to do", so a
    // caller holding one always has work.
    assert_eq!(Rollback::new(4, 4), None);
    assert_eq!(Rollback::new(5, 4), None);
}

#[test]
fn a_rejected_draft_rewinds_to_the_record_after_the_last_accepted_row() {
    let rollback = Rollback::new(2, 4).expect("two of four were rejected");
    assert_eq!(rollback.slot(), 2);
    // The slab holds one record per drafted row in row order.
    assert_eq!(rollback.offset(1024), 2048);
    assert_eq!(rollback.offset(0), 0);
}

#[test]
fn a_draft_rejected_outright_rewinds_to_the_state_the_step_started_from() {
    let rollback = Rollback::new(0, 3).expect("nothing was accepted");
    assert_eq!(rollback.slot(), 0);
    assert_eq!(
        rollback.offset(512),
        0,
        "record zero is the state after the committed row, which is where the \
         session stood before any draft ran"
    );
}

// ---------------------------------------------------------------------------
// settling a step
// ---------------------------------------------------------------------------

#[test]
fn a_step_commits_the_sampled_token_and_the_agreed_prefix_and_nothing_else() {
    let outcome = StepOutcome::settle(
        TokenId::new(5),
        &toks(&[7, 8, 9]),
        &toks(&[7, 42, 9]),
        SPECIAL,
        Some(DraftSource::Head),
    )
    .expect("equal lengths");
    assert_eq!(outcome.committed, toks(&[5, 7]));
    assert_eq!(outcome.committed_len(), 2);
    assert_eq!(
        outcome.logits_row, 1,
        "the row the caller samples next is the one after the last committed token"
    );
    assert_eq!(outcome.rollback.map(Rollback::slot), Some(1));
}

#[test]
fn a_step_whose_draft_was_accepted_whole_owes_no_rewind() {
    let drafts = toks(&[7, 8]);
    let outcome = StepOutcome::settle(
        TokenId::new(5),
        &drafts,
        &drafts,
        SPECIAL,
        Some(DraftSource::Head),
    )
    .expect("equal lengths");
    assert_eq!(outcome.committed, toks(&[5, 7, 8]));
    assert_eq!(outcome.rollback, None);
    assert_eq!(outcome.logits_row, 2);
    assert_eq!(outcome.failed_lookup(), None);
}

#[test]
fn a_step_that_drafted_nothing_commits_one_token_and_reads_row_zero() {
    // The plain decode path, expressed through the same type: K = 0 is not a
    // separate code path, it is a step whose draft list is empty.
    let outcome = StepOutcome::settle(TokenId::new(5), &[], &[], SPECIAL, None)
        .expect("no drafts is no comparison");
    assert_eq!(outcome.committed, toks(&[5]));
    assert_eq!(outcome.rollback, None);
    assert_eq!(outcome.logits_row, 0);
}

#[test]
fn a_lookup_that_was_only_partly_accepted_is_written_off() {
    let id = ProposalId::new(9, &toks(&[7, 8, 9]));
    let outcome = StepOutcome::settle(
        TokenId::new(5),
        &toks(&[7, 8, 9]),
        &toks(&[7, 8, 42]),
        SPECIAL,
        Some(DraftSource::Lookup(id)),
    )
    .expect("equal lengths");
    assert_eq!(
        outcome.failed_lookup(),
        Some(id),
        "a partial acceptance is a failure: the tokens past the divergence are \
         wrong and would be wrong the same way next time"
    );
}

#[test]
fn a_lookup_accepted_whole_is_not_written_off() {
    let drafts = toks(&[7, 8]);
    let id = ProposalId::new(3, &drafts);
    let outcome = StepOutcome::settle(
        TokenId::new(5),
        &drafts,
        &drafts,
        SPECIAL,
        Some(DraftSource::Lookup(id)),
    )
    .expect("equal lengths");
    assert_eq!(outcome.failed_lookup(), None);
}

// ---------------------------------------------------------------------------
// the failure ring
// ---------------------------------------------------------------------------

#[test]
fn the_ring_remembers_a_rejection_and_forgets_it_when_asked() {
    let mut ring = FailRing::new();
    let id = ProposalId::new(4, &toks(&[1, 2]));
    assert!(!ring.rejected(id));
    assert!(ring.is_empty());
    ring.remember(id);
    assert!(ring.rejected(id));
    assert_eq!(ring.len(), 1);
    ring.clear();
    assert!(
        !ring.rejected(id),
        "a new generation starts with no history"
    );
}

#[test]
fn the_same_place_with_a_different_draft_is_a_different_bet() {
    // The identity is the pair, not the position: the same continuation is
    // reachable through keys of several lengths, and a shorter draft from the
    // same place has not been tried.
    let mut ring = FailRing::new();
    ring.remember(ProposalId::new(4, &toks(&[1, 2, 3])));
    assert!(!ring.rejected(ProposalId::new(4, &toks(&[1, 2]))));
    assert!(!ring.rejected(ProposalId::new(5, &toks(&[1, 2, 3]))));
}

#[test]
fn the_ring_stays_bounded_and_keeps_the_newest_entries() {
    let mut ring = FailRing::new();
    for position in 0..600u32 {
        ring.remember(ProposalId::new(position, &toks(&[position])));
    }
    assert_eq!(
        ring.len(),
        256,
        "a ring, not a set that grows with the turn"
    );
    assert!(ring.rejected(ProposalId::new(599, &toks(&[599]))));
    assert!(
        !ring.rejected(ProposalId::new(0, &toks(&[0]))),
        "forgetting the oldest failure costs one more rejected draft, which is \
         what the bound buys"
    );
}

// ---------------------------------------------------------------------------
// the prompt lookup
// ---------------------------------------------------------------------------

#[test]
fn the_lookup_is_off_until_a_caller_asks_for_it() {
    let history = toks(&[1, 2, 3, 4, 1, 2, 3]);
    assert_eq!(
        propose(
            &history,
            TokenId::new(4),
            0,
            &FailRing::new(),
            NgramPolicy::off()
        ),
        None
    );
}

#[test]
fn a_repeated_phrase_proposes_what_followed_it_last_time() {
    // "1 2 3 4 5" earlier; the sequence now ends with "1 2 3" plus the token
    // being committed, so the continuation is "5".
    let history = toks(&[1, 2, 3, 4, 5, 9, 9, 1, 2, 3]);
    let proposal = propose(&history, TokenId::new(4), 0, &FailRing::new(), lookup(4))
        .expect("the key 1 2 3 4 occurs earlier");
    assert_eq!(proposal.tokens(), toks(&[5, 9, 9, 1]).as_slice());
    assert!(!proposal.is_empty());
}

#[test]
fn the_most_recent_occurrence_wins() {
    // The same six-token phrase three times: followed by 55 the first time, by
    // 77 the second, and running to the end of the sequence the third. Both
    // earlier occurrences match the longest key, so which one is read decides
    // the answer.
    let history = toks(&[8, 8, 1, 2, 3, 4, 55, 8, 8, 1, 2, 3, 4, 77, 8, 8, 1, 2, 3]);
    let proposal = propose(&history, TokenId::new(4), 0, &FailRing::new(), lookup(1))
        .expect("the key occurs twice");
    assert_eq!(
        proposal.tokens(),
        toks(&[77]).as_slice(),
        "the text being echoed is the one just said, not the first one like it"
    );
}

#[test]
fn a_proposal_the_ring_has_seen_fail_is_not_made_again() {
    let history = toks(&[1, 2, 3, 4, 5, 9, 9, 1, 2, 3]);
    let first = propose(&history, TokenId::new(4), 0, &FailRing::new(), lookup(1))
        .expect("the key occurs earlier");
    let mut ring = FailRing::new();
    ring.remember(first.id());
    // Only one occurrence, and its only continuation is the one just written
    // off, so every key length ends at the same refusal.
    assert_eq!(
        propose(&history, TokenId::new(4), 0, &ring, lookup(1)),
        None
    );
}

#[test]
fn a_continuation_that_runs_into_the_marker_range_is_dropped_whole() {
    // The earlier occurrence is followed by a turn marker. Truncating the
    // proposal just before it would be a bet that the model ends its turn in the
    // same place; that is a different claim from "this text is being echoed".
    let history = toks(&[1, 2, 3, 4, 1001, 7, 7, 1, 2, 3]);
    assert_eq!(
        propose(&history, TokenId::new(4), 0, &FailRing::new(), lookup(4)),
        None
    );
}

#[test]
fn a_key_reaching_back_into_the_prompt_is_refused() {
    // The prompt ends with a turn header, and a key that reaches into it matches
    // every earlier turn boundary in the context. The lookup stays quiet until
    // the generation is longer than the key.
    let history = toks(&[1, 2, 3, 4, 5, 9, 9, 1, 2, 3]);
    let generated_from = 9;
    assert_eq!(
        propose(
            &history,
            TokenId::new(4),
            generated_from,
            &FailRing::new(),
            lookup(4)
        ),
        None,
        "with only two generated tokens there is no three-token key inside them"
    );
    // The same history with the generation starting far enough back does fire.
    assert!(propose(&history, TokenId::new(4), 0, &FailRing::new(), lookup(4)).is_some());
}

#[test]
fn a_history_shorter_than_the_shortest_key_proposes_nothing() {
    let history = toks(&[1, 2]);
    assert_eq!(
        propose(&history, TokenId::new(3), 0, &FailRing::new(), lookup(4)),
        None
    );
}

#[test]
fn a_minimum_longer_than_the_longest_key_is_clamped_rather_than_silently_disabling() {
    // Obeying it would try no key at all and the lookup would be off while
    // reporting that it is on.
    let policy = NgramPolicy::new(4, 99, SPECIAL);
    assert_eq!(policy.min_match, NGRAM_MAX_MATCH);
    assert!(policy.is_on());
    assert_eq!(NgramPolicy::new(99, 0, SPECIAL).max_k, MAX_DRAFT);
    assert_eq!(NgramPolicy::new(4, 0, SPECIAL).min_match, 1);
}

// ---------------------------------------------------------------------------
// the gate between the two drafters
// ---------------------------------------------------------------------------

#[test]
fn the_lookup_is_taken_only_when_it_is_longer_and_extends_the_heads_whole_chain() {
    let history = toks(&[1, 2, 3, 4, 5, 9, 9, 1, 2, 3]);
    let proposal = propose(&history, TokenId::new(4), 0, &FailRing::new(), lookup(4))
        .expect("the key occurs earlier");
    let echoed = proposal.tokens().to_vec();
    assert!(
        lookup_beats_head(&proposal, &echoed[..2]),
        "four proposed tokens extending the head's two"
    );
    assert!(
        !lookup_beats_head(&proposal, &echoed),
        "no longer than the head's chain is the same verify width for a worse guess"
    );
    assert!(
        !lookup_beats_head(&proposal, &[TokenId::new(123)]),
        "without the agreement there is no evidence that this is the echoed text"
    );
    assert!(
        !lookup_beats_head(&proposal, &[echoed[0], TokenId::new(123)]),
        "agreeing on the first guess and not the second would replace a guess the \
         head usually gets right"
    );
    assert!(
        !lookup_beats_head(&proposal, &[]),
        "a step where the head drafted nothing has no evidence either"
    );
}

// ---------------------------------------------------------------------------
// the single-pass scan against the one-pass-per-key-length scan it replaced
// ---------------------------------------------------------------------------

/// The lookup as it was first written: every key length its own pass over the
/// history, longest first, newest window first. Slow at a long context and
/// obviously right, which is what makes it the reference for the scan that
/// replaced it.
fn propose_per_length(
    history: &[TokenId],
    first: TokenId,
    generated_from: u32,
    memory: &FailRing,
    policy: NgramPolicy,
) -> Option<Vec<TokenId>> {
    if !policy.is_on() {
        return None;
    }
    let mut seq = history.to_vec();
    seq.push(first);
    let seq_len = u32::try_from(seq.len()).expect("a test history fits in u32");
    let mut key = NGRAM_MAX_MATCH;
    while key >= policy.min_match {
        if key < seq_len && seq_len >= key + generated_from {
            for end in (key - 1..=seq_len - 2).rev() {
                let matches = (0..key)
                    .all(|back| seq[(end - back) as usize] == seq[(seq_len - 1 - back) as usize]);
                if !matches {
                    continue;
                }
                let mut tokens = Vec::new();
                let mut marker = false;
                for &token in &seq[end as usize + 1..] {
                    if tokens.len() == policy.max_k as usize {
                        break;
                    }
                    if token >= policy.special_floor {
                        marker = true;
                        break;
                    }
                    tokens.push(token);
                }
                if marker || tokens.is_empty() || memory.rejected(ProposalId::new(end, &tokens)) {
                    continue;
                }
                return Some(tokens);
            }
        }
        key -= 1;
    }
    None
}

#[test]
fn the_single_pass_proposes_exactly_what_a_pass_per_key_length_did() {
    // A tiny vocabulary so that keys of every length match somewhere, markers
    // among the text so that some continuations are refused, and failures
    // remembered so that the newest window is sometimes skipped for an older
    // one - the three ways the answer can move to a different window.
    let mut seed = 0x5eed_u64;
    let mut next = move |bound: u32| {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(seed >> 33).expect("a 31-bit value fits") % bound
    };
    let mut compared = 0_u32;
    let mut proposed = 0_u32;
    for case in 0..4000_u32 {
        let vocab = 2 + next(4);
        let len = next(80);
        let history = (0..len)
            .map(|_| {
                if next(40) == 0 {
                    SPECIAL
                } else {
                    TokenId::new(next(vocab))
                }
            })
            .collect::<Vec<_>>();
        let first = TokenId::new(next(vocab));
        let generated_from = next(len + 2);
        let policy = NgramPolicy::new(1 + next(MAX_DRAFT), 1 + next(NGRAM_MAX_MATCH), SPECIAL);
        let mut ring = FailRing::new();
        // Remember whatever the reference proposes a few times over, so later
        // rounds have to fall back past those windows.
        for _ in 0..3 {
            let expected = propose_per_length(&history, first, generated_from, &ring, policy);
            let got = propose(&history, first, generated_from, &ring, policy);
            assert_eq!(
                got.as_ref().map(|proposal| proposal.tokens().to_vec()),
                expected,
                "case {case}: history {history:?} first {first:?} from {generated_from} \
                 policy {policy:?}"
            );
            compared += 1;
            let Some(proposal) = got else { break };
            proposed += 1;
            ring.remember(proposal.id());
        }
    }
    assert!(
        proposed > compared / 4,
        "the cases have to exercise proposals, not only misses: {proposed} of {compared}"
    );
}

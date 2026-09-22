//! `lookup` is the only verb that moves page bytes, and the only one that can evict
//! while it is still deciding what to copy. Both are measured here: the number of
//! whole-page copies a resume costs, and the guarantee that the span a resume lands on
//! is never the one the victim scan drops to pay for it.

mod common;

use shiso_core::TokenId;
use shiso_kvstore::{CommitOpts, KvError, ReuseSource};

use common::{PAGE_BYTES, PAGE_TOKENS, blob, commit_span, open, prefill, resume, seq, spans};

/// A prompt whose first `shared` tokens are `start`'s run and whose tail is a different
/// run, so two prompts match for exactly `shared` positions and diverge after that.
fn prompt_with_branch(start: u32, shared: u32, tail: u32, tail_start: u32) -> Vec<TokenId> {
    let mut prompt = seq(start, shared);
    prompt.extend(seq(tail_start, tail));
    prompt
}

#[test]
fn a_prompt_nothing_has_seen_is_cold_and_claims_nothing() {
    let (cache, pages) = open(8, 4);
    let prompt = seq(1_000, 300);

    let plan = cache.plan(&prompt, 1.0);
    assert_eq!(plan.source, ReuseSource::Cold);
    assert_eq!(plan.reusable(), 0);
    assert!(plan.ckpt.is_none(), "there is no state to load");
    assert_eq!(cache.stats().pages_used, 0, "planning takes no page");
    assert_eq!(pages.copies(), 0, "planning moves no bytes");

    assert!(
        resume(&cache, &prompt, 1.0).is_none(),
        "a cold prompt has no checkpoint to move onto, which is the normal answer and not a failure"
    );
    assert_eq!(cache.stats().pages_used, 0);
}

#[test]
fn a_second_turn_of_the_same_conversation_recomputes_nothing() {
    let (cache, _pages) = open(8, 4);
    let prompt = seq(2_000, 300);
    let frontier = prefill(&cache, &prompt, 300, 1.0);

    // `plan` alone can never answer this: "the same conversation" is a fact about a
    // sequence, and only the standing frontier says which one.
    let plan = cache.plan_from(&frontier, &prompt, 2.0);
    assert_eq!(plan.matched, 300);
    assert_eq!(plan.source, ReuseSource::Live);
    assert_eq!(
        plan.reusable(),
        300,
        "the sequence is already standing where the prompt ends"
    );
}

#[test]
fn planning_reports_a_match_without_changing_the_trees_shape() {
    let (cache, _pages) = open(8, 4);
    let prompt = seq(3_000, 700);
    let writer = prefill(&cache, &prompt, 700, 1.0);
    let before = cache.stats();
    assert_eq!(spans(&cache), 1, "one turn wrote one span above the root");
    assert_eq!(before.pages_used, 3, "700 positions is three pages");

    let mut shorter = seq(3_000, 300);
    shorter.extend(seq(90_000, 40));
    let plan = cache.plan(&shorter, 2.0);
    assert_eq!(
        plan.matched, 300,
        "the match stops where the two prompts diverge, mid-span"
    );

    let after = cache.stats();
    assert_eq!(
        after.nodes, before.nodes,
        "a plan is a pure answer: the span a match stops inside stays one span until a \
         commit needs the boundary"
    );
    assert_eq!(after.pages_used, before.pages_used, "and it claims nothing");
    drop(writer);
}

#[test]
fn a_turn_that_stops_inside_a_span_splits_it_and_claims_nothing_for_the_shared_half() {
    let (cache, _pages) = open(8, 4);
    let prompt = seq(3_000, 700);
    let writer = prefill(&cache, &prompt, 700, 1.0);
    drop(writer);
    let shared_pages = cache.stats().pages_used;

    // A second conversation that shares the first 300 positions and then diverges. It
    // writes its own pages for the whole prompt, because it never resumed onto the
    // first conversation's path.
    let mut other = seq(3_000, 300);
    other.extend(seq(90_000, 40));
    let mut frontier = cache.cold_frontier();
    assert!(frontier.grow_to(340).expect("four pages are in the pool"));
    commit_span(&cache, &mut frontier, &other, 340, 2.0);

    assert_eq!(
        spans(&cache),
        3,
        "the span it stopped inside became two, and its own tail is the third"
    );
    assert_eq!(
        cache.stats().pages_used,
        shared_pages + 2,
        "the split moves page ids between halves; only the span above the branch point \
         costs a page"
    );
}

#[test]
fn resuming_inside_a_shared_page_copies_exactly_that_one_page() {
    let (cache, pages) = open(8, 4);
    let prompt = prompt_with_branch(4_000, 300, 400, 50_000);
    let writer = prefill(&cache, &prompt, 300, 1.0);
    let first = writer
        .table()
        .page_of(0)
        .expect("the writer's table covers position 0");
    let shared = writer
        .table()
        .page_of(299)
        .expect("the writer's table covers the last position before the checkpoint");
    let pattern: Vec<u8> = (0..u32::try_from(PAGE_BYTES).expect("fits"))
        .map(|i| u8::try_from(i % 251).expect("remainder fits"))
        .collect();
    pages.seed(shared, &pattern);
    // The conversation goes idle. The tree still lists these pages, so it still holds
    // its own reference on each of them -- without that reference the copy below would
    // be over a page the pool had already handed to somebody else.
    drop(writer);
    assert_eq!(cache.stats().pages_used, 2, "the tree holds what it lists");

    let resume = resume(&cache, &prompt, 2.0).expect("the checkpoint at 300 is reusable");

    assert_eq!(resume.ckpt_pos, 300);
    assert_eq!(resume.matched, 300);
    assert_eq!(resume.source, ReuseSource::Tree);
    assert_eq!(resume.frontier.pages(), 2, "positions [0, 300)");
    assert_eq!(
        pages.copies(),
        1,
        "the one page the resume point lands inside is the only page copied"
    );

    let private = resume
        .frontier
        .table()
        .page_of(299)
        .expect("the resumed table covers the same position");
    assert_ne!(
        shared, private,
        "a sequence resuming inside a shared page writes its next rows into a page of its own"
    );
    assert_eq!(
        pages
            .bytes_of(private)
            .expect("the private copy holds a page"),
        pattern,
        "the private copy is the shared page's bytes, row for row"
    );
    assert_eq!(
        resume.frontier.table().page_of(0),
        Some(first),
        "the pages entirely below the resume point stay shared: only the page the resume \
         point lands inside becomes private, and the writer's reference on it is untouched"
    );
}

#[test]
fn a_resume_that_cannot_pay_for_its_page_fails_without_losing_its_target() {
    // Two pages, which is exactly what the one committed span occupies, so the pool has
    // nothing to give the copy -- and the only span the victim scan could drop is the one
    // the resume is standing on.
    let (cache, _pages) = open(2, 4);
    let prompt = seq(13_000, 300);
    let writer = prefill(&cache, &prompt, 300, 1.0);
    drop(writer);
    assert_eq!(cache.stats().pages_used, 2, "the pool is at capacity");

    let error = cache
        .lookup(&prompt, 2.0)
        .expect_err("a resume that needs a page and cannot evict anything says so");

    assert!(
        matches!(error, KvError::PoolEmpty),
        "the caller is told the pool is out, rather than being handed a frontier over a \
         span the scan had just dropped: {error:?}"
    );
    assert_eq!(
        spans(&cache),
        1,
        "the victim scan ran with the target pinned, so it had nothing it was allowed to drop"
    );
    assert_eq!(
        cache.stats().ckpt_used,
        1,
        "the checkpoint the resume wanted is still in the store"
    );
    assert_eq!(
        cache.plan(&prompt, 3.0).source,
        ReuseSource::Tree,
        "a failed resume leaves the checkpoint where the next request can find it"
    );
}

#[test]
fn resuming_on_a_page_boundary_copies_nothing() {
    let (cache, pages) = open(8, 4);
    let prompt = prompt_with_branch(6_000, 2 * PAGE_TOKENS, 100, 70_000);
    let writer = prefill(&cache, &prompt, 512, 1.0);
    drop(writer);

    let resume = resume(&cache, &prompt, 2.0).expect("the checkpoint at 512 is reusable");

    assert_eq!(resume.ckpt_pos, 512);
    assert_eq!(resume.frontier.pages(), 2);
    assert_eq!(
        pages.copies(),
        0,
        "a checkpoint on a page boundary shares no page with the writer, so a resume at one is free"
    );
}

#[test]
fn a_resume_buys_its_page_from_eviction_without_dropping_its_own_target() {
    // Three pages, which is exactly what the one conversation below occupies: the pool
    // is full before the resume starts looking for the page it needs to copy into.
    let (cache, pages) = open(3, 4);
    let prompt = prompt_with_branch(8_000, 300, 400, 8_300);
    let mut writer = prefill(&cache, &prompt, 300, 1.0);
    let grew = writer
        .grow_to(700)
        .expect("the pool had the page the second span needs");
    assert!(grew);
    commit_span(&cache, &mut writer, &prompt, 700, 1.0);
    drop(writer);
    let before = cache.stats();
    assert_eq!(before.pages_used, 3, "the pool is exactly full");
    assert_eq!(before.ckpt_used, 2, "one checkpoint per committed boundary");

    // The second conversation wants the checkpoint at 300, which is not on a page
    // boundary, so the resume must find a page. The only evictable span is the one above
    // its own target; if the target were still an unpinned childless leaf when the victim
    // scan ran, this is where a resume would land on a node the cache had just dropped.
    let other = prompt_with_branch(8_000, 300, 400, 99_000);
    let resume = resume(&cache, &other, 2.0).expect("the checkpoint at 300 survives the shortage");

    assert_eq!(resume.ckpt_pos, 300);
    assert_eq!(
        pages.copies(),
        1,
        "the shortage was paid for with one page copy"
    );
    let after = cache.stats();
    assert_eq!(
        after.page_evictions, 1,
        "exactly the span above the target went"
    );
    assert_eq!(
        after.ckpt_used, 1,
        "the target's checkpoint is the one still in the store"
    );
    assert_eq!(
        cache.plan(&prompt, 3.0).matched,
        300,
        "the path the resume is standing on still matches as far as it did"
    );
}

#[test]
fn a_resume_onto_a_dropped_checkpoint_is_simply_not_offered() {
    let (cache, _pages) = open(8, 4);
    let prompt = seq(11_000, 300);
    let writer = prefill(&cache, &prompt, 300, 1.0);
    drop(writer);
    let freed = cache.evict_to(8, 1.0);
    assert_eq!(freed.nodes_dropped, 1, "the only span in the tree went");

    let plan = cache.plan(&prompt, 2.0);
    assert_eq!(plan.source, ReuseSource::Cold);
    assert!(resume(&cache, &prompt, 2.0).is_none());
    assert_eq!(
        cache.stats().ckpt_used,
        0,
        "the slot went with the span that owned it"
    );
}

#[test]
fn a_state_of_the_wrong_size_is_refused_before_anything_is_claimed() {
    let (cache, _pages) = open(8, 4);
    let prompt = seq(12_000, 300);
    let mut frontier = cache.cold_frontier();
    assert!(frontier.grow_to(300).expect("two pages are in the pool"));

    let short = blob()[..16].to_vec();
    let mut opts = CommitOpts::default();
    opts.state = Some(&short);
    let error = cache
        .commit(&mut frontier, &prompt, 300, opts, 1.0)
        .expect_err("a blob that is not a checkpoint blob is a caller bug");
    assert!(
        matches!(error, KvError::BufferLen { .. }),
        "the caller is told which of its two buffers was wrong, not that the pool is out"
    );
    assert_eq!(cache.stats().ckpt_used, 0);
    assert_eq!(spans(&cache), 0, "a refused commit leaves no orphan span");
}

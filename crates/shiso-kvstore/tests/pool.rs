//! Who holds a page, and when it goes back.
//!
//! The pool is the only part of the cache with a refcount, and the only part whose
//! mistakes are invisible until memory is gone. Every scenario here counts pages through
//! [`CacheStats::pages_used`](shiso_kvstore::CacheStats::pages_used), which is the pool's
//! own answer rather than a caller's bookkeeping.

mod common;

use shiso_kvstore::ReuseSource;

use common::{open, prefill, resume, seq};

#[test]
fn a_grow_claims_exactly_the_pages_the_positions_need() {
    let (cache, _pages) = open(8, 4);
    let mut frontier = cache.cold_frontier();
    assert_eq!(frontier.pages(), 0, "a cold frontier holds nothing");

    assert!(frontier.grow_to(700).expect("eight pages are in the pool"));

    assert_eq!(frontier.pages(), 3, "700 positions is three pages");
    assert_eq!(cache.stats().pages_used, 3);
    let table = frontier.table();
    assert_eq!(
        table.page_of(0),
        table.page_of(255),
        "one page covers a whole page's worth of positions"
    );
    assert_ne!(
        table.page_of(255),
        table.page_of(256),
        "and the next position is a different page"
    );
    assert_eq!(
        table.page_of(768),
        None,
        "the table covers whole pages, so the first position past the last one has no page \
         -- a position, not a zero"
    );
    assert!(
        table.page_of(767).is_some(),
        "and 767 is the last one it covers"
    );
}

#[test]
fn a_short_pool_refuses_a_grow_and_claims_nothing() {
    let (cache, _pages) = open(2, 2);
    let mut frontier = cache.cold_frontier();

    let grew = frontier
        .grow_to(700)
        .expect("a shortage is answered with false, not with an error");

    assert!(!grew, "three pages are needed and the pool has two");
    assert_eq!(
        frontier.pages(),
        0,
        "a failed grow leaves the table exactly as long as it was"
    );
    assert_eq!(
        cache.stats().pages_used,
        0,
        "and hands back whatever it had already taken"
    );

    assert!(
        frontier.grow_to(512).expect("two pages are in the pool"),
        "the same frontier can still be grown to what the pool can cover"
    );
    assert_eq!(cache.stats().pages_used, 2);
}

#[test]
fn a_second_holder_keeps_a_page_out_of_the_free_stack() {
    let (cache, _pages) = open(8, 4);
    let prompt = seq(14_000, 300);
    let writer = prefill(&cache, &prompt, 300, 1.0);
    assert_eq!(
        cache.stats().pages_used,
        2,
        "the tree's reference and the writer's are on the same two pages"
    );

    // The resume shares both pages below its checkpoint and takes one private copy of the
    // page the checkpoint sits inside, so it is a second holder of two pages and the sole
    // holder of one.
    let reader = resume(&cache, &prompt, 2.0)
        .expect("the checkpoint at 300 is reusable")
        .frontier;
    assert_eq!(cache.stats().pages_used, 3);

    drop(writer);
    assert_eq!(
        cache.stats().pages_used,
        3,
        "the tree and the reader still hold the shared pages, so neither is free"
    );

    drop(reader);
    assert_eq!(
        cache.stats().pages_used,
        2,
        "the private copy was the reader's alone and goes back with it"
    );

    let freed = cache.evict_to(8, 3.0);
    assert_eq!(
        freed.pages_freed, 2,
        "only the tree's own references were left"
    );
    assert_eq!(
        cache.stats().pages_used,
        0,
        "every page the cache handed out is back"
    );
}

#[test]
fn a_truncated_frontier_returns_the_pages_above_its_branch() {
    let (cache, _pages) = open(8, 4);
    let prompt = seq(15_000, 300);
    let writer = prefill(&cache, &prompt, 300, 1.0);
    let mut reader = resume(&cache, &prompt, 2.0)
        .expect("the checkpoint at 300 is reusable")
        .frontier;
    assert!(reader.grow_to(700).expect("the pool has the pages"));
    let used = cache.stats().pages_used;
    assert_eq!(used, 4, "two shared, one private, one above the branch");

    reader.truncate(2);

    assert_eq!(
        reader.pages(),
        2,
        "the frontier is back to the branch point"
    );
    assert_eq!(
        cache.stats().pages_used,
        used - 1,
        "the page above the branch was the frontier's alone and came back with it"
    );
    drop(writer);
}

#[test]
fn a_live_sequence_keeps_its_pages_out_of_the_eviction_pass() {
    let (cache, _pages) = open(8, 4);
    let prompt = seq(16_000, 700);
    let writer = prefill(&cache, &prompt, 700, 1.0);

    let freed = cache.evict_to(8, 2.0);

    assert_eq!(
        freed.pages_freed, 0,
        "the only span in the tree is standing under a live sequence"
    );
    assert_eq!(freed.nodes_dropped, 0);
    assert_eq!(cache.stats().pages_used, 3);
    assert_eq!(cache.plan(&prompt, 3.0).matched, 700);
    drop(writer);
}

#[test]
fn dropping_the_last_holder_lets_a_span_be_evicted_and_every_page_returned() {
    let (cache, _pages) = open(8, 4);
    let prompt = seq(17_000, 700);
    let writer = prefill(&cache, &prompt, 700, 1.0);
    drop(writer);

    let freed = cache.evict_to(8, 2.0);

    assert_eq!(freed.nodes_dropped, 1);
    assert_eq!(freed.pages_freed, 3);
    assert_eq!(cache.stats().pages_used, 0);
    assert_eq!(
        cache.stats().ckpt_used,
        0,
        "the checkpoint went with its span"
    );
    assert_eq!(cache.plan(&prompt, 3.0).source, ReuseSource::Cold);
}

//! The span tree, scenario for scenario.
//!
//! These are the shapes the cache exists to exploit: two 700-position conversations that
//! share their first 300, the page both halves of a split span own, and the order a
//! crowded pool gives up its spans in. The refcount-level assertions the same scenarios
//! need -- a straddled page held by both halves, a live count that stays with the guard
//! that minted it -- are in `src/tree.rs`, where the refcounts are visible; what is
//! asserted here is what a caller can see of them.

mod common;

use shiso_core::TokenId;
use shiso_kvstore::{Frontier, KvCache, ReuseSource};

use common::{commit_span, open, prefill, seq, spans};

/// Writes a conversation as two turns with a checkpoint at each boundary: 300 positions,
/// then the rest of the way to 700.
///
/// Two spans rather than one, because that is what a real conversation looks like -- one
/// chunk per boundary -- and because a tree with an interior node is the only tree where
/// "an interior span is not a victim while its child lives" means anything.
fn two_turns(cache: &KvCache, prompt: &[TokenId], now: f64) -> Frontier {
    let mut frontier = prefill(cache, prompt, 300, now);
    assert!(frontier.grow_to(700).expect("the pool was sized for this"));
    commit_span(cache, &mut frontier, prompt, 700, now);
    frontier
}

#[test]
fn two_paths_sharing_a_prefix_share_the_pages_below_the_branch_only() {
    let (cache, _pages) = open(8, 4);
    let first = seq(20_000, 300);
    let mut prompt_a = first.clone();
    prompt_a.extend(seq(21_000, 400));
    let mut prompt_b = first.clone();
    prompt_b.extend(seq(31_000, 400));

    let writer = two_turns(&cache, &prompt_a, 1.0);
    let head_page = writer
        .table()
        .page_of(0)
        .expect("the writer covers the first position");
    assert_eq!(spans(&cache), 2, "two boundaries, two spans");
    assert_eq!(cache.stats().pages_used, 3, "700 positions is three pages");

    // B resumes onto the checkpoint at 300, which is where sharing comes from: a cold
    // frontier would have written its own pages for the shared prefix and shared nothing.
    let reader = common::continue_prompt(&cache, &prompt_b, 700, 2.0);

    assert_eq!(
        spans(&cache),
        3,
        "the shared prefix is one span and each conversation owns its own tail"
    );
    assert_eq!(
        cache.stats().pages_used,
        5,
        "B's path is two pages of its own -- its tail, and the page it resumed inside -- \\
         on top of A's three"
    );
    assert_eq!(
        reader.table().page_of(0),
        Some(head_page),
        "the page below the branch point is the same page, not a copy of it"
    );
    assert_ne!(
        reader.table().page_of(299),
        writer.table().page_of(299),
        "the page the resume point sits inside is copied, because both conversations are \\
         about to write different rows of it"
    );
    assert_eq!(cache.plan_from(&reader, &prompt_b, 3.0).matched, 700);
    assert_eq!(cache.plan_from(&writer, &prompt_a, 3.0).matched, 700);
}

#[test]
fn an_interior_span_is_not_a_victim_while_a_child_lives() {
    let (cache, _pages) = open(8, 4);
    let mut prompt = seq(40_000, 300);
    prompt.extend(seq(41_000, 400));
    let writer = two_turns(&cache, &prompt, 1.0);
    drop(writer);
    let before = cache.stats();
    assert_eq!(before.pages_used, 3, "two spans, three pages");

    // Five of the eight pages are free and six are wanted, so one more page has to come
    // from somewhere. The only leaf is the tail span, and dropping it returns exactly the
    // one page no other span shares: the page it straddles with its parent stays held by
    // the parent, which is also why the parent survives a pass that met its target.
    let freed = cache.evict_to(6, 2.0);
    assert_eq!(freed.nodes_dropped, 1, "the leaf went");
    assert_eq!(
        freed.pages_freed, 1,
        "the straddled page belongs to the surviving half as much as to the dropped one"
    );

    assert_eq!(
        cache.plan(&prompt, 3.0).matched,
        300,
        "the interior span is still there with its checkpoint, and only the tail is gone"
    );
    assert_eq!(cache.plan(&prompt, 3.0).source, ReuseSource::Tree);
    assert_eq!(
        cache.stats().ckpt_used,
        1,
        "the tail's checkpoint went with it"
    );
}

#[test]
fn a_crowded_pool_gives_up_the_stalest_path_first() {
    let (cache, _pages) = open(8, 4);
    let prompt_a = seq(50_000, 700);
    let prompt_b = seq(60_000, 700);

    let writer_a = prefill(&cache, &prompt_a, 700, 1.0);
    drop(writer_a);
    let writer_b = prefill(&cache, &prompt_b, 700, 2.0);
    drop(writer_b);
    assert_eq!(spans(&cache), 2, "two unrelated conversations");
    assert_eq!(cache.stats().pages_used, 6, "three pages each");

    // Both spans are the same shape -- 700 positions over three pages, one checkpoint --
    // so the only thing that separates them is how recently each was read.
    let freed = cache.evict_to(5, 3.0);

    assert_eq!(
        freed.nodes_dropped, 1,
        "one span was enough to meet the target"
    );
    assert_eq!(
        cache.plan(&prompt_b, 4.0).matched,
        700,
        "the path read more recently is the one still standing"
    );
    assert_eq!(
        cache.plan(&prompt_a, 4.0).matched,
        0,
        "the older one paid for the pages"
    );
}

#[test]
fn a_dropped_path_returns_its_pages_and_leaves_no_span_behind() {
    let (cache, _pages) = open(8, 4);
    let mut prompt = seq(70_000, 300);
    prompt.extend(seq(71_000, 400));
    let writer = two_turns(&cache, &prompt, 1.0);
    drop(writer);

    let freed = cache.evict_to(8, 2.0);

    assert_eq!(
        freed.nodes_dropped, 2,
        "the leaf went first, then its parent"
    );
    assert_eq!(freed.pages_freed, 3, "every page the tree listed is back");
    assert_eq!(spans(&cache), 0);
    assert_eq!(cache.stats().pages_used, 0);
    assert_eq!(cache.stats().ckpt_used, 0);
    assert_eq!(cache.plan(&prompt, 3.0).source, ReuseSource::Cold);
}

//! The disk tier as a caller sees it: persist, pump, drain, and a resume in a
//! fresh cache over the same directory.
//!
//! Every scenario reopens the cache before it resumes, so what is asserted is
//! what the files say rather than what the tree remembers.

mod common;

use std::path::Path;
use std::time::Duration;

use shiso_core::TokenId;
use shiso_kvstore::{
    CommitOpts, DiskConfig, Frontier, KvCache, PathLayout, PersistOutcome, PersistRefusal, Reason,
    ReuseSource,
};

use common::{CKPT_BYTES, FakeBacking, PAGE_BYTES, Pages, blob, geometry, logits_row, seq};

/// The layout the scenarios' identity is built from. Any consistent set of
/// words will do; what matters is that two caches agree or deliberately differ.
fn layout() -> PathLayout {
    PathLayout {
        model_id: 9,
        quant_bits: 4,
        ctx_size: 65_536,
        kv_width: 16,
        n_layers: 4,
        n_attn_layers: 1,
        n_gdn_layers: 3,
        vocab: 16,
        gdn_conv_elements: 8,
        gdn_state_elements: 64,
        ple_window_elements: 0,
        hc_width: 0,
        indexer: false,
        draft_head: false,
    }
}

/// A disk tier with the floor lowered to fixture sizes.
fn disk(dir: &Path) -> DiskConfig {
    let mut config = DiskConfig::new(dir, 256 << 20);
    config.min_tokens = 16;
    config.staging_bytes = 64 * PAGE_BYTES;
    "fixture-kernels".clone_into(&mut config.numerics);
    config
}

/// A cache over a fresh fake device with a disk tier at `config`.
fn open(config: DiskConfig, pool_pages: u32) -> (KvCache, Pages) {
    let (backing, pages) = FakeBacking::new();
    let cache = KvCache::open(
        &geometry().with_layout(layout()),
        Box::new(backing),
        pool_pages,
        8,
        Some(config),
    )
    .expect("a fixture geometry with a layout opens a disk tier");
    (cache, pages)
}

/// The bytes a forward pass "wrote" into page `rank` of path `path`.
fn page_fill(path: u32, rank: usize) -> Vec<u8> {
    let seed = u8::try_from((usize::try_from(path).expect("fits") * 31 + rank) % 251)
        .expect("a remainder fits");
    vec![seed; usize::try_from(PAGE_BYTES).expect("a fixture page fits")]
}

/// Prefills `tokens` on a cold frontier, seeding every page it claims with
/// bytes `path` names, and commits a checkpoint at the end queued for disk.
fn prefill(cache: &KvCache, pages: &Pages, tokens: &[TokenId], path: u32) -> Frontier {
    let q = u32::try_from(tokens.len()).expect("fits");
    let mut frontier = cache.cold_frontier();
    assert!(frontier.grow_to(q).expect("the pool answers"));
    seed(pages, &frontier, path, 0);
    let state = blob();
    let row = logits_row();
    cache
        .commit(
            &mut frontier,
            tokens,
            q,
            CommitOpts::span(&[])
                .checkpoint(&state, true)
                .logits_row(&row)
                .persist(Reason::Cold),
            1.0,
        )
        .expect("a well-sized commit");
    frontier
}

/// Seeds every page of `frontier` from `from` on with bytes `path` names.
fn seed(pages: &Pages, frontier: &Frontier, path: u32, from: usize) {
    for (rank, page) in frontier.table().as_slice().iter().enumerate().skip(from) {
        pages.seed(*page, &page_fill(path, rank));
    }
}

/// Grows `frontier` to cover `tokens` and commits a checkpoint at the end,
/// queued for disk: the next turn of a conversation.
fn extend(cache: &KvCache, pages: &Pages, frontier: &mut Frontier, tokens: &[TokenId], path: u32) {
    let q = u32::try_from(tokens.len()).expect("fits");
    let before = frontier.pages();
    assert!(frontier.grow_to(q).expect("the pool answers"));
    seed(pages, frontier, path, before.saturating_sub(1));
    let state = blob();
    cache
        .commit(
            frontier,
            tokens,
            q,
            CommitOpts::span(&[])
                .checkpoint(&state, true)
                .persist(Reason::Continued),
            2.0,
        )
        .expect("a well-sized commit");
}

/// Everything queued reaches disk.
fn settle(cache: &KvCache) {
    assert!(cache.drain(Duration::from_secs(10)), "the writer caught up");
}

#[test]
fn a_stored_path_resumes_in_a_fresh_cache_with_the_bytes_it_was_written_with() {
    let dir = tempfile::tempdir().expect("a directory");
    let prompt = seq(1000, 700);
    {
        let (cache, pages) = open(disk(dir.path()), 16);
        let _frontier = prefill(&cache, &pages, &prompt, 1);
        settle(&cache);
        let stats = cache.disk_stats().expect("a tier is open");
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.pages_written, 3);
        assert_eq!(
            stats.entries_by_reason[usize::from(Reason::Cold.as_u8())],
            1
        );
    }
    let (cache, pages) = open(disk(dir.path()), 16);
    let resume = cache
        .resume(&prompt, 5.0)
        .expect("not an error")
        .expect("found on disk");
    assert_eq!(resume.source, ReuseSource::Disk);
    assert_eq!(resume.ckpt_pos, 700);
    for (rank, page) in resume.frontier.table().as_slice().iter().enumerate() {
        assert_eq!(
            pages.bytes_of(*page).expect("uploaded"),
            page_fill(1, rank),
            "page {rank} came back as written"
        );
    }
    let reference = resume.ckpt.expect("the state was re-admitted");
    let mut state = vec![0u8; usize::try_from(CKPT_BYTES).expect("fits")];
    cache.ckpt_state_into(reference, &mut state).expect("live");
    assert_eq!(state, blob());
    let mut row = vec![0u8; logits_row().len()];
    cache
        .ckpt_logits_into(reference, &mut row)
        .expect("carried");
    assert_eq!(row, logits_row());
    // A second request for the same prompt is served by the tree the restore
    // filled, not by the disk again.
    drop(resume);
    let again = cache.resume(&prompt, 6.0).expect("ok").expect("found");
    assert_eq!(again.source, ReuseSource::Tree);
}

#[test]
fn a_growing_conversation_writes_only_its_new_pages() {
    let dir = tempfile::tempdir().expect("a directory");
    let (cache, pages) = open(disk(dir.path()), 64);
    let full = seq(5000, 2048);
    let mut frontier = prefill(&cache, &pages, &full[..512], 7);
    settle(&cache);
    for end in [1024usize, 1536, 2048] {
        extend(&cache, &pages, &mut frontier, &full[..end], 7);
        settle(&cache);
    }
    let stats = cache.disk_stats().expect("open");
    assert_eq!(stats.entries, 4);
    // Eight pages cover 2048 tokens, and each was written exactly once.
    assert_eq!(
        stats.pages_written, 8,
        "no turn rewrote an earlier turn's pages"
    );
    assert_eq!(stats.pages, 8);
}

#[test]
fn a_prompt_that_continues_a_stored_one_resumes_at_its_end_without_logits() {
    let dir = tempfile::tempdir().expect("a directory");
    let full = seq(9000, 1200);
    {
        let (cache, pages) = open(disk(dir.path()), 32);
        let mut frontier = prefill(&cache, &pages, &full[..600], 2);
        // The turn's end carries no logits row: a later prompt always runs past it.
        extend(&cache, &pages, &mut frontier, &full[..900], 2);
        settle(&cache);
    }
    let (cache, _pages) = open(disk(dir.path()), 32);
    let resume = cache.resume(&full, 3.0).expect("ok").expect("found");
    assert_eq!(resume.ckpt_pos, 900, "the deepest entry on the path wins");
    // Exactly the turn's end as a prompt: that entry has no row, so the one
    // below it is used.
    drop(resume);
    let (cache, _pages) = open(disk(dir.path()), 32);
    let resume = cache.resume(&full[..900], 3.0).expect("ok").expect("found");
    assert_eq!(resume.ckpt_pos, 600);
}

#[test]
fn a_different_build_never_reads_the_files() {
    let dir = tempfile::tempdir().expect("a directory");
    let prompt = seq(100, 600);
    {
        let (cache, pages) = open(disk(dir.path()), 16);
        let _frontier = prefill(&cache, &pages, &prompt, 3);
        settle(&cache);
    }
    let mut other = disk(dir.path());
    "other-kernels".clone_into(&mut other.numerics);
    let (cache, _pages) = open(other, 16);
    assert!(cache.resume(&prompt, 1.0).expect("ok").is_none());
    let stats = cache.disk_stats().expect("open");
    assert_eq!(
        stats.entries, 0,
        "the other build's entry is not usable here"
    );
    assert!(
        stats.durable_bytes > 0,
        "but it still counts against the budget"
    );
}

#[test]
fn a_corrupt_page_is_a_miss_and_the_entry_is_discarded() {
    let dir = tempfile::tempdir().expect("a directory");
    let prompt = seq(200, 600);
    {
        let (cache, pages) = open(disk(dir.path()), 16);
        let _frontier = prefill(&cache, &pages, &prompt, 4);
        settle(&cache);
    }
    // Flip one byte near the end of every page file.
    for fan in std::fs::read_dir(dir.path().join("pages")).expect("pages") {
        for file in std::fs::read_dir(fan.expect("fan").path()).expect("files") {
            let path = file.expect("file").path();
            let mut bytes = std::fs::read(&path).expect("readable");
            let last = bytes.len() - 1;
            bytes[last] ^= 0xFF;
            std::fs::write(&path, bytes).expect("writable");
        }
    }
    let (cache, _pages) = open(disk(dir.path()), 16);
    assert!(
        cache
            .resume(&prompt, 1.0)
            .expect("a bad file is a miss")
            .is_none(),
        "a corrupt page must not be uploaded"
    );
    let stats = cache.disk_stats().expect("open");
    assert_eq!(stats.restore_failures, 1);
    assert_eq!(stats.entries, 0, "the entry was discarded");
}

#[test]
fn debris_from_an_interrupted_write_is_removed_at_open() {
    let dir = tempfile::tempdir().expect("a directory");
    let prompt = seq(300, 600);
    {
        let (cache, pages) = open(disk(dir.path()), 16);
        let _frontier = prefill(&cache, &pages, &prompt, 5);
        settle(&cache);
    }
    let fan = dir.path().join("pages").join("00");
    std::fs::create_dir_all(&fan).expect("a fan directory");
    let debris = fan.join("0000000000000000000000000000000000000000.pg.tmp.1.0");
    std::fs::write(&debris, b"half a page").expect("debris");
    let orphan = fan.join("0000000000000000000000000000000000000001.pg");
    std::fs::write(&orphan, b"a page nothing lists").expect("orphan");
    let (cache, _pages) = open(disk(dir.path()), 16);
    assert!(!debris.exists(), "the temporary file is gone");
    assert!(!orphan.exists(), "the orphan page is gone");
    assert!(
        cache.resume(&prompt, 1.0).expect("ok").is_some(),
        "the good entry survived"
    );
}

#[test]
fn a_path_below_the_floor_is_not_stored() {
    let dir = tempfile::tempdir().expect("a directory");
    let (cache, pages) = open(disk(dir.path()), 16);
    let frontier = prefill(&cache, &pages, &seq(0, 10), 6);
    assert_eq!(
        cache.persist(&frontier, Reason::Cold).expect("asked"),
        PersistOutcome::Refused(PersistRefusal::BelowMinTokens {
            tokens: 10,
            min: 16
        })
    );
    settle(&cache);
    assert_eq!(cache.disk_stats().expect("open").entries, 0);
}

#[test]
fn an_evicted_checkpoint_that_was_never_stored_is_written_on_its_way_out() {
    let dir = tempfile::tempdir().expect("a directory");
    let prompt = seq(4000, 700);
    let (cache, pages) = open(disk(dir.path()), 3);
    {
        // Committed with no persist request: only eviction will store it.
        let mut frontier = cache.cold_frontier();
        assert!(frontier.grow_to(700).expect("the pool answers"));
        seed(&pages, &frontier, 8, 0);
        let state = blob();
        let row = logits_row();
        cache
            .commit(
                &mut frontier,
                &prompt,
                700,
                CommitOpts::span(&[])
                    .checkpoint(&state, true)
                    .logits_row(&row),
                1.0,
            )
            .expect("a well-sized commit");
    }
    assert_eq!(cache.disk_stats().expect("open").queued_jobs, 0);
    let report = cache.evict_to(3, 2.0);
    assert!(report.pages_freed > 0, "the idle path was evicted");
    settle(&cache);
    let stats = cache.disk_stats().expect("open");
    assert_eq!(
        stats.entries_by_reason[usize::from(Reason::Evict.as_u8())],
        1
    );
    let resume = cache
        .resume(&prompt, 3.0)
        .expect("ok")
        .expect("found on disk");
    assert_eq!(resume.source, ReuseSource::Disk);
    assert_eq!(
        pages.bytes_of(resume.frontier.table().as_slice()[2]),
        Some(page_fill(8, 2))
    );
}

#[test]
fn the_tree_is_preferred_when_it_already_holds_the_checkpoint() {
    let dir = tempfile::tempdir().expect("a directory");
    let (cache, pages) = open(disk(dir.path()), 16);
    let prompt = seq(700, 600);
    drop(prefill(&cache, &pages, &prompt, 9));
    settle(&cache);
    let resume = cache.resume(&prompt, 2.0).expect("ok").expect("found");
    assert_eq!(resume.source, ReuseSource::Tree);
    assert_eq!(cache.disk_stats().expect("open").restores, 0);
}

#[test]
fn a_full_budget_releases_the_older_superseded_entry_before_a_lone_one() {
    let dir = tempfile::tempdir().expect("a directory");
    let mut config = disk(dir.path());
    // Eight page files of 5200 bytes and three entries of about 1.2 kB fill
    // 45 288 bytes; 500 fewer means one entry file has to go.
    config.budget_bytes = 45_288 - 500;
    let (cache, pages) = open(config, 64);
    let single = seq(20_000, 512);
    drop(prefill(&cache, &pages, &single, 10));
    settle(&cache);
    let long = seq(30_000, 1536);
    let mut frontier = prefill(&cache, &pages, &long[..512], 11);
    settle(&cache);
    extend(&cache, &pages, &mut frontier, &long[..1536], 11);
    settle(&cache);
    let stats = cache.disk_stats().expect("open");
    assert_eq!(stats.entries_evicted, 1, "the budget made room: {stats:?}");
    drop(frontier);
    let (cache, _pages) = open(disk(dir.path()), 64);
    assert!(
        cache.resume(&single, 1.0).expect("ok").is_some(),
        "the lone entry stayed"
    );
    // One token past the turn's end, because that entry carries no logits row.
    let mut next_turn = long.clone();
    next_turn.push(TokenId::new(1));
    assert_eq!(
        cache
            .resume(&next_turn, 1.0)
            .expect("ok")
            .map(|resume| resume.ckpt_pos),
        Some(1536),
        "the deeper entry on the long path stayed"
    );
    let (cache, _pages) = open(disk(dir.path()), 64);
    assert!(
        cache.resume(&long[..1000], 1.0).expect("ok").is_none(),
        "the entry the deeper one subsumed is the one that went"
    );
}

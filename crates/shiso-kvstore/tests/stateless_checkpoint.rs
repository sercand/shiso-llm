//! A checkpoint taken by a model with no recurrent state at all.
//!
//! Every scenario elsewhere in this directory carries a state blob, because the
//! families the cache was built for have a recurrent stack to snapshot. A purely
//! attentional family has none: its `ckpt_bytes` is zero and what a checkpoint
//! carries is the frontier logits row alone. The cache still has to admit that
//! checkpoint, hand it back on a resume, and round-trip it through the disk tier --
//! otherwise such a family re-prefills every prompt from position zero.

mod common;

use std::time::Duration;

use shiso_kvstore::{
    CacheGeometry, CommitOpts, DiskConfig, KvCache, PathLayout, PersistOutcome, Reason, ReuseSource,
};

use common::{FakeBacking, PAGE_BYTES, PAGE_TOKENS, Pages, seq};

/// The frontier logits row: four `f32`s, which is the `vocab` the layout below
/// declares. The bytes are arbitrary; what matters is that they come back.
const ROW: [u8; 16] = [1u8; 16];

/// Positions every scenario here prefills, two whole pages so the checkpoint sits
/// on a page start and a resume shares rather than copies.
const POSITIONS: u32 = 512;

/// The geometry of a stack with no recurrent state: pages as usual, and a
/// checkpoint blob of nothing.
#[must_use]
fn geometry() -> CacheGeometry {
    CacheGeometry::new(PAGE_TOKENS, PAGE_BYTES, 0, 0.004, 600.0)
}

/// The on-disk description that geometry implies: no recurrent layers, so the
/// payload's state section is empty and its `state_bytes` is zero.
#[must_use]
fn layout() -> PathLayout {
    PathLayout {
        model_id: 9,
        quant_bits: 4,
        ctx_size: 4096,
        kv_width: 16,
        n_layers: 2,
        n_attn_layers: 2,
        n_gdn_layers: 0,
        vocab: 4,
        gdn_conv_elements: 0,
        gdn_state_elements: 0,
        ple_window_elements: 0,
        hc_width: 0,
        indexer: false,
        draft_head: false,
    }
}

/// A cache over [`FakeBacking`] with the geometry a scenario asks for, plus the
/// handle it reads the moved bytes through.
#[must_use]
fn cache_over(
    geometry: &CacheGeometry,
    pool_pages: u32,
    ckpt_slots: u32,
    disk: Option<DiskConfig>,
) -> (KvCache, Pages) {
    let (backing, pages) = FakeBacking::new();
    let cache = KvCache::open(geometry, Box::new(backing), pool_pages, ckpt_slots, disk)
        .expect("a stateless geometry is one a cache can be built from");
    (cache, pages)
}

#[test]
fn a_geometry_with_no_recurrent_state_is_one_the_cache_can_be_built_from() {
    geometry()
        .validate()
        .expect("a model with no recurrent state has nothing to put in a state blob");
}

#[test]
fn a_stack_with_no_recurrent_state_commits_and_resumes_a_logits_only_checkpoint() {
    let (cache, _pages) = cache_over(&geometry(), 8, 2, None);
    let prompt = seq(21_000, POSITIONS);
    let mut writer = cache.cold_frontier();
    assert!(
        writer
            .grow_to(POSITIONS)
            .expect("a pool of eight pages covers two"),
        "the pool covered every position of the prompt"
    );

    cache
        .commit(
            &mut writer,
            &prompt,
            POSITIONS,
            CommitOpts::span(&[]).checkpoint(&[], true).logits_row(&ROW),
            1.0,
        )
        .expect("a checkpoint whose state is nothing is still a checkpoint");

    let resume = cache
        .lookup(&prompt, 2.0)
        .expect("a resume over a pool sized for the prompt does not fail")
        .expect("the prompt was just committed");
    assert_eq!(resume.matched, POSITIONS);
    assert_eq!(
        resume.ckpt_pos, POSITIONS,
        "the checkpoint is at the end of the span that was committed"
    );
    let reference = resume
        .ckpt
        .expect("the frontier row alone is what a stateless checkpoint carries");

    cache
        .ckpt_state_into(reference, &mut [])
        .expect("a blob of zero bytes copies into a buffer of zero bytes");
    let mut row = [0u8; 16];
    cache
        .ckpt_logits_into(reference, &mut row)
        .expect("the row was committed beside the empty blob");
    assert_eq!(row, ROW, "the row comes back exactly as it was handed over");
}

/// A disk tier small enough for a fixture and with the floor lowered to the
/// fixture's prompt.
fn disk(dir: &std::path::Path) -> DiskConfig {
    let mut config = DiskConfig::new(dir, 64 << 20);
    config.min_tokens = 8;
    config.staging_bytes = 1 << 20;
    config
}

#[test]
fn a_logits_only_checkpoint_survives_the_disk_tier() {
    let dir = tempfile::TempDir::new().expect("a temporary disk directory");
    let geometry = geometry().with_layout(layout());
    let prompt = seq(22_000, POSITIONS);
    {
        let (cache, pages) = cache_over(&geometry, 8, 2, Some(disk(dir.path())));
        let mut writer = cache.cold_frontier();
        assert!(
            writer
                .grow_to(POSITIONS)
                .expect("a pool of eight pages covers two"),
            "the pool covered every position of the prompt"
        );
        // The pump reads whole pages out of the backing, so the backing has to
        // hold what a forward pass would have written into them.
        let page_bytes = usize::try_from(PAGE_BYTES).expect("a fixture page fits a usize");
        for (rank, page) in writer.table().as_slice().iter().enumerate() {
            let fill = u8::try_from(rank % 251).expect("a remainder fits a byte");
            pages.seed(*page, &vec![fill; page_bytes]);
        }
        cache
            .commit(
                &mut writer,
                &prompt,
                POSITIONS,
                CommitOpts::span(&[])
                    .checkpoint(&[], true)
                    .logits_row(&ROW)
                    .persist(Reason::Cold),
                1.0,
            )
            .expect("a checkpoint whose state is nothing is still a checkpoint");
        assert!(matches!(
            cache.persist(&writer, Reason::Cold),
            Ok(PersistOutcome::AlreadyStored { .. })
        ));
        assert!(cache.drain(Duration::from_secs(10)), "the writer caught up");
    }

    // A fresh process: nothing in the tree, everything on disk.
    let (cache, pages) = cache_over(&geometry, 8, 2, Some(disk(dir.path())));
    let resume = cache
        .resume(&prompt, 4.0)
        .expect("a restore of a well-formed entry is not an error")
        .expect("the entry is in the directory");
    assert_eq!(resume.source, ReuseSource::Disk);
    assert_eq!(resume.ckpt_pos, POSITIONS);
    let reference = resume
        .ckpt
        .expect("a restored stateless checkpoint is re-admitted rather than dropped");
    cache
        .ckpt_state_into(reference, &mut [])
        .expect("a blob of zero bytes copies into a buffer of zero bytes");
    let mut row = [0u8; 16];
    cache
        .ckpt_logits_into(reference, &mut row)
        .expect("the row came back with the entry");
    assert_eq!(row, ROW);
    let first = resume.frontier.table().as_slice()[1];
    assert_eq!(
        pages.bytes_of(first).expect("the page was uploaded"),
        vec![1u8; usize::try_from(PAGE_BYTES).expect("fits")],
        "the second page came back holding the bytes it was written with"
    );
}

#[test]
fn the_stateless_checkpoint_worth_least_is_the_one_a_full_store_gives_up() {
    // Two slots, both filled, then a third earned checkpoint asks for one. Every
    // stateless checkpoint costs the same, so the victim has to be the one that
    // saves the least prefill -- the short path -- and not merely the first slot
    // the scan happened to look at, which is what an infinite utility would leave
    // it with. The long path is committed *first* so the two answers differ.
    let (cache, _pages) = cache_over(&geometry(), 8, 2, None);
    let long = seq(30_000, POSITIONS);
    let short = seq(31_000, POSITIONS / 2);
    let third = seq(32_000, POSITIONS);

    for (prompt, now) in [(&long, 1.0), (&short, 2.0), (&third, 3.0)] {
        let positions = u32::try_from(prompt.len()).expect("a fixture prompt is short");
        let mut frontier = cache.cold_frontier();
        assert!(
            frontier
                .grow_to(positions)
                .expect("a pool of eight pages covers five"),
            "the pool covered every position of the prompt"
        );
        cache
            .commit(
                &mut frontier,
                prompt,
                positions,
                CommitOpts::span(&[]).checkpoint(&[], true).logits_row(&ROW),
                now,
            )
            .expect("a checkpoint whose state is nothing is still a checkpoint");
    }

    assert!(
        cache.plan(&long, 4.0).ckpt.is_some(),
        "the path that saves twice the prefill keeps its checkpoint"
    );
    assert!(
        cache.plan(&short, 4.0).ckpt.is_none(),
        "the path that saves the least is the one the third admission displaced"
    );
    assert!(
        cache.plan(&third, 4.0).ckpt.is_some(),
        "and the admission that displaced it got its slot"
    );
}

#[test]
fn a_prompt_final_checkpoint_off_the_resume_grid_answers_only_its_own_prompt() {
    // Two checkpoints on one path: one on the grid at 512, and the end of a
    // prompt at 600, with its logits row. The prompt that ended at 600 gets
    // 600 back and needs no prefill; a longer prompt would cut a run there
    // that a cold prefill of it never cuts, so it gets 512.
    let geometry = geometry().with_resume_grid(64);
    let (cache, _pages) = cache_over(&geometry, 8, 4, None);
    let longer = seq(22_000, 700);
    let mut writer = cache.cold_frontier();
    assert!(
        writer.grow_to(600).expect("the pool covers three pages"),
        "the pool covered every position of the prompt"
    );
    cache
        .commit(
            &mut writer,
            &longer,
            POSITIONS,
            CommitOpts::span(&[]).checkpoint(&[], true),
            1.0,
        )
        .expect("the grid checkpoint commits");
    cache
        .commit(
            &mut writer,
            &longer,
            600,
            CommitOpts::span(&[]).checkpoint(&[], true).logits_row(&ROW),
            2.0,
        )
        .expect("the prompt-final checkpoint commits");

    let own = cache
        .lookup(&longer[..600], 3.0)
        .expect("not an error")
        .expect("the prompt that wrote it");
    assert_eq!(own.ckpt_pos, 600, "its own prompt resumes at its end");
    drop(own);
    let continued = cache
        .lookup(&longer, 4.0)
        .expect("not an error")
        .expect("the grid checkpoint beneath it");
    assert_eq!(
        continued.ckpt_pos, POSITIONS,
        "a longer prompt resumes on the grid"
    );
}

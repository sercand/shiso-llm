//! The byte geometry of this family's cache and checkpoints.
//!
//! Every number here is *derived* from the shape table rather than stated, and
//! that is exactly why it is worth a test: a page size that drifted from the
//! geometry would not fail - it would allocate a pool whose pages are the wrong
//! size and put one layer's key rows where another's belong.
//!
//! The figures are checked against the ones recorded for the shipping
//! checkpoint, so a shape-table edit that changes a page is a failing test here
//! rather than a silent cross-load with a checkpoint written by an older build.

use shiso_engine::shape::LayerKind;
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::paged::{
    PagePlanes, ckpt_bytes, draft_row_bytes, page_bytes, ple_window_bytes,
};
use shiso_models::qwen4exp::shape;

#[test]
fn a_page_carries_every_attention_layer_s_four_planes() {
    let shape = shape();
    assert_eq!(
        shape.count(LayerKind::Attn),
        12,
        "forty-eight slots at one in four"
    );

    // Twelve attention layers, a key run and a value run each, 256 positions of
    // a 512-wide row at two bytes an element.
    let kv_only = page_bytes(&shape, PagePlanes::default());
    assert_eq!(kv_only, 12 * 2 * 256 * 512 * 2);
    assert_eq!(kv_only, 6_291_456);

    // Plus the indexer's raw key row per position and its pooled row per block
    // of four, 128 wide at two bytes.
    let with_indexer = page_bytes(&shape, PagePlanes::indexer_only());
    assert_eq!(with_indexer - kv_only, 12 * (256 + 64) * 128 * 2);
    assert_eq!(with_indexer - kv_only, 983_040);

    // Plus a draft head's own four planes, which are one attention layer's
    // worth: it runs the same indexer over its own rows.
    let full = page_bytes(&shape, PagePlanes::full());
    assert_eq!(
        full - with_indexer,
        2 * 256 * 512 * 2 + (256 + 64) * 128 * 2
    );
    assert_eq!(full, 7_880_704);
}

#[test]
fn a_page_without_the_indexer_still_pays_for_a_draft_head_s_rows() {
    let shape = shape();
    let draft_only = page_bytes(
        &shape,
        PagePlanes {
            indexer: false,
            draft: true,
            ..PagePlanes::default()
        },
    );
    let kv_only = page_bytes(&shape, PagePlanes::default());
    // One extra attention layer's keys and values, and no indexer planes at all.
    assert_eq!(draft_only - kv_only, 2 * 256 * 512 * 2);
}

#[test]
fn a_checkpoint_blob_carries_what_the_cache_cannot_reconstruct() {
    let shape = shape();
    let gdn = u64::from(shape.count(LayerKind::Gdn));
    assert_eq!(gdn, 36);

    // Per recurrent slot: a convolution window of three rows of the projection
    // width, and a square state matrix per value head.
    let conv = 3 * u64::from(shape.gdn_in_dim()) * 4;
    let state = 48 * 128 * 128 * 4;
    // Plus the n-gram block's dilated window, and the pending draft row when the
    // stack speculates. The frontier logits row is not part of the blob: the
    // store commits it beside the state and checks the blob's length against
    // this number, so a blob that carried the row would be refused at every
    // save.
    let ngram = ple_window_bytes(&shape);
    let draft = u64::from(shape.stream_dim()) * 4;

    assert_eq!(
        ckpt_bytes(&shape, SpecDepth::NONE),
        gdn * (conv + state) + ngram
    );
    assert_eq!(
        ckpt_bytes(&shape, SpecDepth::new(4)),
        gdn * (conv + state) + ngram + draft,
        "a speculating stack's blob carries the row its next step drafts from"
    );
    assert_eq!(draft, draft_row_bytes(&shape));
    // Nine dilated taps of the whole residual: the block's convolution reaches
    // back three positions at a stride of three grams.
    assert_eq!(ngram, 9 * 10_240 * 4);
}

#[test]
fn a_resume_of_this_family_is_not_key_value_rows_alone() {
    // Stated as its own assertion because it is the one thing about this
    // family's checkpoint that differs in kind from the dense family's: a blob
    // that carried only the recurrent stack would resume a session whose n-gram
    // convolution window is somebody else's, and the block reads that window
    // before it writes it.
    let shape = shape();
    let recurrent_only = {
        let mut bare = shape.clone();
        bare.ple = None;
        ckpt_bytes(&bare, SpecDepth::NONE)
    };
    assert!(
        ckpt_bytes(&shape, SpecDepth::NONE) > recurrent_only,
        "the n-gram window is part of what a resume has to restore"
    );
    assert_eq!(
        ckpt_bytes(&shape, SpecDepth::NONE) - recurrent_only,
        ple_window_bytes(&shape)
    );
}

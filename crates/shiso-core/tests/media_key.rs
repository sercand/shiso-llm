//! The prefix-cache key a media prompt is filed under.
//!
//! What is pinned is the one property the key exists for: two different
//! pictures never share a key path, the same picture always does, and a key
//! that would file one prompt under another's path is refused when it is built
//! rather than discovered as a wrong answer.

use shiso_core::ids::TokenId;
use shiso_core::media::{KEY_ID_FLOOR, MediaDigest, MediaKey, MediaSpan, VisionGrid};

const PAD: TokenId = TokenId::new(248_056);

fn digest(seed: u8) -> MediaDigest {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    MediaDigest::new(bytes)
}

fn run(digest: MediaDigest, n: u32) -> Vec<TokenId> {
    (0..n).map(|k| digest.key_id(k)).collect()
}

#[test]
fn a_key_id_is_never_a_vocabulary_id_and_is_the_same_on_every_call() {
    let one = digest(1);
    for k in 0..64 {
        assert!(one.key_id(k).get() >= KEY_ID_FLOOR);
        assert_eq!(one.key_id(k), one.key_id(k));
    }
}

#[test]
fn two_pictures_part_at_their_first_placeholder_and_one_picture_never_does() {
    assert_ne!(digest(1).key_id(0), digest(2).key_id(0));
    assert_eq!(run(digest(7), 256), run(digest(7), 256));
    // Placeholders of one run are told apart, so agreeing on one is not
    // agreeing on the run.
    let ids = run(digest(3), 256);
    let distinct: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(distinct.len(), ids.len());
}

#[test]
fn a_key_that_rewrites_only_its_spans_is_accepted() {
    let text = TokenId::new(11);
    let mut path = vec![text, text];
    path.extend(run(digest(1), 4));
    path.push(text);
    let spans = [MediaSpan::new(2, VisionGrid::new(1, 2, 2))];
    let key = MediaKey::new(&path, &spans).expect("a well-formed key");
    assert_eq!(key.spans()[0].end(), 6);
    assert!(key.spans()[0].splits(3) && !key.spans()[0].splits(2) && !key.spans()[0].splits(6));
}

#[test]
fn a_bare_placeholder_inside_a_span_is_refused() {
    let path = [TokenId::new(11), PAD, PAD, PAD, PAD];
    let spans = [MediaSpan::new(1, VisionGrid::new(1, 2, 2))];
    MediaKey::new(&path, &spans).expect_err("an unkeyed run would match any picture");
}

#[test]
fn a_key_id_outside_every_span_is_refused() {
    let mut path = run(digest(1), 4);
    path.push(digest(1).key_id(9));
    let spans = [MediaSpan::new(0, VisionGrid::new(1, 2, 2))];
    MediaKey::new(&path, &spans).expect_err("a stray key id");
}

#[test]
fn a_span_past_the_path_or_out_of_order_is_refused() {
    let path = run(digest(1), 4);
    let past = [MediaSpan::new(2, VisionGrid::new(1, 2, 2))];
    MediaKey::new(&path, &past).expect_err("past the end");
    let mut two = run(digest(1), 4);
    two.extend(run(digest(2), 4));
    let backwards = [
        MediaSpan::new(4, VisionGrid::new(1, 2, 2)),
        MediaSpan::new(0, VisionGrid::new(1, 2, 2)),
    ];
    MediaKey::new(&two, &backwards).expect_err("out of order");
}

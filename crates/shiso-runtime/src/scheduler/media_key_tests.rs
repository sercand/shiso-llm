//! The key a media prompt is filed under, built from its prompt and items.
//!
//! The builder is the one place a placeholder run is matched to the picture it
//! stands for, so what is pinned is that it matches them one for one and that
//! every doubt turns into no key - which keeps the request off the tier - and
//! never into a key naming the wrong picture.

use shiso_core::{MediaDigest, MediaKey, PadTokens, TokenId, VisionGrid};

use super::media_key;

const PAD: TokenId = TokenId::new(248_056);
const TXT: TokenId = TokenId::new(9);

fn pads() -> PadTokens {
    PadTokens::new(Some(PAD), None)
}

fn digest(seed: u8) -> MediaDigest {
    let mut bytes = [0u8; 32];
    bytes[31] = seed;
    MediaDigest::new(bytes)
}

/// Text, a 2x2 still, text, a 1x3 still, text.
fn prompt() -> Vec<TokenId> {
    [
        vec![TXT; 2],
        vec![PAD; 4],
        vec![TXT],
        vec![PAD; 3],
        vec![TXT; 2],
    ]
    .concat()
}

fn grids() -> [VisionGrid; 2] {
    [VisionGrid::new(1, 2, 2), VisionGrid::new(1, 1, 3)]
}

#[test]
fn each_run_is_written_in_its_own_items_ids_and_the_text_is_left_alone() {
    let prompt = prompt();
    let (path, spans) =
        media_key(&prompt, &grids(), &[digest(1), digest(2)], pads()).expect("a key");
    assert_eq!(path.len(), prompt.len());
    assert_eq!(
        [
            spans[0].start(),
            spans[0].end(),
            spans[1].start(),
            spans[1].end()
        ],
        [2, 6, 7, 10]
    );
    assert_eq!(path[..2], prompt[..2]);
    assert_eq!(path[6], TXT);
    assert_eq!(path[10..], prompt[10..]);
    assert_eq!(path[2], digest(1).key_id(0));
    assert_eq!(path[7], digest(2).key_id(0));
    MediaKey::new(&path, &spans).expect("the builder's key is well-formed");
}

#[test]
fn a_different_second_picture_keeps_the_first_ones_path() {
    let prompt = prompt();
    let (one, _) = media_key(&prompt, &grids(), &[digest(1), digest(2)], pads()).expect("a key");
    let (two, _) = media_key(&prompt, &grids(), &[digest(1), digest(3)], pads()).expect("a key");
    let shared = one.iter().zip(&two).take_while(|(a, b)| a == b).count();
    assert_eq!(
        shared, 7,
        "the paths part at the second picture's first placeholder"
    );
}

#[test]
fn a_run_that_does_not_match_its_item_gives_no_key() {
    let prompt = prompt();
    let wrong = [VisionGrid::new(1, 2, 3), VisionGrid::new(1, 1, 3)];
    assert!(media_key(&prompt, &wrong, &[digest(1), digest(2)], pads()).is_none());
    let one = [VisionGrid::new(1, 2, 2)];
    assert!(
        media_key(&prompt, &one, &[digest(1)], pads()).is_none(),
        "a placeholder run left over after every item"
    );
    assert!(
        media_key(&prompt, &grids(), &[digest(1)], pads()).is_none(),
        "an item with no digest"
    );
}

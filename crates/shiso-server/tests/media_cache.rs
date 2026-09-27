//! A picture is decoded and encoded once, however many requests carry it.
//!
//! Two properties are pinned here. The cache keeps rows under its byte budget
//! and forgets the least recently used first. A request that was handed rows
//! keeps them however the cache moves after that. On top of those, the
//! renderer must stop decoding a payload once the cache knows its rows, and
//! must still hand the prefix cache the digest the decoded item had.

mod common;

use std::sync::Arc;

use base64::Engine as _;
use shiso_core::media::{MediaDigest, VisionGrid};
use shiso_core::{Backend, HfConfig};
use shiso_ir::{Block, MediaSource, Message, Role};
use shiso_server::media::MediaItem;
use shiso_server::media_cache::{CachedRows, MediaCache, payload_key};
use shiso_server::tokenizer::message_preprocess::{MediaPolicy, preprocess};
use shiso_vision::{ImageLimits, PreprocessSpec};

const RECORDED_CONFIG: &str = include_str!("../../../test-vectors/loader/qwen35-config.json");

fn backend() -> Arc<dyn Backend> {
    common::FakeBackend::new()
}

fn digest(seed: u8) -> MediaDigest {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    MediaDigest::new(bytes)
}

/// One item's rows of `bytes` bytes.
fn rows(backend: &Arc<dyn Backend>, bytes: u64) -> CachedRows {
    let buffer = shiso_core::alloc(backend, bytes).expect("the fake allocates");
    CachedRows::new(buffer, VisionGrid::new(1, 2, 2), false)
}

#[test]
fn rows_are_kept_under_the_budget_and_the_least_recently_used_go_first() {
    let backend = backend();
    let cache = MediaCache::new(300);
    cache.insert(digest(1), rows(&backend, 100));
    cache.insert(digest(2), rows(&backend, 100));
    cache.insert(digest(3), rows(&backend, 100));
    // Touch the oldest, so the next insert evicts the second instead.
    assert!(cache.rows(&digest(1)).is_some());
    cache.insert(digest(4), rows(&backend, 100));
    assert!(
        cache.rows(&digest(2)).is_none(),
        "the least recently used went"
    );
    for kept in [1, 3, 4] {
        assert!(cache.rows(&digest(kept)).is_some(), "{kept} stayed");
    }
    let stats = cache.stats();
    assert_eq!((stats.entries, stats.bytes), (3, 300));
}

#[test]
fn an_item_larger_than_the_budget_is_handed_back_and_not_kept() {
    let backend = backend();
    let cache = MediaCache::new(300);
    cache.insert(digest(1), rows(&backend, 100));
    let big = cache.insert(digest(2), rows(&backend, 400));
    assert_eq!(big.rows().len(), 400, "the caller still gets its rows");
    assert!(cache.rows(&digest(2)).is_none());
    assert!(
        cache.rows(&digest(1)).is_some(),
        "and nothing was evicted for it"
    );
}

#[test]
fn rows_a_request_holds_outlive_their_eviction() {
    let backend = backend();
    let cache = MediaCache::new(100);
    let held = cache.insert(digest(1), rows(&backend, 100));
    cache.insert(digest(2), rows(&backend, 100));
    assert!(cache.rows(&digest(1)).is_none(), "evicted from the cache");
    let mut bytes = vec![0u8; 100];
    backend
        .copy_d2h(held.rows().view(), &mut bytes)
        .expect("the request's rows are still allocated");
}

#[test]
fn a_payload_is_found_again_only_while_its_rows_are_resident() {
    let backend = backend();
    let cache = MediaCache::new(100);
    let key = payload_key("an image", "data:image/png;base64,AAAA");
    assert!(cache.rows_for_payload(&key).is_none(), "never seen");
    cache.remember_payload(key, digest(1));
    assert!(
        cache.rows_for_payload(&key).is_none(),
        "seen, but never encoded"
    );
    cache.insert(digest(1), rows(&backend, 100));
    let (found, _rows) = cache.rows_for_payload(&key).expect("seen and resident");
    assert_eq!(found, digest(1));
    assert_ne!(key, payload_key("a video", "data:image/png;base64,AAAA"));
}

fn png_data_uri(shade: u8) -> String {
    let image = image::RgbImage::from_pixel(64, 64, image::Rgb([shade, 90, 160]));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("a solid-colour PNG encodes");
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    )
}

fn spec() -> PreprocessSpec {
    let vision = HfConfig::from_json_str(RECORDED_CONFIG)
        .expect("the recorded config parses")
        .vision_config()
        .expect("the vision block parses")
        .expect("the recorded config carries a vision block");
    PreprocessSpec::from_vision_config(&vision).expect("the geometry drives a preprocessor")
}

fn image_turn(uri: &str) -> Message {
    Message::new(
        Role::User,
        vec![Block::Image(MediaSource::classify(uri.to_owned()))],
    )
}

#[test]
fn a_picture_whose_rows_are_resident_is_not_decoded_again() {
    let backend = backend();
    let spec = spec();
    let cache = MediaCache::new(1 << 30);
    let policy = MediaPolicy {
        vision: Some(&spec),
        limits: ImageLimits::default(),
        cache: Some(&cache),
        ..MediaPolicy::default()
    };
    let uri = png_data_uri(40);

    let first = preprocess(&[], &[image_turn(&uri)], &policy).expect("prepares");
    assert!(
        matches!(first.media[0], MediaItem::Vision(_)),
        "decoded the first time"
    );
    let digest = first.digests[0];
    assert_eq!(digest, first.media[0].digest());

    // Before the engine has encoded it, the payload is known but its rows are
    // not, so it is decoded again.
    let again = preprocess(&[], &[image_turn(&uri)], &policy).expect("prepares");
    assert!(matches!(again.media[0], MediaItem::Vision(_)));

    // What the engine thread does once the tower has run.
    let grid = first.media[0].grid(spec.geometry.spatial_merge_size());
    let bytes = u64::from(grid.pad_count()) * 16;
    let buffer = shiso_core::alloc(&backend, bytes).expect("allocates");
    cache.insert(digest, CachedRows::new(buffer, grid, false));

    let third = preprocess(&[], &[image_turn(&uri)], &policy).expect("prepares");
    let MediaItem::Encoded(item) = &third.media[0] else {
        panic!("a picture whose rows are resident was decoded again");
    };
    assert_eq!(item.digest, digest, "and it is filed under the same digest");
    assert_eq!(third.digests, first.digests);
    assert_eq!(third.pad_counts, first.pad_counts);

    // A different picture is still decoded.
    let other = preprocess(&[], &[image_turn(&png_data_uri(41))], &policy).expect("prepares");
    assert!(matches!(other.media[0], MediaItem::Vision(_)));
    assert_ne!(other.digests[0], digest);
}

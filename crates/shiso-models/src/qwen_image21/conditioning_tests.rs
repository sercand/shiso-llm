//! A job's conditioning on the host: the prefix layout an edit prompt
//! implies, the text rows the transformer reads, the cache key that keeps
//! an edit's features from answering a text job, and the budget refusal.

use shiso_vision::generator::{ImageGenError, InputReferenceImage};

use super::conditioning::{
    Conditioning, FeatureCache, FeatureKey, JobPrompt, edit_refusal, prepare_references,
    text_rows_of,
};
use super::prompt::{EditPromptIds, PromptIds};
use super::rope::Segment;
use super::vae::LatentGrid;
use crate::diffusion::residency::Shortfall;

/// An edit prompt of 40 ids: a 14-token system turn, 6 text ids, a 2x3
/// slot picture, 4 text ids, a 1x1 slot picture and 9 text ids.
fn edit_ids() -> EditPromptIds {
    EditPromptIds {
        ids: (0..40).collect(),
        drop: 14,
        images: vec![(20..26, (2, 3)), (30..31, (1, 1))],
    }
}

fn edit_prompt() -> JobPrompt {
    JobPrompt::Edit {
        ids: edit_ids(),
        references: [7; 32],
        grids: vec![LatentGrid { h: 4, w: 6 }, LatentGrid { h: 2, w: 2 }],
    }
}

#[test]
fn an_edit_prompt_lays_its_pictures_between_its_text_runs() {
    let prompt = edit_prompt();
    assert_eq!(
        prompt.layout(),
        vec![
            Segment::Text(6),
            Segment::Image(LatentGrid { h: 4, w: 6 }),
            Segment::Text(4),
            Segment::Image(LatentGrid { h: 2, w: 2 }),
            Segment::Text(9),
        ]
    );
    // Every slot is four latent tokens: 24 and 4.
    assert_eq!(prompt.text_rows(), 19);
    assert_eq!(prompt.prefix_tokens(), 19 + 24 + 4);
    assert_eq!(prompt.encoder_ids().len(), 40);

    let text = JobPrompt::Text(PromptIds {
        ids: (0..20).collect(),
        drop: 14,
    });
    assert_eq!(text.layout(), vec![Segment::Text(6)]);
    assert_eq!((text.text_rows(), text.prefix_tokens()), (6, 6));
}

#[test]
fn the_transformer_reads_the_text_rows_alone() {
    // Three kept rows of width 2, the middle one an image slot.
    let features = [1.0f32, 2.0, 9.0, 9.0, 3.0, 4.0];
    assert_eq!(
        text_rows_of(&features, &[false, true, false], 2).expect("rows"),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert!(text_rows_of(&features, &[false, true], 2).is_err());

    // An edit prompt's mask covers its kept rows.
    let prompt = edit_prompt();
    let kept = 40 - 14;
    let rows: Vec<f32> = (0..kept)
        .map(|r| f32::from(u8::try_from(r).expect("small")))
        .collect();
    let text = prompt.text_features(&rows, 1).expect("rows");
    assert_eq!(text.len(), 19);
    // Kept rows 6..12 and 16 are the pictures' slots.
    assert_eq!(&text[..7], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0]);
}

fn conditioning(tag: f32) -> Conditioning {
    Conditioning {
        text: vec![tag],
        images: Vec::new(),
    }
}

#[test]
fn text_job_after_edit_does_not_reuse_edit_features() {
    let kept = vec![1u32, 2, 3];
    let edit = FeatureKey {
        kept: kept.clone(),
        references: Some([7; 32]),
    };
    let text = FeatureKey {
        kept: kept.clone(),
        references: None,
    };
    let other_pictures = FeatureKey {
        kept,
        references: Some([8; 32]),
    };
    let mut cache = FeatureCache::new(4);
    cache.put(edit.clone(), conditioning(1.0));
    assert!(
        cache.get(&text).is_none(),
        "an edit's features answer no text job"
    );
    assert!(
        cache.get(&other_pictures).is_none(),
        "nor an edit of other pictures"
    );
    assert_eq!(cache.get(&edit).expect("the edit itself").text, vec![1.0]);

    // And the reverse.
    let mut cache = FeatureCache::new(4);
    cache.put(text.clone(), conditioning(2.0));
    assert!(
        cache.get(&edit).is_none(),
        "a text job's features answer no edit"
    );
    assert_eq!(cache.get(&text).expect("the text job").text, vec![2.0]);
}

#[test]
fn the_cache_keeps_the_most_recent_entries() {
    let key = |n: u32| FeatureKey {
        kept: vec![n],
        references: None,
    };
    let mut cache = FeatureCache::new(2);
    cache.put(key(1), conditioning(1.0));
    cache.put(key(2), conditioning(2.0));
    assert!(
        cache.get(&key(1)).is_some(),
        "reading moves an entry to the front"
    );
    cache.put(key(3), conditioning(3.0));
    assert!(cache.get(&key(2)).is_none(), "the least recent entry left");
    assert!(cache.get(&key(1)).is_some() && cache.get(&key(3)).is_some());
}

#[test]
fn over_budget_edit_names_bytes_and_count() {
    let error = edit_refusal(
        Shortfall {
            needed: 20 << 30,
            budget: 17 << 30,
        },
        3,
    );
    match error {
        ImageGenError::ReferenceImage(message) => {
            assert!(message.contains("3 reference images"), "{message}");
            assert!(message.contains("20480 MiB"), "{message}");
            assert!(message.contains("17408 MiB"), "{message}");
        }
        other => panic!("expected a reference-image refusal, got {other:?}"),
    }
}

#[test]
fn too_many_references_are_refused_before_any_is_decoded() {
    // Eleven byte strings that are not pictures: the count is judged first,
    // so the refusal names the count, not a decode failure.
    let garbage = vec![
        InputReferenceImage {
            bytes: b"not a picture".to_vec()
        };
        11
    ];
    match prepare_references(&garbage) {
        Err(ImageGenError::ReferenceImage(message)) => {
            assert!(message.contains("11 reference images"), "{message}");
        }
        other => panic!("expected the count refusal, got {other:?}"),
    }
    // Within the limit, the first unreadable one is named.
    match prepare_references(&garbage[..2]) {
        Err(ImageGenError::ReferenceImage(message)) => {
            assert!(message.contains("reference image 1"), "{message}");
        }
        other => panic!("expected a decode refusal, got {other:?}"),
    }
}

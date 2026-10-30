//! The speaker cache's host-side state: the FIFO's spill into the cache and
//! the compression, on hand-made activities.

#![expect(
    clippy::float_cmp,
    reason = "the embeddings hold their own row index, compared exactly"
)]

use super::{Advance, CACHE_LEN, State, compress};

const SILENCE: f32 = -1.0;

/// 190 cache rows of one-wide embeddings holding their own index: rows
/// 0..100 confidently speaker 0, rows 100..190 confidently speaker 1.
fn two_speakers() -> State {
    let n = 190;
    let mut preds = Vec::with_capacity(n * 2);
    for t in 0..n {
        preds.extend(if t < 100 { [0.9, 0.1] } else { [0.1, 0.9] });
    }
    State {
        cache: (0..n).map(|t| t as f32).collect(),
        cache_preds: preds,
        ..State::default()
    }
}

#[test]
fn compression_keeps_each_speakers_best_frames_grouped_with_silence_last() {
    let mut state = two_speakers();
    compress(&mut state, 190, 1, 2, &[SILENCE]);
    // A speaker's budget is 188 / 2 - 1 = 93: its first 69 frames are
    // boosted four- and two-fold, the next 70 two-fold. Speaker 1's frames
    // 188 and 189 are past the cache and boosted first. Of the 52 equal
    // two-fold frames left (speaker 0's 69..100, speaker 1's 167..188) the
    // 48 that fit are taken lowest flat index first, so speaker 1 loses
    // 184..188. Each speaker's block ends in its silence slot.
    let mut want: Vec<f32> = (0..100).map(|t| t as f32).collect();
    want.push(SILENCE);
    want.extend((100..184).map(|t| t as f32));
    want.extend([188.0, 189.0, SILENCE]);
    assert_eq!(state.cache.len(), CACHE_LEN);
    assert_eq!(state.cache, want);
    // A kept frame keeps its activities; a silence slot has none.
    assert_eq!(&state.cache_preds[..2], &[0.9, 0.1]);
    assert_eq!(&state.cache_preds[100 * 2..101 * 2], &[0.0, 0.0]);
    assert_eq!(&state.cache_preds[101 * 2..102 * 2], &[0.1, 0.9]);
    assert_eq!(&state.cache_preds[187 * 2..], &[0.0, 0.0]);
}

#[test]
fn a_speaker_with_no_frames_keeps_only_its_silence_slot() {
    // Every frame is speaker 0's: speaker 1 has nothing to keep but its
    // silence slot, and speaker 0's frames fill every other one.
    let n = 190;
    let mut state = State {
        cache: (0..n).map(|t| t as f32).collect(),
        cache_preds: (0..n).flat_map(|_| [0.9, 0.1]).collect(),
        ..State::default()
    };
    compress(&mut state, n, 1, 2, &[SILENCE]);
    let kept: Vec<f32> = state.cache.iter().copied().filter(|v| *v >= 0.0).collect();
    // Speaker 0's 190 frames compete for the 186 slots the two silence
    // slots leave: the newest two are boosted first, and of the equal rest
    // the latest four lose.
    let mut want: Vec<f32> = (0..184).map(|t| t as f32).collect();
    want.extend([188.0, 189.0]);
    assert_eq!(kept, want);
    assert_eq!(state.cache.iter().filter(|v| **v == SILENCE).count(), 2);
}

/// `rows` rows of activities `[r, r + 0.5]` offset by `base`.
fn p80(rows: usize, base: f32) -> Vec<f32> {
    (0..rows)
        .flat_map(|r| [base + r as f32, base + r as f32 + 0.5])
        .collect()
}

#[test]
fn the_fifo_spills_into_the_cache_past_its_length() {
    let silence = [SILENCE];
    let mut state = State::default();
    // Window 1: 30 core rows after 2 of left context join the FIFO, under its
    // 40, so nothing spills.
    let emb: Vec<f32> = (0..40).map(|r| 100.0 + r as f32).collect();
    state.advance(&Advance {
        p80: &p80(40, 0.0),
        emb: &emb,
        left_drop: 2,
        core: 30,
        d: 1,
        speakers: 2,
        silence: &silence,
    });
    assert!(state.cache.is_empty());
    assert_eq!(state.fifo, (102..132).map(|r| r as f32).collect::<Vec<_>>());
    assert_eq!(state.fifo_preds, p80(32, 0.0)[2 * 2..]);
    // Window 2 runs behind the 30 FIFO rows: their activities are rescored
    // by it, and 30 + 20 core rows overflow the FIFO, which spills whole
    // (the period of 300 is more than it holds).
    let emb: Vec<f32> = (0..25).map(|r| 200.0 + r as f32).collect();
    state.advance(&Advance {
        p80: &p80(30 + 25, 1000.0),
        emb: &emb,
        left_drop: 1,
        core: 20,
        d: 1,
        speakers: 2,
        silence: &silence,
    });
    assert!(state.fifo.is_empty() && state.fifo_preds.is_empty());
    let mut want: Vec<f32> = (102..132).map(|r| r as f32).collect();
    want.extend((201..221).map(|r| r as f32));
    assert_eq!(state.cache, want);
    let scored = p80(55, 1000.0);
    let mut want_preds = scored[..30 * 2].to_vec();
    want_preds.extend_from_slice(&scored[31 * 2..51 * 2]);
    assert_eq!(state.cache_preds, want_preds);
    // Window 3 runs behind the 50 cache rows: before any compression their
    // activities are the ones it scored.
    let emb: Vec<f32> = (0..50).map(|r| 300.0 + r as f32).collect();
    state.advance(&Advance {
        p80: &p80(50 + 50, 2000.0),
        emb: &emb,
        left_drop: 0,
        core: 50,
        d: 1,
        speakers: 2,
        silence: &silence,
    });
    assert_eq!(state.cache.len(), 100);
    assert_eq!(state.cache_preds, p80(100, 2000.0));
    assert!(!state.compressed);
}

#[test]
fn a_compressed_cache_keeps_its_own_activities() {
    let silence = [SILENCE];
    let mut state = two_speakers();
    compress(&mut state, 190, 1, 2, &[SILENCE]);
    state.compressed = true;
    // This window scores every cache row and its own core as silence; were
    // the cache's activities refreshed from it, the next compression would
    // keep nothing but silence slots.
    let emb = vec![500.0f32; 50];
    state.advance(&Advance {
        p80: &vec![0.0; (CACHE_LEN + 50) * 2],
        emb: &emb,
        left_drop: 0,
        core: 50,
        d: 1,
        speakers: 2,
        silence: &silence,
    });
    assert!(state.fifo.is_empty());
    assert_eq!(state.cache.len(), CACHE_LEN);
    assert_eq!(state.cache[0], 0.0, "speaker 0's first frame survives");
    assert_eq!(state.cache_preds[..2], [0.9, 0.1]);
    assert!(
        !state.cache.contains(&500.0),
        "the silent core rows are not kept"
    );
}

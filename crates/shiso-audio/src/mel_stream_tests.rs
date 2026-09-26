//! A stream fed in awkward pieces reproduces the offline frames exactly.

use std::sync::Arc;

use super::MelStream;
use crate::mel::tests::{centred, left_padded};
use crate::mel::{LogScale, MelFrontend};
use crate::num::{f32_of, f64_of};

#[test]
fn awkward_push_schedules_match_offline_bit_for_bit() {
    let fe = Arc::new(MelFrontend::new(centred()).expect("valid"));
    let n = 3 * 16_000 + 123;
    let signal: Vec<f32> = (0..n)
        .map(|i| f32_of((f64_of(i) * 0.013).sin() * 0.3 + (f64_of(i) * 0.0007).cos() * 0.1))
        .collect();
    let offline = fe.compute(&signal);
    for schedule in [&[1usize][..], &[159, 161], &[4_000], &[7, 512, 33, 160]] {
        let mut stream = MelStream::new(Arc::clone(&fe)).expect("centred");
        let mut rows: Vec<f32> = Vec::new();
        let (mut at, mut i) = (0, 0);
        while at < n {
            let take = schedule[i % schedule.len()].min(n - at);
            rows.extend_from_slice(stream.push(&signal[at..at + take]).as_slice());
            at += take;
            i += 1;
        }
        rows.extend_from_slice(stream.finish().as_slice());
        assert_eq!(
            rows.len(),
            offline.valid() * offline.n_mels(),
            "schedule {schedule:?}"
        );
        assert_eq!(
            &rows[..],
            &offline.as_slice()[..rows.len()],
            "schedule {schedule:?}"
        );
    }
}

#[test]
fn refuses_left_padded_framing() {
    let fe = Arc::new(MelFrontend::new(left_padded()).expect("valid"));
    assert!(MelStream::new(fe).is_err());
}

#[test]
fn refuses_the_whole_clip_log_scale() {
    // Centred framing streams fine on its own; it is the whisper-clamp log
    // scale, which needs every frame's value before it can floor any of
    // them, that a stream cannot offer.
    let mut spec = centred();
    spec.log_scale = LogScale::WhisperClamp;
    let fe = Arc::new(MelFrontend::new(spec).expect("valid"));
    assert!(MelStream::new(fe).is_err());
}

#[test]
fn a_short_stream_emits_nothing_and_finishes_cleanly() {
    let fe = Arc::new(MelFrontend::new(centred()).expect("valid"));
    let mut stream = MelStream::new(fe).expect("centred");
    assert_eq!(stream.push(&[0.1; 50]).total(), 0);
    assert_eq!(stream.finish().total(), 0);
}

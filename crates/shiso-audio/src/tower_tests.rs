//! Building a tower's clip: the frontend, the valid rows, the token count and
//! the tower's limits.

use std::sync::Arc;

use shiso_core::Backend;
use shiso_vision::EncodedRows;

use super::{AudioClip, AudioTower, clip_for};
use crate::error::AudioError;
use crate::mel::{Filterbank, Framing, MelScale, MelSpec, Preemphasis, Spectrum, Window};
use crate::pcm::{MODEL_RATE, Pcm};

/// A tower with a real frontend and the token rule of two stride-2 stages,
/// capped at `cap` tokens.
#[derive(Debug)]
struct Fake {
    spec: MelSpec,
    cap: usize,
}

impl AudioTower for Fake {
    fn mel_spec(&self) -> &MelSpec {
        &self.spec
    }
    fn tokens_for(&self, valid_mel: usize) -> Option<usize> {
        let tokens = valid_mel.div_ceil(2).div_ceil(2);
        (tokens <= self.cap).then_some(tokens)
    }
    fn max_seconds(&self) -> f64 {
        0.4
    }
    fn out_hidden(&self) -> usize {
        8
    }
    fn device_bytes(&self) -> u64 {
        0
    }
    fn encode(&self, _: &Arc<dyn Backend>, _: &[AudioClip]) -> Result<EncodedRows, AudioError> {
        Err(AudioError::Empty)
    }
}

fn fake(cap: usize) -> Fake {
    Fake {
        spec: MelSpec {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 320,
            hop_length: 160,
            n_mels: 128,
            window: Window::HannPeriodic,
            framing: Framing::LeftPadded {
                pad_to_multiple: 128,
            },
            preemphasis: Preemphasis::None,
            spectrum: Spectrum::Magnitude,
            filterbank: Filterbank::Computed {
                scale: MelScale::Htk,
                slaney_norm: false,
                f_min: 0.0,
                f_max: 8_000.0,
            },
            log_guard: 1e-3,
        },
        cap,
    }
}

fn tone(samples: usize) -> Pcm {
    Pcm::new(
        (0..samples)
            .map(|i| if i % 40 < 20 { 0.1 } else { -0.1 })
            .collect(),
        MODEL_RATE,
    )
    .expect("a rate")
}

#[test]
fn a_clip_keeps_the_valid_rows_and_their_token_count() {
    let clip = clip_for(&fake(100), &tone(6_400)).expect("builds");
    // 6400 samples -> 39 valid rows under the left-padded framing -> 10 tokens.
    assert_eq!(clip.mel().total(), clip.mel().valid());
    assert_eq!(clip.mel().valid(), 39);
    assert_eq!(clip.tokens(), 10);
}

#[test]
fn too_short_and_too_long_are_refused() {
    assert!(matches!(
        clip_for(&fake(100), &tone(100)),
        Err(AudioError::Empty)
    ));
    assert!(matches!(
        clip_for(&fake(100), &tone(16_000)),
        Err(AudioError::TooLong { .. })
    ));
    // Within the seconds but past the token cap.
    assert!(matches!(
        clip_for(&fake(5), &tone(6_400)),
        Err(AudioError::TooLong { .. })
    ));
}

#[test]
fn a_clip_cannot_claim_more_tokens_than_rows_allow() {
    let clip = clip_for(&fake(100), &tone(6_400)).expect("builds");
    assert!(AudioClip::new(clip.mel().clone(), 0).is_err());
}

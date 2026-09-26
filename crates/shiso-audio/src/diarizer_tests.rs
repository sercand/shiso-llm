//! The diarizer contract is usable as a trait object, and answers one clip
//! with one activity block of the clip's own length.

use super::{DiarizationPass, Diarizer};
use crate::error::AudioError;
use crate::mel::{
    Filterbank, Framing, MelFrames, MelScale, MelSpec, Preemphasis, Spectrum, Window,
};
use crate::transcript::SpeakerActivity;

/// Every frame belongs to speaker 0.
#[derive(Debug)]
struct Monologue {
    spec: MelSpec,
}

impl Diarizer for Monologue {
    fn mel_spec(&self) -> &MelSpec {
        &self.spec
    }
    fn speakers(&self) -> usize {
        2
    }
    fn frame_secs(&self) -> f64 {
        0.01
    }
    fn begin(&self, mel: MelFrames) -> Result<Box<dyn DiarizationPass>, AudioError> {
        Ok(Box::new(Steps {
            left: 3,
            frames: mel.valid(),
        }))
    }
}

/// Three steps, then the activity.
struct Steps {
    left: usize,
    frames: usize,
}

impl DiarizationPass for Steps {
    // `None`: the pass has more steps to run; it is not a refusal.
    fn step(&mut self) -> Result<Option<SpeakerActivity>, AudioError> {
        if self.left > 0 {
            self.left -= 1;
            return Ok(None);
        }
        let probs = (0..self.frames).flat_map(|_| [1.0, 0.0]).collect();
        SpeakerActivity::new(0.01, 2, probs).map(Some)
    }
}

fn spec() -> MelSpec {
    MelSpec {
        sample_rate: 16_000,
        n_fft: 512,
        win_length: 400,
        hop_length: 160,
        n_mels: 4,
        window: Window::HannSymmetric,
        framing: Framing::Centred,
        preemphasis: Preemphasis::Signal(0.97),
        spectrum: Spectrum::Power,
        filterbank: Filterbank::Computed {
            scale: MelScale::Slaney,
            slaney_norm: true,
            f_min: 0.0,
            f_max: 8_000.0,
        },
        log_guard: 2f32.powi(-24),
    }
}

#[test]
fn one_clip_is_one_activity_block_of_its_own_length() {
    let diarizer: &dyn Diarizer = &Monologue { spec: spec() };
    let mel = MelFrames::from_rows(vec![0.0; 4 * 30], 4).expect("whole rows");
    let activity = diarizer.diarize(&mel).expect("diarizes");
    assert_eq!(activity.frames(), 30);
    assert_eq!(activity.speakers(), diarizer.speakers());
    assert!((activity.frame_secs() - diarizer.frame_secs()).abs() < f64::EPSILON);
    assert!((activity.prob(29, 0) - 1.0).abs() < f32::EPSILON);
}

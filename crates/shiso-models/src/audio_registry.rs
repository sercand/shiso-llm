//! Opens a speech checkpoint as whichever family claims it.
//!
//! The same shape as the language-model factory: an ordered list of openers,
//! each of which either recognises the path or says it is not its own. Adding
//! a family adds a line here and a directory beside this file, nothing else.

use std::path::Path;
use std::sync::Arc;

use shiso_audio::{AudioError, Diarizer, SpeechRecognizer};
use shiso_core::Backend;

/// Opens the speech recognizer at `path` on `backend`.
///
/// `mmproj` is `--asr-mmproj`: the audio-tower file for a family whose GGUF
/// export splits it from the text model. A family that ships in one file
/// ignores it.
///
/// # Errors
///
/// The claiming family's load error, or [`AudioError::Checkpoint`] saying no
/// family recognises the path.
pub fn open_speech_recognizer(
    path: &Path,
    mmproj: Option<&Path>,
    backend: &Arc<dyn Backend>,
) -> Result<Arc<dyn SpeechRecognizer>, AudioError> {
    // Nemotron ships every export as one file or directory; it has no tower
    // file to take.
    let _ = mmproj;
    if let Some(model) = crate::nemotron_asr::open(path, backend)? {
        return Ok(Arc::new(model));
    }
    Err(AudioError::Checkpoint {
        what: format!("no speech recognizer family recognises {}", path.display()),
    })
}

/// Opens the speaker diarizer at `path` on `backend`.
///
/// # Errors
///
/// The claiming family's load error, or [`AudioError::Checkpoint`] saying no
/// family recognises the path.
pub fn open_diarizer(
    path: &Path,
    backend: &Arc<dyn Backend>,
) -> Result<Arc<dyn Diarizer>, AudioError> {
    if let Some(model) = crate::nemotron_diar::open(path, backend)? {
        return Ok(Arc::new(model));
    }
    Err(AudioError::Checkpoint {
        what: format!("no diarizer family recognises {}", path.display()),
    })
}

#[cfg(test)]
mod tests {

    use shiso_audio::AudioError;

    #[test]
    #[ignore = "opens a Metal backend"]
    fn a_directory_no_family_claims_is_refused_saying_so() {
        let Some(backend) = crate::audio_ops::testing::device() else {
            return;
        };
        let dir = tempfile::tempdir().expect("a scratch directory");
        std::fs::write(dir.path().join("config.json"), r#"{"model_type": "other"}"#)
            .expect("writes");
        let error = super::open_diarizer(dir.path(), &backend).expect_err("nothing claims it");
        assert!(
            matches!(&error, AudioError::Checkpoint { what } if what.contains("no diarizer family")),
            "{error}"
        );
    }
}

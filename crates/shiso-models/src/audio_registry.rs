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
/// refuses it rather than ignoring it, since a flag that does nothing hides
/// a wrong command line.
///
/// # Errors
///
/// The claiming family's load error, [`AudioError::Checkpoint`] naming
/// `--asr-mmproj` when the claiming family takes none, or
/// [`AudioError::Checkpoint`] saying no family recognises the path.
pub fn open_speech_recognizer(
    path: &Path,
    mmproj: Option<&Path>,
    backend: &Arc<dyn Backend>,
) -> Result<Arc<dyn SpeechRecognizer>, AudioError> {
    // Nemotron ships every export as one file or directory; it has no tower
    // file to take. Recognising the path loads nothing, so the refusal comes
    // before the weights do.
    if crate::nemotron_asr::config::locate(path)?.is_some() {
        if let Some(given) = mmproj {
            return Err(AudioError::Checkpoint {
                what: format!(
                    "--asr-mmproj names {}, but {} carries its own encoder and takes no mmproj",
                    given.display(),
                    path.display()
                ),
            });
        }
        if let Some(model) = crate::nemotron_asr::open(path, backend)? {
            return Ok(Arc::new(model));
        }
    }
    if let Some(model) = crate::qwen3_asr::open(path, mmproj, backend)? {
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

    #[test]
    #[ignore = "opens a Metal backend"]
    fn an_mmproj_for_a_family_that_takes_none_is_refused_naming_the_flag() {
        let Some(backend) = crate::audio_ops::testing::device() else {
            return;
        };
        // Enough of a Nemotron export to be claimed; the refusal comes before
        // any weight is read.
        let dir = tempfile::tempdir().expect("a scratch directory");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "nemotron3_5_asr"}"#,
        )
        .expect("writes");
        std::fs::write(dir.path().join("processor_config.json"), "{}").expect("writes");
        let mmproj = dir.path().join("mmproj-x.gguf");
        let error = super::open_speech_recognizer(dir.path(), Some(&mmproj), &backend)
            .expect_err("refused");
        assert!(
            matches!(&error, AudioError::Checkpoint { what }
                if what.contains("--asr-mmproj") && what.contains("mmproj-x.gguf")),
            "{error}"
        );
    }
}

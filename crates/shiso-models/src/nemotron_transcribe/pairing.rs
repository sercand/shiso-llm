//! Which encoder export a decoder export is paired with, and where the
//! encoder's rows go in the decoder's prompt.
//!
//! The decoder names its encoder by file name in its own metadata, so the two
//! are found together the way they were published: side by side. The pairing
//! is resolved before either loads, because a decoder whose encoder is missing
//! can answer no audio request at all and finding that after twenty gigabytes
//! of weights moved is the most expensive place to find it.

use std::path::{Path, PathBuf};

use shiso_core::CoreError;
use shiso_modelfile::GgufFile;
use shiso_vision::MediaPlacement;

/// The decoder metadata key naming the encoder export's file.
pub const SPEECH_MODEL_KEY: &str = "nemo_speech.speech_model";

/// Where the decoder's template puts audio: one `<|audio|>` per clip, last in
/// the user turn, no bracket around the run.
///
/// The marker's id is one past the embedding table, so its rows are never
/// gathered: the engine substitutes id 0 at placeholder positions and the
/// encoder's rows overwrite them. This vocabulary declares no picture marker,
/// so the image spelling resolves to nothing and a picture is refused.
pub const PLACEMENT: MediaPlacement = MediaPlacement {
    image_marker: "<|image_pad|>",
    video_marker: None,
    wrap_image: None,
    audio_marker: Some("<|audio|>"),
    wrap_audio: None,
};

/// The encoder export the decoder at `decoder` pairs with.
///
/// `explicit` (the operator's override) wins when given; otherwise the file
/// [`SPEECH_MODEL_KEY`] names, beside the decoder. Either way the file must
/// exist.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] for a decoder that does not parse, or an
/// encoder path that names no file.
// `Ok(None)`: the decoder names no encoder and none was given - a text-only
// export of the family, which serves text and refuses audio.
pub fn speech_encoder_for(
    decoder: &Path,
    explicit: Option<&Path>,
) -> Result<Option<PathBuf>, CoreError> {
    let path = if let Some(path) = explicit {
        path.to_path_buf()
    } else {
        let file = GgufFile::open(decoder).map_err(|err| CoreError::InvalidValue {
            key: "nemotron_h gguf",
            value: format!("{}: {err}", decoder.display()),
        })?;
        let Some(name) = file.metadata().get_str(SPEECH_MODEL_KEY) else {
            return Ok(None);
        };
        decoder
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(name)
    };
    if !path.is_file() {
        return Err(CoreError::InvalidValue {
            key: SPEECH_MODEL_KEY,
            value: format!(
                "the speech encoder {} does not exist; put it beside the decoder or name it \
                 with --speech-encoder",
                path.display()
            ),
        });
    }
    Ok(Some(path))
}

#[cfg(test)]
#[path = "pairing_tests.rs"]
mod tests;

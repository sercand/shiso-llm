//! The instructions the speech decoder was trained to follow.
//!
//! The model card's prompt guide is the only source of these strings and this
//! file is their only home: the decoder answers a paraphrase fluently and
//! worse, so nothing here is reworded, and a request picks one by name rather
//! than writing its own.

/// What a transcription request asks the decoder to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Preset {
    /// The words as spoken, fillers and repetitions included: the checkpoint's
    /// own default.
    Verbatim,
    /// Readable text: fillers and accidental repetitions dropped.
    Clean,
    /// Transcription biased towards the given names and terms.
    Context(String),
    /// Who said what: the transcript carries `<spk:N>` tags.
    Speakers,
    /// Only the speaker nearest the microphone (the card calls it
    /// experimental).
    Nearest,
    /// Transcription with `<ev:*>` and `<sty:*>` event and style tags.
    Rich,
    /// The spoken language, as exactly one of these labels.
    LangId(Vec<String>),
    /// A translation into the named language.
    Translate(String),
}

/// A preset request this server will not turn into an instruction.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PresetError {
    /// No preset of that name.
    #[error(
        "preset {name:?} is not one of verbatim, clean, context, speakers, nearest, rich, \
         langid, translate"
    )]
    Unknown {
        /// The name asked for.
        name: String,
    },
    /// The preset needs a language (`translate`) or a label list (`langid`).
    #[error("preset {preset} needs `language`")]
    LanguageRequired {
        /// The preset.
        preset: &'static str,
    },
    /// A language was given to a preset that takes none.
    #[error("preset {preset} takes no `language`; only translate and langid do")]
    LanguageNotAccepted {
        /// The preset.
        preset: &'static str,
    },
    /// A context prompt was given to a preset the card defines no combined
    /// instruction for.
    #[error(
        "preset {preset} takes no `prompt`: the model card defines context terms only on \
         their own (preset verbatim or context)"
    )]
    PromptNotAccepted {
        /// The preset.
        preset: &'static str,
    },
    /// The `context` preset with no terms.
    #[error("preset context needs `prompt` naming the terms")]
    PromptRequired,
}

impl Preset {
    /// The instruction the user turn carries before the audio, as the card
    /// spells it.
    #[must_use]
    pub fn instruction(&self) -> String {
        match self {
            Self::Verbatim => "Audio transcription. Language: as spoken. Wording: verbatim, \
                               including fillers and repetitions. Formatting: punctuation and \
                               capitalization. Output: transcript only."
                .to_owned(),
            Self::Clean => "Audio transcription. Language: as spoken. Wording: omit fillers \
                            and accidental repetitions; keep the remaining spoken wording \
                            unchanged. Formatting: punctuation and capitalization. Output: \
                            transcript only."
                .to_owned(),
            Self::Context(terms) => {
                format!("Transcribe with punctuation and capitalization. Context terms: {terms}.")
            }
            Self::Speakers => "Transcribe this audio with punctuation and capitalization. \
                               Include all spoken words and verbal fillers. Write numbers in \
                               their standard written form. Put speaker tags <spk:0> <spk:1> \
                               <spk:2> <spk:3> on the transcription."
                .to_owned(),
            Self::Nearest => "Transcribe this audio. Include all spoken words and verbal \
                              fillers. Write numbers as spoken. No punctuation or \
                              capitalization. Transcribe only the speaker nearest to the \
                              microphone. Do not transcribe any other voices."
                .to_owned(),
            Self::Rich => "Transcribe this audio with punctuation and capitalization. Include \
                           all spoken words and verbal fillers. Write numbers in their \
                           standard written form. Use <ev:*> and <sty:*> special tokens to \
                           recognize all non-speech verbatims and speaking style."
                .to_owned(),
            Self::LangId(labels) => format!(
                "Identify the spoken language. Respond with exactly one label from {} and \
                 nothing else.",
                labels.join(", ")
            ),
            Self::Translate(language) => {
                format!("Please translate the given speech to {language}.")
            }
        }
    }

    /// Whether the transcript's `<spk:N>` tags are the answer rather than
    /// noise: only the preset that asks for them keeps them.
    #[must_use]
    pub const fn keeps_speakers(&self) -> bool {
        matches!(self, Self::Speakers)
    }

    /// The part of [`Self::instruction`] the caller wrote, when there is
    /// one: it is text even where it spells one of the vocabulary's markers.
    #[must_use]
    pub fn caller_text(&self) -> Option<String> {
        match self {
            Self::Context(terms) => Some(terms.clone()),
            Self::Translate(language) => Some(language.clone()),
            Self::LangId(labels) => Some(labels.join(", ")),
            _ => None,
        }
    }

    /// The preset's request name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Verbatim => "verbatim",
            Self::Clean => "clean",
            Self::Context(_) => "context",
            Self::Speakers => "speakers",
            Self::Nearest => "nearest",
            Self::Rich => "rich",
            Self::LangId(_) => "langid",
            Self::Translate(_) => "translate",
        }
    }
}

/// The preset a request names, with its `language` and `prompt` fields.
///
/// `language` is the target of `translate` and the comma-separated label list
/// of `langid`; any other preset refuses it. `prompt` turns `verbatim` (the
/// default) into context terms and is what `context` requires; every other
/// preset refuses it, because the card gives no instruction combining the two.
/// An empty or blank field counts as absent.
///
/// # Errors
///
/// A [`PresetError`] naming what was wrong.
pub fn parse_preset(
    name: &str,
    language: Option<&str>,
    prompt: Option<&str>,
) -> Result<Preset, PresetError> {
    let language = language.map(str::trim).filter(|s| !s.is_empty());
    let prompt = prompt.map(str::trim).filter(|s| !s.is_empty());
    let preset = match name.trim() {
        "verbatim" | "" => match prompt {
            Some(terms) => Preset::Context(terms.to_owned()),
            None => Preset::Verbatim,
        },
        "context" => Preset::Context(prompt.ok_or(PresetError::PromptRequired)?.to_owned()),
        "clean" => Preset::Clean,
        "speakers" => Preset::Speakers,
        "nearest" => Preset::Nearest,
        "rich" => Preset::Rich,
        "langid" => {
            let labels = language
                .ok_or(PresetError::LanguageRequired { preset: "langid" })?
                .split(',')
                .map(str::trim)
                .filter(|label| !label.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>();
            if labels.is_empty() {
                return Err(PresetError::LanguageRequired { preset: "langid" });
            }
            Preset::LangId(labels)
        }
        "translate" => Preset::Translate(
            language
                .ok_or(PresetError::LanguageRequired {
                    preset: "translate",
                })?
                .to_owned(),
        ),
        other => {
            return Err(PresetError::Unknown {
                name: other.to_owned(),
            });
        }
    };
    let takes_prompt = matches!(preset, Preset::Context(_));
    if prompt.is_some() && !takes_prompt {
        return Err(PresetError::PromptNotAccepted {
            preset: preset.name(),
        });
    }
    let takes_language = matches!(preset, Preset::Translate(_) | Preset::LangId(_));
    if language.is_some() && !takes_language {
        return Err(PresetError::LanguageNotAccepted {
            preset: preset.name(),
        });
    }
    Ok(preset)
}

#[cfg(test)]
#[path = "presets_tests.rs"]
mod tests;

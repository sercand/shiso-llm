//! The prompt the decoder reads, and what its output means.
//!
//! The prompt is the chat template with an empty system turn and one audio
//! item, its single placeholder expanded to one `<|audio_pad|>` per tower row;
//! a forced language appends `language {Name}<asr_text>` after the generation
//! prompt, so the model writes the transcript alone. Neither needs a
//! tokenizer: the template's ids are fixed, and each language's suffix was
//! captured from the checkpoint's tokenizer (both checkpoints share it) and
//! is spelled back through the vocabulary by a test.
//!
//! The output is parsed as the reference's `parse_asr_output` does: strip,
//! collapse degenerate repetition, then split at the first `<asr_text>`. The
//! collapse works on characters of the decoded text, as the reference's does,
//! not on token ids: a repeated phrase is rarely the same ids twice at a
//! token boundary, and the reference's thresholds count characters.

use shiso_audio::AudioError;

use super::config::{
    ASR_TEXT, AUDIO_END, AUDIO_PAD, AUDIO_START, IM_END, IM_START, language_name, languages,
};

/// `system`.
pub(crate) const SYSTEM: u32 = 8948;
/// `user`.
pub(crate) const USER: u32 = 872;
/// `assistant`.
pub(crate) const ASSISTANT: u32 = 77091;
/// `\n`.
pub(crate) const NEWLINE: u32 = 198;
/// `language`, the first id of a forced suffix and of the model's own answer.
const LANGUAGE: u32 = 11528;
/// The text of the tag the transcript follows.
const ASR_TEXT_TAG: &str = "<asr_text>";
/// What a meta line starts with.
const LANGUAGE_PREFIX: &str = "language ";
/// The collapse threshold the reference parses with.
pub const REPEAT_THRESHOLD: usize = 20;
/// The longest pattern the collapse looks for.
const MAX_PATTERN: usize = 20;

/// Ids of ` {Name}` for each language the model can be forced to, in the
/// config's order: every name is one token but Cantonese and Macedonian.
const NAME_IDS: [(&str, &[u32]); 30] = [
    ("Chinese", &[8453]),
    ("English", &[6364]),
    ("Cantonese", &[72366, 2367]),
    ("Arabic", &[34117]),
    ("German", &[5938]),
    ("French", &[8585]),
    ("Spanish", &[15154]),
    ("Portuguese", &[42188]),
    ("Indonesian", &[58829]),
    ("Italian", &[14811]),
    ("Korean", &[16134]),
    ("Russian", &[8522]),
    ("Thai", &[26392]),
    ("Vietnamese", &[48477]),
    ("Japanese", &[10769]),
    ("Turkish", &[23734]),
    ("Hindi", &[43980]),
    ("Malay", &[79140]),
    ("Dutch", &[23234]),
    ("Swedish", &[30109]),
    ("Danish", &[43680]),
    ("Finnish", &[57853]),
    ("Polish", &[31984]),
    ("Czech", &[33150]),
    ("Filipino", &[62417]),
    ("Persian", &[49861]),
    ("Greek", &[17860]),
    ("Romanian", &[73597]),
    ("Hungarian", &[56769]),
    ("Macedonian", &[56452, 75491]),
];

/// A transcript split into its parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    /// The language the model named, or the one it was forced to; `None` when
    /// it named none or answered `language None` (silence).
    pub language: Option<String>,
    /// The transcript, stripped.
    pub text: String,
}

/// Ids of the prompt around its placeholder run and context: nine before the
/// run, six after.
pub(crate) const FRAME_IDS: usize = 15;

/// The most context ids a prompt carries: some 350 words of mixed Turkish
/// and English, far past any keyword list, and small beside the ~450 audio
/// rows of a 35 s chunk, so a pooled cache sized for it stays small.
pub(crate) const MAX_CONTEXT_IDS: usize = 512;

/// What a prompt is conditioned on beyond its audio.
#[derive(Debug, Clone, Copy, Default)]
pub struct Conditioning<'a> {
    /// The language forced, by the model's name for it; `None` lets the
    /// model name it.
    pub forced: Option<&'a str>,
    /// The system turn's text, as ids ([`super::context`]); empty for none.
    pub context: &'a [u32],
}

impl<'a> Conditioning<'a> {
    /// A forced language (or none) and no context.
    #[must_use]
    pub const fn forced(forced: Option<&'a str>) -> Self {
        Self {
            forced,
            context: &[],
        }
    }
}

/// Ids of the longest [`forced_suffix`].
pub(crate) const LONGEST_SUFFIX: usize = longest_suffix();

const fn longest_suffix() -> usize {
    let mut longest = 0;
    let mut i = 0;
    while i < NAME_IDS.len() {
        if NAME_IDS[i].1.len() > longest {
            longest = NAME_IDS[i].1.len();
        }
        i += 1;
    }
    // `language` before the name, `<asr_text>` after it.
    longest + 2
}

/// The ids of `language {name}<asr_text>` for a language the model supports,
/// named as the model names it (in any case); `None` for any other.
#[must_use]
pub fn forced_suffix(name: &str) -> Option<Vec<u32>> {
    let (_, name_ids) = NAME_IDS
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))?;
    let mut out = Vec::with_capacity(name_ids.len() + 2);
    out.push(LANGUAGE);
    out.extend_from_slice(name_ids);
    out.push(ASR_TEXT);
    Some(out)
}

/// The prompt for `n_audio` tower rows, with the context in the system turn
/// (empty, as the reference captured it, when there is none):
///
/// ```text
/// <|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>
/// <|audio_pad|> x n_audio <|audio_end|><|im_end|>\n<|im_start|>assistant\n
/// ```
///
/// followed, for a forced language, by `language {Name}<asr_text>`. The
/// placeholders are one run starting at index `9 + context.len()`.
///
/// # Errors
///
/// [`AudioError::BadContext`] for more than [`MAX_CONTEXT_IDS`] context ids;
/// [`AudioError::BadSpec`] for a forced language the model does not support
/// (by the name the model writes, e.g. `English`, or an ISO code).
pub fn ids(n_audio: usize, cond: &Conditioning<'_>) -> Result<Vec<u32>, AudioError> {
    if cond.context.len() > MAX_CONTEXT_IDS {
        return Err(AudioError::BadContext {
            reason: format!(
                "{} context ids; a prompt carries at most {MAX_CONTEXT_IDS}",
                cond.context.len()
            ),
        });
    }
    let suffix = match cond.forced {
        None => Vec::new(),
        Some(name) => {
            let canonical = language_name(name).unwrap_or(name);
            forced_suffix(canonical).ok_or_else(|| AudioError::BadSpec {
                reason: format!(
                    "the language {name:?} cannot be forced; supported: {}",
                    languages()
                        .iter()
                        .map(|(_, n)| *n)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            })?
        }
    };
    let mut out = Vec::with_capacity(FRAME_IDS + cond.context.len() + n_audio + suffix.len());
    out.extend_from_slice(&[IM_START, SYSTEM, NEWLINE]);
    out.extend_from_slice(cond.context);
    out.extend_from_slice(&[IM_END, NEWLINE, IM_START, USER, NEWLINE, AUDIO_START]);
    out.extend(std::iter::repeat_n(AUDIO_PAD, n_audio));
    out.extend_from_slice(&[AUDIO_END, IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE]);
    out.extend_from_slice(&suffix);
    Ok(out)
}

/// `Chinese` from `cHINese`: first letter upper, the rest lower, as the
/// reference normalises a language name.
fn normalise(name: &str) -> String {
    let mut chars = name.chars();
    chars.next().map_or_else(String::new, |first| {
        first
            .to_uppercase()
            .chain(chars.flat_map(char::to_lowercase))
            .collect()
    })
}

/// The decoded output, stripped, collapsed and split at the first
/// `<asr_text>` as the reference's output parser does:
///
/// - a forced language: the whole output is the text;
/// - no tag: the whole output is the text, with no language;
/// - `language None` (any case) before the tag: silence, no language, and
///   whatever text follows the tag;
/// - otherwise the language is the first meta line starting `language `.
#[must_use]
pub fn parse(raw: &str, forced: Option<&str>) -> Parsed {
    let stripped = raw.trim();
    if stripped.is_empty() {
        return Parsed {
            language: None,
            text: String::new(),
        };
    }
    let s = collapse_repeats(stripped, REPEAT_THRESHOLD);
    if let Some(language) = forced {
        return Parsed {
            language: Some(language.to_owned()),
            text: s,
        };
    }
    let Some((meta, text)) = s.split_once(ASR_TEXT_TAG) else {
        return Parsed {
            language: None,
            text: s.trim().to_owned(),
        };
    };
    let text = text.trim().to_owned();
    if meta.to_lowercase().contains("language none") {
        return Parsed {
            language: None,
            text,
        };
    }
    let language = meta
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .find(|line| line.to_lowercase().starts_with(LANGUAGE_PREFIX))
        .map(|line| line[LANGUAGE_PREFIX.len()..].trim())
        .filter(|name| !name.is_empty())
        .map(normalise);
    Parsed { language, text }
}

/// Degenerate repetition collapsed as the reference's repetition fix does,
/// over characters: first a run of one character longer than `threshold` becomes
/// that character; then, scanning from the left, the first pattern of 1 to 20
/// characters repeated at least `threshold` times becomes one copy, and the
/// rest of the string after the repeats is collapsed the same way.
#[must_use]
pub fn collapse_repeats(text: &str, threshold: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut runs = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let mut count = 1;
        while i + count < chars.len() && chars[i + count] == chars[i] {
            count += 1;
        }
        if count > threshold {
            runs.push(chars[i]);
        } else {
            runs.extend_from_slice(&chars[i..i + count]);
        }
        i += count;
    }
    let mut out = String::with_capacity(text.len());
    collapse_patterns(&runs, threshold.max(1), &mut out);
    out
}

/// The pattern half of [`collapse_repeats`], appending to `out`.
fn collapse_patterns(s: &[char], threshold: usize, out: &mut String) {
    let n = s.len();
    if n < threshold * 2 {
        out.extend(s);
        return;
    }
    let mut i = 0;
    while i <= n - threshold * 2 {
        for k in 1..=MAX_PATTERN {
            if i + k * threshold > n {
                break;
            }
            let pattern = &s[i..i + k];
            if (1..threshold).all(|rep| &s[i + rep * k..i + rep * k + k] == pattern) {
                let mut end = i + threshold * k;
                while end + k <= n && &s[end..end + k] == pattern {
                    end += k;
                }
                out.extend(pattern);
                collapse_patterns(&s[end..], threshold, out);
                return;
            }
        }
        out.push(s[i]);
        i += 1;
    }
    out.extend(&s[i..]);
}

#[cfg(test)]
#[path = "prompt_tests.rs"]
mod tests;

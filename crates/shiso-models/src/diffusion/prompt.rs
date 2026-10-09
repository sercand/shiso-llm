//! Prompt text to token ids, with the caller's words kept out of the
//! template's control tokens.
//!
//! An image family conditions on a text encoder that reads a templated
//! prompt: turn markers (`<|im_start|>`) written as text that the vocabulary
//! turns into control ids, with the caller's prompt in one slot. A prompt
//! that *spells* a marker must not gain a turn boundary, so the template's
//! own segments encode with added tokens recognised and the caller's text
//! encodes with them read as plain text. [`PromptTokenizer`] holds the
//! vocabulary and both encodings; [`render_user_turn`] renders a
//! checkpoint's chat template around a single user turn and splits it at the
//! content slot, for a family that conditions on the chat template rather
//! than a fixed string.

use std::path::Path;

use shiso_vision::generator::ImageGenError;

/// Private-use character written into the template's content slot instead of
/// the prompt, so the rendered string can be split back around it. No chat
/// template emits a private-use codepoint of its own.
const CONTENT_MARK: char = '\u{F000}';

/// A text encoder's vocabulary, with the spellings it reads as control
/// tokens.
#[derive(Debug)]
pub(crate) struct PromptTokenizer {
    tokenizer: tokenizers::Tokenizer,
    /// The same vocabulary with added tokens read as plain text, built once:
    /// caller text that spells a control token encodes through it.
    as_text: tokenizers::Tokenizer,
    /// The spellings the vocabulary reads as control tokens.
    special_spellings: Vec<String>,
}

impl PromptTokenizer {
    /// Opens `tokenizer.json` at `path`, or inside it when `path` is a
    /// directory.
    pub(crate) fn open(path: &Path) -> Result<Self, ImageGenError> {
        let file = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&file).map_err(|e| {
            ImageGenError::Checkpoint(format!("opening tokenizer {}: {e}", file.display()))
        })?;
        let special_spellings: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter(|token| token.special)
            .map(|token| token.content)
            .collect();
        let mut as_text = tokenizer.clone();
        as_text.set_encode_special_tokens(true);
        Ok(Self {
            tokenizer,
            as_text,
            special_spellings,
        })
    }

    /// Whether `text` spells any control token of this vocabulary.
    pub(crate) fn spells_control_token(&self, text: &str) -> bool {
        self.special_spellings
            .iter()
            .any(|spelling| text.contains(spelling.as_str()))
    }

    /// Encodes template text: control-token spellings become control ids.
    pub(crate) fn encode_template(&self, text: &str) -> Result<Vec<u32>, ImageGenError> {
        encode(&self.tokenizer, text)
    }

    /// Encodes caller text: a control-token spelling stays words.
    ///
    /// Text that spells no control token takes the plain vocabulary, which
    /// gives the same ids: the two differ only in how they read added tokens.
    pub(crate) fn encode_caller(&self, text: &str) -> Result<Vec<u32>, ImageGenError> {
        if self.spells_control_token(text) {
            encode(&self.as_text, text)
        } else {
            encode(&self.tokenizer, text)
        }
    }
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, ImageGenError> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|e| ImageGenError::Checkpoint(format!("tokenizing: {e}")))
}

/// Answers the Python-shaped string methods a checkpoint's chat template
/// calls (`startswith`, `split`, `strip('\n')`, ...) that the Rust engine
/// exposes only as filters or not at all. Mirrors the server's tokenizer
/// bridge; it is duplicated rather than shared because `shiso-models` must
/// not depend on `shiso-server`, and the templates — not our code — decide
/// which methods exist. An argument is treated as a Python character set,
/// which is what the Qwen3-family templates pass (`'\n'`).
fn bridge_python_methods(
    _state: &minijinja::State<'_, '_>,
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> Result<minijinja::Value, minijinja::Error> {
    use minijinja::value::{ValueKind, from_args};
    if value.kind() != ValueKind::String {
        return Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod));
    }
    let text = value.as_str().unwrap_or_default();
    match method {
        "startswith" => {
            let (prefix,): (String,) = from_args(args)?;
            Ok(minijinja::Value::from(text.starts_with(&prefix)))
        }
        "endswith" => {
            let (suffix,): (String,) = from_args(args)?;
            Ok(minijinja::Value::from(text.ends_with(&suffix)))
        }
        "replace" => {
            let (from, to): (String, String) = from_args(args)?;
            Ok(minijinja::Value::from(text.replace(&from, &to)))
        }
        "strip" | "lstrip" | "rstrip" => {
            let (chars,): (Option<String>,) = from_args(args)?;
            let stripped = match chars.as_deref() {
                Some(set) if !set.is_empty() => match method {
                    "strip" => text.trim_matches(|c| set.contains(c)),
                    "lstrip" => text.trim_start_matches(|c| set.contains(c)),
                    _ => text.trim_end_matches(|c| set.contains(c)),
                },
                // Python's no-argument form strips whitespace. `strip('')`
                // is a no-op there, but no template in the wild passes an
                // empty set, so the empty set falls in with whitespace.
                _ => match method {
                    "strip" => text.trim(),
                    "lstrip" => text.trim_start(),
                    _ => text.trim_end(),
                },
            };
            Ok(minijinja::Value::from(stripped))
        }
        "split" => {
            let (sep,): (Option<String>,) = from_args(args)?;
            let parts: Vec<String> = match sep.as_deref() {
                Some(s) if !s.is_empty() => text.split(s).map(str::to_owned).collect(),
                // No separator (or the empty string): Python splits on
                // runs of whitespace, which is how the thinking-block
                // templates never hit this path anyway.
                _ => text.split_whitespace().map(str::to_owned).collect(),
            };
            Ok(minijinja::Value::from(parts))
        }
        _ => Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod)),
    }
}

/// Renders `chat_template` around one user turn and returns the text before
/// and after its content slot.
///
/// The content slot is filled by a private-use mark and the result is split
/// around it. The generation prompt stays on: the assistant turn opener
/// (and, for a Qwen3 template with thinking disabled, its empty `think`
/// block) belongs to the tail that encodes after the prompt.
pub(crate) fn render_user_turn(chat_template: &str) -> Result<(String, String), ImageGenError> {
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(bridge_python_methods);
    let template = env
        .template_from_str(chat_template)
        .map_err(|e| ImageGenError::Checkpoint(format!("chat template: {e}")))?;
    let rendered = template
        .render(serde_json::json!({
            "messages": [
                { "role": "user", "content": CONTENT_MARK.to_string() },
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        }))
        .map_err(|e| ImageGenError::Checkpoint(format!("chat template: {e}")))?;
    rendered
        .split_once(CONTENT_MARK)
        .map(|(head, tail)| (head.to_string(), tail.to_string()))
        .ok_or_else(|| {
            ImageGenError::Checkpoint(
                "the chat template did not emit the prompt content".to_string(),
            )
        })
}

/// Reads the checkpoint's chat template: `chat_template.jinja` beside the
/// tokenizer, else `tokenizer_config.json`'s `chat_template` (a string or an
/// object with a `default`).
///
/// A checkpoint that carries neither is refused: the template decides the
/// turn structure the text encoder was trained to read, and a built-in
/// generic one would silently condition the transformer on markers the
/// checkpoint's own vocabulary may not even carry. Every real checkpoint
/// ships its template beside the tokenizer.
pub(crate) fn load_chat_template(tokenizer_dir: Option<&Path>) -> Result<String, ImageGenError> {
    let dir = tokenizer_dir.ok_or_else(|| {
        ImageGenError::Checkpoint("the tokenizer path names no directory to read".to_string())
    })?;
    let jinja = dir.join("chat_template.jinja");
    if jinja.exists() {
        return std::fs::read_to_string(&jinja)
            .map_err(|e| ImageGenError::Checkpoint(format!("reading {}: {e}", jinja.display())));
    }
    let config = dir.join("tokenizer_config.json");
    if config.exists() {
        let text = std::fs::read_to_string(&config)
            .map_err(|e| ImageGenError::Checkpoint(format!("reading {}: {e}", config.display())))?;
        let document: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ImageGenError::Checkpoint(format!("parsing {}: {e}", config.display())))?;
        match &document["chat_template"] {
            serde_json::Value::String(template) => return Ok(template.clone()),
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::String(template)) = map.get("default") {
                    return Ok(template.clone());
                }
            }
            _ => {}
        }
    }
    Err(ImageGenError::Checkpoint(format!(
        "no chat template found beside the tokenizer (searched {} and its \
         tokenizer_config.json's chat_template); refusing to condition on a \
         built-in template the checkpoint was not trained with",
        jinja.display()
    )))
}

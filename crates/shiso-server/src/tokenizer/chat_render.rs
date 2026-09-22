//! Rendering a prepared conversation through a checkpoint's own chat template.
//!
//! # One rendering path, and the tests use it
//!
//! Prompt bytes are produced here and nowhere else. A second implementation for
//! tests would be a second thing that can drift from what the server actually
//! sends, and the failure mode of that drift is a golden test that passes while
//! production sends a different prompt.
//!
//! # What the render knobs are, and why they are explicit
//!
//! Templates read a handful of context variables beyond the messages -
//! whether to open a thinking block, what effort tier to steer at, whether to
//! keep historical traces. Several of them have in-template defaults, and those
//! defaults are frequently the most expensive option. Passing every one
//! explicitly means the effective default lives here, in code that can be read,
//! rather than inside a document shipped in a checkpoint.
//!
//! The one exception is deliberately tri-state: a variable that is left
//! *undefined* is not the same as one set to null, because a template can test
//! for undefined and fall back to its own behaviour. [`RenderFlags`] spells that
//! as `None` and passes `UNDEFINED`, never a null.

use minijinja::{Environment, Value, context};
use serde_json::Value as Json;

use crate::tokenizer::jinja_helpers::{TEMPLATE_NAME, TemplateError, ToolJsonStyle};

/// The effort tier passed when the caller named none and thinking is on.
///
/// The neutral rung, not the highest one. Templates that steer on effort
/// frequently default to their most expensive tier when the variable is
/// undefined, which means an effort-silent client would silently buy the most
/// expensive directive. Passing an explicit neutral value makes that
/// unreachable.
pub const NEUTRAL_EFFORT: &str = "medium";

/// The effort tier passed when thinking is off.
pub const NO_EFFORT: &str = "none";

/// What a render is told beyond the conversation itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderFlags<'a> {
    /// Whether the model should open a thinking block.
    pub enable_thinking: bool,
    /// The effort tier, when the caller named one.
    pub reasoning_effort: Option<&'a str>,
    /// Whether historical thinking blocks are kept. `None` leaves the variable
    /// undefined so the template's own default applies; see the module docs for
    /// why that is not the same as null.
    pub preserve_thinking: Option<bool>,
    /// Whether the last turn is continued rather than answered.
    ///
    /// Only takes effect when the last turn is an assistant turn, because
    /// continuing a user turn is not a thing a model does.
    pub continue_final: bool,
}

/// A compiled chat template.
#[derive(Debug)]
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    /// Compiles a template.
    ///
    /// # Errors
    ///
    /// [`TemplateError::Compile`] when the template does not parse.
    pub fn new(template: &str, style: ToolJsonStyle) -> Result<Self, TemplateError> {
        Ok(Self {
            env: crate::tokenizer::jinja_helpers::build_environment(template, style)?,
        })
    }

    /// Renders one conversation into prompt text.
    ///
    /// `messages` is what [`super::message_preprocess::preprocess`] produced;
    /// `tools` is the tool list in the wire shape a template expects.
    ///
    /// # Errors
    ///
    /// [`TemplateError::Render`] when the template raises, which includes the
    /// exceptions a template raises for conversations it refuses.
    pub fn render(
        &self,
        messages: &[Json],
        tools: Option<&[Json]>,
        flags: RenderFlags<'_>,
    ) -> Result<String, TemplateError> {
        let template =
            self.env
                .get_template(TEMPLATE_NAME)
                .map_err(|error| TemplateError::Render {
                    message: error.to_string(),
                })?;

        let continue_final = flags.continue_final
            && messages
                .last()
                .and_then(|turn| turn.get("role"))
                .and_then(Json::as_str)
                == Some("assistant");

        let effort: Value = match flags.reasoning_effort {
            Some(named) => named.into(),
            None if flags.enable_thinking => NEUTRAL_EFFORT.into(),
            None => NO_EFFORT.into(),
        };
        let preserve = flags
            .preserve_thinking
            .map_or(Value::UNDEFINED, Value::from);

        let ctx = context! {
            messages => Value::from_serialize(messages),
            tools => tools.map_or(Value::UNDEFINED, Value::from_serialize),
            add_generation_prompt => !continue_final,
            enable_thinking => flags.enable_thinking,
            reasoning_effort => effort,
            preserve_thinking => preserve,
            add_vision_id => false,
            bos_token => "<bos>",
        };

        let rendered = template
            .render(ctx)
            .map_err(|error| TemplateError::Render {
                message: error.to_string(),
            })?;

        if continue_final {
            // The trailing end-of-turn marker is what tells the model the turn
            // is over; leaving it in and then asking the model to continue that
            // turn is a contradiction the model resolves by starting a new one.
            let trimmed = rendered.trim_end();
            let stripped = trimmed.strip_suffix("<|im_end|>").unwrap_or(trimmed);
            return Ok(stripped.to_owned());
        }
        Ok(rendered)
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatTemplate, RenderFlags};
    use crate::tokenizer::jinja_helpers::ToolJsonStyle;
    use serde_json::json;

    /// A minimal stand-in for the templates checkpoints ship: it reads the
    /// same variables, so the context this module builds is what is under test
    /// rather than any one checkpoint's wording.
    const TEMPLATE: &str = concat!(
        "{% for m in messages %}",
        "<|im_start|>{{ m.role }}\n",
        "{% for part in m.content %}{{ part.text }}{% endfor %}",
        "<|im_end|>\n",
        "{% endfor %}",
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        "{% if enable_thinking %}[think:{{ reasoning_effort }}]{% endif %}",
    );

    fn turns() -> Vec<serde_json::Value> {
        vec![json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})]
    }

    #[test]
    fn a_rendered_conversation_ends_with_the_generation_prompt() {
        let template = ChatTemplate::new(TEMPLATE, ToolJsonStyle::Compact).expect("it compiles");
        let out = template
            .render(&turns(), None, RenderFlags::default())
            .expect("it renders");
        assert_eq!(
            out,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// An effort-silent caller with thinking on gets the neutral tier, never
    /// the template's own default.
    #[test]
    fn thinking_without_a_named_effort_renders_the_neutral_tier() {
        let template = ChatTemplate::new(TEMPLATE, ToolJsonStyle::Compact).expect("it compiles");
        let out = template
            .render(
                &turns(),
                None,
                RenderFlags {
                    enable_thinking: true,
                    ..RenderFlags::default()
                },
            )
            .expect("it renders");
        assert!(out.ends_with("[think:medium]"), "{out}");
    }

    #[test]
    fn a_named_effort_reaches_the_template_verbatim() {
        let template = ChatTemplate::new(TEMPLATE, ToolJsonStyle::Compact).expect("it compiles");
        let out = template
            .render(
                &turns(),
                None,
                RenderFlags {
                    enable_thinking: true,
                    reasoning_effort: Some("low"),
                    ..RenderFlags::default()
                },
            )
            .expect("it renders");
        assert!(out.ends_with("[think:low]"), "{out}");
    }

    /// Continuing is only possible when the last turn is the assistant's; a
    /// caller that asks for it on a user turn gets an ordinary render rather
    /// than a prompt with its final marker cut off.
    #[test]
    fn continuing_a_user_turn_renders_as_an_ordinary_turn() {
        let template = ChatTemplate::new(TEMPLATE, ToolJsonStyle::Compact).expect("it compiles");
        let out = template
            .render(
                &turns(),
                None,
                RenderFlags {
                    continue_final: true,
                    ..RenderFlags::default()
                },
            )
            .expect("it renders");
        assert!(out.ends_with("<|im_start|>assistant\n"), "{out}");
    }

    #[test]
    fn continuing_an_assistant_turn_strips_its_end_of_turn_marker() {
        let template = ChatTemplate::new(TEMPLATE, ToolJsonStyle::Compact).expect("it compiles");
        let mut messages = turns();
        messages.push(json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "the answer is"}]
        }));
        let out = template
            .render(
                &messages,
                None,
                RenderFlags {
                    continue_final: true,
                    ..RenderFlags::default()
                },
            )
            .expect("it renders");
        assert!(out.ends_with("the answer is"), "{out}");
    }

    /// A template that raises reports the raise, rather than being reported as
    /// a broken template.
    #[test]
    fn a_template_exception_is_reported_as_a_render_failure() {
        let template = ChatTemplate::new(
            "{{ raise_exception('this conversation is not renderable') }}",
            ToolJsonStyle::Compact,
        )
        .expect("it compiles");
        let error = template
            .render(&turns(), None, RenderFlags::default())
            .expect_err("the template raises");
        assert!(
            error.to_string().contains("not renderable"),
            "the raise message is lost: {error}"
        );
    }
}

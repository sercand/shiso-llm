//! The template environment, and the two things about it that are load bearing.
//!
//! # Key order in `tojson` is a correctness property
//!
//! A chat template writes tool definitions into the prompt with
//! `{{ tool | tojson }}`. The model was trained on prompts produced by a
//! renderer that serialises objects in insertion order, so a serialiser that
//! sorted keys would produce a prompt the model has never seen - the same tools,
//! described in an order it has no training signal for. The `preserve_order`
//! feature on the JSON crate is what keeps insertion order all the way through,
//! and it is a dependency feature rather than a line of code here, which is
//! exactly why it needs saying out loud somewhere.
//!
//! # Why a spaced serialiser exists beside the compact one
//!
//! The reference implementation these templates were written against separates
//! JSON tokens with `", "` and `": "`; the compact form has no spaces at all.
//! It is a pure byte-stream difference with no change in meaning, and models
//! have been observed to be measurably sensitive to it in either direction. So
//! both are available and the choice is an explicit argument to
//! [`build_environment`] rather than a process-wide environment read: a test
//! that flipped a global would change what every concurrently running test
//! renders, and that has produced real intermittent failures elsewhere.

use minijinja::{Environment, Value};

/// How `{{ x | tojson }}` writes JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolJsonStyle {
    /// No spaces: `{"a":1}`, from the template engine's built-in, which also
    /// escapes `<`, `>`, `&` and `'` for HTML.
    ///
    /// The server no longer renders with it (see
    /// [`crate::SERVER_TOOL_JSON_STYLE`]): the templates it serves were written
    /// for the Python engine, whose output is spaced and unescaped, and a schema
    /// rendered this way is one the model was not trained on. It stays for the
    /// tests that pin the context this crate builds independently of the style.
    #[default]
    Compact,
    /// Reference spacing: `{"a": 1}`.
    Spaced,
}

/// Why a template could not be prepared.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TemplateError {
    /// The template does not compile.
    #[error("the chat template does not compile: {message}")]
    Compile {
        /// What the template engine reported.
        message: String,
    },
    /// The template compiled but did not render.
    #[error("the chat template did not render: {message}")]
    Render {
        /// What the template engine reported.
        message: String,
    },
}

/// The name the chat template is registered under.
pub const TEMPLATE_NAME: &str = "chat";

/// Builds an environment with the chat template compiled into it.
///
/// The template text is leaked to `'static`. That is deliberate and is not a
/// leak in any meaningful sense: one of these is built per process and lives
/// for as long as the process serves, so the alternative - a self-referential
/// owner, or a re-parse per render - buys nothing back.
///
/// # Errors
///
/// [`TemplateError::Compile`] when the template does not parse.
pub fn build_environment(
    template: &str,
    style: ToolJsonStyle,
) -> Result<Environment<'static>, TemplateError> {
    let leaked: &'static str = Box::leak(template.to_owned().into_boxed_str());
    let mut env = Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    // Templates raise their own errors for conversations they refuse to render -
    // an assistant turn with no content, a tool result with no call. Without
    // this the raise itself is an unknown-function error, which reports the
    // template as broken rather than the request as wrong.
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    env.add_filter("rtrim", |s: String| s.trim_end_matches('\n').to_owned());
    env.add_filter("ltrim", |s: String| s.trim_start_matches('\n').to_owned());
    env.add_filter("split_first", |s: String, sep: String| {
        s.split(&sep).next().unwrap_or_default().to_owned()
    });
    env.add_filter("split_last", |s: String, sep: String| {
        s.rsplit(&sep).next().unwrap_or_default().to_owned()
    });

    // These templates are written for a Python engine, where strings and maps
    // carry methods that the Rust engine exposes only as filters. Bridging them
    // here costs nothing when a template never calls one, and is the difference
    // between rendering and a 500 when it does.
    env.set_unknown_method_callback(bridge_python_methods);

    env.add_filter("tojson_reference", spaced_tojson);
    if style == ToolJsonStyle::Spaced {
        env.add_filter("tojson", spaced_tojson);
    }

    env.add_template(TEMPLATE_NAME, leaked)
        .map_err(|error| TemplateError::Compile {
            message: error.to_string(),
        })?;
    Ok(env)
}

/// Answers the string and map methods a Python-written template calls.
fn bridge_python_methods(
    state: &minijinja::State<'_, '_>,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, minijinja::Error> {
    use minijinja::value::{ValueKind, from_args};
    if value.kind() == ValueKind::String {
        match method {
            "split" => {
                let (sep,): (Option<Value>,) = from_args(args)?;
                let mut filter_args = vec![value.clone()];
                if let Some(sep) = sep {
                    filter_args.push(sep);
                }
                return state.apply_filter("split", &filter_args);
            }
            // A checkpoint's own template tests whether a turn is a tool
            // response by looking at its first and last characters. Without
            // these two the template does not render at all, and the request
            // fails as if the caller had sent something wrong.
            "startswith" => {
                let (prefix,): (String,) = from_args(args)?;
                return Ok(Value::from(
                    value.as_str().unwrap_or_default().starts_with(&prefix),
                ));
            }
            "endswith" => {
                let (suffix,): (String,) = from_args(args)?;
                return Ok(Value::from(
                    value.as_str().unwrap_or_default().ends_with(&suffix),
                ));
            }
            "replace" => {
                let (from, to): (String, String) = from_args(args)?;
                return Ok(Value::from(
                    value.as_str().unwrap_or_default().replace(&from, &to),
                ));
            }
            "strip" => {
                let (): () = from_args(args)?;
                return Ok(Value::from(value.as_str().unwrap_or_default().trim()));
            }
            "lstrip" => {
                let (): () = from_args(args)?;
                return Ok(Value::from(value.as_str().unwrap_or_default().trim_start()));
            }
            "rstrip" => {
                let (): () = from_args(args)?;
                return Ok(Value::from(value.as_str().unwrap_or_default().trim_end()));
            }
            _ => {}
        }
    }
    if value.kind() == ValueKind::Map {
        match method {
            "items" => {
                let (): () = from_args(args)?;
                return state.apply_filter("items", std::slice::from_ref(value));
            }
            "get" => {
                let (key, default): (Value, Option<Value>) = from_args(args)?;
                return Ok(value
                    .get_item(&key)
                    .ok()
                    .filter(|found| !found.is_undefined())
                    .unwrap_or_else(|| default.unwrap_or(Value::UNDEFINED)));
            }
            _ => {}
        }
    }
    Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod))
}

/// `tojson` with reference spacing.
fn spaced_tojson(value: &Value) -> Result<Value, minijinja::Error> {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, SpacedFormatter);
    serde::Serialize::serialize(value, &mut serializer).map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson failed: {error}"),
        )
    })?;
    let json = String::from_utf8(buf).map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson produced invalid UTF-8: {error}"),
        )
    })?;
    Ok(Value::from_safe_string(json))
}

/// A JSON formatter that writes `", "` between items and `": "` after a key.
#[derive(Clone, Debug)]
struct SpacedFormatter;

impl serde_json::ser::Formatter for SpacedFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}

#[cfg(test)]
mod tests {
    use super::{TEMPLATE_NAME, ToolJsonStyle, build_environment};

    /// Renders a one-expression template over one named value.
    fn render(source: &str, name: &str, value: &minijinja::Value) -> String {
        let env = build_environment(source, ToolJsonStyle::Compact)
            .expect("a one-expression template compiles");
        env.get_template(TEMPLATE_NAME)
            .expect("the template is registered")
            .render(minijinja::context! { text => value.clone(), tool => value })
            .unwrap_or_else(|error| panic!("{name} renders: {error}"))
    }

    /// A checkpoint's own template decides whether a turn is a tool response by
    /// testing its ends, so a missing method here is a checkpoint that cannot
    /// be served at all rather than a feature nobody notices.
    #[test]
    fn a_template_can_test_how_a_string_begins_and_ends() {
        let answer = render(
            "{{ text.startswith('<tool_response>') }}|{{ text.endswith('</tool_response>') }}",
            "the ends of a string",
            &minijinja::Value::from("<tool_response>ok</tool_response>"),
        );
        assert_eq!(answer, "True|True");

        let answer = render(
            "{{ text.startswith('<tool_response>') }}|{{ text.endswith('</tool_response>') }}",
            "the ends of a string",
            &minijinja::Value::from("hello"),
        );
        assert_eq!(answer, "False|False");
    }

    /// The order the keys were written in is the order they render in. A
    /// sorting serialiser would put `description` first here.
    #[test]
    fn tojson_preserves_the_order_the_keys_were_written_in() {
        let env = build_environment("{{ tool | tojson }}", ToolJsonStyle::Compact)
            .expect("a one-expression template compiles");
        let tool = serde_json::json!({"name": "b", "description": "a", "input_schema": {}});
        let rendered = env
            .get_template(TEMPLATE_NAME)
            .expect("the template is registered")
            .render(minijinja::context! { tool => minijinja::Value::from_serialize(&tool) })
            .expect("the template renders");
        assert_eq!(
            rendered,
            r#"{"name":"b","description":"a","input_schema":{}}"#
        );
    }

    #[test]
    fn the_spaced_style_writes_the_reference_separators() {
        let env = build_environment("{{ tool | tojson }}", ToolJsonStyle::Spaced)
            .expect("a one-expression template compiles");
        let tool = serde_json::json!({"name": "b", "args": [1, 2]});
        let rendered = env
            .get_template(TEMPLATE_NAME)
            .expect("the template is registered")
            .render(minijinja::context! { tool => minijinja::Value::from_serialize(&tool) })
            .expect("the template renders");
        assert_eq!(rendered, r#"{"name": "b", "args": [1, 2]}"#);
    }

    #[test]
    fn a_template_that_does_not_parse_is_refused_rather_than_rendered() {
        assert!(build_environment("{% for %}", ToolJsonStyle::Compact).is_err());
    }

    /// The Python-shaped method a template calls on a string reaches the filter
    /// of the same name.
    #[test]
    fn a_python_style_string_method_is_bridged_to_its_filter() {
        let env = build_environment("{{ text.replace('a', 'b') }}", ToolJsonStyle::Compact)
            .expect("the template compiles");
        let rendered = env
            .get_template(TEMPLATE_NAME)
            .expect("the template is registered")
            .render(minijinja::context! { text => "banana" })
            .expect("the template renders");
        assert_eq!(rendered, "bbnbnb");
    }
}

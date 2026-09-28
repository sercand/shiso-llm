//! What a Qwen 3.5 tool call must say next, from the tools a request declared.
//!
//! The XML call body is mostly fixed text: the opening `<function=`, a name
//! that is one of the declared tools, `<parameter=` and a name that is one of
//! that tool's parameters, and the closers. Only the values are free. A
//! speculative decoder that is told the fixed parts can verify them in one
//! pass instead of drafting them, and a name the drafter has never seen - the
//! tools are the request's, not the training data's - is exactly what it would
//! get wrong.
//!
//! [`ToolScaffold::next`] answers from the text the turn has produced so far,
//! and answers only where the answer is certain: a name prefix that more than
//! one declared name continues, or a value, is `None`. Being wrong costs a
//! verify row and nothing else - the proposal is checked against the model's
//! own choices - so the rule is "certain or silent", not "likely".

/// The declared tools, as the names the call body can spell.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolScaffold {
    /// Each tool's name and its parameter names, in declaration order.
    tools: Vec<(String, Vec<String>)>,
}

/// The call's opening and closing tags.
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

impl ToolScaffold {
    /// Reads `(tool name, JSON schema)` pairs; a tool's parameters are its
    /// schema's `properties`, in the order the schema lists them.
    #[must_use]
    pub fn from_schemas<'a>(
        tools: impl IntoIterator<Item = (&'a str, &'a serde_json::Value)>,
    ) -> Self {
        let tools = tools
            .into_iter()
            .map(|(name, schema)| {
                let params = schema
                    .get("properties")
                    .and_then(serde_json::Value::as_object)
                    .map(|properties| properties.keys().cloned().collect())
                    .unwrap_or_default();
                (name.to_owned(), params)
            })
            .collect();
        Self { tools }
    }

    /// Whether the request declared no tools, so nothing is ever proposed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The text that must follow `text`, the turn's output so far, or `None`
    /// where what follows is not fixed.
    ///
    /// Inside an open `<tool_call>`, at the points where the body is fixed:
    /// the function tag after the opening; the rest of a function or
    /// parameter name one declared name alone continues, with its closer;
    /// `\n<` after a parameter's close (a next parameter or the function's
    /// close, which share it), or the whole close once every parameter of the
    /// function has been written; and the call's close after the function's.
    #[must_use]
    pub fn next(&self, text: &str) -> Option<String> {
        if self.tools.is_empty() {
            return None;
        }
        let open = text.rfind(CALL_OPEN)?;
        let body = &text[open + CALL_OPEN.len()..];
        if body.contains(CALL_CLOSE) {
            return None;
        }
        // After the opening: the function tag, and the name when only one
        // tool was declared.
        if body.is_empty() || body == "\n" {
            let lead = if body.is_empty() { "\n" } else { "" };
            let mut next = format!("{lead}{FUNCTION_OPEN}");
            if let [(name, params)] = self.tools.as_slice() {
                next.push_str(&Self::named(name, params.is_empty()));
            }
            return Some(next);
        }
        if body.ends_with(FUNCTION_CLOSE) {
            return Some(format!("\n{CALL_CLOSE}"));
        }
        let function_at = body.rfind(FUNCTION_OPEN)?;
        let after_function = &body[function_at + FUNCTION_OPEN.len()..];
        let Some(name_end) = after_function.find('>') else {
            // Inside the function's name.
            let (name, params) =
                Self::unique(self.tools.iter().map(|(n, p)| (n, p)), after_function)?;
            let rest = &name[after_function.len()..];
            return Some(format!("{rest}{}", Self::closer(params.is_empty())));
        };
        let function = &after_function[..name_end];
        let params = &self.tools.iter().find(|(name, _)| name == function)?.1;
        let inside = &after_function[name_end..];
        if inside.ends_with(PARAMETER_CLOSE) {
            let written = inside.matches(PARAMETER_OPEN).count();
            return Some(if written >= params.len() {
                format!("\n{FUNCTION_CLOSE}\n{CALL_CLOSE}")
            } else {
                "\n<".to_owned()
            });
        }
        let parameter_at = inside.rfind(PARAMETER_OPEN)?;
        let after_parameter = &inside[parameter_at + PARAMETER_OPEN.len()..];
        if after_parameter.contains('>') {
            // Inside a value: free text.
            return None;
        }
        let (param, ()) = Self::unique(params.iter().map(|p| (p, ())), after_parameter)?;
        Some(format!("{}>\n", &param[after_parameter.len()..]))
    }

    /// A function's name, its closing `>` and what follows the name's line.
    fn named(name: &str, no_params: bool) -> String {
        format!("{name}{}", Self::closer(no_params))
    }

    /// What follows a function name: its line's end, then the first
    /// parameter's tag or, for a function with none, the function's close.
    fn closer(no_params: bool) -> String {
        if no_params {
            format!(">\n{FUNCTION_CLOSE}")
        } else {
            format!(">\n{PARAMETER_OPEN}")
        }
    }

    /// The one candidate `prefix` begins, or `None` when there are none or
    /// several.
    fn unique<'a, T>(
        candidates: impl Iterator<Item = (&'a String, T)>,
        prefix: &str,
    ) -> Option<(&'a String, T)> {
        let mut found = None;
        for (name, extra) in candidates {
            if name.starts_with(prefix) {
                if found.is_some() {
                    return None;
                }
                found = Some((name, extra));
            }
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::ToolScaffold;
    use serde_json::json;

    fn scaffold() -> ToolScaffold {
        let search = json!({"properties": {"query": {}, "path": {}}});
        let read = json!({"properties": {"path": {}}});
        let now = json!({"properties": {}});
        ToolScaffold::from_schemas([
            ("search_files", &search),
            ("read_file", &read),
            ("current_time", &now),
        ])
    }

    #[test]
    fn the_function_tag_follows_the_opening() {
        let s = scaffold();
        assert_eq!(
            s.next("I will look.\n<tool_call>").as_deref(),
            Some("\n<function=")
        );
        assert_eq!(s.next("<tool_call>\n").as_deref(), Some("<function="));
    }

    #[test]
    fn a_single_tool_is_named_at_once() {
        let only = json!({"properties": {"city": {}}});
        let s = ToolScaffold::from_schemas([("get_weather", &only)]);
        assert_eq!(
            s.next("<tool_call>").as_deref(),
            Some("\n<function=get_weather>\n<parameter=")
        );
    }

    #[test]
    fn a_name_prefix_one_tool_continues_is_completed() {
        let s = scaffold();
        assert_eq!(
            s.next("<tool_call>\n<function=sea").as_deref(),
            Some("rch_files>\n<parameter=")
        );
        assert_eq!(
            s.next("<tool_call>\n<function=cur").as_deref(),
            Some("rent_time>\n</function>")
        );
        // "re" could only be read_file; "" could be any of the three.
        assert_eq!(
            s.next("<tool_call>\n<function=re").as_deref(),
            Some("ad_file>\n<parameter=")
        );
        assert_eq!(s.next("<tool_call>\n<function="), None);
    }

    #[test]
    fn a_parameter_prefix_of_the_open_function_is_completed() {
        let s = scaffold();
        let open = "<tool_call>\n<function=search_files>\n<parameter=";
        assert_eq!(s.next(&format!("{open}q")).as_deref(), Some("uery>\n"));
        assert_eq!(s.next(open), None);
        // Inside a value nothing is fixed.
        assert_eq!(s.next(&format!("{open}query>\nfoo")), None);
    }

    #[test]
    fn the_closers_follow_the_last_parameter() {
        let s = scaffold();
        let one = "<tool_call>\n<function=search_files>\n<parameter=query>\nx\n</parameter>";
        assert_eq!(s.next(one).as_deref(), Some("\n<"));
        let both = format!("{one}\n<parameter=path>\n.\n</parameter>");
        assert_eq!(
            s.next(&both).as_deref(),
            Some("\n</function>\n</tool_call>")
        );
        assert_eq!(
            s.next("<tool_call>\n<function=current_time>\n</function>")
                .as_deref(),
            Some("\n</tool_call>")
        );
    }

    #[test]
    fn nothing_is_proposed_outside_a_call_or_for_an_unknown_function() {
        let s = scaffold();
        assert_eq!(s.next("plain prose"), None);
        assert_eq!(
            s.next("<tool_call>\n<function=x>\n</function>\n</tool_call> done"),
            None
        );
        assert_eq!(s.next("<tool_call>\n<function=nope>\n<parameter=a"), None);
        assert_eq!(ToolScaffold::default().next("<tool_call>"), None);
    }
}

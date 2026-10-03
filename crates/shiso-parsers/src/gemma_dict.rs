//! Gemma 4 dictionary to JSON converter.
//!
//! Gemma 4 formats tool call arguments as a dictionary syntax:
//! `{key1:value1,key2:value2}` where string values are quoted with `<|"|>` … `<|"|>`.
//! This module parses that grammar and converts it to standard JSON text.

/// The quote a Gemma 4 call body spells every string value between.
pub const STRING_QUOTE: &str = "<|\"|>";

/// An error that occurred while converting a Gemma dictionary to JSON.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("gemma dict parse error at byte offset {offset}: {message}")]
pub struct GemmaDictError {
    /// Byte offset into the input where parsing failed.
    pub offset: usize,
    /// Human-readable description of the error.
    pub message: String,
}

/// Converts Gemma's call body `{k:v,...}` to a JSON object text.
///
/// # Errors
/// Returns [`GemmaDictError`] naming the byte offset where the body stopped parsing.
pub fn gemma_dict_to_json(body: &str) -> Result<String, GemmaDictError> {
    let mut parser = Parser::new(body);
    let json = parser.parse_dict()?;
    parser.skip_whitespace();
    if parser.pos != parser.input.len() {
        return Err(parser.error("unexpected trailing content"));
    }
    Ok(json)
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn error(&self, message: &str) -> GemmaDictError {
        Self::error_at(self.pos, message)
    }

    fn error_at(offset: usize, message: &str) -> GemmaDictError {
        GemmaDictError {
            offset,
            message: message.to_string(),
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if let Some(c) = self.peek()
            && c == expected
        {
            self.pos += c.len_utf8();
            true
        } else {
            false
        }
    }

    fn consume_literal(&mut self, lit: &str) -> bool {
        if self.input[self.pos..].starts_with(lit) {
            let next_idx = self.pos + lit.len();
            if let Some(next_char) = self.input[next_idx..].chars().next()
                && (next_char.is_alphanumeric() || next_char == '_')
            {
                return false;
            }
            self.pos += lit.len();
            true
        } else {
            false
        }
    }

    fn parse_dict(&mut self) -> Result<String, GemmaDictError> {
        self.skip_whitespace();
        if !self.consume_char('{') {
            return Err(self.error("expected '{'"));
        }
        self.skip_whitespace();
        if self.consume_char('}') {
            return Ok("{}".to_string());
        }

        let mut out = String::from("{");
        let mut first = true;

        loop {
            self.skip_whitespace();
            let key_start = self.pos;
            while let Some(c) = self.peek() {
                if c == ':' || c == '}' {
                    break;
                }
                self.pos += c.len_utf8();
            }
            let raw_key = self.input[key_start..self.pos].trim();
            if raw_key.is_empty() {
                return Err(Self::error_at(key_start, "expected key name"));
            }

            self.skip_whitespace();
            if !self.consume_char(':') {
                return Err(self.error("expected ':' after key"));
            }

            self.skip_whitespace();
            let val = self.parse_value()?;

            if !first {
                out.push(',');
            }
            first = false;

            let key_json = serde_json::to_string(raw_key)
                .map_err(|e| Self::error_at(key_start, &e.to_string()))?;
            out.push_str(&key_json);
            out.push(':');
            out.push_str(&val);

            self.skip_whitespace();
            if self.consume_char(',') {
                self.skip_whitespace();
                if self.consume_char('}') {
                    break;
                }
            } else if self.consume_char('}') {
                break;
            } else {
                return Err(self.error("expected ',' or '}' in dict"));
            }
        }

        out.push('}');
        Ok(out)
    }

    fn parse_value(&mut self) -> Result<String, GemmaDictError> {
        self.skip_whitespace();

        // 1. String: <|"|>...<|"|>
        if self.input[self.pos..].starts_with(STRING_QUOTE) {
            let quote = STRING_QUOTE;
            self.pos += quote.len();
            let content_start = self.pos;
            let rest = &self.input[self.pos..];
            if let Some(end) = rest.find(quote) {
                let content = &rest[..end];
                self.pos += end + quote.len();
                let json_str = serde_json::to_string(content)
                    .map_err(|e| Self::error_at(content_start, &e.to_string()))?;
                return Ok(json_str);
            }
            return Err(self.error("unterminated string: missing '<|\"|>'"));
        }

        // 2. Nested dict: { ... }
        if self.input[self.pos..].starts_with('{') {
            return self.parse_dict();
        }

        // 3. Array: [ ... ]
        if self.consume_char('[') {
            self.skip_whitespace();
            if self.consume_char(']') {
                return Ok("[]".to_string());
            }

            let mut out = String::from("[");
            let mut first = true;

            loop {
                self.skip_whitespace();
                let val = self.parse_value()?;

                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&val);

                self.skip_whitespace();
                if self.consume_char(',') {
                    self.skip_whitespace();
                    if self.consume_char(']') {
                        break;
                    }
                } else if self.consume_char(']') {
                    break;
                } else {
                    return Err(self.error("expected ',' or ']' in array"));
                }
            }

            out.push(']');
            return Ok(out);
        }

        // 4. Literals: true, false, null
        if self.consume_literal("true") {
            return Ok("true".to_string());
        }
        if self.consume_literal("false") {
            return Ok("false".to_string());
        }
        if self.consume_literal("null") {
            return Ok("null".to_string());
        }

        // 5. Numbers
        let num_start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '+' {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        let num_str = &self.input[num_start..self.pos];
        if !num_str.is_empty() && serde_json::from_str::<serde_json::Number>(num_str).is_ok() {
            return Ok(num_str.to_string());
        }

        Err(Self::error_at(
            num_start,
            "expected valid value (string, dict, array, boolean, null, or number)",
        ))
    }
}

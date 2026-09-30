//! Just enough `multipart/form-data` to route a form-encoded request:
//! `/v1/audio/transcriptions` carries `model` (and `stream`) as form fields
//! next to the audio file rather than in a JSON body.
//!
//! This is a probe, not a decoder. It finds the two text fields routing
//! needs and remembers where the `model` value sits, so a rename can splice
//! the new name into the original bytes. Every other part — the audio above
//! all — is forwarded byte for byte, the same way a JSON body with no edits
//! is. Splicing is safe in a way it is not for JSON: a part's extent is
//! fixed by the boundary, which the client chose so as not to occur in any
//! part, so there is no escaping to get wrong.

use std::ops::Range;

use bytes::{Bytes, BytesMut};

/// The routing-relevant fields of a form body.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct FormProbe {
    pub model: Option<String>,
    /// Byte range of the `model` field's value within the body.
    pub model_span: Option<Range<usize>>,
    pub stream: Option<bool>,
}

/// The `boundary` parameter of a `multipart/form-data` content type, or
/// `None` for any other content type.
pub(crate) fn boundary(content_type: &str) -> Option<&str> {
    let mut params = content_type.split(';');
    let mime = params.next()?.trim();
    if !mime.eq_ignore_ascii_case("multipart/form-data") {
        return None;
    }
    params.find_map(|p| {
        let (k, v) = p.split_once('=')?;
        if !k.trim().eq_ignore_ascii_case("boundary") {
            return None;
        }
        let v = v.trim();
        let v = v
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(v);
        (!v.is_empty()).then_some(v)
    })
}

/// Scans the form for `model` and `stream`. `None` means the body is not a
/// well-formed form under this boundary. A field that occurs twice keeps
/// its first value.
pub(crate) fn probe(body: &[u8], boundary: &str) -> Option<FormProbe> {
    let delim = [b"--", boundary.as_bytes()].concat();
    // Every delimiter after the first is preceded by a CRLF that belongs to
    // it rather than to the part before.
    let next_delim = [b"\r\n", delim.as_slice()].concat();

    let mut out = FormProbe::default();
    // Anything before the first delimiter is preamble and ignored.
    let mut pos = find(body, &delim, 0)? + delim.len();
    loop {
        let rest = body.get(pos..)?;
        if rest.starts_with(b"--") {
            return Some(out);
        }
        // RFC 2046 allows transport padding between the delimiter and its
        // line break.
        let pad = rest
            .iter()
            .take_while(|b| **b == b' ' || **b == b'\t')
            .count();
        if !rest[pad..].starts_with(b"\r\n") {
            return None;
        }
        let headers_start = pos + pad + 2;
        let headers_end = find(body, b"\r\n\r\n", headers_start)?;
        let content_start = headers_end + 4;
        let content_end = find(body, &next_delim, content_start)?;

        let headers = std::str::from_utf8(&body[headers_start..headers_end]).ok()?;
        match field_name(headers) {
            Some("model") if out.model.is_none() => {
                let value = std::str::from_utf8(&body[content_start..content_end]).ok()?;
                out.model = Some(value.to_string());
                out.model_span = Some(content_start..content_end);
            }
            Some("stream") if out.stream.is_none() => {
                let value = String::from_utf8_lossy(&body[content_start..content_end]);
                out.stream = Some(value.trim().eq_ignore_ascii_case("true"));
            }
            _ => {}
        }
        pos = content_end + next_delim.len();
    }
}

/// The body with `span` (a field value found by `probe`) replaced.
pub(crate) fn splice(body: &Bytes, span: Range<usize>, value: &str) -> Bytes {
    let mut out = BytesMut::with_capacity(body.len() - span.len() + value.len());
    out.extend_from_slice(&body[..span.start]);
    out.extend_from_slice(value.as_bytes());
    out.extend_from_slice(&body[span.end..]);
    out.freeze()
}

/// The `name` parameter of a part's `Content-Disposition: form-data`
/// header.
fn field_name(headers: &str) -> Option<&str> {
    headers.split("\r\n").find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if !name.trim().eq_ignore_ascii_case("content-disposition") {
            return None;
        }
        value.split(';').skip(1).find_map(|p| {
            let (k, v) = p.split_once('=')?;
            if !k.trim().eq_ignore_ascii_case("name") {
                return None;
            }
            let v = v.trim();
            Some(
                v.strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .unwrap_or(v),
            )
        })
    })
}

/// First occurrence of `needle` in `hay` at or after `from`. Scans for the
/// first byte and only then compares, which keeps a pass over a 25 MB
/// audio part cheap without pulling in a search crate.
fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    let (&first, tail) = needle.split_first()?;
    let mut i = from;
    while i + needle.len() <= hay.len() {
        let at = i + hay[i..=hay.len() - needle.len()]
            .iter()
            .position(|&b| b == first)?;
        if hay[at + 1..at + needle.len()] == *tail {
            return Some(at);
        }
        i = at + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(parts: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
        let mut out = b"preamble to ignore\r\n".to_vec();
        for (name, filename, value) in parts {
            out.extend_from_slice(b"--XyZ\r\n");
            match filename {
                Some(f) => out.extend_from_slice(
                    format!(
                        "Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\n\
                         Content-Type: audio/mpeg\r\n\r\n"
                    )
                    .as_bytes(),
                ),
                None => out.extend_from_slice(
                    format!("content-disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                ),
            }
            out.extend_from_slice(value);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"--XyZ--\r\n");
        out
    }

    #[test]
    fn boundary_is_read_from_the_content_type() {
        assert_eq!(boundary("multipart/form-data; boundary=abc"), Some("abc"));
        assert_eq!(
            boundary("Multipart/Form-Data; charset=utf-8; boundary=\"a b\""),
            Some("a b")
        );
        assert_eq!(boundary("application/json"), None);
        assert_eq!(boundary("multipart/form-data"), None);
    }

    #[test]
    fn finds_model_after_a_binary_file_part() {
        // Audio bytes that contain CRLFs, dashes and even a near-miss of
        // the delimiter must not end the part early.
        let audio: &[u8] = b"\x00\xff\r\n--XY\r\n-XyZ\n--XyZ\r\n\r\n\xfe";
        let body = form(&[
            ("file", Some("a.mp3"), audio),
            ("model", None, b"whisper-1"),
            ("stream", None, b"true"),
        ]);
        let p = probe(&body, "XyZ").unwrap();
        assert_eq!(p.model.as_deref(), Some("whisper-1"));
        assert_eq!(&body[p.model_span.unwrap()], b"whisper-1");
        assert_eq!(p.stream, Some(true));
    }

    #[test]
    fn a_form_without_model_or_stream_probes_empty() {
        let body = form(&[("file", Some("a.wav"), b"RIFF"), ("language", None, b"en")]);
        assert_eq!(probe(&body, "XyZ"), Some(FormProbe::default()));
    }

    #[test]
    fn malformed_forms_are_rejected() {
        assert_eq!(probe(b"{\"model\":\"m\"}", "XyZ"), None);
        // Unterminated: no closing delimiter after the value.
        assert_eq!(
            probe(
                b"--XyZ\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm",
                "XyZ"
            ),
            None
        );
    }

    #[test]
    fn splice_replaces_only_the_value() {
        let body = Bytes::from(form(&[
            ("model", None, b"alias"),
            ("file", Some("a.mp3"), b"model alias"),
        ]));
        let span = probe(&body, "XyZ").unwrap().model_span.unwrap();
        let out = splice(&body, span, "gpt-4o-transcribe");
        let p = probe(&out, "XyZ").unwrap();
        assert_eq!(p.model.as_deref(), Some("gpt-4o-transcribe"));
        // The file part is untouched, including its own "model alias" text.
        assert!(out.windows(11).any(|w| w == b"model alias"));
        assert_eq!(
            out.len(),
            body.len() - "alias".len() + "gpt-4o-transcribe".len()
        );
    }
}

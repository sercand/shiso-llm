//! Serialization formats for `/embeddings` and `/v1/embeddings`.
//!
//! The request is the shape `OpenAI` and `OpenRouter` agree on, widened to
//! `OpenRouter`'s multimodal inputs: `input` is a string, a list of strings,
//! a list of token ids, a list of such lists, or a list of `{ "content":
//! [parts] }` objects whose parts are text, pictures, video and audio. A list
//! may mix plain strings and content objects. Like every wire module here, this
//! one owns its serde types and converts them to the IR in
//! [`crate::openai::embeddings`]; nothing past that knows these names.
//!
//! Unknown fields are accepted, as on every other surface here: `provider`,
//! `trace`, `session_id` and `user` are routing and accounting fields of the
//! services that define them, and refusing one would turn a working request
//! into a parse error.

use serde::{Deserialize, Serialize};

/// Request payload of `POST /v1/embeddings`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    /// The model the caller names. Answered by whichever embedder is loaded,
    /// as the image route answers: a client that names the model it believes
    /// is behind this server is not refused for being right about nothing.
    #[serde(default)]
    pub model: Option<String>,
    /// What to embed.
    pub input: EmbeddingInput,
    /// The width to cut each vector to; one of the model's trained widths.
    #[serde(default)]
    pub dimensions: Option<u32>,
    /// `float` (the default) or `base64`.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// The task the inputs serve, selecting the prompt the model was trained
    /// with for it: `search_query`, `search_document`, or any prompt name the
    /// checkpoint ships (`Classification`, `CodeRetrieval`, ...).
    #[serde(default)]
    pub input_type: Option<String>,
}

/// Every spelling `input` takes.
///
/// Untagged, and the order is the order serde tries them: a lone string, then
/// one list of ids, then a list of id lists, then a list of items. An array of
/// strings falls through the two numeric arms to the last, where a string is
/// an item.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// One text.
    One(String),
    /// One input as token ids.
    Tokens(Vec<u32>),
    /// Several inputs as token ids.
    TokenLists(Vec<Vec<u32>>),
    /// Several inputs, each a text or a content list.
    Items(Vec<InputItem>),
}

/// One input of a list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    /// A text.
    Text(String),
    /// Text and media, in reading order, embedded together as one vector.
    Content {
        /// The parts.
        content: Vec<Part>,
    },
}

/// One part of a multimodal input.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    /// Prose.
    Text {
        /// The words.
        text: String,
    },
    /// A picture, by address or data URI.
    ImageUrl {
        /// Where the bytes are.
        image_url: UrlRef,
    },
    /// A video, by address or data URI: this server's chat spelling.
    VideoUrl {
        /// Where the bytes are.
        video_url: UrlRef,
    },
    /// A video, inline: `OpenRouter`'s spelling.
    InputVideo {
        /// The clip.
        input_video: Inline,
    },
    /// A sound clip, inline.
    InputAudio {
        /// The clip.
        input_audio: Inline,
    },
    /// A document. Accepted by the format and refused by this server, which
    /// has no document encoder: the refusal names the part rather than
    /// failing the parse.
    InputFile {
        /// The document.
        input_file: Inline,
    },
}

/// An address or a data URI.
#[derive(Debug, Clone, Deserialize)]
pub struct UrlRef {
    /// `https://...` or `data:<mime>;base64,...`.
    pub url: String,
}

/// Bytes carried in the request.
///
/// `format` is the client's word for the container and is read by nobody:
/// the decoders sniff the bytes, as the chat route's do.
#[derive(Debug, Clone, Deserialize)]
pub struct Inline {
    /// Base64, with or without a `data:` prefix.
    pub data: String,
    /// The container the client says it is.
    #[serde(default)]
    pub format: Option<String>,
}

/// Response payload.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    /// Always `list`.
    pub object: &'static str,
    /// One entry per input, in input order.
    pub data: Vec<EmbeddingData>,
    /// The model that answered.
    pub model: String,
    /// Positions the inputs occupied.
    pub usage: EmbeddingUsage,
}

/// One vector.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingData {
    /// Always `embedding`.
    pub object: &'static str,
    /// The input's position in the request.
    pub index: usize,
    /// The vector, as the request's encoding wrote it.
    pub embedding: Vector,
}

/// A vector as numbers or as base64 of its little-endian `f32` bytes.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Vector {
    /// `encoding_format: float`.
    Float(Vec<f32>),
    /// `encoding_format: base64`.
    Base64(String),
}

/// Token accounting.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingUsage {
    /// Every position the inputs occupied.
    pub prompt_tokens: u32,
    /// The same: an embedding generates nothing.
    pub total_tokens: u32,
    /// Positions by modality, only when two or more contributed - the
    /// `OpenRouter` rule, so a text-only answer keeps the plain `OpenAI` shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<ModalityTokens>,
}

/// Positions by modality; a zero count is left out.
#[derive(Debug, Clone, Serialize)]
pub struct ModalityTokens {
    /// Text tokens, including the model's markers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u32>,
    /// Picture soft tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<u32>,
    /// Video-frame soft tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_tokens: Option<u32>,
    /// Audio soft tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> EmbeddingRequest {
        serde_json::from_str(body).expect("parses")
    }

    #[test]
    fn every_input_spelling_parses_to_its_arm() {
        assert!(matches!(
            parse(r#"{"input": "hello"}"#).input,
            EmbeddingInput::One(_)
        ));
        assert!(matches!(
            parse(r#"{"input": [1, 2, 3]}"#).input,
            EmbeddingInput::Tokens(_)
        ));
        assert!(matches!(
            parse(r#"{"input": [[1, 2], [3]]}"#).input,
            EmbeddingInput::TokenLists(_)
        ));
        let EmbeddingInput::Items(items) = parse(r#"{"input": ["a", "b"]}"#).input else {
            panic!("a list of strings is a list of items");
        };
        assert!(matches!(items[0], InputItem::Text(_)));
    }

    #[test]
    fn the_openrouter_multimodal_body_parses() {
        let body = r#"{
            "model": "google/embeddinggemma-2",
            "input": [
                {"content": [
                    {"type": "text", "text": "A scenic boardwalk"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "input_audio", "input_audio": {"data": "UklG", "format": "wav"}},
                    {"type": "input_video", "input_video": {"data": "AAAA", "format": "mp4"}}
                ]},
                "plain text"
            ],
            "encoding_format": "float",
            "input_type": "search_query",
            "provider": {"allow_fallbacks": true}
        }"#;
        let parsed = parse(body);
        let EmbeddingInput::Items(items) = parsed.input else {
            panic!("a list of items");
        };
        let InputItem::Content { content } = &items[0] else {
            panic!("a content item");
        };
        assert_eq!(content.len(), 4);
        assert!(matches!(content[3], Part::InputVideo { .. }));
        assert!(matches!(items[1], InputItem::Text(_)));
        assert_eq!(parsed.input_type.as_deref(), Some("search_query"));
    }
}

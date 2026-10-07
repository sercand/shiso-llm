//! `POST /embeddings`, `POST /v1/embeddings`.
//!
//! The wire body becomes an [`EmbedRequest`] here and nowhere else, and the
//! service's [`EmbedResponse`] becomes the wire answer here too: the service
//! knows the IR, not these names. A picture is a [`Block::Image`] and a clip
//! a [`Block::Audio`] exactly as in a chat turn, so the decode path a chat
//! request takes is the one an embedding input takes.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse as _, Response};
use base64::Engine as _;
use shiso_ir::{Block, EmbedEncoding, EmbedInput, EmbedRequest, EmbedResponse, MediaSource};

use crate::error::AppError;
use crate::openai::embeddings_wire::{
    EmbeddingData, EmbeddingInput, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, InputItem,
    ModalityTokens, Part, Vector,
};
use crate::state::AppState;

/// The handler for both spellings of the route.
///
/// The body is taken as bytes and parsed here, the chat route's way, so a
/// malformed body answers in this surface's error shape.
pub async fn create_embeddings(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    match answer(&state, &body).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn answer(state: &AppState, body: &Bytes) -> Result<EmbeddingResponse, AppError> {
    let parsed: EmbeddingRequest = serde_json::from_slice(body).map_err(|error| {
        AppError::bad_request(
            format!("the request body is not an embeddings request: {error}"),
            None,
            "invalid_request",
        )
    })?;
    // Refused before the model question: a body this server will never act
    // on is the caller's fault whether or not an embedder is loaded.
    let request = to_ir(parsed)?;
    let service = state.embedding()?;
    if request.model != service.model_id() && !request.model.is_empty() {
        tracing::debug!(
            model = request.model,
            served = service.model_id(),
            "the named model is answered by whichever embedder is loaded"
        );
    }
    Ok(encode(service.embed(request).await?))
}

/// The wire request as the IR.
fn to_ir(wire: EmbeddingRequest) -> Result<EmbedRequest, AppError> {
    let encoding = match wire.encoding_format.as_deref() {
        None | Some("float") => EmbedEncoding::Float,
        Some("base64") => EmbedEncoding::Base64,
        Some(other) => {
            return Err(AppError::bad_request(
                format!("encoding_format {other:?} is not one of \"float\", \"base64\""),
                Some("encoding_format"),
                "invalid_request",
            ));
        }
    };
    let text = |t: String| EmbedInput::Content(vec![Block::Text(t)]);
    let inputs = match wire.input {
        EmbeddingInput::One(t) => vec![text(t)],
        // `[]` parses as an empty id list; it is an empty input, not an
        // input of no tokens.
        EmbeddingInput::Tokens(ids) if ids.is_empty() => Vec::new(),
        EmbeddingInput::Tokens(ids) => vec![EmbedInput::Tokens(ids)],
        EmbeddingInput::TokenLists(lists) => lists.into_iter().map(EmbedInput::Tokens).collect(),
        EmbeddingInput::Items(items) => items
            .into_iter()
            .map(|item| match item {
                InputItem::Text(t) => Ok(text(t)),
                InputItem::Content { content } => content
                    .into_iter()
                    .map(block)
                    .collect::<Result<Vec<_>, _>>()
                    .map(EmbedInput::Content),
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    if wire.dimensions == Some(0) {
        return Err(AppError::bad_request(
            "dimensions must be at least 1",
            Some("dimensions"),
            "invalid_request",
        ));
    }
    Ok(EmbedRequest {
        model: wire.model.unwrap_or_default(),
        inputs,
        dimensions: wire.dimensions,
        encoding,
        task: wire.input_type,
    })
}

/// One wire part as a content block.
fn block(part: Part) -> Result<Block, AppError> {
    Ok(match part {
        Part::Text { text } => Block::Text(text),
        Part::ImageUrl { image_url } => Block::Image(MediaSource::classify(image_url.url)),
        Part::VideoUrl { video_url } => Block::Video(MediaSource::classify(video_url.url)),
        Part::InputVideo { input_video } => Block::Video(MediaSource::classify(input_video.data)),
        Part::InputAudio { input_audio } => Block::Audio(MediaSource::Inline(input_audio.data)),
        Part::InputFile { .. } => {
            return Err(AppError::bad_request(
                "input_file parts are not embedded by this server: it has no document encoder",
                Some("input"),
                "invalid_request",
            ));
        }
    })
}

/// The IR answer as the wire body.
fn encode(response: EmbedResponse) -> EmbeddingResponse {
    let data = response
        .vectors
        .into_iter()
        .enumerate()
        .map(|(index, vector)| EmbeddingData {
            object: "embedding",
            index,
            embedding: match response.encoding {
                EmbedEncoding::Float => Vector::Float(vector),
                EmbedEncoding::Base64 => {
                    let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Vector::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
                }
            },
        })
        .collect();
    let usage = response.usage;
    let some = |n: u32| (n > 0).then_some(n);
    EmbeddingResponse {
        object: "list",
        data,
        model: response.model,
        usage: EmbeddingUsage {
            prompt_tokens: usage.total(),
            total_tokens: usage.total(),
            prompt_tokens_details: (usage.modalities() >= 2).then(|| ModalityTokens {
                text_tokens: some(usage.text),
                image_tokens: some(usage.image),
                video_tokens: some(usage.video),
                audio_tokens: some(usage.audio),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> Result<EmbedRequest, AppError> {
        to_ir(serde_json::from_str(body).expect("parses"))
    }

    #[test]
    fn a_list_of_strings_is_one_text_input_each() {
        let request = parse(r#"{"input": ["a", "b"], "dimensions": 256}"#).expect("valid");
        assert_eq!(request.inputs.len(), 2);
        assert_eq!(request.dimensions, Some(256));
        assert_eq!(
            request.inputs[0],
            EmbedInput::Content(vec![Block::Text("a".to_owned())])
        );
    }

    #[test]
    fn content_parts_keep_their_order_and_kind() {
        let request = parse(
            r#"{"input": [{"content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
            ]}]}"#,
        )
        .expect("valid");
        let EmbedInput::Content(blocks) = &request.inputs[0] else {
            panic!("content");
        };
        assert!(matches!(blocks[1], Block::Image(MediaSource::Inline(_))));
        assert!(matches!(blocks[2], Block::Image(MediaSource::Remote(_))));
    }

    #[test]
    fn a_file_part_and_an_unknown_encoding_are_refused() {
        assert!(
            parse(
                r#"{"input": [{"content": [{"type": "input_file", "input_file": {"data": "x"}}]}]}"#
            )
            .is_err()
        );
        assert!(parse(r#"{"input": "a", "encoding_format": "int8"}"#).is_err());
    }

    #[test]
    fn base64_is_the_little_endian_floats() {
        let body = encode(EmbedResponse {
            model: "m".to_owned(),
            vectors: vec![vec![1.0, -2.0]],
            encoding: EmbedEncoding::Base64,
            usage: shiso_ir::EmbedUsage {
                text: 3,
                ..Default::default()
            },
        });
        let Vector::Base64(text) = &body.data[0].embedding else {
            panic!("base64");
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(text)
            .expect("base64");
        assert_eq!(bytes, [1f32.to_le_bytes(), (-2f32).to_le_bytes()].concat());
        assert!(body.usage.prompt_tokens_details.is_none());
        assert_eq!(body.usage.prompt_tokens, 3);
    }
}

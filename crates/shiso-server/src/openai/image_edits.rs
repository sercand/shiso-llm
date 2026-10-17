//! `POST /v1/images/edits` and `POST /images/edits`: reference pictures and a
//! prompt in, an edited picture out.
//!
//! The body is a multipart form (`image` or `image[]`, repeatable) or the
//! newer JSON shape (`images: [{"image_url": "data:..."}]`); both become the
//! generation routes' own request plus the references, and the shared
//! [`crate::openai::images::run`] answers them, so a streamed or buffered edit
//! is the same response a generation is.
//!
//! What an edit refuses, before any queue slot or device work is spent: no
//! reference at all (that is `/v1/images/generations`), `n` other than 1 (the
//! references are one edit, and `n` stays the generations route's feature),
//! a `mask`, more references than the loaded model takes, and a model that
//! takes none.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{FromRequest as _, Multipart, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse as _, Response};

use crate::error::AppError;
use crate::openai::image_edits_wire::{EditRequest, Field};
use crate::state::AppState;

/// The handler for both spellings of the route.
pub async fn edit_images(State(state): State<Arc<AppState>>, request: Request) -> Response {
    match answer(&state, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// The handler, with its failures still typed.
async fn answer(state: &Arc<AppState>, request: Request) -> Result<Response, AppError> {
    let limit = state.image.as_ref().map_or(
        crate::image::DEFAULT_MAX_UPLOAD_BYTES,
        crate::image::ImageService::max_upload_bytes,
    );
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let edit = if content_type.starts_with("multipart/form-data") {
        let multipart = Multipart::from_request(request, state)
            .await
            .map_err(|error| AppError::bad_request(error.to_string(), None, "invalid_request"))?;
        EditRequest::from_fields(read_fields(multipart, limit).await?)?
    } else if content_type.starts_with("application/json") {
        let body = Bytes::from_request(request, state)
            .await
            .map_err(|error| body_error(&error.into_response(), limit))?;
        EditRequest::from_json(&body)?
    } else {
        return Err(AppError::bad_request(
            "an image edit is sent as multipart/form-data or application/json",
            None,
            "invalid_request",
        ));
    };
    if edit.references.is_empty() {
        return Err(AppError::bad_request(
            "an image edit needs at least one reference image in 'image'; \
             text-to-image is /v1/images/generations",
            Some("image"),
            "invalid_request",
        ));
    }
    if let Some(n) = edit.request.n
        && n != 1
    {
        return Err(AppError::bad_request(
            format!("n must be 1 for an edit, got {n}"),
            Some("n"),
            "invalid_request",
        ));
    }
    crate::openai::images::run(state, edit.request, edit.references).await
}

/// Reads every field of the form; the body limit surfaces here as a read
/// error, so a file too big is told apart from a malformed form.
async fn read_fields(mut multipart: Multipart, limit: usize) -> Result<Vec<Field>, AppError> {
    let bad = |error: axum::extract::multipart::MultipartError| {
        if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
            return AppError::TooLarge {
                what: "an image edit request (--image-max-upload-mb)",
                limit,
            };
        }
        AppError::bad_request(
            format!("the body is not a readable multipart form: {error}"),
            None,
            "invalid_request",
        )
    };
    let mut fields = Vec::new();
    while let Some(field) = multipart.next_field().await.map_err(bad)? {
        let name = field.name().unwrap_or_default().to_owned();
        let bytes = field.bytes().await.map_err(bad)?;
        fields.push(Field { name, bytes });
    }
    Ok(fields)
}

/// A failed body read as this surface's error: the limit by name, anything
/// else as a malformed request.
fn body_error(rejection: &Response, limit: usize) -> AppError {
    if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return AppError::TooLarge {
            what: "an image edit request (--image-max-upload-mb)",
            limit,
        };
    }
    AppError::bad_request(
        "the request body could not be read",
        None,
        "invalid_request",
    )
}

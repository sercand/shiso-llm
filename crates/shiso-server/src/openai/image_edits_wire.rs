//! The bodies `POST /v1/images/edits` takes: a multipart form, or the newer
//! JSON shape with `images: [{"image_url": "data:..."}]`.
//!
//! Both end as the same [`ImageGenerationRequest`] the generation routes
//! parse, plus the reference pictures, so every knob (`size`, `seed`,
//! `steps`, `output_format`, ...) means exactly what it means there and is
//! validated by the same code. This module owns the edit-only field names and
//! converts them toward that request; nothing past it knows they exist.

use axum::body::Bytes;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::Deserialize;

use crate::error::AppError;
use crate::openai::images_wire::ImageGenerationRequest;

/// One multipart field: its name and its raw bytes.
#[derive(Debug)]
pub struct Field {
    /// The form field's name.
    pub name: String,
    /// Its content.
    pub bytes: Bytes,
}

/// An edit request ready for the shared pipeline.
#[derive(Debug)]
pub struct EditRequest {
    /// The generation-shaped fields.
    pub request: ImageGenerationRequest,
    /// The reference pictures, in the order the caller sent them.
    pub references: Vec<Vec<u8>>,
}

/// The JSON body: the generation fields plus the references.
#[derive(Debug, Deserialize)]
struct JsonEdit {
    #[serde(flatten)]
    request: ImageGenerationRequest,
    /// The references, as the newer `OpenAI` shape spells them.
    #[serde(default)]
    images: Vec<JsonImage>,
    /// A mask, which this server cannot honor and so refuses.
    #[serde(default)]
    mask: Option<serde_json::Value>,
}

/// One entry of `images`.
#[derive(Debug, Deserialize)]
struct JsonImage {
    /// A `data:` URL carrying the picture.
    #[serde(default)]
    image_url: Option<String>,
    /// An uploaded-file id, which this server has no store to resolve.
    #[serde(default)]
    file_id: Option<String>,
}

fn invalid(message: impl Into<String>, param: &str) -> AppError {
    AppError::bad_request(message, Some(param), "invalid_request")
}

fn mask_refused() -> AppError {
    AppError::bad_request(
        "mask is not supported: this model edits from reference images and a prompt alone",
        Some("mask"),
        "unsupported_parameter",
    )
}

/// Reads the bytes of a `data:...;base64,` URL.
fn decode_data_url(url: &str, index: usize) -> Result<Vec<u8>, AppError> {
    let payload = url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(','))
        .filter(|(head, _)| head.ends_with(";base64"))
        .map(|(_, payload)| payload)
        .ok_or_else(|| {
            invalid(
                format!(
                    "images[{index}].image_url must be a base64 data: URL; this server fetches no remote pictures"
                ),
                "images",
            )
        })?;
    BASE64_STANDARD.decode(payload.trim()).map_err(|error| {
        invalid(
            format!("images[{index}].image_url is not valid base64: {error}"),
            "images",
        )
    })
}

/// One numeric text field. `None` means the form has no such field, which is
/// not a refusal; a field that is present but not a number is.
fn number<T: std::str::FromStr>(
    text: &std::collections::HashMap<String, String>,
    name: &str,
) -> Result<Option<T>, AppError> {
    text.get(name)
        .map(|raw| {
            raw.trim()
                .parse()
                .map_err(|_| invalid(format!("{name} must be a number, got '{raw}'"), name))
        })
        .transpose()
}

impl EditRequest {
    /// Parses the JSON shape.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for a body that is not an edit request, a
    /// reference that is not a base64 data URL, or a mask.
    pub fn from_json(body: &[u8]) -> Result<Self, AppError> {
        let parsed: JsonEdit = serde_json::from_slice(body).map_err(|error| {
            invalid(
                format!("the request body is not an image edit request: {error}"),
                "images",
            )
        })?;
        if parsed.mask.is_some() {
            return Err(mask_refused());
        }
        let mut references = Vec::with_capacity(parsed.images.len());
        for (index, image) in parsed.images.iter().enumerate() {
            if image.file_id.is_some() {
                return Err(AppError::bad_request(
                    "images[].file_id is not supported: send the picture as an image_url data: URL",
                    Some("images"),
                    "unsupported_parameter",
                ));
            }
            let url = image
                .image_url
                .as_deref()
                .ok_or_else(|| invalid(format!("images[{index}] has no image_url"), "images"))?;
            references.push(decode_data_url(url, index)?);
        }
        Ok(Self {
            request: parsed.request,
            references,
        })
    }

    /// Reads the multipart shape: `image` or `image[]` (repeatable) and the
    /// generation fields as text.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for a missing prompt, a field that does not
    /// parse, or a mask.
    pub fn from_fields(fields: Vec<Field>) -> Result<Self, AppError> {
        let mut references = Vec::new();
        let mut text = std::collections::HashMap::new();
        for field in fields {
            match field.name.as_str() {
                "image" | "image[]" => references.push(field.bytes.to_vec()),
                "mask" => return Err(mask_refused()),
                _ => {
                    let value = String::from_utf8(field.bytes.to_vec()).map_err(|_| {
                        invalid(format!("field '{}' is not text", field.name), "prompt")
                    })?;
                    text.insert(field.name, value);
                }
            }
        }
        let prompt = text
            .get("prompt")
            .cloned()
            .ok_or_else(|| invalid("the form has no 'prompt' field", "prompt"))?;
        let stream = text
            .get("stream")
            .map(|raw| match raw.trim().to_ascii_lowercase().as_str() {
                "true" | "1" => Ok(true),
                "false" | "0" => Ok(false),
                other => Err(invalid(
                    format!("stream must be true or false, got '{other}'"),
                    "stream",
                )),
            })
            .transpose()?;
        let request = ImageGenerationRequest {
            prompt,
            model: text.get("model").cloned(),
            n: number(&text, "n")?,
            size: text.get("size").cloned(),
            aspect_ratio: text.get("aspect_ratio").cloned(),
            quality: text.get("quality").cloned(),
            steps: number(&text, "steps")?,
            seed: number(&text, "seed")?,
            output_format: text.get("output_format").cloned(),
            response_format: text.get("response_format").cloned(),
            output_compression: number(&text, "output_compression")?,
            stream,
            user: text.get("user").cloned(),
            session_id: text.get("session_id").cloned(),
        };
        Ok(Self {
            request,
            references,
        })
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Bytes;
    use base64::Engine as _;
    use shiso_ir::image::{Edit, Generate};

    use super::{EditRequest, Field};

    fn field(name: &str, bytes: &[u8]) -> Field {
        Field {
            name: name.to_owned(),
            bytes: Bytes::copy_from_slice(bytes),
        }
    }

    fn ir(request: &EditRequest) -> Edit {
        Edit {
            generate: request.request.to_ir(),
            references: request.references.clone(),
        }
    }

    fn expected(prompt: &str, size: Option<&str>, references: Vec<Vec<u8>>) -> Edit {
        Edit {
            generate: Generate {
                prompt: prompt.to_owned(),
                n: None,
                size: size.map(str::to_owned),
                aspect_ratio: None,
                steps: None,
                seed: None,
                output_format: None,
                output_compression: None,
            },
            references,
        }
    }

    /// A form and a JSON body that say the same thing become the same IR.
    #[test]
    fn both_body_shapes_become_the_same_ir_edit() {
        let from_form = EditRequest::from_fields(vec![
            field("image", &[1, 2]),
            field("image[]", &[3]),
            field("prompt", b"make it blue"),
            field("size", b"512x512"),
        ])
        .expect("form parses");
        let url = |bytes: &[u8]| {
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(bytes)
            )
        };
        let body = serde_json::json!({
            "prompt": "make it blue",
            "size": "512x512",
            "images": [{"image_url": url(&[1, 2])}, {"image_url": url(&[3])}],
        });
        let from_json = EditRequest::from_json(body.to_string().as_bytes()).expect("json parses");
        let want = expected("make it blue", Some("512x512"), vec![vec![1, 2], vec![3]]);
        assert_eq!(ir(&from_form), want);
        assert_eq!(ir(&from_json), want);
    }

    #[test]
    fn a_form_number_travels_to_the_ir_typed() {
        let request = EditRequest::from_fields(vec![
            field("image", &[1]),
            field("prompt", b"p"),
            field("seed", b"5"),
            field("steps", b"7"),
        ])
        .expect("parses");
        let edit = ir(&request);
        assert_eq!(edit.generate.seed, Some(5));
        assert_eq!(edit.generate.steps, Some(7));
    }
}

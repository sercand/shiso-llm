//! The 3D generation wire: `POST /v1/3d/generations`.
//!
//! One picture in, one encoded `.ply` out. This module owns its serde types
//! and hands them to [`crate::splat::splat`], which converts toward the IR;
//! nothing between the parse and the encode knows these names exist.

pub mod generate;
pub mod splat_wire;

pub use shiso_runtime::splat::{SplatService, SplatServiceDesc};

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::splat_wire::{SplatCompletedEvent, SplatGenerationRequest, SplatGenerationUsage};
    use shiso_ir::splat::Mode;

    #[test]
    fn a_data_url_and_a_bare_base64_body_both_decode() {
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"data:image/png;base64,iVBORw0KGgo="}"#)
                .expect("parses");
        assert_eq!(
            request.into_ir().expect("decodes").image,
            b"\x89PNG\r\n\x1a\n"
        );
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"iVBORw0KGgo="}"#).expect("parses");
        assert_eq!(
            request.into_ir().expect("decodes").image,
            b"\x89PNG\r\n\x1a\n"
        );
    }

    #[test]
    fn an_unknown_mode_names_the_field() {
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"aGk=","mode":"cartoon"}"#).expect("parses");
        let err = request.into_ir().expect_err("refused");
        assert!(err.contains("mode"), "{err}");
    }

    #[test]
    fn every_omitted_knob_stays_none() {
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"aGk="}"#).expect("parses");
        let ir = request.into_ir().expect("decodes");
        assert_eq!((ir.steps, ir.seed, ir.num_gaussians), (None, None, None));
        assert_eq!(ir.mode, Mode::Synthetic);
    }

    #[test]
    fn the_mode_is_case_insensitive() {
        for (text, expected) in [("REAL", Mode::Real), ("Synthetic", Mode::Synthetic)] {
            let request: SplatGenerationRequest =
                serde_json::from_str(&format!(r#"{{"image":"aGk=","mode":"{text}"}}"#))
                    .expect("parses");
            assert_eq!(request.into_ir().expect("decodes").mode, expected);
        }
    }

    #[test]
    fn a_body_that_is_not_base64_is_refused_by_name() {
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"not base64!!"}"#).expect("parses");
        let err = request.into_ir().expect_err("refused");
        assert!(err.contains("base64"), "{err}");
    }

    #[test]
    fn the_completed_frame_matches_the_spec_shape() {
        let json = serde_json::to_value(SplatCompletedEvent {
            r#type: "splat_generation.completed",
            created: 7,
            b64_ply: "cGx5".to_owned(),
            media_type: Some("model/ply".to_owned()),
            usage: Some(SplatGenerationUsage {
                prompt_tokens: 0,
                completion_tokens: 614_700,
                total_tokens: 614_700,
            }),
        })
        .expect("serializes");
        assert_eq!(json["type"], "splat_generation.completed");
        assert_eq!(json["usage"]["total_tokens"], 614_700);
    }

    #[test]
    fn the_response_carries_the_splat_and_its_accounting() {
        let json = json!({
            "created": 7,
            "data": [{"b64_ply": "cGx5"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 614_700, "total_tokens": 614_700}
        });
        assert_eq!(json["data"][0]["b64_ply"], "cGx5");
    }
}

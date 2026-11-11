//! The 3D generation wire: `POST /v1/3d/generations`.
//!
//! One picture in, one encoded `.ply` or `.glb` out. This module owns its serde types
//! and hands them to [`crate::splat::generate`], which converts toward the IR;
//! nothing between the parse and the encode knows these names exist.

pub mod generate;
pub mod splat_wire;

pub use shiso_runtime::splat::{SplatService, SplatServiceDesc};

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::splat_wire::{
        SplatCompletedEvent, SplatData, SplatGenerationRequest, SplatGenerationUsage,
        accepted_format,
    };
    use shiso_ir::splat::{Format, Mode};

    #[test]
    fn a_data_url_and_a_bare_base64_body_both_decode() {
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"data:image/png;base64,iVBORw0KGgo="}"#)
                .expect("parses");
        assert_eq!(
            request.into_ir(None).expect("decodes").image,
            b"\x89PNG\r\n\x1a\n"
        );
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"iVBORw0KGgo="}"#).expect("parses");
        assert_eq!(
            request.into_ir(None).expect("decodes").image,
            b"\x89PNG\r\n\x1a\n"
        );
    }

    #[test]
    fn an_unknown_mode_names_the_field() {
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"aGk=","mode":"cartoon"}"#).expect("parses");
        let err = request.into_ir(None).expect_err("refused");
        assert_eq!(err.field, "mode");
        assert!(err.message.contains("synthetic"), "{}", err.message);
    }

    #[test]
    fn every_omitted_knob_stays_none() {
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"aGk="}"#).expect("parses");
        let ir = request.into_ir(None).expect("decodes");
        assert_eq!((ir.steps, ir.seed, ir.num_gaussians), (None, None, None));
        assert_eq!(ir.mode, Mode::Synthetic);
    }

    #[test]
    fn the_mode_is_case_insensitive() {
        for (text, expected) in [("REAL", Mode::Real), ("Synthetic", Mode::Synthetic)] {
            let request: SplatGenerationRequest =
                serde_json::from_str(&format!(r#"{{"image":"aGk=","mode":"{text}"}}"#))
                    .expect("parses");
            assert_eq!(request.into_ir(None).expect("decodes").mode, expected);
        }
    }

    #[test]
    fn a_body_that_is_not_base64_is_refused_by_name() {
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"not base64!!"}"#).expect("parses");
        let err = request.into_ir(None).expect_err("refused");
        assert_eq!(err.field, "image");
        assert!(err.message.contains("base64"), "{}", err.message);
    }

    #[test]
    fn the_completed_frame_matches_the_spec_shape() {
        let json = serde_json::to_value(SplatCompletedEvent {
            r#type: "splat_generation.completed",
            created: 7,
            file: SplatData::encode(b"ply", "model/ply"),
            usage: SplatGenerationUsage {
                prompt_tokens: 0,
                completion_tokens: 614_700,
                total_tokens: 614_700,
            },
        })
        .expect("serializes");
        assert_eq!(json["type"], "splat_generation.completed");
        assert_eq!(json["b64_ply"], "cGx5");
        assert!(json.get("b64_glb").is_none());
        assert_eq!(json["usage"]["total_tokens"], 614_700);
    }

    #[test]
    fn the_format_defaults_to_ply_and_either_side_may_name_it() {
        let parse =
            |body: &str| -> SplatGenerationRequest { serde_json::from_str(body).expect("parses") };
        let plain = r#"{"image":"aGk="}"#;
        assert_eq!(parse(plain).into_ir(None).expect("ok").format, Format::Ply);
        assert_eq!(
            parse(plain).into_ir(Some(Format::Glb)).expect("ok").format,
            Format::Glb
        );
        let glb = r#"{"image":"aGk=","output_format":"GLB"}"#;
        assert_eq!(parse(glb).into_ir(None).expect("ok").format, Format::Glb);
        assert_eq!(
            parse(glb).into_ir(Some(Format::Glb)).expect("ok").format,
            Format::Glb
        );
    }

    #[test]
    fn a_body_and_a_header_that_disagree_are_refused() {
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"aGk=","output_format":"ply"}"#).expect("parses");
        let err = request.into_ir(Some(Format::Glb)).expect_err("refused");
        assert_eq!(err.field, "output_format");
        assert!(err.message.contains("model/gltf-binary"), "{}", err.message);
        let request: SplatGenerationRequest =
            serde_json::from_str(r#"{"image":"aGk=","output_format":"obj"}"#).expect("parses");
        assert_eq!(
            request.into_ir(None).expect_err("refused").field,
            "output_format"
        );
    }

    #[test]
    fn the_accept_header_names_a_format_only_by_a_splat_type() {
        assert_eq!(accepted_format("model/gltf-binary"), Some(Format::Glb));
        assert_eq!(
            accepted_format("application/json;q=0.5, Model/GLTF-Binary;q=0.9"),
            Some(Format::Glb)
        );
        assert_eq!(
            accepted_format("model/ply, model/gltf-binary"),
            Some(Format::Ply)
        );
        assert_eq!(accepted_format("*/*"), None);
        assert_eq!(accepted_format("application/json"), None);
        assert_eq!(accepted_format("model/gltf+json"), None);
    }

    #[test]
    fn a_glb_is_carried_under_its_own_field() {
        let json = serde_json::to_value(SplatData::encode(b"glTF", "model/gltf-binary"))
            .expect("serializes");
        assert_eq!(json["b64_glb"], "Z2xURg==");
        assert!(json.get("b64_ply").is_none());
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

//! The contract checks, on documents built in memory.

use super::*;

/// A valid version-1 document over widths 1024 and 2048.
fn raw() -> Raw {
    Raw {
        version: Some(1),
        block: Some(1024),
        transform: Some(TRANSFORM.into()),
        axis: Some(AXIS.into()),
        sign_mode: Some(SIGN_MODE.into()),
        forward: vec!["blk.0.ffn_up.weight".into(), "blk.0.ssm_out.weight".into()],
        inverse: vec!["token_embd.weight".into()],
        widths: vec![1024, 2048],
        values: (0..3072)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect(),
        grouped: Some(true),
    }
}

/// The identity rename: these documents already use this family's names.
#[expect(
    clippy::unnecessary_wraps,
    reason = "it stands in for a rename that can fail"
)]
fn same(name: &str) -> Option<String> {
    Some(name.to_owned())
}

/// One edit that takes a valid document out of version 1.
type Edit = Box<dyn Fn(&mut Raw)>;

#[test]
fn a_valid_document_is_read() {
    let plan = RotationPlan::check(raw(), same).expect("valid");
    assert_eq!(plan.block(), 1024);
    assert_eq!(plan.counts(), (2, 1));
    assert_eq!(plan.role("blk.0.ssm_out.weight"), Some(Rotated::Forward));
    assert_eq!(plan.role("token_embd.weight"), Some(Rotated::Inverse));
    assert_eq!(plan.role("blk.0.ssm_alpha.weight"), None);
    assert_eq!(plan.signs(2048).expect("width").len(), 2048);
    assert_eq!(plan.widest(), 2048);
    let heads = plan
        .value_heads(128, 16, 48)
        .expect("grouped")
        .expect("permuted");
    assert_eq!((heads.head_dim, heads.key_heads, heads.group), (128, 16, 3));
    assert_eq!(plan.value_heads(128, 16, 16).expect("equal"), None);
}

#[test]
fn every_departure_from_version_one_is_refused() {
    let cases: Vec<(&str, Edit)> = vec![
        ("version", Box::new(|r| r.version = Some(2))),
        ("missing version", Box::new(|r| r.version = None)),
        (
            "transform",
            Box::new(|r| r.transform = Some("hadamard".into())),
        ),
        ("axis", Box::new(|r| r.axis = Some("output".into()))),
        (
            "sign mode",
            Box::new(|r| r.sign_mode = Some("seeded".into())),
        ),
        ("block", Box::new(|r| r.block = Some(1000))),
        (
            "width not whole blocks",
            Box::new(|r| r.widths = vec![1000, 2072]),
        ),
        ("short values", Box::new(|r| r.values.truncate(3000))),
        ("trailing values", Box::new(|r| r.values.push(1.0))),
        ("a sign that is not one", Box::new(|r| r.values[5] = 0.5)),
        (
            "a width twice",
            Box::new(|r| {
                r.widths = vec![1024, 1024, 1024];
            }),
        ),
        (
            "a name in both lists",
            Box::new(|r| r.inverse.push("blk.0.ffn_up.weight".into())),
        ),
        (
            "nothing rotated",
            Box::new(|r| {
                r.forward.clear();
                r.inverse.clear();
            }),
        ),
    ];
    for (label, edit) in cases {
        let mut doc = raw();
        edit(&mut doc);
        assert!(
            RotationPlan::check(doc, same).is_err(),
            "{label} was accepted"
        );
    }
}

#[test]
fn an_unknown_tensor_name_is_refused() {
    let err = RotationPlan::check(raw(), |n| {
        (n != "blk.0.ffn_up.weight").then(|| n.to_owned())
    });
    assert!(err.is_err());
}

#[test]
fn repeated_value_heads_without_a_grouped_fold_are_refused() {
    let mut doc = raw();
    doc.grouped = None;
    let plan = RotationPlan::check(doc, same).expect("valid otherwise");
    assert!(plan.value_heads(128, 16, 48).is_err());
}

#[test]
fn mlx_names_map_onto_this_family() {
    let mut doc = raw();
    doc.forward = vec![
        "language_model.lm_head.weight".into(),
        "language_model.model.layers.3.self_attn.o_proj.weight".into(),
        "language_model.model.layers.0.linear_attn.out_proj.weight".into(),
    ];
    doc.inverse = vec!["language_model.model.embed_tokens.weight".into()];
    let json = serde_json::json!({
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 1024,
        "prism.hadamard.transform": TRANSFORM,
        "prism.hadamard.axis": AXIS,
        "prism.hadamard.sign_mode": SIGN_MODE,
        "prism.hadamard.weight_names": doc.forward,
        "prism.hadamard.inverse_weight_names": doc.inverse,
        "prism.hadamard.sign_widths": doc.widths,
        "prism.hadamard.sign_values": doc.values,
        "prism.hadamard.gdn_v_grouped": true,
    });
    let dir = std::env::temp_dir().join(format!("shiso-rotation-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    std::fs::write(dir.join(MLX_FILE), json.to_string()).expect("write");
    let plan = RotationPlan::from_mlx_dir(&dir).expect("valid");
    std::fs::remove_dir_all(&dir).expect("clean up");
    assert_eq!(plan.role("output.weight"), Some(Rotated::Forward));
    assert_eq!(
        plan.role("blk.3.attn_output.weight"),
        Some(Rotated::Forward)
    );
    assert_eq!(plan.role("blk.0.ssm_out.weight"), Some(Rotated::Forward));
    assert_eq!(plan.role("token_embd.weight"), Some(Rotated::Inverse));
}

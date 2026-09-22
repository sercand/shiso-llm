//! The architecture seam: which config documents this build will serve.
//!
//! Recognition is pure host code - no device, no weights file - because the
//! factory's job is dispatch and dispatch is a decision about a document. The
//! load itself needs both and is exercised by the boot test.

use std::path::Path;
use std::sync::Arc;

use shiso_core::{Backend, HfConfig};
use shiso_server::cli::{Cli, Command, ServeArgs};
use shiso_server::model_factory::{FactoryError, load_model, recognises};

mod common;

/// The recorded checkpoint config.
const RECORDED_CONFIG: &str = include_str!("../../../test-vectors/loader/qwen35-config.json");

/// The recorded Gemma 4 config, which is the published checkpoint's own
/// `config.json` rather than a minimal stand-in: this family's reader takes the
/// geometry out of the document's `text_config`, so a stub would be recognised
/// and then refused for the wrong reason.
const RECORDED_GEMMA4_CONFIG: &str =
    include_str!("../../../test-vectors/loader/gemma4-config.json");

/// The recorded mixture-of-experts config: the published checkpoint's own
/// `config.json`, for the reason the Gemma 4 one above is the published
/// document rather than a stub. A minimal stand-in spelling only `model_type`
/// would be recognised here and would prove nothing about the arm, because what
/// this family's reader needs beyond recognition is the whole `text_config`
/// subtree and a quantization document beside the weights.
const RECORDED_QWEN35MOE_CONFIG: &str =
    include_str!("../../../test-vectors/loader/qwen35moe-config.json");

/// The command line this test would have been given.
fn args(model: &Path) -> ServeArgs {
    let cli = <Cli as clap::Parser>::try_parse_from([
        "shiso".to_owned(),
        "serve".to_owned(),
        "--model".to_owned(),
        model.display().to_string(),
        "--ctx".to_owned(),
        "1024".to_owned(),
    ])
    .expect("the command line parses");
    let Command::Serve(serve) = cli.command;
    *serve
}

#[test]
fn the_recorded_checkpoint_config_is_one_this_build_serves() {
    let config = HfConfig::from_json_str(RECORDED_CONFIG).expect("the recorded config parses");
    assert!(
        recognises(&config),
        "the config this port was built against is not recognised by its own factory"
    );
}

/// A document that describes something else is refused here rather than
/// somewhere deep inside a loader, where the error would be about a missing
/// tensor rather than about the wrong architecture.
#[test]
fn a_config_for_another_architecture_is_not_recognised() {
    let config = HfConfig::from_json_str(
        r#"{"model_type": "something_else", "architectures": ["SomethingElseForCausalLM"]}"#,
    )
    .expect("a minimal config parses");
    assert!(!recognises(&config));
}

/// The sparse family is recognised from its `model_type` alone.
///
/// Its reader takes the checkpoint path as well as the document, because this
/// family carries no geometry in a config document - every dimension is in the
/// container's own metadata. Recognition is still a decision about the document,
/// which is what lets a caller refuse an unservable checkpoint before it pays
/// for a device.
#[test]
fn the_sparse_family_is_one_this_build_serves_too() {
    let config = HfConfig::from_json_str(
        r#"{"model_type": "qwen4exp", "architectures": ["Qwen4ExpForCausalLM"]}"#,
    )
    .expect("a minimal config parses");
    assert!(
        recognises(&config),
        "the second family's arm is in the factory but its recognition is not"
    );
}

// The arm list is exhaustive by construction: `load_model` tries every family
// reader this build has and ends in an explicit "no architecture matched".
// Adding a family means adding a reader to that chain, and there is no wildcard
// arm that would let a new family be silently unserved. That is a property of
// the source rather than of a value, so it is recorded here rather than
// asserted - a test that tried to assert it would only be re-reading the match.

/// The Gemma 4 family is recognised from the published document, which is what
/// lets the third arm of the factory ever be reached.
#[test]
fn the_gemma_family_is_one_this_build_serves_too() {
    let config =
        HfConfig::from_json_str(RECORDED_GEMMA4_CONFIG).expect("the recorded config parses");
    assert!(
        recognises(&config),
        "the gemma arm is in the factory but its recognition is not"
    );
}

/// A recognised family whose weights are not there fails as a load failure and
/// not as an unknown architecture.
///
/// The distinction is the whole reason the arms match on `UnknownArch` alone: an
/// operator who mistyped a path should be told the checkpoint would not load,
/// not that this build does not serve Gemma 4.
#[test]
fn a_gemma_checkpoint_that_is_not_there_is_a_load_failure() {
    let config =
        HfConfig::from_json_str(RECORDED_GEMMA4_CONFIG).expect("the recorded config parses");
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    let missing = Path::new("/nonexistent/gemma-4-E4B_q4_0-it.gguf");
    let error = load_model(&backend, missing, &config, &args(missing))
        .expect_err("a checkpoint that is not on disk does not load");
    assert!(
        matches!(error, FactoryError::Load(_)),
        "a recognised family with no weights reported {error:?}"
    );
}

/// Qwen 3.5 keeps Qwen 3's reasoning tags and does not keep its tool-call body:
/// a call is XML, `<function=NAME>` wrapping one `<parameter=KEY>` block per
/// argument, and a parser expecting JSON turns every call back into prose.
#[test]
fn qwen35_fixture_yields_qwen35_defaults() {
    let config = HfConfig::from_json_str(RECORDED_CONFIG).expect("the recorded config parses");
    let defaults = shiso_server::model_factory::FamilyDefaults::for_config(&config)
        .expect("qwen35 config has defaults");
    assert_eq!(
        defaults,
        shiso_server::model_factory::FamilyDefaults {
            reasoning: shiso_parsers::ReasoningScheme::Qwen3,
            tools: shiso_parsers::ToolCallScheme::Qwen35,
            thinking_default: true,
            speculates: true,
        }
    );
}

/// The mixture-of-experts document selects the fourth arm, and it does so with
/// no checkpoint and no device.
///
/// **This is the case that says the arm is reachable at all.** Every number
/// this repository has for this family was measured in process through
/// `crates/shiso-bench`, because `for_config` recognised no `qwen3_5_moe`
/// document and `load_model` had no arm for it: the model loaded, decoded and
/// took a picture, and no server could be pointed at it. Recognition is the
/// half of that which needs neither, so it is graded here.
///
/// The ordering claim is asserted rather than described: the dense reader is
/// tried first and must decline this document. If it ever stopped declining -
/// by relaxing one of the three things it asserts, 32 layers, hidden 2560 and a
/// dense `intermediate_size` - this document would resolve to the dense
/// family's defaults, and this family's own arm - which is where the
/// per-container speculation question is asked - would never run.
#[test]
fn qwen35moe_fixture_yields_its_own_defaults() {
    let config =
        HfConfig::from_json_str(RECORDED_QWEN35MOE_CONFIG).expect("the recorded config parses");
    assert!(
        recognises(&config),
        "the mixture-of-experts arm is in the factory but its recognition is not"
    );
    let defaults = shiso_server::model_factory::FamilyDefaults::for_config(&config)
        .expect("the qwen35moe config has defaults");
    assert_eq!(
        defaults,
        shiso_server::model_factory::FamilyDefaults {
            reasoning: shiso_parsers::ReasoningScheme::Qwen3,
            tools: shiso_parsers::ToolCallScheme::Qwen35,
            thinking_default: true,
            // TRUE since this family gained a head this build can run: the
            // separately published `Qwen3.6-35B-A3B-MTP-4bit`, MLX affine 4-bit
            // group 64. It was false while the only head was the `modelopt`
            // export's own, which is BF16 and has no matvec here.
            //
            // The refusal did not go away - it MOVED, to the arm, because the
            // answer now depends on which container is served. `draft_factory`'s
            // `the_mixture_of_experts_family_refuses_a_depth_per_container`
            // asserts both halves of it, which is the case that has to exist
            // for this `true` to be safe.
            speculates: true,
        },
        "the mixture-of-experts document resolved to another family's defaults; \
         the most likely cause is the dense reader no longer declining it"
    );
    assert_ne!(
        defaults,
        shiso_server::model_factory::FamilyDefaults::QWEN35,
        "the document took the dense arm, which asserts 32 layers and hidden 2560"
    );
}

/// The two Qwen 3.5 documents are disjoint in both directions.
///
/// The half above says the routed document does not take the dense arm. This
/// says the dense document does not take the routed one - which is what would
/// happen if
/// `Qwen35MoeConfig::describes` were ever widened to a prefix test, since
/// `qwen3_5_moe` starts with `qwen3_5`.
#[test]
fn the_dense_document_still_takes_the_dense_arm() {
    let config = HfConfig::from_json_str(RECORDED_CONFIG).expect("the recorded config parses");
    let defaults = shiso_server::model_factory::FamilyDefaults::for_config(&config)
        .expect("qwen35 config has defaults");
    assert_eq!(
        defaults,
        shiso_server::model_factory::FamilyDefaults::QWEN35
    );
}

/// A recognised mixture-of-experts document whose directory is not there fails
/// as a load failure, and the message names the directory.
///
/// This family is the one whose `--model` is a DIRECTORY rather than a
/// container, so the two mistakes an operator can make are "nothing there" and
/// "a file where a directory belongs". Both are refused before a device is
/// touched, which is what lets this case run in the offline tier beside the
/// others.
#[test]
fn a_moe_checkpoint_directory_that_is_not_there_is_a_load_failure() {
    let config =
        HfConfig::from_json_str(RECORDED_QWEN35MOE_CONFIG).expect("the recorded config parses");
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    let missing = Path::new("/nonexistent/Ornith-1.5-35B-A3B-NVFP4");
    let error = load_model(&backend, missing, &config, &args(missing))
        .expect_err("a checkpoint directory that is not on disk does not load");
    let FactoryError::Load(inner) = &error else {
        panic!("expected a load failure, got {error:?}");
    };
    assert!(
        inner.to_string().contains("Ornith-1.5-35B-A3B-NVFP4"),
        "the refusal names the directory it looked for, got {inner}"
    );
}

/// A file where this family's directory belongs is read as a GGUF, and a file
/// that is not one is refused for what it is.
///
/// This case used to assert that a file is refused for BEING a file. That was
/// right while the only container this family had was the `modelopt` export - a
/// directory of shards and two documents - and it stopped being right when the
/// family gained a GGUF reader, which takes a file and reads every width out of
/// its own metadata.
///
/// So what is pinned now is the half that still holds: the arm decides on the
/// PATH, and a file that is not a GGUF fails inside the container reader naming
/// what it could not read, rather than one layer down as a missing
/// `config.json`.
#[test]
fn a_moe_checkpoint_named_as_a_file_is_read_as_a_container() {
    let config =
        HfConfig::from_json_str(RECORDED_QWEN35MOE_CONFIG).expect("the recorded config parses");
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    // This crate's own manifest: the one path every run of this case is
    // guaranteed to have, and a file rather than a directory. It is not a GGUF,
    // so the container reader is what refuses it.
    let a_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let a_file = a_file.as_path();
    let error = load_model(&backend, a_file, &config, &args(a_file))
        .expect_err("this crate's manifest is not a gguf");
    let FactoryError::Load(inner) = &error else {
        panic!("expected a load failure, got {error:?}");
    };
    let text = inner.to_string();
    assert!(
        !text.contains("config.json"),
        "the refusal must come from the container reader, not from a document \
         one layer down, got {text}"
    );
}

#[test]
fn gemma4_fixture_yields_gemma4_defaults() {
    let config =
        HfConfig::from_json_str(RECORDED_GEMMA4_CONFIG).expect("the recorded config parses");
    let defaults = shiso_server::model_factory::FamilyDefaults::for_config(&config)
        .expect("gemma4 config has defaults");
    assert_eq!(
        defaults,
        shiso_server::model_factory::FamilyDefaults {
            reasoning: shiso_parsers::ReasoningScheme::Gemma4,
            tools: shiso_parsers::ToolCallScheme::Gemma4,
            thinking_default: false,
            speculates: false,
        }
    );
}

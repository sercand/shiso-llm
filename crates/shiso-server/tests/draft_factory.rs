//! What the factory does with a draft head the dense arm never honoured.
//!
//! The command line has carried `--spec-k`, `--draft-model` and
//! `--draft-vocab` since the sparse family gained speculation, and it has
//! refused the pair where one is given without the other since then too. What
//! it has not done is *read* them on the dense arm, which is what these cases
//! pin.

use std::path::{Path, PathBuf};

use std::sync::Arc;

use shiso_core::{Backend, HfConfig};
use shiso_server::cli::{Cli, Command, ServeArgs};
use shiso_server::model_factory::{self, FactoryError, FamilyDefaults, load_model};

mod common;

/// The recorded Gemma 4 config, which is the published checkpoint's own
/// document: this family's reader takes its geometry out of `text_config`, so
/// a minimal stub would be recognised and then refused for the wrong reason.
const RECORDED_GEMMA4_CONFIG: &str =
    include_str!("../../../test-vectors/loader/gemma4-config.json");

/// The recorded Qwen 3.5 config, the family that does speculate.
const RECORDED_QWEN35_CONFIG: &str =
    include_str!("../../../test-vectors/loader/qwen35-config.json");

/// The recorded mixture-of-experts config, the family that ships a head this
/// build cannot bind.
const RECORDED_QWEN35MOE_CONFIG: &str =
    include_str!("../../../test-vectors/loader/qwen35moe-config.json");

/// The command line these cases would have been given.
///
/// Built through clap rather than field by field, so the defaults every other
/// field gets are the ones an operator would actually have.
fn args(model: &Path, spec_k: u32, draft: Option<&Path>) -> ServeArgs {
    let mut line = vec![
        "shiso".to_owned(),
        "serve".to_owned(),
        "--model".to_owned(),
        model.display().to_string(),
        "--spec-k".to_owned(),
        spec_k.to_string(),
    ];
    if let Some(head) = draft {
        line.push("--draft-model".to_owned());
        line.push(head.display().to_string());
    }
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(serve) = cli.command else {
        panic!("expected Command::Serve");
    };
    *serve
}

/// The sidecar this family's head is read from, when the machine has one.
fn sidecar() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_OPTIQ").ok()?);
    dir.join("config.json").is_file().then_some(dir)
}

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Asking for speculation without a head is a refusal that names the head.
///
/// Not a silent fallback to plain decode: a server that quietly ignores
/// `--spec-k` answers correctly and slowly, and the operator has no way to see
/// which of the two they got.
///
/// No device and no checkpoint: the refusal happens while the manifest is
/// being resolved, which is before a backend is opened, and that ordering is
/// half of what makes it useful.
#[test]
fn speculation_without_a_head_refuses_by_name() {
    let args = args(
        Path::new("/nonexistent/model"),
        3,
        Some(Path::new("/nonexistent/mtp.safetensors")),
    );
    let error = model_factory::draft_manifest(&args).expect_err("a missing head must refuse");
    let text = error.to_string();
    assert!(
        text.contains("mtp.safetensors"),
        "the refusal names the head it looked for, got {text:?}"
    );
}

/// A depth of zero asks for no head, whatever else is on the command line.
///
/// The pair rule on the command line already refuses a head without a depth,
/// so what is left to state here is the other direction: a build that was not
/// asked to speculate resolves no manifest and therefore pays for no head.
#[test]
fn no_depth_resolves_no_head() {
    let args = args(Path::new("/nonexistent/model"), 0, None);
    assert!(
        model_factory::draft_manifest(&args)
            .expect("no depth is not an error")
            .is_none()
    );
}

/// A head with no depth is still refused on the command line.
///
/// It is a gigabyte of weights made resident that nothing will ever ask a
/// question, and no family arm can turn that into something useful, so this
/// half of the old pair rule stays where it was.
#[test]
fn a_head_with_no_depth_is_refused_on_the_command_line() {
    let args = args(
        Path::new("/nonexistent/model"),
        0,
        Some(Path::new("/nonexistent/mtp.safetensors")),
    );
    let error = args
        .validate()
        .expect_err("a head with no depth must refuse");
    assert!(
        error.to_string().contains("spec-k"),
        "the refusal names the setting that is missing, got {error:?}"
    );
}

/// A depth with no head is NOT refused on the command line any more.
///
/// It used to be, and the cost was that this family's own default - the
/// `mtp_file` its sidecar has always named - was unreachable, and
/// `scripts/serve_qwen35.sh` worked around it by shelling out to python3 to
/// read that key before the server could start. Whether a depth needs an
/// explicit head is a question only the family arm can answer: the dense one
/// has a default and the sparse one does not, and each refuses by name where
/// it must.
#[test]
fn a_depth_with_no_head_is_left_to_the_family_arm() {
    let args = args(Path::new("/nonexistent/model"), 3, None);
    args.validate()
        .expect("a depth alone is a complete instruction for a family with a default");
}

/// With a head present and a depth asked for, the sidecar's own `mtp_file` is
/// what gets resolved.
///
/// `--draft-model` is left unset on purpose: the factory defaults it to the
/// document's own `mtp_file`, and this case is what pins that default.
#[test]
fn the_dense_arm_defaults_to_the_sidecar_head() {
    let Some(dir) = sidecar() else {
        skip("SHISO_TEST_MODEL_OPTIQ names no sidecar directory");
        return;
    };
    let args = args(&dir, 3, None);
    let manifest = model_factory::draft_manifest(&args)
        .expect("the document reads")
        .expect("this sidecar ships a head");
    assert_eq!(manifest.file, dir.join("optiq/mtp.safetensors"));
    assert_eq!(manifest.quant.bits, 4);
}

/// A family this build binds no draft head for refuses a depth, rather than
/// serving without one.
///
/// **This is a regression test for a hole the M3 relaxation opened.** The
/// command line used to refuse `--spec-k` without `--draft-model` for every
/// family. Moving that to the arms made the dense family's sidecar default
/// reachable, and left Gemma 4 - whose arm never reads `args.spec_k` at all -
/// passing validation, passing the factory, and serving with **no speculation
/// and no error**. That is the exact failure the relaxation's own rationale
/// named as the thing that must not happen.
///
/// It drives the real path rather than inspecting the flag: `load_model`
/// against a fake backend and a checkpoint that is not on disk. The refusal
/// has to arrive anyway, because it is answered from the config document
/// before any arm reaches for a byte of weights - and the assertion below is
/// what says so, since a missing checkpoint would otherwise be a load failure
/// of its own.
#[test]
fn a_family_that_binds_no_head_refuses_a_speculative_depth() {
    let config = HfConfig::from_json_str(RECORDED_GEMMA4_CONFIG).expect("the recorded config");
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    let missing = Path::new("/nonexistent/gemma-4-E4B_q4_0-it.gguf");
    let asked = args(missing, 3, None);

    let error = load_model(&backend, missing, &config, &asked)
        .expect_err("a family with no draft head must refuse a depth");
    let FactoryError::Load(inner) = &error else {
        panic!("expected a load failure, got {error:?}");
    };
    let text = inner.to_string();
    assert!(
        text.contains("--spec-k") && text.contains("gemma"),
        "the refusal names the setting and the family, got {text:?}"
    );
}

/// The same family with no depth asked for loads as it always did.
///
/// The other half: the refusal must fire on the depth and on nothing else. A
/// check that turned every Gemma 4 start into an error would pass the case
/// above and break every operator who never asked to speculate. The missing
/// checkpoint is what this one fails on, and its message is the proof the
/// speculative refusal did not fire.
#[test]
fn the_same_family_without_a_depth_gets_no_speculative_refusal() {
    let config = HfConfig::from_json_str(RECORDED_GEMMA4_CONFIG).expect("the recorded config");
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    let missing = Path::new("/nonexistent/gemma-4-E4B_q4_0-it.gguf");

    let error = load_model(&backend, missing, &config, &args(missing, 0, None))
        .expect_err("the checkpoint is not on disk");
    let FactoryError::Load(inner) = &error else {
        panic!("expected a load failure, got {error:?}");
    };
    assert!(
        !inner.to_string().contains("--spec-k"),
        "a start with no depth must not be refused for speculation, got {inner}"
    );
}

/// The mixture-of-experts family refuses a depth PER CONTAINER, and names which
/// one it refused.
///
/// This case used to assert the opposite - that the refusal fires on the FAMILY,
/// from `FamilyDefaults::QWEN35MOE`'s `speculates: false`, before any arm looks
/// at a path. That was right while the only head this family had was the
/// `modelopt` export's own, which is BF16 and which this build binds no matvec
/// for: there was no head to find, so making the arm responsible for finding one
/// would have been making it responsible for a file that does not exist.
///
/// It gained a second head - `Qwen3.6-35B-A3B-MTP-4bit`, MLX affine 4-bit group
/// 64, which this build does decode and which drafts at 0.860 acceptance - and
/// that head is published beside the GGUF container rather than the safetensors
/// one. So the answer now depends on which container is served, which is a
/// question only the arm can ask, and the family constant says `true`.
///
/// **Both halves are asserted here**, because a `true` on the constant with an
/// arm that forgot to ask is exactly how Gemma 4 once served with no
/// speculation and no error.
///
/// Still no device and no weights: both containers below are EMPTY - a bare
/// directory and a zero-byte file - so the refusal fires before anything is
/// read, which is the property the previous spelling of this case was protecting
/// and the one worth keeping.
#[test]
fn the_mixture_of_experts_family_refuses_a_depth_per_container() {
    let config = HfConfig::from_json_str(RECORDED_QWEN35MOE_CONFIG).expect("the recorded config");
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    let root = tempfile::tempdir().expect("a temporary directory");

    // The safetensors export: a DIRECTORY, whose own head is BF16 and
    // unrunnable here, so a depth is refused whatever else is passed.
    let dir = root.path().join("Qwen3.6-35B-A3B-NVFP4");
    std::fs::create_dir(&dir).expect("the container directory");
    let error = load_model(&backend, &dir, &config, &args(&dir, 3, None))
        .expect_err("the directory container binds no head this build can run");
    let FactoryError::Load(inner) = &error else {
        panic!("expected a load failure, got {error:?}");
    };
    let text = inner.to_string();
    assert!(
        text.contains("--spec-k") && text.contains("BF16"),
        "the refusal names the setting and why this container cannot serve it, got {text:?}"
    );

    // The GGUF: a FILE, which CAN speculate - but carries no head beside it and
    // no `mtp_file` to default to, so a depth without `--draft-model` is a
    // refusal naming the flag rather than a server that quietly decodes one
    // token a step.
    let gguf = root.path().join("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf");
    std::fs::write(&gguf, b"").expect("the container file");
    let error = load_model(&backend, &gguf, &config, &args(&gguf, 3, None))
        .expect_err("a depth with no head named must be refused");
    let FactoryError::Load(inner) = &error else {
        panic!("expected a load failure, got {error:?}");
    };
    let text = inner.to_string();
    assert!(
        text.contains("--draft-model"),
        "the refusal names the flag that would have supplied the head, got {text:?}"
    );
}

/// The other half, as for Gemma 4: with no depth asked for, the refusal must
/// not fire, and what this start fails on is the missing directory.
#[test]
fn the_mixture_of_experts_family_without_a_depth_gets_no_speculative_refusal() {
    let config = HfConfig::from_json_str(RECORDED_QWEN35MOE_CONFIG).expect("the recorded config");
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    let missing = Path::new("/nonexistent/Qwen3.6-35B-A3B-NVFP4");

    let error = load_model(&backend, missing, &config, &args(missing, 0, None))
        .expect_err("the checkpoint directory is not on disk");
    let FactoryError::Load(inner) = &error else {
        panic!("expected a load failure, got {error:?}");
    };
    assert!(
        !inner.to_string().contains("--spec-k"),
        "a start with no depth must not be refused for speculation, got {inner}"
    );
}

/// The two families that DO speculate say so, so the refusal cannot fire for
/// them.
///
/// A blanket refusal keyed on a hard-coded list of non-speculating families
/// would go stale the day a family gained a head; keying it on the family's
/// own constant means the answer moves with the family. This fails if someone
/// flips the dense family off while its arm still resolves a head.
#[test]
fn the_families_that_speculate_are_not_refused() {
    let config = HfConfig::from_json_str(RECORDED_QWEN35_CONFIG).expect("the recorded config");
    let defaults = FamilyDefaults::for_config(&config).expect("qwen35 has defaults");
    assert!(
        defaults.speculates,
        "the dense family resolves its head from the sidecar, so a depth must reach its arm"
    );
}

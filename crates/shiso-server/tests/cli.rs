//! The command line: defaults, environment fallbacks, and the checks clap
//! cannot make.
//!
//! # Why no test here mutates the environment
//!
//! The harness runs a binary's tests as threads in one process, so a variable
//! set by one test is visible to every other test that reads it, and the
//! ordering between them is not controllable. On this edition setting one is
//! also `unsafe`, which this workspace denies outright.
//!
//! So the fallback is checked two ways that need no mutation. The declaration,
//! meaning that a flag has an environment fallback at all and which variable it
//! is, is visible in the rendered help, which is where an operator reads it
//! too. The one behaviour that cannot be checked that way, that a configured
//! token is never echoed back, is checked by running the real binary in its own
//! process with the variable set.

use std::path::PathBuf;

use clap::Parser as _;
use shiso_server::cli::{Cli, Command, ServeArgs};

/// Parses a `shiso serve` command line.
fn parse(args: &[&str]) -> Result<ServeArgs, clap::Error> {
    let mut line = vec!["shiso", "serve"];
    line.extend_from_slice(args);
    let cli = Cli::try_parse_from(line)?;
    let Command::Serve(serve) = cli.command;
    Ok(*serve)
}

/// The long help for `shiso serve`, which is where clap documents the
/// environment fallbacks.
fn serve_help() -> String {
    let mut command = <Cli as clap::CommandFactory>::command();
    let mut serve = command
        .find_subcommand_mut("serve")
        .expect("serve is a subcommand")
        .clone();
    serve.render_long_help().to_string()
}

#[test]
fn a_bare_serve_takes_the_documented_defaults() {
    let args = parse(&["--model", "m.gguf"]).expect("a model is the only required flag");
    assert_eq!(args.bind, "127.0.0.1");
    assert_eq!(args.port, 8090);
    assert_eq!(args.ctx, 4096);
    assert_eq!(args.kvstore_budget_mb, None, "unset is twice the pool");
    assert_eq!(args.kvstore_min_tokens, 512);
    assert_eq!(args.kvstore_pump_mb, 64);
    assert_eq!(args.spec_k, 0);
    assert!(args.kvstore_dir.is_none());
    assert!(args.auth_tokens.is_empty());
    assert!(args.vision.is_none());
    assert!(!args.video_allow_ffmpeg);
    assert!(!args.vision_allow_remote_images);
}

#[test]
fn the_kv_cache_is_sized_by_tokens_or_by_a_share_of_the_device_never_both() {
    let args = parse(&["--model", "m.gguf"]).expect("it parses");
    assert_eq!(args.max_num_seqs, 4);
    assert_eq!(args.kv_pool_pages().expect("nothing to refuse"), None);
    assert!(args.kv_fit().expect("nothing to refuse").is_none());

    // Tokens round up to whole pages.
    let args = parse(&["--model", "m.gguf", "--kv-cache-tokens", "65537"]).expect("it parses");
    args.validate().expect("a pool wider than one context");
    assert_eq!(args.kv_pool_pages().expect("a valid pool"), Some(257));

    let args = parse(&[
        "--model",
        "m.gguf",
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-seqs",
        "8",
    ])
    .expect("it parses");
    args.validate().expect("a valid share");
    let fit = args.kv_fit().expect("a valid share").expect("asked for");
    assert_eq!(fit.sessions(), 8);
    assert!((fit.fraction() - 0.9).abs() < 1e-12);

    assert!(
        parse(&[
            "--model",
            "m.gguf",
            "--kv-cache-tokens",
            "65536",
            "--gpu-memory-utilization",
            "0.9",
        ])
        .is_err(),
        "a pool and a share are two answers to one question"
    );
}

#[test]
fn the_checkpoint_count_defaults_to_the_engine_s_and_zero_is_a_count_not_a_default() {
    let args = parse(&["--model", "m.gguf"]).expect("it parses");
    assert_eq!(
        args.kv_checkpoints,
        shiso_engine::EngineOptions::new().kv_ckpt_slots,
        "the flag's default and the engine's are one number"
    );
    let args = parse(&["--model", "m.gguf", "--kv-checkpoints", "40"]).expect("it parses");
    args.validate().expect("any count is a valid count");
    assert_eq!(args.kv_checkpoints, 40);
    // Zero turns prefix reuse off; it is not refused the way a zero budget
    // is, because nothing downstream substitutes a default for it.
    let args = parse(&["--model", "m.gguf", "--kv-checkpoints", "0"]).expect("it parses");
    args.validate().expect("zero holds none");
    assert_eq!(args.kv_checkpoints, 0);
    assert!(parse(&["--model", "m.gguf", "--kv-checkpoints=-1"]).is_err());
}

#[test]
fn a_kv_cache_that_cannot_hold_one_context_or_a_share_outside_zero_to_one_is_refused() {
    let small = parse(&[
        "--model",
        "m.gguf",
        "--ctx",
        "8192",
        "--kv-cache-tokens",
        "4096",
    ])
    .expect("it parses");
    assert!(small.validate().is_err());
    // `=` because a leading minus would otherwise read as a flag.
    for share in ["0", "1.5", "-0.2"] {
        let flag = format!("--gpu-memory-utilization={share}");
        let args = parse(&["--model", "m.gguf", &flag]).expect("it parses");
        assert!(args.validate().is_err(), "{share}");
    }
    let none = parse(&["--model", "m.gguf", "--max-num-seqs", "0"]).expect("it parses");
    assert!(none.validate().is_err());
}

#[test]
fn the_model_id_is_the_checkpoint_file_stem() {
    let args = parse(&["--model", "/models/Qwen3.5-4B-Q4_K_M.gguf"]).expect("it parses");
    assert_eq!(args.model_id(), "Qwen3.5-4B-Q4_K_M");
}

/// The default is a poor name exactly where a checkpoint is a directory named
/// after its export rather than its model, which is the case `--model-name`
/// exists for. Both halves are asserted together because the second is only
/// interesting given the first.
#[test]
fn an_explicit_name_overrides_the_stem_a_directory_checkpoint_would_serve_under() {
    let args = parse(&["--model", "/models/Qwen3.5-4B/mlx"]).expect("it parses");
    assert_eq!(
        args.model_id(),
        "mlx",
        "the last path component is all the default has to go on"
    );

    let args = parse(&[
        "--model",
        "/models/Qwen3.5-4B/mlx",
        "--model-name",
        "Qwen3.5-4B",
    ])
    .expect("it parses");
    assert_eq!(args.model_id(), "Qwen3.5-4B");
}

/// A path with no last component is refused for want of a name - unless a name
/// was given, which is the whole question that check was asking.
#[test]
fn a_nameless_checkpoint_path_is_servable_once_it_has_been_named() {
    let args = parse(&["--model", "/"]).expect("it parses");
    args.validate()
        .expect_err("a root path cannot name a model on its own");

    let args = parse(&["--model", "/", "--model-name", "whatever"]).expect("it parses");
    args.validate().expect("an explicit name answers it");
}

/// The config defaults to the checkpoint's own directory, and the vocabulary to
/// the config's - which are frequently not the same place.
#[test]
fn the_config_and_vocabulary_default_beside_the_files_they_describe() {
    let args = parse(&["--model", "/models/weights/m.gguf"]).expect("it parses");
    assert_eq!(
        args.config_path(),
        PathBuf::from("/models/weights/config.json")
    );
    assert_eq!(
        args.tokenizer_path(),
        PathBuf::from("/models/weights/tokenizer.json")
    );

    let args = parse(&[
        "--model",
        "/models/weights/m.gguf",
        "--config",
        "/models/mlx/config.json",
    ])
    .expect("it parses");
    assert_eq!(args.config_path(), PathBuf::from("/models/mlx/config.json"));
    assert_eq!(
        args.tokenizer_path(),
        PathBuf::from("/models/mlx/tokenizer.json"),
        "the vocabulary follows the config, not the weights"
    );
}

/// The sampling defaults sit beside the config, as the checkpoint ships them,
/// unless the operator names another file.
#[test]
fn the_generation_config_defaults_beside_the_config() {
    let args = parse(&["--model", "/models/weights/m.gguf"]).expect("it parses");
    assert_eq!(
        args.generation_config_path(),
        PathBuf::from("/models/weights/generation_config.json")
    );
    let args = parse(&[
        "--model",
        "/models/weights/m.gguf",
        "--generation-config",
        "/elsewhere/card.json",
    ])
    .expect("it parses");
    assert_eq!(
        args.generation_config_path(),
        PathBuf::from("/elsewhere/card.json")
    );
}

/// A checkpoint directory carries its own config, so "beside the checkpoint"
/// means inside it. Getting this wrong points the default one level too high,
/// where the sibling export's config is as likely to be found as nothing at
/// all, and a geometry read from the wrong export is a model that loads and is
/// wrong rather than one that refuses.
#[test]
fn a_checkpoint_directory_finds_its_config_inside_itself() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let checkpoint = root.path().join("mlx");
    std::fs::create_dir(&checkpoint).expect("the fixture is writable");

    let args = parse(&["--model", &checkpoint.display().to_string()]).expect("it parses");
    assert_eq!(args.config_path(), checkpoint.join("config.json"));
    assert_eq!(args.tokenizer_path(), checkpoint.join("tokenizer.json"));
}

#[test]
fn auth_tokens_are_a_comma_separated_list_with_blanks_dropped() {
    let args = parse(&["--model", "m.gguf", "--auth-tokens", "a,b,,c"]).expect("it parses");
    assert_eq!(args.auth_tokens(), vec!["a", "b", "c"]);
}

#[test]
fn a_port_that_is_not_a_number_is_a_parse_failure() {
    let error = parse(&["--model", "m.gguf", "--port", "abc"]).expect_err("clap refuses");
    assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
}

#[test]
fn a_context_of_zero_is_refused_by_validate_rather_than_by_clap() {
    let args = parse(&["--model", "m.gguf", "--ctx", "0"]).expect("clap has no opinion on zero");
    let error = args
        .validate()
        .expect_err("a zero context cannot be served");
    assert!(error.to_string().contains("ctx"), "{error}");
}

/// Zero is silently replaced with a default further down, so it is refused
/// here: a server that quietly served a budget other than the one it was given
/// is worse than one that will not start.
#[test]
fn a_zero_budget_is_refused_because_something_downstream_would_substitute_one() {
    let args = parse(&["--model", "m.gguf", "--kvstore-budget-mb", "0"]).expect("it parses");
    assert!(args.validate().is_err());
}

#[test]
fn a_servable_command_line_validates() {
    let args = parse(&["--model", "m.gguf", "--ctx", "8192"]).expect("it parses");
    args.validate().expect("every value is in range");
}

/// The video flags become a decode policy only when decoding is allowed, and
/// the absence of a policy is what "off" means - there is no policy value that
/// means "configured but not allowed".
#[test]
fn the_video_flags_become_a_policy_only_when_decoding_is_allowed() {
    let off = parse(&["--model", "m.gguf"]).expect("it parses");
    assert!(off.ffmpeg_policy().is_none());

    let on = parse(&[
        "--model",
        "m.gguf",
        "--video-allow-ffmpeg",
        "--video-max-frames",
        "32",
        "--video-decode-timeout-s",
        "5",
    ])
    .expect("it parses");
    let policy = on.ffmpeg_policy().expect("decoding is allowed");
    assert_eq!(policy.max_frames(), 32);
    assert_eq!(policy.timeout_secs(), 5);
}

#[test]
fn the_pixel_ceiling_flag_becomes_the_image_limit() {
    let args = parse(&["--model", "m.gguf", "--vision-max-pixels", "65536"]).expect("it parses");
    assert_eq!(
        args.image_limits()
            .max_pixels()
            .map(std::num::NonZeroU64::get),
        Some(65_536)
    );
}

/// Every flag that is documented as having an environment fallback actually
/// declares one, spelled with the project's prefix.
///
/// A fallback that exists in the documentation and not in the attribute is a
/// setting an operator configures and the server ignores.
#[test]
fn every_documented_flag_declares_its_prefixed_environment_fallback() {
    let help = serve_help();
    for variable in [
        "SHISO_BIND",
        "SHISO_PORT",
        "SHISO_CTX",
        "SHISO_MAX_NUM_SEQS",
        "SHISO_KV_CACHE_TOKENS",
        "SHISO_GPU_MEMORY_UTILIZATION",
        "SHISO_KV_CHECKPOINTS",
        "SHISO_KVSTORE_DIR",
        "SHISO_KVSTORE_BUDGET_MB",
        "SHISO_AUTH_TOKENS",
        "SHISO_SPEC_K",
        "SHISO_VISION",
        "SHISO_VIDEO_ALLOW_FFMPEG",
        "SHISO_GENERATION_CONFIG",
        "SHISO_SOLO_PREFILL_CHUNK",
        "SHISO_MIXED_PREFILL_QUANTUM",
        "SHISO_MAX_NUM_BATCHED_TOKENS",
    ] {
        assert!(
            help.contains(variable),
            "{variable} is not declared as a fallback: {help}"
        );
    }
    assert!(
        !help.contains("RUST_LOG"),
        "every knob this binary reads carries the project's prefix"
    );
}

/// The flags the acceptance list names are all present in the help.
#[test]
fn the_help_lists_every_flag_an_operator_is_told_about() {
    let help = serve_help();
    for flag in [
        "--model",
        "--bind",
        "--port",
        "--ctx",
        "--kvstore-dir",
        "--kvstore-budget-mb",
        "--auth-tokens",
        "--vision",
        "--video-allow-ffmpeg",
        "--generation-config",
    ] {
        assert!(help.contains(flag), "{flag} is missing from the help");
    }
}

/// A bearer token must never be echoed by the help text, which clap fills in
/// with an environment variable's value unless the flag says not to.
///
/// Run as a subprocess so that the variable is set for this check and for
/// nothing else in the harness.
#[test]
fn the_help_text_never_prints_a_configured_bearer_token() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_shiso"))
        .args(["serve", "--help"])
        .env("SHISO_AUTH_TOKENS", "sk-supersecret")
        .output()
        .expect("the binary runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        text.contains("--auth-tokens"),
        "the flag itself is still documented: {text}"
    );
    assert!(
        !text.contains("sk-supersecret"),
        "a configured token leaked into the help text: {text}"
    );
}

/// The two prefill widths default to the scheduler's own constants.
#[test]
fn the_prefill_widths_default_to_the_scheduler_constants() {
    let args = parse(&["--model", "m.gguf"]).expect("it parses");
    let widths = args.prefill_widths().expect("the defaults are servable");
    assert_eq!(
        widths.solo_chunk,
        shiso_server::scheduler::DEFAULT_SOLO_CHUNK
    );
    assert_eq!(
        widths.mixed_quantum,
        shiso_server::scheduler::DEFAULT_MIXED_QUANTUM
    );
}

/// Each width can be set on its own.
#[test]
fn each_prefill_width_is_its_own_flag() {
    let args = parse(&[
        "--model",
        "m.gguf",
        "--solo-prefill-chunk",
        "4096",
        "--mixed-prefill-quantum",
        "1024",
    ])
    .expect("it parses");
    let widths = args.prefill_widths().expect("both are positive");
    assert_eq!((widths.solo_chunk, widths.mixed_quantum), (4096, 1024));
}

/// A per-step budget sets both widths: the whole budget when nothing decodes,
/// and what the decode rows leave of it when something does.
///
/// The decode rows are counted at their worst: every resident sequence
/// verifying a full draft is `max-num-seqs * (1 + spec-k)` rows in one step.
#[test]
fn a_step_budget_leaves_the_prefill_what_the_decode_rows_do_not_take() {
    let args = parse(&[
        "--model",
        "m.gguf",
        "--max-num-batched-tokens",
        "2048",
        "--max-num-seqs",
        "4",
        "--spec-k",
        "3",
    ])
    .expect("it parses");
    let widths = args.prefill_widths().expect("the budget leaves room");
    assert_eq!(widths.solo_chunk, 2048);
    assert_eq!(widths.mixed_quantum, 2048 - 16);
}

/// A budget and an explicit width say two different things about the same
/// step, so the parse refuses rather than picking one.
#[test]
fn a_step_budget_and_an_explicit_width_are_refused_together() {
    for width in ["--mixed-prefill-quantum", "--solo-prefill-chunk"] {
        let parsed = parse(&[
            "--model",
            "m.gguf",
            "--max-num-batched-tokens",
            "4096",
            width,
            "512",
        ]);
        assert!(parsed.is_err(), "{width} beside a budget must be refused");
    }
}

/// A budget the decode rows alone fill leaves no prefill at all, which would
/// starve every waiting prompt; it is refused at boot instead.
#[test]
fn a_step_budget_that_the_decode_rows_fill_is_refused() {
    let args = parse(&[
        "--model",
        "m.gguf",
        "--max-num-batched-tokens",
        "16",
        "--max-num-seqs",
        "4",
        "--spec-k",
        "3",
    ])
    .expect("it parses");
    let error = args.validate().expect_err("no room is left for a prefill");
    assert!(
        error.to_string().contains("max-num-batched-tokens"),
        "{error}"
    );
}

/// Zero widths are refused, never quietly raised to one.
#[test]
fn a_zero_prefill_width_is_refused() {
    for flag in [
        "--solo-prefill-chunk",
        "--mixed-prefill-quantum",
        "--max-num-batched-tokens",
    ] {
        let args = parse(&["--model", "m.gguf", flag, "0"]).expect("it parses");
        let error = args.validate().expect_err("zero is not a width");
        assert!(error.to_string().contains(&flag[2..]), "{flag}: {error}");
    }
}

#[test]
fn an_audio_only_server_needs_no_language_model() {
    let args = parse(&["--asr-model", "/models/nemotron-asr"]).expect("the ASR alone parses");
    assert!(args.model.is_none());
    args.validate().expect("an ASR alone is servable");
    assert_eq!(args.asr_id().as_deref(), Some("nemotron-asr"));
    assert_eq!(args.asr_lookahead, 13);
    assert_eq!(args.audio_max_streams, 4);
    assert_eq!(args.audio_max_seconds.get(), 7200);
    assert_eq!(args.audio_max_upload_bytes(), 512 * 1024 * 1024);
}

#[test]
fn a_server_with_neither_model_is_refused() {
    let error = parse(&[])
        .expect("parses")
        .validate()
        .expect_err("nothing to serve");
    let text = error.to_string();
    assert!(
        text.contains("--model") && text.contains("--asr-model"),
        "{text}"
    );
}

#[test]
fn a_diarizer_without_a_recognizer_is_refused() {
    let error = parse(&["--model", "m.gguf", "--diarization-model", "/models/diar"])
        .expect("parses")
        .validate()
        .expect_err("a diarizer has nothing to attribute");
    assert!(error.to_string().contains("--asr-model"), "{error}");
}

#[test]
fn zero_audio_streams_is_refused() {
    let error = parse(&["--asr-model", "/a", "--audio-max-streams", "0"])
        .expect("parses")
        .validate()
        .expect_err("zero streams serves nothing");
    assert!(error.to_string().contains("audio-max-streams"), "{error}");
}

#[test]
fn the_asr_id_can_be_named() {
    let args = parse(&["--asr-model", "/models/asr/mlx", "--asr-id", "nemotron"]).expect("parses");
    assert_eq!(args.asr_id().as_deref(), Some("nemotron"));
}

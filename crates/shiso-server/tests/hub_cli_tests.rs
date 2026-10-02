#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]

//! CLI argument resolution and recipe default precedence tests (§9.2).

use std::path::PathBuf;

use clap::{CommandFactory as _, FromArgMatches as _, Parser as _};
use shiso_hub::recipe::ServerDefaults;
use shiso_server::cli::{Cli, ModelArgKind, ServeArgs, apply_recipe_defaults, classify_model_arg};
use shiso_server::hub::resolve_serve_model;

#[test]
fn classify_model_argument_path_vs_reference() {
    // Paths
    assert_eq!(
        classify_model_arg("/var/models/qwen"),
        ModelArgKind::Path(PathBuf::from("/var/models/qwen"))
    );
    assert_eq!(
        classify_model_arg("./models/qwen"),
        ModelArgKind::Path(PathBuf::from("./models/qwen"))
    );
    assert_eq!(
        classify_model_arg("~/models/qwen"),
        ModelArgKind::Path(PathBuf::from("~/models/qwen"))
    );
    assert_eq!(
        classify_model_arg("checkpoints/model.gguf"),
        ModelArgKind::Path(PathBuf::from("checkpoints/model.gguf"))
    );

    // Registry references (never checks current dir, no slash/dot/tilde)
    assert_eq!(
        classify_model_arg("qwen3.5-9b"),
        ModelArgKind::RegistryRef("qwen3.5-9b".to_string())
    );
    assert_eq!(
        classify_model_arg("qwen3.5-9b:metal-mlx4"),
        ModelArgKind::RegistryRef("qwen3.5-9b:metal-mlx4".to_string())
    );
}

#[test]
fn component_flag_conflicts_with_model_reference() {
    // Providing model_ref + --vision must be refused
    let cli = Cli::parse_from([
        "shiso",
        "serve",
        "qwen3.5-9b",
        "--vision",
        "/path/to/vision.safetensors",
    ]);
    let shiso_server::cli::Command::Serve(mut args) = cli.command else {
        panic!("expected Serve command");
    };

    let res = resolve_serve_model(&mut args, None);
    assert!(res.is_err());
    let err_str = res.unwrap_err().to_string();
    assert!(err_str.contains("cannot pass component flags together with model reference"));
}

#[test]
fn model_name_defaults_to_canonical_recipe_name() {
    let cli = Cli::parse_from(["shiso", "serve", "qwen3.5-9b"]);
    let shiso_server::cli::Command::Serve(args) = cli.command else {
        panic!("expected Serve command");
    };
    assert_eq!(args.model_ref.as_deref(), Some("qwen3.5-9b"));
    assert_eq!(args.model_name, None);
}

#[test]
fn apply_recipe_defaults_precedence() {
    // 1. Neither flag nor env: recipe default applies
    let cmd = Cli::command();
    let matches = cmd.get_matches_from(["shiso", "serve", "-m", "/test/model"]);
    let serve_matches = matches.subcommand_matches("serve").unwrap();
    let mut args = ServeArgs::from_arg_matches(serve_matches).unwrap();

    let defaults = ServerDefaults {
        ctx: Some(32768),
        max_num_seqs: Some(64),
        spec_k: Some(5),
        turn_replay_tokens: Some(1000),
        vision_max_pixels: Some(1920 * 1080),
        video_fps: Some(15.0),
        video_max_frames: Some(120),
        asr_lookahead: Some(8),
        asr_batch: Some(4),
        expert_residency: Some("resident".to_string()),
    };

    apply_recipe_defaults(&mut args, Some(serve_matches), &defaults);

    assert_eq!(args.ctx, 32768);
    assert_eq!(args.max_num_seqs, 64);
    assert_eq!(args.spec_k, 5);
    assert_eq!(args.turn_replay_tokens, 1000);
    assert_eq!(args.vision_max_pixels, 1920 * 1080);
    assert!((args.video_fps - 15.0).abs() < 1e-6);
    assert_eq!(args.video_max_frames, 120);
    assert_eq!(args.asr_lookahead, 8);
    assert_eq!(args.asr_batch, 4);

    // 2. Explicit CLI flag: wins over recipe default
    let cmd = Cli::command();
    let matches = cmd.get_matches_from(["shiso", "serve", "-m", "/test/model", "--ctx", "16384"]);
    let serve_matches = matches.subcommand_matches("serve").unwrap();
    let mut args = ServeArgs::from_arg_matches(serve_matches).unwrap();

    apply_recipe_defaults(&mut args, Some(serve_matches), &defaults);
    // Explicit 16384 wins over recipe's 32768
    assert_eq!(args.ctx, 16384);
    // Other fields without explicit flags still take recipe defaults
    assert_eq!(args.max_num_seqs, 64);
    assert_eq!(args.spec_k, 5);

    // 3. Recipe doesn't set the field: built-in default is kept
    let empty_defaults = ServerDefaults::default();
    let cmd = Cli::command();
    let matches = cmd.get_matches_from(["shiso", "serve", "-m", "/test/model"]);
    let serve_matches = matches.subcommand_matches("serve").unwrap();
    let mut args = ServeArgs::from_arg_matches(serve_matches).unwrap();

    let builtin_ctx = args.ctx;
    apply_recipe_defaults(&mut args, Some(serve_matches), &empty_defaults);
    assert_eq!(args.ctx, builtin_ctx);
}

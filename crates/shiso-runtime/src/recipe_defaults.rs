//! Folding a registry recipe's `server_defaults` into [`ServeOptions`] (spec §6.6).
//!
//! The precedence is `explicit setting > SHISO_* environment > recipe
//! server_defaults > built-in default`. Which settings are explicit is the
//! caller's [`ExplicitlySet`]: a command line builds it from what it parsed, an
//! embedder passes the names it set itself.

use shiso_hub::recipe::{ExpertResidency, ServerDefaults};

use crate::options::{ExpertResidencyChoice, ExplicitlySet, KvCacheDtype, ServeOptions};

/// Merges recipe `server_defaults` into `args`, leaving every option named in
/// `explicit` as the user set it (§6.6).
pub fn apply_recipe_defaults_unless_set(
    args: &mut ServeOptions,
    explicit: &ExplicitlySet,
    defaults: &ServerDefaults,
) {
    let check_override = |id: &str| !explicit.contains(id);

    if let Some(ctx) = defaults.ctx
        && check_override("ctx")
    {
        args.ctx = u32::try_from(ctx).unwrap_or(args.ctx);
        tracing::info!(ctx = args.ctx, "applied recipe default for ctx");
    }

    if let Some(max_num_seqs) = defaults.max_num_seqs
        && check_override("max_num_seqs")
    {
        args.max_num_seqs = max_num_seqs;
        tracing::info!(
            max_num_seqs = args.max_num_seqs,
            "applied recipe default for max_num_seqs"
        );
    }

    if let Some(spec_k) = defaults.spec_k
        && check_override("spec_k")
    {
        args.spec_k = u32::try_from(spec_k).unwrap_or(args.spec_k);
        tracing::info!(spec_k = args.spec_k, "applied recipe default for spec_k");
    }

    if let Some(turn_replay_tokens) = defaults.turn_replay_tokens
        && check_override("turn_replay_tokens")
    {
        args.turn_replay_tokens = turn_replay_tokens;
        tracing::info!(
            turn_replay_tokens = args.turn_replay_tokens,
            "applied recipe default for turn_replay_tokens"
        );
    }

    if let Some(vision_max_pixels) = defaults.vision_max_pixels
        && check_override("vision_max_pixels")
    {
        args.vision_max_pixels = vision_max_pixels;
        tracing::info!(
            vision_max_pixels = args.vision_max_pixels,
            "applied recipe default for vision_max_pixels"
        );
    }

    if let Some(video_fps) = defaults.video_fps
        && check_override("video_fps")
    {
        args.video_fps = f64::from(video_fps);
        tracing::info!(
            video_fps = args.video_fps,
            "applied recipe default for video_fps"
        );
    }

    if let Some(video_max_frames) = defaults.video_max_frames
        && check_override("video_max_frames")
    {
        args.video_max_frames = video_max_frames;
        tracing::info!(
            video_max_frames = args.video_max_frames,
            "applied recipe default for video_max_frames"
        );
    }

    if let Some(asr_lookahead) = defaults.asr_lookahead
        && check_override("asr_lookahead")
    {
        args.asr_lookahead = asr_lookahead;
        tracing::info!(
            asr_lookahead = args.asr_lookahead,
            "applied recipe default for asr_lookahead"
        );
    }

    if let Some(asr_batch) = defaults.asr_batch
        && check_override("asr_batch")
    {
        args.asr_batch = asr_batch;
        tracing::info!(
            asr_batch = args.asr_batch,
            "applied recipe default for asr_batch"
        );
    }

    apply_engine_defaults(args, explicit, defaults);
}

/// The recipe defaults that tune the engine rather than the request surface:
/// expert residency, speculation and capture, and the memory and prefill
/// budgets.
fn apply_engine_defaults(
    args: &mut ServeOptions,
    explicit: &ExplicitlySet,
    defaults: &ServerDefaults,
) {
    let check_override = |id: &str| !explicit.contains(id);

    if let Some(residency) = defaults.expert_residency
        && check_override("expert_residency")
    {
        args.expert_residency = match residency {
            ExpertResidency::Auto => ExpertResidencyChoice::Auto,
            ExpertResidency::Resident => ExpertResidencyChoice::Resident,
            ExpertResidency::Stream => ExpertResidencyChoice::Stream,
        };
        tracing::info!(?residency, "applied recipe default for expert_residency");
    }

    if let Some(kv) = defaults.kv_cache_dtype
        && check_override("kv_cache_dtype")
    {
        args.kv_cache_dtype = match kv {
            shiso_hub::recipe::KvCacheDtype::F16 => KvCacheDtype::F16,
            shiso_hub::recipe::KvCacheDtype::Fp8 => KvCacheDtype::Fp8,
        };
        tracing::info!(?kv, "applied recipe default for kv_cache_dtype");
    }

    if let Some(reserve) = defaults.expert_reserve_gb
        && check_override("expert_reserve_gb")
    {
        args.expert_reserve_gb = reserve;
        tracing::info!(reserve, "applied recipe default for expert_reserve_gb");
    }

    if let Some(slots) = defaults.expert_slots
        && check_override("expert_slots")
    {
        args.expert_slots = slots;
        tracing::info!(slots, "applied recipe default for expert_slots");
    }

    if let Some(spec_lookup) = defaults.spec_lookup
        && check_override("spec_lookup")
    {
        args.spec_lookup = spec_lookup;
        tracing::info!(spec_lookup, "applied recipe default for spec_lookup");
    }

    if let Some(capture_verify) = defaults.capture_verify
        && check_override("capture_verify")
    {
        args.capture_verify = capture_verify;
        tracing::info!(capture_verify, "applied recipe default for capture_verify");
    }

    if let Some(capture_batch) = defaults.capture_batch
        && check_override("capture_batch")
    {
        args.capture_batch = capture_batch;
        tracing::info!(capture_batch, "applied recipe default for capture_batch");
    }

    // An `Option` flag has no default value source, so "not given" is `None`.
    if let Some(fraction) = defaults.gpu_memory_utilization
        && args.gpu_memory_utilization.is_none()
    {
        args.gpu_memory_utilization = Some(fraction);
        tracing::info!(
            fraction,
            "applied recipe default for gpu_memory_utilization"
        );
    }

    if let Some(chunk) = defaults.solo_prefill_chunk
        && check_override("solo_prefill_chunk")
    {
        args.solo_prefill_chunk = chunk;
        tracing::info!(chunk, "applied recipe default for solo_prefill_chunk");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::DEFAULT_CTX;

    fn recipe() -> ServerDefaults {
        ServerDefaults {
            ctx: Some(32768),
            max_num_seqs: Some(64),
            ..ServerDefaults::default()
        }
    }

    #[test]
    fn recipe_defaults_never_override_an_explicitly_set_option() {
        let mut options = ServeOptions {
            ctx: 16384,
            ..ServeOptions::default()
        };
        let explicit: ExplicitlySet = ["ctx"].into_iter().collect();
        apply_recipe_defaults_unless_set(&mut options, &explicit, &recipe());
        assert_eq!(options.ctx, 16384, "the explicit value wins");
        assert_eq!(
            options.max_num_seqs, 64,
            "an untouched one takes the recipe"
        );
    }

    #[test]
    fn an_embedder_that_sets_nothing_explicitly_gets_the_recipe() {
        let mut options = ServeOptions::default();
        apply_recipe_defaults_unless_set(&mut options, &ExplicitlySet::default(), &recipe());
        assert_eq!(options.ctx, 32768);
    }

    /// An option set to the very value the built-in default has is still the
    /// caller's choice: the recipe must not move it.
    #[test]
    fn an_explicit_value_equal_to_the_default_still_beats_the_recipe() {
        let mut options = ServeOptions {
            ctx: DEFAULT_CTX,
            ..ServeOptions::default()
        };
        let explicit: ExplicitlySet = ["ctx"].into_iter().collect();
        apply_recipe_defaults_unless_set(&mut options, &explicit, &recipe());
        assert_eq!(options.ctx, DEFAULT_CTX);
        assert_eq!(options.max_num_seqs, 64);
    }
}

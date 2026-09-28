//! The one path from a neutral request to a stream of events.
//!
//! # Why every surface goes through here
//!
//! Three wire formats reach this module and none of them reaches the engine any
//! other way. Rendering, the context guard, the sampling preset and the token
//! budget are all decisions that have nothing to do with which format asked, and
//! a surface that made one of them itself would be a second answer to a question
//! that already has one - which is how two endpoints on the same server end up
//! serving subtly different models.
//!
//! What a surface still owns is everything before and after: parsing its body
//! into a [`Request`], and encoding what comes back.
//!
//! # What the engine is not told
//!
//! The sampler behind the queue has four stages. A caller may name eight knobs,
//! and the four with no stage - the sigma cutoff and the three penalties - are
//! dropped with a log line rather than silently approximated by one of the
//! stages that does exist. A penalty quietly implemented as a temperature is a
//! server answering a different question than the one it was asked.

use std::sync::Arc;

use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_ir::{EventStream, Request};

use crate::error::AppError;
use crate::ids;
use crate::scheduler::GenerateRequest;
use crate::state::AppState;

/// A queued generation, and what is already known about it.
///
/// The prompt length rides back with the stream because two surfaces have to
/// state it *before* the turn ends: one opens its stream with an accounting
/// block, and the other reports it in a header frame. Deriving it a second time
/// at the edge would mean rendering the prompt twice, and a server whose opening
/// frame and closing frame disagree about how long the prompt was is a server
/// no client can bill against.
///
/// `Debug` is written out rather than derived: an [`EventStream`] is a boxed
/// trait object with no `Debug` of its own, and a value whose `Debug` shows the
/// accounting but not the stream is more useful in a log than no `Debug` at all.
pub struct Started {
    /// The events the turn will produce.
    pub events: EventStream,
    /// Tokens the rendered prompt occupies.
    pub prompt_tokens: u64,
}

impl std::fmt::Debug for Started {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Started")
            .field("prompt_tokens", &self.prompt_tokens)
            .finish_non_exhaustive()
    }
}

/// The wire field that carried the prompt.
///
/// Carried as an enum rather than a string so that a complaint about a request
/// cannot name a field the surface it came from does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptField {
    /// The chat-completions conversation.
    Messages,
    /// The legacy completions prompt.
    Prompt,
    /// The Responses input.
    Input,
}

impl PromptField {
    /// The field's name, as its own surface spells it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::Prompt => "prompt",
            Self::Input => "input",
        }
    }
}

/// The fallback preset, for a checkpoint that ships no generation config.
///
/// Neutral rather than opinionated: every stage is a no-op and the temperature
/// is one, so an unset knob is the model's own distribution and not this
/// server's taste. A checkpoint that does ship one gets
/// [`preset_from_generation_config`] instead, which is what the published model
/// is served with.
#[must_use]
pub const fn preset() -> SamplingParams {
    SamplingParams::neutral()
}

/// The sampling a checkpoint's `generation_config.json` asks for.
///
/// This is what the published model is served with when a caller names nothing,
/// so it is the preset rather than [`SamplingParams::neutral`]: a neutral preset
/// samples temperature one over the whole vocabulary, and the tail it keeps is
/// where malformed tool tags come from. Keys the file does not name keep the
/// neutral value. `do_sample: false` is greedy.
///
/// # Errors
///
/// The parse error, for a file that is not a JSON object of the expected types.
pub fn preset_from_generation_config(text: &str) -> Result<SamplingParams, serde_json::Error> {
    #[derive(serde::Deserialize)]
    struct Card {
        #[serde(default)]
        do_sample: Option<bool>,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        top_k: Option<u32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        min_p: Option<f32>,
    }
    let value: serde_json::Value = serde_json::from_str(text)?;
    // A struct also deserializes from a JSON array, field by field; a card that
    // is a list is not a card, whatever its members happen to be.
    if !value.is_object() {
        return Err(<serde_json::Error as serde::de::Error>::custom(
            "a generation config is a JSON object",
        ));
    }
    let card: Card = serde_json::from_value(value)?;
    let mut preset = SamplingParams::neutral();
    if let Some(temperature) = card.temperature {
        preset.temperature = temperature;
    }
    if let Some(top_k) = card.top_k {
        // Zero is "no top-k stage" here as on the wire.
        preset.top_k = std::num::NonZeroU32::new(top_k);
    }
    if let Some(top_p) = card.top_p {
        preset.top_p = top_p;
    }
    if let Some(min_p) = card.min_p {
        preset.min_p = min_p;
    }
    if card.do_sample == Some(false) {
        preset.temperature = 0.0;
    }
    Ok(preset)
}

/// Resolves what the caller asked for against the preset.
///
/// Returns the settings the sampler runs and the stream it draws from. The two
/// are returned together because a seed without the settings it was drawn for
/// cannot reproduce anything. Each knob the caller named overrides the preset's
/// one by one, and the rest of the preset stands: a caller who sends only a
/// temperature still gets the checkpoint's top-k.
#[must_use]
pub fn resolve_sampling(request: &Request, preset: SamplingParams) -> (SamplingParams, Seed) {
    let asked = request.sampling;
    // Built by mutation rather than as a literal: the engine's settings type is
    // `#[non_exhaustive]`, so a struct expression for it is not available from
    // this crate. Every field the caller left silent therefore keeps the preset
    // value it starts with, which is the behaviour a literal would have had to
    // spell out anyway.
    let mut resolved = preset;
    if let Some(temperature) = asked.temperature {
        resolved.temperature = temperature;
    }
    if let Some(top_p) = asked.top_p {
        resolved.top_p = top_p;
    }
    if let Some(min_p) = asked.min_p {
        resolved.min_p = min_p;
    }
    if let Some(top_k) = asked.top_k {
        // Zero is the wire's way of spelling "no top-k stage", which is what
        // `None` means here. The two spellings meet at the one place that knows
        // the wire uses a sentinel.
        resolved.top_k = std::num::NonZeroU32::new(top_k);
    }
    warn_about_unserved_knobs(request);
    let seed = Seed::new(request.seed.unwrap_or_else(ids::entropy_u64));
    (resolved, seed)
}

/// Says out loud which knobs the sampler has no stage for.
fn warn_about_unserved_knobs(request: &Request) {
    let asked = request.sampling;
    let unserved = [
        ("top_n_sigma", asked.top_n_sigma.is_some()),
        ("repetition_penalty", asked.repetition_penalty.is_some()),
        ("presence_penalty", asked.presence_penalty.is_some()),
        ("frequency_penalty", asked.frequency_penalty.is_some()),
        ("logit_bias", !request.logit_bias.is_empty()),
        ("min_tokens", request.min_tokens > 0),
        ("top_logprobs", request.top_logprobs.is_some()),
        ("response_format", request.response_format.is_some()),
    ];
    for (name, asked_for) in unserved {
        if asked_for {
            tracing::debug!(
                knob = name,
                "the sampler behind this queue has no stage for this setting; \
                 it is being ignored rather than approximated"
            );
        }
    }
}

/// Renders, guards the context, and queues the generation.
///
/// Answers as soon as the engine has accepted the request, which is before the
/// first token exists: what is awaited here is admission, not completion. The
/// caller decides whether to fold the stream into one response or to encode it
/// frame by frame.
///
/// # Errors
///
/// [`AppError::NotReady`] when no prompt renderer was built,
/// [`AppError::ContextOverflow`] when the rendered prompt leaves no room to
/// generate, whatever [`crate::tokenizer::render_prompt::PromptRenderer::render`]
/// reports for a conversation this server will not render, and
/// [`AppError::Internal`] when the engine queue is gone.
pub async fn start(
    state: &AppState,
    request: &Request,
    field: PromptField,
) -> Result<Started, AppError> {
    // A server with no language model has no renderer either, and "no model is
    // loaded" is the answer that says what to change.
    state.engine()?;
    let renderer = state.renderer.as_ref().ok_or(AppError::NotReady {
        what: "a prompt renderer",
    })?;
    let prompt = renderer.render(request)?;
    let text: Arc<str> = Arc::from(prompt.text);
    let media = (prompt.media, prompt.digests);
    queue(state, request, prompt.tokens, media, text, field).await
}

/// Queues a generation over a prompt that is already tokenized.
///
/// The legacy completions surface reaches the engine this way: its prompt is
/// raw text, or raw ids, and running it through a chat template would wrap it in
/// turn markers the caller never asked for. Everything after the prompt - the
/// context guard, the budget, the preset - is identical, which is why it is this
/// function that both entries share rather than two copies of it.
///
/// # Errors
///
/// [`AppError::ContextOverflow`] when the prompt leaves no room to generate, and
/// [`AppError::Internal`] when the engine queue is gone.
pub async fn start_with_prompt(
    state: &AppState,
    request: &Request,
    tokens: Vec<TokenId>,
    media: Vec<crate::media::MediaItem>,
    field: PromptField,
) -> Result<Started, AppError> {
    // No rendered text, so nothing keys the checkpoint tier and this prompt is
    // always a cold start. Decoding the ids back into text to get a key would
    // produce a string that is *nearly* the prompt, and a cache keyed on nearly
    // the prompt occasionally answers with somebody else's conversation.
    let digests = media.iter().map(crate::media::MediaItem::digest).collect();
    queue(
        state,
        request,
        tokens,
        (media, digests),
        Arc::from(""),
        field,
    )
    .await
}

/// The half both entries share: the context guard, the budget, the preset.
async fn queue(
    state: &AppState,
    request: &Request,
    tokens: Vec<TokenId>,
    (media, digests): (
        Vec<crate::media::MediaItem>,
        Vec<shiso_core::media::MediaDigest>,
    ),
    text: Arc<str>,
    field: PromptField,
) -> Result<Started, AppError> {
    let prompt_tokens = u64::try_from(tokens.len()).unwrap_or(u64::MAX);
    let budget = remaining_budget(state.ctx, prompt_tokens).ok_or(AppError::ContextOverflow {
        param: field.as_str(),
        prompt_tokens,
        ctx: state.ctx,
    })?;
    // Zero is what the IR carries when the caller named no cap, because the
    // field is a resolved count rather than an option. Resolving it needs the
    // prompt length, which is why it happens here and not at the edge.
    let max_tokens = if request.max_tokens == 0 {
        budget
    } else {
        request.max_tokens.min(budget)
    };

    // A request that will wait for a seat warms the page cache for the stored
    // prefix it will resume from, so its restore reads memory rather than the
    // drive. Only then: one admitted at once resumes straight away, usually from
    // the tree, and reading its files would be traffic for nothing.
    let engine = state.engine()?;
    let seats = u64::try_from(engine.max_num_seqs()).unwrap_or(u64::MAX);
    if media.is_empty()
        && state.sched.active() >= seats
        && let Some(prefetcher) = state.kv.as_ref().and_then(|kv| kv.prefetcher())
    {
        let started = prefetcher.prefetch(&tokens);
        tracing::trace!(started, "prefetch for a queued request");
    }
    let (sampling, seed) = resolve_sampling(request, state.sampling_preset);
    let generate = GenerateRequest::new(Arc::from(tokens), sampling, seed, max_tokens)
        .with_text(text)
        .with_stop(Arc::from(request.stop.clone()))
        .with_media(Arc::from(media), Arc::from(digests))
        .with_service_class(request.service_class)
        .with_string_params(Arc::new(shiso_parsers::StringParams::from_schemas(
            request
                .tools
                .iter()
                .map(|tool| (tool.name.as_str(), &tool.input_schema)),
        )))
        .with_tool_scaffold(Arc::new(shiso_parsers::ToolScaffold::from_schemas(
            request
                .tools
                .iter()
                .map(|tool| (tool.name.as_str(), &tool.input_schema)),
        )));
    let events = engine.generate(generate).await?;
    Ok(Started {
        events,
        prompt_tokens,
    })
}

/// How many tokens are left for generation, or `None` when the prompt already
/// fills the context.
///
/// The boundary is deliberately "no room at all" rather than "the prompt is
/// longer than the context": a prompt that exactly fills the window has nowhere
/// to put a first token, and answering it with an empty turn is a 200 that says
/// nothing went wrong.
fn remaining_budget(ctx: u32, prompt_tokens: u64) -> Option<u32> {
    let room = u64::from(ctx).checked_sub(prompt_tokens)?;
    let room = u32::try_from(room).unwrap_or(u32::MAX);
    (room > 0).then_some(room)
}

#[cfg(test)]
mod tests {
    use super::{PromptField, preset, remaining_budget, resolve_sampling};
    use shiso_ir::{Message, Request, Sampling};

    fn request() -> Request {
        Request::new("a-model", vec![Message::user("hi")])
    }

    #[test]
    fn a_caller_who_named_nothing_gets_the_neutral_preset() {
        let (sampling, _seed) = resolve_sampling(&request(), preset());
        assert!((sampling.temperature - 1.0).abs() < f32::EPSILON);
        assert_eq!(sampling.top_k, None);
        assert!((sampling.top_p - 1.0).abs() < f32::EPSILON);
    }

    /// Zero temperature is a request for argmax, and it must survive being
    /// "unset-looking": the preset may only fill in what the caller left silent.
    #[test]
    fn a_temperature_of_zero_is_a_choice_and_not_a_silence() {
        let mut request = request();
        request.sampling = Sampling {
            temperature: Some(0.0),
            ..Sampling::default()
        };
        let (sampling, _seed) = resolve_sampling(&request, preset());
        assert!(sampling.is_greedy());
    }

    /// The wire spells "no top-k stage" as zero, and the engine spells it as
    /// absent. One of the two has to translate, and it is this one.
    #[test]
    fn a_top_k_of_zero_turns_the_stage_off_rather_than_keeping_zero_candidates() {
        let mut request = request();
        request.sampling = Sampling {
            top_k: Some(0),
            ..Sampling::default()
        };
        let (sampling, _seed) = resolve_sampling(&request, preset());
        assert_eq!(sampling.top_k, None);
    }

    #[test]
    fn a_named_seed_is_the_seed_and_a_silent_one_is_not_zero() {
        let mut request = request();
        request.seed = Some(4242);
        let (_sampling, seed) = resolve_sampling(&request, preset());
        assert_eq!(seed.get(), 4242);

        let (_sampling, fresh) = resolve_sampling(&self::request(), preset());
        let (_sampling, again) = resolve_sampling(&self::request(), preset());
        assert_ne!(
            fresh.get(),
            again.get(),
            "a silent seed is fresh entropy, not a constant"
        );
    }

    /// The checkpoint card the sparse family ships, verbatim in its sampling
    /// keys.
    const CARD: &str = r#"{"do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "eos_token_id": [248046, 248044]}"#;

    #[test]
    fn the_checkpoint_card_becomes_the_preset() {
        let preset = super::preset_from_generation_config(CARD).expect("parses");
        assert!((preset.temperature - 1.0).abs() < f32::EPSILON);
        assert_eq!(preset.top_k.map(std::num::NonZeroU32::get), Some(20));
        assert!((preset.top_p - 0.95).abs() < 1e-6);
    }

    /// Each knob the caller names overrides the preset one by one; the rest of
    /// the preset stands, so naming a temperature does not reopen the tail.
    #[test]
    fn a_request_naming_only_temperature_keeps_the_preset_top_k() {
        let preset = super::preset_from_generation_config(CARD).expect("parses");
        let mut request = request();
        request.sampling.temperature = Some(0.7);
        let (sampling, _seed) = resolve_sampling(&request, preset);
        assert!((sampling.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(sampling.top_k.map(std::num::NonZeroU32::get), Some(20));
    }

    #[test]
    fn do_sample_false_makes_the_preset_greedy() {
        let preset =
            super::preset_from_generation_config(r#"{"do_sample": false, "temperature": 0.6}"#)
                .expect("parses");
        assert!(preset.is_greedy());
    }

    #[test]
    fn a_card_that_is_not_a_json_object_is_refused() {
        assert!(super::preset_from_generation_config("[1, 2]").is_err());
        assert!(super::preset_from_generation_config(r#"{"top_k": "twenty"}"#).is_err());
    }

    #[test]
    fn a_prompt_that_fills_the_window_leaves_no_room_to_answer() {
        assert_eq!(remaining_budget(100, 40), Some(60));
        assert_eq!(remaining_budget(100, 100), None);
        assert_eq!(remaining_budget(100, 101), None);
    }

    #[test]
    fn each_surface_blames_the_field_it_actually_has() {
        assert_eq!(PromptField::Messages.as_str(), "messages");
        assert_eq!(PromptField::Prompt.as_str(), "prompt");
        assert_eq!(PromptField::Input.as_str(), "input");
    }
}

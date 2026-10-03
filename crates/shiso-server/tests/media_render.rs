//! Media: what reaches the engine, in what order, and what is refused.
//!
//! The geometry comes from the recorded checkpoint config rather than from
//! numbers written here, because a hand-written geometry would let these tests
//! keep passing after the real one changed - and a grid that disagrees with the
//! tower's is the one media failure that produces a fluent answer about the
//! wrong picture rather than an error.

use std::str::FromStr as _;
use std::sync::Arc;

use base64::Engine as _;
use serde_json::{Value, json};
use shiso_core::{HfConfig, PadTokens, TokenId, VisionConfig};
use shiso_ir::{Block, MediaSource, Message, Request, Role};
use shiso_server::tokenizer::Vocabulary;
use shiso_server::tokenizer::jinja_helpers::ToolJsonStyle;
use shiso_server::tokenizer::message_preprocess::{
    MediaPolicy, MediaWraps, PreprocessError, expand_media_pads, expand_vision_pads, preprocess,
};
use shiso_server::tokenizer::render_prompt::PromptRenderer;
use shiso_vision::{ImageLimits, MediaPlacement, PatchGeometry, PreprocessSpec};

/// The recorded checkpoint config, which is where the tower geometry and the
/// media token ids come from.
const RECORDED_CONFIG: &str = include_str!("../../../test-vectors/loader/qwen35-config.json");

/// The vision block of the recorded config.
fn vision() -> VisionConfig {
    HfConfig::from_json_str(RECORDED_CONFIG)
        .expect("the recorded config parses")
        .vision_config()
        .expect("the vision block parses")
        .expect("the recorded config carries a vision block")
}

/// A solid-colour PNG, as the data URI a request would carry.
fn png_data_uri(width: u32, height: u32) -> String {
    let image = image::RgbImage::from_pixel(width, height, image::Rgb([40, 90, 160]));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("a solid-colour PNG encodes");
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    )
}

/// The preprocessing spec the recorded checkpoint implies.
fn spec() -> PreprocessSpec {
    PreprocessSpec::from_vision_config(&vision())
        .expect("the recorded geometry drives a preprocessor")
}

/// A policy with the tower attached and every optional capability off.
fn policy(spec: &PreprocessSpec) -> MediaPolicy<'_> {
    MediaPolicy {
        vision: Some(spec),
        limits: ImageLimits::default(),
        ..MediaPolicy::default()
    }
}

#[test]
fn an_inline_image_becomes_one_item_whose_pad_run_matches_its_grid() {
    let spec = spec();
    let message = Message::new(
        Role::User,
        vec![Block::Image(MediaSource::classify(png_data_uri(64, 64)))],
    );
    let prepared =
        preprocess(&[], std::slice::from_ref(&message), &policy(&spec)).expect("it preprocesses");

    assert_eq!(prepared.media.len(), 1);
    assert_eq!(prepared.pad_counts.len(), 1);
    let merge = spec.geometry.spatial_merge_size();
    assert_eq!(
        prepared.pad_counts[0],
        prepared.media[0].pad_count(merge),
        "the pad run is the item's own count, never a second calculation"
    );
    assert!(prepared.pad_counts[0] > 0);
}

/// Text, then an image, then text: the parts come out in the order the turn put
/// them in, which is what a pair of counts cannot promise.
#[test]
fn the_content_parts_keep_the_order_the_turn_produced_them_in() {
    let spec = spec();
    let message = Message::new(
        Role::User,
        vec![
            Block::Text("before".to_owned()),
            Block::Image(MediaSource::classify(png_data_uri(32, 32))),
            Block::Text("after".to_owned()),
        ],
    );
    let prepared =
        preprocess(&[], std::slice::from_ref(&message), &policy(&spec)).expect("it preprocesses");

    let turn = prepared.messages.first().expect("one turn was rendered");
    let parts = turn["content"].as_array().expect("the content is a list");
    let kinds: Vec<&str> = parts
        .iter()
        .filter_map(|part| part["type"].as_str())
        .collect();
    assert_eq!(kinds, vec!["text", "image", "text"]);
    assert_eq!(parts[0]["text"], "before");
    assert_eq!(parts[2]["text"], "after");
}

/// Instructions are a field on the request, not a turn, so they are rendered as
/// the leading system turn here and nowhere else.
#[test]
fn the_system_instructions_become_the_leading_turn() {
    let spec = spec();
    let prepared = preprocess(
        &[Block::Text("be brief".to_owned())],
        &[Message::user("hello")],
        &policy(&spec),
    )
    .expect("it preprocesses");
    assert_eq!(prepared.messages.len(), 2);
    assert_eq!(prepared.messages[0]["role"], "system");
    assert_eq!(prepared.messages[1]["role"], "user");
}

// The tool-result turn split has no test here, and cannot have one yet:
// `shiso_ir::ToolResult` is `#[non_exhaustive]` with no constructor, so a block
// carrying one cannot be built outside the IR crate at all. The split itself is
// implemented and is what the adapter tasks need; the missing constructor is
// recorded as an IR gap rather than worked around with a second shape here.

/// A remote address is refused before any socket is opened. The assertion that
/// nothing was fetched is structural: this test has no network available to it
/// and the refusal is reached without one.
#[test]
fn a_remote_address_is_refused_without_a_network_call() {
    let spec = spec();
    let message = Message::new(
        Role::User,
        vec![Block::Image(MediaSource::classify(
            "https://example.test/cat.png",
        ))],
    );
    let error = preprocess(&[], std::slice::from_ref(&message), &policy(&spec))
        .expect_err("remote media is off by default");
    assert!(
        matches!(error, PreprocessError::RemoteRefused { .. }),
        "got {error:?}"
    );
    assert!(
        error.to_string().contains("--vision-allow-remote-images"),
        "the refusal names the flag that would change it: {error}"
    );
}

/// With the flag on, the answer is still no, and it says why: nothing in this
/// build talks outbound.
#[test]
fn allowing_remote_media_reports_that_this_build_cannot_fetch_it() {
    let spec = spec();
    let allowing = MediaPolicy {
        allow_remote: true,
        ..policy(&spec)
    };
    let message = Message::new(
        Role::User,
        vec![Block::Image(MediaSource::classify(
            "https://example.test/cat.png",
        ))],
    );
    let error = preprocess(&[], std::slice::from_ref(&message), &allowing)
        .expect_err("this build has no outbound fetch");
    assert!(
        matches!(error, PreprocessError::RemoteUnsupported { .. }),
        "got {error:?}"
    );
}

/// Video decoding is off unless an operator turned it on, and the refusal is
/// the media layer's own, not a message invented here.
#[test]
fn a_video_is_refused_when_subprocess_decoding_is_off() {
    let spec = spec();
    let message = Message::new(
        Role::User,
        // Not a real container; the policy gate is reached before any decode.
        vec![Block::Video(MediaSource::Inline(
            base64::engine::general_purpose::STANDARD
                .encode([0u8, 0, 0, 0x20, b'f', b't', b'y', b'p']),
        ))],
    );
    let error = preprocess(&[], std::slice::from_ref(&message), &policy(&spec))
        .expect_err("video decoding is off");
    assert!(
        error.to_string().contains("--video-allow-ffmpeg"),
        "the refusal names the flag that would change it: {error}"
    );
}

/// A checkpoint with no tower cannot be sent a picture, and says so rather than
/// decoding one and dropping it.
#[test]
fn a_text_only_checkpoint_refuses_media_rather_than_ignoring_it() {
    let message = Message::new(
        Role::User,
        vec![Block::Image(MediaSource::classify(png_data_uri(16, 16)))],
    );
    let error = preprocess(&[], std::slice::from_ref(&message), &MediaPolicy::default())
        .expect_err("a text-only model has no tower");
    assert!(
        matches!(error, PreprocessError::NoVisionTower { .. }),
        "got {error:?}"
    );
}

#[test]
fn a_payload_that_is_not_base64_is_reported_as_such() {
    let spec = spec();
    let message = Message::new(
        Role::User,
        vec![Block::Image(MediaSource::Inline("not base64!!".to_owned()))],
    );
    let error = preprocess(&[], std::slice::from_ref(&message), &policy(&spec))
        .expect_err("the payload does not decode");
    assert!(matches!(error, PreprocessError::NotBase64), "got {error:?}");
}

/// A turn with no media renders without touching the media path at all: no
/// items, no pad counts, and nothing for the pad expansion to do.
#[test]
fn a_text_only_turn_produces_no_media_and_no_pad_expansion() {
    let prepared = preprocess(&[], &[Message::user("just words")], &MediaPolicy::default())
        .expect("text alone cannot fail");
    assert!(prepared.media.is_empty());
    assert!(prepared.pad_counts.is_empty());

    let tokens: Vec<TokenId> = (1..5).map(TokenId::new).collect();
    let expanded = expand_vision_pads(
        tokens.clone(),
        &[],
        PadTokens::new(Some(TokenId::new(9)), None),
        None,
    )
    .expect("nothing to expand");
    assert_eq!(
        expanded, tokens,
        "the token run is passed through untouched"
    );
}

/// One marker per item goes in; one run per item comes out, and the tokens
/// around them are undisturbed.
#[test]
fn each_pad_marker_expands_to_the_run_its_item_needs() {
    let pad = TokenId::new(9);
    let tokens = vec![TokenId::new(1), pad, TokenId::new(2), pad, TokenId::new(3)];
    let expanded = expand_vision_pads(tokens, &[3, 2], PadTokens::new(Some(pad), None), None)
        .expect("the counts match");
    let ids: Vec<u32> = expanded.iter().map(|token| token.get()).collect();
    assert_eq!(ids, vec![1, 9, 9, 9, 2, 9, 9, 3]);
}

/// A marker count that disagrees with the item count means the template and the
/// preprocessor disagree about the conversation. Guessing which is right
/// produces a prompt that is silently about the wrong picture.
#[test]
fn a_marker_count_that_disagrees_with_the_item_count_is_refused() {
    let pad = TokenId::new(9);
    let error = expand_vision_pads(
        vec![TokenId::new(1), pad],
        &[3, 2],
        PadTokens::new(Some(pad), None),
        None,
    )
    .expect_err("one marker, two items");
    assert_eq!(error.markers, 1);
    assert_eq!(error.items, 2);
}

/// The video marker expands the same way the image marker does, so a turn that
/// mixes the two keeps both runs in place.
#[test]
fn image_and_video_markers_both_expand() {
    let image_pad = TokenId::new(9);
    let video_pad = TokenId::new(10);
    let expanded = expand_vision_pads(
        vec![image_pad, TokenId::new(1), video_pad],
        &[2, 3],
        PadTokens::new(Some(image_pad), Some(video_pad)),
        None,
    )
    .expect("the counts match");
    let ids: Vec<u32> = expanded.iter().map(|token| token.get()).collect();
    assert_eq!(ids, vec![9, 9, 1, 10, 10, 10]);
}

/// An image and two audio clips expand in prompt order, each bracketed by its
/// own pair, and the audio pad is matched although it is neither the image nor
/// the video marker.
#[test]
fn image_and_audio_markers_expand_in_order_with_their_own_brackets() {
    let (image, audio) = (TokenId::new(9), TokenId::new(11));
    let wraps = MediaWraps {
        image: Some((TokenId::new(20), TokenId::new(21))),
        audio: Some((TokenId::new(30), TokenId::new(31))),
    };
    let pads = PadTokens::new(Some(image), None).with_audio(Some(audio));
    assert!(pads.matches(audio));
    let (expanded, _ranges) = expand_media_pads(
        vec![image, TokenId::new(1), audio, audio],
        &[2, 3, 1],
        pads,
        wraps,
    )
    .expect("the counts match");
    let ids: Vec<u32> = expanded.iter().map(|token| token.get()).collect();
    assert_eq!(ids, vec![20, 9, 9, 21, 1, 30, 11, 11, 11, 31, 30, 11, 31]);
}

/// Gemma 4's placement declares the audio marker and its bracket; Qwen's
/// declares neither.
#[test]
fn the_family_placements_declare_audio_where_the_family_hears() {
    assert_eq!(MediaPlacement::GEMMA4.audio_marker, Some("<|audio|>"));
    assert_eq!(
        MediaPlacement::GEMMA4.wrap_audio,
        Some(("<|audio>", "<audio|>"))
    );
    assert_eq!(MediaPlacement::QWEN35.audio_marker, None);
}

// --- the bracketed placement -----------------------------------------------

/// The bracket ids the fake vocabulary below gives Gemma's two wrap markers.
const OPEN: u32 = 2;
/// The image marker's id in that vocabulary.
const MARKER: u32 = 3;
/// The closing bracket's id.
const CLOSE: u32 = 4;

/// A template that writes one marker per image and nothing else around it.
///
/// Deliberately tiny: the assertion is about what the EXPANSION does to the
/// marker, and a real family template would bury the three ids being asserted
/// on under a hundred of its own.
const WRAPPING_TEMPLATE: &str = concat!(
    "{% for m in messages %}{% for part in m.content %}",
    "{% if part.type == 'image' %}<|image|>{% else %}{{ part.text }}{% endif %}",
    "{% endfor %}{% endfor %}",
);

/// A vocabulary that declares Gemma's three media markers, as added tokens.
///
/// Added tokens rather than ordinary vocabulary entries because that is how a
/// real checkpoint declares them: the library matches them before any
/// pre-tokenization, so a marker survives being written into prompt text
/// whatever the surrounding punctuation is.
fn wrapping_vocabulary() -> Arc<Vocabulary> {
    let markers = [
        (
            OPEN,
            MediaPlacement::GEMMA4.wrap_image.expect("gemma wraps").0,
        ),
        (MARKER, MediaPlacement::GEMMA4.image_marker),
        (
            CLOSE,
            MediaPlacement::GEMMA4.wrap_image.expect("gemma wraps").1,
        ),
    ];
    let mut vocab = serde_json::Map::new();
    vocab.insert("<unk>".to_owned(), json!(0));
    vocab.insert("word".to_owned(), json!(1));
    let added: Vec<Value> = markers
        .iter()
        .map(|(id, content)| {
            vocab.insert((*content).to_owned(), json!(id));
            json!({
                "id": id,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();
    let document = json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": added,
        "normalizer": Value::Null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": Value::Null,
        "decoder": Value::Null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    })
    .to_string();
    let inner = tokenizers::Tokenizer::from_str(&document).expect("the vocabulary loads");
    Arc::new(Vocabulary::from_tokenizer(inner))
}

/// A 16-pixel patch pooled 3x3, sized by the grid-unit rule.
///
/// Gemma's pooler over a rule that leaves a 48x48 picture alone, so the picture
/// is exactly one soft token and the whole media run is three ids - every one
/// of them asserted on rather than counted. Gemma's own rule would spend its
/// patch budget by UPSCALING the same picture to 256 tokens, which proves the
/// same thing about the bracket behind a run nobody can read.
fn one_token_spec() -> PreprocessSpec {
    PreprocessSpec::qwen35(PatchGeometry::new(16, 1, 3).expect("a 3x3-pooled 16-pixel patch"))
}

/// A family that brackets its run gets the bracket, around a run whose length
/// is still the item's own pad count.
///
/// The picture is 48x48 against Gemma's 16-pixel patch and 3x3 pooler, which is
/// exactly one soft token - so the whole media run is three ids and every one
/// of them is asserted on rather than counted.
#[test]
fn a_wrapping_placement_brackets_the_pad_run() {
    let vocabulary = wrapping_vocabulary();
    let renderer = PromptRenderer::new(WRAPPING_TEMPLATE, ToolJsonStyle::Compact, vocabulary)
        .expect("the tiny template compiles")
        .with_tower(
            &one_token_spec(),
            MediaPlacement::GEMMA4,
            ImageLimits::default(),
        );
    let request = Request::new(
        "test",
        vec![Message::new(
            Role::User,
            vec![Block::Image(MediaSource::classify(png_data_uri(48, 48)))],
        )],
    );

    let prompt = renderer.render(&request).expect("the prompt renders");
    assert_eq!(prompt.media.len(), 1);
    assert_eq!(
        prompt.media[0].pad_count(one_token_spec().geometry.spatial_merge_size()),
        1,
        "a 48x48 picture is one soft token under this geometry"
    );
    let ids: Vec<u32> = prompt.tokens.iter().map(|token| token.get()).collect();
    assert_eq!(
        ids,
        vec![OPEN, MARKER, CLOSE],
        "the run is bracketed, and the bracket is outside it"
    );
}

/// The same expansion without a bracket is the run alone: a family that writes
/// no bracket must not acquire one, because two tokens the model never saw in
/// this position are invisible in the output.
#[test]
fn an_unwrapped_placement_renders_the_run_alone() {
    let pad = TokenId::new(MARKER);
    let expanded = expand_vision_pads(vec![pad], &[3], PadTokens::new(Some(pad), None), None)
        .expect("the counts match");
    let ids: Vec<u32> = expanded.iter().map(|token| token.get()).collect();
    assert_eq!(ids, vec![MARKER, MARKER, MARKER]);
}

/// The bracket goes around the image run only. A clip in the same prompt keeps
/// its bare run, because no family that brackets has a clip marker at all and
/// inventing one would be a prompt shape no checkpoint was trained on.
#[test]
fn the_bracket_is_written_around_the_image_run_only() {
    let image = TokenId::new(MARKER);
    let video = TokenId::new(9);
    let expanded = expand_vision_pads(
        vec![image, TokenId::new(1), video],
        &[2, 2],
        PadTokens::new(Some(image), Some(video)),
        Some((TokenId::new(OPEN), TokenId::new(CLOSE))),
    )
    .expect("the counts match");
    let ids: Vec<u32> = expanded.iter().map(|token| token.get()).collect();
    assert_eq!(ids, vec![OPEN, MARKER, MARKER, CLOSE, 1, 9, 9]);
}

/// A tiny template writing Qwen's media marker.
const QWEN_WRAPPING_TEMPLATE: &str = concat!(
    "{% for m in messages %}{% for part in m.content %}",
    "{% if part.type == 'image' %}<|image_pad|>{% else %}{{ part.text }}{% endif %}",
    "{% endfor %}{% endfor %}",
);

/// The turn opener's id in the Qwen-shaped vocabulary.
const IM_START: u32 = 6;
/// The turn closer's id in the Qwen-shaped vocabulary.
const IM_END: u32 = 7;

/// A vocabulary declaring Qwen's media marker and turn markers as special
/// added tokens.
fn qwen_vocabulary() -> Arc<Vocabulary> {
    let mut vocab = serde_json::Map::new();
    vocab.insert("<unk>".to_owned(), json!(0));
    vocab.insert("word".to_owned(), json!(1));
    vocab.insert(MediaPlacement::QWEN35.image_marker.to_owned(), json!(2));
    vocab.insert("<|".to_owned(), json!(3));
    vocab.insert("image_pad".to_owned(), json!(4));
    vocab.insert("|>".to_owned(), json!(5));
    let added: Vec<Value> = [
        (2, MediaPlacement::QWEN35.image_marker),
        (IM_START, "<|im_start|>"),
        (IM_END, "<|im_end|>"),
    ]
    .into_iter()
    .map(|(id, content)| {
        vocab.insert(content.to_owned(), json!(id));
        json!({
            "id": id,
            "content": content,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true,
        })
    })
    .collect();
    let document = json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": added,
        "normalizer": Value::Null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": Value::Null,
        "decoder": Value::Null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    })
    .to_string();
    let inner = tokenizers::Tokenizer::from_str(&document).expect("the vocabulary loads");
    Arc::new(Vocabulary::from_tokenizer(inner))
}

/// The Qwen renderer, with a one-token tower attached.
fn qwen_renderer() -> PromptRenderer {
    PromptRenderer::new(
        QWEN_WRAPPING_TEMPLATE,
        ToolJsonStyle::Compact,
        qwen_vocabulary(),
    )
    .expect("the tiny template compiles")
    .with_tower(
        &one_token_spec(),
        MediaPlacement::QWEN35,
        ImageLimits::default(),
    )
}

/// A caller who writes a media marker into a message has written text, and
/// the model is sent that text: neither a refusal nor a placeholder with no
/// picture behind it, which the model reads as nothing at all.
#[test]
fn marker_text_in_a_message_is_encoded_as_text() {
    let renderer = PromptRenderer::new(
        WRAPPING_TEMPLATE,
        ToolJsonStyle::Compact,
        wrapping_vocabulary(),
    )
    .expect("the tiny template compiles")
    .with_tower(
        &one_token_spec(),
        MediaPlacement::GEMMA4,
        ImageLimits::default(),
    );
    let marker = MediaPlacement::GEMMA4.image_marker;
    let request = Request::new("test", vec![Message::user(format!("word {marker} word"))]);
    let prompt = renderer
        .render(&request)
        .expect("a prompt that spells a marker in its text renders");
    assert!(prompt.media.is_empty());
    assert_eq!(prompt.text, format!("word {marker} word"));
    let ids: Vec<u32> = prompt.tokens.iter().map(|token| token.get()).collect();
    assert!(!ids.contains(&MARKER), "{ids:?}");
    assert_eq!(ids.first(), Some(&1));
    assert_eq!(ids.last(), Some(&1));
}

/// The Qwen spelling, which is the report this guards: `<|image_pad|>` typed
/// into a message reaches the model as the pieces of that text.
#[test]
fn qwen_image_pad_text_is_encoded_as_its_pieces() {
    let request = Request::new(
        "test",
        vec![Message::user(format!(
            "word {} word",
            MediaPlacement::QWEN35.image_marker
        ))],
    );
    let prompt = qwen_renderer()
        .render(&request)
        .expect("a prompt that spells a marker in its text renders");
    assert!(prompt.media.is_empty());
    let ids: Vec<u32> = prompt.tokens.iter().map(|token| token.get()).collect();
    assert_eq!(ids, vec![1, 3, 4, 5, 1]);
}

/// Marker text beside an attached picture stays text, and the picture still
/// gets the one marker the template wrote for it.
#[test]
fn marker_text_beside_an_attached_picture_stays_text() {
    let request = Request::new(
        "test",
        vec![Message::new(
            Role::User,
            vec![
                Block::Image(MediaSource::classify(png_data_uri(48, 48))),
                Block::Text(format!("word {} word", MediaPlacement::QWEN35.image_marker)),
            ],
        )],
    );
    let prompt = qwen_renderer()
        .render(&request)
        .expect("one marker for one picture renders");
    assert_eq!(prompt.media.len(), 1);
    let ids: Vec<u32> = prompt.tokens.iter().map(|token| token.get()).collect();
    assert_eq!(ids, vec![2, 1, 3, 4, 5, 1]);
}

// --- the per-item media spans ------------------------------------------------

/// One image, unbracketed family: the span pairs the pad run in the tokens
/// with the marker string in the text, so a caller downstream (the decision
/// seam check) can find the tail after the item without ever seeing a marker.
#[test]
fn a_rendered_image_prompt_reports_its_media_spans() {
    let request = Request::new(
        "test",
        vec![Message::new(
            Role::User,
            vec![
                Block::Text("word".to_owned()),
                Block::Image(MediaSource::classify(png_data_uri(96, 96))),
            ],
        )],
    );
    let prompt = qwen_renderer()
        .render(&request)
        .expect("one picture renders");
    let pads = prompt.media[0].pad_count(one_token_spec().geometry.spatial_merge_size());
    assert_eq!(pads, 4, "a 96x96 picture is four soft tokens here");
    assert_eq!(prompt.media_spans.len(), 1);
    let span = &prompt.media_spans[0];
    assert_eq!(
        &prompt.text[span.text.clone()],
        MediaPlacement::QWEN35.image_marker
    );
    let ids: Vec<u32> = prompt.tokens.iter().map(|token| token.get()).collect();
    assert_eq!(
        &ids[span.tokens.clone()],
        &vec![2u32; pads],
        "the run is the pads"
    );
    // The spans end lines up with the text after the marker: `word` before,
    // nothing after.
    assert_eq!(&ids[..span.tokens.start], &[1]);
    assert_eq!(span.tokens.end, ids.len());
}

/// A bracketed family: the token span covers open bracket + pad run + close
/// bracket, while the text span is still just the marker string - the
/// brackets are added token-side only, which is what keeps the two tails
/// aligned for the seam check (spec §2).
#[test]
fn a_bracketed_family_span_includes_the_brackets() {
    let vocabulary = wrapping_vocabulary();
    let renderer = PromptRenderer::new(WRAPPING_TEMPLATE, ToolJsonStyle::Compact, vocabulary)
        .expect("the tiny template compiles")
        .with_tower(
            &one_token_spec(),
            MediaPlacement::GEMMA4,
            ImageLimits::default(),
        );
    let request = Request::new(
        "test",
        vec![Message::new(
            Role::User,
            vec![
                Block::Text("word".to_owned()),
                Block::Image(MediaSource::classify(png_data_uri(48, 48))),
            ],
        )],
    );
    let prompt = renderer.render(&request).expect("the prompt renders");
    assert_eq!(prompt.media_spans.len(), 1);
    let span = &prompt.media_spans[0];
    let ids: Vec<u32> = prompt.tokens.iter().map(|token| token.get()).collect();
    assert_eq!(
        &ids[span.tokens.clone()],
        &[OPEN, MARKER, CLOSE],
        "the span is bracket + run + bracket, one pad here"
    );
    assert_eq!(span.tokens.len(), 3, "pad_count + 2");
    assert_eq!(
        &prompt.text[span.text.clone()],
        MediaPlacement::GEMMA4.image_marker
    );
    assert_eq!(&ids[..span.tokens.start], &[1]);
    assert_eq!(span.tokens.end, ids.len());
}

/// Two images in one prompt (Review Focus 4): the spans pair index-for-index
/// with `media`, strictly increase in both ranges, and the text between the
/// first span's end and the second's start is exactly the template's
/// inter-marker text.
#[test]
fn two_images_have_ordered_spans() {
    let request = Request::new(
        "test",
        vec![Message::new(
            Role::User,
            vec![
                Block::Text("word".to_owned()),
                Block::Image(MediaSource::classify(png_data_uri(48, 48))),
                Block::Text("mid".to_owned()),
                Block::Image(MediaSource::classify(png_data_uri(48, 48))),
                Block::Text("tail".to_owned()),
            ],
        )],
    );
    let prompt = qwen_renderer()
        .render(&request)
        .expect("two pictures render");
    assert_eq!(prompt.media_spans.len(), 2);
    let (first, second) = (&prompt.media_spans[0], &prompt.media_spans[1]);
    assert!(first.text.end <= second.text.start, "{first:?} {second:?}");
    assert!(
        first.tokens.end <= second.tokens.start,
        "{first:?} {second:?}"
    );
    assert_eq!(&prompt.text[first.text.end..second.text.start], "mid");
    assert_eq!(&prompt.text[second.text.end..], "tail");
}

/// A prompt with no media has no spans - the field is empty, not absent, and
/// nothing downstream may read spans for a text-only prompt.
#[test]
fn a_text_only_render_has_no_spans() {
    let request = Request::new("test", vec![Message::user("word")]);
    let prompt = qwen_renderer()
        .render(&request)
        .expect("a text prompt renders");
    assert!(prompt.media.is_empty());
    assert!(prompt.media_spans.is_empty());
}

/// A template that writes tool descriptions, string call arguments raw and
/// tool results, as Qwen's does.
const QWEN_TOOL_TEMPLATE: &str = concat!(
    "{% for t in tools or [] %}{{ t.function.description }} {% endfor %}",
    "{% for m in messages %}",
    "{% for c in m.tool_calls or [] %}",
    "{% for k, v in c.function.arguments|items %}{{ v }} {% endfor %}",
    "{% endfor %}",
    "{% for part in m.content %}{{ part.text }} {% endfor %}",
    "{% endfor %}",
);

/// An agent editing a file that spells a marker sends it back in a call's
/// arguments, in the tool's result and sometimes in a tool's description.
/// Every one of them is text, so a prompt carrying them and no picture
/// renders, with no marker id in it.
#[test]
fn marker_text_in_tool_calls_results_and_definitions_stays_text() {
    let renderer = PromptRenderer::new(
        QWEN_TOOL_TEMPLATE,
        ToolJsonStyle::Compact,
        qwen_vocabulary(),
    )
    .expect("the tiny template compiles")
    .with_tower(
        &one_token_spec(),
        MediaPlacement::QWEN35,
        ImageLimits::default(),
    );
    let marker = MediaPlacement::QWEN35.image_marker;
    let mut request = Request::new(
        "test",
        vec![
            Message::new(
                Role::Assistant,
                vec![Block::ToolUse(shiso_ir::ToolUse::new(
                    "c1",
                    "write",
                    json!({"content": format!("word {marker}")}),
                ))],
            ),
            Message::new(
                Role::User,
                vec![Block::ToolResult(shiso_ir::ToolResult {
                    tool_use_id: "c1".to_owned(),
                    content: vec![Block::Text(format!("word {marker}"))],
                    is_error: false,
                })],
            ),
        ],
    );
    request.tools.push(shiso_ir::ToolDefinition {
        name: "write".to_owned(),
        description: Some(format!("word {marker}")),
        input_schema: json!({"type": "object"}),
    });
    let prompt = renderer
        .render(&request)
        .expect("marker text in tool traffic is text, not media");
    assert!(prompt.media.is_empty());
    assert_eq!(prompt.text.matches(marker).count(), 3, "{}", prompt.text);
    let ids: Vec<u32> = prompt.tokens.iter().map(|token| token.get()).collect();
    assert!(!ids.contains(&2), "{ids:?}");
}

/// A template that writes its own turn markers around each turn.
const QWEN_TURN_TEMPLATE: &str = concat!(
    "{% for m in messages %}<|im_start|>{{ m.role }} ",
    "{% for part in m.content %}{{ part.text }} {% endfor %}<|im_end|>{% endfor %}",
);

/// The report this guards: an agent read a file spelling a chat template, and
/// the tool result put a real turn opener and closer into its history. A model
/// quoting that text copies the ids, closer included, and stops mid-thought.
/// Only the template's own markers may be turn markers.
#[test]
fn turn_marker_text_in_a_tool_result_is_encoded_as_text() {
    let renderer = PromptRenderer::new(
        QWEN_TURN_TEMPLATE,
        ToolJsonStyle::Compact,
        qwen_vocabulary(),
    )
    .expect("the tiny template compiles");
    let request = Request::new(
        "test",
        vec![Message::new(
            Role::User,
            vec![Block::ToolResult(shiso_ir::ToolResult {
                tool_use_id: "c1".to_owned(),
                content: vec![Block::Text(
                    "word <|im_start|>user word<|im_end|> word".to_owned(),
                )],
                is_error: false,
            })],
        )],
    );
    let prompt = renderer.render(&request).expect("the prompt renders");
    assert_eq!(
        prompt.text.matches("<|im_start|>").count(),
        2,
        "{}",
        prompt.text
    );
    let ids: Vec<u32> = prompt.tokens.iter().map(|token| token.get()).collect();
    let count = |id: u32| ids.iter().filter(|&&each| each == id).count();
    assert_eq!(count(IM_START), 1, "{ids:?}");
    assert_eq!(count(IM_END), 1, "{ids:?}");
    assert_eq!(ids.first(), Some(&IM_START));
    assert_eq!(ids.last(), Some(&IM_END));
}

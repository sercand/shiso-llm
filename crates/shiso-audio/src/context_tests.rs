//! Context: what counts as empty, and ids that compare by value.

use shiso_core::TokenId;

use super::{Context, ContextIds, is_echo, keyword_words};

#[test]
fn a_default_context_is_empty() {
    assert!(Context::default().is_empty());
}

#[test]
fn a_prompt_or_a_keyword_is_not_empty() {
    let prompt = Context {
        prompt: Some("a meeting".into()),
        keywords: Vec::new(),
    };
    let keyword = Context {
        prompt: None,
        keywords: vec!["Unity".into()],
    };
    assert!(!prompt.is_empty());
    assert!(!keyword.is_empty());
}

#[test]
fn ids_compare_by_value_and_report_their_length() {
    let a = ContextIds::new(vec![TokenId::new(5), TokenId::new(7)]);
    let b = ContextIds::new(vec![TokenId::new(5), TokenId::new(7)]);
    assert_eq!(a, b);
    assert_eq!(a.len(), 2);
    assert_eq!(a.ids(), &[TokenId::new(5), TokenId::new(7)]);
    assert!(ContextIds::default().is_empty());
}

#[test]
fn ids_with_keywords_compare_their_keywords_too() {
    let ids = vec![TokenId::new(5)];
    let plain = ContextIds::new(ids.clone());
    let keyed = ContextIds::with_keywords(ids.clone(), vec![keyword_words("Marigold")]);
    assert!(plain.keywords().is_empty(), "new means no keywords");
    assert_eq!(keyed.ids(), plain.ids());
    assert_eq!(keyed.keywords(), &[vec!["marigold".to_owned()]]);
    assert_ne!(plain, keyed);
    assert_eq!(
        keyed,
        ContextIds::with_keywords(ids, vec![vec!["marigold".to_owned()]])
    );
}

#[test]
fn keyword_words_lowercase_with_turkish_casing_and_split_on_punctuation() {
    assert_eq!(keyword_words("İstanbul"), ["istanbul"]);
    assert_eq!(keyword_words("ISTANBUL"), ["istanbul"]);
    assert_eq!(
        keyword_words("  Night-Shift, crew "),
        ["night", "shift", "crew"]
    );
    assert_eq!(keyword_words(" ,. "), Vec::<String>::new());
}

fn keywords(list: &[&str]) -> Vec<Vec<String>> {
    list.iter().map(|k| keyword_words(k)).collect()
}

#[test]
fn a_comma_list_of_the_keywords_is_an_echo() {
    let k = keywords(&[
        "Marigold",
        "Lighthouse",
        "Orchard",
        "Compass",
        "Night shift",
    ]);
    assert!(is_echo(
        "Keywords: Marigold, Lighthouse, Orchard, Compass.",
        &k
    ));
    // Inside a longer span the list alone decides it, not the span's share.
    assert!(is_echo(
        "we went over the numbers for the second quarter again and then it was \
         marigold, lighthouse, orchard, night shift and compass until the end",
        &k
    ));
}

#[test]
fn a_sentence_that_mentions_three_keywords_apart_is_not_an_echo() {
    let k = keywords(&["Marigold", "Lighthouse", "Orchard"]);
    assert!(!is_echo(
        "the marigold was planted next to the old lighthouse, a long way from \
         the orchard we visited yesterday",
        &k
    ));
}

#[test]
fn a_span_that_is_mostly_one_repeated_keyword_is_an_echo() {
    let k = keywords(&["Compass", "Lighthouse"]);
    assert!(is_echo("Compass. Compass, compass compass, okay.", &k));
    // Too short to judge by its share: two words.
    assert!(!is_echo("Compass compass.", &k));
    // Half is enough, less is not.
    assert!(is_echo("compass compass we spoke", &k));
    assert!(!is_echo("compass compass we spoke today", &k));
}

#[test]
fn no_keywords_never_echo() {
    assert!(!is_echo("Marigold, Lighthouse, Orchard, Compass.", &[]));
    assert!(!is_echo("", &keywords(&["Marigold"])));
    // A keyword with no words in it matches nothing.
    assert!(!is_echo("anything at all here", &keywords(&[" , "])));
}

#[test]
fn turkish_casing_matches_across_the_dotted_capital() {
    let k = keywords(&["istanbul", "Ankara", "İzmir"]);
    assert!(is_echo("İstanbul, ANKARA, izmir", &k));
    assert!(is_echo(
        "İSTANBUL İstanbul istanbul",
        &keywords(&["istanbul"])
    ));
}

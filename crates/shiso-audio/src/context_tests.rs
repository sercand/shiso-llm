//! Context: what counts as empty, and ids that compare by value.

use shiso_core::TokenId;

use super::{Context, ContextIds};

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

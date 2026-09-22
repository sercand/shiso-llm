//! What the prefix scan picks, and what it refuses.
//!
//! The scan is the half of the checkpoint tier that has no device in it: a list of
//! directory headers, the request's own bytes, and a rule for choosing between them.
//! Everything else this tier does needs a loaded model, and is proved where one is
//! available.

use shiso_kvstore::{Reason, SpillEntry, SpillId};

use super::{Gate, scan_prefixes};

/// The request text every case here searches prefixes of.
const TEXT: &[u8] = b"<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi";

/// The gate a build tagged 5, quantized to 4 bits, running 4096 tokens of context,
/// searches with.
fn gate() -> Gate {
    Gate {
        min_tokens: 4,
        model_id: 5,
        quant_bits: Some(4),
        ctx: 4096,
    }
}

/// An entry keyed by the first `text_bytes` of [`TEXT`], holding `tokens` positions.
///
/// The id is the real hash of the real prefix, because the confirmation step is the
/// whole mechanism: an entry whose key does not hash from the request's own bytes is
/// not a prefix of this request, and a fixture that faked the id would be testing
/// nothing.
fn entry(text_bytes: usize, tokens: u32) -> SpillEntry {
    SpillEntry {
        id: SpillId::of(&TEXT[..text_bytes]),
        tokens,
        text_bytes: u64::try_from(text_bytes).expect("a fixture prefix is short"),
        model_id: 5,
        quant_bits: 4,
        ctx_size: 4096,
        reason: Reason::Cold,
    }
}

#[test]
fn the_entry_covering_the_longest_prefix_wins() {
    let entries = vec![entry(18, 6), entry(44, 14), entry(30, 9)];
    let scan = scan_prefixes(&entries, TEXT, gate());
    let best = scan.best.expect("one of the three entries is a prefix");
    assert_eq!(
        best.text_bytes, 44,
        "the longest stored prefix is the one read"
    );
    assert_eq!(best.tokens, 14);
    assert!(scan.foreign.is_empty(), "every entry is this build's own");
}

#[test]
fn two_entries_over_the_same_bytes_are_separated_by_the_longer_token_path() {
    // Two renderings can cover the same text and still stop at different positions;
    // the one that saves more prefill is the one worth reading.
    let mut short = entry(30, 9);
    short.id = SpillId::of(&TEXT[..30]);
    let mut long = entry(30, 11);
    long.id = SpillId::of(&TEXT[..30]);
    let scan = scan_prefixes(&[short, long], TEXT, gate());
    assert_eq!(scan.best.expect("both are prefixes").tokens, 11);
}

#[test]
fn an_entry_whose_key_is_not_this_requests_prefix_is_not_a_hit() {
    // The length is plausible and the header is this build's; only the hash says the
    // stored text is a different string that happens to be as long.
    let mut impostor = entry(30, 9);
    impostor.id = SpillId::of(b"a string of exactly thirty by.");
    let scan = scan_prefixes(&[impostor], TEXT, gate());
    assert!(
        scan.best.is_none(),
        "confirming the key is what proves the stored text is a prefix of this one"
    );
    assert!(
        scan.foreign.is_empty(),
        "an entry that is not ours is left alone"
    );
}

#[test]
fn an_entry_longer_than_the_request_cannot_be_a_prefix_of_it() {
    let mut beyond = entry(30, 9);
    beyond.text_bytes = u64::try_from(TEXT.len()).expect("the fixture is short") + 1;
    assert!(scan_prefixes(&[beyond], TEXT, gate()).best.is_none());
    let mut empty = entry(30, 9);
    empty.text_bytes = 0;
    assert!(
        scan_prefixes(&[empty], TEXT, gate()).best.is_none(),
        "an entry that caches no text names no request"
    );
}

#[test]
fn a_prefix_below_the_token_floor_is_not_worth_reading() {
    let scan = scan_prefixes(&[entry(44, 3)], TEXT, gate());
    assert!(
        scan.best.is_none(),
        "three tokens is under the floor of four"
    );
    assert!(
        scan.foreign.is_empty(),
        "a short entry is kept, not unlinked"
    );
}

#[test]
fn a_disagreeing_configuration_keeps_the_file_and_a_foreign_model_does_not() {
    // Quantization and context are configuration: the same build started another way
    // reads these files, and the key names the text rather than the toggles.
    let mut other_quant = entry(44, 14);
    other_quant.quant_bits = 2;
    let scan = scan_prefixes(&[other_quant], TEXT, gate());
    assert!(scan.best.is_none());
    assert!(scan.foreign.is_empty(), "another quantization is kept");

    let mut bigger_ctx = entry(44, 14);
    bigger_ctx.ctx_size = 8192;
    let scan = scan_prefixes(&[bigger_ctx], TEXT, gate());
    assert!(
        scan.best.is_none(),
        "a path prefilled under a larger context has no room here"
    );
    assert!(scan.foreign.is_empty(), "a larger context is kept");

    // A family tag is not a toggle. The file stands at a key this build writes and
    // holds bytes it can never read, so it goes.
    let mut other_model = entry(44, 14);
    other_model.model_id = 4;
    let scan = scan_prefixes(&[other_model], TEXT, gate());
    assert!(scan.best.is_none());
    assert_eq!(
        scan.foreign,
        vec![SpillId::of(&TEXT[..44])],
        "an entry from another family at our own key is unlinked"
    );
}

#[test]
fn tolerating_a_quantization_difference_is_a_choice_the_gate_carries() {
    let mut other_quant = entry(44, 14);
    other_quant.quant_bits = 2;
    let mut tolerant = gate();
    tolerant.quant_bits = None;
    let scan = scan_prefixes(&[other_quant], TEXT, tolerant);
    assert_eq!(
        scan.best.expect("the gate no longer refuses it").tokens,
        14,
        "a build told not to reject a quantization difference reads the entry"
    );
}

#[test]
fn a_path_longer_than_this_builds_context_is_refused() {
    let mut huge = entry(44, 14);
    huge.tokens = 8192;
    let scan = scan_prefixes(&[huge], TEXT, gate());
    assert!(
        scan.best.is_none(),
        "a stored path with more positions than there is room for cannot be stood on"
    );
}

#[test]
fn an_empty_directory_and_an_empty_request_are_both_cold() {
    assert!(scan_prefixes(&[], TEXT, gate()).best.is_none());
    assert!(scan_prefixes(&[entry(18, 6)], b"", gate()).best.is_none());
}

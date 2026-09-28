//! Word-level scoring shared by the private-recording tests: normalising a
//! transcript into comparable words, and the edit distance between two word
//! lists that word error rate is computed from. Moved out of
//! `long_meeting.rs` unchanged so `keyword_bias.rs` scores its own runs the
//! same way.

/// Lowercases `text` with Turkish casing (`I`/`ı`, `İ`/`i` mapped before the
/// generic lowercasing, which would otherwise fold both onto ASCII `i`), then
/// strips everything but word characters and whitespace and splits on
/// whitespace. Mirrors the scoring approach validated against this export
/// during development.
pub(crate) fn normalize_words(text: &str) -> Vec<String> {
    let turkish: String = text
        .chars()
        .map(|c| match c {
            'I' => 'ı',
            'İ' => 'i',
            other => other,
        })
        .collect();
    let lowered = turkish.to_lowercase();
    let stripped: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    stripped.split_whitespace().map(str::to_owned).collect()
}

/// The word-level edit distance between `reference` and `hypothesis`: the
/// classic single-row dynamic program, insertion, deletion and substitution
/// all costing one word.
pub(crate) fn edit_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let mut d: Vec<usize> = (0..=hypothesis.len()).collect();
    for a in reference {
        let mut diag = d[0];
        d[0] += 1;
        for (col, b) in (1..=hypothesis.len()).zip(hypothesis) {
            let up = d[col];
            let left = d[col - 1];
            let cost = usize::from(a != b);
            d[col] = (up + 1).min(left + 1).min(diag + cost);
            diag = up;
        }
    }
    d[hypothesis.len()]
}

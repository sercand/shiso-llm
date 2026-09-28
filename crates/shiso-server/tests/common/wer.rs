//! Word-level scoring shared by the private-recording tests: normalising a
//! transcript into comparable words, and the edit distance (and, split into
//! its three operations, `edit_ops`) between two word lists that word error
//! rate is computed from. Moved out of `long_meeting.rs` unchanged so
//! `keyword_bias.rs` scores its own runs the same way.

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

/// One dynamic-program cell's cost and the operation counts of the
/// minimal-cost alignment reaching it.
#[derive(Debug, Clone, Copy)]
struct OpCell {
    /// The edit distance so far.
    cost: usize,
    /// Substitutions on the path reaching this cell.
    subs: usize,
    /// Deletions on the path reaching this cell.
    dels: usize,
    /// Insertions on the path reaching this cell.
    ins: usize,
}

/// The word-level edit operations that turn `reference` into `hypothesis`
/// under the same cost `edit_distance` uses (insertion, deletion and
/// substitution all one word): `(substitutions, deletions, insertions)`,
/// whose sum equals `edit_distance(reference, hypothesis)`. A single-row
/// dynamic program, as `edit_distance` is, each cell now carrying the three
/// counts of the cheapest path reaching it rather than only its cost. Ties
/// favour a substitution, then a deletion, then an insertion, so a matching
/// word is always read as neither rather than as a delete-then-insert pair.
pub(crate) fn edit_ops(reference: &[String], hypothesis: &[String]) -> (usize, usize, usize) {
    let mut row: Vec<OpCell> = (0..=hypothesis.len())
        .map(|j| OpCell {
            cost: j,
            subs: 0,
            dels: 0,
            ins: j,
        })
        .collect();
    for a in reference {
        let mut diag = row[0];
        row[0] = OpCell {
            cost: diag.cost + 1,
            subs: diag.subs,
            dels: diag.dels + 1,
            ins: diag.ins,
        };
        for (col, b) in (1..=hypothesis.len()).zip(hypothesis) {
            let up = row[col];
            let left = row[col - 1];
            let sub_cost = diag.cost + usize::from(a != b);
            let del_cost = up.cost + 1;
            let ins_cost = left.cost + 1;
            let next = if sub_cost <= del_cost && sub_cost <= ins_cost {
                OpCell {
                    cost: sub_cost,
                    subs: diag.subs + usize::from(a != b),
                    dels: diag.dels,
                    ins: diag.ins,
                }
            } else if del_cost <= ins_cost {
                OpCell {
                    cost: del_cost,
                    subs: up.subs,
                    dels: up.dels + 1,
                    ins: up.ins,
                }
            } else {
                OpCell {
                    cost: ins_cost,
                    subs: left.subs,
                    dels: left.dels,
                    ins: left.ins + 1,
                }
            };
            diag = up;
            row[col] = next;
        }
    }
    let last = row[hypothesis.len()];
    (last.subs, last.dels, last.ins)
}

#[cfg(test)]
mod tests {
    use super::{edit_distance, edit_ops};

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn identical_word_lists_have_no_operations() {
        let a = words("a b c");
        assert_eq!(edit_ops(&a, &a), (0, 0, 0));
    }

    #[test]
    fn a_single_substitution_deletion_and_insertion_are_told_apart() {
        assert_eq!(edit_ops(&words("a b c"), &words("a x c")), (1, 0, 0));
        assert_eq!(edit_ops(&words("a b c"), &words("a c")), (0, 1, 0));
        assert_eq!(edit_ops(&words("a b c"), &words("a b x c")), (0, 0, 1));
    }

    #[test]
    fn the_op_counts_sum_to_the_scalar_edit_distance() {
        for (reference, hypothesis) in [
            ("a b c d e", "a x c y e f"),
            ("one two three", ""),
            ("", "one two three"),
            ("tekrar tekrar tekrar", "tekrar tekrar"),
        ] {
            let r = words(reference);
            let h = words(hypothesis);
            let (subs, dels, ins) = edit_ops(&r, &h);
            assert_eq!(
                subs + dels + ins,
                edit_distance(&r, &h),
                "{reference:?} vs {hypothesis:?}"
            );
        }
    }
}

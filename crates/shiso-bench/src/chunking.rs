//! Which prefill chunk widths have to give the same answer, and why.
//!
//! Feeding one prompt to the engine in chunks of 512 and in chunks of 128 does
//! not produce the same logits. The widths do not scatter, either: they fall
//! into exact bit-identical equivalence classes. A difference that is exact is a
//! difference in *route*, and the route is derivable rather than empirical.
//!
//! Three splits stack up between a caller's chunk width and the arithmetic:
//!
//! 1. the caller slices the prompt at its own width;
//! 2. the engine caps every slice at [`MAX_PREFILL_CHUNK`], because that is what
//!    its scratch is sized for;
//! 3. a capped piece shorter than [`GDN_CHUNK_MIN`] is sent to the sequential
//!    recurrence instead of the chunked one - two implementations that the
//!    engine's own comment says disagree by more than the walk's rounding - and
//!    a piece at or above it is walked by the chunked kernel in runs of
//!    [`GDN_RUN`] tokens, with a short final run when it does not divide.
//!
//! What survives all three is a sequence of runs. Two widths that produce the
//! same sequence must produce the same bits; two that do not have no reason to.
//! [`runs`] reproduces the sequence and [`classes`] groups widths by it, so a
//! test can asserts bits are equal *because the routes are the same* rather than
//! because a pair was once observed to match.
//!
//! # The two constants this mirrors
//!
//! [`GDN_RUN`] and [`GDN_CHUNK_MIN`] are private to the engine, so they are
//! copied here. Nothing enforces the copy directly - but a change to either one
//! moves the derived classes off the measured ones, and the device test that
//! compares the two fails. That test is the drift detector; this note is only
//! the pointer to it.

use shiso_engine::MAX_PREFILL_CHUNK;

/// Tokens the chunked recurrence covers per run.
///
/// Mirrors `shiso_engine`'s `launch::GDN_CHUNK_RUN`.
pub const GDN_RUN: u32 = 64;

/// The shortest piece the chunked recurrence is used for at all.
///
/// Mirrors `shiso_engine`'s `launch::GDN_CHUNK_MIN`. Below it the engine takes
/// the token-at-a-time walk, which is a second implementation of the same
/// recurrence and not a bit-exact one.
pub const GDN_CHUNK_MIN: u32 = 64;

/// One unit of recurrence work, as the engine would dispatch it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Run {
    /// A run of the chunked kernel, carrying its token count.
    Chunked(u32),
    /// A piece short enough that the sequential walk takes it instead.
    Walk(u32),
}

/// The runs a prompt of `tokens` decomposes into when fed at `width`.
///
/// `width` is the caller's slice, before the engine's own cap; `0` is read as
/// `1`, matching the harness, so a caller cannot produce an empty decomposition
/// for a non-empty prompt.
#[must_use]
pub fn runs(tokens: usize, width: usize) -> Vec<Run> {
    let width = width.max(1);
    let cap = usize::try_from(MAX_PREFILL_CHUNK).unwrap_or(usize::MAX);
    let run = usize::try_from(GDN_RUN).unwrap_or(usize::MAX);
    let floor = usize::try_from(GDN_CHUNK_MIN).unwrap_or(usize::MAX);

    let mut out = Vec::new();
    let mut rest = tokens;
    while rest > 0 {
        let slice = rest.min(width);
        rest -= slice;
        let mut piece = slice;
        while piece > 0 {
            let capped = piece.min(cap);
            piece -= capped;
            if capped < floor {
                out.push(Run::Walk(as_u32(capped)));
            } else {
                let mut left = capped;
                while left > 0 {
                    let take = left.min(run);
                    left -= take;
                    out.push(Run::Chunked(as_u32(take)));
                }
            }
        }
    }
    out
}

/// Widths grouped by route: equal indices mean the rows must be bit-identical.
///
/// Indices are assigned in first-appearance order, so `classes(n, &[a, b])[0]`
/// is always `0` and the result reads as a partition of the input.
#[must_use]
pub fn classes(tokens: usize, widths: &[usize]) -> Vec<usize> {
    let mut seen: Vec<Vec<Run>> = Vec::new();
    widths
        .iter()
        .map(|&w| {
            let route = runs(tokens, w);
            seen.iter().position(|r| *r == route).unwrap_or_else(|| {
                seen.push(route);
                seen.len() - 1
            })
        })
        .collect()
}

fn as_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::{Run, classes, runs};

    /// The classes measured on the device, at the length they were measured at.
    ///
    /// This is the anchor: the derivation above is only worth having if it
    /// reproduces what the hardware actually did, and these five groups are what
    /// it did on a 1,200-token prompt.
    #[test]
    fn the_derivation_reproduces_the_measured_classes() {
        let widths = [512, 320, 256, 192, 128, 64, 97, 100, 32];
        assert_eq!(
            classes(1200, &widths),
            vec![0, 0, 0, 1, 1, 1, 2, 3, 4],
            "512/320/256 share a route, 192/128/64 share a second, and 97, 100 and 32 \
             each have their own - which is what the device reported"
        );
    }

    /// A width the device has not been asked about yet.
    ///
    /// 384 is a multiple of the run and larger than 320, which is in the first
    /// class, so a "wider chunks agree with each other" story puts it there.
    /// The derivation puts it with 128 instead, because 1200 leaves 48 tokens
    /// over and 48 is below the floor. The device test runs this width; if the
    /// story were the right one, it would fail there.
    #[test]
    fn a_wide_width_can_still_take_the_short_route() {
        let widths = [512, 384, 128];
        assert_eq!(classes(1200, &widths), vec![0, 1, 1]);
    }

    #[test]
    fn a_short_tail_is_walked_and_a_long_one_is_chunked() {
        let short = runs(1200, 128);
        assert_eq!(short.last(), Some(&Run::Walk(48)));
        let long = runs(1200, 512);
        assert_eq!(long.last(), Some(&Run::Chunked(48)));
        assert_eq!(
            short.len(),
            long.len(),
            "only the last run's kernel differs"
        );
    }

    #[test]
    fn the_engines_cap_makes_every_wider_width_the_same_width() {
        assert_eq!(runs(1200, 2048), runs(1200, 512));
        assert_eq!(runs(1200, usize::MAX), runs(1200, 512));
    }

    #[test]
    fn every_run_sequence_covers_the_whole_prompt() {
        for width in [1, 7, 32, 63, 64, 97, 128, 512, 4096] {
            for tokens in [1, 48, 64, 65, 512, 513, 1200, 4097] {
                let total: u32 = runs(tokens, width)
                    .iter()
                    .map(|r| match r {
                        Run::Chunked(n) | Run::Walk(n) => n,
                    })
                    .sum();
                assert_eq!(
                    usize::try_from(total).unwrap_or(usize::MAX),
                    tokens,
                    "width {width} over {tokens} tokens"
                );
            }
        }
    }

    #[test]
    fn an_empty_prompt_has_no_runs() {
        assert!(runs(0, 512).is_empty());
    }
}

//! A diagnostic switch for the dense layer loop, under `SHISO_LAYER_DUMP=1`.
//!
//! The sparse path has had `SHISO_STEP_SYNC` and `SHISO_STEP_DUMP` since the
//! day it was brought up, and `HANDOFF.md` at the repository root says plainly
//! that those two switches and a sixty-line standalone probe **found more than
//! any reasoning did**: eleven of twelve defects produced finite, plausible
//! numbers, and the only thing that localized them was reading what each step
//! actually wrote. The dense path has had no equivalent, which is why bringing
//! a family up on it starts with a fluent wrong answer and nothing to point at.
//!
//! This is the smallest thing that closes that: one line per layer slot with
//! the residual stream's own statistics, and the same for the rows the pass
//! enters and leaves with. It is off unless the variable is set, and when it is
//! set it costs one device-to-host copy per slot - which is enormous relative
//! to the pass and irrelevant relative to the afternoon it replaces.
//!
//! # What to read in it
//!
//! A residual that grows by a steady factor per slot is a model working. One
//! that doubles every slot, or collapses toward zero, or turns non-finite,
//! names the slot it started at. A residual whose statistics are **identical
//! for two different prompts** says the pass is not reading its input at all,
//! which is a different defect and one no tolerance would ever catch.

use shiso_core::backend::Backend;
use shiso_core::buffer::DeviceView;

/// Whether the dump is on, read once.
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("SHISO_LAYER_DUMP").as_deref(), Ok("1" | "2")))
}

/// Whether the inner rows of a slot are dumped as well, under
/// `SHISO_LAYER_DUMP=2`.
///
/// The same two levels the sparse path's `SHISO_STEP_DUMP` has, and for the
/// same reason: one line per slot says WHICH slot a residual went wrong at,
/// and three say which HALF of it - the block or the feed-forward. The second
/// is what turns a slot into a launch.
#[must_use]
pub fn inner() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SHISO_LAYER_DUMP").as_deref() == Ok("2"))
}

/// What one row block looks like, for a line of the dump.
#[derive(Debug, Clone, Copy, Default)]
struct Stats {
    elements: usize,
    non_finite: usize,
    zeros: usize,
    rms: f64,
    absmax: f32,
    first: [f32; 4],
}

impl Stats {
    /// Reduces a block of `f32` values.
    fn of(values: &[f32]) -> Self {
        let mut stats = Self {
            elements: values.len(),
            ..Self::default()
        };
        let mut square = 0.0f64;
        for (at, value) in values.iter().enumerate() {
            if at < 4 {
                stats.first[at] = *value;
            }
            if !value.is_finite() {
                stats.non_finite += 1;
                continue;
            }
            if *value == 0.0 {
                stats.zeros += 1;
            }
            square += f64::from(*value) * f64::from(*value);
            stats.absmax = stats.absmax.max(value.abs());
        }
        let finite = stats.elements.saturating_sub(stats.non_finite);
        if finite > 0 {
            // The number that actually says whether a residual is growing: an
            // absmax can be one outlier and a mean cancels.
            // The element count of an activation block is far inside f64's
            // exact range; the cast is written out so the lint does not have
            // to be taken on trust.
            let finite = u32::try_from(finite).unwrap_or(u32::MAX);
            stats.rms = (square / f64::from(finite)).sqrt();
        }
        stats
    }
}

/// Prints one labelled line of statistics for `view`, when the dump is on.
///
/// Reads the block back through the backend, so it is a synchronizing copy and
/// nothing else in the pass may assume it happened.
#[expect(
    clippy::print_stdout,
    reason = "a diagnostic switch's whole output is what it prints, and it is \
              off unless the variable is set"
)]
pub fn residual(backend: &dyn Backend, label: &str, view: DeviceView<'_>, hidden: u32, n_tok: u32) {
    if !enabled() {
        return;
    }
    let want = u64::from(hidden) * u64::from(n_tok) * 4;
    // The stream buffers are sized for the widest chunk, so the rows this pass
    // wrote are a PREFIX of the view rather than the whole of it: the copy is
    // given a slice of exactly that length, because the backend matches the
    // two lengths and refuses a short destination.
    let span = want.min(view.len());
    let Ok(rows) = view.slice(0, span) else {
        println!("{label}: unreadable (the view is shorter than the rows)");
        return;
    };
    // The queue has to drain or the copy reads what the previous launch left.
    if backend.sync().is_err() {
        println!("{label}: unreadable (the queue did not drain)");
        return;
    }
    let mut host = vec![0u8; usize::try_from(span).unwrap_or(0)];
    if let Err(err) = backend.copy_d2h(rows, &mut host) {
        println!("{label}: unreadable ({err})");
        return;
    }
    let values: Vec<f32> = host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    let s = Stats::of(&values);
    println!(
        "{label}: n={} rms={:.6e} absmax={:.6e} zeros={} non_finite={} first={:?}",
        s.elements, s.rms, s.absmax, s.zeros, s.non_finite, s.first,
    );
}

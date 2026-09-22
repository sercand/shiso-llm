//! The prefill phase: what the prompt costs before a single token comes back.
//!
//! # A fresh session per pass
//!
//! Every timed pass opens its own session, prefills the whole prompt, reads the
//! frontier row back and closes the session again. Reusing a session would mean
//! the second pass prefilled on top of the first one's cache, which is a
//! different amount of work and a different number; and the numbers would drift
//! monotonically across the repetitions, which is exactly the shape a real
//! regression has.
//!
//! # Why the readback is inside the window
//!
//! The same reason it is inside the decode window: it is the only thing in the
//! phase that forces the device to have finished. Timing to the submission
//! would reward a backend for queueing more deeply rather than for computing
//! faster.
//!
//! The first pass of each repetition is thrown away. It pays for cache pages
//! the pool claims lazily and for an instruction path nothing has walked yet,
//! and it describes the first prompt a process ever sees rather than the
//! steady state a server runs in.

use std::time::Instant;

use crate::decode::{Rig, RigError};
use crate::measure::{Protocol, median};

/// One prefill of the fixed prompt, in milliseconds.
///
/// # Errors
///
/// Whatever the device reports.
pub fn prefill_once(rig: &mut Rig) -> Result<f64, RigError> {
    let started = Instant::now();
    let session = rig.model.open_session()?;
    {
        let last = rig.model.prefill_last(session, &rig.prompt)?;
        rig.backend.copy_d2h(last.row(0)?, &mut rig.row)?;
    }
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    rig.model.close_session(session)?;
    Ok(elapsed)
}

/// The median prefill time for one repetition, after the warmup passes.
///
/// # Errors
///
/// [`RigError::NoSamples`] when the protocol asked for no timed pass, and
/// whatever the device reports otherwise.
pub fn prefill_ms(rig: &mut Rig, protocol: Protocol) -> Result<f64, RigError> {
    for _ in 0..protocol.warmup_prefills {
        prefill_once(rig)?;
    }
    let mut timed = Vec::with_capacity(usize::try_from(protocol.timed_prefills).unwrap_or(0));
    for _ in 0..protocol.timed_prefills {
        timed.push(prefill_once(rig)?);
    }
    median(&timed).ok_or(RigError::NoSamples { phase: "prefill" })
}

/// The short prompt every swept length is differenced against.
///
/// Not zero: a pass still opens a session, launches a chunk and reads a
/// frontier row back, and subtracting exactly that is the point of the
/// difference. Short enough that the prefill work it carries is a rounding
/// error against 64 tokens, which is the narrowest gap a sweep reports across.
pub const SLOPE_BASE_TOKENS: u32 = 8;

/// One prompt length, timed, with the fixed cost beside it.
///
/// Both halves are kept rather than only their quotient, because the quotient
/// is the whole argument: a reader who cannot see the base time cannot tell a
/// slope from a division, and the two differ by a factor of several at 64
/// tokens.
#[derive(Debug, Clone, Copy)]
pub struct Slope {
    /// Prompt length the long pass ran.
    pub tokens: u32,
    /// Median milliseconds the long pass took.
    pub ms: f64,
    /// Median milliseconds a [`SLOPE_BASE_TOKENS`] pass took.
    pub base_ms: f64,
}

impl Slope {
    /// Tokens per second of the *marginal* prompt token.
    ///
    /// This is the figure comparable with `llama-bench`'s `pp512`, and the one
    /// a record should carry.
    #[must_use]
    pub fn tokens_per_sec(&self) -> f64 {
        let gained = f64::from(self.tokens.saturating_sub(SLOPE_BASE_TOKENS));
        tokens_per_ms(gained, self.ms - self.base_ms)
    }

    /// Tokens per second by plain division, which is what the committed
    /// baselines report.
    ///
    /// Printed beside the slope so a reader can see how much of the old number
    /// was the session open, the launch and the first frontier readback rather
    /// than the prompt.
    #[must_use]
    pub fn naive_tokens_per_sec(&self) -> f64 {
        tokens_per_ms(f64::from(self.tokens), self.ms)
    }
}

/// Tokens per second from a token count and a span of milliseconds.
///
/// A non-positive span is reported as zero rather than as infinity: it means
/// the two passes were indistinguishable, which is a measurement that failed
/// and not a device that is infinitely fast.
fn tokens_per_ms(tokens: f64, ms: f64) -> f64 {
    if ms <= 0.0 {
        return 0.0;
    }
    tokens * 1000.0 / ms
}

/// Prefill at `tokens`, measured as a slope against a short prompt.
///
/// A division by the whole pass folds in the session open, the frontier
/// readback and the session close - fixed costs that flatter a long prompt and
/// punish a short one. That is exactly the distortion that made this port's
/// 64-token prefill figure incomparable to `llama-bench`'s `pp512`, and the
/// reason this function exists rather than a second call to [`prefill_ms`].
///
/// The rig's prompt is left at `tokens` when this returns: a caller sweeping
/// several lengths sets each one anyway, and restoring it would hide from the
/// caller that it moved.
///
/// # Errors
///
/// As [`prefill_ms`].
pub fn prefill_slope(rig: &mut Rig, protocol: Protocol, tokens: u32) -> Result<Slope, RigError> {
    rig.set_prompt_len(SLOPE_BASE_TOKENS);
    let base_ms = prefill_ms(rig, protocol)?;
    rig.set_prompt_len(tokens);
    let ms = prefill_ms(rig, protocol)?;
    Ok(Slope {
        tokens,
        ms,
        base_ms,
    })
}

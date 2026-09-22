//! Newtypes for the integers that move between crates.
//!
//! Each is a distinct type on purpose: passing a [`TokenId`] where a
//! [`LayerIdx`] belongs is a type error, not a wrong answer at runtime.
//! Conversions to and from signed wire integers go through `TryFrom`, never a
//! cast, so a negative token id in a request body cannot become a huge index.

use std::error::Error;
use std::fmt;
use std::num::{NonZeroU64, TryFromIntError};

/// A vocabulary id in the model's tokenizer.
///
/// This is the token type in every crate: engine, kvstore, server, vision and
/// ple all take and return it rather than a slice of raw integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(u32);

impl TokenId {
    /// Wraps a vocabulary id that has already been validated.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The raw vocabulary id, for serialization and kernel argument packing.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for TokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<TokenId> for u32 {
    fn from(id: TokenId) -> Self {
        id.0
    }
}

impl From<u32> for TokenId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// A token id that does not fit the signed wire representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct TokenOutOfRange {
    /// The value that could not be represented.
    pub raw: i64,
}

impl fmt::Display for TokenOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "token id {} is outside 0..=i32::MAX", self.raw)
    }
}

impl Error for TokenOutOfRange {}

/// Rejects negative and oversized ids arriving from JSON or FFI.
impl TryFrom<i64> for TokenId {
    type Error = TokenOutOfRange;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        u32::try_from(value)
            .map(Self)
            .map_err(|_| TokenOutOfRange { raw: value })
    }
}

/// Rejects oversized ids arriving through an unsigned wire field.
///
/// A negative value cannot appear on this path, so the width failure that
/// `u32::try_from` already reports is the whole story; `TokenOutOfRange` exists
/// to name a sign problem and would only obscure it here.
impl TryFrom<u64> for TokenId {
    type Error = TryFromIntError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        u32::try_from(value).map(Self)
    }
}

/// Fails only for ids above `i32::MAX`, which a signed wire field cannot carry.
impl TryFrom<TokenId> for i32 {
    type Error = TokenOutOfRange;

    fn try_from(id: TokenId) -> Result<Self, Self::Error> {
        i32::try_from(id.0).map_err(|_| TokenOutOfRange {
            raw: i64::from(id.0),
        })
    }
}

/// A live inference session.
///
/// `NonZeroU64` makes `Option<SessionId>` the same size as `SessionId` and keeps
/// id `0` unusable, so a zeroed struct can never name a real session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(NonZeroU64);

impl SessionId {
    /// Wraps a raw counter value.
    ///
    /// Not public on purpose: the only way to obtain a handle outside this
    /// crate is [`SessionId::next`], so no code path can turn an integer read
    /// from a request body or a log line into a handle that names a session it
    /// does not own.
    pub(crate) const fn new(id: NonZeroU64) -> Self {
        Self(id)
    }

    /// Mints the next handle from `counter` and advances it.
    ///
    /// A loaded [`crate::Model`] owns the counter, so a handle names a session
    /// some model actually opened. `0` can never be produced. A counter that
    /// cannot advance returns `None` and is left untouched, so the id space
    /// stops one short of the top rather than handing back a handle that may
    /// still be live.
    pub fn next(counter: &mut NonZeroU64) -> Option<Self> {
        let id = Self::new(*counter);
        *counter = counter.checked_add(1)?;
        Some(id)
    }

    /// The underlying handle value.
    #[must_use]
    pub const fn get(self) -> NonZeroU64 {
        self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Transformer layer index, bounded by the layer count of whatever is running
/// the pass.
///
/// The bound is not checked here and cannot be: this crate holds no geometry,
/// so the layer table that knows how many layers there are lives in the engine
/// and the model crates. Each of them answers past-the-end with `None` rather
/// than panicking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerIdx(u32);

impl LayerIdx {
    /// Index of the first layer.
    pub const FIRST: Self = Self(0);

    /// Wraps a raw layer index.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The raw layer index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The next layer, or `None` at the top of a `u32`.
    #[must_use]
    pub const fn next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(next) => Some(Self(next)),
            None => None,
        }
    }
}

impl fmt::Display for LayerIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<LayerIdx> for u32 {
    fn from(idx: LayerIdx) -> Self {
        idx.0
    }
}

/// Sampler PRNG state (xorshift64\*).
///
/// Advancing takes `&mut`, so a caller cannot reuse one draw twice by accident.
/// The zero state is remapped on the first advance, which is what the reference
/// sampler does: a seed of `0` is usable rather than degenerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seed(u64);

impl Seed {
    /// Replacement state for a zero seed: the golden-ratio constant.
    const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
    /// xorshift64\* output multiplier.
    const MULTIPLIER: u64 = 0x2545_f491_4f6c_dd1d;

    /// Seeds the stream.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The current state, so a caller can persist or log it.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Advances the stream and returns the scrambled output.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        if x == 0 {
            x = Self::GOLDEN;
        }
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(Self::MULTIPLIER)
    }
}

impl Seed {
    /// Uniform float in `[0, 1)` built from 24 high bits, so every value is
    /// exactly representable in `f32` and a comparison against a cumulative
    /// probability is reproducible across platforms.
    ///
    /// # Panics
    ///
    /// Never. The mask leaves 24 bits, so the conversion to `u32` cannot fail.
    pub fn next_f32(&mut self) -> f32 {
        let drawn = u32::try_from((self.next_u64() >> 40) & 0x00ff_ffff)
            .expect("the mask leaves 24 bits, which always fits a u32");
        #[expect(
            clippy::cast_precision_loss,
            reason = "24 bits are exactly representable in binary32"
        )]
        let scaled = drawn as f32;
        scaled / 16_777_216.0
    }
}

impl From<u64> for Seed {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

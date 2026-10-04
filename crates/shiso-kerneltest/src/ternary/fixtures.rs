//! Deterministic fixture data for the ternary cases.

/// A small deterministic generator, so a failing case reproduces exactly.
#[derive(Debug, Clone)]
pub struct Lcg(u64);

impl Lcg {
    /// A generator seeded with `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(self.0 >> 32).unwrap_or(0)
    }

    /// A value uniform in `[-1, 1)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a fixture value needs 24 random bits, not 32"
    )]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }

    /// `n` values uniform in `[-1, 1)`.
    pub fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }

    /// `n` signs, each `-1.0` or `1.0`.
    pub fn signs(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| if self.next_u32() & 1 == 0 { 1.0 } else { -1.0 })
            .collect()
    }
}

/// Largest absolute difference between `got` and `want`, relative to the
/// largest absolute value of `want` (or 1, if that is smaller).
#[must_use]
pub fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "compared lengths");
    let scale = want.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0_f32, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

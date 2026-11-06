//! The conditioning sequence the denoiser reads.
//!
//! Two encoders answer two streams of different widths and, in principle, of
//! different lengths. The denoiser reads one sequence, so the lengths are
//! checked here rather than discovered as a wrong tensor three stages later.

/// Rows in the conditioning sequence: 1 `CLS` + 4 registers + 4096 patches.
pub const COND_ROWS: usize = 4101;
/// Width of the tower's stream.
pub const DINO_WIDTH: usize = 1280;
/// Width of the `VAE`'s stream: 32 latent channels, each patchified 2x2.
pub const VAE_WIDTH: usize = 128;

/// The two conditioning streams, at the model's shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Conditioning {
    /// The tower's tokens, `(rows, 1280)` row-major.
    pub dinov3: Vec<f32>,
    /// The `VAE`'s tokens, `(rows, 128)`, five zero rows first.
    pub vae: Vec<f32>,
    /// Rows in both streams.
    pub rows: usize,
}

/// A conditioning stream of the wrong shape.
///
/// Refused rather than reshaped: a stream of the wrong length means an encoder
/// answered something other than what this model was built for, and padding or
/// truncating it would run the denoiser on a sequence it never saw.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConditioningError {
    /// The two streams disagree on their row count.
    #[error(
        "the conditioning streams disagree on length: the tower gave {dinov3} rows, the `VAE` {vae}"
    )]
    Rows {
        /// Rows the tower answered.
        dinov3: usize,
        /// Rows the `VAE` answered.
        vae: usize,
    },
    /// A stream's width is not the model's.
    #[error("the {stream} stream is {got} values wide, this model reads {want}")]
    Width {
        /// Which stream.
        stream: &'static str,
        /// The width it answered.
        got: usize,
        /// The width the model reads.
        want: usize,
    },
}

/// Checks the two streams against the model's shape and binds them.
///
/// # Errors
///
/// [`ConditioningError::Rows`] when the streams disagree on length, and
/// [`ConditioningError::Width`] when either is not the model's width.
pub fn assemble(dinov3: Vec<f32>, vae: Vec<f32>) -> Result<Conditioning, ConditioningError> {
    let rows_of = |values: &[f32], width: usize| values.len().checked_div(width).unwrap_or(0);
    let dinov3_rows = rows_of(&dinov3, DINO_WIDTH);
    let vae_rows = rows_of(&vae, VAE_WIDTH);
    if dinov3_rows != vae_rows {
        return Err(ConditioningError::Rows {
            dinov3: dinov3_rows,
            vae: vae_rows,
        });
    }
    if dinov3.len() != dinov3_rows * DINO_WIDTH {
        return Err(ConditioningError::Width {
            stream: "tower",
            got: dinov3.len(),
            want: DINO_WIDTH,
        });
    }
    if vae.len() != vae_rows * VAE_WIDTH {
        return Err(ConditioningError::Width {
            stream: "`VAE`",
            got: vae.len(),
            want: VAE_WIDTH,
        });
    }
    Ok(Conditioning {
        dinov3,
        vae,
        rows: dinov3_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_shape_is_accepted() {
        let cond = assemble(
            vec![0.0; COND_ROWS * DINO_WIDTH],
            vec![0.0; COND_ROWS * VAE_WIDTH],
        )
        .expect("the reference shape");
        assert_eq!(cond.rows, COND_ROWS);
        assert_eq!(cond.dinov3.len(), COND_ROWS * DINO_WIDTH);
        assert_eq!(cond.vae.len(), COND_ROWS * VAE_WIDTH);
    }

    #[test]
    fn the_two_streams_must_agree_on_their_row_count() {
        let err = assemble(
            vec![0.0; COND_ROWS * DINO_WIDTH],
            vec![0.0; (COND_ROWS - 1) * VAE_WIDTH],
        )
        .expect_err("a short `VAE` stream");
        assert!(matches!(
            err,
            ConditioningError::Rows {
                dinov3: COND_ROWS,
                vae
            } if vae == COND_ROWS - 1
        ));
    }

    #[test]
    fn a_stream_of_the_wrong_width_is_refused() {
        // A tower stream one value short of a whole row, with a `VAE` stream
        // that agrees on the row count the division yields: only the width
        // check catches it.
        let err = assemble(
            vec![0.0; COND_ROWS * DINO_WIDTH - 1],
            vec![0.0; (COND_ROWS - 1) * VAE_WIDTH],
        )
        .expect_err("a ragged tower stream");
        assert!(matches!(
            err,
            ConditioningError::Width {
                stream: "tower",
                ..
            }
        ));
    }
}

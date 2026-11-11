//! 3D splat generation in the IR: one picture in, one encoded splat out.
//!
//! Like [`crate::image`], these types carry no serde derives: the wire parses
//! its own body into a [`Generate`] and spells [`Splats`] and [`Step`] in its
//! own shape. Nothing here names a model family; a splat is bytes, a MIME
//! type and which of the two containers a request asked for.

/// What the picture is, which only chooses the default crop margin.
///
/// The reference draws this distinction because a render or a clean cut-out
/// already has an object-tight silhouette, while a photograph needs a wider
/// margin so a shadow or a soft edge is not cropped off. It is an enum rather
/// than a margin because that is the knob the API exposes and the margin is
/// derived from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// A render or a clean cut-out. Default crop margin 1.2.
    #[default]
    Synthetic,
    /// A photograph with background. Default crop margin 1.7.
    Real,
}

/// The file a request wants its splat written as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Format {
    /// The 3D Gaussian-splatting `.ply`.
    #[default]
    Ply,
    /// A binary glTF (`.glb`) with the Gaussian-splatting extension.
    Glb,
}

/// A single-image splat request.
#[derive(Debug, Clone, PartialEq)]
pub struct Generate {
    /// The input picture's encoded bytes (PNG or JPEG).
    pub image: Vec<u8>,
    /// The subject type, which only chooses the default crop margin.
    pub mode: Mode,
    /// A seed that makes the request reproducible; `None` draws one.
    pub seed: Option<u64>,
    /// Euler steps; `None` is the operator's default.
    pub steps: Option<u32>,
    /// Classifier-free guidance strength; `None` is the model default.
    pub cfg_strength: Option<f32>,
    /// Time-grid shift towards high noise; `None` is the model default.
    pub rescale_t: Option<f32>,
    /// How many Gaussians to place; `None` is the operator's default.
    pub num_gaussians: Option<u32>,
    /// Overrides the mode's crop margin when present.
    pub crop_margin: Option<f32>,
    /// The file the splat is written as.
    pub format: Format,
    /// How often a step runs in full; `None` is the operator's default.
    pub refresh_every: Option<u32>,
}

/// One finished splat file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Splat {
    /// The encoded file, in the request's [`Format`].
    pub bytes: Vec<u8>,
    /// Its MIME type.
    pub media_type: String,
}

/// Token accounting for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Denoiser sequence length times steps.
    pub completion_tokens: u32,
    /// Always zero today; kept so the shape matches the image wire.
    pub prompt_tokens: u32,
}

impl Usage {
    /// Sum of both counters, saturating.
    #[must_use]
    pub const fn total_tokens(self) -> u32 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// The splats of a finished request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Splats {
    /// Unix time the request finished.
    pub created: u64,
    /// The request's splats, in request order.
    pub splats: Vec<Splat>,
    /// The request's token accounting.
    pub usage: Usage,
}

/// One event of a streamed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// The job was admitted to the queue.
    Admitted,
    /// One Euler step finished.
    Progress {
        /// Zero-based step index.
        step: u32,
        /// Total steps.
        total_steps: u32,
    },
    /// The splat finished.
    Completed {
        /// Unix time the request started.
        created: u64,
        /// The finished splat.
        splat: Splat,
        /// The request's accounting.
        usage: Usage,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_defaults_to_nothing_named() {
        let request = Generate {
            image: vec![1, 2, 3],
            mode: Mode::Synthetic,
            seed: None,
            steps: None,
            cfg_strength: None,
            rescale_t: None,
            num_gaussians: None,
            crop_margin: None,
            format: Format::default(),
            refresh_every: None,
        };
        assert_eq!(request.format, Format::Ply);
        assert_eq!(request.mode, Mode::Synthetic);
        assert_eq!(request.num_gaussians, None);
    }

    #[test]
    fn usage_sums_without_overflow() {
        let usage = Usage {
            prompt_tokens: u32::MAX,
            completion_tokens: 1,
        };
        assert_eq!(usage.total_tokens(), u32::MAX);
    }

    #[test]
    fn a_completed_step_carries_the_finished_splat() {
        let splat = Splat {
            bytes: vec![0x70, 0x6c, 0x79],
            media_type: "model/ply".to_owned(),
        };
        let step = Step::Completed {
            created: 7,
            splat: splat.clone(),
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 614_700,
            },
        };
        match step {
            Step::Completed {
                created,
                splat: got,
                ..
            } => {
                assert_eq!(created, 7);
                assert_eq!(got, splat);
            }
            Step::Admitted | Step::Progress { .. } => panic!("expected a completed step"),
        }
    }
}

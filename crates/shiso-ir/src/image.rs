//! Image generation and editing in the IR: a prompt (and reference pictures)
//! in, encoded pictures out.
//!
//! These types are what `/v1/images/generations` and `/v1/images/edits` hand
//! to, and read back from, the typed cores in `shiso-runtime`. Like the rest
//! of this crate they carry no serde derives: each wire format parses its own
//! body into a [`Generate`] or an [`Edit`] and spells [`Images`] and [`Step`]
//! out in its own shape.
//!
//! # Wire spellings stay strings
//!
//! `size`, `aspect_ratio` and `output_format` are the caller's own words. The
//! size grammar (`WxH`, a named ratio, a named resolution, `auto`) and the
//! generator's format parser live below this crate, next to the limits they
//! are judged against, so the IR does not pre-chew them and each refusal
//! still names the field the caller wrote.
//!
//! # Streaming is a way of asking, not a field
//!
//! Whether the answer is read as one body or as a sequence of [`Step`]s is
//! chosen by which core is called, so a request carries no `stream` flag.

/// A text-to-image request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generate {
    /// What to draw. A blank prompt is refused by the generator's own
    /// validation.
    pub prompt: String,
    /// How many pictures to make; `None` is one. Between 1 and 10.
    pub n: Option<usize>,
    /// Output dimensions: `WxH`, a named aspect ratio, a named resolution, or
    /// `auto`. `None` is the loaded model's default.
    pub size: Option<String>,
    /// An aspect ratio such as `16:9`, read only when `size` names nothing.
    pub aspect_ratio: Option<String>,
    /// Denoising steps per picture, between 1 and 100; `None` is the
    /// operator's default.
    pub steps: Option<u32>,
    /// A seed that makes the request reproducible: picture `i` of it gets
    /// `seed + i`. `None` draws one so the pictures of one request differ.
    pub seed: Option<u64>,
    /// The encoding, `png` or `jpeg`; `None` is the generator's default.
    pub output_format: Option<String>,
    /// Compression level, between 1 and 100.
    pub output_compression: Option<u32>,
}

/// An image-edit request: a [`Generate`] and the pictures it edits from.
///
/// An edit makes exactly one picture, so [`Generate::n`] is `None` or one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit {
    /// The prompt and every knob that means what it means for a generation.
    pub generate: Generate,
    /// The reference pictures' encoded bytes, in the order the caller sent
    /// them. At least one.
    pub references: Vec<Vec<u8>>,
}

/// Token accounting across the pictures of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Prompt tokens, summed over the pictures.
    pub prompt_tokens: u32,
    /// Completion tokens, summed over the pictures.
    pub completion_tokens: u32,
}

impl Usage {
    /// Prompt plus completion tokens, saturating.
    #[must_use]
    pub const fn total_tokens(self) -> u32 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// One finished picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// The encoded file.
    pub bytes: Vec<u8>,
    /// Its MIME type.
    pub media_type: String,
}

/// The pictures of a finished request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Images {
    /// Unix time the request finished.
    pub created: u64,
    /// The pictures, in request order.
    pub images: Vec<Image>,
    /// The request's token accounting.
    pub usage: Usage,
}

/// One event of a streamed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// The next picture was admitted to the generator's queue. Sent once per
    /// picture, before any of its other events.
    Admitted,
    /// A preview decoded part-way through a picture. Always a PNG, whatever
    /// the request's output format; `index` counts across every picture of
    /// the request.
    Partial {
        /// Which preview this is, counting from zero.
        index: u32,
        /// The encoded preview.
        bytes: Vec<u8>,
    },
    /// A picture finished.
    Completed {
        /// Unix time the request started.
        created: u64,
        /// The finished picture.
        image: Image,
        /// The request's running total, this picture included.
        usage: Usage,
    },
}

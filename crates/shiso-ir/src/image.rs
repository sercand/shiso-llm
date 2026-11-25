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
    /// What the finished pictures' background must be.
    pub background: Background,
    /// The rendering quality the caller named. Echoed back; it changes
    /// nothing about how a picture is made.
    pub quality: Option<Quality>,
}

/// What a request asks of its pictures' background.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Background {
    /// The model decides: its own cut-out when it drew one, opaque
    /// otherwise.
    #[default]
    Auto,
    /// No transparency.
    Opaque,
    /// A cut-out, whatever the model drew. Needs png output.
    Transparent,
}

/// What a picture's background turned out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProducedBackground {
    /// Every pixel opaque.
    Opaque,
    /// A cut-out with real transparency.
    Transparent,
}

/// A rendering quality a caller may name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// `low`.
    Low,
    /// `medium`.
    Medium,
    /// `high`.
    High,
    /// `xhigh`.
    Xhigh,
    /// `max`.
    Max,
    /// `auto`.
    Auto,
    /// `standard`, the older models' spelling.
    Standard,
    /// `hd`, the older models' spelling.
    Hd,
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

/// Token accounting across the pictures of one request, by class: the
/// classes are priced apart, so they are counted apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Text tokens of the prompt, summed over the pictures.
    pub text_input_tokens: u32,
    /// Tokens the reference pictures occupied in the model's input, summed
    /// over the pictures.
    pub image_input_tokens: u32,
    /// Tokens of picture the model produced (latent cells times steps),
    /// summed over the pictures.
    pub image_output_tokens: u32,
}

impl Usage {
    /// Input tokens of both classes, saturating.
    #[must_use]
    pub const fn input_tokens(self) -> u32 {
        self.text_input_tokens
            .saturating_add(self.image_input_tokens)
    }

    /// Every token, saturating.
    #[must_use]
    pub const fn total_tokens(self) -> u32 {
        self.input_tokens().saturating_add(self.image_output_tokens)
    }
}

/// One finished picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// The encoded file.
    pub bytes: Vec<u8>,
    /// Its MIME type.
    pub media_type: String,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Whether it carries real transparency.
    pub background: ProducedBackground,
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
    /// Transparent when any picture is.
    pub background: ProducedBackground,
    /// The encoding every picture is in: `png` or `jpeg`.
    pub output_format: String,
    /// The quality the caller named, echoed.
    pub quality: Option<Quality>,
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
        /// The encoding the picture is in: `png` or `jpeg`.
        output_format: String,
        /// The quality the caller named, echoed.
        quality: Option<Quality>,
    },
}

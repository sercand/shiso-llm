//! The Qwen-Image prompt: a fixed system turn, the caller's text as the user
//! turn, and the assistant opener, encoded for the Qwen3-VL text encoder.
//!
//! The template is a raw string, not the checkpoint's chat template: the
//! reference formats it directly and the transformer was trained on what
//! that produces. The encoder reads the whole sequence, and the first
//! [`DROP_IDX`] rows of its output - the system turn - are dropped before
//! the transformer sees them. That count is the system turn's own token
//! count, recomputed from the vocabulary at load and checked against the
//! reference's 14, so a different vocabulary is refused rather than cut at
//! the wrong row.
//!
//! # Caller text
//!
//! A prompt that spells a control token (`<|im_start|>`, `<|image_pad|>`)
//! has it encoded as plain words. This deliberately differs from the
//! reference, which would read those spellings as control ids and let a
//! prompt open a turn or ask for an image the request never carried. A
//! prompt that spells none is encoded as one rendered string, exactly as the
//! reference encodes it; one that does is encoded in three pieces - the
//! template's head, the caller's text read as words, the template's tail.
//!
//! The kept prompt (everything after the system turn) is capped at
//! [`MAX_PROMPT_TOKENS`]; a longer one is refused with its count, never
//! truncated.
//!
//! # Edits
//!
//! An edit's user turn opens with one bracketed run of `<|image_pad|>` per
//! reference picture - `<image1><|vision_start|>...<|vision_end|>`, later
//! ones after a single space - before the caller's text. Each pad is one
//! image slot: a 32-pixel tile of the picture, a row of the vision tower's
//! output in the encoder, and four latent tokens in the transformer. The
//! pads are not counted against the cap. [`EditPromptIds::positions`] gives
//! the encoder's three rotary axes for the sequence: text advances all
//! three together, a picture holds the time axis and spreads its slots over
//! the other two, and the text after it resumes past the picture's longer
//! side.

use std::path::Path;

use shiso_vision::generator::ImageGenError;

use crate::diffusion::prompt::PromptTokenizer;

/// The system turn the reference prefixes to every text-to-image prompt.
const SYSTEM_TURN: &str =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
/// The user turn's opener, between the system turn and the caller's text.
const USER_OPEN: &str = "<|im_start|>user\n";
/// The user turn's close and the assistant opener, after the caller's text.
const TAIL: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// Encoder output rows dropped before the transformer: the system turn's
/// token count under the reference vocabulary.
pub(crate) const DROP_IDX: usize = 14;

/// The most kept prompt tokens (everything after the system turn) the
/// transformer is given.
pub(crate) const MAX_PROMPT_TOKENS: usize = 512;

/// The control token standing for one image slot.
const IMAGE_PAD: &str = "<|image_pad|>";

/// What an empty prompt is sent as: the vocabulary has no BOS token, so an
/// empty user turn would leave the encoder nothing of the caller's to read.
const EMPTY_PROMPT: &str = " ";

/// One prompt's encoder input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptIds {
    /// The full sequence the encoder reads, system turn included.
    pub ids: Vec<u32>,
    /// Leading encoder rows to drop: always [`DROP_IDX`].
    pub drop: usize,
}

impl PromptIds {
    /// The ids whose encoder rows the transformer reads.
    #[cfg(test)]
    pub(crate) fn kept(&self) -> &[u32] {
        &self.ids[self.drop..]
    }
}

/// One edit prompt's encoder input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EditPromptIds {
    /// The full sequence the encoder reads, system turn included.
    pub ids: Vec<u32>,
    /// Leading encoder rows to drop: always [`DROP_IDX`].
    pub drop: usize,
    /// Per reference, in order: its run of image slots in `ids` and the
    /// slot grid's `(rows, columns)`.
    pub images: Vec<(std::ops::Range<usize>, (usize, usize))>,
}

impl EditPromptIds {
    /// Which of the kept rows (after the drop) are image slots.
    pub(crate) fn image_mask_kept(&self) -> Vec<bool> {
        let mut mask = vec![false; self.ids.len() - self.drop];
        for (run, _) in &self.images {
            for i in run.clone() {
                mask[i - self.drop] = true;
            }
        }
        mask
    }

    /// The encoder's `[time, height, width]` rotary position of every id.
    ///
    /// Text takes the running position on all three axes. A picture whose
    /// first slot is reached at position `p` puts slot `(row, column)` at
    /// `[p, p + row, p + column]`, and the text after it resumes at
    /// `p + max(rows, columns)`.
    pub(crate) fn positions(&self) -> Vec<[u32; 3]> {
        let as_u32 = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        let mut positions = Vec::with_capacity(self.ids.len());
        let mut position = 0usize;
        let mut images = self.images.iter().peekable();
        while positions.len() < self.ids.len() {
            match images.peek() {
                Some((run, (rows, cols))) if run.start == positions.len() => {
                    for row in 0..*rows {
                        for col in 0..*cols {
                            positions.push([
                                as_u32(position),
                                as_u32(position + row),
                                as_u32(position + col),
                            ]);
                        }
                    }
                    position += (*rows).max(*cols);
                    images.next();
                }
                _ => {
                    positions.push([as_u32(position); 3]);
                    position += 1;
                }
            }
        }
        positions
    }
}

/// The Qwen-Image prompt encoder: the vocabulary and the system turn's ids.
#[derive(Debug)]
pub(crate) struct QwenImagePrompt {
    tokenizer: PromptTokenizer,
    system_ids: Vec<u32>,
}

impl QwenImagePrompt {
    /// Opens the tokenizer at `path` (`tokenizer.json`, or the processor
    /// directory holding it) and checks the system turn is [`DROP_IDX`]
    /// tokens under it.
    pub(crate) fn load(path: &Path) -> Result<Self, ImageGenError> {
        let tokenizer = PromptTokenizer::open(path)?;
        let system_ids = tokenizer.encode_template(SYSTEM_TURN)?;
        if system_ids.len() != DROP_IDX {
            return Err(ImageGenError::Checkpoint(format!(
                "the system turn encodes to {} tokens under {}, not the {DROP_IDX} the \
                 transformer was trained to have dropped; this is not the Qwen-Image vocabulary",
                system_ids.len(),
                path.display()
            )));
        }
        Ok(Self {
            tokenizer,
            system_ids,
        })
    }

    /// Encodes `prompt` into the encoder's input.
    ///
    /// # Errors
    /// [`ImageGenError::PromptTooLong`] when the kept tokens exceed
    /// [`MAX_PROMPT_TOKENS`].
    pub(crate) fn tokenize(&self, prompt: &str) -> Result<PromptIds, ImageGenError> {
        let prompt = if prompt.trim().is_empty() {
            EMPTY_PROMPT
        } else {
            prompt
        };
        let ids = if self.tokenizer.spells_control_token(prompt) {
            let mut ids = self
                .tokenizer
                .encode_template(&format!("{SYSTEM_TURN}{USER_OPEN}"))?;
            ids.extend(self.tokenizer.encode_caller(prompt)?);
            ids.extend(self.tokenizer.encode_template(TAIL)?);
            ids
        } else {
            // No control spelling: the one-string encoding is the
            // reference's own, merges across the turn boundaries included.
            self.tokenizer
                .encode_template(&format!("{SYSTEM_TURN}{USER_OPEN}{prompt}{TAIL}"))?
        };
        // The system turn ends in a control token, which no merge crosses,
        // so it encodes the same inside the sequence as alone; checked
        // anyway, because the drop below is only right if it holds.
        if ids.get(..DROP_IDX) != Some(self.system_ids.as_slice()) {
            return Err(ImageGenError::Backend(
                "the encoded prompt does not begin with the system turn's tokens".to_string(),
            ));
        }
        let kept = ids.len() - DROP_IDX;
        if kept > MAX_PROMPT_TOKENS {
            return Err(ImageGenError::PromptTooLong {
                tokens: u32::try_from(kept).unwrap_or(u32::MAX),
                max: u32::try_from(MAX_PROMPT_TOKENS).unwrap_or(u32::MAX),
            });
        }
        Ok(PromptIds {
            ids,
            drop: DROP_IDX,
        })
    }

    /// Encodes an edit's `prompt` with one run of image slots per
    /// reference; `slots` is each reference's `(rows, columns)`.
    ///
    /// # Errors
    /// [`ImageGenError::ReferenceImage`] for no references or an empty
    /// grid, and [`ImageGenError::PromptTooLong`] when the kept tokens
    /// that are not image slots exceed [`MAX_PROMPT_TOKENS`].
    pub(crate) fn tokenize_edit(
        &self,
        prompt: &str,
        slots: &[(usize, usize)],
    ) -> Result<EditPromptIds, ImageGenError> {
        if slots.is_empty() || slots.iter().any(|&(rows, cols)| rows == 0 || cols == 0) {
            return Err(ImageGenError::ReferenceImage(format!(
                "an edit prompt needs at least one reference with a non-empty slot grid, \
                 got {slots:?}"
            )));
        }
        let prompt = if prompt.trim().is_empty() {
            EMPTY_PROMPT
        } else {
            prompt
        };
        // One pad per picture here, widened to the picture's slot count
        // below: control tokens are atomic, so the neighbours encode the
        // same beside one pad as beside a thousand.
        let pictures = (1..=slots.len())
            .map(|n| format!("<image{n}><|vision_start|>{IMAGE_PAD}<|vision_end|>"))
            .collect::<Vec<_>>()
            .join(" ");
        let head = format!("{SYSTEM_TURN}{USER_OPEN}{pictures}");
        let short = if self.tokenizer.spells_control_token(prompt) {
            let mut ids = self.tokenizer.encode_template(&head)?;
            ids.extend(self.tokenizer.encode_caller(prompt)?);
            ids.extend(self.tokenizer.encode_template(TAIL)?);
            ids
        } else {
            self.tokenizer
                .encode_template(&format!("{head}{prompt}{TAIL}"))?
        };
        if short.get(..DROP_IDX) != Some(self.system_ids.as_slice()) {
            return Err(ImageGenError::Backend(
                "the encoded prompt does not begin with the system turn's tokens".to_string(),
            ));
        }
        let image_pad = match self.tokenizer.encode_template(IMAGE_PAD)?.as_slice() {
            [id] => *id,
            other => {
                return Err(ImageGenError::Checkpoint(format!(
                    "{IMAGE_PAD} encodes to {} tokens under this vocabulary, not one",
                    other.len()
                )));
            }
        };
        let found = short.iter().filter(|&&id| id == image_pad).count();
        if found != slots.len() {
            return Err(ImageGenError::Backend(format!(
                "the edit prompt encoded {found} image slots for {} references",
                slots.len()
            )));
        }
        let kept = short.len() - DROP_IDX - slots.len();
        if kept > MAX_PROMPT_TOKENS {
            return Err(ImageGenError::PromptTooLong {
                tokens: u32::try_from(kept).unwrap_or(u32::MAX),
                max: u32::try_from(MAX_PROMPT_TOKENS).unwrap_or(u32::MAX),
            });
        }
        let mut ids =
            Vec::with_capacity(short.len() + slots.iter().map(|s| s.0 * s.1).sum::<usize>());
        let mut images = Vec::with_capacity(slots.len());
        let mut grids = slots.iter();
        for id in short {
            if id == image_pad
                && let Some(&(rows, cols)) = grids.next()
            {
                let start = ids.len();
                ids.resize(start + rows * cols, image_pad);
                images.push((start..ids.len(), (rows, cols)));
            } else {
                ids.push(id);
            }
        }
        Ok(EditPromptIds {
            ids,
            drop: DROP_IDX,
            images,
        })
    }
}

//! Subword tokenization, exactly as the reference runs it.
//!
//! The tokenizer is loaded from `tokenizer.json` unchanged. Its own
//! truncation setting cuts an over-long input at 128 ids and re-appends the
//! closing marker, and it pads to the batch; the reference keeps the ids whose
//! attention mask is 1 and nothing else. Reproducing that is the point: a
//! command longer than the window must give the same 128 ids the classifier
//! was calibrated on, trailing marker included, not the first 128 of an
//! untruncated encoding.

use std::path::Path;

use tokenizers::Tokenizer;

use super::ClassifyError;

/// The longest sequence the classifier is run on.
pub(super) const MAX_TOKENS: usize = 128;

/// The checkpoint's tokenizer.
#[derive(Debug)]
pub(super) struct WordPiece {
    inner: Tokenizer,
}

impl WordPiece {
    /// Loads `tokenizer.json` and leaves its configuration as shipped.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::Tokenizer`] when the file cannot be read as a
    /// tokenizer.
    pub(super) fn load(path: &Path) -> Result<Self, ClassifyError> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| ClassifyError::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    /// Ids the tokenizer can emit, added tokens included.
    pub(super) fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// The ids of `text` with the opening and closing markers, the padding
    /// dropped, at most [`MAX_TOKENS`] long.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::Tokenizer`] when encoding fails.
    pub(super) fn ids(&self, text: &str) -> Result<Vec<u32>, ClassifyError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| ClassifyError::Tokenizer(e.to_string()))?;
        Ok(encoding
            .get_ids()
            .iter()
            .zip(encoding.get_attention_mask())
            .filter(|(_, mask)| **mask == 1)
            .map(|(id, _)| *id)
            .take(MAX_TOKENS)
            .collect())
    }
}

//! One utterance through the encoder and the transducer.
//!
//! Mel rows are buffered on the host and handed to the device as soon as they
//! complete an attention chunk. The encoder gives the same frames under any push
//! schedule, so the buffering changes the number of launches and never adds
//! latency past the chunk the model itself waits for.

use std::sync::Arc;

use shiso_audio::{AudioError, MelFrames, RecognizerStream, TokenEvent};

use super::Model;
use super::encoder::{Encoder, EncoderState};
use super::transducer::{DecoderState, Transducer};
use crate::audio_ops::load::upload_f32;

/// A stream of the Nemotron transducer.
#[derive(Debug)]
pub(crate) struct AsrStream {
    model: Arc<Model>,
    encoder: EncoderState,
    decoder: DecoderState,
    /// Mel rows not yet handed to the encoder, row-major.
    buffered: Vec<f32>,
    /// Mel rows per hand-off once the first chunk is out: `8 * (R + 1)`.
    batch: usize,
    /// Mel rows the first chunk needs: `1 + 8R`.
    first: usize,
    /// Mel rows handed to the encoder so far.
    handed: usize,
    finished: bool,
}

impl AsrStream {
    pub(crate) fn new(
        model: Arc<Model>,
        encoder: EncoderState,
        decoder: DecoderState,
        lookahead: usize,
    ) -> Self {
        Self {
            model,
            encoder,
            decoder,
            buffered: Vec::new(),
            batch: 8 * (lookahead + 1),
            first: 1 + 8 * lookahead,
            handed: 0,
            finished: false,
        }
    }

    fn flush(&mut self, finishing: bool) -> Result<Vec<TokenEvent>, AudioError> {
        let m = &*self.model;
        let mels = m.config.mels;
        let rows = self.buffered.len() / mels;
        // Hand over exactly what completes chunks: the first needs 1 + 8R rows,
        // each later one 8(R + 1). Holding rows back past that would delay
        // every chunk by the difference.
        let take = if finishing {
            rows
        } else {
            let first = self.first.saturating_sub(self.handed);
            if self.handed < self.first && rows < first {
                0
            } else if self.handed < self.first {
                first + (rows - first) / self.batch * self.batch
            } else {
                rows / self.batch * self.batch
            }
        };
        let encoder = Encoder::new(&m.ops, &m.weights, &m.config);
        let transducer = Transducer::new(&m.ops, &m.weights, &m.config);
        let mut events = Vec::new();
        if take > 0 {
            let dev = upload_f32(&m.ops, &self.buffered[..take * mels])?;
            let frames = encoder.push(&mut self.encoder, dev.view(), take, false)?;
            events.extend(transducer.feed(&mut self.decoder, &frames)?);
            self.buffered.drain(..take * mels);
            self.handed += take;
        }
        if finishing {
            let tail = encoder.finish(&mut self.encoder)?;
            events.extend(transducer.feed(&mut self.decoder, &tail)?);
        }
        Ok(events)
    }
}

impl RecognizerStream for AsrStream {
    fn push(&mut self, mel: &MelFrames) -> Result<Vec<TokenEvent>, AudioError> {
        if self.finished {
            return Err(AudioError::Checkpoint {
                what: "push on a finished stream".into(),
            });
        }
        if mel.valid() > 0 && mel.n_mels() != self.model.config.mels {
            return Err(AudioError::BadSpec {
                reason: format!(
                    "{} mel bins, this model reads {}",
                    mel.n_mels(),
                    self.model.config.mels
                ),
            });
        }
        self.buffered
            .extend_from_slice(&mel.as_slice()[..mel.valid() * mel.n_mels()]);
        self.flush(false)
    }

    fn finish(&mut self) -> Result<Vec<TokenEvent>, AudioError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        self.flush(true)
    }

    fn frames(&self) -> u32 {
        u32::try_from(self.encoder.frames()).unwrap_or(u32::MAX)
    }
}

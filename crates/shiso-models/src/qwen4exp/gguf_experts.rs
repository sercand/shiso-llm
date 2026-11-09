//! The routed banks of a GGUF checkpoint, read from disk into the expert
//! stream's slots.
//!
//! A mixed-precision export stores the routed banks in different block
//! formats from layer to layer - a 2-bit gate in one layer, a 3-bit one in
//! the next, a 2-bit down projection in a few - so one expert is not one size
//! everywhere. The stream's pool needs one width per array, so each array is
//! sized at its widest layer, a narrower layer's expert fills the front of its
//! slot, and the routed stages read slots at the pool's stride (see
//! `Operand::ExpertStride`).

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;

use shiso_core::Backend;
use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::sparse::stream::{ExpertSource, ExpertStream};
use shiso_engine::sparse::weights::WeightPart;
use shiso_modelfile::GgufFile;
use shiso_modelfile::st_header::open_uncached;

/// The three routed banks of a layer, in stream array order.
const BANKS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

/// Where one layer's bank lives: the shard file, the bank's offset in it and
/// one expert's bytes.
#[derive(Debug, Clone, Copy)]
struct Bank {
    file: usize,
    offset: u64,
    expert_bytes: u64,
}

/// Reads a GGUF checkpoint's routed experts, every layer, whatever shard
/// holds them.
#[derive(Debug)]
pub struct GgufExperts {
    files: Vec<File>,
    banks: Vec<[Bank; 3]>,
    widths: [u64; 3],
}

impl GgufExperts {
    /// Locates the three routed banks of each of `layers` layers in `file`.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming an absent bank,
    /// [`CoreError::ShapeMismatch`] for a bank that is not `experts` equal
    /// slices, and [`CoreError::Io`] when a shard cannot be opened.
    pub fn open(file: &GgufFile, layers: u32, experts: u32) -> CoreResult<Self> {
        let mut paths: HashMap<PathBuf, usize> = HashMap::new();
        let mut files = Vec::new();
        let mut banks = Vec::with_capacity(layers as usize);
        let mut widths = [0u64; 3];
        for layer in 0..layers {
            let mut row = [Bank {
                file: 0,
                offset: 0,
                expert_bytes: 0,
            }; 3];
            for (i, name) in BANKS.iter().enumerate() {
                let key = format!("blk.{layer}.{name}.weight");
                let desc = file
                    .find(&key)
                    .ok_or_else(|| CoreError::ConfigMissing { key: key.clone() })?;
                let per = desc.bytes / u64::from(experts.max(1));
                if per * u64::from(experts) != desc.bytes {
                    return Err(CoreError::ShapeMismatch {
                        field: "routed expert slice bytes",
                        expected: per * u64::from(experts),
                        got: desc.bytes,
                    });
                }
                let path = file.path_of(desc).to_path_buf();
                let index = if let Some(&index) = paths.get(&path) {
                    index
                } else {
                    files.push(open_uncached(&path)?);
                    paths.insert(path, files.len() - 1);
                    files.len() - 1
                };
                row[i] = Bank {
                    file: index,
                    offset: u64::try_from(file.range_of(desc).start).unwrap_or(u64::MAX),
                    expert_bytes: per,
                };
                widths[i] = widths[i].max(per);
            }
            banks.push(row);
        }
        Ok(Self {
            files,
            banks,
            widths,
        })
    }

    /// Routed layers this source reads.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.banks.len()
    }

    /// Bytes one slot occupies across the three arrays.
    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        self.widths.iter().sum()
    }

    /// Bytes of every routed bank, every layer, as stored.
    #[must_use]
    pub fn bank_bytes(&self, experts: u32) -> u64 {
        self.banks
            .iter()
            .flatten()
            .map(|bank| bank.expert_bytes * u64::from(experts))
            .sum()
    }
}

impl ExpertSource for GgufExperts {
    fn parts(&self) -> &'static [WeightPart] {
        &[WeightPart::Whole]
    }

    fn layout(&self) -> Vec<u64> {
        self.widths.to_vec()
    }

    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()> {
        let bank = self
            .banks
            .get(layer as usize)
            .and_then(|row| row.get(array))
            .ok_or(CoreError::OutOfBounds {
                offset: u64::from(layer),
                len: 1,
                buffer_bytes: self.banks.len() as u64,
            })?;
        let len = usize::try_from(bank.expert_bytes).unwrap_or(usize::MAX);
        let width = dst.len() as u64;
        let head = dst.get_mut(..len).ok_or(CoreError::OutOfBounds {
            offset: 0,
            len: bank.expert_bytes,
            buffer_bytes: width,
        })?;
        let file = self.files.get(bank.file).ok_or(CoreError::OutOfBounds {
            offset: bank.file as u64,
            len: 1,
            buffer_bytes: self.files.len() as u64,
        })?;
        file.read_exact_at(head, bank.offset + u64::from(expert) * bank.expert_bytes)
            .map_err(CoreError::Io)
    }
}

/// Opens the stream over `source`'s banks with `slots` slots, at least one
/// widest pass's worth of experts.
///
/// # Errors
///
/// Whatever [`ExpertStream::open`] reports.
pub fn open_stream(
    backend: &Arc<dyn Backend>,
    source: Arc<GgufExperts>,
    layers: u32,
    experts: u32,
    used: u32,
    slots: u32,
    widest_pass: u32,
) -> CoreResult<ExpertStream> {
    let min_slots = widest_pass.saturating_mul(used).min(experts);
    ExpertStream::open(backend, source, layers, experts, slots, min_slots)
}

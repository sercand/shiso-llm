//! Per-layer residual rows on disk, in the layout the independent reference
//! writes, for bisecting a diverging position layer by layer.
//!
//! A logits row says *that* two engines disagree at a frontier. Which layer
//! they started disagreeing at needs the residual itself at every layer
//! boundary, from both sides, diffed element by element. This module writes the
//! engine's half; `scripts/reference/qwen4exp_fp8_ref.py capture
//! --layer-dump-dir` writes the reference's half, and
//! `scripts/reference/bisect_layers.py` compares the two. The layout is the one
//! `scripts/reference/LAYER_DUMP.md` proposes:
//!
//! ```text
//! <dir>/index.json                        case id -> {prompt_file, frontier, prompt_tokens}
//! <dir>/<case-id>/embed.f32.npy           the residual after the stream broadcast
//! <dir>/<case-id>/L{nn}.mixer.f32.npy     after layer nn's token-mixer combine
//! <dir>/<case-id>/L{nn}.f32.npy           after layer nn's feed-forward combine
//! ```
//!
//! Every file is a numpy `.npy` v1 file, dtype `<f4`, shape `[streams, hidden]`
//! (`[4, 2560]` for the one checkpoint of this family): the raw residual
//! streams of the frontier's token, stream-major, as the stack holds them. `f32`
//! rather than `f16`, because the comparison grades relative L2 error per layer
//! and half-precision rounding, about 5e-4 relative, would bury the early
//! layers' errors the bisection is looking for. The index is the row dump's, so
//! `frontier` is the 0-based position, as in [`crate::rows`].
//!
//! `nn` is the layer index, zero-padded to two digits; a model with a hundred
//! layers or more gets three, and so on, so the names always sort in pass
//! order.
//!
//! # Writes once
//!
//! As with the row dump: the index is checked before the checkpoint loads, each
//! case directory is created fresh, and every file is opened `create_new`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use shiso_core::backend::Backend;
use shiso_core::ids::{LayerIdx, TokenId};
use shiso_core::model::Model;
use shiso_engine::sparse::tap::{ResidualTap, TapPoint};
use shiso_models::qwen4exp::Qwen4ExpModel;

use crate::compare::decode_row;
use crate::replay::ReplayResult;
use crate::rows::{INDEX_FILE, IndexEntry, RowError, check_id, encode_f32_npy, write_new};

/// What can go wrong writing a case's layers.
#[derive(Debug, thiserror::Error)]
pub enum LayerError {
    /// Writing a file, or naming one, failed.
    #[error(transparent)]
    Row(#[from] RowError),
    /// The tap does not hold one row for every boundary of the pass, in order.
    ///
    /// Refused rather than written in part: a bisection over a dump with a
    /// layer missing would name the layer after the gap as the first to
    /// diverge.
    #[error("case {case}: the tap is not a whole pass ({detail})")]
    Incomplete {
        /// The case.
        case: String,
        /// What was missing or out of place.
        detail: String,
    },
}

/// Digits the layer index is padded to in a file name, for `n_layers` layers.
fn layer_digits(n_layers: u32) -> usize {
    n_layers.saturating_sub(1).max(1).to_string().len().max(2)
}

/// The file one tap point is written to, for a model of `n_layers` layers.
#[must_use]
pub fn file_name(point: TapPoint, n_layers: u32) -> String {
    let width = layer_digits(n_layers);
    match point {
        TapPoint::Embed => "embed.f32.npy".to_owned(),
        TapPoint::Mixer(layer) => format!("L{:0width$}.mixer.f32.npy", layer.get()),
        TapPoint::Layer(layer) => format!("L{:0width$}.f32.npy", layer.get()),
    }
}

/// The tap points one whole pass over `n_layers` layers copies, in pass order.
#[must_use]
pub fn pass_points(n_layers: u32) -> Vec<TapPoint> {
    let mut points = Vec::with_capacity(1 + 2 * n_layers as usize);
    points.push(TapPoint::Embed);
    for index in 0..n_layers {
        let layer = LayerIdx::new(index);
        points.push(TapPoint::Mixer(layer));
        points.push(TapPoint::Layer(layer));
    }
    points
}

/// A directory being filled with per-layer rows.
#[derive(Debug)]
pub struct LayerDump {
    dir: PathBuf,
    index: BTreeMap<String, IndexEntry>,
}

impl LayerDump {
    /// Opens `dir` for a dump, creating it if needed.
    ///
    /// # Errors
    ///
    /// [`RowError::Exists`] when the directory already holds an index, and
    /// [`RowError::Io`] when it cannot be created.
    pub fn create(dir: &Path) -> Result<Self, RowError> {
        std::fs::create_dir_all(dir).map_err(|source| RowError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let index = dir.join(INDEX_FILE);
        if index.exists() {
            return Err(RowError::Exists { path: index });
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            index: BTreeMap::new(),
        })
    }

    /// The directory case `id`'s files go in.
    ///
    /// # Errors
    ///
    /// [`RowError::BadId`] when `id` would not be a plain file name.
    pub fn case_dir(&self, id: &str) -> Result<PathBuf, RowError> {
        check_id(id)?;
        Ok(self.dir.join(id))
    }

    /// Refuses up front if any of `ids` already has a directory here.
    ///
    /// # Errors
    ///
    /// [`RowError::Exists`] naming the first one, or [`RowError::BadId`].
    pub fn check_free<'a>(&self, ids: impl IntoIterator<Item = &'a str>) -> Result<(), RowError> {
        for id in ids {
            let path = self.case_dir(id)?;
            if path.exists() {
                return Err(RowError::Exists { path });
            }
        }
        Ok(())
    }

    /// Writes one case's rows and remembers its index entry.
    ///
    /// # Errors
    ///
    /// [`LayerError::Incomplete`] when `tap` is not exactly one row per
    /// boundary of an `n_layers`-layer pass in pass order,
    /// [`RowError::Duplicate`], [`RowError::Exists`] when the case directory or
    /// a file in it appeared since, and [`RowError::Io`].
    pub fn write(
        &mut self,
        id: &str,
        tap: &ResidualTap,
        n_layers: u32,
        entry: IndexEntry,
    ) -> Result<(), LayerError> {
        if self.index.contains_key(id) {
            return Err(RowError::Duplicate { id: id.to_owned() }.into());
        }
        let want = pass_points(n_layers);
        let got: Vec<TapPoint> = tap.records().iter().map(|r| r.point).collect();
        if got != want {
            let at = want.iter().zip(&got).position(|(w, g)| w != g);
            return Err(LayerError::Incomplete {
                case: id.to_owned(),
                detail: format!(
                    "{} rows for {} boundaries, first difference at {at:?}",
                    got.len(),
                    want.len()
                ),
            });
        }
        let dir = self.case_dir(id)?;
        std::fs::create_dir(&dir).map_err(|source| {
            if source.kind() == std::io::ErrorKind::AlreadyExists {
                RowError::Exists { path: dir.clone() }
            } else {
                RowError::Io {
                    path: dir.clone(),
                    source,
                }
            }
        })?;
        let shape = [tap.streams() as usize, tap.hidden() as usize];
        for record in tap.records() {
            let bytes = encode_f32_npy(&shape, &record.values)?;
            write_new(&dir.join(file_name(record.point, n_layers)), &bytes)?;
        }
        self.index.insert(id.to_owned(), entry);
        Ok(())
    }

    /// Writes `index.json` and returns its path.
    ///
    /// # Errors
    ///
    /// [`RowError::Exists`] when an index appeared since
    /// [`LayerDump::create`], and [`RowError::Io`].
    pub fn finish(self) -> Result<PathBuf, RowError> {
        let path = self.dir.join(INDEX_FILE);
        let mut text = serde_json::to_string_pretty(&self.index)?;
        text.push('\n');
        write_new(&path, text.as_bytes())?;
        Ok(path)
    }
}

/// The logits row after `tokens`, and the tap over the pass that produced it.
///
/// Every chunk but the last is fed untapped; the tap is armed for the final
/// single-token pass only, so its rows are the frontier's. That pass runs
/// without device graphs (see [`shiso_engine::sparse::tap`]), which is also why
/// the tap is disarmed on every path out of here: a stack left armed would run
/// every later pass eagerly.
///
/// # Errors
///
/// [`crate::replay::ReplayError::Engine`] when the session, a pass or the
/// readback fails, including the tap's own copies.
pub fn frontier_layers(
    backend: &dyn Backend,
    model: &mut Qwen4ExpModel,
    tokens: &[TokenId],
    chunk: usize,
) -> ReplayResult<(Vec<f32>, ResidualTap)> {
    let session = model.open_session()?;
    let result = frontier_layers_in(backend, model, session, tokens, chunk);
    let tap = model.stack_mut().take_residual_tap();
    let closed = model.close_session(session);
    let row = result?;
    closed?;
    let tap = tap.ok_or(shiso_core::CoreError::Unsupported {
        what: "a residual tap that was disarmed during the pass it was armed for",
        backend: "bench",
    })?;
    Ok((row, tap))
}

/// The body of [`frontier_layers`], so the session is closed and the tap
/// disarmed on either path.
fn frontier_layers_in(
    backend: &dyn Backend,
    model: &mut Qwen4ExpModel,
    session: shiso_core::ids::SessionId,
    tokens: &[TokenId],
    chunk: usize,
) -> ReplayResult<Vec<f32>> {
    let chunk = chunk.max(1);
    let split = tokens.len().saturating_sub(1);
    let (head, tail) = tokens.split_at(split.min(tokens.len()));
    for piece in head.chunks(chunk) {
        model.prefill_chunk(session, piece)?;
    }
    model.stack_mut().arm_residual_tap()?;
    let last = model.prefill_last(session, tail)?;
    let view = last.row(0)?;
    let mut bytes = vec![0u8; usize::try_from(view.len()).unwrap_or(usize::MAX)];
    backend.copy_d2h(view, &mut bytes)?;
    Ok(decode_row(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rows::read_npy_header;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "shiso-layers-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        drop(std::fs::remove_dir_all(&dir));
        dir
    }

    /// A tap over `n_layers` layers whose every value names its point and
    /// its index, so a read-back can tell files apart.
    fn synthetic(n_layers: u32, streams: u32, hidden: u32) -> ResidualTap {
        let mut tap = ResidualTap::new(streams, hidden).expect("a shape");
        for (k, point) in pass_points(n_layers).into_iter().enumerate() {
            let k = f32::from(u16::try_from(k).expect("a small pass"));
            let values = (0..streams * hidden)
                .map(|i| k * 1000.0 + f32::from(u16::try_from(i).expect("a small row")))
                .collect();
            tap.push(point, values).expect("one row");
        }
        tap
    }

    #[test]
    fn names_follow_the_proposed_layout_and_sort_in_pass_order() {
        assert_eq!(file_name(TapPoint::Embed, 48), "embed.f32.npy");
        assert_eq!(
            file_name(TapPoint::Mixer(LayerIdx::new(7)), 48),
            "L07.mixer.f32.npy"
        );
        assert_eq!(
            file_name(TapPoint::Layer(LayerIdx::new(47)), 48),
            "L47.f32.npy"
        );
        assert_eq!(
            file_name(TapPoint::Layer(LayerIdx::new(3)), 120),
            "L003.f32.npy"
        );
        let points = pass_points(48);
        assert_eq!(points.len(), 97);
        assert_eq!(points[1], TapPoint::Mixer(LayerIdx::FIRST));
        assert_eq!(points[96], TapPoint::Layer(LayerIdx::new(47)));
        // Across layers the names sort in pass order. Within one layer the
        // plain name sorts before the mixer one, which is the reverse of pass
        // order, so a reader builds names rather than trusting a listing.
        let mut layers: Vec<String> = (0..48)
            .map(|i| file_name(TapPoint::Layer(LayerIdx::new(i)), 48))
            .collect();
        let before = layers.clone();
        layers.sort();
        assert_eq!(layers, before);
    }

    #[test]
    fn a_case_writes_every_boundary_as_a_stream_major_f32_array() {
        let dir = scratch("write");
        let mut dump = LayerDump::create(&dir).expect("fresh dir");
        let entry = IndexEntry {
            prompt_file: "p.txt".to_owned(),
            frontier: 14,
            prompt_tokens: 15,
        };
        let tap = synthetic(3, 4, 5);
        dump.write("case_a", &tap, 3, entry.clone())
            .expect("a whole pass");
        let index = dump.finish().expect("index");

        let case = dir.join("case_a");
        let mut names: Vec<String> = std::fs::read_dir(&case)
            .expect("case dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "L00.f32.npy",
                "L00.mixer.f32.npy",
                "L01.f32.npy",
                "L01.mixer.f32.npy",
                "L02.f32.npy",
                "L02.mixer.f32.npy",
                "embed.f32.npy",
            ]
        );
        // L01.mixer is the fourth point of the pass: embed, L00.mixer, L00,
        // L01.mixer. Its stream 2, element 3 is flat index 2 * 5 + 3.
        let bytes = std::fs::read(case.join("L01.mixer.f32.npy")).expect("file");
        let header = read_npy_header(&bytes).expect("header");
        assert_eq!(header.descr, "<f4");
        assert!(!header.fortran_order);
        assert_eq!(header.shape, vec![4, 5]);
        assert_eq!(header.data_offset % 64, 0);
        assert_eq!(bytes.len(), header.data_offset + 4 * 20);
        let at = header.data_offset + 4 * (2 * 5 + 3);
        let value = f32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
        assert_eq!(value.to_bits(), 3013.0f32.to_bits());

        let parsed: BTreeMap<String, IndexEntry> =
            serde_json::from_str(&std::fs::read_to_string(&index).expect("index")).expect("json");
        assert_eq!(parsed.get("case_a"), Some(&entry));
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn a_used_directory_or_an_incomplete_tap_is_refused() {
        let dir = scratch("refuse");
        let entry = IndexEntry {
            prompt_file: "p.txt".to_owned(),
            frontier: 0,
            prompt_tokens: 1,
        };
        let mut dump = LayerDump::create(&dir).expect("fresh dir");
        let mut short = ResidualTap::new(1, 2).expect("a shape");
        short.push(TapPoint::Embed, vec![0.0; 2]).expect("row");
        short
            .push(TapPoint::Mixer(LayerIdx::FIRST), vec![0.0; 2])
            .expect("row");
        assert!(matches!(
            dump.write("short", &short, 1, entry.clone()),
            Err(LayerError::Incomplete { .. })
        ));
        assert!(!dir.join("short").exists(), "nothing written for a refusal");
        assert!(matches!(dump.case_dir("../x"), Err(RowError::BadId { .. })));

        dump.write("whole", &synthetic(1, 1, 2), 1, entry.clone())
            .expect("whole");
        assert!(matches!(
            dump.write("whole", &synthetic(1, 1, 2), 1, entry.clone()),
            Err(LayerError::Row(RowError::Duplicate { .. }))
        ));
        dump.finish().expect("index");
        assert!(matches!(
            LayerDump::create(&dir),
            Err(RowError::Exists { .. })
        ));
        std::fs::remove_file(dir.join(INDEX_FILE)).expect("clear index");
        let again = LayerDump::create(&dir).expect("no index now");
        assert!(matches!(
            again.check_free(["whole"]),
            Err(RowError::Exists { .. })
        ));
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}

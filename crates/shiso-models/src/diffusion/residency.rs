//! Which pipeline components stay loaded, and how a swapped one comes back
//! fast.
//!
//! A text-to-image pipeline holds three components: a conditioning encoder
//! (read once per prompt), a denoiser (read every step) and a decoder (read
//! once per image). On a host whose working set holds all three plus a job's
//! activations they stay [`Residency::Resident`]. On a smaller one the
//! encoder and the denoiser take turns ([`Residency::Swap`]): the encoder is
//! loaded for the encode and dropped before the denoiser is imported, and
//! the decoder - small - stays. [`plan`] decides per job from byte counts
//! alone, so it is a pure function the tables below pin.
//!
//! # Swap-in speed
//!
//! Weights are imported as no-copy wraps of the file mapping, so "loading"
//! a swapped component is the page cache filling: left to the faults of the
//! first touch it reads the file in page-sized pieces at a fraction of the
//! disk's sequential rate. [`Prefetch`] reads exactly the byte ranges the
//! component will import, in large chunks on a few threads, into nothing but
//! the page cache, before the import - the import then finds every page
//! cached. It never allocates device memory and never maps anything, which is
//! what makes it safe to run on a background thread beside device work.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Instant;

use shiso_vision::generator::ImageGenError;

/// The operator's residency choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidencyMode {
    /// Resident when everything fits, else swap.
    Auto,
    /// Everything loaded across jobs; a job that does not fit is refused.
    Resident,
    /// Encoder and denoiser take turns, whatever fits.
    Swap,
}

/// Bytes each component holds while loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Footprint {
    /// The conditioning encoder's imported weights.
    pub encoder: u64,
    /// The denoiser's imported weights.
    pub denoiser: u64,
    /// The decoder's (widened) weights.
    pub decoder: u64,
}

/// One job's working memory beyond the weights, per phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JobPeaks {
    /// The encoder's activations over the prompt.
    pub encode: u64,
    /// The denoiser's activations at the job's token count, plus the
    /// prompt's cached keys and values.
    pub denoise: u64,
    /// The decoder's working set at the job's image size.
    pub decode: u64,
}

/// What [`plan`] decides from.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanDesc {
    /// The operator's choice.
    pub mode: ResidencyMode,
    /// The device's working set (the most it keeps resident without paging).
    pub working_set: u64,
    /// Bytes left for everything else on the host.
    pub reserve: u64,
    /// Component sizes.
    pub footprint: Footprint,
    /// The job's activation peaks.
    pub peaks: JobPeaks,
}

/// How a job runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Residency {
    /// Every component stays loaded across jobs.
    Resident,
    /// The encoder is loaded for the encode and dropped before the denoiser
    /// is imported; the decoder stays.
    Swap {
        /// The denoiser fits beside the decode and stays loaded after the
        /// job (dropped when a later job needs the encoder); otherwise it is
        /// dropped before the decode.
        keep_denoiser: bool,
    },
}

/// A job that fits no residency: its largest phase and the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shortfall {
    /// Bytes the cheapest admissible residency needs at its peak.
    pub needed: u64,
    /// `working_set - reserve`.
    pub budget: u64,
}

/// Decides the job's residency.
///
/// # Errors
/// [`Shortfall`] when no residency the mode admits fits the budget.
pub(crate) fn plan(desc: &PlanDesc) -> Result<Residency, Shortfall> {
    let budget = desc.working_set.saturating_sub(desc.reserve);
    let f = desc.footprint;
    let p = desc.peaks;
    let resident = f.encoder + f.denoiser + f.decoder + p.encode.max(p.denoise).max(p.decode);
    let encode_phase = f.encoder + f.decoder + p.encode;
    let denoise_phase = f.denoiser + f.decoder + p.denoise;
    let decode_alone = f.decoder + p.decode;
    let swap = encode_phase.max(denoise_phase).max(decode_alone);
    let keep_denoiser = f.denoiser + f.decoder + p.decode <= budget;
    let shortfall = |needed| Shortfall { needed, budget };
    match desc.mode {
        ResidencyMode::Resident | ResidencyMode::Auto if resident <= budget => {
            Ok(Residency::Resident)
        }
        ResidencyMode::Resident => Err(shortfall(resident)),
        ResidencyMode::Auto | ResidencyMode::Swap if swap <= budget => {
            Ok(Residency::Swap { keep_denoiser })
        }
        ResidencyMode::Auto | ResidencyMode::Swap => Err(shortfall(swap)),
    }
}

/// One contiguous byte range of a weight file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileSpan {
    /// The file.
    pub path: PathBuf,
    /// First byte.
    pub start: u64,
    /// One past the last byte.
    pub end: u64,
}

impl FileSpan {
    /// Bytes the span covers.
    pub(crate) fn len(&self) -> u64 {
        self.end - self.start
    }
}

/// Sorts `spans` and merges ranges of one file whose gap is under
/// [`MERGE_GAP`]: reading a small gap costs less than a second request.
pub(crate) fn coalesce(mut spans: Vec<FileSpan>) -> Vec<FileSpan> {
    spans.sort_by(|a, b| a.path.cmp(&b.path).then(a.start.cmp(&b.start)));
    let mut out: Vec<FileSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        if let Some(last) = out.last_mut()
            && last.path == span.path
            && span.start <= last.end.saturating_add(MERGE_GAP)
        {
            last.end = last.end.max(span.end);
            continue;
        }
        out.push(span);
    }
    out
}

/// Splits `spans` after their first `bytes` bytes: what fits a budget now,
/// and the rest for later.
pub(crate) fn split_at(spans: &[FileSpan], bytes: u64) -> (Vec<FileSpan>, Vec<FileSpan>) {
    let (mut head, mut tail) = (Vec::new(), Vec::new());
    let mut left = bytes;
    for span in spans {
        if left >= span.len() {
            left -= span.len();
            head.push(span.clone());
        } else if left > 0 {
            let cut = span.start + left;
            head.push(FileSpan {
                end: cut,
                ..span.clone()
            });
            tail.push(FileSpan {
                start: cut,
                ..span.clone()
            });
            left = 0;
        } else {
            tail.push(span.clone());
        }
    }
    (head, tail)
}

/// Total bytes of `spans`.
pub(crate) fn total_bytes(spans: &[FileSpan]) -> u64 {
    spans.iter().map(FileSpan::len).sum()
}

/// Gap below which two ranges of one file are read as one.
const MERGE_GAP: u64 = 1 << 20;
/// Bytes one read request asks for: large enough that the disk streams,
/// small enough that four readers interleave across a component.
const CHUNK: u64 = 8 << 20;
/// Concurrent readers: a solid-state drive reaches its sequential rate only with
/// several requests in flight.
const READERS: usize = 4;

/// What a finished prefetch read.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefetchStats {
    /// Bytes read.
    pub bytes: u64,
    /// Wall time of the reads.
    pub seconds: f64,
}

impl PrefetchStats {
    /// Read rate in MB/s (10^6 bytes).
    #[expect(
        clippy::cast_precision_loss,
        reason = "a byte count for a rate printed to one decimal"
    )]
    pub(crate) fn mb_per_s(&self) -> f64 {
        self.bytes as f64 / 1e6 / self.seconds.max(1e-9)
    }
}

/// A background read of byte ranges into the page cache.
///
/// Dropping it stops the readers at the next chunk and joins them, so a
/// prefetch never outlives its owner.
#[derive(Debug)]
pub(crate) struct Prefetch {
    label: &'static str,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<PrefetchStats, String>>>,
}

impl Prefetch {
    /// Starts reading `spans` on a background thread.
    pub(crate) fn spawn(label: &'static str, spans: Vec<FileSpan>) -> Result<Self, ImageGenError> {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name(format!("prefetch-{label}"))
            .spawn(move || read_all(&spans, &flag))
            .map_err(|e| ImageGenError::Backend(format!("starting the {label} prefetch: {e}")))?;
        Ok(Self {
            label,
            stop,
            handle: Some(handle),
        })
    }

    /// Waits for every read to finish.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] when a read failed: the import that
    /// follows would read the same bytes, so the file is refused here.
    pub(crate) fn wait(mut self) -> Result<PrefetchStats, ImageGenError> {
        let handle = self.handle.take().ok_or_else(|| {
            ImageGenError::Backend(format!("the {} prefetch was already joined", self.label))
        })?;
        handle
            .join()
            .map_err(|_| ImageGenError::Backend(format!("the {} prefetch panicked", self.label)))?
            .map_err(|e| ImageGenError::Checkpoint(format!("{} prefetch: {e}", self.label)))
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            // A failed or panicked background read only means the import
            // faults the pages in itself.
            let _ = handle.join();
        }
    }
}

/// Reads `spans` in [`CHUNK`]-sized requests on [`READERS`] threads,
/// discarding the bytes.
fn read_all(spans: &[FileSpan], stop: &AtomicBool) -> Result<PrefetchStats, String> {
    let began = Instant::now();
    let mut chunks: Vec<(usize, u64, u64)> = Vec::new();
    for (i, span) in spans.iter().enumerate() {
        let mut at = span.start;
        while at < span.end {
            let end = (at + CHUNK).min(span.end);
            chunks.push((i, at, end));
            at = end;
        }
    }
    let next = AtomicUsize::new(0);
    let read = AtomicU64::new(0);
    let failure = std::sync::Mutex::new(None::<String>);
    std::thread::scope(|scope| {
        for _ in 0..READERS {
            scope.spawn(|| {
                let mut buf = vec![0u8; usize::try_from(CHUNK).unwrap_or(usize::MAX)];
                let mut open: Option<(usize, File)> = None;
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(span, from, to)) = chunks.get(index) else {
                        return;
                    };
                    let path = &spans[span].path;
                    if open.as_ref().is_none_or(|(s, _)| spans[*s].path != *path) {
                        match File::open(path) {
                            Ok(file) => open = Some((span, file)),
                            Err(e) => {
                                failure
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                                    .get_or_insert(format!("opening {}: {e}", path.display()));
                                stop.store(true, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                    let Some((_, file)) = open.as_ref() else {
                        return;
                    };
                    let len = usize::try_from(to - from).unwrap_or(usize::MAX);
                    if let Err(e) = file.read_exact_at(&mut buf[..len], from) {
                        failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .get_or_insert(format!("reading {} at {from}: {e}", path.display()));
                        // The import that follows would fail on the same
                        // bytes: stop the other readers now.
                        stop.store(true, Ordering::Relaxed);
                        return;
                    }
                    read.fetch_add(to - from, Ordering::Relaxed);
                }
            });
        }
    });
    if let Some(error) = failure
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        return Err(error);
    }
    Ok(PrefetchStats {
        bytes: read.into_inner(),
        seconds: began.elapsed().as_secs_f64(),
    })
}

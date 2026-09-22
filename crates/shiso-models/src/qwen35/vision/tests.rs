//! Tests for the tower contract, over the Qwen family's implementation of it.
//!
//! Nothing here launches: the questions are what the wrapper reports about
//! itself - the placement its family's template writes, the row width it
//! projects to, the spec it wants pictures preprocessed with - and all three
//! are answered from the container's own metadata, on a device that is a table
//! of byte vectors. The encode itself is covered by the encoder's device cases
//! next door and by the server's media goldens, which is where a change to the
//! launch sequence would show up.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use shiso_core::{Backend, CoreError, CoreResult, DeviceView, KernelRef, LaunchArgs, RawBuffer};

use super::Qwen35Tower;
use crate::qwen35::vision::dims::VisionDims;
use crate::qwen35::vision::testing::synthetic_tower;
use shiso_vision::{MediaPlacement, VisionTower};
use shiso_vision::{PatchGeometry, ResizeRule};

/// A device that is a table of byte vectors: enough to accept an upload, and
/// nothing else, because nothing here launches.
#[derive(Debug, Default)]
struct FakeBackend {
    allocations: Mutex<HashMap<u64, Vec<u8>>>,
    next_id: AtomicU64,
}

impl FakeBackend {
    fn fail(op: &'static str, detail: &str) -> CoreError {
        CoreError::Backend {
            backend: "fake",
            op,
            source: Box::new(std::io::Error::other(detail.to_owned())),
        }
    }

    fn range_of(view: DeviceView<'_>, len: u64) -> CoreResult<Range<usize>> {
        let start =
            usize::try_from(view.offset()).map_err(|_| Self::fail("range", "offset too large"))?;
        let width = usize::try_from(len).map_err(|_| Self::fail("range", "length too large"))?;
        Ok(start..start.saturating_add(width))
    }
}

impl Backend for FakeBackend {
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let width =
            usize::try_from(bytes).map_err(|_| Self::fail("alloc_raw", "the size does not fit"))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let id = NonZeroU64::new(id).ok_or_else(|| Self::fail("alloc_raw", "id space empty"))?;
        self.allocations.lock().insert(id.get(), vec![0u8; width]);
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        self.allocations.lock().remove(&raw.id().get());
        Ok(())
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        let len = u64::try_from(src.len()).unwrap_or(u64::MAX);
        let range = Self::range_of(dst, len)?;
        let mut table = self.allocations.lock();
        let buffer = table
            .get_mut(&dst.id().get())
            .ok_or_else(|| Self::fail("copy_h2d", "no live allocation"))?;
        if range.end > buffer.len() {
            return Err(CoreError::OutOfBounds {
                offset: dst.offset(),
                len,
                buffer_bytes: u64::try_from(buffer.len()).unwrap_or(u64::MAX),
            });
        }
        buffer.splice(range, src.iter().copied());
        Ok(())
    }

    fn copy_d2h(&self, _src: DeviceView<'_>, _dst: &mut [u8]) -> CoreResult<()> {
        Err(Self::fail("copy_d2h", "nothing here reads back"))
    }

    fn copy_d2d(&self, _dst: DeviceView<'_>, _src: DeviceView<'_>) -> CoreResult<()> {
        Err(Self::fail("copy_d2d", "nothing here gathers"))
    }

    fn run(&self, _kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
        Err(Self::fail("run", "nothing here launches"))
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}

/// A small tower whose every dimension differs from the shipped one, so a
/// number read off a literal rather than off the container fails here.
fn tiny() -> VisionDims {
    let geometry = PatchGeometry::new(2, 2, 2).expect("a small geometry");
    VisionDims::new(2, 8, 2, 16, 12, 16, geometry).expect("the small tower's dimensions")
}

/// Opens a synthetic standalone tower through the family's own opener.
fn open(dims: VisionDims) -> (tempfile::NamedTempFile, Qwen35Tower) {
    let (file, _values) = synthetic_tower(dims, 11);
    let backend: Arc<dyn Backend> = Arc::new(FakeBackend::default());
    let tower = Qwen35Tower::open(&backend, file.path()).expect("the synthetic tower loads");
    (file, tower)
}

/// The placement is the family's, not the server's: one marker per item, a
/// clip marker beside it, and no bracket.
#[test]
fn the_qwen_tower_reports_its_families_markers() {
    let (_file, tower) = open(tiny());
    assert_eq!(tower.placement(), MediaPlacement::QWEN35);
    assert_eq!(tower.placement().image_marker, "<|image_pad|>");
    assert_eq!(tower.placement().video_marker, Some("<|video_pad|>"));
    assert_eq!(
        tower.placement().wrap_image,
        None,
        "this family writes no bracket; a wrap here would put two unseen tokens in every prompt"
    );
}

/// The projected width is the container's, because it is the number the boot
/// path compares against the language model's residual width - and a literal
/// that happened to be right for one checkpoint would pass every other one.
#[test]
fn the_projected_width_comes_from_the_container() {
    let dims = tiny();
    let (_file, tower) = open(dims);
    assert_eq!(
        u64::from(tower.out_hidden()),
        u64::try_from(dims.out_hidden()).expect("a small width fits"),
    );
    assert_ne!(
        tower.out_hidden(),
        2560,
        "the shipped tower's width leaked in from somewhere other than the file"
    );
}

/// The spec the tower hands the preprocessor is derived from its own geometry,
/// and it is the grid-unit rule with the half statistics: the committed vision
/// goldens were recorded under exactly that, and a spec differing by one number
/// would re-record them rather than fail.
/// The halves are written by the spec constructor, not computed, so an
/// epsilon would accept a spec mutated by rounding and call it the same.
#[expect(
    clippy::float_cmp,
    reason = "0.5 is exactly representable and the spec stores the literal, so equality is the assertion"
)]
#[test]
fn the_preprocess_spec_is_the_towers_own_geometry_under_the_grid_unit_rule() {
    let dims = tiny();
    let (_file, tower) = open(dims);
    let spec = tower.preprocess();
    assert_eq!(spec.geometry, dims.geometry());
    assert_eq!(spec.resize, ResizeRule::GridUnit);
    assert_eq!(spec.mean, [0.5, 0.5, 0.5]);
    assert_eq!(spec.std, [0.5, 0.5, 0.5]);
}

/// Gemma's placement brackets its run. Asserted here beside Qwen's because the
/// pair is the reason the type exists: two families, one renderer.
#[test]
fn the_gemma_placement_brackets_its_run_and_has_no_clip_marker() {
    let placement = MediaPlacement::GEMMA4;
    assert_eq!(placement.image_marker, "<|image|>");
    assert_eq!(placement.video_marker, None);
    assert_eq!(placement.wrap_image, Some(("<|image>", "<image|>")));
}

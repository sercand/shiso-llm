//! Tests for moving a vision tower onto a device.
//!
//! Two fixtures, for two different questions. The local checkpoint answers
//! "does this read the tower that actually ships" and skips on a machine
//! without it. A synthetic standalone container answers everything else -
//! contract enforcement, prefix probing, the refusals - because there is no
//! real one of those to read, and a test that waited for one would be a test
//! that never ran.
//!
//! The synthetic fixture's tensor names come from the manifest the loader
//! itself derives, so it deliberately does NOT gate name derivation. That is
//! the recorded checkpoint manifest's job, next door; this one gates the
//! plumbing, the checks and the device handoff.

use std::collections::HashMap;
use std::io::Write;
use std::num::NonZeroU64;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use shiso_core::{
    Backend, CoreError, CoreResult, DeviceView, KernelRef, LaunchArgs, RawBuffer, VisionConfig,
};
use tempfile::NamedTempFile;

use super::load_vision_weights;
use crate::qwen35::vision::dims::VisionDims;
use crate::qwen35::vision::manifest::{
    AxisOrder, PUBLISHED_SIDECAR, SIDECAR_PREFIX, SIDECAR_PREFIX_CANONICAL, TowerManifest,
};
use shiso_vision::VisionError;
use shiso_vision::{PatchGeometry, VisionSource};

// -- a device that is a table of byte vectors -------------------------------

/// The same shape of fake the other crates use: every question a test has -
/// "was this freed", "how much was copied", "is this still live" - is an
/// atomic or a table lookup rather than a mock framework.
#[derive(Debug, Default)]
struct FakeBackend {
    allocations: Mutex<HashMap<u64, Vec<u8>>>,
    next_id: AtomicU64,
    allocs: AtomicU64,
    frees: AtomicU64,
    h2d_bytes: AtomicU64,
}

impl FakeBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn live(&self) -> usize {
        self.allocations.lock().len()
    }

    fn allocs(&self) -> u64 {
        self.allocs.load(Ordering::Relaxed)
    }

    fn frees(&self) -> u64 {
        self.frees.load(Ordering::Relaxed)
    }

    fn uploaded(&self) -> u64 {
        self.h2d_bytes.load(Ordering::Relaxed)
    }

    fn bytes_of(&self, view: DeviceView<'_>) -> Vec<u8> {
        self.allocations
            .lock()
            .get(&view.id().get())
            .cloned()
            .unwrap_or_default()
    }

    fn fail(op: &'static str, detail: String) -> CoreError {
        CoreError::Backend {
            backend: "fake",
            op,
            source: Box::new(std::io::Error::other(detail)),
        }
    }

    fn range_of(view: DeviceView<'_>, len: u64) -> CoreResult<Range<usize>> {
        let start = usize::try_from(view.offset())
            .map_err(|_| Self::fail("range", "offset too large".into()))?;
        let width =
            usize::try_from(len).map_err(|_| Self::fail("range", "length too large".into()))?;
        Ok(start..start.saturating_add(width))
    }
}

impl Backend for FakeBackend {
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let width = usize::try_from(bytes)
            .map_err(|_| Self::fail("alloc_raw", format!("{bytes} bytes do not fit")))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let id =
            NonZeroU64::new(id).ok_or_else(|| Self::fail("alloc_raw", "id space empty".into()))?;
        self.allocations.lock().insert(id.get(), vec![0u8; width]);
        self.allocs.fetch_add(1, Ordering::Relaxed);
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        if self.allocations.lock().remove(&raw.id().get()).is_some() {
            self.frees.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        Err(Self::fail(
            "free_raw",
            format!("{} was already free", raw.id()),
        ))
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        let len = u64::try_from(src.len()).unwrap_or(u64::MAX);
        let range = Self::range_of(dst, len)?;
        let mut table = self.allocations.lock();
        let buffer = table
            .get_mut(&dst.id().get())
            .ok_or_else(|| Self::fail("copy_h2d", "no live allocation".into()))?;
        if range.end > buffer.len() {
            return Err(CoreError::OutOfBounds {
                offset: dst.offset(),
                len,
                buffer_bytes: u64::try_from(buffer.len()).unwrap_or(u64::MAX),
            });
        }
        buffer.splice(range, src.iter().copied());
        self.h2d_bytes.fetch_add(len, Ordering::Relaxed);
        Ok(())
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        let len = u64::try_from(dst.len()).unwrap_or(u64::MAX);
        let range = Self::range_of(src, len)?;
        let table = self.allocations.lock();
        let buffer = table
            .get(&src.id().get())
            .ok_or_else(|| Self::fail("copy_d2h", "no live allocation".into()))?;
        let slice = buffer
            .get(range)
            .ok_or_else(|| Self::fail("copy_d2h", "range past the end".into()))?;
        dst.copy_from_slice(slice);
        Ok(())
    }

    fn copy_d2d(&self, _dst: DeviceView<'_>, _src: DeviceView<'_>) -> CoreResult<()> {
        Err(Self::fail(
            "copy_d2d",
            "the tower loader never copies".into(),
        ))
    }

    fn run(&self, _kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
        Err(Self::fail("run", "the tower loader never launches".into()))
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}

/// The fake as the trait object every loader takes.
fn device(backend: &Arc<FakeBackend>) -> Arc<dyn Backend> {
    let same = Arc::clone(backend);
    let object: Arc<dyn Backend> = same;
    object
}

// -- a synthetic standalone tower -------------------------------------------

/// Header constants of the standalone container format.
const MAGIC: u32 = 0x4655_4747;
/// Container version this fixture writes.
const VERSION: u32 = 3;
/// Payload alignment the container uses when it declares none.
const ALIGN: usize = 32;
/// Metadata type id for an unsigned 32-bit value.
const KV_U32: u32 = 4;
/// Metadata type id for a string.
const KV_STR: u32 = 8;
/// Tensor type id for single precision.
const TENSOR_F32: u32 = 0;

/// A standalone tower, small enough to build in a test and shaped like a real
/// one in every way the loader looks at.
#[derive(Debug, Clone)]
struct TowerFixture {
    architecture: String,
    prefix: &'static str,
    depth: usize,
    hidden: usize,
    heads: usize,
    intermediate: usize,
    out_hidden: usize,
    positions: usize,
    patch: usize,
    temporal: usize,
    merge: usize,
    /// A dimension key to leave out, for the tower that declares no geometry.
    omit: Option<&'static str>,
}

impl TowerFixture {
    /// A tower of an architecture the published contract says nothing about, so
    /// it loads on whatever dimensions it declares.
    fn tiny() -> Self {
        Self {
            architecture: "tiny-vision".to_owned(),
            prefix: SIDECAR_PREFIX,
            depth: 2,
            hidden: 8,
            heads: 2,
            intermediate: 16,
            out_hidden: 12,
            positions: 16,
            patch: 2,
            temporal: 2,
            merge: 2,
            omit: None,
        }
    }

    /// The same tower, claiming to be the published one.
    fn claiming_the_contract(mut self) -> Self {
        self.architecture = PUBLISHED_SIDECAR.architecture.to_owned();
        self
    }

    fn dims(&self) -> VisionDims {
        let geometry = PatchGeometry::new(self.patch, self.temporal, self.merge)
            .expect("the fixture geometry is valid");
        VisionDims::new(
            self.depth,
            self.hidden,
            self.heads,
            self.intermediate,
            self.out_hidden,
            self.positions,
            geometry,
        )
        .expect("the fixture dimensions are valid")
    }

    /// Serializes the tower to a temporary file that deletes itself.
    fn write(&self) -> NamedTempFile {
        let manifest = TowerManifest::build(self.prefix, self.dims());
        let specs: Vec<_> = manifest.iter().cloned().collect();

        let mut out = Vec::new();
        put_u32(&mut out, MAGIC);
        put_u32(&mut out, VERSION);
        put_u64(&mut out, as_u64(specs.len()));

        let kvs = self.metadata();
        put_u64(&mut out, as_u64(kvs.len() + 1));
        put_str(&mut out, "general.architecture");
        put_u32(&mut out, KV_STR);
        put_str(&mut out, &self.architecture);
        for (key, value) in &kvs {
            put_str(&mut out, key);
            put_u32(&mut out, KV_U32);
            put_u32(&mut out, *value);
        }

        let mut offset = 0u64;
        for spec in &specs {
            put_str(&mut out, spec.name());
            // The container writes the fastest-varying axis first, which is the
            // reverse of how the manifest states a shape.
            let dims: Vec<u64> = spec.shape().iter().rev().copied().collect();
            put_u32(&mut out, as_u32(dims.len()));
            for dim in &dims {
                put_u64(&mut out, *dim);
            }
            put_u32(&mut out, TENSOR_F32);
            put_u64(&mut out, offset);
            offset += spec.elements() * 4;
        }
        while out.len() % ALIGN != 0 {
            out.push(0);
        }
        // Payload bytes are a per-tensor pattern rather than zeros, so a test
        // can prove a tensor reached the device intact instead of proving only
        // that something of the right length did.
        for (index, spec) in specs.iter().enumerate() {
            let fill = u8::try_from(index % 251).unwrap_or(0);
            out.extend(std::iter::repeat_n(fill, usize_of(spec.elements() * 4)));
        }

        let mut file = NamedTempFile::new().expect("a temporary file");
        file.write_all(&out).expect("the fixture is written");
        file.flush().expect("the fixture is flushed");
        file
    }

    fn metadata(&self) -> Vec<(String, u32)> {
        let arch = &self.architecture;
        [
            ("block_count", self.depth),
            ("embedding_length", self.hidden),
            ("feed_forward_length", self.intermediate),
            ("attention.head_count", self.heads),
            ("projection_length", self.out_hidden),
            ("patch_size", self.patch),
            ("temporal_patch_size", self.temporal),
            ("spatial_merge_size", self.merge),
            ("position_embedding_count", self.positions),
        ]
        .into_iter()
        .filter(|(suffix, _)| Some(*suffix) != self.omit)
        .map(|(suffix, value)| (format!("{arch}.{suffix}"), as_u32(value)))
        .collect()
    }
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).expect("a fixture count fits")
}

fn as_u32(value: usize) -> u32 {
    u32::try_from(value).expect("a fixture value fits")
}

fn usize_of(value: u64) -> usize {
    usize::try_from(value).expect("a fixture payload fits")
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_u64(out, as_u64(value.len()));
    out.extend_from_slice(value.as_bytes());
}

// -- the local checkpoint ---------------------------------------------------

/// The real tower must load, with the dimensions and the tensor count the
/// recording pinned. Skips on a machine with no checkpoint, which is normal.
#[test]
fn the_local_checkpoint_tower_loads() {
    let Some(dir) = crate::qwen35::vision::testing::checkpoint_dir() else {
        tracing::warn!("skipping: no local checkpoint");
        return;
    };
    let backend = FakeBackend::new();
    let weights = load_vision_weights(&device(&backend), &VisionSource::OptiqDir(dir))
        .expect("the local tower loads");

    let recorded = crate::qwen35::vision::testing::recorded_tower();
    let expected = recorded["n_tensors"].as_u64().expect("a recorded count");
    assert_eq!(u64::try_from(weights.tensor_count()), Ok(expected));
    assert_eq!(weights.prefix(), "vision_tower");

    let dims = weights.dims();
    assert_eq!(dims.depth(), 24);
    assert_eq!(dims.hidden(), 1024);
    assert_eq!(dims.head_dim(), 64);
    assert_eq!(dims.out_hidden(), 2560);
    assert_eq!(dims.position_grid(), 48);
    assert_eq!(weights.tensors().patch_embed.weight.elements(), 1536 * 1024);

    // Every tensor is its own allocation and the whole tower is bf16, so the
    // device byte count is exactly twice the element count.
    let elements: u64 = weights
        .tensors()
        .iter()
        .map(super::TowerTensor::elements)
        .sum();
    assert_eq!(weights.device_bytes(), elements * 2);
    assert_eq!(u64::try_from(backend.live()), Ok(expected));
}

/// The tower owns its allocations: dropping it frees every one, with no
/// teardown call for a caller to forget.
#[test]
fn dropping_the_tower_frees_every_buffer() {
    let Some(dir) = crate::qwen35::vision::testing::checkpoint_dir() else {
        tracing::warn!("skipping: no local checkpoint");
        return;
    };
    let backend = FakeBackend::new();
    let weights = load_vision_weights(&device(&backend), &VisionSource::OptiqDir(dir))
        .expect("the local tower loads");
    let allocated = backend.allocs();
    assert!(allocated > 0);
    drop(weights);
    assert_eq!(backend.frees(), allocated);
    assert_eq!(backend.live(), 0);
}

/// A checkpoint with no vision block is a correct thing to have, and the caller
/// serves text for it rather than reporting a broken model - so it has to be
/// distinguishable from a tower that failed to load.
#[test]
fn a_checkpoint_without_a_tower_says_so() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut config: serde_json::Value =
        serde_json::from_str(crate::qwen35::vision::testing::RECORDED_CONFIG)
            .expect("the recorded config parses");
    config
        .as_object_mut()
        .expect("the config is an object")
        .remove("vision_config");
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).expect("re-serialized"),
    )
    .expect("the config is written");

    let backend = FakeBackend::new();
    let error = load_vision_weights(
        &device(&backend),
        &VisionSource::OptiqDir(dir.path().to_path_buf()),
    )
    .expect_err("a text-only checkpoint has no tower");
    assert!(
        matches!(error, VisionError::NoVisionConfig { .. }),
        "{error}"
    );
    assert_eq!(backend.allocs(), 0, "nothing was uploaded");
}

// -- the standalone container -----------------------------------------------

/// A standalone tower of an architecture the contract says nothing about loads
/// on the dimensions its own metadata declares.
#[test]
fn a_standalone_tower_loads_on_its_declared_dimensions() {
    let fixture = TowerFixture::tiny();
    let file = fixture.write();
    let backend = FakeBackend::new();
    let weights = load_vision_weights(
        &device(&backend),
        &VisionSource::SidecarGguf(file.path().to_path_buf()),
    )
    .expect("the fixture loads");

    assert_eq!(weights.prefix(), SIDECAR_PREFIX);
    assert_eq!(weights.dims(), fixture.dims());
    assert_eq!(weights.tensor_count(), fixture.dims().tensor_count());
    assert_eq!(weights.tensor_count(), 2 * 12 + 9);
    assert_eq!(weights.device_bytes(), backend.uploaded());
}

/// The bytes that reached the device must be the bytes the file held. A loader
/// that allocated the right sizes and uploaded the wrong tensors would pass
/// every count above.
#[test]
fn the_bytes_that_reach_the_device_are_the_tensors_bytes() {
    let fixture = TowerFixture::tiny().write();
    let backend = FakeBackend::new();
    let weights = load_vision_weights(
        &device(&backend),
        &VisionSource::SidecarGguf(fixture.path().to_path_buf()),
    )
    .expect("the fixture loads");

    // The fixture fills tensor `n` with the byte `n`, in upload order.
    for (index, tensor) in weights.tensors().iter().enumerate() {
        let pattern = u8::try_from(index % 251).unwrap_or(0);
        let resident = backend.bytes_of(tensor.view());
        assert_eq!(
            resident.len(),
            usize_of(tensor.elements() * 4),
            "{} is the wrong size on the device",
            tensor.name()
        );
        assert!(
            resident.iter().all(|&byte| byte == pattern),
            "{} did not arrive intact",
            tensor.name()
        );
    }
}

/// The longer spelling of the prefix is probed when the short one is absent,
/// and the tower is reported under the spelling it was actually found at.
#[test]
fn the_canonical_prefix_is_probed_second() {
    let mut fixture = TowerFixture::tiny();
    fixture.prefix = SIDECAR_PREFIX_CANONICAL;
    let file = fixture.write();
    let backend = FakeBackend::new();
    let weights = load_vision_weights(
        &device(&backend),
        &VisionSource::SidecarGguf(file.path().to_path_buf()),
    )
    .expect("the fixture loads under the longer prefix");
    assert_eq!(weights.prefix(), SIDECAR_PREFIX_CANONICAL);
}

/// A file with neither spelling is not a tower, and the error names both
/// spellings so the operator can see which file they actually have.
#[test]
fn a_file_with_no_tower_prefix_names_both_spellings() {
    let mut fixture = TowerFixture::tiny();
    fixture.prefix = "something.else";
    let file = fixture.write();
    let backend = FakeBackend::new();
    let error = load_vision_weights(
        &device(&backend),
        &VisionSource::SidecarGguf(file.path().to_path_buf()),
    )
    .expect_err("this is not a tower");
    match &error {
        VisionError::NoTowerPrefix {
            short, canonical, ..
        } => {
            assert_eq!(*short, SIDECAR_PREFIX);
            assert_eq!(*canonical, SIDECAR_PREFIX_CANONICAL);
        }
        other => panic!("expected a prefix failure, got {other}"),
    }
    assert_eq!(backend.allocs(), 0);
}

/// A file claiming the published architecture is judged against the published
/// contract, and a tower of the wrong depth is a different artefact rather than
/// a corrupt one - so it is refused by name, with both numbers.
#[test]
fn a_tower_claiming_the_contract_must_meet_it() {
    let shallow: Box<dyn Fn(&mut TowerFixture)> =
        Box::new(|f| f.depth = PUBLISHED_SIDECAR.depth - 1);
    let narrow: Box<dyn Fn(&mut TowerFixture)> = Box::new(|f| f.hidden = 1024);
    for (mutate, field, got) in [
        (shallow, "block_count", PUBLISHED_SIDECAR.depth - 1),
        (narrow, "embedding_length", 1024),
    ] {
        let mut fixture = TowerFixture::tiny().claiming_the_contract();
        // Start from the published shape so exactly one field disagrees.
        fixture.depth = PUBLISHED_SIDECAR.depth;
        fixture.hidden = PUBLISHED_SIDECAR.hidden;
        fixture.intermediate = PUBLISHED_SIDECAR.intermediate;
        fixture.out_hidden = PUBLISHED_SIDECAR.out_hidden;
        mutate(&mut fixture);

        let file = fixture.write();
        let backend = FakeBackend::new();
        let error = load_vision_weights(
            &device(&backend),
            &VisionSource::SidecarGguf(file.path().to_path_buf()),
        )
        .expect_err("the contract must bite");
        match &error {
            VisionError::SidecarContract {
                architecture,
                field: reported,
                got: reported_got,
                ..
            } => {
                assert_eq!(*architecture, PUBLISHED_SIDECAR.architecture);
                assert_eq!(*reported, field);
                assert_eq!(*reported_got, got);
            }
            other => panic!("expected a contract failure for {field}, got {other}"),
        }
        assert_eq!(backend.allocs(), 0, "nothing is uploaded before the check");
    }
}

/// A tower missing metadata its dimensions come from has no derivable geometry,
/// and defaulting the missing key would produce a manifest that happens to
/// match some other tower.
#[test]
fn a_tower_missing_its_metadata_is_refused() {
    let mut fixture = TowerFixture::tiny();
    fixture.omit = Some("embedding_length");
    let file = fixture.write();

    let backend = FakeBackend::new();
    let error = load_vision_weights(
        &device(&backend),
        &VisionSource::SidecarGguf(file.path().to_path_buf()),
    )
    .expect_err("a tower with no declared width cannot load");
    match &error {
        VisionError::TowerMetadata { key } => assert_eq!(key, "embedding_length"),
        other => panic!("expected a missing-metadata failure, got {other}"),
    }
    assert_eq!(backend.allocs(), 0, "nothing is uploaded before the check");
}

/// A path that is not a container at all fails as a container error naming the
/// file, not as a missing tensor halfway through a load.
#[test]
fn a_file_that_is_not_a_container_is_refused_by_path() {
    let mut file = NamedTempFile::new().expect("a temporary file");
    file.write_all(b"this is not a model").expect("written");
    file.flush().expect("flushed");
    let backend = FakeBackend::new();
    let error = load_vision_weights(
        &device(&backend),
        &VisionSource::SidecarGguf(file.path().to_path_buf()),
    )
    .expect_err("arbitrary bytes are not a tower");
    assert!(
        matches!(error, VisionError::TowerContainer { .. }),
        "{error}"
    );
}

// -- the checks that stand between a wrong tower and a wrong answer ----------

/// The shape check is what stops a tower of the wrong width from loading
/// cleanly, so it is proved directly: every tensor of a wider tower is
/// individually well-formed, and only the comparison against the declared
/// dimensions catches it.
#[test]
fn a_manifest_built_for_the_wrong_width_rejects_a_good_tower() {
    let fixture = TowerFixture::tiny();
    let wider = VisionDims::new(
        fixture.depth,
        fixture.hidden * 2,
        fixture.heads,
        fixture.intermediate,
        fixture.out_hidden,
        fixture.positions,
        PatchGeometry::new(fixture.patch, fixture.temporal, fixture.merge).expect("geometry"),
    )
    .expect("a valid but different tower");
    let honest = TowerManifest::build(SIDECAR_PREFIX, fixture.dims());
    let wrong = TowerManifest::build(SIDECAR_PREFIX, wider);

    let mut disagreements = 0;
    for (truth, claim) in honest.iter().zip(wrong.iter()) {
        assert_eq!(
            truth.name(),
            claim.name(),
            "the names are width-independent"
        );
        if !claim.matches(truth.shape(), AxisOrder::SlowestFirst) {
            disagreements += 1;
        }
    }
    assert!(
        disagreements > 0,
        "a tower of twice the width must not satisfy the narrow manifest"
    );
}

/// A source names one path whichever variant it is, so a caller reporting the
/// failure does not have to match the enum to find it.
#[test]
fn a_source_reports_the_path_it_was_given() {
    let path = PathBuf::from("/models/tower");
    assert_eq!(VisionSource::OptiqDir(path.clone()).path(), path);
    assert_eq!(VisionSource::SidecarGguf(path.clone()).path(), path);
}

/// The dimensions a tower is loaded with come from the file, never from the
/// contract: two fixtures of different shapes must produce different towers
/// through the same code path.
#[test]
fn two_towers_of_different_shapes_load_through_the_same_path() {
    let narrow = TowerFixture::tiny();
    let mut wide = TowerFixture::tiny();
    wide.hidden = 16;
    wide.intermediate = 32;
    wide.out_hidden = 20;

    let backend = FakeBackend::new();
    let files = (narrow.write(), wide.write());
    let first = load_vision_weights(
        &device(&backend),
        &VisionSource::SidecarGguf(files.0.path().to_path_buf()),
    )
    .expect("the narrow tower loads");
    let second = load_vision_weights(
        &device(&backend),
        &VisionSource::SidecarGguf(files.1.path().to_path_buf()),
    )
    .expect("the wide tower loads");

    assert_eq!(first.dims().hidden(), 8);
    assert_eq!(second.dims().hidden(), 16);
    assert_eq!(first.tensor_count(), second.tensor_count());
    assert!(second.device_bytes() > first.device_bytes());
}

/// A vision config whose geometry cannot drive a preprocessor cannot describe a
/// tower either, and the tower loader must refuse it at the same boundary.
#[test]
fn an_invalid_geometry_stops_the_load_before_the_device() {
    let mut vcfg: VisionConfig = crate::qwen35::vision::testing::vision_config();
    vcfg.patch_size = 0;
    assert!(matches!(
        VisionDims::try_from(&vcfg),
        Err(VisionError::GeometryZero { .. })
    ));
}

/// The channel-slowest patch weight is transposed on the way to the device, and
/// the check that it was is where each *number* landed.
///
/// A shape check cannot see this: both spellings hold the same count, and the
/// encoder is layout-agnostic, so a row read in the wrong order describes a
/// scrambled picture with nothing anywhere to say so. So the fixture writes a
/// distinct value into every element and the assertion reads each one back at
/// the index the preprocessor's own packing would look for it.
#[test]
fn a_channel_slowest_patch_weight_is_transposed_into_the_order_the_packer_builds() {
    let dims =
        VisionDims::try_from(&crate::qwen35::vision::testing::vision_config()).expect("a tower");
    let manifest = TowerManifest::build(crate::qwen35::vision::manifest::CHECKPOINT_PREFIX, dims);
    let spec = &manifest.patch_embed.weight;
    let geometry = dims.geometry();

    let channels = 3usize;
    let patch = geometry.patch_dim() / channels;
    let rows = dims.hidden();

    // Half precision on purpose: a transpose written against a four-byte
    // element would pass a single-precision fixture and shear a real tower.
    let mut stored = Vec::with_capacity(rows * channels * patch * 2);
    for row in 0..rows {
        for c in 0..channels {
            for p in 0..patch {
                let value = u16::try_from((row * channels * patch + c * patch + p) % 65_536)
                    .expect("a fixture value fits");
                stored.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    let moved = super::channel_minor(&stored, spec, shiso_core::DType::BF16)
        .expect("the patch rows transpose");
    assert_eq!(moved.len(), stored.len());
    let at =
        |bytes: &[u8], index: usize| u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
    for row in [0usize, 1, rows - 1] {
        for c in 0..channels {
            for p in [0usize, 1, patch - 1] {
                let expected = u16::try_from((row * channels * patch + c * patch + p) % 65_536)
                    .expect("a fixture value fits");
                let index = row * channels * patch + p * channels + c;
                assert_eq!(
                    at(&moved, index),
                    expected,
                    "row {row}, channel {c}, patch slot {p} did not land channel-innermost"
                );
            }
        }
    }

    // And the already-correct spelling is not touched, which is the other half
    // of the claim: the loader acts on the spelling, not on the tensor's name.
    assert_eq!(
        spec.spelling(
            &[
                u64::try_from(rows).expect("fits"),
                u64::try_from(geometry.temporal_patch_size()).expect("fits"),
                u64::try_from(geometry.patch_size()).expect("fits"),
                u64::try_from(geometry.patch_size()).expect("fits"),
                3
            ],
            AxisOrder::SlowestFirst
        ),
        Some(crate::qwen35::vision::manifest::PatchRows::AsStored)
    );
}

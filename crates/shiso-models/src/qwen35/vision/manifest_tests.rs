//! Tests for the derived tensor manifest.
//!
//! The manifest's whole job is to state what a tower must contain before the
//! tower is asked. So it is checked against a RECORDING of a real tower rather
//! than against the formula that produced it: a test that rebuilt the expected
//! names with the same code would agree with any bug that was in that code.

use std::collections::BTreeMap;

use super::{
    AxisOrder, CHECKPOINT_PREFIX, PUBLISHED_SIDECAR, PatchRows, SIDECAR_PREFIX, TowerManifest,
};
use crate::qwen35::vision::dims::VisionDims;
use crate::qwen35::vision::testing::{recorded_tower, vision_config};

/// The recorded tower's tensor names and shapes, slowest axis first.
fn recorded_tensors() -> BTreeMap<String, Vec<u64>> {
    recorded_tower()["tensors"]
        .as_object()
        .expect("the recording carries a tensor table")
        .iter()
        .map(|(name, shape)| {
            let dims = shape
                .as_array()
                .expect("every entry has a shape")
                .iter()
                .map(|dim| dim.as_u64().expect("every axis is a count"))
                .collect();
            (name.clone(), dims)
        })
        .collect()
}

/// The manifest for the recorded checkpoint's tower.
fn recorded_manifest() -> TowerManifest {
    let dims = VisionDims::try_from(&vision_config()).expect("the recorded tower is valid");
    TowerManifest::build(CHECKPOINT_PREFIX, dims)
}

/// The names the loader will ask for must be exactly the names the real tower
/// holds. One extra is a load failure on a good file; one missing is a tower
/// uploaded with a hole in it.
#[test]
fn the_manifest_names_exactly_the_recorded_tower() {
    let manifest = recorded_manifest();
    let asked: BTreeMap<&str, &[u64]> = manifest
        .iter()
        .map(|spec| (spec.name(), spec.shape()))
        .collect();
    let recorded = recorded_tensors();

    let asked_names: Vec<&&str> = asked.keys().collect();
    let recorded_names: Vec<&str> = recorded.keys().map(String::as_str).collect();
    assert_eq!(
        asked_names.len(),
        recorded_names.len(),
        "the manifest asks for {} tensors, the tower holds {}",
        asked_names.len(),
        recorded_names.len()
    );
    for name in &recorded_names {
        assert!(
            asked.contains_key(name),
            "the tower holds {name} but the manifest never asks for it"
        );
    }
}

/// Every shape as well as every name. The names alone would pass for a tower of
/// a different width, which is exactly the failure the manifest exists to stop.
#[test]
fn the_manifest_shapes_match_the_recorded_tower() {
    let manifest = recorded_manifest();
    let recorded = recorded_tensors();
    for spec in manifest.iter() {
        let declared = recorded
            .get(spec.name())
            .unwrap_or_else(|| panic!("the recording has no {}", spec.name()));
        assert!(
            spec.matches(declared, AxisOrder::SlowestFirst),
            "{} is recorded as {declared:?} but the manifest requires one of {:?}",
            spec.name(),
            spec.accepted()
        );
    }
}

/// The recording is only a gate while it still describes the file on disk. When
/// the checkpoint is present, the two are compared directly; when it is not,
/// this skips, because a machine without the model is a normal one.
#[test]
fn the_recording_still_describes_the_local_checkpoint() {
    let Some(dir) = crate::qwen35::vision::testing::checkpoint_dir() else {
        tracing::warn!("skipping: no local checkpoint");
        return;
    };
    let recorded = recorded_tower();
    let shard = dir.join(
        recorded["shard"]
            .as_str()
            .expect("the recording names its shard"),
    );
    let Ok(bytes) = std::fs::read(&shard) else {
        tracing::warn!("skipping: the recorded shard is not readable");
        return;
    };
    let length: [u8; 8] = bytes[..8]
        .try_into()
        .expect("the shard has a length prefix");
    let end = 8 + usize::try_from(u64::from_le_bytes(length)).expect("a sane header length");
    let table: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..end]).expect("the header is valid JSON");

    let on_disk: BTreeMap<String, Vec<u64>> = table
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, value)| {
            let shape = value["shape"]
                .as_array()
                .expect("every entry has a shape")
                .iter()
                .map(|dim| dim.as_u64().expect("every axis is a count"))
                .collect();
            (name, shape)
        })
        .collect();
    assert_eq!(
        on_disk,
        recorded_tensors(),
        "the checked-in recording no longer matches the checkpoint on disk"
    );
}

/// The count the dimensions derive and the count the tower actually holds are
/// two independent statements of the same number, which is why both are here.
#[test]
fn the_derived_count_matches_the_recording() {
    let dims = VisionDims::try_from(&vision_config()).expect("valid");
    let recorded = recorded_tower()["n_tensors"]
        .as_u64()
        .expect("the recording counts its tensors");
    assert_eq!(u64::try_from(dims.tensor_count()), Ok(recorded));
    assert_eq!(u64::try_from(recorded_manifest().len()), Ok(recorded));
}

// -- axis order -------------------------------------------------------------

/// The two containers write axes in opposite orders, and for a square matrix
/// the difference is invisible. A non-square tensor is what proves the order is
/// actually applied rather than quietly ignored.
#[test]
fn a_non_square_tensor_distinguishes_the_two_axis_orders() {
    let manifest = recorded_manifest();
    let spec = manifest
        .iter()
        .find(|spec| spec.name().ends_with("blocks.0.attn.qkv.weight"))
        .expect("the tower has a fused projection");
    assert_eq!(spec.shape(), [3072, 1024]);
    assert!(spec.matches(&[3072, 1024], AxisOrder::SlowestFirst));
    assert!(spec.matches(&[1024, 3072], AxisOrder::FastestFirst));
    assert!(!spec.matches(&[1024, 3072], AxisOrder::SlowestFirst));
    assert!(!spec.matches(&[3072, 1024], AxisOrder::FastestFirst));
}

/// A trailing unit axis is punctuation from the exporting framework, not a
/// dimension, so it must not fail a tower.
#[test]
fn a_trailing_unit_axis_is_ignored() {
    let manifest = recorded_manifest();
    let spec = manifest
        .iter()
        .find(|spec| spec.name().ends_with("merger.norm.weight"))
        .expect("the merger has a normalization");
    assert!(spec.matches(&[1024], AxisOrder::SlowestFirst));
    assert!(spec.matches(&[1024, 1, 1], AxisOrder::SlowestFirst));
    assert!(!spec.matches(&[512], AxisOrder::SlowestFirst));
}

/// The patch projection alone has two legitimate spellings, and both must be
/// accepted because both are the same bytes in the same order.
#[test]
fn the_patch_projection_accepts_both_of_its_spellings() {
    let manifest = recorded_manifest();
    let spec = &manifest.patch_embed.weight;
    assert!(spec.matches(&[1024, 2, 16, 16, 3], AxisOrder::SlowestFirst));
    assert!(spec.matches(&[1024, 1536], AxisOrder::SlowestFirst));
    assert!(spec.matches(&[1536, 1024], AxisOrder::FastestFirst));
    // A row width that is not the patch row width is still refused.
    assert!(!spec.matches(&[1024, 768], AxisOrder::SlowestFirst));

    // The two spellings are not interchangeable bytes, and the spec says which
    // is which rather than only that both are legal. The channel-slowest one is
    // the framework-native convolution weight, which is what the standalone
    // container carries.
    assert_eq!(
        spec.spelling(&[1024, 2, 16, 16, 3], AxisOrder::SlowestFirst),
        Some(PatchRows::AsStored)
    );
    assert_eq!(
        spec.spelling(&[1024, 3, 2, 16, 16], AxisOrder::SlowestFirst),
        Some(PatchRows::ChannelSlowest)
    );
    // And as the container writes it, fastest axis first.
    assert_eq!(
        spec.spelling(&[16, 16, 2, 3, 1024], AxisOrder::FastestFirst),
        Some(PatchRows::ChannelSlowest)
    );
    assert_eq!(spec.spelling(&[1024, 768], AxisOrder::SlowestFirst), None);
}

/// Nothing else gets a second spelling; the concession is one tensor wide.
#[test]
fn every_other_tensor_has_exactly_one_accepted_shape() {
    let manifest = recorded_manifest();
    let patch = manifest.patch_embed.weight.name().to_owned();
    for spec in manifest.iter() {
        // Three for the patch projection: the two published axis orders, and
        // the already-flattened row. One for everything else.
        let expected = if spec.name() == patch { 3 } else { 1 };
        assert_eq!(
            spec.accepted().len(),
            expected,
            "{} accepts {} shapes",
            spec.name(),
            spec.accepted().len()
        );
    }
}

// -- the second tower -------------------------------------------------------

/// A different tower must produce a different manifest through the same code.
/// A hard-coded width anywhere above would make these two identical.
#[test]
fn a_wider_tower_produces_its_own_manifest() {
    let geometry = shiso_vision::PatchGeometry::new(16, 2, 2).expect("valid geometry");
    let dims = VisionDims::new(27, 1152, 16, 4304, 2560, 2304, geometry).expect("valid");
    let manifest = TowerManifest::build(SIDECAR_PREFIX, dims);
    assert_eq!(manifest.len(), 333);
    assert_eq!(manifest.blocks.len(), 27);
    assert_eq!(manifest.blocks[0].qkv.weight.shape(), [3456, 1152]);
    assert_eq!(manifest.merger.fc1.weight.shape(), [4608, 4608]);
    assert_eq!(manifest.merger.fc2.weight.shape(), [2560, 4608]);
    assert!(
        manifest
            .iter()
            .all(|spec| spec.name().starts_with(SIDECAR_PREFIX)),
        "every name carries the prefix it was built with"
    );
}

/// The published contract counts its own tensors the same way the dimensions
/// do, so the two cannot drift into disagreeing about what a complete tower is.
#[test]
fn the_published_contract_agrees_with_its_own_dimensions() {
    let geometry = shiso_vision::PatchGeometry::new(16, 2, 2).expect("valid geometry");
    let dims = VisionDims::new(
        PUBLISHED_SIDECAR.depth,
        PUBLISHED_SIDECAR.hidden,
        16,
        PUBLISHED_SIDECAR.intermediate,
        PUBLISHED_SIDECAR.out_hidden,
        2304,
        geometry,
    )
    .expect("the published tower is a valid shape");
    assert_eq!(dims.tensor_count(), PUBLISHED_SIDECAR.tensors);
}

/// The tree's iteration order and its map order must be the same order, or a
/// loader that walks one and assembles the other silently transposes weights.
#[test]
fn the_tree_iterates_in_the_order_it_maps() {
    let manifest = recorded_manifest();
    let walked: Vec<String> = manifest.iter().map(|spec| spec.name().to_owned()).collect();
    let mapped: Vec<String> = manifest
        .clone()
        .try_map(|spec| Ok::<String, ()>(spec.name().to_owned()))
        .expect("an infallible map")
        .iter()
        .cloned()
        .collect();
    assert_eq!(walked, mapped);
    assert_eq!(walked.len(), manifest.len());
}

/// The packer's row order and the projection's axis order are one decision, and
/// this is where the two are held against each other.
///
/// The patch projection is stored `[out, t, y, x, c]`: channel is the FASTEST
/// axis. A patch row therefore has to arrive with its channels innermost, which
/// is what `pack_group` emits. The two orders hold exactly the same numbers in a
/// different arrangement, so nothing else in the system can tell them apart -
/// every shape agrees, every kernel launches, and the tower describes a
/// scrambled picture.
///
/// An earlier version of the packer emitted channel-slowest, which is what a
/// framework-native export of the same layer would want. This test is what makes
/// a checkpoint that ships that way a failure here rather than a quiet wrong
/// answer three crates downstream.
#[test]
fn the_patch_projection_puts_channels_innermost_which_is_what_the_packer_emits() {
    let dims = VisionDims::try_from(&vision_config()).expect("the recorded config is a tower");
    let manifest = TowerManifest::build(CHECKPOINT_PREFIX, dims);
    let spec = &manifest.patch_embed.weight;
    let geometry = dims.geometry();

    let spelled = &spec
        .accepted()
        .iter()
        .find(|shape| shape.rows == PatchRows::AsStored && shape.dims.len() == 5)
        .expect("the patch projection is accepted with its patch axes spelled out")
        .dims;

    // `[out, t, y, x, c]` - and the last axis being the channel count is the
    // whole claim. Reading it off the spec rather than writing `3` here is what
    // makes this track the manifest instead of restating a constant.
    let channels = u64::try_from(
        geometry.patch_dim()
            / (geometry.temporal_patch_size() * geometry.patch_size() * geometry.patch_size()),
    )
    .expect("a channel count fits");
    assert_eq!(
        spelled.last().copied(),
        Some(channels),
        "the channel axis is innermost: {spelled:?}"
    );
    assert_eq!(
        spelled[1],
        u64::try_from(geometry.temporal_patch_size()).expect("fits"),
        "time is the outermost patch axis: {spelled:?}"
    );

    // And the flattened spelling is the same row width, which is the only reason
    // accepting two shapes for this one tensor is safe.
    let flat = &spec
        .accepted()
        .iter()
        .find(|shape| shape.dims.len() == 2)
        .expect("the patch projection is also accepted already flattened")
        .dims;
    assert_eq!(
        flat[1],
        u64::try_from(geometry.patch_dim()).expect("fits"),
        "the flattened spelling is one patch row wide"
    );
}

//! The section reader, and one real layer file on a real device.
//!
//! The spans are tested against byte vectors because a span is arithmetic and
//! wants no device. The handoff is tested against Metal rather than a fake for
//! the opposite reason: what [`load_layer`] is for is putting a file's weights
//! somewhere a kernel can read them, and a fake backend that records a copy
//! would prove the arithmetic twice and the handoff never. Both device cases
//! say why they skipped.

use std::sync::Arc;

use shiso_core::Backend;

use super::super::layout::{FileLayout, Layout, Section, SectionDtype};
use super::{SectionError, load_layer, slice_section};

/// The variable that points at an unpacked Splash snapshot, as
/// `package_tests.rs` spells it.
const ENV: &str = "SHISO_TEST_MODEL_QWEN35MOE_SPLASH";

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// A section with every quantization field absent, which is what a norm is.
fn plain(name: &str, offset: u64, bytes: u64) -> Section {
    Section {
        name: name.to_string(),
        dtype: SectionDtype::Bf16,
        shape: vec![2],
        offset,
        bytes,
        group_size: None,
        storage_n: None,
        actual_output: None,
        expert_stride_bytes: None,
        layout: None,
    }
}

#[test]
fn a_section_lands_at_its_declared_offset_and_length() {
    let mut file = vec![0u8; 4096];
    file[1024..1028].copy_from_slice(&42u32.to_le_bytes());
    let section = plain("input_norm", 1024, 4);
    let got = slice_section(&file, &section).expect("in bounds");
    assert_eq!(got, &42u32.to_le_bytes());
}

#[test]
fn a_section_past_the_end_of_its_file_is_refused_by_name() {
    let file = vec![0u8; 16];
    let section = Section {
        name: "gdn_input".to_string(),
        dtype: SectionDtype::Q4Affine,
        shape: vec![4, 4],
        offset: 8,
        bytes: 64,
        group_size: Some(64),
        storage_n: Some(256),
        actual_output: Some(4),
        expert_stride_bytes: None,
        layout: None,
    };
    let err = slice_section(&file, &section).expect_err("past the end");
    let text = format!("{err}");
    assert!(text.contains("gdn_input"), "{text}");
    // Both numbers, because "it did not fit" without them is a bug report
    // nobody can act on: 64 declared, 8 left after the offset.
    assert!(text.contains("64"), "{text}");
    assert!(text.contains("8 remain"), "{text}");
}

#[test]
fn an_offset_too_large_for_the_host_is_a_refusal_and_not_a_wrapped_span() {
    let file = vec![0u8; 16];
    let section = plain("head", u64::MAX, 4);
    let err = slice_section(&file, &section).expect_err("no file is that long");
    assert!(format!("{err}").contains("head"));
}

#[test]
fn an_empty_section_at_the_very_end_is_in_bounds() {
    let file = vec![7u8; 16];
    assert!(
        slice_section(&file, &plain("tail", 16, 0))
            .expect("empty and in bounds")
            .is_empty()
    );
}

/// The real package, or `None` when it is not on this machine.
fn package() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var(ENV).ok()?);
    path.join("layout.json").is_file().then_some(path)
}

/// A Metal device, or `None` when there is none to be had.
fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        return None;
    }
    let metal: Arc<dyn Backend> = shiso_metal::MetalBackend::new().ok()?;
    Some(metal)
}

#[test]
fn a_file_shorter_than_the_layout_declares_is_refused_before_any_section() {
    let Some(backend) = device() else {
        skip("no Metal device");
        return;
    };
    let directory = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(directory.path().join("layer.bin"), [0u8; 32]).expect("the fixture");
    let file = FileLayout {
        path: "layer.bin".to_string(),
        bytes: 4096,
        sections: vec![plain("input_norm", 0, 4)],
    };
    let err = load_layer(&backend, directory.path(), &file).expect_err("32 bytes, not 4096");
    match err {
        SectionError::Size { want, got, .. } => assert_eq!((want, got), (4096, 32)),
        other => panic!("expected Size, got {other}"),
    }
}

#[test]
fn a_file_that_is_not_there_is_refused_by_path() {
    let Some(backend) = device() else {
        skip("no Metal device");
        return;
    };
    let directory = tempfile::tempdir().expect("a temporary directory");
    let file = FileLayout {
        path: "target/layer-0.bin".to_string(),
        bytes: 16,
        sections: Vec::new(),
    };
    let err = load_layer(&backend, directory.path(), &file).expect_err("nothing is there");
    assert!(format!("{err}").contains("target/layer-0.bin"), "{err}");
}

#[test]
fn every_section_of_a_small_file_lands_on_the_device_holding_its_own_bytes() {
    let Some(backend) = device() else {
        skip("no Metal device");
        return;
    };
    // Two sections on the package's own 16,384-byte alignment, which is also
    // the page size this wrap wants, plus a third that is not aligned - a
    // Splash package never writes one, and the reader must not depend on it.
    let mut bytes = vec![0u8; 3 * 16_384];
    bytes[16_384..16_388].copy_from_slice(&1u32.to_le_bytes());
    bytes[32_768..32_772].copy_from_slice(&2u32.to_le_bytes());
    bytes[40_000..40_004].copy_from_slice(&3u32.to_le_bytes());
    let directory = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(directory.path().join("layer.bin"), &bytes).expect("the fixture");
    let file = FileLayout {
        path: "layer.bin".to_string(),
        bytes: 3 * 16_384,
        sections: vec![
            plain("input_norm", 16_384, 4),
            plain("post_norm", 32_768, 4),
            plain("unaligned", 40_000, 4),
        ],
    };

    let loaded = load_layer(&backend, directory.path(), &file).expect("every section lands");
    assert_eq!(loaded.path, "layer.bin");
    assert_eq!(loaded.sections.len(), 3);
    assert_eq!(loaded.bytes(), 12);
    assert!(loaded.section("nothing_of_the_sort").is_none());

    for (name, want) in [("input_norm", 1u32), ("post_norm", 2), ("unaligned", 3)] {
        let held = loaded.section(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(held.buffer.len(), 4, "{name}");
        let mut read = [0u8; 4];
        backend
            .copy_d2h(held.buffer.view(), &mut read)
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(
            u32::from_le_bytes(read),
            want,
            "{name} is holding another section's bytes"
        );
    }
}

/// One real layer file, landed. Landing all 20.9 GB is a later task's problem;
/// what this pins is that the shipped `layout.json` and the shipped bytes agree
/// well enough for a whole layer - 16 sections, 475 MB, every expert stack - to
/// reach the device with nothing out of bounds and every buffer holding the
/// bytes at its own offset.
///
/// The length assertion alone would not say that last part: a buffer's length
/// is taken from `Section::bytes`, so a systematically shifted offset keeps
/// every length correct and lands every weight one section late. Only reading
/// bytes back and comparing them against the file at that offset can tell the
/// two apart, so every section is read back rather than the first.
#[test]
fn the_shipped_layer_zero_lands_section_by_section() {
    let Some(root) = package() else {
        skip(&format!("{ENV} is unset or names no layout.json"));
        return;
    };
    let Some(backend) = device() else {
        skip("no Metal device");
        return;
    };
    let layout =
        Layout::from_json(&std::fs::read_to_string(root.join("layout.json")).expect("layout.json"))
            .expect("the shipped layout parses");
    let file = &layout.target[0];
    assert_eq!(file.path, "target/layer-0.bin");

    let loaded = load_layer(&backend, &root, file).expect("the shipped layer-0 lands");
    assert_eq!(loaded.sections.len(), file.sections.len());
    let bytes = std::fs::read(root.join(&file.path)).expect("the layer file");
    for (held, declared) in loaded.sections.iter().zip(&file.sections) {
        assert_eq!(held.section.name, declared.name);
        assert_eq!(held.buffer.len(), declared.bytes, "{}", declared.name);
        // Eight bytes off each end of the span, because a shift small enough to
        // leave the head of a section looking right still moves its tail.
        for offset in [0, declared.bytes - 8] {
            let mut read = [0u8; 8];
            backend
                .copy_d2h(
                    held.buffer.slice(offset, 8).expect("eight bytes"),
                    &mut read,
                )
                .unwrap_or_else(|err| panic!("{}: {err}", declared.name));
            let start = usize::try_from(declared.offset + offset).expect("an offset that fits");
            assert_eq!(
                read,
                bytes[start..start + 8],
                "{} is holding the bytes at another offset",
                declared.name
            );
        }
    }
    drop(bytes);

    // The two sections the padding rules are about, checked through what the
    // loader carried rather than through the document a second time.
    let gdn = loaded.section("gdn_input").expect("gdn_input");
    assert_eq!(gdn.section.out_dim(), 12_352);
    assert_ne!(
        u64::from(gdn.section.shape[0]),
        u64::from(gdn.section.out_dim()),
        "the padded width is not the live one"
    );
    let experts = loaded.section("experts.gate_proj").expect("gate_proj");
    assert_eq!(experts.section.experts(), Some(256));
    assert_eq!(experts.section.expert_stride_bytes, Some(589_824));
    assert_eq!(
        experts.buffer.len(),
        256 * 589_824,
        "the stack is its stride times its experts"
    );
}

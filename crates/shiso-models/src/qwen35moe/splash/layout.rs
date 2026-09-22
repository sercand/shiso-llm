//! `layout.json`: what each `.bin` holds and where.
//!
//! A Splash package is self-describing. Every section names a dtype, a shape,
//! a byte offset and - for a quantized one - the group size and the padded row
//! width. **`actual_output` is the live width inside a row padded to
//! `storage_n`**, and a loader that reads `shape[0]` instead reads padding as
//! weights. In `Qwen3.6-35B-A3B-Splash` that is not a corner case: `gdn_input`
//! is declared `[12544, 2048]` with `actual_output` 12352, so reading the shape
//! hands the recurrent block 192 rows of zeros dressed as projection weights.
//!
//! # What the shipped package does that a hand-written fixture would not
//!
//! Three things, all found by parsing the real 250 KB document rather than a
//! sketch of it, and all of them load-bearing here:
//!
//! - `vision` is a **single object**, not an array: the tower is one file, and
//!   the packer wrote it as one file rather than a list of one. `target` and
//!   `draft` are arrays. [`Layout::vision`] is still a `Vec` - one shape for
//!   the three weight sets is worth more than a faithful echo of the JSON - so
//!   the field accepts either spelling and yields a vector of length one.
//! - `embedding.weights` is `q4_affine_weights`, a dtype of its own that
//!   exists because the table's scales and biases are separate sections
//!   (`embedding.scales`, `embedding.biases`) rather than interleaved with the
//!   codes. It also carries [`Section::layout`], which no other section does.
//! - The 120 expert stacks carry [`Section::expert_stride_bytes`]: the distance
//!   from one expert's slab to the next, which a three-dimensional shape alone
//!   does not give once each slab is padded.
//!
//! An unknown dtype is refused rather than skipped. A section this build has no
//! reader for is a model that would answer from bytes nobody decoded, so the
//! failure belongs at the document, by name.

use serde::Deserialize;
use serde::de::{MapAccess, SeqAccess, Visitor};

/// The element formats a Splash section is stored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum SectionDtype {
    /// Two-byte brain float.
    #[serde(rename = "bf16")]
    Bf16,
    /// Four-byte float.
    #[serde(rename = "f32")]
    F32,
    /// Four-bit affine codes with a scale and a bias per group, the scales and
    /// biases carried alongside the codes in the same section.
    #[serde(rename = "q4_affine")]
    Q4Affine,
    /// Four-bit affine codes alone, whose scales and biases are their own
    /// sections. Only the embedding table is stored this way, because it is
    /// read a row at a time at decode and a row's scales are read with it.
    #[serde(rename = "q4_affine_weights")]
    Q4AffineWeights,
    /// Eight-bit affine codes with a scale and a bias per group.
    #[serde(rename = "q8_affine")]
    Q8Affine,
}

/// One named span of one `.bin` file.
#[derive(Debug, Clone, Deserialize)]
pub struct Section {
    /// The role this span plays, as the package spells it.
    pub name: String,
    /// The element format.
    pub dtype: SectionDtype,
    /// Logical dimensions, outermost first.
    pub shape: Vec<u32>,
    /// Byte offset into the file.
    pub offset: u64,
    /// Bytes this span occupies.
    pub bytes: u64,
    /// Codes per quantization group, for a quantized span.
    #[serde(default)]
    pub group_size: Option<u32>,
    /// The width a row is padded to.
    #[serde(default)]
    pub storage_n: Option<u32>,
    /// The live width inside that padding.
    #[serde(default)]
    pub actual_output: Option<u32>,
    /// Bytes from one expert's slab to the next, for a three-dimensional
    /// stack. The shape does not give this once a slab is padded.
    #[serde(default)]
    pub expert_stride_bytes: Option<u64>,
    /// How the rows are ordered, where the packer had to say so. The shipped
    /// package writes it on the embedding table alone, as
    /// `vocabulary_row_major`; it is kept as text rather than an enum because
    /// a spelling this build has never seen is the packer's news to deliver,
    /// not a reason to refuse a document whose weights are all readable.
    #[serde(default)]
    pub layout: Option<String>,
}

impl Section {
    /// The expert count, for a three-dimensional stack.
    ///
    /// `None` is an answer rather than a failure: a two-dimensional section is
    /// not an expert stack and has no expert count to report.
    #[must_use]
    pub fn experts(&self) -> Option<u32> {
        match *self.shape.as_slice() {
            [experts, _, _] => Some(experts),
            _ => None,
        }
    }

    /// The live output width: [`Section::actual_output`] when the row is
    /// padded, the leading dimension otherwise - the second of three for an
    /// expert stack, whose leading dimension counts experts.
    #[must_use]
    pub fn out_dim(&self) -> u32 {
        if let Some(live) = self.actual_output {
            return live;
        }
        // An expert stack's leading dimension counts experts, so the live
        // width is the second of its three; everything else leads with it.
        match *self.shape.as_slice() {
            [_, rows, _] | [rows, ..] => rows,
            [] => 0,
        }
    }
}

/// One `.bin` file and the sections in it.
#[derive(Debug, Clone, Deserialize)]
pub struct FileLayout {
    /// Path relative to the package root.
    pub path: String,
    /// Total bytes.
    pub bytes: u64,
    /// The spans, in file order.
    pub sections: Vec<Section>,
}

impl FileLayout {
    /// The section called `name`.
    ///
    /// `None` is an answer rather than a failure: the caller is asking whether
    /// this file carries that span, and most files carry only some of them.
    #[must_use]
    pub fn section(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.name == name)
    }
}

/// A package's three weight sets.
#[derive(Debug, Clone, Deserialize)]
pub struct Layout {
    /// The target model's files.
    #[serde(deserialize_with = "one_or_many")]
    pub target: Vec<FileLayout>,
    /// The draft model's files.
    #[serde(default, deserialize_with = "one_or_many")]
    pub draft: Vec<FileLayout>,
    /// The vision tower's files. The shipped package writes this one as a bare
    /// object because the tower is a single file; see the module docs.
    #[serde(default, deserialize_with = "one_or_many")]
    pub vision: Vec<FileLayout>,
}

impl Layout {
    /// Reads a `layout.json`.
    ///
    /// # Errors
    ///
    /// [`serde_json::Error`] naming the field, which for an unreadable dtype
    /// names the dtype itself.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

/// Reads a weight set written either as an array of files or as one bare file.
///
/// Written as a visitor rather than an `untagged` enum on purpose: `untagged`
/// reports a document it cannot read as "data did not match any variant",
/// which would turn an unreadable dtype inside the vision tower - the one set
/// the package writes as a bare object - into an error that names neither the
/// dtype nor the field.
fn one_or_many<'de, D>(deserializer: D) -> Result<Vec<FileLayout>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct OneOrMany;

    impl<'de> Visitor<'de> for OneOrMany {
        type Value = Vec<FileLayout>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a list of file layouts, or one file layout")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
            Deserialize::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))
        }

        fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
            let one: FileLayout =
                Deserialize::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
            Ok(vec![one])
        }
    }

    deserializer.deserialize_any(OneOrMany)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layer 0 of the 35B-A3B package, read off its own `layout.json`.
    const LAYER_0: &str = r#"{
      "draft": [], "vision": [],
      "target": [{
        "bytes": 475201536, "path": "target/layer-0.bin",
        "sections": [
          {"bytes": 4096, "dtype": "bf16", "name": "input_norm",
           "offset": 16384, "shape": [2048]},
          {"actual_output": 12352, "bytes": 14450688, "dtype": "q4_affine",
           "group_size": 64, "name": "gdn_input", "offset": 32768,
           "shape": [12544, 2048], "storage_n": 256},
          {"bytes": 150994944, "dtype": "q4_affine", "group_size": 64,
           "expert_stride_bytes": 589824,
           "name": "experts.gate_proj", "offset": 19890176,
           "shape": [256, 512, 2048], "storage_n": 256}
        ]}]}"#;

    /// `embedding.bin` and the vision tower, whose spellings the fixture above
    /// does not contain: a bare object for a weight set, the embedding table's
    /// own dtype, and a row-order note.
    const TAIL: &str = r#"{
      "target": [{
        "bytes": 286081024, "path": "target/embedding.bin",
        "sections": [
          {"bytes": 254279680, "dtype": "q4_affine_weights", "group_size": 64,
           "layout": "vocabulary_row_major", "name": "embedding.weights",
           "offset": 16384, "shape": [248320, 2048]},
          {"bytes": 15892480, "dtype": "bf16", "name": "embedding.scales",
           "offset": 254296064, "shape": [248320, 32]}
        ]}],
      "vision": {
        "bytes": 901939200, "path": "vision/model.bin",
        "sections": [
          {"bytes": 3538944, "dtype": "bf16", "name": "patch_embed.weight",
           "offset": 16384, "shape": [1152, 1536]}
        ]}}"#;

    #[test]
    fn a_section_keeps_the_live_width_apart_from_the_padded_one() {
        let layout = Layout::from_json(LAYER_0).expect("layout parses");
        let gdn = layout.target[0].section("gdn_input").expect("gdn_input");
        assert_eq!(gdn.shape, vec![12544, 2048]);
        // A loader that reads `shape[0]` as the live width reads 192 rows of
        // padding as weights.
        assert_eq!(gdn.actual_output, Some(12352));
        assert_eq!(gdn.storage_n, Some(256));
        assert_eq!(gdn.group_size, Some(64));
        assert_eq!(gdn.out_dim(), 12352);
    }

    #[test]
    fn a_norm_carries_no_quantization_fields() {
        let layout = Layout::from_json(LAYER_0).expect("layout parses");
        let norm = layout.target[0].section("input_norm").expect("input_norm");
        assert_eq!(norm.dtype, SectionDtype::Bf16);
        assert_eq!(norm.group_size, None);
        assert_eq!(norm.actual_output, None);
        assert_eq!(norm.out_dim(), 2048);
    }

    #[test]
    fn an_expert_stack_is_three_dimensional() {
        let layout = Layout::from_json(LAYER_0).expect("layout parses");
        let g = layout.target[0].section("experts.gate_proj").expect("gate");
        assert_eq!(g.shape, vec![256, 512, 2048]);
        assert_eq!(g.experts(), Some(256));
        // No `actual_output` on a stack: the live width is the slab's own
        // leading dimension, not the expert count.
        assert_eq!(g.actual_output, None);
        assert_eq!(g.out_dim(), 512);
        // 256 slabs of 589,824 bytes is the whole 150,994,944-byte section, so
        // the stride is the padded slab and not a figure derived from 512.
        assert_eq!(g.expert_stride_bytes, Some(589_824));
        assert_eq!(256 * 589_824_u64, g.bytes);
    }

    #[test]
    fn an_unknown_dtype_is_refused_by_name() {
        let bad = LAYER_0.replace("\"bf16\"", "\"fp6\"");
        let err = Layout::from_json(&bad).expect_err("fp6 has no reader");
        assert!(
            format!("{err}").contains("fp6"),
            "the error names the dtype"
        );
    }

    #[test]
    fn a_weight_set_written_as_one_object_reads_as_one_file() {
        let layout = Layout::from_json(TAIL).expect("layout parses");
        assert_eq!(layout.vision.len(), 1);
        assert_eq!(layout.vision[0].path, "vision/model.bin");
        assert_eq!(
            layout.draft.len(),
            0,
            "an absent set is empty, not an error"
        );
    }

    #[test]
    fn the_embedding_table_keeps_its_dtype_and_its_row_order() {
        let layout = Layout::from_json(TAIL).expect("layout parses");
        let file = &layout.target[0];
        let codes = file.section("embedding.weights").expect("weights");
        assert_eq!(codes.dtype, SectionDtype::Q4AffineWeights);
        assert_eq!(codes.layout.as_deref(), Some("vocabulary_row_major"));
        // The scales are their own section, which is the whole reason the
        // codes carry a dtype of their own.
        let scales = file.section("embedding.scales").expect("scales");
        assert_eq!(scales.dtype, SectionDtype::Bf16);
        assert_eq!(scales.shape, vec![248_320, 32]);
        assert_eq!(codes.shape[1] / 64, scales.shape[1], "one scale per group");
    }

    #[test]
    fn a_section_nobody_asked_for_is_absent_rather_than_fatal() {
        let layout = Layout::from_json(LAYER_0).expect("layout parses");
        assert!(layout.target[0].section("attention_input").is_none());
    }

    #[test]
    fn an_unreadable_dtype_inside_a_bare_object_still_names_itself() {
        // Only the tower's dtype is spoiled: the point of the case is the one
        // weight set the package writes as a bare object.
        let bad = TAIL.replace(
            "\"bf16\", \"name\": \"patch_embed.weight\"",
            "\"fp6\", \"name\": \"patch_embed.weight\"",
        );
        assert_ne!(bad, TAIL, "the fixture still spells the tower's dtype");
        let err = Layout::from_json(&bad).expect_err("fp6 has no reader");
        let text = format!("{err}");
        assert!(
            text.contains("fp6"),
            "the error names the dtype, not a variant: {text}"
        );
    }
}

//! The `.glb` writer: a Gaussian cloud in, one binary glTF file out.
//!
//! The cloud is one mesh of one `POINTS` primitive carrying the
//! `KHR_gaussian_splatting` extension: a position, a unit rotation, a linear
//! scale, a linear opacity and the degree-0 spherical-harmonic colour per
//! Gaussian, each its own tightly packed `f32` accessor. A reader that knows
//! the extension draws splats; one that does not still finds a valid point
//! primitive and the standard `COLOR_0` beside it, so the file opens as a
//! coloured point cloud everywhere (the extension is listed as used, not as
//! required, for that reason).
//!
//! Three conventions differ from the `.ply` of [`super::ply`] and all three
//! are observable:
//!
//! - The values are **activated**. The extension stores a linear scale and an
//!   opacity in `[0, 1]`, where the `.ply` stores their logarithm and logit.
//! - Rotations are `(x, y, z, w)`, glTF's order, and renormalised, because the
//!   extension lets a renderer use them without normalising.
//! - The object is built z-up and glTF is y-up with `-z` forward, so a
//!   position `(x, y, z)` becomes `(x, z, -y)` and every rotation is
//!   left-multiplied by that `-90`-degree turn about x. (The `.ply` turns the
//!   other way, into the y-down frame its viewers expect.)
//!
//! `COLOR_0` holds what the extension asks a fallback to hold: the diffuse
//! colour `0.5 + C0 * dc`, clamped, decoded from the display's sRGB transfer
//! to linear, with the opacity as alpha - as normalised bytes, since it is
//! only ever a preview.

use serde_json::json;

use super::ply::{GaussianCloud, PlyError};

/// The extension's name.
const EXTENSION: &str = "KHR_gaussian_splatting";
/// The degree-0 real spherical harmonic, `1 / (2 sqrt(pi))`.
const SH_C0: f32 = 0.282_094_8;
/// glTF's `FLOAT` component type.
const FLOAT: u32 = 5126;
/// glTF's `UNSIGNED_BYTE` component type.
const UNSIGNED_BYTE: u32 = 5121;
/// glTF's `ARRAY_BUFFER` target: every view here is vertex data.
const ARRAY_BUFFER: u32 = 34962;
/// The container's magic, `glTF`.
const MAGIC: u32 = 0x4654_6C67;
/// The JSON chunk's type, `JSON`.
const CHUNK_JSON: u32 = 0x4E4F_534A;
/// The binary chunk's type, `BIN\0`.
const CHUNK_BIN: u32 = 0x004E_4942;

/// One attribute's place in the binary chunk.
struct View {
    semantic: String,
    offset: usize,
    length: usize,
    components: &'static str,
    component_type: u32,
    normalized: bool,
    bounds: Option<([f32; 3], [f32; 3])>,
}

/// The binary chunk as it grows, one view per attribute.
struct Chunk {
    bytes: Vec<u8>,
    views: Vec<View>,
}

impl Chunk {
    /// Appends an `f32` attribute of `N` components per Gaussian.
    fn floats<const N: usize>(
        &mut self,
        semantic: &str,
        components: &'static str,
        rows: impl Iterator<Item = [f32; N]>,
    ) -> &mut View {
        let offset = self.bytes.len();
        for row in rows {
            for value in row {
                self.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        self.view(semantic, offset, components, FLOAT, false)
    }

    /// Records the view that ends at the chunk's current length. Every view
    /// here is a whole number of four-byte elements, so each starts aligned.
    fn view(
        &mut self,
        semantic: &str,
        offset: usize,
        components: &'static str,
        component_type: u32,
        normalized: bool,
    ) -> &mut View {
        self.views.push(View {
            semantic: semantic.to_owned(),
            offset,
            length: self.bytes.len() - offset,
            components,
            component_type,
            normalized,
            bounds: None,
        });
        self.views.last_mut().expect("a view was just pushed")
    }
}

/// Writes the cloud as a binary glTF (`.glb`) with `KHR_gaussian_splatting`.
///
/// # Errors
///
/// [`PlyError::Lengths`] when the arrays disagree, and [`PlyError::Empty`] for
/// a cloud with no Gaussians: the refusals are the cloud's, whichever file it
/// was headed for.
pub fn write_glb(cloud: &GaussianCloud) -> Result<Vec<u8>, PlyError> {
    let count = cloud.len();
    if count == 0 {
        return Err(PlyError::Empty);
    }
    if cloud.features_dc.len() != count
        || cloud.scaling.len() != count
        || cloud.rotation.len() != count
        || cloud.opacity.len() != count
    {
        return Err(PlyError::Lengths {
            xyz: count,
            colour: cloud.features_dc.len(),
            scale: cloud.scaling.len(),
            rotation: cloud.rotation.len(),
            opacity: cloud.opacity.len(),
        });
    }

    let mut chunk = Chunk {
        bytes: Vec::with_capacity(count * 60),
        views: Vec::with_capacity(6),
    };
    // (x, y, z) -> (x, z, -y); the accessor must state the positions' bounds.
    let position = |index: usize| {
        let [x, y, z] = cloud.xyz[index];
        [x, z, -y]
    };
    let mut low = [f32::INFINITY; 3];
    let mut high = [f32::NEG_INFINITY; 3];
    for index in 0..count {
        for (axis, value) in position(index).into_iter().enumerate() {
            low[axis] = low[axis].min(value);
            high[axis] = high[axis].max(value);
        }
    }
    chunk
        .floats("POSITION", "VEC3", (0..count).map(position))
        .bounds = Some((low, high));
    chunk.floats(
        &format!("{EXTENSION}:ROTATION"),
        "VEC4",
        cloud.rotation.iter().map(|q| y_up_rotation(*q)),
    );
    chunk.floats(
        &format!("{EXTENSION}:SCALE"),
        "VEC3",
        cloud.scaling.iter().map(|s| s.map(|v| v.max(0.0))),
    );
    chunk.floats(
        &format!("{EXTENSION}:OPACITY"),
        "SCALAR",
        cloud.opacity.iter().map(|o| [o.clamp(0.0, 1.0)]),
    );
    chunk.floats(
        &format!("{EXTENSION}:SH_DEGREE_0_COEF_0"),
        "VEC3",
        cloud.features_dc.iter().copied(),
    );
    let offset = chunk.bytes.len();
    for (dc, opacity) in cloud.features_dc.iter().zip(&cloud.opacity) {
        let [r, g, b] = dc.map(|v| unorm8(srgb_to_linear((0.5 + SH_C0 * v).clamp(0.0, 1.0))));
        chunk
            .bytes
            .extend_from_slice(&[r, g, b, unorm8(opacity.clamp(0.0, 1.0))]);
    }
    chunk.view("COLOR_0", offset, "VEC4", UNSIGNED_BYTE, true);

    Ok(container(&document(&chunk, count), &chunk.bytes))
}

/// The glTF document describing `chunk`'s views over `count` Gaussians.
fn document(chunk: &Chunk, count: usize) -> serde_json::Value {
    let mut attributes = serde_json::Map::new();
    let mut accessors = Vec::with_capacity(chunk.views.len());
    let mut buffer_views = Vec::with_capacity(chunk.views.len());
    for (index, view) in chunk.views.iter().enumerate() {
        attributes.insert(view.semantic.clone(), json!(index));
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": view.offset,
            "byteLength": view.length,
            "target": ARRAY_BUFFER,
        }));
        let mut accessor = json!({
            "bufferView": index,
            "componentType": view.component_type,
            "count": count,
            "type": view.components,
        });
        if view.normalized {
            accessor["normalized"] = json!(true);
        }
        if let Some((low, high)) = view.bounds {
            accessor["min"] = json!(low);
            accessor["max"] = json!(high);
        }
        accessors.push(accessor);
    }
    json!({
        "asset": { "version": "2.0", "generator": "shiso" },
        "extensionsUsed": [EXTENSION],
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{
            "primitives": [{
                "attributes": attributes,
                "mode": 0,
                "extensions": {
                    EXTENSION: {
                        "kernel": "ellipse",
                        "colorSpace": "srgb_rec709_display",
                        "projection": "perspective",
                        "sortingMethod": "cameraDistance",
                    }
                }
            }]
        }],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{ "byteLength": chunk.bytes.len() }],
    })
}

/// The binary container: the header, the JSON chunk padded with spaces and
/// the binary chunk padded with zeros, each to a multiple of four bytes.
///
/// The container's lengths are `u32`. The largest cloud the family places
/// (1048576 Gaussians of 60 bytes) is 63 MB, far inside it.
fn container(document: &serde_json::Value, binary: &[u8]) -> Vec<u8> {
    let mut json = document.to_string().into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let binary_len = binary.len().next_multiple_of(4);
    let total = 12 + 8 + json.len() + 8 + binary_len;
    let word = |value: usize| {
        u32::try_from(value)
            .expect("a splat file is far below the container's 4 GiB")
            .to_le_bytes()
    };
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&word(total));
    out.extend_from_slice(&word(json.len()));
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&word(binary_len));
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(binary);
    out.resize(total, 0);
    out
}

/// A `(w, x, y, z)` rotation of the z-up object as the unit `(x, y, z, w)`
/// rotation of the y-up file: the `-90`-degree turn about x, times the
/// Gaussian's own.
fn y_up_rotation(rotation: [f32; 4]) -> [f32; 4] {
    let half = std::f32::consts::FRAC_1_SQRT_2;
    let [bw, bx, by, bz] = rotation;
    // (half, -half, 0, 0) * (bw, bx, by, bz), the Hamilton product, as
    // (w, x, y, z).
    let turned = [
        half * (bw + bx),
        half * (bx - bw),
        half * (by + bz),
        half * (bz - by),
    ];
    let norm = turned.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        [
            turned[1] / norm,
            turned[2] / norm,
            turned[3] / norm,
            turned[0] / norm,
        ]
    } else {
        // A zero or non-finite rotation has no direction to keep; the
        // identity is the one unit quaternion that claims none.
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// The sRGB transfer function's inverse on `[0, 1]`.
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// A value in `[0, 1]` as a normalised byte, rounded to nearest.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to [0, 255] before the cast"
)]
fn unorm8(value: f32) -> u8 {
    (value * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_gaussian() -> GaussianCloud {
        GaussianCloud {
            xyz: vec![[0.1, 0.2, 0.3]],
            features_dc: vec![[0.0, 1.0, -5.0]],
            scaling: vec![[0.5, 0.25, 0.125]],
            rotation: vec![[2.0, 0.0, 0.0, 0.0]],
            opacity: vec![0.5],
        }
    }

    /// A JSON number as an index.
    fn index(value: &serde_json::Value) -> usize {
        usize::try_from(value.as_u64().expect("an unsigned number")).expect("an index")
    }

    fn word(bytes: &[u8], at: usize) -> usize {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize
    }

    /// The document and the binary chunk of a written file.
    fn parts(bytes: &[u8]) -> (serde_json::Value, &[u8]) {
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(word(bytes, 4), 2, "container version");
        assert_eq!(word(bytes, 8), bytes.len(), "total length");
        let json_len = word(bytes, 12);
        assert_eq!(&bytes[16..20], b"JSON");
        assert_eq!(json_len % 4, 0);
        let json = serde_json::from_slice(&bytes[20..20 + json_len]).expect("a JSON chunk");
        let bin = 20 + json_len;
        let bin_len = word(bytes, bin);
        assert_eq!(&bytes[bin + 4..bin + 8], b"BIN\0");
        assert_eq!(bin + 8 + bin_len, bytes.len());
        (json, &bytes[bin + 8..])
    }

    /// The floats of the accessor bound to `semantic`.
    fn floats(json: &serde_json::Value, bin: &[u8], semantic: &str) -> Vec<f32> {
        let attribute = &json["meshes"][0]["primitives"][0]["attributes"][semantic];
        assert!(attribute.is_u64(), "no {semantic} attribute");
        let accessor = &json["accessors"][index(attribute)];
        assert_eq!(accessor["componentType"], FLOAT);
        let view = &json["bufferViews"][index(&accessor["bufferView"])];
        let offset = index(&view["byteOffset"]);
        let length = index(&view["byteLength"]);
        assert_eq!(offset % 4, 0, "{semantic} is aligned");
        bin[offset..offset + length]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn the_primitive_is_points_with_the_splat_extension() {
        let bytes = write_glb(&one_gaussian()).expect("writes");
        let (json, _) = parts(&bytes);
        assert_eq!(json["asset"]["version"], "2.0");
        assert_eq!(json["extensionsUsed"][0], EXTENSION);
        assert!(json.get("extensionsRequired").is_none());
        let primitive = &json["meshes"][0]["primitives"][0];
        assert_eq!(primitive["mode"], 0);
        let extension = &primitive["extensions"][EXTENSION];
        assert_eq!(extension["kernel"], "ellipse");
        assert_eq!(extension["colorSpace"], "srgb_rec709_display");
        for accessor in json["accessors"].as_array().expect("accessors") {
            assert_eq!(accessor["count"], 1);
        }
    }

    #[test]
    fn the_object_is_turned_from_z_up_to_gltf_y_up() {
        let bytes = write_glb(&one_gaussian()).expect("writes");
        let (json, bin) = parts(&bytes);
        assert_eq!(floats(&json, bin, "POSITION"), [0.1, 0.3, -0.2]);
        let accessor = &json["accessors"][0];
        assert_eq!(accessor["min"], json!([0.1f32, 0.3f32, -0.2f32]));
        assert_eq!(accessor["max"], accessor["min"]);
        // An object's up, +z, must come out as the file's up, +y.
        let mut cloud = one_gaussian();
        cloud.xyz = vec![[0.0, 0.0, 1.0]];
        let bytes = write_glb(&cloud).expect("writes");
        let (json, bin) = parts(&bytes);
        assert_eq!(floats(&json, bin, "POSITION"), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn rotations_are_unit_xyzw_and_carry_the_turn() {
        let bytes = write_glb(&one_gaussian()).expect("writes");
        let (json, bin) = parts(&bytes);
        let q = floats(&json, bin, "KHR_gaussian_splatting:ROTATION");
        let half = std::f32::consts::FRAC_1_SQRT_2;
        // The identity (given unnormalised) becomes the turn itself:
        // (w, x, y, z) = (half, -half, 0, 0), written (x, y, z, w).
        for (got, want) in q.iter().zip([-half, 0.0, 0.0, half]) {
            assert!((got - want).abs() < 1e-6, "{q:?}");
        }
    }

    #[test]
    fn the_turn_moves_a_gaussians_axes_with_its_position() {
        // A Gaussian's local z axis, under the identity rotation, points along
        // the object's +z; after the turn it must point along the file's +y,
        // as the positions do.
        let turn = y_up_rotation([1.0, 0.0, 0.0, 0.0]);
        let (axis, real) = ([turn[0], turn[1], turn[2]], turn[3]);
        let vector = [0.0f32, 0.0, 1.0];
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        // Rotating by a unit quaternion: v + 2 axis x (axis x v + real v).
        let inner = cross(axis, vector);
        let inner = [
            inner[0] + real * vector[0],
            inner[1] + real * vector[1],
            inner[2] + real * vector[2],
        ];
        let outer = cross(axis, inner);
        let turned = [
            vector[0] + 2.0 * outer[0],
            vector[1] + 2.0 * outer[1],
            vector[2] + 2.0 * outer[2],
        ];
        for (got, want) in turned.iter().zip([0.0, 1.0, 0.0]) {
            assert!((got - want).abs() < 1e-6, "{turned:?}");
        }
    }

    #[test]
    fn scale_and_opacity_are_written_activated() {
        let bytes = write_glb(&one_gaussian()).expect("writes");
        let (json, bin) = parts(&bytes);
        assert_eq!(
            floats(&json, bin, "KHR_gaussian_splatting:SCALE"),
            [0.5, 0.25, 0.125]
        );
        assert_eq!(floats(&json, bin, "KHR_gaussian_splatting:OPACITY"), [0.5]);
        assert_eq!(
            floats(&json, bin, "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0"),
            [0.0, 1.0, -5.0]
        );
    }

    #[test]
    fn the_fallback_colour_is_linear_with_opacity_as_alpha() {
        let bytes = write_glb(&one_gaussian()).expect("writes");
        let (json, bin) = parts(&bytes);
        let attribute = &json["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"];
        assert!(attribute.is_u64(), "a fallback colour");
        let accessor = &json["accessors"][index(attribute)];
        assert_eq!(accessor["componentType"], UNSIGNED_BYTE);
        assert_eq!(accessor["normalized"], true);
        assert_eq!(accessor["type"], "VEC4");
        let offset = index(&json["bufferViews"][index(&accessor["bufferView"])]["byteOffset"]);
        let rgba = &bin[offset..offset + 4];
        // dc 0 is mid grey in sRGB, 0.214 linear; dc 1 is 0.782 sRGB, 0.574
        // linear; a colour below zero clamps to black. Alpha is the opacity.
        assert_eq!(rgba, [55, 146, 0, 128]);
    }

    #[test]
    fn a_degenerate_rotation_becomes_the_identity() {
        let identity = [0.0f32, 0.0, 0.0, 1.0].map(f32::to_bits);
        for rotation in [[0.0; 4], [f32::NAN, 0.0, 0.0, 0.0]] {
            assert_eq!(y_up_rotation(rotation).map(f32::to_bits), identity);
        }
    }

    #[test]
    fn every_gaussian_gets_a_row_and_the_chunks_are_padded() {
        let mut cloud = one_gaussian();
        cloud.xyz = vec![[0.0; 3], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]];
        cloud.features_dc = vec![[0.0; 3]; 3];
        cloud.scaling = vec![[1.0; 3]; 3];
        cloud.rotation = vec![[1.0, 0.0, 0.0, 0.0]; 3];
        cloud.opacity = vec![0.5; 3];
        let bytes = write_glb(&cloud).expect("writes");
        assert_eq!(bytes.len() % 4, 0);
        let (json, bin) = parts(&bytes);
        assert_eq!(bin.len(), 3 * 60);
        assert_eq!(json["buffers"][0]["byteLength"], 180);
        // (1, 2, 3) -> (1, 3, -2) and (-1, -2, -3) -> (-1, -3, 2).
        assert_eq!(
            json["accessors"][0]["min"],
            json!([-1.0f32, -3.0f32, -2.0f32])
        );
        assert_eq!(json["accessors"][0]["max"], json!([1.0f32, 3.0f32, 2.0f32]));
    }

    #[test]
    fn an_empty_or_ragged_cloud_is_refused() {
        let mut cloud = one_gaussian();
        cloud.opacity.clear();
        assert!(matches!(
            write_glb(&cloud),
            Err(PlyError::Lengths { opacity: 0, .. })
        ));
        cloud.xyz.clear();
        assert_eq!(write_glb(&cloud), Err(PlyError::Empty));
    }
}

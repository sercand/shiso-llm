//! The `.ply` writer: a Gaussian cloud in, the file a viewer reads out.
//!
//! The format is the one every 3D Gaussian-splatting viewer speaks: a
//! `binary_little_endian` vertex element of seventeen `f32` fields, one row per
//! Gaussian. Two conventions are baked in and both are observable:
//!
//! - The object is built z-up and written y-up, so a position `(x, y, z)`
//!   becomes `(x, -z, y)` and every rotation is left-multiplied by the
//!   corresponding `+90`-degree quaternion.
//! - Opacity and scale are stored as the *inverse* of their activations, which
//!   is what the format's readers expect: a renderer applies a sigmoid to
//!   `opacity` and an exponential to `scale`.
//!
//! Both inverses saturate, so both inputs are clamped first: a sigmoid that
//! reached exactly 1.0 would otherwise be written as `+inf`, and a viewer that
//! rejects the file is worse than one that renders a nearly transparent
//! Gaussian.

/// The smallest value the inverse activations are evaluated at.
///
/// An opacity of exactly 1.0 or a scale of exactly 0.0 is a saturated
/// activation, not a value the format can carry.
pub const PLY_EPS: f32 = 1e-6;

/// The seventeen vertex fields, in the order the header declares them.
pub const PLY_FIELDS: [&str; 17] = [
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
    "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
];

/// One Gaussian cloud, already activated: positions in object space, unit
/// rotations, opacity in `(0, 1)` and positive scales.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianCloud {
    /// Positions, `(x, y, z)` per Gaussian, in the object's z-up space.
    pub xyz: Vec<[f32; 3]>,
    /// Degree-0 spherical-harmonic colour, one `(r, g, b)` per Gaussian.
    pub features_dc: Vec<[f32; 3]>,
    /// Per-axis scales, already softplus-activated and floored.
    pub scaling: Vec<[f32; 3]>,
    /// Unit quaternions in `(w, x, y, z)` order.
    pub rotation: Vec<[f32; 4]>,
    /// Opacities in `(0, 1)`.
    pub opacity: Vec<f32>,
}

impl GaussianCloud {
    /// How many Gaussians the cloud holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.xyz.len()
    }

    /// Whether the cloud is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.xyz.is_empty()
    }

    /// Keeps the first `count` Gaussians.
    ///
    /// The decoder answers one group per anchor and the request counts
    /// Gaussians; this is where the two are reconciled when the anchor budget
    /// rounds up.
    pub fn truncate(&mut self, count: usize) {
        self.xyz.truncate(count);
        self.features_dc.truncate(count);
        self.scaling.truncate(count);
        self.rotation.truncate(count);
        self.opacity.truncate(count);
    }
}

/// A cloud the writer refuses.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlyError {
    /// The per-Gaussian arrays disagree on how many Gaussians there are.
    #[error(
        "the cloud's arrays disagree on length: xyz {xyz}, colour {colour}, scale {scale}, \
         rotation {rotation}, opacity {opacity}"
    )]
    Lengths {
        /// Positions.
        xyz: usize,
        /// Colours.
        colour: usize,
        /// Scales.
        scale: usize,
        /// Rotations.
        rotation: usize,
        /// Opacities.
        opacity: usize,
    },
    /// The cloud holds no Gaussians, which is not a splat a viewer can open.
    #[error("the cloud is empty: a splat needs at least one Gaussian")]
    Empty,
}

/// Writes the cloud as a binary little-endian `.ply`.
///
/// # Errors
///
/// [`PlyError::Lengths`] when the arrays disagree, and [`PlyError::Empty`] for
/// a cloud with no Gaussians.
pub fn write_ply(cloud: &GaussianCloud) -> Result<Vec<u8>, PlyError> {
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

    let mut out = Vec::with_capacity(header_len(count) + count * PLY_FIELDS.len() * 4);
    out.extend_from_slice(b"ply\nformat binary_little_endian 1.0\n");
    out.extend_from_slice(format!("element vertex {count}\n").as_bytes());
    for field in PLY_FIELDS {
        out.extend_from_slice(format!("property float {field}\n").as_bytes());
    }
    out.extend_from_slice(b"end_header\n");

    // The z-up -> y-up rotation, as a quaternion: +90 degrees about x.
    let half = std::f32::consts::FRAC_1_SQRT_2;
    for index in 0..count {
        let [x, y, z] = cloud.xyz[index];
        // (x, y, z) -> (x, -z, y).
        push(&mut out, [x, -z, y]);
        // Normals are unused by every renderer and are written as zero.
        push(&mut out, [0.0, 0.0, 0.0]);
        push(&mut out, cloud.features_dc[index]);
        push(
            &mut out,
            [inverse_sigmoid(
                cloud.opacity[index].clamp(PLY_EPS, 1.0 - PLY_EPS),
            )],
        );
        let scale = cloud.scaling[index];
        push(
            &mut out,
            [
                scale[0].max(PLY_EPS).ln(),
                scale[1].max(PLY_EPS).ln(),
                scale[2].max(PLY_EPS).ln(),
            ],
        );
        push(
            &mut out,
            quaternion_multiply([half, half, 0.0, 0.0], cloud.rotation[index]),
        );
    }
    Ok(out)
}

/// How long the header is, for the capacity hint.
fn header_len(count: usize) -> usize {
    "ply\nformat binary_little_endian 1.0\n".len()
        + format!("element vertex {count}\n").len()
        + PLY_FIELDS
            .iter()
            .map(|field| format!("property float {field}\n").len())
            .sum::<usize>()
        + "end_header\n".len()
}

/// Appends `values` little-endian.
fn push<const N: usize>(out: &mut Vec<u8>, values: [f32; N]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

/// `log(p / (1 - p))`, the inverse of the sigmoid.
fn inverse_sigmoid(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

/// The Hamilton product `a * b` of `(w, x, y, z)` quaternions.
///
/// `a` is the fixed y-up rotation and `b` the Gaussian's own; the order matters
/// because the product is not commutative, and rotating the other way puts the
/// object on its side.
fn quaternion_multiply(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_gaussian() -> GaussianCloud {
        GaussianCloud {
            xyz: vec![[0.1, 0.2, 0.3]],
            features_dc: vec![[0.25, 0.5, 0.75]],
            scaling: vec![[1.0, 1.0, 1.0]],
            rotation: vec![[1.0, 0.0, 0.0, 0.0]],
            opacity: vec![0.5],
        }
    }

    /// The header, then the vertices, as a byte slice.
    fn split(bytes: &[u8]) -> (&str, &[u8]) {
        let end = bytes
            .windows(b"end_header\n".len())
            .position(|window| window == b"end_header\n")
            .expect("a header terminator")
            + b"end_header\n".len();
        (
            std::str::from_utf8(&bytes[..end]).expect("the header is ASCII"),
            &bytes[end..],
        )
    }

    /// The seventeen floats of vertex `index`.
    fn vertex(bytes: &[u8], index: usize) -> Vec<f32> {
        let start = index * PLY_FIELDS.len() * 4;
        bytes[start..start + PLY_FIELDS.len() * 4]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn the_header_names_every_field_in_order() {
        let bytes = write_ply(&one_gaussian()).expect("writes");
        let (header, body) = split(&bytes);
        assert!(header.starts_with("ply\nformat binary_little_endian 1.0\n"));
        assert!(header.contains("element vertex 1\n"));
        for field in PLY_FIELDS {
            assert!(
                header.contains(&format!("property float {field}\n")),
                "{field} is missing from the header"
            );
        }
        assert_eq!(body.len(), PLY_FIELDS.len() * 4);
    }

    #[test]
    fn the_object_is_turned_from_z_up_to_y_up() {
        let bytes = write_ply(&one_gaussian()).expect("writes");
        let (_, body) = split(&bytes);
        let values = vertex(body, 0);
        assert_eq!(&values[0..3], &[0.1, -0.3, 0.2], "xyz");
        assert_eq!(&values[3..6], &[0.0, 0.0, 0.0], "normals");
        assert_eq!(&values[6..9], &[0.25, 0.5, 0.75], "colour");
    }

    #[test]
    fn the_rotations_are_left_multiplied_by_the_y_up_turn() {
        let bytes = write_ply(&one_gaussian()).expect("writes");
        let (_, body) = split(&bytes);
        let values = vertex(body, 0);
        let half = std::f32::consts::FRAC_1_SQRT_2;
        // (sqrt(1/2), sqrt(1/2), 0, 0) * (1, 0, 0, 0) = (sqrt(1/2), sqrt(1/2), 0, 0).
        assert_eq!(&values[13..17], &[half, half, 0.0, 0.0]);
    }

    #[test]
    fn a_half_opacity_is_written_as_zero_logit() {
        let bytes = write_ply(&one_gaussian()).expect("writes");
        let (_, body) = split(&bytes);
        let values = vertex(body, 0);
        assert!(values[9].abs() < 1e-6, "opacity {}", values[9]);
        // A unit scale is a zero log-scale.
        assert!(values[10].abs() < 1e-6, "scale {}", values[10]);
    }

    #[test]
    fn saturated_activations_are_clamped_rather_than_written_as_infinity() {
        let mut cloud = one_gaussian();
        cloud.opacity = vec![1.0];
        cloud.scaling = vec![[0.0, 0.0, 0.0]];
        let bytes = write_ply(&cloud).expect("writes");
        let (_, body) = split(&bytes);
        let values = vertex(body, 0);
        assert!(values[9].is_finite(), "opacity {}", values[9]);
        assert!(values[10].is_finite(), "scale {}", values[10]);
        assert!(values[9] > 10.0, "a saturated opacity is a large logit");
        assert!(values[10] < -10.0, "a zero scale is a large negative log");
    }

    #[test]
    fn every_gaussian_gets_a_row() {
        let mut cloud = one_gaussian();
        cloud.xyz = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        cloud.features_dc = vec![[0.0; 3]; 3];
        cloud.scaling = vec![[1.0; 3]; 3];
        cloud.rotation = vec![[1.0, 0.0, 0.0, 0.0]; 3];
        cloud.opacity = vec![0.5; 3];
        let bytes = write_ply(&cloud).expect("writes");
        let (header, body) = split(&bytes);
        assert!(header.contains("element vertex 3\n"));
        assert_eq!(body.len(), 3 * PLY_FIELDS.len() * 4);
        assert!((vertex(body, 2)[0] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn the_bytes_match_the_reference_writer_exactly() {
        // Produced by the reference's own writer (`plyfile` + numpy) for
        // `one_gaussian`: the header, then these 68 bytes.
        const REFERENCE_BODY: [u8; 68] = [
            205, 204, 204, 61, 154, 153, 153, 190, 205, 204, 76, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 128, 62, 0, 0, 0, 63, 0, 0, 64, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 243, 4, 53, 63, 243, 4, 53, 63, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let bytes = write_ply(&one_gaussian()).expect("writes");
        let (header, body) = split(&bytes);
        assert_eq!(
            header,
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n\
             property float x\nproperty float y\nproperty float z\n\
             property float nx\nproperty float ny\nproperty float nz\n\
             property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n\
             property float opacity\nproperty float scale_0\nproperty float scale_1\n\
             property float scale_2\nproperty float rot_0\nproperty float rot_1\n\
             property float rot_2\nproperty float rot_3\nend_header\n"
        );
        assert_eq!(body, REFERENCE_BODY);
    }

    #[test]
    fn an_empty_cloud_is_refused() {
        let mut cloud = one_gaussian();
        cloud.xyz.clear();
        cloud.features_dc.clear();
        cloud.scaling.clear();
        cloud.rotation.clear();
        cloud.opacity.clear();
        assert_eq!(write_ply(&cloud), Err(PlyError::Empty));
    }

    #[test]
    fn arrays_that_disagree_are_refused() {
        let mut cloud = one_gaussian();
        cloud.opacity.clear();
        assert!(matches!(
            write_ply(&cloud),
            Err(PlyError::Lengths { opacity: 0, .. })
        ));
    }
}

//! The Gaussian decoder's activations: raw per-anchor outputs to a cloud.
//!
//! The decoder's last layer answers fifteen numbers per Gaussian - an offset, a
//! colour, a scale, a rotation, an opacity and an offset scale - and none of
//! them is usable until it is activated. The activations are the reference's,
//! including the two biases baked into the checkpoint's representation config:
//! an object whose opacities started at zero would be invisible, and one whose
//! scales started at zero would be points.

/// The checkpoint's representation config: how the decoder's outputs become a
/// cloud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepresentationConfig {
    /// Gaussians per anchor.
    pub num_gaussians: usize,
    /// The decoder's output multiplier for the offset.
    pub lr_xyz: f32,
    /// The decoder's output multiplier for the colour.
    pub lr_features: f32,
    /// The decoder's output multiplier for the scale.
    pub lr_scaling: f32,
    /// The decoder's output multiplier for the rotation.
    pub lr_rotation: f32,
    /// The decoder's output multiplier for the opacity.
    pub lr_opacity: f32,
    /// How far an offset may reach, as a fraction of its cell.
    pub perturbe_size: f32,
    /// The floor added in quadrature to every scale.
    pub kernel_size: f32,
    /// The scale the raw scale is biased towards.
    pub scaling_bias: f32,
    /// The opacity the raw opacity is biased towards.
    pub opacity_bias: f32,
}

impl RepresentationConfig {
    /// The shipped checkpoint's config.
    #[must_use]
    pub const fn idol() -> Self {
        Self {
            num_gaussians: 32,
            lr_xyz: 1.0,
            lr_features: 1.0,
            lr_scaling: 1.0,
            lr_rotation: 0.1,
            lr_opacity: 1.0,
            perturbe_size: 1.5,
            kernel_size: 0.0009,
            scaling_bias: 0.004,
            opacity_bias: 0.1,
        }
    }

    /// The decoder's output width per anchor.
    #[must_use]
    pub const fn outputs_per_anchor(&self) -> usize {
        self.num_gaussians * OUTPUTS_PER_GAUSSIAN
    }
}

/// Fifteen numbers per Gaussian: offset 3, colour 3, scale 3, rotation 4,
/// opacity 1, offset scale 1.
pub const OUTPUTS_PER_GAUSSIAN: usize = 15;

/// The activations' inputs: the decoder's raw output, the anchors it was asked
/// about, and the two tensors the checkpoint stores beside the weights.
#[derive(Debug)]
pub struct RawOutput<'a> {
    /// The decoder's output, `(anchors * num_gaussians * 15)` row-major.
    pub features: &'a [f32],
    /// The anchor each group of `num_gaussians` belongs to.
    pub anchors: &'a [[f32; 3]],
    /// The fixed per-Gaussian offset pattern, `(num_gaussians, 3)`.
    pub perturbation: &'a [[f32; 3]],
    /// The checkpoint's base offset scale.
    pub base_offset_scale: f32,
}

/// A raw output whose shape is not the config's.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GaussianError {
    /// The decoder answered a different number of values than the config
    /// expects.
    #[error("the decoder answered {got} values for {anchors} anchors, expected {want}")]
    Shape {
        /// The anchors the decoder was asked about.
        anchors: usize,
        /// The values it answered.
        got: usize,
        /// The values the config expects.
        want: usize,
    },
    /// The perturbation table is not `num_gaussians` long.
    #[error("the offset perturbation table is {got} long, expected {want}")]
    Perturbation {
        /// The table's length.
        got: usize,
        /// The config's Gaussian count.
        want: usize,
    },
}

/// Activates the decoder's raw output into a cloud.
///
/// # Errors
///
/// [`GaussianError::Shape`] when the output is not
/// `anchors * num_gaussians * 15`, and [`GaussianError::Perturbation`] when the
/// offset pattern is the wrong length.
pub fn activate(
    raw: &RawOutput<'_>,
    config: &RepresentationConfig,
) -> Result<crate::idol::ply::GaussianCloud, GaussianError> {
    let anchors = raw.anchors.len();
    let want = anchors * config.outputs_per_anchor();
    if raw.features.len() != want {
        return Err(GaussianError::Shape {
            anchors,
            got: raw.features.len(),
            want,
        });
    }
    if raw.perturbation.len() != config.num_gaussians {
        return Err(GaussianError::Perturbation {
            got: raw.perturbation.len(),
            want: config.num_gaussians,
        });
    }

    let total = anchors * config.num_gaussians;
    let mut cloud = crate::idol::ply::GaussianCloud {
        xyz: Vec::with_capacity(total),
        features_dc: Vec::with_capacity(total),
        scaling: Vec::with_capacity(total),
        rotation: Vec::with_capacity(total),
        opacity: Vec::with_capacity(total),
    };
    let opacity_offset = inverse_sigmoid(config.opacity_bias);
    for (anchor_index, anchor) in raw.anchors.iter().enumerate() {
        for gaussian in 0..config.num_gaussians {
            let base = anchor_index * config.outputs_per_anchor() + gaussian * OUTPUTS_PER_GAUSSIAN;
            let at = |offset: usize| raw.features[base + offset];

            // The offset is bounded by the cell's half-width times the
            // activated offset scale, which is what keeps a Gaussian inside
            // the region its anchor was placed in.
            let offset_scale = softplus(at(14) + raw.base_offset_scale);
            let perturbation = raw.perturbation[gaussian];
            let reach = 0.5 * config.perturbe_size * offset_scale;
            let mut position = [0.0f32; 3];
            for axis in 0..3 {
                let offset = (at(axis) * config.lr_xyz + perturbation[axis]).tanh() * reach;
                // The object is built in [-0.5, 0.5]^3 around the origin.
                position[axis] = offset + anchor[axis] - 0.5;
            }
            cloud.xyz.push(position);

            cloud.features_dc.push([
                at(3) * config.lr_features,
                at(4) * config.lr_features,
                at(5) * config.lr_features,
            ]);

            let mut scale = [0.0f32; 3];
            for (axis, slot) in scale.iter_mut().enumerate() {
                let activated = softplus(at(6 + axis) * config.lr_scaling + config.scaling_bias);
                *slot = (activated * activated + config.kernel_size * config.kernel_size).sqrt();
            }
            cloud.scaling.push(scale);

            let rotation = normalize_quaternion([
                at(9) * config.lr_rotation + 1.0,
                at(10) * config.lr_rotation,
                at(11) * config.lr_rotation,
                at(12) * config.lr_rotation,
            ]);
            cloud.rotation.push(rotation);

            cloud
                .opacity
                .push(sigmoid(at(13) * config.lr_opacity + opacity_offset));
        }
    }
    Ok(cloud)
}

/// `ln(1 + exp(x))`, in the form that does not overflow.
fn softplus(x: f32) -> f32 {
    // For large x, `exp(x)` overflows and the answer is x to f32 precision.
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// `1 / (1 + exp(-x))`, in the form that does not overflow.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `ln(p / (1 - p))`.
fn inverse_sigmoid(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

/// Scales a quaternion to unit length, falling back to identity.
fn normalize_quaternion(q: [f32; 4]) -> [f32; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm <= f32::EPSILON {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_zero(anchors: usize) -> Vec<f32> {
        vec![0.0f32; anchors * RepresentationConfig::idol().outputs_per_anchor()]
    }

    fn perturbation(config: &RepresentationConfig) -> Vec<[f32; 3]> {
        vec![[0.0; 3]; config.num_gaussians]
    }

    #[test]
    fn a_zero_output_is_the_biased_default_not_a_zero() {
        let config = RepresentationConfig::idol();
        let features = raw_zero(1);
        let anchors = [[0.5f32, 0.5, 0.5]];
        let perturbation = perturbation(&config);
        let cloud = activate(
            &RawOutput {
                features: &features,
                anchors: &anchors,
                perturbation: &perturbation,
                base_offset_scale: 0.0,
            },
            &config,
        )
        .expect("activates");

        assert_eq!(cloud.len(), config.num_gaussians);
        // The anchor sits at the cube's centre, so a zero offset is the origin.
        for position in &cloud.xyz {
            for axis in position {
                assert!(axis.abs() < 1e-6, "position {axis}");
            }
        }
        // Opacity is biased to 0.1, not 0.
        for opacity in &cloud.opacity {
            assert!((opacity - 0.1).abs() < 1e-6, "opacity {opacity}");
        }
        // Scale is softplus(0.004) floored in quadrature by the kernel size.
        let expected = (0.004f32.exp().ln_1p().powi(2) + 0.0009f32.powi(2)).sqrt();
        for scale in &cloud.scaling {
            for axis in scale {
                assert!(
                    (axis - expected).abs() < 1e-6,
                    "scale {axis}, want {expected}"
                );
            }
        }
        // A zero raw rotation plus the identity offset is identity.
        for rotation in &cloud.rotation {
            assert!((rotation[0] - 1.0).abs() < 1e-6, "{rotation:?}");
            assert!(rotation[1].abs() < 1e-6);
        }
    }

    #[test]
    fn the_offset_cannot_leave_its_cell() {
        let config = RepresentationConfig::idol();
        // A raw offset far outside the tanh's linear range still saturates at
        // the reach, and the reach is half the perturbation size times the
        // activated offset scale: softplus(0) = ln 2.
        let mut features = raw_zero(1);
        for gaussian in 0..config.num_gaussians {
            let base = gaussian * OUTPUTS_PER_GAUSSIAN;
            features[base] = 1000.0;
        }
        let anchors = [[0.5f32, 0.5, 0.5]];
        let perturbation = perturbation(&config);
        let cloud = activate(
            &RawOutput {
                features: &features,
                anchors: &anchors,
                perturbation: &perturbation,
                base_offset_scale: 0.0,
            },
            &config,
        )
        .expect("activates");
        let reach = 0.5 * config.perturbe_size * 2.0f32.ln();
        for position in &cloud.xyz {
            assert!(
                position[0] <= reach + 1e-5,
                "offset {} exceeds the reach {reach}",
                position[0]
            );
            assert!(position[0] > 0.0);
        }
    }

    #[test]
    fn the_rotation_is_normalised() {
        let config = RepresentationConfig::idol();
        let mut features = raw_zero(1);
        for gaussian in 0..config.num_gaussians {
            let base = gaussian * OUTPUTS_PER_GAUSSIAN;
            features[base + 9] = 3.0;
            features[base + 10] = 4.0;
        }
        let anchors = [[0.5f32, 0.5, 0.5]];
        let perturbation = perturbation(&config);
        let cloud = activate(
            &RawOutput {
                features: &features,
                anchors: &anchors,
                perturbation: &perturbation,
                base_offset_scale: 0.0,
            },
            &config,
        )
        .expect("activates");
        for rotation in &cloud.rotation {
            let norm = rotation.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "norm {norm}");
        }
    }

    #[test]
    fn a_saturated_opacity_stays_inside_the_unit_interval() {
        let config = RepresentationConfig::idol();
        let mut features = raw_zero(1);
        for gaussian in 0..config.num_gaussians {
            features[gaussian * OUTPUTS_PER_GAUSSIAN + 13] = 100.0;
        }
        let anchors = [[0.5f32, 0.5, 0.5]];
        let perturbation = perturbation(&config);
        let cloud = activate(
            &RawOutput {
                features: &features,
                anchors: &anchors,
                perturbation: &perturbation,
                base_offset_scale: 0.0,
            },
            &config,
        )
        .expect("activates");
        for opacity in &cloud.opacity {
            assert!(*opacity <= 1.0 && *opacity > 0.99, "opacity {opacity}");
        }
    }

    #[test]
    fn each_anchor_gets_its_own_gaussians() {
        let config = RepresentationConfig::idol();
        let features = raw_zero(2);
        let anchors = [[0.0f32, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let perturbation = perturbation(&config);
        let cloud = activate(
            &RawOutput {
                features: &features,
                anchors: &anchors,
                perturbation: &perturbation,
                base_offset_scale: 0.0,
            },
            &config,
        )
        .expect("activates");
        assert_eq!(cloud.len(), 2 * config.num_gaussians);
        // The first anchor is at the cube's low corner, the second at the high.
        assert!((cloud.xyz[0][0] + 0.5).abs() < 1e-6);
        assert!((cloud.xyz[config.num_gaussians][0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_short_output_is_refused() {
        let config = RepresentationConfig::idol();
        let features = raw_zero(1);
        let short = &features[..features.len() - 1];
        let anchors = [[0.5f32, 0.5, 0.5]];
        let perturbation = perturbation(&config);
        assert!(matches!(
            activate(
                &RawOutput {
                    features: short,
                    anchors: &anchors,
                    perturbation: &perturbation,
                    base_offset_scale: 0.0,
                },
                &config
            ),
            Err(GaussianError::Shape { anchors: 1, .. })
        ));
    }

    #[test]
    fn a_short_perturbation_table_is_refused() {
        let config = RepresentationConfig::idol();
        let features = raw_zero(1);
        let anchors = [[0.5f32, 0.5, 0.5]];
        let perturbation = vec![[0.0f32; 3]; config.num_gaussians - 1];
        assert!(matches!(
            activate(
                &RawOutput {
                    features: &features,
                    anchors: &anchors,
                    perturbation: &perturbation,
                    base_offset_scale: 0.0,
                },
                &config
            ),
            Err(GaussianError::Perturbation { .. })
        ));
    }
}

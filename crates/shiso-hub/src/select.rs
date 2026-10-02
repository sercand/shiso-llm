//! Hardware-aware variant selection (§5.4).

use crate::error::HubError;
use crate::host::{Backend, HostProfile};
use crate::recipe::{Recipe, Variant, VariantKey};
use std::fmt::Write;

/// Result of selecting a variant for a host.
#[derive(Debug, Clone)]
pub struct SelectedVariant<'a> {
    /// The chosen variant from the recipe.
    pub variant: &'a Variant,
    /// Human-readable explanation of why this variant was chosen.
    pub explanation: String,
    /// Warning if the user explicitly requested a variant that exceeds host RAM.
    pub warning: Option<String>,
}

/// Selects the best variant for `host` from `recipe`.
///
/// If `explicit_key` is provided, selects that variant if hard requirements are met.
/// Otherwise, filters and ranks variants according to host memory and backend availability.
///
/// # Errors
/// Returns [`HubError::VariantNotFound`] if an explicit variant is not found in the recipe,
/// or [`HubError::NoCompatibleVariant`] if no compatible variant fits the host.
pub fn select_variant<'a>(
    recipe: &'a Recipe,
    host: &HostProfile,
    explicit_key: Option<&VariantKey>,
) -> Result<SelectedVariant<'a>, HubError> {
    if let Some(target_key) = explicit_key {
        let variant = recipe
            .variants
            .iter()
            .find(|v| &v.key == target_key)
            .ok_or_else(|| HubError::VariantNotFound {
                model: recipe.name.to_string(),
                variant: target_key.to_string(),
            })?;

        // Hard checks: backend and CUDA SM
        check_hard_constraints(variant, host, recipe.name.as_ref())?;

        let warning = if variant.requirements.min_ram_bytes > host.ram_bytes {
            Some(format!(
                "warning: variant '{}' requires {} bytes of RAM, but host only has {} bytes (overcommitting as requested)",
                variant.key.as_str(),
                variant.requirements.min_ram_bytes,
                host.ram_bytes
            ))
        } else {
            None
        };

        return Ok(SelectedVariant {
            variant,
            explanation: format!("explicitly selected variant '{}'", variant.key.as_str()),
            warning,
        });
    }

    // Auto-selection (§5.4)
    let mut rejected = Vec::new();
    let mut compatible = Vec::new();

    for variant in &recipe.variants {
        match check_candidate_variant(variant, host) {
            Ok(()) => compatible.push(variant),
            Err(reason) => rejected.push((variant, reason)),
        }
    }

    if compatible.is_empty() {
        let mut msg = String::new();
        for (v, reason) in rejected {
            let _ = writeln!(msg, "  - {}: {}", v.key.as_str(), reason);
        }
        return Err(HubError::NoCompatibleVariant {
            model: recipe.name.to_string(),
            reasons: msg,
        });
    }

    // Pick first in recipe order whose recommended_ram_bytes fits; else first remaining
    for &candidate in &compatible {
        let rec = candidate
            .requirements
            .recommended_ram_bytes
            .unwrap_or(candidate.requirements.min_ram_bytes);
        if rec <= host.ram_bytes {
            return Ok(SelectedVariant {
                variant: candidate,
                explanation: format!(
                    "selected variant '{}' (backend '{}', quant '{}') fitting recommended RAM ({} MB)",
                    candidate.key.as_str(),
                    candidate.backend,
                    candidate.quant,
                    rec / (1024 * 1024)
                ),
                warning: None,
            });
        }
    }

    let first = compatible[0];
    Ok(SelectedVariant {
        variant: first,
        explanation: format!(
            "selected variant '{}' (backend '{}', quant '{}') as best available fit",
            first.key.as_str(),
            first.backend,
            first.quant
        ),
        warning: None,
    })
}

fn check_hard_constraints(
    variant: &Variant,
    host: &HostProfile,
    model_name: &str,
) -> Result<(), HubError> {
    let target_backend = match variant.backend.as_str() {
        "metal" => Backend::Metal,
        "cuda" => Backend::Cuda,
        other => {
            return Err(HubError::NoCompatibleVariant {
                model: model_name.to_string(),
                reasons: format!("unknown backend '{other}'"),
            });
        }
    };

    if !host.backends.contains(&target_backend) {
        return Err(HubError::NoCompatibleVariant {
            model: model_name.to_string(),
            reasons: format!(
                "backend '{}' is not supported by this build of shiso",
                variant.backend
            ),
        });
    }

    if target_backend == Backend::Cuda {
        let cuda = host
            .cuda
            .as_ref()
            .ok_or_else(|| HubError::NoCompatibleVariant {
                model: model_name.to_string(),
                reasons: "CUDA backend is enabled but no CUDA device was detected".to_string(),
            })?;

        if let Some(min_sm) = variant.requires.as_ref().and_then(|r| r.min_cuda_sm)
            && cuda.sm < min_sm
        {
            return Err(HubError::NoCompatibleVariant {
                model: model_name.to_string(),
                reasons: format!(
                    "requires CUDA compute capability sm{min_sm} or higher, but host device is sm{}",
                    cuda.sm
                ),
            });
        }
    }

    Ok(())
}

fn check_candidate_variant(variant: &Variant, host: &HostProfile) -> Result<(), String> {
    let target_backend = match variant.backend.as_str() {
        "metal" => Backend::Metal,
        "cuda" => Backend::Cuda,
        other => return Err(format!("unknown backend '{other}'")),
    };

    if !host.backends.contains(&target_backend) {
        return Err(format!(
            "backend '{}' not available in this build",
            variant.backend
        ));
    }

    if target_backend == Backend::Cuda {
        let Some(cuda) = &host.cuda else {
            return Err("no CUDA device detected".to_string());
        };

        if let Some(min_sm) = variant.requires.as_ref().and_then(|r| r.min_cuda_sm)
            && cuda.sm < min_sm
        {
            return Err(format!("requires CUDA sm{min_sm}, host has sm{}", cuda.sm));
        }

        if let Some(min_vram) = variant.requirements.min_vram_bytes
            && cuda.vram_bytes < min_vram
        {
            return Err(format!(
                "requires {} MB VRAM, host has {} MB",
                min_vram / (1024 * 1024),
                cuda.vram_bytes / (1024 * 1024)
            ));
        }
    }

    if variant.requirements.min_ram_bytes > host.ram_bytes {
        return Err(format!(
            "requires {} MB RAM, host has {} MB",
            variant.requirements.min_ram_bytes / (1024 * 1024),
            host.ram_bytes / (1024 * 1024)
        ));
    }

    Ok(())
}

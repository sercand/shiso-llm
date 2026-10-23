//! Hardware-aware variant selection (§5.4).

use crate::error::HubError;
use crate::host::{Backend, HostProfile, MetalInfo, ServingDevice};
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
                    "selected variant '{}' ({}, quant '{}') fitting recommended RAM ({} MB)",
                    candidate.key.as_str(),
                    where_it_runs(candidate, host),
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
            "selected variant '{}' ({}, quant '{}') as best available fit",
            first.key.as_str(),
            where_it_runs(first, host),
            first.quant
        ),
        warning: None,
    })
}

/// The explanation's account of where the variant runs.
fn where_it_runs(variant: &Variant, host: &HostProfile) -> String {
    match host.device {
        ServingDevice::Cpu => "on the CPU device".to_owned(),
        ServingDevice::Gpu => format!("backends {}", backend_list(variant)),
    }
}

fn backend_list(variant: &Variant) -> String {
    let names: Vec<&str> = variant.backends.iter().map(|b| b.as_str()).collect();
    names.join("/")
}

/// The backend `variant` would run on here: the first one it names that this
/// build has, so the author's preference order decides between two the host
/// offers.
///
/// # Errors
/// The reason no named backend is available, for the caller to wrap.
pub fn host_backend(variant: &Variant, host: &HostProfile) -> Result<Backend, String> {
    variant
        .backends
        .iter()
        .copied()
        .find(|b| host.backends.contains(b))
        .ok_or_else(|| {
            format!(
                "backends {} are not supported by this build of shiso",
                backend_list(variant)
            )
        })
}

/// Whether `variant` can run on `host` at all: a backend the host has, or any
/// variant on the CPU device. The coarse answer `shiso list` prints; the
/// selection itself also weighs memory.
#[must_use]
pub fn fits_host(variant: &Variant, host: &HostProfile) -> bool {
    backends_fit_host(&variant.backends, host)
}

/// [`fits_host`] for a registry index row, which carries a variant's backends
/// but not its recipe.
#[must_use]
pub fn backends_fit_host(backends: &[Backend], host: &HostProfile) -> bool {
    host.device == ServingDevice::Cpu || backends.iter().any(|b| host.backends.contains(b))
}

fn check_hard_constraints(
    variant: &Variant,
    host: &HostProfile,
    model_name: &str,
) -> Result<(), HubError> {
    // The CPU runs every variant's weights; there is no backend, SM or family
    // to check, and RAM is only a warning for an explicit key.
    if host.device == ServingDevice::Cpu {
        return Ok(());
    }
    let target_backend =
        host_backend(variant, host).map_err(|reasons| HubError::NoCompatibleVariant {
            model: model_name.to_string(),
            reasons,
        })?;

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

    if target_backend == Backend::Metal {
        let metal = host
            .metal
            .as_ref()
            .ok_or_else(|| HubError::NoCompatibleVariant {
                model: model_name.to_string(),
                reasons: "Metal backend is enabled but no Metal device was detected".to_string(),
            })?;
        if let Some(reason) = metal_family_shortfall(variant, metal) {
            return Err(HubError::NoCompatibleVariant {
                model: model_name.to_string(),
                reasons: reason,
            });
        }
    }

    Ok(())
}

/// Why the device's Apple GPU family is too old for `variant`, if it is.
fn metal_family_shortfall(variant: &Variant, metal: &MetalInfo) -> Option<String> {
    let min = variant.requires.as_ref().and_then(|r| r.min_metal_family)?;
    (metal.family < min).then(|| {
        format!(
            "requires Apple GPU family {min} ({}) or newer, but {} is family {}",
            apple_family_label(min),
            metal.name,
            metal.family
        )
    })
}

/// The first chip generation of an Apple GPU family, for messages.
#[must_use]
pub const fn apple_family_label(family: u32) -> &'static str {
    match family {
        0..=6 => "pre-M1",
        7 => "M1",
        8 => "M2",
        9 => "M3/M4",
        10 => "M5",
        _ => "newer than M5",
    }
}

fn check_candidate_variant(variant: &Variant, host: &HostProfile) -> Result<(), String> {
    if host.device == ServingDevice::Cpu {
        return check_ram(variant, host);
    }
    let target_backend = host_backend(variant, host)?;

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

    if target_backend == Backend::Metal {
        let Some(metal) = &host.metal else {
            return Err("no Metal device detected".to_string());
        };
        if let Some(reason) = metal_family_shortfall(variant, metal) {
            return Err(reason);
        }
        if let Some(min_set) = variant.requirements.min_vram_bytes
            && metal.working_set_bytes < min_set
        {
            return Err(format!(
                "requires {} MB GPU working set, {} recommends {} MB",
                min_set / (1024 * 1024),
                metal.name,
                metal.working_set_bytes / (1024 * 1024)
            ));
        }
    }

    check_ram(variant, host)
}

fn check_ram(variant: &Variant, host: &HostProfile) -> Result<(), String> {
    if variant.requirements.min_ram_bytes > host.ram_bytes {
        return Err(format!(
            "requires {} MB RAM, host has {} MB",
            variant.requirements.min_ram_bytes / (1024 * 1024),
            host.ram_bytes / (1024 * 1024)
        ));
    }

    Ok(())
}

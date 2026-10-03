//! Manifest and recipe validation rules.
//!
//! Enforces structural, component, role dependency, and path-safety rules
//! for both client parsing and strict maintainer validation.

use std::collections::{BTreeSet, HashSet};

use crate::error::HubError;
use crate::recipe::{CURRENT_SCHEMA_VERSION, Recipe, Role, Variant};

/// External family facts passed to validator to verify a recipe matches engine capabilities.
#[derive(Debug, Clone, Copy)]
pub struct FamilyValidationFacts<'a> {
    /// Family name as `shiso-models` spells it.
    pub name: &'a str,
}

/// Validation mode: client (tolerant of additive changes) or strict (denies typos).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationMode {
    /// Client mode: unknown fields tolerated, schema version check enforced.
    Client,
    /// Strict mode: unknown fields denied, all cross-checks enforced.
    Strict,
}

/// Validates a recipe against the specification rules (§3).
///
/// # Errors
/// Returns [`HubError::Validation`] on rule violations, or [`HubError::UnsupportedSchemaVersion`]
/// if the manifest schema version exceeds what this build supports.
pub fn validate_recipe(recipe: &Recipe, mode: ValidationMode) -> Result<(), HubError> {
    validate_metadata(recipe)?;
    validate_variants(recipe, mode)?;
    Ok(())
}

fn validate_metadata(recipe: &Recipe) -> Result<(), HubError> {
    if recipe.schema_version > CURRENT_SCHEMA_VERSION {
        return Err(HubError::UnsupportedSchemaVersion {
            found: recipe.schema_version,
            supported: CURRENT_SCHEMA_VERSION,
        });
    }
    if recipe.schema_version == 0 {
        return Err(HubError::Validation {
            reason: "schema_version must be >= 1".to_string(),
        });
    }

    let mut names = HashSet::new();
    names.insert(recipe.name.as_str());
    for alias in &recipe.aliases {
        if !names.insert(alias.as_str()) {
            return Err(HubError::Validation {
                reason: format!("duplicate name or alias '{}' in recipe", alias.as_str()),
            });
        }
    }

    if recipe.description.trim().is_empty() {
        return Err(HubError::Validation {
            reason: "recipe description cannot be empty".to_string(),
        });
    }
    if recipe.family.trim().is_empty() {
        return Err(HubError::Validation {
            reason: "recipe family cannot be empty".to_string(),
        });
    }

    validate_family_facts(recipe)?;

    Ok(())
}

fn validate_family_facts(recipe: &Recipe) -> Result<(), HubError> {
    if let Some(default_effort) = &recipe.features.default_reasoning_effort
        && !recipe.features.reasoning_efforts.contains(default_effort)
    {
        return Err(HubError::Validation {
            reason: format!(
                "default reasoning effort '{default_effort}' is not in declared reasoning_efforts"
            ),
        });
    }

    Ok(())
}

fn validate_variants(recipe: &Recipe, mode: ValidationMode) -> Result<(), HubError> {
    if recipe.variants.is_empty() {
        return Err(HubError::Validation {
            reason: "recipe must declare at least one variant".to_string(),
        });
    }

    let mut variant_keys = BTreeSet::new();
    for variant in &recipe.variants {
        if !variant_keys.insert(variant.key.as_str()) {
            return Err(HubError::Validation {
                reason: format!("duplicate variant key '{}' in recipe", variant.key.as_str()),
            });
        }
        validate_single_variant(variant, mode)?;
    }
    Ok(())
}

fn validate_single_variant(variant: &Variant, mode: ValidationMode) -> Result<(), HubError> {
    if variant.backends.is_empty() {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' must name at least one backend",
                variant.key.as_str()
            ),
        });
    }
    let mut seen = BTreeSet::new();
    for backend in &variant.backends {
        if !seen.insert(*backend) {
            return Err(HubError::Validation {
                reason: format!(
                    "variant '{}' names backend '{backend}' twice",
                    variant.key.as_str()
                ),
            });
        }
    }

    if variant.requirements.min_ram_bytes == 0 {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' min_ram_bytes must be greater than zero",
                variant.key.as_str()
            ),
        });
    }

    validate_role_dependencies(variant)?;
    validate_components(variant, mode)?;
    Ok(())
}

fn validate_role_dependencies(variant: &Variant) -> Result<(), HubError> {
    // Roles a bound file supplies count exactly as a component's would: the
    // server cannot tell the two apart, so neither can these rules.
    let roles = variant.role_paths()?;
    let has_base = roles.contains_key(&Role::Base);
    let has_asr = roles.contains_key(&Role::Asr);
    let has_draft = roles.contains_key(&Role::Draft);
    let has_draft_vocab = roles.contains_key(&Role::DraftVocab);
    let has_asr_mmproj = roles.contains_key(&Role::AsrMmproj);
    let has_diarization = roles.contains_key(&Role::Diarization);

    // `--config` and `--ngram` name files; a directory there would be read as
    // one and fail at boot, so the recipe has to say which file it is.
    for role in [Role::Config, Role::Ngram] {
        if let Some(place) = roles.get(&role)
            && place.file.is_none()
        {
            return Err(HubError::Validation {
                reason: format!(
                    "variant '{}' role '{role}' must name its file with 'entry' or 'binds'",
                    variant.key.as_str()
                ),
            });
        }
    }

    if !has_base && !has_asr && !roles.contains_key(&Role::Image) {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' must contain a 'base' component unless it has 'asr' or 'image'",
                variant.key.as_str()
            ),
        });
    }

    if has_draft {
        let spec_k = variant
            .server_defaults
            .as_ref()
            .and_then(|sd| sd.spec_k)
            .unwrap_or(0);
        if spec_k == 0 {
            return Err(HubError::Validation {
                reason: format!(
                    "variant '{}' supplies 'draft' but server_defaults.spec_k is 0 or missing",
                    variant.key.as_str()
                ),
            });
        }
    }

    if has_draft_vocab && !has_draft {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' supplies 'draft_vocab' which requires 'draft'",
                variant.key.as_str()
            ),
        });
    }

    if has_asr_mmproj && !has_asr {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' supplies 'asr_mmproj' which requires 'asr'",
                variant.key.as_str()
            ),
        });
    }

    if has_diarization && !has_asr {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' supplies 'diarization' which requires 'asr'",
                variant.key.as_str()
            ),
        });
    }

    Ok(())
}

fn validate_components(variant: &Variant, mode: ValidationMode) -> Result<(), HubError> {
    let mut total_disk_bytes: u64 = 0;
    for (role, comp) in &variant.components {
        if comp.provider != "huggingface" {
            return Err(HubError::Validation {
                reason: format!(
                    "variant '{}' role '{}' provider must be 'huggingface', got '{}'",
                    variant.key.as_str(),
                    role.as_str(),
                    comp.provider
                ),
            });
        }

        if !comp.repo.contains('/') || comp.repo.starts_with('/') || comp.repo.ends_with('/') {
            return Err(HubError::Validation {
                reason: format!(
                    "variant '{}' role '{}' repo '{}' must be in 'owner/name' format",
                    variant.key.as_str(),
                    role.as_str(),
                    comp.repo
                ),
            });
        }

        if comp.files.is_empty() {
            return Err(HubError::Validation {
                reason: format!(
                    "variant '{}' role '{}' must declare at least one file",
                    variant.key.as_str(),
                    role.as_str()
                ),
            });
        }

        let mut file_names = HashSet::new();
        for file in &comp.files {
            validate_file_path(&file.name)?;
            if !file_names.insert(&file.name) {
                return Err(HubError::Validation {
                    reason: format!(
                        "variant '{}' role '{}' declares duplicate file '{}'",
                        variant.key.as_str(),
                        role.as_str(),
                        file.name
                    ),
                });
            }
            total_disk_bytes = total_disk_bytes.saturating_add(file.size);
        }

        if let Some(entry) = &comp.entry
            && !file_names.contains(entry)
        {
            return Err(HubError::Validation {
                reason: format!(
                    "variant '{}' role '{}' entry '{entry}' is not present in its files list",
                    variant.key.as_str(),
                    role.as_str()
                ),
            });
        }
    }

    if mode == ValidationMode::Strict && variant.requirements.disk_bytes != total_disk_bytes {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' declared disk_bytes ({}) does not match sum of file sizes ({})",
                variant.key.as_str(),
                variant.requirements.disk_bytes,
                total_disk_bytes
            ),
        });
    }

    Ok(())
}

/// Checks that a file path is safe: no `..`, no leading `/`, no empty segments.
fn validate_file_path(path: &str) -> Result<(), HubError> {
    if path.is_empty() || path.starts_with('/') || path.ends_with('/') {
        return Err(HubError::Validation {
            reason: format!(
                "invalid file path '{path}': cannot be empty or have leading/trailing slash"
            ),
        });
    }
    for segment in path.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(HubError::Validation {
                reason: format!("invalid file path '{path}': contains unsafe segment '{segment}'"),
            });
        }
    }
    Ok(())
}

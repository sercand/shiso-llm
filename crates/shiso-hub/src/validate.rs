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
    /// Roles accepted by this family.
    pub accepted_roles: &'a [&'a str],
    /// API tasks served by this family.
    pub tasks: &'a [&'a str],
    /// Whether tool calling is implemented for this family.
    pub tools: bool,
    /// Reasoning effort labels implemented for this family.
    pub reasoning_efforts: &'a [&'a str],
    /// Whether structured output is implemented for this family.
    pub structured_output: bool,
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
pub fn validate_recipe(
    recipe: &Recipe,
    mode: ValidationMode,
    family_facts: Option<&[FamilyValidationFacts<'_>]>,
) -> Result<(), HubError> {
    validate_metadata(recipe, family_facts)?;
    validate_variants(recipe, mode, family_facts)?;
    Ok(())
}

fn validate_metadata(
    recipe: &Recipe,
    family_facts: Option<&[FamilyValidationFacts<'_>]>,
) -> Result<(), HubError> {
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

    if let Some(facts_list) = family_facts {
        validate_family_facts(recipe, facts_list)?;
    }

    Ok(())
}

fn validate_family_facts(
    recipe: &Recipe,
    facts_list: &[FamilyValidationFacts<'_>],
) -> Result<(), HubError> {
    let facts = facts_list
        .iter()
        .find(|f| f.name == recipe.family)
        .ok_or_else(|| HubError::Validation {
            reason: format!("unknown model family '{}'", recipe.family),
        })?;

    for task in &recipe.tasks {
        if !facts.tasks.contains(&task.as_str()) {
            return Err(HubError::Validation {
                reason: format!(
                    "family '{}' does not implement task '{}'",
                    recipe.family,
                    task.as_str()
                ),
            });
        }
    }

    if recipe.features.tools && !facts.tools {
        return Err(HubError::Validation {
            reason: format!("family '{}' does not implement tool calling", recipe.family),
        });
    }
    if recipe.features.structured_output && !facts.structured_output {
        return Err(HubError::Validation {
            reason: format!(
                "family '{}' does not implement structured output",
                recipe.family
            ),
        });
    }
    for effort in &recipe.features.reasoning_efforts {
        if !facts.reasoning_efforts.contains(&effort.as_str()) {
            return Err(HubError::Validation {
                reason: format!(
                    "family '{}' does not implement reasoning effort '{effort}'",
                    recipe.family
                ),
            });
        }
    }
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

fn validate_variants(
    recipe: &Recipe,
    mode: ValidationMode,
    family_facts: Option<&[FamilyValidationFacts<'_>]>,
) -> Result<(), HubError> {
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
        validate_single_variant(recipe, variant, mode, family_facts)?;
    }
    Ok(())
}

fn validate_single_variant(
    recipe: &Recipe,
    variant: &Variant,
    mode: ValidationMode,
    family_facts: Option<&[FamilyValidationFacts<'_>]>,
) -> Result<(), HubError> {
    if variant.backend != "metal" && variant.backend != "cuda" {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' backend must be 'metal' or 'cuda', got '{}'",
                variant.key.as_str(),
                variant.backend
            ),
        });
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

    if let Some(facts_list) = family_facts
        && let Some(facts) = facts_list.iter().find(|f| f.name == recipe.family)
    {
        for role in variant.components.keys() {
            if !facts.accepted_roles.contains(&role.as_str()) {
                return Err(HubError::Validation {
                    reason: format!(
                        "variant '{}' uses role '{}' which family '{}' does not accept",
                        variant.key.as_str(),
                        role.as_str(),
                        recipe.family
                    ),
                });
            }
        }
    }

    validate_components(variant, mode)?;
    Ok(())
}

fn validate_role_dependencies(variant: &Variant) -> Result<(), HubError> {
    let has_base = variant.components.contains_key(&Role::Base);
    let has_asr = variant.components.contains_key(&Role::Asr);
    let has_draft = variant.components.contains_key(&Role::Draft);
    let has_draft_vocab = variant.components.contains_key(&Role::DraftVocab);
    let has_asr_mmproj = variant.components.contains_key(&Role::AsrMmproj);
    let has_diarization = variant.components.contains_key(&Role::Diarization);

    if !has_base && !has_asr && !variant.components.contains_key(&Role::Image) {
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
                    "variant '{}' has 'draft' component but server_defaults.spec_k is 0 or missing",
                    variant.key.as_str()
                ),
            });
        }
    }

    if has_draft_vocab && !has_draft {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' has 'draft_vocab' component which requires 'draft'",
                variant.key.as_str()
            ),
        });
    }

    if has_asr_mmproj && !has_asr {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' has 'asr_mmproj' component which requires 'asr'",
                variant.key.as_str()
            ),
        });
    }

    if has_diarization && !has_asr {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' has 'diarization' component which requires 'asr'",
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

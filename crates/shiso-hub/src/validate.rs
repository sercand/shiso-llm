//! Manifest and recipe validation rules.
//!
//! Enforces structural, component, role dependency, and path-safety rules
//! for both client parsing and strict maintainer validation.

use std::collections::{BTreeSet, HashSet};

use crate::error::HubError;
use crate::recipe::{
    CURRENT_SCHEMA_VERSION, Component, HUB_FILE_MAX_BYTES, Provider, Recipe, Role, Variant,
    hub_file_name_ok,
};

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
    let uses_hub = recipe
        .variants
        .iter()
        .flat_map(|v| v.components.values())
        .any(|c| c.provider == Provider::Shiso);
    if uses_hub && recipe.min_shiso_version < hub_files_min_version() {
        return Err(HubError::Validation {
            reason: format!(
                "recipe '{}' uses hub-hosted files, which need min_shiso_version >= {}",
                recipe.name.as_str(),
                hub_files_min_version()
            ),
        });
    }
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
    validate_info(recipe)?;
    validate_quality(recipe)?;

    Ok(())
}

fn invalid(reason: String) -> HubError {
    HubError::Validation { reason }
}

/// A link the hub site renders must be `https://`: it is shown to every
/// visitor, and a `javascript:` or plain-HTTP link there is the registry's
/// doing, not the visitor's.
fn require_https(what: &str, url: &str) -> Result<(), HubError> {
    let ok = url
        .strip_prefix("https://")
        .is_some_and(|rest| !rest.is_empty() && !rest.contains(char::is_whitespace));
    if ok {
        Ok(())
    } else {
        Err(invalid(format!("{what} '{url}' must be an https:// URL")))
    }
}

fn validate_info(recipe: &Recipe) -> Result<(), HubError> {
    let Some(info) = &recipe.info else {
        return Ok(());
    };
    for (what, value) in [
        ("info.publisher", &info.publisher),
        ("info.license", &info.license),
    ] {
        if value.as_ref().is_some_and(|v| v.trim().is_empty()) {
            return Err(invalid(format!("{what} cannot be empty when present")));
        }
    }
    if let Some(homepage) = &info.homepage {
        require_https("info.homepage", homepage)?;
    }
    if let Some(card) = &info.card {
        if card.provider != "huggingface" {
            return Err(invalid(format!(
                "info.card provider '{}' is not supported; only 'huggingface' is",
                card.provider
            )));
        }
        if card.repo.split('/').count() != 2
            || card
                .repo
                .split('/')
                .any(|seg| seg.is_empty() || seg == "." || seg == "..")
        {
            return Err(invalid(format!(
                "info.card repo '{}' must be 'owner/name'",
                card.repo
            )));
        }
    }
    for (what, value) in [
        ("info.parameters", info.parameters),
        ("info.active_parameters", info.active_parameters),
        ("info.context_length", info.context_length),
    ] {
        if value == Some(0) {
            return Err(invalid(format!("{what} must be greater than zero")));
        }
    }
    if let (Some(total), Some(active)) = (info.parameters, info.active_parameters)
        && active > total
    {
        return Err(invalid(format!(
            "info.active_parameters ({active}) exceeds info.parameters ({total})"
        )));
    }
    Ok(())
}

/// `YYYY-MM-DD` with a plausible month and day; a score without a readable
/// date cannot be told apart from a stale one.
fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let num = |r: std::ops::Range<usize>| s.get(r).and_then(|p| p.parse::<u32>().ok());
    matches!(
        (num(0..4), num(5..7), num(8..10)),
        (Some(_), Some(1..=12), Some(1..=31))
    )
}

fn validate_quality(recipe: &Recipe) -> Result<(), HubError> {
    use crate::recipe::QualityScope;
    for (i, q) in recipe.quality.iter().enumerate() {
        let at = format!("quality[{i}]");
        if q.source.trim().is_empty() || q.metric.trim().is_empty() {
            return Err(invalid(format!("{at} needs a non-empty source and metric")));
        }
        if !q.value.is_finite() {
            return Err(invalid(format!("{at} value must be a finite number")));
        }
        if let Some(max) = q.max
            && (!max.is_finite() || max <= 0.0 || q.value > max)
        {
            return Err(invalid(format!(
                "{at} max ({max}) must be positive and at least the value ({})",
                q.value
            )));
        }
        require_https(&format!("{at} url"), &q.url)?;
        if !is_iso_date(&q.as_of) {
            return Err(invalid(format!(
                "{at} as_of '{}' must be a YYYY-MM-DD date",
                q.as_of
            )));
        }
        match (q.scope, &q.variant) {
            (QualityScope::BaseModel, None) => {}
            (QualityScope::BaseModel, Some(key)) => {
                return Err(invalid(format!(
                    "{at} is scoped to the base model but names variant '{key}'"
                )));
            }
            (QualityScope::Variant, None) => {
                return Err(invalid(format!(
                    "{at} is scoped to a variant but names none"
                )));
            }
            (QualityScope::Variant, Some(key)) => {
                if !recipe.variants.iter().any(|v| v.key == *key) {
                    return Err(invalid(format!("{at} names unknown variant '{key}'")));
                }
            }
        }
    }
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
    let has_image = roles.contains_key(&Role::Image);

    // `--config` and `--ngram` name files; a directory there would be read as
    // one and fail at boot, so the recipe has to say which file it is.
    for role in [Role::Config, Role::Ngram, Role::DraftVocab] {
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

    // `--image-assets-dir` names a directory; a file there finds nothing.
    if let Some(place) = roles.get(&Role::ImageAssets)
        && place.file.is_some()
    {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' role 'image_assets' must be a directory: no 'entry' or 'binds'",
                variant.key.as_str()
            ),
        });
    }

    if !has_base && !has_asr && !has_image {
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

    let speculates = variant
        .server_defaults
        .as_ref()
        .and_then(|sd| sd.spec_k)
        .unwrap_or(0)
        > 0;
    if has_draft_vocab && !has_draft && !speculates {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' supplies 'draft_vocab' but has no 'draft' and server_defaults.spec_k is 0 or missing",
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

    if roles.contains_key(&Role::ImageAssets) && !has_image {
        return Err(HubError::Validation {
            reason: format!(
                "variant '{}' supplies 'image_assets' which requires 'image'",
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
        validate_provider(variant, *role, comp)?;

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

/// The rules that differ by where a component's files come from.
fn validate_provider(variant: &Variant, role: Role, comp: &Component) -> Result<(), HubError> {
    match comp.provider {
        Provider::HuggingFace => {
            let Some(repo) = comp.repo.as_deref() else {
                return Err(component_error(
                    variant,
                    role,
                    "a Hugging Face component needs 'repo'",
                ));
            };
            if comp.revision.is_none() {
                return Err(component_error(
                    variant,
                    role,
                    "a Hugging Face component needs 'revision'",
                ));
            }
            if !repo.contains('/') || repo.starts_with('/') || repo.ends_with('/') {
                return Err(component_error(
                    variant,
                    role,
                    &format!("repo '{repo}' must be in 'owner/name' format"),
                ));
            }
        }
        Provider::Shiso => {
            if comp.repo.is_some() || comp.revision.is_some() {
                return Err(component_error(
                    variant,
                    role,
                    "a hub component has no 'repo' or 'revision': its files live beside the manifest",
                ));
            }
            for file in &comp.files {
                if !hub_file_name_ok(&file.name) {
                    return Err(component_error(
                        variant,
                        role,
                        &format!(
                            "hub file '{}' must be one plain name, not index.json or benchmarks.json",
                            file.name
                        ),
                    ));
                }
                if file.size > HUB_FILE_MAX_BYTES {
                    return Err(component_error(
                        variant,
                        role,
                        &format!(
                            "hub file '{}' is {} bytes; the hub hosts at most {HUB_FILE_MAX_BYTES}",
                            file.name, file.size
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// A component-rule refusal naming the variant and role.
fn component_error(variant: &Variant, role: Role, why: &str) -> HubError {
    HubError::Validation {
        reason: format!(
            "variant '{}' role '{}': {why}",
            variant.key.as_str(),
            role.as_str()
        ),
    }
}

/// The first shiso release that can download hub-hosted files.
#[must_use]
pub fn hub_files_min_version() -> semver::Version {
    semver::Version::new(0, 2, 0)
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

//! The service class a caller asks for, on whichever surface it asked.
//!
//! # Why the vocabulary lives here and not beside one wire format
//!
//! The body field is the `OpenAI` family's, but the `X-Service-Tier` header is this
//! server's, and every surface reads it: a gateway that marks a batch of work
//! as flex should not have to know which API each request in the batch speaks.
//! So the words, the header and the one mapping into the neutral
//! [`ServiceClass`] are defined once, and every surface refuses a bad value
//! with the same wording.
//!
//! # A body field wins over the header
//!
//! The header is a fallback, not an override: a caller that named a class in
//! the body meant that one, and a gateway that adds a header must not silently
//! change it.

use axum::http::HeaderMap;
use serde::Deserialize;
use shiso_ir::ServiceClass;

use crate::error::AppError;

/// The header a caller may name a service class in.
pub const SERVICE_TIER_HEADER: &str = "x-service-tier";

/// The service class a caller asked to be served in.
///
/// Parsed rather than echoed as an opaque value, because a mistyped tier is a
/// caller that thinks it bought something it did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceTier {
    /// Whatever the server would do anyway.
    Auto,
    /// The ordinary class, named explicitly.
    Default,
    /// The class that trades latency for capacity.
    Flex,
}

impl ServiceTier {
    /// Parses the wire vocabulary, case insensitively.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "default" => Some(Self::Default),
            "flex" => Some(Self::Flex),
            _ => None,
        }
    }

    /// The spelling a response echoes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Default => "default",
            Self::Flex => "flex",
        }
    }

    /// How the scheduler serves this tier. `auto` is standard: this server has
    /// no spare capacity to hand out on its own initiative, so "whatever the
    /// server would do" is what it does for everybody.
    #[must_use]
    pub const fn class(self) -> ServiceClass {
        match self {
            Self::Auto | Self::Default => ServiceClass::Standard,
            Self::Flex => ServiceClass::Flex,
        }
    }
}

impl<'de> Deserialize<'de> for ServiceTier {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "service_tier must be one of \"auto\", \"default\" or \"flex\", not {raw:?}"
            ))
        })
    }
}

/// The class a request is served in: the body's tier when it named one, the
/// header's otherwise.
///
/// `Ok(None)` means neither named one, which is not a refusal: the request is
/// served in the standard class.
///
/// # Errors
///
/// A 400 when the header is present and is not a tier this server knows.
pub fn resolve(
    body: Option<ServiceTier>,
    headers: &HeaderMap,
) -> Result<Option<ServiceTier>, AppError> {
    match body {
        Some(tier) => Ok(Some(tier)),
        None => header_tier(headers),
    }
}

/// The service class a header named, when it named one this server knows.
///
/// `Ok(None)` means the header was absent, which is not a refusal: the request
/// simply did not ask for a tier and takes the server's default.
///
/// # Errors
///
/// A 400 when the header is not text or names a tier this server does not have.
pub fn header_tier(headers: &HeaderMap) -> Result<Option<ServiceTier>, AppError> {
    let Some(raw) = headers.get(SERVICE_TIER_HEADER) else {
        return Ok(None);
    };
    let value = raw.to_str().map_err(|_not_text| {
        AppError::bad_request(
            "the X-Service-Tier header is not text",
            Some("service_tier"),
            "invalid_request",
        )
    })?;
    ServiceTier::parse(value).map(Some).ok_or_else(|| {
        AppError::bad_request(
            format!(
                "the X-Service-Tier header must be \"auto\", \"default\" or \
                 \"flex\", not {value:?}"
            ),
            Some("service_tier"),
            "invalid_request",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{ServiceTier, header_tier, resolve};
    use axum::http::HeaderMap;
    use shiso_ir::ServiceClass;

    fn headers(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("x-service-tier", value.parse().expect("a header value"));
        headers
    }

    #[test]
    fn a_header_names_a_service_class_case_insensitively() {
        assert_eq!(
            header_tier(&headers("Flex")).expect("a known class"),
            Some(ServiceTier::Flex)
        );
        assert_eq!(header_tier(&HeaderMap::new()).expect("no header"), None);
    }

    #[test]
    fn a_header_naming_a_class_this_server_does_not_have_is_refused() {
        let error = header_tier(&headers("premium")).expect_err("an unknown class");
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn the_body_wins_over_the_header() {
        assert_eq!(
            resolve(Some(ServiceTier::Default), &headers("flex")).expect("a body tier"),
            Some(ServiceTier::Default)
        );
        assert_eq!(
            resolve(None, &headers("flex")).expect("a header tier"),
            Some(ServiceTier::Flex)
        );
    }

    #[test]
    fn only_flex_is_served_as_flex() {
        assert_eq!(ServiceTier::Flex.class(), ServiceClass::Flex);
        assert_eq!(ServiceTier::Auto.class(), ServiceClass::Standard);
        assert_eq!(ServiceTier::Default.class(), ServiceClass::Standard);
    }
}

//! What can stop a boot, as one typed error.
//!
//! The boot path touches a device, a checkpoint, three tokenizers and a handful
//! of worker threads, and every step has its own failure type. [`BootError`]
//! keeps each of them as the `source` of a short sentence saying which step it
//! was, so a caller can print the whole chain or match on the leading variant
//! without this crate depending on a catch-all error type.

use crate::model_factory::FactoryError;
use crate::options::ConfigError;
use crate::profile::ProfileError;
use crate::vision::TowerError;

/// Why the engine did not boot.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BootError {
    /// A setting the build or the checkpoint cannot honour, named by its flag.
    #[error("{0}")]
    Refused(String),
    /// A step failed; `context` says which one and `source` says why.
    #[error("{context}")]
    Step {
        /// The step, in the operator's words.
        context: String,
        /// The failure underneath.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The options did not validate.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The checkpoint could not be turned into a model.
    #[error(transparent)]
    Factory(#[from] FactoryError),
    /// The profile directory could not be opened.
    #[error(transparent)]
    Profile(#[from] ProfileError),
    /// The vision tower did not load or did not fit the checkpoint.
    #[error(transparent)]
    Vision(#[from] TowerError),
    /// The device or a model layer refused.
    #[error(transparent)]
    Core(#[from] shiso_core::CoreError),
}

/// Builds a [`BootError::Refused`] from a format string.
macro_rules! refuse {
    ($($arg:tt)*) => {
        $crate::boot_error::BootError::Refused(format!($($arg)*))
    };
}
pub(crate) use refuse;

/// Attaches the step that failed to an error on its way out of boot.
pub trait BootContext<T> {
    /// Wraps the error as a [`BootError::Step`] reading `context`.
    ///
    /// # Errors
    ///
    /// The original error, under `context`.
    fn context(self, context: impl Into<String>) -> Result<T, BootError>;

    /// As [`Self::context`], building the sentence only on failure.
    ///
    /// # Errors
    ///
    /// The original error, under the built sentence.
    fn with_context<C: Into<String>>(self, context: impl FnOnce() -> C) -> Result<T, BootError>;
}

impl<T, E> BootContext<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context(self, context: impl Into<String>) -> Result<T, BootError> {
        self.map_err(|source| BootError::Step {
            context: context.into(),
            source: Box::new(source),
        })
    }

    fn with_context<C: Into<String>>(self, context: impl FnOnce() -> C) -> Result<T, BootError> {
        self.map_err(|source| BootError::Step {
            context: context().into(),
            source: Box::new(source),
        })
    }
}

//! Why a decision could not be answered.

use thiserror::Error;

/// Why a decision could not be answered.
///
/// The variants say whose fault it was, so a caller can tell a request it
/// should fix from a machine it should not retry on.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DecideError {
    /// No device could run the model.
    #[error("no device can run the model: {0}")]
    NoDevice(String),
    /// The model could not be loaded.
    #[error("the model could not be loaded: {0}")]
    Load(String),
    /// The request is not one a decision can answer: no questions, a
    /// question without options, a temperature that is not positive.
    #[error("invalid decision request: {0}")]
    Request(String),
    /// The model's layout refused the schema, or laid out something other
    /// than what was asked.
    #[error("the decision prompt could not be laid out: {0}")]
    Layout(String),
    /// Whoever runs the model failed or answered something other than what
    /// was asked.
    #[error("the engine failed: {0}")]
    Engine(String),
    /// The registry has no model of that name.
    #[error("no model named '{0}' in the registry")]
    UnknownModel(String),
    /// Resolving, downloading or installing a model failed. The message names
    /// the step; files already downloaded are kept, so a retry resumes.
    #[error("the model could not be installed: {0}")]
    Download(String),
    /// The caller's progress callback asked to stop. Partial downloads are
    /// kept for the next attempt.
    #[error("the download was cancelled")]
    Cancelled,
    /// Reading a model file or a cache failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

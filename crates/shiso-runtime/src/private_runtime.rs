//! A tokio runtime an embedder-facing call owns privately: [`crate::Runtime`]'s
//! executor and the blocking pull's worker are both one of these.

use tokio::runtime::{Builder, Handle, Runtime};

/// A tokio runtime that is shut down without waiting when dropped, so dropping
/// it is legal on any thread and on every path, unwinding included (a plain
/// drop panics inside an async context, and again while unwinding it).
#[derive(Debug)]
pub(crate) struct PrivateRuntime {
    handle: Handle,
    runtime: Option<Runtime>,
}

impl PrivateRuntime {
    /// Builds the runtime `builder` describes.
    pub(crate) fn build(builder: &mut Builder) -> std::io::Result<Self> {
        let runtime = builder.build()?;
        Ok(Self {
            handle: runtime.handle().clone(),
            runtime: Some(runtime),
        })
    }

    /// The runtime's handle: to spawn onto, or to enter.
    pub(crate) const fn handle(&self) -> &Handle {
        &self.handle
    }
}

impl Drop for PrivateRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

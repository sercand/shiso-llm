//! What a tensor-name to role mapping needs from a model file.
//!
//! One trait, and it is the whole seam between "which container is this" and
//! "which weight is this". A container that can report its architecture and
//! answer `find` and `get` by name can be bound by any role mapping, and a role
//! mapping written against this trait never learns which file format it is
//! reading.
//!
//! The mappings themselves are not here. Role names - `attn_qkv`, `ssm_conv1d`,
//! `post_attention_norm` - are facts about one checkpoint family, and a reader
//! that knew them could not open a checkpoint from another. They live with the
//! crate that knows the family.

use crate::reader::{TensorDesc, TensorView};

/// What a tensor-name to role mapping needs from a model file.
///
/// Implemented by [`crate::reader::GgufFile`] and by any other container that
/// can describe named tensors, so the binding layer never learns which file
/// format it is reading.
pub trait TensorSource {
    /// Declared architecture, `None` when the file declared none.
    fn architecture(&self) -> Option<&str>;

    /// Integer metadata key under the architecture prefix.
    fn arch_u32(&self, suffix: &str) -> Option<u32>;

    /// Directory entry for `name`.
    fn find(&self, name: &str) -> Option<&TensorDesc>;

    /// Borrowed payload for `name`.
    fn get(&self, name: &str) -> Option<TensorView<'_>>;
}

impl TensorSource for crate::reader::GgufFile {
    fn architecture(&self) -> Option<&str> {
        self.metadata().architecture()
    }

    fn arch_u32(&self, suffix: &str) -> Option<u32> {
        self.metadata().arch_u32(suffix)
    }

    fn find(&self, name: &str) -> Option<&TensorDesc> {
        self.find(name)
    }

    fn get(&self, name: &str) -> Option<TensorView<'_>> {
        self.get(name)
    }
}

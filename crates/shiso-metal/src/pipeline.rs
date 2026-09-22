//! Pipeline cache: `KernelRef` in, resolved `MTLComputePipelineState` out.
//!
//! Specialization comes from the `KernelRef`'s constant table, never from a
//! suffix on the kernel name, so nothing here parses a string.

use std::collections::HashMap;
use std::fmt;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLFunctionConstantValues, MTLLibrary};
use shiso_core::{CoreError, CoreResult, FnConst, KernelRef};

use crate::device::MetalError;

/// Cache key: an entry point plus its specialization table.
///
/// Both halves are `'static` data carried by the [`KernelRef`], so the key is
/// `Copy`, hashes without allocating, and needs no parsing of anything.
///
/// `KernelRef::module` is deliberately absent: this backend compiles every
/// kernel into one flat library, so the entry name already identifies the
/// function uniquely and the module is registry metadata, not a lookup dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    entry: &'static str,
    consts: &'static [FnConst],
}

impl PipelineKey {
    /// Derives the key from a kernel reference.
    pub(crate) const fn from_kernel(kernel: KernelRef) -> Self {
        Self {
            entry: kernel.entry,
            consts: kernel.consts,
        }
    }
}

/// A resolved pipeline handle.
pub(crate) type PipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Cache of resolved pipelines, keyed by entry point and specialization.
///
/// A miss costs a hash lookup and two Objective-C message sends, so an engine
/// warms the cache by launching each `KernelRef` it uses once at load time.
pub(crate) struct PipelineCache {
    entries: HashMap<PipelineKey, PipelineState>,
}

impl fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineCache")
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCache {
    /// An empty cache; the first launch of a kernel warms it.
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::default(),
        }
    }

    /// Number of resolved pipelines.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Looks up an already-resolved pipeline.
    pub(crate) fn get(&self, kernel: KernelRef) -> Option<PipelineState> {
        self.entries.get(&PipelineKey::from_kernel(kernel)).cloned()
    }

    /// Resolves `kernel` through `library` on `device` and caches the result.
    ///
    /// `constants` must have been built from exactly `kernel.consts`, and is
    /// `None` only when that table is empty.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the entry point is absent from the library or
    /// the device refuses the pipeline.
    pub(crate) fn insert(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        kernel: KernelRef,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> CoreResult<PipelineState> {
        let name = NSString::from_str(kernel.entry);
        let function = match constants {
            None => library
                .newFunctionWithName(&name)
                .ok_or_else(|| missing(kernel.entry))?,
            Some(constants) => library
                .newFunctionWithName_constantValues_error(&name, constants)
                .map_err(|err| backend_error("newFunctionWithName", &err))?,
        };
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| backend_error("newComputePipelineState", &err))?;
        tracing::debug!(
            entry = kernel.entry,
            consts = kernel.consts.len(),
            max_threads = pipeline.maxTotalThreadsPerThreadgroup(),
            "resolved Metal pipeline",
        );
        self.entries
            .insert(PipelineKey::from_kernel(kernel), pipeline.clone());
        Ok(pipeline)
    }
}

/// Wraps an `NSError` as the structured backend failure Metal reports.
pub(crate) fn backend_error(op: &'static str, err: &NSError) -> CoreError {
    CoreError::Backend {
        backend: "metal",
        op,
        source: Box::new(MetalError::from(err.localizedDescription().to_string())),
    }
}

fn missing(entry: &str) -> CoreError {
    CoreError::Backend {
        backend: "metal",
        op: "newFunctionWithName",
        source: Box::new(MetalError::from(format!(
            "kernel {entry:?} is not present in the compiled library"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::PipelineKey;
    use shiso_core::{FnConst, FnConstValue, KernelRef};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const NARROW: [FnConst; 1] = [FnConst {
        index: 600,
        value: FnConstValue::I16(16),
    }];
    const WIDE: [FnConst; 1] = [FnConst {
        index: 600,
        value: FnConstValue::I16(32),
    }];

    const NARROW_REF: KernelRef = KernelRef::specialized("generic", "kernel_k", &NARROW);
    /// The same entry point and table under a different registry module.
    const OTHER_MODULE: KernelRef = KernelRef::specialized("prefill", "kernel_k", &NARROW);
    const WIDE_REF: KernelRef = KernelRef::specialized("generic", "kernel_k", &WIDE);

    /// Deriving the key in a `const` context is the point: no allocation and no
    /// parsing happens on the launch path.
    const NARROW_KEY: PipelineKey = PipelineKey::from_kernel(NARROW_REF);
    const OTHER_MODULE_KEY: PipelineKey = PipelineKey::from_kernel(OTHER_MODULE);
    const WIDE_KEY: PipelineKey = PipelineKey::from_kernel(WIDE_REF);

    fn hash(key: &PipelineKey) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn module_is_not_a_lookup_dimension() {
        // Every kernel lands in one flat library, so the entry name already
        // identifies the function and the module must not split the cache.
        assert_eq!(NARROW_KEY, OTHER_MODULE_KEY);
        assert_eq!(hash(&NARROW_KEY), hash(&OTHER_MODULE_KEY));
    }

    #[test]
    fn specialization_is_part_of_the_key() {
        assert_ne!(NARROW_KEY, WIDE_KEY);
        assert_ne!(hash(&NARROW_KEY), hash(&WIDE_KEY));
    }
}

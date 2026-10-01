//! The module registry: kernel images in, entry-point handles out.
//!
//! # The module is the lookup key
//!
//! This is the one place where this backend and its sibling deliberately
//! disagree. The sibling compiles every kernel into one flat library, so a
//! [`KernelRef`]'s module is metadata and its entry name alone identifies the
//! function. Here each module is a separately compiled image with its own
//! symbol table, so `module` is the key of the image and `entry` is the key
//! inside it. Both halves of a `KernelRef` therefore matter, and the handle
//! cache is keyed on the pair.
//!
//! # Fail closed
//!
//! The kernel crate answers an unknown module with zero bytes rather than an
//! error, which is right for a crate that must build on a machine with no
//! device compiler. It is not right for a loader: zero bytes are not a kernel,
//! and handing them to the driver produces a message about an unrecognised
//! image instead of one about a build that never compiled the module. So an
//! empty image is refused here, by name, with the kernel set's fingerprint in
//! the message — because the usual cause is a binary built from a different
//! kernel set than the one the caller is reading its entry names from.
//!
//! # Shared memory
//!
//! An entry point that wants more than the default dynamic shared memory has to
//! be told so once, before its first launch, and the driver remembers it for
//! the life of the module. Rather than a table of which kernels need it — which
//! would have to be kept in step with the kernels by hand — the rule is derived
//! from the launch itself: the first launch asking for more than the default
//! raises that entry point's ceiling and is remembered, and every later launch
//! of it is a set lookup rather than a driver call.
//!
//! The opt-in has to happen outside a graph capture, which it does, because the
//! caller's first pass over a region is always an eager one and a capture only
//! ever re-runs work that has already run once.

use std::collections::{HashMap, HashSet};
use std::ffi::CString;

use parking_lot::Mutex;
use shiso_core::{CoreResult, KernelRef};
use std::sync::OnceLock;

use crate::sys::nvidia;
use crate::sys::{CuFunction, CuModule, DEFAULT_SMEM_LIMIT, host_error};

/// A module image as the kernel crate hands it over.
type Image = (&'static str, &'static [u8]);

/// Refuses a module set that contains an image the build never produced.
///
/// Pure, and separate from the loading, because this is the check that catches
/// the common mistake — a binary whose kernel set was skipped or partially
/// built — and it is worth being able to test it on a machine that has no
/// driver to load the good images with.
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] naming the first empty module and the
/// fingerprint of the kernel set it came from.
fn validate_images(images: &[Image]) -> CoreResult<()> {
    for (name, bytes) in images {
        if bytes.is_empty() {
            return Err(host_error(
                "cuModuleLoadData",
                format!(
                    "kernel module {name:?} is empty in this build (kernel set {}); \
                     the module was registered but never compiled",
                    shiso_kernels::KERNEL_SET_HASH
                ),
            ));
        }
    }
    Ok(())
}

/// A loaded module that unloads itself.
#[derive(Debug)]
struct OwnedModule {
    name: &'static str,
    module: CuModule,
}

impl Drop for OwnedModule {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the handle, and unloading happens
        // here and only here.
        if let Err(err) = unsafe { nvidia::module_unload(self.module) }
            && !err.is_teardown_noop()
        {
            tracing::error!(module = self.name, error = %err, "module unload failed");
        }
    }
}

/// What the registry remembers between launches.
#[derive(Debug, Default)]
struct Cache {
    /// Resolved entry points, keyed by the whole [`KernelRef`] identity that
    /// names them.
    functions: HashMap<(&'static str, &'static str), CuFunction>,
    /// Entry points whose dynamic shared-memory ceiling has already been
    /// raised. Membership is the whole of the state: the driver keeps the
    /// attribute itself.
    opted_in: HashSet<CuFunction>,
    /// Entry-point resolutions served from the map rather than the driver.
    /// Exists so a caller can prove the cache is doing its job.
    hits: usize,
}

/// Every kernel module this build embeds, loaded and ready.
#[derive(Debug)]
pub(crate) struct ModuleRegistry {
    /// Immutable after construction, so it needs no lock: the set of modules a
    /// binary embeds is fixed at build time.
    modules: HashMap<&'static str, OwnedModule>,
    cache: Mutex<Cache>,
}

impl ModuleRegistry {
    /// Loads every module this build embeds.
    ///
    /// All of them, at open time, rather than lazily on first use: a kernel set
    /// that does not load is a deployment error, and it should surface while a
    /// model is being loaded rather than part-way through answering a request.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] when a module is empty in this build,
    /// or when the driver refuses an image — which is where an image built for
    /// a different architecture than the device surfaces.
    pub(crate) fn load_all() -> CoreResult<Self> {
        let images = shiso_kernels::ptx_modules();
        validate_images(&images)?;
        if images.is_empty() {
            tracing::warn!(
                kernel_set = shiso_kernels::KERNEL_SET_HASH,
                "this build embeds no kernel modules; every launch will be refused",
            );
        }
        let mut modules = HashMap::with_capacity(images.len());
        for (name, bytes) in images {
            let module = nvidia::module_load(bytes)?;
            modules.insert(name, OwnedModule { name, module });
        }
        tracing::debug!(
            modules = modules.len(),
            kernel_set = shiso_kernels::KERNEL_SET_HASH,
            "loaded kernel modules",
        );
        Ok(Self {
            modules,
            cache: Mutex::new(Cache::default()),
        })
    }

    /// Number of loaded modules.
    pub(crate) fn len(&self) -> usize {
        self.modules.len()
    }

    /// Entry-point resolutions this registry served from its cache.
    pub(crate) fn cache_hits(&self) -> usize {
        self.cache.lock().hits
    }

    /// Resolves one entry point, caching the handle.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] when this build embeds no such module
    /// — the message lists the ones it does embed — or when the module declares
    /// no such entry point.
    pub(crate) fn function(&self, kernel: KernelRef) -> CoreResult<CuFunction> {
        let key = (kernel.module, kernel.entry);
        {
            let mut cache = self.cache.lock();
            if let Some(func) = cache.functions.get(&key).copied() {
                cache.hits += 1;
                return Ok(func);
            }
        }
        let module = self
            .modules
            .get(kernel.module)
            .ok_or_else(|| host_error("cuModuleGetFunction", self.unknown_module(kernel.module)))?;
        let name = CString::new(kernel.entry).map_err(|_| {
            host_error(
                "cuModuleGetFunction",
                format!("entry name {:?} contains a NUL byte", kernel.entry),
            )
        })?;
        // SAFETY: `module` is owned by this registry and stays loaded for at
        // least the duration of this borrow; `name` is a live NUL-terminated
        // buffer for the whole call.
        let func = unsafe { nvidia::module_get_function(module.module, &name) }.map_err(|err| {
            tracing::debug!(
                module = kernel.module,
                entry = kernel.entry,
                error = %err,
                "entry point not found in module",
            );
            // The driver's own message is "named symbol not found" and names no
            // symbol, so the name is put back here. `docs/superpowers/plans/`
            // records this trap biting four times - a module is the lookup key
            // of a separately compiled image on this backend, and an entry the
            // engine asks for under a name has to be in an image of that name.
            // A refusal that says WHICH pair is missing is the difference
            // between a one-line fix and an afternoon.
            host_error(
                "cuModuleGetFunction",
                format!(
                    "module {:?} has no entry {:?}: {err}",
                    kernel.module, kernel.entry
                ),
            )
        })?;
        self.cache.lock().functions.insert(key, func);
        Ok(func)
    }

    /// Threads per block one entry point may be launched with, or `None`.
    ///
    /// `None` covers a module this build does not embed and an entry point the
    /// module does not export, which is what a caller choosing among compiled
    /// variants needs: a variant that is simply absent has no limit to report
    /// and drops out of the candidate set, rather than failing the whole
    /// selection. The driver's answer is the `__launch_bounds__` the entry was
    /// compiled with when there is one, and the device maximum otherwise.
    ///
    /// The resolve failure is logged rather than swallowed, the way the Metal
    /// twin logs a pipeline that does not build: a variant dropping out of the
    /// candidate set is worth seeing.
    pub(crate) fn max_block_threads(&self, kernel: KernelRef) -> Option<u32> {
        let func = match self.function(kernel) {
            Ok(func) => func,
            Err(error) => {
                tracing::debug!(
                    module = kernel.module,
                    entry = kernel.entry,
                    error = %error.chain(),
                    "no block limit: the entry point did not resolve",
                );
                return None;
            }
        };
        // SAFETY: `func` was resolved from a module this registry owns and has
        // not unloaded.
        match unsafe { nvidia::func_max_threads_per_block(func) } {
            Ok(limit) => Some(limit),
            Err(error) => {
                tracing::debug!(
                    module = kernel.module,
                    entry = kernel.entry,
                    %error,
                    "no block limit: the driver would not answer",
                );
                None
            }
        }
    }

    /// Makes sure `func` is allowed the dynamic shared memory this launch wants.
    ///
    /// A no-op — and not a driver call — for the overwhelming majority of
    /// launches, which stay inside the default. The first launch above it pays
    /// one attribute call and every later launch of the same entry point pays a
    /// set lookup.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] when the device will not give that
    /// entry point that much shared memory.
    /// The entry names a module actually exposes, read out of its own text.
    ///
    /// This is the only place those names exist. Kernels with C linkage carry the
    /// name they were written with; template instantiations carry whatever their
    /// compiler mangled from the argument list, and nothing outside the produced
    /// module knows what that is.
    pub(crate) fn entries(module: &str) -> Vec<String> {
        const MARK: &str = ".entry ";
        // Asked of the registry rather than by name, because the by-name accessor
        // treats an unknown module as a caller bug and says so loudly. Here an
        // absent module is the ordinary case - it is every module, on a machine
        // with no toolchain - so it has to be an empty answer.
        let Some((_, ptx)) = shiso_kernels::ptx_modules()
            .into_iter()
            .find(|(name, _)| *name == module)
        else {
            return Vec::new();
        };
        let Ok(text) = core::str::from_utf8(ptx) else {
            return Vec::new();
        };
        text.lines()
            .filter(|line| line.contains(".visible") && line.contains(MARK))
            .filter_map(|line| line.split(MARK).nth(1))
            .map(|rest| {
                rest.split(['(', ' ', '\t'])
                    .next()
                    .unwrap_or_default()
                    .trim()
                    .to_owned()
            })
            .filter(|name| !name.is_empty())
            .collect()
    }

    /// Resolves `logical` against the entry names `module` exposes.
    ///
    /// Two answers are possible and a third deliberately is not. An exact match is
    /// the name itself, which is every kernel with C linkage. A single entry that
    /// *contains* the logical name is that kernel's mangled instantiation, and is
    /// returned because there is nothing else it could be.
    ///
    /// Several containing entries is `None`, not a pick. That is the template case
    /// with more than one instantiation, and choosing between them needs the
    /// demangled argument list - which this cannot see. A wrong instantiation
    /// launches happily and computes the wrong thing, so an ambiguous name has to
    /// come back unresolved and be reported.
    pub(crate) fn resolve(module: &'static str, logical: &str) -> Option<KernelRef> {
        let names = Self::entries(module);
        let picked = Self::pick(&names, logical)?;
        Some(KernelRef::new(module, intern(picked)))
    }

    /// The choosing rule on its own, over a list of names.
    ///
    /// Separated so it can be exercised without a built module, which is the only
    /// way it can be exercised at all on a machine with no toolchain.
    #[must_use]
    pub(crate) fn pick<'a>(names: &'a [String], logical: &str) -> Option<&'a str> {
        if let Some(exact) = names.iter().find(|name| name.as_str() == logical) {
            return Some(exact);
        }
        let mut hits = names.iter().filter(|name| name.contains(logical));
        let only = hits.next()?;
        if hits.next().is_some() {
            tracing::warn!(
                logical,
                "several entry points carry this name; the argument list is needed to choose"
            );
            return None;
        }
        Some(only)
    }

    pub(crate) fn ensure_shared_memory(&self, func: CuFunction, smem_bytes: u32) -> CoreResult<()> {
        if smem_bytes <= DEFAULT_SMEM_LIMIT {
            return Ok(());
        }
        {
            let cache = self.cache.lock();
            if cache.opted_in.contains(&func) {
                return Ok(());
            }
        }
        // SAFETY: `func` was resolved from a module this registry owns and has
        // not unloaded.
        unsafe { nvidia::func_set_max_dynamic_smem(func, smem_bytes) }?;
        tracing::debug!(
            smem_bytes,
            "raised an entry point's dynamic shared-memory ceiling",
        );
        self.cache.lock().opted_in.insert(func);
        Ok(())
    }

    /// The message for a module this build does not embed.
    ///
    /// It lists what *is* embedded and the kernel set's fingerprint, because
    /// the two causes — a name that was never a module, and a binary built from
    /// a different kernel set — look identical from the call site and are told
    /// apart by exactly those two facts.
    fn unknown_module(&self, wanted: &str) -> String {
        let mut known: Vec<&str> = self.modules.keys().copied().collect();
        known.sort_unstable();
        format!(
            "this build embeds no kernel module {wanted:?} (kernel set {}); it embeds: {}",
            shiso_kernels::KERNEL_SET_HASH,
            if known.is_empty() {
                "nothing".to_owned()
            } else {
                known.join(", ")
            }
        )
    }
}

/// Gives a resolved entry name the `'static` life a [`KernelRef`] requires.
///
/// Leaking is the right shape here rather than a concession: the set is the entry
/// points of the modules this process loaded, it is bounded by the kernel set, and
/// each name is resolved once and then lives exactly as long as the loaded module
/// it names. The cache means a name is leaked once however often it is asked for.
fn intern(name: &str) -> &'static str {
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut set = NAMES.get_or_init(Default::default).lock();
    if let Some(found) = set.get(name) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    set.insert(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use shiso_core::{CoreError, KernelRef};

    use super::{ModuleRegistry, validate_images};

    #[test]
    fn an_empty_module_image_is_refused_by_name_before_any_driver_call() {
        let images = [("q4e_hc", &b"//ptx"[..]), ("mmq", &b""[..])];
        match validate_images(&images) {
            Err(CoreError::Backend {
                backend,
                op,
                source,
            }) => {
                assert_eq!(backend, "cuda");
                assert_eq!(op, "cuModuleLoadData");
                let text = source.to_string();
                assert!(text.contains("\"mmq\""), "{text}");
                assert!(
                    text.contains(shiso_kernels::KERNEL_SET_HASH),
                    "the message must carry the kernel set fingerprint: {text}",
                );
            }
            other => panic!("expected a fail-closed error, got {other:?}"),
        }
    }

    #[test]
    fn a_module_set_with_no_empty_images_passes_validation() {
        let images = [("q4e_hc", &b"//ptx"[..]), ("exl3", &b"\x7fELFrest"[..])];
        validate_images(&images).expect("a fully built module set is accepted");
    }

    /// The registry is constructible wherever the build embeds no modules,
    /// which is every machine without a device compiler, and that is the state
    /// this test needs: a lookup miss has to be reported without the driver
    /// ever being asked.
    #[test]
    fn an_unknown_module_is_reported_with_the_kernel_set_and_never_reaches_the_driver() {
        if !shiso_kernels::ptx_modules().is_empty() {
            // This build embeds real modules, and loading them needs an
            // initialised driver this host-only test does not set up; the
            // device tests load them.
            return;
        }
        let registry = ModuleRegistry::load_all().expect("an empty module set loads");
        assert_eq!(registry.len(), 0);
        let kernel = KernelRef::new("not_a_module", "not_an_entry");
        match registry.function(kernel) {
            Err(CoreError::Backend {
                backend,
                op,
                source,
            }) => {
                assert_eq!(backend, "cuda");
                assert_eq!(op, "cuModuleGetFunction");
                let text = source.to_string();
                assert!(text.contains("\"not_a_module\""), "{text}");
                assert!(text.contains(shiso_kernels::KERNEL_SET_HASH), "{text}");
            }
            other => panic!("expected a fail-closed error, got {other:?}"),
        }
        assert_eq!(
            registry.cache_hits(),
            0,
            "a miss must not be recorded as a cache hit",
        );
    }
}

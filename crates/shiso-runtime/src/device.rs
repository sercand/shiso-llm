//! Which device a server runs on: the user's choice and the pure rule that
//! turns it, plus what the build and the host offer, into a decision.
//!
//! The rule is a function of three plain values so the whole table is a unit
//! test; probing the host (does libcuda exist, does the device open) happens in
//! the caller and arrives here as a [`GpuState`].

use crate::boot_error::{BootError, refuse};
use crate::options::ServeOptions;

/// `--device`. The variants that exist are the backends this build compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum DeviceChoice {
    /// The GPU when it is usable, the CPU otherwise.
    #[default]
    Auto,
    /// The Metal GPU; refused when it is unavailable.
    #[cfg(feature = "metal")]
    Metal,
    /// The CUDA GPU; refused when it is unavailable.
    #[cfg(feature = "cuda")]
    Cuda,
    /// The CPU backend, even when a GPU is present.
    #[cfg(feature = "cpu")]
    Cpu,
}

/// A device a backend can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// Apple GPU through Metal.
    Metal,
    /// NVIDIA GPU through the CUDA driver.
    Cuda,
    /// The host CPU.
    Cpu,
}

/// What the user asked for, with the backend's name erased.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Want {
    /// No preference.
    Auto,
    /// The GPU this build was compiled for, by name.
    Gpu,
    /// The CPU.
    Cpu,
}

impl From<DeviceChoice> for Want {
    fn from(choice: DeviceChoice) -> Self {
        match choice {
            DeviceChoice::Auto => Self::Auto,
            #[cfg(feature = "metal")]
            DeviceChoice::Metal => Self::Gpu,
            #[cfg(feature = "cuda")]
            DeviceChoice::Cuda => Self::Gpu,
            #[cfg(feature = "cpu")]
            DeviceChoice::Cpu => Self::Cpu,
        }
    }
}

/// What the GPU backend offers on this host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuState {
    /// This build has no GPU backend.
    NotCompiled,
    /// A GPU backend is compiled but unusable; the string says why (a missing
    /// driver library, or the error from opening the device).
    Absent(String),
    /// A GPU backend is compiled and its device opened.
    Present,
}

/// The device [`resolve`] picked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Use the GPU.
    Gpu,
    /// Use the CPU. `fallback` carries the reason an `auto` run gave up on a
    /// GPU that was compiled in, so the caller can log it once.
    Cpu {
        /// Why the GPU was passed over; `None` when there was none to pass over.
        fallback: Option<String>,
    },
}

/// Why no device could be chosen.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeviceError {
    /// A GPU was required (named, or the only compiled device) and is unusable.
    #[error("the GPU is unavailable: {reason}")]
    GpuUnavailable {
        /// The probe's reason.
        reason: String,
    },
    /// The build compiled no device at all.
    #[error("this build has no compute backend; rebuild with --features metal or --features cuda")]
    NoDevice,
}

/// Picks the device. Refuses rather than substituting: a named GPU that is
/// unusable is an error, not a silent CPU run.
///
/// # Errors
///
/// [`DeviceError::GpuUnavailable`] when the GPU is required and not usable;
/// [`DeviceError::NoDevice`] when the build has neither a CPU nor a GPU backend.
pub fn resolve(want: Want, cpu_compiled: bool, gpu: GpuState) -> Result<Decision, DeviceError> {
    let cpu = || Decision::Cpu { fallback: None };
    match (want, gpu) {
        // The CPU was asked for, or it is the only device there could be.
        (Want::Cpu, _) | (Want::Auto, GpuState::NotCompiled) => {
            cpu_compiled.then(cpu).ok_or(DeviceError::NoDevice)
        }
        (_, GpuState::Present) => Ok(Decision::Gpu),
        (Want::Gpu, GpuState::NotCompiled) => Err(DeviceError::GpuUnavailable {
            reason: "this build has no GPU backend".to_owned(),
        }),
        // Only `auto` may pass over an unusable GPU, and only for a CPU that
        // was compiled in.
        (Want::Auto, GpuState::Absent(reason)) if cpu_compiled => Ok(Decision::Cpu {
            fallback: Some(reason),
        }),
        (Want::Gpu | Want::Auto, GpuState::Absent(reason)) => {
            Err(DeviceError::GpuUnavailable { reason })
        }
    }
}

/// What a boot step needs the device for; picks the CUDA options and the words
/// an open failure is reported in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Purpose {
    /// The language model and the embedding models: decode graphs follow the flag.
    Language,
    /// The speech models: their launches are never replayed.
    Audio,
    /// The image generation model: a diffusion step is never replayed.
    Image,
}

impl Purpose {
    /// The model this device is for, in the operator's words.
    const fn subject(self) -> &'static str {
        match self {
            Self::Language => "the language model",
            Self::Audio => "the speech models",
            Self::Image => "the image generation model",
        }
    }
}

/// An open backend and the device it runs on.
pub(crate) struct OpenedDevice {
    /// The backend every model of this step holds.
    pub backend: std::sync::Arc<dyn shiso_core::Backend>,
    /// Which device it is, for the errors that name one.
    pub kind: DeviceKind,
}

/// What probing the GPU found: the state for [`resolve`], the backend when it
/// opened, and the typed error when opening it failed (kept so a refused boot
/// reports the driver's own error rather than its flattened text).
struct GpuProbe {
    state: GpuState,
    backend: Option<std::sync::Arc<dyn shiso_core::Backend>>,
    open_error: Option<shiso_core::CoreError>,
}

impl GpuProbe {
    fn absent(reason: String) -> Self {
        Self {
            state: GpuState::Absent(reason),
            backend: None,
            open_error: None,
        }
    }

    fn failed(error: shiso_core::CoreError) -> Self {
        Self {
            state: GpuState::Absent(error.chain().to_string()),
            backend: None,
            open_error: Some(error),
        }
    }

    fn opened(backend: std::sync::Arc<dyn shiso_core::Backend>) -> Self {
        Self {
            state: GpuState::Present,
            backend: Some(backend),
            open_error: None,
        }
    }
}

/// Probes and opens the GPU this build was compiled for.
///
/// Ordinal zero for CUDA: which device to run on is a placement decision that
/// belongs outside the process (`CUDA_VISIBLE_DEVICES`).
#[cfg_attr(
    not(any(feature = "metal", feature = "cuda")),
    expect(
        unused_variables,
        reason = "a build with no GPU backend reads neither argument"
    )
)]
fn open_gpu(args: &ServeOptions, purpose: Purpose) -> GpuProbe {
    #[cfg(feature = "metal")]
    {
        let _ = (args, purpose);
        if !shiso_metal::metal_available() {
            return GpuProbe::absent("no Metal device found".to_owned());
        }
        match shiso_metal::MetalBackend::new() {
            Ok(backend) => GpuProbe::opened(backend),
            Err(error) => GpuProbe::failed(error),
        }
    }
    #[cfg(feature = "cuda")]
    {
        if !shiso_cuda::driver_library_present() {
            return GpuProbe::absent("the CUDA driver library (libcuda) was not found".to_owned());
        }
        let graphs = purpose == Purpose::Language && !args.no_decode_graphs;
        let options = shiso_cuda::CudaOptions::default().with_decode_graphs(graphs);
        match shiso_cuda::CudaBackend::new(0, options) {
            Ok(backend) => GpuProbe::opened(backend),
            Err(error) => GpuProbe::failed(error),
        }
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        GpuProbe {
            state: GpuState::NotCompiled,
            backend: None,
            open_error: None,
        }
    }
}

/// The GPU this build was compiled for, by the name `--device` spells it.
const fn gpu_name() -> &'static str {
    if cfg!(feature = "metal") {
        "metal"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "gpu"
    }
}

/// The sentence for a GPU that was found and would not open.
fn open_context(purpose: Purpose) -> String {
    if cfg!(feature = "cuda") {
        match purpose {
            Purpose::Language => "opening the CUDA device".to_owned(),
            Purpose::Audio => "opening the CUDA device for the speech models".to_owned(),
            Purpose::Image => "opening the CUDA device for the image generation model".to_owned(),
        }
    } else {
        "opening the Metal device".to_owned()
    }
}

/// The kind of device `Decision::Gpu` means in this build.
const fn gpu_kind() -> DeviceKind {
    if cfg!(feature = "cuda") {
        DeviceKind::Cuda
    } else {
        DeviceKind::Metal
    }
}

/// Opens the device `--device` names, or the best one for `auto`.
///
/// The GPU is only probed when the CPU was not asked for, so `--device cpu`
/// never touches a driver. Each call opens its own backend: the speech, image
/// and language models want command streams of their own.
pub(crate) fn open_device(
    args: &ServeOptions,
    purpose: Purpose,
) -> Result<OpenedDevice, BootError> {
    let want = Want::from(args.device);
    let probe = if want == Want::Cpu {
        GpuProbe {
            state: GpuState::NotCompiled,
            backend: None,
            open_error: None,
        }
    } else {
        open_gpu(args, purpose)
    };
    let GpuProbe {
        state,
        backend: opened,
        open_error,
    } = probe;
    let decision = resolve(want, cfg!(feature = "cpu"), state).map_err(|error| match error {
        DeviceError::GpuUnavailable { reason } => match open_error {
            // The device was there and refused to open: keep the typed error.
            Some(source) => BootError::Step {
                context: open_context(purpose),
                source: Box::new(source),
            },
            None => refuse!(
                "the {} device is unavailable for {}: {reason}",
                gpu_name(),
                purpose.subject()
            ),
        },
        DeviceError::NoDevice => refuse!("{error}"),
    })?;
    match decision {
        Decision::Gpu => {
            let backend = opened.ok_or_else(|| refuse!("the GPU backend did not open"))?;
            Ok(OpenedDevice {
                backend,
                kind: gpu_kind(),
            })
        }
        Decision::Cpu { fallback } => {
            if let Some(reason) = fallback {
                tracing::info!("device auto chose cpu because {reason}");
                if cfg!(feature = "cuda") {
                    tracing::warn!(
                        "this CUDA build is serving on the CPU because {reason}; \
                         it will be much slower than the GPU"
                    );
                }
            }
            open_cpu(purpose)
        }
    }
}

/// Opens the CPU backend.
#[cfg(feature = "cpu")]
fn open_cpu(purpose: Purpose) -> Result<OpenedDevice, BootError> {
    use crate::boot_error::BootContext as _;
    let backend = shiso_cpu::CpuBackend::new(shiso_cpu::CpuOptions::default())
        .with_context(|| format!("opening the CPU device for {}", purpose.subject()))?;
    Ok(OpenedDevice {
        backend,
        kind: DeviceKind::Cpu,
    })
}

/// Unreachable: [`resolve`] only returns a CPU decision when one is compiled.
#[cfg(not(feature = "cpu"))]
fn open_cpu(_purpose: Purpose) -> Result<OpenedDevice, BootError> {
    Err(refuse!("this build has no CPU backend"))
}

/// The kernel a CPU run found unported, if `err` is or wraps that refusal.
///
/// Checked per link rather than by walking `source` alone: the wrappers on the
/// boot paths (`BootError::Core`, `EmbedError::Core`, `AudioError::Device`) are
/// `#[error(transparent)]`, so their `source` skips the `CoreError` they hold.
fn unported_kernel(err: &(dyn std::error::Error + 'static)) -> Option<&'static str> {
    use shiso_core::CoreError;
    let core = if let Some(core) = err.downcast_ref::<CoreError>() {
        Some(core)
    } else if let Some(BootError::Core(core)) = err.downcast_ref::<BootError>() {
        Some(core)
    } else if let Some(shiso_models::embedder::EmbedError::Core(core)) =
        err.downcast_ref::<shiso_models::embedder::EmbedError>()
    {
        Some(core)
    } else if let Some(shiso_models::embedder::EmbedError::Audio(inner)) =
        err.downcast_ref::<shiso_models::embedder::EmbedError>()
    {
        // Transparent as well, so the audio error is not a `source` of it.
        return unported_kernel(inner);
    } else if let Some(shiso_audio::AudioError::Device(core)) =
        err.downcast_ref::<shiso_audio::AudioError>()
    {
        Some(core)
    } else {
        None
    };
    match core {
        Some(CoreError::Unsupported {
            what,
            backend: "cpu",
        }) => Some(what),
        _ => None,
    }
}

/// Turns a CPU run's "kernel not ported" into the sentence an operator needs.
///
/// The CPU backend ports kernels family by family, so a checkpoint can reach
/// one that is not there yet. That is a property of the model and the device,
/// not a fault, and it should end boot saying so rather than as a backend
/// error title. Errors on any other device pass through untouched.
///
/// Covered: the language, embedding and speech paths, whose errors keep the
/// `CoreError` typed. Not covered: the image path, whose device errors are
/// strings (`ImageGenError::Backend(String)`), so an unported kernel there
/// surfaces as the raw "... is not supported by the cpu backend" message.
pub(crate) fn on_device<T>(kind: DeviceKind, r: Result<T, BootError>) -> Result<T, BootError> {
    let Err(error) = r else { return r };
    if kind == DeviceKind::Cpu {
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        while let Some(err) = cause {
            if let Some(what) = unported_kernel(err) {
                return Err(refuse!(
                    "this model is not available on the CPU device ({what} is not ported)"
                ));
            }
            cause = err.source();
        }
    }
    Err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn absent(why: &str) -> GpuState {
        GpuState::Absent(why.to_owned())
    }

    #[test]
    fn auto_uses_the_gpu_when_present() {
        assert_eq!(
            resolve(Want::Auto, true, GpuState::Present),
            Ok(Decision::Gpu)
        );
        assert_eq!(
            resolve(Want::Auto, false, GpuState::Present),
            Ok(Decision::Gpu)
        );
    }

    #[test]
    fn auto_falls_back_when_the_library_is_absent() {
        assert_eq!(
            resolve(Want::Auto, true, absent("libcuda.so.1 not found")),
            Ok(Decision::Cpu {
                fallback: Some("libcuda.so.1 not found".to_owned())
            })
        );
    }

    #[test]
    fn auto_falls_back_when_the_cuda_open_errs() {
        assert_eq!(
            resolve(Want::Auto, true, absent("cuInit failed: no device")),
            Ok(Decision::Cpu {
                fallback: Some("cuInit failed: no device".to_owned())
            })
        );
    }

    #[test]
    fn auto_without_a_gpu_feature_is_cpu_without_a_fallback_note() {
        assert_eq!(
            resolve(Want::Auto, true, GpuState::NotCompiled),
            Ok(Decision::Cpu { fallback: None })
        );
    }

    #[test]
    fn auto_with_neither_device_is_no_device() {
        assert_eq!(
            resolve(Want::Auto, false, GpuState::NotCompiled),
            Err(DeviceError::NoDevice)
        );
        assert_eq!(
            DeviceError::NoDevice.to_string(),
            "this build has no compute backend; rebuild with --features metal or --features cuda"
        );
    }

    #[test]
    fn auto_with_an_unusable_gpu_and_no_cpu_refuses_with_the_probe_reason() {
        assert_eq!(
            resolve(Want::Auto, false, absent("no driver")),
            Err(DeviceError::GpuUnavailable {
                reason: "no driver".to_owned()
            })
        );
    }

    #[test]
    fn a_named_gpu_that_is_absent_refuses_with_the_probe_reason() {
        for cpu in [true, false] {
            assert_eq!(
                resolve(Want::Gpu, cpu, absent("libcuda.so.1 not found")),
                Err(DeviceError::GpuUnavailable {
                    reason: "libcuda.so.1 not found".to_owned()
                })
            );
        }
        assert!(matches!(
            resolve(Want::Gpu, true, GpuState::NotCompiled),
            Err(DeviceError::GpuUnavailable { .. })
        ));
        assert_eq!(
            resolve(Want::Gpu, true, GpuState::Present),
            Ok(Decision::Gpu)
        );
    }

    #[test]
    fn cpu_is_cpu_even_with_a_gpu_present() {
        for gpu in [GpuState::Present, GpuState::NotCompiled, absent("x")] {
            assert_eq!(
                resolve(Want::Cpu, true, gpu),
                Ok(Decision::Cpu { fallback: None })
            );
        }
        assert_eq!(
            resolve(Want::Cpu, false, GpuState::Present),
            Err(DeviceError::NoDevice)
        );
    }

    #[test]
    fn an_unported_kernel_names_the_cpu_device() {
        use crate::boot_error::BootError;
        use shiso_core::CoreError;
        let unported = || {
            BootError::Core(CoreError::Unsupported {
                what: "kernel_x",
                backend: "cpu",
            })
        };
        let text = on_device(DeviceKind::Cpu, Err::<(), _>(unported()))
            .expect_err("a CPU run that hits an unported kernel is refused")
            .to_string();
        assert!(text.contains("not available on the CPU device"), "{text}");
        assert!(text.contains("kernel_x"), "{text}");
        // Another device keeps its own error word for word.
        let other = on_device(DeviceKind::Metal, Err::<(), _>(unported()))
            .expect_err("the error passes through")
            .to_string();
        assert_eq!(other, unported().to_string());
    }

    #[test]
    fn an_unported_kernel_is_found_behind_a_step_context() {
        use crate::boot_error::{BootContext as _, BootError};
        use shiso_core::CoreError;
        let nested: Result<(), BootError> = Err(CoreError::Unsupported {
            what: "kernel_y",
            backend: "cpu",
        })
        .context("loading the checkpoint");
        let text = on_device(DeviceKind::Cpu, nested)
            .expect_err("refused")
            .to_string();
        assert!(
            text.contains("kernel_y") && text.contains("CPU device"),
            "{text}"
        );
    }

    #[test]
    fn an_unported_kernel_is_found_behind_each_transparent_wrapper() {
        use crate::boot_error::BootError;
        use shiso_core::CoreError;
        let core = || CoreError::Unsupported {
            what: "kernel_z",
            backend: "cpu",
        };
        let step = |source: Box<dyn std::error::Error + Send + Sync>| BootError::Step {
            context: "opening".to_owned(),
            source,
        };
        let wrapped = [
            step(Box::new(shiso_models::embedder::EmbedError::Core(core()))),
            step(Box::new(shiso_audio::AudioError::Device(core()))),
            step(Box::new(shiso_models::embedder::EmbedError::Audio(
                shiso_audio::AudioError::Device(core()),
            ))),
        ];
        for error in wrapped {
            let text = on_device(DeviceKind::Cpu, Err::<(), _>(error))
                .expect_err("refused")
                .to_string();
            assert!(
                text.contains("kernel_z") && text.contains("CPU device"),
                "{text}"
            );
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn device_cpu_opens_the_cpu_without_probing_a_gpu() {
        let args = ServeOptions {
            device: DeviceChoice::Cpu,
            ..ServeOptions::default()
        };
        for purpose in [Purpose::Language, Purpose::Audio, Purpose::Image] {
            let opened = open_device(&args, purpose).expect("the CPU opens");
            assert_eq!(opened.kind, DeviceKind::Cpu);
        }
    }

    #[test]
    fn the_default_choice_is_auto() {
        assert_eq!(DeviceChoice::default(), DeviceChoice::Auto);
        assert_eq!(Want::from(DeviceChoice::Auto), Want::Auto);
    }
}

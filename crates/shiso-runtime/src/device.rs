//! Which device a server runs on: the user's choice and the pure rule that
//! turns it, plus what the build and the host offer, into a decision.
//!
//! The rule is a function of three plain values so the whole table is a unit
//! test; probing the host (does libcuda exist, does the device open) happens in
//! the caller and arrives here as a [`GpuState`].

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
    fn the_default_choice_is_auto() {
        assert_eq!(DeviceChoice::default(), DeviceChoice::Auto);
        assert_eq!(Want::from(DeviceChoice::Auto), Want::Auto);
    }
}

//! Every device-opening kernel case in one test target.
//!
//! Each file used to be its own target. A target is a process, and each process
//! compiled the whole kernel library at runtime (about a minute) and linked its
//! own copy of the backend, so the gate paid both 33 times. As modules of one
//! target they share the compile (`compile_library` caches per process) and the
//! link. The files keep their names: `--test kernels dense_attn::` runs one.
//!
//! A file not listed here is not built, so a new kernel file has to be added.

#[path = "audio_kernels.rs"]
mod audio_kernels;
#[path = "contract.rs"]
mod contract;
#[path = "dense_affine.rs"]
mod dense_affine;
#[path = "dense_attn.rs"]
mod dense_attn;
#[path = "dense_block_draft.rs"]
mod dense_block_draft;
#[path = "dense_exact_affine.rs"]
mod dense_exact_affine;
#[path = "dense_iq.rs"]
mod dense_iq;
#[path = "dense_misc.rs"]
mod dense_misc;
#[path = "dense_ple.rs"]
mod dense_ple;
#[path = "dense_q4_0.rs"]
mod dense_q4_0;
#[path = "dense_ssm_moe.rs"]
mod dense_ssm_moe;
#[path = "dense_vision.rs"]
mod dense_vision;
#[path = "dit_kernels.rs"]
mod dit_kernels;
#[path = "generic_q8.rs"]
mod generic_q8;
#[path = "metal_kernels.rs"]
mod metal_kernels;
#[path = "q35_attn.rs"]
mod q35_attn;
#[path = "q35_compile.rs"]
mod q35_compile;
#[path = "q35_gdn.rs"]
mod q35_gdn;
#[path = "q35_gemv.rs"]
mod q35_gemv;
#[path = "q35_norm_rope.rs"]
mod q35_norm_rope;
#[path = "q35moe_moe.rs"]
mod q35moe_moe;
#[path = "q4e_affine.rs"]
mod q4e_affine;
#[path = "q4e_fp8.rs"]
mod q4e_fp8;
#[path = "q4e_gdn.rs"]
mod q4e_gdn;
#[path = "q4e_hc.rs"]
mod q4e_hc;
#[path = "q4e_idx.rs"]
mod q4e_idx;
#[path = "q4e_idxring.rs"]
mod q4e_idxring;
#[path = "q4e_matvec.rs"]
mod q4e_matvec;
#[path = "q4e_misc.rs"]
mod q4e_misc;
#[path = "q4e_ple.rs"]
mod q4e_ple;
#[path = "q4e_qsa.rs"]
mod q4e_qsa;
#[path = "q4e_route.rs"]
mod q4e_route;
#[path = "sample_kernels.rs"]
mod sample_kernels;
#[path = "ternary.rs"]
mod ternary;
#[path = "vision_kernels.rs"]
mod vision_kernels;

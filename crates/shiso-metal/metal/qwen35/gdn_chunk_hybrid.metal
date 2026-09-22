// A chunk variant that fits every device: the deltas spilled to a device
// scratch buffer, 128 lanes.
//
// It is the gdn_chunk.metal body at CSPLIT = 1 -- one lane per value column,
// all eight of the row's four-slot units on that lane, the same per-unit
// rotation, the same q35_canonical4 pairing, the same fold -- and the one
// thing that differs is WHERE the deltas live.
//
// They cannot live in threadgroup memory, which is what the recurrent hybrid
// does with its staged quarter. The deltas are per value column: at CSPLIT = 1
// each of the 128 lanes owns a whole run of Q35_GDN_CHUNK of them, so the
// threadgroup would need 128 * 64 * 4 B = 32768 B of staging -- the entire
// threadgroup memory every Apple Silicon GPU (Mac GPU family 2) is built to,
// with s_A and the two gate arrays already holding 16896 B of it. There is no
// arrangement of that; the 15872 B left is less than half of what is needed.
// So the scratch is a device buffer at index 6, one row per value head, which
// makes this the one delta-rule entry point whose signature is not the
// six-buffer one.
//
// The row is laid out [token][column] rather than [column][token]: the 128
// lanes read the delta of the same token at the same moment in the
// substitution and the output phases, so that order turns 128 scattered loads
// into one contiguous span. It does not change the arithmetic -- it is the
// same `DSTRIDE` template argument the register variants pass as 1 -- but it
// is the difference between slow and unusable.
//
// This exists so that no device is refused. It is expected to lose to the
// wider register-resident shapes, since a device round trip per delta costs
// more than a register read on every fma that touches one -- but not to every
// shape: measured on this M1 Max, `chunk_hybrid` at 2 282 950 ns beat `c1`,
// its own 128-lane sibling, at 2 295 625 ns. "Sole survivor" was the wrong
// bar; "narrower than the widest register-resident shape" is the one the data
// supports.

// The order this file claims is the order gdn_chunk.metal claims, and the
// library's default fast math is just as free to reassociate the additions
// reached from this file as it is there. The body is parsed inside that
// file's safe region, but the instantiation point is here, so the region is
// reopened around it rather than assumed. See the note at
// #pragma METAL fp math_mode(safe) in gdn_chunk.metal, and at compile_library
// in shiso/crates/shiso-metal/src/compile.rs, before touching either pragma:
// `fast` below restores the BUILD DEFAULT, not a saved state, and full_source
// concatenates files in a fixed order, so a stray `fast` here would silently
// turn fast math back on for every q35 and vision source that follows this
// file.
#pragma METAL fp math_mode(safe)

// The deltas of one threadgroup: one run of Q35_GDN_CHUNK tokens for each of
// the Q35_GDN_D value columns. The host sizes the scratch as
// `n_head_v * Q35_GDN_CHUNK * Q35_GDN_D` floats so that two value heads never
// share a row.
#define Q35_GDN_CHUNK_SPILL (Q35_GDN_CHUNK * Q35_GDN_D)

/// The CSPLIT = 1 body with its deltas in device memory: the always-fits
/// chunk variant.
///
/// 128 lanes, eight four-slot units each. The seventh buffer is the scratch
/// the deltas spill to; everything above it is the six-buffer contract the
/// other chunk entry points keep.
kernel void kernel_q35_gdn_chunk_hybrid(
        constant q35_args_gdn_chunk & args [[buffer(0)]],
        device float *attn_out [[buffer(1)]],
        device float *state [[buffer(2)]],
        device const float *qkv [[buffer(3)]],
        device const float *decay [[buffer(4)]],
        device const float *beta [[buffer(5)]],
        device float *dreg_spill [[buffer(6)]],
        threadgroup char *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    // One block of deltas per value head, indexed by the head this threadgroup
    // owns, so two heads never share; inside it, this lane's column, reached
    // with the Q35_GDN_D pitch the body is given as DSTRIDE. At CSPLIT = 1 the
    // column index is the lane index.
    device float *dreg = dreg_spill
            + (uint64_t) tgpig.x * Q35_GDN_CHUNK_SPILL
            + tpitg.x;
    q35_gdn_chunk_impl<1u, Q35_GDN_D>(
            args, attn_out, state, qkv, decay, beta, dreg, tbuf,
            tgpig, tpitg, ntg);
}

// Back to the library-wide default for the files concatenated after this
// one -- `fast` means the build's default, not a saved state; see
// compile_library in shiso/crates/shiso-metal/src/compile.rs.
#pragma METAL fp math_mode(fast)

/* shs_repack: the aligned-SoA derived-weight layout library shared by the
 * offline weight server and the engine's in-process
 * self-load build.  The GGUF tensor catalog walk, the repack
 * candidate predicates, the S5 parallel repack driver, the FNV-1a artifact
 * hash, and the three aligned repack builds (IQ2_XXS gate/up, Q2_K down,
 * Q8_0 dense) live here so both producers emit bit-identical artifacts
 * (gate: --repack-hash / SHISO_WS_REPACK_HASH FNV lines).  Layout contracts
 * are shared with the consuming kernels in cuda/mmq/mmq.h. */
#ifndef SHISO_REPACK_H
#define SHISO_REPACK_H

#include <cstdint>
#include <string>
#include <vector>

struct shs_repack_file {
    int fd = -1;
    int direct_fd = -1;
    const uint8_t *data = nullptr;
    uint64_t size = 0;
    uint64_t direct_align = 1;
};

struct shs_repack_span {
    uint64_t off = 0;
    uint64_t end = 0;
};

struct shs_repack_tensor {
    std::string name;
    uint32_t type = 0;
    uint32_t ndim = 0;
    uint64_t dims[8] = {};
    uint64_t elements = 0;
    uint64_t off = 0;
    uint64_t bytes = 0;
};

/* Aligned-artifact kinds.  Numbering is shared with the weight-server
 * manifest format and the engine's cuda_derived_kind; the
 * layout comments live on the builders in repack.cu. */
enum {
    SHS_REPACK_Q8_0_F16_COLMAJOR = 2,
    SHS_REPACK_IQ2_XXS_ALIGNED_MOE = 4,
    SHS_REPACK_Q8_0_ALIGNED_DENSE = 5,
    SHS_REPACK_Q2_K_ALIGNED_MOE = 6,
};

struct shs_repack_artifact {
    const shs_repack_tensor *t = nullptr; /* borrowed from the caller's catalog */
    uint32_t kind = 0;                    /* SHS_REPACK_* */
    uint64_t bytes = 0;                   /* artifact byte size */
    uint64_t in_dim = 0;                  /* dims[0] */
    uint64_t out_dim = 0;                 /* dims[1] */
    uint32_t group_count = 0;             /* dims[2] for MoE kinds, 1 for dense,
                                             0 for f16 colmajor (the engine's
                                             lazy-cache lookup passes 0) */
    void *dev = nullptr;                  /* device artifact buffer */
};

/* Optional device-allocation hooks (weight server VMM backend).  alloc must
 * fill art->dev with at least art->bytes of device memory and return true;
 * free releases an allocation made by alloc.  Leave both NULL for plain
 * cudaMalloc/cudaFree. */
typedef bool (*shs_repack_alloc_fn)(void *ctx, shs_repack_artifact *art);
typedef void (*shs_repack_free_fn)(void *ctx, shs_repack_artifact *art);

struct shs_repack_build_args {
    const char *log_prefix = "shiso";  /* stderr line prefix ("weight-server" | "shiso") */
    const char *model_id = "base";   /* log label */
    const char *path = nullptr;      /* GGUF path (each builder re-maps it) */
    const std::vector<shs_repack_tensor> *records = nullptr;
    int device = 0;
    uint64_t copy_chunk_bytes = 0;   /* staged read chunk (weight server --copy-chunk-mb) */
    shs_repack_alloc_fn alloc_fn = nullptr;
    shs_repack_free_fn free_fn = nullptr;
    void *alloc_ctx = nullptr;
};

bool shs_repack_map_file(const char *log_prefix, const char *path, shs_repack_file &m);
void shs_repack_unmap_file(shs_repack_file &m);

/* Parse the GGUF header + tensor table into whole-tensor spans and records
 * with absolute file offsets.  Either out pointer may be NULL. */
bool shs_repack_collect_catalog(const char *log_prefix,
                                const shs_repack_file &m,
                                std::vector<shs_repack_span> *spans,
                                std::vector<shs_repack_tensor> *records);

bool shs_repack_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset);
/* Read [file_off, file_off+bytes) into stage, preferring the O_DIRECT fd;
 * *payload points at the requested bytes inside stage. */
bool shs_repack_read_stage(const shs_repack_file &m, void *stage, uint64_t stage_bytes,
                           uint64_t file_off, uint64_t bytes, const char **payload);

/* Candidate predicates for the three aligned repacks. */
bool shs_repack_iq2_candidate(const shs_repack_tensor &t);
bool shs_repack_q2k_candidate(const shs_repack_tensor &t);
bool shs_repack_q8_candidate(const shs_repack_tensor &t);
bool shs_repack_q8_f16_candidate(const shs_repack_tensor &t);

/* S5 driver configuration.  Unset (0 / false) defers to the env knobs
 * SHISO_WS_REPACK_THREADS and SHISO_WS_REPACK_HASH. */
void shs_repack_set_threads(int n);
void shs_repack_set_hash(bool on);

uint64_t shs_repack_fnv1a(const unsigned char *p, uint64_t n, uint64_t h);

/* Build every candidate's aligned artifact.  On success appends the built
 * artifacts to out, adds their byte total to *repacked_bytes_out (may be
 * NULL), and returns true; on any failure releases every artifact this call
 * allocated and returns false without touching out. */
bool shs_repack_build_q8_aligned(const shs_repack_build_args &a,
                                 std::vector<shs_repack_artifact> &out,
                                 uint64_t *repacked_bytes_out);
bool shs_repack_build_iq2_aligned(const shs_repack_build_args &a,
                                  std::vector<shs_repack_artifact> &out,
                                  uint64_t *repacked_bytes_out);
bool shs_repack_build_q2k_aligned(const shs_repack_build_args &a,
                                  std::vector<shs_repack_artifact> &out,
                                  uint64_t *repacked_bytes_out);
bool shs_repack_build_q8_f16(const shs_repack_build_args &a,
                             std::vector<shs_repack_artifact> &out,
                             uint64_t *repacked_bytes_out);

#endif /* SHISO_REPACK_H */

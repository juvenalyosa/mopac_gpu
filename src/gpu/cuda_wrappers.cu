// Portable CUDA interop for MOPAC: cuBLAS/cuBLASLt GEMM, cuBLAS SYRK,
// cuSOLVER SYEVD, and basic GPU info.
// Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
//
// Production policy: dense linear algebra used by chemistry code is routed
// through NVIDIA libraries in FP64. Project-local low-precision kernels are
// not part of the production path unless a future chemistry validation
// explicitly accepts lower precision.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cublasXt.h>
#include <cusolverDn.h>
#if defined(HAVE_CUSOLVER_MG)
#include <cusolverMg.h>
#endif
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <vector>
#include <array>
#include <cctype>
#include <mutex>
#include <limits>

#include "grad_launch.h"

// Fortran section-timer hook (mozyme_section_timers): lets the resident Fock
// plan packing report its upload / count / build split in the [PROFILE] table.
extern "C" void mozyme_section_timer_add_c(const char *name, int name_len, double ms);
namespace {
inline void mz_add_section_ms(const char *name, double ms) {
  mozyme_section_timer_add_c(name, static_cast<int>(std::strlen(name)), ms);
}
inline double mz_host_ms_since(const std::chrono::steady_clock::time_point &t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace


#if defined(MOPAC_ENABLE_NVTX)
#  if defined(__has_include)
#    if __has_include(<nvToolsExt.h>)
#      include <nvToolsExt.h>
#      define MOPAC_HAVE_NVTX 1
#    else
#      define MOPAC_HAVE_NVTX 0
#    endif
#  else
#    include <nvToolsExt.h>
#    define MOPAC_HAVE_NVTX 1
#  endif
#else
#  define MOPAC_HAVE_NVTX 0
#endif

#if !defined(MOPAC_UNUSED)
#  if defined(__GNUC__) || defined(__clang__)
#    define MOPAC_UNUSED __attribute__((unused))
#  else
#    define MOPAC_UNUSED
#  endif
#endif

enum {
  kMozymeDirectMaxW = 2025,
  kMozymeDirectSpdScratchSp = 0,
  kMozymeDirectSpdScratchPp = kMozymeDirectSpdScratchSp + 9,
  kMozymeDirectSpdScratchSd = kMozymeDirectSpdScratchPp + 54,
  kMozymeDirectSpdScratchDp = kMozymeDirectSpdScratchSd + 25,
  kMozymeDirectSpdScratchDdrot = kMozymeDirectSpdScratchDp + 225,
  kMozymeDirectSpdScratchRi = kMozymeDirectSpdScratchDdrot + 375,
  kMozymeDirectSpdScratchRep = kMozymeDirectSpdScratchRi + 22,
  kMozymeDirectSpdScratchV = kMozymeDirectSpdScratchRep + 492,
  kMozymeDirectSpdScratchWw = kMozymeDirectSpdScratchV + 2116,
  kMozymeDirectSpdScratchReppdArg =
      kMozymeDirectSpdScratchWw + kMozymeDirectMaxW,
  kMozymeDirectSpdScratchReppdSqr = kMozymeDirectSpdScratchReppdArg + 72,
  kMozymeDirectSpdScratchRotP = kMozymeDirectSpdScratchReppdSqr + 72,
  kMozymeDirectSpdScratchRotD = kMozymeDirectSpdScratchRotP + 9,
  kMozymeDirectSpdScratchDoubles = kMozymeDirectSpdScratchRotD + 25,
  kMozymeResidentDirectPackScratchDoubles =
      kMozymeDirectMaxW + kMozymeDirectSpdScratchDoubles
};

extern "C" bool mopac_cuda_fetch_fock(double *host_ptr, size_t linear);

// Lightweight verbose/timing control for BLAS wrappers
static int w_verbose = 0; static int w_inited = 0;
static inline void ensure_w_verbose() {
  if (!w_inited) {
    const char* v = std::getenv("MOPAC_GPU_VERBOSE");
    if (v && (std::strcmp(v, "1")==0 || std::strcmp(v, "on")==0 || std::strcmp(v, "true")==0)) w_verbose = 1;
    w_inited = 1;
  }
}

static inline bool equals_ci(const char* a, const char* b) {
  if (!a || !b) return false;
  while (*a && *b) {
    if (std::tolower(static_cast<unsigned char>(*a)) != std::tolower(static_cast<unsigned char>(*b))) return false;
    ++a; ++b;
  }
  return *a == '\0' && *b == '\0';
}

static inline bool env_token_ci(const char *value, char *out, std::size_t out_size) {
  if (!value || !out || out_size == 0) return false;
  while (*value && std::isspace(static_cast<unsigned char>(*value))) ++value;
  std::size_t len = 0;
  while (value[len] && !std::isspace(static_cast<unsigned char>(value[len])) &&
         len + 1 < out_size) {
    out[len] = static_cast<char>(
        std::tolower(static_cast<unsigned char>(value[len])));
    ++len;
  }
  out[len] = '\0';
  return len > 0;
}

static inline bool env_truthy_ci(const char *value) {
  char token[16] = {};
  if (!env_token_ci(value, token, sizeof(token))) return false;
  return std::strcmp(token, "0") != 0 && std::strcmp(token, "f") != 0 &&
         std::strcmp(token, "false") != 0 && std::strcmp(token, "n") != 0 &&
         std::strcmp(token, "no") != 0 && std::strcmp(token, "off") != 0;
}

static inline bool env_true_ci(const char *value) {
  char token[16] = {};
  if (!env_token_ci(value, token, sizeof(token))) return false;
  return std::strcmp(token, "1") == 0 || std::strcmp(token, "t") == 0 ||
         std::strcmp(token, "true") == 0 || std::strcmp(token, "y") == 0 ||
         std::strcmp(token, "yes") == 0 || std::strcmp(token, "on") == 0;
}

static inline bool report_cuda_error(const char *where, cudaError_t status) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "[GPU ERROR] %s: %s\n", where, cudaGetErrorString(status));
  return false;
}

static inline bool report_cublas_error(const char *where, cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "[GPU ERROR] %s: cuBLAS status %d\n", where, static_cast<int>(status));
  return false;
}

static inline bool report_cusolver_error(const char *where, cusolverStatus_t status) {
  if (status == CUSOLVER_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "[GPU ERROR] %s: cuSOLVER status %d\n", where, static_cast<int>(status));
  return false;
}

static inline bool mozyme_sparse_fock_basis_supported(int nbasis) {
  return nbasis == 1 || nbasis == 4 || nbasis == 9;
}

static inline void poison_host_doubles(double *values, std::size_t count) {
  if (!values) return;
  const double bad = std::numeric_limits<double>::quiet_NaN();
  std::fill(values, values + count, bad);
}

static int g_gpu_profile_level = 0;
static int g_gpu_profile_inited = 0;
static int g_gpu_profile_env_requested MOPAC_UNUSED = 0;

static inline int gpu_profile_level() {
  if (!g_gpu_profile_inited) {
    const char* s = std::getenv("MOPAC_GPU_PROFILE");
    if (s && *s) {
      g_gpu_profile_env_requested = 1;
      if (equals_ci(s, "0") || equals_ci(s, "off") || equals_ci(s, "false")) {
        g_gpu_profile_level = 0;
      } else if (equals_ci(s, "2") || equals_ci(s, "full") || equals_ci(s, "2+") || equals_ci(s, "verbose")) {
        g_gpu_profile_level = 2;
      } else {
        g_gpu_profile_level = 1;
      }
    } else {
      g_gpu_profile_level = 0;
    }
    g_gpu_profile_inited = 1;
  }
  return g_gpu_profile_level;
}

static inline bool gpu_profile_enabled() {
  return gpu_profile_level() >= 2;
}

struct NvtxRange {
#if MOPAC_HAVE_NVTX
  bool active;
  NvtxRange(const char* name, uint32_t color) : active(false) {
    if (name && gpu_profile_enabled()) {
      nvtxEventAttributes_t attr{};
      attr.version = NVTX_VERSION;
      attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      attr.colorType = NVTX_COLOR_ARGB;
      attr.color = color;
      attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
      attr.message.ascii = name;
      nvtxRangePushEx(&attr);
      active = true;
    }
  }
  ~NvtxRange() {
    if (active) nvtxRangePop();
  }
#else
  NvtxRange(const char*, uint32_t) {}
#endif
};

struct BlasProfileEntry {
  long long calls = 0;
  long long tiled_calls = 0;
  long long tiles = 0;
  double total_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double total_flops = 0.0;
};

static BlasProfileEntry g_prof_gemm_single;
static BlasProfileEntry g_prof_syrk_single;
static BlasProfileEntry g_prof_gemm_pair;
static BlasProfileEntry g_prof_syrk_pair;
static BlasProfileEntry g_prof_disp_eval;

struct ScopedBlasProfile {
  BlasProfileEntry* entry;
  double flops;
  bool active;
  bool noted_tiled = false;
  long long tile_accum = 0;
  std::chrono::high_resolution_clock::time_point t0;
  ScopedBlasProfile(BlasProfileEntry* e, double flop_count)
      : entry(e), flops(flop_count) {
    active = gpu_profile_enabled() && entry;
    if (!active) return;
    entry->calls += 1;
    entry->total_flops += flop_count;
    t0 = std::chrono::high_resolution_clock::now();
  }
  void note_tiles(long long tiles) {
    if (!active || tiles <= 0) return;
    tile_accum += tiles;
    if (!noted_tiled) {
      noted_tiled = true;
      entry->tiled_calls += 1;
    }
  }
  ~ScopedBlasProfile() {
    if (!active) return;
    if (noted_tiled) entry->tiles += tile_accum;
    double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    entry->total_ms += ms;
    if (entry->calls == 1) {
      entry->min_ms = entry->max_ms = ms;
    } else {
      entry->min_ms = std::min(entry->min_ms, ms);
      entry->max_ms = std::max(entry->max_ms, ms);
    }
  }
};

static void print_blas_profile(const char* label, const BlasProfileEntry& e) {
  if (e.calls == 0) return;
  double avg_ms = e.total_ms / (double)e.calls;
  double gflops_eff = (e.total_ms > 1e-12) ? (e.total_flops / 1.0e9) / (e.total_ms / 1000.0) : 0.0;
  std::fprintf(stderr,
               "[GPU] profile %-12s calls=%lld avg_ms=%.3f min=%.3f max=%.3f eff_GF/s=%.2f",
               label, e.calls, avg_ms, e.min_ms, e.max_ms, gflops_eff);
  if (e.tiled_calls > 0) {
    double avg_tiles = e.tiles > 0 ? (double)e.tiles / (double)e.tiled_calls : 0.0;
    std::fprintf(stderr, " tiled=%lld avg_tiles=%.2f", e.tiled_calls, avg_tiles);
  }
  std::fprintf(stderr, "\n");
}

static int g_resident_mode = -1; // -1=unset, 0=off, 1=on
static inline bool resident_mode_enabled() {
  if (g_resident_mode >= 0) return g_resident_mode != 0;
  const char* env = std::getenv("MOPAC_RESIDENT_SCF");
  if (env && *env) {
    g_resident_mode = env_truthy_ci(env) ? 1 : 0;
  } else {
    g_resident_mode = 1; // default on when not specified
  }
  return g_resident_mode != 0;
}

extern "C" __global__ void unpack_lower_to_full_kernel(const double *packed, double *full, int n);
extern "C" bool mopac_gpu_cart_gradient_cpu(int numat, int l123, const double *coord, double *grad, const double *qbld);
extern "C" bool mopac_cuda_cart_gradient_launch(int numat, int l123,
                                                const double *coord, double *grad,
                                                const double *charges,
                                                const void *near_pairs, int near_count,
                                                const void *far_pairs, int far_count);

// Simple grow-only device buffer cache helper (C++ only; placed outside C linkage)
template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  size_t cap = 0; // capacity in bytes
  bool ensure(size_t bytes) {
    if (bytes <= cap && ptr) return true;
    if (ptr) cudaFree(ptr);
    ptr = nullptr; cap = 0;
    if (bytes > 0) {
      cudaError_t status = cudaMalloc((void**)&ptr, bytes);
      if (status == cudaSuccess && ptr) {
        cap = bytes;
      } else {
        report_cuda_error("DevBuf::ensure cudaMalloc", status);
        ptr = nullptr;
        return false;
      }
    }
    return true;
  }
  void release() {
    if (ptr) cudaFree(ptr);
    ptr = nullptr; cap = 0;
  }
};

// Simple grow-only pinned host buffer cache
template <typename T>
struct HostBuf {
  T* ptr = nullptr;
  size_t cap = 0; // capacity in bytes
  bool pinned = false;
  void ensure(size_t bytes) {
    if (bytes <= cap && ptr) return;
    if (ptr) {
      if (pinned) cudaFreeHost(ptr); else std::free(ptr);
    }
    ptr = nullptr; cap = 0; pinned = false;
    if (bytes > 0) {
      cudaError_t e = cudaHostAlloc((void**)&ptr, bytes, cudaHostAllocDefault);
      if (e == cudaSuccess && ptr) {
        cap = bytes; pinned = true;
      } else {
        ptr = (T*)std::malloc(bytes);
        cap = ptr ? bytes : 0;
        pinned = false;
      }
    }
  }
  void release() {
    if (ptr) {
      if (pinned) cudaFreeHost(ptr); else std::free(ptr);
    }
    ptr = nullptr; cap = 0; pinned = false;
  }
};


// Default device pair for 2-GPU MOZYME operations
static int g_pair_dev0 = 0;
static int g_pair_dev1 = 1;

extern "C" {

// Configure the default device pair used by 2-GPU MOZYME paths
// Exposed to Fortran via bind(C, name='set_mozyme_gpu_pair') in mod_gpu_info.F90
void set_mozyme_gpu_pair(int dev0, int dev1) {
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count <= 0) {
    // No devices; leave defaults (0,1) as placeholders
    return;
  }
  // Clamp to valid device indices when possible; negative values ignored
  if (dev0 >= 0 && dev0 < count) g_pair_dev0 = dev0;
  if (dev1 >= 0 && dev1 < count) g_pair_dev1 = dev1;
}

// Query basic GPU capabilities
void getGPUInfo(bool *hasGpu,
                bool hasDouble[6],
                int *nDevices,
                char name[6][256],
                int name_size[6],
                size_t totalMem[6],
                int clockRate[6],
                int major[6],
                int minor[6]) {
  int count = 0;
  cudaError_t cerr = cudaGetDeviceCount(&count);
  if (cerr != cudaSuccess || count <= 0) {
    if (hasGpu) *hasGpu = false;
    if (nDevices) *nDevices = 0;
    return;
  }
  if (hasGpu) *hasGpu = true;
  if (nDevices) *nDevices = (count > 6 ? 6 : count);

  for (int i = 0; i < *nDevices; ++i) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, i);
    std::strncpy(name[i], prop.name, 255);
    name[i][255] = '\0';
    name_size[i] = static_cast<int>(std::strlen(name[i]));
    totalMem[i] = prop.totalGlobalMem;
    clockRate[i] = prop.clockRate;
    major[i] = prop.major;
    minor[i] = prop.minor;
    // FP64 support heuristic: CC >= 2.0 generally has native FP64
    hasDouble[i] = (prop.major >= 2);
  }
  for (int i = *nDevices; i < 6; ++i) {
    name[i][0] = '\0';
    name_size[i] = 0;
    totalMem[i] = 0;
    clockRate[i] = 0;
    major[i] = 0;
    minor[i] = 0;
    hasDouble[i] = false;
  }
}

// Query compute capability of current CUDA device (simple helper for policy decisions)
// Set an environment default without overriding a value the user provided.
extern "C" int mopac_setenv_default(const char *name, const char *value) {
#ifdef _WIN32
  (void)name;
  (void)value;
  return -1;
#else
  return setenv(name, value, 0);
#endif
}

void get_current_device_cc(int *major, int *minor) {
  int dev = -1;
  if (major) *major = 0;
  if (minor) *minor = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) return;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return;
  if (major) *major = prop.major;
  if (minor) *minor = prop.minor;
}

bool mopac_cuda_has_cusolvermg() {
#if defined(HAVE_CUSOLVER_MG)
  return true;
#else
  return false;
#endif
}

// Select device by index
void setDevice(int idevice, bool *stat) {
  cudaError_t cerr = cudaSetDevice(idevice);
  if (stat) *stat = (cerr == cudaSuccess);
}

// Global cuBLAS handle
static cublasHandle_t  g_blas = nullptr;
static cublasLtHandle_t g_blasLt = nullptr;
static cublasXtHandle_t g_blasXt = nullptr;
static cudaStream_t   g_stream = nullptr;     // single-GPU general stream
static cudaStream_t   g_stream0 = nullptr;    // 2-GPU device0 stream
static cudaStream_t   g_stream1 = nullptr;    // 2-GPU device1 stream
static bool           g_streams_enabled = true;
static bool           g_pin_user = false;     // Pin user memory for direct H2D/D2H if requested
static cudaStream_t   g_stream_backup = nullptr;
static int            g_stream_override_depth = 0;

static inline void ensure_pair_streams() {
  int dev_count = 0;
  cudaGetDeviceCount(&dev_count);
  if (dev_count <= 0) return;
  if (!g_streams_enabled) return;
  // Device 0 stream
  if (!g_stream0) {
    cudaSetDevice(g_pair_dev0);
    cudaStreamCreate(&g_stream0);
  }
  // Device 1 stream
  if (!g_stream1) {
    cudaSetDevice(g_pair_dev1);
    cudaStreamCreate(&g_stream1);
  }
}

// Cached buffers for single-GPU BLAS wrappers
static DevBuf<double> g_gemm_A, g_gemm_B, g_gemm_C;
static DevBuf<double> g_syrk_A, g_syrk_C;
static HostBuf<double> h_gemm_A, h_gemm_B, h_gemm_C;
static HostBuf<double> h_syrk_A, h_syrk_C;
static DevBuf<double> g_disp_sum2, g_disp_sum3, g_disp_r, g_disp_val, g_disp_der;
static HostBuf<double> h_disp_val, h_disp_der;
static DevBuf<double> g_mz_fock_ptot, g_mz_fock_w, g_mz_fock_out;
static HostBuf<double> h_mz_fock_out;
static DevBuf<int> g_mz_fock1_iab, g_mz_fock1_ilim, g_mz_fock1_pair_off, g_mz_fock1_w_off;
static DevBuf<double> g_mz_fock1_batch_ptot, g_mz_fock1_batch_w, g_mz_fock1_batch_out;
static DevBuf<double> g_mz_fock2_4x1_diag, g_mz_fock2_4x1_light, g_mz_fock2_4x1_cross;
static DevBuf<double> g_mz_fock2_4x1_wj, g_mz_fock2_4x1_wk, g_mz_fock2_4x1_out;
static DevBuf<int> g_mz_fillij_iorbs, g_mz_fillij_nijbo, g_mz_fillij_out;
static DevBuf<double> g_mz_fillij_coord, g_mz_fillij_tvec;
static DevBuf<int> g_mz_res_count_iorbs, g_mz_res_count_kopt;
static DevBuf<int> g_mz_res_count_nijbo, g_mz_res_count_out, g_mz_res_count_fallback;
static DevBuf<int> g_mz_res_pack_nat, g_mz_res_pack_jindex, g_mz_res_pack_status;
static DevBuf<double> g_mz_res_pack_coord, g_mz_res_pack_wj, g_mz_res_pack_wk;
static DevBuf<double> g_mz_res_pack_am, g_mz_res_pack_ad, g_mz_res_pack_dd;
static DevBuf<double> g_mz_res_pack_aq, g_mz_res_pack_qq, g_mz_res_pack_tore;
static DevBuf<double> g_mz_res_pack_po, g_mz_res_pack_ddp;
static DevBuf<double> g_mz_res_pack_direct_scratch;
static DevBuf<int> g_mz_res_pack_iod;
struct MozymeSparseFockPlan {
  DevBuf<int> one_f, one_w, one_iab, one_ilim;
  DevBuf<int> pair_iab, pair_jba, pair_i, pair_j;
  DevBuf<int> pair_cross, pair_diag, pair_w;
  DevBuf<int> pair4_heavy, pair4_light, pair4_cross;
  DevBuf<int> point_iab, point_jba, point_i_atom, point_j_atom;
  DevBuf<int> point_i, point_j, point_addr;
  DevBuf<double> one_w_values, pair_wj, pair_wk;
  DevBuf<double> pair4_wj, pair4_wk, point_w;
  DevBuf<double> ptot, qe, f;
  // Point-pair Fock terms: per-atom accumulators (8 doubles per atom) and the
  // per-atom diagonal-block offset / basis size derived from the point tasks.
  DevBuf<double> point_acc;
  DevBuf<int> point_atom_off, point_atom_iab;
  bool point_index_ready = false;
  int mpack = 0;
  int natoms = 0;
  int one_count = 0;
  int pair_count = 0;
  int pair4_count = 0;
  int point_count = 0;
  int point_dipole_count = 0;
  int point_monopole_count = 0;
  int64_t signature = 0;
  bool ready = false;
  bool full_coverage = false;
  bool has_executable_work = false;

  void release() {
    one_f.release(); one_w.release(); one_iab.release(); one_ilim.release();
    pair_iab.release(); pair_jba.release(); pair_i.release(); pair_j.release();
    pair_cross.release(); pair_diag.release(); pair_w.release();
    pair4_heavy.release(); pair4_light.release(); pair4_cross.release();
    point_iab.release(); point_jba.release();
    point_i_atom.release(); point_j_atom.release();
    point_i.release(); point_j.release(); point_addr.release();
    one_w_values.release(); pair_wj.release(); pair_wk.release();
    pair4_wj.release(); pair4_wk.release(); point_w.release();
    ptot.release(); qe.release(); f.release();
    point_acc.release(); point_atom_off.release(); point_atom_iab.release();
    point_index_ready = false;
    mpack = 0;
    natoms = 0;
    one_count = 0;
    pair_count = 0;
    pair4_count = 0;
    point_count = 0;
    point_dipole_count = 0;
    point_monopole_count = 0;
    signature = 0;
    ready = false;
    full_coverage = false;
    has_executable_work = false;
  }
};

static constexpr int kMozymeSparseFockPlanCount = 2;
static constexpr int kMozymeSparseFockPlanDefault = 0;
static MozymeSparseFockPlan g_mz_res_plans[kMozymeSparseFockPlanCount];

// ---------------------------------------------------------------------------
// Device copy of the MOZYME block index nijbo (numat x numat ints), shared by
// the hcore, Fock-plan and resident-SCF entry points.  fillij is the only
// writer of the host array and calls mopac_cuda_mozyme_nijbo_touch after each
// fill; consumers ask for the device copy and it is re-uploaded only when the
// host array was touched (or is a different array).  For a 7000-atom protein
// the array is 179 MB and used to be uploaded three times per geometry step.
// ---------------------------------------------------------------------------
namespace {
struct MozymeNijboCache {
  DevBuf<int> buf;
  const int *host = nullptr;
  int numat = 0;
  long long gen_host = 0;       // bumped by every touch / new array
  long long gen_uploaded = -1;  // generation currently on the device
};
MozymeNijboCache g_mz_nijbo_cache;
}  // namespace

extern "C" void mopac_cuda_mozyme_nijbo_touch(const int *host, int numat) {
  g_mz_nijbo_cache.host = host;
  g_mz_nijbo_cache.numat = numat;
  ++g_mz_nijbo_cache.gen_host;
}

// Generation of the host array as known to the cache (-1 if unknown).
extern "C" long long mopac_cuda_mozyme_nijbo_generation(const int *host, int numat) {
  if (!host || numat <= 0 || host != g_mz_nijbo_cache.host ||
      numat != g_mz_nijbo_cache.numat) {
    return -1;
  }
  return g_mz_nijbo_cache.gen_host;
}

// Device pointer to the current nijbo; uploads only when stale.  nullptr on failure.
extern "C" const int *mopac_cuda_mozyme_nijbo_device(const int *host, int numat) {
  if (!host || numat <= 0) return nullptr;
  MozymeNijboCache &c = g_mz_nijbo_cache;
  if (host != c.host || numat != c.numat) {
    // Not announced by fillij: treat as a new array (always re-uploaded).
    c.host = host;
    c.numat = numat;
    ++c.gen_host;
  }
  const size_t bytes = sizeof(int) * static_cast<size_t>(numat) * static_cast<size_t>(numat);
  if (c.gen_uploaded == c.gen_host && c.buf.ptr && c.buf.cap >= bytes) return c.buf.ptr;
  if (!c.buf.ensure(bytes)) return nullptr;
  cudaStream_t s = g_stream ? g_stream : 0;
  if (cudaMemcpyAsync(c.buf.ptr, host, bytes, cudaMemcpyHostToDevice, s) != cudaSuccess ||
      cudaStreamSynchronize(s) != cudaSuccess) {
    c.gen_uploaded = -1;
    return nullptr;
  }
  c.gen_uploaded = c.gen_host;
  return c.buf.ptr;
}

static inline MozymeSparseFockPlan *mozyme_sparse_fock_plan(int plan_id) {
  if (plan_id < 0 || plan_id >= kMozymeSparseFockPlanCount) return nullptr;
  return &g_mz_res_plans[plan_id];
}

static inline void mozyme_sparse_fock_invalidate_plan(MozymeSparseFockPlan *plan) {
  if (!plan) return;
  plan->point_index_ready = false;
  plan->mpack = 0;
  plan->natoms = 0;
  plan->one_count = 0;
  plan->pair_count = 0;
  plan->pair4_count = 0;
  plan->point_count = 0;
  plan->point_dipole_count = 0;
  plan->point_monopole_count = 0;
  plan->signature = 0;
  plan->ready = false;
  plan->full_coverage = false;
  plan->has_executable_work = false;
}

extern "C" int mopac_cuda_mozyme_sparse_fock_plan_ready(int plan_id,
                                                         int mpack,
                                                         int full_coverage_required,
                                                         int64_t signature) {
  const MozymeSparseFockPlan *plan = mozyme_sparse_fock_plan(plan_id);
  if (!plan || mpack <= 0) return 0;
  if (full_coverage_required != 0 && !plan->full_coverage) return 0;
  return (plan->ready && plan->has_executable_work &&
          plan->mpack == mpack && plan->signature == signature)
             ? 1
             : 0;
}
static DevBuf<double> g_mz_fock2_pii, g_mz_fock2_pjj, g_mz_fock2_pij;
static DevBuf<double> g_mz_fock2_wj, g_mz_fock2_wk;
static DevBuf<double> g_mz_fock2_fii, g_mz_fock2_fjj, g_mz_fock2_fij;
static HostBuf<double> h_mz_fock2_fii, h_mz_fock2_fjj, h_mz_fock2_fij;
static DevBuf<double> g_mz_dfock2_fii, g_mz_dfock2_fjj, g_mz_dfock2_fij;
static HostBuf<double> h_mz_dfock2_fii, h_mz_dfock2_fjj, h_mz_dfock2_fij;
// 2-GPU caches
static DevBuf<double> g2_gemm_a0, g2_gemm_b0, g2_gemm_c0;
static DevBuf<double> g2_gemm_a1, g2_gemm_b1, g2_gemm_c1;
static DevBuf<double> g2_syrk_v0, g2_syrk_c0;
static DevBuf<double> g2_syrk_v1, g2_syrk_c1;
static HostBuf<double> h2_gemm_A, h2_gemm_B, h2_gemm_C;
static HostBuf<double> h2_syrk_A, h2_syrk_C;
static HostBuf<double> h2_rot_V;

static DevBuf<uint8_t> g_lt_workspace;

struct HmtrStreamSlot {
  int device = -1;
  int thread_id = -1;
  cudaStream_t stream = nullptr;
};

static std::vector<HmtrStreamSlot> g_hmtr_stream_slots;
static std::mutex g_hmtr_stream_mutex;

void mopac_cuda_hmtr_bind_thread(int device,
                                 int thread_id,
                                 void **stream_out,
                                 int *device_changed) {
  if (stream_out) *stream_out = nullptr;
  if (device_changed) *device_changed = 0;
  if (device < 0) return;
  if (thread_id < 0) thread_id = 0;

  std::lock_guard<std::mutex> guard(g_hmtr_stream_mutex);

  int current_dev = -1;
  cudaGetDevice(&current_dev);
  if (current_dev != device) {
    if (cudaSetDevice(device) == cudaSuccess) {
      if (device_changed) *device_changed = 1;
    }
  }

  for (auto &slot : g_hmtr_stream_slots) {
    if (slot.device == device && slot.thread_id == thread_id) {
      if (stream_out) *stream_out = reinterpret_cast<void*>(slot.stream);
      return;
    }
  }

  cudaStream_t stream = nullptr;
  cudaError_t cerr = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cerr != cudaSuccess || !stream) {
    return;
  }

  HmtrStreamSlot slot;
  slot.device = device;
  slot.thread_id = thread_id;
  slot.stream = stream;
  g_hmtr_stream_slots.push_back(slot);
  if (stream_out) *stream_out = reinterpret_cast<void*>(stream);
}

void mopac_cuda_hmtr_clear_streams(void) {
  std::lock_guard<std::mutex> guard(g_hmtr_stream_mutex);
  int saved_device = -1;
  cudaGetDevice(&saved_device);
  for (auto &slot : g_hmtr_stream_slots) {
    cudaSetDevice(slot.device);
    if (slot.stream) cudaStreamDestroy(slot.stream);
  }
  g_hmtr_stream_slots.clear();
  if (saved_device >= 0) cudaSetDevice(saved_device);
}

// Density residency cache (full matrix + packed upper triangle)
static DevBuf<double> g_density_full;
static int g_density_full_n = 0;
static int g_density_full_ld = 0;
static bool g_density_full_valid = false;

struct PackedDensitySlot {
  DevBuf<double> buf;
  size_t len = 0;
  const double* host_ptr = nullptr;
  bool valid = false;
  unsigned long long stamp = 0;
};

static constexpr int kPackedDensitySlots = 4;
static std::array<PackedDensitySlot, kPackedDensitySlots> g_packed_density{};
static unsigned long long g_packed_density_tick = 0;

static inline void invalidate_packed_density() {
  for (auto &slot : g_packed_density) {
    slot.valid = false;
    slot.len = 0;
    slot.stamp = 0;
  }
}

static PackedDensitySlot* find_packed_slot(const double *host_ptr, size_t len) {
  PackedDensitySlot *fallback = nullptr;
  for (auto &slot : g_packed_density) {
    if (!slot.valid) continue;
    if (slot.len != len) continue;
    if (host_ptr && slot.host_ptr == host_ptr) return &slot;
    if (!fallback || slot.stamp > fallback->stamp) fallback = &slot;
  }
  return fallback;
}

static bool resident_debug_enabled() {
  static int inited = 0;
  static bool enabled = false;
  if (!inited) {
    const char *s = std::getenv("MOPAC_GPU_RESIDENT_DEBUG");
    if (s && *s) {
      if (!(std::strcmp(s, "0") == 0 || std::strcmp(s, "off") == 0 ||
            std::strcmp(s, "false") == 0 || std::strcmp(s, "n") == 0 ||
            std::strcmp(s, "N") == 0)) {
        enabled = true;
      }
    }
    inited = 1;
  }
  return enabled;
}

static PackedDensitySlot* acquire_packed_slot(const double *host_ptr) {
  if (host_ptr) {
    for (auto &slot : g_packed_density) {
      if (slot.host_ptr == host_ptr) return &slot;
    }
  }
  for (auto &slot : g_packed_density) {
    if (!slot.valid && slot.host_ptr == nullptr) {
      slot.host_ptr = host_ptr;
      return &slot;
    }
  }
  PackedDensitySlot *victim = &g_packed_density[0];
  for (auto &slot : g_packed_density) {
    if (slot.stamp < victim->stamp) victim = &slot;
  }
  victim->host_ptr = host_ptr;
  victim->valid = false;
  victim->len = 0;
  victim->stamp = 0;
  return victim;
}

struct PackedCache {
  DevBuf<double> buf;
  size_t len = 0;
  const double* host_ptr = nullptr;
  bool valid = false;
};

static PackedCache g_fock_cache;

static inline void invalidate_fock_cache() {
  g_fock_cache.valid = false;
  g_fock_cache.host_ptr = nullptr;
  g_fock_cache.len = 0;
}

static void register_fock_cache(int linear, const double *host_ptr, const double *src_dev) {
  if (!resident_mode_enabled()) return;
  if (linear <= 0 || !src_dev) {
    invalidate_fock_cache();
    return;
  }
  size_t bytes = sizeof(double) * (size_t)linear;
  cudaStream_t s = g_stream ? g_stream : 0;
  g_fock_cache.buf.ensure(bytes);
  cudaMemcpyAsync(g_fock_cache.buf.ptr, src_dev, bytes, cudaMemcpyDeviceToDevice, s);
  cudaStreamSynchronize(s);
  g_fock_cache.len = (size_t)linear;
  g_fock_cache.host_ptr = host_ptr;
  g_fock_cache.valid = true;
  if (resident_debug_enabled() && host_ptr) {
    std::vector<double> src_copy(linear);
    if (cudaMemcpy(src_copy.data(), src_dev, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
      double max_src = 0.0;
      double rms_src = 0.0;
      for (int i = 0; i < linear; ++i) {
        double diff = src_copy[i] - host_ptr[i];
        if (std::abs(diff) > max_src) max_src = std::abs(diff);
        rms_src += diff * diff;
      }
      if (linear > 0) rms_src = std::sqrt(rms_src / (double)linear);
      std::printf("[GPU resident debug] fock src compare max=% .5e rms=% .5e\n", max_src, rms_src);
      if (max_src > 1e-6) {
        int limit = std::min(linear, 5);
        std::printf("  src host vs device:");
        for (int i = 0; i < limit; ++i) {
          std::printf(" (% .5e,% .5e)", host_ptr[i], src_copy[i]);
        }
        std::printf("\n");
      }
      std::fflush(stdout);
    }
    std::vector<double> host_copy(linear);
    if (cudaMemcpy(host_copy.data(), g_fock_cache.buf.ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
      double max_diff = 0.0;
      double rms = 0.0;
      for (int i = 0; i < linear; ++i) {
        double diff = host_copy[i] - host_ptr[i];
        if (std::abs(diff) > max_diff) max_diff = std::abs(diff);
        rms += diff * diff;
      }
      if (linear > 0) rms = std::sqrt(rms / (double)linear);
      std::printf("[GPU resident debug] fock register max=% .5e rms=% .5e\n", max_diff, rms);
      if (max_diff > 1e-6) {
        int limit = std::min(linear, 5);
        std::printf("  sample host vs device:");
        for (int i = 0; i < limit; ++i) {
          std::printf(" (% .5e,% .5e)", host_ptr[i], host_copy[i]);
        }
        std::printf("\n");
      }
      std::fflush(stdout);
    }
  }
}

// cuSOLVERMg profiling accumulators (populated when requested)
static long long mg_calls = 0;
static long long mg_failures = 0;
static double mg_total_ms = 0.0;
static long long mg_total_dim = 0;
static long long mg_total_devices = 0;
static int mg_profile_env_requested = 0;
#if defined(HAVE_CUSOLVER_MG)
static int mg_profile_flag = 0;
static int mg_profile_inited = 0;
static inline bool mg_profile_enabled() {
  if (!mg_profile_inited) {
    const char* s = std::getenv("MOPAC_EIG_MG_PROFILE");
    if (s && *s) {
      if (std::strcmp(s, "0") == 0 || std::strcmp(s, "off") == 0 || std::strcmp(s, "false") == 0) {
        mg_profile_flag = 0;
        mg_profile_env_requested = 0;
      } else {
        mg_profile_flag = 1;
        mg_profile_env_requested = 1;
      }
    } else {
      mg_profile_flag = 0;
      mg_profile_env_requested = 0;
    }
    mg_profile_inited = 1;
  }
  return mg_profile_flag != 0;
}
#else
#define mg_profile_enabled() (false)
#endif

// Prefer cuBLASLt for DGEMM when it can choose a valid FP64 algorithm, then
// fall back to classic cuBLAS DGEMM in the caller for portability.
static bool lt_dgemm(cublasOperation_t opA, cublasOperation_t opB,
                     int m, int n, int k,
                     double alpha,
                     const double *d_A, int lda,
                     const double *d_B, int ldb,
                     double beta,
                     double *d_C, int ldc) {
  if (!g_blasLt) return false;
#if CUBLAS_VERSION < 11700
  return false;
#else
  cublasStatus_t st;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr, layoutD = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  bool success = false;

  do {
    st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_64F, CUDA_R_64F);
    if (st != CUBLAS_STATUS_SUCCESS) break;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    int64_t rowsA = (opA == CUBLAS_OP_N) ? m : k;
    int64_t colsA = (opA == CUBLAS_OP_N) ? k : m;
    int64_t rowsB = (opB == CUBLAS_OP_N) ? k : n;
    int64_t colsB = (opB == CUBLAS_OP_N) ? n : k;
    int64_t rowsC = m;
    int64_t colsC = n;

    st = cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_64F, rowsA, colsA, lda);
    if (st != CUBLAS_STATUS_SUCCESS) break;
    st = cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_64F, rowsB, colsB, ldb);
    if (st != CUBLAS_STATUS_SUCCESS) break;
    st = cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_64F, rowsC, colsC, ldc);
    if (st != CUBLAS_STATUS_SUCCESS) break;
    st = cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_64F, rowsC, colsC, ldc);
    if (st != CUBLAS_STATUS_SUCCESS) break;

    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(layoutD, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    st = cublasLtMatmulPreferenceCreate(&pref);
    if (st != CUBLAS_STATUS_SUCCESS) break;

    size_t workspace_limit = 1ULL << 23; // 8 MB
    g_lt_workspace.ensure(workspace_limit);
    cublasLtMatmulPreferenceSetAttribute(pref,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspace_limit,
                                         sizeof(workspace_limit));

    const int requestedAlgoCount = 8;
    cublasLtMatmulHeuristicResult_t heuristics[requestedAlgoCount];
    int returnCount = 0;
    st = cublasLtMatmulAlgoGetHeuristic(g_blasLt,
                                        op_desc,
                                        layoutA,
                                        layoutB,
                                        layoutC,
                                        layoutD,
                                        pref,
                                        requestedAlgoCount,
                                        heuristics,
                                        &returnCount);
    if (st != CUBLAS_STATUS_SUCCESS || returnCount == 0) break;

    for (int idx = 0; idx < returnCount; ++idx) {
      if (heuristics[idx].state != CUBLAS_STATUS_SUCCESS) continue;
      if (heuristics[idx].workspaceSize > workspace_limit) continue;
      st = cublasLtMatmul(g_blasLt,
                          op_desc,
                          &alpha,
                          d_A, layoutA,
                          d_B, layoutB,
                          &beta,
                          d_C, layoutC,
                          d_C, layoutD,
                          &heuristics[idx].algo,
                          g_lt_workspace.ptr,
                          heuristics[idx].workspaceSize,
                          g_stream ? g_stream : 0);
      if (st == CUBLAS_STATUS_SUCCESS) {
        success = true;
        cudaStreamSynchronize(g_stream ? g_stream : 0);
        break;
      }
    }
  } while (0);

  if (pref) cublasLtMatmulPreferenceDestroy(pref);
  if (layoutD) cublasLtMatrixLayoutDestroy(layoutD);
  if (layoutC) cublasLtMatrixLayoutDestroy(layoutC);
  if (layoutB) cublasLtMatrixLayoutDestroy(layoutB);
  if (layoutA) cublasLtMatrixLayoutDestroy(layoutA);
  if (op_desc) cublasLtMatmulDescDestroy(op_desc);
  return success;
}
#endif

void create_handle() {
  if (!g_blas) {
    if (!report_cublas_error("cublasCreate", cublasCreate(&g_blas))) {
      g_blas = nullptr;
      return;
    }
    // Enforce deterministic behavior if requested
    const char* det = std::getenv("MOPAC_DETERMINISTIC");
    if (det && (std::strcmp(det, "1") == 0 || std::strcmp(det, "on") == 0 || std::strcmp(det, "true") == 0)) {
#if defined(CUBLAS_VERSION) && (CUBLAS_VERSION >= 11000)
      cublasSetAtomicsMode(g_blas, CUBLAS_ATOMICS_NOT_ALLOWED);
#endif
      cublasSetPointerMode(g_blas, CUBLAS_POINTER_MODE_HOST);
#if defined(CUBLAS_VERSION)
      cublasSetMathMode(g_blas, CUBLAS_DEFAULT_MATH);
#endif
    }
    if (!g_blasLt) {
      if (cublasLtCreate(&g_blasLt) != CUBLAS_STATUS_SUCCESS) {
        g_blasLt = nullptr;
      }
    }
    const char* env = std::getenv("MOPAC_STREAMS");
    if (env) {
      if (!env_truthy_ci(env)) {
        g_streams_enabled = false;
      }
    }
    const char* env_pin = std::getenv("MOPAC_PIN_USER");
    if (env_pin) {
      if (env_true_ci(env_pin)) {
        g_pin_user = true;
      }
    }
    if (g_streams_enabled) {
      if (!g_stream) {
        cudaStreamCreate(&g_stream);
      }
      if (g_stream) report_cublas_error("cublasSetStream", cublasSetStream(g_blas, g_stream));
    }
  }
}

static inline void create_handle_xt() {
  if (!g_blasXt) {
    if (!report_cublas_error("cublasXtCreate", cublasXtCreate(&g_blasXt))) {
      g_blasXt = nullptr;
      return;
    }
    // Optional block size tuning for cuBLASXt (default picked by library)
    const char* blk = std::getenv("MOPAC_CUBLASXT_BLOCK");
    if (blk) {
      int b = std::max(64, std::atoi(blk));
      cublasXtSetBlockDim(g_blasXt, b);
    }
    // Optional CPU ratio (0.0 .. 1.0), 0 means pure GPU
    // NOTE: cublasXtSetCpuRatio signature varies across versions; skip configuring it for portability.
    int devCount = 0; cudaGetDeviceCount(&devCount);
    int devList[8]; int nDevs = 0;
    const char* list = std::getenv("MOPAC_CUBLASXT_DEVICES");
    if (list && devCount > 0) {
      char buf[256]; std::strncpy(buf, list, sizeof(buf)-1); buf[sizeof(buf)-1] = '\0';
      char* tok = std::strtok(buf, ",; :");
      while (tok && nDevs < 8) {
        int d = std::atoi(tok);
        if (d >= 0 && d < devCount) { devList[nDevs++] = d; }
        tok = std::strtok(nullptr, ",; :");
      }
    }
    if (nDevs == 0 && devCount > 0) {
      // Auto-select up to 8 devices, ordered by capability and size
      int cand = std::min(devCount, 8);
      int order[8];
      for (int i = 0; i < cand; ++i) order[i] = i;
      // Simple selection sort by (major, multiprocessors, totalGlobalMem)
      for (int i = 0; i < cand; ++i) {
        int best = i;
        cudaDeviceProp propBest{}; cudaGetDeviceProperties(&propBest, order[best]);
        for (int j = i+1; j < cand; ++j) {
          cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, order[j]);
          bool better = (prop.major > propBest.major) ||
                        (prop.major == propBest.major && prop.multiProcessorCount > propBest.multiProcessorCount) ||
                        (prop.major == propBest.major && prop.multiProcessorCount == propBest.multiProcessorCount && prop.totalGlobalMem > propBest.totalGlobalMem);
          if (better) { best = j; propBest = prop; }
        }
        int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
      }
      for (int i = 0; i < cand; ++i) devList[nDevs++] = order[i];
    }
    if (nDevs > 0) {
      cublasXtDeviceSelect(g_blasXt, nDevs, devList);
    }
  }
}

void destroy_handle() {
  if (g_blas) {
    cublasDestroy(g_blas);
    g_blas = nullptr;
  }
  if (g_blasLt) {
    cublasLtDestroy(g_blasLt);
    g_blasLt = nullptr;
  }
  if (g_blasXt) {
    cublasXtDestroy(g_blasXt);
    g_blasXt = nullptr;
  }
  if (g_stream) {
    cudaStreamDestroy(g_stream);
    g_stream = nullptr;
  }
  if (g_stream0) { cudaSetDevice(g_pair_dev0); cudaStreamDestroy(g_stream0); g_stream0 = nullptr; }
  if (g_stream1) { cudaSetDevice(g_pair_dev1); cudaStreamDestroy(g_stream1); g_stream1 = nullptr; }
}

// Cleanup function moved to the end of translation unit (after all static declarations)

// Fortran-callable FP64 GEMM. The wrapper owns device staging and library
// selection so Fortran callers can keep the same column-major BLAS contract.
void call_gemm_cublas(char tra, char trb,
                      int m, int n, int k,
                      double alpha,
                      const double *A, int lda,
                      const double *B, int ldb,
                      double beta,
                      double *C, int ldc) {
  ensure_w_verbose();
  char nv_name[64];
  const char* nv_ptr = nullptr;
  if (gpu_profile_enabled()) {
    std::snprintf(nv_name, sizeof(nv_name), "GEMM %dx%dx%d", m, n, k);
    nv_ptr = nv_name;
  }
  NvtxRange nv_scope(nv_ptr, 0xFF1F77B4);
  ScopedBlasProfile prof_scope(&g_prof_gemm_single, 2.0 * (double)m * (double)n * (double)k);
  if (!g_blas) create_handle();
  if (!g_blas) {
    poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
    return;
  }
  cublasOperation_t opA = (tra == 'T' || tra == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = (trb == 'T' || trb == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  size_t bytesA = (size_t)lda * (size_t)k * sizeof(double);
  size_t bytesB = (size_t)ldb * (size_t)n * sizeof(double);
  size_t bytesC = (size_t)ldc * (size_t)n * sizeof(double);

  size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t reserve = (size_t)(free_mem * 0.8);
  bool can_tile = (opA == CUBLAS_OP_N && opB == CUBLAS_OP_N);
  bool use_tiling = false;
  int tile_n = n;
  if (reserve > 0 && can_tile) {
    if (bytesA + bytesB + bytesC > reserve && bytesA < reserve) {
      size_t span = reserve - bytesA;
      size_t denom = ((size_t)ldb + (size_t)ldc) * sizeof(double);
      size_t max_tile = denom ? span / denom : 0;
      if (max_tile == 0 && span > 0) max_tile = 1;
      if (max_tile > 0 && max_tile < (size_t)n) {
        tile_n = (int)max_tile;
        if (tile_n < 1) tile_n = 1;
        use_tiling = true;
      }
    }
  }

  if (!use_tiling) {
    g_gemm_A.ensure(bytesA);
    g_gemm_B.ensure(bytesB);
    g_gemm_C.ensure(bytesC);
    double *d_A = g_gemm_A.ptr;
    double *d_B = g_gemm_B.ptr;
    double *d_C = g_gemm_C.ptr;
    if (!d_A || !d_B || !d_C) {
      poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
      return;
    }
    bool ok = true;
    bool pinned = false;
    auto cleanup_pinned = [&]() {
      if (!pinned) return;
      cudaHostUnregister((void*)A);
      cudaHostUnregister((void*)B);
      if (beta != 0.0) cudaHostUnregister((void*)C);
      pinned = false;
    };
    if (g_pin_user) {
      if (cudaHostRegister((void*)A, bytesA, cudaHostRegisterDefault) == cudaSuccess &&
          cudaHostRegister((void*)B, bytesB, cudaHostRegisterDefault) == cudaSuccess) {
        if (beta != 0.0) {
          if (cudaHostRegister((void*)C, bytesC, cudaHostRegisterDefault) == cudaSuccess) {
            pinned = true;
          } else {
            cudaHostUnregister((void*)A);
            cudaHostUnregister((void*)B);
          }
        } else {
          pinned = true;
        }
      }
    }
    if (pinned) {
      ok = report_cuda_error("DGEMM copy A host->device",
                             cudaMemcpyAsync(d_A, A, bytesA, cudaMemcpyHostToDevice, g_stream)) && ok;
      ok = report_cuda_error("DGEMM copy B host->device",
                             cudaMemcpyAsync(d_B, B, bytesB, cudaMemcpyHostToDevice, g_stream)) && ok;
      if (beta != 0.0) {
        ok = report_cuda_error("DGEMM copy C host->device",
                               cudaMemcpyAsync(d_C, C, bytesC, cudaMemcpyHostToDevice, g_stream)) && ok;
      }
    } else {
      h_gemm_A.ensure(bytesA);
      h_gemm_B.ensure(bytesB);
      h_gemm_C.ensure(bytesC);
      if (!h_gemm_A.ptr || !h_gemm_B.ptr || !h_gemm_C.ptr) {
        poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
        return;
      }
      std::memcpy(h_gemm_A.ptr, A, bytesA);
      std::memcpy(h_gemm_B.ptr, B, bytesB);
      ok = report_cuda_error("DGEMM copy A host->device",
                             cudaMemcpyAsync(d_A, h_gemm_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream)) && ok;
      ok = report_cuda_error("DGEMM copy B host->device",
                             cudaMemcpyAsync(d_B, h_gemm_B.ptr, bytesB, cudaMemcpyHostToDevice, g_stream)) && ok;
      if (beta != 0.0) {
        std::memcpy(h_gemm_C.ptr, C, bytesC);
        ok = report_cuda_error("DGEMM copy C host->device",
                               cudaMemcpyAsync(d_C, h_gemm_C.ptr, bytesC, cudaMemcpyHostToDevice, g_stream)) && ok;
      }
    }
    if (!ok) {
      poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
      cleanup_pinned();
      return;
    }

    bool lt_used = lt_dgemm(opA, opB, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    if (lt_used && w_verbose) {
      std::fprintf(stderr, "[GPU] DGEMM %dx%dx%d: cuBLASLt path\n", m, n, k);
    }

    if (!lt_used) {
      float ms = 0.0f; cudaEvent_t ev0 = nullptr, ev1 = nullptr;
      cudaStream_t s = g_stream ? g_stream : 0;
      if (w_verbose) {
        if (cudaEventCreate(&ev0) != cudaSuccess) { ev0 = nullptr; }
        if (cudaEventCreate(&ev1) != cudaSuccess) { if (ev0) cudaEventDestroy(ev0); ev0 = nullptr; ev1 = nullptr; }
        if (ev0) cudaEventRecord(ev0, s);
      }
      cublasStatus_t st = cublasDgemm(g_blas, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
      ok = report_cublas_error("cublasDgemm", st) && ok;
      if (w_verbose && ev0 && ev1 && st == CUBLAS_STATUS_SUCCESS) {
        cudaEventRecord(ev1, s);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&ms, ev0, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        double flops = 2.0 * (double)m * (double)n * (double)k;
        double gflops = flops / 1.0e9 / (ms/1000.0);
        std::fprintf(stderr, "[GPU] DGEMM %dx%dx%d: %.3f ms, %.1f GF/s\n", m, n, k, ms, gflops);
      } else if (w_verbose) {
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
      }
    }
    if (!ok) {
      poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
      cleanup_pinned();
      return;
    }

    if (pinned) {
      ok = report_cuda_error("DGEMM copy C device->host",
                             cudaMemcpyAsync(C, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream)) && ok;
      ok = report_cuda_error("DGEMM stream synchronize", cudaStreamSynchronize(g_stream)) && ok;
      cleanup_pinned();
    } else {
      ok = report_cuda_error("DGEMM copy C device->host",
                             cudaMemcpyAsync(h_gemm_C.ptr, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream)) && ok;
      ok = report_cuda_error("DGEMM stream synchronize", cudaStreamSynchronize(g_stream)) && ok;
      if (ok) std::memcpy(C, h_gemm_C.ptr, bytesC);
    }
    if (!ok) poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
  } else {
    if (w_verbose) {
      std::fprintf(stderr, "[GPU] DGEMM %dx%dx%d: tiled columns (tile_n=%d)\n", m, n, k, tile_n);
    }
    if (bytesA > reserve || tile_n <= 0) {
      g_gemm_A.ensure(bytesA);
      g_gemm_B.ensure(bytesB);
      g_gemm_C.ensure(bytesC);
      double *d_A = g_gemm_A.ptr;
      double *d_B = g_gemm_B.ptr;
      double *d_C = g_gemm_C.ptr;
      h_gemm_A.ensure(bytesA);
      h_gemm_B.ensure(bytesB);
      h_gemm_C.ensure(bytesC);
      if (!d_A || !d_B || !d_C || !h_gemm_A.ptr || !h_gemm_B.ptr || !h_gemm_C.ptr) {
        poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
        return;
      }
      bool ok = true;
      std::memcpy(h_gemm_A.ptr, A, bytesA);
      std::memcpy(h_gemm_B.ptr, B, bytesB);
      ok = report_cuda_error("DGEMM tiled fallback copy A host->device",
                             cudaMemcpyAsync(d_A, h_gemm_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream)) && ok;
      ok = report_cuda_error("DGEMM tiled fallback copy B host->device",
                             cudaMemcpyAsync(d_B, h_gemm_B.ptr, bytesB, cudaMemcpyHostToDevice, g_stream)) && ok;
      if (beta != 0.0) {
        std::memcpy(h_gemm_C.ptr, C, bytesC);
        ok = report_cuda_error("DGEMM tiled fallback copy C host->device",
                               cudaMemcpyAsync(d_C, h_gemm_C.ptr, bytesC, cudaMemcpyHostToDevice, g_stream)) && ok;
      }
      ok = report_cublas_error("cublasDgemm tiled fallback",
                               cublasDgemm(g_blas, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc)) && ok;
      ok = report_cuda_error("DGEMM tiled fallback copy C device->host",
                             cudaMemcpyAsync(h_gemm_C.ptr, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream)) && ok;
      ok = report_cuda_error("DGEMM tiled fallback stream synchronize", cudaStreamSynchronize(g_stream)) && ok;
      if (ok) {
        std::memcpy(C, h_gemm_C.ptr, bytesC);
      } else {
        poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
      }
      return;
    }

    size_t bytesA_tile = (size_t)lda * (size_t)k * sizeof(double);
    g_gemm_A.ensure(bytesA_tile);
    double *d_A = g_gemm_A.ptr;
    h_gemm_A.ensure(bytesA_tile);
    if (!d_A || !h_gemm_A.ptr) {
      poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
      return;
    }
    bool ok = true;
    std::memcpy(h_gemm_A.ptr, A, bytesA_tile);
    ok = report_cuda_error("DGEMM tiled copy A host->device",
                           cudaMemcpyAsync(d_A, h_gemm_A.ptr, bytesA_tile, cudaMemcpyHostToDevice, g_stream)) && ok;
    ok = report_cuda_error("DGEMM tiled copy A stream synchronize", cudaStreamSynchronize(g_stream)) && ok;
    if (!ok) {
      poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
      return;
    }

    long long tile_chunks = 0;
    for (int col0 = 0; col0 < n; col0 += tile_n) {
      tile_chunks++;
      int tn = std::min(tile_n, n - col0);
      size_t bytesB_tile = (size_t)ldb * (size_t)tn * sizeof(double);
      size_t bytesC_tile = (size_t)ldc * (size_t)tn * sizeof(double);
      g_gemm_B.ensure(bytesB_tile);
      g_gemm_C.ensure(bytesC_tile);
      double *d_B = g_gemm_B.ptr;
      double *d_C = g_gemm_C.ptr;
      if (!d_B || !d_C) {
        ok = false;
        break;
      }

      const double *B_tile = B + (size_t)col0 * (size_t)ldb;
      double *C_tile = C + (size_t)col0 * (size_t)ldc;

      ok = report_cuda_error("DGEMM tiled copy B host->device",
                             cudaMemcpy2DAsync(d_B, ldb * sizeof(double),
                                               B_tile, ldb * sizeof(double),
                                               (size_t)tn * sizeof(double), (size_t)k,
                                               cudaMemcpyHostToDevice, g_stream)) && ok;
      if (beta != 0.0) {
        ok = report_cuda_error("DGEMM tiled copy C host->device",
                               cudaMemcpy2DAsync(d_C, ldc * sizeof(double),
                                                 C_tile, ldc * sizeof(double),
                                                 (size_t)tn * sizeof(double), (size_t)m,
                                                 cudaMemcpyHostToDevice, g_stream)) && ok;
      }
      cublasStatus_t st = cublasDgemm(g_blas, opA, opB, m, tn, k,
                                      &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
      ok = report_cublas_error("cublasDgemm tiled", st) && ok;
      if (w_verbose && st == CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[GPU] DGEMM tile m=%d n=%d k=%d\n", m, tn, k);
      }
      ok = report_cuda_error("DGEMM tiled copy C device->host",
                             cudaMemcpy2DAsync(C_tile, ldc * sizeof(double),
                                               d_C, ldc * sizeof(double),
                                               (size_t)tn * sizeof(double), (size_t)m,
                                               cudaMemcpyDeviceToHost, g_stream)) && ok;
      ok = report_cuda_error("DGEMM tiled stream synchronize", cudaStreamSynchronize(g_stream)) && ok;
      if (!ok) break;
    }
    prof_scope.note_tiles(tile_chunks);
    if (!ok) poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
  }
}

// Fortran-callable FP64 SYRK via cuBLAS. Large panels are tiled only to fit
// device memory; the math operation remains the vendor DSYRK implementation.
void call_syrk_cublas(char uplo, char tra,
                      int n, int k,
                      double alpha,
                      const double *A, int lda,
                      double beta,
                      double *C, int ldc) {
  ensure_w_verbose();
  char nv_name[64];
  const char* nv_ptr = nullptr;
  if (gpu_profile_enabled()) {
    std::snprintf(nv_name, sizeof(nv_name), "SYRK n=%d k=%d", n, k);
    nv_ptr = nv_name;
  }
  NvtxRange nv_scope(nv_ptr, 0xFF2CA02C);
  ScopedBlasProfile prof_scope(&g_prof_syrk_single, 2.0 * (double)n * (double)n * (double)k);
  if (!g_blas) create_handle();
  if (!g_blas) {
    poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
    return;
  }
  cublasFillMode_t u = (uplo == 'U' || uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t opA = (tra == 'T' || tra == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  size_t bytesA = (size_t)lda * (size_t)((opA==CUBLAS_OP_N)?k:n) * sizeof(double);
  size_t bytesC = (size_t)ldc * (size_t)n * sizeof(double);

  size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t reserve = (size_t)(free_mem * 0.8);
  bool can_tile = (opA == CUBLAS_OP_N);
  bool use_tiling = false;
  int tile_k = k;
  if (reserve > 0 && can_tile) {
    if (bytesA + bytesC > reserve && bytesC < reserve) {
      size_t span = reserve - bytesC;
      size_t denom = (size_t)lda * sizeof(double);
      size_t max_tile = denom ? span / denom : 0;
      if (max_tile == 0 && span > 0) max_tile = 1;
      if (max_tile > 0 && max_tile < (size_t)k) {
        tile_k = (int)max_tile;
        if (tile_k < 1) tile_k = 1;
        use_tiling = true;
      }
    }
  }

  g_syrk_C.ensure(bytesC);
  double *d_C = g_syrk_C.ptr;
  cudaStream_t stream = g_stream ? g_stream : 0;
  if (!d_C) {
    poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
    return;
  }

  if (!use_tiling) {
    g_syrk_A.ensure(bytesA);
    double *d_A = g_syrk_A.ptr;
    if (!d_A) {
      poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
      return;
    }
    bool ok = true;
    ok = report_cuda_error("DSYRK copy A host->device",
                           cudaMemcpyAsync(d_A, A, bytesA, cudaMemcpyHostToDevice, stream)) && ok;
    if (beta != 0.0) {
      ok = report_cuda_error("DSYRK copy C host->device",
                             cudaMemcpyAsync(d_C, C, bytesC, cudaMemcpyHostToDevice, stream)) && ok;
    }
    float ms = 0.0f; cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (w_verbose) {
      if (cudaEventCreate(&ev0) != cudaSuccess) { ev0 = nullptr; }
      if (cudaEventCreate(&ev1) != cudaSuccess) { if (ev0) cudaEventDestroy(ev0); ev0 = nullptr; ev1 = nullptr; }
      if (ev0) cudaEventRecord(ev0, stream);
    }
    cublasStatus_t st2 = cublasDsyrk(g_blas, u, opA, n, k, &alpha, d_A, lda, &beta, d_C, ldc);
    ok = report_cublas_error("cublasDsyrk", st2) && ok;
    if (w_verbose && ev0 && ev1 && st2 == CUBLAS_STATUS_SUCCESS) {
      cudaEventRecord(ev1, stream);
      cudaEventSynchronize(ev1);
      cudaEventElapsedTime(&ms, ev0, ev1);
      cudaEventDestroy(ev0); cudaEventDestroy(ev1);
      double flops = 2.0 * (double)n * (double)n * (double)k; // rough upper-bound
      double gflops = flops / 1.0e9 / (ms/1000.0);
      std::fprintf(stderr, "[GPU] DSYRK n=%d k=%d: %.3f ms, %.1f GF/s\n", n,k, ms, gflops);
    } else if (w_verbose) {
      if (ev0) cudaEventDestroy(ev0);
      if (ev1) cudaEventDestroy(ev1);
    }
    ok = report_cuda_error("DSYRK copy C device->host",
                           cudaMemcpyAsync(C, d_C, bytesC, cudaMemcpyDeviceToHost, stream)) && ok;
    ok = report_cuda_error("DSYRK stream synchronize", cudaStreamSynchronize(stream)) && ok;
    if (!ok) poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
  } else {
    if (w_verbose) {
      std::fprintf(stderr, "[GPU] DSYRK n=%d k=%d: tiled panels (tile_k=%d)\n", n, k, tile_k);
    }
    bool ok = report_cuda_error("DSYRK tiled copy C host->device",
                                cudaMemcpyAsync(d_C, C, bytesC, cudaMemcpyHostToDevice, stream));
    long long tile_chunks = 0;
    for (int k0 = 0; k0 < k; k0 += tile_k) {
      tile_chunks++;
      int kc = std::min(tile_k, k - k0);
      size_t bytesA_tile = (size_t)lda * (size_t)kc * sizeof(double);
      g_syrk_A.ensure(bytesA_tile);
      double *d_Atile = g_syrk_A.ptr;
      if (!d_Atile) {
        ok = false;
        break;
      }
      const double *A_tile = A + (size_t)k0 * (size_t)lda;
      ok = report_cuda_error("DSYRK tiled copy A host->device",
                             cudaMemcpy2DAsync(d_Atile, lda * sizeof(double),
                                               A_tile, lda * sizeof(double),
                                               (size_t)kc * sizeof(double), (size_t)n,
                                               cudaMemcpyHostToDevice, stream)) && ok;
      double beta_local = (k0 == 0) ? beta : 1.0;
      cublasStatus_t st2 = cublasDsyrk(g_blas, u, CUBLAS_OP_N, n, kc, &alpha, d_Atile, lda, &beta_local, d_C, ldc);
      ok = report_cublas_error("cublasDsyrk tiled", st2) && ok;
      if (w_verbose && st2 == CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[GPU] DSYRK tile n=%d k=%d beta=%.1f\n", n, kc, beta_local);
      }
      if (!ok) break;
    }
    prof_scope.note_tiles(tile_chunks);
    ok = report_cuda_error("DSYRK tiled copy C device->host",
                           cudaMemcpyAsync(C, d_C, bytesC, cudaMemcpyDeviceToHost, stream)) && ok;
    ok = report_cuda_error("DSYRK tiled stream synchronize", cudaStreamSynchronize(stream)) && ok;
    if (!ok) poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
  }
}

// 2-GPU outer product helpers and wrappers are further below.

__global__ void disp_eval_kernel(int n,
                                 const double *sum2,
                                 const double *sum3,
                                 const double *rab,
                                 double *val_out,
                                 double *deriv_out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  double s2 = sum2[tid];
  double s3 = sum3[tid];
  double r = rab[tid];
  double val = s2 * exp(s3 * r);
  val_out[tid] = val;
  if (deriv_out) deriv_out[tid] = val * s3;
}

extern "C" int mopac_cuda_disp_eval(int npairs,
                                     const double *sum2,
                                     const double *sum3,
                                     const double *rab,
                                     double *val_out,
                                     double *deriv_out) {
  if (npairs <= 0 || !sum2 || !sum3 || !rab || !val_out) return 0;
  char nv_name[48];
  const char* nv_ptr = nullptr;
  if (gpu_profile_enabled()) {
    std::snprintf(nv_name, sizeof(nv_name), "DISP pairs=%d", npairs);
    nv_ptr = nv_name;
  }
  NvtxRange nv_scope(nv_ptr, 0xFF8C564B);
  ScopedBlasProfile prof_scope(&g_prof_disp_eval, (double)npairs);

  cudaStream_t s = g_stream ? g_stream : 0;
  size_t bytes = sizeof(double) * (size_t)npairs;
  g_disp_sum2.ensure(bytes);
  g_disp_sum3.ensure(bytes);
  g_disp_r.ensure(bytes);
  g_disp_val.ensure(bytes);
  if (deriv_out) g_disp_der.ensure(bytes);

  if (cudaMemcpyAsync(g_disp_sum2.ptr, sum2, bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 1;
  if (cudaMemcpyAsync(g_disp_sum3.ptr, sum3, bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 1;
  if (cudaMemcpyAsync(g_disp_r.ptr, rab, bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 1;

  int block = 256;
  int grid = (npairs + block - 1) / block;
  disp_eval_kernel<<<grid, block, 0, s>>>(npairs,
                                          g_disp_sum2.ptr,
                                          g_disp_sum3.ptr,
                                          g_disp_r.ptr,
                                          g_disp_val.ptr,
                                          deriv_out ? g_disp_der.ptr : nullptr);
  if (cudaPeekAtLastError() != cudaSuccess) return 2;

  h_disp_val.ensure(bytes);
  if (cudaMemcpyAsync(h_disp_val.ptr, g_disp_val.ptr, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 1;
  if (deriv_out) {
    h_disp_der.ensure(bytes);
    if (cudaMemcpyAsync(h_disp_der.ptr, g_disp_der.ptr, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 1;
  }
  if (cudaStreamSynchronize(s) != cudaSuccess) return 1;

  std::memcpy(val_out, h_disp_val.ptr, bytes);
  if (deriv_out) std::memcpy(deriv_out, h_disp_der.ptr, bytes);
  return 0;
}

// ===== MOZYME one-center Coulomb/exchange =====

__device__ __forceinline__ void unpack_pair(int idx, int &i, int &j) {
  int acc = 0;
  int row = 0;
  while (true) {
    int row_count = row + 1;
    if (idx < acc + row_count) {
      i = row;
      j = idx - acc;
      return;
    }
    acc += row_count;
    row += 1;
  }
}

__device__ __forceinline__ int pack_pair(int a, int b) {
  if (a < b) {
    int tmp = a; a = b; b = tmp;
  }
  return a * (a + 1) / 2 + b;
}

__device__ __forceinline__ int full4_to_pair(int idx) {
  return pack_pair(idx / 4, idx - (idx / 4) * 4);
}

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

__global__ void mozyme_fock1_kernel(int iab,
                                    int pairCount,
                                    const double *ptot,
                                    const double *w,
                                    double *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= pairCount) return;
  int i = 0, j = 0;
  unpack_pair(tid, i, j);
  double sum = 0.0;
  int ilim = pairCount;
  for (int k = 0; k < iab; ++k) {
    for (int l = 0; l < iab; ++l) {
      int ijp = pack_pair(k, l);
      int klw = pack_pair(k, l);
      int ikw = pack_pair(k, j);
      int jlw = pack_pair(l, i);
      double p = ptot[ijp];
      sum += p * w[tid + (size_t)klw * ilim] - 0.5 * p * w[ikw + (size_t)jlw * ilim];
    }
  }
  out[tid] = sum;
}

extern "C" int mopac_cuda_mozyme_fock1(int iab,
                                        int ilim,
                                        const double *ptot,
                                        double *f,
                                        const double *w) {
  if (iab <= 0 || ilim <= 0 || !ptot || !f || !w) return 1;
  int pairCount = ilim;
  if (pairCount <= 0) return 1;
  cudaStream_t s = g_stream ? g_stream : 0;
  size_t bytes_pairs = sizeof(double) * (size_t)pairCount;
  size_t bytes_w = sizeof(double) * (size_t)pairCount * (size_t)pairCount;
  g_mz_fock_ptot.ensure(bytes_pairs);
  g_mz_fock_w.ensure(bytes_w);
  g_mz_fock_out.ensure(bytes_pairs);
  h_mz_fock_out.ensure(bytes_pairs);

  if (cudaMemcpyAsync(g_mz_fock_ptot.ptr, ptot, bytes_pairs, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock_w.ptr, w, bytes_w, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;

  int threads = 128;
  int blocks = (pairCount + threads - 1) / threads;
  mozyme_fock1_kernel<<<blocks, threads, 0, s>>>(iab, pairCount, g_mz_fock_ptot.ptr, g_mz_fock_w.ptr, g_mz_fock_out.ptr);
  if (cudaPeekAtLastError() != cudaSuccess) return 3;

  if (cudaMemcpyAsync(h_mz_fock_out.ptr, g_mz_fock_out.ptr, bytes_pairs, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  if (cudaStreamSynchronize(s) != cudaSuccess) return 2;

  for (int idx = 0; idx < pairCount; ++idx) {
    f[idx] += h_mz_fock_out.ptr[idx];
  }
  return 0;
}

__global__ void mozyme_fock1_batch_kernel(int ntasks,
                                          const int *iabs,
                                          const int *ilims,
                                          const int *pair_offsets,
                                          const int *w_offsets,
                                          const double *ptot,
                                          const double *w,
                                          double *out) {
  int task = blockIdx.x;
  if (task >= ntasks) return;
  int iab = iabs[task];
  int ilim = ilims[task];
  int tid = threadIdx.x;
  if (tid >= ilim) return;

  int pair_base = pair_offsets[task];
  int w_base = w_offsets[task];
  int i = 0, j = 0;
  unpack_pair(tid, i, j);
  double sum = 0.0;
  for (int k = 0; k < iab; ++k) {
    for (int l = 0; l < iab; ++l) {
      int ijp = pack_pair(k, l);
      int klw = pack_pair(k, l);
      int ikw = pack_pair(k, j);
      int jlw = pack_pair(l, i);
      double p = ptot[pair_base + ijp];
      sum += p * w[w_base + tid + (size_t)klw * ilim] -
             0.5 * p * w[w_base + ikw + (size_t)jlw * ilim];
    }
  }
  out[pair_base + tid] = sum;
}

extern "C" int mopac_cuda_mozyme_fock1_batch(int ntasks,
                                              const int *f_offsets,
                                              const int *w_offsets,
                                              const int *iabs,
                                              const int *ilims,
                                              const double *ptot,
                                              double *f,
                                              const double *w) {
  if (ntasks <= 0 || !f_offsets || !w_offsets || !iabs || !ilims || !ptot || !f || !w) return 1;

  std::vector<int> h_iabs((size_t)ntasks);
  std::vector<int> h_ilims((size_t)ntasks);
  std::vector<int> h_pair_offsets((size_t)ntasks);
  std::vector<int> h_w_offsets((size_t)ntasks);

  long long total_pairs_ll = 0;
  long long total_w_ll = 0;
  const long long int_limit = 2147483647LL;
  for (int t = 0; t < ntasks; ++t) {
    int iab = iabs[t];
    int ilim = ilims[t];
    int f0 = f_offsets[t] - 1;
    int w0 = w_offsets[t] - 1;
    if (iab <= 0 || ilim <= 0 || f0 < 0 || w0 < 0) return 1;
    int expected = iab * (iab + 1) / 2;
    if (ilim != expected) return 1;
    if (total_pairs_ll > int_limit - ilim) return 1;
    if (total_w_ll > int_limit - (long long)ilim * (long long)ilim) return 1;
    h_iabs[(size_t)t] = iab;
    h_ilims[(size_t)t] = ilim;
    h_pair_offsets[(size_t)t] = (int)total_pairs_ll;
    h_w_offsets[(size_t)t] = (int)total_w_ll;
    total_pairs_ll += ilim;
    total_w_ll += (long long)ilim * (long long)ilim;
  }
  if (total_pairs_ll <= 0 || total_w_ll <= 0) return 1;

  int total_pairs = (int)total_pairs_ll;
  int total_w = (int)total_w_ll;
  std::vector<double> h_ptot((size_t)total_pairs);
  std::vector<double> h_w((size_t)total_w);
  std::vector<double> h_out((size_t)total_pairs, 0.0);

  for (int t = 0; t < ntasks; ++t) {
    int ilim = h_ilims[(size_t)t];
    int f0 = f_offsets[t] - 1;
    int w0 = w_offsets[t] - 1;
    int pbase = h_pair_offsets[(size_t)t];
    int wbase = h_w_offsets[(size_t)t];
    std::memcpy(h_ptot.data() + pbase, ptot + f0, sizeof(double) * (size_t)ilim);
    std::memcpy(h_w.data() + wbase, w + w0, sizeof(double) * (size_t)ilim * (size_t)ilim);
  }

  cudaStream_t s = g_stream ? g_stream : 0;
  size_t task_bytes = sizeof(int) * (size_t)ntasks;
  size_t pair_bytes = sizeof(double) * (size_t)total_pairs;
  size_t w_bytes = sizeof(double) * (size_t)total_w;

  auto t0 = std::chrono::high_resolution_clock::now();
  g_mz_fock1_iab.ensure(task_bytes);
  g_mz_fock1_ilim.ensure(task_bytes);
  g_mz_fock1_pair_off.ensure(task_bytes);
  g_mz_fock1_w_off.ensure(task_bytes);
  g_mz_fock1_batch_ptot.ensure(pair_bytes);
  g_mz_fock1_batch_w.ensure(w_bytes);
  g_mz_fock1_batch_out.ensure(pair_bytes);

  if (cudaMemcpyAsync(g_mz_fock1_iab.ptr, h_iabs.data(), task_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock1_ilim.ptr, h_ilims.data(), task_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock1_pair_off.ptr, h_pair_offsets.data(), task_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock1_w_off.ptr, h_w_offsets.data(), task_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock1_batch_ptot.ptr, h_ptot.data(), pair_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock1_batch_w.ptr, h_w.data(), w_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemsetAsync(g_mz_fock1_batch_out.ptr, 0, pair_bytes, s) != cudaSuccess) return 2;

  mozyme_fock1_batch_kernel<<<ntasks, 64, 0, s>>>(ntasks,
                                                  g_mz_fock1_iab.ptr,
                                                  g_mz_fock1_ilim.ptr,
                                                  g_mz_fock1_pair_off.ptr,
                                                  g_mz_fock1_w_off.ptr,
                                                  g_mz_fock1_batch_ptot.ptr,
                                                  g_mz_fock1_batch_w.ptr,
                                                  g_mz_fock1_batch_out.ptr);
  if (cudaPeekAtLastError() != cudaSuccess) return 3;
  if (cudaMemcpyAsync(h_out.data(), g_mz_fock1_batch_out.ptr, pair_bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  if (cudaStreamSynchronize(s) != cudaSuccess) return 2;

  for (int t = 0; t < ntasks; ++t) {
    int ilim = h_ilims[(size_t)t];
    int f0 = f_offsets[t] - 1;
    int pbase = h_pair_offsets[(size_t)t];
    for (int pidx = 0; pidx < ilim; ++pidx) {
      f[f0 + pidx] += h_out[(size_t)pbase + (size_t)pidx];
    }
  }

  if (gpu_profile_enabled()) {
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::fprintf(stderr,
                 "[GPU] profile mozyme_fock1_batch tasks=%d pairs=%d w_values=%d ms=%.3f\n",
                 ntasks, total_pairs, total_w, ms);
  }
  return 0;
}

__global__ void mozyme_fock2_4x1_batch_kernel(int ntasks,
                                              const double *diag,
                                              const double *light,
                                              const double *cross,
                                              const double *wj,
                                              const double *wk,
                                              double *out) {
  int task = blockIdx.x;
  int tid = threadIdx.x;
  if (task >= ntasks) return;

  const int diag_base = task * 10;
  const int cross_base = task * 4;
  const int wk_base = task * 16;
  const int out_base = task * 15;
  if (tid < 10) {
    out[out_base + tid] = light[task] * wj[diag_base + tid];
  } else if (tid == 10) {
    const int offdiag[6] = {1, 3, 4, 6, 7, 8};
    const int diagidx[4] = {0, 2, 5, 9};
    double sumoff = 0.0;
    double sumdia = 0.0;
    for (int i = 0; i < 6; ++i) {
      int idx = offdiag[i];
      sumoff += diag[diag_base + idx] * wj[diag_base + idx];
    }
    for (int i = 0; i < 4; ++i) {
      int idx = diagidx[i];
      sumdia += diag[diag_base + idx] * wj[diag_base + idx];
    }
    out[out_base + 10] = 2.0 * sumoff + sumdia;
  } else if (tid >= 11 && tid < 15) {
    int row = tid - 11;
    double sum = 0.0;
    for (int j = 0; j < 4; ++j) {
      sum += cross[cross_base + j] * wk[wk_base + row * 4 + j];
    }
    out[out_base + tid] = -0.5 * sum;
  }
}

extern "C" int mopac_cuda_mozyme_fock2_4x1_batch(int ntasks,
                                                  const int *heavy_offsets,
                                                  const int *light_offsets,
                                                  const int *cross_offsets,
                                                  const double *wj_values,
                                                  const double *wk_values,
                                                  const double *ptot,
                                                  double *f) {
  if (ntasks <= 0 || !heavy_offsets || !light_offsets || !cross_offsets ||
      !wj_values || !wk_values || !ptot || !f) return 1;

  std::vector<double> h_diag((size_t)ntasks * 10u);
  std::vector<double> h_light((size_t)ntasks);
  std::vector<double> h_cross((size_t)ntasks * 4u);
  std::vector<double> h_wj((size_t)ntasks * 10u);
  std::vector<double> h_wk((size_t)ntasks * 16u);
  std::vector<double> h_out((size_t)ntasks * 15u, 0.0);

  for (int t = 0; t < ntasks; ++t) {
    int h0 = heavy_offsets[t] - 1;
    int l0 = light_offsets[t] - 1;
    int c0 = cross_offsets[t] - 1;
    if (h0 < 0 || l0 < 0 || c0 < 0) return 1;
    std::memcpy(h_diag.data() + (size_t)t * 10u, ptot + h0, sizeof(double) * 10u);
    h_light[(size_t)t] = ptot[l0];
    std::memcpy(h_cross.data() + (size_t)t * 4u, ptot + c0, sizeof(double) * 4u);
    std::memcpy(h_wj.data() + (size_t)t * 10u, wj_values + (size_t)t * 10u, sizeof(double) * 10u);
    std::memcpy(h_wk.data() + (size_t)t * 16u, wk_values + (size_t)t * 16u, sizeof(double) * 16u);
  }

  cudaStream_t s = g_stream ? g_stream : 0;
  const size_t diag_bytes = sizeof(double) * h_diag.size();
  const size_t light_bytes = sizeof(double) * h_light.size();
  const size_t cross_bytes = sizeof(double) * h_cross.size();
  const size_t wj_bytes = sizeof(double) * h_wj.size();
  const size_t wk_bytes = sizeof(double) * h_wk.size();
  const size_t out_bytes = sizeof(double) * h_out.size();

  auto t0 = std::chrono::high_resolution_clock::now();
  g_mz_fock2_4x1_diag.ensure(diag_bytes);
  g_mz_fock2_4x1_light.ensure(light_bytes);
  g_mz_fock2_4x1_cross.ensure(cross_bytes);
  g_mz_fock2_4x1_wj.ensure(wj_bytes);
  g_mz_fock2_4x1_wk.ensure(wk_bytes);
  g_mz_fock2_4x1_out.ensure(out_bytes);

  if (cudaMemcpyAsync(g_mz_fock2_4x1_diag.ptr, h_diag.data(), diag_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_4x1_light.ptr, h_light.data(), light_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_4x1_cross.ptr, h_cross.data(), cross_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_4x1_wj.ptr, h_wj.data(), wj_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_4x1_wk.ptr, h_wk.data(), wk_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemsetAsync(g_mz_fock2_4x1_out.ptr, 0, out_bytes, s) != cudaSuccess) return 2;

  mozyme_fock2_4x1_batch_kernel<<<ntasks, 16, 0, s>>>(ntasks,
                                                      g_mz_fock2_4x1_diag.ptr,
                                                      g_mz_fock2_4x1_light.ptr,
                                                      g_mz_fock2_4x1_cross.ptr,
                                                      g_mz_fock2_4x1_wj.ptr,
                                                      g_mz_fock2_4x1_wk.ptr,
                                                      g_mz_fock2_4x1_out.ptr);
  if (cudaPeekAtLastError() != cudaSuccess) return 3;
  if (cudaMemcpyAsync(h_out.data(), g_mz_fock2_4x1_out.ptr, out_bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  if (cudaStreamSynchronize(s) != cudaSuccess) return 2;

  for (int t = 0; t < ntasks; ++t) {
    int h0 = heavy_offsets[t] - 1;
    int l0 = light_offsets[t] - 1;
    int c0 = cross_offsets[t] - 1;
    const double *delta = h_out.data() + (size_t)t * 15u;
    for (int i = 0; i < 10; ++i) f[h0 + i] += delta[i];
    f[l0] += delta[10];
    for (int i = 0; i < 4; ++i) f[c0 + i] += delta[11 + i];
  }

  if (gpu_profile_enabled()) {
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::fprintf(stderr,
                 "[GPU] profile mozyme_fock2_4x1_batch tasks=%d ms=%.3f\n",
                 ntasks, ms);
  }
  return 0;
}

__device__ inline bool mozyme_sparse_guard_allows_work(
    const int *guard_ints, int guard_slot, int guard_continue) {
  return !guard_ints || guard_slot < 0 ||
         guard_ints[guard_slot] == guard_continue;
}

__global__ void mozyme_sparse_one_center_kernel(int ntasks,
                                                const int *f_offsets,
                                                const int *w_offsets,
                                                const int *iabs,
                                                const int *ilims,
                                                const double *w_values,
                                                const double *ptot,
                                                double *f,
                                                const int *guard_ints,
                                                int guard_slot,
                                                int guard_continue) {
  if (!mozyme_sparse_guard_allows_work(guard_ints, guard_slot,
                                       guard_continue)) return;
  int task = blockIdx.x;
  if (task >= ntasks) return;
  int tid = threadIdx.x;
  int iab = iabs[task];
  int ilim = ilims[task];
  int fbase = f_offsets[task] - 1;
  int wbase = w_offsets[task] - 1;
  for (int ij_idx = tid; ij_idx < ilim; ij_idx += blockDim.x) {
    int i = 0, j = 0;
    unpack_pair(ij_idx, i, j);
    double sum = 0.0;
    for (int k = 0; k < iab; ++k) {
      for (int l = 0; l < iab; ++l) {
        int ijp = pack_pair(k, l);
        int klw = pack_pair(k, l);
        int ikw = pack_pair(k, j);
        int jlw = pack_pair(l, i);
        double p = ptot[fbase + ijp];
        sum += p * w_values[wbase + ij_idx + (size_t)klw * ilim] -
               0.5 * p * w_values[wbase + ikw + (size_t)jlw * ilim];
      }
    }
    atomicAdd_double(f + fbase + ij_idx, sum);
  }
}

__global__ void mozyme_sparse_pair_kernel(int ntasks,
                                          const int *iabs,
                                          const int *jbas,
                                          const int *i_offsets,
                                          const int *j_offsets,
                                          const int *cross_offsets,
                                          const int *diag_flags,
                                          const int *w_offsets,
                                          const double *wj_values,
                                          const double *wk_values,
                                          const double *ptot,
                                          double *f,
                                          const int *guard_ints,
                                          int guard_slot,
                                          int guard_continue) {
  if (!mozyme_sparse_guard_allows_work(guard_ints, guard_slot,
                                       guard_continue)) return;
  int task = blockIdx.x;
  if (task >= ntasks) return;
  int iab = iabs[task];
  int jba = jbas[task];
  if (iab <= 0 || jba <= 0) return;
  int ni = iab * (iab + 1) / 2;
  int nj = jba * (jba + 1) / 2;
  int total = ni * nj;
  int ibase = i_offsets[task] - 1;
  int jbase = j_offsets[task] - 1;
  int cbase = cross_offsets[task] - 1;
  int wbase = w_offsets[task] - 1;
  int diagonal = diag_flags[task];

  if (iab == 4 && jba == 4) {
    int tid = threadIdx.x;
    if (tid < 10) {
      double suma = 0.0;
      double sumb = 0.0;
      for (int full = 0; full < 16; ++full) {
        int dens = full4_to_pair(full);
        double pja = ptot[ibase + dens];
        double pjb = ptot[jbase + dens];
        suma += pja * wj_values[wbase + dens * 10 + tid];
        sumb += pjb * wj_values[wbase + tid * 10 + dens];
      }
      atomicAdd_double(f + ibase + tid, sumb);
      if (!diagonal) atomicAdd_double(f + jbase + tid, suma);
    } else if (!diagonal && tid < 26) {
      int out = tid - 10;
      int a = out / 4;
      int b = out - a * 4;
      double sum = 0.0;
      for (int in = 0; in < 16; ++in) {
        int c = in / 4;
        int d = in - c * 4;
        int widx = pack_pair(a, c) * 10 + pack_pair(b, d);
        sum += ptot[cbase + in] * wk_values[wbase + widx];
      }
      atomicAdd_double(f + cbase + out, -0.5 * sum);
    }
    return;
  }

  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    int ij_idx = idx / nj;
    int kl_idx = idx - ij_idx * nj;
    int i0 = 0, j0 = 0, k0 = 0, l0 = 0;
    unpack_pair(ij_idx, i0, j0);
    unpack_pair(kl_idx, k0, l0);
    double aa = (i0 == j0) ? 1.0 : 2.0;
    double bb = (k0 == l0) ? 1.0 : 2.0;
    double coul = wj_values[wbase + idx];
    atomicAdd_double(f + ibase + ij_idx, bb * coul * ptot[jbase + kl_idx]);
    if (!diagonal) {
      atomicAdd_double(f + jbase + kl_idx, aa * coul * ptot[ibase + ij_idx]);
      double exch = wk_values[wbase + idx] * aa * bb * 0.125;
      int ik = i0 * jba + k0;
      int il = i0 * jba + l0;
      int jk = j0 * jba + k0;
      int jl = j0 * jba + l0;
      atomicAdd_double(f + cbase + ik, -exch * ptot[cbase + jl]);
      atomicAdd_double(f + cbase + il, -exch * ptot[cbase + jk]);
      atomicAdd_double(f + cbase + jk, -exch * ptot[cbase + il]);
      atomicAdd_double(f + cbase + jl, -exch * ptot[cbase + ik]);
    }
  }
}

__global__ void mozyme_sparse_4x1_kernel(int ntasks,
                                         const int *heavy_offsets,
                                         const int *light_offsets,
                                         const int *cross_offsets,
                                         const double *wj_values,
                                         const double *wk_values,
                                         const double *ptot,
                                         double *f,
                                         const int *guard_ints,
                                         int guard_slot,
                                         int guard_continue) {
  if (!mozyme_sparse_guard_allows_work(guard_ints, guard_slot,
                                       guard_continue)) return;
  int task = blockIdx.x;
  int tid = threadIdx.x;
  if (task >= ntasks) return;
  int heavy = heavy_offsets[task] - 1;
  int light = light_offsets[task] - 1;
  int cross = cross_offsets[task] - 1;
  int wjbase = task * 10;
  int wkbase = task * 16;
  if (tid < 10) {
    atomicAdd_double(f + heavy + tid, ptot[light] * wj_values[wjbase + tid]);
  } else if (tid == 10) {
    const int offdiag[6] = {1, 3, 4, 6, 7, 8};
    const int diagidx[4] = {0, 2, 5, 9};
    double sumoff = 0.0;
    double sumdia = 0.0;
    for (int i = 0; i < 6; ++i) {
      int idx = offdiag[i];
      sumoff += ptot[heavy + idx] * wj_values[wjbase + idx];
    }
    for (int i = 0; i < 4; ++i) {
      int idx = diagidx[i];
      sumdia += ptot[heavy + idx] * wj_values[wjbase + idx];
    }
    atomicAdd_double(f + light, 2.0 * sumoff + sumdia);
  } else if (tid >= 11 && tid < 15) {
    int row = tid - 11;
    double sum = 0.0;
    for (int j = 0; j < 4; ++j) {
      sum += ptot[cross + j] * wk_values[wkbase + row * 4 + j];
    }
    atomicAdd_double(f + cross + row, -0.5 * sum);
  }
}

__global__ void mozyme_sparse_point_kernel(int ntasks,
                                           const int *iabs,
                                           const int *jbas,
                                           const int *i_atoms,
                                           const int *j_atoms,
                                           const int *i_offsets,
                                           const int *j_offsets,
                                           const int *addr_flags,
                                           const double *w_values,
                                           const double *qe,
                                           const double *ptot,
                                           double *f,
                                           const int *guard_ints,
                                           int guard_slot,
                                           int guard_continue) {
  if (!mozyme_sparse_guard_allows_work(guard_ints, guard_slot,
                                       guard_continue)) return;
  int task = blockIdx.x;
  int tid = threadIdx.x;
  if (task >= ntasks) return;
  int iab = iabs[task];
  int jba = jbas[task];
  int ibase = i_offsets[task] - 1;
  int jbase = j_offsets[task] - 1;
  int iatom = i_atoms[task] - 1;
  int jatom = j_atoms[task] - 1;
  const double *w = w_values + (size_t)task * 7u;
  double w1 = w[0];

  for (int orb = tid; orb < iab; orb += blockDim.x) {
    atomicAdd_double(f + ibase + pack_pair(orb, orb), qe[jatom] * w1);
  }
  for (int orb = tid; orb < jba; orb += blockDim.x) {
    atomicAdd_double(f + jbase + pack_pair(orb, orb), qe[iatom] * w1);
  }

  if (addr_flags[task] != -2) return;

  double w2 = w[1], w3 = w[2], w4 = w[3];
  double w5 = w[4], w6 = w[5], w7 = w[6];
  if (tid == 0) {
    if (iab > 1) {
      atomicAdd_double(f + ibase + 1, qe[jatom] * w5);
      atomicAdd_double(f + ibase + 3, qe[jatom] * w6);
      atomicAdd_double(f + ibase + 6, qe[jatom] * w7);
    }
    if (jba > 1) {
      atomicAdd_double(f + jbase + 1, qe[iatom] * w2);
      atomicAdd_double(f + jbase + 3, qe[iatom] * w3);
      atomicAdd_double(f + jbase + 6, qe[iatom] * w4);
    }
  }
  if (iab > 1) {
    const int diag[4] = {0, 2, 5, 9};
    int ndiag = (jba > 1) ? 4 : 1;
    double sum = 2.0 * (ptot[ibase + 1] * w5 + ptot[ibase + 3] * w6 + ptot[ibase + 6] * w7);
    if (tid < ndiag) atomicAdd_double(f + jbase + diag[tid], sum);
  }
  if (jba > 1) {
    const int diag[4] = {0, 2, 5, 9};
    int ndiag = (iab > 1) ? 4 : 1;
    double sum = 2.0 * (ptot[jbase + 1] * w2 + ptot[jbase + 3] * w3 + ptot[jbase + 6] * w4);
    if (tid < ndiag) atomicAdd_double(f + ibase + diag[tid], sum);
  }
}

// ---------------------------------------------------------------------------
// Point-pair Fock terms without one block per pair.
//
// mozyme_sparse_point_kernel above launches one 32-thread block per point
// pair, and a 7000-atom protein has ~22 million of them: the block scheduling
// alone costs ~25 ms per SCF iteration.  The kernels below keep the same
// arithmetic but (a) run one thread per pair with a grid-stride loop, (b)
// accumulate per-atom scalars instead of touching f directly, reducing the
// row side (pairs of one atom are contiguous in the plan) inside the warp
// before a single atomic, and (c) apply the accumulators to f with one thread
// per atom.  Per atom the accumulator holds
//   [0] value added to every diagonal element of the block (monopole + the
//       density-weighted dipole term),
//   [1] value added to the first min(nb,4) diagonal elements only (the
//       dipole term that the original kernel restricted to diag[0..3]),
//   [2..4] values added to the s-p elements 1, 3, 6 (dipole pairs only).
// ---------------------------------------------------------------------------
static constexpr int kMzPointAccStride = 8;

__global__ void mozyme_sparse_point_atom_index_kernel(int ntasks,
                                                      const int *iabs,
                                                      const int *jbas,
                                                      const int *i_atoms,
                                                      const int *j_atoms,
                                                      const int *i_offsets,
                                                      const int *j_offsets,
                                                      int *atom_off,
                                                      int *atom_iab) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < ntasks;
       t += gridDim.x * blockDim.x) {
    const int ia = i_atoms[t] - 1;
    const int ja = j_atoms[t] - 1;
    atom_off[ia] = i_offsets[t] - 1;
    atom_iab[ia] = iabs[t];
    atom_off[ja] = j_offsets[t] - 1;
    atom_iab[ja] = jbas[t];
  }
}

__global__ void mozyme_sparse_point_accumulate_kernel(int ntasks,
                                                      const int *iabs,
                                                      const int *jbas,
                                                      const int *i_atoms,
                                                      const int *j_atoms,
                                                      const int *i_offsets,
                                                      const int *j_offsets,
                                                      const int *addr_flags,
                                                      const double *w_values,
                                                      const double *qe,
                                                      const double *ptot,
                                                      double *acc,
                                                      const int *guard_ints,
                                                      int guard_slot,
                                                      int guard_continue) {
  if (!mozyme_sparse_guard_allows_work(guard_ints, guard_slot,
                                       guard_continue)) return;
  const int lane = threadIdx.x & 31;
  const unsigned full = 0xffffffffu;
  const int stride = gridDim.x * blockDim.x;
  for (int t0 = blockIdx.x * blockDim.x + (threadIdx.x & ~31); t0 < ntasks;
       t0 += stride) {
    const int t = t0 + lane;
    const bool valid = t < ntasks;
    int ia = -1, ja = -1;
    double ai = 0.0, aj = 0.0, di = 0.0, dj = 0.0;
    if (valid) {
      const int iab = iabs[t];
      const int jba = jbas[t];
      ia = i_atoms[t] - 1;
      ja = j_atoms[t] - 1;
      const double *w = w_values + static_cast<size_t>(t) * 7u;
      const double w1 = w[0];
      const double qi = qe[ia];
      const double qj = qe[ja];
      ai = qj * w1;
      aj = qi * w1;
      if (addr_flags[t] == -2) {
        const int ibase = i_offsets[t] - 1;
        const int jbase = j_offsets[t] - 1;
        const double w2 = w[1], w3 = w[2], w4 = w[3];
        const double w5 = w[4], w6 = w[5], w7 = w[6];
        if (iab > 1) {
          // s-p elements of atom i, and the density-weighted term on the
          // first diagonal elements of atom j
          atomicAdd_double(acc + ia * kMzPointAccStride + 2, qj * w5);
          atomicAdd_double(acc + ia * kMzPointAccStride + 3, qj * w6);
          atomicAdd_double(acc + ia * kMzPointAccStride + 4, qj * w7);
          dj = 2.0 * (ptot[ibase + 1] * w5 + ptot[ibase + 3] * w6 +
                      ptot[ibase + 6] * w7);
        }
        if (jba > 1) {
          atomicAdd_double(acc + ja * kMzPointAccStride + 2, qi * w2);
          atomicAdd_double(acc + ja * kMzPointAccStride + 3, qi * w3);
          atomicAdd_double(acc + ja * kMzPointAccStride + 4, qi * w4);
          di = 2.0 * (ptot[jbase + 1] * w2 + ptot[jbase + 3] * w3 +
                      ptot[jbase + 6] * w4);
        }
      }
    }
    // Row side: keys (ia) are non-decreasing across the warp, so equal keys
    // form one contiguous run.  Segmented inclusive scan, then the last lane
    // of each run adds the run total with a single atomic.
    double v = ai;
    for (int off = 1; off < 32; off <<= 1) {
      const double nv = __shfl_up_sync(full, v, off);
      const int nk = __shfl_up_sync(full, ia, off);
      if (lane >= off && nk == ia) v += nv;
    }
    const int nextkey = __shfl_down_sync(full, ia, 1);
    const bool tail = (lane == 31) || (nextkey != ia);
    if (valid && tail) atomicAdd_double(acc + ia * kMzPointAccStride, v);
    if (valid && di != 0.0) atomicAdd_double(acc + ia * kMzPointAccStride + 1, di);
    // Column side: one atomic per pair.
    if (valid) {
      atomicAdd_double(acc + ja * kMzPointAccStride, aj);
      if (dj != 0.0) atomicAdd_double(acc + ja * kMzPointAccStride + 1, dj);
    }
  }
}

__global__ void mozyme_sparse_point_apply_kernel(int natoms,
                                                 const int *atom_off,
                                                 const int *atom_iab,
                                                 const double *acc,
                                                 double *f,
                                                 const int *guard_ints,
                                                 int guard_slot,
                                                 int guard_continue) {
  if (!mozyme_sparse_guard_allows_work(guard_ints, guard_slot,
                                       guard_continue)) return;
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= natoms) return;
  const int iab = atom_iab[a];
  if (iab <= 0) return;
  const int off = atom_off[a];
  const double *ac = acc + a * kMzPointAccStride;
  const double mono = ac[0];
  const double dip = ac[1];
  const int ndip = (iab > 1) ? 4 : 1;
  for (int orb = 0; orb < iab; ++orb) {
    double add = mono;
    if (orb < ndip) add += dip;
    if (add != 0.0) f[off + pack_pair(orb, orb)] += add;
  }
  if (iab > 1) {
    f[off + 1] += ac[2];
    f[off + 3] += ac[3];
    f[off + 6] += ac[4];
  }
}

__device__ __forceinline__ long long mozyme_fillij_tri_dev(int n) {
  return (static_cast<long long>(n) * static_cast<long long>(n + 1)) / 2ll;
}

__device__ __forceinline__ double mozyme_fillij_coord_at(const double *coord,
                                                         int dim,
                                                         int atom0) {
  return coord[atom0 * 3 + dim];
}

__device__ __forceinline__ double mozyme_fillij_tvec_at(const double *tvec,
                                                        int dim,
                                                        int vec0) {
  return tvec[vec0 * 3 + dim];
}

__device__ __forceinline__ int mozyme_fillij_nij_index(int numat, int row0,
                                                       int col0) {
  return row0 + col0 * numat;
}

__global__ void mozyme_fillij_kernel(int numat, int id, int l1u, int l2u,
                                     int l3u, int ispd, int direct_flag,
                                     int semidr_flag, int count_flag,
                                     int fill_nijbo, double cutof1,
                                     double cutof2, const double *coord,
                                     const double *tvec, const int *iorbs,
                                     int *nijbo, int *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  long long mpack_ll = 0;
  long long n2elec_ll = 0;
  long long ix = 0;
  const bool direct = direct_flag != 0;
  const bool semidr = semidr_flag != 0;
  const bool count = count_flag != 0;
  const bool fill = fill_nijbo != 0 && nijbo;

  for (int iloop = 1; iloop <= numat; ++iloop) {
    const int i0 = iloop - 1;
    const int io = iorbs[i0];
    const double x1 = mozyme_fillij_coord_at(coord, 0, i0);
    const double x2 = mozyme_fillij_coord_at(coord, 1, i0);
    const double x3 = mozyme_fillij_coord_at(coord, 2, i0);
    const long long ii = mozyme_fillij_tri_dev(io);
    n2elec_ll += ii * ii;

    for (int jloop = 1; jloop <= iloop - 1; ++jloop) {
      const int j0 = jloop - 1;
      const int jo = iorbs[j0];
      double r;
      if (id == 0) {
        const double dx = x1 - mozyme_fillij_coord_at(coord, 0, j0);
        const double dy = x2 - mozyme_fillij_coord_at(coord, 1, j0);
        const double dz = x3 - mozyme_fillij_coord_at(coord, 2, j0);
        r = dx * dx + dy * dy + dz * dz;
      } else {
        r = 1.0e10;
        for (int ip = -l1u; ip <= l1u; ++ip) {
          for (int jp = -l2u; jp <= l2u; ++jp) {
            for (int kp = -l3u; kp <= l3u; ++kp) {
              const double xj0 = mozyme_fillij_coord_at(coord, 0, j0) +
                                  mozyme_fillij_tvec_at(tvec, 0, 0) * ip +
                                  mozyme_fillij_tvec_at(tvec, 0, 1) * jp +
                                  mozyme_fillij_tvec_at(tvec, 0, 2) * kp;
              const double xj1 = mozyme_fillij_coord_at(coord, 1, j0) +
                                  mozyme_fillij_tvec_at(tvec, 1, 0) * ip +
                                  mozyme_fillij_tvec_at(tvec, 1, 1) * jp +
                                  mozyme_fillij_tvec_at(tvec, 1, 2) * kp;
              const double xj2 = mozyme_fillij_coord_at(coord, 2, j0) +
                                  mozyme_fillij_tvec_at(tvec, 2, 0) * ip +
                                  mozyme_fillij_tvec_at(tvec, 2, 1) * jp +
                                  mozyme_fillij_tvec_at(tvec, 2, 2) * kp;
              const double dx = x1 - xj0;
              const double dy = x2 - xj1;
              const double dz = x3 - xj2;
              r = fmin(r, dx * dx + dy * dy + dz * dz);
            }
          }
        }
      }

      if (r < cutof2) {
        ++ix;
        if (fill && !count) {
          nijbo[mozyme_fillij_nij_index(numat, i0, j0)] =
              static_cast<int>(mpack_ll);
          nijbo[mozyme_fillij_nij_index(numat, j0, i0)] =
              static_cast<int>(mpack_ll);
        }
        mpack_ll += static_cast<long long>(io) * jo;
        if (!direct) n2elec_ll += mozyme_fillij_tri_dev(jo) * ii;
      } else if (r < cutof1) {
        if (!semidr) {
          if (io > 1) {
            n2elec_ll += (jo > 1) ? 7 : 4;
          } else if (jo > 1) {
            n2elec_ll += 4;
          } else {
            n2elec_ll += 1;
          }
        }
        if (fill && !count) {
          nijbo[mozyme_fillij_nij_index(numat, i0, j0)] = -2;
          nijbo[mozyme_fillij_nij_index(numat, j0, i0)] = -2;
        }
      } else {
        if (!semidr) n2elec_ll += 1;
        if (fill && !count) {
          nijbo[mozyme_fillij_nij_index(numat, i0, j0)] = -1;
          nijbo[mozyme_fillij_nij_index(numat, j0, i0)] = -1;
        }
      }
    }

    ++ix;
    if (fill && !count) {
      nijbo[mozyme_fillij_nij_index(numat, i0, i0)] =
          static_cast<int>(mpack_ll);
    }
    if (id != 0) n2elec_ll += ii * ii;
    mpack_ll += ii;
  }

  if (n2elec_ll < 2025) n2elec_ll = 2025;
  n2elec_ll += 10;
  if (direct && ispd == 0) n2elec_ll += 100;
  if (direct && ispd != 0) n2elec_ll += 2025;
  if (id != 0) n2elec_ll *= 2;

  constexpr long long int_max = 2147483647ll;
  if (mpack_ll > int_max || n2elec_ll > int_max || ix > int_max) {
    out[3] = 3;
    return;
  }
  out[0] = static_cast<int>(mpack_ll);
  out[1] = static_cast<int>(n2elec_ll);
  out[2] = static_cast<int>(ix);
  out[3] = 0;
}

static int mozyme_fillij_gpu_run(int numat, int id, int l1u, int l2u,
                                 int l3u, int ispd, int direct_flag,
                                 int semidr_flag, double cutof1,
                                 double cutof2, const double *coord,
                                 const double *tvec, const int *iorbs,
                                 int *nijbo, int *mpack_out,
                                 int *n2elec_out, int *ij_dim_out) {
  if (numat <= 0 || !coord || !tvec || !iorbs || !mpack_out ||
      !n2elec_out || !ij_dim_out) {
    return 1;
  }
  const bool fill = nijbo != nullptr;
  cudaStream_t s = g_stream ? g_stream : 0;
  const size_t iorbs_bytes = sizeof(int) * static_cast<size_t>(numat);
  const size_t coord_bytes = sizeof(double) * static_cast<size_t>(3 * numat);
  const size_t tvec_bytes = sizeof(double) * 9u;
  const size_t nijbo_bytes =
      sizeof(int) * static_cast<size_t>(numat) * static_cast<size_t>(numat);
  if (!g_mz_fillij_iorbs.ensure(iorbs_bytes) ||
      !g_mz_fillij_coord.ensure(coord_bytes) ||
      !g_mz_fillij_tvec.ensure(tvec_bytes) ||
      !g_mz_fillij_out.ensure(sizeof(int) * 4u)) {
    return 2;
  }
  if (fill && !g_mz_fillij_nijbo.ensure(nijbo_bytes)) return 2;

  int host_out[4] = {0, 0, 0, 2};
  cudaError_t status = cudaMemcpyAsync(g_mz_fillij_iorbs.ptr, iorbs,
                                       iorbs_bytes, cudaMemcpyHostToDevice, s);
  if (status != cudaSuccess) {
    report_cuda_error("mozyme fillij copy iorbs", status);
    return 2;
  }
  status = cudaMemcpyAsync(g_mz_fillij_coord.ptr, coord, coord_bytes,
                           cudaMemcpyHostToDevice, s);
  if (status != cudaSuccess) {
    report_cuda_error("mozyme fillij copy coord", status);
    return 2;
  }
  status = cudaMemcpyAsync(g_mz_fillij_tvec.ptr, tvec, tvec_bytes,
                           cudaMemcpyHostToDevice, s);
  if (status != cudaSuccess) {
    report_cuda_error("mozyme fillij copy tvec", status);
    return 2;
  }
  status = cudaMemcpyAsync(g_mz_fillij_out.ptr, host_out, sizeof(host_out),
                           cudaMemcpyHostToDevice, s);
  if (status != cudaSuccess) {
    report_cuda_error("mozyme fillij init out", status);
    return 2;
  }
  if (fill) {
    status = cudaMemsetAsync(g_mz_fillij_nijbo.ptr, 0, nijbo_bytes, s);
    if (status != cudaSuccess) {
      report_cuda_error("mozyme fillij clear nijbo", status);
      return 2;
    }
  }

  mozyme_fillij_kernel<<<1, 1, 0, s>>>(
      numat, id, l1u, l2u, l3u, ispd, direct_flag, semidr_flag,
      fill ? 0 : 1, fill ? 1 : 0, cutof1, cutof2,
      g_mz_fillij_coord.ptr, g_mz_fillij_tvec.ptr, g_mz_fillij_iorbs.ptr,
      fill ? g_mz_fillij_nijbo.ptr : nullptr, g_mz_fillij_out.ptr);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    report_cuda_error("mozyme fillij kernel launch", status);
    return 2;
  }
  status = cudaMemcpyAsync(host_out, g_mz_fillij_out.ptr, sizeof(host_out),
                           cudaMemcpyDeviceToHost, s);
  if (status != cudaSuccess) {
    report_cuda_error("mozyme fillij copy out", status);
    return 2;
  }
  if (fill) {
    status = cudaMemcpyAsync(nijbo, g_mz_fillij_nijbo.ptr, nijbo_bytes,
                             cudaMemcpyDeviceToHost, s);
    if (status != cudaSuccess) {
      report_cuda_error("mozyme fillij copy nijbo", status);
      return 2;
    }
  }
  status = cudaStreamSynchronize(s);
  if (status != cudaSuccess) {
    report_cuda_error("mozyme fillij synchronize", status);
    return 2;
  }
  if (host_out[3] != 0) return host_out[3];
  *mpack_out = host_out[0];
  *n2elec_out = host_out[1];
  *ij_dim_out = host_out[2];
  if (gpu_profile_enabled()) {
    std::fprintf(stderr,
                 "[GPU] profile mozyme_fillij mode=%s atoms=%d mpack=%d n2elec=%d ij_dim=%d\n",
                 fill ? "fill" : "count", numat, host_out[0], host_out[1],
                 host_out[2]);
  }
  return 0;
}

extern "C" int mopac_cuda_mozyme_fillij_count(
    int numat, int id, int l1u, int l2u, int l3u, int ispd, int direct_flag,
    int semidr_flag, double cutof1, double cutof2, const double *coord,
    const double *tvec, const int *iorbs, int *mpack_out, int *n2elec_out,
    int *ij_dim_out) {
  return mozyme_fillij_gpu_run(numat, id, l1u, l2u, l3u, ispd, direct_flag,
                               semidr_flag, cutof1, cutof2, coord, tvec,
                               iorbs, nullptr, mpack_out, n2elec_out,
                               ij_dim_out);
}

extern "C" int mopac_cuda_mozyme_fillij_nijbo(
    int numat, int id, int l1u, int l2u, int l3u, int ispd, int direct_flag,
    int semidr_flag, double cutof1, double cutof2, const double *coord,
    const double *tvec, const int *iorbs, int *nijbo, int *mpack_out,
    int *n2elec_out, int *ij_dim_out) {
  if (!nijbo) return 1;
  return mozyme_fillij_gpu_run(numat, id, l1u, l2u, l3u, ispd, direct_flag,
                               semidr_flag, cutof1, cutof2, coord, tvec,
                               iorbs, nijbo, mpack_out, n2elec_out,
                               ij_dim_out);
}

__device__ __forceinline__ int mozyme_resident_tri_dev(int n) {
  return (n * (n + 1)) / 2;
}

__device__ __forceinline__ bool mozyme_resident_basis_supported_dev(int nbasis) {
  return nbasis == 1 || nbasis == 4 || nbasis == 9;
}

__device__ __forceinline__ bool mozyme_resident_direct_basis_dev(int nbasis) {
  return nbasis == 1 || nbasis == 4 || nbasis == 9;
}

__device__ __forceinline__ bool mozyme_resident_direct_sp_basis_dev(int nbasis) {
  return nbasis == 1 || nbasis == 4;
}

__device__ __forceinline__ bool mozyme_resident_pair_supported_for_direct_dev(
    int iab, int jba, int direct_flag) {
  if (!mozyme_resident_basis_supported_dev(iab) ||
      !mozyme_resident_basis_supported_dev(jba)) {
    return false;
  }
  if (direct_flag == 0) return true;
  return mozyme_resident_direct_basis_dev(iab) &&
         mozyme_resident_direct_basis_dev(jba);
}

__device__ __forceinline__ bool mozyme_resident_pair_noop_dev(int iab,
                                                              int jba) {
  return iab == 0 || jba == 0;
}

__device__ __forceinline__ bool mozyme_resident_point_supported_dev(int iab,
                                                                    int jba,
                                                                    int addr) {
  return addr < 0 && mozyme_resident_basis_supported_dev(iab) &&
         mozyme_resident_basis_supported_dev(jba);
}

__device__ __forceinline__ bool mozyme_resident_point_supported_for_direct_dev(
    int iab, int jba, int addr, int direct_flag) {
  if (!mozyme_resident_point_supported_dev(iab, jba, addr)) return false;
  if (direct_flag == 0) return true;
  return mozyme_resident_direct_basis_dev(iab) &&
         mozyme_resident_direct_basis_dev(jba);
}

__device__ __forceinline__ int mozyme_resident_pair_integral_count_dev(
    int iab, int jba) {
  if ((iab == 4 && jba == 1) || (iab == 1 && jba == 4)) return 10;
  return mozyme_resident_tri_dev(iab) * mozyme_resident_tri_dev(jba);
}

__device__ __forceinline__ int mozyme_resident_nijbo_at(int numat,
                                                        const int *nijbo,
                                                        int i_fortran,
                                                        int j_fortran) {
  return nijbo[(i_fortran - 1) + (j_fortran - 1) * numat];
}

static constexpr int kMozymeResidentFallbackBasisBins = 11;

__device__ __forceinline__ int mozyme_resident_fallback_basis_bin_dev(
    int nbasis) {
  if (nbasis < 0) return 0;
  if (nbasis > 9) return kMozymeResidentFallbackBasisBins - 1;
  return nbasis;
}

__device__ __forceinline__ void mozyme_resident_increment_fallback_basis_dev(
    int iab, int jba, int *fallback_basis) {
  const int ib = mozyme_resident_fallback_basis_bin_dev(iab);
  const int jb = mozyme_resident_fallback_basis_bin_dev(jba);
  fallback_basis[ib + jb * kMozymeResidentFallbackBasisBins] += 1;
}

__global__ void mozyme_resident_fock_count_kernel(
    int numat, int mode, int ione, int direct_flag, int use_nijbo,
    const int *iorbs, const int *kopt, const int *nijbo, int *out,
    int *fallback_basis) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int i = 0; i < 19; ++i) out[i] = 0;
  for (int i = 0; i < kMozymeResidentFallbackBasisBins *
                          kMozymeResidentFallbackBasisBins; ++i) {
    fallback_basis[i] = 0;
  }
  if (numat <= 0 || !iorbs || !out || !fallback_basis ||
      use_nijbo == 0 || !nijbo || (mode != 0 && !kopt)) {
    out[18] = 1;
    return;
  }

  int one_count = 0;
  int one_center_cpu_count = 0;
  int pair_count = 0;
  int pair4_count = 0;
  int point_count = 0;
  int one_w_count = 0;
  int pair_w_count = 0;
  int real_pair_count = 0;
  int real_pair_gpu_count = 0;
  int real_pair_cpu_count = 0;
  int real_pair_inactive_count = 0;
  int real_pair_basis_limit_count = 0;
  int real_pair_other_count = 0;
  int point_pair_count = 0;
  int point_pair_gpu_count = 0;
  int point_pair_cpu_count = 0;
  int point_pair_basis_limit_count = 0;
  int point_pair_other_count = 0;

  int ired = 1;
  for (int ii = 1; ii <= numat; ++ii) {
    bool calci;
    if (mode == 0) {
      calci = true;
    } else {
      calci = (kopt[ired - 1] == ii);
      if (calci && ired < numat) ++ired;
    }
    const int iab = iorbs[ii - 1];
    if (iab == 0) continue;
    int jred = 1;
    const int iim1 = ii - ione;
    for (int jj = 1; jj <= iim1; ++jj) {
      bool calcj;
      if (mode == 0) {
        calcj = true;
      } else {
        calcj = (kopt[jred - 1] == jj);
        if (calcj && jred < numat) ++jred;
      }
      const int jba = iorbs[jj - 1];
      const int addr = mozyme_resident_nijbo_at(numat, nijbo, ii, jj);
      if (addr >= 0) {
        if (calci || calcj) {
          ++real_pair_count;
          if (mozyme_resident_pair_supported_for_direct_dev(iab, jba,
                                                            direct_flag)) {
            ++real_pair_gpu_count;
            if ((iab == 4 && jba == 1) || (iab == 1 && jba == 4)) {
              ++pair4_count;
            } else {
              ++pair_count;
              pair_w_count += mozyme_resident_pair_integral_count_dev(iab, jba);
            }
          } else if (!mozyme_resident_pair_noop_dev(iab, jba)) {
            ++real_pair_cpu_count;
            if (iab > 9 || jba > 9) {
              ++real_pair_basis_limit_count;
            } else {
              ++real_pair_other_count;
            }
            mozyme_resident_increment_fallback_basis_dev(iab, jba,
                                                        fallback_basis);
          }
        } else {
          ++real_pair_inactive_count;
        }
      } else {
        if ((calci || calcj) && iab * jba > 0) {
          ++point_pair_count;
          if (mozyme_resident_point_supported_for_direct_dev(iab, jba, addr,
                                                             direct_flag)) {
            ++point_pair_gpu_count;
            ++point_count;
          } else {
            ++point_pair_cpu_count;
            if (iab > 9 || jba > 9) {
              ++point_pair_basis_limit_count;
            } else {
              ++point_pair_other_count;
            }
            mozyme_resident_increment_fallback_basis_dev(iab, jba,
                                                        fallback_basis);
          }
        }
      }
    }
    if (direct_flag == 0) {
      const int tri = mozyme_resident_tri_dev(iab);
      if (mozyme_resident_basis_supported_dev(iab)) {
        ++one_count;
        one_w_count += tri * tri;
      } else {
        ++one_center_cpu_count;
        mozyme_resident_increment_fallback_basis_dev(iab, iab,
                                                    fallback_basis);
      }
    }
  }

  if (direct_flag != 0) {
    one_count = 0;
    one_w_count = 0;
    one_center_cpu_count = 0;
    for (int ii = 1; ii <= numat; ++ii) {
      const int iab = iorbs[ii - 1];
      if (iab != 0) {
        const int tri = mozyme_resident_tri_dev(iab);
        if (mozyme_resident_basis_supported_dev(iab)) {
          ++one_count;
          one_w_count += tri * tri;
        } else {
          ++one_center_cpu_count;
          mozyme_resident_increment_fallback_basis_dev(iab, iab,
                                                      fallback_basis);
        }
      }
    }
  }

  out[0] = one_count;
  out[1] = pair_count;
  out[2] = pair4_count;
  out[3] = point_count;
  out[4] = one_w_count;
  out[5] = pair_w_count;
  out[6] = real_pair_count;
  out[7] = real_pair_gpu_count;
  out[8] = real_pair_cpu_count;
  out[9] = real_pair_inactive_count;
  out[10] = real_pair_basis_limit_count;
  out[11] = real_pair_other_count;
  out[12] = point_pair_count;
  out[13] = point_pair_gpu_count;
  out[14] = point_pair_cpu_count;
  out[15] = point_pair_basis_limit_count;
  out[16] = point_pair_other_count;
  out[17] = one_center_cpu_count;
  out[18] = 0;
}

extern "C" int mopac_cuda_mozyme_resident_fock_count_plan(
    int numat, int mode, int ione, int direct_flag, int use_nijbo,
    const int *iorbs, const int *kopt, const int *nijbo, int *counts_out,
    int *fallback_basis_out) {
  if (numat <= 0 || !iorbs || !counts_out || !fallback_basis_out ||
      use_nijbo == 0 || !nijbo || (mode != 0 && !kopt)) {
    return 1;
  }
  cudaStream_t s = g_stream ? g_stream : 0;
  const size_t atom_bytes = sizeof(int) * static_cast<size_t>(numat);
  const size_t nijbo_bytes =
      sizeof(int) * static_cast<size_t>(numat) * static_cast<size_t>(numat);
  if (!g_mz_res_count_iorbs.ensure(atom_bytes) ||
      !g_mz_res_count_nijbo.ensure(nijbo_bytes) ||
      !g_mz_res_count_out.ensure(sizeof(int) * 19u) ||
      !g_mz_res_count_fallback.ensure(
          sizeof(int) * kMozymeResidentFallbackBasisBins *
          kMozymeResidentFallbackBasisBins)) {
    return 2;
  }
  if (mode != 0 && !g_mz_res_count_kopt.ensure(atom_bytes)) return 2;

  cudaError_t status = cudaMemcpyAsync(g_mz_res_count_iorbs.ptr, iorbs,
                                       atom_bytes, cudaMemcpyHostToDevice, s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock count copy iorbs", status);
    return 2;
  }
  status = cudaMemcpyAsync(g_mz_res_count_nijbo.ptr, nijbo, nijbo_bytes,
                           cudaMemcpyHostToDevice, s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock count copy nijbo", status);
    return 2;
  }
  if (mode != 0) {
    status = cudaMemcpyAsync(g_mz_res_count_kopt.ptr, kopt, atom_bytes,
                             cudaMemcpyHostToDevice, s);
    if (status != cudaSuccess) {
      report_cuda_error("resident fock count copy kopt", status);
      return 2;
    }
  }

  mozyme_resident_fock_count_kernel<<<1, 1, 0, s>>>(
      numat, mode, ione, direct_flag, use_nijbo, g_mz_res_count_iorbs.ptr,
      mode != 0 ? g_mz_res_count_kopt.ptr : nullptr, g_mz_res_count_nijbo.ptr,
      g_mz_res_count_out.ptr, g_mz_res_count_fallback.ptr);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    report_cuda_error("resident fock count kernel launch", status);
    return 3;
  }
  status = cudaMemcpyAsync(counts_out, g_mz_res_count_out.ptr,
                           sizeof(int) * 19u, cudaMemcpyDeviceToHost, s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock count copy out", status);
    return 2;
  }
  status = cudaMemcpyAsync(fallback_basis_out, g_mz_res_count_fallback.ptr,
                           sizeof(int) * kMozymeResidentFallbackBasisBins *
                               kMozymeResidentFallbackBasisBins,
                           cudaMemcpyDeviceToHost, s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock count copy fallback", status);
    return 2;
  }
  status = cudaStreamSynchronize(s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock count synchronize", status);
    return 2;
  }
  if (counts_out[18] != 0) return counts_out[18];
  if (gpu_profile_enabled()) {
    const int full =
        (counts_out[8] == 0 && counts_out[14] == 0 &&
         counts_out[17] == 0) ? 1 : 0;
    std::fprintf(stderr,
                 "[GPU] profile mozyme_resident_fock_count atoms=%d mode=%d one=%d pair=%d pair4x1=%d point=%d full=%d\n",
                 numat, mode, counts_out[0], counts_out[1], counts_out[2],
                 counts_out[3], full);
  }
  return 0;
}

__device__ __forceinline__ double mozyme_resident_coord_at_dev(
    const double *coord, int dim, int atom_fortran) {
  return coord[(atom_fortran - 1) * 3 + dim];
}

__device__ __forceinline__ void mozyme_resident_to_point_dev(
    double r, double ev, double a0, double trunc_1, double trunc_2,
    double *point, double *constant) {
  *point = ev * a0 / r;
  if (r < trunc_1) {
    const double delta = r - trunc_1;
    *constant = 1.0 - exp(-(delta * delta) * trunc_2);
  } else {
    *constant = 0.0;
  }
}

__device__ __forceinline__ bool mozyme_resident_valid_plan_range_dev(
    int offset_fortran, int count, int limit) {
  if (offset_fortran < 1 || count < 0 || limit < 0) return false;
  const long long start = static_cast<long long>(offset_fortran) - 1ll;
  return start + static_cast<long long>(count) <= static_cast<long long>(limit);
}

__device__ __forceinline__ bool mozyme_resident_valid_source_range_dev(
    int offset_zero, int count, int limit) {
  if (offset_zero < 0 || count < 0 || limit < 0) return false;
  return static_cast<long long>(offset_zero) + static_cast<long long>(count) <=
         static_cast<long long>(limit);
}

__device__ __forceinline__ void mozyme_resident_advance_point_kr_dev(
    int iab, int jba, int addr, int direct_flag, int semidr_flag, int *kr) {
  if (iab * jba <= 0 || direct_flag != 0 || semidr_flag != 0) return;
  *kr += 1;
  if (addr == -2) {
    if (iab > 1 && jba > 1) {
      *kr += 6;
    } else if (iab > 1 || jba > 1) {
      *kr += 3;
    }
  }
}

static __device__ __constant__ int kMozymeMndodMet[46] = {
  0, 1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6,
  4, 5, 5, 5, 6, 6, 4, 5, 5, 5, 6, 6, 6, 4, 5, 5,
  5, 6, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 6
};

static __device__ __constant__ int kMozymeMndodIpos[35] = {
  0, 1, 5, 11, 12, 12, 2, 6, 13, 14, 14, 3, 8, 16, 18, 18,
  7, 15, 10, 20, 4, 9, 17, 19, 21, 7, 15, 10, 20, 22, 4, 9,
  17, 21, 19
};

static __device__ __constant__ int kMozymeMndodInddd[36] = {
  0, 0, 0, 0, 0, 0, 0, 1, 6, 7, 9, 12, 0, 6, 2, 8,
  10, 13, 0, 7, 8, 3, 11, 14, 0, 9, 10, 11, 4, 15, 0, 12,
  13, 14, 15, 5
};

static __device__ __constant__ int kMozymeMndodInd2RowOffset[47] = {
  0, 0, 14, 28, 38, 48, 62, 72, 82, 91, 98, 112, 122, 132, 146, 156, 166, 175, 182, 199,
  206, 216, 233, 240, 250, 260, 277, 287, 294, 311, 321, 331, 345, 355, 365, 374, 381, 398, 405, 415,
  425, 442, 452, 462, 476, 477, 491
};

static __device__ __constant__ int kMozymeMndodInd2Col[491] = {
  1, 2, 5, 10, 13, 18, 21, 25, 28, 31, 36, 40, 43, 45, 1, 2, 5, 10, 13, 18,
  21, 25, 28, 31, 36, 40, 43, 45, 3, 6, 11, 14, 20, 23, 30, 32, 38, 42, 4, 7,
  12, 15, 24, 26, 29, 33, 39, 41, 1, 2, 5, 10, 13, 18, 21, 25, 28, 31, 36, 40,
  43, 45, 3, 6, 11, 14, 20, 23, 30, 32, 38, 42, 4, 7, 12, 15, 24, 26, 29, 33,
  39, 41, 8, 16, 18, 21, 25, 28, 34, 36, 40, 9, 17, 19, 22, 27, 35, 37, 1, 2,
  5, 10, 13, 18, 21, 25, 28, 31, 36, 40, 43, 45, 3, 6, 11, 14, 20, 23, 30, 32,
  38, 42, 4, 7, 12, 15, 24, 26, 29, 33, 39, 41, 1, 2, 5, 10, 13, 18, 21, 25,
  28, 31, 36, 40, 43, 45, 3, 6, 11, 14, 20, 23, 30, 32, 38, 42, 4, 7, 12, 15,
  24, 26, 29, 33, 39, 41, 8, 16, 18, 21, 25, 28, 34, 36, 40, 9, 17, 19, 22, 27,
  35, 37, 1, 2, 5, 8, 10, 13, 16, 18, 21, 25, 28, 31, 34, 36, 40, 43, 45, 9,
  17, 19, 22, 27, 35, 37, 3, 6, 11, 14, 20, 23, 30, 32, 38, 42, 1, 2, 5, 8,
  10, 13, 16, 18, 21, 25, 28, 31, 34, 36, 40, 43, 45, 9, 17, 19, 22, 27, 35, 37,
  3, 6, 11, 14, 20, 23, 30, 32, 38, 42, 4, 7, 12, 15, 24, 26, 29, 33, 39, 41,
  1, 2, 5, 8, 10, 13, 16, 18, 21, 25, 28, 31, 34, 36, 40, 43, 45, 4, 7, 12,
  15, 24, 26, 29, 33, 39, 41, 9, 17, 19, 22, 27, 35, 37, 1, 2, 5, 8, 10, 13,
  16, 18, 21, 25, 28, 31, 34, 36, 40, 43, 45, 4, 7, 12, 15, 24, 26, 29, 33, 39,
  41, 3, 6, 11, 14, 20, 23, 30, 32, 38, 42, 1, 2, 5, 10, 13, 18, 21, 25, 28,
  31, 36, 40, 43, 45, 3, 6, 11, 14, 20, 23, 30, 32, 38, 42, 4, 7, 12, 15, 24,
  26, 29, 33, 39, 41, 8, 16, 18, 21, 25, 28, 34, 36, 40, 9, 17, 19, 22, 27, 35,
  37, 1, 2, 5, 8, 10, 13, 16, 18, 21, 25, 28, 31, 34, 36, 40, 43, 45, 9, 17,
  19, 22, 27, 35, 37, 3, 6, 11, 14, 20, 23, 30, 32, 38, 42, 4, 7, 12, 15, 24,
  26, 29, 33, 39, 41, 1, 2, 5, 8, 10, 13, 16, 18, 21, 25, 28, 31, 34, 36, 40,
  43, 45, 4, 7, 12, 15, 24, 26, 29, 33, 39, 41, 3, 6, 11, 14, 20, 23, 30, 32,
  38, 42, 1, 2, 5, 10, 13, 18, 21, 25, 28, 31, 36, 40, 43, 45, 44, 1, 2, 5,
  10, 13, 18, 21, 25, 28, 31, 36, 40, 43, 45
};

static __device__ __constant__ int kMozymeMndodInd2Value[491] = {
  1, 2, 35, 3, 36, 4, 38, 5, 40, 37, 39, 41, 42, 43, 6, 7, 44, 8, 45, 9,
  47, 10, 49, 46, 48, 50, 51, 52, 16, 63, 17, 64, 62, 66, 68, 65, 67, 69, 25, 91,
  26, 92, 96, 90, 94, 93, 97, 95, 124, 125, 129, 126, 130, 127, 132, 128, 134, 131, 133, 135,
  136, 137, 186, 189, 187, 190, 188, 192, 194, 191, 193, 195, 257, 260, 258, 261, 265, 259, 263, 262,
  266, 264, 341, 342, 335, 337, 336, 339, 343, 338, 340, 420, 421, 416, 418, 417, 422, 419, 11, 12,
  53, 13, 54, 14, 56, 15, 58, 55, 57, 59, 60, 61, 18, 71, 19, 72, 70, 74, 76, 73,
  75, 77, 27, 99, 28, 100, 104, 98, 102, 101, 105, 103, 138, 139, 143, 140, 144, 141, 146, 142,
  148, 145, 147, 149, 150, 151, 196, 199, 197, 200, 198, 202, 204, 201, 203, 205, 267, 270, 268, 271,
  275, 269, 273, 272, 276, 274, 350, 351, 344, 346, 345, 348, 352, 347, 349, 427, 428, 423, 425, 424,
  429, 426, 20, 21, 78, 85, 22, 79, 86, 23, 81, 24, 83, 80, 87, 82, 84, 88, 89, 109,
  110, 29, 107, 106, 111, 108, 152, 155, 153, 156, 154, 158, 160, 157, 159, 161, 206, 207, 211, 218,
  208, 212, 219, 209, 214, 210, 216, 213, 220, 215, 217, 221, 222, 281, 282, 277, 279, 278, 283, 280,
  353, 356, 354, 357, 355, 359, 361, 358, 360, 362, 430, 433, 431, 434, 438, 432, 436, 435, 439, 437,
  30, 31, 112, 119, 32, 113, 120, 33, 115, 34, 117, 114, 121, 116, 118, 122, 123, 162, 165, 163,
  166, 170, 164, 168, 167, 171, 169, 227, 228, 223, 225, 224, 229, 226, 284, 285, 289, 296, 286, 290,
  297, 287, 292, 288, 294, 291, 298, 293, 295, 299, 300, 363, 366, 364, 367, 371, 365, 369, 368, 372,
  370, 440, 443, 441, 444, 442, 446, 448, 445, 447, 449, 172, 173, 177, 174, 178, 175, 180, 176, 182,
  179, 181, 183, 184, 185, 230, 233, 231, 234, 232, 236, 238, 235, 237, 239, 301, 304, 302, 305, 309,
  303, 307, 306, 310, 308, 379, 380, 373, 375, 374, 377, 381, 376, 378, 454, 455, 450, 452, 451, 456,
  453, 240, 241, 245, 252, 242, 246, 253, 243, 248, 244, 250, 247, 254, 249, 251, 255, 256, 315, 316,
  311, 313, 312, 317, 314, 382, 385, 383, 386, 384, 388, 390, 387, 389, 391, 457, 460, 458, 461, 465,
  459, 463, 462, 466, 464, 318, 319, 323, 330, 320, 324, 331, 321, 326, 322, 328, 325, 332, 327, 329,
  333, 334, 392, 395, 393, 396, 400, 394, 398, 397, 401, 399, 467, 470, 468, 471, 469, 473, 475, 472,
  474, 476, 402, 403, 407, 404, 408, 405, 410, 406, 412, 409, 411, 413, 414, 415, 477, 478, 479, 483,
  480, 484, 481, 486, 482, 488, 485, 487, 489, 490, 491
};

static __device__ __constant__ int kMozymeMndodIsym[492] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  38, 39, 0, 42, 0, 0, 0, 0, 0, 47, 48, 0, 51, 0, 0, 0, 0, 0, 56, 57,
  0, 60, 0, 0, 0, 0, 0, 0, 66, 67, 0, 0, 0, 0, 0, 0, 74, 75, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 62, 63, 64, 65, -66, -67, 66, 67, 70, 71,
  72, 73, -74, -75, 74, 75, 86, 86, 0, 85, 86, 87, 78, 79, 80, 83, 84, 81, 82, -85,
  -86, -87, 88, 88, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 132, 133, 0, 136, 0, 0,
  0, 0, 141, 0, 0, 0, 0, 0, 146, 147, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0,
  158, 159, 152, 153, 154, 155, 156, 157, -158, -159, 158, 159, 0, 0, 0, 0, 175, 0, 0, 0,
  0, 0, 180, 181, 0, 184, 0, 0, 0, 0, 0, 0, 0, 0, 192, 193, 0, 0, 0, 0,
  0, 0, 0, 0, 202, 203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 221, 0, 219, 219, 0, 218, 219, 220, 0, 0, 0, 0, 0, 0, 0, 0, 236, 237,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 186, 187, 188,
  189, 190, 191, -192, -193, 192, 193, 196, 197, 198, 199, 200, 201, -202, -203, 202, 203, 223, 219, 219,
  226, 218, 219, 220, 206, 207, 208, 210, 209, 211, 212, 213, 216, 217, 214, 215, -218, -219, -220, 221,
  221, 230, 231, 232, 233, 234, 235, -236, -237, 236, 237, 0, 253, 253, 0, 252, 253, 254, 240, 241,
  242, 244, 243, 245, 246, 247, 250, 251, 248, 249, -252, -253, -254, 255, 255, 0, -335, 0, 0, -337,
  -338, 0, 337, 0, 223, -223, 219, 226, -219, -226, 218, 219, 220, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, -353, -354, -355, -356, -357, -358, 359, 360, -361, -362, 0, -373, 0, 0, -375, -376, 0,
  375, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -382, -383, -384, -385, -386, -387, 388, 389,
  -390, -391, 0, 0, 0, 0, 405, 0, 0, 0, 0, 0, 410, 411, 0, 0, 335, 337, 337, 338,
  341, 337, 343, 223, 219, 219, 226, 218, 219, 220, 353, 354, 355, 356, 357, 358, -361, -362, 359, 360,
  353, 354, 355, 356, 357, 358, 361, 362, 359, 360, 373, 375, 375, 376, 379, 375, 381, 382, 383, 384,
  385, 386, 387, -390, -391, 388, 389, 382, 383, 384, 385, 386, 387, 390, 391, 388, 389, 0, 402, 403,
  404, 405, 405, 407, 408, 409, 410, 411, 410, 411, 415, 414
};

static __device__ __forceinline__ int mozyme_mndod_indx_dev(int i, int j) {
  const int a = i > j ? i : j;
  const int b = i > j ? j : i;
  return (a * (a - 1)) / 2 + b;
}

static __device__ __forceinline__ int mozyme_mndod_indexd_dev(int i, int j) {
  const int a = i > j ? i : j;
  const int b = i > j ? j : i;
  return (-(b * (b - 1)) / 2) + a + 9 * (b - 1);
}

static __device__ __forceinline__ int mozyme_mndod_met_dev(int idx) {
  if (idx < 1 || idx > 45) return 0;
  return kMozymeMndodMet[idx];
}

static __device__ __forceinline__ int mozyme_mndod_ind2_dev(int ij, int kl) {
  if (ij < 1 || ij > 45 || kl < 1 || kl > 45) return 0;
  const int begin = kMozymeMndodInd2RowOffset[ij];
  const int end = kMozymeMndodInd2RowOffset[ij + 1];
  for (int idx = begin; idx < end; ++idx) {
    if (kMozymeMndodInd2Col[idx] == kl) return kMozymeMndodInd2Value[idx];
  }
  return 0;
}

static __device__ __forceinline__ int mozyme_mndod_isym_dev(int idx) {
  if (idx < 1 || idx > 491) return 0;
  return kMozymeMndodIsym[idx];
}

static __device__ __forceinline__ bool mozyme_resident_pack_point_weights_dev(
    int iab, int jba, int addr, int kr, int ii, int jj, int direct_flag,
    int semidr_flag, int l_feather_flag, double ev, double a0,
    double trunc_1, double trunc_2, int w_count, const int *nat,
    const double *coord, const double *wj, const double *am,
    const double *ad, const double *dd, double *values) {
  for (int i = 0; i < 7; ++i) values[i] = 0.0;
  if (iab * jba <= 0) return true;

  if (direct_flag != 0 || semidr_flag != 0) {
    const int ni = nat[ii - 1];
    const int nj = nat[jj - 1];
    if (ni < 1 || ni > 107 || nj < 1 || nj > 107) return false;
    const double dx0 = mozyme_resident_coord_at_dev(coord, 0, ii) -
                       mozyme_resident_coord_at_dev(coord, 0, jj);
    const double dy0 = mozyme_resident_coord_at_dev(coord, 1, ii) -
                       mozyme_resident_coord_at_dev(coord, 1, jj);
    const double dz0 = mozyme_resident_coord_at_dev(coord, 2, ii) -
                       mozyme_resident_coord_at_dev(coord, 2, jj);
    const double r2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
    const double aee = 0.5 / am[ni - 1] + 0.5 / am[nj - 1];
    values[0] = ev / sqrt(r2 / (a0 * a0) + aee * aee);
    if (l_feather_flag != 0) {
      double point = 0.0;
      double constant = 1.0;
      mozyme_resident_to_point_dev(sqrt(r2), ev, a0, trunc_1, trunc_2,
                                   &point, &constant);
      values[0] = values[0] * constant + (1.0 - constant) * point;
    }

    if (addr == -2) {
      const double r = sqrt(r2);
      double dx = dx0;
      double dy = dy0;
      double dz = dz0;
      if (r > 0.0) {
        dx /= r;
        dy /= r;
        dz /= r;
      }
      if (fabs(dz) > 0.99999999) dz = copysign(1.0, dz);

      if (iab > 1) {
        const double da = dd[ni - 1];
        const double ade = 0.5 / ad[ni - 1] + 0.5 / am[nj - 1];
        const double rp = sqrt((r / a0 + da) * (r / a0 + da) + ade * ade);
        const double rm = sqrt((r / a0 - da) * (r / a0 - da) + ade * ade);
        double ri2 = ev * (0.5 / rp - 0.5 / rm);
        if (l_feather_flag != 0) {
          double point = 0.0;
          double constant = 1.0;
          mozyme_resident_to_point_dev(r, ev, a0, trunc_1, trunc_2, &point,
                                       &constant);
          ri2 *= constant;
        }
        values[4] = ri2 * dx;
        values[5] = ri2 * dy;
        values[6] = ri2 * dz;
      }

      if (jba > 1) {
        const double da = dd[nj - 1];
        const double ade = 0.5 / am[ni - 1] + 0.5 / ad[nj - 1];
        const double rp = sqrt((r / a0 + da) * (r / a0 + da) + ade * ade);
        const double rm = sqrt((r / a0 - da) * (r / a0 - da) + ade * ade);
        double ri5 = -ev * (0.5 / rp - 0.5 / rm);
        if (l_feather_flag != 0) {
          double point = 0.0;
          double constant = 1.0;
          mozyme_resident_to_point_dev(r, ev, a0, trunc_1, trunc_2, &point,
                                       &constant);
          ri5 *= constant;
        }
        values[1] = ri5 * dx;
        values[2] = ri5 * dy;
        values[3] = ri5 * dz;
      }
    }
    return true;
  }

  if (!mozyme_resident_valid_source_range_dev(kr, 1, w_count)) return false;
  values[0] = wj[kr];
  if (addr == -2) {
    int required = 1;
    if (iab > 1 && jba > 1) {
      required = 7;
    } else if (iab > 1 || jba > 1) {
      required = 4;
    }
    if (!mozyme_resident_valid_source_range_dev(kr, required, w_count)) return false;
    if (iab > 1 && jba > 1) {
      values[1] = wj[kr + 1];
      values[2] = wj[kr + 2];
      values[3] = wj[kr + 3];
      values[4] = wj[kr + 4];
      values[5] = wj[kr + 5];
      values[6] = wj[kr + 6];
    } else if (iab > 1) {
      values[4] = wj[kr + 1];
      values[5] = wj[kr + 2];
      values[6] = wj[kr + 3];
    } else if (jba > 1) {
      values[1] = wj[kr + 1];
      values[2] = wj[kr + 2];
      values[3] = wj[kr + 3];
    }
  }
  return true;
}

__device__ __forceinline__ int mozyme_mndod_indx_sp_dev(int i, int j) {
  const int a = i > j ? i : j;
  const int b = i > j ? j : i;
  return (a * (a - 1)) / 2 + b;
}

__device__ __forceinline__ int mozyme_mndod_indexd_sp_dev(int i, int j) {
  const int a = i > j ? i : j;
  const int b = i > j ? j : i;
  return (-(b * (b - 1)) / 2) + a + 9 * (b - 1);
}

__device__ __forceinline__ int mozyme_mndod_met_sp_dev(int idx) {
  switch (idx) {
    case 1: return 1;
    case 2: return 2;
    case 3: return 3;
    case 4: return 2;
    case 5: return 3;
    case 6: return 3;
    case 7: return 2;
    case 8: return 3;
    case 9: return 3;
    case 10: return 3;
    default: return 0;
  }
}

__device__ __forceinline__ int mozyme_mndod_indw_sp_dev(int i, int j,
                                                        int limkl, int kl) {
  return (mozyme_mndod_indx_sp_dev(i, j) - 1) * limkl + (kl - 1);
}

__device__ __forceinline__ int mozyme_mndod_ind2_sp_dev(int ij, int kl) {
  if (ij == 1) {
    if (kl == 1) return 1;
    if (kl == 2) return 2;
    if (kl == 10) return 3;
    if (kl == 18) return 4;
    if (kl == 25) return 5;
  } else if (ij == 2) {
    if (kl == 1) return 6;
    if (kl == 2) return 7;
    if (kl == 10) return 8;
    if (kl == 18) return 9;
    if (kl == 25) return 10;
  } else if (ij == 10) {
    if (kl == 1) return 11;
    if (kl == 2) return 12;
    if (kl == 10) return 13;
    if (kl == 18) return 14;
    if (kl == 25) return 15;
  } else if (ij == 3) {
    if (kl == 3) return 16;
    if (kl == 11) return 17;
  } else if (ij == 11) {
    if (kl == 3) return 18;
    if (kl == 11) return 19;
  } else if (ij == 18) {
    if (kl == 1) return 20;
    if (kl == 2) return 21;
    if (kl == 10) return 22;
    if (kl == 18) return 23;
    if (kl == 25) return 24;
  } else if (ij == 4) {
    if (kl == 4) return 25;
    if (kl == 12) return 26;
  } else if (ij == 12) {
    if (kl == 4) return 27;
    if (kl == 12) return 28;
  } else if (ij == 19) {
    if (kl == 19) return 29;
  } else if (ij == 25) {
    if (kl == 1) return 30;
    if (kl == 2) return 31;
    if (kl == 10) return 32;
    if (kl == 18) return 33;
    if (kl == 25) return 34;
  }
  return 0;
}

__device__ __forceinline__ double mozyme_mndod_rep_sp_dev(int nd,
                                                          const double *ri) {
  switch (nd) {
    case 1: return ri[0];
    case 2: return ri[4];
    case 3: return ri[10];
    case 4: return ri[11];
    case 5: return ri[11];
    case 6: return ri[1];
    case 7: return ri[5];
    case 8: return ri[12];
    case 9: return ri[13];
    case 10: return ri[13];
    case 11: return ri[2];
    case 12: return ri[7];
    case 13: return ri[15];
    case 14: return ri[17];
    case 15: return ri[17];
    case 16: return ri[6];
    case 17: return ri[14];
    case 18: return ri[9];
    case 19: return ri[19];
    case 20: return ri[3];
    case 21: return ri[8];
    case 22: return ri[16];
    case 23: return ri[18];
    case 24: return ri[20];
    case 25: return ri[6];
    case 26: return ri[14];
    case 27: return ri[9];
    case 28: return ri[19];
    case 29: return ri[21];
    case 30: return ri[3];
    case 31: return ri[8];
    case 32: return ri[16];
    case 33: return ri[20];
    case 34: return ri[18];
    default: return 0.0;
  }
}

static __device__ __forceinline__ bool mozyme_direct_reppd_sp_dev(
    int ni, int nj, int iab, int jba, double rij, int l_feather_flag,
    double ev, double a0, double trunc_1, double trunc_2, const double *am,
    const double *ad, const double *aq, const double *dd, const double *qq,
    double *ri, double *arg, double *sqr) {
  if (ni < 1 || ni > 107 || nj < 1 || nj > 107 || rij <= 0.0 ||
      !am || !ad || !aq || !dd || !qq || !ri || !arg || !sqr) {
    return false;
  }
  for (int i = 0; i < 22; ++i) ri[i] = 0.0;

  const double td = 2.0;
  const double pp_half = 0.5;
  const double ev1 = ev / 2.0;
  const double ev2 = ev1 / 2.0;
  const double ev3 = ev2 / 2.0;
  const double ev4 = ev3 / 2.0;
  const double r = rij / a0;
  const bool si = iab >= 3;
  const bool sj = jba >= 3;
  double aee = pp_half / am[ni - 1] + pp_half / am[nj - 1];
  aee *= aee;

  if (!si && !sj) {
    ri[0] = ev / sqrt(r * r + aee);
  } else if (si && !sj) {
    const double da = dd[ni - 1];
    const double qa2 = qq[ni - 1] * td;
    double ade = pp_half / ad[ni - 1] + pp_half / am[nj - 1];
    double aqe = pp_half / aq[ni - 1] + pp_half / am[nj - 1];
    ade *= ade;
    aqe *= aqe;
    const double rsq = r * r;
    arg[0] = rsq + aee;
    double x = r + da;
    arg[1] = x * x + ade;
    x = r - da;
    arg[2] = x * x + ade;
    x = r + qa2;
    arg[3] = x * x + aqe;
    x = r - qa2;
    arg[4] = x * x + aqe;
    arg[5] = rsq + aqe;
    arg[6] = arg[5] + qa2 * qa2;
    for (int i = 0; i < 7; ++i) sqr[i] = sqrt(arg[i]);
    const double ee = ev / sqr[0];
    ri[0] = ee;
    ri[1] = ev1 / sqr[1] - ev1 / sqr[2];
    ri[2] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
    ri[3] = ee + ev1 / sqr[6] - ev1 / sqr[5];
  } else if (!si && sj) {
    const double db = dd[nj - 1];
    const double qb2 = qq[nj - 1] * td;
    double aed = pp_half / am[ni - 1] + pp_half / ad[nj - 1];
    double aeq = pp_half / am[ni - 1] + pp_half / aq[nj - 1];
    aed *= aed;
    aeq *= aeq;
    const double rsq = r * r;
    arg[0] = rsq + aee;
    double x = r - db;
    arg[1] = x * x + aed;
    x = r + db;
    arg[2] = x * x + aed;
    x = r - qb2;
    arg[3] = x * x + aeq;
    x = r + qb2;
    arg[4] = x * x + aeq;
    arg[5] = rsq + aeq;
    arg[6] = arg[5] + qb2 * qb2;
    for (int i = 0; i < 7; ++i) sqr[i] = sqrt(arg[i]);
    const double ee = ev / sqr[0];
    ri[0] = ee;
    ri[4] = ev1 / sqr[1] - ev1 / sqr[2];
    ri[10] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
    ri[11] = ee + ev1 / sqr[6] - ev1 / sqr[5];
  } else {
    const double da = dd[ni - 1];
    const double db = dd[nj - 1];
    double qa2 = qq[ni - 1] * td;
    double qb2 = qq[nj - 1] * td;
    double ade = pp_half / ad[ni - 1] + pp_half / am[nj - 1];
    double aqe = pp_half / aq[ni - 1] + pp_half / am[nj - 1];
    double aed = pp_half / am[ni - 1] + pp_half / ad[nj - 1];
    double aeq = pp_half / am[ni - 1] + pp_half / aq[nj - 1];
    double axx = pp_half / ad[ni - 1] + pp_half / ad[nj - 1];
    double adq = pp_half / ad[ni - 1] + pp_half / aq[nj - 1];
    double aqd = pp_half / aq[ni - 1] + pp_half / ad[nj - 1];
    double aqq = pp_half / aq[ni - 1] + pp_half / aq[nj - 1];
    ade *= ade;
    aqe *= aqe;
    aed *= aed;
    aeq *= aeq;
    axx *= axx;
    adq *= adq;
    aqd *= aqd;
    aqq *= aqq;
    const double rsq = r * r;
    arg[0] = rsq + aee;
    double x = r + da; arg[1] = x * x + ade;
    x = r - da; arg[2] = x * x + ade;
    x = r - qa2; arg[3] = x * x + aqe;
    x = r + qa2; arg[4] = x * x + aqe;
    arg[5] = rsq + aqe;
    arg[6] = arg[5] + qa2 * qa2;
    x = r - db; arg[7] = x * x + aed;
    x = r + db; arg[8] = x * x + aed;
    x = r - qb2; arg[9] = x * x + aeq;
    x = r + qb2; arg[10] = x * x + aeq;
    arg[11] = rsq + aeq;
    arg[12] = arg[11] + qb2 * qb2;
    x = da - db; arg[13] = rsq + axx + x * x;
    x = da + db; arg[14] = rsq + axx + x * x;
    x = r + da - db; arg[15] = x * x + axx;
    x = r - da + db; arg[16] = x * x + axx;
    x = r - da - db; arg[17] = x * x + axx;
    x = r + da + db; arg[18] = x * x + axx;
    x = r + da; arg[19] = x * x + adq; arg[20] = arg[19] + qb2 * qb2;
    x = r - da; arg[21] = x * x + adq; arg[22] = arg[21] + qb2 * qb2;
    x = r - db; arg[23] = x * x + aqd; arg[24] = arg[23] + qa2 * qa2;
    x = r + db; arg[25] = x * x + aqd; arg[26] = arg[25] + qa2 * qa2;
    x = r + da - qb2; arg[27] = x * x + adq;
    x = r - da - qb2; arg[28] = x * x + adq;
    x = r + da + qb2; arg[29] = x * x + adq;
    x = r - da + qb2; arg[30] = x * x + adq;
    x = r + qa2 - db; arg[31] = x * x + aqd;
    x = r + qa2 + db; arg[32] = x * x + aqd;
    x = r - qa2 - db; arg[33] = x * x + aqd;
    x = r - qa2 + db; arg[34] = x * x + aqd;
    arg[35] = rsq + aqq;
    x = qa2 - qb2; arg[36] = arg[35] + x * x;
    x = qa2 + qb2; arg[37] = arg[35] + x * x;
    arg[38] = arg[35] + qa2 * qa2;
    arg[39] = arg[35] + qb2 * qb2;
    arg[40] = arg[38] + qb2 * qb2;
    x = r - qb2; arg[41] = x * x + aqq; arg[42] = arg[41] + qa2 * qa2;
    x = r + qb2; arg[43] = x * x + aqq; arg[44] = arg[43] + qa2 * qa2;
    x = r + qa2; arg[45] = x * x + aqq; arg[46] = arg[45] + qb2 * qb2;
    x = r - qa2; arg[47] = x * x + aqq; arg[48] = arg[47] + qb2 * qb2;
    x = r + qa2 - qb2; arg[49] = x * x + aqq;
    x = r + qa2 + qb2; arg[50] = x * x + aqq;
    x = r - qa2 - qb2; arg[51] = x * x + aqq;
    x = r - qa2 + qb2; arg[52] = x * x + aqq;
    const double qa = qq[ni - 1];
    const double qb = qq[nj - 1];
    x = da - qb; double xxx = x * x;
    x = r - qb; double yyy = x * x;
    x = da + qb; double zzz = x * x;
    x = r + qb; double www = x * x;
    arg[53] = xxx + yyy + adq;
    arg[54] = xxx + www + adq;
    arg[55] = zzz + yyy + adq;
    arg[56] = zzz + www + adq;
    x = qa - db; xxx = x * x;
    x = qa + db; yyy = x * x;
    x = r + qa; zzz = x * x;
    x = r - qa; www = x * x;
    arg[57] = zzz + xxx + aqd;
    arg[58] = www + xxx + aqd;
    arg[59] = zzz + yyy + aqd;
    arg[60] = www + yyy + aqd;
    x = qa - qb; xxx = x * x;
    arg[61] = arg[35] + td * xxx;
    x = qa + qb; yyy = x * x;
    arg[62] = arg[35] + td * yyy;
    arg[63] = arg[35] + td * (qa * qa + qb * qb);
    x = r + qa - qb; zzz = x * x;
    arg[64] = zzz + xxx + aqq;
    arg[65] = zzz + yyy + aqq;
    x = r + qa + qb; zzz = x * x;
    arg[66] = zzz + xxx + aqq;
    arg[67] = zzz + yyy + aqq;
    x = r - qa - qb; zzz = x * x;
    arg[68] = zzz + xxx + aqq;
    arg[69] = zzz + yyy + aqq;
    x = r - qa + qb; zzz = x * x;
    arg[70] = zzz + xxx + aqq;
    arg[71] = zzz + yyy + aqq;
    for (int i = 0; i < 72; ++i) sqr[i] = sqrt(arg[i]);
    const double ee = ev / sqr[0];
    const double dze = (-ev1 / sqr[1]) + ev1 / sqr[2];
    const double qzze = ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
    const double qxxe = ev1 / sqr[6] - ev1 / sqr[5];
    const double edz = (-ev1 / sqr[7]) + ev1 / sqr[8];
    const double eqzz = ev2 / sqr[9] + ev2 / sqr[10] - ev1 / sqr[11];
    const double eqxx = ev1 / sqr[12] - ev1 / sqr[11];
    const double dxdx = ev1 / sqr[13] - ev1 / sqr[14];
    const double dzdz = ev2 / sqr[15] + ev2 / sqr[16] - ev2 / sqr[17] - ev2 / sqr[18];
    const double dzqxx = ev2 / sqr[19] - ev2 / sqr[20] - ev2 / sqr[21] + ev2 / sqr[22];
    const double qxxdz = ev2 / sqr[23] - ev2 / sqr[24] - ev2 / sqr[25] + ev2 / sqr[26];
    const double dzqzz = (-ev3 / sqr[27]) + ev3 / sqr[28] - ev3 / sqr[29] + ev3 / sqr[30] -
                         ev2 / sqr[21] + ev2 / sqr[19];
    const double qzzdz = (-ev3 / sqr[31]) + ev3 / sqr[32] - ev3 / sqr[33] + ev3 / sqr[34] +
                         ev2 / sqr[23] - ev2 / sqr[25];
    const double qxxqxx = ev3 / sqr[36] + ev3 / sqr[37] - ev2 / sqr[38] - ev2 / sqr[39] +
                          ev2 / sqr[35];
    const double qxxqyy = ev2 / sqr[40] - ev2 / sqr[38] - ev2 / sqr[39] + ev2 / sqr[35];
    const double qxxqzz = ev3 / sqr[42] + ev3 / sqr[44] - ev3 / sqr[41] - ev3 / sqr[43] -
                          ev2 / sqr[38] + ev2 / sqr[35];
    const double qzzqxx = ev3 / sqr[46] + ev3 / sqr[48] - ev3 / sqr[45] - ev3 / sqr[47] -
                          ev2 / sqr[39] + ev2 / sqr[35];
    const double qzzqzz = ev4 / sqr[49] + ev4 / sqr[50] + ev4 / sqr[51] + ev4 / sqr[52] -
                          ev3 / sqr[47] - ev3 / sqr[45] - ev3 / sqr[41] - ev3 / sqr[43] +
                          ev2 / sqr[35];
    const double dxqxz = (-ev2 / sqr[53]) + ev2 / sqr[54] + ev2 / sqr[55] - ev2 / sqr[56];
    const double qxzdx = (-ev2 / sqr[57]) + ev2 / sqr[58] + ev2 / sqr[59] - ev2 / sqr[60];
    const double qxzqxz = ev3 / sqr[64] - ev3 / sqr[66] - ev3 / sqr[68] + ev3 / sqr[70] -
                          ev3 / sqr[65] + ev3 / sqr[67] + ev3 / sqr[69] - ev3 / sqr[71];
    ri[0] = ee;
    ri[1] = -dze;
    ri[2] = ee + qzze;
    ri[3] = ee + qxxe;
    ri[4] = -edz;
    ri[5] = dzdz;
    ri[6] = dxdx;
    ri[7] = (-edz) - qzzdz;
    ri[8] = (-edz) - qxxdz;
    ri[9] = -qxzdx;
    ri[10] = ee + eqzz;
    ri[11] = ee + eqxx;
    ri[12] = (-dze) - dzqzz;
    ri[13] = (-dze) - dzqxx;
    ri[14] = -dxqxz;
    ri[15] = ee + eqzz + qzze + qzzqzz;
    ri[16] = ee + eqzz + qxxe + qxxqzz;
    ri[17] = ee + eqxx + qzze + qzzqxx;
    ri[18] = ee + eqxx + qxxe + qxxqxx;
    ri[19] = qxzqxz;
    ri[20] = ee + eqxx + qxxe + qxxqyy;
    ri[21] = pp_half * (qxxqxx - qxxqyy);
  }

  if (l_feather_flag != 0) {
    double point = 0.0;
    double constant = 1.0;
    mozyme_resident_to_point_dev(rij, ev, a0, trunc_1, trunc_2, &point,
                                 &constant);
    ri[0] = ri[0] * constant + (1.0 - constant) * point;
    ri[1] *= constant;
    ri[2] = ri[2] * constant + (1.0 - constant) * point;
    ri[3] = ri[3] * constant + (1.0 - constant) * point;
    ri[4] *= constant;
    ri[5] *= constant;
    ri[6] *= constant;
    ri[7] *= constant;
    ri[8] *= constant;
    ri[9] *= constant;
    ri[10] = ri[10] * constant + (1.0 - constant) * point;
    ri[11] = ri[11] * constant + (1.0 - constant) * point;
    ri[12] *= constant;
    ri[13] *= constant;
    ri[14] *= constant;
    ri[15] = ri[15] * constant + (1.0 - constant) * point;
    ri[16] = ri[16] * constant + (1.0 - constant) * point;
    ri[17] = ri[17] * constant + (1.0 - constant) * point;
    ri[18] = ri[18] * constant + (1.0 - constant) * point;
    ri[19] *= constant;
    ri[20] = ri[20] * constant + (1.0 - constant) * point;
    ri[21] *= constant;
  }

  ri[1] = -ri[1];
  ri[4] = -ri[4];
  ri[7] = -ri[7];
  ri[8] = -ri[8];
  ri[9] = -ri[9];
  ri[12] = -ri[12];
  ri[13] = -ri[13];
  ri[14] = -ri[14];
  return true;
}

static __device__ __forceinline__ bool mozyme_direct_rotmat_sp_dev(
    const double *coord, int ii, int jj, double *rij, double sp[3][3],
    double pp[6][3][3], double *p_scratch) {
  if (!coord || !rij || !p_scratch) return false;
  const double x11 = mozyme_resident_coord_at_dev(coord, 0, jj) -
                     mozyme_resident_coord_at_dev(coord, 0, ii);
  const double x22 = mozyme_resident_coord_at_dev(coord, 1, jj) -
                     mozyme_resident_coord_at_dev(coord, 1, ii);
  const double x33 = mozyme_resident_coord_at_dev(coord, 2, jj) -
                     mozyme_resident_coord_at_dev(coord, 2, ii);
  const double b = x11 * x11 + x22 * x22;
  *rij = sqrt(b + x33 * x33);
  if (*rij <= 0.0) return false;

  const double small = 1.0e-7;
  const double sqb = sqrt(b);
  double sb = sqb / *rij;
  double ca = 0.0;
  double sa = 0.0;
  double cb = 0.0;
  if (sb > small) {
    ca = x11 / sqb;
    sa = x22 / sqb;
    cb = x33 / *rij;
  } else {
    sa = 0.0;
    sb = 0.0;
    if (x33 < 0.0) {
      ca = -1.0;
      cb = -1.0;
    } else if (x33 > 0.0) {
      ca = 1.0;
      cb = 1.0;
    }
  }

  double (*p)[3] = reinterpret_cast<double (*)[3]>(p_scratch);
  p[0][0] = ca * sb;
  p[1][0] = ca * cb;
  p[2][0] = -sa;
  p[0][1] = sa * sb;
  p[1][1] = sa * cb;
  p[2][1] = ca;
  p[0][2] = cb;
  p[1][2] = -sb;
  p[2][2] = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) sp[i][j] = p[i][j];
  }
  for (int a = 0; a < 6; ++a) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) pp[a][i][j] = 0.0;
    }
  }
  for (int k = 0; k < 3; ++k) {
    pp[0][k][k] = p[k][0] * p[k][0];
    pp[1][k][k] = p[k][1] * p[k][1];
    pp[2][k][k] = p[k][2] * p[k][2];
    pp[3][k][k] = p[k][0] * p[k][1];
    pp[4][k][k] = p[k][0] * p[k][2];
    pp[5][k][k] = p[k][1] * p[k][2];
    for (int j = 0; j < k; ++j) {
      pp[0][k][j] = 2.0 * p[k][0] * p[j][0];
      pp[1][k][j] = 2.0 * p[k][1] * p[j][1];
      pp[2][k][j] = 2.0 * p[k][2] * p[j][2];
      pp[3][k][j] = p[k][0] * p[j][1] + p[k][1] * p[j][0];
      pp[4][k][j] = p[k][0] * p[j][2] + p[k][2] * p[j][0];
      pp[5][k][j] = p[k][1] * p[j][2] + p[k][2] * p[j][1];
    }
  }
  return true;
}

__device__ __forceinline__ int mozyme_mndod_lorb_dev(int orb) {
  if (orb <= 1) return 0;
  if (orb <= 4) return 1;
  return 2;
}

__device__ __forceinline__ int mozyme_mndod_met_pair_dev(int i, int j) {
  const int a = i > j ? i : j;
  const int b = i > j ? j : i;
  if (a <= 1) return 1;
  if (a <= 4) return b == 1 ? 2 : 3;
  if (b == 1) return 4;
  if (b <= 4) return 5;
  return 6;
}

__device__ __forceinline__ int mozyme_mndod_inddd_dev(int i, int j) {
  const int a = i > j ? i : j;
  const int b = i > j ? j : i;
  if (a == b) return a;
  return 6 + ((a - 2) * (a - 1)) / 2 + (b - 1);
}

__device__ __forceinline__ double mozyme_mndod_po_at_dev(
    const double *po, int atomic_number, int slot) {
  return po[(atomic_number - 1) * 9 + (slot - 1)];
}

__device__ __forceinline__ double mozyme_mndod_ddp_at_dev(
    const double *ddp, int atomic_number, int slot) {
  return ddp[(atomic_number - 1) * 6 + (slot - 1)];
}

__device__ __forceinline__ double mozyme_mndod_ch_dev(int ij, int l, int m) {
  if (ij < 1 || ij > 45 || l < 0 || l > 2 || m < -2 || m > 2) return 0.0;
  switch (ij * 15 + l * 5 + (m + 2)) {
    case ((1 * 15) + (0 * 5) + (0 + 2)): return 1.0;
    case ((2 * 15) + (1 * 5) + (0 + 2)): return 1.0;
    case ((3 * 15) + (1 * 5) + (1 + 2)): return 1.0;
    case ((4 * 15) + (1 * 5) + (-1 + 2)): return 1.0;
    case ((5 * 15) + (2 * 5) + (0 + 2)): return 1.15470054;
    case ((6 * 15) + (2 * 5) + (1 + 2)): return 1.0;
    case ((7 * 15) + (2 * 5) + (-1 + 2)): return 1.0;
    case ((8 * 15) + (2 * 5) + (2 + 2)): return 1.0;
    case ((9 * 15) + (2 * 5) + (-2 + 2)): return 1.0;
    case ((10 * 15) + (0 * 5) + (0 + 2)): return 1.0;
    case ((10 * 15) + (2 * 5) + (0 + 2)): return 1.33333333;
    case ((11 * 15) + (2 * 5) + (1 + 2)): return 1.0;
    case ((12 * 15) + (2 * 5) + (-1 + 2)): return 1.0;
    case ((13 * 15) + (1 * 5) + (0 + 2)): return 1.15470054;
    case ((14 * 15) + (1 * 5) + (1 + 2)): return 1.0;
    case ((15 * 15) + (1 * 5) + (-1 + 2)): return 1.0;
    case ((18 * 15) + (0 * 5) + (0 + 2)): return 1.0;
    case ((18 * 15) + (2 * 5) + (0 + 2)): return -0.66666667;
    case ((18 * 15) + (2 * 5) + (2 + 2)): return 1.0;
    case ((19 * 15) + (2 * 5) + (-2 + 2)): return 1.0;
    case ((20 * 15) + (1 * 5) + (1 + 2)): return -0.57735027;
    case ((21 * 15) + (1 * 5) + (0 + 2)): return 1.0;
    case ((23 * 15) + (1 * 5) + (1 + 2)): return 1.0;
    case ((24 * 15) + (1 * 5) + (-1 + 2)): return 1.0;
    case ((25 * 15) + (0 * 5) + (0 + 2)): return 1.0;
    case ((25 * 15) + (2 * 5) + (0 + 2)): return -0.66666667;
    case ((25 * 15) + (2 * 5) + (2 + 2)): return -1.0;
    case ((26 * 15) + (1 * 5) + (-1 + 2)): return -0.57735027;
    case ((28 * 15) + (1 * 5) + (0 + 2)): return 1.0;
    case ((29 * 15) + (1 * 5) + (-1 + 2)): return -1.0;
    case ((30 * 15) + (1 * 5) + (1 + 2)): return 1.0;
    case ((31 * 15) + (0 * 5) + (0 + 2)): return 1.0;
    case ((31 * 15) + (2 * 5) + (0 + 2)): return 1.33333333;
    case ((32 * 15) + (2 * 5) + (1 + 2)): return 0.57735027;
    case ((33 * 15) + (2 * 5) + (-1 + 2)): return 0.57735027;
    case ((34 * 15) + (2 * 5) + (2 + 2)): return -1.15470054;
    case ((35 * 15) + (2 * 5) + (-2 + 2)): return -1.15470054;
    case ((36 * 15) + (0 * 5) + (0 + 2)): return 1.0;
    case ((36 * 15) + (2 * 5) + (0 + 2)): return 0.66666667;
    case ((36 * 15) + (2 * 5) + (2 + 2)): return 1.0;
    case ((37 * 15) + (2 * 5) + (-2 + 2)): return 1.0;
    case ((38 * 15) + (2 * 5) + (1 + 2)): return 1.0;
    case ((39 * 15) + (2 * 5) + (-1 + 2)): return 1.0;
    case ((40 * 15) + (0 * 5) + (0 + 2)): return 1.0;
    case ((40 * 15) + (2 * 5) + (0 + 2)): return 0.66666667;
    case ((40 * 15) + (2 * 5) + (2 + 2)): return -1.0;
    case ((41 * 15) + (2 * 5) + (-1 + 2)): return -1.0;
    case ((42 * 15) + (2 * 5) + (1 + 2)): return 1.0;
    case ((43 * 15) + (0 * 5) + (0 + 2)): return 1.0;
    case ((43 * 15) + (2 * 5) + (0 + 2)): return -1.33333333;
    case ((45 * 15) + (0 * 5) + (0 + 2)): return 1.0;
    case ((45 * 15) + (2 * 5) + (0 + 2)): return -1.33333333;
    default: return 0.0;
  }
}

__device__ __forceinline__ double mozyme_mndod_charg_dev(
    double r, int l1, int l2, int m, double da, double db, double add) {
  const double rt2 = 1.4142135623730950488;
  if (l1 == 0 && l2 == 0) {
    return 1.0 / sqrt(r * r + add);
  }
  if (l1 == 1 && l2 == 0) {
    return 0.5 * (-1.0 / sqrt((r + da) * (r + da) + add) +
                  1.0 / sqrt((r - da) * (r - da) + add));
  }
  if (l1 == 0 && l2 == 1) {
    return 0.5 * (1.0 / sqrt((r + db) * (r + db) + add) -
                  1.0 / sqrt((r - db) * (r - db) + add));
  }
  if (l1 == 1 && l2 == 1 && m == 0) {
    const double v = 1.0 / sqrt((r + da - db) * (r + da - db) + add) +
                     1.0 / sqrt((r - da + db) * (r - da + db) + add) -
                     1.0 / sqrt((r - da - db) * (r - da - db) + add) -
                     1.0 / sqrt((r + da + db) * (r + da + db) + add);
    return 0.25 * v;
  }
  if (l1 == 1 && l2 == 1 && m == 1) {
    const double v = 2.0 / sqrt(r * r + (da - db) * (da - db) + add) -
                     2.0 / sqrt(r * r + (da + db) * (da + db) + add);
    return 0.25 * v;
  }
  if (l1 == 0 && l2 == 2) {
    const double v = 1.0 / sqrt((r - db) * (r - db) + add) -
                     2.0 / sqrt(r * r + db * db + add) +
                     1.0 / sqrt((r + db) * (r + db) + add);
    return 0.25 * v;
  }
  if (l1 == 2 && l2 == 0) {
    const double v = 1.0 / sqrt((r - da) * (r - da) + add) -
                     2.0 / sqrt(r * r + da * da + add) +
                     1.0 / sqrt((r + da) * (r + da) + add);
    return 0.25 * v;
  }
  if (l1 == 1 && l2 == 2 && m == 0) {
    const double v = 1.0 / sqrt((r - da - db) * (r - da - db) + add) -
                     2.0 / sqrt((r - da) * (r - da) + db * db + add) +
                     1.0 / sqrt((r + db - da) * (r + db - da) + add) -
                     1.0 / sqrt((r - db + da) * (r - db + da) + add) +
                     2.0 / sqrt((r + da) * (r + da) + db * db + add) -
                     1.0 / sqrt((r + da + db) * (r + da + db) + add);
    return 0.125 * v;
  }
  if (l1 == 2 && l2 == 1 && m == 0) {
    const double v = -1.0 / sqrt((r - da - db) * (r - da - db) + add) +
                     2.0 / sqrt((r - db) * (r - db) + da * da + add) -
                     1.0 / sqrt((r + da - db) * (r + da - db) + add) +
                     1.0 / sqrt((r - da + db) * (r - da + db) + add) -
                     2.0 / sqrt((r + db) * (r + db) + da * da + add) +
                     1.0 / sqrt((r + da + db) * (r + da + db) + add);
    return 0.125 * v;
  }
  if (l1 == 2 && l2 == 2 && m == 0) {
    const double zzzz =
        1.0 / sqrt((r - da - db) * (r - da - db) + add) +
        1.0 / sqrt((r + da + db) * (r + da + db) + add) +
        1.0 / sqrt((r - da + db) * (r - da + db) + add) +
        1.0 / sqrt((r + da - db) * (r + da - db) + add) -
        2.0 / sqrt((r - da) * (r - da) + db * db + add) -
        2.0 / sqrt((r - db) * (r - db) + da * da + add) -
        2.0 / sqrt((r + da) * (r + da) + db * db + add) -
        2.0 / sqrt((r + db) * (r + db) + da * da + add) +
        2.0 / sqrt(r * r + (da - db) * (da - db) + add) +
        2.0 / sqrt(r * r + (da + db) * (da + db) + add);
    const double xyxy =
        4.0 / sqrt(r * r + (da - db) * (da - db) + add) +
        4.0 / sqrt(r * r + (da + db) * (da + db) + add) -
        8.0 / sqrt(r * r + da * da + db * db + add);
    return zzzz / 16.0 - xyxy / 64.0;
  }
  if (l1 == 1 && l2 == 2 && m == 1) {
    const double ab = db / rt2;
    const double v = -2.0 / sqrt((r - ab) * (r - ab) + (da - ab) * (da - ab) + add) +
                     2.0 / sqrt((r + ab) * (r + ab) + (da - ab) * (da - ab) + add) +
                     2.0 / sqrt((r - ab) * (r - ab) + (da + ab) * (da + ab) + add) -
                     2.0 / sqrt((r + ab) * (r + ab) + (da + ab) * (da + ab) + add);
    return 0.125 * v;
  }
  if (l1 == 2 && l2 == 1 && m == 1) {
    const double aa = da / rt2;
    const double v = -2.0 / sqrt((r + aa) * (r + aa) + (aa - db) * (aa - db) + add) +
                     2.0 / sqrt((r - aa) * (r - aa) + (aa - db) * (aa - db) + add) +
                     2.0 / sqrt((r + aa) * (r + aa) + (aa + db) * (aa + db) + add) -
                     2.0 / sqrt((r - aa) * (r - aa) + (aa + db) * (aa + db) + add);
    return 0.125 * v;
  }
  if (l1 == 2 && l2 == 2 && m == 1) {
    const double aa = da / rt2;
    const double ab = db / rt2;
    const double v =
        2.0 / sqrt((r + aa - ab) * (r + aa - ab) + (aa - ab) * (aa - ab) + add) -
        2.0 / sqrt((r + aa + ab) * (r + aa + ab) + (aa - ab) * (aa - ab) + add) -
        2.0 / sqrt((r - aa - ab) * (r - aa - ab) + (aa - ab) * (aa - ab) + add) +
        2.0 / sqrt((r - aa + ab) * (r - aa + ab) + (aa - ab) * (aa - ab) + add) -
        2.0 / sqrt((r + aa - ab) * (r + aa - ab) + (aa + ab) * (aa + ab) + add) +
        2.0 / sqrt((r + aa + ab) * (r + aa + ab) + (aa + ab) * (aa + ab) + add) +
        2.0 / sqrt((r - aa - ab) * (r - aa - ab) + (aa + ab) * (aa + ab) + add) -
        2.0 / sqrt((r - aa + ab) * (r - aa + ab) + (aa + ab) * (aa + ab) + add);
    return v / 16.0;
  }
  if (l1 == 2 && l2 == 2 && m == 2) {
    const double v = 4.0 / sqrt(r * r + (da - db) * (da - db) + add) +
                     4.0 / sqrt(r * r + (da + db) * (da + db) + add) -
                     8.0 / sqrt(r * r + da * da + db * db + add);
    return v / 16.0;
  }
  return 0.0;
}

static __device__ __forceinline__ double mozyme_mndod_rijkl_dev(
    int ni, int nj, int ij, int kl, int li, int lj, int lk, int ll, double r,
    const double *po, const double *ddp) {
  const int l1min0 = li > lj ? li - lj : lj - li;
  const int l2min0 = lk > ll ? lk - ll : ll - lk;
  const int l1min = l1min0 > 2 ? 2 : l1min0;
  const int l2min = l2min0 > 2 ? 2 : l2min0;
  const int l1max0 = li + lj;
  const int l2max0 = lk + ll;
  const int l1max = l1max0 > 2 ? 2 : l1max0;
  const int l2max = l2max0 > 2 ? 2 : l2max0;
  const int lij = mozyme_mndod_indx_sp_dev(li + 1, lj + 1);
  const int lkl = mozyme_mndod_indx_sp_dev(lk + 1, ll + 1);
  double sum = 0.0;
  for (int l1 = l1min; l1 <= l1max; ++l1) {
    double pij = 0.0;
    double dij = 0.0;
    if (l1 == 0) {
      if (lij == 1) {
        pij = mozyme_mndod_po_at_dev(po, ni, 1);
      } else if (lij == 3) {
        pij = mozyme_mndod_po_at_dev(po, ni, 7);
      } else if (lij == 6) {
        pij = mozyme_mndod_po_at_dev(po, ni, 8);
      }
    } else {
      dij = mozyme_mndod_ddp_at_dev(ddp, ni, lij);
      pij = mozyme_mndod_po_at_dev(po, ni, lij);
    }
    for (int l2 = l2min; l2 <= l2max; ++l2) {
      double pkl = 0.0;
      double dkl = 0.0;
      if (l2 == 0) {
        if (lkl == 1) {
          pkl = mozyme_mndod_po_at_dev(po, nj, 1);
        } else if (lkl == 3) {
          pkl = mozyme_mndod_po_at_dev(po, nj, 7);
        } else if (lkl == 6) {
          pkl = mozyme_mndod_po_at_dev(po, nj, 8);
        }
      } else {
        dkl = mozyme_mndod_ddp_at_dev(ddp, nj, lkl);
        pkl = mozyme_mndod_po_at_dev(po, nj, lkl);
      }
      const double add = (pij + pkl) * (pij + pkl);
      const int lmin = l1 < l2 ? l1 : l2;
      for (int m = -lmin; m <= lmin; ++m) {
        const double ccc = mozyme_mndod_ch_dev(ij, l1, m) *
                           mozyme_mndod_ch_dev(kl, l2, m);
        if (ccc == 0.0) continue;
        const int mm = m < 0 ? -m : m;
        sum += mozyme_mndod_charg_dev(r, l1, l2, mm, dij, dkl, add) * ccc;
      }
    }
  }
  return sum;
}

static __device__ __forceinline__ bool mozyme_direct_reppd2_rep_dev(
    int ni, int nj, int iab, int jba, int l_feather_flag, double rij,
    double r_bohr, double ev, double a0, double trunc_1, double trunc_2,
    const double *po, const double *ddp, const double *ri, double *rep) {
  if (!po || !ddp || !ri || !rep || iab < 1 || jba < 1 ||
      iab > 9 || jba > 9) {
    return false;
  }
  for (int i = 0; i < 492; ++i) rep[i] = 0.0;
  for (int i = 1; i <= 34; ++i) rep[i] = ri[kMozymeMndodIpos[i] - 1];

  const int lasti = iab == 9 ? 9 : (iab <= 1 ? 1 : 4);
  const int lastk = jba == 9 ? 9 : (jba <= 1 ? 1 : 4);
  for (int i = 1; i <= lasti; ++i) {
    const int li = mozyme_mndod_lorb_dev(i);
    for (int j = 1; j <= i; ++j) {
      const bool coul = (i == j);
      const int lj = mozyme_mndod_lorb_dev(j);
      const int ij = mozyme_mndod_indexd_dev(i, j);
      for (int k = 1; k <= lastk; ++k) {
        const int lk = mozyme_mndod_lorb_dev(k);
        for (int l = 1; l <= k; ++l) {
          const bool coulomb = (coul && k == l);
          const int ll = mozyme_mndod_lorb_dev(l);
          const int kl = mozyme_mndod_indexd_dev(k, l);
          const int numb = mozyme_mndod_ind2_dev(ij, kl);
          if (numb <= 0) continue;
          if (numb <= 34) continue;
          const int nold = mozyme_mndod_isym_dev(numb);
          if (nold >= 35) {
            rep[numb] = rep[nold];
          } else if (nold <= -35) {
            rep[numb] = -rep[-nold];
          } else {
            double value = mozyme_mndod_rijkl_dev(
                               ni, nj, ij, kl, li, lj, lk, ll, r_bohr, po,
                               ddp) *
                           ev;
            if (l_feather_flag != 0) {
              double point = 0.0;
              double constant = 1.0;
              mozyme_resident_to_point_dev(rij, ev, a0, trunc_1, trunc_2,
                                           &point, &constant);
              if (coulomb) {
                value = value * constant + (1.0 - constant) * point;
              } else {
                value *= constant;
              }
            }
            rep[numb] = value;
          }
        }
      }
    }
  }
  return true;
}

static __device__ __forceinline__ bool mozyme_direct_rotmat_spd_dev(
    const double *coord, int ii, int jj, double *rij, double sp[3][3],
    double pp[6][3][3], double sd[5][5], double dp[15][5][3],
    double ddrot[15][5][5], double *p_scratch, double *d_scratch) {
  if (!coord || !rij || !p_scratch || !d_scratch) return false;
  const double x11 = mozyme_resident_coord_at_dev(coord, 0, jj) -
                     mozyme_resident_coord_at_dev(coord, 0, ii);
  const double x22 = mozyme_resident_coord_at_dev(coord, 1, jj) -
                     mozyme_resident_coord_at_dev(coord, 1, ii);
  const double x33 = mozyme_resident_coord_at_dev(coord, 2, jj) -
                     mozyme_resident_coord_at_dev(coord, 2, ii);
  const double b = x11 * x11 + x22 * x22;
  *rij = sqrt(b + x33 * x33);
  if (*rij <= 0.0) return false;

  const double small = 1.0e-7;
  const double pt5sq3 = 0.8660254037841;
  const double sqb = sqrt(b);
  double sb = sqb / *rij;
  double ca = 0.0;
  double sa = 0.0;
  double cb = 0.0;
  if (sb > small) {
    ca = x11 / sqb;
    sa = x22 / sqb;
    cb = x33 / *rij;
  } else {
    sa = 0.0;
    sb = 0.0;
    if (x33 < 0.0) {
      ca = -1.0;
      cb = -1.0;
    } else if (x33 > 0.0) {
      ca = 1.0;
      cb = 1.0;
    }
  }

  double (*p)[3] = reinterpret_cast<double (*)[3]>(p_scratch);
  p[0][0] = ca * sb;
  p[1][0] = ca * cb;
  p[2][0] = -sa;
  p[0][1] = sa * sb;
  p[1][1] = sa * cb;
  p[2][1] = ca;
  p[0][2] = cb;
  p[1][2] = -sb;
  p[2][2] = 0.0;

  const double c2a = 2.0 * ca * ca - 1.0;
  const double c2b = 2.0 * cb * cb - 1.0;
  const double s2a = 2.0 * sa * ca;
  const double s2b = 2.0 * sb * cb;
  double (*d)[5] = reinterpret_cast<double (*)[5]>(d_scratch);
  d[0][0] = pt5sq3 * c2a * sb * sb;
  d[1][0] = 0.5 * c2a * s2b;
  d[2][0] = -s2a * sb;
  d[3][0] = c2a * (cb * cb + 0.5 * sb * sb);
  d[4][0] = -s2a * cb;
  d[0][1] = pt5sq3 * ca * s2b;
  d[1][1] = ca * c2b;
  d[2][1] = -sa * cb;
  d[3][1] = -0.5 * ca * s2b;
  d[4][1] = sa * sb;
  d[0][2] = cb * cb - 0.5 * sb * sb;
  d[1][2] = -pt5sq3 * s2b;
  d[2][2] = 0.0;
  d[3][2] = pt5sq3 * sb * sb;
  d[4][2] = 0.0;
  d[0][3] = pt5sq3 * sa * s2b;
  d[1][3] = sa * c2b;
  d[2][3] = ca * cb;
  d[3][3] = -0.5 * sa * s2b;
  d[4][3] = -ca * sb;
  d[0][4] = pt5sq3 * s2a * sb * sb;
  d[1][4] = 0.5 * s2a * s2b;
  d[2][4] = c2a * sb;
  d[3][4] = s2a * (cb * cb + 0.5 * sb * sb);
  d[4][4] = c2a * cb;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) sp[i][j] = p[i][j];
  }
  for (int a = 0; a < 6; ++a) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) pp[a][i][j] = 0.0;
    }
  }
  for (int k = 0; k < 3; ++k) {
    pp[0][k][k] = p[k][0] * p[k][0];
    pp[1][k][k] = p[k][1] * p[k][1];
    pp[2][k][k] = p[k][2] * p[k][2];
    pp[3][k][k] = p[k][0] * p[k][1];
    pp[4][k][k] = p[k][0] * p[k][2];
    pp[5][k][k] = p[k][1] * p[k][2];
    for (int j = 0; j < k; ++j) {
      pp[0][k][j] = 2.0 * p[k][0] * p[j][0];
      pp[1][k][j] = 2.0 * p[k][1] * p[j][1];
      pp[2][k][j] = 2.0 * p[k][2] * p[j][2];
      pp[3][k][j] = p[k][0] * p[j][1] + p[k][1] * p[j][0];
      pp[4][k][j] = p[k][0] * p[j][2] + p[k][2] * p[j][0];
      pp[5][k][j] = p[k][1] * p[j][2] + p[k][2] * p[j][1];
    }
  }
  for (int k = 0; k < 5; ++k) {
    for (int i = 0; i < 5; ++i) sd[k][i] = d[k][i];
    dp[0][k][0] = d[k][0] * p[0][0];
    dp[0][k][1] = d[k][0] * p[1][0];
    dp[0][k][2] = d[k][0] * p[2][0];
    dp[1][k][0] = d[k][0] * p[0][1];
    dp[1][k][1] = d[k][0] * p[1][1];
    dp[1][k][2] = d[k][0] * p[2][1];
    dp[2][k][0] = d[k][0] * p[0][2];
    dp[2][k][1] = d[k][0] * p[1][2];
    dp[2][k][2] = d[k][0] * p[2][2];
    dp[3][k][0] = d[k][1] * p[0][0];
    dp[3][k][1] = d[k][1] * p[1][0];
    dp[3][k][2] = d[k][1] * p[2][0];
    dp[4][k][0] = d[k][1] * p[0][1];
    dp[4][k][1] = d[k][1] * p[1][1];
    dp[4][k][2] = d[k][1] * p[2][1];
    dp[5][k][0] = d[k][1] * p[0][2];
    dp[5][k][1] = d[k][1] * p[1][2];
    dp[5][k][2] = d[k][1] * p[2][2];
    dp[6][k][0] = d[k][2] * p[0][0];
    dp[6][k][1] = d[k][2] * p[1][0];
    dp[6][k][2] = d[k][2] * p[2][0];
    dp[7][k][0] = d[k][2] * p[0][1];
    dp[7][k][1] = d[k][2] * p[1][1];
    dp[7][k][2] = d[k][2] * p[2][1];
    dp[8][k][0] = d[k][2] * p[0][2];
    dp[8][k][1] = d[k][2] * p[1][2];
    dp[8][k][2] = d[k][2] * p[2][2];
    dp[9][k][0] = d[k][3] * p[0][0];
    dp[9][k][1] = d[k][3] * p[1][0];
    dp[9][k][2] = d[k][3] * p[2][0];
    dp[10][k][0] = d[k][3] * p[0][1];
    dp[10][k][1] = d[k][3] * p[1][1];
    dp[10][k][2] = d[k][3] * p[2][1];
    dp[11][k][0] = d[k][3] * p[0][2];
    dp[11][k][1] = d[k][3] * p[1][2];
    dp[11][k][2] = d[k][3] * p[2][2];
    dp[12][k][0] = d[k][4] * p[0][0];
    dp[12][k][1] = d[k][4] * p[1][0];
    dp[12][k][2] = d[k][4] * p[2][0];
    dp[13][k][0] = d[k][4] * p[0][1];
    dp[13][k][1] = d[k][4] * p[1][1];
    dp[13][k][2] = d[k][4] * p[2][1];
    dp[14][k][0] = d[k][4] * p[0][2];
    dp[14][k][1] = d[k][4] * p[1][2];
    dp[14][k][2] = d[k][4] * p[2][2];
  }
  for (int a = 0; a < 15; ++a) {
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) ddrot[a][i][j] = 0.0;
    }
  }
  for (int k = 0; k < 5; ++k) {
    ddrot[0][k][k] = d[k][0] * d[k][0];
    ddrot[1][k][k] = d[k][1] * d[k][1];
    ddrot[2][k][k] = d[k][2] * d[k][2];
    ddrot[3][k][k] = d[k][3] * d[k][3];
    ddrot[4][k][k] = d[k][4] * d[k][4];
    ddrot[5][k][k] = d[k][0] * d[k][1];
    ddrot[6][k][k] = d[k][0] * d[k][2];
    ddrot[7][k][k] = d[k][1] * d[k][2];
    ddrot[8][k][k] = d[k][0] * d[k][3];
    ddrot[9][k][k] = d[k][1] * d[k][3];
    ddrot[10][k][k] = d[k][2] * d[k][3];
    ddrot[11][k][k] = d[k][0] * d[k][4];
    ddrot[12][k][k] = d[k][1] * d[k][4];
    ddrot[13][k][k] = d[k][2] * d[k][4];
    ddrot[14][k][k] = d[k][3] * d[k][4];
    for (int j = 0; j < k; ++j) {
      ddrot[0][k][j] = 2.0 * d[k][0] * d[j][0];
      ddrot[1][k][j] = 2.0 * d[k][1] * d[j][1];
      ddrot[2][k][j] = 2.0 * d[k][2] * d[j][2];
      ddrot[3][k][j] = 2.0 * d[k][3] * d[j][3];
      ddrot[4][k][j] = 2.0 * d[k][4] * d[j][4];
      ddrot[5][k][j] = d[k][0] * d[j][1] + d[k][1] * d[j][0];
      ddrot[6][k][j] = d[k][0] * d[j][2] + d[k][2] * d[j][0];
      ddrot[7][k][j] = d[k][1] * d[j][2] + d[k][2] * d[j][1];
      ddrot[8][k][j] = d[k][0] * d[j][3] + d[k][3] * d[j][0];
      ddrot[9][k][j] = d[k][1] * d[j][3] + d[k][3] * d[j][1];
      ddrot[10][k][j] = d[k][2] * d[j][3] + d[k][3] * d[j][2];
      ddrot[11][k][j] = d[k][0] * d[j][4] + d[k][4] * d[j][0];
      ddrot[12][k][j] = d[k][1] * d[j][4] + d[k][4] * d[j][1];
      ddrot[13][k][j] = d[k][2] * d[j][4] + d[k][4] * d[j][2];
      ddrot[14][k][j] = d[k][3] * d[j][4] + d[k][4] * d[j][3];
    }
  }
  return true;
}

static __device__ __forceinline__ bool mozyme_mndod_ww_index_dev(
    int ij, int kl, int limkl, int total, int *out) {
  if (!out || ij < 1 || kl < 1 || limkl <= 0 || total <= 0) return false;
  const long long pos =
      (static_cast<long long>(ij) - 1ll) * static_cast<long long>(limkl) +
      (static_cast<long long>(kl) - 1ll);
  if (pos < 0 || pos >= static_cast<long long>(total)) return false;
  *out = static_cast<int>(pos);
  return true;
}

static __device__ __forceinline__ bool mozyme_direct_pm7_d_w_correction_dev(
    int iab, int jba, int ni, int nj, const int *iod, double *ww) {
  if (!iod || !ww || ni < 1 || ni > 107 || nj < 1 || nj > 107) return false;
  const int limkl = mozyme_resident_tri_dev(jba);
  const int total = mozyme_resident_tri_dev(iab) * limkl;
  int pos = 0;
  if (iab == 9 && iod[ni - 1] > 0) {
    if (jba > 1) {
      double sum = 0.0;
      for (int i = 5; i <= 9; ++i) {
        const int ij = mozyme_resident_tri_dev(i);
        int p3 = 0, p6 = 0, p10 = 0;
        if (!mozyme_mndod_ww_index_dev(ij, 3, limkl, total, &p3) ||
            !mozyme_mndod_ww_index_dev(ij, 6, limkl, total, &p6) ||
            !mozyme_mndod_ww_index_dev(ij, 10, limkl, total, &p10)) return false;
        sum += ww[p3] + ww[p6] + ww[p10];
      }
      sum = ww[0] - sum / 15.0;
      for (int i = 5; i <= 9; ++i) {
        const int ij = mozyme_resident_tri_dev(i);
        for (int l = 2; l <= 4; ++l) {
          if (!mozyme_mndod_ww_index_dev(ij, mozyme_resident_tri_dev(l),
                                         limkl, total, &pos)) return false;
          ww[pos] += sum;
        }
      }
    }
    double sum = 0.0;
    for (int i = 5; i <= 9; ++i) {
      if (!mozyme_mndod_ww_index_dev(mozyme_resident_tri_dev(i), 1,
                                     limkl, total, &pos)) return false;
      sum += ww[pos];
    }
    sum = ww[0] - sum / 5.0;
    for (int i = 5; i <= 9; ++i) {
      if (!mozyme_mndod_ww_index_dev(mozyme_resident_tri_dev(i), 1,
                                     limkl, total, &pos)) return false;
      ww[pos] += sum;
    }
  }
  if (jba == 9 && iod[nj - 1] > 0) {
    if (iab == 9 && iod[ni - 1] > 0) {
      double sum = 0.0;
      for (int i = 5; i <= 9; ++i) {
        const int ij = mozyme_resident_tri_dev(i);
        int p15 = 0, p21 = 0, p28 = 0, p36 = 0, p45 = 0;
        if (!mozyme_mndod_ww_index_dev(ij, 15, limkl, total, &p15) ||
            !mozyme_mndod_ww_index_dev(ij, 21, limkl, total, &p21) ||
            !mozyme_mndod_ww_index_dev(ij, 28, limkl, total, &p28) ||
            !mozyme_mndod_ww_index_dev(ij, 36, limkl, total, &p36) ||
            !mozyme_mndod_ww_index_dev(ij, 45, limkl, total, &p45)) return false;
        sum += ww[p15] + ww[p21] + ww[p28] + ww[p36] + ww[p45];
      }
      sum = ww[0] - sum / 25.0;
      for (int i = 5; i <= 9; ++i) {
        const int ij = mozyme_resident_tri_dev(i);
        for (int k = 5; k <= 9; ++k) {
          if (!mozyme_mndod_ww_index_dev(ij, mozyme_resident_tri_dev(k),
                                         limkl, total, &pos)) return false;
          ww[pos] += sum;
        }
      }
    }
    if (iab > 1) {
      double sum = 0.0;
      for (int i = 2; i <= 4; ++i) {
        const int ij = mozyme_resident_tri_dev(i);
        int p15 = 0, p21 = 0, p28 = 0, p36 = 0, p45 = 0;
        if (!mozyme_mndod_ww_index_dev(ij, 15, limkl, total, &p15) ||
            !mozyme_mndod_ww_index_dev(ij, 21, limkl, total, &p21) ||
            !mozyme_mndod_ww_index_dev(ij, 28, limkl, total, &p28) ||
            !mozyme_mndod_ww_index_dev(ij, 36, limkl, total, &p36) ||
            !mozyme_mndod_ww_index_dev(ij, 45, limkl, total, &p45)) return false;
        sum += ww[p15] + ww[p21] + ww[p28] + ww[p36] + ww[p45];
      }
      sum = ww[0] - sum / 15.0;
      for (int i = 2; i <= 4; ++i) {
        const int ij = mozyme_resident_tri_dev(i);
        for (int k = 5; k <= 9; ++k) {
          if (!mozyme_mndod_ww_index_dev(ij, mozyme_resident_tri_dev(k),
                                         limkl, total, &pos)) return false;
          ww[pos] += sum;
        }
      }
    }
    int p15 = 0, p21 = 0, p28 = 0, p36 = 0, p45 = 0;
    if (!mozyme_mndod_ww_index_dev(1, 15, limkl, total, &p15) ||
        !mozyme_mndod_ww_index_dev(1, 21, limkl, total, &p21) ||
        !mozyme_mndod_ww_index_dev(1, 28, limkl, total, &p28) ||
        !mozyme_mndod_ww_index_dev(1, 36, limkl, total, &p36) ||
        !mozyme_mndod_ww_index_dev(1, 45, limkl, total, &p45)) return false;
    const double sum = ww[0] -
                       (ww[p15] + ww[p21] + ww[p28] + ww[p36] + ww[p45]) / 5.0;
    for (int k = 5; k <= 9; ++k) {
      if (!mozyme_mndod_ww_index_dev(1, mozyme_resident_tri_dev(k),
                                     limkl, total, &pos)) return false;
      ww[pos] += sum;
    }
  }
  return true;
}

static __device__ __forceinline__ bool mozyme_direct_spd_w_dev(
    int iab, int jba, int ni, int nj, int ii, int jj, int l_feather_flag,
    int method_pm7_flag, double ev, double a0, double trunc_1, double trunc_2,
    const double *coord, const double *am, const double *ad, const double *aq,
    const double *dd, const double *qq, const double *po, const double *ddp,
    const int *iod, double *scratch, double *w_out, int w_count) {
  if (!mozyme_resident_basis_supported_dev(iab) ||
      !mozyme_resident_basis_supported_dev(jba) || (iab != 9 && jba != 9) ||
      !po || !ddp || !scratch || !w_out || w_count < 1) {
    return false;
  }
  for (int i = 0; i < w_count; ++i) w_out[i] = 0.0;

  const double dx = mozyme_resident_coord_at_dev(coord, 0, ii) -
                    mozyme_resident_coord_at_dev(coord, 0, jj);
  const double dy = mozyme_resident_coord_at_dev(coord, 1, ii) -
                    mozyme_resident_coord_at_dev(coord, 1, jj);
  const double dz = mozyme_resident_coord_at_dev(coord, 2, ii) -
                    mozyme_resident_coord_at_dev(coord, 2, jj);
  const double r2 = dx * dx + dy * dy + dz * dz;
  if (r2 < 0.00002) return true;

  double (*sp)[3] =
      reinterpret_cast<double (*)[3]>(scratch + kMozymeDirectSpdScratchSp);
  double (*pp)[3][3] =
      reinterpret_cast<double (*)[3][3]>(scratch + kMozymeDirectSpdScratchPp);
  double (*sd)[5] =
      reinterpret_cast<double (*)[5]>(scratch + kMozymeDirectSpdScratchSd);
  double (*dp)[5][3] =
      reinterpret_cast<double (*)[5][3]>(scratch + kMozymeDirectSpdScratchDp);
  double (*ddrot)[5][5] =
      reinterpret_cast<double (*)[5][5]>(scratch + kMozymeDirectSpdScratchDdrot);
  double *rot_p = scratch + kMozymeDirectSpdScratchRotP;
  double *rot_d = scratch + kMozymeDirectSpdScratchRotD;
  double rij = 0.0;
  if (!mozyme_direct_rotmat_spd_dev(coord, ii, jj, &rij, sp, pp, sd, dp,
                                    ddrot, rot_p, rot_d)) {
    return false;
  }

  double *ri = scratch + kMozymeDirectSpdScratchRi;
  double *reppd_arg = scratch + kMozymeDirectSpdScratchReppdArg;
  double *reppd_sqr = scratch + kMozymeDirectSpdScratchReppdSqr;
  if (!mozyme_direct_reppd_sp_dev(ni, nj, iab, jba, rij, l_feather_flag, ev,
                                  a0, trunc_1, trunc_2, am, ad, aq, dd, qq,
                                  ri, reppd_arg, reppd_sqr)) {
    return false;
  }
  double *rep = scratch + kMozymeDirectSpdScratchRep;
  const double r_bohr = rij / a0;
  if (!mozyme_direct_reppd2_rep_dev(ni, nj, iab, jba, l_feather_flag, rij,
                                    r_bohr, ev, a0, trunc_1, trunc_2, po,
                                    ddp, ri, rep)) {
    return false;
  }

  const int limij = mozyme_resident_tri_dev(iab);
  const int limkl = mozyme_resident_tri_dev(jba);
  const int total = limij * limkl;
  if (total > w_count || total > 2025) return false;

  double (*v)[46] =
      reinterpret_cast<double (*)[46]>(scratch + kMozymeDirectSpdScratchV);
  for (int i = 0; i < 46; ++i) {
    for (int j = 0; j < 46; ++j) {
      v[i][j] = 0.0;
    }
  }

  for (int i1 = 1; i1 <= iab; ++i1) {
    for (int j1 = 1; j1 <= i1; ++j1) {
      const int ij = mozyme_mndod_indexd_sp_dev(i1, j1);
      for (int k1 = 1; k1 <= jba; ++k1) {
        for (int l1 = 1; l1 <= k1; ++l1) {
          const int kl_indexd = mozyme_mndod_indexd_dev(k1, l1);
          const int nd = mozyme_mndod_ind2_dev(ij, kl_indexd);
          if (nd <= 0) continue;
          const double wrepp = rep[nd];
          if (wrepp == 0.0) continue;
          const int mm = mozyme_mndod_met_pair_dev(k1, l1);
          if (mm == 1) {
            v[ij][1] = wrepp;
          } else if (mm == 2) {
            const int k = k1 - 2;
            v[ij][2] += sp[k][0] * wrepp;
            v[ij][4] += sp[k][1] * wrepp;
            v[ij][7] += sp[k][2] * wrepp;
          } else if (mm == 3) {
            const int k = k1 - 2;
            const int l = l1 - 2;
            v[ij][3] += pp[0][k][l] * wrepp;
            v[ij][6] += pp[1][k][l] * wrepp;
            v[ij][10] += pp[2][k][l] * wrepp;
            v[ij][5] += pp[3][k][l] * wrepp;
            v[ij][8] += pp[4][k][l] * wrepp;
            v[ij][9] += pp[5][k][l] * wrepp;
          } else if (mm == 4) {
            const int k = k1 - 5;
            const int cols[5] = {11, 16, 22, 29, 37};
            for (int c = 0; c < 5; ++c) v[ij][cols[c]] += sd[k][c] * wrepp;
          } else if (mm == 5) {
            const int k = k1 - 5;
            const int l = l1 - 2;
            const int cols[15] = {12, 13, 14, 17, 18, 19, 23, 24,
                                  25, 30, 31, 32, 38, 39, 40};
            for (int c = 0; c < 15; ++c) v[ij][cols[c]] += dp[c][k][l] * wrepp;
          } else if (mm == 6) {
            const int k = k1 - 5;
            const int l = l1 - 5;
            const int cols[15] = {15, 21, 28, 36, 45, 20, 26, 27,
                                  33, 34, 35, 41, 42, 43, 44};
            for (int c = 0; c < 15; ++c) v[ij][cols[c]] += ddrot[c][k][l] * wrepp;
          } else {
            return false;
          }
        }
      }
    }
  }

  double *ww = scratch + kMozymeDirectSpdScratchWw;
  for (int i = 0; i < kMozymeDirectMaxW; ++i) ww[i] = 0.0;
  for (int i1 = 1; i1 <= iab; ++i1) {
    for (int j1 = 1; j1 <= i1; ++j1) {
      const int ij = mozyme_mndod_indexd_sp_dev(i1, j1);
      const int mm = mozyme_mndod_met_pair_dev(i1, j1);
      for (int k = 1; k <= jba; ++k) {
        for (int l = 1; l <= k; ++l) {
          const int kl = mozyme_mndod_indx_sp_dev(k, l);
          const double wrepp = v[ij][kl];
          if (wrepp == 0.0) continue;
          if (mm == 1) {
            ww[mozyme_mndod_indw_sp_dev(1, 1, limkl, kl)] = wrepp;
          } else if (mm == 2) {
            const int pidx = i1 - 2;
            for (int i = 1; i <= 3; ++i) {
              ww[mozyme_mndod_indw_sp_dev(i + 1, 1, limkl, kl)] +=
                  sp[pidx][i - 1] * wrepp;
            }
          } else if (mm == 3) {
            const int p_i = i1 - 2;
            const int p_j = j1 - 2;
            for (int i = 1; i <= 3; ++i) {
              ww[mozyme_mndod_indw_sp_dev(i + 1, i + 1, limkl, kl)] +=
                  pp[i - 1][p_i][p_j] * wrepp;
              for (int j = 1; j < i; ++j) {
                ww[mozyme_mndod_indw_sp_dev(i + 1, j + 1, limkl, kl)] +=
                    pp[i + j][p_i][p_j] * wrepp;
              }
            }
          } else if (mm == 4) {
            const int d_i = i1 - 5;
            for (int i = 1; i <= 5; ++i) {
              ww[mozyme_mndod_indw_sp_dev(i + 4, 1, limkl, kl)] +=
                  sd[d_i][i - 1] * wrepp;
            }
          } else if (mm == 5) {
            const int d_i = i1 - 5;
            const int p_j = j1 - 2;
            for (int i = 1; i <= 5; ++i) {
              for (int j = 1; j <= 3; ++j) {
                const int ij1 = 3 * (i - 1) + j;
                ww[mozyme_mndod_indw_sp_dev(i + 4, j + 1, limkl, kl)] +=
                    dp[ij1 - 1][d_i][p_j] * wrepp;
              }
            }
          } else if (mm == 6) {
            const int d_i = i1 - 5;
            const int d_j = j1 - 5;
            for (int i = 1; i <= 5; ++i) {
              ww[mozyme_mndod_indw_sp_dev(i + 4, i + 4, limkl, kl)] +=
                  ddrot[i - 1][d_i][d_j] * wrepp;
              for (int j = 1; j < i; ++j) {
                const int ij1 = mozyme_mndod_inddd_dev(i, j);
                ww[mozyme_mndod_indw_sp_dev(i + 4, j + 4, limkl, kl)] +=
                    ddrot[ij1 - 1][d_i][d_j] * wrepp;
              }
            }
          } else {
            return false;
          }
        }
      }
    }
  }
  if (method_pm7_flag != 0 &&
      !mozyme_direct_pm7_d_w_correction_dev(iab, jba, ni, nj, iod, ww)) {
    return false;
  }
  for (int i = 0; i < total; ++i) w_out[i] = ww[i];
  return true;
}

static __device__ __forceinline__ bool mozyme_direct_sp_w_dev(
    int iab, int jba, int ni, int nj, int ii, int jj, int l_feather_flag,
    double ev, double a0, double trunc_1, double trunc_2, const double *coord,
    const double *am, const double *ad, const double *aq, const double *dd,
    const double *qq, double *scratch, double *w_out, int w_count) {
  if (!mozyme_resident_direct_sp_basis_dev(iab) ||
      !mozyme_resident_direct_sp_basis_dev(jba) || !scratch || !w_out ||
      w_count < 1) {
    return false;
  }
  for (int i = 0; i < w_count; ++i) w_out[i] = 0.0;

  const double dx = mozyme_resident_coord_at_dev(coord, 0, ii) -
                    mozyme_resident_coord_at_dev(coord, 0, jj);
  const double dy = mozyme_resident_coord_at_dev(coord, 1, ii) -
                    mozyme_resident_coord_at_dev(coord, 1, jj);
  const double dz = mozyme_resident_coord_at_dev(coord, 2, ii) -
                    mozyme_resident_coord_at_dev(coord, 2, jj);
  const double r2 = dx * dx + dy * dy + dz * dz;
  if (r2 < 0.00002) return true;

  double (*sp)[3] =
      reinterpret_cast<double (*)[3]>(scratch + kMozymeDirectSpdScratchSp);
  double (*pp)[3][3] =
      reinterpret_cast<double (*)[3][3]>(scratch + kMozymeDirectSpdScratchPp);
  double rij = 0.0;
  double *rot_p = scratch + kMozymeDirectSpdScratchRotP;
  if (!mozyme_direct_rotmat_sp_dev(coord, ii, jj, &rij, sp, pp, rot_p)) {
    return false;
  }

  double *ri = scratch + kMozymeDirectSpdScratchRi;
  double *reppd_arg = scratch + kMozymeDirectSpdScratchReppdArg;
  double *reppd_sqr = scratch + kMozymeDirectSpdScratchReppdSqr;
  if (!mozyme_direct_reppd_sp_dev(ni, nj, iab, jba, rij, l_feather_flag, ev,
                                  a0, trunc_1, trunc_2, am, ad, aq, dd, qq,
                                  ri, reppd_arg, reppd_sqr)) {
    return false;
  }

  const int limij = mozyme_resident_tri_dev(iab);
  const int limkl = mozyme_resident_tri_dev(jba);
  const int total = limij * limkl;
  if (total > w_count || total > 100) return false;

  double (*v)[11] =
      reinterpret_cast<double (*)[11]>(scratch + kMozymeDirectSpdScratchV);
  for (int i = 0; i < 26; ++i) {
    for (int j = 0; j < 11; ++j) {
      v[i][j] = 0.0;
    }
  }
  for (int i1 = 1; i1 <= iab; ++i1) {
    for (int j1 = 1; j1 <= i1; ++j1) {
      const int ij = mozyme_mndod_indexd_sp_dev(i1, j1);
      for (int k1 = 1; k1 <= jba; ++k1) {
        for (int l1 = 1; l1 <= k1; ++l1) {
          const int kl_indexd = mozyme_mndod_indexd_sp_dev(k1, l1);
          const int nd = mozyme_mndod_ind2_sp_dev(ij, kl_indexd);
          if (nd == 0) continue;
          const double wrepp = mozyme_mndod_rep_sp_dev(nd, ri);
          const int ll = mozyme_mndod_indx_sp_dev(k1, l1);
          const int mm = mozyme_mndod_met_sp_dev(ll);
          if (mm == 1) {
            v[ij][1] = wrepp;
          } else if (mm == 2) {
            const int k = k1 - 2;
            v[ij][2] += sp[k][0] * wrepp;
            v[ij][4] += sp[k][1] * wrepp;
            v[ij][7] += sp[k][2] * wrepp;
          } else if (mm == 3) {
            const int k = k1 - 2;
            const int l = l1 - 2;
            v[ij][3] += pp[0][k][l] * wrepp;
            v[ij][6] += pp[1][k][l] * wrepp;
            v[ij][10] += pp[2][k][l] * wrepp;
            v[ij][5] += pp[3][k][l] * wrepp;
            v[ij][8] += pp[4][k][l] * wrepp;
            v[ij][9] += pp[5][k][l] * wrepp;
          } else {
            return false;
          }
        }
      }
    }
  }

  double *ww = scratch + kMozymeDirectSpdScratchWw;
  for (int i = 0; i < 100; ++i) ww[i] = 0.0;
  for (int i1 = 1; i1 <= iab; ++i1) {
    for (int j1 = 1; j1 <= i1; ++j1) {
      const int ij = mozyme_mndod_indexd_sp_dev(i1, j1);
      const int jj_std = mozyme_mndod_indx_sp_dev(i1, j1);
      const int mm = mozyme_mndod_met_sp_dev(jj_std);
      for (int k = 1; k <= jba; ++k) {
        for (int l = 1; l <= k; ++l) {
          const int kl = mozyme_mndod_indx_sp_dev(k, l);
          const double wrepp = v[ij][kl];
          if (wrepp == 0.0) continue;
          if (mm == 1) {
            ww[mozyme_mndod_indw_sp_dev(1, 1, limkl, kl)] = wrepp;
          } else if (mm == 2) {
            const int pidx = i1 - 2;
            for (int i = 1; i <= 3; ++i) {
              ww[mozyme_mndod_indw_sp_dev(i + 1, 1, limkl, kl)] +=
                  sp[pidx][i - 1] * wrepp;
            }
          } else if (mm == 3) {
            const int p_i = i1 - 2;
            const int p_j = j1 - 2;
            for (int i = 1; i <= 3; ++i) {
              ww[mozyme_mndod_indw_sp_dev(i + 1, i + 1, limkl, kl)] +=
                  pp[i - 1][p_i][p_j] * wrepp;
              for (int j = 1; j < i; ++j) {
                ww[mozyme_mndod_indw_sp_dev(i + 1, j + 1, limkl, kl)] +=
                    pp[i + j][p_i][p_j] * wrepp;
              }
            }
          } else {
            return false;
          }
        }
      }
    }
  }
  for (int i = 0; i < total; ++i) w_out[i] = ww[i];
  return true;
}

__global__ void mozyme_resident_fock_pack_plan_kernel(
    int numat, int mpack, int mode, int ione, int direct_flag, int semidr_flag,
    int l_feather_flag, double ev, double a0, double trunc_1, double trunc_2,
    int w_count, const int *iorbs, const int *nat, const int *kopt,
    const int *nijbo, const int *jindex, const double *coord,
    const double *wj, const double *wk, const double *am, const double *ad,
    const double *aq, const double *dd, const double *qq, const double *po,
    const double *ddp, const double *tore, const int *iod,
    int method_pm7_flag, double *direct_scratch, const int *counts,
    int *one_f, int *one_w, int *one_iab, int *one_ilim,
    double *one_w_values, int *pair_iab, int *pair_jba, int *pair_i,
    int *pair_j, int *pair_cross, int *pair_diag, int *pair_w,
    double *pair_wj, double *pair_wk, int *pair4_heavy, int *pair4_light,
    int *pair4_cross, double *pair4_wj, double *pair4_wk, int *point_iab,
    int *point_jba, int *point_i_atom, int *point_j_atom, int *point_i,
    int *point_j, int *point_addr, double *point_w, int *status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (!status) return;
  for (int i = 0; i < 12; ++i) status[i] = 0;
  if (numat <= 0 || mpack <= 0 || w_count <= 0 || !iorbs || !nat || !nijbo ||
      !jindex || !coord || !wj || (direct_flag == 0 && !wk) || !am || !ad ||
      !aq || !dd || !qq || !po || !ddp || !tore || !iod || !counts ||
      (direct_flag != 0 && !direct_scratch) ||
      (mode != 0 && !kopt)) {
    status[0] = 1;
    return;
  }
  (void)tore;

  const int one_count = counts[0];
  const int pair_count = counts[1];
  const int pair4_count = counts[2];
  const int point_count = counts[3];
  const int one_w_count = counts[4];
  const int pair_w_count = counts[5];
  int one_pos = 0;
  int pair_pos = 0;
  int pair4_pos = 0;
  int point_pos = 0;
  int one_w_pos = 0;
  int pair_w_pos = 0;
  int point_dipole_count = 0;
  int point_monopole_count = 0;
  int kr = 0;
  int one_kr = 0;
  int ired = 1;

  for (int ii = 1; ii <= numat; ++ii) {
    bool calci;
    if (mode == 0) {
      calci = true;
    } else {
      calci = (kopt[ired - 1] == ii);
      if (calci && ired < numat) ++ired;
    }
    const int iab = iorbs[ii - 1];
    if (iab == 0) continue;

    int jred = 1;
    const int iim1 = ii - ione;
    for (int jj = 1; jj <= iim1; ++jj) {
      bool calcj;
      if (mode == 0) {
        calcj = true;
      } else {
        calcj = (kopt[jred - 1] == jj);
        if (calcj && jred < numat) ++jred;
      }
      const int jba = iorbs[jj - 1];
      const int addr = mozyme_resident_nijbo_at(numat, nijbo, ii, jj);
      if (addr >= 0) {
        if ((calci || calcj) &&
            mozyme_resident_pair_supported_for_direct_dev(iab, jba,
                                                          direct_flag)) {
          const int ioff = mozyme_resident_nijbo_at(numat, nijbo, ii, ii) + 1;
          const int joff = mozyme_resident_nijbo_at(numat, nijbo, jj, jj) + 1;
          const int coff = addr + 1;
          double *direct_w = nullptr;
          double *direct_spd_scratch = nullptr;
          const int direct_total = mozyme_resident_pair_integral_count_dev(iab, jba);
          if (direct_flag != 0) {
            direct_w = direct_scratch;
            direct_spd_scratch = direct_scratch + kMozymeDirectMaxW;
            const bool direct_ok =
                (iab == 9 || jba == 9)
                    ? mozyme_direct_spd_w_dev(iab, jba, nat[ii - 1],
                                              nat[jj - 1], ii, jj,
                                              l_feather_flag, method_pm7_flag,
                                              ev, a0, trunc_1, trunc_2, coord,
                                              am, ad, aq, dd, qq, po, ddp, iod,
                                              direct_spd_scratch, direct_w,
                                              direct_total)
                    : mozyme_direct_sp_w_dev(iab, jba, nat[ii - 1],
                                             nat[jj - 1], ii, jj,
                                             l_feather_flag, ev, a0, trunc_1,
                                             trunc_2, coord, am, ad, aq, dd,
                                             qq, direct_spd_scratch, direct_w,
                                             direct_total);
            if (!direct_ok) {
              status[0] = 9;
              return;
            }
          }
          if ((iab == 4 && jba == 1) || (iab == 1 && jba == 4)) {
            if (pair4_pos >= pair4_count || !pair4_heavy || !pair4_light ||
                !pair4_cross || !pair4_wj || !pair4_wk ||
                (direct_flag == 0 &&
                 !mozyme_resident_valid_source_range_dev(kr, 10, w_count))) {
              status[0] = 2;
              return;
            }
            if (iab == 4 && jba == 1) {
              if (!mozyme_resident_valid_plan_range_dev(ioff, 10, mpack) ||
                  !mozyme_resident_valid_plan_range_dev(joff, 1, mpack) ||
                  !mozyme_resident_valid_plan_range_dev(coff, 4, mpack)) {
                status[0] = 3;
                return;
              }
              pair4_heavy[pair4_pos] = ioff;
              pair4_light[pair4_pos] = joff;
            } else {
              if (!mozyme_resident_valid_plan_range_dev(joff, 10, mpack) ||
                  !mozyme_resident_valid_plan_range_dev(ioff, 1, mpack) ||
                  !mozyme_resident_valid_plan_range_dev(coff, 4, mpack)) {
                status[0] = 3;
                return;
              }
              pair4_heavy[pair4_pos] = joff;
              pair4_light[pair4_pos] = ioff;
            }
            pair4_cross[pair4_pos] = coff;
            for (int m = 0; m < 10; ++m) {
              pair4_wj[pair4_pos * 10 + m] =
                  direct_flag != 0 ? direct_w[m] : wj[kr + m];
            }
            for (int m = 0; m < 16; ++m) {
              const int idx = jindex[m] - 1;
              if (idx < 0 ||
                  (direct_flag != 0 && idx >= direct_total) ||
                  (direct_flag == 0 &&
                   !mozyme_resident_valid_source_range_dev(kr + idx, 1, w_count))) {
                status[0] = 2;
                return;
              }
              pair4_wk[pair4_pos * 16 + m] =
                  direct_flag != 0 ? direct_w[idx] : wk[kr + idx];
            }
            ++pair4_pos;
          } else {
            const int ni = mozyme_resident_tri_dev(iab);
            const int nj = mozyme_resident_tri_dev(jba);
            const int total = ni * nj;
            if (pair_pos >= pair_count || pair_w_pos + total > pair_w_count ||
                !pair_iab || !pair_jba || !pair_i || !pair_j || !pair_cross ||
                !pair_diag || !pair_w || !pair_wj || !pair_wk ||
                (direct_flag == 0 &&
                 !mozyme_resident_valid_source_range_dev(kr, total, w_count)) ||
                (direct_flag != 0 && total > direct_total) ||
                !mozyme_resident_valid_plan_range_dev(ioff, ni, mpack) ||
                !mozyme_resident_valid_plan_range_dev(joff, nj, mpack) ||
                (ii != jj &&
                 !mozyme_resident_valid_plan_range_dev(coff, iab * jba, mpack)) ||
                (ii == jj && coff < 1)) {
              status[0] = 4;
              return;
            }
            pair_iab[pair_pos] = iab;
            pair_jba[pair_pos] = jba;
            pair_i[pair_pos] = ioff;
            pair_j[pair_pos] = joff;
            pair_cross[pair_pos] = coff;
            pair_diag[pair_pos] = (ii == jj) ? 1 : 0;
            pair_w[pair_pos] = pair_w_pos + 1;
            for (int m = 0; m < total; ++m) {
              const double value = direct_flag != 0 ? direct_w[m] : wj[kr + m];
              pair_wj[pair_w_pos + m] = value;
              pair_wk[pair_w_pos + m] = direct_flag != 0 ? value : wk[kr + m];
            }
            pair_w_pos += total;
            ++pair_pos;
          }
        }
        if (direct_flag == 0) kr += mozyme_resident_pair_integral_count_dev(iab, jba);
      } else {
        if ((calci || calcj) &&
            mozyme_resident_point_supported_for_direct_dev(iab, jba, addr,
                                                           direct_flag)) {
          const int ioff = mozyme_resident_nijbo_at(numat, nijbo, ii, ii) + 1;
          const int joff = mozyme_resident_nijbo_at(numat, nijbo, jj, jj) + 1;
          const int ni = mozyme_resident_tri_dev(iab);
          const int nj = mozyme_resident_tri_dev(jba);
          if (point_pos >= point_count || !point_iab || !point_jba ||
              !point_i_atom || !point_j_atom || !point_i || !point_j ||
              !point_addr || !point_w ||
              !mozyme_resident_valid_plan_range_dev(ioff, ni, mpack) ||
              !mozyme_resident_valid_plan_range_dev(joff, nj, mpack)) {
            status[0] = 5;
            return;
          }
          point_iab[point_pos] = iab;
          point_jba[point_pos] = jba;
          point_i_atom[point_pos] = ii;
          point_j_atom[point_pos] = jj;
          point_i[point_pos] = ioff;
          point_j[point_pos] = joff;
          point_addr[point_pos] = addr;
          if (!mozyme_resident_pack_point_weights_dev(
                  iab, jba, addr, kr, ii, jj, direct_flag, semidr_flag,
                  l_feather_flag, ev, a0, trunc_1, trunc_2, w_count, nat,
                  coord, wj, am, ad, dd,
                  point_w + static_cast<size_t>(point_pos) * 7u)) {
            status[0] = 6;
            return;
          }
          if (addr == -2) {
            ++point_dipole_count;
          } else {
            ++point_monopole_count;
          }
          ++point_pos;
        }
        mozyme_resident_advance_point_kr_dev(iab, jba, addr, direct_flag,
                                             semidr_flag, &kr);
      }
    }

    const int ilim = mozyme_resident_tri_dev(iab);
    const int one_total = ilim * ilim;
    const int ioff = mozyme_resident_nijbo_at(numat, nijbo, ii, ii) + 1;
    const int source_kr = direct_flag != 0 ? one_kr : kr;
    if (mozyme_resident_basis_supported_dev(iab)) {
      if (one_pos >= one_count || one_w_pos + one_total > one_w_count ||
          !one_f || !one_w || !one_iab || !one_ilim || !one_w_values ||
          !mozyme_resident_valid_plan_range_dev(ioff, ilim, mpack) ||
          !mozyme_resident_valid_source_range_dev(source_kr, one_total, w_count)) {
        status[0] = 7;
        return;
      }
      one_f[one_pos] = ioff;
      one_w[one_pos] = one_w_pos + 1;
      one_iab[one_pos] = iab;
      one_ilim[one_pos] = ilim;
      for (int m = 0; m < one_total; ++m) {
        one_w_values[one_w_pos + m] = wj[source_kr + m];
      }
      ++one_pos;
      one_w_pos += one_total;
    }
    if (direct_flag != 0) {
      one_kr += one_total;
    } else {
      kr += one_total;
    }
  }

  status[1] = one_pos;
  status[2] = pair_pos;
  status[3] = pair4_pos;
  status[4] = point_pos;
  status[5] = one_w_pos;
  status[6] = pair_w_pos;
  status[7] = point_dipole_count;
  status[8] = point_monopole_count;
  status[9] = kr;
  status[10] = one_kr;
  if (one_pos != one_count || pair_pos != pair_count ||
      pair4_pos != pair4_count || point_pos != point_count ||
      one_w_pos != one_w_count || pair_w_pos != pair_w_count) {
    status[0] = 8;
  }
}


// ---------------------------------------------------------------------------
// Parallel direct-mode plan packing.  Rows (atoms ii) are classified in
// parallel, the per-row counts are scanned into task and integral offsets,
// descriptors are written per row, and the two-centre integrals are
// evaluated one task per thread.  The serial kernels above remain in use for
// the non-direct (host integral array) source path.
//
// Row count slots: 0 pair, 1 pair4, 2 point, 3 pair_w, 4 one_sup,
// 5 one_w_sup (destination), 6 one_w_all (source offset into wj).
static constexpr int kMzParRowSlots = 7;
static constexpr int kMzParScratchSlots = 1024;
static constexpr int kMzParThreads = 128;

__device__ __forceinline__ void mozyme_resident_increment_fallback_basis_atomic_dev(
    int iab, int jba, int *fallback_basis) {
  const int ib = mozyme_resident_fallback_basis_bin_dev(iab);
  const int jb = mozyme_resident_fallback_basis_bin_dev(jba);
  atomicAdd(fallback_basis + ib + jb * kMozymeResidentFallbackBasisBins, 1);
}

__global__ void mozyme_resident_calc_flags_kernel(int numat, int mode,
                                                  const int *kopt, int *calc) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int ired = 1;
  for (int ii = 1; ii <= numat; ++ii) {
    bool c = true;
    if (mode != 0) {
      c = (kopt[ired - 1] == ii);
      if (c && ired < numat) ++ired;
    }
    calc[ii - 1] = c ? 1 : 0;
  }
}

__global__ void mozyme_resident_row_count_kernel(
    int numat, int ione, int direct_flag, const int *iorbs, const int *calc,
    const int *nijbo, int *row_counts, int *out, int *fallback_basis) {
  const int ii = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (ii > numat) return;
  int pair = 0, pair4 = 0, point = 0, pairw = 0;
  int real_pair = 0, real_gpu = 0, real_cpu = 0, real_inactive = 0;
  int real_basis = 0, real_other = 0;
  int ppair = 0, pgpu = 0, pcpu = 0, pbasis = 0, pother = 0;
  const int iab = iorbs[ii - 1];
  const bool calci = calc[ii - 1] != 0;
  if (iab != 0) {
    for (int jj = 1; jj <= ii - ione; ++jj) {
      const bool calcj = calc[jj - 1] != 0;
      const int jba = iorbs[jj - 1];
      const int addr = mozyme_resident_nijbo_at(numat, nijbo, ii, jj);
      if (addr >= 0) {
        if (calci || calcj) {
          ++real_pair;
          if (mozyme_resident_pair_supported_for_direct_dev(iab, jba,
                                                            direct_flag)) {
            ++real_gpu;
            if ((iab == 4 && jba == 1) || (iab == 1 && jba == 4)) {
              ++pair4;
            } else {
              ++pair;
              pairw += mozyme_resident_pair_integral_count_dev(iab, jba);
            }
          } else if (!mozyme_resident_pair_noop_dev(iab, jba)) {
            ++real_cpu;
            if (iab > 9 || jba > 9) ++real_basis; else ++real_other;
            mozyme_resident_increment_fallback_basis_atomic_dev(iab, jba,
                                                               fallback_basis);
          }
        } else {
          ++real_inactive;
        }
      } else if ((calci || calcj) && iab * jba > 0) {
        ++ppair;
        if (mozyme_resident_point_supported_for_direct_dev(iab, jba, addr,
                                                           direct_flag)) {
          ++pgpu;
          ++point;
        } else {
          ++pcpu;
          if (iab > 9 || jba > 9) ++pbasis; else ++pother;
          mozyme_resident_increment_fallback_basis_atomic_dev(iab, jba,
                                                             fallback_basis);
        }
      }
    }
  }
  int one_sup = 0, one_w_sup = 0, one_w_all = 0, one_cpu = 0;
  if (iab != 0) {
    const int tri = mozyme_resident_tri_dev(iab);
    one_w_all = tri * tri;
    if (mozyme_resident_basis_supported_dev(iab)) {
      one_sup = 1;
      one_w_sup = tri * tri;
    } else {
      one_cpu = 1;
      mozyme_resident_increment_fallback_basis_atomic_dev(iab, iab,
                                                         fallback_basis);
    }
  }
  row_counts[0 * numat + ii - 1] = pair;
  row_counts[1 * numat + ii - 1] = pair4;
  row_counts[2 * numat + ii - 1] = point;
  row_counts[3 * numat + ii - 1] = pairw;
  row_counts[4 * numat + ii - 1] = one_sup;
  row_counts[5 * numat + ii - 1] = one_w_sup;
  row_counts[6 * numat + ii - 1] = one_w_all;
  if (real_pair) atomicAdd(out + 6, real_pair);
  if (real_gpu) atomicAdd(out + 7, real_gpu);
  if (real_cpu) atomicAdd(out + 8, real_cpu);
  if (real_inactive) atomicAdd(out + 9, real_inactive);
  if (real_basis) atomicAdd(out + 10, real_basis);
  if (real_other) atomicAdd(out + 11, real_other);
  if (ppair) atomicAdd(out + 12, ppair);
  if (pgpu) atomicAdd(out + 13, pgpu);
  if (pcpu) atomicAdd(out + 14, pcpu);
  if (pbasis) atomicAdd(out + 15, pbasis);
  if (pother) atomicAdd(out + 16, pother);
  if (one_cpu) atomicAdd(out + 17, one_cpu);
}

// Single block: exclusive scans of the row-count slots into row_bases and
// the totals into the count/status words.
__global__ void mozyme_resident_row_scan_kernel(int numat, const int *row_counts,
                                                int *row_bases, int *out,
                                                int *status) {
  __shared__ int warp_sums[32];
  __shared__ int carry;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  int totals[kMzParRowSlots];
  for (int slot = 0; slot < kMzParRowSlots; ++slot) {
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    const int *src = row_counts + slot * numat;
    int *dst = row_bases + slot * numat;
    for (int chunk = 0; chunk < numat; chunk += blockDim.x) {
      const int i = chunk + threadIdx.x;
      int v = i < numat ? src[i] : 0;
      int incl = v;
      for (int off = 1; off < 32; off <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += n;
      }
      if (lane == 31) warp_sums[warp] = incl;
      __syncthreads();
      if (warp == 0) {
        int ws = lane < warps ? warp_sums[lane] : 0;
        __syncwarp();
        for (int off = 1; off < 32; off <<= 1) {
          const int n = __shfl_up_sync(0xffffffffu, ws, off);
          if (lane >= off) ws += n;
        }
        if (lane < warps) warp_sums[lane] = ws;
      }
      __syncthreads();
      const int prefix = (warp > 0 ? warp_sums[warp - 1] : 0) + incl - v;
      const int total = warp_sums[warps - 1];
      if (i < numat) dst[i] = carry + prefix;
      __syncthreads();
      if (threadIdx.x == 0) carry += total;
      __syncthreads();
    }
    totals[slot] = carry;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[0] = totals[4];
    out[1] = totals[0];
    out[2] = totals[1];
    out[3] = totals[2];
    out[4] = totals[5];
    out[5] = totals[3];
    out[18] = 0;
    if (status) {
      status[1] = totals[4];
      status[2] = totals[0];
      status[3] = totals[1];
      status[4] = totals[2];
      status[5] = totals[5];
      status[6] = totals[3];
      status[9] = 0;
      status[10] = totals[6];
    }
  }
}

__global__ void mozyme_resident_row_pack_kernel(
    int numat, int mpack, int ione, int direct_flag, int w_count,
    const int *iorbs, const int *calc, const int *nijbo, const double *wj,
    const int *row_bases, int *one_f, int *one_w, int *one_iab, int *one_ilim,
    double *one_w_values, int *pair_iab, int *pair_jba, int *pair_i,
    int *pair_j, int *pair_cross, int *pair_diag, int *pair_w, int *task_ii,
    int *task_jj, int *pair4_heavy, int *pair4_light, int *pair4_cross,
    int *task4_ii, int *task4_jj, int *point_iab, int *point_jba,
    int *point_i_atom, int *point_j_atom, int *point_i, int *point_j,
    int *point_addr, int *status) {
  const int ii = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (ii > numat) return;
  const int iab = iorbs[ii - 1];
  if (iab == 0) return;
  const bool calci = calc[ii - 1] != 0;
  int pair_pos = row_bases[0 * numat + ii - 1];
  int pair4_pos = row_bases[1 * numat + ii - 1];
  int point_pos = row_bases[2 * numat + ii - 1];
  int pair_w_pos = row_bases[3 * numat + ii - 1];
  const int ioff = mozyme_resident_nijbo_at(numat, nijbo, ii, ii) + 1;
  for (int jj = 1; jj <= ii - ione; ++jj) {
    const bool calcj = calc[jj - 1] != 0;
    const int jba = iorbs[jj - 1];
    const int addr = mozyme_resident_nijbo_at(numat, nijbo, ii, jj);
    if (addr >= 0) {
      if (!(calci || calcj) ||
          !mozyme_resident_pair_supported_for_direct_dev(iab, jba, direct_flag)) {
        continue;
      }
      const int joff = mozyme_resident_nijbo_at(numat, nijbo, jj, jj) + 1;
      const int coff = addr + 1;
      if ((iab == 4 && jba == 1) || (iab == 1 && jba == 4)) {
        const bool heavy_i = (iab == 4);
        if (!mozyme_resident_valid_plan_range_dev(heavy_i ? ioff : joff, 10, mpack) ||
            !mozyme_resident_valid_plan_range_dev(heavy_i ? joff : ioff, 1, mpack) ||
            !mozyme_resident_valid_plan_range_dev(coff, 4, mpack)) {
          atomicMax(status, 3);
          return;
        }
        pair4_heavy[pair4_pos] = heavy_i ? ioff : joff;
        pair4_light[pair4_pos] = heavy_i ? joff : ioff;
        pair4_cross[pair4_pos] = coff;
        task4_ii[pair4_pos] = ii;
        task4_jj[pair4_pos] = jj;
        ++pair4_pos;
      } else {
        const int ni = mozyme_resident_tri_dev(iab);
        const int nj = mozyme_resident_tri_dev(jba);
        const int total = ni * nj;
        if (total > mozyme_resident_pair_integral_count_dev(iab, jba) ||
            !mozyme_resident_valid_plan_range_dev(ioff, ni, mpack) ||
            !mozyme_resident_valid_plan_range_dev(joff, nj, mpack) ||
            (ii != jj &&
             !mozyme_resident_valid_plan_range_dev(coff, iab * jba, mpack)) ||
            (ii == jj && coff < 1)) {
          atomicMax(status, 4);
          return;
        }
        pair_iab[pair_pos] = iab;
        pair_jba[pair_pos] = jba;
        pair_i[pair_pos] = ioff;
        pair_j[pair_pos] = joff;
        pair_cross[pair_pos] = coff;
        pair_diag[pair_pos] = (ii == jj) ? 1 : 0;
        pair_w[pair_pos] = pair_w_pos + 1;
        task_ii[pair_pos] = ii;
        task_jj[pair_pos] = jj;
        pair_w_pos += total;
        ++pair_pos;
      }
    } else if ((calci || calcj) &&
               mozyme_resident_point_supported_for_direct_dev(iab, jba, addr,
                                                              direct_flag)) {
      const int joff = mozyme_resident_nijbo_at(numat, nijbo, jj, jj) + 1;
      const int ni = mozyme_resident_tri_dev(iab);
      const int nj = mozyme_resident_tri_dev(jba);
      if (!mozyme_resident_valid_plan_range_dev(ioff, ni, mpack) ||
          !mozyme_resident_valid_plan_range_dev(joff, nj, mpack)) {
        atomicMax(status, 5);
        return;
      }
      point_iab[point_pos] = iab;
      point_jba[point_pos] = jba;
      point_i_atom[point_pos] = ii;
      point_j_atom[point_pos] = jj;
      point_i[point_pos] = ioff;
      point_j[point_pos] = joff;
      point_addr[point_pos] = addr;
      ++point_pos;
    }
  }
  if (mozyme_resident_basis_supported_dev(iab)) {
    const int ilim = mozyme_resident_tri_dev(iab);
    const int one_total = ilim * ilim;
    const int one_pos = row_bases[4 * numat + ii - 1];
    const int one_w_pos = row_bases[5 * numat + ii - 1];
    const int source_kr = row_bases[6 * numat + ii - 1];
    if (!mozyme_resident_valid_plan_range_dev(ioff, ilim, mpack) ||
        !mozyme_resident_valid_source_range_dev(source_kr, one_total, w_count)) {
      atomicMax(status, 7);
      return;
    }
    one_f[one_pos] = ioff;
    one_w[one_pos] = one_w_pos + 1;
    one_iab[one_pos] = iab;
    one_ilim[one_pos] = ilim;
    for (int m = 0; m < one_total; ++m) {
      one_w_values[one_w_pos + m] = wj[source_kr + m];
    }
  }
}

// One thread per pair task; each thread owns a scratch slot (grid-stride).
__global__ void mozyme_resident_pair_integrals_kernel(
    int pair_count, int l_feather_flag, int method_pm7_flag, double ev,
    double a0, double trunc_1, double trunc_2, const int *nat,
    const double *coord, const double *am, const double *ad, const double *aq,
    const double *dd, const double *qq, const double *po, const double *ddp,
    const int *iod, double *scratch, const int *pair_iab, const int *pair_jba,
    const int *task_ii, const int *task_jj, const int *pair_w,
    double *pair_wj, double *pair_wk, int *status) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  double *direct_w = scratch +
      static_cast<size_t>(slot) * kMozymeResidentDirectPackScratchDoubles;
  double *spd_scratch = direct_w + kMozymeDirectMaxW;
  for (int t = slot; t < pair_count; t += stride) {
    const int iab = pair_iab[t];
    const int jba = pair_jba[t];
    const int ii = task_ii[t];
    const int jj = task_jj[t];
    const int direct_total = mozyme_resident_pair_integral_count_dev(iab, jba);
    const bool ok =
        (iab == 9 || jba == 9)
            ? mozyme_direct_spd_w_dev(iab, jba, nat[ii - 1], nat[jj - 1], ii,
                                      jj, l_feather_flag, method_pm7_flag, ev,
                                      a0, trunc_1, trunc_2, coord, am, ad, aq,
                                      dd, qq, po, ddp, iod, spd_scratch,
                                      direct_w, direct_total)
            : mozyme_direct_sp_w_dev(iab, jba, nat[ii - 1], nat[jj - 1], ii,
                                     jj, l_feather_flag, ev, a0, trunc_1,
                                     trunc_2, coord, am, ad, aq, dd, qq,
                                     spd_scratch, direct_w, direct_total);
    if (!ok) {
      atomicMax(status, 9);
      continue;
    }
    const int total = mozyme_resident_tri_dev(iab) * mozyme_resident_tri_dev(jba);
    const int wpos = pair_w[t] - 1;
    for (int m = 0; m < total; ++m) {
      pair_wj[wpos + m] = direct_w[m];
      pair_wk[wpos + m] = direct_w[m];
    }
  }
}

__global__ void mozyme_resident_pair4_integrals_kernel(
    int pair4_count, int l_feather_flag, double ev, double a0,
    double trunc_1, double trunc_2, const int *nat, const double *coord,
    const double *am, const double *ad, const double *aq, const double *dd,
    const double *qq, const int *iorbs, const int *jindex, double *scratch,
    const int *task4_ii, const int *task4_jj, double *pair4_wj,
    double *pair4_wk, int *status) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  double *direct_w = scratch +
      static_cast<size_t>(slot) * kMozymeResidentDirectPackScratchDoubles;
  double *sp_scratch = direct_w + kMozymeDirectMaxW;
  for (int t = slot; t < pair4_count; t += stride) {
    const int ii = task4_ii[t];
    const int jj = task4_jj[t];
    const int iab = iorbs[ii - 1];
    const int jba = iorbs[jj - 1];
    const int direct_total = 10;
    if (!mozyme_direct_sp_w_dev(iab, jba, nat[ii - 1], nat[jj - 1], ii, jj,
                                l_feather_flag, ev, a0, trunc_1, trunc_2,
                                coord, am, ad, aq, dd, qq, sp_scratch,
                                direct_w, direct_total)) {
      atomicMax(status, 9);
      continue;
    }
    for (int m = 0; m < 10; ++m) pair4_wj[t * 10 + m] = direct_w[m];
    for (int m = 0; m < 16; ++m) {
      const int idx = jindex[m] - 1;
      if (idx < 0 || idx >= direct_total) {
        atomicMax(status, 2);
        break;
      }
      pair4_wk[t * 16 + m] = direct_w[idx];
    }
  }
}

__global__ void mozyme_resident_point_weights_kernel(
    int point_count, int direct_flag, int semidr_flag, int l_feather_flag,
    double ev, double a0, double trunc_1, double trunc_2, int w_count,
    const int *nat, const double *coord, const double *wj, const double *am,
    const double *ad, const double *dd, const int *point_iab,
    const int *point_jba, const int *point_i_atom, const int *point_j_atom,
    const int *point_addr, double *point_w, int *status) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= point_count) return;
  const int iab = point_iab[t];
  const int jba = point_jba[t];
  const int addr = point_addr[t];
  if (!mozyme_resident_pack_point_weights_dev(
          iab, jba, addr, 0, point_i_atom[t], point_j_atom[t], direct_flag,
          semidr_flag, l_feather_flag, ev, a0, trunc_1, trunc_2, w_count, nat,
          coord, wj, am, ad, dd, point_w + static_cast<size_t>(t) * 7u)) {
    atomicMax(status, 6);
    return;
  }
  if (addr == -2) atomicAdd(status + 7, 1); else atomicAdd(status + 8, 1);
}

static DevBuf<int> g_mz_par_calc, g_mz_par_rows, g_mz_par_bases;
static DevBuf<int> g_mz_par_task_ii, g_mz_par_task_jj, g_mz_par_task4_ii, g_mz_par_task4_jj;
static DevBuf<double> g_mz_par_scratch;

extern "C" int mopac_cuda_mozyme_resident_fock_pack_plan(
    int plan_id, int mpack, int natoms, int mode, int ione, int direct_flag,
    int semidr_flag, int l_feather_flag, double ev, double a0, double trunc_1,
    double trunc_2, int w_count, const int *iorbs, const int *nat,
    const int *kopt, const int *nijbo, const int *jindex, const double *coord,
    const double *wj, const double *wk, const double *am, const double *ad,
    const double *aq, const double *dd, const double *qq, int *counts_out,
    int *fallback_basis_out, const double *po, const double *ddp,
    const double *tore, const int *iod, int method_pm7_flag,
    int64_t signature, int *full_coverage_out) {
  MozymeSparseFockPlan *plan = mozyme_sparse_fock_plan(plan_id);
  if (!plan) return 1;
  mozyme_sparse_fock_invalidate_plan(plan);
  if (mpack <= 0 || natoms <= 0 || w_count <= 0 || !iorbs || !nat || !nijbo ||
      !jindex || !coord || !wj || (direct_flag == 0 && !wk) || !am || !ad ||
      !aq || !dd || !qq || !counts_out || !fallback_basis_out || !po || !ddp || !tore || !iod ||
      !full_coverage_out || (mode != 0 && !kopt)) {
    return 1;
  }

  cudaStream_t s = g_stream ? g_stream : 0;
  const auto t_pack0 = std::chrono::steady_clock::now();
  const size_t atom_bytes = sizeof(int) * static_cast<size_t>(natoms);
  const size_t nijbo_bytes =
      sizeof(int) * static_cast<size_t>(natoms) * static_cast<size_t>(natoms);
  const size_t coord_bytes = sizeof(double) * static_cast<size_t>(3 * natoms);
  const size_t w_bytes = sizeof(double) * static_cast<size_t>(w_count);
  const size_t param_bytes = sizeof(double) * 107u;
  const size_t po_bytes = sizeof(double) * 9u * 107u;
  const size_t ddp_bytes = sizeof(double) * 6u * 107u;
  const size_t iod_bytes = sizeof(int) * 107u;
  const size_t direct_scratch_bytes =
      sizeof(double) *
      static_cast<size_t>(kMozymeResidentDirectPackScratchDoubles);

  if (!g_mz_res_count_iorbs.ensure(atom_bytes) ||
      !g_mz_res_count_out.ensure(sizeof(int) * 19u) ||
      !g_mz_res_count_fallback.ensure(
          sizeof(int) * kMozymeResidentFallbackBasisBins *
          kMozymeResidentFallbackBasisBins) ||
      !g_mz_res_pack_nat.ensure(atom_bytes) ||
      !g_mz_res_pack_jindex.ensure(sizeof(int) * 16u) ||
      !g_mz_res_pack_status.ensure(sizeof(int) * 12u) ||
      !g_mz_res_pack_coord.ensure(coord_bytes) ||
      !g_mz_res_pack_wj.ensure(w_bytes) ||
      (direct_flag == 0 && !g_mz_res_pack_wk.ensure(w_bytes)) ||
      !g_mz_res_pack_am.ensure(param_bytes) ||
      !g_mz_res_pack_ad.ensure(param_bytes) ||
      !g_mz_res_pack_aq.ensure(param_bytes) ||
      !g_mz_res_pack_dd.ensure(param_bytes) ||
      !g_mz_res_pack_qq.ensure(param_bytes) ||
      !g_mz_res_pack_tore.ensure(param_bytes) ||
      !g_mz_res_pack_po.ensure(po_bytes) ||
      !g_mz_res_pack_ddp.ensure(ddp_bytes) ||
      (direct_flag != 0 &&
       !g_mz_res_pack_direct_scratch.ensure(direct_scratch_bytes)) ||
      !g_mz_res_pack_iod.ensure(iod_bytes)) {
    return 2;
  }
  if (mode != 0 && !g_mz_res_count_kopt.ensure(atom_bytes)) return 2;

  int code = 0;
  auto copy_int = [s](DevBuf<int>& dst, const int *src, size_t bytes,
                      const char *label) -> int {
    cudaError_t status = cudaMemcpyAsync(dst.ptr, src, bytes,
                                         cudaMemcpyHostToDevice, s);
    if (status != cudaSuccess) {
      report_cuda_error(label, status);
      return 2;
    }
    return 0;
  };
  auto copy_double = [s](DevBuf<double>& dst, const double *src, size_t bytes,
                         const char *label) -> int {
    cudaError_t status = cudaMemcpyAsync(dst.ptr, src, bytes,
                                         cudaMemcpyHostToDevice, s);
    if (status != cudaSuccess) {
      report_cuda_error(label, status);
      return 2;
    }
    return 0;
  };
  code |= copy_int(g_mz_res_count_iorbs, iorbs, atom_bytes,
                   "resident fock pack copy iorbs");
  code |= copy_int(g_mz_res_pack_nat, nat, atom_bytes,
                   "resident fock pack copy nat");
  const int *nijbo_dev = mopac_cuda_mozyme_nijbo_device(nijbo, natoms);
  if (!nijbo_dev) {
    if (!g_mz_res_count_nijbo.ensure(nijbo_bytes)) return 2;
    code |= copy_int(g_mz_res_count_nijbo, nijbo, nijbo_bytes,
                     "resident fock pack copy nijbo");
    nijbo_dev = g_mz_res_count_nijbo.ptr;
  }
  code |= copy_int(g_mz_res_pack_jindex, jindex, sizeof(int) * 16u,
                   "resident fock pack copy jindex");
  if (mode != 0) {
    code |= copy_int(g_mz_res_count_kopt, kopt, atom_bytes,
                     "resident fock pack copy kopt");
  }
  code |= copy_double(g_mz_res_pack_coord, coord, coord_bytes,
                      "resident fock pack copy coord");
  code |= copy_double(g_mz_res_pack_wj, wj, w_bytes,
                      "resident fock pack copy wj");
  if (direct_flag == 0) {
    code |= copy_double(g_mz_res_pack_wk, wk, w_bytes,
                        "resident fock pack copy wk");
  }
  code |= copy_double(g_mz_res_pack_am, am, param_bytes,
                      "resident fock pack copy am");
  code |= copy_double(g_mz_res_pack_ad, ad, param_bytes,
                      "resident fock pack copy ad");
  code |= copy_double(g_mz_res_pack_aq, aq, param_bytes,
                      "resident fock pack copy aq");
  code |= copy_double(g_mz_res_pack_dd, dd, param_bytes,
                      "resident fock pack copy dd");
  code |= copy_double(g_mz_res_pack_qq, qq, param_bytes,
                      "resident fock pack copy qq");
  code |= copy_double(g_mz_res_pack_tore, tore, param_bytes,
                      "resident fock pack copy tore");
  code |= copy_double(g_mz_res_pack_po, po, po_bytes,
                      "resident fock pack copy po");
  code |= copy_double(g_mz_res_pack_ddp, ddp, ddp_bytes,
                      "resident fock pack copy ddp");
  code |= copy_int(g_mz_res_pack_iod, iod, iod_bytes,
                   "resident fock pack copy iod");
  if (code != 0) return code;
  // nijbo (natoms^2 ints) and wj (n2elec doubles) dominate this upload.
  if (cudaStreamSynchronize(s) != cudaSuccess) return 2;
  mz_add_section_ms("fock_pack_upload", mz_host_ms_since(t_pack0));
  const auto t_pack1 = std::chrono::steady_clock::now();

  const bool parallel_pack = direct_flag != 0;
  const size_t row_bytes = sizeof(int) * static_cast<size_t>(natoms);
  if (parallel_pack) {
    if (!g_mz_par_calc.ensure(row_bytes) ||
        !g_mz_par_rows.ensure(row_bytes * kMzParRowSlots) ||
        !g_mz_par_bases.ensure(row_bytes * kMzParRowSlots)) {
      return 2;
    }
    if (cudaMemsetAsync(g_mz_res_count_out.ptr, 0, sizeof(int) * 19u, s) != cudaSuccess ||
        cudaMemsetAsync(g_mz_res_count_fallback.ptr, 0,
                        sizeof(int) * kMozymeResidentFallbackBasisBins *
                            kMozymeResidentFallbackBasisBins, s) != cudaSuccess ||
        cudaMemsetAsync(g_mz_res_pack_status.ptr, 0, sizeof(int) * 12u, s) != cudaSuccess) {
      return 2;
    }
    mozyme_resident_calc_flags_kernel<<<1, 1, 0, s>>>(
        natoms, mode, mode != 0 ? g_mz_res_count_kopt.ptr : nullptr,
        g_mz_par_calc.ptr);
    const int row_blocks = (natoms + kMzParThreads - 1) / kMzParThreads;
    mozyme_resident_row_count_kernel<<<row_blocks, kMzParThreads, 0, s>>>(
        natoms, ione, 1, g_mz_res_count_iorbs.ptr, g_mz_par_calc.ptr,
        nijbo_dev, g_mz_par_rows.ptr, g_mz_res_count_out.ptr,
        g_mz_res_count_fallback.ptr);
    mozyme_resident_row_scan_kernel<<<1, 1024, 0, s>>>(
        natoms, g_mz_par_rows.ptr, g_mz_par_bases.ptr, g_mz_res_count_out.ptr,
        g_mz_res_pack_status.ptr);
  } else {
    mozyme_resident_fock_count_kernel<<<1, 1, 0, s>>>(
        natoms, mode, ione, direct_flag != 0 ? 1 : 0, 1, g_mz_res_count_iorbs.ptr,
        mode != 0 ? g_mz_res_count_kopt.ptr : nullptr, nijbo_dev,
        g_mz_res_count_out.ptr, g_mz_res_count_fallback.ptr);
  }
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    report_cuda_error("resident fock pack count kernel launch", status);
    return 3;
  }
  status = cudaMemcpyAsync(counts_out, g_mz_res_count_out.ptr,
                           sizeof(int) * 19u, cudaMemcpyDeviceToHost, s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock pack copy counts", status);
    return 2;
  }
  status = cudaMemcpyAsync(fallback_basis_out, g_mz_res_count_fallback.ptr,
                           sizeof(int) * kMozymeResidentFallbackBasisBins *
                               kMozymeResidentFallbackBasisBins,
                           cudaMemcpyDeviceToHost, s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock pack copy fallback", status);
    return 2;
  }
  status = cudaStreamSynchronize(s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock pack count synchronize", status);
    return 2;
  }
  if (counts_out[18] != 0) return counts_out[18];
  mz_add_section_ms("fock_pack_count", mz_host_ms_since(t_pack1));
  const auto t_pack2 = std::chrono::steady_clock::now();

  const int one_count = counts_out[0];
  const int pair_count = counts_out[1];
  const int pair4_count = counts_out[2];
  const int point_count = counts_out[3];
  const int one_w_count = counts_out[4];
  const int pair_w_count = counts_out[5];
  const int full_coverage =
      (counts_out[8] == 0 && counts_out[14] == 0 &&
       counts_out[17] == 0) ? 1 : 0;
  const long long executable_work =
      static_cast<long long>(one_count) + static_cast<long long>(pair_count) +
      static_cast<long long>(pair4_count) + static_cast<long long>(point_count);
  if (full_coverage != 0 && executable_work <= 0) return 5;

  auto ensure_int_count = [](DevBuf<int>& dst, int count) -> bool {
    return dst.ensure(sizeof(int) * static_cast<size_t>(std::max(count, 0)));
  };
  auto ensure_double_count = [](DevBuf<double>& dst, int count) -> bool {
    return dst.ensure(sizeof(double) * static_cast<size_t>(std::max(count, 0)));
  };
  if (!ensure_int_count(plan->one_f, one_count) ||
      !ensure_int_count(plan->one_w, one_count) ||
      !ensure_int_count(plan->one_iab, one_count) ||
      !ensure_int_count(plan->one_ilim, one_count) ||
      !ensure_double_count(plan->one_w_values, one_w_count) ||
      !ensure_int_count(plan->pair_iab, pair_count) ||
      !ensure_int_count(plan->pair_jba, pair_count) ||
      !ensure_int_count(plan->pair_i, pair_count) ||
      !ensure_int_count(plan->pair_j, pair_count) ||
      !ensure_int_count(plan->pair_cross, pair_count) ||
      !ensure_int_count(plan->pair_diag, pair_count) ||
      !ensure_int_count(plan->pair_w, pair_count) ||
      !ensure_double_count(plan->pair_wj, pair_w_count) ||
      !ensure_double_count(plan->pair_wk, pair_w_count) ||
      !ensure_int_count(plan->pair4_heavy, pair4_count) ||
      !ensure_int_count(plan->pair4_light, pair4_count) ||
      !ensure_int_count(plan->pair4_cross, pair4_count) ||
      !ensure_double_count(plan->pair4_wj, pair4_count * 10) ||
      !ensure_double_count(plan->pair4_wk, pair4_count * 16) ||
      !ensure_int_count(plan->point_iab, point_count) ||
      !ensure_int_count(plan->point_jba, point_count) ||
      !ensure_int_count(plan->point_i_atom, point_count) ||
      !ensure_int_count(plan->point_j_atom, point_count) ||
      !ensure_int_count(plan->point_i, point_count) ||
      !ensure_int_count(plan->point_j, point_count) ||
      !ensure_int_count(plan->point_addr, point_count) ||
      !ensure_double_count(plan->point_w, point_count * 7)) {
    return 2;
  }
  if (parallel_pack) {
    const size_t task_bytes = sizeof(int) * static_cast<size_t>(std::max(pair_count, 1));
    const size_t task4_bytes = sizeof(int) * static_cast<size_t>(std::max(pair4_count, 1));
    const size_t scratch_bytes = sizeof(double) *
        static_cast<size_t>(kMzParScratchSlots) *
        static_cast<size_t>(kMozymeResidentDirectPackScratchDoubles);
    if (!g_mz_par_task_ii.ensure(task_bytes) || !g_mz_par_task_jj.ensure(task_bytes) ||
        !g_mz_par_task4_ii.ensure(task4_bytes) || !g_mz_par_task4_jj.ensure(task4_bytes) ||
        !g_mz_par_scratch.ensure(scratch_bytes)) {
      return 2;
    }
    const int row_blocks = (natoms + kMzParThreads - 1) / kMzParThreads;
    mozyme_resident_row_pack_kernel<<<row_blocks, kMzParThreads, 0, s>>>(
        natoms, mpack, ione, 1, w_count, g_mz_res_count_iorbs.ptr,
        g_mz_par_calc.ptr, nijbo_dev, g_mz_res_pack_wj.ptr,
        g_mz_par_bases.ptr, plan->one_f.ptr, plan->one_w.ptr,
        plan->one_iab.ptr, plan->one_ilim.ptr, plan->one_w_values.ptr,
        plan->pair_iab.ptr, plan->pair_jba.ptr, plan->pair_i.ptr,
        plan->pair_j.ptr, plan->pair_cross.ptr, plan->pair_diag.ptr,
        plan->pair_w.ptr, g_mz_par_task_ii.ptr, g_mz_par_task_jj.ptr,
        plan->pair4_heavy.ptr, plan->pair4_light.ptr, plan->pair4_cross.ptr,
        g_mz_par_task4_ii.ptr, g_mz_par_task4_jj.ptr, plan->point_iab.ptr,
        plan->point_jba.ptr, plan->point_i_atom.ptr, plan->point_j_atom.ptr,
        plan->point_i.ptr, plan->point_j.ptr, plan->point_addr.ptr,
        g_mz_res_pack_status.ptr);
    const int slot_blocks = kMzParScratchSlots / kMzParThreads;
    if (pair_count > 0) {
      mozyme_resident_pair_integrals_kernel<<<slot_blocks, kMzParThreads, 0, s>>>(
          pair_count, l_feather_flag, method_pm7_flag, ev, a0, trunc_1, trunc_2,
          g_mz_res_pack_nat.ptr, g_mz_res_pack_coord.ptr, g_mz_res_pack_am.ptr,
          g_mz_res_pack_ad.ptr, g_mz_res_pack_aq.ptr, g_mz_res_pack_dd.ptr,
          g_mz_res_pack_qq.ptr, g_mz_res_pack_po.ptr, g_mz_res_pack_ddp.ptr,
          g_mz_res_pack_iod.ptr, g_mz_par_scratch.ptr, plan->pair_iab.ptr,
          plan->pair_jba.ptr, g_mz_par_task_ii.ptr, g_mz_par_task_jj.ptr,
          plan->pair_w.ptr, plan->pair_wj.ptr, plan->pair_wk.ptr,
          g_mz_res_pack_status.ptr);
    }
    if (pair4_count > 0) {
      mozyme_resident_pair4_integrals_kernel<<<slot_blocks, kMzParThreads, 0, s>>>(
          pair4_count, l_feather_flag, ev, a0, trunc_1, trunc_2,
          g_mz_res_pack_nat.ptr, g_mz_res_pack_coord.ptr, g_mz_res_pack_am.ptr,
          g_mz_res_pack_ad.ptr, g_mz_res_pack_aq.ptr, g_mz_res_pack_dd.ptr,
          g_mz_res_pack_qq.ptr, g_mz_res_count_iorbs.ptr,
          g_mz_res_pack_jindex.ptr, g_mz_par_scratch.ptr,
          g_mz_par_task4_ii.ptr, g_mz_par_task4_jj.ptr, plan->pair4_wj.ptr,
          plan->pair4_wk.ptr, g_mz_res_pack_status.ptr);
    }
    if (point_count > 0) {
      const int point_blocks = (point_count + kMzParThreads - 1) / kMzParThreads;
      mozyme_resident_point_weights_kernel<<<point_blocks, kMzParThreads, 0, s>>>(
          point_count, 1, semidr_flag, l_feather_flag, ev, a0, trunc_1,
          trunc_2, w_count, g_mz_res_pack_nat.ptr, g_mz_res_pack_coord.ptr,
          g_mz_res_pack_wj.ptr, g_mz_res_pack_am.ptr, g_mz_res_pack_ad.ptr,
          g_mz_res_pack_dd.ptr, plan->point_iab.ptr, plan->point_jba.ptr,
          plan->point_i_atom.ptr, plan->point_j_atom.ptr, plan->point_addr.ptr,
          plan->point_w.ptr, g_mz_res_pack_status.ptr);
    }
  } else {
  status = cudaMemsetAsync(g_mz_res_pack_status.ptr, 0, sizeof(int) * 12u, s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock pack clear status", status);
    return 2;
  }

  mozyme_resident_fock_pack_plan_kernel<<<1, 1, 0, s>>>(
      natoms, mpack, mode, ione, direct_flag != 0 ? 1 : 0, semidr_flag,
      l_feather_flag, ev, a0, trunc_1, trunc_2, w_count,
      g_mz_res_count_iorbs.ptr, g_mz_res_pack_nat.ptr,
      mode != 0 ? g_mz_res_count_kopt.ptr : nullptr, nijbo_dev,
      g_mz_res_pack_jindex.ptr, g_mz_res_pack_coord.ptr, g_mz_res_pack_wj.ptr,
      direct_flag == 0 ? g_mz_res_pack_wk.ptr : nullptr, g_mz_res_pack_am.ptr, g_mz_res_pack_ad.ptr,
      g_mz_res_pack_aq.ptr, g_mz_res_pack_dd.ptr, g_mz_res_pack_qq.ptr,
      g_mz_res_pack_po.ptr, g_mz_res_pack_ddp.ptr, g_mz_res_pack_tore.ptr,
      g_mz_res_pack_iod.ptr, method_pm7_flag,
      direct_flag != 0 ? g_mz_res_pack_direct_scratch.ptr : nullptr,
      g_mz_res_count_out.ptr, plan->one_f.ptr, plan->one_w.ptr,
      plan->one_iab.ptr, plan->one_ilim.ptr, plan->one_w_values.ptr,
      plan->pair_iab.ptr, plan->pair_jba.ptr, plan->pair_i.ptr,
      plan->pair_j.ptr, plan->pair_cross.ptr, plan->pair_diag.ptr,
      plan->pair_w.ptr, plan->pair_wj.ptr, plan->pair_wk.ptr,
      plan->pair4_heavy.ptr, plan->pair4_light.ptr, plan->pair4_cross.ptr,
      plan->pair4_wj.ptr, plan->pair4_wk.ptr, plan->point_iab.ptr,
      plan->point_jba.ptr, plan->point_i_atom.ptr, plan->point_j_atom.ptr,
      plan->point_i.ptr, plan->point_j.ptr, plan->point_addr.ptr,
      plan->point_w.ptr, g_mz_res_pack_status.ptr);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    report_cuda_error("resident fock pack kernel launch", status);
    return 3;
  }
  int host_status[12] = {0};
  status = cudaMemcpyAsync(host_status, g_mz_res_pack_status.ptr,
                           sizeof(host_status), cudaMemcpyDeviceToHost, s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock pack copy status", status);
    return 2;
  }
  status = cudaStreamSynchronize(s);
  if (status != cudaSuccess) {
    report_cuda_error("resident fock pack synchronize", status);
    return 2;
  }
  if (host_status[0] != 0) return 10 + host_status[0];
  mz_add_section_ms("fock_pack_build", mz_host_ms_since(t_pack2));

  plan->mpack = mpack;
  plan->natoms = natoms;
  plan->one_count = one_count;
  plan->pair_count = pair_count;
  plan->pair4_count = pair4_count;
  plan->point_count = point_count;
  plan->point_dipole_count = host_status[7];
  plan->point_monopole_count = host_status[8];
  plan->signature = signature;
  plan->full_coverage = full_coverage != 0;
  plan->has_executable_work = executable_work > 0;
  plan->ready = true;
  *full_coverage_out = full_coverage;
  if (gpu_profile_enabled()) {
    std::fprintf(stderr,
                 "[GPU] profile mozyme_resident_fock_pack one=%d pair=%d pair4x1=%d point=%d point_dipole=%d point_monopole=%d one_w=%d pair_w=%d\n",
                 one_count, pair_count, pair4_count, point_count,
                 plan->point_dipole_count, plan->point_monopole_count,
                 one_w_count, pair_w_count);
  }
  return 0;
}

extern "C" int mopac_cuda_mozyme_sparse_fock_setup_plan(
                                                     int plan_id,
                                                     int mpack,
                                                     int natoms,
                                                     int one_count,
                                                     const int *one_f_offsets,
                                                     const int *one_w_offsets,
                                                     const int *one_iabs,
                                                     const int *one_ilims,
                                                     int one_w_values_count,
                                                     const double *one_w_values,
                                                     int pair_count,
                                                     const int *pair_iabs,
                                                     const int *pair_jbas,
                                                     const int *pair_i_offsets,
                                                     const int *pair_j_offsets,
                                                     const int *pair_cross_offsets,
                                                     const int *pair_diag_flags,
                                                     const int *pair_w_offsets,
                                                     int pair_w_values_count,
                                                     const double *pair_wj_values,
                                                     const double *pair_wk_values,
                                                     int pair4x1_count,
                                                     const int *pair4x1_heavy_offsets,
                                                     const int *pair4x1_light_offsets,
                                                     const int *pair4x1_cross_offsets,
                                                     const double *pair4x1_wj_values,
                                                     const double *pair4x1_wk_values,
                                                     int point_count,
                                                     const int *point_iabs,
                                                     const int *point_jbas,
                                                     const int *point_i_atoms,
                                                     const int *point_j_atoms,
                                                     const int *point_i_offsets,
                                                     const int *point_j_offsets,
                                                     const int *point_addr_flags,
                                                     const double *point_w_values,
                                                     int64_t signature,
                                                     int full_coverage) {
  MozymeSparseFockPlan *plan = mozyme_sparse_fock_plan(plan_id);
  if (!plan) return 1;
  mozyme_sparse_fock_invalidate_plan(plan);
  if (mpack <= 0 || natoms <= 0 || one_count < 0 || pair_count < 0 ||
      pair4x1_count < 0 || point_count < 0 || one_w_values_count < 0 ||
      pair_w_values_count < 0 || (full_coverage != 0 && full_coverage != 1)) return 1;
  auto fail_setup = [plan](int code) -> int {
    mozyme_sparse_fock_invalidate_plan(plan);
    return code;
  };
  auto tri = [](int n) -> long long {
    return (static_cast<long long>(n) * static_cast<long long>(n + 1)) / 2ll;
  };
  auto valid_range = [](int offset, long long count, int limit) -> bool {
    if (count < 0 || offset < 1) return false;
    const long long start = static_cast<long long>(offset) - 1ll;
    return start + count <= static_cast<long long>(limit);
  };
  if (one_count > 0 &&
      (!one_f_offsets || !one_w_offsets || !one_iabs || !one_ilims)) return fail_setup(3);
  for (int idx = 0; idx < one_count; ++idx) {
    const int iab = one_iabs[idx];
    const int ilim = one_ilims[idx];
    if (iab <= 0 || !mozyme_sparse_fock_basis_supported(iab) ||
        ilim != tri(iab)) return fail_setup(3);
    if (!valid_range(one_f_offsets[idx], ilim, mpack)) return fail_setup(3);
    if (!valid_range(one_w_offsets[idx], static_cast<long long>(ilim) * ilim,
                     one_w_values_count)) return fail_setup(3);
  }
  if (pair_count > 0 &&
      (!pair_iabs || !pair_jbas || !pair_i_offsets || !pair_j_offsets ||
       !pair_cross_offsets || !pair_diag_flags || !pair_w_offsets)) return fail_setup(3);
  for (int idx = 0; idx < pair_count; ++idx) {
    const int iab = pair_iabs[idx];
    const int jba = pair_jbas[idx];
    const long long ni = tri(iab);
    const long long nj = tri(jba);
    if (iab <= 0 || jba <= 0 ||
        !mozyme_sparse_fock_basis_supported(iab) ||
        !mozyme_sparse_fock_basis_supported(jba) ||
        ni <= 0 || nj <= 0) return fail_setup(3);
    if (pair_diag_flags[idx] != 0 && pair_diag_flags[idx] != 1) return fail_setup(3);
    if (!valid_range(pair_i_offsets[idx], ni, mpack)) return fail_setup(3);
    if (!valid_range(pair_j_offsets[idx], nj, mpack)) return fail_setup(3);
    if (pair_diag_flags[idx] == 0 &&
        !valid_range(pair_cross_offsets[idx],
                     static_cast<long long>(iab) * jba, mpack)) {
      return fail_setup(3);
    }
    if (pair_diag_flags[idx] == 1 && pair_cross_offsets[idx] < 1) {
      return fail_setup(3);
    }
    if (!valid_range(pair_w_offsets[idx], ni * nj, pair_w_values_count)) return fail_setup(3);
  }
  if (pair4x1_count > 0 &&
      (!pair4x1_heavy_offsets || !pair4x1_light_offsets ||
       !pair4x1_cross_offsets)) return fail_setup(3);
  for (int idx = 0; idx < pair4x1_count; ++idx) {
    if (!valid_range(pair4x1_heavy_offsets[idx], 10, mpack)) return fail_setup(3);
    if (!valid_range(pair4x1_light_offsets[idx], 1, mpack)) return fail_setup(3);
    if (!valid_range(pair4x1_cross_offsets[idx], 4, mpack)) return fail_setup(3);
  }
  if (point_count > 0 &&
      (!point_iabs || !point_jbas || !point_i_atoms || !point_j_atoms ||
       !point_i_offsets || !point_j_offsets || !point_addr_flags)) return fail_setup(3);
  for (int idx = 0; idx < point_count; ++idx) {
    const int iab = point_iabs[idx];
    const int jba = point_jbas[idx];
    const int addr_flag = point_addr_flags[idx];
    if (iab <= 0 || jba <= 0) return fail_setup(3);
    if (!mozyme_sparse_fock_basis_supported(iab) ||
        !mozyme_sparse_fock_basis_supported(jba)) return fail_setup(3);
    if (point_i_atoms[idx] < 1 || point_i_atoms[idx] > natoms ||
        point_j_atoms[idx] < 1 || point_j_atoms[idx] > natoms) return fail_setup(3);
    if (!valid_range(point_i_offsets[idx], tri(iab), mpack)) return fail_setup(3);
    if (!valid_range(point_j_offsets[idx], tri(jba), mpack)) return fail_setup(3);
    if (addr_flag >= 0) return fail_setup(3);
    if (addr_flag == -2) {
      ++plan->point_dipole_count;
    } else {
      ++plan->point_monopole_count;
    }
  }
  const long long executable_work =
      static_cast<long long>(one_count) + static_cast<long long>(pair_count) +
      static_cast<long long>(pair4x1_count) +
      static_cast<long long>(point_count);
  if (full_coverage != 0 && executable_work <= 0) return fail_setup(3);

  cudaStream_t s = g_stream ? g_stream : 0;
  auto copy_int = [s](DevBuf<int>& dst, const int *src, size_t count) -> int {
    size_t bytes = sizeof(int) * count;
    dst.ensure(bytes);
    if (bytes == 0) return 0;
    if (!src || !dst.ptr) return 1;
    return cudaMemcpyAsync(dst.ptr, src, bytes, cudaMemcpyHostToDevice, s) == cudaSuccess ? 0 : 2;
  };
  auto copy_double = [s](DevBuf<double>& dst, const double *src, size_t count) -> int {
    size_t bytes = sizeof(double) * count;
    dst.ensure(bytes);
    if (bytes == 0) return 0;
    if (!src || !dst.ptr) return 1;
    return cudaMemcpyAsync(dst.ptr, src, bytes, cudaMemcpyHostToDevice, s) == cudaSuccess ? 0 : 2;
  };

  int code = 0;
  code |= copy_int(plan->one_f, one_f_offsets, (size_t)one_count);
  code |= copy_int(plan->one_w, one_w_offsets, (size_t)one_count);
  code |= copy_int(plan->one_iab, one_iabs, (size_t)one_count);
  code |= copy_int(plan->one_ilim, one_ilims, (size_t)one_count);
  code |= copy_double(plan->one_w_values, one_w_values, (size_t)one_w_values_count);
  code |= copy_int(plan->pair_iab, pair_iabs, (size_t)pair_count);
  code |= copy_int(plan->pair_jba, pair_jbas, (size_t)pair_count);
  code |= copy_int(plan->pair_i, pair_i_offsets, (size_t)pair_count);
  code |= copy_int(plan->pair_j, pair_j_offsets, (size_t)pair_count);
  code |= copy_int(plan->pair_cross, pair_cross_offsets, (size_t)pair_count);
  code |= copy_int(plan->pair_diag, pair_diag_flags, (size_t)pair_count);
  code |= copy_int(plan->pair_w, pair_w_offsets, (size_t)pair_count);
  code |= copy_double(plan->pair_wj, pair_wj_values, (size_t)pair_w_values_count);
  code |= copy_double(plan->pair_wk, pair_wk_values, (size_t)pair_w_values_count);
  code |= copy_int(plan->pair4_heavy, pair4x1_heavy_offsets, (size_t)pair4x1_count);
  code |= copy_int(plan->pair4_light, pair4x1_light_offsets, (size_t)pair4x1_count);
  code |= copy_int(plan->pair4_cross, pair4x1_cross_offsets, (size_t)pair4x1_count);
  code |= copy_double(plan->pair4_wj, pair4x1_wj_values, (size_t)pair4x1_count * 10u);
  code |= copy_double(plan->pair4_wk, pair4x1_wk_values, (size_t)pair4x1_count * 16u);
  code |= copy_int(plan->point_iab, point_iabs, (size_t)point_count);
  code |= copy_int(plan->point_jba, point_jbas, (size_t)point_count);
  code |= copy_int(plan->point_i_atom, point_i_atoms, (size_t)point_count);
  code |= copy_int(plan->point_j_atom, point_j_atoms, (size_t)point_count);
  code |= copy_int(plan->point_i, point_i_offsets, (size_t)point_count);
  code |= copy_int(plan->point_j, point_j_offsets, (size_t)point_count);
  code |= copy_int(plan->point_addr, point_addr_flags, (size_t)point_count);
  code |= copy_double(plan->point_w, point_w_values, (size_t)point_count * 7u);
  if (code != 0) {
    mozyme_sparse_fock_invalidate_plan(plan);
    return code;
  }
  if (cudaStreamSynchronize(s) != cudaSuccess) {
    mozyme_sparse_fock_invalidate_plan(plan);
    return 2;
  }
  plan->mpack = mpack;
  plan->natoms = natoms;
  plan->one_count = one_count;
  plan->pair_count = pair_count;
  plan->pair4_count = pair4x1_count;
  plan->point_count = point_count;
  plan->signature = signature;
  plan->full_coverage = full_coverage != 0;
  plan->has_executable_work = executable_work > 0;
  plan->ready = true;
  if (gpu_profile_enabled()) {
    std::fprintf(stderr,
                 "[GPU] profile mozyme_sparse_fock_setup one=%d pair=%d pair4x1=%d point=%d point_dipole=%d point_monopole=%d one_w=%d pair_w=%d\n",
                 one_count, pair_count, pair4x1_count, point_count,
                 plan->point_dipole_count, plan->point_monopole_count,
                 one_w_values_count, pair_w_values_count);
  }
  return 0;
}

extern "C" int mopac_cuda_mozyme_sparse_fock_setup(int mpack,
                                                     int natoms,
                                                     int one_count,
                                                     const int *one_f_offsets,
                                                     const int *one_w_offsets,
                                                     const int *one_iabs,
                                                     const int *one_ilims,
                                                     int one_w_values_count,
                                                     const double *one_w_values,
                                                     int pair_count,
                                                     const int *pair_iabs,
                                                     const int *pair_jbas,
                                                     const int *pair_i_offsets,
                                                     const int *pair_j_offsets,
                                                     const int *pair_cross_offsets,
                                                     const int *pair_diag_flags,
                                                     const int *pair_w_offsets,
                                                     int pair_w_values_count,
                                                     const double *pair_wj_values,
                                                     const double *pair_wk_values,
                                                     int pair4x1_count,
                                                     const int *pair4x1_heavy_offsets,
                                                     const int *pair4x1_light_offsets,
                                                     const int *pair4x1_cross_offsets,
                                                     const double *pair4x1_wj_values,
                                                     const double *pair4x1_wk_values,
                                                     int point_count,
                                                     const int *point_iabs,
                                                     const int *point_jbas,
                                                     const int *point_i_atoms,
                                                     const int *point_j_atoms,
                                                     const int *point_i_offsets,
                                                     const int *point_j_offsets,
                                                     const int *point_addr_flags,
                                                     const double *point_w_values) {
  return mopac_cuda_mozyme_sparse_fock_setup_plan(
      kMozymeSparseFockPlanDefault, mpack, natoms, one_count, one_f_offsets,
      one_w_offsets, one_iabs, one_ilims, one_w_values_count, one_w_values,
      pair_count, pair_iabs, pair_jbas, pair_i_offsets, pair_j_offsets,
      pair_cross_offsets, pair_diag_flags, pair_w_offsets,
      pair_w_values_count, pair_wj_values, pair_wk_values, pair4x1_count,
      pair4x1_heavy_offsets, pair4x1_light_offsets, pair4x1_cross_offsets,
      pair4x1_wj_values, pair4x1_wk_values, point_count, point_iabs,
      point_jbas, point_i_atoms, point_j_atoms, point_i_offsets,
      point_j_offsets, point_addr_flags, point_w_values, 0, 1);
}

extern "C" void mopac_cuda_mozyme_sparse_fock_set_full_coverage_plan(
    int plan_id, int complete) {
  MozymeSparseFockPlan *plan = mozyme_sparse_fock_plan(plan_id);
  if (plan && complete == 0) plan->full_coverage = false;
}

extern "C" void mopac_cuda_mozyme_sparse_fock_set_full_coverage(int complete) {
  mopac_cuda_mozyme_sparse_fock_set_full_coverage_plan(
      kMozymeSparseFockPlanDefault, complete);
}

// Launches the point-pair Fock terms of a plan (see the kernels above).
// Returns 0 on success, 2 on allocation failure, 3 on launch failure.
static int mozyme_sparse_point_terms_launch(MozymeSparseFockPlan *plan,
                                            const double *qe_dev,
                                            const double *ptot_dev,
                                            double *f_dev,
                                            const int *guard_ints,
                                            int guard_slot,
                                            int guard_continue,
                                            cudaStream_t s) {
  if (plan->point_count <= 0) return 0;
  const size_t natoms = static_cast<size_t>(plan->natoms);
  if (natoms == 0) return 2;
  const size_t acc_bytes = sizeof(double) * natoms * kMzPointAccStride;
  const size_t atom_bytes = sizeof(int) * natoms;
  if (!plan->point_acc.ensure(acc_bytes) ||
      !plan->point_atom_off.ensure(atom_bytes) ||
      !plan->point_atom_iab.ensure(atom_bytes)) {
    return 2;
  }
  constexpr int kThreads = 256;
  const int task_blocks = static_cast<int>(
      std::min<long long>((plan->point_count + kThreads - 1) / kThreads, 8192ll));
  const int atom_blocks = static_cast<int>((natoms + kThreads - 1) / kThreads);
  if (!plan->point_index_ready) {
    if (cudaMemsetAsync(plan->point_atom_iab.ptr, 0, atom_bytes, s) != cudaSuccess ||
        cudaMemsetAsync(plan->point_atom_off.ptr, 0, atom_bytes, s) != cudaSuccess) {
      return 2;
    }
    mozyme_sparse_point_atom_index_kernel<<<task_blocks, kThreads, 0, s>>>(
        plan->point_count, plan->point_iab.ptr, plan->point_jba.ptr,
        plan->point_i_atom.ptr, plan->point_j_atom.ptr, plan->point_i.ptr,
        plan->point_j.ptr, plan->point_atom_off.ptr, plan->point_atom_iab.ptr);
    if (cudaPeekAtLastError() != cudaSuccess) return 3;
    plan->point_index_ready = true;
  }
  if (cudaMemsetAsync(plan->point_acc.ptr, 0, acc_bytes, s) != cudaSuccess) return 2;
  mozyme_sparse_point_accumulate_kernel<<<task_blocks, kThreads, 0, s>>>(
      plan->point_count, plan->point_iab.ptr, plan->point_jba.ptr,
      plan->point_i_atom.ptr, plan->point_j_atom.ptr, plan->point_i.ptr,
      plan->point_j.ptr, plan->point_addr.ptr, plan->point_w.ptr, qe_dev,
      ptot_dev, plan->point_acc.ptr, guard_ints, guard_slot, guard_continue);
  if (cudaPeekAtLastError() != cudaSuccess) return 3;
  mozyme_sparse_point_apply_kernel<<<atom_blocks, kThreads, 0, s>>>(
      plan->natoms, plan->point_atom_off.ptr, plan->point_atom_iab.ptr,
      plan->point_acc.ptr, f_dev, guard_ints, guard_slot, guard_continue);
  if (cudaPeekAtLastError() != cudaSuccess) return 3;
  return 0;
}

extern "C" int mopac_cuda_mozyme_sparse_fock_run(int mpack,
                                                  const double *ptot,
                                                  const double *qe,
                                                  double *f) {
  MozymeSparseFockPlan *plan =
      mozyme_sparse_fock_plan(kMozymeSparseFockPlanDefault);
  if (!plan || !plan->ready || !plan->full_coverage ||
      !plan->has_executable_work ||
      mpack <= 0 || mpack != plan->mpack ||
      !ptot || !qe || !f) return 1;
  cudaStream_t s = g_stream ? g_stream : 0;
  size_t bytes = sizeof(double) * (size_t)mpack;
  size_t qe_bytes = sizeof(double) * (size_t)plan->natoms;
  auto t0 = std::chrono::high_resolution_clock::now();
  plan->ptot.ensure(bytes);
  plan->f.ensure(bytes);
  plan->qe.ensure(qe_bytes);
  if (!plan->ptot.ptr || !plan->qe.ptr || !plan->f.ptr) return 2;
  if (cudaMemcpyAsync(plan->ptot.ptr, ptot, bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(plan->qe.ptr, qe, qe_bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(plan->f.ptr, f, bytes, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (plan->one_count > 0) {
    mozyme_sparse_one_center_kernel<<<plan->one_count, 64, 0, s>>>(
        plan->one_count, plan->one_f.ptr, plan->one_w.ptr, plan->one_iab.ptr,
        plan->one_ilim.ptr, plan->one_w_values.ptr, plan->ptot.ptr,
        plan->f.ptr, nullptr, -1, 0);
    if (cudaPeekAtLastError() != cudaSuccess) return 3;
  }
  if (plan->pair_count > 0) {
    mozyme_sparse_pair_kernel<<<plan->pair_count, 256, 0, s>>>(
        plan->pair_count, plan->pair_iab.ptr, plan->pair_jba.ptr, plan->pair_i.ptr,
        plan->pair_j.ptr, plan->pair_cross.ptr, plan->pair_diag.ptr, plan->pair_w.ptr,
        plan->pair_wj.ptr, plan->pair_wk.ptr, plan->ptot.ptr, plan->f.ptr,
        nullptr, -1, 0);
    if (cudaPeekAtLastError() != cudaSuccess) return 3;
  }
  if (plan->pair4_count > 0) {
    mozyme_sparse_4x1_kernel<<<plan->pair4_count, 16, 0, s>>>(
        plan->pair4_count, plan->pair4_heavy.ptr, plan->pair4_light.ptr, plan->pair4_cross.ptr,
        plan->pair4_wj.ptr, plan->pair4_wk.ptr, plan->ptot.ptr, plan->f.ptr,
        nullptr, -1, 0);
    if (cudaPeekAtLastError() != cudaSuccess) return 3;
  }
  {
    const int point_code = mozyme_sparse_point_terms_launch(
        plan, plan->qe.ptr, plan->ptot.ptr, plan->f.ptr, nullptr, -1, 0, s);
    if (point_code != 0) return point_code;
  }
  if (cudaMemcpyAsync(f, plan->f.ptr, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  if (cudaStreamSynchronize(s) != cudaSuccess) return 2;
  if (gpu_profile_enabled()) {
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::fprintf(stderr,
                 "[GPU] profile mozyme_sparse_fock_run one=%d pair=%d pair4x1=%d point=%d point_dipole=%d point_monopole=%d ms=%.3f\n",
                 plan->one_count, plan->pair_count, plan->pair4_count,
                 plan->point_count, plan->point_dipole_count,
                 plan->point_monopole_count, ms);
  }
  return 0;
}

static int mozyme_sparse_fock_run_device_plan_guarded_impl(
                                                         int plan_id,
                                                         int mpack,
                                                         const double *ptot_dev,
                                                         const double *qe_dev,
                                                         double *f_dev,
                                                         const int *guard_ints,
                                                         int guard_slot,
                                                         int guard_continue,
                                                         bool force_default_stream,
                                                         int synchronize,
                                                         double *wall_ms) {
  if (wall_ms) *wall_ms = 0.0;
  MozymeSparseFockPlan *plan = mozyme_sparse_fock_plan(plan_id);
  if (!plan || !plan->ready || !plan->full_coverage ||
      !plan->has_executable_work || mpack <= 0 ||
      mpack != plan->mpack || !ptot_dev || !qe_dev || !f_dev) {
    return 1;
  }

  cudaStream_t s = force_default_stream ? 0 : (g_stream ? g_stream : 0);
  const bool measure = wall_ms != nullptr;
  const bool wait_for_completion = (synchronize != 0) || measure;
  auto t0 = std::chrono::high_resolution_clock::now();
  if (plan->one_count > 0) {
    mozyme_sparse_one_center_kernel<<<plan->one_count, 64, 0, s>>>(
        plan->one_count, plan->one_f.ptr, plan->one_w.ptr,
        plan->one_iab.ptr, plan->one_ilim.ptr,
        plan->one_w_values.ptr, ptot_dev, f_dev, guard_ints, guard_slot,
        guard_continue);
    if (!report_cuda_error("mozyme_sparse_fock_run_device one-center kernel",
                           cudaPeekAtLastError())) return 3;
  }
  if (plan->pair_count > 0) {
    mozyme_sparse_pair_kernel<<<plan->pair_count, 256, 0, s>>>(
        plan->pair_count, plan->pair_iab.ptr, plan->pair_jba.ptr,
        plan->pair_i.ptr, plan->pair_j.ptr, plan->pair_cross.ptr,
        plan->pair_diag.ptr, plan->pair_w.ptr, plan->pair_wj.ptr,
        plan->pair_wk.ptr, ptot_dev, f_dev, guard_ints, guard_slot,
        guard_continue);
    if (!report_cuda_error("mozyme_sparse_fock_run_device pair kernel",
                           cudaPeekAtLastError())) return 3;
  }
  if (plan->pair4_count > 0) {
    mozyme_sparse_4x1_kernel<<<plan->pair4_count, 16, 0, s>>>(
        plan->pair4_count, plan->pair4_heavy.ptr,
        plan->pair4_light.ptr, plan->pair4_cross.ptr,
        plan->pair4_wj.ptr, plan->pair4_wk.ptr, ptot_dev, f_dev, guard_ints,
        guard_slot, guard_continue);
    if (!report_cuda_error("mozyme_sparse_fock_run_device 4x1 kernel",
                           cudaPeekAtLastError())) return 3;
  }
  {
    const int point_code = mozyme_sparse_point_terms_launch(
        plan, qe_dev, ptot_dev, f_dev, guard_ints, guard_slot, guard_continue, s);
    if (point_code != 0) {
      report_cuda_error("mozyme_sparse_fock_run_device point terms",
                        cudaPeekAtLastError());
      return point_code;
    }
  }
  if (wait_for_completion &&
      !report_cuda_error("mozyme_sparse_fock_run_device synchronize",
                         cudaStreamSynchronize(s))) return 2;

  const double elapsed = measure
      ? std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0)
            .count()
      : 0.0;
  if (wall_ms) *wall_ms = elapsed;
  if (gpu_profile_enabled()) {
    std::fprintf(stderr,
                 "[GPU] profile mozyme_sparse_fock_run one=%d pair=%d pair4x1=%d point=%d point_dipole=%d point_monopole=%d ms=%.3f\n",
                 plan->one_count, plan->pair_count,
                 plan->pair4_count, plan->point_count,
                 plan->point_dipole_count, plan->point_monopole_count,
                 elapsed);
  }
  return 0;
}

extern "C" int mopac_cuda_mozyme_sparse_fock_run_device_plan_guarded(
                                                         int plan_id,
                                                         int mpack,
                                                         const double *ptot_dev,
                                                         const double *qe_dev,
                                                         double *f_dev,
                                                         const int *guard_ints,
                                                         int guard_slot,
                                                         int guard_continue,
                                                         double *wall_ms) {
  return mozyme_sparse_fock_run_device_plan_guarded_impl(
      plan_id, mpack, ptot_dev, qe_dev, f_dev, guard_ints, guard_slot,
      guard_continue, false, 1, wall_ms);
}

extern "C" int mopac_cuda_mozyme_sparse_fock_run_device_plan_guarded_resident(
                                                         int plan_id,
                                                         int mpack,
                                                         const double *ptot_dev,
                                                         const double *qe_dev,
                                                         double *f_dev,
                                                         const int *guard_ints,
                                                         int guard_slot,
                                                         int guard_continue,
                                                         int synchronize,
                                                         double *wall_ms) {
  return mozyme_sparse_fock_run_device_plan_guarded_impl(
      plan_id, mpack, ptot_dev, qe_dev, f_dev, guard_ints, guard_slot,
      guard_continue, true, synchronize, wall_ms);
}

extern "C" int mopac_cuda_mozyme_sparse_fock_run_device_plan(
                                                         int plan_id,
                                                         int mpack,
                                                         const double *ptot_dev,
                                                         const double *qe_dev,
                                                         double *f_dev,
                                                         double *wall_ms) {
  return mopac_cuda_mozyme_sparse_fock_run_device_plan_guarded(
      plan_id, mpack, ptot_dev, qe_dev, f_dev, nullptr, -1, 0, wall_ms);
}

extern "C" int mopac_cuda_mozyme_sparse_fock_run_device(int mpack,
                                                         const double *ptot_dev,
                                                         const double *qe_dev,
                                                         double *f_dev,
                                                         double *wall_ms) {
  return mopac_cuda_mozyme_sparse_fock_run_device_plan(
      kMozymeSparseFockPlanDefault, mpack, ptot_dev, qe_dev, f_dev, wall_ms);
}

static int g_mozyme_f2_flag = -1;
static inline bool mozyme_f2_enabled() {
  if (g_mozyme_f2_flag < 0) {
    const char* env = std::getenv("MOPAC_MOZYME_F2_GPU");
    if (env && *env) {
      g_mozyme_f2_flag = env_truthy_ci(env) ? 1 : 0;
    } else {
      // Legacy MOZYME Fock wrappers are opt-in only; production uses CPU until
      // the GPU-first batched MOZYME path replaces this fine-grained wrapper.
      g_mozyme_f2_flag = 0;
    }
  }
  return g_mozyme_f2_flag != 0;
}

__global__ void mozyme_fock2_kernel(int iab, int jba,
                                     int n_ij, int n_kl,
                                     const double *pii,
                                     const double *pjj,
                                     const double *pij,
                                     const double *wj,
                                     const double *wk,
                                     double *out_fii,
                                     double *out_fjj,
                                     double *out_fij,
                                     int flag_diagonal) {
  int total = n_ij * n_kl;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int ij_idx = idx / n_kl;
  int kl_idx = idx % n_kl;

  int i0 = 0, j0 = 0;
  unpack_pair(ij_idx, i0, j0);
  int k0 = 0, l0 = 0;
  unpack_pair(kl_idx, k0, l0);

  double aa = (i0 == j0) ? 1.0 : 2.0;
  double bb = (k0 == l0) ? 1.0 : 2.0;
  double coul = wj[idx];

  atomicAdd_double(out_fii + ij_idx, bb * coul * pjj[kl_idx]);

  if (!flag_diagonal) {
    atomicAdd_double(out_fjj + kl_idx, aa * coul * pii[ij_idx]);

    double exch = wk[idx] * aa * bb * 0.125;
    int ik = i0 * jba + k0;
    int il = i0 * jba + l0;
    int jk = j0 * jba + k0;
    int jl = j0 * jba + l0;
    atomicAdd_double(out_fij + ik, -exch * pij[jl]);
    atomicAdd_double(out_fij + il, -exch * pij[jk]);
    atomicAdd_double(out_fij + jk, -exch * pij[il]);
    atomicAdd_double(out_fij + jl, -exch * pij[ik]);
  }
}

__global__ void mozyme_dfock2_kernel(int iab, int jba,
                                      int n_ij, int n_kl,
                                      const double *pii,
                                      const double *pjj,
                                      const double *pij,
                                      const double *wj,
                                      const double *wk,
                                      double *out_fii,
                                      double *out_fjj,
                                      double *out_fij,
                                      int flag_diagonal) {
  int total = n_ij * n_kl;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int ij_idx = idx / n_kl;
  int kl_idx = idx % n_kl;

  int i0 = 0, j0 = 0;
  unpack_pair(ij_idx, i0, j0);
  int k0 = 0, l0 = 0;
  unpack_pair(kl_idx, k0, l0);

  double aa = (i0 == j0) ? 1.0 : 2.0;
  double bb = (k0 == l0) ? 1.0 : 2.0;
  double coul = wj[idx];

  double val = bb * coul * pjj[kl_idx];
  atomicAdd_double(out_fii + ij_idx, val);

  if (!flag_diagonal) {
    double val_j = aa * coul * pii[ij_idx];
    atomicAdd_double(out_fjj + kl_idx, val_j);

    double exch = wk[idx] * aa * bb * 0.125;
    int ik = i0 * jba + k0;
    int il = i0 * jba + l0;
    int jk = j0 * jba + k0;
    int jl = j0 * jba + l0;
    atomicAdd_double(out_fij + ik, -exch * pij[jl]);
    atomicAdd_double(out_fij + il, -exch * pij[jk]);
    atomicAdd_double(out_fij + jk, -exch * pij[il]);
    atomicAdd_double(out_fij + jl, -exch * pij[ik]);
  }
}

extern "C" int mopac_cuda_mozyme_fock2(int iab, int jba,
                                        bool diagonal,
                                        const double *pii,
                                        const double *pjj,
                                        const double *pij,
                                        double *fii,
                                        double *fjj,
                                        double *fij,
                                        const double *wj,
                                        const double *wk) {
  if (!mozyme_f2_enabled()) return 1;
  if (iab <= 0 || jba <= 0 || !pii || !pjj || !pij || !fii || !fjj || !fij || !wj || !wk) return 1;
  int n_ij = iab * (iab + 1) / 2;
  int n_kl = jba * (jba + 1) / 2;
  int total = n_ij * n_kl;
  if (total <= 0) return 1;

  cudaStream_t s = g_stream ? g_stream : 0;
  size_t bytes_pii = sizeof(double) * (size_t)n_ij;
  size_t bytes_pjj = sizeof(double) * (size_t)n_kl;
  size_t bytes_pij = sizeof(double) * (size_t)iab * (size_t)jba;
  size_t bytes_w = sizeof(double) * (size_t)total;

  g_mz_fock2_pii.ensure(bytes_pii);
  g_mz_fock2_pjj.ensure(bytes_pjj);
  g_mz_fock2_pij.ensure(bytes_pij);
  g_mz_fock2_wj.ensure(bytes_w);
  g_mz_fock2_wk.ensure(bytes_w);
  g_mz_fock2_fii.ensure(bytes_pii);
  g_mz_fock2_fjj.ensure(bytes_pjj);
  g_mz_fock2_fij.ensure(bytes_pij);
  h_mz_fock2_fii.ensure(bytes_pii);
  h_mz_fock2_fjj.ensure(bytes_pjj);
  h_mz_fock2_fij.ensure(bytes_pij);

  if (cudaMemcpyAsync(g_mz_fock2_pii.ptr, pii, bytes_pii, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_pjj.ptr, pjj, bytes_pjj, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_pij.ptr, pij, bytes_pij, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_wj.ptr, wj, bytes_w, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_wk.ptr, wk, bytes_w, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;

  cudaMemsetAsync(g_mz_fock2_fii.ptr, 0, bytes_pii, s);
  cudaMemsetAsync(g_mz_fock2_fjj.ptr, 0, bytes_pjj, s);
  cudaMemsetAsync(g_mz_fock2_fij.ptr, 0, bytes_pij, s);

  int threads = 128;
  int blocks = (total + threads - 1) / threads;
  mozyme_fock2_kernel<<<blocks, threads, 0, s>>>(iab, jba, n_ij, n_kl,
                                                 g_mz_fock2_pii.ptr,
                                                 g_mz_fock2_pjj.ptr,
                                                 g_mz_fock2_pij.ptr,
                                                 g_mz_fock2_wj.ptr,
                                                 g_mz_fock2_wk.ptr,
                                                 g_mz_fock2_fii.ptr,
                                                 g_mz_fock2_fjj.ptr,
                                                 g_mz_fock2_fij.ptr,
                                                 diagonal ? 1 : 0);
  if (cudaPeekAtLastError() != cudaSuccess) return 3;

  if (cudaMemcpyAsync(h_mz_fock2_fii.ptr, g_mz_fock2_fii.ptr, bytes_pii, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  if (!diagonal) {
    if (cudaMemcpyAsync(h_mz_fock2_fjj.ptr, g_mz_fock2_fjj.ptr, bytes_pjj, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
    if (cudaMemcpyAsync(h_mz_fock2_fij.ptr, g_mz_fock2_fij.ptr, bytes_pij, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  }
  if (cudaStreamSynchronize(s) != cudaSuccess) return 2;

  for (int i = 0; i < n_ij; ++i) fii[i] += h_mz_fock2_fii.ptr[i];
  if (!diagonal) {
    for (int i = 0; i < n_kl; ++i) fjj[i] += h_mz_fock2_fjj.ptr[i];
    for (int i = 0; i < iab * jba; ++i) fij[i] += h_mz_fock2_fij.ptr[i];
  }
  return 0;
}

extern "C" int mopac_cuda_mozyme_dfock2(int iab, int jba,
                                        bool diagonal,
                                        const double *pii,
                                        const double *pjj,
                                        const double *pij,
                                        double *dfii,
                                        double *dfjj,
                                        double *dfij,
                                        const double *wj,
                                        const double *wk) {
  if (!mozyme_f2_enabled()) return 1;
  if (iab <= 0 || jba <= 0 || !pii || !pjj || !pij || !dfii || !dfjj || !dfij || !wj || !wk) return 1;
  int n_ij = iab * (iab + 1) / 2;
  int n_kl = jba * (jba + 1) / 2;
  int total = n_ij * n_kl;
  if (total <= 0) return 1;

  cudaStream_t s = g_stream ? g_stream : 0;
  size_t bytes_pii = sizeof(double) * (size_t)n_ij;
  size_t bytes_pjj = sizeof(double) * (size_t)n_kl;
  size_t bytes_pij = sizeof(double) * (size_t)iab * (size_t)jba;
  size_t bytes_w = sizeof(double) * (size_t)total;

  g_mz_fock2_pii.ensure(bytes_pii);
  g_mz_fock2_pjj.ensure(bytes_pjj);
  g_mz_fock2_pij.ensure(bytes_pij);
  g_mz_fock2_wj.ensure(bytes_w);
  g_mz_fock2_wk.ensure(bytes_w);
  g_mz_dfock2_fii.ensure(bytes_pii);
  g_mz_dfock2_fjj.ensure(bytes_pjj);
  g_mz_dfock2_fij.ensure(bytes_pij);
  h_mz_dfock2_fii.ensure(bytes_pii);
  h_mz_dfock2_fjj.ensure(bytes_pjj);
  h_mz_dfock2_fij.ensure(bytes_pij);

  if (cudaMemcpyAsync(g_mz_fock2_pii.ptr, pii, bytes_pii, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_pjj.ptr, pjj, bytes_pjj, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_pij.ptr, pij, bytes_pij, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_wj.ptr, wj, bytes_w, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;
  if (cudaMemcpyAsync(g_mz_fock2_wk.ptr, wk, bytes_w, cudaMemcpyHostToDevice, s) != cudaSuccess) return 2;

  cudaMemsetAsync(g_mz_dfock2_fii.ptr, 0, bytes_pii, s);
  cudaMemsetAsync(g_mz_dfock2_fjj.ptr, 0, bytes_pjj, s);
  cudaMemsetAsync(g_mz_dfock2_fij.ptr, 0, bytes_pij, s);

  int threads = 128;
  int blocks = (total + threads - 1) / threads;
  mozyme_dfock2_kernel<<<blocks, threads, 0, s>>>(iab, jba, n_ij, n_kl,
                                                 g_mz_fock2_pii.ptr,
                                                 g_mz_fock2_pjj.ptr,
                                                 g_mz_fock2_pij.ptr,
                                                 g_mz_fock2_wj.ptr,
                                                 g_mz_fock2_wk.ptr,
                                                 g_mz_dfock2_fii.ptr,
                                                 g_mz_dfock2_fjj.ptr,
                                                 g_mz_dfock2_fij.ptr,
                                                 diagonal ? 1 : 0);
  if (cudaPeekAtLastError() != cudaSuccess) return 3;

  if (cudaMemcpyAsync(h_mz_dfock2_fii.ptr, g_mz_dfock2_fii.ptr, bytes_pii, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  if (!diagonal) {
    if (cudaMemcpyAsync(h_mz_dfock2_fjj.ptr, g_mz_dfock2_fjj.ptr, bytes_pjj, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
    if (cudaMemcpyAsync(h_mz_dfock2_fij.ptr, g_mz_dfock2_fij.ptr, bytes_pij, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  }
  if (cudaStreamSynchronize(s) != cudaSuccess) return 2;

  for (int i = 0; i < n_ij; ++i) dfii[i] += h_mz_dfock2_fii.ptr[i];
  if (!diagonal) {
    for (int i = 0; i < n_kl; ++i) dfjj[i] += h_mz_dfock2_fjj.ptr[i];
    for (int i = 0; i < iab * jba; ++i) dfij[i] += h_mz_dfock2_fij.ptr[i];
  }
  return 0;
}

// Device kernels for outer product updates
__global__ void outer_update_rows(double *Csub, int rows, int ncols,
                                  const double *a, const double *b,
                                  double alpha, double beta, int row_offset) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total = rows * ncols;
  if (tid >= total) return;
  int r = tid % rows;
  int c = tid / rows;
  double val = alpha * a[row_offset + r] * b[c];
  double old = Csub[(size_t)c * (size_t)rows + r];
  Csub[(size_t)c * (size_t)rows + r] = val + beta * old;
}

void call_gemm_cublas_2gpu(char tra, char trb,
                           int m, int n, int k,
                           double alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           double beta,
                           double *C, int ldc) {
  int dev_count = 0;
  cudaGetDeviceCount(&dev_count);
  if (dev_count < 2 || g_pair_dev0 >= dev_count || g_pair_dev1 >= dev_count ||
      k != 1 || !(tra=='N'||tra=='n') || !(trb=='T'||trb=='t')) {
    call_gemm_cublas(tra, trb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }
  char nv_name[64];
  const char* nv_ptr = nullptr;
  if (gpu_profile_enabled()) {
    std::snprintf(nv_name, sizeof(nv_name), "GEMM-2GPU %dx%dx%d", m, n, k);
    nv_ptr = nv_name;
  }
  NvtxRange nv_scope(nv_ptr, 0xFF9467BD);
  ScopedBlasProfile prof_scope(&g_prof_gemm_pair, 2.0 * (double)m * (double)n * (double)k);
  ensure_pair_streams();
  if (w_verbose) {
    std::fprintf(stderr, "[GPU] DGEMM %dx%dx%d: 2-GPU outer split (%d,%d)\n",
                 m, n, k, g_pair_dev0, g_pair_dev1);
  }
  int n0 = m / 2;
  int n1 = m - n0;
  // Device allocations and copies with caching per device
  double *d_a0=nullptr, *d_b0=nullptr, *d_c0=nullptr;
  double *d_a1=nullptr, *d_b1=nullptr, *d_c1=nullptr;
  // Device 0
  cudaSetDevice(g_pair_dev0);
  g2_gemm_a0.ensure(sizeof(double) * (size_t)m);
  g2_gemm_b0.ensure(sizeof(double) * (size_t)n);
  g2_gemm_c0.ensure(sizeof(double) * (size_t)n0 * (size_t)ldc);
  d_a0 = g2_gemm_a0.ptr; d_b0 = g2_gemm_b0.ptr; d_c0 = g2_gemm_c0.ptr;
  size_t bytesAm = sizeof(double) * (size_t)m;
  size_t bytesBn = sizeof(double) * (size_t)n;
  size_t bytesCfull = sizeof(double) * (size_t)ldc * (size_t)n;
  h2_gemm_A.ensure(bytesAm);
  h2_gemm_B.ensure(bytesBn);
  h2_gemm_C.ensure(bytesCfull);
  std::memcpy(h2_gemm_A.ptr, A, bytesAm);
  std::memcpy(h2_gemm_B.ptr, B, bytesBn);
  std::memcpy(h2_gemm_C.ptr, C, bytesCfull);
  cudaMemcpyAsync(d_a0, h2_gemm_A.ptr, bytesAm, cudaMemcpyHostToDevice, g_stream0);
  cudaMemcpyAsync(d_b0, h2_gemm_B.ptr, bytesBn, cudaMemcpyHostToDevice, g_stream0);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(d_c0 + (size_t)col * (size_t)n0,
                    h2_gemm_C.ptr + (size_t)col * (size_t)ldc,
                    sizeof(double) * (size_t)n0,
                    cudaMemcpyHostToDevice, g_stream0);
  }
  // Device 1
  cudaSetDevice(g_pair_dev1);
  g2_gemm_a1.ensure(sizeof(double) * (size_t)m);
  g2_gemm_b1.ensure(sizeof(double) * (size_t)n);
  g2_gemm_c1.ensure(sizeof(double) * (size_t)n1 * (size_t)ldc);
  d_a1 = g2_gemm_a1.ptr; d_b1 = g2_gemm_b1.ptr; d_c1 = g2_gemm_c1.ptr;
  cudaMemcpyAsync(d_a1, h2_gemm_A.ptr, bytesAm, cudaMemcpyHostToDevice, g_stream1);
  cudaMemcpyAsync(d_b1, h2_gemm_B.ptr, bytesBn, cudaMemcpyHostToDevice, g_stream1);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(d_c1 + (size_t)col * (size_t)n1,
                    h2_gemm_C.ptr + (size_t)col * (size_t)ldc + (size_t)n0,
                    sizeof(double) * (size_t)n1,
                    cudaMemcpyHostToDevice, g_stream1);
  }

  // Launch kernels
  cudaSetDevice(g_pair_dev0);
  {
    int rows = n0;
    int total = rows * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    outer_update_rows<<<grid, block, 0, g_stream0>>>(d_c0, rows, n, d_a0, d_b0, alpha, beta, 0);
  }
  cudaSetDevice(g_pair_dev1);
  {
    int rows = n1;
    int total = rows * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    outer_update_rows<<<grid, block, 0, g_stream1>>>(d_c1, rows, n, d_a1, d_b1, alpha, beta, n0);
  }

  // Sync and copy back row slices
  cudaSetDevice(g_pair_dev0); cudaStreamSynchronize(g_stream0);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(h2_gemm_C.ptr + (size_t)col * (size_t)ldc,
               d_c0 + (size_t)col * (size_t)n0,
               sizeof(double) * (size_t)n0,
               cudaMemcpyDeviceToHost, g_stream0);
  }
  cudaSetDevice(g_pair_dev1); cudaStreamSynchronize(g_stream1);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(h2_gemm_C.ptr + (size_t)col * (size_t)ldc + (size_t)n0,
               d_c1 + (size_t)col * (size_t)n1,
               sizeof(double) * (size_t)n1,
               cudaMemcpyDeviceToHost, g_stream1);
  }
  cudaSetDevice(g_pair_dev0); cudaStreamSynchronize(g_stream0);
  cudaSetDevice(g_pair_dev1); cudaStreamSynchronize(g_stream1);

  // No frees here; cached buffers are released at process cleanup
  // Copy back to user output
  std::memcpy(C, h2_gemm_C.ptr, bytesCfull);
}

// 2-GPU outer product for SYRK with k==1, tra=='N': C[nxn] += alpha*v*v^T + beta*C
void call_syrk_cublas_2gpu(char uplo, char tra,
                           int n, int k,
                           double alpha,
                           const double *A, int lda,
                           double beta,
                           double *C, int ldc) {
  int dev_count = 0;
  cudaGetDeviceCount(&dev_count);
  if (dev_count < 2 || g_pair_dev0 >= dev_count || g_pair_dev1 >= dev_count ||
      k != 1 || !(tra=='N'||tra=='n')) {
    call_syrk_cublas(uplo, tra, n, k, alpha, A, lda, beta, C, ldc);
    return;
  }
  char nv_name[64];
  const char* nv_ptr = nullptr;
  if (gpu_profile_enabled()) {
    std::snprintf(nv_name, sizeof(nv_name), "SYRK-2GPU n=%d k=%d", n, k);
    nv_ptr = nv_name;
  }
  NvtxRange nv_scope(nv_ptr, 0xFFE377C2);
  ScopedBlasProfile prof_scope(&g_prof_syrk_pair, 2.0 * (double)n * (double)n * (double)k);
  ensure_pair_streams();
  if (w_verbose) {
    std::fprintf(stderr, "[GPU] DSYRK n=%d k=%d: 2-GPU outer split (%d,%d)\n",
                 n, k, g_pair_dev0, g_pair_dev1);
  }
  int n0 = n / 2;
  int n1 = n - n0;
  // Copy full vector v to both devices and split C by rows (cached per device)
  double *d_v0=nullptr, *d_c0=nullptr;
  double *d_v1=nullptr, *d_c1=nullptr;
  // Device 0
  cudaSetDevice(g_pair_dev0);
  g2_syrk_v0.ensure(sizeof(double) * (size_t)n);
  g2_syrk_c0.ensure(sizeof(double) * (size_t)n0 * (size_t)ldc);
  d_v0 = g2_syrk_v0.ptr; d_c0 = g2_syrk_c0.ptr;
  size_t bytesAn = sizeof(double) * (size_t)n;
  size_t bytesCfull = sizeof(double) * (size_t)ldc * (size_t)n;
  h2_syrk_A.ensure(bytesAn);
  h2_syrk_C.ensure(bytesCfull);
  std::memcpy(h2_syrk_A.ptr, A, bytesAn);
  std::memcpy(h2_syrk_C.ptr, C, bytesCfull);
  cudaMemcpyAsync(d_v0, h2_syrk_A.ptr, bytesAn, cudaMemcpyHostToDevice, g_stream0);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(d_c0 + (size_t)col * (size_t)n0,
                    h2_syrk_C.ptr + (size_t)col * (size_t)ldc,
                    sizeof(double) * (size_t)n0,
                    cudaMemcpyHostToDevice, g_stream0);
  }
  // Device 1
  cudaSetDevice(g_pair_dev1);
  g2_syrk_v1.ensure(sizeof(double) * (size_t)n);
  g2_syrk_c1.ensure(sizeof(double) * (size_t)n1 * (size_t)ldc);
  d_v1 = g2_syrk_v1.ptr; d_c1 = g2_syrk_c1.ptr;
  cudaMemcpyAsync(d_v1, h2_syrk_A.ptr, bytesAn, cudaMemcpyHostToDevice, g_stream1);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(d_c1 + (size_t)col * (size_t)n1,
                    h2_syrk_C.ptr + (size_t)col * (size_t)ldc + (size_t)n0,
                    sizeof(double) * (size_t)n1,
                    cudaMemcpyHostToDevice, g_stream1);
  }

  // Launch outer product kernels per device
  cudaSetDevice(g_pair_dev0);
  {
    int rows = n0;
    int total = rows * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    outer_update_rows<<<grid, block, 0, g_stream0>>>(d_c0, rows, n, d_v0, d_v0, alpha, beta, 0);
  }
  cudaSetDevice(g_pair_dev1);
  {
    int rows = n1;
    int total = rows * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    outer_update_rows<<<grid, block, 0, g_stream1>>>(d_c1, rows, n, d_v1, d_v1, alpha, beta, n0);
  }

  // Sync and copy back row slices
  cudaSetDevice(g_pair_dev0); cudaStreamSynchronize(g_stream0);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(h2_syrk_C.ptr + (size_t)col * (size_t)ldc,
               d_c0 + (size_t)col * (size_t)n0,
               sizeof(double) * (size_t)n0,
               cudaMemcpyDeviceToHost, g_stream0);
  }
  cudaSetDevice(g_pair_dev1); cudaStreamSynchronize(g_stream1);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(h2_syrk_C.ptr + (size_t)col * (size_t)ldc + (size_t)n0,
               d_c1 + (size_t)col * (size_t)n1,
               sizeof(double) * (size_t)n1,
               cudaMemcpyDeviceToHost, g_stream1);
  }
  cudaSetDevice(g_pair_dev0); cudaStreamSynchronize(g_stream0);
  cudaSetDevice(g_pair_dev1); cudaStreamSynchronize(g_stream1);

  // No frees; cached buffers are released at process cleanup
  // Copy back to user output
std::memcpy(C, h2_syrk_C.ptr, bytesCfull);
}

// Multi-GPU GEMM via cuBLASXt (host pointers)
extern "C" void call_gemm_cublas_multi(char tra, char trb,
                           int m, int n, int k,
                           double alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           double beta,
                           double *C, int ldc) {
  ensure_w_verbose();
  create_handle_xt();
  if (!g_blasXt) {
    poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
    return;
  }
  cublasOperation_t opA = (tra == 'T' || tra == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = (trb == 'T' || trb == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto t0 = std::chrono::high_resolution_clock::now();
  cublasStatus_t status = cublasXtDgemm(g_blasXt, opA, opB, m, n, k,
                                        &alpha, A, lda, B, ldb, &beta, C, ldc);
  if (!report_cublas_error("cublasXtDgemm", status)) {
    poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
    return;
  }
  if (w_verbose) {
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double flops = 2.0 * (double)m * (double)n * (double)k;
    double gflops = flops / 1.0e9 / (ms/1000.0);
    std::fprintf(stderr, "[MGPU] DGEMM %dx%dx%d: %.3f ms, %.1f GF/s\n", m,n,k, ms, gflops);
  }
}

// Multi-GPU SYRK via cuBLASXt (host pointers)
extern "C" void call_syrk_cublas_multi(char uplo, char tra,
                           int n, int k,
                           double alpha,
                           const double *A, int lda,
                           double beta,
                           double *C, int ldc) {
  ensure_w_verbose();
  create_handle_xt();
  if (!g_blasXt) {
    poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
    return;
  }
  cublasFillMode_t U = (uplo == 'U' || uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t opA = (tra == 'T' || tra == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto t0 = std::chrono::high_resolution_clock::now();
  cublasStatus_t status = cublasXtDsyrk(g_blasXt, U, opA, n, k,
                                        &alpha, A, lda, &beta, C, ldc);
  if (!report_cublas_error("cublasXtDsyrk", status)) {
    poison_host_doubles(C, static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n));
    return;
  }
  if (w_verbose) {
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double flops = 2.0 * (double)n * (double)n * (double)k; // rough
    double gflops = flops / 1.0e9 / (ms/1000.0);
    std::fprintf(stderr, "[MGPU] DSYRK n=%d k=%d: %.3f ms, %.1f GF/s\n", n,k, ms, gflops);
  }
}
// Symmetric eigensolver (upper triangle) using cuSOLVER Dsyevd; A overwritten with eigenvectors
// Cached handles and workspaces for Dsyevd
static cusolverDnHandle_t g_solver = nullptr;
static DevBuf<double> g_dsyevd_A, g_dsyevd_W, g_dsyevd_work;
static DevBuf<int>    g_dsyevd_info;
static int g_dsyevd_lwork_cap = 0; // elements, not bytes
static bool g_have_device_eigvecs = false;
static int  g_device_eigvecs_n = 0;

void mopac_cuda_dsyevd(int n, double *A, int lda, double *W, int *info) {
  if (info) *info = -1000;
  if (!A || !W || !info || n <= 0 || lda < n) {
    return;
  }
  if (!g_solver && !report_cusolver_error("cusolverDnCreate", cusolverDnCreate(&g_solver))) {
    poison_host_doubles(A, static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
    poison_host_doubles(W, static_cast<std::size_t>(n));
    *info = -1001;
    return;
  }

  size_t bytesA = sizeof(double) * (size_t)lda * (size_t)n;
  size_t bytesW = sizeof(double) * (size_t)n;
  g_dsyevd_A.ensure(bytesA);
  g_dsyevd_W.ensure(bytesW);
  g_dsyevd_info.ensure(sizeof(int));
  double *d_A = g_dsyevd_A.ptr;
  double *d_W = g_dsyevd_W.ptr;
  int *d_info = g_dsyevd_info.ptr;
  if (!d_A || !d_W || !d_info) {
    poison_host_doubles(A, static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
    poison_host_doubles(W, static_cast<std::size_t>(n));
    *info = -1002;
    return;
  }

  static HostBuf<double> h_dsyevd_A, h_dsyevd_W;
  h_dsyevd_A.ensure(bytesA);
  h_dsyevd_W.ensure(bytesW);
  if (!h_dsyevd_A.ptr || !h_dsyevd_W.ptr) {
    poison_host_doubles(A, static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
    poison_host_doubles(W, static_cast<std::size_t>(n));
    *info = -1003;
    return;
  }
  std::memcpy(h_dsyevd_A.ptr, A, bytesA);
  bool ok = true;
  ok = report_cuda_error("DSYEVD copy A host->device",
                         cudaMemcpyAsync(d_A, h_dsyevd_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream)) && ok;

  int lwork = 0;
  ok = report_cusolver_error("cusolverDnSetStream", cusolverDnSetStream(g_solver, g_stream)) && ok;
  ok = report_cusolver_error("cusolverDnDsyevd_bufferSize",
                             cusolverDnDsyevd_bufferSize(g_solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                                         n, d_A, lda, d_W, &lwork)) && ok;
  if (!ok || lwork <= 0) {
    poison_host_doubles(A, static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
    poison_host_doubles(W, static_cast<std::size_t>(n));
    *info = -1004;
    return;
  }
  if (lwork > g_dsyevd_lwork_cap) {
    g_dsyevd_work.ensure(sizeof(double) * (size_t)lwork);
    g_dsyevd_lwork_cap = lwork;
  }
  double *d_work = g_dsyevd_work.ptr;
  if (!d_work) {
    poison_host_doubles(A, static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
    poison_host_doubles(W, static_cast<std::size_t>(n));
    *info = -1005;
    return;
  }

  ok = report_cusolver_error("cusolverDnDsyevd",
                             cusolverDnDsyevd(g_solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                              n, d_A, lda, d_W, d_work, lwork, d_info)) && ok;
  ok = report_cuda_error("DSYEVD copy A device->host",
                         cudaMemcpyAsync(h_dsyevd_A.ptr, d_A, bytesA, cudaMemcpyDeviceToHost, g_stream)) && ok;
  ok = report_cuda_error("DSYEVD copy W device->host",
                         cudaMemcpyAsync(h_dsyevd_W.ptr, d_W, bytesW, cudaMemcpyDeviceToHost, g_stream)) && ok;
  ok = report_cuda_error("DSYEVD copy info device->host",
                         cudaMemcpyAsync(info, d_info, sizeof(int), cudaMemcpyDeviceToHost, g_stream)) && ok;
  ok = report_cuda_error("DSYEVD stream synchronize", cudaStreamSynchronize(g_stream)) && ok;
  if (ok && *info == 0) {
    std::memcpy(A, h_dsyevd_A.ptr, bytesA);
    std::memcpy(W, h_dsyevd_W.ptr, bytesW);
  } else {
    if (ok) {
      std::fprintf(stderr, "[GPU ERROR] DSYEVD returned info=%d\n", *info);
    } else {
      *info = -1006;
    }
    poison_host_doubles(A, static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
    poison_host_doubles(W, static_cast<std::size_t>(n));
  }
}

// Variant: keep eigenvectors on device, return eigenvalues only
void mopac_cuda_dsyevd_keep(int n, double *A, int lda, double *W, int *info) {
  if (info) *info = -1100;
  if (!A || !W || !info || n <= 0 || lda < n) {
    return;
  }
  if (!g_solver && !report_cusolver_error("cusolverDnCreate keep", cusolverDnCreate(&g_solver))) {
    poison_host_doubles(W, static_cast<std::size_t>(n));
    g_have_device_eigvecs = false;
    *info = -1101;
    return;
  }

  size_t bytesA = sizeof(double) * (size_t)lda * (size_t)n;
  size_t bytesW = sizeof(double) * (size_t)n;
  g_dsyevd_A.ensure(bytesA);
  g_dsyevd_W.ensure(bytesW);
  g_dsyevd_info.ensure(sizeof(int));
  double *d_A = g_dsyevd_A.ptr;
  double *d_W = g_dsyevd_W.ptr;
  int *d_info = g_dsyevd_info.ptr;
  if (!d_A || !d_W || !d_info) {
    poison_host_doubles(W, static_cast<std::size_t>(n));
    g_have_device_eigvecs = false;
    *info = -1102;
    return;
  }

  static HostBuf<double> h_dsyevd_A, h_dsyevd_W;
  h_dsyevd_A.ensure(bytesA);
  h_dsyevd_W.ensure(bytesW);
  if (!h_dsyevd_A.ptr || !h_dsyevd_W.ptr) {
    poison_host_doubles(W, static_cast<std::size_t>(n));
    g_have_device_eigvecs = false;
    *info = -1103;
    return;
  }
  // Copy host-packed upper triangle (in A) that was unpacked by caller into full matrix; here A is ignored.
  // Caller should have already unpacked; we accept A as a full matrix buffer for simplicity.
  std::memcpy(h_dsyevd_A.ptr, A, bytesA);
  bool ok = true;
  ok = report_cuda_error("DSYEVD keep copy A host->device",
                         cudaMemcpyAsync(d_A, h_dsyevd_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream)) && ok;

  int lwork = 0;
  ok = report_cusolver_error("cusolverDnSetStream keep", cusolverDnSetStream(g_solver, g_stream)) && ok;
  ok = report_cusolver_error("cusolverDnDsyevd_bufferSize keep",
                             cusolverDnDsyevd_bufferSize(g_solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                                         n, d_A, lda, d_W, &lwork)) && ok;
  if (!ok || lwork <= 0) {
    poison_host_doubles(W, static_cast<std::size_t>(n));
    g_have_device_eigvecs = false;
    *info = -1104;
    return;
  }
  if (lwork > g_dsyevd_lwork_cap) {
    g_dsyevd_work.ensure(sizeof(double) * (size_t)lwork);
    g_dsyevd_lwork_cap = lwork;
  }
  double *d_work = g_dsyevd_work.ptr;
  if (!d_work) {
    poison_host_doubles(W, static_cast<std::size_t>(n));
    g_have_device_eigvecs = false;
    *info = -1105;
    return;
  }

  ok = report_cusolver_error("cusolverDnDsyevd keep",
                             cusolverDnDsyevd(g_solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                              n, d_A, lda, d_W, d_work, lwork, d_info)) && ok;
  ok = report_cuda_error("DSYEVD keep copy W device->host",
                         cudaMemcpyAsync(h_dsyevd_W.ptr, d_W, bytesW, cudaMemcpyDeviceToHost, g_stream)) && ok;
  ok = report_cuda_error("DSYEVD keep copy info device->host",
                         cudaMemcpyAsync(info, d_info, sizeof(int), cudaMemcpyDeviceToHost, g_stream)) && ok;
  ok = report_cuda_error("DSYEVD keep stream synchronize", cudaStreamSynchronize(g_stream)) && ok;
  if (ok && *info == 0) {
    std::memcpy(W, h_dsyevd_W.ptr, bytesW);
    g_have_device_eigvecs = true;
    g_device_eigvecs_n = n;
  } else {
    if (ok) {
      std::fprintf(stderr, "[GPU ERROR] DSYEVD keep returned info=%d\n", *info);
    } else {
      *info = -1106;
    }
    poison_host_doubles(W, static_cast<std::size_t>(n));
    g_have_device_eigvecs = false;
  }
}

// Build density C = alpha * V(:,1:ndubl) * V(:,1:ndubl)^T on device from last eigenvectors
void mopac_cuda_density_from_dev_syrk(int n, int ndubl, double alpha, double *C, int ldc) {
  if (!g_blas) create_handle();
  if (!g_have_device_eigvecs || n != g_device_eigvecs_n) {
    // Fallback: just zero C
    size_t bytesC = (size_t)ldc * (size_t)n * sizeof(double);
    std::memset(C, 0, bytesC);
    g_density_full_valid = false;
    invalidate_packed_density();
    return;
  }
  size_t bytesC = (size_t)ldc * (size_t)n * sizeof(double);
  g_density_full.ensure(bytesC);
  double *d_A = g_dsyevd_A.ptr; // eigenvectors on device
  double *d_C = g_density_full.ptr;
  double beta = 0.0;
  cublasDsyrk(g_blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, ndubl, &alpha, d_A, n, &beta, d_C, ldc);
  g_density_full_valid = true;
  g_density_full_n = n;
  g_density_full_ld = ldc;
  invalidate_packed_density();
  if (!resident_mode_enabled()) {
    cudaMemcpyAsync(C, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
  }
}

// Build full X = 2*sign*V(:,nl2:nu2)V(:,nl2:nu2)^T + frac*sign*V(:,nl1:nu1)V(:,nl1:nu1)^T
// Caller adds cst to the diagonal on host.
void mopac_cuda_density_from_dev_gemm(int n,
                                      int nl2, int nu2,
                                      int nl1, int nu1,
                                      double sign,
                                      double frac,
                                      double *C, int ldc) {
  if (!g_blas) create_handle();
  if (!g_have_device_eigvecs || n != g_device_eigvecs_n) {
    size_t bytesC = (size_t)ldc * (size_t)n * sizeof(double);
    std::memset(C, 0, bytesC);
    g_density_full_valid = false;
    invalidate_packed_density();
    return;
  }
  size_t bytesC = (size_t)ldc * (size_t)n * sizeof(double);
  g_density_full.ensure(bytesC);
  double *d_A = g_dsyevd_A.ptr; // eigenvectors on device
  double *d_C = g_density_full.ptr;
  // Zero C (beta=0 in first SYRK covers it)
  // First block: columns [nl2..nu2]
  int k1 = (nu2 >= nl2) ? (nu2 - nl2 + 1) : 0;
  if (k1 > 0) {
    double alpha1 = 2.0 * sign;
    double beta = 0.0;
    const double *d_block1 = d_A + (size_t)(nl2 - 1) * (size_t)n;
    cublasDsyrk(g_blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k1, &alpha1, d_block1, n, &beta, d_C, ldc);
  } else {
    // Initialize d_C to zero if first block absent
    cudaMemsetAsync(d_C, 0, bytesC, g_stream);
  }
  // Second block: columns [nl1..nu1]
  int k2 = (nu1 >= nl1) ? (nu1 - nl1 + 1) : 0;
  if (k2 > 0) {
    double alpha2 = frac * sign;
    double beta = 1.0;
    const double *d_block2 = d_A + (size_t)(nl1 - 1) * (size_t)n;
    cublasDsyrk(g_blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k2, &alpha2, d_block2, n, &beta, d_C, ldc);
  }
  g_density_full_valid = true;
  g_density_full_n = n;
  g_density_full_ld = ldc;
  invalidate_packed_density();
  if (!resident_mode_enabled()) {
    cudaMemcpyAsync(C, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
  }
}

// Fetch device-resident eigenvectors into host buffer A (ld=lda)
void mopac_cuda_fetch_eigenvectors(int n, double *A, int lda) {
  if (!g_have_device_eigvecs || n != g_device_eigvecs_n) {
    // Nothing to fetch; leave A unchanged
    return;
  }
  size_t bytesA = (size_t)lda * (size_t)n * sizeof(double);
  cudaMemcpyAsync(A, g_dsyevd_A.ptr, bytesA, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
}

// --- MOZYME rotation: GPU-assisted drot over two columns ---
__global__ void drot_cols_kernel(double *V, int n, int i_col, int j_col, double alpha, double beta) {
  int r = blockDim.x * blockIdx.x + threadIdx.x;
  if (r < n) {
    double vi = V[r + i_col * (size_t)n];
    double vj = V[r + j_col * (size_t)n];
    double vi_new = alpha * vi + beta * vj;
    double vj_new = alpha * vj - beta * vi;
    V[r + i_col * (size_t)n] = vi_new;
    V[r + j_col * (size_t)n] = vj_new;
  }
}

// (helper was unused; removed to silence warnings)

// Batched sequential rotations applied within one kernel launch
__global__ void drot_cols_batch_kernel(double *V, int n, int npairs,
                                       const int *i_cols, const int *j_cols,
                                       const double *alphas, const double *betas) {
  int r = blockDim.x * blockIdx.x + threadIdx.x;
  if (r >= n) return;
  for (int p = 0; p < npairs; ++p) {
    int ic = i_cols[p];
    int jc = j_cols[p];
    double alpha = alphas[p];
    double beta  = betas[p];
    double vi = V[r + ic * (size_t)n];
    double vj = V[r + jc * (size_t)n];
    double vi_new = alpha * vi + beta * vj;
    double vj_new = alpha * vj - beta * vi;
    V[r + ic * (size_t)n] = vi_new;
    V[r + jc * (size_t)n] = vj_new;
  }
}

// Cached buffers for single-GPU rotation
static DevBuf<double> g_rot_V;
static DevBuf<int>    g_rot_i, g_rot_j;
static DevBuf<double> g_rot_a, g_rot_b;

void call_rot_cuda_gpu(const double *fmo, const double *eig,
                       double *vector, const double *ci0, const double *ca0,
                       int nocc, int lumo, int n,
                       double bigeps, double tiny) {
  (void)ci0; (void)ca0; // unused for now
  auto fail = [&](const char *where) {
    std::fprintf(stderr, "[GPU ERROR] ROT single %s\n", where);
    const std::size_t count = (n > 0) ? static_cast<std::size_t>(n) * static_cast<std::size_t>(n) : 0u;
    poison_host_doubles(vector, count);
  };
  if (!fmo || !eig || !vector || nocc <= 0 || lumo <= 0 || n <= 0) {
    fail("bad arguments");
    return;
  }
  size_t bytesV = sizeof(double) * (size_t)n * (size_t)n;
  if (!g_rot_V.ensure(bytesV)) {
    fail("device allocation V");
    return;
  }
  double *d_V = g_rot_V.ptr;
  static HostBuf<double> h_rot_V;
  h_rot_V.ensure(bytesV);
  if (!d_V || !h_rot_V.ptr) {
    fail("host/device allocation V");
    return;
  }
  std::memcpy(h_rot_V.ptr, vector, bytesV);
  bool ok = report_cuda_error("ROT single copy V host->device",
                              cudaMemcpyAsync(d_V, h_rot_V.ptr, bytesV, cudaMemcpyHostToDevice, g_stream));

  // Walk pairs sequentially; batch to reduce kernel launches
  const int max_batch = 256;
  int   *h_i = (int*)malloc(sizeof(int) * max_batch);
  int   *h_j = (int*)malloc(sizeof(int) * max_batch);
  double *h_a = (double*)malloc(sizeof(double) * max_batch);
  double *h_b = (double*)malloc(sizeof(double) * max_batch);
  if (!h_i || !h_j || !h_a || !h_b) {
    if (h_i) free(h_i);
    if (h_j) free(h_j);
    if (h_a) free(h_a);
    if (h_b) free(h_b);
    fail("host batch allocation");
    return;
  }
  ok = g_rot_i.ensure(sizeof(int) * max_batch) && ok;
  ok = g_rot_j.ensure(sizeof(int) * max_batch) && ok;
  ok = g_rot_a.ensure(sizeof(double) * max_batch) && ok;
  ok = g_rot_b.ensure(sizeof(double) * max_batch) && ok;
  int   *d_i = g_rot_i.ptr, *d_j = g_rot_j.ptr;
  double *d_a = g_rot_a.ptr, *d_b = g_rot_b.ptr;
  if (!d_i || !d_j || !d_a || !d_b) ok = false;

  auto copy_and_launch = [&](int batch) -> bool {
    if (batch <= 0) return true;
    bool batch_ok = true;
    batch_ok = report_cuda_error("ROT single copy i host->device",
                                 cudaMemcpyAsync(d_i, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream)) && batch_ok;
    batch_ok = report_cuda_error("ROT single copy j host->device",
                                 cudaMemcpyAsync(d_j, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream)) && batch_ok;
    batch_ok = report_cuda_error("ROT single copy alpha host->device",
                                 cudaMemcpyAsync(d_a, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream)) && batch_ok;
    batch_ok = report_cuda_error("ROT single copy beta host->device",
                                 cudaMemcpyAsync(d_b, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream)) && batch_ok;
    if (!batch_ok) return false;
    int block = 256;
    int grid = (n + block - 1) / block;
    drot_cols_batch_kernel<<<grid, block, 0, g_stream>>>(d_V, n, batch, d_i, d_j, d_a, d_b);
    return report_cuda_error("ROT single kernel launch", cudaPeekAtLastError());
  };

  int ij = 0;
  for (int i = 0; i < nocc && ok; ++i) {
    int batch = 0;
    for (int j = lumo - 1; j < n && ok; ++j) {
      ij += 1;
      double x = fmo[ij - 1]; // Fortran 1-based to C 0-based
      if (fabs(x) < tiny) continue;
      double a = eig[i];
      double b = eig[j];
      double d = a - b;
      if (fabs(x / d) < bigeps) continue;
      double e = copysign(sqrt(4.0 * x * x + d * d), d);
      double alpha = sqrt(0.5 * (1.0 + d / e));
      double beta = -copysign(sqrt(1.0 - alpha * alpha), x);
      h_i[batch] = i;
      h_j[batch] = j;
      h_a[batch] = alpha;
      h_b[batch] = beta;
      batch++;
      if (batch == max_batch) {
        ok = copy_and_launch(batch) && ok;
        batch = 0;
      }
    }
    if (ok && batch > 0) ok = copy_and_launch(batch) && ok;
  }
  ok = report_cuda_error("ROT single stream synchronize", cudaStreamSynchronize(g_stream)) && ok;
  ok = report_cuda_error("ROT single copy V device->host",
                         cudaMemcpyAsync(h_rot_V.ptr, d_V, bytesV, cudaMemcpyDeviceToHost, g_stream)) && ok;
  ok = report_cuda_error("ROT single final stream synchronize", cudaStreamSynchronize(g_stream)) && ok;
  if (ok) {
    std::memcpy(vector, h_rot_V.ptr, bytesV);
  } else {
    fail("execution failed");
  }
  free(h_i);
  free(h_j);
  free(h_a);
  free(h_b);
}

// Strided versions for multi-GPU row-partitioned layout
__global__ void drot_cols_kernel_strided(double *V, int nloc, int ncols,
                                         int i_col, int j_col,
                                         double alpha, double beta) {
  int r = blockDim.x * blockIdx.x + threadIdx.x;
  if (r < nloc) {
    size_t stride = (size_t)nloc;
    double vi = V[r + (size_t)i_col * stride];
    double vj = V[r + (size_t)j_col * stride];
    double vi_new = alpha * vi + beta * vj;
    double vj_new = alpha * vj - beta * vi;
    V[r + (size_t)i_col * stride] = vi_new;
    V[r + (size_t)j_col * stride] = vj_new;
  }
}

__global__ void drot_cols_batch_kernel_strided(double *V, int nloc, int ncols, int npairs,
                                               const int *i_cols, const int *j_cols,
                                               const double *alphas, const double *betas) {
  int r = blockDim.x * blockIdx.x + threadIdx.x;
  if (r >= nloc) return;
  size_t stride = (size_t)nloc;
  for (int p = 0; p < npairs; ++p) {
    int ic = i_cols[p];
    int jc = j_cols[p];
    double alpha = alphas[p];
    double beta  = betas[p];
    double vi = V[r + (size_t)ic * stride];
    double vj = V[r + (size_t)jc * stride];
    double vi_new = alpha * vi + beta * vj;
    double vj_new = alpha * vj - beta * vi;
    V[r + (size_t)ic * stride] = vi_new;
    V[r + (size_t)jc * stride] = vj_new;
  }
}

void call_rot_cuda_2gpu_gpu(const double *fmo, const double *eig,
                            double *vector, const double *ci0, const double *ca0,
                            int nocc, int lumo, int n,
                            double bigeps, double tiny) {
  (void)ci0; (void)ca0; // unused for now
  auto fail = [&](const char *where) {
    std::fprintf(stderr, "[GPU ERROR] ROT 2-GPU %s\n", where);
    const std::size_t count = (n > 0) ? static_cast<std::size_t>(n) * static_cast<std::size_t>(n) : 0u;
    poison_host_doubles(vector, count);
  };
  if (!fmo || !eig || !vector || nocc <= 0 || lumo <= 0 || n <= 0) {
    fail("bad arguments");
    return;
  }
  int dev_count = 0;
  if (!report_cuda_error("ROT 2-GPU device count", cudaGetDeviceCount(&dev_count))) {
    fail("device query failed");
    return;
  }
  if (dev_count < 2) {
    // Fallback to single-GPU path if we don't have at least 2 devices
    call_rot_cuda_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny);
    return;
  }

  // Partition rows across two devices
  int n0 = n / 2;
  int n1 = n - n0;
  size_t bytes0 = (size_t)n0 * (size_t)n * sizeof(double);
  size_t bytes1 = (size_t)n1 * (size_t)n * sizeof(double);

  // Allocate device slices (cached per device)
  double *d_V0 = nullptr, *d_V1 = nullptr;
  int *d_i0 = nullptr, *d_j0 = nullptr, *d_i1 = nullptr, *d_j1 = nullptr;
  double *d_a0 = nullptr, *d_b0 = nullptr, *d_a1 = nullptr, *d_b1 = nullptr;

  // Use configured device pair
  int dev0 = g_pair_dev0;
  int dev1 = g_pair_dev1;
  static DevBuf<double> g2_rot_V0, g2_rot_V1;
  static DevBuf<int>    g2_rot_i0, g2_rot_j0, g2_rot_i1, g2_rot_j1;
  static DevBuf<double> g2_rot_a0, g2_rot_b0, g2_rot_a1, g2_rot_b1;
  ensure_pair_streams();
  bool ok = true;
  ok = report_cuda_error("ROT 2-GPU set device 0", cudaSetDevice(dev0)) && ok;
  ok = g2_rot_V0.ensure(bytes0) && ok;
  ok = g2_rot_i0.ensure(sizeof(int) * 256) && ok;
  ok = g2_rot_j0.ensure(sizeof(int) * 256) && ok;
  ok = g2_rot_a0.ensure(sizeof(double) * 256) && ok;
  ok = g2_rot_b0.ensure(sizeof(double) * 256) && ok;
  d_V0 = g2_rot_V0.ptr; d_i0 = g2_rot_i0.ptr; d_j0 = g2_rot_j0.ptr; d_a0 = g2_rot_a0.ptr; d_b0 = g2_rot_b0.ptr;
  // Pinned staging for full matrix
  size_t bytesV = sizeof(double) * (size_t)n * (size_t)n;
  h2_rot_V.ensure(bytesV);
  if (!d_V0 || !d_i0 || !d_j0 || !d_a0 || !d_b0 || !h2_rot_V.ptr) {
    fail("device 0 allocation");
    return;
  }
  std::memcpy(h2_rot_V.ptr, vector, bytesV);
  // Copy top slice rows [0..n0) on device 0 using its stream
  for (int col = 0; col < n; ++col) {
    const double *col_ptr = h2_rot_V.ptr + (size_t)col * (size_t)n;
    ok = report_cuda_error("ROT 2-GPU copy V0 host->device",
                           cudaMemcpyAsync(d_V0 + (size_t)col * (size_t)n0, col_ptr, sizeof(double) * n0,
                                           cudaMemcpyHostToDevice, g_stream0)) && ok;
  }

  ok = report_cuda_error("ROT 2-GPU set device 1", cudaSetDevice(dev1)) && ok;
  ok = g2_rot_V1.ensure(bytes1) && ok;
  ok = g2_rot_i1.ensure(sizeof(int) * 256) && ok;
  ok = g2_rot_j1.ensure(sizeof(int) * 256) && ok;
  ok = g2_rot_a1.ensure(sizeof(double) * 256) && ok;
  ok = g2_rot_b1.ensure(sizeof(double) * 256) && ok;
  d_V1 = g2_rot_V1.ptr; d_i1 = g2_rot_i1.ptr; d_j1 = g2_rot_j1.ptr; d_a1 = g2_rot_a1.ptr; d_b1 = g2_rot_b1.ptr;
  if (!d_V1 || !d_i1 || !d_j1 || !d_a1 || !d_b1) {
    fail("device 1 allocation");
    return;
  }
  // Copy bottom slice rows [n0..n)
  for (int col = 0; col < n; ++col) {
    const double *col_ptr = h2_rot_V.ptr + (size_t)col * (size_t)n + (size_t)n0;
    ok = report_cuda_error("ROT 2-GPU copy V1 host->device",
                           cudaMemcpyAsync(d_V1 + (size_t)col * (size_t)n1, col_ptr, sizeof(double) * n1,
                                           cudaMemcpyHostToDevice, g_stream1)) && ok;
  }

  // Host batching buffers
  const int max_batch = 256;
  int   *h_i = (int*)malloc(sizeof(int) * max_batch);
  int   *h_j = (int*)malloc(sizeof(int) * max_batch);
  double *h_a = (double*)malloc(sizeof(double) * max_batch);
  double *h_b = (double*)malloc(sizeof(double) * max_batch);
  if (!h_i || !h_j || !h_a || !h_b) {
    if (h_i) free(h_i);
    if (h_j) free(h_j);
    if (h_a) free(h_a);
    if (h_b) free(h_b);
    fail("host batch allocation");
    return;
  }

  auto launch_batch = [&](int batch) -> bool {
    if (batch <= 0) return true;
    bool batch_ok = true;
    batch_ok = report_cuda_error("ROT 2-GPU set device 0 launch", cudaSetDevice(dev0)) && batch_ok;
    batch_ok = report_cuda_error("ROT 2-GPU copy i0 host->device",
                                 cudaMemcpyAsync(d_i0, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream0)) && batch_ok;
    batch_ok = report_cuda_error("ROT 2-GPU copy j0 host->device",
                                 cudaMemcpyAsync(d_j0, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream0)) && batch_ok;
    batch_ok = report_cuda_error("ROT 2-GPU copy alpha0 host->device",
                                 cudaMemcpyAsync(d_a0, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream0)) && batch_ok;
    batch_ok = report_cuda_error("ROT 2-GPU copy beta0 host->device",
                                 cudaMemcpyAsync(d_b0, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream0)) && batch_ok;
    if (batch_ok) {
      int block = 256;
      int grid = (n0 + block - 1) / block;
      drot_cols_batch_kernel_strided<<<grid, block, 0, g_stream0>>>(d_V0, n0, n, batch, d_i0, d_j0, d_a0, d_b0);
      batch_ok = report_cuda_error("ROT 2-GPU kernel device 0", cudaPeekAtLastError()) && batch_ok;
    }

    batch_ok = report_cuda_error("ROT 2-GPU set device 1 launch", cudaSetDevice(dev1)) && batch_ok;
    batch_ok = report_cuda_error("ROT 2-GPU copy i1 host->device",
                                 cudaMemcpyAsync(d_i1, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream1)) && batch_ok;
    batch_ok = report_cuda_error("ROT 2-GPU copy j1 host->device",
                                 cudaMemcpyAsync(d_j1, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream1)) && batch_ok;
    batch_ok = report_cuda_error("ROT 2-GPU copy alpha1 host->device",
                                 cudaMemcpyAsync(d_a1, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream1)) && batch_ok;
    batch_ok = report_cuda_error("ROT 2-GPU copy beta1 host->device",
                                 cudaMemcpyAsync(d_b1, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream1)) && batch_ok;
    if (batch_ok) {
      int block = 256;
      int grid = (n1 + block - 1) / block;
      drot_cols_batch_kernel_strided<<<grid, block, 0, g_stream1>>>(d_V1, n1, n, batch, d_i1, d_j1, d_a1, d_b1);
      batch_ok = report_cuda_error("ROT 2-GPU kernel device 1", cudaPeekAtLastError()) && batch_ok;
    }
    return batch_ok;
  };

  int ij = 0;
  for (int i = 0; i < nocc && ok; ++i) {
    int batch = 0;
    for (int j = lumo - 1; j < n && ok; ++j) {
      ij += 1;
      double x = fmo[ij - 1];
      if (fabs(x) < tiny) continue;
      double a = eig[i];
      double b = eig[j];
      double d = a - b;
      if (fabs(x / d) < bigeps) continue;
      double e = copysign(sqrt(4.0 * x * x + d * d), d);
      double alpha = sqrt(0.5 * (1.0 + d / e));
      double beta = -copysign(sqrt(1.0 - alpha * alpha), x);
      h_i[batch] = i;
      h_j[batch] = j;
      h_a[batch] = alpha;
      h_b[batch] = beta;
      batch++;
      if (batch == max_batch) {
        ok = launch_batch(batch) && ok;
        batch = 0;
      }
    }
    if (ok && batch > 0) ok = launch_batch(batch) && ok;
  }

  // Synchronize both devices
  ok = report_cuda_error("ROT 2-GPU set device 0 sync", cudaSetDevice(dev0)) && ok;
  ok = report_cuda_error("ROT 2-GPU stream0 synchronize", cudaStreamSynchronize(g_stream0)) && ok;
  ok = report_cuda_error("ROT 2-GPU set device 1 sync", cudaSetDevice(dev1)) && ok;
  ok = report_cuda_error("ROT 2-GPU stream1 synchronize", cudaStreamSynchronize(g_stream1)) && ok;

  // Copy results back into pinned host matrix
  ok = report_cuda_error("ROT 2-GPU set device 0 copyback", cudaSetDevice(dev0)) && ok;
  for (int col = 0; col < n; ++col) {
    double *col_ptr = h2_rot_V.ptr + (size_t)col * (size_t)n;
    ok = report_cuda_error("ROT 2-GPU copy V0 device->host",
                           cudaMemcpyAsync(col_ptr, d_V0 + (size_t)col * (size_t)n0, sizeof(double) * n0,
                                           cudaMemcpyDeviceToHost, g_stream0)) && ok;
  }
  ok = report_cuda_error("ROT 2-GPU set device 1 copyback", cudaSetDevice(dev1)) && ok;
  for (int col = 0; col < n; ++col) {
    double *col_ptr = h2_rot_V.ptr + (size_t)col * (size_t)n + (size_t)n0;
    ok = report_cuda_error("ROT 2-GPU copy V1 device->host",
                           cudaMemcpyAsync(col_ptr, d_V1 + (size_t)col * (size_t)n1, sizeof(double) * n1,
                                           cudaMemcpyDeviceToHost, g_stream1)) && ok;
  }
  ok = report_cuda_error("ROT 2-GPU final set device 0", cudaSetDevice(dev0)) && ok;
  ok = report_cuda_error("ROT 2-GPU final stream0 synchronize", cudaStreamSynchronize(g_stream0)) && ok;
  ok = report_cuda_error("ROT 2-GPU final set device 1", cudaSetDevice(dev1)) && ok;
  ok = report_cuda_error("ROT 2-GPU final stream1 synchronize", cudaStreamSynchronize(g_stream1)) && ok;
  // Copy back staged matrix to user memory
  if (ok) {
    std::memcpy(vector, h2_rot_V.ptr, bytesV);
  } else {
    fail("execution failed");
  }

  // Cleanup host buffers only; device buffers are retained in cache
  free(h_i);
  free(h_j);
  free(h_a);
  free(h_b);
}

// Provide a single cleanup entry point for Fortran.
__global__ void pack_upper_kernel(const double *full, int ld, int n, double *packed) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t total = (size_t)n * (n + 1) / 2;
  if (idx >= total) return;
  double d = (double)idx;
  double col_d = floor((sqrt(8.0 * d + 1.0) - 1.0) * 0.5);
  int col = (int)col_d;
  size_t start = (size_t)col * (col + 1) / 2;
  int row = (int)(idx - start);
  packed[idx] = full[row + (size_t)col * (size_t)ld];
}

__global__ void add_diag_kernel(double *full, int ld, int n, double value) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n) {
    full[col + (size_t)col * (size_t)ld] += value;
  }
}

extern "C" void mopac_cuda_set_active_stream(void* stream_ptr) {
  cudaStream_t new_stream = static_cast<cudaStream_t>(stream_ptr);
  if (!new_stream) return;
  create_handle();
  if (g_stream_override_depth == 0) {
    g_stream_backup = g_stream;
  }
  g_stream_override_depth++;
  g_stream = new_stream;
  if (g_blas) cublasSetStream(g_blas, g_stream);
}

extern "C" void mopac_cuda_clear_active_stream(void) {
  if (g_stream_override_depth <= 0) return;
  g_stream_override_depth--;
  if (g_stream_override_depth == 0) {
    g_stream = g_stream_backup;
    if (g_blas) cublasSetStream(g_blas, g_stream);
    g_stream_backup = nullptr;
  }
}

extern "C" void mopac_cuda_destroy_resources() {
  static bool already = false;
  // Allow skipping destroy on some platforms where driver teardown is fragile
  const char* skip = std::getenv("MOPAC_SKIP_GPU_DESTROY");
  if (env_truthy_ci(skip)) return;
  if (already) return;
  already = true;
  if ((mg_profile_env_requested || w_verbose) && (mg_calls > 0 || mg_failures > 0)) {
    double avg_ms = (mg_calls > 0) ? (mg_total_ms / (double)mg_calls) : 0.0;
    double avg_dim = (mg_calls > 0) ? (mg_total_dim / (double)mg_calls) : 0.0;
    double avg_dev = (mg_calls > 0) ? (mg_total_devices / (double)mg_calls) : 0.0;
    std::fprintf(stderr,
                 "[MGPU] summary: calls=%lld failures=%lld avg_ms=%.3f total_ms=%.3f avg_dim=%.1f avg_devices=%.2f\n",
                 mg_calls, mg_failures, avg_ms, mg_total_ms, avg_dim, avg_dev);
  }
  if (gpu_profile_enabled()) {
    print_blas_profile("gemm", g_prof_gemm_single);
    print_blas_profile("gemm-2gpu", g_prof_gemm_pair);
    print_blas_profile("syrk", g_prof_syrk_single);
    print_blas_profile("syrk-2gpu", g_prof_syrk_pair);
    print_blas_profile("disp", g_prof_disp_eval);
  }
  // Try to quiesce all pending GPU work before releasing resources
  // This helps avoid tearing down streams/handles while async copies are in-flight.
  cudaDeviceSynchronize();
  mopac_cuda_hmtr_clear_streams();
  // BLAS handle and streams
  destroy_handle();
  // cuSOLVER handle
  if (g_solver) {
    cusolverDnDestroy(g_solver);
    g_solver = nullptr;
  }
  // Release cached device buffers
  g_gemm_A.release(); g_gemm_B.release(); g_gemm_C.release();
  g_syrk_A.release(); g_syrk_C.release();
  g_dsyevd_A.release(); g_dsyevd_W.release(); g_dsyevd_work.release(); g_dsyevd_info.release();
  g_rot_V.release(); g_rot_i.release(); g_rot_j.release(); g_rot_a.release(); g_rot_b.release();
  g_density_full.release();
  g_density_full_valid = false; g_density_full_n = 0; g_density_full_ld = 0;
  for (auto &slot : g_packed_density) {
    slot.buf.release();
  }
  invalidate_packed_density();
  g_fock_cache.buf.release();
  invalidate_fock_cache();
  g_lt_workspace.release();
  // Release cached pinned host buffers
  h_gemm_A.release(); h_gemm_B.release(); h_gemm_C.release();
  h_syrk_A.release(); h_syrk_C.release();
  // 2-GPU caches
  // DSYEVD stages are static locals; nothing to release here on purpose
  // 2-GPU caches
  g2_gemm_a0.release(); g2_gemm_b0.release(); g2_gemm_c0.release();
  g2_gemm_a1.release(); g2_gemm_b1.release(); g2_gemm_c1.release();
  g2_syrk_v0.release(); g2_syrk_c0.release(); g2_syrk_v1.release(); g2_syrk_c1.release();
  h2_gemm_A.release(); h2_gemm_B.release(); h2_gemm_C.release();
  h2_syrk_A.release(); h2_syrk_C.release();
  h2_rot_V.release();
  g_disp_sum2.release(); g_disp_sum3.release(); g_disp_r.release();
  g_disp_val.release(); g_disp_der.release();
  h_disp_val.release(); h_disp_der.release();
  g_mz_fock_ptot.release(); g_mz_fock_w.release(); g_mz_fock_out.release();
  h_mz_fock_out.release();
  g_mz_fock1_iab.release(); g_mz_fock1_ilim.release();
  g_mz_fock1_pair_off.release(); g_mz_fock1_w_off.release();
  g_mz_fock1_batch_ptot.release(); g_mz_fock1_batch_w.release(); g_mz_fock1_batch_out.release();
  g_mz_fock2_4x1_diag.release(); g_mz_fock2_4x1_light.release(); g_mz_fock2_4x1_cross.release();
  g_mz_fock2_4x1_wj.release(); g_mz_fock2_4x1_wk.release(); g_mz_fock2_4x1_out.release();
  g_mz_res_count_iorbs.release(); g_mz_res_count_kopt.release();
  g_mz_res_count_nijbo.release(); g_mz_res_count_out.release(); g_mz_res_count_fallback.release();
  g_mz_res_pack_nat.release(); g_mz_res_pack_jindex.release(); g_mz_res_pack_status.release();
  g_mz_res_pack_coord.release(); g_mz_res_pack_wj.release(); g_mz_res_pack_wk.release();
  g_mz_res_pack_am.release(); g_mz_res_pack_ad.release(); g_mz_res_pack_dd.release();
  g_mz_res_pack_aq.release(); g_mz_res_pack_qq.release(); g_mz_res_pack_tore.release();
  g_mz_res_pack_po.release(); g_mz_res_pack_ddp.release();
  g_mz_res_pack_direct_scratch.release(); g_mz_res_pack_iod.release();
  g_mz_par_calc.release(); g_mz_par_rows.release(); g_mz_par_bases.release();
  g_mz_par_task_ii.release(); g_mz_par_task_jj.release(); g_mz_par_task4_ii.release();
  g_mz_par_task4_jj.release(); g_mz_par_scratch.release();
  for (auto &plan : g_mz_res_plans) plan.release();
  g_mz_fock2_pii.release(); g_mz_fock2_pjj.release(); g_mz_fock2_pij.release();
  g_mz_fock2_wj.release(); g_mz_fock2_wk.release();
  g_mz_fock2_fii.release(); g_mz_fock2_fjj.release(); g_mz_fock2_fij.release();
  h_mz_fock2_fii.release(); h_mz_fock2_fjj.release(); h_mz_fock2_fij.release();
  g_mz_dfock2_fii.release(); g_mz_dfock2_fjj.release(); g_mz_dfock2_fij.release();
  h_mz_dfock2_fii.release(); h_mz_dfock2_fjj.release(); h_mz_dfock2_fij.release();
  g_resident_mode = -1;
  g_stream_backup = nullptr;
  g_stream_override_depth = 0;
  resident_grad_release_impl();
}

extern "C" void* mopac_cuda_get_fock_device_ptr(void) {
  if (!resident_mode_enabled()) return nullptr;
  if (!g_fock_cache.valid) return nullptr;
  return static_cast<void*>(g_fock_cache.buf.ptr);
}

extern "C" void* mopac_cuda_get_density_device_ptr(void) {
  if (!resident_mode_enabled()) return nullptr;
  if (!g_density_full_valid) return nullptr;
  return static_cast<void*>(g_density_full.ptr);
}

extern "C" bool mopac_cuda_fetch_fock(double *host_ptr, size_t linear) {
  if (!resident_mode_enabled()) return false;
  if (!g_fock_cache.valid) return false;
  if (!host_ptr) return false;
  if (g_fock_cache.len != linear) return false;
  cudaStream_t s = g_stream ? g_stream : 0;
  if (cudaMemcpyAsync(host_ptr, g_fock_cache.buf.ptr, sizeof(double)*linear, cudaMemcpyDeviceToHost, s) != cudaSuccess) return false;
  cudaStreamSynchronize(s);
  return true;
}


extern "C" bool mopac_cuda_fetch_density(double *host_ptr, int n, int ld) {
  if (!resident_mode_enabled()) return false;
  if (!g_density_full_valid) return false;
  if (!host_ptr) return false;
  if (n != g_density_full_n || ld != g_density_full_ld) return false;
  size_t bytes = sizeof(double) * (size_t)ld * (size_t)n;
  cudaStream_t s = g_stream ? g_stream : 0;
  if (cudaMemcpyAsync(host_ptr, g_density_full.ptr, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return false;
  cudaStreamSynchronize(s);
  return true;
}

extern "C" bool mopac_cuda_fetch_packed_density(double *host_ptr, size_t linear) {
  if (!resident_mode_enabled()) return false;
  if (!host_ptr) return false;
  PackedDensitySlot *slot = find_packed_slot(host_ptr, linear);
  if (!slot) return false;
  cudaStream_t s = g_stream ? g_stream : 0;
  if (cudaMemcpyAsync(host_ptr, slot->buf.ptr, sizeof(double)*linear, cudaMemcpyDeviceToHost, s) != cudaSuccess) return false;
  cudaStreamSynchronize(s);
  return true;
}

bool mopac_cuda_cart_gradient(int numat, int l123, const double *coord,
                              double *grad, const double *charges,
                              const void *near_pairs, int near_count,
                              const void *far_pairs, int far_count) {
  if (!coord || !grad || !charges) return false;
  if (l123 == 1) {
    const auto *near = static_cast<const GradPairPod*>(near_pairs);
    const auto *far  = static_cast<const GradPairPod*>(far_pairs);
    if (resident_grad_launch_impl(numat, l123, coord, grad, charges,
                                  near, near_count, far, far_count)) {
      return true;
    }
  }
  return mopac_gpu_cart_gradient_cpu(numat, l123, coord, grad, charges);
}



void mopac_cuda_clear_density_cache() {
  g_density_full_valid = false;
  g_density_full_n = 0;
  g_density_full_ld = 0;
  for (auto &slot : g_packed_density) {
    slot.buf.release();
    slot.valid = false;
    slot.len = 0;
    slot.stamp = 0;
    slot.host_ptr = nullptr;
  }
  invalidate_packed_density();
}

void mopac_cuda_density_add_diag(int n, double value) {
  if (!resident_mode_enabled()) return;
  if (!g_density_full_valid) return;
  if (n != g_density_full_n) return;
  if (value == 0.0) return;
  cudaStream_t s = g_stream ? g_stream : 0;
  int block = 256;
  int grid = (n + block - 1) / block;
  add_diag_kernel<<<grid, block, 0, s>>>(g_density_full.ptr, g_density_full_ld, n, value);
  invalidate_packed_density();
}

extern "C" void mopac_cuda_update_density_from_host(int linear, const double *packed_host) {
  if (!resident_mode_enabled()) return;
  if (!g_density_full_valid) return;
  if (!packed_host) return;
  int n = g_density_full_n;
  size_t expected = (size_t)n * (size_t)(n + 1) / 2;
  if (linear <= 0 || (size_t)linear != expected) {
    invalidate_packed_density();
    return;
  }
  size_t bytes = (size_t)linear * sizeof(double);
  double *packed_dev = nullptr;
  cudaStream_t s = g_stream ? g_stream : 0;
  if (cudaMalloc((void**)&packed_dev, bytes) != cudaSuccess) {
    invalidate_packed_density();
    return;
  }
  cudaError_t copy_status = cudaMemcpyAsync(packed_dev, packed_host, bytes, cudaMemcpyHostToDevice, s);
  if (copy_status != cudaSuccess) {
    cudaFree(packed_dev);
    invalidate_packed_density();
    return;
  }
  int total = n * n;
  int block = 256;
  int grid = (total + block - 1) / block;
  unpack_lower_to_full_kernel<<<grid, block, 0, s>>>(packed_dev, g_density_full.ptr, n);
  cudaError_t kernel_status = cudaGetLastError();
  cudaFree(packed_dev);
  if (kernel_status != cudaSuccess) {
    invalidate_packed_density();
    return;
  }
  invalidate_packed_density();
}

void mopac_cuda_register_packed_density(int linear, double *packed_host) {
  if (!resident_mode_enabled()) {
    invalidate_packed_density();
    return;
  }
  if (!g_density_full_valid) {
    invalidate_packed_density();
    return;
  }
  if (linear <= 0) {
    invalidate_packed_density();
    return;
  }
  size_t expected = (size_t)g_density_full_n * (g_density_full_n + 1) / 2;
  if ((size_t)linear != expected) {
    invalidate_packed_density();
    return;
  }
  size_t bytes = (size_t)linear * sizeof(double);
  PackedDensitySlot *slot = acquire_packed_slot(packed_host);
  slot->buf.ensure(bytes);
  cudaStream_t s = g_stream ? g_stream : 0;
  int block = 256;
  int grid = ((size_t)linear + block - 1) / block;
  pack_upper_kernel<<<grid, block, 0, s>>>(g_density_full.ptr, g_density_full_ld, g_density_full_n, slot->buf.ptr);
  cudaStreamSynchronize(s);
  slot->len = (size_t)linear;
  slot->host_ptr = packed_host;
  slot->valid = true;
  slot->stamp = ++g_packed_density_tick;
  if (resident_debug_enabled() && packed_host) {
    std::vector<double> host_copy(linear);
    if (cudaMemcpy(host_copy.data(), slot->buf.ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
      double max_diff = 0.0;
      double rms = 0.0;
      for (int i = 0; i < linear; ++i) {
        double diff = host_copy[i] - packed_host[i];
        if (std::abs(diff) > max_diff) max_diff = std::abs(diff);
        rms += diff * diff;
      }
      if (linear > 0) rms = std::sqrt(rms / (double)linear);
      std::printf("[GPU resident debug] density register max=% .5e rms=% .5e\n", max_diff, rms);
      if (max_diff > 1e-6) {
        int limit = std::min(linear, 5);
        std::printf("  sample host vs device:");
        for (int i = 0; i < limit; ++i) {
          std::printf(" (% .5e,% .5e)", packed_host[i], host_copy[i]);
        }
        std::printf("\n");
      }
      std::fflush(stdout);
    }
  }
}

bool mopac_cuda_density_copy_cached(double *dest, size_t len, const double *host_ptr) {
  if (!dest || !host_ptr) return false;
  if (!resident_mode_enabled()) return false;
  PackedDensitySlot *slot = find_packed_slot(host_ptr, len);
  if (!slot) return false;
  if (slot->host_ptr != host_ptr) return false;
  if (cudaMemcpy(dest, slot->buf.ptr, len * sizeof(double), cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
  return true;
}

void mopac_cuda_clear_fock_cache() {
  invalidate_fock_cache();
}

void mopac_cuda_set_resident_mode(int flag) {
  g_resident_mode = (flag > 0) ? 1 : 0;
  if (flag <= 0) {
    invalidate_packed_density();
    invalidate_fock_cache();
  }
}

int mopac_cuda_get_resident_mode() {
  return resident_mode_enabled() ? 1 : 0;
}

bool mopac_cuda_fock_copy_cached(double *dest, size_t len, const double *host_ptr) {
  if (!dest || !host_ptr) return false;
  if (!resident_mode_enabled()) return false;
  if (!g_fock_cache.valid) return false;
  if (g_fock_cache.host_ptr != host_ptr) return false;
  if (g_fock_cache.len != len) return false;
  if (cudaMemcpy(dest, g_fock_cache.buf.ptr, len * sizeof(double), cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
  return true;
}

void mopac_cuda_register_fock_device(int linear, double *host_ptr, const double *src_dev) {
  register_fock_cache(linear, host_ptr, src_dev);
}

} // extern "C"

// =============== Additional GPU Orthogonalization Helpers (Phase 2) ===============
// These helpers provide Cholesky + triangular-solve based orthogonalization primitives
// for future integration of fully GPU-resident SCF orthonormalization paths.

extern "C" {

// =================== cuSOLVERMg multi-GPU symmetric eigensolver (stub) ===================
// Fortran calls mopac_cusolvermg_dsyevd when ngpus>1, n>=threshold, and env enables MG.
// If cuSOLVERMg headers/support are not available, return a nonzero info to trigger fallback.
void mopac_cusolvermg_dsyevd(int n, double *A, int lda, double *W, int *info) {
#if !defined(HAVE_CUSOLVER_MG)
  if (info) *info = -777;
  (void)n; (void)A; (void)lda; (void)W;
  return;
#else
  char nv_name[64];
  const char* nv_ptr = nullptr;
  if (gpu_profile_enabled()) {
    std::snprintf(nv_name, sizeof(nv_name), "MG-DSYEVD n=%d", n);
    nv_ptr = nv_name;
  }
  NvtxRange nv_scope(nv_ptr, 0xFF17BECF);
  ensure_w_verbose();
  if (info) *info = -1;
  bool profile_enabled = mg_profile_enabled();
  bool want_log = (w_verbose || profile_enabled);
  int gx = 2, gy = 1, blksz = 256;
  {
    const char* g = std::getenv("MOPAC_EIG_MG_GRID");
    if (g && *g) {
      int a=0,b=0; if (std::sscanf(g, "%dx%d", &a, &b) == 2 && a>0 && b>0) { gx=a; gy=b; }
    }
    const char* bs = std::getenv("MOPAC_EIG_MG_BLKSIZE");
    if (bs && *bs) { int tmp = std::atoi(bs); if (tmp > 0) blksz = tmp; }
  }
  int devCount = 0;
  cudaGetDeviceCount(&devCount);
  if (devCount <= 0) {
    if (profile_enabled) mg_failures++;
    if (want_log) std::fprintf(stderr, "[MGPU] no CUDA devices detected; fallback to single-GPU DSYEVD\n");
    mopac_cuda_dsyevd(n, A, lda, W, info);
    return;
  }
  int need = gx * gy;
  if (need > devCount) {
    gx = std::max(1, std::min(devCount, gx));
    gy = std::max(1, devCount / gx);
    need = gx * gy;
  }

  int orig_dev = -1;
  cudaGetDevice(&orig_dev);

  cusolverMgHandle_t mh = nullptr;
  cudaLibMgGrid_t grid = nullptr;
  cudaLibMgMatrixDesc_t desc = nullptr;
  std::vector<int> devs(need);
  std::vector<double*> Adev(need, nullptr);
  std::vector<double*> Work(need, nullptr);
  auto cleanup = [&]() {
    for (int did = 0; did < need; ++did) {
      if (!devs.empty()) {
        int cur = -1;
        cudaGetDevice(&cur);
        cudaSetDevice(devs[did]);
        if (Work[did]) cudaFree(Work[did]);
        if (Adev[did]) cudaFree(Adev[did]);
        if (cur >= 0) cudaSetDevice(cur);
      }
    }
    if (desc) cudaLibMgDestroyMatrixDesc(desc);
    if (grid) cudaLibMgDestroyGrid(grid);
    if (mh) cusolverMgDestroy(mh);
    if (orig_dev >= 0) cudaSetDevice(orig_dev);
  };
  auto record_failure = [&]() {
    if (profile_enabled) mg_failures++;
  };

  cusolverStatus_t s = cusolverMgCreate(&mh);
  if (s != CUSOLVER_STATUS_SUCCESS || !mh) {
    if (want_log) std::fprintf(stderr, "[MGPU] cusolverMgCreate failed; fallback to single-GPU DSYEVD\n");
    if (profile_enabled) mg_failures++;
    mopac_cuda_dsyevd(n, A, lda, W, info);
    return;
  }

  for (int i = 0; i < need; ++i) devs[i] = i;

#if defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION >= 11000)
  s = cusolverMgDeviceSelect(mh, need, devs.data());
  if (s != CUSOLVER_STATUS_SUCCESS) {
    if (want_log) std::fprintf(stderr, "[MGPU] cusolverMgDeviceSelect failed; fallback to single-GPU DSYEVD\n");
    record_failure();
    cleanup();
    mopac_cuda_dsyevd(n, A, lda, W, info);
    return;
  }
#endif

  cudaError_t cerr = cudaLibMgCreateGrid(&grid, gx, gy, devs.data());
  if (cerr != cudaSuccess || !grid) {
    if (want_log) std::fprintf(stderr, "[MGPU] cudaLibMgCreateGrid failed; fallback to single-GPU DSYEVD\n");
    record_failure();
    cleanup();
    mopac_cuda_dsyevd(n, A, lda, W, info);
    return;
  }

  cerr = cudaLibMgCreateMatrixDesc(&desc,
                                   CUBLAS_FILL_MODE_UPPER,
                                   CUDA_R_64F,
                                   n, n,
                                   lda,
                                   blksz, blksz,
                                   grid);
  if (cerr != cudaSuccess || !desc) {
    if (want_log) std::fprintf(stderr, "[MGPU] cudaLibMgCreateMatrixDesc failed; fallback to single-GPU DSYEVD\n");
    record_failure();
    cleanup();
    mopac_cuda_dsyevd(n, A, lda, W, info);
    return;
  }

  // Allocate distributed buffers and determine local extents
  for (int did = 0; did < need; ++did) {
    int prow = did % gx;
    int pcol = did / gx;
    int64_t rows = 0, cols = 0;
    cudaError_t szerr = cudaLibMgGetLocalMatrixSize(n, n, blksz, blksz, gx, gy, prow, pcol, &rows, &cols);
    if (szerr != cudaSuccess) {
      if (want_log) std::fprintf(stderr, "[MGPU] cudaLibMgGetLocalMatrixSize failed for device %d; fallback to single-GPU DSYEVD\n", devs[did]);
      record_failure();
      cleanup();
      mopac_cuda_dsyevd(n, A, lda, W, info);
      return;
    }
    size_t ld_local = (rows > 0) ? static_cast<size_t>(rows) : 1u;
    size_t cd_local = (cols > 0) ? static_cast<size_t>(cols) : 1u;
    size_t bytes = ld_local * cd_local * sizeof(double);
    if (bytes == 0) {
      bytes = sizeof(double);
    }
    int cur = -1;
    cudaGetDevice(&cur);
    cudaSetDevice(devs[did]);
    cerr = cudaMalloc(reinterpret_cast<void**>(&Adev[did]), bytes);
    if (cur >= 0) cudaSetDevice(cur);
    if (cerr != cudaSuccess) {
      if (want_log) std::fprintf(stderr, "[MGPU] cudaMalloc tile failed on device %d; fallback to single-GPU DSYEVD\n", devs[did]);
      record_failure();
      cleanup();
      mopac_cuda_dsyevd(n, A, lda, W, info);
      return;
    }
  }

  const int64_t IA = 1;
  const int64_t JA = 1;
  cusolverStatus_t st_copy_h2d = cusolverMgMemcpyH2D(mh,
                                                     reinterpret_cast<void* const*>(Adev.data()),
                                                     IA, JA,
                                                     desc,
                                                     A,
                                                     lda);
  if (st_copy_h2d != CUSOLVER_STATUS_SUCCESS) {
    if (want_log) std::fprintf(stderr, "[MGPU] cusolverMgMemcpyH2D failed; fallback to single-GPU DSYEVD\n");
    record_failure();
    cleanup();
    mopac_cuda_dsyevd(n, A, lda, W, info);
    return;
  }

  int64_t lwork = 0;
  s = cusolverMgSyevd_bufferSize(mh,
                                 CUSOLVER_EIG_MODE_VECTOR,
                                 CUBLAS_FILL_MODE_UPPER,
                                 n,
                                 reinterpret_cast<double* const*>(Adev.data()),
                                 IA,
                                 JA,
                                 desc,
                                 W,
                                 &lwork);
  if (s != CUSOLVER_STATUS_SUCCESS || lwork <= 0) {
    if (want_log) std::fprintf(stderr, "[MGPU] cusolverMgSyevd_bufferSize failed; fallback to single-GPU DSYEVD\n");
    record_failure();
    cleanup();
    mopac_cuda_dsyevd(n, A, lda, W, info);
    return;
  }

  for (int did = 0; did < need; ++did) {
    int cur = -1;
    cudaGetDevice(&cur);
    cudaSetDevice(devs[did]);
    size_t work_bytes = sizeof(double) * static_cast<size_t>(std::max<int64_t>(lwork, 1));
    cerr = cudaMalloc(reinterpret_cast<void**>(&Work[did]), work_bytes);
    if (cur >= 0) cudaSetDevice(cur);
    if (cerr != cudaSuccess) {
      if (want_log) std::fprintf(stderr, "[MGPU] cudaMalloc workspace failed on device %d; fallback to single-GPU DSYEVD\n", devs[did]);
      record_failure();
      cleanup();
      mopac_cuda_dsyevd(n, A, lda, W, info);
      return;
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  int linfo = 0;
  s = cusolverMgSyevd(mh,
                      CUSOLVER_EIG_MODE_VECTOR,
                      CUBLAS_FILL_MODE_UPPER,
                      n,
                      reinterpret_cast<double* const*>(Adev.data()),
                      IA,
                      JA,
                      desc,
                      W,
                      reinterpret_cast<double* const*>(Work.data()),
                      lwork,
                      &linfo);

  if (s != CUSOLVER_STATUS_SUCCESS || linfo != 0) {
    if (want_log) std::fprintf(stderr, "[MGPU] cusolverMgSyevd error (stat=%d, info=%d); fallback to single-GPU DSYEVD\n", (int)s, linfo);
    record_failure();
    cleanup();
    mopac_cuda_dsyevd(n, A, lda, W, info);
    return;
  }

  cusolverStatus_t st_copy_d2h = cusolverMgMemcpyD2H(mh,
                                                     A,
                                                     lda,
                                                     reinterpret_cast<void* const*>(Adev.data()),
                                                     IA,
                                                     JA,
                                                     desc);
  if (st_copy_d2h != CUSOLVER_STATUS_SUCCESS) {
    if (want_log) std::fprintf(stderr, "[MGPU] cusolverMgMemcpyD2H failed; fallback to single-GPU DSYEVD\n");
    record_failure();
    cleanup();
    mopac_cuda_dsyevd(n, A, lda, W, info);
    return;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  cleanup();
  if (profile_enabled) {
    mg_calls += 1;
    mg_total_ms += elapsed_ms;
    mg_total_dim += n;
    mg_total_devices += need;
  }
  if (want_log) {
    std::fprintf(stderr, "[MGPU] DSYEVD n=%d grid=%dx%d blksz=%d: %.3f ms\n", n, gx, gy, blksz, elapsed_ms);
  }

  if (info) *info = linfo;
  return;
#endif
}

// Perform Cholesky factorization of symmetric positive definite matrix S (upper)
// On return, S contains U (upper) such that S = U^T U (host memory)
void mopac_cuda_potrf_upper(int n, double *S, int ld, int *info) {
  if (info) *info = 0;
  if (!g_blas) create_handle();
  static cusolverDnHandle_t solver = nullptr;
  if (!solver) {
    cusolverStatus_t st = cusolverDnCreate(&solver);
    if (st != CUSOLVER_STATUS_SUCCESS) { if (info) *info = -101; return; }
  }

  size_t bytes = (size_t)ld * (size_t)n * sizeof(double);
  DevBuf<double> dS;
  dS.ensure(bytes);
  double *d_S = dS.ptr;
  cudaStream_t s = g_stream ? g_stream : 0;
  cudaMemcpyAsync(d_S, S, bytes, cudaMemcpyHostToDevice, s);
  cusolverDnSetStream(solver, s);
  int lwork = 0;
  cusolverStatus_t st_b = cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, n, d_S, ld, &lwork);
  if (st_b != CUSOLVER_STATUS_SUCCESS || lwork <= 0) { if (info) *info = -102; return; }
  DevBuf<double> dW; dW.ensure(sizeof(double) * (size_t)lwork);
  DevBuf<int> dI; dI.ensure(sizeof(int));
  cusolverStatus_t st = cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, d_S, ld, dW.ptr, lwork, dI.ptr);
  if (st != CUSOLVER_STATUS_SUCCESS) { if (info) *info = -103; return; }
  cudaMemcpyAsync(S, d_S, bytes, cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(info, dI.ptr, sizeof(int), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
}

// F' = X^T F X where X = U^{-1} and S = U^T U (upper). S is overwritten with U (upper).
// All matrices provided in column-major host memory.
void mopac_cuda_transform_fock_with_s(int n,
                                      double *S, int lds,
                                      double *F, int ldf,
                                      int *info) {
  if (info) *info = 0;
  if (!g_blas) create_handle();
  // 1) Cholesky on S -> S := U (upper)
  mopac_cuda_potrf_upper(n, S, lds, info);
  if (info && *info != 0) return;
  // 2) Copy S and F to device
  size_t bytesS = (size_t)lds * (size_t)n * sizeof(double);
  size_t bytesF = (size_t)ldf * (size_t)n * sizeof(double);
  DevBuf<double> dS, dF;
  dS.ensure(bytesS); dF.ensure(bytesF);
  cudaStream_t s = g_stream ? g_stream : 0;
  cudaMemcpyAsync(dS.ptr, S, bytesS, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(dF.ptr, F, bytesF, cudaMemcpyHostToDevice, s);
  cudaStreamSynchronize(s);
  // 3) Y = U^{-T} F  -> solve (U^T) Y = F  => left TRSM with trans=Trans
  const double one = 1.0;
  cublasStatus_t bst1 = cublasDtrsm(g_blas,
              CUBLAS_SIDE_LEFT,
              CUBLAS_FILL_MODE_UPPER,
              CUBLAS_OP_T,
              CUBLAS_DIAG_NON_UNIT,
              n, n,
              &one,
              dS.ptr, lds,
              dF.ptr, ldf);
  if (bst1 != CUBLAS_STATUS_SUCCESS) { if (info) *info = -104; return; }
  // 4) F' = Y U^{-1} -> solve Z = Y * U^{-1} => right TRSM with trans=NoTrans
  cublasStatus_t bst2 = cublasDtrsm(g_blas,
              CUBLAS_SIDE_RIGHT,
              CUBLAS_FILL_MODE_UPPER,
              CUBLAS_OP_N,
              CUBLAS_DIAG_NON_UNIT,
              n, n,
              &one,
              dS.ptr, lds,
              dF.ptr, ldf);
  if (bst2 != CUBLAS_STATUS_SUCCESS) { if (info) *info = -105; return; }
  // 5) Copy back F' to host
  cudaMemcpyAsync(F, dF.ptr, bytesF, cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
}

// Solve for Cocc in AO: U * Cocc = Uocc  (U upper from Cholesky of S)
void mopac_cuda_build_c_from_u(int n, int nocc,
                               const double *U, int ldu,
                               const double *Uocc, int lduocc,
                               double *Cocc, int ldc) {
  if (!g_blas) create_handle();
  size_t bytesU = (size_t)ldu * (size_t)n * sizeof(double);
  size_t bytesUocc = (size_t)lduocc * (size_t)nocc * sizeof(double);
  size_t bytesC = (size_t)ldc * (size_t)nocc * sizeof(double);
  DevBuf<double> dU, dUocc, dC;
  dU.ensure(bytesU); dUocc.ensure(bytesUocc); dC.ensure(bytesC);
  cudaMemcpyAsync(dU.ptr, U, bytesU, cudaMemcpyHostToDevice, g_stream);
  cudaMemcpyAsync(dUocc.ptr, Uocc, bytesUocc, cudaMemcpyHostToDevice, g_stream);
  cudaMemcpyAsync(dC.ptr, dUocc.ptr, bytesUocc, cudaMemcpyDeviceToDevice, g_stream);
  const double one = 1.0;
  // Solve U * C = Uocc  -> Left TRSM with trans=NoTrans
  cublasDtrsm(g_blas,
              CUBLAS_SIDE_LEFT,
              CUBLAS_FILL_MODE_UPPER,
              CUBLAS_OP_N,
              CUBLAS_DIAG_NON_UNIT,
              n, nocc,
              &one,
              dU.ptr, ldu,
              dC.ptr, ldc);
  cudaMemcpyAsync(Cocc, dC.ptr, bytesC, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
}

// P = 2 * Cocc * Cocc^T (upper sym)
void mopac_cuda_density_from_c(int n, int nocc, const double *Cocc, int ldc,
                               double *P, int ldp, double scale) {
  if (!g_blas) create_handle();
  size_t bytesC = (size_t)ldc * (size_t)nocc * sizeof(double);
  size_t bytesP = (size_t)ldp * (size_t)n * sizeof(double);
  DevBuf<double> dC;
  dC.ensure(bytesC);
  g_density_full.ensure(bytesP);
  cudaMemcpyAsync(dC.ptr, Cocc, bytesC, cudaMemcpyHostToDevice, g_stream);
  cudaMemsetAsync(g_density_full.ptr, 0, bytesP, g_stream);
  double alpha = scale;
  double beta  = 0.0;
  cublasDsyrk(g_blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, nocc,
              &alpha, dC.ptr, ldc, &beta, g_density_full.ptr, ldp);
  g_density_full_valid = true;
  g_density_full_n = n;
  g_density_full_ld = ldp;
  invalidate_packed_density();
  cudaMemcpyAsync(P, g_density_full.ptr, bytesP, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
}

} // extern "C"

// =============== Small dense linear solve for DIIS (GPU) ===============
extern "C" {

// Solve A x = b in-place on b using LU (getrf/getrs); A overwritten
void mopac_cuda_solve_linear(int n, double *A, int lda, double *b, int *info) {
  if (!g_blas) create_handle();
  cusolverDnHandle_t solver = nullptr;
  cusolverDnCreate(&solver);
  cusolverDnSetStream(solver, g_stream);
  size_t bytesA = (size_t)lda * (size_t)n * sizeof(double);
  size_t bytesB = sizeof(double) * (size_t)n;
  DevBuf<double> dA, dB;
  dA.ensure(bytesA); dB.ensure(bytesB);
  cudaMemcpyAsync(dA.ptr, A, bytesA, cudaMemcpyHostToDevice, g_stream);
  cudaMemcpyAsync(dB.ptr, b, bytesB, cudaMemcpyHostToDevice, g_stream);
  int lwork = 0;
  DevBuf<int> dIpiv, dInfo;
  dIpiv.ensure(sizeof(int) * (size_t)n);
  dInfo.ensure(sizeof(int));
  cusolverDnDgetrf_bufferSize(solver, n, n, (double*)dA.ptr, lda, &lwork);
  DevBuf<double> dWork; dWork.ensure(sizeof(double) * (size_t)lwork);
  cusolverDnDgetrf(solver, n, n, (double*)dA.ptr, lda, dWork.ptr, dIpiv.ptr, dInfo.ptr);
  // NRHS = 1
  cusolverDnDgetrs(solver, CUBLAS_OP_N, n, 1, (double*)dA.ptr, lda, dIpiv.ptr, dB.ptr, n, dInfo.ptr);
  cudaMemcpyAsync(b, dB.ptr, bytesB, cudaMemcpyDeviceToHost, g_stream);
  cudaMemcpyAsync(info, dInfo.ptr, sizeof(int), cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
  cusolverDnDestroy(solver);
}

} // extern "C"

// =============== Build DIIS B-column on GPU (R^T r_lfock) ===============
extern "C" {

void mopac_cuda_bcol_from_residuals(int linear, int nfock,
                                    const double *fppf, int lfock,
                                    double *out) {
  if (!g_blas) create_handle();
  size_t bytesR = (size_t)linear * (size_t)nfock * sizeof(double);
  DevBuf<double> dR, dY;
  dR.ensure(bytesR);
  dY.ensure(sizeof(double) * (size_t)nfock);
  for (int col = 0; col < nfock; ++col) {
    const double *src = fppf + (size_t)col * (size_t)linear;
    double *dst = dR.ptr + (size_t)col * (size_t)linear;
    cudaMemcpyAsync(dst, src, sizeof(double) * (size_t)linear, cudaMemcpyHostToDevice, g_stream);
  }
  const double *dr = dR.ptr + (size_t)(lfock - 1) * (size_t)linear;
  const double alpha = 1.0;
  const double beta  = 0.0;
  cublasDgemv(g_blas, CUBLAS_OP_T, linear, nfock, &alpha, dR.ptr, linear, dr, 1, &beta, dY.ptr, 1);
  cudaMemcpyAsync(out, dY.ptr, sizeof(double) * (size_t)nfock, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
}

} // extern "C"

// =============== Persistent DIIS residual buffer on GPU ===============
extern "C" {

static DevBuf<double> g_diis_R;
static DevBuf<double> g_diis_full_f;
static DevBuf<double> g_diis_fp;
static DevBuf<double> g_diis_pf;
static DevBuf<double> g_diis_residual_pack;
static int g_diis_linear_cap = 0;

void mopac_cuda_diis_init(int linear, int maxfock) {
  if (!g_blas) create_handle();
  size_t bytes = (size_t)linear * (size_t)maxfock * sizeof(double);
  g_diis_R.ensure(bytes);
  g_diis_linear_cap = linear;
}

void mopac_cuda_diis_store(int linear, int col, const double *r_host) {
  if (g_diis_linear_cap < linear || !g_diis_R.ptr) {
    // Not initialized or too small; ignore store safely
    return;
  }
  size_t offset = (size_t)(col - 1) * (size_t)g_diis_linear_cap;
  double *dst = g_diis_R.ptr + offset;
  cudaMemcpyAsync(dst, r_host, sizeof(double) * (size_t)linear, cudaMemcpyHostToDevice, g_stream);
  cudaStreamSynchronize(g_stream);
}

void mopac_cuda_diis_bcol(int linear, int nfock, int lfock, double *out_host) {
  if (!g_blas) create_handle();
  if (!g_diis_R.ptr || g_diis_linear_cap < linear) {
    // Not initialized; zero output
    for (int i = 0; i < nfock; ++i) out_host[i] = 0.0;
    return;
  }
  DevBuf<double> dY;
  dY.ensure(sizeof(double) * (size_t)nfock);
  const double alpha = 1.0;
  const double beta  = 0.0;
  const double *dr = g_diis_R.ptr + (size_t)(lfock - 1) * (size_t)g_diis_linear_cap;
  cublasDgemv(g_blas, CUBLAS_OP_T, linear, nfock, &alpha, g_diis_R.ptr, g_diis_linear_cap, dr, 1, &beta, dY.ptr, 1);
  cudaMemcpyAsync(out_host, dY.ptr, sizeof(double) * (size_t)nfock, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
}

void mopac_cuda_diis_release() {
  g_diis_R.release();
  g_diis_full_f.release();
  g_diis_fp.release();
  g_diis_pf.release();
  g_diis_residual_pack.release();
  g_diis_linear_cap = 0;
}

} // extern "C"

// =============== Full B = R^T R assembly on GPU ===============
extern "C" {

void mopac_cuda_bfull_from_host(int linear, int nfock,
                                const double *fppf,
                                double *b_out) {
  if (!g_blas) create_handle();
  size_t bytesR = (size_t)linear * (size_t)nfock * sizeof(double);
  DevBuf<double> dR, dB;
  dR.ensure(bytesR);
  dB.ensure(sizeof(double) * (size_t)nfock * (size_t)nfock);
  for (int col = 0; col < nfock; ++col) {
    const double *src = fppf + (size_t)col * (size_t)linear;
    double *dst = dR.ptr + (size_t)col * (size_t)linear;
    cudaMemcpyAsync(dst, src, sizeof(double) * (size_t)linear, cudaMemcpyHostToDevice, g_stream);
  }
  const double alpha = 1.0;
  const double beta  = 0.0;
  cublasDgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
              nfock, nfock, linear,
              &alpha,
              dR.ptr, linear,
              dR.ptr, linear,
              &beta,
              dB.ptr, nfock);
  cudaMemcpyAsync(b_out, dB.ptr, sizeof(double) * (size_t)nfock * (size_t)nfock, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
}

void mopac_cuda_bfull_from_device(int linear, int nfock, double *b_out) {
  if (!g_blas) create_handle();
  if (!g_diis_R.ptr || g_diis_linear_cap < linear) {
    for (int i = 0; i < nfock*nfock; ++i) b_out[i] = 0.0;
    return;
  }
  DevBuf<double> dB;
  dB.ensure(sizeof(double) * (size_t)nfock * (size_t)nfock);
  const double alpha = 1.0;
  const double beta  = 0.0;
  cublasDgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
              nfock, nfock, linear,
              &alpha,
              g_diis_R.ptr, g_diis_linear_cap,
              g_diis_R.ptr, g_diis_linear_cap,
              &beta,
              dB.ptr, nfock);
  cudaMemcpyAsync(b_out, dB.ptr, sizeof(double) * (size_t)nfock * (size_t)nfock, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
}

bool mopac_cuda_diis_residual_resident(int n, int linear, int col,
                                       const double *f_host_ptr,
                                       const double *p_host_ptr,
                                       double *host_out,
                                       int copy_back_flag) {
  if (!resident_mode_enabled()) return false;
  if (!g_blas) create_handle();
  if (!g_fock_cache.valid || g_fock_cache.len != (size_t)linear) return false;
  if (f_host_ptr && g_fock_cache.host_ptr && g_fock_cache.host_ptr != f_host_ptr) return false;
  PackedDensitySlot *density_slot = find_packed_slot(p_host_ptr, (size_t)linear);
  if (!density_slot) return false;
  if (p_host_ptr && density_slot->host_ptr && density_slot->host_ptr != p_host_ptr) return false;
  if (!g_density_full_valid || g_density_full_n != n) return false;
  if (!g_diis_R.ptr || g_diis_linear_cap < linear) return false;

  cudaStream_t s = g_stream ? g_stream : 0;

  size_t nn = (size_t)n * (size_t)n;
  g_diis_full_f.ensure(sizeof(double) * nn);
  g_diis_fp.ensure(sizeof(double) * nn);
  g_diis_pf.ensure(sizeof(double) * nn);
  g_diis_residual_pack.ensure(sizeof(double) * (size_t)linear);

  int block = 256;
  int grid = static_cast<int>((nn + block - 1) / block);
  unpack_lower_to_full_kernel<<<grid, block, 0, s>>>(g_fock_cache.buf.ptr, g_diis_full_f.ptr, n);
  if (cudaGetLastError() != cudaSuccess) return false;

  const double alpha = 1.0;
  const double beta  = 0.0;
  int ldP = g_density_full_ld;

  cublasStatus_t st = cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, n, n,
                                  &alpha,
                                  g_diis_full_f.ptr, n,
                                  g_density_full.ptr, ldP,
                                  &beta,
                                  g_diis_fp.ptr, n);
  if (st != CUBLAS_STATUS_SUCCESS) return false;

  st = cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                   n, n, n,
                   &alpha,
                   g_density_full.ptr, ldP,
                   g_diis_full_f.ptr, n,
                   &beta,
                   g_diis_pf.ptr, n);
  if (st != CUBLAS_STATUS_SUCCESS) return false;

  const double minus_one = -1.0;
  st = cublasDaxpy(g_blas, n * n, &minus_one, g_diis_pf.ptr, 1, g_diis_fp.ptr, 1);
  if (st != CUBLAS_STATUS_SUCCESS) return false;

  size_t total = (size_t)linear;
  grid = static_cast<int>((total + block - 1) / block);
  pack_upper_kernel<<<grid, block, 0, s>>>(g_diis_fp.ptr, n, n, g_diis_residual_pack.ptr);
  if (cudaGetLastError() != cudaSuccess) return false;

  size_t offset = (size_t)(col - 1) * (size_t)g_diis_linear_cap;
  double *dst = g_diis_R.ptr + offset;
  if (cudaMemcpyAsync(dst, g_diis_residual_pack.ptr,
                      sizeof(double) * (size_t)linear,
                      cudaMemcpyDeviceToDevice, s) != cudaSuccess) return false;
  if (host_out && copy_back_flag != 0) {
    if (cudaMemcpyAsync(host_out, g_diis_residual_pack.ptr,
                        sizeof(double) * (size_t)linear,
                        cudaMemcpyDeviceToHost, s) != cudaSuccess) return false;
  }
  cudaStreamSynchronize(s);
  if (host_out && copy_back_flag == 0) {
    for (int i = 0; i < linear; ++i) host_out[i] = 0.0;
  }
  return true;
}

} // extern "C"

// =============== F*C MO transform helper ===============
extern "C" {

// Compute W = F * C, where F is given in packed lower-triangular form (size n(n+1)/2)
// and C, W are n x n (column-major). Uses cuBLAS GEMM on an unpacked full symmetric F.
void mopac_cuda_fmulC(int n, const double *F_packed, const double *C, int ldc, double *W, int ldw) {
  if (!g_blas) create_handle();
  size_t bytesN = (size_t)n * (size_t)n * sizeof(double);
  size_t linear = (size_t)n * ((size_t)n + 1) / 2;
  cudaStream_t s = g_stream ? g_stream : 0;

  bool used_cache = false;
  DevBuf<double> dPacked;
  if (resident_mode_enabled()) {
    dPacked.ensure(sizeof(double) * linear);
    if (mopac_cuda_fock_copy_cached(dPacked.ptr, linear, F_packed)) {
      used_cache = true;
    }
  }

  DevBuf<double> dF, dC, dW;
  dF.ensure(bytesN);
  dC.ensure(bytesN);
  dW.ensure(bytesN);
  cudaMemcpyAsync(dC.ptr, C, bytesN, cudaMemcpyHostToDevice, s);

  if (used_cache) {
    int total = n * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    unpack_lower_to_full_kernel<<<grid, block, 0, s>>>(dPacked.ptr, dF.ptr, n);
    cudaStreamSynchronize(s);
  } else {
    static HostBuf<double> hF;
    hF.ensure(bytesN);
    size_t idx = 0;
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < n; ++row) {
        hF.ptr[row + (size_t)col * (size_t)n] = 0.0;
      }
    }
    for (int col = 0; col < n; ++col) {
      for (int row = col; row < n; ++row) {
        double v = F_packed[idx++];
        hF.ptr[row + (size_t)col * (size_t)n] = v;
        hF.ptr[col + (size_t)row * (size_t)n] = v;
      }
    }
    cudaMemcpyAsync(dF.ptr, hF.ptr, bytesN, cudaMemcpyHostToDevice, s);
  }

  double alpha = 1.0, beta = 0.0;
  cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
              n, n, n, &alpha,
              dF.ptr, n,
              dC.ptr, ldc,
              &beta,
              dW.ptr, ldw);
  cudaMemcpyAsync(W, dW.ptr, bytesN, cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
}

} // extern "C"

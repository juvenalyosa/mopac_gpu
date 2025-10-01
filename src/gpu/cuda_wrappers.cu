// Portable CUDA interop for MOPAC: cuBLAS GEMM, cuSOLVER SYEVD, and basic GPU info
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
#include <cctype>
#include <mutex>

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
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "n") == 0 || std::strcmp(env, "N") == 0) {
      g_resident_mode = 0;
    } else {
      g_resident_mode = 1;
    }
  } else {
    g_resident_mode = 1; // default on when not specified
  }
  return g_resident_mode != 0;
}

extern "C" __global__ void unpack_lower_to_full_kernel(const double *packed, double *full, int n);

// Simple grow-only device buffer cache helper (C++ only; placed outside C linkage)
template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  size_t cap = 0; // capacity in bytes
  void ensure(size_t bytes) {
    if (bytes <= cap && ptr) return;
    if (ptr) cudaFree(ptr);
    ptr = nullptr; cap = 0;
    if (bytes > 0) {
      cudaMalloc((void**)&ptr, bytes);
      cap = bytes;
    }
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

struct PackedDensityCache {
  DevBuf<double> buf;
  size_t len = 0;
  const double* host_ptr = nullptr;
  bool valid = false;
};

static PackedDensityCache g_packed_density;

static inline void invalidate_packed_density() {
  g_packed_density.valid = false;
  g_packed_density.host_ptr = nullptr;
  g_packed_density.len = 0;
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
    cublasCreate(&g_blas);
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
      if (std::strcmp(env, "off") == 0 || std::strcmp(env, "0") == 0) {
        g_streams_enabled = false;
      }
    }
    const char* env_pin = std::getenv("MOPAC_PIN_USER");
    if (env_pin) {
      if (std::strcmp(env_pin, "on") == 0 || std::strcmp(env_pin, "1") == 0 || std::strcmp(env_pin, "true") == 0) {
        g_pin_user = true;
      }
    }
    if (g_streams_enabled) {
      if (!g_stream) {
        cudaStreamCreate(&g_stream);
      }
      cublasSetStream(g_blas, g_stream);
    }
  }
}

static inline void create_handle_xt() {
  if (!g_blasXt) {
    cublasXtCreate(&g_blasXt);
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

// Fortran-callable DGEMM via cuBLAS (uses cached device buffers)
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
    bool pinned = false;
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
      cudaMemcpyAsync(d_A, A, bytesA, cudaMemcpyHostToDevice, g_stream);
      cudaMemcpyAsync(d_B, B, bytesB, cudaMemcpyHostToDevice, g_stream);
      if (beta != 0.0) cudaMemcpyAsync(d_C, C, bytesC, cudaMemcpyHostToDevice, g_stream);
    } else {
      h_gemm_A.ensure(bytesA);
      h_gemm_B.ensure(bytesB);
      h_gemm_C.ensure(bytesC);
      std::memcpy(h_gemm_A.ptr, A, bytesA);
      std::memcpy(h_gemm_B.ptr, B, bytesB);
      cudaMemcpyAsync(d_A, h_gemm_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream);
      cudaMemcpyAsync(d_B, h_gemm_B.ptr, bytesB, cudaMemcpyHostToDevice, g_stream);
      if (beta != 0.0) {
        std::memcpy(h_gemm_C.ptr, C, bytesC);
        cudaMemcpyAsync(d_C, h_gemm_C.ptr, bytesC, cudaMemcpyHostToDevice, g_stream);
      }
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

    if (pinned) {
      cudaMemcpyAsync(C, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream);
      cudaStreamSynchronize(g_stream);
      cudaHostUnregister((void*)A);
      cudaHostUnregister((void*)B);
      if (beta != 0.0) cudaHostUnregister((void*)C);
    } else {
      cudaMemcpyAsync(h_gemm_C.ptr, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream);
      cudaStreamSynchronize(g_stream);
      std::memcpy(C, h_gemm_C.ptr, bytesC);
    }
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
      std::memcpy(h_gemm_A.ptr, A, bytesA);
      std::memcpy(h_gemm_B.ptr, B, bytesB);
      cudaMemcpyAsync(d_A, h_gemm_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream);
      cudaMemcpyAsync(d_B, h_gemm_B.ptr, bytesB, cudaMemcpyHostToDevice, g_stream);
      if (beta != 0.0) {
        std::memcpy(h_gemm_C.ptr, C, bytesC);
        cudaMemcpyAsync(d_C, h_gemm_C.ptr, bytesC, cudaMemcpyHostToDevice, g_stream);
      }
      cublasDgemm(g_blas, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
      cudaMemcpyAsync(h_gemm_C.ptr, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream);
      cudaStreamSynchronize(g_stream);
      std::memcpy(C, h_gemm_C.ptr, bytesC);
      return;
    }

    size_t bytesA_tile = (size_t)lda * (size_t)k * sizeof(double);
    g_gemm_A.ensure(bytesA_tile);
    double *d_A = g_gemm_A.ptr;
    h_gemm_A.ensure(bytesA_tile);
    std::memcpy(h_gemm_A.ptr, A, bytesA_tile);
    cudaMemcpyAsync(d_A, h_gemm_A.ptr, bytesA_tile, cudaMemcpyHostToDevice, g_stream);
    cudaStreamSynchronize(g_stream);

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

      const double *B_tile = B + (size_t)col0 * (size_t)ldb;
      double *C_tile = C + (size_t)col0 * (size_t)ldc;

      cudaMemcpy2DAsync(d_B, ldb * sizeof(double),
                        B_tile, ldb * sizeof(double),
                        (size_t)tn * sizeof(double), (size_t)k,
                        cudaMemcpyHostToDevice, g_stream);
      if (beta != 0.0) {
        cudaMemcpy2DAsync(d_C, ldc * sizeof(double),
                          C_tile, ldc * sizeof(double),
                          (size_t)tn * sizeof(double), (size_t)m,
                          cudaMemcpyHostToDevice, g_stream);
      }
      cublasStatus_t st = cublasDgemm(g_blas, opA, opB, m, tn, k,
                                      &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
      if (w_verbose && st == CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[GPU] DGEMM tile m=%d n=%d k=%d\n", m, tn, k);
      }
      cudaMemcpy2DAsync(C_tile, ldc * sizeof(double),
                        d_C, ldc * sizeof(double),
                        (size_t)tn * sizeof(double), (size_t)m,
                        cudaMemcpyDeviceToHost, g_stream);
      cudaStreamSynchronize(g_stream);
    }
    prof_scope.note_tiles(tile_chunks);
  }
}

// SYRK via cuBLAS (uses cached device buffers)
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

  if (!use_tiling) {
    g_syrk_A.ensure(bytesA);
    double *d_A = g_syrk_A.ptr;
    cudaMemcpyAsync(d_A, A, bytesA, cudaMemcpyHostToDevice, stream);
    if (beta != 0.0) {
      cudaMemcpyAsync(d_C, C, bytesC, cudaMemcpyHostToDevice, stream);
    }
    float ms = 0.0f; cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (w_verbose) {
      if (cudaEventCreate(&ev0) != cudaSuccess) { ev0 = nullptr; }
      if (cudaEventCreate(&ev1) != cudaSuccess) { if (ev0) cudaEventDestroy(ev0); ev0 = nullptr; ev1 = nullptr; }
      if (ev0) cudaEventRecord(ev0, stream);
    }
    cublasStatus_t st2 = cublasDsyrk(g_blas, u, opA, n, k, &alpha, d_A, lda, &beta, d_C, ldc);
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
    cudaMemcpyAsync(C, d_C, bytesC, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  } else {
    if (w_verbose) {
      std::fprintf(stderr, "[GPU] DSYRK n=%d k=%d: tiled panels (tile_k=%d)\n", n, k, tile_k);
    }
    cudaMemcpyAsync(d_C, C, bytesC, cudaMemcpyHostToDevice, stream);
    long long tile_chunks = 0;
    for (int k0 = 0; k0 < k; k0 += tile_k) {
      tile_chunks++;
      int kc = std::min(tile_k, k - k0);
      size_t bytesA_tile = (size_t)lda * (size_t)kc * sizeof(double);
      g_syrk_A.ensure(bytesA_tile);
      double *d_Atile = g_syrk_A.ptr;
      const double *A_tile = A + (size_t)k0 * (size_t)lda;
      cudaMemcpy2DAsync(d_Atile, lda * sizeof(double),
                        A_tile, lda * sizeof(double),
                        (size_t)kc * sizeof(double), (size_t)n,
                        cudaMemcpyHostToDevice, stream);
      double beta_local = (k0 == 0) ? beta : 1.0;
      cublasStatus_t st2 = cublasDsyrk(g_blas, u, CUBLAS_OP_N, n, kc, &alpha, d_Atile, lda, &beta_local, d_C, ldc);
      if (w_verbose && st2 == CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[GPU] DSYRK tile n=%d k=%d beta=%.1f\n", n, kc, beta_local);
      }
    }
    prof_scope.note_tiles(tile_chunks);
    cudaMemcpyAsync(C, d_C, bytesC, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
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

static int g_mozyme_f2_flag = -1;
static inline bool mozyme_f2_enabled() {
  if (g_mozyme_f2_flag < 0) {
    const char* env = std::getenv("MOPAC_MOZYME_F2_GPU");
    if (env && *env) {
      if (equals_ci(env, "0") || equals_ci(env, "off") || equals_ci(env, "false")) {
        g_mozyme_f2_flag = 0;
      } else {
        g_mozyme_f2_flag = 1;
      }
    } else {
      g_mozyme_f2_flag = 1;
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
  cublasOperation_t opA = (tra == 'T' || tra == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = (trb == 'T' || trb == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto t0 = std::chrono::high_resolution_clock::now();
  cublasXtDgemm(g_blasXt, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
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
  cublasFillMode_t U = (uplo == 'U' || uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t opA = (tra == 'T' || tra == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto t0 = std::chrono::high_resolution_clock::now();
  cublasXtDsyrk(g_blasXt, U, opA, n, k, &alpha, A, lda, &beta, C, ldc);
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
  if (!g_solver) cusolverDnCreate(&g_solver);

  size_t bytesA = sizeof(double) * (size_t)lda * (size_t)n;
  size_t bytesW = sizeof(double) * (size_t)n;
  g_dsyevd_A.ensure(bytesA);
  g_dsyevd_W.ensure(bytesW);
  g_dsyevd_info.ensure(sizeof(int));
  double *d_A = g_dsyevd_A.ptr;
  double *d_W = g_dsyevd_W.ptr;
  int *d_info = g_dsyevd_info.ptr;

  static HostBuf<double> h_dsyevd_A, h_dsyevd_W;
  h_dsyevd_A.ensure(bytesA);
  h_dsyevd_W.ensure(bytesW);
  std::memcpy(h_dsyevd_A.ptr, A, bytesA);
  cudaMemcpyAsync(d_A, h_dsyevd_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream);

  int lwork = 0;
  cusolverDnSetStream(g_solver, g_stream);
  cusolverDnDsyevd_bufferSize(g_solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                              n, d_A, lda, d_W, &lwork);
  if (lwork > g_dsyevd_lwork_cap) {
    g_dsyevd_work.ensure(sizeof(double) * (size_t)lwork);
    g_dsyevd_lwork_cap = lwork;
  }
  double *d_work = g_dsyevd_work.ptr;

  cusolverDnDsyevd(g_solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                   n, d_A, lda, d_W, d_work, lwork, d_info);
  cudaMemcpyAsync(h_dsyevd_A.ptr, d_A, bytesA, cudaMemcpyDeviceToHost, g_stream);
  cudaMemcpyAsync(h_dsyevd_W.ptr, d_W, bytesW, cudaMemcpyDeviceToHost, g_stream);
  cudaMemcpyAsync(info, d_info, sizeof(int), cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
  std::memcpy(A, h_dsyevd_A.ptr, bytesA);
  std::memcpy(W, h_dsyevd_W.ptr, bytesW);
}

// Variant: keep eigenvectors on device, return eigenvalues only
void mopac_cuda_dsyevd_keep(int n, double *A, int lda, double *W, int *info) {
  if (!g_solver) cusolverDnCreate(&g_solver);

  size_t bytesA = sizeof(double) * (size_t)lda * (size_t)n;
  size_t bytesW = sizeof(double) * (size_t)n;
  g_dsyevd_A.ensure(bytesA);
  g_dsyevd_W.ensure(bytesW);
  g_dsyevd_info.ensure(sizeof(int));
  double *d_A = g_dsyevd_A.ptr;
  double *d_W = g_dsyevd_W.ptr;
  int *d_info = g_dsyevd_info.ptr;

  static HostBuf<double> h_dsyevd_A, h_dsyevd_W;
  h_dsyevd_A.ensure(bytesA);
  h_dsyevd_W.ensure(bytesW);
  // Copy host-packed upper triangle (in A) that was unpacked by caller into full matrix; here A is ignored.
  // Caller should have already unpacked; we accept A as a full matrix buffer for simplicity.
  std::memcpy(h_dsyevd_A.ptr, A, bytesA);
  cudaMemcpyAsync(d_A, h_dsyevd_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream);

  int lwork = 0;
  cusolverDnSetStream(g_solver, g_stream);
  cusolverDnDsyevd_bufferSize(g_solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                              n, d_A, lda, d_W, &lwork);
  if (lwork > g_dsyevd_lwork_cap) {
    g_dsyevd_work.ensure(sizeof(double) * (size_t)lwork);
    g_dsyevd_lwork_cap = lwork;
  }
  double *d_work = g_dsyevd_work.ptr;

  cusolverDnDsyevd(g_solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                   n, d_A, lda, d_W, d_work, lwork, d_info);
  cudaMemcpyAsync(h_dsyevd_W.ptr, d_W, bytesW, cudaMemcpyDeviceToHost, g_stream);
  cudaMemcpyAsync(info, d_info, sizeof(int), cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
  std::memcpy(W, h_dsyevd_W.ptr, bytesW);
  // Mark device eigenvectors available
  g_have_device_eigvecs = true;
  g_device_eigvecs_n = n;
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
  // Allocate and copy eigenvector matrix to device (async, pinned if possible)
  size_t bytesV = sizeof(double) * (size_t)n * (size_t)n;
  g_rot_V.ensure(bytesV);
  double *d_V = g_rot_V.ptr;
  static HostBuf<double> h_rot_V;
  h_rot_V.ensure(bytesV);
  std::memcpy(h_rot_V.ptr, vector, bytesV);
  cudaMemcpyAsync(d_V, h_rot_V.ptr, bytesV, cudaMemcpyHostToDevice, g_stream);

  // Walk pairs sequentially; batch to reduce kernel launches
  const int max_batch = 256;
  int   *h_i = (int*)malloc(sizeof(int) * max_batch);
  int   *h_j = (int*)malloc(sizeof(int) * max_batch);
  double *h_a = (double*)malloc(sizeof(double) * max_batch);
  double *h_b = (double*)malloc(sizeof(double) * max_batch);
  g_rot_i.ensure(sizeof(int) * max_batch);
  g_rot_j.ensure(sizeof(int) * max_batch);
  g_rot_a.ensure(sizeof(double) * max_batch);
  g_rot_b.ensure(sizeof(double) * max_batch);
  int   *d_i = g_rot_i.ptr, *d_j = g_rot_j.ptr;
  double *d_a = g_rot_a.ptr, *d_b = g_rot_b.ptr;

  int ij = 0;
  for (int i = 0; i < nocc; ++i) {
    int batch = 0;
    for (int j = lumo - 1; j < n; ++j) {
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
        cudaMemcpyAsync(d_i, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream);
        cudaMemcpyAsync(d_j, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream);
        cudaMemcpyAsync(d_a, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream);
        cudaMemcpyAsync(d_b, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream);
        int block = 256;
        int grid = (n + block - 1) / block;
        drot_cols_batch_kernel<<<grid, block, 0, g_stream>>>(d_V, n, batch, d_i, d_j, d_a, d_b);
        batch = 0;
      }
    }
    if (batch > 0) {
      cudaMemcpyAsync(d_i, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream);
      cudaMemcpyAsync(d_j, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream);
      cudaMemcpyAsync(d_a, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream);
      cudaMemcpyAsync(d_b, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream);
      int block = 256;
      int grid = (n + block - 1) / block;
      drot_cols_batch_kernel<<<grid, block, 0, g_stream>>>(d_V, n, batch, d_i, d_j, d_a, d_b);
    }
  }
  cudaStreamSynchronize(g_stream);
  // Copy back result
  cudaMemcpyAsync(h_rot_V.ptr, d_V, sizeof(double) * (size_t)n * (size_t)n, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
  std::memcpy(vector, h_rot_V.ptr, bytesV);
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
  int dev_count = 0;
  cudaGetDeviceCount(&dev_count);
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
  cudaSetDevice(dev0);
  g2_rot_V0.ensure(bytes0);
  g2_rot_i0.ensure(sizeof(int) * 256);
  g2_rot_j0.ensure(sizeof(int) * 256);
  g2_rot_a0.ensure(sizeof(double) * 256);
  g2_rot_b0.ensure(sizeof(double) * 256);
  d_V0 = g2_rot_V0.ptr; d_i0 = g2_rot_i0.ptr; d_j0 = g2_rot_j0.ptr; d_a0 = g2_rot_a0.ptr; d_b0 = g2_rot_b0.ptr;
  // Pinned staging for full matrix
  size_t bytesV = sizeof(double) * (size_t)n * (size_t)n;
  h2_rot_V.ensure(bytesV);
  std::memcpy(h2_rot_V.ptr, vector, bytesV);
  // Copy top slice rows [0..n0) on device 0 using its stream
  for (int col = 0; col < n; ++col) {
    const double *col_ptr = h2_rot_V.ptr + (size_t)col * (size_t)n;
    cudaMemcpyAsync(d_V0 + (size_t)col * (size_t)n0, col_ptr, sizeof(double) * n0,
                    cudaMemcpyHostToDevice, g_stream0);
  }

  cudaSetDevice(dev1);
  g2_rot_V1.ensure(bytes1);
  g2_rot_i1.ensure(sizeof(int) * 256);
  g2_rot_j1.ensure(sizeof(int) * 256);
  g2_rot_a1.ensure(sizeof(double) * 256);
  g2_rot_b1.ensure(sizeof(double) * 256);
  d_V1 = g2_rot_V1.ptr; d_i1 = g2_rot_i1.ptr; d_j1 = g2_rot_j1.ptr; d_a1 = g2_rot_a1.ptr; d_b1 = g2_rot_b1.ptr;
  // Copy bottom slice rows [n0..n)
  for (int col = 0; col < n; ++col) {
    const double *col_ptr = h2_rot_V.ptr + (size_t)col * (size_t)n + (size_t)n0;
    cudaMemcpyAsync(d_V1 + (size_t)col * (size_t)n1, col_ptr, sizeof(double) * n1,
                    cudaMemcpyHostToDevice, g_stream1);
  }

  // Host batching buffers
  const int max_batch = 256;
  int   *h_i = (int*)malloc(sizeof(int) * max_batch);
  int   *h_j = (int*)malloc(sizeof(int) * max_batch);
  double *h_a = (double*)malloc(sizeof(double) * max_batch);
  double *h_b = (double*)malloc(sizeof(double) * max_batch);

  int ij = 0;
  for (int i = 0; i < nocc; ++i) {
    int batch = 0;
    for (int j = lumo - 1; j < n; ++j) {
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
        // Launch on device dev0
        cudaSetDevice(dev0);
        cudaMemcpyAsync(d_i0, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream0);
        cudaMemcpyAsync(d_j0, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream0);
        cudaMemcpyAsync(d_a0, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream0);
        cudaMemcpyAsync(d_b0, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream0);
        {
          int block = 256;
          int grid = (n0 + block - 1) / block;
          drot_cols_batch_kernel_strided<<<grid, block, 0, g_stream0>>>(d_V0, n0, n, batch, d_i0, d_j0, d_a0, d_b0);
        }
        // Launch on device dev1
        cudaSetDevice(dev1);
        cudaMemcpyAsync(d_i1, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream1);
        cudaMemcpyAsync(d_j1, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream1);
        cudaMemcpyAsync(d_a1, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream1);
        cudaMemcpyAsync(d_b1, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream1);
        {
          int block = 256;
          int grid = (n1 + block - 1) / block;
          drot_cols_batch_kernel_strided<<<grid, block, 0, g_stream1>>>(d_V1, n1, n, batch, d_i1, d_j1, d_a1, d_b1);
        }
        batch = 0;
      }
    }
    if (batch > 0) {
      cudaSetDevice(dev0);
      cudaMemcpyAsync(d_i0, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream0);
      cudaMemcpyAsync(d_j0, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream0);
      cudaMemcpyAsync(d_a0, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream0);
      cudaMemcpyAsync(d_b0, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream0);
      {
        int block = 256;
        int grid = (n0 + block - 1) / block;
        drot_cols_batch_kernel_strided<<<grid, block, 0, g_stream0>>>(d_V0, n0, n, batch, d_i0, d_j0, d_a0, d_b0);
      }
      cudaSetDevice(dev1);
      cudaMemcpyAsync(d_i1, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream1);
      cudaMemcpyAsync(d_j1, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice, g_stream1);
      cudaMemcpyAsync(d_a1, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream1);
      cudaMemcpyAsync(d_b1, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice, g_stream1);
      {
        int block = 256;
        int grid = (n1 + block - 1) / block;
        drot_cols_batch_kernel_strided<<<grid, block, 0, g_stream1>>>(d_V1, n1, n, batch, d_i1, d_j1, d_a1, d_b1);
      }
    }
  }

  // Synchronize both devices
  cudaSetDevice(dev0); cudaStreamSynchronize(g_stream0);
  cudaSetDevice(dev1); cudaStreamSynchronize(g_stream1);

  // Copy results back into pinned host matrix
  cudaSetDevice(dev0);
  for (int col = 0; col < n; ++col) {
    double *col_ptr = h2_rot_V.ptr + (size_t)col * (size_t)n;
    cudaMemcpyAsync(col_ptr, d_V0 + (size_t)col * (size_t)n0, sizeof(double) * n0,
               cudaMemcpyDeviceToHost, g_stream0);
  }
  cudaSetDevice(dev1);
  for (int col = 0; col < n; ++col) {
    double *col_ptr = h2_rot_V.ptr + (size_t)col * (size_t)n + (size_t)n0;
    cudaMemcpyAsync(col_ptr, d_V1 + (size_t)col * (size_t)n1, sizeof(double) * n1,
               cudaMemcpyDeviceToHost, g_stream1);
  }
  cudaSetDevice(dev0); cudaStreamSynchronize(g_stream0);
  cudaSetDevice(dev1); cudaStreamSynchronize(g_stream1);
  // Copy back staged matrix to user memory
  std::memcpy(vector, h2_rot_V.ptr, bytesV);

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
  if (skip && *skip) return;
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
  g_packed_density.buf.release();
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
  g_mz_fock2_pii.release(); g_mz_fock2_pjj.release(); g_mz_fock2_pij.release();
  g_mz_fock2_wj.release(); g_mz_fock2_wk.release();
  g_mz_fock2_fii.release(); g_mz_fock2_fjj.release(); g_mz_fock2_fij.release();
  h_mz_fock2_fii.release(); h_mz_fock2_fjj.release(); h_mz_fock2_fij.release();
  g_mz_dfock2_fii.release(); g_mz_dfock2_fjj.release(); g_mz_dfock2_fij.release();
  h_mz_dfock2_fii.release(); h_mz_dfock2_fjj.release(); h_mz_dfock2_fij.release();
  g_resident_mode = -1;
  g_stream_backup = nullptr;
  g_stream_override_depth = 0;
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
  if (!g_packed_density.valid) return false;
  if (!host_ptr) return false;
  if (g_packed_density.len != linear) return false;
  cudaStream_t s = g_stream ? g_stream : 0;
  if (cudaMemcpyAsync(host_ptr, g_packed_density.buf.ptr, sizeof(double)*linear, cudaMemcpyDeviceToHost, s) != cudaSuccess) return false;
  cudaStreamSynchronize(s);
  return true;
}

bool mopac_cuda_cart_gradient(int numat, int l123, const double *coord,
                              double *grad, const double *charges,
                              const void *near_pairs, int near_count,
                              const void *far_pairs, int far_count) {
  (void)near_pairs;
  (void)near_count;
  (void)far_pairs;
  (void)far_count;
  if (!coord || !grad || !charges) return false;
  return mopac_gpu_cart_gradient_cpu(numat, l123, coord, grad, charges);
}

void mopac_cuda_clear_density_cache() {
  g_density_full_valid = false;
  g_density_full_n = 0;
  g_density_full_ld = 0;
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
  g_packed_density.buf.ensure(bytes);
  cudaStream_t s = g_stream ? g_stream : 0;
  int block = 256;
  int grid = ((size_t)linear + block - 1) / block;
  pack_upper_kernel<<<grid, block, 0, s>>>(g_density_full.ptr, g_density_full_ld, g_density_full_n, g_packed_density.buf.ptr);
  g_packed_density.len = (size_t)linear;
  g_packed_density.host_ptr = packed_host;
  g_packed_density.valid = true;
}

bool mopac_cuda_density_copy_cached(double *dest, size_t len, const double *host_ptr) {
  if (!dest || !host_ptr) return false;
  if (!resident_mode_enabled()) return false;
  if (!g_packed_density.valid) return false;
  if (g_packed_density.host_ptr != host_ptr) return false;
  if (g_packed_density.len != len) return false;
  if (cudaMemcpy(dest, g_packed_density.buf.ptr, len * sizeof(double), cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
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
  if (!g_packed_density.valid || g_packed_density.len != (size_t)linear) return false;
  if (p_host_ptr && g_packed_density.host_ptr && g_packed_density.host_ptr != p_host_ptr) return false;
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

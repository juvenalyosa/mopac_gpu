// Minimal light-light branch for Fock derivative (dfock2) on GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>
#include <array>
#include <cmath>

#include "packed_utils.h"

// Silence intentional placeholders to avoid noisy nvcc warnings
#if !defined(MOPAC_UNUSED)
#  if defined(__GNUC__) || defined(__clang__)
#    define MOPAC_UNUSED __attribute__((unused))
#  else
#    define MOPAC_UNUSED
#  endif
#endif

extern "C" {


extern "C" bool mopac_cuda_density_copy_cached(double *dest, size_t len, const double *host_ptr);
extern "C" int mopac_cuda_get_resident_mode();
extern "C" void mopac_cuda_clear_fock_cache();
extern "C" void mopac_cuda_register_fock_device(int linear, double *host_ptr, const double *src_dev);
extern "C" bool mopac_cuda_fock_copy_cached(double *dest, size_t len, const double *host_ptr);
extern "C" bool mopac_cuda_fetch_fock(double *host_ptr, size_t linear);

// Atomic add for double that works on pre-6.0 architectures via CAS
__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    double sum = val + __longlong_as_double(assumed);
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(sum));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

// Persistent buffer helpers and verbose/tuning controls
static int    *s_d_nf = nullptr, *s_d_nl = nullptr;
static double *s_d_ptot = nullptr, *s_d_p = nullptr, *s_d_w = nullptr, *s_d_wj = nullptr,
              *s_d_wk = nullptr, *s_d_f = nullptr;
static int    *s_d_pair_i = nullptr, *s_d_pair_j = nullptr, *s_d_pair_off = nullptr,
              *s_d_pair_type = nullptr, *s_d_pair_wj_off = nullptr,
              *s_d_pair_wk_off = nullptr;
static size_t cap_nf = 0, cap_nl = 0;
static size_t cap_ptot = 0, cap_p = 0, cap_f = 0, cap_w = 0, cap_wj = 0, cap_wk = 0;
static size_t cap_pairs = 0;

enum PairTypeCodes {
  PAIR_LIGHT_LIGHT = 0,
  PAIR_HEAVY_LIGHT = 1,
  PAIR_LIGHT_HEAVY = 2,
  PAIR_HEAVY_HEAVY = 3,
  PAIR_GENERAL = 4,
  PAIR_PERIODIC = 5
};

__host__ __device__ inline int jab_suma_index(int row, int col) {
  constexpr int map[10][16] = {
    { 0, 10, 30, 60, 10, 20, 40, 70, 30, 40, 50, 80, 60, 70, 80, 90 },
    { 1, 11, 31, 61, 11, 21, 41, 71, 31, 41, 51, 81, 61, 71, 81, 91 },
    { 2, 12, 32, 62, 12, 22, 42, 72, 32, 42, 52, 82, 62, 72, 82, 92 },
    { 3, 13, 33, 63, 13, 23, 43, 73, 33, 43, 53, 83, 63, 73, 83, 93 },
    { 4, 14, 34, 64, 14, 24, 44, 74, 34, 44, 54, 84, 64, 74, 84, 94 },
    { 5, 15, 35, 65, 15, 25, 45, 75, 35, 45, 55, 85, 65, 75, 85, 95 },
    { 6, 16, 36, 66, 16, 26, 46, 76, 36, 46, 56, 86, 66, 76, 86, 96 },
    { 7, 17, 37, 67, 17, 27, 47, 77, 37, 47, 57, 87, 67, 77, 87, 97 },
    { 8, 18, 38, 68, 18, 28, 48, 78, 38, 48, 58, 88, 68, 78, 88, 98 },
    { 9, 19, 39, 69, 19, 29, 49, 79, 39, 49, 59, 89, 69, 79, 89, 99 }
  };
  return map[row][col];
}

__host__ __device__ inline int jab_sumb_index(int row, int col) {
  constexpr int map[10][16] = {
    { 0, 1, 3, 6, 1, 2, 4, 7, 3, 4, 5, 8, 6, 7, 8, 9 },
    { 10, 11, 13, 16, 11, 12, 14, 17, 13, 14, 15, 18, 16, 17, 18, 19 },
    { 20, 21, 23, 26, 21, 22, 24, 27, 23, 24, 25, 28, 26, 27, 28, 29 },
    { 30, 31, 33, 36, 31, 32, 34, 37, 33, 34, 35, 38, 36, 37, 38, 39 },
    { 40, 41, 43, 46, 41, 42, 44, 47, 43, 44, 45, 48, 46, 47, 48, 49 },
    { 50, 51, 53, 56, 51, 52, 54, 57, 53, 54, 55, 58, 56, 57, 58, 59 },
    { 60, 61, 63, 66, 61, 62, 64, 67, 63, 64, 65, 68, 66, 67, 68, 69 },
    { 70, 71, 73, 76, 71, 72, 74, 77, 73, 74, 75, 78, 76, 77, 78, 79 },
    { 80, 81, 83, 86, 81, 82, 84, 87, 83, 84, 85, 88, 86, 87, 88, 89 },
    { 90, 91, 93, 96, 91, 92, 94, 97, 93, 94, 95, 98, 96, 97, 98, 99 }
  };
  return map[row][col];
}

__host__ __device__ inline int kab_sum_index(int row, int col) {
  constexpr int map[16][16] = {
    { 0, 1, 3, 6, 10, 11, 13, 16, 30, 31, 33, 36, 60, 61, 63, 66 },
    { 1, 2, 4, 7, 11, 12, 14, 17, 31, 32, 34, 37, 61, 62, 64, 67 },
    { 3, 4, 5, 8, 13, 14, 15, 18, 33, 34, 35, 38, 63, 64, 65, 68 },
    { 6, 7, 8, 9, 16, 17, 18, 19, 36, 37, 38, 39, 66, 67, 68, 69 },
    { 10, 11, 13, 16, 20, 21, 23, 26, 40, 41, 43, 46, 70, 71, 73, 76 },
    { 11, 12, 14, 17, 21, 22, 24, 27, 41, 42, 44, 47, 71, 72, 74, 77 },
    { 13, 14, 15, 18, 23, 24, 25, 28, 43, 44, 45, 48, 73, 74, 75, 78 },
    { 16, 17, 18, 19, 26, 27, 28, 29, 46, 47, 48, 49, 76, 77, 78, 79 },
    { 30, 31, 33, 36, 40, 41, 43, 46, 50, 51, 53, 56, 80, 81, 83, 86 },
    { 31, 32, 34, 37, 41, 42, 44, 47, 51, 52, 54, 57, 81, 82, 84, 87 },
    { 33, 34, 35, 38, 43, 44, 45, 48, 53, 54, 55, 58, 83, 84, 85, 88 },
    { 36, 37, 38, 39, 46, 47, 48, 49, 56, 57, 58, 59, 86, 87, 88, 89 },
    { 60, 61, 63, 66, 70, 71, 73, 76, 80, 81, 83, 86, 90, 91, 93, 96 },
    { 61, 62, 64, 67, 71, 72, 74, 77, 81, 82, 84, 87, 91, 92, 94, 97 },
    { 63, 64, 65, 68, 73, 74, 75, 78, 83, 84, 85, 88, 93, 94, 95, 98 },
    { 66, 67, 68, 69, 76, 77, 78, 79, 86, 87, 88, 89, 96, 97, 98, 99 }
  };
  return map[row][col];
}
__device__ __constant__ int c_jindex[256];
static int jindex_ready = 0;
static std::array<int,256> host_jindex;
static bool host_jindex_ready = false;
static int verbose = 0; static int verbose_inited = 0;
static int csv_enabled = 0; static int csv_inited = 0;
static int prof_collect = 0; static int prof_inited = 0;
static int prof_env_requested = 0;
static int th_ll MOPAC_UNUSED = 64;
static int th_lh MOPAC_UNUSED = 32;
static int th_hh MOPAC_UNUSED = 16;
static int th_inited MOPAC_UNUSED = 0;
static int verify_fock_inited = 0;
static int verify_fock_enabled = 0;
// Profiling accumulators
static long long prof_atoms = 0;
static long long prof_ll_pairs = 0, prof_lh_pairs = 0, prof_hh_pairs = 0;
static double prof_total_ms = 0.0, prof_ll_ms = 0.0, prof_lh_ms = 0.0, prof_hh_ms = 0.0;

static bool resident_debug_enabled_local() {
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

__host__ __device__ inline int span_count(int first, int last) {
  return mopac_gpu::span_count(first, last);
}

__host__ __device__ inline int pair_count(int span) {
  return mopac_gpu::pair_count(span);
}

__host__ __device__ inline int ifact_val(int n) {
  return (n * (n - 1)) / 2;
}

static inline void ensure_verbose() {
  if (!verbose_inited) {
    const char* v = std::getenv("MOPAC_GPU_VERBOSE");
    if (v && (std::strcmp(v, "1") == 0 || std::strcmp(v, "on") == 0 || std::strcmp(v, "true") == 0)) verbose = 1;
    verbose_inited = 1;
  }
}
static inline void ensure_thresholds() {
  if (!th_inited) {
    const char* s;
    s = std::getenv("MOPAC_GPU_TH_LL"); if (s) th_ll = std::max(1, atoi(s));
    s = std::getenv("MOPAC_GPU_TH_LH"); if (s) th_lh = std::max(1, atoi(s));
    s = std::getenv("MOPAC_GPU_TH_HH"); if (s) th_hh = std::max(1, atoi(s));
    th_inited = 1;
  }
}
static inline void ensure_csv() {
  if (!csv_inited) {
    const char* s = std::getenv("MOPAC_GPU_CSV");
    if (s && (std::strcmp(s,"1")==0 || std::strcmp(s,"on")==0 || std::strcmp(s,"true")==0)) csv_enabled = 1;
    csv_inited = 1;
  }
}
static inline void ensure_profile_collect() {
  if (!prof_inited) {
    const char* s = std::getenv("MOPAC_GPU_PROFILE");
    if (s && *s) {
      prof_env_requested = 1;
      if (std::strcmp(s, "0") == 0 || std::strcmp(s, "off") == 0 || std::strcmp(s, "false") == 0) {
        prof_collect = 0;
      } else {
        prof_collect = 1;
      }
    } else {
      prof_collect = 1;
    }
    prof_inited = 1;
  }
}

static inline bool fock_verification_enabled() {
  if (!verify_fock_inited) {
    const char* s = std::getenv("MOPAC_GPU_VERIFY_FOCK");
    if (s && *s) {
      if (!(std::strcmp(s, "0") == 0 || std::strcmp(s, "off") == 0 ||
            std::strcmp(s, "false") == 0 || std::strcmp(s, "n") == 0 ||
            std::strcmp(s, "N") == 0)) {
        verify_fock_enabled = 1;
      }
    }
    verify_fock_inited = 1;
  }
  return verify_fock_enabled != 0;
}

static inline bool ensure_buf_int(int **ptr, size_t *cap_elems, size_t need_elems) {
  if (*cap_elems < need_elems) {
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr; *cap_elems = 0;
    if (need_elems > 0) {
      if (cudaMalloc((void**)ptr, sizeof(int) * need_elems) != cudaSuccess) return false;
      *cap_elems = need_elems;
    }
  }
  return true;
}
static inline bool ensure_buf_double(double **ptr, size_t *cap_elems, size_t need_elems) {
  if (*cap_elems < need_elems) {
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr; *cap_elems = 0;
    if (need_elems > 0) {
      if (cudaMalloc((void**)ptr, sizeof(double) * need_elems) != cudaSuccess) return false;
      *cap_elems = need_elems;
    }
  }
  return true;
}

__host__ __device__ inline int packed_index_zero(int a, int b);
static void ensure_jindex_device();

static inline size_t packed_index_host(int a, int b) {
  return static_cast<size_t>(packed_index_zero(a, b));
}

static inline void host_pair_light_light(int ia, int ja,
                                         const double *ptot, const double *p,
                                         const double *w_block,
                                         double *f_host) {
  if (!w_block) return;
  size_t ii = packed_index_host(ia, ia);
  size_t jj = packed_index_host(ja, ja);
  size_t ij = packed_index_host(std::max(ia, ja), std::min(ia, ja));
  double val = w_block[0];
  f_host[ii] += val * ptot[jj];
  f_host[jj] += val * ptot[ii];
  f_host[ij] -= val * p[ij];
}

static inline void host_pair_general(int ia, int ib, int ja, int jb,
                                     const double *ptot, const double *p,
                                     const double *w_block,
                                     double *f_host) {
  if (!w_block) return;
  int span_i = span_count(ia, ib);
  int span_j = span_count(ja, jb);
  if (span_i <= 0 || span_j <= 0) return;

  if (ia > ja) {
    size_t kr = 0;
    for (int i = ia; i <= ib; ++i) {
      for (int j = ia; j <= i; ++j) {
        double aa = (i == j) ? 1.0 : 2.0;
        size_t ij = packed_index_host(i, j);
        for (int k = ja; k <= jb; ++k) {
          for (int l = ja; l <= k; ++l) {
            double bb = (k == l) ? 1.0 : 2.0;
            size_t kl = packed_index_host(k, l);
            double a = w_block[kr++];
            f_host[ij] += bb * a * ptot[kl];
            f_host[kl] += aa * a * ptot[ij];
            double exch = a * aa * bb * 0.25;
            size_t ik = packed_index_host(i, k);
            size_t il = packed_index_host(i, l);
            size_t jk = packed_index_host(j, k);
            size_t jl = packed_index_host(j, l);
            f_host[ik] -= exch * p[jl];
            f_host[il] -= exch * p[jk];
            f_host[jk] -= exch * p[il];
            f_host[jl] -= exch * p[ik];
          }
        }
      }
    }
  } else {
    int nn = pair_count(span_j);
    if (nn <= 0) return;
    int n1 = 0;
    for (int i = ja; i <= jb; ++i) {
      for (int j = ja; j <= i; ++j) {
        ++n1;
        double aa = (i == j) ? 1.0 : 2.0;
        size_t ij = packed_index_host(i, j);
        int n2 = 0;
        for (int k = ia; k <= ib; ++k) {
          for (int l = ia; l <= k; ++l) {
            ++n2;
            double bb = (k == l) ? 1.0 : 2.0;
            size_t kl = packed_index_host(k, l);
            size_t idx = static_cast<size_t>(n2 - 1) * static_cast<size_t>(nn)
                       + static_cast<size_t>(n1 - 1);
            double a = w_block[idx];
            f_host[ij] += bb * a * ptot[kl];
            f_host[kl] += aa * a * ptot[ij];
            double exch = a * aa * bb * 0.25;
            size_t ik = packed_index_host(i, k);
            size_t il = packed_index_host(i, l);
            size_t jk = packed_index_host(j, k);
            size_t jl = packed_index_host(j, l);
            f_host[ik] -= exch * p[jl];
            f_host[il] -= exch * p[jk];
            f_host[jk] -= exch * p[il];
            f_host[jl] -= exch * p[ik];
          }
        }
      }
    }
  }
}

static inline void host_pair_heavy_light(int heavy_start, int heavy_end, int light_atom,
                                         const double *ptot, const double *p,
                                         const double *w_block,
                                         double *f_host) {
  if (!w_block || !ptot || !p || !f_host) return;
  if (!host_jindex_ready) ensure_jindex_device();
  int span = heavy_end - heavy_start + 1;
  if (span <= 0) return;
  int coulomb_len = span * (span + 1) / 2;
  size_t ll = packed_index_host(light_atom, light_atom);
  double ptot_ll = ptot[ll];
  double sumoff = 0.0;
  double sumdia = 0.0;
  int offset = 0;
  for (int rel = 0; rel < span; ++rel) {
    int orb_i = heavy_start + rel;
    if (rel > 0) {
      for (int relj = 0; relj < rel; ++relj) {
        int orb_j = heavy_start + relj;
        size_t idx = packed_index_host(orb_i, orb_j);
        double val = (offset < coulomb_len) ? w_block[offset] : 0.0;
        offset++;
        f_host[idx] += ptot_ll * val;
        sumoff += ptot[idx] * val;
      }
    }
    double val = (offset < coulomb_len) ? w_block[offset] : 0.0;
    offset++;
    size_t idx_ii = packed_index_host(orb_i, orb_i);
    f_host[idx_ii] += ptot_ll * val;
    sumdia += ptot[idx_ii] * val;
  }
  f_host[ll] += sumoff * 2.0 + sumdia;

  // Heavy-light integrals are stored in fixed 10-entry blocks (Fortran legacy layout).
  const int heavy_light_block_len = 10;
  int table_index = 0;
  for (int rel = 0; rel < span; ++rel) {
    int orb_i = heavy_start + rel;
    size_t idx_il = packed_index_host(orb_i, light_atom);
    double acc = 0.0;
    for (int relj = 0; relj < span; ++relj) {
      int map = host_jindex[table_index + relj];
      if (map <= 0 || map > heavy_light_block_len) continue;
      double wij = w_block[map - 1];
      int orb_j = heavy_start + relj;
      size_t idx_pl = packed_index_host(orb_j, light_atom);
      acc += p[idx_pl] * wij;
    }
    table_index += span;
    f_host[idx_il] -= acc;
  }
}

static inline void host_pair_heavy_heavy(int ia, int ib, int ja, int jb,
                                         const double *ptot, const double *p,
                                         const double *w_block,
                                         double *f_host) {
  if (!w_block || !ptot || !p || !f_host) return;
  int span_i = span_count(ia, ib);
  int span_j = span_count(ja, jb);
  if (span_i != 4 || span_j != 4) {
    host_pair_general(ia, ib, ja, jb, ptot, p, w_block, f_host);
    return;
  }

  double p_block_a[16];
  double p_block_b[16];
  double p_cross[16];

  int idx = 0;
  for (int row = ia; row <= ib; ++row) {
    for (int col = ia; col <= ib; ++col) {
      p_block_a[idx++] = ptot[packed_index_host(row, col)];
    }
  }

  idx = 0;
  for (int row = ja; row <= jb; ++row) {
    for (int col = ja; col <= jb; ++col) {
      p_block_b[idx++] = ptot[packed_index_host(row, col)];
    }
  }

  idx = 0;
  for (int row = ia; row <= ib; ++row) {
    for (int col = ja; col <= jb; ++col) {
      size_t packed = packed_index_host(row, col);
      p_cross[idx++] = p[packed];
    }
  }

  double suma[10];
  double sumb[10];
  for (int row = 0; row < 10; ++row) {
    double sa = 0.0;
    double sb = 0.0;
    for (int col = 0; col < 16; ++col) {
      sa += p_block_a[col] * w_block[jab_suma_index(row, col)];
      sb += p_block_b[col] * w_block[jab_sumb_index(row, col)];
    }
    suma[row] = sa;
    sumb[row] = sb;
  }

  int pair_idx = 0;
  for (int offset = 0; offset < span_i; ++offset) {
    int orb_a = ia + offset;
    int orb_b = ja + offset;
    for (int inner = 0; inner <= offset; ++inner) {
      int orb_a_j = ia + inner;
      int orb_b_j = ja + inner;
      size_t idx_a = packed_index_host(orb_a, orb_a_j);
      size_t idx_b = packed_index_host(orb_b, orb_b_j);
      f_host[idx_a] += sumb[pair_idx];
      f_host[idx_b] += suma[pair_idx];
      ++pair_idx;
    }
  }

  double sums[16];
  for (int row = 0; row < 16; ++row) {
    double total = 0.0;
    for (int col = 0; col < 16; ++col) {
      total += p_cross[col] * w_block[kab_sum_index(row, col)];
    }
    sums[row] = total;
  }

  int sum_idx = 0;
  if (ia > ja) {
    for (int i = ia; i <= ib; ++i) {
      for (int j = ja; j <= jb; ++j) {
        size_t pos = packed_index_host(i, j);
        f_host[pos] -= sums[sum_idx++];
      }
    }
  } else {
    for (int i = ia; i <= ib; ++i) {
      for (int j = ja; j <= jb; ++j) {
        size_t pos = packed_index_host(j, i);
        f_host[pos] -= sums[sum_idx++];
      }
    }
  }
}

static inline void host_pair_periodic(int ia, int ib, int ja, int jb,
                                      const double *ptot, const double *p,
                                      const double *wj_block, const double *wk_block,
                                      double *f_host) {
  if (!wj_block || !wk_block || !ptot || !p || !f_host) return;
  size_t idx = 0;
  for (int i = ia; i <= ib; ++i) {
    for (int j = ia; j <= i; ++j) {
      double aa = (i == j) ? 1.0 : 2.0;
      size_t ij = packed_index_host(i, j);
      for (int k = ja; k <= jb; ++k) {
        for (int l = ja; l <= k; ++l) {
          double bb = (k == l) ? 1.0 : 2.0;
          size_t kl = packed_index_host(k, l);
          double aj = wj_block[idx];
          double ak = wk_block[idx];
          idx++;
          if (kl > ij) continue;
          if (i == k && (aa + bb) < 2.1) {
            f_host[ij] += aj * ptot[kl];
          } else {
            f_host[ij] += bb * aj * ptot[kl];
            f_host[kl] += aa * aj * ptot[ij];
            double exch = ak * aa * bb * 0.25;
            if (i >= k && j >= l) {
              size_t ik = packed_index_host(i, k);
              size_t jl = packed_index_host(j, l);
              f_host[ik] -= exch * p[jl];
            }
            if (i >= l && j >= k) {
              size_t il = packed_index_host(i, l);
              size_t jk = packed_index_host(j, k);
              f_host[il] -= exch * p[jk];
              f_host[jk] -= exch * p[il];
            }
            if (j >= l && i >= k) {
              size_t jl = packed_index_host(j, l);
              size_t ik = packed_index_host(i, k);
              f_host[jl] -= exch * p[ik];
            }
          }
        }
      }
    }
  }
}
static inline int ifact_host(int n) {
  return (n * (n - 1)) / 2;
}

static void ensure_jindex_device() {
  if (jindex_ready) return;
  int host_idx[256];
  int m = 0;
  for (int i = 1; i <= 4; ++i) {
    for (int j = 1; j <= 4; ++j) {
      int ij = std::min(i, j);
      int ji = i + j - ij;
      for (int k = 1; k <= 4; ++k) {
        int ik = std::min(i, k);
        for (int l = 1; l <= 4; ++l) {
          int kl = std::min(k, l);
          int lk = k + l - kl;
          ++m;
          host_idx[m - 1] = (ifact_host(ji) + ij) * 10 + ifact_host(lk) + kl - 10;
        }
      }
    }
  }
  for (int t = 0; t < 256; ++t) host_jindex[t] = host_idx[t];
  host_jindex_ready = true;
  cudaMemcpyToSymbol(c_jindex, host_idx, sizeof(host_idx));
  jindex_ready = 1;
}

static inline bool ensure_pair_buffers(size_t need_pairs) {
  if (cap_pairs < need_pairs) {
    if (s_d_pair_i) cudaFree(s_d_pair_i);
    if (s_d_pair_j) cudaFree(s_d_pair_j);
    if (s_d_pair_off) cudaFree(s_d_pair_off);
    if (s_d_pair_type) cudaFree(s_d_pair_type);
    if (s_d_pair_wj_off) cudaFree(s_d_pair_wj_off);
    if (s_d_pair_wk_off) cudaFree(s_d_pair_wk_off);
    s_d_pair_i = s_d_pair_j = s_d_pair_off = nullptr;
    s_d_pair_type = s_d_pair_wj_off = s_d_pair_wk_off = nullptr;
    cap_pairs = 0;
    if (need_pairs > 0) {
      if (cudaMalloc((void**)&s_d_pair_i, sizeof(int) * need_pairs) != cudaSuccess) return false;
      if (cudaMalloc((void**)&s_d_pair_j, sizeof(int) * need_pairs) != cudaSuccess) return false;
      if (cudaMalloc((void**)&s_d_pair_off, sizeof(int) * need_pairs) != cudaSuccess) return false;
      if (cudaMalloc((void**)&s_d_pair_type, sizeof(int) * need_pairs) != cudaSuccess) return false;
      if (cudaMalloc((void**)&s_d_pair_wj_off, sizeof(int) * need_pairs) != cudaSuccess) return false;
      if (cudaMalloc((void**)&s_d_pair_wk_off, sizeof(int) * need_pairs) != cudaSuccess) return false;
      cap_pairs = need_pairs;
    }
  }
  return true;
}

__host__ __device__ inline int packed_index_zero(int a, int b) {
  return static_cast<int>(mopac_gpu::packed_index(a, b));
}

__device__ void fock_pair_general(int ia, int ib, int ja, int jb,
                                 const double *ptot, const double *p,
                                 const double *w, double *f, int dbg_tid = -1) {
  if (!w || !ptot || !p || !f) {
    if (dbg_tid >= 0 && dbg_tid < 2) {
      printf("[GPU GEN] null ptr check w=%p ptot=%p p=%p f=%p\n",
             (const void*)w, (const void*)ptot, (const void*)p, (void*)f);
    }
    return;
  }
  if (dbg_tid >= 0 && dbg_tid < 2) {
    printf("[GPU GEN] entry ptrs w=%p ptot=%p p=%p f=%p ia=%d ib=%d ja=%d jb=%d\n",
           (const void*)w, (const void*)ptot, (const void*)p, (void*)f,
           ia, ib, ja, jb);
  }
  int span_i = span_count(ia, ib);
  int span_j = span_count(ja, jb);
  if (span_i <= 0 || span_j <= 0) return;

  if (ia > ja) {
    int kr = 0;
    for (int i = ia; i <= ib; ++i) {
      for (int j = ia; j <= i; ++j) {
        double aa = (i == j) ? 1.0 : 2.0;
        int ij = packed_index_zero(i, j);
        for (int k = ja; k <= jb; ++k) {
          for (int l = ja; l <= k; ++l) {
            double bb = (k == l) ? 1.0 : 2.0;
            int kl = packed_index_zero(k, l);
            double a = w[kr++];
            atomicAdd_double(&f[ij], bb * a * ptot[kl]);
            atomicAdd_double(&f[kl], aa * a * ptot[ij]);
            double exch = a * aa * bb * 0.25;
            int ik = packed_index_zero(i, k);
            int il = packed_index_zero(i, l);
            int jk = packed_index_zero(j, k);
            int jl = packed_index_zero(j, l);
            atomicAdd_double(&f[ik], -exch * p[jl]);
            atomicAdd_double(&f[il], -exch * p[jk]);
            atomicAdd_double(&f[jk], -exch * p[il]);
            atomicAdd_double(&f[jl], -exch * p[ik]);
          }
        }
      }
    }
  } else {
    int nn = pair_count(span_j);
    if (nn <= 0) return;
    int n1 = 0;
    for (int i = ja; i <= jb; ++i) {
      for (int j = ja; j <= i; ++j) {
        ++n1;
        double aa = (i == j) ? 1.0 : 2.0;
        int ij = packed_index_zero(i, j);
        int n2 = 0;
        for (int k = ia; k <= ib; ++k) {
          for (int l = ia; l <= k; ++l) {
            ++n2;
            double bb = (k == l) ? 1.0 : 2.0;
            int kl = packed_index_zero(k, l);
            int idx = (n2 - 1) * nn + (n1 - 1);
            double a = w[idx];
            atomicAdd_double(&f[ij], bb * a * ptot[kl]);
            atomicAdd_double(&f[kl], aa * a * ptot[ij]);
            double exch = a * aa * bb * 0.25;
            int ik = packed_index_zero(i, k);
            int il = packed_index_zero(i, l);
            int jk = packed_index_zero(j, k);
            int jl = packed_index_zero(j, l);
            atomicAdd_double(&f[ik], -exch * p[jl]);
            atomicAdd_double(&f[il], -exch * p[jk]);
            atomicAdd_double(&f[jk], -exch * p[il]);
            atomicAdd_double(&f[jl], -exch * p[ik]);
          }
        }
      }
    }
  }
}

__device__ inline void fock_pair_light_light(int ia, int ja,
                                             const double *ptot, const double *p,
                                             const double *w, double *f,
                                             int dbg_tid) {
  if (dbg_tid >= 0 && dbg_tid < 2) {
    printf("[GPU LL] entry w=%p ptot=%p p=%p f=%p\n",
           (const void*)w, (const void*)ptot, (const void*)p, (void*)f);
  }
  if (!w || !ptot || !p || !f) {
    if (dbg_tid >= 0 && dbg_tid < 2) {
      printf("[GPU LL] null ptr abort\n");
    }
    return;
  }
  double val = w[0];
  int ii = packed_index_zero(ia, ia);
  int jj = packed_index_zero(ja, ja);
  int ij = (ia >= ja) ? packed_index_zero(ia, ja)
                      : packed_index_zero(ja, ia);
  double contrib_ii = val * ptot[jj];
  double contrib_jj = val * ptot[ii];
  double contrib_ij = -val * p[ij];
  f[ii] += contrib_ii;
  f[jj] += contrib_jj;
  f[ij] += contrib_ij;
  if (dbg_tid >= 0 && dbg_tid < 2) {
    printf("[GPU LL] tid=%d val=% .5e ptrs(w=%p ptot=%p p=%p f=%p) ii=%d jj=%d ij=%d contribs=% .5e % .5e % .5e\n",
           dbg_tid, val, (const void*)w, (const void*)ptot, (const void*)p, (void*)f,
           ii, jj, ij, contrib_ii, contrib_jj, contrib_ij);
    printf("[GPU LL] after f[ii]=% .5e f[jj]=% .5e f[ij]=% .5e\n",
           f[ii], f[jj], f[ij]);
  }
}

__device__ inline void fock_pair_heavy_light(int heavy_start, int heavy_end, int light_atom,
                                             const double *ptot, const double *p,
                                             const double *w_block,
                                             double *f,
                                             int dbg_tid) {
  if (!w_block || !ptot || !p || !f) return;
  int span = heavy_end - heavy_start + 1;
  if (span <= 0) return;

  int coulomb_len = span * (span + 1) / 2;
  int ll = packed_index_zero(light_atom, light_atom);
  double ptot_ll = ptot[ll];
  // Coulomb contribution: mirror CPU order
  int wpos = 0;
  double sumoff = 0.0;
  double sumdia = 0.0;
  for (int rel = 0; rel < span; ++rel) {
    int orb_i = heavy_start + rel;
    for (int relj = 0; relj < rel; ++relj) {
      int orb_j = heavy_start + relj;
      double val = (wpos < coulomb_len) ? w_block[wpos] : 0.0;
      wpos++;
      int idx = packed_index_zero(orb_i, orb_j);
      atomicAdd_double(&f[idx], ptot_ll * val);
      sumoff += ptot[idx] * val;
    }
    double diag = (wpos < coulomb_len) ? w_block[wpos] : 0.0;
    wpos++;
    int idx_ii = packed_index_zero(orb_i, orb_i);
    atomicAdd_double(&f[idx_ii], ptot_ll * diag);
    sumdia += ptot[idx_ii] * diag;
  }
  atomicAdd_double(&f[ll], 2.0 * sumoff + sumdia);

  // Exchange contraction using jindex table
  // Heavy-light integrals are stored in fixed 10-entry blocks (Fortran legacy layout).
  constexpr int heavy_light_block_len = 10;
  int table_index = 0;
  for (int rel = 0; rel < span; ++rel) {
    int orb_i = heavy_start + rel;
    int idx_il = packed_index_zero(orb_i, light_atom);
    double acc = 0.0;
    for (int relj = 0; relj < span; ++relj) {
      int map = c_jindex[table_index + relj];
      if (map <= 0 || map > heavy_light_block_len) continue;
      double val = w_block[map - 1];
      int orb_j = heavy_start + relj;
      int idx_pl = packed_index_zero(orb_j, light_atom);
      acc += p[idx_pl] * val;
    }
    table_index += span;
    atomicAdd_double(&f[idx_il], -acc);
    if (dbg_tid >= 0 && dbg_tid < 2) {
      printf("[GPU heavy-light] rel=%d acc=% .5e\n", rel, acc);
    }
  }
}

__device__ void fock_pair_heavy_heavy(int ia, int ib, int ja, int jb,
                                      const double *ptot, const double *p,
                                      const double *w_block,
                                      double *f) {
  if (!w_block || !ptot || !p || !f) return;
  int span_i = span_count(ia, ib);
  int span_j = span_count(ja, jb);
  if (span_i != 4 || span_j != 4) {
    fock_pair_general(ia, ib, ja, jb, ptot, p, w_block, f);
    return;
  }

  double p_block_a[16];
  double p_block_b[16];
  double p_cross[16];

  int idx = 0;
  for (int row = ia; row <= ib; ++row) {
    for (int col = ia; col <= ib; ++col) {
      p_block_a[idx++] = ptot[packed_index_zero(row, col)];
    }
  }

  idx = 0;
  for (int row = ja; row <= jb; ++row) {
    for (int col = ja; col <= jb; ++col) {
      p_block_b[idx++] = ptot[packed_index_zero(row, col)];
    }
  }

  idx = 0;
  for (int row = ia; row <= ib; ++row) {
    for (int col = ja; col <= jb; ++col) {
      int packed = (row >= col) ? packed_index_zero(row, col)
                                : packed_index_zero(col, row);
      p_cross[idx++] = p[packed];
    }
  }

  double suma[10];
  double sumb[10];
  for (int row = 0; row < 10; ++row) {
    double sa = 0.0;
    double sb = 0.0;
    for (int col = 0; col < 16; ++col) {
      sa += p_block_a[col] * w_block[jab_suma_index(row, col)];
      sb += p_block_b[col] * w_block[jab_sumb_index(row, col)];
    }
    suma[row] = sa;
    sumb[row] = sb;
  }

  int pair_idx = 0;
  for (int offset = 0; offset < span_i; ++offset) {
    int orb_a = ia + offset;
    int orb_b = ja + offset;
    for (int inner = 0; inner <= offset; ++inner) {
      int orb_a_j = ia + inner;
      int orb_b_j = ja + inner;
      int idx_a = packed_index_zero(orb_a, orb_a_j);
      int idx_b = packed_index_zero(orb_b, orb_b_j);
      atomicAdd_double(&f[idx_a], sumb[pair_idx]);
      atomicAdd_double(&f[idx_b], suma[pair_idx]);
      ++pair_idx;
    }
  }

  double sums[16];
  for (int row = 0; row < 16; ++row) {
    double total = 0.0;
    for (int col = 0; col < 16; ++col) {
      total += p_cross[col] * w_block[kab_sum_index(row, col)];
    }
    sums[row] = total;
  }

  int sum_idx = 0;
  if (ia > ja) {
    for (int i = ia; i <= ib; ++i) {
      for (int j = ja; j <= jb; ++j) {
        int pos = packed_index_zero(i, j);
        atomicAdd_double(&f[pos], -sums[sum_idx++]);
      }
    }
  } else {
    for (int i = ia; i <= ib; ++i) {
      for (int j = ja; j <= jb; ++j) {
        int pos = packed_index_zero(j, i);
        atomicAdd_double(&f[pos], -sums[sum_idx++]);
      }
    }
  }
}

__device__ void fock_pair_periodic(int ia, int ib, int ja, int jb,
                                   const double *ptot, const double *p,
                                   const double *wj_block, const double *wk_block,
                                   double *f) {
  if (!wj_block || !wk_block) return;
  int span_i = span_count(ia, ib);
  int span_j = span_count(ja, jb);
  if (span_i <= 0 || span_j <= 0) return;
  size_t idx = 0;
  for (int i = ia; i <= ib; ++i) {
    for (int j = ia; j <= i; ++j) {
      double aa = (i == j) ? 1.0 : 2.0;
      int ij = packed_index_zero(i, j);
      for (int k = ja; k <= jb; ++k) {
        for (int l = ja; l <= k; ++l) {
          double bb = (k == l) ? 1.0 : 2.0;
          int kl = packed_index_zero(k, l);
          double aj = wj_block[idx];
          double ak = wk_block[idx];
          idx++;
          if (kl > ij) continue;
          int ik = (i >= k) ? packed_index_zero(i, k) : -1;
          int il = (i >= l) ? packed_index_zero(i, l) : -1;
          int jk = (j >= k) ? packed_index_zero(j, k) : -1;
          int jl = (j >= l) ? packed_index_zero(j, l) : -1;
          if (i == k && (aa + bb) < 2.1) {
            atomicAdd_double(&f[ij], aj * ptot[kl]);
          } else {
            atomicAdd_double(&f[ij], bb * aj * ptot[kl]);
            atomicAdd_double(&f[kl], aa * aj * ptot[ij]);
            double exch = ak * aa * bb * 0.25;
            if (jl >= 0 && ik >= 0) atomicAdd_double(&f[ik], -exch * p[jl]);
            if (jk >= 0 && il >= 0) atomicAdd_double(&f[il], -exch * p[jk]);
            if (jk >= 0 && il >= 0) atomicAdd_double(&f[jk], -exch * p[il]);
            if (jl >= 0 && ik >= 0) atomicAdd_double(&f[jl], -exch * p[ik]);
          }
        }
      }
    }
  }
}

__global__ void fock_pairs_kernel(int npairs,
                                  const int *pair_i,
                                  const int *pair_j,
                                  const int *pair_type,
                                  const int *pair_w_off,
                                  const int *pair_wj_off,
                                  const int *pair_wk_off,
                                  const int *nfirst,
                                  const int *nlast,
                                  const double *ptot,
                                  const double *p,
                                  const double *w,
                                  const double *wj,
                                  const double *wk,
                                  double *f,
                                  int debug_flag) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npairs) return;
  int ii = pair_i[tid];
  int jj = pair_j[tid];
  int ia = nfirst[ii - 1];
  int ib = nlast[ii - 1];
  int ja = nfirst[jj - 1];
  int jb = nlast[jj - 1];
  if ((ib - ia) < 0 || (jb - ja) < 0) return;

  int type = pair_type[tid];
  const double *w_block = (w && pair_w_off) ? (w + pair_w_off[tid]) : nullptr;
  const double *wj_block = (wj && pair_wj_off) ? (wj + pair_wj_off[tid]) : nullptr;
  const double *wk_block = (wk && pair_wk_off) ? (wk + pair_wk_off[tid]) : nullptr;

  if (debug_flag && tid < 5) {
    printf("[GPU resident debug] kernel tid=%d ii=%d jj=%d type=%d ia=%d ib=%d ja=%d jb=%d\n",
           tid, ii, jj, type, ia, ib, ja, jb);
  }

  switch (type) {
    case PAIR_LIGHT_LIGHT:
      fock_pair_light_light(ia, ja, ptot, p, w_block, f, debug_flag ? tid : -1);
      break;
    case PAIR_HEAVY_LIGHT:
      fock_pair_heavy_light(ia, ib, ja, ptot, p, w_block, f, debug_flag ? tid : -1);
      break;
    case PAIR_LIGHT_HEAVY:
      fock_pair_heavy_light(ja, jb, ia, ptot, p, w_block, f, debug_flag ? tid : -1);
      break;
    case PAIR_HEAVY_HEAVY:
      fock_pair_heavy_heavy(ia, ib, ja, jb, ptot, p, w_block, f);
      break;
    case PAIR_GENERAL:
      fock_pair_general(ia, ib, ja, jb, ptot, p, w_block, f, debug_flag ? tid : -1);
      break;
    case PAIR_PERIODIC:
      fock_pair_periodic(ia, ib, ja, jb, ptot, p, wj_block, wk_block, f);
      break;
    default:
      fock_pair_general(ia, ib, ja, jb, ptot, p, w_block, f, debug_flag ? tid : -1);
      break;
  }
}

// ================= Device-resident gradient buffers and ops =================
extern "C" {

void mopac_cuda_cart_gradient_release(void);

static double *g_lastF_dev = nullptr;
static size_t g_lastF_bytes MOPAC_UNUSED = 0;
static int g_lastF_n = 0;
static cublasHandle_t g_blas_local = nullptr;
static cudaStream_t g_stream_local = nullptr;

static inline void ensure_local_handles() {
  if (!g_blas_local) {
    cublasCreate(&g_blas_local);
    if (!g_stream_local) cudaStreamCreate(&g_stream_local);
    cublasSetStream(g_blas_local, g_stream_local);
  }
}

// Unpack packed lower-triangular matrix (length n*(n+1)/2) to full n x n (column-major)
__global__ void unpack_lower_to_full_kernel(const double *packed, double *full, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total = n * n;
  if (tid >= total) return;
  int r = tid % n;     // row index [0..n-1]
  int c = tid / n;     // col index [0..n-1]
  size_t idx;
  if (r >= c) {
    idx = (size_t)c + ((size_t)r * ((size_t)r + 1)) / 2;
  } else {
    idx = (size_t)r + ((size_t)c * ((size_t)c + 1)) / 2;
  }
  full[(size_t)r + (size_t)c * (size_t)n] = packed[idx];
}

bool mopac_cuda_fock2_keep(int norbs, int mpack, int numat,
                           const int *nfirst, const int *nlast,
                           const double *ptot, const double *p,
                           const double *w, int nati) {
  ensure_verbose();
  int ia = nfirst[nati - 1];
  int ib = nlast[nati - 1];
  if ((ib - ia) < 0) return false;
  int span_i = span_count(ia, ib);
  if (span_i <= 0) return true;

  const size_t max_index = static_cast<size_t>(std::numeric_limits<int>::max());

  std::vector<int> pair_i;
  std::vector<int> pair_j;
  std::vector<int> pair_off;
  pair_i.reserve(std::max(0, nati - 1));

  size_t w_len = 0;
  int pairs_i = pair_count(span_i);
  for (int jj = 1; jj < nati; ++jj) {
    int ja = nfirst[jj - 1];
    int jb = nlast[jj - 1];
    int span_j = span_count(ja, jb);
    if (span_j <= 0) continue;
    int pairs_j = pair_count(span_j);
    int chunk = pairs_i * pairs_j;
    if (chunk <= 0) continue;
    if (pair_i.size() >= max_index) return false;
    if (w_len > max_index) return false;
    pair_i.push_back(nati);
    pair_j.push_back(jj);
    pair_off.push_back(static_cast<int>(w_len));
    w_len += static_cast<size_t>(chunk);
  }

  if (w_len == 0) return true;

  size_t atoms_e = (size_t)numat;
  size_t mpack_e = (size_t)mpack;
  if (!ensure_buf_int(&s_d_nf, &cap_nf, atoms_e)) return false;
  if (!ensure_buf_int(&s_d_nl, &cap_nl, atoms_e)) return false;
  if (!ensure_buf_double(&s_d_ptot, &cap_ptot, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_p, &cap_p, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_f, &cap_f, mpack_e)) return false;
  if (pair_i.size() > max_index) return false;
  if (!ensure_buf_double(&s_d_w, &cap_w, w_len)) return false;
  if (!ensure_pair_buffers(pair_i.size())) return false;

  if (cudaMemset(s_d_f, 0, sizeof(double)*mpack_e) != cudaSuccess) return false;

  cudaMemcpy(s_d_nf, nfirst, sizeof(int)*atoms_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_nl, nlast, sizeof(int)*atoms_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_ptot, ptot, sizeof(double)*mpack_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_p, p, sizeof(double)*mpack_e, cudaMemcpyHostToDevice);
  if (w_len > 0) {
    cudaMemcpy(s_d_w, w, sizeof(double)*w_len, cudaMemcpyHostToDevice);
    if (resident_debug_enabled_local()) {
      size_t limit = std::min(w_len, (size_t)5);
      std::vector<double> host_w(limit);
      cudaMemcpy(host_w.data(), s_d_w, sizeof(double)*limit, cudaMemcpyDeviceToHost);
      std::printf("[GPU resident debug] w sample:");
      for (size_t idx = 0; idx < limit; ++idx) {
        std::printf(" % .5e", host_w[idx]);
      }
      std::printf("\n");
      std::fflush(stdout);
    }
  }
  if (!pair_i.empty()) {
    std::vector<int> pair_type(pair_i.size(), PAIR_GENERAL);
    std::vector<int> zeros(pair_i.size(), 0);
    cudaMemcpy(s_d_pair_i, pair_i.data(), sizeof(int)*pair_i.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_j, pair_j.data(), sizeof(int)*pair_j.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_off, pair_off.data(), sizeof(int)*pair_off.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_type, pair_type.data(), sizeof(int)*pair_type.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_wj_off, zeros.data(), sizeof(int)*zeros.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_wk_off, zeros.data(), sizeof(int)*zeros.size(), cudaMemcpyHostToDevice);
    int threads = 64;
    int blocks = static_cast<int>((pair_i.size() + threads - 1) / threads);
    int debug_flag = resident_debug_enabled_local() ? 1 : 0;
    fock_pairs_kernel<<<blocks, threads>>>(static_cast<int>(pair_i.size()),
                                           s_d_pair_i, s_d_pair_j, s_d_pair_type,
                                           s_d_pair_off, s_d_pair_wj_off, s_d_pair_wk_off,
                                           s_d_nf, s_d_nl,
                                           s_d_ptot, s_d_p,
                                           s_d_w, nullptr, nullptr,
                                           s_d_f, debug_flag);
    cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) return false;
  }

  g_lastF_dev = s_d_f; g_lastF_bytes = sizeof(double)*mpack_e; g_lastF_n = norbs;
  return true;
}

void mopac_cuda_fmulC_from_dev(int n, const double *C, int ldc, double *W, int ldw) {
  if (!g_lastF_dev || g_lastF_n != n) {
    // Zero W if no resident F
    size_t nn = (size_t)n * (size_t)n;
    for (size_t i = 0; i < nn; ++i) W[i] = 0.0;
    return;
  }
  ensure_local_handles();
  size_t bytesN = sizeof(double) * (size_t)n * (size_t)n;
  double *dC=nullptr, *dW=nullptr;
  if (cudaMalloc((void**)&dC, bytesN) != cudaSuccess) return;
  if (cudaMalloc((void**)&dW, bytesN) != cudaSuccess) { cudaFree(dC); return; }
  cudaMemcpyAsync(dC, C, bytesN, cudaMemcpyHostToDevice, g_stream_local);
  // Unpack device-resident packed-lower F to full matrix before GEMM
  double *dFfull = nullptr;
  if (cudaMalloc((void**)&dFfull, bytesN) != cudaSuccess) { cudaFree(dC); cudaFree(dW); return; }
  {
    int total = n * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    unpack_lower_to_full_kernel<<<grid, block, 0, g_stream_local>>>(g_lastF_dev, dFfull, n);
  }
  double alpha=1.0, beta=0.0;
  cublasDgemm(g_blas_local, CUBLAS_OP_N, CUBLAS_OP_N,
              n, n, n, &alpha,
              dFfull, n,
              dC, ldc,
              &beta,
              dW, ldw);
  cudaMemcpyAsync(W, dW, bytesN, cudaMemcpyDeviceToHost, g_stream_local);
  cudaStreamSynchronize(g_stream_local);
  cudaFree(dC); cudaFree(dW); cudaFree(dFfull);
}

void mopac_cuda_grad_buffers_release() {
  // Print end-of-run profiling summary if enabled
  if ((verbose || prof_env_requested) && prof_atoms > 0) {
    double avg_ms = prof_total_ms / (double)prof_atoms;
    printf("GPU grad summary: atoms=%lld total_ms=%.3f avg_ms=%.3f LL_pairs=%lld LH_pairs=%lld HH_pairs=%lld LL_ms=%.3f LH_ms=%.3f HH_ms=%.3f\n",
           prof_atoms, prof_total_ms, avg_ms, prof_ll_pairs, prof_lh_pairs, prof_hh_pairs, prof_ll_ms, prof_lh_ms, prof_hh_ms);
  }
  if (csv_enabled && prof_atoms > 0) {
    // CSV-style summary
    printf("summary,atoms,%lld,total_ms,%.6f,avg_ms,%.6f,LL_pairs,%lld,LH_pairs,%lld,HH_pairs,%lld,LL_ms,%.6f,LH_ms,%.6f,HH_ms,%.6f\n",
           prof_atoms, prof_total_ms, prof_total_ms/(double)prof_atoms,
           prof_ll_pairs, prof_lh_pairs, prof_hh_pairs,
           prof_ll_ms, prof_lh_ms, prof_hh_ms);
  }
  // Reset accumulators
  prof_atoms = 0; prof_total_ms = 0.0;
  prof_ll_pairs = 0; prof_lh_pairs = 0; prof_hh_pairs = 0;
  prof_ll_ms = 0.0; prof_lh_ms = 0.0; prof_hh_ms = 0.0;
  if (s_d_nf) cudaFree(s_d_nf); s_d_nf = nullptr; cap_nf = 0;
  if (s_d_nl) cudaFree(s_d_nl); s_d_nl = nullptr; cap_nl = 0;
  if (s_d_ptot) cudaFree(s_d_ptot); s_d_ptot = nullptr; cap_ptot = 0;
  if (s_d_p) cudaFree(s_d_p); s_d_p = nullptr; cap_p = 0;
  if (s_d_w) cudaFree(s_d_w); s_d_w = nullptr; cap_w = 0;
  if (s_d_wj) cudaFree(s_d_wj); s_d_wj = nullptr; cap_wj = 0;
  if (s_d_wk) cudaFree(s_d_wk); s_d_wk = nullptr; cap_wk = 0;
  if (s_d_f) cudaFree(s_d_f); s_d_f = nullptr; cap_f = 0;
  if (s_d_pair_i) cudaFree(s_d_pair_i); s_d_pair_i = nullptr;
  if (s_d_pair_j) cudaFree(s_d_pair_j); s_d_pair_j = nullptr;
  if (s_d_pair_off) cudaFree(s_d_pair_off); s_d_pair_off = nullptr;
  if (s_d_pair_type) cudaFree(s_d_pair_type); s_d_pair_type = nullptr;
  if (s_d_pair_wj_off) cudaFree(s_d_pair_wj_off); s_d_pair_wj_off = nullptr;
  if (s_d_pair_wk_off) cudaFree(s_d_pair_wk_off); s_d_pair_wk_off = nullptr;
  cap_pairs = 0;
  g_lastF_dev = nullptr; g_lastF_bytes = 0; g_lastF_n = 0;
  if (g_blas_local) { cublasDestroy(g_blas_local); g_blas_local = nullptr; }
  if (g_stream_local) { cudaStreamDestroy(g_stream_local); g_stream_local = nullptr; }
  mopac_cuda_cart_gradient_release();
}

} // extern "C"

// =============== SCF Fock (J/K) on GPU (experimental) ===============
extern "C" {

bool mopac_cuda_fock2_scf(int norbs, int mpack, int numat,
                          const int *nfirst, const int *nlast,
                          const double *ptot, const double *p,
                          const double *w, const double *wj, const double *wk,
                          int periodic_flag,
                          double *fout) {
  ensure_verbose(); ensure_thresholds(); ensure_csv(); ensure_profile_collect();
  ensure_jindex_device();

  const bool periodic = (periodic_flag != 0);
  size_t atoms_e = static_cast<size_t>(numat);
  size_t mpack_e = static_cast<size_t>(mpack);

  std::vector<int> pair_i;
  std::vector<int> pair_j;
  std::vector<int> pair_type;
  std::vector<int> pair_w_off;
  std::vector<int> pair_wj_off;
  std::vector<int> pair_wk_off;
  pair_i.reserve(std::max(1, numat));
  pair_j.reserve(std::max(1, numat));
  pair_type.reserve(std::max(1, numat));
  pair_w_off.reserve(std::max(1, numat));
  pair_wj_off.reserve(std::max(1, numat));
  pair_wk_off.reserve(std::max(1, numat));

  size_t w_len = 0;
  size_t wj_len = 0;
  size_t wk_len = 0;
  const size_t max_index = static_cast<size_t>(std::numeric_limits<int>::max());
  long long ll_pairs = 0, lh_pairs = 0, hh_pairs = 0;

  for (int ii = 1; ii <= numat; ++ii) {
    int ia = nfirst[ii - 1];
    int ib = nlast[ii - 1];
    int span_i = span_count(ia, ib);
    if (span_i <= 0) continue;
    int pairs_i = pair_count(span_i);
    for (int jj = 1; jj < ii; ++jj) {
      int ja = nfirst[jj - 1];
      int jb = nlast[jj - 1];
      int span_j = span_count(ja, jb);
      if (span_j <= 0) continue;
      int pairs_j = pair_count(span_j);
      if (pairs_i <= 0 || pairs_j <= 0) continue;

      int type = PAIR_GENERAL;
      int chunk_w = 0;
      int chunk_wj = 0;
      int chunk_wk = 0;

      if (periodic) {
        type = PAIR_PERIODIC;
        chunk_w = pairs_i * pairs_j;
        chunk_wj = chunk_w;
        chunk_wk = chunk_w;
      } else {
        bool has_d = (span_i >= 7) || (span_j >= 7);
        if (has_d) {
          type = PAIR_GENERAL;
          chunk_w = pairs_i * pairs_j;
        } else if (span_i >= 4 && span_j >= 4) {
          type = PAIR_HEAVY_HEAVY;
          chunk_w = 100;
        } else if (span_i >= 4 && span_j == 1) {
          type = PAIR_HEAVY_LIGHT;
          chunk_w = 10;
        } else if (span_j >= 4 && span_i == 1) {
          type = PAIR_LIGHT_HEAVY;
          chunk_w = 10;
        } else if (span_i == 1 && span_j == 1) {
          type = PAIR_LIGHT_LIGHT;
          chunk_w = 1;
        } else {
          type = PAIR_GENERAL;
          chunk_w = pairs_i * pairs_j;
        }
      }

      if (chunk_w < 0 || chunk_wj < 0 || chunk_wk < 0) return false;
      if (chunk_w == 0 && chunk_wj == 0 && chunk_wk == 0) continue;

      if (pair_i.size() >= max_index) return false;
      if (w_len >= max_index || w_len + static_cast<size_t>(chunk_w) > max_index) return false;
      if (wj_len >= max_index || wj_len + static_cast<size_t>(chunk_wj) > max_index) return false;
      if (wk_len >= max_index || wk_len + static_cast<size_t>(chunk_wk) > max_index) return false;

      pair_i.push_back(ii);
      pair_j.push_back(jj);
      pair_type.push_back(type);
      pair_w_off.push_back(static_cast<int>(w_len));
      pair_wj_off.push_back(static_cast<int>(wj_len));
      pair_wk_off.push_back(static_cast<int>(wk_len));

      w_len += static_cast<size_t>(chunk_w);
      wj_len += static_cast<size_t>(chunk_wj);
      wk_len += static_cast<size_t>(chunk_wk);

      if (span_i == 1 && span_j == 1) {
        ll_pairs++;
      } else if (span_i == 1 || span_j == 1) {
        lh_pairs++;
      } else {
        hh_pairs++;
      }
    }
  }

  bool want_verify = fock_verification_enabled() || resident_debug_enabled_local();
  std::vector<double> f_host;
  if (want_verify) f_host.assign(mpack_e, 0.0);
  bool unsupported_kind = false;
  if (want_verify) {
    for (size_t idx = 0; idx < pair_i.size(); ++idx) {
      int type = pair_type[idx];
      int ii = pair_i[idx];
      int jj = pair_j[idx];
      int ia = nfirst[ii - 1];
      int ib = nlast[ii - 1];
      int ja = nfirst[jj - 1];
      int jb = nlast[jj - 1];
      const double *w_host = (w && pair_w_off.size() > idx) ? (w + pair_w_off[idx]) : nullptr;
      const double *wj_host = (wj && pair_wj_off.size() > idx) ? (wj + pair_wj_off[idx]) : nullptr;
      const double *wk_host = (wk && pair_wk_off.size() > idx) ? (wk + pair_wk_off[idx]) : nullptr;
      switch (type) {
        case PAIR_LIGHT_LIGHT:
          host_pair_light_light(ia, ja, ptot, p, w_host, f_host.data());
          break;
        case PAIR_GENERAL:
          host_pair_general(ia, ib, ja, jb, ptot, p, w_host, f_host.data());
          break;
        case PAIR_HEAVY_HEAVY:
          host_pair_heavy_heavy(ia, ib, ja, jb, ptot, p, w_host, f_host.data());
          break;
        case PAIR_HEAVY_LIGHT:
          host_pair_heavy_light(ia, ib, ja, ptot, p, w_host, f_host.data());
          break;
        case PAIR_LIGHT_HEAVY:
          host_pair_heavy_light(ja, jb, ia, ptot, p, w_host, f_host.data());
          break;
        case PAIR_PERIODIC:
          host_pair_periodic(ia, ib, ja, jb, ptot, p, wj_host, wk_host, f_host.data());
          break;
        default:
          unsupported_kind = true;
          break;
      }
      if (unsupported_kind) break;
    }
  }
  if (unsupported_kind) return false;

  if (pair_i.size() > max_index) return false;

  if (!ensure_buf_int(&s_d_nf, &cap_nf, atoms_e)) return false;
  if (!ensure_buf_int(&s_d_nl, &cap_nl, atoms_e)) return false;
  if (!ensure_buf_double(&s_d_ptot, &cap_ptot, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_p, &cap_p, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_f, &cap_f, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_w, &cap_w, w_len)) return false;
  if (!ensure_buf_double(&s_d_wj, &cap_wj, wj_len)) return false;
  if (!ensure_buf_double(&s_d_wk, &cap_wk, wk_len)) return false;
  if (!ensure_pair_buffers(pair_i.size())) return false;

  cudaMemset(s_d_f, 0, sizeof(double) * mpack_e);

    cudaMemcpy(s_d_nf, nfirst, sizeof(int) * atoms_e, cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_nl, nlast, sizeof(int) * atoms_e, cudaMemcpyHostToDevice);
    if (!mopac_cuda_density_copy_cached(s_d_ptot, mpack_e, ptot)) {
      cudaMemcpy(s_d_ptot, ptot, sizeof(double) * mpack_e, cudaMemcpyHostToDevice);
    }
    if (!mopac_cuda_density_copy_cached(s_d_p, mpack_e, p)) {
      cudaMemcpy(s_d_p, p, sizeof(double) * mpack_e, cudaMemcpyHostToDevice);
    }

  if (resident_debug_enabled_local()) {
    std::vector<double> host_ptot(mpack);
    std::vector<double> host_p(mpack);
    cudaMemcpy(host_ptot.data(), s_d_ptot, sizeof(double) * mpack, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_p.data(), s_d_p, sizeof(double) * mpack, cudaMemcpyDeviceToHost);
    std::printf("[GPU resident debug] density sample ptot:% .5e % .5e % .5e\n",
                host_ptot[0], host_ptot[1], host_ptot[2]);
    std::printf("[GPU resident debug] density sample p:% .5e % .5e % .5e\n",
                host_p[0], host_p[1], host_p[2]);
    std::fflush(stdout);
  }

  if (w_len > 0) {
    cudaMemcpy(s_d_w, w, sizeof(double) * w_len, cudaMemcpyHostToDevice);
  }
  if (wj_len > 0) {
    cudaMemcpy(s_d_wj, wj, sizeof(double) * wj_len, cudaMemcpyHostToDevice);
  }
  if (wk_len > 0) {
    cudaMemcpy(s_d_wk, wk, sizeof(double) * wk_len, cudaMemcpyHostToDevice);
  }

  if (resident_debug_enabled_local()) {
    if (w_len > 0) {
      size_t limit = std::min(w_len, static_cast<size_t>(5));
      std::vector<double> host_w(limit);
      cudaMemcpy(host_w.data(), s_d_w, sizeof(double) * limit, cudaMemcpyDeviceToHost);
      std::printf("[GPU resident debug] w sample:");
      for (size_t idx = 0; idx < limit; ++idx) std::printf(" % .5e", host_w[idx]);
      std::printf("\n");
    }
    std::printf("[GPU resident debug] fock2_scf pairs=%zu w_len=%zu\n",
                pair_i.size(), w_len);
  }

  if (!pair_i.empty()) {
    if (cudaMemcpy(s_d_pair_i, pair_i.data(), sizeof(int) * pair_i.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(s_d_pair_j, pair_j.data(), sizeof(int) * pair_j.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(s_d_pair_off, pair_w_off.data(), sizeof(int) * pair_w_off.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(s_d_pair_type, pair_type.data(), sizeof(int) * pair_type.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(s_d_pair_wj_off, pair_wj_off.data(), sizeof(int) * pair_wj_off.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(s_d_pair_wk_off, pair_wk_off.data(), sizeof(int) * pair_wk_off.size(), cudaMemcpyHostToDevice) != cudaSuccess) return false;
  }

  if (resident_debug_enabled_local() && !pair_i.empty()) {
    size_t limit = std::min(pair_i.size(), static_cast<size_t>(3));
    for (size_t t = 0; t < limit; ++t) {
      int ii = pair_i[t];
      int jj = pair_j[t];
      int ia = nfirst[ii - 1];
      int ib = nlast[ii - 1];
      int ja = nfirst[jj - 1];
      int jb = nlast[jj - 1];
      std::printf("  pair[%zu]: ii=%d (%d-%d) jj=%d (%d-%d) type=%d w_off=%d wj_off=%d wk_off=%d\n",
                  t, ii, ia, ib, jj, ja, jb, pair_type[t], pair_w_off[t], pair_wj_off[t], pair_wk_off[t]);
    }
    std::fflush(stdout);
  }

  bool want_timing = (verbose != 0) || (csv_enabled != 0) || (prof_collect != 0);
  cudaEvent_t t_start = nullptr, t_stop = nullptr;
  if (want_timing && !pair_i.empty()) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start);
  }

  if (!pair_i.empty()) {
    int threads = 128;
    int blocks = static_cast<int>((pair_i.size() + threads - 1) / threads);
    int debug_flag = resident_debug_enabled_local() ? 1 : 0;
    fock_pairs_kernel<<<blocks, threads>>>(static_cast<int>(pair_i.size()),
                                           s_d_pair_i, s_d_pair_j, s_d_pair_type,
                                           s_d_pair_off, s_d_pair_wj_off, s_d_pair_wk_off,
                                           s_d_nf, s_d_nl,
                                           s_d_ptot, s_d_p,
                                           s_d_w, s_d_wj, s_d_wk,
                                           s_d_f, debug_flag);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;
    if (debug_flag) {
      std::vector<double> probe(5, 0.0);
      size_t probe_len = std::min(probe.size(), static_cast<size_t>(mpack));
      if (probe_len > 0) {
        cudaMemcpy(probe.data(), s_d_f, sizeof(double) * probe_len, cudaMemcpyDeviceToHost);
        std::printf("[GPU resident debug] post-kernel f sample:");
        for (size_t idx = 0; idx < probe_len; ++idx) {
          std::printf(" % .5e", probe[idx]);
        }
        std::printf("\n");
        std::fflush(stdout);
      }
    }
  }

  if (want_timing && t_start && t_stop) {
    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);
    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, t_start, t_stop);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);
    if (prof_collect) {
      prof_atoms += numat;
      prof_total_ms += ms_total;
      prof_ll_pairs += ll_pairs;
      prof_lh_pairs += lh_pairs;
      prof_hh_pairs += hh_pairs;
      long long total_pairs = ll_pairs + lh_pairs + hh_pairs;
      if (total_pairs > 0) {
        double share = ms_total / static_cast<double>(total_pairs);
        prof_ll_ms += share * static_cast<double>(ll_pairs);
        prof_lh_ms += share * static_cast<double>(lh_pairs);
        prof_hh_ms += share * static_cast<double>(hh_pairs);
      }
    }
  }

  bool resident = (mopac_cuda_get_resident_mode() != 0);
  if (resident) {
    if (resident_debug_enabled_local()) {
      size_t limit = std::min(mpack_e, static_cast<size_t>(5));
      std::vector<double> host_f(limit);
      cudaMemcpy(host_f.data(), s_d_f, sizeof(double) * limit, cudaMemcpyDeviceToHost);
      std::printf("[GPU resident debug] f device sample:");
      for (size_t idx = 0; idx < limit; ++idx) std::printf(" % .5e", host_f[idx]);
      std::printf("\n");
      std::fflush(stdout);
    }
    mopac_cuda_register_fock_device(mpack, fout, s_d_f);
    if (!mopac_cuda_fetch_fock(fout, mpack_e)) {
      cudaMemcpy(fout, s_d_f, sizeof(double) * mpack_e, cudaMemcpyDeviceToHost);
    }
  } else {
    cudaMemcpy(fout, s_d_f, sizeof(double) * mpack_e, cudaMemcpyDeviceToHost);
    mopac_cuda_clear_fock_cache();
  }

  if (want_verify && !f_host.empty()) {
    double max_diff = 0.0;
    for (size_t k = 0; k < mpack_e; ++k) {
      double diff = std::abs(fout[k] - f_host[k]);
      if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 1.0e-9) {
      std::printf("[GPU FOCK verify] max diff=% .5e\n", max_diff);
      return false;
    }
  }

  return true;
}


} // extern "C"

bool mopac_cuda_fock2(int norbs, int mpack, int numat,
                      const int *nfirst, const int *nlast,
                      const double *ptot, const double *p,
                      const double *w, int nati,
                      double *f) {
  (void)norbs; (void)mpack;
  ensure_jindex_device();
  int ia = nfirst[nati - 1];
  int ib = nlast[nati - 1];
  if ((ib - ia) < 0) return false;

  int span_i = span_count(ia, ib);
  if (span_i <= 0) return true;

  std::vector<int> pair_i;
  std::vector<int> pair_j;
  std::vector<int> pair_off;
  pair_i.reserve(std::max(0, nati - 1));

  const size_t max_index = static_cast<size_t>(std::numeric_limits<int>::max());

  size_t w_len = 0;
  int pairs_i = pair_count(span_i);
  for (int jj = 1; jj < nati; ++jj) {
    int ja = nfirst[jj - 1];
    int jb = nlast[jj - 1];
    int span_j = span_count(ja, jb);
    if (span_j <= 0) continue;
    int pairs_j = pair_count(span_j);
    int chunk = pairs_i * pairs_j;
    if (chunk <= 0) continue;
    if (pair_i.size() >= max_index) return false;
    if (w_len > max_index) return false;
    pair_i.push_back(nati);
    pair_j.push_back(jj);
    pair_off.push_back(static_cast<int>(w_len));
    w_len += static_cast<size_t>(chunk);
  }

  if (pair_i.size() > max_index) return false;

  size_t atoms_e = (size_t)numat;
  size_t mpack_e = (size_t)mpack;
  if (!ensure_buf_int(&s_d_nf, &cap_nf, atoms_e)) return false;
  if (!ensure_buf_int(&s_d_nl, &cap_nl, atoms_e)) return false;
  if (!ensure_buf_double(&s_d_ptot, &cap_ptot, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_p, &cap_p, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_f, &cap_f, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_w, &cap_w, w_len)) return false;
  if (!ensure_pair_buffers(pair_i.size())) return false;

  if (cudaMemset(s_d_f, 0, sizeof(double)*mpack_e) != cudaSuccess) return false;

  cudaMemcpy(s_d_nf, nfirst, sizeof(int)*atoms_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_nl, nlast, sizeof(int)*atoms_e, cudaMemcpyHostToDevice);
  if (!mopac_cuda_density_copy_cached(s_d_ptot, mpack_e, ptot)) {
    cudaMemcpy(s_d_ptot, ptot, sizeof(double)*mpack_e, cudaMemcpyHostToDevice);
  }
  if (!mopac_cuda_density_copy_cached(s_d_p, mpack_e, p)) {
    cudaMemcpy(s_d_p, p, sizeof(double)*mpack_e, cudaMemcpyHostToDevice);
  }
  cudaMemcpy(s_d_w, w, sizeof(double)*w_len, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_f, f, sizeof(double)*mpack_e, cudaMemcpyHostToDevice);
  if (!pair_i.empty()) {
    std::vector<int> pair_type(pair_i.size(), PAIR_GENERAL);
    std::vector<int> zeros(pair_i.size(), 0);
    cudaMemcpy(s_d_pair_i, pair_i.data(), sizeof(int)*pair_i.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_j, pair_j.data(), sizeof(int)*pair_j.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_off, pair_off.data(), sizeof(int)*pair_off.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_type, pair_type.data(), sizeof(int)*pair_type.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_wj_off, zeros.data(), sizeof(int)*zeros.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_wk_off, zeros.data(), sizeof(int)*zeros.size(), cudaMemcpyHostToDevice);
    int threads = 64;
    int blocks = static_cast<int>((pair_i.size() + threads - 1) / threads);
    fock_pairs_kernel<<<blocks, threads>>>(static_cast<int>(pair_i.size()),
                                           s_d_pair_i, s_d_pair_j, s_d_pair_type,
                                           s_d_pair_off, s_d_pair_wj_off, s_d_pair_wk_off,
                                           s_d_nf, s_d_nl,
                                           s_d_ptot, s_d_p,
                                           s_d_w, nullptr, nullptr,
                                           s_d_f, 1);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;
    {
      std::vector<double> probe(5, 0.0);
      size_t probe_len = std::min(probe.size(), static_cast<size_t>(mpack));
      if (probe_len > 0) {
        cudaMemcpy(probe.data(), s_d_f, sizeof(double) * probe_len, cudaMemcpyDeviceToHost);
        std::printf("[GPU resident debug] post-kernel (keep) f sample:");
        for (size_t idx = 0; idx < probe_len; ++idx) {
          std::printf(" % .5e", probe[idx]);
        }
        std::printf("\n");
        std::fflush(stdout);
      }
    }
  }

  cudaMemcpy(f, s_d_f, sizeof(double)*mpack_e, cudaMemcpyDeviceToHost);
  return true;
}

}

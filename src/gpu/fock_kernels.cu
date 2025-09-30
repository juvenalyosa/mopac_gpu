// Minimal light-light branch for Fock derivative (dfock2) on GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>

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
static double *s_d_ptot = nullptr, *s_d_p = nullptr, *s_d_w = nullptr, *s_d_f = nullptr;
static int    *s_d_pair_i = nullptr, *s_d_pair_j = nullptr, *s_d_pair_off = nullptr;
static size_t cap_nf = 0, cap_nl = 0;
static size_t cap_ptot = 0, cap_p = 0, cap_f = 0, cap_w = 0, cap_pairs = 0;
static int verbose = 0; static int verbose_inited = 0;
static int csv_enabled = 0; static int csv_inited = 0;
static int prof_collect = 0; static int prof_inited = 0;
static int prof_env_requested = 0;
static int th_ll MOPAC_UNUSED = 64;
static int th_lh MOPAC_UNUSED = 32;
static int th_hh MOPAC_UNUSED = 16;
static int th_inited MOPAC_UNUSED = 0;
// Profiling accumulators
static long long prof_atoms = 0;
static long long prof_ll_pairs = 0, prof_lh_pairs = 0, prof_hh_pairs = 0;
static double prof_total_ms = 0.0, prof_ll_ms = 0.0, prof_lh_ms = 0.0, prof_hh_ms = 0.0;

__host__ __device__ inline int span_count(int first, int last) {
  return (last >= first) ? (last - first + 1) : 0;
}

__host__ __device__ inline int pair_count(int span) {
  return (span > 0) ? (span * (span + 1)) / 2 : 0;
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

static inline bool ensure_pair_buffers(size_t need_pairs) {
  if (cap_pairs < need_pairs) {
    if (s_d_pair_i) cudaFree(s_d_pair_i);
    if (s_d_pair_j) cudaFree(s_d_pair_j);
    if (s_d_pair_off) cudaFree(s_d_pair_off);
    s_d_pair_i = s_d_pair_j = s_d_pair_off = nullptr;
    cap_pairs = 0;
    if (need_pairs > 0) {
      if (cudaMalloc((void**)&s_d_pair_i, sizeof(int) * need_pairs) != cudaSuccess) return false;
      if (cudaMalloc((void**)&s_d_pair_j, sizeof(int) * need_pairs) != cudaSuccess) return false;
      if (cudaMalloc((void**)&s_d_pair_off, sizeof(int) * need_pairs) != cudaSuccess) return false;
      cap_pairs = need_pairs;
    }
  }
  return true;
}

__host__ __device__ inline int packed_index_zero(int a, int b) {
  int aa = a - 1;
  int bb = b - 1;
  if (aa >= bb) {
    return (aa * (aa + 1)) / 2 + bb;
  } else {
    return (bb * (bb + 1)) / 2 + aa;
  }
}

__device__ void fock_pair_update(int ia, int ib, int ja, int jb,
                                 const double *ptot, const double *p,
                                 const double *w, double *f) {
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
            int ik = packed_index_zero(i, k);
            int il = packed_index_zero(i, l);
            int jk = packed_index_zero(j, k);
            int jl = packed_index_zero(j, l);
            double a = w[kr++];
            atomicAdd_double(&f[ij], bb * a * ptot[kl]);
            atomicAdd_double(&f[kl], aa * a * ptot[ij]);
            double exch = a * aa * bb * 0.25;
            atomicAdd_double(&f[ik], -exch * p[jl]);
            atomicAdd_double(&f[il], -exch * p[jk]);
            atomicAdd_double(&f[jk], -exch * p[il]);
            atomicAdd_double(&f[jl], -exch * p[ik]);
          }
        }
      }
    }
  } else {
    int nn = pair_count(jb - ja + 1);
    if (nn <= 0) return;
    int n1 = 0;
    for (int i = ja; i <= jb; ++i) {
      for (int j = ja; j <= i; ++j) {
        n1 += 1;
        double aa = (i == j) ? 1.0 : 2.0;
        int ij = packed_index_zero(i, j);
        int n2 = 0;
        for (int k = ia; k <= ib; ++k) {
          for (int l = ia; l <= k; ++l) {
            n2 += 1;
            double bb = (k == l) ? 1.0 : 2.0;
            int kl = packed_index_zero(k, l);
            int ik = packed_index_zero(i, k);
            int il = packed_index_zero(i, l);
            int jk = packed_index_zero(j, k);
            int jl = packed_index_zero(j, l);
            int idx = (n2 - 1) * nn + (n1 - 1);
            double a = w[idx];
            atomicAdd_double(&f[ij], bb * a * ptot[kl]);
            atomicAdd_double(&f[kl], aa * a * ptot[ij]);
            double exch = a * aa * bb * 0.25;
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

__global__ void fock_pairs_kernel(int npairs,
                                  const int *pair_i,
                                  const int *pair_j,
                                  const int *pair_off,
                                  const int *nfirst,
                                  const int *nlast,
                                  const double *ptot,
                                  const double *p,
                                  const double *w,
                                  double *f) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npairs) return;
  int ii = pair_i[tid];
  int jj = pair_j[tid];
  int ia = nfirst[ii - 1];
  int ib = nlast[ii - 1];
  int ja = nfirst[jj - 1];
  int jb = nlast[jj - 1];
  if ((ib - ia) < 0 || (jb - ja) < 0) return;
  const double *w_block = w + pair_off[tid];
  fock_pair_update(ia, ib, ja, jb, ptot, p, w_block, f);
}

// ================= Device-resident gradient buffers and ops =================
extern "C" {

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
  }
  if (!pair_i.empty()) {
    cudaMemcpy(s_d_pair_i, pair_i.data(), sizeof(int)*pair_i.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_j, pair_j.data(), sizeof(int)*pair_j.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_off, pair_off.data(), sizeof(int)*pair_off.size(), cudaMemcpyHostToDevice);
    int threads = 64;
    int blocks = static_cast<int>((pair_i.size() + threads - 1) / threads);
    fock_pairs_kernel<<<blocks, threads>>>(static_cast<int>(pair_i.size()),
                                           s_d_pair_i, s_d_pair_j, s_d_pair_off,
                                           s_d_nf, s_d_nl,
                                           s_d_ptot, s_d_p,
                                           s_d_w, s_d_f);
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
  if (s_d_f) cudaFree(s_d_f); s_d_f = nullptr; cap_f = 0;
  if (s_d_pair_i) cudaFree(s_d_pair_i); s_d_pair_i = nullptr;
  if (s_d_pair_j) cudaFree(s_d_pair_j); s_d_pair_j = nullptr;
  if (s_d_pair_off) cudaFree(s_d_pair_off); s_d_pair_off = nullptr;
  cap_pairs = 0;
  g_lastF_dev = nullptr; g_lastF_bytes = 0; g_lastF_n = 0;
  if (g_blas_local) { cublasDestroy(g_blas_local); g_blas_local = nullptr; }
  if (g_stream_local) { cudaStreamDestroy(g_stream_local); g_stream_local = nullptr; }
}

} // extern "C"

// =============== SCF Fock (J/K) on GPU (experimental) ===============
extern "C" {

bool mopac_cuda_fock2_scf(int norbs, int mpack, int numat,
                          const int *nfirst, const int *nlast,
                          const double *ptot, const double *p,
                          const double *w, double *fout) {
  ensure_verbose(); ensure_thresholds(); ensure_csv(); ensure_profile_collect();
  size_t atoms_e = (size_t)numat;
  size_t mpack_e = (size_t)mpack;

  std::vector<int> pair_i;
  std::vector<int> pair_j;
  std::vector<int> pair_off;
  pair_i.reserve(std::max(1, numat));

  size_t w_len = 0;
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
      int chunk = pairs_i * pairs_j;
      if (chunk <= 0) continue;
      if (pair_i.size() >= max_index) return false;
      if (w_len > max_index) return false;
      pair_i.push_back(ii);
      pair_j.push_back(jj);
      pair_off.push_back(static_cast<int>(w_len));
      w_len += static_cast<size_t>(chunk);
      if (span_i == 1 && span_j == 1) {
        ll_pairs++;
      } else if (span_i == 1 || span_j == 1) {
        lh_pairs++;
      } else {
        hh_pairs++;
      }
    }
  }

  if (pair_i.size() > max_index) return false;

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
  if (w_len > 0) {
    cudaMemcpy(s_d_w, w, sizeof(double)*w_len, cudaMemcpyHostToDevice);
  }
  if (!pair_i.empty()) {
    cudaMemcpy(s_d_pair_i, pair_i.data(), sizeof(int)*pair_i.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_j, pair_j.data(), sizeof(int)*pair_j.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_off, pair_off.data(), sizeof(int)*pair_off.size(), cudaMemcpyHostToDevice);
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
    fock_pairs_kernel<<<blocks, threads>>>(static_cast<int>(pair_i.size()),
                                           s_d_pair_i, s_d_pair_j, s_d_pair_off,
                                           s_d_nf, s_d_nl,
                                           s_d_ptot, s_d_p,
                                           s_d_w, s_d_f);
    cudaError_t err = cudaDeviceSynchronize(); if (err != cudaSuccess) return false;
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
        double share = ms_total / (double)total_pairs;
        prof_ll_ms += share * (double)ll_pairs;
        prof_lh_ms += share * (double)lh_pairs;
        prof_hh_ms += share * (double)hh_pairs;
      }
    }
  }

  cudaMemcpy(fout, s_d_f, sizeof(double)*mpack_e, cudaMemcpyDeviceToHost);
  if (mopac_cuda_get_resident_mode() != 0) {
    mopac_cuda_register_fock_device(mpack, fout, s_d_f);
  } else {
    mopac_cuda_clear_fock_cache();
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
    cudaMemcpy(s_d_pair_i, pair_i.data(), sizeof(int)*pair_i.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_j, pair_j.data(), sizeof(int)*pair_j.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_pair_off, pair_off.data(), sizeof(int)*pair_off.size(), cudaMemcpyHostToDevice);
    int threads = 64;
    int blocks = static_cast<int>((pair_i.size() + threads - 1) / threads);
    fock_pairs_kernel<<<blocks, threads>>>(static_cast<int>(pair_i.size()),
                                           s_d_pair_i, s_d_pair_j, s_d_pair_off,
                                           s_d_nf, s_d_nl,
                                           s_d_ptot, s_d_p,
                                           s_d_w, s_d_f);
    cudaError_t err = cudaDeviceSynchronize(); if (err != cudaSuccess) return false;
  }

  cudaMemcpy(f, s_d_f, sizeof(double)*mpack_e, cudaMemcpyDeviceToHost);
  return true;
}

}

// Minimal light-light branch for Fock derivative (dfock2) on GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>

extern "C" {

// Forward declarations for dfock2 kernels used before their definitions (C linkage)
__global__ void dfock2_ll_lh_kernel(int norbs, int mpack, int numat,
                                    const int* nfirst, const int* nlast,
                                    const double* ptot, const double* p, const double* w,
                                    int nati, double* f);
__global__ void dfock2_ll_parallel_kernel(int norbs, int mpack, int numat,
                                          const int* nfirst, const int* nlast,
                                          const double* ptot, const double* p, const double* w,
                                          int ii, const int* jlist, const int* offlist, int count, double* f);
__global__ void dfock2_lh_parallel_kernel(int norbs, int mpack, int numat,
                                          const int* nfirst, const int* nlast,
                                          const double* ptot, const double* p, const double* w,
                                          int ii, const int* jlist, const int* offlist, int count, double* f);
__global__ void dfock2_hh_parallel_kernel(int norbs, int mpack, int numat,
                                          const int* nfirst, const int* nlast,
                                          const double* ptot, const double* p, const double* w,
                                          int ii, const int* jlist, const int* offlist, int count, double* f);

__device__ __host__ inline int ifact_idx(int i) { return (i * (i - 1)) / 2; }

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
static size_t cap_atoms = 0, cap_mpack = 0, cap_w = 0;
static int verbose = 0; static int verbose_inited = 0;
static int csv_enabled = 0; static int csv_inited = 0;
static int th_ll = 64, th_lh = 32, th_hh = 16; static int th_inited = 0;
// Profiling accumulators
static long long prof_atoms = 0;
static long long prof_ll_pairs = 0, prof_lh_pairs = 0, prof_hh_pairs = 0;
static double prof_total_ms = 0.0, prof_ll_ms = 0.0, prof_lh_ms = 0.0, prof_hh_ms = 0.0;

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

// ================= Device-resident gradient buffers and ops =================
extern "C" {

static double *g_lastF_dev = nullptr;
static size_t g_lastF_bytes = 0;
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

bool mopac_cuda_fock2_keep(int norbs, int mpack, int numat,
                           const int *nfirst, const int *nlast,
                           const double *ptot, const double *p,
                           const double *w, int nati) {
  ensure_verbose();
  int ia = nfirst[nati - 1];
  int ib = nlast[nati - 1];
  if ((ib - ia) < 0) return false;
  // Compute w_len
  size_t w_len = 0;
  for (int jj = 1; jj <= numat; ++jj) {
    if (jj == nati) continue;
    int ja = nfirst[jj - 1];
    int jb = nlast[jj - 1];
    if ((jb - ja) < 0) continue;
    int di = ib - ia;
    int dj = jb - ja;
    if (di >= 3 && dj >= 3) w_len += 100; else if (di >= 3 || dj >= 3) w_len += 10; else w_len += 1;
  }
  if (w_len == 0) return true;
  // Ensure persistent buffers
  size_t atoms_e = (size_t)numat, mpack_e = (size_t)mpack, w_e = w_len;
  if (!ensure_buf_int(&s_d_nf, &cap_atoms, atoms_e)) return false;
  if (!ensure_buf_int(&s_d_nl, &cap_atoms, atoms_e)) return false;
  if (!ensure_buf_double(&s_d_ptot, &cap_mpack, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_p, &cap_mpack, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_f, &cap_mpack, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_w, &cap_w, w_e)) return false;
  cudaMemcpy(s_d_nf, nfirst, sizeof(int)*atoms_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_nl, nlast, sizeof(int)*atoms_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_ptot, ptot, sizeof(double)*mpack_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_p, p, sizeof(double)*mpack_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_w, w, sizeof(double)*w_e, cudaMemcpyHostToDevice);
  if (verbose) printf("GPU grad keep: serial kernel\n");
  dfock2_ll_lh_kernel<<<1,1>>>(norbs, mpack, numat, s_d_nf, s_d_nl, s_d_ptot, s_d_p, s_d_w, nati, s_d_f);
  cudaError_t e = cudaDeviceSynchronize(); if (e!=cudaSuccess) return false;
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
  double alpha=1.0, beta=0.0;
  cublasDgemm(g_blas_local, CUBLAS_OP_N, CUBLAS_OP_N,
              n, n, n, &alpha,
              g_lastF_dev, n,
              dC, ldc,
              &beta,
              dW, ldw);
  cudaMemcpyAsync(W, dW, bytesN, cudaMemcpyDeviceToHost, g_stream_local);
  cudaStreamSynchronize(g_stream_local);
  cudaFree(dC); cudaFree(dW);
}

void mopac_cuda_grad_buffers_release() {
  // Print end-of-run profiling summary if enabled
  if (verbose && prof_atoms > 0) {
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
  if (s_d_nf) cudaFree(s_d_nf); s_d_nf = nullptr; cap_atoms = 0;
  if (s_d_nl) cudaFree(s_d_nl); s_d_nl = nullptr;
  if (s_d_ptot) cudaFree(s_d_ptot); s_d_ptot = nullptr; cap_mpack = 0;
  if (s_d_p) cudaFree(s_d_p); s_d_p = nullptr;
  if (s_d_w) cudaFree(s_d_w); s_d_w = nullptr; cap_w = 0;
  if (s_d_f) cudaFree(s_d_f); s_d_f = nullptr;
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
  ensure_verbose(); ensure_thresholds(); ensure_csv();
  // Prepare buffers
  size_t atoms_e = (size_t)numat;
  size_t mpack_e = (size_t)mpack;
  // Estimate total w length: emulate Fortran kk progression (jj<ii only)
  size_t w_len = 0;
  for (int ii = 1; ii <= numat; ++ii) {
    int ia = nfirst[ii - 1]; int ib = nlast[ii - 1]; if ((ib - ia) < 0) continue;
    for (int jj = 1; jj < ii; ++jj) {
      int ja = nfirst[jj - 1]; int jb = nlast[jj - 1]; if ((jb - ja) < 0) continue;
      int di = ib - ia; int dj = jb - ja;
      if (di >= 3 && dj >= 3) w_len += 100; else if (di >= 3 || dj >= 3) w_len += 10; else w_len += 1;
    }
  }
  if (!ensure_buf_int(&s_d_nf, &cap_atoms, atoms_e)) return false;
  if (!ensure_buf_int(&s_d_nl, &cap_atoms, atoms_e)) return false;
  if (!ensure_buf_double(&s_d_ptot, &cap_mpack, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_p, &cap_mpack, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_f, &cap_mpack, mpack_e)) return false;
  if (!ensure_buf_double(&s_d_w, &cap_w, w_len)) return false;
  // Copy inputs and zero F
  cudaMemcpy(s_d_nf, nfirst, sizeof(int)*atoms_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_nl, nlast, sizeof(int)*atoms_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_ptot, ptot, sizeof(double)*mpack_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_p, p, sizeof(double)*mpack_e, cudaMemcpyHostToDevice);
  cudaMemcpy(s_d_w, w, sizeof(double)*w_len, cudaMemcpyHostToDevice);
  cudaMemset(s_d_f, 0, sizeof(double)*mpack_e);

  // For each atom ii, build pair lists (jj<ii) with absolute w offsets and launch kernels
  size_t kk_cursor = 0;
  for (int ii = 1; ii <= numat; ++ii) {
    int ia = nfirst[ii - 1]; int ib = nlast[ii - 1]; if ((ib - ia) < 0) continue;
    // Count pairs for this ii
    int ll_count = 0, lh_count = 0, hh_count = 0;
    size_t kk_tmp = kk_cursor;
    for (int jj = 1; jj < ii; ++jj) {
      int ja = nfirst[jj - 1]; int jb = nlast[jj - 1]; if ((jb - ja) < 0) continue;
      int di = ib - ia; int dj = jb - ja;
      if (di < 3 && dj < 3) { ll_count++; kk_tmp += 1; }
      else if (di >= 3 && dj >= 3) { hh_count++; kk_tmp += 100; }
      else { lh_count++; kk_tmp += 10; }
    }
    if (ll_count + lh_count + hh_count == 0) continue;
    int *h_ll_j = nullptr, *h_ll_off=nullptr, *h_lh_j=nullptr, *h_lh_off=nullptr, *h_hh_j=nullptr, *h_hh_off=nullptr;
    if (ll_count>0) { h_ll_j=(int*)malloc(sizeof(int)*ll_count); h_ll_off=(int*)malloc(sizeof(int)*ll_count); }
    if (lh_count>0) { h_lh_j=(int*)malloc(sizeof(int)*lh_count); h_lh_off=(int*)malloc(sizeof(int)*lh_count); }
    if (hh_count>0) { h_hh_j=(int*)malloc(sizeof(int)*hh_count); h_hh_off=(int*)malloc(sizeof(int)*hh_count); }
    // Fill
    int il=0, ih=0, ihh=0;
    for (int jj = 1; jj < ii; ++jj) {
      int ja = nfirst[jj - 1]; int jb = nlast[jj - 1]; if ((jb - ja) < 0) continue;
      int di = ib - ia; int dj = jb - ja;
      if (di < 3 && dj < 3) { if (h_ll_j){ h_ll_j[il]=jj; h_ll_off[il]=(int)kk_cursor; il++; } kk_cursor += 1; }
      else if (di >= 3 && dj >= 3) { if (h_hh_j){ h_hh_j[ihh]=jj; h_hh_off[ihh]=(int)kk_cursor; ihh++; } kk_cursor += 100; }
      else { if (h_lh_j){ h_lh_j[ih]=jj; h_lh_off[ih]=(int)kk_cursor; ih++; } kk_cursor += 10; }
    }
    // Launch kernels in sequence
    int *d_j=nullptr, *d_off=nullptr;
    if (ll_count>0) {
      cudaMalloc((void**)&d_j, sizeof(int)*ll_count);
      cudaMalloc((void**)&d_off, sizeof(int)*ll_count);
      cudaMemcpy(d_j, h_ll_j, sizeof(int)*ll_count, cudaMemcpyHostToDevice);
      cudaMemcpy(d_off, h_ll_off, sizeof(int)*ll_count, cudaMemcpyHostToDevice);
      int block=256, grid=(ll_count+block-1)/block;
      dfock2_ll_parallel_kernel<<<grid, block>>>(norbs, mpack, numat, s_d_nf, s_d_nl, s_d_ptot, s_d_p, s_d_w, ii, d_j, d_off, ll_count, s_d_f);
      cudaDeviceSynchronize(); cudaFree(d_j); cudaFree(d_off);
    }
    if (lh_count>0) {
      cudaMalloc((void**)&d_j, sizeof(int)*lh_count);
      cudaMalloc((void**)&d_off, sizeof(int)*lh_count);
      cudaMemcpy(d_j, h_lh_j, sizeof(int)*lh_count, cudaMemcpyHostToDevice);
      cudaMemcpy(d_off, h_lh_off, sizeof(int)*lh_count, cudaMemcpyHostToDevice);
      int block=256, grid=(lh_count+block-1)/block;
      dfock2_lh_parallel_kernel<<<grid, block>>>(norbs, mpack, numat, s_d_nf, s_d_nl, s_d_ptot, s_d_p, s_d_w, ii, d_j, d_off, lh_count, s_d_f);
      cudaDeviceSynchronize(); cudaFree(d_j); cudaFree(d_off);
    }
    if (hh_count>0) {
      cudaMalloc((void**)&d_j, sizeof(int)*hh_count);
      cudaMalloc((void**)&d_off, sizeof(int)*hh_count);
      cudaMemcpy(d_j, h_hh_j, sizeof(int)*hh_count, cudaMemcpyHostToDevice);
      cudaMemcpy(d_off, h_hh_off, sizeof(int)*hh_count, cudaMemcpyHostToDevice);
      int block=128, grid=(hh_count+block-1)/block;
      dfock2_hh_parallel_kernel<<<grid, block>>>(norbs, mpack, numat, s_d_nf, s_d_nl, s_d_ptot, s_d_p, s_d_w, ii, d_j, d_off, hh_count, s_d_f);
      cudaDeviceSynchronize(); cudaFree(d_j); cudaFree(d_off);
    }
    if (h_ll_j) free(h_ll_j); if (h_ll_off) free(h_ll_off);
    if (h_lh_j) free(h_lh_j); if (h_lh_off) free(h_lh_off);
    if (h_hh_j) free(h_hh_j); if (h_hh_off) free(h_hh_off);
  }
  // Copy back result
  cudaMemcpy(fout, s_d_f, sizeof(double)*mpack_e, cudaMemcpyDeviceToHost);
  return true;
}

} // extern "C"

// Device helpers: heavy–heavy Coulomb (jab) and exchange (kab)
__device__ void jab_update(int ia, int ja, const double *pja, const double *pjb,
                           const double *w, double *f) {
  double suma[10];
  double sumb[10];
  // Ported from src/integrals/jab.F90 (1-based -> 0-based)
  suma[0] = pja[0]*w[0] + pja[1]*w[10] + pja[2]*w[30] + pja[3]*w[60] + pja[4]*w[10] + pja[5]*w[20] + pja[6]*w[40] + pja[7]*w[70] + pja[8]*w[30] + pja[9]*w[40] + pja[10]*w[50] + pja[11]*w[80] + pja[12]*w[60] + pja[13]*w[70] + pja[14]*w[80] + pja[15]*w[90];
  suma[1] = pja[0]*w[1] + pja[1]*w[11] + pja[2]*w[31] + pja[3]*w[61] + pja[4]*w[11] + pja[5]*w[21] + pja[6]*w[41] + pja[7]*w[71] + pja[8]*w[31] + pja[9]*w[41] + pja[10]*w[51] + pja[11]*w[81] + pja[12]*w[61] + pja[13]*w[71] + pja[14]*w[81] + pja[15]*w[91];
  suma[2] = pja[0]*w[2] + pja[1]*w[12] + pja[2]*w[32] + pja[3]*w[62] + pja[4]*w[12] + pja[5]*w[22] + pja[6]*w[42] + pja[7]*w[72] + pja[8]*w[32] + pja[9]*w[42] + pja[10]*w[52] + pja[11]*w[82] + pja[12]*w[62] + pja[13]*w[72] + pja[14]*w[82] + pja[15]*w[92];
  suma[3] = pja[0]*w[3] + pja[1]*w[13] + pja[2]*w[33] + pja[3]*w[63] + pja[4]*w[13] + pja[5]*w[23] + pja[6]*w[43] + pja[7]*w[73] + pja[8]*w[33] + pja[9]*w[43] + pja[10]*w[53] + pja[11]*w[83] + pja[12]*w[63] + pja[13]*w[73] + pja[14]*w[83] + pja[15]*w[93];
  suma[4] = pja[0]*w[4] + pja[1]*w[14] + pja[2]*w[34] + pja[3]*w[64] + pja[4]*w[14] + pja[5]*w[24] + pja[6]*w[44] + pja[7]*w[74] + pja[8]*w[34] + pja[9]*w[44] + pja[10]*w[54] + pja[11]*w[84] + pja[12]*w[64] + pja[13]*w[74] + pja[14]*w[84] + pja[15]*w[94];
  suma[5] = pja[0]*w[5] + pja[1]*w[15] + pja[2]*w[35] + pja[3]*w[65] + pja[4]*w[15] + pja[5]*w[25] + pja[6]*w[45] + pja[7]*w[75] + pja[8]*w[35] + pja[9]*w[45] + pja[10]*w[55] + pja[11]*w[85] + pja[12]*w[65] + pja[13]*w[75] + pja[14]*w[85] + pja[15]*w[95];
  suma[6] = pja[0]*w[6] + pja[1]*w[16] + pja[2]*w[36] + pja[3]*w[66] + pja[4]*w[16] + pja[5]*w[26] + pja[6]*w[46] + pja[7]*w[76] + pja[8]*w[36] + pja[9]*w[46] + pja[10]*w[56] + pja[11]*w[86] + pja[12]*w[66] + pja[13]*w[76] + pja[14]*w[86] + pja[15]*w[96];
  suma[7] = pja[0]*w[7] + pja[1]*w[17] + pja[2]*w[37] + pja[3]*w[67] + pja[4]*w[17] + pja[5]*w[27] + pja[6]*w[47] + pja[7]*w[77] + pja[8]*w[37] + pja[9]*w[47] + pja[10]*w[57] + pja[11]*w[87] + pja[12]*w[67] + pja[13]*w[77] + pja[14]*w[87] + pja[15]*w[97];
  suma[8] = pja[0]*w[8] + pja[1]*w[18] + pja[2]*w[38] + pja[3]*w[68] + pja[4]*w[18] + pja[5]*w[28] + pja[6]*w[48] + pja[7]*w[78] + pja[8]*w[38] + pja[9]*w[48] + pja[10]*w[58] + pja[11]*w[88] + pja[12]*w[68] + pja[13]*w[78] + pja[14]*w[88] + pja[15]*w[98];
  suma[9] = pja[0]*w[9] + pja[1]*w[19] + pja[2]*w[39] + pja[3]*w[69] + pja[4]*w[19] + pja[5]*w[29] + pja[6]*w[49] + pja[7]*w[79] + pja[8]*w[39] + pja[9]*w[49] + pja[10]*w[59] + pja[11]*w[89] + pja[12]*w[69] + pja[13]*w[79] + pja[14]*w[89] + pja[15]*w[99];
  sumb[0] = pjb[0]*w[0] + pjb[1]*w[1] + pjb[2]*w[3] + pjb[3]*w[6] + pjb[4]*w[1] + pjb[5]*w[2] + pjb[6]*w[4] + pjb[7]*w[7] + pjb[8]*w[3] + pjb[9]*w[4] + pjb[10]*w[5] + pjb[11]*w[8] + pjb[12]*w[6] + pjb[13]*w[7] + pjb[14]*w[8] + pjb[15]*w[9];
  sumb[1] = pjb[0]*w[10] + pjb[1]*w[11] + pjb[2]*w[13] + pjb[3]*w[16] + pjb[4]*w[11] + pjb[5]*w[12] + pjb[6]*w[14] + pjb[7]*w[17] + pjb[8]*w[13] + pjb[9]*w[14] + pjb[10]*w[15] + pjb[11]*w[18] + pjb[12]*w[16] + pjb[13]*w[17] + pjb[14]*w[18] + pjb[15]*w[19];
  sumb[2] = pjb[0]*w[20] + pjb[1]*w[21] + pjb[2]*w[23] + pjb[3]*w[26] + pjb[4]*w[21] + pjb[5]*w[22] + pjb[6]*w[24] + pjb[7]*w[27] + pjb[8]*w[23] + pjb[9]*w[24] + pjb[10]*w[25] + pjb[11]*w[28] + pjb[12]*w[26] + pjb[13]*w[27] + pjb[14]*w[28] + pjb[15]*w[29];
  sumb[3] = pjb[0]*w[30] + pjb[1]*w[31] + pjb[2]*w[33] + pjb[3]*w[36] + pjb[4]*w[31] + pjb[5]*w[32] + pjb[6]*w[34] + pjb[7]*w[37] + pjb[8]*w[33] + pjb[9]*w[34] + pjb[10]*w[35] + pjb[11]*w[38] + pjb[12]*w[36] + pjb[13]*w[37] + pjb[14]*w[38] + pjb[15]*w[39];
  sumb[4] = pjb[0]*w[40] + pjb[1]*w[41] + pjb[2]*w[43] + pjb[3]*w[46] + pjb[4]*w[41] + pjb[5]*w[42] + pjb[6]*w[44] + pjb[7]*w[47] + pjb[8]*w[43] + pjb[9]*w[44] + pjb[10]*w[45] + pjb[11]*w[48] + pjb[12]*w[46] + pjb[13]*w[47] + pjb[14]*w[48] + pjb[15]*w[49];
  sumb[5] = pjb[0]*w[50] + pjb[1]*w[51] + pjb[2]*w[53] + pjb[3]*w[56] + pjb[4]*w[51] + pjb[5]*w[52] + pjb[6]*w[54] + pjb[7]*w[57] + pjb[8]*w[53] + pjb[9]*w[54] + pjb[10]*w[55] + pjb[11]*w[58] + pjb[12]*w[56] + pjb[13]*w[57] + pjb[14]*w[58] + pjb[15]*w[59];
  sumb[6] = pjb[0]*w[60] + pjb[1]*w[61] + pjb[2]*w[63] + pjb[3]*w[66] + pjb[4]*w[61] + pjb[5]*w[62] + pjb[6]*w[64] + pjb[7]*w[67] + pjb[8]*w[63] + pjb[9]*w[64] + pjb[10]*w[65] + pjb[11]*w[68] + pjb[12]*w[66] + pjb[13]*w[67] + pjb[14]*w[68] + pjb[15]*w[69];
  sumb[7] = pjb[0]*w[70] + pjb[1]*w[71] + pjb[2]*w[73] + pjb[3]*w[76] + pjb[4]*w[71] + pjb[5]*w[72] + pjb[6]*w[74] + pjb[7]*w[77] + pjb[8]*w[73] + pjb[9]*w[74] + pjb[10]*w[75] + pjb[11]*w[78] + pjb[12]*w[76] + pjb[13]*w[77] + pjb[14]*w[78] + pjb[15]*w[79];
  sumb[8] = pjb[0]*w[80] + pjb[1]*w[81] + pjb[2]*w[83] + pjb[3]*w[86] + pjb[4]*w[81] + pjb[5]*w[82] + pjb[6]*w[84] + pjb[7]*w[87] + pjb[8]*w[83] + pjb[9]*w[84] + pjb[10]*w[85] + pjb[11]*w[88] + pjb[12]*w[86] + pjb[13]*w[87] + pjb[14]*w[88] + pjb[15]*w[89];
  sumb[9] = pjb[0]*w[90] + pjb[1]*w[91] + pjb[2]*w[93] + pjb[3]*w[96] + pjb[4]*w[91] + pjb[5]*w[92] + pjb[6]*w[94] + pjb[7]*w[97] + pjb[8]*w[93] + pjb[9]*w[94] + pjb[10]*w[95] + pjb[11]*w[98] + pjb[12]*w[96] + pjb[13]*w[97] + pjb[14]*w[98] + pjb[15]*w[99];

  int i = 0;
  for (int i5 = 1; i5 <= 4; ++i5) {
    int iia = ia + i5 - 1;
    int ija = ja + i5 - 1;
    int ioff = ifact_idx(iia) + ia - 1;
    int joff = ifact_idx(ija) + ja - 1;
    for (int i6 = 1; i6 <= i5; ++i6) {
      ioff += 1; joff += 1; i += 1;
      atomicAdd_double(&f[ioff - 1], sumb[i - 1]);
      atomicAdd_double(&f[joff - 1], suma[i - 1]);
    }
  }
}

__device__ void kab_update(int ia, int ja, const double *pk, const double *w, double *f) {
  double sum[16];
  // Ported from src/integrals/kab.F90 (1-based -> 0-based)
  sum[0] = pk[0]*w[0] + pk[1]*w[1] + pk[2]*w[3] + pk[3]*w[6] + pk[4]*w[10] + pk[5]*w[11] + pk[6]*w[13] + pk[7]*w[16] + pk[8]*w[30] + pk[9]*w[31] + pk[10]*w[33] + pk[11]*w[36] + pk[12]*w[60] + pk[13]*w[61] + pk[14]*w[63] + pk[15]*w[66];
  sum[1] = pk[0]*w[1] + pk[1]*w[2] + pk[2]*w[4] + pk[3]*w[7] + pk[4]*w[11] + pk[5]*w[12] + pk[6]*w[14] + pk[7]*w[17] + pk[8]*w[31] + pk[9]*w[32] + pk[10]*w[34] + pk[11]*w[37] + pk[12]*w[61] + pk[13]*w[62] + pk[14]*w[64] + pk[15]*w[67];
  sum[2] = pk[0]*w[3] + pk[1]*w[4] + pk[2]*w[5] + pk[3]*w[8] + pk[4]*w[13] + pk[5]*w[14] + pk[6]*w[15] + pk[7]*w[18] + pk[8]*w[33] + pk[9]*w[34] + pk[10]*w[35] + pk[11]*w[38] + pk[12]*w[63] + pk[13]*w[64] + pk[14]*w[65] + pk[15]*w[68];
  sum[3] = pk[0]*w[6] + pk[1]*w[7] + pk[2]*w[8] + pk[3]*w[9] + pk[4]*w[16] + pk[5]*w[17] + pk[6]*w[18] + pk[7]*w[19] + pk[8]*w[36] + pk[9]*w[37] + pk[10]*w[38] + pk[11]*w[39] + pk[12]*w[66] + pk[13]*w[67] + pk[14]*w[68] + pk[15]*w[69];
  sum[4] = pk[0]*w[10] + pk[1]*w[11] + pk[2]*w[13] + pk[3]*w[16] + pk[4]*w[20] + pk[5]*w[21] + pk[6]*w[23] + pk[7]*w[26] + pk[8]*w[40] + pk[9]*w[41] + pk[10]*w[43] + pk[11]*w[46] + pk[12]*w[70] + pk[13]*w[71] + pk[14]*w[73] + pk[15]*w[76];
  sum[5] = pk[0]*w[11] + pk[1]*w[12] + pk[2]*w[14] + pk[3]*w[17] + pk[4]*w[21] + pk[5]*w[22] + pk[6]*w[24] + pk[7]*w[27] + pk[8]*w[41] + pk[9]*w[42] + pk[10]*w[44] + pk[11]*w[47] + pk[12]*w[71] + pk[13]*w[72] + pk[14]*w[74] + pk[15]*w[77];
  sum[6] = pk[0]*w[13] + pk[1]*w[14] + pk[2]*w[15] + pk[3]*w[18] + pk[4]*w[23] + pk[5]*w[24] + pk[6]*w[25] + pk[7]*w[28] + pk[8]*w[43] + pk[9]*w[44] + pk[10]*w[45] + pk[11]*w[48] + pk[12]*w[73] + pk[13]*w[74] + pk[14]*w[75] + pk[15]*w[78];
  sum[7] = pk[0]*w[16] + pk[1]*w[17] + pk[2]*w[18] + pk[3]*w[19] + pk[4]*w[26] + pk[5]*w[27] + pk[6]*w[28] + pk[7]*w[29] + pk[8]*w[46] + pk[9]*w[47] + pk[10]*w[48] + pk[11]*w[49] + pk[12]*w[76] + pk[13]*w[77] + pk[14]*w[78] + pk[15]*w[79];
  sum[8] = pk[0]*w[30] + pk[1]*w[31] + pk[2]*w[33] + pk[3]*w[36] + pk[4]*w[40] + pk[5]*w[41] + pk[6]*w[43] + pk[7]*w[46] + pk[8]*w[50] + pk[9]*w[51] + pk[10]*w[53] + pk[11]*w[56] + pk[12]*w[80] + pk[13]*w[81] + pk[14]*w[83] + pk[15]*w[86];
  sum[9] = pk[0]*w[31] + pk[1]*w[32] + pk[2]*w[34] + pk[3]*w[37] + pk[4]*w[41] + pk[5]*w[42] + pk[6]*w[44] + pk[7]*w[47] + pk[8]*w[51] + pk[9]*w[52] + pk[10]*w[54] + pk[11]*w[57] + pk[12]*w[81] + pk[13]*w[82] + pk[14]*w[84] + pk[15]*w[87];
  sum[10] = pk[0]*w[33] + pk[1]*w[34] + pk[2]*w[35] + pk[3]*w[38] + pk[4]*w[43] + pk[5]*w[44] + pk[6]*w[45] + pk[7]*w[48] + pk[8]*w[53] + pk[9]*w[54] + pk[10]*w[55] + pk[11]*w[58] + pk[12]*w[83] + pk[13]*w[84] + pk[14]*w[85] + pk[15]*w[88];
  sum[11] = pk[0]*w[36] + pk[1]*w[37] + pk[2]*w[38] + pk[3]*w[39] + pk[4]*w[46] + pk[5]*w[47] + pk[6]*w[48] + pk[7]*w[49] + pk[8]*w[56] + pk[9]*w[57] + pk[10]*w[58] + pk[11]*w[59] + pk[12]*w[86] + pk[13]*w[87] + pk[14]*w[88] + pk[15]*w[89];
  sum[12] = pk[0]*w[60] + pk[1]*w[61] + pk[2]*w[63] + pk[3]*w[66] + pk[4]*w[70] + pk[5]*w[71] + pk[6]*w[73] + pk[7]*w[76] + pk[8]*w[80] + pk[9]*w[81] + pk[10]*w[83] + pk[11]*w[86] + pk[12]*w[90] + pk[13]*w[91] + pk[14]*w[93] + pk[15]*w[96];
  sum[13] = pk[0]*w[61] + pk[1]*w[62] + pk[2]*w[64] + pk[3]*w[67] + pk[4]*w[71] + pk[5]*w[72] + pk[6]*w[74] + pk[7]*w[77] + pk[8]*w[81] + pk[9]*w[82] + pk[10]*w[84] + pk[11]*w[87] + pk[12]*w[91] + pk[13]*w[92] + pk[14]*w[94] + pk[15]*w[97];
  sum[14] = pk[0]*w[63] + pk[1]*w[64] + pk[2]*w[65] + pk[3]*w[68] + pk[4]*w[73] + pk[5]*w[74] + pk[6]*w[75] + pk[7]*w[78] + pk[8]*w[83] + pk[9]*w[84] + pk[10]*w[85] + pk[11]*w[88] + pk[12]*w[93] + pk[13]*w[94] + pk[14]*w[95] + pk[15]*w[98];
  sum[15] = pk[0]*w[66] + pk[1]*w[67] + pk[2]*w[68] + pk[3]*w[69] + pk[4]*w[76] + pk[5]*w[77] + pk[6]*w[78] + pk[7]*w[79] + pk[8]*w[86] + pk[9]*w[87] + pk[10]*w[88] + pk[11]*w[89] + pk[12]*w[96] + pk[13]*w[97] + pk[14]*w[98] + pk[15]*w[99];

  if (ia > ja) {
    int m = 0;
    for (int j1 = ia; j1 <= ia + 3; ++j1) {
      int j = ifact_idx(j1);
      for (int off = 0; off < 4; ++off) {
        atomicAdd_double(&f[(j + ja + off) - 1], -sum[m + off]);
      }
      m += 4;
    }
  } else {
    int m = 0;
    for (int j1 = ia; j1 <= ia + 3; ++j1) {
      for (int j2 = ja; j2 <= ja + 3; ++j2) {
        m += 1;
        int j3 = ifact_idx(j2) + j1;
        atomicAdd_double(&f[j3 - 1], -sum[m - 1]);
      }
    }
  }
}

__global__ void dfock2_ll_lh_kernel(int norbs, int mpack, int numat,
                                    const int *nfirst, const int *nlast,
                                    const double *ptot, const double *p,
                                    const double *w, int nati, double *f) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid != 0) return; // single-thread kernel
  int ii = nati;
  int ia = nfirst[ii - 1];
  int ib = nlast[ii - 1];
  int kk = 0;
  // Precompute jindex mapping (only relevant for single-heavy cases)
  int jindex[256];
  int m = 0;
  for (int i = 1; i <= 4; ++i) {
    for (int j = 1; j <= 4; ++j) {
      int ij = (i < j) ? i : j;
      int ji = i + j - ij;
      for (int k = 1; k <= 4; ++k) {
        for (int l = 1; l <= 4; ++l) {
          m += 1;
          int kl = (k < l) ? k : l;
          int lk = k + l - kl;
          int ifact_ji = ifact_idx(ji);
          int ifact_lk = ifact_idx(lk);
          jindex[m-1] = (ifact_ji + ij) * 10 + ifact_lk + kl - 10; // Fortran 1-based
        }
      }
    }
  }
  for (int jj = 1; jj <= numat; ++jj) {
    if (jj == ii) continue;
    int ja = nfirst[jj - 1];
    int jb = nlast[jj - 1];
    // Skip sparkles
    if ((ib - ia) < 0 || (jb - ja) < 0) continue;
    int di = ib - ia;
    int dj = jb - ja;
    if (di >= 3 && dj >= 3) {
      // heavy-heavy branch: build pja, pjb (16) and pk (16); use 100 w-terms
      double pja[16], pjb[16], pk[16];
      // pja from ptot
      {
        int mloc = 0;
        for (int j = ia; j <= ib; ++j) {
          for (int k = ia; k <= ib; ++k) {
            mloc += 1;
            int jk = (j < k) ? j : k;
            int kj = j + k - jk;
            int packed = jk + ifact_idx(kj);
            pja[mloc - 1] = ptot[packed - 1];
          }
        }
      }
      // pjb from ptot
      {
        int mloc = 0;
        for (int j = ja; j <= jb; ++j) {
          for (int k = ja; k <= jb; ++k) {
            mloc += 1;
            int jk = (j < k) ? j : k;
            int kj = j + k - jk;
            int packed = jk + ifact_idx(kj);
            pjb[mloc - 1] = ptot[packed - 1];
          }
        }
      }
      // pk from p intersection
      if (ia > ja) {
        int l = 0;
        for (int i2 = ia; i2 <= ib; ++i2) {
          int base = ifact_idx(i2) + ja;
          for (int col = 0; col <= (jb - ja); ++col) {
            l += 1; pk[l - 1] = p[(base + col) - 1];
          }
        }
      } else {
        int l = 0;
        for (int i2 = ia; i2 <= ib; ++i2) {
          for (int j2 = ja; j2 <= jb; ++j2) {
            l += 1; pk[l - 1] = p[(ifact_idx(j2) + i2) - 1];
          }
        }
      }
      // Apply J (Coulomb) and K (exchange)
      jab_update(ia, ja, pja, pjb, w + kk, f);
      kab_update(ia, ja, pk, w + kk, f);
      kk += 100;
    } else if (di >= 3 || dj >= 3) {
      // Single-heavy branch: Coulomb + exchange using 10 w-terms
      double sumdia = 0.0, sumoff = 0.0;
      int k = 0;
      if (di >= 3) {
        // ii heavy, jj light
        int ll = ifact_idx(ja) + ja; // i1fact(ja)
        for (int ii2 = 0; ii2 <= 3; ++ii2) {
          int j1 = ifact_idx(ia + ii2) + ia - 1;
          for (int j2 = 0; j2 < ii2; ++j2) {
            k += 1; j1 += 1;
            f[j1 - 1] += ptot[ll - 1] * w[kk + (k - 1)];
            sumoff += ptot[j1 - 1] * w[kk + (k - 1)];
          }
          j1 = j1 + 1; k += 1;
          f[j1 - 1] += ptot[ll - 1] * w[kk + (k - 1)];
          sumdia += ptot[j1 - 1] * w[kk + (k - 1)];
        }
        f[ll - 1] += sumoff * 2.0 + sumdia;
        // Exchange
        if (ia > ja) {
          int k2 = 0;
          for (int i3 = ia; i3 <= ib; ++i3) {
            int i1 = ifact_idx(i3) + ja;
            double sum = 0.0;
            for (int j3 = ia; j3 <= ib; ++j3) {
              k2 += 1;
              int j1 = ifact_idx(j3) + ja;
              sum += p[j1 - 1] * w[kk + (jindex[k2 - 1] - 1)];
            }
            f[i1 - 1] -= sum;
          }
        } else {
          int k2 = 0;
          for (int i3 = ia; i3 <= ib; ++i3) {
            int i1 = ifact_idx(ja) + i3;
            double sum = 0.0;
            for (int j3 = ia; j3 <= ib; ++j3) {
              k2 += 1;
              int j1 = ifact_idx(ja) + j3;
              sum += p[j1 - 1] * w[kk + (jindex[k2 - 1] - 1)];
            }
            f[i1 - 1] -= sum;
          }
        }
      } else {
        // jj heavy, ii light
        int ll = ifact_idx(ia) + ia; // i1fact(ia)
        for (int ii2 = 0; ii2 <= 3; ++ii2) {
          int j1 = ifact_idx(ja + ii2) + ja - 1;
          for (int j2 = 0; j2 < ii2; ++j2) {
            k += 1; j1 += 1;
            f[j1 - 1] += ptot[ll - 1] * w[kk + (k - 1)];
            sumoff += ptot[j1 - 1] * w[kk + (k - 1)];
          }
          j1 = j1 + 1; k += 1;
          f[j1 - 1] += ptot[ll - 1] * w[kk + (k - 1)];
          sumdia += ptot[j1 - 1] * w[kk + (k - 1)];
        }
        f[ll - 1] += sumoff * 2.0 + sumdia;
        // Exchange
        if (ia > ja) {
          int k2 = ifact_idx(ia) + ja; // starting pos
          int jacc = 0;
          int base = ifact_idx(ia) + ja;
          for (int irow = base; irow <= base + 3; ++irow) {
            double sum = 0.0;
            for (int lcol = base; lcol <= base + 3; ++lcol) {
              jacc += 1;
              sum += p[lcol - 1] * w[kk + (jindex[jacc - 1] - 1)];
            }
            f[irow - 1] -= sum;
          }
        } else {
          int jacc = 0;
          for (int k2 = ja; k2 <= ja + 3; ++k2) {
            int iidx = ifact_idx(k2) + ia;
            double sum = 0.0;
            for (int ll2 = ja; ll2 <= ja + 3; ++ll2) {
              int lidx = ifact_idx(ll2) + ia;
              jacc += 1;
              sum += p[lidx - 1] * w[kk + (jindex[jacc - 1] - 1)];
            }
            f[iidx - 1] -= sum;
          }
        }
      }
      kk += 10;
    } else {
      // light-light branch
      double elrep = w[kk]; // Fortran w(kk+1)
      int i1 = ifact_idx(ia) + ia;
      int j1 = ifact_idx(ja) + ja;
      int pos_i1 = i1 - 1;
      int pos_j1 = j1 - 1;
      // Diagonal updates
      f[pos_i1] += ptot[pos_j1] * elrep;
      f[pos_j1] += ptot[pos_i1] * elrep;
      // Off-diagonal update
      int ij;
      if (ia > ja) {
        ij = i1 + (ja - ia);
      } else {
        ij = j1 + (ia - ja);
      }
      int pos_ij = ij - 1;
      f[pos_ij] -= p[pos_ij] * elrep;
      kk += 1;
    }
  }
}

// Parallel LL kernel: one thread per contributing jj pair
__global__ void dfock2_ll_parallel_kernel(int norbs, int mpack, int numat,
                                          const int *nfirst, const int *nlast,
                                          const double *ptot, const double *p,
                                          const double *w, int nati,
                                          const int *jlist, const int *woff,
                                          int nn, double *f) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nn) return;
  int ia = nfirst[nati - 1];
  int ib = nlast[nati - 1];
  int jj = jlist[tid];
  int ja = nfirst[jj - 1];
  int jb = nlast[jj - 1];
  if ((ib - ia) < 0 || (jb - ja) < 0) return;
  int k0 = woff[tid];
  double elrep = w[k0];
  int i1 = ifact_idx(ia) + ia;
  int j1 = ifact_idx(ja) + ja;
  int pos_i1 = i1 - 1;
  int pos_j1 = j1 - 1;
  atomicAdd_double(&f[pos_i1], ptot[pos_j1] * elrep);
  atomicAdd_double(&f[pos_j1], ptot[pos_i1] * elrep);
  int ij = (ia > ja) ? (i1 + (ja - ia)) : (j1 + (ia - ja));
  int pos_ij = ij - 1;
  atomicAdd_double(&f[pos_ij], -p[pos_ij] * elrep);
}

// Parallel single-heavy kernel: one thread per (nati, jj) LH pair, 10-term block
__global__ void dfock2_lh_parallel_kernel(int norbs, int mpack, int numat,
                                          const int *nfirst, const int *nlast,
                                          const double *ptot, const double *p,
                                          const double *w, int nati,
                                          const int *jlist, const int *woff,
                                          int nn, double *f) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nn) return;
  int ii = nati;
  int ia = nfirst[ii - 1];
  int ib = nlast[ii - 1];
  int jj = jlist[tid];
  int ja = nfirst[jj - 1];
  int jb = nlast[jj - 1];
  if ((ib - ia) < 0 || (jb - ja) < 0) return;
  int di = ib - ia;
  int dj = jb - ja;
  int kk = woff[tid];
  // Build jindex mapping (256)
  __shared__ int jindex[256];
  if (threadIdx.x == 0) {
    int m = 0;
    for (int i = 1; i <= 4; ++i) {
      for (int j = 1; j <= 4; ++j) {
        int ij = (i < j) ? i : j;
        int ji = i + j - ij;
        for (int k = 1; k <= 4; ++k) {
          for (int l = 1; l <= 4; ++l) {
            m += 1;
            int kl = (k < l) ? k : l;
            int lk = k + l - kl;
            int ifact_ji = ifact_idx(ji);
            int ifact_lk = ifact_idx(lk);
            jindex[m-1] = (ifact_ji + ij) * 10 + ifact_lk + kl - 10;
          }
        }
      }
    }
  }
  __syncthreads();
  // Coulomb terms
  double sumdia = 0.0, sumoff = 0.0;
  int k = 0;
  if (di >= 3 && dj < 3) {
    // ii heavy, jj light
    int ll = ifact_idx(ja) + ja; // i1fact(ja)
    for (int i = 0; i <= 3; ++i) {
      int j1 = ifact_idx(ia + i) + ia - 1;
      for (int j = 0; j < i; ++j) {
        k += 1; j1 += 1;
        atomicAdd_double(&f[j1 - 1], ptot[ll - 1] * w[kk + (k - 1)]);
        sumoff += ptot[j1 - 1] * w[kk + (k - 1)];
      }
      j1 = j1 + 1; k += 1;
      atomicAdd_double(&f[j1 - 1], ptot[ll - 1] * w[kk + (k - 1)]);
      sumdia += ptot[j1 - 1] * w[kk + (k - 1)];
    }
    atomicAdd_double(&f[ll - 1], sumoff * 2.0 + sumdia);
    // Exchange
    if (ia > ja) {
      int k2 = 0;
      for (int i3 = ia; i3 <= ib; ++i3) {
        int i1 = ifact_idx(i3) + ja;
        double sum = 0.0;
        for (int j3 = ia; j3 <= ib; ++j3) {
          k2 += 1;
          int j1 = ifact_idx(j3) + ja;
          sum += p[j1 - 1] * w[kk + (jindex[k2 - 1] - 1)];
        }
        atomicAdd_double(&f[i1 - 1], -sum);
      }
    } else {
      int k2 = 0;
      for (int i3 = ia; i3 <= ib; ++i3) {
        int i1 = ifact_idx(ja) + i3;
        double sum = 0.0;
        for (int j3 = ia; j3 <= ib; ++j3) {
          k2 += 1;
          int j1 = ifact_idx(ja) + j3;
          sum += p[j1 - 1] * w[kk + (jindex[k2 - 1] - 1)];
        }
        atomicAdd_double(&f[i1 - 1], -sum);
      }
    }
  } else if (dj >= 3 && di < 3) {
    // jj heavy, ii light
    int ll = ifact_idx(ia) + ia; // i1fact(ia)
    for (int i = 0; i <= 3; ++i) {
      int j1 = ifact_idx(ja + i) + ja - 1;
      for (int j = 0; j < i; ++j) {
        k += 1; j1 += 1;
        atomicAdd_double(&f[j1 - 1], ptot[ll - 1] * w[kk + (k - 1)]);
        sumoff += ptot[j1 - 1] * w[kk + (k - 1)];
      }
      j1 = j1 + 1; k += 1;
      atomicAdd_double(&f[j1 - 1], ptot[ll - 1] * w[kk + (k - 1)]);
      sumdia += ptot[j1 - 1] * w[kk + (k - 1)];
    }
    atomicAdd_double(&f[ll - 1], sumoff * 2.0 + sumdia);
    // Exchange
    if (ia > ja) {
      int jacc = 0;
      int base = ifact_idx(ia) + ja;
      for (int irow = base; irow <= base + 3; ++irow) {
        double sum = 0.0;
        for (int lcol = base; lcol <= base + 3; ++lcol) {
          jacc += 1;
          sum += p[lcol - 1] * w[kk + (jindex[jacc - 1] - 1)];
        }
        atomicAdd_double(&f[irow - 1], -sum);
      }
    } else {
      int jacc = 0;
      for (int k2 = ja; k2 <= ja + 3; ++k2) {
        int iidx = ifact_idx(k2) + ia;
        double sum = 0.0;
        for (int ll2 = ja; ll2 <= ja + 3; ++ll2) {
          int lidx = ifact_idx(ll2) + ia;
          jacc += 1;
          sum += p[lidx - 1] * w[kk + (jindex[jacc - 1] - 1)];
        }
        atomicAdd_double(&f[iidx - 1], -sum);
      }
    }
  }
}

// Parallel heavy–heavy kernel: one thread per HH pair, 100 w-terms
__global__ void dfock2_hh_parallel_kernel(int norbs, int mpack, int numat,
                                          const int *nfirst, const int *nlast,
                                          const double *ptot, const double *p,
                                          const double *w, int nati,
                                          const int *jlist, const int *woff,
                                          int nn, double *f) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nn) return;
  int ii = nati;
  int ia = nfirst[ii - 1];
  int ib = nlast[ii - 1];
  int jj = jlist[tid];
  int ja = nfirst[jj - 1];
  int jb = nlast[jj - 1];
  if ((ib - ia) < 0 || (jb - ja) < 0) return;
  int kk = woff[tid];
  // Build pja/pjb
  double pja[16], pjb[16], pk[16];
  int mloc = 0;
  for (int j = ia; j <= ib; ++j) {
    for (int k = ia; k <= ib; ++k) {
      mloc += 1;
      int jk = (j < k) ? j : k;
      int kj = j + k - jk;
      int packed = jk + ifact_idx(kj);
      pja[mloc - 1] = ptot[packed - 1];
    }
  }
  mloc = 0;
  for (int j = ja; j <= jb; ++j) {
    for (int k = ja; k <= jb; ++k) {
      mloc += 1;
      int jk = (j < k) ? j : k;
      int kj = j + k - jk;
      int packed = jk + ifact_idx(kj);
      pjb[mloc - 1] = ptot[packed - 1];
    }
  }
  // pk
  if (ia > ja) {
    int l = 0;
    for (int i2 = ia; i2 <= ib; ++i2) {
      int base = ifact_idx(i2) + ja;
      for (int col = 0; col <= (jb - ja); ++col) {
        l += 1; pk[l - 1] = p[(base + col) - 1];
      }
    }
  } else {
    int l = 0;
    for (int i2 = ia; i2 <= ib; ++i2) {
      for (int j2 = ja; j2 <= jb; ++j2) {
        l += 1; pk[l - 1] = p[(ifact_idx(j2) + i2) - 1];
      }
    }
  }
  // Apply updates using atomic add inside helpers
  jab_update(ia, ja, pja, pjb, w + kk, f);
  kab_update(ia, ja, pk, w + kk, f);
}

// Packed inputs; currently supports only all light-light pairs for atom nati.
// Returns true if GPU handled the update; false to fall back to CPU dfock2.
bool mopac_cuda_fock2(int norbs, int mpack, int numat,
                      const int *nfirst, const int *nlast,
                      const double *ptot, const double *p,
                      const double *w, int nati,
                      double *f) {
  (void)norbs; (void)mpack;
  // Pre-scan to categorize pairs and estimate w length
  int ia = nfirst[nati - 1];
  int ib = nlast[nati - 1];
  if ((ib - ia) < 0) return false; // sparkle
  size_t w_len = 0;
  int nn_pairs = 0;
  bool all_ll = true;
  for (int jj = 1; jj <= numat; ++jj) {
    if (jj == nati) continue;
    int ja = nfirst[jj - 1];
    int jb = nlast[jj - 1];
    if ((jb - ja) < 0) continue; // sparkle
    int di = ib - ia;
    int dj = jb - ja;
    if (di >= 3 && dj >= 3) { w_len += 100; all_ll = false; }
    else if (di >= 3 || dj >= 3) { w_len += 10; all_ll = false; }
    else { w_len += 1; }
    nn_pairs++;
  }
  if (w_len == 0) return true; // nothing to do

  // Stage inputs to persistent device buffers
  // Predeclare timing and host list buffers before any goto to satisfy nvcc
  cudaEvent_t t_all_start = nullptr, t_all_stop = nullptr; float ms_all = 0.f;
  int *h_ll_j = nullptr, *h_ll_off = nullptr;
  int *h_lh_j = nullptr, *h_lh_off = nullptr;
  int *h_hh_j = nullptr, *h_hh_off = nullptr;
  int ll_count = 0, lh_count = 0, hh_count = 0;
  size_t atoms_e = (size_t)numat;
  size_t mpack_e = (size_t)mpack;
  size_t w_e = w_len;
  cudaError_t e;
  if (!ensure_buf_int(&s_d_nf, &cap_atoms, atoms_e)) goto FAIL;
  if (!ensure_buf_int(&s_d_nl, &cap_atoms, atoms_e)) goto FAIL;
  if (!ensure_buf_double(&s_d_ptot, &cap_mpack, mpack_e)) goto FAIL;
  if (!ensure_buf_double(&s_d_p, &cap_mpack, mpack_e)) goto FAIL;
  if (!ensure_buf_double(&s_d_f, &cap_mpack, mpack_e)) goto FAIL;
  if (!ensure_buf_double(&s_d_w, &cap_w, w_e)) goto FAIL;

  // Copy arrays
  e = cudaMemcpy(s_d_nf, nfirst, sizeof(int)*atoms_e, cudaMemcpyHostToDevice); if (e!=cudaSuccess) goto FAIL;
  e = cudaMemcpy(s_d_nl, nlast, sizeof(int)*atoms_e, cudaMemcpyHostToDevice); if (e!=cudaSuccess) goto FAIL;
  e = cudaMemcpy(s_d_ptot, ptot, sizeof(double)*mpack_e, cudaMemcpyHostToDevice); if (e!=cudaSuccess) goto FAIL;
  e = cudaMemcpy(s_d_p, p, sizeof(double)*mpack_e, cudaMemcpyHostToDevice); if (e!=cudaSuccess) goto FAIL;
  e = cudaMemcpy(s_d_f, f, sizeof(double)*mpack_e, cudaMemcpyHostToDevice); if (e!=cudaSuccess) goto FAIL;
  e = cudaMemcpy(s_d_w, w, sizeof(double)*w_e, cudaMemcpyHostToDevice); if (e!=cudaSuccess) goto FAIL;
  // Total timing start
  if (verbose) { cudaEventCreate(&t_all_start); cudaEventCreate(&t_all_stop); cudaEventRecord(t_all_start); }

  // Build compact pair lists with w offsets
  if (!all_ll) {
    // We still may have a mix of LL and LH without HH
  }
  // Second pass to fill lists and offsets
  {
    int kk_cursor = 0;
    // Count first
    for (int jj = 1; jj <= numat; ++jj) {
      if (jj == nati) continue;
      int ja = nfirst[jj - 1];
      int jb = nlast[jj - 1];
      if ((jb - ja) < 0) continue;
      int di = ib - ia;
      int dj = jb - ja;
      if (di < 3 && dj < 3) { ll_count++; kk_cursor += 1; }
      else if (di >= 3 && dj >= 3) { hh_count++; kk_cursor += 100; }
      else { lh_count++; kk_cursor += 10; }
    }
    if (ll_count > 0) { h_ll_j = (int*)malloc(sizeof(int)*ll_count); h_ll_off = (int*)malloc(sizeof(int)*ll_count); }
    if (lh_count > 0) { h_lh_j = (int*)malloc(sizeof(int)*lh_count); h_lh_off = (int*)malloc(sizeof(int)*lh_count); }
    if (hh_count > 0) { h_hh_j = (int*)malloc(sizeof(int)*hh_count); h_hh_off = (int*)malloc(sizeof(int)*hh_count); }
    // Fill
    kk_cursor = 0; int il = 0, ih = 0;
    for (int jj = 1; jj <= numat; ++jj) {
      if (jj == nati) continue;
      int ja = nfirst[jj - 1];
      int jb = nlast[jj - 1];
      if ((jb - ja) < 0) continue;
      int di = ib - ia;
      int dj = jb - ja;
      if (di < 3 && dj < 3) {
        if (h_ll_j) { h_ll_j[il] = jj; h_ll_off[il] = kk_cursor; il++; }
        kk_cursor += 1;
      } else if (di >= 3 && dj >= 3) {
        if (h_hh_j) { h_hh_j[ih] = jj; h_hh_off[ih] = kk_cursor; ih++; }
        kk_cursor += 100;
      } else {
        if (h_lh_j) { h_lh_j[ih] = jj; h_lh_off[ih] = kk_cursor; ih++; }
        kk_cursor += 10;
      }
    }
  }
  // Launch parallel LL/LH/HH kernels if possible; otherwise serial
  if ((lh_count == 0) && (hh_count == 0) && (ll_count >= 64)) {
    if (verbose) printf("GPU grad: LL-only; counts LL=%d LH=%d HH=%d\n", ll_count, lh_count, hh_count);
    // Parallel LL-only
    int *d_j = nullptr, *d_off = nullptr;
    e = cudaMalloc((void**)&d_j, sizeof(int)*ll_count); if (e!=cudaSuccess) goto SERIAL;
    e = cudaMalloc((void**)&d_off, sizeof(int)*ll_count); if (e!=cudaSuccess) { cudaFree(d_j); goto SERIAL; }
    e = cudaMemcpy(d_j, h_ll_j, sizeof(int)*ll_count, cudaMemcpyHostToDevice); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto SERIAL; }
    e = cudaMemcpy(d_off, h_ll_off, sizeof(int)*ll_count, cudaMemcpyHostToDevice); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto SERIAL; }
    int block = 256, grid = (ll_count + block - 1) / block;
    if (verbose) printf("GPU grad: LL-only parallel, pairs=%d\n", ll_count);
    cudaEvent_t ev1, ev2; float ms=0.f; if (verbose) { cudaEventCreate(&ev1); cudaEventCreate(&ev2); cudaEventRecord(ev1); }
    dfock2_ll_parallel_kernel<<<grid, block>>>(norbs, mpack, numat, s_d_nf, s_d_nl, s_d_ptot, s_d_p, s_d_w, nati, d_j, d_off, ll_count, s_d_f);
    if (verbose) { cudaEventRecord(ev2); cudaEventSynchronize(ev2); cudaEventElapsedTime(&ms, ev1, ev2); printf("GPU grad: LL kernel time = %.3f ms\n", ms); cudaEventDestroy(ev1); cudaEventDestroy(ev2); }
    e = cudaDeviceSynchronize(); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto FAIL; }
    cudaFree(d_j); cudaFree(d_off);
  } else if (ll_count + lh_count + hh_count > 0 && (lh_count >= 32 || ll_count >= 32 || hh_count >= 16)) {
    if (verbose) printf("GPU grad: mixed LL/LH/HH; counts L=%d H=%d HH=%d\n", ll_count, lh_count, hh_count);
    // Mixed LL/LH/HH: launch in sequence
    int *d_j = nullptr, *d_off = nullptr;
    if (ll_count > 0) {
      e = cudaMalloc((void**)&d_j, sizeof(int)*ll_count); if (e!=cudaSuccess) goto SERIAL;
      e = cudaMalloc((void**)&d_off, sizeof(int)*ll_count); if (e!=cudaSuccess) { cudaFree(d_j); goto SERIAL; }
      e = cudaMemcpy(d_j, h_ll_j, sizeof(int)*ll_count, cudaMemcpyHostToDevice); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto SERIAL; }
      e = cudaMemcpy(d_off, h_ll_off, sizeof(int)*ll_count, cudaMemcpyHostToDevice); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto SERIAL; }
      int block = 256, grid = (ll_count + block - 1) / block;
      if (verbose) printf("GPU grad: LL-parallel, pairs=%d\n", ll_count);
      cudaEvent_t e1,e2; float ms=0.f; if (verbose) { cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventRecord(e1); }
      dfock2_ll_parallel_kernel<<<grid, block>>>(norbs, mpack, numat, s_d_nf, s_d_nl, s_d_ptot, s_d_p, s_d_w, nati, d_j, d_off, ll_count, s_d_f);
      if (verbose) { cudaEventRecord(e2); cudaEventSynchronize(e2); cudaEventElapsedTime(&ms, e1, e2); printf("GPU grad: LL kernel time = %.3f ms\n", ms); cudaEventDestroy(e1); cudaEventDestroy(e2); }
      e = cudaDeviceSynchronize(); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto FAIL; }
      cudaFree(d_j); cudaFree(d_off); d_j = nullptr; d_off = nullptr;
    }
    if (lh_count > 0) {
      e = cudaMalloc((void**)&d_j, sizeof(int)*lh_count); if (e!=cudaSuccess) goto SERIAL;
      e = cudaMalloc((void**)&d_off, sizeof(int)*lh_count); if (e!=cudaSuccess) { cudaFree(d_j); goto SERIAL; }
      e = cudaMemcpy(d_j, h_lh_j, sizeof(int)*lh_count, cudaMemcpyHostToDevice); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto SERIAL; }
      e = cudaMemcpy(d_off, h_lh_off, sizeof(int)*lh_count, cudaMemcpyHostToDevice); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto SERIAL; }
      int block = 256, grid = (lh_count + block - 1) / block;
      if (verbose) printf("GPU grad: LH-parallel, pairs=%d\n", lh_count);
      cudaEvent_t e3,e4; float ms2=0.f; if (verbose) { cudaEventCreate(&e3); cudaEventCreate(&e4); cudaEventRecord(e3); }
      dfock2_lh_parallel_kernel<<<grid, block>>>(norbs, mpack, numat, s_d_nf, s_d_nl, s_d_ptot, s_d_p, s_d_w, nati, d_j, d_off, lh_count, s_d_f);
      if (verbose) { cudaEventRecord(e4); cudaEventSynchronize(e4); cudaEventElapsedTime(&ms2, e3, e4); printf("GPU grad: LH kernel time = %.3f ms\n", ms2); cudaEventDestroy(e3); cudaEventDestroy(e4); }
      e = cudaDeviceSynchronize(); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto FAIL; }
      cudaFree(d_j); cudaFree(d_off);
    }
    if (hh_count > 0) {
      e = cudaMalloc((void**)&d_j, sizeof(int)*hh_count); if (e!=cudaSuccess) goto SERIAL;
      e = cudaMalloc((void**)&d_off, sizeof(int)*hh_count); if (e!=cudaSuccess) { cudaFree(d_j); goto SERIAL; }
      e = cudaMemcpy(d_j, h_hh_j, sizeof(int)*hh_count, cudaMemcpyHostToDevice); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto SERIAL; }
      e = cudaMemcpy(d_off, h_hh_off, sizeof(int)*hh_count, cudaMemcpyHostToDevice); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto SERIAL; }
      int block = 128, grid = (hh_count + block - 1) / block;
      if (verbose) printf("GPU grad: HH-parallel, pairs=%d\n", hh_count);
      cudaEvent_t e5,e6; float ms3=0.f; if (verbose) { cudaEventCreate(&e5); cudaEventCreate(&e6); cudaEventRecord(e5); }
      dfock2_hh_parallel_kernel<<<grid, block>>>(norbs, mpack, numat, s_d_nf, s_d_nl, s_d_ptot, s_d_p, s_d_w, nati, d_j, d_off, hh_count, s_d_f);
      if (verbose) { cudaEventRecord(e6); cudaEventSynchronize(e6); cudaEventElapsedTime(&ms3, e5, e6); printf("GPU grad: HH kernel time = %.3f ms\n", ms3); cudaEventDestroy(e5); cudaEventDestroy(e6); }
      e = cudaDeviceSynchronize(); if (e!=cudaSuccess) { cudaFree(d_j); cudaFree(d_off); goto FAIL; }
      cudaFree(d_j); cudaFree(d_off);
    }
  } else {
SERIAL:
    if (verbose) printf("GPU grad: serial kernel; counts LL=%d LH=%d HH=%d\n", ll_count, lh_count, hh_count);
    dfock2_ll_lh_kernel<<<1,1>>>(norbs, mpack, numat, s_d_nf, s_d_nl, s_d_ptot, s_d_p, s_d_w, nati, s_d_f);
    e = cudaDeviceSynchronize(); if (e!=cudaSuccess) goto FAIL;
  }

  // Total timing stop and summary
  if (verbose) {
    cudaEventRecord(t_all_stop); cudaEventSynchronize(t_all_stop);
    cudaEventElapsedTime(&ms_all, t_all_start, t_all_stop);
    printf("GPU grad: atom %d total = %.3f ms (LL=%d LH=%d HH=%d)\n", nati, ms_all, ll_count, lh_count, hh_count);
    cudaEventDestroy(t_all_start); cudaEventDestroy(t_all_stop);
  }

  if (h_ll_j) free(h_ll_j);
  if (h_ll_off) free(h_ll_off);
  if (h_lh_j) free(h_lh_j);
  if (h_lh_off) free(h_lh_off);
  if (h_hh_j) free(h_hh_j);
  if (h_hh_off) free(h_hh_off);

  // Copy updated F back
  e = cudaMemcpy(f, s_d_f, sizeof(double)*mpack_e, cudaMemcpyDeviceToHost); if (e!=cudaSuccess) goto FAIL;
  return true;

FAIL:
  return false;
}

}

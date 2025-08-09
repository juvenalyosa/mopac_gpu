// Portable CUDA interop for MOPAC: cuBLAS GEMM, cuSOLVER SYEVD, and basic GPU info
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

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
  void ensure(size_t bytes) {
    if (bytes <= cap && ptr) return;
    if (ptr) cudaFreeHost(ptr);
    ptr = nullptr; cap = 0;
    if (bytes > 0) {
      cudaHostAlloc((void**)&ptr, bytes, cudaHostAllocDefault);
      cap = bytes;
    }
  }
  void release() {
    if (ptr) cudaFreeHost(ptr);
    ptr = nullptr; cap = 0;
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

// Select device by index
void setDevice(int idevice, bool *stat) {
  cudaError_t cerr = cudaSetDevice(idevice);
  if (stat) *stat = (cerr == cudaSuccess);
}

// Global cuBLAS handle
static cublasHandle_t g_blas = nullptr;
static cudaStream_t   g_stream = nullptr;     // single-GPU general stream
static cudaStream_t   g_stream0 = nullptr;    // 2-GPU device0 stream
static cudaStream_t   g_stream1 = nullptr;    // 2-GPU device1 stream
static bool           g_streams_enabled = true;

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
// 2-GPU caches
static DevBuf<double> g2_gemm_a0, g2_gemm_b0, g2_gemm_c0;
static DevBuf<double> g2_gemm_a1, g2_gemm_b1, g2_gemm_c1;
static DevBuf<double> g2_syrk_v0, g2_syrk_c0;
static DevBuf<double> g2_syrk_v1, g2_syrk_c1;
static HostBuf<double> h2_gemm_A, h2_gemm_B, h2_gemm_C;
static HostBuf<double> h2_syrk_A, h2_syrk_C;
static HostBuf<double> h2_rot_V;

void create_handle() {
  if (!g_blas) {
    cublasCreate(&g_blas);
    const char* env = std::getenv("MOPAC_STREAMS");
    if (env) {
      if (std::strcmp(env, "off") == 0 || std::strcmp(env, "0") == 0) {
        g_streams_enabled = false;
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

void destroy_handle() {
  if (g_blas) {
    cublasDestroy(g_blas);
    g_blas = nullptr;
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
  if (!g_blas) create_handle();
  cublasOperation_t opA = (tra == 'T' || tra == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = (trb == 'T' || trb == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  // Allocate device buffers and copy inputs
  size_t bytesA = (size_t)lda * (size_t)k * sizeof(double);
  size_t bytesB = (size_t)ldb * (size_t)n * sizeof(double);
  size_t bytesC = (size_t)ldc * (size_t)n * sizeof(double);
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
  std::memcpy(h_gemm_C.ptr, C, bytesC);
  cudaMemcpyAsync(d_A, h_gemm_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream);
  cudaMemcpyAsync(d_B, h_gemm_B.ptr, bytesB, cudaMemcpyHostToDevice, g_stream);
  cudaMemcpyAsync(d_C, h_gemm_C.ptr, bytesC, cudaMemcpyHostToDevice, g_stream);
  // Compute C = alpha*op(A)*op(B) + beta*C
  cublasDgemm(g_blas, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
  cudaMemcpyAsync(h_gemm_C.ptr, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
  std::memcpy(C, h_gemm_C.ptr, bytesC);
}

// SYRK via cuBLAS (uses cached device buffers)
void call_syrk_cublas(char uplo, char tra,
                      int n, int k,
                      double alpha,
                      const double *A, int lda,
                      double beta,
                      double *C, int ldc) {
  if (!g_blas) create_handle();
  cublasFillMode_t u = (uplo == 'U' || uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t opA = (tra == 'T' || tra == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  size_t bytesA = (size_t)lda * (size_t)((opA==CUBLAS_OP_N)?k:n) * sizeof(double);
  size_t bytesC = (size_t)ldc * (size_t)n * sizeof(double);
  g_syrk_A.ensure(bytesA);
  g_syrk_C.ensure(bytesC);
  double *d_A = g_syrk_A.ptr;
  double *d_C = g_syrk_C.ptr;
  h_syrk_A.ensure(bytesA);
  h_syrk_C.ensure(bytesC);
  std::memcpy(h_syrk_A.ptr, A, bytesA);
  std::memcpy(h_syrk_C.ptr, C, bytesC);
  cudaMemcpyAsync(d_A, h_syrk_A.ptr, bytesA, cudaMemcpyHostToDevice, g_stream);
  cudaMemcpyAsync(d_C, h_syrk_C.ptr, bytesC, cudaMemcpyHostToDevice, g_stream);
  cublasDsyrk(g_blas, u, opA, n, k, &alpha, d_A, lda, &beta, d_C, ldc);
  cudaMemcpyAsync(h_syrk_C.ptr, d_C, bytesC, cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
  std::memcpy(C, h_syrk_C.ptr, bytesC);
}

// 2-GPU outer product helpers and wrappers are further below.

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
  ensure_pair_streams();
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
  ensure_pair_streams();
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
// Symmetric eigensolver (upper triangle) using cuSOLVER Dsyevd; A overwritten with eigenvectors
// Cached handles and workspaces for Dsyevd
static cusolverDnHandle_t g_solver = nullptr;
static DevBuf<double> g_dsyevd_A, g_dsyevd_W, g_dsyevd_work;
static DevBuf<int>    g_dsyevd_info;
static int g_dsyevd_lwork_cap = 0; // elements, not bytes

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
  // Copy top slice rows [0..n0)
  for (int col = 0; col < n; ++col) {
    const double *col_ptr = h2_rot_V.ptr + (size_t)col * (size_t)n;
    cudaMemcpyAsync(d_V0 + (size_t)col * (size_t)n0, col_ptr, sizeof(double) * n0,
                    cudaMemcpyHostToDevice);
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
void mopac_cuda_destroy_resources() {
  static bool already = false;
  if (already) return;
  already = true;
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
  // Release cached pinned host buffers
  h_gemm_A.release(); h_gemm_B.release(); h_gemm_C.release();
  h_syrk_A.release(); h_syrk_C.release();
  // DSYEVD stages are static locals; nothing to release here on purpose
  // 2-GPU caches
  g2_gemm_a0.release(); g2_gemm_b0.release(); g2_gemm_c0.release();
  g2_gemm_a1.release(); g2_gemm_b1.release(); g2_gemm_c1.release();
  g2_syrk_v0.release(); g2_syrk_c0.release(); g2_syrk_v1.release(); g2_syrk_c1.release();
  h2_gemm_A.release(); h2_gemm_B.release(); h2_gemm_C.release();
  h2_syrk_A.release(); h2_syrk_C.release();
}

} // extern "C"

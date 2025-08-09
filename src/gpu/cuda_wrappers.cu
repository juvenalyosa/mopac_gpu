// Portable CUDA interop for MOPAC: cuBLAS GEMM, cuSOLVER SYEVD, and basic GPU info
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstring>
#include <cmath>
#include <algorithm>

// Default device pair for 2-GPU MOZYME operations
static int g_pair_dev0 = 0;
static int g_pair_dev1 = 1;

extern "C" {

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

void create_handle() {
  if (!g_blas) {
    cublasCreate(&g_blas);
  }
}

void destroy_handle() {
  if (g_blas) {
    cublasDestroy(g_blas);
    g_blas = nullptr;
  }
}

// Fortran-callable DGEMM via cuBLAS
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
  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc((void**)&d_A, bytesA);
  cudaMalloc((void**)&d_B, bytesB);
  cudaMalloc((void**)&d_C, bytesC);
  cudaMemcpy(d_A, A, bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, bytesB, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, bytesC, cudaMemcpyHostToDevice);
  // Compute C = alpha*op(A)*op(B) + beta*C
  cublasDgemm(g_blas, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
  cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// SYRK via cuBLAS
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
  double *d_A = nullptr, *d_C = nullptr;
  cudaMalloc((void**)&d_A, bytesA);
  cudaMalloc((void**)&d_C, bytesC);
  cudaMemcpy(d_A, A, bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, bytesC, cudaMemcpyHostToDevice);
  cublasDsyrk(g_blas, u, opA, n, k, &alpha, d_A, lda, &beta, d_C, ldc);
  cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_C);
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
  int n0 = m / 2;
  int n1 = m - n0;
  // Device allocations and copies as above
  double *d_a0=nullptr, *d_b0=nullptr, *d_c0=nullptr;
  double *d_a1=nullptr, *d_b1=nullptr, *d_c1=nullptr;
  // Device 0
  cudaSetDevice(g_pair_dev0);
  cudaMalloc((void**)&d_a0, sizeof(double) * (size_t)m);
  cudaMalloc((void**)&d_b0, sizeof(double) * (size_t)n);
  cudaMalloc((void**)&d_c0, sizeof(double) * (size_t)n0 * (size_t)ldc);
  cudaMemcpy(d_a0, A, sizeof(double) * (size_t)m, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b0, B, sizeof(double) * (size_t)n, cudaMemcpyHostToDevice);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(d_c0 + (size_t)col * (size_t)n0,
                    C + (size_t)col * (size_t)ldc,
                    sizeof(double) * (size_t)n0,
                    cudaMemcpyHostToDevice);
  }
  // Device 1
  cudaSetDevice(g_pair_dev1);
  cudaMalloc((void**)&d_a1, sizeof(double) * (size_t)m);
  cudaMalloc((void**)&d_b1, sizeof(double) * (size_t)n);
  cudaMalloc((void**)&d_c1, sizeof(double) * (size_t)n1 * (size_t)ldc);
  cudaMemcpy(d_a1, A, sizeof(double) * (size_t)m, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, B, sizeof(double) * (size_t)n, cudaMemcpyHostToDevice);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(d_c1 + (size_t)col * (size_t)n1,
                    C + (size_t)col * (size_t)ldc + (size_t)n0,
                    sizeof(double) * (size_t)n1,
                    cudaMemcpyHostToDevice);
  }

  // Launch kernels
  cudaSetDevice(g_pair_dev0);
  {
    int rows = n0;
    int total = rows * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    outer_update_rows<<<grid, block>>>(d_c0, rows, n, d_a0, d_b0, alpha, beta, 0);
  }
  cudaSetDevice(g_pair_dev1);
  {
    int rows = n1;
    int total = rows * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    outer_update_rows<<<grid, block>>>(d_c1, rows, n, d_a1, d_b1, alpha, beta, n0);
  }

  // Sync and copy back row slices
  cudaSetDevice(g_pair_dev0); cudaDeviceSynchronize();
  for (int col = 0; col < n; ++col) {
    cudaMemcpy(C + (size_t)col * (size_t)ldc,
               d_c0 + (size_t)col * (size_t)n0,
               sizeof(double) * (size_t)n0,
               cudaMemcpyDeviceToHost);
  }
  cudaSetDevice(g_pair_dev1); cudaDeviceSynchronize();
  for (int col = 0; col < n; ++col) {
    cudaMemcpy(C + (size_t)col * (size_t)ldc + (size_t)n0,
               d_c1 + (size_t)col * (size_t)n1,
               sizeof(double) * (size_t)n1,
               cudaMemcpyDeviceToHost);
  }

  // Free
  cudaSetDevice(g_pair_dev0);
  cudaFree(d_a0); cudaFree(d_b0); cudaFree(d_c0);
  cudaSetDevice(g_pair_dev1);
  cudaFree(d_a1); cudaFree(d_b1); cudaFree(d_c1);
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
  int n0 = n / 2;
  int n1 = n - n0;
  // Copy full vector v to both devices and split C by rows
  double *d_v0=nullptr, *d_c0=nullptr;
  double *d_v1=nullptr, *d_c1=nullptr;
  // Device 0
  cudaSetDevice(g_pair_dev0);
  cudaMalloc((void**)&d_v0, sizeof(double) * (size_t)n);
  cudaMalloc((void**)&d_c0, sizeof(double) * (size_t)n0 * (size_t)ldc);
  cudaMemcpy(d_v0, A, sizeof(double) * (size_t)n, cudaMemcpyHostToDevice);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(d_c0 + (size_t)col * (size_t)n0,
                    C + (size_t)col * (size_t)ldc,
                    sizeof(double) * (size_t)n0,
                    cudaMemcpyHostToDevice);
  }
  // Device 1
  cudaSetDevice(g_pair_dev1);
  cudaMalloc((void**)&d_v1, sizeof(double) * (size_t)n);
  cudaMalloc((void**)&d_c1, sizeof(double) * (size_t)n1 * (size_t)ldc);
  cudaMemcpy(d_v1, A, sizeof(double) * (size_t)n, cudaMemcpyHostToDevice);
  for (int col = 0; col < n; ++col) {
    cudaMemcpyAsync(d_c1 + (size_t)col * (size_t)n1,
                    C + (size_t)col * (size_t)ldc + (size_t)n0,
                    sizeof(double) * (size_t)n1,
                    cudaMemcpyHostToDevice);
  }

  // Launch outer product kernels per device
  cudaSetDevice(g_pair_dev0);
  {
    int rows = n0;
    int total = rows * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    outer_update_rows<<<grid, block>>>(d_c0, rows, n, d_v0, d_v0, alpha, beta, 0);
  }
  cudaSetDevice(g_pair_dev1);
  {
    int rows = n1;
    int total = rows * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    outer_update_rows<<<grid, block>>>(d_c1, rows, n, d_v1, d_v1, alpha, beta, n0);
  }

  // Sync and copy back row slices
  cudaSetDevice(g_pair_dev0); cudaDeviceSynchronize();
  for (int col = 0; col < n; ++col) {
    cudaMemcpy(C + (size_t)col * (size_t)ldc,
               d_c0 + (size_t)col * (size_t)n0,
               sizeof(double) * (size_t)n0,
               cudaMemcpyDeviceToHost);
  }
  cudaSetDevice(g_pair_dev1); cudaDeviceSynchronize();
  for (int col = 0; col < n; ++col) {
    cudaMemcpy(C + (size_t)col * (size_t)ldc + (size_t)n0,
               d_c1 + (size_t)col * (size_t)n1,
               sizeof(double) * (size_t)n1,
               cudaMemcpyDeviceToHost);
  }

  // Free
  cudaSetDevice(g_pair_dev0);
  cudaFree(d_v0); cudaFree(d_c0);
  cudaSetDevice(g_pair_dev1);
  cudaFree(d_v1); cudaFree(d_c1);
}
// Symmetric eigensolver (upper triangle) using cuSOLVER Dsyevd; A overwritten with eigenvectors
void mopac_cuda_dsyevd(int n, double *A, int lda, double *W, int *info) {
  cusolverDnHandle_t handle = nullptr;
  cusolverDnCreate(&handle);

  double *d_A = nullptr;
  double *d_W = nullptr;
  int *d_info = nullptr;
  int lwork = 0;
  cudaMalloc((void **)&d_A, sizeof(double) * lda * n);
  cudaMalloc((void **)&d_W, sizeof(double) * n);
  cudaMalloc((void **)&d_info, sizeof(int));

  cudaMemcpy(d_A, A, sizeof(double) * lda * n, cudaMemcpyHostToDevice);

  cusolverDnDsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                              n, d_A, lda, d_W, &lwork);
  double *d_work = nullptr;
  cudaMalloc((void **)&d_work, sizeof(double) * lwork);

  cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                   n, d_A, lda, d_W, d_work, lwork, d_info);

  cudaMemcpy(A, d_A, sizeof(double) * lda * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(W, d_W, sizeof(double) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_work);
  cudaFree(d_A);
  cudaFree(d_W);
  cudaFree(d_info);
  cusolverDnDestroy(handle);
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

void call_rot_cuda_gpu(const double *fmo, const double *eig,
                       double *vector, const double *ci0, const double *ca0,
                       int nocc, int lumo, int n,
                       double bigeps, double tiny) {
  (void)ci0; (void)ca0; // unused for now
  // Allocate and copy eigenvector matrix to device
  double *d_V = nullptr;
  cudaMalloc((void**)&d_V, sizeof(double) * (size_t)n * (size_t)n);
  cudaMemcpy(d_V, vector, sizeof(double) * (size_t)n * (size_t)n, cudaMemcpyHostToDevice);

  // Walk pairs sequentially; batch to reduce kernel launches
  const int max_batch = 256;
  int   *h_i = (int*)malloc(sizeof(int) * max_batch);
  int   *h_j = (int*)malloc(sizeof(int) * max_batch);
  double *h_a = (double*)malloc(sizeof(double) * max_batch);
  double *h_b = (double*)malloc(sizeof(double) * max_batch);
  int   *d_i = nullptr, *d_j = nullptr;
  double *d_a = nullptr, *d_b = nullptr;
  cudaMalloc((void**)&d_i, sizeof(int) * max_batch);
  cudaMalloc((void**)&d_j, sizeof(int) * max_batch);
  cudaMalloc((void**)&d_a, sizeof(double) * max_batch);
  cudaMalloc((void**)&d_b, sizeof(double) * max_batch);

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
        cudaMemcpy(d_i, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_j, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice);
        int block = 256;
        int grid = (n + block - 1) / block;
        drot_cols_batch_kernel<<<grid, block>>>(d_V, n, batch, d_i, d_j, d_a, d_b);
        batch = 0;
      }
    }
    if (batch > 0) {
      cudaMemcpy(d_i, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice);
      cudaMemcpy(d_j, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice);
      cudaMemcpy(d_a, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice);
      int block = 256;
      int grid = (n + block - 1) / block;
      drot_cols_batch_kernel<<<grid, block>>>(d_V, n, batch, d_i, d_j, d_a, d_b);
    }
  }
  cudaDeviceSynchronize();
  // Copy back result
  cudaMemcpy(vector, d_V, sizeof(double) * (size_t)n * (size_t)n, cudaMemcpyDeviceToHost);
  cudaFree(d_V);
  cudaFree(d_i);
  cudaFree(d_j);
  cudaFree(d_a);
  cudaFree(d_b);
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

  // Allocate device slices
  double *d_V0 = nullptr, *d_V1 = nullptr;
  int *d_i0 = nullptr, *d_j0 = nullptr, *d_i1 = nullptr, *d_j1 = nullptr;
  double *d_a0 = nullptr, *d_b0 = nullptr, *d_a1 = nullptr, *d_b1 = nullptr;

  cudaSetDevice(0);
  cudaMalloc((void**)&d_V0, bytes0);
  cudaMalloc((void**)&d_i0, sizeof(int) * 256);
  cudaMalloc((void**)&d_j0, sizeof(int) * 256);
  cudaMalloc((void**)&d_a0, sizeof(double) * 256);
  cudaMalloc((void**)&d_b0, sizeof(double) * 256);
  // Copy top slice rows [0..n0)
  for (int col = 0; col < n; ++col) {
    const double *col_ptr = vector + (size_t)col * (size_t)n;
    cudaMemcpyAsync(d_V0 + (size_t)col * (size_t)n0, col_ptr, sizeof(double) * n0,
                    cudaMemcpyHostToDevice);
  }

  cudaSetDevice(1);
  cudaMalloc((void**)&d_V1, bytes1);
  cudaMalloc((void**)&d_i1, sizeof(int) * 256);
  cudaMalloc((void**)&d_j1, sizeof(int) * 256);
  cudaMalloc((void**)&d_a1, sizeof(double) * 256);
  cudaMalloc((void**)&d_b1, sizeof(double) * 256);
  // Copy bottom slice rows [n0..n)
  for (int col = 0; col < n; ++col) {
    const double *col_ptr = vector + (size_t)col * (size_t)n + (size_t)n0;
    cudaMemcpyAsync(d_V1 + (size_t)col * (size_t)n1, col_ptr, sizeof(double) * n1,
                    cudaMemcpyHostToDevice);
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
        // Launch on device 0
        cudaSetDevice(0);
        cudaMemcpy(d_i0, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_j0, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_a0, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b0, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice);
        {
          int block = 256;
          int grid = (n0 + block - 1) / block;
          drot_cols_batch_kernel_strided<<<grid, block>>>(d_V0, n0, n, batch, d_i0, d_j0, d_a0, d_b0);
        }
        // Launch on device 1
        cudaSetDevice(1);
        cudaMemcpy(d_i1, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_j1, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_a1, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice);
        {
          int block = 256;
          int grid = (n1 + block - 1) / block;
          drot_cols_batch_kernel_strided<<<grid, block>>>(d_V1, n1, n, batch, d_i1, d_j1, d_a1, d_b1);
        }
        batch = 0;
      }
    }
    if (batch > 0) {
      cudaSetDevice(0);
      cudaMemcpy(d_i0, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice);
      cudaMemcpy(d_j0, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice);
      cudaMemcpy(d_a0, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice);
      cudaMemcpy(d_b0, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice);
      {
        int block = 256;
        int grid = (n0 + block - 1) / block;
        drot_cols_batch_kernel_strided<<<grid, block>>>(d_V0, n0, n, batch, d_i0, d_j0, d_a0, d_b0);
      }
      cudaSetDevice(1);
      cudaMemcpy(d_i1, h_i, sizeof(int) * batch, cudaMemcpyHostToDevice);
      cudaMemcpy(d_j1, h_j, sizeof(int) * batch, cudaMemcpyHostToDevice);
      cudaMemcpy(d_a1, h_a, sizeof(double) * batch, cudaMemcpyHostToDevice);
      cudaMemcpy(d_b1, h_b, sizeof(double) * batch, cudaMemcpyHostToDevice);
      {
        int block = 256;
        int grid = (n1 + block - 1) / block;
        drot_cols_batch_kernel_strided<<<grid, block>>>(d_V1, n1, n, batch, d_i1, d_j1, d_a1, d_b1);
      }
    }
  }

  // Synchronize both devices
  cudaSetDevice(0);
  cudaDeviceSynchronize();
  cudaSetDevice(1);
  cudaDeviceSynchronize();

  // Copy results back into host matrix
  cudaSetDevice(0);
  for (int col = 0; col < n; ++col) {
    double *col_ptr = vector + (size_t)col * (size_t)n;
    cudaMemcpy(col_ptr, d_V0 + (size_t)col * (size_t)n0, sizeof(double) * n0,
               cudaMemcpyDeviceToHost);
  }
  cudaSetDevice(1);
  for (int col = 0; col < n; ++col) {
    double *col_ptr = vector + (size_t)col * (size_t)n + (size_t)n0;
    cudaMemcpy(col_ptr, d_V1 + (size_t)col * (size_t)n1, sizeof(double) * n1,
               cudaMemcpyDeviceToHost);
  }

  // Cleanup
  cudaSetDevice(0);
  cudaFree(d_V0);
  cudaFree(d_i0);
  cudaFree(d_j0);
  cudaFree(d_a0);
  cudaFree(d_b0);
  cudaSetDevice(1);
  cudaFree(d_V1);
  cudaFree(d_i1);
  cudaFree(d_j1);
  cudaFree(d_a1);
  cudaFree(d_b1);
  free(h_i);
  free(h_j);
  free(h_a);
  free(h_b);
}

} // extern "C"
// Allow Fortran to set the 2-GPU device pair (0-based device indices)
void set_mozyme_gpu_pair(int dev0, int dev1) {
  if (dev0 >= 0 && dev1 >= 0 && dev0 != dev1) {
    g_pair_dev0 = dev0;
    g_pair_dev1 = dev1;
  }
}

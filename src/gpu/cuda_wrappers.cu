// Portable CUDA interop for MOPAC: cuBLAS GEMM, cuSOLVER SYEVD, and basic GPU info
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstring>

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
  // Note: Arguments follow Fortran column-major order; cuBLAS expects column-major by default
  cublasDgemm(g_blas, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
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
  cublasDsyrk(g_blas, u, opA, n, k, &alpha, A, lda, &beta, C, ldc);
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

static void rot_apply_pair_on_device(double *d_V, int n, int i_idx, int j_idx, double alpha, double beta) {
  int block = 256;
  int grid = (n + block - 1) / block;
  drot_cols_kernel<<<grid, block>>>(d_V, n, i_idx, j_idx, alpha, beta);
}

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

void call_rot_cuda_2gpu_gpu(const double *fmo, const double *eig,
                            double *vector, const double *ci0, const double *ca0,
                            int nocc, int lumo, int n,
                            double bigeps, double tiny) {
  // For now, use single GPU path
  call_rot_cuda_gpu(fmo, eig, vector, ci0, ca0, nocc, lumo, n, bigeps, tiny);
}

} // extern "C"

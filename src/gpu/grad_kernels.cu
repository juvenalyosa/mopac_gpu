// Stage-1 CUDA scaffolding for SCF gradient accumulation.
// Currently provides a Coulomb-only experimental kernel guarded by
// MOPAC_GPU_GRAD_EXPERIMENTAL. The default behaviour is to return false so
// callers can fall back to the trusted CPU implementation.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace {

struct GradPairPod {
  int atom_i;
  int atom_j;
  int span_i_first;
  int span_i_last;
  int span_j_first;
  int span_j_last;
  int image_code;
  int flags;
  double displacement[3];
  double distance2;
  double weight;
};

static GradPairPod *d_near_pairs = nullptr;
static size_t near_capacity = 0;
static double *d_charges = nullptr;
static size_t charge_capacity = 0;
static double *d_grad = nullptr;
static size_t grad_capacity = 0;
static cudaStream_t grad_stream = nullptr;
static int experimental_mode = -1;  // -1 unset, 0 disabled, 1 enabled

constexpr double kCoulombKcalPerAng = 332.063712949;  // kcal/mol * Å / e^2

inline bool experimental_enabled() {
  if (experimental_mode >= 0) return experimental_mode == 1;
  const char *env = std::getenv("MOPAC_GPU_GRAD_EXPERIMENTAL");
  if (env && env[0] != '\0' && env[0] != '0') {
    experimental_mode = 1;
  } else {
    experimental_mode = 0;
  }
  return experimental_mode == 1;
}

template <typename T>
bool ensure_capacity(T *&ptr, size_t &capacity, size_t need) {
  if (need <= capacity) return true;
  if (ptr) cudaFree(ptr);
  ptr = nullptr;
  capacity = 0;
  if (need == 0) return true;
  if (cudaMalloc(reinterpret_cast<void **>(&ptr), need * sizeof(T)) != cudaSuccess) {
    ptr = nullptr;
    capacity = 0;
    return false;
  }
  capacity = need;
  return true;
}

inline bool ensure_stream() {
  if (grad_stream) return true;
  if (cudaStreamCreateWithFlags(&grad_stream, cudaStreamNonBlocking) != cudaSuccess) {
    grad_stream = nullptr;
    return false;
  }
  return true;
}

__global__ void coulomb_gradient_kernel(int pair_count,
                                        const GradPairPod *pairs,
                                        const double *charges,
                                        double *grad)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= pair_count) return;
  const GradPairPod pair = pairs[tid];
  int atom_i = pair.atom_i - 1;  // Fortran -> C indexing
  int atom_j = pair.atom_j - 1;
  if (atom_i < 0 || atom_j < 0) return;
  double dx = pair.displacement[0];
  double dy = pair.displacement[1];
  double dz = pair.displacement[2];
  double r2 = pair.distance2;
  if (r2 <= 1.0e-20) return;
  double inv_r = 1.0 / ::sqrt(r2);
  double inv_r3 = inv_r * inv_r * inv_r;
  double weight = pair.weight;
  double qi = charges[atom_i];
  double qj = charges[atom_j];
  double scale = kCoulombKcalPerAng * qi * qj * inv_r3 * weight;
  double gx = scale * dx;
  double gy = scale * dy;
  double gz = scale * dz;
  int offset_i = atom_i * 3;
  int offset_j = atom_j * 3;
  atomicAdd(&grad[offset_i + 0], gx);
  atomicAdd(&grad[offset_i + 1], gy);
  atomicAdd(&grad[offset_i + 2], gz);
  atomicAdd(&grad[offset_j + 0], -gx);
  atomicAdd(&grad[offset_j + 1], -gy);
  atomicAdd(&grad[offset_j + 2], -gz);
}

}  // namespace

extern "C" bool mopac_cuda_cart_gradient_launch(int numat,
                                                 int l123,
                                                 const double *coord_host,
                                                 double *grad_host,
                                                 const double *charges_host,
                                                 const void *near_pairs,
                                                 int near_count,
                                                 const void *far_pairs,
                                                 int far_count)
{
  (void)far_pairs;
  (void)far_count;
  if (numat <= 0 || !coord_host || !grad_host || !charges_host || !near_pairs) {
    return false;
  }
  if (l123 != 1) return false;
  if (!experimental_enabled()) return false;
  if (!ensure_stream()) return false;
  size_t grad_elems = static_cast<size_t>(numat) * 3;
  if (!ensure_capacity(d_charges, charge_capacity, static_cast<size_t>(numat))) return false;
  if (!ensure_capacity(d_grad, grad_capacity, grad_elems)) return false;
  if (!ensure_capacity(d_near_pairs, near_capacity, static_cast<size_t>(near_count))) return false;

  size_t charge_bytes = static_cast<size_t>(numat) * sizeof(double);
  size_t grad_bytes = grad_elems * sizeof(double);
  size_t pair_bytes = static_cast<size_t>(near_count) * sizeof(GradPairPod);

  if (cudaMemsetAsync(d_grad, 0, grad_bytes, grad_stream) != cudaSuccess) return false;
  if (cudaMemcpyAsync(d_charges, charges_host, charge_bytes, cudaMemcpyHostToDevice, grad_stream) != cudaSuccess) return false;
  if (cudaMemcpyAsync(d_near_pairs, near_pairs, pair_bytes, cudaMemcpyHostToDevice, grad_stream) != cudaSuccess) return false;

  int threads = 128;
  int blocks = (near_count + threads - 1) / threads;
  if (blocks > 0) {
    coulomb_gradient_kernel<<<blocks, threads, 0, grad_stream>>>(near_count, d_near_pairs, d_charges, d_grad);
    if (cudaGetLastError() != cudaSuccess) return false;
  }

  if (cudaMemcpyAsync(grad_host, d_grad, grad_bytes, cudaMemcpyDeviceToHost, grad_stream) != cudaSuccess) return false;
  if (cudaStreamSynchronize(grad_stream) != cudaSuccess) return false;
  return true;
}

extern "C" void mopac_cuda_cart_gradient_release(void) {
  if (d_near_pairs) cudaFree(d_near_pairs);
  if (d_charges) cudaFree(d_charges);
  if (d_grad) cudaFree(d_grad);
  d_near_pairs = nullptr;
  d_charges = nullptr;
  d_grad = nullptr;
  near_capacity = 0;
  charge_capacity = 0;
  grad_capacity = 0;
  if (grad_stream) cudaStreamDestroy(grad_stream);
  grad_stream = nullptr;
  experimental_mode = -1;
}

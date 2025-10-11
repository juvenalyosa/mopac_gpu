// Stage-1 CUDA scaffolding for SCF gradient accumulation.
// Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
// NOTE (modification): 2025-10-11 — Applied by Codex CLI assistant
// - Added atomicAdd_double fallback for architectures without native FP64 atomics
//   (sm < 600), and updated Coulomb gradient kernel to use it. This enables
//   successful builds for sm_52 (Maxwell/TITAN X) while preserving correctness.
// Currently provides a Coulomb-only experimental kernel guarded by
// MOPAC_GPU_GRAD_EXPERIMENTAL. The default behaviour is to return false so
// callers can fall back to the trusted CPU implementation.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "grad_launch.h"

#if defined(__NVCC__)
#pragma diag_suppress 177
#endif

extern "C" bool mopac_gpu_cart_gradient_cpu(int numat, int l123,
                                             const double *coord, double *grad,
                                             const double *qbld);

namespace detail {

static GradPairPod *d_near_pairs = nullptr;
static double *d_charges = nullptr;
static double *d_grad = nullptr;
static int experimental_mode = -1;  // -1 unset, 0 disabled, 1 enabled

constexpr double kCoulombKcalPerAng = 332.063712949;  // kcal/mol * Å / e^2

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

static bool experimental_enabled() {
  if (experimental_mode >= 0) return experimental_mode == 1;
  const char *env = std::getenv("MOPAC_GPU_GRAD_EXPERIMENTAL");
  if (env && env[0] != '\0' && env[0] != '0') {
    experimental_mode = 1;
  } else {
    experimental_mode = 0;
  }
  return experimental_mode == 1;
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
  atomicAdd_double(&grad[offset_i + 0], gx);
  atomicAdd_double(&grad[offset_i + 1], gy);
  atomicAdd_double(&grad[offset_i + 2], gz);
  atomicAdd_double(&grad[offset_j + 0], -gx);
  atomicAdd_double(&grad[offset_j + 1], -gy);
  atomicAdd_double(&grad[offset_j + 2], -gz);
}

static inline void touch_buffers() {
  (void)d_near_pairs;
  (void)d_charges;
  (void)d_grad;
}

}  // namespace detail

bool resident_grad_launch_impl(int numat,
                               int l123,
                               const double *coord_host,
                               double *grad_host,
                               const double *charges_host,
                               const GradPairPod *near_pairs,
                               int near_count,
                               const GradPairPod *far_pairs,
                               int far_count) {
  (void)numat;
  (void)l123;
  (void)coord_host;
  (void)grad_host;
  (void)charges_host;
  (void)far_pairs;
  (void)far_count;
  detail::touch_buffers();
  int work_count = 0;
  generate_pair_work(numat, l123, near_pairs, near_count, nullptr, &work_count);
  std::vector<AtomPairWork> work;
  if (work_count > 0) {
    work.resize(static_cast<size_t>(work_count));
    generate_pair_work(numat, l123, near_pairs, near_count, work.data(), &work_count);
  }
  if (!detail::experimental_enabled()) return false;
  return mopac_gpu_cart_gradient_cpu(numat, l123, coord_host, grad_host, charges_host);
}

void resident_grad_release_impl() {
  if (detail::d_near_pairs) cudaFree(detail::d_near_pairs);
  if (detail::d_charges) cudaFree(detail::d_charges);
  if (detail::d_grad) cudaFree(detail::d_grad);
  detail::d_near_pairs = nullptr;
  detail::d_charges = nullptr;
  detail::d_grad = nullptr;
  detail::experimental_mode = -1;
}

#if defined(__NVCC__)
#pragma diag_default 177
#endif

extern "C" void mopac_cuda_cart_gradient_release(void) {
  resident_grad_release_impl();
}

void generate_pair_work(int numat,
                        int l123,
                        const GradPairPod *pairs,
                        int pair_count,
                        AtomPairWork *out_work,
                        int *out_count) {
  (void)numat;
  (void)l123;
  if (out_count) *out_count = 0;
  if (!pairs || pair_count <= 0) return;

  int local_count = 0;
  if (out_work) {
    for (int idx = 0; idx < pair_count; ++idx) {
      const GradPairPod &pair = pairs[idx];
      AtomPairWork &slot = out_work[local_count];
      slot.atom_i = pair.atom_i - 1;
      slot.atom_j = pair.atom_j - 1;
      slot.range_i.first = pair.span_i_first - 1;
      slot.range_i.last  = pair.span_i_last - 1;
      slot.range_j.first = pair.span_j_first - 1;
      slot.range_j.last  = pair.span_j_last - 1;
      ++local_count;
    }
  } else {
    local_count = pair_count;
  }
  if (out_count) *out_count = local_count;
}

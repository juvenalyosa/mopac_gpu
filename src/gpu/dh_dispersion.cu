// Molecular Orbital PACkage (MOPAC)
// Copyright 2021 Virginia Polytechnic Institute and State University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// GPU evaluation of the PM6-DH / PM7 pairwise dispersion correction
// (src/corrections/H_bond_correction_PM6_DH_Dispersion.F90, PM6_DH_Disp) and
// its analytic Cartesian gradient.  The CPU code differentiates numerically
// (delta = 1e-5 A, 6 extra O(N) energy sweeps per atom); here one thread per
// atom sums the pair energies (j > i) and forces (all j) with the closed-form
// derivative.  Non-periodic systems only.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace {

// Tables C(86), R(86), N(86) of PM6_DH_Disp (index = atomic number - 1).
__constant__ double c_disp_C[86] = {
    0.16, 0.084, 0.00, 0.00, 5.79, 1.65, 1.11, 0.70,
    0.57, 0.45, 0.00, 0.00, 0.00, 0.00, 3.25, 5.79,
    5.97, 3.71, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 11.60, 4.47, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 25.80, 16.50, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
__constant__ double c_disp_R[86] = {
    156.0, 140.0, 0.0, 0.0, 180.0, 170.0, 155.0, 152.0,
    147.0, 154.0, 0.0, 0.0, 0.0, 0.0, 180.0, 180.0,
    175.0, 188.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 140.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 185.0, 202.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 198.0, 216.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
__constant__ double c_disp_N[86] = {
    0.80, 1.42, 0.00, 0.00, 2.16, 2.50, 2.82, 3.15,
    3.48, 3.81, 0.00, 0.00, 0.00, 0.00, 4.50, 4.80,
    5.10, 5.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 2.90, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 6.00, 6.30, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 6.95, 7.25, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00};

constexpr int kThreads = 128;

struct DispArgs {
  int numat;
  const int *nat;
  const int *nbonds;
  const double *coord;  // 3 x numat, Angstrom
  double alpha, s, cscale;
  int l_grad;
  double *energy;       // accumulated (already scaled by cscale)
  double *dxyz;         // 3 x numat, accumulated (kcal/mol/A)
};

__device__ __forceinline__ bool disp_param_ok(int n) {
  return n >= 1 && n <= 86 && c_disp_R[n - 1] != 0.0 && c_disp_C[n - 1] != 0.0 &&
         c_disp_N[n - 1] != 0.0;
}

__device__ __forceinline__ double disp_c6_atom(int n, int nbond) {
  if (n == 6) return (nbond == 4) ? 0.95 : 1.65;
  return c_disp_C[n - 1];
}

__global__ void dh_dispersion_kernel(DispArgs a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double e_i = 0.0;
  double g[3] = {0.0, 0.0, 0.0};
  const double third = 1.0 / 3.0;
  if (i < a.numat) {
    const int ni = a.nat[i];
    if (disp_param_ok(ni)) {
      const double xi = a.coord[3 * i], yi = a.coord[3 * i + 1], zi = a.coord[3 * i + 2];
      const double c6i = disp_c6_atom(ni, a.nbonds[i]);
      const double Ni = c_disp_N[ni - 1];
      const double Ri = c_disp_R[ni - 1];
      for (int j = 0; j < a.numat; ++j) {
        if (j == i) continue;
        const int nj = a.nat[j];
        if (!disp_param_ok(nj)) continue;
        const double dx = xi - a.coord[3 * j];
        const double dy = yi - a.coord[3 * j + 1];
        const double dz = zi - a.coord[3 * j + 2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (!(r2 < 100.0 * 100.0)) continue;  // connected(i, j, 100**2)
        const double c6j = disp_c6_atom(nj, a.nbonds[j]);
        const double Nj = c_disp_N[nj - 1];
        const double Rj = c_disp_R[nj - 1];
        const double C6 = 2.0 * pow(c6i * c6i * c6j * c6j * Ni * Nj, third) /
                          (pow(c6i * Nj * Nj, third) + pow(c6j * Ni * Ni, third));
        const double R0 = (Ri * Ri * Ri + Rj * Rj * Rj) / (Ri * Ri + Rj * Rj) / 1000.0 * 2.0;
        const double rab = sqrt(r2);
        const double rnm = rab * 0.1;
        const double ex = exp(-a.alpha * (rnm / (a.s * R0) - 1.0));
        const double damp = 1.0 / (1.0 + ex);
        const double rnm6 = rnm * rnm * rnm * rnm * rnm * rnm;
        const double e_tmp = C6 / rnm6 * damp / (1000.0 * 4.184);
        if (j > i) e_i -= e_tmp;
        if (a.l_grad) {
          // d(e_tmp)/d(rnm); E_pair = -cscale * e_tmp
          const double ddamp = damp * damp * ex * a.alpha / (a.s * R0);
          const double de = C6 / (1000.0 * 4.184) * (-6.0 * damp / (rnm6 * rnm) + ddamp / rnm6);
          const double f = -a.cscale * de * 0.1 / rab;  // dE_pair/drab * 1/rab
          g[0] += f * dx;
          g[1] += f * dy;
          g[2] += f * dz;
        }
      }
    }
    if (a.l_grad) {
      a.dxyz[3 * i] += g[0];
      a.dxyz[3 * i + 1] += g[1];
      a.dxyz[3 * i + 2] += g[2];
    }
  }
  __shared__ double red[kThreads];
  red[threadIdx.x] = e_i;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0 && red[0] != 0.0) atomicAdd(a.energy, red[0] * a.cscale);
}

template <typename T>
struct DevPtr {
  T *ptr = nullptr;
  bool upload(const T *host, size_t n) {
    if (cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)) != cudaSuccess) return false;
    return cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }
  ~DevPtr() {
    if (ptr) cudaFree(ptr);
  }
};

}  // namespace

// Fortran entry point.  Returns 0 on success (energy in *energy_out, gradient
// accumulated into dxyz when l_grad != 0), 1 on bad arguments, 2 on CUDA failure.
extern "C" int mopac_cuda_dh_dispersion(int numat, const int *nat, const int *nbonds,
                                        const double *coord, int method_pm7, int l_grad,
                                        double *energy_out, double *dxyz, double *ms_out) {
  if (numat <= 0 || !nat || !nbonds || !coord || !energy_out || (l_grad && !dxyz)) return 1;
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0, 0);
  const size_t na = static_cast<size_t>(numat);
  DevPtr<int> d_nat, d_nbonds;
  DevPtr<double> d_coord, d_dxyz, d_energy;
  const double zero = 0.0;
  bool ok = d_nat.upload(nat, na) && d_nbonds.upload(nbonds, na) && d_coord.upload(coord, 3 * na) &&
            d_energy.upload(&zero, 1) && (!l_grad || d_dxyz.upload(dxyz, 3 * na));
  int code = ok ? 0 : 2;
  if (code == 0) {
    DispArgs a;
    a.numat = numat;
    a.nat = d_nat.ptr;
    a.nbonds = d_nbonds.ptr;
    a.coord = d_coord.ptr;
    if (method_pm7) {
      a.alpha = 15.450118;
      a.s = 1.226593;
      a.cscale = 2.286419;
    } else {
      a.alpha = 20.0;
      a.s = 1.04;
      a.cscale = 0.89;
    }
    a.l_grad = l_grad;
    a.energy = d_energy.ptr;
    a.dxyz = d_dxyz.ptr;
    dh_dispersion_kernel<<<(numat + kThreads - 1) / kThreads, kThreads>>>(a);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[GPU ERROR] dh dispersion: %s\n", cudaGetErrorString(err));
      code = 2;
    } else {
      if (cudaMemcpy(energy_out, d_energy.ptr, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) code = 2;
      if (code == 0 && l_grad &&
          cudaMemcpy(dxyz, d_dxyz.ptr, 3 * na * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        code = 2;
      }
    }
  }
  cudaEventRecord(t1, 0);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (ms_out) *ms_out = static_cast<double>(ms);
  return code;
}

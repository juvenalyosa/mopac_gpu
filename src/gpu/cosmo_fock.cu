// Molecular Orbital PACkage (MOPAC)
// Copyright 2025-2026 Juvenal Yosa, PhD (juvenal.yosa@gmail.com)
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

// The dense algebra of the COSMO Fock terms (addfck, src/solvation/cosmo.F90) on the device:
//
//   phi = B^T qd,  A x = phi (A = U^T U, coscl1's packed Cholesky factor),  qsc = -fepsi x,
//   fim = B qsc
//
// B (lm61 x nps) and U (nps x nps, unpacked, diagonal restored from coscl1's inverse pivots) stay
// on the device for one cosmo_mat_gen (one cavity / one geometry); each SCF iteration moves only
// the vectors.  On the CPU these are two passes over B and one over the packed A per iteration
// (~180 ms per iteration for a 650-atom protein).

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cstdio>
#include <cstring>

#include "pinned_copy.h"

extern "C" void mozyme_section_timer_add_c(const char *name, int name_len, double ms);

namespace {

using Clock = std::chrono::steady_clock;
void add_section(const char *name, const Clock::time_point &t0) {
  mozyme_section_timer_add_c(name, static_cast<int>(std::strlen(name)),
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
}

// ap: coscl1's packed upper factor, U(i, j) (0-based, i <= j) at j(j+1)/2 + i, with 1/U(j, j) on
// the diagonal.  U: column-major nps x nps, zero below the diagonal.
__global__ void unpack_factor_kernel(const double *ap, double *u, int n) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(n) * n) return;
  const size_t i = idx % n, j = idx / n;
  double v = 0.0;
  if (i <= j) {
    v = ap[j * (j + 1) / 2 + i];
    if (i == j) v = 1.0 / v;
  }
  u[idx] = v;
}

struct State {
  long long gen = -1;
  int lm61 = 0, nps = 0;
  double *b = nullptr, *u = nullptr, *qd = nullptr, *phi = nullptr, *x = nullptr, *fim = nullptr;
  cublasHandle_t blas = nullptr;
  void release() {
    double *ptrs[] = {b, u, qd, phi, x, fim};
    for (double *q : ptrs) {
      if (q) cudaFree(q);
    }
    b = u = qd = phi = x = fim = nullptr;
    gen = -1;
    lm61 = nps = 0;
  }
};
State g;

bool ok(cudaError_t e) { return e == cudaSuccess; }
bool ok(cublasStatus_t e) { return e == CUBLAS_STATUS_SUCCESS; }

}  // namespace

// bmat: B(lm61, nps) with leading dimension lm61; amat: coscl1's packed factor (nps(nps+1)/2);
// gen: cosmo_mat_gen of both.  qd: qdenet(:, 2).  Out: phi (nps) = B^T qd, qsc (nps) = -fepsi x,
// fim (lm61) = B qsc.  Returns 1 on success, 0 when the device cannot hold B and U or a CUDA call
// failed (the caller then does it on the CPU).
extern "C" int mopac_cuda_cosmo_fock(int lm61, int nps, long long gen, const double *bmat, const double *amat,
                                     const double *qd, double fepsi, double *phi_out, double *qsc_out,
                                     double *fim_out) {
  if (lm61 <= 0 || nps <= 0 || !bmat || !amat || !qd || !phi_out || !qsc_out || !fim_out) return 0;
  const auto t0 = Clock::now();
  if (!g.blas && !ok(cublasCreate(&g.blas))) {
    g.blas = nullptr;
    return 0;
  }
  const size_t nb = static_cast<size_t>(lm61) * nps;
  const size_t nu = static_cast<size_t>(nps) * nps;
  if (g.gen != gen || g.lm61 != lm61 || g.nps != nps) {
    g.release();
    size_t free_b = 0, total_b = 0;
    const size_t npk = static_cast<size_t>(nps) * (nps + 1) / 2;
    const size_t need = (nb + nu + npk + 2 * static_cast<size_t>(nps) + 2 * static_cast<size_t>(lm61)) *
                        sizeof(double);
    if (!ok(cudaMemGetInfo(&free_b, &total_b)) || need > free_b / 2) {
      cudaGetLastError();
      return 0;
    }
    double *ap = nullptr;
    bool good = ok(cudaMalloc(reinterpret_cast<void **>(&g.b), nb * sizeof(double))) &&
                ok(cudaMalloc(reinterpret_cast<void **>(&g.u), nu * sizeof(double))) &&
                ok(cudaMalloc(reinterpret_cast<void **>(&g.qd), lm61 * sizeof(double))) &&
                ok(cudaMalloc(reinterpret_cast<void **>(&g.fim), lm61 * sizeof(double))) &&
                ok(cudaMalloc(reinterpret_cast<void **>(&g.phi), nps * sizeof(double))) &&
                ok(cudaMalloc(reinterpret_cast<void **>(&g.x), nps * sizeof(double))) &&
                ok(cudaMalloc(reinterpret_cast<void **>(&ap), npk * sizeof(double))) &&
                mopac_pinned_h2d(g.b, bmat, nb * sizeof(double)) == 1 &&
                mopac_pinned_h2d(ap, amat, npk * sizeof(double)) == 1;
    if (good) {
      unpack_factor_kernel<<<static_cast<unsigned>((nu + 255) / 256), 256>>>(ap, g.u, nps);
      good = ok(cudaGetLastError()) && ok(cudaDeviceSynchronize());
    }
    if (ap) cudaFree(ap);
    if (!good) {
      cudaGetLastError();
      g.release();
      return 0;
    }
    g.gen = gen;
    g.lm61 = lm61;
    g.nps = nps;
    add_section("cosmo_fock_gpu_setup", t0);
  }
  const auto t1 = Clock::now();
  const double one = 1.0, zero = 0.0, mf = -fepsi;
  bool good = ok(cublasSetStream(g.blas, 0)) &&
              ok(cudaMemcpy(g.qd, qd, lm61 * sizeof(double), cudaMemcpyHostToDevice)) &&
              // phi = B^T qd
              ok(cublasDgemv(g.blas, CUBLAS_OP_T, lm61, nps, &one, g.b, lm61, g.qd, 1, &zero, g.phi, 1)) &&
              ok(cudaMemcpy(g.x, g.phi, nps * sizeof(double), cudaMemcpyDeviceToDevice)) &&
              // A x = phi: U^T y = phi, then U x = y (coscl2)
              ok(cublasDtrsv(g.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, nps, g.u, nps,
                             g.x, 1)) &&
              ok(cublasDtrsv(g.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, nps, g.u, nps,
                             g.x, 1)) &&
              ok(cublasDscal(g.blas, nps, &mf, g.x, 1)) &&
              // fim = B qsc
              ok(cublasDgemv(g.blas, CUBLAS_OP_N, lm61, nps, &one, g.b, lm61, g.x, 1, &zero, g.fim, 1)) &&
              ok(cudaMemcpy(phi_out, g.phi, nps * sizeof(double), cudaMemcpyDeviceToHost)) &&
              ok(cudaMemcpy(qsc_out, g.x, nps * sizeof(double), cudaMemcpyDeviceToHost)) &&
              ok(cudaMemcpy(fim_out, g.fim, lm61 * sizeof(double), cudaMemcpyDeviceToHost));
  if (!good) {
    cudaGetLastError();
    g.release();
    return 0;
  }
  add_section("cosmo_fock_gpu", t1);
  return 1;
}

// Frees the device copies (end of a calculation or a new cavity of another size).
extern "C" void mopac_cuda_cosmo_fock_release() { g.release(); }

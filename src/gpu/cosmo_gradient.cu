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

// GPU evaluation of the COSMO gradient (src/solvation/cosmo.F90 diegrd, swig.F90 swig_diegrd):
//   1. surface charge - surface charge: 1/r (standard surface) or erf(z r)/r (SWIG), points of
//      different atoms;
//   2. SWIG only: the switching of the diagonal, dA(k,k)/dX through F(k);
//   3. surface charge - atomic multipoles (the B matrix of mkbmat / get_bvec).
// The closure term of the standard surface (isude/sude) stays on the CPU.  All O(N**2) loops run
// over every pair in both directions, so each thread accumulates the force on its own point's
// atom (or its own atom) without per-pair atomics.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace {

constexpr int kThreads = 128;

struct CosmoGradArgs {
  int numat, nps, swig;
  const double *coord;    // (3, numat)
  const double *srad;     // numat
  const int *idel;        // orbitals per atom
  const int *iden;        // 0-based offset of the atom in qdenet
  const double *ddi;      // dd * a0
  const double *qqi2;     // (a0 * qq)**2
  const double *px, *py, *pz, *q, *zeta, *fsw;  // per surface point
  const int *iatsp;       // 0-based atom of each point
  const double *qden;     // qdenet(:, 3)
  double fact, fepsi;
  double *dxyz;           // (3, numat), accumulated
};

__device__ inline double atomic_add(double *address, double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  unsigned long long *p = reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *p, assumed;
  do {
    assumed = old;
    old = atomicCAS(p, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

// 1. point-point term: thread k, all points l of other atoms (tiles in shared memory)
__global__ void cosmo_grad_points_kernel(CosmoGradArgs a) {
  __shared__ double sx[kThreads], sy[kThreads], sz[kThreads], sq[kThreads], sze[kThreads];
  __shared__ int sat[kThreads];
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = k < a.nps;
  const double xk = active ? a.px[k] : 0.0, yk = active ? a.py[k] : 0.0, zk = active ? a.pz[k] : 0.0;
  const double qk = active ? a.q[k] : 0.0;
  const double zetak = (active && a.swig) ? a.zeta[k] : 0.0;
  const int iak = active ? a.iatsp[k] : -1;
  double g0 = 0.0, g1 = 0.0, g2 = 0.0;
  for (int base = 0; base < a.nps; base += kThreads) {
    const int l = base + threadIdx.x;
    if (l < a.nps) {
      sx[threadIdx.x] = a.px[l];
      sy[threadIdx.x] = a.py[l];
      sz[threadIdx.x] = a.pz[l];
      sq[threadIdx.x] = a.q[l];
      sze[threadIdx.x] = a.swig ? a.zeta[l] : 0.0;
      sat[threadIdx.x] = a.iatsp[l];
    }
    __syncthreads();
    const int nl = min(kThreads, a.nps - base);
    if (active) {
      for (int t = 0; t < nl; ++t) {
        if (sat[t] == iak) continue;
        const double xl0 = sx[t] - xk, xl1 = sy[t] - yk, xl2 = sz[t] - zk;
        const double r2 = xl0 * xl0 + xl1 * xl1 + xl2 * xl2;
        const double r = sqrt(r2);
        double g;  // -(dA/dr)/r
        if (a.swig) {
          const double z = zetak * sze[t] / sqrt(zetak * zetak + sze[t] * sze[t]);
          if (z * r > 6.0) {
            g = 1.0 / (r2 * r);
          } else {
            const double da = (2.0 * z / sqrt(M_PI) * exp(-(z * r) * (z * r)) * r - erf(z * r)) / r2;
            g = -da / r;
          }
        } else {
          g = 1.0 / (r2 * r);
        }
        const double ff = qk * sq[t] * a.fact * g / a.fepsi;
        g0 -= xl0 * ff;
        g1 -= xl1 * ff;
        g2 -= xl2 * ff;
      }
    }
    __syncthreads();
  }
  if (active) {
    atomic_add(a.dxyz + 3 * iak, g0);
    atomic_add(a.dxyz + 3 * iak + 1, g1);
    atomic_add(a.dxyz + 3 * iak + 2, g2);
  }
}

// 2. SWIG switching term of A(k,k): thread k, all atoms j close to the point
__global__ void cosmo_grad_switch_kernel(CosmoGradArgs a) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= a.nps) return;
  const int iak = a.iatsp[k];
  const double zeta = a.zeta[k];
  const double qk = a.q[k];
  const double coef = a.fact / (2.0 * a.fepsi) * qk * qk * zeta * sqrt(2.0 / M_PI) / a.fsw[k];
  const double xk = a.px[k], yk = a.py[k], zk = a.pz[k];
  double g0 = 0.0, g1 = 0.0, g2 = 0.0;
  for (int j = 0; j < a.numat; ++j) {
    if (j == iak) continue;
    const double rj = a.srad[j];
    const double x0 = a.coord[3 * j] - xk, x1 = a.coord[3 * j + 1] - yk, x2 = a.coord[3 * j + 2] - zk;
    const double r = sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    if (r > rj + 8.0 / zeta) continue;
    const double e1 = exp(-(zeta * (rj - r)) * (zeta * (rj - r)));
    const double e2 = exp(-(zeta * (rj + r)) * (zeta * (rj + r)));
    const double t = 1.0 - 0.5 * (erf(zeta * (rj - r)) + erf(zeta * (rj + r)));
    const double tp = zeta / sqrt(M_PI) * (e1 - e2);
    const double ff = coef * tp / (t * r);
    atomic_add(a.dxyz + 3 * j, x0 * ff);
    atomic_add(a.dxyz + 3 * j + 1, x1 * ff);
    atomic_add(a.dxyz + 3 * j + 2, x2 * ff);
    g0 -= x0 * ff;
    g1 -= x1 * ff;
    g2 -= x2 * ff;
  }
  atomic_add(a.dxyz + 3 * iak, g0);
  atomic_add(a.dxyz + 3 * iak + 1, g1);
  atomic_add(a.dxyz + 3 * iak + 2, g2);
}

// 3. derivative of the potential of atom i at point k times q(k): diegrd's db terms (the
//    contribution to the point's atom; the atom itself gets minus this)
__device__ void cosmo_bterm(const CosmoGradArgs &a, int k, int i, double out[3]) {
  out[0] = out[1] = out[2] = 0.0;
  const double xx[3] = {a.px[k] - a.coord[3 * i], a.py[k] - a.coord[3 * i + 1],
                        a.pz[k] - a.coord[3 * i + 2]};
  const double dist2 = xx[0] * xx[0] + xx[1] * xx[1] + xx[2] * xx[2];
  const double ff0 = -a.q[k] * a.fact / (dist2 * sqrt(dist2));
  const int idel = a.idel[i];
  const int iden = a.iden[i];
  double db0[10], db[3][10];
  for (int j = 0; j < 10; ++j) {
    db0[j] = 0.0;
    db[0][j] = db[1][j] = db[2][j] = 0.0;
  }
  db0[0] = 1.0;
  if (idel > 1) {
    const double ddi = a.ddi[i], qqi2 = a.qqi2[i];
    const double rm2 = 1.0 / dist2, rm4 = rm2 * rm2;
    db0[1] = ddi * 3 * xx[0] * rm2;
    db0[3] = ddi * 3 * xx[1] * rm2;
    db0[6] = ddi * 3 * xx[2] * rm2;
    db0[2] = 1.0 + qqi2 * (15 * xx[0] * xx[0] * rm2 - 3.0) * rm2;
    db0[5] = 1.0 + qqi2 * (15 * xx[1] * xx[1] * rm2 - 3.0) * rm2;
    db0[9] = 1.0 + qqi2 * (15 * xx[2] * xx[2] * rm2 - 3.0) * rm2;
    db0[4] = qqi2 * 15 * xx[0] * xx[1] * rm4;
    db0[7] = qqi2 * 15 * xx[0] * xx[2] * rm4;
    db0[8] = qqi2 * 15 * xx[2] * xx[1] * rm4;
    db[0][1] = ddi;
    db[1][3] = ddi;
    db[2][6] = ddi;
    db[0][2] = 6 * qqi2 * xx[0] * rm2;
    db[1][5] = 6 * qqi2 * xx[1] * rm2;
    db[2][9] = 6 * qqi2 * xx[2] * rm2;
    db[0][4] = 0.5 * db[1][5];
    db[1][4] = 0.5 * db[0][2];
    db[0][7] = 0.5 * db[2][9];
    db[2][7] = 0.5 * db[0][2];
    db[1][8] = 0.5 * db[2][9];
    db[2][8] = 0.5 * db[1][5];
  }
  const int nterm = min(10, (idel * (idel + 1)) / 2);
  for (int j = 0; j < nterm; ++j) {
    double ff = -ff0 * a.qden[iden + j];
    if (j == 0 && idel == 9) {
      for (int iii = 5; iii <= 9; ++iii) ff -= ff0 * a.qden[iden + (iii * (iii + 1)) / 2 - 1];
    }
    for (int ix = 0; ix < 3; ++ix) out[ix] += (xx[ix] * db0[j] - db[ix][j]) * ff;
  }
}

__global__ void cosmo_grad_b_point_kernel(CosmoGradArgs a) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= a.nps) return;
  const int iak = a.iatsp[k];
  double g[3] = {0.0, 0.0, 0.0}, d[3];
  for (int i = 0; i < a.numat; ++i) {
    if (i == iak) continue;
    cosmo_bterm(a, k, i, d);
    g[0] += d[0];
    g[1] += d[1];
    g[2] += d[2];
  }
  for (int ix = 0; ix < 3; ++ix) atomic_add(a.dxyz + 3 * iak + ix, g[ix]);
}

__global__ void cosmo_grad_b_atom_kernel(CosmoGradArgs a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= a.numat) return;
  double g[3] = {0.0, 0.0, 0.0}, d[3];
  for (int k = 0; k < a.nps; ++k) {
    if (a.iatsp[k] == i) continue;
    cosmo_bterm(a, k, i, d);
    g[0] -= d[0];
    g[1] -= d[1];
    g[2] -= d[2];
  }
  for (int ix = 0; ix < 3; ++ix) atomic_add(a.dxyz + 3 * i + ix, g[ix]);
}

template <typename T>
struct DevPtr {
  T *ptr = nullptr;
  bool upload(const T *host, std::size_t n) {
    if (cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)) != cudaSuccess) return false;
    return cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }
  ~DevPtr() {
    if (ptr) cudaFree(ptr);
  }
};

}  // namespace

// Fortran entry point (cosmo.F90 diegrd).  Adds the COSMO gradient, except the closure term of
// the standard surface, to dxyz.  Returns 0 on success, 1 on bad arguments, 2 on CUDA failure
// (dxyz is then unchanged).
extern "C" int mopac_cuda_cosmo_gradient(
    int numat, int nps, int swig, int nden, const double *coord, const double *srad,
    const int *idel, const int *iden, const double *ddi, const double *qqi2,
    const double *pxyz, const double *q, const double *zeta, const double *fsw,
    const int *iatsp, const double *qden, double fact, double fepsi, double *dxyz,
    double *ms_out) {
  if (numat <= 0 || nps <= 0 || !coord || !srad || !idel || !iden || !ddi || !qqi2 || !pxyz ||
      !q || !iatsp || !qden || !dxyz || (swig && (!zeta || !fsw))) {
    return 1;
  }
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0, 0);
  const std::size_t na = static_cast<std::size_t>(numat), np = static_cast<std::size_t>(nps);
  DevPtr<double> d_coord, d_srad, d_ddi, d_qqi2, d_px, d_py, d_pz, d_q, d_zeta, d_fsw, d_qden,
      d_dxyz;
  DevPtr<int> d_idel, d_iden, d_iatsp;
  bool ok = d_coord.upload(coord, 3 * na) && d_srad.upload(srad, na) && d_ddi.upload(ddi, na) &&
            d_qqi2.upload(qqi2, na) && d_idel.upload(idel, na) && d_iden.upload(iden, na) &&
            d_px.upload(pxyz, np) && d_py.upload(pxyz + np, np) && d_pz.upload(pxyz + 2 * np, np) &&
            d_q.upload(q, np) && d_iatsp.upload(iatsp, np) &&
            d_qden.upload(qden, static_cast<std::size_t>(nden)) && d_dxyz.upload(dxyz, 3 * na);
  if (ok && swig) ok = d_zeta.upload(zeta, np) && d_fsw.upload(fsw, np);
  int code = ok ? 0 : 2;
  if (code == 0) {
    CosmoGradArgs a;
    a.numat = numat;
    a.nps = nps;
    a.swig = swig;
    a.coord = d_coord.ptr;
    a.srad = d_srad.ptr;
    a.idel = d_idel.ptr;
    a.iden = d_iden.ptr;
    a.ddi = d_ddi.ptr;
    a.qqi2 = d_qqi2.ptr;
    a.px = d_px.ptr;
    a.py = d_py.ptr;
    a.pz = d_pz.ptr;
    a.q = d_q.ptr;
    a.zeta = d_zeta.ptr;
    a.fsw = d_fsw.ptr;
    a.iatsp = d_iatsp.ptr;
    a.qden = d_qden.ptr;
    a.fact = fact;
    a.fepsi = fepsi;
    a.dxyz = d_dxyz.ptr;
    const int pblocks = (nps + kThreads - 1) / kThreads;
    cosmo_grad_points_kernel<<<pblocks, kThreads>>>(a);
    if (swig) cosmo_grad_switch_kernel<<<pblocks, kThreads>>>(a);
    cosmo_grad_b_point_kernel<<<pblocks, kThreads>>>(a);
    cosmo_grad_b_atom_kernel<<<(numat + kThreads - 1) / kThreads, kThreads>>>(a);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(dxyz, d_dxyz.ptr, 3 * na * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      cudaGetLastError();
      code = 2;
    }
  }
  cudaEventRecord(t1, 0);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);
  if (ms_out) *ms_out = ms;
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return code;
}

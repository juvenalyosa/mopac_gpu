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

// GPU MOZYME Cartesian gradient (Phase 4 of the MOZYME GPU plan).
//
// Ports dcart_build_scf_gradient_cpu (src/forces/dcart.F90) for the MOZYME,
// non-periodic (id = 0), RHF, sp-basis case:
//   * one thread per interacting atom pair (ijbo >= 0): forward/central finite
//     differences of the diatomic energy dhc() with respect to the three
//     coordinates of the second atom, exactly as the CPU code does;
//   * one thread per remaining atom pair within cutofp: the point-charge
//     derivative delsta().
// Contributions are accumulated with atomicAdd into dxyz(3, numat).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mozyme_pair_overlap.cuh"
#include "mozyme_pair_core.cuh"
#include "mozyme_pair_fock.cuh"

namespace {

constexpr int kPairThreads = 128;
constexpr int kPointThreads = 256;

struct PairGradArgs {
  int numat;
  int npairs;
  const int *pair_i;     // 1-based atom index (ii > jj)
  const int *pair_j;     // 1-based atom index
  const int *pair_off;   // ijbo(ii,jj) offset of the ii-jj block in p (0-based start)
  const int *row_start;  // 0-based CSR row start per atom ii (numat+1 entries)
  const int *diag_off;   // ijbo(ii,ii) per atom (0-based start)
  const int *iorbs;
  const int *nat;
  const double *coord;   // 3 x numat
  const double *p;       // packed MOZYME density (mpack)
  const double *tore;    // tore(107)
  MozymePairOverlapParams ovl;
  MozymePairCoreParams core;
  int distance_gate;     // 1 when ijbo() applies the cutof1/cutof2 tests (compact index route)
  double cutof1;         // squared
  double cutof2;         // squared
  double cutofp;         // Angstrom (point-charge cutoff)
  double chnge;
  double chnge2;
  double cnst;           // fpc_9
  double fpc_9;
  double ev;
  int force;             // central differences (FORCE/PRECISE)
  double *dxyz;          // 3 x numat, accumulated
  int *status;           // status[0] = failed pairs, status[1] = d-orbital pairs left to the CPU
};

__device__ __forceinline__ double coord_at(const double *coord, int atom1, int k) {
  return coord[3 * (atom1 - 1) + k];
}

// Diatomic energy for atoms (jj = atom 1, ii = atom 2) at coordinates x1/x2.
__device__ bool pair_energy(const PairGradArgs &a, int nat1, int nat2, int n1, int n2,
                            const double *x1, const double *x2, const double *pdi,
                            double *dener) {
  double smat[81];
  const double *smat_ptr = nullptr;
  if (nat1 != 102 && nat2 != 102) {
    if (!mozyme_pair_h1elec_sp_dev(nat1, nat2, x1, x2, a.ovl, smat)) return false;
    smat_ptr = smat;
  }
  double e_at2[10], e_at1[10], w[mozyme_pair::kMaxW];
  double enuc = 0.0;
  int w_count = 0;
  // rotate(ni = nat(ii), nj = nat(jj), xi = coord(ii), xj = coord(jj), w, kr, e2a, e1b, enuc)
  // in dhc(): rotate's first block (its "e1b") belongs to atom ii (= atom 2 here).
  if (!mozyme_pair_core_sp_dev(nat2, nat1, x2, x1, a.core, w, &w_count, e_at2, e_at1, &enuc)) {
    return false;
  }
  // w_count == 0 only for coincident atoms (rotate's small-rij exit): all
  // integrals are zero and the pair contributes nothing.
  const int expect = mozyme_pair::tri1(n2 + 1) * mozyme_pair::tri1(n1 + 1);
  if (w_count != 0 && w_count != expect) return false;
  *dener = mozyme_pair::diatomic_energy_sp(n1, n2, smat_ptr, e_at1, e_at2, w, enuc, pdi);
  return true;
}

__global__ void mozyme_pair_gradient_kernel(PairGradArgs a) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= a.npairs) return;
  const int ii = a.pair_i[idx];
  const int jj = a.pair_j[idx];
  const int n2 = a.iorbs[ii - 1];
  const int n1 = a.iorbs[jj - 1];
  if ((n1 != 1 && n1 != 4) || (n2 != 1 && n2 != 4)) {
    // d-orbital pair: left to the CPU (dcart_build_scf_gradient_cpu with
    // d_pairs_only); counted so the host knows the CPU pass is needed.
    atomicAdd(a.status + 1, 1);
    return;
  }
  if (a.distance_gate) {
    double r2 = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double d = coord_at(a.coord, ii, k) - coord_at(a.coord, jj, k);
      r2 += d * d;
    }
    // ijbo() returns -1/-2 (point pair) for these; handled by the point kernel.
    if (r2 > a.cutof1 || r2 > a.cutof2) return;
  }
  const int nat1 = a.nat[jj - 1];
  const int nat2 = a.nat[ii - 1];

  // Packed diatomic density, atom jj first (mirrors dcart.F90).
  double pdi[mozyme_pair::kMaxLinear];
  {
    int k = a.diag_off[jj - 1];
    int ij = 0;
    for (int i = 1; i <= n1; ++i) {
      for (int j = 1; j <= i; ++j) pdi[ij++] = a.p[k++];
    }
    ij = n1;
    k = a.pair_off[idx];
    for (int i = 1; i <= n2; ++i) {
      ++ij;
      int l = (ij * (ij - 1)) / 2;
      for (int j = 1; j <= n1; ++j) pdi[l++] = a.p[k++];
    }
    k = a.diag_off[ii - 1];
    ij = n1;
    for (int i = 1; i <= n2; ++i) {
      ++ij;
      int l = (ij * (ij - 1)) / 2 + n1;
      for (int j = 1; j <= i; ++j) pdi[l++] = a.p[k++];
    }
  }

  double x1[3], x2[3];
  for (int k = 0; k < 3; ++k) {
    x1[k] = coord_at(a.coord, jj, k);
    x2[k] = coord_at(a.coord, ii, k);
  }
  double aa = 0.0, ee = 0.0;
  bool ok = true;
  if (!a.force) {
    x1[0] += a.chnge2;
    x1[1] += a.chnge2;
    x1[2] += a.chnge2;
    ok = pair_energy(a, nat1, nat2, n1, n2, x1, x2, pdi, &aa);
  }
  for (int k = 0; k < 3 && ok; ++k) {
    const double x0 = x2[k];
    if (a.force) {
      x2[k] = x0 - a.chnge2;
      ok = pair_energy(a, nat1, nat2, n1, n2, x1, x2, pdi, &aa);
      if (!ok) break;
      x2[k] = x0 + a.chnge2;
    } else {
      x2[k] = x0 + a.chnge;
    }
    ok = pair_energy(a, nat1, nat2, n1, n2, x1, x2, pdi, &ee);
    x2[k] = x0;
    if (!ok) break;
    const double deriv = (aa - ee) * a.cnst / a.chnge;
    atomicAdd(&a.dxyz[3 * (ii - 1) + k], -deriv);
    atomicAdd(&a.dxyz[3 * (jj - 1) + k], deriv);
  }
  if (!ok) atomicAdd(a.status, 1);
}

// Is (ii, jj) an interacting pair (present in the CSR list and, on the
// compact-index route, inside the cutoffs)?
__device__ __forceinline__ bool is_block_pair(const PairGradArgs &a, int ii, int jj, double r2) {
  if (a.distance_gate && (r2 > a.cutof1 || r2 > a.cutof2)) return false;
  int lo = a.row_start[ii - 1];
  int hi = a.row_start[ii] - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    const int v = a.pair_j[mid];
    if (v == jj) return true;
    if (v < jj) lo = mid + 1; else hi = mid - 1;
  }
  return false;
}

// delsta(): point-charge derivative for non-interacting pairs within cutofp.
__global__ void mozyme_point_gradient_kernel(PairGradArgs a, const double *qatom) {
  const int ii = blockIdx.y + 2;  // 2..numat
  const int jj = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (ii > a.numat || jj >= ii) return;
  double d[3];
  double r2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    d[k] = coord_at(a.coord, jj, k) - coord_at(a.coord, ii, k);
    r2 += d[k] * d[k];
  }
  if (is_block_pair(a, ii, jj, r2)) return;
  const double rij = sqrt(r2);
  if (rij > a.cutofp) return;
  const double sum = a.fpc_9 * a.ev / (rij * rij);
  const double qii = qatom[ii - 1];
  const double qjj = qatom[jj - 1];
  for (int k = 0; k < 3; ++k) {
    const double vect = d[k] / rij;
    const double dstat = -0.5 * qjj * qii * sum * vect;
    atomicAdd(&a.dxyz[3 * (ii - 1) + k], -dstat);
    atomicAdd(&a.dxyz[3 * (jj - 1) + k], dstat);
  }
}

// qatom(i) = tore(nat(i)) - sum of the diagonal density of atom i (delsta()).
__global__ void mozyme_atom_charge_kernel(PairGradArgs a, double *qatom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= a.numat) return;
  double q = a.tore[a.nat[i] - 1];
  int l = a.diag_off[i] - 1;  // 0-based index of the element before the block
  for (int k = 1; k <= a.iorbs[i]; ++k) {
    l += k;
    q -= a.p[l];
  }
  qatom[i] = q;
}

template <typename T>
struct DeviceArray {
  T *ptr = nullptr;
  size_t count = 0;
  bool upload(const T *host, size_t n) {
    if (n == 0) return true;
    if (cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)) != cudaSuccess) return false;
    count = n;
    return cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }
  bool alloc(size_t n) {
    if (n == 0) return true;
    if (cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)) != cudaSuccess) return false;
    count = n;
    return true;
  }
  ~DeviceArray() {
    if (ptr) cudaFree(ptr);
  }
};

bool grad_verbose() {
  const char *v = std::getenv("MOPAC_GPU_VERBOSE");
  return v && *v && *v != '0';
}

}  // namespace

// Fortran entry point.  All arrays are host arrays; parameter tables are the
// parameters_C arrays (1-based by atomic number, column-major for 2-D:
// npq(107,3), po(9,107), ddp(6,107), guess1/2/3(107,4), alpb/xfac(100,100),
// v_par(60)).  method_flags = {pm7, pm6, pm8, pm6_org, am1, mndod, l_feather}.
// dxyz (3 x numat) is accumulated in place.
// Returns 0 on success, 1 on bad arguments, 2 on CUDA failure, 3 when some
// pair is unsupported (caller must recompute on the CPU).
extern "C" int mopac_cuda_mozyme_pair_gradient(
    int numat, int mpack, int npairs, const int *pair_i, const int *pair_j,
    const int *pair_off, const int *row_start, const int *diag_off,
    const int *iorbs, const int *nat, const double *coord, const double *p,
    int distance_gate, double cutof1, double cutof2, double cutofp,
    double cutofs, double chnge, double cnst, double fpc_9, double ev,
    double a0, double trunc_1, double trunc_2, int force,
    const int *method_flags,
    // overlap parameters
    const int *natorb, const int *npq, const double *zs, const double *zp,
    const double *zd, const double *betas, const double *betap,
    const double *betad,
    // core parameters
    const int *iod, const double *tore, const double *alp, const double *am,
    const double *ad, const double *aq, const double *dd, const double *qq,
    const double *po, const double *ddp, const double *guess1,
    const double *guess2, const double *guess3, const double *alpb,
    const double *xfac, const double *v_par,
    double *dxyz, double *ms_out, int *d_pairs_out) {
  if (numat <= 0 || mpack <= 0 || !iorbs || !nat || !coord || !p || !dxyz ||
      !natorb || !npq || !zs || !zp || !zd || !betas || !betap || !betad ||
      !iod || !tore || !alp || !am || !ad || !aq || !dd || !qq || !po || !ddp ||
      !guess1 || !guess2 || !guess3 || !alpb || !xfac || !v_par ||
      !method_flags || !row_start || !diag_off) {
    return 1;
  }
  if (npairs > 0 && (!pair_i || !pair_j || !pair_off)) return 1;

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0, 0);

  DeviceArray<int> d_pair_i, d_pair_j, d_pair_off, d_row_start, d_diag_off, d_iorbs, d_nat;
  DeviceArray<int> d_natorb, d_npq, d_iod, d_status;
  DeviceArray<double> d_coord, d_p, d_zs, d_zp, d_zd, d_betas, d_betap, d_betad, d_tore;
  DeviceArray<double> d_alp, d_am, d_ad, d_aq, d_dd, d_qq, d_po, d_ddp, d_guess1, d_guess2;
  DeviceArray<double> d_guess3, d_alpb, d_xfac, d_v_par;
  DeviceArray<double> d_dxyz, d_q;
  const size_t na = static_cast<size_t>(numat);
  const size_t np = static_cast<size_t>(npairs);
  bool ok = d_pair_i.upload(pair_i, np) && d_pair_j.upload(pair_j, np) &&
            d_pair_off.upload(pair_off, np) && d_row_start.upload(row_start, na + 1) &&
            d_diag_off.upload(diag_off, na) && d_iorbs.upload(iorbs, na) &&
            d_nat.upload(nat, na) && d_natorb.upload(natorb, 107) &&
            d_npq.upload(npq, 107 * 3) && d_iod.upload(iod, 107) &&
            d_coord.upload(coord, 3 * na) &&
            d_p.upload(p, static_cast<size_t>(mpack)) && d_zs.upload(zs, 107) &&
            d_zp.upload(zp, 107) && d_zd.upload(zd, 107) && d_betas.upload(betas, 107) &&
            d_betap.upload(betap, 107) && d_betad.upload(betad, 107) &&
            d_tore.upload(tore, 107) && d_alp.upload(alp, 107) && d_am.upload(am, 107) &&
            d_ad.upload(ad, 107) && d_aq.upload(aq, 107) && d_dd.upload(dd, 107) &&
            d_qq.upload(qq, 107) && d_po.upload(po, 9 * 107) && d_ddp.upload(ddp, 6 * 107) &&
            d_guess1.upload(guess1, 107 * 4) && d_guess2.upload(guess2, 107 * 4) &&
            d_guess3.upload(guess3, 107 * 4) && d_alpb.upload(alpb, 100 * 100) &&
            d_xfac.upload(xfac, 100 * 100) && d_v_par.upload(v_par, 60) &&
            d_dxyz.upload(dxyz, 3 * na) && d_q.alloc(na) && d_status.alloc(2);
  if (!ok) return 2;
  if (cudaMemset(d_status.ptr, 0, 2 * sizeof(int)) != cudaSuccess) return 2;
  if (d_pairs_out) *d_pairs_out = 0;

  PairGradArgs a;
  std::memset(&a, 0, sizeof(a));
  a.core.natorb = d_natorb.ptr;
  a.core.iod = d_iod.ptr;
  a.core.tore = d_tore.ptr;
  a.core.alp = d_alp.ptr;
  a.core.am = d_am.ptr;
  a.core.ad = d_ad.ptr;
  a.core.aq = d_aq.ptr;
  a.core.dd = d_dd.ptr;
  a.core.qq = d_qq.ptr;
  a.core.po = d_po.ptr;
  a.core.ddp = d_ddp.ptr;
  a.core.guess1 = d_guess1.ptr;
  a.core.guess2 = d_guess2.ptr;
  a.core.guess3 = d_guess3.ptr;
  a.core.alpb = d_alpb.ptr;
  a.core.xfac = d_xfac.ptr;
  a.core.v_par = d_v_par.ptr;
  a.core.a0 = a0;
  a.core.ev = ev;
  a.core.trunc_1 = trunc_1;
  a.core.trunc_2 = trunc_2;
  a.core.method_pm7 = method_flags[0];
  a.core.method_pm6 = method_flags[1];
  a.core.method_pm8 = method_flags[2];
  a.core.method_pm6_org = method_flags[3];
  a.core.method_am1 = method_flags[4];
  a.core.method_mndod = method_flags[5];
  a.core.l_feather = method_flags[6];

  a.numat = numat;
  a.npairs = npairs;
  a.pair_i = d_pair_i.ptr;
  a.pair_j = d_pair_j.ptr;
  a.pair_off = d_pair_off.ptr;
  a.row_start = d_row_start.ptr;
  a.diag_off = d_diag_off.ptr;
  a.iorbs = d_iorbs.ptr;
  a.nat = d_nat.ptr;
  a.coord = d_coord.ptr;
  a.p = d_p.ptr;
  a.tore = d_tore.ptr;
  a.ovl.natorb = d_natorb.ptr;
  a.ovl.npq = d_npq.ptr;
  a.ovl.zs = d_zs.ptr;
  a.ovl.zp = d_zp.ptr;
  a.ovl.zd = d_zd.ptr;
  a.ovl.betas = d_betas.ptr;
  a.ovl.betap = d_betap.ptr;
  a.ovl.betad = d_betad.ptr;
  a.ovl.a0 = a0;
  a.ovl.cutofs = cutofs;
  a.ovl.cutof1 = cutof1;
  a.distance_gate = distance_gate;
  a.cutof1 = cutof1;
  a.cutof2 = cutof2;
  a.cutofp = cutofp;
  a.chnge = chnge;
  a.chnge2 = 0.5 * chnge;
  a.cnst = cnst;
  a.fpc_9 = fpc_9;
  a.ev = ev;
  a.force = force;
  a.dxyz = d_dxyz.ptr;
  a.status = d_status.ptr;

  mozyme_atom_charge_kernel<<<(numat + 255) / 256, 256>>>(a, d_q.ptr);
  if (npairs > 0) {
    mozyme_pair_gradient_kernel<<<(npairs + kPairThreads - 1) / kPairThreads, kPairThreads>>>(a);
  }
  if (numat >= 2) {
    dim3 grid((numat + kPointThreads - 1) / kPointThreads, numat - 1);
    mozyme_point_gradient_kernel<<<grid, kPointThreads>>>(a, d_q.ptr);
  }
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "[GPU ERROR] mozyme pair gradient: %s\n", cudaGetErrorString(status));
    return 2;
  }
  int counts[2] = {0, 0};
  ok = cudaMemcpy(counts, d_status.ptr, 2 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
  const int bad = counts[0];
  if (ok && d_pairs_out) *d_pairs_out = counts[1];
  if (ok && bad == 0) {
    ok = cudaMemcpy(dxyz, d_dxyz.ptr, 3 * na * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  cudaEventRecord(t1, 0);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (ms_out) *ms_out = static_cast<double>(ms);
  if (!ok) return 2;
  if (bad != 0) {
    if (grad_verbose()) {
      std::fprintf(stderr, "[MOZYME GPU gradient] %d pairs failed on device, falling back to CPU\n", bad);
    }
    return 3;
  }
  return 0;
}

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

// GPU MOZYME per-pair work (Phase 4/5 of the MOZYME GPU plan):
//
//   mopac_cuda_mozyme_pair_gradient — dcart_build_scf_gradient_cpu
//     (src/forces/dcart.F90) for the MOZYME, non-periodic, RHF, sp-basis case:
//     one thread per interacting atom pair (ijbo >= 0) doing the finite
//     differences of the diatomic energy dhc(); one thread per remaining pair
//     within cutofp doing the point-charge derivative delsta().
//
//   mopac_cuda_mozyme_hcore_pairs — the block-pair part of hcore_for_MOZYME
//     (src/MOZYME/hcore_for_MOZYME.F90): h1elec() into the off-diagonal h
//     block, rotate()'s e1b/e2a into the diagonal blocks and enuc summed.
//
// Contributions are accumulated with atomicAdd.  sp-sp pairs and pairs with a
// d-orbital atom run in separate kernels (the d one with larger local arrays);
// sparkles (or d pairs when d_on_device == 0) are counted and left to the CPU.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mozyme_pair_overlap.cuh"
#include "mozyme_pair_core.cuh"
#include "mozyme_pair_fock.cuh"

// Mirror of the Fortran bind(C) type mozyme_pair_tables_c (mozyme_gpu_gradient.F90).
struct MozymePairTablesC {
  const int *natorb;
  const int *npq;
  const int *iod;
  const double *zs;
  const double *zp;
  const double *zd;
  const double *betas;
  const double *betap;
  const double *betad;
  const double *tore;
  const double *alp;
  const double *am;
  const double *ad;
  const double *aq;
  const double *dd;
  const double *qq;
  const double *po;
  const double *ddp;
  const double *guess1;
  const double *guess2;
  const double *guess3;
  const double *alpb;
  const double *xfac;
  const double *v_par;
  double a0;
  double ev;
  double cutofs;   // squared overlap cutoff (h1elec)
  double cutof1;   // squared
  double trunc_1;
  double trunc_2;
  int method_flags[7];  // pm7, pm6, pm8, pm6_org, am1, mndod, l_feather
};

namespace {

constexpr int kPairThreads = 128;
constexpr int kPointThreads = 256;
// The d-pair kernels keep ~60 KB of local storage per thread (W up to 2025
// entries, rotation scratch).  Small blocks plus a dummy dynamic shared
// allocation cap the number of resident threads per SM, which bounds the
// local-memory reservation the driver makes at launch.
constexpr int kPairThreadsD = 32;
constexpr size_t kPairSharedD = 40 * 1024;

template <typename T>
struct DeviceArray {
  T *ptr = nullptr;
  size_t count = 0;
  bool upload(const T *host, size_t n) {
    if (n == 0) return true;
    if (!host) return false;
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

// Device copies of the parameter tables plus the filled parameter blocks.
struct PairTables {
  DeviceArray<int> natorb, npq, iod;
  DeviceArray<double> zs, zp, zd, betas, betap, betad, tore, alp, am, ad, aq, dd, qq;
  DeviceArray<double> po, ddp, guess1, guess2, guess3, alpb, xfac, v_par;
  MozymePairOverlapParams ovl;
  MozymePairCoreParams core;

  bool upload(const MozymePairTablesC &t) {
    std::memset(&ovl, 0, sizeof(ovl));
    std::memset(&core, 0, sizeof(core));
    const bool ok = natorb.upload(t.natorb, 107) && npq.upload(t.npq, 107 * 3) &&
                    iod.upload(t.iod, 107) && zs.upload(t.zs, 107) && zp.upload(t.zp, 107) &&
                    zd.upload(t.zd, 107) && betas.upload(t.betas, 107) &&
                    betap.upload(t.betap, 107) && betad.upload(t.betad, 107) &&
                    tore.upload(t.tore, 107) && alp.upload(t.alp, 107) && am.upload(t.am, 107) &&
                    ad.upload(t.ad, 107) && aq.upload(t.aq, 107) && dd.upload(t.dd, 107) &&
                    qq.upload(t.qq, 107) && po.upload(t.po, 9 * 107) && ddp.upload(t.ddp, 6 * 107) &&
                    guess1.upload(t.guess1, 107 * 4) && guess2.upload(t.guess2, 107 * 4) &&
                    guess3.upload(t.guess3, 107 * 4) && alpb.upload(t.alpb, 100 * 100) &&
                    xfac.upload(t.xfac, 100 * 100) && v_par.upload(t.v_par, 60);
    if (!ok) return false;
    ovl.natorb = natorb.ptr;
    ovl.npq = npq.ptr;
    ovl.zs = zs.ptr;
    ovl.zp = zp.ptr;
    ovl.zd = zd.ptr;
    ovl.betas = betas.ptr;
    ovl.betap = betap.ptr;
    ovl.betad = betad.ptr;
    ovl.a0 = t.a0;
    ovl.cutofs = t.cutofs;
    ovl.cutof1 = t.cutof1;
    core.natorb = natorb.ptr;
    core.iod = iod.ptr;
    core.tore = tore.ptr;
    core.alp = alp.ptr;
    core.am = am.ptr;
    core.ad = ad.ptr;
    core.aq = aq.ptr;
    core.dd = dd.ptr;
    core.qq = qq.ptr;
    core.po = po.ptr;
    core.ddp = ddp.ptr;
    core.guess1 = guess1.ptr;
    core.guess2 = guess2.ptr;
    core.guess3 = guess3.ptr;
    core.alpb = alpb.ptr;
    core.xfac = xfac.ptr;
    core.v_par = v_par.ptr;
    core.a0 = t.a0;
    core.ev = t.ev;
    core.trunc_1 = t.trunc_1;
    core.trunc_2 = t.trunc_2;
    core.method_pm7 = t.method_flags[0];
    core.method_pm6 = t.method_flags[1];
    core.method_pm8 = t.method_flags[2];
    core.method_pm6_org = t.method_flags[3];
    core.method_am1 = t.method_flags[4];
    core.method_mndod = t.method_flags[5];
    core.l_feather = t.method_flags[6];
    return true;
  }
};

// Geometry / block index shared by both kernels.
struct PairGeom {
  int numat;
  int npairs;
  const int *pair_i;     // 1-based atom index (ii > jj)
  const int *pair_j;     // 1-based atom index
  const int *pair_off;   // ijbo(ii,jj): 0-based start of the ii-jj block
  const int *row_start;  // 0-based CSR row start per atom ii (numat+1 entries)
  const int *diag_off;   // ijbo(ii,ii): 0-based start of the diagonal block
  const int *iorbs;
  const int *nat;
  const double *coord;   // 3 x numat
  int distance_gate;     // 1 when ijbo() applies the cutof1/cutof2 tests (compact index route)
  int d_on_device;       // 1: pairs with a d-orbital atom are evaluated by the d kernels
  double cutof1;         // squared
  double cutof2;         // squared
  int *status;           // status[0] = failed pairs, status[1] = pairs left to the CPU
};

struct PairGradArgs {
  PairGeom g;
  const double *p;       // packed MOZYME density (mpack)
  const double *tore;
  MozymePairOverlapParams ovl;
  MozymePairCoreParams core;
  double cutofp;         // Angstrom (point-charge cutoff)
  double chnge;
  double chnge2;
  double cnst;           // fpc_9
  double fpc_9;
  double ev;
  int force;             // central differences (FORCE/PRECISE)
  double *dxyz;          // 3 x numat, accumulated
};

struct HcoreArgs {
  PairGeom g;
  MozymePairOverlapParams ovl;
  MozymePairCoreParams core;
  double *h;             // packed MOZYME one-electron matrix (mpack), accumulated
  double *enuc;          // accumulated core-core repulsion
};

__device__ __forceinline__ double coord_at(const double *coord, int atom1, int k) {
  return coord[3 * (atom1 - 1) + k];
}

__device__ __forceinline__ bool sp_orbital_count(int n) { return n == 1 || n == 4; }
__device__ __forceinline__ bool spd_orbital_count(int n) { return n == 1 || n == 4 || n == 9; }

// Classifies pair idx: 0 = not a device pair (sparkle, d pair with the d path
// off, or outside the compact-route cutoffs), 1 = sp-sp, 2 = involves a d atom
// (both atoms in {1,4,9}).  Pairs left to the CPU are counted in status[1]
// when `count_cpu` is set (only one kernel per launch must count).
__device__ __forceinline__ int pair_class(const PairGeom &g, int idx, int *ii_out, int *jj_out,
                                          bool count_cpu) {
  const int ii = g.pair_i[idx];
  const int jj = g.pair_j[idx];
  const int ni = g.iorbs[ii - 1];
  const int nj = g.iorbs[jj - 1];
  const bool has_d = (ni == 9 || nj == 9);
  if (!spd_orbital_count(ni) || !spd_orbital_count(nj) || (has_d && !g.d_on_device)) {
    if (count_cpu) atomicAdd(g.status + 1, 1);
    return 0;
  }
  if (g.distance_gate) {
    double r2 = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double d = coord_at(g.coord, ii, k) - coord_at(g.coord, jj, k);
      r2 += d * d;
    }
    // ijbo() returns -1/-2 (point pair) for these.
    if (r2 > g.cutof1 || r2 > g.cutof2) return 0;
  }
  *ii_out = ii;
  *jj_out = jj;
  return has_d ? 2 : 1;
}

// Diatomic energy for atoms (jj = atom 1, ii = atom 2) at coordinates x1/x2.
// D = false: sp-sp pair (small local arrays); D = true: pair with a d atom.
template <bool D>
__device__ bool pair_energy(const PairGradArgs &a, int nat1, int nat2, int n1, int n2,
                            const double *x1, const double *x2, const double *pdi,
                            double *dener) {
  constexpr int kL = D ? mozyme_pair::kMaxLinearD : mozyme_pair::kMaxLinear;
  constexpr int kW = D ? mozyme_pair::kMaxWD : mozyme_pair::kMaxW;
  double smat[81];
  const double *smat_ptr = nullptr;
  if (nat1 != 102 && nat2 != 102) {
    const bool ok = D ? mozyme_pair_h1elec_dev(nat1, nat2, x1, x2, a.ovl, smat)
                      : mozyme_pair_h1elec_sp_dev(nat1, nat2, x1, x2, a.ovl, smat);
    if (!ok) return false;
    smat_ptr = smat;
  }
  double e_at2[45], e_at1[45], w[kW];
  double enuc = 0.0;
  int w_count = 0;
  // dhc(): rotate(ni = nat(ii), nj = nat(jj), xi = coord(ii), xj = coord(jj), w, kr, e2a, e1b, enuc)
  // i.e. rotate's first block belongs to atom ii (= atom 2 here).
  const bool ok = D ? mozyme_pair_core_dev(nat2, nat1, x2, x1, a.core, w, &w_count, e_at2, e_at1, &enuc)
                    : mozyme_pair_core_sp_dev(nat2, nat1, x2, x1, a.core, w, &w_count, e_at2, e_at1, &enuc);
  if (!ok) return false;
  // w_count == 0 only for coincident atoms (rotate's small-rij exit): all
  // integrals are zero and the pair contributes nothing.
  const int expect = mozyme_pair::tri1(n2 + 1) * mozyme_pair::tri1(n1 + 1);
  if (w_count != 0 && w_count != expect) return false;
  double h[kL], f[kL], pa[kL];
  *dener = mozyme_pair::diatomic_energy_generic(n1, n2, smat_ptr, e_at1, e_at2, w, enuc, pdi, h, f, pa);
  return true;
}

template <bool D>
__global__ void mozyme_pair_gradient_kernel(PairGradArgs a) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= a.g.npairs) return;
  int ii, jj;
  if (pair_class(a.g, idx, &ii, &jj, !D) != (D ? 2 : 1)) return;
  const int n2 = a.g.iorbs[ii - 1];
  const int n1 = a.g.iorbs[jj - 1];
  const int nat1 = a.g.nat[jj - 1];
  const int nat2 = a.g.nat[ii - 1];

  // Packed diatomic density, atom jj first (mirrors dcart.F90).
  double pdi[D ? mozyme_pair::kMaxLinearD : mozyme_pair::kMaxLinear];
  {
    int k = a.g.diag_off[jj - 1];
    int ij = 0;
    for (int i = 1; i <= n1; ++i) {
      for (int j = 1; j <= i; ++j) pdi[ij++] = a.p[k++];
    }
    ij = n1;
    k = a.g.pair_off[idx];
    for (int i = 1; i <= n2; ++i) {
      ++ij;
      int l = (ij * (ij - 1)) / 2;
      for (int j = 1; j <= n1; ++j) pdi[l++] = a.p[k++];
    }
    k = a.g.diag_off[ii - 1];
    ij = n1;
    for (int i = 1; i <= n2; ++i) {
      ++ij;
      int l = (ij * (ij - 1)) / 2 + n1;
      for (int j = 1; j <= i; ++j) pdi[l++] = a.p[k++];
    }
  }

  double x1[3], x2[3];
  for (int k = 0; k < 3; ++k) {
    x1[k] = coord_at(a.g.coord, jj, k);
    x2[k] = coord_at(a.g.coord, ii, k);
  }
  double aa = 0.0, ee = 0.0;
  bool ok = true;
  if (!a.force) {
    x1[0] += a.chnge2;
    x1[1] += a.chnge2;
    x1[2] += a.chnge2;
    ok = pair_energy<D>(a, nat1, nat2, n1, n2, x1, x2, pdi, &aa);
  }
  for (int k = 0; k < 3 && ok; ++k) {
    const double x0 = x2[k];
    if (a.force) {
      x2[k] = x0 - a.chnge2;
      ok = pair_energy<D>(a, nat1, nat2, n1, n2, x1, x2, pdi, &aa);
      if (!ok) break;
      x2[k] = x0 + a.chnge2;
    } else {
      x2[k] = x0 + a.chnge;
    }
    ok = pair_energy<D>(a, nat1, nat2, n1, n2, x1, x2, pdi, &ee);
    x2[k] = x0;
    if (!ok) break;
    const double deriv = (aa - ee) * a.cnst / a.chnge;
    atomicAdd(&a.dxyz[3 * (ii - 1) + k], -deriv);
    atomicAdd(&a.dxyz[3 * (jj - 1) + k], deriv);
  }
  if (!ok) atomicAdd(a.g.status, 1);
}

// Is (ii, jj) an interacting pair (present in the CSR list and, on the
// compact-index route, inside the cutoffs)?
__device__ __forceinline__ bool is_block_pair(const PairGeom &g, int ii, int jj, double r2) {
  if (g.distance_gate && (r2 > g.cutof1 || r2 > g.cutof2)) return false;
  int lo = g.row_start[ii - 1];
  int hi = g.row_start[ii] - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    const int v = g.pair_j[mid];
    if (v == jj) return true;
    if (v < jj) lo = mid + 1; else hi = mid - 1;
  }
  return false;
}

// delsta(): point-charge derivative for non-interacting pairs within cutofp.
__global__ void mozyme_point_gradient_kernel(PairGradArgs a, const double *qatom) {
  const int ii = blockIdx.y + 2;  // 2..numat
  const int jj = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (ii > a.g.numat || jj >= ii) return;
  double d[3];
  double r2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    d[k] = coord_at(a.g.coord, jj, k) - coord_at(a.g.coord, ii, k);
    r2 += d[k] * d[k];
  }
  if (is_block_pair(a.g, ii, jj, r2)) return;
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
  if (i >= a.g.numat) return;
  double q = a.tore[a.g.nat[i] - 1];
  int l = a.g.diag_off[i] - 1;  // 0-based index of the element before the block
  for (int k = 1; k <= a.g.iorbs[i]; ++k) {
    l += k;
    q -= a.p[l];
  }
  qatom[i] = q;
}

// hcore_for_MOZYME block pair: h(i,j) += h1elec, h(i,i) += e1b, h(j,j) += e2a,
// enuclr += enuc, for atom i = ii (first in the pair loop) and j = jj < ii.
template <bool D>
__global__ void mozyme_hcore_pairs_kernel(HcoreArgs a) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  bool ok = true;
  double enuc = 0.0;
  int ii = 0, jj = 0;
  const bool active =
      idx < a.g.npairs && pair_class(a.g, idx, &ii, &jj, !D) == (D ? 2 : 1);
  if (active) {
    const int ni = a.g.nat[ii - 1];
    const int nj = a.g.nat[jj - 1];
    const int norb_i = a.g.iorbs[ii - 1];
    const int norb_j = a.g.iorbs[jj - 1];
    const double *xi = a.g.coord + 3 * (ii - 1);
    const double *xj = a.g.coord + 3 * (jj - 1);
    double smat[81];
    ok = D ? mozyme_pair_h1elec_dev(ni, nj, xi, xj, a.ovl, smat)
           : mozyme_pair_h1elec_sp_dev(ni, nj, xi, xj, a.ovl, smat);
    if (ok) {
      double *hij = a.h + a.g.pair_off[idx];
      for (int i1 = 1; i1 <= norb_i; ++i1) {
        for (int j1 = 1; j1 <= norb_j; ++j1) {
          // Off-diagonal block: this thread is the only writer.
          hij[(i1 - 1) * norb_j + (j1 - 1)] += smat[(i1 - 1) + 9 * (j1 - 1)];
        }
      }
      double e1b[45], e2a[45], w[D ? mozyme_pair::kMaxWD : mozyme_pair::kMaxW];
      int w_count = 0;
      ok = D ? mozyme_pair_core_dev(ni, nj, xi, xj, a.core, w, &w_count, e1b, e2a, &enuc)
             : mozyme_pair_core_sp_dev(ni, nj, xi, xj, a.core, w, &w_count, e1b, e2a, &enuc);
      if (ok) {
        const int ti = (norb_i * (norb_i + 1)) / 2;
        const int tj = (norb_j * (norb_j + 1)) / 2;
        double *hii = a.h + a.g.diag_off[ii - 1];
        double *hjj = a.h + a.g.diag_off[jj - 1];
        for (int k = 0; k < ti; ++k) atomicAdd(&hii[k], e1b[k]);
        for (int k = 0; k < tj; ++k) atomicAdd(&hjj[k], e2a[k]);
      }
    }
    if (!ok) {
      atomicAdd(a.g.status, 1);
      enuc = 0.0;
    }
  }
  // Block reduction of enuc.
  __shared__ double red[kPairThreads];
  red[threadIdx.x] = enuc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0 && red[0] != 0.0) atomicAdd(a.enuc, red[0]);
}

bool grad_verbose() {
  const char *v = std::getenv("MOPAC_GPU_VERBOSE");
  return v && *v && *v != '0';
}

struct GeomBuffers {
  DeviceArray<int> pair_i, pair_j, pair_off, row_start, diag_off, iorbs, nat, status;
  DeviceArray<double> coord;
  bool upload(int numat, int npairs, const int *pi, const int *pj, const int *po,
              const int *rs, const int *dof, const int *io, const int *na,
              const double *co, PairGeom &g) {
    const size_t n = static_cast<size_t>(numat);
    const size_t np = static_cast<size_t>(npairs);
    const bool ok = pair_i.upload(pi, np) && pair_j.upload(pj, np) && pair_off.upload(po, np) &&
                    row_start.upload(rs, n + 1) && diag_off.upload(dof, n) &&
                    iorbs.upload(io, n) && nat.upload(na, n) && coord.upload(co, 3 * n) &&
                    status.alloc(2) && cudaMemset(status.ptr, 0, 2 * sizeof(int)) == cudaSuccess;
    if (!ok) return false;
    g.numat = numat;
    g.npairs = npairs;
    g.pair_i = pair_i.ptr;
    g.pair_j = pair_j.ptr;
    g.pair_off = pair_off.ptr;
    g.row_start = row_start.ptr;
    g.diag_off = diag_off.ptr;
    g.iorbs = iorbs.ptr;
    g.nat = nat.ptr;
    g.coord = coord.ptr;
    g.status = status.ptr;
    return true;
  }
};

struct EventTimer {
  cudaEvent_t t0, t1;
  EventTimer() {
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);
  }
  double stop_ms() {
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    return static_cast<double>(ms);
  }
  ~EventTimer() {
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
  }
};

// Sync, read the status counters; returns the return code (0 ok, 2 CUDA
// failure, 3 some pair failed on the device).
int finish_launch(const char *label, const DeviceArray<int> &status, int *d_pairs_out) {
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[GPU ERROR] %s: %s\n", label, cudaGetErrorString(err));
    return 2;
  }
  int counts[2] = {0, 0};
  if (cudaMemcpy(counts, status.ptr, 2 * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
  if (d_pairs_out) *d_pairs_out = counts[1];
  if (counts[0] != 0) {
    if (grad_verbose()) {
      std::fprintf(stderr, "[%s] %d pairs failed on device, falling back to CPU\n", label, counts[0]);
    }
    return 3;
  }
  return 0;
}

}  // namespace

// Fortran entry point for the MOZYME gradient.  All arrays are host arrays;
// dxyz (3 x numat) is accumulated in place.  Returns 0 on success, 1 on bad
// arguments, 2 on CUDA failure, 3 when some pair failed on the device
// (caller must recompute on the CPU).  d_pairs_out receives the number of
// pairs left to the CPU (d orbitals / sparkles).
extern "C" int mopac_cuda_mozyme_pair_gradient(
    int numat, int mpack, int npairs, const int *pair_i, const int *pair_j,
    const int *pair_off, const int *row_start, const int *diag_off,
    const int *iorbs, const int *nat, const double *coord, const double *p,
    int distance_gate, int d_on_device, double cutof2, double cutofp, double chnge,
    double cnst, double fpc_9, int force, const MozymePairTablesC *tables,
    double *dxyz, double *ms_out, int *d_pairs_out) {
  if (numat <= 0 || mpack <= 0 || !iorbs || !nat || !coord || !p || !dxyz || !tables ||
      !row_start || !diag_off) {
    return 1;
  }
  if (npairs > 0 && (!pair_i || !pair_j || !pair_off)) return 1;
  if (d_pairs_out) *d_pairs_out = 0;

  EventTimer timer;
  GeomBuffers geom;
  PairTables tab;
  DeviceArray<double> d_p, d_dxyz, d_q;
  PairGradArgs a;
  std::memset(&a, 0, sizeof(a));
  const size_t na = static_cast<size_t>(numat);
  if (!geom.upload(numat, npairs, pair_i, pair_j, pair_off, row_start, diag_off, iorbs, nat,
                   coord, a.g) ||
      !tab.upload(*tables) || !d_p.upload(p, static_cast<size_t>(mpack)) ||
      !d_dxyz.upload(dxyz, 3 * na) || !d_q.alloc(na)) {
    return 2;
  }
  a.g.distance_gate = distance_gate;
  a.g.d_on_device = d_on_device;
  a.g.cutof1 = tables->cutof1;
  a.g.cutof2 = cutof2;
  a.p = d_p.ptr;
  a.tore = tab.tore.ptr;
  a.ovl = tab.ovl;
  a.core = tab.core;
  a.cutofp = cutofp;
  a.chnge = chnge;
  a.chnge2 = 0.5 * chnge;
  a.cnst = cnst;
  a.fpc_9 = fpc_9;
  a.ev = tables->ev;
  a.force = force;
  a.dxyz = d_dxyz.ptr;

  mozyme_atom_charge_kernel<<<(numat + 255) / 256, 256>>>(a, d_q.ptr);
  if (npairs > 0) {
    const int grid = (npairs + kPairThreads - 1) / kPairThreads;
    mozyme_pair_gradient_kernel<false><<<grid, kPairThreads>>>(a);
    if (d_on_device) {
      const int grid_d = (npairs + kPairThreadsD - 1) / kPairThreadsD;
      mozyme_pair_gradient_kernel<true><<<grid_d, kPairThreadsD, kPairSharedD>>>(a);
    }
  }
  if (numat >= 2) {
    dim3 grid((numat + kPointThreads - 1) / kPointThreads, numat - 1);
    mozyme_point_gradient_kernel<<<grid, kPointThreads>>>(a, d_q.ptr);
  }
  int code = finish_launch("MOZYME GPU gradient", geom.status, d_pairs_out);
  if (code == 0) {
    if (cudaMemcpy(dxyz, d_dxyz.ptr, 3 * na * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
      code = 2;
    }
  }
  const double ms = timer.stop_ms();
  if (ms_out) *ms_out = ms;
  return code;
}

// Fortran entry point for the block-pair part of hcore_for_MOZYME.  h (mpack)
// is accumulated in place; *enuc_out receives the summed core-core repulsion
// of the device pairs.  Return codes as for the gradient.
extern "C" int mopac_cuda_mozyme_hcore_pairs(
    int numat, int mpack, int npairs, const int *pair_i, const int *pair_j,
    const int *pair_off, const int *row_start, const int *diag_off,
    const int *iorbs, const int *nat, const double *coord, int distance_gate,
    int d_on_device, double cutof2, const MozymePairTablesC *tables, double *h,
    double *enuc_out, double *ms_out, int *d_pairs_out) {
  if (numat <= 0 || mpack <= 0 || !iorbs || !nat || !coord || !h || !tables || !enuc_out ||
      !row_start || !diag_off) {
    return 1;
  }
  if (npairs > 0 && (!pair_i || !pair_j || !pair_off)) return 1;
  if (d_pairs_out) *d_pairs_out = 0;
  *enuc_out = 0.0;

  EventTimer timer;
  GeomBuffers geom;
  PairTables tab;
  DeviceArray<double> d_h, d_enuc;
  HcoreArgs a;
  std::memset(&a, 0, sizeof(a));
  if (!geom.upload(numat, npairs, pair_i, pair_j, pair_off, row_start, diag_off, iorbs, nat,
                   coord, a.g) ||
      !tab.upload(*tables) || !d_h.upload(h, static_cast<size_t>(mpack)) || !d_enuc.alloc(1) ||
      cudaMemset(d_enuc.ptr, 0, sizeof(double)) != cudaSuccess) {
    return 2;
  }
  a.g.distance_gate = distance_gate;
  a.g.d_on_device = d_on_device;
  a.g.cutof1 = tables->cutof1;
  a.g.cutof2 = cutof2;
  a.ovl = tab.ovl;
  a.core = tab.core;
  a.h = d_h.ptr;
  a.enuc = d_enuc.ptr;

  if (npairs > 0) {
    const int grid = (npairs + kPairThreads - 1) / kPairThreads;
    mozyme_hcore_pairs_kernel<false><<<grid, kPairThreads>>>(a);
    if (d_on_device) {
      const int grid_d = (npairs + kPairThreadsD - 1) / kPairThreadsD;
      mozyme_hcore_pairs_kernel<true><<<grid_d, kPairThreadsD, kPairSharedD>>>(a);
    }
  }
  int code = finish_launch("MOZYME GPU hcore", geom.status, d_pairs_out);
  if (code == 0) {
    if (cudaMemcpy(h, d_h.ptr, static_cast<size_t>(mpack) * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(enuc_out, d_enuc.ptr, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
      code = 2;
    }
  }
  const double ms = timer.stop_ms();
  if (ms_out) *ms_out = ms;
  return code;
}

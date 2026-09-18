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

// GPU evaluation of the PM6-DH+ / PM7 hydrogen-bond correction
// (src/corrections/H_bond_correction_EH_plus.F90, EH_plus) and its
// finite-difference Cartesian gradient, as done in Hydrogen_bond_corrections
// (delta = 1e-5 A on each of the up to nine atoms of a pair, forward
// difference).  One thread per pair for the energies, one thread per
// (pair, atom slot, coordinate) probe for the gradient.  Non-periodic only.
// The geometry helpers replicate bangle / dihed / dang of src/geometry.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <new>

namespace {

constexpr int kThreads = 128;
constexpr int kSlots = 9;         // hblist(ii, 1..9): A, A1..A3, D, D1..D3, H
constexpr int kProbes = 3 * kSlots;

struct HbArgs {
  int numat;
  int nrpairs;
  int max_h_bonds;
  const int *nat;       // numat
  const double *coord;  // 3 x numat
  const int *hblist;    // Fortran (max_h_bonds, 10), column-major
  const int *nrbondsa;  // nrpairs
  const int *nrbondsb;  // nrpairs
  int pm7;
  double delta;
  double a0, ev, fpc9;  // funcon_C constants (runtime CODATA set)
  double *e_pair;       // nrpairs (energy of each pair)
  double *dxyz;         // 3 x numat, accumulated
};

// Local copy of the pair's atoms (slot s holds the coordinates of atom
// hblist(ii, s+1)); a displaced probe applies delta to every slot that
// refers to the displaced atom.
struct PairGeom {
  double x[kSlots][3];
  int atom[kSlots];
  int natA, natD, nbondsa, nbondsb, flag;
};

__device__ __forceinline__ double dist2_s(const PairGeom &g, int a, int b) {
  const double dx = g.x[a][0] - g.x[b][0];
  const double dy = g.x[a][1] - g.x[b][1];
  const double dz = g.x[a][2] - g.x[b][2];
  return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ double distance_s(const PairGeom &g, int a, int b) {
  return sqrt(dist2_s(g, a, b));
}

// bangle(xyz, i, j, k): angle i-j-k
__device__ double angle_s(const PairGeom &g, int i, int j, int k) {
  const double d2ij = dist2_s(g, i, j);
  const double d2jk = dist2_s(g, j, k);
  const double d2ik = dist2_s(g, i, k);
  const double xy = sqrt(d2ij * d2jk);
  if (xy < 1.0e-20) return 0.0;
  double temp = 0.5 * (d2ij + d2jk - d2ik) / xy;
  temp = fmin(1.0, temp);
  temp = fmax(-1.0, temp);
  return acos(temp);
}

// dang(a1, a2, b1, b2)
__device__ double dang_d(double a1, double a2, double b1, double b2) {
  const double zero = 1.0e-6;
  if (fabs(a1) >= zero || fabs(a2) >= zero) {
    if (fabs(b1) >= zero || fabs(b2) >= zero) {
      const double anorm = 1.0 / sqrt(a1 * a1 + a2 * a2);
      const double bnorm = 1.0 / sqrt(b1 * b1 + b2 * b2);
      a1 *= anorm;
      a2 *= anorm;
      b1 *= bnorm;
      b2 *= bnorm;
      const double sinth = a1 * b2 - a2 * b1;
      double costh = a1 * b1 + a2 * b2;
      costh = fmin(1.0, costh);
      costh = fmax(-1.0, costh);
      double rcos = acos(costh);
      if (fabs(rcos) >= 4.0e-5) {
        if (sinth > 0.0) rcos = 6.28318530717959 - rcos;
        return -rcos;
      }
    }
  }
  return 0.0;
}

// dihed(xyz, i, j, k, l): dihedral angle in [0, 2 pi)
__device__ double torsion_s(const PairGeom &g, int i, int j, int k, int l) {
  const double pi = 3.14159265358979323846;
  const double xi1 = g.x[i][0] - g.x[k][0];
  const double xj1 = g.x[j][0] - g.x[k][0];
  const double xl1 = g.x[l][0] - g.x[k][0];
  const double yi1 = g.x[i][1] - g.x[k][1];
  const double yj1 = g.x[j][1] - g.x[k][1];
  const double yl1 = g.x[l][1] - g.x[k][1];
  const double zi1 = g.x[i][2] - g.x[k][2];
  const double zj1 = g.x[j][2] - g.x[k][2];
  const double zl1 = g.x[l][2] - g.x[k][2];
  const double dist = sqrt(xj1 * xj1 + yj1 * yj1 + zj1 * zj1);
  double cosa = zj1 / dist;
  cosa = fmin(1.0, cosa);
  cosa = fmax(-1.0, cosa);
  const double ddd = 1.0 - cosa * cosa;
  double xi2, xl2, yi2, yl2, costh, sinth;
  bool general = false;
  double yxdist = 0.0;
  if (ddd > 0.0) {
    yxdist = dist * sqrt(ddd);
    if (yxdist > 1.0e-6) general = true;
  }
  if (general) {
    const double cosph = yj1 / yxdist;
    const double sinph = xj1 / yxdist;
    xi2 = xi1 * cosph - yi1 * sinph;
    xl2 = xl1 * cosph - yl1 * sinph;
    yi2 = xi1 * sinph + yi1 * cosph;
    const double yj2 = xj1 * sinph + yj1 * cosph;
    yl2 = xl1 * sinph + yl1 * cosph;
    costh = cosa;
    sinth = yj2 / dist;
  } else {
    xi2 = xi1;
    xl2 = xl1;
    yi2 = yi1;
    yl2 = yl1;
    costh = cosa;
    sinth = 0.0;
  }
  const double yi3 = yi2 * costh - zi1 * sinth;
  const double yl3 = yl2 * costh - zl1 * sinth;
  double angle = dang_d(xl2, yl3, xi2, yi3);
  if (angle < 0.0) angle = pi * 2.0 + angle;
  if (angle >= 6.28318530717959) angle = 0.0;
  return angle;
}

// One side of the correction (acceptor A = slots 0..3 with neighbours 1..3,
// or donor D = slots 4..7): returns false when the pair contributes nothing.
// s0 = base slot (0 or 4), nb = nrbonds of that heavy atom, natx = its Z.
struct SideResult {
  double angle2_cos;
  double torsion_cos;
};

__device__ bool side_terms(const PairGeom &g, int s0, int nb, int natx, bool second,
                           SideResult *out) {
  const double pi = 3.14159265358979323846;
  const int H = 8;
  // The Fortran writes the acceptor-side shifts as pi/(180.d0/109.48) and
  // pi/(180.d0/54.74): single-precision literals (109.480003..., 54.740001...),
  // while the donor side uses 109.48d0 / 54.74d0.  Reproduce both exactly.
  const double a109 = second ? 109.48 : static_cast<double>(109.48f);
  const double a54 = second ? 54.74 : static_cast<double>(54.74f);
  double torsion_shift = 0.0, angle2_shift = 0.0, angle2_shift_2 = 0.0;
  double torsion_check = 0.0, torsion_check_bac;
  bool torsion_check_set = false, torsion_check_set2 = false;
  if (natx == 8) {
    if (nb == 1) {
      angle2_shift = pi;
      angle2_shift_2 = pi / (180.0 / 120.0);
      torsion_shift = 0.0;
      torsion_check_set2 = true;
    } else {
      angle2_shift = pi / (180.0 / a109);
      angle2_shift_2 = angle2_shift;
      torsion_shift = pi / (180.0 / a54);
    }
  } else if (natx == 7) {
    if (nb == 2) {
      angle2_shift = pi / (180.0 / 120.0);
      angle2_shift_2 = angle2_shift;
      torsion_shift = 0.0;
    } else {
      angle2_shift = pi / (180.0 / a109);
      angle2_shift_2 = angle2_shift;
      torsion_shift = pi / (180.0 / a54);
      torsion_check_set = true;
    }
  }
  if (torsion_check_set) {
    torsion_check = torsion_s(g, s0 + 2, s0 + 1, s0, s0 + 3);
    if (torsion_check <= -pi) torsion_check = torsion_check + 2.0 * pi;
    if (torsion_check > pi) torsion_check = torsion_check - 2.0 * pi;
    if (torsion_check < 0.0) {
      torsion_check = -pi - torsion_check;
    } else {
      torsion_check = pi - torsion_check;
    }
    torsion_check_bac = torsion_check;
    if (torsion_check < 0.0) torsion_check = -1.0 * torsion_check;
    torsion_check = torsion_check * 180.0 / pi;
    torsion_shift = torsion_shift + pi / (180.0 / ((54.74 - torsion_check) / 54.74 * 35.26));
    angle2_shift = angle2_shift - pi / (180.0 / ((54.74 - torsion_check) / 54.74 * 19.48));
    angle2_shift_2 = angle2_shift;
    torsion_check = torsion_check_bac;
  }
  const double angle2 = angle_s(g, s0 + 1, s0, H);
  double angle2_cos = cos(angle2_shift - angle2);
  const double angle2_cos_2 = cos(angle2_shift_2 - angle2);
  if (angle2_cos_2 > angle2_cos) angle2_cos = angle2_cos_2;
  if (angle2_cos <= 0.0) return false;
  double torsion_correct = torsion_s(g, s0 + 2, s0 + 1, s0, H);
  if (torsion_correct <= -pi) torsion_correct = torsion_correct + 2.0 * pi;
  if (torsion_correct > pi) torsion_correct = torsion_correct - 2.0 * pi;
  if ((!torsion_check_set2) || (fabs(torsion_correct * 180.0 / pi) > 90.0)) {
    if (torsion_correct < 0.0) {
      torsion_correct = -pi - torsion_correct;
    } else {
      torsion_correct = pi - torsion_correct;
    }
  }
  double torsion_cos;
  if (torsion_check < 0.0) {
    double tv = torsion_shift - torsion_correct;
    if (tv <= -pi) tv = tv + 2.0 * pi;
    if (tv > pi) tv = tv - 2.0 * pi;
    torsion_cos = cos(tv);
  } else if (torsion_check > 0.0) {
    double tv = -torsion_shift - torsion_correct;
    if (tv <= -pi) tv = tv + 2.0 * pi;
    if (tv > pi) tv = tv - 2.0 * pi;
    torsion_cos = cos(tv);
  } else {
    double tv = torsion_shift - torsion_correct;
    double tv2 = -torsion_shift - torsion_correct;
    if (tv <= -pi) tv = tv + 2.0 * pi;
    if (tv > pi) tv = tv - 2.0 * pi;
    if (tv2 <= -pi) tv2 = tv2 + 2.0 * pi;
    if (tv2 > pi) tv2 = tv2 - 2.0 * pi;
    torsion_cos = cos(tv);
    const double torsion_cos_2 = cos(tv2);
    if (torsion_cos_2 > torsion_cos) torsion_cos = torsion_cos_2;
  }
  if (distance_s(g, H, s0) > distance_s(g, H, s0 + 1) && torsion_check_set2) torsion_cos = 0.0;
  if (g.atom[s0 + 2] == g.atom[s0 + 3] || g.atom[H] == g.atom[s0 + 3]) torsion_cos = 1.0;
  if (second) {
    torsion_cos = fabs(torsion_cos);
  } else if (torsion_cos < 0.0) {
    return false;
  }
  out->angle2_cos = angle2_cos;
  out->torsion_cos = torsion_cos;
  return true;
}

__device__ double eh_plus_d(const PairGeom &g, const HbArgs &a) {
  const int pm7 = a.pm7;
  const double a0 = a.a0;
  const double eV = a.ev;
  const double fpc_9 = a.fpc9;
  const double shortcut = 2.4, longcut = 7.0, covcut = 1.2;
  const int A = 0, D = 4, H = 8;
  double scale_nsp3, scale_osp3, scale_nsp2, scale_osp2;
  if (pm7) {
    scale_nsp3 = -0.171271 * a0 * a0;
    scale_osp3 = -0.098822 * a0 * a0;
    scale_nsp2 = -0.171271 * a0 * a0;
  } else {
    scale_nsp3 = -0.16 * a0 * a0;
    scale_osp3 = -0.12 * a0 * a0;
    scale_nsp2 = scale_nsp3;
  }
  scale_osp2 = scale_osp3;
  const double hartree2kcal = eV * fpc_9;
  if (g.flag == -666) return 0.0;
  const double angle_cos = -cos(angle_s(g, A, H, D));
  if (angle_cos <= 0.0) return 0.0;
  SideResult sa, sd;
  if (!side_terms(g, A, g.nbondsa, g.natA, false, &sa)) return 0.0;
  if (!side_terms(g, D, g.nbondsb, g.natD, true, &sd)) return 0.0;
  double scale_a, scale_b;
  if (g.natA == 7) {
    scale_a = (g.nbondsa >= 3) ? scale_nsp3 : scale_nsp2;
  } else {
    scale_a = (g.nbondsa >= 2) ? scale_osp3 : scale_osp2;
  }
  if (g.natD == 7) {
    scale_b = (g.nbondsb >= 3) ? scale_nsp3 : scale_nsp2;
  } else {
    scale_b = (g.nbondsb >= 2) ? scale_osp3 : scale_osp2;
  }
  const double scale_c = (scale_a + scale_b) / 2.0;
  const double ha_dist = distance_s(g, H, A);
  const double hb_dist = distance_s(g, H, D);
  double xc_dist = fmin(ha_dist, hb_dist);
  double e;
  if (pm7) {
    double XY_dist = fmax(ha_dist, hb_dist) - xc_dist;
    double damping;
    if (XY_dist > 0.5) {
      damping = 1.0 - 1.0 / (1.0 + exp(-60.0 * (xc_dist / covcut - 1.0)));
    } else {
      damping = 1.0;
    }
    xc_dist = distance_s(g, A, D);
    damping = damping / (1.0 + exp(-100.0 * (xc_dist / shortcut - 1.0)));
    damping = damping * (1.0 - 1.0 / (1.0 + exp(-10.0 * (xc_dist / longcut - 1.0))));
    XY_dist = distance_s(g, A, D);
    const double prod = sa.angle2_cos * sa.torsion_cos * sd.angle2_cos * sd.torsion_cos;
    e = scale_c / pow(XY_dist, 2.0) * angle_cos * angle_cos *
        (1.0 - (1.0 - prod) * (1.0 - prod)) * hartree2kcal * damping;
    if (g.natA == 8 && g.natD == 8) {
      const double d = fmax(XY_dist - 2.67, 0.0);
      const double short_t = -2.5 * exp(-80.0 * d * d) * angle_cos * angle_cos * angle_cos * angle_cos;
      e = e + short_t;
    }
  } else {
    double damping = 1.0 - 1.0 / (1.0 + exp(-60.0 * (xc_dist / covcut - 1.0)));
    xc_dist = distance_s(g, A, D);
    damping = damping / (1.0 + exp(-100.0 * (xc_dist / shortcut - 1.0)));
    damping = damping * (1.0 - 1.0 / (1.0 + exp(-10.0 * (xc_dist / longcut - 1.0))));
    const double dAD = distance_s(g, A, D);
    e = scale_c / pow(dAD, 2.0) * angle_cos * angle_cos * sa.angle2_cos * sa.angle2_cos *
        sa.torsion_cos * sa.torsion_cos * sd.angle2_cos * sd.angle2_cos * sd.torsion_cos *
        sd.torsion_cos * hartree2kcal * damping;
  }
  return e;
}

__device__ void load_pair(const HbArgs &a, int ii, PairGeom *g) {
  for (int s = 0; s < kSlots; ++s) {
    const int atom = a.hblist[s * a.max_h_bonds + ii];
    g->atom[s] = atom;
    if (atom >= 1 && atom <= a.numat) {
      g->x[s][0] = a.coord[3 * (atom - 1)];
      g->x[s][1] = a.coord[3 * (atom - 1) + 1];
      g->x[s][2] = a.coord[3 * (atom - 1) + 2];
    } else {
      g->x[s][0] = 0.0;
      g->x[s][1] = 0.0;
      g->x[s][2] = 0.0;
    }
  }
  g->flag = a.hblist[9 * a.max_h_bonds + ii];
  const int atA = g->atom[0], atD = g->atom[4];
  g->natA = (atA >= 1 && atA <= a.numat) ? a.nat[atA - 1] : 0;
  g->natD = (atD >= 1 && atD <= a.numat) ? a.nat[atD - 1] : 0;
  g->nbondsa = a.nrbondsa[ii];
  g->nbondsb = a.nrbondsb[ii];
}

__global__ void hbond_energy_kernel(HbArgs a) {
  const int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii >= a.nrpairs) return;
  PairGeom g;
  load_pair(a, ii, &g);
  a.e_pair[ii] = eh_plus_d(g, a);
}

// Thread per (pair, slot, coordinate): forward difference of the pair energy
// with atom hblist(ii, slot) displaced by delta, added to dxyz of that atom.
__global__ void hbond_gradient_kernel(HbArgs a) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(a.nrpairs) * kProbes;
  if (t >= total) return;
  const int ii = static_cast<int>(t / kProbes);
  const int probe = static_cast<int>(t % kProbes);
  const int slot = probe / 3;
  const int comp = probe % 3;
  const double e0 = a.e_pair[ii];
  if (!(e0 < -0.01)) return;
  PairGeom g;
  load_pair(a, ii, &g);
  const int k = g.atom[slot];
  if (k < 1 || k > a.numat) return;
  // each distinct atom once (the CPU d_list); the H is slot 8
  for (int s = 0; s < slot; ++s) {
    if (g.atom[s] == k) return;
  }
  // connected(H, k, 8**2)
  const int hatom = g.atom[8];
  if (hatom < 1 || hatom > a.numat) return;
  {
    const double dx = a.coord[3 * (hatom - 1)] - a.coord[3 * (k - 1)];
    const double dy = a.coord[3 * (hatom - 1) + 1] - a.coord[3 * (k - 1) + 1];
    const double dz = a.coord[3 * (hatom - 1) + 2] - a.coord[3 * (k - 1) + 2];
    if (!(dx * dx + dy * dy + dz * dz < 64.0)) return;
  }
  for (int s = 0; s < kSlots; ++s) {
    if (g.atom[s] == k) g.x[s][comp] += a.delta;
  }
  double e1 = eh_plus_d(g, a);
  if (fabs(e1) > 1.0e-5) {
    e1 = (e1 - e0) / a.delta;
    atomicAdd(&a.dxyz[3 * (k - 1) + comp], e1);
  }
}

template <typename T>
struct DevPtr {
  T *ptr = nullptr;
  bool upload(const T *host, size_t n) {
    if (cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)) != cudaSuccess) return false;
    return cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }
  bool alloc(size_t n) {
    return cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)) == cudaSuccess;
  }
  ~DevPtr() {
    if (ptr) cudaFree(ptr);
  }
};

}  // namespace

// Fortran entry point.  Returns 0 on success (E_hb in *energy_out, number of
// pairs with energy < -1 kcal/mol in *nhb_out, gradient accumulated into dxyz
// when l_grad != 0), 1 on bad arguments, 2 on CUDA failure.
extern "C" int mopac_cuda_dh_plus_hbonds(int numat, int nrpairs, int max_h_bonds, const int *nat,
                                         const double *coord, const int *hblist,
                                         const int *nrbondsa, const int *nrbondsb, int method_pm7,
                                         int l_grad, double delta, double a0, double ev, double fpc9,
                                         double *energy_out, int *nhb_out, double *dxyz,
                                         double *ms_out) {
  if (numat <= 0 || nrpairs < 0 || max_h_bonds <= 0 || !nat || !coord || !hblist ||
      !nrbondsa || !nrbondsb || !energy_out || !nhb_out || (l_grad && !dxyz)) {
    return 1;
  }
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0, 0);
  *energy_out = 0.0;
  *nhb_out = 0;
  int code = 0;
  if (nrpairs > 0) {
    const size_t na = static_cast<size_t>(numat);
    const size_t np = static_cast<size_t>(nrpairs);
    DevPtr<int> d_nat, d_hblist, d_nra, d_nrb;
    DevPtr<double> d_coord, d_dxyz, d_e;
    bool ok = d_nat.upload(nat, na) && d_coord.upload(coord, 3 * na) &&
              d_hblist.upload(hblist, static_cast<size_t>(max_h_bonds) * 10) &&
              d_nra.upload(nrbondsa, np) && d_nrb.upload(nrbondsb, np) && d_e.alloc(np) &&
              (!l_grad || d_dxyz.upload(dxyz, 3 * na));
    if (!ok) code = 2;
    if (code == 0) {
      HbArgs a;
      a.numat = numat;
      a.nrpairs = nrpairs;
      a.max_h_bonds = max_h_bonds;
      a.nat = d_nat.ptr;
      a.coord = d_coord.ptr;
      a.hblist = d_hblist.ptr;
      a.nrbondsa = d_nra.ptr;
      a.nrbondsb = d_nrb.ptr;
      a.pm7 = method_pm7;
      a.delta = delta;
      a.a0 = a0;
      a.ev = ev;
      a.fpc9 = fpc9;
      a.e_pair = d_e.ptr;
      a.dxyz = d_dxyz.ptr;
      hbond_energy_kernel<<<(nrpairs + kThreads - 1) / kThreads, kThreads>>>(a);
      cudaError_t err = cudaGetLastError();
      if (err == cudaSuccess && l_grad) {
        const long long total = static_cast<long long>(nrpairs) * kProbes;
        const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
        hbond_gradient_kernel<<<blocks, kThreads>>>(a);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::fprintf(stderr, "[GPU ERROR] dh+ hbonds: %s\n", cudaGetErrorString(err));
        code = 2;
      } else {
        // The CPU sums the pair energies in pair order: do the same on the host.
        double *e_host = new (std::nothrow) double[np];
        if (!e_host || cudaMemcpy(e_host, d_e.ptr, np * sizeof(double), cudaMemcpyDeviceToHost) !=
                           cudaSuccess) {
          code = 2;
        } else {
          double e_hb = 0.0;
          int nhb = 0;
          for (size_t i = 0; i < np; ++i) {
            e_hb += e_host[i];
            if (e_host[i] < -1.0) ++nhb;
          }
          *energy_out = e_hb;
          *nhb_out = nhb;
        }
        delete[] e_host;
        if (code == 0 && l_grad &&
            cudaMemcpy(dxyz, d_dxyz.ptr, 3 * na * sizeof(double), cudaMemcpyDeviceToHost) !=
                cudaSuccess) {
          code = 2;
        }
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

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

// Device port of the diatomic pieces of src/forces/dhc.F90:
//   * the numat = -2 ("deriv") branch of src/SCF/fock2.F90 (two-centre
//     two-electron Fock contribution of one atom pair, RHF, sp basis only)
//   * src/integrals/helect.F90 (electronic energy of the diatomic system)
//
// Conventions (all packed arrays are Fortran lower-triangle order, 0-based in
// memory, i.e. element (i,j) with i >= j lives at i*(i-1)/2 + j - 1):
//   atom 1 owns orbitals 1..n1, atom 2 owns orbitals n1+1..n1+n2.
//   w holds the two-electron integrals in rotate()'s order: (ij|kl) with
//   i >= j running over atom-2 orbitals in the outer loops and k >= l over
//   atom-1 orbitals in the inner loops (this is fock2's general branch, which
//   is algebraically identical to the jab/kab fast paths used on the CPU).
#pragma once

namespace mozyme_pair {

constexpr int kMaxOrbitals = 8;                                   // sp + sp
constexpr int kMaxLinear = kMaxOrbitals * (kMaxOrbitals + 1) / 2; // 36
constexpr int kMaxW = 100;

__device__ __forceinline__ int tri1(int i) { return (i * (i - 1)) / 2; }

// Two-centre two-electron Fock terms for the pair; f is accumulated in place.
// ptot: total density, pa: alpha density (= 0.5 * ptot for RHF).
__device__ __forceinline__ void fock2_diatomic_sp(int n1, int n2, const double *w,
                                                  const double *ptot,
                                                  const double *pa, double *f) {
  const int ja = 1, jb = n1;
  const int ia = n1 + 1, ib = n1 + n2;
  int kk = 0;
  for (int i = ia; i <= ib; ++i) {
    const int ka = tri1(i);
    for (int j = ia; j <= i; ++j) {
      const int kb = tri1(j);
      const int ij = ka + j;
      const double aa = (i == j) ? 1.0 : 2.0;
      for (int k = ja; k <= jb; ++k) {
        const int kc = tri1(k);
        const int ik = ka + k;
        const int jk = kb + k;
        for (int l = ja; l <= k; ++l) {
          const int il = ka + l;
          const int jl = kb + l;
          const int kl = kc + l;
          const double bb = (k == l) ? 1.0 : 2.0;
          const double wv = w[kk++];
          // Coulomb
          f[ij - 1] += bb * wv * ptot[kl - 1];
          f[kl - 1] += aa * wv * ptot[ij - 1];
          // Exchange
          const double a = wv * aa * bb * 0.25;
          f[ik - 1] -= a * pa[jl - 1];
          f[il - 1] -= a * pa[jk - 1];
          f[jk - 1] -= a * pa[il - 1];
          f[jl - 1] -= a * pa[ik - 1];
        }
      }
    }
  }
}

// helect(): 0.5 * sum_ii p(h+f) + sum_{i>j} p(h+f) over the packed triangle.
__device__ __forceinline__ double helect_packed(int n, const double *pa,
                                                const double *h,
                                                const double *f) {
  double ed = 0.0, ee = 0.0;
  int k = 0;
  for (int i = 1; i <= n; ++i) {
    for (int j = 1; j < i; ++j, ++k) ee += pa[k] * (h[k] + f[k]);
    ed += pa[k] * (h[k] + f[k]);
    ++k;
  }
  return ee + 0.5 * ed;
}

// Assemble the diatomic energy given the already-evaluated integrals:
//   smat  : 9x9 column-major overlap-derived one-electron block, row = atom-1
//           orbital, column = atom-2 orbital (h1elec(nat1, nat2, x1, x2)); may
//           be nullptr when the overlap term is to be skipped (Cb atoms).
//   e_at1 : packed one-centre attraction block of atom 1 (rotate's e2a when
//           rotate is called as rotate(nat2, nat1, x2, x1, ...)).
//   e_at2 : packed block of atom 2 (rotate's e1b in that call).
//   w     : two-electron integrals in rotate order (see header comment).
//   pdi   : packed total density of the diatomic system (atom 1 first).
// Returns 2*helect + enuc, i.e. dhc()'s dener for RHF.
__device__ __forceinline__ double diatomic_energy_sp(int n1, int n2,
                                                     const double *smat,
                                                     const double *e_at1,
                                                     const double *e_at2,
                                                     const double *w,
                                                     double enuc,
                                                     const double *pdi) {
  const int n = n1 + n2;
  const int linear = tri1(n) + n;
  double h[kMaxLinear];
  double f[kMaxLinear];
  double pa[kMaxLinear];
  for (int i = 0; i < linear; ++i) {
    h[i] = 0.0;
    pa[i] = 0.5 * pdi[i];
  }
  if (smat) {
    for (int j1 = 1; j1 <= n2; ++j1) {
      const int base = tri1(n1 + j1);
      for (int i1 = 1; i1 <= n1; ++i1) {
        h[base + i1 - 1] = smat[(i1 - 1) + 9 * (j1 - 1)];
      }
    }
  }
  {
    int i2 = 0;
    for (int i1 = 1; i1 <= n1; ++i1) {
      const int base = tri1(i1);
      for (int j1 = 1; j1 <= i1; ++j1) h[base + j1 - 1] += e_at1[i2++];
    }
    i2 = 0;
    for (int i1 = n1 + 1; i1 <= n; ++i1) {
      const int base = tri1(i1) + n1;
      for (int j1 = 1; j1 <= i1 - n1; ++j1) h[base + j1 - 1] += e_at2[i2++];
    }
  }
  for (int i = 0; i < linear; ++i) f[i] = h[i];
  fock2_diatomic_sp(n1, n2, w, pdi, pa, f);
  const double ee = helect_packed(n, pa, h, f);
  return 2.0 * ee + enuc;
}

}  // namespace mozyme_pair

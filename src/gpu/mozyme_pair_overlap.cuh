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
//
// mozyme_pair_overlap.cuh
// -----------------------
// Self-contained CUDA device port of MOPAC's one-electron two-centre matrix
// h1elec() (src/integrals/h1elec.F90) for s/p-only atom pairs, including the
// routines it calls:
//   diat(), diat2(), ss()          src/integrals/diat.F90
//   set(), aintgs(), bintgs()      src/integrals/set.F90
//   bfn()                          src/integrals/bfn.F90
//   coe()                          src/integrals/coe.F90
//
// Numerics are double precision and every expression is transcribed with the
// same association order as the Fortran (gfortran):
//   * constant integer powers x**n are expanded to the multiplication chains
//     GCC's optimiser (powi_as_mults, -O2 and above, i.e. MOPAC's Release
//     build) generates: x^2=x*x, x^3=x^2*x, x^4=x^2*x^2, x^5=x^3*x^2,
//     x^6=x^3*x^3, x^7=x^4*x^3.  (The gfortran front end only inlines
//     |n| <= 2 itself; at -O0 the remaining powers go through libgcc's
//     __powidf2 instead, which forms x^5 and x^6 in a different order and
//     therefore differs from -O2/-O3 by a few ulp in diat2 cases 3, 4, 5.)
//   * variable integer powers ((-x)**m, r**(na+nb+1), ua**na, ub**nb) use
//     the libgcc __powidf2 square-and-multiply algorithm (mpo_powi), which
//     is what gfortran emits at every optimisation level;
//   * Fortran and C both associate '*', '/', '+', '-' left-to-right, so the
//     Fortran expressions are copied verbatim (only '**' needed rewriting).
// Bit-for-bit agreement with a CPU build additionally requires that neither
// side contracts a*b+c into FMA (nvcc: -fmad=false; gfortran/clang:
// -ffp-contract=off) and CUDA's exp() is not correctly rounded, so at most
// ulp-level differences are to be expected on real hardware.
//
// The header only needs <cuda_runtime.h> (under nvcc/clang-cuda) and
// <math.h>. It can also be compiled as plain C++ for host-side testing: the
// CUDA attribute keywords are defined away when __CUDACC__ is not set.
//
// Fortran -> C index conventions used throughout (1-based Fortran on the
// left, 0-based C on the right):
//   natorb(n), zs(n), zp(n), zd(n), betas(n), betap(n), betad(n)
//                                  -> ptr[n-1]                 (n = 1..107)
//   npq(n,l)   (integer(107,3))    -> npq[(n-1) + 107*(l-1)]   (l = 1..3)
//   fact(m)    (0:17)              -> mpo_fact[m]
//   a(i), b(i) (dimension(7))      -> st.a[i-1], st.b[i-1]
//   s(i,j,k)   (3,3,3)             -> s[(i-1) + 3*(j-1) + 9*(k-1)]
//   s1/s2/s3(i,j) = s(i,j,1/2/3)   (Fortran EQUIVALENCE)
//   c(i,k,l)   (3,5,5) = c(75)     -> c[(i-1) + 3*(k-1) + 15*(l-1)]
//   c1..c5(i,k) = c(i,k,1..5)      (Fortran EQUIVALENCE)
//   ival(i,k)  (3,5)               -> mpo_ival[(i-1) + 3*(k-1)]
//   inmb(n)    (17)                -> mpo_inmb[n-1]
//   iii(n)     (78)                -> mpo_iii[n-1]
//   di(i,j) / smat(i,j) (9,9)      -> smat[(i-1) + 9*(j-1)]
//   af(0:19), bf(0:19)             -> af[n], bf[n]
//   aff(la,m,i) (0:2,0:2,0:2)      -> mpo_aff(la,m,i)
//   bi(n,k)    (0:12,0:12)         -> mpo_binom(n,k)

#ifndef MOZYME_PAIR_OVERLAP_CUH
#define MOZYME_PAIR_OVERLAP_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif
#include <math.h>

#define MPO_DEV static __device__ __forceinline__

// ---------------------------------------------------------------------------
// Parameter block.  All pointers are device pointers to element-indexed
// tables copied from the Fortran modules (1-based by atomic number in
// Fortran; use ptr[n-1] here).
// ---------------------------------------------------------------------------
struct MozymePairOverlapParams {
  // parameters_C::natorb(107)  - number of atomic orbitals per element
  //   (used by h1elec, diat, coe and for the use_diat2 test in diat).
  const int *natorb;
  // parameters_C::npq(107,3)   - principal quantum number of the s, p and d
  //   level of each element, column-major: npq(n,l) -> npq[(n-1)+107*(l-1)]
  //   (used by diat: npq(ni,1)/npq(nj,1) and npq(ni,i)/npq(nj,j) in ss path).
  const int *npq;
  // parameters_C::zs, zp, zd(107) - Slater exponents (used by diat).
  //   zd only feeds the ss path as max(zd(n),0.3) for the l=3 level, which
  //   diat evaluates whenever npq(n,1) >= 2 even for sp-only elements.
  const double *zs;
  const double *zp;
  const double *zd;
  // parameters_C::betas, betap, betad(107) - resonance parameters (used by
  //   h1elec for the (bi+bj) scaling).  betad only scales orbitals 5..9 and
  //   is never read on the sp path (natorb <= 4); kept so that the struct
  //   mirrors h1elec's use list exactly.
  const double *betas;
  const double *betap;
  const double *betad;
  // funcon_C::a0     - Bohr radius in Angstrom (used by diat2 and ss).
  double a0;
  // MOZYME_C::cutofs - squared overlap cutoff distance (Angstrom^2), h1elec.
  double cutofs;
  // overlaps_C::cutof1 - squared distance beyond which diat returns zero.
  double cutof1;
};

// ---------------------------------------------------------------------------
// Constant tables (verbatim copies of the Fortran DATA statements).
// ---------------------------------------------------------------------------

// overlaps_C::fact(0:17).  NOTE: the Fortran DATA statement gives fact(16) and
// fact(17) as the truncated literals 2.092278989D13 and 3.556874281D14 (not
// the exact factorials); the literals are reproduced, not recomputed.
static __device__ const double mpo_fact[18] = {
  1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0,
  362880.0, 3628800.0, 39916800.0, 479001600.0, 6227020800.0,
  8.71782912e10, 1.307674368e12, 2.092278989e13, 3.556874281e14 };

// diat: ival(3,5), data ival/ 1,0,9, 1,3,8, 1,4,7, 1,2,6, 0,0,5 /
static __device__ const int mpo_ival[15] = {
  1, 0, 9, 1, 3, 8, 1, 4, 7, 1, 2, 6, 0, 0, 5 };

// diat2: inmb(17)
static __device__ const int mpo_inmb[17] = {
  1, 0, 2, 2, 3, 4, 5, 6, 7, 0, 8, 8, 8, 9, 10, 11, 12 };

// diat2: iii(78)
static __device__ const int mpo_iii[78] = {
  1,                                     // jmax = 1
  2, 4,                                  // jmax = 2
  2, 4, 4,                               // jmax = 3
  2, 4, 4, 4,                            // jmax = 4
  2, 4, 4, 4, 4,                         // jmax = 5
  2, 4, 4, 4, 4, 4,                      // jmax = 6
  2, 4, 4, 4, 4, 4, 4,                   // jmax = 7
  3, 5, 5, 5, 5, 5, 5, 6,                // jmax = 8
  3, 5, 5, 5, 5, 5, 5, 6, 6,             // jmax = 9
  3, 5, 5, 5, 5, 5, 5, 6, 6, 6,          // jmax = 10
  3, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,       // jmax = 11
  3, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6 };  // jmax = 12

// ---------------------------------------------------------------------------
// Small arithmetic helpers reproducing gfortran's integer-power expansion.
// ---------------------------------------------------------------------------

// x**n for a run-time n: libgcc __powidf2 (square-and-multiply, LSB first).
MPO_DEV double mpo_powi(double x, int m)
{
  unsigned int n = (m < 0) ? (unsigned int)(-m) : (unsigned int)m;
  double y = (n % 2u) ? x : 1.0;
  while (n >>= 1) {
    x = x * x;
    if (n % 2u) y = y * x;
  }
  return (m < 0) ? 1.0 / y : y;
}

// x**n for compile-time n: gfortran powi_table addition chains.
MPO_DEV double mpo_pow2(double x) { return x * x; }
MPO_DEV double mpo_pow3(double x) { return (x * x) * x; }
MPO_DEV double mpo_pow4(double x) { const double x2 = x * x; return x2 * x2; }
MPO_DEV double mpo_pow5(double x) { const double x2 = x * x; return (x2 * x) * x2; }
MPO_DEV double mpo_pow6(double x) { const double x3 = (x * x) * x; return x3 * x3; }
MPO_DEV double mpo_pow7(double x)
{
  const double x2 = x * x;
  return (x2 * x2) * (x2 * x);
}

MPO_DEV int mpo_imin(int a, int b) { return (a < b) ? a : b; }
MPO_DEV int mpo_imax(int a, int b) { return (a > b) ? a : b; }

// ---------------------------------------------------------------------------
// overlaps_C state written by set()/aintgs()/bintgs() and read by diat2().
// In Fortran these are module variables (sa, sb, isp, ips, a(7), b(7)); the
// value of isp/ips used by diat2 is whatever the *last* call to set() left.
// ---------------------------------------------------------------------------
struct MpoSetState {
  double sa, sb;
  double a[7];   // a(1..7) -> a[0..6]
  double b[7];   // b(1..7) -> b[0..6]
  int isp, ips;
};

// aintgs(x, k): a(1) = exp(-x)/x ; a(i+1) = (a(i)*i + c)/x, i = 1..k
MPO_DEV void mpo_aintgs(double x, int k, double *a)
{
  const double c = exp(-x);
  a[0] = c / x;
  for (int i = 1; i <= k; ++i) {
    a[i] = (a[i - 1] * (double)i + c) / x;
  }
}

// bintgs(x, k): B integrals, b(1..k+1) -> b[0..k]
MPO_DEV void mpo_bintgs(double x, int k, double *b)
{
  const int io = 0;
  const double absx = fabs(x);
  int mode;   // 0: closed form (label 40), 1: series (label 60), 2: x ~ 0 (label 90)
  int last = 0;
  if (absx > 3.0) {
    mode = 0;
  } else if (absx > 2.0) {
    if (k <= 10) { mode = 0; } else { last = 15; mode = 1; }
  } else if (absx > 1.0) {
    if (k <= 7) { mode = 0; } else { last = 12; mode = 1; }
  } else if (absx > 0.5) {
    if (k <= 5) { mode = 0; } else { last = 7; mode = 1; }
  } else if (absx <= 1.0e-6) {
    mode = 2;
  } else {
    last = 6; mode = 1;
  }
  if (mode == 0) {
    const double expx = exp(x);
    const double expmx = 1.0 / expx;
    b[0] = (expx - expmx) / x;
    for (int i = 1; i <= k; ++i) {
      // (-1.D0)**i * expx is exactly +/- expx
      const double sgn_expx = (i & 1) ? -expx : expx;
      b[i] = ((double)i * b[i - 1] + sgn_expx - expmx) / x;
    }
  } else if (mode == 1) {
    for (int i = io; i <= k; ++i) {
      double y = 0.0;
      for (int m = io; m <= last; ++m) {
        double xf = 1.0;
        if (m != 0) xf = mpo_fact[m];
        y = y + mpo_powi(-x, m) * (double)(2 * ((m + i + 1) % 2)) / (xf * (double)(m + i + 1));
      }
      b[i] = y;
    }
  } else {
    for (int i = io; i <= k; ++i) {
      b[i] = (double)(2 * ((i + 1) % 2)) / ((double)i + 1.0);
    }
  }
}

// bfn(x, bf): B integrals for ss(), fills bf(1..13) -> bf[0..12] (k = 12)
MPO_DEV void mpo_bfn(double x, double *bf)
{
  const int k = 12;
  const int io = 0;
  const double absx = fabs(x);
  int mode;
  int last = 0;
  if (absx <= 3.0) {
    if (absx > 2.0) {
      last = 15; mode = 1;
    } else if (absx > 1.0) {
      last = 12; mode = 1;
    } else if (absx > 0.5) {
      last = 7; mode = 1;
    } else if (absx <= 1.0e-6) {
      mode = 2;
    } else {
      last = 6; mode = 1;
    }
  } else {
    mode = 0;
  }
  if (mode == 0) {
    const double expx = exp(x);
    const double expmx = 1.0 / expx;
    bf[0] = (expx - expmx) / x;
    for (int i = 1; i <= k; ++i) {
      const double sgn_expx = (i & 1) ? -expx : expx;
      bf[i] = ((double)i * bf[i - 1] + sgn_expx - expmx) / x;
    }
  } else if (mode == 1) {
    for (int i = io; i <= k; ++i) {
      double y = 0.0;
      for (int m = io; m <= last; ++m) {
        double xf = 1.0;
        if (m != 0) xf = mpo_fact[m];
        y = y + mpo_powi(-x, m) * (double)(2 * ((m + i + 1) % 2)) / (xf * (double)(m + i + 1));
      }
      bf[i] = y;
    }
  } else {
    for (int i = io; i <= k; ++i) {
      bf[i] = (double)(2 * ((i + 1) % 2)) / ((double)i + 1.0);
    }
  }
}

// set(s1, s2, na, nb, rab, ii)
MPO_DEV void mpo_set(double s1, double s2, int na, int nb, double rab, int ii,
                     MpoSetState &st)
{
  if (na <= nb) {
    st.isp = 1;
    st.ips = 2;
    st.sa = s1;
    st.sb = s2;
  } else {
    st.isp = 2;
    st.ips = 1;
    st.sa = s2;
    st.sb = s1;
  }
  int j = ii + 2;
  if (ii > 3) j = j - 1;
  const double alpha = 0.5 * rab * (st.sa + st.sb);
  const double beta = 0.5 * rab * (st.sb - st.sa);
  const int jcall = j - 1;
  mpo_aintgs(alpha, jcall, st.a);
  mpo_bintgs(beta, jcall, st.b);
}

// ---------------------------------------------------------------------------
// diat2(na, esa, epa, r12, nb, esb, epb, s, a0): s/p overlaps for first,
// second and third row elements (na, nb are atomic numbers 1..17).
// s is the 27-element s(3,3,3) array.
// ---------------------------------------------------------------------------
#define MPO_S(i, j, k) s[((i) - 1) + 3 * ((j) - 1) + 9 * ((k) - 1)]
#define MPO_A(n) st.a[(n) - 1]
#define MPO_B(n) st.b[(n) - 1]

MPO_DEV void mpo_diat2(int na, double esa, double epa, double r12,
                       int nb, double esb, double epb, double *s, double a0)
{
  MpoSetState st;
  const int jmax = mpo_imax(mpo_inmb[na - 1], mpo_inmb[nb - 1]);
  const int jmin = mpo_imin(mpo_inmb[na - 1], mpo_inmb[nb - 1]);
  const int nbond = (jmax * (jmax - 1)) / 2 + jmin;
  const int ii = mpo_iii[nbond - 1];
  for (int i = 0; i < 27; ++i) s[i] = 0.0;
  const double rab = r12 / a0;
  double rab4, rab6, w, rt3, d, e;
  switch (ii) {
  case 2: {
    // *** FIRST ROW - SECOND ROW OVERLAPS
    mpo_set(esa, esb, na, nb, rab, ii, st);
    rab4 = mpo_pow4(rab) * 0.125;
    w = sqrt(mpo_pow3(st.sa) * mpo_pow5(st.sb)) * rab4;
    MPO_S(1, 1, 1) = sqrt(1.0 / 3.0);
    MPO_S(1, 1, 1) = w * MPO_S(1, 1, 1) *
      (MPO_A(4) * MPO_B(1) - MPO_B(4) * MPO_A(1) + MPO_A(3) * MPO_B(2) - MPO_B(3) * MPO_A(2));
    if (na > 1) mpo_set(epa, esb, na, nb, rab, ii, st);
    if (nb > 1) mpo_set(esa, epb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow3(st.sa) * mpo_pow5(st.sb)) * rab4;
    MPO_S(st.isp, st.ips, 1) = w *
      (MPO_A(3) * MPO_B(1) - MPO_B(3) * MPO_A(1) + MPO_A(4) * MPO_B(2) - MPO_B(4) * MPO_A(2));
    return;
  }
  case 3: {
    // *** FIRST ROW - THIRD ROW OVERLAPS
    mpo_set(esa, esb, na, nb, rab, ii, st);
    rab4 = mpo_pow5(rab) * 0.0625;
    w = sqrt(mpo_pow3(st.sa) * mpo_pow7(st.sb) / 22.5) * rab4;
    MPO_S(1, 1, 1) = w * (MPO_A(5) * MPO_B(1) - MPO_B(5) * MPO_A(1) +
                          (MPO_A(4) * MPO_B(2) - MPO_B(4) * MPO_A(2)) * 2.0);
    if (na > 1) mpo_set(epa, esb, na, nb, rab, ii, st);
    if (nb > 1) mpo_set(esa, epb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow3(st.sa) * mpo_pow7(st.sb) / 7.5) * rab4;
    MPO_S(st.isp, st.ips, 1) = w *
      (MPO_A(4) * (MPO_B(1) + MPO_B(3)) - MPO_B(4) * (MPO_A(1) + MPO_A(3)) +
       MPO_B(2) * (MPO_A(3) + MPO_A(5)) - MPO_A(2) * (MPO_B(3) + MPO_B(5)));
    return;
  }
  case 4: {
    // *** SECOND ROW - SECOND ROW OVERLAPS
    mpo_set(esa, esb, na, nb, rab, ii, st);
    rab4 = mpo_pow5(rab) * 0.0625;
    w = sqrt(mpo_pow5(st.sa * st.sb)) * rab4;
    MPO_S(1, 1, 1) = w * (MPO_A(5) * MPO_B(1) + MPO_B(5) * MPO_A(1) - 2.0 * MPO_A(3) * MPO_B(3)) / 3.0;
    mpo_set(esa, epb, na, nb, rab, ii, st);
    if (na > nb) mpo_set(epa, esb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow5(st.sa * st.sb)) * rab4;
    rt3 = 1.0 / sqrt(3.0);
    d = MPO_A(4) * (MPO_B(1) - MPO_B(3)) - MPO_A(2) * (MPO_B(3) - MPO_B(5));
    e = MPO_B(4) * (MPO_A(1) - MPO_A(3)) - MPO_B(2) * (MPO_A(3) - MPO_A(5));
    MPO_S(st.isp, st.ips, 1) = w * rt3 * (d + e);
    mpo_set(epa, esb, na, nb, rab, ii, st);
    if (na > nb) mpo_set(esa, epb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow5(st.sa * st.sb)) * rab4;
    d = MPO_A(4) * (MPO_B(1) - MPO_B(3)) - MPO_A(2) * (MPO_B(3) - MPO_B(5));
    e = MPO_B(4) * (MPO_A(1) - MPO_A(3)) - MPO_B(2) * (MPO_A(3) - MPO_A(5));
    MPO_S(st.ips, st.isp, 1) = w * rt3 * (d - e);
    mpo_set(epa, epb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow5(st.sa * st.sb)) * rab4;
    MPO_S(2, 2, 1) = -w * (MPO_B(3) * (MPO_A(5) + MPO_A(1)) - MPO_A(3) * (MPO_B(5) + MPO_B(1)));
    MPO_S(2, 2, 2) = 0.5 * w * (MPO_A(5) * (MPO_B(1) - MPO_B(3)) - MPO_B(5) * (MPO_A(1) - MPO_A(3)) -
                                MPO_A(3) * MPO_B(1) + MPO_B(3) * MPO_A(1));
    return;
  }
  case 5: {
    // *** SECOND ROW - THIRD ROW OVERLAPS
    mpo_set(esa, esb, na, nb, rab, ii, st);
    rab6 = mpo_pow6(rab) * 0.03125 / sqrt(7.5);
    w = sqrt(mpo_pow5(st.sa) * mpo_pow7(st.sb)) * rab6;
    rt3 = 1.0 / sqrt(3.0);
    MPO_S(1, 1, 1) = w * (MPO_A(6) * MPO_B(1) + MPO_A(5) * MPO_B(2) -
                          2.0 * (MPO_A(4) * MPO_B(3) + MPO_A(3) * MPO_B(4)) +
                          MPO_A(2) * MPO_B(5) + MPO_A(1) * MPO_B(6)) / 3.0;
    mpo_set(esa, epb, na, nb, rab, ii, st);
    if (na > nb) mpo_set(epa, esb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow5(st.sa) * mpo_pow7(st.sb)) * rab6;
    MPO_S(st.isp, st.ips, 1) = w * rt3 * (MPO_A(6) * MPO_B(2) + MPO_A(5) * MPO_B(1) -
                                          2.0 * (MPO_A(4) * MPO_B(4) + MPO_A(3) * MPO_B(3)) +
                                          MPO_A(2) * MPO_B(6) + MPO_A(1) * MPO_B(5));
    mpo_set(epa, esb, na, nb, rab, ii, st);
    if (na > nb) mpo_set(esa, epb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow5(st.sa) * mpo_pow7(st.sb)) * rab6;
    MPO_S(st.ips, st.isp, 1) = -w * rt3 * (MPO_A(5) * (2.0 * MPO_B(3) - MPO_B(1)) -
                                           MPO_B(5) * (2.0 * MPO_A(3) - MPO_A(1)) -
                                           MPO_A(2) * (MPO_B(6) - 2.0 * MPO_B(4)) +
                                           MPO_B(2) * (MPO_A(6) - 2.0 * MPO_A(4)));
    mpo_set(epa, epb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow5(st.sa) * mpo_pow7(st.sb)) * rab6;
    MPO_S(2, 2, 1) = -w * (MPO_B(4) * (MPO_A(1) + MPO_A(5)) - MPO_A(4) * (MPO_B(1) + MPO_B(5)) +
                           MPO_B(3) * (MPO_A(2) + MPO_A(6)) - MPO_A(3) * (MPO_B(2) + MPO_B(6)));
    MPO_S(2, 2, 2) = 0.5 * w * (MPO_A(6) * (MPO_B(1) - MPO_B(3)) - MPO_B(6) * (MPO_A(1) - MPO_A(3)) +
                                MPO_A(5) * (MPO_B(2) - MPO_B(4)) - MPO_B(5) * (MPO_A(2) - MPO_A(4)) -
                                MPO_A(4) * MPO_B(1) + MPO_B(4) * MPO_A(1) -
                                MPO_A(3) * MPO_B(2) + MPO_B(3) * MPO_A(2));
    return;
  }
  case 6: {
    // *** THIRD ROW - THIRD ROW OVERLAPS
    mpo_set(esa, esb, na, nb, rab, ii, st);
    rab4 = mpo_pow7(rab) / 480.0;
    w = sqrt(mpo_pow7(st.sa * st.sb)) * rab4;
    rt3 = 1.0 / sqrt(3.0);
    MPO_S(1, 1, 1) = w * (MPO_A(7) * MPO_B(1) - 3.0 * (MPO_A(5) * MPO_B(3) - MPO_A(3) * MPO_B(5)) -
                          MPO_A(1) * MPO_B(7)) / 3.0;
    mpo_set(esa, epb, na, nb, rab, ii, st);
    if (na > nb) mpo_set(epa, esb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow7(st.sa * st.sb)) * rab4;
    d = MPO_A(6) * (MPO_B(1) - MPO_B(3)) - 2.0 * MPO_A(4) * (MPO_B(3) - MPO_B(5)) +
        MPO_A(2) * (MPO_B(5) - MPO_B(7));
    e = MPO_B(6) * (MPO_A(1) - MPO_A(3)) - 2.0 * MPO_B(4) * (MPO_A(3) - MPO_A(5)) +
        MPO_B(2) * (MPO_A(5) - MPO_A(7));
    MPO_S(st.isp, st.ips, 1) = w * rt3 * (d - e);
    mpo_set(epa, esb, na, nb, rab, ii, st);
    if (na > nb) mpo_set(esa, epb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow7(st.sa * st.sb)) * rab4;
    d = MPO_A(6) * (MPO_B(1) - MPO_B(3)) - 2.0 * MPO_A(4) * (MPO_B(3) - MPO_B(5)) +
        MPO_A(2) * (MPO_B(5) - MPO_B(7));
    e = MPO_B(6) * (MPO_A(1) - MPO_A(3)) - 2.0 * MPO_B(4) * (MPO_A(3) - MPO_A(5)) +
        MPO_B(2) * (MPO_A(5) - MPO_A(7));
    MPO_S(st.ips, st.isp, 1) = -w * rt3 * ((-d) - e);
    mpo_set(epa, epb, na, nb, rab, ii, st);
    w = sqrt(mpo_pow7(st.sa * st.sb)) * rab4;
    d = MPO_A(3) * (MPO_B(7) + MPO_B(3) + MPO_B(3)) - MPO_A(5) * (MPO_B(1) + MPO_B(5) + MPO_B(5)) -
        MPO_B(5) * MPO_A(1) + MPO_A(7) * MPO_B(3);
    MPO_S(2, 2, 1) = -w * d;
    d = MPO_A(7) * (MPO_B(1) - MPO_B(3)) + MPO_B(7) * (MPO_A(1) - MPO_A(3));
    e = MPO_A(5) * (MPO_B(5) - MPO_B(3) - MPO_B(1)) + MPO_B(5) * (MPO_A(5) - MPO_A(3) - MPO_A(1)) +
        2.0 * MPO_A(3) * MPO_B(3);
    MPO_S(2, 2, 2) = 0.5 * w * (d + e);
    return;
  }
  default: {
    // *** FIRST ROW - FIRST ROW OVERLAPS (ii == 1)
    mpo_set(esa, esb, na, nb, rab, ii, st);
    const double t = st.sa * st.sb * rab * rab;
    MPO_S(1, 1, 1) = 0.25 * sqrt(mpo_pow3(t)) * (MPO_A(3) * MPO_B(1) - MPO_B(3) * MPO_A(1));
    return;
  }
  }
}

#undef MPO_A
#undef MPO_B

// ---------------------------------------------------------------------------
// ss(): general Slater-orbital overlap (used by diat when either atom is
// outside the diat2 set: Z > 17, He or Ne).
// ---------------------------------------------------------------------------

// aff(la, m, i) constants from ss() (all other entries are zero)
MPO_DEV double mpo_aff(int la, int m, int i)
{
  if (i == 0) {
    if (la == 0 && m == 0) return 1.0;
    if (la == 1 && m == 0) return 1.0;
    if (la == 1 && m == 1) return sqrt(0.5);
    if (la == 2 && m == 0) return 1.5;
    if (la == 2 && m == 1) return sqrt(1.5);
    if (la == 2 && m == 2) return sqrt(0.375);
  } else if (i == 2) {
    if (la == 2 && m == 0) return -0.5;
  }
  return 0.0;
}

// bi(n, k) binomial coefficients (Pascal's triangle in ss(), exact integers,
// 0 <= k <= n <= 12), evaluated exactly with integer arithmetic.
MPO_DEV double mpo_binom(int n, int k)
{
  if (k < 0 || k > n) return 0.0;
  if (k > n - k) k = n - k;
  int r = 1;
  for (int i = 1; i <= k; ++i) {
    r = (r * (n - k + i)) / i;
  }
  return (double)r;
}

MPO_DEV double mpo_ss(int na, int nb, int la1, int lb1, int m1,
                      double ua, double ub, double r1, double a0)
{
  const int m = m1 - 1;
  const int lb = lb1 - 1;
  const int la = la1 - 1;
  const double r = r1 / a0;
  double af[20];
  double bf[20];
  const double p = (ua + ub) * r * 0.5;
  const double b = (ua - ub) * r * 0.5;
  const double quo = 1.0 / p;
  af[0] = quo * exp(-p);
  for (int n = 1; n <= 19; ++n) {
    af[n] = (double)n * quo * af[n - 1] + af[0];
  }
  // bfn fills bf(1..13) -> bf[0..12]; the Fortran leaves bf(14..20)
  // uninitialised (they are only read when na+nb > 12, i.e. two 7th-row
  // atoms).  Zero-fill them here.
  mpo_bfn(b, bf);
  for (int n = 13; n < 20; ++n) bf[n] = 0.0;
  double sum = 0.0;
  const int lam1 = la - m;
  const int lbm1 = lb - m;
  for (int i = 0; i <= lam1; i += 2) {
    const int ia = na + i - la;
    const int ic = la - i - m;
    for (int j = 0; j <= lbm1; j += 2) {
      const int ib = nb + j - lb;
      const int id = lb - j - m;
      double sum1 = 0.0;
      const int iab = ia + ib;
      for (int k1 = 0; k1 <= ia; ++k1) {
        for (int k2 = 0; k2 <= ib; ++k2) {
          for (int k3 = 0; k3 <= ic; ++k3) {
            for (int k4 = 0; k4 <= id; ++k4) {
              for (int k5 = 0; k5 <= m; ++k5) {
                const int iaf = iab - k1 - k2 + k3 + k4 + 2 * k5;
                for (int k6 = 0; k6 <= m; ++k6) {
                  const int ibf = k1 + k2 + k3 + k4 + 2 * k6;
                  sum1 = sum1 + mpo_binom(id, k4) * mpo_binom(ic, k3) * mpo_binom(ib, k2) *
                                mpo_binom(ia, k1) * mpo_binom(m, k5) * mpo_binom(m, k6) *
                                (double)(1 - 2 * ((m + k2 + k4 + k5 + k6) % 2)) * af[iaf] * bf[ibf];
                }
              }
            }
          }
        }
      }
      sum = sum + sum1 * mpo_aff(la, m, i) * mpo_aff(lb, m, j);
    }
  }
  return sum * mpo_powi(r, na + nb + 1) * mpo_powi(ua, na) * mpo_powi(ub, nb) / 2.0 *
         sqrt(ua * ub / (mpo_fact[na + na] * mpo_fact[nb + nb]) *
              (double)((la + la + 1) * (lb + lb + 1)));
}

// ---------------------------------------------------------------------------
// coe(x2, y2, z2, norbi, norbj, c, r): rotation coefficients.  c is the
// 75-element c(3,5,5) array; r receives sqrt(x2^2 + y2^2 + z2^2).
// Only the s/p block (nij <= 4) is implemented; the nij >= 5 (d) block of the
// Fortran is unreachable for natorb <= 4.
// ---------------------------------------------------------------------------
#define MPO_C(i, k, l) c[((i) - 1) + 3 * ((k) - 1) + 15 * ((l) - 1)]

MPO_DEV void mpo_coe(double x2, double y2, double z2, int norbi, int norbj,
                     double *c, double &r)
{
  double xy = x2 * x2 + y2 * y2;
  r = sqrt(xy + z2 * z2);
  xy = sqrt(xy);
  double ca, cb, sa, sb;
  if (xy >= 1.0e-10) {
    ca = x2 / xy;
    cb = z2 / r;
    sa = y2 / xy;
    sb = xy / r;
  } else if (z2 <= 0.0) {
    if (z2 != 0.0) {
      ca = -1.0;
      cb = -1.0;
      sa = 0.0;
      sb = 0.0;
    } else {
      ca = 0.0;
      cb = 0.0;
      sa = 0.0;
      sb = 0.0;
    }
  } else {
    ca = 1.0;
    cb = 1.0;
    sa = 0.0;
    sb = 0.0;
  }
  for (int i = 0; i < 75; ++i) c[i] = 0.0;
  const int nij = mpo_imax(norbi, norbj);
  MPO_C(1, 3, 3) = 1.0;          // c(37)
  if (nij >= 2) {
    MPO_C(2, 4, 4) = ca * cb;    // c(56)
    MPO_C(2, 4, 3) = ca * sb;    // c(41)
    MPO_C(2, 4, 2) = -sa;        // c(26)
    MPO_C(2, 3, 4) = -sb;        // c(53)
    MPO_C(2, 3, 3) = cb;         // c(38)
    MPO_C(2, 3, 2) = 0.0;        // c(23)
    MPO_C(2, 2, 4) = sa * cb;    // c(50)
    MPO_C(2, 2, 3) = sa * sb;    // c(35)
    MPO_C(2, 2, 2) = ca;         // c(20)
    // nij >= 5 (d orbital) block intentionally omitted (natorb <= 4 only).
  }
}

// ---------------------------------------------------------------------------
// diat(ni, nj, xj, di): diatomic overlap matrix, di is the 81-element di(9,9)
// column-major array.  xj is the position of atom nj relative to atom ni.
// ---------------------------------------------------------------------------
MPO_DEV bool mpo_use_diat2(int n, const MozymePairOverlapParams &prm)
{
  // diat: use_diat2(i) = natorb(i) < 5 for i = 1..17, except He and Ne.
  return (n <= 17) && (n != 2) && (n != 10) && (prm.natorb[n - 1] < 5);
}

#define MPO_NPQ(n, l) prm.npq[((n) - 1) + 107 * ((l) - 1)]
#define MPO_DI(i, j) di[((i) - 1) + 9 * ((j) - 1)]
#define MPO_IVAL(i, k) mpo_ival[((i) - 1) + 3 * ((k) - 1)]

MPO_DEV void mpo_diat(int ni, int nj, const double *xj, double *di,
                      const MozymePairOverlapParams &prm)
{
  const double x2 = xj[0];
  const double y2 = xj[1];
  const double z2 = xj[2];
  int pq1 = MPO_NPQ(ni, 1);
  int pq2 = MPO_NPQ(nj, 1);
  for (int i = 0; i < 81; ++i) di[i] = 0.0;
  double r = x2 * x2 + y2 * y2 + z2 * z2;   // squared distance here
  if (pq1 == 0 || pq2 == 0 || r >= prm.cutof1) return;
  const int natorbi = prm.natorb[ni - 1];
  const int natorbj = prm.natorb[nj - 1];
  if (natorbi == 0 || natorbj == 0) return;
  double c[75];
  mpo_coe(x2, y2, z2, natorbi, natorbj, c, r);   // r is now the distance
  if (r < 0.001) return;
  const int ia = mpo_imin(pq1 + 1, 3);
  int ib = mpo_imin(pq2 + 1, 3);
  const int a = ia - 1;
  const int b = ib - 1;
  double s[27];
  if (mpo_use_diat2(ni, prm) && mpo_use_diat2(nj, prm)) {
    mpo_diat2(ni, prm.zs[ni - 1], prm.zp[ni - 1], r,
              nj, prm.zs[nj - 1], prm.zp[nj - 1], s, prm.a0);
  } else {
    double ul1[3], ul2[3];
    ul1[0] = prm.zs[ni - 1];
    ul2[0] = prm.zs[nj - 1];
    ul1[1] = prm.zp[ni - 1];
    ul2[1] = prm.zp[nj - 1];
    ul1[2] = (prm.zd[ni - 1] > 0.3) ? prm.zd[ni - 1] : 0.3;   // max(zd(ni), 0.3)
    ul2[2] = (prm.zd[nj - 1] > 0.3) ? prm.zd[nj - 1] : 0.3;
    for (int i = 0; i < 27; ++i) s[i] = 0.0;
    const int newk = mpo_imin(a, b);
    const int nk1 = newk + 1;
    for (int i = 1; i <= ia; ++i) {
      const int iss = i;
      ib = b + 1;
      pq1 = MPO_NPQ(ni, i);
      for (int j = 1; j <= ib; ++j) {
        const int jss = j;
        pq2 = MPO_NPQ(nj, j);
        for (int k = 1; k <= nk1; ++k) {
          if (k > i || k > j) continue;
          const int kss = k;
          const int pi = mpo_imax(pq1, iss);
          const int pj = mpo_imax(pq2, jss);
          MPO_S(i, j, k) = mpo_ss(pi, pj, iss, jss, kss, ul1[i - 1], ul2[j - 1], r, prm.a0);
        }
      }
    }
  }
  for (int i = 1; i <= ia; ++i) {        // L = s, p, d for atom a
    const int kmin = 4 - i;
    const int kmax = 2 + i;
    for (int j = 1; j <= ib; ++j) {      // L = s, p, d for atom b
      double aa, bb;
      if (j == 2) {
        aa = -1.0;
        bb = 1.0;
      } else {
        aa = 1.0;
        if (j == 3) {
          bb = -1.0;
        } else {
          bb = 1.0;
        }
      }
      const int lmin = 4 - j;
      const int lmax = 2 + j;
      for (int k = kmin; k <= kmax; ++k) {
        for (int l = lmin; l <= lmax; ++l) {
          const int ii = MPO_IVAL(i, k);
          const int jj = MPO_IVAL(j, l);
          MPO_DI(ii, jj) = MPO_S(i, j, 1) * (MPO_C(i, k, 3) * MPO_C(j, l, 3)) * aa +
                           MPO_S(i, j, 2) * (MPO_C(i, k, 4) * MPO_C(j, l, 4) + MPO_C(i, k, 2) * MPO_C(j, l, 2)) * bb +
                           MPO_S(i, j, 3) * (MPO_C(i, k, 5) * MPO_C(j, l, 5) + MPO_C(i, k, 1) * MPO_C(j, l, 1));
        }
      }
    }
  }
}

#undef MPO_IVAL
#undef MPO_DI
#undef MPO_NPQ
#undef MPO_C
#undef MPO_S

// ---------------------------------------------------------------------------
// h1elec(ni, nj, xi, xj, smat) for s/p-only pairs.
//
// Returns false (smat untouched) if ni/nj are out of range or either atom has
// more than 4 orbitals; this test is made before the distance cutoffs so the
// result is independent of geometry.  Returns true otherwise, including the
// early exit that zeroes smat when the pair is beyond the cutoffs.
//
// smat is 9x9 column-major like the Fortran smat(9,9): smat[(i-1) + 9*(j-1)].
// Exactly as in h1elec, only the natorb(ni) x natorb(nj) block is scaled by
// (bi(i)+bj(j)); the remaining entries hold the raw diat() output (zero for
// diat2 pairs; for pairs handled by ss() the l=3 rows/columns 5..9 may carry
// unscaled overlap values, which callers never read).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool mozyme_pair_h1elec_sp_dev(
    int ni, int nj, const double *xi, const double *xj,
    const MozymePairOverlapParams &prm, double *smat)
{
  if (ni < 1 || ni > 107 || nj < 1 || nj > 107) return false;
  const int norbi = prm.natorb[ni - 1];
  const int norbj = prm.natorb[nj - 1];
  if (norbi > 4 || norbj > 4) return false;
  const double dx = xi[0] - xj[0];
  const double dy = xi[1] - xj[1];
  const double dz = xi[2] - xj[2];
  const double rab = dx * dx + dy * dy + dz * dz;
  if (rab > prm.cutofs || (rab > 3.24 && (ni == 102 || nj == 102))) {
    for (int i = 0; i < 81; ++i) smat[i] = 0.0;
    return true;
  }
  double xjuc[3];
  xjuc[0] = xj[0] - xi[0];
  xjuc[1] = xj[1] - xi[1];
  xjuc[2] = xj[2] - xi[2];
  mpo_diat(ni, nj, xjuc, smat, prm);
  double bi[9], bj[9];
  bi[0] = prm.betas[ni - 1] * 0.5;
  bi[1] = prm.betap[ni - 1] * 0.5;
  bi[2] = bi[1];
  bi[3] = bi[1];
  bi[4] = prm.betad[ni - 1] * 0.5;
  bi[5] = bi[4];
  bi[6] = bi[4];
  bi[7] = bi[4];
  bi[8] = bi[4];
  bj[0] = prm.betas[nj - 1] * 0.5;
  bj[1] = prm.betap[nj - 1] * 0.5;
  bj[2] = bj[1];
  bj[3] = bj[1];
  bj[4] = prm.betad[nj - 1] * 0.5;
  bj[5] = bj[4];
  bj[6] = bj[4];
  bj[7] = bj[4];
  bj[8] = bj[4];
  for (int j = 0; j < norbj; ++j) {
    for (int i = 0; i < norbi; ++i) {
      smat[i + 9 * j] = smat[i + 9 * j] * (bi[i] + bj[j]);
    }
  }
  return true;
}

#undef MPO_DEV

#endif // MOZYME_PAIR_OVERLAP_CUH

// Developed by Dr. Juvenal Yosa Reyes, UMCG Groningen, Universidad Simon Bolivar - Barranquilla - Colombia
#ifndef MOPAC_GPU_MOZYME_PAIR_CORE_CUH
#define MOPAC_GPU_MOZYME_PAIR_CORE_CUH
//
// Self-contained device port of MOPAC's rotate() (src/integrals/rotate.F90)
// for atom pairs whose basis sets contain only s and p orbitals
// (natorb <= 4).  For a pair (ni, xi) / (nj, xj) it reproduces the call path
//
//   rotate -> rotatd (src/integrals/mndod.F90)
//               -> rotmat   : rotation matrix p(3,3) and pp(6,3,3)
//               -> reppd    : 22 local-frame two-electron integrals ri + gab
//               -> spcore   : local-frame core integrals cored(1..4, 1..2)
//               -> to_point : l_feather smoothing (reppd, rotatd)
//               -> reppd2   : rep(1..34) = ri(ipos) (no d terms for sp)
//               -> tx + loop: rotation of the integrals to the molecular frame
//               -> PM7 "d" balance block (iod(ni)/iod(nj) > 0 parts only)
//               -> w2mat    : linear packing of W
//               -> ccrep    : core-core repulsion (src/integrals/ccrep.F90)
//          -> elenuc         : assemble e1b / e2a from cored and p / pp
//
// and returns W, e1b, e2a and enuc.  Periodic (id /= 0) terms and
// nddo_to_point are NOT reproduced (id is treated as 0).
//
// Everything is evaluated in double precision with the same operation order
// as the Fortran source (left-to-right evaluation of products / sums, the
// gfortran -O2/-O3 expansion of integer powers, single-precision literal
// 0.0003 promoted to double, ...).  Remaining differences are limited to
// libm (exp / sqrt / pow) last-bit behaviour and FMA contraction.
//
// Fortran -> C index conventions used throughout (all arrays column-major):
//   1-D element tables t(107)              -> t[ni - 1]
//   po(9,107)      po(k, ni)               -> po[(k - 1) + 9 * (ni - 1)]
//   ddp(6,107)     ddp(k, ni)              -> ddp[(k - 1) + 6 * (ni - 1)]
//   guess1/2/3(107,4) g(ni, ig)            -> g[(ni - 1) + 107 * (ig - 1)]
//   alpb/xfac(100,100) a(ni, nj)           -> a[(ni - 1) + 100 * (nj - 1)]
//   v_par(60)      parN                    -> v_par[N - 1]
//
// Only <cuda_runtime.h> and <math.h> are required.  The header also compiles
// as plain C++ (the CUDA qualifiers collapse to nothing) so the same code can
// be unit-tested on the host.
//
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

// ---------------------------------------------------------------------------
// Parameter block.  All pointers are device pointers (or host pointers when
// the header is compiled as plain C++).  The tables must be copied from the
// Fortran modules AFTER the model has been fully initialised, i.e. after
// switch, moldat/natorb setup, calpar (which calls inid) and fordd have run:
// inid rewrites am, po(1,:), po(2,:), po(3,:), po(7,:), po(9,:), ddp(2,:),
// ddp(3,:); calpar fills am, ad, aq, dd, qq.
// ---------------------------------------------------------------------------
struct MozymePairCoreParams {
  // ---- parameters_C: integer tables, dimension(107) --------------------
  const int *natorb;   // natorb(107): number of atomic orbitals on the element
  const int *iod;      // iod(107): d-shell occupancy, gates rotatd's PM7 block
  // ---- parameters_C: real tables, dimension(107) -----------------------
  const double *tore;  // tore(107): core charge
  const double *alp;   // alp(107): MNDO/AM1 core-core exponent (ccrep)
  const double *am;    // am(107): monopole additive term (reppd)
  const double *ad;    // ad(107): dipole additive term (reppd)
  const double *aq;    // aq(107): quadrupole additive term (reppd)
  const double *dd;    // dd(107): dipole charge separation (reppd)
  const double *qq;    // qq(107): quadrupole charge separation (reppd)
  // ---- parameters_C: 2-D real tables ----------------------------------
  const double *po;    // po(9,107): po(k,ni) -> po[(k-1) + 9*(ni-1)]   (spcore, reppd)
  const double *ddp;   // ddp(6,107): ddp(k,ni) -> ddp[(k-1) + 6*(ni-1)] (spcore)
  const double *guess1; // guess1(107,4): guess1(ni,ig) -> guess1[(ni-1) + 107*(ig-1)] (ccrep)
  const double *guess2; // guess2(107,4): same layout
  const double *guess3; // guess3(107,4): same layout
  const double *alpb;  // alpb(100,100): alpb(ni,nj) -> alpb[(ni-1) + 100*(nj-1)] (ccrep)
  const double *xfac;  // xfac(100,100): same layout (ccrep)
  const double *v_par; // v_par(60): par1..par60 -> v_par[0..59] (ccrep, ccrep_PM6_ORG)
  // ---- funcon_C ---------------------------------------------------------
  double a0;           // a0 = fpc(3), Bohr radius in Angstrom
  double ev;           // ev = fpc(4), Hartree in eV
  // ---- molkst_C ---------------------------------------------------------
  double trunc_1;      // to_point switching radius (Angstrom), used if l_feather
  double trunc_2;      // to_point exponent, used if l_feather
  int method_pm7;      // molkst_C method_PM7   (methods(14))
  int method_pm6;      // molkst_C method_PM6   (methods(6))
  int method_pm8;      // molkst_C method_PM8   (methods(19))
  int method_pm6_org;  // molkst_C method_pm6_org (methods(18))
  int method_am1;      // molkst_C method_AM1   (methods(2))
  int method_mndod;    // molkst_C method_MNDOD (methods(5))
  int l_feather;       // molkst_C l_feather (smooth NDDO -> point charge)
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static __device__ __forceinline__ double mpc_po(const MozymePairCoreParams &p, int k, int ni) {
  return p.po[(k - 1) + 9 * (ni - 1)];
}
static __device__ __forceinline__ double mpc_ddp(const MozymePairCoreParams &p, int k, int ni) {
  return p.ddp[(k - 1) + 6 * (ni - 1)];
}
static __device__ __forceinline__ double mpc_g1(const MozymePairCoreParams &p, int ni, int ig) {
  return p.guess1[(ni - 1) + 107 * (ig - 1)];
}
static __device__ __forceinline__ double mpc_g2(const MozymePairCoreParams &p, int ni, int ig) {
  return p.guess2[(ni - 1) + 107 * (ig - 1)];
}
static __device__ __forceinline__ double mpc_g3(const MozymePairCoreParams &p, int ni, int ig) {
  return p.guess3[(ni - 1) + 107 * (ig - 1)];
}
static __device__ __forceinline__ double mpc_alpb(const MozymePairCoreParams &p, int ni, int nj) {
  return p.alpb[(ni - 1) + 100 * (nj - 1)];
}
static __device__ __forceinline__ double mpc_xfac(const MozymePairCoreParams &p, int ni, int nj) {
  return p.xfac[(ni - 1) + 100 * (nj - 1)];
}
static __device__ __forceinline__ double mpc_par(const MozymePairCoreParams &p, int n) {
  return p.v_par[n - 1];
}

// Fortran x**2 -> x*x
static __device__ __forceinline__ double mpc_sq(double x) { return x * x; }
// gfortran (-O2/-O3) expansion of x**6: (x**3)*(x**3) with x**3 = (x*x)*x
static __device__ __forceinline__ double mpc_pow6(double x) {
  const double x3 = (x * x) * x;
  return x3 * x3;
}
// gfortran (-O2/-O3) expansion of x**12: (x**6)*(x**6)
static __device__ __forceinline__ double mpc_pow12(double x) {
  const double x6 = mpc_pow6(x);
  return x6 * x6;
}

// to_point (mndod.F90): smooth NDDO -> point-charge transition.
// r in Angstrom.  point = ev*a0/r, const = fraction of the NDDO term.
static __device__ __forceinline__ void mpc_to_point(const MozymePairCoreParams &p, double r,
                                                    double *point, double *cnst) {
  *point = p.ev * p.a0 / r;
  if (r < p.trunc_1) {
    const double t = r - p.trunc_1;
    *cnst = 1.0 - exp(-(t * t) * p.trunc_2);
  } else {
    *cnst = 0.0;
  }
}

// ---------------------------------------------------------------------------
// rotmat (mndod.F90): rotation matrix for the s/p block.
// Called by rotatd as rotmat(nj, ni, ci, cj, r): the dummy coordi = ci (= xi),
// coordj = cj (= xj), so x = xj - xi.  Returns r in Angstrom.
//   p[k][m]  = Fortran p(k+1, m+1): k = local p orbital (0 sigma, 1 pi, 2 pi*),
//              m = molecular axis (0 x, 1 y, 2 z).  Note sp = p in the module.
//   pp[k][l][c] = Fortran pp(c+1, k+1, l+1), k >= l, c = 0..5 for the
//              molecular pairs xx, yy, zz, xy, xz, yz.  Entries with k < l are
//              never written by rotmat and never read; they are left zero here.
// ---------------------------------------------------------------------------
struct MozymePairRot {
  double p[3][3];
  double pp[3][3][6];
};

static __device__ __forceinline__ double mpc_rotmat_sp(const double *xi, const double *xj,
                                                       MozymePairRot &rot) {
  const double small = 1.0e-7;
  const double x11 = xj[0] - xi[0];
  const double x22 = xj[1] - xi[1];
  const double x33 = xj[2] - xi[2];
  const double b = x11 * x11 + x22 * x22;
  const double r = sqrt(b + x33 * x33);
  const double sqb = sqrt(b);
  double sb = sqb / r;
  double ca, sa, cb;
  if (sb > small) {
    ca = x11 / sqb;
    sa = x22 / sqb;
    cb = x33 / r;
  } else {
    sa = 0.0;
    sb = 0.0;
    if (x33 < 0.0) {
      ca = -1.0;
      cb = -1.0;
    } else if (x33 > 0.0) {
      ca = 1.0;
      cb = 1.0;
    } else {
      ca = 0.0;
      cb = 0.0;
    }
  }
  double (*p)[3] = rot.p;
  p[0][0] = ca * sb;
  p[1][0] = ca * cb;
  p[2][0] = -sa;
  p[0][1] = sa * sb;
  p[1][1] = sa * cb;
  p[2][1] = ca;
  p[0][2] = cb;
  p[1][2] = -sb;
  p[2][2] = 0.0;
  for (int k = 0; k < 3; ++k) {
    for (int l = 0; l < 3; ++l) {
      for (int c = 0; c < 6; ++c) rot.pp[k][l][c] = 0.0;
    }
    rot.pp[k][k][0] = p[k][0] * p[k][0];
    rot.pp[k][k][1] = p[k][1] * p[k][1];
    rot.pp[k][k][2] = p[k][2] * p[k][2];
    rot.pp[k][k][3] = p[k][0] * p[k][1];
    rot.pp[k][k][4] = p[k][0] * p[k][2];
    rot.pp[k][k][5] = p[k][1] * p[k][2];
    for (int l = 0; l < k; ++l) {
      // pp(1,k,l) = 2.D0*p(k,1)*p(l,1)  -> (2*p(k,1))*p(l,1)
      rot.pp[k][l][0] = 2.0 * p[k][0] * p[l][0];
      rot.pp[k][l][1] = 2.0 * p[k][1] * p[l][1];
      rot.pp[k][l][2] = 2.0 * p[k][2] * p[l][2];
      rot.pp[k][l][3] = p[k][0] * p[l][1] + p[k][1] * p[l][0];
      rot.pp[k][l][4] = p[k][0] * p[l][2] + p[k][2] * p[l][0];
      rot.pp[k][l][5] = p[k][1] * p[l][2] + p[k][2] * p[l][1];
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// reppd (mndod.F90): the 22 local-frame two-electron integrals ri(22)
// (returned in ri[0..21] = Fortran ri(1..22)) and gab used by ccrep.
// rij in Angstrom.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void mpc_reppd(const MozymePairCoreParams &prm, int ni, int nj,
                                                 double rij, double *ri, double *gab) {
  const double td = 2.0;
  const double pp = 0.5;
  // nri(22) sign pattern applied at the end: ri = ri*nri
  const double ev = prm.ev;
  const double ev1 = ev / 2;
  const double ev2 = ev1 / 2;
  const double ev3 = ev2 / 2;
  const double ev4 = ev3 / 2;
  for (int i = 0; i < 22; ++i) ri[i] = 0.0;
  const double r = rij / prm.a0;
  const bool si = prm.natorb[ni - 1] >= 3;
  const bool sj = prm.natorb[nj - 1] >= 3;
  double arg[72], sqr[72];

  double aee = mpc_po(prm, 9, ni) + mpc_po(prm, 9, nj);
  aee = aee * aee;
  *gab = ev / sqrt(r * r + aee);  // core-core term only

  aee = pp / prm.am[ni - 1] + pp / prm.am[nj - 1];
  aee = aee * aee;

  if (!si && !sj) {
    // HYDROGEN - HYDROGEN (SS/SS)
    ri[0] = ev / sqrt(r * r + aee);
  } else if (si && !sj) {
    // HEAVY ATOM - HYDROGEN
    const double da = prm.dd[ni - 1];
    const double qa = prm.qq[ni - 1] * td;
    double ade = pp / prm.ad[ni - 1] + pp / prm.am[nj - 1];
    ade = ade * ade;
    double aqe = pp / prm.aq[ni - 1] + pp / prm.am[nj - 1];
    aqe = aqe * aqe;
    const double rsq = r * r;
    double xxx;
    arg[0] = rsq + aee;
    xxx = r + da;
    arg[1] = xxx * xxx + ade;
    xxx = r - da;
    arg[2] = xxx * xxx + ade;
    xxx = r + qa;
    arg[3] = xxx * xxx + aqe;
    xxx = r - qa;
    arg[4] = xxx * xxx + aqe;
    arg[5] = rsq + aqe;
    arg[6] = arg[5] + qa * qa;
    for (int i = 0; i < 7; ++i) sqr[i] = sqrt(arg[i]);
    const double ee = ev / sqr[0];
    ri[0] = ee;
    ri[1] = ev1 / sqr[1] - ev1 / sqr[2];
    ri[2] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
    ri[3] = ee + ev1 / sqr[6] - ev1 / sqr[5];
  } else if (!si && sj) {
    // HYDROGEN - HEAVY ATOM
    const double db = prm.dd[nj - 1];
    const double qb = prm.qq[nj - 1] * td;
    double aed = pp / prm.am[ni - 1] + pp / prm.ad[nj - 1];
    aed = aed * aed;
    double aeq = pp / prm.am[ni - 1] + pp / prm.aq[nj - 1];
    aeq = aeq * aeq;
    const double rsq = r * r;
    double xxx;
    arg[0] = rsq + aee;
    xxx = r - db;
    arg[1] = xxx * xxx + aed;
    xxx = r + db;
    arg[2] = xxx * xxx + aed;
    xxx = r - qb;
    arg[3] = xxx * xxx + aeq;
    xxx = r + qb;
    arg[4] = xxx * xxx + aeq;
    arg[5] = rsq + aeq;
    arg[6] = arg[5] + qb * qb;
    for (int i = 0; i < 7; ++i) sqr[i] = sqrt(arg[i]);
    const double ee = ev / sqr[0];
    ri[0] = ee;
    ri[4] = ev1 / sqr[1] - ev1 / sqr[2];
    ri[10] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
    ri[11] = ee + ev1 / sqr[6] - ev1 / sqr[5];
  } else {
    // HEAVY ATOM - HEAVY ATOM
    const double da = prm.dd[ni - 1];
    const double db = prm.dd[nj - 1];
    double qa = prm.qq[ni - 1] * td;
    double qb = prm.qq[nj - 1] * td;

    double ade = pp / prm.ad[ni - 1] + pp / prm.am[nj - 1];
    ade = ade * ade;
    double aqe = pp / prm.aq[ni - 1] + pp / prm.am[nj - 1];
    aqe = aqe * aqe;
    double aed = pp / prm.am[ni - 1] + pp / prm.ad[nj - 1];
    aed = aed * aed;
    double aeq = pp / prm.am[ni - 1] + pp / prm.aq[nj - 1];
    aeq = aeq * aeq;
    double axx = pp / prm.ad[ni - 1] + pp / prm.ad[nj - 1];
    axx = axx * axx;
    double adq = pp / prm.ad[ni - 1] + pp / prm.aq[nj - 1];
    adq = adq * adq;
    double aqd = pp / prm.aq[ni - 1] + pp / prm.ad[nj - 1];
    aqd = aqd * aqd;
    double aqq = pp / prm.aq[ni - 1] + pp / prm.aq[nj - 1];
    aqq = aqq * aqq;
    const double rsq = r * r;
    double xxx, yyy, zzz, www;
    // arg(n) -> arg[n-1]
    arg[0] = rsq + aee;
    xxx = r + da;
    arg[1] = xxx * xxx + ade;
    xxx = r - da;
    arg[2] = xxx * xxx + ade;
    xxx = r - qa;
    arg[3] = xxx * xxx + aqe;
    xxx = r + qa;
    arg[4] = xxx * xxx + aqe;
    arg[5] = rsq + aqe;
    arg[6] = arg[5] + qa * qa;
    xxx = r - db;
    arg[7] = xxx * xxx + aed;
    xxx = r + db;
    arg[8] = xxx * xxx + aed;
    xxx = r - qb;
    arg[9] = xxx * xxx + aeq;
    xxx = r + qb;
    arg[10] = xxx * xxx + aeq;
    arg[11] = rsq + aeq;
    arg[12] = arg[11] + qb * qb;
    xxx = da - db;
    arg[13] = rsq + axx + xxx * xxx;
    xxx = da + db;
    arg[14] = rsq + axx + xxx * xxx;
    xxx = r + da - db;
    arg[15] = xxx * xxx + axx;
    xxx = r - da + db;
    arg[16] = xxx * xxx + axx;
    xxx = r - da - db;
    arg[17] = xxx * xxx + axx;
    xxx = r + da + db;
    arg[18] = xxx * xxx + axx;
    xxx = r + da;
    arg[19] = xxx * xxx + adq;
    arg[20] = arg[19] + qb * qb;
    xxx = r - da;
    arg[21] = xxx * xxx + adq;
    arg[22] = arg[21] + qb * qb;
    xxx = r - db;
    arg[23] = xxx * xxx + aqd;
    arg[24] = arg[23] + qa * qa;
    xxx = r + db;
    arg[25] = xxx * xxx + aqd;
    arg[26] = arg[25] + qa * qa;
    xxx = r + da - qb;
    arg[27] = xxx * xxx + adq;
    xxx = r - da - qb;
    arg[28] = xxx * xxx + adq;
    xxx = r + da + qb;
    arg[29] = xxx * xxx + adq;
    xxx = r - da + qb;
    arg[30] = xxx * xxx + adq;
    xxx = r + qa - db;
    arg[31] = xxx * xxx + aqd;
    xxx = r + qa + db;
    arg[32] = xxx * xxx + aqd;
    xxx = r - qa - db;
    arg[33] = xxx * xxx + aqd;
    xxx = r - qa + db;
    arg[34] = xxx * xxx + aqd;
    arg[35] = rsq + aqq;
    xxx = qa - qb;
    arg[36] = arg[35] + xxx * xxx;
    xxx = qa + qb;
    arg[37] = arg[35] + xxx * xxx;
    arg[38] = arg[35] + qa * qa;
    arg[39] = arg[35] + qb * qb;
    arg[40] = arg[38] + qb * qb;
    xxx = r - qb;
    arg[41] = xxx * xxx + aqq;
    arg[42] = arg[41] + qa * qa;
    xxx = r + qb;
    arg[43] = xxx * xxx + aqq;
    arg[44] = arg[43] + qa * qa;
    xxx = r + qa;
    arg[45] = xxx * xxx + aqq;
    arg[46] = arg[45] + qb * qb;
    xxx = r - qa;
    arg[47] = xxx * xxx + aqq;
    arg[48] = arg[47] + qb * qb;
    xxx = r + qa - qb;
    arg[49] = xxx * xxx + aqq;
    xxx = r + qa + qb;
    arg[50] = xxx * xxx + aqq;
    xxx = r - qa - qb;
    arg[51] = xxx * xxx + aqq;
    xxx = r - qa + qb;
    arg[52] = xxx * xxx + aqq;
    qa = prm.qq[ni - 1];
    qb = prm.qq[nj - 1];
    xxx = da - qb;
    xxx = xxx * xxx;
    yyy = r - qb;
    yyy = yyy * yyy;
    zzz = da + qb;
    zzz = zzz * zzz;
    www = r + qb;
    www = www * www;
    arg[53] = xxx + yyy + adq;
    arg[54] = xxx + www + adq;
    arg[55] = zzz + yyy + adq;
    arg[56] = zzz + www + adq;
    xxx = qa - db;
    xxx = xxx * xxx;
    yyy = qa + db;
    yyy = yyy * yyy;
    zzz = r + qa;
    zzz = zzz * zzz;
    www = r - qa;
    www = www * www;
    arg[57] = zzz + xxx + aqd;
    arg[58] = www + xxx + aqd;
    arg[59] = zzz + yyy + aqd;
    arg[60] = www + yyy + aqd;
    xxx = qa - qb;
    xxx = xxx * xxx;
    arg[61] = arg[35] + td * xxx;
    yyy = qa + qb;
    yyy = yyy * yyy;
    arg[62] = arg[35] + td * yyy;
    arg[63] = arg[35] + td * (qa * qa + qb * qb);
    zzz = r + qa - qb;
    zzz = zzz * zzz;
    arg[64] = zzz + xxx + aqq;
    arg[65] = zzz + yyy + aqq;
    zzz = r + qa + qb;
    zzz = zzz * zzz;
    arg[66] = zzz + xxx + aqq;
    arg[67] = zzz + yyy + aqq;
    zzz = r - qa - qb;
    zzz = zzz * zzz;
    arg[68] = zzz + xxx + aqq;
    arg[69] = zzz + yyy + aqq;
    zzz = r - qa + qb;
    zzz = zzz * zzz;
    arg[70] = zzz + xxx + aqq;
    arg[71] = zzz + yyy + aqq;
    for (int i = 0; i < 72; ++i) sqr[i] = sqrt(arg[i]);
    // sqr(n) -> sqr[n-1]
    const double ee = ev / sqr[0];
    const double dze = (-ev1 / sqr[1]) + ev1 / sqr[2];
    const double qzze = ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
    const double qxxe = ev1 / sqr[6] - ev1 / sqr[5];
    const double edz = (-ev1 / sqr[7]) + ev1 / sqr[8];
    const double eqzz = ev2 / sqr[9] + ev2 / sqr[10] - ev1 / sqr[11];
    const double eqxx = ev1 / sqr[12] - ev1 / sqr[11];
    const double dxdx = ev1 / sqr[13] - ev1 / sqr[14];
    const double dzdz = ev2 / sqr[15] + ev2 / sqr[16] - ev2 / sqr[17] - ev2 / sqr[18];
    const double dzqxx = ev2 / sqr[19] - ev2 / sqr[20] - ev2 / sqr[21] + ev2 / sqr[22];
    const double qxxdz = ev2 / sqr[23] - ev2 / sqr[24] - ev2 / sqr[25] + ev2 / sqr[26];
    const double dzqzz = (-ev3 / sqr[27]) + ev3 / sqr[28] - ev3 / sqr[29] + ev3 / sqr[30] -
                         ev2 / sqr[21] + ev2 / sqr[19];
    const double qzzdz = (-ev3 / sqr[31]) + ev3 / sqr[32] - ev3 / sqr[33] + ev3 / sqr[34] +
                         ev2 / sqr[23] - ev2 / sqr[25];
    const double qxxqxx = ev3 / sqr[36] + ev3 / sqr[37] - ev2 / sqr[38] - ev2 / sqr[39] +
                          ev2 / sqr[35];
    const double qxxqyy = ev2 / sqr[40] - ev2 / sqr[38] - ev2 / sqr[39] + ev2 / sqr[35];
    const double qxxqzz = ev3 / sqr[42] + ev3 / sqr[44] - ev3 / sqr[41] - ev3 / sqr[43] -
                          ev2 / sqr[38] + ev2 / sqr[35];
    const double qzzqxx = ev3 / sqr[46] + ev3 / sqr[48] - ev3 / sqr[45] - ev3 / sqr[47] -
                          ev2 / sqr[39] + ev2 / sqr[35];
    const double qzzqzz = ev4 / sqr[49] + ev4 / sqr[50] + ev4 / sqr[51] + ev4 / sqr[52] -
                          ev3 / sqr[47] - ev3 / sqr[45] - ev3 / sqr[41] - ev3 / sqr[43] +
                          ev2 / sqr[35];
    const double dxqxz = (-ev2 / sqr[53]) + ev2 / sqr[54] + ev2 / sqr[55] - ev2 / sqr[56];
    const double qxzdx = (-ev2 / sqr[57]) + ev2 / sqr[58] + ev2 / sqr[59] - ev2 / sqr[60];
    const double qxzqxz = ev3 / sqr[64] - ev3 / sqr[66] - ev3 / sqr[68] + ev3 / sqr[70] -
                          ev3 / sqr[65] + ev3 / sqr[67] + ev3 / sqr[69] - ev3 / sqr[71];
    ri[0] = ee;
    ri[1] = -dze;
    ri[2] = ee + qzze;
    ri[3] = ee + qxxe;
    ri[4] = -edz;
    ri[5] = dzdz;
    ri[6] = dxdx;
    ri[7] = (-edz) - qzzdz;
    ri[8] = (-edz) - qxxdz;
    ri[9] = -qxzdx;
    ri[10] = ee + eqzz;
    ri[11] = ee + eqxx;
    ri[12] = (-dze) - dzqzz;
    ri[13] = (-dze) - dzqxx;
    ri[14] = -dxqxz;
    ri[15] = ee + eqzz + qzze + qzzqzz;
    ri[16] = ee + eqzz + qxxe + qxxqzz;
    ri[17] = ee + eqxx + qzze + qzzqxx;
    ri[18] = ee + eqxx + qxxe + qxxqxx;
    ri[19] = qxzqxz;
    ri[20] = ee + eqxx + qxxe + qxxqyy;
    ri[21] = pp * (qxxqxx - qxxqyy);
  }

  if (prm.l_feather) {
    double point, cnst;
    mpc_to_point(prm, rij, &point, &cnst);
    // integrals that tend to a point charge get the (1-const)*point tail
    const double tail = (1.0 - cnst) * point;
    ri[0] = ri[0] * cnst + tail;
    ri[1] = ri[1] * cnst;
    ri[2] = ri[2] * cnst + tail;
    ri[3] = ri[3] * cnst + tail;
    ri[4] = ri[4] * cnst;
    ri[5] = ri[5] * cnst;
    ri[6] = ri[6] * cnst;
    ri[7] = ri[7] * cnst;
    ri[8] = ri[8] * cnst;
    ri[9] = ri[9] * cnst;
    ri[10] = ri[10] * cnst + tail;
    ri[11] = ri[11] * cnst + tail;
    ri[12] = ri[12] * cnst;
    ri[13] = ri[13] * cnst;
    ri[14] = ri[14] * cnst;
    ri[15] = ri[15] * cnst + tail;
    ri[16] = ri[16] * cnst + tail;
    ri[17] = ri[17] * cnst + tail;
    ri[18] = ri[18] * cnst + tail;
    ri[19] = ri[19] * cnst;
    ri[20] = ri[20] * cnst + tail;
    ri[21] = ri[21] * cnst;
    *gab = *gab * cnst + tail;
  }
  // ri = ri*nri, nri = 1,-1,1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1
  ri[1] = -ri[1];
  ri[4] = -ri[4];
  ri[7] = -ri[7];
  ri[8] = -ri[8];
  ri[9] = -ri[9];
  ri[12] = -ri[12];
  ri[13] = -ri[13];
  ri[14] = -ri[14];
}

// ---------------------------------------------------------------------------
// spcore (mndod.F90): local-frame electron-core attraction integrals.
// r in Bohr.  core1[c] = Fortran core(c+1, 1) (electrons on ni, core of nj),
// core2[c] = core(c+1, 2).  Only c = 0..3 ((SS/), (SO/), (OO/), (PP/)) are
// non-zero for s/p atoms.  Branches on the atomic number (ni >= 3), exactly
// like the Fortran.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void mpc_spcore(const MozymePairCoreParams &prm, int ni, int nj,
                                                  double r, double *core1, double *core2) {
  const double pxy[7] = {1.0, -0.5, -0.5, 0.5, 0.25, 0.25, 0.5};
  const double ev = prm.ev;
  for (int i = 0; i < 4; ++i) {
    core1[i] = 0.0;
    core2[i] = 0.0;
  }
  const double r2 = r * r;
  const double aci = mpc_po(prm, 9, ni);
  const double acj = mpc_po(prm, 9, nj);
  const double ssi = mpc_sq(aci + mpc_po(prm, 1, nj));
  const double ssj = mpc_sq(acj + mpc_po(prm, 1, ni));
  core1[0] = -prm.tore[nj - 1] * ev / sqrt(r2 + ssj);
  core2[0] = -prm.tore[ni - 1] * ev / sqrt(r2 + ssi);
  if (ni >= 3 || nj >= 3) {
    if (ni >= 3) {
      double xj[7];
      const double ppj = mpc_sq(acj + mpc_po(prm, 7, ni));
      const double da = mpc_ddp(prm, 2, ni);
      const double qa = mpc_ddp(prm, 3, ni) / sqrt(2.0);
      const double twoqa = qa + qa;
      const double adj = mpc_sq(mpc_po(prm, 2, ni) + acj);
      const double aqj = mpc_sq(mpc_po(prm, 3, ni) + acj);
      xj[0] = r2 + ppj;
      xj[1] = r2 + aqj;
      xj[2] = mpc_sq(r + da) + adj;
      xj[3] = mpc_sq(r - da) + adj;
      xj[4] = mpc_sq(r - twoqa) + aqj;
      xj[5] = mpc_sq(r + twoqa) + aqj;
      xj[6] = r2 + twoqa * twoqa + aqj;
      for (int i = 0; i < 7; ++i) xj[i] = pxy[i] / sqrt(xj[i]);
      const double aj2 = (xj[2] + xj[3]) * ev;
      const double aj3 = (xj[0] + xj[1] + xj[4] + xj[5]) * ev;
      const double aj4 = (xj[0] + xj[1] + xj[6]) * ev;
      core1[1] = -prm.tore[nj - 1] * aj2;
      core1[2] = -prm.tore[nj - 1] * aj3;
      core1[3] = -prm.tore[nj - 1] * aj4;
    }
    if (nj >= 3) {
      double xi[7];
      const double ppi = mpc_sq(aci + mpc_po(prm, 7, nj));
      const double db = mpc_ddp(prm, 2, nj);
      const double qb = mpc_ddp(prm, 3, nj) / sqrt(2.0);
      const double adi = mpc_sq(mpc_po(prm, 2, nj) + aci);
      const double aqi = mpc_sq(mpc_po(prm, 3, nj) + aci);
      const double twoqb = qb + qb;
      xi[0] = r2 + ppi;
      xi[1] = r2 + aqi;
      xi[2] = mpc_sq(r + db) + adi;
      xi[3] = mpc_sq(r - db) + adi;
      xi[4] = mpc_sq(r - twoqb) + aqi;
      xi[5] = mpc_sq(r + twoqb) + aqi;
      xi[6] = r2 + twoqb * twoqb + aqi;
      for (int i = 0; i < 7; ++i) xi[i] = pxy[i] / sqrt(xi[i]);
      const double ai2 = -(xi[2] + xi[3]) * ev;
      const double ai3 = (xi[0] + xi[1] + xi[4] + xi[5]) * ev;
      const double ai4 = (xi[0] + xi[1] + xi[6]) * ev;
      core2[1] = -prm.tore[ni - 1] * ai2;
      core2[2] = -prm.tore[ni - 1] * ai3;
      core2[3] = -prm.tore[ni - 1] * ai4;
    }
  }
}

// ---------------------------------------------------------------------------
// ccrep_PM6_ORG (ccrep.F90): core-core scaling factor for PM6-ORG.
// r in Angstrom.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ double mpc_ccrep_pm6_org(const MozymePairCoreParams &prm, int ni,
                                                           int nj, double r, double fff,
                                                           double abond) {
  const double sum = 0.01;  // "save" local in the Fortran, never modified
  // 0.0003 is a single-precision literal in the Fortran source
  const double c0003 = (double)0.0003f;
  double scale = 1.0 + 2.0 * fff * exp(-abond * (r + c0003 * mpc_pow6(r)));
  const int i = (ni > nj) ? ni : nj;
  const int j = (ni < nj) ? ni : nj;
  const double gauss_r2 = exp(-abond * mpc_sq(r));  // Exp(-abond*r**2)
  switch (j) {
    case 1:
      switch (i) {
        case 1:
          if (r - mpc_par(prm, 18) > 0.0)
            scale = scale + sum * mpc_par(prm, 16) * exp(-mpc_par(prm, 17) * mpc_sq(r - mpc_par(prm, 18)));
          else
            scale = scale + sum * mpc_par(prm, 16);
          break;
        case 6:
          if (r - mpc_par(prm, 12) > 0.0)
            scale = 1.0 + 2.0 * fff * gauss_r2 +
                    sum * mpc_par(prm, 19) * exp(-mpc_par(prm, 11) * mpc_sq(r - mpc_par(prm, 12)));
          else
            scale = 1.0 + 2.0 * fff * gauss_r2 + sum * mpc_par(prm, 19);
          break;
        case 7:
          if (r - mpc_par(prm, 40) > 0.0)
            scale = 1.0 + 2.0 * fff * gauss_r2 +
                    sum * mpc_par(prm, 38) * exp(-mpc_par(prm, 39) * mpc_sq(r - mpc_par(prm, 40)));
          else
            scale = 1.0 + 2.0 * fff * gauss_r2 + sum * mpc_par(prm, 38);
          break;
        case 8:
          if (r - mpc_par(prm, 5) > 0.0)
            scale = 1.0 + 2.0 * fff * gauss_r2 +
                    sum * mpc_par(prm, 3) * exp(-mpc_par(prm, 4) * mpc_sq(r - mpc_par(prm, 5)));
          else
            scale = 1.0 + 2.0 * fff * gauss_r2 + sum * mpc_par(prm, 3);
          break;
        case 16:
          if (r - mpc_par(prm, 46) > 0.0)
            scale = scale + sum * mpc_par(prm, 44) * exp(-mpc_par(prm, 45) * mpc_sq(r - mpc_par(prm, 46)));
          else
            scale = scale + sum * mpc_par(prm, 44);
          break;
        default:
          break;
      }
      break;
    case 6:
      switch (i) {
        case 6:
          scale = scale + mpc_par(prm, 1) * exp(-mpc_par(prm, 2) * r);
          if (r - mpc_par(prm, 15) > 0.0)
            scale = scale + sum * mpc_par(prm, 13) * exp(-mpc_par(prm, 14) * mpc_sq(r - mpc_par(prm, 15)));
          else
            scale = scale + sum * mpc_par(prm, 13);
          break;
        case 7:
          if (r - mpc_par(prm, 37) > 0.0)
            scale = scale + sum * mpc_par(prm, 35) * exp(-mpc_par(prm, 36) * mpc_sq(r - mpc_par(prm, 37)));
          else
            scale = scale + sum * mpc_par(prm, 35);
          break;
        case 8:
          if (r - mpc_par(prm, 22) > 0.0)
            scale = scale + sum * mpc_par(prm, 20) * exp(-mpc_par(prm, 21) * mpc_sq(r - mpc_par(prm, 22)));
          else
            scale = scale + sum * mpc_par(prm, 20);
          break;
        case 16:
          if (r - mpc_par(prm, 31) > 0.0)
            scale = scale + sum * mpc_par(prm, 29) * exp(-mpc_par(prm, 30) * mpc_sq(r - mpc_par(prm, 31)));
          else
            scale = scale + sum * mpc_par(prm, 29);
          break;
        default:
          break;
      }
      break;
    case 7:
      switch (i) {
        case 8:
          if (r - mpc_par(prm, 28) > 0.0)
            scale = scale + sum * mpc_par(prm, 26) * exp(-mpc_par(prm, 27) * mpc_sq(r - mpc_par(prm, 28)));
          else
            scale = scale + sum * mpc_par(prm, 26);
          break;
        case 16:
          if (r - mpc_par(prm, 43) > 0.0)
            scale = scale + sum * mpc_par(prm, 41) * exp(-mpc_par(prm, 42) * mpc_sq(r - mpc_par(prm, 43)));
          else
            scale = scale + sum * mpc_par(prm, 41);
          break;
        default:
          break;
      }
      break;
    case 8:
      switch (i) {
        case 8:
          if (r - mpc_par(prm, 34) > 0.0)
            scale = scale + sum * mpc_par(prm, 32) * exp(-mpc_par(prm, 33) * mpc_sq(r - mpc_par(prm, 34)));
          else
            scale = scale + sum * mpc_par(prm, 32);
          break;
        case 14:
          scale = scale - 0.7e-3 * exp(-mpc_sq(r - 2.9));
          break;
        case 16:
          if (r - mpc_par(prm, 25) > 0.0)
            scale = scale + sum * mpc_par(prm, 23) * exp(-mpc_par(prm, 24) * mpc_sq(r - mpc_par(prm, 25)));
          else
            scale = scale + sum * mpc_par(prm, 23);
          break;
        default:
          break;
      }
      break;
    default:
      break;
  }
  return scale;
}

// ---------------------------------------------------------------------------
// ccrep (ccrep.F90): core-core repulsion.  r_bohr is the distance in Bohr as
// passed by rotatd; it is converted back to Angstrom inside (r = r*a0), which
// is what the Fortran does.  gab is the <ss|ss>-like term from reppd.
// The PM7 "poor guess" fallback that writes into xfac/alpb on the host is
// computed locally without touching the parameter tables.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ double mpc_ccrep(const MozymePairCoreParams &prm, int ni, int nj,
                                                   double r_bohr, double gab) {
  const bool pm6 = prm.method_pm6 != 0;
  const bool pm7 = prm.method_pm7 != 0;
  const bool pm8 = prm.method_pm8 != 0;
  const bool pm6_org = prm.method_pm6_org != 0;
  const bool am1 = prm.method_am1 != 0;
  const bool mndod = prm.method_mndod != 0;
  const double c0003 = (double)0.0003f;  // single-precision literal in Fortran

  const double r = r_bohr * prm.a0;
  const double alpni = prm.alp[ni - 1];
  const double alpnj = prm.alp[nj - 1];
  const double tni = prm.tore[ni - 1];
  const double tnj = prm.tore[nj - 1];
  const double enuc = tni * tnj * gab;
  double enuclr;
  double fff;
  double abond;
  double scale;
  bool use_guess = false;
  double abond_guess = 0.0;

  if (ni < 101 && nj < 101)
    fff = mpc_xfac(prm, ni, nj);
  else
    fff = 0.0;
  if (pm7) {
    if (fabs(fff) < 1.0e-5) {
      if (ni < 99 && nj < 99) {
        // poor guess for alpb(ni,nj) / xfac(ni,nj): host writes these back into
        // the tables; here they are kept local.
        fff = 0.5 * (mpc_xfac(prm, ni, ni) + mpc_xfac(prm, nj, nj));
        abond_guess = 0.5 * (mpc_alpb(prm, ni, ni) + mpc_alpb(prm, nj, nj));
        use_guess = true;
      } else {
        fff = 0.0;  // unreal atoms
      }
    }
  }
  if (fabs(fff) > 1.0e-5) {
    // Bond parameters defined
    abond = use_guess ? abond_guess : mpc_alpb(prm, ni, nj);
    if (abond < 1.0e-6) abond = 1.2;
    if (pm6_org) {
      scale = mpc_ccrep_pm6_org(prm, ni, nj, r, fff, abond);
    } else if (!mndod) {
      if (pm6 || pm7 || pm8) {
        scale = 1.0 + 2.0 * fff * exp(-abond * (r + c0003 * mpc_pow6(r)));
        const int i = (ni > nj) ? ni : nj;
        const int j = (ni < nj) ? ni : nj;
        switch (j) {
          case 1:
            switch (i) {
              case 1:
                break;
              case 6:
              case 7:
                scale = 1.0 + 2.0 * fff * exp(-abond * mpc_sq(r));
                break;
              case 8:
                // Slow O - H term; par3*exp(-par4*r*2) used by PM7
                scale = 1.0 + 2.0 * fff * exp(-abond * mpc_sq(r)) -
                        mpc_par(prm, 3) * exp(-mpc_par(prm, 4) * r * 2);
                break;
              default:
                break;
            }
            break;
          case 6:
            if (i == 6) scale = scale + mpc_par(prm, 1) * exp(-mpc_par(prm, 2) * r);  // C-C triple bond
            break;
          case 7:
            break;
          case 8:
            if (i == 14) scale = scale - 0.7e-3 * exp(-mpc_sq(r - 2.9));  // Si-O long range
            break;
          default:
            break;
        }
      } else {  // Not PM6
        if (am1 && ((ni == 42 && nj == 1) || (ni == 1 && nj == 42))) {
          // AM1-d Mo-H interaction
          scale = 1.0 + r * 2.0 * fff * exp(-abond * r);
        } else {
          scale = 1.0 + 2.0 * fff * exp(-abond * r);
        }
      }
    } else {
      // MNDO/d interactions
      if (ni == nj) {
        scale = 1.0 + 2.0 * exp(-abond * r);
      } else {
        if (nj == 11 || nj == 12 || nj == 13)
          scale = 1.0 + exp(-abond * r) + exp(-prm.alp[ni - 1] * r);
        else
          scale = 1.0 + exp(-abond * r) + exp(-prm.alp[nj - 1] * r);
      }
    }
    enuclr = enuc * scale;
  } else {
    abond = 0.0;
    double eni, enj;
    if (pm6 || pm7 || pm8 || pm6_org) {
      if ((ni > 56 && ni < 72) || (nj > 56 && nj < 72))
        scale = 10.0 * exp(-3.0 * r);
      else
        scale = 10.0 * exp(-2.18 * r);  // generic core-core term
      eni = 0.0;
      enj = 0.0;
    } else {
      eni = exp(-alpni * r);
      enj = exp(-alpnj * r);
      scale = eni + enj;
    }
    // "almost certainly dead code" in the Fortran, kept verbatim
    const int nt = ni + nj;
    if (nt == 8 || nt == 9) {
      if (ni == 7 || ni == 8) scale = scale + (r - 1.0) * eni;
      if (nj == 7 || nj == 8) scale = scale + (r - 1.0) * enj;
    }
    enuclr = fabs(scale * enuc) + enuc;
  }
  scale = 0.0;
  int ngauss;
  if (pm6 || pm7 || pm8 || pm6_org) {
    // VdW term
    double ax = mpc_g2(prm, ni, 1) * mpc_sq(r - mpc_g3(prm, ni, 1));
    if (ax < 25.0) scale = scale + tni * tnj / r * mpc_g1(prm, ni, 1) * exp(-ax);
    ax = mpc_g2(prm, nj, 1) * mpc_sq(r - mpc_g3(prm, nj, 1));
    if (ax < 25.0) scale = scale + tni * tnj / r * mpc_g1(prm, nj, 1) * exp(-ax);
    ngauss = (abond > 1.0e-4) ? 0 : 4;
  } else if (am1 && (ni == 5 || nj == 5) &&
             (ni == 1 || nj == 1 || ni == 6 || nj == 6 || ni == 9 || nj == 9 || ni == 17 ||
              nj == 17 || ni == 35 || nj == 35 || ni == 53 || nj == 53)) {
    // AM1 B-H, B-C, & B-halogen corrections
    double ax;
    if (ni == 1 || nj == 1) {
      ax = 10.0 * mpc_sq(r - 0.832586);
      if (ax <= 25.0) scale = scale + tni * tnj / r * 0.412253 * exp(-ax);
      ax = 6.0 * mpc_sq(r - 1.186220);
      if (ax <= 25.0) scale = scale + tni * tnj / r * (-0.149917) * exp(-ax);
    } else if (ni == 6 || nj == 6) {
      ax = 8.0 * mpc_sq(r - 1.063995);
      if (ax <= 25.0) scale = scale + tni * tnj / r * 0.261751 * exp(-ax);
      ax = 5.0 * mpc_sq(r - 1.936492);
      if (ax <= 25.0) scale = scale + tni * tnj / r * 0.050275 * exp(-ax);
    } else {
      ax = 9.0 * mpc_sq(r - 0.819351);
      if (ax <= 25.0) scale = scale + tni * tnj / r * 0.359244 * exp(-ax);
      ax = 9.0 * mpc_sq(r - 1.574414);
      if (ax <= 25.0) scale = scale + tni * tnj / r * 0.074729 * exp(-ax);
    }
    if (ni == 5) {
      for (int ig = 1; ig <= 4; ++ig) {
        ax = mpc_g2(prm, nj, ig) * mpc_sq(r - mpc_g3(prm, nj, ig));
        if (ax <= 25.0) scale = scale + tni * tnj / r * mpc_g1(prm, nj, ig) * exp(-ax);
      }
    } else {
      for (int ig = 1; ig <= 4; ++ig) {
        ax = mpc_g2(prm, ni, ig) * mpc_sq(r - mpc_g3(prm, ni, ig));
        if (ax <= 25.0) scale = scale + tni * tnj / r * mpc_g1(prm, ni, ig) * exp(-ax);
      }
    }
    ngauss = 0;
  } else {
    ngauss = 4;
    if (fff > 1.0e-4) ngauss = 0;
  }
  for (int ig = 1; ig <= ngauss; ++ig) {
    if (fabs(mpc_g1(prm, ni, ig)) > 0.0) {
      const double ax = mpc_g2(prm, ni, ig) * mpc_sq(r - mpc_g3(prm, ni, ig));
      if (ax <= 25.0) scale = scale + tni * tnj / r * mpc_g1(prm, ni, ig) * exp(-ax);
    }
    if (fabs(mpc_g1(prm, nj, ig)) <= 0.0) continue;
    const double ax = mpc_g2(prm, nj, ig) * mpc_sq(r - mpc_g3(prm, nj, ig));
    if (ax > 25.0) continue;
    scale = scale + tni * tnj / r * mpc_g1(prm, nj, ig) * exp(-ax);
  }
  enuclr = enuclr + scale;
  if (pm6 || pm7 || pm8 || pm6_org) {
    // unpolarizable core - unpolarizable core ("12" part of Lennard-Jones)
    const double ax = r / (pow((double)ni, 0.3333) + pow((double)nj, 0.3333));
    if (ax < 3.0) {
      scale = 1.0e-8 / mpc_pow12(ax);
      enuclr = enuclr + ((scale < 1.0e5) ? scale : 1.0e5);
    }
  }
  return enuclr;
}

// ---------------------------------------------------------------------------
// elenuc (mndod.F90) restricted to s/p: fill the packed lower triangle of the
// electron-core attraction matrix for one atom with nat orbitals.
//   e[ind1*(ind1+1)/2 + ind2] (0-based local orbital indices, ind2 <= ind1)
// core[c] = cored(c+1, n).  indpp (set in fordd) is hard-coded:
//   indpp(1,1)=1 indpp(2,1)=4 indpp(3,1)=5 indpp(2,2)=2 indpp(3,2)=6 indpp(3,3)=3
// ---------------------------------------------------------------------------
static __device__ __forceinline__ int mpc_indpp(int i1, int i2) {  // 1-based, i2 <= i1
  if (i1 == i2) return i1;
  if (i1 == 2) return 4;
  return (i2 == 1) ? 5 : 6;
}

static __device__ __forceinline__ void mpc_elenuc_sp(const MozymePairRot &rot, const double *core,
                                                     int nat, double *e) {
  for (int ind1 = 0; ind1 < nat; ++ind1) {
    for (int ind2 = 0; ind2 <= ind1; ++ind2) {
      const int m = (ind1 * (ind1 + 1)) / 2 + ind2;
      double h = 0.0;
      if (ind1 == 0) {
        h = h + core[0];  // (SS/)
      } else if (ind2 == 0) {
        h = h + rot.p[0][ind1 - 1] * core[1];  // (SP/): sp(1,ind1)*cored(2,n)
      } else {
        const int ipp = mpc_indpp(ind1, ind2) - 1;  // 0-based component
        // cored(3,n)*pp(ipp,1,1) + cored(4,n)*(pp(ipp,2,2) + pp(ipp,3,3))
        h = h + core[2] * rot.pp[0][0][ipp] + core[3] * (rot.pp[1][1][ipp] + rot.pp[2][2][ipp]);
      }
      e[m] = h;
    }
  }
}

// ---------------------------------------------------------------------------
// Two-electron integrals in the molecular frame (rotatd + tx + w2mat for sp).
//
// Local-frame pair slots q = indx(i1,j1) - 1 (0-based, i1 >= j1, orbitals
// 1 = s, 2 = p-sigma, 3 = p-pi, 4 = p-pi*):
//   q: 0=(1,1) 1=(2,1) 2=(2,2) 3=(3,1) 4=(3,2) 5=(3,3) 6=(4,1) 7=(4,2)
//      8=(4,3) 9=(4,4)
// mpc_sp_ri_index[q_ij][q_kl] gives the Fortran ri index (1..22) of
// rep(ind2(indexd(i1,j1), indexd(k1,l1))) = ri(ipos(numb)), or 0 where
// ind2 = 0 (fordd + reppd2 tables folded together).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ int mpc_sp_ri_index(int qij, int qkl) {
  // rows: q_ij, columns: q_kl
  switch (qij) {
    case 0: {  // (ss|
      const int t[10] = {1, 5, 11, 0, 0, 12, 0, 0, 0, 12};
      return t[qkl];
    }
    case 1: {  // (so|
      const int t[10] = {2, 6, 13, 0, 0, 14, 0, 0, 0, 14};
      return t[qkl];
    }
    case 2: {  // (oo|
      const int t[10] = {3, 8, 16, 0, 0, 18, 0, 0, 0, 18};
      return t[qkl];
    }
    case 3: {  // (sp|
      const int t[10] = {0, 0, 0, 7, 15, 0, 0, 0, 0, 0};
      return t[qkl];
    }
    case 4: {  // (po|
      const int t[10] = {0, 0, 0, 10, 20, 0, 0, 0, 0, 0};
      return t[qkl];
    }
    case 5: {  // (pp|
      const int t[10] = {4, 9, 17, 0, 0, 19, 0, 0, 0, 21};
      return t[qkl];
    }
    case 6: {  // (sp*|
      const int t[10] = {0, 0, 0, 0, 0, 0, 7, 15, 0, 0};
      return t[qkl];
    }
    case 7: {  // (p*o|
      const int t[10] = {0, 0, 0, 0, 0, 0, 10, 20, 0, 0};
      return t[qkl];
    }
    case 8: {  // (p*p|
      const int t[10] = {0, 0, 0, 0, 0, 0, 0, 0, 22, 0};
      return t[qkl];
    }
    default: {  // 9: (p*p*|
      const int t[10] = {4, 9, 17, 0, 0, 21, 0, 0, 0, 19};
      return t[qkl];
    }
  }
}

// tx: first rotation step.  v[qij][KL] with KL = indx(k,l) - 1 the molecular
// pair on atom nj.  Loop order and accumulation order follow the Fortran.
static __device__ __forceinline__ void mpc_tx_sp(const MozymePairRot &rot, const double *ri, int ii,
                                                 int kk, double v[10][10]) {
  for (int a = 0; a < 10; ++a)
    for (int b = 0; b < 10; ++b) v[a][b] = 0.0;
  for (int i1 = 0; i1 < ii; ++i1) {
    for (int j1 = 0; j1 <= i1; ++j1) {
      const int qij = (i1 * (i1 + 1)) / 2 + j1;
      double *vij = v[qij];
      for (int k1 = 0; k1 < kk; ++k1) {
        for (int l1 = 0; l1 <= k1; ++l1) {
          const int qkl = (k1 * (k1 + 1)) / 2 + l1;
          const int nd = mpc_sp_ri_index(qij, qkl);
          if (nd == 0) continue;
          const double wrepp = ri[nd - 1];
          if (k1 == 0) {
            // met = 1: (s s)
            vij[0] = wrepp;
          } else if (l1 == 0) {
            // met = 2: (p s), k = k1 - 1 local p index
            const int k = k1 - 1;
            vij[1] = vij[1] + rot.p[k][0] * wrepp;  // v(ij,2) += sp(k,1)*w
            vij[3] = vij[3] + rot.p[k][1] * wrepp;  // v(ij,4) += sp(k,2)*w
            vij[6] = vij[6] + rot.p[k][2] * wrepp;  // v(ij,7) += sp(k,3)*w
          } else {
            // met = 3: (p p)
            const int k = k1 - 1;
            const int l = l1 - 1;
            const double *ppkl = rot.pp[k][l];
            vij[2] = vij[2] + ppkl[0] * wrepp;  // v(ij,3)  += pp(1,k,l)*w
            vij[5] = vij[5] + ppkl[1] * wrepp;  // v(ij,6)  += pp(2,k,l)*w
            vij[9] = vij[9] + ppkl[2] * wrepp;  // v(ij,10) += pp(3,k,l)*w
            vij[4] = vij[4] + ppkl[3] * wrepp;  // v(ij,5)  += pp(4,k,l)*w
            vij[7] = vij[7] + ppkl[4] * wrepp;  // v(ij,8)  += pp(5,k,l)*w
            vij[8] = vij[8] + ppkl[5] * wrepp;  // v(ij,9)  += pp(6,k,l)*w
          }
        }
      }
    }
  }
}

// rotatd second step: ww[(IJ)*limkl + KL], IJ = indx(i,j)-1 on atom ni,
// KL = indx(k,l)-1 on atom nj (this is exactly w2mat's linear order).
static __device__ __forceinline__ void mpc_rotate_w_sp(const MozymePairRot &rot, const double *ri,
                                                       int ii, int kk, double *ww) {
  double v[10][10];
  const int limkl = (kk * (kk + 1)) / 2;
  const int limij = (ii * (ii + 1)) / 2;
  const int istep = limkl * limij;
  for (int n = 0; n < 100; ++n) ww[n] = 0.0;
  if (istep <= 0) return;
  mpc_tx_sp(rot, ri, ii, kk, v);
  for (int i1 = 0; i1 < ii; ++i1) {
    for (int j1 = 0; j1 <= i1; ++j1) {
      const int qij = (i1 * (i1 + 1)) / 2 + j1;
      for (int k = 0; k < kk; ++k) {
        for (int l = 0; l <= k; ++l) {
          const int kl = (k * (k + 1)) / 2 + l;
          const double wrepp = v[qij][kl];
          if (wrepp == 0.0) continue;  // logv
          if (i1 == 0) {
            // met = 1: ww(indw(1,1)) = wrepp
            ww[0 * limkl + kl] = wrepp;
          } else if (j1 == 0) {
            // met = 2: ww(indw(i+1,1)) += sp(i1-1,i)*wrepp, i = 1..3
            const int ki = i1 - 1;
            for (int i = 1; i <= 3; ++i) {
              const int ij = ((i + 1) * i) / 2;  // indx(i+1,1) - 1
              ww[ij * limkl + kl] = ww[ij * limkl + kl] + rot.p[ki][i - 1] * wrepp;
            }
          } else {
            // met = 3
            const int ki = i1 - 1;
            const int li = j1 - 1;
            const double *ppkl = rot.pp[ki][li];
            for (int i = 1; i <= 3; ++i) {
              double cc = ppkl[i - 1];                 // pp(i,i1-1,j1-1)
              int ij = ((i + 1) * i) / 2 + i;          // indx(i+1,i+1) - 1
              ww[ij * limkl + kl] = ww[ij * limkl + kl] + cc * wrepp;
              for (int j = 1; j < i; ++j) {
                cc = ppkl[i + j];                      // pp(1+i+j,i1-1,j1-1)
                ij = ((i + 1) * i) / 2 + j;            // indx(i+1,j+1) - 1
                ww[ij * limkl + kl] = ww[ij * limkl + kl] + cc * wrepp;
              }
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Public entry point.
//
// Inputs : ni, nj      atomic numbers (1..107)
//          xi, xj      Cartesian coordinates in Angstrom
//          prm         parameter block
// Outputs: w[0..99]    two-electron integrals in rotate's order
//                      ((i,j) i>=j on ni outer, (k,l) k>=l on nj inner);
//                      *w_count = natorb(ni)(natorb(ni)+1)/2 *
//                                 natorb(nj)(natorb(nj)+1)/2 entries written
//                      (100 sp-sp, 10 sp-H, 1 H-H, 0 if a sparkle).
//          e1b[0..9]   electron (ni) - core (nj) attraction, packed lower
//                      triangle over natorb(ni) orbitals
//          e2a[0..9]   electron (nj) - core (ni) attraction
//          *enuc       core-core repulsion (eV)
// Returns false (outputs untouched) if either atom has d orbitals
// (natorb > 4) or an atomic number outside 1..107.  For rij^2 < 2e-5 A^2
// (rotate's small-rij exit) all outputs are zeroed, *w_count = 0 (rotate
// does not advance kr in that case) and true is returned.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ bool mozyme_pair_core_sp_dev(
    int ni, int nj, const double *xi, const double *xj, const MozymePairCoreParams &prm,
    double *w, int *w_count, double *e1b, double *e2a, double *enuc) {
  if (ni < 1 || ni > 107 || nj < 1 || nj > 107) return false;
  const int li = prm.natorb[ni - 1];
  const int lj = prm.natorb[nj - 1];
  if (li > 4 || lj > 4 || li < 0 || lj < 0) return false;

  for (int i = 0; i < 10; ++i) {
    e1b[i] = 0.0;
    e2a[i] = 0.0;
  }
  for (int i = 0; i < 100; ++i) w[i] = 0.0;
  *w_count = 0;
  *enuc = 0.0;

  // rotate: small-rij exit on the squared distance (Angstrom^2)
  {
    const double x0 = xi[0] - xj[0];
    const double x1 = xi[1] - xj[1];
    const double x2 = xi[2] - xj[2];
    const double rij2 = x0 * x0 + x1 * x1 + x2 * x2;
    if (rij2 < 0.00002) return true;
  }

  // rotatd: rotmat(nj, ni, ci, cj, r)
  MozymePairRot rot;
  const double rij = mpc_rotmat_sp(xi, xj, rot);

  // reppd: 22 local integrals + gab
  double ri[22];
  double gab;
  mpc_reppd(prm, ni, nj, rij, ri, &gab);

  // spcore in Bohr
  const double r = rij / prm.a0;
  double cored1[4], cored2[4];
  mpc_spcore(prm, ni, nj, r, cored1, cored2);

  // l_feather blending of cored (rotatd)
  {
    double point, cnst;
    if (prm.l_feather) {
      mpc_to_point(prm, rij, &point, &cnst);
    } else {
      cnst = 1.0;
      point = 0.0;
    }
    // reppd2 adds nothing for s/p atoms (rep(1:34) = ri(ipos), core(5:10) untouched)
    point = -(prm.ev / r) * prm.tore[nj - 1];
    cored1[0] = cored1[0] * cnst + (1.0 - cnst) * point;
    cored1[1] = cored1[1] * cnst;
    cored1[2] = cored1[2] * cnst + (1.0 - cnst) * point;
    cored1[3] = cored1[3] * cnst + (1.0 - cnst) * point;
    point = -(prm.ev / r) * prm.tore[ni - 1];
    cored2[0] = cored2[0] * cnst + (1.0 - cnst) * point;
    cored2[1] = cored2[1] * cnst;
    cored2[2] = cored2[2] * cnst + (1.0 - cnst) * point;
    cored2[3] = cored2[3] * cnst + (1.0 - cnst) * point;
  }

  // two-electron integrals in the molecular frame
  double ww[100];
  mpc_rotate_w_sp(rot, ri, li, lj, ww);
  const int limij = (li * (li + 1)) / 2;
  const int limkl = (lj * (lj + 1)) / 2;
  const int istep = limij * limkl;

  // PM7 "d"-orbital balance block of rotatd.  Only the parts that can touch
  // cored(1..4,:) or ww(1..istep) for s/p atoms are reproduced; all other
  // statements in that block address ww entries beyond istep (never copied
  // to w) or cored(7,9,10) (never read by elenuc for s/p).
  if (prm.method_pm7) {
    if (prm.iod[ni - 1] > 0) {
      if (lj > 1) {
        const double sum = cored1[0] - (cored1[2] + 2.0 * cored1[3]) / 3.0;
        cored1[2] = cored1[2] + sum;
        cored1[3] = cored1[3] + sum;
      }
    }
    if (prm.iod[nj - 1] > 0) {
      if (li > 1) {
        const double sum = cored2[0] - (cored2[2] + 2.0 * cored2[3]) / 3.0;
        cored2[2] = cored2[2] + sum;
        cored2[3] = cored2[3] + sum;
      }
      // "s" on ni with "d" on nj: ww(k(k+1)/2) += sum for k = 5..9, i.e.
      // ww(15), ww(21), ww(28), ww(36), ww(45); within istep only for sp-sp.
      {
        const double sum =
            ww[0] - (ww[14] + ww[20] + ww[27] + ww[35] + ww[44]) / 5.0;
        ww[14] = ww[14] + sum;
        ww[20] = ww[20] + sum;
        ww[27] = ww[27] + sum;
        ww[35] = ww[35] + sum;
        ww[44] = ww[44] + sum;
      }
    }
  }

  // w2mat
  for (int n = 0; n < istep; ++n) w[n] = ww[n];
  *w_count = istep;

  // elenuc -> e1b (electrons on ni, core of nj), e2a (electrons on nj, core of ni)
  mpc_elenuc_sp(rot, cored1, li, e1b);
  mpc_elenuc_sp(rot, cored2, lj, e2a);

  // ccrep (distance passed in Bohr, converted back inside)
  *enuc = mpc_ccrep(prm, ni, nj, r, gab);
  return true;
}


// ===========================================================================
//                     Extension to atoms with d orbitals
// ===========================================================================
//
// mozyme_pair_core_dev() reproduces rotate() for any natorb <= 9.  It is a
// separate entry point; mozyme_pair_core_sp_dev() above is untouched.
//
// Additional pieces of the rotatd path reproduced here:
//   fordd    : constant index tables indexd/indx (by formula) and ind2, isym,
//              ch, inddd, inddp (dumped from the Fortran after fordd and
//              embedded below as device constants; they contain no
//              parameter dependence)
//   rotmat   : d(5,5) block, sd, dp(15,5,3), d_d(15,5,5)
//   rijkl    : point-charge multipole interactions (charg) using
//              po(1..9,ni), ddp(1..6,ni) and ch
//   reppd2   : rep(35..491) d integrals + cored(5..10,1..2)
//   tx/rotatd: full two-step rotation into ww(2025), PM7 iod block complete
//   elenuc   : s/p/d blocks
// dorbs(ni) is taken as natorb(ni) == 9 (moldat sets natorb = 9 exactly when
// dorbs is true).
//
#ifdef __CUDACC__
#define MPC_TABLE static __constant__
#else
#define MPC_TABLE static const
#endif
// ind2, Fortran column-major order (ind2(45,45): ind2(ij,kl) -> [(ij-1) + 45*(kl-1)])
MPC_TABLE int mpc_ind2_tab[2025] = {1, 6, 0, 0, 124, 0, 0, 0, 0, 11, 0, 0, 138, 0, 0, 0, 0, 20, 0, 0, 206, 0, 0, 0, 30, 0, 0, 284, 0, 0, 172, 0, 0, 0, 0, 240, 0, 0, 0, 318, 0, 0, 402, 0, 478, 2, 7, 0, 0, 125, 0, 0, 0, 0, 12, 0, 0, 139, 0, 0, 0, 0, 21, 0, 0, 207, 0, 0, 0, 31, 0, 0, 285, 0, 0, 173, 0, 0, 0, 0, 241, 0, 0, 0, 319, 0, 0, 403, 0, 479, 0, 0, 16, 0, 0, 186, 0, 0, 0, 0, 18, 0, 0, 196, 0, 0, 0, 0, 0, 152, 0, 0, 353, 0, 0, 0, 0, 0, 0, 440, 0, 230, 0, 0, 0, 0, 0, 382, 0, 0, 0, 467, 0, 0, 0, 0, 0, 0, 25, 0, 0, 257, 0, 0, 0, 0, 27, 0, 0, 267, 0, 0, 0, 0, 0, 0, 0, 0, 430, 0, 162, 0, 0, 363, 0, 0, 0, 301, 0, 0, 0, 0, 0, 457, 0, 392, 0, 0, 0, 0, 35, 44, 0, 0, 129, 0, 0, 0, 0, 53, 0, 0, 143, 0, 0, 0, 0, 78, 0, 0, 211, 0, 0, 0, 112, 0, 0, 289, 0, 0, 177, 0, 0, 0, 0, 245, 0, 0, 0, 323, 0, 0, 407, 0, 483, 0, 0, 63, 0, 0, 189, 0, 0, 0, 0, 71, 0, 0, 199, 0, 0, 0, 0, 0, 155, 0, 0, 356, 0, 0, 0, 0, 0, 0, 443, 0, 233, 0, 0, 0, 0, 0, 385, 0, 0, 0, 470, 0, 0, 0, 0, 0, 0, 91, 0, 0, 260, 0, 0, 0, 0, 99, 0, 0, 270, 0, 0, 0, 0, 0, 0, 0, 0, 433, 0, 165, 0, 0, 366, 0, 0, 0, 304, 0, 0, 0, 0, 0, 460, 0, 395, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 341, 0, 0, 0, 0, 0, 0, 0, 350, 0, 85, 0, 0, 218, 0, 0, 0, 119, 0, 0, 296, 0, 0, 0, 0, 0, 379, 0, 252, 0, 0, 0, 330, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 420, 0, 0, 0, 0, 0, 0, 0, 427, 0, 109, 0, 0, 281, 0, 0, 0, 0, 227, 0, 0, 0, 0, 0, 0, 0, 454, 0, 315, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 0, 0, 126, 0, 0, 0, 0, 13, 0, 0, 140, 0, 0, 0, 0, 22, 0, 0, 208, 0, 0, 0, 32, 0, 0, 286, 0, 0, 174, 0, 0, 0, 0, 242, 0, 0, 0, 320, 0, 0, 404, 0, 480, 0, 0, 17, 0, 0, 187, 0, 0, 0, 0, 19, 0, 0, 197, 0, 0, 0, 0, 0, 153, 0, 0, 354, 0, 0, 0, 0, 0, 0, 441, 0, 231, 0, 0, 0, 0, 0, 383, 0, 0, 0, 468, 0, 0, 0, 0, 0, 0, 26, 0, 0, 258, 0, 0, 0, 0, 28, 0, 0, 268, 0, 0, 0, 0, 0, 0, 0, 0, 431, 0, 163, 0, 0, 364, 0, 0, 0, 302, 0, 0, 0, 0, 0, 458, 0, 393, 0, 0, 0, 0, 36, 45, 0, 0, 130, 0, 0, 0, 0, 54, 0, 0, 144, 0, 0, 0, 0, 79, 0, 0, 212, 0, 0, 0, 113, 0, 0, 290, 0, 0, 178, 0, 0, 0, 0, 246, 0, 0, 0, 324, 0, 0, 408, 0, 484, 0, 0, 64, 0, 0, 190, 0, 0, 0, 0, 72, 0, 0, 200, 0, 0, 0, 0, 0, 156, 0, 0, 357, 0, 0, 0, 0, 0, 0, 444, 0, 234, 0, 0, 0, 0, 0, 386, 0, 0, 0, 471, 0, 0, 0, 0, 0, 0, 92, 0, 0, 261, 0, 0, 0, 0, 100, 0, 0, 271, 0, 0, 0, 0, 0, 0, 0, 0, 434, 0, 166, 0, 0, 367, 0, 0, 0, 305, 0, 0, 0, 0, 0, 461, 0, 396, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 342, 0, 0, 0, 0, 0, 0, 0, 351, 0, 86, 0, 0, 219, 0, 0, 0, 120, 0, 0, 297, 0, 0, 0, 0, 0, 380, 0, 253, 0, 0, 0, 331, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 421, 0, 0, 0, 0, 0, 0, 0, 428, 0, 110, 0, 0, 282, 0, 0, 0, 0, 228, 0, 0, 0, 0, 0, 0, 0, 455, 0, 316, 0, 0, 0, 0, 0, 0, 0, 0, 4, 9, 0, 0, 127, 0, 0, 335, 0, 14, 0, 0, 141, 0, 0, 344, 0, 23, 0, 0, 209, 0, 0, 0, 33, 0, 0, 287, 0, 0, 175, 0, 0, 373, 0, 243, 0, 0, 0, 321, 0, 0, 405, 0, 481, 0, 0, 0, 0, 0, 0, 0, 0, 416, 0, 0, 0, 0, 0, 0, 0, 423, 0, 29, 0, 0, 277, 0, 0, 0, 0, 223, 0, 0, 0, 0, 0, 0, 0, 450, 0, 311, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 188, 0, 0, 0, 0, 70, 0, 0, 198, 0, 0, 0, 0, 0, 154, 0, 0, 355, 0, 0, 0, 0, 0, 0, 442, 0, 232, 0, 0, 0, 0, 0, 384, 0, 0, 0, 469, 0, 0, 0, 38, 47, 0, 0, 132, 0, 0, 337, 0, 56, 0, 0, 146, 0, 0, 346, 0, 81, 0, 0, 214, 0, 0, 0, 115, 0, 0, 292, 0, 0, 180, 0, 0, 375, 0, 248, 0, 0, 0, 326, 0, 0, 410, 0, 486, 0, 0, 0, 0, 0, 0, 0, 0, 418, 0, 0, 0, 0, 0, 0, 0, 425, 0, 107, 0, 0, 279, 0, 0, 0, 0, 225, 0, 0, 0, 0, 0, 0, 0, 452, 0, 313, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 0, 192, 0, 0, 0, 0, 74, 0, 0, 202, 0, 0, 0, 0, 0, 158, 0, 0, 359, 0, 0, 0, 0, 0, 0, 446, 0, 236, 0, 0, 0, 0, 0, 388, 0, 0, 0, 473, 0, 0, 0, 0, 0, 0, 96, 0, 0, 265, 0, 0, 0, 0, 104, 0, 0, 275, 0, 0, 0, 0, 0, 0, 0, 0, 438, 0, 170, 0, 0, 371, 0, 0, 0, 309, 0, 0, 0, 0, 0, 465, 0, 400, 0, 0, 0, 0, 5, 10, 0, 0, 128, 0, 0, 336, 0, 15, 0, 0, 142, 0, 0, 345, 0, 24, 0, 0, 210, 0, 0, 0, 34, 0, 0, 288, 0, 0, 176, 0, 0, 374, 0, 244, 0, 0, 0, 322, 0, 0, 406, 0, 482, 0, 0, 0, 90, 0, 0, 259, 0, 0, 0, 0, 98, 0, 0, 269, 0, 0, 0, 0, 0, 0, 0, 0, 432, 0, 164, 0, 0, 365, 0, 0, 0, 303, 0, 0, 0, 0, 0, 459, 0, 394, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 417, 0, 0, 0, 0, 0, 0, 0, 424, 0, 106, 0, 0, 278, 0, 0, 0, 0, 224, 0, 0, 0, 0, 0, 0, 0, 451, 0, 312, 0, 0, 0, 0, 0, 0, 0, 0, 40, 49, 0, 0, 134, 0, 0, 339, 0, 58, 0, 0, 148, 0, 0, 348, 0, 83, 0, 0, 216, 0, 0, 0, 117, 0, 0, 294, 0, 0, 182, 0, 0, 377, 0, 250, 0, 0, 0, 328, 0, 0, 412, 0, 488, 0, 0, 0, 94, 0, 0, 263, 0, 0, 0, 0, 102, 0, 0, 273, 0, 0, 0, 0, 0, 0, 0, 0, 436, 0, 168, 0, 0, 369, 0, 0, 0, 307, 0, 0, 0, 0, 0, 463, 0, 398, 0, 0, 0, 0, 0, 0, 68, 0, 0, 194, 0, 0, 0, 0, 76, 0, 0, 204, 0, 0, 0, 0, 0, 160, 0, 0, 361, 0, 0, 0, 0, 0, 0, 448, 0, 238, 0, 0, 0, 0, 0, 390, 0, 0, 0, 475, 0, 0, 0, 37, 46, 0, 0, 131, 0, 0, 0, 0, 55, 0, 0, 145, 0, 0, 0, 0, 80, 0, 0, 213, 0, 0, 0, 114, 0, 0, 291, 0, 0, 179, 0, 0, 0, 0, 247, 0, 0, 0, 325, 0, 0, 409, 0, 485, 0, 0, 65, 0, 0, 191, 0, 0, 0, 0, 73, 0, 0, 201, 0, 0, 0, 0, 0, 157, 0, 0, 358, 0, 0, 0, 0, 0, 0, 445, 0, 235, 0, 0, 0, 0, 0, 387, 0, 0, 0, 472, 0, 0, 0, 0, 0, 0, 93, 0, 0, 262, 0, 0, 0, 0, 101, 0, 0, 272, 0, 0, 0, 0, 0, 0, 0, 0, 435, 0, 167, 0, 0, 368, 0, 0, 0, 306, 0, 0, 0, 0, 0, 462, 0, 397, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 343, 0, 0, 0, 0, 0, 0, 0, 352, 0, 87, 0, 0, 220, 0, 0, 0, 121, 0, 0, 298, 0, 0, 0, 0, 0, 381, 0, 254, 0, 0, 0, 332, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 422, 0, 0, 0, 0, 0, 0, 0, 429, 0, 111, 0, 0, 283, 0, 0, 0, 0, 229, 0, 0, 0, 0, 0, 0, 0, 456, 0, 317, 0, 0, 0, 0, 0, 0, 0, 0, 39, 48, 0, 0, 133, 0, 0, 338, 0, 57, 0, 0, 147, 0, 0, 347, 0, 82, 0, 0, 215, 0, 0, 0, 116, 0, 0, 293, 0, 0, 181, 0, 0, 376, 0, 249, 0, 0, 0, 327, 0, 0, 411, 0, 487, 0, 0, 0, 0, 0, 0, 0, 0, 419, 0, 0, 0, 0, 0, 0, 0, 426, 0, 108, 0, 0, 280, 0, 0, 0, 0, 226, 0, 0, 0, 0, 0, 0, 0, 453, 0, 314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 193, 0, 0, 0, 0, 75, 0, 0, 203, 0, 0, 0, 0, 0, 159, 0, 0, 360, 0, 0, 0, 0, 0, 0, 447, 0, 237, 0, 0, 0, 0, 0, 389, 0, 0, 0, 474, 0, 0, 0, 0, 0, 0, 97, 0, 0, 266, 0, 0, 0, 0, 105, 0, 0, 276, 0, 0, 0, 0, 0, 0, 0, 0, 439, 0, 171, 0, 0, 372, 0, 0, 0, 310, 0, 0, 0, 0, 0, 466, 0, 401, 0, 0, 0, 0, 41, 50, 0, 0, 135, 0, 0, 340, 0, 59, 0, 0, 149, 0, 0, 349, 0, 84, 0, 0, 217, 0, 0, 0, 118, 0, 0, 295, 0, 0, 183, 0, 0, 378, 0, 251, 0, 0, 0, 329, 0, 0, 413, 0, 489, 0, 0, 0, 95, 0, 0, 264, 0, 0, 0, 0, 103, 0, 0, 274, 0, 0, 0, 0, 0, 0, 0, 0, 437, 0, 169, 0, 0, 370, 0, 0, 0, 308, 0, 0, 0, 0, 0, 464, 0, 399, 0, 0, 0, 0, 0, 0, 69, 0, 0, 195, 0, 0, 0, 0, 77, 0, 0, 205, 0, 0, 0, 0, 0, 161, 0, 0, 362, 0, 0, 0, 0, 0, 0, 449, 0, 239, 0, 0, 0, 0, 0, 391, 0, 0, 0, 476, 0, 0, 0, 42, 51, 0, 0, 136, 0, 0, 0, 0, 60, 0, 0, 150, 0, 0, 0, 0, 88, 0, 0, 221, 0, 0, 0, 122, 0, 0, 299, 0, 0, 184, 0, 0, 0, 0, 255, 0, 0, 0, 333, 0, 0, 414, 0, 490, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 477, 0, 43, 52, 0, 0, 137, 0, 0, 0, 0, 61, 0, 0, 151, 0, 0, 0, 0, 89, 0, 0, 222, 0, 0, 0, 123, 0, 0, 300, 0, 0, 185, 0, 0, 0, 0, 256, 0, 0, 0, 334, 0, 0, 415, 0, 491};
// isym, Fortran column-major order (isym(491): isym(n) -> [n-1])
MPC_TABLE int mpc_isym_tab[491] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 39, 0, 42, 0, 0, 0, 0, 0, 47, 48, 0, 51, 0, 0, 0, 0, 0, 56, 57, 0, 60, 0, 0, 0, 0, 0, 0, 66, 67, 0, 0, 0, 0, 0, 0, 74, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 62, 63, 64, 65, -66, -67, 66, 67, 70, 71, 72, 73, -74, -75, 74, 75, 86, 86, 0, 85, 86, 87, 78, 79, 80, 83, 84, 81, 82, -85, -86, -87, 88, 88, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 132, 133, 0, 136, 0, 0, 0, 0, 141, 0, 0, 0, 0, 0, 146, 147, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 158, 159, 152, 153, 154, 155, 156, 157, -158, -159, 158, 159, 0, 0, 0, 0, 175, 0, 0, 0, 0, 0, 180, 181, 0, 184, 0, 0, 0, 0, 0, 0, 0, 0, 192, 193, 0, 0, 0, 0, 0, 0, 0, 0, 202, 203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 221, 0, 219, 219, 0, 218, 219, 220, 0, 0, 0, 0, 0, 0, 0, 0, 236, 237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 186, 187, 188, 189, 190, 191, -192, -193, 192, 193, 196, 197, 198, 199, 200, 201, -202, -203, 202, 203, 223, 219, 219, 226, 218, 219, 220, 206, 207, 208, 210, 209, 211, 212, 213, 216, 217, 214, 215, -218, -219, -220, 221, 221, 230, 231, 232, 233, 234, 235, -236, -237, 236, 237, 0, 253, 253, 0, 252, 253, 254, 240, 241, 242, 244, 243, 245, 246, 247, 250, 251, 248, 249, -252, -253, -254, 255, 255, 0, -335, 0, 0, -337, -338, 0, 337, 0, 223, -223, 219, 226, -219, -226, 218, 219, 220, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -353, -354, -355, -356, -357, -358, 359, 360, -361, -362, 0, -373, 0, 0, -375, -376, 0, 375, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -382, -383, -384, -385, -386, -387, 388, 389, -390, -391, 0, 0, 0, 0, 405, 0, 0, 0, 0, 0, 410, 411, 0, 0, 335, 337, 337, 338, 341, 337, 343, 223, 219, 219, 226, 218, 219, 220, 353, 354, 355, 356, 357, 358, -361, -362, 359, 360, 353, 354, 355, 356, 357, 358, 361, 362, 359, 360, 373, 375, 375, 376, 379, 375, 381, 382, 383, 384, 385, 386, 387, -390, -391, 388, 389, 382, 383, 384, 385, 386, 387, 390, 391, 388, 389, 0, 402, 403, 404, 405, 405, 407, 408, 409, 410, 411, 410, 411, 415, 414};
// inddd, Fortran column-major order (inddd(5,5): inddd(i,j) -> [(i-1) + 5*(j-1)])
MPC_TABLE int mpc_inddd_tab[25] = {1, 6, 7, 9, 12, 6, 2, 8, 10, 13, 7, 8, 3, 11, 14, 9, 10, 11, 4, 15, 12, 13, 14, 15, 5};
// inddp, Fortran column-major order (inddp(5,3): inddp(i,j) -> [(i-1) + 5*(j-1)])
MPC_TABLE int mpc_inddp_tab[15] = {1, 4, 7, 10, 13, 2, 5, 8, 11, 14, 3, 6, 9, 12, 15};
// ch(45,0:2,-2:2): ch(ij,l,m) -> [(ij-1) + 45*l + 135*(m+2)]
MPC_TABLE double mpc_ch_tab[675] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.15470054, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.57735027, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.57735027, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.15470054, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.15470054, 0.0, 0.0, 0.0, 0.0, 1.33333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.66666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.66666667, 0.0, 0.0, 0.0, 0.0, 0.0, 1.33333333, 0.0, 0.0, 0.0, 0.0, 0.66666667, 0.0, 0.0, 0.0, 0.66666667, 0.0, 0.0, -1.33333333, 0.0, -1.33333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.57735027, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.57735027, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.15470054, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

#undef MPC_TABLE
// NOTE: static __constant__ replicates the ~15.6 KB of tables per translation
// unit that includes this header.  If it is included from several .cu files
// under -rdc=true, or constant-cache broadcast is not applicable (one thread
// per pair reads different entries), change MPC_TABLE to `static __device__`.

// indexd(i,j) = -(j(j-1))/2 + i + 9(j-1) for i >= j, symmetric (fordd)
static __device__ __forceinline__ int mpc_indexd(int i, int j) {
  if (i < j) {
    const int t = i;
    i = j;
    j = t;
  }
  return (-(j * (j - 1)) / 2) + i + 9 * (j - 1);
}
// indx(i,j) = i(i-1)/2 + j for i >= j, symmetric (fordd)
static __device__ __forceinline__ int mpc_indx(int i, int j) {
  if (i < j) {
    const int t = i;
    i = j;
    j = t;
  }
  return (i * (i - 1)) / 2 + j;
}
static __device__ __forceinline__ int mpc_ind2(int ij, int kl) {
  return mpc_ind2_tab[(ij - 1) + 45 * (kl - 1)];
}
static __device__ __forceinline__ int mpc_isym(int n) { return mpc_isym_tab[n - 1]; }
static __device__ __forceinline__ double mpc_ch(int ij, int l, int m) {
  return mpc_ch_tab[(ij - 1) + 45 * l + 135 * (m + 2)];
}
static __device__ __forceinline__ int mpc_inddd(int i, int j) {
  return mpc_inddd_tab[(i - 1) + 5 * (j - 1)];
}
static __device__ __forceinline__ int mpc_inddp(int i, int j) {
  return mpc_inddp_tab[(i - 1) + 5 * (j - 1)];
}
static __device__ __forceinline__ int mpc_iabs(int x) { return x < 0 ? -x : x; }

// ---------------------------------------------------------------------------
// charg (mndod.F90): interaction between two point-charge configurations.
// r in Bohr.  Expression order follows the Fortran source line by line.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ double mpc_charg(double r, int l1, int l2, int m, double da,
                                                   double db, double add) {
  double c = 0.0;
  if (l1 == 0 && l2 == 0) {
    c = 1.0 / sqrt(r * r + add);
  } else if (l1 == 1 && l2 == 0) {
    c = (-1.0 / sqrt(mpc_sq(r + da) + add)) + 1.0 / sqrt(mpc_sq(r - da) + add);
    c = c / 2.0;
  } else if (l1 == 0 && l2 == 1) {
    c = 1.0 / sqrt(mpc_sq(r + db) + add) - 1.0 / sqrt(mpc_sq(r - db) + add);
    c = c / 2.0;
  } else if (l1 == 1 && l2 == 1 && m == 0) {
    const double dzdz = 1.0 / sqrt(mpc_sq(r + da - db) + add) + 1.0 / sqrt(mpc_sq(r - da + db) + add) -
                        1.0 / sqrt(mpc_sq(r - da - db) + add) - 1.0 / sqrt(mpc_sq(r + da + db) + add);
    c = dzdz / 4.0;
  } else if (l1 == 1 && l2 == 1 && m == 1) {
    const double dxdx = 2.0 / sqrt(r * r + mpc_sq(da - db) + add) - 2.0 / sqrt(r * r + mpc_sq(da + db) + add);
    c = dxdx * 0.25;
  } else if (l1 == 0 && l2 == 2) {
    const double qqzz = 1.0 / sqrt(mpc_sq(r - db) + add) - 2.0 / sqrt(r * r + db * db + add) +
                        1.0 / sqrt(mpc_sq(r + db) + add);
    c = qqzz / 4.0;
  } else if (l1 == 2 && l2 == 0) {
    const double qzzq = 1.0 / sqrt(mpc_sq(r - da) + add) - 2.0 / sqrt(r * r + da * da + add) +
                        1.0 / sqrt(mpc_sq(r + da) + add);
    c = qzzq / 4.0;
  } else if (l1 == 1 && l2 == 2 && m == 0) {
    const double dzqzz = 1.0 / sqrt(mpc_sq(r - da - db) + add) - 2.0 / sqrt(mpc_sq(r - da) + db * db + add) +
                         1.0 / sqrt(mpc_sq(r + db - da) + add) - 1.0 / sqrt(mpc_sq(r - db + da) + add) +
                         2.0 / sqrt(mpc_sq(r + da) + db * db + add) - 1.0 / sqrt(mpc_sq(r + da + db) + add);
    c = dzqzz / 8.0;
  } else if (l1 == 2 && l2 == 1 && m == 0) {
    const double qzzdz = (-1.0 / sqrt(mpc_sq(r - da - db) + add)) + 2.0 / sqrt(mpc_sq(r - db) + da * da + add) -
                         1.0 / sqrt(mpc_sq(r + da - db) + add) + 1.0 / sqrt(mpc_sq(r - da + db) + add) -
                         2.0 / sqrt(mpc_sq(r + db) + da * da + add) + 1.0 / sqrt(mpc_sq(r + da + db) + add);
    c = qzzdz / 8.0;
  } else if (l1 == 2 && l2 == 2 && m == 0) {
    const double zzzz = 1.0 / sqrt(mpc_sq(r - da - db) + add) + 1.0 / sqrt(mpc_sq(r + da + db) + add) +
                        1.0 / sqrt(mpc_sq(r - da + db) + add) + 1.0 / sqrt(mpc_sq(r + da - db) + add) -
                        2.0 / sqrt(mpc_sq(r - da) + db * db + add) - 2.0 / sqrt(mpc_sq(r - db) + da * da + add) -
                        2.0 / sqrt(mpc_sq(r + da) + db * db + add) - 2.0 / sqrt(mpc_sq(r + db) + da * da + add) +
                        2.0 / sqrt(r * r + mpc_sq(da - db) + add) + 2.0 / sqrt(r * r + mpc_sq(da + db) + add);
    const double xyxy = 4.0 / sqrt(r * r + mpc_sq(da - db) + add) + 4.0 / sqrt(r * r + mpc_sq(da + db) + add) -
                        8.0 / sqrt(r * r + da * da + db * db + add);
    c = zzzz / 16.0 - xyxy / 64.0;
  } else if (l1 == 1 && l2 == 2 && m == 1) {
    const double ab = db / sqrt(2.0);
    const double dxqxz = (-2.0 / sqrt(mpc_sq(r - ab) + mpc_sq(da - ab) + add)) +
                         2.0 / sqrt(mpc_sq(r + ab) + mpc_sq(da - ab) + add) +
                         2.0 / sqrt(mpc_sq(r - ab) + mpc_sq(da + ab) + add) -
                         2.0 / sqrt(mpc_sq(r + ab) + mpc_sq(da + ab) + add);
    c = dxqxz / 8.0;
  } else if (l1 == 2 && l2 == 1 && m == 1) {
    const double aa = da / sqrt(2.0);
    const double qxzdx = (-2.0 / sqrt(mpc_sq(r + aa) + mpc_sq(aa - db) + add)) +
                         2.0 / sqrt(mpc_sq(r - aa) + mpc_sq(aa - db) + add) +
                         2.0 / sqrt(mpc_sq(r + aa) + mpc_sq(aa + db) + add) -
                         2.0 / sqrt(mpc_sq(r - aa) + mpc_sq(aa + db) + add);
    c = qxzdx / 8.0;
  } else if (l1 == 2 && l2 == 2 && m == 1) {
    const double aa = da / sqrt(2.0);
    const double ab = db / sqrt(2.0);
    const double qxzqxz = 2.0 / sqrt(mpc_sq(r + aa - ab) + mpc_sq(aa - ab) + add) -
                          2.0 / sqrt(mpc_sq(r + aa + ab) + mpc_sq(aa - ab) + add) -
                          2.0 / sqrt(mpc_sq(r - aa - ab) + mpc_sq(aa - ab) + add) +
                          2.0 / sqrt(mpc_sq(r - aa + ab) + mpc_sq(aa - ab) + add) -
                          2.0 / sqrt(mpc_sq(r + aa - ab) + mpc_sq(aa + ab) + add) +
                          2.0 / sqrt(mpc_sq(r + aa + ab) + mpc_sq(aa + ab) + add) +
                          2.0 / sqrt(mpc_sq(r - aa - ab) + mpc_sq(aa + ab) + add) -
                          2.0 / sqrt(mpc_sq(r - aa + ab) + mpc_sq(aa + ab) + add);
    c = qxzqxz / 16.0;
  } else if (l1 == 2 && l2 == 2 && m == 2) {
    const double xyxy = 4.0 / sqrt(r * r + mpc_sq(da - db) + add) + 4.0 / sqrt(r * r + mpc_sq(da + db) + add) -
                        8.0 / sqrt(r * r + da * da + db * db + add);
    c = xyxy / 16.0;
  }
  return c;
}

// ---------------------------------------------------------------------------
// rijkl (mndod.F90): two-center integral over local multipoles.
// ij, kl are indexd pair indices; li..ll the l quantum numbers; ic selects
// the core (po(9,·)) exponent for the monopole of atom 1 (ic=1) or 2 (ic=2).
// r in Bohr.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ double mpc_rijkl(const MozymePairCoreParams &prm, int ni, int nj,
                                                   int ij, int kl, int li, int lj, int lk, int ll,
                                                   int ic, double r) {
  double pij = 0.0, pkl = 0.0, dij = 0.0, dkl = 0.0;
  int l1min = mpc_iabs(li - lj);
  int l1max = li + lj;
  const int lij = mpc_indx(li + 1, lj + 1);
  int l2min = mpc_iabs(lk - ll);
  int l2max = lk + ll;
  const int lkl = mpc_indx(lk + 1, ll + 1);
  if (l1max > 2) l1max = 2;
  if (l1min > 2) l1min = 2;
  if (l2max > 2) l2max = 2;
  if (l2min > 2) l2min = 2;
  double sum = 0.0;
  for (int l1 = l1min; l1 <= l1max; ++l1) {
    if (l1 == 0) {
      switch (lij) {
        case 1:
          pij = mpc_po(prm, 1, ni);
          if (ic == 1) pij = mpc_po(prm, 9, ni);
          break;
        case 3:
          pij = mpc_po(prm, 7, ni);
          break;
        case 6:
          pij = mpc_po(prm, 8, ni);
          break;
        default:
          break;
      }
    } else {
      dij = mpc_ddp(prm, lij, ni);
      pij = mpc_po(prm, lij, ni);
    }
    for (int l2 = l2min; l2 <= l2max; ++l2) {
      if (l2 == 0) {
        switch (lkl) {
          case 1:
            pkl = mpc_po(prm, 1, nj);
            if (ic == 2) pkl = mpc_po(prm, 9, nj);
            break;
          case 3:
            pkl = mpc_po(prm, 7, nj);
            break;
          case 6:
            pkl = mpc_po(prm, 8, nj);
            break;
          default:
            break;
        }
      } else {
        dkl = mpc_ddp(prm, lkl, nj);
        pkl = mpc_po(prm, lkl, nj);
      }
      const double add = mpc_sq(pij + pkl);
      const int lmin = (l1 < l2) ? l1 : l2;
      double s1 = 0.0;
      for (int m = -lmin; m <= lmin; ++m) {
        const double ccc = mpc_ch(ij, l1, m) * mpc_ch(kl, l2, m);
        if (ccc == 0.0) continue;
        const int mm = mpc_iabs(m);
        s1 = s1 + mpc_charg(r, l1, l2, mm, dij, dkl, add) * ccc;
      }
      sum = sum + s1;
    }
  }
  return sum;
}

// ---------------------------------------------------------------------------
// rotmat (mndod.F90), full version: p/pp as in mpc_rotmat_sp plus the d(5,5)
// block and the derived sd (= d), dp(15,5,3), d_d(15,5,5) arrays.
//   d[k][c]       = Fortran d(k+1, c+1)
//   dp[k][l][c]   = Fortran dp(c+1, k+1, l+1)   (k: d orbital, l: p orbital)
//   dd[k][l][c]   = Fortran d_d(c+1, k+1, l+1), k >= l (k < l never written)
// The d block is only computed when at least one atom has d orbitals.
// ---------------------------------------------------------------------------
struct MozymePairRotD {
  MozymePairRot sp;
  double d[5][5];
  double dp[5][3][15];
  double dd[5][5][15];
};

static __device__ __forceinline__ double mpc_rotmat_full(const double *xi, const double *xj,
                                                         bool dorb, MozymePairRotD &rot) {
  const double pt5sq3 = 0.8660254037841;
  const double small = 1.0e-7;
  const double x11 = xj[0] - xi[0];
  const double x22 = xj[1] - xi[1];
  const double x33 = xj[2] - xi[2];
  const double b = x11 * x11 + x22 * x22;
  const double r = sqrt(b + x33 * x33);
  const double sqb = sqrt(b);
  double sb = sqb / r;
  double ca, sa, cb;
  if (sb > small) {
    ca = x11 / sqb;
    sa = x22 / sqb;
    cb = x33 / r;
  } else {
    sa = 0.0;
    sb = 0.0;
    if (x33 < 0.0) {
      ca = -1.0;
      cb = -1.0;
    } else if (x33 > 0.0) {
      ca = 1.0;
      cb = 1.0;
    } else {
      ca = 0.0;
      cb = 0.0;
    }
  }
  // p / pp exactly as in mpc_rotmat_sp
  mpc_rotmat_sp(xi, xj, rot.sp);
  if (!dorb) return r;
  const double c2a = 2.0 * ca * ca - 1.0;
  const double c2b = 2.0 * cb * cb - 1.0;
  const double s2a = 2.0 * sa * ca;
  const double s2b = 2.0 * sb * cb;
  double (*d)[5] = rot.d;
  d[0][0] = pt5sq3 * c2a * sb * sb;
  d[1][0] = 0.5 * c2a * s2b;
  d[2][0] = -s2a * sb;
  d[3][0] = c2a * (cb * cb + 0.5 * sb * sb);
  d[4][0] = -s2a * cb;
  d[0][1] = pt5sq3 * ca * s2b;
  d[1][1] = ca * c2b;
  d[2][1] = -sa * cb;
  d[3][1] = -0.5 * ca * s2b;
  d[4][1] = sa * sb;
  d[0][2] = cb * cb - 0.5 * sb * sb;
  d[1][2] = -pt5sq3 * s2b;
  d[2][2] = 0.0;
  d[3][2] = pt5sq3 * sb * sb;
  d[4][2] = 0.0;
  d[0][3] = pt5sq3 * sa * s2b;
  d[1][3] = sa * c2b;
  d[2][3] = ca * cb;
  d[3][3] = -0.5 * sa * s2b;
  d[4][3] = -ca * sb;
  d[0][4] = pt5sq3 * s2a * sb * sb;
  d[1][4] = 0.5 * s2a * s2b;
  d[2][4] = c2a * sb;
  d[3][4] = s2a * (cb * cb + 0.5 * sb * sb);
  d[4][4] = c2a * cb;
  // dp(3*(a-1)+b, k, l) = d(k,a)*p(l,b)
  const double (*p)[3] = rot.sp.p;
  for (int k = 0; k < 5; ++k) {
    for (int l = 0; l < 3; ++l) {
      for (int a = 0; a < 5; ++a) {
        for (int bb = 0; bb < 3; ++bb) rot.dp[k][l][3 * a + bb] = d[k][a] * p[l][bb];
      }
    }
  }
  // d_d
  const int pa[10] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};  // component pairs for c = 6..15
  const int pb[10] = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
  for (int k = 0; k < 5; ++k) {
    for (int l = 0; l < 5; ++l)
      for (int c = 0; c < 15; ++c) rot.dd[k][l][c] = 0.0;
    for (int c = 0; c < 5; ++c) rot.dd[k][k][c] = d[k][c] * d[k][c];
    for (int c = 0; c < 10; ++c) rot.dd[k][k][5 + c] = d[k][pa[c]] * d[k][pb[c]];
    for (int l = 0; l < k; ++l) {
      for (int c = 0; c < 5; ++c) rot.dd[k][l][c] = 2.0 * d[k][c] * d[l][c];
      for (int c = 0; c < 10; ++c)
        rot.dd[k][l][5 + c] = d[k][pa[c]] * d[l][pb[c]] + d[k][pb[c]] * d[l][pa[c]];
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// reppd2 (mndod.F90): rep(1..491) from ri (sp part) and rijkl (d part), and
// cored(5..10, 1..2).  r in Bohr.  core1/core2 are the 10-entry cored columns.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void mpc_reppd2(const MozymePairCoreParams &prm, int ni, int nj,
                                                  double r, const double *ri, double *rep,
                                                  double *core1, double *core2) {
  const int ipos[34] = {1, 5, 11, 12, 12, 2, 6, 13, 14, 14, 3, 8, 16, 18, 18, 7, 15,
                        10, 20, 4, 9, 17, 19, 21, 7, 15, 10, 20, 22, 4, 9, 17, 21, 19};
  const int lorb[9] = {0, 1, 1, 1, 2, 2, 2, 2, 2};
  const double ev = prm.ev;
  for (int n = 0; n < 491; ++n) rep[n] = 0.0;
  for (int n = 0; n < 34; ++n) rep[n] = ri[ipos[n] - 1];
  const bool di = prm.natorb[ni - 1] == 9;
  const bool dj = prm.natorb[nj - 1] == 9;
  if (di || dj) {
    const int lasti = di ? 9 : ((ni < 3) ? 1 : 4);
    const int lastk = dj ? 9 : ((nj < 3) ? 1 : 4);
    int ij = 0;
    for (int i = 1; i <= lasti; ++i) {
      const int li = lorb[i - 1];
      for (int j = 1; j <= i; ++j) {
        const bool coul = (i == j);
        const int lj = lorb[j - 1];
        ij = mpc_indexd(i, j);
        for (int k = 1; k <= lastk; ++k) {
          const int lk = lorb[k - 1];
          for (int l = 1; l <= k; ++l) {
            const bool coulomb = coul && (k == l);
            const int ll = lorb[l - 1];
            const int kl = mpc_indexd(k, l);
            const int numb = mpc_ind2(ij, kl);
            if (numb <= 34) continue;
            const int nold = mpc_isym(numb);
            if (nold >= 35) {
              rep[numb - 1] = rep[nold - 1];
            } else if (nold <= -35) {
              rep[numb - 1] = -rep[(-nold) - 1];
            } else if (nold == 0) {
              rep[numb - 1] = mpc_rijkl(prm, ni, nj, ij, kl, li, lj, lk, ll, 0, r) * ev;
              if (prm.l_feather) {
                double point, cnst;
                mpc_to_point(prm, r * prm.a0, &point, &cnst);
                if (coulomb)
                  rep[numb - 1] = rep[numb - 1] * cnst + (1.0 - cnst) * point;
                else
                  rep[numb - 1] = rep[numb - 1] * cnst;
              }
            }
          }
        }
      }
    }
    for (int c = 4; c < 10; ++c) {
      core1[c] = 0.0;
      core2[c] = 0.0;
    }
    // ij keeps the value of the last loop iteration, as in the Fortran
    if (dj) {
      const double tni = prm.tore[ni - 1];
      core2[4] = -mpc_rijkl(prm, ni, nj, ij, mpc_indexd(5, 1), 0, 0, 2, 0, 1, r) * ev * tni;
      core2[5] = -mpc_rijkl(prm, ni, nj, ij, mpc_indexd(5, 2), 0, 0, 2, 1, 1, r) * ev * tni;
      core2[6] = -mpc_rijkl(prm, ni, nj, ij, mpc_indexd(5, 5), 0, 0, 2, 2, 1, r) * ev * tni;
      core2[7] = -mpc_rijkl(prm, ni, nj, ij, mpc_indexd(6, 3), 0, 0, 2, 1, 1, r) * ev * tni;
      core2[8] = -mpc_rijkl(prm, ni, nj, ij, mpc_indexd(6, 6), 0, 0, 2, 2, 1, r) * ev * tni;
      core2[9] = -mpc_rijkl(prm, ni, nj, ij, mpc_indexd(8, 8), 0, 0, 2, 2, 1, r) * ev * tni;
    }
    if (di) {
      const double tnj = prm.tore[nj - 1];
      core1[4] = -mpc_rijkl(prm, ni, nj, mpc_indexd(5, 1), ij, 2, 0, 0, 0, 2, r) * ev * tnj;
      core1[5] = -mpc_rijkl(prm, ni, nj, mpc_indexd(5, 2), ij, 2, 1, 0, 0, 2, r) * ev * tnj;
      core1[6] = -mpc_rijkl(prm, ni, nj, mpc_indexd(5, 5), ij, 2, 2, 0, 0, 2, r) * ev * tnj;
      core1[7] = -mpc_rijkl(prm, ni, nj, mpc_indexd(6, 3), ij, 2, 1, 0, 0, 2, r) * ev * tnj;
      core1[8] = -mpc_rijkl(prm, ni, nj, mpc_indexd(6, 6), ij, 2, 2, 0, 0, 2, r) * ev * tnj;
      core1[9] = -mpc_rijkl(prm, ni, nj, mpc_indexd(8, 8), ij, 2, 2, 0, 0, 2, r) * ev * tnj;
    }
  }
}

// ---------------------------------------------------------------------------
// tx (mndod.F90), full spd version.  v[ij-1][kl-1]: ij = indexd pair on atom
// ni (1..45), kl = indx pair on atom nj (1..limkl).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void mpc_tx_full(const MozymePairRotD &rot, const double *rep,
                                                   int ii, int kk, double v[45][45]) {
  const int met[45] = {1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, 4, 5, 5, 5, 6, 6, 4, 5,
                       5, 5, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 6};
  const int limkl = mpc_indx(kk, kk);
  for (int a = 0; a < 45; ++a)
    for (int b = 0; b < limkl; ++b) v[a][b] = 0.0;
  for (int i1 = 1; i1 <= ii; ++i1) {
    for (int j1 = 1; j1 <= i1; ++j1) {
      const int ij = mpc_indexd(i1, j1);
      double *vij = v[ij - 1];
      for (int k1 = 1; k1 <= kk; ++k1) {
        for (int l1 = 1; l1 <= k1; ++l1) {
          const int kl = mpc_indexd(k1, l1);
          const int nd = mpc_ind2(ij, kl);
          if (nd == 0) continue;
          const double wrepp = rep[nd - 1];
          const int ll = mpc_indx(k1, l1);
          const int mm = met[ll - 1];
          switch (mm) {
            case 1:
              vij[0] = wrepp;
              break;
            case 2: {
              const int k = k1 - 2;  // Fortran k = k1 - 1 (1-based) -> 0-based
              vij[1] = vij[1] + rot.sp.p[k][0] * wrepp;
              vij[3] = vij[3] + rot.sp.p[k][1] * wrepp;
              vij[6] = vij[6] + rot.sp.p[k][2] * wrepp;
              break;
            }
            case 3: {
              const int k = k1 - 2;
              const int l = l1 - 2;
              const double *ppkl = rot.sp.pp[k][l];
              vij[2] = vij[2] + ppkl[0] * wrepp;
              vij[5] = vij[5] + ppkl[1] * wrepp;
              vij[9] = vij[9] + ppkl[2] * wrepp;
              vij[4] = vij[4] + ppkl[3] * wrepp;
              vij[7] = vij[7] + ppkl[4] * wrepp;
              vij[8] = vij[8] + ppkl[5] * wrepp;
              break;
            }
            case 4: {
              const int k = k1 - 5;  // Fortran k = k1 - 4 -> 0-based
              vij[10] = vij[10] + rot.d[k][0] * wrepp;
              vij[15] = vij[15] + rot.d[k][1] * wrepp;
              vij[21] = vij[21] + rot.d[k][2] * wrepp;
              vij[28] = vij[28] + rot.d[k][3] * wrepp;
              vij[36] = vij[36] + rot.d[k][4] * wrepp;
              break;
            }
            case 5: {
              const int k = k1 - 5;
              const int l = l1 - 2;
              const double *dpkl = rot.dp[k][l];
              vij[11] = vij[11] + dpkl[0] * wrepp;
              vij[12] = vij[12] + dpkl[1] * wrepp;
              vij[13] = vij[13] + dpkl[2] * wrepp;
              vij[16] = vij[16] + dpkl[3] * wrepp;
              vij[17] = vij[17] + dpkl[4] * wrepp;
              vij[18] = vij[18] + dpkl[5] * wrepp;
              vij[22] = vij[22] + dpkl[6] * wrepp;
              vij[23] = vij[23] + dpkl[7] * wrepp;
              vij[24] = vij[24] + dpkl[8] * wrepp;
              vij[29] = vij[29] + dpkl[9] * wrepp;
              vij[30] = vij[30] + dpkl[10] * wrepp;
              vij[31] = vij[31] + dpkl[11] * wrepp;
              vij[37] = vij[37] + dpkl[12] * wrepp;
              vij[38] = vij[38] + dpkl[13] * wrepp;
              vij[39] = vij[39] + dpkl[14] * wrepp;
              break;
            }
            default: {  // 6
              const int k = k1 - 5;
              const int l = l1 - 5;
              const double *ddkl = rot.dd[k][l];
              vij[14] = vij[14] + ddkl[0] * wrepp;
              vij[20] = vij[20] + ddkl[1] * wrepp;
              vij[27] = vij[27] + ddkl[2] * wrepp;
              vij[35] = vij[35] + ddkl[3] * wrepp;
              vij[44] = vij[44] + ddkl[4] * wrepp;
              vij[19] = vij[19] + ddkl[5] * wrepp;
              vij[25] = vij[25] + ddkl[6] * wrepp;
              vij[26] = vij[26] + ddkl[7] * wrepp;
              vij[32] = vij[32] + ddkl[8] * wrepp;
              vij[33] = vij[33] + ddkl[9] * wrepp;
              vij[34] = vij[34] + ddkl[10] * wrepp;
              vij[40] = vij[40] + ddkl[11] * wrepp;
              vij[41] = vij[41] + ddkl[12] * wrepp;
              vij[42] = vij[42] + ddkl[13] * wrepp;
              vij[43] = vij[43] + ddkl[14] * wrepp;
              break;
            }
          }
        }
      }
    }
  }
}

// rotatd second step (full spd): ww[(indx(i,j)-1)*limkl + (kl-1)], 1-based
// Fortran ww(indw(i,j)) with indw = (indx(i,j)-1)*limkl + kl.
static __device__ __forceinline__ void mpc_rotate_w_full(const MozymePairRotD &rot,
                                                         const double *rep, int ii, int kk,
                                                         double *ww) {
  const int met[45] = {1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, 4, 5, 5, 5, 6, 6, 4, 5,
                       5, 5, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 6};
  double v[45][45];
  for (int n = 0; n < 2025; ++n) ww[n] = 0.0;
  if (ii * kk <= 0) return;
  const int limkl = mpc_indx(kk, kk);
  mpc_tx_full(rot, rep, ii, kk, v);
#define MPC_INDW(I, J) ((mpc_indx((I), (J)) - 1) * limkl + (kl - 1))
  for (int i1 = 1; i1 <= ii; ++i1) {
    for (int j1 = 1; j1 <= i1; ++j1) {
      const int ij = mpc_indexd(i1, j1);
      const int jj = mpc_indx(i1, j1);
      const int mm = met[jj - 1];
      for (int k = 1; k <= kk; ++k) {
        for (int l = 1; l <= k; ++l) {
          const int kl = mpc_indx(k, l);
          const double wrepp = v[ij - 1][kl - 1];
          if (wrepp == 0.0) continue;  // logv
          switch (mm) {
            case 1:
              ww[MPC_INDW(1, 1)] = wrepp;
              break;
            case 2:
              for (int i = 1; i <= 3; ++i) {
                const int iw = MPC_INDW(i + 1, 1);
                ww[iw] = ww[iw] + rot.sp.p[i1 - 2][i - 1] * wrepp;  // sp(i1-1,i)
              }
              break;
            case 3:
              for (int i = 1; i <= 3; ++i) {
                double cc = rot.sp.pp[i1 - 2][j1 - 2][i - 1];  // pp(i,i1-1,j1-1)
                int iw = MPC_INDW(i + 1, i + 1);
                ww[iw] = ww[iw] + cc * wrepp;
                for (int j = 1; j < i; ++j) {
                  cc = rot.sp.pp[i1 - 2][j1 - 2][i + j];  // pp(1+i+j,...)
                  iw = MPC_INDW(i + 1, j + 1);
                  ww[iw] = ww[iw] + cc * wrepp;
                }
              }
              break;
            case 4:
              for (int i = 1; i <= 5; ++i) {
                const int iw = MPC_INDW(i + 4, 1);
                ww[iw] = ww[iw] + rot.d[i1 - 5][i - 1] * wrepp;  // sd(i1-4,i)
              }
              break;
            case 5:
              for (int i = 1; i <= 5; ++i) {
                for (int j = 1; j <= 3; ++j) {
                  const int iw = MPC_INDW(i + 4, j + 1);
                  const int ij1 = 3 * (i - 1) + j;
                  ww[iw] = ww[iw] + rot.dp[i1 - 5][j1 - 2][ij1 - 1] * wrepp;  // dp(ij1,i1-4,j1-1)
                }
              }
              break;
            default:  // 6
              for (int i = 1; i <= 5; ++i) {
                double cc = rot.dd[i1 - 5][j1 - 5][i - 1];  // d_d(i,i1-4,j1-4)
                int iw = MPC_INDW(i + 4, i + 4);
                ww[iw] = ww[iw] + cc * wrepp;
                for (int j = 1; j < i; ++j) {
                  const int ij1 = mpc_inddd(i, j);
                  cc = rot.dd[i1 - 5][j1 - 5][ij1 - 1];
                  iw = MPC_INDW(i + 4, j + 4);
                  ww[iw] = ww[iw] + cc * wrepp;
                }
              }
              break;
          }
        }
      }
    }
  }
#undef MPC_INDW
}

// ---------------------------------------------------------------------------
// elenuc (mndod.F90), full spd version for one atom with nat orbitals.
//   e[ind1*(ind1+1)/2 + ind2], 0-based local orbital indices, ind2 <= ind1.
//   core[c] = cored(c+1, n), c = 0..9.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void mpc_elenuc_full(const MozymePairRotD &rot, const double *core,
                                                       int nat, double *e) {
  for (int ind1 = 0; ind1 < nat; ++ind1) {
    for (int ind2 = 0; ind2 <= ind1; ++ind2) {
      const int m = (ind1 * (ind1 + 1)) / 2 + ind2;
      double h = 0.0;
      if (ind1 == 0) {
        h = h + core[0];  // (SS/)
      } else if (ind1 < 4) {
        if (ind2 == 0) {
          h = h + rot.sp.p[0][ind1 - 1] * core[1];  // (SP/) sp(1,ind1)*cored(2)
        } else {
          const int ipp = mpc_indpp(ind1, ind2) - 1;  // (PP/)
          h = h + core[2] * rot.sp.pp[0][0][ipp] + core[3] * (rot.sp.pp[1][1][ipp] + rot.sp.pp[2][2][ipp]);
        }
      } else {
        if (ind2 == 0) {
          h = h + rot.d[0][ind1 - 4] * core[4];  // (SD/) sd(1,ind1-3)*cored(5)
        } else if (ind2 < 4) {
          const int idp = mpc_inddp(ind1 - 3, ind2) - 1;  // (PD/)
          h = h + core[5] * rot.dp[0][0][idp] + core[7] * (rot.dp[1][1][idp] + rot.dp[2][2][idp]);
        } else {
          const int idd = mpc_inddd(ind1 - 3, ind2 - 3) - 1;  // (DD/)
          h = h + core[6] * rot.dd[0][0][idd] + core[8] * (rot.dd[1][1][idd] + rot.dd[2][2][idd]) +
              core[9] * (rot.dd[3][3][idd] + rot.dd[4][4][idd]);
        }
      }
      e[m] = h;
    }
  }
}

// ---------------------------------------------------------------------------
// Public entry point for any natorb <= 9 (s, p, d).
//
// Inputs : ni, nj      atomic numbers (1..107)
//          xi, xj      Cartesian coordinates in Angstrom
//          prm         parameter block (same as for the sp version)
// Outputs: w[0..2024]  two-electron integrals in rotate's order
//                      ((i,j) i>=j on ni outer, (k,l) k>=l on nj inner);
//                      *w_count = natorb(ni)(natorb(ni)+1)/2 *
//                                 natorb(nj)(natorb(nj)+1)/2 entries written
//          e1b[0..44]  electron (ni) - core (nj) attraction, packed lower
//                      triangle over natorb(ni) orbitals
//          e2a[0..44]  electron (nj) - core (ni) attraction
//          *enuc       core-core repulsion (eV)
// Returns false (outputs untouched) for atomic numbers outside 1..107 or
// natorb > 9.  For rij^2 < 2e-5 A^2 all outputs are zeroed, *w_count = 0 and
// true is returned (rotate zeroes w but does not advance kr).
// Local storage: about 40 KB per call (ww, v, rep, rotation matrices).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ bool mozyme_pair_core_dev(
    int ni, int nj, const double *xi, const double *xj, const MozymePairCoreParams &prm,
    double *w, int *w_count, double *e1b, double *e2a, double *enuc) {
  if (ni < 1 || ni > 107 || nj < 1 || nj > 107) return false;
  const int li = prm.natorb[ni - 1];
  const int lj = prm.natorb[nj - 1];
  if (li > 9 || lj > 9 || li < 0 || lj < 0) return false;

  for (int i = 0; i < 45; ++i) {
    e1b[i] = 0.0;
    e2a[i] = 0.0;
  }
  for (int i = 0; i < 2025; ++i) w[i] = 0.0;
  *w_count = 0;
  *enuc = 0.0;

  // rotate: small-rij exit on the squared distance (Angstrom^2)
  {
    const double x0 = xi[0] - xj[0];
    const double x1 = xi[1] - xj[1];
    const double x2 = xi[2] - xj[2];
    const double rij2 = x0 * x0 + x1 * x1 + x2 * x2;
    if (rij2 < 0.00002) return true;
  }
  const bool dorb = (li == 9) || (lj == 9);

  // rotatd: rotmat(nj, ni, ci, cj, r)
  MozymePairRotD rot;
  const double rij = mpc_rotmat_full(xi, xj, dorb, rot);

  // reppd: 22 local integrals + gab
  double ri[22];
  double gab;
  mpc_reppd(prm, ni, nj, rij, ri, &gab);

  // spcore in Bohr -> cored(1..4,:); reppd2 -> rep(1..491), cored(5..10,:)
  const double r = rij / prm.a0;
  double cored1[10], cored2[10];
  mpc_spcore(prm, ni, nj, r, cored1, cored2);
  for (int c = 4; c < 10; ++c) {
    cored1[c] = 0.0;
    cored2[c] = 0.0;
  }
  double point, cnst;
  if (prm.l_feather) {
    mpc_to_point(prm, rij, &point, &cnst);
  } else {
    cnst = 1.0;
    point = 0.0;
  }
  double rep[491];
  mpc_reppd2(prm, ni, nj, r, ri, rep, cored1, cored2);
  point = -(prm.ev / r) * prm.tore[nj - 1];
  cored1[0] = cored1[0] * cnst + (1.0 - cnst) * point;
  cored1[1] = cored1[1] * cnst;
  cored1[2] = cored1[2] * cnst + (1.0 - cnst) * point;
  cored1[3] = cored1[3] * cnst + (1.0 - cnst) * point;
  cored1[4] = cored1[4] * cnst;
  cored1[5] = cored1[5] * cnst;
  cored1[6] = cored1[6] * cnst + (1.0 - cnst) * point;
  cored1[7] = cored1[7] * cnst;
  cored1[8] = cored1[8] * cnst + (1.0 - cnst) * point;
  cored1[9] = cored1[9] * cnst + (1.0 - cnst) * point;
  point = -(prm.ev / r) * prm.tore[ni - 1];
  cored2[0] = cored2[0] * cnst + (1.0 - cnst) * point;
  cored2[1] = cored2[1] * cnst;
  cored2[2] = cored2[2] * cnst + (1.0 - cnst) * point;
  cored2[3] = cored2[3] * cnst + (1.0 - cnst) * point;
  cored2[4] = cored2[4] * cnst;
  cored2[5] = cored2[5] * cnst;
  cored2[6] = cored2[6] * cnst + (1.0 - cnst) * point;
  cored2[7] = cored2[7] * cnst;
  cored2[8] = cored2[8] * cnst + (1.0 - cnst) * point;
  cored2[9] = cored2[9] * cnst + (1.0 - cnst) * point;

  // two-electron integrals in the molecular frame, ww(2025)
  double ww[2025];
  mpc_rotate_w_full(rot, rep, li, lj, ww);

  // PM7 "d"-orbital balance block of rotatd, complete.  Fortran 1-based
  // ww(n) -> ww[n-1].
  if (prm.method_pm7) {
    if (prm.iod[ni - 1] > 0) {
      double sum = 0.0;
      int k;
      if (lj == 9)
        k = 45;
      else if (lj == 4)
        k = 10;
      else
        k = 1;
      if (lj > 1) {
        for (int i = 5; i <= 9; ++i) {  // "p" on nj with "d" on ni
          const int j = k * ((i * (i + 1)) / 2 - 1);
          sum = sum + ww[j + 3 - 1] + ww[j + 6 - 1] + ww[j + 10 - 1];
        }
        sum = (ww[0] - sum / 15.0);
        for (int i = 5; i <= 9; ++i) {
          const int j = k * ((i * (i + 1)) / 2 - 1);
          for (int l = 2; l <= 4; ++l) ww[(l * (l + 1)) / 2 + j - 1] = ww[(l * (l + 1)) / 2 + j - 1] + sum;
        }
        sum = cored1[0] - (cored1[2] + 2.0 * cored1[3]) / 3.0;
        cored1[2] = cored1[2] + sum;
        cored1[3] = cored1[3] + sum;
      }
      sum = 0.0;
      for (int i = 5; i <= 9; ++i) sum = sum + ww[k * ((i * (i + 1)) / 2 - 1) + 1 - 1];
      sum = (ww[0] - sum / 5.0);
      for (int i = 5; i <= 9; ++i) {  // "s" on nj with "d" on ni
        const int n = k * ((i * (i + 1)) / 2 - 1) + 1 - 1;
        ww[n] = ww[n] + sum;
      }
      sum = cored1[0] - (cored1[6] + 2.0 * cored1[8] + 2.0 * cored1[9]) / 5.0;
      cored1[6] = cored1[6] + sum;
      cored1[8] = cored1[8] + sum;
      cored1[9] = cored1[9] + sum;
    }
    if (prm.iod[nj - 1] > 0) {
      double sum = 0.0;
      if (prm.iod[ni - 1] > 0) {
        for (int i = 5; i <= 9; ++i) {  // "d" with "d"
          const int j = 45 * ((i * (i + 1)) / 2 - 1);
          sum = sum + (ww[j + 15 - 1] + ww[j + 21 - 1] + ww[j + 28 - 1] + ww[j + 36 - 1] + ww[j + 45 - 1]);
        }
        sum = (ww[0] - sum / 25.0);
        for (int i = 5; i <= 9; ++i) {
          const int j = 45 * ((i * (i + 1)) / 2 - 1);
          for (int k = 5; k <= 9; ++k) ww[(k * (k + 1)) / 2 + j - 1] = ww[(k * (k + 1)) / 2 + j - 1] + sum;
        }
      }
      if (li > 1) {
        sum = 0.0;
        for (int i = 2; i <= 4; ++i) {  // "p" on ni with "d" on nj
          const int j = 45 * ((i * (i + 1)) / 2 - 1);
          sum = sum + (ww[j + 15 - 1] + ww[j + 21 - 1] + ww[j + 28 - 1] + ww[j + 36 - 1] + ww[j + 45 - 1]);
        }
        sum = (ww[0] - sum / 15.0);
        for (int i = 2; i <= 4; ++i) {
          const int j = 45 * ((i * (i + 1)) / 2 - 1);
          for (int k = 5; k <= 9; ++k) ww[(k * (k + 1)) / 2 + j - 1] = ww[(k * (k + 1)) / 2 + j - 1] + sum;
        }
        sum = cored2[0] - (cored2[2] + 2.0 * cored2[3]) / 3.0;
        cored2[2] = cored2[2] + sum;
        cored2[3] = cored2[3] + sum;
      }
      sum = (ww[0] - (ww[15 - 1] + ww[21 - 1] + ww[28 - 1] + ww[36 - 1] + ww[45 - 1]) / 5.0);
      for (int k = 5; k <= 9; ++k) {  // "s" on ni with "d" on nj
        ww[(k * (k + 1)) / 2 - 1] = ww[(k * (k + 1)) / 2 - 1] + sum;
      }
      sum = (cored2[0] - (cored2[6] + 2.0 * cored2[8] + 2.0 * cored2[9]) / 5.0);
      cored2[6] = cored2[6] + sum;
      cored2[8] = cored2[8] + sum;
      cored2[9] = cored2[9] + sum;
    }
  }

  // w2mat: w(l) = ww(kl, ij), ij outer, kl inner == ww's linear storage
  const int limij = (li * (li + 1)) / 2;
  const int limkl = (lj * (lj + 1)) / 2;
  const int istep = limij * limkl;
  for (int n = 0; n < istep; ++n) w[n] = ww[n];
  *w_count = istep;

  // elenuc
  mpc_elenuc_full(rot, cored1, li, e1b);
  mpc_elenuc_full(rot, cored2, lj, e2a);

  // ccrep (distance passed in Bohr, converted back inside)
  *enuc = mpc_ccrep(prm, ni, nj, r, gab);
  return true;
}

#endif  // MOPAC_GPU_MOZYME_PAIR_CORE_CUH

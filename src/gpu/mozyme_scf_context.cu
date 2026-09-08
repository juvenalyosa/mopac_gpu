#include <cstddef>
#include <climits>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>
#include <new>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define MOPAC_UNUSED_SYMBOL __attribute__((unused))
#else
#define MOPAC_UNUSED_SYMBOL
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cooperative_groups.h>

extern "C" int mopac_cuda_mozyme_sparse_fock_run_device(
    int mpack, const double *ptot_dev, const double *qe_dev, double *f_dev,
    double *wall_ms);

extern "C" int mopac_cuda_mozyme_sparse_fock_run_device_plan(
    int plan_id, int mpack, const double *ptot_dev, const double *qe_dev,
    double *f_dev, double *wall_ms);

extern "C" int mopac_cuda_mozyme_sparse_fock_run_device_plan_guarded(
    int plan_id, int mpack, const double *ptot_dev, const double *qe_dev,
    double *f_dev, const int *guard_ints, int guard_slot, int guard_continue,
    double *wall_ms);

extern "C" int mopac_cuda_mozyme_sparse_fock_run_device_plan_guarded_resident(
    int plan_id, int mpack, const double *ptot_dev, const double *qe_dev,
    double *f_dev, const int *guard_ints, int guard_slot, int guard_continue,
    int synchronize, double *wall_ms);
#endif

#ifdef __CUDACC__
__device__ bool mozyme_tidy_compact_device(
    int nmos, int natoms, int norbs, int n01, int n02, double thresh, int *nc,
    int *ic, double *c, int *nnc, int *ncmo, const int *iorbs, int *iused,
    int *ln_out, int *mn_out, int *status) {
  int ln = 0;
  int mn = 0;
  int previous_count = 0;
  for (int i = 0; i < nmos; ++i) {
    const int old_nnc = nnc[i];
    const int old_nc = nc[i];
    int mm = ncmo[i];
    ncmo[i] = mn;
    nnc[i] = (i == 0) ? 0 : nnc[i - 1] + previous_count;
    int kept_atoms = 0;
    if (old_nc < 0 || old_nnc < 0 || old_nnc + old_nc > n01 || mm < 0 ||
        mm > n02) {
      *status = -502;
      return false;
    }
    for (int local_atom = 0; local_atom < old_nc; ++local_atom) {
      const int ic_slot = old_nnc + local_atom;
      const int atom = ic[ic_slot];
      if (atom < 1 || atom > natoms) {
        *status = -503;
        return false;
      }
      const int norb = iorbs[atom - 1];
      if (norb <= 0 || norb > norbs || mm + norb > n02) {
        *status = -504;
        return false;
      }
      double sum = 0.0;
      for (int orb = 0; orb < norb; ++orb) {
        const double value = c[mm + orb];
        sum += value * value;
      }
      if (sum > thresh) {
        if (ln >= n01 || mn + norb > n02) {
          *status = -505;
          return false;
        }
        ic[ln] = atom;
        for (int orb = 0; orb < norb; ++orb) {
          c[mn + orb] = c[mm + orb];
        }
        ++ln;
        mn += norb;
        ++kept_atoms;
      }
      mm += norb;
    }
    nc[i] = kept_atoms;
    iused[i] = mn;
    previous_count = kept_atoms;
  }
  *ln_out = ln;
  *mn_out = mn;
  return true;
}

__device__ bool mozyme_tidy_compct_device(
    int *nncnew, int *ncnew, int *ncmnew, int itop, int *nc, int *ic,
    int *iws, int n01, double *c, int n02, int nmos, int idone, int *lb,
    int *mb, int icref, int inref, int *status) {
  int inic = icref;
  int inco = inref;
  int inew = itop - 1;
  int ii = idone;
  for (int src = idone - 1; src >= 0; --src) {
    if (nc[src] != 0) {
      --ii;
      ++inew;
      if (ii < 0 || ii >= nmos || inew < 0 || inew >= nmos) {
        *status = -518;
        return false;
      }
      const int natom = nc[src];
      const int ncoef = iws[src];
      if (natom < 0 || ncoef < 0) {
        *status = -519;
        return false;
      }
      inic -= natom;
      inco -= ncoef;
      const int ioic = nncnew[inew];
      const int ioco = ncmnew[inew];
      if (inic < 0 || inic + natom > n01 || ioic < 0 ||
          ioic + natom > n01) {
        *status = -520;
        return false;
      }
      if (inco < 0 || inco + ncoef > n02 || ioco < 0 ||
          ioco + ncoef > n02) {
        *status = -521;
        return false;
      }
      for (int n = natom; n >= 1; --n) ic[inic + n - 1] = ic[ioic + n - 1];
      for (int n = ncoef; n >= 1; --n) c[inco + n - 1] = c[ioco + n - 1];
      ncnew[inew] = natom;
      nncnew[inew] = inic;
      ncmnew[inew] = inco;
      nc[ii] = natom;
      iws[ii] = ncoef;
      if (inew == nmos - 1) break;
    }
  }
  for (int dst = ii - 1; dst >= 0; --dst) nc[dst] = 0;
  *lb = inic;
  *mb = inco;
  return true;
}

__device__ bool mozyme_tidy_selected_lmo_device(
    int lmo, int n01, int nmos, int numred, const int *nc, const int *ic,
    const int *nnc, const int *jopt, int *status) {
  if (lmo < 0 || lmo >= nmos) {
    *status = -512;
    return false;
  }
  if (nc[lmo] <= 0) return false;
  const int first = nnc[lmo];
  if (first < 0 || first >= n01) {
    *status = -513;
    return false;
  }
  const int atom1 = ic[first];
  int atom2 = 0;
  if (nc[lmo] >= 2) {
    if (first + 1 >= n01) {
      *status = -513;
      return false;
    }
    atom2 = ic[first + 1];
  }
  for (int i = 0; i < numred; ++i) {
    if (atom1 == jopt[i] || atom2 == jopt[i]) return true;
  }
  return false;
}

__device__ bool mozyme_tidy_selmos_device(
    int nmos, int n01, int n02, int ln, int mn, int numred, int mode, int *nc,
    int *ic, double *c, int *nnc, int *ncmo, int *iws, const int *jopt,
    int *ncnew, int *ncmnew, int *nncnew, int *selected_out, int *status) {
  if (numred < 0 || numred > n01 || mode < 1 || mode > 2) {
    *status = -514;
    return false;
  }

  const int c_shift = n02 - mn;
  for (int i = mn - 1; i >= 0; --i) c[i + c_shift] = c[i];
  for (int i = 0; i < nmos; ++i) ncmo[i] += c_shift;

  const int ic_shift = n01 - ln;
  for (int i = ln - 1; i >= 0; --i) ic[i + ic_shift] = ic[i];
  for (int i = 0; i < nmos; ++i) nnc[i] += ic_shift;

  int jbot = 0;
  int nbot = 0;
  int jtop = n02 - mn - 1;
  int ntop = n01 - ln - 1;
  int ibot = 0;
  int itop = nmos;
  int nreal = 0;

  for (int i = nmos - 1; i >= 1; --i) iws[i] -= iws[i - 1];

  for (int i = 0; i < nmos; ++i) {
    const int atom_offset = nnc[i];
    const bool selected = mozyme_tidy_selected_lmo_device(
        i, n01, nmos, numred, nc, ic, nnc, jopt, status);
    if (*status != 0) return false;
    if (!selected) {
      --itop;
      if (itop < ibot) {
        *status = -515;
        return false;
      }
      nncnew[itop] = nnc[i];
      ncnew[itop] = nc[i];
      ncmnew[itop] = ncmo[i];
      continue;
    }

    if (nbot + nc[i] > ntop || jbot + iws[i] > jtop) {
      if (!mozyme_tidy_compct_device(nncnew, ncnew, ncmnew, itop, nc, ic, iws,
                                     n01, c, n02, nmos, i, &ntop, &jtop,
                                     atom_offset, ncmo[i], status)) {
        return false;
      }
    }
    if (nbot + nc[i] > ntop || jbot + iws[i] > jtop) {
      *status = -516;
      return false;
    }

    if (ibot >= itop) {
      *status = -517;
      return false;
    }
    nncnew[ibot] = nbot;
    ncnew[ibot] = nc[i];
    ncmnew[ibot] = jbot;
    for (int n = 1; n <= nc[i]; ++n) ic[nbot + n - 1] = ic[atom_offset + n - 1];
    nbot += nc[i];
    nc[i] = 0;
    const int coeff_offset = ncmo[i];
    const int ncoefs = iws[i];
    for (int n = 1; n <= ncoefs; ++n) c[jbot + n - 1] = c[coeff_offset + n - 1];
    jbot += ncoefs;
    ++ibot;
    ++nreal;
  }

  for (int i = 0; i < ibot; ++i) {
    nc[i] = ncnew[i];
    nnc[i] = nncnew[i];
    ncmo[i] = ncmnew[i];
  }
  int j = nmos;
  for (int i = itop; i < nmos; ++i) {
    --j;
    nc[i] = ncnew[j];
    nnc[i] = nncnew[j];
    ncmo[i] = ncmnew[j];
  }
  *selected_out = nreal;
  return true;
}

__device__ bool mozyme_tidy_space_device(
    int nmos, int natoms, int norbs, int n01, int n02, int ln, int mn, int *nc,
    int *ic, double *c, int *nnc, int *ncmo, const int *iused, int *status) {

  const int ispace_initial = (n01 - ln) / nmos;
  const int jspace_initial = (n02 - mn) / nmos;
  int ispace = ispace_initial;
  int jspace = jspace_initial;
  const int average_ic = n01 / nmos;
  const int min_tidy_slack = (average_ic / 5 > 20) ? average_ic / 5 : 20;
  if (average_ic + ispace < natoms && ispace < min_tidy_slack) {
    *status = -506;
    return false;
  }
  if (ispace > natoms && jspace > nmos) {
    ispace = 0;
    jspace = 0;
  }

  int jtop = n01;
  int mtop = n02;
  int jsav = n01 - ln - ispace * nmos;
  int msav = n02 - mn - jspace * nmos;
  for (int i = nmos - 1; i >= 1; --i) {
    const int atom_capacity = nc[i] + ispace;
    int atoms_to_place = (atom_capacity < natoms) ? atom_capacity : natoms;
    if (atoms_to_place < nc[i] + ispace) {
      jsav += nc[i] + ispace - atoms_to_place;
      jtop -= atoms_to_place;
    } else {
      const int jsav_nonnegative = (jsav > 0) ? jsav : 0;
      const int jdash_candidate = jsav_nonnegative + atoms_to_place;
      const int jdash = (jdash_candidate < natoms) ? jdash_candidate : natoms;
      jtop -= jdash;
      jsav = jsav - jdash + atoms_to_place;
    }
    if (jtop < 0 || jtop > n01) {
      *status = -507;
      return false;
    }
    const int source_nnc = nnc[i];
    nnc[i] = jtop;
    for (int k = nc[i]; k >= 1; --k) {
      if (source_nnc + k - 1 < 0 || source_nnc + k - 1 >= n01 ||
          jtop + k - 1 < 0 || jtop + k - 1 >= n01) {
        *status = -508;
        return false;
      }
      ic[jtop + k - 1] = ic[source_nnc + k - 1];
    }

    const int coeff_count = iused[i] - iused[i - 1];
    if (coeff_count < 0) {
      *status = -509;
      return false;
    }
    const int coeff_capacity = coeff_count + jspace;
    int coeffs_to_place = (coeff_capacity < norbs) ? coeff_capacity : norbs;
    if (coeffs_to_place < coeff_count + jspace) {
      msav += coeff_count + jspace - coeffs_to_place;
      mtop -= coeffs_to_place;
    } else {
      const int msav_nonnegative = (msav > 0) ? msav : 0;
      const int mdash_candidate = msav_nonnegative + coeffs_to_place;
      const int mdash = (mdash_candidate < norbs) ? mdash_candidate : norbs;
      mtop -= mdash;
      msav = msav - mdash + coeffs_to_place;
    }
    if (mtop < 0 || mtop > n02) {
      *status = -510;
      return false;
    }
    ncmo[i] = mtop;
    for (int k = coeff_count; k >= 1; --k) {
      if (iused[i - 1] + k - 1 < 0 || iused[i - 1] + k - 1 >= n02 ||
          mtop + k - 1 < 0 || mtop + k - 1 >= n02) {
        *status = -511;
        return false;
      }
      c[mtop + k - 1] = c[iused[i - 1] + k - 1];
    }
  }
  return true;
}

__device__ void mozyme_tidy_run_device(
    int nmos, int natoms, int norbs, int n01, int n02, double thresh,
    int use_selmos, int numred, int mode, int *nc, int *ic, double *c,
    int *nnc, int *ncmo, const int *iorbs, const int *jopt, int *iused,
    int *ncnew, int *ncmnew, int *nncnew, int *result, int *ok_slot) {
  result[0] = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = -1;
  if (nmos == 0) return;
  if (nmos < 0 || natoms <= 0 || norbs <= 0 || n01 <= 0 || n02 <= 0 ||
      !(thresh >= 0.0) || (use_selmos != 0 && numred > 0 && !jopt)) {
    result[0] = -501;
    if (ok_slot) *ok_slot = 0;
    return;
  }

  int ln = 0;
  int mn = 0;
  int status = 0;
  if (!mozyme_tidy_compact_device(nmos, natoms, norbs, n01, n02, thresh, nc,
                                  ic, c, nnc, ncmo, iorbs, iused, &ln, &mn,
                                  &status)) {
    result[0] = status;
    if (ok_slot) *ok_slot = 0;
    return;
  }
  if (use_selmos != 0) {
    int selected = 0;
    if (!mozyme_tidy_selmos_device(nmos, n01, n02, ln, mn, numred, mode, nc,
                                   ic, c, nnc, ncmo, iused, jopt, ncnew,
                                   ncmnew, nncnew, &selected, &status)) {
      result[0] = status;
      if (ok_slot) *ok_slot = 0;
      return;
    }
    result[3] = selected;
    if (!mozyme_tidy_compact_device(nmos, natoms, norbs, n01, n02, thresh, nc,
                                    ic, c, nnc, ncmo, iorbs, iused, &ln, &mn,
                                    &status)) {
      result[0] = status;
      if (ok_slot) *ok_slot = 0;
      return;
    }
  }
  if (!mozyme_tidy_space_device(nmos, natoms, norbs, n01, n02, ln, mn, nc, ic,
                                c, nnc, ncmo, iused, &status)) {
    result[0] = status;
    if (ok_slot) *ok_slot = 0;
    return;
  }
  result[1] = ln;
  result[2] = mn;
}

__global__ void mozyme_tidy_kernel(
    int nmos, int natoms, int norbs, int n01, int n02, double thresh,
    int use_selmos, int numred, int mode, int *nc, int *ic, double *c,
    int *nnc, int *ncmo, const int *iorbs, const int *jopt, int *iused,
    int *ncnew, int *ncmnew, int *nncnew, int *result) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  mozyme_tidy_run_device(nmos, natoms, norbs, n01, n02, thresh, use_selmos,
                         numred, mode, nc, ic, c, nnc, ncmo, iorbs, jopt,
                         iused, ncnew, ncmnew, nncnew, result, nullptr);
}

__global__ void mozyme_reorth_kernel(
    int natoms, int norbs, int nocc, int nvir, int cocc_dim, int icocc_dim,
    int cvir_dim, int icvir_dim, double thresh, double *cocc, int *icocc,
    int *ncf, const int *nncf, const int *ncocc, double *cvir, int *icvir,
    int *nce, const int *nnce, const int *ncvir, const int *iorbs,
    const int *nfirst, double *ws, int *latom, int *iused, int *status,
    double *sumtot_out);
#endif

namespace {

constexpr int kMozymeScfAbiVersion = 33;
constexpr int kMozymeScfSuccess = 0;
constexpr int kMozymeScfNotReady = -1;
constexpr int kMozymeScfBadArgument = -2;
constexpr int kMozymeScfUnsupported = -3;
constexpr int kMozymeScfCpuBoundary = -4;

constexpr int kMozymeScfStageUpload = 1;
constexpr int kMozymeScfStageEimp = 2;
constexpr int kMozymeScfStageDiagg = 4;
constexpr int kMozymeScfStageDensity = 8;
constexpr int kMozymeScfStageFock = 16;
constexpr int kMozymeScfStageCnvgz = 32;
constexpr int kMozymeScfStageHelecz = 64;
constexpr int kMozymeScfStageIsitsc = 128;
constexpr int kMozymeScfStageAddhb = 256;
constexpr int kMozymeScfStageCheck = 512;
constexpr int kMozymeScfStageFull =
    kMozymeScfStageUpload | kMozymeScfStageEimp | kMozymeScfStageDiagg |
    kMozymeScfStageDensity | kMozymeScfStageAddhb |
    kMozymeScfStageFock | kMozymeScfStageCnvgz | kMozymeScfStageHelecz |
    kMozymeScfStageIsitsc | kMozymeScfStageCheck;
constexpr int kResidentStageSlotUpload = 0;
constexpr int kResidentStageSlotEimp = 1;
constexpr int kResidentStageSlotDiagg = 2;
constexpr int kResidentStageSlotDensity = 3;
constexpr int kResidentStageSlotFock = 4;
constexpr int kResidentStageSlotCnvgz = 5;
constexpr int kResidentStageSlotHelecz = 6;
constexpr int kResidentStageSlotIsitsc = 7;
constexpr int kResidentStageSlotAddhb = 8;
constexpr int kResidentStageSlotCheck = 9;
constexpr int kResidentStageSlotCount = 10;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoScalarSolvEnergy = 0;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoScalarEdiel = 1;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoScalarCount = 2;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgRho = 0;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgPq = 1;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgNorm = 2;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgEdiel = 3;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgS1 = 4;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgRhoOld = 5;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgAlpha = 6;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgBeta = 7;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgTargetTol = 8;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgLastResidual = 9;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgScalarCount = 10;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgActive = 0;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgCompletedIterations = 1;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgBreakdown = 2;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgMatvecCalls = 3;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoCgIntCount = 4;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusFockCalls = 0;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusMatvecCalls = 1;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusCgIterations = 2;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusControlResident = 3;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusCgConverged = 4;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusCgBreakdown = 5;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusHostSyncs = 6;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusNewSurface = 7;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusIntCount = 8;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusCurrentTol = 0;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusTargetTol = 1;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusLastResidual = 2;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusSolvEnergy = 3;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusEdiel = 4;
MOPAC_UNUSED_SYMBOL constexpr int kCosmoStatusDoubleCount = 5;
constexpr int kMozymeScfPlsSupervisorLastIter = 10;
constexpr int kMozymeScfFlagInitialSetup = 1;
constexpr int kMozymeScfFlagFinalReorth = 2;
constexpr int kMozymeFockPlanFull = 0;
constexpr int kMozymeFockPlanPartial = 1;
constexpr int kMozymeFockPlanFullMask = 1;
constexpr int kMozymeFockPlanPartialMask = 2;
constexpr int kDiaggIntOk = 0;
constexpr int kDiaggIntNij = 1;
constexpr int kDiaggIntIjc = 2;
constexpr int kDiaggIntNf = 3;
constexpr int kDiaggIntNrej = 4;
constexpr int kDiaggIntRetry = 5;
constexpr int kDiaggIntNextNrej0 = 6;
constexpr int kDiaggIntNextNrej1 = 7;
constexpr int kDiaggIntCount = 8;
constexpr int kDiaggDoubleSumt = 0;
constexpr int kDiaggDoubleTiny = 1;
constexpr int kDiaggDoubleFref = 2;
constexpr int kDiaggDoubleOldlim = 3;
constexpr int kDiaggDoubleSafety = 4;
constexpr int kDiaggDoubleSumb = 5;
constexpr int kDiaggDoubleRotateTiny = 6;
constexpr int kDiaggDoubleBiglim = 7;
constexpr int kDiaggDoubleCount = 8;
constexpr int kAddhbIntOk = 0;
constexpr int kAddhbIntNij = 1;
constexpr int kAddhbIntNrej = 2;
constexpr int kAddhbIntDue = 3;
constexpr int kAddhbIntApplied = 4;
constexpr int kAddhbIntNextNhb = 5;
constexpr int kAddhbIntNextIdiagg = 6;
constexpr int kAddhbIntRetry = 7;
constexpr int kAddhbIntNextNrej0 = 8;
constexpr int kAddhbIntNextNrej1 = 9;
constexpr int kAddhbIntCount = 10;
constexpr int kAddhbDoubleSumb = 0;
constexpr int kAddhbDoubleCutoff = 1;
constexpr int kAddhbDoubleRotateTiny = 2;
constexpr int kAddhbDoubleBiglim = 3;
constexpr int kAddhbDoubleNextTiny = 4;
constexpr int kAddhbDoubleCount = 5;
constexpr int kIsitscIntIemin = 0;
constexpr int kIsitscIntIemax = 1;
constexpr int kIsitscIntScf1 = 2;
constexpr int kIsitscIntOkscf = 3;
constexpr int kIsitscIntIscf = 4;
constexpr int kIsitscIntValid = 5;
constexpr int kIsitscIntCount = 6;
constexpr int kIsitscEnergyScf = 0;
constexpr int kIsitscEnergyDelta = 1;
constexpr int kIsitscDoubleCount = 2;
constexpr int kCheckIntOccBad = 0;
constexpr int kCheckIntVirBad = 1;
constexpr int kCheckIntOk = 2;
constexpr int kCheckIntCount = 3;
constexpr int kCheckDoubleOccError = 0;
constexpr int kCheckDoubleVirError = 1;
constexpr int kCheckDoubleCount = 2;
constexpr int kHeleczIntOk = 0;
constexpr int kHeleczIntCount = 1;
constexpr int kFinalReorthStatusEnergyTotal = 0;
constexpr int kFinalReorthStatusEnergyScf = 1;
constexpr int kFinalReorthStatusEnergyDelta = 2;
constexpr int kFinalReorthStatusDoubleCount = 3;
constexpr int kResidentStageCompleted = 0;
constexpr int kResidentStageRequired = 1;
constexpr int kResidentStageMissing = 2;
constexpr int kResidentStageCode = 3;
constexpr int kResidentStageIntCount = 4;
constexpr int kCnvgzPmax = 0;
constexpr int kCnvgzSumsq = 1;
constexpr int kCnvgzFaca = 2;
constexpr int kCnvgzFacb = 3;
constexpr int kCnvgzFactor = 4;
constexpr int kCnvgzDensityRms = 5;
constexpr int kCnvgzControlCount = 6;
constexpr int kCnvgzIntOk = 0;
constexpr int kCnvgzIntActiveCalls = 1;
constexpr int kCnvgzIntNoopCalls = 2;
constexpr int kCnvgzIntCount = 3;
constexpr int kFockIntOk = 0;
constexpr int kFockIntCount = 1;
constexpr int kResidentDecisionContinue = 0;
constexpr int kResidentDecisionComplete = 1;
constexpr int kResidentDecisionCpuBoundary = 2;
constexpr int kResidentDecisionIterationExhausted = 3;
constexpr int kResidentDecisionPlsRestart = 4;
constexpr int kResidentDecisionStageFailed = 5;
constexpr int kResidentControlDecision = 0;
constexpr int kResidentControlDiaggMode = 1;
constexpr int kResidentControlNhb = 2;
constexpr int kResidentControlDiaggNf = 3;
constexpr int kResidentControlNrej0 = 4;
constexpr int kResidentControlNrej1 = 5;
constexpr int kResidentControlIemin = 6;
constexpr int kResidentControlIemax = 7;
constexpr int kResidentControlScf1 = 8;
constexpr int kResidentControlCurrentIter = 9;
constexpr int kResidentControlAddhbDue = 10;
constexpr int kResidentControlUseThreePoint = 11;
constexpr int kResidentControlLstart = 12;
constexpr int kResidentControlPlsRestartRequired = 13;
constexpr int kResidentControlPlsCalls = 14;
constexpr int kResidentControlPlsHistoryCount = 15;
constexpr int kResidentControlPlsRestartResetCalls = 16;
constexpr int kResidentControlPlsRestartDone = 17;
constexpr int kResidentControlIntCount = 18;
constexpr int kResidentControlDiaggFref = 0;
constexpr int kResidentControlDiaggOldlim = 1;
constexpr int kResidentControlDiaggSafety = 2;
constexpr int kResidentControlOvmax = 3;
constexpr int kResidentControlPreviousEscf = 4;
constexpr int kResidentControlShift = 5;
constexpr int kResidentControlPlsOvmaxDelta = 6;
constexpr int kResidentControlPlsEnergyDelta = 7;
constexpr int kResidentControlDoubleCount = 8;
constexpr int kPlsLoopLimit = 6;
constexpr int kPlsIntLoop = 0;
constexpr int kPlsIntFault = 1;
constexpr int kPlsIntCalls = 2;
constexpr int kPlsIntRestartDone = 3;
constexpr int kPlsIntCount = 4;
constexpr int kPlsDoubleOvmaxOld = 0;
constexpr int kPlsDoubleEscfOld = 1;
constexpr int kPlsDoubleOvmaxHistory = 2;
constexpr int kPlsDoubleEscfHistory = kPlsDoubleOvmaxHistory + kPlsLoopLimit;
constexpr int kPlsDoubleLastOvmaxDelta =
    kPlsDoubleEscfHistory + kPlsLoopLimit;
constexpr int kPlsDoubleLastEnergyDelta = kPlsDoubleLastOvmaxDelta + 1;
constexpr int kPlsDoubleCount = kPlsDoubleLastEnergyDelta + 1;

#ifndef __CUDACC__
template <typename... Args>
inline void ignore_no_cuda_only(const Args &...) {}

MOPAC_UNUSED_SYMBOL constexpr int kNoCudaConstantReferences[] = {
    kMozymeScfAbiVersion,
    kMozymeScfSuccess,
    kMozymeScfNotReady,
    kMozymeScfBadArgument,
    kMozymeScfUnsupported,
    kMozymeScfCpuBoundary,
    kMozymeScfStageUpload,
    kMozymeScfStageEimp,
    kMozymeScfStageDiagg,
    kMozymeScfStageDensity,
    kMozymeScfStageFock,
    kMozymeScfStageCnvgz,
    kMozymeScfStageHelecz,
    kMozymeScfStageIsitsc,
    kMozymeScfStageAddhb,
    kMozymeScfStageCheck,
    kMozymeScfStageFull,
    kResidentStageSlotUpload,
    kResidentStageSlotEimp,
    kResidentStageSlotDiagg,
    kResidentStageSlotDensity,
    kResidentStageSlotFock,
    kResidentStageSlotCnvgz,
    kResidentStageSlotHelecz,
    kResidentStageSlotIsitsc,
    kResidentStageSlotAddhb,
    kResidentStageSlotCheck,
    kResidentStageSlotCount,
    kMozymeScfPlsSupervisorLastIter,
    kMozymeScfFlagInitialSetup,
    kMozymeScfFlagFinalReorth,
    kMozymeFockPlanFull,
    kMozymeFockPlanPartial,
    kDiaggIntOk,
    kDiaggIntNij,
    kDiaggIntIjc,
    kDiaggIntNf,
    kDiaggIntNrej,
    kDiaggIntRetry,
    kDiaggIntNextNrej0,
    kDiaggIntNextNrej1,
    kDiaggIntCount,
    kDiaggDoubleSumt,
    kDiaggDoubleTiny,
    kDiaggDoubleFref,
    kDiaggDoubleOldlim,
    kDiaggDoubleSafety,
    kDiaggDoubleSumb,
    kDiaggDoubleRotateTiny,
    kDiaggDoubleBiglim,
    kDiaggDoubleCount,
    kAddhbIntOk,
    kAddhbIntNij,
    kAddhbIntNrej,
    kAddhbIntDue,
    kAddhbIntApplied,
    kAddhbIntNextNhb,
    kAddhbIntNextIdiagg,
    kAddhbIntRetry,
    kAddhbIntNextNrej0,
    kAddhbIntNextNrej1,
    kAddhbIntCount,
    kAddhbDoubleSumb,
    kAddhbDoubleCutoff,
    kAddhbDoubleRotateTiny,
    kAddhbDoubleBiglim,
    kAddhbDoubleNextTiny,
    kAddhbDoubleCount,
    kIsitscIntIemin,
    kIsitscIntIemax,
    kIsitscIntScf1,
    kIsitscIntOkscf,
    kIsitscIntIscf,
    kIsitscIntValid,
    kIsitscIntCount,
    kIsitscEnergyScf,
    kIsitscEnergyDelta,
    kIsitscDoubleCount,
    kCheckIntOccBad,
    kCheckIntVirBad,
    kCheckIntOk,
    kCheckIntCount,
    kCheckDoubleOccError,
    kCheckDoubleVirError,
    kCheckDoubleCount,
    kHeleczIntOk,
    kHeleczIntCount,
    kFinalReorthStatusEnergyTotal,
    kFinalReorthStatusEnergyScf,
    kFinalReorthStatusEnergyDelta,
    kFinalReorthStatusDoubleCount,
    kResidentStageCompleted,
    kResidentStageRequired,
    kResidentStageMissing,
    kResidentStageCode,
    kResidentStageIntCount,
    kCnvgzPmax,
    kCnvgzSumsq,
    kCnvgzFaca,
    kCnvgzFacb,
    kCnvgzFactor,
    kCnvgzDensityRms,
    kCnvgzControlCount,
    kCnvgzIntOk,
    kCnvgzIntActiveCalls,
    kCnvgzIntNoopCalls,
    kCnvgzIntCount,
    kFockIntOk,
    kFockIntCount,
    kResidentDecisionContinue,
    kResidentDecisionComplete,
    kResidentDecisionCpuBoundary,
    kResidentDecisionIterationExhausted,
    kResidentDecisionPlsRestart,
    kResidentDecisionStageFailed,
    kResidentControlDecision,
    kResidentControlDiaggMode,
    kResidentControlNhb,
    kResidentControlDiaggNf,
    kResidentControlNrej0,
    kResidentControlNrej1,
    kResidentControlIemin,
    kResidentControlIemax,
    kResidentControlScf1,
    kResidentControlCurrentIter,
    kResidentControlAddhbDue,
    kResidentControlUseThreePoint,
    kResidentControlLstart,
    kResidentControlPlsRestartRequired,
    kResidentControlPlsCalls,
    kResidentControlPlsHistoryCount,
    kResidentControlPlsRestartResetCalls,
    kResidentControlPlsRestartDone,
    kResidentControlIntCount,
    kResidentControlDiaggFref,
    kResidentControlDiaggOldlim,
    kResidentControlDiaggSafety,
    kResidentControlOvmax,
    kResidentControlPreviousEscf,
    kResidentControlShift,
    kResidentControlPlsOvmaxDelta,
    kResidentControlPlsEnergyDelta,
    kResidentControlDoubleCount,
    kPlsLoopLimit,
    kPlsIntLoop,
    kPlsIntFault,
    kPlsIntCalls,
    kPlsIntRestartDone,
    kPlsIntCount,
    kPlsDoubleOvmaxOld,
    kPlsDoubleEscfOld,
    kPlsDoubleOvmaxHistory,
    kPlsDoubleEscfHistory,
    kPlsDoubleLastOvmaxDelta,
    kPlsDoubleLastEnergyDelta,
    kPlsDoubleCount,
};
#endif

#ifdef __CUDACC__
__device__ inline double atomicAdd_double(double *address, double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    const double sum = value + __longlong_as_double(assumed);
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(sum));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

__device__ inline int mozyme_diagg2_retry_device(int previous0,
                                                int previous1) {
  return previous0 == previous1 && previous0 != 0 && previous0 < 20 ? 1 : 0;
}

__device__ inline void mozyme_diagg2_thresholds_device(
    int idiagg, double input_tiny, double bigeps, double *rotate_tiny,
    double *biglim) {
  if (idiagg % 5 == 0 || idiagg <= 5) {
    *rotate_tiny = -1.0;
    *biglim = -1.0;
  } else {
    *rotate_tiny = 0.01 * input_tiny;
    *biglim = bigeps;
  }
}

__device__ inline int resident_control_int_or(const int *control_ints,
                                              int slot, int fallback) {
  return control_ints ? control_ints[slot] : fallback;
}

__device__ inline double resident_control_double_or(
    const double *control_scalars, int slot, double fallback) {
  return control_scalars ? control_scalars[slot] : fallback;
}

__device__ inline int resident_control_current_iter_or(
    const int *control_ints, int fallback) {
  return resident_control_int_or(control_ints, kResidentControlCurrentIter,
                                 fallback);
}

__device__ inline int resident_control_completed_iter_or(
    const int *control_ints, int fallback) {
  if (!control_ints) return fallback;
  const int current = resident_control_current_iter_or(control_ints, fallback);
  return current < 2147483647 ? current + 1 : current;
}

__device__ inline bool resident_control_uses_three_point_or(
    const int *control_ints, bool fallback) {
  return resident_control_int_or(control_ints, kResidentControlUseThreePoint,
                                 fallback ? 1 : 0) != 0;
}

__device__ inline bool resident_control_terminal(
    const int *resident_control_ints) {
  return resident_control_ints &&
         resident_control_ints[kResidentControlDecision] !=
             kResidentDecisionContinue;
}

__global__ void mozyme_zero_ints_if_resident_active_kernel(
    int *values, int count, const int *resident_control_ints) {
  if (!values || count <= 0 || resident_control_terminal(resident_control_ints)) {
    return;
  }
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) values[idx] = 0;
}

__global__ void mozyme_resident_control_decision_kernel(int *control_ints,
                                                        int decision) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!control_ints) return;
  control_ints[kResidentControlDecision] = decision;
}

__global__ void mozyme_zero_doubles_if_resident_active_kernel(
    double *values, int count, const int *resident_control_ints) {
  if (!values || count <= 0 || resident_control_terminal(resident_control_ints)) {
    return;
  }
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) values[idx] = 0.0;
}

__global__ void mozyme_set_int_slot_if_resident_active_kernel(
    int *values, int slot, int value, const int *resident_control_ints) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!values || slot < 0 || resident_control_terminal(resident_control_ints)) {
    return;
  }
  values[slot] = value;
}
#endif

struct MozymeScfConfig {
  int version;
  int natoms;
  int norbs;
  int mpack;
  int noccupied;
  int nvirtual;
  int total_occupied;
  int total_virtual;
  int max_iter;
  int current_iter;
  int use_three_point;
  int density_mode;
  int fock_mode;
  int resident_fock_plan_id;
  int resident_fock_plan_full_coverage;
  int resident_fock_plan_partial_coverage;
  int resident_fock_plan_required_mask;
  int resident_fock_plan_covered_mask;
  int flags;
  int diagg_mode;
  int density_indi;
  int lstart;
  double shift;
  double thresh;
  double selcon;
  double diagg_rot_const;
  double diagg_bigeps;
  double diagg_fref;
  double diagg_oldlim;
  double diagg_safety;
  int diagg_retry;
  int diagg_nf;
  int nhb;
  int addhb_due;
  int diagg2_nrejct[2];
  int isitsc_iemin;
  int isitsc_iemax;
  int isitsc_scf1;
  double energy_scale;
  double energy_offset;
  double previous_escf;
  double emin;
  double ovmax;
  double isitsc_escf0[10];
};

struct MozymeScfState {
  int version;
  int flags;
  int use_nijbo;
  int icocc_dim;
  int cocc_dim;
  int icvir_dim;
  int cvir_dim;
  int fmo_dim;
  int partp_dim;
  int partf_dim;
  int nocc_slots;
  int nvir_slots;
  int p_dim;
  int f_dim;
  int h_dim;
  int pold_dim;
  int p1_dim;
  int p2_dim;
  int p3_dim;
  int idiag_dim;
  int iorbs_dim;
  int kopt_dim;
  int ncf_dim;
  int nncf_dim;
  int ncocc_dim;
  int nce_dim;
  int nnce_dim;
  int ncvir_dim;
  int ifmo_rows;
  int ifmo_cols;
  int eigs_dim;
  int nfmo_dim;
  int nfirst_dim;
  int nlast_dim;
  int nijbo_rows;
  int nijbo_cols;
  int coord_rows;
  int coord_cols;
  int nat_dim;
  int cosmo_enabled;
  int cosmo_nps;
  int cosmo_lm61;
  int cosmo_cosurf_rows;
  int cosmo_cosurf_cols;
  int cosmo_phinet_rows;
  int cosmo_phinet_cols;
  int cosmo_qscnet_rows;
  int cosmo_qscnet_cols;
  int cosmo_qdenet_rows;
  int cosmo_qdenet_cols;
  int cosmo_qscat_dim;
  int cosmo_srad_dim;
  int cosmo_npoints_dim;
  int cosmo_a_diag_dim;
  int cosmo_a_part_dim;
  int cosmo_m_vec_dim;
  int cosmo_iblock_pos_dim;
  int cosmo_new_surface;
  int param_dim;
  double cosmo_fepsi;
  double cosmo_disex2;
  double cosmo_solv_energy;
  double cosmo_ediel;
  double cosmo_a0;
  double cosmo_ev;
  void *p;
  void *f;
  void *h;
  void *partp;
  void *partf;
  void *pold;
  void *p1;
  void *p2;
  void *p3;
  void *idiag;
  void *iorbs;
  void *kopt;
  void *ncf;
  void *nncf;
  void *ncocc;
  void *icocc;
  void *cocc;
  void *nce;
  void *nnce;
  void *ncvir;
  void *icvir;
  void *cvir;
  void *fmo;
  void *ifmo;
  void *eigs;
  void *nfmo;
  void *nfirst;
  void *nlast;
  void *nijbo;
  void *coord;
  void *nat;
  void *param_dd;
  void *param_qq;
  void *param_tore;
  void *cosmo_iatsp;
  void *cosmo_ipiden;
  void *cosmo_gden;
  void *cosmo_qscat;
  void *cosmo_srad;
  void *cosmo_cosurf;
  void *cosmo_phinet;
  void *cosmo_qscnet;
  void *cosmo_qdenet;
  void *cosmo_npoints;
  void *cosmo_a_diag;
  void *cosmo_a_part;
  void *cosmo_a_part_i;
  void *cosmo_a_part_j;
  void *cosmo_m_vec;
  void *cosmo_iblock_pos;
  void *cosmo_solv_energy_ptr;
  void *cosmo_ediel_ptr;
};

struct MozymeScfStatus {
  int version;
  int code;
  int ready;
  int resident;
  int device_id;
  int natoms;
  int norbs;
  int mpack;
  int iterations;
  int stage_completed;
  int stage_required;
  int stage_missing;
  int resident_decision;
  int resident_fock_plan_id;
  int resident_fock_plan_full_coverage;
  int resident_fock_plan_partial_coverage;
  int resident_fock_plan_required_mask;
  int resident_fock_plan_covered_mask;
  int final_publication_done;
  int final_publication_arrays;
  std::size_t final_publication_bytes;
  int final_publication_cosmo;
  double energy_total;
  double energy_delta;
  double density_max;
  double density_rms;
  double wall_ms;
  int diagg_nij;
  int diagg_nf;
  int idiagg;
  int nhb;
  int addhb_due;
  int addhb_applied;
  int addhb_nij;
  int diagg2_nrejct[2];
  double diagg_tiny;
  double next_tiny;
  double diagg_fref;
  double diagg_oldlim;
  double diagg_safety;
  double diagg_sumt;
  double diagg_sumb;
  double energy_scf;
  int isitsc_okscf;
  int isitsc_iscf;
  int isitsc_iemin;
  int isitsc_iemax;
  int isitsc_scf1;
  int use_three_point;
  int lstart;
  double shift;
  double isitsc_escf0[10];
  int resident_stage_calls[kResidentStageSlotCount];
  double resident_stage_ms[kResidentStageSlotCount];
  int final_reorth_applied;
  double final_reorth_ms;
  double final_reorth_sum;
  int pls_supervisor_calls;
  int pls_restart_required;
  int pls_history_count;
  double pls_ovmax_delta;
  double pls_energy_delta;
  int pls_restart_reset_device_calls;
  int pls_restart_done;
  int cosmo_enabled;
  int cosmo_fock_calls;
  int cosmo_matvec_calls;
  int cosmo_cg_iterations;
  int cosmo_nps;
  int cosmo_lm61;
  int cosmo_pair_count;
  double cosmo_solv_energy;
  double cosmo_ediel;
  double cosmo_last_residual;
  int cosmo_cg_control_resident;
  int cosmo_cg_converged;
  int cosmo_cg_breakdown;
  int cosmo_cg_host_syncs;
  double cosmo_cg_target_tol;
  int cnvgz_active_calls;
  int cnvgz_noop_calls;
  int strict_resident_host_syncs;
  int strict_resident_control_polls;
};

struct ResidentFinalPublicationProof {
  int arrays = 0;
  std::size_t bytes = 0;
  int cosmo = 0;
};

#ifdef __CUDACC__
inline bool cuda_context_ok(cudaError_t status, const char *where) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "[GPU ERROR] MOZYME SCF %s: %s\n", where,
               cudaGetErrorString(status));
  return false;
}

template <typename T>
struct DeviceBuffer {
  T *ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() { reset(); }

  void reset() {
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
    count = 0;
  }

  bool resize(std::size_t next_count) {
    if (next_count == count && ptr) return true;
    reset();
    if (next_count == 0) return true;
    if (!cuda_context_ok(
            cudaMalloc(reinterpret_cast<void **>(&ptr),
                       next_count * sizeof(T)),
            "cudaMalloc")) {
      return false;
    }
    count = next_count;
    return true;
  }

  bool upload(const T *host, std::size_t next_count) {
    if (!host || next_count == 0) return false;
    if (!resize(next_count)) return false;
    return cuda_context_ok(
        cudaMemcpy(ptr, host, next_count * sizeof(T),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy host-to-device");
  }

};

template <typename T>
bool device_buffer_ready(const DeviceBuffer<T> &device,
                         std::size_t required_count) {
  return device.ptr && device.count >= required_count;
}

template <typename T>
bool copy_device_to_host_raw(void *host, const DeviceBuffer<T> &device,
                             std::size_t count, const char *label) {
  if (count == 0) return true;
  if (!host || !device.ptr || device.count < count) return false;
  return cuda_context_ok(
      cudaMemcpy(host, device.ptr, count * sizeof(T), cudaMemcpyDeviceToHost),
      label);
}

template <typename T>
bool stage_device_to_vector(std::vector<T> &host,
                            const DeviceBuffer<T> &device,
                            std::size_t count, const char *label) {
  host.clear();
  if (count == 0) return true;
  if (!device.ptr || device.count < count) return false;
  host.resize(count);
  return cuda_context_ok(
      cudaMemcpy(host.data(), device.ptr, count * sizeof(T),
                 cudaMemcpyDeviceToHost),
      label);
}

template <typename T>
void commit_staged_vector(void *host, const std::vector<T> &values) {
  if (!host || values.empty()) return;
  std::memcpy(host, values.data(), values.size() * sizeof(T));
}

template <typename T>
bool stage_and_commit_device_vector(void *host, const DeviceBuffer<T> &device,
                                    std::size_t count, const char *label) {
  if (count == 0) return true;
  if (!host || !device.ptr || device.count < count) return false;
  constexpr std::size_t kMaxStagedElements = 1u << 20;
  const std::size_t staged_elements = std::min(count, kMaxStagedElements);
  std::vector<T> values;
  try {
    values.resize(staged_elements);
  } catch (const std::bad_alloc &) {
    return false;
  }
  auto *typed_host = static_cast<T *>(host);
  for (std::size_t offset = 0; offset < count; offset += staged_elements) {
    const std::size_t chunk = std::min(staged_elements, count - offset);
    if (!cuda_context_ok(
            cudaMemcpy(values.data(), device.ptr + offset,
                       chunk * sizeof(T), cudaMemcpyDeviceToHost),
            label)) {
      return false;
    }
    std::memcpy(typed_host + offset, values.data(), chunk * sizeof(T));
  }
  return true;
}

struct MozymeScfDeviceState {
  DeviceBuffer<double> p;
  DeviceBuffer<double> f;
  DeviceBuffer<double> h;
  DeviceBuffer<double> partp;
  DeviceBuffer<double> partf;
  DeviceBuffer<double> pold;
  DeviceBuffer<double> p1;
  DeviceBuffer<double> p2;
  DeviceBuffer<double> p3;
  DeviceBuffer<double> cocc;
  DeviceBuffer<double> cvir;
  DeviceBuffer<double> fmo;
  DeviceBuffer<double> eigs;
  DeviceBuffer<int> idiag;
  DeviceBuffer<int> iorbs;
  DeviceBuffer<int> ncf;
  DeviceBuffer<int> nncf;
  DeviceBuffer<int> ncocc;
  DeviceBuffer<int> icocc;
  DeviceBuffer<int> nce;
  DeviceBuffer<int> nnce;
  DeviceBuffer<int> ncvir;
  DeviceBuffer<int> icvir;
  DeviceBuffer<int> ifmo;
  DeviceBuffer<int> nfmo;
  DeviceBuffer<int> nfirst;
  DeviceBuffer<int> nlast;
  DeviceBuffer<int> nijbo;
  DeviceBuffer<double> coord;
  DeviceBuffer<int> nat;
  DeviceBuffer<double> param_dd;
  DeviceBuffer<double> param_qq;
  DeviceBuffer<double> param_tore;
  DeviceBuffer<int> cosmo_iatsp;
  DeviceBuffer<int> cosmo_ipiden;
  DeviceBuffer<double> cosmo_gden;
  DeviceBuffer<double> cosmo_qscat;
  DeviceBuffer<double> cosmo_srad;
  DeviceBuffer<double> cosmo_cosurf;
  DeviceBuffer<double> cosmo_phinet;
  DeviceBuffer<double> cosmo_qscnet;
  DeviceBuffer<double> cosmo_qdenet;
  DeviceBuffer<int> cosmo_npoints;
  DeviceBuffer<double> cosmo_a_diag;
  DeviceBuffer<double> cosmo_a_part;
  DeviceBuffer<int> cosmo_a_part_i;
  DeviceBuffer<int> cosmo_a_part_j;
  DeviceBuffer<double> cosmo_m_vec;
  DeviceBuffer<int> cosmo_iblock_pos;
  DeviceBuffer<double> cosmo_scalars;
  DeviceBuffer<double> cosmo_cg_x;
  DeviceBuffer<double> cosmo_cg_r;
  DeviceBuffer<double> cosmo_cg_p;
  DeviceBuffer<double> cosmo_cg_q;
  DeviceBuffer<double> cosmo_cg_z;
  DeviceBuffer<double> cosmo_cg_tmp;
  DeviceBuffer<double> cosmo_cg_scalars;
  DeviceBuffer<int> cosmo_cg_ints;
  DeviceBuffer<double> cosmo_status_scalars;
  DeviceBuffer<int> cosmo_status_ints;

  DeviceBuffer<double> checkpoint_p;
  DeviceBuffer<double> checkpoint_f;
  DeviceBuffer<double> checkpoint_partp;
  DeviceBuffer<double> checkpoint_partf;
  DeviceBuffer<double> checkpoint_pold;
  DeviceBuffer<double> checkpoint_p1;
  DeviceBuffer<double> checkpoint_p2;
  DeviceBuffer<double> checkpoint_p3;
  DeviceBuffer<double> checkpoint_cocc;
  DeviceBuffer<double> checkpoint_cvir;
  DeviceBuffer<double> checkpoint_fmo;
  DeviceBuffer<double> checkpoint_eigs;
  DeviceBuffer<int> checkpoint_idiag;
  DeviceBuffer<int> checkpoint_iorbs;
  DeviceBuffer<int> checkpoint_ncf;
  DeviceBuffer<int> checkpoint_nncf;
  DeviceBuffer<int> checkpoint_ncocc;
  DeviceBuffer<int> checkpoint_icocc;
  DeviceBuffer<int> checkpoint_nce;
  DeviceBuffer<int> checkpoint_nnce;
  DeviceBuffer<int> checkpoint_ncvir;
  DeviceBuffer<int> checkpoint_icvir;
  DeviceBuffer<int> checkpoint_ifmo;
  DeviceBuffer<int> checkpoint_nfmo;
  DeviceBuffer<int> checkpoint_nfirst;
  DeviceBuffer<int> checkpoint_nlast;
  DeviceBuffer<double> checkpoint_coord;
  DeviceBuffer<int> checkpoint_nat;
  DeviceBuffer<double> checkpoint_cosmo_qscat;
  DeviceBuffer<double> checkpoint_cosmo_phinet;
  DeviceBuffer<double> checkpoint_cosmo_qscnet;
  DeviceBuffer<double> checkpoint_cosmo_qdenet;
  DeviceBuffer<double> checkpoint_cosmo_scalars;
  DeviceBuffer<double> checkpoint_cosmo_status_scalars;
  DeviceBuffer<int> checkpoint_cosmo_status_ints;
  DeviceBuffer<double> checkpoint_energy_sums;
  DeviceBuffer<double> checkpoint_cnvgz_sums;
  DeviceBuffer<double> checkpoint_diagg_scalars;
  DeviceBuffer<double> checkpoint_addhb_scalars;
  DeviceBuffer<double> checkpoint_isitsc_scalars;
  DeviceBuffer<double> checkpoint_isitsc_escf0;
  DeviceBuffer<double> checkpoint_resident_control_scalars;
  DeviceBuffer<double> checkpoint_pls_scalars;
  DeviceBuffer<int> checkpoint_cnvgz_ints;
  DeviceBuffer<int> checkpoint_diagg_ints;
  DeviceBuffer<int> checkpoint_addhb_ints;
  DeviceBuffer<int> checkpoint_isitsc_ints;
  DeviceBuffer<int> checkpoint_helecz_ints;
  DeviceBuffer<int> checkpoint_resident_stage_ints;
  DeviceBuffer<int> checkpoint_resident_stage_calls;
  DeviceBuffer<int> checkpoint_resident_control_ints;
  DeviceBuffer<int> checkpoint_pls_ints;

  DeviceBuffer<double> eimp_p;
  DeviceBuffer<int> eimp_pair_updates;
  DeviceBuffer<int> density_updates;
  DeviceBuffer<double> qe;
  DeviceBuffer<double> atom_sums;
  DeviceBuffer<double> atom_diag_sums;
  DeviceBuffer<double> energy_sums;
  DeviceBuffer<double> diag_new;
  DeviceBuffer<double> diag_old;
  DeviceBuffer<double> candidate;
  DeviceBuffer<double> block_max;
  DeviceBuffer<double> block_sumsq;
  DeviceBuffer<double> block_faca;
  DeviceBuffer<double> block_facb;
  DeviceBuffer<double> cnvgz_sums;
  DeviceBuffer<int> cnvgz_ints;
  DeviceBuffer<int> diagg_ints;
  DeviceBuffer<double> diagg_scalars;
  DeviceBuffer<int> addhb_ints;
  DeviceBuffer<double> addhb_scalars;
  DeviceBuffer<double> diagg_aocc;
  DeviceBuffer<int> diagg_work_ints;
  DeviceBuffer<double> diagg_work_scalars;
  DeviceBuffer<double> diagg_avir_entry;
  DeviceBuffer<int> diagg_counts;
  DeviceBuffer<int> diagg_offsets;
  DeviceBuffer<int> diagg_pair_state;
  DeviceBuffer<int> diagg_vclaim;
  DeviceBuffer<int> diagg_oclaim;
  DeviceBuffer<int> hb_pair_counts;
  DeviceBuffer<int> hb_pair_offsets;
  DeviceBuffer<int> hb_pair_i;
  DeviceBuffer<int> hb_pair_j;
  DeviceBuffer<int> hb_entry_counts;
  DeviceBuffer<int> hb_entry_offsets;
  DeviceBuffer<int> tidy_iused;
  DeviceBuffer<int> tidy_ncnew;
  DeviceBuffer<int> tidy_ncmnew;
  DeviceBuffer<int> tidy_nncnew;
  DeviceBuffer<int> tidy_result;
  DeviceBuffer<int> isitsc_ints;
  DeviceBuffer<double> isitsc_scalars;
  DeviceBuffer<double> isitsc_escf0;
  DeviceBuffer<int> resident_stage_ints;
  DeviceBuffer<int> resident_stage_calls;
  DeviceBuffer<int> resident_control_ints;
  DeviceBuffer<double> resident_control_scalars;
  DeviceBuffer<int> pls_ints;
  DeviceBuffer<double> pls_scalars;
  DeviceBuffer<int> check_ints;
  DeviceBuffer<double> check_errors;
  DeviceBuffer<int> helecz_ints;
  DeviceBuffer<int> fock_ints;
  DeviceBuffer<double> final_reorth_ws;
  DeviceBuffer<double> final_reorth_sumtot;
  DeviceBuffer<double> final_reorth_status_scalars;
  DeviceBuffer<int> final_reorth_latom;
  DeviceBuffer<int> final_reorth_iused;
  DeviceBuffer<int> final_reorth_status;

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool uploaded = false;

  // Non-blocking per-stage profile: event pairs are only read back after the run.
  struct StageTiming {
    const char *label;
    cudaEvent_t start;
    cudaEvent_t stop;
  };
  std::vector<StageTiming> stage_timings;
  cudaEvent_t pending_stage_start = nullptr;

  MozymeScfDeviceState() = default;
  MozymeScfDeviceState(const MozymeScfDeviceState &) = delete;
  MozymeScfDeviceState &operator=(const MozymeScfDeviceState &) = delete;

  ~MozymeScfDeviceState() { cleanup_events(); }

  void cleanup_stage_timings() {
    for (auto &entry : stage_timings) {
      if (entry.start) cudaEventDestroy(entry.start);
      if (entry.stop) cudaEventDestroy(entry.stop);
    }
    stage_timings.clear();
    if (pending_stage_start) cudaEventDestroy(pending_stage_start);
    pending_stage_start = nullptr;
  }

  void cleanup_events() {
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    start = nullptr;
    stop = nullptr;
    cleanup_stage_timings();
  }

  bool ensure_events() {
    if (!start &&
        !cuda_context_ok(cudaEventCreate(&start),
                         "resident create start event")) {
      return false;
    }
    if (!stop &&
        !cuda_context_ok(cudaEventCreate(&stop),
                         "resident create stop event")) {
      return false;
    }
    return true;
  }
};

template <typename T>
bool resize_checkpoint_like(DeviceBuffer<T> &checkpoint,
                            const DeviceBuffer<T> &source) {
  return checkpoint.resize(source.count);
}

bool resize_resident_checkpoint_buffers(MozymeScfDeviceState &dev) {
  return resize_checkpoint_like(dev.checkpoint_p, dev.p) &&
         resize_checkpoint_like(dev.checkpoint_f, dev.f) &&
         resize_checkpoint_like(dev.checkpoint_partp, dev.partp) &&
         resize_checkpoint_like(dev.checkpoint_partf, dev.partf) &&
         resize_checkpoint_like(dev.checkpoint_pold, dev.pold) &&
         resize_checkpoint_like(dev.checkpoint_p1, dev.p1) &&
         resize_checkpoint_like(dev.checkpoint_p2, dev.p2) &&
         resize_checkpoint_like(dev.checkpoint_p3, dev.p3) &&
         resize_checkpoint_like(dev.checkpoint_idiag, dev.idiag) &&
         resize_checkpoint_like(dev.checkpoint_iorbs, dev.iorbs) &&
         resize_checkpoint_like(dev.checkpoint_ncf, dev.ncf) &&
         resize_checkpoint_like(dev.checkpoint_nncf, dev.nncf) &&
         resize_checkpoint_like(dev.checkpoint_ncocc, dev.ncocc) &&
         resize_checkpoint_like(dev.checkpoint_nce, dev.nce) &&
         resize_checkpoint_like(dev.checkpoint_nnce, dev.nnce) &&
         resize_checkpoint_like(dev.checkpoint_ncvir, dev.ncvir) &&
         resize_checkpoint_like(dev.checkpoint_icocc, dev.icocc) &&
         resize_checkpoint_like(dev.checkpoint_icvir, dev.icvir) &&
         resize_checkpoint_like(dev.checkpoint_cocc, dev.cocc) &&
         resize_checkpoint_like(dev.checkpoint_cvir, dev.cvir) &&
         resize_checkpoint_like(dev.checkpoint_eigs, dev.eigs) &&
         resize_checkpoint_like(dev.checkpoint_nfmo, dev.nfmo) &&
         resize_checkpoint_like(dev.checkpoint_nfirst, dev.nfirst) &&
         resize_checkpoint_like(dev.checkpoint_nlast, dev.nlast) &&
         resize_checkpoint_like(dev.checkpoint_coord, dev.coord) &&
         resize_checkpoint_like(dev.checkpoint_nat, dev.nat) &&
         resize_checkpoint_like(dev.checkpoint_ifmo, dev.ifmo) &&
         resize_checkpoint_like(dev.checkpoint_fmo, dev.fmo) &&
         resize_checkpoint_like(dev.checkpoint_cosmo_qscat,
                                dev.cosmo_qscat) &&
         resize_checkpoint_like(dev.checkpoint_cosmo_phinet,
                                dev.cosmo_phinet) &&
         resize_checkpoint_like(dev.checkpoint_cosmo_qscnet,
                                dev.cosmo_qscnet) &&
         resize_checkpoint_like(dev.checkpoint_cosmo_qdenet,
                                dev.cosmo_qdenet) &&
         resize_checkpoint_like(dev.checkpoint_cosmo_scalars,
                                dev.cosmo_scalars) &&
         resize_checkpoint_like(dev.checkpoint_cosmo_status_scalars,
                                dev.cosmo_status_scalars) &&
         resize_checkpoint_like(dev.checkpoint_cosmo_status_ints,
                                dev.cosmo_status_ints) &&
         resize_checkpoint_like(dev.checkpoint_energy_sums,
                                dev.energy_sums) &&
         resize_checkpoint_like(dev.checkpoint_cnvgz_sums,
                                dev.cnvgz_sums) &&
         resize_checkpoint_like(dev.checkpoint_diagg_scalars,
                                dev.diagg_scalars) &&
         resize_checkpoint_like(dev.checkpoint_addhb_scalars,
                                dev.addhb_scalars) &&
         resize_checkpoint_like(dev.checkpoint_isitsc_scalars,
                                dev.isitsc_scalars) &&
         resize_checkpoint_like(dev.checkpoint_isitsc_escf0,
                                dev.isitsc_escf0) &&
         resize_checkpoint_like(dev.checkpoint_resident_control_scalars,
                                dev.resident_control_scalars) &&
         resize_checkpoint_like(dev.checkpoint_pls_scalars,
                                dev.pls_scalars) &&
         resize_checkpoint_like(dev.checkpoint_cnvgz_ints,
                                dev.cnvgz_ints) &&
         resize_checkpoint_like(dev.checkpoint_diagg_ints,
                                dev.diagg_ints) &&
         resize_checkpoint_like(dev.checkpoint_addhb_ints,
                                dev.addhb_ints) &&
         resize_checkpoint_like(dev.checkpoint_isitsc_ints,
                                dev.isitsc_ints) &&
         resize_checkpoint_like(dev.checkpoint_helecz_ints,
                                dev.helecz_ints) &&
         resize_checkpoint_like(dev.checkpoint_resident_stage_ints,
                                dev.resident_stage_ints) &&
         resize_checkpoint_like(dev.checkpoint_resident_stage_calls,
                                dev.resident_stage_calls) &&
         resize_checkpoint_like(dev.checkpoint_resident_control_ints,
                                dev.resident_control_ints) &&
         resize_checkpoint_like(dev.checkpoint_pls_ints, dev.pls_ints);
}
#endif

struct MozymeScfContext {
  MozymeScfConfig config{};
  MozymeScfState state{};
  bool state_registered = false;
  int cosmo_fock_calls = 0;
  int cosmo_matvec_calls = 0;
  int cosmo_cg_iterations = 0;
  int pls_restart_reset_device_calls = 0;
  int pls_restart_done = 0;
  int strict_resident_host_syncs = 0;
  int strict_resident_control_polls = 0;
  double cosmo_solv_energy = 0.0;
  double cosmo_ediel = 0.0;
  double cosmo_last_residual = 0.0;
#ifdef __CUDACC__
  MozymeScfDeviceState device{};
#endif
};

#ifdef __CUDACC__
struct MozymeScfHostCheckpoint {
  int cosmo_fock_calls = 0;
  int cosmo_matvec_calls = 0;
  int cosmo_cg_iterations = 0;
  int pls_restart_reset_device_calls = 0;
  int pls_restart_done = 0;
  double cosmo_solv_energy = 0.0;
  double cosmo_ediel = 0.0;
  double cosmo_last_residual = 0.0;
  int cosmo_new_surface = 0;
};

bool zero_ints_if_resident_active(MozymeScfContext &ctx, int *values,
                                  int count, const char *label);
bool zero_doubles_if_resident_active(MozymeScfContext &ctx, double *values,
                                     int count, const char *label);
bool resident_stage_timing_enabled(const MozymeScfContext &ctx,
                                   const double *wall_ms);
bool begin_resident_stage_timing(MozymeScfContext &ctx, bool time_stage,
                                 const char *label);
bool finish_resident_stage_timing(MozymeScfContext &ctx, bool time_stage,
                                  const char *stop_label,
                                  const char *kernels_label,
                                  const char *sync_label,
                                  const char *elapsed_label,
                                  double *wall_ms);
bool env_enabled(const char *name);
bool strict_resident_request_enabled();
bool resident_scf_request_enabled();
bool host_commit_marker_enabled();

MozymeScfHostCheckpoint capture_host_checkpoint(
    const MozymeScfContext &context) {
  MozymeScfHostCheckpoint checkpoint{};
  checkpoint.cosmo_fock_calls = context.cosmo_fock_calls;
  checkpoint.cosmo_matvec_calls = context.cosmo_matvec_calls;
  checkpoint.cosmo_cg_iterations = context.cosmo_cg_iterations;
  checkpoint.pls_restart_reset_device_calls =
      context.pls_restart_reset_device_calls;
  checkpoint.pls_restart_done = context.pls_restart_done;
  checkpoint.cosmo_solv_energy = context.cosmo_solv_energy;
  checkpoint.cosmo_ediel = context.cosmo_ediel;
  checkpoint.cosmo_last_residual = context.cosmo_last_residual;
  checkpoint.cosmo_new_surface = context.state.cosmo_new_surface;
  return checkpoint;
}

void restore_host_checkpoint(MozymeScfContext &context,
                             const MozymeScfHostCheckpoint &checkpoint) {
  context.cosmo_fock_calls = checkpoint.cosmo_fock_calls;
  context.cosmo_matvec_calls = checkpoint.cosmo_matvec_calls;
  context.cosmo_cg_iterations = checkpoint.cosmo_cg_iterations;
  context.pls_restart_reset_device_calls =
      checkpoint.pls_restart_reset_device_calls;
  context.pls_restart_done = checkpoint.pls_restart_done;
  context.cosmo_solv_energy = checkpoint.cosmo_solv_energy;
  context.cosmo_ediel = checkpoint.cosmo_ediel;
  context.cosmo_last_residual = checkpoint.cosmo_last_residual;
  context.state.cosmo_new_surface = checkpoint.cosmo_new_surface;
}
#endif

void publish_cosmo_status(MozymeScfStatus *status,
                          const MozymeScfContext *context) {
  if (!status) return;
  status->cosmo_enabled =
      context && context->state.cosmo_enabled != 0 ? 1 : 0;
  status->cosmo_fock_calls = context ? context->cosmo_fock_calls : 0;
  status->cosmo_matvec_calls = context ? context->cosmo_matvec_calls : 0;
  status->cosmo_cg_iterations = context ? context->cosmo_cg_iterations : 0;
  status->cosmo_nps = context ? context->state.cosmo_nps : 0;
  status->cosmo_lm61 = context ? context->state.cosmo_lm61 : 0;
  status->cosmo_pair_count = context ? context->state.cosmo_a_part_dim : 0;
  status->cosmo_solv_energy = context ? context->cosmo_solv_energy : 0.0;
  status->cosmo_ediel = context ? context->cosmo_ediel : 0.0;
  status->cosmo_last_residual = context ? context->cosmo_last_residual : 0.0;
  status->cosmo_cg_control_resident = 0;
  status->cosmo_cg_converged = context && context->state.cosmo_enabled == 0 ? 1 : 0;
  status->cosmo_cg_breakdown = 0;
  status->cosmo_cg_host_syncs = 0;
  status->cosmo_cg_target_tol = 0.0;
  status->cnvgz_active_calls = 0;
  status->cnvgz_noop_calls = 0;
  status->strict_resident_host_syncs =
      context ? context->strict_resident_host_syncs : 0;
  status->strict_resident_control_polls =
      context ? context->strict_resident_control_polls : 0;
}

void publish_pls_runtime_status(MozymeScfStatus *status,
                                const MozymeScfContext *context) {
  if (!status) return;
  if (status->pls_restart_reset_device_calls != 0 ||
      status->pls_restart_done != 0) {
    return;
  }
  status->pls_restart_reset_device_calls =
      context ? context->pls_restart_reset_device_calls : 0;
  status->pls_restart_done = context ? context->pls_restart_done : 0;
}

void fill_status(MozymeScfStatus *status, const MozymeScfContext *context,
                 int code) {
  if (!status) return;
  status->version = kMozymeScfAbiVersion;
  status->code = code;
  status->ready = context && context->state_registered ? 1 : 0;
  status->resident = 0;
  status->device_id = -1;
#ifdef __CUDACC__
  status->resident = context && context->device.uploaded ? 1 : 0;
  if (status->resident) {
    int device_id = -1;
    if (cuda_context_ok(cudaGetDevice(&device_id),
                        "resident status cudaGetDevice")) {
      status->device_id = device_id;
    }
  }
#endif
  status->natoms = context ? context->config.natoms : 0;
  status->norbs = context ? context->config.norbs : 0;
  status->mpack = context ? context->config.mpack : 0;
  status->iterations = 0;
  status->stage_completed = 0;
  status->stage_required = kMozymeScfStageFull;
  status->stage_missing = kMozymeScfStageFull;
  status->resident_decision = kResidentDecisionContinue;
  status->resident_fock_plan_id = context ? context->config.resident_fock_plan_id : 0;
  status->resident_fock_plan_full_coverage =
      context ? context->config.resident_fock_plan_full_coverage : 0;
  status->resident_fock_plan_partial_coverage =
      context ? context->config.resident_fock_plan_partial_coverage : 0;
  status->resident_fock_plan_required_mask =
      context ? context->config.resident_fock_plan_required_mask : 0;
  status->resident_fock_plan_covered_mask =
      context ? context->config.resident_fock_plan_covered_mask : 0;
  status->final_publication_done = 0;
  status->final_publication_arrays = 0;
  status->final_publication_bytes = 0;
  status->final_publication_cosmo = 0;
  status->energy_total = 0.0;
  status->energy_delta = 0.0;
  status->density_max = 0.0;
  status->density_rms = 0.0;
  status->wall_ms = 0.0;
  status->diagg_nij = 0;
  status->diagg_nf = 0;
  status->idiagg = context ? context->config.diagg_mode : 0;
  status->nhb = context ? context->config.nhb : 0;
  status->addhb_due = 0;
  status->addhb_applied = 0;
  status->addhb_nij = 0;
  status->diagg2_nrejct[0] =
      context ? context->config.diagg2_nrejct[0] : 0;
  status->diagg2_nrejct[1] =
      context ? context->config.diagg2_nrejct[1] : 0;
  status->diagg_tiny = 0.0;
  status->next_tiny = 0.0;
  status->diagg_fref = 0.0;
  status->diagg_oldlim = 0.0;
  status->diagg_safety = 0.0;
  status->diagg_sumt = 0.0;
  status->diagg_sumb = 0.0;
  status->energy_scf = 0.0;
  status->isitsc_okscf = 0;
  status->isitsc_iscf = 0;
  status->isitsc_iemin = context ? context->config.isitsc_iemin : 0;
  status->isitsc_iemax = context ? context->config.isitsc_iemax : 0;
  status->isitsc_scf1 = context ? context->config.isitsc_scf1 : 0;
  status->use_three_point = context ? context->config.use_three_point : 0;
  status->lstart = context ? context->config.lstart : 0;
  status->shift = context ? context->config.shift : 0.0;
  for (int i = 0; i < 10; ++i) {
    status->isitsc_escf0[i] =
        context ? context->config.isitsc_escf0[i] : 0.0;
  }
  for (int i = 0; i < kResidentStageSlotCount; ++i) {
    status->resident_stage_calls[i] = 0;
    status->resident_stage_ms[i] = 0.0;
  }
  status->final_reorth_applied = 0;
  status->final_reorth_ms = 0.0;
  status->final_reorth_sum = 0.0;
  status->pls_supervisor_calls = 0;
  status->pls_restart_required = 0;
  status->pls_history_count = 0;
  status->pls_ovmax_delta = 0.0;
  status->pls_energy_delta = 0.0;
  publish_pls_runtime_status(status, context);
  publish_cosmo_status(status, context);
}

bool valid_config(const MozymeScfConfig &config) {
  const int expected_plan_id = (config.fock_mode == 0)
                                   ? kMozymeFockPlanFull
                                   : kMozymeFockPlanPartial;
  const int expected_required_mask =
      (config.fock_mode == 0)
          ? kMozymeFockPlanFullMask
          : (kMozymeFockPlanFullMask | kMozymeFockPlanPartialMask);
  return config.version == kMozymeScfAbiVersion && config.natoms > 0 &&
         config.norbs > 0 && config.mpack > 0 && config.noccupied >= 0 &&
         config.nvirtual >= 0 && config.total_occupied >= config.noccupied &&
         config.total_virtual >= config.nvirtual && config.max_iter >= 0 &&
         config.current_iter >= 0 && config.density_mode >= -1 &&
         config.density_mode <= 1 && config.fock_mode >= -1 &&
         config.fock_mode <= 1 &&
         config.resident_fock_plan_id == expected_plan_id &&
         config.resident_fock_plan_full_coverage == 1 &&
         config.resident_fock_plan_required_mask == expected_required_mask &&
         config.resident_fock_plan_covered_mask == expected_required_mask &&
         ((config.resident_fock_plan_required_mask &
           kMozymeFockPlanPartialMask) == 0 ||
          config.resident_fock_plan_partial_coverage == 1) &&
         config.density_indi >= -1 &&
         config.density_indi <= 1 && config.selcon >= 0.0 &&
         config.lstart >= 0 && config.thresh >= 0.0 &&
         config.diagg_bigeps >= 0.0 &&
         config.diagg_fref >= 0.0 && config.diagg_oldlim >= 0.0 &&
         config.diagg_safety >= 1.0 && config.diagg_nf >= 0 &&
         config.nhb >= 0 && config.nhb <= 4 &&
         (config.addhb_due == 0 || config.addhb_due == 1) &&
         config.diagg2_nrejct[0] >= 0 && config.diagg2_nrejct[1] >= 0 &&
         (config.diagg_retry == 0 || config.diagg_retry == 1) &&
         config.isitsc_iemin >= 0 && config.isitsc_iemin <= 5 &&
         config.isitsc_iemax >= 0 && config.isitsc_iemax <= 5 &&
         (config.isitsc_scf1 == 0 || config.isitsc_scf1 == 1) &&
         config.energy_scale != 0.0;
}

bool valid_state(const MozymeScfConfig &config, const MozymeScfState &state) {
  if (state.version != kMozymeScfAbiVersion) return false;
  if (!state.p || !state.f || !state.h || !state.partp || !state.partf ||
      !state.pold || !state.p1 || !state.p2 || !state.p3 || !state.idiag ||
      !state.iorbs || !state.kopt || !state.ncf || !state.nncf ||
      !state.ncocc || !state.icocc || !state.cocc || !state.nce ||
      !state.nnce || !state.ncvir || !state.icvir || !state.cvir ||
      !state.fmo || !state.ifmo || !state.eigs || !state.nfmo ||
      !state.nfirst || !state.nlast) {
    return false;
  }
  if (state.use_nijbo != 0 && state.use_nijbo != 1) return false;
  if (state.use_nijbo == 1 && !state.nijbo) return false;

  const int numat = config.natoms;
  const int norbs = config.norbs;
  const int mpack = config.mpack;
  const int occ_slots_required = config.total_occupied + 1;
  const int vir_slots_required = config.total_virtual + 1;
  if (numat <= 0 || norbs <= 0 || mpack <= 0 || occ_slots_required <= 0 ||
      vir_slots_required <= 0) {
    return false;
  }
  if (state.p_dim < mpack || state.f_dim < mpack || state.h_dim < mpack ||
      state.pold_dim < mpack) {
    return false;
  }
  if (state.partp_dim < 1 || state.partf_dim < 1) return false;
  if ((config.fock_mode != 0 || config.density_indi != 0) &&
      (state.partp_dim < mpack || state.partf_dim < mpack)) {
    return false;
  }
  if (state.p1_dim < norbs || state.p2_dim < norbs ||
      state.p3_dim < norbs || state.idiag_dim < norbs ||
      state.eigs_dim < norbs || state.nfmo_dim < norbs) {
    return false;
  }
  if (state.iorbs_dim < numat || state.kopt_dim < numat ||
      state.nfirst_dim < numat || state.nlast_dim < numat) {
    return false;
  }
  if (state.nocc_slots < occ_slots_required ||
      state.nvir_slots < vir_slots_required ||
      state.ncf_dim < state.nocc_slots ||
      state.nncf_dim < state.nocc_slots ||
      state.ncocc_dim < state.nocc_slots ||
      state.nce_dim < state.nvir_slots ||
      state.nnce_dim < state.nvir_slots ||
      state.ncvir_dim < state.nvir_slots) {
    return false;
  }
  if (state.icocc_dim <= 0 || state.cocc_dim <= 0 ||
      state.icvir_dim <= 0 || state.cvir_dim <= 0 ||
      state.fmo_dim <= 0 || state.ifmo_rows != 2 ||
      state.ifmo_cols < state.fmo_dim) {
    return false;
  }
  if (state.use_nijbo == 1 &&
      (state.nijbo_rows != numat || state.nijbo_cols < numat)) {
    return false;
  }
  if (state.cosmo_enabled != 0 && state.cosmo_enabled != 1) return false;
  if (state.cosmo_enabled) {
    if (state.use_nijbo != 1 || !state.nijbo) return false;
    if (!state.coord || !state.nat || !state.param_dd || !state.param_qq ||
        !state.param_tore || !state.cosmo_iatsp || !state.cosmo_ipiden ||
        !state.cosmo_gden || !state.cosmo_qscat || !state.cosmo_srad ||
        !state.cosmo_cosurf || !state.cosmo_phinet ||
        !state.cosmo_qscnet || !state.cosmo_qdenet ||
        !state.cosmo_npoints || !state.cosmo_a_diag ||
        !state.cosmo_m_vec || !state.cosmo_iblock_pos ||
        !state.cosmo_solv_energy_ptr || !state.cosmo_ediel_ptr) {
      return false;
    }
    if (state.coord_rows < 3 || state.coord_cols < numat ||
        state.nat_dim < numat || state.param_dim < 107 ||
        state.cosmo_nps <= 0 || state.cosmo_lm61 <= 0 ||
        state.cosmo_qscat_dim < numat || state.cosmo_srad_dim < numat ||
        state.cosmo_cosurf_rows < 4 ||
        state.cosmo_cosurf_cols < state.cosmo_nps ||
        state.cosmo_phinet_rows < state.cosmo_nps ||
        state.cosmo_phinet_cols < 3 ||
        state.cosmo_qscnet_rows < state.cosmo_nps ||
        state.cosmo_qscnet_cols < 3 ||
        state.cosmo_qdenet_rows < state.cosmo_lm61 ||
        state.cosmo_qdenet_cols < 3 ||
        state.cosmo_npoints_dim < numat + 1 ||
        state.cosmo_a_diag_dim < state.cosmo_nps ||
        state.cosmo_m_vec_dim <= 0 ||
        state.cosmo_iblock_pos_dim < numat ||
        state.cosmo_fepsi == 0.0 || state.cosmo_a0 == 0.0 ||
        state.cosmo_ev == 0.0 ||
        (state.cosmo_new_surface != 0 && state.cosmo_new_surface != 1)) {
      return false;
    }
    if (state.cosmo_a_part_dim < 0) return false;
    if (state.cosmo_a_part_dim > 0 &&
        (!state.cosmo_a_part || !state.cosmo_a_part_i ||
         !state.cosmo_a_part_j)) {
      return false;
    }
  }
  return true;
}

#ifdef __CUDACC__
__global__ void mozyme_helecz_kernel(int numat, int mpack, const int *iorbs,
                                     const int *nijbo, const double *p,
                                     const double *h, const double *f,
                                     double *atom_sums,
                                     double *atom_diag_sums, int *ok_out,
                                     const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int atom_i = blockIdx.x + 1;
  const int tid = threadIdx.x;
  if (tid != 0) return;

  double offdiag = 0.0;
  double diag = 0.0;
  if (atom_i <= numat) {
    const int ni = iorbs[atom_i - 1];
    if (ni <= 0) {
      atomicExch(ok_out, 0);
      atom_sums[blockIdx.x] = 0.0;
      atom_diag_sums[blockIdx.x] = 0.0;
      return;
    }
    for (int atom_j = 1; atom_j < atom_i; ++atom_j) {
      const int base = nijbo[(atom_i - 1) + (atom_j - 1) * numat];
      if (base < 0) continue;
      const int nj = iorbs[atom_j - 1];
      if (nj <= 0) {
        atomicExch(ok_out, 0);
        continue;
      }
      const int terms = ni * nj;
      if (base > mpack || terms < 0 || base + terms > mpack) {
        atomicExch(ok_out, 0);
        continue;
      }
      for (int offset = 0; offset < terms; ++offset) {
        const int idx = base + offset;
        offdiag += p[idx] * (h[idx] + f[idx]);
      }
    }

    const int base = nijbo[(atom_i - 1) + (atom_i - 1) * numat];
    if (base < 0) {
      atomicExch(ok_out, 0);
      atom_sums[blockIdx.x] = 0.0;
      atom_diag_sums[blockIdx.x] = 0.0;
      return;
    }
    const int terms = (ni * (ni + 1)) / 2;
    if (base > mpack || terms < 0 || base + terms > mpack) {
      atomicExch(ok_out, 0);
      atom_sums[blockIdx.x] = 0.0;
      atom_diag_sums[blockIdx.x] = 0.0;
      return;
    }
    int packed = 0;
    for (int row = 1; row <= ni; ++row) {
      for (int col = 1; col <= row; ++col) {
        if (row == col) {
          const int idx = base + packed;
          diag += p[idx] * (h[idx] + f[idx]);
        } else {
          const int idx = base + packed;
          offdiag += p[idx] * (h[idx] + f[idx]);
        }
        ++packed;
      }
    }
  }

  atom_sums[blockIdx.x] = offdiag;
  atom_diag_sums[blockIdx.x] = diag;
}

__global__ void mozyme_helecz_reduce_kernel(int count,
                                            const double *atom_sums,
                                            const double *atom_diag_sums,
                                            const int *ok_in,
                                            double *totals,
                                            const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  extern __shared__ double scratch[];
  double *offdiag_sums = scratch;
  double *diag_sums = scratch + blockDim.x;
  const int tid = threadIdx.x;

  if (ok_in && ok_in[kHeleczIntOk] != 1) {
    if (tid == 0) {
      totals[0] = 0.0;
      totals[1] = 0.0;
      totals[2] = 0.0;
    }
    return;
  }

  double offdiag = 0.0;
  double diag = 0.0;
  for (int idx = tid; idx < count; idx += blockDim.x) {
    offdiag += atom_sums[idx];
    diag += atom_diag_sums[idx];
  }
  offdiag_sums[tid] = offdiag;
  diag_sums[tid] = diag;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      offdiag_sums[tid] += offdiag_sums[tid + stride];
      diag_sums[tid] += diag_sums[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    totals[0] = offdiag_sums[0];
    totals[1] = diag_sums[0];
    totals[2] = offdiag_sums[0] + 0.5 * diag_sums[0];
  }
}

__global__ void mozyme_eimp_kernel(int numat, int mpack, const int *iorbs,
                                   const int *nijbo, const double *f,
                                   double *p, int *pair_updates,
                                   int *ok_out, int *expected_updates,
                                   const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int atom_i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int atom_j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (atom_i > numat || atom_j >= atom_i) return;

  const int ni = iorbs[atom_i - 1];
  const int nj = iorbs[atom_j - 1];
  const int terms = ni * nj;
  if (terms <= 0) return;

  const int base = nijbo[(atom_i - 1) + (atom_j - 1) * numat];
  if (base < 0) return;
  if (base + terms > mpack) {
    atomicExch(ok_out, 0);
    return;
  }
  atomicAdd(expected_updates, 1);

  double sum = 0.0;
  for (int offset = 0; offset < terms; ++offset) {
    const double value = f[base + offset];
    sum += value * value;
  }
  p[base] = sum;
  atomicAdd(pair_updates, 1);
}

__global__ void mozyme_update_status_finalize_kernel(int *status,
                                                     int require_work,
                                                     const int *resident_control_ints) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!status) return;
  if (resident_control_terminal(resident_control_ints)) return;
  const int actual = status[0];
  const int ok = status[1];
  const int expected = status[2];
  if (ok != 1 || actual != expected || (require_work && expected <= 0)) {
    status[1] = 0;
  }
}

__global__ void mozyme_set_int_slot_kernel(int *values, int slot, int value) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!values || slot < 0) return;
  values[slot] = value;
}

__global__ void mozyme_check_init_kernel(int nocc, int nvir, int *ints,
                                         const int *resident_control_ints) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!ints) return;
  if (resident_control_terminal(resident_control_ints)) return;
  ints[kCheckIntOccBad] = nocc + 1;
  ints[kCheckIntVirBad] = nvir + 1;
  ints[kCheckIntOk] = 1;
}

__global__ void mozyme_check_finalize_kernel(int nocc, int nvir,
                                             const double *errors,
                                             int *ints,
                                             const int *resident_control_ints) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!errors || !ints) return;
  if (resident_control_terminal(resident_control_ints)) return;
  const int ok = ints[kCheckIntOk];
  const int occ_bad = ints[kCheckIntOccBad];
  const int vir_bad = ints[kCheckIntVirBad];
  const double occ_error = errors[kCheckDoubleOccError];
  const double vir_error = errors[kCheckDoubleVirError];
  if (ok != 1 || occ_error > 0.1 || vir_error > 0.1 ||
      occ_bad <= nocc || vir_bad <= nvir) {
    ints[kCheckIntOk] = 0;
  }
}

__global__ void mozyme_resident_stage_reset_kernel(
    int *stage_ints, const int *stage_calls,
    int *resident_control_ints) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!stage_ints) return;
  if (resident_control_terminal(resident_control_ints)) return;
  if (!stage_calls || stage_calls[kResidentStageSlotUpload] <= 0) {
    stage_ints[kResidentStageCompleted] = 0;
    stage_ints[kResidentStageRequired] = kMozymeScfStageFull;
    stage_ints[kResidentStageMissing] = kMozymeScfStageFull;
    stage_ints[kResidentStageCode] = kMozymeScfUnsupported;
    if (resident_control_ints) {
      resident_control_ints[kResidentControlDecision] =
          kResidentDecisionStageFailed;
    }
    return;
  }
  stage_ints[kResidentStageCompleted] = kMozymeScfStageUpload;
  stage_ints[kResidentStageRequired] = kMozymeScfStageFull;
  stage_ints[kResidentStageMissing] =
      kMozymeScfStageFull & ~kMozymeScfStageUpload;
  stage_ints[kResidentStageCode] = kMozymeScfNotReady;
}

__global__ void mozyme_resident_stage_upload_accept_kernel(
    int *stage_calls, const int *resident_control_ints) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!stage_calls) return;
  if (resident_control_terminal(resident_control_ints)) return;
  stage_calls[kResidentStageSlotUpload] = 1;
}

__global__ void mozyme_resident_stage_mark_if_int_kernel(
    int *stage_ints, int *stage_calls, int stage_bits, const int *values,
    int value_slot, int expected_value, int stage_slot,
    int *resident_control_ints) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!stage_ints || !stage_calls || !values || value_slot < 0) return;
  if (resident_control_terminal(resident_control_ints)) return;
  if (values[value_slot] != expected_value) {
    stage_ints[kResidentStageRequired] = kMozymeScfStageFull;
    stage_ints[kResidentStageMissing] =
        kMozymeScfStageFull & ~stage_ints[kResidentStageCompleted];
    stage_ints[kResidentStageCode] = kMozymeScfUnsupported;
    if (resident_control_ints) {
      resident_control_ints[kResidentControlDecision] =
          kResidentDecisionStageFailed;
    }
    return;
  }
  const int completed = stage_ints[kResidentStageCompleted] | stage_bits;
  stage_ints[kResidentStageCompleted] = completed;
  if (stage_slot >= 0 && stage_slot < kResidentStageSlotCount) {
    stage_calls[stage_slot] += 1;
  }
  stage_ints[kResidentStageRequired] = kMozymeScfStageFull;
  stage_ints[kResidentStageMissing] = kMozymeScfStageFull & ~completed;
  stage_ints[kResidentStageCode] =
      (stage_ints[kResidentStageMissing] == 0) ? kMozymeScfSuccess
                                               : kMozymeScfUnsupported;
}

__global__ void mozyme_diagg1_aocc_kernel(int nocc, int icocc_dim,
                                          int cocc_dim, int numat,
                                          const int *ncf, const int *nncf,
                                          const int *ncocc,
                                          const int *icocc,
                                          const int *iorbs,
                                          const double *cocc,
                                          double *aocc, int *updated) {
  const int lmo = blockIdx.x + 1;
  const int local_atom = threadIdx.x + 1;
  if (lmo > nocc || local_atom > ncf[lmo - 1]) return;

  const int kk = nncf[lmo - 1] + local_atom;
  if (kk < 1 || kk > icocc_dim) return;

  int coeff_offset = 0;
  for (int item = 1; item < local_atom; ++item) {
    const int prev_kk = nncf[lmo - 1] + item;
    if (prev_kk >= 1 && prev_kk <= icocc_dim) {
      const int prev_atom = icocc[prev_kk - 1];
      if (prev_atom < 1 || prev_atom > numat) return;
      coeff_offset += iorbs[prev_atom - 1];
    }
  }

  const int atom = icocc[kk - 1];
  if (atom < 1 || atom > numat) return;
  const int norb = iorbs[atom - 1];
  const int base = ncocc[lmo - 1] + coeff_offset;
  if (norb <= 0 || base < 0 || base + norb > cocc_dim) return;
  double sum = 0.0;
  for (int orb = 0; orb < norb; ++orb) {
    const double value = cocc[base + orb];
    sum += value * value;
  }
  aocc[kk - 1] = sum;
  atomicAdd(updated, 1);
}

__global__ void mozyme_diagg1_avir_kernel(int nvir, int icvir_dim,
                                          int cvir_dim, int numat,
                                          const int *nce, const int *nnce,
                                          const int *ncvir,
                                          const int *icvir,
                                          const int *iorbs,
                                          const double *cvir,
                                          double *avir_cache,
                                          int *updated) {
  const int lmo = blockIdx.x + 1;
  const int local_atom = threadIdx.x + 1;
  if (lmo > nvir || local_atom > nce[lmo - 1]) return;

  const int jj = nnce[lmo - 1] + local_atom;
  if (jj < 1 || jj > icvir_dim) return;

  int coeff_offset = 0;
  for (int item = 1; item < local_atom; ++item) {
    const int prev_jj = nnce[lmo - 1] + item;
    if (prev_jj >= 1 && prev_jj <= icvir_dim) {
      const int prev_atom = icvir[prev_jj - 1];
      if (prev_atom < 1 || prev_atom > numat) return;
      coeff_offset += iorbs[prev_atom - 1];
    }
  }

  const int atom = icvir[jj - 1];
  if (atom < 1 || atom > numat) return;
  const int norb = iorbs[atom - 1];
  const int base = ncvir[lmo - 1] + coeff_offset;
  if (norb <= 0 || base < 0 || base + norb > cvir_dim) return;
  double sum = 0.0;
  for (int orb = 0; orb < norb; ++orb) {
    const double value = cvir[base + orb];
    sum += value * value;
  }
  avir_cache[jj - 1] = sum;
  atomicAdd(updated, 1);
}

__global__ void mozyme_diagg2_rotprep_kernel(
    int nij, int nocc, int nvir, const int *ifmo, const double *fmo, const double *eigs,
    const double *eigv, double shift, double rot_const, double tiny,
    double biglim, int *active, double *alpha, int *active_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nij) return;

  active[idx] = 0;
  alpha[idx] = 0.0;

  const double fmo_value = fmo[idx];
  if (!(fabs(fmo_value) >= tiny)) return;

  const int vir = ifmo[2 * idx];
  const int occ = ifmo[2 * idx + 1];
  if (vir <= 0 || vir > nvir || occ <= 0 || occ > nocc) return;

  const double c = fmo_value * rot_const;
  const double d = eigs[occ - 1] - eigv[vir - 1] - shift;
  if (!(fabs(c / d) >= biglim)) return;

  const double e = copysign(sqrt(4.0 * c * c + d * d), d);
  const double a = sqrt(0.5 * (1.0 + d / e));

  active[idx] = 1;
  alpha[idx] = a;
  atomicAdd(active_count, 1);
}

__device__ int mozyme_nijbo_at(const int *nijbo, int numat, int row,
                               int col) {
  if (!nijbo || row < 1 || row > numat || col < 1 || col > numat) return -1;
  return nijbo[(row - 1) + (col - 1) * numat];
}

__global__ void mozyme_diagg2_prepare_control_kernel(
    int idiagg, double bigeps, int previous_nrej0, int previous_nrej1,
    int fmo_dim, int *control_ints, double *control_scalars,
    const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;

  idiagg = resident_control_int_or(resident_control_ints,
                                   kResidentControlDiaggMode, idiagg);
  previous_nrej0 = resident_control_int_or(
      resident_control_ints, kResidentControlNrej0, previous_nrej0);
  previous_nrej1 = resident_control_int_or(
      resident_control_ints, kResidentControlNrej1, previous_nrej1);
  const int ok = control_ints[kDiaggIntOk];
  const int nij = control_ints[kDiaggIntNij];
  const int nf = control_ints[kDiaggIntNf];
  if (ok != 1 || nij < 0 || nij > fmo_dim || nf < 0 || nf > fmo_dim) {
    control_ints[kDiaggIntOk] = 0;
    control_ints[kDiaggIntNij] = 0;
    control_ints[kDiaggIntRetry] = 0;
    control_scalars[kDiaggDoubleRotateTiny] = 0.0;
    control_scalars[kDiaggDoubleBiglim] = 0.0;
    return;
  }

  mozyme_diagg2_thresholds_device(
      idiagg, control_scalars[kDiaggDoubleTiny], bigeps,
      control_scalars + kDiaggDoubleRotateTiny,
      control_scalars + kDiaggDoubleBiglim);
  control_ints[kDiaggIntRetry] =
      mozyme_diagg2_retry_device(previous_nrej0, previous_nrej1);
}

__global__ void mozyme_diagg2_finalize_control_kernel(
    int previous_nrej0, int fmo_dim, int *control_ints,
    const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;

  previous_nrej0 = resident_control_int_or(
      resident_control_ints, kResidentControlNrej0, previous_nrej0);
  const int nij = control_ints[kDiaggIntNij];
  const int nf = control_ints[kDiaggIntNf];
  if (control_ints[kDiaggIntOk] != 1 || nij < 0 || nij > fmo_dim || nf < 0 ||
      nf > fmo_dim) {
    control_ints[kDiaggIntOk] = 0;
  }
  control_ints[kDiaggIntNextNrej0] = control_ints[kDiaggIntNrej];
  control_ints[kDiaggIntNextNrej1] = previous_nrej0;
}

__global__ void mozyme_addhb_prepare_control_kernel(
    int current_iter, int addhb_requested, int nhb, int idiagg, double bigeps,
    int fmo_dim, const int *diagg_ints, const double *diagg_scalars,
    int *control_ints, double *control_scalars,
    const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) {
    control_ints[kAddhbIntOk] = 1;
    return;
  }

  current_iter = resident_control_int_or(
      resident_control_ints, kResidentControlCurrentIter, current_iter);
  addhb_requested = resident_control_int_or(
      resident_control_ints, kResidentControlAddhbDue, addhb_requested);
  nhb = resident_control_int_or(resident_control_ints,
                                kResidentControlNhb, nhb);
  idiagg = resident_control_int_or(resident_control_ints,
                                   kResidentControlDiaggMode, idiagg);
  const int diagg_ok = diagg_ints ? diagg_ints[kDiaggIntOk] : 0;
  const int diagg_nrej0 = diagg_ints ? diagg_ints[kDiaggIntNextNrej0] : 0;
  const int diagg_nrej1 = diagg_ints ? diagg_ints[kDiaggIntNextNrej1] : 0;
  const int diagg_nij = diagg_ints ? diagg_ints[kDiaggIntNij] : -1;
  const double input_tiny =
      diagg_scalars ? diagg_scalars[kDiaggDoubleRotateTiny] : 0.0;
  const double hblims[4] = {1.0, 0.1, 0.01, 0.001};
  const int diagg_ready =
      (diagg_ok == 1 && diagg_nij >= 0 && diagg_nij <= fmo_dim) ? 1 : 0;
  const int due = (diagg_ready != 0 && addhb_requested != 0 &&
                   (current_iter + 1) % 3 == 0 && nhb < 4)
                      ? 1
                      : 0;
  const int next_nhb = due != 0 ? nhb + 1 : nhb;
  const int next_idiagg = idiagg + 1;

  control_ints[kAddhbIntOk] = diagg_ready != 0 ? 0 : -1;
  control_ints[kAddhbIntNij] = 0;
  control_ints[kAddhbIntNrej] = 0;
  control_ints[kAddhbIntDue] = due;
  control_ints[kAddhbIntApplied] = due;
  control_ints[kAddhbIntNextNhb] = next_nhb;
  control_ints[kAddhbIntNextIdiagg] = next_idiagg;
  control_ints[kAddhbIntRetry] =
      mozyme_diagg2_retry_device(diagg_nrej0, diagg_nrej1);
  control_ints[kAddhbIntNextNrej0] = diagg_nrej0;
  control_ints[kAddhbIntNextNrej1] = diagg_nrej1;

  control_scalars[kAddhbDoubleSumb] = 0.0;
  control_scalars[kAddhbDoubleCutoff] =
      due != 0 ? hblims[next_nhb - 1] : 0.0;
  control_scalars[kAddhbDoubleNextTiny] = input_tiny;
  mozyme_diagg2_thresholds_device(
      next_idiagg, input_tiny, bigeps,
      control_scalars + kAddhbDoubleRotateTiny,
      control_scalars + kAddhbDoubleBiglim);
}

__global__ void mozyme_addhb_finalize_control_kernel(
    const int *diagg_ints, int fmo_dim, int *control_ints,
    double *control_scalars, const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;

  const int previous_nrej0 =
      diagg_ints ? diagg_ints[kDiaggIntNextNrej0] : 0;
  const int nij = control_ints[kAddhbIntNij];
  if (control_ints[kAddhbIntOk] != 1 || nij < 0 || nij > fmo_dim) {
    control_ints[kAddhbIntOk] = 0;
    return;
  }
  if (control_ints[kAddhbIntDue] != 0 && nij > 0) {
    control_ints[kAddhbIntNextNrej0] = control_ints[kAddhbIntNrej];
    control_ints[kAddhbIntNextNrej1] = previous_nrej0;
    control_scalars[kAddhbDoubleNextTiny] =
        control_scalars[kAddhbDoubleRotateTiny];
    if (control_ints[kAddhbIntNextIdiagg] % 2 == 1) {
      ++control_ints[kAddhbIntNextIdiagg];
    }
  }
}

// ---------------------------------------------------------------------------
// Parallel DIAGG (diagg1 / diagg2 / addhb / check) kernels.
//
// diagg1: one block per virtual LMO, threads over its coefficients and over
// occupied candidates; ifmo/fmo emitted with a deterministic two-pass
// (count, scan, fill) scheme so the layout matches the CPU ordering.
//
// diagg2: LMO pair rotations are scheduled with a "smallest pending pair at
// both endpoints" rule inside one cooperative kernel.  Every rotation then
// observes exactly the state the sequential CPU sweep would have produced
// (per-LMO rotation order is identical), while independent pairs run in
// parallel, one warp each.
// ---------------------------------------------------------------------------

namespace cg = cooperative_groups;

constexpr int kDiaggBlockThreads = 256;
constexpr int kDiaggMaxLmoAtoms = 1024;
constexpr int kDiaggMaxLmoCoeffs = 2048;
constexpr int kDiaggRotateThreads = 128;
constexpr int kDiaggRotateWarps = kDiaggRotateThreads / 32;
constexpr int kDiaggWorkIntError = 0;
constexpr int kDiaggWorkIntFlag0 = 1;
constexpr int kDiaggWorkIntFlag1 = 2;
constexpr int kDiaggWorkIntCount = 8;
constexpr int kDiaggWorkDoubleSumt = 0;
constexpr int kDiaggWorkDoubleTiny = 1;
constexpr int kDiaggWorkDoubleCount = 4;
constexpr int kPairStatePending = 0;
constexpr int kPairStateDone = 1;

inline std::size_t hbond_pair_capacity(int numat) {
  return static_cast<std::size_t>(std::max(1024, numat * 32));
}

__device__ inline double atomicMax_double(double *address, double value) {
  auto *addr = reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *addr;
  while (true) {
    const double current = __longlong_as_double(old);
    if (!(value > current)) break;
    const unsigned long long int assumed = old;
    old = atomicCAS(addr, assumed, __double_as_longlong(value));
    if (old == assumed) break;
  }
  return __longlong_as_double(old);
}

__device__ inline int warp_inclusive_scan_int(int value) {
  const int lane = threadIdx.x & 31;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, value, offset);
    if (lane >= offset) value += n;
  }
  return value;
}

__device__ inline double warp_sum_double(double value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ inline double warp_max_double(double value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmax(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

// Exclusive scan across a 256-thread block.  `warp_sums` is shared scratch of
// at least 8 ints.  Returns the exclusive prefix; `*total_out` gets the total.
__device__ inline int block_exclusive_scan_int(int value, int *warp_sums,
                                               int *total_out) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int inclusive = warp_inclusive_scan_int(value);
  if (lane == 31) warp_sums[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    int ws = lane < warps ? warp_sums[lane] : 0;
    __syncwarp();
    ws = warp_inclusive_scan_int(ws);
    if (lane < warps) warp_sums[lane] = ws;
  }
  __syncthreads();
  const int warp_prefix = warp > 0 ? warp_sums[warp - 1] : 0;
  const int total = warp_sums[warps - 1];
  __syncthreads();
  if (total_out) *total_out = total;
  return warp_prefix + inclusive - value;
}

__device__ inline double block_sum_double(double value, double *scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_sum_double(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  double total = 0.0;
  if (warp == 0) {
    double v = lane < warps ? scratch[lane] : 0.0;
    total = warp_sum_double(v);
  }
  __syncthreads();
  return total;
}

__device__ inline double block_max_double(double value, double *scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_max_double(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  double result = 0.0;
  if (warp == 0) {
    double v = lane < warps ? scratch[lane] : 0.0;
    result = warp_max_double(v);
  }
  __syncthreads();
  return result;
}

// Bitonic sort of `count` 64-bit keys held in shared memory (padded to a
// power of two with ~0ULL).  Block-wide; caller must __syncthreads before.
__device__ inline void block_bitonic_sort_u64(unsigned long long *keys,
                                              int padded) {
  for (int k = 2; k <= padded; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int idx = threadIdx.x; idx < padded; idx += blockDim.x) {
        const int ixj = idx ^ j;
        if (ixj > idx) {
          const bool ascending = (idx & k) == 0;
          const unsigned long long a = keys[idx];
          const unsigned long long b = keys[ixj];
          if ((a > b) == ascending) {
            keys[idx] = b;
            keys[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ inline unsigned long long lmo_sort_key(int atom, int entry) {
  return (static_cast<unsigned long long>(static_cast<unsigned>(atom)) << 32) |
         static_cast<unsigned>(entry);
}

// Binary search for `atom` in keys sorted by lmo_sort_key.  Returns the entry
// index or -1.
__device__ inline int find_lmo_entry(const unsigned long long *keys, int count,
                                     int atom) {
  int lo = 0;
  int hi = count - 1;
  const unsigned long long target = lmo_sort_key(atom, 0);
  const unsigned long long limit = lmo_sort_key(atom + 1, 0);
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    const unsigned long long key = keys[mid];
    if (key < target) {
      lo = mid + 1;
    } else if (key >= limit) {
      hi = mid - 1;
    } else {
      return static_cast<int>(key & 0xffffffffu);
    }
  }
  return -1;
}

// Packed Fock/density block element (row atom k1 with orbital i4, column atom
// j1 with orbital jx; both orbitals 0-based) at block base `base` (0-based).
__device__ inline int packed_block_index(int base, int k1, int j1, int nk,
                                         int nj, int i4, int jx) {
  if (k1 > j1) return base + i4 * nj + jx;
  if (k1 < j1) return base + jx * nk + i4;
  return i4 > jx ? base + (i4 * (i4 + 1)) / 2 + jx
                 : base + (jx * (jx + 1)) / 2 + i4;
}

// Per-entry squared norms of an LMO set (aocc for occupied, avir for virtual):
// one warp per LMO, lanes strided over its atom entries.
__global__ void mozyme_lmo_entry_norms_kernel(
    int nvec, int numat, int ic_dim, int c_dim, const int *nnc, const int *nc,
    const int *icvec, const int *ncvec, const int *iorbs, const double *cvec,
    double *entry_norms, int *error_flag,
    const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int lane = threadIdx.x & 31;
  const int lmo = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (lmo >= nvec) return;

  const int base = nnc[lmo];
  const int count = nc[lmo];
  const int coeff_base = ncvec[lmo];
  if (count < 0 || base < 0 || base + count > ic_dim) {
    if (lane == 0) atomicExch(error_flag, 1);
    return;
  }
  int running = 0;
  for (int chunk = 0; chunk < count; chunk += 32) {
    const int entry = chunk + lane;
    int atom = 0;
    int norb = 0;
    if (entry < count) {
      atom = icvec[base + entry];
      if (atom < 1 || atom > numat) {
        atomicExch(error_flag, 1);
      } else {
        norb = iorbs[atom - 1];
      }
    }
    const int inclusive = warp_inclusive_scan_int(norb);
    const int offset = running + inclusive - norb;
    if (entry < count) {
      double sum = 0.0;
      const int first = coeff_base + offset;
      if (first < 0 || first + norb > c_dim) {
        atomicExch(error_flag, 1);
      } else {
        for (int k = 0; k < norb; ++k) {
          const double v = cvec[first + k];
          sum += v * v;
        }
      }
      entry_norms[base + entry] = sum;
    }
    running += __shfl_sync(0xffffffffu, inclusive, 31);
  }
}

// Single-block exclusive scan of `count` ints; offsets[count] = total.  When
// `count_ptr` is given the element count is read from device memory.
__global__ void mozyme_exclusive_scan_kernel(int count, const int *count_ptr,
                                             const int *values, int *offsets,
                                             const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  if (count_ptr && *count_ptr < count) count = *count_ptr;
  if (count < 0) count = 0;
  __shared__ int warp_sums[32];
  __shared__ int carry;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (int chunk = 0; chunk < count; chunk += blockDim.x) {
    const int idx = chunk + threadIdx.x;
    const int v = idx < count ? values[idx] : 0;
    int total = 0;
    const int prefix = block_exclusive_scan_int(v, warp_sums, &total);
    if (idx < count) offsets[idx] = carry + prefix;
    __syncthreads();
    if (threadIdx.x == 0) carry += total;
    __syncthreads();
  }
  if (threadIdx.x == 0) offsets[count] = carry;
}

struct DiaggVirtualArgs {
  int nocc, nvir, numat, norbs, mpack;
  int icocc_dim, icvir_dim, cocc_dim, cvir_dim, fmo_dim;
  int idiagg;     // fallback; overridden by the resident control block
  int fill;       // even mode: 0 = count pass, 1 = fill pass
  int capacity;   // maximum number of ifmo entries
  const double *resident_control_scalars;
  const double *fao;
  const double *p;
  const int *ncf, *nce, *nncf, *nnce, *ncocc, *ncvir, *icocc, *icvir;
  const int *iorbs, *nijbo;
  const double *cocc, *cvir;
  const double *aocc;        // per icocc entry
  const double *avir_entry;  // per icvir entry
  double cutoff, flim, oldlim;
  double *eigv;
  int *nfmo;
  int *counts;         // per virtual (count pass output)
  const int *offsets;  // exclusive scan of counts (fill / odd mode)
  int *ifmo;
  double *fmo;
  double *work_scalars;
  int *work_ints;
  const int *resident_control_ints;
};

__global__ void __launch_bounds__(kDiaggBlockThreads)
mozyme_diagg1_virtual_kernel(DiaggVirtualArgs a) {
  __shared__ int s_atoms[kDiaggMaxLmoAtoms];
  __shared__ int s_off[kDiaggMaxLmoAtoms];
  __shared__ unsigned long long s_sorted[kDiaggMaxLmoAtoms];
  __shared__ double s_ws[kDiaggMaxLmoCoeffs];
  __shared__ double s_aov[kDiaggMaxLmoAtoms];
  __shared__ int s_warp[8];
  __shared__ double s_red[8];
  __shared__ int s_span;
  __shared__ int s_fail;
  __shared__ int s_running;

  if (resident_control_terminal(a.resident_control_ints)) return;
  const int i = blockIdx.x + 1;
  if (i > a.nvir) return;
  const int tid = threadIdx.x;

  const int idiagg = resident_control_int_or(a.resident_control_ints,
                                             kResidentControlDiaggMode,
                                             a.idiagg);
  double cutoff = a.cutoff;
  double flim = a.flim;
  double oldlim = a.oldlim;
  if (a.resident_control_scalars) {
    constexpr double cutlim = 1.0e-8;
    const double ovmax = a.resident_control_scalars[kResidentControlOvmax];
    cutoff = fmax(cutlim, ovmax * 10.0 * cutlim);
    flim = fmin(3.0,
                a.resident_control_scalars[kResidentControlDiaggFref] * 0.5);
    oldlim = a.resident_control_scalars[kResidentControlDiaggOldlim];
    if (idiagg <= 5) cutoff = cutlim;
  }
  const bool even_mode = (idiagg <= 5 || idiagg % 2 == 0);
  if (!even_mode && !a.fill) {
    if (tid == 0) a.counts[i - 1] = a.nfmo[i - 1];
    return;
  }

  const int nce_i = a.nce[i - 1];
  const int ibase = a.nnce[i - 1];
  const int loopi = a.ncvir[i - 1];
  if (tid == 0) {
    s_fail = 0;
    if (nce_i <= 0 || nce_i > kDiaggMaxLmoAtoms || ibase < 0 ||
        ibase + nce_i > a.icvir_dim) {
      s_fail = 1;
    }
  }
  __syncthreads();
  if (s_fail) {
    if (tid == 0) atomicExch(a.work_ints + kDiaggWorkIntError, 1);
    return;
  }

  // Atom list and orbital offsets (4 entries per thread + block scan).
  {
    int local[4];
    int local_sum = 0;
    for (int q = 0; q < 4; ++q) {
      const int e = tid * 4 + q;
      int norb = 0;
      if (e < nce_i) {
        const int atom = a.icvir[ibase + e];
        if (atom < 1 || atom > a.numat) {
          s_fail = 1;
        } else {
          norb = a.iorbs[atom - 1];
          s_atoms[e] = atom;
        }
      }
      local[q] = norb;
      local_sum += norb;
    }
    int total = 0;
    int prefix = block_exclusive_scan_int(local_sum, s_warp, &total);
    for (int q = 0; q < 4; ++q) {
      const int e = tid * 4 + q;
      if (e < nce_i) s_off[e] = prefix;
      prefix += local[q];
    }
    if (tid == 0) s_span = total;
  }
  __syncthreads();
  const int span = s_span;
  if (s_fail || span <= 0 || span > kDiaggMaxLmoCoeffs || loopi < 0 ||
      loopi + span > a.cvir_dim) {
    if (tid == 0) atomicExch(a.work_ints + kDiaggWorkIntError, 1);
    return;
  }

  // Sorted (atom, entry) keys for membership lookups.
  int padded = 1;
  while (padded < nce_i) padded <<= 1;
  for (int e = tid; e < padded; e += blockDim.x) {
    s_sorted[e] = e < nce_i ? lmo_sort_key(s_atoms[e], e) : ~0ULL;
  }
  __syncthreads();
  block_bitonic_sort_u64(s_sorted, padded);

  // ws = F . c_vir restricted to the LMO's atom blocks.
  for (int c = tid; c < span; c += blockDim.x) {
    int lo = 0;
    int hi = nce_i - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (s_off[mid] <= c) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    const int jj = lo;
    const int j1 = s_atoms[jj];
    const int nj = a.iorbs[j1 - 1];
    const int jx = c - s_off[jj];
    double ws = 0.0;
    for (int kk = 0; kk < nce_i; ++kk) {
      const int k1 = s_atoms[kk];
      const int kj = mozyme_nijbo_at(a.nijbo, a.numat, k1, j1);
      if (kj < 0) continue;
      if (!(a.avir_entry[ibase + kk] * a.p[kj] > cutoff)) continue;
      const int nk = a.iorbs[k1 - 1];
      const int cbase = loopi + s_off[kk];
      for (int i4 = 0; i4 < nk; ++i4) {
        const int fidx = packed_block_index(kj, k1, j1, nk, nj, i4, jx);
        if (fidx < 0 || fidx >= a.mpack) {
          s_fail = 1;
          break;
        }
        ws += a.fao[fidx] * a.cvir[cbase + i4];
      }
    }
    s_ws[c] = ws;
  }
  __syncthreads();
  if (s_fail) {
    if (tid == 0) atomicExch(a.work_ints + kDiaggWorkIntError, 1);
    return;
  }

  // aov per entry and the virtual energy level.
  double eig_part = 0.0;
  for (int e = tid; e < nce_i; e += blockDim.x) {
    const int norb = a.iorbs[s_atoms[e] - 1];
    const int off = s_off[e];
    double sum = 0.0;
    for (int k = 0; k < norb; ++k) sum += s_ws[off + k] * s_ws[off + k];
    s_aov[e] = sum;
    if (sum * a.avir_entry[ibase + e] > cutoff) {
      for (int k = 0; k < norb; ++k) {
        eig_part += s_ws[off + k] * a.cvir[loopi + off + k];
      }
    }
  }
  __syncthreads();
  const double eigv_i = block_sum_double(eig_part, s_red);
  if (a.fill && tid == 0) a.eigv[i - 1] = eigv_i;

  double sumt_local = 0.0;
  double tiny_local = 0.0;

  if (even_mode) {
    const int i1 = s_atoms[0];
    const int i2 = nce_i > 1 ? s_atoms[1] : i1;
    const int base = a.fill ? a.offsets[i - 1] : 0;
    if (tid == 0) s_running = 0;
    __syncthreads();

    for (int chunk = 0; chunk < a.nocc; chunk += blockDim.x) {
      const int j = chunk + tid + 1;
      int emit = 0;
      double value = 0.0;
      if (j <= a.nocc) {
        const int jbase = a.nncf[j - 1];
        const int ncf_j = a.ncf[j - 1];
        if (jbase < 0 || ncf_j <= 0 || jbase + ncf_j > a.icocc_dim) {
          s_fail = 1;
        } else {
          const int j1 = a.icocc[jbase];
          const int j2 = ncf_j > 1 ? a.icocc[jbase + 1] : j1;
          const int i1j1 = mozyme_nijbo_at(a.nijbo, a.numat, i1, j1);
          const int i1j2 = mozyme_nijbo_at(a.nijbo, a.numat, i1, j2);
          const int i2j1 = mozyme_nijbo_at(a.nijbo, a.numat, i2, j1);
          const int i2j2 = mozyme_nijbo_at(a.nijbo, a.numat, i2, j2);
          if (i1j1 >= 0 || i1j2 >= 0 || i2j1 >= 0 || i2j2 >= 0) {
            double pre = 0.0;
            if (i1j1 >= 0) pre = fabs(a.fao[i1j1]);
            if (i1j2 >= 0) pre += fabs(a.fao[i1j2]);
            if (i2j1 >= 0) pre += fabs(a.fao[i2j1]);
            if (i2j2 >= 0) pre += fabs(a.fao[i2j2]);
            if (pre >= flim) {
              const int loopj = a.ncocc[j - 1];
              bool lij = false;
              double sum = 0.0;
              int kl = 0;
              for (int kk = 0; kk < ncf_j; ++kk) {
                const int k1 = a.icocc[jbase + kk];
                const int norb = a.iorbs[k1 - 1];
                const int e = find_lmo_entry(s_sorted, padded, k1);
                if (e < 0 || a.aocc[jbase + kk] * s_aov[e] < cutoff) {
                  kl += norb;
                  continue;
                }
                lij = true;
                const int off = s_off[e];
                const int cbase = loopj + kl;
                if (cbase < 0 || cbase + norb > a.cocc_dim) {
                  s_fail = 1;
                  break;
                }
                for (int k = 0; k < norb; ++k) {
                  sum += s_ws[off + k] * a.cocc[cbase + k];
                }
                kl += norb;
              }
              sumt_local += fabs(sum);
              tiny_local = fmax(tiny_local, fabs(sum));
              if (lij && fabs(sum) > oldlim) {
                emit = 1;
                value = sum;
              }
            }
          }
        }
      }
      int chunk_total = 0;
      const int rank = block_exclusive_scan_int(emit, s_warp, &chunk_total);
      if (a.fill && emit) {
        const int pos = base + s_running + rank;
        if (pos < a.capacity) {
          a.ifmo[2 * pos] = i;
          a.ifmo[2 * pos + 1] = j;
          a.fmo[pos] = value;
        }
      }
      __syncthreads();
      if (tid == 0) s_running += chunk_total;
      __syncthreads();
    }
    if (tid == 0) {
      if (a.fill) {
        int emitted = s_running;
        const int room = a.capacity - base;
        if (emitted > room) emitted = room < 0 ? 0 : room;
        a.nfmo[i - 1] = emitted;
      } else {
        a.counts[i - 1] = s_running;
      }
    }
  } else {
    const int base = a.offsets[i - 1];
    const int n = a.nfmo[i - 1];
    for (int t = tid; t < n; t += blockDim.x) {
      const int pos = base + t;
      if (pos >= a.capacity) break;
      const int j = a.ifmo[2 * pos + 1];
      if (j < 1 || j > a.nocc) {
        s_fail = 1;
        break;
      }
      const int jbase = a.nncf[j - 1];
      const int ncf_j = a.ncf[j - 1];
      const int loopj = a.ncocc[j - 1];
      double sum = 0.0;
      int kl = 0;
      for (int kk = 0; kk < ncf_j; ++kk) {
        const int k1 = a.icocc[jbase + kk];
        const int norb = a.iorbs[k1 - 1];
        const int e = find_lmo_entry(s_sorted, padded, k1);
        if (e < 0 || s_aov[e] * a.aocc[jbase + kk] < cutoff) {
          kl += norb;
          continue;
        }
        const int off = s_off[e];
        const int cbase = loopj + kl;
        if (cbase < 0 || cbase + norb > a.cocc_dim) {
          s_fail = 1;
          break;
        }
        for (int k = 0; k < norb; ++k) sum += s_ws[off + k] * a.cocc[cbase + k];
        kl += norb;
      }
      sumt_local += fabs(sum);
      tiny_local = fmax(tiny_local, fabs(sum));
      a.fmo[pos] = sum;
    }
  }
  __syncthreads();
  if (s_fail) {
    if (tid == 0) atomicExch(a.work_ints + kDiaggWorkIntError, 1);
    return;
  }
  if (a.fill) {
    const double sumt_block = block_sum_double(sumt_local, s_red);
    const double tiny_block = block_max_double(tiny_local, s_red);
    if (tid == 0) {
      atomicAdd_double(a.work_scalars + kDiaggWorkDoubleSumt, sumt_block);
      atomicMax_double(a.work_scalars + kDiaggWorkDoubleTiny, tiny_block);
    }
  }
}

// Occupied LMO energy levels (diagg1 tail): one block per occupied LMO.
__global__ void __launch_bounds__(kDiaggBlockThreads)
mozyme_diagg1_occupied_eigs_kernel(
    int nocc, int numat, int mpack, int icocc_dim, int cocc_dim,
    const double *fao, const double *p, const int *ncf, const int *nncf,
    const int *ncocc, const int *icocc, const int *iorbs, const int *nijbo,
    const double *cocc, const double *aocc, double cutoff, int idiagg,
    double *eigs, int *work_ints, const int *resident_control_ints,
    const double *resident_control_scalars) {
  __shared__ int s_atoms[kDiaggMaxLmoAtoms];
  __shared__ int s_off[kDiaggMaxLmoAtoms];
  __shared__ int s_warp[8];
  __shared__ double s_red[8];
  __shared__ int s_fail;

  if (resident_control_terminal(resident_control_ints)) return;
  idiagg = resident_control_int_or(resident_control_ints,
                                   kResidentControlDiaggMode, idiagg);
  if (idiagg > 2 && idiagg % 4 != 0) return;
  if (resident_control_scalars) {
    constexpr double cutlim = 1.0e-8;
    const double ovmax = resident_control_scalars[kResidentControlOvmax];
    cutoff = fmax(cutlim, ovmax * 10.0 * cutlim);
    if (idiagg <= 5) cutoff = cutlim;
  }
  const int i = blockIdx.x + 1;
  if (i > nocc) return;
  const int tid = threadIdx.x;
  const int ncf_i = ncf[i - 1];
  const int jbase = nncf[i - 1];
  const int loopi = ncocc[i - 1];
  if (tid == 0) {
    s_fail = (ncf_i <= 0 || ncf_i > kDiaggMaxLmoAtoms || jbase < 0 ||
              jbase + ncf_i > icocc_dim)
                 ? 1
                 : 0;
  }
  __syncthreads();
  if (s_fail) {
    if (tid == 0) atomicExch(work_ints + kDiaggWorkIntError, 1);
    return;
  }
  {
    int local[4];
    int local_sum = 0;
    for (int q = 0; q < 4; ++q) {
      const int e = tid * 4 + q;
      int norb = 0;
      if (e < ncf_i) {
        const int atom = icocc[jbase + e];
        if (atom < 1 || atom > numat) {
          s_fail = 1;
        } else {
          norb = iorbs[atom - 1];
          s_atoms[e] = atom;
        }
      }
      local[q] = norb;
      local_sum += norb;
    }
    int total = 0;
    int prefix = block_exclusive_scan_int(local_sum, s_warp, &total);
    for (int q = 0; q < 4; ++q) {
      const int e = tid * 4 + q;
      if (e < ncf_i) s_off[e] = prefix;
      prefix += local[q];
    }
    if (tid == 0 && (loopi < 0 || loopi + total > cocc_dim)) s_fail = 1;
  }
  __syncthreads();
  if (s_fail) {
    if (tid == 0) atomicExch(work_ints + kDiaggWorkIntError, 1);
    return;
  }

  double part = 0.0;
  const int pairs = ncf_i * ncf_i;
  for (int pr = tid; pr < pairs; pr += blockDim.x) {
    const int je = pr / ncf_i;
    const int ke = pr - je * ncf_i;
    const int j1 = s_atoms[je];
    const int k1 = s_atoms[ke];
    const int k1j1 = mozyme_nijbo_at(nijbo, numat, k1, j1);
    if (k1j1 < 0) continue;
    const double ak = aocc[jbase + ke];
    if (!(ak * p[k1j1] * ak >= cutoff)) continue;
    const int nj = iorbs[j1 - 1];
    const int nk = iorbs[k1 - 1];
    const int jl = loopi + s_off[je];
    const int kl = loopi + s_off[ke];
    for (int jx = 0; jx < nj; ++jx) {
      double sum1 = 0.0;
      for (int i4 = 0; i4 < nk; ++i4) {
        const int fidx = packed_block_index(k1j1, k1, j1, nk, nj, i4, jx);
        if (fidx < 0 || fidx >= mpack) {
          s_fail = 1;
          break;
        }
        sum1 += fao[fidx] * cocc[kl + i4];
      }
      part += cocc[jl + jx] * sum1;
    }
  }
  __syncthreads();
  const double total = block_sum_double(part, s_red);
  if (tid == 0) {
    if (s_fail) {
      atomicExch(work_ints + kDiaggWorkIntError, 1);
    } else {
      eigs[i - 1] = total;
    }
  }
}

__global__ void mozyme_diagg1_finalize_kernel(
    int nvir, int capacity, int idiagg, int nf_in, double safety_in,
    double oldlim_in, const int *offsets, const int *nfmo,
    const int *work_ints, const double *work_scalars, int *nij_out,
    int *ijc_out, int *nf_out, double *sumt_out, double *tiny_out,
    double *fref_out, double *oldlim_out, double *safety_out, int *ok_out,
    const int *resident_control_ints,
    const double *resident_control_scalars) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) {
    *ok_out = 1;
    return;
  }
  idiagg = resident_control_int_or(resident_control_ints,
                                   kResidentControlDiaggMode, idiagg);
  nf_in = resident_control_int_or(resident_control_ints,
                                  kResidentControlDiaggNf, nf_in);
  if (resident_control_scalars) {
    oldlim_in = resident_control_scalars[kResidentControlDiaggOldlim];
    safety_in = resident_control_scalars[kResidentControlDiaggSafety];
  }
  const int even_mode = (idiagg <= 5 || idiagg % 2 == 0) ? 1 : 0;
  if (work_ints[kDiaggWorkIntError] != 0) {
    *ok_out = 0;
    return;
  }
  int ijc = offsets[nvir];
  if (ijc > capacity) ijc = capacity;
  double safety = safety_in;
  if (ijc == capacity) {
    safety *= 2.0;
  } else {
    safety = fmax(safety * 0.5, 1.0);
  }
  const double tiny = work_scalars[kDiaggWorkDoubleTiny];
  double oldlim = oldlim_in;
  if (!(idiagg > 2 && idiagg % 4 != 0)) oldlim = tiny * safety * 1.0e-3;

  *nij_out = ijc;
  *ijc_out = ijc;
  *nf_out = even_mode ? nfmo[nvir - 1] : nf_in;
  *sumt_out = work_scalars[kDiaggWorkDoubleSumt];
  *tiny_out = tiny;
  *fref_out = tiny * tiny * tiny * tiny;
  *oldlim_out = oldlim;
  *safety_out = safety;
  *ok_out = 1;
}

// ---- diagg2: parallel Jacobi rotations ------------------------------------

struct DiaggRotateArgs {
  const int *control_ints;
  int nij_slot, retry_slot;
  const double *control_scalars;
  int tiny_slot, biglim_slot;
  int nocc, nvir, numat, norbs, icocc_dim, icvir_dim, cocc_dim, cvir_dim;
  const int *ifmo;
  const double *fmo;
  const double *eigs, *eigv;
  const int *nncf;
  int *ncf;
  const int *ncocc;
  int *icocc;
  const int *nnce;
  int *nce;
  const int *ncvir;
  int *icvir;
  const int *iorbs;
  double *cocc, *cvir;
  double shift, rot_const, thresh;
  int *pair_state;
  int *vclaim, *oclaim;
  int *work_ints;
  double *sumb_out;
  int *nrej_out;
  int *ok_slot;
  const int *resident_control_ints;
  const double *resident_control_scalars;
};

// Exclusive orbital offsets for an LMO atom list, written to `off`; returns
// the total coefficient count.  Warp-collective.
// Arrays mutated by other SMs between cooperative rounds (LMO lists,
// coefficients, pair state) must be read through L2, never from a stale L1.
__device__ inline int ld_shared_int(const int *p) { return __ldcg(p); }
__device__ inline double ld_shared_double(const double *p) { return __ldcg(p); }

__device__ inline int warp_build_offsets(const int *list, int count,
                                         const int *iorbs, int numat,
                                         int *off, int *error) {
  const int lane = threadIdx.x & 31;
  int running = 0;
  for (int chunk = 0; chunk < count; chunk += 32) {
    const int e = chunk + lane;
    int norb = 0;
    if (e < count) {
      const int atom = ld_shared_int(list + e);
      if (atom < 1 || atom > numat) {
        *error = 1;
      } else {
        norb = iorbs[atom - 1];
      }
    }
    const int inclusive = warp_inclusive_scan_int(norb);
    if (e < count) off[e] = running + inclusive - norb;
    running += __shfl_sync(0xffffffffu, inclusive, 31);
  }
  return running;
}

__device__ inline int warp_find_in_list(const int *list, int count,
                                        int atom) {
  int pos = -1;
  for (int k = 0; k < count; ++k) {
    if (ld_shared_int(list + k) == atom) pos = k;
  }
  return pos;
}

// One warp rotates occupied LMO j against virtual LMO i (1-based).  Mirrors
// the sequential diagg2 loop body, including LMO growth and rejection.
__device__ void warp_rotate_pair(const DiaggRotateArgs &a, int ij, int retry,
                                 int *joff, int *ioff, double &sumb_acc,
                                 int &nrej_acc, int &error) {
  const int lane = threadIdx.x & 31;
  const int i = a.ifmo[2 * ij];
  const int j = a.ifmo[2 * ij + 1];
  const double c = a.fmo[ij] * a.rot_const;
  const double d = a.eigs[j - 1] - a.eigv[i - 1] - a.shift;

  const int ncfj0 = ld_shared_int(a.ncf + j - 1);
  const int ncei0 = ld_shared_int(a.nce + i - 1);
  const int jbase = a.nncf[j - 1];
  const int ibase = a.nnce[i - 1];
  const int loopj = a.ncocc[j - 1];
  const int loopi = a.ncvir[i - 1];
  if (ncfj0 < 0 || ncei0 < 0 || ncfj0 > kDiaggMaxLmoAtoms ||
      ncei0 > kDiaggMaxLmoAtoms || jbase < 0 || ibase < 0 ||
      jbase + ncfj0 > a.icocc_dim || ibase + ncei0 > a.icvir_dim) {
    error = 1;
    return;
  }
  int jur = (j != a.nocc) ? a.ncocc[j] : a.cocc_dim;
  const int jncf = (j != a.nocc) ? a.nncf[j] : a.icocc_dim;
  if (jur > loopj + a.norbs) jur = loopj + a.norbs;
  int iur = (i != a.nvir) ? a.ncvir[i] : a.cvir_dim;
  const int incv = (i != a.nvir) ? a.nnce[i] : a.icvir_dim;
  if (iur > loopi + a.norbs) iur = loopi + a.norbs;

  const int *jlist = a.icocc + jbase;
  const int *ilist = a.icvir + ibase;
  int local_error = 0;
  const int mlf0 = warp_build_offsets(jlist, ncfj0, a.iorbs, a.numat, joff,
                                      &local_error);
  const int mle0 = warp_build_offsets(ilist, ncei0, a.iorbs, a.numat, ioff,
                                      &local_error);
  __syncwarp();
  if (__any_sync(0xffffffffu, local_error != 0) || loopj < 0 || loopi < 0 ||
      loopj + mlf0 > a.cocc_dim || loopi + mle0 > a.cvir_dim) {
    error = 1;
    return;
  }

  const double e = copysign(sqrt(4.0 * c * c + d * d), d);
  double alpha = sqrt(0.5 * (1.0 + d / e));

  while (true) {
    const double beta = -copysign(sqrt(fmax(0.0, 1.0 - alpha * alpha)), c);
    if (lane == 0) sumb_acc += fabs(beta);
    const double beta2 = beta * beta;

    // Count growth: virtual atoms missing from the occupied LMO.
    int n_new_occ = 0;
    int orb_new_occ = 0;
    for (int chunk = 0; chunk < ncei0; chunk += 32) {
      const int le = chunk + lane;
      int flag = 0;
      int norb = 0;
      if (le < ncei0) {
        const int mie = ld_shared_int(ilist + le);
        if (warp_find_in_list(jlist, ncfj0, mie) < 0) {
          norb = a.iorbs[mie - 1];
          const int cb = loopi + ioff[le];
          double s = 0.0;
          for (int k = 0; k < norb; ++k) {
            const double v = ld_shared_double(a.cvir + cb + k);
            s += v * v;
          }
          flag = (beta2 * s > a.thresh) ? 1 : 0;
        }
      }
      const unsigned mask = __ballot_sync(0xffffffffu, flag);
      n_new_occ += __popc(mask);
      orb_new_occ += __shfl_sync(0xffffffffu,
                                 warp_inclusive_scan_int(flag ? norb : 0), 31);
    }
    // Occupied atoms missing from the virtual LMO.
    int n_new_vir = 0;
    int orb_new_vir = 0;
    for (int chunk = 0; chunk < ncfj0; chunk += 32) {
      const int lf = chunk + lane;
      int flag = 0;
      int norb = 0;
      if (lf < ncfj0) {
        const int ii = ld_shared_int(jlist + lf);
        if (warp_find_in_list(ilist, ncei0, ii) < 0) {
          norb = a.iorbs[ii - 1];
          const int cb = loopj + joff[lf];
          double s = 0.0;
          for (int k = 0; k < norb; ++k) {
            const double v = ld_shared_double(a.cocc + cb + k);
            s += v * v;
          }
          flag = (beta2 * s > a.thresh) ? 1 : 0;
        }
      }
      const unsigned mask = __ballot_sync(0xffffffffu, flag);
      n_new_vir += __popc(mask);
      orb_new_vir += __shfl_sync(0xffffffffu,
                                 warp_inclusive_scan_int(flag ? norb : 0), 31);
    }

    const bool reject = (jbase + ncfj0 + n_new_occ > jncf) ||
                        (loopj + mlf0 + orb_new_occ > jur) ||
                        (ibase + ncei0 + n_new_vir > incv) ||
                        (loopi + mle0 + orb_new_vir > iur);
    if (reject) {
      if (lane == 0) ++nrej_acc;
      if (retry != 0) {
        alpha = 0.5 * (alpha + 1.0);
        continue;
      }
      return;
    }

    // Apply: rotate common atoms, append new occupied atoms in virtual order.
    int running_new = 0;
    int running_orb = 0;
    for (int chunk = 0; chunk < ncei0; chunk += 32) {
      const int le = chunk + lane;
      int flag = 0;
      int norb = 0;
      int jpos = -1;
      int mie = 0;
      if (le < ncei0) {
        mie = ld_shared_int(ilist + le);
        norb = a.iorbs[mie - 1];
        jpos = warp_find_in_list(jlist, ncfj0, mie);
        if (jpos >= 0) {
          const int cb_i = loopi + ioff[le];
          const int cb_j = loopj + joff[jpos];
          for (int k = 0; k < norb; ++k) {
            const double av = ld_shared_double(a.cocc + cb_j + k);
            const double bv = ld_shared_double(a.cvir + cb_i + k);
            a.cocc[cb_j + k] = alpha * av + beta * bv;
            a.cvir[cb_i + k] = alpha * bv - beta * av;
          }
        } else {
          const int cb = loopi + ioff[le];
          double s = 0.0;
          for (int k = 0; k < norb; ++k) {
            const double v = ld_shared_double(a.cvir + cb + k);
            s += v * v;
          }
          flag = (beta2 * s > a.thresh) ? 1 : 0;
        }
      }
      const unsigned mask = __ballot_sync(0xffffffffu, flag);
      const int rank = __popc(mask & ((1u << lane) - 1u));
      const int orb_incl = warp_inclusive_scan_int(flag ? norb : 0);
      const int orb_prefix = orb_incl - (flag ? norb : 0);
      if (flag) {
        const int slot = ncfj0 + running_new + rank;
        a.icocc[jbase + slot] = mie;
        const int cb_j = loopj + mlf0 + running_orb + orb_prefix;
        const int cb_i = loopi + ioff[le];
        for (int k = 0; k < norb; ++k) {
          const double v = ld_shared_double(a.cvir + cb_i + k);
          a.cocc[cb_j + k] = beta * v;
          a.cvir[cb_i + k] = alpha * v;
        }
      }
      running_new += __popc(mask);
      running_orb += __shfl_sync(0xffffffffu, orb_incl, 31);
    }
    // Append new virtual atoms in occupied order (original entries only).
    running_new = 0;
    running_orb = 0;
    for (int chunk = 0; chunk < ncfj0; chunk += 32) {
      const int lf = chunk + lane;
      int flag = 0;
      int norb = 0;
      int ii = 0;
      if (lf < ncfj0) {
        ii = ld_shared_int(jlist + lf);
        if (warp_find_in_list(ilist, ncei0, ii) < 0) {
          norb = a.iorbs[ii - 1];
          const int cb = loopj + joff[lf];
          double s = 0.0;
          for (int k = 0; k < norb; ++k) {
            const double v = ld_shared_double(a.cocc + cb + k);
            s += v * v;
          }
          flag = (beta2 * s > a.thresh) ? 1 : 0;
        }
      }
      const unsigned mask = __ballot_sync(0xffffffffu, flag);
      const int rank = __popc(mask & ((1u << lane) - 1u));
      const int orb_incl = warp_inclusive_scan_int(flag ? norb : 0);
      const int orb_prefix = orb_incl - (flag ? norb : 0);
      if (flag) {
        const int slot = ncei0 + running_new + rank;
        a.icvir[ibase + slot] = ii;
        const int cb_i = loopi + mle0 + running_orb + orb_prefix;
        const int cb_j = loopj + joff[lf];
        for (int k = 0; k < norb; ++k) {
          const double v = ld_shared_double(a.cocc + cb_j + k);
          a.cvir[cb_i + k] = -beta * v;
          a.cocc[cb_j + k] = alpha * v;
        }
      }
      running_new += __popc(mask);
      running_orb += __shfl_sync(0xffffffffu, orb_incl, 31);
    }
    __syncwarp();
    if (lane == 0) {
      a.ncf[j - 1] = ncfj0 + n_new_occ;
      a.nce[i - 1] = ncei0 + n_new_vir;
    }
    return;
  }
}

__global__ void __launch_bounds__(kDiaggRotateThreads)
mozyme_diagg2_parallel_kernel(DiaggRotateArgs a) {
  __shared__ int s_joff[kDiaggRotateWarps][kDiaggMaxLmoAtoms];
  __shared__ int s_ioff[kDiaggRotateWarps][kDiaggMaxLmoAtoms];

  cg::grid_group grid = cg::this_grid();
  if (resident_control_terminal(a.resident_control_ints)) return;

  const int nij = a.control_ints[a.nij_slot];
  const int retry = a.control_ints[a.retry_slot];
  const double tiny = a.control_scalars[a.tiny_slot];
  const double biglim = a.control_scalars[a.biglim_slot];
  a.shift = resident_control_double_or(a.resident_control_scalars,
                                       kResidentControlShift, a.shift);
  if (nij <= 0) return;

  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gthreads = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int gwarp = gtid >> 5;
  const int gwarps = gthreads >> 5;

  for (int idx = gtid; idx < a.nvir; idx += gthreads) a.vclaim[idx] = INT_MAX;
  for (int idx = gtid; idx < a.nocc; idx += gthreads) a.oclaim[idx] = INT_MAX;
  for (int ij = gtid; ij < nij; ij += gthreads) {
    const int i = a.ifmo[2 * ij];
    const int j = a.ifmo[2 * ij + 1];
    int state = kPairStateDone;
    if (i >= 1 && i <= a.nvir && j >= 1 && j <= a.nocc) {
      const double f = a.fmo[ij];
      if (fabs(f) >= tiny) {
        const double c = f * a.rot_const;
        const double d = a.eigs[j - 1] - a.eigv[i - 1] - a.shift;
        if (fabs(c / d) >= biglim) state = kPairStatePending;
      }
    }
    a.pair_state[ij] = state;
  }
  if (gtid == 0) {
    a.work_ints[kDiaggWorkIntFlag0] = 0;
    a.work_ints[kDiaggWorkIntFlag1] = 0;
  }
  grid.sync();

  double sumb_acc = 0.0;
  int nrej_acc = 0;
  int error = 0;
  for (int round = 0; round <= nij; ++round) {
    int *flag = a.work_ints + (round & 1 ? kDiaggWorkIntFlag1
                                         : kDiaggWorkIntFlag0);
    int *next_flag = a.work_ints + (round & 1 ? kDiaggWorkIntFlag0
                                              : kDiaggWorkIntFlag1);
    for (int ij = gtid; ij < nij; ij += gthreads) {
      if (ld_shared_int(a.pair_state + ij) != kPairStatePending) continue;
      atomicMin(a.vclaim + a.ifmo[2 * ij] - 1, ij);
      atomicMin(a.oclaim + a.ifmo[2 * ij + 1] - 1, ij);
      *flag = 1;
    }
    grid.sync();
    if (ld_shared_int(flag) == 0) break;
    if (gtid == 0) *next_flag = 0;

    for (int ij = gwarp; ij < nij; ij += gwarps) {
      if (ld_shared_int(a.pair_state + ij) != kPairStatePending) continue;
      const int i = a.ifmo[2 * ij];
      const int j = a.ifmo[2 * ij + 1];
      if (ld_shared_int(a.vclaim + i - 1) != ij ||
          ld_shared_int(a.oclaim + j - 1) != ij) {
        continue;
      }
      warp_rotate_pair(a, ij, retry, s_joff[warp_in_block],
                       s_ioff[warp_in_block], sumb_acc, nrej_acc, error);
      __syncwarp();
      if (lane == 0) {
        a.pair_state[ij] = kPairStateDone;
        a.vclaim[i - 1] = INT_MAX;
        a.oclaim[j - 1] = INT_MAX;
      }
    }
    grid.sync();
  }

  if (lane == 0) {
    if (sumb_acc != 0.0) atomicAdd_double(a.sumb_out, sumb_acc);
    if (nrej_acc != 0) atomicAdd(a.nrej_out, nrej_acc);
    if (error != 0) {
      atomicExch(a.work_ints + kDiaggWorkIntError, 1);
      if (a.ok_slot) atomicExch(a.ok_slot, 0);
    }
  }
}

// ---- addhb: hydrogen-bond pair discovery ----------------------------------

__device__ inline bool hbond_pair_qualifies(int atom_i, int atom_j, int numat,
                                            int mpack, const int *iorbs,
                                            const int *nijbo,
                                            const double *fao,
                                            const double *p, double cutoff,
                                            int *error) {
  const int base = mozyme_nijbo_at(nijbo, numat, atom_i, atom_j);
  if (base < 0) return false;
  const int ni = iorbs[atom_i - 1];
  const int nj = iorbs[atom_j - 1];
  const int terms = ni * nj;
  if (ni <= 0 || nj <= 0 || base + terms > mpack) {
    *error = 1;
    return false;
  }
  double sumf = 0.0;
  double sump = 0.0;
  for (int t = 0; t < terms; ++t) {
    sumf += fao[base + t] * fao[base + t];
    sump += p[base + t] * p[base + t];
  }
  return sump < 1.0e-10 && sumf > cutoff;
}

// Block per atom i; threads over j < i.  fill=0 counts, fill=1 writes pairs.
__global__ void __launch_bounds__(kDiaggBlockThreads)
mozyme_hbond_pairs_kernel(int fill, int numat, int mpack, int pair_capacity,
                          const int *iorbs, const int *nijbo,
                          const double *fao, const double *p,
                          const int *control_ints,
                          const double *control_scalars, int *counts,
                          const int *offsets, int *pair_i, int *pair_j,
                          int *work_ints, const int *resident_control_ints) {
  __shared__ int s_warp[8];
  __shared__ int s_running;
  if (resident_control_terminal(resident_control_ints)) return;
  if (control_ints[kAddhbIntOk] < 0 || control_ints[kAddhbIntDue] == 0) {
    if (!fill && threadIdx.x == 0) counts[blockIdx.x] = 0;
    return;
  }
  const int atom_i = blockIdx.x + 1;
  if (atom_i > numat) return;
  const double cutoff = control_scalars[kAddhbDoubleCutoff];
  const int base = fill ? offsets[blockIdx.x] : 0;
  if (threadIdx.x == 0) s_running = 0;
  __syncthreads();
  int error = 0;
  for (int chunk = 0; chunk < atom_i - 1; chunk += blockDim.x) {
    const int atom_j = chunk + threadIdx.x + 1;
    int flag = 0;
    if (atom_j < atom_i) {
      flag = hbond_pair_qualifies(atom_i, atom_j, numat, mpack, iorbs, nijbo,
                                  fao, p, cutoff, &error)
                 ? 1
                 : 0;
    }
    int total = 0;
    const int rank = block_exclusive_scan_int(flag, s_warp, &total);
    if (fill && flag) {
      const int pos = base + s_running + rank;
      if (pos < pair_capacity) {
        pair_i[pos] = atom_i;
        pair_j[pos] = atom_j;
      } else {
        error = 1;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) s_running += total;
    __syncthreads();
  }
  if (threadIdx.x == 0 && !fill) counts[blockIdx.x] = s_running;
  if (error) atomicExch(work_ints + kDiaggWorkIntError, 1);
}

__device__ inline bool lmo_head_matches(const int *nnc, const int *nc,
                                        const int *icvec, int ic_dim,
                                        int lmo, int atom_i, int atom_j,
                                        int *error) {
  const int count = nc[lmo];
  const int first = nnc[lmo];
  const int second = first + (count > 1 ? 1 : 0);
  if (count <= 0 || first < 0 || second >= ic_dim) {
    *error = 1;
    return false;
  }
  const int a0 = icvec[first];
  const int a1 = icvec[second];
  return a0 == atom_i || a1 == atom_i || a0 == atom_j || a1 == atom_j;
}

__device__ void hbond_entries_for_pair(
    int fill, int pr, int nocc, int nvir, int icocc_dim, int icvir_dim,
    int fmo_dim, const int *ncf, const int *nncf, const int *icocc,
    const int *nce, const int *nnce, const int *icvir, const int *pair_i,
    const int *pair_j, int *counts, const int *offsets, int *ifmo,
    double *fmo, int *work_ints, int *s_occ, int *s_warp, int *s_m_ptr,
    int *s_running_ptr);

// Grid-stride over qualifying atom pairs (count read from device).  fill=0:
// counts[pair] = m * v.  fill=1: emits (vir, occ) entries in CPU order (vir
// ascending, occ ascending).
__global__ void __launch_bounds__(kDiaggBlockThreads)
mozyme_hbond_entries_kernel(int fill, const int *npairs_ptr, int pair_capacity,
                            int nocc, int nvir, int icocc_dim, int icvir_dim,
                            int fmo_dim, const int *ncf, const int *nncf,
                            const int *icocc, const int *nce, const int *nnce,
                            const int *icvir, const int *pair_i,
                            const int *pair_j, int *counts, const int *offsets,
                            int *ifmo, double *fmo, int *work_ints,
                            const int *resident_control_ints) {
  __shared__ int s_occ[kDiaggMaxLmoAtoms];
  __shared__ int s_warp[8];
  __shared__ int s_m;
  __shared__ int s_running;
  if (resident_control_terminal(resident_control_ints)) return;
  int npairs = *npairs_ptr;
  if (npairs > pair_capacity) npairs = pair_capacity;
  for (int pr = blockIdx.x; pr < npairs; pr += gridDim.x) {
    hbond_entries_for_pair(fill, pr, nocc, nvir, icocc_dim, icvir_dim, fmo_dim,
                           ncf, nncf, icocc, nce, nnce, icvir, pair_i, pair_j,
                           counts, offsets, ifmo, fmo, work_ints, s_occ,
                           s_warp, &s_m, &s_running);
    __syncthreads();
  }
}

__device__ void hbond_entries_for_pair(
    int fill, int pr, int nocc, int nvir, int icocc_dim, int icvir_dim,
    int fmo_dim, const int *ncf, const int *nncf, const int *icocc,
    const int *nce, const int *nnce, const int *icvir, const int *pair_i,
    const int *pair_j, int *counts, const int *offsets, int *ifmo,
    double *fmo, int *work_ints, int *s_occ, int *s_warp, int *s_m_ptr,
    int *s_running_ptr) {
  int &s_m = *s_m_ptr;
  int &s_running = *s_running_ptr;
  const int atom_i = pair_i[pr];
  const int atom_j = pair_j[pr];
  int error = 0;
  if (threadIdx.x == 0) {
    s_m = 0;
    s_running = 0;
  }
  __syncthreads();

  for (int chunk = 0; chunk < nocc; chunk += blockDim.x) {
    const int occ = chunk + threadIdx.x;
    int flag = 0;
    if (occ < nocc) {
      flag = lmo_head_matches(nncf, ncf, icocc, icocc_dim, occ, atom_i, atom_j,
                              &error)
                 ? 1
                 : 0;
    }
    int total = 0;
    const int rank = block_exclusive_scan_int(flag, s_warp, &total);
    if (flag) {
      const int slot = s_m + rank;
      if (slot < kDiaggMaxLmoAtoms) {
        s_occ[slot] = occ + 1;
      } else {
        error = 1;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) s_m += total;
    __syncthreads();
  }
  const int m = s_m;
  if (m == 0) {
    if (!fill && threadIdx.x == 0) counts[pr] = 0;
    if (error) atomicExch(work_ints + kDiaggWorkIntError, 1);
    return;
  }
  const int base = fill ? offsets[pr] : 0;
  for (int chunk = 0; chunk < nvir; chunk += blockDim.x) {
    const int vir = chunk + threadIdx.x;
    int flag = 0;
    if (vir < nvir) {
      flag = lmo_head_matches(nnce, nce, icvir, icvir_dim, vir, atom_i, atom_j,
                              &error)
                 ? 1
                 : 0;
    }
    int total = 0;
    const int rank = block_exclusive_scan_int(flag, s_warp, &total);
    if (fill && flag) {
      const int start = base + (s_running + rank) * m;
      for (int l = 0; l < m; ++l) {
        const int pos = start + l;
        if (pos < fmo_dim) {
          ifmo[2 * pos] = vir + 1;
          ifmo[2 * pos + 1] = s_occ[l];
          fmo[pos] = 0.1;
        } else {
          error = 1;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) s_running += total;
    __syncthreads();
  }
  if (threadIdx.x == 0 && !fill) counts[pr] = s_running * m;
  if (error) atomicExch(work_ints + kDiaggWorkIntError, 1);
}

__global__ void mozyme_hbond_commit_kernel(int fmo_dim, int npairs_capacity,
                                           const int *pair_offsets,
                                           int numat, const int *entry_offsets,
                                           int *control_ints, int *work_ints,
                                           const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;
  if (control_ints[kAddhbIntOk] < 0) {
    control_ints[kAddhbIntNij] = 0;
    return;
  }
  if (control_ints[kAddhbIntDue] == 0) {
    control_ints[kAddhbIntNij] = 0;
    control_ints[kAddhbIntOk] = 1;
    return;
  }
  const int npairs = pair_offsets[numat];
  if (work_ints[kDiaggWorkIntError] != 0 || npairs < 0 ||
      npairs > npairs_capacity) {
    control_ints[kAddhbIntNij] = 0;
    control_ints[kAddhbIntOk] = 0;
    return;
  }
  const int nij = npairs > 0 ? entry_offsets[npairs] : 0;
  if (nij < 0 || nij > fmo_dim) {
    control_ints[kAddhbIntNij] = 0;
    control_ints[kAddhbIntOk] = 0;
    return;
  }
  control_ints[kAddhbIntNij] = nij;
  control_ints[kAddhbIntOk] = 1;
}

// Warp per LMO normalisation check (replaces the one-thread-per-block kernel).
__global__ void __launch_bounds__(kDiaggRotateThreads)
mozyme_check_lmo_warp_kernel(int nvec, int numat, int ic_dim, int c_dim,
                             const int *nnc, const int *nc, const int *icvec,
                             const int *iorbs, const int *ncvec, double *cvec,
                             double *errors, int *bad_index, int *ok_flag,
                             int error_slot,
                             const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int lane = threadIdx.x & 31;
  const int lmo0 = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (lmo0 >= nvec) return;
  const int lmo = lmo0 + 1;

  const int base = nnc[lmo0];
  const int count = nc[lmo0];
  const int coeff_base = ncvec[lmo0];
  int bad = (count <= 0 || base < 0 || base + count > ic_dim) ? 1 : 0;
  int span = 0;
  if (!bad) {
    int running = 0;
    for (int chunk = 0; chunk < count; chunk += 32) {
      const int e = chunk + lane;
      int norb = 0;
      if (e < count) {
        const int atom = icvec[base + e];
        if (atom < 1 || atom > numat) {
          bad = 1;
        } else {
          norb = iorbs[atom - 1];
          if (norb <= 0) bad = 1;
        }
      }
      running += __shfl_sync(0xffffffffu, warp_inclusive_scan_int(norb), 31);
    }
    span = running;
    if (coeff_base < 0 || coeff_base + span > c_dim) bad = 1;
  }
  if (__any_sync(0xffffffffu, bad)) {
    if (lane == 0) {
      *ok_flag = 0;
      atomicMin(&bad_index[error_slot], lmo);
    }
    return;
  }
  double part = 0.0;
  for (int k = lane; k < span; k += 32) {
    const double v = cvec[coeff_base + k];
    part += v * v;
  }
  const double norm = warp_sum_double(part);
  if (!(norm > 0.0) || norm != norm || norm > 1.0e300) {
    if (lane == 0) {
      *ok_flag = 0;
      atomicMin(&bad_index[error_slot], lmo);
    }
    return;
  }
  if (lane == 0) {
    atomicAdd_double(&errors[error_slot], fabs(1.0 - norm));
    if (fabs(norm - 1.0) > 0.1) atomicMin(&bad_index[error_slot], lmo);
  }
  const double scale = 1.0 / sqrt(norm);
  for (int k = lane; k < span; k += 32) cvec[coeff_base + k] *= scale;
}

__global__ void mozyme_tidy_resident_kernel(
    int nmos, int natoms, int norbs, int n01, int n02, double thresh,
    int mode, int *nc, int *ic, double *c, int *nnc, int *ncmo,
    const int *iorbs, int *iused, int *ncnew, int *ncmnew, int *nncnew,
    int *result, const int *resident_control_ints, int *ok_slot) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  result[0] = 0;
  if (resident_control_terminal(resident_control_ints)) return;
  mozyme_tidy_run_device(nmos, natoms, norbs, n01, n02, thresh, 0, 0, mode,
                         nc, ic, c, nnc, ncmo, iorbs, nullptr, iused, ncnew,
                         ncmnew, nncnew, result, ok_slot);
}

__device__ int mozyme_pls_supervisor_device(double ovmax, double escf,
                                            int *pls_ints,
                                            double *pls_scalars) {
  if (!pls_ints || !pls_scalars) return 0;
  int loop = pls_ints[kPlsIntLoop];
  int fault = 0;
  double ovmax_delta = 0.0;
  double energy_delta = 0.0;

  if (loop == -1) {
    for (int i = 0; i < kPlsLoopLimit; ++i) {
      pls_scalars[kPlsDoubleOvmaxHistory + i] = 10.0;
      pls_scalars[kPlsDoubleEscfHistory + i] = 10.0;
    }
    pls_ints[kPlsIntLoop] = 0;
    pls_ints[kPlsIntFault] = 0;
    pls_scalars[kPlsDoubleLastOvmaxDelta] = 0.0;
    pls_scalars[kPlsDoubleLastEnergyDelta] = 0.0;
  } else {
    ++loop;
    if (loop == kPlsLoopLimit + 1) {
      for (int i = 1; i < kPlsLoopLimit; ++i) {
        pls_scalars[kPlsDoubleOvmaxHistory + i - 1] =
            pls_scalars[kPlsDoubleOvmaxHistory + i];
        pls_scalars[kPlsDoubleEscfHistory + i - 1] =
            pls_scalars[kPlsDoubleEscfHistory + i];
      }
      loop = kPlsLoopLimit;
    }

    const int slot = loop - 1;
    ovmax_delta = fabs(ovmax - pls_scalars[kPlsDoubleOvmaxOld]);
    pls_scalars[kPlsDoubleOvmaxOld] = ovmax;
    energy_delta = fabs(escf - pls_scalars[kPlsDoubleEscfOld]);
    pls_scalars[kPlsDoubleEscfOld] = escf;
    if (slot >= 0 && slot < kPlsLoopLimit) {
      pls_scalars[kPlsDoubleOvmaxHistory + slot] = ovmax_delta;
      pls_scalars[kPlsDoubleEscfHistory + slot] = energy_delta;
    }

    int j = 0;
    for (; j < loop; ++j) {
      if (pls_scalars[kPlsDoubleOvmaxHistory + j] > 0.01) break;
    }
    if (j < loop) {
      for (j = 0; j < loop; ++j) {
        if (pls_scalars[kPlsDoubleEscfHistory + j] > 0.1) break;
      }
    }
    fault = (j >= loop && ovmax > 0.1) ? 1 : 0;
    if (fault != 0) {
      for (int i = 0; i < kPlsLoopLimit; ++i) {
        pls_scalars[kPlsDoubleOvmaxHistory + i] = static_cast<double>(i + 1);
        pls_scalars[kPlsDoubleEscfHistory + i] = static_cast<double>(i + 1);
      }
    }
    pls_ints[kPlsIntLoop] = loop;
    pls_ints[kPlsIntFault] = fault;
    pls_scalars[kPlsDoubleLastOvmaxDelta] = ovmax_delta;
    pls_scalars[kPlsDoubleLastEnergyDelta] = energy_delta;
  }
  pls_ints[kPlsIntCalls] += 1;
  return pls_ints[kPlsIntFault];
}

__global__ void mozyme_resident_control_advance_kernel(
    int current_iter, int max_iter, int strict_resident, int use_three_point,
    int lstart, double shift, const int *diagg_ints,
    const double *diagg_scalars, const int *addhb_ints,
    const double *addhb_scalars, const int *isitsc_ints,
    const double *isitsc_scalars, int *pls_ints, double *pls_scalars,
    int *control_ints, double *control_scalars) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(control_ints)) return;

  current_iter = resident_control_int_or(
      control_ints, kResidentControlCurrentIter, current_iter);
  use_three_point = resident_control_int_or(
      control_ints, kResidentControlUseThreePoint, use_three_point);
  lstart = resident_control_int_or(control_ints, kResidentControlLstart,
                                   lstart);
  shift = resident_control_double_or(control_scalars,
                                     kResidentControlShift, shift);
  const int completed_iter =
      current_iter < 2147483647 ? current_iter + 1 : current_iter;
  const int isitsc_ok = isitsc_ints[kIsitscIntOkscf];
  int decision = kResidentDecisionContinue;
  if (isitsc_ok == 1) {
    decision = kResidentDecisionComplete;
  } else if (completed_iter >= max_iter) {
    decision = kResidentDecisionIterationExhausted;
  } else if (strict_resident == 0 &&
             completed_iter > kMozymeScfPlsSupervisorLastIter) {
    decision = kResidentDecisionCpuBoundary;
  } else if (strict_resident != 0 &&
             completed_iter > kMozymeScfPlsSupervisorLastIter &&
             (!pls_ints || pls_ints[kPlsIntRestartDone] == 0)) {
    const double escf = isitsc_scalars[kIsitscEnergyScf];
    const double ovmax = diagg_scalars[kDiaggDoubleTiny];
    if (mozyme_pls_supervisor_device(ovmax, escf, pls_ints, pls_scalars) != 0) {
      decision = kResidentDecisionPlsRestart;
      control_ints[kResidentControlPlsRestartRequired] = 1;
    }
    if (pls_ints && pls_scalars) {
      control_ints[kResidentControlPlsCalls] = pls_ints[kPlsIntCalls];
      control_ints[kResidentControlPlsHistoryCount] = pls_ints[kPlsIntLoop];
      control_scalars[kResidentControlPlsOvmaxDelta] =
          pls_scalars[kPlsDoubleLastOvmaxDelta];
      control_scalars[kResidentControlPlsEnergyDelta] =
          pls_scalars[kPlsDoubleLastEnergyDelta];
    }
  }

  const double energy_diff = isitsc_scalars[kIsitscEnergyDelta];
  int next_use_three_point = use_three_point;
  int next_lstart = lstart;
  double next_shift = shift;
  if (next_use_three_point != 0) {
    if (completed_iter % 3 == 2 && fabs(energy_diff) < 0.1) {
      next_use_three_point = 0;
      next_lstart = completed_iter;
      next_shift = 0.0;
    }
  } else if (energy_diff > 0.0 && next_shift < 11.0 &&
             completed_iter > next_lstart + 2) {
    next_shift += 2.0;
    next_lstart = completed_iter;
  }

  const int next_iter = completed_iter;
  const int next_nhb = addhb_ints[kAddhbIntNextNhb];
  control_ints[kResidentControlDecision] = decision;
  control_ints[kResidentControlDiaggMode] = addhb_ints[kAddhbIntNextIdiagg];
  control_ints[kResidentControlNhb] = next_nhb;
  control_ints[kResidentControlDiaggNf] = diagg_ints[kDiaggIntNf];
  control_ints[kResidentControlNrej0] = addhb_ints[kAddhbIntNextNrej0];
  control_ints[kResidentControlNrej1] = addhb_ints[kAddhbIntNextNrej1];
  control_ints[kResidentControlIemin] = isitsc_ints[kIsitscIntIemin];
  control_ints[kResidentControlIemax] = isitsc_ints[kIsitscIntIemax];
  control_ints[kResidentControlScf1] = isitsc_ints[kIsitscIntScf1];
  control_ints[kResidentControlCurrentIter] = next_iter;
  control_ints[kResidentControlAddhbDue] =
      ((next_iter + 1) % 3 == 0 && next_nhb < 4) ? 1 : 0;
  control_ints[kResidentControlUseThreePoint] = next_use_three_point;
  control_ints[kResidentControlLstart] = next_lstart;

  control_scalars[kResidentControlDiaggFref] =
      diagg_scalars[kDiaggDoubleFref];
  control_scalars[kResidentControlDiaggOldlim] =
      diagg_scalars[kDiaggDoubleOldlim];
  control_scalars[kResidentControlDiaggSafety] =
      diagg_scalars[kDiaggDoubleSafety];
  control_scalars[kResidentControlOvmax] = addhb_scalars[kAddhbDoubleNextTiny];
  control_scalars[kResidentControlPreviousEscf] =
      isitsc_scalars[kIsitscEnergyScf];
  control_scalars[kResidentControlShift] = next_shift;
}

__global__ void mozyme_resident_pls_restart_zero_kernel(
    int pold_count, int p1_count, double *pold, double *p1,
    int *isitsc_ints, const int *control_ints) {
  if (!control_ints ||
      control_ints[kResidentControlDecision] != kResidentDecisionPlsRestart) {
    return;
  }
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pold && idx < pold_count) pold[idx] = 0.0;
  if (p1 && idx < p1_count) p1[idx] = 0.0;
  if (isitsc_ints && idx == 0) {
    isitsc_ints[kIsitscIntIemin] = 0;
    isitsc_ints[kIsitscIntIemax] = 0;
    isitsc_ints[kIsitscIntOkscf] = 0;
    isitsc_ints[kIsitscIntIscf] = 0;
    isitsc_ints[kIsitscIntValid] = 0;
  }
}

__global__ void mozyme_resident_pls_restart_finalize_kernel(
    int *control_ints, double *control_scalars, int *pls_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || !control_ints ||
      !control_scalars ||
      control_ints[kResidentControlDecision] != kResidentDecisionPlsRestart) {
    return;
  }

  if (pls_ints) {
    pls_ints[kPlsIntFault] = 0;
    pls_ints[kPlsIntRestartDone] = 1;
  }

  const int pls_calls = control_ints[kResidentControlPlsCalls];
  const int pls_history = control_ints[kResidentControlPlsHistoryCount];
  const int scf1 = control_ints[kResidentControlScf1];
  const int lstart = control_ints[kResidentControlLstart];
  const double pls_ovmax_delta =
      control_scalars[kResidentControlPlsOvmaxDelta];
  const double pls_energy_delta =
      control_scalars[kResidentControlPlsEnergyDelta];
  const double previous_escf =
      control_scalars[kResidentControlPreviousEscf];
  for (int i = 0; i < kResidentControlIntCount; ++i) control_ints[i] = 0;
  for (int i = 0; i < kResidentControlDoubleCount; ++i) {
    control_scalars[i] = 0.0;
  }

  control_ints[kResidentControlDecision] = kResidentDecisionContinue;
  control_ints[kResidentControlDiaggMode] = 0;
  control_ints[kResidentControlNhb] = 0;
  control_ints[kResidentControlScf1] = scf1;
  control_ints[kResidentControlCurrentIter] = 0;
  control_ints[kResidentControlAddhbDue] = 0;
  control_ints[kResidentControlUseThreePoint] = 1;
  control_ints[kResidentControlLstart] = lstart;
  control_ints[kResidentControlPlsCalls] = pls_calls;
  control_ints[kResidentControlPlsHistoryCount] = pls_history;
  control_ints[kResidentControlPlsRestartResetCalls] = 1;
  control_ints[kResidentControlPlsRestartDone] = 1;
  control_scalars[kResidentControlDiaggFref] = 10.0;
  control_scalars[kResidentControlDiaggOldlim] = 0.0;
  control_scalars[kResidentControlDiaggSafety] = 1.0;
  control_scalars[kResidentControlPreviousEscf] = previous_escf;
  control_scalars[kResidentControlPlsOvmaxDelta] = pls_ovmax_delta;
  control_scalars[kResidentControlPlsEnergyDelta] = pls_energy_delta;
}

__global__ void mozyme_density_values_kernel(
    int task_count, const int *task_j_offset, const int *task_k_offset,
    const int *task_nj, const int *task_nk, const int *task_diag,
    const int *task_value_offset, int value_count, int cocc_dim,
    const double *cocc, double *values) {
  const int task = blockIdx.x;
  if (task >= task_count) return;

  const int nj = task_nj[task];
  const int nk = task_nk[task];
  const int value_offset = task_value_offset[task];
  const int j_offset = task_j_offset[task];
  const int k_offset = task_k_offset[task];
  const bool diag = task_diag[task] != 0;
  const int terms = diag ? (nj * (nj + 1)) / 2 : nj * nk;
  if (nj <= 0 || nk <= 0 || terms <= 0) return;
  if (value_offset < 0 || value_offset + terms > value_count) return;
  if (j_offset < 0 || k_offset < 0) return;
  if (diag) {
    if (j_offset + nj > cocc_dim || k_offset + nj > cocc_dim) return;
  } else {
    if (j_offset + nj > cocc_dim || k_offset + nk > cocc_dim) return;
  }

  for (int term = threadIdx.x; term < terms; term += blockDim.x) {
    if (diag) {
      int packed = 0;
      for (int row = 1; row <= nj; ++row) {
        for (int col = 1; col <= row; ++col) {
          if (packed == term) {
            values[value_offset + term] =
                cocc[j_offset + row - 1] * cocc[k_offset + col - 1];
          }
          ++packed;
        }
      }
    } else {
      const int row = term / nk;
      const int col = term - row * nk;
      values[value_offset + term] = cocc[j_offset + row] * cocc[k_offset + col];
    }
  }
}

__global__ void mozyme_density_init_kernel(int mpack, int mode,
                                           const double *partp, double *p,
                                           const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= mpack) return;

  if (mode == 0) {
    p[idx] = 0.0;
  } else if (mode == -1) {
    p[idx] = -0.5 * partp[idx];
  } else {
    p[idx] = 0.5 * partp[idx];
  }
}

__global__ void mozyme_density_spin_scale_kernel(int mpack, int mode,
                                                 double *p,
                                                 const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= mpack) return;

  double spin_scale = 1.0;
  if (mode == 0 || mode == 1) {
    spin_scale = 2.0;
  } else if (mode == -1) {
    spin_scale = -2.0;
  }
  p[idx] *= spin_scale;
}

__global__ void mozyme_density_resident_kernel(
    int nclose, int numat, int mpack, int icocc_dim, int cocc_dim,
    const int *ncf, const int *nncf, const int *ncocc, const int *icocc,
    const int *iorbs, const int *nijbo, const double *cocc, double *p,
    int *updated_terms, int *ok_out, const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int lmo = blockIdx.x + 1;
  if (lmo > nclose) return;

  const int atom_count = ncf[lmo - 1];
  const int first_atom_slot = nncf[lmo - 1] + 1;
  const int coeff_base = ncocc[lmo - 1];
  if (atom_count <= 0 || first_atom_slot < 1) {
    atomicExch(ok_out, 0);
    return;
  }

  int terms_updated = 0;
  const int pair_count = atom_count * atom_count;
  for (int pair_idx = threadIdx.x; pair_idx < pair_count;
       pair_idx += blockDim.x) {
    const int local_j = pair_idx / atom_count;
    const int local_k = pair_idx - local_j * atom_count;
    const int jj_slot = first_atom_slot + local_j;
    if (jj_slot < 1 || jj_slot > icocc_dim) {
      atomicExch(ok_out, 0);
      return;
    }

    const int atom_j = icocc[jj_slot - 1];
    if (atom_j < 1 || atom_j > numat) {
      atomicExch(ok_out, 0);
      return;
    }

    const int nj = iorbs[atom_j - 1];
    if (nj <= 0) {
      atomicExch(ok_out, 0);
      return;
    }

    int j_coeff_shift = 0;
    for (int idx = 0; idx < local_j; ++idx) {
      const int slot = first_atom_slot + idx;
      if (slot < 1 || slot > icocc_dim) {
        atomicExch(ok_out, 0);
        return;
      }
      const int atom = icocc[slot - 1];
      if (atom < 1 || atom > numat) {
        atomicExch(ok_out, 0);
        return;
      }
      const int atom_orbs = iorbs[atom - 1];
      if (atom_orbs <= 0) {
        atomicExch(ok_out, 0);
        return;
      }
      j_coeff_shift += atom_orbs;
    }

    const int kk_slot = first_atom_slot + local_k;
    if (kk_slot < 1 || kk_slot > icocc_dim) {
      atomicExch(ok_out, 0);
      return;
    }

    const int atom_k = icocc[kk_slot - 1];
    if (atom_k < 1 || atom_k > numat) {
      atomicExch(ok_out, 0);
      return;
    }

    const int nk = iorbs[atom_k - 1];
    const int block_base = nijbo[(atom_j - 1) + (atom_k - 1) * numat];
    if (nk <= 0) {
      atomicExch(ok_out, 0);
      return;
    }

    int k_coeff_base = coeff_base;
    for (int idx = 0; idx < local_k; ++idx) {
      const int slot = first_atom_slot + idx;
      if (slot < 1 || slot > icocc_dim) {
        atomicExch(ok_out, 0);
        return;
      }
      const int atom = icocc[slot - 1];
      if (atom < 1 || atom > numat) {
        atomicExch(ok_out, 0);
        return;
      }
      const int atom_orbs = iorbs[atom - 1];
      if (atom_orbs <= 0) {
        atomicExch(ok_out, 0);
        return;
      }
      k_coeff_base += atom_orbs;
    }

    if (atom_j == atom_k) {
      if (block_base < 0) {
        atomicExch(ok_out, 0);
        return;
      }
      int packed = 0;
      for (int row = 1; row <= nj; ++row) {
        const int cj_idx = coeff_base + j_coeff_shift + row - 1;
        if (cj_idx < 0 || cj_idx >= cocc_dim) {
          atomicExch(ok_out, 0);
          return;
        }
        const double cj = cocc[cj_idx];
        for (int col = 1; col <= row; ++col) {
          const int ck_idx = k_coeff_base + col - 1;
          if (ck_idx < 0 || ck_idx >= cocc_dim) {
            atomicExch(ok_out, 0);
            return;
          }
          const int p_idx = block_base + packed;
          if (p_idx < 0 || p_idx >= mpack) {
            atomicExch(ok_out, 0);
            return;
          }
          atomicAdd_double(&p[p_idx], cj * cocc[ck_idx]);
          ++terms_updated;
          ++packed;
        }
      }
    } else if (atom_j > atom_k && block_base >= 0) {
      int packed = 0;
      for (int row = 1; row <= nj; ++row) {
        const int cj_idx = coeff_base + j_coeff_shift + row - 1;
        if (cj_idx < 0 || cj_idx >= cocc_dim) {
          atomicExch(ok_out, 0);
          return;
        }
        const double cj = cocc[cj_idx];
        for (int col = 1; col <= nk; ++col) {
          const int ck_idx = k_coeff_base + col - 1;
          if (ck_idx < 0 || ck_idx >= cocc_dim) {
            atomicExch(ok_out, 0);
            return;
          }
          const int p_idx = block_base + packed;
          if (p_idx < 0 || p_idx >= mpack) {
            atomicExch(ok_out, 0);
            return;
          }
          atomicAdd_double(&p[p_idx], cj * cocc[ck_idx]);
          ++terms_updated;
          ++packed;
        }
      }
    }
  }

  if (terms_updated > 0) atomicAdd(updated_terms, terms_updated);
}

__global__ void mozyme_density_expected_kernel(
    int nclose, int numat, int mpack, int icocc_dim, int cocc_dim,
    const int *ncf, const int *nncf, const int *ncocc, const int *icocc,
    const int *iorbs, const int *nijbo, int *expected_terms, int *ok_out,
    const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int lmo = blockIdx.x + 1;
  if (lmo > nclose) return;

  const int atom_count = ncf[lmo - 1];
  const int first_atom_slot = nncf[lmo - 1] + 1;
  const int coeff_base = ncocc[lmo - 1];
  if (atom_count <= 0 || first_atom_slot < 1 || coeff_base < 0) {
    atomicExch(ok_out, 0);
    return;
  }

  int lmo_terms = 0;
  const int pair_count = atom_count * atom_count;
  for (int pair_idx = threadIdx.x; pair_idx < pair_count;
       pair_idx += blockDim.x) {
    const int local_j = pair_idx / atom_count;
    const int local_k = pair_idx - local_j * atom_count;
    const int jj_slot = first_atom_slot + local_j;
    const int kk_slot = first_atom_slot + local_k;
    if (jj_slot < 1 || jj_slot > icocc_dim ||
        kk_slot < 1 || kk_slot > icocc_dim) {
      atomicExch(ok_out, 0);
      return;
    }

    const int atom_j = icocc[jj_slot - 1];
    const int atom_k = icocc[kk_slot - 1];
    if (atom_j < 1 || atom_j > numat || atom_k < 1 || atom_k > numat) {
      atomicExch(ok_out, 0);
      return;
    }

    const int nj = iorbs[atom_j - 1];
    const int nk = iorbs[atom_k - 1];
    if (nj <= 0 || nk <= 0) {
      atomicExch(ok_out, 0);
      return;
    }

    int j_coeff_shift = 0;
    for (int idx = 0; idx < local_j; ++idx) {
      const int slot = first_atom_slot + idx;
      if (slot < 1 || slot > icocc_dim) {
        atomicExch(ok_out, 0);
        return;
      }
      const int atom = icocc[slot - 1];
      if (atom < 1 || atom > numat) {
        atomicExch(ok_out, 0);
        return;
      }
      const int atom_orbs = iorbs[atom - 1];
      if (atom_orbs <= 0) {
        atomicExch(ok_out, 0);
        return;
      }
      j_coeff_shift += atom_orbs;
    }

    int k_coeff_shift = 0;
    for (int idx = 0; idx < local_k; ++idx) {
      const int slot = first_atom_slot + idx;
      if (slot < 1 || slot > icocc_dim) {
        atomicExch(ok_out, 0);
        return;
      }
      const int atom = icocc[slot - 1];
      if (atom < 1 || atom > numat) {
        atomicExch(ok_out, 0);
        return;
      }
      const int atom_orbs = iorbs[atom - 1];
      if (atom_orbs <= 0) {
        atomicExch(ok_out, 0);
        return;
      }
      k_coeff_shift += atom_orbs;
    }

    const int j_start = coeff_base + j_coeff_shift;
    const int k_start = coeff_base + k_coeff_shift;
    if (j_start < 0 || j_start + nj > cocc_dim ||
        k_start < 0 || k_start + nk > cocc_dim) {
      atomicExch(ok_out, 0);
      return;
    }

    const int block_base = nijbo[(atom_j - 1) + (atom_k - 1) * numat];
    int terms = 0;
    if (atom_j == atom_k) {
      terms = (nj * (nj + 1)) / 2;
      if (block_base < 0 || block_base + terms > mpack) {
        atomicExch(ok_out, 0);
        return;
      }
    } else if (atom_j > atom_k && block_base >= 0) {
      terms = nj * nk;
      if (block_base + terms > mpack) {
        atomicExch(ok_out, 0);
        return;
      }
    }
    if (terms > 0 && lmo_terms > 2147483647 - terms) {
      atomicExch(ok_out, 0);
      return;
    }
    lmo_terms += terms;
  }

  if (lmo_terms > 0) {
    const int previous = atomicAdd(expected_terms, lmo_terms);
    if (previous < 0 || previous > 2147483647 - lmo_terms) {
      atomicExch(ok_out, 0);
    }
  }
}

__global__ void mozyme_setupk_mark_kernel(int natoms, int nocc, int icocc_dim,
                                          const int *ncf, const int *nncf,
                                          const int *icocc, int *atom_flags) {
  const int lmo = blockIdx.x + 1;
  if (lmo > nocc) return;

  const int atom_count = ncf[lmo - 1];
  const int first_slot = nncf[lmo - 1] + 1;
  for (int local = threadIdx.x; local < atom_count; local += blockDim.x) {
    const int slot = first_slot + local;
    if (slot < 1 || slot > icocc_dim) continue;
    const int atom = icocc[slot - 1];
    if (atom > 0 && atom <= natoms) atomicExch(&atom_flags[atom - 1], 1);
  }
}

__global__ void mozyme_setupk_compress_kernel(int natoms,
                                              const int *atom_flags,
                                              int *kopt) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  int out = 0;
  for (int atom = 1; atom <= natoms; ++atom) {
    if (atom_flags[atom - 1] != 0) {
      kopt[out] = atom;
      ++out;
    }
  }
  if (out < natoms) kopt[out] = 0;
}

__device__ void mozyme_isitsc_apply(
    const double *energy_values, int energy_slot, double energy_scale,
    double energy_offset, double previous_escf, double selcon, double emin,
    double ovmax, int niter, int itrmax, int *iemin, int *iemax, int *scf1,
    double *escf0, double *scalars, int *okscf, int *iscf_out) {
  *okscf = 0;
  const double energy = energy_values[energy_slot];
  const double unclamped_escf = energy * energy_scale + energy_offset;
  const double escf = fmin(999999.0, fmax(-999999.0, unclamped_escf));
  double energy_diff = escf - previous_escf;
  if (fabs(energy_diff) > 9999.0) energy_diff = 0.0;
  scalars[kIsitscEnergyScf] = escf;
  scalars[kIsitscEnergyDelta] = energy_diff;

  const double energy_test = selcon;
  const double fmo_test = selcon * 5.0;
  const bool stable_now =
      ovmax < fmo_test && fabs(energy_diff) < energy_test;

  if ((stable_now && *scf1 != 0) || niter > itrmax) {
    *okscf = 1;
    *iscf_out = (*scf1 != 0) ? 1 : 2;
    return;
  }

  *scf1 = stable_now ? 1 : 0;
  if (emin != 0.0) {
    if (escf < emin) {
      *iemax = 0;
      const int previous_iemin = *iemin;
      const int next_iemin =
          previous_iemin + 1 < 5 ? previous_iemin + 1 : 5;
      if (previous_iemin == 5) {
        for (int i = 1; i < 5; ++i) escf0[i - 1] = escf0[i];
      }
      *iemin = next_iemin;
      escf0[next_iemin - 1] = escf;
      if (next_iemin > 3) {
        for (int i = 1; i < next_iemin; ++i) {
          if (fabs(escf0[i] - escf0[i - 1]) > 0.1 * (emin - escf)) return;
        }
        *okscf = 1;
        *iscf_out = 1;
        return;
      }
    } else {
      *iemin = 0;
      const int previous_iemax = *iemax;
      const int next_iemax =
          previous_iemax + 1 < 5 ? previous_iemax + 1 : 5;
      if (previous_iemax == 5) {
        for (int i = 1; i < 5; ++i) escf0[i - 1] = escf0[i];
      }
      *iemax = next_iemax;
      escf0[next_iemax - 1] = escf;
      if (next_iemax > 3) {
        for (int i = 1; i < next_iemax; ++i) {
          if (fabs(escf0[i] - escf0[i - 1]) > 0.1 * (escf - emin)) return;
        }
        *okscf = 1;
        *iscf_out = 1;
        return;
      }
    }
  }
}

__global__ void mozyme_isitsc_kernel(
    const double *energy_values, int energy_slot, double energy_scale,
    double energy_offset, double previous_escf, double selcon, double emin,
    double ovmax, int niter, int itrmax, int *iemin, int *iemax, int *scf1,
    double *escf0, double *scalars, int *okscf, int *iscf_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  mozyme_isitsc_apply(energy_values, energy_slot, energy_scale, energy_offset,
                      previous_escf, selcon, emin, ovmax, niter, itrmax,
                      iemin, iemax, scf1, escf0, scalars, okscf, iscf_out);
}

__global__ void mozyme_isitsc_resident_kernel(
    const double *energy_values, int energy_slot, double energy_scale,
    double energy_offset, const double *cosmo_scalars, int cosmo_enabled,
    double cosmo_reference_solv_energy, double previous_escf, double selcon,
    double emin, const double *ovmax_values, int ovmax_slot, int niter,
    int itrmax, int *iemin, int *iemax, int *scf1, double *escf0,
    double *scalars, int *okscf, int *iscf_out, int *valid_out,
    const int *resident_control_ints,
    const double *resident_control_scalars) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) {
    // resident terminal stage no-op: the device-side loop decision already
    // completed control flow, so this stage is valid without more work.
    if (valid_out) *valid_out = 1;
    return;
  }
  if (valid_out) *valid_out = 0;
  previous_escf = resident_control_double_or(
      resident_control_scalars, kResidentControlPreviousEscf, previous_escf);
  niter = resident_control_completed_iter_or(resident_control_ints, niter);
  if (cosmo_enabled != 0 && cosmo_scalars) {
    energy_offset +=
        (cosmo_scalars[kCosmoScalarSolvEnergy] - cosmo_reference_solv_energy) *
        energy_scale;
  }
  const double ovmax = ovmax_values ? ovmax_values[ovmax_slot] : 0.0;
  mozyme_isitsc_apply(energy_values, energy_slot, energy_scale, energy_offset,
                      previous_escf, selcon, emin, ovmax, niter, itrmax,
                      iemin, iemax, scf1, escf0, scalars, okscf, iscf_out);
  if (valid_out) *valid_out = 1;
}

__global__ void mozyme_final_reorth_status_kernel(
    const double *energy_values, int energy_slot, double energy_scale,
    double energy_offset, const double *cosmo_scalars, int cosmo_enabled,
    double cosmo_reference_solv_energy, double previous_energy_scf,
    double previous_energy_delta, double *final_scalars, int *ok_out,
    const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (ok_out) *ok_out = 0;
  if (resident_control_terminal(resident_control_ints)) return;
  if (!energy_values || !final_scalars || !ok_out) return;
  if (cosmo_enabled != 0 && cosmo_scalars) {
    energy_offset +=
        (cosmo_scalars[kCosmoScalarSolvEnergy] - cosmo_reference_solv_energy) *
        energy_scale;
  }
  const double final_energy = energy_values[energy_slot];
  const double final_escf = final_energy * energy_scale + energy_offset;
  const double old_previous_escf = previous_energy_scf - previous_energy_delta;
  final_scalars[kFinalReorthStatusEnergyTotal] = final_energy;
  final_scalars[kFinalReorthStatusEnergyScf] = final_escf;
  final_scalars[kFinalReorthStatusEnergyDelta] =
      final_escf - old_previous_escf;
  *ok_out = 1;
}

__global__ void mozyme_chrge_kernel(int numat, int mpack, const int *iorbs,
                                    const int *nijbo, const double *p,
                                    double *qe,
                                    const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int atom = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (atom > numat) return;

  const int base = nijbo[(atom - 1) + (atom - 1) * numat];
  const int norb = iorbs[atom - 1];
  double sum = 0.0;
  if (base >= 0 && norb > 0) {
    int packed = 0;
    for (int orb = 0; orb < norb; ++orb) {
      packed += orb + 1;
      const int idx = base + packed - 1;
      if (idx >= 0 && idx < mpack) sum += p[idx];
    }
  }
  qe[atom - 1] = sum;
}

__global__ void mozyme_fock_init_kernel(int mpack, int mode,
                                        const double *h,
                                        const double *partf,
                                        double *f,
                                        const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= mpack) return;

  if (mode == 0) {
    f[idx] = h[idx];
  } else if (mode == 1) {
    f[idx] = partf[idx] + h[idx];
  } else if (mode == -1) {
    f[idx] = partf[idx] - h[idx];
  }
}

__global__ void mozyme_fock_negate_kernel(int mpack, double *f,
                                          const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= mpack) return;
  f[idx] = -f[idx];
}

__device__ double *atomic_max_double(double *address, double value) {
  unsigned long long int *address_as_ull =
      reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    const double current = __longlong_as_double(
        static_cast<long long>(assumed));
    if (current >= value) break;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long int>(
                        __double_as_longlong(value)));
  } while (assumed != old);
  return address;
}

__device__ double cosmo_coord_at(const double *coord, int rows, int atom,
                                 int component) {
  return coord[(component - 1) + (atom - 1) * rows];
}

__device__ double cosmo_surface_at(const double *cosurf, int rows, int point0,
                                   int component) {
  return cosurf[(component - 1) + point0 * rows];
}

__device__ double cosmo_bvec_value(const double *surface_xyz,
                                   const double *coord, int coord_rows,
                                   const int *nat, const double *dd,
                                   const double *qq, double a0, int atom,
                                   int nao, int term) {
  const double dx0 = surface_xyz[0] - cosmo_coord_at(coord, coord_rows, atom, 1);
  const double dy0 = surface_xyz[1] - cosmo_coord_at(coord, coord_rows, atom, 2);
  const double dz0 = surface_xyz[2] - cosmo_coord_at(coord, coord_rows, atom, 3);
  const double r = 1.0 / sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
  if (term == 1) return r;
  if (nao == 0) return 0.0;

  const int ni = nat[atom - 1];
  if (ni <= 0 || ni > 107) return 0.0;
  const double dx = dx0 * r;
  const double dy = dy0 * r;
  const double dz = dz0 * r;
  const double dip = dd[ni - 1] * a0 * r * r;
  const double quad = (a0 * qq[ni - 1]) * (a0 * qq[ni - 1]) * r * r * r;
  switch (term) {
    case 2: return dx * dip;
    case 3: return r + (3.0 * dx * dx - 1.0) * quad;
    case 4: return dy * dip;
    case 5: return 3.0 * dx * dy * quad;
    case 6: return r + (3.0 * dy * dy - 1.0) * quad;
    case 7: return dz * dip;
    case 8: return 3.0 * dx * dz * quad;
    case 9: return 3.0 * dy * dz * quad;
    case 10: return r + (3.0 * dz * dz - 1.0) * quad;
    case 15:
    case 21:
    case 28:
    case 36:
    case 45:
      return (nao > 3) ? r : 0.0;
    default:
      return 0.0;
  }
}

__global__ void mozyme_cosmo_build_potential_kernel(
    int numat, int nps, int mpack, int coord_rows, int cosurf_rows,
    int phinet_rows, int lm61, double a0, const double *coord, const int *nat,
    const int *nfirst, const int *nlast, const int *nijbo,
    const int *ipiden, const double *gden, const double *dd,
    const double *qq, const double *tore, const double *cosurf,
    const double *density_p, double *phinet,
    const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int s0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (s0 >= nps) return;
  const double surface_xyz[3] = {
      cosmo_surface_at(cosurf, cosurf_rows, s0, 1),
      cosmo_surface_at(cosurf, cosurf_rows, s0, 2),
      cosmo_surface_at(cosurf, cosurf_rows, s0, 3)};
  double nuclear = 0.0;
  double electronic = 0.0;
  int slot_base = 0;
  for (int atom = 1; atom <= numat; ++atom) {
    const int element = nat[atom - 1];
    if (element <= 0 || element > 107) return;
    const double dx = cosmo_coord_at(coord, coord_rows, atom, 1) -
                      surface_xyz[0];
    const double dy = cosmo_coord_at(coord, coord_rows, atom, 2) -
                      surface_xyz[1];
    const double dz = cosmo_coord_at(coord, coord_rows, atom, 3) -
                      surface_xyz[2];
    nuclear += tore[element - 1] * (1.0 / sqrt(dx * dx + dy * dy + dz * dz));

    const int nao = nlast[atom - 1] - nfirst[atom - 1];
    const int nterms = ((nao + 2) * (nao + 1)) / 2;
    const int base = mozyme_nijbo_at(nijbo, numat, atom, atom);
    if (base < 0 || base + nterms > mpack) {
      slot_base += nterms;
      continue;
    }
    int term = 0;
    double atom_electronic = 0.0;
    for (int row = 1; row <= nao + 1; ++row) {
      for (int col = 1; col < row; ++col) {
        ++term;
        const double w = cosmo_bvec_value(surface_xyz, coord, coord_rows,
                                          nat, dd, qq, a0, atom, nao, term);
        const int slot = slot_base + term - 1;
        if (slot >= 0 && slot < lm61) {
          const int pidx = ipiden ? ipiden[slot] - 1 : base + term - 1;
          if (pidx >= 0 && pidx < mpack) {
          atom_electronic += (gden ? gden[slot] : -2.0) * density_p[pidx] * w;
          }
        }
      }
      ++term;
      const double w = cosmo_bvec_value(surface_xyz, coord, coord_rows,
                                        nat, dd, qq, a0, atom, nao, term);
      const int slot = slot_base + term - 1;
      if (slot >= 0 && slot < lm61) {
        const int pidx = ipiden ? ipiden[slot] - 1 : base + term - 1;
        if (pidx >= 0 && pidx < mpack) {
        atom_electronic += (gden ? gden[slot] : -1.0) * density_p[pidx] * w;
        }
      }
    }
    electronic += atom_electronic;
    slot_base += nterms;
  }
  phinet[s0] = 0.0;
  phinet[s0 + phinet_rows] = nuclear + electronic;
  phinet[s0 + 2 * phinet_rows] = nuclear + electronic;
}

__device__ double cosmo_precondition_row(
    int s0, int numat, int nps, const int *iatsp, const int *npoints,
    const int *iblock_pos, const double *m_vec, int m_vec_dim,
    const double *r) {
  const int atom = iatsp[s0];
  if (atom <= 0 || atom > numat) return 0.0;
  const int start = npoints[atom - 1];
  const int end = npoints[atom];
  const int local_i = (s0 + 1) - start + 1;
  const int block_dim = end - start;
  if (start <= 0 || end <= start || local_i < 1 || local_i > block_dim) {
    return 0.0;
  }
  const int base = iblock_pos[atom - 1] - 1;
  double sum = 0.0;
  for (int local_j = 1; local_j <= block_dim; ++local_j) {
    const int packed_row = (local_i > local_j) ? local_i : local_j;
    const int packed_col = (local_i < local_j) ? local_i : local_j;
    const int packed = base + (packed_row * (packed_row - 1)) / 2 +
                       packed_col - 1;
    const int global_j = start + local_j - 2;
    if (global_j >= 0 && global_j < nps && packed >= 0 &&
        packed < m_vec_dim) {
      sum += m_vec[packed] * r[global_j];
    }
  }
  return sum;
}

__global__ void mozyme_cosmo_initial_x_kernel(
    int numat, int nps, int qscnet_rows, int new_surface, double fepsi,
    const int *iatsp, const int *npoints, const int *iblock_pos,
    const double *m_vec, int m_vec_dim, const double *phinet,
    const double *qscnet, double *x, const int *status_ints,
    const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nps) return;
  const bool use_new_surface =
      status_ints ? status_ints[kCosmoStatusNewSurface] != 0
                  : new_surface != 0;
  if (use_new_surface) {
    x[idx] = cosmo_precondition_row(idx, numat, nps, iatsp, npoints, iblock_pos,
                                    m_vec, m_vec_dim, phinet);
  } else {
    x[idx] = qscnet[idx + qscnet_rows] / (-fepsi);
  }
}

__global__ void mozyme_cosmo_matvec_far_kernel(
  int nps, int cosurf_rows, double disex2, const double *cosurf,
  const double *a_diag, const double *x, double *y, int *cg_ints,
  const int *resident_control_ints) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (resident_control_terminal(resident_control_ints)) return;
  if (cg_ints && cg_ints[kCosmoCgActive] == 0) return;
  if (row >= nps) return;
  if (cg_ints && row == 0) atomicAdd(cg_ints + kCosmoCgMatvecCalls, 1);
  double sum = a_diag[row] * x[row];
  const double row_x = cosmo_surface_at(cosurf, cosurf_rows, row, 1);
  const double row_y = cosmo_surface_at(cosurf, cosurf_rows, row, 2);
  const double row_z = cosmo_surface_at(cosurf, cosurf_rows, row, 3);
  for (int col = 0; col < nps; ++col) {
    if (col == row) continue;
    const double dx = row_x - cosmo_surface_at(cosurf, cosurf_rows, col, 1);
    const double dy = row_y - cosmo_surface_at(cosurf, cosurf_rows, col, 2);
    const double dz = row_z - cosmo_surface_at(cosurf, cosurf_rows, col, 3);
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > disex2) sum += (1.0 / sqrt(d2)) * x[col];
  }
  y[row] = sum;
}

__global__ void mozyme_cosmo_matvec_close_kernel(
    int nps, int pair_count, const int *pair_i, const int *pair_j,
    const double *a_part, const double *x, double *y, const int *cg_ints,
    const int *resident_control_ints) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (resident_control_terminal(resident_control_ints)) return;
  if (cg_ints && cg_ints[kCosmoCgActive] == 0) return;
  if (idx >= pair_count) return;
  const int row = pair_i[idx] - 1;
  const int col = pair_j[idx] - 1;
  if (row < 0 || row >= nps || col < 0 || col >= nps) return;
  const double value = a_part[idx];
  atomicAdd_double(y + row, value * x[col]);
  atomicAdd_double(y + col, value * x[row]);
}

__global__ void mozyme_cosmo_residual_kernel(
    int nps, const double *b, const double *ax, double *r,
    const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nps) r[idx] = b[idx] - ax[idx];
}

__global__ void mozyme_cosmo_precondition_kernel(
    int numat, int nps, const int *iatsp, const int *npoints,
    const int *iblock_pos, const double *m_vec, int m_vec_dim,
    const double *r, double *z, const int *cg_ints,
    const int *resident_control_ints) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (resident_control_terminal(resident_control_ints)) return;
  if (cg_ints && cg_ints[kCosmoCgActive] == 0) return;
  if (idx < nps) {
    z[idx] = cosmo_precondition_row(idx, numat, nps, iatsp, npoints, iblock_pos,
                                    m_vec, m_vec_dim, r);
  }
}

__global__ void mozyme_cosmo_zero_scalars_kernel(double *values, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) values[idx] = 0.0;
}

__global__ void mozyme_cosmo_dot_kernel(int nps, const double *a,
                                        const double *b, double *out,
                                        const int *cg_ints,
                                        const int *resident_control_ints) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (resident_control_terminal(resident_control_ints)) return;
  if (cg_ints && cg_ints[kCosmoCgActive] == 0) return;
  if (idx < nps) atomicAdd_double(out, a[idx] * b[idx]);
}

__global__ void mozyme_cosmo_cg_init_control_kernel(
    double fallback_ovmax, double selcon, const double *resident_control_scalars,
    double *status_scalars, int *status_ints, double *scalars, int *ints,
    const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;
  if (!scalars || !ints) return;
  constexpr double kStartTol = 1.0e-2;
  constexpr double kStopTol = 1.0e-6;
  double current_tol = kStartTol;
  if (status_scalars && status_scalars[kCosmoStatusCurrentTol] > 0.0) {
    current_tol = status_scalars[kCosmoStatusCurrentTol];
  }
  if (status_ints && status_ints[kCosmoStatusNewSurface] != 0) {
    current_tol = kStartTol;
  }

  const double ovmax = resident_control_scalars
                           ? resident_control_scalars[kResidentControlOvmax]
                           : fallback_ovmax;
  double c_proc = 1.0;
  if (fabs(ovmax) >= 5.0 * selcon && selcon > 0.0) {
    c_proc = 5.0 * selcon / fabs(ovmax);
  }
  double target_tol = kStopTol;
  if (c_proc < 0.2) {
    target_tol = kStartTol;
  } else if (c_proc < 0.4) {
    target_tol = kStartTol * 0.1;
  } else if (c_proc < 0.6) {
    target_tol = kStartTol * 0.01;
  } else if (c_proc < 0.8) {
    target_tol = kStartTol * 0.001;
  }
  target_tol = fmin(target_tol, current_tol);
  if (status_scalars) {
    status_scalars[kCosmoStatusCurrentTol] = target_tol;
    status_scalars[kCosmoStatusTargetTol] = target_tol;
  }
  if (status_ints) {
    status_ints[kCosmoStatusControlResident] = 1;
    status_ints[kCosmoStatusCgConverged] = 0;
    status_ints[kCosmoStatusCgBreakdown] = 0;
    status_ints[kCosmoStatusHostSyncs] = 0;
  }
  ints[kCosmoCgActive] = 1;
  ints[kCosmoCgCompletedIterations] = 0;
  ints[kCosmoCgBreakdown] = 0;
  ints[kCosmoCgMatvecCalls] = 0;
  scalars[kCosmoCgRhoOld] = 0.0;
  scalars[kCosmoCgAlpha] = 0.0;
  scalars[kCosmoCgBeta] = 0.0;
  scalars[kCosmoCgTargetTol] = target_tol;
  scalars[kCosmoCgLastResidual] = 0.0;
}

__global__ void mozyme_cosmo_cg_prepare_iteration_kernel(
    double *scalars, int *ints, const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;
  if (ints[kCosmoCgActive] == 0) return;
  const int iteration = ints[kCosmoCgCompletedIterations];
  const double rho = scalars[kCosmoCgRho];
  ints[kCosmoCgCompletedIterations] = iteration + 1;
  if (rho == 0.0) {
    ints[kCosmoCgActive] = 0;
    return;
  }
  const double rho_old = scalars[kCosmoCgRhoOld];
  scalars[kCosmoCgBeta] =
      (iteration == 0 || rho_old == 0.0) ? 0.0 : rho / rho_old;
}

__global__ void mozyme_cosmo_cg_direction_kernel(
    int nps, const double *scalars, const int *ints,
    const double *z, double *p, const int *resident_control_ints) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (resident_control_terminal(resident_control_ints)) return;
  if (ints && ints[kCosmoCgActive] == 0) return;
  if (idx >= nps) return;
  const double beta = scalars ? scalars[kCosmoCgBeta] : 0.0;
  const bool first_iteration =
      !ints || ints[kCosmoCgCompletedIterations] <= 1;
  p[idx] = first_iteration ? z[idx] : z[idx] + beta * p[idx];
}

__global__ void mozyme_cosmo_cg_prepare_update_kernel(
    double *scalars, int *ints, const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;
  if (ints[kCosmoCgActive] == 0) return;
  const double pq = scalars[kCosmoCgPq];
  if (pq == 0.0) {
    ints[kCosmoCgActive] = 0;
    ints[kCosmoCgBreakdown] = 1;
    return;
  }
  scalars[kCosmoCgAlpha] = scalars[kCosmoCgRho] / pq;
}

__global__ void mozyme_cosmo_cg_update_kernel(
    int nps, const double *scalars, const int *ints, double *x, double *r,
    const double *p, const double *q, double *norm_out,
    const int *resident_control_ints) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (resident_control_terminal(resident_control_ints)) return;
  if (ints && ints[kCosmoCgActive] == 0) return;
  if (idx >= nps) return;
  const double alpha = scalars ? scalars[kCosmoCgAlpha] : 0.0;
  x[idx] += alpha * p[idx];
  r[idx] -= alpha * q[idx];
  atomic_max_double(norm_out, fabs(r[idx]));
}

__global__ void mozyme_cosmo_cg_finish_iteration_kernel(
    double *scalars, int *ints, const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;
  if (ints[kCosmoCgActive] == 0) return;
  const double residual = scalars[kCosmoCgNorm];
  scalars[kCosmoCgLastResidual] = residual;
  scalars[kCosmoCgRhoOld] = scalars[kCosmoCgRho];
  if (residual < scalars[kCosmoCgTargetTol]) ints[kCosmoCgActive] = 0;
}

__global__ void mozyme_cosmo_cg_finalize_status_kernel(
    const double *cosmo_scalars, const double *cg_scalars, const int *cg_ints,
    double *status_scalars, int *status_ints,
    const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;
  if (!cosmo_scalars || !cg_scalars || !cg_ints || !status_scalars ||
      !status_ints) {
    return;
  }
  const int breakdown = cg_ints[kCosmoCgBreakdown];
  const int converged =
      (cg_ints[kCosmoCgActive] == 0 && breakdown == 0) ? 1 : 0;
  status_ints[kCosmoStatusFockCalls] += 1;
  status_ints[kCosmoStatusMatvecCalls] += cg_ints[kCosmoCgMatvecCalls];
  status_ints[kCosmoStatusCgIterations] +=
      cg_ints[kCosmoCgCompletedIterations];
  status_ints[kCosmoStatusControlResident] = 1;
  status_ints[kCosmoStatusCgConverged] = converged;
  status_ints[kCosmoStatusCgBreakdown] = breakdown;
  status_ints[kCosmoStatusHostSyncs] = 0;
  status_scalars[kCosmoStatusLastResidual] =
      cg_scalars[kCosmoCgLastResidual];
  status_scalars[kCosmoStatusSolvEnergy] =
      cosmo_scalars[kCosmoScalarSolvEnergy];
  status_scalars[kCosmoStatusEdiel] = cosmo_scalars[kCosmoScalarEdiel];
  status_ints[kCosmoStatusNewSurface] = 0;
}

__global__ void mozyme_cosmo_cg_mark_fock_status_kernel(
    int *fock_ints, const int *status_ints,
    const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;
  if (!fock_ints || !status_ints) return;
  fock_ints[kFockIntOk] =
      status_ints[kCosmoStatusCgConverged] == 1 ? 1 : 0;
}

__global__ void mozyme_cosmo_finalize_surface_kernel(
    int numat, int nps, int phinet_rows, int qscnet_rows, double fepsi,
    double fcon, const int *iatsp, double *qscat, double *phinet,
    double *qscnet, const double *x, double *ediel_out,
    const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nps) return;
  const int atom = iatsp[idx] - 1;
  const double q2 = -fepsi * x[idx];
  const double q3 = qscnet[idx] + q2;
  qscnet[idx + qscnet_rows] = q2;
  qscnet[idx + 2 * qscnet_rows] = q3;
  if (atom >= 0 && atom < numat) atomicAdd_double(qscat + atom, q3);
  atomicAdd_double(ediel_out, q3 * phinet[idx + 2 * phinet_rows]);
  if (idx == 0) phinet[idx] = 0.0;
}

__global__ void mozyme_cosmo_fock_correction_kernel(
    int numat, int nps, int mpack, int coord_rows, int cosurf_rows,
    int qscnet_rows, int lm61, double a0, double fcon, const double *coord,
    const int *nat, const int *nfirst, const int *nlast, const int *nijbo,
    const int *ipiden, const double *dd, const double *qq, const double *cosurf,
    const double *qscnet, const double *density_p, const double *gden,
    double *f, double *s1_out, const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int atom = blockIdx.x + 1;
  if (atom < 1 || atom > numat) return;
  const int nao = nlast[atom - 1] - nfirst[atom - 1];
  const int nterms = ((nao + 2) * (nao + 1)) / 2;
  const int base = mozyme_nijbo_at(nijbo, numat, atom, atom);
  if (base < 0 || base + nterms > mpack) return;

  int slot_base = 0;
  for (int prev = 1; prev < atom; ++prev) {
    const int prev_nao = nlast[prev - 1] - nfirst[prev - 1];
    slot_base += ((prev_nao + 2) * (prev_nao + 1)) / 2;
  }

  for (int term = threadIdx.x + 1; term <= nterms; term += blockDim.x) {
    double v = 0.0;
    for (int s0 = 0; s0 < nps; ++s0) {
      const double surface_xyz[3] = {
          cosmo_surface_at(cosurf, cosurf_rows, s0, 1),
          cosmo_surface_at(cosurf, cosurf_rows, s0, 2),
          cosmo_surface_at(cosurf, cosurf_rows, s0, 3)};
      const double w = cosmo_bvec_value(surface_xyz, coord, coord_rows, nat,
                                        dd, qq, a0, atom, nao, term);
      v += w * qscnet[s0 + qscnet_rows];
    }
    v = -v * fcon;
    const int slot = slot_base + term - 1;
    const int fidx = (slot >= 0 && slot < lm61 && ipiden)
                         ? ipiden[slot] - 1
                         : base + term - 1;
    if (fidx >= 0 && fidx < mpack) {
      f[fidx] += v;
      if (slot >= 0 && slot < lm61) {
        atomicAdd_double(s1_out, v * density_p[fidx] * gden[slot]);
      }
    }
  }
}

__global__ void mozyme_cosmo_update_qdenet_kernel(
    int lm61, int mpack, int qdenet_rows, const int *ipiden, const double *gden,
    const double *density_p, double *qdenet,
    const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= lm61) return;
  const int pidx = ipiden[idx] - 1;
  if (pidx < 0 || pidx >= mpack) return;
  const double q2 = gden[idx] * density_p[pidx];
  qdenet[idx + qdenet_rows] = q2;
  qdenet[idx + 2 * qdenet_rows] = qdenet[idx] + q2;
}

__global__ void mozyme_cosmo_finalize_energy_kernel(
  double fcon, const double *ediel_accum, const double *s1_accum,
  double *cosmo_scalars, const int *resident_control_ints) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;
  const double ediel = 0.5 * fcon * ediel_accum[0];
  cosmo_scalars[kCosmoScalarEdiel] = ediel;
  cosmo_scalars[kCosmoScalarSolvEnergy] = 0.5 * s1_accum[0] + ediel;
}

__global__ void mozyme_makvec_init_kernel(
    int natoms, int mpack, const int *iorbs, const int *nfirst,
    const int *nijbo, const double *pdiag, const double *h, double *p,
    double *f) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mpack) {
    p[idx] = 0.0;
    f[idx] = h[idx];
  }
  if (idx >= natoms) return;

  const int atom = idx + 1;
  const int base = mozyme_nijbo_at(nijbo, natoms, atom, atom);
  const int first = nfirst[idx];
  const int norb = iorbs[idx];
  if (base < 0 || first <= 0 || norb <= 0) return;

  int packed = 0;
  for (int orb = 1; orb <= norb; ++orb) {
    packed += orb;
    const int pidx = base + packed - 1;
    if (pidx >= 0 && pidx < mpack) p[pidx] = pdiag[first + orb - 2];
  }
}

__device__ int makvec_upper_index(int row, int col) {
  if (row > col) {
    const int tmp = row;
    row = col;
    col = tmp;
  }
  return (col * (col - 1)) / 2 + row - 1;
}

__device__ int makvec_local_first(int atom) {
  return atom == 1 ? 1 : atom + 3;
}

__device__ int makvec_local_last(int atom) {
  return atom == 1 ? 4 : atom + 3;
}

__device__ bool makvec_jacobi_eigen(const double *packed, int n,
                                    double *eigenvalues,
                                    double *eigenvectors) {
  constexpr int kMaxDim = 20;
  double matrix[kMaxDim * kMaxDim];
  for (int col = 1; col <= n; ++col) {
    for (int row = 1; row <= n; ++row) {
      matrix[(row - 1) + (col - 1) * kMaxDim] =
          packed[makvec_upper_index(row, col)];
      eigenvectors[(row - 1) + (col - 1) * kMaxDim] =
          row == col ? 1.0 : 0.0;
    }
  }

  const int max_rotations = 50 * n * n;
  bool converged = false;
  for (int sweep = 0; sweep < max_rotations; ++sweep) {
    int p = 1;
    int q = 2;
    double max_offdiag = 0.0;
    for (int col = 2; col <= n; ++col) {
      for (int row = 1; row < col; ++row) {
        const double value =
            fabs(matrix[(row - 1) + (col - 1) * kMaxDim]);
        if (value > max_offdiag) {
          max_offdiag = value;
          p = row;
          q = col;
        }
      }
    }
    if (max_offdiag < 1.0e-13) {
      converged = true;
      break;
    }

    const int pp = (p - 1) + (p - 1) * kMaxDim;
    const int qq = (q - 1) + (q - 1) * kMaxDim;
    const int pq = (p - 1) + (q - 1) * kMaxDim;
    const double app = matrix[pp];
    const double aqq = matrix[qq];
    const double apq = matrix[pq];
    if (apq == 0.0) continue;
    const double tau = (aqq - app) / (2.0 * apq);
    const double t = copysign(1.0 / (fabs(tau) + sqrt(1.0 + tau * tau)), tau);
    const double c = 1.0 / sqrt(1.0 + t * t);
    const double s = t * c;

    for (int k = 1; k <= n; ++k) {
      if (k != p && k != q) {
        const int kp = (k - 1) + (p - 1) * kMaxDim;
        const int pk = (p - 1) + (k - 1) * kMaxDim;
        const int kq = (k - 1) + (q - 1) * kMaxDim;
        const int qk = (q - 1) + (k - 1) * kMaxDim;
        const double akp = matrix[kp];
        const double akq = matrix[kq];
        const double next_kp = c * akp - s * akq;
        const double next_kq = s * akp + c * akq;
        matrix[kp] = next_kp;
        matrix[pk] = next_kp;
        matrix[kq] = next_kq;
        matrix[qk] = next_kq;
      }
    }
    matrix[pp] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    matrix[qq] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    matrix[pq] = 0.0;
    matrix[(q - 1) + (p - 1) * kMaxDim] = 0.0;

    for (int row = 1; row <= n; ++row) {
      const int rp = (row - 1) + (p - 1) * kMaxDim;
      const int rq = (row - 1) + (q - 1) * kMaxDim;
      const double vip = eigenvectors[rp];
      const double viq = eigenvectors[rq];
      eigenvectors[rp] = c * vip - s * viq;
      eigenvectors[rq] = s * vip + c * viq;
    }
  }
  if (!converged) {
    double max_offdiag = 0.0;
    for (int col = 2; col <= n; ++col) {
      for (int row = 1; row < col; ++row) {
        const double value =
            fabs(matrix[(row - 1) + (col - 1) * kMaxDim]);
        if (value > max_offdiag) max_offdiag = value;
      }
    }
    converged = max_offdiag < 1.0e-10;
  }

  for (int col = 1; col <= n; ++col) {
    eigenvalues[col - 1] =
        matrix[(col - 1) + (col - 1) * kMaxDim];
  }
  for (int col = 1; col <= n; ++col) {
    int best = col;
    for (int cand = col + 1; cand <= n; ++cand) {
      if (eigenvalues[cand - 1] < eigenvalues[best - 1]) best = cand;
    }
    if (best == col) continue;
    const double ev = eigenvalues[col - 1];
    eigenvalues[col - 1] = eigenvalues[best - 1];
    eigenvalues[best - 1] = ev;
    for (int row = 1; row <= n; ++row) {
      const int a = (row - 1) + (col - 1) * kMaxDim;
      const int b = (row - 1) + (best - 1) * kMaxDim;
      const double value = eigenvectors[a];
      eigenvectors[a] = eigenvectors[b];
      eigenvectors[b] = value;
    }
  }
  return converged;
}

__device__ void makvec_minloc(double *vecs, int nvec, int n) {
  constexpr int kMaxDim = 20;
  double beta;
  double alpha;
  double sum;
  const double rot = 0.999;
  int i = 2;
  if (n != 2) {
    for (i = 2; i <= 4; ++i) {
      sum = vecs[(i - 1) + (2 - 1) * kMaxDim] *
                vecs[(i - 1) + (2 - 1) * kMaxDim] +
            vecs[(i - 1) + (3 - 1) * kMaxDim] *
                vecs[(i - 1) + (3 - 1) * kMaxDim];
      if (sum > 0.1) break;
    }
    if (i <= 4) {
      sum = 1.0 / sqrt(sum);
      alpha = vecs[(i - 1) + (2 - 1) * kMaxDim] * sum;
      beta = vecs[(i - 1) + (3 - 1) * kMaxDim] * sum;
      for (int j = 1; j <= nvec; ++j) {
        const int j2 = (j - 1) + (2 - 1) * kMaxDim;
        const int j3 = (j - 1) + (3 - 1) * kMaxDim;
        const double next = alpha * vecs[j2] + beta * vecs[j3];
        vecs[j3] = -beta * vecs[j2] + alpha * vecs[j3];
        vecs[j2] = next;
      }
    }
    sum = vecs[(i - 1) + (4 - 1) * kMaxDim] *
              vecs[(i - 1) + (4 - 1) * kMaxDim] +
          vecs[(i - 1) + (2 - 1) * kMaxDim] *
              vecs[(i - 1) + (2 - 1) * kMaxDim];
    sum = 1.0 / sqrt(sum);
    alpha = vecs[(i - 1) + (4 - 1) * kMaxDim] * sum;
    beta = vecs[(i - 1) + (2 - 1) * kMaxDim] * sum;
    for (int j = 1; j <= nvec; ++j) {
      const int j4 = (j - 1) + (4 - 1) * kMaxDim;
      const int j2 = (j - 1) + (2 - 1) * kMaxDim;
      const double next = alpha * vecs[j4] + beta * vecs[j2];
      vecs[j4] = -beta * vecs[j4] + alpha * vecs[j2];
      vecs[j2] = next;
    }
  }

  for (i = 2; i <= 4; ++i) {
    sum = vecs[(i - 1) + (4 - 1) * kMaxDim] *
              vecs[(i - 1) + (4 - 1) * kMaxDim] +
          vecs[(i - 1) + (3 - 1) * kMaxDim] *
              vecs[(i - 1) + (3 - 1) * kMaxDim];
    if (sum > 0.1) break;
  }
  if (i > 4) return;
  sum = 1.0 / sqrt(sum);
  alpha = vecs[(i - 1) + (4 - 1) * kMaxDim] * sum;
  beta = vecs[(i - 1) + (3 - 1) * kMaxDim] * sum;
  for (int j = 1; j <= nvec; ++j) {
    const int j4 = (j - 1) + (4 - 1) * kMaxDim;
    const int j3 = (j - 1) + (3 - 1) * kMaxDim;
    const double next = alpha * vecs[j4] + beta * vecs[j3];
    vecs[j4] = -beta * vecs[j4] + alpha * vecs[j3];
    vecs[j3] = next;
  }

  for (int l1 = 1; l1 <= 4; ++l1) {
    for (int l2 = l1 + 1; l2 <= 4; ++l2) {
      alpha = rot;
      beta = sqrt(1.0 - alpha * alpha);
      for (int j = 1; j <= nvec; ++j) {
        const int j1 = (j - 1) + (l1 - 1) * kMaxDim;
        const int j2 = (j - 1) + (l2 - 1) * kMaxDim;
        const double next = alpha * vecs[j1] + beta * vecs[j2];
        vecs[j1] = -beta * vecs[j1] + alpha * vecs[j2];
        vecs[j2] = next;
      }
    }
  }
}

__device__ void makvec_local2(double *c, int mdim, int nmos, int numat) {
  constexpr int kMaxDim = 20;
  double psi1[kMaxDim];
  double psi2[kMaxDim];
  const int n = makvec_local_last(numat);
  const int niter = nmos == 4 ? 19 : 1;
  for (int loop = 1; loop <= niter; ++loop) {
    double total = 0.0;
    for (int i = 1; i <= nmos; ++i) {
      for (int j = 1; j <= nmos; ++j) {
        if (j == i) continue;
        double xijjj = 0.0;
        double xjiii = 0.0;
        double xiiii = 0.0;
        double xjjjj = 0.0;
        double xijij = 0.0;
        double xiijj = 0.0;
        for (int k = 1; k <= n; ++k) {
          psi1[k - 1] = c[(k - 1) + (i - 1) * kMaxDim];
          psi2[k - 1] = c[(k - 1) + (j - 1) * kMaxDim];
        }
        for (int k1 = 1; k1 <= numat; ++k1) {
          double dij = 0.0;
          double dii = 0.0;
          double djj = 0.0;
          for (int k = makvec_local_first(k1); k <= makvec_local_last(k1);
               ++k) {
            dij += psi1[k - 1] * psi2[k - 1];
            dii += psi1[k - 1] * psi1[k - 1];
            djj += psi2[k - 1] * psi2[k - 1];
          }
          xijjj += dij * djj;
          xjiii += dij * dii;
          xiiii += dii * dii;
          xjjjj += djj * djj;
          xijij += dij * dij;
          xiijj += dii * djj;
        }
        double aij = xijij - (xiiii + xjjjj - 2.0 * xiijj) / 4.0;
        double bij = xjiii - xijjj;
        double ca = sqrt(aij * aij + bij * bij);
        double sa = aij + ca;
        if (sa > 1.0e-14) {
          ca = (1.0 + sqrt((1.0 - aij / ca) / 2.0)) / 2.0;
          sa = sqrt(1.0 - ca);
          total += fabs(sa);
          ca = sqrt(ca);
          for (int k = 1; k <= n; ++k) {
            c[(k - 1) + (i - 1) * kMaxDim] =
                ca * psi1[k - 1] + sa * psi2[k - 1];
            c[(k - 1) + (j - 1) * kMaxDim] =
                -sa * psi1[k - 1] + ca * psi2[k - 1];
          }
        }
      }
    }
    if (total < 1.0e-5) break;
  }
}

__device__ void makvec_mlmo(
    int natoms, int norbs, int noccupied, int nvirtual, int ipad2,
    int ipad4, const int *iorbs, int ii, int jj, int &locc, int &lvir,
    int &nf_loc, int &ne, int &nocc, int &nvir, int *nce, int *ncf,
    int *ncocc, int *ncvir, int *icocc, int *icvir, double *cocc,
    double *cvir, int icocc_dim, int icvir_dim, int cocc_dim,
    int cvir_dim, int *ok) {
  const int nes = ne;
  const int nfs = nf_loc;
  const int iocc = locc;
  const int ivir = lvir;

  if (ii != 0) {
    ++nocc;
    if (nocc < 1 || nocc > noccupied) {
      *ok = -201;
      return;
    }
    ncocc[nocc - 1] = locc;
    locc += iorbs[ii - 1];
    ++nf_loc;
    if (nf_loc < 1 || nf_loc > icocc_dim) {
      *ok = -202;
      return;
    }
    icocc[nf_loc - 1] = ii;
    ncf[nocc - 1] = 1;
  }
  if (jj != 0) {
    ++nvir;
    if (nvir < 1 || nvir > nvirtual) {
      *ok = -203;
      return;
    }
    ncvir[nvir - 1] = lvir;
    lvir += iorbs[jj - 1];
    ++ne;
    if (ne < 1 || ne > icvir_dim) {
      *ok = -204;
      return;
    }
    nce[nvir - 1] = 1;
    if (ii != 0) {
      icvir[ne - 1] = ii;
      ncf[nocc - 1] = 2;
      nce[nvir - 1] = 2;
    } else {
      icvir[ne - 1] = jj;
    }
  }
  if (ii != 0 && jj != 0) {
    ++nf_loc;
    ++ne;
    if (nf_loc > icocc_dim || ne > icvir_dim) {
      *ok = -205;
      return;
    }
    icocc[nf_loc - 1] = jj;
    icvir[ne - 1] = jj;
    locc += iorbs[jj - 1];
    lvir += iorbs[ii - 1];
  }

  const int atom_reserve = natoms * 2 < ipad2 ? natoms * 2 : ipad2;
  const int coeff_reserve = norbs * 2 < ipad4 ? norbs * 2 : ipad4;
  if (ii != 0) {
    nf_loc = nfs + atom_reserve;
    if (iocc + coeff_reserve > cocc_dim || nf_loc > icocc_dim) {
      *ok = -206;
      return;
    }
    for (int idx = locc; idx < iocc + coeff_reserve; ++idx) cocc[idx] = 0.0;
    locc = iocc + coeff_reserve;
  }
  if (jj != 0) {
    ne = nes + atom_reserve;
    if (ivir + coeff_reserve > cvir_dim || ne > icvir_dim) {
      *ok = -207;
      return;
    }
    for (int idx = lvir; idx < ivir + coeff_reserve; ++idx) cvir[idx] = 0.0;
    lvir = ivir + coeff_reserve;
  }
}

__device__ double makvec_catom(const double *catom, int morb, int row,
                               int ao_col) {
  return catom[(row - 1) + morb * (ao_col - 1)];
}

__device__ void makvec_set_catom(double *catom, int morb, int row,
                                 int ao_col, double value) {
  catom[(row - 1) + morb * (ao_col - 1)] = value;
}

__device__ bool makvec_build_hybrids(
    int natoms, int norbs, int morb, const int *iorbs, const int *nfirst,
    const int *nlast, const int *nijbo, const int *nbonds,
    const int *ibonds, int ibonds_rows, const double *f, double *catom,
    int *ok) {
  constexpr int kMaxDim = 20;
  double packed[190];
  double eig[kMaxDim];
  double c[kMaxDim * kMaxDim];
  int loop = 0;

  for (int atom = 1; atom <= natoms; ++atom) {
    const int nb = nbonds[atom - 1];
    const int atom_orbs = iorbs[atom - 1];
    if (nb > 15) {
      *ok = -301;
      return false;
    }
    if (atom_orbs == 0) continue;
    if (atom_orbs == 1) {
      ++loop;
      makvec_set_catom(catom, morb, 1, loop, 1.0);
      continue;
    }

    int packed_count = ((nb + 4) * (nb + 5)) / 2;
    for (int idx = 0; idx < 190; ++idx) packed[idx] = 0.0;
    int offset = 10;
    for (int bond = 1; bond <= nb; ++bond) {
      const int other = ibonds[(bond - 1) + ibonds_rows * (atom - 1)];
      const int base = mozyme_nijbo_at(nijbo, natoms, other, atom);
      const int other_orbs = other >= 1 && other <= natoms ? iorbs[other - 1] : 0;
      if (base < 0 || other_orbs <= 0) {
        *ok = -302;
        return false;
      }
      if (other < atom) {
        for (int l = 1; l <= 4; ++l) {
          const int fidx = base - other_orbs + other_orbs * l;
          packed[offset + l - 1] = f[fidx];
        }
      } else {
        for (int l = 1; l <= 4; ++l) {
          packed[offset + l - 1] = f[base + l - 1];
        }
      }
      offset += 4 + bond;
    }
    const int dim = 4 + nb;
    for (int col = 1; col <= 4; ++col) {
      for (int row = 1; row <= 4; ++row) {
        const int idx = (col * (col - 1)) / 2 + row - 1;
        if (idx >= 0 && idx < packed_count) packed[idx] += (idx + 1) * 1.0e-8;
      }
    }
    for (int col = 5; col <= dim; ++col) {
      for (int row = 1; row <= 4; ++row) {
        const int idx = (col * (col - 1)) / 2 + row - 1;
        if (idx >= 0 && idx < packed_count) packed[idx] += (idx + 1) * 2.0e-4;
      }
    }

    if (dim == 1) {
      eig[0] = packed[0];
      c[0] = 1.0;
    } else {
      if (!makvec_jacobi_eigen(packed, dim, eig, c)) {
        *ok = -304;
        return false;
      }
      for (int col = 1; col <= dim; ++col) {
        if (c[(col - 1) * kMaxDim] < 1.0e-14) {
          for (int row = 1; row <= dim; ++row) {
            c[(row - 1) + (col - 1) * kMaxDim] =
                -c[(row - 1) + (col - 1) * kMaxDim];
          }
        }
      }
    }

    const int ill_defined = 8 - dim;
    if (ill_defined > 1) makvec_minloc(c, dim, ill_defined);
    makvec_local2(c, dim, 4, nb + 1);

    for (int col = 1; col <= 4; ++col) {
      double norm = 0.0;
      for (int row = 1; row <= 4; ++row) {
        const double value = c[(row - 1) + (col - 1) * kMaxDim];
        norm += value * value;
      }
      if (!(norm > 0.0)) {
        *ok = -303;
        return false;
      }
      const double scale = 1.0 / sqrt(norm);
      for (int row = 1; row <= 4; ++row) {
        makvec_set_catom(catom, morb, row, loop + col,
                         c[(row - 1) + (col - 1) * kMaxDim] * scale);
      }
    }
    const int sp_orbs = atom_orbs < 4 ? atom_orbs : 4;
    for (int col = 1; col <= sp_orbs; ++col) {
      for (int row = 5; row <= atom_orbs; ++row) {
        makvec_set_catom(catom, morb, row, loop + col, 0.0);
      }
    }
    for (int col = 5; col <= atom_orbs; ++col) {
      for (int row = 1; row <= atom_orbs; ++row) {
        makvec_set_catom(catom, morb, row, loop + col, 0.0);
      }
      makvec_set_catom(catom, morb, col, loop + col, 1.0);
    }
    loop += atom_orbs;
  }
  return loop == norbs;
}

__device__ bool makvec_make_bond_lmo(
    int natoms, int norbs, int morb, int cocc_dim, int cvir_dim,
    const int *iorbs, const int *nfirst, const int *nijbo, const double *f,
    const double *catom, int ii, int jj, unsigned char *used, int &locc,
    int &lvir, double *cocc, double *cvir, int *ok) {
  const int iorbsi = iorbs[ii - 1];
  const int iorbsj = iorbs[jj - 1];
  const int ni = nfirst[ii - 1] - 1;
  const int nj = nfirst[jj - 1] - 1;
  double summin = 0.0;
  double summax = -1.0e5;
  int i1 = 0;
  int j1 = 0;
  int i2 = 0;
  int j2 = 0;
  bool found = false;

  for (int i = 1; i <= iorbsi; ++i) {
    if (used[ni + i - 1]) continue;
    for (int j = 1; j <= iorbsj; ++j) {
      if (used[nj + j - 1]) continue;
      found = true;
      double sum = 0.0;
      int linear = mozyme_nijbo_at(nijbo, natoms, ii, jj);
      if (linear < 0) {
        *ok = -401;
        return false;
      }
      for (int m = 1; m <= iorbsj; ++m) {
        for (int k = 1; k <= iorbsi; ++k) {
          sum += makvec_catom(catom, morb, k, i + ni) * f[linear] *
                 makvec_catom(catom, morb, m, j + nj);
          ++linear;
        }
      }
      if (sum < summin) {
        i1 = i;
        j1 = j;
        summin = sum;
      }
      if (sum > summax) {
        i2 = i;
        j2 = j;
        summax = sum;
      }
    }
  }
  if (!found) return false;

  int used_count = 0;
  for (int i = 1; i <= iorbsi; ++i) {
    if (used[ni + i - 1]) ++used_count;
  }
  for (int i = 1; i <= iorbsj; ++i) {
    if (used[nj + i - 1]) ++used_count;
  }
  if (locc + iorbsi + iorbsj - used_count > cocc_dim ||
      lvir + iorbsi + iorbsj - used_count > cvir_dim) {
    *ok = -402;
    return false;
  }

  double one;
  double e12;
  if (summax > -summin) {
    i1 = i2;
    j1 = j2;
    one = -1.0;
    e12 = -summax;
  } else {
    one = 1.0;
    e12 = summin;
  }

  int linear = mozyme_nijbo_at(nijbo, natoms, ii, ii);
  double e11 = 0.0;
  double e111 = 0.0;
  for (int i = 1; i <= iorbsi; ++i) {
    for (int j = 1; j < i; ++j) {
      e111 += makvec_catom(catom, morb, i, ni + i1) * f[linear] *
              makvec_catom(catom, morb, j, ni + i1);
      ++linear;
    }
    e11 += makvec_catom(catom, morb, i, ni + i1) * f[linear] *
           makvec_catom(catom, morb, i, ni + i1);
    ++linear;
  }
  e11 += e111 * 2.0;

  linear = mozyme_nijbo_at(nijbo, natoms, jj, jj);
  double e22 = 0.0;
  double e221 = 0.0;
  for (int i = 1; i <= iorbsj; ++i) {
    for (int j = 1; j < i; ++j) {
      e221 += makvec_catom(catom, morb, i, nj + j1) * f[linear] *
              makvec_catom(catom, morb, j, nj + j1);
      ++linear;
    }
    e22 += makvec_catom(catom, morb, i, nj + j1) * f[linear] *
           makvec_catom(catom, morb, i, nj + j1);
    ++linear;
  }
  e22 += e221 * 2.0;

  const double d = e11 - e22;
  const double e = copysign(sqrt(4.0 * e12 * e12 + d * d), d);
  const double alpha = fmin(0.866, fmax(sqrt(0.5 * (1.0 + d / e)), 0.5));
  const double beta = -copysign(sqrt(1.0 - alpha * alpha), e12);
  for (int i = 1; i <= iorbsi; ++i) {
    cocc[locc + i - 1] = makvec_catom(catom, morb, i, ni + i1) * alpha;
    cvir[lvir + i - 1] = makvec_catom(catom, morb, i, ni + i1) * beta;
  }
  int max_j = iorbsj;
  if (max_j > cocc_dim - locc - iorbsi) max_j = cocc_dim - locc - iorbsi;
  if (max_j > cvir_dim - lvir - iorbsi) max_j = cvir_dim - lvir - iorbsi;
  for (int i = 1; i <= max_j; ++i) {
    cocc[locc + iorbsi + i - 1] =
        makvec_catom(catom, morb, i, nj + j1) * beta * one;
    cvir[lvir + iorbsi + i - 1] =
        -makvec_catom(catom, morb, i, nj + j1) * alpha * one;
  }
  used[ni + i1 - 1] = 1;
  used[nj + j1 - 1] = 1;
  return true;
}

__global__ void mozyme_makvec_lmo_kernel(
    int natoms, int norbs, int morb, int lewis_tot, int noccupied,
    int nvirtual, int ipad2, int ipad4, int icocc_dim, int cocc_dim,
    int icvir_dim, int cvir_dim, int ibonds_rows, const int *iorbs,
    const int *nfirst, const int *nlast, const int *nijbo,
    const int *nbonds, const int *ibonds, const int *lewis_elem,
    const double *f, double *catom, int *ncf, int *nncf, int *ncocc,
    int *icocc, double *cocc, int *nce, int *nnce, int *ncvir,
    int *icvir, double *cvir, unsigned char *used, int *ok) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *ok = 0;

  if (!makvec_build_hybrids(natoms, norbs, morb, iorbs, nfirst, nlast,
                            nijbo, nbonds, ibonds, ibonds_rows, f, catom,
                            ok)) {
    if (*ok == 0) *ok = -2;
    return;
  }

  int nocc = 0;
  int nvir = 0;
  int nf_loc = 0;
  int ne = 0;
  int locc = 0;
  int lvir = 0;

  for (int item = 1; item <= lewis_tot; ++item) {
    const int ii = lewis_elem[2 * (item - 1)];
    const int jj = lewis_elem[2 * (item - 1) + 1];
    if (ii > 0 && jj > 0) {
      const int ni = nfirst[ii - 1];
      const int nj = nfirst[jj - 1];
      (void)ni;
      (void)nj;
      if (!makvec_make_bond_lmo(natoms, norbs, morb, cocc_dim, cvir_dim,
                                iorbs, nfirst, nijbo, f, catom, ii, jj,
                                used, locc, lvir, cocc, cvir, ok)) {
        if (*ok == 0) *ok = -3;
        return;
      }
      nncf[nocc] = nf_loc;
      nnce[nvir] = ne;
      makvec_mlmo(natoms, norbs, noccupied, nvirtual, ipad2, ipad4, iorbs,
                  ii, jj, locc, lvir, nf_loc, ne, nocc, nvir, nce, ncf,
                  ncocc, ncvir, icocc, icvir, cocc, cvir, icocc_dim,
                  icvir_dim, cocc_dim, cvir_dim, ok);
      if (*ok != 0) return;
    } else if (ii > 0) {
      bool found = false;
      for (int k = nlast[ii - 1]; k >= nfirst[ii - 1]; --k) {
        if (used[k - 1]) continue;
        used[k - 1] = 1;
        int coeff = locc;
        for (int orb = 1; orb <= iorbs[ii - 1]; ++orb) {
          if (coeff >= cocc_dim) {
            *ok = -501;
            return;
          }
          cocc[coeff++] = makvec_catom(catom, morb, orb, k);
        }
        nncf[nocc] = nf_loc;
        makvec_mlmo(natoms, norbs, noccupied, nvirtual, ipad2, ipad4, iorbs,
                    ii, 0, locc, lvir, nf_loc, ne, nocc, nvir, nce, ncf,
                    ncocc, ncvir, icocc, icvir, cocc, cvir, icocc_dim,
                    icvir_dim, cocc_dim, cvir_dim, ok);
        if (*ok != 0) return;
        found = true;
        break;
      }
      if (!found) {
        *ok = -502;
        return;
      }
    } else if (jj > 0) {
      bool found = false;
      for (int k = nlast[jj - 1]; k >= nfirst[jj - 1]; --k) {
        if (used[k - 1]) continue;
        used[k - 1] = 1;
        int coeff = lvir;
        for (int orb = 1; orb <= iorbs[jj - 1]; ++orb) {
          if (coeff >= cvir_dim) {
            *ok = -601;
            return;
          }
          cvir[coeff++] = makvec_catom(catom, morb, orb, k);
        }
        nnce[nvir] = ne;
        makvec_mlmo(natoms, norbs, noccupied, nvirtual, ipad2, ipad4, iorbs,
                    0, jj, locc, lvir, nf_loc, ne, nocc, nvir, nce, ncf,
                    ncocc, ncvir, icocc, icvir, cocc, cvir, icocc_dim,
                    icvir_dim, cocc_dim, cvir_dim, ok);
        if (*ok != 0) return;
        found = true;
        break;
      }
      if (!found) {
        *ok = -602;
        return;
      }
    }
  }

  if (nocc != noccupied || nvir != nvirtual) {
    *ok = -701;
    return;
  }
}

__global__ void mozyme_cnvgz_diag_kernel(int norbs, int mpack,
                                         const int *idiag,
                                         const double *pnew,
                                         const double *pold,
                                         double *diag_new,
                                         double *diag_old,
                                         const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= norbs) return;

  const int packed = idiag[idx] - 1;
  if (packed < 0 || packed >= mpack) {
    diag_new[idx] = 0.0;
    diag_old[idx] = 0.0;
    return;
  }
  diag_new[idx] = pnew[packed];
  diag_old[idx] = pold[packed];
}

__global__ void mozyme_cnvgz_diff_kernel(int mpack, const double *pnew,
                                         const double *pold,
                                         double *block_max,
                                         double *block_sumsq,
                                         const int *resident_control_ints) {
  if (resident_control_terminal(resident_control_ints)) return;
  extern __shared__ double shared[];
  double *max_part = shared;
  double *sum_part = shared + blockDim.x;
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  double local_max = 0.0;
  double local_sum = 0.0;
  if (idx < mpack) {
    const double diff = fabs(pnew[idx] - pold[idx]);
    local_max = diff;
    local_sum = diff * diff;
  }
  max_part[tid] = local_max;
  sum_part[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      max_part[tid] = fmax(max_part[tid], max_part[tid + stride]);
      sum_part[tid] += sum_part[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_max[blockIdx.x] = max_part[0];
    block_sumsq[blockIdx.x] = sum_part[0];
  }
}

__global__ void mozyme_cnvgz_diff_reduce_kernel(int count,
                                                const double *block_max,
                                                const double *block_sumsq,
                                                double *totals,
                                                const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;

  double pmax = 0.0;
  double sumsq = 0.0;
  for (int idx = 0; idx < count; ++idx) {
    pmax = fmax(pmax, block_max[idx]);
    sumsq += block_sumsq[idx];
  }
  totals[0] = pmax;
  totals[1] = sumsq;
}

__global__ void mozyme_cnvgz_factor_kernel(int norbs, const double *diag_old2,
                                           const double *diag_old1,
                                           const double *diag_new,
                                           double *block_faca,
                                           double *block_facb,
                                           const int *resident_control_ints,
                                           int fallback_use_three_point,
                                           int fallback_niter) {
  extern __shared__ double shared[];
  double *faca_part = shared;
  double *facb_part = shared + blockDim.x;
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  if (resident_control_terminal(resident_control_ints)) return;
  const bool use_three_point = resident_control_uses_three_point_or(
      resident_control_ints, fallback_use_three_point != 0);
  const int niter =
      resident_control_completed_iter_or(resident_control_ints, fallback_niter);
  const bool compute_factor = use_three_point && niter % 3 == 0;

  double faca = 0.0;
  double facb = 0.0;
  if (compute_factor && idx < norbs) {
    const double d1 = fabs(diag_new[idx] - diag_old2[idx]);
    const double d2 = diag_new[idx] - 2.0 * diag_old2[idx] + diag_old1[idx];
    faca = d1 * d1;
    facb = d2 * d2;
  }
  faca_part[tid] = faca;
  facb_part[tid] = facb;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      faca_part[tid] += faca_part[tid + stride];
      facb_part[tid] += facb_part[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_faca[blockIdx.x] = faca_part[0];
    block_facb[blockIdx.x] = facb_part[0];
  }
}

__global__ void mozyme_cnvgz_factor_reduce_kernel(int count,
                                                  const double *block_faca,
                                                  const double *block_facb,
                                                  double *control,
                                                  const int *resident_control_ints) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) return;

  double faca = 0.0;
  double facb = 0.0;
  for (int idx = 0; idx < count; ++idx) {
    faca += block_faca[idx];
    facb += block_facb[idx];
  }
  control[kCnvgzFaca] = faca;
  control[kCnvgzFacb] = facb;
}

__global__ void mozyme_cnvgz_finalize_kernel(
    int mpack, bool compute_factor, double *control, int *status_ints,
    const int *resident_control_ints, int fallback_use_three_point,
    int fallback_niter) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (resident_control_terminal(resident_control_ints)) {
    // resident terminal stage no-op: the device-side loop decision already
    // completed control flow, so this stage is valid without more work.
    if (status_ints) {
      status_ints[kCnvgzIntOk] = 1;
      status_ints[kCnvgzIntNoopCalls] += 1;
    }
    return;
  }
  if (status_ints) status_ints[kCnvgzIntOk] = 0;
  bool use_three_point = fallback_use_three_point != 0;
  if (resident_control_ints) {
    use_three_point = resident_control_uses_three_point_or(
        resident_control_ints, use_three_point);
    const int niter =
        resident_control_completed_iter_or(resident_control_ints,
                                           fallback_niter);
    compute_factor = use_three_point && niter % 3 == 0;
  }

  const double sumsq = control[kCnvgzSumsq];
  control[kCnvgzDensityRms] =
      (mpack > 0 && sumsq >= 0.0) ? sqrt(sumsq / static_cast<double>(mpack))
                                  : 0.0;
  double factor = 0.0;
  if (compute_factor) {
    const double faca = control[kCnvgzFaca];
    const double facb = control[kCnvgzFacb];
    if (facb > 0.0 && faca < (100.0 * facb)) factor = sqrt(faca / facb);
  }
  control[kCnvgzFactor] = factor;
  const bool valid = mpack > 0 && sumsq >= 0.0;
  if (status_ints) {
    status_ints[kCnvgzIntOk] = valid ? 1 : 0;
    if (valid) {
      if (use_three_point) {
        status_ints[kCnvgzIntActiveCalls] += 1;
      } else {
        status_ints[kCnvgzIntNoopCalls] += 1;
      }
    }
  }
}

__global__ void mozyme_cnvgz_candidate_kernel(int mpack,
                                              const double *control,
                                              const double *pnew,
                                              const double *pold,
                                              double *candidate,
                                              const int *resident_control_ints,
                                              int fallback_use_three_point) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= mpack) return;

  const bool use_three_point = resident_control_uses_three_point_or(
      resident_control_ints, fallback_use_three_point != 0);
  const double factor = control[kCnvgzFactor];
  candidate[idx] =
      use_three_point ? pnew[idx] + factor * (pnew[idx] - pold[idx])
                      : pnew[idx];
}

__global__ void mozyme_cnvgz_damp_kernel(int norbs, int mpack,
                                         const double *control,
                                         const int *idiag,
                                         const double *diag_new,
                                         const double *diag_old,
                                         double *candidate,
                                         const int *resident_control_ints,
                                         int fallback_use_three_point,
                                         int fallback_niter) {
  if (resident_control_terminal(resident_control_ints)) return;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const bool use_three_point = resident_control_uses_three_point_or(
      resident_control_ints, fallback_use_three_point != 0);
  const int niter =
      resident_control_completed_iter_or(resident_control_ints, fallback_niter);
  const double pmax = control[kCnvgzPmax];
  if (idx >= norbs || !use_three_point || niter <= 3 || pmax <= 0.05) {
    return;
  }

  const double delta = diag_new[idx] - diag_old[idx];
  if (fabs(delta) <= 0.05) return;

  double value = diag_old[idx] + (delta < 0.0 ? -0.05 : 0.05);
  value = fmin(2.0, fmax(value, 0.0));
  const int packed = idiag[idx] - 1;
  if (packed >= 0 && packed < mpack) candidate[packed] = value;
}

__global__ void mozyme_cnvgz_commit_matrix_kernel(
    int mpack, const int *resident_control_ints, const double *candidate,
    double *p, double *pold) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (resident_control_terminal(resident_control_ints)) return;
  if (idx >= mpack ||
      !resident_control_uses_three_point_or(resident_control_ints, false)) {
    return;
  }
  const double value = candidate[idx];
  p[idx] = value;
  pold[idx] = value;
}

__global__ void mozyme_cnvgz_commit_diag_kernel(
    int norbs, const int *resident_control_ints, const double *diag_old,
    const double *diag_new, double *p1, double *p2, double *p3) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (resident_control_terminal(resident_control_ints)) return;
  if (idx >= norbs ||
      !resident_control_uses_three_point_or(resident_control_ints, false)) {
    return;
  }
  p1[idx] = diag_old[idx];
  p2[idx] = diag_old[idx];
  p3[idx] = diag_new[idx];
}

int ceil_div(int value, int divisor) { return (value + divisor - 1) / divisor; }

bool upload_registered_state(MozymeScfContext &ctx) {
  ctx.device.uploaded = false;
  if (!ctx.state_registered) return false;
  if (!valid_state(ctx.config, ctx.state)) return false;

  const int numat = ctx.config.natoms;
  const int norbs = ctx.config.norbs;
  const int mpack = ctx.config.mpack;
  const int nocc = ctx.config.noccupied;
  const int nvir = ctx.config.nvirtual;
  const int total_occ = ctx.config.total_occupied;
  const int total_vir = ctx.config.total_virtual;
  if (numat <= 0 || norbs <= 0 || mpack <= 0 || nocc < 0 || nvir < 0 ||
      total_occ < nocc || total_vir < nvir) {
    return false;
  }

  const std::size_t numat_count = static_cast<std::size_t>(numat);
  const std::size_t norbs_count = static_cast<std::size_t>(norbs);
  const std::size_t mpack_count = static_cast<std::size_t>(mpack);
  const std::size_t nocc_slots =
      static_cast<std::size_t>(ctx.state.nocc_slots);
  const std::size_t nvir_slots =
      static_cast<std::size_t>(ctx.state.nvir_slots);
  const std::size_t fmo_count = static_cast<std::size_t>(ctx.state.fmo_dim);
  const std::size_t nijbo_count = numat_count * numat_count;
  auto &dev = ctx.device;
  if (ctx.state.nocc_slots < total_occ + 1 ||
      ctx.state.nvir_slots < total_vir + 1 ||
      ctx.state.partp_dim < 1 || ctx.state.partf_dim < 1) {
    return false;
  }
  if ((ctx.config.fock_mode != 0 || ctx.config.density_indi != 0) &&
      (ctx.state.partp_dim < mpack || ctx.state.partf_dim < mpack)) {
    return false;
  }
  const std::size_t partp_upload_count =
      std::min(static_cast<std::size_t>(ctx.state.partp_dim), mpack_count);
  const std::size_t partf_upload_count =
      std::min(static_cast<std::size_t>(ctx.state.partf_dim), mpack_count);

  if (!dev.p.upload(static_cast<const double *>(ctx.state.p), mpack_count)) {
    return false;
  }
  if (!dev.f.upload(static_cast<const double *>(ctx.state.f), mpack_count)) {
    return false;
  }
  if (!dev.h.upload(static_cast<const double *>(ctx.state.h), mpack_count)) {
    return false;
  }
  if (!dev.eimp_p.resize(mpack_count)) return false;
  if (!dev.partp.resize(mpack_count)) return false;
  if (!cuda_context_ok(cudaMemset(dev.partp.ptr, 0,
                                  mpack_count * sizeof(double)),
                       "resident upload partp memset")) {
    return false;
  }
  if (!cuda_context_ok(cudaMemcpy(dev.partp.ptr, ctx.state.partp,
                                  partp_upload_count * sizeof(double),
                                  cudaMemcpyHostToDevice),
                       "resident upload partp copy")) {
    return false;
  }
  if (!dev.partf.resize(mpack_count)) return false;
  if (!cuda_context_ok(cudaMemset(dev.partf.ptr, 0,
                                  mpack_count * sizeof(double)),
                       "resident upload partf memset")) {
    return false;
  }
  if (!cuda_context_ok(cudaMemcpy(dev.partf.ptr, ctx.state.partf,
                                  partf_upload_count * sizeof(double),
                                  cudaMemcpyHostToDevice),
                       "resident upload partf copy")) {
    return false;
  }
  if (!dev.pold.upload(static_cast<const double *>(ctx.state.pold),
                       mpack_count)) {
    return false;
  }
  if (!dev.p1.upload(static_cast<const double *>(ctx.state.p1), norbs_count)) {
    return false;
  }
  if (!dev.p2.upload(static_cast<const double *>(ctx.state.p2), norbs_count)) {
    return false;
  }
  if (!dev.p3.upload(static_cast<const double *>(ctx.state.p3), norbs_count)) {
    return false;
  }
  if (!dev.idiag.upload(static_cast<const int *>(ctx.state.idiag),
                        norbs_count)) {
    return false;
  }
  if (!dev.iorbs.upload(static_cast<const int *>(ctx.state.iorbs),
                        numat_count)) {
    return false;
  }
  if (!dev.nfirst.upload(static_cast<const int *>(ctx.state.nfirst),
                         numat_count)) {
    return false;
  }
  if (!dev.nlast.upload(static_cast<const int *>(ctx.state.nlast),
                        numat_count)) {
    return false;
  }
  if (ctx.state.cocc_dim <= 0 || ctx.state.icocc_dim <= 0 ||
      ctx.state.cvir_dim <= 0 || ctx.state.icvir_dim <= 0 ||
      ctx.state.fmo_dim <= 0) {
    return false;
  }
  if (!dev.ncf.upload(static_cast<const int *>(ctx.state.ncf), nocc_slots)) {
    return false;
  }
  if (!dev.nncf.upload(static_cast<const int *>(ctx.state.nncf), nocc_slots)) {
    return false;
  }
  if (!dev.ncocc.upload(static_cast<const int *>(ctx.state.ncocc),
                        nocc_slots)) {
    return false;
  }
  if (!dev.icocc.upload(static_cast<const int *>(ctx.state.icocc),
                        static_cast<std::size_t>(ctx.state.icocc_dim))) {
    return false;
  }
  if (!dev.cocc.upload(static_cast<const double *>(ctx.state.cocc),
                       static_cast<std::size_t>(ctx.state.cocc_dim))) {
    return false;
  }
  if (!dev.nce.upload(static_cast<const int *>(ctx.state.nce), nvir_slots)) {
    return false;
  }
  if (!dev.nnce.upload(static_cast<const int *>(ctx.state.nnce), nvir_slots)) {
    return false;
  }
  if (!dev.ncvir.upload(static_cast<const int *>(ctx.state.ncvir),
                        nvir_slots)) {
    return false;
  }
  if (!dev.icvir.upload(static_cast<const int *>(ctx.state.icvir),
                        static_cast<std::size_t>(ctx.state.icvir_dim))) {
    return false;
  }
  if (!dev.cvir.upload(static_cast<const double *>(ctx.state.cvir),
                       static_cast<std::size_t>(ctx.state.cvir_dim))) {
    return false;
  }
  if (!dev.fmo.upload(static_cast<const double *>(ctx.state.fmo), fmo_count)) {
    return false;
  }
  if (!dev.ifmo.upload(static_cast<const int *>(ctx.state.ifmo),
                       2 * fmo_count)) {
    return false;
  }
  if (!dev.eigs.upload(static_cast<const double *>(ctx.state.eigs),
                       norbs_count)) {
    return false;
  }
  if (!dev.nfmo.upload(static_cast<const int *>(ctx.state.nfmo),
                       norbs_count)) {
    return false;
  }
  if (ctx.state.use_nijbo) {
    if (!dev.nijbo.upload(static_cast<const int *>(ctx.state.nijbo),
                          nijbo_count)) {
      return false;
    }
  } else {
    dev.nijbo.reset();
  }
  if (ctx.state.cosmo_enabled) {
    const int cosmo_nps = ctx.state.cosmo_nps;
    const int cosmo_lm61 = ctx.state.cosmo_lm61;
    if (cosmo_nps <= 0 || cosmo_lm61 <= 0 ||
        ctx.state.coord_rows < 3 || ctx.state.coord_cols < numat ||
        ctx.state.nat_dim < numat || ctx.state.param_dim < 107 ||
        ctx.state.cosmo_cosurf_rows < 4 ||
        ctx.state.cosmo_cosurf_cols < cosmo_nps ||
        ctx.state.cosmo_phinet_rows < cosmo_nps ||
        ctx.state.cosmo_phinet_cols < 3 ||
        ctx.state.cosmo_qscnet_rows < cosmo_nps ||
        ctx.state.cosmo_qscnet_cols < 3 ||
        ctx.state.cosmo_qdenet_rows < cosmo_lm61 ||
        ctx.state.cosmo_qdenet_cols < 3 ||
        ctx.state.cosmo_qscat_dim < numat ||
        ctx.state.cosmo_srad_dim < numat ||
        ctx.state.cosmo_npoints_dim < numat + 1 ||
        ctx.state.cosmo_a_diag_dim < cosmo_nps ||
        ctx.state.cosmo_m_vec_dim <= 0 ||
        ctx.state.cosmo_iblock_pos_dim < numat ||
        !ctx.state.coord || !ctx.state.nat ||
        !ctx.state.param_dd || !ctx.state.param_qq ||
        !ctx.state.param_tore || !ctx.state.cosmo_iatsp ||
        !ctx.state.cosmo_ipiden || !ctx.state.cosmo_gden ||
        !ctx.state.cosmo_qscat || !ctx.state.cosmo_srad ||
        !ctx.state.cosmo_cosurf || !ctx.state.cosmo_phinet ||
        !ctx.state.cosmo_qscnet || !ctx.state.cosmo_qdenet ||
        !ctx.state.cosmo_npoints || !ctx.state.cosmo_a_diag ||
        !ctx.state.cosmo_m_vec || !ctx.state.cosmo_iblock_pos ||
        !ctx.state.cosmo_solv_energy_ptr || !ctx.state.cosmo_ediel_ptr) {
      return false;
    }
    const std::size_t cosmo_nps_count = static_cast<std::size_t>(cosmo_nps);
    const std::size_t cosmo_lm61_count = static_cast<std::size_t>(cosmo_lm61);
    if (!dev.coord.upload(static_cast<const double *>(ctx.state.coord),
                          static_cast<std::size_t>(ctx.state.coord_rows) *
                              static_cast<std::size_t>(ctx.state.coord_cols))) {
      return false;
    }
    if (!dev.nat.upload(static_cast<const int *>(ctx.state.nat),
                        numat_count)) return false;
    if (!dev.param_dd.upload(static_cast<const double *>(ctx.state.param_dd),
                             static_cast<std::size_t>(ctx.state.param_dim))) {
      return false;
    }
    if (!dev.param_qq.upload(static_cast<const double *>(ctx.state.param_qq),
                             static_cast<std::size_t>(ctx.state.param_dim))) {
      return false;
    }
    if (!dev.param_tore.upload(
            static_cast<const double *>(ctx.state.param_tore),
            static_cast<std::size_t>(ctx.state.param_dim))) {
      return false;
    }
    if (!dev.cosmo_iatsp.upload(static_cast<const int *>(ctx.state.cosmo_iatsp),
                                cosmo_nps_count)) return false;
    if (!dev.cosmo_ipiden.upload(
            static_cast<const int *>(ctx.state.cosmo_ipiden),
            cosmo_lm61_count)) return false;
    if (!dev.cosmo_gden.upload(static_cast<const double *>(ctx.state.cosmo_gden),
                               cosmo_lm61_count)) return false;
    if (!dev.cosmo_qscat.upload(
            static_cast<const double *>(ctx.state.cosmo_qscat),
            numat_count)) return false;
    if (!dev.cosmo_srad.upload(
            static_cast<const double *>(ctx.state.cosmo_srad),
            numat_count)) return false;
    if (!dev.cosmo_cosurf.upload(
            static_cast<const double *>(ctx.state.cosmo_cosurf),
            static_cast<std::size_t>(ctx.state.cosmo_cosurf_rows) *
                static_cast<std::size_t>(ctx.state.cosmo_cosurf_cols))) {
      return false;
    }
    if (!dev.cosmo_phinet.upload(
            static_cast<const double *>(ctx.state.cosmo_phinet),
            static_cast<std::size_t>(ctx.state.cosmo_phinet_rows) *
                static_cast<std::size_t>(ctx.state.cosmo_phinet_cols))) {
      return false;
    }
    if (!dev.cosmo_qscnet.upload(
            static_cast<const double *>(ctx.state.cosmo_qscnet),
            static_cast<std::size_t>(ctx.state.cosmo_qscnet_rows) *
                static_cast<std::size_t>(ctx.state.cosmo_qscnet_cols))) {
      return false;
    }
    if (!dev.cosmo_qdenet.upload(
            static_cast<const double *>(ctx.state.cosmo_qdenet),
            static_cast<std::size_t>(ctx.state.cosmo_qdenet_rows) *
                static_cast<std::size_t>(ctx.state.cosmo_qdenet_cols))) {
      return false;
    }
    if (!dev.cosmo_npoints.upload(
            static_cast<const int *>(ctx.state.cosmo_npoints),
            static_cast<std::size_t>(ctx.state.cosmo_npoints_dim))) {
      return false;
    }
    if (!dev.cosmo_a_diag.upload(
            static_cast<const double *>(ctx.state.cosmo_a_diag),
            cosmo_nps_count)) return false;
    if (ctx.state.cosmo_a_part_dim > 0) {
      if (!ctx.state.cosmo_a_part ||
          !dev.cosmo_a_part.upload(
              static_cast<const double *>(ctx.state.cosmo_a_part),
              static_cast<std::size_t>(ctx.state.cosmo_a_part_dim))) {
        return false;
      }
      if (!ctx.state.cosmo_a_part_i || !ctx.state.cosmo_a_part_j ||
          !dev.cosmo_a_part_i.upload(
              static_cast<const int *>(ctx.state.cosmo_a_part_i),
              static_cast<std::size_t>(ctx.state.cosmo_a_part_dim)) ||
          !dev.cosmo_a_part_j.upload(
              static_cast<const int *>(ctx.state.cosmo_a_part_j),
              static_cast<std::size_t>(ctx.state.cosmo_a_part_dim))) {
        return false;
      }
    } else {
      dev.cosmo_a_part.reset();
      dev.cosmo_a_part_i.reset();
      dev.cosmo_a_part_j.reset();
    }
    if (!dev.cosmo_m_vec.upload(
            static_cast<const double *>(ctx.state.cosmo_m_vec),
            static_cast<std::size_t>(ctx.state.cosmo_m_vec_dim))) {
      return false;
    }
    if (!dev.cosmo_iblock_pos.upload(
            static_cast<const int *>(ctx.state.cosmo_iblock_pos),
            static_cast<std::size_t>(ctx.state.cosmo_iblock_pos_dim))) {
      return false;
    }
    double cosmo_scalars[kCosmoScalarCount] = {
        ctx.state.cosmo_solv_energy, ctx.state.cosmo_ediel};
    if (!dev.cosmo_scalars.upload(cosmo_scalars, kCosmoScalarCount)) {
      return false;
    }
    double cosmo_status_scalars[kCosmoStatusDoubleCount] = {};
    cosmo_status_scalars[kCosmoStatusCurrentTol] = 1.0e-2;
    cosmo_status_scalars[kCosmoStatusTargetTol] = 1.0e-2;
    cosmo_status_scalars[kCosmoStatusSolvEnergy] =
        ctx.state.cosmo_solv_energy;
    cosmo_status_scalars[kCosmoStatusEdiel] = ctx.state.cosmo_ediel;
    if (!dev.cosmo_status_scalars.upload(cosmo_status_scalars,
                                         kCosmoStatusDoubleCount)) {
      return false;
    }
    int cosmo_status_ints[kCosmoStatusIntCount] = {};
    cosmo_status_ints[kCosmoStatusControlResident] = 1;
    cosmo_status_ints[kCosmoStatusHostSyncs] = 0;
    cosmo_status_ints[kCosmoStatusNewSurface] =
        ctx.state.cosmo_new_surface ? 1 : 0;
    if (!dev.cosmo_status_ints.upload(cosmo_status_ints,
                                      kCosmoStatusIntCount)) {
      return false;
    }
    if (!dev.cosmo_cg_x.resize(cosmo_nps_count)) return false;
    if (!dev.cosmo_cg_r.resize(cosmo_nps_count)) return false;
    if (!dev.cosmo_cg_p.resize(cosmo_nps_count)) return false;
    if (!dev.cosmo_cg_q.resize(cosmo_nps_count)) return false;
    if (!dev.cosmo_cg_z.resize(cosmo_nps_count)) return false;
    if (!dev.cosmo_cg_tmp.resize(cosmo_nps_count)) return false;
    if (!dev.cosmo_cg_scalars.resize(kCosmoCgScalarCount)) return false;
    if (!dev.cosmo_cg_ints.resize(kCosmoCgIntCount)) return false;
  } else {
    dev.coord.reset();
    dev.nat.reset();
    dev.param_dd.reset();
    dev.param_qq.reset();
    dev.param_tore.reset();
    dev.cosmo_iatsp.reset();
    dev.cosmo_ipiden.reset();
    dev.cosmo_gden.reset();
    dev.cosmo_qscat.reset();
    dev.cosmo_srad.reset();
    dev.cosmo_cosurf.reset();
    dev.cosmo_phinet.reset();
    dev.cosmo_qscnet.reset();
    dev.cosmo_qdenet.reset();
    dev.cosmo_npoints.reset();
    dev.cosmo_a_diag.reset();
    dev.cosmo_a_part.reset();
    dev.cosmo_a_part_i.reset();
    dev.cosmo_a_part_j.reset();
    dev.cosmo_m_vec.reset();
    dev.cosmo_iblock_pos.reset();
    dev.cosmo_scalars.reset();
    dev.cosmo_cg_x.reset();
    dev.cosmo_cg_r.reset();
    dev.cosmo_cg_p.reset();
    dev.cosmo_cg_q.reset();
    dev.cosmo_cg_z.reset();
    dev.cosmo_cg_tmp.reset();
    dev.cosmo_cg_scalars.reset();
    dev.cosmo_cg_ints.reset();
    dev.cosmo_status_scalars.reset();
    dev.cosmo_status_ints.reset();
  }

  constexpr int kCnvgzThreads = 256;
  const int matrix_blocks = ceil_div(mpack, kCnvgzThreads);
  const int diag_blocks = ceil_div(norbs, kCnvgzThreads);
  if (!dev.atom_sums.resize(numat_count)) return false;
  if (!dev.atom_diag_sums.resize(numat_count)) return false;
  if (!dev.energy_sums.resize(3)) return false;
  if (!dev.diag_new.resize(norbs_count)) return false;
  if (!dev.diag_old.resize(norbs_count)) return false;
  if (!dev.candidate.resize(mpack_count)) return false;
  if (!dev.isitsc_escf0.upload(ctx.config.isitsc_escf0, 10)) return false;
  int initial_isitsc_ints[kIsitscIntCount] = {};
  initial_isitsc_ints[kIsitscIntIemin] = ctx.config.isitsc_iemin;
  initial_isitsc_ints[kIsitscIntIemax] = ctx.config.isitsc_iemax;
  initial_isitsc_ints[kIsitscIntScf1] = ctx.config.isitsc_scf1;
  if (!dev.isitsc_ints.upload(initial_isitsc_ints, kIsitscIntCount)) {
    return false;
  }
  int initial_control_ints[kResidentControlIntCount] = {};
  initial_control_ints[kResidentControlDecision] = kResidentDecisionContinue;
  initial_control_ints[kResidentControlDiaggMode] = ctx.config.diagg_mode;
  initial_control_ints[kResidentControlNhb] = ctx.config.nhb;
  initial_control_ints[kResidentControlDiaggNf] = ctx.config.diagg_nf;
  initial_control_ints[kResidentControlNrej0] = ctx.config.diagg2_nrejct[0];
  initial_control_ints[kResidentControlNrej1] = ctx.config.diagg2_nrejct[1];
  initial_control_ints[kResidentControlIemin] = ctx.config.isitsc_iemin;
  initial_control_ints[kResidentControlIemax] = ctx.config.isitsc_iemax;
  initial_control_ints[kResidentControlScf1] = ctx.config.isitsc_scf1;
  initial_control_ints[kResidentControlCurrentIter] = ctx.config.current_iter;
  initial_control_ints[kResidentControlAddhbDue] = ctx.config.addhb_due;
  initial_control_ints[kResidentControlUseThreePoint] =
      ctx.config.use_three_point;
  initial_control_ints[kResidentControlLstart] = ctx.config.lstart;
  if (!dev.resident_control_ints.upload(initial_control_ints,
                                        kResidentControlIntCount)) {
    return false;
  }
  double initial_control_scalars[kResidentControlDoubleCount] = {};
  initial_control_scalars[kResidentControlDiaggFref] = ctx.config.diagg_fref;
  initial_control_scalars[kResidentControlDiaggOldlim] =
      ctx.config.diagg_oldlim;
  initial_control_scalars[kResidentControlDiaggSafety] =
      ctx.config.diagg_safety;
  initial_control_scalars[kResidentControlOvmax] = ctx.config.ovmax;
  initial_control_scalars[kResidentControlPreviousEscf] =
      ctx.config.previous_escf;
  initial_control_scalars[kResidentControlShift] = ctx.config.shift;
  if (!dev.resident_control_scalars.upload(initial_control_scalars,
                                           kResidentControlDoubleCount)) {
    return false;
  }
  int initial_pls_ints[kPlsIntCount] = {};
  initial_pls_ints[kPlsIntLoop] = -1;
  if (!dev.pls_ints.upload(initial_pls_ints, kPlsIntCount)) return false;
  double initial_pls_scalars[kPlsDoubleCount] = {};
  if (!dev.pls_scalars.upload(initial_pls_scalars, kPlsDoubleCount)) {
    return false;
  }
  if (!dev.block_max.resize(static_cast<std::size_t>(matrix_blocks))) {
    return false;
  }
  if (!dev.block_sumsq.resize(static_cast<std::size_t>(matrix_blocks))) {
    return false;
  }
  if (!dev.block_faca.resize(static_cast<std::size_t>(diag_blocks))) {
    return false;
  }
  if (!dev.block_facb.resize(static_cast<std::size_t>(diag_blocks))) {
    return false;
  }
  if (!dev.eimp_pair_updates.resize(3)) return false;
  if (!dev.density_updates.resize(3)) return false;
  if (!dev.qe.resize(numat_count)) return false;
  if (!dev.cnvgz_sums.resize(kCnvgzControlCount)) return false;
  if (!dev.cnvgz_ints.resize(kCnvgzIntCount)) return false;
  if (!cuda_context_ok(
          cudaMemset(dev.cnvgz_ints.ptr, 0, kCnvgzIntCount * sizeof(int)),
          "resident upload cnvgz integer reset")) {
    return false;
  }
  if (!dev.isitsc_scalars.resize(kIsitscDoubleCount)) return false;
  if (!dev.check_ints.resize(kCheckIntCount)) return false;
  if (!dev.check_errors.resize(kCheckDoubleCount)) return false;
  if (!dev.helecz_ints.resize(kHeleczIntCount)) return false;
  if (!dev.fock_ints.resize(kFockIntCount)) return false;
  if (!dev.resident_stage_ints.resize(kResidentStageIntCount)) return false;
  int initial_stage_calls[kResidentStageSlotCount] = {};
  if (!dev.resident_stage_calls.upload(initial_stage_calls,
                                       kResidentStageSlotCount)) {
    return false;
  }
  if (!dev.diagg_ints.resize(kDiaggIntCount)) return false;
  if (!dev.diagg_scalars.resize(kDiaggDoubleCount)) return false;
  if (!dev.addhb_ints.resize(kAddhbIntCount)) return false;
  if (!dev.addhb_scalars.resize(kAddhbDoubleCount)) return false;
  if (!dev.diagg_aocc.resize(
          static_cast<std::size_t>(ctx.state.icocc_dim))) {
    return false;
  }
  {
    const std::size_t nocc_count =
        static_cast<std::size_t>(std::max(1, ctx.config.noccupied));
    const std::size_t nvir_count =
        static_cast<std::size_t>(std::max(1, ctx.config.nvirtual));
    const std::size_t fmo_count =
        static_cast<std::size_t>(std::max(1, ctx.state.fmo_dim));
    const std::size_t hb_capacity = hbond_pair_capacity(numat);
    if (!dev.diagg_work_ints.resize(kDiaggWorkIntCount)) return false;
    if (!dev.diagg_work_scalars.resize(kDiaggWorkDoubleCount)) return false;
    if (!dev.diagg_avir_entry.resize(
            static_cast<std::size_t>(std::max(1, ctx.state.icvir_dim)))) {
      return false;
    }
    if (!dev.diagg_counts.resize(nvir_count)) return false;
    if (!dev.diagg_offsets.resize(nvir_count + 1)) return false;
    if (!dev.diagg_pair_state.resize(fmo_count)) return false;
    if (!dev.diagg_vclaim.resize(nvir_count)) return false;
    if (!dev.diagg_oclaim.resize(nocc_count)) return false;
    if (!dev.hb_pair_counts.resize(numat_count)) return false;
    if (!dev.hb_pair_offsets.resize(numat_count + 1)) return false;
    if (!dev.hb_pair_i.resize(hb_capacity)) return false;
    if (!dev.hb_pair_j.resize(hb_capacity)) return false;
    if (!dev.hb_entry_counts.resize(hb_capacity)) return false;
    if (!dev.hb_entry_offsets.resize(hb_capacity + 1)) return false;
    const std::size_t lmo_count = std::max(nocc_count, nvir_count);
    if (!dev.tidy_iused.resize(lmo_count)) return false;
    if (!dev.tidy_ncnew.resize(lmo_count)) return false;
    if (!dev.tidy_ncmnew.resize(lmo_count)) return false;
    if (!dev.tidy_nncnew.resize(lmo_count)) return false;
    if (!dev.tidy_result.resize(8)) return false;
  }
  if (!dev.final_reorth_ws.resize(norbs_count)) return false;
  if (!dev.final_reorth_sumtot.resize(1)) return false;
  if (!dev.final_reorth_status_scalars.resize(
          kFinalReorthStatusDoubleCount)) return false;
  if (!dev.final_reorth_latom.resize(numat_count)) return false;
  if (!dev.final_reorth_iused.resize(numat_count)) return false;
  if (!dev.final_reorth_status.resize(1)) return false;
  if (!dev.ensure_events()) return false;
  mozyme_resident_stage_upload_accept_kernel<<<1, 1>>>(
      dev.resident_stage_calls.ptr, dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident stage upload accept kernel")) {
    return false;
  }

  // Resident SCF stages operate on these device buffers and copy them back only
  // after a whole resident iteration completes.
  dev.uploaded = true;
  return true;
}

bool compute_eimp_probe_on_gpu(MozymeScfContext &ctx, double *wall_ms) {
  if (!wall_ms || !ctx.device.uploaded || !ctx.state.use_nijbo) {
    return false;
  }

  const int numat = ctx.config.natoms;
  const int mpack = ctx.config.mpack;
  if (numat <= 0 || mpack <= 0) return false;

  auto &dev = ctx.device;
  bool ok = false;
  do {
    const bool time_stage = resident_stage_timing_enabled(ctx, wall_ms);
    const std::size_t mpack_count = static_cast<std::size_t>(mpack);
    if (!device_buffer_ready(dev.eimp_p, mpack_count)) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(dev.eimp_p.ptr, dev.p.ptr,
                            mpack_count * sizeof(double),
                            cudaMemcpyDeviceToDevice, 0),
            "resident eimp shadow density copy")) {
      break;
    }
    if (!device_buffer_ready(dev.eimp_pair_updates, 3)) break;
    if (!zero_ints_if_resident_active(ctx, dev.eimp_pair_updates.ptr, 3,
                                      "resident eimp status reset")) {
      break;
    }
    mozyme_set_int_slot_if_resident_active_kernel<<<1, 1>>>(
        dev.eimp_pair_updates.ptr, 1, 1, dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident eimp status init kernel")) break;

    dim3 block(16, 16);
    dim3 grid(ceil_div(numat, static_cast<int>(block.x)),
              ceil_div(numat, static_cast<int>(block.y)));
    if (!begin_resident_stage_timing(ctx, time_stage,
                                     "resident eimp start event")) break;
    mozyme_eimp_kernel<<<grid, block>>>(numat, mpack, dev.iorbs.ptr,
                                        dev.nijbo.ptr, dev.f.ptr,
                                        dev.eimp_p.ptr,
                                        dev.eimp_pair_updates.ptr,
                                        dev.eimp_pair_updates.ptr + 1,
                                        dev.eimp_pair_updates.ptr + 2,
                                        dev.resident_control_ints.ptr);
    mozyme_update_status_finalize_kernel<<<1, 1>>>(
        dev.eimp_pair_updates.ptr, 0, dev.resident_control_ints.ptr);
    if (!finish_resident_stage_timing(
            ctx, time_stage, "resident eimp stop event",
            "resident eimp kernels", "resident eimp synchronize",
            "resident eimp elapsed time", wall_ms)) break;
    ok = true;
  } while (false);

  return ok;
}

// MOPAC_MOZYME_DIAGG_DEBUG=1: synchronize after each DIAGG kernel group and
// report progress/errors on stderr (diagnostic only; adds host syncs).
bool diagg_debug_enabled() {
  static const bool enabled = env_enabled("MOPAC_MOZYME_DIAGG_DEBUG");
  return enabled;
}

void diagg_debug_checkpoint(const char *where) {
  if (!diagg_debug_enabled()) return;
  const cudaError_t sync = cudaDeviceSynchronize();
  const cudaError_t last = cudaGetLastError();
  std::fprintf(stderr, "[DIAGG DEBUG] %s: sync=%s last=%s\n", where,
               cudaGetErrorString(sync), cudaGetErrorString(last));
  std::fflush(stderr);
}

void diagg_debug_dump_ints(const char *label, const int *device_ptr, int n) {
  if (!diagg_debug_enabled() || !device_ptr || n <= 0) return;
  std::vector<int> host(static_cast<std::size_t>(n), 0);
  cudaDeviceSynchronize();
  if (cudaMemcpy(host.data(), device_ptr, host.size() * sizeof(int),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "[DIAGG DEBUG] %s: copy failed\n", label);
    return;
  }
  std::fprintf(stderr, "[DIAGG DEBUG] %s:", label);
  for (int v : host) std::fprintf(stderr, " %d", v);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

void diagg_debug_dump_doubles(const char *label, const double *device_ptr,
                              int n) {
  if (!diagg_debug_enabled() || !device_ptr || n <= 0) return;
  std::vector<double> host(static_cast<std::size_t>(n), 0.0);
  cudaDeviceSynchronize();
  if (cudaMemcpy(host.data(), device_ptr, host.size() * sizeof(double),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "[DIAGG DEBUG] %s: copy failed\n", label);
    return;
  }
  std::fprintf(stderr, "[DIAGG DEBUG] %s:", label);
  for (double v : host) std::fprintf(stderr, " %.6g", v);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

// Cooperative launch of the parallel diagg2 sweep: every block must be
// resident at once for grid.sync(), so the grid is sized from occupancy.
bool launch_diagg2_parallel(int max_pairs, DiaggRotateArgs args,
                            const char *label) {
  int device = 0;
  if (!cuda_context_ok(cudaGetDevice(&device), label)) return false;
  int cooperative = 0;
  if (!cuda_context_ok(cudaDeviceGetAttribute(
                           &cooperative, cudaDevAttrCooperativeLaunch, device),
                       label)) {
    return false;
  }
  if (!cooperative) return false;
  int sms = 0;
  if (!cuda_context_ok(cudaDeviceGetAttribute(
                           &sms, cudaDevAttrMultiProcessorCount, device),
                       label)) {
    return false;
  }
  int blocks_per_sm = 0;
  if (!cuda_context_ok(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &blocks_per_sm, mozyme_diagg2_parallel_kernel,
                           kDiaggRotateThreads, 0),
                       label)) {
    return false;
  }
  if (blocks_per_sm <= 0 || sms <= 0) return false;
  int grid = blocks_per_sm * sms;
  const int max_useful =
      std::max(1, (max_pairs + kDiaggRotateWarps - 1) / kDiaggRotateWarps);
  if (grid > max_useful) grid = max_useful;
  void *kernel_args[] = {&args};
  return cuda_context_ok(
      cudaLaunchCooperativeKernel(
          reinterpret_cast<void *>(mozyme_diagg2_parallel_kernel), dim3(grid),
          dim3(kDiaggRotateThreads), kernel_args, 0, nullptr),
      label);
}

DiaggRotateArgs make_diagg_rotate_args(MozymeScfContext &ctx,
                                       const int *control_ints, int nij_slot,
                                       int retry_slot,
                                       const double *control_scalars,
                                       int tiny_slot, int biglim_slot,
                                       double *sumb_out, int *nrej_out,
                                       int *ok_slot) {
  auto &dev = ctx.device;
  DiaggRotateArgs a{};
  a.control_ints = control_ints;
  a.nij_slot = nij_slot;
  a.retry_slot = retry_slot;
  a.control_scalars = control_scalars;
  a.tiny_slot = tiny_slot;
  a.biglim_slot = biglim_slot;
  a.nocc = ctx.config.noccupied;
  a.nvir = ctx.config.nvirtual;
  a.numat = ctx.config.natoms;
  a.norbs = ctx.config.norbs;
  a.icocc_dim = ctx.state.icocc_dim;
  a.icvir_dim = ctx.state.icvir_dim;
  a.cocc_dim = ctx.state.cocc_dim;
  a.cvir_dim = ctx.state.cvir_dim;
  a.ifmo = dev.ifmo.ptr;
  a.fmo = dev.fmo.ptr;
  a.eigs = dev.eigs.ptr;
  a.eigv = dev.eigs.ptr + ctx.config.noccupied;
  a.nncf = dev.nncf.ptr;
  a.ncf = dev.ncf.ptr;
  a.ncocc = dev.ncocc.ptr;
  a.icocc = dev.icocc.ptr;
  a.nnce = dev.nnce.ptr;
  a.nce = dev.nce.ptr;
  a.ncvir = dev.ncvir.ptr;
  a.icvir = dev.icvir.ptr;
  a.iorbs = dev.iorbs.ptr;
  a.cocc = dev.cocc.ptr;
  a.cvir = dev.cvir.ptr;
  a.shift = ctx.config.shift;
  a.rot_const = ctx.config.diagg_rot_const;
  a.thresh = ctx.config.thresh;
  a.pair_state = dev.diagg_pair_state.ptr;
  a.vclaim = dev.diagg_vclaim.ptr;
  a.oclaim = dev.diagg_oclaim.ptr;
  a.work_ints = dev.diagg_work_ints.ptr;
  a.sumb_out = sumb_out;
  a.nrej_out = nrej_out;
  a.ok_slot = ok_slot;
  a.resident_control_ints = dev.resident_control_ints.ptr;
  a.resident_control_scalars = dev.resident_control_scalars.ptr;
  return a;
}

bool diagg_parallel_buffers_ready(MozymeScfContext &ctx) {
  auto &dev = ctx.device;
  const std::size_t nocc = static_cast<std::size_t>(ctx.config.noccupied);
  const std::size_t nvir = static_cast<std::size_t>(ctx.config.nvirtual);
  const std::size_t fmo = static_cast<std::size_t>(ctx.state.fmo_dim);
  return device_buffer_ready(dev.diagg_work_ints, kDiaggWorkIntCount) &&
         device_buffer_ready(dev.diagg_work_scalars, kDiaggWorkDoubleCount) &&
         device_buffer_ready(dev.diagg_pair_state, fmo) &&
         device_buffer_ready(dev.diagg_vclaim, nvir) &&
         device_buffer_ready(dev.diagg_oclaim, nocc) &&
         device_buffer_ready(dev.diagg_counts, nvir) &&
         device_buffer_ready(dev.diagg_offsets, nvir + 1);
}

bool compute_check_on_gpu(MozymeScfContext &ctx, double *wall_ms) {
  if (!wall_ms || !ctx.device.uploaded) return false;

  const int nocc = ctx.config.noccupied;
  const int nvir = ctx.config.nvirtual;
  const int numat = ctx.config.natoms;
  if (nocc <= 0 || nvir <= 0 || numat <= 0 || ctx.state.icocc_dim <= 0 ||
      ctx.state.icvir_dim <= 0 || ctx.state.cocc_dim <= 0 ||
      ctx.state.cvir_dim <= 0) {
    return false;
  }

  auto &dev = ctx.device;
  bool ok = false;
  do {
    const bool time_stage = resident_stage_timing_enabled(ctx, wall_ms);
    if (!device_buffer_ready(dev.check_errors, kCheckDoubleCount)) break;
    if (!device_buffer_ready(dev.check_ints, kCheckIntCount)) break;
    if (!zero_ints_if_resident_active(ctx, dev.check_ints.ptr, kCheckIntCount,
                                      "resident check integer reset")) {
      break;
    }
    mozyme_check_init_kernel<<<1, 1>>>(nocc, nvir, dev.check_ints.ptr,
                                       dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident check integer init kernel")) break;
    if (!zero_doubles_if_resident_active(ctx, dev.check_errors.ptr,
                                         kCheckDoubleCount,
                                         "resident check errors reset")) {
      break;
    }

    if (!begin_resident_stage_timing(ctx, time_stage,
                                     "resident check start event")) break;
    // The CPU loop tidies both LMO sets at the top of every iteration
    // (compaction + even redistribution of free space); without it the
    // resident sweep runs out of room to grow LMOs and rejects every rotation.
    {
      const std::size_t lmo_count =
          static_cast<std::size_t>(std::max(nocc, nvir));
      if (!device_buffer_ready(dev.tidy_iused, lmo_count) ||
          !device_buffer_ready(dev.tidy_ncnew, lmo_count) ||
          !device_buffer_ready(dev.tidy_ncmnew, lmo_count) ||
          !device_buffer_ready(dev.tidy_nncnew, lmo_count) ||
          !device_buffer_ready(dev.tidy_result, 8)) {
        break;
      }
      mozyme_tidy_resident_kernel<<<1, 1>>>(
          nocc, numat, ctx.config.norbs, ctx.state.icocc_dim,
          ctx.state.cocc_dim, ctx.config.thresh, 1, dev.ncf.ptr,
          dev.icocc.ptr, dev.cocc.ptr, dev.nncf.ptr, dev.ncocc.ptr,
          dev.iorbs.ptr, dev.tidy_iused.ptr, dev.tidy_ncnew.ptr,
          dev.tidy_ncmnew.ptr, dev.tidy_nncnew.ptr, dev.tidy_result.ptr,
          dev.resident_control_ints.ptr, dev.check_ints.ptr + kCheckIntOk);
      mozyme_tidy_resident_kernel<<<1, 1>>>(
          nvir, numat, ctx.config.norbs, ctx.state.icvir_dim,
          ctx.state.cvir_dim, ctx.config.thresh, 2, dev.nce.ptr,
          dev.icvir.ptr, dev.cvir.ptr, dev.nnce.ptr, dev.ncvir.ptr,
          dev.iorbs.ptr, dev.tidy_iused.ptr, dev.tidy_ncnew.ptr,
          dev.tidy_ncmnew.ptr, dev.tidy_nncnew.ptr, dev.tidy_result.ptr + 4,
          dev.resident_control_ints.ptr, dev.check_ints.ptr + kCheckIntOk);
      if (!cuda_context_ok(cudaGetLastError(), "resident tidy kernels")) break;
      diagg_debug_checkpoint("resident tidy");
      diagg_debug_dump_ints("resident tidy results", dev.tidy_result.ptr, 8);
    }
    const int occ_blocks = (nocc * 32 + kDiaggRotateThreads - 1) /
                           kDiaggRotateThreads;
    const int vir_blocks = (nvir * 32 + kDiaggRotateThreads - 1) /
                           kDiaggRotateThreads;
    mozyme_check_lmo_warp_kernel<<<occ_blocks, kDiaggRotateThreads>>>(
        nocc, numat, ctx.state.icocc_dim, ctx.state.cocc_dim, dev.nncf.ptr,
        dev.ncf.ptr, dev.icocc.ptr, dev.iorbs.ptr, dev.ncocc.ptr,
        dev.cocc.ptr, dev.check_errors.ptr, dev.check_ints.ptr,
        dev.check_ints.ptr + kCheckIntOk, 0, dev.resident_control_ints.ptr);
    mozyme_check_lmo_warp_kernel<<<vir_blocks, kDiaggRotateThreads>>>(
        nvir, numat, ctx.state.icvir_dim, ctx.state.cvir_dim, dev.nnce.ptr,
        dev.nce.ptr, dev.icvir.ptr, dev.iorbs.ptr, dev.ncvir.ptr,
        dev.cvir.ptr, dev.check_errors.ptr, dev.check_ints.ptr,
        dev.check_ints.ptr + kCheckIntOk, 1, dev.resident_control_ints.ptr);
    mozyme_check_finalize_kernel<<<1, 1>>>(nocc, nvir, dev.check_errors.ptr,
                                           dev.check_ints.ptr,
                                           dev.resident_control_ints.ptr);
    if (!finish_resident_stage_timing(
            ctx, time_stage, "resident check stop event",
            "resident check kernels", "resident check synchronize",
            "resident check elapsed time", wall_ms)) break;
    ok = true;
  } while (false);

  return ok;
}

bool compute_diagg_on_gpu(MozymeScfContext &ctx, double *wall_ms) {
  if (!wall_ms || !ctx.device.uploaded || !ctx.state.use_nijbo ||
      !ctx.device.eimp_p.ptr) {
    return false;
  }

  const int nocc = ctx.config.noccupied;
  const int nvir = ctx.config.nvirtual;
  const int numat = ctx.config.natoms;
  const int norbs = ctx.config.norbs;
  const int mpack = ctx.config.mpack;
  const int fmo_dim = ctx.state.fmo_dim;
  if (nocc <= 0 || nvir <= 0 || numat <= 0 || norbs <= 0 || mpack <= 0 ||
      fmo_dim <= 0 || ctx.state.icocc_dim <= 0 || ctx.state.icvir_dim <= 0 ||
      ctx.state.cocc_dim <= 0 || ctx.state.cvir_dim <= 0) {
    return false;
  }
  auto &dev = ctx.device;

  bool ok = false;
  do {
    const bool time_stage = resident_stage_timing_enabled(ctx, wall_ms);
    if (!dev.diagg_aocc.ptr ||
        dev.diagg_aocc.count < static_cast<std::size_t>(ctx.state.icocc_dim)) {
      break;
    }
    if (!dev.diagg_ints.ptr || dev.diagg_ints.count < kDiaggIntCount) break;
    if (!dev.diagg_scalars.ptr ||
        dev.diagg_scalars.count < kDiaggDoubleCount) {
      break;
    }
    if (!zero_ints_if_resident_active(ctx, dev.diagg_ints.ptr, kDiaggIntCount,
                                      "resident diagg ints reset")) break;
    if (!zero_doubles_if_resident_active(ctx, dev.diagg_scalars.ptr,
                                         kDiaggDoubleCount,
                                         "resident diagg scalars reset")) break;

    const double cutlim = 1.0e-8;
    double cutoff = std::max(cutlim, ctx.config.ovmax * 10.0 * cutlim);
    const double flim = std::min(3.0, ctx.config.diagg_fref * 0.5);
    const double oldlim_in = ctx.config.diagg_oldlim;
    const double safety_in = ctx.config.diagg_safety;
    const int nf_in = ctx.config.diagg_nf;
    if (ctx.config.diagg_mode <= 5) cutoff = cutlim;
    double *eigv = dev.eigs.ptr + nocc;
    if (!diagg_parallel_buffers_ready(ctx)) break;
    if (!device_buffer_ready(dev.diagg_avir_entry,
                             static_cast<std::size_t>(ctx.state.icvir_dim))) {
      break;
    }
    if (!zero_ints_if_resident_active(ctx, dev.diagg_work_ints.ptr,
                                      kDiaggWorkIntCount,
                                      "resident diagg work ints reset")) break;
    if (!zero_doubles_if_resident_active(
            ctx, dev.diagg_work_scalars.ptr, kDiaggWorkDoubleCount,
            "resident diagg work scalars reset")) break;

    if (!begin_resident_stage_timing(ctx, time_stage,
                                     "resident diagg start event")) break;
    const int occ_norm_blocks =
        (nocc * 32 + kDiaggBlockThreads - 1) / kDiaggBlockThreads;
    const int vir_norm_blocks =
        (nvir * 32 + kDiaggBlockThreads - 1) / kDiaggBlockThreads;
    mozyme_lmo_entry_norms_kernel<<<occ_norm_blocks, kDiaggBlockThreads>>>(
        nocc, numat, ctx.state.icocc_dim, ctx.state.cocc_dim, dev.nncf.ptr,
        dev.ncf.ptr, dev.icocc.ptr, dev.ncocc.ptr, dev.iorbs.ptr,
        dev.cocc.ptr, dev.diagg_aocc.ptr,
        dev.diagg_work_ints.ptr + kDiaggWorkIntError,
        dev.resident_control_ints.ptr);
    mozyme_lmo_entry_norms_kernel<<<vir_norm_blocks, kDiaggBlockThreads>>>(
        nvir, numat, ctx.state.icvir_dim, ctx.state.cvir_dim, dev.nnce.ptr,
        dev.nce.ptr, dev.icvir.ptr, dev.ncvir.ptr, dev.iorbs.ptr,
        dev.cvir.ptr, dev.diagg_avir_entry.ptr,
        dev.diagg_work_ints.ptr + kDiaggWorkIntError,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident diagg entry norm kernels")) break;
    diagg_debug_checkpoint("resident diagg1 entry norms");

    DiaggVirtualArgs va{};
    va.nocc = nocc;
    va.nvir = nvir;
    va.numat = numat;
    va.norbs = norbs;
    va.mpack = mpack;
    va.icocc_dim = ctx.state.icocc_dim;
    va.icvir_dim = ctx.state.icvir_dim;
    va.cocc_dim = ctx.state.cocc_dim;
    va.cvir_dim = ctx.state.cvir_dim;
    va.fmo_dim = fmo_dim;
    va.idiagg = ctx.config.diagg_mode;
    va.capacity = fmo_dim;
    va.resident_control_scalars = dev.resident_control_scalars.ptr;
    va.fao = dev.f.ptr;
    va.p = dev.eimp_p.ptr;
    va.ncf = dev.ncf.ptr;
    va.nce = dev.nce.ptr;
    va.nncf = dev.nncf.ptr;
    va.nnce = dev.nnce.ptr;
    va.ncocc = dev.ncocc.ptr;
    va.ncvir = dev.ncvir.ptr;
    va.icocc = dev.icocc.ptr;
    va.icvir = dev.icvir.ptr;
    va.iorbs = dev.iorbs.ptr;
    va.nijbo = dev.nijbo.ptr;
    va.cocc = dev.cocc.ptr;
    va.cvir = dev.cvir.ptr;
    va.aocc = dev.diagg_aocc.ptr;
    va.avir_entry = dev.diagg_avir_entry.ptr;
    va.cutoff = cutoff;
    va.flim = flim;
    va.oldlim = oldlim_in;
    va.eigv = eigv;
    va.nfmo = dev.nfmo.ptr;
    va.counts = dev.diagg_counts.ptr;
    va.offsets = dev.diagg_offsets.ptr;
    va.ifmo = dev.ifmo.ptr;
    va.fmo = dev.fmo.ptr;
    va.work_scalars = dev.diagg_work_scalars.ptr;
    va.work_ints = dev.diagg_work_ints.ptr;
    va.resident_control_ints = dev.resident_control_ints.ptr;

    va.fill = 0;
    mozyme_diagg1_virtual_kernel<<<nvir, kDiaggBlockThreads>>>(va);
    mozyme_exclusive_scan_kernel<<<1, 1024>>>(
        nvir, nullptr, dev.diagg_counts.ptr, dev.diagg_offsets.ptr,
        dev.resident_control_ints.ptr);
    va.fill = 1;
    mozyme_diagg1_virtual_kernel<<<nvir, kDiaggBlockThreads>>>(va);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident diagg1 virtual kernels")) break;
    diagg_debug_checkpoint("resident diagg1 virtual passes");
    diagg_debug_dump_ints("resident diagg1 work ints", dev.diagg_work_ints.ptr,
                          kDiaggWorkIntCount);
    diagg_debug_dump_ints("resident diagg1 total candidates",
                          dev.diagg_offsets.ptr + nvir, 1);
    diagg_debug_dump_doubles("resident diagg1 sumt/tiny",
                             dev.diagg_work_scalars.ptr, 2);
    mozyme_diagg1_occupied_eigs_kernel<<<nocc, kDiaggBlockThreads>>>(
        nocc, numat, mpack, ctx.state.icocc_dim, ctx.state.cocc_dim,
        dev.f.ptr, dev.eimp_p.ptr, dev.ncf.ptr, dev.nncf.ptr, dev.ncocc.ptr,
        dev.icocc.ptr, dev.iorbs.ptr, dev.nijbo.ptr, dev.cocc.ptr,
        dev.diagg_aocc.ptr, cutoff, ctx.config.diagg_mode, dev.eigs.ptr,
        dev.diagg_work_ints.ptr, dev.resident_control_ints.ptr,
        dev.resident_control_scalars.ptr);
    mozyme_diagg1_finalize_kernel<<<1, 1>>>(
        nvir, fmo_dim, ctx.config.diagg_mode, nf_in, safety_in, oldlim_in,
        dev.diagg_offsets.ptr, dev.nfmo.ptr, dev.diagg_work_ints.ptr,
        dev.diagg_work_scalars.ptr, dev.diagg_ints.ptr + kDiaggIntNij,
        dev.diagg_ints.ptr + kDiaggIntIjc, dev.diagg_ints.ptr + kDiaggIntNf,
        dev.diagg_scalars.ptr + kDiaggDoubleSumt,
        dev.diagg_scalars.ptr + kDiaggDoubleTiny,
        dev.diagg_scalars.ptr + kDiaggDoubleFref,
        dev.diagg_scalars.ptr + kDiaggDoubleOldlim,
        dev.diagg_scalars.ptr + kDiaggDoubleSafety,
        dev.diagg_ints.ptr + kDiaggIntOk, dev.resident_control_ints.ptr,
        dev.resident_control_scalars.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident diagg1 finalize kernels")) break;
    diagg_debug_checkpoint("resident diagg1 finalize");
    diagg_debug_dump_ints("resident diagg ints after diagg1",
                          dev.diagg_ints.ptr, kDiaggIntCount);
    diagg_debug_dump_doubles("resident diagg scalars after diagg1",
                             dev.diagg_scalars.ptr, kDiaggDoubleCount);
    diagg_debug_dump_ints("resident control ints", dev.resident_control_ints.ptr,
                          8);
    diagg_debug_dump_doubles("resident control scalars",
                             dev.resident_control_scalars.ptr, 8);

    mozyme_diagg2_prepare_control_kernel<<<1, 1>>>(
        ctx.config.diagg_mode, ctx.config.diagg_bigeps,
        ctx.config.diagg2_nrejct[0], ctx.config.diagg2_nrejct[1],
        fmo_dim, dev.diagg_ints.ptr, dev.diagg_scalars.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident diagg control kernel")) break;
    if (!launch_diagg2_parallel(
            fmo_dim,
            make_diagg_rotate_args(
                ctx, dev.diagg_ints.ptr, kDiaggIntNij, kDiaggIntRetry,
                dev.diagg_scalars.ptr, kDiaggDoubleRotateTiny,
                kDiaggDoubleBiglim, dev.diagg_scalars.ptr + kDiaggDoubleSumb,
                dev.diagg_ints.ptr + kDiaggIntNrej,
                dev.diagg_ints.ptr + kDiaggIntOk),
            "resident diagg rotate kernel")) {
      break;
    }
    diagg_debug_checkpoint("resident diagg2 rotate");
    diagg_debug_dump_ints("resident diagg ints after diagg2",
                          dev.diagg_ints.ptr, kDiaggIntCount);
    diagg_debug_dump_doubles("resident diagg scalars after diagg2",
                             dev.diagg_scalars.ptr, kDiaggDoubleCount);
    mozyme_diagg2_finalize_control_kernel<<<1, 1>>>(
        ctx.config.diagg2_nrejct[0], fmo_dim, dev.diagg_ints.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident diagg finalize kernel")) break;

    if (!finish_resident_stage_timing(
            ctx, time_stage, "resident diagg stop event",
            "resident diagg kernels", "resident diagg synchronize",
            "resident diagg elapsed time", wall_ms)) break;
    ok = true;
  } while (false);

  return ok;
}

bool compute_addhb_on_gpu(MozymeScfContext &ctx, double *wall_ms) {
  if (!wall_ms || !ctx.device.uploaded || !ctx.state.use_nijbo ||
      !ctx.device.diagg_ints.ptr || !ctx.device.diagg_scalars.ptr) {
    return false;
  }

  const int nocc = ctx.config.noccupied;
  const int nvir = ctx.config.nvirtual;
  const int numat = ctx.config.natoms;
  const int norbs = ctx.config.norbs;
  const int fmo_dim = ctx.state.fmo_dim;
  if (nocc <= 0 || nvir <= 0 || numat <= 0 || norbs <= 0 ||
      fmo_dim <= 0 || ctx.state.icocc_dim <= 0 || ctx.state.icvir_dim <= 0 ||
      ctx.state.cocc_dim <= 0 || ctx.state.cvir_dim <= 0) {
    return false;
  }

  auto &dev = ctx.device;

  bool ok = false;
  do {
    const bool time_stage = resident_stage_timing_enabled(ctx, wall_ms);
    const std::size_t numat_count = static_cast<std::size_t>(numat);

    if (!dev.addhb_ints.ptr || dev.addhb_ints.count < kAddhbIntCount) break;
    if (!dev.addhb_scalars.ptr ||
        dev.addhb_scalars.count < kAddhbDoubleCount) {
      break;
    }
    if (!zero_ints_if_resident_active(ctx, dev.addhb_ints.ptr, kAddhbIntCount,
                                      "resident addhb ints reset")) break;
    if (!zero_doubles_if_resident_active(ctx, dev.addhb_scalars.ptr,
                                         kAddhbDoubleCount,
                                         "resident addhb scalars reset")) break;

    if (!begin_resident_stage_timing(ctx, time_stage,
                                     "resident addhb start event")) break;
    mozyme_addhb_prepare_control_kernel<<<1, 1>>>(
        ctx.config.current_iter, ctx.config.addhb_due, ctx.config.nhb,
        ctx.config.diagg_mode, ctx.config.diagg_bigeps, fmo_dim,
        dev.diagg_ints.ptr, dev.diagg_scalars.ptr, dev.addhb_ints.ptr,
        dev.addhb_scalars.ptr, dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident addhb control kernel")) break;
    if (!diagg_parallel_buffers_ready(ctx)) break;
    const std::size_t hb_capacity = hbond_pair_capacity(numat);
    if (!device_buffer_ready(dev.hb_pair_counts, numat_count) ||
        !device_buffer_ready(dev.hb_pair_offsets, numat_count + 1) ||
        !device_buffer_ready(dev.hb_pair_i, hb_capacity) ||
        !device_buffer_ready(dev.hb_pair_j, hb_capacity) ||
        !device_buffer_ready(dev.hb_entry_counts, hb_capacity) ||
        !device_buffer_ready(dev.hb_entry_offsets, hb_capacity + 1)) {
      break;
    }
    if (!zero_ints_if_resident_active(ctx, dev.diagg_work_ints.ptr,
                                      kDiaggWorkIntCount,
                                      "resident addhb work ints reset")) break;
    const int hb_cap = static_cast<int>(hb_capacity);
    mozyme_hbond_pairs_kernel<<<numat, kDiaggBlockThreads>>>(
        0, numat, ctx.config.mpack, hb_cap, dev.iorbs.ptr, dev.nijbo.ptr,
        dev.f.ptr, dev.p.ptr, dev.addhb_ints.ptr, dev.addhb_scalars.ptr,
        dev.hb_pair_counts.ptr, nullptr, nullptr, nullptr,
        dev.diagg_work_ints.ptr, dev.resident_control_ints.ptr);
    mozyme_exclusive_scan_kernel<<<1, 1024>>>(
        numat, nullptr, dev.hb_pair_counts.ptr, dev.hb_pair_offsets.ptr,
        dev.resident_control_ints.ptr);
    mozyme_hbond_pairs_kernel<<<numat, kDiaggBlockThreads>>>(
        1, numat, ctx.config.mpack, hb_cap, dev.iorbs.ptr, dev.nijbo.ptr,
        dev.f.ptr, dev.p.ptr, dev.addhb_ints.ptr, dev.addhb_scalars.ptr,
        dev.hb_pair_counts.ptr, dev.hb_pair_offsets.ptr, dev.hb_pair_i.ptr,
        dev.hb_pair_j.ptr, dev.diagg_work_ints.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident addhb pair kernels")) break;
    const int entry_blocks = std::min(4096, hb_cap);
    const int *npairs_ptr = dev.hb_pair_offsets.ptr + numat;
    mozyme_hbond_entries_kernel<<<entry_blocks, kDiaggBlockThreads>>>(
        0, npairs_ptr, hb_cap, nocc, nvir, ctx.state.icocc_dim,
        ctx.state.icvir_dim, fmo_dim, dev.ncf.ptr, dev.nncf.ptr,
        dev.icocc.ptr, dev.nce.ptr, dev.nnce.ptr, dev.icvir.ptr,
        dev.hb_pair_i.ptr, dev.hb_pair_j.ptr, dev.hb_entry_counts.ptr,
        nullptr, dev.ifmo.ptr, dev.fmo.ptr, dev.diagg_work_ints.ptr,
        dev.resident_control_ints.ptr);
    mozyme_exclusive_scan_kernel<<<1, 1024>>>(
        hb_cap, npairs_ptr, dev.hb_entry_counts.ptr,
        dev.hb_entry_offsets.ptr, dev.resident_control_ints.ptr);
    mozyme_hbond_entries_kernel<<<entry_blocks, kDiaggBlockThreads>>>(
        1, npairs_ptr, hb_cap, nocc, nvir, ctx.state.icocc_dim,
        ctx.state.icvir_dim, fmo_dim, dev.ncf.ptr, dev.nncf.ptr,
        dev.icocc.ptr, dev.nce.ptr, dev.nnce.ptr, dev.icvir.ptr,
        dev.hb_pair_i.ptr, dev.hb_pair_j.ptr, dev.hb_entry_counts.ptr,
        dev.hb_entry_offsets.ptr, dev.ifmo.ptr, dev.fmo.ptr,
        dev.diagg_work_ints.ptr, dev.resident_control_ints.ptr);
    mozyme_hbond_commit_kernel<<<1, 1>>>(
        fmo_dim, hb_cap, dev.hb_pair_offsets.ptr, numat,
        dev.hb_entry_offsets.ptr, dev.addhb_ints.ptr,
        dev.diagg_work_ints.ptr, dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident addhb entry kernels")) break;

    if (!launch_diagg2_parallel(
            fmo_dim,
            make_diagg_rotate_args(
                ctx, dev.addhb_ints.ptr, kAddhbIntNij, kAddhbIntRetry,
                dev.addhb_scalars.ptr, kAddhbDoubleRotateTiny,
                kAddhbDoubleBiglim, dev.addhb_scalars.ptr + kAddhbDoubleSumb,
                dev.addhb_ints.ptr + kAddhbIntNrej,
                dev.addhb_ints.ptr + kAddhbIntOk),
            "resident addhb rotate kernel")) {
      break;
    }
    mozyme_addhb_finalize_control_kernel<<<1, 1>>>(
        dev.diagg_ints.ptr, fmo_dim, dev.addhb_ints.ptr, dev.addhb_scalars.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident addhb finalize kernel")) break;

    if (!finish_resident_stage_timing(
            ctx, time_stage, "resident addhb stop event",
            "resident addhb kernels", "resident addhb synchronize",
            "resident addhb elapsed time", wall_ms)) break;
    ok = true;
  } while (false);

  return ok;
}

bool compute_density_on_gpu_impl(MozymeScfContext &ctx, int nclose, int mode,
                                 const double *input_partp, double *output_p,
                                 int *updated_terms, double *wall_ms) {
  if (!updated_terms || !wall_ms || !ctx.device.uploaded ||
      !ctx.state.use_nijbo || !input_partp || !output_p) {
    return false;
  }

  const int numat = ctx.config.natoms;
  const int mpack = ctx.config.mpack;
  if (nclose <= 0 || numat <= 0 || mpack <= 0 ||
      ctx.state.icocc_dim <= 0 || ctx.state.cocc_dim <= 0 ||
      nclose > ctx.state.nocc_slots) {
    return false;
  }

  auto &dev = ctx.device;
  bool ok = false;
  do {
    const bool time_stage = resident_stage_timing_enabled(ctx, wall_ms);
    if (!device_buffer_ready(dev.density_updates, 3)) break;
    if (!zero_ints_if_resident_active(ctx, dev.density_updates.ptr, 3,
                                      "resident density status reset")) {
      break;
    }
    mozyme_set_int_slot_if_resident_active_kernel<<<1, 1>>>(
        dev.density_updates.ptr, 1, 1, dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident density status init kernel")) break;

    constexpr int kThreads = 256;
    const int matrix_blocks = ceil_div(mpack, kThreads);
    if (!begin_resident_stage_timing(ctx, time_stage,
                                     "resident density start event")) break;
    mozyme_density_init_kernel<<<matrix_blocks, kThreads>>>(
        mpack, mode, input_partp, output_p, dev.resident_control_ints.ptr);
    mozyme_density_expected_kernel<<<nclose, kThreads>>>(
        nclose, numat, mpack, ctx.state.icocc_dim, ctx.state.cocc_dim,
        dev.ncf.ptr, dev.nncf.ptr, dev.ncocc.ptr, dev.icocc.ptr,
        dev.iorbs.ptr, dev.nijbo.ptr, dev.density_updates.ptr + 2,
        dev.density_updates.ptr + 1, dev.resident_control_ints.ptr);
    mozyme_density_resident_kernel<<<nclose, kThreads>>>(
        nclose, numat, mpack, ctx.state.icocc_dim, ctx.state.cocc_dim,
        dev.ncf.ptr, dev.nncf.ptr, dev.ncocc.ptr, dev.icocc.ptr,
        dev.iorbs.ptr, dev.nijbo.ptr, dev.cocc.ptr, output_p,
        dev.density_updates.ptr, dev.density_updates.ptr + 1,
        dev.resident_control_ints.ptr);
    mozyme_update_status_finalize_kernel<<<1, 1>>>(
        dev.density_updates.ptr, 1, dev.resident_control_ints.ptr);
    mozyme_density_spin_scale_kernel<<<matrix_blocks, kThreads>>>(
        mpack, mode, output_p, dev.resident_control_ints.ptr);
    if (!finish_resident_stage_timing(
            ctx, time_stage, "resident density stop event",
            "resident density kernels", "resident density synchronize",
            "resident density elapsed time", wall_ms)) break;
    *updated_terms = 0;
    ok = true;
  } while (false);

  return ok;
}

bool compute_density_on_gpu(MozymeScfContext &ctx, int *updated_terms,
                            double *wall_ms) {
  return compute_density_on_gpu_impl(
      ctx, ctx.config.noccupied, ctx.config.density_indi, ctx.device.partp.ptr,
      ctx.device.p.ptr, updated_terms, wall_ms);
}

bool zero_device_scalar(double *device_value, const char *label) {
  return device_value &&
         cuda_context_ok(cudaMemset(device_value, 0, sizeof(double)), label);
}

bool run_cosmo_matvec(MozymeScfContext &ctx, const double *x, double *y,
                      int *cg_ints) {
  if (!x || !y) return false;
  auto &dev = ctx.device;
  const int nps = ctx.state.cosmo_nps;
  if (nps <= 0 || !dev.cosmo_cosurf.ptr || !dev.cosmo_a_diag.ptr) {
    return false;
  }
  constexpr int kThreads = 128;
  const int nps_blocks = ceil_div(nps, kThreads);
  mozyme_cosmo_matvec_far_kernel<<<nps_blocks, kThreads>>>(
      nps, ctx.state.cosmo_cosurf_rows, ctx.state.cosmo_disex2,
      dev.cosmo_cosurf.ptr, dev.cosmo_a_diag.ptr, x, y, cg_ints,
      dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident COSMO matrix-free far matvec kernel")) {
    return false;
  }
  if (ctx.state.cosmo_a_part_dim > 0) {
    if (!dev.cosmo_a_part.ptr || !dev.cosmo_a_part_i.ptr ||
        !dev.cosmo_a_part_j.ptr) {
      return false;
    }
    mozyme_cosmo_matvec_close_kernel<<<ceil_div(
        ctx.state.cosmo_a_part_dim, kThreads), kThreads>>>(
        nps, ctx.state.cosmo_a_part_dim, dev.cosmo_a_part_i.ptr,
        dev.cosmo_a_part_j.ptr, dev.cosmo_a_part.ptr, x, y, cg_ints,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident COSMO matrix-free close matvec kernel")) {
      return false;
    }
  }
  if (!cg_ints) ctx.cosmo_matvec_calls += 1;
  return true;
}

bool apply_cosmo_fock_on_gpu(MozymeScfContext &ctx, const double *density_p,
                             double *output_f) {
  if (!ctx.state.cosmo_enabled) return true;
  if (!density_p || !output_f || !ctx.device.uploaded) return false;

  const int numat = ctx.config.natoms;
  const int mpack = ctx.config.mpack;
  const int nps = ctx.state.cosmo_nps;
  const int lm61 = ctx.state.cosmo_lm61;
  if (numat <= 0 || mpack <= 0 || nps <= 0 || lm61 <= 0) return false;

  auto &dev = ctx.device;
  if (!dev.cosmo_iatsp.ptr || !dev.cosmo_npoints.ptr ||
      !dev.cosmo_iblock_pos.ptr ||
      !dev.cosmo_m_vec.ptr || !dev.cosmo_scalars.ptr) {
    return false;
  }

  constexpr int kThreads = 128;
  const int nps_blocks = ceil_div(nps, kThreads);
  const int lm61_blocks = ceil_div(lm61, kThreads);
  const double fcon = ctx.state.cosmo_a0 * ctx.state.cosmo_ev;
  const std::size_t nps_count = static_cast<std::size_t>(nps);
  if (!device_buffer_ready(dev.cosmo_cg_x, nps_count)) return false;
  if (!device_buffer_ready(dev.cosmo_cg_r, nps_count)) return false;
  if (!device_buffer_ready(dev.cosmo_cg_p, nps_count)) return false;
  if (!device_buffer_ready(dev.cosmo_cg_q, nps_count)) return false;
  if (!device_buffer_ready(dev.cosmo_cg_z, nps_count)) return false;
  if (!device_buffer_ready(dev.cosmo_cg_tmp, nps_count)) return false;
  if (!device_buffer_ready(dev.cosmo_cg_scalars, kCosmoCgScalarCount)) {
    return false;
  }
  if (!device_buffer_ready(dev.cosmo_cg_ints, kCosmoCgIntCount)) return false;
  if (!device_buffer_ready(dev.cosmo_status_scalars,
                           kCosmoStatusDoubleCount)) {
    return false;
  }
  if (!device_buffer_ready(dev.cosmo_status_ints, kCosmoStatusIntCount)) {
    return false;
  }

  if (!zero_doubles_if_resident_active(ctx, dev.cosmo_qscat.ptr, numat,
                                       "resident COSMO qscat reset")) {
    return false;
  }
  if (!zero_doubles_if_resident_active(ctx, dev.cosmo_cg_scalars.ptr,
                                       kCosmoCgScalarCount,
                                       "resident COSMO scalar reset")) {
    return false;
  }
  if (!zero_ints_if_resident_active(ctx, dev.cosmo_cg_ints.ptr,
                                    kCosmoCgIntCount,
                                    "resident COSMO integer reset")) {
    return false;
  }
  mozyme_cosmo_cg_init_control_kernel<<<1, 1>>>(
      ctx.config.ovmax, ctx.config.selcon, dev.resident_control_scalars.ptr,
      dev.cosmo_status_scalars.ptr, dev.cosmo_status_ints.ptr,
      dev.cosmo_cg_scalars.ptr, dev.cosmo_cg_ints.ptr,
      dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident COSMO CG control init kernel")) {
    return false;
  }

  mozyme_cosmo_build_potential_kernel<<<nps_blocks, kThreads>>>(
      numat, nps, mpack, ctx.state.coord_rows, ctx.state.cosmo_cosurf_rows,
      ctx.state.cosmo_phinet_rows, lm61, ctx.state.cosmo_a0, dev.coord.ptr,
      dev.nat.ptr, dev.nfirst.ptr, dev.nlast.ptr, dev.nijbo.ptr,
      dev.cosmo_ipiden.ptr, dev.cosmo_gden.ptr, dev.param_dd.ptr,
      dev.param_qq.ptr, dev.param_tore.ptr, dev.cosmo_cosurf.ptr, density_p,
      dev.cosmo_phinet.ptr, dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident COSMO potential kernel")) {
    return false;
  }

  mozyme_cosmo_initial_x_kernel<<<nps_blocks, kThreads>>>(
      numat, nps, ctx.state.cosmo_qscnet_rows, ctx.state.cosmo_new_surface,
      ctx.state.cosmo_fepsi, dev.cosmo_iatsp.ptr, dev.cosmo_npoints.ptr,
      dev.cosmo_iblock_pos.ptr, dev.cosmo_m_vec.ptr,
      ctx.state.cosmo_m_vec_dim,
      dev.cosmo_phinet.ptr + ctx.state.cosmo_phinet_rows,
      dev.cosmo_qscnet.ptr, dev.cosmo_cg_x.ptr, dev.cosmo_status_ints.ptr,
      dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident COSMO initial CG kernel")) {
    return false;
  }
  if (!run_cosmo_matvec(ctx, dev.cosmo_cg_x.ptr, dev.cosmo_cg_tmp.ptr,
                        dev.cosmo_cg_ints.ptr)) {
    return false;
  }
  mozyme_cosmo_residual_kernel<<<nps_blocks, kThreads>>>(
      nps, dev.cosmo_phinet.ptr + ctx.state.cosmo_phinet_rows,
      dev.cosmo_cg_tmp.ptr, dev.cosmo_cg_r.ptr,
      dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident COSMO initial residual kernels")) {
    return false;
  }

  for (int iter = 0; iter < 100; ++iter) {
    mozyme_cosmo_precondition_kernel<<<nps_blocks, kThreads>>>(
        numat, nps, dev.cosmo_iatsp.ptr, dev.cosmo_npoints.ptr,
        dev.cosmo_iblock_pos.ptr, dev.cosmo_m_vec.ptr,
        ctx.state.cosmo_m_vec_dim, dev.cosmo_cg_r.ptr,
        dev.cosmo_cg_z.ptr, dev.cosmo_cg_ints.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident COSMO precondition kernel")) {
      return false;
    }
    if (!zero_doubles_if_resident_active(
            ctx, dev.cosmo_cg_scalars.ptr + kCosmoCgRho, 1,
            "resident COSMO rho reset")) {
      return false;
    }
    mozyme_cosmo_dot_kernel<<<nps_blocks, kThreads>>>(
        nps, dev.cosmo_cg_r.ptr, dev.cosmo_cg_z.ptr,
        dev.cosmo_cg_scalars.ptr + kCosmoCgRho, dev.cosmo_cg_ints.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident COSMO rho kernel")) {
      return false;
    }
    mozyme_cosmo_cg_prepare_iteration_kernel<<<1, 1>>>(
        dev.cosmo_cg_scalars.ptr, dev.cosmo_cg_ints.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident COSMO CG prepare iteration kernel")) {
      return false;
    }
    mozyme_cosmo_cg_direction_kernel<<<nps_blocks, kThreads>>>(
        nps, dev.cosmo_cg_scalars.ptr, dev.cosmo_cg_ints.ptr,
        dev.cosmo_cg_z.ptr, dev.cosmo_cg_p.ptr,
        dev.resident_control_ints.ptr);
    if (!run_cosmo_matvec(ctx, dev.cosmo_cg_p.ptr, dev.cosmo_cg_q.ptr,
                          dev.cosmo_cg_ints.ptr)) {
      return false;
    }
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident COSMO direction/matvec kernels")) {
      return false;
    }
    if (!zero_doubles_if_resident_active(
            ctx, dev.cosmo_cg_scalars.ptr + kCosmoCgPq, 1,
            "resident COSMO pq reset")) {
      return false;
    }
    mozyme_cosmo_dot_kernel<<<nps_blocks, kThreads>>>(
        nps, dev.cosmo_cg_p.ptr, dev.cosmo_cg_q.ptr,
        dev.cosmo_cg_scalars.ptr + kCosmoCgPq, dev.cosmo_cg_ints.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident COSMO pq kernel")) {
      return false;
    }
    mozyme_cosmo_cg_prepare_update_kernel<<<1, 1>>>(
        dev.cosmo_cg_scalars.ptr, dev.cosmo_cg_ints.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident COSMO CG prepare update kernel")) {
      return false;
    }
    if (!zero_doubles_if_resident_active(
            ctx, dev.cosmo_cg_scalars.ptr + kCosmoCgNorm, 1,
            "resident COSMO residual norm reset")) {
      return false;
    }
    mozyme_cosmo_cg_update_kernel<<<nps_blocks, kThreads>>>(
        nps, dev.cosmo_cg_scalars.ptr, dev.cosmo_cg_ints.ptr,
        dev.cosmo_cg_x.ptr, dev.cosmo_cg_r.ptr, dev.cosmo_cg_p.ptr,
        dev.cosmo_cg_q.ptr,
        dev.cosmo_cg_scalars.ptr + kCosmoCgNorm,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident COSMO CG update kernel")) {
      return false;
    }
    mozyme_cosmo_cg_finish_iteration_kernel<<<1, 1>>>(
        dev.cosmo_cg_scalars.ptr, dev.cosmo_cg_ints.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident COSMO CG finish iteration kernel")) {
      return false;
    }
  }

  if (!zero_doubles_if_resident_active(
          ctx, dev.cosmo_cg_scalars.ptr + kCosmoCgEdiel, 1,
          "resident COSMO ediel reset") ||
      !zero_doubles_if_resident_active(
          ctx, dev.cosmo_cg_scalars.ptr + kCosmoCgS1, 1,
          "resident COSMO s1 reset")) {
    return false;
  }
  mozyme_cosmo_finalize_surface_kernel<<<nps_blocks, kThreads>>>(
      numat, nps, ctx.state.cosmo_phinet_rows, ctx.state.cosmo_qscnet_rows,
      ctx.state.cosmo_fepsi, fcon, dev.cosmo_iatsp.ptr, dev.cosmo_qscat.ptr,
      dev.cosmo_phinet.ptr, dev.cosmo_qscnet.ptr, dev.cosmo_cg_x.ptr,
      dev.cosmo_cg_scalars.ptr + kCosmoCgEdiel,
      dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident COSMO surface finalize kernel")) {
    return false;
  }
  mozyme_cosmo_fock_correction_kernel<<<numat, 64>>>(
      numat, nps, mpack, ctx.state.coord_rows, ctx.state.cosmo_cosurf_rows,
      ctx.state.cosmo_qscnet_rows, lm61, ctx.state.cosmo_a0, fcon,
      dev.coord.ptr, dev.nat.ptr, dev.nfirst.ptr, dev.nlast.ptr,
      dev.nijbo.ptr, dev.cosmo_ipiden.ptr, dev.param_dd.ptr, dev.param_qq.ptr,
      dev.cosmo_cosurf.ptr, dev.cosmo_qscnet.ptr, density_p,
      dev.cosmo_gden.ptr, output_f, dev.cosmo_cg_scalars.ptr + kCosmoCgS1,
      dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident COSMO Fock correction kernel")) {
    return false;
  }
  mozyme_cosmo_update_qdenet_kernel<<<lm61_blocks, kThreads>>>(
      lm61, mpack, ctx.state.cosmo_qdenet_rows, dev.cosmo_ipiden.ptr,
      dev.cosmo_gden.ptr, density_p, dev.cosmo_qdenet.ptr,
      dev.resident_control_ints.ptr);
  mozyme_cosmo_finalize_energy_kernel<<<1, 1>>>(
      fcon, dev.cosmo_cg_scalars.ptr + kCosmoCgEdiel,
      dev.cosmo_cg_scalars.ptr + kCosmoCgS1, dev.cosmo_scalars.ptr,
      dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident COSMO energy finalize kernels")) {
    return false;
  }
  mozyme_cosmo_cg_finalize_status_kernel<<<1, 1>>>(
      dev.cosmo_scalars.ptr, dev.cosmo_cg_scalars.ptr, dev.cosmo_cg_ints.ptr,
      dev.cosmo_status_scalars.ptr, dev.cosmo_status_ints.ptr,
      dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident COSMO CG finalize status kernel")) {
    return false;
  }
  if (!strict_resident_request_enabled()) {
    if (!cuda_context_ok(cudaDeviceSynchronize(),
                         "resident COSMO synchronize")) {
      return false;
    }
  }
  return true;
}

bool compute_fock_on_gpu_impl(MozymeScfContext &ctx, int plan_id, int mode,
                              const double *density_p, const double *base_f,
                              double *output_f, double *wall_ms) {
  if (!wall_ms || !ctx.device.uploaded || !ctx.state.use_nijbo) {
    return false;
  }

  const int numat = ctx.config.natoms;
  const int mpack = ctx.config.mpack;
  if (numat <= 0 || mpack <= 0 || !density_p || !base_f || !output_f ||
      (mode != -1 && mode != 0 && mode != 1)) {
    return false;
  }

  auto &dev = ctx.device;
  bool ok = false;
  do {
    const bool time_stage = resident_stage_timing_enabled(ctx, wall_ms);
    constexpr int kThreads = 256;
    const int matrix_blocks = ceil_div(mpack, kThreads);
    const int atom_blocks = ceil_div(numat, kThreads);
    if (!device_buffer_ready(dev.qe, static_cast<std::size_t>(numat))) break;
    if (!device_buffer_ready(dev.fock_ints, kFockIntCount)) break;
    if (!zero_ints_if_resident_active(ctx, dev.fock_ints.ptr, kFockIntCount,
                                      "resident fock status reset")) {
      break;
    }

    if (!begin_resident_stage_timing(ctx, time_stage,
                                     "resident fock start event")) break;
    mozyme_chrge_kernel<<<atom_blocks, kThreads>>>(
        numat, mpack, dev.iorbs.ptr, dev.nijbo.ptr, density_p, dev.qe.ptr,
        dev.resident_control_ints.ptr);
    mozyme_fock_init_kernel<<<matrix_blocks, kThreads>>>(
        mpack, mode, dev.h.ptr, base_f, output_f,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident fock charge/init kernels")) break;
    if (mode == -1) {
      mozyme_fock_negate_kernel<<<matrix_blocks, kThreads>>>(
          mpack, output_f, dev.resident_control_ints.ptr);
      if (!cuda_context_ok(cudaGetLastError(),
                           "resident fock pre-negate kernel")) break;
    }
    if (time_stage &&
        !cuda_context_ok(cudaDeviceSynchronize(),
                         "resident fock pre-sparse synchronize")) break;

    double sparse_ms = 0.0;
    const bool strict_resident = strict_resident_request_enabled();
    const int synchronize_sparse_fock = strict_resident ? 0 : 1;
    if (strict_resident && synchronize_sparse_fock != 0) {
      ++ctx.strict_resident_host_syncs;
    }
    const int code = time_stage
        ? mopac_cuda_mozyme_sparse_fock_run_device_plan_guarded(
              plan_id, mpack, density_p, dev.qe.ptr, output_f,
              dev.resident_control_ints.ptr, kResidentControlDecision,
              kResidentDecisionContinue, &sparse_ms)
        : mopac_cuda_mozyme_sparse_fock_run_device_plan_guarded_resident(
              plan_id, mpack, density_p, dev.qe.ptr, output_f,
              dev.resident_control_ints.ptr, kResidentControlDecision,
              kResidentDecisionContinue, synchronize_sparse_fock, nullptr);
    if (code != kMozymeScfSuccess) break;
    if (mode == -1) {
      mozyme_fock_negate_kernel<<<matrix_blocks, kThreads>>>(
          mpack, output_f, dev.resident_control_ints.ptr);
      if (!cuda_context_ok(cudaGetLastError(),
                           "resident fock post-negate kernel")) break;
    }
    if (mode == 0 && !apply_cosmo_fock_on_gpu(ctx, density_p, output_f)) {
      break;
    }

    if (!finish_resident_stage_timing(
            ctx, time_stage, "resident fock stop event",
            "resident fock kernels", "resident fock synchronize",
            "resident fock elapsed time", wall_ms)) break;
    if (sparse_ms > 0.0) *wall_ms = sparse_ms;
    if (ctx.state.cosmo_enabled && mode == 0) {
      mozyme_cosmo_cg_mark_fock_status_kernel<<<1, 1>>>(
          dev.fock_ints.ptr, dev.cosmo_status_ints.ptr,
          dev.resident_control_ints.ptr);
    } else {
      mozyme_set_int_slot_if_resident_active_kernel<<<1, 1>>>(
          dev.fock_ints.ptr, kFockIntOk, 1, dev.resident_control_ints.ptr);
    }
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident fock status kernel")) break;
    ok = true;
  } while (false);

  return ok;
}

bool compute_fock_on_gpu(MozymeScfContext &ctx, double *wall_ms) {
  const int expected_plan_id = (ctx.config.fock_mode == 0)
                                   ? kMozymeFockPlanFull
                                   : kMozymeFockPlanPartial;
  if (ctx.config.resident_fock_plan_id != expected_plan_id) return false;
  return compute_fock_on_gpu_impl(ctx, ctx.config.resident_fock_plan_id,
                                  ctx.config.fock_mode,
                                  ctx.device.p.ptr, ctx.device.partf.ptr,
                                  ctx.device.f.ptr, wall_ms);
}

bool compute_helecz_on_gpu(MozymeScfContext &ctx, double *energy,
                           double *wall_ms) {
  const bool publish_energy = energy != nullptr;
  if (!wall_ms || !ctx.device.uploaded || !ctx.state.use_nijbo) {
    return false;
  }

  const int numat = ctx.config.natoms;
  const int mpack = ctx.config.mpack;
  if (numat <= 0 || mpack <= 0) return false;
  bool ok = false;
  auto &dev = ctx.device;

  do {
    const bool time_stage =
        resident_stage_timing_enabled(ctx, wall_ms) || publish_energy;
    constexpr int kThreads = 128;
    if (!device_buffer_ready(dev.helecz_ints, kHeleczIntCount)) break;
    if (!zero_ints_if_resident_active(ctx, dev.helecz_ints.ptr,
                                      kHeleczIntCount,
                                      "resident helecz status reset")) {
      break;
    }
    mozyme_set_int_slot_if_resident_active_kernel<<<1, 1>>>(
        dev.helecz_ints.ptr, kHeleczIntOk, 1,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident helecz status init kernel")) break;
    if (!begin_resident_stage_timing(ctx, time_stage,
                                     "resident helecz start event")) break;
    mozyme_helecz_kernel<<<numat, kThreads>>>(
        numat, mpack, dev.iorbs.ptr, dev.nijbo.ptr, dev.p.ptr, dev.h.ptr,
        dev.f.ptr, dev.atom_sums.ptr, dev.atom_diag_sums.ptr,
        dev.helecz_ints.ptr + kHeleczIntOk, dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident helecz atom kernel")) break;
    mozyme_helecz_reduce_kernel<<<1, kThreads, 2 * kThreads * sizeof(double)>>>(
        numat, dev.atom_sums.ptr, dev.atom_diag_sums.ptr,
        dev.helecz_ints.ptr, dev.energy_sums.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident helecz reduce kernel")) break;
    if (!finish_resident_stage_timing(
            ctx, time_stage, "resident helecz stop event",
            "resident helecz kernels", "resident helecz synchronize",
            "resident helecz elapsed time", wall_ms)) break;
    if (publish_energy) {
      int host_ok = 0;
      if (!cuda_context_ok(
              cudaMemcpy(&host_ok, dev.helecz_ints.ptr, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "resident helecz validity copy")) {
        break;
      }
      if (host_ok != 1) break;
      double host_totals[3] = {0.0, 0.0, 0.0};
      if (!cuda_context_ok(
              cudaMemcpy(host_totals, dev.energy_sums.ptr, sizeof(host_totals),
                         cudaMemcpyDeviceToHost),
              "resident helecz totals copy")) {
        break;
      }
      *energy = host_totals[2];
    }
    ok = true;
  } while (false);

  return ok;
}

bool set_resident_control_decision(MozymeScfContext &ctx, int decision) {
  auto &dev = ctx.device;
  if (!dev.resident_control_ints.ptr ||
      dev.resident_control_ints.count < kResidentControlIntCount) {
    return false;
  }
  mozyme_resident_control_decision_kernel<<<1, 1>>>(
      dev.resident_control_ints.ptr, decision);
  return cuda_context_ok(cudaGetLastError(),
                         "resident control decision kernel");
}

bool publish_cosmo_status_from_gpu(MozymeScfContext &ctx,
                                   MozymeScfStatus *status);

bool validate_final_reorth_rebuild_from_gpu(MozymeScfContext &ctx) {
  auto &dev = ctx.device;
  int density_status[3] = {};
  int fock_status[kFockIntCount] = {};
  int helecz_status[kHeleczIntCount] = {};
  if (!copy_device_to_host_raw(density_status, dev.density_updates, 3,
                               "resident final reorth density status copy")) {
    return false;
  }
  if (!copy_device_to_host_raw(fock_status, dev.fock_ints, kFockIntCount,
                               "resident final reorth fock status copy")) {
    return false;
  }
  if (!copy_device_to_host_raw(helecz_status, dev.helecz_ints,
                               kHeleczIntCount,
                               "resident final reorth helecz status copy")) {
    return false;
  }
  return density_status[1] == 1 && density_status[0] == density_status[2] &&
         density_status[2] > 0 && fock_status[kFockIntOk] == 1 &&
         helecz_status[kHeleczIntOk] == 1;
}

bool apply_final_reorth_on_gpu(MozymeScfContext &ctx, MozymeScfStatus *status,
                               double *accumulated_ms) {
  if (!status || !accumulated_ms || !ctx.device.uploaded) return false;
  const int total_occ = ctx.config.total_occupied;
  const int total_vir = ctx.config.total_virtual;
  const int natoms = ctx.config.natoms;
  const int norbs = ctx.config.norbs;
  if (total_occ < 0 || total_vir < 0 || natoms <= 0 || norbs <= 0 ||
      ctx.state.cocc_dim <= 0 || ctx.state.icocc_dim <= 0 ||
      ctx.state.cvir_dim <= 0 || ctx.state.icvir_dim <= 0 ||
      ctx.state.nocc_slots < total_occ ||
      ctx.state.nvir_slots < total_vir) {
    return false;
  }

  auto &dev = ctx.device;
  bool ok = false;
  bool device_final_reorth_committed = false;
  do {
    if (!set_resident_control_decision(ctx, kResidentDecisionContinue)) break;
    if (!dev.final_reorth_ws.ptr ||
        dev.final_reorth_ws.count < static_cast<std::size_t>(norbs)) {
      break;
    }
    if (!dev.final_reorth_latom.ptr ||
        dev.final_reorth_latom.count < static_cast<std::size_t>(natoms)) {
      break;
    }
    if (!dev.final_reorth_iused.ptr ||
        dev.final_reorth_iused.count < static_cast<std::size_t>(natoms)) {
      break;
    }
    if (!dev.final_reorth_status.ptr || dev.final_reorth_status.count < 1) {
      break;
    }
    if (!dev.final_reorth_sumtot.ptr || dev.final_reorth_sumtot.count < 1) {
      break;
    }
    if (!dev.final_reorth_status_scalars.ptr ||
        dev.final_reorth_status_scalars.count <
            kFinalReorthStatusDoubleCount) {
      break;
    }
    if (!zero_ints_if_resident_active(ctx, dev.final_reorth_status.ptr, 1,
                                      "resident final reorth status reset")) {
      break;
    }
    if (!zero_doubles_if_resident_active(ctx, dev.final_reorth_sumtot.ptr, 1,
                                         "resident final reorth sum reset")) {
      break;
    }
    if (!dev.ensure_events()) break;

    if (!cuda_context_ok(cudaEventRecord(dev.start),
                         "resident final reorth start event")) break;
    ::mozyme_reorth_kernel<<<1, 1>>>(
        natoms, norbs, total_occ, total_vir, ctx.state.cocc_dim,
        ctx.state.icocc_dim, ctx.state.cvir_dim, ctx.state.icvir_dim,
        ctx.config.thresh, dev.cocc.ptr, dev.icocc.ptr, dev.ncf.ptr,
        dev.nncf.ptr, dev.ncocc.ptr, dev.cvir.ptr, dev.icvir.ptr, dev.nce.ptr,
        dev.nnce.ptr, dev.ncvir.ptr, dev.iorbs.ptr, dev.nfirst.ptr,
        dev.final_reorth_ws.ptr, dev.final_reorth_latom.ptr,
        dev.final_reorth_iused.ptr, dev.final_reorth_status.ptr,
        dev.final_reorth_sumtot.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident final reorth kernel")) break;
    if (!cuda_context_ok(cudaEventRecord(dev.stop),
                         "resident final reorth stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(dev.stop),
                         "resident final reorth synchronize")) break;

    int host_status = kMozymeScfNotReady;
    if (!cuda_context_ok(
            cudaMemcpy(&host_status, dev.final_reorth_status.ptr, sizeof(int),
                       cudaMemcpyDeviceToHost),
            "resident final reorth status copy")) break;
    if (host_status != 0) break;

    double host_sumtot = 0.0;
    if (!cuda_context_ok(
            cudaMemcpy(&host_sumtot, dev.final_reorth_sumtot.ptr, sizeof(double),
                       cudaMemcpyDeviceToHost),
            "resident final reorth sum copy")) break;

    float reorth_elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&reorth_elapsed, dev.start,
                                              dev.stop),
                         "resident final reorth elapsed time")) break;

    int density_terms = 0;
    double density_ms = 0.0;
    double fock_ms = 0.0;
    double helecz_ms = 0.0;
    if (!compute_density_on_gpu_impl(ctx, total_occ, 0, dev.partp.ptr,
                                     dev.p.ptr, &density_terms,
                                     &density_ms)) {
      break;
    }
    if (!compute_fock_on_gpu_impl(ctx, kMozymeFockPlanFull, 0, dev.p.ptr,
                                  dev.partf.ptr, dev.f.ptr, &fock_ms)) {
      break;
    }
    if (!compute_helecz_on_gpu(ctx, nullptr, &helecz_ms)) break;
    if (!validate_final_reorth_rebuild_from_gpu(ctx)) break;
    device_final_reorth_committed = true;
    if (!publish_cosmo_status_from_gpu(ctx, status)) break;

    if (!zero_doubles_if_resident_active(
            ctx, dev.final_reorth_status_scalars.ptr,
            kFinalReorthStatusDoubleCount,
            "resident final status scalar reset")) {
      break;
    }
    if (!zero_ints_if_resident_active(ctx, dev.final_reorth_status.ptr, 1,
                                      "resident final status ok reset")) {
      break;
    }
    mozyme_final_reorth_status_kernel<<<1, 1>>>(
        dev.energy_sums.ptr, 2, ctx.config.energy_scale,
        ctx.config.energy_offset,
        ctx.state.cosmo_enabled ? dev.cosmo_scalars.ptr : nullptr,
        ctx.state.cosmo_enabled ? 1 : 0, ctx.state.cosmo_solv_energy,
        status->energy_scf, status->energy_delta,
        dev.final_reorth_status_scalars.ptr, dev.final_reorth_status.ptr,
        dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident final status kernel")) break;
    int final_status_ok = 0;
    if (!cuda_context_ok(
            cudaMemcpy(&final_status_ok, dev.final_reorth_status.ptr,
                       sizeof(int), cudaMemcpyDeviceToHost),
            "resident final status ok copy")) break;
    if (final_status_ok != 1) break;
    double final_status_scalars[kFinalReorthStatusDoubleCount] = {};
    if (!copy_device_to_host_raw(final_status_scalars,
                                 dev.final_reorth_status_scalars,
                                 kFinalReorthStatusDoubleCount,
                                 "resident final status scalar copy")) {
      break;
    }
    status->energy_total =
        final_status_scalars[kFinalReorthStatusEnergyTotal];
    status->energy_scf = final_status_scalars[kFinalReorthStatusEnergyScf];
    status->energy_delta =
        final_status_scalars[kFinalReorthStatusEnergyDelta];
    status->final_reorth_applied = 1;
    status->final_reorth_sum = host_sumtot;
    status->final_reorth_ms =
        static_cast<double>(reorth_elapsed) + density_ms + fock_ms + helecz_ms;
    *accumulated_ms += status->final_reorth_ms;
    status->wall_ms = *accumulated_ms;
    ok = true;
  } while (false);

  const int final_decision =
      (ok || device_final_reorth_committed) ? kResidentDecisionComplete
                                            : kResidentDecisionCpuBoundary;
  if (!set_resident_control_decision(ctx, final_decision)) {
    return false;
  }
  return ok;
}

bool compute_initial_setup_on_gpu(MozymeScfContext &ctx, double *energy,
                                  double *wall_ms) {
  if (!wall_ms || !ctx.device.uploaded || !ctx.state.use_nijbo) {
    return false;
  }

  const int total_occ = ctx.config.total_occupied;
  const int mpack = ctx.config.mpack;
  if (total_occ <= 0 || mpack <= 0) {
    return false;
  }

  int updated_terms = 0;
  double density_ms = 0.0;
  double density_partial_ms = 0.0;
  double fock_ms = 0.0;
  double fock_partial_ms = 0.0;
  double helecz_ms = 0.0;
  double total_ms = 0.0;
  bool ok = false;
  auto &dev = ctx.device;

  do {
    if (!compute_density_on_gpu_impl(ctx, total_occ, 0, dev.partp.ptr,
                                     dev.p.ptr, &updated_terms,
                                     &density_ms)) {
      break;
    }
    total_ms += density_ms;

    if (ctx.config.fock_mode != 0) {
      if (!compute_density_on_gpu_impl(ctx, ctx.config.noccupied, -1,
                                       dev.p.ptr, dev.partp.ptr,
                                       &updated_terms,
                                       &density_partial_ms)) {
        break;
      }
      total_ms += density_partial_ms;
    }

    if (!compute_fock_on_gpu_impl(ctx, kMozymeFockPlanFull, 0,
                                  dev.p.ptr, dev.partf.ptr, dev.f.ptr,
                                  &fock_ms)) {
      break;
    }
    total_ms += fock_ms;

    if (!compute_helecz_on_gpu(ctx, energy, &helecz_ms)) break;
    total_ms += helecz_ms;

    if (ctx.config.fock_mode != 0) {
      if (!compute_fock_on_gpu_impl(ctx, kMozymeFockPlanPartial,
                                    -1, dev.partp.ptr, dev.f.ptr,
                                    dev.partf.ptr, &fock_partial_ms)) {
        break;
      }
      total_ms += fock_partial_ms;
    }

    *wall_ms = total_ms;
    ok = true;
  } while (false);

  return ok;
}

bool compute_isitsc_on_gpu(MozymeScfContext &ctx, double *wall_ms) {
  if (!wall_ms || !ctx.device.uploaded ||
      ctx.config.selcon < 0.0 || ctx.config.max_iter < 0 ||
      !ctx.device.diagg_scalars.ptr) {
    return false;
  }

  auto &dev = ctx.device;

  bool ok = false;
  do {
    const bool time_stage = resident_stage_timing_enabled(ctx, wall_ms);
    if (!dev.isitsc_ints.ptr || dev.isitsc_ints.count < kIsitscIntCount) {
      break;
    }
    if (!zero_ints_if_resident_active(
            ctx, dev.isitsc_ints.ptr + kIsitscIntOkscf,
            kIsitscIntCount - kIsitscIntOkscf,
            "resident isitsc control reset")) break;
    if (!device_buffer_ready(dev.isitsc_scalars, kIsitscDoubleCount)) break;
    if (!dev.isitsc_escf0.ptr || dev.isitsc_escf0.count < 10) break;

    if (!begin_resident_stage_timing(ctx, time_stage,
                                     "resident isitsc start event")) break;
    mozyme_isitsc_resident_kernel<<<1, 1>>>(
        dev.energy_sums.ptr, 2, ctx.config.energy_scale,
        ctx.config.energy_offset, dev.cosmo_scalars.ptr,
        ctx.state.cosmo_enabled, ctx.state.cosmo_solv_energy,
        ctx.config.previous_escf, ctx.config.selcon, ctx.config.emin,
        dev.diagg_scalars.ptr, kDiaggDoubleTiny, ctx.config.current_iter,
        ctx.config.max_iter,
        dev.isitsc_ints.ptr + kIsitscIntIemin,
        dev.isitsc_ints.ptr + kIsitscIntIemax,
        dev.isitsc_ints.ptr + kIsitscIntScf1, dev.isitsc_escf0.ptr,
        dev.isitsc_scalars.ptr, dev.isitsc_ints.ptr + kIsitscIntOkscf,
        dev.isitsc_ints.ptr + kIsitscIntIscf,
        dev.isitsc_ints.ptr + kIsitscIntValid, dev.resident_control_ints.ptr,
        dev.resident_control_scalars.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident isitsc kernel")) break;
    if (!finish_resident_stage_timing(
            ctx, time_stage, "resident isitsc stop event",
            "resident isitsc kernels", "resident isitsc synchronize",
            "resident isitsc elapsed time", wall_ms)) break;
    ok = true;
  } while (false);

  return ok;
}

bool compute_cnvgz_probe_on_gpu(MozymeScfContext &ctx,
                                double *wall_ms,
                                bool update_device_state) {
  if (!wall_ms || !ctx.device.uploaded) {
    return false;
  }

  const int norbs = ctx.config.norbs;
  const int mpack = ctx.config.mpack;
  if (norbs <= 0 || mpack <= 0) return false;

  auto &dev = ctx.device;
  if (!dev.resident_control_ints.ptr ||
      dev.resident_control_ints.count < kResidentControlIntCount) {
    return false;
  }
  if (!device_buffer_ready(dev.cnvgz_ints, kCnvgzIntCount)) {
    return false;
  }
  if (!zero_ints_if_resident_active(ctx, dev.cnvgz_ints.ptr + kCnvgzIntOk, 1,
                                    "resident cnvgz ok reset")) {
    return false;
  }
  if (!device_buffer_ready(dev.cnvgz_sums, kCnvgzControlCount)) return false;
  if (!zero_doubles_if_resident_active(ctx, dev.cnvgz_sums.ptr,
                                       kCnvgzControlCount,
                                       "resident cnvgz sums reset")) {
    return false;
  }

  bool ok = false;

  constexpr int kThreads = 256;
  const int matrix_blocks = ceil_div(mpack, kThreads);
  const int diag_blocks = ceil_div(norbs, kThreads);

  do {
    const bool time_stage = resident_stage_timing_enabled(ctx, wall_ms);
    if (!begin_resident_stage_timing(ctx, time_stage,
                                     "resident cnvgz start event")) break;
    mozyme_cnvgz_diag_kernel<<<diag_blocks, kThreads>>>(
        norbs, mpack, dev.idiag.ptr, dev.p.ptr, dev.pold.ptr, dev.diag_new.ptr,
        dev.diag_old.ptr, dev.resident_control_ints.ptr);
    mozyme_cnvgz_diff_kernel<<<matrix_blocks, kThreads,
                               2 * kThreads * sizeof(double)>>>(
        mpack, dev.p.ptr, dev.pold.ptr, dev.block_max.ptr,
        dev.block_sumsq.ptr, dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "resident cnvgz diff kernels")) break;
    mozyme_cnvgz_diff_reduce_kernel<<<1, 1>>>(
        matrix_blocks, dev.block_max.ptr, dev.block_sumsq.ptr,
        dev.cnvgz_sums.ptr, dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "resident cnvgz reduce kernel")) break;

    const bool fallback_compute_factor = false;
    const int fallback_niter = 0;
    const int fallback_use_three_point = 0;
    mozyme_cnvgz_factor_kernel<<<diag_blocks, kThreads,
                                 2 * kThreads * sizeof(double)>>>(
        norbs, dev.diag_old.ptr, dev.p1.ptr, dev.diag_new.ptr,
        dev.block_faca.ptr, dev.block_facb.ptr, dev.resident_control_ints.ptr,
        fallback_use_three_point, fallback_niter);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident cnvgz factor kernel")) break;
    mozyme_cnvgz_factor_reduce_kernel<<<1, 1>>>(
        diag_blocks, dev.block_faca.ptr, dev.block_facb.ptr,
        dev.cnvgz_sums.ptr, dev.resident_control_ints.ptr);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident cnvgz factor reduce kernel")) break;
    mozyme_cnvgz_finalize_kernel<<<1, 1>>>(
        mpack, fallback_compute_factor, dev.cnvgz_sums.ptr,
        dev.cnvgz_ints.ptr, dev.resident_control_ints.ptr,
        fallback_use_three_point, fallback_niter);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident cnvgz finalize kernel")) break;

    mozyme_cnvgz_candidate_kernel<<<matrix_blocks, kThreads>>>(
        mpack, dev.cnvgz_sums.ptr, dev.p.ptr, dev.pold.ptr,
        dev.candidate.ptr, dev.resident_control_ints.ptr,
        fallback_use_three_point);
    mozyme_cnvgz_damp_kernel<<<diag_blocks, kThreads>>>(
        norbs, mpack, dev.cnvgz_sums.ptr, dev.idiag.ptr, dev.diag_new.ptr,
        dev.diag_old.ptr, dev.candidate.ptr, dev.resident_control_ints.ptr,
        fallback_use_three_point, fallback_niter);
    if (!cuda_context_ok(cudaGetLastError(),
                         "resident cnvgz candidate kernels")) break;
    if (update_device_state) {
      mozyme_cnvgz_commit_matrix_kernel<<<matrix_blocks, kThreads>>>(
          mpack, dev.resident_control_ints.ptr, dev.candidate.ptr, dev.p.ptr,
          dev.pold.ptr);
      mozyme_cnvgz_commit_diag_kernel<<<diag_blocks, kThreads>>>(
          norbs, dev.resident_control_ints.ptr, dev.diag_old.ptr,
          dev.diag_new.ptr, dev.p1.ptr, dev.p2.ptr, dev.p3.ptr);
      if (!cuda_context_ok(cudaGetLastError(),
                           "resident cnvgz commit kernels")) break;
    }
    if (!finish_resident_stage_timing(
            ctx, time_stage, "resident cnvgz stop event",
            "resident cnvgz kernels", "resident cnvgz synchronize",
            "resident cnvgz elapsed time", wall_ms)) break;
    ok = true;
  } while (false);

  return ok;
}

void set_status_loop_control(MozymeScfStatus *status,
                             const MozymeScfContext &ctx) {
  status->use_three_point = ctx.config.use_three_point;
  status->lstart = ctx.config.lstart;
  status->shift = ctx.config.shift;
}

void begin_resident_iteration_status(MozymeScfStatus *status,
                                     const MozymeScfContext &ctx,
                                     double accumulated_ms) {
  int previous_calls[kResidentStageSlotCount] = {};
  double previous_ms[kResidentStageSlotCount] = {};
  for (int i = 0; i < kResidentStageSlotCount; ++i) {
    previous_calls[i] = status->resident_stage_calls[i];
    previous_ms[i] = status->resident_stage_ms[i];
  }
  fill_status(status, &ctx, kMozymeScfNotReady);
  for (int i = 0; i < kResidentStageSlotCount; ++i) {
    status->resident_stage_calls[i] = previous_calls[i];
    status->resident_stage_ms[i] = previous_ms[i];
  }
  status->resident = ctx.device.uploaded ? 1 : 0;
  status->iterations = ctx.config.current_iter;
  status->wall_ms = accumulated_ms;
  set_status_loop_control(status, ctx);
}

void add_stage_time(MozymeScfStatus *status, double *accumulated_ms,
                    double stage_ms, int stage_slot = -1) {
  if (stage_slot >= 0 && stage_slot < kResidentStageSlotCount) {
    if (stage_ms > 0.0) status->resident_stage_ms[stage_slot] += stage_ms;
  }
  if (stage_ms <= 0.0) return;
  *accumulated_ms += stage_ms;
  status->wall_ms = *accumulated_ms;
}

bool completed_full_stage_mask(int completed) {
  return completed == kMozymeScfStageFull;
}

int required_resident_stage_calls(const MozymeScfStatus &status,
                                  int previous_iter, int stage_slot) {
  if (stage_slot == kResidentStageSlotUpload) return 1;
  const int completed_iterations = status.iterations - previous_iter;
  return completed_iterations > 1 ? completed_iterations : 1;
}

bool resident_stage_calls_complete(const MozymeScfStatus &status,
                                   int previous_iter) {
  for (int i = 0; i < kResidentStageSlotCount; ++i) {
    if (status.resident_stage_calls[i] <
        required_resident_stage_calls(status, previous_iter, i)) {
      return false;
    }
  }
  return true;
}

bool resident_stage_status_complete(const MozymeScfStatus &status,
                                    int previous_iter) {
  return status.stage_required == kMozymeScfStageFull &&
         status.stage_missing == 0 &&
         completed_full_stage_mask(status.stage_completed) &&
         resident_stage_calls_complete(status, previous_iter) &&
         status.code == kMozymeScfSuccess;
}

void mark_final_publication_done(MozymeScfStatus *status,
                                 const ResidentFinalPublicationProof &proof) {
  if (!status) return;
  status->final_publication_done = 1;
  status->final_publication_arrays = proof.arrays;
  status->final_publication_bytes = proof.bytes;
  status->final_publication_cosmo = proof.cosmo;
}

int strict_resident_loop_limit(const MozymeScfConfig &config) {
  if (config.max_iter <= 0 || config.current_iter < 0 ||
      config.current_iter > config.max_iter) {
    return 0;
  }

  const long long remaining =
      static_cast<long long>(config.max_iter) -
      static_cast<long long>(config.current_iter);
  const long long restart_budget = static_cast<long long>(config.max_iter);
  const long long limit = (remaining > 0LL ? remaining : 0LL) + restart_budget;
  return limit > 2147483647LL ? 2147483647 : static_cast<int>(limit);
}

bool resident_stage_timing_enabled(const MozymeScfContext &, const double *wall_ms) {
  return wall_ms && !strict_resident_request_enabled();
}

bool resident_stage_profile_enabled() {
  static const bool enabled = env_enabled("MOPAC_MOZYME_SECTION_PROFILE") ||
                              env_enabled("MOPAC_MOZYME_PROFILE") ||
                              env_enabled("MOPAC_GPU_PROFILE");
  return enabled;
}

bool begin_resident_stage_timing(MozymeScfContext &ctx, bool time_stage,
                                 const char *label) {
  if (resident_stage_profile_enabled()) {
    auto &dev = ctx.device;
    if (dev.pending_stage_start) {
      cudaEventDestroy(dev.pending_stage_start);
      dev.pending_stage_start = nullptr;
    }
    if (cudaEventCreate(&dev.pending_stage_start) == cudaSuccess) {
      cudaEventRecord(dev.pending_stage_start);
    }
  }
  if (!time_stage) return true;
  if (!ctx.device.ensure_events()) return false;
  return cuda_context_ok(cudaEventRecord(ctx.device.start), label);
}

void record_resident_stage_profile(MozymeScfContext &ctx,
                                   const char *stop_label) {
  auto &dev = ctx.device;
  if (!dev.pending_stage_start) return;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&stop) != cudaSuccess) return;
  cudaEventRecord(stop);
  dev.stage_timings.push_back({stop_label, dev.pending_stage_start, stop});
  dev.pending_stage_start = nullptr;
}

void report_resident_stage_profile(MozymeScfContext &ctx) {
  auto &dev = ctx.device;
  if (dev.stage_timings.empty()) return;
  cudaDeviceSynchronize();

  struct Total {
    std::string name;
    long calls;
    double ms;
  };
  std::vector<Total> totals;
  for (const auto &entry : dev.stage_timings) {
    float elapsed = 0.0f;
    if (cudaEventElapsedTime(&elapsed, entry.start, entry.stop) != cudaSuccess) {
      continue;
    }
    std::string name = entry.label ? entry.label : "unknown";
    const std::string suffix = " stop event";
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      name.erase(name.size() - suffix.size());
    }
    if (name.rfind("resident ", 0) == 0) name.erase(0, 9);
    std::replace(name.begin(), name.end(), ' ', '_');

    auto it = std::find_if(totals.begin(), totals.end(),
                           [&](const Total &t) { return t.name == name; });
    if (it == totals.end()) {
      totals.push_back({name, 1L, static_cast<double>(elapsed)});
    } else {
      ++it->calls;
      it->ms += static_cast<double>(elapsed);
    }
  }
  for (const auto &t : totals) {
    std::printf("[PROFILE] MOZYME_RESIDENT_STAGE name=%s calls=%ld ms=%.3f\n",
                t.name.c_str(), t.calls, t.ms);
  }
  std::fflush(stdout);
  dev.cleanup_stage_timings();
}

bool finish_resident_stage_timing(MozymeScfContext &ctx, bool time_stage,
                                  const char *stop_label,
                                  const char *kernels_label,
                                  const char *sync_label,
                                  const char *elapsed_label,
                                  double *wall_ms) {
  if (resident_stage_profile_enabled()) {
    record_resident_stage_profile(ctx, stop_label);
  }
  if (!time_stage) {
    if (wall_ms) *wall_ms = 0.0;
    return cuda_context_ok(cudaGetLastError(), kernels_label);
  }
  if (!cuda_context_ok(cudaEventRecord(ctx.device.stop), stop_label)) {
    return false;
  }
  if (!cuda_context_ok(cudaGetLastError(), kernels_label)) {
    return false;
  }
  if (!cuda_context_ok(cudaEventSynchronize(ctx.device.stop), sync_label)) {
    return false;
  }

  float elapsed = 0.0f;
  if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, ctx.device.start,
                                            ctx.device.stop),
                       elapsed_label)) {
    return false;
  }
  if (wall_ms) *wall_ms = static_cast<double>(elapsed);
  return true;
}

bool zero_ints_if_resident_active(MozymeScfContext &ctx, int *values, int count,
                                  const char *label) {
  if (!values || count <= 0) return false;
  constexpr int kThreads = 128;
  mozyme_zero_ints_if_resident_active_kernel<<<ceil_div(count, kThreads),
                                                kThreads>>>(
      values, count, ctx.device.resident_control_ints.ptr);
  return cuda_context_ok(cudaGetLastError(), label);
}

bool zero_doubles_if_resident_active(MozymeScfContext &ctx, double *values,
                                     int count, const char *label) {
  if (!values || count <= 0) return false;
  constexpr int kThreads = 128;
  mozyme_zero_doubles_if_resident_active_kernel<<<ceil_div(count, kThreads),
                                                  kThreads>>>(
      values, count, ctx.device.resident_control_ints.ptr);
  return cuda_context_ok(cudaGetLastError(), label);
}

bool reset_resident_stage_status_on_gpu(MozymeScfContext &ctx) {
  auto &dev = ctx.device;
  if (!device_buffer_ready(dev.resident_stage_ints,
                           kResidentStageIntCount)) {
    return false;
  }
  if (!device_buffer_ready(dev.resident_stage_calls,
                           kResidentStageSlotCount)) {
    return false;
  }
  mozyme_resident_stage_reset_kernel<<<1, 1>>>(
      dev.resident_stage_ints.ptr, dev.resident_stage_calls.ptr,
      dev.resident_control_ints.ptr);
  return cuda_context_ok(cudaGetLastError(), "resident stage reset kernel");
}

bool mark_resident_stage_if_int_on_gpu(MozymeScfContext &ctx, int stage_bits,
                                       const int *values, int value_slot,
                                       int expected_value, int stage_slot) {
  auto &dev = ctx.device;
  if (!dev.resident_stage_ints.ptr ||
      dev.resident_stage_ints.count < kResidentStageIntCount ||
      !dev.resident_stage_calls.ptr ||
      dev.resident_stage_calls.count < kResidentStageSlotCount ||
      !values || value_slot < 0) {
    return false;
  }
  mozyme_resident_stage_mark_if_int_kernel<<<1, 1>>>(
      dev.resident_stage_ints.ptr, dev.resident_stage_calls.ptr, stage_bits,
      values, value_slot, expected_value, stage_slot,
      dev.resident_control_ints.ptr);
  return cuda_context_ok(cudaGetLastError(),
                         "resident conditional stage mark kernel");
}

bool copy_resident_stage_ints_from_gpu(MozymeScfContext &ctx,
                                       int *host_stage) {
  auto &dev = ctx.device;
  if (!host_stage || !dev.resident_stage_ints.ptr ||
      dev.resident_stage_ints.count < kResidentStageIntCount) {
    return false;
  }
  return cuda_context_ok(
      cudaMemcpy(host_stage, dev.resident_stage_ints.ptr,
                 static_cast<std::size_t>(kResidentStageIntCount) * sizeof(int),
                 cudaMemcpyDeviceToHost),
      "resident stage status copy");
}

bool copy_resident_stage_calls_from_gpu(MozymeScfContext &ctx,
                                        int *host_calls) {
  auto &dev = ctx.device;
  if (!host_calls || !dev.resident_stage_calls.ptr ||
      dev.resident_stage_calls.count < kResidentStageSlotCount) {
    return false;
  }
  return cuda_context_ok(
      cudaMemcpy(host_calls, dev.resident_stage_calls.ptr,
                 static_cast<std::size_t>(kResidentStageSlotCount) *
                     sizeof(int),
                 cudaMemcpyDeviceToHost),
      "resident stage device counters copy");
}

int resident_stage_device_call_count(const int *host_calls, int stage_slot) {
  if (!host_calls || stage_slot < 0 || stage_slot >= kResidentStageSlotCount) {
    return 0;
  }
  return host_calls[stage_slot];
}

void publish_resident_stage_status_from_host(MozymeScfContext &ctx,
                                             MozymeScfStatus *status,
                                             const int *host_stage,
                                             const int *host_calls) {
  status->stage_completed = host_stage[kResidentStageCompleted];
  status->stage_required = host_stage[kResidentStageRequired];
  status->stage_missing = host_stage[kResidentStageMissing];
  status->code = host_stage[kResidentStageCode];
  if (host_calls) {
    for (int i = 0; i < kResidentStageSlotCount; ++i) {
      status->resident_stage_calls[i] = host_calls[i];
    }
  }
  publish_pls_runtime_status(status, &ctx);
  publish_cosmo_status(status, &ctx);
}

bool publish_resident_stage_status_from_gpu(MozymeScfContext &ctx,
                                            MozymeScfStatus *status) {
  if (!status) return false;
  int host_stage[kResidentStageIntCount] = {};
  int host_calls[kResidentStageSlotCount] = {};
  if (!copy_resident_stage_ints_from_gpu(ctx, host_stage)) return false;
  if (!copy_resident_stage_calls_from_gpu(ctx, host_calls)) return false;
  publish_resident_stage_status_from_host(ctx, status, host_stage, host_calls);
  return true;
}

bool resident_stage_device_confirmed(MozymeScfContext &ctx, int stage_bits,
                                     int stage_slot, int *completed_out,
                                     MozymeScfStatus *status = nullptr) {
  int host_stage[kResidentStageIntCount] = {};
  int host_calls[kResidentStageSlotCount] = {};
  if (!copy_resident_stage_ints_from_gpu(ctx, host_stage)) return false;
  if (!copy_resident_stage_calls_from_gpu(ctx, host_calls)) return false;
  if ((host_stage[kResidentStageCompleted] & stage_bits) != stage_bits) {
    return false;
  }
  if (resident_stage_device_call_count(host_calls, stage_slot) <= 0) {
    return false;
  }
  if (completed_out) *completed_out = host_stage[kResidentStageCompleted];
  if (status) {
    publish_resident_stage_status_from_host(ctx, status, host_stage,
                                            host_calls);
  }
  return true;
}

bool publish_cosmo_status_from_gpu(MozymeScfContext &ctx,
                                   MozymeScfStatus *status) {
  if (!status) return false;
  publish_cosmo_status(status, &ctx);
  if (!ctx.state.cosmo_enabled) return true;

  auto &dev = ctx.device;
  if (!dev.cosmo_status_ints.ptr ||
      dev.cosmo_status_ints.count < kCosmoStatusIntCount ||
      !dev.cosmo_status_scalars.ptr ||
      dev.cosmo_status_scalars.count < kCosmoStatusDoubleCount) {
    return false;
  }

  int host_ints[kCosmoStatusIntCount] = {};
  double host_scalars[kCosmoStatusDoubleCount] = {};
  if (!cuda_context_ok(
          cudaMemcpy(host_ints, dev.cosmo_status_ints.ptr,
                     sizeof(host_ints), cudaMemcpyDeviceToHost),
          "resident final COSMO CG integer status copy")) {
    return false;
  }
  if (!cuda_context_ok(
          cudaMemcpy(host_scalars, dev.cosmo_status_scalars.ptr,
                     sizeof(host_scalars), cudaMemcpyDeviceToHost),
          "resident final COSMO CG scalar status copy")) {
    return false;
  }

  status->cosmo_fock_calls = host_ints[kCosmoStatusFockCalls];
  status->cosmo_matvec_calls = host_ints[kCosmoStatusMatvecCalls];
  status->cosmo_cg_iterations = host_ints[kCosmoStatusCgIterations];
  status->cosmo_cg_control_resident =
      host_ints[kCosmoStatusControlResident];
  status->cosmo_cg_converged = host_ints[kCosmoStatusCgConverged];
  status->cosmo_cg_breakdown = host_ints[kCosmoStatusCgBreakdown];
  status->cosmo_cg_host_syncs = host_ints[kCosmoStatusHostSyncs];
  status->cosmo_cg_target_tol = host_scalars[kCosmoStatusTargetTol];
  status->cosmo_last_residual = host_scalars[kCosmoStatusLastResidual];
  status->cosmo_solv_energy = host_scalars[kCosmoStatusSolvEnergy];
  status->cosmo_ediel = host_scalars[kCosmoStatusEdiel];
  return true;
}

void ensure_cnvgz_activity_from_resident_stage_calls(MozymeScfStatus *status) {
  if (!status) return;
  if (status->cnvgz_active_calls + status->cnvgz_noop_calls > 0) return;
  const int cnvgz_calls =
      status->resident_stage_calls[kResidentStageSlotCnvgz];
  if (cnvgz_calls > 0) status->cnvgz_noop_calls = cnvgz_calls;
}

bool publish_resident_iteration_outputs_from_gpu(MozymeScfContext &ctx,
                                                 MozymeScfStatus *status) {
  auto &dev = ctx.device;
  if (!status || !dev.diagg_ints.ptr || !dev.diagg_scalars.ptr ||
      !dev.addhb_ints.ptr || !dev.addhb_scalars.ptr ||
      !dev.cnvgz_sums.ptr || !dev.cnvgz_ints.ptr || !dev.helecz_ints.ptr ||
      !dev.energy_sums.ptr || !dev.isitsc_ints.ptr ||
      !dev.isitsc_scalars.ptr || !dev.isitsc_escf0.ptr) {
    return false;
  }

  int diagg_ints[kDiaggIntCount] = {};
  int addhb_ints[kAddhbIntCount] = {};
  int cnvgz_ints[kCnvgzIntCount] = {};
  int helecz_ints[kHeleczIntCount] = {};
  int isitsc_ints[kIsitscIntCount] = {};
  double diagg_scalars[kDiaggDoubleCount] = {};
  double addhb_scalars[kAddhbDoubleCount] = {};
  double cnvgz_scalars[kCnvgzControlCount] = {};
  double energy_totals[3] = {0.0, 0.0, 0.0};
  double isitsc_scalars[kIsitscDoubleCount] = {};

  if (!cuda_context_ok(
          cudaMemcpy(diagg_ints, dev.diagg_ints.ptr, sizeof(diagg_ints),
                     cudaMemcpyDeviceToHost),
          "resident final diagg integer copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(diagg_scalars, dev.diagg_scalars.ptr,
                     sizeof(diagg_scalars), cudaMemcpyDeviceToHost),
          "resident final diagg scalar copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(addhb_ints, dev.addhb_ints.ptr, sizeof(addhb_ints),
                     cudaMemcpyDeviceToHost),
          "resident final addhb integer copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(addhb_scalars, dev.addhb_scalars.ptr,
                     sizeof(addhb_scalars), cudaMemcpyDeviceToHost),
          "resident final addhb scalar copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(cnvgz_ints, dev.cnvgz_ints.ptr, sizeof(cnvgz_ints),
                     cudaMemcpyDeviceToHost),
          "resident final cnvgz integer copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(cnvgz_scalars, dev.cnvgz_sums.ptr,
                     sizeof(cnvgz_scalars), cudaMemcpyDeviceToHost),
          "resident final cnvgz scalar copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(helecz_ints, dev.helecz_ints.ptr, sizeof(helecz_ints),
                     cudaMemcpyDeviceToHost),
          "resident final helecz integer copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(energy_totals, dev.energy_sums.ptr, sizeof(energy_totals),
                     cudaMemcpyDeviceToHost),
          "resident final energy totals copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(isitsc_ints, dev.isitsc_ints.ptr, sizeof(isitsc_ints),
                     cudaMemcpyDeviceToHost),
          "resident final isitsc integer copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(isitsc_scalars, dev.isitsc_scalars.ptr,
                     sizeof(isitsc_scalars), cudaMemcpyDeviceToHost),
          "resident final isitsc scalar copy")) return false;
  if (!cuda_context_ok(
          cudaMemcpy(status->isitsc_escf0, dev.isitsc_escf0.ptr,
                     10 * sizeof(double), cudaMemcpyDeviceToHost),
          "resident final isitsc history copy")) return false;

  if (diagg_ints[kDiaggIntOk] != 1 || addhb_ints[kAddhbIntOk] != 1 ||
      cnvgz_ints[kCnvgzIntOk] != 1 ||
      helecz_ints[kHeleczIntOk] != 1 ||
      isitsc_ints[kIsitscIntValid] != 1) {
    return false;
  }

  status->diagg_nij = diagg_ints[kDiaggIntNij];
  status->diagg_nf = diagg_ints[kDiaggIntNf];
  status->diagg_tiny = diagg_scalars[kDiaggDoubleTiny];
  status->next_tiny = addhb_scalars[kAddhbDoubleNextTiny];
  status->diagg_fref = diagg_scalars[kDiaggDoubleFref];
  status->diagg_oldlim = diagg_scalars[kDiaggDoubleOldlim];
  status->diagg_safety = diagg_scalars[kDiaggDoubleSafety];
  status->diagg_sumt = diagg_scalars[kDiaggDoubleSumt];
  status->diagg_sumb =
      (addhb_ints[kAddhbIntNij] > 0) ? addhb_scalars[kAddhbDoubleSumb]
                                     : diagg_scalars[kDiaggDoubleSumb];
  status->diagg2_nrejct[0] = addhb_ints[kAddhbIntNextNrej0];
  status->diagg2_nrejct[1] = addhb_ints[kAddhbIntNextNrej1];
  status->idiagg = addhb_ints[kAddhbIntNextIdiagg];
  status->nhb = addhb_ints[kAddhbIntNextNhb];
  status->addhb_due = addhb_ints[kAddhbIntDue];
  status->addhb_applied = addhb_ints[kAddhbIntApplied];
  status->addhb_nij = addhb_ints[kAddhbIntNij];
  status->density_max = cnvgz_scalars[kCnvgzPmax];
  status->density_rms = cnvgz_scalars[kCnvgzDensityRms];
  status->cnvgz_active_calls = cnvgz_ints[kCnvgzIntActiveCalls];
  status->cnvgz_noop_calls = cnvgz_ints[kCnvgzIntNoopCalls];
  ensure_cnvgz_activity_from_resident_stage_calls(status);
  status->energy_total = energy_totals[2];
  status->energy_scf = isitsc_scalars[kIsitscEnergyScf];
  status->energy_delta = isitsc_scalars[kIsitscEnergyDelta];
  status->isitsc_okscf = isitsc_ints[kIsitscIntOkscf];
  status->isitsc_iscf = isitsc_ints[kIsitscIntIscf];
  status->isitsc_iemin = isitsc_ints[kIsitscIntIemin];
  status->isitsc_iemax = isitsc_ints[kIsitscIntIemax];
  status->isitsc_scf1 = isitsc_ints[kIsitscIntScf1];
  publish_pls_runtime_status(status, &ctx);
  return publish_cosmo_status_from_gpu(ctx, status);
}

bool env_enabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') return false;
  std::string normalized;
  for (const char *p = value; *p != '\0' && normalized.size() < 16; ++p) {
    if (!std::isspace(static_cast<unsigned char>(*p))) {
      normalized.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
    }
  }
  return normalized != "0" && normalized != "f" &&
         normalized != "false" && normalized != "n" &&
         normalized != "no" && normalized != "off";
}

bool strict_resident_request_enabled() {
  return env_enabled("MOPAC_MOZYME_SCF_STRICT_RESIDENT") ||
         env_enabled("MOPAC_MOZYME_SCF_GPU") ||
         env_enabled("MOPAC_MOZYME_GPU_STRICT") ||
         env_enabled("MOPAC_MOZYME_FULL_SCF_GPU");
}

bool resident_scf_request_enabled() {
  return strict_resident_request_enabled() ||
         env_enabled("MOPAC_MOZYME_RESIDENT_SCF");
}

bool host_commit_marker_enabled() {
  return resident_scf_request_enabled() ||
         env_enabled("MOPAC_GPU_PROFILE") ||
         env_enabled("MOPAC_MOZYME_PROFILE") ||
         env_enabled("MOPAC_MOZYME_SECTION_PROFILE");
}

enum class ResidentReturnDecision {
  ContinueOnDevice,
  CompleteAndPublish,
  CpuBoundary,
  IterationExhausted,
  PlsRestart,
  StageFailed
};

struct ResidentControlSnapshot {
  ResidentReturnDecision decision = ResidentReturnDecision::CpuBoundary;
  int diagg_mode = 0;
  int nhb = 0;
  int diagg_nf = 0;
  int diagg2_nrejct[2] = {0, 0};
  int isitsc_iemin = 0;
  int isitsc_iemax = 0;
  int isitsc_scf1 = 0;
  int current_iter = 0;
  int addhb_due = 0;
  int use_three_point = 0;
  int lstart = 0;
  int pls_supervisor_calls = 0;
  int pls_restart_required = 0;
  int pls_history_count = 0;
  int pls_restart_reset_device_calls = 0;
  int pls_restart_done = 0;
  double diagg_fref = 0.0;
  double diagg_oldlim = 0.0;
  double diagg_safety = 0.0;
  double ovmax = 0.0;
  double previous_escf = 0.0;
  double shift = 0.0;
  double pls_ovmax_delta = 0.0;
  double pls_energy_delta = 0.0;
};

ResidentReturnDecision resident_decision_from_code(int code) {
  if (code == kResidentDecisionComplete) {
    return ResidentReturnDecision::CompleteAndPublish;
  }
  if (code == kResidentDecisionCpuBoundary) {
    return ResidentReturnDecision::CpuBoundary;
  }
  if (code == kResidentDecisionIterationExhausted) {
    return ResidentReturnDecision::IterationExhausted;
  }
  if (code == kResidentDecisionPlsRestart) {
    return ResidentReturnDecision::PlsRestart;
  }
  if (code == kResidentDecisionStageFailed) {
    return ResidentReturnDecision::StageFailed;
  }
  return ResidentReturnDecision::ContinueOnDevice;
}

int resident_decision_to_code(ResidentReturnDecision decision) {
  switch (decision) {
    case ResidentReturnDecision::CompleteAndPublish:
      return kResidentDecisionComplete;
    case ResidentReturnDecision::CpuBoundary:
      return kResidentDecisionCpuBoundary;
    case ResidentReturnDecision::IterationExhausted:
      return kResidentDecisionIterationExhausted;
    case ResidentReturnDecision::PlsRestart:
      return kResidentDecisionPlsRestart;
    case ResidentReturnDecision::StageFailed:
      return kResidentDecisionStageFailed;
    case ResidentReturnDecision::ContinueOnDevice:
    default:
      return kResidentDecisionContinue;
  }
}

bool copy_resident_control_snapshot_from_gpu(MozymeScfContext &ctx,
                                             ResidentControlSnapshot *out,
                                             bool count_strict_poll = true) {
  if (!out || !ctx.device.uploaded) return false;
  auto &dev = ctx.device;
  if (!dev.resident_control_ints.ptr || !dev.resident_control_scalars.ptr ||
      dev.resident_control_ints.count < kResidentControlIntCount ||
      dev.resident_control_scalars.count < kResidentControlDoubleCount) {
    return false;
  }
  if (count_strict_poll && strict_resident_request_enabled()) {
    ++ctx.strict_resident_control_polls;
  }

  int host_ints[kResidentControlIntCount] = {};
  double host_scalars[kResidentControlDoubleCount] = {};
  if (!cuda_context_ok(
          cudaMemcpy(host_ints, dev.resident_control_ints.ptr,
                     sizeof(host_ints), cudaMemcpyDeviceToHost),
          "resident loop control snapshot integer copy")) {
    return false;
  }
  if (!cuda_context_ok(
          cudaMemcpy(host_scalars, dev.resident_control_scalars.ptr,
                     sizeof(host_scalars), cudaMemcpyDeviceToHost),
          "resident loop control snapshot scalar copy")) {
    return false;
  }

  out->decision =
      resident_decision_from_code(host_ints[kResidentControlDecision]);
  out->diagg_mode = host_ints[kResidentControlDiaggMode];
  out->nhb = host_ints[kResidentControlNhb];
  out->diagg_nf = host_ints[kResidentControlDiaggNf];
  out->diagg2_nrejct[0] = host_ints[kResidentControlNrej0];
  out->diagg2_nrejct[1] = host_ints[kResidentControlNrej1];
  out->isitsc_iemin = host_ints[kResidentControlIemin];
  out->isitsc_iemax = host_ints[kResidentControlIemax];
  out->isitsc_scf1 = host_ints[kResidentControlScf1];
  out->current_iter = host_ints[kResidentControlCurrentIter];
  out->addhb_due = host_ints[kResidentControlAddhbDue];
  out->use_three_point = host_ints[kResidentControlUseThreePoint];
  out->lstart = host_ints[kResidentControlLstart];
  out->pls_restart_required = host_ints[kResidentControlPlsRestartRequired];
  out->pls_supervisor_calls = host_ints[kResidentControlPlsCalls];
  out->pls_history_count = host_ints[kResidentControlPlsHistoryCount];
  out->pls_restart_reset_device_calls =
      host_ints[kResidentControlPlsRestartResetCalls];
  out->pls_restart_done = host_ints[kResidentControlPlsRestartDone];
  ctx.pls_restart_reset_device_calls =
      std::max(ctx.pls_restart_reset_device_calls,
               out->pls_restart_reset_device_calls);
  ctx.pls_restart_done = std::max(ctx.pls_restart_done,
                                  out->pls_restart_done);
  out->diagg_fref = host_scalars[kResidentControlDiaggFref];
  out->diagg_oldlim = host_scalars[kResidentControlDiaggOldlim];
  out->diagg_safety = host_scalars[kResidentControlDiaggSafety];
  out->ovmax = host_scalars[kResidentControlOvmax];
  out->previous_escf = host_scalars[kResidentControlPreviousEscf];
  out->shift = host_scalars[kResidentControlShift];
  out->pls_ovmax_delta = host_scalars[kResidentControlPlsOvmaxDelta];
  out->pls_energy_delta = host_scalars[kResidentControlPlsEnergyDelta];
  return true;
}

bool resident_pls_restart_resolved_on_device(
    const ResidentControlSnapshot &control) {
  return control.pls_restart_required == 0 &&
         control.pls_restart_reset_device_calls > 0 &&
         control.pls_restart_done == 1;
}

bool advance_resident_control_on_gpu(MozymeScfContext &ctx,
                                     bool strict_resident) {
  if (!ctx.device.uploaded) return false;
  auto &dev = ctx.device;
  if (!dev.diagg_ints.ptr || !dev.diagg_scalars.ptr || !dev.addhb_ints.ptr ||
      !dev.addhb_scalars.ptr || !dev.isitsc_ints.ptr ||
      !dev.isitsc_scalars.ptr || !dev.pls_ints.ptr || !dev.pls_scalars.ptr) {
    return false;
  }

  if (!device_buffer_ready(dev.resident_control_ints,
                           kResidentControlIntCount)) {
    return false;
  }
  if (!device_buffer_ready(dev.resident_control_scalars,
                           kResidentControlDoubleCount)) {
    return false;
  }

  mozyme_resident_control_advance_kernel<<<1, 1>>>(
      ctx.config.current_iter, ctx.config.max_iter, strict_resident ? 1 : 0,
      ctx.config.use_three_point, ctx.config.lstart, ctx.config.shift,
      dev.diagg_ints.ptr, dev.diagg_scalars.ptr, dev.addhb_ints.ptr,
      dev.addhb_scalars.ptr, dev.isitsc_ints.ptr, dev.isitsc_scalars.ptr,
      dev.pls_ints.ptr, dev.pls_scalars.ptr, dev.resident_control_ints.ptr,
      dev.resident_control_scalars.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident loop control advance kernel")) {
    return false;
  }
  return true;
}

bool apply_resident_pls_restart_if_requested_on_gpu(MozymeScfContext &ctx) {
  if (!ctx.device.uploaded) return false;
  auto &dev = ctx.device;
  if (!dev.resident_control_ints.ptr || !dev.resident_control_scalars.ptr ||
      !dev.pold.ptr || !dev.p1.ptr || !dev.isitsc_ints.ptr ||
      !dev.pls_ints.ptr) {
    return false;
  }

  const int pold_count =
      std::min(ctx.state.pold_dim, ctx.config.mpack);
  const int p1_count =
      std::min(ctx.state.p1_dim, ctx.config.norbs);
  if (pold_count <= 0 || p1_count <= 0) return false;

  constexpr int kThreads = 128;
  const int max_count = std::max(std::max(pold_count, p1_count), 10);
  mozyme_resident_pls_restart_zero_kernel<<<ceil_div(max_count, kThreads),
                                            kThreads>>>(
      pold_count, p1_count, dev.pold.ptr, dev.p1.ptr, dev.isitsc_ints.ptr,
      dev.resident_control_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident PLS restart zero kernel")) {
    return false;
  }
  mozyme_resident_pls_restart_finalize_kernel<<<1, 1>>>(
      dev.resident_control_ints.ptr, dev.resident_control_scalars.ptr,
      dev.pls_ints.ptr);
  if (!cuda_context_ok(cudaGetLastError(),
                       "resident PLS restart finalize kernel")) {
    return false;
  }
  return true;
}

bool compute_resident_control_after_iteration(MozymeScfContext &ctx,
                                              bool strict_resident,
                                              ResidentControlSnapshot *out) {
  if (!out) return false;
  if (!advance_resident_control_on_gpu(ctx, strict_resident)) return false;
  return copy_resident_control_snapshot_from_gpu(ctx, out);
}

bool advance_strict_resident_control_on_gpu(MozymeScfContext &ctx) {
  if (!advance_resident_control_on_gpu(ctx, true)) return false;
  return apply_resident_pls_restart_if_requested_on_gpu(ctx);
}

void publish_resident_control_to_status(MozymeScfStatus *status,
                                        const ResidentControlSnapshot &control) {
  if (!status) return;
  status->resident_decision = resident_decision_to_code(control.decision);
  status->idiagg = control.diagg_mode;
  status->nhb = control.nhb;
  status->diagg_nf = control.diagg_nf;
  status->diagg2_nrejct[0] = control.diagg2_nrejct[0];
  status->diagg2_nrejct[1] = control.diagg2_nrejct[1];
  status->diagg_fref = control.diagg_fref;
  status->diagg_oldlim = control.diagg_oldlim;
  status->diagg_safety = control.diagg_safety;
  status->isitsc_iemin = control.isitsc_iemin;
  status->isitsc_iemax = control.isitsc_iemax;
  status->isitsc_scf1 = control.isitsc_scf1;
  status->use_three_point = control.use_three_point;
  status->lstart = control.lstart;
  status->shift = control.shift;
  status->addhb_due = control.addhb_due;
  status->pls_supervisor_calls = control.pls_supervisor_calls;
  status->pls_restart_required = control.pls_restart_required;
  status->pls_history_count = control.pls_history_count;
  status->pls_restart_reset_device_calls =
      control.pls_restart_reset_device_calls;
  status->pls_restart_done = control.pls_restart_done;
  status->pls_ovmax_delta = control.pls_ovmax_delta;
  status->pls_energy_delta = control.pls_energy_delta;
  if (control.current_iter >= 0) status->iterations = control.current_iter;
}

void advance_config_after_resident_control(
    MozymeScfContext &ctx, const ResidentControlSnapshot &control) {
  ctx.config.diagg_mode = control.diagg_mode;
  ctx.config.nhb = control.nhb;
  ctx.config.diagg_nf = control.diagg_nf;
  ctx.config.diagg_fref = control.diagg_fref;
  ctx.config.diagg_oldlim = control.diagg_oldlim;
  ctx.config.diagg_safety = control.diagg_safety;
  ctx.config.diagg2_nrejct[0] = control.diagg2_nrejct[0];
  ctx.config.diagg2_nrejct[1] = control.diagg2_nrejct[1];
  ctx.config.ovmax = control.ovmax;
  ctx.config.previous_escf = control.previous_escf;
  ctx.config.isitsc_iemin = control.isitsc_iemin;
  ctx.config.isitsc_iemax = control.isitsc_iemax;
  ctx.config.isitsc_scf1 = control.isitsc_scf1;
  ctx.config.use_three_point = control.use_three_point;
  ctx.config.lstart = control.lstart;
  ctx.config.shift = control.shift;
  ctx.config.current_iter = control.current_iter;
  ctx.config.addhb_due = control.addhb_due;
}

void advance_config_after_resident_iteration(
    MozymeScfContext &ctx, const MozymeScfStatus &status,
    const ResidentControlSnapshot &control) {
  advance_config_after_resident_control(ctx, control);
  std::memcpy(ctx.config.isitsc_escf0, status.isitsc_escf0,
              sizeof(ctx.config.isitsc_escf0));
}

template <typename T>
bool copy_device_buffer(DeviceBuffer<T> &dst, const DeviceBuffer<T> &src) {
  if (src.count == 0) {
    dst.reset();
    return true;
  }
  if (!src.ptr || !device_buffer_ready(dst, src.count)) return false;
  return cuda_context_ok(
      cudaMemcpy(dst.ptr, src.ptr, src.count * sizeof(T),
                 cudaMemcpyDeviceToDevice),
      "resident checkpoint device copy");
}

bool save_resident_checkpoint(MozymeScfDeviceState &dev) {
  if (!resize_resident_checkpoint_buffers(dev)) return false;
  return copy_device_buffer(dev.checkpoint_p, dev.p) &&
         copy_device_buffer(dev.checkpoint_f, dev.f) &&
         copy_device_buffer(dev.checkpoint_partp, dev.partp) &&
         copy_device_buffer(dev.checkpoint_partf, dev.partf) &&
         copy_device_buffer(dev.checkpoint_pold, dev.pold) &&
         copy_device_buffer(dev.checkpoint_p1, dev.p1) &&
         copy_device_buffer(dev.checkpoint_p2, dev.p2) &&
         copy_device_buffer(dev.checkpoint_p3, dev.p3) &&
         copy_device_buffer(dev.checkpoint_idiag, dev.idiag) &&
         copy_device_buffer(dev.checkpoint_iorbs, dev.iorbs) &&
         copy_device_buffer(dev.checkpoint_ncf, dev.ncf) &&
         copy_device_buffer(dev.checkpoint_nncf, dev.nncf) &&
         copy_device_buffer(dev.checkpoint_ncocc, dev.ncocc) &&
         copy_device_buffer(dev.checkpoint_nce, dev.nce) &&
         copy_device_buffer(dev.checkpoint_nnce, dev.nnce) &&
         copy_device_buffer(dev.checkpoint_ncvir, dev.ncvir) &&
         copy_device_buffer(dev.checkpoint_icocc, dev.icocc) &&
         copy_device_buffer(dev.checkpoint_icvir, dev.icvir) &&
         copy_device_buffer(dev.checkpoint_cocc, dev.cocc) &&
         copy_device_buffer(dev.checkpoint_cvir, dev.cvir) &&
         copy_device_buffer(dev.checkpoint_eigs, dev.eigs) &&
         copy_device_buffer(dev.checkpoint_nfmo, dev.nfmo) &&
         copy_device_buffer(dev.checkpoint_nfirst, dev.nfirst) &&
         copy_device_buffer(dev.checkpoint_nlast, dev.nlast) &&
         copy_device_buffer(dev.checkpoint_coord, dev.coord) &&
         copy_device_buffer(dev.checkpoint_nat, dev.nat) &&
         copy_device_buffer(dev.checkpoint_ifmo, dev.ifmo) &&
         copy_device_buffer(dev.checkpoint_fmo, dev.fmo) &&
         copy_device_buffer(dev.checkpoint_cosmo_qscat, dev.cosmo_qscat) &&
         copy_device_buffer(dev.checkpoint_cosmo_phinet, dev.cosmo_phinet) &&
         copy_device_buffer(dev.checkpoint_cosmo_qscnet, dev.cosmo_qscnet) &&
         copy_device_buffer(dev.checkpoint_cosmo_qdenet, dev.cosmo_qdenet) &&
         copy_device_buffer(dev.checkpoint_cosmo_scalars,
                            dev.cosmo_scalars) &&
         copy_device_buffer(dev.checkpoint_cosmo_status_scalars,
                            dev.cosmo_status_scalars) &&
         copy_device_buffer(dev.checkpoint_cosmo_status_ints,
                            dev.cosmo_status_ints) &&
         copy_device_buffer(dev.checkpoint_energy_sums, dev.energy_sums) &&
         copy_device_buffer(dev.checkpoint_cnvgz_sums, dev.cnvgz_sums) &&
         copy_device_buffer(dev.checkpoint_diagg_scalars, dev.diagg_scalars) &&
         copy_device_buffer(dev.checkpoint_addhb_scalars, dev.addhb_scalars) &&
         copy_device_buffer(dev.checkpoint_isitsc_scalars,
                            dev.isitsc_scalars) &&
         copy_device_buffer(dev.checkpoint_isitsc_escf0, dev.isitsc_escf0) &&
         copy_device_buffer(dev.checkpoint_resident_control_scalars,
                            dev.resident_control_scalars) &&
         copy_device_buffer(dev.checkpoint_pls_scalars, dev.pls_scalars) &&
         copy_device_buffer(dev.checkpoint_cnvgz_ints, dev.cnvgz_ints) &&
         copy_device_buffer(dev.checkpoint_diagg_ints, dev.diagg_ints) &&
         copy_device_buffer(dev.checkpoint_addhb_ints, dev.addhb_ints) &&
         copy_device_buffer(dev.checkpoint_isitsc_ints, dev.isitsc_ints) &&
         copy_device_buffer(dev.checkpoint_helecz_ints, dev.helecz_ints) &&
         copy_device_buffer(dev.checkpoint_resident_stage_ints,
                            dev.resident_stage_ints) &&
         copy_device_buffer(dev.checkpoint_resident_stage_calls,
                            dev.resident_stage_calls) &&
         copy_device_buffer(dev.checkpoint_resident_control_ints,
                            dev.resident_control_ints) &&
         copy_device_buffer(dev.checkpoint_pls_ints, dev.pls_ints);
}

bool restore_resident_checkpoint(MozymeScfDeviceState &dev) {
  return copy_device_buffer(dev.p, dev.checkpoint_p) &&
         copy_device_buffer(dev.f, dev.checkpoint_f) &&
         copy_device_buffer(dev.partp, dev.checkpoint_partp) &&
         copy_device_buffer(dev.partf, dev.checkpoint_partf) &&
         copy_device_buffer(dev.pold, dev.checkpoint_pold) &&
         copy_device_buffer(dev.p1, dev.checkpoint_p1) &&
         copy_device_buffer(dev.p2, dev.checkpoint_p2) &&
         copy_device_buffer(dev.p3, dev.checkpoint_p3) &&
         copy_device_buffer(dev.idiag, dev.checkpoint_idiag) &&
         copy_device_buffer(dev.iorbs, dev.checkpoint_iorbs) &&
         copy_device_buffer(dev.ncf, dev.checkpoint_ncf) &&
         copy_device_buffer(dev.nncf, dev.checkpoint_nncf) &&
         copy_device_buffer(dev.ncocc, dev.checkpoint_ncocc) &&
         copy_device_buffer(dev.nce, dev.checkpoint_nce) &&
         copy_device_buffer(dev.nnce, dev.checkpoint_nnce) &&
         copy_device_buffer(dev.ncvir, dev.checkpoint_ncvir) &&
         copy_device_buffer(dev.icocc, dev.checkpoint_icocc) &&
         copy_device_buffer(dev.icvir, dev.checkpoint_icvir) &&
         copy_device_buffer(dev.cocc, dev.checkpoint_cocc) &&
         copy_device_buffer(dev.cvir, dev.checkpoint_cvir) &&
         copy_device_buffer(dev.eigs, dev.checkpoint_eigs) &&
         copy_device_buffer(dev.nfmo, dev.checkpoint_nfmo) &&
         copy_device_buffer(dev.nfirst, dev.checkpoint_nfirst) &&
         copy_device_buffer(dev.nlast, dev.checkpoint_nlast) &&
         copy_device_buffer(dev.coord, dev.checkpoint_coord) &&
         copy_device_buffer(dev.nat, dev.checkpoint_nat) &&
         copy_device_buffer(dev.ifmo, dev.checkpoint_ifmo) &&
         copy_device_buffer(dev.fmo, dev.checkpoint_fmo) &&
         copy_device_buffer(dev.cosmo_qscat, dev.checkpoint_cosmo_qscat) &&
         copy_device_buffer(dev.cosmo_phinet, dev.checkpoint_cosmo_phinet) &&
         copy_device_buffer(dev.cosmo_qscnet, dev.checkpoint_cosmo_qscnet) &&
         copy_device_buffer(dev.cosmo_qdenet, dev.checkpoint_cosmo_qdenet) &&
         copy_device_buffer(dev.cosmo_scalars,
                            dev.checkpoint_cosmo_scalars) &&
         copy_device_buffer(dev.cosmo_status_scalars,
                            dev.checkpoint_cosmo_status_scalars) &&
         copy_device_buffer(dev.cosmo_status_ints,
                            dev.checkpoint_cosmo_status_ints) &&
         copy_device_buffer(dev.energy_sums, dev.checkpoint_energy_sums) &&
         copy_device_buffer(dev.cnvgz_sums, dev.checkpoint_cnvgz_sums) &&
         copy_device_buffer(dev.diagg_scalars, dev.checkpoint_diagg_scalars) &&
         copy_device_buffer(dev.addhb_scalars, dev.checkpoint_addhb_scalars) &&
         copy_device_buffer(dev.isitsc_scalars,
                            dev.checkpoint_isitsc_scalars) &&
         copy_device_buffer(dev.isitsc_escf0, dev.checkpoint_isitsc_escf0) &&
         copy_device_buffer(dev.resident_control_scalars,
                            dev.checkpoint_resident_control_scalars) &&
         copy_device_buffer(dev.pls_scalars, dev.checkpoint_pls_scalars) &&
         copy_device_buffer(dev.cnvgz_ints, dev.checkpoint_cnvgz_ints) &&
         copy_device_buffer(dev.diagg_ints, dev.checkpoint_diagg_ints) &&
         copy_device_buffer(dev.addhb_ints, dev.checkpoint_addhb_ints) &&
         copy_device_buffer(dev.isitsc_ints, dev.checkpoint_isitsc_ints) &&
         copy_device_buffer(dev.helecz_ints, dev.checkpoint_helecz_ints) &&
         copy_device_buffer(dev.resident_stage_ints,
                            dev.checkpoint_resident_stage_ints) &&
         copy_device_buffer(dev.resident_stage_calls,
                            dev.checkpoint_resident_stage_calls) &&
         copy_device_buffer(dev.resident_control_ints,
                            dev.checkpoint_resident_control_ints) &&
         copy_device_buffer(dev.pls_ints, dev.checkpoint_pls_ints);
}

bool run_resident_iteration_on_gpu(MozymeScfContext &ctx,
                                   MozymeScfStatus *status,
                                   double *accumulated_ms,
                                   bool publish_iteration_outputs) {
  double energy = 0.0;
  double wall_ms = 0.0;
  double eimp_ms = 0.0;
  double density_ms = 0.0;
  double fock_ms = 0.0;
  double cnvgz_ms = 0.0;
  double check_ms = 0.0;
  double setup_ms = 0.0;
  double diagg_ms = 0.0;
  double isitsc_ms = 0.0;
  double addhb_ms = 0.0;
  int density_terms = 0;
  int completed = 0;
  const bool poll_stage_status = publish_iteration_outputs;

  begin_resident_iteration_status(status, ctx, *accumulated_ms);
  if (!reset_resident_stage_status_on_gpu(ctx)) {
    status->iterations = ctx.config.current_iter;
    status->stage_completed = completed;
    status->stage_required = kMozymeScfStageFull;
    status->stage_missing = kMozymeScfStageFull & ~completed;
    status->code = kMozymeScfUnsupported;
    status->ready = 0;
    set_status_loop_control(status, ctx);
    return false;
  }
  if (poll_stage_status) {
    if (!resident_stage_device_confirmed(
            ctx, kMozymeScfStageUpload, kResidentStageSlotUpload, &completed,
            status)) {
      status->iterations = ctx.config.current_iter;
      status->stage_completed = completed;
      status->stage_required = kMozymeScfStageFull;
      status->stage_missing = kMozymeScfStageFull & ~completed;
      status->code = kMozymeScfUnsupported;
      status->ready = 0;
      set_status_loop_control(status, ctx);
      return false;
    }
  } else {
    completed = kMozymeScfStageUpload;
  }
  add_stage_time(status, accumulated_ms, 0.0, kResidentStageSlotUpload);

  auto publish_stage_status = [&]() {
    if (!publish_iteration_outputs) {
      status->stage_completed = completed;
      status->stage_required = kMozymeScfStageFull;
      status->stage_missing = kMozymeScfStageFull & ~completed;
      status->code =
          completed_full_stage_mask(completed) ? kMozymeScfSuccess
                                               : kMozymeScfUnsupported;
      return;
    }
    if (!publish_resident_stage_status_from_gpu(ctx, status)) {
      status->stage_completed = completed;
      status->stage_required = kMozymeScfStageFull;
      status->stage_missing = kMozymeScfStageFull & ~completed;
      status->code =
          completed_full_stage_mask(completed) ? kMozymeScfSuccess
                                               : kMozymeScfUnsupported;
    }
  };

  auto fail_stage = [&]() -> bool {
    status->iterations = ctx.config.current_iter;
    publish_stage_status();
    status->code = kMozymeScfUnsupported;
    status->ready = 0;
    set_status_loop_control(status, ctx);
    return false;
  };

  auto complete_stage_from_int = [&](int stage_bits, const int *values,
                                     int value_slot,
                                     int expected_value,
                                     int stage_slot) -> bool {
    if (!mark_resident_stage_if_int_on_gpu(ctx, stage_bits, values,
                                           value_slot, expected_value,
                                           stage_slot)) {
      return false;
    }
    if (!poll_stage_status) {
      completed |= stage_bits;
      return true;
    }
    if (!resident_stage_device_confirmed(ctx, stage_bits, stage_slot,
                                         &completed, status)) {
      return false;
    }
    completed = status->stage_completed;
    return true;
  };

  if ((ctx.config.flags & kMozymeScfFlagInitialSetup) != 0) {
    if (!compute_initial_setup_on_gpu(
            ctx, publish_iteration_outputs ? &energy : nullptr, &setup_ms)) {
      return fail_stage();
    }
    ctx.config.flags &= ~kMozymeScfFlagInitialSetup;
    if (publish_iteration_outputs) status->energy_total = energy;
    add_stage_time(status, accumulated_ms, setup_ms);
  }

  if (compute_check_on_gpu(ctx, &check_ms)) {
    if (!complete_stage_from_int(kMozymeScfStageCheck, ctx.device.check_ints.ptr,
                                 kCheckIntOk, 1, kResidentStageSlotCheck)) {
      return fail_stage();
    }
    add_stage_time(status, accumulated_ms, check_ms, kResidentStageSlotCheck);
  } else {
    return fail_stage();
  }

  // EIMP intentionally writes a shadow density buffer; DIAGG consumes that
  // temporary data and the real density is rebuilt later in the iteration.
  if ((completed & kMozymeScfStageCheck) != 0 &&
      compute_eimp_probe_on_gpu(ctx, &eimp_ms)) {
    if (!complete_stage_from_int(kMozymeScfStageEimp,
                                 ctx.device.eimp_pair_updates.ptr, 1, 1,
                                 kResidentStageSlotEimp)) {
      return fail_stage();
    }
    add_stage_time(status, accumulated_ms, eimp_ms, kResidentStageSlotEimp);
  } else {
    return fail_stage();
  }
  if ((completed & kMozymeScfStageEimp) != 0 &&
      compute_diagg_on_gpu(ctx, &diagg_ms)) {
    if (!complete_stage_from_int(kMozymeScfStageDiagg,
                                 ctx.device.diagg_ints.ptr, kDiaggIntOk,
                                 1, kResidentStageSlotDiagg)) {
      return fail_stage();
    }
    add_stage_time(status, accumulated_ms, diagg_ms, kResidentStageSlotDiagg);
  } else {
    return fail_stage();
  }
  if (compute_density_on_gpu(ctx, &density_terms, &density_ms)) {
    if (!complete_stage_from_int(kMozymeScfStageDensity,
                                 ctx.device.density_updates.ptr, 1, 1,
                                 kResidentStageSlotDensity)) {
      return fail_stage();
    }
    add_stage_time(status, accumulated_ms, density_ms,
                   kResidentStageSlotDensity);
  } else {
    return fail_stage();
  }
  if ((completed & (kMozymeScfStageDiagg | kMozymeScfStageDensity)) ==
          (kMozymeScfStageDiagg | kMozymeScfStageDensity) &&
      compute_addhb_on_gpu(ctx, &addhb_ms)) {
    if (!complete_stage_from_int(kMozymeScfStageAddhb,
                                 ctx.device.addhb_ints.ptr, kAddhbIntOk,
                                 1, kResidentStageSlotAddhb)) {
      return fail_stage();
    }
    add_stage_time(status, accumulated_ms, addhb_ms, kResidentStageSlotAddhb);
  } else {
    return fail_stage();
  }
  // Match the CPU loop: cnvgz mutates density history only while
  // three-point extrapolation is active.  Once disabled, the resident stage is
  // a GPU-owned no-op so the SCF contract stays complete without CPU fallback.
  if (compute_cnvgz_probe_on_gpu(ctx, &cnvgz_ms, true)) {
    if (!complete_stage_from_int(kMozymeScfStageCnvgz,
                                 ctx.device.cnvgz_ints.ptr, kCnvgzIntOk,
                                 1, kResidentStageSlotCnvgz)) {
      return fail_stage();
    }
    add_stage_time(status, accumulated_ms, cnvgz_ms, kResidentStageSlotCnvgz);
  } else {
    return fail_stage();
  }
  if (compute_fock_on_gpu(ctx, &fock_ms)) {
    if (!complete_stage_from_int(kMozymeScfStageFock,
                                 ctx.device.fock_ints.ptr, kFockIntOk, 1,
                                 kResidentStageSlotFock)) {
      return fail_stage();
    }
    add_stage_time(status, accumulated_ms, fock_ms, kResidentStageSlotFock);
  } else {
    return fail_stage();
  }
  if ((completed & kMozymeScfStageFock) != 0 &&
      compute_helecz_on_gpu(ctx, nullptr, &wall_ms)) {
    if (!complete_stage_from_int(kMozymeScfStageHelecz,
                                 ctx.device.helecz_ints.ptr, kHeleczIntOk,
                                 1, kResidentStageSlotHelecz)) {
      return fail_stage();
    }
    add_stage_time(status, accumulated_ms, wall_ms, kResidentStageSlotHelecz);
  } else {
    return fail_stage();
  }
  if ((completed & (kMozymeScfStageDiagg | kMozymeScfStageHelecz)) ==
          (kMozymeScfStageDiagg | kMozymeScfStageHelecz) &&
      compute_isitsc_on_gpu(ctx, &isitsc_ms)) {
    if (!complete_stage_from_int(kMozymeScfStageIsitsc,
                                 ctx.device.isitsc_ints.ptr,
                                 kIsitscIntValid, 1,
                                 kResidentStageSlotIsitsc)) {
      return fail_stage();
    }
    add_stage_time(status, accumulated_ms, isitsc_ms,
                   kResidentStageSlotIsitsc);
  } else {
    return fail_stage();
  }

  status->iterations = ctx.config.current_iter;
  if (publish_iteration_outputs) {
    publish_stage_status();
  } else {
    status->stage_completed = completed;
    status->stage_required = kMozymeScfStageFull;
    status->stage_missing = kMozymeScfStageFull & ~completed;
    status->code =
        completed_full_stage_mask(completed) ? kMozymeScfSuccess
                                             : kMozymeScfUnsupported;
  }
  if (publish_iteration_outputs) {
    if (!publish_resident_iteration_outputs_from_gpu(ctx, status)) {
      return fail_stage();
    }
  }
  status->stage_required = kMozymeScfStageFull;
  status->code =
      completed_full_stage_mask(status->stage_completed) ? kMozymeScfSuccess
                                                         : kMozymeScfUnsupported;
  set_status_loop_control(status, ctx);
  if (status->code != kMozymeScfSuccess) status->ready = 0;
  return status->code == kMozymeScfSuccess;
}

bool copy_resident_state_to_host(const MozymeScfContext &ctx,
                                 ResidentFinalPublicationProof *proof = nullptr) {
  if (!ctx.device.uploaded) return false;
  const int norbs = ctx.config.norbs;
  const int mpack = ctx.config.mpack;
  if (norbs <= 0 || mpack <= 0 || ctx.state.cocc_dim <= 0 ||
      ctx.state.cvir_dim <= 0 || ctx.state.icocc_dim <= 0 ||
      ctx.state.icvir_dim <= 0 || ctx.state.fmo_dim <= 0 ||
      ctx.state.partp_dim <= 0 || ctx.state.partf_dim <= 0) {
    return false;
  }

  const auto &dev = ctx.device;
  const std::size_t nocc_slots =
      static_cast<std::size_t>(ctx.state.nocc_slots);
  const std::size_t nvir_slots =
      static_cast<std::size_t>(ctx.state.nvir_slots);
  const std::size_t norbs_count = static_cast<std::size_t>(norbs);
  const std::size_t mpack_count = static_cast<std::size_t>(mpack);
  const std::size_t partp_count = std::min(
      static_cast<std::size_t>(ctx.state.partp_dim), mpack_count);
  const std::size_t partf_count = std::min(
      static_cast<std::size_t>(ctx.state.partf_dim), mpack_count);
  const std::size_t fmo_count = static_cast<std::size_t>(ctx.state.fmo_dim);
  const std::size_t icocc_count =
      static_cast<std::size_t>(ctx.state.icocc_dim);
  const std::size_t icvir_count =
      static_cast<std::size_t>(ctx.state.icvir_dim);
  const std::size_t cocc_count =
      static_cast<std::size_t>(ctx.state.cocc_dim);
  const std::size_t cvir_count =
      static_cast<std::size_t>(ctx.state.cvir_dim);
  const std::size_t numat_count = static_cast<std::size_t>(ctx.config.natoms);

  const bool has_cosmo = ctx.state.cosmo_enabled != 0;
  const std::size_t cosmo_qscat_count = has_cosmo ? numat_count : 0;
  const std::size_t cosmo_phinet_count =
      has_cosmo ? static_cast<std::size_t>(ctx.state.cosmo_phinet_rows) *
                      static_cast<std::size_t>(ctx.state.cosmo_phinet_cols)
                : 0;
  const std::size_t cosmo_qscnet_count =
      has_cosmo ? static_cast<std::size_t>(ctx.state.cosmo_qscnet_rows) *
                      static_cast<std::size_t>(ctx.state.cosmo_qscnet_cols)
                : 0;
  const std::size_t cosmo_qdenet_count =
      has_cosmo ? static_cast<std::size_t>(ctx.state.cosmo_qdenet_rows) *
                      static_cast<std::size_t>(ctx.state.cosmo_qdenet_cols)
                : 0;
  auto host_ready = [](const void *ptr, std::size_t count) {
    return count == 0 || ptr != nullptr;
  };
  if (!host_ready(ctx.state.p, mpack_count) ||
      !host_ready(ctx.state.f, mpack_count) ||
      !host_ready(ctx.state.partp, partp_count) ||
      !host_ready(ctx.state.partf, partf_count) ||
      !host_ready(ctx.state.pold, mpack_count) ||
      !host_ready(ctx.state.p1, norbs_count) ||
      !host_ready(ctx.state.p2, norbs_count) ||
      !host_ready(ctx.state.p3, norbs_count) ||
      !host_ready(ctx.state.idiag, norbs_count) ||
      !host_ready(ctx.state.iorbs, numat_count) ||
      !host_ready(ctx.state.ncf, nocc_slots) ||
      !host_ready(ctx.state.nncf, nocc_slots) ||
      !host_ready(ctx.state.ncocc, nocc_slots) ||
      !host_ready(ctx.state.nce, nvir_slots) ||
      !host_ready(ctx.state.nnce, nvir_slots) ||
      !host_ready(ctx.state.ncvir, nvir_slots) ||
      !host_ready(ctx.state.icocc, icocc_count) ||
      !host_ready(ctx.state.icvir, icvir_count) ||
      !host_ready(ctx.state.cocc, cocc_count) ||
      !host_ready(ctx.state.cvir, cvir_count) ||
      !host_ready(ctx.state.eigs, norbs_count) ||
      !host_ready(ctx.state.nfmo, norbs_count) ||
      !host_ready(ctx.state.nfirst, numat_count) ||
      !host_ready(ctx.state.nlast, numat_count) ||
      !host_ready(ctx.state.ifmo, 2 * fmo_count) ||
      !host_ready(ctx.state.fmo, fmo_count)) {
    return false;
  }

  const std::size_t host_commit_bytes =
      (mpack_count + mpack_count + partp_count + partf_count + mpack_count +
       norbs_count + norbs_count + norbs_count + cocc_count + cvir_count +
       norbs_count + fmo_count + cosmo_qscat_count + cosmo_phinet_count +
       cosmo_qscnet_count + cosmo_qdenet_count) *
          sizeof(double) +
      (norbs_count + numat_count + nocc_slots + nocc_slots +
       nocc_slots + nvir_slots + nvir_slots + nvir_slots + icocc_count +
       icvir_count + norbs_count + numat_count + numat_count +
       2 * fmo_count) *
          sizeof(int) +
      (has_cosmo ? sizeof(double) * kCosmoScalarCount : 0);
  const int host_commit_arrays = has_cosmo ? 32 : 26;
  const ResidentFinalPublicationProof final_publication_proof{
      host_commit_arrays, host_commit_bytes, has_cosmo ? 1 : 0};

  if (!stage_and_commit_device_vector(ctx.state.p, dev.p, mpack_count,
                                      "resident final p stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.f, dev.f, mpack_count,
                                      "resident final f stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.partp, dev.partp, partp_count,
                                      "resident final partp stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.partf, dev.partf, partf_count,
                                      "resident final partf stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.pold, dev.pold, mpack_count,
                                      "resident final pold stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.p1, dev.p1, norbs_count,
                                      "resident final p1 stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.p2, dev.p2, norbs_count,
                                      "resident final p2 stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.p3, dev.p3, norbs_count,
                                      "resident final p3 stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.idiag, dev.idiag, norbs_count,
                                      "resident final idiag stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.iorbs, dev.iorbs, numat_count,
                                      "resident final iorbs stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.ncf, dev.ncf, nocc_slots,
                                      "resident final ncf stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.nncf, dev.nncf, nocc_slots,
                                      "resident final nncf stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.ncocc, dev.ncocc, nocc_slots,
                                      "resident final ncocc stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.nce, dev.nce, nvir_slots,
                                      "resident final nce stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.nnce, dev.nnce, nvir_slots,
                                      "resident final nnce stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.ncvir, dev.ncvir, nvir_slots,
                                      "resident final ncvir stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.icocc, dev.icocc, icocc_count,
                                      "resident final icocc stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.icvir, dev.icvir, icvir_count,
                                      "resident final icvir stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.cocc, dev.cocc, cocc_count,
                                      "resident final cocc stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.cvir, dev.cvir, cvir_count,
                                      "resident final cvir stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.eigs, dev.eigs, norbs_count,
                                      "resident final eigs stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.nfmo, dev.nfmo, norbs_count,
                                      "resident final nfmo stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.nfirst, dev.nfirst,
                                      numat_count,
                                      "resident final nfirst stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.nlast, dev.nlast, numat_count,
                                      "resident final nlast stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.ifmo, dev.ifmo, 2 * fmo_count,
                                      "resident final ifmo stage")) return false;
  if (!stage_and_commit_device_vector(ctx.state.fmo, dev.fmo, fmo_count,
                                      "resident final fmo stage")) return false;
  if (has_cosmo) {
    if (!host_ready(ctx.state.cosmo_qscat, cosmo_qscat_count) ||
        !host_ready(ctx.state.cosmo_phinet, cosmo_phinet_count) ||
        !host_ready(ctx.state.cosmo_qscnet, cosmo_qscnet_count) ||
        !host_ready(ctx.state.cosmo_qdenet, cosmo_qdenet_count) ||
        !ctx.state.cosmo_solv_energy_ptr || !ctx.state.cosmo_ediel_ptr) {
      return false;
    }
    if (!stage_and_commit_device_vector(ctx.state.cosmo_qscat,
                                        dev.cosmo_qscat, cosmo_qscat_count,
                                        "resident final COSMO qscat stage"))
      return false;
    if (!stage_and_commit_device_vector(ctx.state.cosmo_phinet,
                                        dev.cosmo_phinet, cosmo_phinet_count,
                                        "resident final COSMO phinet stage"))
      return false;
    if (!stage_and_commit_device_vector(ctx.state.cosmo_qscnet,
                                        dev.cosmo_qscnet, cosmo_qscnet_count,
                                        "resident final COSMO qscnet stage"))
      return false;
    if (!stage_and_commit_device_vector(ctx.state.cosmo_qdenet,
                                        dev.cosmo_qdenet, cosmo_qdenet_count,
                                        "resident final COSMO qdenet stage"))
      return false;
    double host_cosmo_scalars[kCosmoScalarCount] = {};
    if (!cuda_context_ok(
            cudaMemcpy(host_cosmo_scalars, dev.cosmo_scalars.ptr,
                       sizeof(host_cosmo_scalars), cudaMemcpyDeviceToHost),
            "resident COSMO scalar stage")) return false;
    std::memcpy(ctx.state.cosmo_solv_energy_ptr,
                host_cosmo_scalars + kCosmoScalarSolvEnergy,
                sizeof(double));
    std::memcpy(ctx.state.cosmo_ediel_ptr,
                host_cosmo_scalars + kCosmoScalarEdiel, sizeof(double));
  }
  if (host_commit_marker_enabled()) {
    std::fprintf(stdout,
                 "[MOZYME GPU SCF] host_commit_only=1 "
                 "phase=final_publication arrays=%d bytes=%zu "
                 "cosmo=%d\n",
                 host_commit_arrays, host_commit_bytes, has_cosmo ? 1 : 0);
    std::fflush(stdout);
  }
  if (proof) *proof = final_publication_proof;
  return true;
}
#else
MOPAC_UNUSED_SYMBOL bool upload_registered_state(MozymeScfContext &) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool compute_eimp_probe_on_gpu(MozymeScfContext &,
                                                   double *) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool compute_check_on_gpu(MozymeScfContext &, double *) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool compute_diagg_on_gpu(MozymeScfContext &, double *) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool compute_addhb_on_gpu(MozymeScfContext &, double *) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool compute_density_on_gpu(MozymeScfContext &, int *,
                                                double *) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool compute_fock_on_gpu(MozymeScfContext &, double *) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool compute_helecz_on_gpu(MozymeScfContext &, double *,
                                               double *) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool compute_isitsc_on_gpu(MozymeScfContext &, double *) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool compute_cnvgz_probe_on_gpu(MozymeScfContext &,
                                                    double *, bool) {
  return false;
}
MOPAC_UNUSED_SYMBOL bool copy_resident_state_to_host(
    const MozymeScfContext &, ResidentFinalPublicationProof * = nullptr) {
  return false;
}
#endif

}  // namespace

extern "C" int mopac_cuda_mozyme_scf_setup(const MozymeScfConfig *config,
                                            void **context) {
  if (!config || !context) return kMozymeScfBadArgument;
  if (!valid_config(*config)) return kMozymeScfBadArgument;

  auto *ctx = new (std::nothrow) MozymeScfContext();
  if (!ctx) return kMozymeScfNotReady;

  ctx->config = *config;
  *context = ctx;
  return kMozymeScfSuccess;
}

extern "C" int mopac_cuda_mozyme_scf_register_state(
    void *context, const MozymeScfState *state) {
  auto *ctx = static_cast<MozymeScfContext *>(context);
  if (!ctx || !state) return kMozymeScfBadArgument;
  if (!valid_state(ctx->config, *state)) return kMozymeScfBadArgument;

  ctx->state = *state;
  ctx->state_registered = true;
  ctx->cosmo_fock_calls = 0;
  ctx->cosmo_matvec_calls = 0;
  ctx->cosmo_cg_iterations = 0;
  ctx->pls_restart_reset_device_calls = 0;
  ctx->pls_restart_done = 0;
  ctx->strict_resident_host_syncs = 0;
  ctx->strict_resident_control_polls = 0;
  ctx->cosmo_solv_energy = state->cosmo_solv_energy;
  ctx->cosmo_ediel = state->cosmo_ediel;
  ctx->cosmo_last_residual = 0.0;
#ifdef __CUDACC__
  ctx->device.uploaded = false;
#endif
  return kMozymeScfSuccess;
}

extern "C" int mopac_cuda_mozyme_scf_run(void *context,
                                          MozymeScfStatus *status) {
  auto *ctx = static_cast<MozymeScfContext *>(context);
  if (!status) return kMozymeScfBadArgument;
  if (!ctx || !ctx->state_registered) return kMozymeScfBadArgument;

  fill_status(status, ctx, kMozymeScfNotReady);
  int final_code = kMozymeScfNotReady;

#ifdef __CUDACC__
  if (upload_registered_state(*ctx)) {
    const bool strict_resident = strict_resident_request_enabled();
    double accumulated_ms = 0.0;
    bool have_checkpoint = false;
    MozymeScfStatus checkpoint{};
    MozymeScfConfig checkpoint_config{};
    MozymeScfHostCheckpoint checkpoint_host{};
    final_code = kMozymeScfUnsupported;

    if (strict_resident) {
      int resident_loop_guard = 0;
      const int strict_loop_limit = strict_resident_loop_limit(ctx->config);
      while (resident_loop_guard < strict_loop_limit) {
        ++resident_loop_guard;
        if (!run_resident_iteration_on_gpu(*ctx, status, &accumulated_ms,
                                           false)) {
          final_code = (status->code == kMozymeScfSuccess)
                           ? kMozymeScfUnsupported
                           : status->code;
          status->ready = 0;
          break;
        }
        if (!advance_strict_resident_control_on_gpu(*ctx)) {
          final_code = kMozymeScfNotReady;
          status->ready = 0;
          break;
        }
      }
      if (final_code == kMozymeScfUnsupported) {
        ResidentControlSnapshot control{};
        if (!copy_resident_control_snapshot_from_gpu(*ctx, &control, false)) {
          final_code = kMozymeScfNotReady;
          status->ready = 0;
        } else {
          publish_resident_control_to_status(status, control);
          const ResidentReturnDecision return_decision = control.decision;
          auto publish_strict_stage_status_or_not_ready = [&]() -> bool {
            if (publish_resident_stage_status_from_gpu(*ctx, status)) return true;
            final_code = kMozymeScfNotReady;
            status->ready = 0;
            return false;
          };
          if (return_decision == ResidentReturnDecision::CompleteAndPublish) {
            const bool needs_final_reorth =
                (ctx->config.flags & kMozymeScfFlagFinalReorth) != 0;
            MozymeScfStatus final_status = *status;
            ResidentFinalPublicationProof final_publication_proof{};
            const int previous_iter = ctx->config.current_iter;
            advance_config_after_resident_control(*ctx, control);
            if (!publish_resident_stage_status_from_gpu(*ctx, &final_status) ||
                !resident_stage_status_complete(final_status, previous_iter) ||
                !publish_resident_iteration_outputs_from_gpu(*ctx,
                                                             &final_status) ||
                (needs_final_reorth &&
                 !apply_final_reorth_on_gpu(*ctx, &final_status,
                                            &accumulated_ms)) ||
                !copy_resident_state_to_host(*ctx,
                                             &final_publication_proof)) {
              final_code = kMozymeScfNotReady;
            } else {
              mark_final_publication_done(&final_status,
                                          final_publication_proof);
              publish_resident_control_to_status(&final_status, control);
              *status = final_status;
              final_code = kMozymeScfSuccess;
            }
          } else if (return_decision == ResidentReturnDecision::CpuBoundary) {
            if (publish_strict_stage_status_or_not_ready()) {
              final_code = kMozymeScfCpuBoundary;
              status->ready = 0;
            }
          } else if (return_decision == ResidentReturnDecision::PlsRestart) {
            if (publish_strict_stage_status_or_not_ready()) {
              final_code = kMozymeScfUnsupported;
              status->ready = 0;
            }
          } else if (return_decision ==
                     ResidentReturnDecision::IterationExhausted) {
            if (publish_strict_stage_status_or_not_ready()) {
              final_code = kMozymeScfUnsupported;
              status->ready = 0;
            }
          } else if (return_decision == ResidentReturnDecision::StageFailed) {
            if (publish_strict_stage_status_or_not_ready()) {
              final_code = kMozymeScfUnsupported;
              status->ready = 0;
            }
          } else {
            if (publish_strict_stage_status_or_not_ready()) {
              final_code = kMozymeScfUnsupported;
              status->ready = 0;
            }
          }
        }
      }
    } else {
      int resident_loop_guard = 0;
      while (ctx->config.current_iter > 0 &&
             resident_loop_guard < ctx->config.max_iter) {
        ++resident_loop_guard;
        const bool step_complete =
            run_resident_iteration_on_gpu(*ctx, status, &accumulated_ms, true);
        if (!step_complete) {
          if (have_checkpoint) ctx->config = checkpoint_config;
          if (have_checkpoint && restore_resident_checkpoint(ctx->device) &&
              copy_resident_state_to_host(*ctx)) {
            *status = checkpoint;
            restore_host_checkpoint(*ctx, checkpoint_host);
            final_code = kMozymeScfCpuBoundary;
            status->ready = 0;
          }
          break;
        }

        ResidentControlSnapshot control{};
        if (!compute_resident_control_after_iteration(*ctx, false,
                                                      &control)) {
          final_code = kMozymeScfCpuBoundary;
          status->ready = 0;
          break;
        }
        publish_resident_control_to_status(status, control);
        const ResidentReturnDecision return_decision = control.decision;
        if (return_decision == ResidentReturnDecision::CompleteAndPublish) {
          const bool needs_final_reorth =
              (ctx->config.flags & kMozymeScfFlagFinalReorth) != 0;
          MozymeScfStatus final_status = *status;
          if (!publish_resident_stage_status_from_gpu(*ctx, &final_status) ||
              !resident_stage_status_complete(final_status,
                                              ctx->config.current_iter) ||
              !publish_resident_iteration_outputs_from_gpu(*ctx,
                                                           &final_status) ||
              (needs_final_reorth &&
               !apply_final_reorth_on_gpu(*ctx, &final_status,
                                          &accumulated_ms)) ||
              !copy_resident_state_to_host(*ctx)) {
            final_code = kMozymeScfNotReady;
          } else {
            ResidentControlSnapshot final_control{};
            if (copy_resident_control_snapshot_from_gpu(*ctx, &final_control,
                                                        false)) {
              publish_resident_control_to_status(&final_status,
                                                 final_control);
              *status = final_status;
              final_code = kMozymeScfSuccess;
            } else {
              final_code = kMozymeScfNotReady;
            }
          }
          break;
        }
        if (return_decision == ResidentReturnDecision::CpuBoundary) {
          final_code = kMozymeScfCpuBoundary;
          status->ready = 0;
          break;
        }
        if (return_decision == ResidentReturnDecision::IterationExhausted) {
          final_code = kMozymeScfUnsupported;
          status->ready = 0;
          break;
        }
        if (return_decision == ResidentReturnDecision::StageFailed) {
          final_code = kMozymeScfUnsupported;
          status->ready = 0;
          break;
        }
        if (!save_resident_checkpoint(ctx->device)) {
          if (copy_resident_state_to_host(*ctx)) {
            final_code = kMozymeScfCpuBoundary;
          } else {
            final_code = kMozymeScfNotReady;
          }
          status->ready = 0;
          break;
        }
        advance_config_after_resident_iteration(*ctx, *status, control);
        checkpoint = *status;
        checkpoint_config = ctx->config;
        checkpoint_host = capture_host_checkpoint(*ctx);
        have_checkpoint = true;
      }
    }
  }
  if (resident_stage_profile_enabled()) report_resident_stage_profile(*ctx);
#endif

  status->code = final_code;
  publish_pls_runtime_status(status, ctx);
  const bool preserve_resident_cosmo_cg_status =
      status->cosmo_enabled != 0 && status->cosmo_cg_control_resident != 0;
  if (!preserve_resident_cosmo_cg_status) {
    publish_cosmo_status(status, ctx);
  }
  status->strict_resident_host_syncs =
      ctx ? ctx->strict_resident_host_syncs : 0;
  status->strict_resident_control_polls =
      ctx ? ctx->strict_resident_control_polls : 0;
  if (final_code != kMozymeScfSuccess) status->ready = 0;
  return final_code;
}

extern "C" int mopac_cuda_mozyme_scf_destroy(void *context) {
  delete static_cast<MozymeScfContext *>(context);
  return kMozymeScfSuccess;
}

extern "C" int mopac_cuda_mozyme_scf_status(void *context,
                                             MozymeScfStatus *status) {
  auto *ctx = static_cast<MozymeScfContext *>(context);
  if (!status) return kMozymeScfBadArgument;

  fill_status(status, ctx, ctx && ctx->state_registered ? kMozymeScfSuccess
                                                        : kMozymeScfNotReady);
  return ctx && ctx->state_registered ? kMozymeScfSuccess : kMozymeScfNotReady;
}

extern "C" int mopac_cuda_mozyme_setupk(int natoms, int nocc, int icocc_dim,
                                         const int *ncf, const int *nncf,
                                         const int *icocc, int *kopt,
                                         double *wall_ms) {
  if (wall_ms) *wall_ms = 0.0;
  if (natoms <= 0 || nocc <= 0 || icocc_dim <= 0 || !ncf || !nncf ||
      !icocc || !kopt || !wall_ms) {
    return kMozymeScfBadArgument;
  }

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<int> d_ncf;
  DeviceBuffer<int> d_nncf;
  DeviceBuffer<int> d_icocc;
  DeviceBuffer<int> d_flags;
  DeviceBuffer<int> d_kopt;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    const std::size_t natoms_count = static_cast<std::size_t>(natoms);
    const std::size_t nocc_count = static_cast<std::size_t>(nocc);
    const std::size_t icocc_count = static_cast<std::size_t>(icocc_dim);
    if (!d_ncf.upload(ncf, nocc_count)) break;
    if (!d_nncf.upload(nncf, nocc_count)) break;
    if (!d_icocc.upload(icocc, icocc_count)) break;
    if (!d_flags.resize(natoms_count)) break;
    if (!d_kopt.resize(natoms_count)) break;
    if (!cuda_context_ok(
            cudaMemset(d_flags.ptr, 0, natoms_count * sizeof(int)),
            "setupk flags memset")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemset(d_kopt.ptr, 0, natoms_count * sizeof(int)),
            "setupk kopt memset")) {
      break;
    }
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "setupk create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "setupk create stop event")) break;

    constexpr int kThreads = 128;
    if (!cuda_context_ok(cudaEventRecord(start), "setupk start event")) break;
    mozyme_setupk_mark_kernel<<<nocc, kThreads>>>(
        natoms, nocc, icocc_dim, d_ncf.ptr, d_nncf.ptr, d_icocc.ptr,
        d_flags.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "setupk mark kernel")) break;
    mozyme_setupk_compress_kernel<<<1, 1>>>(natoms, d_flags.ptr, d_kopt.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "setupk compress kernel")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(kopt, d_kopt.ptr, natoms_count * sizeof(int),
                            cudaMemcpyDeviceToHost),
            "setupk kopt copy")) {
      break;
    }
    if (!cuda_context_ok(cudaEventRecord(stop), "setupk stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "setupk synchronize")) break;

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "setupk elapsed time")) break;
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_isitsc(
    double escf, double selcon, double emin, double ovmax, double energy_diff,
    int niter, int itrmax, int *iemin, int *iemax, int *scf1, double *escf0,
    int *okscf, int *iscf_out, double *wall_ms) {
  if (!iemin || !iemax || !scf1 || !escf0 || !okscf || !iscf_out ||
      !wall_ms || selcon < 0.0 || itrmax < 0) {
    return kMozymeScfBadArgument;
  }
  *wall_ms = 0.0;

#ifndef __CUDACC__
  ignore_no_cuda_only(escf, emin, ovmax, energy_diff, niter);
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<int> d_iemin;
  DeviceBuffer<int> d_iemax;
  DeviceBuffer<int> d_scf1;
  DeviceBuffer<int> d_okscf;
  DeviceBuffer<int> d_iscf_out;
  DeviceBuffer<double> d_escf0;
  DeviceBuffer<double> d_energy;
  DeviceBuffer<double> d_scalars;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    if (!d_iemin.upload(iemin, 1)) break;
    if (!d_iemax.upload(iemax, 1)) break;
    if (!d_scf1.upload(scf1, 1)) break;
    if (!d_escf0.upload(escf0, 10)) break;
    if (!d_energy.upload(&escf, 1)) break;
    if (!d_scalars.resize(kIsitscDoubleCount)) break;
    if (!d_okscf.resize(1)) break;
    if (!d_iscf_out.resize(1)) break;
    if (!cuda_context_ok(cudaMemset(d_okscf.ptr, 0, sizeof(int)),
                         "isitsc okscf memset")) break;
    if (!cuda_context_ok(cudaMemset(d_iscf_out.ptr, 0, sizeof(int)),
                         "isitsc iscf memset")) break;
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "isitsc create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "isitsc create stop event")) break;

    if (!cuda_context_ok(cudaEventRecord(start), "isitsc start event")) break;
    mozyme_isitsc_kernel<<<1, 1>>>(
        d_energy.ptr, 0, 1.0, 0.0, escf - energy_diff, selcon, emin, ovmax,
        niter, itrmax, d_iemin.ptr, d_iemax.ptr, d_scf1.ptr, d_escf0.ptr,
        d_scalars.ptr, d_okscf.ptr, d_iscf_out.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "isitsc kernel")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(iemin, d_iemin.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "isitsc iemin copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(iemax, d_iemax.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "isitsc iemax copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(scf1, d_scf1.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "isitsc scf1 copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(escf0, d_escf0.ptr, 10 * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "isitsc escf history copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(okscf, d_okscf.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "isitsc okscf copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(iscf_out, d_iscf_out.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "isitsc iscf copy")) break;
    if (!cuda_context_ok(cudaEventRecord(stop), "isitsc stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "isitsc synchronize")) break;

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "isitsc elapsed time")) break;
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_cnvgz(int mpack, int norbs,
                                        int use_three_point, int niter,
                                        const int *idiag, double *pnew,
                                        double *pold, double *p1, double *p2,
                                        double *p3, double *pmax_out,
                                        double *rms_out, double *wall_ms) {
  if (mpack <= 0 || norbs <= 0 || !idiag || !pnew || !pold || !p1 || !p2 ||
      !p3 || !pmax_out || !rms_out || !wall_ms) {
    return kMozymeScfBadArgument;
  }

#ifndef __CUDACC__
  ignore_no_cuda_only(use_three_point, niter);
  return kMozymeScfUnsupported;
#else
  constexpr int kThreads = 256;
  const int matrix_blocks = ceil_div(mpack, kThreads);
  const int diag_blocks = ceil_div(norbs, kThreads);
  const std::size_t mpack_count = static_cast<std::size_t>(mpack);
  const std::size_t norbs_count = static_cast<std::size_t>(norbs);

  DeviceBuffer<double> d_pnew;
  DeviceBuffer<double> d_pold;
  DeviceBuffer<double> d_p1;
  DeviceBuffer<double> d_p2;
  DeviceBuffer<double> d_p3;
  DeviceBuffer<double> d_diag_new;
  DeviceBuffer<double> d_diag_old;
  DeviceBuffer<double> d_candidate;
  DeviceBuffer<double> d_block_max;
  DeviceBuffer<double> d_block_sumsq;
  DeviceBuffer<double> d_block_faca;
  DeviceBuffer<double> d_block_facb;
  DeviceBuffer<double> d_cnvgz_sums;
  DeviceBuffer<int> d_idiag;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    if (!d_pnew.upload(pnew, mpack_count)) break;
    if (!d_pold.upload(pold, mpack_count)) break;
    if (!d_p1.upload(p1, norbs_count)) break;
    if (!d_idiag.upload(idiag, norbs_count)) break;
    if (!d_p2.resize(norbs_count)) break;
    if (!d_p3.resize(norbs_count)) break;
    if (!d_diag_new.resize(norbs_count)) break;
    if (!d_diag_old.resize(norbs_count)) break;
    if (!d_candidate.resize(mpack_count)) break;
    if (!d_block_max.resize(static_cast<std::size_t>(matrix_blocks))) break;
    if (!d_block_sumsq.resize(static_cast<std::size_t>(matrix_blocks))) break;
    if (!d_block_faca.resize(static_cast<std::size_t>(diag_blocks))) break;
    if (!d_block_facb.resize(static_cast<std::size_t>(diag_blocks))) break;
    if (!d_cnvgz_sums.resize(kCnvgzControlCount)) break;
    if (!cuda_context_ok(cudaEventCreate(&start), "cnvgz create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop), "cnvgz create stop event")) break;

    if (!cuda_context_ok(cudaEventRecord(start), "cnvgz start event")) break;
    mozyme_cnvgz_diag_kernel<<<diag_blocks, kThreads>>>(
        norbs, mpack, d_idiag.ptr, d_pnew.ptr, d_pold.ptr, d_diag_new.ptr,
        d_diag_old.ptr, nullptr);
    mozyme_cnvgz_diff_kernel<<<matrix_blocks, kThreads,
                               2 * kThreads * sizeof(double)>>>(
        mpack, d_pnew.ptr, d_pold.ptr, d_block_max.ptr, d_block_sumsq.ptr,
        nullptr);
    if (!cuda_context_ok(cudaGetLastError(), "cnvgz diff kernels")) break;
    mozyme_cnvgz_diff_reduce_kernel<<<1, 1>>>(
        matrix_blocks, d_block_max.ptr, d_block_sumsq.ptr, d_cnvgz_sums.ptr,
        nullptr);
    if (!cuda_context_ok(cudaGetLastError(), "cnvgz reduce kernel")) break;

    const bool compute_factor = (use_three_point != 0 && niter % 3 == 0);
    if (compute_factor) {
      mozyme_cnvgz_factor_kernel<<<diag_blocks, kThreads,
                                   2 * kThreads * sizeof(double)>>>(
          norbs, d_diag_old.ptr, d_p1.ptr, d_diag_new.ptr, d_block_faca.ptr,
          d_block_facb.ptr, nullptr, use_three_point, niter);
      if (!cuda_context_ok(cudaGetLastError(), "cnvgz factor kernel")) break;
      mozyme_cnvgz_factor_reduce_kernel<<<1, 1>>>(
          diag_blocks, d_block_faca.ptr, d_block_facb.ptr, d_cnvgz_sums.ptr,
          nullptr);
      if (!cuda_context_ok(cudaGetLastError(),
                           "cnvgz factor reduce kernel")) break;
    }
    mozyme_cnvgz_finalize_kernel<<<1, 1>>>(
        mpack, compute_factor, d_cnvgz_sums.ptr, nullptr, nullptr,
        use_three_point, niter);
    if (!cuda_context_ok(cudaGetLastError(), "cnvgz finalize kernel")) break;

    mozyme_cnvgz_candidate_kernel<<<matrix_blocks, kThreads>>>(
        mpack, d_cnvgz_sums.ptr, d_pnew.ptr, d_pold.ptr, d_candidate.ptr,
        nullptr, use_three_point);
    if (use_three_point != 0 && niter > 3) {
      mozyme_cnvgz_damp_kernel<<<diag_blocks, kThreads>>>(
          norbs, mpack, d_cnvgz_sums.ptr, d_idiag.ptr, d_diag_new.ptr,
          d_diag_old.ptr, d_candidate.ptr, nullptr, use_three_point, niter);
    }
    if (!cuda_context_ok(cudaGetLastError(), "cnvgz candidate kernels")) break;

    if (!cuda_context_ok(
            cudaMemcpyAsync(pnew, d_candidate.ptr,
                            mpack_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "cnvgz pnew copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(pold, d_candidate.ptr,
                            mpack_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "cnvgz pold copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(p1, d_diag_old.ptr,
                            norbs_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "cnvgz p1 copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(p2, d_diag_old.ptr,
                            norbs_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "cnvgz p2 copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(p3, d_diag_new.ptr,
                            norbs_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "cnvgz p3 copy")) break;
    if (!cuda_context_ok(cudaEventRecord(stop), "cnvgz stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop), "cnvgz synchronize")) break;

    double cnvgz_totals[kCnvgzControlCount] = {};
    if (!cuda_context_ok(
            cudaMemcpy(cnvgz_totals, d_cnvgz_sums.ptr,
                       sizeof(cnvgz_totals), cudaMemcpyDeviceToHost),
            "cnvgz totals copy")) {
      break;
    }
    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "cnvgz elapsed time")) break;
    *pmax_out = cnvgz_totals[kCnvgzPmax];
    *rms_out = cnvgz_totals[kCnvgzDensityRms];
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_helecz(int mpack, int numat,
                                         const int *iorbs, const int *nijbo,
                                         const double *p, const double *h,
                                         const double *f, double *energy,
                                         double *wall_ms) {
  if (mpack <= 0 || numat <= 0 || !iorbs || !nijbo || !p || !h || !f ||
      !energy || !wall_ms) {
    return kMozymeScfBadArgument;
  }

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  const std::size_t numat_count = static_cast<std::size_t>(numat);
  const std::size_t mpack_count = static_cast<std::size_t>(mpack);
  const std::size_t nijbo_count = numat_count * numat_count;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_nijbo;
  DeviceBuffer<double> d_p;
  DeviceBuffer<double> d_h;
  DeviceBuffer<double> d_f;
  DeviceBuffer<double> d_atom_sums;
  DeviceBuffer<double> d_atom_diag_sums;
  DeviceBuffer<double> d_energy_sums;
  DeviceBuffer<int> d_valid;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    if (!d_iorbs.upload(iorbs, numat_count)) break;
    if (!d_nijbo.upload(nijbo, nijbo_count)) break;
    if (!d_p.upload(p, mpack_count)) break;
    if (!d_h.upload(h, mpack_count)) break;
    if (!d_f.upload(f, mpack_count)) break;
    if (!d_atom_sums.resize(numat_count)) break;
    if (!d_atom_diag_sums.resize(numat_count)) break;
    if (!d_energy_sums.resize(3)) break;
    int host_ok_init[1] = {1};
    if (!d_valid.upload(host_ok_init, 1)) break;
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "helecz create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "helecz create stop event")) break;

    constexpr int kThreads = 128;
    if (!cuda_context_ok(cudaEventRecord(start), "helecz start event")) break;
    mozyme_helecz_kernel<<<numat, kThreads>>>(
        numat, mpack, d_iorbs.ptr, d_nijbo.ptr, d_p.ptr, d_h.ptr, d_f.ptr,
        d_atom_sums.ptr, d_atom_diag_sums.ptr, d_valid.ptr, nullptr);
    if (!cuda_context_ok(cudaGetLastError(), "helecz atom kernel")) break;
    mozyme_helecz_reduce_kernel<<<1, kThreads, 2 * kThreads * sizeof(double)>>>(
        numat, d_atom_sums.ptr, d_atom_diag_sums.ptr, d_valid.ptr,
        d_energy_sums.ptr, nullptr);
    if (!cuda_context_ok(cudaGetLastError(), "helecz reduce kernel")) break;
    if (!cuda_context_ok(cudaEventRecord(stop), "helecz stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "helecz synchronize")) break;

    double host_totals[3] = {0.0, 0.0, 0.0};
    int host_ok = 0;
    if (!cuda_context_ok(
            cudaMemcpy(&host_ok, d_valid.ptr, sizeof(int),
                       cudaMemcpyDeviceToHost),
            "helecz validity copy")) {
      break;
    }
    if (host_ok != 1) break;
    if (!cuda_context_ok(
            cudaMemcpy(host_totals, d_energy_sums.ptr, sizeof(host_totals),
                       cudaMemcpyDeviceToHost),
            "helecz totals copy")) {
      break;
    }
    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "helecz elapsed time")) break;
    *energy = host_totals[2];
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_eimp(int mpack, int numat, const int *iorbs,
                                       const int *nijbo, const double *f,
                                       double *p, int *updated_pairs,
                                       double *wall_ms) {
  if (mpack <= 0 || numat <= 0 || !iorbs || !nijbo || !f || !p ||
      !updated_pairs || !wall_ms) {
    return kMozymeScfBadArgument;
  }

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  const std::size_t numat_count = static_cast<std::size_t>(numat);
  const std::size_t mpack_count = static_cast<std::size_t>(mpack);
  const std::size_t nijbo_count = numat_count * numat_count;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_nijbo;
  DeviceBuffer<int> d_updates;
  DeviceBuffer<double> d_f;
  DeviceBuffer<double> d_p;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    if (!d_iorbs.upload(iorbs, numat_count)) break;
    if (!d_nijbo.upload(nijbo, nijbo_count)) break;
    if (!d_f.upload(f, mpack_count)) break;
    if (!d_p.upload(p, mpack_count)) break;
    int host_status[3] = {0, 1, 0};
    if (!d_updates.upload(host_status, 3)) break;
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "eimp create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "eimp create stop event")) break;

    const dim3 block(16, 16);
    const dim3 grid(ceil_div(numat, block.x), ceil_div(numat, block.y));
    if (!cuda_context_ok(cudaEventRecord(start), "eimp start event")) break;
    mozyme_eimp_kernel<<<grid, block>>>(numat, mpack, d_iorbs.ptr,
                                        d_nijbo.ptr, d_f.ptr, d_p.ptr,
                                        d_updates.ptr, d_updates.ptr + 1,
                                        d_updates.ptr + 2, nullptr);
    if (!cuda_context_ok(cudaGetLastError(), "eimp kernel")) break;
    if (!cuda_context_ok(cudaEventRecord(stop), "eimp stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop), "eimp synchronize")) break;

    if (!cuda_context_ok(
            cudaMemcpy(host_status, d_updates.ptr, sizeof(host_status),
                       cudaMemcpyDeviceToHost),
            "eimp status copy")) break;
    if (host_status[1] != 1 || host_status[0] != host_status[2]) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpy(p, d_p.ptr, mpack_count * sizeof(double),
                       cudaMemcpyDeviceToHost),
            "eimp density copy")) break;
    *updated_pairs = host_status[0];

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "eimp elapsed time")) break;
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_diagg1_aocc(
    int nocc, int icocc_dim, int cocc_dim, int numat, const int *ncf,
    const int *nncf, const int *ncocc, const int *icocc, const int *iorbs,
    const double *cocc, double *aocc, int *updated_terms, double *wall_ms) {
  if (nocc <= 0 || icocc_dim <= 0 || cocc_dim <= 0 || numat <= 0 || !ncf ||
      !nncf || !ncocc || !icocc || !iorbs || !cocc || !aocc ||
      !updated_terms || !wall_ms) {
    return kMozymeScfBadArgument;
  }

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<int> d_ncf;
  DeviceBuffer<int> d_nncf;
  DeviceBuffer<int> d_ncocc;
  DeviceBuffer<int> d_icocc;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_updated;
  DeviceBuffer<double> d_cocc;
  DeviceBuffer<double> d_aocc;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    if (!d_ncf.upload(ncf, static_cast<std::size_t>(nocc))) break;
    if (!d_nncf.upload(nncf, static_cast<std::size_t>(nocc))) break;
    if (!d_ncocc.upload(ncocc, static_cast<std::size_t>(nocc))) break;
    if (!d_icocc.upload(icocc, static_cast<std::size_t>(icocc_dim))) break;
    if (!d_iorbs.upload(iorbs, static_cast<std::size_t>(numat))) break;
    if (!d_cocc.upload(cocc, static_cast<std::size_t>(cocc_dim))) break;
    if (!d_aocc.upload(aocc, static_cast<std::size_t>(icocc_dim))) break;
    if (!d_updated.resize(1)) break;
    if (!cuda_context_ok(cudaMemset(d_updated.ptr, 0, sizeof(int)),
                         "diagg1 aocc updated memset")) break;
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "diagg1 aocc create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "diagg1 aocc create stop event")) break;

    constexpr int kThreads = 128;
    if (!cuda_context_ok(cudaEventRecord(start),
                         "diagg1 aocc start event")) break;
    mozyme_diagg1_aocc_kernel<<<nocc, kThreads>>>(
        nocc, icocc_dim, cocc_dim, numat, d_ncf.ptr, d_nncf.ptr,
        d_ncocc.ptr, d_icocc.ptr, d_iorbs.ptr, d_cocc.ptr, d_aocc.ptr,
        d_updated.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "diagg1 aocc kernel")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(aocc, d_aocc.ptr,
                            static_cast<std::size_t>(icocc_dim) *
                                sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 aocc cache copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(updated_terms, d_updated.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg1 aocc update count copy")) break;
    if (!cuda_context_ok(cudaEventRecord(stop),
                         "diagg1 aocc stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "diagg1 aocc synchronize")) break;

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "diagg1 aocc elapsed time")) break;
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_diagg1_avir(
    int nvir, int icvir_dim, int cvir_dim, int numat, const int *nce,
    const int *nnce, const int *ncvir, const int *icvir, const int *iorbs,
    const double *cvir, double *avir_cache, int *updated_terms,
    double *wall_ms) {
  if (nvir <= 0 || icvir_dim <= 0 || cvir_dim <= 0 || numat <= 0 || !nce ||
      !nnce || !ncvir || !icvir || !iorbs || !cvir || !avir_cache ||
      !updated_terms || !wall_ms) {
    return kMozymeScfBadArgument;
  }

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<int> d_nce;
  DeviceBuffer<int> d_nnce;
  DeviceBuffer<int> d_ncvir;
  DeviceBuffer<int> d_icvir;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_updated;
  DeviceBuffer<double> d_cvir;
  DeviceBuffer<double> d_avir_cache;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    if (!d_nce.upload(nce, static_cast<std::size_t>(nvir))) break;
    if (!d_nnce.upload(nnce, static_cast<std::size_t>(nvir))) break;
    if (!d_ncvir.upload(ncvir, static_cast<std::size_t>(nvir))) break;
    if (!d_icvir.upload(icvir, static_cast<std::size_t>(icvir_dim))) break;
    if (!d_iorbs.upload(iorbs, static_cast<std::size_t>(numat))) break;
    if (!d_cvir.upload(cvir, static_cast<std::size_t>(cvir_dim))) break;
    if (!d_avir_cache.upload(avir_cache, static_cast<std::size_t>(icvir_dim))) {
      break;
    }
    if (!d_updated.resize(1)) break;
    if (!cuda_context_ok(cudaMemset(d_updated.ptr, 0, sizeof(int)),
                         "diagg1 avir updated memset")) break;
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "diagg1 avir create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "diagg1 avir create stop event")) break;

    constexpr int kThreads = 128;
    if (!cuda_context_ok(cudaEventRecord(start),
                         "diagg1 avir start event")) break;
    mozyme_diagg1_avir_kernel<<<nvir, kThreads>>>(
        nvir, icvir_dim, cvir_dim, numat, d_nce.ptr, d_nnce.ptr,
        d_ncvir.ptr, d_icvir.ptr, d_iorbs.ptr, d_cvir.ptr,
        d_avir_cache.ptr, d_updated.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "diagg1 avir kernel")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(avir_cache, d_avir_cache.ptr,
                            static_cast<std::size_t>(icvir_dim) *
                                sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 avir cache copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(updated_terms, d_updated.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg1 avir update count copy")) break;
    if (!cuda_context_ok(cudaEventRecord(stop),
                         "diagg1 avir stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "diagg1 avir synchronize")) break;

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "diagg1 avir elapsed time")) break;
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_diagg1_construct(
    int nocc, int nvir, int numat, int norbs, int mpack, int icocc_dim,
    int icvir_dim, int cocc_dim, int cvir_dim, int fmo_dim, int idiagg,
    int mydisp, const double *fao, const double *p, const int *nfirst,
    const int *nlast, const int *ncf, const int *nce, const int *nncf,
    const int *nnce, const int *ncocc, const int *ncvir, const int *icocc,
    const int *icvir, const int *iorbs, const int *nijbo,
    const double *cocc, const double *cvir, double cutoff, double flim,
    double oldlim_in, double safety_in, int nf_in, double *eigs,
    double *eigv, int *nfmo, int *ifmo, double *fmo, int *nij_io,
    int *ijc_out, int *nf_out, double *sumt_out, double *tiny_out,
    double *fref_out, double *oldlim_out, double *safety_out,
    double *wall_ms) {
  if (nocc <= 0 || nvir <= 0 || numat <= 0 || norbs <= 0 || mpack <= 0 ||
      icocc_dim <= 0 || icvir_dim <= 0 || cocc_dim <= 0 || cvir_dim <= 0 ||
      fmo_dim <= 0 || !fao || !p || !nfirst || !nlast || !ncf || !nce ||
      !nncf || !nnce || !ncocc || !ncvir || !icocc || !icvir || !iorbs ||
      !nijbo || !cocc || !cvir || !eigs || !eigv || !nfmo || !ifmo ||
      !fmo || !nij_io || !ijc_out || !nf_out || !sumt_out || !tiny_out ||
      !fref_out || !oldlim_out || !safety_out || !wall_ms) {
    return kMozymeScfBadArgument;
  }
  const int nij_capacity = *nij_io;
  if (nij_capacity <= 0 || nij_capacity > fmo_dim) return kMozymeScfBadArgument;
  *wall_ms = 0.0;

#ifndef __CUDACC__
  ignore_no_cuda_only(idiagg, mydisp, cutoff, flim, oldlim_in, safety_in,
                      nf_in);
  return kMozymeScfUnsupported;
#else
  const std::size_t nocc_count = static_cast<std::size_t>(nocc);
  const std::size_t nvir_count = static_cast<std::size_t>(nvir);
  const std::size_t numat_count = static_cast<std::size_t>(numat);
  const std::size_t norbs_count = static_cast<std::size_t>(norbs);
  const std::size_t mpack_count = static_cast<std::size_t>(mpack);
  const std::size_t icocc_count = static_cast<std::size_t>(icocc_dim);
  const std::size_t icvir_count = static_cast<std::size_t>(icvir_dim);
  const std::size_t cocc_count = static_cast<std::size_t>(cocc_dim);
  const std::size_t cvir_count = static_cast<std::size_t>(cvir_dim);
  const std::size_t fmo_count = static_cast<std::size_t>(fmo_dim);
  const std::size_t nijbo_count = numat_count * numat_count;

  DeviceBuffer<double> d_fao;
  DeviceBuffer<double> d_p;
  DeviceBuffer<double> d_cocc;
  DeviceBuffer<double> d_cvir;
  DeviceBuffer<double> d_eigs;
  DeviceBuffer<double> d_eigv;
  DeviceBuffer<double> d_fmo;
  DeviceBuffer<double> d_aocc;
  DeviceBuffer<double> d_avir;
  DeviceBuffer<double> d_aov;
  DeviceBuffer<double> d_ws;
  DeviceBuffer<double> d_sumt;
  DeviceBuffer<double> d_tiny;
  DeviceBuffer<double> d_fref;
  DeviceBuffer<double> d_oldlim;
  DeviceBuffer<double> d_safety;
  DeviceBuffer<int> d_nfirst;
  DeviceBuffer<int> d_nlast;
  DeviceBuffer<int> d_ncf;
  DeviceBuffer<int> d_nce;
  DeviceBuffer<int> d_nncf;
  DeviceBuffer<int> d_nnce;
  DeviceBuffer<int> d_ncocc;
  DeviceBuffer<int> d_ncvir;
  DeviceBuffer<int> d_icocc;
  DeviceBuffer<int> d_icvir;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_nijbo;
  DeviceBuffer<int> d_nfmo;
  DeviceBuffer<int> d_ifmo;
  DeviceBuffer<int> d_latoms;
  DeviceBuffer<int> d_nij;
  DeviceBuffer<int> d_ijc;
  DeviceBuffer<int> d_nf;
  DeviceBuffer<int> d_ok;
  DeviceBuffer<double> d_avir_entry;
  DeviceBuffer<int> d_counts;
  DeviceBuffer<int> d_offsets;
  DeviceBuffer<int> d_work_ints;
  DeviceBuffer<double> d_work_scalars;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  // The parallel kernels index cached ifmo entries by the nfmo prefix sum,
  // which only matches the CPU bookkeeping when mydisp is zero.
  if (mydisp != 0) return kMozymeScfUnsupported;

  do {
    if (!d_fao.upload(fao, mpack_count)) break;
    if (!d_p.upload(p, mpack_count)) break;
    if (!d_nfirst.upload(nfirst, numat_count)) break;
    if (!d_nlast.upload(nlast, numat_count)) break;
    if (!d_ncf.upload(ncf, nocc_count)) break;
    if (!d_nce.upload(nce, nvir_count)) break;
    if (!d_nncf.upload(nncf, nocc_count)) break;
    if (!d_nnce.upload(nnce, nvir_count)) break;
    if (!d_ncocc.upload(ncocc, nocc_count)) break;
    if (!d_ncvir.upload(ncvir, nvir_count)) break;
    if (!d_icocc.upload(icocc, icocc_count)) break;
    if (!d_icvir.upload(icvir, icvir_count)) break;
    if (!d_iorbs.upload(iorbs, numat_count)) break;
    if (!d_nijbo.upload(nijbo, nijbo_count)) break;
    if (!d_cocc.upload(cocc, cocc_count)) break;
    if (!d_cvir.upload(cvir, cvir_count)) break;
    if (!d_eigs.upload(eigs, nocc_count)) break;
    if (!d_eigv.upload(eigv, nvir_count)) break;
    if (!d_nfmo.upload(nfmo, nvir_count)) break;
    if (!d_ifmo.upload(ifmo, 2 * fmo_count)) break;
    if (!d_fmo.upload(fmo, fmo_count)) break;
    if (!d_aocc.resize(icocc_count)) break;
    if (!d_avir.resize(norbs_count)) break;
    if (!d_aov.resize(numat_count)) break;
    if (!d_ws.resize(norbs_count)) break;
    if (!d_latoms.resize(numat_count)) break;
    if (!d_nij.resize(1)) break;
    if (!d_ijc.resize(1)) break;
    if (!d_nf.resize(1)) break;
    if (!d_sumt.resize(1)) break;
    if (!d_tiny.resize(1)) break;
    if (!d_fref.resize(1)) break;
    if (!d_oldlim.resize(1)) break;
    if (!d_safety.resize(1)) break;
    if (!d_ok.resize(1)) break;
    if (!d_avir_entry.resize(icvir_count)) break;
    if (!d_counts.resize(nvir_count)) break;
    if (!d_offsets.resize(nvir_count + 1)) break;
    if (!d_work_ints.resize(kDiaggWorkIntCount)) break;
    if (!d_work_scalars.resize(kDiaggWorkDoubleCount)) break;
    if (!cuda_context_ok(cudaMemset(d_ok.ptr, 0, sizeof(int)),
                         "diagg1 construct ok memset")) break;
    if (!cuda_context_ok(
            cudaMemset(d_work_ints.ptr, 0, kDiaggWorkIntCount * sizeof(int)),
            "diagg1 construct work ints memset")) break;
    if (!cuda_context_ok(cudaMemset(d_work_scalars.ptr, 0,
                                    kDiaggWorkDoubleCount * sizeof(double)),
                         "diagg1 construct work scalars memset")) break;
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "diagg1 construct create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "diagg1 construct create stop event")) break;

    if (!cuda_context_ok(cudaEventRecord(start),
                         "diagg1 construct start event")) break;
    {
      const int occ_blocks =
          (nocc * 32 + kDiaggBlockThreads - 1) / kDiaggBlockThreads;
      const int vir_blocks =
          (nvir * 32 + kDiaggBlockThreads - 1) / kDiaggBlockThreads;
      mozyme_lmo_entry_norms_kernel<<<occ_blocks, kDiaggBlockThreads>>>(
          nocc, numat, icocc_dim, cocc_dim, d_nncf.ptr, d_ncf.ptr,
          d_icocc.ptr, d_ncocc.ptr, d_iorbs.ptr, d_cocc.ptr, d_aocc.ptr,
          d_work_ints.ptr + kDiaggWorkIntError, nullptr);
      mozyme_lmo_entry_norms_kernel<<<vir_blocks, kDiaggBlockThreads>>>(
          nvir, numat, icvir_dim, cvir_dim, d_nnce.ptr, d_nce.ptr,
          d_icvir.ptr, d_ncvir.ptr, d_iorbs.ptr, d_cvir.ptr,
          d_avir_entry.ptr, d_work_ints.ptr + kDiaggWorkIntError, nullptr);
      if (diagg_debug_enabled()) {
        std::fprintf(stderr,
                     "[DIAGG DEBUG] standalone diagg1 nocc=%d nvir=%d numat=%d "
                     "norbs=%d mpack=%d icocc=%d icvir=%d cocc=%d cvir=%d "
                     "fmo_dim=%d capacity=%d idiagg=%d\n",
                     nocc, nvir, numat, norbs, mpack, icocc_dim, icvir_dim,
                     cocc_dim, cvir_dim, fmo_dim, nij_capacity, idiagg);
      }
      diagg_debug_checkpoint("standalone diagg1 entry norms");

      DiaggVirtualArgs va{};
      va.nocc = nocc;
      va.nvir = nvir;
      va.numat = numat;
      va.norbs = norbs;
      va.mpack = mpack;
      va.icocc_dim = icocc_dim;
      va.icvir_dim = icvir_dim;
      va.cocc_dim = cocc_dim;
      va.cvir_dim = cvir_dim;
      va.fmo_dim = fmo_dim;
      va.idiagg = idiagg;
      va.capacity = nij_capacity;
      va.resident_control_scalars = nullptr;
      va.fao = d_fao.ptr;
      va.p = d_p.ptr;
      va.ncf = d_ncf.ptr;
      va.nce = d_nce.ptr;
      va.nncf = d_nncf.ptr;
      va.nnce = d_nnce.ptr;
      va.ncocc = d_ncocc.ptr;
      va.ncvir = d_ncvir.ptr;
      va.icocc = d_icocc.ptr;
      va.icvir = d_icvir.ptr;
      va.iorbs = d_iorbs.ptr;
      va.nijbo = d_nijbo.ptr;
      va.cocc = d_cocc.ptr;
      va.cvir = d_cvir.ptr;
      va.aocc = d_aocc.ptr;
      va.avir_entry = d_avir_entry.ptr;
      va.cutoff = cutoff;
      va.flim = flim;
      va.oldlim = oldlim_in;
      va.eigv = d_eigv.ptr;
      va.nfmo = d_nfmo.ptr;
      va.counts = d_counts.ptr;
      va.offsets = d_offsets.ptr;
      va.ifmo = d_ifmo.ptr;
      va.fmo = d_fmo.ptr;
      va.work_scalars = d_work_scalars.ptr;
      va.work_ints = d_work_ints.ptr;
      va.resident_control_ints = nullptr;

      va.fill = 0;
      mozyme_diagg1_virtual_kernel<<<nvir, kDiaggBlockThreads>>>(va);
      diagg_debug_checkpoint("standalone diagg1 count pass");
      mozyme_exclusive_scan_kernel<<<1, 1024>>>(nvir, nullptr, d_counts.ptr,
                                                d_offsets.ptr, nullptr);
      diagg_debug_checkpoint("standalone diagg1 scan");
      diagg_debug_dump_ints("standalone diagg1 total candidates",
                            d_offsets.ptr + nvir, 1);
      va.fill = 1;
      mozyme_diagg1_virtual_kernel<<<nvir, kDiaggBlockThreads>>>(va);
      diagg_debug_checkpoint("standalone diagg1 fill pass");
      mozyme_diagg1_occupied_eigs_kernel<<<nocc, kDiaggBlockThreads>>>(
          nocc, numat, mpack, icocc_dim, cocc_dim, d_fao.ptr, d_p.ptr,
          d_ncf.ptr, d_nncf.ptr, d_ncocc.ptr, d_icocc.ptr, d_iorbs.ptr,
          d_nijbo.ptr, d_cocc.ptr, d_aocc.ptr, cutoff, idiagg, d_eigs.ptr,
          d_work_ints.ptr, nullptr, nullptr);
      diagg_debug_checkpoint("standalone diagg1 occupied eigs");
      mozyme_diagg1_finalize_kernel<<<1, 1>>>(
          nvir, nij_capacity, idiagg, nf_in, safety_in, oldlim_in,
          d_offsets.ptr, d_nfmo.ptr, d_work_ints.ptr, d_work_scalars.ptr,
          d_nij.ptr, d_ijc.ptr, d_nf.ptr, d_sumt.ptr, d_tiny.ptr, d_fref.ptr,
          d_oldlim.ptr, d_safety.ptr, d_ok.ptr, nullptr, nullptr);
      diagg_debug_checkpoint("standalone diagg1 finalize");
      diagg_debug_dump_ints("standalone diagg1 work ints", d_work_ints.ptr,
                            kDiaggWorkIntCount);
      diagg_debug_dump_ints("standalone diagg1 nij/ok", d_nij.ptr, 1);
      diagg_debug_dump_ints("standalone diagg1 ok", d_ok.ptr, 1);
      diagg_debug_dump_doubles("standalone diagg1 sumt/tiny",
                               d_work_scalars.ptr, 2);
    }
    if (!cuda_context_ok(cudaGetLastError(),
                         "diagg1 construct kernels")) break;
    int ok_value = 0;
    if (!cuda_context_ok(
            cudaMemcpy(&ok_value, d_ok.ptr, sizeof(int),
                       cudaMemcpyDeviceToHost),
            "diagg1 construct ok copy")) {
      break;
    }
    if (ok_value != 1) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(nij_io, d_nij.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct nij copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(ijc_out, d_ijc.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct ijc copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(nf_out, d_nf.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct nf copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(sumt_out, d_sumt.ptr, sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct sumt copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(tiny_out, d_tiny.ptr, sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct tiny copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(fref_out, d_fref.ptr, sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct fref copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(oldlim_out, d_oldlim.ptr, sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct oldlim copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(safety_out, d_safety.ptr, sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct safety copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(eigs, d_eigs.ptr, nocc_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct eigs copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(eigv, d_eigv.ptr, nvir_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct eigv copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(nfmo, d_nfmo.ptr, nvir_count * sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct nfmo copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(ifmo, d_ifmo.ptr, 2 * fmo_count * sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct ifmo copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(fmo, d_fmo.ptr, fmo_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg1 construct fmo copy")) {
      break;
    }
    if (!cuda_context_ok(cudaEventRecord(stop),
                         "diagg1 construct stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "diagg1 construct synchronize")) break;

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "diagg1 construct elapsed time")) break;
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_diagg2_rotprep(
    int nij, int nocc, int nvir, const int *ifmo, const double *fmo,
    const double *eigs, const double *eigv, double shift, double rot_const,
    double tiny, double biglim, int *active, double *alpha, int *active_count,
    double *wall_ms) {
  if (nij < 0 || nocc <= 0 || nvir <= 0 || !ifmo || !fmo || !eigs || !eigv ||
      !active || !alpha || !active_count || !wall_ms) {
    return kMozymeScfBadArgument;
  }
  *active_count = 0;
  *wall_ms = 0.0;
  if (nij == 0) return kMozymeScfSuccess;

#ifndef __CUDACC__
  ignore_no_cuda_only(shift, rot_const, tiny, biglim);
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<int> d_ifmo;
  DeviceBuffer<int> d_active;
  DeviceBuffer<int> d_active_count;
  DeviceBuffer<double> d_fmo;
  DeviceBuffer<double> d_eigs;
  DeviceBuffer<double> d_eigv;
  DeviceBuffer<double> d_alpha;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    const auto wall_begin = std::chrono::steady_clock::now();
    const std::size_t nij_count = static_cast<std::size_t>(nij);
    if (!d_ifmo.upload(ifmo, 2 * nij_count)) break;
    if (!d_fmo.upload(fmo, nij_count)) break;
    if (!d_eigs.upload(eigs, static_cast<std::size_t>(nocc))) break;
    if (!d_eigv.upload(eigv, static_cast<std::size_t>(nvir))) break;
    if (!d_active.resize(nij_count)) break;
    if (!d_alpha.resize(nij_count)) break;
    if (!d_active_count.resize(1)) break;
    if (!cuda_context_ok(cudaMemset(d_active_count.ptr, 0, sizeof(int)),
                         "diagg2 rotprep active count memset")) break;
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "diagg2 rotprep create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "diagg2 rotprep create stop event")) break;

    constexpr int kThreads = 256;
    const int blocks = ceil_div(nij, kThreads);
    if (!cuda_context_ok(cudaEventRecord(start),
                         "diagg2 rotprep start event")) break;
    mozyme_diagg2_rotprep_kernel<<<blocks, kThreads>>>(
        nij, nocc, nvir, d_ifmo.ptr, d_fmo.ptr, d_eigs.ptr, d_eigv.ptr,
        shift, rot_const, tiny, biglim, d_active.ptr, d_alpha.ptr,
        d_active_count.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "diagg2 rotprep kernel")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(active, d_active.ptr, nij_count * sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotprep active copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(alpha, d_alpha.ptr, nij_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotprep alpha copy")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(active_count, d_active_count.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotprep active count copy")) break;
    if (!cuda_context_ok(cudaEventRecord(stop),
                         "diagg2 rotprep stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "diagg2 rotprep synchronize")) break;

    const auto wall_end = std::chrono::steady_clock::now();
    *wall_ms = std::chrono::duration<double, std::milli>(
                   wall_end - wall_begin)
                   .count();
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_diagg2_rotate(
    int nij, int nocc, int nvir, int numat, int norbs, int icocc_dim,
    int icvir_dim, int cocc_dim, int cvir_dim, const int *ifmo,
    const double *fmo, const double *eigs, const double *eigv,
    const int *nncf, int *ncf, const int *ncocc, int *icocc,
    const int *nnce, int *nce, const int *ncvir, int *icvir,
    const int *iorbs, double *cocc, double *cvir, double shift,
    double rot_const, double tiny, double biglim, double thresh, int retry,
    double *sumb_out, int *nrej_out, double *wall_ms) {
  if (nij < 0 || nocc <= 0 || nvir <= 0 || numat <= 0 || norbs <= 0 ||
      icocc_dim <= 0 || icvir_dim <= 0 || cocc_dim <= 0 || cvir_dim <= 0 ||
      !ifmo || !fmo || !eigs || !eigv || !nncf || !ncf || !ncocc ||
      !icocc || !nnce || !nce || !ncvir || !icvir || !iorbs || !cocc ||
      !cvir || !sumb_out || !nrej_out || !wall_ms) {
    return kMozymeScfBadArgument;
  }
  *sumb_out = 0.0;
  *nrej_out = 0;
  *wall_ms = 0.0;
  if (nij == 0) return kMozymeScfSuccess;

#ifndef __CUDACC__
  ignore_no_cuda_only(shift, rot_const, tiny, biglim, thresh, retry);
  return kMozymeScfUnsupported;
#else
  const std::size_t nij_count = static_cast<std::size_t>(nij);
  const std::size_t nocc_count = static_cast<std::size_t>(nocc);
  const std::size_t nvir_count = static_cast<std::size_t>(nvir);
  const std::size_t numat_count = static_cast<std::size_t>(numat);
  const std::size_t norbs_count = static_cast<std::size_t>(norbs);
  const std::size_t icocc_count = static_cast<std::size_t>(icocc_dim);
  const std::size_t icvir_count = static_cast<std::size_t>(icvir_dim);
  const std::size_t cocc_count = static_cast<std::size_t>(cocc_dim);
  const std::size_t cvir_count = static_cast<std::size_t>(cvir_dim);

  DeviceBuffer<int> d_ifmo;
  DeviceBuffer<int> d_nncf;
  DeviceBuffer<int> d_ncf;
  DeviceBuffer<int> d_ncocc;
  DeviceBuffer<int> d_icocc;
  DeviceBuffer<int> d_nnce;
  DeviceBuffer<int> d_nce;
  DeviceBuffer<int> d_ncvir;
  DeviceBuffer<int> d_icvir;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_iused;
  DeviceBuffer<int> d_latoms;
  DeviceBuffer<int> d_nrej;
  DeviceBuffer<int> d_control_ints;
  DeviceBuffer<double> d_fmo;
  DeviceBuffer<double> d_eigs;
  DeviceBuffer<double> d_eigv;
  DeviceBuffer<double> d_cocc;
  DeviceBuffer<double> d_cvir;
  DeviceBuffer<double> d_storei;
  DeviceBuffer<double> d_storej;
  DeviceBuffer<double> d_sumb;
  DeviceBuffer<double> d_control_scalars;
  DeviceBuffer<int> d_pair_state;
  DeviceBuffer<int> d_vclaim;
  DeviceBuffer<int> d_oclaim;
  DeviceBuffer<int> d_work_ints;
  DeviceBuffer<int> d_ok;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    if (!d_pair_state.resize(nij_count)) break;
    if (!d_vclaim.resize(nvir_count)) break;
    if (!d_oclaim.resize(nocc_count)) break;
    if (!d_work_ints.resize(kDiaggWorkIntCount)) break;
    {
      const int one = 1;
      if (!d_ok.upload(&one, 1)) break;
    }
    if (!cuda_context_ok(
            cudaMemset(d_work_ints.ptr, 0, kDiaggWorkIntCount * sizeof(int)),
            "diagg2 rotate work ints memset")) break;
    if (!d_ifmo.upload(ifmo, 2 * nij_count)) break;
    if (!d_fmo.upload(fmo, nij_count)) break;
    if (!d_eigs.upload(eigs, nocc_count)) break;
    if (!d_eigv.upload(eigv, nvir_count)) break;
    if (!d_nncf.upload(nncf, nocc_count)) break;
    if (!d_ncf.upload(ncf, nocc_count)) break;
    if (!d_ncocc.upload(ncocc, nocc_count)) break;
    if (!d_icocc.upload(icocc, icocc_count)) break;
    if (!d_nnce.upload(nnce, nvir_count)) break;
    if (!d_nce.upload(nce, nvir_count)) break;
    if (!d_ncvir.upload(ncvir, nvir_count)) break;
    if (!d_icvir.upload(icvir, icvir_count)) break;
    if (!d_iorbs.upload(iorbs, numat_count)) break;
    if (!d_cocc.upload(cocc, cocc_count)) break;
    if (!d_cvir.upload(cvir, cvir_count)) break;
    if (!d_iused.resize(numat_count)) break;
    if (!d_latoms.resize(numat_count)) break;
    if (!d_storei.resize(norbs_count)) break;
    if (!d_storej.resize(norbs_count)) break;
    if (!d_sumb.resize(1)) break;
    if (!d_nrej.resize(1)) break;
    int control_ints[kDiaggIntCount] = {};
    control_ints[kDiaggIntNij] = nij;
    control_ints[kDiaggIntRetry] = retry;
    double control_scalars[kDiaggDoubleCount] = {};
    control_scalars[kDiaggDoubleRotateTiny] = tiny;
    control_scalars[kDiaggDoubleBiglim] = biglim;
    if (!d_control_ints.upload(control_ints, kDiaggIntCount)) break;
    if (!d_control_scalars.upload(control_scalars, kDiaggDoubleCount)) break;
    if (!cuda_context_ok(
            cudaMemset(d_iused.ptr, 0xff, numat_count * sizeof(int)),
            "diagg2 rotate iused memset")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemset(d_latoms.ptr, 0, numat_count * sizeof(int)),
            "diagg2 rotate latoms memset")) {
      break;
    }
    if (!cuda_context_ok(cudaMemset(d_sumb.ptr, 0, sizeof(double)),
                         "diagg2 rotate sumb memset")) break;
    if (!cuda_context_ok(cudaMemset(d_nrej.ptr, 0, sizeof(int)),
                         "diagg2 rotate nrej memset")) break;
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "diagg2 rotate create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "diagg2 rotate create stop event")) break;

    if (!cuda_context_ok(cudaEventRecord(start),
                         "diagg2 rotate start event")) break;
    {
      DiaggRotateArgs ra{};
      ra.control_ints = d_control_ints.ptr;
      ra.nij_slot = kDiaggIntNij;
      ra.retry_slot = kDiaggIntRetry;
      ra.control_scalars = d_control_scalars.ptr;
      ra.tiny_slot = kDiaggDoubleRotateTiny;
      ra.biglim_slot = kDiaggDoubleBiglim;
      ra.nocc = nocc;
      ra.nvir = nvir;
      ra.numat = numat;
      ra.norbs = norbs;
      ra.icocc_dim = icocc_dim;
      ra.icvir_dim = icvir_dim;
      ra.cocc_dim = cocc_dim;
      ra.cvir_dim = cvir_dim;
      ra.ifmo = d_ifmo.ptr;
      ra.fmo = d_fmo.ptr;
      ra.eigs = d_eigs.ptr;
      ra.eigv = d_eigv.ptr;
      ra.nncf = d_nncf.ptr;
      ra.ncf = d_ncf.ptr;
      ra.ncocc = d_ncocc.ptr;
      ra.icocc = d_icocc.ptr;
      ra.nnce = d_nnce.ptr;
      ra.nce = d_nce.ptr;
      ra.ncvir = d_ncvir.ptr;
      ra.icvir = d_icvir.ptr;
      ra.iorbs = d_iorbs.ptr;
      ra.cocc = d_cocc.ptr;
      ra.cvir = d_cvir.ptr;
      ra.shift = shift;
      ra.rot_const = rot_const;
      ra.thresh = thresh;
      ra.pair_state = d_pair_state.ptr;
      ra.vclaim = d_vclaim.ptr;
      ra.oclaim = d_oclaim.ptr;
      ra.work_ints = d_work_ints.ptr;
      ra.sumb_out = d_sumb.ptr;
      ra.nrej_out = d_nrej.ptr;
      ra.ok_slot = d_ok.ptr;
      ra.resident_control_ints = nullptr;
      ra.resident_control_scalars = nullptr;
      if (diagg_debug_enabled()) {
        std::fprintf(stderr, "[DIAGG DEBUG] standalone diagg2 nij=%d nocc=%d "
                             "nvir=%d retry=%d tiny=%.3g biglim=%.3g\n",
                     nij, nocc, nvir, retry, tiny, biglim);
      }
      if (!launch_diagg2_parallel(nij, ra, "diagg2 rotate kernel")) break;
      diagg_debug_checkpoint("standalone diagg2 rotate");
      diagg_debug_dump_ints("standalone diagg2 work ints", d_work_ints.ptr,
                            kDiaggWorkIntCount);
      diagg_debug_dump_ints("standalone diagg2 nrej", d_nrej.ptr, 1);
    }
    int ok_value = 0;
    if (!cuda_context_ok(cudaMemcpy(&ok_value, d_ok.ptr, sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "diagg2 rotate ok copy")) {
      break;
    }
    if (ok_value != 1) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(ncf, d_ncf.ptr, nocc_count * sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotate ncf copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(nce, d_nce.ptr, nvir_count * sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotate nce copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(icocc, d_icocc.ptr, icocc_count * sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotate icocc copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(icvir, d_icvir.ptr, icvir_count * sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotate icvir copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(cocc, d_cocc.ptr, cocc_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotate cocc copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(cvir, d_cvir.ptr, cvir_count * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotate cvir copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(sumb_out, d_sumb.ptr, sizeof(double),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotate sumb copy")) {
      break;
    }
    if (!cuda_context_ok(
            cudaMemcpyAsync(nrej_out, d_nrej.ptr, sizeof(int),
                            cudaMemcpyDeviceToHost),
            "diagg2 rotate nrej copy")) {
      break;
    }
    if (!cuda_context_ok(cudaEventRecord(stop),
                         "diagg2 rotate stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "diagg2 rotate synchronize")) break;

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "diagg2 rotate elapsed time")) break;
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_density_values(
    int task_count, int value_count, int cocc_dim, const int *task_j_offset,
    const int *task_k_offset, const int *task_nj, const int *task_nk,
    const int *task_diag, const int *task_value_offset, const double *cocc,
    double *values, double *wall_ms) {
  if (task_count < 0 || value_count < 0 || cocc_dim <= 0 || !task_j_offset ||
      !task_k_offset || !task_nj || !task_nk || !task_diag ||
      !task_value_offset || !cocc || !values || !wall_ms) {
    return kMozymeScfBadArgument;
  }
  *wall_ms = 0.0;
  if (task_count == 0 || value_count == 0) return kMozymeScfSuccess;

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<int> d_task_j_offset;
  DeviceBuffer<int> d_task_k_offset;
  DeviceBuffer<int> d_task_nj;
  DeviceBuffer<int> d_task_nk;
  DeviceBuffer<int> d_task_diag;
  DeviceBuffer<int> d_task_value_offset;
  DeviceBuffer<double> d_cocc;
  DeviceBuffer<double> d_values;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    const auto wall_begin = std::chrono::steady_clock::now();
    const std::size_t task_count_sz = static_cast<std::size_t>(task_count);
    const std::size_t value_count_sz = static_cast<std::size_t>(value_count);
    if (!d_task_j_offset.upload(task_j_offset, task_count_sz)) break;
    if (!d_task_k_offset.upload(task_k_offset, task_count_sz)) break;
    if (!d_task_nj.upload(task_nj, task_count_sz)) break;
    if (!d_task_nk.upload(task_nk, task_count_sz)) break;
    if (!d_task_diag.upload(task_diag, task_count_sz)) break;
    if (!d_task_value_offset.upload(task_value_offset, task_count_sz)) break;
    if (!d_cocc.upload(cocc, static_cast<std::size_t>(cocc_dim))) break;
    if (!d_values.resize(value_count_sz)) break;
    if (!cuda_context_ok(cudaEventCreate(&start),
                         "density values create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "density values create stop event")) break;

    constexpr int kThreads = 128;
    if (!cuda_context_ok(cudaEventRecord(start),
                         "density values start event")) break;
    mozyme_density_values_kernel<<<task_count, kThreads>>>(
      task_count, d_task_j_offset.ptr, d_task_k_offset.ptr, d_task_nj.ptr,
        d_task_nk.ptr, d_task_diag.ptr, d_task_value_offset.ptr, value_count,
        cocc_dim, d_cocc.ptr, d_values.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "density values kernel")) break;
    if (!cuda_context_ok(
            cudaMemcpyAsync(values, d_values.ptr,
                            value_count_sz * sizeof(double),
                            cudaMemcpyDeviceToHost),
            "density values copy")) break;
    if (!cuda_context_ok(cudaEventRecord(stop),
                         "density values stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "density values synchronize")) break;

    const auto wall_end = std::chrono::steady_clock::now();
    *wall_ms = std::chrono::duration<double, std::milli>(
                   wall_end - wall_begin)
                   .count();
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

#ifdef __CUDACC__
__device__ bool mozyme_relocalize_pass_device(
    int natoms, int norbs, int nmos, int c_dim, int ic_dim, double *c,
    const int *ic, const int *nc, const int *ncstrt, const int *nnc,
    const int *iorbs, double *psi1, double *psi2, double *axiiii, int *nf,
    int *nl, int *ioc, double *totij_out, double *total_out, int *status) {
  int l = 0;
  double xiiii = 0.0;
  for (int k = 1; k <= nmos; ++k) {
    l = nnc[k - 1];
    int m = ncstrt[k - 1];
    if (k < 1 || k > nmos) {
      *status = -101;
      return false;
    }
    axiiii[k - 1] = 0.0;
    for (int ii = 1; ii <= nc[k - 1]; ++ii) {
      ++l;
      if (l < 1 || l > ic_dim) {
        *status = -102;
        return false;
      }
      const int atom = ic[l - 1];
      if (atom < 1 || atom > natoms) {
        *status = -103;
        return false;
      }
      double dii = 0.0;
      for (int j = 1; j <= iorbs[atom - 1]; ++j) {
        ++m;
        if (m < 1 || m > c_dim) {
          *status = -104;
          return false;
        }
        const double coeff = c[m - 1];
        dii += coeff * coeff;
      }
      axiiii[k - 1] += dii * dii;
    }
  }

  double total = 0.0;
  double totij = 0.0;
  for (int iloop = 1; iloop <= nmos; ++iloop) {
    const int i5 = ncstrt[iloop - 1];
    const int i8 = nnc[iloop - 1];
    if (i8 < 0 || i8 + nc[iloop - 1] > ic_dim) {
      *status = -105;
      return false;
    }
    for (int jloop = 1; jloop <= nmos; ++jloop) {
      const int j8 = nnc[jloop - 1];
      const int j5 = ncstrt[jloop - 1];
      if (jloop == iloop) continue;
      if (j8 < 0 || j8 + nc[jloop - 1] > ic_dim) {
        *status = -106;
        return false;
      }
      bool shared_atom = false;
      for (int ii = 1; ii <= 2 && ii <= nc[iloop - 1] && !shared_atom; ++ii) {
        const int atom_i = ic[i8 + ii - 1];
        for (int jj = 1; jj <= 2 && jj <= nc[jloop - 1]; ++jj) {
          if (atom_i == ic[j8 + jj - 1]) {
            shared_atom = true;
            break;
          }
        }
      }
      if (!shared_atom) continue;

      int ij = 0;
      int ijorb = 0;
      int il = 0;
      for (int ii = 1; ii <= nc[iloop - 1]; ++ii) {
        const int atom_i = ic[i8 + ii - 1];
        if (atom_i < 1 || atom_i > natoms) {
          *status = -107;
          return false;
        }
        int jl = 0;
        for (int jj = 1; jj <= nc[jloop - 1]; ++jj) {
          const int atom_j = ic[j8 + jj - 1];
          if (atom_j < 1 || atom_j > natoms) {
            *status = -108;
            return false;
          }
          if (atom_i == atom_j) {
            ++ij;
            if (ij < 1 || ij > natoms) {
              *status = -109;
              return false;
            }
            nf[ij - 1] = ijorb + 1;
            nl[ij - 1] = ijorb + iorbs[atom_i - 1];
            ioc[2 * (ij - 1)] = il;
            ioc[2 * (ij - 1) + 1] = jl;
            int jl1 = jl;
            int il1 = il;
            for (int k = 1; k <= iorbs[atom_i - 1]; ++k) {
              ++ijorb;
              ++il1;
              ++jl1;
              if (ijorb < 1 || ijorb > norbs ||
                  i5 + il1 < 1 || i5 + il1 > c_dim ||
                  j5 + jl1 < 1 || j5 + jl1 > c_dim) {
                *status = -110;
                return false;
              }
              psi1[ijorb - 1] = c[i5 + il1 - 1];
              psi2[ijorb - 1] = c[j5 + jl1 - 1];
            }
          }
          jl += iorbs[atom_j - 1];
        }
        il += iorbs[atom_i - 1];
      }

      double xijjj = 0.0;
      double xjiii = 0.0;
      double xijij = 0.0;
      double xiijj = 0.0;
      for (int k1 = 1; k1 <= ij; ++k1) {
        double dij = 0.0;
        double dii = 0.0;
        double djj = 0.0;
        for (int k = nf[k1 - 1]; k <= nl[k1 - 1]; ++k) {
          if (k < 1 || k > norbs) {
            *status = -111;
            return false;
          }
          dij += psi1[k - 1] * psi2[k - 1];
          dii += psi1[k - 1] * psi1[k - 1];
          djj += psi2[k - 1] * psi2[k - 1];
        }
        xijjj += dij * djj;
        xjiii += dij * dii;
        xijij += dij * dij;
        xiijj += dii * djj;
      }

      if (xiijj >= 0.001) {
        xiiii = axiiii[iloop - 1];
        const double xjjjj = axiiii[jloop - 1];
        const double aij = xijij - (xiiii + xjjjj - 2.0 * xiijj) / 4.0;
        const double bij = xjiii - xijjj;
        const double ca_norm = sqrt(aij * aij + bij * bij);
        double sa = aij + ca_norm;
        if (ca_norm > 0.0 && sa > 1.0e-14) {
          double ca = (1.0 + sqrt((1.0 - aij / ca_norm) / 2.0)) / 2.0;
          sa = sqrt(1.0 - ca);
          ca = sqrt(ca);
          totij += sa;
          int ii = 0;
          for (int k = 1; k <= ij; ++k) {
            il = 0;
            for (int i = nf[k - 1]; i <= nl[k - 1]; ++i) {
              ++il;
              ++ii;
              const int c1 = ioc[2 * (k - 1)] + il + i5;
              const int c2 = ioc[2 * (k - 1) + 1] + il + j5;
              if (ii < 1 || ii > norbs || c1 < 1 || c1 > c_dim ||
                  c2 < 1 || c2 > c_dim) {
                *status = -112;
                return false;
              }
              c[c1 - 1] = ca * psi1[ii - 1] + sa * psi2[ii - 1];
              c[c2 - 1] = -sa * psi1[ii - 1] + ca * psi2[ii - 1];
            }
          }
        }
      }
    }
    total += xiiii;
  }
  *totij_out = totij;
  *total_out = total;
  return true;
}

__device__ bool mozyme_relocalize_eigs_device(
    int natoms, int norbs, int mpack, int nocc, int c_dim, int ic_dim,
    const double *c, const int *ic, const int *nc, const int *ncstrt,
    const int *nnc, const int *iorbs, const int *nfirst, const int *nlast,
    const int *nijbo, const double *p, const double *f, double *eigs,
    double *aocc, double *avir, int *status) {
  for (int idx = 0; idx < ic_dim; ++idx) aocc[idx] = 0.0;
  for (int idx = 0; idx < natoms; ++idx) avir[idx] = 0.0;

  for (int j = 1; j <= nocc; ++j) {
    const int loopj = ncstrt[j - 1];
    int kl = 0;
    for (int kk = nnc[j - 1] + 1; kk <= nnc[j - 1] + nc[j - 1]; ++kk) {
      if (kk < 1 || kk > ic_dim) {
        *status = -201;
        return false;
      }
      const int atom = ic[kk - 1];
      if (atom < 1 || atom > natoms) {
        *status = -202;
        return false;
      }
      double sum = 0.0;
      for (int k = nfirst[atom - 1]; k <= nlast[atom - 1]; ++k) {
        ++kl;
        const int cidx = kl + loopj;
        if (cidx < 1 || cidx > c_dim) {
          *status = -203;
          return false;
        }
        const double coeff = c[cidx - 1];
        sum += coeff * coeff;
      }
      aocc[kk - 1] = sum;
    }
  }

  constexpr double cutoff = 1.0e-8;
  for (int i = 1; i <= nocc; ++i) {
    const int loopi = ncstrt[i - 1];
    int l = 0;
    for (int j = nnc[i - 1] + 1; j <= nnc[i - 1] + nc[i - 1]; ++j) {
      if (j < 1 || j > ic_dim) {
        *status = -204;
        return false;
      }
      const int atom_j = ic[j - 1];
      if (atom_j < 1 || atom_j > natoms) {
        *status = -205;
        return false;
      }
      double sum = 0.0;
      for (int k = l + 1; k <= l + iorbs[atom_j - 1]; ++k) {
        const int cidx = k + loopi;
        if (cidx < 1 || cidx > c_dim) {
          *status = -206;
          return false;
        }
        const double coeff = c[cidx - 1];
        sum += coeff * coeff;
      }
      l += iorbs[atom_j - 1];
      avir[atom_j - 1] = sum;
    }

    double sum = 0.0;
    int jl = loopi;
    for (int j = nnc[i - 1] + 1; j <= nnc[i - 1] + nc[i - 1]; ++j) {
      const int atom_j = ic[j - 1];
      int kl = loopi;
      for (int k = nnc[i - 1] + 1; k <= nnc[i - 1] + nc[i - 1]; ++k) {
        const int atom_k = ic[k - 1];
        const int base = nijbo[(atom_k - 1) + natoms * (atom_j - 1)];
        if (base >= 0) {
          if (base >= mpack) {
            *status = -207;
            return false;
          }
          if (avir[atom_k - 1] * p[base] * aocc[k - 1] >= cutoff) {
            if (atom_k > atom_j) {
              for (int jx = 1; jx <= iorbs[atom_j - 1]; ++jx) {
                double sum1 = 0.0;
                for (int i4 = 1; i4 <= iorbs[atom_k - 1]; ++i4) {
                  const int ii = base + (i4 - 1) * iorbs[atom_j - 1] + jx;
                  const int cidx = kl + i4;
                  if (ii < 1 || ii > mpack || cidx < 1 || cidx > c_dim) {
                    *status = -208;
                    return false;
                  }
                  sum1 += f[ii - 1] * c[cidx - 1];
                }
                const int cidx = jl + jx;
                if (cidx < 1 || cidx > c_dim) {
                  *status = -209;
                  return false;
                }
                sum += c[cidx - 1] * sum1;
              }
            } else if (atom_k < atom_j) {
              for (int jx = 1; jx <= iorbs[atom_j - 1]; ++jx) {
                double sum1 = 0.0;
                for (int i4 = 1; i4 <= iorbs[atom_k - 1]; ++i4) {
                  const int ii = base + (jx - 1) * iorbs[atom_k - 1] + i4;
                  const int cidx = kl + i4;
                  if (ii < 1 || ii > mpack || cidx < 1 || cidx > c_dim) {
                    *status = -210;
                    return false;
                  }
                  sum1 += f[ii - 1] * c[cidx - 1];
                }
                const int cidx = jl + jx;
                if (cidx < 1 || cidx > c_dim) {
                  *status = -211;
                  return false;
                }
                sum += c[cidx - 1] * sum1;
              }
            } else {
              for (int jx = 1; jx <= iorbs[atom_j - 1]; ++jx) {
                double sum1 = 0.0;
                for (int j4 = 1; j4 <= jx; ++j4) {
                  const int ii = base + (jx * (jx - 1)) / 2 + j4;
                  const int cidx = kl + j4;
                  if (ii < 1 || ii > mpack || cidx < 1 || cidx > c_dim) {
                    *status = -212;
                    return false;
                  }
                  sum1 += f[ii - 1] * c[cidx - 1];
                }
                for (int i4 = jx + 1; i4 <= iorbs[atom_k - 1]; ++i4) {
                  const int ii = base + (i4 * (i4 - 1)) / 2 + jx;
                  const int cidx = kl + i4;
                  if (ii < 1 || ii > mpack || cidx < 1 || cidx > c_dim) {
                    *status = -213;
                    return false;
                  }
                  sum1 += f[ii - 1] * c[cidx - 1];
                }
                const int cidx = jl + jx;
                if (cidx < 1 || cidx > c_dim) {
                  *status = -214;
                  return false;
                }
                sum += c[cidx - 1] * sum1;
              }
            }
          }
        }
        kl += iorbs[atom_k - 1];
      }
      jl += iorbs[atom_j - 1];
    }
    eigs[i - 1] = sum;
  }
  return true;
}

__global__ void mozyme_relocalize_kernel(
    int kind, int natoms, int norbs, int mpack, int nmos, int c_dim,
    int ic_dim, int use_nijbo, double *c, const int *ic, const int *nc,
    const int *ncstrt, const int *nnc, const int *iorbs, const int *nfirst,
    const int *nlast, const int *nijbo, const double *p, const double *f,
    double *eigs, double *psi1, double *psi2, double *axiiii, int *nf,
    int *nl, int *ioc, double *aocc, double *avir, int *status,
    int *iterations, double *total_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *status = 0;
  *iterations = 0;
  *total_out = 0.0;
  if (kind != 1 && kind != 2) {
    *status = kMozymeScfBadArgument;
    return;
  }
  if (nmos <= 0) return;
  if (kind == 1 && use_nijbo == 0) {
    *status = kMozymeScfUnsupported;
    return;
  }

  bool converged = false;
  int display_iter = 101;
  double total = 0.0;
  for (int iter = 1; iter <= 100; ++iter) {
    double totij = 0.0;
    total = 0.0;
    if (!mozyme_relocalize_pass_device(
            natoms, norbs, nmos, c_dim, ic_dim, c, ic, nc, ncstrt, nnc,
            iorbs, psi1, psi2, axiiii, nf, nl, ioc, &totij, &total,
            status)) {
      return;
    }
    if (totij < 1.0e-5) {
      converged = true;
      display_iter = iter;
      break;
    }
  }
  if (!converged) display_iter = 101;
  *iterations = display_iter;
  *total_out = total;

  if (kind == 1 &&
      !mozyme_relocalize_eigs_device(
          natoms, norbs, mpack, nmos, c_dim, ic_dim, c, ic, nc, ncstrt, nnc,
          iorbs, nfirst, nlast, nijbo, p, f, eigs, aocc, avir, status)) {
    return;
  }
}
#endif

extern "C" int mopac_cuda_mozyme_relocalize(
    int kind, int natoms, int norbs, int mpack, int nmos, int c_dim,
    int ic_dim, int use_nijbo, double *c, int *ic, int *nc, int *ncstrt,
    int *nnc, int *iorbs, int *nfirst, int *nlast, int *nijbo, double *p,
    double *f, double *eigs, int *iterations, double *total,
    double *wall_ms) {
  if (iterations) *iterations = 0;
  if (total) *total = 0.0;
  if (wall_ms) *wall_ms = 0.0;
  if ((kind != 1 && kind != 2) || (use_nijbo != 0 && use_nijbo != 1) ||
      natoms <= 0 || norbs <= 0 || mpack <= 0 || nmos < 0 || c_dim <= 0 ||
      ic_dim <= 0 || !c || !ic || !nc || !ncstrt || !nnc || !iorbs ||
      !nfirst || !nlast || !nijbo || !p || !f || !eigs || !iterations ||
      !total || !wall_ms) {
    return kMozymeScfBadArgument;
  }
  if (nmos == 0) return kMozymeScfSuccess;

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<double> d_c;
  DeviceBuffer<double> d_p;
  DeviceBuffer<double> d_f;
  DeviceBuffer<double> d_eigs;
  DeviceBuffer<double> d_psi1;
  DeviceBuffer<double> d_psi2;
  DeviceBuffer<double> d_axiiii;
  DeviceBuffer<double> d_aocc;
  DeviceBuffer<double> d_avir;
  DeviceBuffer<double> d_total;
  DeviceBuffer<int> d_ic;
  DeviceBuffer<int> d_nc;
  DeviceBuffer<int> d_ncstrt;
  DeviceBuffer<int> d_nnc;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_nfirst;
  DeviceBuffer<int> d_nlast;
  DeviceBuffer<int> d_nijbo;
  DeviceBuffer<int> d_nf;
  DeviceBuffer<int> d_nl;
  DeviceBuffer<int> d_ioc;
  DeviceBuffer<int> d_status;
  DeviceBuffer<int> d_iterations;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    const std::size_t natoms_sz = static_cast<std::size_t>(natoms);
    const std::size_t norbs_sz = static_cast<std::size_t>(norbs);
    const std::size_t mpack_sz = static_cast<std::size_t>(mpack);
    const std::size_t nmos_sz = static_cast<std::size_t>(nmos);
    const std::size_t c_sz = static_cast<std::size_t>(c_dim);
    const std::size_t ic_sz = static_cast<std::size_t>(ic_dim);
    const std::size_t nijbo_sz = natoms_sz * natoms_sz;
    const int zero_i = 0;
    const double zero_d = 0.0;
    std::vector<double> host_c(c_sz);
    std::vector<double> host_eigs(kind == 1 ? norbs_sz : 0);
    int host_iterations = 0;
    double host_total = 0.0;

    if (!d_c.upload(c, c_sz)) break;
    if (!d_ic.upload(ic, ic_sz)) break;
    if (!d_nc.upload(nc, nmos_sz)) break;
    if (!d_ncstrt.upload(ncstrt, nmos_sz)) break;
    if (!d_nnc.upload(nnc, nmos_sz)) break;
    if (!d_iorbs.upload(iorbs, natoms_sz)) break;
    if (!d_nfirst.upload(nfirst, natoms_sz)) break;
    if (!d_nlast.upload(nlast, natoms_sz)) break;
    if (!d_nijbo.upload(nijbo, nijbo_sz)) break;
    if (!d_p.upload(p, mpack_sz)) break;
    if (!d_f.upload(f, mpack_sz)) break;
    if (!d_eigs.upload(eigs, norbs_sz)) break;
    if (!d_psi1.resize(norbs_sz)) break;
    if (!d_psi2.resize(norbs_sz)) break;
    if (!d_axiiii.resize(nmos_sz)) break;
    if (!d_aocc.resize(ic_sz)) break;
    if (!d_avir.resize(natoms_sz)) break;
    if (!d_nf.resize(natoms_sz)) break;
    if (!d_nl.resize(natoms_sz)) break;
    if (!d_ioc.resize(2 * natoms_sz)) break;
    if (!d_status.upload(&zero_i, 1)) break;
    if (!d_iterations.upload(&zero_i, 1)) break;
    if (!d_total.upload(&zero_d, 1)) break;

    if (!cuda_context_ok(cudaEventCreate(&start),
                         "relocal create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "relocal create stop event")) break;
    if (!cuda_context_ok(cudaEventRecord(start), "relocal start event")) break;

    mozyme_relocalize_kernel<<<1, 1>>>(
        kind, natoms, norbs, mpack, nmos, c_dim, ic_dim, use_nijbo, d_c.ptr,
        d_ic.ptr, d_nc.ptr, d_ncstrt.ptr, d_nnc.ptr, d_iorbs.ptr,
        d_nfirst.ptr, d_nlast.ptr, d_nijbo.ptr, d_p.ptr, d_f.ptr,
        d_eigs.ptr, d_psi1.ptr, d_psi2.ptr, d_axiiii.ptr, d_nf.ptr,
        d_nl.ptr, d_ioc.ptr, d_aocc.ptr, d_avir.ptr, d_status.ptr,
        d_iterations.ptr, d_total.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "relocal kernel")) break;
    if (!cuda_context_ok(cudaEventRecord(stop), "relocal stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "relocal synchronize")) break;

    int host_status = kMozymeScfNotReady;
    if (!cuda_context_ok(cudaMemcpy(&host_status, d_status.ptr, sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "relocal status copy")) break;
    if (host_status != 0) {
      code = host_status;
      break;
    }
    if (!cuda_context_ok(cudaMemcpy(&host_iterations, d_iterations.ptr, sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "relocal iteration copy")) break;
    if (!cuda_context_ok(cudaMemcpy(&host_total, d_total.ptr, sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "relocal total copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_c.data(), d_c.ptr,
                                    c_sz * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "relocal c copy")) break;
    if (kind == 1 &&
        !cuda_context_ok(cudaMemcpy(host_eigs.data(), d_eigs.ptr,
                                    norbs_sz * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "relocal eigs copy")) break;

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "relocal elapsed time")) break;
    std::copy(host_c.begin(), host_c.end(), c);
    if (kind == 1) std::copy(host_eigs.begin(), host_eigs.end(), eigs);
    *iterations = host_iterations;
    *total = host_total;
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

#ifdef __CUDACC__
__device__ bool mozyme_reorth_adjvec_device(
    double *cvecb, int ncvb, int *icvecb, int nib, const int *nncb,
    int *ncb_loc, int nnb, const int *ncvecb, int lmob,
    const int *iorbs, const double *cveca, int ncva, const int *icveca,
    int nia, const int *nnca, const int *nca_loc, int nna,
    const int *ncveca, int lmoa, double beta, int *iused,
    double *sumtot, double thresh, int natoms, int *status) {
  const double cutoff = thresh * 10.0;
  if (fabs(beta) < cutoff) return true;
  *sumtot += fabs(beta);

  if (lmoa < 1 || lmoa > nna || lmob < 1 || lmob > nnb) {
    *status = -301;
    return false;
  }

  for (int la = nnca[lmoa - 1] + 1;
       la <= nnca[lmoa - 1] + nca_loc[lmoa - 1]; ++la) {
    if (la < 1 || la > nia) {
      *status = -302;
      return false;
    }
    const int atom = icveca[la - 1];
    if (atom < 1 || atom > natoms) {
      *status = -303;
      return false;
    }
    iused[atom - 1] = -1;
  }

  int mlb = ncvecb[lmob - 1];
  const int llim = (lmob == nnb) ? nib : nncb[lmob];
  const int mlim = (lmob == nnb) ? ncvb - 4 : ncvecb[lmob] - 4;
  for (int lb = nncb[lmob - 1] + 1;
       lb <= nncb[lmob - 1] + ncb_loc[lmob - 1]; ++lb) {
    if (lb < 1 || lb > nib) {
      *status = -304;
      return false;
    }
    const int atom = icvecb[lb - 1];
    if (atom < 1 || atom > natoms) {
      *status = -305;
      return false;
    }
    iused[atom - 1] = mlb;
    mlb += iorbs[atom - 1];
  }

  int mla = ncveca[lmoa - 1];
  for (int la = nnca[lmoa - 1] + 1;
       la <= nnca[lmoa - 1] + nca_loc[lmoa - 1]; ++la) {
    if (la < 1 || la > nia) {
      *status = -306;
      return false;
    }
    const int atom = icveca[la - 1];
    if (atom < 1 || atom > natoms) {
      *status = -307;
      return false;
    }
    if (iused[atom - 1] >= 0) {
      int mllb = iused[atom - 1];
      for (int mlla = mla + 1; mlla <= mla + iorbs[atom - 1]; ++mlla) {
        ++mllb;
        if (mlla < 1 || mlla > ncva || mllb < 1 || mllb > ncvb) {
          *status = -308;
          return false;
        }
        cvecb[mllb - 1] -= beta * cveca[mlla - 1];
      }
    } else {
      double sum = 0.0;
      for (int mlla = mla + 1; mlla <= mla + iorbs[atom - 1]; ++mlla) {
        if (mlla < 1 || mlla > ncva) {
          *status = -309;
          return false;
        }
        sum += cveca[mlla - 1] * cveca[mlla - 1];
      }
      if (beta * beta * sum > cutoff) {
        if (ncb_loc[lmob - 1] < llim && mlb < mlim) {
          ncb_loc[lmob - 1] += 1;
          const int icpos = nncb[lmob - 1] + ncb_loc[lmob - 1];
          if (icpos < 1 || icpos > nib) {
            *status = -310;
            return false;
          }
          icvecb[icpos - 1] = atom;
          iused[atom - 1] = mlb;
          for (int mlla = mla + 1; mlla <= mla + iorbs[atom - 1]; ++mlla) {
            ++mlb;
            if (mlla < 1 || mlla > ncva || mlb < 1 || mlb > ncvb) {
              *status = -311;
              return false;
            }
            cvecb[mlb - 1] = -beta * cveca[mlla - 1];
          }
        }
      }
    }
    mla += iorbs[atom - 1];
  }
  return true;
}

__global__ void mozyme_reorth_kernel(
    int natoms, int norbs, int nocc, int nvir, int cocc_dim, int icocc_dim,
    int cvir_dim, int icvir_dim, double thresh, double *cocc, int *icocc,
    int *ncf, const int *nncf, const int *ncocc, double *cvir, int *icvir,
    int *nce, const int *nnce, const int *ncvir, const int *iorbs,
    const int *nfirst, double *ws, int *latom, int *iused, int *status,
    double *sumtot_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *status = 0;
  *sumtot_out = 0.0;
  double sumtot = 0.0;

  for (int i = 1; i <= nvir; ++i) {
    int loopi = ncvir[i - 1];
    for (int atom = 0; atom < natoms; ++atom) latom[atom] = 0;
    for (int jj = nnce[i - 1] + 1; jj <= nnce[i - 1] + nce[i - 1]; ++jj) {
      if (jj < 1 || jj > icvir_dim) {
        *status = -401;
        return;
      }
      const int atom = icvir[jj - 1];
      if (atom < 1 || atom > natoms) {
        *status = -402;
        return;
      }
      latom[atom - 1] = 1;
      int j = nfirst[atom - 1] - 1;
      for (int jx = 1; jx <= iorbs[atom - 1]; ++jx) {
        ++loopi;
        ++j;
        if (loopi < 1 || loopi > cvir_dim || j < 1 || j > norbs) {
          *status = -403;
          return;
        }
        ws[j - 1] = cvir[loopi - 1];
      }
    }

    for (int ii = i + 1; ii <= nvir; ++ii) {
      double sum = 0.0;
      int loopii = ncvir[ii - 1];
      for (int jj = nnce[ii - 1] + 1;
           jj <= nnce[ii - 1] + nce[ii - 1]; ++jj) {
        if (jj < 1 || jj > icvir_dim) {
          *status = -404;
          return;
        }
        const int atom = icvir[jj - 1];
        if (atom < 1 || atom > natoms) {
          *status = -405;
          return;
        }
        if (latom[atom - 1] != 0) {
          int j = nfirst[atom - 1] - 1;
          for (int jx = 1; jx <= iorbs[atom - 1]; ++jx) {
            ++loopii;
            ++j;
            if (loopii < 1 || loopii > cvir_dim || j < 1 || j > norbs) {
              *status = -406;
              return;
            }
            sum += ws[j - 1] * cvir[loopii - 1];
          }
        } else {
          loopii += iorbs[atom - 1];
        }
      }
      if (!mozyme_reorth_adjvec_device(
              cvir, cvir_dim, icvir, icvir_dim, nnce, nce, nvir, ncvir, i,
              iorbs, cvir, cvir_dim, icvir, icvir_dim, nnce, nce, nvir,
              ncvir, ii, sum, iused, &sumtot, thresh, natoms, status)) {
        return;
      }
    }

    for (int ii = 1; ii <= nocc; ++ii) {
      double sum = 0.0;
      int loopii = ncocc[ii - 1];
      for (int jj = nncf[ii - 1] + 1;
           jj <= nncf[ii - 1] + ncf[ii - 1]; ++jj) {
        if (jj < 1 || jj > icocc_dim) {
          *status = -407;
          return;
        }
        const int atom = icocc[jj - 1];
        if (atom < 1 || atom > natoms) {
          *status = -408;
          return;
        }
        if (latom[atom - 1] != 0) {
          int j = nfirst[atom - 1] - 1;
          for (int jx = 1; jx <= iorbs[atom - 1]; ++jx) {
            ++loopii;
            ++j;
            if (loopii < 1 || loopii > cocc_dim || j < 1 || j > norbs) {
              *status = -409;
              return;
            }
            sum += ws[j - 1] * cocc[loopii - 1];
          }
        } else {
          loopii += iorbs[atom - 1];
        }
      }
      if (!mozyme_reorth_adjvec_device(
              cvir, cvir_dim, icvir, icvir_dim, nnce, nce, nvir, ncvir, i,
              iorbs, cocc, cocc_dim, icocc, icocc_dim, nncf, ncf, nocc,
              ncocc, ii, sum, iused, &sumtot, thresh, natoms, status)) {
        return;
      }
    }
  }

  for (int i = 1; i <= nocc; ++i) {
    int loopi = ncocc[i - 1];
    for (int atom = 0; atom < natoms; ++atom) latom[atom] = 0;
    for (int jj = nncf[i - 1] + 1; jj <= nncf[i - 1] + ncf[i - 1]; ++jj) {
      if (jj < 1 || jj > icocc_dim) {
        *status = -410;
        return;
      }
      const int atom = icocc[jj - 1];
      if (atom < 1 || atom > natoms) {
        *status = -411;
        return;
      }
      latom[atom - 1] = 1;
      int j = nfirst[atom - 1] - 1;
      for (int jx = 1; jx <= iorbs[atom - 1]; ++jx) {
        ++loopi;
        ++j;
        if (loopi < 1 || loopi > cocc_dim || j < 1 || j > norbs) {
          *status = -412;
          return;
        }
        ws[j - 1] = cocc[loopi - 1];
      }
    }

    for (int ii = i + 1; ii <= nocc; ++ii) {
      double sum = 0.0;
      int loopii = ncocc[ii - 1];
      for (int jj = nncf[ii - 1] + 1;
           jj <= nncf[ii - 1] + ncf[ii - 1]; ++jj) {
        if (jj < 1 || jj > icocc_dim) {
          *status = -413;
          return;
        }
        const int atom = icocc[jj - 1];
        if (atom < 1 || atom > natoms) {
          *status = -414;
          return;
        }
        if (latom[atom - 1] != 0) {
          int j = nfirst[atom - 1] - 1;
          for (int jx = 1; jx <= iorbs[atom - 1]; ++jx) {
            ++loopii;
            ++j;
            if (loopii < 1 || loopii > cocc_dim || j < 1 || j > norbs) {
              *status = -415;
              return;
            }
            sum += ws[j - 1] * cocc[loopii - 1];
          }
        } else {
          loopii += iorbs[atom - 1];
        }
      }
      if (!mozyme_reorth_adjvec_device(
              cocc, cocc_dim, icocc, icocc_dim, nncf, ncf, nocc, ncocc, i,
              iorbs, cocc, cocc_dim, icocc, icocc_dim, nncf, ncf, nocc,
              ncocc, ii, sum, iused, &sumtot, thresh, natoms, status)) {
        return;
      }
    }
  }
  *sumtot_out = sumtot;
}
#endif

extern "C" int mopac_cuda_mozyme_reorth(
    int natoms, int norbs, int nocc, int nvir, int cocc_dim,
    int icocc_dim, int cvir_dim, int icvir_dim, double thresh,
    double *cocc, int *icocc, int *ncf, int *nncf, int *ncocc,
    double *cvir, int *icvir, int *nce, int *nnce, int *ncvir,
    int *iorbs, int *nfirst, double *sumtot, double *wall_ms) {
  if (sumtot) *sumtot = 0.0;
  if (wall_ms) *wall_ms = 0.0;
  if (natoms <= 0 || norbs <= 0 || nocc < 0 || nvir < 0 ||
      cocc_dim <= 0 || icocc_dim <= 0 || cvir_dim <= 0 ||
      icvir_dim <= 0 || !(thresh >= 0.0) || !cocc || !icocc || !ncf ||
      !nncf || !ncocc || !cvir || !icvir || !nce || !nnce || !ncvir || !iorbs ||
      !nfirst || !sumtot || !wall_ms) {
    return kMozymeScfBadArgument;
  }
  if (nocc == 0 && nvir == 0) return kMozymeScfSuccess;

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<double> d_cocc;
  DeviceBuffer<double> d_cvir;
  DeviceBuffer<double> d_ws;
  DeviceBuffer<double> d_sumtot;
  DeviceBuffer<int> d_icocc;
  DeviceBuffer<int> d_ncf;
  DeviceBuffer<int> d_nncf;
  DeviceBuffer<int> d_ncocc;
  DeviceBuffer<int> d_icvir;
  DeviceBuffer<int> d_nce;
  DeviceBuffer<int> d_nnce;
  DeviceBuffer<int> d_ncvir;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_nfirst;
  DeviceBuffer<int> d_latom;
  DeviceBuffer<int> d_iused;
  DeviceBuffer<int> d_status;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    const std::size_t natoms_sz = static_cast<std::size_t>(natoms);
    const std::size_t norbs_sz = static_cast<std::size_t>(norbs);
    const std::size_t nocc_sz = static_cast<std::size_t>(nocc);
    const std::size_t nvir_sz = static_cast<std::size_t>(nvir);
    const std::size_t cocc_sz = static_cast<std::size_t>(cocc_dim);
    const std::size_t icocc_sz = static_cast<std::size_t>(icocc_dim);
    const std::size_t cvir_sz = static_cast<std::size_t>(cvir_dim);
    const std::size_t icvir_sz = static_cast<std::size_t>(icvir_dim);
    const int zero_i = 0;
    const double zero_d = 0.0;
    std::vector<double> host_cocc(cocc_sz);
    std::vector<double> host_cvir(cvir_sz);
    std::vector<int> host_icocc(icocc_sz);
    std::vector<int> host_icvir(icvir_sz);
    std::vector<int> host_ncf(nocc_sz);
    std::vector<int> host_nce(nvir_sz);
    double host_sumtot = 0.0;

    if (!d_cocc.upload(cocc, cocc_sz)) break;
    if (!d_icocc.upload(icocc, icocc_sz)) break;
    if (!d_ncf.upload(ncf, nocc_sz)) break;
    if (!d_nncf.upload(nncf, nocc_sz)) break;
    if (!d_ncocc.upload(ncocc, nocc_sz)) break;
    if (!d_cvir.upload(cvir, cvir_sz)) break;
    if (!d_icvir.upload(icvir, icvir_sz)) break;
    if (!d_nce.upload(nce, nvir_sz)) break;
    if (!d_nnce.upload(nnce, nvir_sz)) break;
    if (!d_ncvir.upload(ncvir, nvir_sz)) break;
    if (!d_iorbs.upload(iorbs, natoms_sz)) break;
    if (!d_nfirst.upload(nfirst, natoms_sz)) break;
    if (!d_ws.resize(norbs_sz)) break;
    if (!d_latom.resize(natoms_sz)) break;
    if (!d_iused.resize(natoms_sz)) break;
    if (!d_status.upload(&zero_i, 1)) break;
    if (!d_sumtot.upload(&zero_d, 1)) break;

    if (!cuda_context_ok(cudaEventCreate(&start),
                         "reorth create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "reorth create stop event")) break;
    if (!cuda_context_ok(cudaEventRecord(start), "reorth start event")) break;

    mozyme_reorth_kernel<<<1, 1>>>(
        natoms, norbs, nocc, nvir, cocc_dim, icocc_dim, cvir_dim,
        icvir_dim, thresh, d_cocc.ptr, d_icocc.ptr, d_ncf.ptr,
        d_nncf.ptr, d_ncocc.ptr, d_cvir.ptr, d_icvir.ptr, d_nce.ptr,
        d_nnce.ptr, d_ncvir.ptr, d_iorbs.ptr, d_nfirst.ptr, d_ws.ptr,
        d_latom.ptr, d_iused.ptr, d_status.ptr, d_sumtot.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "reorth kernel")) break;
    if (!cuda_context_ok(cudaEventRecord(stop), "reorth stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "reorth synchronize")) break;

    int host_status = kMozymeScfNotReady;
    if (!cuda_context_ok(cudaMemcpy(&host_status, d_status.ptr, sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "reorth status copy")) break;
    if (host_status != 0) {
      code = host_status;
      break;
    }
    if (!cuda_context_ok(cudaMemcpy(host_cocc.data(), d_cocc.ptr,
                                    cocc_sz * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "reorth cocc copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_cvir.data(), d_cvir.ptr,
                                    cvir_sz * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "reorth cvir copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_icocc.data(), d_icocc.ptr,
                                    icocc_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "reorth icocc copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_icvir.data(), d_icvir.ptr,
                                    icvir_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "reorth icvir copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_ncf.data(), d_ncf.ptr,
                                    nocc_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "reorth ncf copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_nce.data(), d_nce.ptr,
                                    nvir_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "reorth nce copy")) break;
    if (!cuda_context_ok(cudaMemcpy(&host_sumtot, d_sumtot.ptr,
                                    sizeof(double), cudaMemcpyDeviceToHost),
                         "reorth sumtot copy")) break;
    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "reorth elapsed time")) break;

    std::copy(host_cocc.begin(), host_cocc.end(), cocc);
    std::copy(host_cvir.begin(), host_cvir.end(), cvir);
    std::copy(host_icocc.begin(), host_icocc.end(), icocc);
    std::copy(host_icvir.begin(), host_icvir.end(), icvir);
    std::copy(host_ncf.begin(), host_ncf.end(), ncf);
    std::copy(host_nce.begin(), host_nce.end(), nce);
    *sumtot = host_sumtot;
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_tidy(
    int nmos, int natoms, int norbs, int ic_dim, int c_dim, double thresh,
    int use_selmos, int numred, int mode, int *nc, int *ic, double *c, int *nnc,
    int *ncmo, const int *iorbs, const int *jopt, int *ln_out, int *mn_out,
    int *selected_out, double *wall_ms) {
  if (ln_out) *ln_out = 0;
  if (mn_out) *mn_out = 0;
  if (selected_out) *selected_out = -1;
  if (wall_ms) *wall_ms = 0.0;
  if (nmos < 0 || natoms <= 0 || norbs <= 0 || ic_dim <= 0 || c_dim <= 0 ||
      use_selmos < 0 || use_selmos > 1 || numred < 0 || mode < 1 || mode > 2 ||
      (use_selmos != 0 && numred > natoms) || !(thresh >= 0.0) ||
      !nc || !ic || !c || !nnc || !ncmo || !iorbs ||
      (use_selmos != 0 && numred > 0 && !jopt) || !ln_out || !mn_out || !selected_out ||
      !wall_ms) {
    return kMozymeScfBadArgument;
  }
  if (nmos == 0) return kMozymeScfSuccess;

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<int> d_nc;
  DeviceBuffer<int> d_ic;
  DeviceBuffer<int> d_nnc;
  DeviceBuffer<int> d_ncmo;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_jopt;
  DeviceBuffer<int> d_iused;
  DeviceBuffer<int> d_ncnew;
  DeviceBuffer<int> d_ncmnew;
  DeviceBuffer<int> d_nncnew;
  DeviceBuffer<int> d_result;
  DeviceBuffer<double> d_c;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = -530;

  do {
    const std::size_t nmos_sz = static_cast<std::size_t>(nmos);
    const std::size_t natoms_sz = static_cast<std::size_t>(natoms);
    const std::size_t ic_sz = static_cast<std::size_t>(ic_dim);
    const std::size_t c_sz = static_cast<std::size_t>(c_dim);
    const int result_init[4] = {kMozymeScfNotReady, 0, 0, -1};
    std::vector<int> host_nc(nmos_sz);
    std::vector<int> host_nnc(nmos_sz);
    std::vector<int> host_ncmo(nmos_sz);
    std::vector<int> host_ic(ic_sz);
    std::vector<double> host_c(c_sz);
    int host_result[4] = {kMozymeScfNotReady, 0, 0, -1};

    if (!d_nc.upload(nc, nmos_sz)) break;
    if (!d_ic.upload(ic, ic_sz)) break;
    if (!d_c.upload(c, c_sz)) break;
    if (!d_nnc.upload(nnc, nmos_sz)) break;
    if (!d_ncmo.upload(ncmo, nmos_sz)) break;
    if (!d_iorbs.upload(iorbs, natoms_sz)) break;
    if (use_selmos != 0 && numred > 0 &&
        !d_jopt.upload(jopt, static_cast<std::size_t>(numred))) {
      break;
    }
    if (!d_iused.resize(nmos_sz)) break;
    if (!d_ncnew.resize(nmos_sz)) break;
    if (!d_ncmnew.resize(nmos_sz)) break;
    if (!d_nncnew.resize(nmos_sz)) break;
    if (!d_result.upload(result_init, 4)) break;

    if (!cuda_context_ok(cudaEventCreate(&start),
                         "tidy create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "tidy create stop event")) break;
    if (!cuda_context_ok(cudaEventRecord(start), "tidy start event")) break;

    mozyme_tidy_kernel<<<1, 1>>>(
        nmos, natoms, norbs, ic_dim, c_dim, thresh, use_selmos, numred, mode,
        d_nc.ptr, d_ic.ptr, d_c.ptr, d_nnc.ptr, d_ncmo.ptr, d_iorbs.ptr,
        (use_selmos && numred > 0) ? d_jopt.ptr : nullptr,
        d_iused.ptr, d_ncnew.ptr, d_ncmnew.ptr, d_nncnew.ptr, d_result.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "tidy kernel")) break;
    if (!cuda_context_ok(cudaEventRecord(stop), "tidy stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop), "tidy synchronize")) break;

    if (!cuda_context_ok(cudaMemcpy(host_result, d_result.ptr,
                                    sizeof(host_result),
                                    cudaMemcpyDeviceToHost),
                         "tidy result copy")) break;
    if (host_result[0] != 0) {
      code = host_result[0];
      break;
    }
    if (!cuda_context_ok(cudaMemcpy(host_nc.data(), d_nc.ptr,
                                    nmos_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "tidy nc copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_ic.data(), d_ic.ptr,
                                    ic_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "tidy ic copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_c.data(), d_c.ptr,
                                    c_sz * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "tidy c copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_nnc.data(), d_nnc.ptr,
                                    nmos_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "tidy nnc copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_ncmo.data(), d_ncmo.ptr,
                                    nmos_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "tidy ncmo copy")) break;

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "tidy elapsed time")) break;

    std::memcpy(nc, host_nc.data(), nmos_sz * sizeof(int));
    std::memcpy(ic, host_ic.data(), ic_sz * sizeof(int));
    std::memcpy(c, host_c.data(), c_sz * sizeof(double));
    std::memcpy(nnc, host_nnc.data(), nmos_sz * sizeof(int));
    std::memcpy(ncmo, host_ncmo.data(), nmos_sz * sizeof(int));
    *ln_out = host_result[1];
    *mn_out = host_result[2];
    *selected_out = host_result[3];
    *wall_ms = static_cast<double>(elapsed);
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

extern "C" int mopac_cuda_mozyme_makvec(
    int natoms, int norbs, int mpack, int morb, int lewis_tot,
    int noccupied, int nvirtual, int ipad2, int ipad4, int icocc_dim,
    int cocc_dim, int icvir_dim, int cvir_dim, int ibonds_rows,
    double *pdiag, double *h, double *p, double *f, int *iorbs,
    int *nfirst, int *nlast, int *nijbo, int *nbonds, int *ibonds,
    int *lewis_elem, int *ncf, int *nncf, int *ncocc, int *icocc,
    double *cocc, int *nce, int *nnce, int *ncvir, int *icvir,
    double *cvir, double *wall_ms) {
  if (wall_ms) *wall_ms = 0.0;
  if (natoms <= 0 || norbs <= 0 || mpack <= 0 || morb <= 0 ||
      lewis_tot <= 0 || noccupied <= 0 || nvirtual < 0 || ipad2 <= 0 ||
      ipad4 <= 0 || icocc_dim <= 0 || cocc_dim <= 0 || icvir_dim <= 0 ||
      cvir_dim <= 0 || ibonds_rows <= 0 || !pdiag || !h || !p || !f ||
      !iorbs || !nfirst || !nlast || !nijbo || !nbonds || !ibonds ||
      !lewis_elem || !ncf || !nncf || !ncocc || !icocc || !cocc ||
      !nce || !nnce || !ncvir || !icvir || !cvir || !wall_ms) {
    return kMozymeScfBadArgument;
  }

#ifndef __CUDACC__
  return kMozymeScfUnsupported;
#else
  DeviceBuffer<double> d_pdiag;
  DeviceBuffer<double> d_h;
  DeviceBuffer<double> d_p;
  DeviceBuffer<double> d_f;
  DeviceBuffer<double> d_qe;
  DeviceBuffer<double> d_catom;
  DeviceBuffer<double> d_cocc;
  DeviceBuffer<double> d_cvir;
  DeviceBuffer<int> d_iorbs;
  DeviceBuffer<int> d_nfirst;
  DeviceBuffer<int> d_nlast;
  DeviceBuffer<int> d_nijbo;
  DeviceBuffer<int> d_nbonds;
  DeviceBuffer<int> d_ibonds;
  DeviceBuffer<int> d_lewis;
  DeviceBuffer<int> d_ncf;
  DeviceBuffer<int> d_nncf;
  DeviceBuffer<int> d_ncocc;
  DeviceBuffer<int> d_icocc;
  DeviceBuffer<int> d_nce;
  DeviceBuffer<int> d_nnce;
  DeviceBuffer<int> d_ncvir;
  DeviceBuffer<int> d_icvir;
  DeviceBuffer<int> d_ok;
  DeviceBuffer<unsigned char> d_used;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int code = kMozymeScfNotReady;

  do {
    const std::size_t natoms_sz = static_cast<std::size_t>(natoms);
    const std::size_t norbs_sz = static_cast<std::size_t>(norbs);
    const std::size_t mpack_sz = static_cast<std::size_t>(mpack);
    const std::size_t nijbo_sz = natoms_sz * natoms_sz;
    const std::size_t ibonds_sz =
        static_cast<std::size_t>(ibonds_rows) * natoms_sz;
    const std::size_t lewis_sz = static_cast<std::size_t>(2 * lewis_tot);
    const std::size_t nocc_sz = static_cast<std::size_t>(noccupied + 1);
    const std::size_t nvir_sz = static_cast<std::size_t>(nvirtual + 1);
    const std::size_t catom_sz =
        static_cast<std::size_t>(morb) * norbs_sz;

    if (!d_pdiag.upload(pdiag, norbs_sz)) break;
    if (!d_h.upload(h, mpack_sz)) break;
    if (!d_p.resize(mpack_sz)) break;
    if (!d_f.resize(mpack_sz)) break;
    if (!d_qe.resize(natoms_sz)) break;
    if (!d_catom.resize(catom_sz)) break;
    if (!d_iorbs.upload(iorbs, natoms_sz)) break;
    if (!d_nfirst.upload(nfirst, natoms_sz)) break;
    if (!d_nlast.upload(nlast, natoms_sz)) break;
    if (!d_nijbo.upload(nijbo, nijbo_sz)) break;
    if (!d_nbonds.upload(nbonds, natoms_sz)) break;
    if (!d_ibonds.upload(ibonds, ibonds_sz)) break;
    if (!d_lewis.upload(lewis_elem, lewis_sz)) break;
    if (!d_ncf.resize(nocc_sz)) break;
    if (!d_nncf.resize(nocc_sz)) break;
    if (!d_ncocc.resize(nocc_sz)) break;
    if (!d_icocc.resize(static_cast<std::size_t>(icocc_dim))) break;
    if (!d_cocc.resize(static_cast<std::size_t>(cocc_dim))) break;
    if (!d_nce.resize(nvir_sz)) break;
    if (!d_nnce.resize(nvir_sz)) break;
    if (!d_ncvir.resize(nvir_sz)) break;
    if (!d_icvir.resize(static_cast<std::size_t>(icvir_dim))) break;
    if (!d_cvir.resize(static_cast<std::size_t>(cvir_dim))) break;
    if (!d_ok.resize(1)) break;
    if (!d_used.resize(norbs_sz)) break;

    if (!cuda_context_ok(cudaMemset(d_catom.ptr, 0, catom_sz * sizeof(double)),
                         "makvec catom memset")) break;
    if (!cuda_context_ok(cudaMemset(d_used.ptr, 0,
                                    norbs_sz * sizeof(unsigned char)),
                         "makvec used memset")) break;
    if (!cuda_context_ok(cudaMemset(d_cocc.ptr, 0,
                                    static_cast<std::size_t>(cocc_dim) *
                                        sizeof(double)),
                         "makvec cocc memset")) break;
    if (!cuda_context_ok(cudaMemset(d_cvir.ptr, 0,
                                    static_cast<std::size_t>(cvir_dim) *
                                        sizeof(double)),
                         "makvec cvir memset")) break;
    if (!cuda_context_ok(cudaMemset(d_icocc.ptr, 0,
                                    static_cast<std::size_t>(icocc_dim) *
                                        sizeof(int)),
                         "makvec icocc memset")) break;
    if (!cuda_context_ok(cudaMemset(d_icvir.ptr, 0,
                                    static_cast<std::size_t>(icvir_dim) *
                                        sizeof(int)),
                         "makvec icvir memset")) break;
    if (!cuda_context_ok(cudaMemset(d_ncf.ptr, 0, nocc_sz * sizeof(int)),
                         "makvec ncf memset")) break;
    if (!cuda_context_ok(cudaMemset(d_nncf.ptr, 0, nocc_sz * sizeof(int)),
                         "makvec nncf memset")) break;
    if (!cuda_context_ok(cudaMemset(d_ncocc.ptr, 0, nocc_sz * sizeof(int)),
                         "makvec ncocc memset")) break;
    if (!cuda_context_ok(cudaMemset(d_nce.ptr, 0, nvir_sz * sizeof(int)),
                         "makvec nce memset")) break;
    if (!cuda_context_ok(cudaMemset(d_nnce.ptr, 0, nvir_sz * sizeof(int)),
                         "makvec nnce memset")) break;
    if (!cuda_context_ok(cudaMemset(d_ncvir.ptr, 0, nvir_sz * sizeof(int)),
                         "makvec ncvir memset")) break;

    if (!cuda_context_ok(cudaEventCreate(&start),
                         "makvec create start event")) break;
    if (!cuda_context_ok(cudaEventCreate(&stop),
                         "makvec create stop event")) break;
    if (!cuda_context_ok(cudaEventRecord(start), "makvec start event")) break;

    constexpr int kThreads = 256;
    const int init_blocks = ceil_div(std::max(mpack, natoms), kThreads);
    mozyme_makvec_init_kernel<<<init_blocks, kThreads>>>(
        natoms, mpack, d_iorbs.ptr, d_nfirst.ptr, d_nijbo.ptr, d_pdiag.ptr,
        d_h.ptr, d_p.ptr, d_f.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "makvec init kernel")) break;

    const int atom_blocks = ceil_div(natoms, kThreads);
    mozyme_chrge_kernel<<<atom_blocks, kThreads>>>(
        natoms, mpack, d_iorbs.ptr, d_nijbo.ptr, d_p.ptr, d_qe.ptr, nullptr);
    if (!cuda_context_ok(cudaGetLastError(), "makvec charge kernel")) break;
    if (!cuda_context_ok(cudaDeviceSynchronize(),
                         "makvec pre-fock synchronize")) break;

    double sparse_ms = 0.0;
    const int sparse_code = mopac_cuda_mozyme_sparse_fock_run_device_plan(
        kMozymeFockPlanFull, mpack, d_p.ptr, d_qe.ptr, d_f.ptr, &sparse_ms);
    if (sparse_code != kMozymeScfSuccess) {
      code = sparse_code;
      break;
    }

    mozyme_makvec_lmo_kernel<<<1, 1>>>(
        natoms, norbs, morb, lewis_tot, noccupied, nvirtual, ipad2, ipad4,
        icocc_dim, cocc_dim, icvir_dim, cvir_dim, ibonds_rows, d_iorbs.ptr,
        d_nfirst.ptr, d_nlast.ptr, d_nijbo.ptr, d_nbonds.ptr, d_ibonds.ptr,
        d_lewis.ptr, d_f.ptr, d_catom.ptr, d_ncf.ptr, d_nncf.ptr,
        d_ncocc.ptr, d_icocc.ptr, d_cocc.ptr, d_nce.ptr, d_nnce.ptr,
        d_ncvir.ptr, d_icvir.ptr, d_cvir.ptr, d_used.ptr, d_ok.ptr);
    if (!cuda_context_ok(cudaGetLastError(), "makvec lmo kernel")) break;
    if (!cuda_context_ok(cudaEventRecord(stop), "makvec stop event")) break;
    if (!cuda_context_ok(cudaEventSynchronize(stop),
                         "makvec synchronize")) break;

    int host_ok = 0;
    if (!cuda_context_ok(cudaMemcpy(&host_ok, d_ok.ptr, sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "makvec status copy")) break;
    if (host_ok != 0) {
      code = host_ok;
      break;
    }

    std::vector<double> host_p(mpack_sz);
    std::vector<double> host_f(mpack_sz);
    std::vector<int> host_ncf(nocc_sz);
    std::vector<int> host_nncf(nocc_sz);
    std::vector<int> host_ncocc(nocc_sz);
    std::vector<int> host_icocc(static_cast<std::size_t>(icocc_dim));
    std::vector<double> host_cocc(static_cast<std::size_t>(cocc_dim));
    std::vector<int> host_nce(nvir_sz);
    std::vector<int> host_nnce(nvir_sz);
    std::vector<int> host_ncvir(nvir_sz);
    std::vector<int> host_icvir(static_cast<std::size_t>(icvir_dim));
    std::vector<double> host_cvir(static_cast<std::size_t>(cvir_dim));

    if (!cuda_context_ok(cudaMemcpy(host_p.data(), d_p.ptr,
                                    mpack_sz * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "makvec p copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_f.data(), d_f.ptr,
                                    mpack_sz * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "makvec f copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_ncf.data(), d_ncf.ptr,
                                    nocc_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "makvec ncf copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_nncf.data(), d_nncf.ptr,
                                    nocc_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "makvec nncf copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_ncocc.data(), d_ncocc.ptr,
                                    nocc_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "makvec ncocc copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_icocc.data(), d_icocc.ptr,
                                    static_cast<std::size_t>(icocc_dim) *
                                        sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "makvec icocc copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_cocc.data(), d_cocc.ptr,
                                    static_cast<std::size_t>(cocc_dim) *
                                        sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "makvec cocc copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_nce.data(), d_nce.ptr,
                                    nvir_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "makvec nce copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_nnce.data(), d_nnce.ptr,
                                    nvir_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "makvec nnce copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_ncvir.data(), d_ncvir.ptr,
                                    nvir_sz * sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "makvec ncvir copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_icvir.data(), d_icvir.ptr,
                                    static_cast<std::size_t>(icvir_dim) *
                                        sizeof(int),
                                    cudaMemcpyDeviceToHost),
                         "makvec icvir copy")) break;
    if (!cuda_context_ok(cudaMemcpy(host_cvir.data(), d_cvir.ptr,
                                    static_cast<std::size_t>(cvir_dim) *
                                        sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "makvec cvir copy")) break;

    std::memcpy(p, host_p.data(), mpack_sz * sizeof(double));
    std::memcpy(f, host_f.data(), mpack_sz * sizeof(double));
    std::memcpy(ncf, host_ncf.data(), nocc_sz * sizeof(int));
    std::memcpy(nncf, host_nncf.data(), nocc_sz * sizeof(int));
    std::memcpy(ncocc, host_ncocc.data(), nocc_sz * sizeof(int));
    std::memcpy(icocc, host_icocc.data(),
                static_cast<std::size_t>(icocc_dim) * sizeof(int));
    std::memcpy(cocc, host_cocc.data(),
                static_cast<std::size_t>(cocc_dim) * sizeof(double));
    std::memcpy(nce, host_nce.data(), nvir_sz * sizeof(int));
    std::memcpy(nnce, host_nnce.data(), nvir_sz * sizeof(int));
    std::memcpy(ncvir, host_ncvir.data(), nvir_sz * sizeof(int));
    std::memcpy(icvir, host_icvir.data(),
                static_cast<std::size_t>(icvir_dim) * sizeof(int));
    std::memcpy(cvir, host_cvir.data(),
                static_cast<std::size_t>(cvir_dim) * sizeof(double));

    float elapsed = 0.0f;
    if (!cuda_context_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         "makvec elapsed time")) break;
    *wall_ms = static_cast<double>(elapsed);
    if (sparse_ms > *wall_ms) *wall_ms = sparse_ms;
    code = kMozymeScfSuccess;
  } while (false);

  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return code;
#endif
}

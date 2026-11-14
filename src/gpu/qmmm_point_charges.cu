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

// QM/MM electrostatic embedding of MOZYME on the device: the MM point charges in the one-electron matrix
// (qmmm_hcore) and the gradient of the QM-MM energy on the QM atoms and on the charges (qmmm_grad), with the
// integrals of qmmm_pair (src/MOZYME/qmmm.F90).  On the CPU these loops over all QM atom - charge pairs took
// ~2 ms of a 0.037-s QM/MM step of trypsin-BPTI (59 QM atoms, ~2200 charges, 12 OpenMP threads, Colab A100).
//
// Every sum runs in a fixed order: over a chunk of charges (one thread per QM atom and chunk), then over the
// chunks (one thread per QM atom); the charges' own sums likewise over chunks of QM atoms.  The chunks are sized
// for ~kTargetThreads threads per kernel: with one thread per QM atom over 64 charges (1900 threads for
// trypsin-BPTI) the kernels took 1.0 ms of GPU time per step.  Per QM atom only the elements qmmm_pair sets are kept: the 10 of an sp block and the 5 d
// diagonal elements (kSlots; their packed positions in the atom's diagonal block are slot_pos in qmmm.F90).
// The kernels use no shared memory; the file also compiles as C++ with a shim (tests run the kernels thread
// by thread on the CPU).

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

constexpr int kSlots = 15;
constexpr int kThreads = 128;
constexpr long long kTargetThreads = 32768;

struct PairTerms {
  double rr, u[3];
  double ri1, ri2, ri3, ri4;        // (ss|J), (s p_sigma|J), (p_sigma p_sigma|J), (p_pi p_pi|J), eV
  double d1, d2, d3, d4;            // their derivatives with respect to R, eV/A
  double sw, dsw;                   // switching function and dS/dR
};

// qmmm_pair's radial terms for one QM atom (nao orbitals; par = am, dd, qq, ad, aq, tore) and a unit charge at
// d = x(J) - x(A)
__device__ __forceinline__ void pair_terms(int nao, const double *par, const double d[3], double ev, double a0,
                                           double ron, double roff, bool want_grad, PairTerms &t) {
  t.rr = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  for (int c = 0; c < 3; c++) t.u[c] = d[c] / t.rr;
  const double r = t.rr / a0;
  const double aee = (0.5 / par[0]) * (0.5 / par[0]);
  const double s1 = sqrt(r * r + aee);
  t.ri1 = ev / s1;
  t.d1 = -ev * r / (s1 * s1 * s1) / a0;
  t.ri2 = t.ri3 = t.ri4 = t.d2 = t.d3 = t.d4 = 0.0;
  if (nao >= 4) {
    const double da = par[1], qa = 2.0 * par[2];
    const double ade = (0.5 / par[3]) * (0.5 / par[3]), aqe = (0.5 / par[4]) * (0.5 / par[4]);
    const double s2 = sqrt((r + da) * (r + da) + ade), s3 = sqrt((r - da) * (r - da) + ade);
    const double s4 = sqrt((r + qa) * (r + qa) + aqe), s5 = sqrt((r - qa) * (r - qa) + aqe);
    const double s6 = sqrt(r * r + aqe), s7 = sqrt(r * r + aqe + qa * qa);
    t.ri2 = 0.5 * ev / s2 - 0.5 * ev / s3;
    t.ri3 = t.ri1 + 0.25 * ev / s4 + 0.25 * ev / s5 - 0.5 * ev / s6;
    t.ri4 = t.ri1 + 0.5 * ev / s7 - 0.5 * ev / s6;
    if (want_grad) {
      t.d2 = (-0.5 * ev * (r + da) / (s2 * s2 * s2) + 0.5 * ev * (r - da) / (s3 * s3 * s3)) / a0;
      t.d3 = t.d1 + (-0.25 * ev * (r + qa) / (s4 * s4 * s4) - 0.25 * ev * (r - qa) / (s5 * s5 * s5) +
                     0.5 * ev * r / (s6 * s6 * s6)) / a0;
      t.d4 = t.d1 + (-0.5 * ev * r / (s7 * s7 * s7) + 0.5 * ev * r / (s6 * s6 * s6)) / a0;
    }
  }
  t.sw = 1.0;
  t.dsw = 0.0;
  if (roff > 0.0) {
    if (t.rr >= roff) {
      t.sw = 0.0;
    } else if (t.rr > ron) {
      const double x = (t.rr - ron) / (roff - ron);
      t.sw = 1.0 - 10.0 * x * x * x + 15.0 * x * x * x * x - 6.0 * x * x * x * x * x;
      t.dsw = (-30.0 * x * x + 60.0 * x * x * x - 30.0 * x * x * x * x) / (roff - ron);
    }
  }
}

// Slots: 0 (s,s); 1-3 (p_c,s); 4-9 (p_c,p_e), e <= c, in the order (1,1) (2,1) (2,2) (3,1) (3,2) (3,3); 10-14 the
// d diagonal.  Value v and gradient dv (eV, eV/A) of slot k, as qmmm_pair sets them.
__device__ __forceinline__ void slot_value(int k, const PairTerms &t, bool want_grad, double &v, double dv[3]) {
  const double *u = t.u;
  if (k == 0 || k >= 10) {
    v = t.ri1;
    if (want_grad)
      for (int a = 0; a < 3; a++) dv[a] = t.d1 * u[a];
    return;
  }
  if (k <= 3) {
    const int c = k - 1;
    v = -t.ri2 * u[c];
    if (want_grad) {
      for (int a = 0; a < 3; a++) dv[a] = -t.d2 * u[c] * u[a] + t.ri2 * u[c] * u[a] / t.rr;
      dv[c] = dv[c] - t.ri2 / t.rr;
    }
    return;
  }
  const int ce[6][2] = {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}};
  const int c = ce[k - 4][0], e = ce[k - 4][1];
  const double delta = (c == e) ? 1.0 : 0.0;
  v = t.ri4 * delta + (t.ri3 - t.ri4) * u[c] * u[e];
  if (want_grad) {
    for (int a = 0; a < 3; a++)
      dv[a] = (t.d4 * delta + (t.d3 - t.d4) * u[c] * u[e]) * u[a] -
              (t.ri3 - t.ri4) * 2.0 * u[c] * u[e] * u[a] / t.rr;
    dv[c] = dv[c] + (t.ri3 - t.ri4) * u[e] / t.rr;
    dv[e] = dv[e] + (t.ri3 - t.ri4) * u[c] / t.rr;
  }
}

__device__ __forceinline__ int nslots(int nao) { return nao >= 9 ? 15 : (nao >= 4 ? 10 : 1); }

struct QmmmArgs {
  int numat, n_mm;
  int chunk, nchunk;      // charges per chunk, chunks
  int achunk, nachunk;    // QM atoms per chunk, chunks (the charges' sums)
  const int *nao;
  const double *par;      // 6 per QM atom: am, dd, qq, ad, aq, tore
  const double *coord;    // 3 per QM atom
  const double *mm_xyz;   // 3 per charge
  const double *mm_q;
  const double *pk;       // kSlots per QM atom: weighted density (qmmm_grad's wk * p)
  double ron, roff, ev, a0, fpc9;
  double *part;           // per chunk and QM atom: kSlots + 1 (hcore) or 3 (grad)
  double *out_atom;       // per QM atom: kSlots + 1 (hcore: Sum q S v, Sum tore q S gam) or 3 (grad)
  double *part_mm;        // per chunk of QM atoms and charge: gradient (3), potential, energy
  double *mm_grad, *mm_pot, *mm_e;
};

// One thread per QM atom i and chunk c of charges: Sum_j q_j S v_k(i, j) for every slot and Sum_j tore q_j S gam
__global__ void qmmm_hcore_partial(QmmmArgs a) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= (long long)a.numat * a.nchunk) return;
  const int i = (int)(t % a.numat), c = (int)(t / a.numat);
  const int nao = a.nao[i], ns = nslots(nao);
  const double *par = a.par + 6 * i;
  double acc[kSlots], en = 0.0;
  for (int k = 0; k < kSlots; k++) acc[k] = 0.0;
  const int j1 = (c + 1) * a.chunk < a.n_mm ? (c + 1) * a.chunk : a.n_mm;
  for (int j = c * a.chunk; j < j1; j++) {
    double d[3];
    for (int x = 0; x < 3; x++) d[x] = a.mm_xyz[3 * j + x] - a.coord[3 * i + x];
    PairTerms pt;
    pair_terms(nao, par, d, a.ev, a.a0, a.ron, a.roff, false, pt);
    if (pt.sw == 0.0) continue;
    const double qs = a.mm_q[j] * pt.sw;
    for (int k = 0; k < ns; k++) {
      double v, dv[3];
      slot_value(k, pt, false, v, dv);
      acc[k] += qs * v;
    }
    en += par[5] * a.mm_q[j] * pt.sw * pt.ri1;
  }
  double *p = a.part + (long long)t * (kSlots + 1);
  for (int k = 0; k < kSlots; k++) p[k] = acc[k];
  p[kSlots] = en;
}

// One thread per QM atom: the chunks of qmmm_hcore_partial or qmmm_grad_partial in order (width values each)
__global__ void qmmm_sum_chunks(QmmmArgs a, int width) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= a.numat) return;
  for (int k = 0; k < width; k++) {
    double s = 0.0;
    for (int c = 0; c < a.nchunk; c++) s += a.part[((long long)c * a.numat + i) * width + k];
    a.out_atom[(long long)i * width + k] = s;
  }
}

// dE/dd for one QM atom i and charge j (d = x(J) - x(A), kcal/(mol A)); e and pot: the pair's energy terms
__device__ __forceinline__ void pair_gradient(const QmmmArgs &a, int i, int j, double g[3], double &e,
                                              double &pot, bool &on) {
  const int nao = a.nao[i], ns = nslots(nao);
  const double *par = a.par + 6 * i;
  double d[3];
  for (int x = 0; x < 3; x++) d[x] = a.mm_xyz[3 * j + x] - a.coord[3 * i + x];
  PairTerms pt;
  pair_terms(nao, par, d, a.ev, a.a0, a.ron, a.roff, true, pt);
  on = pt.sw != 0.0;
  if (!on) return;
  double pv = 0.0, dpv[3] = {0.0, 0.0, 0.0};
  const double *pk = a.pk + (long long)kSlots * i;
  for (int k = 0; k < ns; k++) {
    double v, dv[3];
    slot_value(k, pt, true, v, dv);
    pv = pv + pk[k] * v;
    for (int x = 0; x < 3; x++) dpv[x] = dpv[x] + pk[k] * dv[x];
  }
  const double tore = par[5], q = a.mm_q[j];
  const double core = tore * pt.ri1 - pv;
  e = q * pt.sw * core;
  pot = pt.sw * core * a.fpc9;
  for (int x = 0; x < 3; x++)
    g[x] = q * (pt.dsw * pt.u[x] * core + pt.sw * (tore * pt.d1 * pt.u[x] - dpv[x])) * a.fpc9;
}

// One thread per charge j and chunk of QM atoms: the charge's gradient, potential (kcal/(mol e)) and energy over
// the chunk
__global__ void qmmm_grad_mm_partial(QmmmArgs a) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= (long long)a.n_mm * a.nachunk) return;
  const int j = (int)(t % a.n_mm), c = (int)(t / a.n_mm);
  double gj[3] = {0.0, 0.0, 0.0}, ej = 0.0, potj = 0.0;
  const int i1 = (c + 1) * a.achunk < a.numat ? (c + 1) * a.achunk : a.numat;
  for (int i = c * a.achunk; i < i1; i++) {
    double g[3], e, pot;
    bool on;
    pair_gradient(a, i, j, g, e, pot, on);
    if (!on) continue;
    ej = ej + e;
    potj = potj + pot;
    for (int x = 0; x < 3; x++) gj[x] = gj[x] + g[x];
  }
  double *p = a.part_mm + (long long)t * 5;
  for (int x = 0; x < 3; x++) p[x] = gj[x];
  p[3] = potj;
  p[4] = ej;
}

// One thread per charge: the chunks of qmmm_grad_mm_partial in order
__global__ void qmmm_grad_mm_sum(QmmmArgs a) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= a.n_mm) return;
  double s[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int c = 0; c < a.nachunk; c++)
    for (int k = 0; k < 5; k++) s[k] += a.part_mm[((long long)c * a.n_mm + j) * 5 + k];
  for (int x = 0; x < 3; x++) a.mm_grad[3 * j + x] = s[x];
  a.mm_pot[j] = s[3];
  a.mm_e[j] = s[4];
}

// One thread per QM atom i and chunk c of charges: Sum_j dE/dd (to be subtracted from the QM atom's gradient)
__global__ void qmmm_grad_partial(QmmmArgs a) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= (long long)a.numat * a.nchunk) return;
  const int i = (int)(t % a.numat), c = (int)(t / a.numat);
  double s[3] = {0.0, 0.0, 0.0};
  const int j1 = (c + 1) * a.chunk < a.n_mm ? (c + 1) * a.chunk : a.n_mm;
  for (int j = c * a.chunk; j < j1; j++) {
    double g[3], e, pot;
    bool on;
    pair_gradient(a, i, j, g, e, pot, on);
    if (!on) continue;
    for (int x = 0; x < 3; x++) s[x] = s[x] + g[x];
  }
  double *p = a.part + (long long)t * 3;
  for (int x = 0; x < 3; x++) p[x] = s[x];
}

// Device buffers kept between calls, grown as needed
struct Buffer {
  void *ptr = nullptr;
  size_t cap = 0;
  bool reserve(size_t bytes) {
    if (bytes <= cap) return true;
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
    cap = 0;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) return false;
    cap = bytes;
    return true;
  }
  bool put(const void *host, size_t bytes) {
    return reserve(bytes) && (bytes == 0 || cudaMemcpy(ptr, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  }
};

Buffer b_nao, b_par, b_coord, b_xyz, b_q, b_pk, b_part, b_part_mm, b_atom, b_grad, b_pot, b_e;

int blocks(long long n) { return (int)((n + kThreads - 1) / kThreads); }

bool upload_common(QmmmArgs &a, int numat, int n_mm, const int *nao, const double *par, const double *coord,
                   const double *mm_xyz, const double *mm_q) {
  const size_t na = (size_t)numat, nm = (size_t)n_mm;
  if (!b_nao.put(nao, na * sizeof(int)) || !b_par.put(par, 6 * na * sizeof(double)) ||
      !b_coord.put(coord, 3 * na * sizeof(double)) || !b_xyz.put(mm_xyz, 3 * nm * sizeof(double)) ||
      !b_q.put(mm_q, nm * sizeof(double)))
    return false;
  a.numat = numat;
  a.n_mm = n_mm;
  // chunks of >= 8 charges (>= 4 QM atoms), as many as give ~kTargetThreads threads
  long long nc = std::min<long long>((n_mm + 7) / 8, std::max<long long>(1, kTargetThreads / numat));
  a.chunk = (int)((n_mm + nc - 1) / nc);
  a.nchunk = (n_mm + a.chunk - 1) / a.chunk;
  nc = std::min<long long>((numat + 3) / 4, std::max<long long>(1, kTargetThreads / n_mm));
  a.achunk = (int)((numat + nc - 1) / nc);
  a.nachunk = (numat + a.achunk - 1) / a.achunk;
  a.nao = static_cast<const int *>(b_nao.ptr);
  a.par = static_cast<const double *>(b_par.ptr);
  a.coord = static_cast<const double *>(b_coord.ptr);
  a.mm_xyz = static_cast<const double *>(b_xyz.ptr);
  a.mm_q = static_cast<const double *>(b_q.ptr);
  return true;
}

int finish(const char *what) {
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[GPU ERROR] QM/MM %s: %s\n", what, cudaGetErrorString(err));
    return 2;
  }
  return 0;
}

}  // namespace

// Fortran entry points (src/MOZYME/qmmm.F90).  QM atom i: nao(i) orbitals, par(6, i) = am, dd, qq, ad, aq, tore;
// coord(3, i), mm_xyz(3, j), mm_q(j) in Angstroms and electrons.  Return 0 on success, 1 on bad arguments, 2 on
// a CUDA failure (the caller then runs the CPU loops).
//
// hcore: hv(kSlots, i) = Sum_j q_j S v_k(i, j) (eV; h of the slot's element gets -hv) and en(i) = Sum_j tore q_j
// S gam (eV, added to the core energy).
extern "C" int mopac_cuda_qmmm_hcore(int numat, int n_mm, const int *nao, const double *par, const double *coord,
                                     const double *mm_xyz, const double *mm_q, double ron, double roff, double ev,
                                     double a0, double *hv, double *en) {
  if (numat <= 0 || n_mm <= 0 || !nao || !par || !coord || !mm_xyz || !mm_q || !hv || !en) return 1;
  QmmmArgs a = {};
  if (!upload_common(a, numat, n_mm, nao, par, coord, mm_xyz, mm_q)) return 2;
  const int width = kSlots + 1;
  const long long nt = (long long)numat * a.nchunk;
  if (!b_part.reserve((size_t)nt * width * sizeof(double)) ||
      !b_atom.reserve((size_t)numat * width * sizeof(double)))
    return 2;
  a.ron = ron;
  a.roff = roff;
  a.ev = ev;
  a.a0 = a0;
  a.part = static_cast<double *>(b_part.ptr);
  a.out_atom = static_cast<double *>(b_atom.ptr);
  qmmm_hcore_partial<<<blocks(nt), kThreads>>>(a);
  qmmm_sum_chunks<<<blocks(numat), kThreads>>>(a, width);
  int code = finish("hcore");
  if (code) return code;
  static double *host = nullptr;
  static size_t host_cap = 0;
  const size_t n = (size_t)numat * width;
  if (n > host_cap) {
    delete[] host;
    host = new double[n];
    host_cap = n;
  }
  if (cudaMemcpy(host, b_atom.ptr, n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
  for (int i = 0; i < numat; i++) {
    for (int k = 0; k < kSlots; k++) hv[(size_t)i * kSlots + k] = host[(size_t)i * width + k];
    en[i] = host[(size_t)i * width + kSlots];
  }
  return 0;
}

// grad: pk(kSlots, i) = wk * p of the slot's element; gq(3, i) = Sum_j dE/dd (kcal/(mol A); the QM atom's
// gradient gets -gq), mm_grad(3, j) and mm_pot(j) as in qmmm_grad, *e_out = Sum_j q_j S (tore gam - p.v) (eV).
extern "C" int mopac_cuda_qmmm_grad(int numat, int n_mm, const int *nao, const double *par, const double *coord,
                                    const double *mm_xyz, const double *mm_q, const double *pk, double ron,
                                    double roff, double ev, double a0, double fpc9, double *gq, double *mm_grad,
                                    double *mm_pot, double *e_out) {
  if (numat <= 0 || n_mm <= 0 || !nao || !par || !coord || !mm_xyz || !mm_q || !pk || !gq || !mm_grad ||
      !mm_pot || !e_out)
    return 1;
  QmmmArgs a = {};
  if (!upload_common(a, numat, n_mm, nao, par, coord, mm_xyz, mm_q)) return 2;
  const size_t na = (size_t)numat, nm = (size_t)n_mm;
  const long long nt = (long long)numat * a.nchunk;
  if (!b_pk.put(pk, kSlots * na * sizeof(double)) || !b_part.reserve((size_t)nt * 3 * sizeof(double)) ||
      !b_part_mm.reserve((size_t)a.nachunk * nm * 5 * sizeof(double)) ||
      !b_atom.reserve(3 * na * sizeof(double)) || !b_grad.reserve(3 * nm * sizeof(double)) ||
      !b_pot.reserve(nm * sizeof(double)) || !b_e.reserve(nm * sizeof(double)))
    return 2;
  a.pk = static_cast<const double *>(b_pk.ptr);
  a.ron = ron;
  a.roff = roff;
  a.ev = ev;
  a.a0 = a0;
  a.fpc9 = fpc9;
  a.part = static_cast<double *>(b_part.ptr);
  a.out_atom = static_cast<double *>(b_atom.ptr);
  a.part_mm = static_cast<double *>(b_part_mm.ptr);
  a.mm_grad = static_cast<double *>(b_grad.ptr);
  a.mm_pot = static_cast<double *>(b_pot.ptr);
  a.mm_e = static_cast<double *>(b_e.ptr);
  qmmm_grad_mm_partial<<<blocks((long long)n_mm * a.nachunk), kThreads>>>(a);
  qmmm_grad_mm_sum<<<blocks(n_mm), kThreads>>>(a);
  qmmm_grad_partial<<<blocks(nt), kThreads>>>(a);
  qmmm_sum_chunks<<<blocks(numat), kThreads>>>(a, 3);
  int code = finish("gradient");
  if (code) return code;
  if (cudaMemcpy(gq, b_atom.ptr, 3 * na * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(mm_grad, b_grad.ptr, 3 * nm * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(mm_pot, b_pot.ptr, nm * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 2;
  static double *host_e = nullptr;
  static size_t host_cap = 0;
  if (nm > host_cap) {
    delete[] host_e;
    host_e = new double[nm];
    host_cap = nm;
  }
  if (cudaMemcpy(host_e, b_e.ptr, nm * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
  double e = 0.0;
  for (size_t j = 0; j < nm; j++) e += host_e[j];
  *e_out = e;
  return 0;
}

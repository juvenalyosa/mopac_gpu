"""The work over the MM charges of the QM/MM force (openmm_mopac.MopacQMMMForce) in five CUDA kernels.

With array operations (MopacQMMMForce._select_arrays / _forces_arrays, CuPy) a step of trypsin-BPTI (46 800 MM
charges, Colab A100) launched ~190 small operations, each ~20 us of Python and launch overhead on the Xeon host,
~8 ms in all against ~0.3 ms of GPU work.  MMChargesKernels does the same steps with the same arithmetic in five
kernels; the charges of each switching group must be contiguous in the arrays (the atoms of a residue are, in an
OpenMM topology):

- mm_groups: images the group next to the QM centre (its first charge to the image nearest the centre, the others
  to the image nearest the first), its centre, and its switching weight (smooth minimum of the distances from
  the centre to the QM atoms);
- mm_select: the switched charges, in the order of the array code (cumulative sum of the counts);
- mm_forces (after MOPAC): the force on every MM charge (long-range term, MOPAC's gradient, switching);
- mm_qm_partial, mm_qm_sum: the switching force on each QM atom, summed over chunks of CHUNK groups (one thread
  per QM atom and chunk) and then over the chunks in order (one thread per QM atom).  One thread per QM atom over
  all groups took 1.8 ms of GPU time per step in trypsin-BPTI.

Sums over the charges of a group run in the order of the array code and the kernels are compiled without fused
multiply-add; sums over QM atoms or groups are ordered differently there, so the results agree to rounding.
The source also compiles as C++ (a test runs it thread by thread on the CPU)."""

from __future__ import annotations

import numpy as np

_SRC = r"""
extern "C" {

__global__ void mm_groups(const double *pos, const int *idx, const int *gstart, const int *gsize,
                          const double *xr, int ng, int nq, double cx, double cy, double cz, double lx, double ly,
                          double lz, double rmax, double ron, double roff, double lam, double *xm, double *cg,
                          double *w, double *dw, double *rmin, double *sumex, long long *cnt)
{
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= ng) return;
  const double c[3] = {cx, cy, cz}, L[3] = {lx, ly, lz};
  int s = gstart[g], m = gsize[g];
  double ref[3], sum[3] = {0.0, 0.0, 0.0}, cgk[3];
  for (int k = 0; k < 3; k++) {
    double d = pos[3 * (long long)idx[s] + k] * 10.0 - c[k];
    ref[k] = c[k] + (d - L[k] * rint(d / L[k]));
  }
  for (int j = s; j < s + m; j++)
    for (int k = 0; k < 3; k++) {
      double d = pos[3 * (long long)idx[j] + k] * 10.0 - ref[k];
      double x = ref[k] + (d - L[k] * rint(d / L[k]));
      xm[3 * (long long)j + k] = x;
      sum[k] += x;
    }
  for (int k = 0; k < 3; k++) {
    cgk[k] = sum[k] / (double)m;
    cg[3 * (long long)g + k] = cgk[k];
  }
  w[g] = 0.0; dw[g] = 0.0; rmin[g] = 0.0; sumex[g] = 1.0; cnt[g] = 0;
  double e0 = cgk[0] - cx, e1 = cgk[1] - cy, e2 = cgk[2] - cz;
  if (nq == 0 || !(sqrt(e0 * e0 + e1 * e1 + e2 * e2) < rmax)) return;
  double rm = 1e300, se = 0.0;
  for (int a = 0; a < nq; a++) {
    double v0 = cgk[0] - xr[3 * a], v1 = cgk[1] - xr[3 * a + 1], v2 = cgk[2] - xr[3 * a + 2];
    rm = fmin(rm, sqrt(v0 * v0 + v1 * v1 + v2 * v2));
  }
  for (int a = 0; a < nq; a++) {
    double v0 = cgk[0] - xr[3 * a], v1 = cgk[1] - xr[3 * a + 1], v2 = cgk[2] - xr[3 * a + 2];
    se += exp(-(sqrt(v0 * v0 + v1 * v1 + v2 * v2) - rm) / lam);
  }
  double ds = rm - lam * log(se);
  double x = fmin(fmax((ds - ron) / (roff - ron), 0.0), 1.0);
  double x2 = x * x, x3 = x2 * x, x4 = x3 * x;
  double ww = 1.0 - 10 * x3 + 15 * x4 - 6 * (x4 * x);
  if (!(ww > 0.0)) return;
  w[g] = ww;
  dw[g] = (x > 0.0 && x < 1.0) ? (-30 * x2 + 60 * x3 - 30 * x4) / (roff - ron) : 0.0;
  rmin[g] = rm;
  sumex[g] = se;
  cnt[g] = m;
}

__global__ void mm_select(const long long *incl, const long long *cnt, const int *gstart, const double *xm,
                          const double *q, const double *w, int ng, long long nsel, double *out)
{
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= ng || cnt[g] == 0) return;
  long long off = incl[g] - cnt[g];
  int s = gstart[g], m = (int)cnt[g];
  for (int j = 0; j < m; j++) {
    for (int k = 0; k < 3; k++) out[3 * (off + j) + k] = xm[3 * (long long)(s + j) + k];
    out[3 * nsel + off + j] = q[s + j] * w[g];
  }
}

__global__ void mm_forces(const double *gall, int has_far, const double *up, long long nsel, const int *idx,
                          const int *gstart, const int *gsize, const double *q, const long long *incl,
                          const long long *cnt, const double *dw, const double *rmin, const double *sumex,
                          const double *cg, const double *xr, int ng, int nq, double lam, double conv,
                          double *coef_out, double *f)
{
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= ng) return;
  int s = gstart[g], m = gsize[g];
  bool on = cnt[g] > 0;
  long long off = incl[g] - cnt[g];
  const double *gm = up, *pot = up + 3 * nsel;
  double gmem[3] = {0.0, 0.0, 0.0};
  coef_out[g] = 0.0;
  if (on) {
    double dEdw = 0.0;
    for (int j = 0; j < m; j++) dEdw += q[s + j] * pot[off + j];
    double coef = dEdw * dw[g];
    coef_out[g] = coef;
    double c0 = cg[3 * (long long)g], c1 = cg[3 * (long long)g + 1], c2 = cg[3 * (long long)g + 2];
    double t[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < nq; a++) {
      double v0 = c0 - xr[3 * a], v1 = c1 - xr[3 * a + 1], v2 = c2 - xr[3 * a + 2];
      double r = sqrt(v0 * v0 + v1 * v1 + v2 * v2);
      double pA = exp(-(r - rmin[g]) / lam) / sumex[g];
      t[0] += pA * (v0 / r); t[1] += pA * (v1 / r); t[2] += pA * (v2 / r);
    }
    for (int k = 0; k < 3; k++) gmem[k] = (coef * t[k]) / (double)m;
  }
  for (int j = 0; j < m; j++) {
    long long a = idx[s + j];
    for (int k = 0; k < 3; k++) {
      double fk = has_far ? gall[3 * (long long)(s + j) + k] * -conv : 0.0;
      if (on) {
        fk += gm[3 * (off + j) + k] * -conv;
        fk += gmem[k] * -conv;
      }
      f[3 * a + k] = fk;
    }
  }
}

__global__ void mm_qm_partial(const double *cg, const double *coef, const long long *cnt, const double *rmin,
                              const double *sumex, const double *xr, int ng, int nq, int chunk, int nchunk,
                              double lam, double *part)
{
  long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= (long long)nq * nchunk) return;
  int a = (int)(t % nq), c = (int)(t / nq);
  int g1 = (c + 1) * chunk < ng ? (c + 1) * chunk : ng;
  double s0 = 0.0, s1 = 0.0, s2 = 0.0;
  for (int g = c * chunk; g < g1; g++) {
    if (cnt[g] == 0) continue;
    double v0 = cg[3 * (long long)g] - xr[3 * a], v1 = cg[3 * (long long)g + 1] - xr[3 * a + 1],
           v2 = cg[3 * (long long)g + 2] - xr[3 * a + 2];
    double r = sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    double tg = -coef[g] * (exp(-(r - rmin[g]) / lam) / sumex[g]);
    s0 += tg * (v0 / r); s1 += tg * (v1 / r); s2 += tg * (v2 / r);
  }
  part[3 * t] = s0; part[3 * t + 1] = s1; part[3 * t + 2] = s2;
}

__global__ void mm_qm_sum(const double *part, int nq, int nchunk, double *ga)
{
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= nq) return;
  double s0 = 0.0, s1 = 0.0, s2 = 0.0;
  for (int c = 0; c < nchunk; c++) {
    const double *p = part + 3 * ((long long)c * nq + a);
    s0 += p[0]; s1 += p[1]; s2 += p[2];
  }
  ga[3 * a] = s0; ga[3 * a + 1] = s1; ga[3 * a + 2] = s2;
}

}
"""

KERNELS = ("mm_groups", "mm_select", "mm_forces", "mm_qm_partial", "mm_qm_sum")
BLOCK = 128
CHUNK = 64                             # groups per thread in mm_qm_partial


class CuPyBackend:
    """device arrays, transfers and kernel launches with CuPy"""

    def __init__(self):
        import cupy
        import cupyx
        self.cp, self.cpx = cupy, cupyx
        mod = cupy.RawModule(code=_SRC, options=("--fmad=false",))
        self.fn = {k: mod.get_function(k) for k in KERNELS}

    def zeros(self, shape, dtype=np.float64):
        return self.cp.zeros(shape, dtype)

    def asarray(self, a):
        return self.cp.asarray(a)

    def pinned(self, shape):
        a = self.cpx.empty_pinned(shape, dtype=np.float64)
        a[...] = 0.0
        return a

    def upload(self, dst, src):
        dst.set(src)

    def download(self, src, out):
        src.get(out=out)
        return out

    def cumsum(self, a):
        return self.cp.cumsum(a)

    def launch(self, name, n, args):
        if n > 0:
            self.fn[name](((n + BLOCK - 1) // BLOCK,), (BLOCK,), args)


class MMChargesKernels:
    """select() and forces() of MopacQMMMForce's MM charges in the kernels above (orthorhombic boxes).
    mm_idx: atom index of each MM charge; group_of: its switching group (non-decreasing); group_size: charges per
    group; q: the charges; nq: real QM atoms (the switching distances use those, not the link atoms)"""

    def __init__(self, mm_idx, group_of, group_size, q, n_atoms, nq, lam=0.5, backend=None):
        group_of = np.asarray(group_of)
        if np.any(np.diff(group_of) < 0):
            raise ValueError("MMChargesKernels: the charges of each switching group must be contiguous")
        be = self.be = backend or CuPyBackend()
        nm, ng = len(mm_idx), len(group_size)
        self.nm, self.ng, self.n_atoms, self.nq, self.lam = nm, ng, n_atoms, nq, lam
        self.idx = be.asarray(np.asarray(mm_idx, dtype=np.int32))
        self.gstart = be.asarray(np.searchsorted(group_of, np.arange(ng)).astype(np.int32))
        self.gsize = be.asarray(np.asarray(group_size, dtype=np.int32))
        self.q = be.asarray(np.asarray(q, dtype=np.float64))
        self.pos = be.zeros((n_atoms, 3))
        self.xr = be.zeros((max(nq, 1), 3))
        self.xm, self.cg = be.zeros((nm, 3)), be.zeros((ng, 3))
        self.w, self.dw, self.rmin, self.sumex, self.coef = (be.zeros(ng) for _ in range(5))
        self.cnt = be.zeros(ng, np.int64)
        self.sel = be.zeros(4 * nm)                      # switched positions (3 n), then charges (n)
        self.up = be.zeros(4 * nm)                       # MOPAC's gradient on them (3 n), then dE/dq (n)
        self.out = be.zeros((n_atoms + nq, 3))           # forces on all atoms, then on the QM atoms
        self.nchunk = max(1, (ng + CHUNK - 1) // CHUNK)
        self.part = be.zeros(3 * max(nq, 1) * self.nchunk)  # mm_qm_partial: per chunk of groups and QM atom
        self.h_sel, self.h_out = be.pinned(4 * nm), be.pinned((n_atoms + nq, 3))
        self.incl, self.n_sel = None, 0

    def select(self, pos_nm, xr, c, L, rmax, ron, roff):
        """xm (all MM charges, device), xs and qs (switched charges, device), the same two on the host"""
        be, ng, nq = self.be, self.ng, self.nq
        f64, i32 = np.float64, np.int32
        if len(xr) != nq:
            raise ValueError("MMChargesKernels.select: %d QM atoms, expected %d" % (len(xr), nq))
        be.upload(self.pos, np.ascontiguousarray(pos_nm, dtype=np.float64))
        if nq:
            be.upload(self.xr, np.ascontiguousarray(xr, dtype=np.float64))
        be.launch("mm_groups", ng, (self.pos, self.idx, self.gstart, self.gsize, self.xr, i32(ng), i32(nq),
                                    *(f64(v) for v in c), *(f64(v) for v in L), f64(rmax), f64(ron), f64(roff),
                                    f64(self.lam), self.xm, self.cg, self.w, self.dw, self.rmin, self.sumex,
                                    self.cnt))
        self.incl = be.cumsum(self.cnt)
        n = self.n_sel = int(self.incl[-1]) if ng else 0
        h = self.h_sel[:4 * n]
        if n:
            be.launch("mm_select", ng, (self.incl, self.cnt, self.gstart, self.xm, self.q, self.w, i32(ng),
                                        np.int64(n), self.sel))
            be.download(self.sel[:4 * n], h)
        return (self.xm, self.sel[:3 * n].reshape(n, 3), self.sel[3 * n:4 * n], h[:3 * n].reshape(n, 3).copy(),
                h[3 * n:].copy())

    def forces(self, g_all, gm, pot, conv):
        """forces (kJ/(mol nm)) on all atoms from the MM-charge terms, and the switching gradient on the QM
        atoms (kcal/(mol A); None without switched charges); g_all: gradient of the long-range term on all MM
        charges (device) or None, gm and pot: MOPAC's gradient and dE/dq of the switched charges"""
        be, ng, nq, n = self.be, self.ng, self.nq, self.n_sel
        f64, i32 = np.float64, np.int32
        if n:
            be.upload(self.up[:4 * n], np.concatenate([np.ravel(gm), np.ravel(pot)]).astype(np.float64))
        be.launch("mm_forces", ng, (self.xm if g_all is None else g_all, i32(g_all is not None), self.up,
                                    np.int64(n), self.idx, self.gstart, self.gsize, self.q, self.incl, self.cnt,
                                    self.dw, self.rmin, self.sumex, self.cg, self.xr, i32(ng), i32(nq),
                                    f64(self.lam), f64(conv), self.coef, self.out))
        be.launch("mm_qm_partial", nq * self.nchunk, (self.cg, self.coef, self.cnt, self.rmin, self.sumex, self.xr,
                                                      i32(ng), i32(nq), i32(CHUNK), i32(self.nchunk), f64(self.lam),
                                                      self.part))
        be.launch("mm_qm_sum", nq, (self.part, i32(nq), i32(self.nchunk), self.out[self.n_atoms:]))
        h = be.download(self.out, self.h_out)
        return np.array(h[:self.n_atoms]), (h[self.n_atoms:].copy() if n else None)

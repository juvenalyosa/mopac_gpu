"""Smooth particle-mesh Ewald (Essmann et al. 1995) for the long-range QM-MM electrostatics (openmm_mopac.py).

Only what QM/MM needs: the reciprocal-space potential of the MM charges at the QM atoms, and its
derivatives with respect to the positions of both (B-spline interpolation with analytic derivatives, so
that the forces are the exact derivatives of the energy as computed on the grid).  Orthorhombic boxes;
Angstroms; potentials in e/Angstrom (multiply by the Coulomb constant).

The grid work runs on the GPU with CuPy when gpu=True (positions and charges in, potentials and gradients
out, as NumPy arrays), else with NumPy (and SciPy's FFTs) on the CPU.

FarFieldNumPy and FarFieldGPU compute the whole long-range QM-MM term of openmm_mopac.py (reciprocal space,
real-space erfc sum, minus the bare Coulomb potential of the switched charges MOPAC sees) and its gradient;
FarFieldGPU does it with a few fused CUDA kernels (spread, gather, pair sums) instead of array operations, so
that a step costs two charge spreadings, two FFT pairs and four small transfers.
"""

from __future__ import annotations

import numpy as np

from scipy.special import erfc

# Coulomb constant, kcal A/(mol e^2) (OpenMM's ONE_4PI_EPS0 in kJ nm/(mol e^2))
COULOMB = 138.935456 * 10.0 / 4.184

try:                                   # multithreaded FFTs when SciPy is there
    import scipy.fft as _fft
    _FFT_KW = {"workers": -1}
except ImportError:                    # pragma: no cover
    _fft = np.fft
    _FFT_KW = {}


_gpu_ok = None


def gpu_available():
    """True when CuPy is installed, sees a CUDA device and runs a small PME grid to the same potentials and
    gradients as NumPy.  A CuPy that does not match the CUDA installed (cupy-cuda12x on CUDA 13, or two CuPy
    packages at once) imports and counts devices but can fail at the first kernel it compiles, so every
    operation the grid uses (bincount, padding, FFTs, the gather kernel) is tried once here."""
    global _gpu_ok
    if _gpu_ok is None:
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            rng = np.random.default_rng(1)
            box = np.array([12.0, 13.0, 14.0])
            x, q, y = rng.random((20, 3)) * box, rng.standard_normal(20), rng.random((3, 3)) * box
            out = []
            for gpu in (False, True):
                grid = PMEGrid(box, 0.3, gpu=gpu)
                phi = grid.potential(Stencil(grid, x), q)
                st = Stencil(grid, y)
                out.append(np.concatenate([grid.value(st, phi), grid.gradient(st, phi).ravel()]))
            _gpu_ok = bool(np.allclose(out[0], out[1], rtol=1e-8, atol=1e-10))
        except Exception:
            _gpu_ok = False
    return _gpu_ok


_fused_ok = None


def fused_far_field_available():
    """True when the CUDA kernels of FarFieldGPU compile and agree with FarFieldNumPy on a random periodic
    system (potentials and every gradient); needs gpu_available()"""
    global _fused_ok
    if _fused_ok is None:
        try:
            _fused_ok = gpu_available() and _far_field_check()
        except Exception:
            _fused_ok = False
    return _fused_ok


def _far_field_check():
    rng = np.random.default_rng(2)
    L = np.array([40.0, 41.0, 42.0])
    c = L / 2
    xm, qm = rng.random((400, 3)) * L, rng.standard_normal(400)
    xq = c + rng.standard_normal((12, 3))
    xs, qs = c + 4.0 * rng.standard_normal((30, 3)), rng.standard_normal(30)
    Q = 0.3 * rng.standard_normal(12)
    out = []
    for gpu in (False, True):
        grid = PMEGrid(L, 0.25, 1.2, 4, gpu=gpu)
        ff = (FarFieldGPU if gpu else FarFieldNumPy)(grid, 0.25, 13.0)
        phi = ff.potential(xq, xm, qm, xs, qs, c, L)
        gq, gm, pot = np.zeros((12, 3)), np.zeros((30, 3)), np.zeros(30)
        g_all = ff.gradient(Q, gq, gm, pot)
        out.append(np.concatenate([phi, gq.ravel(), gm.ravel(), pot, g_all.ravel()]))
    return bool(np.allclose(out[0], out[1], rtol=1e-8, atol=1e-8))


def bspline(w, n):
    """Cardinal B-spline weights of order n for fractional offsets w: (..., n), element k = M_n(w + n-1-k)"""
    m = np.zeros(w.shape + (n,))
    m[..., 0] = 1.0 - w
    m[..., 1] = w
    for k in range(3, n + 1):
        new = np.zeros_like(m)
        new[..., k - 1] = w * m[..., k - 2] / (k - 1)
        for j in range(1, k - 1):
            new[..., k - 1 - j] = ((w + j) * m[..., k - 2 - j] + (k - j - w) * m[..., k - 1 - j]) / (k - 1)
        new[..., 0] = (1.0 - w) * m[..., 0] / (k - 1)
        m = new
    return m


def _bmod(K, n):
    w = bspline(np.zeros(1), n)[0][::-1]                 # M_n(1), ..., M_n(n)
    arg = 2j * np.pi * np.outer(np.arange(K), np.arange(n - 1)) / K
    return 1.0 / np.abs((w[None, 1:] * np.exp(arg)).sum(axis=1)) ** 2


def _weights(w, n, xp=np):
    """weights and their derivatives (d/dw) for fractional offsets w: (..., n) each"""
    if n == 4:
        v = 1.0 - w
        w2, w3 = w * w, w * w * w
        m = xp.stack([v * v * v, 4.0 - 6.0 * w2 + 3.0 * w3, 1.0 + 3.0 * (w + w2 - w3), w3], axis=-1) / 6.0
        d = xp.stack([-0.5 * v * v, -2.0 * w + 1.5 * w2, 0.5 + w - 1.5 * w2, 0.5 * w2], axis=-1)
        return m, d
    wh = xp.asnumpy(w) if xp is not np else w
    m = bspline(wh, n)
    m1 = bspline(wh, n - 1)
    d = np.zeros_like(m)
    d[..., 1:] += m1
    d[..., :-1] -= m1
    return xp.asarray(m), xp.asarray(d)


class Stencil:
    """B-spline weights (and their derivatives) of a set of positions, and their grid points as indices
    into a grid padded with order - 1 planes in front of each axis (no wrapping per point)"""

    def __init__(self, grid, x):
        xp, n, K, L = grid.xp, grid.order, grid.K, grid.L
        u = xp.mod(xp.asarray(x) * xp.asarray(K / L), xp.asarray(K, dtype=float))
        base = xp.minimum(xp.floor(u).astype(xp.int64), xp.asarray(K - 1))
        self.w, dw = _weights(u - base, n, xp)               # (N, 3, n)
        self.dw = dw * xp.asarray(K / L)[None, :, None]      # d weight / d x (1/Angstrom)
        Kp = grid.Kp
        self.flat = ((base[:, 0] * int(Kp[1]) + base[:, 1]) * int(Kp[2]) + base[:, 2])[:, None, None, None] + \
            grid.offsets


_GATHER_SRC = r"""
int n3 = n * n * n;
const double *wi = &w[i * 3 * n], *di = &dw[i * 3 * n];
double v = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
for (int a = 0; a < n; a++) {
  for (int b = 0; b < n; b++) {
    double wab = wi[a] * wi[n + b], dab = di[a] * wi[n + b], adb = wi[a] * di[n + b];
    for (int c = 0; c < n; c++) {
      double p = phi[flat[(long long)i * n3 + (a * n + b) * n + c]];
      double wc = wi[2 * n + c];
      v += p * wab * wc;
      sx += p * dab * wc;
      sy += p * adb * wc;
      sz += p * wab * di[2 * n + c];
    }
  }
}
val = v; gx = sx; gy = sy; gz = sz;
"""
_gather_kernel = None


def _gather_gpu(st, phi, n):
    """potential and gradient at the stencil's positions, one GPU thread per position"""
    global _gather_kernel
    import cupy
    if _gather_kernel is None:
        _gather_kernel = cupy.ElementwiseKernel(
            "raw float64 phi, raw int64 flat, raw float64 w, raw float64 dw, int32 n",
            "float64 val, float64 gx, float64 gy, float64 gz", _GATHER_SRC, "pme_gather")
    N = st.w.shape[0]
    out = [cupy.empty(N) for _ in range(4)]
    _gather_kernel(phi.ravel(), st.flat, cupy.ascontiguousarray(st.w), cupy.ascontiguousarray(st.dw), n, *out)
    return out


class PMEGrid:
    """Reciprocal-space Ewald sum with the erf kernel, Gaussian splitting parameter alpha (1/Angstrom);
    on the GPU (CuPy) when gpu is True"""

    def __init__(self, box, alpha, spacing=1.5, order=4, gpu=False):
        if order % 2:
            raise ValueError("PME: use an even B-spline order (odd orders vanish at the Nyquist frequency)")
        self.L = np.asarray(box, float)
        self.alpha, self.order, self.gpu = alpha, order, bool(gpu)
        if self.gpu:
            import cupy
            import cupyx.scipy.fft
            self.xp, self.fft, self.fft_kw = cupy, cupyx.scipy.fft, {}
        else:
            self.xp, self.fft, self.fft_kw = np, _fft, _FFT_KW
        xp = self.xp
        self.K = np.array([max(2 * order, int(np.ceil(x / spacing))) for x in self.L])
        self.V = float(np.prod(self.L))
        K = self.K
        mx = np.fft.fftfreq(K[0], 1.0 / K[0]) / self.L[0]
        my = np.fft.fftfreq(K[1], 1.0 / K[1]) / self.L[1]
        mz = np.fft.rfftfreq(K[2], 1.0 / K[2]) / self.L[2]
        m2 = mx[:, None, None] ** 2 + my[None, :, None] ** 2 + mz[None, None, :] ** 2
        b = _bmod(K[0], order)[:, None, None] * _bmod(K[1], order)[None, :, None] * \
            _bmod(K[2], order)[None, None, :K[2] // 2 + 1]
        m2[0, 0, 0] = 1.0
        g = np.exp(-np.pi ** 2 * m2 / alpha ** 2) / (np.pi * self.V * m2) * b
        g[0, 0, 0] = 0.0
        self.kernel = xp.asarray(g * float(np.prod(K)))
        self.Kp = K + order - 1
        t = np.arange(order)
        self.offsets = xp.asarray((t[:, None, None] * self.Kp[1] + t[None, :, None]) * self.Kp[2] + t[None, None, :])

    def matches(self, box, alpha, spacing, order, gpu=False):
        return np.allclose(self.L, box, rtol=0.0, atol=1e-9) and (alpha, order) == (self.alpha, self.order) and \
            np.array_equal(self.K, [max(2 * order, int(np.ceil(x / spacing))) for x in self.L]) and \
            bool(gpu) == self.gpu

    def potential(self, st: Stencil, q):
        """grid of the reciprocal-space potential of charges q at the positions of stencil st (stays on the
        device; padded like the stencils)"""
        xp, n = self.xp, self.order
        q = xp.asarray(q)
        wxy = ((q[:, None] * st.w[:, 0])[:, :, None] * st.w[:, 1, None, :]).reshape(-1, n * n, 1)
        wt = (wxy * st.w[:, 2, None, :]).ravel()
        Qp = xp.bincount(st.flat.ravel(), wt, minlength=int(np.prod(self.Kp))).reshape(tuple(self.Kp))
        m = n - 1
        for ax in range(3):                                  # fold the padding back (periodic)
            Qp = xp.moveaxis(Qp, ax, 0)
            Qp[-m:] += Qp[:m]
            Qp = xp.moveaxis(Qp[m:], 0, ax)
        phi = self.fft.irfftn(self.fft.rfftn(Qp, **self.fft_kw) * self.kernel, s=tuple(int(k) for k in self.K),
                              **self.fft_kw)
        return xp.pad(phi, [(m, 0)] * 3, mode="wrap")         # padded like the stencils

    def value(self, st: Stencil, phi):
        """potential at the stencil's positions (phi from potential()), NumPy"""
        if self.gpu:
            return _gather_gpu(st, phi, self.order)[0].get()
        v = phi.ravel()[st.flat]                             # (N, n, n, n)
        w = st.w
        return np.einsum('nabc,na,nb,nc->n', v, w[:, 0], w[:, 1], w[:, 2], optimize=True)

    def gradient(self, st: Stencil, phi):
        """gradient of the potential at the stencil's positions, (N, 3), NumPy"""
        if self.gpu:
            return self.xp.stack(_gather_gpu(st, phi, self.order)[1:], axis=1).get()
        v = phi.ravel()[st.flat]
        w, dw = st.w, st.dw
        return np.stack([np.einsum('nabc,na,nb,nc->n', v, dw[:, 0], w[:, 1], w[:, 2], optimize=True),
                         np.einsum('nabc,na,nb,nc->n', v, w[:, 0], dw[:, 1], w[:, 2], optimize=True),
                         np.einsum('nabc,na,nb,nc->n', v, w[:, 0], w[:, 1], dw[:, 2], optimize=True)], axis=1)


class FarFieldNumPy:
    """Long-range QM-MM term of openmm_mopac.py: E_far = Sum_A Q_A phi_far(A),
    phi_far(A) = phi_Ewald(A) - Sum_(j switched) S_j q_j / r_Aj, with the grid of `grid` (NumPy, or CuPy array
    operations) and the real-space sums in NumPy; Angstroms, kcal/mol"""

    def __init__(self, grid, alpha, cutoff):
        self.grid, self.alpha, self.cutoff = grid, alpha, cutoff
        self.far = None

    def potential(self, xq, xm, qm_all, xs, qs, c, L):
        """phi_far at the QM atoms xq (kcal/(mol e)).  xm, qm_all: every MM charge; xs, qs: the switched charges
        MOPAC sees (images near the QM region); c: QM centroid; L: box edges"""
        g, a = self.grid, self.alpha
        st_mm = Stencil(g, xm)
        phi_mm = g.potential(st_mm, qm_all)
        st_q = Stencil(g, xq)
        # real space: every MM charge within the cut-off of a QM atom (minimum image of each atom)
        d = xm - c
        d -= L * np.round(d / L)
        span = np.max(np.linalg.norm(xq - c, axis=1))
        near = np.nonzero(np.linalg.norm(d, axis=1) < span + self.cutoff)[0]
        rv = (c + d[near])[None, :, :] - xq[:, None, :]            # QM atom -> MM charge
        r = np.sqrt(np.einsum('ijk,ijk->ij', rv, rv))
        pa, pj = np.nonzero(r < self.cutoff)                        # pairs inside the cut-off
        rv, r = rv[pa, pj], r[pa, pj]
        ec = erfc(a * r)
        qn = qm_all[near[pj]]
        phi_real = np.bincount(pa, qn * ec / r, minlength=len(xq))
        # bare Coulomb potential of the charges MOPAC sees (switched)
        rb = xs[None, :, :] - xq[:, None, :]
        rbn = np.linalg.norm(rb, axis=2)
        phi_bare = (qs[None, :] / rbn).sum(axis=1)
        background = -np.pi * qm_all.sum() / (g.V * a * a)
        self.far = dict(st_mm=st_mm, st_q=st_q, phi_mm=phi_mm, pa=pa, pj=near[pj], rv=rv, r=r, ec=ec, qn=qn,
                        rb=rb, rbn=rbn, qm_all=qm_all, qs=qs)
        return COULOMB * (phi_real + g.value(st_q, phi_mm) + background - phi_bare)

    def gradient(self, Q, gq, gm, pot):
        """Q: Mulliken charges of the QM atoms.  Adds the gradient of E_far (kcal/(mol A)) to gq (QM atoms) and
        gm (switched charges, at fixed weights), dE_far/dq_eff to pot; returns the gradient on every MM charge
        from the Ewald part"""
        g, a, far = self.grid, self.alpha, self.far
        qm_all, qs = far["qm_all"], far["qs"]
        # reciprocal space
        gq += COULOMB * Q[:, None] * g.gradient(far["st_q"], far["phi_mm"])
        psi = g.potential(far["st_q"], Q)
        g_all = COULOMB * qm_all[:, None] * g.gradient(far["st_mm"], psi)
        # real space: E = C Q_A q_j erfc(a r)/r
        r, rv, pa = far["r"], far["rv"], far["pa"]
        dfdr = -2.0 * a / np.sqrt(np.pi) * np.exp(-(a * r) ** 2) / r - far["ec"] / r ** 2
        gr = (COULOMB * Q[pa] * far["qn"] * dfdr / r)[:, None] * rv      # d E / d x(MM charge), per pair
        for k in range(3):
            g_all[:, k] += np.bincount(far["pj"], gr[:, k], minlength=len(g_all))
            gq[:, k] -= np.bincount(pa, gr[:, k], minlength=len(gq))
        # bare Coulomb with the switched charges: E = -C Q_A q_eff_j / r
        rbn, rb = far["rbn"], far["rb"]
        cb = COULOMB * Q[:, None] * qs[None, :] / rbn ** 3
        gb = cb[:, :, None] * rb
        gm += gb.sum(axis=0)
        gq -= gb.sum(axis=1)
        pot -= COULOMB * (Q[:, None] / rbn).sum(axis=0)
        return g_all


_FF_SRC = r"""
// order-4 B-spline weights m[t] = M4(w + 3 - t) and d m[t] / d w; weight t sits on grid point base + t - 3 (mod K)
__device__ __forceinline__ void ff_bsp4(double w, double *m, double *d) {
  double v = 1.0 - w, w2 = w * w, w3 = w2 * w;
  m[0] = v * v * v / 6.0;
  m[1] = (4.0 - 6.0 * w2 + 3.0 * w3) / 6.0;
  m[2] = (1.0 + 3.0 * (w + w2 - w3)) / 6.0;
  m[3] = w3 / 6.0;
  d[0] = -0.5 * v * v;
  d[1] = -2.0 * w + 1.5 * w2;
  d[2] = 0.5 + w - 1.5 * w2;
  d[3] = 0.5 * w2;
}

// grid cell of coordinate x (Angstrom) on an axis with k points (kl = k / L), and the offset w in it
__device__ __forceinline__ int ff_cell(double x, double kl, int k, double *w) {
  double u = x * kl;
  u -= floor(u / k) * k;
  int b = (int)floor(u);
  if (b > k - 1) b = k - 1;
  if (b < 0) b = 0;
  *w = u - b;
  return b;
}

// add q_i times its B-spline weights to the grid (kx, ky, kz), one thread per charge
extern "C" __global__ void ff_spread(const double *x, const double *q, int n, double klx, double kly,
                                     double klz, int kx, int ky, int kz, double *grid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double qi = q[i];
  if (qi == 0.0) return;
  double wx, wy, wz, mx[4], my[4], mz[4], d[4];
  int bx = ff_cell(x[3 * i], klx, kx, &wx);
  int by = ff_cell(x[3 * i + 1], kly, ky, &wy);
  int bz = ff_cell(x[3 * i + 2], klz, kz, &wz);
  ff_bsp4(wx, mx, d);
  ff_bsp4(wy, my, d);
  ff_bsp4(wz, mz, d);
  for (int a = 0; a < 4; a++) {
    int ia = (bx + a - 3 + kx) % kx;
    for (int b = 0; b < 4; b++) {
      int ib = (by + b - 3 + ky) % ky;
      double qab = qi * mx[a] * my[b];
      long long row = ((long long)ia * ky + ib) * kz;
      for (int c = 0; c < 4; c++) {
        int ic = (bz + c - 3 + kz) % kz;
        atomicAdd(&grid[row + ic], qab * mz[c]);
      }
    }
  }
}

// potential val[i] and scale[i] times its gradient grad[i] (d/dx, 1/Angstrom) at the positions x
extern "C" __global__ void ff_gather(const double *x, int n, double klx, double kly, double klz, int kx, int ky,
                                     int kz, const double *phi, const double *scale, double *val, double *grad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double wx, wy, wz, mx[4], my[4], mz[4], dx[4], dy[4], dz[4];
  int bx = ff_cell(x[3 * i], klx, kx, &wx);
  int by = ff_cell(x[3 * i + 1], kly, ky, &wy);
  int bz = ff_cell(x[3 * i + 2], klz, kz, &wz);
  ff_bsp4(wx, mx, dx);
  ff_bsp4(wy, my, dy);
  ff_bsp4(wz, mz, dz);
  double v = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
  for (int a = 0; a < 4; a++) {
    int ia = (bx + a - 3 + kx) % kx;
    for (int b = 0; b < 4; b++) {
      int ib = (by + b - 3 + ky) % ky;
      double wab = mx[a] * my[b], dab = dx[a] * my[b], adb = mx[a] * dy[b];
      long long row = ((long long)ia * ky + ib) * kz;
      for (int c = 0; c < 4; c++) {
        double p = phi[row + (bz + c - 3 + kz) % kz];
        v += p * wab * mz[c];
        gx += p * dab * mz[c];
        gy += p * adb * mz[c];
        gz += p * wab * dz[c];
      }
    }
  }
  double s = scale[i];
  val[i] = v;
  grad[3 * i] = s * gx * klx;
  grad[3 * i + 1] = s * gy * kly;
  grad[3 * i + 2] = s * gz * klz;
}

// phi[A] = Sum_j q_j erfc(alpha r)/r over r < rc, minimum image (ewald = 1), or Sum_j q_j / r (ewald = 0);
// one block (a power of two threads, as many doubles of shared memory) per QM atom
extern "C" __global__ void ff_pair_phi(const double *xq, int nq, const double *xm, const double *qm, int nm,
                                       double lx, double ly, double lz, double alpha, double rc2, int ewald,
                                       double *phi) {
  extern __shared__ double sh[];
  int A = blockIdx.x;
  double xa = xq[3 * A], ya = xq[3 * A + 1], za = xq[3 * A + 2], s = 0.0;
  for (int j = threadIdx.x; j < nm; j += blockDim.x) {
    double q = qm[j];
    if (q == 0.0) continue;
    double dx = xm[3 * j] - xa, dy = xm[3 * j + 1] - ya, dz = xm[3 * j + 2] - za;
    if (ewald) {
      dx -= lx * rint(dx / lx);
      dy -= ly * rint(dy / ly);
      dz -= lz * rint(dz / lz);
    }
    double r2 = dx * dx + dy * dy + dz * dz;
    if (ewald && r2 >= rc2) continue;
    double r = sqrt(r2);
    s += ewald ? q * erfc(alpha * r) / r : q / r;
  }
  sh[threadIdx.x] = s;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) phi[A] = sh[0];
}

// gradient of E = coul Sum_A Sum_j Q_A q_j f(r_Aj), f = erfc(alpha r)/r (ewald = 1, r < rc, minimum image) or
// 1/r: added to gm[j] (one thread per charge j) and gq[A] (atomic); pot[j] += coul Sum_A Q_A / r if want_pot
extern "C" __global__ void ff_pair_grad(const double *xq, const double *Q, int nq, const double *xm,
                                        const double *qm, int nm, double lx, double ly, double lz, double alpha,
                                        double rc2, int ewald, double coul, double *gm, double *gq, double *pot,
                                        int want_pot) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nm) return;
  double q = qm[j];
  if (ewald && q == 0.0) return;
  double xj = xm[3 * j], yj = xm[3 * j + 1], zj = xm[3 * j + 2];
  double gx = 0.0, gy = 0.0, gz = 0.0, pj = 0.0;
  const double tsp = 1.1283791670955126;          // 2 / sqrt(pi)
  for (int A = 0; A < nq; A++) {
    double dx = xj - xq[3 * A], dy = yj - xq[3 * A + 1], dz = zj - xq[3 * A + 2];
    if (ewald) {
      dx -= lx * rint(dx / lx);
      dy -= ly * rint(dy / ly);
      dz -= lz * rint(dz / lz);
    }
    double r2 = dx * dx + dy * dy + dz * dz;
    if (ewald && r2 >= rc2) continue;
    double r = sqrt(r2), dfdr;
    if (ewald) dfdr = -tsp * alpha * exp(-alpha * alpha * r2) / r - erfc(alpha * r) / r2;
    else dfdr = -1.0 / r2;
    double c = coul * Q[A] * q * dfdr / r;
    gx += c * dx;
    gy += c * dy;
    gz += c * dz;
    atomicAdd(&gq[3 * A], -c * dx);
    atomicAdd(&gq[3 * A + 1], -c * dy);
    atomicAdd(&gq[3 * A + 2], -c * dz);
    pj += Q[A] / r;
  }
  gm[3 * j] += gx;
  gm[3 * j + 1] += gy;
  gm[3 * j + 2] += gz;
  if (want_pot) pot[j] += coul * pj;
}
"""
_ff_module_cache = None


def _ff_module():
    global _ff_module_cache
    if _ff_module_cache is None:
        import cupy
        _ff_module_cache = cupy.RawModule(code=_FF_SRC)
    return _ff_module_cache


class FarFieldGPU:
    """FarFieldNumPy on the GPU with fused kernels (CuPy, double precision); same arguments and results"""

    BLOCK = 128
    PAIR_BLOCK = 256

    def __init__(self, grid, alpha, cutoff):
        import cupy
        if not grid.gpu or grid.order != 4:
            raise ValueError("FarFieldGPU: needs a GPU grid of B-spline order 4")
        self.cp, self.grid, self.alpha, self.cutoff = cupy, grid, alpha, cutoff
        mod = _ff_module()
        self.k_spread, self.k_gather, self.k_phi, self.k_grad = (
            mod.get_function(n) for n in ("ff_spread", "ff_gather", "ff_pair_phi", "ff_pair_grad"))
        self.K = tuple(int(k) for k in grid.K)
        self.kl = tuple(np.float64(k / l) for k, l in zip(grid.K, grid.L))
        self.qm_host = None

    def _blocks(self, n):
        return ((n + self.BLOCK - 1) // self.BLOCK,), (self.BLOCK,)

    def _dev(self, a):
        return self.cp.asarray(np.ascontiguousarray(a, dtype=np.float64))

    def _grid_potential(self, x, q):
        """reciprocal-space potential grid of charges q at positions x (both on the device)"""
        cp, g = self.cp, self.grid
        rho = cp.zeros(self.K)
        n = x.shape[0]
        if n:
            self.k_spread(*self._blocks(n), (x, q, np.int32(n), *self.kl, *(np.int32(k) for k in self.K), rho))
        return cp.ascontiguousarray(g.fft.irfftn(g.fft.rfftn(rho) * g.kernel, s=self.K))

    def _gather(self, x, phi, scale):
        cp = self.cp
        n = x.shape[0]
        val, grad = cp.zeros(n), cp.zeros((n, 3))
        if n:
            self.k_gather(*self._blocks(n), (x, np.int32(n), *self.kl, *(np.int32(k) for k in self.K), phi, scale,
                                             val, grad))
        return val, grad

    def _pair_phi(self, xq, xm, qm, ewald, L):
        cp = self.cp
        nq, nm = xq.shape[0], xm.shape[0]
        phi = cp.zeros(nq)
        if nq and nm:
            self.k_phi((nq,), (self.PAIR_BLOCK,), (xq, np.int32(nq), xm, qm, np.int32(nm), *(np.float64(x) for x in L),
                                                   np.float64(self.alpha), np.float64(self.cutoff ** 2),
                                                   np.int32(ewald), phi), shared_mem=8 * self.PAIR_BLOCK)
        return phi

    def _pair_grad(self, xq, Q, xm, qm, ewald, coul, gm, gq, pot, want_pot, L):
        nq, nm = xq.shape[0], xm.shape[0]
        if nq and nm:
            self.k_grad(*self._blocks(nm), (xq, Q, np.int32(nq), xm, qm, np.int32(nm), *(np.float64(x) for x in L),
                                            np.float64(self.alpha), np.float64(self.cutoff ** 2), np.int32(ewald),
                                            np.float64(coul), gm, gq, pot, np.int32(want_pot)))

    def potential(self, xq, xm, qm_all, xs, qs, c, L):
        cp, g, a = self.cp, self.grid, self.alpha
        if self.qm_host is None or len(self.qm_host) != len(qm_all) or not np.array_equal(self.qm_host, qm_all):
            self.qm_host = np.array(qm_all, dtype=np.float64)
            self.qm = self._dev(self.qm_host)
            self.qm_sum = float(self.qm_host.sum())
        self.L = np.asarray(L, dtype=np.float64)
        self.xq, self.xm = self._dev(xq), self._dev(xm)
        self.xs, self.qs = self._dev(np.reshape(xs, (-1, 3))), self._dev(qs)
        phi_mm = self._grid_potential(self.xm, self.qm)
        rec, self.grad_rec = self._gather(self.xq, phi_mm, cp.ones(self.xq.shape[0]))
        real = self._pair_phi(self.xq, self.xm, self.qm, 1, self.L)
        bare = self._pair_phi(self.xq, self.xs, self.qs, 0, self.L)
        background = -np.pi * self.qm_sum / (g.V * a * a)
        return cp.asnumpy(COULOMB * (real + rec + background - bare))

    def gradient(self, Q, gq, gm, pot):
        cp = self.cp
        Qd = self._dev(Q)
        gqd = COULOMB * Qd[:, None] * self.grad_rec                       # reciprocal, QM atoms
        psi = self._grid_potential(self.xq, Qd)
        _, g_all = self._gather(self.xm, psi, COULOMB * self.qm)           # reciprocal, MM charges
        self._pair_grad(self.xq, Qd, self.xm, self.qm, 1, COULOMB, g_all, gqd, g_all, 0, self.L)
        ns = self.xs.shape[0]
        if ns:
            gs, ps = cp.zeros((ns, 3)), cp.zeros(ns)
            self._pair_grad(self.xq, Qd, self.xs, self.qs, 0, -COULOMB, gs, gqd, ps, 1, self.L)
            gm += cp.asnumpy(gs)
            pot += cp.asnumpy(ps)
        gq += cp.asnumpy(gqd)
        return cp.asnumpy(g_all)

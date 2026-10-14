"""Smooth particle-mesh Ewald (Essmann et al. 1995) for the long-range QM-MM electrostatics (openmm_mopac.py).

Only what QM/MM needs: the reciprocal-space potential of the MM charges at the QM atoms, and its
derivatives with respect to the positions of both (B-spline interpolation with analytic derivatives, so
that the forces are the exact derivatives of the energy as computed on the grid).  Orthorhombic boxes;
Angstroms; potentials in e/Angstrom (multiply by the Coulomb constant).
"""

from __future__ import annotations

import numpy as np

try:                                   # multithreaded FFTs when SciPy is there
    import scipy.fft as _fft
    _FFT_KW = {"workers": -1}
except ImportError:                    # pragma: no cover
    _fft = np.fft
    _FFT_KW = {}


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


def _weights(w, n):
    """weights and their derivatives (d/dw) for fractional offsets w: (..., n) each"""
    if n == 4:
        v = 1.0 - w
        w2, w3 = w * w, w * w * w
        m = np.stack([v * v * v, 4.0 - 6.0 * w2 + 3.0 * w3, 1.0 + 3.0 * (w + w2 - w3), w3], axis=-1) / 6.0
        d = np.stack([-0.5 * v * v, -2.0 * w + 1.5 * w2, 0.5 + w - 1.5 * w2, 0.5 * w2], axis=-1)
        return m, d
    m = bspline(w, n)
    m1 = bspline(w, n - 1)
    d = np.zeros_like(m)
    d[..., 1:] += m1
    d[..., :-1] -= m1
    return m, d


class Stencil:
    """B-spline weights (and their derivatives) of a set of positions, and their grid points as indices
    into a grid padded with order - 1 planes in front of each axis (no wrapping per point)"""

    def __init__(self, grid, x):
        n, K, L = grid.order, grid.K, grid.L
        u = np.mod(x / L * K, K)
        base = np.minimum(np.floor(u).astype(np.int64), K - 1)
        self.w, dw = _weights(u - base, n)                   # (N, 3, n)
        self.dw = dw * (K / L)[None, :, None]                # d weight / d x (1/Angstrom)
        Kp = grid.Kp
        self.flat = ((base[:, 0] * Kp[1] + base[:, 1]) * Kp[2] + base[:, 2])[:, None, None, None] + grid.offsets


class PMEGrid:
    """Reciprocal-space Ewald sum with the erf kernel, Gaussian splitting parameter alpha (1/Angstrom)"""

    def __init__(self, box, alpha, spacing=1.5, order=4):
        if order % 2:
            raise ValueError("PME: use an even B-spline order (odd orders vanish at the Nyquist frequency)")
        self.L = np.asarray(box, float)
        self.alpha, self.order = alpha, order
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
        self.kernel = g * float(np.prod(K))
        self.Kp = K + order - 1
        t = np.arange(order)
        self.offsets = (t[:, None, None] * self.Kp[1] + t[None, :, None]) * self.Kp[2] + t[None, None, :]

    def matches(self, box, alpha, spacing, order):
        return np.allclose(self.L, box, rtol=0.0, atol=1e-9) and (alpha, order) == (self.alpha, self.order) and \
            np.array_equal(self.K, [max(2 * order, int(np.ceil(x / spacing))) for x in self.L])

    def potential(self, st: Stencil, q):
        """grid of the reciprocal-space potential of charges q at the positions of stencil st"""
        n = self.order
        wxy = ((q[:, None] * st.w[:, 0])[:, :, None] * st.w[:, 1, None, :]).reshape(-1, n * n, 1)
        wt = (wxy * st.w[:, 2, None, :]).ravel()
        Qp = np.bincount(st.flat.ravel(), wt, minlength=int(np.prod(self.Kp))).reshape(self.Kp)
        m = n - 1
        for ax in range(3):                                  # fold the padding back (periodic)
            Qp = np.moveaxis(Qp, ax, 0)
            Qp[-m:] += Qp[:m]
            Qp = np.moveaxis(Qp[m:], 0, ax)
        phi = _fft.irfftn(_fft.rfftn(Qp, **_FFT_KW) * self.kernel, s=tuple(self.K), **_FFT_KW)
        return np.pad(phi, [(m, 0)] * 3, mode="wrap")         # padded like the stencils

    @staticmethod
    def value(st: Stencil, phi):
        """potential at the stencil's positions (phi from potential())"""
        v = phi.ravel()[st.flat]
        return np.einsum('nabc,na,nb,nc->n', v, st.w[:, 0], st.w[:, 1], st.w[:, 2], optimize=True)

    @staticmethod
    def gradient(st: Stencil, phi):
        """gradient of the potential at the stencil's positions, (N, 3)"""
        v = phi.ravel()[st.flat]
        w, dw = st.w, st.dw
        return np.stack([np.einsum('nabc,na,nb,nc->n', v, dw[:, 0], w[:, 1], w[:, 2], optimize=True),
                         np.einsum('nabc,na,nb,nc->n', v, w[:, 0], dw[:, 1], w[:, 2], optimize=True),
                         np.einsum('nabc,na,nb,nc->n', v, w[:, 0], w[:, 1], dw[:, 2], optimize=True)], axis=1)

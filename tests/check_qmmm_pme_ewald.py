#!/usr/bin/env python3
"""Check the long-range QM-MM electrostatics of scripts/qmmm/openmm_mopac.py (smooth PME, qmmm_pme.py) against
an Ewald sum evaluated directly, without a grid.

    python tests/check_qmmm_pme_ewald.py [--csv out.csv] [--gpu]

1. Reference: Ewald potential of point charges in an orthorhombic box, real space summed over all images within
   40 A and reciprocal space summed over every k vector with exp(-k^2/4a^2) > 1e-16, plus the neutralizing
   background -pi Q/(V a^2) ("tin-foil" boundary).  The reference is checked by itself: it must not depend on the
   Ewald splitting a.
2. Madelung: the potential at a Na site of rock salt from all other ions is -1.747565/r0 (e/A).  It checks the
   reference and the PME of openmm_mopac with no adjustable number.
3. phi_far: the potential that openmm_mopac adds to the Fock matrix (Ewald potential of all MM charges minus the
   bare Coulomb potential of the switched charges that MOPAC already sees), for random neutral and charged MM
   sets, at the default PME settings and on a finer grid (the error must fall with the grid).
4. Forces: the analytic gradient of E_far = Sum_A Q_A phi_far(A) (QM atoms, all MM charges, switched charges) and
   dE/dq of the switched charges, against central finite differences of the same energy.

Needs NumPy and SciPy, and OpenMM (openmm_mopac imports it); no MOPAC library.  --gpu also runs the PME grid with
CuPy.  Exit status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import erfc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "qmmm"))
from openmm_mopac import COULOMB, MopacQMMMForce, QMMMRegion  # noqa: E402
from qmmm_pme import gpu_available  # noqa: E402

MADELUNG_NACL = 1.747564594633


def ewald_reference(points, x, q, L, alpha, rmax=40.0, tol=1e-16):
    """Ewald potential (e/A) at points of charges q at x in an orthorhombic box L.  A charge at the point itself
    contributes its periodic images only (its own Coulomb term and the Ewald self term are left out)."""
    L = np.asarray(L, float)
    V = float(np.prod(L))
    nimg = [int(math.ceil(rmax / l)) for l in L]
    shifts = np.array([(i, j, k) for i in range(-nimg[0], nimg[0] + 1) for j in range(-nimg[1], nimg[1] + 1)
                       for k in range(-nimg[2], nimg[2] + 1)], float) * L
    phi_r = np.zeros(len(points))
    for a, p in enumerate(points):
        d = (x[None, :, :] + shifts[:, None, :]) - p[None, None, :]
        r = np.linalg.norm(d, axis=2)
        m = (r < rmax) & (r > 1e-9)
        phi_r[a] = np.sum((q[None, :] * erfc(alpha * r) / np.where(m, r, 1.0))[m])
        self_q = q[np.linalg.norm(x - p, axis=1) < 1e-9].sum()          # charge sitting on the point
        phi_r[a] -= 2.0 * alpha / math.sqrt(math.pi) * self_q
    kmax = 2.0 * alpha * math.sqrt(-math.log(tol))
    ms = [np.arange(-int(kmax * l / (2 * math.pi)) - 1, int(kmax * l / (2 * math.pi)) + 2) for l in L]
    kv = np.stack(np.meshgrid(*[2 * math.pi * m / l for m, l in zip(ms, L)], indexing="ij"), -1).reshape(-1, 3)
    k2 = np.einsum("ij,ij->i", kv, kv)
    keep = (k2 > 0) & (k2 < kmax * kmax)
    kv, k2 = kv[keep], k2[keep]
    w = 4 * math.pi / V * np.exp(-k2 / (4 * alpha * alpha)) / k2
    S = np.exp(-1j * kv @ x.T) @ q                                   # structure factor, sum_j q_j exp(-i k.x_j)
    phi_k = np.real(np.exp(1j * points @ kv.T) @ (w * S))
    return phi_r + phi_k - math.pi * q.sum() / (V * alpha * alpha)


def far_object(q_all, Q, alpha=0.25, spacing=1.2, order=4, gpu=False):
    """A MopacQMMMForce with only what _pme_far_potential / _pme_far_gradient read (no MOPAC session)."""
    f = MopacQMMMForce.__new__(MopacQMMMForce)
    f.region = QMMMRegion(qm_atoms=[], atomic_numbers=[], mm_charges=np.asarray(q_all, float), periodic=True,
                          pme=True, pme_alpha=alpha, pme_spacing=spacing, pme_order=order, pme_gpu=gpu)
    f.mm_idx_all = np.arange(len(q_all))
    f.grid = None
    f.session = SimpleNamespace(qm_charges=np.asarray(Q, float))
    return f


def far_energy(f, xq, xm, xs, qs, box, Q):
    c = xq.mean(axis=0)
    return float(np.dot(Q, f._pme_far_potential(xq, xm, xs, qs, box, c)["phi"]))


def rock_salt(n, a0):
    """n x n x n conventional cells of NaCl (lattice constant a0); charges +1 (Na) and -1 (Cl)."""
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    cells = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], float)
    na = (cells[:, None, :] + base[None, :, :]).reshape(-1, 3) * a0
    cl = na + np.array([.5, 0, 0]) * a0
    return np.vstack([na, cl]), np.concatenate([np.ones(len(na)), -np.ones(len(cl))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="", help="write the results to this CSV file")
    ap.add_argument("--gpu", action="store_true", help="also run the PME grid on the GPU (CuPy)")
    a = ap.parse_args()
    rows, ok = [], True

    def report(check, value, tol, unit, note=""):
        nonlocal ok
        good = value <= tol
        ok &= good
        rows.append({"check": check, "value": value, "tolerance": tol, "unit": unit, "pass": good, "note": note})
        print(f"{'PASS' if good else 'FAIL'} {check}: {value:.3e} {unit} (<= {tol:.0e}) {note}")

    gpus = [False] + ([True] if a.gpu and gpu_available() else [])
    if a.gpu and len(gpus) == 1:
        print("note: --gpu given but CuPy cannot run the PME grid here; CPU only")

    # 1. the reference does not depend on the Ewald splitting
    rng = np.random.default_rng(7)
    L = np.array([40.0, 41.0, 42.0])                                 # > 2 x (QM radius + 13 A), as compute() requires
    x = rng.random((300, 3)) * L
    q = rng.standard_normal(300)
    q -= q.mean()
    q[:20] += 0.05                                                   # net charge 1.0 e: tests the background
    pts = L / 2 + rng.standard_normal((6, 3)) * 2.0
    r1, r2 = ewald_reference(pts, x, q, L, 0.25), ewald_reference(pts, x, q, L, 0.35)
    report("reference: alpha 0.25 vs 0.35, max |dphi|", float(np.max(np.abs(r1 - r2))), 1e-8, "e/A")

    # 2. Madelung constant of rock salt
    a0 = 5.64
    xs_nacl, qs_nacl = rock_salt(5, a0)                              # box 28.2 A > 2 x (0 + 13 A)
    box_len = np.full(3, 5 * a0)
    site = xs_nacl[0].copy()
    others = np.arange(1, len(qs_nacl))
    exact = -MADELUNG_NACL / (a0 / 2)
    # reference: the whole neutral lattice; the ion on the site contributes its periodic images only
    ref = ewald_reference(site[None, :], xs_nacl, qs_nacl, box_len, 0.3)[0]
    report("Madelung: Ewald reference vs -1.747565/r0", abs(ref - exact), 1e-7, "e/A")
    # openmm_mopac sees the other 999 ions as MM charges (net charge -1); the images of the removed ion are the
    # difference between the two references, so PME(others) + that term must give the Madelung potential
    img = ref - ewald_reference(site[None, :], xs_nacl[others], qs_nacl[others], box_len, 0.3)[0]
    for gpu in gpus:
        f = far_object(qs_nacl[others], [1.0], gpu=gpu)
        phi = f._pme_far_potential(site[None, :], xs_nacl[others], np.zeros((0, 3)), np.zeros(0),
                                   np.diag(box_len), site)["phi"][0] / COULOMB
        report(f"Madelung: openmm_mopac PME ({'GPU' if gpu else 'CPU'}) + images of the site ion vs -1.747565/r0",
               abs(phi + img - exact) / abs(exact), 2e-4, "relative", "default grid (1.2 A, order 4)")

    # 3. phi_far for random MM sets: neutral and charged, default and fine grid
    box = np.diag(L)
    xq = L / 2 + rng.standard_normal((5, 3)) * 1.5
    near = np.nonzero(np.min(np.linalg.norm(x[:, None, :] - xq[None, :, :], axis=2), axis=1) < 9.0)[0]
    xs_sw = x[near]
    for label, qq in (("charged", q), ("neutral", q - q.mean())):
        w = rng.random(len(near))                                    # switching weights of the near charges
        qs_sw = qq[near] * w
        ref = COULOMB * (ewald_reference(xq, x, qq, L, 0.25) -
                         (qs_sw[None, :] / np.linalg.norm(xs_sw[None, :, :] - xq[:, None, :], axis=2)).sum(1))
        scale = float(np.max(np.abs(ref)))
        for spacing, order, tol, tag in ((1.2, 4, 3e-4, "default grid"), (0.6, 6, 1e-5, "fine grid")):
            for gpu in gpus:
                f = far_object(qq, np.zeros(len(xq)), spacing=spacing, order=order, gpu=gpu)
                phi = f._pme_far_potential(xq, x, xs_sw, qs_sw, box, xq.mean(axis=0))["phi"]
                report(f"phi_far {label}, {tag} ({'GPU' if gpu else 'CPU'}), max |dphi| / max |phi|",
                       float(np.max(np.abs(phi - ref))) / scale, tol, "relative",
                       f"spacing {spacing} A, order {order}; max |phi| {scale:.1f} kcal/(mol e)")

    # 4. forces: analytic gradient of E_far against central differences
    Q = rng.standard_normal(len(xq)) * 0.3
    qs_sw = q[near] * rng.random(len(near))
    for gpu in gpus:
        f = far_object(q, Q, gpu=gpu)
        c = xq.mean(axis=0)
        far = f._pme_far_potential(xq, x, xs_sw, qs_sw, box, c)
        gq = np.zeros_like(xq)
        gm = np.zeros_like(xs_sw)
        pot = np.zeros(len(qs_sw))
        g_all = f._pme_far_gradient(far, xq, xs_sw, qs_sw, gq, gm, pot)
        h = 1e-4
        tag = "GPU" if gpu else "CPU"

        def fd(arr_name, idx, k):
            args = {"xq": xq.copy(), "xm": x.copy(), "xs": xs_sw.copy()}
            args[arr_name][idx, k] += h
            ep = far_energy(f, args["xq"], args["xm"], args["xs"], qs_sw, box, Q)
            args[arr_name][idx, k] -= 2 * h
            em = far_energy(f, args["xq"], args["xm"], args["xs"], qs_sw, box, Q)
            return (ep - em) / (2 * h)

        # the centre c moves with xq, but it only picks images: the energy does not depend on it
        num_q = np.array([[fd("xq", i, k) for k in range(3)] for i in range(len(xq))])
        pick_m = rng.choice(len(x), 12, replace=False)
        num_m = np.array([[fd("xm", i, k) for k in range(3)] for i in pick_m])
        pick_s = rng.choice(len(xs_sw), min(8, len(xs_sw)), replace=False)
        num_s = np.array([[fd("xs", i, k) for k in range(3)] for i in pick_s])
        num_p = []
        for i in pick_s:
            qp, qm_ = qs_sw.copy(), qs_sw.copy()
            qp[i] += h
            qm_[i] -= h
            num_p.append((far_energy(f, xq, x, xs_sw, qp, box, Q) - far_energy(f, xq, x, xs_sw, qm_, box, Q)) / (2 * h))
        gscale = max(np.max(np.abs(num_q)), 1.0)
        report(f"forces ({tag}): QM atoms, max |analytic - FD|", float(np.max(np.abs(gq - num_q))), 1e-6 * gscale,
               "kcal/(mol A)", f"max |g| {gscale:.2f}")
        report(f"forces ({tag}): MM charges (Ewald part), max |analytic - FD|",
               float(np.max(np.abs(g_all[pick_m] - num_m))), 1e-6 * gscale, "kcal/(mol A)", "12 random charges")
        report(f"forces ({tag}): switched charges (bare part), max |analytic - FD|",
               float(np.max(np.abs(gm[pick_s] - num_s))), 1e-6 * gscale, "kcal/(mol A)", f"{len(pick_s)} charges")
        report(f"dE/dq of switched charges ({tag}), max |analytic - FD|",
               float(np.max(np.abs(pot[pick_s] - np.array(num_p)))), 1e-6 * max(np.max(np.abs(num_p)), 1.0),
               "kcal/(mol e)", "")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""QM/MM molecular dynamics with OpenMM (MM, integrators, ensembles) and MOPAC/MOZYME (QM, on the GPU).

The QM region is computed by MOPAC (PM7 by default) with electrostatic embedding in the MM point charges
(src/MOZYME/qmmm.F90); everything else (force field, PME, constraints, thermostats, barostats, reporters)
is OpenMM.  The coupling is an OpenMM PythonForce (OpenMM >= 8.4), so every integrator and ensemble of
OpenMM works unchanged.

    from openmm_mopac import add_qmmm
    system = forcefield.createSystem(topology, nonbondedMethod=PME, constraints=HBonds, ...)
    qmmm = add_qmmm(system, topology, positions, qm_atoms, qm_charge=0, libmopac="/path/libmopac.so")
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.5*femtoseconds)
    simulation = Simulation(topology, system, integrator)
    ...

What add_qmmm does to the MM system:
  * bonds, angles, torsions (and CMAP) whose atoms are all QM are switched off, constraints between QM
    atoms are removed (MOPAC describes them);
  * NonbondedForce: the charges of the QM atoms are set to zero (the QM-MM electrostatics come from MOPAC,
    with the QM electron density polarized by the MM charges) and QM-QM pairs are excluded; the
    Lennard-Jones terms between QM and MM atoms are kept;
  * covalent QM/MM boundary: a hydrogen link atom on every QM-MM bond, at Q + g (M - Q) with
    g = (Q-H bond length)/(Q-M bond length); its force is shared between Q and M; the charge of the MM
    boundary atom M is not seen by the QM region (it would sit 1.1 A from the link atom);
  * the MM charges are switched on and off by groups (residues: a water, an ion, an MM residue), all
    charges of a group with the same factor S(d), d = smooth minimum (lambda = 0.5 A) of the distances
    from the group centre to the QM atoms, S = 1 below ron and 0 beyond roff (minimum image).  Switching
    each QM atom - MM charge pair instead let waters in the switching shell turn their H inside and their
    O outside and feel a net charge: the energy of a charged QM peptide fell by 1750 kcal/mol in 10 fs.
    The force from dS/dx uses dE/dq of each charge (the QM potential there), returned by MOPAC.
  * long-range QM-MM electrostatics (pme=True, the default with PME in OpenMM): the Mulliken charges Q_A of
    the QM atoms interact with all MM charges and their periodic images, by Ewald summation (smooth PME,
    qmmm_pme.py), minus the bare Coulomb interaction with the switched charges that MOPAC already sees:
        E_far = Sum_A Q_A phi_far(A),  phi_far(A) = phi_Ewald(A) - Sum_(j near) S_j q_j / r_Aj
    phi_far goes into the Fock matrix of MOPAC (on the diagonal of atom A), so that the energy stays
    variational and the forces are exact; near charges act through the QM integrals with weight S and
    through Q_A with weight 1 - S, and the result hardly depends on ron and roff.  The interactions of the
    QM region with its own periodic images are left out.

Units at the interface: OpenMM nm, kJ/mol; MOPAC Angstrom, kcal/mol.  The QM energy is the PM7 heat of
formation of the QM region (+ link atoms) in the field of the MM charges, so the total potential energy is
offset by a constant; forces and dynamics are unaffected.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mopac_qmmm_api import QMSession  # noqa: E402
from qmmm_pme import PMEGrid, Stencil  # noqa: E402
from scipy.special import erfc  # noqa: E402

KCAL_TO_KJ = 4.184
# Coulomb constant, kcal A/(mol e^2) (OpenMM's ONE_4PI_EPS0)
COULOMB = 138.935456 * 10.0 / KCAL_TO_KJ
# force: kcal/(mol A) -> kJ/(mol nm)
GRAD_KCAL_A_TO_KJ_NM = KCAL_TO_KJ * 10.0
# Q-H bond lengths (A) for link atoms
XH_LENGTH = {6: 1.09, 7: 1.01, 8: 0.96, 16: 1.34, 15: 1.42}


@dataclass
class QMMMRegion:
    qm_atoms: list[int]
    atomic_numbers: list[int]
    links: list[tuple[int, int, float]] = field(default_factory=list)   # (qm atom, mm atom, g)
    mm_charges: np.ndarray | None = None       # charge of every particle as seen by the QM region
    groups: np.ndarray | None = None           # switching group (residue index) of every particle
    ron: float = 10.0                          # A
    roff: float = 12.0                         # A
    periodic: bool = False
    pme: bool = False                          # long-range QM-MM electrostatics by PME (periodic only)
    pme_alpha: float = 0.25                    # Ewald splitting, 1/A
    pme_cutoff: float = 13.0                   # real-space cut-off, A (erfc(alpha rc) ~ 4e-6)
    pme_spacing: float = 1.2                   # grid spacing, A (potential error ~1e-4 relative)
    pme_order: int = 4                         # B-spline order (even)


def _element_number(atom) -> int:
    return atom.element.atomic_number if atom.element is not None else 0


def _qm_bonds(topology, qm_set):
    links = []
    for b in topology.bonds():
        a1, a2 = b[0].index, b[1].index
        if (a1 in qm_set) != (a2 in qm_set):
            q, m = (a1, a2) if a1 in qm_set else (a2, a1)
            links.append((q, m))
    return links


def prepare_mm_system(system: mm.System, topology, qm_atoms) -> np.ndarray:
    """Remove the MM description inside the QM region; returns the MM charges (e) of all particles."""
    qm = set(qm_atoms)
    charges = np.zeros(system.getNumParticles())
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for i in range(f.getNumBonds()):
                a, b, r0, k = f.getBondParameters(i)
                if a in qm and b in qm:
                    f.setBondParameters(i, a, b, r0, 0.0)
        elif isinstance(f, mm.HarmonicAngleForce):
            for i in range(f.getNumAngles()):
                a, b, c, t0, k = f.getAngleParameters(i)
                if a in qm and b in qm and c in qm:
                    f.setAngleParameters(i, a, b, c, t0, 0.0)
        elif isinstance(f, mm.PeriodicTorsionForce):
            for i in range(f.getNumTorsions()):
                a, b, c, d, n, ph, k = f.getTorsionParameters(i)
                if a in qm and b in qm and c in qm and d in qm:
                    f.setTorsionParameters(i, a, b, c, d, n, ph, 0.0)
        elif isinstance(f, mm.NonbondedForce):
            for i in range(f.getNumParticles()):
                q, sig, eps = f.getParticleParameters(i)
                charges[i] = q.value_in_unit(unit.elementary_charge)
                if i in qm:
                    f.setParticleParameters(i, 0.0, sig, eps)
            for i in range(f.getNumExceptions()):
                a, b, qq, sig, eps = f.getExceptionParameters(i)
                if a in qm or b in qm:
                    f.setExceptionParameters(i, a, b, 0.0 if (a in qm or b in qm) else qq, sig,
                                             0.0 if (a in qm and b in qm) else eps)
            existing = {tuple(sorted(f.getExceptionParameters(i)[:2])) for i in range(f.getNumExceptions())}
            ql = sorted(qm)
            for x in range(len(ql)):
                for y in range(x + 1, len(ql)):
                    if (ql[x], ql[y]) not in existing:
                        f.addException(ql[x], ql[y], 0.0, 1.0, 0.0)
        elif isinstance(f, mm.CMAPTorsionForce):
            pass  # CMAP terms span backbone atoms; a QM backbone region would need them removed
    # constraints between QM atoms
    for i in reversed(range(system.getNumConstraints())):
        a, b, d = system.getConstraintParameters(i)
        if a in qm and b in qm:
            system.removeConstraint(i)
    return charges


class MopacQMMMForce:
    """Builds the PythonForce; keep a reference to this object while the simulation runs."""

    def __init__(self, region: QMMMRegion, libmopac: str, qm_charge: int, positions_nm: np.ndarray,
                 model: int = 0, tolerance: float = 1e-2):
        self.region = region
        self.qm_idx = np.array(region.qm_atoms, dtype=int)
        self.link_q = np.array([q for q, m, g in region.links], dtype=int)
        self.link_m = np.array([m for q, m, g in region.links], dtype=int)
        self.link_g = np.array([g for q, m, g in region.links], dtype=float)
        nums = list(region.atomic_numbers) + [1] * len(region.links)
        # MM charges seen by the QM region: not the QM atoms, not the MM boundary atoms
        self.mm_mask = np.ones(len(region.mm_charges), dtype=bool)
        self.mm_mask[self.qm_idx] = False
        if len(self.link_m):
            self.mm_mask[self.link_m] = False
        self.mm_idx_all = np.nonzero(self.mm_mask & (np.abs(region.mm_charges) > 0))[0]
        # switching groups of the MM charges: member lists (indices into mm_idx_all)
        g = region.groups[self.mm_idx_all]
        self.group_ids, self.group_of = np.unique(g, return_inverse=True)
        self.group_size = np.bincount(self.group_of)
        self.group_first = np.unique(self.group_of, return_index=True)[1]
        x0 = self._qm_coords(positions_nm * 10.0)
        self.session = QMSession(libmopac, nums, x0, charge=qm_charge, model=model, tolerance=tolerance)
        self.calls = 0
        self.last = {}
        self.grid = None

    def _qm_coords(self, pos_a):
        xq = pos_a[self.qm_idx]
        if len(self.link_q):
            xl = pos_a[self.link_q] + self.link_g[:, None] * (pos_a[self.link_m] - pos_a[self.link_q])
            xq = np.vstack([xq, xl])
        return xq

    def compute(self, state):
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer) * 10.0   # A
        n = pos.shape[0]
        box = None
        if self.region.periodic:
            box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer) * 10.0
            # positions may be wrapped: make the QM region (and its link partners) whole
            pos = pos.copy()
            ref = pos[self.qm_idx[0]]
            for idx in (self.qm_idx, self.link_m):
                if len(idx):
                    d = pos[idx] - ref
                    for k in (2, 1, 0):
                        d -= np.outer(np.round(d[:, k] / box[k, k]), box[k])
                    pos[idx] = ref + d
        xq = self._qm_coords(pos)
        # MM charges, minimum image around the QM centroid
        mm_idx = self.mm_idx_all
        xm = pos[mm_idx]
        c = xq.mean(axis=0)
        if self.region.periodic:
            # one image of each MM charge: the box must hold the QM region plus roff on every side
            span = np.max(np.linalg.norm(xq - c, axis=1))
            if 2.0 * (span + self.region.roff) > min(box[0, 0], box[1, 1], box[2, 2]):
                raise ValueError("QM/MM: periodic box %.1f A too small for the QM region (radius %.1f A) plus "
                                 "roff = %.1f A on each side; use a larger box or a smaller roff"
                                 % (min(box[0, 0], box[1, 1], box[2, 2]), span, self.region.roff))
            # whole groups: the first charge of each group nearest to the QM centroid, the other charges of
            # the group nearest to it (imaging every atom separately split waters at half a box, and the
            # switching factor of the split group jumped)
            ref = xm[self.group_first]
            d = ref - c
            for k in (2, 1, 0):
                d -= np.outer(np.round(d[:, k] / box[k, k]), box[k])
            ref = c + d
            d = xm - ref[self.group_of]
            for k in (2, 1, 0):
                d -= np.outer(np.round(d[:, k] / box[k, k]), box[k])
            xm = ref[self.group_of] + d
        # group centres and their smooth-minimum distance to the QM atoms (real QM atoms only)
        ng = len(self.group_ids)
        cg = np.zeros((ng, 3))
        np.add.at(cg, self.group_of, xm)
        cg /= self.group_size[:, None]
        xr = xq[:len(self.qm_idx)]
        rmax = np.max(np.linalg.norm(xr - c, axis=1)) + self.region.roff + 3.0
        near_g = np.nonzero(np.linalg.norm(cg - c, axis=1) < rmax)[0]
        lam = 0.5
        rv = cg[near_g, None, :] - xr[None, :, :]
        r = np.linalg.norm(rv, axis=2)
        rmin = r.min(axis=1)
        ex = np.exp(-(r - rmin[:, None]) / lam)
        dsoft = rmin - lam * np.log(ex.sum(axis=1))
        pA = ex / ex.sum(axis=1)[:, None]                       # d dsoft / d r_A
        ron, roff = self.region.ron, self.region.roff
        x = np.clip((dsoft - ron) / (roff - ron), 0.0, 1.0)
        w = 1.0 - 10*x**3 + 15*x**4 - 6*x**5
        dw = np.where((x > 0) & (x < 1), (-30*x**2 + 60*x**3 - 30*x**4) / (roff - ron), 0.0)
        on = w > 0
        near_g, w, dw, pA, rv, r = near_g[on], w[on], dw[on], pA[on], rv[on], r[on]
        wg = np.zeros(ng)
        wg[near_g] = w
        sel = np.nonzero(wg[self.group_of] > 0)[0]
        xs = xm[sel]
        qs = self.region.mm_charges[mm_idx[sel]] * wg[self.group_of[sel]]
        if self.region.pme:
            far = self._pme_far_potential(xq, xm, xs, qs, box, c)
            heat, gq, gm = self.session.step(xq, xs, qs, 0.0, 0.0, qm_pot=far["phi"])
            pot = self.session.mm_pot.copy()                      # dE/dq_eff, kcal/(mol e)
            g_all = self._pme_far_gradient(far, xq, xs, qs, gq, gm, pot)
        else:
            heat, gq, gm = self.session.step(xq, xs, qs, 0.0, 0.0)
            pot = self.session.mm_pot                             # dE/dq_eff, kcal/(mol e)
        forces = np.zeros((n, 3))
        nq = len(self.qm_idx)
        if self.region.pme:
            forces[mm_idx] -= g_all * GRAD_KCAL_A_TO_KJ_NM
        forces[self.qm_idx] -= gq[:nq] * GRAD_KCAL_A_TO_KJ_NM
        if len(self.link_q):
            fl = -gq[nq:] * GRAD_KCAL_A_TO_KJ_NM
            np.add.at(forces, self.link_q, (1.0 - self.link_g)[:, None] * fl)
            np.add.at(forces, self.link_m, self.link_g[:, None] * fl)
        if len(sel):
            np.add.at(forces, mm_idx[sel], -gm * GRAD_KCAL_A_TO_KJ_NM)
            # switching forces: dE/dw_g = Sum_i q_i pot_i, w = w(dsoft(centre, QM atoms))
            dEdw = np.zeros(ng)
            np.add.at(dEdw, self.group_of[sel], self.region.mm_charges[mm_idx[sel]] * pot)
            coef = dEdw[near_g] * dw                              # dE/d dsoft, kcal/(mol A)
            u = rv / r[:, :, None]                                # unit vectors QM atom -> centre
            gc = coef[:, None] * np.einsum('ga,gak->gk', pA, u)    # dE/d centre
            ga = -coef[:, None, None] * pA[:, :, None] * u         # dE/d QM atom, per group
            # centre = mean of the group's charged atoms
            gmem = gc[np.searchsorted(near_g, self.group_of[sel])] / self.group_size[self.group_of[sel]][:, None]
            np.add.at(forces, mm_idx[sel], -gmem * GRAD_KCAL_A_TO_KJ_NM)
            forces[self.qm_idx] -= ga.sum(axis=0) * GRAD_KCAL_A_TO_KJ_NM
        self.calls += 1
        self.last = dict(heat=heat, n_mm=len(sel))
        return heat * KCAL_TO_KJ, forces

    def _pme_far_potential(self, xq, xm, xs, qs, box, c):
        """phi_far at the QM atoms (kcal/(mol e)) and what its gradient needs"""
        reg = self.region
        if box is None or np.any(np.abs(box - np.diag(np.diag(box))) > 1e-9):
            raise ValueError("QM/MM PME: needs an orthorhombic periodic box")
        L = np.diag(box).copy()
        if self.grid is None or not self.grid.matches(L, reg.pme_alpha, reg.pme_spacing, reg.pme_order):
            self.grid = PMEGrid(L, reg.pme_alpha, reg.pme_spacing, reg.pme_order)
        g, a = self.grid, reg.pme_alpha
        qm_all = reg.mm_charges[self.mm_idx_all]
        st_mm = Stencil(g, xm)
        phi_mm = g.potential(st_mm, qm_all)
        st_q = Stencil(g, xq)
        # real space: every MM charge within pme_cutoff of a QM atom (minimum image of each atom)
        d = xm - c
        d -= L * np.round(d / L)
        span = np.max(np.linalg.norm(xq - c, axis=1))
        near = np.nonzero(np.linalg.norm(d, axis=1) < span + reg.pme_cutoff)[0]
        rv = (c + d[near])[None, :, :] - xq[:, None, :]            # QM atom -> MM charge
        r = np.sqrt(np.einsum('ijk,ijk->ij', rv, rv))
        pa, pj = np.nonzero(r < reg.pme_cutoff)                     # pairs inside the cut-off
        rv, r = rv[pa, pj], r[pa, pj]
        ec = erfc(a * r)
        qn = qm_all[near[pj]]
        phi_real = np.bincount(pa, qn * ec / r, minlength=len(xq))
        # bare Coulomb potential of the charges MOPAC sees (switched)
        rb = xs[None, :, :] - xq[:, None, :]
        rbn = np.linalg.norm(rb, axis=2)
        phi_bare = (qs[None, :] / rbn).sum(axis=1)
        background = -np.pi * qm_all.sum() / (g.V * a * a)
        phi = COULOMB * (phi_real + g.value(st_q, phi_mm) + background - phi_bare)
        return dict(phi=phi, st_mm=st_mm, st_q=st_q, phi_mm=phi_mm, pa=pa, pj=near[pj], rv=rv, r=r, ec=ec, qn=qn,
                    rb=rb, rbn=rbn, qm_all=qm_all)

    def _pme_far_gradient(self, far, xq, xs, qs, gq, gm, pot):
        """Add the gradient of E_far (kcal/(mol A)) to gq (QM atoms) and gm (switched charges, at fixed
        weights), dE_far/dq_eff to pot; returns the gradient on all MM charges from the Ewald part"""
        g, a = self.grid, self.region.pme_alpha
        Q = self.session.qm_charges
        qm_all = far["qm_all"]
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

    def close(self):
        self.session.close()


def add_qmmm(system: mm.System, topology, positions, qm_atoms, qm_charge: int, libmopac: str,
             ron: float = 10.0, roff: float = 12.0, model: int = 0, tolerance: float = 1e-2, pme=None):
    """Turn an OpenMM system into a QM/MM system; returns the MopacQMMMForce (keep it alive).
    pme: long-range QM-MM electrostatics by PME (default: when the NonbondedForce uses PME or Ewald)."""
    if pme is None:
        pme = any(isinstance(f, mm.NonbondedForce) and f.getNonbondedMethod() in (mm.NonbondedForce.PME,
                  mm.NonbondedForce.Ewald) for f in system.getForces())
    if pme and not system.usesPeriodicBoundaryConditions():
        raise ValueError("QM/MM PME needs a periodic system")
    qm_atoms = sorted(int(i) for i in qm_atoms)
    qm_set = set(qm_atoms)
    atoms = list(topology.atoms())
    pos_nm = np.array(positions.value_in_unit(unit.nanometer)) if hasattr(positions, "value_in_unit") \
        else np.asarray(positions, float)
    links = []
    for q, m in _qm_bonds(topology, qm_set):
        zq = _element_number(atoms[q])
        r_qm = np.linalg.norm(pos_nm[q] - pos_nm[m]) * 10.0
        links.append((q, m, XH_LENGTH.get(zq, 1.09) / r_qm))
    charges = prepare_mm_system(system, topology, qm_atoms)
    region = QMMMRegion(qm_atoms=qm_atoms, atomic_numbers=[_element_number(atoms[i]) for i in qm_atoms],
                        links=links, mm_charges=charges, ron=ron, roff=roff,
                        groups=np.array([a.residue.index for a in atoms]),
                        periodic=system.usesPeriodicBoundaryConditions(), pme=bool(pme))
    qmmm = MopacQMMMForce(region, libmopac, qm_charge, pos_nm, model=model, tolerance=tolerance)
    force = mm.PythonForce(qmmm.compute)
    force.setUsesPeriodicBoundaryConditions(region.periodic)
    force.setName("MOPAC QM/MM")
    system.addForce(force)
    qmmm.force = force
    return qmmm

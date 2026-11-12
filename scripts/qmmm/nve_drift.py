#!/usr/bin/env python3
"""Where does the total energy of a QM/MM NVE run drift?  (OpenMM + MOPAC, openmm_mopac.py)

    python nve_drift.py prepare protein.pdb --out prep [--padding 1.0 --variants E:57=HID]
    python nve_drift.py run --prep prep --out run1 --libmopac libmopac.so --qm-residues E:57:sc,I:15 \
        [--tolerance 0.001] [--guess linear] [--mm-only] [--platform CUDA --precision mixed] [--nve 10000]

prepare builds the system as run_qmmm_md.py does (amber14 + TIP3P, hydrogens, water box), minimizes and
equilibrates it with MM only, and saves the topology and the state; every `run` starts from that state, so
that variants (SCF tolerance, precision, MM only) differ in nothing else.

run: optional QM/MM force, Langevin NVT (`--nvt` steps), then velocity-Verlet NVE (CustomIntegrator, so that
velocities and positions belong to the same time).  Each step the QM/MM force is called once; the script
keeps its energy E_qm and its work along the path, W_qm = Sum 1/2 (F_n + F_n+1) . (x_n+1 - x_n).  Every
`--every` steps it writes the kinetic energy, the MM energy (from a second, MM-only context, so the QM/MM force
is not called again) and the total energy to nve.txt.  R_qm = E_qm(t) - E_qm(0) + W_qm(t) is the
non-conservative part of the QM/MM force (zero, up to the trapezoid error, when the force is the exact gradient
of E_qm); the total drift minus R_qm is what the MM part and the integration leave.  summary.json holds the
fitted slopes (kJ/mol/ps) and their standard errors from ten block means, and the mean number of SCF
iterations per QM/MM call in NVT and NVE.  --guess linear starts the SCF of each step from LMOs extrapolated
from the last two steps (MOPAC_QMMM_GUESS) instead of the LMOs of the last step.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openmm_mopac as om  # noqa: E402
from run_qmmm_md import BACKBONE, PROTEIN, parse_residues, residue_matches  # noqa: E402

FF = ("amber14-all.xml", "amber14/tip3p.xml")


def make_system(ff, top, lj_switch):
    s = ff.createSystem(top, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * u.nanometer, constraints=app.HBonds)
    if lj_switch:
        for f in s.getForces():
            if isinstance(f, mm.NonbondedForce):
                f.setUseSwitchingFunction(True)
                f.setSwitchingDistance(0.9 * u.nanometer)
    return s


def platform(name, precision):
    p = mm.Platform.getPlatformByName(name)
    props = {"Precision": precision} if name in ("CUDA", "OpenCL") and precision else {}
    return p, props


def prepare(a):
    os.makedirs(a.out, exist_ok=True)
    ff = app.ForceField(*FF)
    pdb = app.PDBFile(a.pdb)
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.delete([r for r in mod.topology.residues() if r.name not in PROTEIN])
    mod.delete([x for x in mod.topology.atoms() if x.element is not None and x.element.symbol == "H"])
    variants = None
    if a.variants:
        want = {}
        for item in a.variants.split(","):
            spec, name = item.split("=")
            f = spec.split(":")
            want[(f[0], f[1]) if len(f) == 2 else (None, f[0])] = name.strip()
        variants = [next((v for k, v in want.items() if residue_matches(r, k)), None)
                    for r in mod.topology.residues()]
    mod.addHydrogens(ff, pH=7.0, variants=variants)
    mod.addSolvent(ff, padding=a.padding * u.nanometer, neutralize=True)
    top, pos = mod.topology, mod.positions
    integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds)
    integ.setRandomNumberSeed(a.seed)
    p, props = platform(a.platform, a.precision)
    sim = app.Simulation(top, make_system(ff, top, False), integ, p, props)
    sim.context.setPositions(pos)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * u.kelvin, a.seed)
    sim.step(int(a.mm_equil_ps * 500))
    st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
    with open(os.path.join(a.out, "top.pdb"), "w") as fh:
        app.PDBFile.writeFile(top, st.getPositions(), fh, keepIds=True)
    with open(os.path.join(a.out, "state.xml"), "w") as fh:
        fh.write(mm.XmlSerializer.serialize(st))
    print(f"prepared: {top.getNumAtoms()} atoms, box "
          f"{top.getPeriodicBoxVectors()[0][0].value_in_unit(u.angstrom):.1f} A", flush=True)


def vv_integrator(dt_fs, ctol):
    vv = mm.CustomIntegrator(dt_fs * u.femtoseconds)
    vv.addPerDofVariable("x1", 0)
    vv.addUpdateContextState()
    vv.addComputePerDof("v", "v+0.5*dt*f/m")
    vv.addComputePerDof("x", "x+dt*v")
    vv.addComputePerDof("x1", "x")
    vv.addConstrainPositions()
    vv.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
    vv.addConstrainVelocities()
    vv.setConstraintTolerance(ctol)
    return vv


class Recorder:
    """energy, forces and positions of the last QM/MM call and the work of the force along the path"""

    def __init__(self):
        self.x = self.f = self.e = None
        self.w = 0.0
        self.calls = self.same = 0
        self.active = False
        self.iters = {False: [], True: []}     # SCF iterations per call in NVT (False) and NVE (True)

    def __call__(self, state, e, f, iters=None):
        x = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        self.calls += 1
        if iters is not None:
            self.iters[self.active].append(iters)
        if self.active and self.x is not None:
            if np.array_equal(x, self.x):
                self.same += 1       # a second call at the same positions: keep the first
                return
            # OpenMM may wrap the positions it hands to a PythonForce into another box (CUDA and OpenCL: every
            # 250 steps); no atom moves half a box in one step
            dx = x - self.x
            box = np.diag(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
            dx -= box * np.round(dx / box)
            self.w += 0.5 * float(np.sum((self.f + f) * dx))
        self.x, self.f, self.e = x, f, e


def slope_se(t, y):
    """slope of a linear fit and its standard error from ten block means"""
    s = np.polyfit(t, y, 1)[0]
    n = len(y) // 10 * 10
    bt, by = t[:n].reshape(10, -1).mean(1), y[:n].reshape(10, -1).mean(1)
    r = by - np.polyval(np.polyfit(bt, by, 1), bt)
    return float(s), float(np.sqrt(np.sum(r ** 2) / 8 / np.sum((bt - bt.mean()) ** 2)))


def run(a):
    os.makedirs(a.out, exist_ok=True)
    ff = app.ForceField(*FF)
    top = app.PDBFile(os.path.join(a.prep, "top.pdb")).topology
    with open(os.path.join(a.prep, "state.xml")) as fh:
        st0 = mm.XmlSerializer.deserialize(fh.read())
    system = make_system(ff, top, a.lj_switch)
    mm_system = make_system(ff, top, a.lj_switch)          # the MM energy alone, without the QM/MM force
    rec = Recorder()
    qmmm = None
    if not a.mm_only:
        orig = om.MopacQMMMForce.compute

        def compute(self, state):
            e, f = orig(self, state)
            rec(state, e, f, self.session.scf_iterations)
            return e, f
        om.MopacQMMMForce.compute = compute
        wanted = parse_residues(a.qm_residues)
        qm = [x.index for x in top.atoms() for spec, sc in wanted
              if residue_matches(x.residue, spec) and not (sc and x.name in BACKBONE)]
        nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
        qch = int(round(sum(nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in qm)))
        qmmm = om.add_qmmm(system, top, st0.getPositions(), qm, qch, a.libmopac, ron=a.ron, roff=a.roff,
                           tolerance=a.tolerance, pme_gpu=False if a.pme_cpu else None, guess=a.guess or None)
        om.prepare_mm_system(mm_system, top, qm)
        print(f"QM region: {len(qm)} atoms + {len(qmmm.region.links)} link atoms, charge {qch}, tolerance "
              f"{a.tolerance}, PME grid on the {'GPU' if qmmm.region.pme_gpu else 'CPU'}", flush=True)
    p, props = platform(a.platform, a.precision)
    masses = np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())])

    integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, a.dt * u.femtoseconds)
    integ.setRandomNumberSeed(a.seed)
    ctx = mm.Context(system, integ, p, props)
    ctx.setPeriodicBoxVectors(*st0.getPeriodicBoxVectors())
    ctx.setPositions(st0.getPositions())
    ctx.setVelocities(st0.getVelocities())
    t0 = time.perf_counter()
    integ.step(a.nvt)
    t_nvt = time.perf_counter() - t0
    st = ctx.getState(getPositions=True, getVelocities=True)
    del ctx, integ

    vv = vv_integrator(a.dt, a.ctol)
    ctx = mm.Context(system, vv, p, props)
    ctx.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
    ctx.setPositions(st.getPositions())
    ctx.setVelocities(st.getVelocities())
    ctx.applyConstraints(a.ctol)
    ctx.applyVelocityConstraints(a.ctol)
    mctx = mm.Context(mm_system, mm.VerletIntegrator(1.0), p, props)
    mctx.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
    rows = []

    def reading(k):
        s = ctx.getState(getPositions=True, getVelocities=True)
        v = s.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
        ke = 0.5 * float(np.sum(masses[:, None] * v * v))
        mctx.setPositions(s.getPositions())
        emm = mctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        eqm = rec.e if qmmm else 0.0
        rows.append((k * a.dt / 1000.0, ke + emm + eqm, ke, emm, eqm, rec.w))

    if qmmm:
        rec.active, rec.x = True, None
        ctx.getState(getForces=True)        # the QM/MM call at the start of the NVE run
    reading(0)
    t0 = time.perf_counter()
    for k in range(1, a.nve + 1):
        vv.step(1)
        if k % a.every == 0:
            reading(k)
    t_nve = time.perf_counter() - t0
    r = np.array(rows)
    np.savetxt(os.path.join(a.out, "nve.txt"), r, header="t_ps E_total KE E_mm E_qm W_qm (kJ/mol)", fmt="%.6f")
    t, et = r[:, 0], r[:, 1]
    fit = np.polyfit(t, et, 1)
    out = dict(vars(a), atoms=system.getNumParticles(), nvt_s_per_step=t_nvt / max(1, a.nvt),
               nve_s_per_step=t_nve / a.nve, drift_total=float(et[-1] - et[0]),
               rms_dev=float(np.std(et - np.polyval(fit, t))), qm_calls=rec.calls, qm_same_position_calls=rec.same)
    for phase, it in (("nvt", rec.iters[False][1:]), ("nve", rec.iters[True])):
        if it:
            out[f"scf_iterations_{phase}"] = float(np.mean(it))
    out["slope_total"], out["slope_total_se"] = slope_se(t, et)
    if qmmm:
        res = (r[:, 4] - r[0, 4]) + (r[:, 5] - r[0, 5])
        out["slope_qm_nonconservative"], out["slope_qm_nonconservative_se"] = slope_se(t, res)
        out["slope_rest"], out["slope_rest_se"] = slope_se(t, et - res)
        qmmm.close()
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("NVE drift: " + ", ".join(f"{k} {out[k]:.3g}" for k in out if k.startswith(("slope", "nve_s", "nvt_s"))),
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "run"])
    ap.add_argument("pdb", nargs="?", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", default="", help="directory written by prepare")
    ap.add_argument("--padding", type=float, default=1.0, help="water padding, nm (prepare)")
    ap.add_argument("--variants", default="", help="protonation states, e.g. E:57=HID (prepare)")
    ap.add_argument("--mm-equil-ps", type=float, default=20.0, help="MM equilibration, 2-fs steps (prepare)")
    ap.add_argument("--libmopac", default="")
    ap.add_argument("--qm-residues", default="", help="as run_qmmm_md.py, e.g. E:57:sc,I:15")
    ap.add_argument("--ron", type=float, default=10.0)
    ap.add_argument("--roff", type=float, default=12.0)
    ap.add_argument("--tolerance", type=float, default=0.01, help="MOPAC relative SCF criterion (RELSCF)")
    ap.add_argument("--mm-only", action="store_true", help="no QM region")
    ap.add_argument("--pme-cpu", action="store_true", help="the QM-MM PME grid on the CPU")
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--precision", default="", help="CUDA/OpenCL: single, mixed or double")
    ap.add_argument("--dt", type=float, default=1.0, help="fs")
    ap.add_argument("--nvt", type=int, default=400)
    ap.add_argument("--nve", type=int, default=10000)
    ap.add_argument("--every", type=int, default=10, help="steps between energy readings")
    ap.add_argument("--ctol", type=float, default=1e-8, help="constraint tolerance of the NVE run")
    ap.add_argument("--lj-switch", action="store_true", help="Lennard-Jones switching from 0.9 nm")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--guess", choices=["previous", "linear", "quadratic", "cubic"], default="",
                    help="starting LMOs of each step's SCF (default: MOPAC's, see openmm_mopac.add_qmmm)")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    main()

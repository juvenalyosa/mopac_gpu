#!/usr/bin/env python3
"""QM/MM molecular dynamics with OpenMM + MOPAC/MOZYME (example and benchmark).

    python run_qmmm_md.py protein.pdb --libmopac /path/libmopac.so --qm-residues 2 \
        [--qm-sidechain] [--qm-charge 0] [--padding 1.6] [--ron 8 --roff 10] \
        [--steps 200] [--dt 0.5] [--temperature 300] [--platform CUDA] [--nve 0]

Prepares the system with OpenMM (hydrogens, amber14 + TIP3P, a water box with `padding` nm, ions), relaxes
and equilibrates it with MM only (minimization + 20 ps NVT, 2 fs), then turns the selected residues (whole,
or only their side chains with an H link atom on CA-CB) into the QM region (PM7, MOZYME, electrostatic
embedding) and runs Langevin NVT (LangevinMiddleIntegrator, 1/ps) with `dt` fs; optionally a short NVE
check of the energy conservation.  Prints the time per step and writes qmmm_traj.pdb.
"""

import argparse
import os
import sys
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openmm_mopac import add_qmmm  # noqa: E402

BACKBONE = ("N", "H", "CA", "HA", "C", "O", "OXT", "H1", "H2", "H3", "HA2", "HA3")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb")
    ap.add_argument("--libmopac", required=True)
    ap.add_argument("--qm-residues", default="", help="residue numbers (PDB), comma separated")
    ap.add_argument("--qm-sidechain", action="store_true", help="only the side chains (link atom on CA-CB)")
    ap.add_argument("--qm-charge", type=int, default=None, help="default: sum of the MM charges, rounded")
    ap.add_argument("--padding", type=float, default=1.6)
    ap.add_argument("--ron", type=float, default=8.0)
    ap.add_argument("--roff", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--nve", type=int, default=0, help="steps of an NVE check at the end")
    ap.add_argument("--mm-equil-ps", type=float, default=20.0)
    a = ap.parse_args()

    plat = mm.Platform.getPlatformByName(a.platform)
    pdb = app.PDBFile(a.pdb)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.delete([x for x in mod.topology.atoms() if x.residue.name == "HOH"])
    mod.delete([x for x in mod.topology.atoms() if x.element is not None and x.element.symbol == "H"])
    mod.addHydrogens(ff, pH=7.0)
    mod.addSolvent(ff, padding=a.padding * u.nanometer, neutralize=True)
    top, pos = mod.topology, mod.positions
    print(f"system: {top.getNumAtoms()} atoms, {top.getNumResidues()} residues, box "
          f"{top.getPeriodicBoxVectors()[0][0].value_in_unit(u.angstrom):.1f} A", flush=True)

    def make_system():
        return ff.createSystem(top, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * u.nanometer,
                               constraints=app.HBonds)

    # MM equilibration
    t0 = time.perf_counter()
    sim = app.Simulation(top, make_system(), mm.LangevinMiddleIntegrator(a.temperature * u.kelvin,
                         1 / u.picosecond, 2 * u.femtoseconds), plat)
    sim.context.setPositions(pos)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(a.temperature * u.kelvin, 1)
    sim.step(int(a.mm_equil_ps * 500))
    st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
    print(f"MM equilibration {a.mm_equil_ps} ps: {time.perf_counter() - t0:.1f} s", flush=True)

    # QM region
    wanted = {r.strip() for r in a.qm_residues.split(",") if r.strip()}
    qm = [x.index for x in top.atoms() if x.residue.id in wanted and
          (not a.qm_sidechain or x.name not in BACKBONE)]
    system = make_system()
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    qcharge = a.qm_charge
    if qcharge is None:
        qcharge = int(round(sum(nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in qm)))
    qmmm = add_qmmm(system, top, st.getPositions(), qm, qcharge, a.libmopac, ron=a.ron, roff=a.roff,
                    tolerance=a.tolerance)
    print(f"QM region: {len(qm)} atoms + {len(qmmm.region.links)} link atoms, charge {qcharge}", flush=True)

    integ = mm.LangevinMiddleIntegrator(a.temperature * u.kelvin, 1 / u.picosecond, a.dt * u.femtoseconds)
    sim = app.Simulation(top, system, integ, plat)
    sim.context.setPositions(st.getPositions())
    sim.context.setVelocities(st.getVelocities())
    sim.reporters.append(app.PDBReporter("qmmm_traj.pdb", max(1, a.steps // 10)))
    sim.step(2)   # first SCF from the Lewis structure
    t0 = time.perf_counter()
    sim.step(a.steps)
    t = time.perf_counter() - t0
    print(f"QM/MM NVT: {a.steps} steps x {a.dt} fs in {t:.1f} s = {t / a.steps:.3f} s/step, "
          f"{a.steps * a.dt / 1000 / (t / 86400):.2f} ps/day; MM charges seen {qmmm.last['n_mm']}", flush=True)

    if a.nve > 0:
        st = sim.context.getState(getPositions=True, getVelocities=True)
        integ2 = mm.VerletIntegrator(a.dt * u.femtoseconds)
        ctx = mm.Context(system, integ2, plat)
        ctx.setPositions(st.getPositions())
        ctx.setVelocities(st.getVelocities())
        e = []
        for _ in range(a.nve):
            s = ctx.getState(getEnergy=True)
            e.append((s.getPotentialEnergy() + s.getKineticEnergy()).value_in_unit(u.kilojoule_per_mole))
            integ2.step(1)
        e = np.array(e)
        de = np.diff(e)
        print(f"NVE check {a.nve} steps: drift {e[-1] - e[0]:.2f} kJ/mol, largest step {np.max(np.abs(de)):.2f}, "
              f"rms step {np.sqrt(np.mean(de ** 2)):.3f}", flush=True)
    qmmm.close()


if __name__ == "__main__":
    main()

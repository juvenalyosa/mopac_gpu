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

Residues: "57" (any chain) or "E:57" (chain E); a suffix ":sc" takes only the side chain (all side chains
with --qm-sidechain).  Waters, ions and other non-protein residues of the PDB file are removed.
--variants E:57=HID sets the protonation state of a residue.  --distances E:195:OG-I:15:C,... prints
the mean, smallest and largest distance during the QM/MM run.  --cutoff-test 8,12,16 recomputes the QM/MM
forces on snapshots with these roff (ron = roff - 2), with and without the long-range part (PME), and
compares them with PME at the largest roff.  --no-pme: cut-off QM-MM electrostatics only; --pme-cpu: the
PME grid on the CPU (NumPy) even when CuPy sees a GPU.
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
PROTEIN = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
           "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN", "ACE",
           "NME"}


def residue_matches(res, spec):
    """spec = (chain or None, residue number)"""
    chain, rid = spec
    return res.id == rid and (chain is None or res.chain.id == chain)


def parse_residues(text):
    """"57,E:102,E:195:sc" -> [((chain, id), side_chain_only)]"""
    out = []
    for item in (x.strip() for x in text.split(",")):
        if not item:
            continue
        f = item.split(":")
        sc = f[-1].lower() == "sc"
        if sc:
            f = f[:-1]
        out.append(((f[0], f[1]) if len(f) == 2 else (None, f[0]), sc))
    return out


def find_atom(top, text):
    """"E:195:OG" -> atom index"""
    chain, rid, name = text.split(":")
    for a in top.atoms():
        if a.name == name and residue_matches(a.residue, (chain, rid)):
            return a.index
    raise ValueError("atom not found: " + text)


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
    ap.add_argument("--variants", default="", help="protonation states, e.g. E:57=HID,E:102=ASP")
    ap.add_argument("--distances", default="", help="atom pairs to follow, e.g. E:195:OG-I:15:C")
    ap.add_argument("--cutoff-test", default="", help="roff values (A) for a force comparison on snapshots")
    ap.add_argument("--snapshots", type=int, default=5, help="snapshots of the cut-off test")
    ap.add_argument("--no-pme", action="store_true", help="no long-range QM-MM electrostatics (cut-off only)")
    ap.add_argument("--pme-cpu", action="store_true", help="PME grid on the CPU even when CuPy sees a GPU")
    a = ap.parse_args()

    plat = mm.Platform.getPlatformByName(a.platform)
    pdb = app.PDBFile(a.pdb)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
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
    wanted = parse_residues(a.qm_residues)
    qm = []
    for x in top.atoms():
        for spec, sc in wanted:
            if residue_matches(x.residue, spec) and not ((sc or a.qm_sidechain) and x.name in BACKBONE):
                qm.append(x.index)
                break
    system = make_system()
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    qcharge = a.qm_charge
    if qcharge is None:
        qcharge = int(round(sum(nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in qm)))
    qmmm = add_qmmm(system, top, st.getPositions(), qm, qcharge, a.libmopac, ron=a.ron, roff=a.roff,
                    tolerance=a.tolerance, pme=False if a.no_pme else None,
                    pme_gpu=False if a.pme_cpu else None)
    lr = "none (cut-off)" if not qmmm.region.pme else "PME on the GPU" if qmmm.region.pme_gpu else "PME"
    print(f"QM region: {len(qm)} atoms + {len(qmmm.region.links)} link atoms, charge {qcharge}; long-range "
          f"QM-MM electrostatics: {lr}", flush=True)

    integ = mm.LangevinMiddleIntegrator(a.temperature * u.kelvin, 1 / u.picosecond, a.dt * u.femtoseconds)
    sim = app.Simulation(top, system, integ, plat)
    sim.context.setPositions(st.getPositions())
    sim.context.setVelocities(st.getVelocities())
    sim.reporters.append(app.PDBReporter("qmmm_traj.pdb", max(1, a.steps // 10)))
    sim.step(2)   # first SCF from the Lewis structure
    pairs = [(find_atom(top, p.split("-")[0]), find_atom(top, p.split("-")[1]), p)
             for p in a.distances.split(",") if p.strip()]
    nsnap = max(1, a.snapshots) if a.cutoff_test else 0
    chunk = max(1, a.steps // max(20, nsnap))
    dist = [[] for _ in pairs]
    snaps = []
    t0 = time.perf_counter()
    done = 0
    while done < a.steps:
        n = min(chunk, a.steps - done)
        sim.step(n)
        done += n
        if pairs or nsnap:
            st2 = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
            x = st2.getPositions(asNumpy=True).value_in_unit(u.angstrom)
            for k, (i, j, _) in enumerate(pairs):
                dist[k].append(np.linalg.norm(x[i] - x[j]))
            if nsnap and done * nsnap >= (len(snaps) + 1) * a.steps:
                snaps.append(st2)
    t = time.perf_counter() - t0
    print(f"QM/MM NVT: {a.steps} steps x {a.dt} fs in {t:.1f} s = {t / a.steps:.3f} s/step, "
          f"{a.steps * a.dt / 1000 / (t / 86400):.2f} ps/day; MM charges seen {qmmm.last['n_mm']}", flush=True)

    for (i, j, name), d in zip(pairs, dist):
        d = np.array(d)
        print(f"distance {name}: mean {d.mean():.2f} A, min {d.min():.2f}, max {d.max():.2f}, last {d[-1]:.2f}",
              flush=True)

    if snaps:
        cuts = sorted(float(c) for c in a.cutoff_test.split(","))
        ron0, roff0, pme0 = qmmm.region.ron, qmmm.region.roff, qmmm.region.pme
        idx = np.array(qmmm.region.qm_atoms)
        modes = (False, True) if pme0 else (False,)
        res = {(m, c): [] for m in modes for c in cuts}
        for st2 in snaps:
            for m in modes:
                for c in cuts:
                    qmmm.region.ron, qmmm.region.roff, qmmm.region.pme = c - 2.0, c, m
                    e, f = qmmm.compute(st2)
                    res[(m, c)].append((e, f[idx], qmmm.last["n_mm"]))
        qmmm.region.ron, qmmm.region.roff, qmmm.region.pme = ron0, roff0, pme0
        ref = (modes[-1], cuts[-1])
        print(f"cut-off test on {len(snaps)} snapshots, reference: {'PME, ' if pme0 else ''}roff = {cuts[-1]:.0f} A "
              f"(forces on the QM atoms, kJ/(mol nm)):", flush=True)
        fr = np.array([q[1] for q in res[ref]])
        for m in modes:
            for c in cuts:
                if (m, c) == ref:
                    continue
                de = [r[0] - q[0] for r, q in zip(res[(m, c)], res[ref])]
                df = np.array([r[1] - q[1] for r, q in zip(res[(m, c)], res[ref])])
                print(f"   {'PME' if m else 'cut-off'} roff {c:4.0f} A ({np.mean([r[2] for r in res[(m, c)]]):.0f} MM "
                      f"charges): rms force error {np.sqrt(np.mean(df ** 2)):.1f} (rms force "
                      f"{np.sqrt(np.mean(fr ** 2)):.0f}), largest {np.max(np.abs(df)):.1f}; energy change between "
                      f"snapshots: spread {np.ptp(de):.1f} kJ/mol", flush=True)

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
        slope = np.polyfit(np.arange(a.nve) * a.dt / 1000.0, e, 1)[0]
        print(f"NVE check {a.nve} steps: drift {e[-1] - e[0]:.2f} kJ/mol, slope of a linear fit {slope:.1f} "
              f"kJ/mol/ps, largest step {np.max(np.abs(de)):.2f}, rms step {np.sqrt(np.mean(de ** 2)):.3f}, "
              f"rms deviation from the fit {np.std(e - np.polyval(np.polyfit(np.arange(a.nve), e, 1), np.arange(a.nve))):.2f}",
              flush=True)
        np.savetxt("nve_energy.txt", e, fmt="%.4f")
    qmmm.close()


if __name__ == "__main__":
    main()

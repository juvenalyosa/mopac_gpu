#!/usr/bin/env python3
"""QM/MM scan of a reaction coordinate (adiabatic mapping) with OpenMM + MOPAC/MOZYME.

    python qmmm_scan.py 2PTC.pdb --libmopac build/libmopac.so \\
        --qm-residues E:57:sc,E:102:sc,E:195:sc,I:15,I:16 --variants E:57=HID \\
        --scan E:195:OG-I:15:C --to 1.5 --step 0.1 --back \\
        --watch E:195:OG-E:195:HG,E:195:HG-E:57:NE2,I:15:C-I:15:O,I:15:C-I:16:N

    # two coordinates: the attack and the proton transfer d(OG-HG) - d(HG-NE2), on a grid
    python qmmm_scan.py 2PTC.pdb ... --scan E:195:OG-I:15:C --to 1.5 --step 0.15 \\
        --scan2 E:195:OG-E:195:HG-E:57:NE2 --from2 -1.1 --to2 1.1 --step2 0.2

Prepares the system as run_qmmm_md.py does (hydrogens, amber14 + TIP3P, a water box, ions, MM equilibration), then
minimizes it with QM/MM (the reactant), and moves the distance given by --scan from its minimized value to --to in
steps of --step with a harmonic restraint (--k, kJ/mol/A^2), minimizing the whole system at each value from the
structure of the value before; with --back it goes back to the start the same way.  A coordinate is a distance
"A-B" or a difference of two distances "A-B-C" = d(A,B) - d(B,C) (for a proton B moving from A to C).  With --scan2 the
two coordinates are restrained together on a grid, row by row: for each value of the first, the second runs from
--from2 to --to2 and back on the next row (each window starts from the one before).  Only the residues with an atom
within --free-radius of the QM region move freely; the others are held at their equilibrated positions by harmonic
restraints (--hold), so that each window relaxes the active site rather than the whole water box.

Writes scan.csv, one row per window: the restraint target and the value reached (target, d; target2, d2 for the
second coordinate), the potential energy without the restraints (kJ/mol, MM + QM/MM), the energy of the scan
restraints, the QM heat of formation (kcal/mol), the watched distances and
the minimizer's time; and scan_frames.pdb, the residues within 5 A of the QM region, one model per window.  Every
energy belongs to one minimized structure (no sampling, no entropy, no zero-point energy): this is a potential-
energy profile along one path, not a free-energy profile.  Differences between the forward and the backward scan
show where the path depends on its history.
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
from run_qmmm_md import PROTEIN, BACKBONE, parse_residues, residue_matches, find_atom  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb")
    ap.add_argument("--libmopac", required=True)
    ap.add_argument("--qm-residues", required=True, help="as in run_qmmm_md.py, e.g. E:57:sc,E:195:sc,I:15")
    ap.add_argument("--qm-charge", type=int, default=None, help="default: sum of the MM charges, rounded")
    ap.add_argument("--variants", default="", help="protonation states, e.g. E:57=HID")
    ap.add_argument("--padding", type=float, default=1.0, help="water around the protein (nm)")
    ap.add_argument("--ron", type=float, default=10.0)
    ap.add_argument("--roff", type=float, default=12.0)
    ap.add_argument("--tolerance", type=float, default=0.001, help="MOPAC's relative SCF criterion")
    ap.add_argument("--scan", required=True, help="the restrained coordinate: a distance E:195:OG-I:15:C, or "
                    "d(A,B) - d(B,C) as A-B-C")
    ap.add_argument("--to", type=float, required=True, help="last target (A)")
    ap.add_argument("--step", type=float, default=0.1, help="A")
    ap.add_argument("--k", type=float, default=2000.0, help="restraint force constant (kJ/mol/A^2)")
    ap.add_argument("--scan2", default="", help="a second coordinate (grid scan)")
    ap.add_argument("--from2", type=float, default=None, help="first value of the second coordinate (default: "
                    "its value in the reactant)")
    ap.add_argument("--to2", type=float, default=None)
    ap.add_argument("--step2", type=float, default=0.2)
    ap.add_argument("--back", action="store_true", help="scan back to the start afterwards")
    ap.add_argument("--watch", default="", help="distances to report, e.g. E:195:HG-E:57:NE2")
    ap.add_argument("--free-radius", type=float, default=10.0, help="residues within this distance (A) of the QM "
                    "region move freely; the others are restrained")
    ap.add_argument("--hold", type=float, default=50.0, help="restraint on the other atoms (kJ/mol/A^2)")
    ap.add_argument("--min-iter", type=int, default=200, help="minimizer iterations per window")
    ap.add_argument("--min-iter-start", type=int, default=1000, help="minimizer iterations for the reactant")
    ap.add_argument("--min-tol", type=float, default=5.0, help="minimizer tolerance (kJ/mol/nm)")
    ap.add_argument("--mm-equil-ps", type=float, default=20.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="scan")
    a = ap.parse_args()

    plat = mm.Platform.getPlatformByName(a.platform)
    props = {"Precision": "mixed"} if a.platform == "CUDA" else {}
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
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
    print(f"system: {top.getNumAtoms()} atoms, box {top.getPeriodicBoxVectors()[0][0].value_in_unit(u.angstrom):.1f} A",
          flush=True)

    def make_system():
        return ff.createSystem(top, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * u.nanometer,
                               constraints=app.HBonds)

    # MM equilibration, as run_qmmm_md.py
    t0 = time.perf_counter()
    integ = mm.LangevinMiddleIntegrator(a.temperature * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds)
    integ.setRandomNumberSeed(a.seed)
    sim = app.Simulation(top, make_system(), integ, plat, props)
    sim.context.setPositions(pos)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(a.temperature * u.kelvin, a.seed)
    sim.step(int(a.mm_equil_ps * 500))
    x0 = sim.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions()
    print(f"MM equilibration {a.mm_equil_ps} ps: {time.perf_counter() - t0:.1f} s", flush=True)

    # QM region and restraint
    wanted = parse_residues(a.qm_residues)
    qm = [x.index for x in top.atoms() for spec, sc in wanted
          if residue_matches(x.residue, spec) and not (sc and x.name in BACKBONE)]
    system = make_system()
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    qcharge = a.qm_charge
    if qcharge is None:
        qcharge = int(round(sum(nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in qm)))
    qmmm = add_qmmm(system, top, x0, qm, qcharge, a.libmopac, ron=a.ron, roff=a.roff, tolerance=a.tolerance)
    print(f"QM region: {len(qm)} atoms + {len(qmmm.region.links)} link atoms, charge {qcharge}", flush=True)
    coords = [[find_atom(top, s) for s in a.scan.split("-")]]
    if a.scan2:
        coords.append([find_atom(top, s) for s in a.scan2.split("-")])
    for n, c in enumerate(coords):
        expr = "distance(p1,p2)" if len(c) == 2 else "distance(p1,p2)-distance(p2,p3)"
        rest = mm.CustomCompoundBondForce(len(c), f"0.5*k_c{n}*({expr}-r0_c{n})^2")
        rest.addGlobalParameter(f"k_c{n}", 0.0)
        rest.addGlobalParameter(f"r0_c{n}", 0.0)
        rest.addBond(c, [])
        rest.setUsesPeriodicBoundaryConditions(True)
        rest.setForceGroup(30)
        system.addForce(rest)
    # hold the residues away from the active site
    xa = np.array(x0.value_in_unit(u.angstrom))
    dmin = np.full(top.getNumAtoms(), np.inf)
    for i in qm:
        dmin = np.minimum(dmin, np.linalg.norm(xa - xa[i], axis=1))
    free_res = {x.residue.index for x in top.atoms() if dmin[x.index] < a.free_radius}
    hold = mm.CustomExternalForce("0.5*k_hold*periodicdistance(x, y, z, x0, y0, z0)^2")
    hold.addGlobalParameter("k_hold", a.hold * 100.0)
    for p_ in ("x0", "y0", "z0"):
        hold.addPerParticleParameter(p_)
    nheld = 0
    for x in top.atoms():
        if x.residue.index not in free_res:
            hold.addParticle(x.index, (xa[x.index] / 10.0).tolist())
            nheld += 1
    hold.setForceGroup(29)
    system.addForce(hold)
    print(f"free: {top.getNumAtoms() - nheld} atoms in {len(free_res)} residues within {a.free_radius} A of the QM "
          f"region; held: {nheld}", flush=True)
    watch = [(find_atom(top, p.split("-")[0]), find_atom(top, p.split("-")[1]), p)
             for p in a.watch.split(",") if p.strip()]
    sim = app.Simulation(top, system, mm.VerletIntegrator(0.5 * u.femtoseconds), plat, props)
    sim.context.setPositions(x0)

    # residues near the QM region, for the frames
    near = {top_atom.residue.index for top_atom in top.atoms()
            if np.min(np.linalg.norm(xa[qm] - xa[top_atom.index], axis=1)) < 5.0}
    sub = app.Modeller(top, x0)
    sub.delete([r for r in sub.topology.residues() if r.index not in near])
    keep = [x.index for x in top.atoms() if x.residue.index in near]
    frames = open(f"{a.out}_frames.pdb", "w")
    app.PDBFile.writeHeader(sub.topology, frames)

    def dist(x, i, j):
        return float(np.linalg.norm(x[i] - x[j]))

    def value(x, c):
        return dist(x, c[0], c[1]) if len(c) == 2 else dist(x, c[0], c[1]) - dist(x, c[1], c[2])

    rows = []

    def window(label, targets, iters):
        """targets: one value per coordinate (A), or None for no restraint"""
        t1 = time.perf_counter()
        for n in range(len(coords)):
            if targets is None:
                sim.context.setParameter(f"k_c{n}", 0.0)
            else:
                sim.context.setParameter(f"k_c{n}", a.k * 100.0)        # kJ/mol/A^2 -> kJ/mol/nm^2
                sim.context.setParameter(f"r0_c{n}", targets[n] / 10.0)
        sim.minimizeEnergy(tolerance=a.min_tol * u.kilojoule_per_mole / u.nanometer, maxIterations=iters)
        st = sim.context.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=False)
        e_all = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        e_rest = sim.context.getState(getEnergy=True, groups={30}).getPotentialEnergy().value_in_unit(
            u.kilojoule_per_mole)
        e_hold = sim.context.getState(getEnergy=True, groups={29}).getPotentialEnergy().value_in_unit(
            u.kilojoule_per_mole)
        x = np.array(st.getPositions().value_in_unit(u.angstrom))
        tg = targets if targets is not None else [float("nan")] * len(coords)
        row = dict(window=label, target=tg[0], d=value(x, coords[0]))
        if len(coords) > 1:
            row.update(target2=tg[1], d2=value(x, coords[1]))
        row.update(e=e_all - e_rest - e_hold, e_rest=e_rest, e_hold=e_hold, heat=qmmm.last.get("heat", float("nan")),
                   seconds=time.perf_counter() - t1, **{w[2]: dist(x, w[0], w[1]) for w in watch})
        rows.append(row)
        app.PDBFile.writeModel(sub.topology, st.getPositions(asNumpy=True)[keep], frames, modelIndex=len(rows),
                               keepIds=True)
        print(f"{label:>8s} target {row['target']:6.3f}  d {row['d']:6.3f}  "
              + (f"target2 {row['target2']:6.3f}  d2 {row['d2']:6.3f}  " if len(coords) > 1 else "")
              + f"E {row['e'] - rows[0]['e']:9.2f} kJ/mol  "
              f"restraint {e_rest:7.2f}  " + "  ".join(f"{w[2]} {row[w[2]]:.3f}" for w in watch)
              + f"  ({row['seconds']:.0f} s)", flush=True)

    window("reactant", None, a.min_iter_start)
    d0 = rows[0]["d"]
    n = int(round(abs(d0 - a.to) / a.step))
    targets = [d0 + np.sign(a.to - d0) * a.step * (k + 1) for k in range(n)]
    if len(coords) == 1:
        for k, r in enumerate(targets):
            window(f"f{k + 1}", [float(r)], a.min_iter)
        if a.back:
            for k, r in enumerate(targets[-2::-1] + [d0]):
                window(f"b{k + 1}", [float(r)], a.min_iter)
    else:
        s0 = rows[0]["d2"] if a.from2 is None else a.from2
        m = int(round(abs(a.to2 - s0) / a.step2))
        col = [s0 + np.sign(a.to2 - s0) * a.step2 * k for k in range(m + 1)]
        for i, r in enumerate([d0] + targets):
            for j, s in enumerate(col if i % 2 == 0 else col[::-1]):
                window(f"g{i}_{j if i % 2 == 0 else m - j}", [float(r), float(s)], a.min_iter)
    app.PDBFile.writeFooter(sub.topology, frames)
    frames.close()
    keys = list(rows[0])
    with open(f"{a.out}.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.6f}" for k in keys) + "\n")
    qmmm.close()


if __name__ == "__main__":
    main()

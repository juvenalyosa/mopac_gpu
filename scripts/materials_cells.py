"""Periodic MOPAC test cells (Tv) for the materials GPU work (tests/data/materials), built with ASE.

Usage: materials_cells.py OUTDIR KEYWORDS [rep]   (rep=1: small/, rep=2: medium/ with TIMES)
"""
import sys
from pathlib import Path
import numpy as np
from ase.build import bulk, add_adsorbate, molecule
from ase.spacegroup import crystal

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
keys = sys.argv[2]
rep = int(sys.argv[3]) if len(sys.argv) > 3 else 1


def urea():
    return crystal(["C", "O", "N", "H", "H"],
                   [(0, 0.5, 0.3260), (0, 0.5, 0.5953), (0.1459, 0.6459, 0.1766),
                    (0.2575, 0.7575, 0.2827), (0.1441, 0.6441, -0.0380)],
                   spacegroup=113, cellpar=[5.565, 5.565, 4.684, 90, 90, 90])


def quartz():
    return crystal(["Si", "O"], [(0.4697, 0, 2 / 3), (0.4135, 0.2669, 0.7857667)], spacegroup=154,
                   cellpar=[4.916, 4.916, 5.405, 90, 90, 120])


def rutile():
    return crystal(["Ti", "O"], [(0, 0, 0), (0.3053, 0.3053, 0)], spacegroup=136,
                   cellpar=[4.594, 4.594, 2.959, 90, 90, 90])


def zno():
    return crystal(["Zn", "O"], [(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.382)], spacegroup=186,
                   cellpar=[3.25, 3.25, 5.207, 90, 90, 120])


def mgo_slab():
    from ase.build import fcc100
    s = bulk("MgO", "rocksalt", a=4.212, cubic=True)
    s = s.repeat((2, 2, 2))          # 64 atoms, 8.4 A
    s.cell[2] = [0, 0, 0]
    w = molecule("H2O")
    w.rotate(180, "x")              # hydrogens away from the surface
    w.translate(s.positions[np.argmax(s.positions[:, 2])] + [0, 0, 2.2] - w.positions[0])
    s += w
    s.pbc = [True, True, False]
    return s


cells = {"urea": (urea(), (2, 2, 3)), "diamond": (bulk("C", "diamond", a=3.567, cubic=True), (2, 2, 2)),
         "silicon": (bulk("Si", "diamond", a=5.431, cubic=True), (2, 2, 2)),
         "quartz": (quartz(), (2, 2, 2)), "rutile": (rutile(), (2, 2, 3)), "zno": (zno(), (3, 3, 2)),
         "mgo_water": (mgo_slab(), (1, 1, 1))}
for name, (a, r) in cells.items():
    if name != "mgo_water":
        a = a.repeat(tuple(x * rep for x in r))
    else:
        a = a.repeat((rep, rep, 1))
    lines = [keys, f"{name} {len(a)} atoms", ""]
    for s, p in zip(a.get_chemical_symbols(), a.positions):
        lines.append(f"{s:2s} {p[0]:12.6f} 1 {p[1]:12.6f} 1 {p[2]:12.6f} 1")
    for v, periodic in zip(a.cell, a.pbc):
        if periodic and np.linalg.norm(v) > 0:
            lines.append(f"Tv {v[0]:12.6f} 0 {v[1]:12.6f} 0 {v[2]:12.6f} 0")
    (out / f"{name}.mop").write_text("\n".join(lines) + "\n")
    print(name, len(a), "atoms", "cell", np.round(a.cell.lengths(), 2))

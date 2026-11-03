#!/usr/bin/env python3
"""Central differences of the heat of formation for selected Cartesian coordinates of a PDB geometry, to decide
between two analytic gradients (e.g. MOPAC 23.1.2 and this fork, which give the same heat but different gradients).

    fd_gradient.py --mopac build/mopac --pdb geom.pdb --keywords "PM7 1SCF PULAY GEO-OK SCFCRT=1.D-8" \
                   --aux a.aux [--compare-aux b.aux] [--n 6] [--step 0.005] [--csv out.csv] [--work DIR]

The coordinates checked are the n with the largest |gradient(a) - gradient(b)| (or of |gradient(a)| without
--compare-aux).  Each displaced geometry is the PDB with one coordinate moved by +-step (the PDB keeps the residue
information, so the Lewis structure and the charge are the same); MOPAC must keep the PDB atom order, which is
checked against ATOM_X of the .aux file.  The energy is the FINAL HEAT OF FORMATION of the .out file (1e-5
kcal/mol, so the differences carry ~1e-3 kcal/(mol A) of rounding with the default step).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def aux_block(text: str, name: str) -> list[float]:
    m = re.search(re.escape(name) + r"\[\d+\]=\s*(.*?)(?:\n\s*[A-Z_]+[:=\[]|\Z)", text, re.S)
    if not m:
        raise ValueError(f"{name} not in the .aux file")
    return [float(v.replace("D", "E")) for v in m.group(1).split()]


def out_heat(text: str) -> float:
    """FINAL HEAT OF FORMATION of the .out file (1e-5 kcal/mol; the .aux file has 6 significant digits only)"""
    m = re.findall(r"FINAL HEAT OF FORMATION\s*=\s*([-+0-9.]+)", text)
    if not m:
        raise ValueError("no FINAL HEAT OF FORMATION in the .out file")
    return float(m[-1])


def atom_lines(pdb_text: str) -> list[int]:
    return [i for i, ln in enumerate(pdb_text.splitlines()) if ln.startswith(("ATOM", "HETATM"))]


def displaced(pdb_text: str, atom: int, axis: int, delta: float) -> str:
    lines = pdb_text.splitlines()
    i = atom_lines(pdb_text)[atom]
    ln = lines[i]
    cols = [(30, 38), (38, 46), (46, 54)]
    a, b = cols[axis]
    lines[i] = ln[:a] + f"{float(ln[a:b]) + delta:8.3f}" + ln[b:]
    return "\n".join(lines) + "\n"


def heat(mopac: str, pdb_text: str, keywords: str, work: Path, tag: str, threads: int) -> float:
    d = work / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "geom.pdb").write_text(pdb_text)
    (d / "job.mop").write_text(f'{keywords} GEO_DAT="geom.pdb"\n{tag}\n\n')
    env = dict(os.environ, MOPAC_NOGPU="1", OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    subprocess.run([mopac, "job.mop"], cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False)
    return out_heat((d / "job.out").read_text(errors="ignore"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mopac", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--keywords", required=True, help="keywords of the energy runs (no GRADIENTS needed)")
    ap.add_argument("--aux", required=True, help=".aux of a 1SCF GRADIENTS run at this geometry (gradient a)")
    ap.add_argument("--compare-aux", default="", help=".aux with a second gradient at the same geometry (b)")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--step", type=float, default=0.005, help="Angstrom (the PDB has 3 decimals)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--work", default="fd_work")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    pdb_text = Path(a.pdb).read_text()
    ta = Path(a.aux).read_text(errors="ignore")
    x = aux_block(ta, "ATOM_X:ANGSTROMS")
    ga = aux_block(ta, "GRADIENTS:KCAL/MOL/ANGSTROM")
    gb = aux_block(Path(a.compare_aux).read_text(errors="ignore"), "GRADIENTS:KCAL/MOL/ANGSTROM") if a.compare_aux else None
    lines = pdb_text.splitlines()
    idx = atom_lines(pdb_text)
    if len(idx) * 3 != len(x):
        sys.exit("the .aux and the PDB have different atom counts")
    for k, i in enumerate(idx):
        p = [float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])]
        if max(abs(p[j] - x[3 * k + j]) for j in range(3)) > 1e-3:
            sys.exit(f"MOPAC did not keep the PDB atom order (atom {k + 1})")
    key = (lambda c: -abs(ga[c] - gb[c])) if gb else (lambda c: -abs(ga[c]))
    coords = sorted(range(len(ga)), key=key)[:a.n]
    work = Path(a.work)
    rows = []
    for c in coords:
        atom, axis = divmod(c, 3)
        ep = heat(a.mopac, displaced(pdb_text, atom, axis, +a.step), a.keywords, work, f"c{c}_p", a.threads)
        em = heat(a.mopac, displaced(pdb_text, atom, axis, -a.step), a.keywords, work, f"c{c}_m", a.threads)
        fd = (ep - em) / (2 * a.step)
        ln = lines[idx[atom]]
        r = {"coord": c, "atom": atom + 1, "name": f"{ln[17:20].strip()}{ln[22:26].strip()} {ln[12:16].strip()}",
             "axis": "xyz"[axis], "fd": fd, a.label_a: ga[c]}
        if gb:
            r[a.label_b] = gb[c]
        rows.append(r)
        print(f"{r['name']:>14s} {r['axis']}  FD {fd:9.4f}   {a.label_a} {ga[c]:9.4f}" +
              (f"   {a.label_b} {gb[c]:9.4f}" if gb else ""), flush=True)
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

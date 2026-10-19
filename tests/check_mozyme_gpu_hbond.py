#!/usr/bin/env python3
"""MOZYME GPU: PM7 H-bond correction (with and without HB-SMOOTH) and dispersion kernels against the CPU loops.

Usage:
  check_mozyme_gpu_hbond.py <mopac> [--work-dir DIR] [--skip-dna]

Runs PM7 MOZYME 1SCF GRADIENTS with MOPAC_GPU_DISP_CHECK=1: every H-bond and dispersion evaluation is done on the
device and again by the CPU loop of the same run, and the .out gets one line per evaluation:
  [MOZYME GPU hbond] check E_gpu= E_cpu= dE= N_gpu= N_cpu= max_abs_dgrad= ms=
  [MOZYME GPU disp] check E_gpu= E_cpu= dE= max_abs_dgrad= ms=
The H-bond kernel repeats the CPU arithmetic (finite-difference gradient, only the summation order differs):
|dE| <= 1e-6 kcal/mol, max |dgrad| <= 1e-5 kcal/(mol A), same number of H-bonds.  The dispersion kernel has an
analytic gradient, the CPU a numerical one: |dE| <= 1e-6, max |dgrad| <= 1e-3.  A run without check lines fails
(the device path did not run).

Systems: chignolin and crambin (S: d orbitals), DNA dodecamer 1BNA neutral (P: d orbitals) and the 1BNA anion
in COSMO water (EPS=78.4 SWIG) at the geometry where an optimization without HB-SMOOTH stuck on an H-bond kink
(tests/data/mozyme_dna_cosmo/1BNA_anion_water_kink.arc); each with the standard correction and with HB-SMOOTH.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
BASE_KEYS = "PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK 1SCF GRADIENTS"
NUM = r"\s*([-+0-9.EeDd]+)"
HB_RE = re.compile(r"\[MOZYME GPU hbond\] check E_gpu=" + NUM + " E_cpu=" + NUM + " dE=" + NUM +
                   r" N_gpu=\s*(\d+) N_cpu=\s*(\d+) max_abs_dgrad=" + NUM)
DISP_RE = re.compile(r"\[MOZYME GPU disp\] check E_gpu=" + NUM + " E_cpu=" + NUM + " dE=" + NUM +
                     " max_abs_dgrad=" + NUM)
HEAT_RE = re.compile(r"FINAL HEAT OF FORMATION =\s*([-0-9.]+)")


def deck(name: str, keywords: str, geometry: Path) -> str:
    """Input deck: a PDB through GEO_DAT, or the geometry lines of a MOPAC deck/.arc (after its 3 header lines)."""
    if geometry.suffix == ".pdb":
        return f'{keywords} GEO_DAT="{geometry}"\n{name}\n\n'
    lines = geometry.read_text().splitlines()
    return f"{keywords}\n{name}\n\n" + "\n".join(lines[3:]) + "\n"


def run(mopac: Path, work: Path, name: str, keywords: str, geometry: Path) -> dict:
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "j.mop").write_text(deck(name, keywords, geometry))
    env = dict(os.environ, MOPAC_GPU_DISP_CHECK="1")
    t0 = time.perf_counter()
    subprocess.run([str(mopac), "j.mop"], cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    out = d / "j.out"
    text = out.read_text(errors="ignore") if out.exists() else ""
    f = lambda s: float(s.replace("D", "E").replace("d", "e"))  # noqa: E731
    heat = HEAT_RE.search(text)
    return {"wall": time.perf_counter() - t0, "heat": heat.group(1) if heat else None,
            "hb": [(f(m[2]), int(m[3]), int(m[4]), f(m[5])) for m in HB_RE.findall(text)],
            "disp": [(f(m[2]), f(m[3])) for m in DISP_RE.findall(text)]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mopac", type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path("mozyme_gpu_hbond"))
    ap.add_argument("--skip-dna", action="store_true", help="only chignolin and crambin")
    a = ap.parse_args()
    mopac = a.mopac.resolve()
    work = a.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    ref, dna = DATA / "mozyme_gpu_reference", DATA / "mozyme_dna_cosmo"
    systems = [("chignolin", "", ref / "1UAO_opt.pdb"), ("crambin", "", ref / "1CRN_opt.pdb")]
    if not a.skip_dna:
        systems += [("dna_neutral", "", dna / "1BNA_neutral.pdb"),
                    ("dna_anion_water", " CHARGE=-22 EPS=78.4 SWIG", dna / "1BNA_anion_water_kink.arc")]
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        if not cond:
            failures.append(msg)

    for system, extra, geometry in systems:
        for smooth in ("", " HB-SMOOTH"):
            name = system + ("_hbsmooth" if smooth else "")
            r = run(mopac, work, name, BASE_KEYS + extra + smooth, geometry)
            print(f"[{name}] {r['wall']:.1f} s, heat {r['heat']}, {len(r['hb'])} H-bond and {len(r['disp'])} "
                  f"dispersion checks", flush=True)
            if r["hb"]:
                de = max(abs(x[0]) for x in r["hb"])
                dg = max(x[3] for x in r["hb"])
                n_ok = all(x[1] == x[2] for x in r["hb"])
                check(de <= 1e-6 and dg <= 1e-5 and n_ok,
                      f"{name}: H-bond GPU - CPU max |dE| {de:.2e} kcal/mol, max |dgrad| {dg:.2e} kcal/(mol A), "
                      f"H-bonds {r['hb'][-1][1]} GPU / {r['hb'][-1][2]} CPU")
            else:
                check(False, f"{name}: no [MOZYME GPU hbond] check line (device H-bond path did not run)")
            if r["disp"]:
                de = max(abs(x[0]) for x in r["disp"])
                dg = max(x[1] for x in r["disp"])
                check(de <= 1e-6 and dg <= 1e-3,
                      f"{name}: dispersion GPU - CPU max |dE| {de:.2e} kcal/mol, max |dgrad| {dg:.2e} kcal/(mol A)")
            else:
                check(False, f"{name}: no [MOZYME GPU disp] check line (device dispersion path did not run)")
            check(r["heat"] is not None, f"{name}: SCF finished (heat {r['heat']})")

    print("all checks passed" if not failures else f"{len(failures)} check(s) failed", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Molecular Orbital PACkage (MOPAC)
# Copyright 2025-2026 Juvenal Yosa, PhD (juvenal.yosa@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Solids (Tv, conventional SCF) on the GPU against the CPU.

  1 check   cells of 144-192 atoms, 1SCF GRADIENTS with MOPAC_GPU_HCORE_CHECK, MOPAC_GPU_FOCK_CHECK and
            MOPAC_GPU_GRAD_CHECK: the device hcore / Fock / gradient against the CPU loops of the same run
  2 medium  576-atom quartz and rutile, 512-atom Si: GPU heat and time against stored Linux CPU references
  3 force   diamond (64 atoms) FORCE: GPU frequencies and ZPE against the same binary with MOPAC_NOGPU=1
  4 drc     diamond DRC TEMPERATURE=300 CYCLES=40: GPU energy conservation, GPU potential against the CPU one

    python3 tests/check_materials_gpu.py build/mopac --work-dir /scratch/$USER/materials [--skip-force] ...

Prints PASS / FAIL lines and [materials] result lines; exit 1 when a check fails.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "materials"

# Linux CPU (4 vCPU EPYC, GPU=OFF build, 2026-10-19) heats of formation, kcal/mol.  Si differs between CPU builds
# by ~0.02 kcal/mol (equidistant nearest images: solrot picks one by the last bit of the distance).
MEDIUM_REF = {"quartz": -40393.65183, "silicon": -3991.12205, "rutile": -38476.65650}
MEDIUM_CPU_TIME = {"quartz": 239.9, "silicon": 808.9, "rutile": 1223.6}

failures = []


def result(ok, text):
    print(("PASS " if ok else "FAIL ") + text, flush=True)
    if not ok:
        failures.append(text)


def run(mopac, work, name, keywords, deck, env=None, timeout=7200):
    """Write deck (first line replaced by keywords) to work/name.mop, run it, return (output text, seconds)."""
    work.mkdir(parents=True, exist_ok=True)
    lines = Path(deck).read_text().splitlines()
    lines[0] = keywords
    (work / f"{name}.mop").write_text("\n".join(lines) + "\n")
    full_env = dict(os.environ)
    full_env.update(env or {})
    t0 = time.time()
    with open(work / f"{name}.stdout", "w") as fh:
        try:
            subprocess.run([mopac, f"{name}.mop"], cwd=work, env=full_env, stdout=fh, stderr=subprocess.STDOUT,
                           timeout=timeout)
        except subprocess.TimeoutExpired:
            fh.write(f"timeout after {timeout} s\n")
    wall = time.time() - t0
    out = work / f"{name}.out"
    return (out.read_text(errors="ignore") if out.exists() else ""), wall


def heat(text):
    m = re.findall(r"FINAL HEAT OF FORMATION =\s+(-?\d+\.\d+)", text)
    return float(m[-1]) if m else None


def floats_after(text, key):
    return [float(x) for x in re.findall(re.escape(key) + r"\s*(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)", text)]


def check_step(mopac, work):
    env = {"MOPAC_GPU_HCORE_CHECK": "1", "MOPAC_GPU_FOCK_CHECK": "1", "MOPAC_GPU_GRAD_CHECK": "1"}
    for name in ("si144", "quartz162", "rutile162", "urea_cluster"):
        text, wall = run(mopac, work / "check" / name, name, "PM7 1SCF GRADIENTS GEO-OK", DATA / "grad" / f"{name}.mop",
                         env)
        hc = re.findall(r"\[GPU hcore\] check .*?enuc=\s*(.*?)\s*\(CPU", text)
        fock = floats_after(text, "check max|F_gpu - F_cpu|=")
        grad = floats_after(text, "max_abs_diff=")
        print(f"[materials] check {name}: {wall:.1f} s, heat {heat(text)}, hcore h w wk e1b/e2a enuc "
              f"{hc[-1] if hc else None}, Fock {max(fock) if fock else None} eV, gradient "
              f"{max(grad) if grad else None} kcal/mol/A", flush=True)
        # wk may differ where two images are equally near (solrot keeps one): not tested
        h_ok = bool(hc) and all(max(float(v) for i, v in enumerate(c.split()) if i != 2) <= 1e-8 for c in hc)
        result(h_ok, f"check {name}: device hcore h, w, e1b/e2a, enuc within 1e-8 of the CPU")
        result(bool(fock) and max(fock) <= 1e-7, f"check {name}: device Fock within 1e-7 eV of the CPU")
        result(bool(grad) and max(grad) <= 1e-5, f"check {name}: device gradient within 1e-5 kcal/mol/A of the CPU")


def medium_step(mopac, work, names):
    for name in names:
        text, wall = run(mopac, work / "medium" / name, name, "PM7 1SCF GRADIENTS GEO-OK TIMES",
                         DATA / "medium" / f"{name}.mop", {"MOPAC_GPU_PROFILE": "1"})
        h = heat(text)
        iters = text.count("TIME FOR ITERATION")
        print(f"[materials] medium {name}: GPU {wall:.1f} s ({iters} iterations), heat {h}, CPU reference "
              f"{MEDIUM_REF[name]} in {MEDIUM_CPU_TIME[name]} s (4 vCPU)", flush=True)
        result(h is not None and abs(h - MEDIUM_REF[name]) <= 0.03,
               f"medium {name}: GPU heat - CPU reference {(h or float('nan')) - MEDIUM_REF[name]:+.5f} kcal/mol")


def force_step(mopac, work):
    deck = DATA / "small" / "diamond.mop"
    res = {}
    for mode, env in (("gpu", {}), ("cpu", {"MOPAC_NOGPU": "1"})):
        text, wall = run(mopac, work / "force" / mode, "diamond", "PM7 FORCE GEO-OK", deck, env)
        freq = floats_after(text, "FREQUENCY  ")
        zpe = floats_after(text, "ZERO POINT ENERGY")
        res[mode] = (freq, zpe[-1] if zpe else None, wall)
        print(f"[materials] force diamond {mode}: {wall:.1f} s, {len(freq)} frequencies "
              f"{freq[:1]}..{freq[-1:]} cm-1, ZPE {res[mode][1]} kcal/mol", flush=True)
    (fg, zg, tg), (fc, zc, tc) = res["gpu"], res["cpu"]
    ok = bool(fg) and len(fg) == len(fc)
    dmax = max(abs(a - b) for a, b in zip(fg, fc)) if ok else float("nan")
    result(ok and dmax <= 1.0, f"force diamond: max |freq GPU - CPU| {dmax:.3f} cm-1 ({len(fg)} / {len(fc)} modes)")
    result(zg is not None and zc is not None and abs(zg - zc) <= 0.01,
           f"force diamond: ZPE GPU - CPU {(zg or float('nan')) - (zc or float('nan')):+.4f} kcal/mol, "
           f"time GPU {tg:.0f} s CPU {tc:.0f} s")


def drc_points(text):
    """(time fs, potential, total, error) of every DRC line."""
    pts = []
    for m in re.finditer(r"^\s+(\d+\.\d+)\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+%",
                         text, re.M):
        pts.append((float(m.group(1)), float(m.group(2)), float(m.group(4)), float(m.group(5))))
    return pts


def drc_step(mopac, work):
    deck = DATA / "small" / "diamond.mop"
    keys = "PM7 DRC TEMPERATURE=300 SEED=7 CYCLES=40 GEO-OK"
    res = {}
    for mode, env in (("gpu", {}), ("cpu", {"MOPAC_NOGPU": "1"})):
        text, wall = run(mopac, work / "drc" / mode, "diamond", keys, deck, env)
        res[mode] = (drc_points(text), wall)
        pts = res[mode][0]
        drift = max(abs(p[3]) for p in pts) if pts else float("nan")
        print(f"[materials] drc diamond {mode}: {wall:.1f} s, {len(pts)} points, max |total energy error| "
              f"{drift:.5f} kcal/mol", flush=True)
    pg, pc = res["gpu"][0], res["cpu"][0]
    drift = max(abs(p[3]) for p in pg) if pg else float("nan")
    result(len(pg) >= 30 and drift <= 0.05, f"drc diamond: GPU {len(pg)} points, total energy kept within "
           f"{drift:.5f} kcal/mol")
    n = min(len(pg), len(pc))
    dpot = max(abs(a[1] - b[1]) for a, b in zip(pg[:n], pc[:n])) if n else float("nan")
    result(n >= 30 and dpot <= 0.01, f"drc diamond: max |potential GPU - CPU| {dpot:.5f} kcal/mol over {n} points")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mopac")
    ap.add_argument("--work-dir", default="materials_gpu")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--medium", default="quartz,silicon,rutile", help="medium cells to run ('' for none)")
    ap.add_argument("--skip-force", action="store_true")
    ap.add_argument("--skip-drc", action="store_true")
    a = ap.parse_args()
    mopac = str(Path(a.mopac).resolve())
    work = Path(a.work_dir).resolve()
    if not a.skip_check:
        check_step(mopac, work)
    if a.medium:
        medium_step(mopac, work, [n for n in a.medium.split(",") if n])
    if not a.skip_force:
        force_step(mopac, work)
    if not a.skip_drc:
        drc_step(mopac, work)
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

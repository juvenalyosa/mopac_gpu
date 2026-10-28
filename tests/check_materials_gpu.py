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
  3 force   diamond (64 atoms) FORCE SCFCRT=1.D-10: GPU frequencies and ZPE against the same binary with MOPAC_NOGPU=1
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
# decks without Tv: only the pair gradient is a device path there
MOLECULES = {"urea_cluster"}
# MOPAC uses the GPU above 100 atoms only (diamond: 64 + 3 Tv); MOPAC_GPU_DEBUG prints "[GPU gradient] pairs="
# and "[GPU Fock] pairs=" for every device call, which shows that the GPU did the work
GPU_ENV = {"MOPAC_FORCEGPU": "1", "MOPAC_GPU_DEBUG": "1"}
# FORCE of a solid with the default SCF criterion (1e-7 x numat/20 for FORCE) is not converged: diamond ZPE
# 269.07 (default), 268.85 (SCFCRT=1e-8), 268.82 (1e-10), 268.80 (1e-12); highest frequency 1262.4 -> 1261.3 cm-1
# (CPU, 2026-10-22).  GPU and CPU are compared at SCFCRT=1e-10, where the remaining spread is <= 0.3 cm-1.
FORCE_KEYS = "PM7 FORCE GEO-OK SCFCRT=1.D-10"

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


def device_calls(text):
    g = text.count("[GPU gradient] pairs=")
    f = text.count("[GPU Fock] pairs=")
    fb = text.count("[GPU gradient] fallback_cpu") + text.count("[GPU Fock] device failure")
    return f"device gradient calls {g}, device Fock builds {f}, CPU fallbacks {fb}"


def gpu_used(text, label):
    g = text.count("[GPU gradient] pairs=")
    f = text.count("[GPU Fock] pairs=")
    fb = text.count("[GPU gradient] fallback_cpu") + text.count("[GPU Fock] device failure")
    result(g > 0 and f > 0 and fb == 0, f"{label}: ran on the GPU ({g} device gradients, {f} device Fock builds, "
           f"{fb} CPU fallbacks)")


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
        result(bool(grad) and max(grad) <= 1e-5, f"check {name}: device gradient within 1e-5 kcal/mol/A of the CPU")
        if name in MOLECULES:
            continue   # no Tv: the device hcore / Fock of solids do not apply
        # wk may differ where two images are equally near (solrot keeps one): not tested
        h_ok = bool(hc) and all(max(float(v) for i, v in enumerate(c.split()) if i != 2) <= 1e-8 for c in hc)
        result(h_ok, f"check {name}: device hcore h, w, e1b/e2a, enuc within 1e-8 of the CPU")
        result(bool(fock) and max(fock) <= 1e-7, f"check {name}: device Fock within 1e-7 eV of the CPU")
    # Fock check alone: the device hcore leaves w / wk on the device, and the CPU Fock build of the
    # check must fetch them (mopac_cuda_conv_w_sync_host)
    for name in ("si144", "rutile162"):
        text, wall = run(mopac, work / "check" / f"{name}_wsync", name, "PM7 1SCF GRADIENTS GEO-OK",
                         DATA / "grad" / f"{name}.mop", {"MOPAC_GPU_FOCK_CHECK": "1", "MOPAC_GPU_PROFILE": "1"})
        fock = floats_after(text, "check max|F_gpu - F_cpu|=")
        synced = "conv_w_sync_host" in text
        print(f"[materials] check {name} w on the device: {wall:.1f} s, heat {heat(text)}, Fock "
              f"{max(fock) if fock else None} eV, host w fetched {synced}", flush=True)
        result(bool(fock) and max(fock) <= 1e-7 and synced,
               f"check {name}: w / wk fetched from the device, CPU Fock within 1e-7 eV of the device one")
    # A molecule (conventional SCF, no Tv): the device Fock build against fock2 on the CPU, then the
    # job with the device Fock build and with MOPAC_GPU_CONV_FOCK=0 (time and heat)
    mol = DATA / "molecules" / "crambin.mop"
    kw = "PM7 1SCF GRADIENTS GEO-OK CHARGE=-1"
    text, wall = run(mopac, work / "check" / "crambin_fockcheck", "crambin", kw, mol, {"MOPAC_GPU_FOCK_CHECK": "1"})
    fock = floats_after(text, "check max|F_gpu - F_cpu|=")
    print(f"[materials] check crambin Fock: {wall:.1f} s, heat {heat(text)}, Fock {max(fock) if fock else None} eV",
          flush=True)
    result(bool(fock) and max(fock) <= 1e-7, "check crambin: device Fock within 1e-7 eV of the CPU")
    runs = {}
    for tag, env in (("gpu_fock", {}), ("cpu_fock", {"MOPAC_GPU_CONV_FOCK": "0"})):
        env = dict(env, MOPAC_GPU_PROFILE="1", MOPAC_GPU_DEBUG="1")
        text, wall = run(mopac, work / "check" / f"crambin_{tag}", "crambin", kw + " TIMES", mol, env)
        runs[tag] = (heat(text), wall, text.count("[GPU Fock] pairs="))
        print(f"[materials] crambin {tag}: {wall:.1f} s, heat {runs[tag][0]}, device Fock builds {runs[tag][2]}",
              flush=True)
    (hg, tg, ng), (hc, tc, _) = runs["gpu_fock"], runs["cpu_fock"]
    dh = "n/a" if hg is None or hc is None else f"{hg - hc:+.5f}"
    result(hg is not None and hc is not None and abs(hg - hc) <= 1e-3 and ng > 0,
           f"crambin: device Fock heat - CPU Fock heat {dh} kcal/mol, time {tg:.1f} s vs {tc:.1f} s")
    # UHF gradient on the device (dhc's uhf energy: pa and pb): an open-shell molecule (neutral
    # crambin, a doublet) and a solid, each against dcart on the CPU in the same run
    for name, kw_u, deck in (("crambin", "PM7 UHF 1SCF GRADIENTS GEO-OK CHARGE=0", mol),
                             ("si144", "PM7 UHF 1SCF GRADIENTS GEO-OK", DATA / "grad" / "si144.mop")):
        text, wall = run(mopac, work / "check" / f"{name}_uhf", name, kw_u, deck,
                         {"MOPAC_GPU_GRAD_CHECK": "1", "MOPAC_GPU_DEBUG": "1"})
        grad = floats_after(text, "max_abs_diff=")
        print(f"[materials] check {name} UHF: {wall:.1f} s, heat {heat(text)}, gradient "
              f"{max(grad) if grad else None} kcal/mol/A, device Fock builds {text.count('[GPU Fock] pairs=')}",
              flush=True)
        result(bool(grad) and max(grad) <= 1e-5,
               f"check {name} UHF: device gradient within 1e-5 kcal/mol/A of the CPU")
    # COSMO in the SCF (addfck's dense algebra on the device): crambin in water, the device terms
    # against the CPU ones, then the job with and without the device COSMO terms
    kw_c = "PM7 1SCF GRADIENTS GEO-OK CHARGE=-1 EPS=78.4"
    text, wall = run(mopac, work / "check" / "crambin_cosmocheck", "crambin", kw_c, mol,
                     {"MOPAC_GPU_COSMO_CHECK": "1"})
    m = re.findall(r"\[GPU COSMO\] check nps=\s*(\d+) max\|phi_gpu - phi_cpu\|=\s*(\S+) "
                   r"max\|qsc_gpu - qsc_cpu\|=\s*(\S+) max\|qsc_cpu\|=\s*(\S+)", text)
    rel = max(float(q) / max(float(s), 1e-30) for _, _, q, s in m) if m else None
    print(f"[materials] check crambin COSMO: {wall:.1f} s, heat {heat(text)}, nps {m[0][0] if m else None}, "
          f"max relative |qsc GPU - CPU| {rel}", flush=True)
    result(rel is not None and rel <= 1e-9, "check crambin COSMO: device COSMO charges within 1e-9 (relative) of the CPU")
    runs = {}
    for tag, env in (("gpu_cosmo", {}), ("cpu_cosmo", {"MOPAC_GPU_COSMO": "0"})):
        text, wall = run(mopac, work / "check" / f"crambin_{tag}", "crambin", kw_c + " TIMES", mol,
                         dict(env, MOPAC_GPU_PROFILE="1"))
        runs[tag] = (heat(text), wall)
        print(f"[materials] crambin {tag}: {wall:.1f} s, heat {runs[tag][0]}", flush=True)
    (hg, tg), (hc, tc) = runs["gpu_cosmo"], runs["cpu_cosmo"]
    dh = "n/a" if hg is None or hc is None else f"{hg - hc:+.5f}"
    result(hg is not None and hc is not None and abs(hg - hc) <= 1e-3,
           f"crambin COSMO: device COSMO heat - CPU COSMO heat {dh} kcal/mol, time {tg:.1f} s vs {tc:.1f} s")
    # PRECISE: central differences (dcart's force), the other probe layout of the d-pair kernel
    for name in ("si144", "rutile162"):
        text, wall = run(mopac, work / "check" / f"{name}_precise", name, "PM7 1SCF GRADIENTS GEO-OK PRECISE",
                         DATA / "grad" / f"{name}.mop", {"MOPAC_GPU_GRAD_CHECK": "1"})
        grad = floats_after(text, "max_abs_diff=")
        print(f"[materials] check {name} PRECISE: {wall:.1f} s, heat {heat(text)}, gradient "
              f"{max(grad) if grad else None} kcal/mol/A", flush=True)
        result(bool(grad) and max(grad) <= 1e-5,
               f"check {name} PRECISE: device gradient within 1e-5 kcal/mol/A of the CPU")


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
    for mode, env in (("gpu", GPU_ENV), ("cpu", {"MOPAC_NOGPU": "1"})):
        text, wall = run(mopac, work / "force" / mode, "diamond", FORCE_KEYS, deck, env)
        freq = floats_after(text, "FREQUENCY  ")
        zpe = floats_after(text, "ZERO POINT ENERGY")
        res[mode] = (freq, zpe[-1] if zpe else None, wall)
        print(f"[materials] force diamond {mode}: {wall:.1f} s, {len(freq)} frequencies "
              f"{freq[:1]}..{freq[-1:]} cm-1, ZPE {res[mode][1]} kcal/mol, {device_calls(text)}", flush=True)
        if mode == "gpu":
            gpu_used(text, "force diamond")
    (fg, zg, tg), (fc, zc, tc) = res["gpu"], res["cpu"]
    ok = bool(fg) and len(fg) == len(fc)
    dmax = max(abs(a - b) for a, b in zip(fg, fc)) if ok else float("nan")
    result(ok and dmax <= 0.5, f"force diamond: max |freq GPU - CPU| {dmax:.3f} cm-1 ({len(fg)} / {len(fc)} modes)")
    result(zg is not None and zc is not None and abs(zg - zc) <= 0.03,
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
    for mode, env in (("gpu", GPU_ENV), ("cpu", {"MOPAC_NOGPU": "1"})):
        text, wall = run(mopac, work / "drc" / mode, "diamond", keys, deck, env)
        res[mode] = (drc_points(text), wall)
        pts = res[mode][0]
        drift = max(abs(p[3]) for p in pts) if pts else float("nan")
        print(f"[materials] drc diamond {mode}: {wall:.1f} s, {len(pts)} points, max |total energy error| "
              f"{drift:.5f} kcal/mol, {device_calls(text)}", flush=True)
        if mode == "gpu":
            gpu_used(text, "drc diamond")
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
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unknown"
    print(f"[materials] GPU: {gpu or 'unknown'}", flush=True)
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

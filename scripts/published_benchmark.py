#!/usr/bin/env python3
"""MOZYME CPU vs GPU single points on the systems of the published MOPAC timing table.

The openmopac.net page "Use of MKL and Multi-Threading to reduce computation time"
(https://openmopac.net/Manual/Reducing_computation_time.html) lists 1SCF times of the
conventional (matrix-diagonalisation, non-MOZYME) MOPAC2016 solver on a
2 x 2.93 GHz 6-core Intel Xeon Mac Pro (2010): single thread, with MKL, and with MKL on
12 threads.  This script runs the same proteins (hydrogenated PDB structures; the page
gives no PDB code for bacteriorhodopsin, 1C3W is used) with MOZYME on the CPU
(MOPAC_NOGPU=1, one core) and on the GPU (defaults) with this binary, and prints one
table with the published numbers next to the measured ones.  The published times are
for a different solver and a different machine: they show what a user of MOPAC2016
would have waited for a single point of the same system, not a same-hardware speedup.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HEAT_RE = re.compile(r"FINAL HEAT OF FORMATION\s*=\s*([-+0-9.EeDd]+)")
JOB_TIME_RE = re.compile(r"TOTAL JOB TIME:\s*([0-9.]+)\s*SECONDS")
STATUS_RE = re.compile(r"\[MOZYME GPU SCF\] status=(\w+)")
REF_RE = re.compile(r'"([^"]+)"')

# (label, deck stem, published atoms, MOPAC2016 1 thread s, with MKL s, MKL + 12 threads s)
CASES = [
    ("Crambin (1CRN)", "protein_crambin_1crn", 642, 468.0, 114.0, 12.0),
    ("1G6X", "protein_1g6x", 1455, 8612.0, 1240.0, 142.0),
    ("Antifreeze protein (1EZG)", "protein_antifreeze_1ezg", 2064, 22959.0, 2118.0, 300.0),
    ("Barnase (1RNB)", "protein_barnase_1rnb", 2066, 34372.0, 4108.0, 411.0),
    ("Bacteriorhodopsin (1C3W)", "protein_bacteriorhodopsin_1c3w", 3352, 141773.0, 11192.0, 1394.0),
    ("Ubiquitin (1UBQ)", "protein_ubiquitin_1ubq", None, None, None, None),
    ("Adenylate kinase apo (1AKE)", "protein_adenylate_kinase_1ake_apo", None, None, None, None),
]

CPU_ENV_OFF = ("MOPAC_NOGPU", "MOZYME_GPU_OFF", "MOPAC_FORCEGPU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mopac", type=Path)
    parser.add_argument("--inputs-dir", type=Path, default=Path("benchmarks/publication_inputs/mop"))
    parser.add_argument("--out-dir", type=Path, default=Path("published_benchmark"))
    parser.add_argument("--modes", default="cpu,gpu", help="comma list of cpu,gpu")
    parser.add_argument("--only", action="append", default=[], help="run only cases whose stem contains this text")
    parser.add_argument("--skip-cpu-above", type=int, default=0,
                        help="skip the CPU run for decks with more atoms than this (0 = never skip)")
    parser.add_argument("--timeout", type=float, default=7200.0)
    return parser.parse_args()


def count_atoms(pdb: Path) -> int:
    n = 0
    with pdb.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                n += 1
    return n


def stage(deck: Path, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(deck, run_dir / deck.name)
    for ref in REF_RE.findall(deck.read_text(errors="ignore")):
        src = deck.parent / Path(ref).name
        if src.exists():
            shutil.copy2(src, run_dir / src.name)
    return run_dir / deck.name


def run(mopac: Path, deck: Path, run_dir: Path, gpu: bool, timeout: float) -> dict:
    staged = stage(deck, run_dir)
    env = os.environ.copy()
    for key in CPU_ENV_OFF:
        env.pop(key, None)
    if not gpu:
        env["MOPAC_NOGPU"] = "1"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([str(mopac), staged.name], cwd=run_dir, capture_output=True, text=True,
                              env=env, timeout=timeout, check=False)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    wall = time.perf_counter() - t0
    out = run_dir / (staged.stem + ".out")
    text = out.read_text(errors="ignore") if out.exists() else ""
    heat = HEAT_RE.search(text)
    job = JOB_TIME_RE.search(text)
    statuses = STATUS_RE.findall(text)
    return {
        "rc": rc,
        "wall_s": wall,
        "job_s": float(job.group(1)) if job else None,
        "heat": float(heat.group(1).replace("D", "E")) if heat else None,
        "gpu_status": ",".join(sorted(set(statuses))) if statuses else "",
    }


def fmt(v, digits=1):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def main() -> int:
    args = parse_args()
    args.mopac = args.mopac.resolve()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, stem, pub_atoms, t1, tmkl, t12 in CASES:
        if args.only and not any(o in stem for o in args.only):
            continue
        deck = args.inputs_dir / f"{stem}.mop"
        if not deck.exists():
            print(f"[skip] {label}: {deck} not found", flush=True)
            continue
        refs = [deck.parent / Path(r).name for r in REF_RE.findall(deck.read_text(errors="ignore"))]
        atoms = count_atoms(refs[0]) if refs and refs[0].exists() else None
        row = {"label": label, "stem": stem, "atoms": atoms, "published_atoms": pub_atoms,
               "mopac2016_1thread_s": t1, "mopac2016_mkl_s": tmkl, "mopac2016_mkl_12threads_s": t12}
        for mode in modes:
            gpu = mode == "gpu"
            if not gpu and args.skip_cpu_above and atoms and atoms > args.skip_cpu_above:
                print(f"[skip] {label}: CPU run skipped ({atoms} atoms > {args.skip_cpu_above})", flush=True)
                continue
            print(f"[{mode}] {label} ({atoms} atoms) ...", flush=True)
            res = run(args.mopac, deck, args.out_dir / stem / mode, gpu, args.timeout)
            print(f"    wall={res['wall_s']:.1f}s rc={res['rc']} heat={res['heat']} {res['gpu_status']}", flush=True)
            for k, v in res.items():
                row[f"{mode}_{k}"] = v
        if row.get("cpu_heat") is not None and row.get("gpu_heat") is not None:
            row["dHf_gpu_minus_cpu"] = row["gpu_heat"] - row["cpu_heat"]
        rows.append(row)

    # table
    head = ("| System | Atoms (ours / published) | MOPAC2016 conventional 1SCF, 1 thread (s) | MKL, 12 threads (s) "
            "| MOZYME CPU, 1 core, this binary (s) | MOZYME GPU (s) | Heat CPU | Heat GPU | dHf |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [head, sep]
    for r in rows:
        lines.append("| {} | {} / {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["label"], r.get("atoms") or "-", r.get("published_atoms") or "-",
            fmt(r["mopac2016_1thread_s"], 0), fmt(r["mopac2016_mkl_12threads_s"], 0),
            fmt(r.get("cpu_wall_s")), fmt(r.get("gpu_wall_s")),
            fmt(r.get("cpu_heat"), 4), fmt(r.get("gpu_heat"), 4), fmt(r.get("dHf_gpu_minus_cpu"), 4)))
    table = "\n".join(lines)
    print()
    print(table)
    (args.out_dir / "published_benchmark.md").write_text(table + "\n")
    (args.out_dir / "published_benchmark.json").write_text(json.dumps(rows, indent=2))
    if rows:
        keys = sorted({k for r in rows for k in r})
        with (args.out_dir / "published_benchmark.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

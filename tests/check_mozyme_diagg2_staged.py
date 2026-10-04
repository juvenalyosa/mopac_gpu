#!/usr/bin/env python3
"""A/B check of the staged diagg2 kernel (MOPAC_MOZYME_DIAGG2_STAGED=1) against the default lock kernel.

Usage:
  check_mozyme_diagg2_staged.py <mopac> [--pdb-ids 1UAO,1CRN] [--repeats 2] [--forcets A10=1]
      [--work-dir DIR] [--keep-going]

Any CUDA GPU; needs internet once to download the PDB files (or put <ID>.pdb in <work-dir>/<ID>/).
For every system (hydrogenated with ADD-H, no optimization needed):
  1. 1SCF GRADIENTS AUX SCFCRT=0.000001 THRESH=1.D-15 (the FORCE/THERMO SCF), default kernel and staged
     kernel, --repeats times each, with MOPAC_MOZYME_SCF_TRACE=1;
  2. for the last system, FORCETS THERMO(298) LET OPT(region) (the warm-start regime of FORCE), both kernels.

Checks (the lock sweep already varies at rounding level from run to run, so the staged kernel is compared
with the spread of the default kernel, not bit for bit):
  - every SCF resident on the GPU and converged (no "FAILED", no fallback);
  - SCF iterations of the staged runs within +-3 of the default runs;
  - heat of formation within 1e-4 kcal/mol of the default runs;
  - gradient |difference| within max(5 x the default run-to-run spread, 0.005) kcal/(mol A);
  - FORCETS frequencies RMS difference <= 0.5 cm-1 and ZPE within 0.01 kcal/mol.
Prints the wall times (speed-up) and the diagg2 sweep statistics of the trace: max_chain (rotations done by
one block, the critical path) and lock_spins (waits on occupied-LMO locks).
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_mozyme_gpu_thermo as th  # noqa: E402
import mozyme_md_workflow as md  # noqa: E402

GRAD_RE = re.compile(r"GRADIENTS:KCAL/MOL/ANGSTROM\[\d+\]=\s*(.*?)(?=\n\s*[A-Z_]+[:=\[])", re.S)
ITER_RE = re.compile(r"\[MOZYME GPU SCF\] status=(\S+) reason=\S+ code=\S+ iterations=\s*(\d+)")
SWEEP_RE = re.compile(r"diagg2 staged=(\d) nij=(\d+) rotated=(\d+) max_chain=(\d+) lock_spins=(\d+)")
SCF_KEYS = "SCFCRT=0.000001 THRESH=1.D-15"


def run(mopac: Path, run_dir: Path, keywords: str, pdb: Path, staged: bool) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / pdb.name
    if not target.exists():
        target.write_bytes(pdb.read_bytes())
    (run_dir / "job.mop").write_text(f'{keywords} GEO_DAT="{pdb.name}"\nA/B diagg2\n\n')
    env = os.environ.copy()
    env.pop("MOPAC_NOGPU", None)
    env["MOPAC_MOZYME_SCF_TRACE"] = "1"
    if staged:
        env["MOPAC_MOZYME_DIAGG2_STAGED"] = "1"
    else:
        env.pop("MOPAC_MOZYME_DIAGG2_STAGED", None)
    t0 = time.perf_counter()
    proc = subprocess.run([str(mopac), "job.mop"], cwd=run_dir, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    out = run_dir / "job.out"
    text = proc.stdout + ("\n" + out.read_text(errors="ignore") if out.exists() else "")
    (run_dir / "stdout_and_out.txt").write_text(text)
    heats = md.HEAT_RE.findall(text)
    aux = run_dir / "job.aux"
    grad = []
    if aux.exists():
        m = GRAD_RE.search(aux.read_text(errors="ignore"))
        if m:
            grad = [float(x) for x in m.group(1).split()]
    statuses = ITER_RE.findall(text)
    sweeps = [tuple(int(x) for x in m) for m in SWEEP_RE.findall(text)]
    return {
        "wall": wall,
        "heat": float(heats[-1].replace("D", "E")) if heats else None,
        "grad": grad,
        "statuses": [s for s, _ in statuses],
        "iterations": [int(n) for _, n in statuses],
        "failed": "SCF CALCULATION FAILED" in text or "FAILED TO ACHIEVE SCF" in text,
        "gpu_error": text.count("[GPU ERROR]"),
        "sweeps": sweeps,
        "text": text,
    }


def gdiff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("inf")
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mopac", type=Path)
    ap.add_argument("--pdb-ids", default="1UAO,1CRN")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--forcets", default="A10=1", help='FORCETS region on the last system, "" to skip')
    ap.add_argument("--work-dir", type=Path, default=Path("mozyme_diagg2_staged"))
    a = ap.parse_args()
    mopac = a.mopac.resolve()
    work = a.work_dir.resolve()
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        if not cond:
            failures.append(msg)

    def sweep_summary(r: dict) -> str:
        if not r["sweeps"]:
            return "no diagg2 trace"
        chains = [s[3] for s in r["sweeps"]]
        spins = [s[4] for s in r["sweeps"]]
        return (f"{len(r['sweeps'])} sweeps, max_chain mean {statistics.mean(chains):.0f} max {max(chains)}, "
                f"lock_spins mean {statistics.mean(spins):.0f}")

    ids = [x.strip() for x in a.pdb_ids.split(",") if x.strip()]
    pdbs: dict[str, Path] = {}
    for pdb_id in ids:
        base = work / pdb_id
        pdbs[pdb_id] = md.prepare_pdb(mopac, th.get_pdb(pdb_id, base), base / "prepare")

    for pdb_id in ids:
        pdb = pdbs[pdb_id]
        keys = f"{md.BASE_KEYS} 1SCF GRADIENTS AUX LET {SCF_KEYS}"
        res: dict[str, list[dict]] = {"default": [], "staged": []}
        for rep in range(1, a.repeats + 1):
            for name in ("default", "staged"):
                r = run(mopac, work / pdb_id / f"scf_{name}_{rep}", keys, pdb, name == "staged")
                res[name].append(r)
                print(f"[{pdb_id}] 1SCF {name} #{rep}: {r['wall']:.2f} s, heat {r['heat']}, "
                      f"iterations {r['iterations']}, statuses {r['statuses']}, GPU errors {r['gpu_error']}; "
                      f"{sweep_summary(r)}", flush=True)
        allr = res["default"] + res["staged"]
        check(all(r["statuses"] and all(s == "success" for s in r["statuses"]) and not r["failed"]
                  and r["gpu_error"] == 0 for r in allr),
              f"{pdb_id}: every SCF resident and converged, no GPU error")
        d0, s0 = res["default"], res["staged"]
        it_d = [r["iterations"][-1] for r in d0 if r["iterations"]]
        it_s = [r["iterations"][-1] for r in s0 if r["iterations"]]
        if it_d and it_s:
            check(max(abs(x - y) for x in it_s for y in it_d) <= 3,
                  f"{pdb_id}: SCF iterations staged {it_s} vs default {it_d} (within 3)")
        h_d = [r["heat"] for r in d0 if r["heat"] is not None]
        h_s = [r["heat"] for r in s0 if r["heat"] is not None]
        if h_d and h_s:
            dh = max(abs(x - y) for x in h_s for y in h_d)
            check(dh <= 1e-4, f"{pdb_id}: heat staged - default max {dh:.2e} kcal/mol (<= 1e-4)")
        spread = max((gdiff(d0[0]["grad"], r["grad"]) for r in d0[1:]), default=0.0)
        dg = max(gdiff(d0[0]["grad"], r["grad"]) for r in s0)
        tol = max(5.0 * spread, 0.005)
        check(dg <= tol, f"{pdb_id}: gradient |staged - default| {dg:.4f} (default run-to-run {spread:.4f}, "
                         f"tolerance {tol:.4f} kcal/(mol A))")
        t_d = statistics.mean(r["wall"] for r in d0)
        t_s = statistics.mean(r["wall"] for r in s0)
        print(f"     {pdb_id}: 1SCF wall default {t_d:.2f} s, staged {t_s:.2f} s, speed-up {t_d / t_s:.2f}x "
              f"(includes start-up; the diagg2 share is in the trace)", flush=True)

    if a.forcets:
        pdb_id = ids[-1]
        sel, _, rad = a.forcets.partition("=")
        keys = f'{md.BASE_KEYS} FORCETS THERMO(298) LET OPT("{sel}"={rad})'
        fr = {}
        for name in ("default", "staged"):
            r = run(mopac, work / pdb_id / f"forcets_{name}", keys, pdbs[pdb_id], name == "staged")
            t = th.thermo(r["text"], 298.0)
            fr[name] = (r, t)
            ok = r["statuses"].count("success")
            print(f"[{pdb_id}] FORCETS {a.forcets} {name}: {r['wall']:.1f} s, {len(t['freqs'])} frequencies, "
                  f"ZPE {t.get('zpe')}, GPU SCF {ok}/{len(r['statuses'])} resident, GPU errors {r['gpu_error']}; "
                  f"{sweep_summary(r)}", flush=True)
        (rd, td), (rs, ts) = fr["default"], fr["staged"]
        check(all(s == "success" for s in rd["statuses"] + rs["statuses"]) and rd["statuses"] and rs["statuses"],
              f"{pdb_id} FORCETS: every SCF resident")
        fd, fs = sorted(td["freqs"]), sorted(ts["freqs"])
        if fd and len(fd) == len(fs):
            rms = (sum((x - y) ** 2 for x, y in zip(fd, fs)) / len(fd)) ** 0.5
            check(rms <= 0.5, f"{pdb_id} FORCETS: frequencies RMS staged - default {rms:.3f} cm-1 (<= 0.5)")
        else:
            check(False, f"{pdb_id} FORCETS: frequency counts {len(fd)} / {len(fs)}")
        if td.get("zpe") is not None and ts.get("zpe") is not None:
            check(abs(ts["zpe"] - td["zpe"]) <= 0.01,
                  f"{pdb_id} FORCETS: ZPE staged - default {ts['zpe'] - td['zpe']:+.4f} kcal/mol (<= 0.01)")
        print(f"     {pdb_id} FORCETS: default {rd['wall']:.1f} s, staged {rs['wall']:.1f} s, "
              f"speed-up {rd['wall'] / rs['wall']:.2f}x", flush=True)

    print(f"work dir: {work}")
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

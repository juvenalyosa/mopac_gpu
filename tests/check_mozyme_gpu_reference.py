#!/usr/bin/env python3
"""MOZYME GPU against stored CPU references (no CPU run needed): SCF, gradient, FORCE / FORCETS thermochemistry,
GPU memory and concurrent processes.

Usage:
  check_mozyme_gpu_reference.py <mopac> [--data tests/data/mozyme_gpu_reference] [--work-dir DIR]
      [--workers 4] [--concurrent 6] [--sequential-force] [--large-workers 0] [--skip-forcets]

References (tests/data/mozyme_gpu_reference/reference.json, CPU runs at the stored geometries):
  chignolin_scf    1SCF GRADIENTS, SCFCRT=1e-6 THRESH=1e-15       heat <= 1e-3 kcal/mol, |grad diff| <= 0.01
  crambin_scf      same, crambin (S atoms: d-orbital pair kernels) heat <= 2e-4, |grad diff| <= 0.01
  chignolin_force  FORCE THERMO(298), --workers GPU processes      ZPE 0.1, S 1, G 0.3, RMS(>=100 cm-1) 2 cm-1
  crambin_forcets  FORCETS THERMO(298) OPT("A46"=4)                 RMS 0.5 cm-1, ZPE 0.01 kcal/mol
plus every SCF resident on the GPU (no CPU fallback, no "[GPU ERROR]"), the GPU memory peak of a crambin
process (<= 8 GB; ~15 GB before cccffe37), and --concurrent crambin FORCETS processes at once with no fallback.
--large-workers N: the full crambin FORCE THERMO with N processes (timing only, ~45 min on an A100).

Environment variables pass through (e.g. MOPAC_MOZYME_DIAGG2_STAGED=1 to test the staged diagg2 kernel).
A JSON summary is written to <work-dir>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_mozyme_gpu_thermo as th  # noqa: E402
import mozyme_parallel_force as pf  # noqa: E402

GRAD_RE = re.compile(r"GRADIENTS:KCAL/MOL/ANGSTROM\[\d+\]=\s*(.*?)(?=\n\s*[A-Z_]+[:=\[])", re.S)
HEAT_RE = re.compile(r"FINAL HEAT OF FORMATION =\s*([-0-9.]+)")
STATUS_RE = re.compile(r"\[MOZYME GPU SCF\] status=(\S+) reason=")
TIME_RE = re.compile(r"TOTAL JOB TIME:\s*([0-9.]+)")


class GpuMemory:
    """Peak of nvidia-smi memory.used (MiB, whole device) while active."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = shutil.which("nvidia-smi") is not None
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        self._args = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if dev:
            self._args.insert(1, f"--id={dev}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(self._args, capture_output=True, text=True, timeout=10).stdout
                vals = [int(x) for x in out.split() if x.strip().isdigit()]
                if vals:
                    self.peak = max(self.peak, vals[0])
            except Exception:
                pass
            self._stop.wait(0.5)

    def __enter__(self) -> "GpuMemory":
        self.peak = 0
        if self.available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join()


def mopac_run(mopac: Path, run_dir: Path, keywords: str, geometry: Path, name: str = "job") -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(geometry, run_dir / geometry.name)
    (run_dir / f"{name}.mop").write_text(f'{keywords} GEO_DAT="{geometry.name}"\nreference check\n\n')
    env = os.environ.copy()
    env.pop("MOPAC_NOGPU", None)
    t0 = time.perf_counter()
    proc = subprocess.run([str(mopac), f"{name}.mop"], cwd=run_dir, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    (run_dir / f"{name}.stdout").write_text(proc.stdout)
    out = run_dir / f"{name}.out"
    text = out.read_text(errors="ignore") if out.exists() else ""
    return parse(text, proc.stdout, run_dir / f"{name}.aux", wall)


def parse(text: str, stdout: str, aux: Path | None, wall: float) -> dict:
    heats = HEAT_RE.findall(text)
    grad = []
    if aux is not None and aux.exists():
        m = GRAD_RE.search(aux.read_text(errors="ignore"))
        if m:
            grad = [float(x) for x in m.group(1).split()]
    statuses = STATUS_RE.findall(text + "\n" + stdout)
    return {"wall": wall, "heat": float(heats[-1]) if heats else None, "grad": grad,
            "resident": statuses.count("success"), "fallback": sum(1 for s in statuses if s != "success"),
            "gpu_errors": (text + stdout).count("[GPU ERROR]"),
            "failed": "SCF CALCULATION FAILED" in text, "text": text}


def gdiff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("inf")
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mopac", type=Path)
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data" / "mozyme_gpu_reference")
    ap.add_argument("--work-dir", type=Path, default=Path("mozyme_gpu_reference"))
    ap.add_argument("--workers", type=int, default=4, help="GPU processes of the chignolin FORCE")
    ap.add_argument("--sequential-force", action="store_true", help="also the chignolin FORCE in one process")
    ap.add_argument("--concurrent", type=int, default=6, help="crambin FORCETS processes at once (0: skip)")
    ap.add_argument("--large-workers", type=int, default=0, help="full crambin FORCE with N processes (timing)")
    ap.add_argument("--skip-forcets", action="store_true")
    a = ap.parse_args()
    mopac = a.mopac.resolve()
    work = a.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    ref = json.loads((a.data / "reference.json").read_text())
    failures: list[str] = []
    summary: dict = {"env": {k: v for k, v in os.environ.items() if k.startswith("MOPAC_")}}
    mem = GpuMemory()

    def check(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        if not cond:
            failures.append(msg)

    def resident_check(label: str, r: dict) -> None:
        check(r["resident"] > 0 and r["fallback"] == 0 and r["gpu_errors"] == 0 and not r["failed"],
              f"{label}: {r['resident']} SCF resident, {r['fallback']} fallback, {r['gpu_errors']} GPU errors")

    # 1-2: SCF + gradient
    for key, heat_tol in (("chignolin_scf", 1e-3), ("crambin_scf", 2e-4)):
        c = ref[key]
        with mem:
            r = mopac_run(mopac, work / key, c["keywords"], a.data / c["geometry"])
        dh = (r["heat"] - c["heat"]) if r["heat"] is not None else float("inf")
        dg = gdiff(c["gradient"], r["grad"])
        print(f"[{key}] {r['wall']:.2f} s (CPU {c['cpu_seconds']} s), heat {r['heat']} (CPU {c['heat']}), "
              f"GPU memory peak {mem.peak} MiB", flush=True)
        resident_check(key, r)
        check(abs(dh) <= heat_tol, f"{key}: heat GPU - CPU {dh:+.6f} kcal/mol (<= {heat_tol:g})")
        check(dg <= 0.01, f"{key}: gradient |GPU - CPU| {dg:.5f} kcal/(mol A) (<= 0.01)")
        if key == "crambin_scf" and mem.available:
            check(mem.peak <= 8192, f"{key}: GPU memory peak {mem.peak} MiB (<= 8192; ~15000 before the d-kernel fix)")
        summary[key] = {"wall": r["wall"], "heat_diff": dh, "grad_diff": dg, "mem_peak_mib": mem.peak,
                        "resident": r["resident"], "fallback": r["fallback"]}

    # 3: chignolin FORCE THERMO, parallel GPU processes (and optionally one process)
    c = ref["chignolin_force"]
    geo = a.data / c["geometry"]
    runs = []
    if a.sequential_force:
        with mem:
            r = mopac_run(mopac, work / "chignolin_force_seq", c["keywords"] + " FORCE", geo, "force")
        runs.append(("1 process", r, r["text"], mem.peak))
    if a.workers > 0:
        with mem:
            t0 = time.perf_counter()
            out, tim = pf.parallel_force(mopac, geo, c["keywords"], a.workers,
                                         work / f"chignolin_force_w{a.workers}")
            wall = time.perf_counter() - t0
        text = out.read_text(errors="ignore")
        agg = {"resident": 0, "fallback": 0, "gpu_errors": 0, "failed": False}
        for wdir in sorted((work / f"chignolin_force_w{a.workers}").glob("worker_*")):
            wr = parse((wdir / "force.out").read_text(errors="ignore") if (wdir / "force.out").exists() else "",
                       (wdir / "force.stdout").read_text(errors="ignore") if (wdir / "force.stdout").exists() else "",
                       None, 0.0)
            for k in ("resident", "fallback", "gpu_errors"):
                agg[k] += wr[k]
            agg["failed"] |= wr["failed"]
        agg["wall"] = wall
        runs.append((f"{a.workers} processes", agg, text, mem.peak))
    for label, r, text, peak in runs:
        t = th.thermo(text, 298.0)
        f, fr = sorted(t["freqs"]), c["freqs"]
        print(f"[chignolin_force {label}] {r['wall']:.1f} s (CPU {c['cpu_processes']} processes {c['cpu_seconds']} s), "
              f"ZPE {t.get('zpe')} (CPU {c['zpe']}), S {t.get('S')} (CPU {c['S']}), G {t.get('G')} (CPU {c['G']}), "
              f"lowest {f[:3]}, GPU memory peak {peak} MiB", flush=True)
        resident_check(f"chignolin_force {label}", r)
        ok = len(f) == len(fr) and bool(f)
        check(ok, f"chignolin_force {label}: {len(f)} frequencies (CPU {len(fr)})")
        if ok:
            hi = [(x, y) for x, y in zip(fr, f) if x >= 100.0]
            rms = (sum((x - y) ** 2 for x, y in hi) / len(hi)) ** 0.5
            check(rms <= 2.0, f"chignolin_force {label}: RMS frequency difference >= 100 cm-1 {rms:.3f} (<= 2)")
            for k, tol in (("zpe", 0.1), ("S", 1.0), ("G", 0.3)):
                d = t[k] - c[k]
                check(abs(d) <= tol, f"chignolin_force {label}: {k} GPU - CPU {d:+.4f} (<= {tol})")
            summary[f"chignolin_force {label}"] = {"wall": r["wall"], "rms": rms, "zpe": t["zpe"] - c["zpe"],
                                                   "S": t["S"] - c["S"], "G": t["G"] - c["G"], "mem_peak_mib": peak}

    # 4: crambin FORCETS (d-orbital pairs, warm-started SCFs)
    if not a.skip_forcets:
        c = ref["crambin_forcets"]
        with mem:
            r = mopac_run(mopac, work / "crambin_forcets", c["keywords"], a.data / c["geometry"], "forcets")
        t = th.thermo(r["text"], 298.0)
        f, fr = sorted(t["freqs"]), c["freqs"]
        print(f"[crambin_forcets] {r['wall']:.1f} s (CPU {c['cpu_seconds']} s, {c['cpu_seconds'] / r['wall']:.0f}x), "
              f"ZPE {t.get('zpe')} (CPU {c['zpe']}), lowest {f[:3]}, GPU memory peak {mem.peak} MiB", flush=True)
        resident_check("crambin_forcets", r)
        if len(f) == len(fr) and f:
            rms = (sum((x - y) ** 2 for x, y in zip(fr, f)) / len(f)) ** 0.5
            check(rms <= 0.5, f"crambin_forcets: RMS frequency difference {rms:.3f} cm-1 (<= 0.5)")
            check(abs(t["zpe"] - c["zpe"]) <= 0.01, f"crambin_forcets: ZPE GPU - CPU {t['zpe'] - c['zpe']:+.4f}")
            summary["crambin_forcets"] = {"wall": r["wall"], "rms": rms, "zpe": t["zpe"] - c["zpe"],
                                          "mem_peak_mib": mem.peak}
        else:
            check(False, f"crambin_forcets: {len(f)} frequencies (CPU {len(fr)})")

    # 5: several crambin processes at once (parallel FORCE regime): no fallback, memory
    if a.concurrent > 0:
        geo = a.data / ref["crambin_scf"]["geometry"]
        regions = [f"A{3 + 7 * k}" for k in range(a.concurrent)]
        dirs = []
        procs = []
        env = os.environ.copy()
        env.pop("MOPAC_NOGPU", None)
        with mem:
            t0 = time.perf_counter()
            for reg in regions:
                d = work / "concurrent" / reg
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(geo, d / geo.name)
                (d / "f.mop").write_text(f'PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK FORCETS '
                                         f'THERMO(298) LET OPT("{reg}"=1) GEO_DAT="{geo.name}"\nconcurrent\n\n')
                log = open(d / "f.stdout", "w")
                procs.append(subprocess.Popen([str(mopac), "f.mop"], cwd=d, env=env, stdout=log,
                                              stderr=subprocess.STDOUT))
                dirs.append(d)
            for p in procs:
                p.wait()
            wall = time.perf_counter() - t0
        tot = {"resident": 0, "fallback": 0, "gpu_errors": 0}
        for d in dirs:
            wr = parse((d / "f.out").read_text(errors="ignore") if (d / "f.out").exists() else "",
                       (d / "f.stdout").read_text(errors="ignore"), None, 0.0)
            for k in tot:
                tot[k] += wr[k]
        print(f"[concurrent] {a.concurrent} crambin FORCETS processes: {wall:.1f} s, {tot['resident']} SCF resident, "
              f"{tot['fallback']} fallback, {tot['gpu_errors']} GPU errors, GPU memory peak {mem.peak} MiB", flush=True)
        check(tot["resident"] > 0 and tot["fallback"] == 0 and tot["gpu_errors"] == 0,
              f"concurrent: {a.concurrent} processes, no fallback and no GPU error")
        summary["concurrent"] = dict(tot, wall=wall, mem_peak_mib=mem.peak)

    # 6: full crambin FORCE (timing)
    if a.large_workers > 0:
        c = ref["crambin_scf"]
        geo = a.data / c["geometry"]
        keys = "PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK THERMO(298) LET"
        with mem:
            t0 = time.perf_counter()
            out, tim = pf.parallel_force(mopac, geo, keys, a.large_workers, work / f"crambin_force_w{a.large_workers}")
            wall = time.perf_counter() - t0
        t = th.thermo(out.read_text(errors="ignore"), 298.0)
        tot = {"resident": 0, "fallback": 0, "gpu_errors": 0}
        for wdir in sorted((work / f"crambin_force_w{a.large_workers}").glob("worker_*")):
            wr = parse((wdir / "force.out").read_text(errors="ignore") if (wdir / "force.out").exists() else "",
                       (wdir / "force.stdout").read_text(errors="ignore") if (wdir / "force.stdout").exists() else "",
                       None, 0.0)
            for k in tot:
                tot[k] += wr[k]
        print(f"[crambin_force] {a.large_workers} processes: {wall:.1f} s (A100 reference: 2642 s with 6), "
              f"{len(t['freqs'])} frequencies, ZPE {t.get('zpe')}, {tot['resident']} SCF resident, "
              f"{tot['fallback']} fallback, GPU memory peak {mem.peak} MiB", flush=True)
        check(tot["fallback"] == 0 and tot["gpu_errors"] == 0, "crambin_force: no fallback and no GPU error")
        summary["crambin_force"] = dict(tot, wall=wall, zpe=t.get("zpe"), mem_peak_mib=mem.peak)

    summary["failures"] = failures
    (work / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"work dir: {work}")
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""MOZYME GPU acceptance test: CPU-vs-GPU tolerance, gradient agreement, hand-back and kill switch.

Usage:
  check_mozyme_gpu_tolerance.py <mopac> <deck.mop> [<deck.mop> ...] [--hof-tol 0.05]
      [--grad-rms-tol 1e-3] [--handback <deck.mop>] [--work-dir DIR]

For every deck: run it once with the GPU disabled (MOPAC_NOGPU=1) and once with MOPAC's
production defaults (GPU on a capable device); the final heats of formation must agree within
--hof-tol kcal/mol and the GPU run must report a successful resident SCF.  Decks whose keywords
request a geometry optimization or GRADIENTS are additionally run with MOPAC_GPU_GRAD_CHECK=1,
and every "[MOZYME GPU gradient] check" line must report rms_diff <= --grad-rms-tol
kcal/mol/A.  The optional --handback deck (DENOUT=n) must complete on the CPU after the resident
SCF hands back, again within --hof-tol of the CPU-only heat.  Finally, the first deck is run in the
GPU build with the NOGPU keyword: no GPU helper may run and the heat must equal the
MOPAC_NOGPU run bit for bit.

Exit status 0 when every check passes; the report is printed either way.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HEAT_RE = re.compile(r"FINAL HEAT OF FORMATION\s*=\s*([-+0-9.EeDd]+)\s*KCAL/MOL")
SCF_STATUS_RE = re.compile(r"\[MOZYME GPU SCF\]\s+status=(\S+)")
GRAD_CHECK_RE = re.compile(r"\[MOZYME GPU gradient\] check .*?rms_diff=\s*([-+0-9.EeDd]+)")
GPU_SUCCESS_RE = re.compile(r"\[MOZYME GPU \w+\]\s+(?:status=)?success")
OPT_RE = re.compile(r"(?i)(^|\s)(GRADIENTS|CYCLES\s*=|BFGS|LBFGS|EF\b|TS\b)")
GEO_DAT_RE = re.compile(r"(?i)GEO_DAT\s*=\s*(?:\"([^\"]+)\"|(\S+))")

GPU_OFF_KEYS = [
    "MOPAC_FORCEGPU", "MOZYME_GPU_FORCE", "MOPAC_MOZYME_SCF_EXPERIMENTAL", "MOPAC_MOZYME_RESIDENT_SCF",
    "MOPAC_MOZYME_SCF_GPU", "MOPAC_MOZYME_RESIDENT_FOCK_GPU", "MOPAC_MOZYME_MAKVEC_GPU",
    "MOPAC_MOZYME_SCF_STRICT_RESIDENT", "MOPAC_MOZYME_GRAD_GPU", "MOPAC_MOZYME_HCORE_GPU",
    "MOPAC_DH_DISP_GPU", "MOPAC_GPU_GRAD_CHECK", "MOPAC_GPU_HCORE_CHECK", "MOPAC_GPU_DISP_CHECK",
    "MOPAC_NOGPU", "MOZYME_GPU_OFF",
]


def to_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in GPU_OFF_KEYS:
        env.pop(key, None)
    if extra:
        env.update(extra)
    return env


def stage(deck: Path, run_dir: Path, keyword_suffix: str = "") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    text = deck.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if keyword_suffix:
        idx = next((i for i, line in enumerate(lines) if line.strip()), 0)
        lines[idx] = lines[idx].rstrip() + " " + keyword_suffix
        text = "\n".join(lines) + "\n"
    staged = run_dir / deck.name
    staged.write_text(text, encoding="utf-8")
    for quoted, bare in GEO_DAT_RE.findall(text):
        ref = deck.parent / (quoted or bare)
        if ref.exists():
            shutil.copy2(ref, run_dir / ref.name)
    for ext in (".res", ".den"):
        ref = deck.with_suffix(ext)
        if ref.exists():
            shutil.copy2(ref, run_dir / ref.name)
    return staged


def run_mopac(mopac: Path, deck: Path, run_dir: Path, env: dict[str, str], keyword_suffix: str = "",
              timeout: float = 3600.0) -> tuple[str, float | None]:
    staged = stage(deck, run_dir, keyword_suffix)
    proc = subprocess.run([str(mopac), staged.name], cwd=run_dir, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=timeout)
    text = proc.stdout
    out = run_dir / (staged.stem + ".out")
    if out.exists():
        text += "\n" + out.read_text(encoding="utf-8", errors="ignore")
    heats = HEAT_RE.findall(text)
    heat = to_float(heats[-1]) if heats else None
    return text, heat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mopac")
    parser.add_argument("decks", nargs="+")
    parser.add_argument("--hof-tol", type=float, default=0.05, help="kcal/mol (default 0.05)")
    parser.add_argument("--grad-rms-tol", type=float, default=1.0e-3, help="kcal/mol/A (default 1e-3)")
    parser.add_argument("--handback", default=None, help="DENOUT deck exercising the CPU hand-back")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    mopac = Path(args.mopac).resolve()
    work = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="mozyme_gpu_tol_"))
    failures: list[str] = []
    report: list[str] = []

    def check(cond: bool, message: str) -> None:
        report.append(("PASS " if cond else "FAIL ") + message)
        if not cond:
            failures.append(message)

    decks = [Path(d).resolve() for d in args.decks]
    cpu_heats: dict[str, float | None] = {}
    for deck in decks:
        name = deck.stem
        cpu_text, cpu_heat = run_mopac(mopac, deck, work / name / "cpu", clean_env({"MOPAC_NOGPU": "1"}),
                                       timeout=args.timeout)
        gpu_text, gpu_heat = run_mopac(mopac, deck, work / name / "gpu", clean_env(), timeout=args.timeout)
        cpu_heats[name] = cpu_heat
        check(cpu_heat is not None, f"{name}: CPU run produced a heat of formation")
        check(gpu_heat is not None, f"{name}: GPU run produced a heat of formation")
        statuses = SCF_STATUS_RE.findall(gpu_text)
        check("success" in statuses, f"{name}: GPU run reports a successful resident SCF (statuses: {sorted(set(statuses))})")
        check("strict_abort" not in statuses and "[GPU ERROR]" not in gpu_text,
              f"{name}: no GPU error or strict abort in the GPU run")
        if cpu_heat is not None and gpu_heat is not None:
            diff = abs(gpu_heat - cpu_heat)
            check(diff <= args.hof_tol,
                  f"{name}: |dHf(GPU - CPU)| = {diff:.4f} kcal/mol <= {args.hof_tol} (CPU {cpu_heat:.5f}, GPU {gpu_heat:.5f})")
        keywords = deck.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        if OPT_RE.search(keywords):
            chk_text, _ = run_mopac(mopac, deck, work / name / "gradcheck",
                                    clean_env({"MOPAC_GPU_GRAD_CHECK": "1"}), timeout=args.timeout)
            rms = [to_float(v) for v in GRAD_CHECK_RE.findall(chk_text)]
            check(len(rms) > 0, f"{name}: GPU gradient check lines found ({len(rms)})")
            if rms:
                check(max(rms) <= args.grad_rms_tol,
                      f"{name}: max gradient rms_diff {max(rms):.3e} kcal/mol/A <= {args.grad_rms_tol:g}")

    if args.handback:
        deck = Path(args.handback).resolve()
        name = deck.stem
        cpu_text, cpu_heat = run_mopac(mopac, deck, work / name / "cpu", clean_env({"MOPAC_NOGPU": "1"}),
                                       timeout=args.timeout)
        gpu_text, gpu_heat = run_mopac(mopac, deck, work / name / "gpu", clean_env(), timeout=args.timeout)
        statuses = SCF_STATUS_RE.findall(gpu_text)
        check("fallback_cpu" in statuses, f"{name}: resident SCF handed back to the CPU (statuses: {sorted(set(statuses))})")
        check("SCF CALCULATION FAILED" not in gpu_text, f"{name}: the CPU continuation converged")
        check(cpu_heat is not None and gpu_heat is not None and abs(gpu_heat - cpu_heat) <= args.hof_tol,
              f"{name}: hand-back heat within {args.hof_tol} of CPU (CPU {cpu_heat}, GPU {gpu_heat})")

    # Kill switch: NOGPU keyword in the GPU build.
    deck = decks[0]
    name = deck.stem
    off_text, off_heat = run_mopac(mopac, deck, work / name / "nogpu_keyword", clean_env(), keyword_suffix="NOGPU",
                                   timeout=args.timeout)
    check(not GPU_SUCCESS_RE.search(off_text), f"{name} + NOGPU: no GPU helper ran")
    check(off_heat is not None and cpu_heats.get(name) is not None and off_heat == cpu_heats[name],
          f"{name} + NOGPU: heat identical to the MOPAC_NOGPU run ({off_heat} vs {cpu_heats.get(name)})")

    print("\n".join(report))
    print(f"work dir: {work}")
    if failures:
        print(f"{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

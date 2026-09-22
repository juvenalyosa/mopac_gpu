#!/usr/bin/env python3
"""Does the MOZYME GPU path give the CPU result for every semiempirical method?

For each method the crambin 1SCF deck is rewritten with that method keyword and run
twice: with the GPU disabled (MOPAC_NOGPU=1) and with the defaults.  The heats of
formation, the resident SCF status and any GPU error line are compared.  With --opt the
3-cycle optimization deck is also run on the GPU with the hcore, gradient and
dispersion/H-bond check modes on, and the reported differences are printed.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HEAT_RE = re.compile(r"(?:FINAL HEAT OF FORMATION|CURRENT VALUE OF HEAT OF FORMATION)\s*=\s*([-+0-9.EeDd]+)")
STATUS_RE = re.compile(r"\[MOZYME GPU SCF\] status=(\w+)")
STATUS_LINE_RE = re.compile(r"\[MOZYME GPU SCF\] (?:status=\w+ reason=|lmo_storage_grown)[^\n]*")
CHECK_RE = re.compile(r"\[MOZYME GPU (hcore|gradient|disp|hbond)\] check.*")
ERROR_RE = re.compile(r"GPU ERROR|strict abort|Backtrace", re.IGNORECASE)
REF_RE = re.compile(r'"([^"]+)"')
CPU_ENV_OFF = ("MOPAC_NOGPU", "MOZYME_GPU_OFF", "MOPAC_FORCEGPU")

DEFAULT_METHODS = ["MNDO", "AM1", "PM3", "RM1", "PM6", "PM6-D3H4", "PM7"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mopac", type=Path)
    parser.add_argument("--deck", type=Path, default=Path("benchmarks/publication_inputs/mop/protein_crambin_1crn.mop"))
    parser.add_argument("--opt-deck", type=Path, default=Path("benchmarks/publication_inputs/mop/protein_crambin_1crn_opt.mop"))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--opt", action="store_true", help="also run the optimization deck with the GPU check modes")
    parser.add_argument("--work-dir", type=Path, default=Path("mozyme_gpu_methods"))
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def rewrite(deck: Path, method: str, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = deck.read_text(errors="ignore").splitlines()
    tokens = lines[0].split()
    tokens[0] = method  # the benchmark decks start with the method keyword
    lines[0] = " ".join(tokens)
    new = run_dir / deck.name
    new.write_text("\n".join(lines) + "\n")
    for ref in REF_RE.findall(lines[0]):
        src = deck.parent / Path(ref).name
        if src.exists():
            shutil.copy2(src, run_dir / src.name)
    return new


def run(mopac: Path, deck: Path, gpu: bool, timeout: float, extra_env: dict | None = None) -> tuple[str, float]:
    env = os.environ.copy()
    for key in CPU_ENV_OFF:
        env.pop(key, None)
    if not gpu:
        env["MOPAC_NOGPU"] = "1"
    if extra_env:
        env.update(extra_env)
    t0 = time.perf_counter()
    try:
        subprocess.run([str(mopac), deck.name], cwd=deck.parent, capture_output=True, text=True, env=env,
                       timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        pass
    out = deck.parent / (deck.stem + ".out")
    return (out.read_text(errors="ignore") if out.exists() else ""), time.perf_counter() - t0


def heat_of(text: str):
    hits = HEAT_RE.findall(text)
    return float(hits[-1].replace("D", "E")) if hits else None


def main() -> int:
    args = parse_args()
    args.mopac = args.mopac.resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    failures = 0
    print(f"{'method':10s} {'CPU heat':>14s} {'GPU heat':>14s} {'dHf':>9s} {'CPU s':>7s} {'GPU s':>7s}  status")
    for method in methods:
        cpu_deck = rewrite(args.deck, method, args.work_dir / method / "cpu")
        gpu_deck = rewrite(args.deck, method, args.work_dir / method / "gpu")
        cpu_text, cpu_s = run(args.mopac, cpu_deck, False, args.timeout)
        gpu_text, gpu_s = run(args.mopac, gpu_deck, True, args.timeout)
        cpu_heat, gpu_heat = heat_of(cpu_text), heat_of(gpu_text)
        statuses = sorted(set(STATUS_RE.findall(gpu_text)))
        errors = ERROR_RE.search(gpu_text) is not None
        note = ",".join(statuses) or "no resident SCF line"
        if errors:
            note += " GPU-ERROR"
        ok = cpu_heat is not None and gpu_heat is not None and abs(gpu_heat - cpu_heat) <= args.tolerance \
            and not errors and statuses == ["success"]
        if not ok:
            failures += 1
            note = "FAIL " + note
        d = (gpu_heat - cpu_heat) if (cpu_heat is not None and gpu_heat is not None) else None
        print(f"{method:10s} {cpu_heat if cpu_heat is not None else float('nan'):14.4f} "
              f"{gpu_heat if gpu_heat is not None else float('nan'):14.4f} "
              f"{d if d is not None else float('nan'):9.4f} {cpu_s:7.1f} {gpu_s:7.1f}  {note}", flush=True)
        if statuses != ["success"] or "lmo_storage_grown" in gpu_text:
            for line in STATUS_LINE_RE.findall(gpu_text):
                print("    " + line.strip()[:200])
        if args.opt:
            opt_deck = rewrite(args.opt_deck, method, args.work_dir / method / "opt_gpu_check")
            text, secs = run(args.mopac, opt_deck, True, args.timeout,
                             {"MOPAC_GPU_GRAD_CHECK": "1", "MOPAC_GPU_HCORE_CHECK": "1", "MOPAC_GPU_DISP_CHECK": "1"})
            checks = CHECK_RE.findall(text)
            lines = [l.strip() for l in text.splitlines() if CHECK_RE.search(l)]
            print(f"    opt ({secs:.1f}s): {len(lines)} check lines" + (", none" if not lines else ""))
            seen = set()
            for l in lines:
                kind = l.split("]")[0]
                if kind in seen:
                    continue
                seen.add(kind)
                print("      " + l[:160])
            if ERROR_RE.search(text):
                failures += 1
                print("      FAIL: GPU error in the optimization run")
    print("all methods agree within tolerance" if failures == 0 else f"{failures} method(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

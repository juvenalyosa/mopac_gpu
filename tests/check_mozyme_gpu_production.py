#!/usr/bin/env python3
"""MOZYME GPU production-workload check: long geometry optimization and molecular dynamics.

Usage:
  check_mozyme_gpu_production.py <mopac> [--opt-deck DECK] [--drc-deck DECK]
      [--drc-gpu-deck DECK] [--hof-tol 0.05] [--work-dir DIR]

--opt-deck (e.g. 1AKE apo, CYCLES=50): run on the GPU only (a CPU optimization of a 6700-atom
protein takes hours).  Every SCF of every cycle must be a successful resident SCF (no CPU
fallback), the heat must go down over the run, and the final geometry (restart file) is
re-evaluated as RESTART 1SCF with the GPU disabled and enabled: those two heats must agree within
--hof-tol.

--drc-deck (e.g. crambin, DRC): the same molecular-dynamics run on the CPU and on the GPU.
Trajectories diverge (the SCF is not bit-identical and MD is chaotic), so the check is energy
conservation: after the first points (DRC start-up), the largest |ERROR| (drift of the total
energy) of the GPU run must not exceed max(1.0, 3 x the CPU run's) kcal/mol, and the first
potential energies must agree within --hof-tol.

--drc-gpu-deck (e.g. 1AKE apo, DRC): GPU only; reports speed and energy conservation.

Exit status 0 when every check passes; the report is printed either way.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mozyme_gpu_tolerance import (GEO_DAT_RE, HEAT_RE, SCF_STATUS_RE,  # noqa: E402
                                        clean_env, run_mopac, to_float)

CYCLE_RE = re.compile(r"CYCLE:\s*(\d+).*?GRAD\.:\s*([-+0-9.EeDd]+)\s+HEAT:\s*([-+0-9.EeDd]+)")
FLOAT_RE = re.compile(r"-?\d+\.\d+")
DRC_START_POINTS = 3   # points skipped before measuring energy conservation


def drc_rows(text: str) -> list[tuple[float, float, float, float]]:
    """(time fs, potential, total, error) of every DRC printout row.

    GPU trace lines ("[MOZYME GPU SCF] ...") are printed between the rows, so every
    row-shaped line after the table header counts, not just a contiguous block.
    """
    rows = []
    in_table = False
    for line in text.splitlines():
        if "FEMTOSECONDS  POINT  POTENTIAL" in line:
            in_table = True
            continue
        if not in_table:
            continue
        head = line.split()
        if len(head) < 2 or not re.fullmatch(r"-?\d+\.\d+", head[0]) or not head[1].isdigit():
            continue
        # glued columns ("-3109.0971-227.41577") -> take the floats by pattern
        values = [float(v) for v in FLOAT_RE.findall(line)]
        if len(values) >= 5:
            fs, potential, _kinetic, total, error = values[:5]
            rows.append((fs, potential, total, error))
    return rows


def restart_deck(deck: Path, res: Path, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    name = deck.stem + "_restart_1scf"
    shutil.copy2(res, run_dir / f"{name}.res")
    text = deck.read_text(encoding="utf-8", errors="ignore")
    for quoted, bare in GEO_DAT_RE.findall(text):
        ref = deck.parent / (quoted or bare)
        if ref.exists():
            shutil.copy2(ref, run_dir / ref.name)
    keywords = text.splitlines()[0]
    keywords = re.sub(r"(?i)(^|\s)(CYCLES\s*=\s*\S+|GRADIENTS|BFGS|LBFGS|EF|TS)(?=\s|$)", " ", keywords)
    out = run_dir / f"{name}.mop"
    out.write_text(keywords.rstrip() + " 1SCF RESTART\n"
                   "1SCF at the GPU optimization's final geometry (restart file)\n\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mopac")
    parser.add_argument("--opt-deck", type=Path)
    parser.add_argument("--drc-deck", type=Path)
    parser.add_argument("--drc-gpu-deck", type=Path)
    parser.add_argument("--hof-tol", type=float, default=0.05)
    parser.add_argument("--work-dir", type=Path, default=Path("mozyme_gpu_production"))
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--reuse", action="store_true",
                        help="re-analyse the outputs already in --work-dir instead of running MOPAC again")
    args = parser.parse_args()

    mopac = Path(args.mopac).resolve()
    work = args.work_dir.resolve()
    if work.exists() and not args.reuse:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    def check(cond: bool, message: str) -> None:
        print(("PASS " if cond else "FAIL ") + message, flush=True)
        if not cond:
            failures.append(message)

    def timed(deck: Path, run_dir: Path, env: dict[str, str]) -> tuple[str, float | None, float]:
        out = run_dir / (deck.stem + ".out")
        if args.reuse and out.exists():
            text = out.read_text(encoding="utf-8", errors="ignore")
            heats = HEAT_RE.findall(text)
            wall = re.findall(r"TOTAL JOB TIME:\s*([0-9.]+)", text)
            return text, (to_float(heats[-1]) if heats else None), (float(wall[-1]) if wall else 0.0)
        t0 = time.perf_counter()
        text, heat = run_mopac(mopac, deck, run_dir, env, timeout=args.timeout)
        return text, heat, time.perf_counter() - t0

    if args.opt_deck:
        deck = args.opt_deck.resolve()
        name = deck.stem
        text, heat, wall = timed(deck, work / name / "gpu", clean_env())
        cycles = [(int(c), to_float(g), to_float(h)) for c, g, h in CYCLE_RE.findall(text)]
        statuses = SCF_STATUS_RE.findall(text)
        n_ok = statuses.count("success")
        print(f"     {name}: GPU wall {wall:.1f} s, {len(cycles)} cycles, {len(statuses)} SCF status lines "
              f"({n_ok} success)", flush=True)
        if cycles:
            first, last = cycles[0], cycles[-1]
            warm = [t for t in re.findall(r"CYCLE:\s*\d+\s+TIME:\s*([0-9.]+)", text)][1:]
            warm_avg = sum(float(t) for t in warm) / len(warm) if warm else float("nan")
            print(f"     cycle 1: heat {first[2]:.3f}, GRAD {first[1]:.1f}; cycle {last[0]}: heat {last[2]:.3f}, "
                  f"GRAD {last[1]:.1f}; mean time per cycle after the first {warm_avg:.2f} s", flush=True)
        check(len(cycles) > 1, f"{name}: optimization ran ({len(cycles)} cycles)")
        check(len(statuses) > 0 and n_ok == len(statuses),
              f"{name}: every SCF was a successful resident SCF ({sorted(set(statuses))})")
        if len(cycles) > 1:
            check(cycles[-1][2] < cycles[0][2],
                  f"{name}: heat went down ({cycles[0][2]:.3f} -> {cycles[-1][2]:.3f})")
        res = next(iter((work / name / "gpu").glob("*.res")), None)
        check(res is not None, f"{name}: GPU optimization left a restart file (.res)")
        if res is not None:
            rdeck = restart_deck(deck, res, work / name / "restart")
            _, rc_heat, rc_wall = timed(rdeck, work / name / "restart" / "cpu", clean_env({"MOPAC_NOGPU": "1"}))
            _, rg_heat, rg_wall = timed(rdeck, work / name / "restart" / "gpu", clean_env())
            check(rc_heat is not None and rg_heat is not None,
                  f"{name}: RESTART 1SCF at the final geometry (CPU {rc_heat} in {rc_wall:.0f} s, "
                  f"GPU {rg_heat} in {rg_wall:.1f} s)")
            if rc_heat is not None and rg_heat is not None:
                diff = rg_heat - rc_heat
                check(abs(diff) <= args.hof_tol,
                      f"{name}: |dHf(GPU - CPU)| at the final geometry = {abs(diff):.4f} kcal/mol <= {args.hof_tol}")

    def drc_summary(label: str, text: str, wall: float) -> tuple[list, float | None]:
        rows = drc_rows(text)
        tail = rows[DRC_START_POINTS:]
        max_err = max((abs(r[3]) for r in tail), default=None)
        if rows:
            print(f"     {label}: wall {wall:.1f} s, {len(rows)} points to {rows[-1][0]:.1f} fs, "
                  f"total energy {rows[0][2]:.3f} -> {rows[-1][2]:.3f}, max |ERROR| after point "
                  f"{DRC_START_POINTS}: {max_err if max_err is None else round(max_err, 4)} kcal/mol", flush=True)
        else:
            print(f"     {label}: wall {wall:.1f} s, no DRC rows found", flush=True)
        return rows, max_err

    if args.drc_deck:
        deck = args.drc_deck.resolve()
        name = deck.stem
        ctext, _, cwall = timed(deck, work / name / "cpu", clean_env({"MOPAC_NOGPU": "1"}))
        gtext, _, gwall = timed(deck, work / name / "gpu", clean_env())
        crows, cerr = drc_summary(f"{name} CPU", ctext, cwall)
        grows, gerr = drc_summary(f"{name} GPU", gtext, gwall)
        statuses = SCF_STATUS_RE.findall(gtext)
        check(len(grows) > DRC_START_POINTS and len(crows) > DRC_START_POINTS,
              f"{name}: DRC ran on CPU ({len(crows)} points) and GPU ({len(grows)} points)")
        check(len(statuses) > 0 and all(s == "success" for s in statuses),
              f"{name}: every GPU SCF was a successful resident SCF ({sorted(set(statuses))})")
        if crows and grows:
            d0 = abs(grows[0][1] - crows[0][1])
            check(d0 <= args.hof_tol, f"{name}: starting potential energy CPU vs GPU differs by {d0:.4f} kcal/mol")
        if cerr is not None and gerr is not None:
            limit = max(1.0, 3.0 * cerr)
            check(gerr <= limit, f"{name}: GPU energy conservation max |ERROR| {gerr:.4f} <= {limit:.4f} "
                  f"(CPU {cerr:.4f}) kcal/mol")
        if cwall > 0 and gwall > 0:
            print(f"     {name}: speedup {cwall / gwall:.1f}x", flush=True)

    if args.drc_gpu_deck:
        deck = args.drc_gpu_deck.resolve()
        name = deck.stem
        gtext, _, gwall = timed(deck, work / name / "gpu", clean_env())
        grows, gerr = drc_summary(f"{name} GPU", gtext, gwall)
        statuses = SCF_STATUS_RE.findall(gtext)
        check(len(grows) > DRC_START_POINTS, f"{name}: DRC ran on the GPU ({len(grows)} points)")
        check(len(statuses) > 0 and all(s == "success" for s in statuses),
              f"{name}: every GPU SCF was a successful resident SCF ({sorted(set(statuses))})")
        if gerr is not None:
            check(gerr <= 1.0, f"{name}: energy conservation max |ERROR| {gerr:.4f} <= 1.0 kcal/mol")

    print(f"work dir: {work}")
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

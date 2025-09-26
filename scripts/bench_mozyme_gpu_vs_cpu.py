#!/usr/bin/env python3
"""Benchmark MOZYME CPU vs GPU execution and generate a bar plot."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError as exc:  # pragma: no cover
    plt = None

MODES = {
    "GPU": {"MOZYME_GPU_FORCE": "1", "MOZYME_GPU_OFF": None},
    "CPU": {"MOZYME_GPU_OFF": "1", "MOZYME_GPU_FORCE": None},
}

POSSIBLE_TIME_KEYS = (
    "TOTAL JOB TIME",
    "TOTAL CPU TIME",
    "TOTAL WALL CLOCK",
    "WALL CLOCK TIME",
    "TOTAL ELAPSED TIME",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "binary",
        nargs="?",
        default="./build-gpu/mopac",
        help="Path to the MOPAC executable (default: ./build-gpu/mopac)",
    )
    parser.add_argument(
        "--input",
        default="examples/mozyme_protein_auto.mop",
        help="Input deck to benchmark (default: examples/mozyme_protein_auto.mop)",
    )
    parser.add_argument(
        "--log-dir",
        default="bench_logs",
        help="Directory to store stdout logs (default: bench_logs)",
    )
    parser.add_argument(
        "--plot",
        default="bench_mozyme_gpu_vs_cpu.png",
        help="Path of the output PNG plot (default: bench_mozyme_gpu_vs_cpu.png)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of runs per mode to average (default: 1)",
    )
    return parser.parse_args()


def ensure_matplotlib() -> None:
    if plt is None:
        raise SystemExit(
            "matplotlib is required for plotting. Install it (e.g. pip install matplotlib) "
            "or rerun with --plot /dev/null to skip plotting."
        )


def run_job(mode: str, binary: Path, input_file: Path, log_dir: Path) -> Dict[str, object]:
    env = os.environ.copy()
    env.setdefault("MOPAC_FORCEGPU", "1")  # keep GPU context initialised
    env.setdefault("MOPAC_DETERMINISTIC", "1")
    env.setdefault("MOPAC_GPU_VERBOSE", "1")

    for key, value in MODES[mode].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    log_path = log_dir / f"mozyme_{mode.lower()}.out"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            [str(binary), str(input_file)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    elapsed = time.perf_counter() - start
    return {
        "mode": mode,
        "elapsed": elapsed,
        "returncode": proc.returncode,
        "log": log_path,
    }


def parse_log_time(log_path: Path) -> Optional[float]:
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    lines = content.splitlines()
    for key in POSSIBLE_TIME_KEYS:
        for line in lines[::-1]:  # scan from bottom up
            if key in line:
                tokens = line.split()
                for token in reversed(tokens):
                    try:
                        return float(token)
                    except ValueError:
                        continue
    return None


def aggregate_runs(results: List[Dict[str, object]], repeats: int) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for mode in MODES:
        mode_runs = [r for r in results if r["mode"] == mode]
        avg_wall = sum(r["elapsed"] for r in mode_runs) / len(mode_runs)
        parsed = [parse_log_time(r["log"]) for r in mode_runs]
        parsed = [p for p in parsed if p is not None]
        stats[mode] = {
            "wall": avg_wall,
            "log_time": sum(parsed) / len(parsed) if parsed else float("nan"),
        }
    return stats


def make_plot(stats: Dict[str, Dict[str, float]], out_path: Path, title: str) -> None:
    ensure_matplotlib()
    modes = list(stats.keys())
    wall_times = [stats[m]["wall"] for m in modes]
    colors = ["#1f77b4" if m == "GPU" else "#ff7f0e" for m in modes]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(modes, wall_times, color=colors)
    ax.set_ylabel("Wall-clock seconds")
    ax.set_title(title)
    for bar, seconds in zip(bars, wall_times):
        ax.text(bar.get_x() + bar.get_width() / 2, seconds, f"{seconds:.1f}s", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    binary = Path(args.binary).resolve()
    input_file = Path(args.input).resolve()
    log_dir = Path(args.log_dir).resolve()
    plot_path = Path(args.plot).resolve()

    if not binary.exists():
        raise SystemExit(f"MOPAC executable not found: {binary}")
    if not input_file.exists():
        raise SystemExit(f"Input file not found: {input_file}")
    log_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for mode in MODES:
        for rep in range(args.repeats):
            print(f"[{mode}] run {rep + 1}/{args.repeats}:", flush=True)
            result = run_job(mode, binary, input_file, log_dir)
            if result["returncode"] != 0:
                print(f"  Warning: return code {result['returncode']}, see {result['log']}")
            print(f"  Wall time: {result['elapsed']:.2f}s (log: {result['log']})")
            results.append(result)

    stats = aggregate_runs(results, args.repeats)
    print("\nSummary (average over runs):")
    for mode in MODES:
        wall = stats[mode]["wall"]
        log_time = stats[mode]["log_time"]
        print(f"  {mode:>3} wall = {wall:8.2f}s", end="")
        if log_time == log_time:  # not NaN
            print(f", log reported ≈ {log_time:.2f}")
        else:
            print("")

    try:
        make_plot(stats, plot_path, title=f"MOZYME GPU vs CPU — {input_file.name}")
        print(f"Plot saved to {plot_path}")
    except SystemExit as exc:  # matplotlib missing
        print(exc)
        print("Skipping plot generation.")


if __name__ == "__main__":
    main()

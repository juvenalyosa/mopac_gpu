#!/usr/bin/env python3
"""Run MOPAC GPU wrapper benchmarks and generate CSV, JSON, plots, and a report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchCase:
    label: str
    gemm: str
    syrk: str
    dsyevd: str
    rot1: str
    rot2: str


PROFILES: dict[str, list[BenchCase]] = {
    "quick": [
        BenchCase("small", "512,512,128,8", "512,128,8", "128,2", "256,2", "256,2"),
        BenchCase("medium", "1024,1024,256,8", "1024,256,5", "256,1", "512,2", "512,2"),
    ],
    "standard": [
        BenchCase("small", "512,512,128,8", "512,128,8", "128,2", "256,2", "256,2"),
        BenchCase("medium", "1024,1024,256,8", "1024,256,5", "256,1", "512,2", "512,2"),
        BenchCase("large", "2048,2048,512,5", "2048,512,3", "512,1", "1024,1", "1024,1"),
    ],
    "large": [
        BenchCase("small", "1024,1024,256,8", "1024,256,5", "256,1", "512,2", "512,2"),
        BenchCase("medium", "2048,2048,512,5", "2048,512,3", "512,1", "1024,1", "1024,1"),
        BenchCase("xlarge", "4096,4096,1024,3", "4096,1024,2", "768,1", "2048,1", "2048,1"),
    ],
}

EXPECTED_TIMING_OPS = ("GEMM", "SYRK", "DSYEVD", "ROT single", "ROT 2-GPU")
ACCURACY_TOLERANCES: dict[str, dict[str, float]] = {
    "GEMM": {"max_abs": 1.0e-10, "rms_abs": 1.0e-11, "rel_rms": 1.0e-12},
    "SYRK": {"max_abs": 1.0e-10, "rms_abs": 1.0e-11, "rel_rms": 1.0e-12},
    "DSYEVD": {"residual": 1.0e-10, "orthogonality": 1.0e-10},
    "ROT_SINGLE_VS_2GPU": {"max_abs": 1.0e-10, "rms_abs": 1.0e-11, "rel_rms": 1.0e-10},
}
EXPECTED_ACCURACY_METRICS: dict[str, tuple[str, ...]] = {
    check: tuple(metrics) for check, metrics in ACCURACY_TOLERANCES.items()
}
GPU_ERROR_MARKERS = (
    "[GPU ERROR]",
    "ACCURACY_FAIL",
    "BENCH_FAIL",
    "cuBLAS status",
    "cuSOLVER status",
    "CUBLAS_STATUS_",
    "CUSOLVER_STATUS_",
    "CUDA error",
    "cudaError",
    "illegal memory access",
    "device-side assert",
    "device assert",
    "misaligned address",
    "unspecified launch failure",
    "out of memory",
    "Segmentation fault",
    "SIGSEGV",
    "core dumped",
)

GEMM_RE = re.compile(r"^GEMM size m=\s*(\d+)\s+n=\s*(\d+)\s+k=\s*(\d+)")
SYRK_RE = re.compile(r"^SYRK size n=\s*(\d+)\s+k=\s*(\d+)")
DSYEVD_RE = re.compile(r"^DSYEVD size n=\s*(\d+)")
ROT_RE = re.compile(r"^ROT\s+(single|2-GPU) n=\s*(\d+)\s+nocc=\s*(\d+)")
FIRST_RE = re.compile(r"first call:\s*([+\-0-9.Ee]+)\s*s(?:,\s*([+\-0-9.Ee]+)\s*GF/s)?")
AVG_RE = re.compile(r"avg \(cached\):\s*([+\-0-9.Ee]+)\s*s(?:,\s*([+\-0-9.Ee]+)\s*GF/s)?")
ACCURACY_RE = re.compile(r"^ACCURACY\s+(\S+)\s+(.*)$")
NUMBER_TOKEN = r"[+\-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+\-]?\d+)?|Inf(?:inity)?|NaN)"
KEYVAL_RE = re.compile(rf"([A-Za-z_]+)=\s*({NUMBER_TOKEN})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bench",
        nargs="?",
        default="/content/mopac_colab_build/mopac-gpu-bench",
        help="Path to mopac-gpu-bench.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="standard",
        help="Benchmark size profile.",
    )
    parser.add_argument(
        "--accuracy-size",
        default="256,128",
        help="Accuracy check size as n,k. Keep this modest because CPU references are used.",
    )
    parser.add_argument(
        "--out-dir",
        default="gpu_benchmark_report",
        help="Output directory for CSV, JSON, PNG, and Markdown files.",
    )
    parser.add_argument(
        "--bundle-zip",
        default="",
        help="Publication/analysis zip path. Default: <out-dir>_publication_data.zip.",
    )
    parser.add_argument(
        "--no-bundle-zip",
        action="store_true",
        help="Do not create the publication/analysis zip bundle.",
    )
    parser.add_argument(
        "--verbose-library",
        action="store_true",
        help="Enable MOPAC_GPU_VERBOSE and MOPAC_GPU_PROFILE for library timing messages.",
    )
    parser.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Skip accuracy checks.",
    )
    parser.add_argument(
        "--no-echo-output",
        action="store_true",
        help="Do not print raw benchmark output while running.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], env: dict[str, str], echo: bool) -> str:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    if echo:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with return code {proc.returncode}: {' '.join(cmd)}")
    lower_stdout = proc.stdout.lower()
    for marker in GPU_ERROR_MARKERS:
        if marker.lower() in lower_stdout:
            raise SystemExit(f"GPU benchmark output contained error marker {marker!r}: {' '.join(cmd)}")
    return proc.stdout


def iters_from_arg(value: str) -> int:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return int(parts[-1]) if parts else 0


def parse_timing_output(text: str, case: BenchCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    iters = {
        "GEMM": iters_from_arg(case.gemm),
        "SYRK": iters_from_arg(case.syrk),
        "DSYEVD": iters_from_arg(case.dsyevd),
        "ROT single": iters_from_arg(case.rot1),
        "ROT 2-GPU": iters_from_arg(case.rot2),
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = GEMM_RE.match(line)
        if match:
            current = new_timing_row(case.label, "GEMM", iters["GEMM"])
            current.update({"m": int(match.group(1)), "n": int(match.group(2)), "k": int(match.group(3))})
            rows.append(current)
            continue

        match = SYRK_RE.match(line)
        if match:
            current = new_timing_row(case.label, "SYRK", iters["SYRK"])
            current.update({"m": "", "n": int(match.group(1)), "k": int(match.group(2))})
            rows.append(current)
            continue

        match = DSYEVD_RE.match(line)
        if match:
            current = new_timing_row(case.label, "DSYEVD", iters["DSYEVD"])
            current.update({"m": "", "n": int(match.group(1)), "k": ""})
            rows.append(current)
            continue

        match = ROT_RE.match(line)
        if match:
            op = f"ROT {match.group(1)}"
            current = new_timing_row(case.label, op, iters[op])
            current.update({"m": "", "n": int(match.group(2)), "k": int(match.group(3))})
            rows.append(current)
            continue

        if current is None:
            continue

        match = FIRST_RE.search(line)
        if match:
            current["first_s"] = float(match.group(1))
            current["first_gflops"] = to_float_or_blank(match.group(2))
            continue

        match = AVG_RE.search(line)
        if match:
            current["avg_s"] = float(match.group(1))
            current["avg_gflops"] = to_float_or_blank(match.group(2))
            continue

    return rows


def new_timing_row(case_label: str, operation: str, iters: int) -> dict[str, Any]:
    return {
        "case": case_label,
        "operation": operation,
        "m": "",
        "n": "",
        "k": "",
        "iters": iters,
        "first_s": "",
        "avg_s": "",
        "first_gflops": "",
        "avg_gflops": "",
    }


def to_float_or_blank(value: str | None) -> float | str:
    return float(value) if value else ""


def parse_accuracy_output(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = ACCURACY_RE.match(line)
        if not match:
            continue
        check = match.group(1)
        values = {key: parse_metric_float(value) for key, value in KEYVAL_RE.findall(match.group(2))}
        n_value = metric_to_int(values.pop("n", 0.0), f"{check} n")
        k_value = metric_to_int(values.pop("k", 0.0), f"{check} k") if "k" in values else ""
        info_value = metric_to_int(values.pop("info", 0.0), f"{check} info") if "info" in values else ""
        for metric, value in values.items():
            rows.append(
                {
                    "check": check,
                    "n": n_value,
                    "k": k_value,
                    "metric": metric,
                    "value": value,
                    "info": info_value,
                }
            )
    return rows


def parse_metric_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def metric_to_int(value: float, label: str) -> int:
    if not math.isfinite(value):
        raise SystemExit(f"Invalid integer accuracy value for {label}: {value!r}.")
    return int(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def gpu_info() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    except OSError:
        return "nvidia-smi unavailable"
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "nvidia-smi unavailable"


def require_gpu_info(info: str) -> None:
    if not info or "unavailable" in info.lower():
        raise SystemExit("nvidia-smi did not report a CUDA GPU; refusing to write a GPU benchmark bundle.")


def require_finite_positive(value: Any, label: str) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Missing numeric benchmark value for {label}.") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise SystemExit(f"Invalid benchmark value for {label}: {value!r}.")


def require_finite_metric(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Missing numeric accuracy value for {label}.") from exc
    if not math.isfinite(number):
        raise SystemExit(f"Invalid accuracy value for {label}: {value!r}.")
    return number


def require_accuracy_within_tolerance(check: str, metric: str, value: Any) -> None:
    number = require_finite_metric(value, f"{check} {metric}")
    tolerance = ACCURACY_TOLERANCES[check][metric]
    if abs(number) > tolerance:
        raise SystemExit(
            f"Accuracy check {check} {metric}={number:.6e} exceeds tolerance {tolerance:.6e}."
        )


def validate_timing_rows(rows: list[dict[str, Any]], profile: str) -> None:
    expected_cases = {case.label for case in PROFILES[profile]}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        case = str(row.get("case", ""))
        operation = str(row.get("operation", ""))
        if case in expected_cases and operation in EXPECTED_TIMING_OPS:
            seen.add((case, operation))
            require_finite_positive(row.get("first_s"), f"{case} {operation} first_s")
            require_finite_positive(row.get("avg_s"), f"{case} {operation} avg_s")
            if operation in {"GEMM", "SYRK", "DSYEVD"}:
                require_finite_positive(row.get("avg_gflops"), f"{case} {operation} avg_gflops")

    missing = [
        f"{case.label} {operation}"
        for case in PROFILES[profile]
        for operation in EXPECTED_TIMING_OPS
        if (case.label, operation) not in seen
    ]
    if missing:
        raise SystemExit("Benchmark output did not include expected timing rows: " + ", ".join(missing))


def validate_accuracy_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        check = str(row.get("check", ""))
        metric = str(row.get("metric", ""))
        if check in ACCURACY_TOLERANCES and metric in ACCURACY_TOLERANCES[check]:
            seen.add((check, metric))
            require_accuracy_within_tolerance(check, metric, row.get("value"))
            if check == "DSYEVD" and int(row.get("info") or 0) != 0:
                raise SystemExit(f"DSYEVD accuracy check returned nonzero info={row.get('info')}.")

    missing = [
        f"{check} {metric}"
        for check, metrics in EXPECTED_ACCURACY_METRICS.items()
        for metric in metrics
        if (check, metric) not in seen
    ]
    if missing:
        raise SystemExit("Accuracy output did not include expected metrics: " + ", ".join(missing))


def make_plots(timing_rows: list[dict[str, Any]], accuracy_rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plots. In Colab it is preinstalled.") from exc

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    paths = [
        plot_gflops(timing_rows, out_dir / "gpu_throughput_gflops.png", plt),
        plot_times(timing_rows, out_dir / "gpu_wrapper_times.png", plt),
        plot_first_call(timing_rows, out_dir / "gpu_first_call_overhead.png", plt),
    ]
    if accuracy_rows:
        paths.append(plot_accuracy(accuracy_rows, out_dir / "gpu_accuracy.png", plt))
    return paths


def plot_gflops(rows: list[dict[str, Any]], path: Path, plt: Any) -> Path:
    data = [row for row in rows if row.get("avg_gflops") != ""]
    labels = [f"{row['case']} {row['operation']}" for row in data]
    values = [float(row["avg_gflops"]) for row in data]
    colors = [color_for_operation(row["operation"]) for row in data]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Cached end-to-end GF/s")
    ax.set_title("MOPAC GPU wrapper throughput")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_times(rows: list[dict[str, Any]], path: Path, plt: Any) -> Path:
    data = [row for row in rows if row.get("avg_s") != ""]
    labels = [f"{row['case']} {row['operation']}" for row in data]
    values = [float(row["avg_s"]) for row in data]
    colors = [color_for_operation(row["operation"]) for row in data]

    fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.33 * len(labels))))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Cached end-to-end seconds")
    ax.set_title("MOPAC GPU wrapper time per call")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_first_call(rows: list[dict[str, Any]], path: Path, plt: Any) -> Path:
    data = [
        row for row in rows
        if row.get("first_s") != "" and row.get("avg_s") != "" and float(row["avg_s"]) > 0.0
    ]
    labels = [f"{row['case']} {row['operation']}" for row in data]
    values = [float(row["first_s"]) / float(row["avg_s"]) for row in data]
    colors = [color_for_operation(row["operation"]) for row in data]

    fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.33 * len(labels))))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("First call / cached call")
    ax.set_title("Initialization and cache overhead")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy(rows: list[dict[str, Any]], path: Path, plt: Any) -> Path:
    data = [row for row in rows if row["metric"] != "info"]
    labels = [f"{row['check']} {row['metric']}" for row in data]
    values = [max(abs(float(row["value"])), 1.0e-18) for row in data]
    colors = [color_for_accuracy(row["metric"]) for row in data]

    fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.36 * len(labels))))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Absolute value, log scale")
    ax.set_title("GPU numerical accuracy checks")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def color_for_operation(operation: str) -> str:
    if operation == "GEMM":
        return "#2F6B9A"
    if operation == "SYRK":
        return "#D9822B"
    if operation == "DSYEVD":
        return "#4B8B3B"
    return "#7D4E8A"


def color_for_accuracy(metric: str) -> str:
    if metric == "max_abs":
        return "#2F6B9A"
    if metric == "rel_rms":
        return "#D9822B"
    if metric == "residual":
        return "#4B8B3B"
    if metric == "orthogonality":
        return "#7D4E8A"
    return "#666666"


def write_report(
    path: Path,
    bench: Path,
    profile: str,
    timing_rows: list[dict[str, Any]],
    accuracy_rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    lines = [
        "# MOPAC GPU Benchmark Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Benchmark binary: `{bench}`",
        f"Profile: `{profile}`",
        "",
        "## Benchmark Method",
        "",
        "This benchmark measures the production MOPAC GPU wrapper path. The measured operations are:",
        "",
        "- GEMM: FP64 dense matrix multiplication through the MOPAC cuBLAS/cuBLASLt wrapper.",
        "- SYRK: FP64 symmetric rank-k update through the MOPAC cuBLAS wrapper.",
        "- DSYEVD: FP64 dense symmetric eigensolver through the MOPAC cuSOLVER wrapper.",
        "- ROT: orbital-rotation wrapper timing for the single-GPU and 2-GPU entry points.",
        "",
        "For each benchmark case, the first call is reported separately because it includes CUDA",
        "context creation, cuBLAS/cuSOLVER handle setup, memory-cache growth, and other one-time",
        "initialization costs. The cached average is the primary timing to compare steady-state",
        "performance. Cached wrapper timings are end-to-end timings and include host/device",
        "staging, synchronization, and library calls.",
        "",
        "Scope guardrail: this report does not run a molecule, does not enter the MOPAC",
        "SCF driver, and does not validate complete SCF execution on GPU. Its machine-readable",
        "`summary.json` therefore sets `benchmark_scope=low_level_gpu_wrappers` and",
        "`full_scf_gpu_status=not_measured`. Use `scripts/molecule_benchmark_report.py`",
        "with `--require-full-scf-gpu` for a strict complete-SCF readiness gate.",
        "",
        "Accuracy checks are intentionally run at a modest matrix size because CPU FP64 reference",
        "calculations are used. GEMM and SYRK compare GPU FP64 results against CPU FP64 references.",
        "DSYEVD reports the normalized residual ||A V - V D|| / ||A|| and eigenvector",
        "orthogonality. ROT validates the single-GPU and 2-GPU wrapper results against a CPU reference.",
        "",
        "## GPU",
        "",
        "```",
        gpu_info(),
        "```",
        "",
        "## Timing",
        "",
        "Cached times are end-to-end wrapper timings.",
        "",
        "| Case | Operation | Size | Avg s | Avg GF/s | First s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in timing_rows:
        lines.append(
            "| {case} | {operation} | {size} | {avg_s} | {avg_gflops} | {first_s} |".format(
                case=row["case"],
                operation=row["operation"],
                size=size_text(row),
                avg_s=format_number(row["avg_s"]),
                avg_gflops=format_number(row["avg_gflops"]),
                first_s=format_number(row["first_s"]),
            )
        )

    if accuracy_rows:
        lines.extend(
            [
                "",
                "## Accuracy",
                "",
                "GEMM and SYRK compare GPU FP64 results against CPU FP64 references.",
                "DSYEVD reports eigensolver residual and eigenvector orthogonality.",
                "ROT validates the single-GPU and 2-GPU wrapper results against a CPU reference.",
                "The benchmark executable and this report both enforce these tolerances:",
                "",
                "| Check | Metric | Tolerance |",
                "|---|---:|---:|",
            ]
        )
        for check, metrics in ACCURACY_TOLERANCES.items():
            for metric, tolerance in metrics.items():
                lines.append(f"| {check} | {metric} | {tolerance:.1e} |")
        lines.extend(
            [
                "",
                "| Check | n | k | Metric | Value |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in accuracy_rows:
            lines.append(
                f"| {row['check']} | {row['n']} | {row['k']} | {row['metric']} | "
                f"{float(row['value']):.6e} |"
            )

    lines.extend(["", "## Plots", ""])
    for plot_path in plot_paths:
        rel = plot_path.name
        lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Data Files",
            "",
            "- `timing.csv`: parsed benchmark timing table.",
            "- `accuracy.csv`: parsed numerical accuracy metrics.",
            "- `summary.json`: machine-readable benchmark summary and metadata.",
            "  It explicitly marks complete SCF GPU readiness as `not_measured`.",
            "- `raw_outputs.json`: raw stdout from each benchmark command.",
            "- `README.md`: this report and benchmark-method description.",
            "- `*.png`: generated publication/analysis plots.",
        ]
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, generated_files: list[Path], profile: str, bench: Path) -> None:
    lines = [
        "MOPAC GPU publication/analysis data bundle",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Benchmark binary: {bench}",
        f"Profile: {profile}",
        "",
        "Files:",
    ]
    for item in sorted(generated_files):
        lines.append(f"- {item.name}")
    lines.append("")
    lines.append("Use timing.csv and accuracy.csv for statistical analysis.")
    lines.append("Use README.md for benchmark-method text and quick interpretation.")
    path.write_text("\n".join(lines), encoding="utf-8")


def create_bundle_zip(bundle_path: Path, out_dir: Path) -> Path:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.iterdir()):
            if path.resolve() == bundle_path.resolve() or not path.is_file():
                continue
            zf.write(path, arcname=f"{out_dir.name}/{path.name}")
    return bundle_path


def size_text(row: dict[str, Any]) -> str:
    op = row["operation"]
    if op == "GEMM":
        return f"{row['m']}x{row['n']}x{row['k']}"
    if op == "SYRK":
        return f"n={row['n']},k={row['k']}"
    if op == "DSYEVD":
        return f"n={row['n']}"
    return f"n={row['n']},nocc={row['k']}"


def format_number(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.6g}"


def main() -> int:
    args = parse_args()
    bench = Path(args.bench).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not bench.exists():
        raise SystemExit(f"Benchmark binary not found: {bench}")
    detected_gpu = gpu_info()
    require_gpu_info(detected_gpu)

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.verbose_library:
        env["MOPAC_GPU_VERBOSE"] = "1"
        env["MOPAC_GPU_PROFILE"] = "2"

    timing_rows: list[dict[str, Any]] = []
    raw_outputs: dict[str, str] = {}
    for case in PROFILES[args.profile]:
        cmd = [
            str(bench),
            f"--gemm={case.gemm}",
            f"--syrk={case.syrk}",
            f"--dsyevd={case.dsyevd}",
            f"--rot1={case.rot1}",
            f"--rot2={case.rot2}",
        ]
        output = run_command(cmd, env=env, echo=not args.no_echo_output)
        raw_outputs[case.label] = output
        timing_rows.extend(parse_timing_output(output, case))
    validate_timing_rows(timing_rows, args.profile)

    accuracy_rows: list[dict[str, Any]] = []
    if not args.no_accuracy:
        acc_cmd = [str(bench), "--accuracy-only", f"--accuracy={args.accuracy_size}"]
        accuracy_output = run_command(acc_cmd, env=env, echo=not args.no_echo_output)
        raw_outputs["accuracy"] = accuracy_output
        accuracy_rows = parse_accuracy_output(accuracy_output)
        validate_accuracy_rows(accuracy_rows)

    write_csv(
        out_dir / "timing.csv",
        timing_rows,
        ["case", "operation", "m", "n", "k", "iters", "first_s", "avg_s", "first_gflops", "avg_gflops"],
    )
    write_csv(out_dir / "accuracy.csv", accuracy_rows, ["check", "n", "k", "metric", "value", "info"])
    (out_dir / "raw_outputs.json").write_text(json.dumps(raw_outputs, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bench": str(bench),
                "profile": args.profile,
                "gpu": detected_gpu,
                "benchmark_scope": "low_level_gpu_wrappers",
                "full_scf_gpu_status": "not_measured",
                "full_scf_gpu_ready": 0,
                "publication_claim": "wrapper throughput only; not evidence of complete SCF GPU execution",
                "accuracy_tolerances": ACCURACY_TOLERANCES,
                "timing": timing_rows,
                "accuracy": accuracy_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_paths = make_plots(timing_rows, accuracy_rows, out_dir)
    report_path = out_dir / "README.md"
    write_report(report_path, bench, args.profile, timing_rows, accuracy_rows, plot_paths)
    generated_files = [
        out_dir / "timing.csv",
        out_dir / "accuracy.csv",
        out_dir / "raw_outputs.json",
        out_dir / "summary.json",
        report_path,
        *plot_paths,
    ]
    manifest_path = out_dir / "MANIFEST.txt"
    write_manifest(manifest_path, generated_files, args.profile, bench)
    generated_files.append(manifest_path)

    bundle_path: Path | None = None
    if not args.no_bundle_zip:
        bundle_path = Path(args.bundle_zip).resolve() if args.bundle_zip else out_dir.with_name(
            f"{out_dir.name}_publication_data.zip"
        )
        create_bundle_zip(bundle_path, out_dir)

    print("")
    print(f"Wrote report directory: {out_dir}")
    print(f"  {report_path}")
    for plot_path in plot_paths:
        print(f"  {plot_path}")
    if bundle_path is not None:
        print(f"Wrote publication/analysis bundle: {bundle_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

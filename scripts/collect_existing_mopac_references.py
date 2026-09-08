#!/usr/bin/env python3
"""Collect already-computed MOPAC output references into CSV/JSON tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


HEAT_RE = re.compile(r"(?:FINAL\s+)?HEAT\s+OF\s+FORMATION\s*=\s*([+\-0-9.EeDd]+)", re.IGNORECASE)
WALL_RE = re.compile(r"WALL-CLOCK TIME\s*=\s*(.+)", re.IGNORECASE)
TOTAL_JOB_RE = re.compile(r"TOTAL JOB TIME:\s*(.+)", re.IGNORECASE)

DEFAULT_FILES = [
    "examples/benzene.arc",
    "examples/h2o_gpu_force.arc",
    "examples/halogen_disp.arc",
    "examples/large_dense.arc",
    "examples/mozyme_protein_auto.arc",
    "examples/water_pm7_gpu.arc",
    "tests/Crambin_1SCF.out",
    "tests/test_Lewis_for_Proteins.out",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Specific .out/.arc files. Defaults to curated existing outputs.")
    parser.add_argument("--include-tests", action="store_true", help="Also scan tests/**/*.out and tests/**/*.arc.")
    parser.add_argument("--out-csv", default="benchmarks/existing_mopac_references.csv")
    parser.add_argument("--out-json", default="benchmarks/existing_mopac_references.json")
    return parser.parse_args()


def parse_file(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    heat_values = [parse_float(value) for value in HEAT_RE.findall(text)]
    heat_values = [value for value in heat_values if value is not None]
    wall_values = [parse_time(match.group(1)) for match in WALL_RE.finditer(text)]
    wall_values = [value for value in wall_values if value is not None]
    job_values = [parse_time(match.group(1)) for match in TOTAL_JOB_RE.finditer(text)]
    job_values = [value for value in job_values if value is not None]
    if not heat_values and not wall_values and not job_values:
        return None

    return {
        "name": path.stem,
        "source_path": str(path),
        "source_type": path.suffix.lstrip("."),
        "heat_kcal_mol": heat_values[-1] if heat_values else "",
        "wall_clock_s": wall_values[-1] if wall_values else "",
        "total_job_s": job_values[-1] if job_values else "",
        "has_gpu_keyword": contains_any(text, ["MOZYME_GPU", "MOPAC_FORCEGPU", "GPU"]),
        "has_mozyme": "MOZYME" in text.upper(),
        "normal_end": "JOB ENDED NORMALLY" in text or "== MOPAC DONE ==" in text,
        "notes": "existing output; use as historical reference, not same-session benchmark",
    }


def contains_any(text: str, needles: list[str]) -> bool:
    upper = text.upper()
    return any(needle.upper() in upper for needle in needles)


def parse_float(value: str) -> float | None:
    try:
        return float(value.replace("D", "E").replace("d", "E"))
    except ValueError:
        return None


def parse_time(value: str) -> float | None:
    text = value.upper().replace(",", " ")
    numbers = [parse_float(token) for token in re.findall(r"[+\-]?\d+(?:\.\d*)?(?:[Ee][+\-]?\d+)?", text)]
    numbers = [number for number in numbers if number is not None]
    if not numbers:
        return None
    seconds = numbers[-1]
    if "MINUTE" in text and len(numbers) >= 2:
        seconds += 60.0 * numbers[-2]
    if "HOUR" in text and len(numbers) >= 3:
        seconds += 3600.0 * numbers[-3]
    return seconds


def collect_paths(args: argparse.Namespace) -> list[Path]:
    if args.files:
        paths = [Path(item) for item in args.files]
    else:
        paths = [Path(item) for item in DEFAULT_FILES]
    if args.include_tests:
        paths.extend(Path("tests").glob("**/*.out"))
        paths.extend(Path("tests").glob("**/*.arc"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "source_path",
        "source_type",
        "heat_kcal_mol",
        "wall_clock_s",
        "total_job_s",
        "has_gpu_keyword",
        "has_mozyme",
        "normal_end",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    rows = [row for path in collect_paths(args) if (row := parse_file(path)) is not None]
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    write_csv(out_csv, rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv} with {len(rows)} existing reference rows")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

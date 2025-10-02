#!/usr/bin/env python3
"""
Compare MOPAC gradient blocks ("CARTESIAN COORDINATE DERIVATIVES")
between two outputs (e.g., CPU vs GPU). Prints max absolute diff and RMS.

Usage:
  python compare_gradients.py gpu_test_logs/gradient_cpu_target.mopac_out \
                              gpu_test_logs/gradient_gpu_target.mopac_out
"""

from __future__ import annotations
import argparse
import math
import sys
from typing import List, Tuple, Optional


def ffloat(s: str) -> float:
    """Fortran 'D' exponent to Python float; tolerates stray spaces."""
    return float(s.replace("D", "E").replace("d", "e"))


def extract_gradients(path: str) -> List[Tuple[float, float, float]]:
    """
    Extract the XYZ gradient tuples from a MOPAC output file.

    We look for the section header:
      'CARTESIAN COORDINATE DERIVATIVES'
    Then find the header line containing tokens 'ATOM', 'X', 'Y', 'Z'
    to determine the XYZ column indices. We read numeric rows until a blank
    line after data has started.
    """
    data: List[Tuple[float, float, float]] = []
    capture = False
    cols: Optional[Tuple[int, int, int]] = None
    seen_data = False

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()

                # Arm capture when the section header appears
                if not capture and "CARTESIAN COORDINATE DERIVATIVES" in raw:
                    capture = True
                    cols = None
                    seen_data = False
                    continue

                if not capture:
                    continue

                # Find the header that defines column positions
                if cols is None:
                    if not line:
                        continue
                    parts = line.split()
                    # Require all tokens to be present in this header line
                    if all(tok in parts for tok in ("ATOM", "X", "Y", "Z")):
                        try:
                            x = parts.index("X")
                            y = parts.index("Y")
                            z = parts.index("Z")
                            cols = (x, y, z)
                        except ValueError:
                            # Keep scanning; malformed header line
                            cols = None
                    # Keep looping until we detect the header line
                    continue

                # After columns are known, stop at the first blank line
                if not line:
                    if seen_data:
                        break
                    else:
                        continue

                parts = line.split()

                # Rows usually start with an integer index in column 0.
                # If not numeric, skip (header/footnote noise).
                head = parts[0].lstrip("+-")
                if not head.isdigit():
                    continue

                # Ensure we have enough columns
                if max(cols) >= len(parts):
                    continue

                try:
                    xyz = tuple(ffloat(parts[i]) for i in cols)
                except ValueError:
                    # Non-numeric payload in data row; skip
                    continue

                data.append(xyz)  # (x, y, z)
                seen_data = True

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except OSError as e:
        raise OSError(f"Error reading {path}: {e}")

    if not data:
        raise ValueError(
            f"No gradient data found in {path}. "
            "Make sure it contains 'CARTESIAN COORDINATE DERIVATIVES'."
        )
    return data


def flatten_xyz(triples: List[Tuple[float, float, float]]) -> List[float]:
    out: List[float] = []
    for x, y, z in triples:
        out.extend((x, y, z))
    return out


def rms(values: List[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare MOPAC Cartesian coordinate derivatives between two outputs."
    )
    p.add_argument("cpu_file", help="Reference gradients (e.g., CPU)")
    p.add_argument("gpu_file", help="Target gradients (e.g., GPU)")
    args = p.parse_args()

    try:
        cpu_xyz = extract_gradients(args.cpu_file)
        gpu_xyz = extract_gradients(args.gpu_file)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if len(cpu_xyz) != len(gpu_xyz):
        print(
            f"WARNING: Different atom counts: CPU={len(cpu_xyz)} vs GPU={len(gpu_xyz)}. "
            "Comparing up to the shortest length.",
            file=sys.stderr,
        )

    n = min(len(cpu_xyz), len(gpu_xyz))
    cpu_flat = flatten_xyz(cpu_xyz[:n])
    gpu_flat = flatten_xyz(gpu_xyz[:n])

    diffs = [abs(a - b) for a, b in zip(cpu_flat, gpu_flat)]
    max_abs = max(diffs) if diffs else 0.0
    rms_val = rms(diffs)
    mean_abs = (sum(diffs) / len(diffs)) if diffs else 0.0

    print(f"atoms_compared={n} components={len(diffs)}")
    print(f"max_abs={max_abs:.6e}  rms={rms_val:.6e}  mean_abs={mean_abs:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


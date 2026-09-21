#!/usr/bin/env python3
"""Write a MOPAC deck for a water cluster of N molecules on a cubic lattice.

Rigid water molecules (O-H 0.9572 A, H-O-H 104.52 deg) are placed on a simple cubic
lattice (default spacing 3.1 A, the O-O distance of liquid water) with a
deterministic pseudo-random orientation per site.  Meant for size-scaling
benchmarks only (1SCF), not for physically meaningful energies.  7052 molecules give
42,312 orbitals (6 per water), the size of the largest system of Maia, Cabral and
Rocha, J Mol Model 26, 313 (2020).
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def water_geometry() -> list[tuple[str, float, float, float]]:
    roh = 0.9572
    half = math.radians(104.52 / 2.0)
    return [("O", 0.0, 0.0, 0.0),
            ("H", roh * math.sin(half), roh * math.cos(half), 0.0),
            ("H", -roh * math.sin(half), roh * math.cos(half), 0.0)]


def rotation(rng: random.Random) -> list[list[float]]:
    # random rotation from three Euler angles (uniform enough for a benchmark)
    a, b, c = (rng.uniform(0.0, 2.0 * math.pi) for _ in range(3))
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return [[ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
            [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
            [-sb, cb * sc, cb * cc]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("n_waters", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=3.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keywords", default="PM7 1SCF MOZYME MOZYME_GPU MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 XYZ GEO-OK NOCOMMENTS")
    args = parser.parse_args()
    n = args.n_waters
    side = int(math.ceil(n ** (1.0 / 3.0)))
    rng = random.Random(args.seed)
    template = water_geometry()
    lines = [args.keywords, f"Water cluster benchmark: {n} H2O on a {side}^3 cubic lattice, spacing {args.spacing} A", ""]
    placed = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if placed >= n:
                    break
                rot = rotation(rng)
                ox, oy, oz = i * args.spacing, j * args.spacing, k * args.spacing
                for el, x, y, z in template:
                    rx = rot[0][0] * x + rot[0][1] * y + rot[0][2] * z
                    ry = rot[1][0] * x + rot[1][1] * y + rot[1][2] * z
                    rz = rot[2][0] * x + rot[2][1] * y + rot[2][2] * z
                    lines.append(f"{el:<2} {ox + rx:12.6f} 1 {oy + ry:12.6f} 1 {oz + rz:12.6f} 1")
                placed += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {placed} waters, {3 * placed} atoms, {6 * placed} orbitals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

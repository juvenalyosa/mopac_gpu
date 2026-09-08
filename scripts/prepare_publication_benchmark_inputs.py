#!/usr/bin/env python3
"""Prepare medium/large molecule inputs for publication-scale GPU benchmarks."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


@dataclass(frozen=True)
class PdbCase:
    label: str
    pdb_id: str
    title: str
    category: str
    keyword_tail: str


PDB_CASES = [
    PdbCase(
        "protein_crambin_1crn",
        "1CRN",
        "Protein benchmark: crambin, PDB 1CRN",
        "protein",
        "1SCF MOZYME MOZYME_GPU MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 NEWPDB PDB GEO-OK NOCOMMENTS",
    ),
    PdbCase(
        "protein_ubiquitin_1ubq",
        "1UBQ",
        "Protein benchmark: ubiquitin, PDB 1UBQ",
        "protein",
        "1SCF MOZYME MOZYME_GPU MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 NEWPDB PDB GEO-OK NOCOMMENTS",
    ),
    PdbCase(
        "protein_adenylate_kinase_1ake",
        "1AKE",
        "Large protein benchmark: adenylate kinase, PDB 1AKE",
        "protein",
        "1SCF MOZYME MOZYME_GPU MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 NEWPDB PDB GEO-OK NOCOMMENTS",
    ),
    PdbCase(
        "dna_dodecamer_1bna",
        "1BNA",
        "DNA benchmark: Dickerson dodecamer, PDB 1BNA",
        "dna",
        "1SCF MOZYME MOZYME_GPU MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 NEWPDB PDB GEO-OK NOCOMMENTS",
    ),
    PdbCase(
        "rna_trna_1ehz",
        "1EHZ",
        "RNA benchmark: yeast phenylalanine tRNA, PDB 1EHZ",
        "rna",
        "1SCF MOZYME MOZYME_GPU MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 NEWPDB PDB GEO-OK NOCOMMENTS",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="benchmarks/publication_inputs")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-download", action="store_true", help="Use existing raw PDB files only.")
    return parser.parse_args()


def download_pdb(case: PdbCase, raw_dir: Path, force: bool, no_download: bool) -> Path:
    raw_path = raw_dir / f"{case.pdb_id}.pdb"
    if raw_path.exists() and not force:
        return raw_path
    if no_download:
        raise FileNotFoundError(f"Missing raw PDB and --no-download was set: {raw_path}")
    url = RCSB_URL.format(pdb_id=case.pdb_id)
    print(f"Downloading {case.pdb_id} from {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        raw_path.write_bytes(response.read())
    return raw_path


def clean_pdb(raw_path: Path, clean_path: Path) -> int:
    atoms = 0
    with raw_path.open("r", encoding="utf-8", errors="ignore") as src, clean_path.open("w", encoding="utf-8") as dst:
        for line in src:
            record = line[:6].strip()
            if record not in {"ATOM", "HETATM", "TER"}:
                continue
            if record == "HETATM":
                residue = line[17:20].strip().upper()
                if residue in {"HOH", "WAT", "DOD"}:
                    continue
            if record in {"ATOM", "HETATM"}:
                atoms += 1
            dst.write(line.rstrip() + "\n")
        dst.write("END\n")
    return atoms


def write_pdb_mop(case: PdbCase, clean_pdb: Path, mop_dir: Path) -> Path:
    mop_path = mop_dir / f"{case.label}.mop"
    keywords = f'PM7 GEO_DAT="{clean_pdb.name}" {case.keyword_tail}'
    text = "\n".join([keywords, case.title, ""]) + "\n"
    mop_path.write_text(text, encoding="utf-8")
    return mop_path


def write_graphene_mop(mop_dir: Path) -> tuple[str, Path, int]:
    atoms = graphene_flake(nx=12, ny=8)
    carbon_count = sum(1 for atom in atoms if atom[0] == "C")
    hydrogen_count = sum(1 for atom in atoms if atom[0] == "H")
    label = f"material_graphene_nanoflake_c{carbon_count}h{hydrogen_count}"
    mop_path = mop_dir / f"{label}.mop"
    lines = [
        "PM7 XYZ 1SCF PULAY BONDS GEO-OK",
        f"Material-like benchmark: hydrogen-terminated graphene nanoflake C{carbon_count}H{hydrogen_count}",
        "",
    ]
    for element, x, y, z in atoms:
        lines.append(f"{element:<2} {x:12.6f} {y:12.6f} {z:12.6f}")
    lines.append("")
    mop_path.write_text("\n".join(lines), encoding="utf-8")
    return label, mop_path, len(atoms)


def graphene_flake(nx: int, ny: int) -> list[tuple[str, float, float, float]]:
    bond = 1.42
    basis = [(0.0, 0.0), (math.sqrt(3.0) * bond / 2.0, 0.5 * bond)]
    a1 = (math.sqrt(3.0) * bond, 0.0)
    a2 = (math.sqrt(3.0) * bond / 2.0, 1.5 * bond)
    carbons: list[tuple[float, float]] = []
    for i in range(nx):
        for j in range(ny):
            ox = i * a1[0] + j * a2[0]
            oy = i * a1[1] + j * a2[1]
            for bx, by in basis:
                carbons.append((ox + bx, oy + by))

    cx = sum(x for x, _ in carbons) / len(carbons)
    cy = sum(y for _, y in carbons) / len(carbons)
    carbons = [(x - cx, y - cy) for x, y in carbons]
    atoms: list[tuple[str, float, float, float]] = [("C", x, y, 0.0) for x, y in carbons]

    edge_h: list[tuple[str, float, float, float]] = []
    for x, y in carbons:
        neighbors = 0
        for x2, y2 in carbons:
            if x == x2 and y == y2:
                continue
            dist = math.hypot(x - x2, y - y2)
            if 1.15 < dist < 1.65:
                neighbors += 1
        if neighbors < 3:
            norm = math.hypot(x, y) or 1.0
            hx = x + 1.09 * x / norm
            hy = y + 1.09 * y / norm
            edge_h.append(("H", hx, hy, 0.0))

    return atoms + edge_h


def write_readme(out_dir: Path, rows: list[tuple[str, str, int, Path]]) -> None:
    lines = [
        "# Publication Benchmark Inputs",
        "",
        "Prepared medium/large inputs for end-to-end MOPAC CPU/GPU publication benchmarks.",
        "PDB structures are downloaded from the RCSB Protein Data Bank and cleaned to keep",
        "coordinate records relevant for MOPAC single-point benchmarking.",
        "",
        "| Label | Category | Heavy/input atoms | MOPAC input |",
        "|---|---|---:|---|",
    ]
    for label, category, atoms, mop_path in rows:
        lines.append(f"| {label} | {category} | {atoms} | `{mop_path.name}` |")
    lines.extend(
        [
            "",
            "The DNA/RNA/protein inputs are intended as reproducible performance and CPU/GPU",
            "agreement benchmarks, not as curated thermochemical reference geometries.",
            "Before timing these cases, run `scripts/hydrogenate_publication_benchmark_inputs.py`",
            "with the built MOPAC executable; it creates hydrogenated geometries and rewrites",
            "the compute decks to reference them via `GEO_DAT`.",
            "The graphene nanoflake is generated locally as a material-like cluster benchmark.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "pdb_raw"
    clean_dir = out_dir / "pdb_clean"
    mop_dir = out_dir / "mop"
    for path in (raw_dir, clean_dir, mop_dir):
        path.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, int, Path]] = []
    for case in PDB_CASES:
        raw = download_pdb(case, raw_dir, args.force_download, args.no_download)
        clean = clean_dir / f"{case.label}.pdb"
        atom_count = clean_pdb(raw, clean)
        shutil.copy2(clean, mop_dir / clean.name)
        mop = write_pdb_mop(case, clean, mop_dir)
        rows.append((case.label, case.category, atom_count, mop))

    graphene_label, graphene, graphene_atoms = write_graphene_mop(mop_dir)
    rows.append((graphene_label, "material_cluster", graphene_atoms, graphene))
    write_readme(out_dir, rows)

    print(f"Prepared publication benchmark inputs in {out_dir}")
    for label, category, atoms, mop in rows:
        print(f"  {label:36s} {category:16s} atoms={atoms:5d} input={mop}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

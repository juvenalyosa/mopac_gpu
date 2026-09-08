#!/usr/bin/env python3
"""Hydrogenate publication PDB inputs with MOPAC before timing CPU/GPU runs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_KEYWORD_TAIL = "1SCF MOZYME MOZYME_GPU MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 NEWPDB PDB GEO-OK NOCOMMENTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mopac", help="MOPAC executable used for ADD-H preparation.")
    parser.add_argument("--inputs-dir", default="benchmarks/publication_inputs")
    parser.add_argument("--only", action="append", default=[], help="Hydrogenate only labels containing this text.")
    parser.add_argument("--force", action="store_true", help="Re-run ADD-H even if a hydrogenated geometry exists.")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--keep-runs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mopac = Path(args.mopac).resolve()
    root = Path(args.inputs_dir)
    clean_dir = root / "pdb_clean"
    mop_dir = root / "mop"
    prep_dir = root / "hydrogenation_runs"
    hydro_dir = root / "hydrogenated"

    if not mopac.exists():
        raise SystemExit(f"MOPAC executable not found: {mopac}")
    if not clean_dir.exists():
        raise SystemExit(f"Clean PDB directory not found: {clean_dir}")

    prep_dir.mkdir(parents=True, exist_ok=True)
    hydro_dir.mkdir(parents=True, exist_ok=True)
    mop_dir.mkdir(parents=True, exist_ok=True)

    labels = [path.stem for path in sorted(clean_dir.glob("*.pdb"))]
    if args.only:
        labels = [label for label in labels if any(token in label for token in args.only)]
    if not labels:
        raise SystemExit("No PDB labels selected for hydrogenation.")

    failures: list[str] = []
    for label in labels:
        clean_pdb = clean_dir / f"{label}.pdb"
        hydro_pdb = hydro_dir / f"{label}_hydrogenated.pdb"
        hydro_arc = hydro_dir / f"{label}_hydrogenated.arc"
        compute_mop = mop_dir / f"{label}.mop"

        if hydro_pdb.exists() and not args.force:
            write_compute_deck(compute_mop, hydro_pdb.name, label)
            shutil.copy2(hydro_pdb, mop_dir / hydro_pdb.name)
            print(f"[SKIP] {label}: using existing {hydro_pdb}")
            continue

        run_dir = prep_dir / label
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clean_pdb, run_dir / clean_pdb.name)

        addh_mop = run_dir / f"{label}_addh.mop"
        addh_mop.write_text(
            "\n".join(
                [
                    f'PM7 ADD-H PDBOUT NEWPDB 0SCF GEO_DAT="{clean_pdb.name}" GEO-OK NOCOMMENTS',
                    f"Hydrogenation preparation for {label}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        print(f"[ADD-H] {label}: running MOPAC hydrogenation", flush=True)
        proc = subprocess.run(
            [str(mopac), addh_mop.name],
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        (hydro_dir / f"{label}_addh.stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="ignore")
        copy_outputs(run_dir, hydro_dir, label)

        produced = choose_hydrogenated_geometry(run_dir, label)
        if proc.returncode != 0 or produced is None:
            failures.append(label)
            print(f"[FAIL] {label}: ADD-H did not produce a usable PDB/ARC")
            continue

        if produced.suffix.lower() == ".pdb":
            shutil.copy2(produced, hydro_pdb)
            shutil.copy2(hydro_pdb, mop_dir / hydro_pdb.name)
            write_compute_deck(compute_mop, hydro_pdb.name, label)
            print(f"[OK] {label}: wrote {hydro_pdb}")
        else:
            shutil.copy2(produced, hydro_arc)
            shutil.copy2(hydro_arc, mop_dir / hydro_arc.name)
            write_compute_deck(compute_mop, hydro_arc.name, label)
            print(f"[OK] {label}: wrote {hydro_arc}")

        if not args.keep_runs:
            shutil.rmtree(run_dir, ignore_errors=True)

    write_readme(root, labels)
    if failures:
        raise SystemExit("Hydrogenation failed for: " + ", ".join(failures))
    print("Hydrogenated publication benchmark inputs are ready.")
    return 0


def choose_hydrogenated_geometry(run_dir: Path, label: str) -> Path | None:
    pdb_candidates = sorted(run_dir.glob(f"{label}_addh*.pdb")) + sorted(run_dir.glob("*.pdb"))
    for path in pdb_candidates:
        if path.name != f"{label}.pdb" and has_atoms(path):
            return path
    arc_candidates = sorted(run_dir.glob(f"{label}_addh*.arc")) + sorted(run_dir.glob("*.arc"))
    for path in arc_candidates:
        if path.stat().st_size > 0:
            return path
    return None


def has_atoms(path: Path) -> bool:
    try:
        return any(line.startswith(("ATOM", "HETATM")) for line in path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError:
        return False


def copy_outputs(run_dir: Path, target_dir: Path, label: str) -> None:
    for suffix in ("*.out", "*.arc", "*.pdb", "*.log"):
        for path in run_dir.glob(suffix):
            target = target_dir / f"{label}_addh{path.suffix}"
            try:
                shutil.copy2(path, target)
            except OSError:
                pass


def write_compute_deck(path: Path, geometry_name: str, label: str) -> None:
    path.write_text(
        "\n".join(
            [
                f'PM7 GEO_DAT="{geometry_name}" {DEFAULT_KEYWORD_TAIL}',
                f"Publication benchmark compute deck for {label}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_readme(root: Path, labels: list[str]) -> None:
    path = root / "hydrogenated" / "README.md"
    lines = [
        "# Hydrogenated Publication Inputs",
        "",
        "These geometries were generated by running MOPAC `ADD-H PDBOUT NEWPDB 0SCF`",
        "on the cleaned RCSB PDB files before CPU/GPU timing. The benchmark `.mop`",
        "files in `mop/` reference these hydrogenated geometries through `GEO_DAT`.",
        "",
        "Prepared labels:",
        "",
    ]
    lines.extend(f"- `{label}`" for label in labels)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

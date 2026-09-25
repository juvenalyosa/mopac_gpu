#!/usr/bin/env python3
"""MOZYME (GPU) workflow: prepare a protein PDB, optimize it, then NVT and NVE dynamics with DRC.

Steps (each one is a function, the Colab notebook colab/mozyme_md_colab.ipynb calls them):

  prepare_pdb     clean a PDB (waters, alternate conformers, half groups on crystallographic
                  special positions) and hydrogenate it with MOPAC ADD-H (optionally
                  SITE=(IONIZE) for pH ~7); MOPAC's INPUT CHEMISTRY CHECK must pass
  optimize        geometry optimization (CYCLES=n); the final geometry is written as a PDB
                  (from the .arc when the optimization converged, else RESTART 1SCF PDBOUT
                  from the restart file)
  dynamics        DRC molecular dynamics from Maxwell-Boltzmann velocities (TEMPERATURE=T):
                  NVT adds the Bussi thermostat (BUSSI=tau), NVE keeps the energy constant;
                  returns the DRC table, the temperature log and the trajectory (.xyz);
                  last_frame_pdb writes the final frame as a PDB

Implicit solvent: pass eps=78.4 (COSMO, water).  No barostat exists (NPT): MOZYME runs a
finite molecule, with implicit solvent there is no box and no pressure.

Command line (runs the whole chain):
  mozyme_md_workflow.py <mopac> <input.pdb> --work-dir DIR [--temperature 300]
      [--opt-cycles 100] [--nvt-points 40] [--nve-points 40] [--eps 78.4] [--ionize]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_publication_benchmark_inputs import clean_pdb  # noqa: E402

HEAT_RE = re.compile(r"(?:FINAL HEAT OF FORMATION|CURRENT VALUE OF HEAT OF FORMATION)\s*=\s*([-+0-9.EeDd]+)")
CYCLE_RE = re.compile(r"CYCLE:\s*(\d+)\s+TIME:\s*([0-9.]+).*?GRAD\.:\s*([-+0-9.EeDd]+)\s+HEAT:\s*([-+0-9.EeDd]+)")
STATUS_RE = re.compile(r"\[MOZYME GPU SCF\]\s+status=(\S+)")
ENSEMBLE_RE = re.compile(r"TEMPERATURE: STEP\s+(\d+)\s+TIME\(FS\)\s+([-0-9.]+)\s+T\(K\)\s+([-0-9.]+)\s+<T>\(K\)\s+([-0-9.]+)"
                         r"\s+ENERGY TO BATH \(KCAL/MOL\)\s+([-0-9.]+)")
FLOAT_RE = re.compile(r"-?\d+\.\d+")
CHECK_ERROR = "INPUT CHEMISTRY CHECK FOUND ERRORS"
BASE_KEYS = "PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK"


@dataclass
class RunResult:
    deck: Path
    out_text: str
    wall: float
    heat: float | None
    statuses: list[str] = field(default_factory=list)

    @property
    def gpu_ok(self) -> bool:
        return bool(self.statuses) and all(s == "success" for s in self.statuses)


def run_mopac(mopac: Path, run_dir: Path, name: str, keywords: str, title: str, files: list[Path],
              env_extra: dict[str, str] | None = None, timeout: float = 86400.0) -> RunResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.resolve() != (run_dir / f.name).resolve():
            shutil.copy2(f, run_dir / f.name)
    deck = run_dir / f"{name}.mop"
    deck.write_text(f"{keywords}\n{title}\n\n", encoding="utf-8")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.perf_counter()
    proc = subprocess.run([str(mopac), deck.name], cwd=run_dir, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    out = run_dir / f"{name}.out"
    text = proc.stdout + ("\n" + out.read_text(encoding="utf-8", errors="ignore") if out.exists() else "")
    heats = HEAT_RE.findall(text)
    heat = float(heats[-1].replace("D", "E")) if heats else None
    return RunResult(deck, text, wall, heat, STATUS_RE.findall(text))


def solvent_keys(eps: float | None) -> str:
    return f" EPS={eps:g}" if eps else ""


def prepare_pdb(mopac: Path, raw_pdb: Path, work: Path, ionize: bool = False) -> Path:
    """Clean + ADD-H.  Returns the hydrogenated PDB; raises if MOPAC's input check fails."""
    work.mkdir(parents=True, exist_ok=True)
    clean = work / f"{raw_pdb.stem}_clean.pdb"
    n = clean_pdb(raw_pdb, clean)
    print(f"[prepare] {raw_pdb.name}: {n} atoms after cleaning", flush=True)
    site = " SITE=(IONIZE)" if ionize else ""
    name = f"{raw_pdb.stem}_addh"
    r = run_mopac(mopac, work, name, f'PM7 ADD-H{site} PDBOUT NEWPDB 0SCF GEO_DAT="{clean.name}" GEO-OK NOCOMMENTS',
                  "hydrogenation", [clean])
    if CHECK_ERROR in r.out_text:
        errors = [l.strip() for l in r.out_text.splitlines() if l.strip().startswith("ERROR ")]
        raise RuntimeError("MOPAC input chemistry check failed:\n  " + "\n  ".join(errors))
    produced = work / f"{name}.pdb"
    if not produced.exists():
        raise RuntimeError(f"ADD-H did not write {produced.name}; see {name}.out")
    hydro = work / f"{raw_pdb.stem}_hydrogenated.pdb"
    shutil.copy2(produced, hydro)
    atoms = sum(1 for l in hydro.read_text().splitlines() if l.startswith(("ATOM", "HETATM")))
    charge = re.findall(r"COMPUTED CHARGE ON SYSTEM:\s*([-+]?\d+)", r.out_text)
    print(f"[prepare] hydrogenated: {atoms} atoms{', charge ' + charge[-1] if charge else ''} -> {hydro.name}",
          flush=True)
    return hydro


def optimize(mopac: Path, pdb: Path, work: Path, cycles: int = 100, eps: float | None = None,
             env_extra: dict[str, str] | None = None, extra: str = "") -> tuple[Path, RunResult]:
    """Geometry optimization; returns (optimized PDB, run).  extra: more keywords, e.g. "GNORM=0.5"."""
    name = "opt"
    extra = f" {extra.strip()}" if extra.strip() else ""
    r = run_mopac(mopac, work, name,
                  f'{BASE_KEYS}{solvent_keys(eps)} GEO_DAT="{pdb.name}" CYCLES={cycles} PDBOUT{extra}',
                  "geometry optimization", [pdb], env_extra)
    cycles_seen = CYCLE_RE.findall(r.out_text)
    if cycles_seen:
        first, last = cycles_seen[0], cycles_seen[-1]
        print(f"[opt] {len(cycles_seen)} cycles in {r.wall:.1f} s: heat {float(first[3]):.3f} -> {float(last[3]):.3f}, "
              f"GRAD {float(first[2]):.1f} -> {float(last[2]):.1f}; GPU SCF all resident: {r.gpu_ok}", flush=True)
    final = work / f"{name}.pdb"
    if final.exists() and "JOB ENDED NORMALLY" in r.out_text and (work / f"{name}.arc").exists():
        return final, r
    res = work / f"{name}.res"
    if not res.exists():
        raise RuntimeError("optimization left neither a final geometry nor a restart file; see opt.out")
    rdir = work / "opt_final"
    rdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(res, rdir / "opt_final.res")
    rr = run_mopac(mopac, rdir, "opt_final", f'{BASE_KEYS}{solvent_keys(eps)} GEO_DAT="{pdb.name}" 1SCF RESTART PDBOUT',
                   "final geometry of the optimization", [pdb], env_extra)
    out_pdb = rdir / "opt_final.pdb"
    if not out_pdb.exists():
        raise RuntimeError("RESTART 1SCF PDBOUT did not write the final geometry")
    print(f"[opt] cycle limit reached; final geometry from the restart file: heat {rr.heat}", flush=True)
    return out_pdb, r


@dataclass
class Dynamics:
    run: RunResult
    rows: list[tuple[float, float, float, float, float]]   # fs, potential, kinetic, total, error
    temps: list[tuple[int, float, float, float, float]]    # step, fs, T, <T>, E_to_bath
    xyz: Path | None


def drc_rows(text: str) -> list[tuple[float, float, float, float, float]]:
    rows = []
    started = False
    for line in text.splitlines():
        if "FEMTOSECONDS  POINT  POTENTIAL" in line:
            started = True
            continue
        if not started:
            continue
        head = line.split()
        if len(head) < 2 or not re.fullmatch(r"-?\d+\.\d+", head[0]) or not head[1].isdigit():
            continue
        v = [float(x) for x in FLOAT_RE.findall(line)]
        if len(v) >= 5:
            rows.append((v[0], v[1], v[2], v[3], v[4]))
    return rows


def dynamics(mopac: Path, pdb: Path, work: Path, ensemble: str, temperature: float, points: int,
             interval_fs: float = 0.5, tau_fs: float = 100.0, eps: float | None = None, seed: int = 1,
             env_extra: dict[str, str] | None = None) -> Dynamics:
    """DRC dynamics; ensemble 'NVT' or 'NVE'.  points = number of table rows (interval_fs apart)."""
    ens = ensemble.upper()
    if ens not in ("NVT", "NVE"):
        raise ValueError("ensemble must be NVT or NVE (no barostat exists for NPT)")
    extra = f" TEMPERATURE={temperature:g}" + (f" BUSSI={tau_fs:g}" if ens == "NVT" else "")
    name = ens.lower()
    r = run_mopac(mopac, work, name,
                  f'{BASE_KEYS}{solvent_keys(eps)} GEO_DAT="{pdb.name}" DRC{extra} T-PRIORITY={interval_fs:g} '
                  f'CYCLES={points} SEED={seed}',
                  f"{ens} dynamics at {temperature:g} K", [pdb], env_extra)
    rows = drc_rows(r.out_text)
    temps = [(int(a), float(b), float(c), float(d), float(e)) for a, b, c, d, e in ENSEMBLE_RE.findall(r.out_text)]
    xyz = work / f"{name}.xyz"
    natoms = sum(1 for l in pdb.read_text().splitlines() if l.startswith(("ATOM", "HETATM")))
    if rows:
        errs = [abs(x[4]) for x in rows[3:]]
        t_pts = temperatures(rows, natoms)[1:]
        t_avg = sum(t_pts) / len(t_pts) if t_pts else float("nan")
        print(f"[{name}] {len(rows)} points to {rows[-1][0]:.1f} fs in {r.wall:.1f} s "
              f"({r.wall / max(1, len(rows) - 1):.2f} s per {interval_fs:g} fs); "
              f"max |ERROR| {max(errs) if errs else 0:.3f} kcal/mol; mean T {t_avg:.1f} K; "
              f"GPU SCF all resident: {r.gpu_ok}", flush=True)
    return Dynamics(r, rows, temps, xyz if xyz.exists() else None)


def temperatures(rows: list[tuple[float, float, float, float, float]], natoms: int) -> list[float]:
    """Instantaneous temperature (K) of every DRC row from its kinetic energy, 3N-3 degrees of freedom."""
    kb = 0.0019872041
    ndof = max(1, 3 * natoms - 3)
    return [2.0 * r[2] / (ndof * kb) for r in rows]


def xyz_frames(xyz: Path) -> list[list[tuple[str, float, float, float]]]:
    lines = xyz.read_text(encoding="utf-8", errors="ignore").splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        atoms = []
        for line in lines[i + 2:i + 2 + n]:
            p = line.split()
            atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
        frames.append(atoms)
        i += 2 + n
    return frames


def last_frame_pdb(xyz: Path, template_pdb: Path, out_pdb: Path) -> Path:
    """Final DRC frame written into the template PDB's records (same atom order as the input)."""
    frame = xyz_frames(xyz)[-1]
    records = [l for l in template_pdb.read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
    if len(records) != len(frame):
        raise RuntimeError(f"atom count mismatch: {len(records)} PDB records vs {len(frame)} in the trajectory")
    out = []
    for rec, (_el, x, y, z) in zip(records, frame):
        out.append(f"{rec[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{rec[54:]}")
    out_pdb.write_text("\n".join(out) + "\nEND\n", encoding="utf-8")
    return out_pdb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mopac", type=Path)
    ap.add_argument("pdb", type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path("mozyme_md"))
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--opt-cycles", type=int, default=100)
    ap.add_argument("--nvt-points", type=int, default=40)
    ap.add_argument("--nve-points", type=int, default=40)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--ionize", action="store_true")
    ap.add_argument("--already-hydrogenated", action="store_true")
    a = ap.parse_args()
    mopac = a.mopac.resolve()
    work = a.work_dir.resolve()
    pdb = a.pdb.resolve() if a.already_hydrogenated else prepare_pdb(mopac, a.pdb.resolve(), work / "prepare", a.ionize)
    opt_pdb, _ = optimize(mopac, pdb, work / "opt", a.opt_cycles, a.eps)
    nvt = dynamics(mopac, opt_pdb, work / "nvt", "NVT", a.temperature, a.nvt_points, eps=a.eps)
    start = opt_pdb
    if nvt.xyz:
        start = last_frame_pdb(nvt.xyz, opt_pdb, work / "nvt" / "nvt_last.pdb")
    dynamics(mopac, start, work / "nve", "NVE", a.temperature, a.nve_points, eps=a.eps, seed=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

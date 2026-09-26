#!/usr/bin/env python3
"""FORCE (vibrational frequencies / THERMO) with the Hessian rows computed by several MOPAC
processes at the same time, using only standard MOPAC keywords.

MOPAC builds the Hessian one Cartesian coordinate at a time (src/forces/fmat.F90): coordinate i
adds its finite-difference gradient change to row i and column i of the packed force matrix,
so the matrix is a SUM of independent per-coordinate contributions.  `FORCE CYCLES=n` stops
after n coordinates and writes the partial matrix to the restart file (src/forces/forsav.F90),
and `FORCE RESTART` continues from the line stored there.  This driver:

  1. runs `FORCE CYCLES=1` once (template: the restart file with the right geometry, sizes and
     the contribution of coordinate 1);
  2. writes one restart file per worker that starts at the worker's first coordinate with a
     zero force matrix, and runs the workers concurrently with `FORCE RESTART CYCLES=<n>`;
  3. adds the partial force matrices, copies each worker's dipole derivatives, writes a restart
     file whose line counter is the last coordinate, and runs `FORCE RESTART THERMO(...)`:
     MOPAC finds nothing left to compute and goes straight to the frequencies and THERMO.

Several processes share one GPU (the resident SCF of a 600-atom protein leaves most of an A100
idle); `nvidia-cuda-mps-control -d` lets their kernels overlap instead of time-slicing.

Usage:
  mozyme_parallel_force.py <mopac> <geometry.pdb> --keywords "PM7 MOZYME ... THERMO(298)" \\
      --workers 4 --work-dir DIR
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

MAX_RECORD = 2**31 - 9   # gfortran splits longer records into sub-records (not handled here)


# ---------------- gfortran unformatted sequential records ----------------
def read_records(path: Path) -> list[bytes]:
    data = path.read_bytes()
    out, pos = [], 0
    while pos < len(data):
        (n,) = struct.unpack_from("<i", data, pos)
        if n < 0:
            raise NotImplementedError("record longer than 2 GB (gfortran sub-records) in " + str(path))
        out.append(data[pos + 4:pos + 4 + n])
        pos += 8 + n
    return out


def write_records(path: Path, records: list[bytes]) -> None:
    with path.open("wb") as fh:
        for rec in records:
            if len(rec) > MAX_RECORD:
                raise NotImplementedError("record longer than 2 GB")
            fh.write(struct.pack("<i", len(rec)) + rec + struct.pack("<i", len(rec)))


class ForceRestart:
    """The six records written by forsav: header, coordinates, packed force matrix, dipole
    derivatives of the lines done, evecs, jstart + fconst."""

    def __init__(self, path: Path):
        rec = read_records(path)
        if len(rec) != 6:
            raise ValueError(f"{path}: {len(rec)} records, a FORCE restart file has 6")
        self.time, self.ipt, self.refh, self.numat, self.norbs = struct.unpack("<didii", rec[0])
        self.nvar = len(rec[1]) // 8
        self.coord = rec[1]
        n_f = self.nvar * (self.nvar + 1) // 2
        self.fmatrx = list(struct.unpack(f"<{n_f}d", rec[2]))
        self.deldip = list(struct.unpack(f"<{3 * self.ipt}d", rec[3])) if self.ipt > 0 else []
        self.evecs = rec[4]
        self.jstart = struct.unpack_from("<i", rec[5])[0]
        self.fconst = rec[5][4:]

    def write(self, path: Path, ipt: int, fmatrx: list[float], deldip: list[float], time_s: float) -> None:
        write_records(path, [
            struct.pack("<didii", time_s, ipt, self.refh, self.numat, self.norbs),
            self.coord,
            struct.pack(f"<{len(fmatrx)}d", *fmatrx),
            struct.pack(f"<{len(deldip)}d", *deldip),
            self.evecs,
            struct.pack("<i", self.jstart) + self.fconst,
        ])


# ---------------- driver ----------------
def deck(path: Path, keywords: str, title: str) -> None:
    path.write_text(f"{keywords}\n{title}\n\n", encoding="utf-8")


def launch(mopac: Path, run_dir: Path, name: str, env: dict) -> subprocess.Popen:
    return subprocess.Popen([str(mopac), f"{name}.mop"], cwd=run_dir, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def parallel_force(mopac: Path, geometry: Path, keywords: str, workers: int, work: Path,
                   env_extra: dict | None = None) -> tuple[Path, dict]:
    """Returns (final .out file, timings).  `keywords` must contain FORCE's usual companions
    (method, MOZYME, THERMO(...), SCFCRT...) but not FORCE, CYCLES or RESTART."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    work.mkdir(parents=True, exist_ok=True)
    kw = f'{keywords} GEO_DAT="{geometry.name}"'
    timings = {}

    # 1. template: coordinate 1 and the restart file layout
    t0 = time.perf_counter()
    tdir = work / "template"
    tdir.mkdir(exist_ok=True)
    shutil.copy2(geometry, tdir / geometry.name)
    deck(tdir / "force.mop", f"{kw} FORCE CYCLES=1", "parallel FORCE template (coordinate 1)")
    if launch(mopac, tdir, "force", env).wait() != 0 or not (tdir / "force.res").exists():
        raise RuntimeError(f"template run failed, see {tdir / 'force.out'}")
    tpl = ForceRestart(tdir / "force.res")
    if tpl.ipt != 1:
        raise RuntimeError(f"template stopped at line {tpl.ipt}, expected 1")
    nvar = tpl.nvar
    timings["template_s"] = time.perf_counter() - t0

    # 2. workers over coordinates 2..nvar
    todo = list(range(2, nvar + 1))
    workers = max(1, min(workers, len(todo)))
    size = -(-len(todo) // workers)
    chunks = [todo[i:i + size] for i in range(0, len(todo), size)]
    zero_f = [0.0] * len(tpl.fmatrx)
    procs = []
    t0 = time.perf_counter()
    for k, chunk in enumerate(chunks):
        wdir = work / f"worker_{k:02d}"
        wdir.mkdir(exist_ok=True)
        shutil.copy2(geometry, wdir / geometry.name)
        first = chunk[0]
        tpl.write(wdir / "force.res", first - 1, zero_f, [0.0] * (3 * (first - 1)), 0.0)
        deck(wdir / "force.mop", f"{kw} FORCE RESTART CYCLES={len(chunk)}",
             f"parallel FORCE worker {k}: coordinates {first}-{chunk[-1]}")
        procs.append((wdir, chunk, launch(mopac, wdir, "force", env)))
    for wdir, chunk, proc in procs:
        if proc.wait() != 0:
            raise RuntimeError(f"worker failed, see {wdir / 'force.out'}")
    timings["workers_s"] = time.perf_counter() - t0

    # 3. merge and finish
    fsum = list(tpl.fmatrx)
    deldip = list(tpl.deldip) + [0.0] * (3 * (nvar - 1))
    total_time = tpl.time
    for wdir, chunk, _ in procs:
        part = ForceRestart(wdir / "force.res")
        if part.ipt != chunk[-1]:
            raise RuntimeError(f"{wdir}: stopped at line {part.ipt}, expected {chunk[-1]}")
        fsum = [a + b for a, b in zip(fsum, part.fmatrx)]
        lo = 3 * (chunk[0] - 1)
        deldip[lo:3 * chunk[-1]] = part.deldip[lo:3 * chunk[-1]]
        total_time += part.time
    t0 = time.perf_counter()
    fdir = work / "final"
    fdir.mkdir(exist_ok=True)
    shutil.copy2(geometry, fdir / geometry.name)
    tpl.write(fdir / "force.res", nvar, fsum, deldip, total_time)
    deck(fdir / "force.mop", f"{kw} FORCE RESTART", f"parallel FORCE: {len(chunks)} workers, merged")
    if launch(mopac, fdir, "force", env).wait() != 0:
        raise RuntimeError(f"final run failed, see {fdir / 'force.out'}")
    timings["final_s"] = time.perf_counter() - t0
    timings["workers"] = len(chunks)
    timings["coordinates"] = nvar
    return fdir / "force.out", timings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mopac", type=Path)
    ap.add_argument("geometry", type=Path)
    ap.add_argument("--keywords", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--work-dir", type=Path, default=Path("parallel_force"))
    a = ap.parse_args()
    t0 = time.perf_counter()
    out, timings = parallel_force(a.mopac.resolve(), a.geometry.resolve(), a.keywords, a.workers,
                                  a.work_dir.resolve())
    print(f"final output: {out}")
    print(f"wall {time.perf_counter() - t0:.1f} s: " + ", ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}"
                                                           for k, v in timings.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

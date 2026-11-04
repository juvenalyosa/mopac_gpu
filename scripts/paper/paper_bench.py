#!/usr/bin/env python3
"""Benchmarks for the Barranquilla MOPAC paper: speed and accuracy of every GPU path against the CPU.

    paper_bench.py --mopac build/mopac --out RESULTS [--libmopac build/libmopac.so]
                   [--exp all | env,mozyme_sp,...] [--repeats 3] [--quick] [--no-cpu]

Each MOPAC run goes to RESULTS/runs/<experiment>/<case>/<mode>[_r<k>]/ with its input, output and a
result.json; a run whose result.json exists is read back instead of repeated (delete the directory to redo
it), so an interrupted session resumes where it stopped.  Every experiment writes RESULTS/data/<name>.csv
(one row per run, plus detail tables: optimization cycles, dynamics rows, frequencies, profiles), which
paper_figures.py turns into the figures and tables.

Modes: cpu = the same binary with MOPAC_NOGPU=1; MOZYME on one core (it has no threading) and the
conventional SCF with all cores (OpenMP + BLAS); gpu = the defaults (MOPAC_GPU_DEBUG=1 for the status lines;
MOPAC_FORCEGPU=1 for cells under 100 atoms).  GPU timings are repeated --repeats times (median in the
figures); the CPU runs once.

Experiments (--exp):
  env           hardware and software of the session (env.json)
  mozyme_sp     MOZYME 1SCF: proteins and DNA (642-6689 atoms) and water clusters (375-21156 atoms)
  mozyme_grad   MOZYME 1SCF GRADIENTS: heat and gradient GPU vs CPU
  methods       MNDO, AM1, PM3, RM1, PM6, PM6-D3H4, PM7 (crambin, DNA): heat GPU vs CPU
  cosmo         MOZYME with COSMO (EPS=78.4)
  opt           crambin optimization (100 cycles) CPU and GPU, 1AKE (50 cycles) GPU; heats at the same geometry
  md            crambin NVE CPU and GPU, crambin NVT and 1AKE NVE on the GPU
  thermo        chignolin FORCE and crambin FORCETS on the GPU against the stored CPU references
  solids        conventional SCF of solids: Si supercells 64-1728 atoms and 9 other cells, 1SCF GRADIENTS
  solids_profile  where the time of a 512-768-atom cell goes (MOPAC_GPU_PROFILE stage timers)
  solids_force  diamond FORCE (Gamma-point vibrations) CPU and GPU
  solids_drc    diamond DRC TEMPERATURE=300 CPU and GPU
  conv_mol      conventional SCF of molecules: crambin RHF / UHF / COSMO, graphene flake, chignolin C.I.
  qmmm          OpenMM + MOPAC QM/MM MD: QM regions of 16-1231 atoms in water, MOPAC on the GPU and the CPU
  qmmm_enzyme   trypsin-BPTI (2PTC): cut-off vs QM-PME, time per step and the cut-off test
  mozyme_threads  MOZYME on the CPU with 1, 4 and all threads (is the one-thread reference fair?)
  upstream      MOPAC 23.1.2 (--upstream-mopac) against the CPU path of this fork: heats, gradients, iterations, time
  qmmm_pme_check  QM-MM PME against a direct Ewald sum (tests/check_qmmm_pme_ewald.py)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUB = REPO / "benchmarks" / "publication_inputs" / "mop"
MAT = REPO / "tests" / "data" / "materials"
REF = REPO / "tests" / "data" / "mozyme_gpu_reference"

HEAT_RE = re.compile(r"FINAL HEAT OF FORMATION\s*=\s*([-+0-9.EeDd]+)")
JOB_RE = re.compile(r"TOTAL JOB TIME:\s*([0-9.]+)\s*SECONDS")
ATOMS_RE = re.compile(r"Empirical Formula:.*?=\s*(\d+)\s+atoms")
STATUS_RE = re.compile(r"\[MOZYME GPU SCF\] status=(\S+) reason=\S+.*?iterations=(\d+)")
ITER_RE = re.compile(r"TIME FOR ITERATION:")
GNORM_RE = re.compile(r"GRADIENT NORM\s*=\s*([0-9.]+)")
AUX_GRAD_RE = re.compile(r"GRADIENTS:KCAL/MOL/ANGSTROM\[\d+\]=\s*(.*?)(?=\n\s*[A-Z_]+[:=\[])", re.S)
CYCLE_RE = re.compile(r"CYCLE:\s*(\d+)\s+TIME:\s*([0-9.]+).*?GRAD\.:\s*([-+0-9.EeDd]+)\s+HEAT:\s*([-+0-9.EeDd]+)")
TEMP_RE = re.compile(r"TEMPERATURE: STEP\s+(\d+)\s+TIME\(FS\)\s+([0-9.]+)\s+T\(K\)\s+([0-9.]+)\s+<T>\(K\)\s+([0-9.]+)"
                     r"\s+ENERGY TO BATH \(KCAL/MOL\)\s+([-0-9.]+)")
FLOAT_RE = re.compile(r"-?\d+\.\d+")
SECTION_RE = re.compile(r"\[PROFILE\] MOZYME_SECTION name=(\S+) calls=(\d+) ms=([0-9.]+)")
TIMES_RE = re.compile(r"\s+(BEFORE|AFTER)\s+(.+?)\s+CPU INTERVAL:.*WALL-CLOCK INTERVAL:\s+([\d.]+)")
GEO_REF_RE = re.compile(r'GEO_DAT\s*=\s*"([^"]+)"')
CPU_ENV_OFF = ("MOPAC_NOGPU", "MOZYME_GPU_OFF", "MOPAC_FORCEGPU", "MOPAC_GPU_PROFILE")
METHODS = ("MNDO", "AM1", "PM3", "RM1", "PM6", "PM6-D3H4", "PM7")
BASE_MOZ = "MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK"

# (case, deck stem in benchmarks/publication_inputs/mop, label, kind)
PROTEINS = [
    ("crambin", "protein_crambin_1crn", "Crambin (1CRN)", "protein"),
    ("dna", "dna_dodecamer_1bna", "DNA dodecamer (1BNA)", "nucleic acid"),
    ("ubiquitin", "protein_ubiquitin_1ubq", "Ubiquitin (1UBQ)", "protein"),
    ("bpti", "protein_1g6x_noions", "BPTI (1G6X)", "protein"),
    ("barnase", "protein_barnase_1rnb", "Barnase (1RNB)", "protein"),
    ("antifreeze", "protein_antifreeze_1ezg", "Antifreeze protein (1EZG)", "protein"),
    ("bacteriorhodopsin", "protein_bacteriorhodopsin_1c3w", "Bacteriorhodopsin (1C3W)", "protein"),
    ("adk", "protein_adenylate_kinase_1ake_apo", "Adenylate kinase (1AKE)", "protein"),
]
WATERS = (125, 250, 500, 1000, 2000, 4000, 7052)


# ----------------------------------------------------------------------------------------------- helpers

def say(msg: str) -> None:
    print(msg, flush=True)


class GpuMemory:
    """Peak of nvidia-smi memory.used (MiB) while active; None without nvidia-smi."""

    def __init__(self) -> None:
        self.peak = None
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.split()
                if out and out[0].strip().isdigit():
                    self.peak = max(self.peak or 0, int(out[0]))
            except Exception:
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self.peak = None
        if self.available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread:
            self._stop.set()
            self._thread.join()


def fnum(text: str | None):
    if text is None:
        return None
    try:
        return float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def last(regex: re.Pattern, text: str):
    m = regex.findall(text)
    return m[-1] if m else None


def drc_rows(text: str) -> list[tuple]:
    """(fs, point, potential, kinetic, total, error) of every DRC printout row."""
    rows, in_table = [], False
    for line in text.splitlines():
        if "FEMTOSECONDS  POINT  POTENTIAL" in line:
            in_table = True
            continue
        if not in_table:
            continue
        head = line.split()
        if len(head) < 2 or not re.fullmatch(r"-?\d+\.\d+", head[0]) or not head[1].isdigit():
            continue
        values = [float(v) for v in FLOAT_RE.findall(line)]
        if len(values) >= 5:
            fs, potential, kinetic, total, error = values[:5]
            rows.append((fs, int(head[1]), potential, kinetic, total, error))
    return rows


def frequencies(text: str) -> list[float]:
    sys.path.insert(0, str(REPO / "tests"))
    import check_mozyme_gpu_thermo as th  # noqa: E402
    return th.frequencies(text)


def stage_times(text: str) -> dict:
    """Wall-clock seconds summed per TIMES stage ('AFTER X' = stage X, 'BEFORE X' = untimed work before it)."""
    sums: dict = {}
    for m in TIMES_RE.finditer(text):
        key = ("" if m.group(1) == "AFTER" else "gap before ") + m.group(2).strip()
        sums[key] = sums.get(key, 0.0) + float(m.group(3))
    return sums


# ----------------------------------------------------------------------------------------------- runner

def cpu_limit() -> int:
    """CPUs this process may use: the cgroup quota, else the affinity mask, else os.cpu_count() (which in a
    container is the host's count)."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            return max(1, round(int(q) / int(p)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class Bench:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.mopac = Path(a.mopac).resolve()
        self.out = Path(a.out).resolve()
        self.runs = self.out / "runs"
        self.data = self.out / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        self.ncpu = a.cpus or cpu_limit()

    def env_for(self, mode: str, threads: int | None, force_gpu: bool, extra: dict | None) -> dict:
        env = os.environ.copy()
        for k in CPU_ENV_OFF:
            env.pop(k, None)
        n = str(threads or self.ncpu)
        env.update(OMP_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, MKL_NUM_THREADS=n)
        if mode != "gpu":
            env["MOPAC_NOGPU"] = "1"
        else:
            env["MOPAC_GPU_DEBUG"] = "1"
            if force_gpu:
                env["MOPAC_FORCEGPU"] = "1"
        env.update(extra or {})
        return env

    def run(self, exp: str, case: str, mode: str, deck: str, files=(), rep: int = 0, threads: int | None = None,
            force_gpu: bool = False, extra_env: dict | None = None, timeout: float | None = None,
            want_grad: bool = False, meta: dict | None = None, name: str = "job", binary: Path | None = None) -> dict:
        """Run one MOPAC deck (text) in its own directory; returns the parsed result (cached in result.json).
        binary: another MOPAC executable (default --mopac); every mode but gpu runs on the CPU."""
        d = self.runs / exp / case / (mode + (f"_r{rep}" if rep else ""))
        res_file = d / "result.json"
        mop = d / f"{name}.mop"
        # reuse a finished run only if it had exactly this input (a --quick run in the same directory
        # must not stand in for the full one)
        if res_file.exists() and mop.exists() and mop.read_text() == deck:
            try:
                return json.loads(res_file.read_text())
            except json.JSONDecodeError:
                pass
        if mode != "gpu" and self.a.no_cpu:
            return {}
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for f in files:
            shutil.copy2(f, d / Path(f).name)
        (d / f"{name}.mop").write_text(deck)
        env = self.env_for(mode, threads, force_gpu, extra_env)
        say(f"  [{exp}/{case}] {mode}{f' rep {rep}' if rep else ''} ...")
        mem = GpuMemory()
        rc = 0
        with mem, open(d / f"{name}.stdout", "w") as fh:
            t0 = time.perf_counter()
            try:
                rc = subprocess.run([str(binary or self.mopac), f"{name}.mop"], cwd=d, env=env, stdout=fh,
                                    stderr=subprocess.STDOUT,
                                    timeout=timeout or (self.a.gpu_timeout if mode == "gpu" else self.a.timeout)
                                    ).returncode
            except subprocess.TimeoutExpired:
                rc = 124
            wall = time.perf_counter() - t0
        out = d / f"{name}.out"
        text = out.read_text(errors="ignore") if out.exists() else ""
        stdout = (d / f"{name}.stdout").read_text(errors="ignore")
        statuses = STATUS_RE.findall(text + "\n" + stdout)
        r = {
            "exp": exp, "case": case, "mode": mode, "rep": rep, "rc": rc, "wall_s": round(wall, 3),
            "job_s": fnum(last(JOB_RE, text)), "heat": fnum(last(HEAT_RE, text)),
            "atoms": int(last(ATOMS_RE, text)) if last(ATOMS_RE, text) else None,
            "gnorm": fnum(last(GNORM_RE, text)),
            "scf_iterations": (sum(int(i) for _, i in statuses) if statuses else (len(ITER_RE.findall(text)) or None)),
            "gpu_scf_resident": sum(1 for s, _ in statuses if s == "success"),
            "gpu_scf_fallback": sum(1 for s, _ in statuses if s != "success"),
            "gpu_device_grad": text.count("[GPU gradient] pairs="),
            "gpu_device_fock": text.count("[GPU Fock] pairs="),
            "gpu_errors": (text + stdout).count("[GPU ERROR]"),
            "scf_failed": "SCF CALCULATION FAILED" in text or "NOT ENOUGH TIME" in text,
            "gpu_mem_mib": mem.peak, "threads": int(env["OMP_NUM_THREADS"]),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        r.update(meta or {})
        if want_grad:
            aux = d / f"{name}.aux"
            m = AUX_GRAD_RE.search(aux.read_text(errors="ignore")) if aux.exists() else None
            if m:
                (d / "grad.json").write_text(json.dumps([float(x) for x in m.group(1).split()]))
        # a failed run (crash, timeout) is kept as result_failed.json and tried again on the next call
        (res_file if rc == 0 else d / "result_failed.json").write_text(json.dumps(r, indent=1))
        say(f"      {wall:8.1f} s  heat {r['heat']}  atoms {r['atoms']}"
            f"{'  GPU mem %s MiB' % r['gpu_mem_mib'] if r['gpu_mem_mib'] else ''}"
            f"{'  FALLBACK' if r['gpu_scf_fallback'] else ''}{'  rc=%d' % rc if rc else ''}")
        return r

    def path(self, exp: str, case: str, mode: str, rep: int = 0) -> Path:
        return self.runs / exp / case / (mode + (f"_r{rep}" if rep else ""))

    def text(self, exp: str, case: str, mode: str, rep: int = 0, name: str = "job") -> str:
        f = self.path(exp, case, mode, rep) / f"{name}.out"
        return f.read_text(errors="ignore") if f.exists() else ""

    def grad(self, exp: str, case: str, mode: str) -> list | None:
        f = self.path(exp, case, mode) / "grad.json"
        return json.loads(f.read_text()) if f.exists() else None

    def write_csv(self, name: str, rows: list[dict]) -> None:
        rows = [r for r in rows if r]
        if not rows:
            return
        fields: list[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(self.data / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        say(f"  -> data/{name}.csv ({len(rows)} rows)")

    def timed(self, exp: str, case: str, deck: str, files=(), cpu: bool = True, gpu: bool = True,
              cpu_threads: int | None = None, force_gpu: bool = False, want_grad: bool = False,
              meta: dict | None = None, extra_env: dict | None = None) -> list[dict]:
        """CPU once, GPU --repeats times."""
        rows = []
        if cpu:
            rows.append(self.run(exp, case, "cpu", deck, files, threads=cpu_threads, want_grad=want_grad, meta=meta,
                                 extra_env=extra_env))
        if gpu:
            for k in range(1, max(1, self.a.repeats) + 1):
                rows.append(self.run(exp, case, "gpu", deck, files, rep=0 if k == 1 else k, force_gpu=force_gpu,
                                     want_grad=want_grad and k == 1, meta=meta, extra_env=extra_env))
                if rows[-1].get("rc") == 124:   # a run that hit the time limit is not repeated
                    break
        return rows


def deck_text(stem: str, keywords: str | None = None, title: str | None = None) -> tuple[str, list[Path]]:
    """The deck of benchmarks/publication_inputs/mop (first line optionally replaced) and its GEO_DAT file."""
    src = PUB / f"{stem}.mop"
    lines = src.read_text().splitlines()
    if keywords is not None:
        lines[0] = keywords
    if title is not None:
        lines[1] = title
    text = "\n".join(lines) + "\n"
    files = [PUB / Path(f).name for f in GEO_REF_RE.findall(lines[0]) if (PUB / Path(f).name).exists()]
    return text, files


def with_keywords(text: str, add: str = "", drop: tuple = ()) -> str:
    lines = text.splitlines()
    kw = lines[0]
    for d in drop:
        kw = re.sub(rf"(^|\s){re.escape(d)}(?=\s|$)", " ", kw)
    lines[0] = re.sub(r"\s+", " ", kw).strip() + (" " + add if add else "")
    return "\n".join(lines) + "\n"


def mat_deck(path: Path, keywords: str) -> str:
    lines = path.read_text().splitlines()
    lines[0] = keywords
    return "\n".join(lines) + "\n"


def silicon_supercell(n: int, a: float = 5.431, shake: float = 0.01) -> tuple[str, int]:
    """Diamond-structure Si, n x n x n conventional cells (8 n^3 atoms), atoms displaced by up to `shake` A
    (deterministic) so that no two images are exactly equidistant."""
    basis = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0), (.25, .25, .25), (.25, .75, .75), (.75, .25, .75),
             (.75, .75, .25)]
    lines, k = [], 0
    for i in range(n):
        for j in range(n):
            for l in range(n):
                for b in basis:
                    k += 1
                    x, y, z = ((i + b[0]) * a, (j + b[1]) * a, (l + b[2]) * a)
                    dx, dy, dz = (shake * ((k * 7919 + c * 104729) % 1000 / 500.0 - 1.0) for c in (1, 2, 3))
                    lines.append(f"Si {x + dx:12.6f} 1 {y + dy:12.6f} 1 {z + dz:12.6f} 1")
    L = n * a
    lines += [f"Tv {L:12.6f} 0 {0:12.6f} 0 {0:12.6f} 0", f"Tv {0:12.6f} 0 {L:12.6f} 0 {0:12.6f} 0",
              f"Tv {0:12.6f} 0 {0:12.6f} 0 {L:12.6f} 0"]
    return "PM7 1SCF GRADIENTS GEO-OK AUX TIMES\nsilicon %d atoms\n\n" % (8 * n ** 3) + "\n".join(lines) + "\n", 8 * n ** 3


# ----------------------------------------------------------------------------------------------- experiments

def exp_env(b: Bench) -> None:
    def sh(cmd: str) -> str:
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"n/a ({exc})"
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "commit": sh(f"git -C {REPO} log -1 --format='%h %ad %s' --date=short"),
        "tag": sh(f"git -C {REPO} describe --tags --always"),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,clocks.max.sm "
                  "--format=csv,noheader"),
        "cuda": sh("nvcc --version | tail -2 | head -1"),
        "cpu_model": (sh("lscpu | grep 'Model name' | sed 's/.*: *//'")
                      or sh("grep -m1 'model name' /proc/cpuinfo | sed 's/.*: *//'") or platform.processor()),
        "cpu_count": b.ncpu,
        "cpu_host_count": os.cpu_count(),
        "memory": sh("free -g | awk '/Mem:/ {print $2 \" GB\"}'"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "gfortran": sh("gfortran --version | head -1"),
        "mopac": sh(f"{b.mopac} --version"),
        "openmm": sh(f"{sys.executable} -c 'import openmm; print(openmm.__version__)'"),
    }
    (b.out / "env.json").write_text(json.dumps(info, indent=1))
    for k, v in info.items():
        say(f"  {k:10s} {v}")


def exp_mozyme_sp(b: Bench) -> None:
    exp, rows = "mozyme_sp", []
    cases = PROTEINS[:1] if b.a.quick else PROTEINS
    for case, stem, label, kind in cases:
        text, files = deck_text(stem)
        # atoms: the hydrogenated PDB
        atoms = sum(1 for ln in files[0].read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))) if files else 0
        rows += b.timed(exp, case, text, files, cpu=atoms <= b.a.cpu_max_atoms, cpu_threads=1,
                        meta={"label": label, "kind": kind})
    waters = (125,) if b.a.quick else WATERS
    for n in waters:
        wdir = b.runs / exp / "_decks"
        wdir.mkdir(parents=True, exist_ok=True)
        deck = wdir / f"water_{n}.mop"
        if not deck.exists():
            subprocess.run([sys.executable, str(REPO / "scripts" / "make_water_cluster.py"), str(n), "--out", str(deck)],
                           check=True, capture_output=True)
        rows += b.timed(exp, f"water_{n}", deck.read_text(), cpu=3 * n <= b.a.cpu_max_atoms, cpu_threads=1,
                        meta={"label": f"Water cluster, {n} H$_2$O", "kind": "water cluster"})
    b.write_csv(exp, rows)


def exp_mozyme_grad(b: Bench) -> None:
    exp, rows, comp = "mozyme_grad", [], []
    cases = PROTEINS[:1] if b.a.quick else [p for p in PROTEINS if p[0] in ("crambin", "dna", "ubiquitin", "barnase")]
    for case, stem, label, kind in cases:
        text, files = deck_text(stem)
        # tight SCF (as the stored references): the default criterion leaves ~0.1 kcal/(mol A) of SCF noise in
        # the gradient of either run, which would hide the GPU - CPU difference itself
        text = with_keywords(text, "GRADIENTS AUX SCFCRT=1.D-6 THRESH=1.D-15", drop=("NEWPDB", "PDB"))
        rows += b.timed(exp, case, text, files, cpu_threads=1, want_grad=True, meta={"label": label, "kind": kind})
        gc, gg = b.grad(exp, case, "cpu"), b.grad(exp, case, "gpu")
        if gc and gg and len(gc) == len(gg):
            d = [x - y for x, y in zip(gg, gc)]
            comp.append({"case": case, "label": label, "n": len(d), "max_abs": max(abs(x) for x in d),
                         "rms": (sum(x * x for x in d) / len(d)) ** 0.5,
                         "rms_grad": (sum(x * x for x in gc) / len(gc)) ** 0.5})
    b.write_csv(exp, rows)
    b.write_csv("mozyme_grad_compare", comp)


def exp_methods(b: Bench) -> None:
    exp, rows = "methods", []
    methods = ("AM1", "PM7") if b.a.quick else METHODS
    systems = [("crambin", "protein_crambin_1crn")] + ([] if b.a.quick else [("dna", "dna_dodecamer_1bna")])
    for case, stem in systems:
        text, files = deck_text(stem)
        for m in methods:
            t = text.split("\n", 1)
            t = m + " " + t[0].split(" ", 1)[1] + "\n" + t[1]
            for mode in ("cpu", "gpu"):
                rows.append(b.run(exp, f"{case}_{m}", mode, t, files, threads=1 if mode == "cpu" else None,
                                  meta={"system": case, "method": m}))
    b.write_csv(exp, rows)


def exp_cosmo(b: Bench) -> None:
    exp, rows = "cosmo", []
    cases = PROTEINS[:1] if b.a.quick else [p for p in PROTEINS if p[0] in ("crambin", "dna", "ubiquitin")]
    for case, stem, label, kind in cases:
        text, files = deck_text(stem)
        rows += b.timed(exp, case, with_keywords(text, "EPS=78.4", drop=("NEWPDB", "PDB")), files, cpu_threads=1,
                        meta={"label": label, "kind": kind})
    b.write_csv(exp, rows)


def restart_1scf(b: Bench, exp: str, case: str, from_mode: str, stem: str) -> list[dict]:
    """1SCF at the final geometry of an optimization (its .res), on the CPU and on the GPU."""
    res = b.path(exp, case, from_mode) / "job.res"
    if not res.exists():
        return []
    text, files = deck_text(stem)
    kw = re.sub(r"(?i)(^|\s)(CYCLES\s*=\s*\S+|GRADIENTS)(?=\s|$)", " ", text.splitlines()[0])
    digest = hashlib.sha1(res.read_bytes()).hexdigest()[:12]   # a new optimization invalidates the cached 1SCF
    deck = re.sub(r"\s+", " ", kw).strip() + \
        f" 1SCF RESTART\nfinal geometry of the {from_mode} optimization (restart file {digest})\n\n"
    out = []
    for mode in ("cpu", "gpu"):
        d = f"{case}_final_{from_mode}"
        out.append(b.run(exp, d, mode, deck, [*files, res], threads=1 if mode == "cpu" else None,
                         meta={"geometry_of": from_mode, "system": case}))
    return out


def exp_opt(b: Bench) -> None:
    exp, rows, cycles = "opt", [], []
    plan = [("crambin", "protein_crambin_1crn_opt" if b.a.quick else "protein_crambin_1crn_optfull", True)]
    if not b.a.quick:
        plan.append(("adk", "protein_adenylate_kinase_1ake_apo_optlong", b.a.cpu_long))
    for case, stem, do_cpu in plan:
        text, files = deck_text(stem)
        modes = (("cpu",) if do_cpu else ()) + ("gpu",)
        for mode in modes:
            r = b.run(exp, case, mode, text, files, threads=1 if mode == "cpu" else None, meta={"system": case})
            these = [{"system": case, "mode": mode, "cycle": int(m.group(1)), "cycle_s": float(m.group(2)),
                      "grad": fnum(m.group(3)), "heat": fnum(m.group(4))}
                     for m in CYCLE_RE.finditer(b.text(exp, case, mode))]
            cycles += these
            if r and these:
                # a run stopped by CYCLES prints no FINAL HEAT: the last cycle's heat and gradient
                r = dict(r, cycles=len(these), heat=r.get("heat") or these[-1]["heat"], gnorm_last=these[-1]["grad"])
            rows.append(r)
            rows += restart_1scf(b, exp, case, mode, stem)
    b.write_csv(exp, rows)
    b.write_csv("opt_cycles", cycles)


def exp_md(b: Bench) -> None:
    exp, rows, md, temps = "md", [], [], []
    geo = REF / "1CRN_opt.pdb"
    nrows = 4 if b.a.quick else b.a.md_rows
    plan = [("crambin_nve", f"PM7 {BASE_MOZ} DRC TEMPERATURE=300 SEED=11 T-PRIORITY=0.5 CYCLES={nrows}", geo, True)]
    if not b.a.quick:
        plan += [("crambin_nvt", f"PM7 {BASE_MOZ} DRC TEMPERATURE=300 BUSSI=50 SEED=11 T-PRIORITY=0.5 CYCLES=200",
                  geo, False)]
        adk, _ = deck_text("protein_adenylate_kinase_1ake_apo")
        plan += [("adk_nve", f"PM7 {BASE_MOZ} DRC TEMPERATURE=300 SEED=11 T-PRIORITY=0.5 CYCLES=20",
                  PUB / "protein_adenylate_kinase_1ake_apo_hydrogenated.pdb", False)]
    for case, kw, geom, do_cpu in plan:
        deck = f'{kw} GEO_DAT="{geom.name}"\n{case}\n\n'
        for mode in (("cpu",) if do_cpu else ()) + ("gpu",):
            r = b.run(exp, case, mode, deck, [geom], threads=1 if mode == "cpu" else None, meta={"system": case})
            t = b.text(exp, case, mode)
            pts = drc_rows(t)
            if pts:
                r = dict(r, steps=pts[-1][1], fs=pts[-1][0], s_per_step=r["wall_s"] / max(1, pts[-1][1]),
                         max_abs_error=max(abs(p[5]) for p in pts))
            rows.append(r)
            md += [{"system": case, "mode": mode, "fs": p[0], "point": p[1], "potential": p[2], "kinetic": p[3],
                    "total": p[4], "error": p[5]} for p in pts]
            temps += [{"system": case, "mode": mode, "step": int(m[0]), "fs": float(m[1]), "T": float(m[2]),
                       "T_mean": float(m[3]), "e_bath": float(m[4])} for m in TEMP_RE.findall(t)]
    b.write_csv(exp, rows)
    b.write_csv("md_rows", md)
    b.write_csv("md_temperature", temps)


def exp_thermo(b: Bench) -> None:
    """chignolin FORCE (parallel GPU processes) and crambin FORCETS against the stored CPU references."""
    exp = "thermo"
    work = b.runs / exp / "reference_check"
    summary = work / "summary.json"
    opts = f"workers={b.a.force_workers} quick={b.a.quick}"
    if summary.exists() and (work / "options.txt").exists() and (work / "options.txt").read_text() != opts:
        shutil.rmtree(work)
    if not summary.exists():
        work.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(REPO / "tests" / "check_mozyme_gpu_reference.py"), str(b.mopac), "--data", str(REF),
               "--work-dir", str(work), "--workers", str(b.a.force_workers), "--concurrent", "0"]
        if b.a.quick:
            cmd += ["--workers", "0", "--skip-forcets"]
        say("  " + " ".join(cmd))
        with open(b.runs / exp / "reference_check.log", "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
        (work / "options.txt").write_text(opts)
    ref = json.loads((REF / "reference.json").read_text())
    s = json.loads(summary.read_text()) if summary.exists() else {}
    rows, freqs = [], []
    for key, sysname, rkey, glob in (("chignolin_force %d processes" % b.a.force_workers, "chignolin FORCE",
                                      "chignolin_force", "chignolin_force_w*/**/*.out"),
                                     ("crambin_forcets", "crambin FORCETS", "crambin_forcets", "crambin_forcets/*.out")):
        if key not in s:
            continue
        c = ref[rkey]
        rows.append({"system": sysname, "gpu_s": s[key]["wall"], "cpu_ref_s": c["cpu_seconds"],
                     "cpu_ref_processes": c.get("cpu_processes", 1), "rms_cm1": s[key]["rms"],
                     "dzpe": s[key]["zpe"], "gpu_mem_mib": s[key].get("mem_peak_mib")})
        for f in sorted(work.glob(glob)):
            g = sorted(frequencies(f.read_text(errors="ignore")))
            if len(g) == len(c["freqs"]):
                freqs += [{"system": sysname, "mode": i + 1, "cpu": x, "gpu": y} for i, (x, y) in
                          enumerate(zip(c["freqs"], g))]
                break
    for key in ("chignolin_scf", "crambin_scf"):
        if key in s:
            rows.append({"system": key.replace("_scf", " 1SCF GRADIENTS"), "gpu_s": s[key]["wall"],
                         "cpu_ref_s": ref[key]["cpu_seconds"], "dheat": s[key]["heat_diff"],
                         "grad_norm_diff": s[key]["grad_diff"], "gpu_mem_mib": s[key].get("mem_peak_mib")})
    b.write_csv(exp, rows)
    b.write_csv("thermo_freqs", freqs)


SOLID_CELLS = [  # (case, deck, label, kind)
    ("diamond", "small/diamond.mop", "Diamond", "covalent"),
    ("graphene", "small/graphene.mop", "Graphene (2D)", "semimetal"),
    ("mgo_water", "small/mgo_water.mop", "MgO slab + water", "slab"),
    ("zno", "small/zno.mop", "ZnO", "ionic"),
    ("quartz162", "grad/quartz162.mop", "$\\alpha$-quartz", "covalent"),
    ("rutile162", "grad/rutile162.mop", "Rutile TiO$_2$", "ionic"),
    ("urea192", "small/urea.mop", "Urea crystal", "molecular"),
    ("quartz576", "medium/quartz.mop", "$\\alpha$-quartz", "covalent"),
    ("rutile576", "medium/rutile.mop", "Rutile TiO$_2$", "ionic"),
    ("urea768", "medium/urea768.mop", "Urea crystal", "molecular"),
]


def exp_solids(b: Bench) -> None:
    exp, rows, comp = "solids", [], []
    kw = "PM7 1SCF GRADIENTS GEO-OK AUX TIMES"
    series = (2,) if b.a.quick else b.a.si_sizes
    plan = []
    for n in series:
        text, atoms = silicon_supercell(n)
        plan.append((f"si{atoms}", text, "Silicon", "covalent", atoms))
    cells = SOLID_CELLS[:1] if b.a.quick else SOLID_CELLS
    for case, rel, label, kind in cells:
        plan.append((case, mat_deck(MAT / rel, kw), label, kind, None))
    for case, text, label, kind, atoms in plan:
        n = atoms or sum(1 for ln in text.splitlines()[3:] if ln.strip() and not ln.startswith("Tv"))
        rows += b.timed(exp, case, text, cpu=n <= b.a.solids_cpu_max_atoms, force_gpu=True, want_grad=True,
                        meta={"label": label, "kind": kind, "series": "Si" if case.startswith("si") else "cells"})
        gc, gg = b.grad(exp, case, "cpu"), b.grad(exp, case, "gpu")
        if gc and gg and len(gc) == len(gg):
            d = [x - y for x, y in zip(gg, gc)]
            comp.append({"case": case, "label": label, "n": len(d), "max_abs": max(abs(x) for x in d),
                         "rms": (sum(x * x for x in d) / len(d)) ** 0.5})
    b.write_csv(exp, rows)
    b.write_csv("solids_grad_compare", comp)


def exp_solids_profile(b: Bench) -> None:
    exp, rows, sections = "solids_profile", [], []
    kw = "PM7 1SCF GRADIENTS GEO-OK TIMES"
    plan = [("si512", silicon_supercell(4)[0].replace("AUX TIMES", "TIMES"))] if not b.a.quick else \
        [("si64", silicon_supercell(2)[0].replace("AUX TIMES", "TIMES"))]
    if not b.a.quick:
        plan += [(c, mat_deck(MAT / r, kw)) for c, r, *_ in SOLID_CELLS if c in ("quartz576", "rutile576", "urea768")]
    for case, text in plan:
        r = b.run(exp, case, "gpu", text, force_gpu=True, extra_env={"MOPAC_GPU_PROFILE": "1"})
        rows.append(r)
        t = b.text(exp, case, "gpu")
        for k, v in stage_times(t).items():
            sections.append({"case": case, "source": "TIMES", "name": k, "calls": None, "s": v})
        for name, calls, ms in SECTION_RE.findall(t):
            sections.append({"case": case, "source": "section", "name": name, "calls": int(calls),
                             "s": float(ms) / 1000.0})
    b.write_csv(exp, rows)
    b.write_csv("solids_profile_sections", sections)


def exp_solids_force(b: Bench) -> None:
    exp, rows, freqs = "solids_force", [], []
    text = mat_deck(MAT / "small" / "diamond.mop", "PM7 FORCE GEO-OK SCFCRT=1.D-10")
    res = {}
    for mode in ("cpu", "gpu"):
        rows.append(b.run(exp, "diamond", mode, text, force_gpu=True, meta={"system": "diamond"}))
        res[mode] = sorted(frequencies(b.text(exp, "diamond", mode)))
    if res["cpu"] and len(res["cpu"]) == len(res["gpu"]):
        freqs = [{"system": "diamond FORCE", "mode": i + 1, "cpu": x, "gpu": y}
                 for i, (x, y) in enumerate(zip(res["cpu"], res["gpu"]))]
    b.write_csv(exp, rows)
    b.write_csv("solids_force_freqs", freqs)


def exp_solids_drc(b: Bench) -> None:
    exp, rows, md = "solids_drc", [], []
    text = mat_deck(MAT / "small" / "diamond.mop", f"PM7 DRC TEMPERATURE=300 SEED=7 CYCLES={4 if b.a.quick else 40} GEO-OK")
    for mode in ("cpu", "gpu"):
        r = b.run(exp, "diamond", mode, text, force_gpu=True, meta={"system": "diamond"})
        pts = drc_rows(b.text(exp, "diamond", mode))
        if pts:
            r = dict(r, steps=pts[-1][1], s_per_step=r["wall_s"] / max(1, pts[-1][1]),
                     max_abs_error=max(abs(p[5]) for p in pts))
        rows.append(r)
        md += [{"system": "diamond", "mode": mode, "fs": p[0], "point": p[1], "potential": p[2], "kinetic": p[3],
                "total": p[4], "error": p[5]} for p in pts]
    b.write_csv(exp, rows)
    b.write_csv("solids_drc_rows", md)


def exp_conv_mol(b: Bench) -> None:
    # SCFCRT=1.D-6: with the default criterion the SCF of the graphene flake stops early enough for its
    # gradient to depend on the path (max. 1.3 kcal/(mol A) between full diagonalizations and the GPU's
    # pseudo-diagonalization, also on the CPU; 0.06 with SCFCRT=1.D-6), which would hide the GPU - CPU
    # difference this compares
    exp, rows, comp = "conv_mol", [], []
    crambin = MAT / "molecules" / "crambin.mop"
    tight = "SCFCRT=1.D-6 AUX TIMES"
    plan = [("crambin_rhf", mat_deck(crambin, f"PM7 1SCF GRADIENTS GEO-OK CHARGE=-1 {tight}"), "Crambin, RHF"),
            ("crambin_uhf", mat_deck(crambin, f"PM7 UHF 1SCF GRADIENTS GEO-OK CHARGE=0 {tight}"), "Crambin, UHF doublet"),
            ("crambin_cosmo", mat_deck(crambin, f"PM7 1SCF GRADIENTS GEO-OK CHARGE=-1 EPS=78.4 {tight}"),
             "Crambin, COSMO"),
            # 1.D-8: at 1.D-6 the gradients of repeated GPU runs of this SCF still differ by ~0.01 and GPU vs CPU by
            # 0.2 kcal/(mol A) (Modal A100), SCF noise of a difficult SCF rather than the device code
            ("graphene_flake", with_keywords(deck_text("material_graphene_nanoflake_c192h38")[0],
                                             "GRADIENTS SCFCRT=1.D-8 AUX TIMES"),
             "Graphene flake C$_{192}$H$_{38}$"),
            ("chignolin_ci", mat_deck(MAT / "molecules" / "chignolin.mop",
                                      f"PM7 1SCF GRADIENTS CHARGE=3 DOUBLET RHF GEO-OK {tight}"),
             "Chignolin, half-electron C.I. gradient")]
    if b.a.quick:
        plan = plan[3:4]
    for case, text, label in plan:
        rows += b.timed(exp, case, text, force_gpu=True, want_grad=True, meta={"label": label})
        gc, gg = b.grad(exp, case, "cpu"), b.grad(exp, case, "gpu")
        if gc and gg and len(gc) == len(gg):
            d = [x - y for x, y in zip(gg, gc)]
            comp.append({"case": case, "label": label, "n": len(d), "max_abs": max(abs(x) for x in d),
                         "rms": (sum(x * x for x in d) / len(d)) ** 0.5})
    b.write_csv(exp, rows)
    b.write_csv("conv_mol_grad_compare", comp)


QMMM_LINE = {
    "system": re.compile(r"system: (\d+) atoms, (\d+) residues, box ([0-9.]+) A"),
    "qm": re.compile(r"QM region: (\d+) atoms \+ (\d+) link atoms, charge (-?\d+); long-range QM-MM electrostatics: (.+)"),
    "nvt": re.compile(r"QM/MM NVT: (\d+) steps x ([0-9.]+) fs in ([0-9.]+) s = ([0-9.]+) s/step, ([0-9.]+) ps/day"),
    "nve": re.compile(r"NVE check (\d+) steps: drift (-?[0-9.]+) kJ/mol, slope of a linear fit (-?[0-9.]+) kJ/mol/ps, "
                      r"largest step ([0-9.]+), rms step ([0-9.]+), rms deviation from the fit ([0-9.]+)"),
    "mm": re.compile(r"MM equilibration ([0-9.]+) ps: ([0-9.]+) s"),
    "cut": re.compile(r"(PME|cut-off) roff\s+(\d+) A \((\d+) MM charges\): rms force error ([0-9.]+) \(rms force "
                      r"(\d+)\), largest ([0-9.]+); energy change between snapshots: spread ([0-9.]+) kJ/mol"),
    "dist": re.compile(r"distance (\S+): mean ([0-9.]+) A, min ([0-9.]+), max ([0-9.]+)"),
}


def qmmm_run(b: Bench, exp: str, case: str, mode: str, pdb: Path, opts: str, steps: int, nve: int,
             mm_ps: float = 10.0) -> dict:
    d = b.path(exp, case, mode)
    res = d / "result.json"
    cmd = (f"{sys.executable} -u {REPO / 'scripts' / 'qmmm' / 'run_qmmm_md.py'} {pdb} --libmopac {b.a.libmopac} "
           f"{opts} --platform {b.a.openmm_platform} --steps {steps} --nve {nve} --mm-equil-ps {mm_ps}")
    if res.exists() and (d / "cmd.txt").exists() and (d / "cmd.txt").read_text() == cmd:
        return json.loads(res.read_text())
    if mode == "cpu" and b.a.no_cpu:
        return {}
    if not b.a.libmopac:
        say("  qmmm: --libmopac not given, skipped")
        return {}
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "cmd.txt").write_text(cmd)
    env = b.env_for(mode, 1 if mode == "cpu" else None, False, None)
    say(f"  [{exp}/{case}] {mode}: {cmd}")
    t0 = time.perf_counter()
    with open(d / "log.txt", "w") as fh:
        rc = subprocess.run(cmd, shell=True, cwd=d, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    text = (d / "log.txt").read_text(errors="ignore")
    r = {"exp": exp, "case": case, "mode": mode, "rc": rc, "total_s": round(time.perf_counter() - t0, 1)}
    if (m := QMMM_LINE["system"].search(text)):
        r.update(total_atoms=int(m.group(1)), box_A=float(m.group(3)))
    if (m := QMMM_LINE["qm"].search(text)):
        r.update(qm_atoms=int(m.group(1)), link_atoms=int(m.group(2)), qm_charge=int(m.group(3)),
                 long_range=m.group(4).strip())
    if (m := QMMM_LINE["nvt"].search(text)):
        r.update(steps=int(m.group(1)), dt_fs=float(m.group(2)), md_s=float(m.group(3)),
                 s_per_step=float(m.group(4)), ps_per_day=float(m.group(5)))
    if (m := QMMM_LINE["nve"].search(text)):
        r.update(nve_steps=int(m.group(1)), nve_drift_kj=float(m.group(2)), nve_slope_kj_ps=float(m.group(3)),
                 nve_rms_dev_kj=float(m.group(6)))
    if (m := QMMM_LINE["mm"].search(text)):
        r.update(mm_equil_s=float(m.group(2)))
    r["cutoff_test"] = [dict(zip(("method", "roff", "n_mm", "rms_err", "rms_force", "max_err", "spread"),
                                 (g[0], float(g[1]), int(g[2]), float(g[3]), float(g[4]), float(g[5]), float(g[6]))))
                        for g in QMMM_LINE["cut"].findall(text)]
    r["distances"] = [dict(zip(("pair", "mean", "min", "max"), (g[0], *map(float, g[1:])))) for g in
                      QMMM_LINE["dist"].findall(text)]
    (res if rc == 0 and "s_per_step" in r else d / "result_failed.json").write_text(json.dumps(r, indent=1))
    if rc != 0:
        say("      failed; end of the log:\n" + "\n".join(text.splitlines()[-8:]))
    say(f"      {r.get('s_per_step')} s/step, {r.get('ps_per_day')} ps/day, QM {r.get('qm_atoms')} atoms, rc {rc}")
    return r


def get_pdb(code: str, where: Path) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    f = where / f"{code}.pdb"
    if not f.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{code}.pdb", f)
    return f


def exp_qmmm(b: Bench) -> None:
    exp, rows, nve = "qmmm", [], []
    q = b.a.quick
    chig, cram = REF / "1UAO_opt.pdb", REF / "1CRN_opt.pdb"
    # QM-PME needs a box of at least 2 (QM radius + 13 A): chignolin R 10.6 A -> 47.2 A (padding 2.2: 44.0),
    # crambin R 16.5-16.9 A -> 59.8 A (padding 1.8: 51.6); 0.1 nm of padding adds 2 A
    plan = [  # (case, label, pdb, options, GPU steps, GPU NVE steps, CPU steps)
        ("tyr_sc", "Tyr side chain (chignolin)", chig, "--qm-residues 2 --qm-sidechain --padding 1.8",
         20 if q else 1000, 0 if q else 400, 0),
        ("chignolin", "Chignolin", chig, "--qm-residues 1,2,3,4,5,6,7,8,9,10 --padding 2.5 --ron 6 --roff 8",
         400, 400, 40),
        ("crambin", "Crambin", cram, "--qm-residues " + ",".join(map(str, range(1, 47))) +
         " --padding 2.3 --ron 6 --roff 8", 100, 200, 10),
    ]
    if b.a.qmmm_large:
        plan.append(("ubiquitin", "Ubiquitin", get_pdb("1UBQ", b.runs / exp / "_pdb"),
                     "--qm-residues " + ",".join(map(str, range(1, 77))) + " --padding 1.8 --ron 6 --roff 8",
                     40, 0, 0))
    if q:
        plan = plan[:1]
    for case, label, pdb, opts, steps, nsteps, csteps in plan:
        modes = [("gpu", steps, nsteps)] + ([("cpu", csteps, 0)] if csteps else [])
        for mode, s, n in modes:
            r = qmmm_run(b, exp, case, mode, pdb, opts, s, n)
            if r:
                rows.append(dict({k: v for k, v in r.items() if k not in ("cutoff_test", "distances")}, label=label))
            e = b.path(exp, case, mode) / "nve_energy.txt"
            if e.exists():
                vals = [float(x) for x in e.read_text().split()]
                dt = r.get("dt_fs", 0.5)
                nve += [{"case": case, "mode": mode, "step": i, "fs": i * dt, "E_kj": v} for i, v in enumerate(vals)]
    b.write_csv(exp, rows)
    b.write_csv("qmmm_nve", nve)


def exp_qmmm_enzyme(b: Bench) -> None:
    exp, rows, cut, dist = "qmmm_enzyme", [], [], []
    pdb = get_pdb("2PTC", b.runs / exp / "_pdb")
    base = ("--qm-residues E:57:sc,E:102:sc,E:195:sc,I:15,I:16 --variants E:57=HID --padding 1.0 --ron 10 --roff 12 "
            "--distances E:195:OG-I:15:C,E:57:NE2-E:195:OG,E:57:HD1-E:102:OD2")
    steps = 20 if b.a.quick else b.a.enzyme_steps
    for case, extra in (("cutoff", "--no-pme"), ("pme", "--cutoff-test 8,12,16,20,24 --snapshots 8")):
        r = qmmm_run(b, exp, case, "gpu", pdb, f"{base} {extra}", steps, 0 if b.a.quick else 400,
                     mm_ps=10.0 if b.a.quick else 100.0)
        if not r:
            continue
        rows.append({k: v for k, v in r.items() if k not in ("cutoff_test", "distances")})
        cut += [dict(c, run=case) for c in r.get("cutoff_test", [])]
        dist += [dict(x, run=case) for x in r.get("distances", [])]
    b.write_csv(exp, rows)
    b.write_csv("qmmm_enzyme_cutoff", cut)
    b.write_csv("qmmm_enzyme_distances", dist)


def exp_mozyme_threads(b: Bench) -> None:
    """MOZYME 1SCF on the CPU with 1, 4 and all threads: is the one-thread CPU reference of mozyme_sp fair?
    (The MOZYME code has no OpenMP; BLAS calls could still use threads.)"""
    exp, rows = "mozyme_threads", []
    cases = PROTEINS[:1] if b.a.quick else [p for p in PROTEINS if p[0] in ("crambin", "ubiquitin", "adk")]
    for case, stem, label, kind in cases:
        text, files = deck_text(stem)
        for n in sorted({1, 4, b.ncpu}):
            rows.append(b.run(exp, f"{case}_t{n}", "cpu", text, files, threads=n,
                              meta={"system": case, "label": label, "threads_requested": n}))
    b.write_csv(exp, rows)


def upstream_deck(text: str) -> str:
    """The deck without the keywords that only this fork knows (they act on the GPU path only)."""
    lines = text.split("\n", 1)
    kw = re.sub(r"(?i)(^|\s)(MOZYME_GPU|MOZYME_MINBLK=\S+)(?=\s|$)", " ", lines[0])
    return re.sub(r"\s+", " ", kw).strip() + ("\n" + lines[1] if len(lines) > 1 else "\n")


def upstream_rows(b: Bench, exp: str, case: str, kind: str, label: str, res: dict) -> list[dict]:
    """One row per fork mode: heat, gradient, iterations and time against the upstream run of the same deck."""
    out, ru = [], res.get("upstream") or {}
    gu = b.grad(exp, case, "upstream")
    for m, r in res.items():
        if m == "upstream" or not r:
            continue
        g = b.grad(exp, case, m)
        hu, hm = ru.get("heat"), r.get("heat")
        row = {"case": case, "kind": kind, "label": label, "mode": m, "heat_upstream": hu, "heat": hm,
               "dheat": (hm - hu) if hu is not None and hm is not None else None,
               "upstream_failed": bool(ru.get("scf_failed")) or hu is None, "failed": bool(r.get("scf_failed")) or hm is None,
               "grad_max_abs": None, "grad_rms": None,
               "iter_upstream": ru.get("scf_iterations"), "iter": r.get("scf_iterations"),
               "wall_upstream": ru.get("wall_s"), "wall": r.get("wall_s")}
        if g and gu and len(g) == len(gu):
            dg = [x - y for x, y in zip(g, gu)]
            row["grad_max_abs"] = max(abs(x) for x in dg)
            row["grad_rms"] = (sum(x * x for x in dg) / len(dg)) ** 0.5
        out.append(row)
    return out


def exp_upstream(b: Bench) -> None:
    """MOPAC 23.1.2 (commit 78dc4572, the base of the fork; CPU build, --upstream-mopac) against the CPU path of
    this fork, same decks and threads: heats, gradients, SCF iterations and times.  Solids also with SHIFT=80 in
    the fork (the original level shift, without the damped start), which isolates the new solid-state SCF."""
    exp, rows, comp = "upstream", [], []
    if not b.a.upstream_mopac:
        say("  upstream: no --upstream-mopac given, skipped")
        return
    up = Path(b.a.upstream_mopac).resolve()
    tight = "SCFCRT=1.D-6 THRESH=1.D-15"
    plan = []   # (case, kind, deck, files, threads, label, solid)
    mozyme = PROTEINS[:1] if b.a.quick else [p for p in PROTEINS if p[0] in ("crambin", "dna", "ubiquitin", "adk")]
    for case, stem, label, _ in mozyme:
        text, files = deck_text(stem)
        plan.append((case, "MOZYME 1SCF", upstream_deck(text), files, 1, label, False))
    if not b.a.quick:
        text, files = deck_text("protein_crambin_1crn")
        plan.append(("crambin_grad", "MOZYME 1SCF GRADIENTS",
                     upstream_deck(with_keywords(text, f"GRADIENTS AUX {tight}", drop=("NEWPDB", "PDB"))), files, 1,
                     "Crambin (1CRN)", False))
        plan.append(("graphene_flake", "conventional 1SCF GRADIENTS",
                     with_keywords(deck_text("material_graphene_nanoflake_c192h38")[0], "GRADIENTS SCFCRT=1.D-6 AUX TIMES"),
                     [], None, "Graphene flake C$_{192}$H$_{38}$", False))
    kw = "PM7 1SCF GRADIENTS GEO-OK AUX TIMES"
    solids = [(f"si{8 * n ** 3}", silicon_supercell(n)[0], "Silicon") for n in ((2,) if b.a.quick else (2, 3))]
    if not b.a.quick:
        solids += [(c, mat_deck(MAT / r, kw), lab) for c, r, lab, _ in SOLID_CELLS
                   if c in ("diamond", "graphene", "mgo_water", "zno", "quartz162", "rutile162", "urea192")]
    for case, text, label in solids:
        plan.append((case, "solid 1SCF GRADIENTS", text, [], None, label, True))
    for case, kind, deck, files, threads, label, solid in plan:
        modes = [("upstream", deck, up), ("fork", deck, None)]
        if solid:
            modes.append(("fork_shift80", with_keywords(deck, "SHIFT=80"), None))
        res = {}
        for mode, d, binary in modes:
            r = b.run(exp, case, mode, d, files, threads=threads, binary=binary, want_grad=True,
                      meta={"system": case, "kind": kind, "label": label})
            res[mode] = r
            rows.append(r)
        comp += upstream_rows(b, exp, case, kind, label, res)
    # same heat, different gradient (the PM7 H-bond gradient was fixed in the fork): central differences of the
    # heat decide which gradient is right (chignolin, conventional SCF converged to 1e-8)
    case, pdb = "chignolin_fd", REF / "1UAO_opt.pdb"
    kw = "PM7 1SCF PULAY GEO-OK SCFCRT=1.D-8"
    deck = f'{kw} GRADIENTS AUX GEO_DAT="{pdb.name}"\nchignolin (1UAO), conventional SCF\n\n'
    res = {}
    for mode, binary in (("upstream", up), ("fork", None)):
        res[mode] = b.run(exp, case, mode, deck, [pdb], binary=binary, want_grad=True,
                          meta={"system": case, "kind": "conventional 1SCF GRADIENTS", "label": "Chignolin (1UAO)"})
        rows.append(res[mode])
    comp += upstream_rows(b, exp, case, "conventional 1SCF GRADIENTS", "Chignolin (1UAO)", res)
    fd_csv = b.data / "upstream_fd.csv"
    fa, fb = b.path(exp, case, "fork") / "job.aux", b.path(exp, case, "upstream") / "job.aux"
    if fa.exists() and fb.exists() and not b.a.quick:
        cmd = [sys.executable, str(REPO / "scripts" / "paper" / "fd_gradient.py"), "--mopac", str(b.mopac),
               "--pdb", str(pdb), "--keywords", kw, "--aux", str(fa), "--compare-aux", str(fb), "--label-a", "fork",
               "--label-b", "upstream", "--n", "8", "--work", str(b.runs / exp / case / "fd"), "--csv", str(fd_csv)]
        say("  " + " ".join(cmd))
        subprocess.run(cmd, check=False)
    b.write_csv(exp, rows)
    b.write_csv("upstream_compare", comp)


def exp_qmmm_pme_check(b: Bench) -> None:
    """tests/check_qmmm_pme_ewald.py: the QM-MM PME against a direct Ewald sum, rock salt and finite differences."""
    exp = "qmmm_pme_check"
    d = b.runs / exp
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPO / "tests" / "check_qmmm_pme_ewald.py"), "--csv", str(b.data / "qmmm_pme_check.csv"),
           "--gpu"]
    say("  " + " ".join(cmd))
    with open(d / "log.txt", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
    say(f"  -> data/qmmm_pme_check.csv (rc {rc})")


EXPERIMENTS = {
    "env": exp_env, "mozyme_sp": exp_mozyme_sp, "mozyme_grad": exp_mozyme_grad, "methods": exp_methods,
    "cosmo": exp_cosmo, "opt": exp_opt, "md": exp_md, "thermo": exp_thermo, "solids": exp_solids,
    "solids_profile": exp_solids_profile, "solids_force": exp_solids_force, "solids_drc": exp_solids_drc,
    "conv_mol": exp_conv_mol, "qmmm": exp_qmmm, "qmmm_enzyme": exp_qmmm_enzyme,
    "mozyme_threads": exp_mozyme_threads, "upstream": exp_upstream, "qmmm_pme_check": exp_qmmm_pme_check,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mopac", required=True)
    ap.add_argument("--libmopac", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", default="all", help="comma list of " + ", ".join(EXPERIMENTS) + " (default all)")
    ap.add_argument("--repeats", type=int, default=3, help="GPU repetitions of every timed run")
    ap.add_argument("--quick", action="store_true", help="one small case per experiment (test of the pipeline)")
    ap.add_argument("--no-cpu", action="store_true", help="skip every CPU run (GPU only)")
    ap.add_argument("--cpu-max-atoms", type=int, default=7000, help="MOZYME CPU runs up to this size")
    ap.add_argument("--solids-cpu-max-atoms", type=int, default=800, help="conventional-SCF CPU runs up to this size")
    ap.add_argument("--si-sizes", default="2,3,4,5,6", help="Si supercells n x n x n conventional cells (8 n^3 atoms)")
    ap.add_argument("--cpu-long", action="store_true", help="also the 1AKE optimization on the CPU (hours)")
    ap.add_argument("--cpus", type=int, default=0, help="threads of the all-core CPU runs (default: the CPU quota)")
    ap.add_argument("--upstream-mopac", default="", help="MOPAC 23.1.2 executable (CPU build) for the upstream experiment")
    ap.add_argument("--md-rows", type=int, default=40, help="DRC rows of the crambin NVE run (0.5 fs each)")
    ap.add_argument("--force-workers", type=int, default=4, help="GPU processes of the chignolin FORCE")
    ap.add_argument("--qmmm-large", action="store_true", help="also the whole ubiquitin as QM region")
    ap.add_argument("--enzyme-steps", type=int, default=2000)
    ap.add_argument("--openmm-platform", default="CUDA")
    ap.add_argument("--timeout", type=float, default=6 * 3600)
    # the longest GPU run of the A100 benchmarks takes ~7 min; a GPU run that lasts an hour is stuck
    # (PM7 Si512 with displaced atoms ran 6 h without SCF convergence before the PDIAG safeguard)
    ap.add_argument("--gpu-timeout", type=float, default=3600)
    a = ap.parse_args()
    a.si_sizes = tuple(int(x) for x in a.si_sizes.split(",") if x.strip())
    if a.libmopac:
        a.libmopac = str(Path(a.libmopac).resolve())
    names = list(EXPERIMENTS) if a.exp == "all" else [x.strip() for x in a.exp.split(",") if x.strip()]
    if a.exp == "all" and not a.libmopac:
        names = [n for n in names if not n.startswith("qmmm")]
    bad = [n for n in names if n not in EXPERIMENTS]
    if bad:
        ap.error(f"unknown experiment(s): {bad}")
    b = Bench(a)
    for n in names:
        t0 = time.time()
        say(f"== {n}")
        EXPERIMENTS[n](b)
        say(f"== {n} done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

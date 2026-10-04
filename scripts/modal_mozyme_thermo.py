#!/usr/bin/env python3
"""MOZYME GPU thermochemistry check on Modal (the Colab notebook colab/mozyme_thermo_colab.ipynb, on a Modal GPU).

The image is built from this working tree (not from GitHub), so the run uses exactly the local commit.
Results go to the Modal volume "mopac-thermo-results" (one directory per run label).

  modal run scripts/modal_mozyme_thermo.py::smoke                 # build + GPU 1SCF chignolin, CPU vs GPU
  modal run --detach scripts/modal_mozyme_thermo.py::thermo       # tests/check_mozyme_gpu_thermo.py
  modal run --detach scripts/modal_mozyme_thermo.py::thermo --large --label crambin   # + crambin timing
  modal run scripts/modal_mozyme_thermo.py::scfprobe --no-cpu --envs MOPAC_MOZYME_SCF_TRACE=1   # SCF per iteration
  modal run --detach scripts/modal_mozyme_thermo.py::force_gpu --workers 4          # GPU-only FORCE THERMO
  modal run scripts/modal_mozyme_thermo.py::shell --cmd "nvidia-smi"                # diagnostics
  modal volume get mopac-thermo-results <label> ./modal_results

Only GPU work belongs here: CPU references are cheaper locally.

GPU type: MOPAC_MODAL_GPU (A100 by default; H100, L40S); the CUDA architecture follows it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

GPU = os.environ.get("MOPAC_MODAL_GPU", "A100")
SM = {"A100": "80", "A100-80GB": "80", "H100": "90", "H200": "90", "L40S": "89", "L4": "89", "A10G": "86"}[GPU]
REPO = Path(__file__).resolve().parents[1]
SRC = "/opt/mopac_src"
BUILD = "/opt/mopac_build"
MOPAC = f"{BUILD}/mopac"
RESULTS = "/results"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("gfortran", "cmake", "ninja-build", "libblas-dev", "liblapack-dev")
    .add_local_file(REPO / "CMakeLists.txt", f"{SRC}/CMakeLists.txt", copy=True)
    .add_local_dir(REPO / "cmake", f"{SRC}/cmake", copy=True)
    .add_local_dir(REPO / "include", f"{SRC}/include", copy=True)
    .add_local_dir(REPO / "src", f"{SRC}/src", copy=True)
    # files the CMake install/CPack section references
    .add_local_dir(REPO / ".github", f"{SRC}/.github", copy=True)
    .add_local_file(REPO / "LICENSE", f"{SRC}/LICENSE", copy=True)
    .add_local_file(REPO / "CITATION.cff", f"{SRC}/CITATION.cff", copy=True)
    # GPU benchmark drivers built with GPU=ON even when TESTS=OFF
    .add_local_file(REPO / "tests/gpu_bench.F90", f"{SRC}/tests/gpu_bench.F90", copy=True)
    .add_local_file(REPO / "tests/gpu_resident_fock_pair_compare.F90",
                    f"{SRC}/tests/gpu_resident_fock_pair_compare.F90", copy=True)
    .run_commands(
        f"cmake -S {SRC} -B {BUILD} -GNinja -DGPU=ON -DTESTS=OFF -DGIT_HASH=OFF "
        f"-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCUDA_ARCHS={SM} > /tmp/cmake.log 2>&1 "
        "|| (grep -A12 'CMake Error' /tmp/cmake.log; exit 1)",
        "grep -i -E 'cuda|gpu' /tmp/cmake.log | head -20",
        # print the compiler errors when the build fails
        f"cmake --build {BUILD} --target mopac --parallel 8 > /tmp/build.log 2>&1 "
        f"|| (grep -i -B2 -A6 'error' /tmp/build.log | head -150; tail -40 /tmp/build.log; exit 1)",
        f"test -x {MOPAC}",
    )
    # scripts last: editing them does not rebuild MOPAC
    .add_local_dir(REPO / "scripts", f"{SRC}/scripts", ignore=["__pycache__"])
    .add_local_file(REPO / "tests/check_mozyme_gpu_thermo.py", f"{SRC}/tests/check_mozyme_gpu_thermo.py")
)

app = modal.App("mopac-mozyme-thermo", image=image)
volume = modal.Volume.from_name("mopac-thermo-results", create_if_missing=True)


def _stream(cmd: list[str], cwd: str, log: Path) -> int:
    """Run cmd, echo every line and keep a copy in log; commit the volume every 2 minutes."""
    print("$", " ".join(cmd), flush=True)
    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(120):
            try:
                volume.commit()
            except Exception as exc:  # a failed commit must not stop the run
                print("volume commit failed:", exc, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    with log.open("a") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
            fh.flush()
        rc = proc.wait()
    stop.set()
    volume.commit()
    print("exit status", rc, flush=True)
    return rc


def _gpu_info() -> None:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
    r = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True, text=True)
    print("MPS:", "active" if r.returncode == 0 else f"not available ({r.stderr.strip() or r.returncode})", flush=True)


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=3600, volumes={RESULTS: volume})
def smoke() -> None:
    """Chignolin (1UAO, 140 atoms): prepare, then 1SCF on the GPU and on the CPU; the GPU SCF must be resident."""
    _gpu_info()
    sys.path.insert(0, f"{SRC}/scripts")
    sys.path.insert(0, f"{SRC}/tests")
    import check_mozyme_gpu_thermo as th
    import mozyme_md_workflow as md

    work = Path(RESULTS) / "smoke"
    raw = th.get_pdb("1UAO", work)
    pdb = md.prepare_pdb(Path(MOPAC), raw, work / "prepare")
    keys = f'{md.BASE_KEYS} 1SCF GEO_DAT="{pdb.name}"'
    g = md.run_mopac(Path(MOPAC), work / "gpu", "scf", keys, "scf", [pdb], None)
    c = md.run_mopac(Path(MOPAC), work / "cpu", "scf", keys, "scf", [pdb], {"MOPAC_NOGPU": "1"})
    print(f"GPU: heat {g.heat} kcal/mol, {g.wall:.1f} s, GPU SCF statuses {g.statuses}")
    print(f"CPU: heat {c.heat} kcal/mol, {c.wall:.1f} s")
    volume.commit()
    if not g.statuses or any(s != "success" for s in g.statuses):
        raise SystemExit("GPU SCF not resident")
    if g.heat is None or c.heat is None or abs(g.heat - c.heat) > 1e-3:
        raise SystemExit("CPU and GPU heats differ")
    print("smoke OK")


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=24 * 3600, volumes={RESULTS: volume})
def thermo(label: str = "chignolin", large: bool = False, extra: str = "", seed_opt: str = "") -> int:
    """tests/check_mozyme_gpu_thermo.py with the notebook settings (SCFCRT=1e-6 THRESH=1e-15, 4 parallel workers).
    seed_opt: an optimized PDB in the volume (e.g. optcheck/gpu/opt.pdb) used as the small-system geometry
    instead of a new optimization (--reuse then only skips that step)."""
    import shutil

    _gpu_info()
    work = Path(RESULTS) / label
    work.mkdir(parents=True, exist_ok=True)
    if seed_opt:
        (work / "1UAO" / "opt").mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(RESULTS) / seed_opt, work / "1UAO" / "opt" / "opt.pdb")
        extra = (extra + " --reuse").strip()
    cmd = [sys.executable, f"{SRC}/tests/check_mozyme_gpu_thermo.py", MOPAC,
           "--pdb-id", "1UAO", "--partial", "A9=4", "--large-pdb-id", "1CRN", "--large-partial", "A25=5",
           "--temperature", "298", "--gnorm", "1", "--work-dir", str(work),
           "--scfcrt", "0.000001", "--thresh", "1e-15", "--gpu-repeat", "1",
           "--parallel-workers", "4", "--parallel-scfcrt", "0.000001", "--parallel-itry", "300"]
    if large:
        cmd += ["--large-workers", "6"]
    else:
        cmd.append("--skip-large")
    cmd += extra.split()
    t0 = time.time()
    rc = _stream(cmd, SRC, work / "check.log")
    print(f"total wall {time.time() - t0:.0f} s", flush=True)
    return rc


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=6 * 3600, volumes={RESULTS: volume})
def optcheck(label: str = "optcheck", pdb_id: str = "1UAO", gnorm: float = 1.0, cycles: int = 3000,
             extra: str = "", repeats: int = 1, cpu: bool = True, env: str = "") -> None:
    """The thermo check's GNORM optimization on the GPU and on the CPU from the same prepared PDB:
    does the optimization reach GNORM on both?  Prints every 10th cycle and the jumps."""
    import re

    sys.path.insert(0, f"{SRC}/scripts")
    sys.path.insert(0, f"{SRC}/tests")
    import check_mozyme_gpu_thermo as th
    import mozyme_md_workflow as md

    work = Path(RESULTS) / label
    pdb = md.prepare_pdb(Path(MOPAC), th.get_pdb(pdb_id, work), work / "prepare")
    cyc_re = re.compile(r"CYCLE:\s+(\d+).*GRAD\.:\s+([0-9.]+)\s+HEAT:\s+([-0-9.]+)")
    genv = dict(kv.split("=", 1) for kv in env.split(",") if "=" in kv) or None  # GPU runs only
    runs = [(f"gpu{i}" if repeats > 1 else "gpu", genv) for i in range(1, repeats + 1)]
    if cpu:
        runs.append(("cpu", {"MOPAC_NOGPU": "1"}))
    for name, env in runs:
        keys = f'{md.BASE_KEYS} GEO_DAT="{pdb.name}" CYCLES={cycles} PDBOUT GNORM={gnorm:g} {extra}'
        r = md.run_mopac(Path(MOPAC), work / name, "opt", keys, "geometry optimization", [pdb], env)
        rows = [(int(c), float(g), float(h)) for c, g, h in cyc_re.findall(r.out_text)]
        best = min(h for _, _, h in rows) if rows else None
        jumps = [(c, g, h) for (c, g, h), (_, _, hp) in zip(rows[1:], rows) if h - hp > 5.0]
        gn = re.findall(r"GRADIENT NORM\s*=\s*([-0-9.]+)", r.out_text)
        print(f"[{name}] {len(rows)} cycles, {r.wall:.1f} s, final heat {r.heat}, best {best}, "
              f"final GNORM {gn[-1] if gn else '?'}, heat jumps > 5 kcal/mol: {len(jumps)} "
              f"(first {jumps[:3]})", flush=True)
        for c, g, h in rows:
            if c % 25 == 0:
                print(f"   [{name}] cycle {c}: grad {g:.3f} heat {h:.4f}", flush=True)
        volume.commit()


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=3600, volumes={RESULTS: volume})
def gradcheck(pdb: str = "optcheck/gpu/opt.pdb", label: str = "gradcheck", repeats: int = 3, extra: str = "",
              cpu: bool = True) -> None:
    """1SCF GRADIENTS at one geometry (a PDB in the volume): CPU once, GPU `repeats` times.
    Heat and the Cartesian gradient (AUX) are compared: GPU - CPU and GPU run to run."""
    import re

    sys.path.insert(0, f"{SRC}/scripts")
    import mozyme_md_workflow as md

    src = Path(RESULTS) / pdb
    work = Path(RESULTS) / label

    def grad(run_dir: Path) -> list[float]:
        aux = (run_dir / "g.aux").read_text(errors="ignore")
        m = re.search(r"GRADIENTS:KCAL/MOL/ANGSTROM\[\d+\]=\s*(.*?)(?=\n\s*[A-Z_]+[:=\[])", aux, re.S)
        return [float(x) for x in m.group(1).split()] if m else []

    keys = f'{md.BASE_KEYS} 1SCF GRADIENTS AUX GEO_DAT="{src.name}" {extra}'
    results = {}
    runs = ([("cpu", {"MOPAC_NOGPU": "1"})] if cpu else []) + [(f"gpu{i}", None) for i in range(1, repeats + 1)]
    for name, env in runs:
        r = md.run_mopac(Path(MOPAC), work / name, "g", keys, "gradient check", [src], env)
        g = grad(work / name)
        gn = sum(x * x for x in g) ** 0.5
        results[name] = (r.heat, g)
        print(f"[{name}] heat {r.heat} kcal/mol, {len(g)} gradient components, |g| {gn:.4f}, "
              f"GPU SCF {r.statuses}", flush=True)
    volume.commit()

    def diff(a: str, b: str) -> None:
        (ha, ga), (hb, gb) = results[a], results[b]
        if len(ga) != len(gb) or not ga:
            print(f"{b} vs {a}: gradient lengths differ ({len(ga)} / {len(gb)})")
            return
        d = [y - x for x, y in zip(ga, gb)]
        rms = (sum(x * x for x in d) / len(d)) ** 0.5
        k = max(range(len(d)), key=lambda i: abs(d[i]))
        print(f"{b} - {a}: heat {hb - ha:+.6f} kcal/mol, gradient |diff| {sum(x * x for x in d) ** 0.5:.4f}, "
              f"RMS {rms:.4f}, max {d[k]:+.4f} kcal/(mol A) at component {k} (atom {k // 3 + 1}; "
              f"{a} {ga[k]:+.4f}, {b} {gb[k]:+.4f})", flush=True)

    for i in range(1, repeats + 1):
        if cpu:
            diff("cpu", f"gpu{i}")
    for i in range(2, repeats + 1):
        diff("gpu1", f"gpu{i}")


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=3600, volumes={RESULTS: volume})
def scfprobe(pdb: str = "optcheck/gpu/opt.pdb", label: str = "scfprobe", repeats: int = 2,
             configs: str = "SCFCRT=0.0001;SCFCRT=0.00001;SCFCRT=0.00001 THRESH=1.D-15;"
                            "SCFCRT=0.000001 THRESH=1.D-15;SCFCRT=0.000001 THRESH=1.D-15 ITRY=1000",
             envs: str = "", cpu: bool = True, drop: str = "", job: str = "1SCF") -> None:
    """GPU 1SCF at one geometry for several SCF criteria (CPU once each): iterations, final ovmax/energy
    change reported by the device, heat, and whether the host accepted the SCF."""
    import re

    sys.path.insert(0, f"{SRC}/scripts")
    import mozyme_md_workflow as md

    src = Path(RESULTS) / pdb
    base = md.BASE_KEYS.replace(" ITRY=200", "")
    for word in drop.split(","):  # keywords left out of the base set, e.g. PULAY,SHIFT=-50
        if word.strip():
            base = base.replace(" " + word.strip(), "")
    print("keywords:", base, flush=True)
    for ci, cfg in enumerate(c.strip() for c in configs.split(";") if c.strip()):
        itry = "" if "ITRY=" in cfg else " ITRY=200"
        keys = f'{base}{itry} {job} {cfg} GEO_DAT="{src.name}"'
        heats = {}
        # envs: variants "K=V,K=V;K=V" for the GPU runs (bisect the GPU stages)
        variants = [dict(kv.split("=", 1) for kv in v.split(",") if "=" in kv)
                    for v in envs.split(";")] if envs else [{}]
        runs = [("cpu", {"MOPAC_NOGPU": "1"})] if cpu else []
        for vi, venv in enumerate(variants):
            tag = ",".join(f"{k}={v}" for k, v in venv.items()) or "default"
            runs += [(f"gpu{i}[{tag}]" if venv else f"gpu{i}", venv or None) for i in range(1, repeats + 1)]
        for name, env in runs:
            sub = re.sub(r"[^A-Za-z0-9_.=-]+", "_", name)
            run_dir = Path(RESULTS) / label / f"c{ci}" / sub
            r = md.run_mopac(Path(MOPAC), run_dir, "s", keys, "scf probe", [src], env)
            (run_dir / "stdout_and_out.txt").write_text(r.out_text)  # device printf traces go to stdout
            it = re.findall(r"status=\S+ reason=\S+ code=\S+ iterations=\s*(\d+)", r.out_text)
            ov = re.findall(r"pls_ovmax_delta=\s*(\S+)\s+pls_energy_delta=\s*(\S+)", r.out_text)
            failed = "FAILED TO ACHIEVE SCF" in r.out_text or "SCF CALCULATION FAILED" in r.out_text
            heats[name] = r.heat
            print(f"[{cfg}] {name}: heat {r.heat}, {r.wall:.1f} s, iterations {it[-1] if it else '-'}, "
                  f"device ovmax/energy delta {ov[-1] if ov else '-'}, {'FAILED' if failed else 'ok'}", flush=True)
        if heats.get("cpu") is not None:
            d = {k: (v - heats["cpu"] if v is not None else None) for k, v in heats.items() if k != "cpu"}
            print(f"[{cfg}] GPU - CPU heat: {d}", flush=True)
        volume.commit()


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=12 * 3600, volumes={RESULTS: volume})
def force_gpu(pdb: str = "optcheck/gpu/opt.pdb", label: str = "force_gpu", temperature: float = 298.0,
              extra: str = "", workers: int = 0, itry: int = 200, sequential: bool = True,
              pdb_id: str = "", gnorm: float = 2.0) -> None:
    """GPU only: FORCE THERMO(T) at one geometry (a PDB in the volume) with the default MOZYME FORCE SCF
    (FORCE/THERMO set SCFCRT=1e-6 and THRESH=1.D-15 themselves), sequential and, with workers > 0, the
    parallel FORCE driver.  Prints the frequencies summary, ZPE, H, S, G and the time."""
    sys.path.insert(0, f"{SRC}/scripts")
    sys.path.insert(0, f"{SRC}/tests")
    import check_mozyme_gpu_thermo as th
    import mozyme_md_workflow as md
    import mozyme_parallel_force as pf

    _gpu_info()
    work = Path(RESULTS) / label
    if pdb_id:
        # prepare and optimize here (GPU), with the SCF tight enough for a meaningful GNORM
        raw = th.get_pdb(pdb_id, work)
        hydro = md.prepare_pdb(Path(MOPAC), raw, work / "prepare")
        opt_pdb, r = md.optimize(Path(MOPAC), hydro, work / "opt", cycles=3000,
                                 extra=f"GNORM={gnorm:g} SCFCRT=0.0001")
        print(f"[opt] {r.wall:.1f} s, final heat {r.heat}", flush=True)
        src = opt_pdb
    else:
        src = Path(RESULTS) / pdb
    base = md.BASE_KEYS.replace("ITRY=200", f"ITRY={itry}") + f" THERMO({temperature:g}) {extra}".rstrip()

    def report(name: str, text: str, wall: float) -> None:
        t = th.thermo(text, temperature)
        f = sorted(t["freqs"])
        stat = md.STATUS_RE.findall(text)
        print(f"[{name}] {wall:.1f} s, {len(f)} frequencies ({sum(1 for x in f if x < 0)} imaginary, "
              f"lowest {f[:3] if f else '-'}), ZPE {t.get('zpe')}, H {t.get('H')}, S {t.get('S')}, "
              f"G {t.get('G')}, GPU SCF {stat.count('success')}/{len(stat)} resident", flush=True)

    if sequential:
        r = md.run_mopac(Path(MOPAC), work / "seq", "force", f'{base} FORCE GEO_DAT="{src.name}"', "force",
                         [src], None)
        (work / "seq" / "stdout_and_out.txt").write_text(r.out_text)
        report("sequential", r.out_text, r.wall)
        volume.commit()
    if workers > 0:
        out, tim = pf.parallel_force(Path(MOPAC), src, base, workers, work / f"parallel_w{workers}")
        wall = tim["template_s"] + tim["workers_s"] + tim["final_s"]
        report(f"parallel {workers}", out.read_text(errors="ignore"), wall)
        volume.commit()


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=4 * 3600, volumes={RESULTS: volume})
def forcets_gpu(pdb: str = "crambin_force/opt/opt.pdb", label: str = "crambin_forcets", region: str = "A46=4",
                temperature: float = 298.0, extra: str = "LET") -> None:
    """GPU only: FORCETS THERMO(T) OPT("chain+residue"=radius) at one geometry (partial Hessian)."""
    sys.path.insert(0, f"{SRC}/scripts")
    sys.path.insert(0, f"{SRC}/tests")
    import check_mozyme_gpu_thermo as th
    import mozyme_md_workflow as md

    src = Path(RESULTS) / pdb
    sel, _, rad = region.partition("=")
    keys = f'{md.BASE_KEYS} FORCETS THERMO({temperature:g}) {extra} OPT("{sel}"={rad}) GEO_DAT="{src.name}"'
    r = md.run_mopac(Path(MOPAC), Path(RESULTS) / label, "forcets", keys, "forcets", [src], None)
    t = th.thermo(r.out_text, temperature)
    f = sorted(t["freqs"])
    print(f"[forcets {region}] {r.wall:.1f} s, {len(f)} frequencies, lowest {f[:4]}, ZPE {t.get('zpe')}", flush=True)
    volume.commit()


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=3600, volumes={RESULTS: volume})
def shell(cmd: str, cwd: str = RESULTS) -> None:
    """Run a shell command in the GPU container (diagnostics, profilers); output streamed."""
    rc = _stream(["bash", "-lc", cmd], cwd, Path("/tmp/shell.log"))
    print("rc", rc)

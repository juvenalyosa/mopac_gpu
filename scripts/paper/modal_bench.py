"""The paper benchmarks (paper_bench.py) on a Modal A100, as colab/paper_benchmarks_colab.ipynb does on Colab.

    pip install modal && modal token new                                   # once
    modal run scripts/paper/modal_bench.py::main --name prueba_modal --quick     # pipeline test (~30 min)
    modal run --detach scripts/paper/modal_bench.py::main --name a100_modal    # full run (hours)
    modal run scripts/paper/modal_bench.py::fetch --name a100_modal            # light zip -> ~/Downloads

The image builds this fork (GPU, OpenBLAS, the commit of origin/BRANCH at launch time) and MOPAC 23.1.2 (commit
78dc4572, CPU) once per commit.  The results go to the Modal volume "mopac-paper-bench" (<name>/runs, data, figures,
tables); a run whose input is unchanged is read back, so launching the same name again resumes, and each experiment
group is committed to the volume when it finishes.

CPU reference: 6 physical cores are reserved (12 vCPUs, as the Colab runtime) and the all-core CPU runs use 12
threads (--cpus 12); MOZYME CPU runs use one thread.  A container may burst above its reservation, so the CPU times
are those of 12 threads with at least 6 reserved cores.  The GPU is an A100 40 GB (--gpu A100-80GB to change).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

REPO_URL = "https://github.com/juvenalyosa/mopac_gpu.git"
BRANCH = os.environ.get("MOPAC_BENCH_BRANCH", "paper-bench")
UPSTREAM_URL = "https://github.com/openmopac/mopac.git"
UPSTREAM_COMMIT = "78dc4572"
CPU_CORES = 6            # physical cores (Modal); 12 vCPUs
THREADS = 12
GPU = os.environ.get("MOPAC_BENCH_GPU", "A100-40GB")


def _commit() -> str:
    """The commit of origin/BRANCH, resolved on the machine that launches the run (it keys the image cache)."""
    if os.environ.get("MOPAC_BENCH_COMMIT"):
        return os.environ["MOPAC_BENCH_COMMIT"]
    try:
        out = subprocess.run(["git", "ls-remote", REPO_URL, f"refs/heads/{BRANCH}"], capture_output=True, text=True,
                             timeout=60).stdout.split()
        return out[0] if out else BRANCH
    except Exception:  # noqa: BLE001  (inside the container the image is already built)
        return BRANCH


COMMIT = _commit()

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "gfortran", "ninja-build", "cmake", "libopenblas-dev", "liblapack-dev", "zip")
    .pip_install("numpy", "scipy", "pandas", "matplotlib", "openmm[cuda12]==8.6.1", "cupy-cuda12x")
    .run_commands(
        f"git clone -q {REPO_URL} /opt/mopac_gpu && git -C /opt/mopac_gpu checkout -q {COMMIT}",
        "cmake -S /opt/mopac_gpu -B /opt/build -G Ninja -DGPU=ON -DTESTS=OFF -DCMAKE_BUILD_TYPE=Release "
        "-DCUDA_ARCHS=80 -DENABLE_MKL=OFF -DBLA_VENDOR=OpenBLAS > /opt/cmake.log 2>&1 || (tail -40 /opt/cmake.log; exit 1)",
        "cmake --build /opt/build --target mopac --parallel 8 > /opt/build.log 2>&1 "
        "|| (grep -i -B2 -A6 error /opt/build.log | head -80; exit 1)",
        f"git clone -q {UPSTREAM_URL} /opt/mopac_upstream && git -C /opt/mopac_upstream checkout -q {UPSTREAM_COMMIT}",
        "cmake -S /opt/mopac_upstream -B /opt/build_upstream -G Ninja -DTESTS=OFF -DCMAKE_BUILD_TYPE=Release "
        "-DENABLE_MKL=OFF -DBLA_VENDOR=OpenBLAS > /opt/cmake_up.log 2>&1 || (tail -40 /opt/cmake_up.log; exit 1)",
        "cmake --build /opt/build_upstream --target mopac --parallel 8 > /opt/build_up.log 2>&1 "
        "|| (grep -i -B2 -A6 error /opt/build_up.log | head -80; exit 1)",
    )
    .env({"MOPAC_BENCH_COMMIT": COMMIT, "MOPAC_BENCH_BRANCH": BRANCH})
)

app = modal.App("mopac-paper-bench", image=image)
volume = modal.Volume.from_name("mopac-paper-bench", create_if_missing=True)
RESULTS = Path("/results")

# experiment groups in the order of the notebook; each is committed to the volume when it ends
GROUPS = ["env", "mozyme_sp,mozyme_grad,methods,cosmo", "opt,md,thermo",
          "solids,solids_profile,solids_force,solids_drc,conv_mol", "qmmm,qmmm_enzyme",
          "mozyme_threads,upstream,qmmm_pme_check"]


def _sh(cmd: str) -> int:
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, executable="/bin/bash").returncode


@app.function(gpu=GPU, cpu=CPU_CORES, memory=49152, timeout=24 * 3600, volumes={str(RESULTS): volume})
def bench(name: str, repeats: int = 3, quick: bool = False, groups: str = "") -> str:
    import glob
    src, out = Path("/opt/mopac_gpu"), RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    lib = sorted(glob.glob("/opt/build/libmopac.so*"))[0]
    _sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; "
        "cat /sys/fs/cgroup/cpu.max 2>/dev/null; git -C /opt/mopac_gpu log -1 --format='%h %ad %s' --date=short")
    _sh(f"cd {src}/scripts/qmmm && python -c \"from qmmm_pme import gpu_available; "
        f"print('QM-PME grid on the GPU (CuPy):', gpu_available())\"")
    base = (f"python -u {src}/scripts/paper/paper_bench.py --mopac /opt/build/mopac --libmopac {lib} --out {out} "
            f"--repeats {repeats} --cpus {THREADS} --upstream-mopac /opt/build_upstream/mopac"
            + (" --quick" if quick else ""))
    for g in (groups.split(";") if groups else GROUPS):
        _sh(f"{base} --exp {g}")
        volume.commit()
    _sh(f"python {src}/scripts/paper/paper_figures.py {out}")
    volume.commit()
    return (out / "summary.md").read_text() if (out / "summary.md").exists() else "no summary"


@app.function(volumes={str(RESULTS): volume}, timeout=1800)
def package(name: str) -> bytes:
    """The light zip of the notebook: data, figures, tables, summary, env, and the logs of failed and QM/MM runs."""
    import io
    import zipfile
    volume.reload()
    out = RESULTS / name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("data", "figures", "tables"):
            for f in sorted((out / sub).rglob("*")) if (out / sub).exists() else []:
                if f.is_file():
                    z.write(f, f"{name}/{f.relative_to(out)}")
        for f in ("summary.md", "env.json"):
            if (out / f).exists():
                z.write(out / f, f"{name}/{f}")
        runs = out / "runs"
        logs = list(runs.rglob("result_failed.json")) + list(runs.rglob("log.txt")) + list(runs.rglob("reference_check.log"))
        for f in logs:
            z.write(f, f"{name}/logs/{f.relative_to(runs)}")
            if f.name == "result_failed.json":
                for g in f.parent.glob("*.stdout"):
                    z.write(g, f"{name}/logs/{g.relative_to(runs)}")
    return buf.getvalue()


@app.local_entrypoint()
def main(name: str = "a100_modal", repeats: int = 3, quick: bool = False, groups: str = ""):
    print(f"commit {COMMIT} ({BRANCH}), GPU {GPU}, results volume mopac-paper-bench/{name}")
    print(bench.remote(name, repeats, quick, groups))
    fetch_zip(name)


def fetch_zip(name: str) -> None:
    data = package.remote(name)
    dst = Path.home() / "Downloads" / f"paper_bench_{name}.zip"
    dst.write_bytes(data)
    print(f"{len(data) / 1e6:.1f} MB -> {dst}")


@app.local_entrypoint()
def fetch(name: str = "a100_modal"):
    fetch_zip(name)

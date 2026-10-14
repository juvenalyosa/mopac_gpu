"""Kaggle GPU test: thermochemistry of chignolin in COSMO water with the smooth SWIG surface.

Builds MOPAC (GPU=ON) from https://github.com/juvenalyosa/mopac_gpu and, at the same time:
  GPU 0: chignolin (1UAO_opt.pdb) optimized with mozyme_md_workflow.optimize_stationary (GNORM=0.1, EPS=78.4
         SWIG HB-SMOOTH, stage 0 with the standard COSMO surface), then FORCE THERMO(298) of the minimum on
         both GPUs (mozyme_parallel_force, 4 workers);
  GPU 1: the DNA anion in water from the geometry where the run without HB-SMOOTH stuck on an H-bond kink
         (tests/data/mozyme_dna_cosmo/1BNA_anion_water_kink.arc), SWIG stages only, at most 2 x 45 min.
REPORT_CHIG_WATER.txt has every number.

    kaggle kernels push -p scripts/kaggle/mopac_chig_water
    kaggle kernels output juvenalyosa/mopac-chig-water -p ./chig_water_results
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

BRANCH = "main"
BASE = "/tmp/cw"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_CHIG_WATER.txt"


def say(msg):
    print(msg, flush=True)
    with open(REPORT, "a") as fh:
        fh.write(msg + "\n")


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        if check:
            raise SystemExit(f"failed: {cmd}")
    return r.stdout


def main():
    t_start = time.time()
    os.makedirs(BASE, exist_ok=True)
    open(REPORT, "w").close()
    say(sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader"))
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    sh("apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
       "--no-install-recommends gfortran libblas-dev liblapack-dev ninja-build >/dev/null 2>&1", check=False)
    sh(f"rm -rf {SRC} && git clone -q --depth 1 --branch {BRANCH} https://github.com/juvenalyosa/mopac_gpu.git {SRC}")
    say("commit: " + sh(f"git -C {SRC} log -1 --format='%h %s'").strip())
    cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1").strip().replace(".", "")
    sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=ON -DTESTS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo "
       f"-DCUDA_ARCHS={cc} > {BASE}/cmake.log 2>&1 && cmake --build {BUILD} --target mopac --parallel $(nproc) "
       f"> {BASE}/build.log 2>&1 || (grep -i error {BASE}/build.log | head -40; exit 1)")
    ngpu = int(sh("nvidia-smi -L | wc -l").strip() or 1)
    sys.path.insert(0, f"{SRC}/scripts")
    sys.path.insert(0, f"{SRC}/tests")
    import mozyme_md_workflow as md  # noqa: E402
    import mozyme_parallel_force as pf  # noqa: E402
    import check_mozyme_gpu_thermo as th  # noqa: E402
    starts = {"base": Path(f"{SRC}/tests/data/mozyme_gpu_reference/1UAO_opt.pdb"),
              "dna": Path(f"{SRC}/tests/data/mozyme_dna_cosmo/1BNA_anion_water_kink.arc")}
    os.makedirs(f"{BASE}/runs", exist_ok=True)
    final = {}

    def opt(gpu, key):
        try:
            t0 = time.perf_counter()
            kw = dict(gnorm=0.1, eps=78.4) if key == "base" else \
                dict(gnorm=0.2, eps=78.4, max_stages=2, extra="T=45M", prerelax_gnorm=None)
            geo, stages = md.optimize_stationary(Path(f"{BUILD}/mopac"), starts[key], Path(f"{BASE}/runs/{key}"),
                                                 env_extra={"CUDA_VISIBLE_DEVICES": str(gpu)}, **kw)
            final[key] = (geo, stages, time.perf_counter() - t0)
        except Exception as exc:
            say(f"{key}: optimization failed: {exc!r}")

    threads = [threading.Thread(target=opt, args=(k % ngpu, key)) for k, key in enumerate(starts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say("")
    say(f"1. optimize_stationary in COSMO water ({md.BASE_KEYS} EPS=78.4 SWIG); stage 0 = standard COSMO")
    for key in starts:
        if key not in final:
            continue
        geo, stages, wall = final[key]
        for st in stages:
            say(f"{key} stage {st['stage']}: {st['cycles']} cycles {st['wall']:.1f} s, heat {st['heat']}, "
                f"GNORM {st['gnorm_opt']}, fresh tight SCF {st['gnorm_fresh']}")
        say(f"{key}: {wall / 60:.1f} min in all")
    say("")
    say("2. FORCE THERMO(298) EPS=78.4 SWIG at the chignolin minimum (4 workers, both GPUs)")
    for key in ("base",):
        if key not in final:
            continue
        geo = final[key][0]
        t0 = time.perf_counter()
        try:
            out, _ = pf.parallel_force(Path(f"{BUILD}/mopac"), geo, f"{md.BASE_KEYS} EPS=78.4 SWIG THERMO(298)",
                                       2 * ngpu, Path(f"{BASE}/runs/{key}/force"))
            text = open(out, errors="ignore").read()
            t = th.thermo(text, 298.0)
            f = sorted(t["freqs"])
            say(f"{key}: FORCE {time.perf_counter() - t0:.0f} s, {len(f)} frequencies, "
                f"{sum(1 for x in f if x < 0)} imaginary, lowest {f[:6]}, ZPE {t.get('zpe')}, S {t.get('S')}, "
                f"H {t.get('H')}, G {t.get('G')}")
        except Exception as exc:
            say(f"{key}: FORCE failed after {time.perf_counter() - t0:.0f} s: {exc!r}")
    say(f"total {(time.time() - t_start) / 60:.0f} min")
    for p in Path(f"{BASE}/runs").rglob("*"):
        if p.suffix in (".den", ".res", ".brz") and p.is_file():
            p.unlink()
    shutil.make_archive(f"{OUT}/chig_water_runs", "gztar", f"{BASE}/runs")


main()

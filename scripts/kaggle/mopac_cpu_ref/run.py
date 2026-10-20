"""CPU references for the periodic cells of tests/data/materials on a Kaggle CPU machine (no GPU quota).

Builds https://github.com/juvenalyosa/mopac_gpu with GPU=OFF and runs the medium cells, so the GPU timings and
heats of the mopac-ctest kernel have a reference from a Linux CPU build.  Outputs go to /kaggle/working.

    kaggle kernels push -p scripts/kaggle/mopac_cpu_ref
    kaggle kernels output juvenalyosa/mopac-cpu-ref -p ./cpu_ref
"""

import os
import subprocess
import time

BASE = "/tmp/cr"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_CPU_REF.txt"
T0 = time.time()


def say(msg):
    print(msg, flush=True)
    with open(REPORT, "a") as fh:
        fh.write(msg + "\n")


def sh(cmd, check=False):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    if r.returncode != 0 and check:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"failed: {cmd}")
    return r.stdout + r.stderr


os.makedirs(BASE, exist_ok=True)
sh("apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
   "gfortran libblas-dev liblapack-dev ninja-build >/dev/null 2>&1")
sh(f"git clone -q --depth 1 https://github.com/juvenalyosa/mopac_gpu.git {SRC}", check=True)
say("commit: " + sh(f"git -C {SRC} log -1 --format='%h %s'").strip())
say("cpu: " + sh("nproc; grep -m1 'model name' /proc/cpuinfo").replace("\n", " ").strip())
sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=OFF -DCMAKE_BUILD_TYPE=Release > {OUT}/cmake.log 2>&1 && "
   f"cmake --build {BUILD} --parallel $(nproc) --target mopac > {OUT}/build.log 2>&1", check=True)
say(f"build: {time.time() - T0:.0f} s")

for deck in ("quartz.mop", "silicon.mop", "rutile.mop"):
    d = f"{BASE}/medium"
    sh(f"mkdir -p {d} && cp {SRC}/tests/data/materials/medium/{deck} {d}/")
    t = time.time()
    sh(f"cd {d} && {BUILD}/mopac {deck} > /dev/null 2>&1")
    out = f"{d}/{deck[:-4]}.out"
    txt = open(out, errors="ignore").read() if os.path.exists(out) else ""
    h = [l.split()[5] for l in txt.splitlines() if "FINAL HEAT OF FORMATION" in l]
    n = txt.count("TIME FOR ITERATION")
    say(f"medium {deck[:-4]:8s} {time.time() - t:7.1f} s  heat {h[-1] if h else None}  iterations {n}")
    sh(f"cp {out} {OUT}/cpu_medium_{deck[:-4]}.out")
say(f"total {time.time() - T0:.0f} s")

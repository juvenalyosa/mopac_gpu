"""Pair gradient of the conventional SCF on the GPU (MOPAC_GPU_CONV_GRAD=1), checked against the CPU on a Kaggle GPU.

With MOPAC_GPU_GRAD_CHECK=1 dcart evaluates the pair loop on the CPU and on the device, prints
"[GPU gradient] check pairs=.. ms=.. cpu_ms=.. max_abs_diff=.." and keeps the CPU result.  Decks:
tests/data/materials/grad (si144 and quartz162: solids with d orbitals; urea_cluster: a 192-atom molecule)
and tests/data/materials/small/urea.mop (sp solid), then PRECISE (central differences) for urea and quartz162.

    kaggle kernels push -p scripts/kaggle/mopac_grad_gpu
    kaggle kernels output juvenalyosa/mopac-grad-gpu -p ./grad_results
"""

import os
import subprocess
import time

BASE = "/tmp/gg"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_GRAD.txt"
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
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
sh("apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
   "gfortran libblas-dev liblapack-dev ninja-build >/dev/null 2>&1")
sh(f"git clone -q --depth 1 https://github.com/juvenalyosa/mopac_gpu.git {SRC}", check=True)
say("commit: " + sh(f"git -C {SRC} log -1 --format='%h %s'").strip())
say("gpu: " + sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1").strip())
cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1").strip().replace(".", "")
sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS={cc} > {OUT}/cmake.log 2>&1 "
   f"&& cmake --build {BUILD} --parallel $(nproc) --target mopac > {OUT}/build.log 2>&1", check=True)
say(f"build: {time.time() - T0:.0f} s")

MAT = f"{SRC}/tests/data/materials"
runs = [("si144", f"{MAT}/grad/si144.mop", ""), ("quartz162", f"{MAT}/grad/quartz162.mop", ""),
        ("urea_cluster", f"{MAT}/grad/urea_cluster.mop", ""), ("urea", f"{MAT}/small/urea.mop", ""),
        ("urea", f"{MAT}/small/urea.mop", " PRECISE"), ("quartz162", f"{MAT}/grad/quartz162.mop", " PRECISE")]
for name, path, extra in runs:
    label = name + extra.replace(" ", "_")
    d = f"{BASE}/{label}"
    lines = open(path).read().splitlines()
    lines[0] += extra
    os.makedirs(d, exist_ok=True)
    open(f"{d}/{name}.mop", "w").write("\n".join(lines) + "\n")
    t = time.time()
    sh(f"cd {d} && MOPAC_GPU_DEBUG=1 MOPAC_GPU_GRAD_CHECK=1 timeout 900 "
       f"{BUILD}/mopac {name}.mop > stdout.txt 2>&1")
    out = open(f"{d}/{name}.out", errors="ignore").read().splitlines() if os.path.exists(f"{d}/{name}.out") else []
    heat = [l.split()[5] for l in out if "FINAL HEAT OF FORMATION" in l]
    chk = [l.strip() for l in out if "[GPU gradient]" in l]
    say(f"{label:16s} {time.time() - t:7.1f} s  heat {heat[-1] if heat else None}")
    for c in chk:
        say("    " + c)
    say("    stderr: " + " | ".join(sh(f"tail -3 {d}/stdout.txt").splitlines()))
    sh(f"cp {d}/{name}.out {OUT}/{label}.out")
say(f"total {time.time() - T0:.0f} s")

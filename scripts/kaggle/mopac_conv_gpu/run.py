"""Conventional (non-MOZYME) SCF with the GPU on a Kaggle GPU: which device path stalls the urea crystal.

tests/data/materials/small/urea.mop (192 atoms, 480 orbitals, 288 occupied) ran past 600 s with the GPU and in
20 s with MOPAC_NOGPU=1 (mopac-ctest v3); v1 of this kernel: only MOPAC_CPU_DENSITY=1 converged.  Runs it with PL and MOPAC_GPU_DEBUG=1, each at most 120 s, switching off one
device path at a time; REPORT_CONV.txt gets the time, the heat, the last SCF iteration lines and stderr.

    kaggle kernels push -p scripts/kaggle/mopac_conv_gpu
    kaggle kernels output juvenalyosa/mopac-conv-gpu -p ./conv_results
"""

import os
import subprocess
import time

BRANCH = "main"
BASE = "/tmp/cg"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_CONV.txt"
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
sh(f"git clone -q --depth 1 --branch {BRANCH} https://github.com/juvenalyosa/mopac_gpu.git {SRC}", check=True)
say("commit: " + sh(f"git -C {SRC} log -1 --format='%h %s'").strip())
cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1").strip().replace(".", "")
sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS={cc} > {OUT}/cmake.log 2>&1 "
   f"&& cmake --build {BUILD} --parallel $(nproc) --target mopac > {OUT}/build.log 2>&1", check=True)
say(f"build: {time.time() - T0:.0f} s")

deck = open(f"{SRC}/tests/data/materials/small/urea.mop").read().splitlines()
deck[0] = deck[0] + " PL"
variants = [("NOGPU", "MOPAC_NOGPU=1"), ("default", ""), ("RESIDENT_SCF=0", "MOPAC_RESIDENT_SCF=0"),
            ("EIGEN_MIN=1e6", "MOPAC_GPU_EIGEN_MIN=1000000"),
            ("DENSITY_DEBUG", "MOPAC_GPU_DENSITY_DEBUG=1")]
for label, env in variants:
    d = f"{BASE}/{label.replace('=', '_')}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/urea.mop", "w").write("\n".join(deck) + "\n")
    t = time.time()
    sh(f"cd {d} && MOPAC_GPU_DEBUG=1 {env} timeout 120 {BUILD}/mopac urea.mop > stdout.txt 2>&1")
    out = open(f"{d}/urea.out", errors="ignore").read().splitlines() if os.path.exists(f"{d}/urea.out") else []
    heat = [l.strip() for l in out if "FINAL HEAT OF FORMATION" in l]
    iters = [l.strip() for l in out if "ITERATION" in l or "PLS" in l or l.strip().startswith("ITER")]
    say(f"\n== {label}: {time.time() - t:.0f} s, {heat[-1] if heat else 'no heat'}, {len(iters)} iteration lines")
    for l in iters[:3] + ["..."] + iters[-4:]:
        say("   " + l)
    say("   stdout/stderr: " + " | ".join(open(f"{d}/stdout.txt", errors="ignore").read().splitlines()[-6:]))
    sh(f"cp {d}/urea.out {OUT}/urea_{label.replace('=', '_')}.out")
say(f"total {time.time() - T0:.0f} s")

"""Solids, conventional SCF on a Kaggle GPU: two-centre Fock terms on the device (MOPAC_GPU_CONV_FOCK=1).

MOPAC_GPU_FOCK_CHECK=1 builds F on the device and on the CPU and prints "[GPU Fock] check max|F_gpu - F_cpu|=" for
the first SCF iterations (CPU result kept) on Si, quartz and rutile cells of 144-162 atoms; then medium quartz with
the device Fock build on and MOPAC_GPU_PROFILE=1.  (v1 validated the pair gradient, v2 the hcore image sums, v3
profiled the medium cells.)

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
CHECK = "MOPAC_GPU_DEBUG=1 MOPAC_GPU_CONV_FOCK=1 MOPAC_GPU_FOCK_CHECK=1"
ON = "MOPAC_GPU_CONV_FOCK=1 MOPAC_GPU_PROFILE=1"
runs = [("si144", f"{MAT}/grad/si144.mop", "", CHECK), ("quartz162", f"{MAT}/grad/quartz162.mop", "", CHECK),
        ("rutile162", f"{MAT}/grad/rutile162.mop", "", CHECK), ("quartz", f"{MAT}/medium/quartz.mop", "", ON)]
for name, path, extra, env in runs:
    label = name + extra.replace(" ", "_") + ("_check" if "CHECK" in env else "_on")
    d = f"{BASE}/{label}"
    lines = open(path).read().splitlines()
    lines[0] += extra
    os.makedirs(d, exist_ok=True)
    open(f"{d}/{name}.mop", "w").write("\n".join(lines) + "\n")
    t = time.time()
    sh(f"cd {d} && {env} timeout 900 {BUILD}/mopac {name}.mop > stdout.txt 2>&1")
    out = open(f"{d}/{name}.out", errors="ignore").read().splitlines() if os.path.exists(f"{d}/{name}.out") else []
    heat = [l.split()[5] for l in out if "FINAL HEAT OF FORMATION" in l]
    chk = [l.strip() for l in out if "[GPU hcore]" in l or "[GPU gradient]" in l or "check max|F_gpu" in l or
           ("[PROFILE]" in l and any(k in l for k in ("conv_", "hcore_total", "deriv_dcart", "dcart_")))]
    stages = [l.strip() for l in out if ("AFTER  HCORE" in l or "AFTER  DERIV" in l or "AFTER  ITER" in l)]
    stages += [l.strip() for l in out if "AFTER  FOCKS" in l][-1:] + [l.strip() for l in out if "AFTER  FULL DIAG" in l][-1:]
    stages += [f"iterations {sum('TIME FOR ITERATION' in l for l in out)}"]
    say(f"{label:18s} {time.time() - t:7.1f} s  heat {heat[-1] if heat else None}")
    for c in chk + stages:
        say("    " + c)
    say("    stderr: " + " | ".join(sh(f"tail -3 {d}/stdout.txt").splitlines()))
    sh(f"cp {d}/{name}.out {OUT}/{label}.out")
say(f"total {time.time() - T0:.0f} s")

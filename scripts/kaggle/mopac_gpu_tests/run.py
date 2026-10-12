"""MOPAC MOZYME GPU tests on a Kaggle GPU (P100 by default; T4 is slow in double precision).

Runs scripts/habrok/mopac_gpu_tests.sbatch as a plain bash script (no SLURM, no modules): installs the
build tools, clones https://github.com/juvenalyosa/mopac_gpu, builds MOPAC with GPU=ON for this GPU and runs
every test.  REPORT.txt, the full console log and results_<id>.tar.gz end up in /kaggle/working (the kernel
output).  Push from a computer with the Kaggle CLI:

    kaggle kernels push -p scripts/kaggle/mopac_gpu_tests
    kaggle kernels status juvenalyosa/mopac-gpu-tests
    kaggle kernels output juvenalyosa/mopac-gpu-tests -p ./kaggle_results

Settings below: BRANCH, QUICK (short pipeline check), RUN_LARGE (full crambin FORCE), RUN_NSYS.
"""

import os
import shutil
import subprocess
import sys
import time

BRANCH = "main"
QUICK = os.environ.get("QUICK", "0")
RUN_LARGE = os.environ.get("RUN_LARGE", "0")
RUN_NSYS = os.environ.get("RUN_NSYS", "0")

BASE = "/tmp/mopac_kaggle"          # work area (not saved)
OUT = "/kaggle/working"             # saved as the kernel output
SRC = f"{BASE}/mopac_gpu"
LOG = f"{OUT}/console.log"


def sh(cmd, check=False, log=True):
    print(f"$ {cmd}", flush=True)
    with open(LOG, "a") as fh:
        fh.write(f"$ {cmd}\n")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             executable="/bin/bash")
        for line in p.stdout:
            if log:
                print(line, end="", flush=True)
            fh.write(line)
        rc = p.wait()
    if check and rc != 0:
        raise SystemExit(f"command failed ({rc}): {cmd}")
    return rc


os.makedirs(BASE, exist_ok=True)
os.makedirs(f"{BASE}/mopac_gpu_runs", exist_ok=True)
t0 = time.time()
sh("nvidia-smi; nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv")
# CUDA toolkit of the image
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
sh("nvcc --version || ls /usr/local | grep -i cuda")
# build tools: gfortran matching the system gcc, BLAS/LAPACK, ninja, a recent cmake
sh("apt-get update -qq >/dev/null 2>&1; "
   "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
   "gfortran libblas-dev liblapack-dev ninja-build >/dev/null 2>&1; gcc --version | head -1; gfortran --version | head -1",
   log=True)
sh("cmake --version | head -1 || pip install -q cmake")
sh(f"rm -rf {SRC} && git clone -q --depth 1 --branch {BRANCH} https://github.com/juvenalyosa/mopac_gpu.git {SRC} "
   f"&& git -C {SRC} log -1 --format='%h %ad %s' --date=short", check=True)

env = (f"SCRATCH_BASE={BASE} SRC={SRC} NO_PULL=1 QUICK={QUICK} RUN_LARGE={RUN_LARGE} RUN_NSYS={RUN_NSYS} "
       f"SLURM_CPUS_PER_TASK=$(nproc) SLURM_JOB_ID=kaggle")
rc = sh(f"{env} bash {SRC}/scripts/habrok/mopac_gpu_tests.sbatch")

# results into the kernel output
runs = sorted(d for d in os.listdir(f"{BASE}/mopac_gpu_runs") if d.endswith("_jobkaggle"))
if runs:
    run = f"{BASE}/mopac_gpu_runs/{runs[-1]}"
    for name in ("REPORT.txt",):
        if os.path.exists(f"{run}/{name}"):
            shutil.copy2(f"{run}/{name}", f"{OUT}/{name}")
tar = f"{BASE}/mopac_gpu_runs/results_kaggle.tar.gz"
if os.path.exists(tar):
    shutil.copy2(tar, f"{OUT}/results_kaggle.tar.gz")
print(f"\nexit {rc}, total {time.time() - t0:.0f} s; outputs: {os.listdir(OUT)}", flush=True)

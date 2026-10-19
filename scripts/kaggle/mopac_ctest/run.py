"""MOPAC test suite (ctest) with a GPU build on a Kaggle GPU, keeping every output file.

Builds https://github.com/juvenalyosa/mopac_gpu with GPU=ON TESTS=ON, runs ctest without the periodic test
`port` with the GPU in use and with MOPAC_NOGPU=1, then `port` alone (GPU, MOPAC_NOGPU=1, the opt-in device Fock
build MOPAC_GPU_EXACT_SC=1; each at most 15 min), keywords/PM3 with MOPAC_GPU_EXACT_SC=1, and the API test under
gdb.  The kernel output gets ctest_outputs_gpu.tar.gz / ctest_outputs_nogpu.tar.gz (every .out of each ctest
run), ctest_references.tar.gz, the ctest logs and REPORT_CTEST.txt.

    kaggle kernels push -p scripts/kaggle/mopac_ctest
    kaggle kernels output juvenalyosa/mopac-ctest -p ./ctest_results
"""

import os
import subprocess
import time

BRANCH = "main"
BASE = "/tmp/ct"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_CTEST.txt"
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
   "gfortran libblas-dev liblapack-dev ninja-build gdb >/dev/null 2>&1")
sh(f"git clone -q --depth 1 --branch {BRANCH} https://github.com/juvenalyosa/mopac_gpu.git {SRC}", check=True)
say("commit: " + sh(f"git -C {SRC} log -1 --format='%h %s'").strip())
say("gpu: " + sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1").strip())
cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1").strip().replace(".", "")
sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=ON -DTESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCUDA_ARCHS={cc} "
   f"> {OUT}/cmake.log 2>&1 && cmake --build {BUILD} --parallel $(nproc) -- -k 0 > {OUT}/build.log 2>&1", check=True)
say(f"build: {time.time() - T0:.0f} s")

t = time.time()
sh(f"cd {BUILD} && ctest -j $(nproc) -E '^port$' --timeout 900 --output-on-failure > {OUT}/ctest_gpu.log 2>&1")
say(f"ctest without port: {time.time() - t:.0f} s, " + sh(f"grep 'tests passed' {OUT}/ctest_gpu.log").strip())
sh(f"cd {BUILD}/tests && find . -name '*.out' | tar -czf {OUT}/ctest_outputs_gpu.tar.gz -T -")
t = time.time()
sh(f"cd {BUILD} && MOPAC_NOGPU=1 ctest -j $(nproc) -E '^port$' --timeout 900 > {OUT}/ctest_nogpu.log 2>&1")
say(f"ctest without port, MOPAC_NOGPU=1: {time.time() - t:.0f} s, " + sh(f"grep 'tests passed' {OUT}/ctest_nogpu.log").strip())

# the periodic test with the GPU in use and off, then the opt-in device Fock build (MOPAC_GPU_EXACT_SC=1)
for label, env in (("GPU", ""), ("MOPAC_NOGPU=1", "MOPAC_NOGPU=1"), ("MOPAC_GPU_EXACT_SC=1", "MOPAC_GPU_EXACT_SC=1")):
    tag = label.replace("=", "_")
    d = f"{BASE}/port_{tag}"
    sh(f"mkdir -p {d} && cp {SRC}/tests/port.mop {d}/")
    t = time.time()
    sh(f"cd {d} && {env} timeout {300 if 'EXACT' in env else 900} {BUILD}/mopac port.mop > stdout.txt 2>&1")
    heats = sh(f"grep 'FINAL HEAT OF FORMATION' {d}/port.out | awk '{{print $6}}'").split()
    say(f"port, {label}: {time.time() - t:.0f} s, {len(heats)} heats {heats[:8]}")
    sh(f"cp {d}/port.out {OUT}/port_{tag}.out")
# a molecule (keywords/PM3) with the opt-in device Fock build, against the reference
d = f"{BASE}/pm3_exact_sc"
sh(f"mkdir -p {d} && cp {SRC}/tests/keywords/PM3.mop {d}/")
sh(f"cd {d} && MOPAC_GPU_EXACT_SC=1 timeout 300 {BUILD}/mopac PM3.mop > stdout.txt 2>&1; cp PM3.out {OUT}/PM3_exact_sc.out")
say("PM3, MOPAC_GPU_EXACT_SC=1: " + sh(f"grep 'FINAL HEAT OF FORMATION' {d}/PM3.out").strip() + " | reference: "
    + sh(f"grep 'FINAL HEAT OF FORMATION' {SRC}/tests/keywords/PM3.out").strip())
say("port reference heats: " + str(sh(f"grep 'FINAL HEAT OF FORMATION' {SRC}/tests/port.out | awk '{{print $6}}'").split()[:8]))

# backtrace of the API test
say("mopac-api-test under gdb:")
say(sh(f"cd {BUILD}/tests && gdb -batch -ex run -ex bt {BUILD}/mopac-api-test 2>&1 | tail -25"))

sh(f"cd {BUILD}/tests && find . -name '*.out' | tar -czf {OUT}/ctest_outputs_nogpu.tar.gz -T - ; "
   f"cd {SRC}/tests && find . -name '*.out' | tar -czf {OUT}/ctest_references.tar.gz -T -")
say(f"total {time.time() - T0:.0f} s")

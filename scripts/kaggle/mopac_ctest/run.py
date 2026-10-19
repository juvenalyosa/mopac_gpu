"""MOPAC test suite (ctest) with a GPU build on a Kaggle GPU, keeping every output file.

Builds https://github.com/juvenalyosa/mopac_gpu with GPU=ON TESTS=ON, runs ctest without the periodic test
`port`, then `port` alone with the device Fock build on and off (MOPAC_GPU_EXACT_SC=0), each at most 15 min,
and the API test under gdb for a backtrace.  The kernel output gets ctest_outputs.tar.gz (every .out of the
tests and the reference .out next to it), the ctest logs and REPORT_CTEST.txt.

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
t = time.time()
sh(f"cd {BUILD} && MOPAC_NOGPU=1 ctest -j $(nproc) -E '^port$' --timeout 900 > {OUT}/ctest_nogpu.log 2>&1")
say(f"ctest without port, MOPAC_NOGPU=1: {time.time() - t:.0f} s, " + sh(f"grep 'tests passed' {OUT}/ctest_nogpu.log").strip())

# the periodic test: device Fock build on (default) and off
for label, env in (("device Fock on", ""), ("MOPAC_GPU_EXACT_SC=0", "MOPAC_GPU_EXACT_SC=0"),
                   ("MOPAC_NOGPU=1", "MOPAC_NOGPU=1")):
    d = f"{BASE}/port_{label.split()[0].replace('=', '_')}"
    sh(f"mkdir -p {d} && cp {SRC}/tests/port.mop {d}/")
    t = time.time()
    sh(f"cd {d} && {env} timeout 900 {BUILD}/mopac port.mop > stdout.txt 2>&1")
    heats = sh(f"grep 'FINAL HEAT OF FORMATION' {d}/port.out | awk '{{print $6}}'").split()
    say(f"port, {label}: {time.time() - t:.0f} s, {len(heats)} heats {heats}")
    sh(f"cp {d}/port.out {OUT}/port_{label.split()[0].replace('=', '_')}.out")
say("port reference heats: " + str(sh(f"grep 'FINAL HEAT OF FORMATION' {SRC}/tests/port.out | awk '{{print $6}}'").split()))

# backtrace of the API test
say("mopac-api-test under gdb:")
say(sh(f"cd {BUILD}/tests && gdb -batch -ex run -ex bt {BUILD}/mopac-api-test 2>&1 | tail -40"))

sh(f"cd {BUILD}/tests && find . -name '*.out' | tar -czf {OUT}/ctest_outputs.tar.gz -T - ; "
   f"cd {SRC}/tests && find . -name '*.out' | tar -czf {OUT}/ctest_references.tar.gz -T -")
say(f"total {time.time() - T0:.0f} s")

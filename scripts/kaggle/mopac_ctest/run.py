"""MOPAC test suite (ctest) with a GPU build on a Kaggle GPU, keeping every output file.

Builds https://github.com/juvenalyosa/mopac_gpu with GPU=ON TESTS=ON, runs ctest with the GPU in use and
with MOPAC_NOGPU=1, then the periodic cells of tests/data/materials (small: GPU and MOPAC_NOGPU=1; medium: GPU,
and quartz also MOPAC_NOGPU=1) and the API test under gdb.  The kernel output gets ctest_outputs_gpu.tar.gz / ctest_outputs_nogpu.tar.gz (every .out of each ctest
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
sh(f"cd {BUILD} && ctest -j $(nproc) --timeout 900 --output-on-failure > {OUT}/ctest_gpu.log 2>&1")
say(f"ctest: {time.time() - t:.0f} s, " + sh(f"grep 'tests passed' {OUT}/ctest_gpu.log").strip())
sh(f"cd {BUILD}/tests && find . -name '*.out' | tar -czf {OUT}/ctest_outputs_gpu.tar.gz -T -")
t = time.time()
sh(f"cd {BUILD} && MOPAC_NOGPU=1 ctest -j $(nproc) --timeout 900 > {OUT}/ctest_nogpu.log 2>&1")
say(f"ctest, MOPAC_NOGPU=1: {time.time() - t:.0f} s, " + sh(f"grep 'tests passed' {OUT}/ctest_nogpu.log").strip())

# periodic cells (tests/data/materials): GPU against MOPAC_NOGPU=1; MOPAC_GPU_DEBUG=1 shows whether lgpu stays on
def heat_grad(path):
    txt = open(path, errors="ignore").read() if os.path.exists(path) else ""
    h = [l.split()[5] for l in txt.splitlines() if "FINAL HEAT OF FORMATION" in l]
    g = [l.split()[-1] for l in txt.splitlines() if "GRADIENT NORM" in l]
    gpu = [l.strip() for l in txt.splitlines() if l.strip().startswith(("lgpu", "hasGPU", "[GPU SCF] norbs"))]
    return (h[-1] if h else None), (g[-1] if g else None), gpu[:2]


for size, decks, modes, limit in (("small", sorted(os.listdir(f"{SRC}/tests/data/materials/small")), ("GPU", "NOGPU"), 600),
                                  ("medium", ["quartz.mop", "silicon.mop", "rutile.mop"], ("GPU",), 1500),
                                  ("medium", ["quartz.mop"], ("NOGPU",), 1500)):
    for deck in decks:
        for mode in modes:
            d = f"{BASE}/mat_{size}_{mode}"
            sh(f"mkdir -p {d} && cp {SRC}/tests/data/materials/{size}/{deck} {d}/")
            env = "MOPAC_GPU_DEBUG=1 " + ("MOPAC_NOGPU=1" if mode == "NOGPU" else "")
            t = time.time()
            sh(f"cd {d} && {env} timeout {limit} {BUILD}/mopac {deck} > /dev/null 2>&1")
            h, g, gpu = heat_grad(f"{d}/{deck[:-4]}.out")
            say(f"{size} {deck[:-4]:10s} {mode:5s} {time.time() - t:7.1f} s  heat {h}  gnorm {g}  {gpu}")
            sh(f"cp {d}/{deck[:-4]}.out {OUT}/mat_{size}_{mode}_{deck[:-4]}.out")

# backtrace of the API test
say("mopac-api-test under gdb:")
say(sh(f"cd {BUILD}/tests && gdb -batch -ex run -ex bt {BUILD}/mopac-api-test 2>&1 | grep -v Thread | tail -8"))

sh(f"cd {BUILD}/tests && find . -name '*.out' | tar -czf {OUT}/ctest_outputs_nogpu.tar.gz -T - ; "
   f"cd {SRC}/tests && find . -name '*.out' | tar -czf {OUT}/ctest_references.tar.gz -T -")
say(f"total {time.time() - T0:.0f} s")

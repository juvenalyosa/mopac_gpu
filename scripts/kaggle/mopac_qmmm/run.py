"""Kaggle GPU test of QM/MM molecular dynamics: OpenMM (CUDA) + MOPAC/MOZYME (GPU or CPU) through
scripts/qmmm/run_qmmm_md.py.

Builds MOPAC (GPU=ON, libmopac) from https://github.com/juvenalyosa/mopac_gpu, installs OpenMM 8.6.1 with
CUDA, and runs Langevin NVT (0.5 fs) plus an NVE check for QM regions of increasing size in TIP3P water:
Tyr side chain of chignolin (16 atoms with the link atom), the whole chignolin (138), the whole crambin
(~640); MOPAC on the GPU and on the CPU (MOPAC_NOGPU=1).  REPORT_QMMM.txt has every number.

    kaggle kernels push -p scripts/kaggle/mopac_qmmm
    kaggle kernels output juvenalyosa/mopac-qmmm -p ./qmmm_results
"""

import os
import subprocess
import time

BRANCH = "main"
BASE = "/tmp/qmmm"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_QMMM.txt"


def say(msg):
    print(msg, flush=True)
    with open(REPORT, "a") as fh:
        fh.write(msg + "\n")


def sh(cmd, check=True, env=None):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
        if check:
            raise SystemExit(f"failed: {cmd}")
    return r.stdout + r.stderr


def main():
    t_start = time.time()
    os.makedirs(BASE, exist_ok=True)
    open(REPORT, "w").close()
    say(sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader"))
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    sh("apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
       "--no-install-recommends gfortran libopenblas-dev liblapack-dev ninja-build >/dev/null 2>&1", check=False)
    sh("pip install -q 'openmm[cuda12]==8.6.1' 2>&1 | tail -2", check=False)
    say(sh("python -c \"import openmm as m; print('OpenMM', m.__version__, [m.Platform.getPlatform(i).getName() "
           "for i in range(m.Platform.getNumPlatforms())])\""))
    sh(f"rm -rf {SRC} && git clone -q --depth 1 --branch {BRANCH} https://github.com/juvenalyosa/mopac_gpu.git {SRC}")
    say("commit: " + sh(f"git -C {SRC} log -1 --format='%h %s'").strip())
    cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1").strip().replace(".", "")
    # OpenBLAS, not the image's MKL: libmopac is loaded into Python (with numpy), where MKL failed to
    # load its own kernels ("Cannot load libmkl_avx512.so")
    sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=ON -DTESTS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo "
       f"-DENABLE_MKL=OFF -DBLA_VENDOR=OpenBLAS -DCUDA_ARCHS={cc} > {BASE}/cmake.log 2>&1 && cmake --build {BUILD} --target mopac --parallel $(nproc) "
       f"> {BASE}/build.log 2>&1 || (grep -i error {BASE}/build.log | head -40; exit 1)")
    lib = sh(f"ls {BUILD}/libmopac.so* | head -1").strip()
    say(f"libmopac: {lib}")
    say(sh(f"ldd {lib} | grep -i -E 'blas|lapack|mkl'", check=False).strip())
    data = f"{SRC}/tests/data/mozyme_gpu_reference"
    script = f"{SRC}/scripts/qmmm/run_qmmm_md.py"
    cases = [
        ("Tyr2 side chain (chignolin)", f"{data}/1UAO_opt.pdb", "--qm-residues 2 --qm-sidechain --padding 1.4", 1000, 200),
        ("whole chignolin", f"{data}/1UAO_opt.pdb", "--qm-residues 1,2,3,4,5,6,7,8,9,10 --padding 1.8 --ron 6 --roff 8",
         400, 100),
        ("whole crambin", f"{data}/1CRN_opt.pdb",
         "--qm-residues " + ",".join(str(i) for i in range(1, 47)) + " --padding 1.8 --ron 6 --roff 8", 100, 50),
    ]
    for name, pdb, opts, steps, nve in cases:
        for dev in ("GPU", "CPU"):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
            if dev == "CPU":
                env["MOPAC_NOGPU"] = "1"
                steps_run, nve_run = max(10, steps // 10), 0
            else:
                steps_run, nve_run = steps, nve
            d = f"{BASE}/runs/{name.split()[0]}_{name.split()[-1]}_{dev}".replace("(", "").replace(")", "")
            os.makedirs(d, exist_ok=True)
            t0 = time.perf_counter()
            out = sh(f"cd {d} && python -u {script} {pdb} --libmopac {lib} {opts} --platform CUDA "
                     f"--steps {steps_run} --nve {nve_run} --mm-equil-ps 10", check=False, env=env)
            open(f"{d}/log.txt", "w").write(out)
            say("")
            say(f"== {name}, MOPAC on the {dev} ({time.perf_counter() - t0:.0f} s in all)")
            for line in out.splitlines():
                if line.startswith(("system:", "QM region", "QM/MM NVT", "NVE check", "MM equilibration")) or \
                        "Error" in line or "error" in line:
                    say("   " + line.strip()[:300])
    say(f"total {(time.time() - t_start) / 60:.0f} min")


main()

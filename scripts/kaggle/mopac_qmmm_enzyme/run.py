"""Kaggle example of QM/MM molecular dynamics of an enzyme: trypsin with its inhibitor BPTI (PDB 2PTC) in
TIP3P water, OpenMM (CUDA) + MOPAC/MOZYME, through scripts/qmmm/run_qmmm_md.py.

QM region (PM7, 54 atoms + 5 H link atoms, charge 0): the catalytic triad side chains His57 (HID), Asp102,
Ser195 of trypsin and Lys15-Ala16 of BPTI (the scissile peptide bond, attacked by Ser195 OG).  Langevin NVT,
an NVE check, the distances of the active site, and a cut-off test of the QM/MM electrostatics (with and
without the long-range part, QM-PME).

Builds MOPAC (GPU=ON, libmopac) from https://github.com/juvenalyosa/mopac_gpu and installs OpenMM 8.6.1 with
CUDA.  MM equilibration 100 ps, then 5 ps of QM/MM NVT (0.5 fs) and 0.5 ps NVE.  REPORT_QMMM_ENZYME.txt has
every number, log_trypsin.txt the whole output, qmmm_traj.pdb the trajectory.

    kaggle kernels push -p scripts/kaggle/mopac_qmmm_enzyme
    kaggle kernels output juvenalyosa/mopac-qmmm-enzyme -p ./enzyme_results
"""

import os
import subprocess
import time

BRANCH = "main"
BASE = "/tmp/qmmm"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_QMMM_ENZYME.txt"


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
    pdb = f"{BASE}/2PTC.pdb"
    sh(f"curl -sf -o {pdb} https://files.rcsb.org/download/2PTC.pdb")
    script = f"{SRC}/scripts/qmmm/run_qmmm_md.py"
    d = f"{BASE}/runs/trypsin"
    os.makedirs(d, exist_ok=True)
    opts = ("--qm-residues E:57:sc,E:102:sc,E:195:sc,I:15,I:16 --variants E:57=HID --padding 1.0 "
            "--ron 10 --roff 12 --distances E:195:OG-I:15:C,E:57:NE2-E:195:OG,E:57:HD1-E:102:OD1,"
            "E:57:HD1-E:102:OD2 --cutoff-test 8,12,16,20,24 --snapshots 8")
    t0 = time.perf_counter()
    out = sh(f"cd {d} && python -u {script} {pdb} --libmopac {lib} {opts} --platform CUDA "
             f"--steps 10000 --nve 1000 --mm-equil-ps 100", check=False)
    open(f"{d}/log.txt", "w").write(out)
    os.system(f"cp {d}/log.txt {OUT}/log_trypsin.txt; cp {d}/qmmm_traj.pdb {OUT}/ 2>/dev/null")
    say("")
    say(f"== trypsin-BPTI ({time.perf_counter() - t0:.0f} s in all)")
    for line in out.splitlines():
        if line.startswith(("system:", "QM region", "QM/MM NVT", "NVE check", "MM equilibration", "distance",
                            "cut-off", "   cut-off", "   PME")) or "Error" in line or "error" in line:
            say("   " + line.rstrip()[:300])
    say(f"total {(time.time() - t_start) / 60:.0f} min")


main()

"""Kaggle profile of a QM/MM step on the GPU (why whole crambin took 9 s per step).

Builds MOPAC (GPU, OpenBLAS), installs OpenMM 8.6.1 CUDA; runs (1) a standalone 1SCF GRADIENTS of crambin
on the GPU, (2) 6 QM/MM steps of whole crambin in water with MOPAC_API_OUTPUT (section timers and GPU status of
the API calls), (3) the same without MM charges (roff tiny), (4) whole chignolin (bigger box), and prints the
largest timers and the GPU SCF status lines.  REPORT_QMMM_PROF.txt.
"""

import os
import re
import subprocess
import time

BASE = "/tmp/qp"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_QMMM_PROF.txt"


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


def timers(path, n=14):
    sec = {}
    if not os.path.exists(path):
        return ["(no output)"]
    txt = open(path, errors="ignore").read()
    for m in re.finditer(r"name=(\S+) calls=(\d+) ms=([0-9.]+)", txt):
        sec[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    lines = [f"{k:36s} {c:6d} calls {ms / 1000:9.2f} s" for k, (c, ms) in sorted(sec.items(), key=lambda x: -x[1][1])[:n]]
    st = sorted(set(re.findall(r"\[MOZYME GPU[^\]]*\][^\n]*status=\S+[^\n]*", txt)))[:6]
    return lines + ["status: " + s[:200] for s in st] + \
        [f"fallback lines: {txt.count('fallback')}, SCF iterations lines: {len(re.findall('ITERATIONS', txt))}"]


def main():
    os.makedirs(BASE, exist_ok=True)
    open(REPORT, "w").close()
    say(sh("nvidia-smi --query-gpu=index,name --format=csv,noheader"))
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    sh("apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
       "--no-install-recommends gfortran libopenblas-dev liblapack-dev ninja-build >/dev/null 2>&1", check=False)
    sh("pip install -q 'openmm[cuda12]==8.6.1' 2>&1 | tail -2", check=False)
    sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/juvenalyosa/mopac_gpu.git {SRC}")
    say("commit: " + sh(f"git -C {SRC} log -1 --format='%h %s'").strip())
    cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1").strip().replace(".", "")
    sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=ON -DTESTS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_MKL=OFF "
       f"-DBLA_VENDOR=OpenBLAS -DCUDA_ARCHS={cc} > {BASE}/cmake.log 2>&1 && cmake --build {BUILD} --target mopac "
       f"--parallel $(nproc) > {BASE}/build.log 2>&1 || (grep -i error {BASE}/build.log | head -40; exit 1)")
    lib = sh(f"ls {BUILD}/libmopac.so* | head -1").strip()
    data = f"{SRC}/tests/data/mozyme_gpu_reference"
    env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MOPAC_MOZYME_SECTION_PROFILE="1", MOPAC_GPU_VERBOSE="1")
    # 1. standalone
    d = f"{BASE}/sa"
    os.makedirs(d, exist_ok=True)
    sh(f"cp {data}/1CRN_opt.pdb {d}/")
    for tag, kw in (("default", ""), ("tight", "RELSCF=0.01 THRESH=1.D-15 HB-SMOOTH")):
        open(f"{d}/{tag}.mop", "w").write(f'PM7 MOZYME 1SCF GRADIENTS GEO-OK {kw} GEO_DAT="1CRN_opt.pdb"\nx\n\n')
        t0 = time.perf_counter()
        sh(f"cd {d} && {BUILD}/mopac {tag}.mop", check=False, env=env0)
        say(f"\n1. standalone crambin 1SCF GRADIENTS ({tag}): {time.perf_counter() - t0:.1f} s")
        for line in timers(f"{d}/{tag}.out", 10):
            say("   " + line)
    # 2-4. QM/MM
    script = f"{SRC}/scripts/qmmm/run_qmmm_md.py"
    res = ",".join(str(i) for i in range(1, 47))
    runs = [("crambin QM/MM", f"{data}/1CRN_opt.pdb", f"--qm-residues {res} --padding 1.8 --ron 6 --roff 8"),
            ("crambin no MM charges", f"{data}/1CRN_opt.pdb", f"--qm-residues {res} --padding 1.8 --ron 0.05 --roff 0.1"),
            ("chignolin QM/MM", f"{data}/1UAO_opt.pdb", "--qm-residues 1,2,3,4,5,6,7,8,9,10 --padding 2.2 --ron 6 --roff 8")]
    for name, pdb, opts in runs:
        d = f"{BASE}/" + name.replace(" ", "_").replace("/", "")
        os.makedirs(d, exist_ok=True)
        env = dict(env0, MOPAC_API_OUTPUT=f"{d}/api.out")
        out = sh(f"cd {d} && python -u {script} {pdb} --libmopac {lib} {opts} --platform CUDA --steps 6 --nve 40 "
                 f"--mm-equil-ps 5", check=False, env=env)
        say(f"\n== {name}")
        for line in out.splitlines():
            if line.startswith(("system:", "QM region", "QM/MM NVT", "NVE check")) or "rror" in line:
                say("   " + line.strip()[:300])
        for line in timers(f"{d}/api.out"):
            say("   " + line)


main()

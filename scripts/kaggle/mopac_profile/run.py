"""Kaggle GPU profile of MOZYME geometry steps: where the time of a cycle goes (SCF, Fock, COSMO, gradient).

Builds MOPAC (GPU=ON) from https://github.com/juvenalyosa/mopac_gpu and runs short optimizations (CYCLES=6)
with MOPAC_MOZYME_SECTION_PROFILE=1 and MOPAC_GPU_PROFILE=1: the DNA dodecamer (1BNA neutral in the gas
phase, 1BNA anion with standard COSMO and with SWIG) and chignolin (gas, COSMO, SWIG), one job per GPU at a
time.  REPORT_PROFILE.txt: wall time per cycle and the largest [PROFILE] MOZYME_SECTION timers per case.

    kaggle kernels push -p scripts/kaggle/mopac_profile
    kaggle kernels output juvenalyosa/mopac-profile -p ./profile_results
"""

import os
import re
import shutil
import subprocess
import threading
import time

BRANCH = "main"
BASE = "/tmp/prof"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_PROFILE.txt"
KEYS = "PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK CYCLES=6 GNORM=0.01"
CASES = [
    ("dna_gas", "mozyme_dna_cosmo/1BNA_neutral.pdb", ""),
    ("dna_cosmo", "mozyme_dna_cosmo/1BNA_anion.pdb", "EPS=78.4"),
    ("dna_swig", "mozyme_dna_cosmo/1BNA_anion.pdb", "EPS=78.4 SWIG"),
    ("chig_gas", "mozyme_gpu_reference/1UAO_opt.pdb", ""),
    ("chig_cosmo", "mozyme_gpu_reference/1UAO_opt.pdb", "EPS=78.4"),
    ("chig_swig", "mozyme_gpu_reference/1UAO_opt.pdb", "EPS=78.4 SWIG"),
]


def say(msg):
    print(msg, flush=True)
    with open(REPORT, "a") as fh:
        fh.write(msg + "\n")


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        if check:
            raise SystemExit(f"failed: {cmd}")
    return r.stdout


def run_case(gpu, case, res):
    cid, geo, extra = case
    d = f"{BASE}/runs/{cid}"
    os.makedirs(d, exist_ok=True)
    shutil.copy2(f"{SRC}/tests/data/{geo}", d)
    with open(f"{d}/j.mop", "w") as fh:
        fh.write(f'{KEYS} {extra} GEO_DAT="{os.path.basename(geo)}"\nprofile\n\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MOPAC_MOZYME_SECTION_PROFILE="1", MOPAC_GPU_PROFILE="1")
    t0 = time.perf_counter()
    subprocess.run([f"{BUILD}/mopac", "j.mop"], cwd=d, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = open(f"{d}/j.out", errors="ignore").read() if os.path.exists(f"{d}/j.out") else ""
    sec = {}
    for m in re.finditer(r"name=(\S+) calls=(\d+) ms=([0-9.]+)", out):
        sec[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    cyc = [float(x) for x in re.findall(r"CYCLE:\s+\d+\s+TIME:\s+([0-9.]+)", out)]
    stage = re.findall(r"resident_stage_ms.*", out)
    res[cid] = dict(wall=wall, sec=sec, cyc=cyc, stage=stage[-3:], fallback=out.count("status=fallback"))


def main():
    os.makedirs(BASE, exist_ok=True)
    open(REPORT, "w").close()
    say(sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader"))
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    sh("apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
       "--no-install-recommends gfortran libblas-dev liblapack-dev ninja-build >/dev/null 2>&1", check=False)
    sh(f"rm -rf {SRC} && git clone -q --depth 1 --branch {BRANCH} https://github.com/juvenalyosa/mopac_gpu.git {SRC}")
    say("commit: " + sh(f"git -C {SRC} log -1 --format='%h %s'").strip())
    cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1").strip().replace(".", "")
    sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=ON -DTESTS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo "
       f"-DCUDA_ARCHS={cc} > {BASE}/cmake.log 2>&1 && cmake --build {BUILD} --target mopac --parallel $(nproc) "
       f"> {BASE}/build.log 2>&1 || (grep -i error {BASE}/build.log | head -40; exit 1)")
    ngpu = int(sh("nvidia-smi -L | wc -l").strip() or 1)
    # checks: 1SCF GRADIENTS against the CPU references (dense COSMO matrix) and the COSMO gradient
    # on the GPU against the CPU code (MOPAC_GPU_GRAD_CHECK=1)
    import json
    ref = json.load(open(f"{SRC}/tests/data/mozyme_dna_cosmo/reference.json"))
    ref["cases"].update(json.load(open(f"{SRC}/tests/data/mozyme_dna_cosmo/swig_reference.json"))["cases"])
    say("")
    say("checks: 1SCF GRADIENTS against CPU references; COSMO gradient GPU - CPU (MOPAC_GPU_GRAD_CHECK)")
    for cid in ("chignolin_cosmo_scf", "chignolin_swig_scf", "chignolin_swig_scf_tight", "dna_cosmo_scf",
                "dna_swig_scf_tight"):
        c = ref["cases"][cid]
        d = f"{BASE}/checks/{cid}"
        os.makedirs(d, exist_ok=True)
        geo = "mozyme_gpu_reference/1UAO_opt.pdb" if "1UAO" in c["geometry"] else f"mozyme_dna_cosmo/{c['geometry']}"
        shutil.copy2(f"{SRC}/tests/data/{geo}", d)
        open(f"{d}/j.mop", "w").write(f'{ref["keywords"]} {c["extra_keywords"]} GEO_DAT="{c["geometry"]}"\ncheck\n\n')
        t0 = time.perf_counter()
        subprocess.run([f"{BUILD}/mopac", "j.mop"], cwd=d, capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", MOPAC_GPU_GRAD_CHECK="1"))
        wall = time.perf_counter() - t0
        out = open(f"{d}/j.out", errors="ignore").read()
        h = re.findall(r"FINAL HEAT OF FORMATION =\s*([-0-9.]+)", out)
        g = re.findall(r"GRADIENT NORM\s*=\s*([-0-9.]+)", out)
        chk = re.findall(r"COSMO gradient\] check max_abs_dgrad=\s*(\S+)", out)
        dense = re.findall(r"cosmo_matvec_calls=\s*(\d+)", out)
        say(f"{cid:28s} {wall:7.1f} s  d heat {float(h[-1]) - c['heat'] if h else 'none'}  "
            f"d GNORM {float(g[-1]) - c['gnorm'] if g else 'none'}  GPU-CPU COSMO grad {chk[-1] if chk else 'none'}")
    res = {}
    for first in range(0, len(CASES), ngpu):
        th = [threading.Thread(target=run_case, args=(k, c, res)) for k, c in enumerate(CASES[first:first + ngpu])]
        for t in th:
            t.start()
        for t in th:
            t.join()
    for cid, _, extra in CASES:
        r = res[cid]
        c = r["cyc"][1:] or r["cyc"]
        say("")
        say(f"== {cid} ({extra or 'gas'}): wall {r['wall']:.1f} s, {len(r['cyc'])} cycles, "
            f"{sum(c) / max(1, len(c)):.2f} s/cycle (MOPAC TIME), fallbacks {r['fallback']}")
        tot = r["sec"].get("run_total", (1, 1.0))[1]
        for name, (calls, ms) in sorted(r["sec"].items(), key=lambda x: -x[1][1])[:18]:
            say(f"   {name:36s} {calls:6d} calls {ms / 1000:9.2f} s  {100 * ms / tot:5.1f} %")
        for line in r["stage"]:
            say("   " + line.strip()[:300])
    shutil.make_archive(f"{OUT}/profile_runs", "gztar", f"{BASE}/runs")


main()

"""Short Kaggle A/B of diagg2 variants (two T4: the two variants run at the same time, then swap GPUs).

Builds MOPAC (GPU=ON) from https://github.com/juvenalyosa/mopac_gpu (BRANCH) and runs, for each variant in
VARIANTS (environment settings), crambin and chignolin 1SCF at the FORCE criterion (SCFCRT=1e-6 THRESH=1e-15)
and a small crambin FORCETS, with MOPAC_MOZYME_SCF_TRACE=1.  REPORT_AB.txt (kernel output): wall times,
heats, iterations and the mean diagg2 sweep statistics: span (wall time of a sweep), wait / rotate (mean per
block: lock waits and rotations), max_chain, grid.

    kaggle kernels push -p scripts/kaggle/mopac_diagg2_ab
    kaggle kernels output juvenalyosa/mopac-diagg2-ab -p ./ab_results
"""

import os
import re
import shutil
import statistics
import subprocess
import threading
import time

BRANCH = "diagg2-lpt"
# base: the sweep before the fast path and the LPT order; fast: fast path only; fast+lpt: the new default
VARIANTS = {
    "base": {"MOPAC_MOZYME_DIAGG2_LPT": "0", "MOPAC_MOZYME_DIAGG2_FASTPATH": "0"},
    "fast": {"MOPAC_MOZYME_DIAGG2_LPT": "0"},
    "fast+lpt": {},
}
REPEATS = 2
BASE = "/tmp/ab"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_AB.txt"
KEYS = "PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK"
JOBS = [
    ("1CRN_scf", "1CRN_opt.pdb", f"{KEYS} 1SCF GRADIENTS LET SCFCRT=0.000001 THRESH=1.D-15"),
    ("1UAO_scf", "1UAO_opt.pdb", f"{KEYS} 1SCF GRADIENTS SCFCRT=0.000001 THRESH=1.D-15"),
    ("1CRN_forcets", "1CRN_opt.pdb", f'{KEYS} FORCETS THERMO(298) LET OPT("A10"=1)'),
]
SWEEP_RE = re.compile(r"diagg2 staged=\d nij=(\d+) rotated=(\d+) max_chain=(\d+) lock_spins=(\d+) nrej=-?\d+ "
                      r"lpt=(\d) grid=(\d+) span_us=([0-9.]+) wait_us=([0-9.]+) rotate_us=([0-9.]+)")


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


def run_job(variant, gpu, rep, name, geo, keys, results):
    d = f"{BASE}/runs/{variant}/{name}_{rep}"
    os.makedirs(d, exist_ok=True)
    shutil.copy2(f"{SRC}/tests/data/mozyme_gpu_reference/{geo}", d)
    with open(f"{d}/j.mop", "w") as fh:
        fh.write(f'{keys} GEO_DAT="{geo}"\nab\n\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MOPAC_MOZYME_SCF_TRACE="1", **VARIANTS[variant])
    t0 = time.perf_counter()
    p = subprocess.run([f"{BUILD}/mopac", "j.mop"], cwd=d, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = open(f"{d}/j.out", errors="ignore").read() if os.path.exists(f"{d}/j.out") else ""
    heat = re.findall(r"FINAL HEAT OF FORMATION =\s*([-0-9.]+)", out)
    zpe = re.findall(r"ZERO POINT ENERGY\s+([-0-9.]+)", out)
    its = [int(x) for x in re.findall(r"status=success reason=\S+ code=\S+ iterations=\s*(\d+)", out)]
    sweeps = [tuple(float(v) for v in m) for m in SWEEP_RE.findall(p.stdout)]
    fallback = len(re.findall(r"status=fallback", out))
    results.append(dict(variant=variant, gpu=gpu, rep=rep, name=name, wall=wall,
                        heat=heat[-1] if heat else None, zpe=zpe[-1] if zpe else None,
                        iterations=its, sweeps=sweeps, fallback=fallback,
                        errors=p.stdout.count("[GPU ERROR]")))


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
    t0 = time.time()
    sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=ON -DTESTS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo "
       f"-DCUDA_ARCHS={cc} > {BASE}/cmake.log 2>&1 && cmake --build {BUILD} --target mopac --parallel $(nproc) "
       f"> {BASE}/build.log 2>&1 || (grep -i error {BASE}/build.log | head -40; exit 1)")
    say(f"build {time.time() - t0:.0f} s")
    ngpu = int(sh("nvidia-smi -L | wc -l").strip() or 1)
    names = list(VARIANTS)
    results = []
    for rep in range(1, REPEATS + 1):
        for name, geo, keys in JOBS:
            if name.endswith("forcets") and rep > 1:
                continue
            # ngpu variants at a time, one per GPU, the GPUs rotated on the next repeat
            for first in range(0, len(names), ngpu):
                threads = []
                for k, variant in enumerate(names[first:first + ngpu]):
                    gpu = (k + rep - 1) % ngpu
                    th = threading.Thread(target=run_job, args=(variant, gpu, rep, name, geo, keys, results))
                    th.start()
                    threads.append(th)
                for th in threads:
                    th.join()
            say(f"done {name} repeat {rep}")
    say("")
    say(f"{'job':14s} {'variant':9s} {'gpu':>3s} {'wall s':>8s} {'heat':>12s} {'zpe':>9s} {'iters':>12s} "
        f"{'sweeps':>6s} {'span ms':>8s} {'wait ms':>8s} {'rot ms':>8s} {'chain':>6s} {'grid':>5s} {'fb':>3s}")
    for r in sorted(results, key=lambda x: (x["name"], x["rep"], x["variant"])):
        big = [s for s in r["sweeps"] if s[0] > 50000] or r["sweeps"]
        mean = lambda k: statistics.mean(s[k] for s in big) if big else float("nan")  # noqa: E731
        it = r["iterations"][-1] if len(r["iterations"]) == 1 else (len(r["iterations"]) or "-")
        say(f"{r['name']:14s} {r['variant']:9s} {r['gpu']:3d} {r['wall']:8.1f} {str(r['heat']):>12s} "
            f"{str(r['zpe']):>9s} {str(it):>12s} {len(r['sweeps']):6d} {mean(6) / 1e3:8.2f} {mean(7) / 1e3:8.2f} "
            f"{mean(8) / 1e3:8.2f} {mean(2):6.0f} {mean(5):5.0f} {r['fallback'] + r['errors']:3d}")
    say("")
    say("totals per variant (sum of wall time over all jobs; sum of diagg2 sweep spans):")
    for v in names:
        rs = [r for r in results if r["variant"] == v]
        span = sum(s[6] for r in rs for s in r["sweeps"]) / 1e6
        say(f"  {v:9s} wall {sum(r['wall'] for r in rs):8.1f} s   diagg2 spans {span:7.2f} s")
    shutil.make_archive(f"{OUT}/ab_runs", "gztar", f"{BASE}/runs")


main()

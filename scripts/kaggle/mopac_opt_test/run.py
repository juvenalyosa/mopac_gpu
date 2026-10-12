"""Kaggle GPU test of the MOZYME geometry optimizer (trust-region L-BFGS) and of FORCE-ready minima.

Builds MOPAC (GPU=ON) from https://github.com/juvenalyosa/mopac_gpu and runs on the GPUs (two T4: two jobs at a time):
  1. reliability: chignolin (tests/data/mozyme_gpu_reference/1UAO_start.pdb) from 8 starting points 0.002 Angstrom
     apart (random.seed 1..8, as in the CPU study), GNORM=1, with the trust-region optimizer and with L-BFGS-B
     (MOPAC_LBFGS_CLASSIC=1);
  2. FORCE-ready minimum, 2 of those starts: mozyme_md_workflow.optimize_stationary (GNORM=0.1, optimizations
     restarted fresh until a fresh tightly converged SCF agrees; the interaction lists only grow during one
     optimization) and FORCE THERMO(298) on the result: imaginary modes.
REPORT_OPT.txt (kernel output) has every number.

    kaggle kernels push -p scripts/kaggle/mopac_opt_test
    kaggle kernels output juvenalyosa/mopac-opt-test -p ./opt_results
"""

import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time

BRANCH = "main"
SEEDS = list(range(1, 9))
FORCE_SEEDS = [1, 2]
BASE = "/tmp/opt"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_OPT.txt"
KEYS = "PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK"


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


def mopac(run_dir, keywords, geo, gpu, env_extra=None, name="j"):
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/{name}.mop", "w") as fh:
        fh.write(f'{keywords} GEO_DAT="{geo}"\nopt test\n\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), **(env_extra or {}))
    t0 = time.perf_counter()
    subprocess.run([f"{BUILD}/mopac", f"{name}.mop"], cwd=run_dir, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = open(f"{run_dir}/{name}.out", errors="ignore").read() if os.path.exists(f"{run_dir}/{name}.out") else ""
    heat = re.findall(r"FINAL HEAT OF FORMATION =\s*([-0-9.]+)", out)
    gn = re.findall(r"GRADIENT NORM\s*=\s*([-0-9.]+)", out)
    cycles = re.findall(r"CYCLE:\s+(\d+)", out)
    fallbacks = len(re.findall(r"status=fallback", out))
    return dict(wall=wall, heat=float(heat[-1]) if heat else None, gnorm=float(gn[-1]) if gn else None,
                cycles=int(cycles[-1]) if cycles else 0, fallbacks=fallbacks, text=out)


def perturbed(seed, src, dst):
    random.seed(seed)
    out = []
    for line in open(src).read().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            x += random.uniform(-.002, .002)
            y += random.uniform(-.002, .002)
            z += random.uniform(-.002, .002)
            line = line[:30] + "%8.3f%8.3f%8.3f" % (x, y, z) + line[54:]
        out.append(line)
    open(dst, "w").write("\n".join(out) + "\n")


def in_parallel(tasks, ngpu):
    """tasks: callables taking a GPU id; ngpu at a time, one per GPU."""
    for first in range(0, len(tasks), ngpu):
        threads = [threading.Thread(target=t, args=(k,)) for k, t in enumerate(tasks[first:first + ngpu])]
        for th in threads:
            th.start()
        for th in threads:
            th.join()


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
    start = f"{SRC}/tests/data/mozyme_gpu_reference/1UAO_start.pdb"

    # 1. reliability
    res = {}
    tasks = []
    for opt in ("trust", "classic"):
        for seed in SEEDS:
            d = f"{BASE}/runs/{opt}_r{seed}"
            os.makedirs(d, exist_ok=True)
            perturbed(seed, start, f"{d}/start.pdb")
            env = {"MOPAC_LBFGS_CLASSIC": "1"} if opt == "classic" else {}

            def job(gpu, d=d, env=env, key=(opt, seed)):
                res[key] = mopac(d, f"{KEYS} CYCLES=3000 PDBOUT GNORM=1", "start.pdb", gpu, env)
            tasks.append(job)
    in_parallel(tasks, ngpu)
    say("")
    say("1. chignolin GNORM=1 from 8 starts 0.002 A apart (GPU)")
    say(f"{'opt':8s} {'seed':>4s} {'cycles':>6s} {'wall s':>7s} {'heat':>11s} {'GNORM':>9s} {'fb':>3s}")
    for opt in ("trust", "classic"):
        ok = 0
        for seed in SEEDS:
            r = res[(opt, seed)]
            conv = r["gnorm"] is not None and r["gnorm"] < 1.0
            ok += conv
            say(f"{opt:8s} {seed:4d} {r['cycles']:6d} {r['wall']:7.1f} {str(r['heat']):>11s} {str(r['gnorm']):>9s} "
                f"{r['fallbacks']:3d}")
        heats = [res[(opt, s)]["heat"] for s in SEEDS if res[(opt, s)]["heat"] is not None]
        say(f"   {opt}: {ok} of {len(SEEDS)} reach GNORM < 1; heats {min(heats):.2f} .. {max(heats):.2f}; "
            f"mean wall {sum(res[(opt, s)]['wall'] for s in SEEDS) / len(SEEDS):.1f} s")

    # 2. FORCE-ready minima: optimize_stationary from the GNORM=1 result, then FORCE
    say("")
    say("2. stationary point (mozyme_md_workflow.optimize_stationary, GNORM=0.1: optimizations restarted fresh")
    say("   until a fresh, tightly converged SCF has GNORM < 0.1), then FORCE THERMO(298)")
    sys.path.insert(0, f"{SRC}/scripts")
    sys.path.insert(0, f"{SRC}/tests")
    from pathlib import Path
    import mozyme_md_workflow as md  # noqa: E402
    import mozyme_parallel_force as pf  # noqa: E402
    import check_mozyme_gpu_thermo as th  # noqa: E402
    final = {}

    def stationary(gpu, seed):
        d1 = Path(f"{BASE}/runs/trust_r{seed}")
        geo, stages = md.optimize_stationary(Path(f"{BUILD}/mopac"), d1 / "j.arc", Path(f"{BASE}/runs/stat_r{seed}"),
                                             gnorm=0.1, env_extra={"CUDA_VISIBLE_DEVICES": str(gpu)})
        final[seed] = (geo, stages)
    in_parallel([lambda gpu, s=s: stationary(gpu, s) for s in FORCE_SEEDS], ngpu)
    for seed in FORCE_SEEDS:
        geo, stages = final[seed]
        for st in stages:
            say(f"seed {seed} stage {st['stage']}: {st['cycles']} cycles {st['wall']:.1f} s, heat {st['heat']}, "
                f"GNORM {st['gnorm_opt']}, fresh tight SCF {st['gnorm_fresh']}")
    for seed in FORCE_SEEDS:
        geo, _ = final[seed]
        t0 = time.perf_counter()
        out, tim = pf.parallel_force(Path(f"{BUILD}/mopac"), geo, f"{KEYS} THERMO(298)", 2 * ngpu,
                                     Path(f"{BASE}/runs/stat_r{seed}/force"))
        t = th.thermo(open(out, errors="ignore").read(), 298.0)
        f = sorted(t["freqs"])
        say(f"seed {seed}: FORCE {time.perf_counter() - t0:.0f} s, {len(f)} frequencies, "
            f"{sum(1 for x in f if x < 0)} imaginary, lowest {f[:4]}, ZPE {t.get('zpe')}, S {t.get('S')}")
    shutil.make_archive(f"{OUT}/opt_runs", "gztar", f"{BASE}/runs")


main()

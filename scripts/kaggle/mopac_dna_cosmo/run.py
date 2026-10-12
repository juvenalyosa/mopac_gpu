"""Kaggle GPU test of MOZYME on DNA and with COSMO (EPS=78.4) against CPU references.

Builds MOPAC (GPU=ON) from https://github.com/juvenalyosa/mopac_gpu and runs on the GPUs (two T4):
  1. 1SCF GRADIENTS, default and FORCE criterion (SCFCRT=1e-6 THRESH=1e-15), against
     tests/data/mozyme_dna_cosmo/reference.json (CPU): DNA dodecamer 1BNA neutral (780 atoms, 22 P: d orbitals),
     1BNA anion (758 atoms, charge -22) in COSMO water, chignolin in COSMO water;
  2. at the same time, GPU 0: chignolin in water, mozyme_md_workflow.optimize_stationary (GNORM=0.1) and
     FORCE THERMO(298) (imaginary modes; geometry and output kept for a CPU FORCE on the same geometry);
     GPU 1: the DNA anion in water, optimize_stationary (GNORM=0.2, at most 3 stages of T=2H);
  3. FORCE THERMO(298) of the DNA anion in water on both GPUs, if less than 6 h have passed.
GPU memory peak per phase, GPU status lines (fallbacks) per run.  REPORT_DNA.txt (kernel output) has every number.

    kaggle kernels push -p scripts/kaggle/mopac_dna_cosmo
    kaggle kernels output juvenalyosa/mopac-dna-cosmo -p ./dna_results
"""

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

BRANCH = "main"
BASE = "/tmp/dna"
OUT = "/kaggle/working"
SRC = f"{BASE}/mopac_gpu"
BUILD = f"{BASE}/build"
REPORT = f"{OUT}/REPORT_DNA.txt"
T0 = time.time()
mem_peak = collections.defaultdict(int)   # phase -> MiB (max over GPUs)
phase = ["start"]


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


def sample_memory():
    while True:
        try:
            out = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                                 capture_output=True, text=True).stdout
            used = max(int(x) for x in out.split())
            mem_peak[phase[0]] = max(mem_peak[phase[0]], used)
        except Exception:
            pass
        time.sleep(2)


def gpu_status(text):
    """Counts of the status= values in the [MOZYME GPU ...] lines."""
    c = collections.Counter(re.findall(r"\[MOZYME GPU[^\]]*\].*?status=(\S+)", text))
    return dict(c)


def mopac(run_dir, keywords, geo, gpu, name="j"):
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/{name}.mop", "w") as fh:
        fh.write(f'{keywords} GEO_DAT="{geo}"\ndna/cosmo test\n\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    t0 = time.perf_counter()
    subprocess.run([f"{BUILD}/mopac", f"{name}.mop"], cwd=run_dir, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = open(f"{run_dir}/{name}.out", errors="ignore").read() if os.path.exists(f"{run_dir}/{name}.out") else ""
    heat = re.findall(r"FINAL HEAT OF FORMATION =\s*([-0-9.]+)", out)
    gn = re.findall(r"GRADIENT NORM\s*=\s*([-0-9.]+)", out)
    return dict(wall=wall, heat=float(heat[-1]) if heat else None, gnorm=float(gn[-1]) if gn else None,
                failed="SCF CALCULATION FAILED" in out or "FAILED TO ACHIEVE SCF" in out,
                status=gpu_status(out))


def in_parallel(tasks, ngpu):
    """tasks: callables taking a GPU id; ngpu at a time, one per GPU."""
    for first in range(0, len(tasks), ngpu):
        threads = [threading.Thread(target=t, args=(k,)) for k, t in enumerate(tasks[first:first + ngpu])]
        for th in threads:
            th.start()
        for th in threads:
            th.join()


def force(pf, th, geo, keys, workers, work, gpus, label):
    t0 = time.perf_counter()
    try:
        out, _ = pf.parallel_force(Path(f"{BUILD}/mopac"), geo, keys, workers, work, gpus=gpus)
        text = open(out, errors="ignore").read()
        t = th.thermo(text, 298.0)
        f = sorted(t["freqs"])
        say(f"{label}: FORCE {time.perf_counter() - t0:.0f} s, {len(f)} frequencies, "
            f"{sum(1 for x in f if x < 0)} imaginary, lowest {f[:6]}, ZPE {t.get('zpe')}, S {t.get('S')}, "
            f"H {t.get('H')}, G {t.get('G')}; GPU status {gpu_status(text)}")
    except Exception as exc:  # report and go on
        say(f"{label}: FORCE failed after {time.perf_counter() - t0:.0f} s: {exc!r}")


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
    threading.Thread(target=sample_memory, daemon=True).start()
    data = Path(SRC) / "tests/data/mozyme_dna_cosmo"
    ref = json.load(open(data / "reference.json"))
    geos = [data / "1BNA_neutral.pdb", data / "1BNA_anion.pdb",
            Path(SRC) / "tests/data/mozyme_gpu_reference/1UAO_opt.pdb"]

    # 1. 1SCF against the CPU
    phase[0] = "1scf"
    res = {}
    tasks = []
    for cid, case in ref["cases"].items():
        d = f"{BASE}/runs/{cid}"
        os.makedirs(d, exist_ok=True)
        for g in geos:
            shutil.copy2(g, d)

        def job(gpu, d=d, cid=cid, case=case):
            res[cid] = mopac(d, f"{ref['keywords']} {case['extra_keywords']}", case["geometry"], gpu)
        tasks.append(job)
    in_parallel(tasks, ngpu)
    say("")
    say(f"1. 1SCF GRADIENTS against the CPU ({ref['keywords']}); memory peak {mem_peak['1scf']} MiB")
    say(f"{'case':28s} {'GPU s':>6s} {'CPU s':>6s} {'heat GPU':>12s} {'d heat':>9s} {'d GNORM':>9s}  ok  status")
    for cid, case in ref["cases"].items():
        r = res[cid]
        tight = "SCFCRT" in case["extra_keywords"]
        tol = 1e-3 if tight else 0.1
        if r["heat"] is None:
            say(f"{cid:28s} {r['wall']:6.1f} {case['cpu_job_s']:6.1f} {'none':>12s} {'':9s} {'':9s}  NO  "
                f"{r['status']} failed={r['failed']}")
            continue
        dh = r["heat"] - case["heat"]
        dg = (r["gnorm"] or 0) - case["gnorm"]
        ok = abs(dh) <= tol and abs(dg) <= max(0.05, 1e-4 * case["gnorm"]) * (1 if tight else 20)
        say(f"{cid:28s} {r['wall']:6.1f} {case['cpu_job_s']:6.1f} {r['heat']:12.5f} {dh:9.5f} {dg:9.4f}  "
            f"{'yes' if ok else 'NO ':3s} {r['status']}")

    # 2. stationary points: chignolin in water (GPU 0, then FORCE on GPU 0) and the DNA anion in water (GPU 1)
    sys.path.insert(0, f"{SRC}/scripts")
    sys.path.insert(0, f"{SRC}/tests")
    import mozyme_md_workflow as md  # noqa: E402
    import mozyme_parallel_force as pf  # noqa: E402
    import check_mozyme_gpu_thermo as th  # noqa: E402
    phase[0] = "opt"
    keys_w = f"{md.BASE_KEYS} EPS=78.4 THERMO(298)"
    final = {}
    say("")
    say("2. stationary points in COSMO water (optimize_stationary), then FORCE THERMO(298)")

    def chig(gpu):
        try:
            geo, stages = md.optimize_stationary(Path(f"{BUILD}/mopac"), geos[2], Path(f"{BASE}/runs/chig_w"),
                                                 gnorm=0.1, eps=78.4,
                                                 env_extra={"CUDA_VISIBLE_DEVICES": str(gpu)})
            final["chig"] = (geo, stages)
            force(pf, th, geo, keys_w, 2, Path(f"{BASE}/runs/chig_w/force"), str(gpu), "chignolin, water")
        except Exception as exc:
            say(f"chignolin, water: failed: {exc!r}")

    def dna(gpu):
        try:
            geo, stages = md.optimize_stationary(Path(f"{BUILD}/mopac"), geos[1], Path(f"{BASE}/runs/dna_w"),
                                                 gnorm=0.2, eps=78.4, max_stages=3, extra="T=2H",
                                                 env_extra={"CUDA_VISIBLE_DEVICES": str(gpu)})
            final["dna"] = (geo, stages)
        except Exception as exc:
            say(f"DNA anion, water: optimization failed: {exc!r}")
    in_parallel([chig, dna] if ngpu > 1 else [chig], ngpu)
    if ngpu == 1:
        in_parallel([dna], 1)
    for key in ("chig", "dna"):
        if key in final:
            for st in final[key][1]:
                say(f"{key} stage {st['stage']}: {st['cycles']} cycles {st['wall']:.1f} s, heat {st['heat']}, "
                    f"GNORM {st['gnorm_opt']}, fresh tight SCF {st['gnorm_fresh']}")
    say(f"memory peak (optimizations, chignolin FORCE) {mem_peak['opt']} MiB")

    # 3. FORCE of the DNA anion in water
    phase[0] = "dna_force"
    if "dna" not in final:
        say("3. DNA FORCE skipped: no optimized geometry")
    elif time.time() - T0 > 6 * 3600:
        say(f"3. DNA FORCE skipped: {(time.time() - T0) / 3600:.1f} h used")
    else:
        say("")
        say("3. DNA anion in water, FORCE THERMO(298)")
        force(pf, th, final["dna"][0], keys_w, 2 * ngpu, Path(f"{BASE}/runs/dna_w/force"), "auto", "DNA anion, water")
        say(f"memory peak (DNA FORCE) {mem_peak['dna_force']} MiB")
    say(f"total {(time.time() - T0) / 60:.0f} min")
    for p in Path(f"{BASE}/runs").rglob("*"):
        if p.suffix in (".den", ".res", ".brz") and p.is_file():
            p.unlink()
    shutil.make_archive(f"{OUT}/dna_runs", "gztar", f"{BASE}/runs")


main()

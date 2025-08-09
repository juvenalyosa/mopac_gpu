#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys

HEAT_RE = re.compile(r"FINAL HEAT OF FORMATION\s*=\s*([\-+0-9\.Ee]+)")
HEAT_AUX_RE = re.compile(r"HEAT_OF_FORMATION:KCAL/MOL=\s*([\-+0-9\.Ee]+)")

def run(cmd, env=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    out, _ = p.communicate()
    return p.returncode, out

def parse_heat(output):
    for line in output.splitlines():
        m = HEAT_RE.search(line)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None

def parse_heat_from_files(stem):
    # Try .out, .arc, then .aux in current working directory
    for ext, regex in ((".out", HEAT_RE), (".arc", HEAT_RE), (".aux", HEAT_AUX_RE)):
        path = f"{stem}{ext}"
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m = regex.search(line)
                    if m:
                        try:
                            return float(m.group(1))
                        except Exception:
                            pass
        except Exception:
            continue
    return None

def main():
    ap = argparse.ArgumentParser(description="Compare CPU vs GPU MOPAC heats of formation")
    ap.add_argument("input", help="Path to .mop input file")
    ap.add_argument("--mopac", default="./mopac", help="Path to mopac executable (default: ./mopac)")
    ap.add_argument("--tolerance", type=float, default=1e-6, help="Absolute tolerance on heat of formation (kcal/mol)")
    ap.add_argument("--keep", action="store_true", help="Keep output files (default: delete)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        return 2

    env_base = os.environ.copy()

    # CPU run (disable GPU)
    env_cpu = env_base.copy()
    env_cpu["MOPAC_NOGPU"] = "1"
    stem = os.path.splitext(os.path.basename(args.input))[0]

    rc_cpu, out_cpu = run([args.mopac, args.input], env=env_cpu)
    if rc_cpu != 0:
        print("CPU run failed:\n" + out_cpu)
        return 2
    heat_cpu = parse_heat(out_cpu)
    if heat_cpu is None:
        heat_cpu = parse_heat_from_files(stem)
    if heat_cpu is None:
        print("CPU run: could not parse FINAL HEAT OF FORMATION (stdout/.out/.arc/.aux)\n" + out_cpu)
        return 2

    # GPU run (force GPU)
    env_gpu = env_base.copy()
    env_gpu["MOPAC_FORCEGPU"] = "1"
    rc_gpu, out_gpu = run([args.mopac, args.input], env=env_gpu)
    if rc_gpu != 0:
        print("GPU run failed:\n" + out_gpu)
        return 2
    heat_gpu = parse_heat(out_gpu)
    if heat_gpu is None:
        heat_gpu = parse_heat_from_files(stem)
    if heat_gpu is None:
        print("GPU run: could not parse FINAL HEAT OF FORMATION (stdout/.out/.arc/.aux)\n" + out_gpu)
        return 2

    diff = abs(heat_cpu - heat_gpu)
    ok = diff <= args.tolerance

    print(f"CPU heat: {heat_cpu:.10f} kcal/mol")
    print(f"GPU heat: {heat_gpu:.10f} kcal/mol")
    print(f"Abs diff: {diff:.10e} kcal/mol (tol {args.tolerance})")
    print("RESULT: PASS" if ok else "RESULT: FAIL")

    # Cleanup typical MOPAC side files unless --keep
    if not args.keep:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        for ext in (".arc", ".aux", ".out", ".den", ".res"):
            f = stem + ext
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import os
import subprocess
import sys

def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_gpu_scf_fallback.py <mopac_exe>")
        return 2
    mopac_exe = sys.argv[1]
    input_path = os.path.join(os.path.dirname(__file__), "gpu_scf_disk_fallback.mop")
    env = os.environ.copy()
    env.setdefault("MOPAC_GPU_SCF_EXPERIMENTAL", "on")
    env.setdefault("MOPAC_GPU_SCFTASK", "gpu")
    env.setdefault("MOPAC_FORCEGPU", "1")
    proc = subprocess.run(
        [mopac_exe, input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        print("fallback test: mopac returned non-zero status", file=sys.stderr)
        return proc.returncode
    out_path = os.path.splitext(input_path)[0] + ".out"
    try:
        with open(out_path, "r", encoding="utf-8", errors="ignore") as fh:
            out_text = fh.read()
    except OSError as exc:
        print(f"fallback test: unable to open output file: {exc}", file=sys.stderr)
        return 1
    if "[GPU SCF] Disabled: integral disk mode not supported" not in out_text:
        print("fallback test: expected GPU SCF fallback message not found", file=sys.stderr)
        return 1
    if "SCF FIELD WAS ACHIEVED" not in out_text:
        print("fallback test: SCF did not converge", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

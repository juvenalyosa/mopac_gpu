#!/usr/bin/env python3
"""Per-section MOZYME timing table: CPU vs GPU (current default) vs GPU resident SCF.

Usage:
  python3 scripts/mozyme_section_profile.py ./build-gpu/mopac \
      benchmarks/publication_inputs/mop/protein_crambin_1crn.mop [more.mop ...]

Each input is run once per mode in an isolated directory with
MOPAC_MOZYME_SECTION_PROFILE=1, and the [PROFILE] MOZYME_SECTION markers are
tabulated so the dominant sections can be ranked per mode.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from molecule_benchmark_report import parse_mozyme_section_times, stage_input  # noqa: E402

MODES: dict[str, dict[str, str | None]] = {
    "cpu": {"MOPAC_NOGPU": "1", "MOZYME_GPU_OFF": "1", "MOPAC_FORCEGPU": None, "MOZYME_GPU_FORCE": None},
    "gpu": {"MOPAC_FORCEGPU": "1", "MOZYME_GPU_FORCE": "1", "MOPAC_NOGPU": None, "MOZYME_GPU_OFF": None},
    # CPU-owned SCF loop with only the DIAGG stages offloaded to the parallel
    # kernels: the cleanest A/B correctness test for those kernels.
    "gpu-diagg": {
        "MOPAC_FORCEGPU": "1",
        "MOZYME_GPU_FORCE": "1",
        "MOPAC_NOGPU": None,
        "MOZYME_GPU_OFF": None,
        "MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU": "1",
        "MOPAC_MOZYME_DIAGG2_ROTATE_GPU": "1",
        "MOPAC_GPU_VERBOSE": "1",
    },
    "resident": {
        "MOPAC_FORCEGPU": "1",
        "MOZYME_GPU_FORCE": "1",
        "MOPAC_NOGPU": None,
        "MOZYME_GPU_OFF": None,
        "MOPAC_MOZYME_SCF_EXPERIMENTAL": "1",
        "MOPAC_MOZYME_RESIDENT_SCF": "1",
        "MOPAC_MOZYME_SCF_GPU": "1",
        "MOPAC_MOZYME_RESIDENT_FOCK_GPU": "1",
        "MOPAC_MOZYME_MAKVEC_GPU": "1",
        "MOPAC_MOZYME_SCF_STRICT_RESIDENT": "1",
        "MOPAC_MOZYME_SCF_EARLY_PROBE": "0",
    },
}

HEAT_RE = re.compile(r"FINAL HEAT OF FORMATION\s*=\s*([+\-0-9.EeDd]+)")
SCF_STATUS_RE = re.compile(r"\[MOZYME GPU SCF\]\s+status=(\S+)(?:.*?reason=(\S+))?")
HELPER_RE = re.compile(r"\[MOZYME GPU (\w+)\]\s+(success|fallback_cpu)")
RESIDENT_STAGE_RE = re.compile(r"\[PROFILE\]\s+MOZYME_RESIDENT_STAGE\s+name=(\S+)\s+calls=(\d+)\s+ms=([+\-0-9.Ee]+)")


def run_mode(mopac: Path, input_path: Path, mode: str, out_dir: Path, timeout: float) -> dict:
    run_dir = out_dir / input_path.stem / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_input(input_path, run_dir)

    env = os.environ.copy()
    env["MOPAC_MOZYME_SECTION_PROFILE"] = "1"
    env["MOPAC_GPU_DEBUG"] = "1"
    env["MOPAC_DETERMINISTIC"] = "1"
    for key, value in MODES[mode].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(mopac), staged.name],
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
        stdout, rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, rc = (exc.stdout or "") + "\nTIMEOUT\n", 124
    wall = time.perf_counter() - t0

    out_file = run_dir / (staged.stem + ".out")
    text = stdout + ("\n" + out_file.read_text(errors="ignore") if out_file.exists() else "")
    (run_dir / "combined.log").write_text(text)
    if rc != 0:
        print(f"    rc={rc}; last stdout/stderr lines:")
        for line in stdout.splitlines()[-15:]:
            print("      " + line[:200])
    for line in stdout.splitlines():
        if "[GPU ERROR]" in line or "runtime error" in line:
            print("      " + line[:200])

    heat = HEAT_RE.search(text)
    status = SCF_STATUS_RE.findall(text)
    resident_stages: dict[str, tuple[int, float]] = {}
    for m in RESIDENT_STAGE_RE.finditer(text):
        calls, ms = resident_stages.get(m.group(1), (0, 0.0))
        resident_stages[m.group(1)] = (calls + int(m.group(2)), ms + float(m.group(3)))
    helpers: dict[str, dict[str, int]] = {}
    for m in HELPER_RE.finditer(text):
        entry = helpers.setdefault(m.group(1), {"success": 0, "fallback_cpu": 0})
        entry[m.group(2)] += 1
    return {
        "mode": mode,
        "wall": wall,
        "rc": rc,
        "heat": float(heat.group(1)) if heat else None,
        "scf_status": status[-1] if status else None,
        "helpers": helpers,
        "sections": {r["name"]: r for r in parse_mozyme_section_times(text)},
        "resident_stages": resident_stages,
    }


def print_table(input_path: Path, results: list[dict]) -> None:
    modes = [r["mode"] for r in results]
    names: set[str] = set()
    for r in results:
        names.update(r["sections"])
    ref = results[0]["sections"]
    ordered = sorted(names, key=lambda n: -(ref.get(n, {}).get("ms", 0.0)))

    print(f"\n=== {input_path.name} ===")
    for r in results:
        heat = f"{r['heat']:.4f}" if r["heat"] is not None else "n/a"
        status = ""
        if r["scf_status"]:
            code, reason = r["scf_status"]
            status = f" scf_status={code}" + (f" reason={reason}" if reason else "")
        print(f"  {r['mode']:>9}: wall={r['wall']:9.2f}s rc={r['rc']} heat={heat}{status}")
        for name, counts in sorted(r["helpers"].items()):
            print(f"             [{name}] success={counts['success']} fallback_cpu={counts['fallback_cpu']}")
    if results[0]["heat"] is not None:
        for r in results[1:]:
            if r["heat"] is not None:
                print(f"  dHf({r['mode']}-{results[0]['mode']}) = {r['heat'] - results[0]['heat']:+.4f} kcal/mol")

    header = f"{'section':<32}" + "".join(f"{m + ' ms':>14}{'calls':>8}" for m in modes)
    print("\n" + header)
    print("-" * len(header))
    for name in ordered:
        line = f"{name:<32}"
        for r in results:
            row = r["sections"].get(name)
            line += f"{row['ms']:>14.1f}{row['calls']:>8}" if row else f"{'-':>14}{'-':>8}"
        print(line)

    for r in results:
        if r["resident_stages"]:
            print(f"\n  resident GPU stages ({r['mode']}):")
            for name, (calls, ms) in sorted(r["resident_stages"].items(), key=lambda kv: -kv[1][1]):
                print(f"    {name:<28}{ms:>12.1f} ms{calls:>8} calls")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mopac")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--modes",
        default="cpu,gpu-diagg,resident",
        help="Comma-separated subset of: cpu,gpu,gpu-diagg,resident",
    )
    parser.add_argument("--out-dir", default="mozyme_section_profile")
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()

    mopac = Path(args.mopac).resolve()
    if not mopac.exists():
        raise SystemExit(f"MOPAC executable not found: {mopac}")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"Unknown modes: {unknown}")
    out_dir = Path(args.out_dir).resolve()

    for raw in args.inputs:
        input_path = Path(raw).resolve()
        results = []
        for mode in modes:
            print(f"[{mode}] {input_path.name} ...", flush=True)
            results.append(run_mode(mopac, input_path, mode, out_dir, args.timeout))
            print(f"    {results[-1]['wall']:.2f}s", flush=True)
        print_table(input_path, results)


if __name__ == "__main__":
    main()

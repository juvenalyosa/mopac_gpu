#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HEAT_RE = re.compile(r"HEAT\s+OF\s+FORMATION\s*=*\s*([\-+0-9Ee\.]+)")

def parse_heat(path: Path):
    try:
        txt = path.read_text(errors='ignore')
    except Exception as e:
        return None
    m = HEAT_RE.search(txt)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def run(cmd, env=None, cwd=None):
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    return p.returncode, out

def main():
    ap = argparse.ArgumentParser(description='Run MOPAC examples with CPU vs GPU and compare heats of formation')
    ap.add_argument('mopac', help='Path to mopac executable (e.g., build-gpu/mopac)')
    ap.add_argument('inputs', nargs='*', help='Input .mop files (default: examples/*.mop)')
    ap.add_argument('--workdir', default='run_compare', help='Working directory to store outputs')
    args = ap.parse_args()

    mopac = Path(args.mopac).resolve()
    if not mopac.exists():
        print(f'Executable not found: {mopac}', file=sys.stderr)
        sys.exit(2)

    inputs = [Path(p) for p in (args.inputs or sorted(Path('examples').glob('*.mop')))]
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    print(f'Using workdir: {work}')
    print(f'Found {len(inputs)} inputs')
    rows = []
    for inp in inputs:
        name = inp.stem
        out_cpu = work / f'{name}.cpu.out'
        out_gpu = work / f'{name}.gpu.out'

        # CPU run
        env = os.environ.copy()
        env['MOPAC_NOGPU'] = '1'
        rc, out = run([str(mopac), str(inp.resolve())], env=env)
        out_cpu.write_text(out)
        e_cpu = parse_heat(out_cpu)

        # GPU run (force)
        env = os.environ.copy()
        env['MOPAC_FORCEGPU'] = '1'
        rc2, out2 = run([str(mopac), str(inp.resolve())], env=env)
        out_gpu.write_text(out2)
        e_gpu = parse_heat(out_gpu)

        if e_cpu is None or e_gpu is None:
            rows.append((name, e_cpu, e_gpu, None, None))
        else:
            diff = abs(e_cpu - e_gpu)
            rel = diff / (abs(e_cpu) + 1e-16)
            rows.append((name, e_cpu, e_gpu, diff, rel))

    # Print summary
    print('\nSummary (Heat of Formation, kcal/mol)')
    print(f"{'Input':<24} {'CPU':>16} {'GPU':>16} {'AbsDiff':>12} {'RelDiff':>12}")
    for name, e_cpu, e_gpu, diff, rel in rows:
        def fmt(x):
            return 'n/a' if x is None else f'{x: .10f}'
        print(f"{name:<24} {fmt(e_cpu):>16} {fmt(e_gpu):>16} {fmt(diff):>12} {fmt(rel):>12}")

    # Basic check: warn if any rel diff > 1e-12
    bad = [r for r in rows if r[4] is not None and r[4] > 1e-12]
    if bad:
        print('\nNote: Some differences exceed 1e-12 (expected ~1e-15 to 1e-14). Review outputs above.')

if __name__ == '__main__':
    main()


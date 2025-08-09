#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HEAT_RE = re.compile(r"HEAT\s+OF\s+FORMATION\s*=*\s*([\-+0-9Ee\.]+)")

def parse_heat(text: str):
    m = HEAT_RE.search(text)
    return float(m.group(1)) if m else None

def rewrite_keywords(line: str, two_gpu: bool, minblk: int|None, pair: str|None) -> str:
    key = line.strip()
    # Ensure MOZYME present
    if ' MOZYME' not in f' {key}':
        key = key + ' MOZYME'
    # Two GPU toggle
    if two_gpu and ' MOZYME_2GPU' not in f' {key}':
        key = key + ' MOZYME_2GPU'
    # Replace or add MOZYME_MINBLK
    if minblk is not None:
        if 'MOZYME_MINBLK' in key:
            key = re.sub(r'MOZYME_MINBLK\s*=\s*\d+', f'MOZYME_MINBLK={minblk}', key)
        else:
            key = key + f' MOZYME_MINBLK={minblk}'
    # Replace or add MOZYME_GPUPAIR
    if pair:
        if 'MOZYME_GPUPAIR' in key:
            key = re.sub(r'MOZYME_GPUPAIR\s*=\s*[^\s]+', f'MOZYME_GPUPAIR={pair}', key)
        else:
            key = key + f' MOZYME_GPUPAIR={pair}'
    return key

def run_mopac(mopac: Path, input_text: str, env: dict) -> tuple[int,str,float]:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)/'bench.mop'
        tmp.write_text(input_text)
        t0 = time.perf_counter()
        p = subprocess.Popen([str(mopac), str(tmp)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = p.communicate()
        dt = time.perf_counter() - t0
        return p.returncode, out, dt

def main():
    ap = argparse.ArgumentParser(description='Benchmark MOZYME peptide with varying MOZYME_MINBLK and GPU modes')
    ap.add_argument('mopac', help='Path to mopac executable (e.g., build-gpu/mopac)')
    ap.add_argument('input', help='Peptide .mop file (uses first line keywords; NEWPDB recommended)')
    ap.add_argument('--minblk', default='8,16,24,32,48,64', help='Comma-separated MOZYME_MINBLK values')
    ap.add_argument('--two-gpu', action='store_true', help='Enable MOZYME_2GPU for all runs')
    ap.add_argument('--pair', default='', help='Explicit MOZYME_GPUPAIR=a,b (1-based). Implies --two-gpu.')
    ap.add_argument('--devices', default='', help='CUDA_VISIBLE_DEVICES (e.g., 0 or 0,1). Defaults to visible devices')
    ap.add_argument('--streams', default='', help='MOPAC_STREAMS value ("off" to disable)')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat each run N times and report the best time (default 1)')
    ap.add_argument('--csv', default='', help='Optional CSV output path to append results')
    args = ap.parse_args()

    mopac = Path(args.mopac).resolve()
    if not mopac.exists():
        print(f'Executable not found: {mopac}', file=sys.stderr)
        sys.exit(2)

    inp_path = Path(args.input)
    if not inp_path.exists():
        print(f'Input not found: {inp_path}', file=sys.stderr)
        sys.exit(2)

    raw = inp_path.read_text().splitlines()
    if not raw:
        print('Empty input file', file=sys.stderr)
        sys.exit(2)
    key_line = raw[0]
    body = '\n'.join(raw[1:])

    pair = args.pair.strip() or None
    two_gpu = args.two_gpu or bool(pair)
    minblks = [int(x) for x in args.minblk.split(',') if x.strip()]

    print('Peptide bench:')
    print(f'  input   : {inp_path}')
    print(f'  mopac   : {mopac}')
    print(f'  two-gpu : {two_gpu}  pair={pair or "auto"}')
    print(f'  minblk  : {minblks}')
    print('')
    print(f"{'Mode':<10} {'MinBlk':>6} {'Time(s)':>10} {'HoF(kcal/mol)':>16}")
    csv_path = Path(args.csv) if args.csv else None
    if csv_path and not csv_path.exists():
        csv_path.write_text('mode,minblk,time_s,hof_kcal_mol,devices,pair,streams,input\n')

    for mb in minblks:
        new_key = rewrite_keywords(key_line, two_gpu, mb, pair)
        new_text = new_key + '\n' + body
        env = os.environ.copy()
        # Force GPU usage
        env['MOPAC_FORCEGPU'] = '1'
        if args.devices:
            env['CUDA_VISIBLE_DEVICES'] = args.devices
        if args.streams:
            env['MOPAC_STREAMS'] = args.streams
        best_dt = None
        best_hof = None
        for _ in range(max(1, args.repeat)):
            rc, out, dt = run_mopac(mopac, new_text, env)
            hof = parse_heat(out)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_hof = hof
        mode = '2GPU' if two_gpu else '1GPU'
        print(f"{mode:<10} {mb:>6d} {best_dt:>10.3f} {('n/a' if best_hof is None else f'{best_hof: 12.6f}'):>16}")
        if csv_path:
            with csv_path.open('a') as f:
                f.write(f"{mode},{mb},{best_dt},{'' if best_hof is None else best_hof},{args.devices or os.environ.get('CUDA_VISIBLE_DEVICES','')},{pair or ''},{args.streams},{inp_path}\n")

if __name__ == '__main__':
    main()

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
import platform

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

def run_mopac(mopac: Path, input_text: str, env: dict, keep_tmp: bool = False, tmp_root: str | None = None) -> tuple[int,str,float,Path,Path]:
    """Run MOPAC in a temporary directory. If keep_tmp is True, the directory is preserved.

    Returns: (returncode, output_text, elapsed_time, out_path, tmp_dir)
    """
    if keep_tmp:
        td = Path(tempfile.mkdtemp(dir=tmp_root))
        tmp_dir = td
        tmp = tmp_dir / 'bench.mop'
        tmp.write_text(input_text)
        t0 = time.perf_counter()
        p = subprocess.Popen([str(mopac), str(tmp)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = p.communicate()
        dt = time.perf_counter() - t0
        out_path = tmp.with_suffix('.out')
        out_text = out
        try:
            if out_path.exists():
                out_text = out_path.read_text(errors='ignore')
        except Exception:
            pass
        return p.returncode, out_text, dt, out_path, tmp_dir
    else:
        with tempfile.TemporaryDirectory(dir=tmp_root) as td:
            tmp_dir = Path(td)
            tmp = tmp_dir / 'bench.mop'
            tmp.write_text(input_text)
            t0 = time.perf_counter()
            p = subprocess.Popen([str(mopac), str(tmp)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out, _ = p.communicate()
            dt = time.perf_counter() - t0
            out_path = tmp.with_suffix('.out')
            out_text = out
            try:
                if out_path.exists():
                    out_text = out_path.read_text(errors='ignore')
            except Exception:
                pass
            return p.returncode, out_text, dt, out_path, tmp_dir

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
    ap.add_argument('--keep-tmp', action='store_true', help='Keep temporary run directory (prints path)')
    ap.add_argument('--tmp-root', default='', help='Optional root folder for temporary directories')
    ap.add_argument('--csv-append-meta', action='store_true', help='Write provenance comments to new CSV files (GPU, driver, MOPAC version, platform)')
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
        # Optionally write provenance as comment lines
        if args.csv_append_meta:
            meta_lines = []
            # Platform / Python
            meta_lines.append(f"# timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            meta_lines.append(f"# platform: {platform.platform()}")
            meta_lines.append(f"# python: {platform.python_version()}")
            # CUDA / NVIDIA driver (best-effort)
            try:
                smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                if smi.returncode == 0 and smi.stdout.strip():
                    meta_lines.append('# nvidia-smi:')
                    for line in smi.stdout.strip().splitlines():
                        meta_lines.append(f"#   {line.strip()}")
            except Exception:
                pass
            # MOPAC version (best-effort)
            try:
                mv = subprocess.run([str(mopac), '-V'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
                if mv.stdout:
                    meta_lines.append(f"# mopac: {mv.stdout.strip()}")
            except Exception:
                pass
            # Benchmark params
            meta_lines.append(f"# input: {inp_path}")
            meta_lines.append(f"# two_gpu: {two_gpu} pair: {pair or ''} devices: {args.devices or os.environ.get('CUDA_VISIBLE_DEVICES','')}")
            meta_lines.append(f"# streams: {args.streams or ''}")
            meta_lines.append(f"# minblk_list: {','.join(str(x) for x in minblks)}")
            csv_path.write_text('\n'.join(meta_lines) + '\n')
        # Write header
        with csv_path.open('a') as f:
            f.write('mode,minblk,time_s,hof_kcal_mol,devices,pair,streams,input,out_path,tmp_dir\n')

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
        best_out_path = None
        best_tmp_dir = None
        for _ in range(max(1, args.repeat)):
            rc, out_text, dt, out_path, tmp_dir = run_mopac(mopac, new_text, env, keep_tmp=args.keep_tmp, tmp_root=(args.tmp_root or None))
            hof = parse_heat(out_text)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_hof = hof
                best_out_path = out_path
                best_tmp_dir = tmp_dir
        mode = '2GPU' if two_gpu else '1GPU'
        print(f"{mode:<10} {mb:>6d} {best_dt:>10.3f} {('n/a' if best_hof is None else f'{best_hof: 12.6f}'):>16}")
        if args.keep_tmp and best_out_path:
            print(f"  kept: {best_out_path}")
        if csv_path:
            with csv_path.open('a') as f:
                f.write(
                    f"{mode},{mb},{best_dt},{'' if best_hof is None else best_hof},{args.devices or os.environ.get('CUDA_VISIBLE_DEVICES','')},"
                    f"{pair or ''},{args.streams},{inp_path},{best_out_path or ''},{best_tmp_dir or ''}\n"
                )

if __name__ == '__main__':
    main()

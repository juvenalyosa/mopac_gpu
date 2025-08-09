#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

def load_csv(path: Path):
    rows = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    'mode': row.get('mode','').strip(),
                    'minblk': int(row.get('minblk','0')),
                    'time_s': float(row.get('time_s','nan')),
                    'hof': row.get('hof_kcal_mol',''),
                    'devices': row.get('devices',''),
                    'pair': row.get('pair',''),
                    'streams': row.get('streams',''),
                    'input': row.get('input',''),
                })
            except Exception:
                continue
    return rows

def main():
    ap = argparse.ArgumentParser(description='Plot MOZYME peptide benchmark CSV (minblk vs time)')
    ap.add_argument('csvs', nargs='+', help='CSV file(s) from peptide_bench.py')
    ap.add_argument('--labels', default='', help='Comma-separated labels for each CSV')
    ap.add_argument('--out', default='', help='Output image path (PNG). Default: derived from first CSV')
    ap.add_argument('--title', default='MOZYME Benchmark', help='Plot title')
    ap.add_argument('--show', action='store_true', help='Show the plot interactively')
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib is required: pip install matplotlib', flush=True)
        return

    labels = [s.strip() for s in args.labels.split(',')] if args.labels else []
    if labels and len(labels) != len(args.csvs):
        print('Number of labels must match number of CSVs', flush=True)
        return

    fig, ax = plt.subplots(figsize=(7,4.2))
    for i, csv_path in enumerate(args.csvs):
        rows = load_csv(Path(csv_path))
        if not rows:
            continue
        # group by minblk; if multiple modes exist, prefer 2GPU over 1GPU mark in label
        xs = sorted({r['minblk'] for r in rows if r['minblk']})
        # pick best (min) time per minblk
        ys = []
        for mb in xs:
            tbest = min([r['time_s'] for r in rows if r['minblk']==mb])
            ys.append(tbest)
        label = labels[i] if labels else Path(csv_path).stem
        ax.plot(xs, ys, marker='o', label=label)

    ax.set_xlabel('MOZYME_MINBLK')
    ax.set_ylabel('Time (s)')
    ax.set_title(args.title)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if args.out:
        out = Path(args.out)
    else:
        out = Path(args.csvs[0]).with_suffix('.png')
    fig.savefig(out, dpi=150)
    print(f'Wrote {out}')

    if args.show:
        plt.show()

if __name__ == '__main__':
    main()


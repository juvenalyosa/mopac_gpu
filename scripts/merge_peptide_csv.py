#!/usr/bin/env python3
import argparse
from pathlib import Path

def read_rows(csv_path: Path):
    rows = []
    header = None
    with csv_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith('#'):
                continue
            if header is None:
                header = line.strip()
            else:
                rows.append(line.strip())
    return header, rows

def main():
    ap = argparse.ArgumentParser(description='Merge peptide_bench CSVs and annotate with labels')
    ap.add_argument('csvs', nargs='+', help='Input CSV files')
    ap.add_argument('--labels', default='', help='Comma-separated labels for each CSV (default: filename stems)')
    ap.add_argument('--out', default='combined.csv', help='Output CSV path (default combined.csv)')
    ap.add_argument('--prepend', action='store_true', help='Insert label as first column (default: append at end)')
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(',')] if args.labels else []
    if labels and len(labels) != len(args.csvs):
        print('Number of labels must match number of CSVs')
        return

    csv_paths = [Path(p) for p in args.csvs]
    outputs = []
    base_header = None
    for idx, p in enumerate(csv_paths):
        header, rows = read_rows(p)
        if header is None:
            continue
        if base_header is None:
            base_header = header
        elif base_header != header:
            # Allow different headers if they share a common prefix; otherwise prefer first
            pass
        lab = labels[idx] if labels else p.stem
        for r in rows:
            if args.prepend:
                outputs.append(f'{lab},{r}')
            else:
                outputs.append(f'{r},{lab}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        if base_header is None:
            f.write('')
            return
        if args.prepend:
            f.write(f'label,{base_header}\n')
        else:
            f.write(f'{base_header},label\n')
        for r in outputs:
            f.write(r + '\n')
    print(f'Wrote {out_path} ({len(outputs)} rows)')

if __name__ == '__main__':
    main()


#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 -e <mopac_exec> -i <input.mop> -o <basename> [-m "8,16,24,32,48,64"] \
          [--2gpu] [--pair 1,2] [--devices 0,1] [--streams off] [--repeat 3] [--title "Plot Title"]

Runs peptide_bench.py to sweep MOZYME_MINBLK values and then plots minblk vs time.

Required:
  -e, --exec     Path to mopac executable (e.g., build-gpu/mopac)
  -i, --input    Peptide .mop input (uses keyword line and NEWPDB)
  -o, --out      Output basename (produces <basename>.csv and <basename>.png)

Optional:
  -m, --minblk   Comma-separated list (default: 8,16,24,32,48,64)
      --2gpu     Enable MOZYME_2GPU
      --pair     1-based pair (e.g., 1,2)
      --devices  CUDA_VISIBLE_DEVICES (e.g., 0,1)
      --streams  MOPAC_STREAMS value ("off" to disable)
      --repeat   Repeat each run; take best (default 1)
      --title    Plot title (default derived from input)

Examples:
  $0 -e ./build-gpu/mopac -i examples/peptide_gg.mop -o gg_sweep
  $0 -e ./build-gpu/mopac -i examples/peptide_aaa.mop -o aaa_2gpu --2gpu --pair 1,2 --devices 0,1 --repeat 3
EOF
}

MOPAC_EXEC=""
INPUT=""
OUTBASE=""
MINBLK="8,16,24,32,48,64"
TWOGPU="0"
PAIR=""
DEVICES=""
STREAMS=""
REPEAT="1"
TITLE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--exec) MOPAC_EXEC="$2"; shift 2;;
    -i|--input) INPUT="$2"; shift 2;;
    -o|--out) OUTBASE="$2"; shift 2;;
    -m|--minblk) MINBLK="$2"; shift 2;;
    --2gpu) TWOGPU="1"; shift;;
    --pair) PAIR="$2"; shift 2;;
    --devices) DEVICES="$2"; shift 2;;
    --streams) STREAMS="$2"; shift 2;;
    --repeat) REPEAT="$2"; shift 2;;
    --title) TITLE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "$MOPAC_EXEC" || -z "$INPUT" || -z "$OUTBASE" ]]; then
  usage; exit 2
fi

CSV="${OUTBASE}.csv"
PNG="${OUTBASE}.png"

ARGS=("scripts/peptide_bench.py" "$MOPAC_EXEC" "$INPUT" --minblk "$MINBLK" --csv "$CSV" --repeat "$REPEAT")
if [[ "$TWOGPU" == "1" ]]; then ARGS+=(--two-gpu); fi
if [[ -n "$PAIR" ]]; then ARGS+=(--pair "$PAIR"); fi
if [[ -n "$DEVICES" ]]; then ARGS+=(--devices "$DEVICES"); fi
if [[ -n "$STREAMS" ]]; then ARGS+=(--streams "$STREAMS"); fi

echo "[peptide_sweep_plot] Running: ${ARGS[*]}"
python3 "${ARGS[@]}"

PTITLE="$TITLE"
if [[ -z "$PTITLE" ]]; then
  PTITLE="MOZYME minblk sweep: $(basename "$INPUT")"
fi

echo "[peptide_sweep_plot] Plotting to $PNG"
python3 scripts/plot_peptide_csv.py "$CSV" --out "$PNG" --title "$PTITLE"
echo "Done. CSV: $CSV, PNG: $PNG"


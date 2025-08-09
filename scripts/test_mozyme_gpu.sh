#!/usr/bin/env bash
set -euo pipefail

# Simple verification for MOZYME CPU vs 1-GPU vs 2-GPU energy agreement.
# Usage:
#   scripts/test_mozyme_gpu.sh /path/to/mopac /path/to/input.mop [gpu_pair] [tolerance]
#
# Notes:
# - The input should already contain MOZYME keywords; this script appends NOGPU/MOZYME_GPU/MOZYME_2GPU as needed.
# - If your machine has >2 GPUs and you want a specific pair, pass a 1-based pair as third arg, e.g. "1,2".
# - Results are parsed from the generated .out files by grepping "FINAL HEAT OF FORMATION".

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /path/to/mopac /path/to/input.mop [gpu_pair] [tolerance]" >&2
  exit 1
fi

MOPAC_BIN="$1"
INPUT="$2"
PAIR="${3:-}"
TOL="${4:-1e-4}"

if [[ ! -x "$MOPAC_BIN" ]]; then
  echo "mopac binary not executable: $MOPAC_BIN" >&2
  exit 1
fi
if [[ ! -f "$INPUT" ]]; then
  echo "input file not found: $INPUT" >&2
  exit 1
fi

base=$(basename "$INPUT")
stem="${base%.*}"
tmpdir=$(mktemp -d ${stem}_testgpu.XXXXXX)
cp "$INPUT" "$tmpdir/$stem.cpu.mop"
cp "$INPUT" "$tmpdir/$stem.gpu1.mop"
cp "$INPUT" "$tmpdir/$stem.gpu2.mop"

# Helper: prepend keywords to the first line
prepend_kw() {
  local file="$1" kw="$2"
  awk -v pre="$kw" 'NR==1{print pre" " $0; next} {print}' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

# CPU run: enforce NOGPU
prepend_kw "$tmpdir/$stem.cpu.mop" "NOGPU"

# 1-GPU run: enable MOZYME GPU
prepend_kw "$tmpdir/$stem.gpu1.mop" "MOZYME_GPU"

# 2-GPU run: force multi-GPU; optionally set device pair
kw2="MOZYME_GPU MOZYME_2GPU"
if [[ -n "$PAIR" ]]; then
  kw2+=" MOZYME_GPUPAIR=$PAIR"
fi
prepend_kw "$tmpdir/$stem.gpu2.mop" "$kw2"

pushd "$tmpdir" >/dev/null

echo "Running CPU (NOGPU)…"
"$MOPAC_BIN" "$stem.cpu.mop" > cpu.log 2>&1 || true
echo "Running 1-GPU (MOZYME_GPU)…"
"$MOPAC_BIN" "$stem.gpu1.mop" > gpu1.log 2>&1 || true
echo "Running 2-GPU (MOZYME_2GPU)…"
"$MOPAC_BIN" "$stem.gpu2.mop" > gpu2.log 2>&1 || true

# Try to find energies in .out or logs
extract_energy() {
  local tag="$1"
  local out_file
  out_file=$(ls -1 ${tag%.*}*.out 2>/dev/null | head -n1 || true)
  if [[ -n "$out_file" ]]; then
    grep -m1 -E "FINAL HEAT OF FORMATION" "$out_file" | awk '{print $(NF-1)}' || true
  else
    grep -m1 -E "FINAL HEAT OF FORMATION" "$tag".log | awk '{print $(NF-1)}' || true
  fi
}

ECPU=$(extract_energy "$stem.cpu")
E1G=$(extract_energy "$stem.gpu1")
E2G=$(extract_energy "$stem.gpu2")

echo "CPU   HOF: $ECPU"
echo "1-GPU HOF: $E1G"
echo "2-GPU HOF: $E2G"

# Basic validation
if [[ -z "$ECPU" || -z "$E1G" || -z "$E2G" ]]; then
  echo "ERROR: Failed to extract energies. Check logs in $tmpdir" >&2
  exit 2
fi

abs_diff() {
  awk -v x="$1" -v y="$2" 'BEGIN{d=(x-y); if(d<0)d=-d; print d}'
}

cmp_le() {
  awk -v d="$1" -v t="$2" 'BEGIN{exit (d<=t)?0:1}'
}

D1=$(abs_diff "$ECPU" "$E1G")
D2=$(abs_diff "$ECPU" "$E2G")

echo "|1-GPU - CPU| = $D1 (tol=$TOL)"
echo "|2-GPU - CPU| = $D2 (tol=$TOL)"

rc=0
if ! cmp_le "$D1" "$TOL"; then
  echo "FAIL: 1-GPU energy deviates beyond tolerance" >&2
  rc=3
fi
if ! cmp_le "$D2" "$TOL"; then
  echo "FAIL: 2-GPU energy deviates beyond tolerance" >&2
  rc=4
fi

if [[ $rc -eq 0 ]]; then
  echo "PASS: GPU energies within tolerance."
else
  echo "See logs in: $tmpdir" >&2
fi

exit $rc
popd >/dev/null

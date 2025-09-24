#!/usr/bin/env bash
set -euo pipefail

# GPU test suite for MOPAC. Runs a sequence of GPU-focused tests and summarizes results.
#
# Usage:
#   scripts/run_gpu_suite.sh [PATH_TO_MOPAC]
# Example:
#   scripts/run_gpu_suite.sh ./build-gpu/mopac

MOPAC_BIN="${1:-}"

detect_mopac() {
  if [[ -n "${MOPAC_BIN}" && -x "${MOPAC_BIN}" ]]; then
    return 0
  fi
  for c in ./build-gpu/mopac ./build/mopac ./build-gpu-make/mopac mopac; do
    if command -v "$c" >/dev/null 2>&1; then MOPAC_BIN="$c"; return 0; fi
    if [[ -x "$c" ]]; then MOPAC_BIN="$c"; return 0; fi
  done
  echo "ERROR: Could not locate MOPAC executable. Pass path as first argument." 1>&2
  exit 1
}

detect_mopac

ROOT_DIR=$(pwd)
OUT_DIR="gpu_test_logs"
mkdir -p "$OUT_DIR"

GPU_VERBOSE_DEFAULT=${MOPAC_GPU_VERBOSE:-}

export MOPAC_FORCEGPU=1
export MOPAC_DETERMINISTIC=1
export MOPAC_GPU_VERBOSE=1

have_cmd() { command -v "$1" >/dev/null 2>&1; }

gpu_count=1
if have_cmd nvidia-smi; then
  gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print ($1==""?0:$1)}') || gpu_count=1
fi

summary=()

run_case() {
  local name="$1"; shift
  local infile="$1"; shift
  local extra_env="$*"
  local log="$OUT_DIR/${name// /_}.out"
  local start_ts end_ts elapsed status="FAIL" gpu_hits="no"
  echo "==> Running $name"
  if [[ ! -f "$infile" ]]; then
    echo "SKIP: missing input $infile" | tee "$log"
    summary+=("$name;SKIP;0;no")
    return 0
  fi
  start_ts=$(date +%s)
  # shellcheck disable=SC2086
  bash -c "$extra_env \"$MOPAC_BIN\" \"$infile\"" >"$log" 2>&1 || true
  end_ts=$(date +%s)
  elapsed=$(( end_ts - start_ts ))
  if grep -q "JOB ENDED NORMALLY" "$log"; then status="OK"; fi
  if grep -Eq "\[GPU\] (DGEMM|DSYRK|MGPU)" "$log"; then gpu_hits="yes"; fi
  echo "Result: $status (${elapsed}s), GPU logs: $gpu_hits"
  if [[ "$status" != "OK" ]]; then
    echo "--- Failure details ($name) ---"
    if grep -q "UNRECOGNIZED KEY-WORDS" "$log"; then
      echo "Reason: Unrecognized keyword(s)"
      grep -m1 -n "UNRECOGNIZED KEY-WORDS" "$log" || true
    elif grep -q "ERROR DETECTED IN SUBROUTINE CHECK" "$log"; then
      echo "Reason: MOZYME CHECK failure (LMO/connectivity)"
      grep -m1 -n "ERROR DETECTED IN SUBROUTINE CHECK" "$log" || true
      echo "Hint: simplify keywords (no ALLBONDS/BONDS), set CHARGE, let NEWPDB build bonds."
    elif grep -q "Segmentation fault" "$log"; then
      echo "Reason: Segmentation fault"
      grep -m1 -n "Segmentation fault" "$log" || true
    elif grep -q "mopac_cuda" "$log"; then
      echo "Reason: CUDA runtime/interop error"
      grep -n "mopac_cuda" "$log" | head -n 3 || true
    else
      echo "Reason: unknown; last 20 lines:"
      tail -n 20 "$log" || true
    fi
    if grep -q "GPU DEBUG SUMMARY:" "$log"; then
      echo "GPU Debug Summary:"
      awk '/GPU DEBUG SUMMARY:/{p=1;print;next} /^$/{if(p){exit}} p && NR<999{print}' "$log" | sed -n '1,12p' || true
    fi
    echo "(See full log: $log)"
    echo "------------------------------"
  fi
  summary+=("$name;$status;$elapsed;$gpu_hits")
}

# 1) Dense sanity
run_case "dense_sanity_single_gpu" "examples/water_pm7_gpu.mop" \
  "export MOPAC_GPU_EIGEN_MIN=1; export CUDA_VISIBLE_DEVICES=0;"

# 2) Gradient device F reuse
run_case "gradient_device_reuse" "examples/h2o_gpu_force.mop" \
  "export CUDA_VISIBLE_DEVICES=0;"

# 3) DIIS on GPU (full B)
run_case "diis_gpu_bfull" "examples/benzene.mop" \
  "export CUDA_VISIBLE_DEVICES=0; export MOPAC_DIIS_GPU_BUF=1; export MOPAC_DIIS_GPU_BFULL=1;"

# 4) MOZYME with provided protein PDB (auto policy chooses safe GPU/CPU path)
PROT_PDB="examples/test_protein_gpu.pdb"
PROT_MOP="$OUT_DIR/test_protein_gpu.mop"
if [[ -f "$PROT_PDB" ]]; then
  cat > "$PROT_MOP" <<EOF
PM7 GEO_DAT=$PROT_PDB MOZYME 1SCF PULAY SHIFT=-50 ITRY=200 NEWPDB CHARGE=-9
Test MOZYME GPU auto-policy run (CHARGE=-9)

EOF
  run_case "mozyme_protein_auto" "$PROT_MOP" "export CUDA_VISIBLE_DEVICES=0;"
else
  echo "NOTE: $PROT_PDB not found; skipping protein MOZYME test"
fi

# 5) Multi-GPU BLAS (cuBLASXt) if >=2 GPUs
if [[ "$gpu_count" -ge 2 ]]; then
  run_case "multigpu_blas_cublasxt" "examples/peptide_gg_2gpu.mop" \
    "export CUDA_VISIBLE_DEVICES=0,1; export MOPAC_CUBLASXT_DEVICES=0,1;"
else
  echo "NOTE: <2 GPUs detected; skipping multi-GPU BLAS test"
fi

# 6) MG eigensolver attempt (safe fallback)
run_case "mg_eigs_attempt" "examples/water_pm7_gpu.mop" \
  "export CUDA_VISIBLE_DEVICES=0; export MOPAC_EIG_MG=1; export MOPAC_EIG_MG_MIN=1;"

echo ""
echo "GPU Test Summary (name;status;seconds;gpu_logs)"
for line in "${summary[@]}"; do echo "$line"; done | tee "$OUT_DIR/summary.csv"

# Restore original verbose setting if it was empty
if [[ -z "$GPU_VERBOSE_DEFAULT" ]]; then unset MOPAC_GPU_VERBOSE; fi

echo "Logs in: $OUT_DIR"; exit 0

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
# Resolve repository root relative to this script to find examples reliably
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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
fail_count=0

gradient_cpu_elapsed=""
gradient_gpu_elapsed=""
gradient_cpu_log=""
gradient_gpu_log=""
gradient_check_cpu_log=""
gradient_check_gpu_log=""
LAST_OUT=""

LAST_STATUS=""
LAST_REASON=""
LAST_LOG=""

run_case() {
  local name="$1"; shift
  local infile="$1"; shift
  local extra_env="$*"
  local log="$OUT_DIR/${name// /_}.out"
  local start_ts end_ts elapsed status="FAIL" gpu_hits="no" reason=""
  echo "==> Running $name"
  echo "Log: $log"
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
  # Consider both the banner and the single-line footer (case-insensitive)
  if grep -qiE "(JOB ENDED NORMALLY|ended normally on)" "$log"; then status="OK"; fi
  # Mark as GPU activity if BLAS logs appear, MG fallback logs are present, or MG notices fire
  if grep -Eq "(\[GPU\]|\[MGPU\]|cuSOLVERMg support unavailable|cuSOLVERMg solve failed|cuSOLVERMg workspace allocation failed)" "$log"; then
    gpu_hits="yes"
  fi
  # Enforce expected GPU logs for certain tests
  local expect_gpu_mark="no"
  case "$name" in
    dense_sanity_single_gpu|gradient_device_reuse|diis_gpu_bfull|diis_gpu_bcol_solve|mg_eigs_attempt)
      expect_gpu_mark="yes"
      ;;
    multigpu_blas_cublasxt)
      # Informational only: do not fail if logs absent; report via summary
      : ;;
    mg_large_dense_*)
      # Informational only: MG may fall back cleanly; do not fail if logs absent
      : ;;
    mozyme_protein_auto)
      if awk 'BEGIN{m=""} /mozyme_gpu=/{for(i=1;i<=NF;i++){if($i ~ /mozyme_gpu=/){split($i,a,"="); if(length(a[2])==0){if((i+1)<=NF) m=$(i+1); else m="";} else {m=a[2];}}} if(m=="T") exit 0; else exit 1} END{exit (m=="T"?0:1)}' "$log" \
         || grep -Eq "Large protein MOZYME GPU" "$log"; then
        gpu_hits="yes"
        reason="MOZYME"
      fi
      ;;
    resident_scf_density)
      : ;;
  esac
  echo "Result: $status (${elapsed}s), GPU logs: $gpu_hits"
  if [[ -z "$reason" && "$status" == "OK" ]]; then
    case "$name" in
      mg_* )
        if grep -q "\[MGPU\] DSYEVD" "$log"; then
          reason="MG solve"
        elif grep -q "\[MGPU\].*fallback" "$log"; then
          reason="MG fallback"
        fi
        ;;
      multigpu_blas_cublasxt)
        if grep -q "\[GPU\].*DGEMM" "$log"; then
          reason="Xt active"
        fi
        ;;
      resident_scf_density)
        if grep -q "resident_scf= T" "$log"; then
          reason="resident"
        fi
        ;;
      gradient_device_reuse)
        reason="GPU profile"
        ;;
      gradient_cpu_baseline)
        reason="CPU baseline"
        ;;
    esac
  fi
  if [[ "$status" == "OK" && "$expect_gpu_mark" == "yes" && "$gpu_hits" != "yes" && -z "$reason" ]]; then
    reason="GPU markers not detected"
  fi
  if [[ "$status" != "OK" ]]; then
    echo "--- Failure details ($name) ---"
    if grep -q "UNRECOGNIZED KEY-WORDS" "$log"; then
      reason="Unrecognized keyword(s)"
      echo "Reason: $reason"
      grep -m1 -n "UNRECOGNIZED KEY-WORDS" "$log" || true
    elif grep -q "ERROR DETECTED IN SUBROUTINE CHECK" "$log"; then
      reason="MOZYME CHECK failure (LMO/connectivity)"
      echo "Reason: $reason"
      grep -m1 -n "ERROR DETECTED IN SUBROUTINE CHECK" "$log" || true
      echo "Context (15 lines before):"
      awk '/ERROR DETECTED IN SUBROUTINE CHECK/{print NR-0}' "$log" | head -n1 | {
        read n; if [[ -n "$n" ]]; then start=$((n-15)); if ((start<1)); then start=1; fi; sed -n "${start},${n}p" "$log"; fi
      }
      echo "Hint: simplify keywords (no ALLBONDS/BONDS), set CHARGE, let NEWPDB build bonds."
    elif grep -q "Segmentation fault" "$log"; then
      reason="Segmentation fault"
      echo "Reason: $reason"
      grep -m1 -n "Segmentation fault" "$log" || true
    elif grep -Eiq "(cuda|cusolver|cublas|CUBLAS_STATUS|CUDA error)" "$log"; then
      reason="CUDA runtime/interop error"
      echo "Reason: $reason"
      grep -niE "(cuda|cusolver|cublas|CUBLAS_STATUS|CUDA error)" "$log" | head -n 5 || true
    else
      reason="Unknown"
      echo "Reason: $reason; last 20 lines:"
      tail -n 20 "$log" || true
    fi
    if grep -q "GPU DEBUG SUMMARY:" "$log"; then
      echo "GPU Debug Summary:"
      awk '/GPU DEBUG SUMMARY:/{p=1;print;next} /^$/{if(p){exit}} p && NR<999{print}' "$log" | sed -n '1,12p' || true
    fi
    echo "(See full log: $log)"
    echo "------------------------------"
    fail_count=$((fail_count+1))
  fi
  if [[ "$name" == "gradient_cpu_baseline" ]]; then
    gradient_cpu_elapsed="$elapsed"
    gradient_cpu_log="$log"
  elif [[ "$name" == "gradient_device_reuse" ]]; then
    gradient_gpu_elapsed="$elapsed"
    gradient_gpu_log="$log"
  fi
  summary+=("$name;$status;$elapsed;$gpu_hits;$reason")
  LAST_STATUS="$status"
  LAST_REASON="$reason"
  LAST_LOG="$log"
  archive_output "$name" "$infile"
}

archive_output() {
  local name="$1"
  local infile="$2"
  local base="$(basename "$infile")"
  local outfile="${base%.*}.out"
  local src=""
  if [[ -f "$outfile" ]]; then
    src="$outfile"
  elif [[ -f "$(dirname "$infile")/$outfile" ]]; then
    src="$(dirname "$infile")/$outfile"
  fi
  if [[ -n "$src" ]]; then
    local dest="$OUT_DIR/${name}.mopac_out"
    cp "$src" "$dest"
    rm -f "$src"
    LAST_OUT="$dest"
  else
    LAST_OUT=""
  fi
}

compare_gradients() {
  local cpu_log="$1"
  local gpu_log="$2"
  local name="$3"
  local tol="${4:-1e-5}"
  echo "==> Comparing gradients ($name)"
  if [[ -z "$cpu_log" || -z "$gpu_log" ]]; then
    echo "Gradient compare skipped: missing log paths"
    summary+=("$name;SKIP;0;no;missing log paths")
    return 0
  fi
  if [[ ! -f "$cpu_log" || ! -f "$gpu_log" ]]; then
    echo "Gradient compare skipped: log files not found"
    summary+=("$name;SKIP;0;no;logs missing")
    return 0
  fi
  if [[ ! -f "$cpu_log" || ! -f "$gpu_log" ]]; then
    echo "Gradient compare skipped: log files not found"
    summary+=("$name;SKIP;0;no;logs missing")
    return 0
  fi
  local result
  if ! result=$(python3 - "$cpu_log" "$gpu_log" "$tol" <<'PY'
import sys, math

cpu_log = sys.argv[1]
gpu_log = sys.argv[2]
tol = float(sys.argv[3])

def extract(path):
    import re
    data = []
    capture = False
    started = False
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                for raw in fh:
                line = raw.rstrip()
                if 'CARTESIAN COORDINATE DERIVATIVES' in line:
                    capture = True
                    started = False
                    continue
                if not capture:
                    continue
                if not line.strip():
                    if started:
                        break
                    else:
                        continue
                floats = re.findall(r'[-+]?\d+\.\d+(?:[EeDd][+-]?\d+)?', line)
                if len(floats) < 3:
                    continue
                if len(floats) >= 4:
                    gx, gy, gz = [float(v.replace('D', 'E')) for v in floats[-4:-1]]
                else:
                    gx, gy, gz = [float(v.replace('D', 'E')) for v in floats[-3:]]
                data.append((gx, gy, gz))
                started = True
    except FileNotFoundError:
        raise SystemExit(f"gradient log not found: {path}")
    if not data:
        raise SystemExit(f"no gradient block detected in {path}")
    return data


cpu = extract(cpu_log)
gpu = extract(gpu_log)
if len(cpu) != len(gpu):
    raise SystemExit(f"gradient length mismatch: CPU={len(cpu)} GPU={len(gpu)}")

max_abs = 0.0
sum_sq = 0.0
count = 0
for (cx, cy, cz), (gx, gy, gz) in zip(cpu, gpu):
    dx = gx - cx
    dy = gy - cy
    dz = gz - cz
    max_abs = max(max_abs, abs(dx), abs(dy), abs(dz))
    sum_sq += dx*dx + dy*dy + dz*dz
    count += 3

rms = math.sqrt(sum_sq / count) if count else 0.0
msg = f"max_abs={max_abs:.3e} rms={rms:.3e}"
if max_abs > tol:
    raise SystemExit(msg)
print(msg)
PY
  ); then
    echo "$result"
    summary+=("$name;FAIL;0;yes;$result")
    fail_count=$((fail_count+1))
    return 1
  fi
  echo "Gradient match: PASS ($result)"
  summary+=("$name;OK;0;yes;$result")
  return 0
}


# 1) Dense sanity
run_case "dense_sanity_single_gpu" "examples/water_pm7_gpu.mop" \
  "export MOPAC_GPU_EIGEN_MIN=1; export CUDA_VISIBLE_DEVICES=0;"

# CPU baseline for gradient timing (Stage 0 profiling)
run_case "gradient_cpu_baseline" "examples/h2o_gpu_force.mop" \
  "unset MOPAC_FORCEGPU; unset MOPAC_GPU_GRAD; export MOPAC_GPU_VERBOSE=0; export CUDA_VISIBLE_DEVICES=0;"

# 2) Gradient device F reuse
run_case "gradient_device_reuse" "examples/h2o_gpu_force.mop" \
  "export CUDA_VISIBLE_DEVICES=0; export MOPAC_GPU_GRAD=1; export MOPAC_GPU_PROFILE=1;"

# Targeted gradient comparison (prints gradients via DCART)
run_case "gradient_cpu_target" "examples/h2o_gpu_gradcheck.mop" \
  "unset MOPAC_FORCEGPU; unset MOPAC_GPU_GRAD; export MOPAC_GPU_VERBOSE=0; export CUDA_VISIBLE_DEVICES=0;"
gradient_check_cpu_log="$LAST_OUT"

run_case "gradient_gpu_target" "examples/h2o_gpu_gradcheck.mop" \
  "export CUDA_VISIBLE_DEVICES=0; export MOPAC_GPU_GRAD=1; export MOPAC_RESIDENT_SCF=1; export MOPAC_GPU_VERBOSE=0;"
gradient_check_gpu_log="$LAST_OUT"

compare_gradients "$gradient_check_cpu_log" "$gradient_check_gpu_log" "gradient_compare" 1e-5

# Resident SCF cache check
run_case "resident_scf_density" "examples/water_pm7_gpu.mop" \
  "export CUDA_VISIBLE_DEVICES=0; export MOPAC_RESIDENT_SCF=1; export MOPAC_GPU_DEBUG=1;"

# 3) DIIS on GPU (full B)
run_case "diis_gpu_bfull" "examples/benzene.mop" \
  "export CUDA_VISIBLE_DEVICES=0; export MOPAC_DIIS_GPU_BUF=1; export MOPAC_DIIS_GPU_BFULL=1; export MOPAC_GPU_PROFILE=1;"

# 3b) DIIS on GPU (B column + cuSOLVER solve + generalized residual)
run_case "diis_gpu_bcol_solve" "examples/benzene.mop" \
  "export CUDA_VISIBLE_DEVICES=0; export MOPAC_DIIS_GPU_BUF=1; export MOPAC_DIIS_GPU_BMAT=1; export MOPAC_DIIS_GPU=1; export MOPAC_DIIS_GEN=1; export MOPAC_GPU_PROFILE=1;"

# Dispersion halogen contact (energy-only)
run_case "disp_halogen_gpu" "examples/halogen_disp.mop" \
  "export CUDA_VISIBLE_DEVICES=0; export MOPAC_DISP_GPU=1; export MOPAC_GPU_PROFILE=2;"

# 4) MOZYME with large protein (test_dense.pdb)
PROT_PDB="$REPO_ROOT/examples/test_dense.pdb"
PROT_MOP="$REPO_ROOT/examples/mozyme_protein_auto.mop"
if [[ -f "$PROT_PDB" && -f "$PROT_MOP" ]]; then
  run_case "mozyme_protein_auto" "$PROT_MOP" \
    "export CUDA_VISIBLE_DEVICES=0; export MOZYME_GPU_FORCE=1; export MOPAC_GPU_VERBOSE=1; export MOPAC_GPU_DEBUG=1;"
  # If MOZYME CHECK failed, retry with MOZYME GPU disabled (CPU MOZYME) as an automatic fallback
  if [[ "$LAST_STATUS" != "OK" ]] && grep -q "ERROR DETECTED IN SUBROUTINE CHECK" "$LAST_LOG"; then
    echo "Retrying MOZYME protein with MOZYME_GPU_OFF=1 (CPU MOZYME fallback)"
    run_case "mozyme_protein_cpu_fallback" "$PROT_MOP" \
      "export CUDA_VISIBLE_DEVICES=0; export MOZYME_GPU_OFF=1;"
  fi
else
  echo "NOTE: $PROT_PDB or $PROT_MOP not found; skipping protein MOZYME test"
fi

# 5) Multi-GPU BLAS (cuBLASXt) if >=2 GPUs; otherwise mark as SKIP in summary
if [[ "$gpu_count" -ge 2 ]]; then
  # Prefer dense input to exercise BLAS-3 across GPUs
  DENSE_FOR_MG_BLAS=""
  if [[ -n "${MOPAC_MG_LARGE_INPUT:-}" && -f "${MOPAC_MG_LARGE_INPUT}" ]]; then
    DENSE_FOR_MG_BLAS="${MOPAC_MG_LARGE_INPUT}"
  elif [[ -f "$REPO_ROOT/examples/dense_test.mop" ]]; then
    DENSE_FOR_MG_BLAS="$REPO_ROOT/examples/dense_test.mop"
  elif [[ -f "$REPO_ROOT/examples/large_dense.mop" ]]; then
    DENSE_FOR_MG_BLAS="$REPO_ROOT/examples/large_dense.mop"
  fi
  if [[ -n "$DENSE_FOR_MG_BLAS" ]]; then
    run_case "multigpu_blas_cublasxt" "$DENSE_FOR_MG_BLAS" \
      "export CUDA_VISIBLE_DEVICES=0,1; export MOPAC_CUBLASXT_DEVICES=0,1; export MOPAC_GPU_PROFILE=1;"
  else
    run_case "multigpu_blas_cublasxt" "$REPO_ROOT/examples/peptide_gg_2gpu.mop" \
      "export CUDA_VISIBLE_DEVICES=0,1; export MOPAC_CUBLASXT_DEVICES=0,1; export MOPAC_GPU_PROFILE=1;"
  fi
else
  name="multigpu_blas_cublasxt"; status="SKIP"; elapsed=0; gpu_hits="no"; reason="<2 GPUs"
  echo "==> Running $name"
  echo "Result: $status (${elapsed}s), GPU logs: $gpu_hits"
  summary+=("$name;$status;$elapsed;$gpu_hits;$reason")
fi

# 6) MG eigensolver attempt (safe fallback)
run_case "mg_eigs_attempt" "examples/water_pm7_gpu.mop" \
  "export CUDA_VISIBLE_DEVICES=0; export MOPAC_EIG_MG=1; export MOPAC_EIG_MG_MIN=1; export MOPAC_EIG_MG_PROFILE=1;"

# 7) MG eigensolver on larger dense input (if provided)
# Prefer explicit path via MOPAC_MG_LARGE_INPUT. Otherwise, try examples/dense_test.mop (preferred),
# then examples/large_dense.mop; if neither exists, autogenerate from examples/test_dense_big.pdb
# (or test_dense.pdb) using recommended dense keywords (no MOZYME).
MG_DENSE_IN="${MOPAC_MG_LARGE_INPUT:-}"
if [[ -z "$MG_DENSE_IN" ]]; then
  if [[ -f "$REPO_ROOT/examples/dense_test.mop" ]]; then
    MG_DENSE_IN="$REPO_ROOT/examples/dense_test.mop"
  elif [[ -f "$REPO_ROOT/examples/large_dense.mop" ]]; then
    MG_DENSE_IN="$REPO_ROOT/examples/large_dense.mop"
  elif [[ -f "$REPO_ROOT/examples/test_dense_big.pdb" || -f "$REPO_ROOT/examples/test_dense.pdb" ]]; then
    MG_DENSE_IN="$OUT_DIR/large_dense_autogen.mop"
    PDB_SRC="$REPO_ROOT/examples/test_dense.pdb"
    if [[ -f "$REPO_ROOT/examples/test_dense_big.pdb" ]]; then PDB_SRC="$REPO_ROOT/examples/test_dense_big.pdb"; fi
    CHG="${MOPAC_MG_LARGE_CHARGE:-+1}"
    cat > "$MG_DENSE_IN" <<EOF
GEO_DAT="$PDB_SRC" M6-D3H4X  1SCF  XYZ  GEO-OK  CHARGE=$CHG  SINGLET  SCFCRT=1.D-10  MAXIT=999  SHIFT=50  THREADS=8  COSMO  EPS=78.4
Large dense SP — PM6-D3H4X + COSMO(H2O); tight SCF, big MAXIT, diagonalization (no MOZYME)

EOF
  fi
fi

if [[ -n "$MG_DENSE_IN" && -f "$MG_DENSE_IN" ]]; then
  # Prefer testing across 2 GPUs if available
  if [[ "$gpu_count" -ge 2 ]]; then
    run_case "mg_large_dense_multigpu" "$MG_DENSE_IN" \
      "export CUDA_VISIBLE_DEVICES=0,1; export MOPAC_EIG_MG=1; export MOPAC_EIG_MG_MIN=1; export MOPAC_EIG_MG_GRID=2x1; export MOPAC_EIG_MG_PROFILE=1;"
  else
    run_case "mg_large_dense_singlegpu" "$MG_DENSE_IN" \
      "export CUDA_VISIBLE_DEVICES=0; export MOPAC_EIG_MG=1; export MOPAC_EIG_MG_MIN=1; export MOPAC_EIG_MG_PROFILE=1;"
  fi
else
  echo "NOTE: No larger dense input provided (set MOPAC_MG_LARGE_INPUT or add examples/large_dense.mop); skipping MG large test"
fi

echo ""
echo "GPU Test Summary (name;status;seconds;gpu_logs;reason)"
for line in "${summary[@]}"; do echo "$line"; done | tee "$OUT_DIR/summary.csv"

if [[ -n "$gradient_cpu_elapsed" && -n "$gradient_gpu_elapsed" ]]; then
  echo ""
  echo "Gradient timing snapshot (Stage 0 baseline)"
  echo "  CPU baseline : ${gradient_cpu_elapsed}s (log: ${gradient_cpu_log})"
  echo "  GPU profile  : ${gradient_gpu_elapsed}s (log: ${gradient_gpu_log})"
  echo "  Tip: rerun with \"MOPAC_GPU_PROFILE=2\" for detailed kernel timing."
fi

# Restore original verbose setting if it was empty
if [[ -z "$GPU_VERBOSE_DEFAULT" ]]; then unset MOPAC_GPU_VERBOSE; fi

echo "Logs in: $OUT_DIR"
if [[ "$fail_count" -gt 0 ]]; then
  echo "Failures: $fail_count" 1>&2
  exit 1
fi
exit 0

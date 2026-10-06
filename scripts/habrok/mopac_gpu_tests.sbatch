#!/bin/bash
#SBATCH --job-name=mopac_gpu_tests
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/%u/mopac_gpu_runs/slurm-%j.out
#
# MOPAC MOZYME GPU: build and full test on Habrok (University of Groningen).  Everything under /scratch/$USER.
#
#   1. on a login node (internet):
#        mkdir -p /scratch/$USER/mopac_gpu_runs && cd /scratch/$USER
#        git clone https://github.com/juvenalyosa/mopac_gpu.git      # or: cd mopac_gpu && git pull
#   2. submit:
#        sbatch /scratch/$USER/mopac_gpu/scripts/habrok/mopac_gpu_tests.sbatch
#      V100 instead of A100:   sbatch --gpus-per-node=v100:1 .../mopac_gpu_tests.sbatch
#      also the full crambin FORCE (~45-90 min more):  sbatch --export=ALL,RUN_LARGE=1 ...
#      also an Nsight Systems kernel profile:           sbatch --export=ALL,RUN_NSYS=1 ...
#   3. read /scratch/$USER/mopac_gpu_runs/<date>_job<id>/REPORT.txt  (and results_<id>.tar.gz to send back)
#
# Optional overrides (sbatch --export=ALL,VAR=value): SRC (source tree, default /scratch/$USER/mopac_gpu),
# MODULES (space-separated module list to load instead of the automatic choice), CUDA_ARCHS (default native),
# CONCURRENT (crambin processes at once in the memory test, default from the GPU memory), MOPAC_BIN (use this
# mopac executable, no build), QUICK=1 (short version of every step, ~10 min: checks the pipeline first).
#
# Steps (each logged under the run directory; PASS / FAIL lines and the key numbers go to REPORT.txt):
#   0 environment   node, GPU, driver, modules, compilers, git commit
#   1 build         cmake -DGPU=ON, mopac target
#   2 smoke         chignolin 1SCF, GPU vs CPU (same binary, MOPAC_NOGPU=1)
#   3 reference     tests/check_mozyme_gpu_reference.py: SCF + gradient (chignolin, crambin), chignolin FORCE THERMO
#                   (1 and 4 GPU processes), crambin FORCETS, GPU memory peak, concurrent processes - all against
#                   stored CPU references
#   4 staged A/B    tests/check_mozyme_diagg2_staged.py: default vs MOPAC_MOZYME_DIAGG2_STAGED=1 diagg2 kernel
#   5 reference     step 3 again with the staged kernel (timings side by side in the report)
#   6 trace         chignolin and crambin SCF per iteration and per diagg2 sweep (MOPAC_MOZYME_SCF_TRACE=1)
#   7 (RUN_NSYS=1)  nsys kernel summary of a crambin SCF, both kernels
#   8 (RUN_LARGE=1) full crambin FORCE THERMO with 6 processes, both kernels

set -u
STAMP=$(date +%Y%m%d_%H%M%S)
JOB=${SLURM_JOB_ID:-local}
SCRATCH_BASE=${SCRATCH_BASE:-/scratch/$USER}
SRC=${SRC:-$SCRATCH_BASE/mopac_gpu}
RUN=$SCRATCH_BASE/mopac_gpu_runs/${STAMP}_job${JOB}
BUILD=$RUN/build
REPORT=$RUN/REPORT.txt
DATA=$SRC/tests/data/mozyme_gpu_reference
NCPU=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$RUN" "$RUN/logs"
cd "$RUN" || exit 1

# ---------------------------------------------------------------- helpers
say() { echo "$*" | tee -a "$REPORT"; }
section() { say ""; say "================================================================================"; say "$*"; say "================================================================================"; }
# step NAME LOG cmd...: run, keep the full log, copy PASS/FAIL and [bracketed] result lines to the report
step() {
  local name=$1 log=$RUN/logs/$2; shift 2
  section "STEP $name   ($(date '+%F %T'))"
  say "\$ $*"
  local t0=$SECONDS
  "$@" > "$log" 2>&1
  local rc=$?
  local dt=$((SECONDS - t0))
  grep -E "^(PASS|FAIL) |^\[|^ {5}|all checks passed|check\(s\) failed|^work dir" "$log" >> "$REPORT"
  if [ $rc -ne 0 ]; then
    # what the PASS/FAIL lines do not show: errors, tracebacks, the last lines of a crash
    local extra; extra=$(grep -v -E "^(PASS|FAIL) |^\[|^ {5}|all checks passed|check\(s\) failed|^work dir" "$log" | tail -25)
    if [ -n "$extra" ]; then say "--- other output (last 25 lines) of logs/$(basename "$log"):"; echo "$extra" >> "$REPORT"; fi
  fi
  say "STEP $name: exit $rc, ${dt} s, log logs/$(basename "$log")"
  STEP_RESULTS+=("$name: exit $rc (${dt} s)")
  return $rc
}
STEP_RESULTS=()

finish() {
  section "SUMMARY"
  for r in ${STEP_RESULTS[@]+"${STEP_RESULTS[@]}"}; do say "  $r"; done
  say "  PASS lines: $(grep -c '^PASS ' "$REPORT")   FAIL lines: $(grep -c '^FAIL ' "$REPORT")"
  grep '^FAIL ' "$REPORT" | sed 's/^/  /' >> "$REPORT.fails"
  if [ -s "$REPORT.fails" ]; then say "  failed checks:"; tee -a "$REPORT" < "$REPORT.fails"; fi
  rm -f "$REPORT.fails"
  if [ -n "${MPS_STARTED:-}" ]; then echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; fi
  # results archive without the large restart / density / aux / profile files
  tar -czf "$SCRATCH_BASE/mopac_gpu_runs/results_${JOB}.tar.gz" -C "$SCRATCH_BASE/mopac_gpu_runs" \
      --exclude='*.res' --exclude='*.den' --exclude='*.aux' --exclude='*.nsys-rep' --exclude='*.sqlite' \
      --exclude='build' "$(basename "$RUN")" 2>/dev/null
  say ""
  say "report:  $REPORT"
  say "archive: $SCRATCH_BASE/mopac_gpu_runs/results_${JOB}.tar.gz"
}
trap finish EXIT

# ---------------------------------------------------------------- 0 environment
section "MOPAC MOZYME GPU tests on Habrok - job $JOB - $(date '+%F %T')"
say "run dir: $RUN"
say "source:  $SRC"
if [ ! -d "$SRC/.git" ]; then
  say "source tree missing: cloning (needs internet on this node)"
  git clone -q https://github.com/juvenalyosa/mopac_gpu.git "$SRC" >> "$RUN/logs/git.log" 2>&1 || { say "FAIL clone"; exit 1; }
fi
if [ "${NO_PULL:-0}" != 1 ]; then
  ( cd "$SRC" && timeout 60 git pull -q --ff-only >> "$RUN/logs/git.log" 2>&1 ) || say "(git pull failed: using the tree as it is)"
fi
say "commit:  $(cd "$SRC" && git log -1 --format='%h %ad %s' --date=short)"
say "dirty:   $(cd "$SRC" && git status --short | wc -l) changed files"

# modules: the newest CUDA 12.x and a GCC 12/13 toolchain with BLAS/LAPACK, CMake and Python
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1
  if [ -n "${MODULES:-}" ]; then
    module load $MODULES
  else
    pick() { module -t avail "$1" 2>&1 | grep -E "^$1/$2" | sort -V | tail -1; }
    CUDA_MOD=$(pick CUDA '12\.')
    FOSS_MOD=$(pick foss '2023a')
    [ -z "$FOSS_MOD" ] && FOSS_MOD=$(pick foss '2022b')
    GCC_TAG=GCCcore-12
    [[ $FOSS_MOD == *2023a* ]] && GCC_TAG=GCCcore-12.3.0
    [[ $FOSS_MOD == *2022b* ]] && GCC_TAG=GCCcore-12.2.0
    CMAKE_MOD=$(module -t avail CMake 2>&1 | grep "$GCC_TAG" | sort -V | tail -1)
    PY_MOD=$(module -t avail Python 2>&1 | grep -E "^Python/3\..*$GCC_TAG" | grep -v bare | sort -V | tail -1)
    NINJA_MOD=$(module -t avail Ninja 2>&1 | grep "$GCC_TAG" | sort -V | tail -1)
    for m in $FOSS_MOD $CUDA_MOD $CMAKE_MOD $PY_MOD $NINJA_MOD; do module load "$m" >> "$RUN/logs/modules.log" 2>&1 || say "module load $m failed"; done
  fi
  say "modules: $(module -t list 2>&1 | tr '\n' ' ')"
fi
say "node:    $(hostname)   cpus: $NCPU"
say "gpu:     $(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>&1 | head -1)"
say "nvcc:    $(nvcc --version 2>&1 | tail -1)"
say "gcc:     $(gcc --version 2>&1 | head -1)   gfortran: $(gfortran --version 2>&1 | head -1)"
say "cmake:   $(cmake --version 2>&1 | head -1)   python: $(python3 --version 2>&1)"
nvidia-smi > "$RUN/logs/nvidia-smi.txt" 2>&1
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_MEM=${GPU_MEM:-0}

# MPS lets the parallel FORCE processes share the GPU; optional (the tests run without it)
export CUDA_MPS_PIPE_DIRECTORY=$RUN/mps/pipe CUDA_MPS_LOG_DIRECTORY=$RUN/mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if nvidia-cuda-mps-control -d >/dev/null 2>&1; then MPS_STARTED=1; say "MPS:     started"; else say "MPS:     not available (processes time-slice the GPU)"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; fi

# ---------------------------------------------------------------- 1 build
build() {
  local gen=()
  command -v ninja >/dev/null 2>&1 && gen=(-G Ninja)
  local arch=${CUDA_ARCHS:-native}
  if [ "$arch" = native ]; then
    # the compute capability of this node's GPU (cmake's 'native' needs CMake >= 3.24)
    local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    [ -n "$cc" ] && arch=$cc
  fi
  echo "CUDA_ARCHS=$arch"
  local opts=(-DGPU=ON -DTESTS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCUDA_ARCHS="$arch")
  # EasyBuild toolchains (foss) provide BLAS/LAPACK through FlexiBLAS
  if [ -n "${EBROOTFLEXIBLAS:-}" ] && \
     cmake -S "$SRC" -B "$BUILD" ${gen[@]+"${gen[@]}"} "${opts[@]}" -DBLA_VENDOR=FlexiBLAS; then
    :
  else
    rm -rf "$BUILD"
    cmake -S "$SRC" -B "$BUILD" ${gen[@]+"${gen[@]}"} "${opts[@]}" || return 1
  fi
  cmake --build "$BUILD" --target mopac --parallel "$NCPU" || return 1
  test -x "$BUILD/mopac"
}
if [ -n "${MOPAC_BIN:-}" ]; then
  say "build skipped: MOPAC_BIN=$MOPAC_BIN"
  MOPAC=$MOPAC_BIN
else
  if ! step "1 build" build.log build; then
    say "--- compiler errors:"
    grep -i -E "error|Error" "$RUN/logs/build.log" | head -60 >> "$REPORT"
    exit 1
  fi
  MOPAC=$BUILD/mopac
fi
say "mopac:   $MOPAC"

# ---------------------------------------------------------------- 2 smoke
smoke() {
  local d=$RUN/smoke; mkdir -p "$d/gpu" "$d/cpu"
  for m in gpu cpu; do
    cp "$DATA/1UAO_opt.pdb" "$d/$m/"
    printf 'PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK 1SCF GEO_DAT="1UAO_opt.pdb"\nsmoke\n\n' > "$d/$m/s.mop"
  done
  ( cd "$d/gpu" && "$MOPAC" s.mop > s.stdout 2>&1 )
  ( cd "$d/cpu" && MOPAC_NOGPU=1 "$MOPAC" s.mop > s.stdout 2>&1 )
  local hg hc st
  hg=$(grep "FINAL HEAT" "$d/gpu/s.out" | awk '{print $6}')
  hc=$(grep "FINAL HEAT" "$d/cpu/s.out" | awk '{print $6}')
  st=$(grep -c "status=success reason" "$d/gpu/s.out")
  echo "[smoke] chignolin 1SCF: GPU heat $hg ($st resident SCF), CPU heat $hc"
  python3 - "$hg" "$hc" "$st" <<'EOF'
import sys
hg, hc, st = sys.argv[1], sys.argv[2], int(sys.argv[3] or 0)
ok = bool(hg) and bool(hc) and abs(float(hg) - float(hc)) <= 1e-3 and st > 0
print(("PASS " if ok else "FAIL ") + f"smoke: GPU - CPU heat {float(hg or 'nan') - float(hc or 'nan'):+.5f} kcal/mol, GPU SCF resident {st}")
sys.exit(0 if ok else 1)
EOF
}
step "2 smoke" smoke.log smoke || say "(smoke failed: the remaining steps still run, read their logs)"

# ---------------------------------------------------------------- 3-5 tests
if [ -z "${CONCURRENT:-}" ]; then
  if [ "$GPU_MEM" -ge 32000 ]; then CONCURRENT=6; else CONCURRENT=3; fi
fi
if [ "${QUICK:-0}" = 1 ]; then
  say "QUICK=1: no FORCE / FORCETS, one repeat"
  REF_OPTS=(--workers 0 --concurrent 0 --skip-forcets)
  REF_STAGED_OPTS=(--workers 0 --concurrent 0 --skip-forcets)
  AB_OPTS=(--repeats 1 --forcets "")
else
  REF_OPTS=(--workers 4 --sequential-force --concurrent "$CONCURRENT")
  REF_STAGED_OPTS=(--workers 4 --sequential-force --concurrent 0)
  AB_OPTS=(--repeats 2 --forcets "A10=1")
fi
step "3 reference (default diagg2)" reference_default.log \
  python3 "$SRC/tests/check_mozyme_gpu_reference.py" "$MOPAC" --data "$DATA" --work-dir "$RUN/reference_default" \
  "${REF_OPTS[@]}"

step "4 staged diagg2 A/B" staged_ab.log \
  python3 "$SRC/tests/check_mozyme_diagg2_staged.py" "$MOPAC" --work-dir "$RUN/staged_ab" "${AB_OPTS[@]}" \
  --geometry "1UAO=$DATA/1UAO_opt.pdb" --geometry "1CRN=$DATA/1CRN_opt.pdb"

step "5 reference (staged diagg2)" reference_staged.log \
  env MOPAC_MOZYME_DIAGG2_STAGED=1 python3 "$SRC/tests/check_mozyme_gpu_reference.py" "$MOPAC" --data "$DATA" \
  --work-dir "$RUN/reference_staged" "${REF_STAGED_OPTS[@]}"

# timings side by side
section "TIMINGS default vs staged diagg2 (seconds; from summary.json)"
python3 - "$RUN/reference_default/summary.json" "$RUN/reference_staged/summary.json" >> "$REPORT" 2>&1 <<'EOF'
import json, sys
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}
d, s = load(sys.argv[1]), load(sys.argv[2])
print(f"{'case':38s} {'default':>10s} {'staged':>10s} {'speed-up':>9s}")
for k in d:
    if isinstance(d[k], dict) and "wall" in d[k]:
        a = d[k]["wall"]; b = s.get(k, {}).get("wall") if isinstance(s.get(k), dict) else None
        print(f"{k:38s} {a:10.1f} {b if b is None else round(b, 1)!s:>10s} " + (f"{a / b:8.2f}x" if b else ""))
EOF

# ---------------------------------------------------------------- 6 trace
trace() {
  for sys_id in 1UAO 1CRN; do
    for mode in default staged; do
      local d=$RUN/trace/${sys_id}_$mode; mkdir -p "$d"; cp "$DATA/${sys_id}_opt.pdb" "$d/"
      printf 'PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK 1SCF LET SCFCRT=0.000001 THRESH=1.D-15 GEO_DAT="%s_opt.pdb"\ntrace\n\n' "$sys_id" > "$d/t.mop"
      if [ $mode = staged ]; then
        ( cd "$d" && MOPAC_MOZYME_SCF_TRACE=1 MOPAC_MOZYME_DIAGG2_STAGED=1 MOPAC_MOZYME_SECTION_PROFILE=1 "$MOPAC" t.mop > t.stdout 2>&1 )
      else
        ( cd "$d" && MOPAC_MOZYME_SCF_TRACE=1 MOPAC_MOZYME_SECTION_PROFILE=1 "$MOPAC" t.mop > t.stdout 2>&1 )
      fi
      local it sweeps chain spins
      it=$(grep -c "SCF trace\] iter=" "$d/t.stdout")
      sweeps=$(grep -c "diagg2 staged=" "$d/t.stdout")
      chain=$(grep "diagg2 staged=" "$d/t.stdout" | sed 's/.*max_chain=\([0-9]*\).*/\1/' | sort -n | tail -1)
      spins=$(grep "diagg2 staged=" "$d/t.stdout" | sed 's/.*lock_spins=\([0-9]*\).*/\1/' | awk '{s+=$1} END{print s+0}')
      echo "[trace] $sys_id $mode: $it iterations, $sweeps diagg2 sweeps, max_chain ${chain:-?}, lock_spins total ${spins:-?}, $(grep 'TOTAL JOB TIME' "$d/t.out" | awk '{print $4}') s, heat $(grep 'FINAL HEAT' "$d/t.out" | awk '{print $6}')"
      grep "MOZYME_SECTION name=resident_run_total\|name=compfg_deriv \|name=run_total" "$d/t.stdout" | sort -u | sed "s/^/     $sys_id $mode /"
    done
  done
}
step "6 SCF / diagg2 trace" trace.log trace

# ---------------------------------------------------------------- 7 nsys (optional)
if [ "${RUN_NSYS:-0}" = 1 ]; then
  nsys_prof() {
    command -v nsys >/dev/null 2>&1 || module load Nsight-Systems >/dev/null 2>&1 || true
    command -v nsys >/dev/null 2>&1 || { echo "FAIL nsys not found"; return 1; }
    for mode in default staged; do
      local d=$RUN/nsys/$mode; mkdir -p "$d"; cp "$DATA/1CRN_opt.pdb" "$d/"
      printf 'PM7 MOZYME MOZYME_MINBLK=16 PULAY SHIFT=-50 ITRY=200 GEO-OK 1SCF LET SCFCRT=0.000001 THRESH=1.D-15 GEO_DAT="1CRN_opt.pdb"\nnsys\n\n' > "$d/s.mop"
      local envs=""; [ $mode = staged ] && envs="MOPAC_MOZYME_DIAGG2_STAGED=1"
      ( cd "$d" && env $envs nsys profile -o crambin -f true --stats=false "$MOPAC" s.mop > nsys.log 2>&1 )
      echo "[nsys] crambin 1SCF $mode: kernel summary (top 12)"
      ( cd "$d" && nsys stats -r cuda_gpu_kern_sum crambin.nsys-rep 2>/dev/null | sed -n '1,20p' | cut -c1-160 | sed 's/^/     /' )
    done
  }
  step "7 nsys profile" nsys.log nsys_prof
fi

# ---------------------------------------------------------------- 8 full crambin FORCE (optional)
if [ "${RUN_LARGE:-0}" = 1 ]; then
  step "8a crambin FORCE (default diagg2)" crambin_force_default.log \
    python3 "$SRC/tests/check_mozyme_gpu_reference.py" "$MOPAC" --data "$DATA" --work-dir "$RUN/crambin_default" \
    --workers 0 --concurrent 0 --skip-forcets --large-workers 6
  step "8b crambin FORCE (staged diagg2)" crambin_force_staged.log \
    env MOPAC_MOZYME_DIAGG2_STAGED=1 python3 "$SRC/tests/check_mozyme_gpu_reference.py" "$MOPAC" --data "$DATA" \
    --work-dir "$RUN/crambin_staged" --workers 0 --concurrent 0 --skip-forcets --large-workers 6
fi

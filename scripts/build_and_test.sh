#!/usr/bin/env bash
set -euo pipefail

# Build MOPAC (CPU or GPU) and run a quick test.
#
# Usage examples:
#  - GPU (P4): ./scripts/build_and_test.sh --gpu on --arch 61 \
#              --pdb modeljuv.B99990166_withH_rep0_step5000.pdb
#  - CPU only: ./scripts/build_and_test.sh --gpu off
#  - Custom .mop: ./scripts/build_and_test.sh --gpu on --mop examples/water_pm7_gpu.mop
#
# Options
#   --gpu on|off          Enable GPU build (default: on)
#   --arch CCLIST         CMAKE_CUDA_ARCHITECTURES (default: 61;70;75;80;86)
#   --build-dir DIR       Build directory (default: build)
#   --type Release|Debug  Build type (default: Release)
#   --mop FILE            Use an existing .mop input file
#   --pdb FILE            Build a temporary .mop with GEO_DAT=FILE
#   --fastgpu             Enable keep-on-GPU eigensolve/density
#   --partial             Enable partial eigensolve (RHF) for occupied subspace
#   --purify              Enable diagonalization-free purification (RHF)
#   --fock-gpu            Enable experimental GPU Fock build
#   --diis-gpu            Enable DIIS GPU residuals/B/solve
#   --no-streams          Disable CUDA streams

GPU=on
ARCH="61;70;75;80;86"
BUILDDIR=build
TYPE=Release
MOPFILE=""
PDBFILE=""
FASTGPU=0
PARTIAL=0
PURIFY=0
FOCKGPU=0
DIISGPU=0
NOSTREAMS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --arch) ARCH="$2"; shift 2;;
    --build-dir) BUILDDIR="$2"; shift 2;;
    --type) TYPE="$2"; shift 2;;
    --mop) MOPFILE="$2"; shift 2;;
    --pdb) PDBFILE="$2"; shift 2;;
    --fastgpu) FASTGPU=1; shift;;
    --partial) PARTIAL=1; shift;;
    --purify) PURIFY=1; shift;;
    --fock-gpu) FOCKGPU=1; shift;;
    --diis-gpu) DIISGPU=1; shift;;
    --no-streams) NOSTREAMS=1; shift;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

echo "[build_and_test] GPU=${GPU} ARCH=${ARCH} BUILDDIR=${BUILDDIR} TYPE=${TYPE}"

cmake -S . -B "${BUILDDIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${TYPE}" \
  -DGPU=$( [[ "${GPU}" == on ]] && echo ON || echo OFF ) \
  $( [[ "${GPU}" == on ]] && echo -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" )

cmake --build "${BUILDDIR}" -j

BIN="${BUILDDIR}/mopac"
if [[ ! -x "${BIN}" ]]; then
  echo "[build_and_test] error: built binary not found: ${BIN}" >&2
  exit 1
fi

# GPU runtime toggles
if [[ "${GPU}" == on ]]; then
  export MOPAC_FORCEGPU=1
  export MOPAC_DETERMINISTIC=1
  [[ ${FASTGPU} -eq 1 ]] && export MOPAC_FASTGPU=1
  [[ ${PARTIAL} -eq 1 ]] && export MOPAC_PARTIAL_EIG=1
  if [[ ${PURIFY} -eq 1 ]]; then
    export MOPAC_PURIFY=1
    export MOPAC_PURIFY_GPU=1
  fi
  if [[ ${DIISGPU} -eq 1 ]]; then
    export MOPAC_DIIS_GEN=1
    export MOPAC_DIIS_GPU_BUF=1
    export MOPAC_DIIS_GPU_BFULL=1
    export MOPAC_DIIS_GPU=1
  fi
  [[ ${FOCKGPU} -eq 1 ]] && export MOPAC_FOCK_GPU=1
  [[ ${NOSTREAMS} -eq 1 ]] && export MOPAC_STREAMS=off
fi

# Prepare input
TMPDIR=$(mktemp -d -t mopac_test_XXXXXX)
TESTMOP="${TMPDIR}/test.mop"

if [[ -n "${MOPFILE}" ]]; then
  TESTMOP="${MOPFILE}"
elif [[ -n "${PDBFILE}" ]]; then
  # Create a minimal PM7 input referencing the PDB via GEO_DAT
  {
    echo "PM7 GEO_DAT=${PDBFILE} 1SCF EIGS VECTORS"
    echo "GPU test with GEO_DAT"
    echo
  } > "${TESTMOP}"
else
  # Default example
  if [[ -f examples/water_pm7_gpu.mop ]]; then
    TESTMOP="examples/water_pm7_gpu.mop"
  else
    {
      echo "PM7 1SCF EIGS VECTORS"
      echo "GPU test H2O"
      echo
      echo "O   0.000000  0.000000  0.000000"
      echo "H   0.000000  0.757160  0.586260"
      echo "H   0.000000 -0.757160  0.586260"
    } > "${TESTMOP}"
  fi
fi

echo "[build_and_test] Running: ${BIN} ${TESTMOP}"
"${BIN}" "${TESTMOP}" || {
  echo "[build_and_test] run failed" >&2
  exit 1
}

echo "[build_and_test] Done. Outputs are next to input file (.out/.arc)."


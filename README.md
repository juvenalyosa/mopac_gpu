# Molecular Orbital PACkage (MOPAC)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/177640376.svg)](https://zenodo.org/badge/latestdoi/177640376)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/mopac/badges/version.svg)](https://anaconda.org/conda-forge/mopac)
![build](https://github.com/openmopac/mopac/actions/workflows/CI.yaml/badge.svg)
[![codecov](https://codecov.io/gh/openmopac/mopac/branch/main/graph/badge.svg?token=qM2KeRvw06)](https://codecov.io/gh/openmopac/mopac)

This is the official repository of the modern open-source version of MOPAC, which is now released under an Apache license
(versions 22.0.0 through 23.0.3 are available under an LGPL license).
This is a direct continuation of the commercial development and distribution of MOPAC, which ended at MOPAC 2016.
Commercial versions of MOPAC are no longer supported, and all MOPAC users are encouraged to switch to the most recent open-source version.

[![mopac_at_molssi](.github/mopac_at_molssi.png)](https://molssi.org)

MOPAC is actively maintained and curated by the [Molecular Sciences Software Institute (MolSSI)](https://molssi.org).

## Quick Start (Ubuntu + GPU)

1) Install prerequisites
- NVIDIA driver + CUDA Toolkit 11.2+ (verify `nvidia-smi` and `nvcc --version`)
- Build tools and BLAS/LAPACK:
  - `sudo apt-get install -y build-essential cmake gfortran ninja-build libopenblas-dev liblapack-dev`

2) Configure and build (one build for everything)
- `export CUDAToolkit_ROOT=/usr/local/cuda`   # adjust if needed
- `cmake -S . -B build -G Ninja -DGPU=ON -DCMAKE_BUILD_TYPE=Release \\
         -DCMAKE_CUDA_ARCHITECTURES=70;80;86`  # match your GPU(s)
- `cmake --build build --parallel`

3) Run a calculation (manual, no scripts)
- CPU run (explicitly disable GPU):
  - Add `NOGPU` to the first keyword line of your `.mop` input
  - Run: `./build/mopac path/to/your_input.mop`
- GPU run (recommended for larger systems):
  - Keep your input unchanged (no `NOGPU`)
  - Export: `export MOPAC_FORCEGPU=1`  # enable GPU if available
  - Optional: `export MOPAC_FASTGPU=1` # faster SCF by keeping data on GPU
  - Run: `./build/mopac path/to/your_input.mop`

Examples
- Minimal water input (PM7): `examples/water_pm7_gpu.mop`
  - Prints eigenvectors (includes `EIGS VECTORS`).
  - GPU (recommended on larger systems):
    - `export MOPAC_FORCEGPU=1`
    - `./build/mopac examples/water_pm7_gpu.mop`
  - CPU only:
    - Edit the first line to prepend the keyword `NOGPU`, e.g., `NOGPU PM7 1SCF EIGS VECTORS`
    - `./build/mopac examples/water_pm7_gpu.mop`

Useful runtime toggles (set only if needed)
- `MOPAC_GPU_EIGEN_MIN=400`   # GPU eigensolve cutoff in AOs (default 400)
- `MOPAC_EIG2HOST=1`          # copy eigenvectors back to host immediately
- `MOPAC_PIN_USER=1`          # pin user arrays to reduce extra host memcpy
- `MOPAC_STREAMS=off`         # disable CUDA streams (debugging)
- `MOPAC_DETERMINISTIC=1`     # enforce deterministic cuBLAS settings (no atomics, host pointer mode)

Printing eigenvectors
- Add `EIGS VECTORS` to the keyword line; if eigenvectors were kept on the GPU, MOPAC fetches them automatically before printing.

## Installation

Open-source MOPAC is available through multiple distributon channels, and it can also be compiled from source using CMake.
In addition to continuing the distribution of self-contained installers on the
[old commercial website](http://openmopac.net/Download_MOPAC_Executable_Step2.html) and here on GitHub,
MOPAC can also be installed using multiple package managers and accessed through containers.

### Self-contained installers

Self-contained graphical installers for Linux, Mac, and Windows are available on GitHub for each release,
which are constructed using the [Qt Installer Framework](https://doc.qt.io/qtinstallerframework/).

While the installers are meant to be run from a desktop environment by default, they can also be run from a command line without user input.
On Linux, the basic command-line installation syntax is:

`./mopac-x.y.z-linux.run install --accept-licenses --confirm-command --root type_installation_directory_here`

For more information on command-line installation, see the [Qt Installer Framework Documentation](https://doc.qt.io/qtinstallerframework/ifw-cli.html).

Linux installations without a desktop environment may not have the shared libraries required for the graphical installers,
and there have also been isolated reports of problems with the Qt installer on other platforms. A minimal, compressed-archive installer
is available for each platform as an alternative for users that have problems with the Qt installer.

The minimum glibc version required for the precompiled version of MOPAC on Linux is currently 2.17.

#### Library path issues

The pre-built MOPAC executables use the RPATH system on Mac and Linux to connect with its shared libraries,
including the `libiomp5` Intel OpenMP redistributable library. The `libiomp5` library is not properly versioned, and the recent version used by
MOPAC is not compatible with older versions that might also exist on a user's machine. If a directory containing an old version of `libiomp5`
is in the shared library path (`LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on Mac), this will override the RPATH system, link MOPAC to the
wrong library, and cause an error in MOPAC execution. On Mac, this can be fixed by switching the offending directories to the failsafe shared library
path, `DYLD_FALLBACK_LIBRARY_PATH`. On Linux, the use of `LD_LIBRARY_PATH` is generally discouraged for widespread use, and there is no simple
workaround available. The newer version of `libiomp5` is backwards compatible, so replacing the offending version with the version used by MOPAC
should preserve the functionality of other software that depends on the library.

### Package managers

The officially supported package manager for MOPAC is the [conda-forge channel of Conda](https://anaconda.org/conda-forge/mopac).
MOPAC is also packaged by major Linux distributions including
[Fedora](https://packages.fedoraproject.org/pkgs/mopac/mopac/) and
[Debian](https://tracker.debian.org/pkg/mopac).
It is also available in the [Google Play store](https://play.google.com/store/apps/details?id=cz.m).

[![Packaging status](https://repology.org/badge/vertical-allrepos/mopac.svg?columns=2)](https://repology.org/project/mopac/versions)

### Docker/Apptainer Containers

The official [Docker](https://www.docker.com) and [Apptainer](https://apptainer.org) ([Singularity](https://sylabs.io)) containers for MOPAC 22.0.6 ([Conda version](https://anaconda.org/conda-forge/mopac)) are developed and
maintained by [MolSSI Container Hub](https://molssi.github.io/molssi-hub/index.html) and are distributed by the MolSSI Docker Hub [repository](https://hub.docker.com/r/molssi/mopac220-mamba141).

### CMake

MOPAC uses a CMake build system. For a single, unified build (CPU+GPU when available), follow the Quick Start above. If you need CPU‑only:

```
cmake -S . -B build-cpu -G Ninja -DGPU=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
```

### GPU Support (CUDA)

Experimental CUDA acceleration is available and can be enabled at configure time:

```
cmake -DGPU=ON ..
make
```

This builds CUDA wrappers for selected linear algebra routines (GEMM, SYRK) and an accelerated eigenvector rotation used in the SCF procedure. If multiple compatible NVIDIA GPUs are present, MOPAC will use up to two devices to speed up select steps.

MOZYME-specific GPU controls
- MOZYME_GPU: enable MOZYME GPU acceleration (rank-1 GEMM/SYRK in density construction).
- MOZYME_2GPU: force MOZYME density updates to use two GPUs when at least two suitable devices exist.
- MOZYME_MINBLK=INT: minimum localized block size to offload rank-1 operations (default: 16).
- MOZYME_GPUPAIR=a,b: explicitly select two 1-based GPU device IDs for MOZYME 2-GPU density (e.g., 1,2).
- MOZYME_GPUIGNORE=a,b,c: 1-based device IDs to ignore for auto-selection (applies to single and multi-GPU).

General GPU toggles
- NOGPU: disable all GPU usage.
- Environment MOPAC_FORCEGPU=1: force-enable GPU when supported (overrides small-system heuristic).

Examples
- Single GPU: `MOZYME MOZYME_GPU`
- Force two GPUs with explicit pair: `MOZYME MOZYME_GPU MOZYME_2GPU MOZYME_GPUPAIR=1,2`
- Increase offload threshold: `MOZYME MOZYME_GPU MOZYME_MINBLK=32`

Note: Optional verification helpers and CI recipes have been removed from this quick path to keep usage simple. See scripts/ and .github/ for advanced options.

### GPU Usage and Examples

Build-time options
- `-DGPU=ON`: enables CUDA wrappers and GPU-aware code paths.
- `-DAUTO_BLAS=ON`: let CMake discover BLAS/LAPACK (recommended). If OFF, set `-DMOPAC_LINK` and `-DMOPAC_LINK_PATH` manually.
- `-DENABLE_GPU_TESTS=ON`: registers GPU checks with `ctest` (requires `GPU=ON`).

Runtime environment knobs
- `MOPAC_NOGPU=1`: disable GPU paths entirely.
- `MOPAC_FORCEGPU=1`: force-enable GPU (bypasses small-system heuristic if any).
- `MOPAC_STREAMS=off` (or `0`): disable custom CUDA streams (helpful for debugging ordering). Default uses streams for overlap.
- `CUDA_VISIBLE_DEVICES=...`: standard CUDA device masking (e.g., `0` or `0,1`).

MOZYME-specific GPU keywords (in the MOPAC keyword line)
- `MOZYME_2GPU`: use two GPUs for MOZYME density rank‑1 updates when available.
- `MOZYME_MINBLK=INT`: minimum localized block size to offload (default 16).
- `MOZYME_GPUPAIR=a,b`: explicit 1‑based GPU IDs (e.g., `1,2`).

Notes
- 1‑GPU SCF and MOZYME paths offload dense BLAS and rotations using cuBLAS/cuSOLVER.
- 2‑GPU MOZYME density uses a row‑sliced outer‑product implementation; device pair defaults to `0,1` or can be set via `MOZYME_GPUPAIR`.
- Internally, MOPAC uses grow‑only device and pinned‑host caches to avoid repeated allocations and to overlap copies with compute.
 - Experimental (phase 2): GPU orthogonalization helpers (Cholesky + TRSM) are available via new interfaces for future fully GPU‑resident SCF wiring.

Common build recipes
- CPU only, auto BLAS:
  - `cmake -S . -B build-cpu -DAUTO_BLAS=ON`
  - `cmake --build build-cpu -j`
- GPU build (CUDA on PATH), auto BLAS:
  - `cmake -S . -B build-gpu -DGPU=ON -DAUTO_BLAS=ON`
  - `cmake --build build-gpu -j`

Quick verification executables (when `GPU=ON`)
- 1‑GPU rotation check: `./build-gpu/mopac-gpu-rot-verify`
- 2‑GPU rotation check: `CUDA_VISIBLE_DEVICES=0,1 ./build-gpu/mopac-gpu-rot-2gpu-verify`
- Density check: `./build-gpu/mopac-gpu-density-verify`
- SCF (density + Fock) compare: `./build-gpu/mopac-gpu-scf-compare`

Benchmark tool (with CLI flags)
- Build target: `./build-gpu/mopac-gpu-bench`
- Default run prints first‑call vs cached timings and GFLOP/s for GEMM/SYRK:
  - `./build-gpu/mopac-gpu-bench`
- Custom sizes/iterations and options:
  - `./build-gpu/mopac-gpu-bench --gemm=2048,2048,128,10 --syrk=2048,128,10 --syrk-full --dsyevd=1024,3 --rot1=2048,5 --rot2=4096,5`

End‑to‑end examples and benchmarking tools are available in `scripts/` and `tests/`, but are intentionally omitted here to keep usage simple. Refer to those directories if you need automated comparisons or performance studies.

Expected correctness
- The GPU paths are designed to match the CPU numerics to double‑precision round‑off. Typical diffs:
  - Densities/Fock: 0 or ~1e‑15
  - Rotations/Eigenvalues: ~1e‑15 to 1e‑14
  - If diffs exceed ~1e‑12 consistently, please open an issue with input and environment details.

## Enhanced GPU SCF (cuSOLVER + Reduced Transfers)

This release adds a high‑performance GPU SCF path that keeps heavy linear‑algebra on the device and reduces PCIe transfers while preserving double‑precision accuracy.

What’s new
- Exact eigensolve on GPU: SCF uses cuSOLVER `Dsyevd` for symmetric eigenvalue problems in double precision.
- Size‑aware routing: A configurable cutoff avoids GPU overhead on small matrices; CPU LAPACK is used below the threshold.
- Keep‑on‑GPU mode: Optionally keep eigenvectors on the GPU after diagonalization and form the density on device (cuBLAS `Dsyrk`/`Dgemm`), copying only the density back.
- Smarter H2D/D2H: Skip copying `C` when `beta=0` and optionally pin user arrays to reduce extra memcpy.
- Auto‑fetch for printing: When `EIGS VECTORS` is requested, MOPAC automatically fetches eigenvectors to host for printing even if they were kept on GPU.

Build on Ubuntu (GPU)
1) Install dependencies
   - NVIDIA drivers + CUDA Toolkit 11.2+ (verify `nvidia-smi` and `nvcc --version`)
   - `sudo apt-get install -y build-essential cmake gfortran ninja-build libopenblas-dev liblapack-dev`
2) Configure and build
   - `export CUDAToolkit_ROOT=/usr/local/cuda`            # adjust if needed
   - `cmake -S . -B build -G Ninja -DGPU=ON -DCMAKE_BUILD_TYPE=Release \\
            -DCMAKE_CUDA_ARCHITECTURES=70;80;86`         # match your GPUs
   - `cmake --build build --parallel`
3) Binaries: `build/mopac`, `build/mopac-param`

Runtime controls (environment)
- `MOPAC_FORCEGPU=1`            Force‑enable GPU if any suitable device exists.
- `MOPAC_GPU_EIGEN_MIN=400`     Size cutoff (AOs) for GPU eigensolve; smaller sizes stay on CPU. Default: 400.
- `MOPAC_FASTGPU=1`             Keep eigenvectors on the GPU; build density on device; copy back only the density.
- `MOPAC_EIG2HOST=1`            Also fetch eigenvectors to host immediately after GPU diagonalization (optional).
- `MOPAC_PIN_USER=1`            Pin user arrays to cut extra host memcpy (falls back safely if unsupported).
- `MOPAC_STREAMS=off`           Disable CUDA streams (debug/diagnostics).
- `MOPAC_NOGPU=1`               Disable all GPU functionality.
- `MOPAC_ORTHO_GPU=1`           Experimental: orthogonalize F with S on GPU (Cholesky + TRSM) before eigensolve.
- `MOPAC_DIIS_GEN=1`            Experimental: use generalized Pulay residual R = F P S − S P F (GPU-assisted).
- `MOPAC_DIIS_GPU=1`            Experimental: solve DIIS linear system on GPU (cuSOLVER small dense solve).

Typical usage
- Large system, fast SCF:
  - `export MOPAC_FORCEGPU=1`
  - `export MOPAC_FASTGPU=1`
  - Optional tuning: `export MOPAC_GPU_EIGEN_MIN=600`
  - Optional: `export MOPAC_PIN_USER=1`
  - Run: `./build/mopac examples/your_input.mop`
- Print eigenvectors on fast path:
  - Add `EIGS VECTORS` to the keyword line; MOPAC auto‑fetches eigenvectors if they were kept on GPU.
  - Or set `MOPAC_EIG2HOST=1` to fetch right after eigensolve.
- Disable GPU (for comparison): `MOPAC_NOGPU=1 ./build/mopac examples/your_input.mop`

Performance guidance
- Small (< 300–400 AOs): CPU LAPACK/BLAS is often as fast due to GPU overheads. The cutoff avoids regressions.
- Medium (≈ 500–1500 AOs): 1.5–4× faster SCF with cuSOLVER + cuBLAS, especially with `MOPAC_FASTGPU=1`.
- Large (≥ 2000 AOs): 3–8× faster SCF; benefits grow with basis size; ensure sufficient VRAM (see below).
- Gradients: When ported selectively to cuBLAS (symmetric GEMM/SYRK contractions) expect 1.5–3× on large systems.

VRAM sizing (double precision)
- One dense `n×n` matrix ≈ `8·n²` bytes.
- Typical SCF peak (eigensolve + density) ≈ `40–56·n²` bytes per GPU (includes cuSOLVER workspace and caches).
- Examples (per device):
  - n=2000: one matrix ~32 MB; SCF peak ~160–225 MB; worst‑case caches ~300–500 MB.
  - n=3000: one matrix ~72 MB; peak ~360–500 MB; worst‑case ~0.7–1.2 GB.
  - n=5000: one matrix ~200 MB; peak ~1.0–1.4 GB; worst‑case ~2–3 GB.
- 2 GPUs halve per‑device footprint for certain outer‑products; eigensolve remains single‑GPU.

Accuracy and reproducibility
- All GPU math runs in double precision (cuBLAS D* and cuSOLVER `Dsyevd`).
- Results match CPU within ulp‑level differences; eigenvectors can differ by phase/sign (expected behavior).
- SCF tolerances and convergence criteria are unchanged.

Troubleshooting
- CUDA not found at configure time: set `CUDAToolkit_ROOT` or ensure CUDA is on PATH/LD_LIBRARY_PATH.
- Link errors to `cusolver/cublas`: ensure `/usr/local/cuda/lib64` is in `LD_LIBRARY_PATH`.
- Architecture mismatch: set `-DCMAKE_CUDA_ARCHITECTURES` to your GPU’s compute capability.
- Streams/ordering issues: run with `MOPAC_STREAMS=off`.
- Disable GPU quickly: `MOPAC_NOGPU=1` or add `NOGPU` to the input keywords.

### Local GPU Verification Helper (MOZYME)

To quickly compare CPU vs 1‑GPU vs 2‑GPU energies on a small MOZYME case:

- Build the helper target (from your build dir):
  - `cmake --build build --target mozyme-gpu-verify`

- Or run the script manually for more control:
  - `bash scripts/test_mozyme_gpu.sh ./build/mopac tests/mozyme_h2o.mop [pair] [tol]`
  - Examples:
    - Default devices, default tolerance (`1e-4`):
      - `bash scripts/test_mozyme_gpu.sh ./build/mopac tests/mozyme_h2o.mop`
    - Force device pair `1,2` (1‑based) for the 2‑GPU check and set tolerance to `5e-4`:
      - `bash scripts/test_mozyme_gpu.sh ./build/mopac tests/mozyme_h2o.mop 1,2 5e-4`

What it does
- Makes three copies of the input and prepends the appropriate keywords: `NOGPU`, `MOZYME_GPU`, and `MOZYME_2GPU` (and `MOZYME_GPUPAIR=…` if you passed a pair).
- Runs each, then extracts the (final) heat of formation from either the `.out` file or the `.log` if no `.out` was produced.
- Prints CPU/1‑GPU/2‑GPU energies and their absolute differences against the CPU value.

Troubleshooting the helper
- If energies are blank, the script prints the temp folder name (e.g., `mozyme_h2o_testgpu.XXXXXX`). Inspect `cpu.log`, `gpu1.log`, `gpu2.log` and any `.out` files in that directory.
- The script matches either `FINAL HEAT OF FORMATION` or `HEAT OF FORMATION` and takes the last occurrence.
- “bash: …/libtinfo.so.6: no version information available”: typically harmless (conda/bash libtinfo mismatch). You can invoke the system shell explicitly: `/bin/bash scripts/test_mozyme_gpu.sh …`.
- If you use a nested build directory, pass the correct path to `mopac` (e.g., `./build/mopac`).

## Documentation

The main source for MOPAC documentation is presently its old [online user manual](http://openmopac.net/manual/index.html).

There is a [new documentation website](https://openmopac.github.io) under development, but it is not yet ready for general use.

## Interfaces

While MOPAC is primarily a self-contained command-line program whose behavior is specified by an input file, it also has other modes of
operation, some of which only require the MOPAC shared library and not the executable. Note that API calls to the MOPAC library are not
thread safe. Each thread must load its own instance of the MOPAC library, such as by running independent calling programs.

### MDI Engine

MOPAC can be compiled to run as an MDI Engine through the [MolSSI Driver Interface Library](https://github.com/MolSSI-MDI/MDI_Library)
by setting `-DMDI=ON` when running CMake. See [MDI documentation](https://molssi-mdi.github.io/MDI_Library) for more information.

### Run from library

MOPAC calculations can be run as a C-like library call to `run_mopac_from_input(path_to_file)` where `path_to_file` is a C string
containing the system path to a MOPAC input file. Alternatively, a Fortran wrapper in the `include` directory allows this to be run as
the subroutine `run_mopac_from_input_f(path_to_file)` in the `mopac_api_f` module where `path_to_file` is a Fortran string.

### Diskless/stateless API

A subset of MOPAC calculations can be run through a C-like Application Programming Interface (API) defined by the `mopac.h` C header file
in the `include` directory, which also has a Fortran wrapper for convenience to Fortran software developers. Calculations run through this API
do not use any input or output files or any other form of disk access, and the data structures of the API contain all relevant information
regarding the input and output of the MOPAC calculation. The functionality and data exposed by this API is limited and has been designed to
align with the most common observed uses of MOPAC. Future expansion of this functionality and data will be considered upon request.

## Citation

To cite the use of open-source MOPAC in scientific publications, see the `CITATION.cff` file in this repository.

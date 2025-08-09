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

## Quick Start (GPU)

- Build (GPU):
  - `cmake -S . -B build-gpu -DGPU=ON -DAUTO_BLAS=ON`
  - `cmake --build build-gpu -j`
- Try examples (in `examples/`):
  - CPU baseline: `./build-gpu/mopac examples/h2o_cpu.mop`
  - Force GPU: `MOPAC_FORCEGPU=1 ./build-gpu/mopac examples/h2o_gpu_force.mop`
  - MOZYME on 1 GPU: `./build-gpu/mopac examples/mozyme_1gpu.mop`
  - MOZYME on 2 GPUs (devices 0,1): `CUDA_VISIBLE_DEVICES=0,1 ./build-gpu/mopac examples/mozyme_2gpu_pair.mop`
- Optional debug switch: disable streams (serialize copies/compute)
  - `MOPAC_STREAMS=off ./build-gpu/mopac examples/h2o_gpu_force.mop`

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

MOPAC is now built using a CMake 3.x build system with tests orchestrated using CTest.
The minimum required CMake version is presently 3.14.

CMake performs out-of-source builds, with the canonical sequence of commands:

```
mkdir build
cd build
cmake ..
make
```

starting from the root directory of the MOPAC repository. MOPAC should build without any additional options
if CMake successfully detects a Fortran compiler and BLAS/LAPACK libraries. Otherwise, the `cmake ..` command
may require additional command-line options to specify a Fortran compiler (`-DCMAKE_Fortran_COMPILER=...`)
or the path (`-DMOPAC_LINK_PATH=...`) and linker options (`-DMOPAC_LINK=...`) to link BLAS and LAPACK libraries to the MOPAC executable.

The CTest-based testing requires an installation of Python 3.x and Numpy that can be detected by CMake.

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

GPU verification (local)
- With GPU=ON, a local target runs a quick MOZYME energy check across CPU, 1-GPU, and 2-GPU and enforces a tolerance:
  - Configure: `cmake -S . -B build-gpu -DGPU=ON -DGPU_VERIFY_PAIR=1,2 -DGPU_VERIFY_TOL=1e-4`
  - Build: `cmake --build build-gpu -j`
  - Verify: `cmake --build build-gpu --target mozyme-gpu-verify`
  - Optional CTest: configure with `-DENABLE_GPU_TESTS=ON` then `ctest -V -L gpu`

GPU verification (CI)
- A GitHub Actions job `gpu-verify` is included for self-hosted GPU runners. It triggers only when:
  - Manually dispatched (workflow_dispatch), or
  - Pushed to a designated branch (see CI.yaml for the current condition), on a runner labeled `self-hosted` and `gpu`.
  - Customize device pair/tolerance via repository variables `GPU_VERIFY_PAIR`, `GPU_VERIFY_TOL` or edit the workflow env.
  - From the Actions UI: select the “CI” workflow → “Run workflow”, then set inputs:
    - GPU pair: e.g., `1,2` (optional; overrides default or env)
    - Energy tolerance: e.g., `1e-4` (optional; default `1e-4`)
    - Choose target branch and click “Run workflow”.

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

End‑to‑end run examples
- Single‑GPU default (auto device selection):
  - `./build-gpu/mopac my_system.mop`
- Force GPU on small system:
  - `MOPAC_FORCEGPU=1 ./build-gpu/mopac my_small_system.mop`
- Disable GPU explicitly:
  - `MOPAC_NOGPU=1 ./build-gpu/mopac my_system.mop`
- Debug with streams disabled (serialize copies/compute on default stream):
  - `MOPAC_STREAMS=off ./build-gpu/mopac my_system.mop`
- Restrict to one device with CUDA:
  - `CUDA_VISIBLE_DEVICES=1 ./build-gpu/mopac my_system.mop`
- MOZYME on two specific GPUs with custom threshold:
  - Put on the first line of the input: `MOZYME MOZYME_2GPU MOZYME_MINBLK=32 MOZYME_GPUPAIR=1,2`
  - Run: `CUDA_VISIBLE_DEVICES=0,1 ./build-gpu/mopac protein.mop`

Compare CPU vs GPU (quick script)
- Run the included helper to execute all examples with CPU and with GPU forced, then compare heats of formation:
  - `python3 scripts/run_examples_compare.py ./build-gpu/mopac`
  - Or specify files: `python3 scripts/run_examples_compare.py ./build-gpu/mopac examples/h2o_cpu.mop examples/ethanol.mop`
  - Set a custom workdir: `python3 scripts/run_examples_compare.py ./build-gpu/mopac examples/*.mop --workdir out_compare`

MOZYME peptide benchmark (CSV export)
- Sweep MOZYME_MINBLK values on a peptide input and export results to CSV for plotting:
  - `python3 scripts/peptide_bench.py ./build-gpu/mopac examples/peptide_gg.mop --minblk 8,16,24,32,48,64 --csv gg_bench.csv`
  - Two GPUs (pair 0,1) and average best of 3: `python3 scripts/peptide_bench.py ./build-gpu/mopac examples/peptide_aaa.mop --two-gpu --pair 1,2 --devices 0,1 --repeat 3 --csv aaa_2gpu.csv`
  - Disable streams for a control run: `--streams off`
  
Plotting the benchmark results
- Use the helper to plot minblk vs time from one or more CSV files (requires matplotlib):
  - `python3 scripts/plot_peptide_csv.py gg_bench.csv --out gg_bench.png --title "Gly–Gly MOZYME minblk sweep"`
  - Compare two datasets: `python3 scripts/plot_peptide_csv.py gg_bench.csv aaa_2gpu.csv --labels GlyGly,Ala3 --out compare.png`

One‑shot sweep + plot helper
- Run a minblk sweep and plot in one go (produces .csv and .png):
  - `bash scripts/peptide_sweep_plot.sh -e ./build-gpu/mopac -i examples/peptide_gg.mop -o gg_sweep`
  - 2 GPUs pair 0,1, repeat 3: `bash scripts/peptide_sweep_plot.sh -e ./build-gpu/mopac -i examples/peptide_aaa.mop -o aaa_sweep --2gpu --pair 1,2 --devices 0,1 --repeat 3`

ctest integration (optional)
- Configure with `-DENABLE_GPU_TESTS=ON -DGPU=ON`.
- List and run tests labeled `gpu`:
  - `ctest -N -L gpu`
  - `ctest -V -L gpu`

Expected correctness
- The GPU paths are designed to match the CPU numerics to double‑precision round‑off. Typical diffs:
  - Densities/Fock: 0 or ~1e‑15
  - Rotations/Eigenvalues: ~1e‑15 to 1e‑14
  - If diffs exceed ~1e‑12 consistently, please open an issue with input and environment details.

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

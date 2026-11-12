<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/barranquilla_mopac_logo_dark.svg">
    <img src="logo/barranquilla_mopac_logo.svg" alt="Barranquilla MOPAC GPU" width="520">
  </picture>
</p>

# Barranquilla MOPAC — MOZYME, solids and QM/MM on the GPU

Barranquilla MOPAC is [MOPAC](https://github.com/openmopac/mopac) with its linear-scaling **MOZYME**
method (proteins, DNA/RNA, large complexes) running on an NVIDIA GPU. The self-consistent field,
the one-electron matrix, the gradient and the dispersion and hydrogen-bond corrections all run on
the device, and the electronic state stays on the GPU between geometry steps. You do not change
your input: any `MOZYME` job on a capable GPU uses it automatically.

The **conventional SCF** runs on the GPU too, for **solids** (translation vectors `Tv`: crystals,
slabs, polymers) and for molecules of 300 orbitals or more: one-electron matrix, Fock matrix, the
pseudo-diagonalization between full diagonalizations, and the gradient, RHF and UHF, with COSMO,
`FORCE`, `DRC`, Fermi smearing (`SMEAR`) and analytical C.I. gradients.

It also adds **molecular dynamics at a temperature** (NVE and NVT with the Bussi thermostat),
**QM/MM molecular dynamics with OpenMM** (MOPAC computes the QM region, with long-range QM–MM
electrostatics by PME), and an **input chemistry check** that stops broken PDB-derived inputs before
they waste hours of SCF.

**How much faster?** One NVIDIA A100-SXM4-40GB against the same program on the CPU of the same Google
Colab machine (Intel Xeon, 2.2 GHz, 12 threads); the same binary with `MOPAC_NOGPU=1` gives the CPU
numbers. PM7; GPU times are the median of three runs.

| MOZYME job | Atoms | CPU (1 core) | GPU (A100) | Speed-up |
|---|---:|---:|---:|---:|
| Crambin, single point (1CRN) | 642 | 25.4 s | 1.22 s | **21×** |
| DNA dodecamer, single point (1BNA) | 781 | 19.7 s | 1.17 s | **17×** |
| Ubiquitin, single point (1UBQ) | 1 231 | 49.2 s | 1.42 s | **35×** |
| Barnase, single point (1RNB) | 1 778 | 78.4 s | 2.07 s | **38×** |
| Antifreeze protein, single point (1EZG) | 2 064 | 140 s | 3.07 s | **45×** |
| Bacteriorhodopsin, single point (1C3W) | 4 473 | 294 s | 5.53 s | **53×** |
| Adenylate kinase, single point (1AKE) | 6 689 | 695 s | 9.09 s | **76×** |
| Water cluster, 7 052 H₂O (42 312 orbitals) | 21 156 | — | 29.4 s | |
| Crambin in implicit water (COSMO), single point | 642 | 30.0 s | 1.67 s | **18×** |
| Crambin, geometry optimization, 100 cycles | 642 | 993 s | 27.0 s | **37×** |
| Adenylate kinase, geometry optimization, 50 cycles | 6 689 | — | 83 s (1.7 s/cycle) | |
| Crambin, NVE dynamics, 0.1-fs steps | 642 | 11.0 s/step | 0.27 s/step | **40×** |
| Adenylate kinase, NVE dynamics, 95 steps | 6 689 | — | 2.7 s/step | |

The MOZYME CPU code does not use threads (crambin, ubiquitin and adenylate kinase took the same time
with 1, 4 and 12 threads), so its reference is one core. Several CPU jobs side by side on a many-core
node give more throughput per node; that comparison was not measured. Small systems gain little (125
water molecules: 3.1×).

**Solids and the conventional SCF** (PM7 `1SCF GRADIENTS`; CPU on all 12 threads):

| Job | Atoms | CPU (12 threads) | GPU (A100) | Speed-up |
|---|---:|---:|---:|---:|
| Silicon supercell | 512 | 369 s | 22.0 s | **17×** |
| Silicon supercell | 1 000 | — | 76.8 s | |
| Rutile TiO₂ supercell | 576 | 179 s | 14.0 s | **13×** |
| α-quartz SiO₂ supercell | 576 | 172 s | 13.8 s | **12×** |
| Urea crystal | 768 | 80.6 s | 9.4 s | **8.6×** |
| Diamond | 64 | 5.2 s | 1.2 s | **4.2×** |
| Graphene (2D, ~2 800 SCF iterations) | 72 | 102 s | 77.7 s | 1.3× |
| Diamond, `FORCE` (Γ-point vibrations) | 64 | 745 s | 99 s | **7.5×** |
| Diamond, `DRC` dynamics | 64 | 4.2 s/step | 0.22 s/step | **19×** |
| Crambin, conventional RHF (`SCFCRT=1.D-6`) | 650 | 17.6 s | 4.9 s | 3.6× |

The solid timings include writing the `AUX` file, about half of the GPU time of the 512–576-atom cells
(without it: silicon 512 10.0 s, quartz 7.7 s, rutile 7.8 s, urea 7.1 s).

**QM/MM dynamics** (OpenMM + MOPAC, protein in a water box; time per MD step):

| QM region | QM atoms | Atoms in total | MOPAC on the CPU | MOPAC on the GPU |
|---|---:|---:|---:|---:|
| Tyrosine side chain (chignolin) | 15 | 4 730 | 0.028 s ¹ | |
| Chignolin, whole peptide | 138 | 12 095 | 2.0 s | 0.185 s (**11×**) |
| Crambin, whole protein | 642 | 24 813 | 34.5 s | 1.33 s (**26×**) |
| Trypsin–BPTI complex, with PME | 54 + 5 link ¹ | 46 853 | 0.231 s | |

0.5-fs steps. ¹ In this run QM regions of up to 100 atoms (link atoms included) ran MOPAC on the CPU in GPU
mode as well; OpenMM and the PME grid still use the GPU. The CPU QM/MM runs were shorter than the GPU runs.
Since 2026-11-12 the limit is 20 atoms: a later A100 run (commit `52c7f964`, 400 steps) gave
0.129 s per step for trypsin–BPTI with MOPAC on the GPU; regions of 10 atoms are still faster on the CPU
(0.022 against 0.033 s per step, chignolin residue 1).

All numbers are from one benchmark run (commit `c85eb96d`); see [Benchmarks](#8-benchmarks) to
repeat them.

---

**[Wiki: 64 worked examples](https://github.com/juvenalyosa/mopac_gpu/wiki)**, most of them with the new keywords
(dynamics, `TRUST`, `SWIG`, `SMEAR`, solids, QM/MM), plus a keyword reference and troubleshooting.

## Contents

1. [Try it in Google Colab](#1-try-it-in-google-colab)
2. [Install](#2-install)
3. [Your first calculation](#3-your-first-calculation)
4. [How it works](#4-how-it-works)
5. [New keywords](#5-new-keywords)
6. [Examples](#6-examples)
7. [QM/MM with OpenMM](#7-qmmm-with-openmm)
8. [Benchmarks](#8-benchmarks)
9. [Accuracy](#9-accuracy)
10. [Limitations](#10-limitations)
11. [Repository map, tests and documentation](#11-repository-map-tests-and-documentation)
12. [Citing and license](#12-citing-and-license)

---

## 1. Try it in Google Colab

No local GPU needed. Open a notebook, choose **Runtime → Change runtime type → A100 (or any GPU)**,
and run the cells in order (the build takes about 10 minutes):

| Notebook | What it does |
|---|---|
| [`colab/mozyme_md_colab.ipynb`](https://colab.research.google.com/github/juvenalyosa/mopac_gpu/blob/main/colab/mozyme_md_colab.ipynb) | Downloads a PDB, prepares it (hydrogens, chemistry check), optimizes it, runs NVT and NVE dynamics, plots temperature and energies, animates the trajectory. |
| [`colab/materials_gpu_a100_colab.ipynb`](https://colab.research.google.com/github/juvenalyosa/mopac_gpu/blob/main/colab/materials_gpu_a100_colab.ipynb) | Solids: device hcore, Fock and gradient against the CPU, the 512-576-atom cells, `FORCE`, `DRC`, MOZYME with `Tv`, `SMEAR`, C.I. gradients and the test suite; options at the top of the notebook. |
| [`colab/paper_benchmarks_colab.ipynb`](https://colab.research.google.com/github/juvenalyosa/mopac_gpu/blob/main/colab/paper_benchmarks_colab.ipynb) | The complete benchmark suite of the tables above (MOZYME, methods, COSMO, optimization, dynamics, frequencies, solids, QM/MM, comparison with MOPAC 23.1.2); keeps results on Google Drive and resumes an interrupted session. |
| [`colab/mozyme_diagg_profile_colab.ipynb`](https://colab.research.google.com/github/juvenalyosa/mopac_gpu/blob/main/colab/mozyme_diagg_profile_colab.ipynb) | Profiling and validation of the MOZYME SCF: CPU-vs-GPU timings, the acceptance test, all semiempirical methods, long optimizations and dynamics. |

---

## 2. Install

### Requirements

- Linux with an NVIDIA GPU of compute capability 7.0 or newer (V100, T4, A100, H100, RTX 20xx and
  later). All GPU arithmetic is double precision, so data-centre GPUs (A100, H100, H200, V100) are much
  faster than gaming cards: see [Which GPU?](#which-gpu).
- CUDA toolkit 12 or 13 (the benchmarks used 13.0), CMake ≥ 3.14, a Fortran compiler (gfortran),
  BLAS/LAPACK, Ninja (optional).
- Device memory: the largest readings in the benchmarks were 8.9 GiB for the 6 689-atom protein and
  28.6 GiB for the 21 156-atom water cluster (A100, 40 GB); silicon 1000 atoms 27.3 GiB.
- For QM/MM: Python ≥ 3.10, OpenMM ≥ 8.4, NumPy, SciPy, and optionally CuPy (PME grid on the GPU).

### Which GPU?

MOZYME does all its GPU arithmetic in double precision (FP64), without tensor cores, and much of the SCF
is limited by memory bandwidth and by the latency of the serial orbital rotations. What decides the speed
is therefore the GPU's plain FP64 rate and memory bandwidth, not the AI (FP16/FP8) figures on the box.
Gaming and inference cards run FP64 at 1/32 or 1/64 of their FP32 rate and are slow here.

| GPU | Arch. (`CUDA_ARCHS`) | FP64, TFLOPS | Memory bandwidth | Memory | Here |
|---|---|---:|---:|---:|---|
| **H200** | Hopper (90) | 34 | 4.8 TB/s | 141 GB | faster than A100 (expected) |
| **GH200** | Hopper (90) | 34 | 4.0–4.9 TB/s | 96–144 GB | faster than A100 (expected) |
| **H100 SXM** | Hopper (90) | 34 | 3.35 TB/s | 80 GB | faster than A100 (expected) |
| **H100 NVL / PCIe** | Hopper (90) | 30 / 26 | 3.9 / 2.0 TB/s | 94 / 80 GB | faster than A100 (expected) |
| **B200 / GB200** | Blackwell (100) | ~37–40 | 8 TB/s | 180–192 GB | faster than A100 (expected; needs CUDA ≥ 12.8, untested) |
| **A100** | Ampere (80) | 9.7 | 1.6–2.0 TB/s | 40 / 80 GB | **reference, measured** (tables above) |
| V100 | Volta (70) | 7.8 | 0.9 TB/s | 16 / 32 GB | somewhat slower than A100 (expected) |
| A30 | Ampere (80) | 5.2 | 0.9 TB/s | 24 GB | slower (expected) |
| L40S, RTX 6000 Ada, RTX 4090 | Ada (89) | 1.3–1.4 | 0.9–1.0 TB/s | 24–48 GB | much slower (expected) |
| A10, L4 | Ampere / Ada | 0.5–1.0 | 0.3–0.6 TB/s | 24 GB | much slower (expected) |
| T4 | Turing (75) | 0.25 | 0.32 TB/s | 16 GB | **measured**: crambin FORCETS 27 min against 7.5 min on an A100 (3.7× slower); fine for testing (free on Kaggle) |

FP64 and bandwidth are the vendors' data-sheet values (FP64 without tensor cores). Only the A100 and the
T4 have been timed with this code. For Hopper and Blackwell the gain over an A100 should be real but
smaller than the FP64 ratio, because part of each SCF waits on latency, not arithmetic. Check the data
sheet before buying newer parts: some recent NVIDIA products (for example Blackwell Ultra, B300) cut FP64
sharply. AMD Instinct cards (MI250X, MI300) have high FP64 but are not supported: the GPU code is CUDA.
Frequency jobs (`FORCE`, `THERMO`) scale with the number of GPUs (`scripts/mozyme_parallel_force.py
--gpus auto`), so two or four A100s beat one faster card there; a single SCF, optimization or dynamics
run uses one GPU.

### Build with GPU support

```bash
# 1. tools (Ubuntu / Debian)
sudo apt-get update
sudo apt-get install -y cmake gfortran ninja-build libblas-dev liblapack-dev
# the CUDA toolkit comes from NVIDIA (https://developer.nvidia.com/cuda-downloads)
nvcc --version            # must print a CUDA 12.x or 13.x release

# 2. source
git clone https://github.com/juvenalyosa/mopac_gpu.git
cd mopac_gpu

# 3. configure and build (CUDA_ARCHS=native compiles for the GPU in this machine)
cmake -S . -B build -G Ninja -DGPU=ON -DCUDA_ARCHS=native -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mopac --parallel

# 4. the program (and the library build/libmopac.so, used by QM/MM)
./build/mopac --version     # MOPAC version 23.1.2 commit ...
```

Other `CUDA_ARCHS` values: `all` (a binary for every common GPU generation) or an explicit list
such as `70;80;90` (V100, A100, H100) when you build on one machine and run on another.

### Build for the CPU only

```bash
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target mopac --parallel
```

The CPU build accepts every input in this README; the new keywords (`TEMPERATURE=`, `BUSSI=`,
`SEED=`, `TRUST`, `SWIG`, `SMEAR`) and the input chemistry check work there too, just without the GPU
speed.

### Python packages for QM/MM

```bash
conda install -c conda-forge openmm numpy scipy     # or: pip install 'openmm[cuda12]' numpy scipy
pip install cupy-cuda12x                             # optional: PME grid of the QM-MM electrostatics on the GPU
```

### Check the installation

```bash
cd tests
python3 check_mozyme_gpu_tolerance.py ../build/mopac \
    ../benchmarks/publication_inputs/mop/protein_crambin_1crn.mop \
    ../benchmarks/publication_inputs/mop/protein_crambin_1crn_opt.mop \
    --handback ../benchmarks/publication_inputs/mop/protein_crambin_1crn_denout.mop
```

It runs crambin on the CPU and on the GPU and must end with `all checks passed`. For solids,
`python3 tests/check_materials_gpu.py build/mopac` compares the device paths with the CPU loops; for
QM/MM, `python3 tests/check_qmmm_pme_ewald.py --gpu` checks the long-range electrostatics against a
direct Ewald sum (no MOPAC needed).

---

## 3. Your first calculation

A MOPAC input file has three header lines: the **keywords**, a **title** and a **comment** (may be
empty). With `GEO_DAT` the geometry is read from a PDB file next to the input.

**`crambin.mop`**
```
PM7 MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
Crambin, single point energy
```

```bash
mopac crambin.mop          # writes crambin.out (and crambin.arc)
```

What to look for in `crambin.out`:

```
 [MOZYME GPU] resident SCF enabled by default (disable with NOGPU or MOPAC_NOGPU=1)
 ...
 [MOZYME GPU SCF] status=success reason=backend_complete code=0 iterations=42
 ...
          FINAL HEAT OF FORMATION =      -2901.67894 KCAL/MOL
```

`status=success` means the SCF ran on the GPU. If you see `status=fallback_cpu`, that SCF was
finished by the CPU code (correct result, CPU speed); see [Limitations](#10-limitations).

> The PDB must contain hydrogen atoms. Crystal structures usually do not; add them first with
> `ADD-H` ([example 6.2](#62-prepare-a-pdb-from-the-protein-data-bank)). Ready-made hydrogenated
> structures of the benchmark proteins are in `benchmarks/publication_inputs/`.

---

## 4. How it works

MOZYME describes the electrons with **localized molecular orbitals** (LMOs): each orbital lives on a
few neighbouring atoms, so the work grows linearly with the size of the molecule instead of with
its cube. One SCF iteration is a sequence of steps over those orbitals:

```
            ┌──────────────────────── on the GPU, every iteration ───────────────────────┐
 geometry → │ tidy/check LMOs → rotate occupied/virtual pairs (DIAGG) → density matrix → │ → energy,
            │ → extrapolate (CNVGZ) → Fock matrix → energy → converged? ─── no ──┐        │   gradient
            └────────────────────────────────────────────────────────────────────┘        ┘
```

- **Resident SCF.** The LMOs, density and Fock matrices stay in GPU memory for the whole SCF and
  between geometry steps. The CPU only launches work and reads back a yes/no convergence flag, so
  no large arrays cross the PCIe bus inside the loop.
- **Parallel kernels.** The orbital rotations (the dominant cost) run as thousands of independent
  2×2 rotations, one GPU thread block per virtual orbital, with locks on the occupied orbitals they
  share.
- **Everything around the SCF also runs on the GPU:** the one-electron (core) matrix, the gradient
  of every atom pair (including d orbitals of S and P), the PM6/PM7 dispersion and hydrogen-bond
  corrections.
- **Warm steps.** In an optimization or a dynamics run the next geometry starts from the previous
  orbitals already on the GPU: a cycle of a 6 700-atom protein optimization takes about 1.7 s.
- **Automatic fallback.** If a step cannot run on the GPU (for example an SCF that does not
  converge within its iteration budget), MOPAC hands the current state to the ordinary CPU code,
  which finishes the job. The answer is still correct.
- **Off switch.** The `NOGPU` keyword or the environment variable `MOPAC_NOGPU=1` gives the
  CPU code path of the same program.

The conventional SCF of solids follows the same idea: the one-electron and Fock matrices, the
diagonalization (cuSOLVER), the pseudo-diagonalization and the gradient run on the device; the
one-centre terms and the far periodic images (point charges) stay on the host. Details in
`docs/GPU_GUIDE.md`.

---

## 5. New keywords

These keywords are **new in this fork**. They follow MOPAC's usual form (`WORD=value`) and are
listed in the output header like every other keyword.

- `SWIG` (with `EPS=`): smooth COSMO surface with a consistent gradient (Lange and Herbert).
- `HB-SMOOTH`: the PM6-DH+ / PM7 hydrogen-bond correction without kinks (for optimization, FORCE, MD).
- `TRUST`: trust-region L-BFGS geometry optimizer, recommended for MOZYME (the line searches of the default
  L-BFGS-B fail on the noise of the MOZYME SCF; with `TRUST` the radius follows the directional derivative
  and no step is ever undone, so the LMOs of the last SCF stay valid).
- `SMEAR=kT` (eV; `SMEAR` alone: 0.1): Fermi–Dirac occupations for metals and small-gap solids
  (closed-shell RHF). The heat is the Mermin free energy E − TS; the output also prints E(kT → 0).
  See `docs/BACKGROUND.md`.
- `MM_CHARGES="file"`: QM/MM electrostatic embedding in MM point charges (x y z q per line); the OpenMM
  driver passes them through the API instead (`mozyme_qmmm_init/step/finalize` in `include/mopac.h`).

### Molecular dynamics at a temperature (used with `DRC`)

| Keyword | Meaning | Default |
|---|---|---|
| `TEMPERATURE=n` | Start the dynamics from velocities drawn from the **Maxwell–Boltzmann distribution at n kelvin** (zero net momentum, kinetic energy exactly ½·(3N−3)·k·n). Without `BUSSI` the energy is then conserved: **NVE**. | — |
| `BUSSI=n` | Keep the temperature at the `TEMPERATURE` value with the **Bussi–Donadio–Parrinello stochastic velocity rescaling thermostat** (canonical ensemble): **NVT**. n is the coupling time in femtoseconds; small n = tight coupling. `BUSSI` without a value means 100 fs. | 100 fs |
| `SEED=n` | Seed of the random numbers used for the starting velocities and the thermostat. Same seed + same input = same trajectory (on the CPU; the GPU SCF is not bit-reproducible). | 20260925 |

Rules: `TEMPERATURE` needs `DRC`; `BUSSI` needs `TEMPERATURE`. Otherwise the job stops with
`TEMPERATURE AND NO DRC` or `BUSSI AND NO TEMPERATURE`.

During the run a line is printed every 50 steps (numbers for illustration):

```
          TEMPERATURE: STEP      50  TIME(FS)     4.912  T(K)   302.14  <T>(K)   301.77  ENERGY TO BATH (KCAL/MOL)     -3.1275
```

`T(K)` is the instantaneous temperature, `<T>(K)` the running average and `ENERGY TO BATH` the
energy the thermostat has taken out (negative) or put in (positive). In the DRC table the column
`ERROR` stays the **integration error only**: the energy exchanged with the bath is booked
separately, so a small `ERROR` means the dynamics is accurate in NVT as well as in NVE.

### GPU control

| Keyword / environment variable | Meaning |
|---|---|
| *(nothing)* | A `MOZYME` job uses the GPU when one with compute capability ≥ 7.0 is present. |
| `NOGPU` (keyword) | Run this job entirely on the CPU (identical to the CPU build). |
| `MOPAC_NOGPU=1` (environment) | Same, for every job started from this shell. |
| `MOPAC_FORCEGPU=1` (environment) | Use the GPU even if the input says `NOGPU`, and for solids under 100 atoms. |

### Input chemistry check (automatic, no keyword needed)

Every PDB-derived input is checked before the calculation. When the check finds errors, the job stops
and the output shows the report (warnings included); inputs without errors run with no report:

```
          INPUT CHEMISTRY CHECK
          ERROR   SO4 A  64: incomplete group, 3 heavy atoms instead of 5.  Remove the fragment or complete it.
          WARNING all 14 ARG/LYS/ASP/GLU/HIS residues are neutral (gas-phase protonation);
                  for a system at pH ~7 hydrogenate with ADD-H SITE=(IONIZE).
          Errors: 1   Warnings: 1
```

| Finding | Level |
|---|---|
| A group of several atoms on a crystallographic special position (`REMARK 375` in the PDB: only part of it is in the file, the rest is a symmetry copy) | **error** |
| An incomplete small group: SO4, PO4, NO3, EDO, GOL, ACT, FMT, DMS, MPD, TRS | **error** |
| A formal charge of ±2 or more on H, B, C, N, O or F in the Lewis structure | **error** |
| Amino acid with missing side-chain atoms | warning |
| Atoms with partial occupancy | warning |
| All Arg/Lys/Asp/Glu/His neutral; sulfate/phosphate in acid form | warning |

**Errors stop the job.** Add the existing MOPAC keyword `LET` to run anyway (the report stays in the output). Why this matters: in
the PDB entry 1G6X two half sulfates on a symmetry axis made the SCF take 112 iterations and give a
different energy on every GPU run; after removing them it converges in 29.

---

## 6. Examples

All examples are complete input files. Save the lines in a file, e.g. `job.mop`, put the PDB file
in the same directory and run `mopac job.mop`. Lines are shown exactly as they go in the file.
More inputs: `examples/`, `benchmarks/publication_inputs/mop/` (the benchmark proteins) and
`tests/data/materials/` (the periodic cells).

### 6.1 Single-point energy of a protein

```
PM7 MOZYME 1SCF GEO_DAT="1ubq_hydrogenated.pdb"
Ubiquitin, single point

```

### 6.2 Prepare a PDB from the Protein Data Bank

Download, then let MOPAC add the hydrogen atoms. `0SCF` stops after the preparation; `PDBOUT`
writes the result as `prepare.pdb`.

```bash
wget https://files.rcsb.org/download/1CRN.pdb
```

**`prepare.mop`** — neutral groups (as in the benchmarks):
```
PM7 ADD-H PDBOUT NEWPDB 0SCF GEO-OK GEO_DAT="1CRN.pdb"
Add hydrogens to crambin

```

**`prepare.mop`** — charged groups as at pH ≈ 7 (Arg⁺, Lys⁺, Asp⁻, Glu⁻, termini, SO4²⁻, PO4²⁻):
```
PM7 ADD-H SITE=(IONIZE) PDBOUT NEWPDB 0SCF GEO-OK GEO_DAT="1CRN.pdb"
Add hydrogens to crambin, pH 7

```

Remove waters and alternate conformations first (or use `scripts/mozyme_md_workflow.py`, which
does it). If the chemistry check reports an error, fix the PDB rather than adding `LET`.

### 6.3 Other semiempirical methods

Replace `PM7` by the method. MNDO, AM1, PM3, RM1, PM6, PM6-D3H4 and PM7 were compared on the GPU and
the CPU for crambin and the DNA dodecamer (11–33× faster; see [Accuracy](#9-accuracy)):

```
PM6-D3H4 MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
AM1 MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
RM1 MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
PM3 MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
MNDO MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
```

(one method per input file, each followed by a title line and an empty line).

### 6.4 Implicit water (COSMO)

`EPS=78.4` is the dielectric constant of water. Add `SWIG` for optimizations, frequencies and dynamics in
water: the standard COSMO surface changes in steps as atoms move, the smooth SWIG surface does not (see
`docs/MOZYME_GPU_USER_GUIDE.md`).

```
PM7 MOZYME 1SCF EPS=78.4 SWIG GEO_DAT="1crn_hydrogenated.pdb"
Crambin in implicit water

```

Explicit water and the rest of the protein can be molecular mechanics instead: see
[QM/MM with OpenMM](#7-qmmm-with-openmm).

### 6.5 Geometry optimization

```
PM7 MOZYME TRUST CYCLES=200 GEO_DAT="1crn_hydrogenated.pdb" PDBOUT
Crambin, optimization

```

The final geometry is written to `job.arc` and, with `PDBOUT`, to `job.pdb`. If the job stops at
the cycle limit, MOPAC leaves a restart file `job.res`; continue with:

```
PM7 MOZYME CYCLES=200 RESTART GEO_DAT="1crn_hydrogenated.pdb" PDBOUT
Crambin, optimization continued

```

Only the final geometry, as a PDB, from the restart file:

```
PM7 MOZYME 1SCF RESTART PDBOUT GEO_DAT="1crn_hydrogenated.pdb"
Final geometry of the optimization

```

### 6.6 Optimization in implicit water with PM6-D3H4

```
PM6-D3H4 MOZYME EPS=78.4 TRUST CYCLES=300 GEO_DAT="protein_hydrogenated.pdb" PDBOUT
Protein optimized in water

```

### 6.7 NVE dynamics (constant energy) at 300 K

`T-PRIORITY=0.5` prints one row of the table every 0.5 fs; `CYCLES=200` stops after 200 rows
(100 fs). The trajectory, one frame per row, is written to `job.xyz`.

```
PM7 MOZYME DRC TEMPERATURE=300 T-PRIORITY=0.5 CYCLES=200 GEO_DAT="1crn_optimized.pdb"
Crambin, NVE at 300 K

```

Output (the `TOTAL` column is constant up to the `ERROR` column):

```
 FEMTOSECONDS  POINT  POTENTIAL + KINETIC  =   TOTAL     ERROR    REF%   MOVEMENT
     0.000       1  -2901.6813  573.20898  -2328.4723   0.00000     1   %  0.0000
     0.500       5  -2899.3389  570.86939  -2328.4696   0.00294     2   %  5.1095
     1.500      15  -2888.9053  560.40901  -2328.4962  -0.02336     4   % 15.0601
```

The kinetic energy at 0 fs, 573.21 kcal/mol, is exactly ½ · 1923 · k · 300 K for the 642 atoms of
crambin (1923 = 3·642 − 3 degrees of freedom).

### 6.8 NVT dynamics (constant temperature) at 300 K

```
PM7 MOZYME DRC TEMPERATURE=300 BUSSI=100 T-PRIORITY=0.5 CYCLES=200 GEO_DAT="1crn_optimized.pdb"
Crambin, NVT at 300 K, Bussi thermostat, 100 fs coupling

```

**Equilibrate first.** Starting from an optimized structure (an energy minimum), half of the
kinetic energy flows into potential energy in the first ~10 fs, so the temperature drops to about
T/2; the thermostat brings it back with an effective time of about 2 × `BUSSI` (it has to heat the
potential energy too). Measured on crambin: with `BUSSI=100` the mean temperature after 100 fs was
still 199 K. Use a tight coupling to equilibrate, then a loose one (or NVE) to sample:

```
PM7 MOZYME DRC TEMPERATURE=300 BUSSI=10 T-PRIORITY=0.5 CYCLES=100 GEO_DAT="1crn_optimized.pdb"
Crambin, NVT equilibration, 10 fs coupling

```

### 6.9 NVT dynamics in implicit water, reproducible

```
PM7 MOZYME EPS=78.4 DRC TEMPERATURE=310 BUSSI=100 SEED=42 T-PRIORITY=1 CYCLES=500 GEO_DAT="protein_optimized.pdb"
Protein in water at 310 K, 500 fs, seed 42

```

Change `SEED` to get an independent trajectory from the same structure (useful for replicas).

### 6.10 Heating a structure

Several short NVT runs, each starting from the last frame of the previous one (the
`last_frame_pdb` function of `scripts/mozyme_md_workflow.py` writes it as a PDB):

```
PM7 MOZYME DRC TEMPERATURE=100 BUSSI=20 T-PRIORITY=0.5 CYCLES=100 GEO_DAT="step0.pdb"
PM7 MOZYME DRC TEMPERATURE=200 BUSSI=20 T-PRIORITY=0.5 CYCLES=100 GEO_DAT="step1.pdb"
PM7 MOZYME DRC TEMPERATURE=300 BUSSI=20 T-PRIORITY=0.5 CYCLES=100 GEO_DAT="step2.pdb"
```

(three separate input files).

### 6.11 The original DRC is still there

Without `TEMPERATURE`, `DRC` behaves as in standard MOPAC, for example adding 20 kcal/mol of
kinetic energy along the initial gradient:

```
PM7 MOZYME DRC KINETIC=20 T-PRIORITY=0.5 CYCLES=40 GEO_DAT="1crn_hydrogenated.pdb"
Crambin, classic DRC with 20 kcal/mol of kinetic energy

```

### 6.12 Compare the GPU with the CPU

```bash
mopac job.mop                       # GPU
MOPAC_NOGPU=1 mopac job.mop         # CPU, same program
grep "FINAL HEAT OF FORMATION" job.out
```

or, inside the input, add `NOGPU` to the keyword line.

### 6.13 A solid

A crystal is an ordinary MOPAC input with three translation vectors (`Tv`, in Å) after the atoms of
the cell; the GPU is used above 100 atoms (`MOPAC_FORCEGPU=1` for smaller cells). Complete cells are
in `tests/data/materials/` (`small/`, 64-192 atoms; `medium/`, 512-768 atoms):

```
PM7 1SCF GRADIENTS GEO-OK
quartz 576 atoms

Si     2.309045 1     0.000000 1     3.603333 1
...
O      8.159577 1    16.405392 1    20.976264 1
Tv    19.664000 0     0.000000 0     0.000000 0
Tv    -9.832000 0    17.029524 0     0.000000 0
Tv     0.000000 0     0.000000 0    21.620000 0
```

Use `PRECISE` for cell optimizations (central differences in the gradient; see [Accuracy](#9-accuracy)),
`FORCE` for Γ-point vibrations (the cell is held fixed), `DRC TEMPERATURE=300` for dynamics and
`SMEAR=0.1` for a metal or a small-gap cell. MOZYME with `Tv` suits molecular crystals; for dense
covalent solids (oxides, silicon) leave `MOZYME` out.

### 6.14 Vibrational frequencies on several GPUs

`FORCE` and `THERMO` are many independent gradient evaluations; this script spreads them over GPU
processes (several per GPU, or several GPUs):

```bash
python3 scripts/mozyme_parallel_force.py --help
```

### 6.15 The whole workflow from a PDB code (Python)

`scripts/mozyme_md_workflow.py` chains preparation, optimization, NVT and NVE:

```bash
wget https://files.rcsb.org/download/1CRN.pdb
python3 scripts/mozyme_md_workflow.py build/mopac 1CRN.pdb --work-dir crambin_md \
    --temperature 300 --opt-cycles 100 --nvt-points 200 --nve-points 200
# add --eps 78.4 for implicit water, --ionize for pH 7 protonation
```

It prints one summary line per stage (preparation: atoms and charge; optimization: cycles, heat
and gradient at the first and last cycle; NVT and NVE: simulated time, wall time, largest
integration error, mean temperature, and whether every SCF ran on the GPU):

```
[prepare] hydrogenated: 642 atoms -> 1CRN_hydrogenated.pdb
[opt] <cycles> cycles in <s> s: heat <first> -> <last>, GRAD <first> -> <last>; GPU SCF all resident: True
[nvt] <points> points to <fs> fs in <s> s (...); max |ERROR| <x> kcal/mol; mean T <T> K; GPU SCF all resident: True
[nve] ...
```

and leaves `opt/`, `nvt/nvt.xyz`, `nve/nve.xyz` and the `.out` files in the work directory. The
same functions are available from Python:

```python
import sys; sys.path.insert(0, "scripts")
import mozyme_md_workflow as md
from pathlib import Path

mopac = Path("build/mopac")
pdb = md.prepare_pdb(mopac, Path("1CRN.pdb"), Path("run/prepare"), ionize=False)
opt_pdb, _ = md.optimize(mopac, pdb, Path("run/opt"), cycles=100, eps=78.4)
nvt = md.dynamics(mopac, opt_pdb, Path("run/nvt"), "NVT", 300, points=200, tau_fs=100, eps=78.4)
temps = md.temperatures(nvt.rows, natoms=642)   # K, one per table row
```

---

## 7. QM/MM with OpenMM

The QM region is computed by MOPAC (PM7, MOZYME, on the GPU above 20 atoms) in the field of the MM
point charges (electrostatic embedding); the force field, water, constraints, integrators,
thermostats and reporters are OpenMM's; the coupling is an OpenMM `PythonForce`. QM–MM covalent bonds get hydrogen link atoms; the
long-range QM–MM electrostatics use PME (on the GPU with CuPy). Full description:
`docs/QMMM_OPENMM.md`.

**From a PDB file, in one command** (hydrogens, water box, MM equilibration, QM region from residue
numbers, NVT, then an NVE check):

```bash
python3 scripts/qmmm/run_qmmm_md.py protein.pdb --libmopac build/libmopac.so \
    --qm-residues 45,72 --qm-sidechain --steps 2000 --dt 0.5 --nve 400 --platform CUDA
```

Useful options: `--qm-charge`, `--ron/--roff` (switching of the MM charges, Å), `--tolerance`
(SCF criterion), `--no-pme` (cut-off only), `--pme-cpu`, `--variants E:57=HID` (protonation states),
`--distances E:195:OG-I:15:C` (distances to follow), `--cutoff-test 8,12,16,24` (force error against
the cut-off), `--seed`. `--help` lists them all.

**From your own OpenMM script:**

```python
import sys; sys.path.insert(0, "scripts/qmmm")
import openmm as mm, openmm.app as app, openmm.unit as u
from openmm_mopac import add_qmmm

system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=1*u.nanometer,
                                 constraints=app.HBonds)
qmmm = add_qmmm(system, topology, positions, qm_atoms, qm_charge=0, libmopac="build/libmopac.so",
                ron=8.0, roff=10.0, tolerance=0.01)      # Angstroms
integrator = mm.LangevinMiddleIntegrator(300*u.kelvin, 1/u.picosecond, 0.5*u.femtoseconds)
simulation = app.Simulation(topology, system, integrator, mm.Platform.getPlatformByName("CUDA"))
simulation.context.setPositions(positions)
simulation.step(1000)
qmmm.close()
```

At the measured rates, 1 ns with 0.5-fs steps takes about 4 days with a 138-atom QM region and about a
month with 642 QM atoms on one A100.

---

## 8. Benchmarks

`scripts/paper/paper_bench.py` runs every benchmark of the tables at the top (speed and accuracy of each
GPU path against the CPU path of the same binary) and `scripts/paper/paper_figures.py` turns the
results into tables, figures and a `summary.md`. The easiest way is the Colab notebook
[`colab/paper_benchmarks_colab.ipynb`](https://colab.research.google.com/github/juvenalyosa/mopac_gpu/blob/main/colab/paper_benchmarks_colab.ipynb)
(A100 runtime): it builds MOPAC at a fixed commit, runs everything, keeps the results on Google Drive
and resumes an interrupted session. The complete run takes several hours, most of it CPU reference
runs.

On your own machine:

```bash
# GPU build with the library (QM/MM experiments), then:
python3 scripts/paper/paper_bench.py --mopac build/mopac --libmopac build/libmopac.so --out results \
    --exp all                       # or a comma list, e.g. mozyme_sp,methods,solids
python3 scripts/paper/paper_figures.py results
```

| Option | Effect |
|---|---|
| `--quick` | One small case per experiment: a test of the pipeline (about 15 min on a CPU) |
| `--no-cpu` | GPU runs only (no CPU references) |
| `--repeats 3` | GPU repetitions of every timed run (median reported) |
| `--cpu-max-atoms`, `--solids-cpu-max-atoms` | Largest systems run on the CPU |
| `--upstream-mopac path` | Also compare with a CPU build of MOPAC 23.1.2 |

Experiments (`--exp`): `mozyme_sp`, `mozyme_grad`, `methods`, `cosmo`, `opt`, `md`, `thermo`, `solids`,
`solids_profile`, `solids_force`, `solids_drc`, `conv_mol`, `qmmm`, `qmmm_enzyme` and the controls;
`paper_bench.py --help` lists them all, and `scripts/paper/README.md` describes what each one measures
and the layout of the results directory (one directory per run with its input, output and
`result.json`; CSV files; tables and figures).

Smaller, older benchmark scripts: `scripts/published_benchmark.py` (protein single points),
`scripts/check_mozyme_gpu_methods.py` (all methods).

---

## 9. Accuracy

The GPU SCF is not bit-identical to the CPU (the order of the parallel orbital rotations varies),
so energies differ between runs at the level of the SCF convergence criterion. Measured on the A100
(benchmark run above; GPU − CPU for the GPU run farthest from the CPU result):

| Check | GPU − CPU |
|---|---|
| MOZYME single points, water clusters 375–6 000 atoms | ≤ 0.006 kcal/mol |
| MOZYME single points, crambin, DNA, BPTI, ubiquitin, barnase, bacteriorhodopsin | ≤ 0.02 kcal/mol |
| Antifreeze protein (1EZG) / adenylate kinase (1AKE) | 0.088 / 0.41 kcal/mol (spread of three GPU runs: 0.08 / 0.21) |
| Same, with a tight criterion (`SCFCRT=1.D-6`) | heats ≤ 5·10⁻⁵ kcal/mol, gradients ≤ 5·10⁻⁴ kcal/mol/Å |
| MNDO, AM1, PM3, RM1, PM6, PM6-D3H4, PM7 on crambin and DNA | ≤ 0.006 kcal/mol, except crambin PM6 / PM6-D3H4 (below) |
| COSMO (crambin, DNA, ubiquitin) | 0.008, 0.012, 0.030 kcal/mol |
| Crambin, 100 optimization cycles: heat at the final geometry, CPU vs GPU | 0.001 kcal/mol |
| Adenylate kinase, 50 GPU cycles: heat at the final geometry, CPU vs GPU | 0.18 kcal/mol |
| Crambin NVE, 19.5 fs: largest integration `ERROR` | CPU 0.14, GPU 0.27 kcal/mol |
| Frequencies, chignolin `FORCE` / crambin `FORCETS` (against stored CPU references) | rms 0.027 / 0.012 cm⁻¹ |
| Solids, 13 cells (64–768 atoms): heats | ≤ 0.007 kcal/mol (graphene); others ≤ 0.0005 |
| QM/MM PME: 20 checks against a direct Ewald sum (Madelung constant, potentials, forces) | all pass |

For crambin with PM6 and PM6-D3H4 the two results were 0.46–0.49 kcal/mol apart because the **CPU**
SCF at the default criterion stopped 0.51 kcal/mol above a tightly converged reference, while the GPU
was 0.02–0.05 from it. When differences of 0.1 kcal/mol matter (for example adenylate kinase above),
use a tighter criterion such as `SCFCRT=1.D-6`.

The default MOPAC gradient uses one-sided differences for the pair derivatives: about 0.05
kcal/mol/Å from the exact derivative of the energy, in molecules and solids, CPU and GPU alike, as in
upstream MOPAC (chignolin, PM7: largest deviation from central differences 0.055 here, 0.29 in MOPAC
23.1.2). `PRECISE` (and `FORCE`) switch to central differences.

Optimization and dynamics trajectories themselves diverge between CPU and GPU (the tiny SCF
differences are amplified, as in any chaotic dynamics); compare energies at the same geometry, or
statistical averages.

**Differences from MOPAC 23.1.2 on the CPU.** This fork also corrects the CPU code (a MOZYME SCF that
stalled near 10⁻³ eV, the PM7 hydrogen-bond, D3, H4 and COSMO gradients, the solid gradient and
dispersion, the solid SCF level shift, converged SCFs in `FORCE` and dynamics). Results can therefore
differ from MOPAC 23.1.2 with the same keywords: MOZYME heats by 0.003–0.32 kcal/mol at the default
criterion; with `SCFCRT=1.D-6` MOPAC 23.1.2 did not converge crambin. Each change is described in its
commit message.

---

## 10. Limitations

- **One GPU vendor and model tested.** CUDA only (NVIDIA); timings are from one A100 (and a T4).
- **Run-to-run variation.** GPU runs are not bit-reproducible (order of the parallel rotations);
  `SEED` fixes the random numbers but not that order. Differences stay at the level of the SCF
  criterion.
- **No NPT in MOPAC dynamics.** MOZYME treats a finite molecule; with implicit solvent there is no box
  and no pressure. NVE and NVT are available.
- **Time step.** `DRC` chooses its own step, about 0.1 fs when hydrogen atoms move. One picosecond
  of crambin is about 10 000 steps (≈ 45 minutes on an A100); 1 ps of a 7 000-atom protein about
  7.5 hours. The dynamics tests so far cover 4–100 fs (MOPAC) and 0.1–0.2 ps (QM/MM); long-time
  energy conservation has not been established.
- **NVE after NVT** restarts from the last NVT geometry with fresh velocities at the same
  temperature: MOPAC cannot read velocities together with a PDB geometry.
- **Runs on the CPU instead of the GPU:** the Fock matrix of conventional-SCF molecules under 300
  orbitals, solids under 100 atoms (unless `MOPAC_FORCEGPU=1`), pKa calculations, and the programming
  interface (`libmopac` API) except the MOZYME QM/MM session (QM regions above 20 atoms).
- **Small systems** gain little: 3.1× for 125 water molecules; the 64–72-atom solids 1.3–4.2×.
- **Inputs.** Hydrogens are required; broken crystal fragments are refused (see the chemistry
  check).
- **Solids, known limits.**
  - **PM7 gradients of molecular crystals.** For a 192-atom urea cell the PM7 analytic gradient
    deviates from central differences by up to 4.4 kcal/mol/Å (MOPAC 23.1.2: up to 7.3); with PM6
    and PM6-DH+ it agrees within 0.07. The cause has not been found; check PM7 gradients of such
    solids against finite differences.
  - Slow SCFs: graphene needs about 2 800 iterations; its heats agree with the CPU (0.007 kcal/mol)
    but its gradient differed by 0.79 kcal/mol/Å. Check such results with a tighter criterion.
  - MOZYME with `Tv` in dense covalent solids: the localized orbitals spread over most of the cell
    (quartz 576: ~480 atoms each), beyond the device limit (2048 coefficients per orbital), and the
    SCF finishes on the CPU, much more slowly than the conventional SCF on the GPU. The output says
    so (`lmo_extent`); use the conventional SCF there.
  - Several RHF solutions: some cells have more than one SCF solution (Li 54 atoms: the default SCF
    ends at 1898.11 kcal/mol, a solution at 1885.15 exists; graphene). Compare heats and HOMO–LUMO gaps when in doubt.
  - Aluminium with PM7 (Al 108) does not converge, with or without `SMEAR`.
  - The pseudo-diagonalization starts only once the density change PL is below 0.05; in the test
    cells it then reaches the same states as full diagonalizations, but a hard SCF could still be
    steered elsewhere: compare with `MOPAC_PDIAG=0`, which turns it off.
  - The half-electron SCF of large open-shell proteins (C.I. on a crambin doublet) needs hundreds of
    iterations.

---

## 11. Repository map, tests and documentation

| Path | Content |
|---|---|
| `src/gpu/mozyme_scf_context.cu` | Resident SCF on the GPU (kernels and loop) |
| `src/gpu/mozyme_pair_*.cu(h)`, `dh_dispersion.cu`, `hbond_dh_plus.cu` | Core matrix, gradient, dispersion and H-bond kernels |
| `src/MOZYME/mozyme_gpu_*.F90`, `iter_for_MOZYME.F90` | Fortran side of the GPU SCF |
| `src/gpu/periodic_fock.cu`, `gpu_conv_fock.F90`, `grad_kernels.cu`, `src/SCF/pseudo_diag_gemm.F90` | Conventional SCF on the GPU: Fock matrix of solids and large molecules, gradient kernels, pseudo-diagonalization |
| `src/reactions/drc.F90`, `drc_ensemble.F90` | Dynamics; `TEMPERATURE=`, `BUSSI=`, `SEED=` |
| `src/chemistry/input_chemistry_check.F90` | Input chemistry check |
| `src/forces/deri_zvector.F90` | Analytical C.I. gradients (Z-vector) |
| `src/interface/` | `libmopac` API, including the MOZYME QM/MM session (`mozyme_qmmm_*`) |
| `scripts/qmmm/` | OpenMM coupling (`openmm_mopac.py`), QM-PME (`qmmm_pme.py`), driver (`run_qmmm_md.py`) |
| `scripts/paper/` | Benchmark suite (`paper_bench.py`) and its tables and figures (`paper_figures.py`) |
| `scripts/mozyme_md_workflow.py` | PDB → optimization → NVT → NVE |
| `scripts/mozyme_parallel_force.py` | `FORCE`/`THERMO` over several GPU processes |
| `tests/check_mozyme_gpu_tolerance.py` | Acceptance test (CPU vs GPU, gradient, fallback, `NOGPU`) |
| `tests/check_mozyme_gpu_production.py` | Long optimization and dynamics check |
| `tests/check_materials_gpu.py`, `tests/data/materials/` | Solids check (GPU vs CPU) and its cells |
| `tests/check_qmmm_pme_ewald.py` | QM/MM long-range electrostatics against a direct Ewald sum |
| `benchmarks/publication_inputs/` | The benchmark proteins, hydrogenated, and their input files |
| `colab/` | Colab notebooks (see section 1) |
| `docs/MOZYME_GPU_USER_GUIDE.md` | Full user guide of the MOZYME GPU path |
| `docs/GPU_GUIDE.md` | Build, runtime switches and measurements of every GPU path, solids included |
| `docs/QMMM_OPENMM.md` | QM/MM with OpenMM: method, options, limits |
| `docs/BACKGROUND.md` | Theory (`SMEAR`, gradient accuracy), earlier GPU work (HMTR optimizer) |

Everything else is standard MOPAC; its manual at [openmopac.net](http://openmopac.net) describes all
other keywords.

---

## 12. Citing and license

Barranquilla MOPAC is an independent derivative of MOPAC; it is not affiliated with or endorsed by
MolSSI or Virginia Tech. It is based on MOPAC (Stewart Computational Chemistry / Virginia Tech, Apache
License 2.0; see `LICENSE` and `NOTICE`). Please cite MOPAC as described in `CITATION.cff`
(doi: [10.5281/zenodo.6511958](https://doi.org/10.5281/zenodo.6511958)), for the thermostat
G. Bussi, D. Donadio and M. Parrinello, *J. Chem. Phys.* **126**, 014101 (2007), and for QM/MM
OpenMM (P. Eastman et al., *J. Phys. Chem. B* **128**, 109–116 (2024), doi:
[10.1021/acs.jpcb.3c06662](https://doi.org/10.1021/acs.jpcb.3c06662)).

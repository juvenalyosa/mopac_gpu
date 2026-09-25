# Barranquilla MOPAC — MOZYME on the GPU

<img src="logo/barranquilla_logo.png" alt="Barranquilla MOPAC Logo" style="max-width: 100%; width: 720px; height: auto;" />

Barranquilla MOPAC is [MOPAC](https://github.com/openmopac/mopac) with its linear-scaling **MOZYME**
method (proteins, DNA/RNA, large complexes) running on an NVIDIA GPU. The self-consistent field,
the one-electron matrix, the gradient and the dispersion and hydrogen-bond corrections all run on
the device, and the electronic state stays on the GPU between geometry steps. You do not change
your input: any `MOZYME` job on a capable GPU uses it automatically.

It also adds **molecular dynamics at a temperature** (NVE and NVT with the Bussi thermostat) and an
**input chemistry check** that stops broken PDB-derived inputs before they waste hours of SCF.

**How much faster?** On one NVIDIA A100 against the same program on one CPU core:

| Job | Atoms | CPU (1 core) | GPU (A100) | Speed-up |
|---|---:|---:|---:|---:|
| Crambin, single point (1CRN) | 642 | 28.3 s | 1.1 s | **26×** |
| Ubiquitin, single point (1UBQ) | 1 231 | 41.1 s | 1.3 s | **32×** |
| Barnase, single point (1RNB) | 1 778 | 71.0 s | 1.8 s | **39×** |
| Antifreeze protein, single point (1EZG) | 2 064 | 130.1 s | 2.5 s | **52×** |
| Bacteriorhodopsin, single point (1C3W) | 4 473 | 343.3 s | 4.5 s | **76×** |
| Adenylate kinase, single point (1AKE, protein) | 6 689 | 449 s | 7.6 s | **59×** |
| Water cluster, 7 052 H₂O (42 312 orbitals) | 21 156 | — | 25.7 s | |
| Crambin, geometry optimization, 100 cycles | 642 | 1 042 s | 17.7 s | **59×** |
| Adenylate kinase, geometry optimization, 50 cycles | 6 689 | — | 57 s (1.0 s/cycle) | |
| Crambin, molecular dynamics, 195 steps (19.5 fs) | 642 | 1 273 s | 21.9 s | **58×** |
| Crambin in implicit water (COSMO), single point | 642 | 33.0 s | 2.1 s | **16×** |
| Crambin, NVT or NVE dynamics, 100 fs (~1 000 steps) | 642 | — | 114 s | |
| Adenylate kinase, molecular dynamics, 95 steps | 6 689 | — | 87 s (0.9 s/step) | |

Energies agree with the CPU within 0.05 kcal/mol (details in [Accuracy](#7-accuracy)). The CPU
column is a single core because MOZYME has no multi-threading; a many-core CPU node would
narrow the gap to roughly 5–10×.

---

## Contents

1. [Try it in Google Colab](#1-try-it-in-google-colab)
2. [Install](#2-install)
3. [Your first calculation](#3-your-first-calculation)
4. [How it works](#4-how-it-works)
5. [New keywords](#5-new-keywords)
6. [Examples](#6-examples)
7. [Accuracy](#7-accuracy)
8. [Limitations](#8-limitations)
9. [Repository map, tests and documentation](#9-repository-map-tests-and-documentation)
10. [Citing and license](#10-citing-and-license)

---

## 1. Try it in Google Colab

No local GPU needed. Open a notebook, choose **Runtime → Change runtime type → A100 (or any GPU)**,
and run the cells in order (the build takes about 10 minutes):

| Notebook | What it does |
|---|---|
| [`colab/mozyme_md_colab.ipynb`](https://colab.research.google.com/github/juvenalyosa/mopac_gpu/blob/main/colab/mozyme_md_colab.ipynb) | Downloads a PDB, prepares it (hydrogens, chemistry check), optimizes it, runs NVT and NVE dynamics, plots temperature and energies, animates the trajectory. |
| [`colab/mozyme_diagg_profile_colab.ipynb`](https://colab.research.google.com/github/juvenalyosa/mopac_gpu/blob/main/colab/mozyme_diagg_profile_colab.ipynb) | Benchmarks and validation: CPU-vs-GPU timings, the acceptance test, all semiempirical methods, the published-benchmark table, long optimizations and dynamics. |

---

## 2. Install

### Requirements

- Linux with an NVIDIA GPU of compute capability 7.0 or newer (V100, T4, A100, H100, RTX 20xx and
  later). All GPU arithmetic is double precision, so data-centre GPUs (A100, H100, V100) are much
  faster than gaming cards.
- CUDA toolkit 12, CMake ≥ 3.14, a Fortran compiler (gfortran), BLAS/LAPACK, Ninja (optional).
- Device memory: about 2 GB for a 7 000-atom protein.

### Build with GPU support

```bash
# 1. tools (Ubuntu / Debian)
sudo apt-get update
sudo apt-get install -y cmake gfortran ninja-build libblas-dev liblapack-dev
# the CUDA toolkit comes from NVIDIA (https://developer.nvidia.com/cuda-downloads)
nvcc --version            # must print a CUDA 12.x release

# 2. source
git clone https://github.com/juvenalyosa/mopac_gpu.git
cd mopac_gpu

# 3. configure and build (CUDA_ARCHS=native compiles for the GPU in this machine)
cmake -S . -B build -G Ninja -DGPU=ON -DCUDA_ARCHS=native -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mopac --parallel

# 4. the program
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
`SEED=`) and the input chemistry check work there too, just without the GPU speed.

### Check the installation

```bash
cd tests
python3 check_mozyme_gpu_tolerance.py ../build/mopac \
    ../benchmarks/publication_inputs/mop/protein_crambin_1crn.mop \
    ../benchmarks/publication_inputs/mop/protein_crambin_1crn_opt.mop \
    --handback ../benchmarks/publication_inputs/mop/protein_crambin_1crn_denout.mop
```

It runs crambin on the CPU and on the GPU and must end with `all checks passed`.

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
finished by the CPU code (correct result, CPU speed); see [Limitations](#8-limitations).

> The PDB must contain hydrogen atoms. Crystal structures usually do not; add them first with
> `ADD-H` ([example 6.2](#62-prepare-a-pdb-from-the-protein-data-bank)).

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
  orbitals already on the GPU, so a step of a 6 700-atom protein takes about 1 second.
- **Automatic fallback.** If a step cannot run on the GPU (for example an SCF that does not
  converge within its iteration budget), MOPAC hands the current state to the ordinary CPU code,
  which finishes the job. The answer is still correct.
- **Off switch.** The `NOGPU` keyword or the environment variable `MOPAC_NOGPU=1` gives the
  original CPU program, bit for bit.

---

## 5. New keywords

These keywords are **new in this fork**. They follow MOPAC's usual form (`WORD=value`) and are
listed in the output header like every other keyword.

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
| `MOPAC_FORCEGPU=1` (environment) | Use the GPU even if the input says `NOGPU`. |

### Input chemistry check (automatic, no keyword needed)

Every PDB-derived input is checked before the calculation. The result appears as a block:

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

**Errors stop the job.** Add the existing MOPAC keyword `LET` to run anyway. Why this matters: in
the PDB entry 1G6X two half sulfates on a symmetry axis made the SCF take 112 iterations and give a
different energy on every GPU run; after removing them it converges in 29.

---

## 6. Examples

All examples are complete input files. Save the lines in a file, e.g. `job.mop`, put the PDB file
in the same directory and run `mopac job.mop`. Lines are shown exactly as they go in the file.

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

Replace `PM7` by the method. All of these were validated on the GPU (energies within 0.03 kcal/mol
of the CPU on crambin and DNA):

```
PM6-D3H4 MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
AM1 MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
RM1 MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
PM3 MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
MNDO MOZYME 1SCF GEO_DAT="1crn_hydrogenated.pdb"
```

(one method per input file, each followed by a title line and an empty line).

### 6.4 Implicit water (COSMO)

`EPS=78.4` is the dielectric constant of water.

```
PM7 MOZYME 1SCF EPS=78.4 GEO_DAT="1crn_hydrogenated.pdb"
Crambin in implicit water

```

### 6.5 Geometry optimization

```
PM7 MOZYME CYCLES=200 GEO_DAT="1crn_hydrogenated.pdb" PDBOUT
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
PM6-D3H4 MOZYME EPS=78.4 CYCLES=300 GEO_DAT="protein_hydrogenated.pdb" PDBOUT
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

### 6.13 The whole workflow from a PDB code (Python)

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

## 7. Accuracy

The GPU SCF is not bit-identical to the CPU (the order of the parallel orbital rotations varies),
so energies differ at the 0.01 kcal/mol level between runs. The acceptance criterion is
|ΔHf| ≤ 0.05 kcal/mol against the CPU and a gradient RMS difference ≤ 10⁻³ kcal/mol/Å. Measured on an
A100:

| Check | GPU − CPU |
|---|---|
| Crambin / ubiquitin / DNA 1BNA, single point | +0.003 / −0.002 / +0.003 kcal/mol |
| 1G6X, 1EZG, 1RNB, 1C3W single points | +0.001, −0.007, −0.002, −0.004 kcal/mol |
| MNDO, AM1, PM3, RM1, PM6, PM6-D3H4, PM7 (crambin and DNA) | all within 0.024 kcal/mol |
| Gradient vs CPU (crambin) | RMS 5·10⁻⁸, max 3·10⁻⁷ kcal/mol/Å |
| Adenylate kinase after 50 GPU optimization cycles, same geometry on CPU and GPU | 0.009 kcal/mol |
| Crambin after 100 GPU optimization cycles, same geometry | 0.004 kcal/mol |
| Dynamics, crambin, 19.5 fs: largest integration ERROR | GPU 0.24, CPU 0.40 kcal/mol |

Optimization and dynamics trajectories themselves diverge between CPU and GPU (the tiny SCF
differences are amplified, as in any chaotic dynamics); compare energies at the same geometry, or
statistical averages.

---

## 8. Limitations

- **No NPT.** MOZYME treats a finite molecule; with implicit solvent there is no box and no
  pressure. NVE and NVT are available.
- **Time step.** `DRC` chooses its own step, about 0.1 fs when hydrogen atoms move. One picosecond
  of crambin is about 10 000 steps (≈ 20 minutes on an A100); 1 ps of a 7 000-atom protein about
  2.5 hours.
- **NVE after NVT** restarts from the last NVT geometry with fresh velocities at the same
  temperature: MOPAC cannot read velocities together with a PDB geometry.
- **Runs on the CPU instead of the GPU:** periodic systems, pKa calculations, and the programming
  interface (`libmopac` API). COSMO implicit water does run on the GPU (crambin: 2.1 s vs 33 s on the
  CPU, energies within 0.002 kcal/mol).
- **Reproducibility.** GPU runs are not bit-reproducible; `SEED` fixes the random numbers but not
  the order of the parallel rotations.
- **Inputs.** Hydrogens are required; broken crystal fragments are refused (see the chemistry
  check).

---

## 9. Repository map, tests and documentation

| Path | Content |
|---|---|
| `src/gpu/mozyme_scf_context.cu` | Resident SCF on the GPU (kernels and loop) |
| `src/gpu/mozyme_pair_*.cu(h)`, `dh_dispersion.cu`, `hbond_dh_plus.cu` | Core matrix, gradient, dispersion and H-bond kernels |
| `src/MOZYME/mozyme_gpu_*.F90`, `iter_for_MOZYME.F90` | Fortran side of the GPU SCF |
| `src/reactions/drc.F90`, `drc_ensemble.F90` | Dynamics; `TEMPERATURE=`, `BUSSI=`, `SEED=` |
| `src/chemistry/input_chemistry_check.F90` | Input chemistry check |
| `tests/check_mozyme_gpu_tolerance.py` | Acceptance test (CPU vs GPU, gradient, fallback, `NOGPU`) |
| `tests/check_mozyme_gpu_production.py` | Long optimization and dynamics check |
| `scripts/published_benchmark.py`, `scripts/check_mozyme_gpu_methods.py` | Benchmark table, all methods |
| `scripts/mozyme_md_workflow.py` | PDB → optimization → NVT → NVE |
| `benchmarks/publication_inputs/` | The benchmark proteins, hydrogenated, and their input files |
| `docs/MOZYME_GPU_USER_GUIDE.md` | Full user guide with all measurements |
| `docs/BACKGROUND.md` | Theory, earlier GPU work (conventional SCF, HMTR optimizer) |

Everything else is standard MOPAC; its manual at [openmopac.net](http://openmopac.net) describes all
other keywords.

---

## 10. Citing and license

This fork is based on MOPAC (Stewart Computational Chemistry / Virginia Tech, Apache License 2.0;
see `LICENSE` and `NOTICE`). Please cite MOPAC as described in `CITATION.cff`
(doi: [10.5281/zenodo.6511958](https://doi.org/10.5281/zenodo.6511958)) and, for the thermostat,
G. Bussi, D. Donadio and M. Parrinello, *J. Chem. Phys.* **126**, 014101 (2007).

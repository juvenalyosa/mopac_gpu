# MOZYME on the GPU: user guide

MOPAC's linear-scaling MOZYME method (proteins, DNA/RNA, large complexes) runs its SCF, the
one-electron matrix, the analytic and finite-difference gradients and the PM6-DH/PM7 corrections
on an NVIDIA GPU. Nothing has to be set: a `MOZYME` job on a capable device takes the GPU path,
and any job or environment can switch it off.

## Requirements

- NVIDIA GPU with compute capability 7.0 or newer (Volta, Turing, Ampere, Hopper). All GPU
  arithmetic is double precision; the numbers below are from an A100 (40 GB).
- CUDA toolkit 12 and a CMake build with `-DGPU=ON` (see `docs/GPU_GUIDE.md`, section "Build").
- Memory: the resident SCF keeps the packed density, Fock and LMO arrays on the device; a
  6700-atom protein uses about 2 GB of device memory.

## Running

Use an ordinary MOZYME deck, for example a PDB read through `GEO_DAT`:

```
PM7 GEO_DAT="protein.pdb" MOZYME 1SCF
```

or a geometry optimization:

```
PM7 GEO_DAT="protein.pdb" MOZYME CYCLES=200
```

The output reports `[MOZYME GPU] resident SCF enabled by default (disable with NOGPU or
MOPAC_NOGPU=1)` and one `[MOZYME GPU SCF] status=success ...` line per SCF.

Switching the GPU off, for a job or for a whole environment:

- keyword `NOGPU` in the deck, or
- `MOPAC_NOGPU=1` (or `MOZYME_GPU_OFF=1`) in the environment.

Either one disables every GPU helper (resident SCF, hcore, gradient, dispersion, hydrogen-bond
correction) and reproduces the CPU result bit for bit. `MOPAC_FORCEGPU=1` in the environment
overrides the keyword.

Inputs from the PDB must be complete: heavy-atom-only structures do not converge in MOZYME on
the CPU either. Run `ADD-H` first (`scripts/hydrogenate_publication_benchmark_inputs.py` does it
for the benchmark decks).

MOPAC checks PDB-derived inputs before the calculation (`INPUT CHEMISTRY CHECK` in the output).
Errors stop the job unless `LET` is present:

- a group of several atoms on a crystallographic special position (`REMARK 375`): the file holds
  only part of it, the rest is a symmetry mate;
- an incomplete small group (SO4, PO4, NO3, EDO, GOL, ACT, ...);
- a formal charge of ±2 or more on H, B, C, N, O or F in the Lewis structure.

Warnings only: amino acids with missing side-chain atoms, partial occupancy, all ARG/LYS/ASP/GLU/HIS
neutral (use `ADD-H SITE=(IONIZE)` for pH ~7) and sulfate/phosphate in acid form. The check exists
because of 1G6X: two half sulfates on special positions became "SO2(2-)" fragments, and the SCF
took 112 iterations (CPU) and gave run-dependent energies (GPU). Without them it converges in 29.

## Molecular dynamics

`DRC` runs the dynamics (adaptive time step, of the order of 0.1 fs with hydrogens; `T-PRIORITY=0.5` prints a
row every 0.5 fs and the trajectory is written to `<name>.xyz`). Two ensembles:

- `DRC TEMPERATURE=300`: Maxwell-Boltzmann velocities at 300 K (zero net momentum, exact kinetic
  energy), then constant energy (NVE).
- `DRC TEMPERATURE=300 BUSSI=100`: the same start plus the Bussi-Donadio-Parrinello stochastic
  velocity rescaling thermostat (NVT, canonical ensemble), time constant in fs (`BUSSI` alone: 100 fs).
  The energy exchanged with the bath is booked separately, so the ERROR column stays the integration
  error.

`SEED=n` sets the random seed. There is no barostat (NPT): MOZYME treats a finite molecule, and with implicit
solvent there is no box. Implicit water is COSMO (`EPS=78.4`). `scripts/mozyme_md_workflow.py` and
`colab/mozyme_md_colab.ipynb` chain PDB preparation, optimization, NVT equilibration and NVE production.

## Vibrational frequencies and thermochemistry (FORCE, FORCETS, THERMO)

MOPAC builds the Hessian by finite differences of the gradient (two gradients per Cartesian coordinate),
so every displaced SCF must be converged far more tightly than for an energy or an optimization. With the
MOZYME default criterion (`SCFCRT=0.01`) the zero-point energy of chignolin came out 52 kcal/mol too high.
With `FORCE`, `FORCETS` or `THERMO` in the keywords, MOZYME therefore uses:

- `SCFCRT=0.000001` (printed as `SCF CRITERION = 0.1000E-05 (FORCE: ...)`), unless `SCFCRT` or `RELSCF`
  is given;
- `THRESH=1.D-15` for the LMO coefficients, unless `THRESH` or `RELTHR` is given. With the default
  1.D-13 the MOZYME SCF cannot converge below about 1e-5 eV (rotations smaller than sqrt(THRESH)
  cannot add atoms to the LMOs), so a 1e-6 criterion would never be met.

Against the conventional (non-MOZYME) SCF, CPU, same geometry:

| System | ZPE | RMS freq. ≥ 100 cm⁻¹ | S (modes ≥ 20 cm⁻¹) | G = ZPE + H − TS |
|---|---|---|---|---|
| Chignolin, 140 atoms (protein) | +0.05 kcal/mol | 0.2 cm⁻¹ | −0.6 cal/(mol K) (all modes) | +0.19 kcal/mol |
| PET oligomer, 76 atoms (polymer) | +0.04 | 0.15 | −0.2 | +0.13 |
| d(TpA), 64 atoms (DNA) | +0.05 | 0.18 | −0.2 | +1.2 (see below) |

The harmonic entropy of the modes below ~20 cm⁻¹ (residual translations and rotations, which should be
zero) is not reliable in either method: finite differences leave them at ±10 cm⁻¹, and whether one comes
out as +3 or −1 cm⁻¹ changes S by several cal/(mol K). In d(TpA) three such modes change sign and account
for all of the S and G difference. For free energies of large flexible molecules use the quasi-harmonic
treatment of your choice on the printed frequencies, or compare only systems with the same low modes.

Cost: the tight criterion makes each gradient about three times slower than the default. For large
systems compute the partial Hessian of the region that matters (`FORCETS` with `OPT("A25"=5)`: atoms within
5 Å of residue 25 of chain A) and spread the Hessian rows over several processes with
`scripts/mozyme_parallel_force.py` (several processes can share one GPU through NVIDIA MPS).

## What stays on the CPU

The GPU path silently hands the work back to the CPU code (same results, CPU speed) for:

- periodic systems (`id > 0`) and pKa calculations (COSMO solvation runs on the GPU: crambin 1SCF
  2.1 s vs 33 s on the CPU, |ΔHf| 0.001 kcal/mol);
- the per-bond hydrogen-bond printout (`0SCF` / `PRT` with `DISP`);
- sparkles and any atom without device parameters in the gradient and hcore pair kernels;
- an SCF that does not converge on the device (iteration budget, PLS restart): the CPU continues
  from the device state.

## Accuracy

The GPU SCF is not bit-identical to the CPU one: the DIAGG rotation order on the device is
nondeterministic, and the initial LMO construction on the device also varies discretely between
runs (the same candidate pairs, slightly different first-iteration sums), so heats of formation
vary at the 0.01 kcal/mol level between runs (about 0.05 at 7000 atoms). The
acceptance criterion is |ΔHf| ≤ 0.05 kcal/mol against the single-core CPU run, and gradients
within 1e-3 kcal/mol/Å RMS. Measured (A100):

| Check | Result |
|---|---|
| Crambin 1SCF (642 atoms) | +0.002 kcal/mol |
| Ubiquitin 1SCF (1231) | −0.001 |
| DNA dodecamer 1BNA 1SCF (781) | +0.003 |
| Adenylate kinase apo 1SCF (6689) | within 0.1 (the CPU reference itself is converged to ~0.1 at this size) |
| GPU gradient vs CPU finite differences (crambin) | RMS 5e-8, max 3e-7 kcal/mol/Å |
| hcore, dispersion, H-bond energies vs CPU | 8e-12 eV, 4e-12 kcal/mol, 1e-12 kcal/mol |
| Crambin after 100 GPU optimization cycles, re-evaluated at the same geometry on the CPU | −3468.5422 (CPU) vs −3468.5459 (GPU), 0.004 kcal/mol |
| Crambin 1SCF, other methods: MNDO, AM1, PM3, RM1, PM6, PM6-D3H4 | within 0.024 kcal/mol (PM3 +0.023, PM6-D3H4 −0.011, others ≤ 0.004) |
| DNA dodecamer 1BNA 1SCF, the same seven methods | within 0.005 |
| 1G6X, 1EZG, 1RNB, 1C3W 1SCF (944 to 4473 atoms) | +0.001, −0.007, −0.002, −0.004 |
| Water cluster, 1000 H2O | +0.034 |
| Adenylate kinase apo, 50-cycle GPU optimization, final geometry re-evaluated on the CPU | −35936.8859 (CPU) vs −35936.8946 (GPU), 0.009 kcal/mol; all 100 SCFs resident |
| Molecular dynamics (DRC), crambin, 40 points / 19.5 fs | energy conservation max \|ERROR\| 0.24 kcal/mol (GPU) vs 0.40 (CPU) |
| Molecular dynamics (DRC), adenylate kinase apo, 20 points / 9.5 fs | max \|ERROR\| 1.49 kcal/mol = 5.6e-4 of the kinetic energy (CPU crambin: 1.4e-3) |

Optimization trajectories diverge between CPU and GPU (the rounding-level SCF differences are
amplified by the line search), so cycle-by-cycle heats are not comparable; only energies at the
same geometry are. `tests/check_mozyme_gpu_tolerance.py` (CTest `mozyme-gpu-tolerance`) runs all
of these checks on crambin, including a forced CPU hand-back and the `NOGPU` kill switch.

## Speed

Wall-clock times on an A100 (40 GB) against the same binary with `NOGPU`. The CPU reference is a
single core: MOZYME has no OpenMP, so a 16-core CPU would be roughly 5 to 10 times slower than
the GPU rather than the 20 to 60 times shown here.

| Job | CPU | GPU | Speedup |
|---|---|---|---|
| Crambin 1SCF (642 atoms) | 28.6 s | 1.13 s | 25x |
| Ubiquitin 1SCF (1231) | 41.7 s | 1.35 s | 31x |
| DNA dodecamer 1BNA 1SCF (781) | 24.6 s | 1.13 s | 22x |
| Adenylate kinase apo 1SCF (6689) | ~600 s | 7.2 to 7.6 s | ~80x |
| 1G6X 1SCF (944) | 30.2 s | 1.2 s | 25x |
| Barnase 1RNB 1SCF (1778) | 71.0 s | 1.8 s | 39x |
| Antifreeze protein 1EZG 1SCF (2064) | 130.1 s | 2.5 s | 52x |
| Bacteriorhodopsin 1C3W 1SCF (4473) | 343.3 s | 4.5 s | 76x |
| Water cluster 1000 H2O 1SCF (3000 atoms) | 52.9 s | 1.7 s | 31x |
| Water cluster 7052 H2O 1SCF (21156 atoms, 42,312 orbitals) | (not measured) | 25.7 s | |
| Crambin optimization, 3 cycles | 48 s | 1.5 s | 32x |
| Crambin optimization, 100 cycles | 1042 s | 17.7 s | 59x |
| Crambin, one warm optimization cycle | 6.9 to 8.5 s | 0.15 to 0.17 s | ~45x |
| Adenylate kinase apo, optimization, 3 cycles | (not measured) | 8.6 s | |
| Adenylate kinase apo, optimization, 50 cycles | (not measured) | 57 s (1.0 s per cycle) | |
| Adenylate kinase apo, one warm optimization cycle | (not measured) | 0.88 s | |
| Crambin molecular dynamics (DRC), 195 steps | 1273 s | 21.9 s | 58x |
| Adenylate kinase apo molecular dynamics (DRC), 95 steps | (not measured) | 87 s (0.9 s per step) | |

In the small systems about one second is fixed cost (MOPAC start-up, PDB reading, the initial LMO
construction on the device); the SCF itself is 0.35 s for crambin. For optimizations and
molecular dynamics the number that matters is the warm cycle: 0.15 s for crambin, 0.88 s for a
6700-atom protein, of which about 55 % is the SCF (7 iterations of about 68 ms), 13 % the
one-electron matrix, 13 % the gradient, 7 % the two-electron integral pack, and the rest
dispersion, hydrogen bonds and host/device transfers. Between geometry steps the LMOs and the
density stay on the device (the host tidy pass is skipped and the post-SCF corrections are
evaluated once per geometry, for the energy and the gradient together).

## Profiling and debugging

- `MOPAC_MOZYME_SECTION_PROFILE=1` prints per-section timers (`[PROFILE] MOZYME_SECTION ...`) and
  `MOPAC_GPU_PROFILE=1` the per-stage device timings; `scripts/mozyme_section_profile.py <mopac>
  <deck> --modes cpu,resident` tabulates them side by side.
- `MOPAC_GPU_GRAD_CHECK=1`, `MOPAC_GPU_HCORE_CHECK=1`, `MOPAC_GPU_DISP_CHECK=1` evaluate the GPU
  and CPU versions of the gradient, hcore, dispersion and H-bond terms and print their
  differences.
- The remaining `MOPAC_MOZYME_*_GPU` variables are development overrides; `docs/GPU_GUIDE.md`
  lists them. `MOPAC_MOZYME_SCF_STRICT_RESIDENT=1` is the fail-closed proof mode used to develop
  the resident SCF (it aborts instead of handing anything back to the CPU) and is not meant for
  production runs.
- `colab/mozyme_diagg_profile_colab.ipynb` builds and runs the whole benchmark and test set on a
  Colab GPU.

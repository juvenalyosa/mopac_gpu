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

## What stays on the CPU

The GPU path silently hands the work back to the CPU code (same results, CPU speed) for:

- periodic systems (`id > 0`), COSMO solvation and pKa calculations;
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
| Crambin optimization, 3 cycles | 48 s | 1.5 s | 32x |
| Crambin optimization, 100 cycles | 1042 s | 17.7 s | 59x |
| Crambin, one warm optimization cycle | 6.9 to 8.5 s | 0.15 to 0.17 s | ~45x |
| Adenylate kinase apo, optimization, 3 cycles | (not measured) | 8.6 s | |
| Adenylate kinase apo, one warm optimization cycle | (not measured) | 0.88 s | |

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

# GPU Gradient Roadmap

This document captures the work needed to make the DCART gradient path run on the
GPU.  It distils the current Fortran workflow into GPU-ready components,
outlines the Stage 1 prototype, and records the validation plan we will follow
before the CUDA implementation replaces the CPU fallback.

## 1. CPU DCART Workflow (Current Behaviour)

The existing Cartesian gradient is built in `dcart.F90` and depends on several
auxiliary routines:

1. **Input preparation (`dcart`)**
   - Zero/initialise `dxyz`, fetch AO populations (`chrge`/`chrge_for_MOZYME`),
     assemble the `qbld` charge deficit vector, and decide whether the GPU route
     is allowed (`resident_scf`, `MOPAC_GPU_GRAD`, etc.).
   - Allocate the `gpu_grad_pair` lists via `gpu_build_gradient_pairs` when the
     GPU path is requested.

2. **Finite-difference outer loop (`dcart_build_scf_gradient_cpu`)**
   - Iterate over all atom pairs (including lattice replicas for periodic
     systems).  For each pair:
     - Construct AO-block snapshots (`pdi/padi/pbdi`, `cdi`, `ndi`) from the
       global packed density arrays `p`, `pa`, `pb`.
     - Call `dhc` / `h1elec` / `rotate` / `point` to obtain the one- and
       two-electron integral derivatives at displaced geometries.
     - Evaluate the Coulomb fallback (`derp`) when interatomic distances exceed
       cutoffs.
     - Perform the finite-difference accumulation (`aa` vs `ee`) and add the
       contributions to both atoms in `dxyz`.
     - Apply MOZYME-specific logic (`jopt`, `part_dxyz`) when localised orbitals
       are in use.

3. **Downstream consumers**
   - `dcart` writes the gradient table for FORCE/DCART output, and `deri1`
     consumes the same data path when analytical derivatives are needed.  Any
     GPU implementation must preserve the packed AO layout expected by DIIS and
     MOZYME helpers.

### Key dependencies to mirror on the GPU

- `dhc` interaction with `h1elec`, `rotate`, `point`: these routines build the
  derivative integrals used in the finite difference.  Their logic (especially
  the lattice loop and point-charge shortcut) must be available on device.
- Packed density/Fock access (`p`, `pa`, `pb`, `ptot`): currently host only; we
  already keep resident copies of the density/Fock matrices through the “keep”
  cache in `fock_kernels.cu`.
- Finite-difference bookkeeping (`aa`, `ee`, `chnge`, `chnge2`, `const=fpc_9`).
  These are scalar operations but need to be evaluated in the same order on the
  GPU to guarantee identical results.

## 2. Stage 1 GPU Design (Resident Reuse Prototype)

Goal: provide a CUDA path that mirrors the CPU algorithm for near-field pairs
while reusing resident density/Fock buffers, but still falls back to Fortran for
unsupported corner cases.

### Data structures

- **Resident density/Fock**: reuse existing device buffers populated by
  `mopac_cuda_fock2_keep`.  Expose read-only handles for the gradient kernel.
- **Pair metadata**: extend `gpu_grad_pair` (already built in `dcart`) with
  flags for near/far, lattice offsets, and AO span indices.  Copy the near-field
  slice to device for kernel launch.
- **AO density slices**: Stage 1 will fetch the required packed AO blocks on the
  host and upload them per batch.  Subsequent stages can construct them directly
  on device once the integral builders exist.

### Kernel sketch

```
__global__ void gradient_nearfield_kernel(
    int pair_count,
    const GradPairPod *pairs,
    const DeviceDensityView dens,
    const DeviceFockView fock,
    GradientAccum *out);
```

- Each thread handles one atom pair (`ii`, `jj`).  The kernel will:
  1. Reconstruct AO block views from `span_i_first/last`, `span_j_first/last`.
  2. Evaluate the finite-difference integrals (initially using device versions
     of `dhc`/`rotate` ported verbatim).
  3. Accumulate the resulting derivatives into a temporary per-pair buffer.
  4. Use atomics to add the contribution to the shared `out` array (size
     `3*numat`).

- **Host wrapper**:
  - Checks `MOPAC_GPU_GRAD_STAGE1` and resident mode before launching.
  - Copies `GradPairPod` array to device, zeros the gradient scratch, launches
    the kernel, and copies the reduced gradient back.
  - On failure or unsupported cases (far-field, MOZYME, UHF), falls back to
    `mopac_gpu_cart_gradient_cpu`.

### Fallback logic

- Far-field pairs (`flags` with point-charge indicator) and MOZYME cases stay on
  the CPU path until Stage 2.  The wrapper can split the pair list so the GPU
  handles only near-field terms and the CPU processes the rest.

## 3. Validation Plan

### Test decks

1. `examples/h2o_gpu_gradcheck.mop` (RHF, small molecule).
2. A larger closed-shell case (e.g., benzene from the suite).
3. Open-shell/UHF example (ensure spin-dependent logic is intact).
4. MOZYME/large protein once the dedicated path is implemented.

### Metrics

- For each case: compare GPU vs CPU `CARTESIAN COORDINATE DERIVATIVES` table.
- Acceptance: `max_abs ≤ 1e-6`, `rms ≤ 1e-7` for Stage 1.  Record tolerance in
  the suite so experimental runs can opt out.

### Harness updates

- Add an explicit env gate (`MOPAC_GPU_GRAD_STAGE1=1`) so regression runs can
  flip the CUDA path on/off without editing input decks.
- Extend `scripts/run_gpu_suite.sh` to run the gradient comparison twice when
  the flag is set: once for baseline, once for GPU, reporting both metrics.
- Provide a helper script (or extend `compare_grad.py`) to summarise per-case
  differences and print a warning when tolerances are exceeded.

### Manual sanity checks

- Cross-check heats of formation and energy components to ensure the GPU path
  doesn’t unexpectedly mutate the SCF state.
- Profile kernel timings on a medium-sized system to confirm the resident reuse
  actually avoids host/device copies.

---

With this baseline mapped out, the next step is to start porting the integral
routines to CUDA, implement the Stage 1 kernel + wrapper, and wire the validation
hooks into the test harness.

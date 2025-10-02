# GPU SCF Re-Architecture Plan

## Motivation and Goals
- Deliver numerically identical SCF energies/gradients to the reference CPU implementation.
- Execute the entire SCF micro-iteration loop on GPU, avoiding host/device ping-pong.
- Support both PM7 and MNDO-family Hamiltonians initially; extendable to MOZYME/linear-scaling paths.
- Preserve existing Fortran interfaces so that downstream modules (gradients, properties) remain unchanged.
- Provide robust debugging, determinism controls, and graceful fallback to CPU paths.

## Current Pain Points
- `iter.F90` performs the SCF loop on CPU, invoking piecemeal GPU helpers (density build, diagonalisation).
- Resident caches for density/Fock matrices are fragile and diverge from the CPU state, breaking gradients.
- DIIS and Pulay mixing still run on the host (`iter_C` work arrays), forcing density fetches each iteration.
- Multiple environment-variable toggles complicate reproducibility (`MOPAC_RESIDENT_SCF`, `MOPAC_GPU_DIIS`, etc.).

## High-Level Design
1. **GPU SCF Driver Layer (C++/CUDA)**
   - New module `src/gpu/scf_driver.cu` exposes `mopac_cuda_scf_run(context)` to Fortran.
   - Context struct bundles Hamiltonian data, integral evaluators, initial density guess, convergence tolerances,
     and output slots for converged density, orbital energies, total energy, and diagnostics.
   - Uses existing GPU kernels (`fock_kernels.cu`, `hmtr_optimizer.cu`, `gpu_transform_interfaces`) but orchestrates
     them centrally.

2. **Device-Resident State Objects**
   - `FockState`: packed/full Fock matrices, two copies for DIIS error accumulation.
   - `DensityState`: current, previous, and extrapolated densities (alpha/beta when UHF).
   - `OverlapState`: `S`, `S^{-1/2}`, and transformation buffers prepared once per job.
   - `EigenState`: device workspaces for diagonalisation (cuSolver) and eigenvector storage.
   - All objects manage device memory with deterministic allocation; they expose handles to Fortran for diagnostics.

3. **SCF Iteration Loop (GPU)**
   - Sequence per iteration:
     1. Build Fock (existing GPU kernels already support resident density).
     2. Solve generalised eigenproblem: `F C = S C E` via cuSolver (single or multi-GPU).
     3. Occupy orbitals -> update density fully on device, supporting fractional occupations.
     4. Compute electronic energy, `Tr[(F + H_core) P]`, entirely on GPU with reductions.
     5. DIIS acceleration: implement DIIS solver in CUDA (reuse linear algebra libs), maintain history slices.
     6. Convergence tests (density RMS, energy change) executed on GPU; only scalars copied to host.
   - Provide hooks to dump intermediate matrices for debugging when `MOPAC_GPU_SCF_DEBUG` is set.

4. **CPU Interoperability Layer (Fortran)**
   - Introduce module `gpu_scf_interfaces.F90` mirroring current `gpu_density_interfaces`.
   - `iter.F90` detects GPU mode and, instead of executing host SCF, prepares context arrays and calls
     `mopac_cuda_scf_run`. On success it receives converged density/Fock, updates legacy globals, and exits early.
   - Fallback: on failure or when GPU disabled, existing CPU path runs unchanged.

5. **Determinism and Precision Controls**
   - Support both double precision and mixed precision (optional) builds; default to double.
   - Deterministic reductions achieved with segmented parallel reductions and explicit ordering.
   - Environment variable `MOPAC_GPU_SCF_DETERMINISTIC=1` forces deterministic kernels at small performance cost.

6. **Diagnostics and Logging**
   - Unified logging macro writing to `stderr` with `[GPU SCF]` prefix; severity levels controlled via env var.
   - Device/host parity checks selectable via `MOPAC_GPU_SCF_DEBUG=matrix` to dump max/rms discrepancies.
   - Profiling counters (iteration count, time spent in Fock, diagonalisation, DIIS) accessible through existing
     GPU profile framework.

## Integration Roadmap
1. **Phase 0 – Instrumentation Baseline**
   - Add lightweight wrappers in `iter.F90` capturing all arrays required for SCF (H core, overlap, AO integrals).
   - Establish regression tests comparing CPU and GPU SCF for small molecules (H2O, NH3, benzene) and edge cases
     (open-shell, fractional occupation).

2. **Phase 1 – GPU Density/Fock Residency**
   - Promote current resident caches into `DensityState`/`FockState` classes; ensure pack/unpack consistency.
   - Provide Fortran accessors for diagnostics (`mopac_cuda_fetch_density` etc.) through new interface module.

3. **Phase 2 – GPU DIIS and Convergence**
   - Implement DIIS error matrix assembly on device using cuBLAS; support both RHF and UHF simultaneously.
   - Validate against CPU DIIS for identical inputs by mirroring residual vectors.

4. **Phase 3 – Full GPU Iteration Driver**
   - Replace host SCF loop with GPU driver call in `iter.F90` when `lgpu` and capability >= target compute.
   - Support optional CPU fallback each iteration for mixed debugging (env flag `MOPAC_GPU_SCF_VALIDATE`).

5. **Phase 4 – Extended Features**
   - Re-enable resident gradients leveraging converged device densities.
   - Integrate with MOZYME semi-empirical linear-scaling SCF (sparse matrices) where applicable.
   - Explore multi-GPU scaling for large systems using cuSolverMg and domain decomposition.

## Testing Strategy
- Expand `compare_grad.py` harness to accept GPU SCF path and record per-iteration diagnostics.
- Add CTest entries: `ctest -R scf_gpu_h2o`, `-R scf_gpu_open_shell`, running with `MOPAC_FORCEGPU=1`.
- Continuous integration stub: CPU-only environments run host SCF; GPU CI executes GPU tests with Gold files.
- Provide deterministic random seed injection for initial density perturbations (`MOPAC_GPU_SCF_SEED`).

## Deliverables for First Implementation Slice
- New CMake targets compiling `scf_driver.cu` and exposing Fortran bindings.
- Refactored `iter.F90` entry that delegates to GPU driver when available.
- Device DIIS implementation validating on RHF water molecule within 1e-8 RMS density difference vs CPU.
- Documentation updates (this file + `docs/dev/gpu_scf.md`) and developer checklist for debugging.

## Stub & Validation Hook (current status)
- `src/gpu/scf_driver.cu` currently provides a stub implementation that logs invocations when
  `MOPAC_GPU_SCF_STUB_LOG` is set. The stub returns `false` while capturing the most recent status message
  via `mopac_cuda_scf_last_error`.
- `gpu_scf_interfaces` exposes `gpu_scf_run` and `gpu_scf_last_error` helpers so Fortran callers can exercise
  the path today without changing the legacy SCF loop. This enables incremental wiring and unit tests that
  assert the bridge works before the full GPU solver is ready.
- Developers can run `ctest -R gpu-scf-stub-smoke` (which sets `MOPAC_GPU_SCF_STUB_LOG=1`) to verify the
  binding and error plumbing until the real driver is implemented.
- In a full MOPAC run, export `MOPAC_GPU_SCF_EXPERIMENTAL=on` and launch `examples/test_dense_gpu_stub.mop`
  (which loads `examples/test_dense.pdb` via `GEO_DAT=`) to see the stub invoked from the Fortran SCF driver.

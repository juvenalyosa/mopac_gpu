# MOZYME GPU Reformulation Plan

## Current Finding

The existing MOZYME GPU path is not a production accelerator for proteins,
DNA, RNA, or other large sparse biomolecules.

The molecule benchmark showed:

- GPU device detection works.
- `MOZYME_GPU` can be enabled.
- MOZYME density blocks are too small for cuBLAS (`max_block <= 9` in the
  publication biomolecule inputs).
- The legacy MOZYME Fock GPU wrappers are unsafe on realistic inputs and are
  disabled by default.
- The global dense-SCF GPU switch (`lgpu`) is too coarse for MOZYME.  MOZYME
  must disable generic GPU paths unless a MOZYME-specific production kernel can
  run.

This means the production problem is not dense matrix multiplication. It is a
many-small-block sparse workload.

## Production Policy

`MOZYME_GPU` may remain enabled for profiling and future planning, but legacy
Fock GPU execution is opt-in only through `MOPAC_MOZYME_FOCK_GPU`.
The experimental `check_gpu` MOZYME LMO retry path is also opt-in through
`MOPAC_MOZYME_CHECK_GPU`; it does not launch CUDA work and should not be used
as evidence of GPU acceleration.

Publication and production benchmarks must stop during preflight unless they
observe a real, successful GPU kernel. They must not spend hours on a CPU-bound
run labeled as GPU.
When `MOPAC_MOZYME_GPU_PREFLIGHT_STOP=1` is set, the MOZYME plan exits
immediately if no production GPU work is eligible.
The same preflight now reports a Fock workload plan:

- one-center tasks, two-center sparse atom-pair tasks, and skipped pairs,
- common block-shape counts (`4x4`, `4x1`, `9x4`, `9x9`),
- candidate GPU tasks versus production GPU tasks.

For the current implementation, production GPU Fock coverage has three
validated slices. `MOPAC_MOZYME_FOCK1_BATCH_GPU` batches independent one-center
atom Fock tasks. `MOPAC_MOZYME_FOCK2_4X1_BATCH_GPU` batches the common `4x1`
and `1x4` two-center atom-pair branches. The resident sparse Fock path now uses
an optimized `4x4` branch plus the generic exact `focd2z` formula for MOZYME
basis-block sizes `{1,4,9}`, including the `9x4` and `9x9` d-orbital
buckets that previously forced CPU fallback.
For direct-integral resident packing, the current W-only GPU generators cover
the same `{1,4,9}` resident basis range. The s/p path remains separate from the
s/p/d path; any future direct basis shape outside that range is still reported
as `unsupported_direct_basis` and sets resident Fock executable work to zero
unless full resident coverage is available.

The planner has two phases:

1. Early policy gate, after orbital block sizes are known and before MOZYME
   sparse setup.  This can safely disable `lgpu`/`MOZYME_GPU`.
2. Full profiler, after `fillij(.false.)`, when the real sparse interaction map
   exists.  This is the only phase allowed to inspect `nijbo`.

The implementation keeps separate state for:

- `mozyme_gpu_requested`: user/default policy requested MOZYME GPU work.
- `mozyme_gpu`: production-safe MOZYME GPU execution is enabled.
- `mozyme_gpu_plan_ready`: the sparse workload has been profiled.
- `mozyme_gpu_disable_reason`: numeric reason code for automated reports.

Reason codes are:

- `0`: none.
- `1`: not requested.
- `2`: no usable GPU device.
- `3`: no production MOZYME GPU work is eligible for this molecule.
- `4`: disabled by device or policy gate.

Profiling/preflight builds emit a machine-readable block delimited by
`[MOZYME_GPU_PLAN_BEGIN]` and `[MOZYME_GPU_PLAN_END]`.  The benchmark parser
uses that block first and only falls back to older human-readable log lines.

## Target Architecture

The GPU implementation should create a persistent MOZYME GPU plan before SCF:

1. Build atom-pair and LMO-pair task lists.
2. Bucket tasks by block shape: `1x1`, `4x1`, `4x4`, `9x4`, `9x9`, and rare
   larger cases.
3. Move invariant geometry, orbital ranges, parameters, and task metadata to
   the GPU once.
4. Keep density/Fock buffers resident across SCF iterations.
5. Launch fused batched kernels over thousands or millions of small tasks.
6. Accumulate Fock contributions with either atomics, coloring, or staged
   reductions, depending on the conflict rate.

## First Kernel To Replace

Start with the two-center MOZYME Fock contribution because it dominates the
end-to-end biomolecular runs. The first production kernel should be a grouped
atom-pair kernel, not a cuBLAS call:

- one thread block or warp group per atom-pair task,
- specialized branches for common `4x4`, `4x1`, and `9x4` cases,
- one launch per bucket,
- no host/device transfers inside the inner loop.

Only after this kernel is stable should density construction be moved to a
similar batched sparse kernel.

The implemented slices are still intentionally conservative.  They use exact
FP64 arithmetic and CPU fallback, and they scatter compact GPU deltas back to
the packed Fock matrix.  The final biomolecule speedup still depends on keeping
density/Fock buffers resident across the SCF loop and validating the full
resident-SCF path on publication molecules so host/device transfers do not
dominate.

## Resident SCF Boundary

The first whole-SCF integration point is now present but intentionally disabled
for production. `iter_for_MOZYME` first tries a strict full-active-space
initial resident setup before CPU `tidy`, density, and Fock setup, and then
tries resident iterations before the CPU LMO
`check`/`eimp/diagg/cnvgz/buildf` sequence. Initial LMO construction now has an
experimental GPU helper for `makvec`; canonicalization and metadata setup
(`tidy` and `setupk`) remain tracked by the strict readiness contract. A
complete workflow claim passes only when the CPU section timers are absent
because the resident path owned the work. Strict proof mode now requires
`[MOZYME GPU makvec] status=success` and aborts before CPU `makvec` or
`OLD_SCF` can bypass the GPU initial LMO construction.
The pre-`tidy` probe is intentionally non-consuming: if backend preparation
fails before CPU setup has built the conventional state, the driver does not
mark the SCF as blocked, so the existing post-setup resident attempt can still
run after the validated CPU setup path.

The boundary is gated by both `MOPAC_MOZYME_SCF_EXPERIMENTAL=1` and
`MOPAC_MOZYME_RESIDENT_SCF=1` (`MOPAC_MOZYME_SCF_GPU=1` is accepted as a
development alias). `MOZYME_GPU` or `MOPAC_FORCEGPU` alone must not enable this
path. The v26 CUDA ABI currently exports `setup`, `register_state`, `run`,
`status`, and `destroy` entry points. `register_state` is the hand-off for packed matrices
(`p`, `f`, `h`, `partp`, `partf`), density history (`pold`, `p1`, `p2`, `p3`),
diagonal indices, orbital counts, occupied and virtual sparse LMO lists,
FMO/IFMO storage, `nfmo`, AO range maps (`nfirst/nlast`), DIAGG control
(`idiagg`, density `indi`, `shift`, `thresh`, `selcon`, and DIAGG2 rotation
constants), explicit active and total occupied/virtual counts, direct-COSMO
resident state for the `addfckz` correction, and eigenvalue
scratch. The current CUDA context validates and stores these host pointers,
uploads resident shadow buffers, and runs resident stage kernels.
`run` can now advance multiple resident iterations when every resident substage
completes, including the resident LMO `check` normalization that CPU performs
at the start of each SCF iteration. Completed resident checkpoints are copied
back to Fortran only at
controlled boundaries: SCF convergence, iteration-budget return, or a clean
resident-step continuation. The PLS supervisor now performs its restart reset
inside resident CUDA state in strict proof mode instead of creating a host
checkpoint. This keeps the
validated CPU recovery paths available while the hot SCF stages remain resident
on the device.

The Fortran boundary is deliberately conservative. A requested resident-SCF
attempt emits one `[MOZYME GPU SCF] status=...` line per SCF attempt even when
profiling is disabled. Common fallback reasons include `experimental_gate_closed`,
`not_gpu_build`, `denout_checkpoint`, `solvent_fock`,
`resident_fock_partial_coverage`, `gpu_disabled`,
`state_incomplete`, `backend_not_ready`, `backend_missing_stages`, and
`backend_unsupported`.
`denout_checkpoint` is an immediate controlled fallback at the requested CPU
`.den` checkpoint; resident GPU iterations may run until just before that
observable checkpoint boundary.
ABI v30 applies the direct-COSMO `addfckz` Fock correction inside the resident
CUDA Fock stage, using a matrix-free GPU CG path and exported COSMO
preconditioner state. The status ABI reports COSMO Fock calls, matrix-vector
calls, CG iterations, surface dimensions, close-pair counts, final residual,
final COSMO energy terms, and resident CG control/convergence fields so Colab
proof artifacts can distinguish ordinary EPS/COSMO execution from a non-solvent
resident SCF run and reject host-synchronized CG control. `solvent_fock`
remains a hard resident-SCF fallback for solvent modes outside that direct
resident contract, currently LPKA.
`resident_fock_partial_coverage` is a hard resident-SCF fallback because strict
resident SCF requires every positive-orbital real Fock pair in the active plan
to be covered by the resident sparse device kernels. Real pairs with zero
orbital count are no-op pairs and do not count as CPU fallback coverage.
Partial initial setup uses explicit resident Fock plan slots: a full-plan slot
for the initial full Fock/energy build and a partial-plan slot for the
mode-specific `partf` rebuild before resident iterations continue.
CPU `makvec`, `OLDEN`/`OLDENS`, `OLD_SCF`, and `RE-LOC` have CPU bookend
timers, and `REORTH` is also tracked as a CPU bookend section rather than a
resident-SCF entry fallback reason. The strict report now also rejects CPU
`isitsc` and PLS restart timers when claiming a complete GPU workflow.
Strict readiness still detects and rejects a complete GPU SCF claim if those
sections run outside the resident backend, but the main driver no longer emits
`reason=relocalization_bookend` or `reason=reorth_bookend`.
`backend_missing_stages` means the diagnostic resident iteration reached the
implemented substages but cannot claim full-SCF ownership. Profiling adds
backend status details such as `code_name`,
`ready`, `resident`, `iterations`, density probes, and timing.

The CPU loop can be replaced only when `mopac_cuda_mozyme_scf_run` returns
`SUCCESS` and the status also reports the matching ABI version, `SUCCESS`,
`ready=1`, `resident=1`, a nonnegative CUDA `device_id`, `stage_missing=0`,
any PLS restart completed on GPU with final `pls_restart_required=0`, a final
iteration count greater than the incoming iteration count, and a usable
electronic energy. Only then
does Fortran update `ee` and the normal MOZYME energy bookkeeping before
entering the existing post-SCF cleanup. Any incomplete or ambiguous backend
status falls back to CPU.

The resident-SCF ABI now reports stage bitmasks: `stage_completed`,
`stage_required`, and `stage_missing`. These make partial progress explicit in
logs and machine-readable reports. The current backend can mark resident upload
plus LMO `check`, `eimp`, DIAGG rebuild/rotation, resident density rebuild,
resident ADDHB, full-coverage sparse Fock rebuild, `cnvgz`, `helecz`, and
`isitsc` as completed.
It reports `status=success` only when ISITSC convergence is reached on the
resident path and any resident PLS restart request was completed by the
device-side reset path.
Outside strict proof mode, an incomplete but successful resident iteration can
report `resident_step` so Fortran can continue through existing supervisor
logic. In `MOPAC_MOZYME_SCF_STRICT_RESIDENT=1`, `resident_step` is demoted to
unsupported by the driver and the Fortran loop aborts before CPU
SCF/body/bookend work, so the Colab proof cannot pass by silently continuing on
CPU.

The resident kernels include initial full-density/full-Fock setup, production
LMO `check`, `eimp`, DIAGG rebuild/rotation, density rebuild, ADDHB/DIAGG2
updates, full-coverage sparse Fock rebuild, `helecz`, `cnvgz`, and `isitsc`
stages.
`mopac_cuda_mozyme_scf_run` can copy the current packed
`p/h/f`, `partp/partf`, occupied and virtual LMO inputs, `fmo/ifmo`, `nfmo`,
density history, diagonal maps, `nfirst/nlast`, and `nijbo/iorbs` maps to the
device. It normalizes occupied and virtual LMO coefficients on device, runs
`eimp` into a resident screening buffer, applies the
`diagg1_construct`/`diagg2_rotate` sequence to resident LMO buffers, rebuilds
the resident packed density, initializes `f` with the same mode as `buildf`,
and can run the sparse Fock kernels directly on resident device buffers when
the existing Fock planner has certified full GPU pair coverage. It evaluates
the electronic energy reduction and convergence metrics, then reports
`energy_total`, `density_max`, `density_rms`, and `wall_ms` in the SCF status.
When the resident iteration is complete, it copies `p/f`, density history,
LMO coefficients, sparse LMO metadata, FMO/IFMO, `nfmo`, eigenvalue scratch,
DIAGG1/DIAGG2 saved state, `sumt`, `sumb`, `ijc`, and CNVGZ `pmax` back to the
canonical Fortran side at controlled boundaries. Device-to-host array copyback
uses temporary host buffers and commits to Fortran only after every CUDA copy
succeeds, so CPU fallback cannot resume from a mixed GPU/CPU state. If resident
ISITSC converges after either no historical PLS restart correction or a completed
device-side PLS restart reset, the backend returns `SUCCESS` with the full stage
mask completed and `stage_missing=0`; the Fortran final-density bookend then
reuses the resident GPU density and emits `final_density=current_resident`
instead of rebuilding that density on CPU. If a PLS restart remains pending
after the resident reset path, the backend reports `backend_pls_restart_required`
and fails closed before CPU SCF continuation.
Outside strict proof mode, incomplete resident progress can still return a clean
`resident_step` so the Fortran loop can skip the CPU duplicate of the completed
iteration and continue through existing convergence and recovery logic.
The strict readiness report also rejects runs that emit CPU MOZYME
state-mutating setup or bookend section timers outside the resident GPU
boundary, including `makvec`, `tidy`, `setupk`, initial density/Fock setup, `check`,
`OLDEN` load/density, `OLD_SCF`, `RE-LOC`, `eimp`, DIAGG, `cnvgz`, `isitsc`,
PLS restart, iterative Fock, final density rebuilds, and reorthogonalization.
Section profile markers must be present, `[MOZYME GPU makvec] status=success`
must be present, and raw CUDA/cuBLAS/cuSOLVER error markers are fatal. Those
timers must remain absent for a complete GPU SCF workflow claim; recovery and
unsupported bookend paths continue to fall back to CPU outside strict proof
mode, while partial initial setup is routed through dual resident Fock plan
slots.

Separately, standalone `eimp`, `density_batch`, `diagg1_construct`,
`diagg1_aocc`, `diagg1_avir`, `diagg2_rotate`, `diagg2_rotprep`, `cnvgz`,
`helecz`, and `isitsc` GPU stage calls run inside the existing CPU-owned SCF
loop when MOZYME GPU is active.
They can be controlled individually with
`MOPAC_MOZYME_EIMP_GPU=0`, `MOPAC_MOZYME_DENSITY_BATCH_GPU=0`,
`MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU=0`,
`MOPAC_MOZYME_DIAGG1_AOCC_GPU=0`, `MOPAC_MOZYME_DIAGG1_AVIR_GPU=0`,
`MOPAC_MOZYME_DIAGG2_ROTATE_GPU=0`, `MOPAC_MOZYME_DIAGG2_ROTPREP_GPU=0`,
`MOPAC_MOZYME_CNVGZ_GPU=0`, and `MOPAC_MOZYME_HELECZ_GPU=0`. The standalone
`isitsc` helper is opt-in with `MOPAC_MOZYME_ISITSC_GPU=1`. `eimp` writes the packed `p` scratch entries
consumed by `diagg`, `diagg1_construct` builds `fmo/ifmo`, pseudo-eigenvalues,
`nfmo`, and DIAGG control scalars on GPU, `diagg1_aocc` and `diagg1_avir`
remain as substage fallbacks, `density_batch` builds the sparse packed density
contribution for small MOZYME atom blocks, `diagg2_rotate` performs the sparse
LMO accept/retry rotation and copies the mutated LMO arrays back to Fortran,
`diagg2_rotprep` remains as a coefficient-prep fallback, `cnvgz` copies the
updated packed density and diagonal history back to the canonical Fortran
arrays before the next SCF step, `helecz` returns the GPU electronic-energy
reduction, and `isitsc` evaluates the scalar SCF convergence decision on the
GPU while the Fortran wrapper retains the saved convergence history. These are
real production offloads of individual SCF stages, not a complete SCF
replacement.
The Colab notebook exercises this boundary through the strict full-SCF
readiness probe and a direct-COSMO readiness probe by default. The long
publication molecule benchmark remains opt-in until those readiness artifacts
pass.
Benchmark reports must keep this boundary separate from production timing:
`backend_missing_stages` means the experimental hand-off completed only the
implemented resident substages; `fallback_cpu` means the boundary executed and
returned to CPU without taking over the iteration.
In strict mode the corresponding failure status is `strict_abort`, which fails
closed before CPU SCF continuation.
`success_pending_contract` means the raw backend emitted `status=success`, but
the report has not yet proven the strict resident-SCF contract. Only
`full_scf_gpu_status=complete` with `full_scf_gpu_ready=1`, contract version
`resident-scf-strict-final-resident-reorth-tidy-selmos-pls-reset-cosmo-direct-resident-cg-point-kind-cnvgz-active-or-noop-cpu-compare-explicit-proof-v58`,
zero parsed fallback counters, and the Colab same-input CPU companion
comparison artifact is evidence of a complete MOZYME SCF GPU run. None of the
intermediate states is evidence of
production speedup unless the same run also emits production MOZYME GPU kernel
success markers.

The backend may return success only after it has updated all CPU-visible state
that the post-SCF cleanup expects: density/Fock arrays, LMO coefficients,
DIAGG1 saved state, iteration count, total energy, convergence metadata, and
any diagnostics needed by the existing output path.
Those final/status host commits publish the completed resident result back to
MOPAC; they are not treated as CPU SCF compute fallback when the strict
readiness contract has otherwise passed with zero parsed fallback counters.

## Profiling Gate For The 10x Work

Before adding more GPU kernels, benchmark runs should enable
`MOPAC_MOZYME_SECTION_PROFILE=1`.  This emits cumulative markers such as
`[PROFILE] MOZYME_SECTION name=buildf calls=<n> ms=<total>`, which the molecule
benchmark turns into `mozyme_section_times.csv` and `mozyme_section_times.png`.

The first instrumented sections are `buildf`, `fock2z`, and the active `fz2` or
`fz2n` branch.  These separate the current resident sparse-Fock work from the
larger MOZYME SCF loop.  If these sections do not explain most wall time, the
next instrumentation layer should be added around `iter_for_MOZYME` calls to
`check`, `eimp`, `diagg`, `cnvgz`, `density_for_MOZYME`, `helecz`, and `isitsc`.
That profile determines which arrays must become GPU-resident before a 10x
speedup is technically plausible.

## Acceptance Criteria

A MOZYME GPU implementation is considered production-ready only when:

- GPU preflight sees `lgpu=T`, `MOZYME_GPU=T`, and at least one successful
  profiled MOZYME GPU kernel.
- CPU and GPU heats of formation agree within publication tolerance.
- No fallback or segmentation fault occurs on Crambin, Ubiquitin, DNA 1BNA,
  and adenylate kinase.
- End-to-end speedup is measured on the same Colab hardware/session.

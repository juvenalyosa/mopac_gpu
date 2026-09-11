GPU Build and Runtime Guide (CUDA)

This page summarizes how to build MOPAC with GPU support and the runtime switches that control GPU, multi‑GPU, and related behaviors.

Build
- CMake GPU: configure with `-DGPU=ON`.
- CUDA arch: set `-DCMAKE_CUDA_ARCHITECTURES=native` or an explicit list like `61;70;75;80;86;89;90`.
- BLAS/LAPACK: either let `find_package` discover system BLAS/LAPACK (default) or pass your own via `MOPAC_LINK`.
- OpenMP (THREADS keyword): enabled by default if found.

Minimal examples
- Single unified build (CPU+GPU):
  - `cmake -S . -B build -G Ninja -DGPU=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native`
  - `cmake --build build -j`
- GPU‑only build dir: `build-gpu` is fine; the source tree supports multiple build dirs.

MOZYME Resident SCF (production default)
- A `MOZYME` job on a GPU with compute capability >= 7.0 (Volta or newer) runs the resident SCF by default: the SCF loop (tidy, check, eimp, parallel diagg1/diagg2, density by atom-pair blocks, sparse Fock, cnvgz, helecz, isitsc) executes on the device, initial LMOs come from the GPU makvec and the Fock plan is packed in parallel. Per-geometry host routines (hcore, add_more_interactions, tidy/check bookends, OLD_SCF warm starts, gradients) still run on the CPU, so single points and geometry optimizations work without any environment setup. Measured on Colab (T4-class): crambin 1SCF 3.2 s vs 29 s CPU, ubiquitin 4.7 s vs 43 s, 3-cycle crambin optimization 8 s vs 47 s, heats within 0.01 kcal/mol.
- Disable with the `NOGPU` keyword or `MOPAC_NOGPU=1` / `MOZYME_GPU_OFF=1`; the defaults only fill in `MOPAC_MOZYME_*` variables the user left unset, so explicit settings still win. `MOPAC_MOZYME_SCF_STRICT_RESIDENT=1` keeps the old fail-closed proof mode, which aborts on any CPU setup work.
- Profile with `scripts/mozyme_section_profile.py <mopac> <deck.mop> --modes cpu,resident`; see "Verbose/Profiling" below.
- MOZYME gradients on the GPU: the defaults also set `MOPAC_MOZYME_GRAD_GPU=1`, which evaluates the pair loop of `dcart` (finite differences of the diatomic energy `dhc` plus the point-charge `delsta` term) on the device — one thread per interacting atom pair, with the overlap (`h1elec`), core-attraction/core-core (`rotate`/`ccrep`) and diatomic Fock/energy (`fock2`/`helect`) routines ported to device code (`src/gpu/mozyme_pair_*.cuh`). Non-periodic RHF sp-basis systems only; anything else (or any device failure) silently uses the CPU loop. `MOPAC_GPU_GRAD_CHECK=1` evaluates both and prints `[MOZYME GPU gradient] check ... max_abs_diff= rms_diff=` (the CPU result is kept); `--modes resident-gradcheck` in the profiler does this for every gradient of a run.
- Pairs involving d-orbital atoms (S, P, Cl, Si, transition metals in PM6/PM7) are evaluated by separate d kernels in both the gradient and hcore paths (full `rotatd` port with d integrals); `MOPAC_MOZYME_DPAIRS_GPU=0` sends them back to the CPU pass. Only sparkles (0 orbitals) remain on the CPU.
- `hcore_for_MOZYME` block pairs on the GPU: `MOPAC_MOZYME_HCORE_GPU=1` (also a default) evaluates the sp-sp interacting pairs of the one-electron matrix (`h1elec` off-diagonal block, `rotate` e1b/e2a diagonal blocks, core-core repulsion) with the same device routines; the CPU loop keeps the point-charge/dipole pairs, one-centre terms and d-orbital pairs. `MOPAC_GPU_HCORE_CHECK=1` prints `[MOZYME GPU hcore] check ... max_abs_dh= denuc=` against a CPU reference of the same pairs.
- DIAGG rotations (`diagg2`) use a lock-based sweep by default: one warp per virtual LMO walks its candidate pairs and holds a spin lock on the occupied LMO while rotating, with no grid-wide synchronisation (the previous cooperative sweep needed as many `grid.sync()` rounds as the largest LMO degree). Rotation order between different virtuals on the same occupied LMO depends on lock timing, so energies vary at rounding level between runs. `MOPAC_MOZYME_DIAGG2_COOPERATIVE=1` restores the deterministic cooperative sweep.
- PM6-DH/PM7 dispersion on the GPU: `MOPAC_DH_DISP_GPU=1` (also a default) evaluates the pairwise dispersion energy and its analytic gradient on the device (`src/gpu/dh_dispersion.cu`; the CPU code uses forward differences, six O(N) sweeps per atom). Non-periodic only. `MOPAC_GPU_DISP_CHECK=1` prints `[MOZYME GPU disp] check E_gpu= E_cpu= dE= max_abs_dgrad=` (differences of order 1e-5 kcal/mol/Å are the CPU finite-difference error).

Device Selection and Enablement
- Default behavior (CLI run):
  - GPU is enabled automatically when at least one suitable GPU is present and the system is not tiny (heuristic `natoms > 100`).
  - Up to 2 GPUs are used for select paths (cuBLASXt uses all selected devices for BLAS‑3; special 2‑GPU paths exist for MOZYME/outer‑products).
- Keywords and explicit control:
  - `NOGPU`: disable GPU for the job.
  - `SETGPU=n`: pick device `n` (1‑based) when multiple GPUs exist (skips unsuitable devices).
  - `MOZYME`: enables MOZYME algorithm (recommended for large biomolecules). GPU MOZYME is used automatically if GPU is on.
  - `MOZYME_GPUIGNORE=a,b,...`: ignore these 1‑based device indices during auto‑selection.
- Environment variables:
  - `MOPAC_FORCEGPU=1`: force‑enable GPU if any suitable device exists.
  - `MOPAC_NOGPU=1`: disable GPU (overrides auto‑enable).
  - `MOPAC_MIN_CC=7.0`: require minimum compute capability (e.g., 7.0 for Volta+).
  - `MOPAC_GPU_DEBUG=1`: print a summary of detected GPUs, chosen devices, and MOZYME GPU settings in the output.
  - `MOPAC_RESIDENT_SCF=1`: keep density/Fock/DIIS work buffers on the GPU across SCF iterations (default on; set to 0 to force host copies).

Eigensolver (cuSOLVER) and Density
- Threshold to use GPU eigensolver: `MOPAC_GPU_EIGEN_MIN` (default 400 AOs).
- Keep‑on‑device mode (faster SCF): `MOPAC_FASTGPU=1` keeps eigenvectors on device; density builds from device avoid host transfers.
- Fetch eigenvectors to host (when kept on device): `MOPAC_EIG2HOST=1`.
- Optional GPU orthogonalization (Cholesky + transforms): `MOPAC_ORTHO_GPU=1`.

Fock Build and Gradients
- Default: the two-center Fock build runs on GPU when `lgpu` is true.
- Opt-out: `MOPAC_NOFOCKGPU=1` forces CPU Fock build; legacy opt-in `MOPAC_FOCK_GPU=1` is still honored.
- Gradient reuse: device-resident Fock can be multiplied with C on device (`fmulC_from_dev`).
- Experimental gradients: set `MOPAC_GPU_GRAD=1` to request the CUDA gradient path. If no GPU implementation is available the code automatically falls back to the existing CPU routine and continues normally.

BLAS Acceleration and Multi‑GPU
- Single‑GPU: BLAS‑3 calls (GEMM, SYRK, TRSM) go through cuBLAS when `lgpu=true`.
- Multi‑GPU BLAS (cuBLASXt): used when multiple GPUs are selected.
  - `MOPAC_CUBLASXT_DEVICES="0,1"`: choose devices by index (0‑based).
  - `MOPAC_CUBLASXT_BLOCK=256`: set Xt block size.
  - CPU ratio (kept at 0 by default for portability).
- `MOZYME_GPU_FORCE=1`: keep MOZYME GPU enabled even on GPUs with compute capability < 6.0 (default auto policy disables it).
- `MOZYME_MINBLK=n`: lower the minimum MOZYME block size for GPU rank-1 GEMM/SYRK (default `n=16`).
- `MOPAC_DISP_GPU=1`: enable GPU evaluation of the DnX halogen dispersion term (energy and gradients); set to `0`/`off` to force the CPU path.
- `MOPAC_MOZYME_RESIDENT_FOCK_GPU=1`: enable the production MOZYME sparse Fock path. It uploads the sparse plan/integrals once per geometry and launches fused kernels for one-center, `4x1`/`1x4`, optimized `4x4`, generic exact two-center pairs for the fail-closed MOZYME basis-block sizes `{1,4,9}` including `9x4` and `9x9`, diagonal pair plans, and MOZYME point-charge/dipole Fock work. In direct-integral mode the resident W-only integral generators cover the same `{1,4,9}` basis range; d-shell pairs are routed through the s/p/d generator and retain fail-closed unsupported-basis counters for any future basis shape outside that range. Positive-orbital pairs outside device coverage are reported as CPU fallback; zero-orbital inactive pairs are reported separately as inactive/no-op coverage.
- `MOPAC_MOZYME_CNVGZ_GPU=0`: disable the exact GPU implementation of MOZYME `cnvgz`, the density convergence/history update inside the SCF loop. When MOZYME GPU is active, the default is enabled. The GPU path copies the updated packed density and diagonal history back to the canonical Fortran arrays before the next SCF step.
- `MOPAC_MOZYME_HELECZ_GPU=0`: disable the GPU reduction for MOZYME `helecz`, the electronic energy evaluation. When MOZYME GPU is active, the default is enabled. It requires the sparse `nijbo` map and falls back to CPU if that map is unavailable.
- `MOPAC_MOZYME_EIMP_GPU=0`: disable the exact GPU implementation of MOZYME `eimp`, the per-atom-pair `F**2` reduction used by the diagonalizer. When MOZYME GPU is active, the default is enabled. It requires the sparse `nijbo` map and copies the updated packed `p` entries back before `diagg`.
- `MOPAC_MOZYME_DENSITY_BATCH_GPU=0`: disable the sparse MOZYME density builder for small atom blocks. This path uses the `lijbo/nijbo` sparse map, computes per-block density products on the GPU, then scatters those products back into the packed density in the original CPU order for deterministic results. It falls back to the original CPU loops if the sparse map is unavailable or the CUDA helper fails.
- `MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU=1`: enable the opt-in full `diagg1` construction kernel. It builds `fmo/ifmo`, occupied and virtual pseudo-eigenvalues, `nfmo`, and the `sumt/tiny/ijc/ovmax` control values on the GPU, then copies those canonical outputs back before `diagg2`. It requires the sparse `nijbo` map and falls back to the CPU path if unavailable.
- `MOPAC_MOZYME_DIAGG1_AOCC_GPU=0`: disable the older exact GPU substage of `diagg1` that computes occupied-LMO atom contribution screening terms (`aocc`). When MOZYME GPU is active, the default is enabled. This helper is bypassed when `MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU=1` succeeds.
- `MOPAC_MOZYME_DIAGG1_AVIR_GPU=0`: disable the older exact GPU substage of `diagg1` that computes virtual-LMO atom contribution screening terms (`avir`). When MOZYME GPU is active, the default is enabled. This helper is bypassed when `MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU=1` succeeds.
- `MOPAC_MOZYME_DIAGG2_ROTATE_GPU=1`: enable the opt-in full `diagg2` sparse LMO rotation kernel. It performs the accept/retry rotation and sparse `cocc`/`cvir`/`ncf`/`nce` mutation on the GPU, then copies those canonical arrays back to Fortran. It is off unless explicitly set because this side-effect path needs CUDA benchmark validation across the publication set.
- `MOPAC_MOZYME_DIAGG2_ROTPREP_GPU=0`: disable the older GPU substage of `diagg2` that only prepares two-by-two LMO rotation coefficients. This helper is bypassed when `MOPAC_MOZYME_DIAGG2_ROTATE_GPU=1` succeeds.
- `MOPAC_MOZYME_ISITSC_GPU=1`: enable the opt-in GPU implementation of MOZYME `isitsc`, the scalar SCF convergence decision. It is off by default until the resident-SCF ABI owns convergence state for the full loop.
- `MOPAC_MOZYME_RELOCAL_GPU=1`: enable the standalone GPU-backed MOZYME `RE-LOCAL` helper outside strict proof mode. Strict resident proof mode enables this path automatically and aborts rather than falling back to CPU if the GPU helper cannot run.
- `MOPAC_MOZYME_REORTH_GPU=1`: enable the standalone GPU-backed MOZYME `REORTH` helper outside strict proof mode. Strict resident proof mode enables this path automatically and requires the following density/Fock/energy rebuilds to stay on GPU as well.
- `MOPAC_MOZYME_FOCK1_BATCH_GPU=1`: enable the batched one-center MOZYME Fock kernel for profiling. It is off by default because the work is too fine-grained to amortize GPU launch and transfer overhead in end-to-end molecule runs.
- `MOPAC_MOZYME_FOCK2_4X1_BATCH_GPU=1`: enable the batched `4x1`/`1x4` MOZYME two-center Fock kernel; set to `0`/`off` to force CPU.
- `MOPAC_MOZYME_FOCK2_4X1_BATCH_CHUNK=n`: set the 4x1 batch size. The default is 65536 tasks to reduce CUDA launch/synchronization overhead on large MOZYME systems.
- `MOPAC_MOZYME_SCF_EXPERIMENTAL=1 MOPAC_MOZYME_RESIDENT_SCF=1`: enable the experimental MOZYME resident-SCF boundary. This is an ABI/scaffold for moving the full MOZYME SCF loop onto the GPU. The v32 boundary carries active and total occupied/virtual counts separately, explicit host-buffer capacities and 2-D leading dimensions, DIAGG working state (`cvir`, `nce/nnce/ncvir/icvir`, `fmo/ifmo`, `eigs`, `nfmo`, and `nfirst/nlast`), DIAGG threshold/retry/output state (`nf`, `fref`, `oldlim`, `safety`, `sumt`, `sumb`, `ijc`, and `nrejct`), CNVGZ `pmax` and active/no-op counters, `addhb` loop-control state (`nhb` and `idiagg`), ISITSC convergence state (`iemin/iemax`, `scf1`, `escf0`, `emin`, previous heat, and heat conversion constants), SCF loop-control state (`use_three_point`, `lstart`, and `shift`), typed final host publication proof (`final_publication_done`, `final_publication_arrays`, `final_publication_bytes`, and `final_publication_cosmo`), resident final reorthogonalization status (`final_reorth_applied`, `final_reorth_ms`, and `final_reorth_sum`), PLS supervisor status (`pls_supervisor_calls`, `pls_restart_required`, `pls_history_count`, `pls_ovmax_delta`, `pls_energy_delta`, `pls_restart_reset_device_calls`, and `pls_restart_done`), resident COSMO runtime proof counters (`cosmo_fock_calls`, `cosmo_matvec_calls`, `cosmo_cg_iterations`, and final COSMO energies), and resident COSMO CG control/convergence status (`cosmo_cg_control_resident`, `cosmo_cg_converged`, `cosmo_cg_breakdown`, `cosmo_cg_host_syncs`, and `cosmo_cg_target_tol`) in addition to the density/Fock state. The current backend can enter a strict pre-`tidy` full-active-space initial setup path, use dual resident sparse Fock plan slots for partial initial setup, and complete resident iterations through upload, resident LMO `check` normalization, `eimp`, DIAGG rebuild/rotation, density rebuild, resident `addhb`/DIAGG2, `cnvgz`, full-coverage sparse Fock rebuild, `helecz`, integrated `isitsc`, direct EPS/COSMO Fock correction with resident CG state/control, GPU-side PLS supervision, and device-side PLS restart reset. If the resident PLS supervisor requests the historical restart correction, strict mode resets `pold`, `p1`, loop-control state, and PLS completion state on the device and continues resident SCF; `backend_pls_restart_required` is reserved for a pending or failed reset, not for a successful proof. When final reorthogonalization is due in strict mode, it must be handled by the resident backend and emit a `[MOZYME GPU reorth] status=success resident=1` final reorth marker. `MOPAC_MOZYME_SCF_FORCE_FINAL_REORTH=1` is a strict-proof diagnostic override that exercises that final resident bookend when the input also requests `REORTH`; leave it unset for normal MOPAC behavior. Unsupported CPU-owned canonicalization/setup/recovery paths such as LPKA and final bookends still gate blanket full-workflow GPU claims. Do not treat Fock, density, DIAGG, cuBLAS, cuSOLVER, or standalone host-copy REORTH offload as proof that the whole SCF workflow ran on GPU.
- `MOPAC_MOZYME_FOCK_GPU=1`: enable the legacy MOZYME Fock GPU wrapper for development only. It is off by default for production molecule benchmarks.
- `MOPAC_MOZYME_F2_GPU=1`: enable the legacy two-centre MOZYME Fock GPU wrapper only when `MOPAC_MOZYME_FOCK_GPU=1` is also set.
- `MOPAC_MOZYME_CHECK_GPU=1`: enable the experimental MOZYME LMO check/retry path. This is off by default and does not represent CUDA acceleration.

Streams, Pinning, and Determinism
- Streams: enable/disable CUDA streams for wrappers with `MOPAC_STREAMS=off` (defaults on).
- Pinning: `MOPAC_PIN_USER=1` attempts to pin user buffers for fewer copies (fallbacks safely).
- Deterministic cuBLAS settings: `MOPAC_DETERMINISTIC=1` (no atomics, host pointer mode, default math).

Verbose/Profiling
- `MOPAC_GPU_VERBOSE=1`: prints per‑call timings/GF/s for GEMM/SYRK and high‑level kernels; MG eigensolver logs under `[MGPU]`.
- `MOPAC_GPU_CSV=1`: prints a CSV‑style summary for gradient kernels at teardown.
- `MOPAC_GPU_PROFILE=1`: collect aggregated gradient kernel statistics (atom count, pair mix, cumulative ms) and emit a summary at teardown.
- `MOPAC_GPU_PROFILE=2`: extend profiling to cuBLAS wrappers (GEMM/SYRK, 2‑GPU splits) and cuSOLVERMg; timing/GFLOP summaries print at shutdown and NVTX ranges are emitted when `nvToolsExt` is available (for Nsight traces).
- `MOPAC_MOZYME_SECTION_PROFILE=1`: emit cumulative MOZYME section timers as `[PROFILE] MOZYME_SECTION name=<name> calls=<n> ms=<total>`. The molecule benchmark parses these into `mozyme_section_times.csv` and `mozyme_section_times.png` so remaining CPU bottlenecks can be ranked before adding more GPU kernels.
- Whole-run phases appear in the same report: `run_setup` (input, geometry and MOZYME set-up up to the first Hamiltonian), `run_writmo` (final output) and `run_total`; the hydrogen-bond correction is split into `hbonds_find_pairs`, `hbonds_setup_dh_plus` and `hbonds_energy_grad`, and the GPU hcore call into `hcore_gpu_pairlist`, `hcore_gpu_upload`, `hcore_gpu_kernels`, `hcore_gpu_download`.
- The PM6-DH+/PM7 hydrogen-bond correction (CPU) uses cell-grid neighbour lists for its covalent-bond searches and per-hydrogen chains for duplicate pairs (`src/corrections/H_bond_neighbours.F90`), so it scales linearly with the atom count instead of O(pairs x atoms); the lists are visited in ascending atom order and the results are bit-identical to the original scans. Periodic systems keep the original loops.
- With the same profile switches, the resident MOZYME SCF backend also prints non-blocking per-stage GPU timings (`[PROFILE] MOZYME_RESIDENT_STAGE name=<stage> calls=<n> ms=<total>`) on stdout once per `mopac_cuda_mozyme_scf_run` call. `scripts/mozyme_section_profile.py <mopac> <input.mop>` runs CPU / GPU / strict-resident GPU on the same deck and prints a section-by-mode table with both marker families.
- `MOPAC_EIG_MG_PROFILE=1`: gather cuSOLVERMg solve statistics (calls, failures, average runtime/devices) and print a single summary when GPU resources are released.

Example checks
- `scripts/run_gpu_suite.sh ./build-gpu/mopac` now includes `disp_halogen_gpu`, which exercises the GPU dispersion path using `examples/halogen_disp.mop` (`MOPAC_DISP_GPU=1`).
- `scripts/bench_mozyme_gpu_vs_cpu.py` benchmarks MOZYME GPU vs CPU on a chosen deck and emits a timing plot.

Benchmark Reporting Guardrails
- Colab proof zip:
  - Clean public proof: start from empty `git status --porcelain --untracked-files=all`, run `python3 scripts/create_colab_gpu_zip.py`, paste the printed `source_zip_sha256` into Colab as `trusted_expected_source_zip_sha256`, upload the zip plus `.zip.expected.json`, and leave `allow_development_dirty_zip = False`.
  - Clean proof snapshot from an active development tree: run `python3 scripts/create_colab_gpu_zip.py --snapshot-clean-proof --output mopac_colab_gpu_bench_proof.zip`. The packager copies only allowlisted source files into a temporary git worktree, commits that snapshot locally, and writes a proof-eligible zip plus `.zip.expected.json` without relaxing Colab proof checks.
  - Development dirty diagnostic: run `python3 scripts/create_colab_gpu_zip.py --allow-dirty`, set `allow_non_proof_run = True` and the dirty-zip override in Colab, and report `source_dirty_status_sha256`. Do not label this as proof.
  - If dirty status changes after packaging, regenerate the zip. The manifest proves packaged file bytes; `dirty_status_sha256` is a fingerprint of the `git status --porcelain --untracked-files=all` output.
- `scripts/gpu_benchmark_report.py` is a low-level wrapper benchmark only. Its `summary.json` sets `benchmark_scope=low_level_gpu_wrappers`, `full_scf_gpu_status=not_measured`, and `full_scf_gpu_ready=0`.
- `scripts/molecule_benchmark_report.py` reports production molecule speedups only when profiled MOZYME GPU work was emitted. Experimental resident-SCF fallback or resident-step execution is marked in parseable `full_scf_gpu_*` fields and is excluded from production speedup.
- Molecule benchmark reports emit machine-readable provenance: `feature_set=mozyme-full-scf-gpu-makvec-relocal-final-resident-reorth-tidy-selmos-pls-reset-cosmo-direct-point-kind-cnvgz-active-or-noop-cpu-compare-explicit-proof-v38-20260702`, `full_scf_contract_version=resident-scf-strict-final-resident-reorth-tidy-selmos-pls-reset-cosmo-direct-resident-cg-point-kind-cnvgz-active-or-noop-cpu-compare-explicit-proof-v58`, and `benchmark_scope=molecule_mozyme_full_scf_gpu`.
- Complete MOZYME full-SCF GPU compute is claimed only for rows that pass the strict resident-SCF readiness contract with no parsed CPU fallback and, in the Colab proof path, a same-input CPU companion energy comparison; final/status host commits remain allowed for MOPAC state publication.
- Use `scripts/molecule_benchmark_report.py --require-full-scf-gpu ...` only when a run is intended to claim complete SCF execution on GPU. The script runs a readiness probe first and aborts before the publication benchmark unless `full_scf_gpu_status=complete` and `full_scf_gpu_ready=1`.
- Current resident-SCF ABI v33 carries resident initial setup, `makvec`, `check`, DIAGG, ADDHB, CNVGZ/`pmax`, CNVGZ active/no-op counters, ISITSC convergence state, DIAGG `sumt`/`sumb`/`ijc`, SCF loop-control state and return decision, strict resident host synchronization/control-poll counters, typed final publication proof, resident final reorthogonalization status, GPU-side PLS supervisor status, direct-COSMO resident state/runtime counters and resident CG control/convergence status for the `addfckz` Fock correction, and explicit host buffer capacities for every raw pointer registered with CUDA.
  `complete` is reserved for a normal MOPAC exit with `status=success`, backend `ready=1`, `resident=1`, a nonnegative CUDA `device_id`, the full stage mask `1023`, `stage_missing=0`, `isitsc_okscf=1`, at least one resident iteration, positive device-side CNVGZ stage work (`cnvgz_active_calls + cnvgz_noop_calls > 0`), `[MOZYME GPU makvec] status=success`, `[MOZYME GPU setupk] success` with `initial_setup=1 all_initial_setup_paths=1`, a MOZYME plan that reports complete resident Fock coverage, resident sparse Fock work on GPU every resident iteration, `final_density=current_resident`, any PLS restart completed on GPU with final `pls_restart_required=0` and `pls_restart_done=1` when reset calls are nonzero, no CPU MOZYME state-mutating setup or bookend sections, no raw GPU error markers, and zero parsed fallback counters. Reporting, Colab checks, and the CTest smoke gate accept active-only, no-op-only, or mixed CNVGZ work only when both counters are present, nonnegative, and their sum is positive.
  The strict readiness contract fails if CPU `makvec`, `tidy`, `setupk`, `OLDEN` density rebuild, initial density/Fock, `check`, `eimp`, `diagg`, `cnvgz`, `isitsc`, PLS restart fallback, iterative Fock, final density, or reorthogonalization sections run outside the resident backend. `OLDEN`/`OLDENS` is accepted only as a marked setup-only host LMO restore before resident upload. `RE-LOCAL` must emit successful `[MOZYME GPU relocal]` markers when the corresponding `iter_reloc_*` timer is present, and `REORTH` must emit `[MOZYME GPU reorth] status=success resident=1` plus resident GPU density/Fock/energy rebuild markers when the final reorthogonalization bookend is requested; the `resident=1` final reorth marker is what distinguishes strict resident proof from the standalone host-copy helper.
  Outside strict proof mode, a `resident_step` row means one or more resident iterations completed on GPU but control returned to Fortran because ISITSC did not yet converge, the iteration budget was reached, or an immediate CPU checkpoint boundary is next.
  In `MOPAC_MOZYME_SCF_STRICT_RESIDENT=1`, `resident_step` is not accepted as success; the driver aborts before CPU SCF/body/bookend work instead of silently continuing.
  Controlled fallback reasons include `denout_checkpoint`, `solvent_fock`, `resident_fock_partial_coverage`, `strict_density_batch_fallback`, `early_probe`, `backend_missing_stages`, `backend_not_converged`, `backend_cnvgz_no_device_work`, `backend_cpu_boundary`, and `backend_pls_restart_required` for pending or failed device-side PLS restart reset.
  Partial initial setup now uses explicit full and partial resident Fock plan slots, `denout_checkpoint` is an immediate controlled fallback at the requested CPU `.den` checkpoint after any resident GPU iterations that could run before that boundary, `OLDEN`/`OLDENS` setup-only restore is marked with `[MOZYME GPU SCF] olden_setup=host_lmo_restore setup_only=1`, strict `RE-LOCAL` and `REORTH` use GPU helpers, strict proof rejects `OLD_SCF` host-existing LMOs and requires `[MOZYME GPU makvec] status=success`, any nonresident `density_for_MOZYME` call fails closed under strict proof if the density batch GPU path cannot complete, and `resident_fock_partial_coverage` is blocked when the resident Fock plan found any positive-orbital real or point-charge/dipole pair outside the device-covered basis range. Any positive `cpu_point_pairs` field on a `[MOZYME GPU resident_fock]` marker is fatal regardless of marker field order. In direct mode, base-9 d-shell pairs are covered by the resident s/p/d W-only generator; `unsupported_direct_basis` is reserved for future direct basis shapes outside `{1,4,9}`. `solvent_fock` is reserved for solvent modes that are still outside the resident contract, currently LPKA.

Multi‑GPU Eigensolver (cuSOLVERMg)
- Enable: `MOPAC_EIG_MG=1` with `ngpus>1` and set `MOPAC_EIG_MG_MIN` (e.g., `3000`).
- Tuning: `MOPAC_EIG_MG_GRID=PxQ` (e.g., `2x1`, `2x2`) and `MOPAC_EIG_MG_BLKSIZE=256`.
- Logging: with `MOPAC_GPU_VERBOSE=1`, prints `[MGPU] DSYEVD n=… grid=PxQ blksz=B: … ms`.
- Fallbacks: on any MG error or missing library, MOPAC safely falls back to the single‑GPU cuSOLVER path and notes it in logs.
- Profiling: enable `MOPAC_EIG_MG_PROFILE=1` to receive a one-line aggregate summary of MG solves (call/failure counts, average wall time, average active devices).
- Distribution: when cuSOLVERMg is active, the dense matrix is block-cyclic distributed across the selected GPUs, solved in place, and gathered back automatically—no user tiling required.

Cleanup and Safety
- GPU resources are released automatically at end of run. You can skip teardown via `MOPAC_SKIP_GPU_DESTROY=1` for debugging on fragile drivers.

Large Biomolecules (Proteins/DNA/RNA)
- Keywords: Prefer `MOZYME 1SCF EIGS VECTORS` for very large systems. Add `PULAY`, `DAMP`, and `SHIFT=-50` if SCF is tough.
- Multi‑GPU BLAS: If multiple GPUs are available, set `MOPAC_CUBLASXT_DEVICES="0,1"` (or leave unset to auto‑select). cuBLASXt accelerates density GEMM/SYRK across devices.
- Keep data on GPU: `MOPAC_FASTGPU=1` keeps eigenvectors/device data to reduce PCIe traffic. Use `MOPAC_EIG2HOST=1` only when you need vectors printed.
- Memory sizing: Peak VRAM (double) is roughly `40–56·n²` bytes during SCF for dense paths; with MOZYME, scaling is closer to linear in atoms. Use tiling or reduce `THREADS` for constrained VRAM.
- Determinism: `MOPAC_DETERMINISTIC=1` enforces cuBLAS settings to avoid atomics for reproducibility.
- Troubleshooting: Enable `MOPAC_GPU_VERBOSE=1` to see per‑call timings. If timing causes issues on old drivers, unset `MOPAC_STREAMS` (use default stream) and re‑run.

Best‑Practice Recipe
- Export: `MOPAC_FORCEGPU=1 MOPAC_FASTGPU=1 MOPAC_DETERMINISTIC=1`
- Optionally: `MOPAC_CUBLASXT_DEVICES="0,1"` on multi‑GPU nodes; set `CUDA_VISIBLE_DEVICES` accordingly.
- For big matrices: set `MOPAC_EIG_MG=1` and `MOPAC_EIG_MG_MIN=4000` once cuSOLVERMg is enabled in your build.
- Use example inputs in `examples/` (e.g., `peptide_gg.mop`, `mozyme_1gpu.mop`, `peptide_gg_2gpu.mop`).

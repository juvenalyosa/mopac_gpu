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

Eigensolver (cuSOLVER) and Density
- Threshold to use GPU eigensolver: `MOPAC_GPU_EIGEN_MIN` (default 400 AOs).
- Keep‑on‑device mode (faster SCF): `MOPAC_FASTGPU=1` keeps eigenvectors on device; density builds from device avoid host transfers.
- Fetch eigenvectors to host (when kept on device): `MOPAC_EIG2HOST=1`.
- Optional GPU orthogonalization (Cholesky + transforms): `MOPAC_ORTHO_GPU=1`.

Fock Build and Gradients
- Default: the two‑center Fock build runs on GPU when `lgpu` is true.
- Opt‑out: `MOPAC_NOFOCKGPU=1` forces CPU Fock build; legacy opt‑in `MOPAC_FOCK_GPU=1` is still honored.
- Gradient reuse: device‑resident Fock can be multiplied with C on device (`fmulC_from_dev`).

BLAS Acceleration and Multi‑GPU
- Single‑GPU: BLAS‑3 calls (GEMM, SYRK, TRSM) go through cuBLAS when `lgpu=true`.
- Multi‑GPU BLAS (cuBLASXt): used when multiple GPUs are selected.
  - `MOPAC_CUBLASXT_DEVICES="0,1"`: choose devices by index (0‑based).
  - `MOPAC_CUBLASXT_BLOCK=256`: set Xt block size.
  - CPU ratio (kept at 0 by default for portability).

Streams, Pinning, and Determinism
- Streams: enable/disable CUDA streams for wrappers with `MOPAC_STREAMS=off` (defaults on).
- Pinning: `MOPAC_PIN_USER=1` attempts to pin user buffers for fewer copies (fallbacks safely).
- Deterministic cuBLAS settings: `MOPAC_DETERMINISTIC=1` (no atomics, host pointer mode, default math).

Verbose/Profiling
- `MOPAC_GPU_VERBOSE=1`: prints per‑call timings/GF/s for GEMM/SYRK and high‑level kernels; MG eigensolver logs under `[MGPU]`.
- `MOPAC_GPU_CSV=1`: prints a CSV‑style summary for gradient kernels at teardown.

Multi‑GPU Eigensolver (cuSOLVERMg)
- Enable: `MOPAC_EIG_MG=1` with `ngpus>1` and set `MOPAC_EIG_MG_MIN` (e.g., `3000`).
- Tuning: `MOPAC_EIG_MG_GRID=PxQ` (e.g., `2x1`, `2x2`) and `MOPAC_EIG_MG_BLKSIZE=256`.
- Logging: with `MOPAC_GPU_VERBOSE=1`, prints `[MGPU] DSYEVD n=… grid=PxQ blksz=B: … ms`.
- Fallbacks: on any MG error or missing library, MOPAC safely falls back to the single‑GPU cuSOLVER path and notes it in logs.

Cleanup and Safety
- GPU resources are released automatically at end of run. You can skip teardown via `MOPAC_SKIP_GPU_DESTROY=1` for debugging on fragile drivers.

Large Biomolecules (Proteins/DNA/RNA)
- Keywords: Prefer `MOZYME 1SCF EIGS VECTORS` for very large systems. Add `PULAY`, `DAMP`, and `SHIFT=-50` if SCF is tough.
- Multi‑GPU BLAS: If multiple GPUs are available, set `MOPAC_CUBLASXT_DEVICES="0,1"` (or leave unset to auto‑select). cuBLASXt accelerates density GEMM/SYRK across devices.
- Keep data on GPU: `MOPAC_FASTGPU=1` keeps eigenvectors/device data to reduce PCIe traffic. Use `MOPAC_EIG2HOST=1` only when you need vectors printed.
- Memory sizing: Peak VRAM (double) is roughly `40–56·n²` bytes during SCF for dense paths; with MOZYME, scaling is closer to linear in atoms. Use tiling or reduce `THREADS` for constrained VRAM.
- Determinism: `MOPAC_DETERMINISTIC=1` enforces cuBLAS settings to avoid atomics for reproducibility.
- Troubleshooting: Enable `MOPAC_GPU_VERBOSE=1` to see per‑call timings. If timing causes issues on old drivers, unset `MOPAC_STREAMS` (use default stream) and re‑run.

Multi‑GPU Eigensolver (cuSOLVERMg)
- Implemented using cuSOLVERMg with 2D block‑cyclic distribution and robust fallbacks.
- Enable with `MOPAC_EIG_MG=1` (and `ngpus>1`), set threshold via `MOPAC_EIG_MG_MIN`. Use `MOPAC_EIG_MG_GRID`/`MOPAC_EIG_MG_BLKSIZE` to tune.
- If cuSOLVERMg is not available at build or runtime errors occur, a single‑GPU cuSOLVER solve is used automatically.

Best‑Practice Recipe
- Export: `MOPAC_FORCEGPU=1 MOPAC_FASTGPU=1 MOPAC_DETERMINISTIC=1`
- Optionally: `MOPAC_CUBLASXT_DEVICES="0,1"` on multi‑GPU nodes; set `CUDA_VISIBLE_DEVICES` accordingly.
- For big matrices: set `MOPAC_EIG_MG=1` and `MOPAC_EIG_MG_MIN=4000` once cuSOLVERMg is enabled in your build.
- Use example inputs in `examples/` (e.g., `peptide_gg.mop`, `mozyme_1gpu.mop`, `peptide_gg_2gpu.mop`).

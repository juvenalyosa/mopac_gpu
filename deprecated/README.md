# Deprecated sources

Files moved out of the build because nothing in the program uses them any more. They keep their
original path below this directory (`deprecated/src/...` was `src/...`) and their git history, so
any of them can be brought back with `git mv`.

This directory is not part of the build: CMake does not look here.

Not to be confused with `src/deprecated/`, which is part of the build: `mod_vars_cuda.F90`,
`mod_gpu_info.F90` and `mod_calls_cublas.F90` there are still used by the GPU code (the module
`mod_vars_cuda` alone by 44 source files).

| File | Was | Why it left | Replaced by |
|---|---|---|---|
| `src/matrix/diag_for_GPU.F90` | GPU pseudo-diagonalization for the conventional SCF (fast Jacobi rotations of the occupied-virtual block) | No caller: the only references were commented-out calls in `src/SCF/iter.F90` | The conventional SCF diagonalizes through `eigenvectors_LAPACK` (cuSOLVER on the GPU) |
| `src/matrix/call_rot_cuda.F90` | Fortran interfaces to the CUDA rotation kernels `call_rot_cuda_gpu` / `call_rot_cuda_2gpu_gpu` | Used only by `diag_for_GPU.F90` | — (the C++ functions are still in `src/gpu/cuda_wrappers.cu`, unused) |
| `tests/gpu_rot_verify.F90`, `tests/gpu_rot_2gpu_verify.F90` | Stand-alone checks of the CUDA rotation kernels (not in `tests/CMakeLists.txt`) | They `use call_rot_cuda` | — |
| `src/deprecated/pulay_for_gpu.F90` | GPU variant of the Pulay (DIIS) converger | No caller | `src/SCF/pulay.F90`, which calls the GPU DIIS helpers itself (`gpu_diis_interfaces`, `gpu_bmat_interfaces`, `gpu_small_solve_interfaces`) |

Note: `tests/check_gpu_source_build_contract.py`
already failed before this move (three unrelated contract assertions), and still fails the same way.

Moved on 2026-09-25 after a usage scan of every GPU source file (module `use` statements, calls of
every subroutine and of every `bind(C)` interface).

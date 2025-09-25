cuSOLVERMg Integration Notes (Draft)

Goals
- Add multi-GPU symmetric eigensolver (DSYEVD) to MOPAC when multiple GPUs are present.
- Maintain compatibility with existing single-GPU cuSOLVER (keep-on-device option).

Data layout considerations
- cuSOLVERMg expects a distributed 2D block-cyclic layout across devices.
- MOPAC currently forms dense full matrices (host and device) with column-major storage.
- A staging step is needed to redistribute the full dense matrix into Mg descriptors (grid, block sizes, device contexts).

Plan (status)
1) Detection: ENABLE_CUSOLVER_MG defines HAVE_CUSOLVER_MG when library is found. (done)
2) Plumbing: Fortran interface + C wrapper in `src/gpu/cuda_wrappers.cu` (`mopac_cusolvermg_dsyevd`). (done)
3) Distribution: 2D block-cyclic with `MOPAC_EIG_MG_GRID` and `MOPAC_EIG_MG_BLKSIZE`. (done)
4) Host↔Mg copy: use cuSOLVERMg descriptors and memcpy helpers. (done)
5) DSYEVD call: `cusolverMgSyevd[_bufferSize]`. (done)
6) Gather: copy eigenvectors back to host full column-major and return eigenvalues. (done)
7) Fallback: on any error or insufficient devices, revert to single-GPU DSYEVD with a clear log note. (done)

Initial scope
- RHF/UHF core eigensolve (square, symmetric, upper triangle input), double precision.
- Use the same GPU streams policy as existing wrappers; simple single stream per device at first.

Open questions
- Memory pressure vs. replication: whether to duplicate the full matrix on host and distribute, or construct directly in distributed memory for large n.
- Cross-device communication: NCCL vs. peer-to-peer; cuSOLVERMg manages its internal transfers but upstream paths may benefit from pinned buffers.

Testing
- Use `scripts/run_gpu_suite.sh` with `MOPAC_EIG_MG=1` on 2+ GPUs; expect `[MGPU] DSYEVD …` logs and parity with single-GPU.
- Compare eigenvalue spectrum and orthogonality checks vs. single-GPU reference.

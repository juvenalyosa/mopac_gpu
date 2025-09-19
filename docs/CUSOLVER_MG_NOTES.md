cuSOLVERMg Integration Notes (Draft)

Goals
- Add multi-GPU symmetric eigensolver (DSYEVD) to MOPAC when multiple GPUs are present.
- Maintain compatibility with existing single-GPU cuSOLVER (keep-on-device option).

Data layout considerations
- cuSOLVERMg expects a distributed 2D block-cyclic layout across devices.
- MOPAC currently forms dense full matrices (host and device) with column-major storage.
- A staging step is needed to redistribute the full dense matrix into Mg descriptors (grid, block sizes, device contexts).

Plan (staged)
1) Detection (done): add CMake option ENABLE_CUSOLVER_MG and define HAVE_CUSOLVER_MG if library is found.
2) Plumbing (placeholder present): Fortran interface + C wrapper function that will own MG setup.
3) Distribution design: choose block size (e.g., 128 or 256) and form device grid (e.g., PxQ devices).
4) Implement host-to-Mg copy: allocate distributed buffers, scatter columns/rows according to block-cyclic mapping.
5) Call cusolverMgSyevd (or equivalent) to compute eigenpairs on-device.
6) Gather back the full eigenvector matrix into host column-major layout (and optionally leave on one device).
7) Fall back: if Mg reports an error (or resources insufficient), revert to the single-GPU path.

Initial scope
- RHF/UHF core eigensolve (square, symmetric, upper triangle input), double precision.
- Use the same GPU streams policy as existing wrappers; simple single stream per device at first.

Open questions
- Memory pressure vs. replication: whether to duplicate the full matrix on host and distribute, or construct directly in distributed memory for large n.
- Cross-device communication: NCCL vs. peer-to-peer; cuSOLVERMg manages its internal transfers but upstream paths may benefit from pinned buffers.

Testing
- Add a moderate-size test (n≈4k) on 2x GPUs gated by ENABLE_GPU_TESTS and ENABLE_CUSOLVER_MG.
- Compare eigenvalue spectrum and orthogonality checks vs. single-GPU reference.


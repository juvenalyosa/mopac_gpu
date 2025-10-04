# CPU Reference for SCF/Density/Fock Builds

This document captures the canonical behaviour of the trusted CPU code paths
that the new GPU implementation must reproduce exactly. Each section lists the
entry points, the key data structures they consume/produce, and the ordering
constraints that downstream routines rely on.

## Fock Build (`src/SCF/fock2.F90`)

- **Inputs**
  - `ptot`, `p`: packed lower-triangle (length `mpack`) using 1-based Fortran
    ordering where element `(i,j)` with `i>=j` lives at index `ifact(i) + j`.
  - Two-electron integral tiles `w`, `wj`, `wk` with the same layout expected by
    the branch that calls them (general, periodic, heavy/light, etc.).
  - Atom block ranges `nfirst`, `nlast` (1-based, inclusive) for each atom.
- **Outputs**
  - `f`: packed lower-triangle Fock matrix. The routine overwrites the full
    buffer and zero-initialises it before accumulation.
- **Branch structure**
  - General branch (`call fockdorbs`) when either atom spans >= 6 orbitals.
  - Heavy–heavy, heavy–light, light–heavy (each with bespoke Coulomb + exchange
    loops) when spans fall in MNDO/AM1 shortcuts.
  - Light–light trivial case writes the single Coulomb contribution and the
    exchange subtraction.
  - Periodic branch uses `wj`/`wk` matrices; otherwise `wk` is unused.
- **Exchange updates**
  - For every contributing quartet `(i,j,k,l)` the code updates the symmetric
    set `(ij, kl, ik, il, jk, jl)` guarded by the packed indexing conditions so
    each element is touched exactly once.
  - The exchange factor is `a * aa * bb * 0.25` where `aa`, `bb` account for
    equality of orbital indices.

## Density Formation (`src/matrix/densit.F90` and `src/matrix/density_for_GPU.F90`)

- CPU reference (`densit`) performs
  - Full occupation block: `2 * C(:, nl2:nu2) * C(:, nl2:nu2)^T`
  - Fractional block: `frac * C(:, nl1:nu1) * C(:, nl1:nu1)^T`
  - Adds `cst` to the diagonal, then packs via `dtrttp('u', ...)` into `pp`.
- GPU helper currently mirrors this flow using cuBLAS DGEMM/DSYRK when allowed.
- **Critical details to preserve**
  - Packed storage is generated after the full symmetric matrix is built in
    column-major order.
  - `mopac_cuda_density_add_diag` must be semantically equivalent to adding
    `cst` before packing; the resident cache must reflect the updated values.
  - Pulay arrays (`pold`, etc.) assume the packed data is ready immediately
    after the density call returns.

## SCF Loop (`src/SCF/iter.F90`)

- **Sequence per iteration (closed-shell)**
  1. Build alpha Fock matrix (`fock2`) using the current density `p`.
  2. Diagonalise Fock → eigenvectors `c`, eigenvalues `eigs` (LAPACK by default).
  3. Form density (`density_for_GPU`/`densit`) and update Pulay histories.
  4. Optionally apply DIIS / mixing before next iteration.
- **Open-shell** adds beta-channel repetitions with shared Pulay machinery.
- Any GPU path must honour the same ordering, especially when resident caches
  are introduced. When Pulay modifies `p`, GPU copies must be invalidated before
  the next build.

## Gradient Builders (`src/forces/dfock2.F90`, etc.)

- Use the same packed indexing helper arrays (`ifact`, `i1fact`) as `fock2`.
- All derivatives assume the Fock matrix in packed form matches the CPU layout.

## Shared Helper Requirements

- We need a single set of index utilities (`packed_index(i,j)`, `pair_count`) in
  a header/module that both Fortran (via ISO_C_BINDING) and CUDA include to
  avoid divergent definitions.
- Device kernels must operate on 1-based inputs but convert to 0-based indices
  internally in a single canonical way.

The new GPU implementation sits behind the opt-in environment flag
`MOPAC_GPU_EXACT_SC`. With the flag unset the CPU reference remains the only
execution path, ensuring a clean baseline while the CUDA port is brought to
parity. Once the GPU path reproduces these reference behaviours, the flag can be
used to validate energy/gradient matches before any broader enablement.

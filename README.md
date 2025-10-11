# Barranquilla MOPAC — A GPU‑Accelerated Semiempirical Flavor of MOPAC
<img src="logo/barranquilla_logo.png" alt="Barranquilla MOPAC Logo" style="max-width: 100%; width: 720px; height: auto;" />
<!-- NOTE (modification): 2025-10-11 — Added GPU usage cheat sheet and
     configuration matrix for keywords/envs and per‑GPU guidance. -->

Barranquilla MOPAC is an implementation of MOPAC specialized for large systems (proteins, DNA/RNA, polymers, and extended materials) with deterministic GPU acceleration and smart SCF. It preserves the physics and accuracy of the semiempirical framework while exploiting locality and tensor contraction structure on modern GPUs.

Contents
- Why Barranquilla MOPAC? (Large‑system specialization)
- Physics background (semiempirical PMx, Fock build, SCF equations)
- HMTR geometry optimization (trust‑region, models, update rules)
- GPU implementation (conventional SCF and MOZYME)
- Automatic policies (no‑tuning defaults)
- Build and run (with examples)
- Advanced controls
- Performance expectations and best practices

---

## Why Barranquilla MOPAC?

Barranquilla MOPAC is a semiempirical, GPU‑accelerated MOPAC flavor designed for very large systems — proteins, DNA/RNA, polymers, and extended materials — where conventional ab‑initio scaling is prohibitive. It couples the physics of PMx/NDDO Hamiltonians with architectures that favor massive, batched linear algebra, and adds SCF/optimizer logic that reduces iteration counts reliably.

1) Semiempirical physics that fits large systems
- PMx (e.g., PM7) under the NDDO approximation replaces explicit four‑center ERIs with closed‑form, parameterized two‑center expressions. The Fock build becomes a sum of two‑center “atom‑pair” contractions rather than dense rank‑4 tensor algebra.
- The electronic nearsightedness principle in condensed/biological matter implies locality: far‑separated fragments weakly interact. With localized orbitals (MOZYME), the Fock/density operators become block‑sparse with compact support on atom blocks/torsional neighborhoods.
- For biomolecules (proteins, nucleic acids) and polymers, repeating motifs and short‑range couplings yield a natural hierarchy of blocks — exactly what modern GPUs process well when streamed as independent batches.

2) Algorithmic structure aligned to GPUs
- Two‑center J/K build dominates the SCF wall‑clock on large systems. In Barranquilla, the “general” two‑center blocks are fed to the GPU as batched pair kernels, while compact corner cases (LL/HL/HH) remain on the CPU by default. This split maximizes GPU throughput where it matters and preserves numerical exactness for the fragile corner cases.
- Streaming/resident modes: very large cases can stream pair slices (J/K blocks) from host/disk, staging them into device memory with predictable memory footprints; medium cases keep SCF data resident to amortize H2D/D2H traffic.
- Determinism and verification: when enabled (developer mode), per‑pair verifiers compare device and host contributions without impacting production runs.

3) Robust self‑consistency (fewer iterations, same physics)
- CDIIS (Pulay) converges very fast near the fixed point but can overshoot early; EDIIS (energy DIIS) is safer far from convergence. Barranquilla uses a hybrid EDIIS→CDIIS strategy with residual‑based switching, adaptive negative level shifts for virtuals, and residual‑proportional damping. You get faster, more predictable SCF with no change to the underlying Hamiltonian.

4) Geometry optimization that scales
- The HMTR optimizer (Hierarchical Memetic Trust‑Region) marries population‑based, torsion‑aware exploration with a rigorous trust‑region micro‑solver. It batches energy/gradient calls — which are the dominant cost at each geometry step — across GPU(s), and it adapts the trust radius using the standard acceptance ratio ρ. This approach is particularly effective for proteins (many low‑curvature torsions) and soft materials.

5) Automatic policies (no tuning required)
- Barranquilla selects CPU/GPU policies from problem size and device CC: small SCFs stay on CPU; medium cases keep data resident; large cases stream J/K; MOZYME pair work is enabled on newer GPUs and can be forced when needed. You typically only choose the number of GPUs; the code chooses the rest.

In short: Barranquilla MOPAC brings semiempirical physics to large systems with a GPU‑first execution model where it counts (general two‑center J/K), combines it with smarter SCF/optimizer logic to cut iterations, and keeps correctness safeguards for the fragile corners.

---

## Background — SCF, Fock Build, and Energy

Let `S` be the AO overlap matrix, `H` the one‑electron (core) Hamiltonian, `F` the Fock matrix, and `P` the (spin‑summed) density. Semiempirical PMx/NDDO Hamiltonians replace expensive four‑center electron‑repulsion integrals with analytic, parameterized two‑center forms, so the Fock assembly becomes a sum of two‑center “atom‑pair” contractions.

For a closed‑shell (RHF) problem, the Fock operator has the form
```
F[P] = H + G[P] = H + J[P] − K[P]
```
where `J` and `K` are the Coulomb and exchange contributions constructed by contracting the AO density over two‑center kernels derived from the NDDO/PMx parameter set. In practice, these contractions are evaluated as independent batches over atom pairs `A,B`, which is the key to GPU offload.

Given `F`, the SCF solves the generalized eigenvalue problem
```
F C = S C ε
```
which yields molecular orbitals `C` and eigenvalues `ε`. Occupied subspaces define the density. For RHF the density is
```
P = 2 · C_occ · C_occ^T
```
(with finite‑temperature or fractional occupations when required). For UHF the spin channels are separated, `P = P^α + P^β` with corresponding `F^α, F^β` and occupations per spin.

The total electronic energy in AO (packed‑lower) storage is
```
E_elec = 1/2 · Tr[ P · (H + F) ]
E_total = E_elec + E_nuclear
```
and in UHF the natural variant is `E_elec = 1/2 · Tr[ P^α · (H + F^α) + P^β · (H + F^β) ]`. Self‑consistency is monitored via the generalized commutator residual
```
R = F P S − S P F
```
with convergence declared when ‖R‖ and the changes in `E` and `P` fall below user‑defined tolerances.

The SCF loop is: start from a guess `P`, assemble `F[P]`, solve `F C = S C ε`, update occupations and form a new `P`, mix the density/Fock information using a convergence accelerator, measure `R`, and iterate. Near the fixed point, Pulay’s CDIIS (commutator DIIS) constructs an optimal linear combination of recent Fock/density pairs that minimizes the residual norm. Far from self‑consistency, CDIIS can be over‑aggressive; energy DIIS (EDIIS) forms a convex combination that lowers the total energy and is more stable.

Barranquilla MOPAC adopts a hybrid strategy: begin with EDIIS‑like damped mixing and an adaptive negative level shift on the virtual space to stabilize the spectral gap, then switch on CDIIS automatically once the residual is small (or after a few iterations). This strategy reduces iterations while preserving the underlying semiempirical physics and the final fixed point.

---

## HMTR Geometry Optimization — Trust‑Region with GPU Support

HMTR (Hierarchical Memetic Trust‑Region) is a geometry optimizer that minimizes the potential energy surface \(E(R)\) with respect to the nuclear coordinates \(R\) by coupling two complementary mechanisms. A memetic, population‑based stage proposes diverse candidate geometries (with an emphasis on torsional subspaces for large biomolecules), and a rigorous trust‑region micro‑solver refines each candidate using a quadratic model. The design is well‑suited to GPUs because many energy/gradient evaluations for different candidates can be batched.

At a current geometry \(R_k\), the optimizer builds a local quadratic model of the energy around \(R_k\)
```
m_k(p) = E(R_k) + g_k^T p + (1/2) p^T B_k p
```
where \(g_k = \partial E/\partial R |_{R_k}\) is the gradient and \(B_k\) is a Hessian or quasi‑Newton approximation. The trial step \(p\) is obtained by solving the trust‑region subproblem
```
minimize_p  m_k(p)  subject to  ‖p‖ ≤ Δ_k
```
with a norm chosen for the coordinate representation (Cartesian, internal, or torsional; a diagonal scaling can be used to account for units). If \(B_k\) is positive definite and the unconstrained Newton step \(p_N = -B_k^{-1} g_k\) satisfies \(‖p_N‖ ≤ Δ_k\), then \(p = p_N\) is used. Otherwise, a truncated step on the trust‑region boundary is computed, for example via the dogleg path between the Cauchy (steepest‑descent) point and the Newton point, or with truncated conjugate gradients.

The model quality is assessed by the ratio of actual to predicted decrease
```
ρ_k = [ E(R_k) − E(R_k + p_k) ] / [ m_k(0) − m_k(p_k) ]
```
and both the step and the trust‑region radius are updated accordingly. A step is accepted when \(ρ_k\) exceeds a small threshold \(η_0\) (e.g., 0.1), in which case \(R_{k+1} = R_k + p_k\); otherwise the step is rejected and \(R_{k+1} = R_k\). The trust radius is adapted: if \(ρ_k\) is large (e.g., \(ρ_k ≥ η_2\)), then the radius is expanded \(Δ_{k+1} = τ_{\text{expand}} Δ_k\); if \(ρ_k\) is poor (e.g., \(ρ_k ≤ η_1\)), then the radius is shrunk \(Δ_{k+1} = τ_{\text{shrink}} Δ_k\); otherwise the radius is left unchanged. Typical values are \(η_1 \approx 0.25\), \(η_2 \approx 0.75\), \(τ_{\text{expand}} \approx 2\), \(τ_{\text{shrink}} \approx 1/2\).

To update the curvature model, HMTR employs a quasi‑Newton formula such as BFGS. Let \(s_k = p_k\) and \(y_k = g(R_k + p_k) − g(R_k)\). The BFGS update reads
```
B_{k+1} = B_k − (B_k s_k s_k^T B_k) / (s_k^T B_k s_k) + (y_k y_k^T) / (y_k^T s_k)
```
with a standard damping if \(y_k^T s_k\) is too small, to maintain positive definiteness. The gradients \(g\) are obtained from the semiempirical electronic structure (Hellmann–Feynman plus Pulay terms), and the trust‑region solve treats the nuclear surface only; the SCF is converged at each geometry evaluation.

The “memetic” component maintains a population of candidate geometries and uses simple evolutionary moves in a low‑curvature subspace (e.g., torsions). A particle‑swarm‑like update for torsions is convenient:
```
v_{i}^{t+1} = ω v_{i}^{t} + c1·r1 ⊙ (pbest_i − θ_i^t) + c2·r2 ⊙ (gbest − θ_i^t)
θ_{i}^{t+1} = wrap( θ_i^t + v_{i}^{t+1} )
```
where \(θ_i\) are the torsions of particle \(i\), \(v_i\) their velocities, \(pbest_i\) the best torsions previously found by particle \(i\), and \(gbest\) the global best in the population. The operator `wrap` enforces periodicity and any box constraints; \(ω, c1, c2\) are inertial and cognitive/social weights and \(r1, r2\) are uniform random vectors. Each particle is then refined by the trust‑region micro‑optimizer described above. The population proposal stage is trivially parallel and is evaluated in batches on GPU(s); the trust‑region steps are cheap relative to the energy/gradient calls and run on the host.

In practice, HMTR’s effectiveness on proteins and soft materials comes from combining torsional diversity (to escape narrow basins separated by small barriers) with a principled local model to accept or reject moves. The acceptance ratio \(ρ\) acts as a universal quality indicator independent of the particular coordinate choice or step generator, and the trust‑region radius adapts the aggressiveness of local refinement automatically. All energy and gradient evaluations use the same semiempirical Hamiltonian and SCF stack described earlier; the GPU acceleration enters only through batched evaluations across the population or internal micro‑batches.

### Math Appendix: Semiempirical (NDDO/PMx) Essentials

Under the Neglect of Diatomic Differential Overlap (NDDO), the AO basis is orthogonalized within atoms and overlap is neglected between different atoms beyond specific terms. Key parameterizations appear as:

- On‑site Coulomb integrals α_A and resonance integrals β_A (orbital dependent in extended models). For a diatomic pair (A,B), two‑center quantities depend on the interatomic distance R_AB.
- Two‑center Coulomb γ_AB (electron–electron repulsion between net charges localized on A and B):
```
γ_AB(R) ≈ 1 / sqrt( R^2 + δ_A^2 + δ_B^2 )
```
with screening radii δ_A, δ_B from parameters. In PMx, distance‑dependent forms are used to fit heats of formation and other observables.

In matrix language (suppressing AO indices), the semiempirical Fock for a closed shell is:
```
F = H_core + J(P) − K(P)
H_core(AO) ≈ α + β terms (on‑site and nearest‑neighbor resonance)
J(P)  ≈ ∑_{A≤B}  Γ_AB : P_AB    (pairwise Coulomb contraction)
K(P)  ≈ ∑_{A≤B}  Λ_AB : P_AB    (pairwise exchange contraction)
```
where `Γ_AB, Λ_AB` are small pairwise tensors generated by NDDO/PMx formulas, and `P_AB` is the block of the density over orbitals centered on atoms A and B. This “pairwise” structure is what enables efficient two‑center batching on GPUs.

---

## Build Prerequisites (CPU/GPU)

Toolchain
- CMake ≥ 3.14 and a Fortran compiler (e.g., `gfortran` ≥ 9).
- C/C++ compilers for wrappers (e.g., `gcc`/`g++`).
- Optional Ninja generator (`ninja-build`).

Math libraries
- BLAS and LAPACK (auto-detected): Intel MKL, OpenBLAS, or system BLAS/LAPACK.
- OpenMP (optional) to enable the `THREADS` keyword.

GPU (when `-DGPU=ON`)
- NVIDIA CUDA Toolkit (matching your driver), including cuBLAS and cuSOLVER.
- Set CUDA architectures via `-DCUDA_ARCHS=` (e.g., 52, 70, 80, 90) or `native`.
- Optional: cuSOLVERMg for multi‑GPU eigensolver (`-DUSE_CUSOLVER_MG=ON`).

Linux example: GPU build with GCC 12 and Ninja
```
/usr/bin/cmake -S . -B build-gpu -G Ninja \
  -DGPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build-gpu --target all --parallel
```

Tip: pass `-DCUDA_ARCHS=<sm>` for your GPU (e.g., `52` for Maxwell/TITAN X, `80` for A100).

Ubuntu quickstart
```
sudo apt-get update
sudo apt-get install -y build-essential gfortran cmake ninja-build \
    gcc-12 g++-12 libopenblas-dev liblapack-dev
# Install the NVIDIA driver and CUDA Toolkit matching your GPU (see NVIDIA docs),
# or use the distribution's CUDA packages if suitable for your environment.

/usr/bin/cmake -S . -B build-gpu -G Ninja \
  -DGPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build-gpu --target all --parallel
```

CPU‑only quickstart (Ubuntu)
```
sudo apt-get update
sudo apt-get install -y build-essential gfortran cmake ninja-build \
    libopenblas-dev liblapack-dev

/usr/bin/cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all --parallel
```

CUDA architectures (examples for `-DCUDA_ARCHS`)
- Maxwell TITAN X: `52`
- Pascal P100: `60`
- Turing (e.g., RTX 2080 Ti): `75`
- Volta V100: `70`
- Ampere A100: `80`
- Ampere RTX 30 (e.g., 3080/3090): `86`
- Ada RTX 40 (e.g., 4080/4090): `89`
- Hopper H100: `90`

Examples
```
# Single architecture
/usr/bin/cmake -S . -B build-gpu -G Ninja -DGPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCUDA_ARCHS=80

# Fat binary for two SMs (e.g., V100 + A100 farm)
/usr/bin/cmake -S . -B build-gpu -G Ninja -DGPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCHS="70;80"
```

CUDA install (Ubuntu)

Official NVIDIA repository (recommended)
- Ensure kernel headers and common tools are present:
  - `sudo apt-get update && sudo apt-get install -y build-essential dkms gnupg software-properties-common`
- Pick the repo that matches your Ubuntu release:
  - Ubuntu 22.04 (Jammy):
    ```
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install nvidia-driver-535
    sudo apt-get -y install cuda-toolkit-12-2
    ```
  - Ubuntu 20.04 (Focal): replace `ubuntu2204` with `ubuntu2004` in the keyring URL, then run the same commands.
- After install, set your PATH/LD_LIBRARY_PATH (or add to your shell profile):
  ```
  export PATH=/usr/local/cuda-12.2/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
  ```
- Reboot if you installed/changed the driver, then verify:
  - `nvidia-smi` (driver and GPUs present)
  - `nvcc --version` (CUDA host tools available)

Distribution packages (quick test; often older CUDA)
- `sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit`
- This may install an older toolkit; prefer NVIDIA’s repo for matching versions and features.

## GPU Implementation

The GPU acceleration in Barranquilla MOPAC is designed around the two‑center nature of the semiempirical J/K build and the fact that the vast majority of work on large systems is “general” atom‑pair contractions. Corner cases with compact tables (LL/HL/HH) are intentionally left on the CPU by default; this preserves numerical exactness and has negligible performance cost because their work share is tiny.

Conventional SCF (non‑MOZYME)
The Fock assembly is written as a stream of atom‑pair blocks. For each pair of atoms \((A,B)\), a block of Coulomb and exchange weights (denoted \(W\), or split as \(W_J, W_K\) for periodic cases) contracts against the corresponding sub‑blocks of the density. Barranquilla enumerates these pairs in a deterministic order, classifies them by size (compact LL/HL/HH versus general), and transfers the “general” blocks to the GPU in batches. The compact LL/HL/HH cases — which use small fixed tables of size 1, 10, or 100 — are accumulated on the CPU for correctness. On large systems, the general blocks dominate the wall‑clock, so moving them to the GPU captures most of the speedup without risking accuracy.

Two data‑movement strategies are used. For medium problems, resident mode is preferred: the packed‑lower Fock and density arrays are uploaded once and updated in place on the GPU; cuSOLVER is used to diagonalize \(F\) (Dsyevd), and a SYRK/GEMM densitizer forms \(P\) directly on device. Keeping the eigenvectors resident avoids repeated large host/device transfers. For very large problems, a streaming interface enumerates pair blocks and publishes just the \(W\) or \(W_J/W_K\) slices needed for the next kernel launch; the device holds a single Fock accumulation buffer and pulls density/metadata once per SCF iteration. In both cases, the resident/stream decision is automatic and can be overridden.

For diagonalization and density formation, the device path mirrors the CPU algebra: solve \(F C = S C \varepsilon\) in the transformed orthonormal basis (Cholesky or canonical orthogonalization is unchanged), then form \(P\) using an upper‑triangular SYRK of the occupied block or a linear combination of occupied and fractionally occupied blocks if smearing is used. In RHF, \(P = 2 C_{\text{occ}} C_{\text{occ}}^T\); in UHF, the two spin channels are treated separately and combined after the SCF sweep. The device holds the packed‑lower form of \(F\) and uses a packed‑to‑full unpacker when forming \(F C\) products on device.

To guarantee numerical parity, Barranquilla separates compact cases (LL/HL/HH) from general cases. LL/HL/HH use small fixed tables and delicate index maps historically; they are accumulated on the CPU where the legacy mappings are exact. “General” two‑center blocks — which make up the dominant work share — are offloaded to the GPU and accumulate into the same packed‑lower Fock array. Optional verification hooks (disabled by default) can compare GPU and CPU contributions pair‑by‑pair without impacting production runs.

MOZYME (proteins)
In MOZYME, localized orbitals yield block‑sparse density/Fock operators with strictly local interactions. Barranquilla offloads the localized pair kernels (F2/DF2) to the GPU whenever the localized block exceeds a minimum threshold (set by `MOZYME_MINBLK`), leaving very small blocks on the CPU to avoid launch overhead. The pair kernels operate on reduced‑dimension blocks (torsional neighborhoods) and are batched across atom groups; this pattern maps well to single‑GPU and can be extended to multi‑GPU when beneficial. On older GPUs with limited double‑precision throughput, the default policy is conservative and can be overridden via environment.

HMTR (geometry optimization)
HMTR evaluates many candidate geometries per outer iteration and uses a trust‑region micro‑solver to refine them. Barranquilla batches the energy/gradient calls across the candidate set and assigns them to GPU(s) in round‑robin fashion, using one CUDA stream per host thread. The trust‑region algebra (model assembly, radius adaptation, BFGS updates) remains on the host, while the dominant cost — the electronic energy and gradient at each candidate geometry — is evaluated on device. This division leverages GPU throughput where it counts without complicating the optimizer math.

Memory, determinism, and precision
All SCF arrays that live on the GPU use double precision. Packed‑lower formats are used for \(F\) and (optionally) \(P\) and are unpacked only for GEMM/SYRK operations that require full column‑major storage. Deterministic cuBLAS/cuSOLVER settings are recommended for regression testing; they can be enabled via environment. Resident mode minimizes host/device transfers on medium problems; streaming minimizes peak device memory on very large problems by staging only the needed pair blocks at a time.

Multi‑GPU
For dense BLAS‑3 (density formation) in conventional SCF, cuBLASXt can utilize multiple GPUs when available and configured. For MOZYME, pair batches can be partitioned across devices explicitly or via a simple device map. The current eigensolver path uses single‑GPU cuSOLVER; multi‑GPU cuSOLVERMg can be enabled on supported platforms.

### Code Dataflow Summaries

Conventional SCF — resident mode (medium systems)
```
host:   P(0) → Fock(P) → F, C, ε → Density(C) → P(1)
device:        ↑                 ↘            ↘
               J/K (general)      Dsyevd(F)    SYRK/GEMM(C_occ)
               (batched pairs)                 → P

Resident buffers: F(dev), P(dev), V(dev) (eigenvectors). Packed‑lower storage for F, unpacked temporaries for BLAS as needed.
```

Conventional SCF — streaming mode (very large systems)
```
host:   enumerate (A,B) pairs → publish W or (WJ,WK) slices → finalize → F(host)
device:                                    accumulate J/K into F(dev)

Only F(dev) and small staging buffers exist on device; density/P are uploaded once per SCF iteration. Disk/host streaming reduces VRAM pressure.
```

MOZYME
```
host:   build localized blocks → if size ≥ MINBLK: launch F2/DF2 kernel → reduce to F
device: batch over block list (single/multi‑GPU), small dense kernels over local neighborhoods.

Very small blocks remain on CPU; large blocks are offloaded to maximize GPU occupancy.
```

HMTR
```
host:   propose θ_i (torsions) → form R_i candidates → batch E,∇E on device → trust‑region refine per candidate → accept/reject by ρ → update Δ
device: evaluate many E,∇E calls concurrently (one stream per host thread) using the same SCF stack as single‑geometry runs.

All trust‑region algebra (model m_k, radius/ρ update, BFGS) remains on host; GPU is used where the cost lies (E,∇E).
```

---

## Appendix: Streaming API (Conventional SCF)

The streaming path feeds atom‑pair J/K slices to the device without staging a monolithic integral table. The Fortran side (`src/gpu/gpu_scf_stream_driver.F90`) wraps a C++ driver (`src/gpu/scf_driver.cu`).

Registration
```
call mopac_cuda_scf_stream_register(cookie_ptr)

cookie carries:
  norbs, mpack, numat
  ptot, p               (packed densities)
  f                     (packed Fock accumulation target)
  nfirst, nlast         (per‑atom AO ranges)
  periodic_flag         (0=non‑periodic, 1=periodic)
```

Publish blocks
```
call mopac_cuda_scf_stream_publish(cookie_ptr, ia, ib, ja, jb, len, wj_slice, wk_slice, status)

non‑periodic (molecules): publish legacy W slices for compact/general blocks
  wj_slice = W(kk+1:kk+len), wk_slice = W(kk+1:kk+len)
periodic: publish split WJ/WK slices (wk optional; falls back to WJ if absent)

Constraints:
  len must match the enumerated pair block length for (ia..ib, ja..jb)
  publish order must match the driver’s enumeration order
```

Finalize and fetch
```
call mopac_cuda_scf_stream_finalize(cookie_ptr, status)

On success, F(dev) is registered with the runtime and copied back to F(host) if resident cache fetch fails.
```

Pair classification and lengths (C++, `scf_driver.cu`)
The driver builds an expected block list over all `1 ≤ jj < ii ≤ numat`, computes spans `span_i = ib−ia+1` and `span_j = jb−ja+1`, and assigns a kind:
```
  LL: span_i=1, span_j=1,   len=1             (legacy W)
  HL/LH: one span=4, other=1, len=10          (legacy W)
  HH: span_i=4, span_j=4,    len=100          (legacy W)
  d‑containing: any span≥7,   len=pair_i×pair_j (legacy W)
  periodic general:           len=pair_i×pair_j (split WJ/WK)
  fallback general:           len=pair_i×pair_j (legacy W)
```
Offsets (`kk`) are advanced by `len` per published block. On the device, only a small staging buffer for `W` (or `WJ/WK`) and the packed Fock exist; densities (`P`, `P_tot`) and `nfirst/nlast` are uploaded once per SCF iteration.

Determinism and verification
Set `MOPAC_GPU_VERIFY_FOCK=1` to compare device and host J/K contributions (debug‑only). `MOPAC_GPU_STREAM_TRACE=on` writes a trace of published blocks (atom ranges and lengths) to help audit publish ordering.

Memory model and layouts
Packed‑lower storage is used for symmetric matrices (F, optionally P). When a BLAS operation requires column‑major full storage, an unpack step materializes the full matrix on device, the BLAS kernel runs, and results are repacked as needed.


---

## Automatic Policies

At startup Barranquilla MOPAC measures the system and GPU and applies:
- Small SCF (very small AO basis): GPU off (CPU is faster). Default threshold `norbs < 30`.
- Medium SCF: GPU on for general J/K; resident SCF on; streaming not forced.
- Large SCF: GPU on + streaming (fast NVMe TMPDIR recommended).
- MOZYME: GPU pair kernels enabled on newer GPUs; can be forced via env on older ones.

These are printed in the run header when `MOPAC_GPU_DEBUG=1` is set.

---

## GPU Usage Cheat Sheet (Keywords, Envs, GPUs)

Key ideas
- Conventional SCF (no MOZYME) offloads the dominant two‑center J/K build to GPU and shows the clearest speedups.
- MOZYME accelerates density (batched SYRK/GEMM) on GPU for large localized blocks; current pair kernels primarily benefit d‑orbital cases, while protein s/p pair work remains CPU‑side.
- Resident vs. streaming is automatic by size; both can be overridden with environment flags.

Build configuration
- Enable: `-DGPU=ON`
- Target SMs: `-DCUDA_ARCHS=<sm>`
  - Maxwell TITAN X: `52`
  - V100: `70`  •  A100: `80`  •  H100: `90`
  - RTX 30/40 (limited FP64): `86`/`89`

Common environment flags
- Force/disable GPU: `MOPAC_FORCEGPU=1`, `MOPAC_NOGPU=1`
- SCF task: `MOPAC_GPU_SCFTASK=gpu|cpu|auto`, optional driver: `MOPAC_GPU_SCF_EXPERIMENTAL=1`
- Residency: `MOPAC_RESIDENT_SCF=1` (prefer resident on medium systems)
- Diagnostics: `MOPAC_GPU_PROFILE=1|2`, `MOPAC_GPU_VERBOSE=1`, `MOPAC_GPU_SCF_DEBUG=1`, `MOPAC_GPU_STREAM_TRACE=1`
- Eigenvectors host fetch (printing): set `MOPAC_EIG2HOST` non‑empty

MOZYME (localized) flags
- Enable/disable: keyword `MOZYME_GPU`, or env `MOZYME_GPU_FORCE=1` / `MOZYME_GPU_OFF=1`
- Pair kernels (F2/DF2): `MOPAC_MOZYME_F2_GPU=1` (on). Current kernels mainly benefit d‑orbital cases.
- Block threshold: `MOZYME_MINBLK=<n>` (2–4 for big blocks; 1 to be aggressive)
- Two GPUs: `MOZYME_GPUPAIR=a,b` (1‑based)

Recommended profiles by GPU
- Maxwell sm_52 (TITAN X):
  - Build with `-DCUDA_ARCHS=52`.
  - Conventional SCF: `MOPAC_FORCEGPU=1 MOPAC_RESIDENT_SCF=1` (visible speedups on large cases).
  - MOZYME proteins: prefer CPU density (`MOZYME_GPU_OFF=1`) and use threaded BLAS.

- Volta V100 / Ampere A100 / Hopper H100 (strong FP64):
  - Build with `-DCUDA_ARCHS=70|80|90`.
  - Conventional SCF: `MOPAC_GPU_SCFTASK=gpu MOPAC_GPU_SCF_EXPERIMENTAL=1`.
  - MOZYME: `MOZYME_GPU MOZYME_MINBLK=2–4` and `MOPAC_MOZYME_F2_GPU=1`; multi‑GPU via `MOZYME_GPUPAIR`.

- RTX 30/40 (sm_86/89, limited FP64):
  - Conventional SCF still helps for large systems; MOZYME density offload helps on bigger blocks; s/p pair work remains CPU.

Quick recipes
- Conventional SCF (GPU, medium): `MOPAC_FORCEGPU=1 MOPAC_RESIDENT_SCF=1`
- Conventional SCF (GPU, very large/streaming): add `MOPAC_GPU_PROFILE=2`
- MOZYME (proteins) density on GPU: `MOZYME_GPU MOZYME_MINBLK=2–4` and `MOPAC_MOZYME_F2_GPU=1`

## GPU Configuration Matrix (Scenarios → Settings)

- Conventional SCF — molecular (non‑periodic)
  - Small (norbs < 30): default CPU; optional test `MOPAC_FORCEGPU=1` (little/no speedup).
  - Medium (30–800): `MOPAC_FORCEGPU=1 MOPAC_RESIDENT_SCF=1`; leave `MOPAC_GPU_SCFTASK=auto` (or `gpu`).
  - Large (> 800): same as medium + profiling: `MOPAC_GPU_PROFILE=2`; log shows `[STREAM] engaged` when streaming.

- Conventional SCF — periodic (solids)
  - Uses split `WJ/WK` blocks and prefers streaming. Keep `MOPAC_FORCEGPU=1`; add `MOPAC_GPU_PROFILE=2` for visibility.
  - Optional debug: `MOPAC_GPU_SCF_DEBUG=1` and `MOPAC_GPU_STREAM_TRACE=1` to trace block publishes.

- MOZYME — proteins (mostly s/p shells)
  - GPU helps density BLAS on large localized blocks: set `MOZYME_GPU` and `MOZYME_MINBLK=2–4`.
  - Pair kernels (F2/DF2) remain CPU for s/p; `MOPAC_MOZYME_F2_GPU=1` mainly helps d‑orbital cases.
  - On weak‑FP64 GPUs (Maxwell, consumer RTX), CPU MOZYME often faster: `MOZYME_GPU_OFF=1` + threaded BLAS (`THREADS` keyword, `OMP_NUM_THREADS`/`MKL_NUM_THREADS`).

- MOZYME — with transition metals (d shells present)
  - Enable F2 GPU: `MOPAC_MOZYME_F2_GPU=1`; keep `MOZYME_GPU` and moderate `MOZYME_MINBLK` (2–4).
  - Multi‑GPU density: `MOZYME_GPUPAIR=1,2` when two devices are available.

- Gradients
  - Default uses same algebra as energies. An experimental Coulomb‑only GPU gradient exists: `MOPAC_GPU_GRAD_EXPERIMENTAL=1` (for development/diagnostics).

- Multi‑GPU eigensolver (experimental)
  - Build: `-DUSE_CUSOLVER_MG=ON`. Runtime: set `MOPAC_EIG_MG=1` (and optionally `MOPAC_EIG_MG_GRID`, `MOPAC_EIG_MG_BLKSIZE`, `MOPAC_EIG_MG_MIN`).

- Device selection and ignore lists
  - Select a device: input keyword `SETGPU=<n>` (1‑based). Ignore devices: `MOZYME_GPUIGNORE=a,b,c`.
  - Two‑GPU pairing for MOZYME: `MOZYME_GPUPAIR=a,b`.

- Troubleshooting no GPU utilization
  - Verify build targets your SM (`-DCUDA_ARCHS` in compile lines).
  - Check profile lines in the log: `MOZYME_GPU`, `MOZYME_F2_GPU`, `resident_scf`, `[STREAM] engaged`.
  - Integral‑on‑disk disables the full GPU SCF driver; streaming Fock can still be active.
  - Enable `MOPAC_GPU_SCF_DEBUG=1` / `MOPAC_GPU_STREAM_TRACE=1` to see driver/stream activity.

---

## Build & Run

Build (CPU+GPU)
```
cmake -S . -B build-gpu -G Ninja -DGPU=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-gpu --parallel
```

Small molecule (CPU preferred)
```
./build-gpu/mopac ethanol.mop
```

Medium/large molecule (SCF on GPU)
```
export CUDA_VISIBLE_DEVICES=0
./build-gpu/mopac big.mop
```

Protein (MOZYME)
```
PM7 GEO_DAT=protein.pdb MOZYME MOZYME_GPU MOZYME_MINBLK=4 PULAY ITRY=200
./build-gpu/mopac protein_gpu.mop
```

Tip: remove `1SCF` and raise `ITRY` to observe actual convergence.

---

## Advanced Controls (Optional)

GPU/SCF policy
- `MOPAC_GPU_MIN_NORBS=NN`  Auto‑guard cutoff (default 30).
- `MOPAC_GPU_AUTOPOLICY_OFF=1`  Disable auto policy.
- `MOPAC_FORCEGPU=1`  Force GPU even if small.
- `MOPAC_RESIDENT_SCF=1`  Keep SCF data resident on GPU.

SCF hybrid mixer
- `EDIIS` keyword: enable hybrid mode.
- `MOPAC_SCF_HYBRID=on`  Hybrid without keyword.
- `MOPAC_SCF_SWITCH_PL`, `MOPAC_SCF_EDIIS_ITERS`  Control EDIIS→CDIIS switch.
- `MOPAC_SCF_DAMP=on`, `MOPAC_SCF_ALPHA_MIN/MAX/K`  Damping parameters.
- `MOPAC_SCF_ADAPTIVE_SHIFT=on`  Adaptive level shift schedule.

MOZYME
- `MOZYME`, `MOZYME_GPU`, `MOZYME_MINBLK=INT`  Enable and tune MOZYME GPU.
- `MOPAC_MOZYME_F2_GPU=1`  Force pair GPU on older GPUs.

Verification and profiling
- `MOPAC_GPU_PROFILE=1`  Prints GPU pair counts and timings.
- `MOPAC_GPU_VERIFY_FOCK=1`  Debug verifier for GPU J/K vs CPU.

---

## Performance Expectations

- Older GPUs (e.g., TITAN X CC 5.2): modest DP throughput; expect small–moderate gains from MOZYME and general J/K offload.
- Modern DP GPUs (V100/A100/H100): conventional SCF 2–6×; MOZYME 2–6× on large systems with resident/streaming and tuned block sizes.

Best practices
- Use MOZYME for biomolecules; tune `MOZYME_MINBLK` (3–6).
- Let the auto‑guard keep tiny SCFs on CPU.
- Place TMPDIR on fast NVMe for streaming.
- Use `EDIIS` for tough SCFs.

---

## Roadmap

- Host‑side pre‑expansion of compact LL/HL/HH blocks to the “general” order for one robust GPU kernel.
- cuSOLVERMg multi‑GPU diagonalization on supported platforms.
- Deeper overlap and batched pair launches to reduce kernel overhead.
---

## Development & Collaboration

- Developed by: Dr. Juvenal Yosa (UMCG — Groningen, Netherlands)
  - Address: Hanzeplein 1, 9713 GZ Groningen, The Netherlands
- In collaboration with: Universidad Simón Bolívar — Barranquilla, Colombia
  - Address: Carrera 54 # 64 - 222, Barranquilla, Atlántico, Colombia
- And: Protyon — Groningen, Netherlands
  - Address: Winschoterdiep 50, 9723 AB Groningen, The Netherlands

---

© 2025 Barranquilla MOPAC (OpenMOPAC). Apache‑2.0 License.

---

Publication note
- A peer‑reviewed publication describing Barranquilla MOPAC’s algorithms and GPU implementation is in preparation.
- In the meantime, please cite this repository in scholarly work that uses this software.

BibTeX (repository)
```
@misc{barranquilla_mopac_repo,
  author       = {Yosa, Juvenal and Melcr, Josef and van der Wekken, Anthonie and Groves, M. R.},
  title        = {Barranquilla MOPAC — A GPU-Accelerated Semiempirical Flavor of MOPAC},
  year         = {2025},
  howpublished = {GitHub repository},
  note         = {UMCG Groningen; Universidad Simón Bolívar; Protyon},
  url          = {https://github.com/juvenalyosa/mopac_gpu}
}
```

Also consider citing the original MOPAC (Apache 2.0) distribution when appropriate:
```
@software{mopac_software,
  title        = {MOPAC},
  author       = {Jonathan E. Moussa and James J. P. Stewart},
  year         = {2025},
  version      = {23.1.2},
  doi          = {10.5281/zenodo.6511958}
}
```

Forthcoming Barranquilla paper (placeholder BibTeX)
```
@misc{barranquilla_mopac_yosa_2025,
  author       = {Yosa, Juvenal and Melcr, Josef and van der Wekken, Anthonie and Groves, M. R.},
  title        = {Barranquilla MOPAC: GPU-Accelerated Semiempirical Calculations for Large Systems},
  year         = {2025},
  note         = {In preparation},
  url          = {<preprint_or_doi_when_available>}
}
```

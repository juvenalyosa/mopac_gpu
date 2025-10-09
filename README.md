# Barranquilla MOPAC — A GPU‑Accelerated Semiempirical Flavor of MOPAC

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

## GPU Implementation

Conventional SCF
- Two‑center J/K build: the heavy “general” blocks are offloaded to the GPU. Compact corner cases (LL/HL/HH) run on CPU by default to preserve accuracy (and because they are cheap on CPU). Streaming is used for very large problems to feed J/K slices to the device without building a giant in‑memory buffer.
- Eigensolver and density: cuSOLVER Dsyevd on device, with eigenvectors kept resident; SYRK/GEMM used to form densities. Near the fixed point, resident SCF reduces PCIe transfers.

MOZYME (proteins)
- Localized pair kernels (F2/DF2) offloaded to GPU when blocks exceed a minimum size (MOZYME_MINBLK), with single‑ or multi‑GPU policies depending on architecture. Large proteins benefit substantially from this path.

Design choices for correctness
- Compact LL/HL/HH pairs remain on CPU by default (exact by construction) while general J/K dominates runtime on GPU.
- Verification hooks allow A/B testing (debug) without runtime cost in production.

---

## Automatic Policies

At startup Barranquilla MOPAC measures the system and GPU and applies:
- Small SCF (very small AO basis): GPU off (CPU is faster). Default threshold `norbs < 30`.
- Medium SCF: GPU on for general J/K; resident SCF on; streaming not forced.
- Large SCF: GPU on + streaming (fast NVMe TMPDIR recommended).
- MOZYME: GPU pair kernels enabled on newer GPUs; can be forced via env on older ones.

These are printed in the run header when `MOPAC_GPU_DEBUG=1` is set.

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

© 2025 Barranquilla MOPAC (OpenMOPAC). Apache‑2.0 License.

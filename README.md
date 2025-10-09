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

Semiempirical NDDO/PMx Hamiltonians (e.g., PM7) balance physics‑based interactions with parametrized approximations. They replace costly four‑center integrals with analytic formulas and tabulated parameters, dramatically reducing the expense of assembling the Fock matrix. This makes semiempirical methods ideal for large systems:

- Biomolecules (proteins, DNA/RNA): localized electronic structure and nearsightedness of matter make interactions short‑ranged; MOZYME’s localized orbital machinery gives near‑linear scaling in practice.
- Polymers and materials: repeated motifs and short‑range couplings map well to block‑sparse contractions.
- Long trajectories and scans: faster SCF cycles and robust convergence reduce wall‑clock and failure modes.

Barranquilla MOPAC layers on top of that:
- GPU acceleration of the heavy “general” two‑center J/K build (dominant cost on large systems) while keeping compact corner cases on CPU for correctness.
- A hybrid SCF mixer (EDIIS→CDIIS) with adaptive level shifting and damping to reduce iterations without sacrificing predictability.
- Automatic policies so you only choose the number of GPUs; the code picks the rest.

---

## Physics Background — SCF, Fock Build, and Energy

Let `S` be the overlap, `H` the one‑electron core Hamiltonian, `P` the (spin‑summed) density, and `F` the Fock matrix. For RHF (closed shell) in an AO basis:

- Fock build (semiempirical PMx):
```
F = H + G[P] = H + J[P] − K[P]
```
where the two‑electron contribution `G[P]` is evaluated with NDDO/PMx formulas and pretabulated parameters. In practice we evaluate `J`/`K` as batched two‑center contractions over atom blocks.

- Density build (RHF):
```
P = 2 · C_occ · C_occ^T
```
or with fractional occupations for open shells/temperature smearing as needed.

- Total electronic energy (AO packed form):
```
E_elec = 1/2 · Tr[P · (H + F)]
E_total = E_elec + E_nuclear
```

- Commutator residual (generalized):
```
R = F P S − S P F
```
and SCF convergence is judged by ‖R‖ (plus energy and density deltas).

CDIIS (Pulay): builds an optimal linear combination of past Fock/density pairs by minimizing the residual norm subject to sum‑to‑one constraints. It converges rapidly near the fixed point but can overshoot far from it. EDIIS forms a convex energy‑minimizing combination and is more stable far from self‑consistency.

Barranquilla MOPAC uses a hybrid: begin with EDIIS‑like damped mixing and adaptive level shift, then switch on CDIIS automatically when the residual is small enough.

---

## HMTR Geometry Optimization — Trust‑Region with GPU Support

HMTR stands for Hierarchical Memetic Trust‑Region:
- H (Hierarchical): multi‑level organization of candidates (global → local), enabling coarse‑to‑fine searches.
- M (Memetic): population‑based proposals combined with local refinement (a “memetic” hybrid of global and local search).
- TR (Trust‑Region): rigorous local models (quadratic) solved within a radius Δ, with radius adaptation via the acceptance ratio ρ.

Geometry optimization minimizes the potential energy surface `E(R)` with respect to nuclear coordinates `R`. HMTR combines a trust‑region model with population‑based exploration (torsional memetics) and batched GPU evaluation.

1) Quadratic model around `R_k`:
```
m_k(p) = E(R_k) + g_k^T p + 1/2 p^T B_k p
```
where `g_k = ∂E/∂R |_(R_k)` and `B_k` is a Hessian (or quasi‑Newton) approximation. The trial step `p` solves the trust‑region subproblem `‖p‖ ≤ Δ_k`.

2) Acceptance ratio:
```
ρ_k = (E(R_k) − E(R_k + p)) / (m_k(0) − m_k(p))
```
Update the radius Δ_k by comparing `ρ_k` to thresholds η₁ < η₂ (e.g., η₁=0.25, η₂=0.75): shrink Δ if ρ is small (poor model), expand if ρ is large (good model), otherwise keep Δ.

3) Quasi‑Newton update of `B_k` (e.g., BFGS):
```
s_k = p,   y_k = g(R_k + p) − g(R_k)
B_{k+1} = B_k − (B_k s_k s_k^T B_k) / (s_k^T B_k s_k)
          + (y_k y_k^T) / (y_k^T s_k)
```

HMTR augments 1–3 with a memetic torsion subspace and a particle‑swarm‑like outer loop to propose diverse trial moves, then refines locally with the trust‑region micro‑optimizer (radius/rho thresholds are in `src/optimization/hmtr.F90`). Energies and gradients are evaluated batched on the GPU (when enabled) for the trial population, which is effective for large biomolecules where many local proposals can be examined concurrently.

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

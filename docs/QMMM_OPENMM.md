# QM/MM molecular dynamics with OpenMM and MOPAC/MOZYME

The QM region is computed by MOPAC (PM7, MOZYME, on the GPU when there is one) in the field of the MM
point charges (electrostatic embedding); the force field, water, PME, constraints, integrators,
thermostats, barostats and reporters are OpenMM's. Any OpenMM integrator or ensemble works, because the
coupling is an OpenMM `PythonForce`.

## Requirements

- OpenMM ≥ 8.4 (tested with 8.6.1; `conda install -c conda-forge openmm` or `pip install 'openmm[cuda12]'`),
  NumPy and SciPy; CuPy (`pip install cupy-cuda12x`) puts the QM-PME grid on the GPU
- this MOPAC built as a shared library (`libmopac.so` / `libmopac.dylib`, built with the `mopac` target;
  `-DGPU=ON` for the GPU)
- `scripts/qmmm/openmm_mopac.py`, `mopac_qmmm_api.py` and `qmmm_pme.py` on the Python path

## Use

```python
import openmm as mm, openmm.app as app, openmm.unit as u
from openmm_mopac import add_qmmm

system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=1*u.nanometer,
                                 constraints=app.HBonds)
qmmm = add_qmmm(system, topology, positions, qm_atoms, qm_charge=0, libmopac="/path/libmopac.so",
                ron=8.0, roff=10.0, tolerance=0.01)      # Angstroms
integrator = mm.LangevinMiddleIntegrator(300*u.kelvin, 1/u.picosecond, 0.5*u.femtoseconds)
simulation = app.Simulation(topology, system, integrator, mm.Platform.getPlatformByName("CUDA"))
...
qmmm.close()
```

`scripts/qmmm/run_qmmm_md.py` does the whole job from a PDB file (hydrogens, water box, MM
equilibration, QM region from residue numbers, optionally side chains only, NVT, an NVE check) and prints
the time per step:

```
python scripts/qmmm/run_qmmm_md.py protein.pdb --libmopac build/libmopac.so --qm-residues 45,72 \
    --qm-sidechain --steps 2000 --dt 0.5 --platform CUDA
```

## What `add_qmmm` does

- MM bonds, angles and torsions whose atoms are all QM are switched off; constraints between QM atoms are
  removed (MOPAC describes them).
- The QM atoms keep their Lennard-Jones parameters (QM-MM repulsion and dispersion) but get zero charge in
  `NonbondedForce`; QM-QM pairs are excluded. The QM-MM electrostatics come from MOPAC, with the QM density
  polarized by the MM charges.
- A QM-MM covalent bond gets a hydrogen link atom at Q + g (M - Q), g = (Q-H)/(Q-M) bond length; its force
  is shared between Q and M. The QM region does not see the charge of the MM boundary atom M.
- The MM charges are switched on and off by groups (residues: a water, an ion, an MM residue): all charges
  of a group are multiplied by the same S(d), d = smooth minimum of the distances from the group centre to
  the QM atoms, S = 1 below `ron` and 0 beyond `roff` (quintic). Each group is imaged as a whole.
- Long-range electrostatics (QM-PME, on by default when the `NonbondedForce` uses PME or Ewald; `pme=False`
  turns it off): the Mulliken charges Q_A of the QM atoms interact with all MM charges and their periodic
  images by Ewald summation (smooth PME in `qmmm_pme.py`), minus the bare Coulomb interaction with the
  switched charges that MOPAC already sees:
  E_far = Σ_A Q_A φ_far(A), φ_far(A) = φ_Ewald(A) − Σ_j S_j q_j / r_Aj.
  φ_far enters MOPAC's Fock matrix (diagonal of atom A, as MOL_QMMM does), so the energy stays variational
  and the forces are its exact derivatives. A charge near the QM region acts through the QM integrals with
  weight S and through the Mulliken charges with weight 1 − S, so the result hardly depends on `ron` and
  `roff`. The QM region does not interact with its own periodic images.

## Details and limits

- Units: OpenMM nm and kJ/mol, MOPAC Angstroms and kcal/mol. The QM energy is the PM7 heat of formation
  of the QM region in the field of the MM charges, so the potential energy is offset by a constant.
- The periodic box must hold the QM region plus `roff` on every side (checked; an error says so).
- `tolerance` is MOPAC's relative SCF criterion (default 0.01); below 0.1 MOPAC also uses `THRESH=1.D-15`.
  An SCF that does not converge stops the run with an error. 0.001 costs ~1.8 x the SCF time of 0.01.
- `guess` (`add_qmmm(..., guess=...)`, `run_qmmm_md.py --guess`, or the environment variable
  `MOPAC_QMMM_GUESS`) sets where the SCF of each step starts: `previous` (default), the converged LMOs of
  the last step; `linear` or `aspc`, LMOs extrapolated from the last 2 or 4 steps with the predictor
  coefficients of Kolafa's always stable predictor-corrector (J. Comput. Chem. 25, 335 (2004); `aspc`:
  2.8, -2.8, 1.2, -0.2). The extrapolation acts on the projectors onto the occupied space, P(n-k), applied
  to the LMOs of the last step (occupied C' = B1 C(n) + Sum B(k+1) P(n-k) C(n), virtual
  V' = V(n) - Sum B(k+1) P(n-k) V(n)), which does not depend on how the LMOs of a step are rotated among
  themselves. Three things made earlier versions fail (SCF error in CHECK): extrapolating the coefficients
  themselves (the rotations among occupied LMOs, which the SCF never corrects, grew); polynomial
  extrapolation of degree 2 or 3 (B = 3, -3, 1 or 4, -6, 4, -1: trypsin-BPTI diverged within 35-85 steps,
  whereas `linear` and `aspc` have only a double root at 1); and keeping the converged LMOs as they were
  (each SCF leaves them non-orthogonal by ~1e-6 to 1e-5, so the P(n-k) were not projectors; they are now
  re-orthonormalized first). The time steps must be equal: a call at the same geometry is skipped, and a
  step more than twice as long or short as the last restarts the history. **`linear` and `aspc` are not
  recommended**: they halve the SCF iterations, but on the GPU they made the step slower and in NVE runs of
  trypsin-BPTI they heated the system in episodes. NVE of 1 fs, SCF iterations per step and the drift from
  the QM/MM force (kJ/mol/ps):

  | system, run | `guess` | `tolerance=0.01` | | `tolerance=0.001` | |
  |---|---|---:|---:|---:|---:|
  | chignolin Tyr2 (21 + 2 link atoms), CPU, 10 ps | `previous` | 17.3 | -4.12 ± 0.03 | | |
  | | `aspc` | 8.4 | -0.09 ± 0.02 | 17.3 | +0.03 ± 0.02 |
  | trypsin-BPTI (54 + 5), CPU, 2 ps | `previous` | 15.2 | -8.5 ± 0.2 | | |
  | | `aspc` | 8.4 | +1.5 ± 0.6 | 14.6 | -0.8 ± 0.4 |
  | trypsin-BPTI (54 + 5), A100, 5 ps | `previous` | 14.9 | -8.4 ± 0.4 | 24.2 | +2.2 ± 0.2 |
  | | `aspc` | 8.4 | +176 ± 17 | 14.4 | +8.5 ± 2.4 |

  (CPU: MOPAC and OpenMM on an Apple M-series CPU, trypsin after 2 ps of MM equilibration only; A100: MOPAC
  on the GPU, OpenMM CUDA in mixed precision, 50 ps of MM equilibration, commit 6ea3c4b5.) With `previous`
  the QM/MM part drifts steadily; with `aspc` it came in episodes: on the A100, 0.5-ps blocks of +340 to
  +640 kJ/mol/ps at 0.01, and +28 to +79 in the last 1.5 ps at 0.001, the earlier blocks being like
  `previous`; on the CPU, one 0.2-ps block of +20 in the 2-ps run at 0.01. Chignolin showed none in 10 ps.
  A likely reason, not proven: both schemes reproduce a linear trend exactly, so an error that the SCF
  leaves in place (one that the energy criterion does not see) is carried forward as if it were motion.
  The extrapolation runs on the host: on the A100 it took 44 ms per step (`MOPAC_GPU_PROFILE=1`, section
  `qmmm_lmo_guess`), more than the whole SCF, so a 0.5-fs step of trypsin-BPTI took 0.095 s with `aspc`
  against 0.059 s with `previous` at 0.01 (5.7 and 13.5 iterations), and 0.109 against 0.074 s at 0.001
  (11.1 and 21.6). An SCF stopped at the tolerance keeps part of the error of its starting point: LMOs
  that lag behind the geometry lose energy, and a tighter tolerance makes the error smaller.
  `mozyme_qmmm_scf_iterations()` (C API) / `QMSession.scf_iterations` return the iterations of the last
  step.
- Time step: the bonds of the QM region are free (`add_qmmm` removes the constraints between QM atoms;
  MOPAC describes them), and with 1-fs steps velocity Verlet makes energy errors of up to 1-3 kJ/mol in
  single steps in which an atom of the QM region moves fast (trypsin-BPTI: Hδ1 of His57, H of Lys15 and
  Ala16). The QM/MM force is the exact gradient there: the QM/MM energy along such a step, integrated in ten
  sub-steps, changed by +0.015 kJ/mol against +2.18 from the trapezoid rule over the whole step (three
  steps checked). These errors partly cancel and partly add up like a random walk. Trypsin-BPTI,
  tolerance 0.001, MOPAC on the CPU, OpenMM OpenCL (single precision), energy and work of the QM/MM force
  checked at every step (local preparation, 1 to 3 ps each):

  | | length | SD per step, kJ/mol | SD of the sums over 0.2 ps (blocks) |
  |---|---:|---:|---:|
  | 1 fs | 2.8 ps | 0.108 | 1.09 (14) |
  | 0.5 fs | 1.0 ps | 0.034 | 0.23 (5) |
  | 1 fs, `qm_constraints=True` | 1.5 ps | 0.151 | 0.65 (7) |

  The same check on the A100 (MOPAC on the GPU, OpenMM CUDA in mixed precision, the preparation of the
  A100 rows above, tolerance 0.001, 5 ps of NVE after 0.4 ps at 300 K, seeds 11 to 13, commit 229b6fe4,
  `colab/qmmm_dt_colab.ipynb`), from E_qm + W read every 10 fs (mean ± SD of the three runs, or the three
  values):

  | | SD of the 10-fs increments, kJ/mol | SD of the sums over 0.2 ps | increments > 1 kJ/mol | drift from the QM/MM force, kJ/mol/ps | s per step | s per ps |
  |---|---:|---:|---:|---:|---:|---:|
  | 1 fs | 0.29 ± 0.01 | 1.24 ± 0.30 | 6, 3, 7 | +1.6, -0.3, +3.9 | 0.089 | 89 |
  | 1 fs, `qm_constraints=True` | 0.22 ± 0.05 | 0.82 ± 0.17 | 4, 2, 9 | +1.5, 0.0, -0.2 | 0.088 | 88 |
  | 0.5 fs | 0.072 ± 0.007 | 0.20 ± 0.02 | 0, 0, 0 | -0.6, -0.2, -0.8 | 0.077 | 155 |
  | 0.5 fs, `tolerance=0.01` (seed 11) | 0.072 | 0.21 | 0 | -8.8 | 0.061 | 122 |

  (Step times of these NVE runs, which read energies and forces every 10 fs; in their NVT part a 1-fs step
  took 0.072-0.077 s and a 0.5-fs step 0.067 s, or 0.051 s at 0.01.) With 0.5 fs the increments had no
  tail (none above 0.5 kJ/mol; excess kurtosis -0.2 to 0.8, against 23 to 43 at 1 fs), the drift from the
  QM/MM force had the same sign in the three runs, and the slopes of the total energy were -0.2 ± 0.7,
  +0.6 ± 0.5 and -1.0 ± 0.6 kJ/mol/ps. The loss left at 0.5 fs (-0.2 to -0.8 kJ/mol/ps) is probably the
  SCF's: at 0.01 it was -8.8, against -8.4 with 1-fs steps in the table above, so it did not depend on the
  step; a run at 0.0001 would be needed to show it. 0.5-fs steps take 1.7-1.8 times as long per ps as 1-fs
  steps (fewer SCF iterations per step, 21 against 24, do not make up for twice as many steps).

  Use 0.5 fs (the default of `run_qmmm_md.py`), and in NVE also `tolerance=0.001` (26-31% more per step
  than 0.01 here). `qm_constraints=True` (`--qm-constraints`) keeps the force field's X-H constraints
  inside the QM region, as Amber's `qmshake=1` (its default when SHAKE is on) does, and `qm_free`
  (`--qm-free E:57:HD1`) leaves chosen atoms free (a proton that may move); the constrained lengths are the
  force field's, not PM7's. At 1 fs it made the accumulated errors about a third smaller but left as many
  jumps (15 increments above 1 kJ/mol in the three runs, against 16): heavy atoms of the QM region move fast
  too (the ring of His57 in the one constrained jump checked).
- The H-bond correction of PM7 is used with `HB-SMOOTH` (its original form has kinks that MD and FORCE
  cannot cross).
- QM-PME needs an orthorhombic box. Its parameters are fields of `qmmm.region` (`pme_alpha` = 0.25 1/A,
  `pme_cutoff` = 13 A real space, `pme_spacing` = 1.2 A, `pme_order` = 4): potential errors ~1e-4 of the
  potential, and the forces are exact derivatives of the energy as computed. It runs on the GPU with CuPy
  when CuPy sees one (`add_qmmm(..., pme_gpu=False)` or `run_qmmm_md.py --pme-cpu` keep it on the CPU),
  else in NumPy: grid work for 46 000 MM atoms 13 ms on a T4 against 100 ms in NumPy on the Kaggle CPU
  (~45 ms on an Apple M-series CPU); CuPy and NumPy agree to 1e-15. With PME the QM energy includes the far field and a
  constant (net-charge background), so absolute energies of runs with and without PME differ; only
  differences and forces compare.
- One QM region per process; do not run two MOPAC sessions in the same directory.
- Energy conservation depends on `tolerance`: an SCF converged only to the tolerance gives forces that are
  not exactly the derivatives of the energy, and the total energy falls. NVE with velocity Verlet, 1 fs,
  5 ps after 400 steps of NVT (`scripts/qmmm/nve_drift.py`), trypsin-BPTI, two preparations (46 790 and
  46 859 atoms, 50 ps of MM equilibration each), A100, OpenMM in mixed precision; slope of a linear fit ±
  standard error from ten block means, one run per setting and preparation (kJ/mol/ps):

  | `tolerance` | total, prep. 1 | QM/MM part, prep. 1 | total, prep. 2 | QM/MM part, prep. 2 | s per step (NVT) |
  |---|---:|---:|---:|---:|---:|
  | MM only (20 ps) | -0.06 ± 0.12 | - | -0.08 ± 0.06 | - | - |
  | 0.01 | -4.4 ± 0.8 | -5.5 ± 0.3 | -7.6 ± 0.8 | -8.4 ± 0.4 | 0.059 / 0.058 |
  | 0.001 | -0.6 ± 0.6 | +0.8 ± 0.2 | +3.0 ± 0.7 | +2.2 ± 0.2 | 0.076 / 0.071 |
  | 0.0001 | +0.3 ± 0.6 | -0.3 ± 0.2 | +2.7 ± 0.7 | +2.5 ± 0.3 | 0.096 / 0.090 |

  The QM/MM part is the energy of the QM/MM force plus the work it did along the path (zero for an exact
  gradient); in preparation 1 it was estimated after excluding the 40 of 500 readings where OpenMM had moved
  molecules by a box vector, which `nve_drift.py` now handles (preparation 2). The standard errors do not
  include the difference between preparations. With 0.01 the QM/MM part falls steadily (median ten-step
  increment -0.08 kJ/mol in preparation 2). With 0.001 and 0.0001 the median increment is about zero and the
  drift comes from a few ten-step intervals in which the energy of the QM/MM force plus its work jumps by
  1 to 3 kJ/mol: without the ten largest increments (of 460 or 500), the sum over 5 ps was -0.1 and -2.6
  kJ/mol at 0.001 and 0.0001 in preparation 1 and -0.6 and +0.6 in preparation 2, against +5.0, -1.1,
  +10.5 and +11.2 with them. In preparation 2 these jumps were mostly positive. They are errors of the
  1-fs time step, not of the force (see the time-step item below). OpenMM's single precision on CUDA, its
  default, drifts by itself: +11.1 ± 0.4 kJ/mol/ps for the same system with MM only (mixed -0.06, double
  -0.12), so `run_qmmm_md.py` uses mixed on CUDA (`--precision`); the QM/MM step took the same time in
  both. With single precision and 0.01 the two drifts partly cancel (+5.7 ± 0.8 in total). Every step
  converges the SCF to `tolerance`; before 2026-10-14 the SCF of a step could stop early (a rapid exit
  meant for geometry optimization), and the total energy fell by ~30 kJ/mol per 100 fs.

## Speed

### Current state (commit `e6fddf48`, Colab A100 40 GB)

Seconds per 0.5-fs step, with MOPAC and QM-PME on the GPU. Trypsin–BPTI is the system of the example below.

| system | QM atoms | `tolerance=0.01` | `tolerance=0.001` | simulated time per day at 0.001 |
|---|---:|---:|---:|---:|
| trypsin–BPTI | 59 | 0.021 | 0.027 | ~1.6 ns |
| whole chignolin | 138 | 0.031 | — | ~1.4 ns (at 0.01) |
| whole crambin | 642 | 0.245 | — | ~0.18 ns (at 0.01) |

Use 0.5-fs steps and a tolerance of 0.001 for NVE or anything that needs energy conservation (see the
time-step and tolerance items in "Details and limits"). The same trypsin–BPTI step took 0.052 s (0.067 s at
0.001) at commit `89a30daf`, in another session.

These parts of the speed work are on by default. Each switch below turns one off, and each was compared with the
code before it in one session (subsections below):

| what | on by default | off with |
|---|---|---|
| MM-charge work (imaging, switching, forces) in fused CUDA kernels | when CuPy sees a GPU | `QMMM_MM_GPU=arrays` (CuPy array code) or `QMMM_MM_GPU=0` (NumPy) |
| long-range FFTs through cuFFT plans | idem | `QMMM_FFT_PLANS=0` |
| MOPAC's QM atom – MM charge loops on the GPU | MOZYME on the GPU | `MOPAC_QMMM_GPU=0` |
| device buffers from a stream-ordered pool; pair tables kept on the device | GPU builds | `MOPAC_GPU_ASYNC_ALLOC=0`, `MOPAC_GPU_PAIR_TABLE_CACHE=0` |
| `diagg2` with shared-memory staging, staggered chains, dense LMO rows | QM/MM sessions | `MOPAC_MOZYME_DIAGG2_STAGED=0`, `MOPAC_MOZYME_DIAGG2_STAGGER=0`, `MOPAC_MOZYME_DIAGG2_DENSE=0` |
| `diagg1` from dense matrix products | QM/MM sessions | `MOPAC_MOZYME_DIAGG1_DENSE=0` |

Diagnostics:
- `MOPAC_GPU_PROFILE=1`: MOPAC's section and SCF stage times. It adds synchronizations.
- `MOPAC_MOZYME_SCF_TRACE=1`: one line per `diagg2` sweep.
- `MOPAC_NVTX=1`: NVTX ranges of MOPAC's sections and marks at the SCF stage boundaries, for Nsight Systems.
- The notebooks `colab/qmmm_prof_colab.ipynb`, `qmmm_pyprof_colab.ipynb` and `qmmm_speed2`–`qmmm_speed10` hold the
  profiling and validation runs.

The rest of this section records how the time per step came down, with the checks of each change.

### Earlier measurements

OpenMM CUDA + MOPAC, Kaggle T4, 0.5 fs steps, TIP3P water box (`scripts/kaggle/mopac_qmmm`):

| QM region | atoms (QM / all) | MOPAC on the GPU | MOPAC on the CPU (1 core) | NVE drift (GPU run) |
|---|---:|---:|---:|---:|
| Tyr side chain of chignolin (link atom) | 16 / 3 300 | - | **0.017 s/step** (2.6 ns/day) | +2.5 kJ/mol in 200 fs |
| whole chignolin | 138 / 8 100 | **0.19 s/step** (226 ps/day) | 1.66 s/step | -2.2 kJ/mol in 200 fs |
| whole crambin | 642 / 13 000 | **2.47 s/step** (17.5 ps/day) | - | +0.4 kJ/mol in 100 fs |

(`tolerance=0.01`, every step's SCF converged; the drifts are within the per-step fluctuation of the MM
water.)

QM regions of up to 20 atoms (link atoms included) run on the CPU by default; larger ones on the GPU
(`MOPAC_NOGPU=1` / `MOPAC_FORCEGPU=1` override it). A100, chignolin in water with PME, s per step, CPU (12
Xeon threads) / GPU: 10 QM atoms 0.022 / 0.033, 31 atoms 0.076 / 0.055, 43 atoms 0.149 / 0.065, 86 atoms
0.589 / 0.111, 138 atoms 0.176 on the GPU. The crossover lies between 10 and 31 atoms; 20 was not measured.
Until 2026-11-12 the limit was 100 atoms, and the GPU had a floor of ~0.13 s per step below ~100 atoms:
after every call the pair kernels put the CUDA stack limit back, and the next call made the driver grow it
again (~90 ms per step on an A100). Since commit `52c7f964` the QM/MM session keeps it between steps
(`MOPAC_GPU_STACK_RESTORE=d`, set unless already defined; `=all` restores after every call), which holds
~1.2 GB more device memory on an A100. A T4 is weak in double precision; an A100 should be several times faster (README, "Which GPU?").

Trypsin-BPTI (below; 59 QM atoms with the link atoms), Colab A100 40 GB, MOPAC and QM-PME on the GPU,
0.5 fs, `tolerance=0.01`: 0.063 s per step (680 ps/day) at commit `a6a4992e`, 0.127 s before it. That
commit moved the real-space sums of QM-PME into CUDA kernels (0.105 s per step with them in NumPy) and
spread the QM-MM loops of MOPAC over the CPU threads with OpenMP. Of the 0.063 s, MOPAC takes 0.035 s
(the SCF ~0.030 s, 13 iterations on average). The 0.063 s include `run_qmmm_md.py` writing ten frames of the
whole system to `qmmm_traj.pdb` in a 400-step run, ~0.24 s per frame or 6.5 ms per step averaged (measured at
commit `89a30daf`); in long runs the frames are `steps / 10` apart and cost nothing per step.

Profile at commit `89a30daf` (`colab/qmmm_prof_colab.ipynb`, same machine type, one prepared state, 400 NVT steps,
wall time over the steps): 0.052 s at `tolerance=0.01` (two runs) and 0.067 s at 0.001 (21.5 SCF iterations),
against 0.0022 s for OpenMM alone. At 0.01, medians per call: MOPAC 0.032 s (13.3 SCF iterations, 2.4 ms each),
the Python part of the force 0.013 s (0.012 s with MOPAC replaced by an instant stub: imaging and switching over
the ~46 000 MM charges on the CPU), the long-range term 0.003 s, OpenMM with the transfer of positions and
forces ~0.003 s. Inside the SCF (`MOPAC_GPU_PROFILE=1`, which adds ~7 ms per step) the LMO rotations of `diagg2` take 11.7 ms per step and
`diagg1` ~5 ms; `MOPAC_MOZYME_SCF_TRACE=1` shows 2460 rotations per sweep, a longest chain of 58 rotations
on one LMO, ~14 us per rotation on that chain, and ~230 us of lock waits per block. Under `nsys` the GPU was busy
26 ms of a 54-ms step, with 1700 kernel launches, 300 copies and 100 `cudaMalloc`/`cudaFree` pairs per step.
`MOPAC_MOZYME_DIAGG2_STAGED=1` (coefficients in shared memory) made `diagg2` 1.9 ms and the step ~2 ms faster
(49.6 and 49.9 ms against 51.6 and 52.4, two runs each); at the first call (same geometry in every run) its
heats differed from the default by 6e-6 and 2e-6 kcal/mol, two default runs by 2e-6, and the profiled default
run by 4e-6. `MOPAC_MOZYME_DIAGG2_COOPERATIVE=1` doubled the step (101-107 ms).

Four changes at commit `46637c1f`, each with an off switch:

- The imaging, switching and force assembly over the MM charges run on the GPU with CuPy when the
  long-range term does (`QMMM_MM_GPU=0` keeps them in NumPy), and the long-range term takes its input from
  the device.
- QM/MM sessions use the staged `diagg2`: `MOPAC_MOZYME_DIAGG2_STAGED=1` is set unless already defined.
- Device buffers that live for one call (pair gradient, hcore and solrot pair terms, dispersion and H-bond
  corrections) come from CUDA's stream-ordered pool (`MOPAC_GPU_ASYNC_ALLOC=0` gives `cudaMalloc`/`cudaFree`).
- The parameter tables of the MOZYME pair kernels stay on the device while the parameters do not change
  (`MOPAC_GPU_PAIR_TABLE_CACHE=0` rebuilds them every call).

The ordering kernel of the default (unstaged) `diagg2` also scans in parallel now (1.1 ms per step before,
under `nsys`); the staged kernel does not use that ordering.

Measured with `colab/qmmm_speed2_colab.ipynb` on the same machine type, prepared state and protocol as the
profile above. Each figure is one run unless two are given. Two runs of the same setting differed by up to
0.1 ms per step.

| `tolerance=0.01` | s per step | MOPAC | Python | long-range |
|---|---:|---:|---:|---:|
| all four changes | 0.0429 / 0.0428 | 0.0285 | 0.0090 | 0.0025 |
| all four off | 0.0507 / 0.0507 | 0.0317 | 0.0129 | 0.0034 |
| only `QMMM_MM_GPU=0` | 0.0480 | 0.0287 | 0.0131 | 0.0034 |
| only staged off | 0.0440 | 0.0297 | 0.0091 | 0.0025 |
| only async alloc off | 0.0437 | 0.0293 | 0.0091 | 0.0025 |
| only table cache off | 0.0435 | 0.0291 | 0.0090 | 0.0025 |

MOPAC, Python and long-range are medians per call, in seconds. At a tolerance of 0.001 a step took 0.057 s,
against 0.066 s with all four off.

Under `nsys`, 6.8 `cudaMalloc`/`cudaFree` pairs per step remain, against 100 before, and allocation calls take
0.36 ms per step against 1.8 ms. The staged kernel took 9.9 ms per step, against 10.6 ms for the default block
kernel.

The changes do not alter the first call, which sees the same geometry in every run. Heats of all nine runs at
0.01 lay within 6.1e-6 kcal/mol of the profile above (−297.0127265). Over the 36 pairs of runs at 0.01:
- heats differed by up to 9e-6 kcal/mol, and the largest force component by up to 0.043 kJ/mol/nm (largest
  force 2607);
- the two pairs with the same setting differed by 1.2e-6 and 4.9e-6 kcal/mol and by 0.021 kJ/mol/nm.

With MOPAC replaced by an instant stub, the NumPy and the GPU path of the Python part gave the same energies
and forces within 1e-12 on three position sets, one of them with molecules split across the box. The stubbed
call took 11.0 ms on the GPU path against 14.8 ms in NumPy.

Larger QM regions, all four changes against all four off (one run each):

| QM region | s per step | with staged off | all four off | Python per call |
|---|---:|---:|---:|---|
| whole chignolin (138 QM atoms, 200 steps) | 0.080 | 0.083 | 0.095 | 8 ms, against 18 |
| whole crambin (642 QM atoms, 40 steps) | 0.49 | 0.53 | 0.66 | 13 ms, against 143 |

Their first-call forces differed from the all-off run by up to 0.077 and 0.79 kJ/mol/nm on QM atoms
(largest force 4538 and 5836), and by up to 0.002 and 0.024 on MM atoms. For crambin the all-off SCF
converged in 31 iterations, the others in 32.
These two regions had no repeat runs, so how much of these differences is run-to-run variation was not
measured.

A line profile with MOPAC replaced by an instant stub (`colab/qmmm_pyprof_colab.ipynb`, same machine type) found
the remaining Python time to be host work. The GPU was busy 1.75 ms of a 13.6-ms call. About 190 CuPy operations per
step cost ~19 us of host time each, and the four FFTs of the long-range term ~0.28 ms each. Two changes followed:

- At commit `1caa9976`, the work over the MM charges moved into fused CUDA kernels (`scripts/qmmm/qmmm_mm_gpu.py`);
  `QMMM_MM_GPU=arrays` keeps the CuPy array code.
- The long-range FFTs now go through cuFFT plans made once; `QMMM_FFT_PLANS=0` keeps CuPy's `rfftn`/`irfftn`.

Both are compared with the old code at the first call and switched off, with a message, if they differ. In every
run below they agreed:
- the kernels within 1.4e-14 (switched charges, and forces in kJ/mol/nm);
- the potential grids from the plans within 6.7e-16 (absolute) of those from `rfftn`/`irfftn`.

Measured with `colab/qmmm_speed3_colab.ipynb`, same protocol, all runs in one Colab session; two runs of the same
setting differed by up to 0.6 ms per step:

| `tolerance=0.01` | s per step | MOPAC | Python | long-range |
|---|---:|---:|---:|---:|
| both changes | 0.0372 / 0.0374 | 0.0285 | 0.0037 | 0.0019 |
| both off (as at `46637c1f`) | 0.0437 / 0.0431 | 0.0291 | 0.0089 | 0.0026 |
| only the kernels off | 0.0426 | 0.0289 | 0.0091 | 0.0021 |
| only the plans off | 0.0374 | 0.0286 | 0.0037 | 0.0025 |

MOPAC, Python and long-range are medians per call. With a tolerance of 0.001, a step took 0.0509 s against 0.0564 s.
Whole chignolin took 0.074 s per step against 0.080 s (Python 2.2 ms per call against 8.1 ms), and whole crambin
0.490 s against 0.496 s (Python 5.0 ms against 12.5 ms). Each of these was one run.

First call, the same geometry in every run:
- Over the 15 pairs of the six trypsin runs at 0.01, heats differed by up to 7.0e-6 kcal/mol and the largest force
  component by up to 0.038 kJ/mol/nm.
- The two pairs with the same setting differed by 4.6e-6 and 6.0e-6 kcal/mol, and by 0.031 and 0.022 kJ/mol/nm.
- All heats lay within 6.3e-6 of −297.0127265.
- Whole chignolin and crambin, both changes against both off: up to 0.054 and 0.34 kJ/mol/nm on QM atoms, and 0.0035
  and 0.011 on MM atoms (no repeats).

Under `nsys` with the stub, one kernel took 1.84 ms of GPU time per call: the switching force on the QM atoms, one
thread per QM atom over all 14 536 groups. Commit `92e8f2e6` sums it over chunks of 64 groups instead, and the two
kernels now take 0.09 ms. With the stub (`qmmm_pyprof2`, another session), a call then took 3.9 ms against 11.4 ms
with both changes off: Python 1.9 against 8.7 ms, long-range 1.8 against 2.5 ms. The kernels' forces agreed with NumPy
within 2.2e-13 on the three position sets. Per call the stubbed path runs 38 kernels and 12 copies (194 and 33 with
the array code), and the GPU is busy 1.5 ms of 4.2 ms. The full step with MOPAC was not measured at `92e8f2e6`.

At commit `c484ceee`, MOPAC's loops over all QM atom – MM charge pairs run in CUDA kernels
(`src/gpu/qmmm_point_charges.cu`). These are `qmmm_hcore`, which puts the charges into the one-electron matrix,
and `qmmm_grad`, the gradient on both. Before, they ran on the CPU with OpenMP. `MOPAC_QMMM_GPU=0` keeps them there.
The first call compares the two and stays on the CPU if they differ. In every run below they agreed: the hcore terms
within 6.8e-13 eV, and the gradient, potentials and energy within 9.9e-13.

Measured with `colab/qmmm_speed4_colab.ipynb`, same protocol, one Colab session:

| | point charges on the GPU | on the CPU |
|---|---:|---:|
| trypsin–BPTI, `tolerance=0.01` (two runs each) | 0.0343 / 0.0342 s | 0.0381 / 0.0388 s |
| trypsin–BPTI, 0.001 | 0.0484 s | 0.0519 s |
| whole chignolin | 0.0706 s | 0.0752 s |
| whole crambin | 0.451 s | 0.483 s |

With `MOPAC_GPU_PROFILE=1`, `qmmm_grad` took 1.05 ms per step against 2.76 ms, and `qmmm_hcore` 0.30 ms against
1.80 ms.

The first call of the session's first run took 5.6 s, against 0.23 s in the later runs. CuPy compiles its kernels at
first use and caches them, which would account for this; it was not checked.

First call:
- Over the six pairs of the four trypsin runs at 0.01, heats differed by up to 2.9e-6 kcal/mol, and the largest
  force component by up to 0.028 kJ/mol/nm.
- All four heats lay within 7.1e-6 of −297.0127265.
- Whole chignolin, GPU against CPU: forces up to 0.99 kJ/mol/nm on QM atoms and 0.059 on MM atoms. Its SCF converged
  in 30 iterations against 29 (no repeats).
- Whole crambin: 0.48 and 0.020 kJ/mol/nm; 31 iterations both.

Inside MOPAC (trypsin–BPTI, `nsys` of 38 steps with `MOPAC_NVTX=1`, which makes every section timer an NVTX
range): a MOPAC call took 26.6 ms of wall time, with the GPU busy for 21.1 ms. Of the 5.5 ms of idle GPU, 4.0 ms
fell inside the resident SCF loop. There MOPAC launches ~1590 kernels per step, many of them shorter than the
~5 us a launch costs the host.

The largest kernels per step:

| kernel | time per step | launches per step |
|---|---:|---:|
| staged `diagg2` | 9.7 ms | 15 |
| `diagg1_virtual` | 4.4 ms | 30 |
| pair gradient | 1.0 ms | 1 |
| the point-charge kernels, together | 1.0 ms | |

The `diagg2` trace (`MOPAC_MOZYME_SCF_TRACE=1`, 20 steps) counted per sweep:
- ~2400 rotations, 61 thread blocks, and a longest chain of 58 rotations on one LMO;
- with the block kernel, a sweep spanned 790 us, 13.8 us per rotation of that chain, and each block waited 226 us
  for locks.

Two later changes made the point-charge kernels faster on the GPU:
- At commit `48bc1100`, their chunks are sized for ~32 000 threads. Before, there was one thread per QM atom over
  64 charges, ~1900 threads in all. In `colab/qmmm_speed5_colab.ipynb` the main gradient kernel then took 0.09 ms
  per step against 0.53 ms, and the `qmmm_grad` section 0.39 ms against 1.05 ms.
- At commit `da705b9d`, the sums over the chunks run one thread per QM atom and value. They had taken 0.32 ms.

Gaps inside the resident SCF loop, from `nsys` with NVTX marks at the SCF's stage boundaries (`MOPAC_NVTX=1`, the
gaps charged by `analyze_prof.scf_gaps` in the notebooks):
- The GPU was idle 4.0 ms per step in all.
- 1800 of the gaps were shorter than 5 us and added up to 2.5 ms. These are the boundaries between ~2100 GPU
  operations per step: kernels, copies and memsets.
- Only 3 % of the operations were issued by the host after the GPU had gone idle.
- 1.2 ms of the idle time fell while the host was inside `cudaEventSynchronize` after the convergence stage, and
  0.7 ms inside `cudaMemcpy`.

`QMMM_CUDA_BLOCKING_SYNC=0` or `1` sets OpenMM's CUDA property `UseBlockingSync` (`colab/qmmm_speed6_colab.ipynb`).
It changed neither the step nor those idle times:

| synchronization | s per step |
|---|---:|
| OpenMM's default | 0.0322 / 0.0320 |
| spinning (`0`) | 0.0323 / 0.0321 |
| blocking (`1`) | 0.0321 |

Whole chignolin and crambin took 0.068 and 0.445 s with the default, and 0.068 and 0.447 s spinning. The first-call
heats of the five trypsin runs lay within 6.1e-6 kcal/mol of −297.0127265.

`diagg2`, the rotations between occupied and virtual LMOs, was the largest kernel, at 9.7 ms per step. Two
attempts followed:

- **One warp per virtual LMO** (`colab/qmmm_speed7_colab.ipynb`, commit `cdc2f84d`, reverted). The rotation was the
  one of the cooperative kernel; the locks were those of the block kernel. It was slower: 31 us per rotation on the
  critical chain against ~14 us, `diagg2` 24 against 10 ms per step, and the trypsin step 46.7 against 32.4 ms.
  Its first-call heats and its NVE drift were as with the default kernel.
- **Phase timers** in the trace (`colab/qmmm_speed8_colab.ipynb`, commit `d03f8d72`). In trypsin–BPTI the LMOs grew
  during the first SCF from ~15 to ~50 of the 59 QM atoms, and stayed at ~53 atoms on average (occupied and
  virtual alike). The pair list held ~3400 of the 76 × 61 = 4636 occupied–virtual pairs.
  - Per sweep, each block (one virtual) did ~39 rotations. Inside the staged kernel they cost 80 us in staging the
    occupied LMO, 81 us in the growth tests and 72 us in applying and writing back, ~6 us per rotation in all.
  - Each block also waited 207 us for the occupied LMOs' locks.
  - The pairs of a virtual are stored in ascending occupied index (`diagg1`), so every block began its chain on the
    same occupied LMOs.

At commit `2cb2f2be`, each virtual in a QM/MM session starts its chain at its own offset
(`MOPAC_MOZYME_DIAGG2_STAGGER=1`, set unless already defined). With it:
- the lock waits per block fell to 100 us, and the critical chain from 13.6 to 10.7 us per rotation;
- `diagg2` took 8.1 instead of 10.2 ms per step (stage profile).

| | staggered | list order |
|---|---:|---:|
| trypsin–BPTI, `tolerance=0.01` (two runs each) | 0.0313 / 0.0311 s | 0.0335 / 0.0334 s |
| trypsin–BPTI, 0.001 | 0.0434 s | 0.0471 s |
| whole chignolin | 0.0620 s | 0.0700 s |
| whole crambin | 0.442 s | 0.457 s |

The order of the rotations changes the SCF's path:
- SCF iterations per step: 13.42 against 13.29, and 21.76 against 21.48 at 0.001.
- First-call heats: within 3.5e-6 kcal/mol.
- First-call forces: up to 0.054 kJ/mol/nm apart, against 0.025 between two runs in list order.
- NVE, 1 ps at 0.001 and 0.5 fs from the same state: the QM/MM force drifted by −0.91 ± 0.11 kJ/mol/ps staggered,
  against −0.94 ± 0.11 in list order.

At commit `bba4e09c`, `diagg2` runs on dense LMO rows in QM/MM sessions (`MOPAC_MOZYME_DIAGG2_DENSE=1`, set unless
already defined; commit `c8fd1164`). Each sweep goes through four kernels:
- The LMOs are copied into rows with one column per basis function.
- The lock sweep rotates whole rows, with the same angles, locks and staggered order. This drops the atom-list
  bookkeeping of the sparse rotation.
- An atom missing from an LMO is added where the squared norm of its coefficients exceeds `thresh`. This is the
  test the sparse rotation applies to the coefficients a rotation would add, but here it is applied at the end of the
  sweep instead of rotation by rotation.
- The rows are written back.

If an LMO's storage cannot take its new atoms, the dense result is dropped and the lock kernel does the sweep.
Limits: at most 4096 atoms, and 6144 basis functions (one row in 48 KB of shared memory).

Measured with `colab/qmmm_speed9_colab.ipynb`, one Colab session:
- Every dense sweep of trypsin–BPTI was written back, with a median of 12 new atoms per sweep.
- `diagg2` took 3.2 instead of 7.9 ms per step (stage profile).

| | dense | lock kernel |
|---|---:|---:|
| trypsin–BPTI, `tolerance=0.01` (two runs each) | 0.0256 / 0.0258 s | 0.0305 / 0.0305 s |
| trypsin–BPTI, 0.001 | 0.0349 s | 0.0430 s |
| whole chignolin | 0.0464 s | 0.0628 s |
| whole crambin | 0.354 s | 0.448 s |

Checks against the lock kernel:
- SCF iterations per step: 13.43 against 13.42 at 0.01, and 21.75 against 21.76 at 0.001.
- First call, over the six pairs of the four trypsin runs at 0.01: heats within 1.5e-6 kcal/mol, forces within
  0.032 kJ/mol/nm. Two runs with the lock kernel were 0.030 apart.
- NVE, 1 ps at 0.001 and 0.5 fs from the same state: the QM/MM force drifted by −0.90 ± 0.12 kJ/mol/ps, against
  −0.91 ± 0.11.
- Whole chignolin and crambin, dense against the lock kernel at the first call: forces up to 0.39 and 0.53 kJ/mol/nm
  apart (largest force 4538 and 5836), with no repeat runs.

At commit `e6fddf48`, `diagg1` also runs on dense matrices in QM/MM sessions (`MOPAC_MOZYME_DIAGG1_DENSE=1`, set
unless already defined; commit `977d7e25`). `diagg1` computes the Fock elements between occupied and virtual LMOs and
lists the pairs `diagg2` rotates.
- The Fock matrix is built dense from its atom-pair blocks.
- The elements come from two DGEMMs (cuBLAS), F_vo = (C_vir F) C_occ^T, on the dense LMO rows of `diagg2`.
- The pair list follows the sparse kernel's rules: the prescreen on the first two atoms of each LMO, |F_ij| >
  `oldlim`, ascending occupied order, and odd iterations recomputing only the listed pairs. So do `sumt`, `tiny` and
  the virtual energies.
- Only the density screen of the sparse sums (terms below `cutoff`, at most 1e-8) is gone.
- If the dense tables cannot be built, the sparse passes run.

Measured with `colab/qmmm_speed10_colab.ipynb`, one Colab session; `diagg1` took ~1.5 instead of 5.1 ms per step:

| | dense `diagg1` | sparse `diagg1` |
|---|---:|---:|
| trypsin–BPTI, `tolerance=0.01` (two runs each) | 0.0215 / 0.0211 s | 0.0259 / 0.0259 s |
| trypsin–BPTI, 0.001 | 0.0269 s | 0.0343 s |
| whole chignolin | 0.0312 s | 0.0445 s |
| whole crambin | 0.245 s | 0.352 s |

Checks against the sparse `diagg1`:
- SCF iterations per step: 13.4 at 0.01 and 21.7 at 0.001 with both. Whole crambin took 14.5 against 15.9.
- First-call heats of the four trypsin runs at 0.01: within 3e-6 kcal/mol.
- First-call forces at 0.01: the dense runs were 0.16–0.18 kJ/mol/nm from the sparse ones. The two sparse runs, whose
  first SCF took 38 and 39 iterations, were 0.17 apart, and the two dense runs 0.016.
- At 0.001 the dense and sparse first-call forces were 0.015 apart, and the heats 1e-7 kcal/mol.
- NVE, 1 ps at 0.001 and 0.5 fs from the same state: the QM/MM force drifted by −0.92 ± 0.11 kJ/mol/ps, against −0.90
  ± 0.11.

## Example: trypsin with BPTI

`scripts/kaggle/mopac_qmmm_enzyme` (PDB 2PTC, 46 637 atoms with TIP3P water):

```
python scripts/qmmm/run_qmmm_md.py 2PTC.pdb --libmopac build/libmopac.so \
    --qm-residues E:57:sc,E:102:sc,E:195:sc,I:15,I:16 --variants E:57=HID --padding 1.0 --ron 10 --roff 12 \
    --distances E:195:OG-I:15:C,E:57:NE2-E:195:OG,E:57:HD1-E:102:OD2 --cutoff-test 8,12,16,20,24 \
    --steps 10000 --nve 1000 --mm-equil-ps 100 --platform CUDA
```

QM region: side chains of the catalytic triad (His57 as HID, Asp102, Ser195) and Lys15-Ala16 of BPTI
(54 atoms + 5 H link atoms, charge 0). Kaggle T4, MOPAC on the CPU, 5 ps of NVT:

| | cut-off only (roff 12 A) | QM-PME, grid in NumPy | QM-PME, grid on the GPU |
|---|---:|---:|---:|
| time per step | 0.21 s (206 ps/day) | 0.31 s (140 ps/day) | 0.24 s (181 ps/day) |
| His57 HD1 - Asp102 OD2 (mean) | 1.95 A | 1.76 A | 1.80 A |
| His57 NE2 - Ser195 OG | 2.85 A | 3.09 A | 3.21 A |
| Ser195 OG - Lys15 C (attacked carbonyl) | 4.0 A | 2.5 A | 2.56 A |

(one 5 ps trajectory each, from different MM equilibrations.) Three independent replicas of each (Mac CPU,
20 ps MM equilibration, 5 ps NVT) give similar geometries with and without QM-PME, mean ± standard deviation
of the replica averages: Ser195 OG - Lys15 C 2.42 ± 0.05 / 2.55 ± 0.08 A, His57 NE2 - Ser195 OG 3.03 ± 0.06 /
3.15 ± 0.06 A, His57 HD1 - Asp102 OD2 1.76 ± 0.06 / 1.83 ± 0.03 A (cut-off only / QM-PME); the 4.0 A of the
single cut-off run above was not reproduced. The NumPy PME costs ~0.1 s per step on the Kaggle CPU for 46 000 atoms, the CuPy one ~0.03 s. NVE (Mac, 3000 steps =
1.5 ps): linear-fit slope -1.6 kJ/mol/ps with QM-PME, -7.2 without, fluctuations ~5 kJ/mol rms (the Kaggle
check, +20 kJ/mol over 0.5 ps with PME, is within them).

Cut-off test (8 snapshots of the PME run; forces on the QM atoms against PME with roff = 24 A, rms force
796 kJ/(mol nm); "spread" = how much the QM/MM energy differences between snapshots change):

| roff | cut-off only: rms force error | spread | QM-PME: rms force error | spread |
|---:|---:|---:|---:|---:|
| 8 A | 33.6 | 29.5 kJ/mol | 8.2 | 2.8 kJ/mol |
| 12 A | 28.0 | 44.5 kJ/mol | 6.4 | 1.7 kJ/mol |
| 16 A | 24.6 | 23.9 kJ/mol | 5.8 | 1.9 kJ/mol |
| 20 A | 19.9 | 23.9 kJ/mol | 4.0 | 1.9 kJ/mol |
| 24 A | 19.2 | 26.0 kJ/mol | (reference) | |

A second run (GPU grid, other MM equilibration) gave the same picture: cut-off only 45/27/30/22/20, QM-PME
10.0/7.0/6.7/3.4 kJ/(mol nm) at 8/12/16/20 A, spreads 14-43 against 1.1-2.6 kJ/mol.

Without the long-range part, the energy differences depend on the cut-off by 20-45 kJ/mol even at 24 A
(the charged residues of a protein converge slowly); with QM-PME the result is nearly independent of roff,
and roff = 12 A is enough.

## Reaction scans (`qmmm_scan.py`)

`scripts/qmmm/qmmm_scan.py` follows a reaction coordinate by adiabatic mapping: it prepares the system as
`run_qmmm_md.py` does, minimizes the reactant with QM/MM, then moves one distance (`--scan A-B`) to `--to` in steps of
`--step` with a harmonic restraint (`--k`, default 2000 kJ/mol/A^2), minimizing at each value from the structure of the
value before, and with `--back` returns the same way. A coordinate can also be a difference of two distances,
`A-B-C` = d(A,B) − d(B,C), for a proton B moving from A to C; `--scan2` (with `--from2`, `--to2`, `--step2`) adds a
second coordinate and scans the two on a grid, row by row in alternating directions, and `--back` then returns along
the last row and the first column. Only residues with an atom within `--free-radius` (10 A) of the QM region move
freely; the rest are held (`--hold`, 50 kJ/mol/A^2) at their positions after an MM minimization of the equilibrated
system, so that each window relaxes the active site and not the whole water box (minimizing all of trypsin-BPTI in
water lowered the energy by thousands of kJ/mol per window, from the relaxation of the box). Held at the equilibrated
snapshot itself, as in the first version of the script, the held atoms kept moving toward it for several windows:
in a trypsin-BPTI map their restraint energy fell by 58 kJ/mol, most of it over the first five windows, and two
neighbouring points of the map whose active sites differed by less than 0.3 A differed by 50 kJ/mol in the energy
without the restraints. `scan.csv` gives,
per window, the targets and the values reached, the energy without the restraints, the QM heat and the distances
named in `--watch`; `scan_frames.pdb` the residues within 5 A of the QM region. The energies belong to single
minimized structures: a potential-energy map along one path, not a free energy; a difference between the forward and
the backward scan shows where the path depends on its history. Example: the wiki, QM/MM example 59 (Ser195 of trypsin
attacking the carbonyl carbon of Lys15 of BPTI while His57 takes its proton).

## Validation

- QM water + TIP3P water (point charges): all gradient components on the QM atoms and the MM charges agree
  with finite differences to 0.01 kcal/(mol Angstrom).
- Chignolin in TIP3P (QM = whole peptide, 138 atoms; QM = Tyr side chain with a link atom): OpenMM forces
  agree with finite differences of the total energy, including atoms in the switching shell and the
  boundary atoms CA/CB.
- Along an MD trajectory the gradient of each step agrees with that of a fresh calculation at the same
  geometry (largest difference 0.035 kcal/(mol Angstrom)) and with finite differences of the energy.

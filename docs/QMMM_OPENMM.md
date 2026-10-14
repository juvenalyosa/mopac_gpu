# QM/MM molecular dynamics with OpenMM and MOPAC/MOZYME

The QM region is computed by MOPAC (PM7, MOZYME, on the GPU when there is one) in the field of the MM
point charges (electrostatic embedding); the force field, water, PME, constraints, integrators,
thermostats, barostats and reporters are OpenMM's. Any OpenMM integrator or ensemble works, because the
coupling is an OpenMM `PythonForce`.

## Requirements

- OpenMM ≥ 8.4 (tested with 8.6.1; `conda install -c conda-forge openmm` or `pip install 'openmm[cuda12]'`),
  NumPy and SciPy
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
- The H-bond correction of PM7 is used with `HB-SMOOTH` (its original form has kinks that MD and FORCE
  cannot cross).
- QM-PME needs an orthorhombic box. Its parameters are fields of `qmmm.region` (`pme_alpha` = 0.25 1/A,
  `pme_cutoff` = 13 A real space, `pme_spacing` = 1.2 A, `pme_order` = 4): potential errors ~1e-4 of the
  potential, and the forces are exact derivatives of the energy as computed. It runs in NumPy on the CPU:
  ~45 ms per step for 46 000 MM atoms (Apple M-series). With PME the QM energy includes the far field and a
  constant (net-charge background), so absolute energies of runs with and without PME differ; only
  differences and forces compare.
- One QM region per process; do not run two MOPAC sessions in the same directory.
- Energy conservation (NVE, chignolin in water, 0.5 fs, 100 fs): the total energy fluctuates by a few kJ/mol
  as with pure MM and does not drift (-0.5 kJ/mol per 100 fs at `tolerance=0.01`, 0.0 at 0.001). Every
  step converges the SCF to `tolerance`; before 2026-10-14 the SCF of a step could stop early (a rapid exit
  meant for geometry optimization, when the heat is far from the lowest heat so far), the forces were not
  the derivatives of the energy and the total energy fell by ~30 kJ/mol per 100 fs.

## Speed

OpenMM CUDA + MOPAC, Kaggle T4, 0.5 fs steps, TIP3P water box (`scripts/kaggle/mopac_qmmm`):

| QM region | atoms (QM / all) | MOPAC on the GPU | MOPAC on the CPU (1 core) | NVE drift (GPU run) |
|---|---:|---:|---:|---:|
| Tyr side chain of chignolin (link atom) | 16 / 3 300 | - | **0.017 s/step** (2.6 ns/day) | +2.5 kJ/mol in 200 fs |
| whole chignolin | 138 / 8 100 | **0.19 s/step** (226 ps/day) | 1.66 s/step | -2.2 kJ/mol in 200 fs |
| whole crambin | 642 / 13 000 | **2.47 s/step** (17.5 ps/day) | - | +0.4 kJ/mol in 100 fs |

(`tolerance=0.01`, every step's SCF converged; the drifts are within the per-step fluctuation of the MM
water.)

QM regions of up to 100 atoms run on the CPU by default (as MOPAC does for small systems); larger ones on
the GPU. A T4 is weak in double precision; an A100 should be several times faster (README, "Which GPU?").

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

| | cut-off only (roff 12 A) | with QM-PME |
|---|---:|---:|
| time per step | 0.21 s (206 ps/day) | 0.31 s (140 ps/day) |
| His57 HD1 - Asp102 OD2 (mean) | 1.95 A | 1.76 A |
| His57 NE2 - Ser195 OG | 2.85 A | 3.09 A |
| Ser195 OG - Lys15 C (attacked carbonyl) | 4.0 A | 2.5 A |

(one 5 ps trajectory each, from different MM equilibrations: the distances are indications, not converged
averages.) The NumPy PME costs ~0.1 s per step on the Kaggle CPU for 46 000 atoms. NVE (Mac, 3000 steps =
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

Without the long-range part, the energy differences depend on the cut-off by 20-45 kJ/mol even at 24 A
(the charged residues of a protein converge slowly); with QM-PME the result is nearly independent of roff,
and roff = 12 A is enough.

## Validation

- QM water + TIP3P water (point charges): all gradient components on the QM atoms and the MM charges agree
  with finite differences to 0.01 kcal/(mol Angstrom).
- Chignolin in TIP3P (QM = whole peptide, 138 atoms; QM = Tyr side chain with a link atom): OpenMM forces
  agree with finite differences of the total energy, including atoms in the switching shell and the
  boundary atoms CA/CB.
- Along an MD trajectory the gradient of each step agrees with that of a fresh calculation at the same
  geometry (largest difference 0.035 kcal/(mol Angstrom)) and with finite differences of the energy.

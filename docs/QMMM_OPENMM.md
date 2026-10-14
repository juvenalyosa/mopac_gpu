# QM/MM molecular dynamics with OpenMM and MOPAC/MOZYME

The QM region is computed by MOPAC (PM7, MOZYME, on the GPU when there is one) in the field of the MM
point charges (electrostatic embedding); the force field, water, PME, constraints, integrators,
thermostats, barostats and reporters are OpenMM's. Any OpenMM integrator or ensemble works, because the
coupling is an OpenMM `PythonForce`.

## Requirements

- OpenMM ≥ 8.4 (tested with 8.6.1; `conda install -c conda-forge openmm` or `pip install 'openmm[cuda12]'`)
- this MOPAC built as a shared library (`libmopac.so` / `libmopac.dylib`, built with the `mopac` target;
  `-DGPU=ON` for the GPU)
- `scripts/qmmm/openmm_mopac.py` and `scripts/qmmm/mopac_qmmm_api.py` on the Python path

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

## Details and limits

- Units: OpenMM nm and kJ/mol, MOPAC Angstroms and kcal/mol. The QM energy is the PM7 heat of formation
  of the QM region in the field of the MM charges, so the potential energy is offset by a constant.
- The periodic box must hold the QM region plus `roff` on every side (checked; an error says so).
- `tolerance` is MOPAC's relative SCF criterion; below 0.1 MOPAC also uses `THRESH=1.D-15`. An SCF that
  does not converge stops the run with an error.
- The H-bond correction of PM7 is used with `HB-SMOOTH` (its original form has kinks that MD and FORCE
  cannot cross).
- Long-range QM-MM electrostatics: cut-off with switching (no QM-PME yet).
- One QM region per process; do not run two MOPAC sessions in the same directory.
- Energy conservation (NVE, chignolin in water): no jumps; a slow drift (~ -10 kJ/mol per 100 fs for a
  138-atom QM region) comes from MOZYME's own gradient, which is not exactly the derivative of its energy
  (also without MM charges). Use a thermostat (Langevin NVT) for production.

## Validation

- QM water + TIP3P water (point charges): all gradient components on the QM atoms and the MM charges agree
  with finite differences to 0.01 kcal/(mol Angstrom).
- Chignolin in TIP3P (QM = whole peptide, 138 atoms; QM = Tyr side chain with a link atom): OpenMM forces
  agree with finite differences of the total energy, including atoms in the switching shell and the
  boundary atoms CA/CB; NVE at 0.25 fs: drift and per-step noise as for pure MM over 10-50 fs.

Examples for running MOPAC with and without GPU

CPU baseline
- File: `h2o_cpu.mop`
- Run: `./build-gpu/mopac examples/h2o_cpu.mop`

Force GPU on small system
- File: `h2o_gpu_force.mop`
- Run: `MOPAC_FORCEGPU=1 ./build-gpu/mopac examples/h2o_gpu_force.mop`

MOZYME on 1 GPU
- File: `mozyme_1gpu.mop`
- Run: `./build-gpu/mopac examples/mozyme_1gpu.mop`

MOZYME on 2 GPUs (explicit pair 1,2)
- File: `mozyme_2gpu_pair.mop`
- Run: `CUDA_VISIBLE_DEVICES=0,1 ./build-gpu/mopac examples/mozyme_2gpu_pair.mop`

Small peptide (Gly–Gly) from PDB with hydrogens added
- File: `peptide_gg.mop` (uses `peptide_gg.pdb`)
- Run (1 GPU auto): `./build-gpu/mopac examples/peptide_gg.mop`
- Run (2 GPUs 0,1): `CUDA_VISIBLE_DEVICES=0,1 ./build-gpu/mopac examples/peptide_gg_2gpu.mop`

Tri‑alanine (Ala–Ala–Ala) from PDB with hydrogens added
- File: `peptide_aaa.mop` (uses `peptide_aaa.pdb`)
- Run (1 GPU auto): `./build-gpu/mopac examples/peptide_aaa.mop`
- Run (2 GPUs 0,1): `CUDA_VISIBLE_DEVICES=0,1 ./build-gpu/mopac examples/peptide_aaa_2gpu.mop`

MOZYME peptide benchmark helper
- Vary MOZYME_MINBLK and summarize timing and heat of formation:
  - `python3 scripts/peptide_bench.py ./build-gpu/mopac examples/peptide_gg.mop`
  - Two GPUs pair 0,1: `python3 scripts/peptide_bench.py ./build-gpu/mopac examples/peptide_gg.mop --two-gpu --pair 1,2 --devices 0,1`
  - Custom minblk list: `--minblk 8,16,24,32,48,64`
  - Disable streams for debugging: `--streams off`

Notes
- These examples use Cartesian (XYZ) coordinates and perform single‑point calculations (1SCF).
- MOZYME can be used on small systems but typically shows the most benefit on larger molecules.

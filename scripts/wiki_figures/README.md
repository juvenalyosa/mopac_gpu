# Figures of the wiki examples

One figure per worked example of the [wiki](https://github.com/juvenalyosa/mopac_gpu/wiki), in a light and a dark
version (`images/exNN.png`, `images/exNN-dark.png` in the wiki repository).

```bash
# 1. the calculations (CPU build of this repository; OpenMM for QM/MM): ~290 jobs, a few hours on 8 cores
python jobs.py build-cpu/mopac build-cpu/libmopac.dylib work --workers 8
# 2. the figures (needs PyMOL, matplotlib, NumPy; in <ext>: 2PTC.pdb from the PDB, and for example 59 the maps of
#    colab/enzyme_scan_colab.ipynb, enzyme_scan_l4.zip unzipped: enzyme_scan_l4/seed_{1,2,3}/)
python make_figures.py work path/to/mopac_gpu.wiki --ext ext
# 3. put them at the end of each example section of the wiki pages
python embed_wiki.py path/to/mopac_gpu.wiki
```

- `jobs.py`: every calculation behind a figure; each job runs in `work/<name>/` and is skipped once `done.json`
  exists. MOPAC runs with one thread unless the job says otherwise.
- `figures.py`: figures drawn from numbers published in `README.md` and `docs/QMMM_OPENMM.md` (examples 2, 39, 55, 56,
  60, 64, 65) and the diagram of example 57; `figures_runs.py`: the figures that read the calculations (example 59:
  the QM/MM maps that the Colab notebook ran on an L4 GPU, commit `e799de85`).
- `figlib.py`: the two themes (the validated reference palette of the data-visualization method, light and dark
  steps), readers of MOPAC output and PyMOL renders.
- `embed_wiki.py`: the captions; it replaces the figure blocks it wrote before, so it can be run again.

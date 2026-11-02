# Paper benchmarks

`paper_bench.py` runs every benchmark of the paper (speed and accuracy of each GPU path against the CPU, same
binary with `MOPAC_NOGPU=1`) and `paper_figures.py` turns the results into figures and tables. The Colab
notebook [`colab/paper_benchmarks_colab.ipynb`](https://colab.research.google.com/github/juvenalyosa/mopac_gpu/blob/main/colab/paper_benchmarks_colab.ipynb)
builds MOPAC, runs them on an A100 and keeps the results on Google Drive, so an interrupted session resumes.

```bash
python3 scripts/paper/paper_bench.py --mopac build/mopac --libmopac build/libmopac.so --out results \
    --exp all            # or a comma list: mozyme_sp,methods,solids,...
python3 scripts/paper/paper_figures.py results
```

`--quick` runs one small case per experiment (a test of the pipeline, ~15 min on a CPU); `--no-cpu` only the
GPU runs. The list of experiments and their options is in `paper_bench.py --help`.

## What is measured

| Experiment | Runs | Compared |
|---|---|---|
| `mozyme_sp` | MOZYME 1SCF, 8 proteins / DNA (642–6689 atoms), water clusters 125–7052 H2O | time (1 CPU core vs GPU), heat, GPU memory |
| `mozyme_grad` | 1SCF GRADIENTS, 4 systems | heat, every gradient component |
| `methods` | MNDO, AM1, PM3, RM1, PM6, PM6-D3H4, PM7 on crambin and DNA | heat |
| `cosmo` | EPS=78.4, 3 systems | time, heat |
| `opt` | crambin 100 L-BFGS cycles (CPU, GPU), 1AKE 50 cycles (GPU) | time per cycle, energy path, heat at the same final geometry |
| `md` | crambin NVE (CPU, GPU), crambin NVT and 1AKE NVE (GPU) | time per step, energy conservation, temperature |
| `thermo` | chignolin FORCE (parallel GPU processes), crambin FORCETS | frequencies, ZPE against stored CPU references |
| `solids` | Si supercells 64–1728 atoms and 10 cells, PM7 1SCF GRADIENTS | time (all CPU cores vs GPU), heat, gradient |
| `solids_profile` | 512–768-atom cells with `MOPAC_GPU_PROFILE=1` | time per stage |
| `solids_force`, `solids_drc` | diamond FORCE and DRC | frequencies, energy conservation |
| `conv_mol` | crambin RHF / UHF / COSMO, graphene flake, chignolin C.I. | time, heat, gradient |
| `qmmm` | OpenMM + MOPAC, QM regions of 16–642 (1231) atoms in water | time per step, NVE drift |
| `qmmm_enzyme` | trypsin–BPTI, cut-off vs QM-PME | time per step, force error vs cut-off |

## Layout of a results directory

- `runs/<experiment>/<case>/<mode>[_r<k>]/`: the MOPAC input and output of every run and its `result.json`
  (a run with a `result.json` is not repeated; delete the directory to redo it)
- `data/*.csv`: one row per run, plus detail tables (optimization cycles, dynamics rows, frequencies, profiles,
  gradient comparisons)
- `figures/*.pdf|png`, `tables/*.tex|csv`, `summary.md`, `env.json` (GPU, CPU, versions, commit)

GPU times are the median of `--repeats` runs (error bars: fastest and slowest); CPU times are one run. MOZYME
has no threading, so its CPU reference is one core; the conventional SCF uses all cores (OpenMP and the
threaded OpenBLAS the notebook builds with).

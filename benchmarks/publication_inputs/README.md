# Publication Benchmark Inputs

Prepared medium/large inputs for end-to-end MOPAC CPU/GPU publication benchmarks.
PDB structures are downloaded from the RCSB Protein Data Bank and cleaned to keep
coordinate records relevant for MOPAC single-point benchmarking.

| Label | Category | Heavy/input atoms | MOPAC input |
|---|---|---:|---|
| protein_crambin_1crn | protein | 327 | `protein_crambin_1crn.mop` |
| protein_ubiquitin_1ubq | protein | 602 | `protein_ubiquitin_1ubq.mop` |
| protein_adenylate_kinase_1ake | protein | 3426 | `protein_adenylate_kinase_1ake.mop` |
| dna_dodecamer_1bna | dna | 486 | `dna_dodecamer_1bna.mop` |
| protein_1g6x | protein | 496 | `protein_1g6x.mop` |
| protein_antifreeze_1ezg | protein | 1106 | `protein_antifreeze_1ezg.mop` |
| protein_barnase_1rnb | protein | 912 | `protein_barnase_1rnb.mop` |
| protein_bacteriorhodopsin_1c3w | protein | 2050 | `protein_bacteriorhodopsin_1c3w.mop` |
| rna_trna_1ehz | rna | 1661 | `rna_trna_1ehz.mop` |
| material_graphene_nanoflake_c192h38 | material_cluster | 230 | `material_graphene_nanoflake_c192h38.mop` |

The DNA/RNA/protein inputs are intended as reproducible performance and CPU/GPU
agreement benchmarks, not as curated thermochemical reference geometries.
Before timing these cases, run `scripts/hydrogenate_publication_benchmark_inputs.py`
with the built MOPAC executable; it creates hydrogenated geometries and rewrites
the compute decks to reference them via `GEO_DAT`.
The graphene nanoflake is generated locally as a material-like cluster benchmark.

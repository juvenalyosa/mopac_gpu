"""Put the figure of each example at the end of its section in the wiki pages (idempotent).

    python embed_wiki.py <wiki dir>

The images are referenced through raw.githubusercontent.com/wiki/..., with a dark variant for the dark theme.
"""
import re, sys
from pathlib import Path

RAW = "https://raw.githubusercontent.com/wiki/juvenalyosa/mopac_gpu/images"
CPU = "CPU build of v1.2.1-gpu, laptop (Intel Core i7-8850H)"
README = "numbers published in the [README](https://github.com/juvenalyosa/mopac_gpu#readme) (A100 benchmark, commit `c85eb96d`)"
DOCS = "numbers published in [`docs/QMMM_OPENMM.md`](https://github.com/juvenalyosa/mopac_gpu/blob/main/docs/QMMM_OPENMM.md)"

CAP = {
    1: "Atomic charges of chignolin from this job, on the structure and summed per residue.",
    2: f"Time of MOZYME single points on the CPU code path (one core) and on an A100; {README}.",
    3: "The gradient of every atom from this job: its size on the structure and its distribution by element.",
    4: "Heat of formation against the SCF criterion, relative to `SCFCRT=1.D-6` (heats are printed with five decimals), and the job time; all with `THRESH=1.D-15`.",
    5: "Crambin coloured by the charge of each residue (sum of its atomic charges), and the same values along the chain.",
    6: "The DNA dodecamer coloured by the charge of each nucleotide, and the charges along both strands. The +1 is the charge MOZYME derived from the Lewis structure of this hydrogenated file.",
    7: "Heat of formation of chignolin with each method, MOZYME 1SCF at the geometry of the file.",
    8: "Job time of the same single points; AM1 highlighted.",
    9: "The 50-molecule cluster written by the script, and the job time against the number of molecules.",
    10: "Crambin before and after `ADD-H`.",
    11: "The sites that `SITE=(IONIZE)` changed in crambin, as listed in the output (*Changes in Ionization caused by keyword SITE*).",
    12: "The sulfate groups of 1G6X (grey) and the two incomplete ones that the input chemistry check reports (red).",
    13: "SCF of 1G6X (MOZYME, PM7) with the two half sulfates, run with `LET`, and with them removed: energy change per iteration. These are CPU runs; the README's 112 and 29 iterations are from GPU runs.",
    14: "`scripts/mozyme_md_workflow.py` on chignolin with hydrogens added to its heavy atoms: 30 optimization cycles, then 60 NVT and 60 NVE points (shorter than the command above). In NVE the total energy stays within 0.012 kcal/mol while potential and kinetic energy exchange 36 kcal/mol (largest integration `ERROR` 0.010).",
    15: "Chignolin in its COSMO cavity (the surface is PyMOL's) and the heats in vacuum and in water.",
    16: "Heat while atom 50 moves in 0.001-Å steps, and the tangent given by the analytic gradient at 0. With the standard surface the analytic gradient (+0.358 kcal/(mol Å)) differs from the slope of the heats (+0.232); with `SWIG` they agree (−0.322 and −0.325). For this atom the heats are smooth with both surfaces at this resolution.",
    17: "Heat and gradient norm over 40 cycles (the input asks for 100).",
    18: "Heat and gradient norm over 40 cycles (the input asks for 300).",
    19: "The DNA dodecamer (solvent-accessible surface drawn by PyMOL) and the change of the heat from vacuum to water.",
    20: "`TRUST` and the default optimizer, 150 cycles each, from chignolin with hydrogens added again to its heavy atoms (gradient norm 246 at the start).",
    21: "Gradient norm of the `TRUST` run of example 20, with the `GNORM` values 5 and 1.",
    22: "Water dimer: the hydrogen-bond correction and the heat against the O···O distance. The PM7 correction drops to zero at 7.0 Å; with `HB-SMOOTH` it is switched off over the last ångström.",
    23: "An optimization in two jobs: `TRUST CYCLES=25`, then `RESTART` with 25 more cycles (start as in example 20).",
    24: "Start (grey) and `final.pdb` of the runs of example 23, and the rms displacement of each residue.",
    25: "The atoms that `OPT(\"A46\"=4)` lets move (orange carbons) and the heat over the cycles; `TRUST` converged after 10 cycles.",
    26: "NVE from 300 K, 60 fs (the input asks for 100): potential, kinetic and total energy on one scale, and the total alone, magnified: it stays within 0.23 kcal/mol (single-step jumps) while potential and kinetic energy exchange 75 kcal/mol.",
    27: "NVT with `BUSSI=100`, 60 fs: temperature, and `ENERGY TO BATH` (negative: put into the system).",
    28: "Temperature with `BUSSI=10`, `BUSSI=100` and without thermostat, from the same optimized structure (60 fs each).",
    29: "Two runs with `SEED=42`, identical on the CPU (the curves coincide), and one with `SEED=43`; 30 fs.",
    30: "Three seeds, 30 fs: temperature and potential energy.",
    31: "In water at 310 K (`BUSSI=100`), 30 fs: temperature and the integration error.",
    32: "Three heating stages of 20 fs, each started from the last frame of the one before.",
    33: "`KINETIC=20`: the atoms start at rest; the 20 kcal/mol are added once the kinetic energy has reached 0.2 kcal/mol.",
    34: "Crambin, NVE at 300 K, the first 2.5 fs (the input asks for 100): total and kinetic energy.",
    35: "Vibrational spectrum of chignolin from `FORCE THERMO(298)` (transition dipoles), 71 minutes on one core. The one imaginary mode shows that this geometry is not an exact minimum.",
    36: "Frequencies from the parallel script (4 workers) against one MOPAC job, and the wall times. The largest differences are in the lowest modes.",
    37: "The atoms selected by `OPT(\"A46\"=4)`, whose Hessian `FORCETS` computes (FORCETS itself was not run for this figure).",
    38: "The 64-atom silicon cell and the density of states from its orbital energies.",
    39: f"Solids, conventional SCF on 12 CPU threads and on an A100; {README}.",
    40: "Graphene with its periodic images (the cell darker), and the energy change per SCF iteration: 1 059 iterations on the CPU.",
    41: "`SMEAR=0.1`: occupations of the levels near the Fermi level, and the SCF iterations. With `SMEAR` the SCF converged in 23 iterations, but to a state 16.6 kcal/mol above the one of the default SCF; compare heats and gaps (see Troubleshooting).",
    42: "Diamond: every gradient component with the default one-sided differences against `PRECISE` (central differences).",
    43: "Rutile, atoms relaxed in a fixed cell: heat and gradient norm; converged after 15 cycles.",
    44: "Γ-point frequencies of the 64-atom diamond cell (`FORCE`, 40 minutes on 4 CPU threads).",
    45: "Diamond, NVE from 300 K in a fixed cell, run with `CYCLES=60` (6 fs): temperature, and potential, kinetic and total energy on one scale (the total stays within 0.022 kcal/mol).",
    46: "Silicon: energy change per SCF iteration with the default SCF of solids and with `SHIFT=80`.",
    47: "The urea cell, and its heat with MOZYME and with the conventional SCF.",
    48: "α-quartz cell: SCF with and without the pseudo-diagonalization, both on the CPU (`MOPAC_PDIAG=1` turns it on there): same iterations and heats.",
    49: "The C₁₉₂H₃₈ flake and its SCF with `PULAY`. Without `PULAY` the same SCF on this laptop converged only after MOPAC had forced all its convergers on, after 49 minutes, 8.4 kcal/mol higher.",
    50: "Spin populations of the UHF cation: the unpaired electron is on Trp9.",
    51: "Gradient of the half-electron doublet: the analytic one (default) and `NOANCI` against central differences of the heat (±0.005 Å, checked with 0.002 and 0.01 Å and `SCFCRT=1.D-8`) for 12 coordinates. The analytic gradient agrees within 0.4 kcal/(mol Å) at 10 coordinates and is off by 1.0 and 1.25 at two (MOPAC 23.1.2 gives the same values there); `NOANCI` is off by up to 29, in MOPAC 23.1.2 too.",
    52: "C–C bond orders of the flake from `BONDS`.",
    53: "The QM region (thick, orange carbons), and the total energy of the whole system (3 992 atoms) in the NVE check: 200 NVT and 400 NVE steps with MOPAC and OpenMM on the CPU, default `--tolerance 0.01`. The driver reports a drift of 0.39 kJ/mol over the 400 steps and an rms deviation of 0.80 kJ/mol from a linear fit; `docs/QMMM_OPENMM.md` explains the sources and recommends 0.001 for NVE.",
    54: "Whole peptide as QM region: total energy of the whole system (12 095 atoms) in the NVE check after 100 NVT steps, at `--tolerance 0.001`, MOPAC and OpenMM on the CPU. 200 steps are too few for a drift rate; the driver reports a change of 1.17 kJ/mol and an rms deviation of 1.48 kJ/mol from a linear fit.",
    55: f"The QM region in trypsin–BPTI (PDB 2PTC), and three distances with and without the long-range QM–MM term: means of three 5-ps replicas each; {DOCS}.",
    56: f"Forces on the QM atoms against the cut-off, without and with the long-range term (trypsin–BPTI, 8 snapshots, reference PME at 24 Å); {DOCS}.",
    57: "What `add_qmmm` does to an OpenMM system (`scripts/qmmm/openmm_mopac.py`).",
    58: "QM/MM electrostatic energy of chignolin with one TIP3P water as point charges at increasing distance (one job per distance).",
    60: f"Largest difference between GPU and CPU heats in the benchmark checks; {README}.",
    61: "The cells of `tests/data/materials/small`.",
    62: "The 11 CPU checks of `tests/check_qmmm_pme_ewald.py`: value and tolerance of each (unit in brackets).",
    63: "Time of the stages of the SCF reported by `TIMES` for this job.",
    64: "FP64 rate and memory bandwidth of NVIDIA GPUs, vendor data as listed in the README (*Which GPU?*); only the A100 and the T4 were timed with this code.",
    65: f"Speed-up of the A100 against the system size; {README}.",
}
HEAD = ("<!-- figures -->\n<sub>Figures: calculations with the " + CPU + ", one core unless a caption says otherwise, "
        "or published numbers where the caption says so. Scripts: "
        "[`scripts/wiki_figures`](https://github.com/juvenalyosa/mopac_gpu/tree/main/scripts/wiki_figures).</sub>\n"
        "<!-- /figures -->\n\n")


def block(n):
    alt = re.sub(r"[`*]|\[|\]\([^)]*\)", "", CAP[n]).split(";")[0].split(". ")[0]
    return (f"<!-- fig {n} -->\n<picture>\n"
            f"  <source media=\"(prefers-color-scheme: dark)\" srcset=\"{RAW}/ex{n:02d}-dark.png\">\n"
            f"  <img src=\"{RAW}/ex{n:02d}.png\" alt=\"{alt.replace(chr(34), '')}\" width=\"760\">\n"
            f"</picture>\n\n<sub>{CAP[n]}</sub>\n<!-- /fig {n} -->\n")


def main():
    wiki = Path(sys.argv[1])
    done = []
    for page in sorted(wiki.glob("Examples-*.md")):
        t = page.read_text()
        t = re.sub(r"<!-- fig \d+ -->.*?<!-- /fig \d+ -->\n\n?", "", t, flags=re.S)
        t = re.sub(r"<!-- figures -->.*?<!-- /figures -->\n\n", "", t, flags=re.S)
        parts = re.split(r"(?m)^(?=## \d+\. )", t)
        out = [parts[0].rstrip("\n") + "\n\n" + HEAD]
        for sec in parts[1:]:
            n = int(re.match(r"## (\d+)\.", sec).group(1))
            out.append(sec.rstrip("\n") + "\n\n" + block(n) + "\n")
            done.append(n)
        page.write_text("".join(out).rstrip("\n") + "\n")
    print(len(done), "figures placed")


if __name__ == "__main__":
    main()

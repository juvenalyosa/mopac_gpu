"""Figures of the wiki examples.

    python figures.py <work dir of jobs.py> <wiki dir> [--ext dir] [--only 1,2,3]

Writes <wiki>/images/exNN.png and exNN-dark.png.  Figures from calculations read the outputs that jobs.py
left in the work directory; the others use numbers published in README.md and docs/QMMM_OPENMM.md (each
function says which).  --ext holds downloaded structures (2PTC.pdb).
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as L

REPO = Path(__file__).resolve().parents[2]
CHIG = REPO / "tests/data/mozyme_gpu_reference/1UAO_opt.pdb"
CRN_OPT = REPO / "tests/data/mozyme_gpu_reference/1CRN_opt.pdb"
CRN_H = REPO / "benchmarks/publication_inputs/hydrogenated/protein_crambin_1crn_hydrogenated.pdb"
CRN_CLEAN = REPO / "benchmarks/publication_inputs/pdb_clean/protein_crambin_1crn.pdb"
DNA = REPO / "benchmarks/publication_inputs/hydrogenated/dna_dodecamer_1bna_hydrogenated.pdb"
G6X = REPO / "benchmarks/publication_inputs/mop/protein_1g6x.pdb"
CELLS = REPO / "tests/data/materials/small"
FIG = {}


def figure(n):
    def deco(f):
        FIG[n] = f
        return f
    return deco


class Ctx:
    def __init__(self, work, ext, renders):
        self.work, self.ext, self.renders = work, ext, renders

    def out(self, job, name="job.out"):
        return L.text(self.work / job / name)

    def has(self, *jobs):
        return all((self.work / j / "done.json").exists() for j in jobs)


def label_bars(ax, bars, texts, c, inside=False):
    for b, t in zip(bars, texts):
        x = b.get_x() + b.get_width() if b.get_width() >= 0 else b.get_x()
        ax.text(x * (1.08 if ax.get_xscale() == "log" else 1) + (0 if ax.get_xscale() == "log" else 0.01 * ax.get_xlim()[1]),
                b.get_y() + b.get_height() / 2, t, va="center", ha="left", fontsize=8.5, color=c["text2"])


# ------------------------------------------------------------------------------------------- renders
def render_values(ctx, theme, pdb, values, name, vmax=None, style="sticks", cartoon=False, palette="div",
                  width=1300, height=1000, turn=None, extra=None):
    """PyMOL render of a PDB coloured by per-atom values (B-factor spectrum)"""
    out = ctx.renders / f"{name}-{theme}.png"
    vals = list(map(float, values))
    v = vmax or max(abs(min(vals)), abs(max(vals)))

    def build(cmd, c):
        cmd.load(str(pdb), "m")
        cmd.remove("solvent")
        cmd.alter("m", "b=vals[int(index)-1]", space={"vals": vals})
        cmd.hide("everything")
        if cartoon:
            cmd.dss("m")
            cmd.show("cartoon")
            if style != "cartoon_only":
                cmd.set("cartoon_transparency", 0.55)
                cmd.set("cartoon_color", "mid_t")
        if style == "sticks":
            cmd.show("sticks")
            cmd.set("stick_radius", 0.17)
            cmd.set("stick_h_scale", 0.6)
        elif style == "spheres":
            cmd.show("spheres")
            cmd.set("sphere_scale", 0.3)
        names = "d1 d2 d3 d4 d5" if palette == "div" else "q1 q2 q3 q4 q5"
        lo = -v if palette == "div" else 0.0
        cmd.spectrum("b", names, "m", minimum=lo, maximum=v)
        if extra:
            extra(cmd, c)
        cmd.orient("m")
    L.pymol_render(out, build, width, height, theme, turn)
    return out, v


def colorbar(fig, ax, c, vmin, vmax, label, palette="div"):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cm = LinearSegmentedColormap.from_list("p", L.rdiv(c) if palette == "div" else c["seq"])
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cm)
    b = ax.get_position()
    cax = fig.add_axes([b.x1 - 0.012, b.y0 + 0.2 * b.height, 0.012, 0.6 * b.height])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, color=c["text2"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=c["text2"], length=2)
    return cb


# ------------------------------------------------------------------------------------------- README data
# MOZYME single points, PM7, one A100-SXM4-40GB against one core of the same Colab machine (README.md,
# benchmark run at commit c85eb96d)
MOZYME_SP = [("Crambin (1CRN)", 642, 25.4, 1.22), ("DNA dodecamer (1BNA)", 781, 19.7, 1.17),
             ("Ubiquitin (1UBQ)", 1231, 49.2, 1.42), ("Barnase (1RNB)", 1778, 78.4, 2.07),
             ("Antifreeze protein (1EZG)", 2064, 140, 3.07), ("Bacteriorhodopsin (1C3W)", 4473, 294, 5.53),
             ("Adenylate kinase (1AKE)", 6689, 695, 9.09)]
# conventional SCF of solids, PM7 1SCF GRADIENTS, A100 against 12 CPU threads (README.md, c85eb96d)
SOLIDS = [("Diamond", 64, 5.2, 1.2), ("Graphene (2D)", 72, 102, 77.7), ("Silicon", 512, 369, 22.0),
          ("Rutile TiO2", 576, 179, 14.0), ("alpha-quartz SiO2", 576, 172, 13.8), ("Urea crystal", 768, 80.6, 9.4)]


@figure(2)
def ex02(theme, ctx):
    c = L.style(theme)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    names = [n for n, *_ in MOZYME_SP][::-1]
    cpu = np.array([x[2] for x in MOZYME_SP][::-1])
    gpu = np.array([x[3] for x in MOZYME_SP][::-1])
    y = np.arange(len(names))
    h = 0.38
    b1 = ax.barh(y + h / 2 + 0.01, cpu, h, color=c["series"][1], label="CPU code path (NOGPU), 1 core")
    b2 = ax.barh(y - h / 2 - 0.01, gpu, h, color=c["series"][0], label="GPU (A100)")
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{n}  {a:,} atoms".replace(",", " ") for n, a, *_ in MOZYME_SP][::-1])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    speed = [21, 17, 35, 38, 45, 53, 76][::-1]           # as printed in README.md
    for yy, cp, gp, sp in zip(y, cpu, gpu, speed):
        ax.text(cp * 1.12, yy + h / 2, f"{cp:g} s", va="center", fontsize=8, color=c["text2"])
        ax.text(gp * 1.12, yy - h / 2, f"{gp:g} s  ({sp}x)", va="center", fontsize=8, color=c["text2"])
    ax.set_xlim(0.6, 3000)
    L.plain_log(ax, "x", [1, 10, 100, 1000])
    ax.set_xlabel("time of a PM7 MOZYME single point (s, log scale)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return fig


@figure(39)
def ex39(theme, ctx):
    c = L.style(theme)
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    data = SOLIDS[::-1]
    y = np.arange(len(data))
    h = 0.38
    cpu = np.array([d[2] for d in data]); gpu = np.array([d[3] for d in data])
    ax.barh(y + h / 2 + 0.01, cpu, h, color=c["series"][1], label="CPU, 12 threads")
    ax.barh(y - h / 2 - 0.01, gpu, h, color=c["series"][0], label="GPU (A100)")
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{n}  {a} atoms" for n, a, *_ in data])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    for yy, cp, gp in zip(y, cpu, gpu):
        ax.text(cp * 1.12, yy + h / 2, f"{cp:g} s", va="center", fontsize=8, color=c["text2"])
        ax.text(gp * 1.12, yy - h / 2, f"{gp:g} s  ({cp / gp:.1f}x)", va="center", fontsize=8, color=c["text2"])
    ax.set_xlim(0.6, 1500)
    L.plain_log(ax, "x", [1, 10, 100, 1000])
    ax.set_xlabel("time of PM7 1SCF GRADIENTS (s, log scale)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.axhline(4.5, color=c["axis"], lw=0.8, ls=(0, (3, 3)))
    ax.text(1400, 4.62, "cells of 100 atoms or fewer: on the GPU only with MOPAC_FORCEGPU=1", ha="right",
            va="bottom", fontsize=8, color=c["text2"])
    return fig


@figure(59)
def ex59(theme, ctx):
    """largest |GPU - CPU| of the heat of formation, README.md Accuracy (A100, c85eb96d)"""
    c = L.style(theme)
    rows = [("Water clusters, 375-6 000 atoms", 0.006), ("Crambin, DNA, BPTI, ubiquitin,\nbarnase, bacteriorhodopsin", 0.02),
            ("Antifreeze protein (1EZG)", 0.088), ("Adenylate kinase (1AKE)", 0.41),
            ("Same, SCFCRT=1.D-6", 5e-5), ("MNDO ... PM7, crambin and DNA", 0.006),
            ("COSMO: crambin, DNA, ubiquitin", 0.030), ("Solids, 13 cells (graphene)", 0.007),
            ("Solids, the other 12 cells", 0.0005)][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(rows))
    v = [r[1] for r in rows]
    ax.hlines(y, 1e-5, v, color=c["grid"], lw=2)
    ax.plot(v, y, "o", color=c["series"][0], ms=7, mec=c["surface"], mew=1.5)
    for yy, vv in zip(y, v):
        ax.text(vv * 1.4, yy, f"{vv:g}", va="center", fontsize=8.5, color=c["text2"])
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 2)
    L.plain_log(ax, "x", [1e-4, 1e-3, 0.01, 0.1, 1])
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.axvline(0.1, color=c["series"][1], lw=1, ls=(0, (3, 3)))
    ax.text(0.1, len(rows) - 0.35, " 0.1 kcal/mol", color=c["series"][1], fontsize=8.5, va="bottom")
    ax.set_xlabel("largest |GPU − CPU| of the heat of formation (kcal/mol, log scale)")
    return fig


@figure(63)
def ex63(theme, ctx):
    """FP64 rate and memory bandwidth of NVIDIA GPUs, vendor data sheets as listed in README.md (Which GPU?)"""
    c = L.style(theme)
    gpus = [("H200", 34, 4.8), ("H100 SXM", 34, 3.35), ("H100 PCIe", 26, 2.0), ("A100", 9.7, 1.8),
            ("V100", 7.8, 0.9), ("A30", 5.2, 0.9), ("L40S", 1.4, 0.86), ("RTX 4090", 1.3, 1.0), ("L4", 0.5, 0.3),
            ("T4", 0.25, 0.32)]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for n, f, b in gpus:
        measured = n in ("A100", "T4")
        ax.plot(b, f, "o", ms=8 if measured else 6, color=c["series"][0] if measured else c["axis"],
                mec=c["surface"], mew=1.2, zorder=3)
        off = {"H100 SXM": (6, -10), "H100 PCIe": (6, -4), "L40S": (6, -8), "RTX 4090": (6, 2), "A30": (6, -8),
               "V100": (6, 3), "L4": (6, -2), "T4": (6, -9)}.get(n, (6, -3))
        if n == "L40S":
            off = (-30, -12)
        ax.annotate(n + ("  (timed with this code)" if measured else ""), (b, f), xytext=off,
                    textcoords="offset points", fontsize=8.5, color=c["text"] if measured else c["text2"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.2, 8); ax.set_ylim(0.15, 60)
    L.plain_log(ax, "x", [0.3, 1, 3])
    L.plain_log(ax, "y", [0.3, 1, 3, 10, 30])
    ax.set_xlabel("memory bandwidth (TB/s)")
    ax.set_ylabel("FP64 rate without tensor cores (TFLOPS)")
    return fig


@figure(64)
def ex64(theme, ctx):
    c = L.style(theme)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, data, ref, col in ((a1, MOZYME_SP, "MOZYME single points\nagainst 1 CPU core", 0),
                               (a2, SOLIDS, "Solids, conventional SCF\nagainst 12 CPU threads", 2)):
        n = np.array([d[1] for d in data]); s = np.array([d[2] / d[3] for d in data])
        ax.plot(n, s, "o", color=c["series"][col], ms=7, mec=c["surface"], mew=1.2)
        for d, nn, ss in zip(data, n, s):
            name = d[0].split(" (")[0]
            if name == "alpha-quartz SiO2":
                continue
            if name == "Rutile TiO2":
                name = "Rutile and quartz (576)"
            ax.annotate(name, (nn, ss), xytext=(5, -3), textcoords="offset points", fontsize=7.5, color=c["text2"])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ref, fontsize=9.5)
        ax.set_xlabel("atoms")
    a1.set_ylabel("speed-up of the A100")
    a1.set_xlim(400, 15000); a1.set_ylim(8, 120)
    a2.set_xlim(40, 2500); a2.set_ylim(0.8, 30)
    L.plain_log(a1, "x", [500, 1000, 2000, 5000, 10000]); L.plain_log(a1, "y", [10, 20, 50, 100])
    L.plain_log(a2, "x", [50, 100, 200, 500, 1000, 2000]); L.plain_log(a2, "y", [1, 2, 5, 10, 20])
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------- docs data
@figure(56)
def ex56(theme, ctx):
    """cut-off test of trypsin-BPTI, docs/QMMM_OPENMM.md (Example: trypsin with BPTI)"""
    c = L.style(theme)
    roff = [8, 12, 16, 20]
    cut = [33.6, 28.0, 24.6, 19.9]; pme = [8.2, 6.4, 5.8, 4.0]
    spread_cut = [29.5, 44.5, 23.9, 23.9]; spread_pme = [2.8, 1.7, 1.9, 1.9]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for ax, a, b, lab in ((a1, cut, pme, "rms error of the forces on the QM atoms\n(kJ/(mol nm); rms force 796)"),
                          (a2, spread_cut, spread_pme, "change of the QM/MM energy differences\nbetween snapshots (kJ/mol)")):
        ax.plot(roff, a, "o-", color=c["series"][1], label="cut-off only")
        ax.plot(roff, b, "o-", color=c["series"][0], label="with long-range QM–MM (PME)")
        ax.set_xticks(roff)
        ax.set_xlabel("roff (Å)")
        ax.set_ylabel(lab)
        ax.set_ylim(0, None)
    a1.text(20.3, cut[-1], "cut-off only", color=c["series"][1], fontsize=8.5, va="center")
    a1.text(20.3, pme[-1], "PME", color=c["series"][0], fontsize=8.5, va="center")
    a1.set_xlim(7, 24)
    a2.set_xlim(7, 21)
    a2.legend(loc="center right")
    fig.tight_layout()
    return fig


@figure(55)
def ex55(theme, ctx):
    """2PTC with the QM region; distances: three replicas, 5 ps NVT each (docs/QMMM_OPENMM.md)"""
    c = L.style(theme)
    pdb = ctx.ext / "2PTC.pdb"
    out = ctx.renders / f"ex55-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(pdb), "m")
        cmd.remove("solvent or hetatm")
        cmd.hide("everything")
        cmd.dss("m")
        cmd.show("cartoon")
        cmd.color("carbon_t", "chain E")
        cmd.color("s1", "chain I")
        cmd.set("cartoon_transparency", 0.6, "chain E")
        qm = "(chain E and resi 57+102+195 and not name N+C+O+CA) or (chain I and resi 15+16)"
        cmd.show("sticks", qm)
        cmd.set("stick_radius", 0.3)
        L.element_colours(cmd, qm)
        cmd.color("s2", f"({qm}) and elem C")
        cmd.orient("chain E and resi 57+102+195 or chain I and resi 15+16")
        cmd.zoom("chain E and resi 57+102+195 or chain I and resi 15+16", 5)
    L.pymol_render(out, build, 1200, 1000, theme)
    fig = plt.figure(figsize=(8.6, 3.6))
    a0 = fig.add_axes([0.0, 0.02, 0.42, 0.96])
    L.image_axes(a0, out)
    a0.text(0.02, 0.02, "trypsin (grey), BPTI (blue);\nQM region as sticks (orange carbons)", transform=a0.transAxes,
            fontsize=8, color=c["text2"])
    ax = fig.add_axes([0.62, 0.17, 0.36, 0.72])
    lab = ["Ser195 OG – Lys15 C", "His57 NE2 – Ser195 OG", "His57 HD1 – Asp102 OD2"]
    cut = [(2.42, 0.05), (3.03, 0.06), (1.76, 0.06)]
    pme = [(2.55, 0.08), (3.15, 0.06), (1.83, 0.03)]
    y = np.arange(3)[::-1]
    ax.errorbar([m for m, s in cut], y + 0.12, xerr=[s for m, s in cut], fmt="o", color=c["series"][1], capsize=3,
                label="cut-off only")
    ax.errorbar([m for m, s in pme], y - 0.12, xerr=[s for m, s in pme], fmt="o", color=c["series"][0], capsize=3,
                label="with PME")
    ax.set_yticks(y, lab)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean distance (Å), mean ± SD of 3 replicas")
    ax.set_xlim(1.5, 3.4)
    ax.set_ylim(-0.6, 2.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.2), ncol=2)
    return fig


def box(ax, x, y, w, h, title, body, c, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=c["surface"],
                                ec=col, lw=1.4))
    ax.text(x + 0.015, y + h - 0.03, title, fontsize=9.5, fontweight="bold", color=c["text"], va="top")
    ax.text(x + 0.015, y + h - 0.12, body, fontsize=8.3, color=c["text2"], va="top", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, c):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c["axis"], lw=1.3,
                                                        shrinkA=2, shrinkB=2))


@figure(57)
def ex57(theme, ctx):
    """what add_qmmm does (scripts/qmmm/openmm_mopac.py)"""
    c = L.style(theme)
    fig = plt.figure(figsize=(8.6, 2.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, 0.01, 0.32, 0.20, 0.40, "OpenMM System", "force field, water, PME,\nconstraints\n(createSystem)", c,
        c["axis"])
    box(ax, 0.27, 0.03, 0.40, 0.94, "add_qmmm(system, topology, ...)",
        "• MM bonds, angles, torsions inside the\n  QM region switched off; constraints\n  between QM atoms removed\n"
        "• QM atoms: charge 0, Lennard-Jones kept\n• hydrogen link atom on each QM–MM bond\n"
        "• MM charges switched off by groups\n  between ron and roff\n• long-range QM–MM by PME (default)\n"
        "• MOPAC (PM7, MOZYME, HB-SMOOTH) as\n  an OpenMM PythonForce", c, c["series"][0])
    box(ax, 0.73, 0.32, 0.26, 0.40, "Simulation", "any integrator, thermostat,\nbarostat, reporter\n"
        "(LangevinMiddle, Verlet, ...)", c, c["axis"])
    arrow(ax, 0.215, 0.51, 0.265, 0.51, c)
    arrow(ax, 0.675, 0.51, 0.725, 0.51, c)
    return fig


# ------------------------------------------------------------------------------------------- run on demand
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("wiki", type=Path)
    ap.add_argument("--ext", type=Path, required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    renders = a.work.parent / "wikifig_renders"
    renders.mkdir(exist_ok=True)
    ctx = Ctx(a.work, a.ext, renders)
    want = [int(x) for x in a.only.split(",")] if a.only else sorted(FIG)
    images = a.wiki / "images"
    for n in want:
        for theme in ("light", "dark"):
            try:
                fig = FIG[n](theme, ctx)
            except FileNotFoundError as e:
                print(f"ex{n:02d}: missing data ({e.filename})")
                break
            if fig is None:
                print(f"ex{n:02d}: no data yet")
                break
            fn = L.save(fig, images, f"ex{n:02d}", theme)
        else:
            print(f"ex{n:02d}: {fn.name}")


if __name__ == "__main__":
    main()

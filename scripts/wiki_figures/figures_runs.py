"""Figures of the wiki examples that read the calculations of jobs.py (CPU build of this repository)."""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figlib as L
from figures import (figure, render_values, colorbar, CHIG, CRN_H, CRN_CLEAN, CRN_OPT, DNA, G6X, CELLS, REPO,
                     box, arrow)

KB = 0.0019872041          # kcal/(mol K), as in drc_ensemble.F90


def two(fig_w=8.6, fig_h=3.5, split=0.46):
    """figure with a render on the left and a plot on the right"""
    fig = plt.figure(figsize=(fig_w, fig_h))
    a0 = fig.add_axes([0.0, 0.0, split - 0.06, 1.0])
    a1 = fig.add_axes([split + 0.12, 0.16, 0.98 - split - 0.12, 0.74])
    return fig, a0, a1


def energy_panel(ax, rows, c, title=True):
    """potential, kinetic and total energy (changes from the first row) on one scale, so that the conservation of
    the total is seen against the energy the other two exchange"""
    t = np.array([r[0] for r in rows])
    pot = np.array([r[1] for r in rows]) - rows[0][1]
    kin = np.array([r[2] for r in rows]) - rows[0][2]
    tot = np.array([r[3] for r in rows]) - rows[0][3]
    ax.plot(t, pot, color=c["series"][0], lw=1.4, label="potential")
    ax.plot(t, kin, color=c["series"][1], lw=1.4, label="kinetic")
    ax.plot(t, tot, color=c["series"][2], lw=2.4, label="total")
    ax.axhline(0, color=c["axis"], lw=0.8)
    ax.set_xlabel("time (fs)"); ax.set_ylabel("change from 0 fs (kcal/mol)")
    lo, hi = min(pot.min(), kin.min()), max(pot.max(), kin.max())
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.35 * (hi - lo))          # room for the legend above the curves
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    if title:
        ax.set_title(f"total within {np.abs(tot).max():.3f} kcal/mol; potential and kinetic\nexchange up to "
                     f"{max(np.ptp(pot), np.ptp(kin)):.0f} kcal/mol", fontsize=8.5, fontweight="normal")
    return tot


def note(ax, s, c, x=0.02, y=0.02, ha="left"):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, color=c["text2"], ha=ha, va="bottom")


# --------------------------------------------------------------------------------------------- getting started
@figure(1)
def ex01(theme, ctx):
    c = L.style(theme)
    t = ctx.out("sp_chig")
    q = L.charges(t)
    png, v = render_values(ctx, theme, CHIG, [x[1] for x in q], "ex01", vmax=0.8)
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    note(a0, f"PM7 MOZYME, heat of formation {L.heat(t):.2f} kcal/mol", c)
    colorbar(fig, a0, c, -0.8, 0.8, "atomic charge (e)")
    res = L.residue_sums(q)
    y = np.arange(len(res))[::-1]
    vals = [r[3] for r in res]
    a1.barh(y, vals, 0.65, color=[c["series"][7] if v > 0 else c["series"][0] for v in vals])
    a1.set_yticks(y, [f"{r[0].title()}{r[2]}" for r in res])
    a1.axvline(0, color=c["axis"], lw=0.8)
    a1.grid(axis="y", visible=False)
    a1.tick_params(axis="y", length=0)
    a1.set_xlabel("sum of the atomic charges of the residue (e)")
    return fig


@figure(3)
def ex03(theme, ctx):
    c = L.style(theme)
    g = L.cart_gradients(ctx.out("grad_chig"))
    n = np.linalg.norm(g, axis=1)
    png, _ = render_values(ctx, theme, CHIG, n, "ex03", vmax=float(np.percentile(n, 98)), palette="seq")
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    colorbar(fig, a0, c, 0, float(np.percentile(n, 98)), "|gradient| (kcal/(mol Å))", palette="seq")
    els = [a[4] for a in L.pdb_atoms(CHIG)]
    bins = np.linspace(0, n.max() * 1.02, 30)
    for k, el in enumerate(["H", "C", "N", "O"]):
        sel = n[[e == el for e in els]]
        a1.hist(sel, bins, histtype="step", lw=1.6, color=c["series"][k], label=f"{el} ({len(sel)})")
    a1.set_xlabel("|gradient| of the atom (kcal/(mol Å))")
    a1.set_ylabel("atoms")
    a1.legend()
    a1.set_title(f"norm of the whole gradient {np.linalg.norm(g):.1f} kcal/(mol Å)", fontsize=9, fontweight="normal")
    return fig


@figure(4)
def ex04(theme, ctx):
    c = L.style(theme)
    crit = ["1.D-2", "1.D-3", "1.D-4", "1.D-5", "1.D-6"]
    h = {k: L.heat(ctx.out(f"scfcrt_{k}")) for k in crit}
    s = {k: L.job_seconds(ctx.out(f"scfcrt_{k}")) for k in crit}
    ref = h["1.D-6"]
    x = [float(k.replace("D", "E")) for k in crit[:-1]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    dh = [abs(h[k] - ref) for k in crit[:-1]]
    floor = 5e-6                       # heats are printed with 5 decimals
    a1.plot(x, [max(d, floor) for d in dh], "o-", color=c["series"][0])
    for xx, d in zip(x, dh):
        a1.annotate("same to 5 decimals" if d < 1e-5 else f"{d:.5f}", (xx, max(d, floor)), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=c["text2"])
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlim(2e-2, 4e-6)
    L.plain_log(a1, "x", [1e-2, 1e-3, 1e-4, 1e-5])
    a1.set_xticklabels(["1.D-2", "1.D-3", "1.D-4", "1.D-5"])
    a1.set_xlabel("SCFCRT")
    a1.set_ylabel("|heat − heat at SCFCRT=1.D-6| (kcal/mol)")
    a2.bar(range(len(crit)), [s[k] for k in crit], 0.6, color=c["series"][0])
    a2.set_xticks(range(len(crit)), crit)
    a2.set_xlabel("SCFCRT")
    a2.set_ylabel("job time on one CPU core (s)")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


@figure(5)
def ex05(theme, ctx):
    c = L.style(theme)
    t = ctx.out("sp_crambin")
    q = L.charges(t)
    res = L.residue_sums(q)
    key = {(r[1], r[2]): r[3] for r in res}
    atoms = L.pdb_atoms(CRN_H)
    vals = [key.get((a[2], a[3]), 0.0) for a in atoms]
    vm = 0.12
    png, _ = render_values(ctx, theme, CRN_H, vals, "ex05", vmax=vm, style="cartoon_only", cartoon=True)
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    colorbar(fig, a0, c, -vm, vm, "charge of the residue (e)")
    note(a0, f"crambin, 642 atoms; heat {L.heat(t):.1f} kcal/mol; {L.job_seconds(t):.0f} s on one CPU core", c)
    v = np.array([r[3] for r in res])
    a1.bar(np.arange(1, len(v) + 1), v, 0.8, color=[c["series"][7] if x > 0 else c["series"][0] for x in v])
    a1.axhline(0, color=c["axis"], lw=0.8)
    a1.set_xlabel("residue")
    a1.set_ylabel("sum of the atomic charges (e)")
    a1.grid(axis="x", visible=False)
    for i, r in enumerate(res):
        if abs(r[3]) > 0.45:
            a1.text(i + 1, r[3] + (0.05 if r[3] > 0 else -0.05), f"{r[0].title()}{r[2]}", ha="center",
                    va="bottom" if r[3] > 0 else "top", fontsize=7.5, color=c["text2"])
    return fig


@figure(6)
def ex06(theme, ctx):
    c = L.style(theme)
    t = ctx.out("sp_dna")
    q = L.charges(t)
    res = L.residue_sums(q)
    key = {(r[1], r[2]): r[3] for r in res}
    atoms = L.pdb_atoms(DNA)
    vals = [key.get((a[2], a[3]), 0.0) for a in atoms]
    png, _ = render_values(ctx, theme, DNA, vals, "ex06", vmax=1.0, turn=[("z", 90)])
    m = re.findall(r"COMPUTED CHARGE ON SYSTEM:\s*([+-]?\d+)", t)
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    colorbar(fig, a0, c, -1, 1, "charge of the nucleotide (e)")
    note(a0, f"computed charge on the system {m[-1] if m else '?'}", c)
    chains = sorted({r[1] for r in res})
    for k, ch in enumerate(chains):
        rr = [r for r in res if r[1] == ch]
        a1.plot([r[2] for r in rr], [r[3] for r in rr], "o-", color=c["series"][k], label=f"strand {ch}")
        for r in rr:
            a1.annotate(r[0].strip()[-1], (r[2], r[3]), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=7, color=c["text2"])
    a1.axhline(0, color=c["axis"], lw=0.8)
    a1.set_xlabel("residue number")
    a1.set_ylabel("sum of the atomic charges (e)")
    a1.legend()
    return fig


METHODS = ["MNDO", "AM1", "PM3", "RM1", "PM6", "PM6-D3H4", "PM7"]


@figure(7)
def ex07(theme, ctx):
    c = L.style(theme)
    h = [L.heat(ctx.out(f"method_{m}")) for m in METHODS]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(METHODS))[::-1]
    bars = ax.barh(y, h, 0.6, color=[c["series"][0] if m != "PM6-D3H4" else c["series"][1] for m in METHODS])
    for yy, v in zip(y, h):
        ax.text(v - 12, yy, f"{v:.1f}", va="center", ha="right", fontsize=8.5, color=c["text2"])
    ax.set_xlim(min(h) * 1.18, 0)
    ax.set_yticks(y, METHODS)
    ax.axvline(0, color=c["axis"], lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("heat of formation of chignolin, MOZYME 1SCF (kcal/mol)")
    return fig


@figure(8)
def ex08(theme, ctx):
    c = L.style(theme)
    s = [L.job_seconds(ctx.out(f"method_{m}")) for m in METHODS]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(METHODS))
    ax.bar(x, s, 0.6, color=[c["series"][1] if m == "AM1" else c["series"][0] for m in METHODS])
    for xx, v in zip(x, s):
        ax.text(xx, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=c["text2"])
    ax.set_xticks(x, METHODS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("job time, chignolin 1SCF,\none CPU core (s)")
    return fig


@figure(9)
def ex09(theme, ctx):
    c = L.style(theme)
    ns = [50, 100, 200, 400, 800]
    s = [L.job_seconds(ctx.out(f"water_{n}")) for n in ns]
    mop = (ctx.work / "water_50" / "job.mop").read_text().splitlines()[3:]
    atoms = [(f.split()[0], np.array([float(f.split()[1]), float(f.split()[3]), float(f.split()[5])]))
             for f in mop if len(f.split()) >= 6]
    pdb = ctx.renders / "water50.pdb"
    L.write_pdb(atoms, pdb)
    out = ctx.renders / f"ex09-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(pdb), "w")
        cmd.hide("everything"); cmd.show("sticks"); cmd.show("spheres")
        cmd.set("sphere_scale", 0.22); cmd.set("stick_radius", 0.12)
        L.element_colours(cmd)
        cmd.orient(); cmd.turn("x", -25); cmd.turn("y", 30)
    L.pymol_render(out, build, 1100, 1000, theme)
    fig, a0, a1 = two()
    L.image_axes(a0, out)
    note(a0, "50 water molecules on a 3.1-Å cubic lattice", c)
    a1.plot(ns, s, "o-", color=c["series"][0], label="this Mac, one CPU core")
    a1.plot(ns, np.array(ns) * s[-1] / ns[-1], color=c["axis"], lw=1, ls=(0, (3, 3)), label="proportional to N")
    a1.set_xscale("log"); a1.set_yscale("log")
    L.plain_log(a1, "x", ns)
    a1.set_xlabel("water molecules")
    a1.set_ylabel("PM7 MOZYME 1SCF job time (s)")
    a1.legend()
    return fig


# --------------------------------------------------------------------------------------------- preparing
def render_pdb(ctx, theme, pdb, name, build_extra=None, width=1100, height=1000, h=True, turn=None):
    out = ctx.renders / f"{name}-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(pdb), "m")
        cmd.remove("solvent")
        if not h:
            cmd.remove("hydro")
        cmd.hide("everything")
        cmd.show("sticks")
        cmd.set("stick_radius", 0.14)
        L.element_colours(cmd)
        if build_extra:
            build_extra(cmd, cc)
        cmd.orient("m")
    L.pymol_render(out, build, width, height, theme, turn)
    return out


@figure(10)
def ex10(theme, ctx):
    c = L.style(theme)
    after = ctx.work / "prep_neutral" / "job.pdb"
    n0 = len(L.pdb_atoms(CRN_CLEAN)); n1 = len(L.pdb_atoms(after))
    p0 = render_pdb(ctx, theme, CRN_CLEAN, "ex10a")
    p1 = render_pdb(ctx, theme, after, "ex10b")
    fig = plt.figure(figsize=(8.6, 3.6))
    for k, (p, s) in enumerate(((p0, f"crambin from the PDB: {n0} atoms, no hydrogens"),
                                (p1, f"after ADD-H: {n1} atoms ({n1 - n0} hydrogens added)"))):
        ax = fig.add_axes([k * 0.5, 0.06, 0.5, 0.94])
        L.image_axes(ax, p)
        ax.text(0.5, -0.02, s, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=c["text"])
    return fig


@figure(11)
def ex11(theme, ctx):
    c = L.style(theme)
    t = ctx.out("prep_ionize")
    i = t.find("Hydrogen atoms added to")
    added, removed = [], []
    for line in t[i:].splitlines()[2:]:
        m = re.match(r"\s+\d+\s+(\S+)\s+(\w{3})\s+(\w)\s+(\d+)(?:\s+(\S+)\s+(\w{3})\s+(\w)\s+(\d+))?", line)
        if not m:
            break
        added.append((m.group(2), m.group(3), int(m.group(4)), m.group(1)))
        if m.group(5):
            removed.append((m.group(6), m.group(7), int(m.group(8)), m.group(5)))
    pdb = ctx.work / "prep_ionize" / "job.pdb"

    def extra(cmd, cc):
        cmd.dss("m")
        cmd.show("cartoon"); cmd.set("cartoon_transparency", 0.6); cmd.color("carbon_t", "m and rep cartoon")
        cmd.hide("sticks", "m")
        for res, ch, num, at in added:
            cmd.show("sticks", f"chain {ch} and resi {num} and not name N+C+O" + (" or (chain %s and resi %d and name N+H*)" % (ch, num) if at == "N" else ""))
            cmd.color("s1", f"chain {ch} and resi {num} and elem C")
        for res, ch, num, at in removed:
            cmd.show("sticks", f"chain {ch} and resi {num} and not name N+C+O" + (" or (chain %s and resi %d and name C+O+OXT)" % (ch, num) if at == "OXT" else ""))
            cmd.color("s8", f"chain {ch} and resi {num} and elem C")
        cmd.set("stick_radius", 0.2)
    png = render_pdb(ctx, theme, pdb, "ex11", extra)
    fig, a0, a1 = two(split=0.5)
    L.image_axes(a0, png)
    note(a0, "carbon atoms of the sites: blue = protonated (+), red = deprotonated (−)", c)
    a1.set_axis_off()
    rows = [("+1", f"{r} {n}" + (" (N-terminus)" if at == "N" else ""), c["series"][0]) for r, _, n, at in added] + \
           [("−1", f"{r} {n}" + (" (C-terminus)" if at == "OXT" else ""), c["series"][7]) for r, _, n, at in removed]
    a1.text(0, 1.02, "SITE=(IONIZE) on crambin", fontsize=10, fontweight="bold", color=c["text"], transform=a1.transAxes)
    for k, (q, s, col) in enumerate(rows):
        yy = 0.86 - k * 0.12
        a1.text(0.02, yy, q, color=col, fontsize=11, fontweight="bold", transform=a1.transAxes)
        a1.text(0.16, yy, s.title().replace("(N-Terminus)", "(N-terminus)").replace("(C-Terminus)", "(C-terminus)"),
                color=c["text"], fontsize=10, transform=a1.transAxes)
    a1.text(0.02, 0.86 - len(rows) * 0.12 - 0.02, f"net charge {len(added) - len(removed):+d}", color=c["text2"],
            fontsize=9.5, transform=a1.transAxes)
    return fig


@figure(12)
def ex12(theme, ctx):
    c = L.style(theme)
    t = ctx.out("check_1g6x")
    bad = [(m[0], m[1], int(m[2])) for m in re.findall(r"ERROR\s+(\w{3})\s+(\w)\s+(\d+):", t)]
    errs = re.findall(r"ERROR\s+(\w{3} \w\s+\d+: [^.]*)\.", t)
    out = ctx.renders / f"ex12-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(G6X), "m")
        cmd.remove("solvent")
        cmd.hide("everything")
        cmd.dss("m"); cmd.show("cartoon", "polymer"); cmd.color("carbon_t", "polymer")
        cmd.set("cartoon_transparency", 0.5)
        cmd.show("sticks", "resn SO4"); cmd.set("stick_radius", 0.25)
        cmd.color("grey60", "resn SO4")
        for r, ch, n in bad:
            cmd.color("s8", f"resn {r} and chain {ch} and resi {n}")
            cmd.show("spheres", f"resn {r} and chain {ch} and resi {n}"); cmd.set("sphere_scale", 0.45)
        cmd.orient("m")
        cmd.zoom("m", -4)
    L.pymol_render(out, build, 1200, 1000, theme)
    fig, a0, a1 = two(split=0.5)
    L.image_axes(a0, out)
    note(a0, "BPTI (1G6X) with its sulfate groups; the two flagged groups in red", c)
    a1.set_axis_off()
    a1.text(0, 1.02, "INPUT CHEMISTRY CHECK", fontsize=10, fontweight="bold", color=c["text"], transform=a1.transAxes)
    for k, e in enumerate(errs):
        a1.text(0.0, 0.84 - k * 0.2, "ERROR", color=c["critical"], fontsize=9.5, fontweight="bold", transform=a1.transAxes)
        a1.text(0.18, 0.84 - k * 0.2, e.replace(", ", ",\n"), color=c["text"], fontsize=9, transform=a1.transAxes,
                va="center" if False else "baseline")
    a1.text(0.0, 0.84 - len(errs) * 0.2 - 0.05, "The job stops before the SCF;\nLET runs it anyway.", color=c["text2"],
            fontsize=9, transform=a1.transAxes)
    return fig


@figure(13)
def ex13(theme, ctx):
    c = L.style(theme)
    w = L.scf_trace(ctx.out("let_1g6x", "scf_with.out"))
    wo = L.scf_trace(ctx.out("let_1g6x", "scf_without.out"))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for tr, col, lab in ((w, c["series"][1], "with the two half sulfates (LET)"),
                         (wo, c["series"][0], "half sulfates removed")):
        it = [x[0] for x in tr if x[2] != 0.0]
        de = [abs(x[2]) for x in tr if x[2] != 0.0]
        ax.plot(it, de, color=col, lw=1.5)
        ax.text(it[-1] + 1, de[-1], f"{lab}\n{len(tr)} iterations", color=col, fontsize=8.5, va="center")
    ax.set_yscale("log")
    ax.set_xlim(0, max(len(w), len(wo)) + 30)
    ax.set_xlabel("SCF iteration")
    ax.set_ylabel("|energy change| per iteration (kcal/mol)")
    L.plain_log(ax, "y", [1e-4, 1e-2, 1, 100, 1e4])
    return fig


@figure(14)
def ex14(theme, ctx):
    c = L.style(theme)
    wf = ctx.work / "workflow" / "wf"
    opt = L.opt_cycles(L.text(next((wf / "opt").glob("*.out"))))
    nvt = L.drc_rows(L.text(next((wf / "nvt").glob("*.out"))))
    nve = L.drc_rows(L.text(next((wf / "nve").glob("*.out"))))
    natoms = len(L.pdb_atoms(next((wf / "prepare").glob("*.pdb"))))
    ndof = 3 * natoms - 3
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    axs[0].plot([x[0] for x in opt], [x[2] - opt[0][2] for x in opt], "-", color=c["series"][0])
    axs[0].set_title("optimization (TRUST)", fontsize=9.5)
    axs[0].set_xlabel("cycle"); axs[0].set_ylabel("heat − start (kcal/mol)")
    tt = [r[0] for r in nvt]; T = [2 * r[2] / (ndof * KB) for r in nvt]
    axs[1].plot(tt, T, color=c["series"][1]); axs[1].axhline(300, color=c["axis"], lw=0.8, ls=(0, (3, 3)))
    axs[1].set_title("NVT, 300 K", fontsize=9.5); axs[1].set_xlabel("time (fs)"); axs[1].set_ylabel("temperature (K)")
    energy_panel(axs[2], nve, c, title=False)
    tot = np.array([r[3] for r in nve]) - nve[0][3]
    axs[2].set_title(f"NVE: total within {np.abs(tot).max():.3f} kcal/mol", fontsize=9.5)
    fig.suptitle(f"scripts/mozyme_md_workflow.py on chignolin ({natoms} atoms after ADD-H)", x=0.01, ha="left",
                 fontsize=9.5, color=c["text2"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------------- implicit water
@figure(15)
def ex15(theme, ctx):
    c = L.style(theme)
    tv, tw = ctx.out("sp_chig"), ctx.out("cosmo_chig")
    area = re.findall(r"COSMO AREA\s+=\s+([\d.]+)", tw)[-1]
    vol = re.findall(r"COSMO VOLUME\s+=\s+([\d.]+)", tw)[-1]
    out = ctx.renders / f"ex15-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(CHIG), "m")
        cmd.hide("everything"); cmd.show("sticks"); cmd.set("stick_radius", 0.13)
        L.element_colours(cmd)
        cmd.show("surface"); cmd.set("transparency", 0.55); cmd.set("surface_color", "s3")
        cmd.orient("m")
    L.pymol_render(out, build, 1100, 1000, theme)
    fig, a0, a1 = two()
    L.image_axes(a0, out)
    note(a0, f"COSMO cavity: {float(area):.0f} Å², {float(vol):.0f} Å³ (the surface shown is PyMOL's)", c)
    hv, hw = L.heat(tv), L.heat(tw)
    de = float(re.findall(r"DIELECTRIC ENERGY\s+=\s+(-?[\d.]+)", tw)[-1])
    a1.set_axis_off()
    rows = [("vacuum", f"{hv:.2f} kcal/mol", c["series"][0]), ("water, EPS=78.4", f"{hw:.2f} kcal/mol", c["series"][2]),
            ("water − vacuum", f"{hw - hv:.2f} kcal/mol", c["text"]),
            ("dielectric energy", f"{de:.3f} eV = {de * 23.0605:.1f} kcal/mol", c["text2"])]
    a1.text(0, 0.98, "heat of formation, PM7 MOZYME 1SCF", fontsize=10, fontweight="bold", color=c["text"],
            transform=a1.transAxes, va="top")
    for k, (a, b, col) in enumerate(rows):
        a1.text(0, 0.78 - 0.15 * k, a, fontsize=10, color=col, transform=a1.transAxes)
        a1.text(0.5, 0.78 - 0.15 * k, b, fontsize=10, color=c["text"], transform=a1.transAxes)
    return fig


@figure(16)
def ex16(theme, ctx):
    c = L.style(theme)
    ks = range(-20, 21)
    dx = np.array([k * 0.001 for k in ks])
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for tag, job, col, lab in (("std", "cosmograd_std", c["series"][1], "standard COSMO surface"),
                               ("swig", "cosmograd_swig", c["series"][0], "SWIG")):
        h = np.array([L.heat(ctx.out(f"cosmoscan_{tag}_{k:+03d}")) for k in ks]) - L.heat(ctx.out(f"cosmoscan_{tag}_+00"))
        slope = np.polyfit(dx, h, 2)[1]
        g = L.cart_gradients(ctx.out(job))[49, 0]
        ax.plot(dx, h, "o", ms=3.2, color=col, label=f"{lab}: heats")
        xx = np.array([-0.012, 0.012])
        ax.plot(xx, g * xx, "-", lw=1.6, color=col, alpha=0.9,
                label=f"{lab}: analytic gradient {g:+.3f}, slope of the heats {slope:+.3f}")
    ax.set_xlabel("displacement of atom 50 along x (Å)")
    ax.set_ylabel("heat − heat at 0 (kcal/mol)")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_ylim(-0.006, 0.05)
    ax.set_title("chignolin in water, PM7 MOZYME; lines: tangents from the analytic gradient (kcal/(mol Å))",
                 fontsize=9, fontweight="normal")
    return fig


def opt_panels(ctx, theme, runs, title=None, gnorm_lines=()):
    c = L.style(theme)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for k, (job, lab) in enumerate(runs):
        cy = L.opt_cycles(ctx.out(job))
        x = [r[0] for r in cy]
        a1.plot(x, [r[2] - cy[0][2] for r in cy], color=c["series"][k], label=lab)
        a2.plot(x, [r[1] for r in cy], color=c["series"][k], label=lab)
    a1.set_xlabel("cycle"); a1.set_ylabel("heat − heat at the start (kcal/mol)")
    a2.set_xlabel("cycle"); a2.set_ylabel("gradient norm (kcal/(mol Å))")
    a2.set_yscale("log")
    for g in gnorm_lines:
        a2.axhline(g, color=c["axis"], lw=0.9, ls=(0, (3, 3)))
        a2.text(a2.get_xlim()[1], g, f" GNORM={g:g}", va="bottom", ha="right", fontsize=8, color=c["text2"])
    if len(runs) > 1:
        a1.legend()
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, color=c["text2"])
    fig.tight_layout()
    return fig


@figure(17)
def ex17(theme, ctx):
    return opt_panels(ctx, theme, [("wopt_pm7", "PM7")], "chignolin in water: PM7 MOZYME EPS=78.4 SWIG TRUST, 40 cycles")


@figure(18)
def ex18(theme, ctx):
    return opt_panels(ctx, theme, [("wopt_pm6", "PM6-D3H4")],
                      "chignolin in water: PM6-D3H4 MOZYME EPS=78.4 SWIG TRUST, 40 cycles")


@figure(19)
def ex19(theme, ctx):
    c = L.style(theme)
    d = [("chignolin, 140 atoms", L.heat(ctx.out("sp_chig")), L.heat(ctx.out("cosmo_chig"))),
         ("DNA dodecamer, 758 atoms", L.heat(ctx.out("sp_dna")), L.heat(ctx.out("cosmo_dna")))]
    n_dna = len(L.pdb_atoms(DNA))
    d[1] = (f"DNA dodecamer, {n_dna} atoms", d[1][1], d[1][2])
    out = ctx.renders / f"ex19-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(DNA), "m")
        cmd.hide("everything"); cmd.show("surface"); cmd.set("surface_color", "s3"); cmd.set("transparency", 0.25)
        cmd.orient("m"); cmd.turn("z", 90)
    L.pymol_render(out, build, 900, 1100, theme)
    fig, a0, a1 = two(split=0.38)
    L.image_axes(a0, out)
    y = np.arange(2)[::-1]
    v = [w - vac for _, vac, w in d]
    a1.barh(y, v, 0.5, color=c["series"][2])
    for yy, vv in zip(y, v):
        a1.text(vv - 8, yy, f"{vv:.1f}", ha="right", va="center", fontsize=9, color=c["text2"])
    a1.set_xlim(min(v) * 1.25, 0)
    a1.set_yticks(y, [x[0] for x in d])
    a1.tick_params(axis="y", length=0)
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("heat in water (EPS=78.4) − heat in vacuum (kcal/mol)")
    return fig


# --------------------------------------------------------------------------------------------- optimization
@figure(20)
def ex20(theme, ctx):
    return opt_panels(ctx, theme, [("opt2_trust", "TRUST"), ("opt2_lbfgs", "default (L-BFGS-B)")],
                      "chignolin with hydrogens added again (gradient norm 246 at the start), PM7 MOZYME")


@figure(21)
def ex21(theme, ctx):
    c = L.style(theme)
    cy = L.opt_cycles(ctx.out("opt2_trust"))
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    x = [r[0] for r in cy]; g = [r[1] for r in cy]
    ax.plot(x, g, color=c["series"][0])
    ax.set_yscale("log")
    for k, gn in enumerate((5, 1)):
        stop = next((r for r in cy if r[1] < gn), None)
        ax.axhline(gn, color=c["axis"], lw=0.9, ls=(0, (3, 3)))
        if stop:
            ax.plot(stop[0], stop[1], "o", color=c["series"][1], ms=7, mec=c["surface"], mew=1.2, zorder=4)
            ax.annotate(f"GNORM={gn}: stops at cycle {stop[0]}", (stop[0], stop[1]), xytext=(0, -26),
                        textcoords="offset points", fontsize=8.5, color=c["text"], ha="center",
                        arrowprops=dict(arrowstyle="-", color=c["axis"], lw=0.8))
        else:
            ax.text(x[-1], gn, f"GNORM={gn}: not reached in {x[-1]} cycles ", ha="right", va="bottom", fontsize=8.5,
                    color=c["text2"])
    ax.set_xlabel("cycle")
    ax.set_ylabel("gradient norm (kcal/(mol Å))")
    L.plain_log(ax, "y", [0.5, 1, 5, 10, 50, 100, 200])
    ax.set_title("TRUST on chignolin; the default GNORM is 1", fontsize=9.5, fontweight="normal")
    return fig


@figure(22)
def ex22(theme, ctx):
    c = L.style(theme)
    ks = range(61)
    r = np.array([5.0 + 0.06 * k for k in ks])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for tag, col, lab in (("orig", c["series"][1], "PM7"), ("smooth", c["series"][0], "PM7 with HB-SMOOTH")):
        t = [ctx.out(f"hbscan_{tag}_{k:02d}") for k in ks]
        hb = np.array([float((re.findall(r"H-BOND ENERGY\s+=\s+(-?[\d.]+)", x) or ["0"])[-1]) for x in t])
        h = np.array([L.heat(x) for x in t])
        a1.plot(r, hb, "o-", ms=2.5, lw=1.4, color=col, label=lab)
        a2.plot(r, h - h[-1], "o-", ms=2.5, lw=1.4, color=col, label=lab)
    a1.set_xlabel("O···O distance of the water dimer (Å)"); a1.set_ylabel("H-bond correction (kcal/mol)")
    a2.set_xlabel("O···O distance of the water dimer (Å)"); a2.set_ylabel("heat − heat at 8.6 Å (kcal/mol)")
    a1.legend()
    fig.tight_layout()
    return fig


@figure(23)
def ex23(theme, ctx):
    c = L.style(theme)
    p1 = L.opt_cycles(ctx.out("opt2_restart", "part1.out"))
    p2 = L.opt_cycles(ctx.out("opt2_restart", "part2.out"))
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.plot([r[0] for r in p1], [r[2] for r in p1], "o-", ms=3, color=c["series"][0], label="first job, CYCLES=25")
    ax.plot([r[0] for r in p2], [r[2] for r in p2], "o-", ms=3, color=c["series"][1], label="RESTART, CYCLES=25")
    ax.axvline(p2[0][0] - 0.5, color=c["axis"], lw=0.9, ls=(0, (3, 3)))
    ax.text(p2[0][0], ax.get_ylim()[1], "  restart file read", va="top", fontsize=8.5, color=c["text2"])
    ax.set_xlabel("cycle (the restarted job goes on counting)")
    ax.set_ylabel("heat of formation (kcal/mol)")
    ax.legend(loc="center right")
    return fig


@figure(24)
def ex24(theme, ctx):
    c = L.style(theme)
    start = ctx.work / "opt2_restart" / "prep.pdb"
    final = ctx.work / "opt2_restart" / "final.pdb"
    a0 = L.pdb_atoms(start); a1_ = L.pdb_atoms(final)
    d = np.array([np.linalg.norm(p[5] - q[5]) for p, q in zip(a0, a1_)])
    out = ctx.renders / f"ex24-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(start), "s"); cmd.load(str(final), "f")
        cmd.hide("everything"); cmd.show("sticks")
        cmd.set("stick_radius", 0.12)
        cmd.color("grey60", "s"); cmd.set("stick_transparency", 0.4, "s")
        L.element_colours(cmd, "f")
        cmd.orient("f")
    L.pymol_render(out, build, 1100, 1000, theme)
    fig, a0x, a1x = two()
    L.image_axes(a0x, out)
    note(a0x, "grey: start (hydrogens added); colours: after 50 cycles (final.pdb)", c)
    res = sorted({(p[2], p[3], p[1]) for p in a0}, key=lambda x: x[1])
    rms = [np.sqrt(np.mean([dd ** 2 for p, dd in zip(a0, d) if p[3] == r[1]])) for r in res]
    a1x.bar([r[1] for r in res], rms, 0.65, color=c["series"][0])
    a1x.set_xticks([r[1] for r in res], [f"{r[2].title()}{r[1]}" for r in res], rotation=45, ha="right")
    a1x.grid(axis="x", visible=False)
    a1x.set_ylabel("rms displacement of the atoms (Å)")
    return fig


def flagged_atoms(t):
    """indices (0-based) of the atoms whose coordinates are optimized, from the first geometry print"""
    i = t.find("CARTESIAN COORDINATES")
    j = t.rfind("ATOM   CHEMICAL", 0, i) if i > 0 else -1
    idx = []
    for line in t.splitlines():
        m = re.match(r"\s+(\d+)\s+[A-Z][a-z]?\(.*?\)\s+-?\d+\.\d+\s+\+?1", line)
        if m:
            idx.append(int(m.group(1)) - 1)
    return sorted(set(idx))


@figure(25)
def ex25(theme, ctx):
    c = L.style(theme)
    t = ctx.out("opt_part")
    a0 = L.pdb_atoms(CRN_OPT); a1 = L.pdb_atoms(ctx.work / "opt_part" / "job.pdb")
    d = np.array([np.linalg.norm(p[5] - q[5]) for p, q in zip(a0, a1)])
    sel = np.where(d > 1e-4)[0]
    ids = "+".join(str(i + 1) for i in sel)
    out = ctx.renders / f"ex25-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(CRN_OPT), "m")
        cmd.hide("everything"); cmd.dss("m"); cmd.show("cartoon"); cmd.set("cartoon_color", "mid_t")
        cmd.set("cartoon_transparency", 0.4)
        cmd.show("sticks", f"index {ids}"); cmd.set("stick_radius", 0.17)
        L.element_colours(cmd, f"index {ids}"); cmd.color("s2", f"index {ids} and elem C")
        cmd.orient("m")
    L.pymol_render(out, build, 1200, 1000, theme)
    fig, ax0, ax1 = two()
    L.image_axes(ax0, out)
    note(ax0, f"crambin: the {len(sel)} atoms within 4 Å of residue 46 move (orange carbons); the other "
              f"{len(d) - len(sel)} stay", c)
    cy = L.opt_cycles(t)
    ax1.plot([r[0] for r in cy], [r[2] - cy[0][2] for r in cy], "o-", ms=3, color=c["series"][0])
    ax1.set_xlabel("cycle"); ax1.set_ylabel("heat − heat at the start (kcal/mol)")
    ax1.set_title(f"largest displacement {d.max():.4f} Å (the start is already optimized)", fontsize=9,
                  fontweight="normal")
    return fig


# --------------------------------------------------------------------------------------------- dynamics
def temps(rows, natoms):
    return np.array([2 * r[2] / ((3 * natoms - 3) * KB) for r in rows])


NCHIG = 140


@figure(26)
def ex26(theme, ctx):
    c = L.style(theme)
    r = L.drc_rows(ctx.out("md_nve"))
    t = np.array([x[0] for x in r])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    energy_panel(a1, r, c)
    a2.plot(t, [x[3] - r[0][3] for x in r], color=c["series"][2])
    a2.set_xlabel("time (fs)"); a2.set_ylabel("total energy − at 0 fs (kcal/mol)")
    a2.set_title(f"the total alone, magnified; largest |ERROR| {max(abs(x[4]) for x in r):.3f}", fontsize=8.5,
                 fontweight="normal")
    fig.suptitle("chignolin, NVE from Maxwell–Boltzmann velocities at 300 K (TEMPERATURE=300)", x=0.01, ha="left",
                 fontsize=9.5, color=c["text2"])
    fig.tight_layout()
    return fig


def tplot(ax, ctx, runs, c, natoms=NCHIG, target=None):
    for k, (job, lab) in enumerate(runs):
        r = L.drc_rows(ctx.out(job))
        ax.plot([x[0] for x in r], temps(r, natoms), color=c["series"][k], label=lab)
    if target:
        ax.axhline(target, color=c["axis"], lw=0.9, ls=(0, (3, 3)))
    ax.set_xlabel("time (fs)"); ax.set_ylabel("temperature (K)")


@figure(27)
def ex27(theme, ctx):
    c = L.style(theme)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    tplot(a1, ctx, [("md_bussi100", "BUSSI=100")], c, target=300)
    tl = L.temperature_lines(ctx.out("md_bussi100"))
    a2.plot([x[1] for x in tl], [x[4] for x in tl], "o-", color=c["series"][1])
    a2.axhline(0, color=c["axis"], lw=0.8)
    a2.set_xlabel("time (fs)"); a2.set_ylabel("ENERGY TO BATH (kcal/mol)")
    a2.set_title("negative: the thermostat put energy in", fontsize=9.5, fontweight="normal")
    fig.tight_layout()
    return fig


@figure(28)
def ex28(theme, ctx):
    c = L.style(theme)
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    tplot(ax, ctx, [("md_bussi10", "BUSSI=10"), ("md_bussi100", "BUSSI=100"), ("md_nve", "no thermostat (NVE)")], c,
          target=300)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    return fig


@figure(29)
def ex29(theme, ctx):
    c = L.style(theme)
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ra = L.drc_rows(ctx.out("md_seed42a")); rb = L.drc_rows(ctx.out("md_seed42b")); r3 = L.drc_rows(ctx.out("md_seed43"))
    ax.plot([x[0] for x in ra], temps(ra, NCHIG), color=c["series"][0], lw=3.2, alpha=0.45, label="SEED=42, run 1")
    ax.plot([x[0] for x in rb], temps(rb, NCHIG), color=c["series"][0], lw=1.2, ls=(0, (4, 2)), label="SEED=42, run 2")
    ax.plot([x[0] for x in r3], temps(r3, NCHIG), color=c["series"][1], label="SEED=43")
    same = all(abs(a[3] - b[3]) < 1e-9 for a, b in zip(ra, rb))
    ax.set_xlabel("time (fs)"); ax.set_ylabel("temperature (K)")
    ax.legend(loc="lower right", ncol=3)
    ax.set_title("CPU: the two SEED=42 runs are " + ("identical" if same else "not identical"), fontsize=9.5,
                 fontweight="normal")
    return fig


@figure(30)
def ex30(theme, ctx):
    c = L.style(theme)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    runs = [("md_rep1", "SEED=1"), ("md_rep2", "SEED=2"), ("md_rep3", "SEED=3")]
    tplot(a1, ctx, runs, c, target=300)
    for k, (job, lab) in enumerate(runs):
        r = L.drc_rows(ctx.out(job))
        a2.plot([x[0] for x in r], [x[1] - r[0][1] for x in r], color=c["series"][k], label=lab)
    a2.set_xlabel("time (fs)"); a2.set_ylabel("potential energy − at 0 fs (kcal/mol)")
    a2.legend()
    fig.tight_layout()
    return fig


@figure(31)
def ex31(theme, ctx):
    c = L.style(theme)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    tplot(a1, ctx, [("md_water", "310 K, BUSSI=100")], c, target=310)
    r = L.drc_rows(ctx.out("md_water"))
    a2.plot([x[0] for x in r], [x[4] for x in r], color=c["series"][2])
    a2.set_xlabel("time (fs)"); a2.set_ylabel("integration ERROR (kcal/mol)")
    fig.suptitle("chignolin in water (EPS=78.4 SWIG HB-SMOOTH), 30 fs", x=0.01, ha="left", fontsize=9.5,
                 color=c["text2"])
    fig.tight_layout()
    return fig


@figure(32)
def ex32(theme, ctx):
    c = L.style(theme)
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    t0 = 0.0
    for k, temp in enumerate((100, 200, 300)):
        r = L.drc_rows(L.text(ctx.work / "md_heat" / f"stage{k}.out"))
        t = np.array([x[0] for x in r]) + t0
        ax.plot(t, temps(r, NCHIG), color=c["series"][k], label=f"TEMPERATURE={temp}")
        ax.hlines(temp, t[0], t[-1], color=c["series"][k], lw=0.9, ls=(0, (3, 3)))
        t0 = t[-1] + 0.5
    ax.set_xlabel("time (fs), stages one after the other"); ax.set_ylabel("temperature (K)")
    ax.legend(loc="upper left", ncol=3)
    return fig


@figure(33)
def ex33(theme, ctx):
    c = L.style(theme)
    r = L.drc_rows(ctx.out("md_kinetic"))
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.plot([x[0] for x in r], [x[2] for x in r], color=c["series"][1], label="kinetic")
    ax.plot([x[0] for x in r], [x[3] - r[0][3] for x in r], color=c["series"][2], label="total − total at 0 fs")
    ax.set_xlabel("time (fs)"); ax.set_ylabel("energy (kcal/mol)")
    ax.legend()
    ax.set_title("classic DRC with KINETIC=20: the atoms start at rest", fontsize=9.5, fontweight="normal")
    return fig


@figure(34)
def ex34(theme, ctx):
    c = L.style(theme)
    t = ctx.out("md_crambin")
    r = L.drc_rows(t)
    png = ctx.renders / f"ex34-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(CRN_OPT), "m")
        cmd.hide("everything"); cmd.dss("m"); cmd.show("cartoon"); cmd.color("s1", "m")
        cmd.show("sticks", "not name N+C+O"); cmd.set("stick_radius", 0.1)
        L.element_colours(cmd, "m and rep sticks and not elem C")
        cmd.color("carbon_t", "elem C and rep sticks")
        cmd.orient("m")
    L.pymol_render(png, build, 1100, 1000, theme)
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    secs = L.job_seconds(t)
    note(a0, f"crambin, 642 atoms; {r[-1][0]:.1f} fs in {secs / 60:.0f} min on one CPU core", c)
    a1.plot([x[0] for x in r], [x[3] - r[0][3] for x in r], "o-", color=c["series"][2], label="total")
    a1.plot([x[0] for x in r], [x[2] - r[0][2] for x in r], "o-", color=c["series"][1], label="kinetic")
    a1.set_xlabel("time (fs)"); a1.set_ylabel("change from 0 fs (kcal/mol)")
    a1.legend()
    return fig


# --------------------------------------------------------------------------------------------- vibrations
def stick_spectrum(ax, f, d, c, col):
    f = np.array(f); d = np.array(d)
    ax.vlines(f, 0, d, color=col, lw=1)
    ax.set_xlabel("frequency (cm⁻¹)"); ax.set_ylabel("transition dipole (D)")


@figure(35)
def ex35(theme, ctx):
    c = L.style(theme)
    t = ctx.out("vib_force")
    f, d = L.frequencies(t)
    m = re.search(r"298\.00\s+VIB\..*?TOT\.\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t, re.S)
    zpe = re.findall(r"ZERO POINT ENERGY\s+([\d.]+)", t)
    fig, ax = plt.subplots(figsize=(8.0, 3.3))
    stick_spectrum(ax, f, d, c, c["series"][0])
    nimag = sum(1 for x in f if x < 0)
    s = f"{len(f)} vibrations"
    if nimag:
        s += f", {nimag} imaginary ({min(f):.1f} cm⁻¹)"
    if m:
        s += f"\nTHERMO(298): enthalpy {float(m.group(2)) / 1000:.2f} kcal/mol, heat capacity {float(m.group(3)):.1f}, " \
             f"entropy {float(m.group(4)):.1f} cal/(K mol)"
    if zpe:
        s += f"; zero-point energy {float(zpe[-1]):.1f} kcal/mol"
    ax.set_title(s, fontsize=9, fontweight="normal")
    return fig


@figure(36)
def ex36(theme, ctx):
    c = L.style(theme)
    fs, _ = L.frequencies(ctx.out("vib_force"))
    fp, _ = L.frequencies(L.text(ctx.work / "vib_parallel" / "pf" / "final" / "force.out"))
    n = min(len(fs), len(fp))
    fs, fp = np.array(fs[:n]), np.array(fp[:n])
    ts = json_seconds(ctx, "vib_force"); tp = json_seconds(ctx, "vib_parallel")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    a1.plot(fs, fp - fs, "o", ms=3, color=c["series"][0])
    a1.axhline(0, color=c["axis"], lw=0.8)
    a1.set_xlabel("frequency, one MOPAC job (cm⁻¹)"); a1.set_ylabel("parallel − one job (cm⁻¹)")
    a1.set_title(f"largest difference {np.max(np.abs(fp - fs)):.3f} cm⁻¹", fontsize=9.5, fontweight="normal")
    a2.bar([0, 1], [ts / 60, tp / 60], 0.55, color=[c["series"][1], c["series"][0]])
    a2.set_xticks([0, 1], ["one job", "4 workers"])
    for x, v in zip([0, 1], [ts / 60, tp / 60]):
        a2.text(x, v, f"{v:.0f} min", ha="center", va="bottom", fontsize=9, color=c["text2"])
    a2.set_ylabel("wall time on this Mac (min)")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def json_seconds(ctx, job):
    import json
    return json.loads((ctx.work / job / "done.json").read_text())["seconds"]


@figure(37)
def ex37(theme, ctx):
    c = L.style(theme)
    a0 = L.pdb_atoms(CRN_OPT); a1 = L.pdb_atoms(ctx.work / "opt_part" / "job.pdb")
    d = np.array([np.linalg.norm(p[5] - q[5]) for p, q in zip(a0, a1)])
    sel = np.where(d > 1e-3)[0]
    out = ctx.renders / f"ex37-{theme}.png"
    ids = "+".join(str(i + 1) for i in sel)

    def build(cmd, cc):
        cmd.load(str(CRN_OPT), "m")
        cmd.hide("everything"); cmd.dss("m"); cmd.show("cartoon"); cmd.set("cartoon_color", "mid_t")
        cmd.set("cartoon_transparency", 0.4)
        cmd.show("sticks", f"index {ids}"); cmd.set("stick_radius", 0.16)
        L.element_colours(cmd, f"index {ids}")
        cmd.color("s2", f"index {ids} and elem C")
        cmd.show("sticks", "resi 46"); cmd.color("s7", "resi 46 and elem C")
        cmd.orient("m")
    L.pymol_render(out, build, 1300, 1000, theme)
    fig = plt.figure(figsize=(7.6, 3.6))
    ax = fig.add_axes([0, 0, 0.7, 1]); L.image_axes(ax, out)
    tx = fig.add_axes([0.68, 0.1, 0.3, 0.8]); tx.set_axis_off()
    tx.text(0, 0.8, f"{len(sel)} atoms selected\nby OPT(\"A46\"=4)", fontsize=10, color=c["text"], fontweight="bold",
            transform=tx.transAxes)
    tx.text(0, 0.55, f"FORCETS: Hessian of\n{3 * len(sel)} coordinates\n(the whole protein: {3 * len(a0)})",
            fontsize=9.5, color=c["text2"], transform=tx.transAxes)
    tx.text(0, 0.3, "orange carbons: selected atoms\nviolet carbons: Asn46", fontsize=8.5, color=c["text2"],
            transform=tx.transAxes)
    return fig


# --------------------------------------------------------------------------------------------- solids
def cell_pdb(ctx, name, reps=(1, 1, 1)):
    """PDB of a repository cell (optionally repeated), with CRYST1 from the translation vectors"""
    atoms, tv = L.mop_cell(CELLS / f"{name}.mop")
    out = ctx.renders / f"cell_{name}_{reps[0]}{reps[1]}{reps[2]}.pdb"
    allat, central = [], []
    tv3 = list(tv) + [np.array([0, 0, 30.0])] * (3 - len(tv))
    rng = lambda n: range(-(n // 2), n - n // 2)
    for i in rng(reps[0]):
        for j in rng(reps[1]):
            for k in rng(reps[2] if len(tv) == 3 else 1):
                sh = i * tv3[0] + j * tv3[1] + k * tv3[2]
                for el, x in atoms:
                    allat.append((el, x + sh))
                    central.append(1.0 if (i, j, k) == (0, 0, 0) else 0.0)
    L.write_pdb(allat, out, bfac=central)
    a, b, cc = (np.linalg.norm(v) for v in tv3)
    ang = lambda u, v: np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
    cryst = (f"CRYST1{a:9.3f}{b:9.3f}{cc:9.3f}{ang(tv3[1], tv3[2]):7.2f}{ang(tv3[0], tv3[2]):7.2f}"
             f"{ang(tv3[0], tv3[1]):7.2f} P 1           1\n")
    out.write_text(cryst + out.read_text())
    return out, atoms, tv


def render_cell(ctx, theme, name, reps=(1, 1, 1), turn=None, show_cell=False, spheres=False):
    pdb, atoms, tv = cell_pdb(ctx, name, reps)
    out = ctx.renders / f"{name}-{reps}-{theme}.png".replace(" ", "")

    def build(cmd, cc):
        cmd.load(str(pdb), "cl")
        cmd.hide("everything")
        cmd.show("sticks"); cmd.set("stick_radius", 0.13)
        if spheres:
            cmd.show("spheres"); cmd.set("sphere_scale", 0.28)
        L.element_colours(cmd)
        cmd.color("s1", "elem Si"); cmd.color("s6", "elem Ti"); cmd.color("s3", "elem Zn"); cmd.color("s5", "elem Mg")
        if reps != (1, 1, 1):
            cmd.set("stick_transparency", 0.65, "b < 0.5"); cmd.set("sphere_transparency", 0.65, "b < 0.5")
        if show_cell and len(tv) == 3:
            cmd.show("cell"); cmd.set("cgo_line_width", 1.5)
            cmd.set_color("cellc", L._rgb(cc["text2"])); cmd.set("cell_color", "cellc")
        cmd.orient("cl")
    L.pymol_render(out, build, 1100, 1000, theme, turn)
    return out


def dos(ax, e, c, homo=None, lumo=None, col=None, width=0.15, label=None):
    grid = np.linspace(e.min() - 1, min(e.max(), (lumo or e.max()) + 12) + 1, 1200)
    g = np.exp(-((grid[:, None] - e[None, :]) / width) ** 2).sum(1)
    ax.fill_between(grid, g, color=col or c["series"][0], alpha=0.35, lw=0)
    ax.plot(grid, g, color=col or c["series"][0], lw=1.2, label=label)
    ax.set_xlabel("orbital energy (eV)"); ax.set_ylabel("density of states (arb.)")
    ax.set_yticks([])


@figure(38)
def ex38(theme, ctx):
    c = L.style(theme)
    t = ctx.out("sol_si")
    png = render_cell(ctx, theme, "silicon", turn=[("x", -20), ("y", 25)], spheres=True)
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    note(a0, f"silicon, 64 atoms; heat {L.heat(t):.2f} kcal/mol", c)
    e = L.eigenvalues(t); h, l = L.homo_lumo(t)
    dos(a1, e[(e > h - 14) & (e < l + 8)], c)
    a1.axvline(h, color=c["series"][1], lw=1); a1.axvline(l, color=c["series"][1], lw=1, ls=(0, (3, 3)))
    a1.text(h, a1.get_ylim()[1] * 0.95, "HOMO ", ha="right", va="top", fontsize=8.5, color=c["series"][1])
    a1.text(l, a1.get_ylim()[1] * 0.95, " LUMO", ha="left", va="top", fontsize=8.5, color=c["series"][1])
    a1.set_title(f"gap {l - h:.2f} eV (orbital energies, broadened 0.15 eV)", fontsize=9.5, fontweight="normal")
    return fig


@figure(40)
def ex40(theme, ctx):
    c = L.style(theme)
    t = ctx.out("sol_graphene")
    png = render_cell(ctx, theme, "graphene", reps=(3, 3, 1), show_cell=False)
    tr = L.scf_trace(t)
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    note(a0, "graphene, 72 atoms per cell (solid), with its periodic images", c)
    it = [x[0] for x in tr if x[2] != 0]; de = [abs(x[2]) for x in tr if x[2] != 0]
    a1.plot(it, de, lw=0.5, color=c["series"][0])
    a1.set_yscale("log")
    a1.set_xlabel("SCF iteration"); a1.set_ylabel("|energy change| per iteration (kcal/mol)")
    a1.set_title(f"{len(tr)} SCF iterations; heat {L.heat(t):.3f} kcal/mol", fontsize=9.5, fontweight="normal")
    return fig


@figure(41)
def ex41(theme, ctx):
    c = L.style(theme)
    t = ctx.out("sol_graphene_smear")
    ef = float(re.findall(r"FERMI LEVEL =\s+(-?[\d.]+)", t)[-1])
    mts = float(re.findall(r"-TS =\s+(-?[\d.]+)", t)[-1])
    e = L.eigenvalues(t)
    kt = 0.1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    near = e[(e > ef - 3) & (e < ef + 3)]
    x = np.linspace(ef - 3, ef + 3, 400)
    a1.plot(x, 2 / (1 + np.exp((x - ef) / kt)), color=c["axis"], lw=1, label="2 / (1 + exp((ε − μ)/kT))")
    a1.plot(near, 2 / (1 + np.exp((near - ef) / kt)), "o", ms=4, color=c["series"][0], label="levels of the cell")
    a1.axvline(ef, color=c["series"][1], lw=0.9, ls=(0, (3, 3)))
    a1.text(ef, 1.05, " Fermi level", color=c["series"][1], fontsize=8.5)
    a1.set_xlabel("orbital energy (eV)"); a1.set_ylabel("occupation"); a1.legend(loc="lower left", fontsize=8)
    a1.set_title(f"kT = 0.1 eV; −TS = {mts:.6f} kcal/mol", fontsize=9.5, fontweight="normal")
    n_default = len(L.scf_trace(ctx.out("sol_graphene"))); n_smear = len(L.scf_trace(t))
    a2.bar([0, 1], [n_default, n_smear], 0.55, color=[c["series"][1], c["series"][0]])
    a2.set_xticks([0, 1], ["default SCF", "SMEAR=0.1"])
    for xx, v in zip([0, 1], [n_default, n_smear]):
        a2.text(xx, v, f"{v}", ha="center", va="bottom", fontsize=9, color=c["text2"])
    a2.set_ylabel("SCF iterations, graphene 72 atoms")
    a2.grid(axis="x", visible=False)
    hd, hs = L.heat(ctx.out("sol_graphene")), L.heat(t)
    a2.set_title(f"heats {hd:.2f} / {hs:.2f} kcal/mol: SMEAR ended\nin another SCF state, {hs - hd:.1f} kcal/mol higher",
                 fontsize=9, fontweight="normal")
    fig.tight_layout()
    return fig


def tv_gradients(t):
    g = re.findall(r"\s+\d+\s+\d+\s+[A-Z][a-z]?\s+CARTESIAN [XYZ]\s+-?[\d.]+\s+(-?[\d.]+)\s+KCAL/ANGSTROM", t)
    return np.array([float(x) for x in g])


@figure(42)
def ex42(theme, ctx):
    c = L.style(theme)
    gd = tv_gradients(ctx.out("sol_diamond_grad")); gp = tv_gradients(ctx.out("sol_diamond_precise"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    a1.plot(gp, gd, "o", ms=3, color=c["series"][0])
    lim = max(abs(gp).max(), abs(gd).max()) * 1.1
    a1.plot([-lim, lim], [-lim, lim], color=c["axis"], lw=0.8)
    a1.set_xlabel("gradient with PRECISE (kcal/(mol Å))"); a1.set_ylabel("default gradient (kcal/(mol Å))")
    a2.hist(gd - gp, 30, color=c["series"][0])
    a2.set_xlabel("default − PRECISE (kcal/(mol Å))"); a2.set_ylabel("components")
    a2.set_title(f"largest difference {np.max(np.abs(gd - gp)):.3f}", fontsize=9.5, fontweight="normal")
    fig.suptitle("diamond, 64 atoms, every gradient component", x=0.01, ha="left", fontsize=9.5, color=c["text2"])
    fig.tight_layout()
    return fig


@figure(43)
def ex43(theme, ctx):
    n = len(L.opt_cycles(ctx.out("sol_rutile_opt")))
    return opt_panels(ctx, theme, [("sol_rutile_opt", "rutile")], f"rutile, 72 atoms relaxed in a fixed cell, PM7 "
                      f"PRECISE: converged after {n} cycles (gradient norm below GNORM = 1)")


@figure(44)
def ex44(theme, ctx):
    c = L.style(theme)
    t = ctx.out("sol_diamond_force")
    f = np.array([x for x in L.freq_table(t) if abs(x) > 1.0])
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.hist(f, bins=np.arange(0, f.max() + 25, 25), color=c["series"][0])
    ax.set_xlabel("Γ-point frequency (cm⁻¹)"); ax.set_ylabel("modes per 25 cm⁻¹")
    ax.set_title(f"diamond, 64-atom cell: {len(f)} modes; FORCE SCFCRT=1.D-10, "
                 f"{json_seconds(ctx, 'sol_diamond_force') / 60:.0f} min on 4 CPU threads", fontsize=9.5, fontweight="normal")
    return fig


@figure(45)
def ex45(theme, ctx):
    c = L.style(theme)
    r = L.drc_rows(ctx.out("sol_diamond_drc"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    a1.plot([x[0] for x in r], temps(r, 64), color=c["series"][1]); a1.axhline(300, color=c["axis"], lw=0.9, ls=(0, (3, 3)))
    a1.set_xlabel("time (fs)"); a1.set_ylabel("temperature (K)")
    energy_panel(a2, r, c)
    fig.suptitle("diamond, 64 atoms, NVE at 300 K in a fixed cell", x=0.01, ha="left", fontsize=9.5, color=c["text2"])
    fig.tight_layout()
    return fig


@figure(46)
def ex46(theme, ctx):
    c = L.style(theme)
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for k, (job, lab) in enumerate((("sol_si_default", "default: 20 eV, damped start"),
                                    ("sol_si_shift80", "SHIFT=80 (MOPAC 23.1.2)"))):
        t = ctx.out(job); tr = L.scf_trace(t)
        it = [x[0] for x in tr if x[2] != 0]; de = [abs(x[2]) for x in tr if x[2] != 0]
        ax.plot(it, de, "o-", ms=2.5, lw=1.3, color=c["series"][k], label=f"{lab}: {len(tr)} iterations, "
                f"{L.heat(t):.5f} kcal/mol")
    ax.set_yscale("log")
    ax.set_xlabel("SCF iteration"); ax.set_ylabel("|energy change| (kcal/mol)")
    ax.legend(fontsize=8)
    ax.set_title("silicon, 64 atoms", fontsize=9.5, fontweight="normal")
    return fig


@figure(47)
def ex47(theme, ctx):
    c = L.style(theme)
    png = render_cell(ctx, theme, "urea", turn=[("x", -15), ("y", 20)], spheres=True)
    hm, hc = L.heat(ctx.out("sol_urea_moz")), L.heat(ctx.out("sol_urea_conv"))
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    note(a0, "urea crystal, 192 atoms per cell", c)
    a1.set_axis_off()
    rows = [("MOZYME", hm, L.job_seconds(ctx.out("sol_urea_moz"))), ("conventional SCF", hc,
                                                                       L.job_seconds(ctx.out("sol_urea_conv")))]
    a1.text(0, 0.95, "PM7 1SCF, one CPU core", fontsize=10, fontweight="bold", color=c["text"], transform=a1.transAxes)
    for k, (n, h, s) in enumerate(rows):
        a1.text(0, 0.72 - 0.17 * k, n, fontsize=10, color=c["series"][k], transform=a1.transAxes)
        a1.text(0.45, 0.72 - 0.17 * k, f"{h:.3f} kcal/mol\n{s:.1f} s", fontsize=9.5, color=c["text"],
                transform=a1.transAxes, va="center")
    a1.text(0, 0.3, f"difference {hm - hc:+.3f} kcal/mol ({(hm - hc) / abs(hc):+.1e} relative)", fontsize=9.5,
            color=c["text2"], transform=a1.transAxes)
    return fig


@figure(48)
def ex48(theme, ctx):
    c = L.style(theme)
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for k, (v, lab) in enumerate((("0", "MOPAC_PDIAG=0: full diagonalization every iteration"),
                                  ("1", "MOPAC_PDIAG=1: pseudo-diagonalization (the GPU default)"))):
        t = ctx.out(f"sol_quartz_pdiag{v}"); tr = L.scf_trace(t)
        it = [x[0] for x in tr if x[2] != 0]; de = [abs(x[2]) for x in tr if x[2] != 0]
        ax.plot(it, de, "o-", ms=2.5, lw=1.3, color=c["series"][k], label=f"{lab}\n{len(tr)} iterations, "
                f"{L.heat(t):.5f} kcal/mol")
    ax.set_yscale("log")
    ax.set_xlabel("SCF iteration"); ax.set_ylabel("|energy change| (kcal/mol)")
    ax.legend(fontsize=8)
    ax.set_title("α-quartz cell of the repository (CPU run)", fontsize=9.5, fontweight="normal")
    return fig


# --------------------------------------------------------------------------------------------- conventional SCF
FLAKE = REPO / "benchmarks/publication_inputs/mop/material_graphene_nanoflake_c192h38.mop"


def flake_atoms():
    out = []
    for l in FLAKE.read_text().splitlines()[3:]:
        f = l.split()
        if len(f) >= 4:
            out.append((f[0], np.array([float(f[1]), float(f[2]), float(f[3])])))
    return out


@figure(49)
def ex49(theme, ctx):
    c = L.style(theme)
    t = ctx.out("conv_flake")
    tr = L.scf_trace(t)
    at = flake_atoms()
    fig = plt.figure(figsize=(8.6, 3.4))
    a0 = fig.add_axes([0.0, 0.05, 0.4, 0.9]); a0.set_aspect("equal"); a0.set_axis_off()
    xy = np.array([x[1][:2] for x in at])
    for i in range(len(at)):
        for j in range(i + 1, len(at)):
            dd = np.linalg.norm(at[i][1] - at[j][1])
            if dd < (1.2 if "H" in (at[i][0], at[j][0]) else 1.6):
                a0.plot(*xy[[i, j]].T, color=c["carbon"], lw=0.9, zorder=1)
    a0.scatter(*xy[[k for k, a in enumerate(at) if a[0] == "C"]].T, s=5, color=c["text2"], zorder=2)
    a0.text(0.0, 0.0, "C₁₉₂H₃₈, 230 atoms", transform=a0.transAxes, fontsize=8, color=c["text2"])
    a1 = fig.add_axes([0.52, 0.16, 0.46, 0.74])
    it = [x[0] for x in tr if x[2] != 0]; de = [abs(x[2]) for x in tr if x[2] != 0]
    a1.plot(it, de, lw=1.2, color=c["series"][0])
    a1.set_yscale("log")
    a1.set_xlabel("SCF iteration"); a1.set_ylabel("|energy change| (kcal/mol)")
    a1.set_title(f"with PULAY: {len(tr)} iterations, {L.heat(t):.2f} kcal/mol", fontsize=9.5, fontweight="normal")
    return fig


@figure(50)
def ex50(theme, ctx):
    c = L.style(theme)
    t = ctx.out("conv_uhf")
    i = t.rfind("ATOMIC ORBITAL SPIN POPULATIONS")
    sp = []
    for line in t[i:].splitlines()[4:]:
        m = re.match(r"\s+(\d+)\s+[A-Z][a-z]?\s+(-?\d+\.\d+)", line)
        if not m:
            if sp:
                break
            continue
        sp.append(float(m.group(2)))
    sp = np.array(sp)
    vm = float(np.abs(sp).max())
    png, _ = render_values(ctx, theme, CHIG, sp, "ex50", vmax=vm)
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    colorbar(fig, a0, c, -vm, vm, "spin population")
    res = {}
    for a, s in zip(L.pdb_atoms(CHIG), sp):
        res[(a[1], a[3])] = res.get((a[1], a[3]), 0.0) + s
    keys = sorted(res, key=lambda k: k[1])
    y = np.arange(len(keys))[::-1]
    a1.barh(y, [res[k] for k in keys], 0.65, color=c["series"][0])
    a1.set_yticks(y, [f"{k[0].title()}{k[1]}" for k in keys])
    a1.tick_params(axis="y", length=0); a1.grid(axis="y", visible=False)
    a1.set_xlabel("spin population of the residue")
    a1.set_title(f"chignolin cation, UHF doublet: {L.heat(t):.2f} kcal/mol", fontsize=9.5, fontweight="normal")
    return fig


@figure(51)
def ex51(theme, ctx):
    import json
    c = L.style(theme)
    ga = tv_gradients(ctx.out("conv_he")); gn = tv_gradients(ctx.out("conv_he_num"))
    fd = {int(k): v for k, v in json.loads((ctx.work / "he_fd" / "fd.json").read_text()).items()}
    ks = sorted(fd)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    x = np.array([fd[k] for k in ks])
    a1.plot(x, [ga[k] for k in ks], "o", ms=6, color=c["series"][0], label="analytic (default, Z-vector)")
    a1.plot(x, [gn[k] for k in ks], "s", ms=5, color=c["series"][1], label="NOANCI")
    lim = np.abs(np.concatenate([x, ga[ks], gn[ks]])).max() * 1.08
    a1.plot([-lim, lim], [-lim, lim], color=c["axis"], lw=0.8)
    a1.set_xlabel("central difference of the heat, ±0.005 Å (kcal/(mol Å))")
    a1.set_ylabel("gradient printed by MOPAC (kcal/(mol Å))")
    a1.legend(fontsize=8)
    ea = np.abs(ga[ks] - x).max(); en = np.abs(gn[ks] - x).max()
    na = int(np.sum(np.abs(ga[ks] - x) < 0.4))
    a1.set_title(f"{len(ks)} coordinates: analytic within 0.4 at {na}, largest deviation {ea:.2f};\n"
                 f"NOANCI largest deviation {en:.1f}", fontsize=9, fontweight="normal")
    ta, tn = json_seconds(ctx, "conv_he"), json_seconds(ctx, "conv_he_num")
    a2.bar([0, 1], [ta, tn], 0.55, color=[c["series"][0], c["series"][1]])
    a2.set_xticks([0, 1], ["analytic (default)", "NOANCI"])
    for xx, v in zip([0, 1], [ta, tn]):
        a2.text(xx, v, f"{v:.0f} s", ha="center", va="bottom", fontsize=9, color=c["text2"])
    a2.set_ylabel("job time, one CPU core (s)"); a2.grid(axis="x", visible=False)
    fig.suptitle("chignolin cation, RHF doublet (half-electron), 1SCF GRADIENTS", x=0.01, ha="left", fontsize=9.5,
                 color=c["text2"])
    fig.tight_layout()
    return fig


@figure(52)
def ex52(theme, ctx):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    c = L.style(theme)
    bo = L.bond_orders(ctx.out("conv_flake"))
    at = flake_atoms()
    xy = np.array([x[1][:2] for x in at])
    segs, vals = [], []
    for (i, j), o in bo.items():
        if at[i - 1][0] == "C" and at[j - 1][0] == "C" and o > 0.6:
            segs.append(xy[[i - 1, j - 1]]); vals.append(o)
    fig = plt.figure(figsize=(7.6, 4.0))
    ax = fig.add_axes([0.0, 0.0, 0.86, 1.0]); ax.set_aspect("equal"); ax.set_axis_off()
    cm = LinearSegmentedColormap.from_list("s", c["seq"])
    norm = Normalize(min(vals), max(vals))
    ax.add_collection(LineCollection(segs, colors=cm(norm(vals)), linewidths=3.2, capstyle="round"))
    ax.autoscale()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    cax = fig.add_axes([0.86, 0.2, 0.018, 0.6])
    cb = fig.colorbar(sm, cax=cax); cb.set_label("C–C bond order (BONDS)", color=c["text2"]); cb.outline.set_visible(False)
    cb.ax.tick_params(colors=c["text2"])
    return fig


# --------------------------------------------------------------------------------------------- QM/MM
def qmmm_render(ctx, theme, job, qm_sel, name):
    out = ctx.renders / f"{name}-{theme}.png"

    def build(cmd, cc):
        cmd.load(str(CHIG), "m")
        cmd.hide("everything")
        cmd.show("sticks"); cmd.set("stick_radius", 0.13)
        L.element_colours(cmd, "m")
        cmd.color("carbon_t", "elem C")
        cmd.set_bond("stick_radius", 0.24, f"m and ({qm_sel})")
        cmd.color("s2", f"m and ({qm_sel}) and elem C")
        cmd.orient("m")
    L.pymol_render(out, build, 1100, 1000, theme)
    traj = L.text(ctx.work / job / "qmmm_traj.pdb")
    first = traj.split("ENDMDL")[0]
    nwat = len({l[17:26] for l in first.splitlines() if l.startswith(("ATOM", "HETATM")) and l[17:20] == "HOH"})
    return out, nwat


def nve_panel(ax, ctx, job, c):
    e = np.loadtxt(ctx.work / job / "nve_energy.txt")
    t = np.arange(len(e)) * 0.5
    ax.plot(t, e - e[0], color=c["series"][2], lw=1.2)
    ax.set_xlabel("NVE time (fs), 0.5-fs steps"); ax.set_ylabel("total energy − start (kJ/mol)")
    log = L.text(ctx.work / job / "run.log")
    m = re.search(r"QM/MM NVT: .*?= ([\d.]+) s/step", log)
    return m.group(1) if m else "?"


@figure(53)
def ex53(theme, ctx):
    c = L.style(theme)
    png, nwat = qmmm_render(ctx, theme, "qmmm_sc", "resi 2 and not name N+C+O+CA+H+HA", "ex53")
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    s = nve_panel(a1, ctx, "qmmm_sc", c)
    note(a0, f"QM region: the Tyr2 side chain (orange carbons, thick);\nMM: the rest and {nwat} TIP3P waters (not shown)", c)
    a1.set_title(f"MOPAC and OpenMM on the CPU: {s} s per NVT step", fontsize=9.5, fontweight="normal")
    return fig


@figure(54)
def ex54(theme, ctx):
    c = L.style(theme)
    png, nwat = qmmm_render(ctx, theme, "qmmm_pep", "all", "ex54")
    fig, a0, a1 = two()
    L.image_axes(a0, png)
    s = nve_panel(a1, ctx, "qmmm_pep", c)
    note(a0, f"QM region: the whole peptide (138 atoms);\nMM: {nwat} TIP3P waters (not shown); --tolerance 0.001", c)
    a1.set_title(f"on the CPU: {s} s per step (an A100: 0.031 s at 0.01)", fontsize=9.5, fontweight="normal")
    return fig


ENZ = "enzyme_scan_l4"          # results of colab/enzyme_scan_colab.ipynb (an L4 GPU), unzipped in <ext>


def scan_rows(ctx, seed):
    import csv
    with open(ctx.ext / ENZ / f"seed_{seed}" / "scan.csv") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k != "window":
                r[k] = float(v)
    return rows


def scan_grid(rows):
    """{(row, column): window} of the grid windows g<i>_<j>"""
    return {tuple(int(v) for v in r["window"][1:].split("_")): r for r in rows if r["window"].startswith("g")}


def lowest_path(g, start, goal):
    """the path through neighbouring grid points (diagonals too) whose highest energy is lowest"""
    import heapq
    best, prev, pq = {start: g[start]["e"]}, {}, [(g[start]["e"], start)]
    while pq:
        h, p = heapq.heappop(pq)
        if h > best[p]:
            continue
        for q in ((p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b):
            if q in g and max(h, g[q]["e"]) < best.get(q, np.inf):
                best[q], prev[q] = max(h, g[q]["e"]), p
                heapq.heappush(pq, (best[q], q))
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def scan_end(g):
    """the lowest point of the last row with the proton on His57"""
    last = max(i for i, _ in g)
    return min((k for k in g if k[0] == last and g[k]["d2"] > 0), key=lambda k: g[k]["e"])


@figure(59)
def ex59(theme, ctx):
    from matplotlib.colors import LinearSegmentedColormap
    c = L.style(theme)
    kcal = 1 / 4.184
    runs = {s: scan_rows(ctx, s) for s in (1, 2, 3)}
    rows = runs[1]
    e0 = rows[0]["e"]
    g = scan_grid(rows)
    end = scan_end(g)
    frames = ctx.ext / ENZ / "seed_1" / "scan_frames.pdb"
    picks = [(rows[0], "reactant", "A"), (g[end], "end", "B")]
    pngs = []
    for row, name, _ in picks:
        state = rows.index(row) + 1           # one model per window, in order
        out = ctx.renders / f"ex59-{name}-{theme}.png"

        def build(cmd, cc, state=state):
            cmd.load(str(frames), "m0")
            cmd.create("m", "m0", state, 1); cmd.delete("m0")
            cmd.hide("everything"); cmd.set("depth_cue", 0)
            # the catalytic triad's side chains and the scissile peptide of BPTI (Lys15 C=O, Ala16 N)
            act = ("(chain E and resi 57+102+195 and not name N+C+O+H+HA) or "
                   "(chain I and ((resi 15 and name N+H+CA+HA+CB+C+O) or (resi 16 and name N+H+CA)))")
            # the bonds that form and break: drawn as bonds when short, as dashes when not
            hg, og = "chain E and resi 195 and name HG", "chain E and resi 195 and name OG"
            ne2, cc = "chain E and resi 57 and name NE2", "chain I and resi 15 and name C"
            for k, (s1, s2, cut) in enumerate(((og, cc, 1.7), (hg, ne2, 1.3), (hg, og, 1.3))):
                cmd.unbond(s1, s2)
                if cmd.get_distance(s1, s2) < cut:
                    cmd.bond(s1, s2)
                elif k < 2:
                    cmd.distance(f"d{k}", s1, s2)
            cmd.show("sticks", f"({act}) and not (elem H and neighbor elem C)"); cmd.set("stick_radius", 0.17)
            L.element_colours(cmd, f"({act})"); cmd.color("carbon_t", f"({act}) and elem C")
            cmd.color("s2", "chain I and resi 15+16 and elem C")
            cmd.show("spheres", hg); cmd.set("sphere_scale", 0.2); cmd.color("s3", hg)
            cmd.hide("labels"); cmd.set("dash_color", "s7"); cmd.set("dash_width", 3); cmd.set("dash_gap", 0.25)
            cmd.orient("(chain E and resi 57+195 and not name N+C+O) or (chain I and resi 15 and name CA+C+O)")
            cmd.zoom(f"({act})", 0.5, complete=1)
        L.pymol_render(out, build, 900, 800, theme)
        pngs.append(out)
    fig = plt.figure(figsize=(10.6, 4.3))
    for k, (png, (row, _, letter)) in enumerate(zip(pngs, picks)):
        ax = fig.add_axes([0.0, 0.53 - 0.5 * k, 0.25, 0.4]); L.image_axes(ax, png, crop=True)
        where = "Ser195" if row["d2"] < 0 else "His57"
        ax.text(0.0, -0.04, f"{letter}  OG···C {row['d']:.2f} Å, H on {where}", transform=ax.transAxes, fontsize=8,
                color=c["text2"], va="top")
    # the map of replica 1, against the values reached (not the targets)
    am = fig.add_axes([0.32, 0.14, 0.33, 0.74])
    x = np.array([r["d"] for r in g.values()]); y = np.array([r["d2"] for r in g.values()])
    z = np.array([(r["e"] - e0) * kcal for r in g.values()])
    lv = np.arange(np.floor(z.min() / 4) * 4, z.max() + 4, 4)
    cm = LinearSegmentedColormap.from_list("s", c["seq"])
    cs = am.tricontourf(x, y, z, levels=lv, cmap=cm)
    am.tricontour(x, y, z, levels=lv, colors=c["surface"], linewidths=0.5)
    am.plot(x, y, ".", ms=2, color=c["text2"], alpha=0.5)
    path = lowest_path(g, (0, 0), end)
    am.plot([g[p]["d"] for p in path], [g[p]["d2"] for p in path], "-", lw=1.6, color=c["series"][1])
    for row, _, letter in picks:
        am.plot(row["d"], row["d2"], "o", ms=8, color=c["series"][1], mec=c["surface"], mew=1.5)
        am.annotate(letter, (row["d"], row["d2"]), xytext=(7 if letter == "A" else -16, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=c["text"])
    ys = list(y) + [rows[0]["d2"]]
    am.set_ylim(min(ys) - 0.06, max(ys) + 0.06); am.set_xlim(max(list(x) + [rows[0]["d"]]) + 0.04, min(x) - 0.04)
    am.grid(False)
    am.set_xlabel("Ser195 OG – Lys15 C (Å)"); am.set_ylabel("d(OG–HG) − d(HG–NE2) (Å)")
    for yy, va, lab in ((0.03, "bottom", "H on Ser195"), (0.97, "top", "H on His57")):
        am.text(0.97, yy, lab, transform=am.transAxes, ha="right", va=va, fontsize=8, color=c["text"],
                bbox=dict(boxstyle="round,pad=0.25", fc=c["surface"], ec="none", alpha=0.85))
    am.set_title("QM/MM energy of the minimized structures (replica 1)", fontsize=9.5, fontweight="normal")
    cax = fig.add_axes([0.66, 0.14, 0.012, 0.74])
    cb = fig.colorbar(cs, cax=cax); cb.outline.set_visible(False)
    cb.set_label("energy − reactant (kcal/mol)", color=c["text2"]); cb.ax.tick_params(colors=c["text2"])
    # the energy along the lowest path of each replica
    ah = fig.add_axes([0.79, 0.14, 0.2, 0.74])
    for k, (s, rr) in enumerate(runs.items()):
        gs = scan_grid(rr)
        ps = lowest_path(gs, (0, 0), scan_end(gs))
        ah.plot([gs[p]["d"] for p in ps], [(gs[p]["e"] - rr[0]["e"]) * kcal for p in ps], "o-", ms=3.5,
                color=c["series"][[1, 0, 2][k]], label=f"replica {s}")
    ah.invert_xaxis(); ah.legend(fontsize=8, loc="upper left")
    ah.set_xlabel("Ser195 OG – Lys15 C (Å)"); ah.set_ylabel("energy − reactant (kcal/mol)")
    ah.set_title("along the lowest path", fontsize=9.5, fontweight="normal")
    return fig


@figure(58)
def ex58(theme, ctx):
    c = L.style(theme)
    ds = [4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
    e = []
    for k, d in enumerate(ds):
        t = ctx.out(f"mmq_{k:02d}")
        e.append(float(re.findall(r"QM/MM ELECTROSTATIC ENERGY =\s+(-?[\d.]+)", t)[-1]))
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.plot(ds, e, "o-", color=c["series"][0])
    ax.axhline(0, color=c["axis"], lw=0.8)
    ax.set_xlabel("distance of the TIP3P water (point charges) from the nearest peptide atom (Å)")
    ax.set_ylabel("QM/MM electrostatic energy\n(kcal/mol)")
    ax.set_title("chignolin, PM7 MOZYME 1SCF GRADIENTS with MM_CHARGES (3 charges)", fontsize=9.5, fontweight="normal")
    return fig


# --------------------------------------------------------------------------------------------- checks
@figure(61)
def ex61(theme, ctx):
    c = L.style(theme)
    names = [("diamond", "diamond, 64"), ("silicon", "silicon, 64"), ("graphene", "graphene, 72 (2D)"),
             ("rutile", "rutile, 72"), ("quartz", "α-quartz, 72"), ("zno", "ZnO, 72"),
             ("urea", "urea, 192"), ("mgo_water", "MgO slab + water")]
    fig = plt.figure(figsize=(8.8, 4.6))
    for k, (n, lab) in enumerate(names):
        png = render_cell(ctx, theme, n, turn=[("x", -20), ("y", 20)], spheres=True)
        ax = fig.add_axes([(k % 4) * 0.25, 0.53 - (k // 4) * 0.5, 0.25, 0.44])
        L.image_axes(ax, png)
        ax.text(0.5, -0.02, lab, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=c["text"])
    return fig


@figure(62)
def ex62(theme, ctx):
    c = L.style(theme)
    log = L.text(ctx.work / "pme_check" / "run.log")
    rows = re.findall(r"PASS (.*?): ([\d.e+-]+) (.+?) \(<= ([\d.e+-]+)\)", log)
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    y = np.arange(len(rows))[::-1]
    v = [float(r[1]) for r in rows]; tol = [float(r[3]) for r in rows]
    ax.hlines(y, [x * 1e-1 for x in v], tol, color=c["grid"], lw=2)
    ax.plot(tol, y, "|", ms=12, mew=2, color=c["series"][1], label="tolerance")
    ax.plot(v, y, "o", color=c["series"][0], ms=6, label="value")
    ax.set_xscale("log")
    def short(s):
        s = s.replace(" (CPU)", "").replace("openmm_mopac PME + images of the site ion vs -1.747565/r0",
                                            "QM-PME vs the Madelung constant")
        s = s.replace("Ewald reference vs -1.747565/r0", "Ewald reference vs the Madelung constant")
        s = re.sub(r", max \|.*|, analytic - FD.*", "", s)
        return s.replace("max |analytic - FD|", "").replace(":", "").strip()
    ax.set_yticks(y, [f"{short(r[0])}  [{r[2]}]" for r in rows], fontsize=8)
    ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("value of the check (its unit) and its tolerance, log scale")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return fig


@figure(63)
def ex63(theme, ctx):
    c = L.style(theme)
    t = ctx.out("times_chig")
    stages = {}
    for m in re.finditer(r"^\s*(.{19}?)\s*CPU INTERVAL:\s+[\d.]+, INTEGRAL:\s+[\d.]+\s+WALL-CLOCK INTERVAL:\s+([\d.]+)",
                         t, re.M):
        key = m.group(1).strip()
        key = re.sub(r"\s+IN\s*ITE?$|\s+in MAK$", "", key).strip()
        key = re.sub(r"^(?i:after)\s+", "", key).upper()
        stages[key] = stages.get(key, 0.0) + float(m.group(2))
    top = sorted(stages.items(), key=lambda kv: -kv[1])[:8]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    y = np.arange(len(top))[::-1]
    ax.barh(y, [v for _, v in top], 0.6, color=c["series"][0])
    ax.set_yticks(y, [k for k, _ in top])
    ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("wall-clock time up to each timer, summed over the SCF (s)")
    ax.set_title(f"TIMES, chignolin 1SCF on one CPU core; job {L.job_seconds(t):.2f} s", fontsize=9.5,
                 fontweight="normal")
    return fig

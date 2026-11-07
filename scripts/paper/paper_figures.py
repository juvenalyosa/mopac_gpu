#!/usr/bin/env python3
"""Figures and tables of the Barranquilla MOPAC paper from the results of paper_bench.py.

    paper_figures.py RESULTS        ->  RESULTS/figures/*.pdf|png, RESULTS/tables/*.tex|csv, RESULTS/summary.md

Every figure is made only when its data exist, so it can run after any subset of the experiments.  GPU times
are the median of the repetitions (the error bars span the fastest and slowest); CPU times are one run.
Figures are sized for a journal page (single column 3.4 in, double column 7.0 in), vector PDF plus a 300 dpi
PNG.  Colours: GPU blue, CPU orange (a colour-blind-safe pair that also differs in grey); every series also
has its own marker, so the figures read in black and white.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GPU, CPU, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
STACK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER = "#b4b2ac"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SINGLE, DOUBLE = 3.4, 7.0

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.3,
    "lines.markersize": 4.5, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
})
MK = {"gpu": dict(color=GPU, marker="s", mec="white", mew=0.7), "cpu": dict(color=CPU, marker="o", mec="white", mew=0.7)}


class Out:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data = root / "data"
        self.fig = root / "figures"
        self.tab = root / "tables"
        self.fig.mkdir(exist_ok=True)
        self.tab.mkdir(exist_ok=True)
        self.summary: list[str] = []

    def csv(self, name: str) -> pd.DataFrame | None:
        f = self.data / f"{name}.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f)
        return df if len(df) else None

    def save(self, fig, name: str) -> None:
        fig.savefig(self.fig / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.fig / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  figures/{name}.pdf")

    def table(self, name: str, df: pd.DataFrame, caption: str, notes: str = "") -> None:
        df.to_csv(self.tab / f"{name}.csv", index=False)
        cols = "l" + "r" * (len(df.columns) - 1)
        lines = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}",
                 f"\\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(df.columns) + r" \\", r"\midrule"]
        for _, row in df.iterrows():
            lines.append(" & ".join("--" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        if notes:
            lines.append(f"\\par\\smallskip\\footnotesize {notes}")
        lines.append(r"\end{table}")
        (self.tab / f"{name}.tex").write_text("\n".join(lines) + "\n")
        print(f"  tables/{name}.tex")


def timing(df: pd.DataFrame, key: str = "case") -> pd.DataFrame:
    """One row per case: CPU time, GPU median/min/max, heats, atoms, speed-up, GPU memory."""
    out = []
    for case, g in df.groupby(key, sort=False):
        c, gp = g[g["mode"] == "cpu"], g[g["mode"] == "gpu"]
        gp_ok = gp[gp["wall_s"].notna()] if "wall_s" in gp else gp
        row = {key: case}
        for col in ("label", "kind", "series"):
            if col in g:
                row[col] = g[col].dropna().iloc[0] if g[col].notna().any() else ""
        row["atoms"] = g["atoms"].dropna().max() if "atoms" in g and g["atoms"].notna().any() else np.nan
        row["cpu_s"] = c["wall_s"].iloc[0] if len(c) else np.nan
        row["gpu_s"] = gp_ok["wall_s"].median() if len(gp_ok) else np.nan
        row["gpu_min"] = gp_ok["wall_s"].min() if len(gp_ok) else np.nan
        row["gpu_max"] = gp_ok["wall_s"].max() if len(gp_ok) else np.nan
        row["cpu_heat"] = c["heat"].iloc[0] if len(c) else np.nan
        first = gp[gp["rep"] == 0] if "rep" in gp else gp
        row["gpu_heat"] = first["heat"].iloc[0] if len(first) else np.nan
        row["dheat"] = row["gpu_heat"] - row["cpu_heat"]
        row["speedup"] = row["cpu_s"] / row["gpu_s"] if row["gpu_s"] else np.nan
        row["gpu_mem_mib"] = gp["gpu_mem_mib"].max() if "gpu_mem_mib" in gp else np.nan
        row["scf_iterations"] = first["scf_iterations"].iloc[0] if "scf_iterations" in first and len(first) else np.nan
        row["fallback"] = gp["gpu_scf_fallback"].sum() if "gpu_scf_fallback" in gp else 0
        out.append(row)
    return pd.DataFrame(out)


def fmt_s(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def fmt_e(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:+.4f}" if abs(x) >= 1e-4 else f"{x:+.1e}"


def loglog_fit(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return None
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def errbars(t: pd.DataFrame):
    return np.vstack([(t["gpu_s"] - t["gpu_min"]).clip(lower=0), (t["gpu_max"] - t["gpu_s"]).clip(lower=0)])


# ------------------------------------------------------------------------------------------------ figures

def fig_mozyme_scaling(o: Out) -> None:
    df = o.csv("mozyme_sp")
    if df is None:
        return
    t = timing(df)
    t = t[t["atoms"].notna()].sort_values("atoms")
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), layout="constrained")
    for kind, filled in (("biomolecule", True), ("water cluster", False)):
        sel = t[(t["kind"] == "water cluster") == (kind == "water cluster")]
        if not len(sel):
            continue
        cpu = sel[sel["cpu_s"].notna()]
        a.plot(cpu["atoms"], cpu["cpu_s"], ls="none", label=f"CPU, 1 core ({kind}s)", **dict(
            MK["cpu"], mfc=CPU if filled else "white", mec=CPU if not filled else "white"))
        a.errorbar(sel["atoms"], sel["gpu_s"], yerr=errbars(sel), ls="none", ecolor=GPU, elinewidth=0.8, capsize=0,
                   label=f"GPU ({kind}s)", **dict(MK["gpu"], mfc=GPU if filled else "white",
                                                    mec=GPU if not filled else "white"))
    for mode, col, ys in (("GPU", GPU, t["gpu_s"]), ("CPU", CPU, t["cpu_s"])):
        p = loglog_fit(t["atoms"], ys)
        if p:
            xx = np.array([t["atoms"].min(), t["atoms"].max()])
            ok = ys.notna() & (ys > 0)
            c0 = np.exp(np.mean(np.log(ys[ok]) - p * np.log(t["atoms"][ok])))
            a.plot(xx, c0 * xx ** p, color=col, lw=0.8, alpha=0.6, zorder=1)
            a.text(xx[-1], c0 * xx[-1] ** p, f"  {mode} $\\propto N^{{{p:.2f}}}$", color=INK2, fontsize=7,
                   va="center")
    a.set(xscale="log", yscale="log", xlabel="Atoms", ylabel="Wall time, 1SCF (s)")
    a.set_title("a  MOZYME single point")
    a.legend(loc="upper left")
    both = t[t["cpu_s"].notna() & t["gpu_s"].notna()]
    for kind, filled in (("biomolecule", True), ("water cluster", False)):
        sel = both[(both["kind"] == "water cluster") == (kind == "water cluster")]
        b.plot(sel["atoms"], sel["speedup"], ls="none", label=kind + "s",
               **dict(MK["gpu"], mfc=GPU if filled else "white", mec=GPU if not filled else "white"))
    if len(both):
        top = both.loc[both["speedup"].idxmax()]
        b.annotate(f"{top['speedup']:.0f}$\\times$", (top["atoms"], top["speedup"]), textcoords="offset points",
                   xytext=(4, 4), fontsize=7, color=INK)
    b.set(xscale="log", xlabel="Atoms", ylabel="Speed-up, GPU vs 1 CPU core")
    b.set_ylim(bottom=0)
    b.set_title("b  Speed-up")
    b.legend(loc="upper left")
    o.save(fig, "fig_mozyme_scaling")

    tab = pd.DataFrame({
        "System": t["label"], "Atoms": t["atoms"].astype(int), "CPU (s)": t["cpu_s"].map(fmt_s),
        "GPU (s)": t["gpu_s"].map(fmt_s), "Speed-up": t["speedup"].map(lambda x: "--" if math.isnan(x) else f"{x:.0f}"),
        "$\\Delta H_f$ (kcal/mol)": t["dheat"].map(fmt_e),
        "GPU mem. (GB)": (t["gpu_mem_mib"] / 1024).map(lambda x: "--" if math.isnan(x) else f"{x:.1f}")})
    o.table("table_mozyme_sp", tab, "MOZYME single points (PM7 1SCF): one CPU core against one GPU.",
            "GPU: median of the repetitions. $\\Delta H_f$: GPU $-$ CPU heat of formation. GPU memory: peak of the "
            "whole device.")
    if len(both):
        o.summary.append(f"- MOZYME 1SCF: speed-up {both['speedup'].min():.0f}-{both['speedup'].max():.0f}x over one "
                         f"CPU core ({len(both)} systems, {int(both['atoms'].min())}-{int(both['atoms'].max())} atoms); "
                         f"max |dHf| {both['dheat'].abs().max():.4f} kcal/mol; largest GPU run "
                         f"{int(t['atoms'].max())} atoms in {t.loc[t['atoms'].idxmax(), 'gpu_s']:.1f} s.")


def fig_accuracy(o: Out) -> None:
    rows = []
    for name, group in (("mozyme_sp", "MOZYME 1SCF"), ("cosmo", "MOZYME COSMO"), ("solids", "Solids"),
                        ("conv_mol", "Conventional SCF")):
        df = o.csv(name)
        if df is None:
            continue
        t = timing(df)
        for _, r in t.iterrows():
            if np.isfinite(r["dheat"]):
                rows.append((group, str(r.get("label") or r["case"]) + (f" ({int(r['atoms'])})" if np.isfinite(
                    r["atoms"]) and group == "Solids" else ""), r["dheat"]))
    m = o.csv("methods")
    if m is not None:
        for (sysname, meth), g in m.groupby(["system", "method"], sort=False):
            c, gp = g[g["mode"] == "cpu"]["heat"], g[g["mode"] == "gpu"]["heat"]
            if len(c) and len(gp):
                rows.append(("Methods (MOZYME)", f"{meth}, {'DNA' if sysname == 'dna' else sysname}",
                             gp.iloc[0] - c.iloc[0]))
    grads = []
    for name, group in (("mozyme_grad_compare", "MOZYME"), ("solids_grad_compare", "Solids"),
                        ("conv_mol_grad_compare", "Conventional")):
        df = o.csv(name)
        if df is not None:
            # n = gradient components: the atom count tells the supercells of one solid apart
            grads += [(group, str(r["label"]) + (f" ({int(r['n']) // 3})" if group == "Solids" else ""),
                       r["max_abs"], r["rms"]) for _, r in df.iterrows()]
    if not rows and not grads:
        return
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, max(2.6, 0.13 * max(len(rows), len(grads)) + 0.8)),
                             gridspec_kw={"width_ratios": [1.15, 1]}, layout="constrained")
    floor = 1e-7
    for ax, data, xlabel, tol, title in (
            (axes[0], [(g, l, abs(v)) for g, l, v in rows], "|$\\Delta H_f$| GPU $-$ CPU (kcal/mol)", 0.05,
             "a  Heats of formation"),
            (axes[1], [(g, l, mx) for g, l, mx, _ in grads], "max |$\\Delta g$| GPU $-$ CPU\n(kcal mol$^{-1}$ Å$^{-1}$)",
             0.01, "b  Gradients")):
        if not data:
            ax.set_visible(False)
            continue
        groups = list(dict.fromkeys(g for g, _, _ in data))
        y, ticks, labels, heads = 0, [], [], []
        for grp in groups:
            ticks.append(y)
            labels.append(grp)
            heads.append(len(labels) - 1)
            y -= 1
            for l, v in [(l, v) for g, l, v in data if g == grp]:
                ax.plot(max(v, floor), y, ls="none", **MK["gpu"])
                ticks.append(y)
                labels.append(l)
                y -= 1
            y -= 0.4
        ax.axvline(tol, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.text(tol, 1.0, f" {tol:g}", transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK2,
                va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(floor / 2, max(tol * 5, max(v for *_, v in data) * 3))
        ax.set_yticks(ticks, labels, fontsize=6.5)
        for i, lab in enumerate(ax.get_yticklabels()):
            if i in heads:
                lab.set(fontweight="bold", color=INK, fontsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(y + 0.6, 0.6)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    o.save(fig, "fig_accuracy")
    if rows:
        o.summary.append(f"- Heats GPU - CPU: largest |dHf| {max(abs(v) for *_, v in rows):.2e} kcal/mol over "
                         f"{len(rows)} comparisons.")
    if grads:
        o.summary.append(f"- Gradients GPU - CPU: largest component difference {max(g[2] for g in grads):.2e} "
                         f"kcal/(mol A) over {len(grads)} systems.")
    tight = o.csv("methods_tight")
    if tight is not None:
        for _, r in tight.iterrows():
            o.summary.append(f"- Methods, GPU and CPU > 0.1 kcal/mol apart: {r['case']} CPU {r['d_cpu']:+.4f}, "
                             f"GPU {r['d_gpu']:+.4f} kcal/mol from a CPU SCF with SCFCRT=1.D-5 THRESH=1.D-15")


def fig_opt(o: Out) -> None:
    cy = o.csv("opt_cycles")
    if cy is None:
        return
    systems = list(dict.fromkeys(cy["system"]))
    fig, axes = plt.subplots(len(systems), 3, figsize=(DOUBLE, 2.3 * len(systems)), squeeze=False, layout="constrained")
    for i, s in enumerate(systems):
        g = cy[cy["system"] == s]
        hmin = g["heat"].min()
        for mode in ("cpu", "gpu"):
            h = g[g["mode"] == mode]
            if not len(h):
                continue
            kw = dict(color=MK[mode]["color"], label=mode.upper())
            axes[i, 0].plot(h["cycle"], h["heat"] - hmin, **kw)
            axes[i, 1].plot(h["cycle"], h["grad"], **kw)
            axes[i, 2].plot(h["cycle"], h["cycle_s"].cumsum() / 60.0, **kw)
        axes[i, 0].set(yscale="symlog", xlabel="Cycle", ylabel="$H_f - H_{f,\\min}$ (kcal/mol)")
        axes[i, 1].set(yscale="log", xlabel="Cycle", ylabel="Gradient norm (kcal mol$^{-1}$ Å$^{-1}$)")
        axes[i, 2].set(xlabel="Cycle", ylabel="Elapsed (min)")
        for j, t in enumerate(("Energy", "Gradient", "Cost")):
            axes[i, j].set_title(f"{chr(97 + 3 * i + j)}  {s}: {t.lower()}")
        axes[i, 0].legend()
        for ax in axes[i]:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    o.save(fig, "fig_optimization")
    df = o.csv("opt")
    if df is not None:
        rows = []
        for s in systems:
            g = df[(df["case"] == s)]
            r = {"System": s}
            for mode in ("cpu", "gpu"):
                h = g[g["mode"] == mode]
                c = cy[(cy["system"] == s) & (cy["mode"] == mode)]
                r[f"{mode.upper()} cycles"] = int(c["cycle"].max()) if len(c) else "--"
                r[f"{mode.upper()} (s)"] = fmt_s(h["wall_s"].iloc[0]) if len(h) else "--"
                r[f"{mode.upper()} s/cycle"] = f"{h['wall_s'].iloc[0] / max(1, len(c)):.2f}" if len(h) and len(c) else "--"
            for src in ("gpu", "cpu"):
                f = df[df["case"] == f"{s}_final_{src}"]
                if len(f) == 2:
                    hc, hg = f[f["mode"] == "cpu"]["heat"].iloc[0], f[f["mode"] == "gpu"]["heat"].iloc[0]
                    r[f"$\\Delta H_f$ at {src.upper()} geometry"] = fmt_e(hg - hc)
            rows.append(r)
        o.table("table_optimization", pd.DataFrame(rows), "Geometry optimizations (MOZYME, PM7, L-BFGS).",
                "$\\Delta H_f$: GPU $-$ CPU 1SCF at the final geometry of the optimization (restart file).")


def fig_md(o: Out) -> None:
    md, tmp, sd = o.csv("md_rows"), o.csv("md_temperature"), o.csv("solids_drc_rows")
    panels = [p for p in (md is not None and "crambin_nve" in set(md["system"]), tmp is not None,
                          sd is not None) if p]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE if len(panels) > 1 else SINGLE, 2.4), squeeze=False, layout="constrained")
    k = 0
    for data, system, title in ((md, "crambin_nve", "crambin NVE (MOZYME)"), (sd, "diamond", "diamond NVE (solid)")):
        if data is None or system not in set(data["system"]):
            continue
        ax = axes[0, k]
        for mode in ("cpu", "gpu"):
            g = data[(data["system"] == system) & (data["mode"] == mode)]
            if len(g):
                ax.plot(g["fs"], g["total"] - g["total"].iloc[0], color=MK[mode]["color"], label=mode.upper(),
                        lw=1.6 if mode == "cpu" else 1.0)
        ax.set(xlabel="Time (fs)", ylabel="$E_{tot} - E_{tot}(0)$ (kcal/mol)")
        ax.set_title(f"{chr(97 + k)}  {title}")
        ax.legend()
        k += 1
    if tmp is not None:
        ax = axes[0, k]
        for system, g in tmp.groupby("system"):
            ax.plot(g["fs"], g["T"], color=GPU, label=f"{system}, T")
            ax.plot(g["fs"], g["T_mean"], color=INK2, lw=0.8, label="running mean")
        ax.set(xlabel="Time (fs)", ylabel="Temperature (K)")
        ax.set_title(f"{chr(97 + k)}  NVT, Bussi thermostat (GPU)")
        ax.legend()
    o.save(fig, "fig_dynamics")
    rows = []
    for name in ("md", "solids_drc"):
        df = o.csv(name)
        if df is None or "s_per_step" not in df:
            continue
        for case, g in df.groupby("case", sort=False):
            c, gp = g[g["mode"] == "cpu"], g[g["mode"] == "gpu"]
            r = {"Run": case, "Steps": int(gp["steps"].iloc[0]) if len(gp) and "steps" in gp else "--",
                 "CPU s/step": f"{c['s_per_step'].iloc[0]:.2f}" if len(c) else "--",
                 "GPU s/step": f"{gp['s_per_step'].iloc[0]:.3f}" if len(gp) else "--"}
            if len(c) and len(gp):
                r["Speed-up"] = f"{c['s_per_step'].iloc[0] / gp['s_per_step'].iloc[0]:.0f}"
            for mode, d in (("CPU", c), ("GPU", gp)):
                ok = len(d) and "max_abs_error" in d and pd.notna(d["max_abs_error"].iloc[0])
                r[f"{mode} max |error| (kcal/mol)"] = f"{d['max_abs_error'].iloc[0]:.3f}" if ok else "--"
            rows.append(r)
    if rows:
        o.table("table_dynamics", pd.DataFrame(rows), "Molecular dynamics (DRC, 0.1--0.5 fs steps).",
                "Error: MOPAC's integration error of the total energy (energy given to the bath excluded).")


def fig_solids(o: Out) -> None:
    df = o.csv("solids")
    if df is None:
        return
    t = timing(df)
    si = t[t["series"] == "Si"].sort_values("atoms")
    cells = t[t["series"] != "Si"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), gridspec_kw={"width_ratios": [1, 1.2]}, layout="constrained")
    if len(si):
        c = si[si["cpu_s"].notna()]
        a.plot(c["atoms"], c["cpu_s"], label=f"CPU, {int(df[df['mode'] == 'cpu']['threads'].max())} threads"
               if (df["mode"] == "cpu").any() else "CPU", **MK["cpu"])
        a.errorbar(si["atoms"], si["gpu_s"], yerr=errbars(si), ecolor=GPU, elinewidth=0.8, label="GPU", **MK["gpu"])
        for mode, ys in (("GPU", si["gpu_s"]), ("CPU", si["cpu_s"])):
            p = loglog_fit(si["atoms"], ys)
            if p:
                ok = ys.notna()
                a.annotate(f"{mode} $\\propto N^{{{p:.1f}}}$", (si["atoms"][ok].iloc[-1], ys[ok].iloc[-1]),
                           textcoords="offset points", xytext=(-4, 6), ha="right", fontsize=7, color=INK2)
        a.set(xscale="log", yscale="log", xlabel="Atoms", ylabel="Wall time, 1SCF GRADIENTS (s)")
        a.set_title("a  Silicon supercells (PM7)")
        a.legend(loc="upper left")
    both = t[t["cpu_s"].notna() & t["gpu_s"].notna()].sort_values("atoms")
    if len(both):
        lab = [f"{r['label']} ({int(r['atoms'])})" for _, r in both.iterrows()]
        y = np.arange(len(both))
        b.barh(y, both["speedup"], height=0.6, color=GPU)
        for yi, v in zip(y, both["speedup"]):
            b.text(v, yi, f" {v:.0f}$\\times$", va="center", fontsize=6.5, color=INK)
        b.set_yticks(y, lab, fontsize=6.5)
        b.set_xlabel("Speed-up, GPU vs CPU (same binary)")
        b.set_title("b  Solids, 1SCF GRADIENTS")
        b.grid(axis="y", visible=False)
    o.save(fig, "fig_solids")
    tab = t.sort_values(["series", "atoms"])
    o.table("table_solids", pd.DataFrame({
        "Cell": [f"{r['label']}" for _, r in tab.iterrows()], "Atoms": tab["atoms"].map(lambda x: int(x) if x == x else "--"),
        "SCF it.": tab["scf_iterations"].map(lambda x: int(x) if x == x else "--"),
        "CPU (s)": tab["cpu_s"].map(fmt_s), "GPU (s)": tab["gpu_s"].map(fmt_s),
        "Speed-up": tab["speedup"].map(lambda x: "--" if math.isnan(x) else f"{x:.0f}"),
        "$\\Delta H_f$": tab["dheat"].map(fmt_e)}),
        "Conventional SCF of solids (PM7, 1SCF GRADIENTS, $\\Gamma$ point): CPU (all cores) against the GPU.",
        "Cells under 100 atoms use the GPU with MOPAC\\_FORCEGPU=1.")
    if len(both):
        o.summary.append(f"- Solids (conventional SCF): speed-up {both['speedup'].min():.0f}-{both['speedup'].max():.0f}x "
                         f"against all CPU cores; max |dHf| {both['dheat'].abs().max():.4f} kcal/mol.")


STAGES = [("one-electron matrix", ("HCORE",)), ("Fock matrix", ("FOCKS",)),
          ("diagonalization", ("FULL DIAG", "FINAL DIAG")), ("pseudo-diagonalization", ("PSEUDO DIAG",)),
          ("density", ("DENSIT",)), ("gradient", ("DERIV",))]


def fig_profile(o: Out) -> None:
    sec, runs = o.csv("solids_profile_sections"), o.csv("solids_profile")
    if sec is None or runs is None:
        return
    sec = sec[sec["source"] == "TIMES"]
    cases = list(dict.fromkeys(sec["case"]))
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.45 * len(cases) + 0.9), layout="constrained")
    labels = []
    for i, case in enumerate(cases):
        g = sec[sec["case"] == case].set_index("name")["s"]
        total = runs[runs["case"] == case]["job_s"].iloc[0] if (runs["case"] == case).any() else g.sum()
        left = 0.0
        for k, (name, keys) in enumerate(STAGES):
            v = sum(g.get(x, 0.0) for x in keys)
            ax.barh(i, v, left=left, height=0.55, color=STACK[k], edgecolor="white", linewidth=1.0,
                    label=name if i == 0 else None)
            left += v
        ax.barh(i, max(0.0, total - left), left=left, height=0.55, color=OTHER, edgecolor="white", linewidth=1.0,
                label="other (setup, SCF control, I/O)" if i == 0 else None)
        ax.text(max(total, left), i, f" {total:.1f} s", va="center", fontsize=7)
        a = runs[runs["case"] == case]["atoms"]
        labels.append(f"{case} ({int(a.iloc[0])} atoms)" if len(a) and a.notna().any() else case)
    ax.set_yticks(range(len(cases)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall time on the GPU (s)")
    env = o.root / "env.json"
    gpu = json.loads(env.read_text()).get("gpu", "").split(",")[0].replace("NVIDIA ", "") if env.exists() else ""
    ax.set_title(f"Where the time of a 1SCF GRADIENTS goes ({gpu or 'GPU'}, PM7)")
    fig.legend(ncol=4, loc="outside lower center")
    ax.grid(axis="y", visible=False)
    o.save(fig, "fig_profile")


def fig_freq(o: Out) -> None:
    frames = [f for f in (o.csv("thermo_freqs"), o.csv("solids_force_freqs")) if f is not None]
    if not frames:
        return
    fq = pd.concat(frames)
    systems = list(dict.fromkeys(fq["system"]))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), layout="constrained")
    marks = ["s", "o", "^", "D"]
    for i, s in enumerate(systems):
        g = fq[fq["system"] == s]
        axes[0].plot(g["cpu"], g["gpu"], ls="none", marker=marks[i % 4], ms=3, mec="white", mew=0.4,
                     color=STACK[i], label=s)
        axes[1].plot(g["cpu"], g["gpu"] - g["cpu"], ls="none", marker=marks[i % 4], ms=3, mec="white", mew=0.4,
                     color=STACK[i], label=s)
    lim = [fq[["cpu", "gpu"]].min().min(), fq[["cpu", "gpu"]].max().max()]
    axes[0].plot(lim, lim, color=INK2, lw=0.6, zorder=0)
    axes[0].set(xlabel="CPU frequency (cm$^{-1}$)", ylabel="GPU frequency (cm$^{-1}$)")
    axes[0].set_title("a  Harmonic frequencies")
    axes[0].legend(loc="upper left")
    axes[1].axhline(0, color=INK2, lw=0.6)
    axes[1].set(xlabel="CPU frequency (cm$^{-1}$)", ylabel="GPU $-$ CPU (cm$^{-1}$)")
    axes[1].set_title("b  Difference")
    o.save(fig, "fig_frequencies")
    th = o.csv("thermo")
    if th is not None:
        o.table("table_thermo", th.rename(columns={"system": "Run", "gpu_s": "GPU (s)", "cpu_ref_s": "CPU ref. (s)"}),
                "MOZYME FORCE / FORCETS on the GPU against stored CPU references.",
                "CPU references stored in tests/data/mozyme\\_gpu\\_reference/reference.json, computed on "
                "laptop CPUs (FORCE and FORCETS: Intel Core i7-8850H), not on the benchmark machine: "
                "the times are not a speed-up.")


def fig_qmmm(o: Out) -> None:
    q, nve, cut = o.csv("qmmm"), o.csv("qmmm_nve"), o.csv("qmmm_enzyme_cutoff")
    if q is not None:
        q = q[q["s_per_step"].notna()] if "s_per_step" in q else None
        q = q if q is not None and len(q) else None
    panels = [x for x in (q, nve, cut) if x is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE if len(panels) > 1 else SINGLE, 2.6), squeeze=False, layout="constrained")
    k = 0
    if q is not None:
        ax = axes[0, k]
        # QM regions of <= 100 atoms (link atoms included) run MOPAC on the CPU even in the GPU mode
        # (mozyme_qmmm_init); there only OpenMM and the PME grid use the GPU
        small = (q["qm_atoms"] + q.get("link_atoms", 0).fillna(0)) <= 100
        for mode in ("cpu", "gpu"):
            g = q[(q["mode"] == mode) & ~small].sort_values("qm_atoms")
            if len(g):
                ax.plot(g["qm_atoms"], g["s_per_step"], label=f"MOPAC on the {mode.upper()}", **MK[mode])
        g = q[(q["mode"] == "gpu") & small]
        if len(g):
            ax.plot(g["qm_atoms"], g["s_per_step"], ls="none", color=CPU, marker="D", mfc="white", mew=0.9,
                    label="$\\leq$100 QM atoms: MOPAC on the CPU,\nMM and PME on the GPU")
        for _, r in q[q["mode"] == "gpu"].iterrows():
            ax.annotate(f"{r['label']}\n{r['ps_per_day']:.0f} ps/day", (r["qm_atoms"], r["s_per_step"]),
                        textcoords="offset points", xytext=(5, 2), va="bottom", fontsize=6, color=INK2)
        ax.set(xscale="log", yscale="log", xlabel="QM atoms", ylabel="Wall time per MD step (s)")
        ax.set_title(f"{chr(97 + k)}  QM/MM MD (OpenMM + MOPAC)")
        ax.legend(loc="upper left")
        k += 1
    if nve is not None:
        ax = axes[0, k]
        for i, (case, g) in enumerate(nve.groupby("case", sort=False)):
            name = q[q["case"] == case]["label"].iloc[0] if q is not None and (q["case"] == case).any() else case
            ax.plot(g["fs"], g["E_kj"] - g["E_kj"].iloc[0], color=STACK[i], lw=0.9, label=name)
        ax.set(xlabel="Time (fs)", ylabel="$E_{tot} - E_{tot}(0)$ (kJ/mol)")
        ax.set_title(f"{chr(97 + k)}  NVE energy conservation")
        ax.legend()
        k += 1
    if cut is not None:
        ax = axes[0, k]
        for i, (meth, g) in enumerate(cut.groupby("method", sort=False)):
            g = g.sort_values("roff")
            ax.plot(g["roff"], g["rms_err"], marker="os"[i % 2], mec="white", mew=0.6, color=(CPU, GPU)[i % 2],
                    label="QM-PME" if meth == "PME" else "cut-off only")
        ax.set(xlabel="QM-MM cut-off (Å)", ylabel="RMS force error (kJ mol$^{-1}$ nm$^{-1}$)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{chr(97 + k)}  Trypsin-BPTI: long-range part")
        ax.legend()
    o.save(fig, "fig_qmmm")
    if q is not None:
        tab = q.sort_values(["qm_atoms", "mode"])
        o.table("table_qmmm", pd.DataFrame({
            "QM region": tab["label"], "QM atoms": tab["qm_atoms"], "Total atoms": tab["total_atoms"],
            "MOPAC on": tab["mode"].str.upper(), "s/step": tab["s_per_step"].map(lambda x: f"{x:.3f}"),
            "ps/day": tab["ps_per_day"].map(lambda x: f"{x:.1f}"),
            "NVE drift (kJ/mol/ps)": tab.get("nve_slope_kj_ps", pd.Series(dtype=float)).map(
                lambda x: "--" if x != x else f"{x:.1f}")}),
            "QM/MM molecular dynamics: OpenMM (CUDA) with MOPAC/MOZYME (PM7) for the QM region, TIP3P water.",
            "0.5 fs steps, Langevin NVT; NVE: linear-fit slope of the total energy.")


def tables_controls(o: Out) -> None:
    """Controls asked for by the review: CPU threads of MOZYME, MOPAC 23.1.2 against the fork's CPU path (with
    central differences where the gradients differ), and the QM-MM PME against a direct Ewald sum."""
    th = o.csv("mozyme_threads")
    if th is not None:
        rows = []
        for system, g in th.groupby("system", sort=False):
            t1 = g.loc[g["threads"] == g["threads"].min(), "wall_s"].iloc[0]
            for _, r in g.sort_values("threads").iterrows():
                rows.append({"System": r["label"], "Threads": int(r["threads"]), "CPU (s)": f"{r['wall_s']:.1f}",
                             "vs 1 thread": f"{t1 / r['wall_s']:.2f}", "$H_f$ (kcal/mol)": f"{r['heat']:.5f}"})
        o.table("table_mozyme_threads", pd.DataFrame(rows), "MOZYME 1SCF on the CPU with 1 to all threads.",
                "Same executable with MOPAC\\_NOGPU=1; OMP\\_NUM\\_THREADS and OPENBLAS\\_NUM\\_THREADS set to the count.")
        o.summary.append("- MOZYME CPU threads: " + "; ".join(
            f"{r['System']} {r['Threads']} thr {r['CPU (s)']} s" for r in rows))
    up = o.csv("upstream_compare")
    if up is not None:
        rows = []
        for _, r in up.iterrows():
            rows.append({"System": r["label"], "Calculation": r["kind"], "Fork run": str(r["mode"]).replace("_", " "),
                         "$H_f$ 23.1.2": "failed" if r.get("upstream_failed") is True else f"{r['heat_upstream']:.5f}",
                         "$H_f$ fork": "failed" if r.get("failed") is True else f"{r['heat']:.5f}",
                         "max $|\\Delta g|$": fmt_e(r["grad_max_abs"]) if pd.notna(r.get("grad_max_abs")) else "--",
                         "SCF it. (23.1.2 / fork)": (f"{int(r['iter_upstream'])} / {int(r['iter'])}"
                                                    if pd.notna(r.get("iter_upstream")) and pd.notna(r.get("iter")) else "--"),
                         "CPU s (23.1.2 / fork)": f"{r['wall_upstream']:.1f} / {r['wall']:.1f}"})
        o.table("table_upstream", pd.DataFrame(rows), "MOPAC 23.1.2 (commit 78dc4572) against the CPU path of the fork.",
                "Same decks, threads and machine. Gradients in kcal/(mol \\AA). fork shift80: the original 80 eV level "
                "shift of solids, without the damped start.")
    fd = o.csv("upstream_fd")
    if fd is not None:
        df = pd.DataFrame({"Atom": fd["name"], "Coord.": fd["axis"], "Central diff.": fd["fd"].map("{:.3f}".format),
                           "Fork": fd["fork"].map("{:.3f}".format), "23.1.2": fd["upstream"].map("{:.3f}".format)})
        o.table("table_upstream_fd", df, "Chignolin, PM7 gradient (kcal/(mol \\AA)) of the coordinates where MOPAC 23.1.2 "
                "and the fork differ most, against central differences of the heat.",
                "Conventional SCF, SCFCRT=1.D-8; step 0.005 \\AA; both programs give the same heat.")
        o.summary.append(f"- Gradient vs central differences (chignolin, {len(fd)} coords): max |fork - FD| "
                         f"{(fd['fork'] - fd['fd']).abs().max():.3f}, max |23.1.2 - FD| {(fd['upstream'] - fd['fd']).abs().max():.3f}")
    pme = o.csv("qmmm_pme_check")
    if pme is not None:
        df = pd.DataFrame({"Check": pme["check"].str.replace("_", " "), "Value": pme["value"].map(fmt_e),
                           "Tolerance": pme["tolerance"].map(fmt_e), "Unit": pme["unit"],
                           "Pass": pme["pass"].map(lambda v: "yes" if str(v) == "True" else "NO")})
        o.table("table_qmmm_pme_check", df, "QM--MM PME against a direct Ewald sum, the rock-salt Madelung constant "
                "and finite differences.", "tests/check\\_qmmm\\_pme\\_ewald.py.")
        o.summary.append(f"- QM-MM PME checks: {int((pme['pass'].astype(str) == 'True').sum())} of {len(pme)} pass")


def write_summary(o: Out) -> None:
    env = o.root / "env.json"
    head = ["# Benchmark summary", ""]
    if env.exists():
        e = json.loads(env.read_text())
        head += [f"- GPU: {e.get('gpu')}", f"- CPU: {e.get('cpu_model')} ({e.get('cpu_count')} logical cores), "
                 f"{e.get('memory')}", f"- Code: {e.get('commit')} ({e.get('tag')}); {e.get('mopac')}",
                 f"- CUDA: {e.get('cuda')}; OpenMM {e.get('openmm')}; {e.get('date')}", ""]
    (o.root / "summary.md").write_text("\n".join(head + o.summary) + "\n")
    print("\n".join(head + o.summary))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    o = Out(Path(sys.argv[1]))
    for f in (fig_mozyme_scaling, fig_accuracy, fig_opt, fig_md, fig_solids, fig_profile, fig_freq, fig_qmmm,
              tables_controls):
        try:
            f(o)
        except Exception as exc:  # noqa: BLE001  (one broken figure must not stop the others)
            print(f"  {f.__name__} failed: {exc!r}")
    write_summary(o)
    return 0


if __name__ == "__main__":
    sys.exit(main())

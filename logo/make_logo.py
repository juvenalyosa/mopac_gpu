"""Barranquilla MOPAC logo: a p orbital on a chip, in the red, yellow and green of Barranquilla.

    python logo/make_logo.py <dir with Montserrat-Bold.ttf, -SemiBold.ttf, -Medium.ttf>

Montserrat (SIL Open Font License 1.1) comes from https://github.com/JulietaUla/Montserrat; the text is written
as outlines, so the font is not needed to display the files. Writes, next to this script:
    barranquilla_mopac_logo.svg/.png        mark + wordmark, for light backgrounds
    barranquilla_mopac_logo_dark.svg/.png   the same for dark backgrounds
    barranquilla_mopac_mark.svg/.png        the mark alone (avatar, favicon)
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle

RED, YEL, GRN = "#C8102E", "#F2A900", "#00843D"
HERE = Path(__file__).resolve().parent
FONTS = Path(sys.argv[1])
BOLD, SEMI, MED = (FontProperties(fname=FONTS / f"Montserrat-{w}.ttf") for w in ("Bold", "SemiBold", "Medium"))
plt.rcParams["svg.fonttype"] = "path"


def chip(ax, cx, cy, s, lw, ink, fill="none"):
    """Rounded die with three pins per side, a p orbital (red and green lobes) and a yellow nucleus."""
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s, boxstyle=f"round,pad=0,rounding_size={s * 0.16}",
                                fc=fill, ec=ink, lw=lw))
    for k in (-0.25, 0, 0.25):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x0 = cx + dx * s / 2 + (k * s if dy else 0)
            y0 = cy + dy * s / 2 + (k * s if dx else 0)
            ax.plot([x0, x0 + dx * s * 0.12], [y0, y0 + dy * s * 0.12], color=ink, lw=lw * 0.8,
                    solid_capstyle="round")
    ax.add_patch(Ellipse((cx, cy + s * 0.18), s * 0.26, s * 0.36, color=RED))
    ax.add_patch(Ellipse((cx, cy - s * 0.18), s * 0.26, s * 0.36, color=GRN))
    ax.add_patch(Circle((cx, cy), s * 0.07, color=YEL, zorder=5))


def logo(path, ink, mute):
    fig, ax = plt.subplots(figsize=(6.4, 1.6))
    ax.set(xlim=(0, 8), ylim=(0, 2), aspect="equal")
    ax.axis("off")
    chip(ax, 1.0, 1.0, 1.1, 4, ink)
    ax.text(2.12, 1.42, "BARRANQUILLA", fontproperties=MED, fontsize=15, color=mute, va="bottom", ha="left")
    t = ax.text(2.05, 0.6, "MOPAC", fontproperties=BOLD, fontsize=50, color=ink, va="baseline", ha="left")
    fig.canvas.draw()
    x1 = t.get_window_extent().transformed(ax.transData.inverted()).x1
    ax.text(x1 + 0.08, 0.6, "GPU", fontproperties=SEMI, fontsize=21, color=GRN, va="baseline", ha="left")
    for ext, kw in ((".svg", {}), (".png", {"dpi": 300})):
        fig.savefig(path.with_suffix(ext), bbox_inches="tight", pad_inches=0.05, transparent=True, **kw)
    plt.close(fig)


def mark(path, ink):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set(xlim=(0, 2), ylim=(0, 2), aspect="equal")
    ax.axis("off")
    chip(ax, 1.0, 1.0, 1.3, 6, ink, fill="white")   # white die: readable on any background
    for ext, kw in ((".svg", {}), (".png", {"dpi": 256})):
        fig.savefig(path.with_suffix(ext), pad_inches=0, transparent=True, **kw)
    plt.close(fig)


logo(HERE / "barranquilla_mopac_logo", "#1d1d1f", "#6e6e73")
logo(HERE / "barranquilla_mopac_logo_dark", "#f5f5f7", "#a1a1a6")
mark(HERE / "barranquilla_mopac_mark", "#1d1d1f")
print("written in", HERE)

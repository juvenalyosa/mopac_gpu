"""Shared pieces of the wiki figures: two themes (light / dark), MOPAC output readers, PyMOL renders.

Colours are the validated reference palette of the dataviz method (categorical order blue, orange, aqua, ...;
sequential blue; diverging blue <-> red with a grey midpoint), stepped separately for the light and the dark
surface.  Every figure is drawn twice, <name>.png (light) and <name>-dark.png (dark).
"""
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e6e5e1", axis="#b9b8b2",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
                  mid="#f0efec", seq=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
                  div=["#1c5cab", "#86b6ef", "#f0efec", "#ee8e8c", "#b42f2e"], good="#0ca30c", critical="#d03b3b",
                  carbon="#9a9a95"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", grid="#2f2f2d", axis="#5c5b56",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
                 mid="#383835", seq=["#184f95", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
                 div=["#5598e7", "#256abf", "#383835", "#b04545", "#e66767"], good="#0ca30c", critical="#d03b3b",
                 carbon="#b4b3ad"),
}
_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans") if f in _fonts), "DejaVu Sans")


def style(theme):
    c = THEMES[theme]
    plt.rcParams.update({
        "font.family": FONT, "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"], "savefig.facecolor": c["surface"],
        "text.color": c["text"], "axes.labelcolor": c["text2"], "xtick.color": c["text2"], "ytick.color": c["text2"],
        "axes.edgecolor": c["axis"], "axes.linewidth": 0.8, "axes.grid": True, "grid.color": c["grid"],
        "grid.linewidth": 0.7, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.8, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 9,
        "xtick.major.size": 3, "ytick.major.size": 3, "axes.titlelocation": "left", "axes.titleweight": "bold",
        "axes.titlecolor": c["text"], "axes.prop_cycle": matplotlib.cycler(color=c["series"]),
    })
    return c


def save(fig, out_dir, name, theme):
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / (f"{name}.png" if theme == "light" else f"{name}-dark.png")
    fig.savefig(fn, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return fn


def image_axes(ax, png, crop=False):
    """show a rendered (transparent) PNG in an axes without frame; crop: to the drawn pixels, with a small margin"""
    import matplotlib.image as mpimg
    im = mpimg.imread(png)
    if crop:
        ys, xs = np.nonzero(im[..., 3] > 0.02)
        m = int(0.04 * max(im.shape[:2]))
        im = im[max(ys.min() - m, 0):ys.max() + m + 1, max(xs.min() - m, 0):xs.max() + m + 1]
    ax.imshow(im)
    ax.set_axis_off()
    return ax


# ---------------------------------------------------------------------------------- MOPAC output readers
def text(path):
    return Path(path).read_text(errors="replace")


def heat(t):
    m = re.findall(r"FINAL HEAT OF FORMATION =\s+(-?[\d.]+)", t)
    return float(m[-1]) if m else None


def job_seconds(t):
    m = re.findall(r"TOTAL JOB TIME:\s+([\d.]+)", t)
    return float(m[-1]) if m else None


def charges(t):
    """[(atom label, charge)] from the last NET ATOMIC CHARGES table"""
    i = t.rfind("NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS")
    out = []
    for line in t[i:].splitlines()[3:]:
        m = re.match(r"\s+\d+\s+(\S.*?)\s+(-?\d+\.\d+)\s+\d+\.\d+", line)
        if not m:
            if out:
                break
            continue
        out.append((m.group(1), float(m.group(2))))
    return out


def cart_gradients(t):
    """array (n, 3) of Cartesian gradients (kcal/(mol A)) from FINAL POINT AND DERIVATIVES"""
    i = t.rfind("FINAL  POINT  AND  DERIVATIVES")
    g = []
    for line in t[i:].splitlines()[5:]:
        m = re.match(r"\s+\d+\s+\S.*?\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+-?\d", line)
        if not m:
            m = re.match(r"\s+\d+\s+[A-Z][a-z]?\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+-?\d", line)
        if not m:
            if g:
                break
            continue
        g.append([float(m.group(k)) for k in (1, 2, 3)])
    return np.array(g)


def opt_cycles(t):
    """[(cycle, gnorm, heat)] from the CYCLE lines of an optimization"""
    return [(int(a), float(b), float(c)) for a, b, c in
            re.findall(r"CYCLE:\s+(\d+)\s+TIME:.*?GRAD\.:\s+([\d.]+)\s+HEAT:\s*(-?[\d.]+)", t)]


def scf_trace(t):
    """[(iteration, energy, delta E, PL)] from the PL / ITER lines of the conventional or MOZYME SCF"""
    out = []
    for line in t.splitlines():
        m = re.match(r"\s*ITER(?:ATION|\.)\s+(\d+)\s+PLS?=\s*(\S+)(?:\s+[\d.E+-]+)?\s+ENERGY\s+(-?[\d.]+)"
                     r"\s+DELTAE\s*(-?[\d.E+-]+)", line)
        if m:
            # MOPAC prints the iteration number in three digits (1000 shows as 1): count the lines instead
            out.append((len(out) + 1, float(m.group(3)), float(m.group(4)), float(m.group(2))))
    return out


def freq_table(t):
    """all frequencies (cm-1) of the NORMAL COORDINATE ANALYSIS table"""
    i = t.find("NORMAL COORDINATE ANALYSIS"); j = t.find("MASS-WEIGHTED COORDINATE ANALYSIS", i)
    lines = t[i:j].splitlines()
    out = []
    for k, l in enumerate(lines):
        if l.strip().startswith("Root No."):
            for l2 in lines[k + 1:k + 6]:
                if re.fullmatch(r"\s*(-?\d+\.\d+\s*)+", l2):
                    out += [float(x) for x in l2.split()]
                    break
    return out


def eigenvalues(t):
    i = t.rfind("EIGENVALUES")
    vals = []
    for line in t[i:].splitlines()[1:]:
        nums = re.findall(r"-?\d+\.\d+", line)
        if not nums or re.search(r"[A-Za-z]", line):
            if vals:
                break
            continue
        vals += [float(x) for x in nums]
    return np.array(vals)


def homo_lumo(t):
    m = re.findall(r"HOMO LUMO ENERGIES \(EV\) =\s+(-?[\d.]+)\s+(-?[\d.]+)", t)
    return tuple(map(float, m[-1])) if m else None


def bond_orders(t):
    """{(i, j): order} (1-based, i < j) from the BONDS table"""
    i = t.rfind("(VALENCIES)   BOND ORDERS")
    bo, cur = {}, None
    for line in t[i:].splitlines()[2:]:
        if not line.strip():
            continue
        m = re.match(r"\s+(\d+)\s+[A-Z][a-z]?\s+\(\s*[\d.]+\)(.*)", line)
        if m:
            cur, rest = int(m.group(1)), m.group(2)
        elif cur and re.match(r"\s{20,}\d", line):
            rest = line
        else:
            break
        for j, o in re.findall(r"(\d+)\s+[A-Z][a-z]?\s+(\d+\.\d+)", rest):
            j = int(j)
            bo[(min(cur, j), max(cur, j))] = float(o)
    return bo


def frequencies(t):
    """[(frequency cm-1, T-dipole)] from DESCRIPTION OF VIBRATIONS"""
    f = [float(x) for x in re.findall(r"FREQUENCY\s+(-?[\d.]+)", t)]
    d = [float(x) for x in re.findall(r"(?:T-DIPOLE|TRANSITION DIPOLE)\s+([\d.]+)", t)]
    return f, d


def drc_rows(t):
    """[(fs, potential, kinetic, total, error)] of the DRC table"""
    rows = []
    for line in t.splitlines():
        m = re.match(r"\s+(\d+\.\d+)\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+%", line)
        if m:
            rows.append(tuple(float(m.group(k)) for k in range(1, 6)))
    return rows


def temperature_lines(t):
    """[(step, fs, T, <T>, energy to bath)]"""
    return [tuple(float(x) for x in m) for m in re.findall(
        r"TEMPERATURE: STEP\s+(\d+)\s+TIME\(FS\)\s+([\d.]+)\s+T\(K\)\s+([\d.]+)\s+<T>\(K\)\s+([\d.]+)\s+"
        r"ENERGY TO BATH \(KCAL/MOL\)\s+(-?[\d.]+)", t)]


def pdb_atoms(path):
    """[(name, resname, chain, resseq, element, xyz)] of ATOM/HETATM records"""
    out = []
    for l in Path(path).read_text().splitlines():
        if l.startswith(("ATOM", "HETATM")):
            el = l[76:78].strip() or re.sub(r"[^A-Za-z]", "", l[12:16])[:1]
            out.append((l[12:16].strip(), l[17:20].strip(), l[21], int(l[22:26]), el,
                        np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return out


def mop_cell(path):
    """atoms [(element, xyz)] and translation vectors of a repository cell (.mop)"""
    atoms, tv = [], []
    for l in Path(path).read_text().splitlines()[3:]:
        f = l.split()
        if len(f) < 6:
            continue
        xyz = np.array([float(f[1]), float(f[3]), float(f[5])])
        (tv if f[0] == "Tv" else atoms).append(xyz if f[0] == "Tv" else (f[0], xyz))
    return atoms, np.array(tv)


def write_pdb(atoms, path, bfac=None):
    """atoms [(element, xyz)] as HETATM records (one residue), optional B-factors"""
    lines = []
    for k, (el, x) in enumerate(atoms):
        b = 0.0 if bfac is None else bfac[k]
        lines.append(f"HETATM{k + 1:5d} {el:>2s}   UNK A   1    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}"
                     f"  1.00{b:6.2f}          {el:>2s}\n")
    Path(path).write_text("".join(lines) + "END\n")


# ---------------------------------------------------------------------------------- PyMOL
def pymol_render(out_png, build, width=1400, height=1000, theme="light", turn=None):
    """build(cmd, colours) loads and styles the scene; rendered with a transparent background"""
    import pymol2
    c = THEMES[theme]
    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 0)
        cmd.set("antialias", 2)
        cmd.set("ray_shadows", 0)
        cmd.set("specular", 0.15)
        cmd.set("ambient", 0.45)
        cmd.set("direct", 0.55)
        cmd.set("cartoon_transparency", 0.0)
        cmd.set("stick_radius", 0.16)
        cmd.set("sphere_scale", 0.25)
        cmd.set("ray_trace_mode", 0)
        for k, h in enumerate(c["series"]):
            cmd.set_color(f"s{k + 1}", _rgb(h))
        for k, h in enumerate(rdiv(c)):
            cmd.set_color(f"d{k + 1}", _rgb(h))
        for k, h in enumerate(c["seq"]):
            cmd.set_color(f"q{k + 1}", _rgb(h))
        cmd.set_color("carbon_t", _rgb(c["carbon"]))
        cmd.set_color("mid_t", _rgb(c["mid"]))
        build(cmd, c)
        if turn:
            for axis, ang in turn:
                cmd.turn(axis, ang)
            cmd.zoom("all", 1.0, complete=1)
        cmd.png(str(out_png), width=width, height=height, dpi=200, ray=1)
    return out_png


def _rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def element_colours(cmd, sel="all"):
    cmd.color("carbon_t", f"{sel} and elem C")
    cmd.color("s1", f"{sel} and elem N")
    cmd.color("s8", f"{sel} and elem O")
    cmd.color("s4", f"{sel} and elem S")
    cmd.color("grey80", f"{sel} and elem H")


def plain_log(ax, axis, ticks):
    """log axis with plain tick labels (1, 10, 100 instead of 10^0 ...)"""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    a.set_minor_formatter(NullFormatter())


RES_RE = re.compile(r"\((?:ATOM|HETATM)\s+\d+\s+\S+\s+(\w{1,3})\s+(\w)\s*(-?\d+)\)")


def residue_sums(charge_rows):
    """[(resname, chain, resseq, sum of charges)] in order, from labels such as N(ATOM 1 N GLY A 1)"""
    out, idx = [], {}
    for lab, q in charge_rows:
        m = RES_RE.search(lab)
        if not m:
            continue
        key = (m.group(1), m.group(2), int(m.group(3)))
        if key not in idx:
            idx[key] = len(out)
            out.append([*key, 0.0])
        out[idx[key]][3] += q
    return [tuple(r) for r in out]


def time_axis_fs(rows):
    return np.array([r[0] for r in rows])


def rdiv(c):
    """diverging palette for 3D renders and their colour bars: the grey midpoint stays visible on both surfaces"""
    return [c["div"][0], c["div"][1] if c is THEMES["light"] else "#4a85cf", c["carbon"],
            c["div"][3] if c is THEMES["light"] else "#c95a59", c["div"][4]]

"""Draw the wiki figures: python make_figures.py <work dir of jobs.py> <wiki dir> --ext <dir> [--only 1,2]"""
import argparse, sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as L
import figures
import figures_runs  # noqa: F401  (registers the figures that read calculations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("wiki", type=Path)
    ap.add_argument("--ext", type=Path, required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    renders = a.work.parent / "wikifig_renders"
    renders.mkdir(exist_ok=True)
    ctx = figures.Ctx(a.work, a.ext, renders)
    want = [int(x) for x in a.only.split(",")] if a.only else sorted(figures.FIG)
    for n in want:
        if n not in figures.FIG:
            print(f"ex{n:02d}: no figure function")
            continue
        try:
            for theme in ("light", "dark"):
                fig = figures.FIG[n](theme, ctx)
                fn = L.save(fig, a.wiki / "images", f"ex{n:02d}", theme)
            print(f"ex{n:02d}: ok")
        except Exception as e:
            print(f"ex{n:02d}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MOZYME GPU thermochemistry check: FORCE / FORCETS + THERMO, CPU versus GPU.

Usage:
  check_mozyme_gpu_thermo.py <mopac> [--pdb-id 1UAO] [--partial A9=4] [--large-pdb-id 1CRN]
      [--temperature 298] [--gnorm 1] [--opt-cycles 3000] [--precise] [--scfcrt 0.0001]
      [--gpu-repeat 2] [--skip-large]
      [--work-dir DIR] [--reuse]

Small system (default chignolin, PDB 1UAO, 140 atoms with hydrogens; the GPU path needs more
than 100 atoms):
  1. prepare the PDB (first NMR model, ADD-H, MOPAC input chemistry check) and optimize it on the
     GPU to GNORM=--gnorm (THERMO needs a stationary point);
  2. FORCE THERMO(T) at that geometry with the GPU disabled and enabled: the harmonic
     frequencies, the zero-point energy, H, Cp, S and the free-energy correction
     G = ZPE + H(T) - T*S are compared;
  3. FORCETS THERMO(T) OPT("chain+residue"=radius): partial Hessian of the atoms within the
     radius of the selected residue (e.g. A9=4: 4 Angstrom around residue 9 of chain A), CPU vs GPU.

Large system (default crambin, 642 atoms), GPU only: optimization, full FORCE THERMO and the
partial FORCETS, to measure the time.

The tolerances are provisional (this is the first validation of FORCE on the GPU): ZPE within
0.1 kcal/mol, entropy within 1 cal/(mol K), RMS difference of the frequencies above 100 cm-1
within 2 cm-1.  Every number is printed, pass or fail.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mozyme_md_workflow as md  # noqa: E402
import mozyme_parallel_force as pf  # noqa: E402

ZPE_RE = re.compile(r"ZERO POINT ENERGY\s+([-0-9.]+)\s+KCAL/MOL")
TOT_RE = re.compile(r"^\s+TOT\.\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)", re.M)
FLOAT_RE = re.compile(r"-?\d+\.\d+")
CPU_ENV = {"MOPAC_NOGPU": "1"}


def frequencies(text: str) -> list[float]:
    """Harmonic frequencies (cm-1) from the 'Root No.' blocks of the normal-coordinate analysis
    (the first section only: the mass-weighted section that follows repeats the same modes)."""
    start = text.find("NORMAL COORDINATE ANALYSIS")
    if start < 0:
        return []
    end = text.find("MASS-WEIGHTED COORDINATE ANALYSIS", start)
    lines = text[start:end if end > 0 else len(text)].splitlines()
    freqs: list[float] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Root No."):
            j = i + 1
            seen_labels = False
            while j < len(lines) and j < i + 8:
                s = lines[j].strip()
                if s and not seen_labels:
                    seen_labels = True          # the "1 A  2 A ..." (or "1 2 ...") label line
                elif s:
                    freqs.extend(float(x) for x in FLOAT_RE.findall(s))
                    break
                j += 1
            i = j
        i += 1
    return freqs


def thermo(text: str, temperature: float) -> dict:
    zpe = ZPE_RE.findall(text)
    tot = TOT_RE.findall(text)
    out = {"zpe": float(zpe[-1]) if zpe else None, "freqs": frequencies(text)}
    if tot:
        hof, h_cal, cp, s = (float(x) for x in tot[0])
        # G: free-energy correction ZPE + H(T) - T*S (kcal/mol), the quantity added to a heat of formation
        zp = out["zpe"] or 0.0
        out.update(hof=hof, H=h_cal / 1000.0, Cp=cp, S=s, G=zp + h_cal / 1000.0 - temperature * s / 1000.0)
    return out


def get_pdb(pdb_id: str, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    raw = work / f"{pdb_id}.pdb"
    if not raw.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", raw)
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mopac", type=Path)
    ap.add_argument("--pdb-id", default="1UAO")
    ap.add_argument("--partial", default="A9=4", help='FORCETS region: "chain+residue=radius"')
    ap.add_argument("--large-pdb-id", default="1CRN")
    ap.add_argument("--large-partial", default="A25=5")
    ap.add_argument("--temperature", type=float, default=298.0)
    ap.add_argument("--gnorm", type=float, default=1.0)
    ap.add_argument("--opt-cycles", type=int, default=3000)
    ap.add_argument("--precise", action="store_true", help="also run the GPU FORCE with PRECISE")
    ap.add_argument("--scfcrt", type=float, default=None,
                    help="SCF criterion (kcal/mol) for every FORCE/FORCETS run, CPU and GPU (MOZYME with FORCE: 1e-6)")
    ap.add_argument("--thresh", type=float, default=None,
                    help="MOZYME LMO threshold THRESH for every FORCE/FORCETS run (MOZYME with FORCE: 1.D-15); "
                         "an SCF criterion below 1.D-5 needs a smaller one than the 1.D-13 of other jobs")
    ap.add_argument("--gpu-repeat", type=int, default=1, help="number of GPU FORCE runs (run-to-run spread)")
    ap.add_argument("--itry-scan", default="",
                    help="comma-separated ITRY values: GPU FORCE with SCFCRT=0.00001 (never met, so every SCF "
                         "runs exactly ITRY iterations) against the conventional reference, e.g. 30,60,100,200")
    ap.add_argument("--parallel-workers", type=int, default=0,
                    help="also compute the GPU FORCE (SCFCRT=--parallel-scfcrt) with this many MOPAC processes at once")
    ap.add_argument("--parallel-scfcrt", type=float, default=0.00001)
    ap.add_argument("--parallel-itry", type=int, default=200)
    ap.add_argument("--large-workers", type=int, default=0,
                    help="crambin: accurate FORCE (SCFCRT=--parallel-scfcrt, ITRY=--parallel-itry) with this many processes")
    ap.add_argument("--scfcrt-scan", default="",
                    help="comma-separated SCF criteria for a GPU-only FORCE convergence scan, e.g. 0.01,0.001,0.0001")
    ap.add_argument("--skip-large", action="store_true")
    ap.add_argument("--work-dir", type=Path, default=Path("mozyme_gpu_thermo"))
    ap.add_argument("--reuse", action="store_true", help="skip MOPAC runs whose .out already exists")
    a = ap.parse_args()
    mopac = a.mopac.resolve()
    work = a.work_dir.resolve()
    T = a.temperature
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        if not cond:
            failures.append(msg)

    def run(run_dir: Path, name: str, keywords: str, pdb: Path, env: dict | None) -> tuple[str, float, list[str]]:
        out = run_dir / f"{name}.out"
        if a.reuse and out.exists():
            text = out.read_text(errors="ignore")
            wall = re.findall(r"TOTAL JOB TIME:\s*([0-9.]+)", text)
            return text, float(wall[-1]) if wall else 0.0, md.STATUS_RE.findall(text)
        r = md.run_mopac(mopac, run_dir, name, keywords, name, [pdb], env)
        return r.out_text, r.wall, r.statuses

    def prepared_and_optimized(pdb_id: str) -> Path:
        base = work / pdb_id
        final = base / "opt" / "opt.pdb"
        if a.reuse and final.exists():
            return final
        hydro = md.prepare_pdb(mopac, get_pdb(pdb_id, base), base / "prepare")
        opt_pdb, r = md.optimize(mopac, hydro, base / "opt", cycles=a.opt_cycles, extra=f"GNORM={a.gnorm:g}")
        gn = re.findall(r"GRADIENT NORM\s*=\s*([-0-9.]+)", r.out_text)
        print(f"[{pdb_id}] optimization: final gradient norm {gn[-1] if gn else '?'} (target GNORM={a.gnorm:g})",
              flush=True)
        return opt_pdb

    def summary(label: str, th: dict, wall: float, statuses: list[str]) -> None:
        ok = statuses.count("success")
        n_imag = sum(1 for f in th["freqs"] if f < 0)
        print(f"     {label}: {wall:.1f} s, {len(th['freqs'])} frequencies ({n_imag} imaginary), ZPE {th.get('zpe')}, "
              f"H {th.get('H')} kcal/mol, S {th.get('S')} cal/(mol K), G(T) {th.get('G')} kcal/mol, "
              f"GPU SCF {ok}/{len(statuses)} resident", flush=True)

    def compare(label: str, cpu: dict, gpu: dict) -> None:
        check(len(cpu["freqs"]) == len(gpu["freqs"]) and len(cpu["freqs"]) > 0,
              f"{label}: same number of frequencies (CPU {len(cpu['freqs'])}, GPU {len(gpu['freqs'])})")
        if len(cpu["freqs"]) == len(gpu["freqs"]) and cpu["freqs"]:
            pairs = list(zip(sorted(cpu["freqs"]), sorted(gpu["freqs"])))
            hi = [(c, g) for c, g in pairs if c >= 100.0]
            rms_hi = (sum((c - g) ** 2 for c, g in hi) / len(hi)) ** 0.5 if hi else 0.0
            worst = max(pairs, key=lambda p: abs(p[0] - p[1]))
            mean_hi = sum(g - c for c, g in hi) / len(hi) if hi else 0.0
            lo = [(c, g) for c, g in pairs if c < 100.0]
            print(f"     {label}: frequencies >= 100 cm-1: mean shift {mean_hi:+.3f} cm-1, RMS diff {rms_hi:.3f} cm-1; "
                  f"largest diff "
                  f"{abs(worst[0] - worst[1]):.2f} cm-1 at {worst[0]:.1f}; {len(lo)} modes below 100 cm-1 "
                  f"(lowest CPU {pairs[0][0]:.1f} / GPU {pairs[0][1]:.1f})", flush=True)
            check(rms_hi <= 2.0, f"{label}: RMS frequency difference above 100 cm-1 = {rms_hi:.3f} <= 2 cm-1")
        for key, tol, unit in (("zpe", 0.1, "kcal/mol"), ("S", 1.0, "cal/(mol K)"), ("G", 0.3, "kcal/mol")):
            if cpu.get(key) is not None and gpu.get(key) is not None:
                d = gpu[key] - cpu[key]
                check(abs(d) <= tol, f"{label}: {key} GPU - CPU = {d:+.4f} {unit} (tolerance {tol})")

    ref_cache: dict = {}

    def reference() -> dict:
        """Conventional (non-MOZYME) SCF FORCE, fully converged at every displaced geometry."""
        if "ref" in ref_cache:
            return ref_cache["ref"]
        # Reference: the conventional (non-MOZYME) SCF, fully converged at every displaced geometry.
        # Converger aids: the plain conventional SCF oscillated on the GPU-optimized geometry
        # ("THE SCF CALCULATION FAILED" after 2542 s on Colab); PULAY first, then CAMP KING.
        ref, rwall = {"freqs": []}, 0.0
        for attempt, aids in enumerate(("PULAY ITRY=500", "CAMP KING ITRY=500"), start=1):
            conv_kw = f'PM7 FORCE THERMO({T:g}) {aids} GEO-OK GEO_DAT="{opt_pdb.name}"'
            rtext, rwall, _ = run(work / small / f"reference_conventional_cpu_{attempt}", "force", conv_kw,
                                  opt_pdb, CPU_ENV)
            ref = thermo(rtext, T)
            if ref["freqs"]:
                break
            print(f"     conventional reference with {aids}: SCF failed, "
                  f"{'trying the next converger' if attempt == 1 else 'no reference'}", flush=True)
        summary(f"{small} FORCE conventional SCF (reference, CPU)", ref, rwall, [])
        ref_cache["ref"] = ref
        return ref

    thresh_kw = f" THRESH={a.thresh:g}" if a.thresh else ""

    def keywords(itry: int, scfcrt: float) -> str:
        return md.BASE_KEYS.replace("ITRY=200", f"ITRY={itry}") + f" THERMO({T:g}) SCFCRT={scfcrt:g}" + thresh_kw

    def versus_reference(label: str, th: dict, wall: float) -> None:
        ref = reference()
        if len(th["freqs"]) == len(ref["freqs"]) and th["freqs"]:
            d = [x - y for y, x in zip(sorted(ref["freqs"]), sorted(th["freqs"])) if y >= 100.0]
            print(f"     {label} vs conventional: {wall:.1f} s, mean frequency shift {sum(d) / len(d):+.2f} cm-1, "
                  f"ZPE {th['zpe'] - ref['zpe']:+.3f}, S {th['S'] - ref['S']:+.3f} cal/(mol K), "
                  f"G {th['G'] - ref['G']:+.3f} kcal/mol", flush=True)

    # ---------------- small system: CPU vs GPU ----------------
    small = a.pdb_id
    opt_pdb = prepared_and_optimized(small)
    thermo_kw = f"THERMO({T:g})" + (f" SCFCRT={a.scfcrt:g}" if a.scfcrt else "") + thresh_kw
    # separate directories per SCF criterion and LMO threshold
    tag = (f"_scfcrt{a.scfcrt:g}" if a.scfcrt else "") + (f"_thresh{a.thresh:g}" if a.thresh else "")
    full_kw = f'{md.BASE_KEYS} FORCE {thermo_kw} GEO_DAT="{opt_pdb.name}"'
    ctext, cwall, _ = run(work / small / f"force_cpu{tag}", "force", full_kw, opt_pdb, CPU_ENV)
    gtext, gwall, gstat = run(work / small / f"force_gpu{tag}", "force", full_kw, opt_pdb, None)
    cth, gth = thermo(ctext, T), thermo(gtext, T)
    for rep in range(2, a.gpu_repeat + 1):
        rtext, rwall, rstat = run(work / small / f"force_gpu_rep{rep}{tag}", "force", full_kw, opt_pdb, None)
        rth = thermo(rtext, T)
        summary(f"{small} FORCE GPU repeat {rep}", rth, rwall, rstat)
        print(f"     {small} FORCE GPU repeat {rep} - first GPU run: ZPE {rth['zpe'] - gth['zpe']:+.4f}, "
              f"G {rth['G'] - gth['G']:+.4f} kcal/mol", flush=True)
    summary(f"{small} FORCE CPU", cth, cwall, [])
    summary(f"{small} FORCE GPU", gth, gwall, gstat)
    check(bool(gstat) and all(s == "success" for s in gstat),
          f"{small} FORCE: every GPU SCF resident ({gstat.count('success')}/{len(gstat)})")
    compare(f"{small} FORCE", cth, gth)
    if gwall > 0:
        print(f"     {small} FORCE: speed-up {cwall / gwall:.1f}x", flush=True)
    if a.scfcrt:
        # an explicit (tight) criterion: both against the conventional SCF
        versus_reference(f"{small} FORCE CPU", cth, cwall)
        versus_reference(f"{small} FORCE GPU", gth, gwall)
        ref = reference()
        compare(f"{small} FORCE GPU vs conventional", ref, gth)
    if a.precise:
        ptext, pwall, pstat = run(work / small / f"force_gpu_precise{tag}", "force", full_kw + " PRECISE", opt_pdb, None)
        pth = thermo(ptext, T)
        summary(f"{small} FORCE GPU PRECISE", pth, pwall, pstat)
        compare(f"{small} FORCE PRECISE(GPU) vs CPU", cth, pth)

    if a.scfcrt_scan:
        # How the harmonic frequencies depend on the SCF criterion of every displaced gradient
        # (MOZYME default 0.01 kcal/mol): GPU only, each value against the tightest one.
        scan = []
        for value in sorted((float(v) for v in a.scfcrt_scan.split(",") if v.strip()), reverse=True):
            kw = f'{md.BASE_KEYS} FORCE THERMO({T:g}) SCFCRT={value:g}{thresh_kw} GEO_DAT="{opt_pdb.name}"'
            text, wall, stat = run(work / small / f"scan_gpu_scfcrt{value:g}{tag}", "force", kw, opt_pdb, None)
            th = thermo(text, T)
            summary(f"{small} FORCE GPU SCFCRT={value:g}", th, wall, stat)
            scan.append((value, th))
        ref = reference()
        compare(f"{small} MOZYME GPU SCFCRT={scan[-1][0]:g} vs conventional", ref, scan[-1][1])
        for value, th in scan:
            if len(th["freqs"]) == len(ref["freqs"]) and th["freqs"]:
                d = [x - y for y, x in zip(sorted(ref["freqs"]), sorted(th["freqs"])) if y >= 100.0]
                print(f"     SCFCRT={value:g} vs conventional: mean frequency shift {sum(d) / len(d):+.2f} cm-1, "
                      f"ZPE {th['zpe'] - ref['zpe']:+.3f}, S {th['S'] - ref['S']:+.3f} cal/(mol K), "
                      f"G {th['G'] - ref['G']:+.3f} kcal/mol", flush=True)

    if a.itry_scan:
        # With an unreachable criterion every SCF runs exactly ITRY iterations: how many are needed?
        for itry in sorted(int(v) for v in a.itry_scan.split(",") if v.strip()):
            kw = keywords(itry, 0.00001) + f' FORCE GEO_DAT="{opt_pdb.name}"'
            text, wall, stat = run(work / small / f"itry_gpu_{itry}", "force", kw, opt_pdb, None)
            th = thermo(text, T)
            summary(f"{small} FORCE GPU ITRY={itry}", th, wall, stat)
            versus_reference(f"{small} ITRY={itry}", th, wall)

    if a.parallel_workers > 0:
        # sequential counterpart: the ITRY scan run with the same ITRY (SCFCRT=0.00001)
        seq_dir = work / small / f"itry_gpu_{a.parallel_itry}"
        out, tim = pf.parallel_force(mopac, opt_pdb, keywords(a.parallel_itry, a.parallel_scfcrt), a.parallel_workers,
                                     work / small / f"parallel_gpu_w{a.parallel_workers}")
        th = thermo(out.read_text(errors="ignore"), T)
        wall = tim["template_s"] + tim["workers_s"] + tim["final_s"]
        summary(f"{small} parallel FORCE GPU, {tim['workers']} workers", th, wall, [])
        versus_reference(f"{small} parallel FORCE", th, wall)
        seq_out = seq_dir / "force.out"
        if seq_out.exists():
            seq_text = seq_out.read_text(errors="ignore")
            seq_wall = re.findall(r"TOTAL JOB TIME:\s*([0-9.]+)", seq_text)
            seq = thermo(seq_text, T)
            # Informational: each worker starts its SCF from scratch, so with a fixed ITRY the two runs
            # differ at the level of the ITRY error itself; both are checked against the reference.
            if len(seq["freqs"]) == len(th["freqs"]) and th["freqs"]:
                print(f"     {small} parallel - sequential: ZPE {th['zpe'] - seq['zpe']:+.3f}, "
                      f"S {th['S'] - seq['S']:+.3f} cal/(mol K), G {th['G'] - seq['G']:+.3f} kcal/mol", flush=True)
            if seq_wall:
                print(f"     {small} parallel FORCE: {float(seq_wall[-1]):.1f} s sequential -> {wall:.1f} s "
                      f"({float(seq_wall[-1]) / wall:.1f}x)", flush=True)

    sel, _, rad = a.partial.partition("=")
    part_kw = f'{md.BASE_KEYS} FORCETS {thermo_kw} OPT("{sel}"={rad}) GEO_DAT="{opt_pdb.name}"'
    ctext, cwall, _ = run(work / small / f"forcets_cpu{tag}", "forcets", part_kw, opt_pdb, CPU_ENV)
    gtext, gwall, gstat = run(work / small / f"forcets_gpu{tag}", "forcets", part_kw, opt_pdb, None)
    cth, gth = thermo(ctext, T), thermo(gtext, T)
    summary(f"{small} FORCETS {a.partial} CPU", cth, cwall, [])
    summary(f"{small} FORCETS {a.partial} GPU", gth, gwall, gstat)
    compare(f"{small} FORCETS", cth, gth)

    # ---------------- large system: GPU timing ----------------
    if not a.skip_large:
        big = a.large_pdb_id
        big_pdb = prepared_and_optimized(big)
        natoms = sum(1 for l in big_pdb.read_text().splitlines() if l.startswith(("ATOM", "HETATM")))
        # timing run: LET so that FORCE continues if the optimization stopped above GNORM
        kw = f'{md.BASE_KEYS} FORCE {thermo_kw} LET GEO_DAT="{big_pdb.name}"'
        t0 = time.perf_counter()
        text, wall, stat = run(work / big / f"force_gpu{tag}", "force", kw, big_pdb, None)
        th = thermo(text, T)
        summary(f"{big} FORCE GPU ({natoms} atoms, {6 * natoms} gradients)", th, wall, stat)
        if wall > 0:
            print(f"     {big} FORCE GPU: {wall / (6 * natoms):.3f} s per gradient", flush=True)
        check(len(th["freqs"]) == 3 * natoms - 6, f"{big} FORCE GPU: {len(th['freqs'])} frequencies "
              f"(expected {3 * natoms - 6})")
        if a.large_workers > 0:
            out, tim = pf.parallel_force(mopac, big_pdb, keywords(a.parallel_itry, a.parallel_scfcrt) + " LET",
                                         a.large_workers, work / big / f"parallel_gpu_w{a.large_workers}")
            th = thermo(out.read_text(errors="ignore"), T)
            wall = tim["template_s"] + tim["workers_s"] + tim["final_s"]
            summary(f"{big} accurate FORCE GPU (SCFCRT={a.parallel_scfcrt:g}, ITRY={a.parallel_itry}), "
                    f"{tim['workers']} workers", th, wall, [])
        sel, _, rad = a.large_partial.partition("=")
        kw = f'{md.BASE_KEYS} FORCETS {thermo_kw} LET OPT("{sel}"={rad}) GEO_DAT="{big_pdb.name}"'
        text, wall, stat = run(work / big / f"forcets_gpu{tag}", "forcets", kw, big_pdb, None)
        summary(f"{big} FORCETS {a.large_partial} GPU", thermo(text, T), wall, stat)

    print(f"work dir: {work}")
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

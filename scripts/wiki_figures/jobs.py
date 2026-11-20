"""Calculations behind the figures of the wiki examples (scripts/wiki_figures/README.md).

    python jobs.py <mopac> <libmopac> <work dir> [--workers 10] [--only name,name] [--list]

Every job runs in <work dir>/<name>/ and is skipped when its done.json exists.  MOPAC runs on the CPU
(OMP_NUM_THREADS=1 per job unless the job says otherwise); OpenMM jobs use its CPU platform.
"""
import argparse, json, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHIG = REPO / "tests/data/mozyme_gpu_reference/1UAO_opt.pdb"
CRN_OPT = REPO / "tests/data/mozyme_gpu_reference/1CRN_opt.pdb"
CRN_H = REPO / "benchmarks/publication_inputs/hydrogenated/protein_crambin_1crn_hydrogenated.pdb"
CRN_CLEAN = REPO / "benchmarks/publication_inputs/pdb_clean/protein_crambin_1crn.pdb"
DNA = REPO / "benchmarks/publication_inputs/hydrogenated/dna_dodecamer_1bna_hydrogenated.pdb"
G6X = REPO / "benchmarks/publication_inputs/mop/protein_1g6x.pdb"
FLAKE = REPO / "benchmarks/publication_inputs/mop/material_graphene_nanoflake_c192h38.mop"
CELLS = REPO / "tests/data/materials/small"
METHODS = ["MNDO", "AM1", "PM3", "RM1", "PM6", "PM6-D3H4", "PM7"]


def deck(keywords, title, body=""):
    return f"{keywords}\n{title}\n\n{body}"


def cell_body(name):
    """atom and Tv lines of a repository cell (after its two header lines and the blank line)"""
    return "".join((CELLS / f"{name}.mop").read_text().splitlines(True)[3:])


def flake_body():
    return "".join(FLAKE.read_text().splitlines(True)[3:])


def pdb_job(kw, title, pdb=CHIG, **extra):
    return dict(steps=[dict(mop=deck(f'{kw} GEO_DAT="{pdb.name}"', title), files=[pdb])], **extra)


def shifted_pdb(pdb, atom, dx):
    """the PDB text with atom (1-based serial order) moved by dx Angstrom along x"""
    out, k = [], 0
    for line in pdb.read_text().splitlines(True):
        if line.startswith(("ATOM", "HETATM")):
            k += 1
            if k == atom:
                x = float(line[30:38]) + dx
                line = line[:30] + f"{x:8.3f}" + line[38:]
        out.append(line)
    return "".join(out)


def noh_text():
    """chignolin without its hydrogen atoms"""
    return "".join(l for l in CHIG.read_text().splitlines(True)
                   if not (l.startswith(("ATOM", "HETATM")) and l[76:78].strip() == "H"))


def water_dimer(r):
    """water dimer, donor O at the origin, O-H...O along x, acceptor O at r (Angstrom); planar, Cs"""
    import math
    a = math.radians(104.5)
    c, s = 0.96 * math.cos(a / 2), 0.96 * math.sin(a / 2)       # acceptor: bisector along +x, in the xz plane
    return "\n".join([
        "O  0.000 1 0.000 1 0.000 1",
        "H  0.960 1 0.000 1 0.000 1",
        f"H  {0.96 * math.cos(a):.4f} 1 {0.96 * math.sin(a):.4f} 1 0.000 1",
        f"O  {r:.4f} 1 0.000 1 0.000 1",
        f"H  {r + c:.4f} 1 0.000 1 {s:.4f} 1",
        f"H  {r + c:.4f} 1 0.000 1 {-s:.4f} 1", ""])


def jobs():
    J = {}
    # getting started
    J["sp_chig"] = pdb_job("PM7 MOZYME 1SCF", "Chignolin")
    J["grad_chig"] = pdb_job("PM7 MOZYME 1SCF GRADIENTS", "Chignolin, gradient")
    for c in ["1.D-2", "1.D-3", "1.D-4", "1.D-5", "1.D-6"]:
        J[f"scfcrt_{c}"] = pdb_job(f"PM7 MOZYME 1SCF SCFCRT={c} THRESH=1.D-15", f"Chignolin, SCFCRT={c}")
    J["sp_crambin"] = pdb_job("PM7 MOZYME 1SCF", "Crambin", CRN_H)
    J["sp_dna"] = pdb_job("PM7 MOZYME 1SCF", "DNA dodecamer", DNA)
    for m in METHODS:
        J[f"method_{m}"] = pdb_job(f"{m} MOZYME 1SCF", f"Chignolin, {m}")
    for n in [50, 100, 200, 400, 800]:
        J[f"water_{n}"] = dict(script=["python", str(REPO / "scripts/make_water_cluster.py"), str(n),
                                       "--out", "job.mop"], steps=[dict(mop=None, files=[])])
    # preparing structures
    J["prep_neutral"] = pdb_job("PM7 ADD-H PDBOUT NEWPDB 0SCF GEO-OK", "Crambin, neutral", CRN_CLEAN)
    J["prep_ionize"] = pdb_job("PM7 ADD-H SITE=(IONIZE) PDBOUT NEWPDB 0SCF GEO-OK", "Crambin, pH 7", CRN_CLEAN)
    J["check_1g6x"] = pdb_job("PM7 ADD-H PDBOUT NEWPDB 0SCF GEO-OK", "1G6X", G6X)
    J["let_1g6x"] = dict(steps=[
        dict(mop=deck('PM7 ADD-H PDBOUT NEWPDB 0SCF GEO-OK LET GEO_DAT="protein_1g6x.pdb"', "1G6X, LET"), files=[G6X]),
        dict(mop=deck('PM7 MOZYME 1SCF LET PL GEO_DAT="job.pdb"', "1G6X with its sulfates"), name="scf_with"),
        dict(mop=deck('PM7 MOZYME 1SCF PL GEO_DAT="nosulf.pdb"', "1G6X without the two half sulfates"), name="scf_without",
             prepare="nosulf")])
    J["workflow"] = dict(workflow=True, steps=[dict(mop=None, files=[CHIG])])
    # implicit water
    J["cosmo_chig"] = pdb_job("PM7 MOZYME 1SCF EPS=78.4", "Chignolin in water")
    J["swig_chig"] = pdb_job("PM7 MOZYME 1SCF EPS=78.4 SWIG", "Chignolin in water, SWIG")
    for k in range(-20, 21):
        dx = k * 0.001
        for tag, kw in (("std", "EPS=78.4"), ("swig", "EPS=78.4 SWIG")):
            J[f"cosmoscan_{tag}_{k:+03d}"] = dict(steps=[dict(
                mop=deck(f'PM7 MOZYME 1SCF {kw} GEO_DAT="scan.pdb"', f"x + {dx:.3f}"),
                text_files={"scan.pdb": shifted_pdb(CHIG, 50, dx)})])
    J["cosmograd_std"] = pdb_job("PM7 MOZYME 1SCF GRADIENTS EPS=78.4", "Chignolin in water, gradient")
    J["cosmograd_swig"] = pdb_job("PM7 MOZYME 1SCF GRADIENTS EPS=78.4 SWIG", "Chignolin in water, SWIG, gradient")
    J["wopt_pm7"] = pdb_job("PM7 MOZYME EPS=78.4 SWIG TRUST CYCLES=40 PDBOUT", "Chignolin optimized in water")
    J["wopt_pm6"] = pdb_job("PM6-D3H4 MOZYME EPS=78.4 SWIG TRUST CYCLES=40 PDBOUT", "Chignolin, PM6-D3H4, water")
    J["cosmo_dna"] = pdb_job("PM7 MOZYME 1SCF EPS=78.4", "DNA in water", DNA)
    # geometry optimization
    J["opt_trust"] = pdb_job("PM7 MOZYME TRUST CYCLES=80 PDBOUT", "Chignolin, TRUST")
    J["opt_lbfgs"] = pdb_job("PM7 MOZYME CYCLES=80 PDBOUT", "Chignolin, L-BFGS-B")
    for k in range(0, 61):
        r = 5.0 + 0.06 * k
        for tag, kw in (("orig", ""), ("smooth", " HB-SMOOTH")):
            J[f"hbscan_{tag}_{k:02d}"] = dict(steps=[dict(mop=deck(f"PM7 1SCF DISP GEO-OK{kw}", f"O-O {r:.2f}",
                                                                   water_dimer(r)))])
    J["opt_restart"] = dict(steps=[
        dict(mop=deck(f'PM7 MOZYME TRUST CYCLES=20 GEO_DAT="{CHIG.name}" PDBOUT', "first part"), files=[CHIG],
             name="part1"),
        dict(mop=deck(f'PM7 MOZYME TRUST CYCLES=20 RESTART GEO_DAT="{CHIG.name}" PDBOUT', "continued"),
             name="part2"),
        dict(mop=deck(f'PM7 MOZYME 1SCF RESTART PDBOUT GEO_DAT="{CHIG.name}"', "final geometry"), name="final")])
    # optimizations from chignolin with hydrogens added again to its heavy atoms (GNORM ~245), as the workflow does
    prep = dict(mop=deck('PM7 ADD-H PDBOUT NEWPDB 0SCF GEO-OK GEO_DAT="noH.pdb"', "chignolin, hydrogens added"),
                text_files={"noH.pdb": noh_text()}, name="prep")
    J["opt2_trust"] = dict(steps=[prep, dict(mop=deck('PM7 MOZYME TRUST CYCLES=150 GEO_DAT="prep.pdb" PDBOUT',
                                                       "TRUST"))])
    J["opt2_lbfgs"] = dict(steps=[prep, dict(mop=deck('PM7 MOZYME CYCLES=150 GEO_DAT="prep.pdb" PDBOUT',
                                                       "L-BFGS-B"))])
    J["opt2_restart"] = dict(steps=[
        prep,
        dict(mop=deck('PM7 MOZYME TRUST CYCLES=25 GEO_DAT="prep.pdb" PDBOUT', "first part"), name="part1"),
        dict(mop=deck('PM7 MOZYME TRUST CYCLES=25 RESTART GEO_DAT="prep.pdb" PDBOUT', "continued"), name="part2"),
        dict(mop=deck('PM7 MOZYME 1SCF RESTART PDBOUT GEO_DAT="prep.pdb"', "final geometry"), name="final")])
    J["opt_part"] = pdb_job('PM7 MOZYME TRUST CYCLES=15 OPT("A46"=4) PDBOUT', "Crambin, around residue 46", CRN_OPT)
    # molecular dynamics (chignolin)
    md = "PM7 MOZYME DRC TEMPERATURE=300 T-PRIORITY=0.5"
    J["md_nve"] = pdb_job(f"{md} CYCLES=120", "NVE 300 K")
    J["md_bussi100"] = pdb_job(f"{md} BUSSI=100 CYCLES=120", "NVT, BUSSI=100")
    J["md_bussi10"] = pdb_job(f"{md} BUSSI=10 CYCLES=120", "NVT, BUSSI=10")
    for name, seed in (("md_seed42a", 42), ("md_seed42b", 42), ("md_seed43", 43),
                       ("md_rep1", 1), ("md_rep2", 2), ("md_rep3", 3)):
        J[name] = pdb_job(f"{md} BUSSI=100 SEED={seed} CYCLES=60", f"NVT, seed {seed}")
    J["md_water"] = pdb_job("PM7 MOZYME EPS=78.4 SWIG HB-SMOOTH DRC TEMPERATURE=310 BUSSI=100 SEED=7 "
                            "T-PRIORITY=1 CYCLES=30", "Chignolin in water at 310 K")
    J["md_heat"] = dict(heating=[100, 200, 300], steps=[dict(mop=None, files=[CHIG])])
    J["md_kinetic"] = pdb_job("PM7 MOZYME DRC KINETIC=20 T-PRIORITY=0.5 CYCLES=60", "Classic DRC")
    J["md_crambin"] = pdb_job("PM7 MOZYME DRC TEMPERATURE=300 SEED=11 T-PRIORITY=0.5 CYCLES=6", "Crambin NVE",
                              CRN_OPT)
    # vibrations
    J["vib_force"] = pdb_job("PM7 MOZYME FORCE THERMO(298)", "Chignolin, frequencies")
    J["vib_parallel"] = dict(parallel_force=4, steps=[dict(mop=None, files=[CHIG])])
    # periodic solids
    for name, kw in (("sol_si", "PM7 1SCF GRADIENTS GEO-OK"), ("sol_si_shift80", "PM7 1SCF SHIFT=80 GEO-OK PL"),
                     ("sol_si_default", "PM7 1SCF GEO-OK PL")):
        J[name] = dict(steps=[dict(mop=deck(kw, name, cell_body("silicon")))])
    J["sol_graphene"] = dict(steps=[dict(mop=deck("PM7 1SCF GEO-OK PL", "graphene", cell_body("graphene")))])
    J["sol_graphene_smear"] = dict(steps=[dict(mop=deck("PM7 1SCF SMEAR=0.1 GEO-OK PL", "graphene SMEAR",
                                                        cell_body("graphene")))])
    J["sol_diamond_grad"] = dict(steps=[dict(mop=deck("PM7 1SCF GRADIENTS GEO-OK", "diamond", cell_body("diamond")))])
    J["sol_diamond_precise"] = dict(steps=[dict(mop=deck("PM7 1SCF GRADIENTS PRECISE GEO-OK", "diamond PRECISE",
                                                         cell_body("diamond")))])
    J["sol_rutile_opt"] = dict(steps=[dict(mop=deck("PM7 PRECISE CYCLES=30 GEO-OK", "rutile", cell_body("rutile")))])
    J["sol_diamond_force"] = dict(steps=[dict(mop=deck("PM7 FORCE SCFCRT=1.D-10 GEO-OK", "diamond FORCE",
                                                       cell_body("diamond")))], threads=4)
    J["sol_diamond_drc"] = dict(steps=[dict(mop=deck("PM7 DRC TEMPERATURE=300 SEED=7 CYCLES=60 GEO-OK", "diamond DRC",
                                                     cell_body("diamond")))])
    J["sol_urea_moz"] = dict(steps=[dict(mop=deck("PM7 MOZYME 1SCF GEO-OK", "urea MOZYME", cell_body("urea")))])
    J["sol_urea_conv"] = dict(steps=[dict(mop=deck("PM7 1SCF GEO-OK", "urea", cell_body("urea")))])
    for v in ("0", "1"):
        J[f"sol_quartz_pdiag{v}"] = dict(steps=[dict(mop=deck("PM7 1SCF GEO-OK PL", f"quartz PDIAG={v}",
                                                              cell_body("quartz")))], env={"MOPAC_PDIAG": v})
    # conventional SCF of large molecules
    J["conv_flake"] = dict(steps=[dict(mop=deck("PM7 XYZ 1SCF PULAY PL BONDS GEO-OK", "flake", flake_body()))],
                           threads=4)
    J["conv_uhf"] = pdb_job("PM7 UHF 1SCF GRADIENTS CHARGE=1 DOUBLET", "Chignolin cation, UHF")
    J["conv_he"] = pdb_job("PM7 RHF 1SCF GRADIENTS CHARGE=1 DOUBLET GEO-OK", "half-electron, analytic")
    J["conv_he_num"] = pdb_job("PM7 RHF 1SCF GRADIENTS CHARGE=1 DOUBLET GEO-OK NOANCI", "half-electron, numerical")
    J["he_fd"] = dict(fd=[0, 54, 120, 200, 260, 300, 336, 341, 348, 355, 400, 419], steps=[dict(mop=None, files=[CHIG])],
                      threads=1)
    # QM/MM
    J["qmmm_sc"] = dict(qmmm=["--qm-residues", "2", "--qm-sidechain", "--steps", "200", "--nve", "400",
                              "--dt", "0.5", "--platform", "CPU", "--mm-equil-ps", "1"],
                        steps=[dict(mop=None, files=[CHIG])], threads=4)
    J["qmmm_pep"] = dict(qmmm=["--qm-residues", "1,2,3,4,5,6,7,8,9,10", "--padding", "2.5", "--ron", "6",
                               "--roff", "8", "--tolerance", "0.001", "--steps", "100", "--nve", "200",
                               "--platform", "CPU", "--mm-equil-ps", "1"],
                         steps=[dict(mop=None, files=[CHIG])], threads=4)
    for k, d in enumerate([4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]):
        J[f"mmq_{k:02d}"] = dict(mmq=d, steps=[dict(mop=None, files=[CHIG])])
    # an enzyme reaction: Ser195 OG of trypsin attacks the carbonyl C of Lys15 of BPTI (PDB 2PTC), QM/MM scan
    J["enzyme_scan"] = dict(enzyme=["--qm-residues", "E:57:sc,E:102:sc,E:195:sc,I:15,I:16", "--variants", "E:57=HID",
                                    "--scan", "E:195:OG-I:15:C", "--to", "1.5", "--step", "0.1", "--back",
                                    "--watch", "E:195:OG-E:195:HG,E:195:HG-E:57:NE2,I:15:C-I:15:O,I:15:C-I:16:N,"
                                               "E:57:HD1-E:102:OD2",
                                    "--min-iter", "150", "--min-iter-start", "600", "--mm-equil-ps", "20",
                                    "--platform", "OpenCL"],
                            steps=[dict(mop=None)], threads=6)
    J["times_chig"] = pdb_job("PM7 MOZYME 1SCF TIMES", "Chignolin, stage times")
    J["pme_check"] = dict(script=["python", str(REPO / "tests/check_qmmm_pme_ewald.py")], steps=[dict(mop=None)])
    return J


# ----------------------------------------------------------------------------------------------------------
def run(name, job, mopac, libmopac, work, python):
    d = work / name
    if (d / "done.json").exists():
        return name, "cached", 0.0
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(job.get("threads", 1)), OPENBLAS_NUM_THREADS=str(job.get("threads", 1)),
               **job.get("env", {}))
    t0 = time.time()
    log = open(d / "run.log", "w")

    def sh(cmd):
        return subprocess.run(cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT).returncode

    for st in job["steps"]:
        for f in st.get("files", []):
            shutil.copy(f, d / Path(f).name)
        for fn, text in st.get("text_files", {}).items():
            (d / fn).write_text(text)
    rc = 0
    if "script" in job:
        cmd = [python if c == "python" else c for c in job["script"]]
        rc = sh(cmd)
        if rc == 0 and (d / "job.mop").exists():
            rc = sh([mopac, "job.mop"])
    elif job.get("workflow"):
        noh = "".join(l for l in CHIG.read_text().splitlines(True)
                      if not (l.startswith(("ATOM", "HETATM")) and l[76:78].strip() == "H"))
        (d / "1UAO_noH.pdb").write_text(noh)
        rc = sh([python, str(REPO / "scripts/mozyme_md_workflow.py"), mopac, "1UAO_noH.pdb", "--work-dir", "wf",
                 "--temperature", "300", "--opt-cycles", "30", "--nvt-points", "60", "--nve-points", "60"])
    elif "heating" in job:
        sys.path.insert(0, str(REPO / "scripts"))
        import mozyme_md_workflow as wf
        geo = CHIG.name
        for k, temp in enumerate(job["heating"]):
            sub = f"stage{k}"
            (d / f"{sub}.mop").write_text(deck(f'PM7 MOZYME DRC TEMPERATURE={temp} BUSSI=20 T-PRIORITY=0.5 '
                                               f'CYCLES=40 GEO_DAT="{geo}"', f"Heating, {temp} K"))
            rc = sh([mopac, f"{sub}.mop"])
            if rc:
                break
            geo = f"{sub}_last.pdb"
            wf.last_frame_pdb(d / f"{sub}.xyz", d / CHIG.name, d / geo)
    elif "parallel_force" in job:
        rc = sh([python, str(REPO / "scripts/mozyme_parallel_force.py"), mopac, CHIG.name, "--keywords",
                 "PM7 MOZYME GEO-OK THERMO(298)", "--workers", str(job["parallel_force"]), "--work-dir", "pf",
                 "--gpus", "none"])
    elif "enzyme" in job:
        import urllib.request
        urllib.request.urlretrieve("https://files.rcsb.org/download/2PTC.pdb", d / "2PTC.pdb")
        rc = sh([python, str(REPO / "scripts/qmmm/qmmm_scan.py"), "2PTC.pdb", "--libmopac", libmopac] + job["enzyme"])
    elif "qmmm" in job:
        rc = sh([python, str(REPO / "scripts/qmmm/run_qmmm_md.py"), CHIG.name, "--libmopac", libmopac]
                + job["qmmm"])
    elif "fd" in job:
        # central differences (+-0.005 A) of the half-electron heat for selected Cartesian coordinates
        import re as _re
        src = CHIG.read_text().splitlines(True)
        res = {}
        for k in job["fd"]:
            for sgn in (1, -1):
                out, n = [], 0
                for l in src:
                    if l.startswith(("ATOM", "HETATM")):
                        n += 1
                        if n == k // 3 + 1:
                            col = 30 + 8 * (k % 3)
                            l = l[:col] + f"{float(l[col:col + 8]) + sgn * 0.005:8.3f}" + l[col + 8:]
                    out.append(l)
                stem = f"c{k}_{'p' if sgn > 0 else 'm'}"
                (d / f"{stem}.pdb").write_text("".join(out))
                (d / f"{stem}.mop").write_text(deck(f'PM7 RHF 1SCF CHARGE=1 DOUBLET GEO-OK GEO_DAT="{stem}.pdb"', stem))
                rc |= sh([mopac, f"{stem}.mop"])
                res[stem] = float(_re.findall(r"FINAL HEAT OF FORMATION =\s+(-?[\d.]+)", (d / f"{stem}.out").read_text())[-1])
        (d / "fd.json").write_text(json.dumps({str(k): (res[f"c{k}_p"] - res[f"c{k}_m"]) / 0.01 for k in job["fd"]}))
    elif "mmq" in job:
        import numpy as np
        xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in CHIG.read_text().splitlines()
                        if l.startswith(("ATOM", "HETATM"))])
        c = xyz.mean(axis=0)
        u = np.array([1.0, 0.0, 0.0])
        # the TIP3P oxygen at distance d (Angstrom) from the nearest peptide atom, along +x from the centre
        s = 0.0
        while np.min(np.linalg.norm(xyz - (c + s * u), axis=1)) < job["mmq"]:
            s += 0.01
        o = c + s * u
        q = [(o, -0.834), (o + [0.586, 0.757, 0.0], 0.417), (o + [0.586, -0.757, 0.0], 0.417)]
        (d / "charges.txt").write_text("".join(f"{p[0]:10.4f} {p[1]:10.4f} {p[2]:10.4f} {v:8.3f}\n" for p, v in q))
        (d / "job.mop").write_text(deck(f'PM7 MOZYME 1SCF GRADIENTS MM_CHARGES="charges.txt" GEO_DAT="{CHIG.name}"',
                                        f"water {job['mmq']} A from the peptide"))
        rc = sh([mopac, "job.mop"])
    else:
        for k, st in enumerate(job["steps"]):
            stem = st.get("name", "job")
            if st.get("prepare") == "nosulf":
                # the two incomplete sulfates that the input chemistry check reports (SO4 A 64 and 65)
                keep = [l for l in (d / "job.pdb").read_text().splitlines(True)
                        if not (l[17:20] == "SO4" and l[22:26].strip() in ("64", "65"))]
                (d / "nosulf.pdb").write_text("".join(keep))
            (d / f"{stem}.mop").write_text(st["mop"])
            if "RESTART" in st["mop"] and k > 0:
                # the restart and density files of the previous step, under this step's name
                prev = job["steps"][k - 1].get("name", "job")
                for ext in (".res", ".den"):
                    if (d / f"{prev}{ext}").exists():
                        shutil.copy(d / f"{prev}{ext}", d / f"{stem}{ext}")
            rc = sh([mopac, f"{stem}.mop"])
    dt = time.time() - t0
    (d / "done.json").write_text(json.dumps(dict(rc=rc, seconds=round(dt, 2), threads=job.get("threads", 1))))
    return name, rc, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mopac")
    ap.add_argument("libmopac")
    ap.add_argument("work", type=Path)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", default="")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    J = jobs()
    if a.only:
        want = a.only.split(",")
        J = {k: v for k, v in J.items() if any(k == w or (w.endswith("*") and k.startswith(w[:-1])) for w in want)}
    if a.list:
        print("\n".join(J))
        return
    a.work.mkdir(parents=True, exist_ok=True)
    # long jobs first
    order = ["vib_force", "vib_parallel", "conv_he_num", "sol_diamond_force", "md_water", "qmmm_pep", "md_crambin",
             "opt_part", "wopt_pm6", "wopt_pm7", "workflow", "let_1g6x", "qmmm_sc", "conv_flake"]
    names = [n for n in order if n in J] + [n for n in J if n not in order]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run, n, J[n], a.mopac, a.libmopac, a.work.resolve(), a.python): n for n in names}
        for f in as_completed(futs):
            try:
                n, rc, dt = f.result()
                print(f"{n}: rc={rc} {dt:.0f} s", flush=True)
            except Exception as e:
                print(f"{futs[f]}: EXCEPTION {e!r}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a fail-closed MOZYME strict resident-SCF GPU smoke test."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


STRICT_GPU_ENV = {
    "MOPAC_FORCEGPU": "1",
    "MOZYME_GPU_FORCE": "1",
    "MOPAC_GPU_PROFILE": "1",
    "MOPAC_MOZYME_SECTION_PROFILE": "1",
    "MOPAC_MOZYME_RESIDENT_FOCK_GPU": "1",
    "MOPAC_MOZYME_DENSITY_BATCH_GPU": "1",
    "MOPAC_MOZYME_CNVGZ_GPU": "1",
    "MOPAC_MOZYME_HELECZ_GPU": "1",
    "MOPAC_MOZYME_EIMP_GPU": "1",
    "MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU": "1",
    "MOPAC_MOZYME_DIAGG1_AOCC_GPU": "1",
    "MOPAC_MOZYME_DIAGG1_AVIR_GPU": "1",
    "MOPAC_MOZYME_DIAGG2_ROTATE_GPU": "1",
    "MOPAC_MOZYME_DIAGG2_ROTPREP_GPU": "1",
    "MOPAC_MOZYME_ISITSC_GPU": "1",
    "MOPAC_MOZYME_MAKVEC_GPU": "1",
    "MOPAC_MOZYME_RELOCAL_GPU": "1",
    "MOPAC_MOZYME_REORTH_GPU": "1",
    "MOPAC_MOZYME_TIDY_GPU": "1",
    "MOPAC_MOZYME_SCF_EXPERIMENTAL": "1",
    "MOPAC_MOZYME_SCF_GPU": "1",
    "MOPAC_MOZYME_SCF_STRICT_RESIDENT": "1",
    "MOPAC_MOZYME_FULL_SCF_GPU": "1",
    "MOPAC_MOZYME_GPU_STRICT": "1",
    "MOPAC_MOZYME_RESIDENT_SCF": "1",
    "MOPAC_MOZYME_SCF_EARLY_PROBE": "0",
    "MOPAC_MOZYME_SCF_FORCE_FINAL_REORTH": "1",
}

FATAL_STATUS_RE = re.compile(r"\[MOZYME GPU SCF\].*status=(fallback_cpu|strict_abort|resident_step)\b")
HELPER_FATAL_STATUS_RE = re.compile(
    r"\[MOZYME GPU (?:makvec|relocal|reorth|setupk|density(?:_batch)?|cnvgz|helecz|eimp|"
    r"diagg1_(?:aocc|avir|construct)|diagg2_(?:rotprep|rotate)|isitsc|tidy)\][^\n]*"
    r"(?:\bstatus\s*=\s*)?(fallback_cpu|strict_abort)\b",
    re.IGNORECASE,
)
MOZYME_TIDY_RE = re.compile(
    r"\[MOZYME GPU tidy\]\s+status=(success|fallback_cpu|strict_abort)\s+mode=([^\s]+)",
    re.IGNORECASE,
)
FOCK_FAMILY_FALLBACK_RE = re.compile(
    r"\[MOZYME GPU (?:fock1|fock2|fock1_batch|fock2_4x1_batch)\]\s+.*\bfallback\b",
    re.IGNORECASE,
)
RESIDENT_FOCK_FALLBACK_REAL_PAIRS_RE = re.compile(
    r"\[MOZYME GPU resident_fock\]\s+fallback_real_pairs\s+total=\s*(\d+)\b",
    re.IGNORECASE,
)
RESIDENT_FOCK_CPU_POINT_PAIRS_RE = re.compile(
    r"\[MOZYME GPU resident_fock\][^\n]*\bcpu_point_pairs\s*=\s*(\d+)\b",
    re.IGNORECASE,
)
SUCCESS_STATUS_RE = re.compile(
    r"\[MOZYME GPU SCF\].*status=success\b.*code=\s*(-?\d+)\b.*"
    r"ready=\s*(-?\d+)\b.*resident=\s*(-?\d+)\b",
    re.IGNORECASE,
)
GPU_ERROR_MARKERS = (
    "[GPU ERROR]",
    "ACCURACY_FAIL",
    "BENCH_FAIL",
    "cuBLAS status",
    "cuSOLVER status",
    "CUBLAS_STATUS_",
    "CUSOLVER_STATUS_",
    "CUDA error",
    "cudaError",
    "illegal memory access",
    "device-side assert",
    "device assert",
    "misaligned address",
    "unspecified launch failure",
    "out of memory",
    "Segmentation fault",
    "SIGSEGV",
    "core dumped",
)
STAGE_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+stage_completed=\s*(\d+)\s+"
    r"stage_required=\s*(\d+)\s+stage_missing=\s*(\d+)"
)
STAGE_NAMES_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+stage_completed_names=\s*([^\s]+)\s+"
    r"stage_missing_names=\s*([^\s]+)"
)
STRICT_PROOF_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+strict_proof\s+strict_resident=\s*(\d+)\s+"
    r"no_fallback_required=\s*(\d+)\s+full_stage_mask=\s*(\d+)\s+"
    r"resident_decision_complete=\s*(\d+)\s+strict_host_syncs=\s*(\d+)\s+"
    r"strict_control_polls=\s*(\d+)"
)
RESIDENT_DECISION_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+resident_decision=\s*(-?\d+)"
)
RESIDENT_FOCK_PLAN_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+resident_fock_plan_id=\s*(-?\d+)\s+"
    r"resident_fock_plan_full_coverage=\s*(\d+)"
    r"(?:\s+resident_fock_plan_partial_coverage=\s*(\d+)\s+"
    r"resident_fock_plan_required_mask=\s*(\d+)\s+"
    r"resident_fock_plan_covered_mask=\s*(\d+))?"
)
FINAL_DENSITY_RE = re.compile(r"\[MOZYME GPU SCF\]\s+final_density=current_resident\b")
HOST_COMMIT_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+host_commit_only=1\s+phase=final_publication\s+"
    r"arrays=\s*(\d+)\s+bytes=\s*(\d+)\s+cosmo=\s*(\d+)",
    re.IGNORECASE,
)
FINAL_PUBLICATION_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+final_publication_done=\s*(\d+)\s+"
    r"arrays=\s*(\d+)\s+bytes=\s*(\d+)\s+cosmo=\s*(\d+)",
    re.IGNORECASE,
)
DEVICE_ITERATIONS_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+device_id=\s*(-?\d+)\s+natoms=\s*\d+\s+"
    r"norbs=\s*\d+\s+iterations=\s*(-?\d+)",
    re.IGNORECASE,
)
CNVGZ_ACTIVITY_RE = re.compile(
    r"\[MOZYME GPU SCF\](?=[^\n]*\bcnvgz_active_calls\s*=\s*-?\d+\b)"
    r"(?=[^\n]*\bcnvgz_noop_calls\s*=\s*-?\d+\b)[^\n]*",
    re.IGNORECASE,
)
MARKER_FIELD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s]+)")
RESIDENT_STAGE_CALLS_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+resident_stage_calls\s+upload=\s*(\d+)\s+"
    r"eimp=\s*(\d+)\s+diagg=\s*(\d+)\s+density=\s*(\d+)\s+"
    r"fock=\s*(\d+)\s+cnvgz=\s*(\d+)\s+helecz=\s*(\d+)\s+"
    r"isitsc=\s*(\d+)\s+addhb=\s*(\d+)\s+check=\s*(\d+)",
    re.IGNORECASE,
)
SPARSE_FOCK_RUN_RE = re.compile(
    r"\[GPU\]\s+profile\s+mozyme_sparse_fock_run\s+one=(\d+)\s+"
    r"pair=(\d+)\s+pair4x1=(\d+)\s+point=(\d+)\s+"
    r"point_dipole=(\d+)\s+point_monopole=(\d+)\s+"
    r"ms=([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_RELOCAL_RE = re.compile(
    r"\[MOZYME GPU relocal\]\s+status=(success|fallback_cpu|strict_abort)\s+kind=([^\s]+)",
    re.IGNORECASE,
)
MOZYME_REORTH_RE = re.compile(
    r"\[MOZYME GPU reorth\]\s+status=(success|fallback_cpu|strict_abort)\b([^\n]*)",
    re.IGNORECASE,
)
MOZYME_SECTION_RE = re.compile(
    r"\[PROFILE\]\s+MOZYME_SECTION\s+name=(?:\"([^\"]+)\"|([^\s]+))\s+"
    r"calls=(\d+)\s+ms=([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_STRICT_PROOF_ALLOWED_SECTION_NAMES = {
    "iter_resident_scf_boundary",
    "iter_density_final_resident",
}
MOZYME_SCF_STAGE_FULL = 1023
MOZYME_SCF_STAGE_FULL_NAMES = "upload+eimp+diagg+density+fock+cnvgz+helecz+isitsc+addhb+check"
MOZYME_SCF_STAGE_NAMES = (
    "upload",
    "eimp",
    "diagg",
    "density",
    "fock",
    "cnvgz",
    "helecz",
    "isitsc",
    "addhb",
    "check",
)
MOZYME_SCF_RESIDENT_DECISION_COMPLETE = 1


def cnvgz_active_noop_ready(active_calls: int | None, noop_calls: int | None) -> bool:
    return (
        active_calls is not None
        and noop_calls is not None
        and active_calls >= 0
        and noop_calls >= 0
        and active_calls + noop_calls > 0
    )


def marker_fields(line: str) -> dict[str, str]:
    return dict(MARKER_FIELD_RE.findall(line or ""))


def write_strict_input(src: Path, dst: Path) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"empty MOPAC input: {src}")
    forced_keywords = (
        "MOZYME_GPU MOZYME_MINBLK=1 PULAY SHIFT=-50 ITRY=40 "
        "RE-LOCAL=1 REORTH NOCOMMENTS"
    )
    lines[0] = f"{lines[0]} {forced_keywords}"
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strict_disallowed_section_names(text: str) -> list[str]:
    relocal_counts: dict[str, int] = {}
    for status, kind in MOZYME_RELOCAL_RE.findall(text):
        if status.lower() == "success":
            key = kind.upper()
            relocal_counts[key] = relocal_counts.get(key, 0) + 1
    tidy_counts: dict[str, int] = {}
    for status, mode in MOZYME_TIDY_RE.findall(text):
        if status.lower() == "success":
            key = mode.lower()
            tidy_counts[key] = tidy_counts.get(key, 0) + 1

    disallowed: set[str] = set()
    for quoted_name, bare_name, calls_text, _ms_text in MOZYME_SECTION_RE.findall(text):
        name = quoted_name or bare_name
        calls = int(calls_text)
        if calls <= 0:
            continue
        allowed = name in MOZYME_STRICT_PROOF_ALLOWED_SECTION_NAMES
        if name == "iter_reloc_occ":
            allowed = relocal_counts.get("OCCUPIED", 0) >= calls
        elif name == "iter_reloc_virt":
            allowed = relocal_counts.get("VIRTUAL", 0) >= calls
        elif name == "iter_tidy_occ":
            allowed = tidy_counts.get("occupied", 0) >= calls
        elif name == "iter_tidy_virt":
            allowed = tidy_counts.get("virtual", 0) >= calls
        if not allowed:
            disallowed.add(name)
    return sorted(disallowed)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_mozyme_strict_resident_scf.py <mopac_exe> <input.mop>", file=sys.stderr)
        return 2

    mopac_exe = Path(sys.argv[1]).resolve()
    input_path = Path(sys.argv[2]).resolve()
    if not mopac_exe.exists():
        print(f"mopac executable not found: {mopac_exe}", file=sys.stderr)
        return 2
    if not input_path.exists():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="mopac_strict_resident_scf_") as tmp:
        workdir = Path(tmp)
        test_input = workdir / "strict_resident_scf.mop"
        write_strict_input(input_path, test_input)

        env = os.environ.copy()
        env.update(STRICT_GPU_ENV)
        proc = subprocess.run(
            [str(mopac_exe), test_input.name],
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        sys.stdout.write(proc.stdout)

        out_path = test_input.with_suffix(".out")
        out_text = out_path.read_text(encoding="utf-8", errors="ignore") if out_path.exists() else ""
        combined = proc.stdout + "\n" + out_text
        lowered = combined.lower()

        artifact_dir = Path.cwd() / "mozyme_strict_resident_scf_failure_artifacts"
        if proc.returncode != 0:
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            shutil.copytree(workdir, artifact_dir)
            print(f"strict resident SCF: MOPAC failed; artifacts copied to {artifact_dir}", file=sys.stderr)
            return proc.returncode

        gpu_error_markers = [marker for marker in GPU_ERROR_MARKERS if marker.lower() in lowered]
        if gpu_error_markers:
            print(
                "strict resident SCF: GPU error marker(s) appeared: "
                + ", ".join(gpu_error_markers),
                file=sys.stderr,
            )
            return 1
        fatal = FATAL_STATUS_RE.search(combined)
        if fatal:
            print(f"strict resident SCF: forbidden status={fatal.group(1)} marker appeared", file=sys.stderr)
            return 1
        helper_fatal = HELPER_FATAL_STATUS_RE.search(combined)
        if helper_fatal:
            print(
                "strict resident SCF: forbidden helper marker appeared: "
                f"{helper_fatal.group(0)}",
                file=sys.stderr,
            )
            return 1
        tidy_success_modes = {
            mode.lower()
            for status, mode in MOZYME_TIDY_RE.findall(combined)
            if status.lower() == "success"
        }
        if "occupied" not in tidy_success_modes:
            print("strict resident SCF: occupied GPU TIDY success marker was not found", file=sys.stderr)
            return 1
        if "virtual" not in tidy_success_modes:
            print("strict resident SCF: virtual GPU TIDY success marker was not found", file=sys.stderr)
            return 1
        fock_fallback = FOCK_FAMILY_FALLBACK_RE.search(combined)
        if fock_fallback:
            print(
                "strict resident SCF: forbidden MOZYME GPU Fock fallback marker appeared: "
                f"{fock_fallback.group(0)}",
                file=sys.stderr,
            )
            return 1
        resident_fock_fallback_pairs = [
            int(match) for match in RESIDENT_FOCK_FALLBACK_REAL_PAIRS_RE.findall(combined)
        ]
        if any(count > 0 for count in resident_fock_fallback_pairs):
            print(
                "strict resident SCF: resident_fock reported CPU fallback real pairs: "
                + ", ".join(str(count) for count in resident_fock_fallback_pairs),
                file=sys.stderr,
            )
            return 1
        resident_fock_cpu_point_pairs = [
            int(match) for match in RESIDENT_FOCK_CPU_POINT_PAIRS_RE.findall(combined)
        ]
        if any(count > 0 for count in resident_fock_cpu_point_pairs):
            print(
                "strict resident SCF: resident_fock reported CPU point pairs: "
                + ", ".join(str(count) for count in resident_fock_cpu_point_pairs),
                file=sys.stderr,
            )
            return 1
        if "backend_cpu_boundary" in combined:
            print("strict resident SCF: backend_cpu_boundary appeared in output", file=sys.stderr)
            return 1
        success_matches = SUCCESS_STATUS_RE.findall(combined)
        if not success_matches:
            print("strict resident SCF: success code=0 ready=1 resident=1 marker was not found", file=sys.stderr)
            return 1
        backend_code, backend_ready, backend_resident = map(int, success_matches[-1])
        if backend_code != 0 or backend_ready != 1 or backend_resident != 1:
            print(
                "strict resident SCF: success marker did not prove code=0 ready=1 resident=1 "
                f"(code={backend_code}, ready={backend_ready}, resident={backend_resident})",
                file=sys.stderr,
            )
            return 1
        if not FINAL_DENSITY_RE.search(combined):
            print("strict resident SCF: final_density=current_resident marker was not found", file=sys.stderr)
            return 1
        host_commit_matches = HOST_COMMIT_RE.findall(combined)
        if not host_commit_matches:
            print(
                "strict resident SCF: host_commit_only=1 phase=final_publication marker was not found",
                file=sys.stderr,
            )
            return 1
        if len(host_commit_matches) != 1:
            print(
                "strict resident SCF: host_commit_only=1 phase=final_publication marker "
                f"appeared {len(host_commit_matches)} times",
                file=sys.stderr,
            )
            return 1
        host_commit_arrays, host_commit_bytes, host_commit_cosmo = map(
            int, host_commit_matches[-1]
        )
        final_publication_matches = FINAL_PUBLICATION_RE.findall(combined)
        if not final_publication_matches:
            print(
                "strict resident SCF: final_publication_done typed marker was not found",
                file=sys.stderr,
            )
            return 1
        (
            final_publication_done,
            final_publication_arrays,
            final_publication_bytes,
            final_publication_cosmo,
        ) = map(int, final_publication_matches[-1])
        if final_publication_done != 1 or final_publication_arrays <= 0 or final_publication_bytes <= 0:
            print(
                "strict resident SCF: final_publication_done marker did not prove final publication "
                f"(done={final_publication_done}, arrays={final_publication_arrays}, "
                f"bytes={final_publication_bytes})",
                file=sys.stderr,
            )
            return 1
        if (
            host_commit_arrays != final_publication_arrays
            or host_commit_bytes != final_publication_bytes
            or host_commit_cosmo != final_publication_cosmo
        ):
            print(
                "strict resident SCF: host commit marker and typed final publication disagree "
                f"(host arrays={host_commit_arrays}, bytes={host_commit_bytes}, cosmo={host_commit_cosmo}; "
                f"typed arrays={final_publication_arrays}, bytes={final_publication_bytes}, "
                f"cosmo={final_publication_cosmo})",
                file=sys.stderr,
            )
            return 1

        stage_matches = STAGE_RE.findall(combined)
        if not stage_matches:
            print("strict resident SCF: stage mask marker was not found", file=sys.stderr)
            return 1
        stage_completed, stage_required, stage_missing = map(int, stage_matches[-1])
        if stage_required != MOZYME_SCF_STAGE_FULL or stage_missing != 0:
            print(
                "strict resident SCF: incomplete full stage mask "
                f"completed={stage_completed} required={stage_required} missing={stage_missing}",
                file=sys.stderr,
            )
            return 1
        if stage_completed != stage_required or stage_completed != MOZYME_SCF_STAGE_FULL:
            print(
                "strict resident SCF: stage_completed must exactly equal stage_required "
                f"and the full stage mask (completed={stage_completed}, "
                f"required={stage_required}, full={MOZYME_SCF_STAGE_FULL})",
                file=sys.stderr,
            )
            return 1
        stage_name_matches = STAGE_NAMES_RE.findall(combined)
        if not stage_name_matches:
            print("strict resident SCF: stage name marker was not found", file=sys.stderr)
            return 1
        stage_completed_names, stage_missing_names = stage_name_matches[-1]
        if (
            stage_completed_names != MOZYME_SCF_STAGE_FULL_NAMES
            or stage_missing_names.lower() != "none"
        ):
            print(
                "strict resident SCF: stage names do not prove the full mask "
                f"(completed={stage_completed_names}, missing={stage_missing_names})",
                file=sys.stderr,
            )
            return 1
        iteration_matches = DEVICE_ITERATIONS_RE.findall(combined)
        if not iteration_matches:
            print("strict resident SCF: device iteration marker was not found", file=sys.stderr)
            return 1
        device_id, final_iterations = map(int, iteration_matches[-1])
        if device_id < 0:
            print(
                f"strict resident SCF: device_id must be nonnegative, got {device_id}",
                file=sys.stderr,
            )
            return 1
        min_stage_calls = max(1, final_iterations)
        stage_call_matches = RESIDENT_STAGE_CALLS_RE.findall(combined)
        if not stage_call_matches:
            print("strict resident SCF: resident_stage_calls marker was not found", file=sys.stderr)
            return 1
        stage_calls = [int(value) for value in stage_call_matches[-1]]
        deficient_stages = [
            f"{name}={count}"
            for name, count in zip(MOZYME_SCF_STAGE_NAMES, stage_calls)
            if count < (1 if name == "upload" else min_stage_calls)
        ]
        if deficient_stages:
            print(
                "strict resident SCF: resident stage call counters are below "
                f"the required count (upload >= 1, other stages >= {min_stage_calls}): "
                + ", ".join(deficient_stages),
                file=sys.stderr,
            )
            return 1
        cnvgz_activity_matches = CNVGZ_ACTIVITY_RE.findall(combined)
        if not cnvgz_activity_matches:
            print("strict resident SCF: CNVGZ active/no-op marker was not found", file=sys.stderr)
            return 1
        cnvgz_activity_fields = marker_fields(cnvgz_activity_matches[-1])
        cnvgz_active_calls = int(cnvgz_activity_fields["cnvgz_active_calls"])
        cnvgz_noop_calls = int(cnvgz_activity_fields["cnvgz_noop_calls"])
        if not cnvgz_active_noop_ready(cnvgz_active_calls, cnvgz_noop_calls):
            print(
                "strict resident SCF: CNVGZ active/no-op marker did not prove GPU stage work "
                f"(active={cnvgz_active_calls}, noop={cnvgz_noop_calls})",
                file=sys.stderr,
            )
            return 1
        strict_proof_matches = STRICT_PROOF_RE.findall(combined)
        if not strict_proof_matches:
            print("strict resident SCF: strict_proof marker was not found", file=sys.stderr)
            return 1
        (
            strict_resident,
            no_fallback_required,
            full_stage_mask,
            resident_decision_complete,
            strict_host_syncs,
            strict_control_polls,
        ) = map(
            int, strict_proof_matches[-1]
        )
        if (
            strict_resident != 1
            or no_fallback_required != 1
            or full_stage_mask != 1
            or resident_decision_complete != 1
            or strict_host_syncs != 0
            or strict_control_polls != 0
        ):
            print(
                "strict resident SCF: strict_proof marker did not prove resident/no-fallback/full-mask execution "
                f"(strict={strict_resident}, no_fallback={no_fallback_required}, "
                f"full_stage={full_stage_mask}, decision={resident_decision_complete}, "
                f"strict_host_syncs={strict_host_syncs}, "
                f"strict_control_polls={strict_control_polls})",
                file=sys.stderr,
            )
            return 1
        decision_matches = RESIDENT_DECISION_RE.findall(combined)
        if not decision_matches:
            print("strict resident SCF: resident_decision marker was not found", file=sys.stderr)
            return 1
        resident_decision = int(decision_matches[-1])
        if resident_decision != MOZYME_SCF_RESIDENT_DECISION_COMPLETE:
            print(
                "strict resident SCF: resident_decision must be CompleteAndPublish "
                f"({MOZYME_SCF_RESIDENT_DECISION_COMPLETE}), got {resident_decision}",
                file=sys.stderr,
            )
            return 1
        fock_plan_matches = RESIDENT_FOCK_PLAN_RE.findall(combined)
        if not fock_plan_matches:
            print("strict resident SCF: resident Fock plan marker was not found", file=sys.stderr)
            return 1
        fock_plan = fock_plan_matches[-1]
        fock_plan_id = int(fock_plan[0])
        fock_plan_full = int(fock_plan[1])
        if fock_plan_id < 0 or fock_plan_full != 1:
            print(
                "strict resident SCF: resident Fock plan did not prove full coverage "
                f"(plan_id={fock_plan_id}, full_coverage={fock_plan_full})",
                file=sys.stderr,
            )
            return 1
        if not fock_plan[3] or not fock_plan[4]:
            print("strict resident SCF: resident Fock plan masks were not found", file=sys.stderr)
            return 1
        required_mask = int(fock_plan[3])
        covered_mask = int(fock_plan[4])
        if required_mask == 0 or covered_mask != required_mask:
            print(
                "strict resident SCF: resident Fock plan masks must exactly match "
                f"(required={required_mask}, covered={covered_mask})",
                file=sys.stderr,
            )
            return 1
        if required_mask & 2 and int(fock_plan[2] or 0) != 1:
            print("strict resident SCF: partial resident Fock plan was not covered", file=sys.stderr)
            return 1
        sparse_fock_runs = SPARSE_FOCK_RUN_RE.findall(combined)
        if not sparse_fock_runs:
            print("strict resident SCF: resident sparse Fock GPU run profile was not found", file=sys.stderr)
            return 1
        if len(sparse_fock_runs) < min_stage_calls:
            print(
                "strict resident SCF: resident sparse Fock GPU run count is below "
                f"the final iteration count {min_stage_calls}: {len(sparse_fock_runs)}",
                file=sys.stderr,
            )
            return 1
        zero_work_runs = []
        total_sparse_work = 0
        for match in sparse_fock_runs:
            work = sum(int(value) for value in match[:4])
            total_sparse_work += work
            if work <= 0:
                zero_work_runs.append(match)
        if zero_work_runs or total_sparse_work <= 0:
            print("strict resident SCF: resident sparse Fock GPU reported zero work", file=sys.stderr)
            return 1
        resident_reorth_success = False
        for status, extra in MOZYME_REORTH_RE.findall(combined):
            if status.lower() == "success" and marker_fields(extra).get("resident") == "1":
                resident_reorth_success = True
                break
        if not resident_reorth_success:
            print(
                "strict resident SCF: resident=1 final GPU REORTH success marker was not found",
                file=sys.stderr,
            )
            return 1
        disallowed_sections = strict_disallowed_section_names(combined)
        if disallowed_sections:
            print(
                "strict resident SCF: disallowed CPU MOZYME section(s) ran: "
                + ", ".join(disallowed_sections),
                file=sys.stderr,
            )
            return 1
        den_files = sorted(path.name for path in workdir.glob("*.den"))
        if den_files:
            print(
                "strict resident SCF: host .den checkpoint artifacts were produced: "
                + ", ".join(den_files),
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

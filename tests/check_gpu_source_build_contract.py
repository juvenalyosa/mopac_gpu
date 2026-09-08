#!/usr/bin/env python3
"""Static guards for GPU/MOZYME source inclusion in the build graph."""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GPU_CORE_SOURCES = (
    "gpu/cuda_wrappers.cu",
    "gpu/cublas_interfaces.F90",
    "gpu/gpu_density_interfaces.F90",
    "gpu/gpu_ortho_interfaces.F90",
    "gpu/gpu_runtime_interfaces.F90",
    "gpu/gpu_transform_interfaces.F90",
    "gpu/gpu_eig_mg_interfaces.F90",
    "gpu/gpu_small_solve_interfaces.F90",
    "gpu/gpu_bmat_interfaces.F90",
    "gpu/gpu_grad_interfaces.F90",
    "gpu/gpu_diis_interfaces.F90",
    "gpu/gpu_fock_interfaces.F90",
    "gpu/gpu_hmtr_interfaces.F90",
    "gpu/gpu_scf_types.F90",
    "gpu/gpu_scf_interfaces.F90",
    "gpu/gpu_scf_stream_interfaces.F90",
    "gpu/gpu_scf_stream_driver.F90",
    "gpu/gpu_scf_stream_trace.F90",
    "gpu/gpu_mozyme_scf_interfaces.F90",
    "gpu/scf_driver.cu",
    "gpu/mozyme_scf_context.cu",
    "gpu/fock_kernels.cu",
    "gpu/grad_kernels.cu",
    "gpu/hmtr_optimizer.cu",
)

CUDA_LANGUAGE_SOURCES = (
    "gpu/cuda_wrappers.cu",
    "gpu/scf_driver.cu",
    "gpu/mozyme_scf_context.cu",
    "gpu/fock_kernels.cu",
    "gpu/grad_kernels.cu",
    "gpu/hmtr_optimizer.cu",
)

MOZYME_CORE_MODULES = (
    "mozyme_gpu_int_utils",
    "mozyme_section_timers",
    "mozyme_diagg1_state",
    "mozyme_diagg2_state",
    "mozyme_isitsc_state",
    "mozyme_resident_fock",
    "mozyme_gpu_makvec",
    "mozyme_gpu_relocalize",
    "mozyme_gpu_reorth",
    "mozyme_gpu_tidy",
    "mozyme_gpu_scf_driver",
    "fillij",
    "check",
    "buildf",
    "add_more_interactions",
    "isitsc",
    "setupk",
    "reorth",
    "mozyme_gpu_plan",
    "set_up_MOZYME_arrays",
    "eimp",
    "cnvgz",
    "diagg",
    "density_for_MOZYME",
    "addhb",
    "mozyme_fock1_batch",
    "mozyme_fock2_4x1_batch",
    "fock2z",
    "helecz",
    "iter_for_MOZYME",
    "pinout",
    "fock1_for_MOZYME",
    "tidy",
    "makvec",
    "diagg2",
    "diagg1",
)

SOURCE_MARKER_CONTRACT_VERSION = "mopac-colab-source-markers-explicit-proof-v83"
SOURCE_MARKER_CONTRACT_SHA256 = "17aa11cd4550db1b4d2bbf2f4b15bfc13b9c6e06faad28f6609ee42c7ab69240"
FORTRAN_FREE_FORM_LINE_LIMIT = 132
CRITICAL_GPU_FORTRAN_SOURCES = (
    "src/MOZYME/addhb.F90",
    "src/MOZYME/check.F90",
    "src/MOZYME/cnvgz.F90",
    "src/MOZYME/density_for_MOZYME.F90",
    "src/MOZYME/diagg.F90",
    "src/MOZYME/diagg1.F90",
    "src/MOZYME/diagg2.F90",
    "src/MOZYME/eimp.F90",
    "src/MOZYME/add_more_interactions.F90",
    "src/MOZYME/fillij.F90",
    "src/MOZYME/fock1_for_MOZYME.F90",
    "src/MOZYME/fock2z.F90",
    "src/MOZYME/helecz.F90",
    "src/MOZYME/isitsc.F90",
    "src/MOZYME/iter_for_MOZYME.F90",
    "src/MOZYME/mozyme_diagg1_state.F90",
    "src/MOZYME/mozyme_diagg2_state.F90",
    "src/MOZYME/mozyme_fock1_batch.F90",
    "src/MOZYME/mozyme_fock2_4x1_batch.F90",
    "src/MOZYME/mozyme_gpu_int_utils.F90",
    "src/MOZYME/mozyme_gpu_makvec.F90",
    "src/MOZYME/mozyme_gpu_plan.F90",
    "src/MOZYME/mozyme_gpu_relocalize.F90",
    "src/MOZYME/mozyme_gpu_reorth.F90",
    "src/MOZYME/mozyme_gpu_tidy.F90",
    "src/MOZYME/mozyme_gpu_scf_driver.F90",
    "src/MOZYME/mozyme_isitsc_state.F90",
    "src/MOZYME/mozyme_resident_fock.F90",
    "src/MOZYME/mozyme_section_timers.F90",
    "src/MOZYME/pinout.F90",
    "src/MOZYME/reorth.F90",
    "src/MOZYME/set_up_MOZYME_arrays.F90",
    "src/MOZYME/setupk.F90",
    "src/MOZYME/tidy.F90",
    "src/gpu/cublas_interfaces.F90",
    "src/gpu/gpu_fock_interfaces.F90",
    "src/gpu/gpu_mozyme_scf_interfaces.F90",
)


def normalize(text: str) -> str:
    return " ".join(text.split())


def notebook_source_code(text: str) -> str:
    payload = json.loads(text)
    return "\n".join("".join(cell.get("source", ())) for cell in payload.get("cells", ()))


def cmake_if_block(text: str, condition: str) -> str:
    pattern = re.compile(r"^\s*(if|endif)\s*\(([^)]*)\)", re.IGNORECASE | re.MULTILINE)
    stack: list[tuple[str, int]] = []
    condition_norm = normalize(condition).lower()
    for match in pattern.finditer(text):
        keyword = match.group(1).lower()
        expr_norm = normalize(match.group(2)).lower()
        if keyword == "if":
            stack.append((expr_norm, match.end()))
        elif stack:
            expr, start = stack.pop()
            if expr == condition_norm:
                return text[start : match.start()]
    return ""


def cmake_list_items(text: str, list_name: str) -> set[str]:
    match = re.search(
        rf"set\s*\(\s*{re.escape(list_name)}\s+(.*?)\n\s*\)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return set()
    body = re.sub(r"#.*", " ", match.group(1))
    return {item for item in re.split(r"\s+", body.strip()) if item}


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def source_region_between(text: str, start_fragment: str, end_fragment: str) -> str:
    start = text.find(start_fragment)
    if start < 0:
        return ""
    end = text.find(end_fragment, start + len(start_fragment))
    if end < 0:
        return ""
    return text[start:end]


def final_reorth_commit_flag_scoped(cuda_text: str) -> bool:
    body = source_region_between(
        cuda_text,
        "bool apply_final_reorth_on_gpu(",
        "\nbool compute_initial_setup_on_gpu(",
    )
    declaration = "bool device_final_reorth_committed = false;"
    assignment = "device_final_reorth_committed = true;"
    decision = "ok || device_final_reorth_committed"
    if not body:
        return False
    declaration_index = body.find(declaration)
    assignment_index = body.find(assignment)
    decision_index = body.find(decision)
    return (
        cuda_text.count(declaration) == 1
        and body.count(declaration) == 1
        and declaration_index >= 0
        and assignment_index > declaration_index
        and decision_index > assignment_index
    )


def critical_fortran_line_length_failures() -> list[str]:
    failures: list[str] = []
    for rel in CRITICAL_GPU_FORTRAN_SOURCES:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"{rel} is missing from the critical GPU Fortran line-length guard")
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(),
            start=1,
        ):
            if len(line) > FORTRAN_FREE_FORM_LINE_LIMIT:
                failures.append(
                    f"{rel}:{line_number} line length {len(line)} exceeds "
                    f"{FORTRAN_FREE_FORM_LINE_LIMIT}"
                )
    return failures


def main() -> int:
    src_cmake = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
    mozyme_cmake = (ROOT / "src/MOZYME/CMakeLists.txt").read_text(encoding="utf-8")
    tests_cmake = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
    cuda_text = (ROOT / "src/gpu/mozyme_scf_context.cu").read_text(encoding="utf-8")
    cuda_wrappers_text = (ROOT / "src/gpu/cuda_wrappers.cu").read_text(encoding="utf-8")
    scf_driver_text = (ROOT / "src/gpu/scf_driver.cu").read_text(encoding="utf-8")
    scf_interface_text = (ROOT / "src/gpu/gpu_mozyme_scf_interfaces.F90").read_text(
        encoding="utf-8"
    )
    mozyme_scf_driver_text = (ROOT / "src/MOZYME/mozyme_gpu_scf_driver.F90").read_text(
        encoding="utf-8"
    )
    resident_fock_text = (ROOT / "src/MOZYME/mozyme_resident_fock.F90").read_text(encoding="utf-8")
    colab_text = (ROOT / "colab/mopac_cublas_gpu_bench_colab.ipynb").read_text(encoding="utf-8")
    colab_code_text = notebook_source_code(colab_text)
    create_zip_text = (ROOT / "scripts/create_colab_gpu_zip.py").read_text(encoding="utf-8")
    verify_zip_text = (ROOT / "scripts/verify_colab_gpu_proof_zip.py").read_text(encoding="utf-8")
    report_text = (ROOT / "scripts/molecule_benchmark_report.py").read_text(encoding="utf-8")
    gpu_block = cmake_if_block(src_cmake, "GPU")
    gpu_block_flat = normalize(gpu_block)
    src_cmake_flat = normalize(src_cmake)
    failures: list[str] = []

    require(bool(gpu_block), "src/CMakeLists.txt is missing an if(GPU) block", failures)
    for rel in GPU_CORE_SOURCES:
        fragment = f"target_sources(mopac-core PRIVATE ${{CMAKE_CURRENT_SOURCE_DIR}}/{rel})"
        require(
            fragment in gpu_block_flat,
            f"{rel} is not added to mopac-core inside if(GPU)",
            failures,
        )

    for rel in CUDA_LANGUAGE_SOURCES:
        require(
            rel in src_cmake and "PROPERTIES LANGUAGE CUDA" in src_cmake_flat,
            f"{rel} is not covered by the CUDA language source-properties block",
            failures,
        )

    mozyme_sources = cmake_list_items(mozyme_cmake, "src_list")
    for module in MOZYME_CORE_MODULES:
        require(
            module in mozyme_sources,
            f"{module}.F90 is not listed in src/MOZYME/CMakeLists.txt src_list",
            failures,
        )

    require(
        'add_test(NAME "gpu-source-build-contract"' in tests_cmake,
        "tests/CMakeLists.txt is missing gpu-source-build-contract",
        failures,
    )
    require(
        "check_gpu_source_build_contract.py" in tests_cmake,
        "tests/CMakeLists.txt does not run check_gpu_source_build_contract.py",
        failures,
    )
    require(
        f'SOURCE_MARKER_CONTRACT_VERSION = "{SOURCE_MARKER_CONTRACT_VERSION}"' in create_zip_text,
        "create_colab_gpu_zip.py has an unexpected source marker contract version",
        failures,
    )
    require(
        SOURCE_MARKER_CONTRACT_SHA256 in create_zip_text
        and SOURCE_MARKER_CONTRACT_SHA256 in report_text
        and SOURCE_MARKER_CONTRACT_SHA256 in colab_text,
        "source marker contract SHA is not synchronized across packager, report, and notebook",
        failures,
    )
    require(
        "fallback_real_pairs\\s+total\\s*=\\s*[1-9]\\d*" in report_text
        and "fallback_real_pairs\\\\s+total\\\\s*=\\\\s*[1-9]\\\\d*" in colab_text
        and "fallback_real_pairs\\b|" not in report_text
        and "fallback_real_pairs\\\\b|" not in colab_text,
        "helper fatal parsing must only treat positive fallback_real_pairs as fatal",
        failures,
    )
    require(
        "DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_ABS_TOL = 5.0e-3" in report_text
        and "DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_REL_TOL = 1.0e-5" in report_text
        and "DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_PER_ATOM_TOL = 5.0e-4" in report_text
        and "default=DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_ABS_TOL" in report_text
        and "default=DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_REL_TOL" in report_text
        and "default=DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_PER_ATOM_TOL" in report_text,
        "strict readiness CPU companion tolerance defaults are not synchronized",
        failures,
    )
    require(
        "or '..' in posix_path.parts" in colab_text
        and "or '..' in rel_path.parts" in colab_text
        and "or '..' in normalized" not in colab_text
        and "or '..' in rel:" not in colab_text,
        "Colab zip path safety must reject only real '..' path components",
        failures,
    )
    require(
        "path.is_symlink()" in create_zip_text
        and "Refusing to package symlinked source files" in create_zip_text
        and "path.resolve().relative_to(root_resolved)" in create_zip_text,
        "Colab source packager must reject symlinks and outside-root resolved paths",
        failures,
    )
    require(
        "zip member set does not match manifest" in verify_zip_text
        and "def source_marker_contract_sha256(markers: list[dict[str, Any]]) -> str:" in verify_zip_text
        and "required_source_markers sha256 mismatch" in verify_zip_text
        and "notebook contains stale unsafe-path substring check" in verify_zip_text,
        "Colab proof zip verifier must reject extra members and recompute marker contract hash",
        failures,
    )
    require(
        'FULL_SCF_PROBE_FORCED_KEYWORDS = ("MOZYME_MINBLK=1", "RE-LOCAL=1", "REORTH")'
        in report_text
        and "RE-LOCAL=1 REORTH MOZYME_MINBLK=1" in colab_text
        and "RE-LOCAL=1 REORTH NOCOMMENTS" in (ROOT / "tests/check_mozyme_strict_resident_scf.py").read_text(
            encoding="utf-8"
        )
        and "RE-LOC=1" not in report_text
        and "RE-LOC=1" not in colab_text,
        "strict readiness probes must use recognized RE-LOCAL=1 keyword, not RE-LOC=1",
        failures,
    )
    writmo_colab_marker_block = source_region_between(
        colab_code_text,
        "'writmo_no_optional_cpu_outputs': (writmo_text, (",
        "'strict_resident_scf_ctest_fail_closed':",
    )
    require(
        "writmo_relocal_output=skipped resident_gpu=1" in writmo_colab_marker_block
        and "writmo_pops_output=skipped resident_gpu=1" in writmo_colab_marker_block
        and "'strict_writmo_reloc_host_output'" not in writmo_colab_marker_block
        and "'strict_writmo_pops_host_output'" not in writmo_colab_marker_block,
        "Colab hardcoded writmo validation must require strict GPU skip markers, not retired writmo abort markers",
        failures,
    )
    require(
        "mozyme_final_reorth_status_kernel" in cuda_text
        and "final_reorth_status_scalars" in cuda_text
        and "resident final status scalar copy" in cuda_text
        and "current_resident_energy_offset" not in cuda_text,
        "resident final reorth status energy must be computed on GPU",
        failures,
    )
    require(
        "device_final_reorth_committed" in cuda_text
        and "ok || device_final_reorth_committed" in cuda_text,
        "resident final reorth must not report CPU boundary after device finalization",
        failures,
    )
    require(
        final_reorth_commit_flag_scoped(cuda_text),
        "final reorth commit flag must be scoped inside apply_final_reorth_on_gpu before use",
        failures,
    )
    failures.extend(critical_fortran_line_length_failures())
    require(
        "use mozyme_gpu_int_utils, only: mozyme_c_int_checked" in mozyme_scf_driver_text
        and "config%resident_fock_plan_id = mozyme_c_int_checked(resident_fock_plan_id)"
        in mozyme_scf_driver_text,
        "resident SCF driver must import mozyme_c_int_checked before setting resident_fock_plan_id",
        failures,
    )
    require(
        "CPU count reference" not in resident_fock_text
        and "strict_resident_fock_gpu_pack_mismatch" not in resident_fock_text
        and "strict_resident_fock_gpu_count_mismatch" not in resident_fock_text
        and "strict_resident_fock_gpu_point_weights_mismatch" not in resident_fock_text
        and "cpu_plan_constructed=1 setup_only=1 plan_id=" not in resident_fock_text
        and "strict_resident_fock_cpu_plan_setup" not in resident_fock_text
        and "mopac_cuda_mozyme_resident_fock_count_plan" not in resident_fock_text
        and "mopac_cuda_mozyme_resident_fock_point_weights" not in resident_fock_text
        and "signature = last_signature(plan_id) + 1_c_int64_t" in resident_fock_text
        and "log_resident_coverage_gpu_counts" in resident_fock_text,
        "strict resident Fock pack must use GPU-authored setup/counts without CPU count reference or CPU plan marker",
        failures,
    )
    require(
        "import :: max_resident_fallback_basis" in resident_fock_text
        and "fallback_basis_c(0:max_resident_fallback_basis,0:*)" in resident_fock_text,
        "resident Fock C interface must import max_resident_fallback_basis before using it as an array bound",
        failures,
    )
    require(
        "gpu_count=1 plan_id=" in resident_fock_text
        and "gpu_pack=1 plan_id=" in resident_fock_text
        and "gpu_point_weights=1 point=" in resident_fock_text
        and "source=pack" in resident_fock_text,
        "strict resident Fock proof markers must come from the GPU pack path",
        failures,
    )
    require(
        "mozyme_resident_pack_point_weights_dev" in cuda_wrappers_text
        and "mopac_cuda_mozyme_resident_fock_point_weights" not in cuda_wrappers_text
        and "mozyme_resident_point_weights_kernel" not in cuda_wrappers_text
        and "g_mz_res_point_out" not in cuda_wrappers_text,
        "resident point weights must only be built inside the GPU pack path",
        failures,
    )
    require(
        "bool strict_resident_stream_required()" in scf_driver_text
        and 'env_enabled_ci("MOPAC_MOZYME_SCF_STRICT_RESIDENT")' in scf_driver_text
        and 'env_enabled_ci("MOPAC_MOZYME_FULL_SCF_GPU")' in scf_driver_text
        and "if (strict_resident_stream_required())" in scf_driver_text
        and "GPU SCF stream finalize: strict resident Fock cache miss" in scf_driver_text
        and "cudaMemcpy fock resident fallback" in scf_driver_text,
        "resident stream finalizer must fail closed on strict Fock cache miss",
        failures,
    )
    require(
        "GPU_MOZYME_SCF_ABI_VERSION = 33_c_int" in scf_interface_text
        and "integer(c_int) :: strict_resident_host_syncs = 0_c_int" in scf_interface_text
        and "integer(c_int) :: strict_resident_control_polls = 0_c_int" in scf_interface_text
        and "kMozymeScfAbiVersion = 33" in cuda_text
        and "int strict_resident_host_syncs;" in cuda_text
        and "int strict_resident_control_polls;" in cuda_text
        and "copy_resident_control_snapshot_from_gpu(*ctx, &control, false)" in cuda_text
        and "completed |= stage_bits;" in cuda_text
        and "const int synchronize_sparse_fock = strict_resident ? 0 : 1;" in cuda_text
        and "backend_strict_host_control_clean(status)" in mozyme_scf_driver_text
        and "backend_strict_host_control_polling" in mozyme_scf_driver_text
        and "strict_host_syncs=" in mozyme_scf_driver_text
        and "strict_control_polls=" in mozyme_scf_driver_text,
        "strict resident SCF ABI v33 must reject host sync/control polling in proof mode",
        failures,
    )
    require(
        "ensure_cnvgz_activity_from_resident_stage_calls" in cuda_text
        and "status->resident_stage_calls[kResidentStageSlotCnvgz]" in cuda_text
        and "status->cnvgz_noop_calls = cnvgz_calls" in cuda_text,
        "resident SCF must publish CNVGZ no-op activity from device stage calls when fine counters are empty",
        failures,
    )
    require(
        "subroutine normalize_cnvgz_activity(status)" in mozyme_scf_driver_text
        and "call normalize_cnvgz_activity(status)" in mozyme_scf_driver_text
        and "status%resident_stage_calls(6) > 0_c_int" in mozyme_scf_driver_text
        and "status%cnvgz_noop_calls = status%resident_stage_calls(6)" in mozyme_scf_driver_text,
        "resident SCF driver must normalize CNVGZ no-op activity before strict success checks",
        failures,
    )
    require(
        "final_density=current_resident" in mozyme_scf_driver_text
        and "index(prefix, 'status=success') > 0" in mozyme_scf_driver_text,
        "resident SCF success trace must report final_density=current_resident",
        failures,
    )

    if failures:
        for failure in failures:
            print(f"GPU source build contract failure: {failure}", file=sys.stderr)
        return 1
    print("GPU source build/proof contract static check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

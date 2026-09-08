#!/usr/bin/env python3
"""Create the Colab GPU proof source zip with a SHA-256 manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT = "mopac_colab_gpu_bench.zip"
MANIFEST_NAME = "MOPAC_COLAB_SOURCE_MANIFEST.sha256"
PROVENANCE_NAME = "MOPAC_COLAB_SOURCE_PROVENANCE.json"
FEATURES_NAME = "MOPAC_COLAB_FEATURES.json"
PROVENANCE_SCHEMA = "mopac-colab-source-provenance-v1"
FEATURES_SCHEMA = "mopac-colab-feature-manifest-v1"
ZIP_SOURCE_TREE_MARKER = "mopac-colab-gpu-proof-source-v1"
FEATURE_SET = "mozyme-full-scf-gpu-makvec-relocal-final-resident-reorth-tidy-selmos-pls-reset-cosmo-direct-point-kind-cnvgz-active-or-noop-cpu-compare-explicit-proof-v38-20260702"
FULL_SCF_CONTRACT_VERSION = "resident-scf-strict-final-resident-reorth-tidy-selmos-pls-reset-cosmo-direct-resident-cg-point-kind-cnvgz-active-or-noop-cpu-compare-explicit-proof-v58"
SOURCE_MARKER_CONTRACT_VERSION = "mopac-colab-source-markers-explicit-proof-v83"
SOURCE_ALLOWLIST_POLICY_SCHEMA = "mopac-colab-source-allowlist-v1"
FORTRAN_FREE_FORM_LINE_LIMIT = 132
ALLOWED_SOURCE_ROOT_FILES = {
    ".gitignore",
    "AUTHORS.rst",
    "CITATION.cff",
    "CMakeLists.txt",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.rst",
    "Dockerfile",
    "LICENSE",
    "NOTICE",
    "README.md",
}
ALLOWED_SOURCE_ROOT_DIRS = {
    ".github",
    "benchmarks",
    "cmake",
    "colab",
    "data",
    "docs",
    "examples",
    "include",
    "logo",
    "scripts",
    "src",
    "tests",
}
CRITICAL_FEATURE_FILES = (
    "CMakeLists.txt",
    "src/CMakeLists.txt",
    "src/run_mopac.F90",
    "src/MOZYME/CMakeLists.txt",
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
    "src/output/writmo.F90",
    "src/solvation/linear_cosmo.F90",
    "src/gpu/cublas_interfaces.F90",
    "src/gpu/cuda_wrappers.cu",
    "src/gpu/fock_kernels.cu",
    "src/gpu/grad_launch.h",
    "src/gpu/gpu_bmat_interfaces.F90",
    "src/gpu/gpu_density_interfaces.F90",
    "src/gpu/gpu_diis_interfaces.F90",
    "src/gpu/gpu_eig_mg_interfaces.F90",
    "src/gpu/gpu_fock_interfaces.F90",
    "src/gpu/gpu_grad_interfaces.F90",
    "src/gpu/gpu_hmtr_interfaces.F90",
    "src/gpu/gpu_mozyme_scf_interfaces.F90",
    "src/gpu/gpu_ortho_interfaces.F90",
    "src/gpu/gpu_runtime_interfaces.F90",
    "src/gpu/gpu_scf_interfaces.F90",
    "src/gpu/gpu_scf_stream_driver.F90",
    "src/gpu/gpu_scf_stream_interfaces.F90",
    "src/gpu/gpu_scf_stream_trace.F90",
    "src/gpu/gpu_scf_types.F90",
    "src/gpu/gpu_small_solve_interfaces.F90",
    "src/gpu/gpu_transform_interfaces.F90",
    "src/gpu/grad_kernels.cu",
    "src/gpu/hmtr_optimizer.cu",
    "src/gpu/mozyme_scf_context.cu",
    "src/gpu/packed_utils.h",
    "src/gpu/scf_driver.cu",
    "tests/CMakeLists.txt",
    "tests/gpu_bench.F90",
    "tests/gpu_resident_fock_pair_compare.F90",
    "tests/check_gpu_env_flag_parsing.py",
    "tests/check_gpu_source_build_contract.py",
    "tests/check_mozyme_strict_resident_scf.py",
    "docs/GPU_GUIDE.md",
    "scripts/collect_existing_mopac_references.py",
    "scripts/molecule_benchmark_report.py",
    "scripts/create_colab_gpu_zip.py",
    "scripts/verify_colab_gpu_proof_zip.py",
    "scripts/gpu_benchmark_report.py",
    "scripts/hydrogenate_publication_benchmark_inputs.py",
    "scripts/prepare_publication_benchmark_inputs.py",
    "colab/mopac_cublas_gpu_bench_colab.ipynb",
)
REQUIRED_SOURCE_MARKERS = (
    {
        "name": "standalone_gpu_benchmark_target_before_tests",
        "path": "CMakeLists.txt",
        "fragments": (
            "# GPU functionality",
            "if(GPU)",
            "CUDA_STANDARD 14",
            "CUDA_STANDARD_REQUIRED ON",
            "add_executable(mopac-gpu-bench tests/gpu_bench.F90)",
            "target_link_libraries(mopac-gpu-bench mopac-core)",
            "add_dependencies(mopac-gpu-bench mopac-core)",
            "option(TESTS ",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "standalone_resident_fock_pair_target_before_tests",
        "path": "CMakeLists.txt",
        "fragments": (
            "# GPU functionality",
            "if(GPU)",
            "add_executable(mopac-gpu-resident-fock-pair-compare tests/gpu_resident_fock_pair_compare.F90)",
            "target_link_libraries(mopac-gpu-resident-fock-pair-compare mopac-core)",
            "add_dependencies(mopac-gpu-resident-fock-pair-compare mopac-core)",
            "option(TESTS ",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "ctest_gpu_tests_are_gpu_gated",
        "path": "tests/CMakeLists.txt",
        "fragments": (
            "if(GPU AND ENABLE_GPU_TESTS)",
            "function(add_mopac_gpu_test test_name target_name source_file)",
            "set_tests_properties(\"${test_name}\" PROPERTIES LABELS \"gpu\")",
            "if(TARGET mopac-gpu-resident-fock-pair-compare)",
            "set_tests_properties(gpu-resident-fock-pair-compare PROPERTIES LABELS \"gpu\")",
            'add_test(NAME "gpu-mozyme-strict-resident-scf"',
            "check_mozyme_strict_resident_scf.py",
            "set_tests_properties(gpu-mozyme-strict-resident-scf PROPERTIES",
        ),
        "forbidden_fragments": (
            "print *, 'GPU support not enabled; skipping gpu_resident_fock_pair_compare'",
        ),
    },
    {
        "name": "gpu_env_flag_parsing_static_ctest",
        "path": "tests/CMakeLists.txt",
        "fragments": (
            'add_test(NAME "gpu-env-flag-parsing"',
            "check_gpu_env_flag_parsing.py",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "gpu_env_flag_parsing_static_guard",
        "path": "tests/check_gpu_env_flag_parsing.py",
        "fragments": (
            "Static guards for GPU environment-variable parsing.",
            "MOPAC_FASTGPU",
            "MOPAC_RESIDENT_SCF",
            "MOPAC_MOZYME_RESIDENT_FOCK_GPU",
            "MOPAC_GPU_AUTOPOLICY_OFF",
            "MOPAC_GPU_GRAD_EXPERIMENTAL",
            "TRUE_TOKENS_FORTRAN",
            "FALSE_TOKENS_FORTRAN",
            "REQUIRED_FRAGMENTS",
            "FORBIDDEN_REGEXES",
            "env_truthy_ci(skip)",
            "forbidden GPU env parsing pattern",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "gpu_source_build_contract_static_ctest",
        "path": "tests/CMakeLists.txt",
        "fragments": (
            'add_test(NAME "gpu-source-build-contract"',
            "check_gpu_source_build_contract.py",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "gpu_source_build_contract_static_guard",
        "path": "tests/check_gpu_source_build_contract.py",
        "fragments": (
            "Static guards for GPU/MOZYME source inclusion in the build graph.",
            "GPU_CORE_SOURCES",
            "CUDA_LANGUAGE_SOURCES",
            "MOZYME_CORE_MODULES",
            "cmake_if_block",
            "target_sources(mopac-core PRIVATE ${{CMAKE_CURRENT_SOURCE_DIR}}",
            "PROPERTIES LANGUAGE CUDA",
            "src/MOZYME/CMakeLists.txt src_list",
            "gpu-source-build-contract",
            "SOURCE_MARKER_CONTRACT_SHA256",
            "or '..' in posix_path.parts",
            "mozyme_final_reorth_status_kernel",
            "device_final_reorth_committed",
            "strict resident Fock pack must use GPU-authored setup/counts",
            "resident stream finalizer must fail closed on strict Fock cache miss",
            "strict resident SCF ABI v33 must reject host sync/control polling in proof mode",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_resident_stream_finalizer_fail_closed",
        "path": "src/gpu/scf_driver.cu",
        "fragments": (
            "bool strict_resident_stream_required()",
            'env_enabled_ci("MOPAC_MOZYME_SCF_STRICT_RESIDENT")',
            'env_enabled_ci("MOPAC_MOZYME_SCF_GPU")',
            'env_enabled_ci("MOPAC_MOZYME_GPU_STRICT")',
            'env_enabled_ci("MOPAC_MOZYME_FULL_SCF_GPU")',
            "mopac_cuda_register_fock_device(session.mpack, session.f_host, session.d_f)",
            "if (!mopac_cuda_fetch_fock(session.f_host, static_cast<size_t>(session.mpack)))",
            "if (strict_resident_stream_required())",
            "GPU SCF stream finalize: strict resident Fock cache miss",
            "STREAM_STATUS_NOT_READY",
            "cudaMemcpy fock resident fallback",
        ),
        "forbidden_fragments": (
            "if (!mopac_cuda_fetch_fock(session.f_host, static_cast<size_t>(session.mpack))) {\n        cudaError_t err = cudaMemcpy",
        ),
    },
    {
        "name": "colab_proof_zip_verifier",
        "path": "scripts/verify_colab_gpu_proof_zip.py",
        "fragments": (
            "Validate a MOZYME GPU Colab proof source zip before upload.",
            "SOURCE_MANIFEST_NAME = \"MOPAC_COLAB_SOURCE_MANIFEST.sha256\"",
            "SOURCE_FEATURES_NAME = \"MOPAC_COLAB_FEATURES.json\"",
            "SOURCE_PROVENANCE_NAME = \"MOPAC_COLAB_SOURCE_PROVENANCE.json\"",
            "def safe_zip_member(name: str) -> bool:",
            "\"..\" not in posix_path.parts",
            "def verify_manifest(",
            "zip member set does not match manifest",
            "def source_marker_contract_sha256(markers: list[dict[str, Any]]) -> str:",
            "\"required_source_markers\": markers",
            "required_source_markers sha256 mismatch",
            "def verify_source_markers(",
            "proof zip verifier is not listed as a critical file",
            "notebook contains stale unsafe-path substring check",
            "Colab proof zip verification passed:",
        ),
        "forbidden_fragments": (
            "or '..' in normalized",
            "or '..' in rel:",
        ),
    },
    {
        "name": "gpu_core_source_inclusion_contract",
        "path": "src/CMakeLists.txt",
        "fragments": (
            "if(GPU)",
            "target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda_wrappers.cu)",
            "target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_mozyme_scf_interfaces.F90)",
            "target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/scf_driver.cu)",
            "target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/mozyme_scf_context.cu)",
            "target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/fock_kernels.cu)",
            "target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/grad_kernels.cu)",
            "target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/hmtr_optimizer.cu)",
            "set_source_files_properties(",
            "${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda_wrappers.cu",
            "${CMAKE_CURRENT_SOURCE_DIR}/gpu/mozyme_scf_context.cu",
            "PROPERTIES LANGUAGE CUDA",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "mozyme_core_source_inclusion_contract",
        "path": "src/MOZYME/CMakeLists.txt",
        "fragments": (
            "set(src_list",
            "mozyme_gpu_int_utils",
            "mozyme_section_timers",
            "mozyme_resident_fock",
            "mozyme_gpu_makvec",
            "mozyme_gpu_relocalize",
            "mozyme_gpu_reorth",
            "mozyme_gpu_tidy",
            "mozyme_gpu_scf_driver",
            "fillij",
            "check",
            "buildf",
            "mozyme_gpu_plan",
            "density_for_MOZYME",
            "mozyme_fock1_batch",
            "mozyme_fock2_4x1_batch",
            "fock2z",
            "iter_for_MOZYME",
            "target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${idx}.F90)",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_resident_scf_ctest_fail_closed",
        "path": "tests/check_mozyme_strict_resident_scf.py",
        "fragments": (
            "MOPAC_MOZYME_SCF_STRICT_RESIDENT",
            "MOPAC_MOZYME_SCF_GPU",
            "MOPAC_MOZYME_GPU_STRICT",
            "MOPAC_MOZYME_FULL_SCF_GPU",
            "FATAL_STATUS_RE",
            "HELPER_FATAL_STATUS_RE",
            "FOCK_FAMILY_FALLBACK_RE",
            "RESIDENT_FOCK_FALLBACK_REAL_PAIRS_RE",
            "RESIDENT_FOCK_CPU_POINT_PAIRS_RE",
            "resident_fock\\][^\\n]*\\bcpu_point_pairs",
            "SUCCESS_STATUS_RE",
            "GPU_ERROR_MARKERS",
            "MARKER_FIELD_RE",
            "marker_fields",
            "cnvgz_active_noop_ready",
            "STAGE_NAMES_RE",
            "STRICT_PROOF_RE",
            "RESIDENT_DECISION_RE",
            "RESIDENT_FOCK_PLAN_RE",
            "FINAL_DENSITY_RE",
            "HOST_COMMIT_RE",
            "FINAL_PUBLICATION_RE",
            "DEVICE_ITERATIONS_RE",
            "CNVGZ_ACTIVITY_RE",
            "RESIDENT_STAGE_CALLS_RE",
            "SPARSE_FOCK_RUN_RE",
            "MOZYME_SECTION_RE",
            "strict_disallowed_section_names",
            "stage_completed != stage_required",
            "MOZYME_SCF_STAGE_FULL = 1023",
            "MOZYME_SCF_STAGE_FULL_NAMES",
            "resident_decision must be CompleteAndPublish",
            "resident Fock plan did not prove full coverage",
            "resident Fock plan masks were not found",
            "covered_mask != required_mask",
            "resident Fock plan masks must exactly match",
            "stage_required != MOZYME_SCF_STAGE_FULL",
            "backend_cpu_boundary",
            "final_density=current_resident marker was not found",
            "host_commit_only=1 phase=final_publication marker was not found",
            "final_publication_done typed marker was not found",
            "final_publication_done marker did not prove final publication",
            "host commit marker and typed final publication disagree",
            "resident_stage_calls marker was not found",
            "success code=0 ready=1 resident=1 marker was not found",
            "device_id must be nonnegative",
            "CNVGZ active/no-op marker did not prove GPU stage work",
            "resident stage call counters are below",
            "resident sparse Fock GPU run profile was not found",
            "resident sparse Fock GPU reported zero work",
            "disallowed CPU MOZYME section(s) ran",
            "host .den checkpoint artifacts were produced",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "api_initialize_mopac_nogpu_false_values",
        "path": "src/interface/mopac_api_initialize.F90",
        "fragments": (
            "call get_environment_variable('MOPAC_NOGPU', env, status=i)",
            "call upcase(env, len_trim(env))",
            "case ('0','FALSE','F','NO','N','OFF')",
            "lgpu = .false.",
            "return",
        ),
        "forbidden_fragments": (
            "if (trim(adjustl(env)) /= '') then\n        lgpu = .false.",
        ),
    },
    {
        "name": "colab_zip_clean_snapshot_proof_generator",
        "path": "scripts/create_colab_gpu_zip.py",
        "fragments": (
            "--snapshot-clean-proof",
            "def create_clean_snapshot(root: Path, output: Path) -> Path:",
            "collect_source_files(root, output)",
            "run_checked([\"git\", \"init\", \"-q\"], snapshot)",
            "run_checked([\"git\", \"add\", \"-A\"], snapshot)",
            "Snapshot MOZYME GPU Colab proof source",
            "package_root(snapshot, output, allow_dirty=False)",
            "shutil.rmtree(snapshot, ignore_errors=True)",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "gpu_guide_resident_scf_abi_v33",
        "path": "docs/GPU_GUIDE.md",
        "fragments": (
            "Current resident-SCF ABI v33",
            "SCF loop-control state and return decision",
            "strict resident host synchronization/control-poll counters",
            "typed final publication proof",
            "full stage mask `1023`",
            "stage_missing=0",
            "final_reorth_applied",
            "PLS supervisor status",
            "resident COSMO CG control/convergence status",
            "`resident=1` final reorth marker",
            "resident_step` is not accepted as success",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "colab_strict_proof_defaults",
        "path": "colab/mopac_cublas_gpu_bench_colab.ipynb",
        "fragments": (
            "allow_development_dirty_zip = False",
            "allow_expected_dirty_source_zip = False",
            "trusted_expected_source_zip_sha256",
            "trusted_expected_source_zip_sha256 = str(trusted_expected_source_zip_sha256 or '').strip().lower()",
            "if not re.fullmatch(r'[0-9a-f]{64}', trusted_expected_source_zip_sha256):",
            "require_pinned_source_zip=True requires trusted_expected_source_zip_sha256 as a 64-character lowercase SHA-256 hex string.",
            "the uploaded sidecar alone is not a trusted freshness pin",
            "expected_source_proof_eligible",
            "expected_source_proof_ineligible_reason",
            "require_full_scf_gpu = True",
            "run_full_scf_readiness_probe = True",
            "allow_non_proof_run = False",
            "proof_eligible",
            "proof_ineligible_reason",
            "dirty_matches_expected_sidecar",
            "source_dirty_matches_expected_sidecar",
            "if require_full_scf_gpu and source_zip_dirty:",
            "Dirty source zip is proof-ineligible",
            "NON-PROOF DEVELOPMENT: dirty source zip accepted only because proof mode is disabled.",
            "if require_full_scf_gpu and not run_full_scf_readiness_probe:",
            "require_full_scf_gpu=True requires run_full_scf_readiness_probe=True.",
            "assert_full_scf_readiness_artifact",
            "assert_full_scf_readiness_cpu_compare_artifact",
            "assert_direct_cosmo_readiness_artifact",
            "full_scf_gpu_final_publication_done",
            "full_scf_gpu_final_publication_arrays",
            "full_scf_gpu_final_publication_bytes",
            "full_scf_gpu_final_publication_cosmo",
            "resident Fock plan mask must exactly match",
            "'mopac_executable_sha256', 'cmake_gpu_bool', 'cmake_cuda_architectures'",
            "run_direct_cosmo_readiness_probe = True",
            "direct_cosmo_readiness_passed = False",
            "BUILD_DIR / 'mopac-gpu-resident-fock-pair-compare'",
            "mopac_colab_resident_fock_pair_compare.log",
            "require_full_scf_gpu=True requires run_direct_cosmo_readiness_probe=True",
            "full_scf_probe_input = direct_cosmo_input",
            "direct_cosmo_readiness_passed = True",
            "Proof section ended without direct_cosmo_readiness_passed=True.",
            "--require-direct-cosmo-gpu",
            "cmd.append('--require-direct-cosmo-gpu')",
            "cmd.extend(['--full-scf-probe-input', str(direct_cosmo_input)])",
            "Direct COSMO readiness artifact did not prove resident GPU COSMO execution",
            "require_pinned_source_zip = True",
            "if require_full_scf_gpu and not require_pinned_source_zip:",
            "require_full_scf_gpu=True requires require_pinned_source_zip=True",
            "if require_full_scf_gpu and not expected_source_zip_sha256:",
            "require_full_scf_gpu=True requires an expected source zip SHA",
            "EXPECTED_MOPAC_GPU_FEATURE_SET",
            "EXPECTED_MOPAC_GPU_CONTRACT_VERSION",
            "EXPECTED_MOPAC_SOURCE_MARKER_CONTRACT_VERSION",
            "mopac-colab-expected-source-v1",
            "expected-source sidecar feature_set does not match this notebook contract",
            "expected-source sidecar full_scf_contract_version does not match this notebook contract",
            "expected-source sidecar source_marker_contract_version does not match this notebook contract",
            ".zip.expected.json",
            "expected_source_zip_sha256",
            "expected_source_git_dirty",
            "expected_source_dirty_status_sha256",
            "expected_source_proof_eligible",
            "expected_source_proof_ineligible_reason",
            "git.dirty_expected",
            "git.dirty_status_sha256_expected",
            "source.proof_eligible",
            "source.proof_ineligible_reason",
            "source_zip_sha256",
            "MOPAC_COLAB_SOURCE_ZIP_PATH",
            "source_zip_verified",
            "source_metadata_from_zip",
            "source_manifest_verified_files",
            "source_critical_files_verified",
            "source_required_critical_file_count",
            "source_required_critical_missing_count",
            "full_scf_gpu_cpu_mutating_call_count",
            "full_scf_gpu_cpu_mutating_ms",
            "full_scf_gpu_host_commit_cosmo",
            "MOZYME_STRICT_PROOF_ALLOWED_SECTION_NAMES",
            "mozyme_disallowed_strict_section_stats",
            "proof_identity_violation_reasons",
            "strict_readiness_fatal_status_lines",
            "helper_fatal_re",
            "fock1_batch",
            "fock2_4x1_batch",
            "fallback_real_pairs",
            "cpu_point_pairs",
            "_cnvgz_active_noop_ready",
            "required_stage_mask = 1023",
            "Experimental resident SCF required stage mask must be",
            "mozyme_gpu_helper_fatal_marker_count",
            "strict host route marker coverage",
            "strict_writmo_mecip_host_density",
            "strict_writmo_pm7ts_host_compfg",
            "strict_writmo_deriv_host_output",
            "strict helper aliases and direct CPU guards",
            "strict_cpu_makvec_direct",
            "strict_density_direct_host_rebuild",
            "strict_fock1_cpu_fallback",
            "stage mask mismatch: missing=",
            "full_scf_gpu_stage_completed has extra bits",
            "run_molecule_benchmark=True requires the strict readiness probe to pass first",
            "direct COSMO point-weight calls",
            "direct COSMO point-weight point",
            "hardcoded_critical_source_markers",
            "The uploaded zip failed hardcoded critical source validation",
            "run_mopac_text",
            "run_mopac has no raw stop",
            "source_git_dirty is not true/false",
            "source_metadata_valid",
            "source_manifest_file_sha256",
            "source feature contract version does not match current contract",
            "CUDA architecture identity was not recorded",
            "row.update(proof_identity)",
            "SOURCE_ALLOWLIST_POLICY_SCHEMA",
            "colab_source_path_allowed",
            "source allowlist policy",
            "density_cpu_diag_blocks",
            "density_cpu_offdiag_blocks",
            "strict_denout_host_output",
            "strict_olden_host_lmo_restore",
            "strict_pka_host_output",
            "resident loop control snapshot integer copy",
            "full_scf_gpu_pls_restart_reset_device_calls",
            "full_scf_gpu_pls_restart_done",
            "full_scf_gpu_compact_index_route",
            "full_scf_gpu_use_nijbo",
            "pair_count <= 0",
            "advance_strict_resident_control_on_gpu",
            "if (!advance_strict_resident_control_on_gpu(*ctx))",
            "return apply_resident_pls_restart_if_requested_on_gpu(ctx);",
            "return copy_resident_control_snapshot_from_gpu(ctx, out);",
            "MOZYME GPU relocal source contract",
            "mozyme_relocal_gpu_success_calls",
            "mozyme_relocal_gpu_fallback_calls",
            "mozyme_relocal_gpu_occupied_success_calls",
            "mozyme_relocal_gpu_virtual_success_calls",
            "mozyme_reorth_gpu_success_calls",
            "mozyme_reorth_gpu_fallback_calls",
            "MOPAC_MOZYME_RELOCAL_GPU",
            "MOPAC_MOZYME_REORTH_GPU",
            "--require-full-scf-gpu",
            "--full-scf-readiness-only",
            "--full-scf-readiness-cpu-compare",
            "full_scf_gpu_readiness_cpu_compare.json",
            "Strict full-SCF readiness CPU companion verified:",
            "Proof section ended without full_scf_readiness_passed=True.",
            "Direct COSMO readiness proof flag:",
            "_ordered_source_markers_present",
            "flat_pos = flat_text.find(flat_fragment, flat_cursor)",
            "DEFERRED_LOW_LEVEL_BENCHMARKS = {}",
            "def run_or_defer_low_level(name, callback):",
            "def run_deferred_low_level_benchmarks():",
            "Deferred {name} until strict readiness proof passes",
            "run_or_defer_low_level('quick low-level benchmark', _run_low_level_quick)",
            "run_or_defer_low_level('larger low-level benchmark', _run_low_level_large)",
            "run_or_defer_low_level('library timing benchmark', _run_low_level_library_timing)",
            "run_or_defer_low_level('low-level benchmark report', _run_low_level_report)",
            "Direct COSMO resident readiness already satisfied by strict full-SCF readiness probe; skipping duplicate probe.",
            "run_deferred_low_level_benchmarks()",
            "_scf_success_lines",
            "status=success code=0 ready=1 resident=1",
            "fatal_status_lines",
            "isitsc|tidy",
            "cnvgz_active_calls",
            "CNVGZ GPU stage work",
            "Experimental resident SCF reported fallback_cpu, strict_abort, or resident_step in a fatal smoke check.",
            "Experimental resident SCF did not report [MOZYME GPU SCF] status=success code=0 ready=1 resident=1.",
            "or '..' in posix_path.parts",
            "or '..' in rel_path.parts",
        ),
        "forbidden_fragments": (
            "allow_development_dirty_zip = True",
            "allow_expected_dirty_source_zip = True",
            "require_full_scf_gpu = False",
            "allow_non_proof_run = True",
            "DEVELOPMENT PROOF:",
            "if require_full_scf_gpu and not (run_full_scf_readiness_probe or run_molecule_benchmark):",
            "'if (!strict_resident && !copy_resident_state_to_host(*ctx))' in scf_cuda_text",
            "or '..' in normalized",
            "or '..' in rel:",
        ),
        "ordered_fragments": (
            "source_zip_sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()",
            "trusted_expected_source_zip_sha256 = str(trusted_expected_source_zip_sha256 or '').strip().lower()",
            "if require_pinned_source_zip:",
            "if not re.fullmatch(r'[0-9a-f]{64}', trusted_expected_source_zip_sha256):",
            "if source_zip_sha256 != trusted_expected_source_zip_sha256:",
            "with zipfile.ZipFile(zip_path) as zf:",
            "zf.extractall(SOURCE_ROOT)",
            "run_or_defer_low_level('quick low-level benchmark', _run_low_level_quick)",
            "run_or_defer_low_level('larger low-level benchmark', _run_low_level_large)",
            "run_or_defer_low_level('library timing benchmark', _run_low_level_library_timing)",
            "run_or_defer_low_level('low-level benchmark report', _run_low_level_report)",
            "run_direct_cosmo_readiness_probe = True",
            "direct_cosmo_readiness_passed = False",
            "if require_full_scf_gpu and not run_direct_cosmo_readiness_probe:",
            "readiness_cmd = [",
            "--full-scf-readiness-cpu-compare",
            "direct_cosmo_payload = assert_direct_cosmo_readiness_artifact(full_scf_readiness_report_dir / 'full_scf_gpu_readiness.json')",
            "assert_full_scf_readiness_cpu_compare_artifact(full_scf_readiness_report_dir / 'full_scf_gpu_readiness_cpu_compare.json', readiness_payload)",
            "direct_cosmo_readiness_passed = True",
            "Direct COSMO resident readiness already satisfied by strict full-SCF readiness probe; skipping duplicate probe.",
            "run_deferred_low_level_benchmarks()",
            "if require_full_scf_gpu and not direct_cosmo_readiness_passed:",
            "Direct COSMO readiness proof flag:",
        ),
    },
    {
        "name": "colab_source_allowlist_policy",
        "path": "scripts/create_colab_gpu_zip.py",
        "fragments": (
            'SOURCE_ALLOWLIST_POLICY_SCHEMA = "mopac-colab-source-allowlist-v1"',
            'SOURCE_MARKER_CONTRACT_VERSION = "mopac-colab-source-markers-explicit-proof-v83"',
            'SOURCE_MARKER_CONTRACT_SHA256 = "',
            "def source_marker_contract_sha256() -> str:",
            "validate_source_marker_contract_sha256()",
            "ALLOWED_SOURCE_ROOT_FILES",
            "ALLOWED_SOURCE_ROOT_DIRS",
            "def allowed_by_source_policy(path: Path, root: Path) -> bool:",
            '"source_allowlist_policy": source_policy_payload()',
            '"source_marker_contract_version": SOURCE_MARKER_CONTRACT_VERSION',
            '"source_marker_contract_sha256": SOURCE_MARKER_CONTRACT_SHA256',
            "Refusing to package files outside the Colab source allowlist policy",
            "def write_expected_source_sidecar(",
            "mopac-colab-expected-source-v1",
            '"feature_set": FEATURE_SET',
            '"full_scf_contract_version": FULL_SCF_CONTRACT_VERSION',
            '"source_marker_contract_version": SOURCE_MARKER_CONTRACT_VERSION',
            '"proof_eligible": proof_eligible',
            '"proof_ineligible_reason": proof_ineligible_reason',
            "output.name + \".expected.json\"",
            "Wrote {expected_sidecar}",
            "def validate_notebook_code_cells(",
            "ast.parse(code, filename=f\"colab-cell-{index}\")",
            "Colab notebook code cell failed Python syntax check",
            "path.is_symlink()",
            "Refusing to package symlinked source files",
            "path.resolve().relative_to(root_resolved)",
            "Refusing to package files that resolve outside the source root",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_sparse_fock_setup_host_bounds",
        "path": "src/gpu/cuda_wrappers.cu",
        "fragments": (
            "mozyme_sparse_fock_basis_supported",
            "auto valid_range = [](int offset, long long count, int limit) -> bool",
            "pair_diag_flags[idx] != 0 && pair_diag_flags[idx] != 1",
            "valid_range(pair_cross_offsets[idx], static_cast<long long>(iab) * jba",
            "!mozyme_sparse_fock_basis_supported(iab)",
            "valid_range(point_i_offsets[idx], tri(iab), mpack)",
            "int full_coverage",
            "plan->full_coverage = full_coverage != 0",
            "bool has_executable_work = false",
            "plan->has_executable_work = false;",
            "const long long executable_work =",
            "if (full_coverage != 0 && executable_work <= 0) return fail_setup(3);",
            "plan->has_executable_work = executable_work > 0",
            "if (plan && complete == 0) plan->full_coverage = false;",
            "!plan || !plan->ready || !plan->full_coverage",
            "!plan->has_executable_work",
            "return fail_setup(3)",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_scf_partial_fock_bounded_copy_back",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "const std::size_t partp_count = std::min(",
            "static_cast<std::size_t>(ctx.state.partp_dim), mpack_count);",
            "const std::size_t partf_count = std::min(",
            "static_cast<std::size_t>(ctx.state.partf_dim), mpack_count);",
            "stage_and_commit_device_vector(ctx.state.partp, dev.partp, partp_count",
            "stage_and_commit_device_vector(ctx.state.partf, dev.partf, partf_count",
        ),
        "forbidden_fragments": (
            "stage_device_to_vector(host_partp, dev.partp, partp_count",
            "stage_device_to_vector(host_partf, dev.partf, partf_count",
            "commit_staged_vector(ctx.state.partp, host_partp)",
            "commit_staged_vector(ctx.state.partf, host_partf)",
            "std::vector<double> host_partp(mpack_count);",
            "std::vector<double> host_partf(mpack_count);",
            "copy_device_to_host_vector(host_partp, dev.partp)",
            "commit_host_vector(ctx.state.partp, host_partp)",
        ),
    },
    {
        "name": "strict_full_scf_report_contract_final_resident_reorth_tidy_selmos_pls_cosmo_direct_resident_cg_point_kind_cnvgz_active_or_noop_cpu_compare_explicit_proof_v58",
        "path": "scripts/molecule_benchmark_report.py",
        "fragments": (
            'MOPAC_GPU_READINESS_CONTRACT_VERSION = "resident-scf-strict-final-resident-reorth-tidy-selmos-pls-reset-cosmo-direct-resident-cg-point-kind-cnvgz-active-or-noop-cpu-compare-explicit-proof-v58"',
            'MOPAC_GPU_FEATURE_SET = "mozyme-full-scf-gpu-makvec-relocal-final-resident-reorth-tidy-selmos-pls-reset-cosmo-direct-point-kind-cnvgz-active-or-noop-cpu-compare-explicit-proof-v38-20260702"',
            'MOPAC_GPU_BENCHMARK_SCOPE = "molecule_mozyme_full_scf_gpu"',
            '"full_scf_contract_version": MOPAC_GPU_READINESS_CONTRACT_VERSION',
            '"publication_claim": MOPAC_GPU_PUBLICATION_CLAIM',
            "MOZYME_SCF_PLS_RE",
            "MOZYME_SCF_PLS_RESET_RE",
            "MOZYME_SCF_COSMO_RE",
            "MOZYME_SCF_COSMO_CG_RE",
            "MOZYME_SCF_COMPACT_RE",
            "MOZYME_SCF_COSMO_ENERGY_RE",
            "MOZYME_SCF_OLDEN_SETUP_RE",
            "MOZYME_SCF_CPU_SETUP_RE",
            "MOZYME_SCF_FILLIJ_GPU_RE",
            "MOZYME_RESIDENT_FOCK_GPU_COUNT_RE",
            "MOZYME_RESIDENT_FOCK_GPU_PACK_RE",
            "MOZYME_RESIDENT_FOCK_GPU_POINT_WEIGHTS_RE",
            "MOZYME_RESIDENT_FOCK_CPU_PLAN_RE",
            "MOZYME_SCF_HOST_COMMIT_RE",
            "MOZYME_SCF_FINAL_PUBLICATION_RE",
            "MOZYME_CPU_PINOUT_RE",
            "full_scf_gpu_final_publication_done",
            "full_scf_gpu_final_publication_arrays",
            "full_scf_gpu_final_publication_bytes",
            "full_scf_gpu_final_publication_cosmo",
            "full_scf_gpu_pls_restart_required",
            "resident PLS restart must be unnecessary or completed on GPU",
            "full_scf_gpu_pls_restart_required must finish at 0",
            "full_scf_gpu_pls_restart_reset_device_calls",
            "full_scf_gpu_pls_restart_done",
            "full_scf_gpu_compact_index_route",
            "full_scf_gpu_use_nijbo",
            "full_scf_gpu_cosmo_fock_calls",
            "full_scf_gpu_cosmo_matvec_calls",
            "full_scf_gpu_cosmo_last_residual",
            "full_scf_gpu_cosmo_cg_control_resident",
            "full_scf_gpu_cosmo_cg_converged",
            "full_scf_gpu_cosmo_cg_breakdown",
            "full_scf_gpu_cosmo_cg_host_syncs",
            "full_scf_gpu_cosmo_cg_target_tol",
            "validate_full_scf_cosmo_fields",
            "validate_direct_cosmo_contract",
            "validate_resident_plan_coverage",
            "requires_direct_cosmo_gpu",
            "mozyme_plan_direct_mode",
            "mozyme_fock_resident_full_coverage_planned",
            "mozyme_fock_resident_executable_tasks",
            "mozyme_fock_resident_direct_unsupported_pairs",
            "mozyme_fock_resident_direct_unsupported_point_pairs",
            "mozyme_resident_fock_direct_basis_fallback_pairs",
            "mozyme_resident_fock_direct_basis_point_fallback_pairs",
            'parse_int_value(row.get("requires_direct_cosmo_gpu")) == 1',
            "row_uses_eps",
            "--require-full-scf-gpu requires --require-direct-cosmo-gpu for the v58",
            "--require-full-scf-gpu requires --full-scf-readiness-cpu-compare",
            "--require-direct-cosmo-gpu",
            "--full-scf-readiness-cpu-compare",
            "--full-scf-readiness-cpu-compare requires --require-full-scf-gpu",
            "run_full_scf_readiness_cpu_compare",
            "cpu_companion_gpu_leakage_reasons",
            "cpu_only_reasons",
            "CPU companion reported GPU work units",
            "CPU companion finished with lgpu=T",
            "full_scf_gpu_readiness_cpu_compare.json",
            "strict_full_scf_readiness_cpu_companion",
            "Full SCF readiness CPU companion passed",
            "same-input CPU/GPU energy comparison",
            "DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_ABS_TOL = 5.0e-3",
            "DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_REL_TOL = 1.0e-5",
            "DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_PER_ATOM_TOL = 5.0e-4",
            "CPU companion did not finish normally with a parsed heat of formation",
            "strict readiness CPU/GPU heat comparison failed",
            "strict readiness CPU/GPU heat comparison cannot pass on",
            "per-atom tolerance alone",
            "requires a full SCF probe input with EPS=78.4",
            "direct COSMO proof planned point-charge/dipole Fock work but did not run",
            "direct COSMO proof did not report direct integral mode in the MOZYME plan",
            "direct COSMO proof encountered resident Fock direct-basis fallback",
            "MOZYME GPU plan reported incomplete resident Fock coverage",
            "positive resident point work",
            "full_scf_gpu_stage_completed must exactly equal",
            "input_eps",
            "source_zip_sha256",
            "source_zip_verified",
            "source_metadata_from_zip",
            "source_manifest_verified_files",
            "source_manifest_missing_file_count",
            "source_manifest_hash_mismatch_count",
            "source_critical_files_verified",
            "source_required_critical_file_count",
            "source_required_critical_missing_count",
            "source_required_critical_missing_files",
            "source_required_marker_count",
            "source_required_markers_verified",
            "source_required_marker_violation_count",
            "verify_required_source_markers",
            "required source markers invalid",
            "mozyme_gpu_relocal_fortran_wrapper marker is missing",
            "success|fallback_cpu|strict_abort",
            "MOZYME_GPU_HELPER_FATAL_RE",
            "MOZYME_TIDY_RE",
            "fock[12]\\]\\s+fallback",
            "fock1_batch\\]\\s+fallback",
            "fock2_4x1_batch\\]\\s+fallback",
            "resident_fock\\]\\s+fallback_real_pairs\\s+total\\s*=\\s*[1-9]\\d*",
            'return "helper_fatal"',
            "mozyme_gpu_helper_fatal_marker_count",
            "mozyme_gpu_helper_fatal_markers",
            "source_manifest_sha256",
            "source_manifest_file_sha256",
            "source_metadata_valid",
            "source_metadata_present",
            "source zip path was not present and hash-verified",
            "packaged source metadata was not verified from the uploaded zip",
            "source manifest verified file count does not match entry count",
            "critical source file verification count does not match",
            "required critical source files are missing from the feature manifest",
            "source_provenance_contract_version",
            "source_features_contract_version",
            "SOURCE_MARKER_CONTRACT_VERSION",
            "source_provenance_marker_contract_version",
            "source_features_marker_contract_version",
            "source marker contract version mismatch",
            "source feature contract version does not match current contract",
            "cmake_cuda_architectures",
            "full_scf_gpu_olden_setup_only",
            "full_scf_gpu_fillij_gpu_count_calls",
            "full_scf_gpu_fillij_gpu_fill_calls",
            "full_scf_gpu_resident_fock_gpu_count_calls",
            "full_scf_gpu_resident_fock_gpu_count_plan_id",
            "full_scf_gpu_resident_fock_gpu_count_full_coverage",
            "full_scf_gpu_resident_fock_gpu_pack_calls",
            "full_scf_gpu_resident_fock_gpu_pack_plan_id",
            "full_scf_gpu_resident_fock_gpu_pack_full_coverage",
            "full_scf_gpu_resident_fock_gpu_point_weight_calls",
            "full_scf_gpu_resident_fock_gpu_point_weight_max_abs_diff",
            "full_scf_gpu_cpu_mozyme_setup_only_calls",
            "full_scf_gpu_cpu_resident_fock_plan_setup_calls",
            "full_scf_gpu_cpu_resident_fock_plan_setup_full_coverage",
            "full_scf_gpu_host_commit_only_calls",
            "full_scf_gpu_host_commit_phase",
            "full_scf_gpu_host_commit_arrays",
            "full_scf_gpu_host_commit_bytes",
            "full_scf_gpu_host_commit_cosmo",
            "full_scf_gpu_cpu_pinout_calls",
            "validate_strict_host_route_contract",
            "validate_full_scf_host_commit_contract",
            "resident-SCF final host publication did not emit host_commit_only=1",
            "resident-SCF host publication did not report phase=final_publication",
            "host commit arrays and typed final publication arrays disagree",
            "host commit bytes and typed final publication bytes disagree",
            "host commit COSMO flag and typed final publication COSMO flag disagree",
            "direct COSMO strict proof did not publish COSMO state in the final host commit",
            "OLDEN/OLDENS host LMO restore marker appeared in strict GPU proof",
            "strict resident SCF did not report GPU fillij count setup",
            "strict resident SCF did not report GPU fillij nijbo setup",
            "strict resident SCF did not report GPU resident-Fock plan pack",
            "CPU MOZYME array setup marker appeared in strict GPU proof",
            "CPU resident Fock plan construction marker appeared in strict GPU proof",
            "CPU pinout host I/O marker appeared",
            "strict GPU proof produced .den host checkpoint artifact(s)",
            "backend_pls_restart_required",
            "gpu_error_marker_count was not reported",
            "full_scf_gpu_stage_required is not",
            "MOZYME section profile markers were not present",
            "MOZYME makvec initial LMO construction did not report GPU success",
            "OLD_SCF existing-LMO marker is not accepted as complete GPU makvec proof",
            "MOZYME_MAKVEC_EXISTING_RE",
            "mozyme_makvec_gpu_existing_lmo_calls",
            "MOZYME_SETUPK_RE",
            "mozyme_setupk_gpu_success_calls",
            "MOZYME setupk GPU did not report a success marker",
            "mozyme_setupk_gpu_fallback_calls",
            "mozyme_setupk_gpu_initial_setup_fallback_calls",
            "mozyme_setupk_gpu_initial_setup_success_calls",
            "MOZYME setupk GPU did not report an initial_setup=1 success marker",
            "mozyme_setupk_gpu_initial_setup_fallback_calls",
            "MOZYME setupk GPU reported an initial_setup=1 fallback marker",
            "mozyme_setupk_gpu_initial_setup_all_paths_calls",
            "MOZYME setupk all-initial-setup path marker was not reported",
            "mozyme_setupk_gpu_initial_setup_last_ms",
            "MOZYME_RELOCAL_RE",
            "mozyme_relocal_gpu_success_calls",
            "mozyme_relocal_gpu_fallback_calls",
            "mozyme_relocal_gpu_occupied_success_calls",
            "mozyme_relocal_gpu_virtual_success_calls",
            "mozyme_relocal_success_types",
            "MOZYME_REORTH_RE",
            "mozyme_reorth_gpu_success_calls",
            "mozyme_reorth_gpu_resident_success_calls",
            "mozyme_reorth_gpu_fallback_calls",
            "mozyme_tidy_gpu_success_calls",
            "mozyme_tidy_gpu_occupied_success_calls",
            "mozyme_tidy_gpu_virtual_success_calls",
            "mozyme_tidy_gpu_selmos_success_calls",
            "mozyme_tidy_gpu_fallback_calls",
            "strict resident SCF did not report occupied GPU TIDY success",
            "strict resident SCF did not report virtual GPU TIDY success",
            'marker_field_equals(extra, "resident", "1")',
            "FULL_SCF_PROBE_FORCED_KEYWORDS",
            "FULL_SCF_PROBE_FORCED_ENV",
            "MOPAC_MOZYME_SCF_STRICT_RESIDENT=1",
            "MOPAC_MOZYME_GPU_STRICT=1",
            "MOPAC_MOZYME_FULL_SCF_GPU=1",
            "MOPAC_MOZYME_RESIDENT_FOCK_GPU=1",
            "MOPAC_MOZYME_DIAGG1_AOCC_GPU=1",
            "MOPAC_MOZYME_DIAGG1_AVIR_GPU=1",
            "MOPAC_MOZYME_DIAGG2_ROTPREP_GPU=1",
            "MOPAC_MOZYME_TIDY_GPU=1",
            "force_full_scf_probe_keywords",
            "full_scf_probe_forced_keywords",
            "full_scf_probe_forced_env",
            'env["MOPAC_MOZYME_SCF_GPU"] = "1"',
            'env["MOPAC_MOZYME_FULL_SCF_GPU"] = "1"',
            'env["MOPAC_MOZYME_GPU_STRICT"] = "1"',
            'env["MOPAC_MOZYME_SCF_FORCE_FINAL_REORTH"] = "1"',
            "forced REORTH probe did not report a resident=1 final reorth success marker",
            "CPU final reorthogonalization sections ran in strict proof",
            "[MOZYME GPU reorth] status=success resident=1",
            "mozyme_disallowed_strict_section_stats(mozyme_section_times, combined_text)",
            'env["MOPAC_MOZYME_RELOCAL_GPU"] = "1"',
            'env["MOPAC_MOZYME_REORTH_GPU"] = "1"',
            "select_resident_fock_coverage",
            "real_pairs != gpu_pairs + cpu_pairs",
            "resident sparse Fock GPU reported zero run calls",
            "resident sparse Fock GPU reported zero work tasks",
            "resident sparse Fock GPU reported {sparse_zero_work_calls} zero-work run call(s)",
            "semantic point-charge/dipole coverage",
            "validate_resident_point_charge_coverage",
            "resident sparse Fock point-charge/dipole work must be semantically covered when present",
            "full_scf_gpu_resident_fock_gpu_point_weight_point must be positive",
            "GPU resident-Fock point-weight coverage is incomplete",
            "resident sparse Fock point-charge/dipole setup coverage is incomplete",
            "resident sparse Fock point-charge/dipole run coverage is incomplete",
            "resident sparse Fock point-dipole setup coverage is incomplete",
            "resident sparse Fock point-dipole run coverage is incomplete",
            "MOZYME point-charge/dipole plan counters are inconsistent",
            "mozyme_fock_plan_point_charge_pairs was not reported",
            "mozyme_fock_plan_point_charge_pairs",
            "mozyme_resident_fock_point_pairs",
            "mozyme_resident_fock_gpu_point_pairs",
            "mozyme_resident_fock_cpu_point_pairs",
            "mozyme_resident_fock_cpu_point_pair_fatal_max",
            "MOZYME_RESIDENT_FOCK_POINT_COVERAGE_RE",
            "MOZYME_RESIDENT_FOCK_CPU_POINT_PAIRS_RE",
            "parse_resident_fock_point_coverage",
            "resident_fock\\][^\\n]*\\bcpu_point_pairs",
            "mozyme_fock_plan_point_dipole_pairs",
            "mozyme_fock_plan_point_monopole_pairs",
            "mozyme_sparse_fock_setup_point_tasks",
            "mozyme_sparse_fock_setup_point_dipole_tasks",
            "mozyme_sparse_fock_setup_point_monopole_tasks",
            "mozyme_sparse_fock_run_point_tasks",
            "mozyme_sparse_fock_run_point_dipole_tasks",
            "mozyme_sparse_fock_run_point_monopole_tasks",
            "point_dipole=(\\d+)",
            "point_monopole=(\\d+)",
            "resident sparse Fock GPU did not report runtime",
            "resident sparse Fock GPU runtime is negative",
            "mozyme_sparse_fock_run_zero_work_calls",
            "resident sparse Fock must run real GPU work every resident iteration",
            "MOPAC returned non-zero exit code",
            "MOPAC did not finish normally with a parsed heat of formation",
            "strict readiness probe decision was not complete",
            "full_scf_probe_decision",
            "full_scf_gpu_requested is not 1",
            "full_scf_gpu_executed is not 1",
            "mozyme_scf_experimental_executed is not 1",
            "full_scf_gpu_scf_success_calls",
            "full_scf_gpu_scf_fallback_calls",
            "full_scf_gpu_resident_step_calls",
            "FULL_SCF_GPU_FALLBACK_KEYS",
            "RESIDENT_FOCK_FALLBACK_KEYS",
            "dict.fromkeys((*FULL_SCF_GPU_FALLBACK_KEYS, *RESIDENT_FOCK_FALLBACK_KEYS))",
            "full_scf_gpu_cpu_boundary_calls",
            'mode == "GPU" and parse_int_value(row.get("full_scf_gpu_requested")) == 1',
            'parse_int_value(row.get("full_scf_gpu_ready")) != 1',
            "full_scf_gpu_contract_violation_reasons(row)",
            "MOZYME_SCF_STRICT_ABORT_REASON_RE",
            "full_scf_gpu_strict_host_route_marker_count",
            "full_scf_gpu_strict_host_route_markers",
            "full_scf_gpu_pls_restart_required_calls",
            "full_scf_gpu_pls_restart_required",
            "full_scf_gpu_pls_restart_reset_device_calls",
            "full_scf_gpu_pls_restart_done",
            "full_scf_gpu_compact_index_route",
            "full_scf_gpu_use_nijbo",
            "full_scf_gpu_cpu_mutating_call_count",
            "full_scf_gpu_cpu_mutating_ms",
            '"*.out", "*.arc", "*.aux", "*.res", "*.den", "*.DEN"',
            "density_cpu_diag_blocks",
            "density_cpu_offdiag_blocks",
            "mozyme_fock1_gpu_fallback_seen",
            "mozyme_fock2_gpu_fallback_seen",
            "full_scf_gpu_wall_ms was not reported",
            "full_scf_gpu_wall_ms is negative",
            "MOZYME_SCF_STAGE_NAMES",
            "MOZYME_SCF_STAGE_NAMES_RAW_RE",
            "MOZYME_SCF_STRICT_PROOF_RE",
            "MOZYME_SCF_RESIDENT_FOCK_PLAN_RE",
            "MOZYME_SCF_CNVGZ_ACTIVITY_RE",
            "full_scf_gpu_stage_completed_names_raw",
            "full_scf_gpu_strict_resident",
            "full_scf_gpu_resident_fock_plan_full_coverage",
            "full_scf_gpu_resident_fock_plan_partial_coverage",
            "full_scf_gpu_resident_fock_plan_required_mask",
            "full_scf_gpu_resident_fock_plan_covered_mask",
            "resident Fock covered mask must exactly match required plans",
            "FULL_SCF_READINESS_CPU_COMPARE_BINDING_FIELDS",
            "readiness_binding",
            "MOZYME_SCF_STAGE_CALL_FIELDS",
            "MOZYME_SCF_STAGE_MS_FIELDS",
            "MOZYME_SCF_STAGE_CALLS_RE",
            "MOZYME_SCF_STAGE_MS_RE",
            "success|fallback_cpu|resident_step|strict_abort",
            "strict_abort",
            "cnvgz_active_noop_ready",
            "append_cnvgz_active_noop_reasons",
            "parse_scf_cnvgz_activity",
            "full_scf_gpu_stage_{name}_calls",
            "full_scf_gpu_stage_{name}_ms",
            "full_scf_gpu_cnvgz_active_calls",
            "full_scf_gpu_cnvgz_noop_calls",
            "resident-SCF CNVGZ active-call counter was not reported",
            "resident-SCF CNVGZ did not report active or no-op GPU stage work",
            "resident-SCF stage {stage_name} calls=",
            "required_resident_stage_calls(stage_name, min_stage_calls)",
            "Resident SCF stage counters:",
            "full_scf_gpu_stage_completed_names",
            "full_scf_gpu_stage_missing_names",
            "full_scf_gpu_energy_total was not parseable",
            "full_scf_gpu_diagg_sumt was not parseable",
            "full_scf_gpu_diagg_sumb was not parseable",
            "backend_cpu_boundary",
            "backend_pls_restart_required",
            "strict_denout_host_output",
            "strict_olden_host_lmo_restore",
            "strict_pka_host_output",
            "strict_no_gpu_device",
            "strict_not_gpu_build",
            "strict_gpu_disabled_by_nogpu_keyword",
            "strict_gpu_disabled_by_mopac_nogpu",
            "strict_gpu_disabled_by_scftask_cpu",
            "strict_gpu_disabled_by_mozyme_gpu_off",
            "strict_resident_fock_gpu_disabled",
            "strict_setup_mozyme_arrays_cpu_setup",
            "strict_resident_fock_cpu_plan_setup",
            "strict_resident_fock_gpu_pack_failed",
            "strict_fillij_gpu_failed",
            "strict_fillij_gpu_unavailable",
            "strict_fillij_nijbo_missing",
            "strict_add_more_interactions_cpu_fallback",
            "strict_addhb_cpu_fallback",
            "strict_check_cpu_fallback",
            "strict_check_gpu_host_fallback",
            "strict_diagg_cpu_fallback",
            "strict_fock1_cpu_fallback",
            "strict_fock2z_cpu_fallback",
            "strict_fock2_4x1_batch_cpu_fallback",
            "strict_buildf_cpu_fallback",
            "strict_setupk_cpu_fallback",
            "strict_cnvgz_cpu_fallback",
            "strict_eimp_cpu_fallback",
            "strict_helecz_cpu_fallback",
            "strict_isitsc_cpu_fallback",
            "strict_diagg1_cpu_fallback",
            "strict_diagg2_cpu_fallback",
            "strict_density_batch_fallback",
            "strict_reorth_cpu_fallback",
            "strict_tidy_cpu_fallback",
            "strict_pinout_cpu_fallback",
            "strict_nijbo_alloc_failed",
            "strict_no_gpu_work",
            "strict_cpu_makvec",
            "strict_lewis_gpu_makvec_failed",
            "strict_cpu_relocalization",
            "strict_early_probe_fallback",
            "strict_pre_tidy_failed",
            "strict_pre_tidy_not_started",
            "strict_post_tidy_failed",
            "strict_cpu_iteration_work",
            "strict_cpu_lmo_check",
            "strict_cpu_pls_supervisor",
            "strict_cpu_pls_restart",
            "strict_cpu_loop_control",
            "strict_cpu_scf_body",
            "strict_missing_final_density",
            "strict_cpu_reorthogonalization",
            "strict_solvent_fock",
            "strict_resident_fock_disabled",
            "strict_resident_fock_legacy_host_copy",
            "strict_resident_fock_prepare_failed",
            "strict_resident_fock_partial_coverage",
            "strict_resident_fock_work_alloc_failed",
            "strict_resident_fock_run_failed",
            "strict_resident_fock_cpu_fallback",
            "strict_legacy_fock2_failed",
            "strict_density_direct_host_rebuild",
            "strict_cpu_makvec_direct",
            "strict_run_mopac_olden_host_restore",
            "strict_writmo_reloc_host_output",
            "strict_writmo_mecip_host_density",
            "strict_writmo_pm7ts_host_compfg",
            "strict_writmo_deriv_host_output",
            "strict_writmo_fock_host_output",
            "strict_writmo_denout_host_output",
            "[MOZYME GPU SCF] host_commit_only=1",
            "phase=final_publication",
            "source_zip_sha256",
            "source_manifest_sha256",
            "mopac_executable_sha256",
            "cmake_cuda_architectures",
            "FULL_SCF_GPU_PROVEN",
            "FULL_SCF_GPU_DEVELOPMENT_PROOF",
            "NOT_FULL_SCF_GPU_PROOF",
            "energy_status",
            "proof_status",
            "no resident-SCF fallback_cpu, strict_abort, or resident_step marker may appear anywhere in the log",
            "GPU debug state did not confirm hasGPU=T",
            "final MOPAC GPU switch lgpu is not T",
            "resident sparse Fock GPU was not enabled in the MOZYME GPU plan",
            "resident sparse Fock coverage reported zero GPU real pairs",
            "mozyme_resident_fock_real_pairs was not reported",
            "resident sparse Fock real-pair coverage is incomplete",
            "resident sparse Fock coverage counters are inconsistent",
            "all parsed fallback counters",
            "full_scf_gpu_final_iterations must be at least 1",
            'env["MOPAC_MOZYME_ISITSC_GPU"] = "1"',
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_fock_4x1_pair_compare",
        "path": "tests/gpu_resident_fock_pair_compare.F90",
        "fragments": (
            "run_pair4x1_case(.true., '4x1 resident')",
            "run_pair4x1_case(.false., '1x4 resident')",
            "run_case(1, 1, '1x1')",
            "run_case(1, 9, '1x9')",
            "run_case(9, 1, '9x1')",
            "run_case(4, 9, '4x9')",
            "run_case(9, 4, '9x4')",
            "run_case(9, 9, '9x9')",
            "run_diag_case(4, '4x4 diagonal')",
            "run_diag_case(9, '9x9 diagonal')",
            "run_point_case(1, 9, -2",
            "run_point_case(9, 1, -2",
            "pair4_heavy_offsets(1) = int(heavy_offset, c_int)",
            "pair4_light_offsets(1) = int(light_offset, c_int)",
            "pair4_cross_offsets(1) = int(cross_offset, c_int)",
            "apply_pair4x1_cpu",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_fock_diagonal_pair_validation",
        "path": "src/gpu/cuda_wrappers.cu",
        "fragments": (
            "pair_diag_flags[idx] == 0 &&",
            "static_cast<long long>(iab) * jba, mpack",
            "pair_diag_flags[idx] == 1 && pair_cross_offsets[idx] < 1",
        ),
        "forbidden_fragments": (
            "if (!valid_range(pair_cross_offsets[idx], static_cast<long long>(iab) * jba,\n                     mpack))",
        ),
    },
    {
        "name": "resident_fock_point_dipole_pair_compare",
        "path": "tests/gpu_resident_fock_pair_compare.F90",
        "fragments": (
            "run_point_case(9, 4, -2",
            "run_point_case(9, 9, -2",
            "run_point_case(9, 9, -1",
            "apply_point_cpu",
            "point_addr_flags(1) = int(addr_flag, c_int)",
            "resident sparse point-charge/dipole 9-orbital comparison",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_fock_point_addr_preserved",
        "path": "src/MOZYME/mozyme_resident_fock.F90",
        "fragments": (
            "mozyme_resident_basis_supported",
            "nbasis == 1 .or. nbasis == 4 .or. nbasis == 9",
            "point_addr_flags(point_pos) = mozyme_c_int_checked(addr)",
            "mozyme_resident_point_supported(iab, jba, addr)",
            "mozyme_resident_direct_basis_supported",
            "mozyme_resident_direct_basis_supported = nbasis == 1 .or. nbasis == 4 .or. nbasis == 9",
            "mozyme_resident_direct_basis_fallback",
            "max_resident_fallback_basis = max_resident_diag_basis + 1",
            "resident_fallback_basis_bin",
            "real_pair_direct_basis_count",
            "point_pair_direct_basis_count",
            "reason=unsupported_direct_basis",
            "logical :: plan_counts_ok",
            "setup rejected count mismatch",
            "counts_ok = one_pos == one_count .and. pair_pos == pair_count",
            "resident_full_coverage(plan_id) = one_center_cpu_count == 0 .and. &",
            "real_pair_cpu_count == 0 .and. point_pair_cpu_count == 0",
            "int(gpu_pack_counts(18)) == one_center_cpu_count",
            "one_center_coverage gpu_one_center=",
            "if (one_center_cpu_count == 0 .and. real_pair_cpu_count == 0 .and. point_pair_cpu_count == 0) then",
            "point_coverage point_pairs=",
            "call increment_fallback_basis(iab, jba, fallback_basis)",
        ),
        "forbidden_fragments": (
            "point_addr_flags(point_pos) = mozyme_c_int_nonnegative_or_zero(addr)",
        ),
    },
    {
        "name": "resident_fock_direct_d_shell_gpu_pack",
        "path": "src/gpu/cuda_wrappers.cu",
        "fragments": (
            "mozyme_direct_spd_w_dev",
            "mozyme_direct_reppd2_rep_dev",
            "mozyme_mndod_ind2_dev",
            "mozyme_mndod_isym_dev",
            "kMozymeMndodIpos",
            "kMozymeResidentDirectPackScratchDoubles",
            "g_mz_res_pack_direct_scratch",
            "direct_scratch + kMozymeDirectMaxW",
            "kMozymeDirectSpdScratchReppdArg",
            "kMozymeDirectSpdScratchReppdSqr",
            "kMozymeDirectSpdScratchRotP",
            "kMozymeDirectSpdScratchRotD",
            "method_pm7_flag",
            "po, ddp, iod",
            "(iab == 9 || jba == 9)\n                    ? mozyme_direct_spd_w_dev",
        ),
        "forbidden_fragments": (
            "if (direct_flag != 0 && semidr_flag == 0) return 4;",
            "double direct_w[2025]",
            "double ww[2025]",
            "double rep[492]",
            "double v[46][46]",
            "double direct_w[100]",
            "double arg[72]",
            "double sqr[72]",
            "double arg[7]",
            "double sqr[7]",
            "double p[3][3]",
            "double d[5][5]",
            "double v[26][11]",
            "double ww[100]",
            "bool logv",
        ),
    },
    {
        "name": "strict_resident_fock_gpu_pack_plan",
        "path": "src/MOZYME/mozyme_resident_fock.F90",
        "fragments": (
            "mopac_cuda_mozyme_resident_fock_pack_plan",
            "gpu_count=1 plan_id=",
            "gpu_pack=1 plan_id=",
            "gpu_point_weights=1 point=",
            "source=pack",
            "mozyme_resident_direct_sp_basis_supported",
            "mozyme_resident_direct_sp_basis_supported = nbasis == 1 .or. nbasis == 4",
            "merge(1_c_int, 0_c_int, direct)",
            "am, ad, aq, dd, qq",
            "jindex(m) = ifact(lk) + kl",
            "strict_resident_fock_gpu_pack_failed",
            "strict_resident_fock_gpu_pack_required",
            "resident_gpu_pack_ready",
            "mopac_cuda_mozyme_sparse_fock_plan_ready",
            "use iso_c_binding, only: c_int64_t",
            "integer(c_int64_t), save :: last_signature",
            "signature = last_signature(plan_id) + 1_c_int64_t",
            "signature = mozyme_resident_signature(iorbs, nat, ifact, wj, wk,",
            "signature, gpu_pack_full_coverage",
            "point_w_values, signature, coverage_complete",
            "gpu_pack_stale=1 plan_id=",
            "gpu_pack_required=1 plan_id=",
            "resident_full_coverage(plan_id) = gpu_pack_full_coverage /= 0_c_int",
            "log_resident_coverage_gpu_counts",
        ),
        "forbidden_fragments": (
            "mopac_cuda_mozyme_resident_fock_count_plan",
            "mopac_cuda_mozyme_resident_fock_point_weights",
            "MOZYME GPU strict resident Fock pack setup disagreed with CPU count reference",
            "strict_resident_fock_cpu_count_reference",
            "cpu_plan_constructed=1 setup_only=1 plan_id=",
            "strict_resident_fock_cpu_plan_setup",
            "strict_resident_fock_gpu_count_mismatch",
            "strict_resident_fock_gpu_point_weights_mismatch",
        ),
    },
    {
        "name": "strict_fillij_uses_gpu_builder",
        "path": "src/MOZYME/fillij.F90",
        "fragments": (
            "mopac_cuda_mozyme_fillij_count",
            "mopac_cuda_mozyme_fillij_nijbo",
            "mozyme_gpu_scf_no_fallback_required()",
            "fillij_gpu=1 count=",
            "strict_fillij_gpu_failed",
            "strict_fillij_gpu_unavailable",
        ),
        "forbidden_fragments": (
            "cpu_setup=mozyme_arrays setup_only=1",
        ),
    },
    {
        "name": "strict_fillij_cuda_builder",
        "path": "src/gpu/cuda_wrappers.cu",
        "fragments": (
            "mozyme_fillij_kernel",
            "mopac_cuda_mozyme_fillij_count",
            "mopac_cuda_mozyme_fillij_nijbo",
            "mozyme_fillij_gpu_run",
            "g_mz_fillij_nijbo",
            "cudaMemcpyDeviceToHost",
            "mozyme_fillij mode=%s",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_resident_fock_cuda_count_builder",
        "path": "src/gpu/cuda_wrappers.cu",
        "fragments": (
            "mozyme_resident_fock_count_kernel",
            "mopac_cuda_mozyme_resident_fock_count_plan",
            "g_mz_res_count_nijbo",
            "mozyme_resident_fock_count atoms=%d",
            "mozyme_resident_point_supported_dev",
            "kMozymeResidentFallbackBasisBins = 11",
            "mozyme_resident_fallback_basis_bin_dev",
            "if (mozyme_resident_basis_supported_dev(iab))",
            "++one_center_cpu_count;",
            "out[17] = one_center_cpu_count;",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_resident_fock_cuda_pack_point_weight_builder",
        "path": "src/gpu/cuda_wrappers.cu",
        "fragments": (
            "mozyme_resident_to_point_dev",
            "mozyme_resident_pack_point_weights_dev",
            "if (!mozyme_resident_pack_point_weights_dev(",
            "point_w + static_cast<size_t>(point_pos) * 7u",
            "status[0] = 6;",
        ),
        "forbidden_fragments": (
            "mozyme_resident_point_weights_kernel",
            "mopac_cuda_mozyme_resident_fock_point_weights",
            "g_mz_res_point_out",
        ),
    },
    {
        "name": "strict_resident_fock_cuda_pack_builder",
        "path": "src/gpu/cuda_wrappers.cu",
        "fragments": (
            "mozyme_resident_fock_pack_plan_kernel",
            "mopac_cuda_mozyme_resident_fock_pack_plan",
            "mozyme_sparse_fock_invalidate_plan",
            "mopac_cuda_mozyme_sparse_fock_plan_ready",
            "int64_t signature",
            "plan->ready && plan->has_executable_work",
            "plan->signature == signature",
            "g_mz_res_pack_status",
            "g_mz_res_pack_aq",
            "g_mz_res_pack_qq",
            "g_mz_res_pack_direct_scratch",
            "kMozymeResidentDirectPackScratchDoubles",
            "kMozymeDirectSpdScratchReppdArg",
            "kMozymeDirectSpdScratchRotP",
            "mozyme_resident_pack_point_weights_dev",
            "mozyme_direct_reppd_sp_dev",
            "mozyme_direct_sp_w_dev",
            "mozyme_resident_pair_supported_for_direct_dev",
            "(direct_flag == 0 && !wk)",
            "(direct_flag == 0 && !g_mz_res_pack_wk.ensure(w_bytes))",
            "if (direct_flag == 0) {\n    code |= copy_double(g_mz_res_pack_wk, wk, w_bytes",
            "direct_flag == 0 ? g_mz_res_pack_wk.ptr : nullptr",
            "mozyme_resident_fock_pack one=%d",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_fock_point_kind_profile",
        "path": "src/gpu/cuda_wrappers.cu",
        "fragments": (
            "int point_dipole_count = 0;",
            "plan->point_dipole_count = 0;",
            "if (addr_flag >= 0) return fail_setup(3);",
            "if (addr_flag == -2) {",
            "++plan->point_dipole_count;",
            "++plan->point_monopole_count;",
            "point=%d point_dipole=%d point_monopole=%d",
            "point_dipole=%d point_monopole=%d",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_fock_integral_signature",
        "path": "src/MOZYME/mozyme_resident_fock.F90",
        "fragments": (
            "mozyme_resident_signature(iorbs, nat, ifact, wj, wk,",
            "integer(c_int64_t) function mozyme_resident_signature",
            "mozyme_resident_integral_signature_hash(iorbs, wj, wk, use_nijbo, h1, h2)",
            "mozyme_integral_slice_signature_hash(wj, kr + 1, term_count, 41, h1, h2)",
            "mozyme_integral_slice_signature_hash(wk, kr + 1, term_count, 43, h1, h2)",
            "mozyme_mix_real64(h1, h2, values(first + idx - 1))",
            "mozyme_sig_mod1 = 2147483647_c_int64_t",
            "mozyme_point_charge_advance_kr(iab, jba, addr, kr)",
            "last_signature(plan_id) = signature",
        ),
        "forbidden_fragments": (
            "double precision function mozyme_resident_signature",
            "mozyme_integral_slice_signature(wk, kr + 1, term_count, 43.0d0)",
        ),
    },
    {
        "name": "resident_scf_cuda_bounds_and_fail_closed",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "mozyme_helecz_kernel(int numat, int mpack",
            "double *atom_diag_sums, int *ok_out",
            "base + terms > mpack",
            "mozyme_hbond_pairs_kernel(int fill, int numat, int mpack, int pair_capacity,",
            "mozyme_diagg1_aocc_kernel(int nocc, int icocc_dim, int cocc_dim, int numat",
            "mozyme_diagg1_avir_kernel(int nvir, int icvir_dim, int cvir_dim, int numat",
            "const int *task_value_offset, int value_count, int cocc_dim",
            "auto fail_stage = [&]() -> bool",
            "return fail_stage();",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_scf_planner_and_makvec_controls",
        "path": "src/MOZYME/mozyme_gpu_plan.F90",
        "fragments": (
            "if (mozyme_resident_fock_gpu) then\n        resident_scf = .true.\n        gpu_scf_stream_available = .true.\n      else\n        lgpu = .false.\n        resident_scf = .false.\n        gpu_scf_stream_available = .false.\n      end if",
            "mozyme_gpu_enabled = mozyme_resident_fock_gpu .or. mozyme_fock1_batch_gpu .or. mozyme_fock2_4x1_batch_gpu .or. &\n      (mozyme_gpu .and. lgpu)",
            ".not. (mozyme_fock_gpu .and. mozyme_f2_gpu) .and. eligible_density_pairs == 0",
            "fock_resident_full_coverage_planned = &",
            "fock_resident_supported_one_center",
            "fock_resident_unsupported_one_center",
            "fock_resident_unsupported_one_center == 0 .and. &",
            "fock_resident_executable_tasks = fock_resident_supported_one_center + fock_resident_supported_pairs + &",
            "fock_production_gpu_tasks = fock_resident_executable_tasks",
            "if (mozyme_fock_gpu .and. mozyme_f2_gpu) fock_production_gpu_tasks = fock_candidate_gpu_tasks",
            "if (.not. (mozyme_fock_gpu .and. mozyme_f2_gpu)) then",
            "mozyme_fock_gpu .and. mozyme_f2_gpu",
            "'f2_gpu=', mozyme_f2_gpu",
            "fock_resident_supported_point_pairs",
            "fock_resident_unsupported_point_pairs",
            "fock_resident_direct_basis_pairs",
            "fock_resident_direct_basis_point_pairs",
            "fock_resident_noop_pairs",
            "fock_resident_full_coverage_planned",
            "fock_resident_executable_tasks",
            "mozyme_plan_resident_basis_supported",
            "mozyme_plan_resident_basis_supported(iab) .and. &",
            "nbasis == 1 .or. nbasis == 4 .or. nbasis == 9",
            "mozyme_plan_resident_direct_basis_supported",
            "mozyme_plan_resident_direct_basis_supported(iab) .and. &",
            "mozyme_plan_resident_direct_basis_supported = nbasis == 1 .or. nbasis == 4 .or. nbasis == 9",
            "mozyme_plan_resident_direct_basis_fallback",
            "mozyme_plan_resident_point_supported",
            "if (direct) then",
            "[MOZYME GPU plan] direct=",
            "fock_resident_direct_unsupported_pairs",
            "fock_resident_direct_unsupported_point_pairs",
            "fock_point_dipole_pairs",
            "fock_point_monopole_pairs",
            "resident_one_center supported=",
            "' point_dipole_pairs='",
            "' point_monopole_pairs='",
            "call mozyme_gpu_strict_abort('strict_no_gpu_work', &",
            "Strict MOZYME full-SCF GPU requested, but the MOZYME GPU plan has no GPU work",
            "production two-center GPU Fock uses resident sparse Fock when resident_fock_gpu=T",
        ),
        "forbidden_fragments": (
            "if (mozyme_resident_fock_gpu) then\n        lgpu = .false.",
            "production two-center GPU Fock is limited to batched 4x1",
            "iab > 0 .and. jba > 0 .and. iab <= 9 .and. jba <= 9",
            "addr < 0 .and. iab > 0 .and. jba > 0 .and. &\n      iab <= 9 .and. jba <= 9",
        ),
    },
    {
        "name": "strict_resident_fock_abort_marker",
        "path": "src/MOZYME/mozyme_resident_fock.F90",
        "fragments": (
            "subroutine strict_resident_fock_abort(reason, message)",
            "[MOZYME GPU SCF] status=strict_abort reason=",
            "strict_resident_fock_disabled",
            "strict_resident_fock_legacy_host_copy",
            "strict_resident_fock_work_alloc_failed",
            "error stop 'MOZYME GPU strict resident Fock abort'",
            "allocate(f_work(mpack), stat=alloc_stat)",
            "f_work(1:mpack) = f(1:mpack)",
            "f(1:mpack) = f_work(1:mpack)",
        ),
        "forbidden_fragments": (
            "[MOZYME GPU SCF] status=fallback_cpu reason=",
        ),
    },
    {
        "name": "strict_legacy_fock2_abort_marker",
        "path": "src/MOZYME/fock2z.F90",
        "fragments": (
            "resident_strict_requested, &\n     strict_resident_fock_abort",
            "strict_legacy_fock2_failed",
            "MOZYME GPU strict legacy Fock2 failed",
        ),
        "forbidden_fragments": (),
    },
    {
            "name": "full_resident_request_blocks_run_mopac_cpu_disables",
            "path": "src/run_mopac.F90",
            "fragments": (
                "MOPAC_MOZYME_SCF_STRICT_RESIDENT",
                "MOPAC_MOZYME_SCF_GPU",
                "mozyme_gpu_scf_no_fallback_required",
                "mozyme_gpu_scf_reset_request_state",
            "strict_gpu_disabled_by_nogpu_keyword",
            "strict_no_gpu_device",
            "strict_gpu_disabled_by_mopac_nogpu",
            "strict_gpu_disabled_by_scftask_cpu",
            "strict_gpu_disabled_by_mozyme_gpu_off",
            "case ('0','FALSE','F','NO','N','OFF')",
            "strict_full_gpu_required = mozyme .and. strict_full_gpu_required",
            "strict_full_gpu_required = strict_mozyme_scf",
            "Preserve ordinary MOZYME behavior unless GPU execution was requested",
            "call mozyme_gpu_scf_reset_request_state()",
            'write(*,*) "This MOPAC executable was not compiled with MDI support"\n          return',
            "call geout (iarc)\n        goto 100",
            "read(list,*,iostat=stat_env)",
            "read(keyup(pos:),*,iostat=stat_env)",
            "read(pair,*,iostat=stat_env)",
            "strict_mozyme_scf = mozyme .and. mozyme_gpu_scf_no_fallback_required()",
            "strict_not_gpu_build",
            "strict_resident_fock_gpu_disabled",
            "if (strict_mozyme_scf) then\n        l_OLDDEN = .true.",
            "strict_run_mopac_olden_host_restore",
            "MOZYME GPU strict resident SCF does not support OLDEN/OLDENS host restore",
            "goto 101",
            "if (.not. strict_mozyme_scf .and. .not. l_OLDDEN .and.",
        ),
        "forbidden_fragments": (
            "If MOZYME is active and GPU is enabled, default to MOZYME GPU unless explicitly disabled",
            "call get_environment_variable('MOPAC_MOZYME_RESIDENT_SCF', line, status=i)\n        if (i == 0) then\n          line = adjustl(line)\n          if (len_trim(line) /= 0) then\n            call upcase(line, len_trim(line))\n            select case (trim(line))\n            case ('0','FALSE','F','NO','N','OFF')\n            case default\n              strict_full_gpu_required = .true.",
        ),
    },
    {
        "name": "full_resident_request_blocks_fock2z_cpu_fallback",
        "path": "src/MOZYME/fock2z.F90",
        "fragments": (
            "resident_strict_requested, &\n      strict_resident_fock_abort",
            "strict_resident_fock_cpu_fallback",
            "MOZYME GPU strict resident Fock did not complete before CPU Fock",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "full_resident_request_blocks_fock1_cpu_fallback",
        "path": "src/MOZYME/fock1_for_MOZYME.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "mozyme_gpu_scf_no_fallback_required()",
            "strict_fock1_cpu_fallback",
            "MOZYME GPU strict resident SCF does not support CPU one-center Fock construction",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_resident_blocks_cpu_pinout",
        "path": "src/MOZYME/pinout.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "use mozyme_section_timers, only: mozyme_section_timers_enabled",
            "[MOZYME CPU pinout]",
            "mozyme_gpu_scf_no_fallback_required()",
            "strict_pinout_cpu_fallback",
            "MOZYME GPU strict resident SCF does not support CPU pinout",
            "call mozyme_gpu_strict_abort",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_resident_blocks_cpu_interaction_promotion",
        "path": "src/MOZYME/add_more_interactions.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "external :: mozyme_gpu_strict_abort",
            "strict_add_more_interactions_cpu_fallback",
            "MOZYME GPU strict resident SCF does not support CPU interaction promotion",
            "call fillij (.false.)",
        ),
        "forbidden_fragments": (
            "call fillij (.false.)\n  if (mozyme_gpu_scf_no_fallback_required())",
        ),
    },
    {
        "name": "strict_resident_blocks_cpu_writmo_outputs",
        "path": "src/output/writmo.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only : mozyme_gpu_scf_no_fallback_required",
            "strict_mozyme_gpu_scf = mozyme .and. mozyme_gpu_scf_no_fallback_required()",
            "subroutine mozyme_gpu_strict_writmo_abort(reason)",
            "call mopend(reason)",
            "error stop 'MOZYME GPU strict writmo abort'",
            "writmo_relocal_output=skipped resident_gpu=1",
            "writmo_pops_output=skipped resident_gpu=1",
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_vec_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_mecip_host_density")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_pm7ts_host_compfg")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_deriv_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_fock_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_dens_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_pi_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_spin_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_bonds_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_local_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_1ele_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_enpart_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_denout_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_mullik_host_output")',
        ),
        "forbidden_fragments": (
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_reloc_host_output")',
            'call mozyme_gpu_strict_writmo_abort("strict_writmo_pops_host_output")',
            'call mopend ("strict_writmo_reloc_host_output")',
            'call mopend ("strict_writmo_vec_host_output")',
            'call mopend ("strict_writmo_mecip_host_density")',
            'call mopend ("strict_writmo_pm7ts_host_compfg")',
            'call mopend ("strict_writmo_deriv_host_output")',
            'call mopend ("strict_writmo_fock_host_output")',
            'call mopend ("strict_writmo_dens_host_output")',
            'call mopend ("strict_writmo_pops_host_output")',
            'call mopend ("strict_writmo_pi_host_output")',
            'call mopend ("strict_writmo_spin_host_output")',
            'call mopend ("strict_writmo_bonds_host_output")',
            'call mopend ("strict_writmo_local_host_output")',
            'call mopend ("strict_writmo_1ele_host_output")',
            'call mopend ("strict_writmo_enpart_host_output")',
            'call mopend ("strict_writmo_denout_host_output")',
            'call mopend ("strict_writmo_mullik_host_output")',
        ),
    },
    {
        "name": "resident_scf_makvec_periodic_ione",
        "path": "src/MOZYME/mozyme_gpu_makvec.F90",
        "fragments": (
            "resident_plan_ione = 0",
            "if (id == 0) resident_plan_ione = 1",
            "kopt, resident_plan_ione, coord",
            "env_is_one('MOPAC_MOZYME_SCF_GPU')",
            "env_is_one('MOPAC_MOZYME_FULL_SCF_GPU')",
            "env_is_one('MOPAC_MOZYME_GPU_STRICT')",
            "env_is_one('MOPAC_MOZYME_RESIDENT_SCF')",
        ),
        "forbidden_fragments": (
            "kopt, 1, coord",
        ),
    },
    {
        "name": "strict_resident_implies_request",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "mozyme_gpu_scf_strict_resident = &\n      env_is_one('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &\n      env_is_one('MOPAC_MOZYME_SCF_GPU')",
            "strict_requested = env_is_one('MOPAC_MOZYME_SCF_STRICT_RESIDENT')",
            "env_is_one('MOPAC_MOZYME_GPU_STRICT')",
            "full_scf_requested = env_is_one('MOPAC_MOZYME_SCF_GPU')",
            "env_is_one('MOPAC_MOZYME_FULL_SCF_GPU')",
            "resident_requested = env_is_one('MOPAC_MOZYME_RESIDENT_SCF')",
            "request_present = resident_requested .or. full_scf_requested .or.",
            "no_fallback_required = strict_requested .or. full_scf_requested",
            ".not. (strict_requested .or. full_scf_requested) .and.",
            "env_is_one = env_enabled(var_name)",
            "scf_failure_status = 'status=strict_abort'",
            "scf_failure_message = 'status=strict_abort '//trim(detail)",
        ),
        "forbidden_fragments": (
            "mozyme_gpu_scf_strict_resident = &\n      env_is_one('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &\n      env_is_one('MOPAC_MOZYME_RESIDENT_SCF')",
            "mozyme_gpu_scf_strict_resident = &\n      env_is_one('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &\n      env_is_one('MOPAC_MOZYME_SCF_GPU') .or. &\n      env_is_one('MOPAC_MOZYME_RESIDENT_SCF')",
        ),
    },
    {
        "name": "strict_lewis_uses_gpu_makvec",
        "path": "src/MOZYME/iter_for_MOZYME.F90",
        "fragments": (
            'if (Index (keywrd, " LEWIS") /= 0) then',
            "if (resident_strict_required) then",
            "makvec_gpu_done = mozyme_gpu_makvec_try()",
            "strict_lewis_gpu_makvec_failed",
            "could not complete LEWIS setup on GPU",
        ),
        "forbidden_fragments": (
            "strict_lewis_setup",
            "does not support LEWIS setup",
        ),
    },
    {
        "name": "strict_direct_cpu_makvec_guard",
        "path": "src/MOZYME/makvec.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "mozyme_gpu_scf_no_fallback_required()",
            "strict_cpu_makvec_direct",
            "does not support direct CPU makvec",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_density_rebuild_fail_closed",
        "path": "src/MOZYME/density_for_MOZYME.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "strict_resident_density = mozyme_gpu_scf_no_fallback_required()",
            "strict_resident_density",
            "strict_density_direct_host_rebuild",
            "strict_density_batch_fallback",
            "does not support direct host density rebuild",
            "does not support CPU density rebuild",
            "if (strict_resident_density .and. .not. density_batch_gpu_done) then",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_old_scf_rejected_in_proof",
        "path": "src/MOZYME/iter_for_MOZYME.F90",
        "fragments": (
            'if (index(keywrd, "OLD_SCF") /= 0) then',
            "if (resident_strict_required) then",
            "strict_old_scf_existing_lmo",
            "does not accept OLD_SCF host-existing LMOs as makvec proof",
        ),
        "forbidden_fragments": (
            "status=existing_lmo reason=old_scf",
        ),
    },
    {
        "name": "strict_relocal_uses_gpu",
        "path": "src/MOZYME/iter_for_MOZYME.F90",
        "fragments": (
            "use mozyme_gpu_relocalize, only : mozyme_gpu_relocalize_try",
            'mozyme_gpu_relocalize_try("OCCUPIED")',
            'mozyme_gpu_relocalize_try("VIRTUAL")',
            "could not complete occupied re-localization on GPU",
            "could not complete virtual re-localization on GPU",
        ),
        "forbidden_fragments": (
            "does not support CPU re-localization",
        ),
    },
    {
        "name": "mozyme_gpu_relocal_fortran_wrapper",
        "path": "src/MOZYME/mozyme_gpu_relocalize.F90",
        "fragments": (
            "module mozyme_gpu_relocalize",
            "bind(C,name='mopac_cuda_mozyme_relocalize')",
            "env_is_one('MOPAC_MOZYME_SCF_GPU')",
            "env_is_one('MOPAC_MOZYME_FULL_SCF_GPU')",
            "env_is_one('MOPAC_MOZYME_GPU_STRICT')",
            "env_is_one('MOPAC_MOZYME_RESIDENT_SCF')",
            "MOPAC_MOZYME_RELOCAL_GPU",
            "kind_mismatch",
            "[MOZYME GPU relocal]",
            "kind=",
            "if (code == 0_c_int) ijc = 0",
        ),
        "forbidden_fragments": (
            "type=",
        ),
    },
    {
        "name": "mozyme_gpu_relocal_cuda_helper",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "mozyme_relocalize_pass_device",
            "mozyme_relocalize_eigs_device",
            "mozyme_relocalize_kernel",
            "for (int iter = 1; iter <= 100; ++iter)",
            "extern \"C\" int mopac_cuda_mozyme_relocalize",
            "std::vector<double> host_c(c_sz);",
            "std::copy(host_c.begin(), host_c.end(), c);",
            "relocal c copy",
            "relocal eigs copy",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_final_reorth_uses_resident_gpu",
        "path": "src/MOZYME/iter_for_MOZYME.F90",
        "fragments": (
            "resident_final_reorth_due",
            "resident_final_reorth_done",
            "mozyme_gpu_scf_force_final_reorth",
            "final_reorth=resident_strict_required",
            "final_reorth_done=resident_final_reorth_done",
            "strict_cpu_reorthogonalization",
            "did not complete resident reorthogonalization",
        ),
        "forbidden_fragments": (
            "does not support CPU reorthogonalization",
        ),
    },
    {
        "name": "strict_final_reorth_success_marker_requires_scf_success",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "index(prefix, 'status=success') > 0",
            "[MOZYME GPU reorth]",
            "status=success resident=1",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_or_full_request_is_no_cpu_fallback",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
            "fragments": (
            "public :: mozyme_gpu_scf_no_fallback_required",
            "public :: mozyme_gpu_scf_reset_request_state",
            "mozyme_gpu_scf_no_fallback_required = no_fallback_required",
            "blocked_nscf = -1",
            "env_is_one('MOPAC_MOZYME_RESIDENT_SCF')",
            "env_is_one('MOPAC_MOZYME_SCF_GPU')",
            "env_is_one('MOPAC_MOZYME_FULL_SCF_GPU')",
            "env_is_one('MOPAC_MOZYME_GPU_STRICT')",
            "mozyme_gpu_scf_no_fallback_required()) then",
            "scf_failure_status = 'status=strict_abort'",
            "scf_failure_message = 'status=strict_abort '//trim(detail)",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_eimp_cpu_fallback_guard",
        "path": "src/MOZYME/eimp.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "if (mozyme_gpu_scf_no_fallback_required()) then",
            "[MOZYME GPU SCF] status=strict_abort reason=strict_eimp_cpu_fallback",
            "error stop 'MOZYME GPU strict eimp abort'",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_cnvgz_cpu_fallback_guard",
        "path": "src/MOZYME/cnvgz.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "if (mozyme_gpu_scf_no_fallback_required()) then",
            "[MOZYME GPU SCF] status=strict_abort reason=strict_cnvgz_cpu_fallback",
            "error stop 'MOZYME GPU strict cnvgz abort'",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_helecz_cpu_fallback_guard",
        "path": "src/MOZYME/helecz.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "if (mozyme_gpu_scf_no_fallback_required()) then",
            "[MOZYME GPU SCF] status=strict_abort reason=strict_helecz_cpu_fallback",
            "error stop 'MOZYME GPU strict helecz abort'",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_isitsc_cpu_fallback_guard",
        "path": "src/MOZYME/isitsc.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "if (mozyme_gpu_scf_no_fallback_required()) then",
            "[MOZYME GPU SCF] status=strict_abort reason=strict_isitsc_cpu_fallback",
            "error stop 'MOZYME GPU strict isitsc abort'",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_addhb_cpu_fallback_guard",
        "path": "src/MOZYME/addhb.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "mozyme_gpu_scf_no_fallback_required()",
            "strict_addhb_cpu_fallback",
            "MOZYME GPU strict resident SCF does not support CPU ADDHB",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_check_cpu_fallback_guard",
        "path": "src/MOZYME/check.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "mozyme_gpu_scf_no_fallback_required()",
            "strict_check_cpu_fallback",
            "strict_check_gpu_host_fallback",
            "MOZYME GPU strict resident SCF does not support CPU LMO normalization check",
            "MOZYME GPU strict resident SCF does not support host LMO normalization check",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_diagg_cpu_fallback_guard",
        "path": "src/MOZYME/diagg.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "mozyme_gpu_scf_no_fallback_required()",
            "strict_diagg_cpu_fallback",
            "MOZYME GPU strict resident SCF does not support CPU DIAGG",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_reorth_cpu_fallback_guard",
        "path": "src/MOZYME/reorth.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "mozyme_gpu_scf_no_fallback_required()",
            "strict_reorth_cpu_fallback",
            "MOZYME GPU strict resident SCF does not support CPU reorthogonalization",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_tidy_cpu_fallback_guard",
        "path": "src/MOZYME/tidy.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "mozyme_gpu_scf_no_fallback_required()",
            "strict_tidy_cpu_fallback",
            "MOZYME GPU strict resident SCF does not support CPU LMO tidy",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_diagg1_cpu_fallback_guard",
        "path": "src/MOZYME/diagg1.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "if (mozyme_gpu_scf_no_fallback_required()) then",
            "[MOZYME GPU SCF] status=strict_abort reason=strict_diagg1_cpu_fallback",
            "error stop 'MOZYME GPU strict diagg1 abort'",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_diagg2_cpu_fallback_guard",
        "path": "src/MOZYME/diagg2.F90",
        "fragments": (
            "use mozyme_gpu_scf_driver, only: mozyme_gpu_scf_no_fallback_required",
            "if (mozyme_gpu_scf_no_fallback_required()) then",
            "[MOZYME GPU SCF] status=strict_abort reason=strict_diagg2_cpu_fallback",
            "error stop 'MOZYME GPU strict diagg2 abort'",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "full_resident_request_blocks_fortran_cpu_fallbacks",
        "path": "src/MOZYME/iter_for_MOZYME.F90",
        "fragments": (
            "mozyme_gpu_scf_no_fallback_required",
            "resident_strict_required = mozyme_gpu_scf_no_fallback_required()",
            "block_on_failure=resident_strict_required",
            "strict_cpu_iteration_work",
            "strict_cpu_scf_body",
            "strict_missing_final_density",
        ),
        "forbidden_fragments": (
            "resident_strict_required = mozyme_gpu_scf_strict_resident()",
        ),
    },
    {
        "name": "full_resident_request_blocks_density_and_index_cpu_fallbacks",
        "path": "src/MOZYME/density_for_MOZYME.F90",
        "fragments": (
            "mozyme_gpu_scf_no_fallback_required",
            "strict_resident_density = mozyme_gpu_scf_no_fallback_required()",
            "strict_density_batch_fallback",
        ),
        "forbidden_fragments": (
            "get_environment_variable('MOPAC_MOZYME_SCF_STRICT_RESIDENT'",
        ),
    },
    {
        "name": "full_resident_request_requires_nijbo_and_plan_work",
        "path": "src/MOZYME/fillij.F90",
        "fragments": (
            "mozyme_gpu_scf_no_fallback_required",
            "strict_nijbo_alloc_failed",
            "compact_index_route=0 use_nijbo=",
        ),
        "forbidden_fragments": (
            "mozyme_gpu_scf_strict_resident()",
        ),
    },
    {
        "name": "full_resident_request_blocks_plan_and_fock_fallbacks",
        "path": "src/MOZYME/mozyme_gpu_plan.F90",
        "fragments": (
            "mozyme_plan_env_enabled('MOPAC_MOZYME_SCF_GPU')",
            "mozyme_plan_env_enabled('MOPAC_MOZYME_FULL_SCF_GPU')",
            "mozyme_plan_env_enabled('MOPAC_MOZYME_GPU_STRICT')",
            "strict_required = mozyme_plan_env_enabled('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &",
            "case ('0', 'f', 'F', 'false', 'FALSE', 'False', &",
            "'n', 'N', 'no', 'NO', 'No', 'off', 'OFF', 'Off')",
            "strict_no_gpu_work",
            "preflight_stop = strict_required",
            "error stop 'MOZYME_GPU preflight found no GPU production work'",
        ),
        "forbidden_fragments": (
            "strict_required = mozyme_plan_env_enabled('MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &\n      mozyme_plan_env_enabled('MOPAC_MOZYME_RESIDENT_SCF')",
        ),
    },
    {
        "name": "resident_fock_no_fallback_request_envs",
        "path": "src/MOZYME/mozyme_resident_fock.F90",
        "fragments": (
            "resident_strict_requested = resident_env_requested( &",
            "'MOPAC_MOZYME_SCF_STRICT_RESIDENT') .or. &",
            "resident_env_requested('MOPAC_MOZYME_SCF_GPU')",
            "resident_env_requested('MOPAC_MOZYME_FULL_SCF_GPU')",
            "resident_env_requested('MOPAC_MOZYME_GPU_STRICT')",
            "strict_resident_fock_run_failed",
        ),
        "forbidden_fragments": (
            "resident_env_requested('MOPAC_MOZYME_RESIDENT_SCF') .or. &",
        ),
    },
    {
        "name": "mozyme_gpu_reorth_fortran_wrapper",
        "path": "src/MOZYME/mozyme_gpu_reorth.F90",
        "fragments": (
            "module mozyme_gpu_reorth",
            "bind(C,name='mopac_cuda_mozyme_reorth')",
            "env_is_one('MOPAC_MOZYME_SCF_GPU')",
            "env_is_one('MOPAC_MOZYME_FULL_SCF_GPU')",
            "env_is_one('MOPAC_MOZYME_GPU_STRICT')",
            "env_is_one('MOPAC_MOZYME_RESIDENT_SCF')",
            "MOPAC_MOZYME_REORTH_GPU",
            "kind_mismatch",
            "[MOZYME GPU reorth]",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "mozyme_gpu_reorth_cuda_helper",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "kMozymeScfFlagFinalReorth",
            "apply_final_reorth_on_gpu",
            "resident final reorth kernel",
            "status->final_reorth_applied = 1",
            "status->final_reorth_ms",
            "status->final_reorth_sum",
            "DeviceBuffer<double> final_reorth_ws;",
            "DeviceBuffer<double> final_reorth_sumtot;",
            "DeviceBuffer<int> final_reorth_status;",
            "dev.final_reorth_ws.resize(norbs_count)",
            "dev.final_reorth_status.resize(1)",
            "dev.final_reorth_ws.ptr",
            "dev.final_reorth_status.ptr",
            "resident final reorth status reset",
            "resident final reorth sum reset",
            "mozyme_reorth_adjvec_device",
            "mozyme_reorth_kernel",
            "extern \"C\" int mopac_cuda_mozyme_reorth",
        ),
        "forbidden_fragments": (
            "DeviceBuffer<double> ws;\n  DeviceBuffer<double> sumtot;\n  DeviceBuffer<int> latom;",
            "if (!reorth_status.upload(&zero_i, 1))",
            "if (!sumtot.upload(&zero_d, 1))",
        ),
    },
    {
        "name": "resident_final_reorth_validates_recomputed_gpu_stages",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "validate_final_reorth_rebuild_from_gpu",
            "resident final reorth density status copy",
            "resident final reorth fock status copy",
            "resident final reorth helecz status copy",
            "density_status[1] == 1",
            "density_status[0] == density_status[2]",
            "fock_status[kFockIntOk] == 1",
            "helecz_status[kHeleczIntOk] == 1",
            "publish_cosmo_status_from_gpu(ctx, status)",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "mozyme_gpu_tidy_wrapper_and_cuda_helper",
        "path": "src/MOZYME/mozyme_gpu_tidy.F90",
        "fragments": (
            "module mozyme_gpu_tidy",
            "public :: mozyme_gpu_tidy_try",
            "bind(C,name='mopac_cuda_mozyme_tidy')",
            "MOPAC_MOZYME_TIDY_GPU",
            "MOPAC_MOZYME_FULL_SCF_GPU",
            "MOPAC_MOZYME_GPU_STRICT",
            "[MOZYME GPU tidy]",
            "status=success mode=",
            "selmos=",
            "selected=",
            "use_selmos_c",
            "type(c_ptr), value :: jopt_c",
            "jopt_ptr = c_null_ptr",
            "jopt_ptr = c_loc(jopt(1))",
            "numred_out_of_range",
            "error_code",
            "status=fallback_cpu mode=",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "mozyme_gpu_tidy_cuda_helper",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "mozyme_tidy_kernel",
            "mopac_cuda_mozyme_tidy",
            "mozyme_tidy_selmos_device",
            "mozyme_tidy_compct_device",
            "mozyme_tidy_space_device",
            "if (nbot + nc[i] > ntop || jbot + iws[i] > jtop)",
            "nc[lmo] <= 0",
            "use_selmos != 0 && numred > natoms",
            "*status = -520",
            "*status = -521",
            "int code = -530",
            "jopt",
            "selected_out",
            "tidy kernel",
            "tidy result copy",
            "tidy nc copy",
            "tidy nnc copy",
            "tidy ncmo copy",
            "tidy elapsed time",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "mozyme_gpu_tidy_selmos_strict_path",
        "path": "src/MOZYME/iter_for_MOZYME.F90",
        "fragments": (
            "resident_pre_tidy_attempt_needed = mozyme_gpu_scf_requested()",
            "resident_tidy_imode(2) = 0",
            "step_num > 1+step_num0",
            "step_num /= resident_tidy_imode(1)",
            "step_num /= resident_tidy_imode(2)",
            "mozyme_gpu_tidy_try(1, lno, mn, &",
            "mozyme_gpu_tidy_try(2, lnv, mn, &",
            "use_selmos=resident_tidy_select_lmos",
            "error_code=resident_tidy_code",
            "resident_tidy_code == -506",
            "mozyme_gpu_grow_lmo_storage",
            "resident_tidy_done = .true.",
            "initial_tidy_done=resident_tidy_done",
        ),
        "forbidden_fragments": (
            "strict_resident_tidy_selmos_missing",
            "MOZYME GPU strict resident SCF does not yet support TIDY SELMOS on GPU",
        ),
    },
    {
        "name": "resident_scf_abi_v33_final_publication_reorth_pls_cosmo_direct",
        "path": "src/gpu/gpu_mozyme_scf_interfaces.F90",
        "fragments": (
            "GPU_MOZYME_SCF_ABI_VERSION = 33_c_int",
            "GPU_MOZYME_SCF_RESIDENT_DECISION_COMPLETE = 1_c_int",
            "integer(c_int) :: resident_decision = 0_c_int",
            "integer(c_int) :: resident_fock_plan_id = 0_c_int",
            "integer(c_int) :: resident_fock_plan_full_coverage = 0_c_int",
            "integer(c_int) :: resident_fock_plan_partial_coverage = 0_c_int",
            "integer(c_int) :: resident_fock_plan_required_mask = 0_c_int",
            "integer(c_int) :: resident_fock_plan_covered_mask = 0_c_int",
            "integer(c_int) :: final_publication_done = 0_c_int",
            "integer(c_int) :: final_publication_arrays = 0_c_int",
            "integer(c_size_t) :: final_publication_bytes = 0_c_size_t",
            "integer(c_int) :: final_publication_cosmo = 0_c_int",
            "integer(c_int) :: cnvgz_active_calls = 0_c_int",
            "integer(c_int) :: cnvgz_noop_calls = 0_c_int",
            "integer(c_int) :: partp_dim = 0_c_int",
            "integer(c_int) :: partf_dim = 0_c_int",
            "integer(c_int) :: nocc_slots = 0_c_int",
            "integer(c_int) :: nvir_slots = 0_c_int",
            "integer(c_int) :: p_dim = 0_c_int",
            "integer(c_int) :: f_dim = 0_c_int",
            "integer(c_int) :: h_dim = 0_c_int",
            "integer(c_int) :: pold_dim = 0_c_int",
            "integer(c_int) :: p1_dim = 0_c_int",
            "integer(c_int) :: p2_dim = 0_c_int",
            "integer(c_int) :: p3_dim = 0_c_int",
            "integer(c_int) :: idiag_dim = 0_c_int",
            "integer(c_int) :: iorbs_dim = 0_c_int",
            "integer(c_int) :: kopt_dim = 0_c_int",
            "integer(c_int) :: ncf_dim = 0_c_int",
            "integer(c_int) :: nncf_dim = 0_c_int",
            "integer(c_int) :: ncocc_dim = 0_c_int",
            "integer(c_int) :: nce_dim = 0_c_int",
            "integer(c_int) :: nnce_dim = 0_c_int",
            "integer(c_int) :: ncvir_dim = 0_c_int",
            "integer(c_int) :: ifmo_rows = 0_c_int",
            "integer(c_int) :: ifmo_cols = 0_c_int",
            "integer(c_int) :: eigs_dim = 0_c_int",
            "integer(c_int) :: nfmo_dim = 0_c_int",
            "integer(c_int) :: nfirst_dim = 0_c_int",
            "integer(c_int) :: nlast_dim = 0_c_int",
            "integer(c_int) :: nijbo_rows = 0_c_int",
            "integer(c_int) :: nijbo_cols = 0_c_int",
            "integer(c_int) :: resident_stage_calls(10) = 0_c_int",
            "real(c_double) :: resident_stage_ms(10) = 0.0_c_double",
            "integer(c_int) :: final_reorth_applied = 0_c_int",
            "real(c_double) :: final_reorth_ms = 0.0_c_double",
            "real(c_double) :: final_reorth_sum = 0.0_c_double",
            "integer(c_int) :: pls_supervisor_calls = 0_c_int",
            "integer(c_int) :: pls_restart_required = 0_c_int",
            "integer(c_int) :: pls_history_count = 0_c_int",
            "real(c_double) :: pls_ovmax_delta = 0.0_c_double",
            "real(c_double) :: pls_energy_delta = 0.0_c_double",
            "integer(c_int) :: pls_restart_reset_device_calls = 0_c_int",
            "integer(c_int) :: pls_restart_done = 0_c_int",
            "integer(c_int) :: cosmo_fock_calls = 0_c_int",
            "integer(c_int) :: cosmo_matvec_calls = 0_c_int",
            "integer(c_int) :: cosmo_cg_iterations = 0_c_int",
            "integer(c_int) :: cosmo_pair_count = 0_c_int",
            "real(c_double) :: cosmo_last_residual = 0.0_c_double",
            "integer(c_int) :: cosmo_cg_control_resident = 0_c_int",
            "integer(c_int) :: cosmo_cg_converged = 0_c_int",
            "integer(c_int) :: cosmo_cg_breakdown = 0_c_int",
            "integer(c_int) :: cosmo_cg_host_syncs = 0_c_int",
            "real(c_double) :: cosmo_cg_target_tol = 0.0_c_double",
            "integer(c_int) :: strict_resident_host_syncs = 0_c_int",
            "integer(c_int) :: strict_resident_control_polls = 0_c_int",
            "integer(c_int) :: coord_rows = 0_c_int",
            "integer(c_int) :: coord_cols = 0_c_int",
            "integer(c_int) :: nat_dim = 0_c_int",
            "integer(c_int) :: cosmo_enabled = 0_c_int",
            "integer(c_int) :: cosmo_nps = 0_c_int",
            "integer(c_int) :: cosmo_lm61 = 0_c_int",
            "integer(c_int) :: cosmo_cosurf_rows = 0_c_int",
            "integer(c_int) :: cosmo_cosurf_cols = 0_c_int",
            "integer(c_int) :: cosmo_phinet_rows = 0_c_int",
            "integer(c_int) :: cosmo_phinet_cols = 0_c_int",
            "integer(c_int) :: cosmo_qscnet_rows = 0_c_int",
            "integer(c_int) :: cosmo_qscnet_cols = 0_c_int",
            "integer(c_int) :: cosmo_qdenet_rows = 0_c_int",
            "integer(c_int) :: cosmo_qdenet_cols = 0_c_int",
            "integer(c_int) :: cosmo_qscat_dim = 0_c_int",
            "integer(c_int) :: cosmo_srad_dim = 0_c_int",
            "integer(c_int) :: cosmo_npoints_dim = 0_c_int",
            "integer(c_int) :: cosmo_a_diag_dim = 0_c_int",
            "integer(c_int) :: cosmo_a_part_dim = 0_c_int",
            "integer(c_int) :: cosmo_m_vec_dim = 0_c_int",
            "integer(c_int) :: cosmo_iblock_pos_dim = 0_c_int",
            "integer(c_int) :: cosmo_new_surface = 0_c_int",
            "integer(c_int) :: param_dim = 0_c_int",
            "real(c_double) :: cosmo_fepsi = 0.0_c_double",
            "real(c_double) :: cosmo_disex2 = 0.0_c_double",
            "real(c_double) :: cosmo_solv_energy = 0.0_c_double",
            "real(c_double) :: cosmo_ediel = 0.0_c_double",
            "real(c_double) :: cosmo_a0 = 0.0_c_double",
            "real(c_double) :: cosmo_ev = 0.0_c_double",
            "type(c_ptr) :: coord = c_null_ptr",
            "type(c_ptr) :: nat = c_null_ptr",
            "type(c_ptr) :: param_dd = c_null_ptr",
            "type(c_ptr) :: param_qq = c_null_ptr",
            "type(c_ptr) :: param_tore = c_null_ptr",
            "type(c_ptr) :: cosmo_iatsp = c_null_ptr",
            "type(c_ptr) :: cosmo_ipiden = c_null_ptr",
            "type(c_ptr) :: cosmo_gden = c_null_ptr",
            "type(c_ptr) :: cosmo_qscat = c_null_ptr",
            "type(c_ptr) :: cosmo_srad = c_null_ptr",
            "type(c_ptr) :: cosmo_cosurf = c_null_ptr",
            "type(c_ptr) :: cosmo_phinet = c_null_ptr",
            "type(c_ptr) :: cosmo_qscnet = c_null_ptr",
            "type(c_ptr) :: cosmo_qdenet = c_null_ptr",
            "type(c_ptr) :: cosmo_npoints = c_null_ptr",
            "type(c_ptr) :: cosmo_a_diag = c_null_ptr",
            "type(c_ptr) :: cosmo_a_part = c_null_ptr",
            "type(c_ptr) :: cosmo_a_part_i = c_null_ptr",
            "type(c_ptr) :: cosmo_a_part_j = c_null_ptr",
            "type(c_ptr) :: cosmo_m_vec = c_null_ptr",
            "type(c_ptr) :: cosmo_iblock_pos = c_null_ptr",
            "type(c_ptr) :: cosmo_solv_energy_ptr = c_null_ptr",
            "type(c_ptr) :: cosmo_ediel_ptr = c_null_ptr",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_scf_fortran_cosmo_direct_state_v32",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "use linear_cosmo, only: mozyme_cosmo_gpu_state",
            "mozyme_cosmo_prepare_gpu_state",
            "if (lpka) then",
            "scf_failure_message('reason=solvent_fock')",
            "if (useps) then",
            "call mozyme_cosmo_prepare_gpu_state(cosmo_prepare_ok)",
            "cosmo_state_supported = .false.",
            "scf_failure_message('reason=solvent_fock detail=cosmo_prepare')",
            "call mozyme_cosmo_gpu_state(cosmo_npoints_dim",
            "state%cosmo_enabled = 1_c_int",
            "c_int_or_zero(size1_or_zero_real_2d(cosurf))",
            "c_int_or_zero(size1_or_zero_real_2d(phinet))",
            "c_int_or_zero(size1_or_zero_real_2d(qscnet))",
            "c_int_or_zero(size1_or_zero_real_2d(qdenet))",
            "state%cosmo_a_part_i = cosmo_a_part_i_ptr",
            "state%cosmo_solv_energy_ptr = c_loc(solv_energy)",
            "state%cosmo_ediel_ptr = c_loc(ediel)",
            "state_layout_has_required_sizes(config, state, missing_field)",
            "state%cosmo_enabled == 1_c_int",
            "missing_field = 'cosmo_npoints_dim'",
            "cosmo_fock_calls=",
            "cosmo_matvec_calls=",
            "cosmo_last_residual=",
            "cosmo_cg_control_resident=",
            "cosmo_cg_converged=",
            "cosmo_cg_breakdown=",
            "cosmo_cg_host_syncs=",
            "cosmo_cg_target_tol=",
        ),
        "forbidden_fragments": (
            "if (useps .or. lpka) then",
        ),
    },
    {
        "name": "linear_cosmo_gpu_direct_state_v32",
        "path": "src/solvation/linear_cosmo.F90",
        "fragments": (
            "mozyme_cosmo_prepare_gpu_state",
            "mozyme_cosmo_gpu_state",
            "mozyme_cosmo_allocate_a_part_pairs",
            "mozyme_cosmo_build_a_part_pairs",
            "count_short_ints(cosurf, 4, simulate_aq_dir_int, .true.)",
            "call precondition(cosurf, nps, iatsp, numat, a_diag, m_vec,",
            "if (.not. precondition_ok) return",
            "if (.not. allocated(a_part) .or. npos > size(a_part)) return",
            "a_part_i(npos) = ii",
            "a_part_j(npos) = jj",
            "a_part_i",
            "a_part_j",
            "npoints_ptr = c_loc(npoints(1))",
            "a_diag_ptr = c_loc(a_diag(1))",
            "m_vec_ptr = c_loc(m_vec(1))",
            "iblock_pos_ptr = c_loc(iblock_pos(1))",
            "new_surface_flag = merge(1_c_int, 0_c_int, new_surface)",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_scf_cuda_abi_v33_publication_pls_cosmo_direct_resident_cg",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "kMozymeScfAbiVersion = 33",
            "int resident_decision;",
            "int resident_fock_plan_id;",
            "int resident_fock_plan_full_coverage;",
            "int resident_fock_plan_partial_coverage;",
            "int resident_fock_plan_required_mask;",
            "int resident_fock_plan_covered_mask;",
            "int final_publication_done;",
            "int final_publication_arrays;",
            "std::size_t final_publication_bytes;",
            "int final_publication_cosmo;",
            "int cnvgz_active_calls;",
            "int cnvgz_noop_calls;",
            "int strict_resident_host_syncs;",
            "int strict_resident_control_polls;",
            "ctx->strict_resident_host_syncs = 0;",
            "ctx->strict_resident_control_polls = 0;",
            "status->strict_resident_host_syncs =",
            "status->strict_resident_control_polls =",
            "copy_resident_control_snapshot_from_gpu(*ctx, &control, false)",
            "completed |= stage_bits;",
            "const int synchronize_sparse_fock = strict_resident ? 0 : 1;",
            "ResidentFinalPublicationProof",
            "mark_final_publication_done",
            "resident_decision_to_code",
            "status->resident_decision = resident_decision_to_code(control.decision);",
            "status->resident_fock_plan_full_coverage",
            "status->resident_fock_plan_required_mask",
            "status->cnvgz_active_calls = cnvgz_ints[kCnvgzIntActiveCalls];",
            "status->cnvgz_noop_calls = cnvgz_ints[kCnvgzIntNoopCalls];",
            "config.resident_fock_plan_covered_mask == expected_required_mask",
            "kMozymeScfFlagFinalReorth",
            "mozyme_pls_supervisor_device",
            "ResidentReturnDecision::PlsRestart",
            "cosmo_enabled",
            "param_dd",
            "param_qq",
            "param_tore",
            "cosmo_qscnet",
            "publish_cosmo_status",
            "publish_cosmo_status_from_gpu",
            "preserve_resident_cosmo_cg_status",
            "capture_host_checkpoint",
            "restore_host_checkpoint",
            "status->cosmo_fock_calls",
            "kCosmoCgMatvecCalls",
            "kCosmoStatusControlResident",
            "kCosmoStatusCgConverged",
            "kCosmoStatusHostSyncs",
            "status->cosmo_cg_control_resident",
            "status->cosmo_cg_converged",
            "status->cosmo_cg_host_syncs",
            "mozyme_cosmo_cg_finalize_status_kernel",
            "mozyme_cosmo_cg_mark_fock_status_kernel",
            "resident final COSMO CG scalar status copy",
            "mozyme_cosmo_cg_prepare_iteration_kernel",
            "mozyme_cosmo_cg_prepare_update_kernel",
            "mozyme_cosmo_cg_finish_iteration_kernel",
            "resident terminal stage no-op",
            "state.use_nijbo != 1 || !state.nijbo",
            "apply_cosmo_fock_on_gpu",
            "mozyme_final_reorth_status_kernel",
            "final_reorth_status_scalars",
            "resident final status scalar copy",
            "device_final_reorth_committed",
            "ok || device_final_reorth_committed",
            "mozyme_cosmo_build_potential_kernel",
            "run_cosmo_matvec",
            "mozyme_cosmo_matvec_far_kernel",
            "mozyme_cosmo_matvec_close_kernel",
            "mozyme_cosmo_fock_correction_kernel",
            "cosmo_surface_at",
            "atomicAdd_double(out",
            "atomicAdd_double(qscat",
            "dev.cosmo_ipiden.upload",
            "dev.cosmo_qdenet.upload",
            "dev.cosmo_cg_x.resize(cosmo_nps_count)",
            "dev.cosmo_cg_scalars.resize(kCosmoCgScalarCount)",
            "checkpoint_cosmo_qscnet",
            "copy_device_buffer(dev.cosmo_qscnet, dev.checkpoint_cosmo_qscnet)",
            "stage_and_commit_device_vector(ctx.state.cosmo_qscat",
            "stage_and_commit_device_vector(ctx.state.cosmo_phinet",
            "stage_and_commit_device_vector(ctx.state.cosmo_qdenet",
            "std::memcpy(ctx.state.cosmo_solv_energy_ptr",
            "std::memcpy(ctx.state.cosmo_ediel_ptr",
            "if (mode == 0 && !apply_cosmo_fock_on_gpu(ctx, density_p, output_f))",
            "state.cosmo_a_part_dim > 0",
            "state.cosmo_npoints_dim < numat + 1",
            "kResidentControlPlsRestartRequired",
            "kResidentControlPlsRestartResetCalls",
            "kResidentControlPlsRestartDone",
            "status->pls_restart_required",
            "status->pls_restart_reset_device_calls",
            "status->pls_restart_done",
            "mozyme_resident_pls_restart_zero_kernel",
            "mozyme_resident_pls_restart_finalize_kernel",
            "control_ints[kResidentControlDiaggMode] = 0;",
            "control_scalars[kResidentControlDiaggOldlim] = 0.0;",
            "const long long restart_budget = static_cast<long long>(config.max_iter);",
            "const long long limit = (remaining > 0LL ? remaining : 0LL) + restart_budget;",
            "apply_resident_pls_restart_if_requested_on_gpu",
            "resident PLS restart zero kernel",
            "resident PLS restart finalize kernel",
            "bool valid_state(const MozymeScfConfig &config, const MozymeScfState &state)",
            "state.p_dim < mpack",
            "state.f_dim < mpack",
            "state.h_dim < mpack",
            "state.pold_dim < mpack",
            "state.partp_dim < 1",
            "state.partf_dim < 1",
            "state.p1_dim < norbs",
            "state.p2_dim < norbs",
            "state.p3_dim < norbs",
            "state.idiag_dim < norbs",
            "state.eigs_dim < norbs",
            "state.nfmo_dim < norbs",
            "state.iorbs_dim < numat",
            "state.kopt_dim < numat",
            "state.nfirst_dim < numat",
            "state.nlast_dim < numat",
            "state.ncf_dim < state.nocc_slots",
            "state.nncf_dim < state.nocc_slots",
            "state.ncocc_dim < state.nocc_slots",
            "state.nce_dim < state.nvir_slots",
            "state.nnce_dim < state.nvir_slots",
            "state.ncvir_dim < state.nvir_slots",
            "state.ifmo_rows != 2",
            "state.ifmo_cols < state.fmo_dim",
            "state.nijbo_rows != numat",
            "state.nijbo_cols < numat",
            "resident_stage_calls[kResidentStageSlotCount]",
            "resident_stage_ms[kResidentStageSlotCount]",
            "final_reorth_applied",
            "final_reorth_ms",
            "final_reorth_sum",
            "if (!valid_state(ctx->config, *state)) return kMozymeScfBadArgument;",
            "if (!valid_state(ctx.config, ctx.state)) return false;",
        ),
        "forbidden_fragments": (
            "bool valid_state(const MozymeScfState &state)",
            "(config.resident_fock_plan_covered_mask &\n          config.resident_fock_plan_required_mask) == expected_required_mask",
            "publish_cosmo_status(status, ctx);\n  if (final_code != kMozymeScfSuccess)",
        ),
    },
    {
        "name": "resident_scf_fortran_layout_kind_validation",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "state_layout_has_required_sizes(config, state, missing_field)",
            "resident_scf_state_layout_preflight",
            "reason=state_layout_invalid missing=",
            "storage_size(0) == storage_size(0_c_int)",
            "storage_size(0.0d0) == storage_size(0.0_c_double)",
            "size(p) >= mpack",
            "size(f) >= mpack",
            "size(h) >= mpack",
            "size(partp) >= mpack",
            "size(partf) >= mpack",
            "size(pold) >= mpack",
            "size(p1) >= norbs",
            "size(p2) >= norbs",
            "size(p3) >= norbs",
            "size(idiag) >= norbs",
            "size(iorbs) >= numat",
            "size(kopt) >= numat",
            "size(nfirst) >= numat",
            "size(nlast) >= numat",
            "size(ncf) < int(state%nocc_slots)",
            "size(nce) < int(state%nvir_slots)",
            "size(ifmo, 1) == 2",
            "size(ifmo, 2) >= fmo_dim",
            "size(nijbo, 1) == numat",
            "size(nijbo, 2) >= numat",
        ),
        "forbidden_fragments": (
            "cosmo_cg_target_tolerance(ctx)",
            "ctx.cosmo_matvec_calls += host_cg_ints[kCosmoCgMatvecCalls]",
            "cudaMemcpy(host_cg_scalars",
            "cudaMemcpy(host_cg_ints",
            "resident COSMO CG scalar status copy",
            "resident COSMO CG integer status copy",
            "iter == 0 ? 1 : 0",
        ),
    },
    {
        "name": "resident_scf_fortran_c_int_nfirst_nlast",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "type(c_ptr) function ptr_or_null_c_int(values) result(ptr)",
            "integer(c_int), allocatable, target, intent(inout) :: values(:)",
            "state%nfirst = ptr_or_null_c_int(nfirst)",
            "state%nlast = ptr_or_null_c_int(nlast)",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_scf_eimp_density_fail_closed",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "mozyme_eimp_kernel(int numat, int mpack",
            "int *pair_updates,",
            "int *ok_out, int *expected_updates",
            "atomicExch(ok_out, 0);",
            "if (base < 0) return;",
            "atomicAdd(expected_updates, 1);",
            "mozyme_update_status_finalize_kernel",
            "const int actual = status[0];",
            "const int expected = status[2];",
            "actual != expected",
            "resident eimp status reset",
            "mozyme_set_int_slot_if_resident_active_kernel<<<1, 1>>>(\n        dev.eimp_pair_updates.ptr, 1, 1, dev.resident_control_ints.ptr)",
            "dev.eimp_pair_updates.ptr + 2",
            "mozyme_update_status_finalize_kernel<<<1, 1>>>(\n        dev.eimp_pair_updates.ptr, 0, dev.resident_control_ints.ptr)",
            "d_updates.upload(host_status, 3)",
            "d_updates.ptr + 2",
            "host_status[1] != 1",
            "host_status[0] != host_status[2]",
            "mozyme_density_resident_kernel(",
            "mozyme_density_expected_kernel(",
            "int *updated_terms, int *ok_out",
            "resident density status reset",
            "mozyme_set_int_slot_if_resident_active_kernel<<<1, 1>>>(\n        dev.density_updates.ptr, 1, 1, dev.resident_control_ints.ptr)",
            "dev.density_updates.ptr + 2",
            "mozyme_update_status_finalize_kernel<<<1, 1>>>(\n        dev.density_updates.ptr, 1, dev.resident_control_ints.ptr)",
            "complete_stage_from_int(kMozymeScfStageEimp,",
            "kResidentStageSlotEimp",
            "complete_stage_from_int(kMozymeScfStageDensity,",
            "kResidentStageSlotDensity",
        ),
        "forbidden_fragments": (
            "if (base < 0 || base + terms > mpack) return;",
            "expected_pair_updates",
            "expected_updates_ll",
            "int host_pair_status[3] = {0, 1, 0};",
            "dev.eimp_pair_updates.upload(host_pair_status, 3)",
            "dev.eimp_pair_updates.upload(host_pair_status, 2)",
            "mozyme_set_int_slot_kernel<<<1, 1>>>(dev.eimp_pair_updates.ptr, 1, 1)",
            "d_updates.upload(host_status, 2)",
            "const int *host_ncf = static_cast<const int *>(ctx.state.ncf);",
            "const int *host_nijbo = static_cast<const int *>(ctx.state.nijbo);",
            "host_pair_status[0] != host_pair_status[2]",
            "host_pair_status[1] != 1",
            "cudaMemcpy(host_pair_status, dev.eimp_pair_updates.ptr",
            "expected_density_updates",
            "host_updates != expected_density_updates",
            "int host_density_status[3] = {0, 1, 0};",
            "dev.density_updates.upload(host_density_status, 3)",
            "mozyme_set_int_slot_kernel<<<1, 1>>>(dev.density_updates.ptr, 1, 1)",
            "host_density_status[1] != 1",
            "cudaMemcpy(host_density_status, dev.density_updates.ptr",
        ),
    },
    {
        "name": "resident_scf_complete_state_copyback_checkpoint",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "DeviceBuffer<int> checkpoint_idiag;",
            "DeviceBuffer<int> checkpoint_nncf;",
            "DeviceBuffer<int> checkpoint_ncocc;",
            "DeviceBuffer<int> checkpoint_nnce;",
            "DeviceBuffer<int> checkpoint_ncvir;",
            "DeviceBuffer<int> checkpoint_isitsc_ints;",
            "DeviceBuffer<double> checkpoint_isitsc_scalars;",
            "DeviceBuffer<double> checkpoint_resident_control_scalars;",
            "copy_device_buffer(dev.checkpoint_idiag, dev.idiag)",
            "copy_device_buffer(dev.checkpoint_nncf, dev.nncf)",
            "copy_device_buffer(dev.checkpoint_isitsc_ints, dev.isitsc_ints)",
            "copy_device_buffer(dev.checkpoint_isitsc_escf0, dev.isitsc_escf0)",
            "copy_device_buffer(dev.checkpoint_resident_control_ints,\n                            dev.resident_control_ints)",
            "copy_device_buffer(dev.idiag, dev.checkpoint_idiag)",
            "copy_device_buffer(dev.isitsc_ints, dev.checkpoint_isitsc_ints)",
            "copy_device_buffer(dev.isitsc_escf0, dev.checkpoint_isitsc_escf0)",
            "copy_device_buffer(dev.resident_control_ints,\n                            dev.checkpoint_resident_control_ints)",
            "resize_resident_checkpoint_buffers",
            "resident checkpoint device copy",
            "stage_and_commit_device_vector(ctx.state.idiag, dev.idiag, norbs_count",
            "stage_and_commit_device_vector(ctx.state.nncf, dev.nncf, nocc_slots",
            "stage_and_commit_device_vector(ctx.state.ncvir, dev.ncvir, nvir_slots",
        ),
        "forbidden_fragments": (
            "return src.ptr && dst.copy_from_device(src.ptr, src.count);",
            "copy_device_to_host_vector(host_idiag, dev.idiag)",
            "stage_device_to_vector(host_idiag, dev.idiag, norbs_count",
            "stage_device_to_vector(host_nncf, dev.nncf, nocc_slots",
            "stage_device_to_vector(host_ncvir, dev.ncvir, nvir_slots",
            "commit_staged_vector(ctx.state.idiag, host_idiag)",
            "commit_staged_vector(ctx.state.nncf, host_nncf)",
            "commit_staged_vector(ctx.state.ncvir, host_ncvir)",
            "commit_host_vector(ctx.state.idiag, host_idiag)",
        ),
    },
    {
        "name": "resident_scf_preallocated_checkpoints",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "bool resize_checkpoint_like(DeviceBuffer<T> &checkpoint",
            "bool resize_resident_checkpoint_buffers(MozymeScfDeviceState &dev)",
            "resize_checkpoint_like(dev.checkpoint_p, dev.p)",
            "resize_checkpoint_like(dev.checkpoint_cosmo_qscnet",
            "resize_checkpoint_like(dev.checkpoint_resident_control_ints",
            "if (!resize_resident_checkpoint_buffers(dev)) return false;",
            "if (!src.ptr || !device_buffer_ready(dst, src.count)) return false;",
            "cudaMemcpy(dst.ptr, src.ptr, src.count * sizeof(T),",
            "resident checkpoint device copy",
        ),
        "forbidden_fragments": (
            "return src.ptr && dst.copy_from_device(src.ptr, src.count);",
        ),
    },
    {
        "name": "strict_resident_blocks_cpu_check_and_pls",
        "path": "src/MOZYME/iter_for_MOZYME.F90",
        "fragments": (
            "strict_cpu_lmo_check",
            "MOZYME GPU strict resident SCF does not support CPU LMO check",
            "strict_cpu_pls_supervisor",
            "MOZYME GPU strict resident SCF does not support CPU PLS supervisor",
            "resident_pls_restart_needed = PLS_faulty()",
            "olden_setup=host_lmo_restore setup_only=1",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_resident_no_partial_host_publish",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "enum class ResidentReturnDecision",
            "CompleteAndPublish",
            "CpuBoundary",
            "IterationExhausted",
            "PlsRestart",
            "StageFailed",
            "kResidentDecisionIterationExhausted",
            "kResidentDecisionPlsRestart",
            "kResidentDecisionStageFailed",
            "compute_resident_control_after_iteration",
            "mozyme_resident_control_advance_kernel",
            "if (code == kResidentDecisionIterationExhausted) {\n    return ResidentReturnDecision::IterationExhausted;\n  }",
            "if (code == kResidentDecisionStageFailed) {\n    return ResidentReturnDecision::StageFailed;\n  }",
            "return_decision == ResidentReturnDecision::CompleteAndPublish",
            "return_decision == ResidentReturnDecision::CpuBoundary",
            "return_decision == ResidentReturnDecision::IterationExhausted",
            "return_decision == ResidentReturnDecision::StageFailed",
            "strict_resident_request_enabled",
            "resident_scf_request_enabled",
            'env_enabled("MOPAC_MOZYME_SCF_GPU")',
            'env_enabled("MOPAC_MOZYME_FULL_SCF_GPU")',
            'env_enabled("MOPAC_MOZYME_GPU_STRICT")',
            'env_enabled("MOPAC_MOZYME_RESIDENT_SCF")',
            "kMozymeScfCpuBoundary",
            "if (!run_resident_iteration_on_gpu(*ctx, status, &accumulated_ms,\n                                           false))",
            "advance_strict_resident_control_on_gpu",
            "return apply_resident_pls_restart_if_requested_on_gpu(ctx);",
            "if (!advance_strict_resident_control_on_gpu(*ctx))",
            "} else if (return_decision == ResidentReturnDecision::PlsRestart) {",
            "MozymeScfStatus final_status = *status;",
            "publish_resident_control_to_status(&final_status, control);",
            "*status = final_status;",
            "resident_stage_status_complete",
            "!resident_stage_status_complete(final_status,",
            "status.stage_missing == 0",
            "publish_resident_stage_status_from_gpu(*ctx, status)",
            "publish_strict_stage_status_or_not_ready",
            "final_code = kMozymeScfNotReady;",
            "[MOZYME GPU SCF] host_commit_only=1",
            "phase=final_publication",
            "host_commit_marker_enabled",
            "const bool step_complete =\n            run_resident_iteration_on_gpu(*ctx, status, &accumulated_ms, true);",
            "mozyme_resident_pls_restart_zero_kernel",
            "mozyme_resident_pls_restart_finalize_kernel",
            "apply_resident_pls_restart_if_requested_on_gpu",
            "device_buffer_ready(dev.resident_control_ints",
            "device_buffer_ready(dev.resident_control_scalars",
        ),
        "forbidden_fragments": (
            "if (must_return_to_fortran_after_iteration(*ctx, *status))",
            "if (!strict_resident && !copy_resident_state_to_host(*ctx))",
            "if (strict_resident) {\n          final_code = kMozymeScfCpuBoundary;",
            "if (strict_resident) continue;",
            "run_resident_iteration_on_gpu(*ctx, status, &accumulated_ms,\n                                        !strict_resident)",
            "if (!run_resident_iteration_on_gpu(*ctx, status, &accumulated_ms,\n                                           true))",
            "resident PLS restart state copy",
            "reset_resident_after_pls_restart(*ctx)",
            "dev.resident_control_ints.upload(control_ints,",
            "dev.resident_control_scalars.upload(control_scalars,",
            "dev.isitsc_ints.upload(isitsc_ints,",
            "dev.pls_ints.upload(pls_ints,",
            "compute_resident_control_after_iteration(*ctx, true, false",
            "compute_resident_control_after_iteration(*ctx, true, true",
            "resident loop control decision copy",
            "sizeof(host_decision)",
            "resident_decision_from_code(host_decision)",
            "copy_full_snapshot ? copy_resident_control_snapshot_from_gpu(ctx, out)",
        ),
    },
    {
        "name": "resident_scf_cpu_boundary_status",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "GPU_MOZYME_SCF_CPU_BOUNDARY",
            "backend_cpu_boundary",
            "backend_pls_restart_required",
            "backend_resident_decision",
            "backend_resident_tidy_missing",
            "backend_pls_resolved(status)",
            "backend_resident_tidy_complete(initial_setup_requested, &",
            "full_success = backend_completed_scf(code, status, niter, &\n      initial_setup_requested, initial_tidy_completed)",
            "status, niter, initial_setup_requested, initial_tidy_completed))",
            "backend_resident_fock_plans_complete(status)",
            "status%resident_fock_plan_covered_mask == &\n      status%resident_fock_plan_required_mask",
            "status%resident_decision == &\n      GPU_MOZYME_SCF_RESIDENT_DECISION_COMPLETE",
            "pls_restart_required",
            "CPU_BOUNDARY",
            "status%stage_missing == 0_c_int",
        ),
        "forbidden_fragments": (
            'env_enabled("MOPAC_MOZYME_SCF_GPU") ||\n         env_enabled("MOPAC_MOZYME_RESIDENT_SCF")',
            "iand(status%resident_fock_plan_covered_mask, &\n      status%resident_fock_plan_required_mask) == &\n      status%resident_fock_plan_required_mask",
        ),
    },
    {
        "name": "resident_scf_cnvgz_stage_call_normalization",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "call normalize_cnvgz_activity(status)",
            "subroutine normalize_cnvgz_activity(status)",
            "status%cnvgz_active_calls + status%cnvgz_noop_calls <= 0_c_int",
            "status%resident_stage_calls(6) > 0_c_int",
            "status%cnvgz_noop_calls = status%resident_stage_calls(6)",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "strict_resident_no_timing_sync",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "resident_stage_timing_enabled",
            "strict_resident_request_enabled",
            "return wall_ms && !strict_resident_request_enabled();",
            "cudaMemcpyAsync(dev.eimp_p.ptr, dev.p.ptr",
            "resident eimp shadow density copy",
            "begin_resident_stage_timing",
            "finish_resident_stage_timing",
            "mopac_cuda_mozyme_sparse_fock_run_device_plan_guarded_resident",
            "if (time_stage &&",
            "compute_helecz_on_gpu(ctx, nullptr, &wall_ms)",
        ),
        "forbidden_fragments": (
            "if (strict_resident_request_enabled()) {\n    if (!cuda_context_ok(cudaDeviceSynchronize()",
            "if (strict_resident_request_enabled() &&\n        !cuda_context_ok(cudaDeviceSynchronize()",
            "if (strict_resident &&\n        !cuda_context_ok(cudaDeviceSynchronize()",
            "if (strict_resident_request_enabled()) {\n    if (!cuda_context_ok(cudaEventSynchronize(",
            "if (strict_resident_request_enabled() &&\n        !cuda_context_ok(cudaEventSynchronize(",
            "if (strict_resident &&\n        !cuda_context_ok(cudaEventSynchronize(",
        ),
    },
    {
        "name": "resident_scf_diagg_addhb_persistent_scratch",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "DeviceBuffer<double> diagg_aocc;",
            "DeviceBuffer<int> diagg_work_ints;",
            "DeviceBuffer<double> diagg_avir_entry;",
            "DeviceBuffer<int> diagg_pair_state;",
            "DeviceBuffer<int> hb_pair_i;",
            "dev.diagg_aocc.resize(",
            "if (!dev.diagg_pair_state.resize(fmo_count)) return false;",
            "if (!dev.diagg_vclaim.resize(nvir_count)) return false;",
            "if (!dev.diagg_oclaim.resize(nocc_count)) return false;",
            "if (!dev.hb_entry_offsets.resize(hb_capacity + 1)) return false;",
            "dev.diagg_ints.resize(kDiaggIntCount)",
            "dev.addhb_ints.resize(kAddhbIntCount)",
            "dev.diagg_aocc.ptr,\n        dev.diagg_work_ints.ptr + kDiaggWorkIntError",
            "va.avir_entry = dev.diagg_avir_entry.ptr;",
            "a.pair_state = dev.diagg_pair_state.ptr;",
        ),
        "forbidden_fragments": (
            "DeviceBuffer<double> aocc;",
            "DeviceBuffer<double> avir;",
            "DeviceBuffer<double> aov;",
            "if (!aocc.resize(",
            "if (!avir.resize(",
            "if (!aov.resize(",
            "if (!iused.resize(static_cast<std::size_t>(max_iused)))",
            "DeviceBuffer<int> iused;\n  DeviceBuffer<int> latoms;\n  DeviceBuffer<double> storei;",
        ),
    },
    {
        "name": "resident_scf_preallocated_stage_buffers",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "dev.eimp_p.resize(mpack_count)",
            "dev.eimp_pair_updates.resize(3)",
            "dev.density_updates.resize(3)",
            "dev.qe.resize(numat_count)",
            "dev.cnvgz_ints.resize(kCnvgzIntCount)",
            "dev.isitsc_scalars.resize(kIsitscDoubleCount)",
            "dev.check_ints.resize(kCheckIntCount)",
            "dev.check_errors.resize(kCheckDoubleCount)",
            "dev.helecz_ints.resize(kHeleczIntCount)",
            "dev.fock_ints.resize(kFockIntCount)",
            "dev.resident_stage_ints.resize(kResidentStageIntCount)",
            "dev.resident_stage_calls.upload(initial_stage_calls",
            "mozyme_resident_stage_upload_accept_kernel",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_scf_no_late_stage_resize",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "bool device_buffer_ready(const DeviceBuffer<T> &device",
            "cudaMemcpyAsync(dev.eimp_p.ptr, dev.p.ptr",
            "device_buffer_ready(dev.eimp_p, mpack_count)",
            "device_buffer_ready(dev.eimp_pair_updates, 3)",
            "device_buffer_ready(dev.check_ints, kCheckIntCount)",
            "device_buffer_ready(dev.density_updates, 3)",
            "device_buffer_ready(dev.cosmo_cg_x, nps_count)",
            "device_buffer_ready(dev.cosmo_status_scalars",
            "device_buffer_ready(dev.qe, static_cast<std::size_t>(numat))",
            "device_buffer_ready(dev.fock_ints, kFockIntCount)",
            "device_buffer_ready(dev.helecz_ints, kHeleczIntCount)",
            "device_buffer_ready(dev.isitsc_scalars, kIsitscDoubleCount)",
            "device_buffer_ready(dev.cnvgz_ints, kCnvgzIntCount)",
            "device_buffer_ready(dev.resident_stage_ints",
            "device_buffer_ready(dev.resident_control_ints",
        ),
        "forbidden_fragments": (
            "copy_from_device(const T *device_src",
            "copy_from_device_async(const T *device_src",
            "if (!dev.eimp_p.copy_from_device_async(",
            "if (!dev.eimp_pair_updates.resize(3)) break;",
            "if (!dev.check_errors.resize(kCheckDoubleCount)) break;",
            "if (!dev.check_ints.resize(kCheckIntCount)) break;",
            "if (!dev.density_updates.resize(3)) break;",
            "if (!dev.cosmo_cg_x.resize(static_cast<std::size_t>(nps)))",
            "if (!dev.cosmo_status_scalars.resize(kCosmoStatusDoubleCount))",
            "if (!dev.qe.resize(static_cast<std::size_t>(numat))) break;",
            "if (!dev.fock_ints.resize(kFockIntCount)) break;",
            "if (!dev.helecz_ints.resize(kHeleczIntCount)) break;",
            "if (!dev.isitsc_scalars.resize(kIsitscDoubleCount)) break;",
            "if (!dev.resident_control_ints.resize(kResidentControlIntCount)) return false;",
            "if (!dev.resident_control_scalars.resize(kResidentControlDoubleCount))",
        ),
    },
    {
        "name": "strict_resident_sparse_fock_default_stream",
        "path": "src/gpu/cuda_wrappers.cu",
        "fragments": (
            "mozyme_sparse_fock_run_device_plan_guarded_impl",
            "force_default_stream ? 0 : (g_stream ? g_stream : 0)",
            "wait_for_completion = (synchronize != 0) || measure",
            "mopac_cuda_mozyme_sparse_fock_run_device_plan_guarded_resident",
            "force_default_stream",
            "true, synchronize, wall_ms",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_scf_stage_status_device",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "DeviceBuffer<int> resident_stage_ints;",
            "resident_stage_calls[kResidentStageSlotCount]",
            "resident_stage_ms[kResidentStageSlotCount]",
            "kResidentStageSlotCount = 10",
            "kResidentStageCompleted",
            "DeviceBuffer<int> resident_stage_calls;",
            "mozyme_resident_stage_reset_kernel",
            "mozyme_resident_stage_mark_if_int_kernel",
            "resident_control_ints[kResidentControlDecision] =\n          kResidentDecisionStageFailed;",
            "DeviceBuffer<int> cnvgz_ints;",
            "DeviceBuffer<int> fock_ints;",
            "kCnvgzIntOk",
            "kFockIntOk",
            "reset_resident_stage_status_on_gpu",
            "mark_resident_stage_if_int_on_gpu",
            "publish_resident_stage_status_from_gpu",
            "copy_resident_stage_calls_from_gpu",
            "status->resident_stage_calls[i] = host_calls[i]",
            "resident_stage_status_complete",
            "resident_stage_calls_complete",
            "required_resident_stage_calls",
            "status.resident_stage_calls[i] <",
            "ctx->config.current_iter",
            "return completed == kMozymeScfStageFull;",
            "status.stage_missing == 0",
            "complete_stage_from_int",
            "const bool poll_stage_status = publish_iteration_outputs;",
            "if (publish_iteration_outputs) {\n    publish_stage_status();",
            "completed_full_stage_mask(completed) ? kMozymeScfSuccess",
            "add_stage_time(status, accumulated_ms, check_ms, kResidentStageSlotCheck)",
            "add_stage_time(status, accumulated_ms, eimp_ms, kResidentStageSlotEimp)",
            "add_stage_time(status, accumulated_ms, diagg_ms, kResidentStageSlotDiagg)",
            "add_stage_time(status, accumulated_ms, density_ms,\n                   kResidentStageSlotDensity)",
            "add_stage_time(status, accumulated_ms, addhb_ms, kResidentStageSlotAddhb)",
            "add_stage_time(status, accumulated_ms, cnvgz_ms, kResidentStageSlotCnvgz)",
            "add_stage_time(status, accumulated_ms, fock_ms, kResidentStageSlotFock)",
            "add_stage_time(status, accumulated_ms, wall_ms, kResidentStageSlotHelecz)",
            "add_stage_time(status, accumulated_ms, isitsc_ms,\n                   kResidentStageSlotIsitsc)",
            "complete_stage_from_int(kMozymeScfStageCheck, ctx.device.check_ints.ptr",
            "complete_stage_from_int(kMozymeScfStageEimp,",
            "complete_stage_from_int(kMozymeScfStageDiagg,",
            "complete_stage_from_int(kMozymeScfStageDensity,",
            "complete_stage_from_int(kMozymeScfStageAddhb,",
            "complete_stage_from_int(kMozymeScfStageCnvgz,",
            "complete_stage_from_int(kMozymeScfStageFock,",
            "complete_stage_from_int(kMozymeScfStageHelecz,",
            "complete_stage_from_int(kMozymeScfStageIsitsc,",
        ),
        "forbidden_fragments": (
            "mozyme_resident_stage_mark_kernel",
            "mark_resident_stage_on_gpu",
            "complete_stage(kMozymeScfStage",
            "status->iterations = ctx.config.current_iter;\n  publish_stage_status();\n  if (publish_iteration_outputs) {",
            "if (!mark_resident_stage_if_int_on_gpu(ctx, stage_bits, values,\n                                           value_slot, expected_value)) {\n      return false;\n    }\n    if (!publish_resident_stage_status_from_gpu(ctx, status)) return false;",
        ),
    },
    {
        "name": "resident_scf_stage_runtime_trace",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "status%resident_stage_calls = 0_c_int",
            "status%resident_stage_ms = 0.0d0",
            "'[MOZYME GPU SCF]', 'resident_stage_calls'",
            "'upload=', int(status%resident_stage_calls(1))",
            "'check=', int(status%resident_stage_calls(10))",
            "'[MOZYME GPU SCF]', 'resident_stage_ms'",
            "'helecz=', status%resident_stage_ms(7)",
            "'isitsc=', status%resident_stage_ms(8)",
            "backend_stage_calls_complete(status, previous_iter)",
            "backend_stage_call_counts",
            "int(status%resident_stage_calls(idx)) < min_calls",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_scf_setupk_all_initial_setup",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "if (initial_setup_requested) then\n      if (.not. mozyme_gpu_setupk_try(nocc, fock_mode)) then",
            "scf_failure_message('reason=backend_not_ready detail=setupk')",
            "resident_fock_required_mask = 1_c_int",
            "if (fock_mode /= 0) resident_fock_required_mask = ior(resident_fock_required_mask, 2_c_int)",
            "resident_fock_plan_ready = mozyme_resident_fock_prepare_plan(",
            "resident_fock_plan_full, iorbs, nat, ifact,",
            "resident_fock_plan_partial, iorbs, nat, ifact,",
            "if (fock_mode /= 0) then\n      if (id == 0) then",
            "[MOZYME GPU setupk]",
            "initial_setup=1",
            "fock_mode=",
            "all_initial_setup_paths=1",
            "trim(scf_failure_status())",
            "printed_ms = max(wall_ms, 0.001_c_double)",
        ),
        "forbidden_fragments": (
            "if (initial_setup_requested .and. fock_mode /= 0) then\n      if (.not. mozyme_gpu_setupk_try(nocc",
        ),
    },
    {
        "name": "resident_scf_fock_density_source",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "const double *density_p, const double *base_f",
            "dev.nijbo.ptr, density_p, dev.qe.ptr",
            "plan_id, mpack, density_p, dev.qe.ptr, output_f",
            "ctx.device.p.ptr, ctx.device.partf.ptr",
            "dev.p.ptr, dev.partf.ptr, dev.f.ptr",
            "-1, dev.partp.ptr, dev.f.ptr",
        ),
        "forbidden_fragments": (),
    },
    {
        "name": "resident_fock_partial_point_cpu_completion",
        "path": "src/MOZYME/fock2z.F90",
        "fragments": (
            "mozyme_resident_point_supported, mozyme_point_charge_advance_kr",
            "resident_fock_done .and. &\n                  mozyme_resident_point_supported(iab, jba, ijbo(ii, jj))",
            "resident_fock_done .and. &\n                  mozyme_resident_point_supported(iab, jba, nijbo(ii, jj))",
        ),
        "forbidden_fragments": (
            "if (resident_fock_done) then\n                call mozyme_point_charge_advance_kr",
        ),
    },
    {
        "name": "resident_scf_cnvgz_device_control",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "constexpr int kCnvgzControlCount = 6;",
            "constexpr int kCnvgzIntActiveCalls = 1;",
            "constexpr int kCnvgzIntNoopCalls = 2;",
            "constexpr int kCnvgzIntCount = 3;",
            "mozyme_cnvgz_finalize_kernel",
            "control[kCnvgzDensityRms]",
            "control[kCnvgzFactor] = factor;",
            "status_ints[kCnvgzIntActiveCalls] += 1;",
            "status_ints[kCnvgzIntNoopCalls] += 1;",
            "mozyme_cnvgz_candidate_kernel(int mpack,\n                                              const double *control",
            "const double factor = control[kCnvgzFactor];",
            "mozyme_cnvgz_damp_kernel(int norbs, int mpack,\n                                         const double *control",
            "const double pmax = control[kCnvgzPmax];",
            "resident_control_uses_three_point_or",
            "mozyme_cnvgz_commit_matrix_kernel",
            "mozyme_cnvgz_commit_diag_kernel",
            "dev.cnvgz_sums.resize(kCnvgzControlCount)",
            "d_cnvgz_sums.resize(kCnvgzControlCount)",
            "mozyme_cnvgz_candidate_kernel<<<matrix_blocks, kThreads>>>(\n        mpack, dev.cnvgz_sums.ptr",
            "dev.candidate.ptr, dev.resident_control_ints.ptr",
            "mozyme_cnvgz_damp_kernel<<<diag_blocks, kThreads>>>(\n        norbs, mpack, dev.cnvgz_sums.ptr",
            "dev.diag_old.ptr, dev.candidate.ptr, dev.resident_control_ints.ptr",
            "mozyme_cnvgz_commit_matrix_kernel<<<matrix_blocks, kThreads>>>",
            "mozyme_cnvgz_commit_diag_kernel<<<diag_blocks, kThreads>>>",
            "resident final cnvgz scalar copy",
        ),
        "forbidden_fragments": (
            "double factor = 0.0;\n    if (use_three_point && niter % 3 == 0)",
            "double factor = 0.0;\n    if (use_three_point != 0 && niter % 3 == 0)",
            "std::sqrt(faca / facb)",
            "mpack, factor, dev.p.ptr",
            "mpack, factor, d_pnew.ptr",
            "norbs, mpack, pmax, dev.idiag.ptr",
            "norbs, mpack, pmax, d_idiag.ptr",
            "resident cnvgz factor totals copy",
            "cnvgz factor totals copy",
            "resident cnvgz totals copy",
        ),
    },
    {
        "name": "resident_scf_helecz_device_total",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "totals[2] = offdiag_sums[0] + 0.5 * diag_sums[0];",
            "const int *ok_in",
            "ok_in[kHeleczIntOk] != 1",
            "DeviceBuffer<int> helecz_ints;",
            "dev.helecz_ints.ptr + kHeleczIntOk",
            "const bool publish_energy = energy != nullptr;",
            "dev.energy_sums.resize(3)",
            "d_energy_sums.resize(3)",
            "d_valid.ptr,\n        d_energy_sums.ptr",
            "double host_totals[3] = {0.0, 0.0, 0.0};",
            "*energy = host_totals[2];",
            "resident helecz status reset",
            "compute_helecz_on_gpu(ctx, nullptr, &wall_ms)",
            "resident final energy totals copy",
            "status->energy_total = energy_totals[2];",
        ),
        "forbidden_fragments": (
            "const double offdiag = host_totals[0];",
            "const double diag = host_totals[1];",
            "const double total = offdiag + 0.5 * diag;",
            "dev.check_ints.ptr);\n    if (!cuda_context_ok(cudaGetLastError(),\n                         \"resident helecz atom kernel\"",
            "resident isitsc helecz totals copy",
        ),
    },
    {
        "name": "resident_scf_diagg_addhb_device_control",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "mozyme_diagg2_prepare_control_kernel",
            "mozyme_diagg2_finalize_control_kernel",
            "mozyme_addhb_prepare_control_kernel",
            "mozyme_addhb_finalize_control_kernel",
            "const int *diagg_ints, const double *diagg_scalars",
            "dev.diagg_ints.ptr, dev.diagg_scalars.ptr, dev.addhb_ints.ptr",
            "if (control_ints[kAddhbIntOk] < 0 || control_ints[kAddhbIntDue] == 0) {",
            "make_diagg_rotate_args(\n                ctx, dev.diagg_ints.ptr, kDiaggIntNij, kDiaggIntRetry,",
            "make_diagg_rotate_args(\n                ctx, dev.addhb_ints.ptr, kAddhbIntNij, kAddhbIntRetry,",
            "kDiaggDoubleRotateTiny",
            "kAddhbDoubleNextTiny",
            "publish_resident_iteration_outputs_from_gpu",
            "resident final diagg integer copy",
            "resident final addhb integer copy",
        ),
        "forbidden_fragments": (
            "if (host_nij > 0)",
            "const bool due = ctx.config.addhb_due",
            "diagg2_thresholds(",
            "diagg2_retry_from_state",
            "diagg2_set_rejections",
            "compute_addhb_on_gpu(ctx, next_tiny, diagg2_nrejct",
            "resident diagg integer copy",
            "resident addhb integer copy",
        ),
    },
    {
        "name": "resident_scf_isitsc_device_energy_control",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "DeviceBuffer<double> isitsc_escf0;",
            "dev.isitsc_escf0.upload(ctx.config.isitsc_escf0, 10)",
            "dev.isitsc_ints.upload(initial_isitsc_ints, kIsitscIntCount)",
            "mozyme_isitsc_resident_kernel",
            "mozyme_isitsc_resident_kernel<<<1, 1>>>(\n        dev.energy_sums.ptr, 2",
            "dev.diagg_scalars.ptr, kDiaggDoubleTiny",
            "zero_ints_if_resident_active(\n            ctx, dev.isitsc_ints.ptr + kIsitscIntOkscf",
            "resident isitsc control reset",
            "kIsitscIntValid",
            "dev.isitsc_ints.ptr + kIsitscIntValid",
            "complete_stage_from_int(kMozymeScfStageIsitsc,",
            "compute_isitsc_on_gpu(ctx, &isitsc_ms)",
            "resident final isitsc history copy",
            "const double energy = energy_values[energy_slot];",
            "resident terminal stage no-op",
        ),
        "forbidden_fragments": (
            "DeviceBuffer<double> escf0;",
            "if (!escf0.upload(ctx.config.isitsc_escf0",
            "energy, ctx.config.energy_scale",
            "compute_isitsc_on_gpu(ctx, diagg_tiny",
            "compute_isitsc_on_gpu(ctx, status, &isitsc_ms)",
            "int host_ints[kIsitscIntCount] = {}",
            "dev.isitsc_ints.upload(host_ints, kIsitscIntCount)",
            "resident isitsc integer copy",
            "resident isitsc scalar copy",
        ),
    },
    {
        "name": "resident_scf_iteration_accounting",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "call init_config(config, nocc, nvir, resident_max_iter, niter, &",
            "config%current_iter = mozyme_c_int_nonnegative_or_zero(current_iter)",
            "mod(current_iter + 1, 3) == 0 .and. nhb < 4",
            "status%stage_required == GPU_MOZYME_SCF_STAGE_FULL .and.",
            "status%stage_completed == GPU_MOZYME_SCF_STAGE_FULL .and.",
        ),
        "forbidden_fragments": (
            "resident_max_iter, niter + 1",
        ),
    },
    {
        "name": "resident_scf_direct_wk_guard",
        "path": "src/MOZYME/mozyme_gpu_scf_driver.F90",
        "fragments": (
            "logical :: resident_fock_plan_ready",
            "if (id == 0) then\n      resident_fock_plan_ready = mozyme_resident_fock_prepare_plan(",
            "resident_fock_plan_full, iorbs, nat, ifact, wj, wj, 0, kopt,",
            "resident_fock_plan_full, iorbs, nat, ifact, wj, wk, 0, kopt,",
            "resident_fock_plan_partial, iorbs, nat, ifact, wj, wj, fock_mode,",
            "resident_fock_plan_partial, iorbs, nat, ifact, wj, wk, fock_mode,",
        ),
        "forbidden_fragments": (
            "mozyme_resident_fock_prepare_plan(resident_fock_plan_full, &\n        iorbs, nat, ifact, wj, wk, 0,",
            "mozyme_resident_fock_prepare_plan(resident_fock_plan_partial, &\n          iorbs, nat, ifact, wj, wk, fock_mode,",
        ),
    },
    {
        "name": "resident_scf_loop_control_device",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "mozyme_resident_control_advance_kernel",
            "compute_resident_control_after_iteration",
            "if (!advance_resident_control_on_gpu(ctx, strict_resident)) return false;\n  return copy_resident_control_snapshot_from_gpu(ctx, out);",
            "advance_strict_resident_control_on_gpu",
            "if (!advance_resident_control_on_gpu(ctx, true)) return false;",
            "return apply_resident_pls_restart_if_requested_on_gpu(ctx);",
            "if (!advance_strict_resident_control_on_gpu(*ctx))",
            "while (resident_loop_guard < strict_loop_limit)",
            "ResidentControlSnapshot control{}",
            "copy_resident_control_snapshot_from_gpu(*ctx, &control, false)",
            "const ResidentReturnDecision return_decision = control.decision;",
            "const int previous_iter = ctx->config.current_iter;",
            "advance_config_after_resident_control(*ctx, control)",
            "resident_stage_status_complete(final_status, previous_iter)",
            "resident loop control advance kernel",
            "resident loop control snapshot integer copy",
            "const int completed_iter =\n      current_iter < 2147483647 ? current_iter + 1 : current_iter;",
            "else if (completed_iter >= max_iter) {\n    decision = kResidentDecisionIterationExhausted;\n  } else if (strict_resident == 0 &&\n             completed_iter > kMozymeScfPlsSupervisorLastIter) {\n    decision = kResidentDecisionCpuBoundary;\n  }",
            "current_iter = resident_control_int_or(\n      control_ints, kResidentControlCurrentIter, current_iter);",
            "control_ints[kResidentControlDecision] = decision;",
            "kResidentControlCurrentIter",
            "control_ints[kResidentControlCurrentIter] = next_iter;",
            "copy_resident_control_snapshot_from_gpu",
            "strict_resident_loop_limit",
            "config.current_iter < 0",
            "const long long remaining =",
            "static_cast<long long>(config.current_iter);",
            "const long long restart_budget = static_cast<long long>(config.max_iter);",
            "const long long limit = (remaining > 0LL ? remaining : 0LL) + restart_budget;",
        ),
        "forbidden_fragments": (
            "apply_cpu_loop_control_for_continuation",
            "addhb_due_for_current_iteration",
            "resident_return_decision_after_iteration",
            "compute_resident_control_after_iteration(*ctx, true, false",
            "compute_resident_control_after_iteration(*ctx, true, true",
            "bool compute_resident_control_after_iteration(MozymeScfContext &ctx,\n                                              bool strict_resident,\n                                              const MozymeScfStatus &status",
            "ctx->config.max_iter > 0 ? ctx->config.max_iter * 2 : 0",
            "config.current_iter <= 0",
            "static_cast<long long>(config.max_iter) + remaining",
            "resident loop control decision copy",
            "resident loop control integer copy",
            "resident loop control shift copy",
            "sizeof(host_decision)",
            "resident_decision_from_code(host_decision)",
            "copy_full_snapshot ? copy_resident_control_snapshot_from_gpu(ctx, out)",
        ),
    },
    {
        "name": "resident_scf_runtime_risk_fixes",
        "path": "src/gpu/mozyme_scf_context.cu",
        "fragments": (
            "atomicExch(&atom_flags[atom - 1], 1)",
            "mozyme_update_status_finalize_kernel",
            "mozyme_check_finalize_kernel",
            "mozyme_check_init_kernel",
            "resident check integer reset",
            "resident check errors reset",
            "mozyme_check_init_kernel<<<1, 1>>>(nocc, nvir, dev.check_ints.ptr,\n                                       dev.resident_control_ints.ptr)",
            "mozyme_check_finalize_kernel<<<1, 1>>>(nocc, nvir, dev.check_errors.ptr,",
            "complete_stage_from_int(kMozymeScfStageCheck, ctx.device.check_ints.ptr",
            "mozyme_resident_stage_upload_accept_kernel",
            "stage_calls[kResidentStageSlotUpload] = 1",
            "stage_calls[stage_slot] += 1",
            "DeviceBuffer<int> checkpoint_resident_stage_calls;",
            "copy_device_buffer(dev.checkpoint_resident_stage_calls",
            "copy_device_buffer(dev.resident_stage_calls",
            "int completed = 0;",
            "resident_stage_device_call_count",
            "copy_resident_stage_ints_from_gpu",
            "copy_resident_stage_calls_from_gpu",
            "resident stage device counters copy",
            "resident_stage_device_confirmed",
            "mozyme_set_int_slot_if_resident_active_kernel<<<1, 1>>>(\n        dev.eimp_pair_updates.ptr",
            "mozyme_set_int_slot_if_resident_active_kernel<<<1, 1>>>(\n        dev.density_updates.ptr",
            "mozyme_resident_stage_mark_if_int_kernel",
            "resident_control_ints[kResidentControlDecision] =\n          kResidentDecisionStageFailed;",
            "resident fock status reset",
            "if (!strict_resident_request_enabled()) {\n    if (!cuda_context_ok(cudaDeviceSynchronize(),\n                         \"resident COSMO synchronize\"",
            "bool save_resident_checkpoint(MozymeScfDeviceState &dev) {\n  if (!resize_resident_checkpoint_buffers(dev)) return false;",
            "resident cnvgz ok reset",
            "if (base < 0) {\n      atomicExch(ok_out, 0);",
            "resident cnvgz candidate kernels",
            "resident terminal stage no-op",
            "restore_host_checkpoint(*ctx, checkpoint_host);",
            "final_code = kMozymeScfCpuBoundary;\n            status->ready = 0;",
            "*status = final_status;\n              final_code = kMozymeScfSuccess;",
            "if (copy_resident_state_to_host(*ctx)) {\n            final_code = kMozymeScfCpuBoundary;",
        ),
        "forbidden_fragments": (
            "const bool poll_stage_status =\n      publish_iteration_outputs || strict_resident_request_enabled();",
            "host_pair_status[2] <= 0",
            "host_status[2] <= 0",
            "atom_flags[atom - 1] = 1",
            "host_errors[kCheckDoubleOccError]",
            "int host_ints[kCheckIntCount] = {}",
            "host_ints[kCheckIntOk] == 1",
            "cudaMemcpy(host_ints, dev.check_ints.ptr, sizeof(host_ints)",
            "cudaMemset(dev.check_ints.ptr",
            "cudaMemset(dev.check_errors.ptr",
            "cudaMemset(dev.eimp_pair_updates.ptr",
            "cudaMemset(dev.density_updates.ptr",
            "cudaMemset(dev.fock_ints.ptr",
            "if (!resize_resident_checkpoint_buffers(dev)) return false;\n  if (!dev.ensure_events()) return false;",
            "int host_fock_status[kFockIntCount] = {0}",
            "dev.fock_ints.upload(host_fock_status, kFockIntCount)",
            "int host_ok_init[kHeleczIntCount] = {1}",
            "dev.helecz_ints.upload(host_ok_init, kHeleczIntCount)",
            "int host_cnvgz_status[kCnvgzIntCount] = {0}",
            "dev.cnvgz_ints.upload(host_cnvgz_status, kCnvgzIntCount)",
        ),
    },
)


def source_marker_contract_payload() -> dict[str, object]:
    return {
        "version": SOURCE_MARKER_CONTRACT_VERSION,
        "required_source_markers": REQUIRED_SOURCE_MARKERS,
    }


def source_marker_contract_sha256() -> str:
    canonical = json.dumps(
        source_marker_contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


SOURCE_MARKER_CONTRACT_SHA256 = "f94e2542b0a0e3e4ea5cb34ddf07513bbcc50f99962585f09a63551fe6dbf69f"


def validate_source_marker_contract_sha256() -> None:
    actual = source_marker_contract_sha256()
    if actual != SOURCE_MARKER_CONTRACT_SHA256:
        raise SystemExit(
            "SOURCE_MARKER_CONTRACT_SHA256 does not match REQUIRED_SOURCE_MARKERS: "
            f"expected {SOURCE_MARKER_CONTRACT_SHA256}, computed {actual}"
        )


EXCLUDED_PARTS = {".git", ".agents", ".codex", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".o", ".a", ".dylib", ".mod", ".smod", ".zip"}
EXCLUDED_ROOTS = ("build", "gpu_test_logs", "colab_test")
EXCLUDED_NAMES = {
    ".DS_Store",
    "compare_grad.py",
    "cpu_fock_debug.patch",
    "modeljuv.B99990166_withH_rep0_step5000.pdb",
    "existing_mopac_references.csv",
    "existing_mopac_references.json",
    "protein_gpu.mop",
    "protein_gpu_2gpu.mop",
    "test_cusolver.F90",
    MANIFEST_NAME,
    PROVENANCE_NAME,
    FEATURES_NAME,
}
GENERATED_BENCHMARK_SUFFIXES = {".arc", ".out", ".log"}


def source_policy_payload() -> dict[str, object]:
    return {
        "schema": SOURCE_ALLOWLIST_POLICY_SCHEMA,
        "root_files": sorted(ALLOWED_SOURCE_ROOT_FILES),
        "root_directories": sorted(ALLOWED_SOURCE_ROOT_DIRS),
    }


def allowed_by_source_policy(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if len(rel.parts) == 1:
        return rel.as_posix() in ALLOWED_SOURCE_ROOT_FILES
    return bool(rel.parts) and rel.parts[0] in ALLOWED_SOURCE_ROOT_DIRS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output zip path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root to package. Default: current directory.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Allow packaging a dirty git worktree. The dirty status is recorded "
            "in the embedded provenance so Colab can distinguish a development "
            "zip from a clean release proof; dirty zips are marked proof-ineligible."
        ),
    )
    parser.add_argument(
        "--snapshot-clean-proof",
        action="store_true",
        help=(
            "Create a temporary clean git snapshot from the current source tree "
            "and package that snapshot. This is useful when the working tree is "
            "dirty but the Colab proof artifact must remain proof-eligible. The "
            "snapshot still uses the same source allowlist and marker contract."
        ),
    )
    parser.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="Keep the temporary source snapshot created by --snapshot-clean-proof.",
    )
    return parser.parse_args()


def excluded(path: Path, root: Path, output: Path) -> bool:
    rel = path.relative_to(root)
    if path == output:
        return True
    if rel.name.endswith(".zip.expected.json"):
        return True
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return True
    if rel.name in EXCLUDED_NAMES:
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    if rel.parts and any(rel.parts[0].startswith(prefix) for prefix in EXCLUDED_ROOTS):
        return True
    if (
        len(rel.parts) >= 3
        and rel.parts[0] == "benchmarks"
        and rel.parts[1] == "publication_inputs"
    ):
        if path.suffix in GENERATED_BENCHMARK_SUFFIXES:
            return True
        if rel.name.endswith("_addh.pdb") or rel.name.endswith(".stdout.txt"):
            return True
    return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_source_files(root: Path, output: Path) -> list[Path]:
    root_resolved = root.resolve()
    candidate_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not excluded(path, root, output)
    ]
    symlink_files = [
        path.relative_to(root).as_posix() for path in candidate_files if path.is_symlink()
    ]
    if symlink_files:
        preview = ", ".join(sorted(symlink_files)[:20])
        raise SystemExit(
            "Refusing to package symlinked source files in the Colab proof zip: "
            f"{preview}"
        )
    escaped_files = []
    for path in candidate_files:
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError:
            escaped_files.append(path.relative_to(root).as_posix())
    if escaped_files:
        preview = ", ".join(sorted(escaped_files)[:20])
        raise SystemExit(
            "Refusing to package files that resolve outside the source root: "
            f"{preview}"
        )
    disallowed_files = [
        path.relative_to(root).as_posix()
        for path in candidate_files
        if not allowed_by_source_policy(path, root)
    ]
    if disallowed_files:
        preview = ", ".join(sorted(disallowed_files)[:20])
        raise SystemExit(
            "Refusing to package files outside the Colab source allowlist policy: "
            f"{preview}"
        )
    return sorted(candidate_files, key=lambda item: item.relative_to(root).as_posix())


def run_checked(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        raise SystemExit(
            f"Command failed in {cwd}: {' '.join(command)}"
            + (f"\n{details}" if details else "")
        )


def create_clean_snapshot(root: Path, output: Path) -> Path:
    snapshot = Path(tempfile.mkdtemp(prefix="mopac-colab-proof-src.")).resolve()
    files = collect_source_files(root, output)
    for path in files:
        rel = path.relative_to(root)
        destination = snapshot / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    run_checked(["git", "init", "-q"], snapshot)
    run_checked(["git", "config", "user.name", "MOPAC Colab Proof"], snapshot)
    run_checked(
        ["git", "config", "user.email", "mopac-colab-proof@example.invalid"],
        snapshot,
    )
    run_checked(["git", "add", "-A"], snapshot)
    run_checked(
        ["git", "commit", "-q", "-m", "Snapshot MOZYME GPU Colab proof source"],
        snapshot,
    )
    return snapshot


def write_expected_source_sidecar(
    output: Path,
    source_zip_sha256: str,
    manifest_sha256: str,
    git_commit: str,
    dirty: bool | None,
    dirty_status_sha256: str | None,
) -> Path:
    sidecar = output.with_name(output.name + ".expected.json")
    proof_eligible = dirty is False
    proof_ineligible_reason = "" if proof_eligible else "dirty_source_zip"
    payload = {
        "schema": "mopac-colab-expected-source-v1",
        "feature_set": FEATURE_SET,
        "full_scf_contract_version": FULL_SCF_CONTRACT_VERSION,
        "source_marker_contract_version": SOURCE_MARKER_CONTRACT_VERSION,
        "source_marker_contract_sha256": SOURCE_MARKER_CONTRACT_SHA256,
        "proof_eligible": proof_eligible,
        "proof_ineligible_reason": proof_ineligible_reason,
        "source_zip_sha256": source_zip_sha256,
        "source_manifest_sha256": manifest_sha256,
        "source_git_commit": git_commit,
        "source_git_dirty": dirty,
        "source_dirty_status_sha256": dirty_status_sha256 or "clean",
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def has_source_fragment(text: str, flat_text: str, fragment: object) -> bool:
    value = str(fragment)
    return value in text or " ".join(value.split()) in flat_text


def normalized_source_fragment(fragment: object) -> str:
    return " ".join(str(fragment).split())


def ordered_source_fragments_missing(flat_text: str, fragments: object) -> str | None:
    start = 0
    for fragment in fragments:
        normalized = normalized_source_fragment(fragment)
        index = flat_text.find(normalized, start)
        if index < 0:
            return normalized
        start = index + len(normalized)
    return None


def cmake_if_block_ranges(text: str, condition: str) -> list[tuple[int, int]]:
    pattern = re.compile(r"^\s*(if|endif)\s*\(([^)]*)\)", re.IGNORECASE | re.MULTILINE)
    stack: list[int | None] = []
    ranges: list[tuple[int, int]] = []
    condition_norm = " ".join(condition.lower().split())
    for match in pattern.finditer(text):
        keyword = match.group(1).lower()
        expr_norm = " ".join(match.group(2).lower().split())
        if keyword == "if":
            stack.append(match.end() if expr_norm == condition_norm else None)
        elif keyword == "endif":
            start = stack.pop() if stack else None
            if start is not None:
                ranges.append((start, match.start()))
    return ranges


def cmake_if_block_range(text: str, condition: str) -> tuple[int, int] | None:
    ranges = cmake_if_block_ranges(text, condition)
    return ranges[0] if ranges else None


def cmake_target_def_matches(text: str, target_name: str, source_name: str) -> list[re.Match[str]]:
    pattern = re.compile(
        rf"(?m)^\s*add_executable\s*\(\s*{re.escape(target_name)}\s+"
        rf"{re.escape(source_name)}\s*\)"
    )
    return list(pattern.finditer(text))


def validate_cmake_gpu_target(
    failures: list[str], cmake_text: str, target_name: str, source_name: str
) -> None:
    target_matches = cmake_target_def_matches(cmake_text, target_name, source_name)
    target_count = len(target_matches)
    target_index = target_matches[0].start() if target_matches else -1
    label = f"standalone_{target_name.replace('-', '_')}_target_before_tests"
    gpu_blocks = cmake_if_block_ranges(cmake_text, "GPU")
    tests_block_match = re.search(r"(?mi)^\s*if\s*\(\s*TESTS\b", cmake_text)
    tests_block_index = tests_block_match.start() if tests_block_match else -1
    if target_index < 0:
        failures.append(f"{label}: missing {target_name} target")
    elif target_count != 1:
        failures.append(f"{label}: expected exactly one {target_name} target definition, found {target_count}")
    elif not any(block_start <= target_index < block_end for block_start, block_end in gpu_blocks):
        failures.append(f"{label}: target is outside the GPU block")
    elif tests_block_index >= 0 and target_index > tests_block_index:
        failures.append(f"{label}: target is gated behind TESTS")


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


def validate_critical_fortran_line_lengths(root: Path) -> list[str]:
    failures: list[str] = []
    for rel in CRITICAL_FEATURE_FILES:
        if not rel.endswith(".F90"):
            continue
        path = root / rel
        if not path.exists():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(),
            start=1,
        ):
            if len(line) > FORTRAN_FREE_FORM_LINE_LIMIT:
                failures.append(
                    f"{rel}:{line_number}: line length {len(line)} exceeds "
                    f"{FORTRAN_FREE_FORM_LINE_LIMIT}"
                )
    return failures


def validate_required_source_markers(root: Path) -> list[str]:
    failures: list[str] = []
    for marker in REQUIRED_SOURCE_MARKERS:
        rel = str(marker["path"])
        path = root / rel
        if not path.exists():
            failures.append(f"{marker['name']}: missing {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        flat_text = " ".join(text.split())
        for fragment in marker.get("fragments", ()):
            if not has_source_fragment(text, flat_text, fragment):
                failures.append(f"{marker['name']}: missing {fragment!r}")
        for fragment in marker.get("forbidden_fragments", ()):
            if has_source_fragment(text, flat_text, fragment):
                failures.append(f"{marker['name']}: forbidden {fragment!r}")
        ordered_missing = ordered_source_fragments_missing(
            flat_text,
            marker.get("ordered_fragments", ()),
        )
        if ordered_missing is not None:
            failures.append(
                f"{marker['name']}: ordered fragment missing or out of order "
                f"{ordered_missing!r}"
            )
    cmake_text = (root / "CMakeLists.txt").read_text(encoding="utf-8", errors="ignore")
    validate_cmake_gpu_target(failures, cmake_text, "mopac-gpu-bench", "tests/gpu_bench.F90")
    validate_cmake_gpu_target(
        failures,
        cmake_text,
        "mopac-gpu-resident-fock-pair-compare",
        "tests/gpu_resident_fock_pair_compare.F90",
    )
    scf_cuda_text = (root / "src/gpu/mozyme_scf_context.cu").read_text(
        encoding="utf-8", errors="ignore"
    )
    if not final_reorth_commit_flag_scoped(scf_cuda_text):
        failures.append(
            "resident_final_reorth_commit_scope: final reorth commit flag must "
            "be scoped inside apply_final_reorth_on_gpu before use"
        )
    resident_fock_text = (root / "src/MOZYME/mozyme_resident_fock.F90").read_text(
        encoding="utf-8", errors="ignore"
    )
    scf_driver_text = (root / "src/MOZYME/mozyme_gpu_scf_driver.F90").read_text(
        encoding="utf-8", errors="ignore"
    )
    if (
        "use mozyme_gpu_int_utils, only: mozyme_c_int_checked" not in scf_driver_text
        or "config%resident_fock_plan_id = mozyme_c_int_checked(resident_fock_plan_id)"
        not in scf_driver_text
    ):
        failures.append(
            "resident_scf_driver_int_import: driver must import "
            "mozyme_c_int_checked before setting resident_fock_plan_id"
        )
    if (
        "import :: max_resident_fallback_basis" not in resident_fock_text
        or "fallback_basis_c(0:max_resident_fallback_basis,0:*)"
        not in resident_fock_text
    ):
        failures.append(
            "resident_fock_interface_import: C interface must import "
            "max_resident_fallback_basis before using it as an array bound"
        )
    failures.extend(validate_critical_fortran_line_lengths(root))
    all_cmake_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted(root.rglob("CMakeLists.txt"))
        if not excluded(path, root, root / DEFAULT_OUTPUT)
    )
    for target_name, source_name, label in (
        (
            "mopac-gpu-bench",
            "tests/gpu_bench.F90",
            "standalone_gpu_benchmark_target_before_tests",
        ),
        (
            "mopac-gpu-resident-fock-pair-compare",
            "tests/gpu_resident_fock_pair_compare.F90",
            "standalone_resident_fock_pair_target_before_tests",
        ),
    ):
        target_count_all = len(
            cmake_target_def_matches(all_cmake_text, target_name, source_name)
        )
        if target_count_all != 1:
            failures.append(
                f"{label}: expected exactly one {target_name} target definition "
                "across all CMakeLists.txt files, "
                f"found {target_count_all}"
            )
    return failures


def run_git(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_dirty_status(root: Path, output: Path) -> tuple[bool | None, list[str], str | None]:
    status = run_git(root, ["status", "--porcelain", "--untracked-files=all"])
    if status is None:
        return None, [], None
    ignored = {MANIFEST_NAME, PROVENANCE_NAME, FEATURES_NAME}
    try:
        output_rel = output.relative_to(root).as_posix()
        ignored.add(output_rel)
        ignored.add(output_rel + ".expected.json")
    except ValueError:
        pass
    dirty_lines = []
    for raw_line in status.splitlines():
        rel = raw_line[3:].strip()
        paths = [item.strip().strip('"') for item in rel.split(" -> ")]
        if not paths or all(path in ignored for path in paths):
            continue
        dirty_lines.append(raw_line)
    dirty_text = "\n".join(dirty_lines) + ("\n" if dirty_lines else "")
    dirty_sha256 = hashlib.sha256(dirty_text.encode("utf-8")).hexdigest()
    return bool(dirty_lines), dirty_lines, dirty_sha256


def is_git_commit(value: str | None) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdefABCDEF" for char in value)


def create_provenance(
    manifest_sha256: str,
    file_count: int,
    feature_set: str,
    contract_version: str,
    git_commit: str | None,
    dirty: bool | None,
    dirty_allowed: bool,
    dirty_entries: list[str],
    dirty_status_sha256: str | None,
) -> str:
    proof_eligible = dirty is False
    proof_ineligible_reason = "" if proof_eligible else "dirty_source_zip"
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "generator": "scripts/create_colab_gpu_zip.py",
        "zip_source_tree_marker": ZIP_SOURCE_TREE_MARKER,
        "feature_set": feature_set,
        "full_scf_contract_version": contract_version,
        "source_marker_contract_version": SOURCE_MARKER_CONTRACT_VERSION,
        "source_marker_contract_sha256": SOURCE_MARKER_CONTRACT_SHA256,
        "proof_eligible": proof_eligible,
        "proof_ineligible_reason": proof_ineligible_reason,
        "manifest": {
            "path": MANIFEST_NAME,
            "sha256": manifest_sha256,
            "file_count": file_count,
        },
        "git": {
            "commit": git_commit,
            "dirty": dirty,
            "dirty_allowed": dirty_allowed,
            "dirty_entries": dirty_entries,
            "dirty_status_sha256": dirty_status_sha256,
        },
    }
    return json.dumps(provenance, indent=2, sort_keys=True) + "\n"


def create_feature_manifest(root: Path, manifest_sha256: str, file_count: int) -> str:
    validate_source_marker_contract_sha256()
    critical_files: dict[str, dict[str, object]] = {}
    missing = []
    for rel in CRITICAL_FEATURE_FILES:
        path = root / rel
        if not path.exists():
            missing.append(rel)
            continue
        critical_files[rel] = {
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
    if missing:
        raise SystemExit(
            "Cannot create Colab feature manifest; missing critical file(s): "
            + ", ".join(missing)
        )
    marker_failures = validate_required_source_markers(root)
    if marker_failures:
        raise SystemExit(
            "Cannot create Colab feature manifest; source marker contract failed: "
            + "; ".join(marker_failures)
        )
    features = {
        "schema": FEATURES_SCHEMA,
        "feature_set": FEATURE_SET,
        "full_scf_contract_version": FULL_SCF_CONTRACT_VERSION,
        "source_marker_contract_version": SOURCE_MARKER_CONTRACT_VERSION,
        "source_marker_contract_sha256": SOURCE_MARKER_CONTRACT_SHA256,
        "zip_source_tree_marker": ZIP_SOURCE_TREE_MARKER,
        "source_allowlist_policy": source_policy_payload(),
        "manifest": {
            "path": MANIFEST_NAME,
            "sha256": manifest_sha256,
            "file_count": file_count,
        },
        "critical_files": critical_files,
        "required_source_markers": REQUIRED_SOURCE_MARKERS,
    }
    return json.dumps(features, indent=2, sort_keys=True) + "\n"


def validate_written_zip(
    output: Path,
    files: list[Path],
    root: Path,
    manifest_text: str,
    feature_manifest_text: str,
    provenance_text: str,
) -> None:
    expected_names = {path.relative_to(root).as_posix() for path in files}
    expected_names.update({MANIFEST_NAME, FEATURES_NAME, PROVENANCE_NAME})
    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise SystemExit(
                "Created zip contains duplicate member(s): "
                + ", ".join(duplicate_names[:20])
            )
        bad_member = archive.testzip()
        if bad_member is not None:
            raise SystemExit(f"Created zip failed integrity check at member: {bad_member}")
        actual_names = set(names)
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            details = []
            if missing:
                details.append("missing=" + ", ".join(missing[:20]))
            if extra:
                details.append("extra=" + ", ".join(extra[:20]))
            raise SystemExit("Created zip file set mismatch: " + "; ".join(details))
        embedded_manifest = archive.read(MANIFEST_NAME).decode("utf-8")
        embedded_features = archive.read(FEATURES_NAME).decode("utf-8")
        embedded_provenance = archive.read(PROVENANCE_NAME).decode("utf-8")
        if embedded_manifest != manifest_text:
            raise SystemExit(f"Created zip embedded {MANIFEST_NAME} does not round-trip")
        if embedded_features != feature_manifest_text:
            raise SystemExit(f"Created zip embedded {FEATURES_NAME} does not round-trip")
        if embedded_provenance != provenance_text:
            raise SystemExit(f"Created zip embedded {PROVENANCE_NAME} does not round-trip")
        validate_notebook_code_cells(
            archive.read("colab/mopac_cublas_gpu_bench_colab.ipynb").decode("utf-8")
        )


def validate_notebook_code_cells(notebook_text: str) -> None:
    try:
        notebook = json.loads(notebook_text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Colab notebook JSON is invalid: {exc}") from exc
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise SystemExit("Colab notebook JSON does not contain a cells list")
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            code = "".join(str(part) for part in source)
        else:
            code = str(source)
        try:
            ast.parse(code, filename=f"colab-cell-{index}")
        except SyntaxError as exc:
            raise SystemExit(
                "Colab notebook code cell failed Python syntax check: "
                f"cell={index} line={exc.lineno} offset={exc.offset} {exc.msg}"
            ) from exc


def package_root(root: Path, output: Path, allow_dirty: bool) -> int:
    git_commit = run_git(root, ["rev-parse", "HEAD"])
    dirty, dirty_entries, dirty_status_sha256 = git_dirty_status(root, output)
    if not is_git_commit(git_commit):
        raise SystemExit(
            "Refusing to package a source tree without a valid git commit. "
            "Run this script from the repository checkout used for the Colab proof."
        )
    if dirty is None:
        raise SystemExit(
            "Refusing to package a source tree without git dirty-state provenance. "
            "Run this script from a valid git worktree."
        )
    if dirty and not allow_dirty:
        raise SystemExit(
            "Refusing to package a dirty git worktree without --allow-dirty. "
            "Commit/stash unrelated changes or pass --allow-dirty for a development Colab zip."
        )

    files = collect_source_files(root, output)
    manifest_text = (
        "\n".join(
            f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files
        )
        + "\n"
    )
    manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    feature_manifest_text = create_feature_manifest(root, manifest_sha256, len(files))
    provenance_text = create_provenance(
        manifest_sha256,
        len(files),
        FEATURE_SET,
        FULL_SCF_CONTRACT_VERSION,
        git_commit,
        dirty,
        allow_dirty,
        dirty_entries,
        dirty_status_sha256,
    )

    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(MANIFEST_NAME, manifest_text)
        archive.writestr(FEATURES_NAME, feature_manifest_text)
        archive.writestr(PROVENANCE_NAME, provenance_text)
    validate_written_zip(
        output,
        files,
        root,
        manifest_text,
        feature_manifest_text,
        provenance_text,
    )

    source_zip_sha256 = sha256(output)
    expected_sidecar = write_expected_source_sidecar(
        output,
        source_zip_sha256,
        manifest_sha256,
        git_commit,
        dirty,
        dirty_status_sha256,
    )

    print(f"Wrote {output}")
    print(f"Wrote {expected_sidecar}")
    print(f"Embedded {MANIFEST_NAME}")
    print(f"Embedded {FEATURES_NAME}")
    print(f"Embedded {PROVENANCE_NAME}")
    print(f"source_zip_sha256={source_zip_sha256}")
    print(f"source_manifest_sha256={manifest_sha256}")
    print(f"source_git_commit={git_commit}")
    print(f"source_git_dirty={str(dirty).lower()}")
    print(f"source_dirty_status_sha256={dirty_status_sha256 or 'clean'}")
    print(f"manifest_sha256={manifest_sha256}")
    print(f"git_commit={git_commit}")
    print(f"git_dirty={dirty}")
    print(f"Packaged {len(files) + 3} files")
    return 0


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if args.snapshot_clean_proof:
        if args.allow_dirty:
            raise SystemExit(
                "--snapshot-clean-proof creates a clean proof snapshot; do not "
                "combine it with --allow-dirty."
            )
        snapshot = create_clean_snapshot(root, output)
        try:
            result = package_root(snapshot, output, allow_dirty=False)
        finally:
            if args.keep_snapshot:
                print(f"Kept clean proof snapshot: {snapshot}")
            else:
                shutil.rmtree(snapshot, ignore_errors=True)
        return result
    return package_root(root, output, allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run end-to-end MOPAC molecule CPU/GPU benchmarks and package results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROFILES: dict[str, list[str]] = {
    "quick": [
        "examples/h2o_gpu_force.mop",
        "examples/benzene.mop",
        "examples/peptide_gg.mop",
    ],
    "standard": [
        "examples/h2o_gpu_force.mop",
        "examples/benzene.mop",
        "examples/ethanol.mop",
        "examples/halogen_disp.mop",
        "examples/peptide_gg.mop",
        "examples/peptide_aaa.mop",
        "examples/mozyme_protein_auto.mop",
    ],
    "protein": [
        "examples/peptide_gg.mop",
        "examples/peptide_aaa.mop",
        "examples/mozyme_protein_auto.mop",
    ],
    "publication_medium": [
        "benchmarks/publication_inputs/mop/protein_crambin_1crn.mop",
        "benchmarks/publication_inputs/mop/protein_ubiquitin_1ubq.mop",
        "benchmarks/publication_inputs/mop/dna_dodecamer_1bna.mop",
    ],
    "publication_large": [
        "benchmarks/publication_inputs/mop/protein_crambin_1crn.mop",
        "benchmarks/publication_inputs/mop/protein_ubiquitin_1ubq.mop",
        "benchmarks/publication_inputs/mop/protein_adenylate_kinase_1ake.mop",
        "benchmarks/publication_inputs/mop/dna_dodecamer_1bna.mop",
        "benchmarks/publication_inputs/mop/rna_trna_1ehz.mop",
    ],
    "publication_material_diagnostic": [
        "benchmarks/publication_inputs/mop/material_graphene_nanoflake_c192h38.mop",
    ],
}

HEAT_RE = re.compile(r"(?:FINAL\s+)?HEAT\s+OF\s+FORMATION\s*=\s*([+\-0-9.EeDd]+)", re.IGNORECASE)
HEAT_AUX_RE = re.compile(r"HEAT_OF_FORMATION:KCAL/MOL\s*=\s*([+\-0-9.EeDd]+)", re.IGNORECASE)
WALL_RE = re.compile(r"WALL-CLOCK TIME\s*=\s*(.+)", re.IGNORECASE)
TOTAL_JOB_RE = re.compile(r"TOTAL JOB TIME:\s*(.+)", re.IGNORECASE)
GEO_DAT_RE = re.compile(r"GEO_DAT\s*=\s*(?:\"([^\"]+)\"|([^\s]+))", re.IGNORECASE)
PDB_LINE_RE = re.compile(r"^\s*([^\s]+\.pdb)\s*$", re.IGNORECASE)
ATOM_LINE_RE = re.compile(r"^\s*([A-Z][a-z]?)\s+[-+0-9.]", re.IGNORECASE)
MOZYME_PROFILE_RE = re.compile(r"\[PROFILE\]\s+MOZYME_GPU=\s*([TF])\s+minblk=\s*(\d+)", re.IGNORECASE)
MOZYME_FOCK_PROFILE_RE = re.compile(r"\[PROFILE\]\s+MOZYME_FOCK_GPU=\s*([TF])", re.IGNORECASE)
MOZYME_CHECK_PROFILE_RE = re.compile(r"\[PROFILE\]\s+MOZYME_CHECK_GPU=\s*([TF])", re.IGNORECASE)
MOZYME_REQUEST_PROFILE_RE = re.compile(
    r"\[PROFILE\]\s+MOZYME_GPU_REQUESTED=\s*([TF])\s+disable_reason=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_FOCK1_BATCH_PROFILE_RE = re.compile(r"\[PROFILE\]\s+MOZYME_FOCK1_BATCH_GPU=\s*([TF])", re.IGNORECASE)
MOZYME_RESIDENT_FOCK_PROFILE_RE = re.compile(r"\[PROFILE\]\s+MOZYME_RESIDENT_FOCK_GPU=\s*([TF])", re.IGNORECASE)
MOZYME_FOCK2_4X1_BATCH_PROFILE_RE = re.compile(
    r"\[PROFILE\]\s+MOZYME_FOCK2_4X1_BATCH_GPU=\s*([TF])",
    re.IGNORECASE,
)
MOZYME_PLAN_BLOCK_RE = re.compile(
    r"\[MOZYME_GPU_PLAN_BEGIN\](.*?)\[MOZYME_GPU_PLAN_END\]",
    re.IGNORECASE | re.DOTALL,
)
MOZYME_PLAN_RE = re.compile(
    r"\[MOZYME GPU plan\]\s+lgpu=\s*([TF])\s+fock_gpu=\s*([TF])\s+"
    r"minblk=\s*(\d+)\s+max_block=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_PLAN_DENSITY_RE = re.compile(
    r"\[MOZYME GPU plan\]\s+density_pairs_meeting_minblk=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_PLAN_AFTER_RE = re.compile(
    r"\[MOZYME GPU plan\]\s+mozyme_gpu_after_plan=\s*([TF])\s+disabled_no_work=\s*([TF])",
    re.IGNORECASE,
)
MOZYME_FOCK_PLAN_RE = re.compile(
    r"\[MOZYME GPU fock plan\]\s+one_center=\s*(\d+)\s+two_center=\s*(\d+)\s+"
    r"skipped=\s*(\d+)\s+d_pairs=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_FOCK_PLAN_SHAPES_RE = re.compile(
    r"\[MOZYME GPU fock plan\]\s+pairs_4x4=\s*(\d+)\s+pairs_4x1=\s*(\d+)\s+"
    r"pairs_9x4=\s*(\d+)\s+pairs_9x9=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_FOCK_PLAN_TASKS_RE = re.compile(
    r"\[MOZYME GPU fock plan\]\s+candidate_gpu_tasks=\s*(\d+)\s+"
    r"production_gpu_tasks=\s*(\d+)\s+one_center_terms=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_FOCK_PLAN_TWO_CENTER_TERMS_RE = re.compile(
    r"\[MOZYME GPU fock plan\]\s+two_center_terms=\s*(\d+)",
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
MOZYME_DENSITY_CALL_RE = re.compile(
    r"\[MOZYME GPU density\]\s+mode=\s*(-?\d+)\s+minblk=\s*(\d+)\s+"
    r"gpu_syrk=\s*(\d+)\s+gpu_gemm=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_DENSITY_SKIP_RE = re.compile(
    r"\[MOZYME GPU density\]\s+skipped_diag=\s*(\d+)\s+skipped_offdiag=\s*(\d+)\s+"
    r"cpu_diag=\s*(\d+)\s+cpu_offdiag=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_DENSITY_MAX_RE = re.compile(
    r"\[MOZYME GPU density\]\s+max_diag=\s*(\d+)\s+max_offdiag_j=\s*(\d+)\s+max_offdiag_k=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_DENSITY_BATCH_RE = re.compile(
    r"\[MOZYME GPU density_batch\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+mode=\s*(-?\d+)\s+blocks=\s*(\d+)\s+terms=\s*(\d+)\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
GPU_DEBUG_RE = re.compile(
    r"hasGPU=\s*([TF])\s+nDevices=\s*(\d+)\s+lgpu=\s*([TF])",
    re.IGNORECASE,
)
GPU_REQUEST_DEBUG_RE = re.compile(
    r"mozyme_gpu_requested=\s*([TF])\s+disable_reason=\s*(\d+)",
    re.IGNORECASE,
)
RESIDENT_DEBUG_RE = re.compile(r"resident_scf=\s*([TF])", re.IGNORECASE)
MOZYME_SCF_RE = re.compile(r"\[MOZYME GPU SCF\]", re.IGNORECASE)
MOZYME_SCF_STATUS_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+status=(success|fallback_cpu|resident_step|strict_abort)\b"
    r"(?:[^\n]*?\breason=([^\s]+))?",
    re.IGNORECASE,
)
MOZYME_SCF_STATUS_ITERATIONS_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+status=(success|fallback_cpu|resident_step|strict_abort)\b[^\n]*?\biterations=\s*(-?\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_STRICT_ABORT_REASON_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+status=strict_abort\s+reason=([^\s]+)",
    re.IGNORECASE,
)
MOZYME_SCF_FINAL_DENSITY_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+final_density=current_resident",
    re.IGNORECASE,
)
MOZYME_SCF_OLDEN_SETUP_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+olden_setup=host_lmo_restore\s+setup_only=1",
    re.IGNORECASE,
)
MOZYME_SCF_CPU_SETUP_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+cpu_setup=mozyme_arrays\s+setup_only=1",
    re.IGNORECASE,
)
MOZYME_SCF_FILLIJ_GPU_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+fillij_gpu=1\s+count=\s*([TF])\s+"
    r"mpack=\s*(-?\d+)\s+n2elec=\s*(-?\d+)\s+ij_dim=\s*(-?\d+)",
    re.IGNORECASE,
)
MOZYME_RESIDENT_FOCK_GPU_COUNT_RE = re.compile(
    r"\[MOZYME GPU resident_fock\]\s+gpu_count=1\s+plan_id=\s*(-?\d+)\s+"
    r"one=\s*(\d+)\s+pair=\s*(\d+)\s+pair4x1=\s*(\d+)\s+"
    r"point=\s*(\d+)\s+full_coverage=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_RESIDENT_FOCK_GPU_PACK_RE = re.compile(
    r"\[MOZYME GPU resident_fock\]\s+gpu_pack=1\s+plan_id=\s*(-?\d+)\s+"
    r"one=\s*(\d+)\s+pair=\s*(\d+)\s+pair4x1=\s*(\d+)\s+"
    r"point=\s*(\d+)\s+full_coverage=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_RESIDENT_FOCK_GPU_POINT_WEIGHTS_RE = re.compile(
    r"\[MOZYME GPU resident_fock\]\s+gpu_point_weights=1\s+point=\s*(\d+)\s+"
    r"max_abs_diff=\s*([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_RESIDENT_FOCK_CPU_PLAN_RE = re.compile(
    r"\[MOZYME GPU resident_fock\]\s+cpu_plan_constructed=1\s+setup_only=1\s+"
    r"plan_id=\s*(-?\d+)\s+one=\s*(\d+)\s+pair=\s*(\d+)\s+"
    r"pair4x1=\s*(\d+)\s+point=\s*(\d+)\s+full_coverage=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_HOST_COMMIT_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+host_commit_only=1(?:\s+phase=([^\s]+))?\s+"
    r"arrays=(\d+)\s+bytes=(\d+)\s+cosmo=(\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_FINAL_PUBLICATION_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+final_publication_done=\s*(\d+)\s+"
    r"arrays=\s*(\d+)\s+bytes=\s*(\d+)\s+cosmo=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_CPU_PINOUT_RE = re.compile(r"\[MOZYME CPU pinout\]", re.IGNORECASE)
MOZYME_SCF_CODE_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+status=(success|fallback_cpu|resident_step|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+code_name=\s*[^\s]+)?\s+ready=\s*(\d+)\s+resident=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_METRICS_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+wall_ms=\s*([+\-0-9.EeDd]+)\s+"
    r"density_max=\s*([+\-0-9.EeDd]+)\s+density_rms=\s*([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_SCF_DIAGG_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+diagg_sumt=\s*([+\-0-9.EeDd]+)\s+diagg_sumb=\s*([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_SCF_STAGE_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+stage_completed=\s*(\d+)\s+"
    r"stage_required=\s*(\d+)\s+stage_missing=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_STAGE_NAMES_RAW_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+stage_completed_names=\s*([^\s]+)\s+"
    r"stage_missing_names=\s*([^\s]+)",
    re.IGNORECASE,
)
MOZYME_SCF_STRICT_PROOF_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+strict_proof\s+strict_resident=\s*(\d+)\s+"
    r"no_fallback_required=\s*(\d+)\s+full_stage_mask=\s*(\d+)\s+"
    r"resident_decision_complete=\s*(\d+)\s+strict_host_syncs=\s*(\d+)\s+"
    r"strict_control_polls=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_RESIDENT_FOCK_PLAN_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+resident_fock_plan_id=\s*(-?\d+)\s+"
    r"resident_fock_plan_full_coverage=\s*(\d+)"
    r"(?:\s+resident_fock_plan_partial_coverage=\s*(\d+)\s+"
    r"resident_fock_plan_required_mask=\s*(\d+)\s+"
    r"resident_fock_plan_covered_mask=\s*(\d+))?",
    re.IGNORECASE,
)
MOZYME_SCF_RESIDENT_DECISION_COMPLETE = 1
MOZYME_SCF_DECISION_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+resident_decision=\s*(-?\d+)",
    re.IGNORECASE,
)
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
MOZYME_SCF_STAGE_CALL_FIELDS = (
    "full_scf_gpu_stage_upload_calls",
    "full_scf_gpu_stage_eimp_calls",
    "full_scf_gpu_stage_diagg_calls",
    "full_scf_gpu_stage_density_calls",
    "full_scf_gpu_stage_fock_calls",
    "full_scf_gpu_stage_cnvgz_calls",
    "full_scf_gpu_stage_helecz_calls",
    "full_scf_gpu_stage_isitsc_calls",
    "full_scf_gpu_stage_addhb_calls",
    "full_scf_gpu_stage_check_calls",
)
MOZYME_SCF_STAGE_MS_FIELDS = (
    "full_scf_gpu_stage_upload_ms",
    "full_scf_gpu_stage_eimp_ms",
    "full_scf_gpu_stage_diagg_ms",
    "full_scf_gpu_stage_density_ms",
    "full_scf_gpu_stage_fock_ms",
    "full_scf_gpu_stage_cnvgz_ms",
    "full_scf_gpu_stage_helecz_ms",
    "full_scf_gpu_stage_isitsc_ms",
    "full_scf_gpu_stage_addhb_ms",
    "full_scf_gpu_stage_check_ms",
)
MOZYME_SCF_STAGE_CALLS_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+resident_stage_calls\s+"
    r"upload=\s*(-?\d+)\s+eimp=\s*(-?\d+)\s+diagg=\s*(-?\d+)\s+"
    r"density=\s*(-?\d+)\s+fock=\s*(-?\d+)\s+cnvgz=\s*(-?\d+)\s+"
    r"helecz=\s*(-?\d+)\s+isitsc=\s*(-?\d+)\s+addhb=\s*(-?\d+)\s+"
    r"check=\s*(-?\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_STAGE_MS_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+resident_stage_ms\s+"
    r"upload=\s*([+\-0-9.EeDd]+)\s+eimp=\s*([+\-0-9.EeDd]+)\s+"
    r"diagg=\s*([+\-0-9.EeDd]+)\s+density=\s*([+\-0-9.EeDd]+)\s+"
    r"fock=\s*([+\-0-9.EeDd]+)\s+cnvgz=\s*([+\-0-9.EeDd]+)\s+"
    r"helecz=\s*([+\-0-9.EeDd]+)\s+isitsc=\s*([+\-0-9.EeDd]+)\s+"
    r"addhb=\s*([+\-0-9.EeDd]+)\s+check=\s*([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_SCF_CNVGZ_ACTIVITY_RE = re.compile(
    r"\[MOZYME GPU SCF\](?=[^\n]*\bcnvgz_active_calls\s*=\s*-?\d+\b)"
    r"(?=[^\n]*\bcnvgz_noop_calls\s*=\s*-?\d+\b)[^\n]*",
    re.IGNORECASE,
)
MOZYME_SCF_ENERGY_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+energy_total=\s*([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_SCF_ISITSC_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+isitsc_okscf=\s*(-?\d+)\s+isitsc_iscf=\s*(-?\d+)\s+"
    r"isitsc_iemin=\s*(-?\d+)\s+isitsc_iemax=\s*(-?\d+)\s+isitsc_scf1=\s*(-?\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_PLS_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+pls_supervisor_calls=\s*(-?\d+)\s+"
    r"pls_restart_required=\s*(-?\d+)\s+pls_history_count=\s*(-?\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_PLS_DELTA_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+pls_ovmax_delta=\s*([+\-0-9.EeDd]+)\s+"
    r"pls_energy_delta=\s*([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_SCF_PLS_RESET_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+pls_restart_reset_device_calls=\s*(-?\d+)\s+"
    r"pls_restart_done=\s*(-?\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_DEVICE_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+device_id=\s*(-?\d+)\s+natoms=\s*(\d+)\s+"
    r"norbs=\s*(\d+)\s+iterations=\s*(-?\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_COSMO_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+cosmo_enabled=\s*(-?\d+)\s+"
    r"cosmo_fock_calls=\s*(-?\d+)\s+cosmo_matvec_calls=\s*(-?\d+)\s+"
    r"cosmo_cg_iterations=\s*(-?\d+)\s+cosmo_nps=\s*(-?\d+)\s+"
    r"cosmo_lm61=\s*(-?\d+)\s+cosmo_pair_count=\s*(-?\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_COMPACT_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+compact_index_route=\s*(-?\d+)\s+use_nijbo=\s*(-?\d+)",
    re.IGNORECASE,
)
MOZYME_SCF_COSMO_ENERGY_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+cosmo_solv_energy=\s*([+\-0-9.EeDd]+)\s+"
    r"cosmo_ediel=\s*([+\-0-9.EeDd]+)\s+"
    r"cosmo_last_residual=\s*([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_SCF_COSMO_CG_RE = re.compile(
    r"\[MOZYME GPU SCF\]\s+cosmo_cg_control_resident=\s*(-?\d+)\s+"
    r"cosmo_cg_converged=\s*(-?\d+)\s+cosmo_cg_breakdown=\s*(-?\d+)\s+"
    r"cosmo_cg_host_syncs=\s*(-?\d+)\s+"
    r"cosmo_cg_target_tol=\s*([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_MAKVEC_SUCCESS_RE = re.compile(
    r"\[MOZYME GPU makvec\]\s+status=success\s+wall_ms=\s*([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_MAKVEC_EXISTING_RE = re.compile(
    r"\[MOZYME GPU makvec\]\s+status=existing_lmo\s+reason=([^\s]+)",
    re.IGNORECASE,
)
MOZYME_MAKVEC_FALLBACK_RE = re.compile(
    r"\[MOZYME GPU makvec\]\s+status=(?:fallback_cpu|strict_abort)(?:\s+(?:reason=([^\s]+)|code=\s*(-?\d+)))?",
    re.IGNORECASE,
)
MOZYME_RELOCAL_RE = re.compile(
    r"\[MOZYME GPU relocal\]\s+status=(success|fallback_cpu|strict_abort)\s+kind=([^\s]+)([^\n]*)",
    re.IGNORECASE,
)
MOZYME_REORTH_RE = re.compile(
    r"\[MOZYME GPU reorth\]\s+status=(success|fallback_cpu|strict_abort)([^\n]*)",
    re.IGNORECASE,
)
MOZYME_SETUPK_RE = re.compile(
    r"\[MOZYME GPU setupk\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+ms=\s*([+\-0-9.EeDd]+))?([^\n]*)?",
    re.IGNORECASE,
)
MOZYME_CNVGZ_RE = re.compile(
    r"\[MOZYME GPU cnvgz\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+pmax=\s*([+\-0-9.EeDd]+)\s+rms=\s*([+\-0-9.EeDd]+)\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
MOZYME_HELECZ_RE = re.compile(
    r"\[MOZYME GPU helecz\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+energy=\s*([+\-0-9.EeDd]+)\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
MOZYME_EIMP_RE = re.compile(
    r"\[MOZYME GPU eimp\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+pairs=\s*(\d+)\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
MOZYME_DIAGG1_AOCC_RE = re.compile(
    r"\[MOZYME GPU diagg1_aocc\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+terms=\s*(\d+)\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
MOZYME_DIAGG1_AVIR_RE = re.compile(
    r"\[MOZYME GPU diagg1_avir\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+terms=\s*(\d+)\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
MOZYME_DIAGG1_CONSTRUCT_RE = re.compile(
    r"\[MOZYME GPU diagg1_construct\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+nij=\s*(\d+)\s+sumt=\s*([+\-0-9.EeDd]+)"
    r"\s+tiny=\s*([+\-0-9.EeDd]+)\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
MOZYME_DIAGG2_ROTPREP_RE = re.compile(
    r"\[MOZYME GPU diagg2_rotprep\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+active=\s*(\d+)\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
MOZYME_DIAGG2_ROTATE_RE = re.compile(
    r"\[MOZYME GPU diagg2_rotate\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+nrej=\s*(\d+)\s+sumb=\s*([+\-0-9.EeDd]+)\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
MOZYME_ISITSC_RE = re.compile(
    r"\[MOZYME GPU isitsc\]\s+(?:status=)?(success|fallback_cpu|strict_abort)\s+code=\s*(-?\d+)"
    r"(?:\s+okscf=\s*([TF])\s+ms=\s*([+\-0-9.EeDd]+))?",
    re.IGNORECASE,
)
MOZYME_TIDY_RE = re.compile(
    r"\[MOZYME GPU tidy\]\s+status=(success|fallback_cpu|strict_abort)\s+"
    r"mode=([^\s]+)([^\n]*)",
    re.IGNORECASE,
)
MOZYME_GPU_HELPER_FATAL_RE = re.compile(
    r"\[MOZYME GPU (?:(?:makvec|relocal|reorth|setupk|density(?:_batch)?|cnvgz|helecz|eimp|"
    r"diagg1_(?:aocc|avir|construct)|diagg2_(?:rotprep|rotate)|isitsc|tidy)\][^\n]*"
    r"(?:\bstatus\s*=\s*)?(?:fallback_cpu|strict_abort)\b|fock[12]\]\s+fallback\b|"
    r"fock1_batch\]\s+fallback\b|fock2_4x1_batch\]\s+fallback\b|"
    r"resident_fock\]\s+fallback_real_pairs\s+total\s*=\s*[1-9]\d*\b|"
    r"resident_fock\][^\n]*\bcpu_point_pairs\s*=\s*[1-9]\d*\b)",
    re.IGNORECASE,
)
MOZYME_FOCK_RE = re.compile(r"\[MOZYME GPU fock([12])\]\s+(attempt|success|fallback)", re.IGNORECASE)
MOZYME_FOCK1_BATCH_RE = re.compile(
    r"\[MOZYME GPU fock1_batch\]\s+(attempt|success|fallback)(?:\s+code=\s*(\d+))?"
    r"\s+tasks=\s*(\d+)\s+pairs=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_FOCK2_4X1_BATCH_RE = re.compile(
    r"\[MOZYME GPU fock2_4x1_batch\]\s+(attempt|success|fallback)(?:\s+code=\s*(\d+))?"
    r"\s+tasks=\s*(\d+)",
    re.IGNORECASE,
)
MOZYME_SPARSE_FOCK_SETUP_RE = re.compile(
    r"\[GPU\]\s+profile\s+mozyme_sparse_fock_setup\s+one=(\d+)\s+pair=(\d+)\s+pair4x1=(\d+)"
    r"(?:\s+point=(\d+)(?:\s+point_dipole=(\d+)\s+point_monopole=(\d+))?)?",
    re.IGNORECASE,
)
MOZYME_SPARSE_FOCK_RUN_RE = re.compile(
    r"\[GPU\]\s+profile\s+mozyme_sparse_fock_run\s+one=(\d+)\s+pair=(\d+)\s+pair4x1=(\d+)"
    r"(?:\s+point=(\d+)(?:\s+point_dipole=(\d+)\s+point_monopole=(\d+))?)?\s+ms=([0-9.eE+-]+)",
    re.IGNORECASE,
)
MOZYME_RESIDENT_FOCK_COVERAGE_RE = re.compile(
    r"\[MOZYME GPU resident_fock\]\s+coverage\s+mode=(\d+)\s+use_nijbo=([TF])\s+"
    r"real_pairs=(\d+)\s+gpu_real_pairs=(\d+)\s+cpu_real_pairs=(\d+)\s+inactive_real_pairs=(\d+)",
    re.IGNORECASE,
)
MOZYME_RESIDENT_FOCK_FALLBACK_RE = re.compile(
    r"\[MOZYME GPU resident_fock\]\s+fallback_real_pairs\s+total=(\d+)\s+"
    r"reason=(?:unsupported_basis_limit|d_orbital)\s+count=(\d+)\s+"
    r"(?:reason=unsupported_direct_basis\s+count=(\d+)\s+)?"
    r"reason=(?:unsupported_other|unsupported_sp_mix)\s+count=(\d+)",
    re.IGNORECASE,
)
MOZYME_RESIDENT_FOCK_POINT_COVERAGE_RE = re.compile(
    r"\[MOZYME GPU resident_fock\]\s+point_coverage\b([^\n]*)",
    re.IGNORECASE,
)
MOZYME_RESIDENT_FOCK_CPU_POINT_PAIRS_RE = re.compile(
    r"\[MOZYME GPU resident_fock\][^\n]*\bcpu_point_pairs\s*=\s*(\d+)\b",
    re.IGNORECASE,
)
MARKER_FIELD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s]+)")
MARKER_REASON_COUNT_RE = re.compile(r"\breason=([A-Za-z0-9_]+)\s+count=(\d+)")
MOZYME_SECTION_RE = re.compile(
    r"\[PROFILE\]\s+MOZYME_SECTION\s+name=(?:\"([^\"]+)\"|([^\s]+))\s+calls=(\d+)\s+ms=([+\-0-9.EeDd]+)",
    re.IGNORECASE,
)
MOZYME_CPU_MUTATING_SECTION_NAMES = {
    "iter_makvec",
    "iter_tidy_occ",
    "iter_tidy_virt",
    "iter_olden_load",
    "iter_density_olden",
    "iter_reloc_occ",
    "iter_reloc_virt",
    "iter_setupk",
    "iter_check_occ",
    "iter_check_virt",
    "iter_density_initial",
    "iter_density_remove",
    "iter_buildf_initial",
    "iter_helecz_initial",
    "iter_buildf_partial",
    "iter_density_iter",
    "iter_diagg",
    "iter_eimp",
    "iter_addhb",
    "iter_cnvgz",
    "iter_isitsc",
    "iter_helecz_iter",
    "iter_pls_faulty",
    "iter_buildf_iter_partial",
    "iter_buildf_iter_full",
    "iter_density_final_partial",
    "iter_density_final_full",
    "iter_reorth",
    "iter_density_reorth",
    "iter_buildf_reorth",
    "iter_helecz_reorth",
}
MOZYME_FINAL_REORTH_SECTION_NAMES = {
    "iter_reorth",
    "iter_density_reorth",
    "iter_buildf_reorth",
    "iter_helecz_reorth",
}
MOZYME_STRICT_PROOF_ALLOWED_SECTION_NAMES = {
    "iter_resident_scf_boundary",
    "iter_density_final_resident",
}
MOPAC_GPU_READINESS_CONTRACT_VERSION = "resident-scf-strict-final-resident-reorth-tidy-selmos-pls-reset-cosmo-direct-resident-cg-point-kind-cnvgz-active-or-noop-cpu-compare-explicit-proof-v58"
MOPAC_GPU_FEATURE_SET = "mozyme-full-scf-gpu-makvec-relocal-final-resident-reorth-tidy-selmos-pls-reset-cosmo-direct-point-kind-cnvgz-active-or-noop-cpu-compare-explicit-proof-v38-20260702"
MOPAC_GPU_BENCHMARK_SCOPE = "molecule_mozyme_full_scf_gpu"
MOPAC_GPU_PUBLICATION_CLAIM = (
    "Complete MOZYME full-SCF GPU compute is claimed only for rows that pass "
    "the strict resident-SCF readiness contract with no parsed CPU fallback "
    "and, in the Colab proof path, a same-input CPU companion energy comparison; "
    "final/status host commits remain allowed for MOPAC state publication."
)
DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_ABS_TOL = 5.0e-3
DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_REL_TOL = 1.0e-5
DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_PER_ATOM_TOL = 5.0e-4
FULL_SCF_READINESS_CPU_COMPARE_BINDING_FIELDS = (
    "molecule",
    "input",
    "full_scf_gpu_status",
    "full_scf_gpu_code",
    "full_scf_gpu_backend_ready",
    "full_scf_gpu_resident",
    "full_scf_gpu_scf_success_calls",
    "full_scf_gpu_final_iterations",
    "full_scf_gpu_stage_completed",
    "full_scf_gpu_stage_required",
    "full_scf_gpu_stage_missing",
    "full_scf_gpu_stage_completed_names_raw",
    "full_scf_gpu_stage_missing_names_raw",
    "full_scf_gpu_strict_resident",
    "full_scf_gpu_no_fallback_required",
    "full_scf_gpu_full_stage_mask",
    "full_scf_gpu_resident_decision_complete",
    "full_scf_gpu_strict_resident_host_syncs",
    "full_scf_gpu_strict_resident_control_polls",
    "full_scf_gpu_resident_fock_plan_id",
    "full_scf_gpu_resident_fock_plan_full_coverage",
    "full_scf_gpu_resident_fock_plan_partial_coverage",
    "full_scf_gpu_resident_fock_plan_required_mask",
    "full_scf_gpu_resident_fock_plan_covered_mask",
    "full_scf_gpu_resident_decision",
    "full_scf_gpu_final_density_resident",
    "full_scf_gpu_final_publication_done",
    "full_scf_gpu_final_publication_arrays",
    "full_scf_gpu_final_publication_bytes",
    "full_scf_gpu_final_publication_cosmo",
    "full_scf_gpu_host_commit_only_calls",
    "full_scf_gpu_fillij_gpu_count_calls",
    "full_scf_gpu_fillij_gpu_fill_calls",
    "full_scf_gpu_resident_fock_gpu_count_calls",
    "full_scf_gpu_resident_fock_gpu_count_plan_id",
    "full_scf_gpu_resident_fock_gpu_count_one",
    "full_scf_gpu_resident_fock_gpu_count_pair",
    "full_scf_gpu_resident_fock_gpu_count_pair4x1",
    "full_scf_gpu_resident_fock_gpu_count_point",
    "full_scf_gpu_resident_fock_gpu_count_full_coverage",
    "full_scf_gpu_resident_fock_gpu_pack_calls",
    "full_scf_gpu_resident_fock_gpu_pack_plan_id",
    "full_scf_gpu_resident_fock_gpu_pack_one",
    "full_scf_gpu_resident_fock_gpu_pack_pair",
    "full_scf_gpu_resident_fock_gpu_pack_pair4x1",
    "full_scf_gpu_resident_fock_gpu_pack_point",
    "full_scf_gpu_resident_fock_gpu_pack_full_coverage",
    "full_scf_gpu_resident_fock_gpu_point_weight_calls",
    "full_scf_gpu_resident_fock_gpu_point_weight_point",
    "full_scf_gpu_resident_fock_gpu_point_weight_max_abs_diff",
    "full_scf_gpu_cnvgz_active_calls",
    "full_scf_gpu_cnvgz_noop_calls",
    "full_scf_gpu_cpu_mozyme_setup_only_calls",
    "full_scf_gpu_cpu_resident_fock_plan_setup_calls",
    "mozyme_tidy_gpu_success_calls",
    "mozyme_tidy_gpu_occupied_success_calls",
    "mozyme_tidy_gpu_virtual_success_calls",
    "mozyme_tidy_gpu_selmos_success_calls",
    "mozyme_tidy_gpu_fallback_calls",
    "mozyme_plan_direct_mode",
    "mozyme_fock_resident_supported_one_center",
    "mozyme_fock_resident_unsupported_one_center",
    "mozyme_fock_resident_full_coverage_planned",
    "mozyme_fock_resident_executable_tasks",
    "mozyme_fock_resident_direct_unsupported_pairs",
    "mozyme_fock_resident_direct_unsupported_point_pairs",
    "mozyme_resident_fock_direct_basis_fallback_pairs",
    "mozyme_resident_fock_direct_basis_point_fallback_pairs",
    "mopac_executable_sha256",
    "cmake_gpu_bool",
    "cmake_cuda_architectures",
    "heat_kcal_mol",
    "source_zip_sha256",
    "source_manifest_sha256",
    "source_provenance_marker_contract_version",
    "source_features_marker_contract_version",
    "source_provenance_marker_contract_sha256",
    "source_features_marker_contract_sha256",
)
SOURCE_MANIFEST_NAME = "MOPAC_COLAB_SOURCE_MANIFEST.sha256"
SOURCE_FEATURES_NAME = "MOPAC_COLAB_FEATURES.json"
SOURCE_PROVENANCE_NAME = "MOPAC_COLAB_SOURCE_PROVENANCE.json"
SOURCE_FEATURES_SCHEMA = "mopac-colab-feature-manifest-v1"
SOURCE_PROVENANCE_SCHEMA = "mopac-colab-source-provenance-v1"
ZIP_SOURCE_TREE_MARKER = "mopac-colab-gpu-proof-source-v1"
SOURCE_MARKER_CONTRACT_VERSION = "mopac-colab-source-markers-explicit-proof-v83"
SOURCE_MARKER_CONTRACT_SHA256 = "17aa11cd4550db1b4d2bbf2f4b15bfc13b9c6e06faad28f6609ee42c7ab69240"
REQUIRED_CRITICAL_SOURCE_FILES = (
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

MOZYME_GPU_REASON_TEXT = {
    0: "none",
    1: "not_requested",
    2: "no_device",
    3: "no_production_work",
    4: "device_policy",
}

MOZYME_SCF_STAGE_BITS = (
    (1, "upload"),
    (2, "eimp"),
    (4, "diagg"),
    (8, "density"),
    (16, "fock"),
    (32, "cnvgz"),
    (64, "helecz"),
    (128, "isitsc"),
    (256, "addhb"),
    (512, "check"),
)
MOZYME_SCF_STAGE_FULL = sum(bit for bit, _name in MOZYME_SCF_STAGE_BITS)

FULL_SCF_GPU_FALLBACK_KEYS = (
    "full_scf_gpu_cpu_boundary_calls",
    "full_scf_gpu_pls_restart_required_calls",
    "mozyme_gpu_helper_fatal_marker_count",
    "mozyme_makvec_gpu_fallback_calls",
    "mozyme_relocal_gpu_fallback_calls",
    "mozyme_reorth_gpu_fallback_calls",
    "mozyme_tidy_gpu_fallback_calls",
    "mozyme_setupk_gpu_fallback_calls",
    "mozyme_setupk_gpu_initial_setup_fallback_calls",
    "density_cpu_diag_blocks",
    "density_cpu_offdiag_blocks",
    "density_batch_gpu_fallback_calls",
    "mozyme_eimp_gpu_fallback_calls",
    "mozyme_diagg1_aocc_gpu_fallback_calls",
    "mozyme_diagg1_avir_gpu_fallback_calls",
    "mozyme_diagg1_construct_gpu_fallback_calls",
    "mozyme_diagg2_rotprep_gpu_fallback_calls",
    "mozyme_diagg2_rotate_gpu_fallback_calls",
    "mozyme_isitsc_gpu_fallback_calls",
    "mozyme_cnvgz_gpu_fallback_calls",
    "mozyme_helecz_gpu_fallback_calls",
    "mozyme_fock1_batch_gpu_fallback_calls",
    "mozyme_fock1_batch_gpu_fallback_tasks",
    "mozyme_fock1_batch_gpu_fallback_pairs",
    "mozyme_fock2_4x1_batch_gpu_fallback_calls",
    "mozyme_fock2_4x1_batch_gpu_fallback_tasks",
    "mozyme_fock1_gpu_fallback_seen",
    "mozyme_fock2_gpu_fallback_seen",
)
RESIDENT_FOCK_FALLBACK_KEYS = (
    "mozyme_resident_fock_cpu_real_pairs",
    "mozyme_resident_fock_fallback_pairs",
    "mozyme_resident_fock_basis_limit_fallback_pairs",
    "mozyme_resident_fock_direct_basis_fallback_pairs",
    "mozyme_resident_fock_other_fallback_pairs",
    "mozyme_resident_fock_direct_basis_point_fallback_pairs",
)
STRICT_HOST_ROUTE_MARKERS = (
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
    "strict_resident_fock_gpu_count_setup_failed",
    "strict_resident_fock_gpu_count_mismatch",
    "strict_resident_fock_gpu_pack_failed",
    "strict_resident_fock_gpu_pack_mismatch",
    "strict_resident_fock_gpu_point_weights_failed",
    "strict_resident_fock_gpu_point_weights_mismatch",
    "strict_fillij_gpu_failed",
    "strict_fillij_gpu_unavailable",
    "strict_fillij_nijbo_missing",
    "strict_add_more_interactions_cpu_fallback",
    "strict_addhb_cpu_fallback",
    "strict_check_cpu_fallback",
    "strict_check_gpu_host_fallback",
    "strict_density_direct_host_rebuild",
    "strict_cpu_makvec_direct",
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
    "strict_run_mopac_olden_host_restore",
    "strict_writmo_reloc_host_output",
    "strict_writmo_vec_host_output",
    "strict_writmo_mecip_host_density",
    "strict_writmo_pm7ts_host_compfg",
    "strict_writmo_deriv_host_output",
    "strict_writmo_fock_host_output",
    "strict_writmo_dens_host_output",
    "strict_writmo_pops_host_output",
    "strict_writmo_pi_host_output",
    "strict_writmo_spin_host_output",
    "strict_writmo_bonds_host_output",
    "strict_writmo_local_host_output",
    "strict_writmo_1ele_host_output",
    "strict_writmo_enpart_host_output",
    "strict_writmo_denout_host_output",
    "strict_writmo_mullik_host_output",
)


@dataclass(frozen=True)
class Mode:
    name: str
    env_set: dict[str, str]
    env_unset: tuple[str, ...]


MODES = [
    Mode("CPU", {"MOPAC_NOGPU": "1", "MOZYME_GPU_OFF": "1"}, ("MOPAC_FORCEGPU", "MOZYME_GPU_FORCE")),
    Mode("GPU", {"MOPAC_FORCEGPU": "1", "MOZYME_GPU_FORCE": "1"}, ("MOPAC_NOGPU", "MOZYME_GPU_OFF")),
]
FULL_SCF_PROBE_FORCED_KEYWORDS = ("MOZYME_MINBLK=1", "RE-LOCAL=1", "REORTH")
FULL_SCF_PROBE_FORCED_ENV = (
    "MOPAC_FORCEGPU=1",
    "MOZYME_GPU_FORCE=1",
    "MOPAC_GPU_PROFILE=1",
    "MOPAC_GPU_DEBUG=1",
    "MOPAC_MOZYME_SECTION_PROFILE=1",
    "MOPAC_MOZYME_SCF_GPU=1",
    "MOPAC_MOZYME_FULL_SCF_GPU=1",
    "MOPAC_MOZYME_SCF_EXPERIMENTAL=1",
    "MOPAC_MOZYME_RESIDENT_SCF=1",
    "MOPAC_MOZYME_RESIDENT_FOCK_GPU=1",
    "MOPAC_MOZYME_MAKVEC_GPU=1",
    "MOPAC_MOZYME_SCF_STRICT_RESIDENT=1",
    "MOPAC_MOZYME_GPU_STRICT=1",
    "MOPAC_MOZYME_SCF_EARLY_PROBE=0",
    "MOPAC_MOZYME_SCF_FORCE_FINAL_REORTH=1",
    "MOPAC_MOZYME_CNVGZ_GPU=1",
    "MOPAC_MOZYME_HELECZ_GPU=1",
    "MOPAC_MOZYME_EIMP_GPU=1",
    "MOPAC_MOZYME_DENSITY_BATCH_GPU=1",
    "MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU=1",
    "MOPAC_MOZYME_DIAGG1_AOCC_GPU=1",
    "MOPAC_MOZYME_DIAGG1_AVIR_GPU=1",
    "MOPAC_MOZYME_DIAGG2_ROTATE_GPU=1",
    "MOPAC_MOZYME_DIAGG2_ROTPREP_GPU=1",
    "MOPAC_MOZYME_ISITSC_GPU=1",
    "MOPAC_MOZYME_RELOCAL_GPU=1",
    "MOPAC_MOZYME_REORTH_GPU=1",
    "MOPAC_MOZYME_TIDY_GPU=1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mopac", nargs="?", default="/content/mopac_colab_build/mopac", help="MOPAC executable.")
    parser.add_argument("inputs", nargs="*", help="Input .mop files. Defaults come from --profile.")
    parser.add_argument("--profile", choices=sorted(DEFAULT_PROFILES), default="standard")
    parser.add_argument("--out-dir", default="molecule_benchmark_report")
    parser.add_argument("--bundle-zip", default="", help="Default: <out-dir>_publication_data.zip.")
    parser.add_argument("--no-bundle-zip", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0, help="Seconds per molecule/mode run.")
    parser.add_argument(
        "--no-gpu-preflight",
        action="store_true",
        help="Skip the single-molecule GPU-use check before the full benchmark.",
    )
    parser.add_argument(
        "--preflight-input",
        default="",
        help="Input .mop file for the GPU preflight. Default: first benchmark input.",
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=600.0,
        help="Seconds allowed for the GPU preflight molecule.",
    )
    parser.add_argument(
        "--preflight-min-speedup",
        type=float,
        default=1.0,
        help=(
            "Run the preflight molecule once on CPU and require CPU_wall/GPU_wall to be at least this value. "
            "Set to 0 to skip the speed gate."
        ),
    )
    parser.add_argument("--keep-run-dirs", action="store_true", help="Keep per-run working directories in output.")
    parser.add_argument("--verbose-gpu", action="store_true", help="Enable MOPAC_GPU_VERBOSE=1 for GPU runs.")
    parser.add_argument(
        "--gpu-profile",
        action="store_true",
        help="Enable MOPAC_GPU_PROFILE=2 for GPU runs.",
    )
    parser.add_argument(
        "--mozyme-section-profile",
        action="store_true",
        help=(
            "Enable MOPAC_MOZYME_SECTION_PROFILE=1 so profiling builds emit MOZYME section timing markers. "
            "The executable also enables section timers from MOPAC_GPU_PROFILE or MOPAC_MOZYME_PROFILE."
        ),
    )
    parser.add_argument(
        "--require-full-scf-gpu",
        action="store_true",
        help=(
            "Require the opt-in complete MOZYME resident-SCF GPU backend before running the publication "
            "molecule benchmark. The script runs a streaming readiness probe with "
            "MOPAC_MOZYME_SCF_EXPERIMENTAL=1 and aborts on controlled CPU fallback or missing markers. "
            "Requires --full-scf-readiness-cpu-compare."
        ),
    )
    parser.add_argument(
        "--full-scf-readiness-only",
        action="store_true",
        help=(
            "With --require-full-scf-gpu, run only the strict full resident-SCF GPU readiness probe "
            "and exit before GPU preflight or molecule benchmarks."
        ),
    )
    parser.add_argument(
        "--full-scf-readiness-cpu-compare",
        action="store_true",
        help=(
            "After the strict full resident-SCF GPU readiness probe, run the same forced probe input "
            "once with GPU disabled and abort if heat of formation is outside the configured "
            "CPU/GPU tolerances."
        ),
    )
    parser.add_argument(
        "--require-direct-cosmo-gpu",
        action="store_true",
        help=(
            "For --require-full-scf-gpu, require the readiness input to be a direct "
            "COSMO proof input with EPS=78.4. This is required by the v58 COSMO-direct "
            "point-kind CPU-compare proof path."
        ),
    )
    parser.add_argument(
        "--full-scf-probe-input",
        default="",
        help="Input .mop file for --require-full-scf-gpu. Default: --preflight-input or first benchmark input.",
    )
    parser.add_argument(
        "--full-scf-probe-timeout",
        type=float,
        default=0.0,
        help="Seconds allowed for --require-full-scf-gpu. Default: --preflight-timeout.",
    )
    parser.add_argument(
        "--energy-abs-tol",
        type=float,
        default=DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_ABS_TOL,
        help="Maximum accepted absolute CPU/GPU heat-of-formation difference in kcal/mol.",
    )
    parser.add_argument(
        "--energy-rel-tol",
        type=float,
        default=DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_REL_TOL,
        help="Maximum accepted relative CPU/GPU heat-of-formation difference.",
    )
    parser.add_argument(
        "--energy-per-atom-tol",
        type=float,
        default=DEFAULT_FULL_SCF_READINESS_CPU_COMPARE_PER_ATOM_TOL,
        help="Maximum accepted absolute heat-of-formation difference per atom in kcal/mol/atom.",
    )
    parser.add_argument(
        "--legacy-reference-csv",
        default="benchmarks/existing_mopac_references.csv",
        help="Existing MOPAC output reference CSV to copy into the report bundle when present.",
    )
    parser.add_argument(
        "--legacy-reference-json",
        default="benchmarks/existing_mopac_references.json",
        help="Existing MOPAC output reference JSON to copy into the report bundle when present.",
    )
    return parser.parse_args()


def build_mopac_env(
    mode: Mode,
    verbose_gpu: bool,
    gpu_profile: bool,
    mozyme_section_profile: bool,
    gpu_preflight_stop: bool = False,
    full_scf_gpu: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    env["MOPAC_DETERMINISTIC"] = "1"
    for key in mode.env_unset:
        env.pop(key, None)
    env.update(mode.env_set)
    if mozyme_section_profile:
        env["MOPAC_MOZYME_SECTION_PROFILE"] = "1"
    if mode.name == "GPU" and verbose_gpu:
        env["MOPAC_GPU_VERBOSE"] = "1"
    if mode.name == "GPU" and gpu_profile:
        env["MOPAC_GPU_PROFILE"] = "2"
        env["MOPAC_GPU_DEBUG"] = "1"
    if mode.name == "GPU":
        env["MOPAC_MOZYME_CNVGZ_GPU"] = "1"
        env["MOPAC_MOZYME_HELECZ_GPU"] = "1"
        env["MOPAC_MOZYME_EIMP_GPU"] = "1"
        env["MOPAC_MOZYME_DENSITY_BATCH_GPU"] = "1"
        env["MOPAC_MOZYME_DIAGG1_CONSTRUCT_GPU"] = "1"
        env["MOPAC_MOZYME_DIAGG1_AOCC_GPU"] = "1"
        env["MOPAC_MOZYME_DIAGG1_AVIR_GPU"] = "1"
        env["MOPAC_MOZYME_DIAGG2_ROTATE_GPU"] = "1"
        env["MOPAC_MOZYME_DIAGG2_ROTPREP_GPU"] = "1"
        env["MOPAC_MOZYME_ISITSC_GPU"] = "1"
        env["MOPAC_MOZYME_RELOCAL_GPU"] = "1"
        env["MOPAC_MOZYME_REORTH_GPU"] = "1"
    if mode.name == "GPU" and gpu_preflight_stop:
        env["MOPAC_MOZYME_GPU_PREFLIGHT_STOP"] = "1"
    if mode.name == "GPU" and full_scf_gpu:
        env["MOPAC_MOZYME_SCF_EXPERIMENTAL"] = "1"
        env["MOPAC_MOZYME_SCF_GPU"] = "1"
        env["MOPAC_MOZYME_FULL_SCF_GPU"] = "1"
        env["MOPAC_MOZYME_RESIDENT_SCF"] = "1"
        env["MOPAC_MOZYME_RESIDENT_FOCK_GPU"] = "1"
        env["MOPAC_MOZYME_MAKVEC_GPU"] = "1"
        env["MOPAC_MOZYME_SCF_STRICT_RESIDENT"] = "1"
        env["MOPAC_MOZYME_GPU_STRICT"] = "1"
        env["MOPAC_MOZYME_SCF_EARLY_PROBE"] = "0"
        env["MOPAC_MOZYME_SCF_FORCE_FINAL_REORTH"] = "1"
        env.setdefault("MOPAC_GPU_PROFILE", "1")
        env.setdefault("MOPAC_GPU_DEBUG", "1")
    return env


def run_mopac(
    mopac: Path,
    input_path: Path,
    mode: Mode,
    repeat: int,
    out_dir: Path,
    timeout: float,
    keep_run_dirs: bool,
    verbose_gpu: bool,
    gpu_profile: bool,
    mozyme_section_profile: bool = False,
    gpu_preflight_stop: bool = False,
    full_scf_gpu: bool = False,
    proof_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_root = out_dir / "runs"
    run_dir = run_root / input_path.stem / mode.name.lower() / f"rep{repeat}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    staged_input = stage_input(input_path, run_dir)

    env = build_mopac_env(
        mode,
        verbose_gpu=verbose_gpu,
        gpu_profile=gpu_profile,
        mozyme_section_profile=mozyme_section_profile,
        gpu_preflight_stop=gpu_preflight_stop,
        full_scf_gpu=full_scf_gpu,
    )

    cmd = [str(mopac), staged_input.name]
    print(f"[{mode.name}] {input_path.name} repeat {repeat}: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        stdout = proc.stdout
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - t0
        stdout = ensure_text(exc.stdout or "") + f"\nTIMEOUT after {timeout:.1f} seconds\n"
        returncode = 124
        timed_out = True

    log_path = out_dir / "logs" / f"{input_path.stem}.{mode.name.lower()}.rep{repeat}.stdout.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout, encoding="utf-8", errors="ignore")

    combined_text = stdout + "\n" + read_outputs(run_dir)
    archived_outputs = archive_output_files(run_dir, log_path.parent, f"{input_path.stem}.{mode.name.lower()}.rep{repeat}")
    gpu_error_markers = gpu_error_markers_in_text(combined_text)
    heat = parse_heat(combined_text)
    reported_s = parse_reported_time(combined_text)
    mozyme_diag = parse_mozyme_gpu_diagnostics(combined_text)
    if mode.name == "GPU" and full_scf_gpu:
        mozyme_diag["full_scf_gpu_requested"] = 1
        mozyme_diag["full_scf_probe_decision"] = classify_full_scf_probe_decision(combined_text)
        if mozyme_diag.get("full_scf_gpu_status") == "not_requested":
            mozyme_diag["full_scf_gpu_status"] = "not_executed"
    mozyme_section_times = parse_mozyme_section_times(combined_text)
    if mode.name == "GPU" and full_scf_gpu:
        (
            cpu_mutating_sections,
            cpu_mutating_call_count,
            cpu_mutating_ms,
        ) = mozyme_disallowed_strict_section_stats(mozyme_section_times, combined_text)
    else:
        (
            cpu_mutating_sections,
            cpu_mutating_call_count,
            cpu_mutating_ms,
        ) = mozyme_cpu_mutating_section_stats(mozyme_section_times)
    normal_marker = "JOB ENDED NORMALLY" in combined_text or "== MOPAC DONE ==" in combined_text
    # MOPAC can return code 0 for preparation-only or diagnostic exits.  For this
    # benchmark a run is only usable when a molecular heat of formation was printed.
    normal_end = normal_marker and heat is not None

    archive_dir = ""
    if keep_run_dirs:
        archive_dir = str(run_dir)
    else:
        shutil.rmtree(run_dir, ignore_errors=True)

    row = {
        "molecule": input_path.stem,
        "input": str(input_path),
        "input_eps": parse_input_eps(input_path) or "",
        "mode": mode.name,
        "repeat": repeat,
        "returncode": returncode,
        "timed_out": timed_out,
        "normal_end": normal_end,
        "wall_s": elapsed,
        "reported_s": reported_s if reported_s is not None else "",
        "heat_kcal_mol": heat if heat is not None else "",
        "atoms": count_atoms(input_path),
        **mozyme_diag,
        "gpu_error_markers": ";".join(gpu_error_markers),
        "gpu_error_marker_count": len(gpu_error_markers),
        "mozyme_section_times": mozyme_section_times,
        "full_scf_gpu_cpu_mutating_sections": ";".join(cpu_mutating_sections),
        "full_scf_gpu_cpu_mutating_section_count": len(cpu_mutating_sections),
        "full_scf_gpu_cpu_mutating_call_count": cpu_mutating_call_count,
        "full_scf_gpu_cpu_mutating_ms": cpu_mutating_ms,
        "log_path": str(log_path),
        "output_files": ";".join(str(path) for path in archived_outputs),
        "run_dir": archive_dir,
    }
    if mode.name == "GPU" and full_scf_gpu and proof_identity is not None:
        row.update(proof_identity)
    finalize_full_scf_gpu_contract(row)
    return row


def stage_input(input_path: Path, run_dir: Path) -> Path:
    staged = run_dir / input_path.name
    shutil.copy2(input_path, staged)
    for ref in referenced_files(input_path):
        src = input_path.parent / ref
        if src.exists():
            shutil.copy2(src, run_dir / src.name)
    return staged


def force_full_scf_probe_keywords(input_path: Path) -> str:
    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return ""
    keyword_index = next((idx for idx, line in enumerate(lines) if line.strip()), 0)
    keyword_line = lines[keyword_index]
    keyword_line = re.sub(r"(?i)(^|\s)MOZYME_MINBLK\s*=\s*[^\s]+", " ", keyword_line)
    keyword_line = re.sub(r"(?i)(^|\s)RE-LOC(?:AL)?(?:\s*=\s*[^\s]+)?", " ", keyword_line)
    keyword_line = re.sub(r"(?i)(^|\s)REORTH(?=\s|$)", " ", keyword_line)
    lines[keyword_index] = keyword_line.rstrip() + " " + " ".join(FULL_SCF_PROBE_FORCED_KEYWORDS)
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ";".join(FULL_SCF_PROBE_FORCED_KEYWORDS)


def parse_input_eps(input_path: Path) -> float | None:
    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    keyword_line = next((line for line in lines if line.strip()), "")
    match = re.search(r"(?i)(?:^|\s)EPS\s*=\s*([+\-0-9.EeDd]+)", keyword_line)
    return parse_float(match.group(1)) if match else None


def ensure_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def referenced_files(input_path: Path) -> list[str]:
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    refs: list[str] = []
    for quoted, bare in GEO_DAT_RE.findall(text):
        refs.append(quoted or bare)
    for line in text.splitlines():
        match = PDB_LINE_RE.match(line)
        if match:
            refs.append(match.group(1))
    return sorted(set(refs))


def read_outputs(run_dir: Path) -> str:
    chunks: list[str] = []
    for pattern in ("*.out", "*.arc", "*.aux"):
        for path in sorted(run_dir.glob(pattern)):
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    return "\n".join(chunks)


def text_tail(text: str, max_lines: int = 180) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def archive_output_files(run_dir: Path, log_dir: Path, prefix: str) -> list[Path]:
    archived: list[Path] = []
    for pattern in ("*.out", "*.arc", "*.aux", "*.res", "*.den", "*.DEN"):
        for path in sorted(run_dir.glob(pattern)):
            target = log_dir / f"{prefix}{path.suffix}"
            try:
                shutil.copy2(path, target)
                archived.append(target)
            except OSError:
                pass
    return archived


def parse_heat(text: str) -> float | None:
    for regex in (HEAT_RE, HEAT_AUX_RE):
        matches = regex.findall(text)
        if matches:
            return parse_float(matches[-1])
    return None


def parse_reported_time(text: str) -> float | None:
    matches: list[str] = []
    for line in text.splitlines():
        wall = WALL_RE.search(line)
        if wall:
            matches.append(wall.group(1))
        total = TOTAL_JOB_RE.search(line)
        if total:
            matches.append(total.group(1))
    if not matches:
        return None
    return parse_time_expression(matches[-1])


def parse_mozyme_plan_block(text: str) -> dict[str, str]:
    blocks = MOZYME_PLAN_BLOCK_RE.findall(text)
    if not blocks:
        return {}
    result: dict[str, str] = {}
    for raw_line in blocks[-1].splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip().lower()] = value.strip()
    return result


def plan_bool(plan: dict[str, str], key: str, fallback: str = "") -> str:
    value = plan.get(key, fallback)
    if isinstance(value, str) and value.upper() in {"T", "F"}:
        return value.upper()
    return fallback


def plan_int(plan: dict[str, str], key: str, fallback: Any = "") -> Any:
    value = plan.get(key)
    if value in (None, ""):
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def mozyme_reason_text(reason: Any) -> str:
    if reason in ("", None):
        return ""
    try:
        return MOZYME_GPU_REASON_TEXT.get(int(reason), f"unknown_{reason}")
    except (TypeError, ValueError):
        return ""


def parse_int_value(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_resident_real_pair_coverage(row: dict[str, Any], reasons: list[str]) -> None:
    real_pairs = parse_int_value(row.get("mozyme_resident_fock_real_pairs"))
    gpu_pairs = parse_int_value(row.get("mozyme_resident_fock_gpu_real_pairs"))
    cpu_pairs = parse_int_value(row.get("mozyme_resident_fock_cpu_real_pairs"))
    inactive_pairs = parse_int_value(row.get("mozyme_resident_fock_inactive_real_pairs"))

    if real_pairs is None:
        reasons.append("mozyme_resident_fock_real_pairs was not reported")
    if gpu_pairs is None:
        reasons.append("mozyme_resident_fock_gpu_real_pairs was not reported")
    if cpu_pairs is None:
        reasons.append("mozyme_resident_fock_cpu_real_pairs was not reported")
    if inactive_pairs is None:
        reasons.append("mozyme_resident_fock_inactive_real_pairs was not reported")
    if None in (real_pairs, gpu_pairs, cpu_pairs, inactive_pairs):
        return

    if real_pairs <= 0:
        reasons.append("resident sparse Fock coverage reported zero real pairs")
    if gpu_pairs <= 0:
        reasons.append("resident sparse Fock coverage reported zero GPU real pairs")
    if cpu_pairs != 0:
        reasons.append(f"mozyme_resident_fock_cpu_real_pairs is nonzero ({cpu_pairs}); resident Fock is not all-GPU")
    if real_pairs != gpu_pairs:
        reasons.append(
            "resident sparse Fock real-pair coverage is incomplete "
            f"(gpu_real_pairs={gpu_pairs}, real_pairs={real_pairs})"
        )
    if real_pairs != gpu_pairs + cpu_pairs:
        reasons.append(
            "resident sparse Fock coverage counters are inconsistent "
            f"(real={real_pairs}, gpu={gpu_pairs}, cpu={cpu_pairs})"
        )


def validate_resident_plan_coverage(row: dict[str, Any], reasons: list[str]) -> None:
    if row.get("mozyme_plan_resident_fock_gpu") != "T":
        return
    full = row.get("mozyme_fock_resident_full_coverage_planned")
    if full in ("", None):
        reasons.append("MOZYME GPU plan did not report resident Fock full-coverage planning")
        return
    if full != "T":
        unsupported_one = parse_int_value(
            row.get("mozyme_fock_resident_unsupported_one_center")
        ) or 0
        unsupported = parse_int_value(row.get("mozyme_fock_resident_unsupported_pairs")) or 0
        unsupported_point = parse_int_value(
            row.get("mozyme_fock_resident_unsupported_point_pairs")
        ) or 0
        direct_basis = parse_int_value(
            row.get("mozyme_fock_resident_direct_unsupported_pairs")
        ) or 0
        direct_point = parse_int_value(
            row.get("mozyme_fock_resident_direct_unsupported_point_pairs")
        ) or 0
        reasons.append(
            "MOZYME GPU plan reported incomplete resident Fock coverage "
            f"(unsupported_one_center={unsupported_one}, unsupported_real={unsupported}, unsupported_point={unsupported_point}, "
            f"direct_basis_real={direct_basis}, direct_basis_point={direct_point})"
        )
    executable_tasks = parse_int_value(row.get("mozyme_fock_resident_executable_tasks"))
    production_tasks = parse_int_value(row.get("mozyme_fock_production_gpu_tasks"))
    if full == "F" and (executable_tasks or 0) != 0:
        reasons.append(
            "MOZYME GPU plan reported resident executable tasks despite incomplete coverage "
            f"({executable_tasks})"
        )
    if full == "F" and row.get("mozyme_plan_fock_gpu") != "T" and (production_tasks or 0) != 0:
        reasons.append(
            "MOZYME GPU plan reported production resident Fock tasks despite incomplete coverage "
            f"({production_tasks})"
        )


def validate_resident_point_charge_coverage(
    row: dict[str, Any], sparse_calls: int, reasons: list[str]
) -> None:
    planned_point_pairs_value = parse_int_value(row.get("mozyme_fock_plan_point_charge_pairs"))
    planned_dipole_pairs_value = parse_int_value(row.get("mozyme_fock_plan_point_dipole_pairs"))
    planned_monopole_pairs_value = parse_int_value(row.get("mozyme_fock_plan_point_monopole_pairs"))
    setup_point_tasks_value = parse_int_value(row.get("mozyme_sparse_fock_setup_point_tasks"))
    setup_dipole_tasks_value = parse_int_value(row.get("mozyme_sparse_fock_setup_point_dipole_tasks"))
    setup_monopole_tasks_value = parse_int_value(row.get("mozyme_sparse_fock_setup_point_monopole_tasks"))
    run_point_tasks_value = parse_int_value(row.get("mozyme_sparse_fock_run_point_tasks"))
    run_dipole_tasks_value = parse_int_value(row.get("mozyme_sparse_fock_run_point_dipole_tasks"))
    run_monopole_tasks_value = parse_int_value(row.get("mozyme_sparse_fock_run_point_monopole_tasks"))
    resident_point_pairs_value = parse_int_value(row.get("mozyme_resident_fock_point_pairs"))
    resident_gpu_point_pairs_value = parse_int_value(row.get("mozyme_resident_fock_gpu_point_pairs"))
    resident_cpu_point_pairs_value = parse_int_value(row.get("mozyme_resident_fock_cpu_point_pairs"))

    if planned_point_pairs_value is None:
        reasons.append("mozyme_fock_plan_point_charge_pairs was not reported")
    if setup_point_tasks_value is None:
        reasons.append("mozyme_sparse_fock_setup_point_tasks was not reported")
    if run_point_tasks_value is None:
        reasons.append("mozyme_sparse_fock_run_point_tasks was not reported")

    planned_point_pairs = planned_point_pairs_value or 0
    setup_point_tasks = setup_point_tasks_value or 0
    run_point_tasks = run_point_tasks_value or 0
    if planned_point_pairs_value is None or planned_point_pairs <= 0:
        return

    if resident_point_pairs_value is None:
        reasons.append("mozyme_resident_fock_point_pairs was not reported")
    if resident_gpu_point_pairs_value is None:
        reasons.append("mozyme_resident_fock_gpu_point_pairs was not reported")
    if resident_cpu_point_pairs_value is None:
        reasons.append("mozyme_resident_fock_cpu_point_pairs was not reported")

    if planned_dipole_pairs_value is None:
        reasons.append("mozyme_fock_plan_point_dipole_pairs was not reported")
    if planned_monopole_pairs_value is None:
        reasons.append("mozyme_fock_plan_point_monopole_pairs was not reported")
    if setup_dipole_tasks_value is None:
        reasons.append("mozyme_sparse_fock_setup_point_dipole_tasks was not reported")
    if setup_monopole_tasks_value is None:
        reasons.append("mozyme_sparse_fock_setup_point_monopole_tasks was not reported")
    if run_dipole_tasks_value is None:
        reasons.append("mozyme_sparse_fock_run_point_dipole_tasks was not reported")
    if run_monopole_tasks_value is None:
        reasons.append("mozyme_sparse_fock_run_point_monopole_tasks was not reported")

    planned_dipole_pairs = planned_dipole_pairs_value or 0
    planned_monopole_pairs = planned_monopole_pairs_value or 0
    setup_dipole_tasks = setup_dipole_tasks_value or 0
    setup_monopole_tasks = setup_monopole_tasks_value or 0
    run_dipole_tasks = run_dipole_tasks_value or 0
    run_monopole_tasks = run_monopole_tasks_value or 0
    resident_point_pairs = resident_point_pairs_value or 0
    resident_gpu_point_pairs = resident_gpu_point_pairs_value or 0
    resident_cpu_point_pairs = resident_cpu_point_pairs_value or 0

    if resident_point_pairs_value is not None and resident_point_pairs != planned_point_pairs:
        reasons.append(
            "resident sparse Fock point coverage disagrees with the MOZYME plan "
            f"(resident_point_pairs={resident_point_pairs}, planned_point_pairs={planned_point_pairs})"
        )
    if resident_cpu_point_pairs_value is not None and resident_cpu_point_pairs != 0:
        reasons.append(
            f"mozyme_resident_fock_cpu_point_pairs is nonzero ({resident_cpu_point_pairs}); "
            "resident point-charge/dipole Fock is not all-GPU"
        )
    if (
        resident_point_pairs_value is not None
        and resident_gpu_point_pairs_value is not None
        and resident_cpu_point_pairs_value is not None
        and resident_point_pairs != resident_gpu_point_pairs + resident_cpu_point_pairs
    ):
        reasons.append(
            "resident sparse Fock point coverage counters are inconsistent "
            f"(point={resident_point_pairs}, gpu={resident_gpu_point_pairs}, cpu={resident_cpu_point_pairs})"
        )
    if resident_gpu_point_pairs_value is not None and resident_gpu_point_pairs < planned_point_pairs:
        reasons.append(
            "resident sparse Fock point-charge/dipole resident coverage is incomplete "
            f"({resident_gpu_point_pairs}/{planned_point_pairs})"
        )

    if (
        planned_dipole_pairs_value is not None
        and planned_monopole_pairs_value is not None
        and planned_dipole_pairs + planned_monopole_pairs != planned_point_pairs
    ):
        reasons.append(
            "MOZYME point-charge/dipole plan counters are inconsistent "
            f"(point={planned_point_pairs}, dipole={planned_dipole_pairs}, monopole={planned_monopole_pairs})"
        )
    if (
        setup_point_tasks_value is not None
        and setup_dipole_tasks_value is not None
        and setup_monopole_tasks_value is not None
        and setup_dipole_tasks + setup_monopole_tasks != setup_point_tasks
    ):
        reasons.append(
            "resident sparse Fock point setup counters are inconsistent "
            f"(point={setup_point_tasks}, dipole={setup_dipole_tasks}, monopole={setup_monopole_tasks})"
        )
    if (
        run_point_tasks_value is not None
        and run_dipole_tasks_value is not None
        and run_monopole_tasks_value is not None
        and run_dipole_tasks + run_monopole_tasks != run_point_tasks
    ):
        reasons.append(
            "resident sparse Fock point run counters are inconsistent "
            f"(point={run_point_tasks}, dipole={run_dipole_tasks}, monopole={run_monopole_tasks})"
        )

    if setup_point_tasks < planned_point_pairs:
        reasons.append(
            "resident sparse Fock point-charge/dipole setup coverage is incomplete "
            f"({setup_point_tasks}/{planned_point_pairs})"
        )
    if planned_dipole_pairs > 0 and setup_dipole_tasks < planned_dipole_pairs:
        reasons.append(
            "resident sparse Fock point-dipole setup coverage is incomplete "
            f"({setup_dipole_tasks}/{planned_dipole_pairs})"
        )
    if planned_monopole_pairs > 0 and setup_monopole_tasks < planned_monopole_pairs:
        reasons.append(
            "resident sparse Fock point-monopole setup coverage is incomplete "
            f"({setup_monopole_tasks}/{planned_monopole_pairs})"
        )

    required_point_runs = planned_point_pairs * max(1, sparse_calls)
    required_dipole_runs = planned_dipole_pairs * max(1, sparse_calls)
    required_monopole_runs = planned_monopole_pairs * max(1, sparse_calls)
    if run_point_tasks < required_point_runs:
        reasons.append(
            "resident sparse Fock point-charge/dipole run coverage is incomplete "
            f"({run_point_tasks}/{required_point_runs})"
        )
    if planned_dipole_pairs > 0 and run_dipole_tasks < required_dipole_runs:
        reasons.append(
            "resident sparse Fock point-dipole run coverage is incomplete "
            f"({run_dipole_tasks}/{required_dipole_runs})"
        )
    if planned_monopole_pairs > 0 and run_monopole_tasks < required_monopole_runs:
        reasons.append(
            "resident sparse Fock point-monopole run coverage is incomplete "
            f"({run_monopole_tasks}/{required_monopole_runs})"
        )


def select_resident_fock_coverage(matches: list[tuple[str, ...]]) -> tuple[str, ...] | None:
    for match in reversed(matches):
        if match[0] == "0":
            return match
    return matches[-1] if matches else None


def gpu_error_markers_in_text(text: str) -> list[str]:
    lowered = text.lower()
    return [marker for marker in GPU_ERROR_MARKERS if marker.lower() in lowered]


def mozyme_scf_stage_names(mask: Any) -> str:
    mask_value = parse_int_value(mask)
    if mask_value is None:
        return ""
    names = [name for bit, name in MOZYME_SCF_STAGE_BITS if mask_value & bit]
    unknown_mask = mask_value & ~MOZYME_SCF_STAGE_FULL
    if unknown_mask:
        names.append(f"unknown_{unknown_mask}")
    return "+".join(names) if names else "none"


def cnvgz_active_noop_ready(active_calls: int | None, noop_calls: int | None) -> bool:
    return (
        active_calls is not None
        and noop_calls is not None
        and active_calls >= 0
        and noop_calls >= 0
        and active_calls + noop_calls > 0
    )


def append_cnvgz_active_noop_reasons(
    reasons: list[str], active_calls: int | None, noop_calls: int | None
) -> None:
    if active_calls is None:
        reasons.append("resident-SCF CNVGZ active-call counter was not reported")
    elif active_calls < 0:
        reasons.append(f"resident-SCF CNVGZ active-call counter is negative ({active_calls})")
    if noop_calls is None:
        reasons.append("resident-SCF CNVGZ no-op counter was not reported")
    elif noop_calls < 0:
        reasons.append(f"resident-SCF CNVGZ no-op counter is negative ({noop_calls})")
    if (
        active_calls is not None
        and noop_calls is not None
        and active_calls + noop_calls <= 0
    ):
        reasons.append(
            "resident-SCF CNVGZ did not report active or no-op GPU stage work "
            f"(active={active_calls}, noop={noop_calls})"
        )


def classify_full_scf_gpu(
    raw_status: str,
    scf_code: tuple[str, ...] | None,
    scf_stage: tuple[str, ...] | None,
    scf_decision: tuple[str, ...] | None,
    scf_isitsc: tuple[str, ...] | None,
    scf_device: tuple[str, ...] | None,
    scf_cnvgz_activity: tuple[str, ...] | None,
    scf_final_iterations: int | None,
    final_density_current: bool,
    has_marker: bool,
) -> tuple[str, int, int]:
    if not raw_status:
        if has_marker:
            return "marker_without_status", 0, 0
        return "not_requested", 0, 0

    status = raw_status.lower()
    if status == "fallback_cpu":
        return "fallback_cpu", 0, 1
    if status == "strict_abort":
        return "strict_abort", 0, 1
    if status == "resident_step":
        return "resident_step", 0, 0
    if status != "success":
        return "marker_without_status", 0, 0

    backend_code = parse_int_value(scf_code[1]) if scf_code else None
    backend_ready = parse_int_value(scf_code[2]) if scf_code else None
    resident = parse_int_value(scf_code[3]) if scf_code else None
    stage_completed = parse_int_value(scf_stage[0]) if scf_stage else None
    stage_required = parse_int_value(scf_stage[1]) if scf_stage else None
    stage_missing = parse_int_value(scf_stage[2]) if scf_stage else None
    resident_decision = parse_int_value(scf_decision[0]) if scf_decision else None
    isitsc_okscf = parse_int_value(scf_isitsc[0]) if scf_isitsc else None
    device_id = parse_int_value(scf_device[0]) if scf_device else None
    cnvgz_active_calls = (
        parse_int_value(scf_cnvgz_activity[0])
        if scf_cnvgz_activity
        else None
    )
    cnvgz_noop_calls = (
        parse_int_value(scf_cnvgz_activity[1])
        if scf_cnvgz_activity
        else None
    )
    if (
        backend_code == 0
        and backend_ready == 1
        and resident == 1
        and device_id is not None
        and device_id >= 0
        and stage_required == MOZYME_SCF_STAGE_FULL
        and stage_completed == MOZYME_SCF_STAGE_FULL
        and stage_missing == 0
        and resident_decision == MOZYME_SCF_RESIDENT_DECISION_COMPLETE
        and isitsc_okscf == 1
        and cnvgz_active_noop_ready(cnvgz_active_calls, cnvgz_noop_calls)
        and scf_final_iterations is not None
        and scf_final_iterations >= 1
        and final_density_current
    ):
        return "complete", 1, 0
    if (
        backend_code is None
        or backend_ready is None
        or resident is None
        or stage_completed is None
        or stage_required is None
        or stage_missing is None
        or resident_decision is None
        or isitsc_okscf is None
        or cnvgz_active_calls is None
        or cnvgz_noop_calls is None
        or device_id is None
        or scf_final_iterations is None
    ):
        return "unverified_success", 0, 0
    return "incomplete_success", 0, 0


def classify_full_scf_probe_decision(text: str) -> str:
    has_marker = bool(MOZYME_SCF_RE.search(text))
    if not has_marker:
        return "not_executed"
    if MOZYME_MAKVEC_FALLBACK_RE.search(text):
        return "makvec_fallback_cpu"
    if MOZYME_GPU_HELPER_FATAL_RE.search(text):
        return "helper_fatal"
    makvec_ready = bool(MOZYME_MAKVEC_SUCCESS_RE.search(text))
    if MOZYME_MAKVEC_EXISTING_RE.search(text) and not makvec_ready:
        return "makvec_existing_lmo_not_gpu_proof"
    scf_status_matches = MOZYME_SCF_STATUS_RE.findall(text)
    if not scf_status_matches:
        return "marker_without_status"
    if any(status.lower() == "fallback_cpu" for status, _reason in scf_status_matches):
        return "fallback_cpu"
    if any(status.lower() == "strict_abort" for status, _reason in scf_status_matches):
        return "strict_abort"
    if any(status.lower() == "resident_step" for status, _reason in scf_status_matches):
        return "resident_step"

    raw_status = scf_status_matches[-1][0].lower()
    if raw_status != "success":
        return "marker_without_status"
    diagnostics = parse_mozyme_gpu_diagnostics(text)
    resident_decision_matches = MOZYME_SCF_DECISION_RE.findall(text)
    resident_decision_complete = (
        parse_int_value(resident_decision_matches[-1])
        == MOZYME_SCF_RESIDENT_DECISION_COMPLETE
        if resident_decision_matches
        else False
    )

    if (
        diagnostics.get("full_scf_gpu_status") == "complete"
        and parse_int_value(diagnostics.get("full_scf_gpu_ready")) == 1
        and makvec_ready
    ):
        return "complete"
    if (
        MOZYME_SCF_CODE_RE.search(text)
        and MOZYME_SCF_STAGE_RE.search(text)
        and resident_decision_complete
        and MOZYME_SCF_ISITSC_RE.search(text)
        and MOZYME_SCF_STATUS_ITERATIONS_RE.search(text)
        and MOZYME_SCF_FINAL_DENSITY_RE.search(text)
    ):
        return "success_pending_makvec"
    if (
        MOZYME_SCF_CODE_RE.search(text)
        and MOZYME_SCF_STAGE_RE.search(text)
        and resident_decision_complete
        and MOZYME_SCF_ISITSC_RE.search(text)
        and MOZYME_SCF_STATUS_ITERATIONS_RE.search(text)
    ):
        return "success_pending_final_density"
    if (
        MOZYME_SCF_CODE_RE.search(text)
        and MOZYME_SCF_STAGE_RE.search(text)
        and resident_decision_complete
        and MOZYME_SCF_ISITSC_RE.search(text)
    ):
        return "success_pending_final_iterations"
    if MOZYME_SCF_CODE_RE.search(text) and MOZYME_SCF_STAGE_RE.search(text) and resident_decision_matches:
        return "success_pending_resident_decision"
    return "success_pending_masks"


def marker_field_equals(extra: str, field: str, value: str) -> bool:
    if not extra:
        return False
    return re.search(rf"(?:^|\s){re.escape(field)}\s*=\s*{re.escape(value)}(?:\s|$)", extra) is not None


def marker_fields(extra: str) -> dict[str, str]:
    return dict(MARKER_FIELD_RE.findall(extra or ""))


def parse_resident_fock_point_coverage(extra: str) -> tuple[str, ...] | None:
    fields = marker_fields(extra)
    required_fields = ("point_pairs", "gpu_point_pairs", "cpu_point_pairs")
    if any(field not in fields for field in required_fields):
        return None
    reason_counts = {
        reason: count for reason, count in MARKER_REASON_COUNT_RE.findall(extra or "")
    }
    return (
        fields["point_pairs"],
        fields["gpu_point_pairs"],
        fields["cpu_point_pairs"],
        reason_counts.get("unsupported_basis_limit", "0"),
        reason_counts.get("unsupported_direct_basis", "0"),
        reason_counts.get("unsupported_other", "0"),
    )


def parse_scf_cnvgz_activity(line: str) -> tuple[str, str] | None:
    fields = marker_fields(line)
    active_calls = fields.get("cnvgz_active_calls")
    noop_calls = fields.get("cnvgz_noop_calls")
    if active_calls is None or noop_calls is None:
        return None
    return active_calls, noop_calls


def parse_mozyme_gpu_diagnostics(text: str) -> dict[str, Any]:
    """Extract MOZYME GPU counters emitted by profiling builds.

    Empty fields mean the executable did not emit the profiling markers, which
    usually indicates an old build or a non-MOZYME input.
    """
    profile_matches = MOZYME_PROFILE_RE.findall(text)
    fock_profile_matches = MOZYME_FOCK_PROFILE_RE.findall(text)
    check_profile_matches = MOZYME_CHECK_PROFILE_RE.findall(text)
    request_profile_matches = MOZYME_REQUEST_PROFILE_RE.findall(text)
    fock1_batch_profile_matches = MOZYME_FOCK1_BATCH_PROFILE_RE.findall(text)
    resident_fock_profile_matches = MOZYME_RESIDENT_FOCK_PROFILE_RE.findall(text)
    fock2_4x1_batch_profile_matches = MOZYME_FOCK2_4X1_BATCH_PROFILE_RE.findall(text)
    plan_matches = MOZYME_PLAN_RE.findall(text)
    plan_block = parse_mozyme_plan_block(text)
    plan_density_matches = MOZYME_PLAN_DENSITY_RE.findall(text)
    plan_after_matches = MOZYME_PLAN_AFTER_RE.findall(text)
    fock_plan_matches = MOZYME_FOCK_PLAN_RE.findall(text)
    fock_plan_shape_matches = MOZYME_FOCK_PLAN_SHAPES_RE.findall(text)
    fock_plan_task_matches = MOZYME_FOCK_PLAN_TASKS_RE.findall(text)
    fock_plan_two_center_term_matches = MOZYME_FOCK_PLAN_TWO_CENTER_TERMS_RE.findall(text)
    density_calls = MOZYME_DENSITY_CALL_RE.findall(text)
    density_skips = MOZYME_DENSITY_SKIP_RE.findall(text)
    density_maxima = MOZYME_DENSITY_MAX_RE.findall(text)
    density_batch_matches = MOZYME_DENSITY_BATCH_RE.findall(text)
    gpu_debug_matches = GPU_DEBUG_RE.findall(text)
    gpu_request_debug_matches = GPU_REQUEST_DEBUG_RE.findall(text)
    resident_matches = RESIDENT_DEBUG_RE.findall(text)
    scf_status_matches = MOZYME_SCF_STATUS_RE.findall(text)
    scf_status_iteration_matches = MOZYME_SCF_STATUS_ITERATIONS_RE.findall(text)
    scf_code_matches = MOZYME_SCF_CODE_RE.findall(text)
    scf_metric_matches = MOZYME_SCF_METRICS_RE.findall(text)
    scf_diagg_matches = MOZYME_SCF_DIAGG_RE.findall(text)
    scf_stage_matches = MOZYME_SCF_STAGE_RE.findall(text)
    scf_stage_names_raw_matches = MOZYME_SCF_STAGE_NAMES_RAW_RE.findall(text)
    scf_strict_proof_matches = MOZYME_SCF_STRICT_PROOF_RE.findall(text)
    scf_resident_fock_plan_matches = MOZYME_SCF_RESIDENT_FOCK_PLAN_RE.findall(text)
    scf_decision_matches = MOZYME_SCF_DECISION_RE.findall(text)
    scf_stage_calls_matches = MOZYME_SCF_STAGE_CALLS_RE.findall(text)
    scf_stage_ms_matches = MOZYME_SCF_STAGE_MS_RE.findall(text)
    scf_cnvgz_activity_matches = MOZYME_SCF_CNVGZ_ACTIVITY_RE.findall(text)
    scf_energy_matches = MOZYME_SCF_ENERGY_RE.findall(text)
    scf_isitsc_matches = MOZYME_SCF_ISITSC_RE.findall(text)
    scf_pls_matches = MOZYME_SCF_PLS_RE.findall(text)
    scf_pls_delta_matches = MOZYME_SCF_PLS_DELTA_RE.findall(text)
    scf_device_matches = MOZYME_SCF_DEVICE_RE.findall(text)
    scf_cosmo_matches = MOZYME_SCF_COSMO_RE.findall(text)
    scf_cosmo_energy_matches = MOZYME_SCF_COSMO_ENERGY_RE.findall(text)
    scf_cosmo_cg_matches = MOZYME_SCF_COSMO_CG_RE.findall(text)
    scf_final_density_current = bool(MOZYME_SCF_FINAL_DENSITY_RE.search(text))
    scf_host_commit_matches = MOZYME_SCF_HOST_COMMIT_RE.findall(text)
    scf_final_publication_matches = MOZYME_SCF_FINAL_PUBLICATION_RE.findall(text)
    scf_cpu_setup_matches = MOZYME_SCF_CPU_SETUP_RE.findall(text)
    scf_fillij_gpu_matches = MOZYME_SCF_FILLIJ_GPU_RE.findall(text)
    resident_fock_gpu_count_matches = MOZYME_RESIDENT_FOCK_GPU_COUNT_RE.findall(text)
    resident_fock_gpu_pack_matches = MOZYME_RESIDENT_FOCK_GPU_PACK_RE.findall(text)
    resident_fock_gpu_point_weight_matches = (
        MOZYME_RESIDENT_FOCK_GPU_POINT_WEIGHTS_RE.findall(text)
    )
    resident_fock_cpu_plan_matches = MOZYME_RESIDENT_FOCK_CPU_PLAN_RE.findall(text)
    cpu_pinout_matches = MOZYME_CPU_PINOUT_RE.findall(text)
    makvec_success_matches = MOZYME_MAKVEC_SUCCESS_RE.findall(text)
    makvec_existing_matches = MOZYME_MAKVEC_EXISTING_RE.findall(text)
    makvec_fallback_matches = MOZYME_MAKVEC_FALLBACK_RE.findall(text)
    helper_fatal_markers = MOZYME_GPU_HELPER_FATAL_RE.findall(text)
    setupk_matches = MOZYME_SETUPK_RE.findall(text)
    setupk_initial_matches = [
        match for match in setupk_matches if marker_field_equals(match[3], "initial_setup", "1")
    ]
    setupk_all_initial_matches = [
        match for match in setupk_matches if marker_field_equals(match[3], "all_initial_setup_paths", "1")
    ]
    setupk_initial_success_matches = [
        match for match in setupk_initial_matches if match[0].lower() == "success"
    ]
    setupk_initial_fallback_matches = [
        match for match in setupk_initial_matches if match[0].lower() in {"fallback_cpu", "strict_abort"}
    ]
    relocal_matches = MOZYME_RELOCAL_RE.findall(text)
    relocal_success_counts = mozyme_relocal_success_counts(text)
    reorth_matches = MOZYME_REORTH_RE.findall(text)
    tidy_matches = MOZYME_TIDY_RE.findall(text)
    cnvgz_matches = MOZYME_CNVGZ_RE.findall(text)
    helecz_matches = MOZYME_HELECZ_RE.findall(text)
    eimp_matches = MOZYME_EIMP_RE.findall(text)
    diagg1_aocc_matches = MOZYME_DIAGG1_AOCC_RE.findall(text)
    diagg1_avir_matches = MOZYME_DIAGG1_AVIR_RE.findall(text)
    diagg1_construct_matches = MOZYME_DIAGG1_CONSTRUCT_RE.findall(text)
    diagg2_rotprep_matches = MOZYME_DIAGG2_ROTPREP_RE.findall(text)
    diagg2_rotate_matches = MOZYME_DIAGG2_ROTATE_RE.findall(text)
    isitsc_matches = MOZYME_ISITSC_RE.findall(text)
    fock_markers = MOZYME_FOCK_RE.findall(text)
    fock1_batch_markers = MOZYME_FOCK1_BATCH_RE.findall(text)
    fock2_4x1_batch_markers = MOZYME_FOCK2_4X1_BATCH_RE.findall(text)
    sparse_fock_setup_matches = MOZYME_SPARSE_FOCK_SETUP_RE.findall(text)
    sparse_fock_run_matches = MOZYME_SPARSE_FOCK_RUN_RE.findall(text)
    resident_fock_coverage_matches = MOZYME_RESIDENT_FOCK_COVERAGE_RE.findall(text)
    resident_fock_fallback_matches = MOZYME_RESIDENT_FOCK_FALLBACK_RE.findall(text)
    resident_fock_point_coverage_matches = MOZYME_RESIDENT_FOCK_POINT_COVERAGE_RE.findall(text)
    resident_fock_cpu_point_pair_matches = MOZYME_RESIDENT_FOCK_CPU_POINT_PAIRS_RE.findall(text)
    resident_coverage = select_resident_fock_coverage(resident_fock_coverage_matches)
    resident_fallback = resident_fock_fallback_matches[-1] if resident_fock_fallback_matches else None
    resident_point_coverage = (
        parse_resident_fock_point_coverage(resident_fock_point_coverage_matches[-1])
        if resident_fock_point_coverage_matches
        else None
    )
    resident_cpu_point_pair_counts = [
        int(value) for value in resident_fock_cpu_point_pair_matches
    ]
    resident_cpu_point_pair_max = (
        max(resident_cpu_point_pair_counts)
        if resident_cpu_point_pair_counts
        else None
    )

    mozyme_gpu_profile = ""
    mozyme_minblk: int | str = ""
    if profile_matches:
        mozyme_gpu_profile = profile_matches[-1][0].upper()
        mozyme_minblk = int(profile_matches[-1][1])

    if density_calls:
        mozyme_minblk = int(density_calls[-1][1])

    if "minblk" in plan_block:
        mozyme_minblk = plan_int(plan_block, "minblk", mozyme_minblk)

    requested = ""
    disable_reason: Any = ""
    if request_profile_matches:
        requested = request_profile_matches[-1][0].upper()
        disable_reason = int(request_profile_matches[-1][1])
    if gpu_request_debug_matches:
        requested = gpu_request_debug_matches[-1][0].upper()
        disable_reason = int(gpu_request_debug_matches[-1][1])
    requested = plan_bool(plan_block, "requested", requested)
    disable_reason = plan_int(plan_block, "disable_reason", disable_reason)

    has_scf_marker = bool(MOZYME_SCF_RE.search(text))
    strict_host_route_markers = list(
        dict.fromkeys(
            [marker for marker in STRICT_HOST_ROUTE_MARKERS if marker in text]
            + [
                reason
                for reason in MOZYME_SCF_STRICT_ABORT_REASON_RE.findall(text)
                if reason.lower().startswith("strict_")
            ]
        )
    )
    scf_olden_setup_only = bool(MOZYME_SCF_OLDEN_SETUP_RE.search(text))
    scf_fillij_gpu_count_matches = [
        match for match in scf_fillij_gpu_matches if match[0].upper() == "T"
    ]
    scf_fillij_gpu_fill_matches = [
        match for match in scf_fillij_gpu_matches if match[0].upper() == "F"
    ]
    scf_fillij_gpu_last = (
        scf_fillij_gpu_matches[-1] if scf_fillij_gpu_matches else None
    )
    resident_fock_gpu_count = (
        resident_fock_gpu_count_matches[-1] if resident_fock_gpu_count_matches else None
    )
    resident_fock_gpu_pack = (
        resident_fock_gpu_pack_matches[-1] if resident_fock_gpu_pack_matches else None
    )
    resident_fock_gpu_point_weights = (
        resident_fock_gpu_point_weight_matches[-1]
        if resident_fock_gpu_point_weight_matches
        else None
    )
    resident_fock_cpu_plan = (
        resident_fock_cpu_plan_matches[-1] if resident_fock_cpu_plan_matches else None
    )
    host_commit_phase = (
        scf_host_commit_matches[-1][0] if scf_host_commit_matches else ""
    )
    host_commit_arrays = (
        int(scf_host_commit_matches[-1][1]) if scf_host_commit_matches else ""
    )
    host_commit_bytes = (
        sum(int(match[2]) for match in scf_host_commit_matches)
        if scf_host_commit_matches
        else ""
    )
    host_commit_cosmo = (
        int(scf_host_commit_matches[-1][3]) if scf_host_commit_matches else ""
    )
    final_publication = (
        scf_final_publication_matches[-1] if scf_final_publication_matches else None
    )
    scf_code = scf_code_matches[-1] if scf_code_matches else None
    scf_final_iterations = int(scf_status_iteration_matches[-1][1]) if scf_status_iteration_matches else None
    scf_metrics = scf_metric_matches[-1] if scf_metric_matches else None
    scf_diagg = scf_diagg_matches[-1] if scf_diagg_matches else None
    scf_stage = scf_stage_matches[-1] if scf_stage_matches else None
    scf_stage_names_raw = scf_stage_names_raw_matches[-1] if scf_stage_names_raw_matches else None
    scf_strict_proof = scf_strict_proof_matches[-1] if scf_strict_proof_matches else None
    scf_resident_fock_plan = (
        scf_resident_fock_plan_matches[-1] if scf_resident_fock_plan_matches else None
    )
    scf_decision = (scf_decision_matches[-1],) if scf_decision_matches else None
    scf_stage_calls = scf_stage_calls_matches[-1] if scf_stage_calls_matches else None
    scf_stage_ms = scf_stage_ms_matches[-1] if scf_stage_ms_matches else None
    scf_cnvgz_activity = (
        parse_scf_cnvgz_activity(scf_cnvgz_activity_matches[-1])
        if scf_cnvgz_activity_matches
        else None
    )
    scf_isitsc = scf_isitsc_matches[-1] if scf_isitsc_matches else None
    scf_pls = scf_pls_matches[-1] if scf_pls_matches else None
    scf_pls_delta = scf_pls_delta_matches[-1] if scf_pls_delta_matches else None
    scf_pls_reset_matches = MOZYME_SCF_PLS_RESET_RE.findall(text)
    scf_pls_reset = scf_pls_reset_matches[-1] if scf_pls_reset_matches else None
    scf_device = scf_device_matches[-1] if scf_device_matches else None
    scf_cosmo = scf_cosmo_matches[-1] if scf_cosmo_matches else None
    scf_cosmo_cg = scf_cosmo_cg_matches[-1] if scf_cosmo_cg_matches else None
    scf_compact_matches = MOZYME_SCF_COMPACT_RE.findall(text)
    scf_compact = scf_compact_matches[-1] if scf_compact_matches else None
    scf_cosmo_energy = scf_cosmo_energy_matches[-1] if scf_cosmo_energy_matches else None
    scf_status = ""
    scf_reason = ""
    scf_success_calls = sum(1 for status, _reason in scf_status_matches if status.lower() == "success")
    scf_fallback_calls = sum(
        1
        for status, _reason in scf_status_matches
        if status.lower() in {"fallback_cpu", "strict_abort"}
    )
    scf_resident_step_calls = sum(
        1 for status, _reason in scf_status_matches if status.lower() == "resident_step"
    )
    scf_cpu_boundary_calls = sum(
        1
        for status, reason in scf_status_matches
        if status.lower() == "fallback_cpu" and reason == "backend_cpu_boundary"
    )
    scf_pls_restart_required_calls = sum(
        1
        for status, reason in scf_status_matches
        if status.lower() == "fallback_cpu" and reason == "backend_pls_restart_required"
    )
    if scf_status_matches:
        raw_status, raw_reason = scf_status_matches[-1]
        if raw_status.lower() == "success":
            scf_status = "success_pending_contract"
        elif raw_status.lower() == "resident_step":
            scf_status = "resident_step"
        elif raw_status.lower() == "strict_abort":
            scf_status = "strict_abort"
        else:
            scf_status = "fallback_cpu"
        scf_reason = raw_reason
        full_scf_status, full_scf_ready, full_scf_fallback = classify_full_scf_gpu(
            raw_status,
            scf_code,
            scf_stage,
            scf_decision,
            scf_isitsc,
            scf_device,
            scf_cnvgz_activity,
            scf_final_iterations,
            scf_final_density_current,
            has_scf_marker,
        )
    elif has_scf_marker:
        scf_status = "executed"
        full_scf_status, full_scf_ready, full_scf_fallback = classify_full_scf_gpu(
            "",
            scf_code,
            scf_stage,
            scf_decision,
            scf_isitsc,
            scf_device,
            scf_cnvgz_activity,
            scf_final_iterations,
            scf_final_density_current,
            has_scf_marker,
        )
    else:
        full_scf_status, full_scf_ready, full_scf_fallback = classify_full_scf_gpu(
            "",
            scf_code,
            scf_stage,
            scf_decision,
            scf_isitsc,
            scf_device,
            scf_cnvgz_activity,
            scf_final_iterations,
            scf_final_density_current,
            has_scf_marker,
        )

    return {
        "mozyme_gpu_profile": mozyme_gpu_profile,
        "mozyme_gpu_requested": requested,
        "mozyme_minblk": mozyme_minblk,
        "mozyme_fock1_batch_gpu_profile": (
            fock1_batch_profile_matches[-1].upper() if fock1_batch_profile_matches else ""
        ),
        "mozyme_resident_fock_gpu_profile": (
            resident_fock_profile_matches[-1].upper() if resident_fock_profile_matches else ""
        ),
        "mozyme_fock2_4x1_batch_gpu_profile": (
            fock2_4x1_batch_profile_matches[-1].upper() if fock2_4x1_batch_profile_matches else ""
        ),
        "mozyme_fock_gpu_profile": fock_profile_matches[-1].upper() if fock_profile_matches else "",
        "mozyme_check_gpu_profile": check_profile_matches[-1].upper() if check_profile_matches else "",
        "mozyme_plan_lgpu": plan_bool(plan_block, "lgpu", plan_matches[-1][0].upper() if plan_matches else ""),
        "mozyme_plan_enabled": plan_bool(plan_block, "enabled", ""),
        "mozyme_plan_ready": plan_bool(plan_block, "plan_ready", ""),
        "mozyme_plan_direct_mode": plan_bool(plan_block, "direct", ""),
        "mozyme_plan_fock_gpu": plan_bool(plan_block, "fock_gpu", plan_matches[-1][1].upper() if plan_matches else ""),
        "mozyme_plan_f2_gpu": plan_bool(plan_block, "f2_gpu", ""),
        "mozyme_plan_resident_fock_gpu": plan_bool(plan_block, "resident_fock_gpu", ""),
        "mozyme_plan_fock1_batch_gpu": plan_bool(plan_block, "fock1_batch_gpu", ""),
        "mozyme_plan_fock2_4x1_batch_gpu": plan_bool(plan_block, "fock2_4x1_batch_gpu", ""),
        "mozyme_plan_check_gpu": plan_bool(plan_block, "check_gpu", ""),
        "mozyme_plan_minblk": plan_int(plan_block, "minblk", int(plan_matches[-1][2]) if plan_matches else ""),
        "mozyme_plan_max_block": plan_int(plan_block, "max_block", int(plan_matches[-1][3]) if plan_matches else ""),
        "mozyme_plan_density_pairs_meeting_minblk": plan_int(
            plan_block,
            "density_pairs_meeting_minblk",
            int(plan_density_matches[-1]) if plan_density_matches else "",
        ),
        "mozyme_gpu_after_plan": plan_bool(
            plan_block, "mozyme_gpu", plan_after_matches[-1][0].upper() if plan_after_matches else ""
        ),
        "mozyme_plan_disabled_no_work": plan_bool(
            plan_block, "disabled_no_work", plan_after_matches[-1][1].upper() if plan_after_matches else ""
        ),
        "mozyme_plan_disable_reason": disable_reason,
        "mozyme_plan_disable_reason_text": mozyme_reason_text(disable_reason),
        "mozyme_fock_plan_one_center": plan_int(
            plan_block, "fock_one_center_tasks", int(fock_plan_matches[-1][0]) if fock_plan_matches else ""
        ),
        "mozyme_fock_resident_supported_one_center": plan_int(
            plan_block, "fock_resident_supported_one_center", ""
        ),
        "mozyme_fock_resident_unsupported_one_center": plan_int(
            plan_block, "fock_resident_unsupported_one_center", ""
        ),
        "mozyme_fock_plan_two_center": plan_int(
            plan_block, "fock_two_center_tasks", int(fock_plan_matches[-1][1]) if fock_plan_matches else ""
        ),
        "mozyme_fock_plan_skipped": plan_int(
            plan_block, "fock_skipped_pairs", int(fock_plan_matches[-1][2]) if fock_plan_matches else ""
        ),
        "mozyme_fock_plan_point_charge_pairs": plan_int(plan_block, "fock_point_charge_pairs", ""),
        "mozyme_fock_plan_point_dipole_pairs": plan_int(plan_block, "fock_point_dipole_pairs", ""),
        "mozyme_fock_plan_point_monopole_pairs": plan_int(plan_block, "fock_point_monopole_pairs", ""),
        "mozyme_fock_plan_d_pairs": plan_int(
            plan_block, "fock_d_pairs", int(fock_plan_matches[-1][3]) if fock_plan_matches else ""
        ),
        "mozyme_fock_plan_pairs_4x4": plan_int(
            plan_block, "fock_pairs_4x4", int(fock_plan_shape_matches[-1][0]) if fock_plan_shape_matches else ""
        ),
        "mozyme_fock_plan_pairs_4x1": plan_int(
            plan_block, "fock_pairs_4x1", int(fock_plan_shape_matches[-1][1]) if fock_plan_shape_matches else ""
        ),
        "mozyme_fock_plan_pairs_9x4": plan_int(
            plan_block, "fock_pairs_9x4", int(fock_plan_shape_matches[-1][2]) if fock_plan_shape_matches else ""
        ),
        "mozyme_fock_plan_pairs_9x9": plan_int(
            plan_block, "fock_pairs_9x9", int(fock_plan_shape_matches[-1][3]) if fock_plan_shape_matches else ""
        ),
        "mozyme_fock_resident_supported_pairs": plan_int(plan_block, "fock_resident_supported_pairs", ""),
        "mozyme_fock_resident_unsupported_pairs": plan_int(plan_block, "fock_resident_unsupported_pairs", ""),
        "mozyme_fock_resident_noop_pairs": plan_int(plan_block, "fock_resident_noop_pairs", ""),
        "mozyme_fock_resident_basis_limit_unsupported_pairs": plan_int(
            plan_block, "fock_resident_basis_limit_unsupported_pairs", ""
        ),
        "mozyme_fock_resident_direct_unsupported_pairs": plan_int(
            plan_block, "fock_resident_direct_unsupported_pairs", ""
        ),
        "mozyme_fock_resident_other_unsupported_pairs": plan_int(
            plan_block, "fock_resident_other_unsupported_pairs", ""
        ),
        "mozyme_fock_resident_supported_point_pairs": plan_int(
            plan_block, "fock_resident_supported_point_pairs", ""
        ),
        "mozyme_fock_resident_unsupported_point_pairs": plan_int(
            plan_block, "fock_resident_unsupported_point_pairs", ""
        ),
        "mozyme_fock_resident_basis_limit_unsupported_point_pairs": plan_int(
            plan_block, "fock_resident_basis_limit_unsupported_point_pairs", ""
        ),
        "mozyme_fock_resident_direct_unsupported_point_pairs": plan_int(
            plan_block, "fock_resident_direct_unsupported_point_pairs", ""
        ),
        "mozyme_fock_resident_other_unsupported_point_pairs": plan_int(
            plan_block, "fock_resident_other_unsupported_point_pairs", ""
        ),
        "mozyme_fock_resident_full_coverage_planned": plan_bool(
            plan_block, "fock_resident_full_coverage_planned", ""
        ),
        "mozyme_fock_resident_executable_tasks": plan_int(
            plan_block, "fock_resident_executable_tasks", ""
        ),
        "mozyme_fock_candidate_gpu_tasks": plan_int(
            plan_block,
            "fock_candidate_gpu_tasks",
            int(fock_plan_task_matches[-1][0]) if fock_plan_task_matches else "",
        ),
        "mozyme_fock_production_gpu_tasks": plan_int(
            plan_block,
            "fock_production_gpu_tasks",
            int(fock_plan_task_matches[-1][1]) if fock_plan_task_matches else "",
        ),
        "mozyme_fock_one_center_terms": plan_int(
            plan_block, "fock_one_center_terms", int(fock_plan_task_matches[-1][2]) if fock_plan_task_matches else ""
        ),
        "mozyme_fock_two_center_terms": plan_int(
            plan_block,
            "fock_two_center_terms",
            int(fock_plan_two_center_term_matches[-1]) if fock_plan_two_center_term_matches else "",
        ),
        "gpu_has_device": gpu_debug_matches[-1][0].upper() if gpu_debug_matches else "",
        "gpu_device_count": int(gpu_debug_matches[-1][1]) if gpu_debug_matches else "",
        "gpu_lgpu_final": gpu_debug_matches[-1][2].upper() if gpu_debug_matches else "",
        "gpu_resident_scf_final": resident_matches[-1].upper() if resident_matches else "",
        "mozyme_scf_experimental_executed": int(has_scf_marker),
        "mozyme_scf_experimental_status": scf_status,
        "mozyme_scf_experimental_reason": scf_reason,
        "full_scf_gpu_requested": 0,
        "full_scf_gpu_executed": int(has_scf_marker),
        "full_scf_gpu_ready": full_scf_ready,
        "full_scf_gpu_status": full_scf_status,
        "full_scf_gpu_reason": scf_reason,
        "full_scf_gpu_fallback": full_scf_fallback,
        "full_scf_gpu_scf_success_calls": scf_success_calls,
        "full_scf_gpu_scf_fallback_calls": scf_fallback_calls,
        "full_scf_gpu_resident_step_calls": scf_resident_step_calls,
        "full_scf_gpu_cpu_boundary_calls": scf_cpu_boundary_calls,
        "mozyme_gpu_helper_fatal_marker_count": len(helper_fatal_markers),
        "mozyme_gpu_helper_fatal_markers": ";".join(helper_fatal_markers[-20:]),
        "full_scf_gpu_strict_host_route_marker_count": len(strict_host_route_markers),
        "full_scf_gpu_strict_host_route_markers": ";".join(strict_host_route_markers),
        "full_scf_gpu_pls_restart_required_calls": scf_pls_restart_required_calls,
        "full_scf_gpu_code": int(scf_code[1]) if scf_code else "",
        "full_scf_gpu_backend_ready": int(scf_code[2]) if scf_code else "",
        "full_scf_gpu_resident": int(scf_code[3]) if scf_code else "",
        "full_scf_gpu_compact_index_route": int(scf_compact[0]) if scf_compact else "",
        "full_scf_gpu_use_nijbo": int(scf_compact[1]) if scf_compact else "",
        "full_scf_gpu_device_id": int(scf_device[0]) if scf_device else "",
        "full_scf_gpu_stage_completed": int(scf_stage[0]) if scf_stage else "",
        "full_scf_gpu_stage_required": int(scf_stage[1]) if scf_stage else "",
        "full_scf_gpu_stage_missing": int(scf_stage[2]) if scf_stage else "",
        "full_scf_gpu_resident_decision": int(scf_decision[0]) if scf_decision else "",
        "full_scf_gpu_stage_completed_names": mozyme_scf_stage_names(scf_stage[0]) if scf_stage else "",
        "full_scf_gpu_stage_missing_names": mozyme_scf_stage_names(scf_stage[2]) if scf_stage else "",
        "full_scf_gpu_stage_completed_names_raw": scf_stage_names_raw[0] if scf_stage_names_raw else "",
        "full_scf_gpu_stage_missing_names_raw": scf_stage_names_raw[1] if scf_stage_names_raw else "",
        "full_scf_gpu_strict_resident": int(scf_strict_proof[0]) if scf_strict_proof else "",
        "full_scf_gpu_no_fallback_required": int(scf_strict_proof[1]) if scf_strict_proof else "",
        "full_scf_gpu_full_stage_mask": int(scf_strict_proof[2]) if scf_strict_proof else "",
        "full_scf_gpu_resident_decision_complete": int(scf_strict_proof[3]) if scf_strict_proof else "",
        "full_scf_gpu_strict_resident_host_syncs": int(scf_strict_proof[4]) if scf_strict_proof else "",
        "full_scf_gpu_strict_resident_control_polls": int(scf_strict_proof[5]) if scf_strict_proof else "",
        "full_scf_gpu_resident_fock_plan_id": int(scf_resident_fock_plan[0]) if scf_resident_fock_plan else "",
        "full_scf_gpu_resident_fock_plan_full_coverage": (
            int(scf_resident_fock_plan[1]) if scf_resident_fock_plan else ""
        ),
        "full_scf_gpu_resident_fock_plan_partial_coverage": (
            int(scf_resident_fock_plan[2]) if scf_resident_fock_plan and scf_resident_fock_plan[2] else ""
        ),
        "full_scf_gpu_resident_fock_plan_required_mask": (
            int(scf_resident_fock_plan[3]) if scf_resident_fock_plan and scf_resident_fock_plan[3] else ""
        ),
        "full_scf_gpu_resident_fock_plan_covered_mask": (
            int(scf_resident_fock_plan[4]) if scf_resident_fock_plan and scf_resident_fock_plan[4] else ""
        ),
        **{
            f"full_scf_gpu_stage_{name}_calls": int(scf_stage_calls[idx]) if scf_stage_calls else ""
            for idx, name in enumerate(MOZYME_SCF_STAGE_NAMES)
        },
        **{
            f"full_scf_gpu_stage_{name}_ms": parse_float(scf_stage_ms[idx]) if scf_stage_ms else ""
            for idx, name in enumerate(MOZYME_SCF_STAGE_NAMES)
        },
        "full_scf_gpu_cnvgz_active_calls": (
            int(scf_cnvgz_activity[0]) if scf_cnvgz_activity else ""
        ),
        "full_scf_gpu_cnvgz_noop_calls": (
            int(scf_cnvgz_activity[1]) if scf_cnvgz_activity else ""
        ),
        "full_scf_gpu_isitsc_okscf": int(scf_isitsc[0]) if scf_isitsc else "",
        "full_scf_gpu_isitsc_iscf": int(scf_isitsc[1]) if scf_isitsc else "",
        "full_scf_gpu_isitsc_iemin": int(scf_isitsc[2]) if scf_isitsc else "",
        "full_scf_gpu_isitsc_iemax": int(scf_isitsc[3]) if scf_isitsc else "",
        "full_scf_gpu_isitsc_scf1": int(scf_isitsc[4]) if scf_isitsc else "",
        "full_scf_gpu_pls_supervisor_calls": int(scf_pls[0]) if scf_pls else "",
        "full_scf_gpu_pls_restart_required": int(scf_pls[1]) if scf_pls else 0,
        "full_scf_gpu_pls_history_count": int(scf_pls[2]) if scf_pls else "",
        "full_scf_gpu_pls_ovmax_delta": parse_float(scf_pls_delta[0]) if scf_pls_delta else "",
        "full_scf_gpu_pls_energy_delta": parse_float(scf_pls_delta[1]) if scf_pls_delta else "",
        "full_scf_gpu_pls_restart_reset_device_calls": (
            int(scf_pls_reset[0]) if scf_pls_reset else 0
        ),
        "full_scf_gpu_pls_restart_done": int(scf_pls_reset[1]) if scf_pls_reset else 0,
        "full_scf_gpu_final_iterations": scf_final_iterations if scf_final_iterations is not None else "",
        "full_scf_gpu_final_density_resident": int(scf_final_density_current),
        "full_scf_gpu_final_publication_done": (
            int(final_publication[0]) if final_publication else ""
        ),
        "full_scf_gpu_final_publication_arrays": (
            int(final_publication[1]) if final_publication else ""
        ),
        "full_scf_gpu_final_publication_bytes": (
            int(final_publication[2]) if final_publication else ""
        ),
        "full_scf_gpu_final_publication_cosmo": (
            int(final_publication[3]) if final_publication else ""
        ),
        "full_scf_gpu_olden_setup_only": int(scf_olden_setup_only),
        "full_scf_gpu_fillij_gpu_count_calls": len(scf_fillij_gpu_count_matches),
        "full_scf_gpu_fillij_gpu_fill_calls": len(scf_fillij_gpu_fill_matches),
        "full_scf_gpu_fillij_gpu_last_mpack": (
            int(scf_fillij_gpu_last[1]) if scf_fillij_gpu_last else ""
        ),
        "full_scf_gpu_fillij_gpu_last_n2elec": (
            int(scf_fillij_gpu_last[2]) if scf_fillij_gpu_last else ""
        ),
        "full_scf_gpu_fillij_gpu_last_ij_dim": (
            int(scf_fillij_gpu_last[3]) if scf_fillij_gpu_last else ""
        ),
        "full_scf_gpu_resident_fock_gpu_count_calls": len(resident_fock_gpu_count_matches),
        "full_scf_gpu_resident_fock_gpu_count_plan_id": (
            int(resident_fock_gpu_count[0]) if resident_fock_gpu_count else ""
        ),
        "full_scf_gpu_resident_fock_gpu_count_one": (
            int(resident_fock_gpu_count[1]) if resident_fock_gpu_count else ""
        ),
        "full_scf_gpu_resident_fock_gpu_count_pair": (
            int(resident_fock_gpu_count[2]) if resident_fock_gpu_count else ""
        ),
        "full_scf_gpu_resident_fock_gpu_count_pair4x1": (
            int(resident_fock_gpu_count[3]) if resident_fock_gpu_count else ""
        ),
        "full_scf_gpu_resident_fock_gpu_count_point": (
            int(resident_fock_gpu_count[4]) if resident_fock_gpu_count else ""
        ),
        "full_scf_gpu_resident_fock_gpu_count_full_coverage": (
            int(resident_fock_gpu_count[5]) if resident_fock_gpu_count else ""
        ),
        "full_scf_gpu_resident_fock_gpu_pack_calls": len(resident_fock_gpu_pack_matches),
        "full_scf_gpu_resident_fock_gpu_pack_plan_id": (
            int(resident_fock_gpu_pack[0]) if resident_fock_gpu_pack else ""
        ),
        "full_scf_gpu_resident_fock_gpu_pack_one": (
            int(resident_fock_gpu_pack[1]) if resident_fock_gpu_pack else ""
        ),
        "full_scf_gpu_resident_fock_gpu_pack_pair": (
            int(resident_fock_gpu_pack[2]) if resident_fock_gpu_pack else ""
        ),
        "full_scf_gpu_resident_fock_gpu_pack_pair4x1": (
            int(resident_fock_gpu_pack[3]) if resident_fock_gpu_pack else ""
        ),
        "full_scf_gpu_resident_fock_gpu_pack_point": (
            int(resident_fock_gpu_pack[4]) if resident_fock_gpu_pack else ""
        ),
        "full_scf_gpu_resident_fock_gpu_pack_full_coverage": (
            int(resident_fock_gpu_pack[5]) if resident_fock_gpu_pack else ""
        ),
        "full_scf_gpu_resident_fock_gpu_point_weight_calls": len(
            resident_fock_gpu_point_weight_matches
        ),
        "full_scf_gpu_resident_fock_gpu_point_weight_point": (
            int(resident_fock_gpu_point_weights[0])
            if resident_fock_gpu_point_weights
            else ""
        ),
        "full_scf_gpu_resident_fock_gpu_point_weight_max_abs_diff": (
            parse_float(resident_fock_gpu_point_weights[1])
            if resident_fock_gpu_point_weights
            else ""
        ),
        "full_scf_gpu_cpu_mozyme_setup_only_calls": len(scf_cpu_setup_matches),
        "full_scf_gpu_cpu_resident_fock_plan_setup_calls": len(resident_fock_cpu_plan_matches),
        "full_scf_gpu_cpu_resident_fock_plan_setup_plan_id": (
            int(resident_fock_cpu_plan[0]) if resident_fock_cpu_plan else ""
        ),
        "full_scf_gpu_cpu_resident_fock_plan_setup_one": (
            int(resident_fock_cpu_plan[1]) if resident_fock_cpu_plan else ""
        ),
        "full_scf_gpu_cpu_resident_fock_plan_setup_pair": (
            int(resident_fock_cpu_plan[2]) if resident_fock_cpu_plan else ""
        ),
        "full_scf_gpu_cpu_resident_fock_plan_setup_pair4x1": (
            int(resident_fock_cpu_plan[3]) if resident_fock_cpu_plan else ""
        ),
        "full_scf_gpu_cpu_resident_fock_plan_setup_point": (
            int(resident_fock_cpu_plan[4]) if resident_fock_cpu_plan else ""
        ),
        "full_scf_gpu_cpu_resident_fock_plan_setup_full_coverage": (
            int(resident_fock_cpu_plan[5]) if resident_fock_cpu_plan else ""
        ),
        "full_scf_gpu_host_commit_only_calls": len(scf_host_commit_matches),
        "full_scf_gpu_host_commit_phase": host_commit_phase,
        "full_scf_gpu_host_commit_arrays": host_commit_arrays,
        "full_scf_gpu_host_commit_bytes": host_commit_bytes,
        "full_scf_gpu_host_commit_cosmo": host_commit_cosmo,
        "full_scf_gpu_cpu_pinout_calls": len(cpu_pinout_matches),
        "full_scf_gpu_wall_ms": parse_float(scf_metrics[0]) if scf_metrics else "",
        "full_scf_gpu_density_max": parse_float(scf_metrics[1]) if scf_metrics else "",
        "full_scf_gpu_density_rms": parse_float(scf_metrics[2]) if scf_metrics else "",
        "full_scf_gpu_diagg_sumt": parse_float(scf_diagg[0]) if scf_diagg else "",
        "full_scf_gpu_diagg_sumb": parse_float(scf_diagg[1]) if scf_diagg else "",
        "full_scf_gpu_energy_total": parse_float(scf_energy_matches[-1]) if scf_energy_matches else "",
        "full_scf_gpu_cosmo_enabled": int(scf_cosmo[0]) if scf_cosmo else 0,
        "full_scf_gpu_cosmo_fock_calls": int(scf_cosmo[1]) if scf_cosmo else 0,
        "full_scf_gpu_cosmo_matvec_calls": int(scf_cosmo[2]) if scf_cosmo else 0,
        "full_scf_gpu_cosmo_cg_iterations": int(scf_cosmo[3]) if scf_cosmo else 0,
        "full_scf_gpu_cosmo_nps": int(scf_cosmo[4]) if scf_cosmo else 0,
        "full_scf_gpu_cosmo_lm61": int(scf_cosmo[5]) if scf_cosmo else 0,
        "full_scf_gpu_cosmo_pair_count": int(scf_cosmo[6]) if scf_cosmo else 0,
        "full_scf_gpu_cosmo_solv_energy": parse_float(scf_cosmo_energy[0]) if scf_cosmo_energy else "",
        "full_scf_gpu_cosmo_ediel": parse_float(scf_cosmo_energy[1]) if scf_cosmo_energy else "",
        "full_scf_gpu_cosmo_last_residual": parse_float(scf_cosmo_energy[2]) if scf_cosmo_energy else "",
        "full_scf_gpu_cosmo_cg_control_resident": int(scf_cosmo_cg[0]) if scf_cosmo_cg else 0,
        "full_scf_gpu_cosmo_cg_converged": int(scf_cosmo_cg[1]) if scf_cosmo_cg else 0,
        "full_scf_gpu_cosmo_cg_breakdown": int(scf_cosmo_cg[2]) if scf_cosmo_cg else 0,
        "full_scf_gpu_cosmo_cg_host_syncs": int(scf_cosmo_cg[3]) if scf_cosmo_cg else 0,
        "full_scf_gpu_cosmo_cg_target_tol": parse_float(scf_cosmo_cg[4]) if scf_cosmo_cg else "",
        "mozyme_makvec_gpu_success_calls": len(makvec_success_matches),
        "mozyme_makvec_gpu_existing_lmo_calls": len(makvec_existing_matches),
        "mozyme_makvec_gpu_existing_lmo_last_reason": (
            makvec_existing_matches[-1] if makvec_existing_matches else ""
        ),
        "mozyme_makvec_gpu_fallback_calls": len(makvec_fallback_matches),
        "mozyme_makvec_gpu_last_ms": parse_float(makvec_success_matches[-1]) if makvec_success_matches else "",
        "mozyme_makvec_gpu_last_fallback_reason": (
            makvec_fallback_matches[-1][0] if makvec_fallback_matches and makvec_fallback_matches[-1][0] else ""
        ),
        "mozyme_makvec_gpu_last_code": (
            int(makvec_fallback_matches[-1][1]) if makvec_fallback_matches and makvec_fallback_matches[-1][1] else ""
        ),
        "mozyme_relocal_gpu_success_calls": sum(
            1 for status, _kind, _extra in relocal_matches if status.lower() == "success"
        ),
        "mozyme_relocal_gpu_fallback_calls": sum(
            1 for status, _kind, _extra in relocal_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_relocal_gpu_occupied_success_calls": relocal_success_counts.get("OCCUPIED", 0),
        "mozyme_relocal_gpu_virtual_success_calls": relocal_success_counts.get("VIRTUAL", 0),
        "mozyme_reorth_gpu_success_calls": sum(
            1 for status, _extra in reorth_matches if status.lower() == "success"
        ),
        "mozyme_reorth_gpu_resident_success_calls": sum(
            1
            for status, extra in reorth_matches
            if status.lower() == "success" and marker_field_equals(extra, "resident", "1")
        ),
        "mozyme_reorth_gpu_fallback_calls": sum(
            1 for status, _extra in reorth_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_tidy_gpu_success_calls": sum(
            1 for status, _mode, _extra in tidy_matches if status.lower() == "success"
        ),
        "mozyme_tidy_gpu_occupied_success_calls": sum(
            1
            for status, mode, _extra in tidy_matches
            if status.lower() == "success" and mode.lower() == "occupied"
        ),
        "mozyme_tidy_gpu_virtual_success_calls": sum(
            1
            for status, mode, _extra in tidy_matches
            if status.lower() == "success" and mode.lower() == "virtual"
        ),
        "mozyme_tidy_gpu_selmos_success_calls": sum(
            1
            for status, _mode, extra in tidy_matches
            if status.lower() == "success" and marker_field_equals(extra, "selmos", "1")
        ),
        "mozyme_tidy_gpu_fallback_calls": sum(
            1
            for status, _mode, _extra in tidy_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_setupk_gpu_success_calls": sum(
            1 for status, _code, _ms, _extra in setupk_matches if status.lower() == "success"
        ),
        "mozyme_setupk_gpu_fallback_calls": sum(
            1 for status, _code, _ms, _extra in setupk_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_setupk_gpu_last_code": int(setupk_matches[-1][1]) if setupk_matches else "",
        "mozyme_setupk_gpu_last_ms": (
            parse_float(setupk_matches[-1][2]) if setupk_matches and setupk_matches[-1][2] else ""
        ),
        "mozyme_setupk_gpu_initial_setup_success_calls": len(setupk_initial_success_matches),
        "mozyme_setupk_gpu_initial_setup_fallback_calls": len(setupk_initial_fallback_matches),
        "mozyme_setupk_gpu_initial_setup_all_paths_calls": len(setupk_all_initial_matches),
        "mozyme_setupk_gpu_initial_setup_last_ms": (
            parse_float(setupk_initial_success_matches[-1][2])
            if setupk_initial_success_matches and setupk_initial_success_matches[-1][2]
            else ""
        ),
        "mozyme_cnvgz_gpu_success_calls": sum(1 for status, _code, _pmax, _rms, _ms in cnvgz_matches if status.lower() == "success"),
        "mozyme_cnvgz_gpu_fallback_calls": sum(
            1 for status, _code, _pmax, _rms, _ms in cnvgz_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_cnvgz_gpu_last_code": int(cnvgz_matches[-1][1]) if cnvgz_matches else "",
        "mozyme_cnvgz_gpu_last_pmax": parse_float(cnvgz_matches[-1][2]) if cnvgz_matches and cnvgz_matches[-1][2] else "",
        "mozyme_cnvgz_gpu_last_rms": parse_float(cnvgz_matches[-1][3]) if cnvgz_matches and cnvgz_matches[-1][3] else "",
        "mozyme_cnvgz_gpu_last_ms": parse_float(cnvgz_matches[-1][4]) if cnvgz_matches and cnvgz_matches[-1][4] else "",
        "mozyme_helecz_gpu_success_calls": sum(1 for status, _code, _energy, _ms in helecz_matches if status.lower() == "success"),
        "mozyme_helecz_gpu_fallback_calls": sum(
            1 for status, _code, _energy, _ms in helecz_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_helecz_gpu_last_code": int(helecz_matches[-1][1]) if helecz_matches else "",
        "mozyme_helecz_gpu_last_energy": parse_float(helecz_matches[-1][2]) if helecz_matches and helecz_matches[-1][2] else "",
        "mozyme_helecz_gpu_last_ms": parse_float(helecz_matches[-1][3]) if helecz_matches and helecz_matches[-1][3] else "",
        "mozyme_eimp_gpu_success_calls": sum(1 for status, _code, _pairs, _ms in eimp_matches if status.lower() == "success"),
        "mozyme_eimp_gpu_fallback_calls": sum(
            1 for status, _code, _pairs, _ms in eimp_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_eimp_gpu_last_code": int(eimp_matches[-1][1]) if eimp_matches else "",
        "mozyme_eimp_gpu_last_pairs": int(eimp_matches[-1][2]) if eimp_matches and eimp_matches[-1][2] else "",
        "mozyme_eimp_gpu_last_ms": parse_float(eimp_matches[-1][3]) if eimp_matches and eimp_matches[-1][3] else "",
        "mozyme_diagg1_aocc_gpu_success_calls": sum(
            1 for status, _code, _terms, _ms in diagg1_aocc_matches if status.lower() == "success"
        ),
        "mozyme_diagg1_aocc_gpu_fallback_calls": sum(
            1 for status, _code, _terms, _ms in diagg1_aocc_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_diagg1_aocc_gpu_last_code": int(diagg1_aocc_matches[-1][1]) if diagg1_aocc_matches else "",
        "mozyme_diagg1_aocc_gpu_last_terms": (
            int(diagg1_aocc_matches[-1][2]) if diagg1_aocc_matches and diagg1_aocc_matches[-1][2] else ""
        ),
        "mozyme_diagg1_aocc_gpu_last_ms": (
            parse_float(diagg1_aocc_matches[-1][3]) if diagg1_aocc_matches and diagg1_aocc_matches[-1][3] else ""
        ),
        "mozyme_diagg1_avir_gpu_success_calls": sum(
            1 for status, _code, _terms, _ms in diagg1_avir_matches if status.lower() == "success"
        ),
        "mozyme_diagg1_avir_gpu_fallback_calls": sum(
            1 for status, _code, _terms, _ms in diagg1_avir_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_diagg1_avir_gpu_last_code": int(diagg1_avir_matches[-1][1]) if diagg1_avir_matches else "",
        "mozyme_diagg1_avir_gpu_last_terms": (
            int(diagg1_avir_matches[-1][2]) if diagg1_avir_matches and diagg1_avir_matches[-1][2] else ""
        ),
        "mozyme_diagg1_avir_gpu_last_ms": (
            parse_float(diagg1_avir_matches[-1][3]) if diagg1_avir_matches and diagg1_avir_matches[-1][3] else ""
        ),
        "mozyme_diagg1_construct_gpu_success_calls": sum(
            1 for status, _code, _nij, _sumt, _tiny, _ms in diagg1_construct_matches
            if status.lower() == "success"
        ),
        "mozyme_diagg1_construct_gpu_fallback_calls": sum(
            1 for status, _code, _nij, _sumt, _tiny, _ms in diagg1_construct_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_diagg1_construct_gpu_last_code": (
            int(diagg1_construct_matches[-1][1]) if diagg1_construct_matches else ""
        ),
        "mozyme_diagg1_construct_gpu_last_nij": (
            int(diagg1_construct_matches[-1][2])
            if diagg1_construct_matches and diagg1_construct_matches[-1][2]
            else ""
        ),
        "mozyme_diagg1_construct_gpu_last_sumt": (
            parse_float(diagg1_construct_matches[-1][3])
            if diagg1_construct_matches and diagg1_construct_matches[-1][3]
            else ""
        ),
        "mozyme_diagg1_construct_gpu_last_tiny": (
            parse_float(diagg1_construct_matches[-1][4])
            if diagg1_construct_matches and diagg1_construct_matches[-1][4]
            else ""
        ),
        "mozyme_diagg1_construct_gpu_last_ms": (
            parse_float(diagg1_construct_matches[-1][5])
            if diagg1_construct_matches and diagg1_construct_matches[-1][5]
            else ""
        ),
        "mozyme_diagg2_rotprep_gpu_success_calls": sum(
            1 for status, _code, _active, _ms in diagg2_rotprep_matches if status.lower() == "success"
        ),
        "mozyme_diagg2_rotprep_gpu_fallback_calls": sum(
            1 for status, _code, _active, _ms in diagg2_rotprep_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_diagg2_rotprep_gpu_last_code": int(diagg2_rotprep_matches[-1][1]) if diagg2_rotprep_matches else "",
        "mozyme_diagg2_rotprep_gpu_last_active": (
            int(diagg2_rotprep_matches[-1][2])
            if diagg2_rotprep_matches and diagg2_rotprep_matches[-1][2]
            else ""
        ),
        "mozyme_diagg2_rotprep_gpu_last_ms": (
            parse_float(diagg2_rotprep_matches[-1][3])
            if diagg2_rotprep_matches and diagg2_rotprep_matches[-1][3]
            else ""
        ),
        "mozyme_diagg2_rotate_gpu_success_calls": sum(
            1 for status, _code, _nrej, _sumb, _ms in diagg2_rotate_matches if status.lower() == "success"
        ),
        "mozyme_diagg2_rotate_gpu_fallback_calls": sum(
            1 for status, _code, _nrej, _sumb, _ms in diagg2_rotate_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_diagg2_rotate_gpu_last_code": int(diagg2_rotate_matches[-1][1]) if diagg2_rotate_matches else "",
        "mozyme_diagg2_rotate_gpu_last_nrej": (
            int(diagg2_rotate_matches[-1][2])
            if diagg2_rotate_matches and diagg2_rotate_matches[-1][2]
            else ""
        ),
        "mozyme_diagg2_rotate_gpu_last_sumb": (
            parse_float(diagg2_rotate_matches[-1][3])
            if diagg2_rotate_matches and diagg2_rotate_matches[-1][3]
            else ""
        ),
        "mozyme_diagg2_rotate_gpu_last_ms": (
            parse_float(diagg2_rotate_matches[-1][4])
            if diagg2_rotate_matches and diagg2_rotate_matches[-1][4]
            else ""
        ),
        "mozyme_isitsc_gpu_success_calls": sum(
            1 for status, _code, _okscf, _ms in isitsc_matches if status.lower() == "success"
        ),
        "mozyme_isitsc_gpu_fallback_calls": sum(
            1 for status, _code, _okscf, _ms in isitsc_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "mozyme_isitsc_gpu_last_code": int(isitsc_matches[-1][1]) if isitsc_matches else "",
        "mozyme_isitsc_gpu_last_okscf": isitsc_matches[-1][2].upper() if isitsc_matches and isitsc_matches[-1][2] else "",
        "mozyme_isitsc_gpu_last_ms": (
            parse_float(isitsc_matches[-1][3]) if isitsc_matches and isitsc_matches[-1][3] else ""
        ),
        "mozyme_fock1_gpu_success_seen": int(any(kind == "1" and status.lower() == "success" for kind, status in fock_markers)),
        "mozyme_fock2_gpu_success_seen": int(any(kind == "2" and status.lower() == "success" for kind, status in fock_markers)),
        "mozyme_fock1_gpu_attempt_seen": int(any(kind == "1" and status.lower() == "attempt" for kind, status in fock_markers)),
        "mozyme_fock2_gpu_attempt_seen": int(any(kind == "2" and status.lower() == "attempt" for kind, status in fock_markers)),
        "mozyme_fock1_gpu_fallback_seen": int(any(kind == "1" and status.lower() == "fallback" for kind, status in fock_markers)),
        "mozyme_fock2_gpu_fallback_seen": int(any(kind == "2" and status.lower() == "fallback" for kind, status in fock_markers)),
        "mozyme_fock1_batch_gpu_attempt_calls": sum(
            1 for status, _code, _tasks, _pairs in fock1_batch_markers if status.lower() == "attempt"
        ),
        "mozyme_fock1_batch_gpu_attempt_tasks": sum(
            int(tasks) for status, _code, tasks, _pairs in fock1_batch_markers if status.lower() == "attempt"
        ),
        "mozyme_fock1_batch_gpu_attempt_pairs": sum(
            int(pairs) for status, _code, _tasks, pairs in fock1_batch_markers if status.lower() == "attempt"
        ),
        "mozyme_fock1_batch_gpu_success_calls": sum(
            1 for status, _code, _tasks, _pairs in fock1_batch_markers if status.lower() == "success"
        ),
        "mozyme_fock1_batch_gpu_success_tasks": sum(
            int(tasks) for status, _code, tasks, _pairs in fock1_batch_markers if status.lower() == "success"
        ),
        "mozyme_fock1_batch_gpu_success_pairs": sum(
            int(pairs) for status, _code, _tasks, pairs in fock1_batch_markers if status.lower() == "success"
        ),
        "mozyme_fock1_batch_gpu_fallback_calls": sum(
            1 for status, _code, _tasks, _pairs in fock1_batch_markers if status.lower() == "fallback"
        ),
        "mozyme_fock1_batch_gpu_fallback_tasks": sum(
            int(tasks) for status, _code, tasks, _pairs in fock1_batch_markers if status.lower() == "fallback"
        ),
        "mozyme_fock1_batch_gpu_fallback_pairs": sum(
            int(pairs) for status, _code, _tasks, pairs in fock1_batch_markers if status.lower() == "fallback"
        ),
        "mozyme_fock2_4x1_batch_gpu_attempt_calls": sum(
            1 for status, _code, _tasks in fock2_4x1_batch_markers if status.lower() == "attempt"
        ),
        "mozyme_fock2_4x1_batch_gpu_attempt_tasks": sum(
            int(tasks) for status, _code, tasks in fock2_4x1_batch_markers if status.lower() == "attempt"
        ),
        "mozyme_fock2_4x1_batch_gpu_success_calls": sum(
            1 for status, _code, _tasks in fock2_4x1_batch_markers if status.lower() == "success"
        ),
        "mozyme_fock2_4x1_batch_gpu_success_tasks": sum(
            int(tasks) for status, _code, tasks in fock2_4x1_batch_markers if status.lower() == "success"
        ),
        "mozyme_fock2_4x1_batch_gpu_fallback_calls": sum(
            1 for status, _code, _tasks in fock2_4x1_batch_markers if status.lower() == "fallback"
        ),
        "mozyme_fock2_4x1_batch_gpu_fallback_tasks": sum(
            int(tasks) for status, _code, tasks in fock2_4x1_batch_markers if status.lower() == "fallback"
        ),
        "mozyme_sparse_fock_setup_calls": len(sparse_fock_setup_matches),
        "mozyme_sparse_fock_setup_one_tasks": sum(int(match[0]) for match in sparse_fock_setup_matches),
        "mozyme_sparse_fock_setup_pair_tasks": sum(int(match[1]) for match in sparse_fock_setup_matches),
        "mozyme_sparse_fock_setup_4x1_tasks": sum(int(match[2]) for match in sparse_fock_setup_matches),
        "mozyme_sparse_fock_setup_point_tasks": sum(int(match[3] or 0) for match in sparse_fock_setup_matches),
        "mozyme_sparse_fock_setup_point_dipole_tasks": sum(
            int(match[4] or 0) for match in sparse_fock_setup_matches
        ),
        "mozyme_sparse_fock_setup_point_monopole_tasks": sum(
            int(match[5] or 0) for match in sparse_fock_setup_matches
        ),
        "mozyme_sparse_fock_run_calls": len(sparse_fock_run_matches),
        "mozyme_sparse_fock_run_one_tasks": sum(int(match[0]) for match in sparse_fock_run_matches),
        "mozyme_sparse_fock_run_pair_tasks": sum(int(match[1]) for match in sparse_fock_run_matches),
        "mozyme_sparse_fock_run_4x1_tasks": sum(int(match[2]) for match in sparse_fock_run_matches),
        "mozyme_sparse_fock_run_point_tasks": sum(int(match[3] or 0) for match in sparse_fock_run_matches),
        "mozyme_sparse_fock_run_point_dipole_tasks": sum(
            int(match[4] or 0) for match in sparse_fock_run_matches
        ),
        "mozyme_sparse_fock_run_point_monopole_tasks": sum(
            int(match[5] or 0) for match in sparse_fock_run_matches
        ),
        "mozyme_sparse_fock_run_zero_work_calls": sum(
            1
            for match in sparse_fock_run_matches
            if int(match[0]) + int(match[1]) + int(match[2]) + int(match[3] or 0) <= 0
        ),
        "mozyme_sparse_fock_run_ms": sum(float(match[6]) for match in sparse_fock_run_matches),
        "mozyme_resident_fock_coverage_mode": int(resident_coverage[0]) if resident_coverage else "",
        "mozyme_resident_fock_coverage_use_nijbo": resident_coverage[1].upper() if resident_coverage else "",
        "mozyme_resident_fock_real_pairs": int(resident_coverage[2]) if resident_coverage else "",
        "mozyme_resident_fock_gpu_real_pairs": int(resident_coverage[3]) if resident_coverage else "",
        "mozyme_resident_fock_cpu_real_pairs": int(resident_coverage[4]) if resident_coverage else "",
        "mozyme_resident_fock_inactive_real_pairs": int(resident_coverage[5]) if resident_coverage else "",
        "mozyme_resident_fock_fallback_pairs": int(resident_fallback[0]) if resident_fallback else "",
        "mozyme_resident_fock_basis_limit_fallback_pairs": (
            int(resident_fallback[1]) if resident_fallback else ""
        ),
        "mozyme_resident_fock_direct_basis_fallback_pairs": (
            int(resident_fallback[2] or 0) if resident_fallback else ""
        ),
        "mozyme_resident_fock_other_fallback_pairs": int(resident_fallback[3]) if resident_fallback else "",
        "mozyme_resident_fock_point_pairs": int(resident_point_coverage[0]) if resident_point_coverage else "",
        "mozyme_resident_fock_gpu_point_pairs": int(resident_point_coverage[1]) if resident_point_coverage else "",
        "mozyme_resident_fock_cpu_point_pairs": (
            max(
                int(resident_point_coverage[2]) if resident_point_coverage else 0,
                resident_cpu_point_pair_max if resident_cpu_point_pair_max is not None else 0,
            )
            if resident_point_coverage or resident_cpu_point_pair_max is not None
            else ""
        ),
        "mozyme_resident_fock_cpu_point_pair_fatal_max": (
            resident_cpu_point_pair_max if resident_cpu_point_pair_max is not None else ""
        ),
        "mozyme_resident_fock_basis_limit_point_fallback_pairs": (
            int(resident_point_coverage[3]) if resident_point_coverage else ""
        ),
        "mozyme_resident_fock_direct_basis_point_fallback_pairs": (
            int(resident_point_coverage[4] or 0) if resident_point_coverage else ""
        ),
        "mozyme_resident_fock_other_point_fallback_pairs": (
            int(resident_point_coverage[5]) if resident_point_coverage else ""
        ),
        "density_calls": len(density_calls) if density_calls else "",
        "density_gpu_syrk_calls": sum(int(match[2]) for match in density_calls) if density_calls else "",
        "density_gpu_gemm_calls": sum(int(match[3]) for match in density_calls) if density_calls else "",
        "density_skipped_diag_blocks": sum(int(match[0]) for match in density_skips) if density_skips else "",
        "density_skipped_offdiag_blocks": sum(int(match[1]) for match in density_skips) if density_skips else "",
        "density_cpu_diag_blocks": sum(int(match[2]) for match in density_skips) if density_skips else "",
        "density_cpu_offdiag_blocks": sum(int(match[3]) for match in density_skips) if density_skips else "",
        "density_max_diag_block": max((int(match[0]) for match in density_maxima), default=""),
        "density_max_offdiag_j": max((int(match[1]) for match in density_maxima), default=""),
        "density_max_offdiag_k": max((int(match[2]) for match in density_maxima), default=""),
        "density_batch_gpu_success_calls": sum(
            1 for status, _code, _mode, _blocks, _terms, _ms in density_batch_matches if status.lower() == "success"
        ),
        "density_batch_gpu_fallback_calls": sum(
            1 for status, _code, _mode, _blocks, _terms, _ms in density_batch_matches
            if status.lower() in {"fallback_cpu", "strict_abort"}
        ),
        "density_batch_gpu_last_code": int(density_batch_matches[-1][1]) if density_batch_matches else "",
        "density_batch_gpu_last_mode": (
            int(density_batch_matches[-1][2]) if density_batch_matches and density_batch_matches[-1][2] else ""
        ),
        "density_batch_gpu_last_blocks": (
            int(density_batch_matches[-1][3]) if density_batch_matches and density_batch_matches[-1][3] else ""
        ),
        "density_batch_gpu_last_terms": (
            int(density_batch_matches[-1][4]) if density_batch_matches and density_batch_matches[-1][4] else ""
        ),
        "density_batch_gpu_last_ms": (
            parse_float(density_batch_matches[-1][5])
            if density_batch_matches and density_batch_matches[-1][5]
            else ""
        ),
    }


def parse_mozyme_section_times(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in MOZYME_SECTION_RE.finditer(text):
        name = match.group(1) or match.group(2) or ""
        calls = int(match.group(3))
        ms = parse_float(match.group(4))
        if ms is None:
            continue
        rows.append(
            {
                "name": name,
                "calls": calls,
                "ms": ms,
                "ms_per_call": ms / calls if calls > 0 else "",
            }
        )
    return rows


def mozyme_cpu_mutating_section_stats(section_rows: Any) -> tuple[list[str], int, float]:
    if not isinstance(section_rows, list):
        return [], 0, 0.0

    names: set[str] = set()
    total_calls = 0
    total_ms = 0.0
    for row in section_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        calls = parse_int_value(row.get("calls")) or 0
        if calls > 0 and name in MOZYME_CPU_MUTATING_SECTION_NAMES:
            names.add(name)
            total_calls += calls
            total_ms += parse_float(row.get("ms")) or 0.0
    return sorted(names), total_calls, total_ms


def mozyme_relocal_success_types(text: str) -> set[str]:
    return set(mozyme_relocal_success_counts(text))


def mozyme_relocal_success_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status, kind, _extra in MOZYME_RELOCAL_RE.findall(text):
        if status.lower() != "success":
            continue
        key = kind.upper()
        counts[key] = counts.get(key, 0) + 1
    return counts


def mozyme_disallowed_strict_section_stats(
    section_rows: Any, text: str = ""
) -> tuple[list[str], int, float]:
    if not isinstance(section_rows, list):
        return [], 0, 0.0

    allowed_names = set(MOZYME_STRICT_PROOF_ALLOWED_SECTION_NAMES)
    relocal_counts = mozyme_relocal_success_counts(text)
    tidy_matches = MOZYME_TIDY_RE.findall(text)
    tidy_success_counts = {
        "occupied": sum(
            1 for status, mode, _extra in tidy_matches
            if status.lower() == "success" and mode.lower() == "occupied"
        ),
        "virtual": sum(
            1 for status, mode, _extra in tidy_matches
            if status.lower() == "success" and mode.lower() == "virtual"
        ),
    }

    names: set[str] = set()
    total_calls = 0
    total_ms = 0.0
    for row in section_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        calls = parse_int_value(row.get("calls")) or 0
        allowed = name in allowed_names
        if name == "iter_reloc_occ":
            allowed = relocal_counts.get("OCCUPIED", 0) >= calls
        elif name == "iter_reloc_virt":
            allowed = relocal_counts.get("VIRTUAL", 0) >= calls
        elif name == "iter_tidy_occ":
            allowed = tidy_success_counts.get("occupied", 0) >= calls
        elif name == "iter_tidy_virt":
            allowed = tidy_success_counts.get("virtual", 0) >= calls
        if calls > 0 and not allowed:
            names.add(name)
            total_calls += calls
            total_ms += parse_float(row.get("ms")) or 0.0
    return sorted(names), total_calls, total_ms


def mozyme_final_reorth_section_names(section_rows: Any) -> list[str]:
    if not isinstance(section_rows, list):
        return []
    names: set[str] = set()
    for row in section_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        calls = parse_int_value(row.get("calls")) or 0
        if calls > 0 and name in MOZYME_FINAL_REORTH_SECTION_NAMES:
            names.add(name)
    return sorted(names)


def proof_identity_violation_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []

    def require_hex(key: str, length: int) -> None:
        value = str(row.get(key) or "")
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            reasons.append(f"{key} is not a {length}-character lowercase hex digest")

    require_hex("source_zip_sha256", 64)
    require_hex("source_manifest_sha256", 64)
    require_hex("source_manifest_file_sha256", 64)
    require_hex("source_features_file_sha256", 64)
    require_hex("source_provenance_file_sha256", 64)
    require_hex("source_provenance_manifest_sha256", 64)
    require_hex("source_features_manifest_sha256", 64)
    require_hex("source_provenance_marker_contract_sha256", 64)
    require_hex("source_features_marker_contract_sha256", 64)
    require_hex("source_git_commit", 40)
    require_hex("source_provenance_git_commit", 40)
    require_hex("mopac_executable_sha256", 64)
    if parse_int_value(row.get("source_zip_verified")) != 1:
        reasons.append("source zip path was not present and hash-verified")
    if parse_int_value(row.get("source_metadata_from_zip")) != 1:
        reasons.append("packaged source metadata was not verified from the uploaded zip")
    source_zip_env = str(row.get("source_zip_sha256_env") or "")
    if source_zip_env and source_zip_env != str(row.get("source_zip_sha256") or ""):
        reasons.append("source_zip_sha256 does not match MOPAC_COLAB_SOURCE_ZIP_SHA256")
    if parse_int_value(row.get("source_metadata_present")) != 1:
        reasons.append("packaged source metadata files were not present")
    if parse_int_value(row.get("source_metadata_valid")) != 1:
        detail = str(row.get("source_metadata_violation_reasons") or "").strip()
        reasons.append(
            "packaged source metadata did not validate"
            + (f": {detail}" if detail else "")
        )
    if str(row.get("source_manifest_sha256") or "") != str(
        row.get("source_manifest_file_sha256") or ""
    ):
        reasons.append("source_manifest_sha256 does not match packaged manifest file hash")
    if str(row.get("source_provenance_manifest_sha256") or "") != str(
        row.get("source_manifest_file_sha256") or ""
    ):
        reasons.append("source provenance manifest hash does not match packaged manifest")
    if str(row.get("source_features_manifest_sha256") or "") != str(
        row.get("source_manifest_file_sha256") or ""
    ):
        reasons.append("source feature manifest hash does not match packaged manifest")
    if str(row.get("source_git_commit") or "") != str(
        row.get("source_provenance_git_commit") or ""
    ):
        reasons.append("source_git_commit does not match packaged provenance")
    dirty = str(row.get("source_git_dirty") or "").lower()
    if dirty not in {"true", "false"}:
        reasons.append("source_git_dirty is not true/false")
    provenance_dirty = str(row.get("source_provenance_git_dirty") or "").lower()
    if provenance_dirty not in {"true", "false"}:
        reasons.append("source_provenance_git_dirty is not true/false")
    elif dirty != provenance_dirty:
        reasons.append("source_git_dirty does not match packaged provenance")
    if dirty == "true":
        require_hex("source_dirty_status_sha256", 64)
        require_hex("source_provenance_dirty_status_sha256", 64)
        if str(row.get("source_dirty_status_sha256") or "") != str(
            row.get("source_provenance_dirty_status_sha256") or ""
        ):
            reasons.append("source_dirty_status_sha256 does not match packaged provenance")
    if not str(row.get("source_generated_at_utc") or "").strip():
        reasons.append("source_generated_at_utc was not recorded")
    if str(row.get("source_generated_at_utc") or "") != str(
        row.get("source_provenance_generated_at_utc") or ""
    ):
        reasons.append("source_generated_at_utc does not match packaged provenance")
    if str(row.get("source_provenance_contract_version") or "") != MOPAC_GPU_READINESS_CONTRACT_VERSION:
        reasons.append("source provenance contract version does not match current contract")
    if str(row.get("source_features_contract_version") or "") != MOPAC_GPU_READINESS_CONTRACT_VERSION:
        reasons.append("source feature contract version does not match current contract")
    if str(row.get("source_provenance_marker_contract_sha256") or "") != SOURCE_MARKER_CONTRACT_SHA256:
        reasons.append("source provenance marker contract sha does not match current contract")
    if str(row.get("source_features_marker_contract_sha256") or "") != SOURCE_MARKER_CONTRACT_SHA256:
        reasons.append("source feature marker contract sha does not match current contract")
    if str(row.get("source_features_feature_set") or "") != MOPAC_GPU_FEATURE_SET:
        reasons.append("source feature set does not match current feature set")
    source_manifest_entries = parse_int_value(row.get("source_manifest_entry_count"))
    source_manifest_verified = parse_int_value(row.get("source_manifest_verified_files"))
    if source_manifest_entries is None or source_manifest_entries <= 0:
        reasons.append("source manifest did not contain verified file entries")
    elif source_manifest_verified != source_manifest_entries:
        reasons.append(
            "source manifest verified file count does not match entry count "
            f"({source_manifest_verified} != {source_manifest_entries})"
        )
    if (parse_int_value(row.get("source_manifest_missing_file_count")) or 0) != 0:
        reasons.append("source manifest references missing source file(s)")
    if (parse_int_value(row.get("source_manifest_hash_mismatch_count")) or 0) != 0:
        reasons.append("source manifest file hash mismatch was detected")
    critical_count = parse_int_value(row.get("source_critical_file_count"))
    critical_verified = parse_int_value(row.get("source_critical_files_verified"))
    required_critical_count = parse_int_value(row.get("source_required_critical_file_count"))
    required_critical_missing = parse_int_value(row.get("source_required_critical_missing_count"))
    if critical_count is None or critical_count <= 0:
        reasons.append("source feature manifest did not list critical files")
    elif critical_verified != critical_count:
        reasons.append(
            "critical source file verification count does not match "
            f"({critical_verified} != {critical_count})"
        )
    if required_critical_count != len(REQUIRED_CRITICAL_SOURCE_FILES):
        reasons.append(
            "required critical source file count does not match current contract "
            f"({required_critical_count} != {len(REQUIRED_CRITICAL_SOURCE_FILES)})"
        )
    if (
        critical_count is not None
        and required_critical_count is not None
        and critical_count < required_critical_count
    ):
        reasons.append(
            "source feature manifest listed fewer critical files than required "
            f"({critical_count} < {required_critical_count})"
        )
    if required_critical_missing != 0:
        reasons.append("required critical source files are missing from the feature manifest")
    if str(row.get("cmake_gpu_bool") or "").upper() != "ON":
        reasons.append(f"cmake_gpu_bool is not ON ({row.get('cmake_gpu_bool')!r})")
    if not str(row.get("cmake_cuda_architectures") or row.get("cmake_cuda_archs") or "").strip():
        reasons.append("CUDA architecture identity was not recorded")
    return reasons


def den_output_artifacts(row: dict[str, Any]) -> list[str]:
    output_files = str(row.get("output_files") or "")
    if not output_files:
        return []
    return [
        item
        for item in output_files.split(";")
        if item and Path(item).suffix.lower() == ".den"
    ]


def validate_strict_host_route_contract(row: dict[str, Any], reasons: list[str]) -> None:
    if (parse_int_value(row.get("full_scf_gpu_olden_setup_only")) or 0) != 0:
        reasons.append("OLDEN/OLDENS host LMO restore marker appeared in strict GPU proof")
    fillij_count_calls = parse_int_value(row.get("full_scf_gpu_fillij_gpu_count_calls"))
    if fillij_count_calls is None:
        reasons.append("full_scf_gpu_fillij_gpu_count_calls was not reported")
    elif fillij_count_calls <= 0:
        reasons.append("strict resident SCF did not report GPU fillij count setup")
    fillij_fill_calls = parse_int_value(row.get("full_scf_gpu_fillij_gpu_fill_calls"))
    if fillij_fill_calls is None:
        reasons.append("full_scf_gpu_fillij_gpu_fill_calls was not reported")
    elif fillij_fill_calls <= 0:
        reasons.append("strict resident SCF did not report GPU fillij nijbo setup")
    resident_count_calls = parse_int_value(
        row.get("full_scf_gpu_resident_fock_gpu_count_calls")
    )
    if resident_count_calls is None:
        reasons.append("full_scf_gpu_resident_fock_gpu_count_calls was not reported")
    elif resident_count_calls <= 0:
        reasons.append("strict resident SCF did not report GPU resident-Fock count setup")
    resident_pack_calls = parse_int_value(
        row.get("full_scf_gpu_resident_fock_gpu_pack_calls")
    )
    if resident_pack_calls is None:
        reasons.append("full_scf_gpu_resident_fock_gpu_pack_calls was not reported")
    elif resident_pack_calls <= 0:
        reasons.append("strict resident SCF did not report GPU resident-Fock plan pack")
    resident_point_count_keys = (
        "full_scf_gpu_resident_fock_gpu_count_point",
        "full_scf_gpu_resident_fock_gpu_pack_point",
        "full_scf_gpu_cpu_resident_fock_plan_setup_point",
        "mozyme_fock_plan_point_charge_pairs",
        "mozyme_sparse_fock_setup_point_tasks",
        "mozyme_resident_fock_point_pairs",
        "mozyme_resident_fock_gpu_point_pairs",
    )
    resident_point_counts = [
        value
        for value in (
            parse_int_value(row.get(key)) for key in resident_point_count_keys
        )
        if value is not None and value > 0
    ]
    resident_point_count = max(resident_point_counts, default=0)
    resident_point_run_tasks = parse_int_value(
        row.get("mozyme_sparse_fock_run_point_tasks")
    )
    resident_point_work_present = (
        resident_point_count > 0
        or (resident_point_run_tasks is not None and resident_point_run_tasks > 0)
    )
    point_weight_calls = parse_int_value(
        row.get("full_scf_gpu_resident_fock_gpu_point_weight_calls")
    )
    point_weight_point = parse_int_value(
        row.get("full_scf_gpu_resident_fock_gpu_point_weight_point")
    )
    if resident_point_work_present:
        if point_weight_calls is None:
            reasons.append(
                "full_scf_gpu_resident_fock_gpu_point_weight_calls was not reported"
            )
        elif point_weight_calls <= 0:
            reasons.append(
                "strict resident SCF did not report GPU resident-Fock point weights"
            )
        if point_weight_point is None:
            reasons.append(
                "full_scf_gpu_resident_fock_gpu_point_weight_point was not reported"
            )
        elif point_weight_point <= 0:
            reasons.append(
                "full_scf_gpu_resident_fock_gpu_point_weight_point must be positive"
            )
        elif resident_point_count > 0 and point_weight_point < resident_point_count:
            reasons.append(
                "GPU resident-Fock point-weight coverage is incomplete "
                f"({point_weight_point}/{resident_point_count})"
            )
    cpu_setup_calls = parse_int_value(row.get("full_scf_gpu_cpu_mozyme_setup_only_calls"))
    if cpu_setup_calls is None:
        reasons.append("full_scf_gpu_cpu_mozyme_setup_only_calls was not reported")
    elif cpu_setup_calls != 0:
        reasons.append(
            "CPU MOZYME array setup marker appeared in strict GPU proof "
            f"{cpu_setup_calls} time(s)"
        )
    cpu_plan_calls = parse_int_value(row.get("full_scf_gpu_cpu_resident_fock_plan_setup_calls"))
    if cpu_plan_calls is None:
        reasons.append("full_scf_gpu_cpu_resident_fock_plan_setup_calls was not reported")
    elif cpu_plan_calls != 0:
        reasons.append(
            "CPU resident Fock plan construction marker appeared in strict GPU proof "
            f"{cpu_plan_calls} time(s)"
        )
    pinout_calls = parse_int_value(row.get("full_scf_gpu_cpu_pinout_calls"))
    if pinout_calls is None:
        reasons.append("full_scf_gpu_cpu_pinout_calls was not reported")
    elif pinout_calls != 0:
        reasons.append(f"CPU pinout host I/O marker appeared {pinout_calls} time(s)")
    den_outputs = den_output_artifacts(row)
    if den_outputs:
        reasons.append("strict GPU proof produced .den host checkpoint artifact(s): " + ";".join(den_outputs))


def validate_full_scf_host_commit_contract(row: dict[str, Any], reasons: list[str]) -> None:
    host_commit_calls = parse_int_value(row.get("full_scf_gpu_host_commit_only_calls"))
    if host_commit_calls is None:
        reasons.append("full_scf_gpu_host_commit_only_calls was not reported")
    elif host_commit_calls != 1:
        reasons.append(
            "resident-SCF final host publication did not emit host_commit_only=1 "
            f"exactly once ({host_commit_calls})"
        )
    host_commit_arrays = parse_int_value(row.get("full_scf_gpu_host_commit_arrays"))
    host_commit_phase = str(row.get("full_scf_gpu_host_commit_phase") or "")
    if host_commit_phase != "final_publication":
        reasons.append(
            "resident-SCF host publication did not report phase=final_publication "
            f"({host_commit_phase!r})"
        )
    if host_commit_arrays is None:
        reasons.append("full_scf_gpu_host_commit_arrays was not reported")
    elif host_commit_arrays <= 0:
        reasons.append(f"full_scf_gpu_host_commit_arrays is not positive ({host_commit_arrays})")
    host_commit_bytes = parse_int_value(row.get("full_scf_gpu_host_commit_bytes"))
    if host_commit_bytes is None:
        reasons.append("full_scf_gpu_host_commit_bytes was not reported")
    elif host_commit_bytes <= 0:
        reasons.append(f"full_scf_gpu_host_commit_bytes is not positive ({host_commit_bytes})")
    host_commit_cosmo = parse_int_value(row.get("full_scf_gpu_host_commit_cosmo"))
    if host_commit_cosmo is None:
        reasons.append("full_scf_gpu_host_commit_cosmo was not reported")
    final_publication_done = parse_int_value(row.get("full_scf_gpu_final_publication_done"))
    if final_publication_done is None:
        reasons.append("full_scf_gpu_final_publication_done was not reported")
    elif final_publication_done != 1:
        reasons.append(
            "typed final publication marker did not report done=1 "
            f"({final_publication_done})"
        )
    final_publication_arrays = parse_int_value(row.get("full_scf_gpu_final_publication_arrays"))
    if final_publication_arrays is None:
        reasons.append("full_scf_gpu_final_publication_arrays was not reported")
    elif final_publication_arrays <= 0:
        reasons.append(
            f"full_scf_gpu_final_publication_arrays is not positive ({final_publication_arrays})"
        )
    final_publication_bytes = parse_int_value(row.get("full_scf_gpu_final_publication_bytes"))
    if final_publication_bytes is None:
        reasons.append("full_scf_gpu_final_publication_bytes was not reported")
    elif final_publication_bytes <= 0:
        reasons.append(
            f"full_scf_gpu_final_publication_bytes is not positive ({final_publication_bytes})"
        )
    final_publication_cosmo = parse_int_value(row.get("full_scf_gpu_final_publication_cosmo"))
    if final_publication_cosmo is None:
        reasons.append("full_scf_gpu_final_publication_cosmo was not reported")
    if (
        host_commit_arrays is not None
        and final_publication_arrays is not None
        and host_commit_arrays != final_publication_arrays
    ):
        reasons.append(
            "host commit arrays and typed final publication arrays disagree "
            f"({host_commit_arrays} != {final_publication_arrays})"
        )
    if (
        host_commit_bytes is not None
        and final_publication_bytes is not None
        and host_commit_bytes != final_publication_bytes
    ):
        reasons.append(
            "host commit bytes and typed final publication bytes disagree "
            f"({host_commit_bytes} != {final_publication_bytes})"
        )
    if (
        host_commit_cosmo is not None
        and final_publication_cosmo is not None
        and host_commit_cosmo != final_publication_cosmo
    ):
        reasons.append(
            "host commit COSMO flag and typed final publication COSMO flag disagree "
            f"({host_commit_cosmo} != {final_publication_cosmo})"
        )
    if row_uses_eps(row) or row_requires_direct_cosmo_proof(row):
        if host_commit_cosmo != 1:
            reasons.append(
                "direct COSMO strict proof did not publish COSMO state in the final host commit: "
                f"full_scf_gpu_host_commit_cosmo={row.get('full_scf_gpu_host_commit_cosmo')!r}"
            )
        if final_publication_cosmo != 1:
            reasons.append(
                "direct COSMO strict proof did not publish COSMO state in the typed final publication: "
                f"full_scf_gpu_final_publication_cosmo={row.get('full_scf_gpu_final_publication_cosmo')!r}"
            )


def required_resident_stage_calls(stage_name: str, min_stage_calls: int) -> int:
    return 1 if stage_name == "upload" else min_stage_calls


def validate_full_scf_cosmo_fields(row: dict[str, Any], reasons: list[str], *, require_enabled: bool = False) -> None:
    enabled = parse_int_value(row.get("full_scf_gpu_cosmo_enabled")) or 0
    if require_enabled and enabled != 1:
        reasons.append(f"full_scf_gpu_cosmo_enabled={row.get('full_scf_gpu_cosmo_enabled')!r}")
    if enabled == 0 and not require_enabled:
        return
    required_positive = (
        "full_scf_gpu_cosmo_fock_calls",
        "full_scf_gpu_cosmo_matvec_calls",
        "full_scf_gpu_cosmo_cg_iterations",
        "full_scf_gpu_cosmo_nps",
        "full_scf_gpu_cosmo_lm61",
    )
    for key in required_positive:
        value = parse_int_value(row.get(key))
        if value is None or value <= 0:
            reasons.append(f"{key}={row.get(key)!r}")
    pair_count = parse_int_value(row.get("full_scf_gpu_cosmo_pair_count"))
    if pair_count is None or pair_count < 0:
        reasons.append(f"full_scf_gpu_cosmo_pair_count={row.get('full_scf_gpu_cosmo_pair_count')!r}")
    for key in (
        "full_scf_gpu_cosmo_solv_energy",
        "full_scf_gpu_cosmo_ediel",
        "full_scf_gpu_cosmo_last_residual",
        "full_scf_gpu_cosmo_cg_target_tol",
    ):
        value = parse_float(row.get(key))
        if value is None:
            reasons.append(f"{key}={row.get(key)!r}")
        elif key == "full_scf_gpu_cosmo_last_residual" and value < 0.0:
            reasons.append(f"{key}={value}")
        elif key == "full_scf_gpu_cosmo_cg_target_tol" and value <= 0.0:
            reasons.append(f"{key}={value}")
    expected_ints = {
        "full_scf_gpu_cosmo_cg_control_resident": 1,
        "full_scf_gpu_cosmo_cg_converged": 1,
        "full_scf_gpu_cosmo_cg_breakdown": 0,
        "full_scf_gpu_cosmo_cg_host_syncs": 0,
    }
    for key, expected in expected_ints.items():
        value = parse_int_value(row.get(key))
        if value != expected:
            reasons.append(f"{key}={row.get(key)!r}")
    residual = parse_float(row.get("full_scf_gpu_cosmo_last_residual"))
    target_tol = parse_float(row.get("full_scf_gpu_cosmo_cg_target_tol"))
    if residual is not None and target_tol is not None and residual > target_tol:
        reasons.append(
            "full_scf_gpu_cosmo_last_residual="
            f"{residual} > full_scf_gpu_cosmo_cg_target_tol={target_tol}"
        )


def row_input_basename(row: dict[str, Any]) -> str:
    value = str(row.get("input") or "")
    if not value:
        return ""
    return Path(value).name


def row_requires_direct_cosmo_proof(row: dict[str, Any]) -> bool:
    return (
        parse_int_value(row.get("requires_direct_cosmo_gpu")) == 1
        or str(row.get("requires_direct_cosmo_gpu") or "").strip().lower()
        in {"true", "t", "yes", "on"}
        or str(row.get("molecule") or "") == "direct_cosmo_peptide_gg"
        or row_input_basename(row) == "direct_cosmo_peptide_gg.mop"
    )


def row_uses_eps(row: dict[str, Any]) -> bool:
    value = parse_float(row.get("input_eps"))
    return value is not None and value > 0.0


def validate_direct_cosmo_contract(row: dict[str, Any], reasons: list[str]) -> None:
    if not row_requires_direct_cosmo_proof(row):
        return
    if row.get("mozyme_plan_direct_mode") != "T":
        reasons.append(
            "direct COSMO proof did not report direct integral mode in the MOZYME plan: "
            f"mozyme_plan_direct_mode={row.get('mozyme_plan_direct_mode')!r}"
        )
    eps = parse_float(row.get("input_eps"))
    if eps is None:
        reasons.append("direct COSMO proof input did not report input_eps")
    elif abs(eps - 78.4) > 1.0e-9:
        reasons.append(f"direct COSMO proof input_eps={eps}; expected 78.4")
    pair_count = parse_int_value(row.get("full_scf_gpu_cosmo_pair_count"))
    if pair_count is None or pair_count <= 0:
        reasons.append(
            "direct COSMO proof did not report positive resident COSMO pair work: "
            f"full_scf_gpu_cosmo_pair_count={row.get('full_scf_gpu_cosmo_pair_count')!r}"
        )
    planned_point_pairs = parse_int_value(row.get("mozyme_fock_plan_point_charge_pairs")) or 0
    if planned_point_pairs > 0:
        setup_point_tasks = parse_int_value(row.get("mozyme_sparse_fock_setup_point_tasks")) or 0
        run_point_tasks = parse_int_value(row.get("mozyme_sparse_fock_run_point_tasks")) or 0
        if setup_point_tasks <= 0 or run_point_tasks <= 0:
            reasons.append(
                "direct COSMO proof planned point-charge/dipole Fock work but did not run "
                "positive resident point work: "
                f"planned={planned_point_pairs} setup={setup_point_tasks} "
                f"run={run_point_tasks}"
            )
    direct_real_fallback = parse_int_value(
        row.get("mozyme_fock_resident_direct_unsupported_pairs")
    ) or 0
    direct_point_fallback = parse_int_value(
        row.get("mozyme_fock_resident_direct_unsupported_point_pairs")
    ) or 0
    if direct_real_fallback != 0 or direct_point_fallback != 0:
        reasons.append(
            "direct COSMO proof encountered resident Fock direct-basis fallback "
            f"(real={direct_real_fallback}, point={direct_point_fallback}); "
            "current CUDA direct support is limited to basis blocks 1, 4, and 9"
        )


def full_scf_gpu_contract_violation_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    returncode = parse_int_value(row.get("returncode"))
    if row.get("timed_out"):
        reasons.append(f"MOPAC timed out after {float(row.get('wall_s') or 0.0):.1f} s")
    if returncode != 0:
        reasons.append(f"MOPAC returned non-zero exit code {row.get('returncode')}")
    if not row.get("normal_end"):
        reasons.append("MOPAC did not finish normally with a parsed heat of formation")
    error_count = parse_int_value(row.get("gpu_error_marker_count"))
    if error_count is None:
        reasons.append("gpu_error_marker_count was not reported")
    elif error_count != 0:
        markers = str(row.get("gpu_error_markers") or "")
        suffix = f": {markers}" if markers else ""
        reasons.append(f"gpu_error_marker_count is {error_count}{suffix}")

    status = str(row.get("full_scf_gpu_status") or "not_requested")
    if status != "complete":
        reason = row.get("full_scf_gpu_reason") or "no_reason"
        reasons.append(f"complete GPU SCF is not available: full_scf_gpu_status={status} reason={reason}")
    if parse_int_value(row.get("full_scf_gpu_requested")) != 1:
        reasons.append("full_scf_gpu_requested is not 1")
    if parse_int_value(row.get("full_scf_gpu_executed")) != 1:
        reasons.append("full_scf_gpu_executed is not 1")
    if parse_int_value(row.get("mozyme_scf_experimental_executed")) != 1:
        reasons.append("mozyme_scf_experimental_executed is not 1")
    if parse_int_value(row.get("full_scf_gpu_ready")) != 1:
        reasons.append("full_scf_gpu_ready is not 1")
    if parse_int_value(row.get("full_scf_gpu_backend_ready")) != 1:
        reasons.append("backend ready marker is not 1")
    if parse_int_value(row.get("full_scf_gpu_resident")) != 1:
        reasons.append("resident execution marker is not 1")
    compact_route = parse_int_value(row.get("full_scf_gpu_compact_index_route"))
    if compact_route != 0:
        reasons.append(
            "strict resident SCF did not prove the nijbo route: "
            f"full_scf_gpu_compact_index_route={row.get('full_scf_gpu_compact_index_route')!r}"
        )
    if parse_int_value(row.get("full_scf_gpu_use_nijbo")) != 1:
        reasons.append(
            f"strict resident SCF use_nijbo marker is not 1 ({row.get('full_scf_gpu_use_nijbo')!r})"
        )
    if parse_int_value(row.get("full_scf_gpu_isitsc_okscf")) != 1:
        reasons.append("resident-SCF ISITSC convergence marker is not 1")
    if (parse_int_value(row.get("full_scf_gpu_pls_restart_required")) or 0) != 0:
        reasons.append("resident-SCF detected a PLS restart requirement that was not completed on GPU")
    pls_reset_calls = parse_int_value(row.get("full_scf_gpu_pls_restart_reset_device_calls")) or 0
    if pls_reset_calls > 0 and parse_int_value(row.get("full_scf_gpu_pls_restart_done")) != 1:
        reasons.append(
            "resident-SCF reported device PLS reset calls without pls_restart_done=1"
        )
    if parse_int_value(row.get("full_scf_gpu_final_density_resident")) != 1:
        reasons.append("final MOZYME density was not reported as the resident GPU density")
    forced_probe_keywords = str(row.get("full_scf_probe_forced_keywords") or "")
    if "RE-LOCAL=1" in forced_probe_keywords:
        if (parse_int_value(row.get("mozyme_relocal_gpu_occupied_success_calls")) or 0) <= 0:
            reasons.append("forced RE-LOCAL=1 probe did not report occupied GPU relocalization success")
        if (parse_int_value(row.get("mozyme_relocal_gpu_virtual_success_calls")) or 0) <= 0:
            reasons.append("forced RE-LOCAL=1 probe did not report virtual GPU relocalization success")
    forced_probe_env = str(row.get("full_scf_probe_forced_env") or "")
    if "MOPAC_MOZYME_SCF_FORCE_FINAL_REORTH=1" in forced_probe_env:
        if (parse_int_value(row.get("mozyme_reorth_gpu_resident_success_calls")) or 0) <= 0:
            reasons.append("forced REORTH probe did not report a resident=1 final reorth success marker")
    if (parse_int_value(row.get("mozyme_tidy_gpu_occupied_success_calls")) or 0) <= 0:
        reasons.append("strict resident SCF did not report occupied GPU TIDY success")
    if (parse_int_value(row.get("mozyme_tidy_gpu_virtual_success_calls")) or 0) <= 0:
        reasons.append("strict resident SCF did not report virtual GPU TIDY success")
    tidy_fallback_calls = parse_int_value(row.get("mozyme_tidy_gpu_fallback_calls"))
    if tidy_fallback_calls is None:
        reasons.append("mozyme_tidy_gpu_fallback_calls was not reported")
    elif tidy_fallback_calls != 0:
        reasons.append(f"MOZYME GPU TIDY emitted {tidy_fallback_calls} fallback marker(s)")
    probe_decision = row.get("full_scf_probe_decision")
    if probe_decision != "complete":
        reasons.append(f"strict readiness probe decision was not complete: {probe_decision}")
    if row.get("gpu_has_device") != "T":
        reasons.append("GPU debug state did not confirm hasGPU=T")
    if row.get("gpu_lgpu_final") != "T":
        reasons.append("final MOPAC GPU switch lgpu is not T")
    reasons.extend(proof_identity_violation_reasons(row))
    scf_fallback_calls = parse_int_value(row.get("full_scf_gpu_scf_fallback_calls"))
    if scf_fallback_calls is None:
        reasons.append("full_scf_gpu_scf_fallback_calls was not reported")
    elif scf_fallback_calls != 0:
        reasons.append(
            f"resident-SCF emitted {scf_fallback_calls} fallback/strict-abort status marker(s)"
        )
    strict_host_count = parse_int_value(row.get("full_scf_gpu_strict_host_route_marker_count"))
    if strict_host_count is None:
        reasons.append("full_scf_gpu_strict_host_route_marker_count was not reported")
    elif strict_host_count != 0:
        markers = str(row.get("full_scf_gpu_strict_host_route_markers") or "")
        suffix = f": {markers}" if markers else ""
        reasons.append(
            f"strict resident proof encountered {strict_host_count} host route marker(s){suffix}"
        )
    validate_strict_host_route_contract(row, reasons)
    validate_full_scf_host_commit_contract(row, reasons)
    scf_success_calls = parse_int_value(row.get("full_scf_gpu_scf_success_calls"))
    if scf_success_calls is None:
        reasons.append("full_scf_gpu_scf_success_calls was not reported")
    elif scf_success_calls <= 0:
        reasons.append("resident-SCF emitted no success status marker")
    resident_step_calls = parse_int_value(row.get("full_scf_gpu_resident_step_calls"))
    if resident_step_calls is None:
        reasons.append("full_scf_gpu_resident_step_calls was not reported")
    elif resident_step_calls != 0:
        reasons.append(f"resident-SCF emitted {resident_step_calls} resident_step status marker(s)")
    if row.get("mozyme_plan_resident_fock_gpu") != "T":
        reasons.append("resident sparse Fock GPU was not enabled in the MOZYME GPU plan")

    backend_code = parse_int_value(row.get("full_scf_gpu_code"))
    if backend_code != 0:
        reasons.append(f"full_scf_gpu_code is not 0 ({row.get('full_scf_gpu_code')})")

    device_id = parse_int_value(row.get("full_scf_gpu_device_id"))
    if device_id is None or device_id < 0:
        reasons.append(f"full_scf_gpu_device_id is not a nonnegative CUDA device ({row.get('full_scf_gpu_device_id')})")
    wall_ms = parse_float(row.get("full_scf_gpu_wall_ms"))
    if wall_ms is None:
        reasons.append("full_scf_gpu_wall_ms was not reported")
    elif wall_ms < 0.0:
        reasons.append(f"full_scf_gpu_wall_ms is negative ({wall_ms})")
    if parse_float(row.get("full_scf_gpu_energy_total")) is None:
        reasons.append("full_scf_gpu_energy_total was not parseable")
    if parse_float(row.get("full_scf_gpu_diagg_sumt")) is None:
        reasons.append("full_scf_gpu_diagg_sumt was not parseable")
    if parse_float(row.get("full_scf_gpu_diagg_sumb")) is None:
        reasons.append("full_scf_gpu_diagg_sumb was not parseable")
    validate_direct_cosmo_contract(row, reasons)
    validate_full_scf_cosmo_fields(
        row,
        reasons,
        require_enabled=row_uses_eps(row) or row_requires_direct_cosmo_proof(row),
    )

    stage_required = parse_int_value(row.get("full_scf_gpu_stage_required"))
    if stage_required != MOZYME_SCF_STAGE_FULL:
        reasons.append(f"full_scf_gpu_stage_required is not {MOZYME_SCF_STAGE_FULL} ({row.get('full_scf_gpu_stage_required')})")

    stage_completed = parse_int_value(row.get("full_scf_gpu_stage_completed"))
    if stage_completed is None or (stage_completed & MOZYME_SCF_STAGE_FULL) != MOZYME_SCF_STAGE_FULL:
        missing = MOZYME_SCF_STAGE_FULL if stage_completed is None else MOZYME_SCF_STAGE_FULL & ~stage_completed
        reasons.append(
            "full_scf_gpu_stage_completed does not include required mask "
            f"{MOZYME_SCF_STAGE_FULL} ({mozyme_scf_stage_names(missing)})"
        )
    elif stage_completed != MOZYME_SCF_STAGE_FULL:
        reasons.append(
            "full_scf_gpu_stage_completed has extra bits "
            f"{stage_completed & ~MOZYME_SCF_STAGE_FULL}"
        )

    stage_missing = parse_int_value(row.get("full_scf_gpu_stage_missing"))
    if stage_missing != 0:
        names = row.get("full_scf_gpu_stage_missing_names") or mozyme_scf_stage_names(stage_missing)
        reasons.append(f"full_scf_gpu_stage_missing is not 0 ({stage_missing}, {names})")

    resident_decision = parse_int_value(row.get("full_scf_gpu_resident_decision"))
    if resident_decision != MOZYME_SCF_RESIDENT_DECISION_COMPLETE:
        reasons.append(
            "full_scf_gpu_resident_decision is not CompleteAndPublish "
            f"({row.get('full_scf_gpu_resident_decision')})"
        )
    expected_completed_names = mozyme_scf_stage_names(MOZYME_SCF_STAGE_FULL)
    raw_completed_names = str(row.get("full_scf_gpu_stage_completed_names_raw") or "")
    if raw_completed_names != expected_completed_names:
        reasons.append(
            "raw stage_completed_names did not prove the full resident stage mask "
            f"({raw_completed_names or 'missing'} != {expected_completed_names})"
        )
    raw_missing_names = str(row.get("full_scf_gpu_stage_missing_names_raw") or "")
    if raw_missing_names != "none":
        reasons.append(
            "raw stage_missing_names did not prove zero missing stages "
            f"({raw_missing_names or 'missing'})"
        )
    for key in (
        "full_scf_gpu_strict_resident",
        "full_scf_gpu_no_fallback_required",
        "full_scf_gpu_full_stage_mask",
        "full_scf_gpu_resident_decision_complete",
        "full_scf_gpu_resident_fock_plan_full_coverage",
    ):
        if parse_int_value(row.get(key)) != 1:
            reasons.append(f"{key} is not 1 ({row.get(key)})")
    for key in (
        "full_scf_gpu_strict_resident_host_syncs",
        "full_scf_gpu_strict_resident_control_polls",
    ):
        value = parse_int_value(row.get(key))
        if value is None:
            reasons.append(f"{key} was not reported")
        elif value != 0:
            reasons.append(f"{key} is not 0 ({row.get(key)})")
    if parse_int_value(row.get("full_scf_gpu_resident_fock_plan_id")) is None:
        reasons.append("full_scf_gpu_resident_fock_plan_id was not reported")
    required_plan_mask = parse_int_value(
        row.get("full_scf_gpu_resident_fock_plan_required_mask")
    )
    covered_plan_mask = parse_int_value(
        row.get("full_scf_gpu_resident_fock_plan_covered_mask")
    )
    if required_plan_mask is None or required_plan_mask == 0:
        reasons.append("full_scf_gpu_resident_fock_plan_required_mask was not reported")
    if covered_plan_mask is None:
        reasons.append("full_scf_gpu_resident_fock_plan_covered_mask was not reported")
    if required_plan_mask is not None and covered_plan_mask is not None:
        if covered_plan_mask != required_plan_mask:
            reasons.append(
                "resident Fock covered mask must exactly match required plans "
                f"(required={required_plan_mask}, covered={covered_plan_mask})"
            )
        if required_plan_mask & 2:
            partial_coverage = parse_int_value(
                row.get("full_scf_gpu_resident_fock_plan_partial_coverage")
            )
            if partial_coverage != 1:
                reasons.append(
                    "full_scf_gpu_resident_fock_plan_partial_coverage is not 1 "
                    f"({row.get('full_scf_gpu_resident_fock_plan_partial_coverage')})"
                )

    min_stage_calls = parse_int_value(row.get("full_scf_gpu_final_iterations")) or 1
    for stage_name in MOZYME_SCF_STAGE_NAMES:
        expected_calls = required_resident_stage_calls(stage_name, min_stage_calls)
        calls = parse_int_value(row.get(f"full_scf_gpu_stage_{stage_name}_calls"))
        ms = parse_float(row.get(f"full_scf_gpu_stage_{stage_name}_ms"))
        if calls is None:
            reasons.append(f"resident-SCF stage {stage_name} did not report a call counter")
        elif calls < expected_calls:
            reasons.append(
                f"resident-SCF stage {stage_name} calls={calls}; expected at least {expected_calls}"
            )
        if ms is None:
            reasons.append(f"resident-SCF stage {stage_name} did not report runtime ms")
        elif ms < 0.0:
            reasons.append(f"resident-SCF stage {stage_name} runtime is negative ({ms})")
    cnvgz_active_calls = parse_int_value(row.get("full_scf_gpu_cnvgz_active_calls"))
    cnvgz_noop_calls = parse_int_value(row.get("full_scf_gpu_cnvgz_noop_calls"))
    append_cnvgz_active_noop_reasons(reasons, cnvgz_active_calls, cnvgz_noop_calls)

    section_rows = row.get("mozyme_section_times")
    if not isinstance(section_rows, list) or not section_rows:
        reasons.append("MOZYME section profile markers were not present")
    else:
        final_reorth_sections = mozyme_final_reorth_section_names(section_rows)
        if final_reorth_sections:
            reasons.append(
                "CPU final reorthogonalization sections ran in strict proof: "
                + ";".join(final_reorth_sections)
            )

    cpu_mutating_count = parse_int_value(row.get("full_scf_gpu_cpu_mutating_section_count"))
    cpu_mutating_sections = str(row.get("full_scf_gpu_cpu_mutating_sections") or "")
    if cpu_mutating_count is None:
        reasons.append("full_scf_gpu_cpu_mutating_section_count was not reported")
    elif cpu_mutating_count != 0:
        suffix = f": {cpu_mutating_sections}" if cpu_mutating_sections else ""
        reasons.append(f"full_scf_gpu_cpu_mutating_section_count is {cpu_mutating_count}{suffix}")
    elif cpu_mutating_sections:
        reasons.append(f"full_scf_gpu_cpu_mutating_sections is nonempty: {cpu_mutating_sections}")
    cpu_mutating_call_count = parse_int_value(row.get("full_scf_gpu_cpu_mutating_call_count"))
    if cpu_mutating_call_count is None:
        reasons.append("full_scf_gpu_cpu_mutating_call_count was not reported")
    elif cpu_mutating_call_count != 0:
        reasons.append(f"full_scf_gpu_cpu_mutating_call_count is {cpu_mutating_call_count}")
    cpu_mutating_ms = parse_float(row.get("full_scf_gpu_cpu_mutating_ms"))
    if cpu_mutating_ms is None:
        reasons.append("full_scf_gpu_cpu_mutating_ms was not reported")
    elif cpu_mutating_ms != 0.0:
        reasons.append(f"full_scf_gpu_cpu_mutating_ms is {cpu_mutating_ms}")

    final_iterations = parse_int_value(row.get("full_scf_gpu_final_iterations"))
    if final_iterations is None:
        reasons.append("full_scf_gpu_final_iterations was not reported")
    elif final_iterations < 1:
        reasons.append(f"full_scf_gpu_final_iterations is below 1 ({final_iterations})")

    makvec_success_calls = parse_int_value(row.get("mozyme_makvec_gpu_success_calls")) or 0
    makvec_existing_calls = parse_int_value(row.get("mozyme_makvec_gpu_existing_lmo_calls")) or 0
    if makvec_success_calls <= 0:
        reasons.append("MOZYME makvec initial LMO construction did not report GPU success")
    if makvec_existing_calls > 0:
        reasons.append("OLD_SCF existing-LMO marker is not accepted as complete GPU makvec proof")
    makvec_ms = parse_float(row.get("mozyme_makvec_gpu_last_ms"))
    if makvec_success_calls > 0 and makvec_ms is None:
        reasons.append("MOZYME makvec GPU did not report runtime")
    elif makvec_ms is not None and makvec_ms < 0.0:
        reasons.append(f"MOZYME makvec GPU runtime is negative ({makvec_ms})")
    setupk_success_calls = parse_int_value(row.get("mozyme_setupk_gpu_success_calls"))
    if setupk_success_calls is None:
        reasons.append("mozyme_setupk_gpu_success_calls was not reported")
    elif setupk_success_calls <= 0:
        reasons.append("MOZYME setupk GPU did not report a success marker")
    setupk_fallback_calls = parse_int_value(row.get("mozyme_setupk_gpu_fallback_calls"))
    if setupk_fallback_calls is None:
        reasons.append("mozyme_setupk_gpu_fallback_calls was not reported")
    elif setupk_fallback_calls != 0:
        reasons.append(f"mozyme_setupk_gpu_fallback_calls is nonzero ({setupk_fallback_calls})")
    setupk_initial_success_calls = parse_int_value(row.get("mozyme_setupk_gpu_initial_setup_success_calls"))
    if setupk_initial_success_calls is None:
        reasons.append("mozyme_setupk_gpu_initial_setup_success_calls was not reported")
    elif setupk_initial_success_calls <= 0:
        reasons.append("MOZYME setupk GPU did not report an initial_setup=1 success marker")
    setupk_initial_fallback_calls = parse_int_value(row.get("mozyme_setupk_gpu_initial_setup_fallback_calls"))
    if setupk_initial_fallback_calls is None:
        reasons.append("mozyme_setupk_gpu_initial_setup_fallback_calls was not reported")
    elif setupk_initial_fallback_calls != 0:
        reasons.append("MOZYME setupk GPU reported an initial_setup=1 fallback marker")
    setupk_initial_all_paths_calls = parse_int_value(
        row.get("mozyme_setupk_gpu_initial_setup_all_paths_calls")
    )
    if setupk_initial_all_paths_calls is None:
        reasons.append("mozyme_setupk_gpu_initial_setup_all_paths_calls was not reported")
    elif setupk_initial_all_paths_calls <= 0:
        reasons.append("MOZYME setupk all-initial-setup path marker was not reported")
    setupk_initial_ms = parse_float(row.get("mozyme_setupk_gpu_initial_setup_last_ms"))
    if setupk_initial_ms is None:
        reasons.append("MOZYME setupk initial_setup=1 GPU did not report runtime")
    elif setupk_initial_ms < 0.0:
        reasons.append(f"MOZYME setupk initial_setup=1 GPU runtime is negative ({setupk_initial_ms})")
    sparse_calls = parse_int_value(row.get("mozyme_sparse_fock_run_calls")) or 0
    sparse_work = sum(
        parse_int_value(row.get(key)) or 0
        for key in (
            "mozyme_sparse_fock_run_one_tasks",
            "mozyme_sparse_fock_run_pair_tasks",
            "mozyme_sparse_fock_run_4x1_tasks",
            "mozyme_sparse_fock_run_point_tasks",
        )
    )
    sparse_ms = parse_float(row.get("mozyme_sparse_fock_run_ms"))
    if sparse_calls <= 0:
        reasons.append("resident sparse Fock GPU reported zero run calls")
    elif final_iterations is not None and sparse_calls < final_iterations:
        reasons.append(
            f"resident sparse Fock GPU ran {sparse_calls} time(s), below resident iteration count {final_iterations}"
        )
    if sparse_work <= 0:
        reasons.append("resident sparse Fock GPU reported zero work tasks")
    sparse_zero_work_calls = parse_int_value(row.get("mozyme_sparse_fock_run_zero_work_calls")) or 0
    if sparse_zero_work_calls != 0:
        reasons.append(f"resident sparse Fock GPU reported {sparse_zero_work_calls} zero-work run call(s)")
    validate_resident_point_charge_coverage(row, sparse_calls, reasons)
    if sparse_ms is None:
        reasons.append("resident sparse Fock GPU did not report runtime")
    elif sparse_ms < 0.0:
        reasons.append(f"resident sparse Fock GPU runtime is negative ({sparse_ms})")

    validate_resident_plan_coverage(row, reasons)
    validate_resident_real_pair_coverage(row, reasons)

    boundary_fallback = parse_int_value(row.get("full_scf_gpu_fallback")) or 0
    if boundary_fallback != 0:
        reasons.append(f"full_scf_gpu_fallback is nonzero ({boundary_fallback})")
    for key in (*RESIDENT_FOCK_FALLBACK_KEYS, *FULL_SCF_GPU_FALLBACK_KEYS):
        value = parse_int_value(row.get(key)) or 0
        if value != 0:
            reasons.append(f"{key} is nonzero ({value})")

    return reasons


def finalize_full_scf_gpu_contract(row: dict[str, Any]) -> None:
    status = str(row.get("full_scf_gpu_status") or "not_requested")
    if (
        status == "not_requested"
        and int(row.get("full_scf_gpu_requested") or 0) == 0
        and int(row.get("full_scf_gpu_executed") or 0) == 0
    ):
        row["full_scf_gpu_contract_violations"] = ""
        return
    violations = full_scf_gpu_contract_violation_reasons(row)
    row["full_scf_gpu_contract_violations"] = "; ".join(violations)
    if status != "complete" or not violations:
        return
    row["full_scf_gpu_status"] = "incomplete_success"
    row["full_scf_gpu_ready"] = 0
    row["full_scf_gpu_reason"] = row.get("full_scf_gpu_reason") or "strict_contract_violation"


def parse_time_expression(value: str) -> float | None:
    text = value.upper().replace(",", " ")
    nums = [parse_float(token) for token in re.findall(r"[+\-]?\d+(?:\.\d*)?(?:[Ee][+\-]?\d+)?", text)]
    nums = [num for num in nums if num is not None]
    if not nums:
        return None
    seconds = nums[-1]
    if "MINUTE" in text and len(nums) >= 2:
        seconds += 60.0 * nums[-2]
    if "HOUR" in text and len(nums) >= 3:
        seconds += 3600.0 * nums[-3]
    return seconds


def parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace("D", "E").replace("d", "E"))
    except (TypeError, ValueError):
        return None


def count_atoms(input_path: Path) -> int:
    refs = referenced_files(input_path)
    for ref in refs:
        pdb = input_path.parent / ref
        if pdb.exists() and pdb.suffix.lower() == ".pdb":
            count = count_pdb_atoms(pdb)
            if count:
                return count
    return count_mop_atoms(input_path)


def count_pdb_atoms(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            count += 1
    return count


def count_mop_atoms(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[2:]:
        if ATOM_LINE_RE.match(line):
            count += 1
    return count


def summarize(
    rows: list[dict[str, Any]],
    energy_abs_tol: float,
    energy_rel_tol: float,
    energy_per_atom_tol: float,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    molecules = sorted({row["molecule"] for row in rows})
    for molecule in molecules:
        mol_rows = [row for row in rows if row["molecule"] == molecule]
        cpu_rows = successful_rows(mol_rows, "CPU")
        gpu_rows = successful_rows(mol_rows, "GPU")
        atoms_value = next((row["atoms"] for row in mol_rows if row["atoms"]), "")
        atoms = int(atoms_value) if atoms_value not in ("", None) else 0
        cpu_best = min((row["wall_s"] for row in cpu_rows), default=None)
        gpu_best = min((row["wall_s"] for row in gpu_rows), default=None)
        production_gpu_rows = [row for row in gpu_rows if production_gpu_work_units(row) > 0]
        production_gpu_best = min((row["wall_s"] for row in production_gpu_rows), default=None)
        cpu_heat = first_heat(cpu_rows)
        gpu_heat = first_heat(gpu_rows)
        abs_diff = abs(cpu_heat - gpu_heat) if cpu_heat is not None and gpu_heat is not None else None
        rel_diff = abs_diff / max(abs(cpu_heat), 1.0e-16) if abs_diff is not None and cpu_heat is not None else None
        abs_per_atom = abs_diff / atoms if abs_diff is not None and atoms > 0 else None
        speedup = cpu_best / production_gpu_best if cpu_best and production_gpu_best and production_gpu_best > 0.0 else None
        speedup_scope = "production_gpu" if speedup is not None else ""
        if gpu_rows and not production_gpu_rows:
            speedup_scope = "no_profiled_production_gpu_work"
        production_units = sum(production_gpu_work_units(row) for row in gpu_rows)
        experimental_scf_status = summarize_experimental_scf(gpu_rows)
        full_scf_gpu_status, full_scf_gpu_ready, full_scf_gpu_fallback = summarize_full_scf_gpu(gpu_rows)
        full_scf_gpu_device_id = summarize_full_scf_gpu_device_id(gpu_rows)
        status, accuracy_basis = molecule_status(
            cpu_rows,
            gpu_rows,
            abs_diff,
            rel_diff,
            abs_per_atom,
            energy_abs_tol,
            energy_rel_tol,
            energy_per_atom_tol,
        )
        energy_status = status
        proof_status = molecule_proof_status(gpu_rows)
        combined_status = status
        if proof_status in {
            "FULL_SCF_GPU_PROVEN",
            "FULL_SCF_GPU_DEVELOPMENT_PROOF",
        }:
            combined_status = f"{status}+{proof_status}"
        elif proof_status == "NOT_FULL_SCF_GPU_PROOF":
            combined_status = f"{status}+{proof_status}"
        summary.append(
            {
                "molecule": molecule,
                "atoms": atoms_value,
                "cpu_best_s": cpu_best if cpu_best is not None else "",
                "gpu_best_s": gpu_best if gpu_best is not None else "",
                "production_gpu_best_s": production_gpu_best if production_gpu_best is not None else "",
                "speedup": speedup if speedup is not None else "",
                "production_speedup": speedup if speedup is not None else "",
                "speedup_scope": speedup_scope,
                "production_gpu_work_units": production_units,
                "experimental_scf_status": experimental_scf_status,
                "full_scf_gpu_status": full_scf_gpu_status,
                "full_scf_gpu_ready": full_scf_gpu_ready,
                "full_scf_gpu_fallback": full_scf_gpu_fallback,
                "full_scf_gpu_device_id": full_scf_gpu_device_id,
                "cpu_heat_kcal_mol": cpu_heat if cpu_heat is not None else "",
                "gpu_heat_kcal_mol": gpu_heat if gpu_heat is not None else "",
                "abs_heat_diff_kcal_mol": abs_diff if abs_diff is not None else "",
                "rel_heat_diff": rel_diff if rel_diff is not None else "",
                "abs_heat_diff_per_atom_kcal_mol": abs_per_atom if abs_per_atom is not None else "",
                "accuracy_basis": accuracy_basis,
                "energy_status": energy_status,
                "proof_status": proof_status,
                "status": combined_status,
            }
        )
    return summary


def production_gpu_work_units(row: dict[str, Any]) -> int:
    keys = [
        "mozyme_sparse_fock_run_calls",
        "mozyme_fock1_batch_gpu_success_calls",
        "mozyme_fock2_4x1_batch_gpu_success_calls",
        "density_gpu_syrk_calls",
        "density_gpu_gemm_calls",
        "density_batch_gpu_success_calls",
        "mozyme_makvec_gpu_success_calls",
        "mozyme_makvec_gpu_existing_lmo_calls",
        "mozyme_relocal_gpu_success_calls",
        "mozyme_reorth_gpu_success_calls",
        "mozyme_cnvgz_gpu_success_calls",
        "mozyme_helecz_gpu_success_calls",
        "mozyme_eimp_gpu_success_calls",
        "mozyme_isitsc_gpu_success_calls",
        "mozyme_diagg1_construct_gpu_success_calls",
        "mozyme_diagg1_aocc_gpu_success_calls",
        "mozyme_diagg1_avir_gpu_success_calls",
        "mozyme_diagg2_rotate_gpu_success_calls",
        "mozyme_diagg2_rotprep_gpu_success_calls",
        "mozyme_isitsc_gpu_success_calls",
    ]
    total = 0
    for key in keys:
        value = row.get(key, "")
        if value in ("", None):
            continue
        total += int(value)
    return total


def cpu_companion_gpu_leakage_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("mode") != "CPU":
        reasons.append(f"CPU companion row mode is {row.get('mode')!r}, expected 'CPU'")
    error_count = parse_int_value(row.get("gpu_error_marker_count"))
    if error_count is None:
        reasons.append("CPU companion did not report gpu_error_marker_count")
    elif error_count != 0:
        reasons.append(f"CPU companion emitted GPU error markers ({error_count})")
    if row.get("gpu_lgpu_final") == "T":
        reasons.append("CPU companion finished with lgpu=T; MOPAC_NOGPU/MOZYME_GPU_OFF was not honored")
    if row.get("mozyme_gpu_profile") == "T" or row.get("mozyme_gpu_after_plan") == "T":
        reasons.append("CPU companion reported MOZYME GPU active")
    if row.get("mozyme_plan_enabled") == "T":
        reasons.append("CPU companion MOZYME GPU planner reported enabled=T")
    for key in (
        "full_scf_gpu_requested",
        "full_scf_gpu_executed",
        "mozyme_scf_experimental_executed",
        "full_scf_gpu_scf_success_calls",
        "full_scf_gpu_resident_step_calls",
        "full_scf_gpu_cpu_boundary_calls",
        "mozyme_sparse_fock_setup_calls",
        "mozyme_sparse_fock_run_calls",
    ):
        value = parse_int_value(row.get(key)) or 0
        if value != 0:
            reasons.append(f"CPU companion {key} is nonzero ({value})")
    work_units = production_gpu_work_units(row)
    if work_units != 0:
        reasons.append(f"CPU companion reported GPU work units ({work_units})")
    for key in (*FULL_SCF_GPU_FALLBACK_KEYS, *RESIDENT_FOCK_FALLBACK_KEYS):
        value = parse_int_value(row.get(key)) or 0
        if value != 0:
            reasons.append(f"CPU companion {key} is nonzero ({value})")
    return reasons


def summarize_experimental_scf(rows: list[dict[str, Any]]) -> str:
    statuses = [str(row.get("mozyme_scf_experimental_status") or "") for row in rows]
    statuses = [status for status in statuses if status]
    if not statuses:
        return "not_run"
    if "success_pending_contract" in statuses:
        return "success_pending_contract"
    if "strict_abort" in statuses:
        return "strict_abort"
    if "fallback_cpu" in statuses:
        return "fallback_cpu"
    if "resident_step" in statuses:
        return "resident_step"
    return "executed"


def summarize_full_scf_gpu(rows: list[dict[str, Any]]) -> tuple[str, int, int]:
    statuses = [str(row.get("full_scf_gpu_status") or "not_requested") for row in rows]
    requested_statuses = [status for status in statuses if status != "not_requested"]
    if not requested_statuses:
        return "not_requested", 0, 0
    if "strict_abort" in requested_statuses:
        return "strict_abort", 0, 1
    if "fallback_cpu" in requested_statuses:
        return "fallback_cpu", 0, 1
    if "not_executed" in requested_statuses:
        return "not_executed", 0, 0
    if "marker_without_status" in requested_statuses:
        return "marker_without_status", 0, 0
    if "incomplete_success" in requested_statuses:
        return "incomplete_success", 0, 0
    if "unverified_success" in requested_statuses:
        return "unverified_success", 0, 0
    if "resident_step" in requested_statuses:
        return "resident_step", 0, 0
    if all(status == "complete" for status in requested_statuses):
        complete_rows = [row for row in rows if str(row.get("full_scf_gpu_status") or "") == "complete"]
        if any(row.get("full_scf_gpu_contract_violations") or full_scf_gpu_contract_violation_reasons(row) for row in complete_rows):
            return "incomplete_success", 0, 0
        return "complete", 1, 0
    return "marker_without_status", 0, int(
        "fallback_cpu" in requested_statuses or "strict_abort" in requested_statuses
    )


def summarize_full_scf_gpu_device_id(rows: list[dict[str, Any]]) -> int | str:
    device_ids = [
        value
        for value in (parse_int_value(row.get("full_scf_gpu_device_id")) for row in rows)
        if value is not None
    ]
    if not device_ids:
        return ""
    unique_ids = sorted(set(device_ids))
    if len(unique_ids) == 1:
        return unique_ids[0]
    return ";".join(str(value) for value in unique_ids)


def molecule_proof_status(gpu_rows: list[dict[str, Any]]) -> str:
    if not gpu_rows:
        return "NOT_FULL_SCF_GPU_PROOF"
    dirty_source_seen = False
    for row in gpu_rows:
        if (
            parse_int_value(row.get("full_scf_gpu_requested")) != 1
            or str(row.get("full_scf_gpu_status") or "") != "complete"
            or parse_int_value(row.get("full_scf_gpu_ready")) != 1
            or str(row.get("full_scf_gpu_contract_violations") or "")
            or full_scf_gpu_contract_violation_reasons(row)
            or diagnose_full_scf_gpu_readiness(row)
        ):
            return "NOT_FULL_SCF_GPU_PROOF"
        if str(row.get("source_git_dirty") or "").lower() == "true":
            dirty_source_seen = True
    if dirty_source_seen:
        return "FULL_SCF_GPU_DEVELOPMENT_PROOF"
    return "FULL_SCF_GPU_PROVEN"


def molecule_status(
    cpu_rows: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
    abs_diff: float | None,
    rel_diff: float | None,
    abs_per_atom: float | None,
    energy_abs_tol: float,
    energy_rel_tol: float,
    energy_per_atom_tol: float,
) -> tuple[str, str]:
    if not cpu_rows or not gpu_rows:
        return "INCOMPLETE", "missing_cpu_or_gpu"
    if abs_diff is None or rel_diff is None:
        return "NO_ENERGY", "missing_heat"
    basis: list[str] = []
    if abs_diff <= energy_abs_tol:
        basis.append("absolute")
    if rel_diff <= energy_rel_tol:
        basis.append("relative")
    if abs_per_atom is not None and abs_per_atom <= energy_per_atom_tol:
        basis.append("per_atom")
    if basis:
        return "PASS", "+".join(basis)
    return "ENERGY_MISMATCH", "outside_tolerance"


def successful_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    successful: list[dict[str, Any]] = []
    for row in rows:
        if not (
            row["mode"] == mode
            and row["returncode"] == 0
            and row["normal_end"]
            and row["heat_kcal_mol"] != ""
        ):
            continue
        if mode == "GPU" and parse_int_value(row.get("full_scf_gpu_requested")) == 1:
            if parse_int_value(row.get("full_scf_gpu_ready")) != 1:
                continue
            violations = row.get("full_scf_gpu_contract_violations")
            if violations is None:
                violations = "; ".join(full_scf_gpu_contract_violation_reasons(row))
            if str(violations or "").strip():
                continue
        successful.append(row)
    return successful


def first_heat(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        value = row["heat_kcal_mol"]
        if value != "":
            return float(value)
    return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in fieldnames:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise SystemExit(f"Duplicate CSV field(s) for {path.name}: {', '.join(duplicates)}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_mozyme_section_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, int, str, str], dict[str, Any]] = {}
    for row in rows:
        for section in row.get("mozyme_section_times") or []:
            calls = int(section["calls"])
            ms = float(section["ms"])
            key = (
                str(row["molecule"]),
                str(row["input"]),
                str(row["mode"]),
                int(row["repeat"]),
                str(section["name"]),
                str(row["log_path"]),
            )
            # Fortran emits cumulative section counters after each profiled call.
            # Keep only the last marker for each run/section so report totals are
            # not inflated by summing intermediate cumulative values.
            latest[key] = {
                "molecule": row["molecule"],
                "input": row["input"],
                "mode": row["mode"],
                "repeat": row["repeat"],
                "name": section["name"],
                "calls": calls,
                "ms": ms,
                "ms_per_call": ms / calls if calls > 0 else "",
                "input_eps": row.get("input_eps", ""),
                "log_path": row["log_path"],
            }
    return list(latest.values())


def summarize_mozyme_sections(section_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in section_rows:
        key = (str(row["mode"]), str(row["name"]))
        item = grouped.setdefault(
            key,
            {
                "mode": row["mode"],
                "name": row["name"],
                "calls": 0,
                "ms": 0.0,
                "run_keys": set(),
            },
        )
        item["calls"] += int(row["calls"])
        item["ms"] += float(row["ms"])
        item["run_keys"].add((row["molecule"], row["mode"], row["repeat"], row["log_path"]))

    summary: list[dict[str, Any]] = []
    for item in grouped.values():
        calls = int(item["calls"])
        summary.append(
            {
                "mode": item["mode"],
                "name": item["name"],
                "runs": len(item["run_keys"]),
                "calls": calls,
                "ms": item["ms"],
                "ms_per_call": item["ms"] / calls if calls > 0 else "",
            }
        )
    return sorted(summary, key=lambda row: (str(row["mode"]), str(row["name"])))


def make_plots(
    summary: list[dict[str, Any]],
    section_summary: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping molecule benchmark plots.", file=sys.stderr)
        return []

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    paths = [
        plot_wall_times(summary, out_dir / "molecule_wall_times.png", plt),
        plot_speedups(summary, out_dir / "molecule_gpu_speedup.png", plt),
        plot_accuracy(summary, out_dir / "molecule_heat_accuracy.png", plt),
    ]
    if section_summary:
        paths.append(plot_mozyme_section_times(section_summary, out_dir / "mozyme_section_times.png", plt))
    return paths


def plot_wall_times(summary: list[dict[str, Any]], path: Path, plt: Any) -> Path:
    rows = [row for row in summary if row["cpu_best_s"] != "" and row["gpu_best_s"] != ""]
    labels = [row["molecule"] for row in rows]
    x = list(range(len(rows)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 0.8 * len(labels)), 4.8))
    ax.bar([i - width / 2 for i in x], [float(row["cpu_best_s"]) for row in rows], width, label="CPU", color="#8A8F98")
    ax.bar([i + width / 2 for i in x], [float(row["gpu_best_s"]) for row in rows], width, label="GPU", color="#2F6B9A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Best wall time, seconds")
    ax.set_title("End-to-end molecule runtime")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_speedups(summary: list[dict[str, Any]], path: Path, plt: Any) -> Path:
    rows = [row for row in summary if row["speedup"] != ""]
    labels = [row["molecule"] for row in rows]
    values = [float(row["speedup"]) for row in rows]
    colors = ["#2F6B9A" if value >= 1.0 else "#B94E48" for value in values]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.8 * len(labels)), 4.8))
    ax.bar(labels, values, color=colors)
    ax.axhline(1.0, color="#333333", linewidth=1.0)
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel("CPU wall time / GPU wall time")
    ax.set_title("End-to-end production GPU speedup by molecule")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy(summary: list[dict[str, Any]], path: Path, plt: Any) -> Path:
    rows = [row for row in summary if row["abs_heat_diff_kcal_mol"] != ""]
    labels = [row["molecule"] for row in rows]
    total_values = [max(float(row["abs_heat_diff_kcal_mol"]), 1.0e-12) for row in rows]
    atom_values = [
        max(float(row["abs_heat_diff_per_atom_kcal_mol"]), 1.0e-12)
        for row in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(max(10.0, 1.2 * len(labels)), 4.8))
    axes[0].bar(labels, total_values, color="#4B8B3B")
    axes[0].set_ylabel("Absolute heat difference, kcal/mol")
    axes[0].set_title("Total energy agreement")
    axes[1].bar(labels, atom_values, color="#2F6B9A")
    axes[1].set_ylabel("Absolute heat difference per atom, kcal/mol/atom")
    axes[1].set_title("Size-normalized agreement")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mozyme_section_times(section_summary: list[dict[str, Any]], path: Path, plt: Any) -> Path:
    rows = [row for row in section_summary if row["ms"] != ""]
    labels = [f"{row['mode']}:{row['name']}" for row in rows]
    values = [float(row["ms"]) for row in rows]
    colors = ["#8A8F98" if row["mode"] == "CPU" else "#2F6B9A" for row in rows]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 4.8))
    ax.bar(labels, values, color=colors)
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel("Total section time, ms")
    ax.set_title("MOZYME section timing markers")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    section_summary: list[dict[str, Any]],
    plots: list[Path],
    legacy_rows: list[dict[str, Any]],
    energy_abs_tol: float,
    energy_rel_tol: float,
    energy_per_atom_tol: float,
    preflight_min_speedup: float,
    proof_identity: dict[str, Any] | None = None,
) -> None:
    if preflight_min_speedup > 0.0:
        preflight_speed_text = (
            "The preflight also runs the same molecule once on CPU and aborts if "
            f"`CPU_wall/GPU_wall < {preflight_min_speedup:g}`."
        )
    else:
        preflight_speed_text = (
            "The optional preflight speed gate was disabled for this run; enable it with "
            "`--preflight-min-speedup` when the benchmark should abort before long CPU/GPU comparisons."
        )
    lines = [
        "# MOPAC End-to-End Molecule GPU Benchmark",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Benchmark Method",
        "",
        "This benchmark runs complete MOPAC input decks twice: once with GPU disabled",
        "(`MOPAC_NOGPU=1`, `MOZYME_GPU_OFF=1`) and once with GPU enabled",
        "(`MOPAC_FORCEGPU=1`, `MOZYME_GPU_FORCE=1`). It measures wall-clock time",
        "around the full executable process, then parses MOPAC output files for reported",
        "wall time and heat of formation.",
        "For PDB-derived biomolecules, the Colab workflow first runs MOPAC ADD-H preparation",
        "and the timed benchmark uses the resulting hydrogenated geometry through `GEO_DAT`.",
        "",
        "The molecule benchmark is complementary to the low-level cuBLAS/cuSOLVER benchmark.",
        "It includes parser setup, integral work, SCF/MOZYME flow, file I/O, data transfer,",
        "GPU library calls, and any CPU-side work that remains in the production path.",
        "Summary `speedup` is a production GPU speedup: it is left blank unless the GPU",
        "run emitted profiled production MOZYME GPU work. Experimental resident-SCF probes",
        "are reported separately and are never counted as production speedup.",
        "",
        "Complete GPU SCF readiness is not inferred from Fock, density, cuBLAS, or cuSOLVER",
        "offload counters. It is marked ready only when the opt-in resident-SCF boundary",
        "emits `status=success` with backend `ready=1`, `resident=1`, `stage_missing=0`,",
        f"`stage_required={MOZYME_SCF_STAGE_FULL}`, all required stage bits completed,",
        "`isitsc_okscf=1`, a nonnegative CUDA device id, zero raw GPU error markers,",
        "complete resident Fock plan coverage, resident sparse Fock calls/work/timing,",
        "semantic point-charge/dipole coverage, zero fallback and CPU-boundary counters, at least one",
        "resident iteration, any PLS restart resolved on GPU with final `pls_restart_required=0`,",
        "`final_density=current_resident`, a typed `[MOZYME GPU SCF] final_publication_done=1` "
        "marker with positive arrays/bytes, and a final `[MOZYME GPU SCF] host_commit_only=1 "
        "phase=final_publication` publication marker.",
        MOPAC_GPU_PUBLICATION_CLAIM,
        "The current experimental backend is expected to report controlled",
        "`fallback_cpu` with a reason such as `early_probe`, `backend_missing_stages`,",
        "`backend_not_converged`, `backend_cnvgz_no_device_work`, `backend_cpu_boundary`, "
        "`backend_pls_restart_required`, `backend_resident_tidy_missing`, "
        "`denout_checkpoint`,",
        "`resident_fock_partial_coverage`, or `solvent_fock` for solvent modes outside the direct resident-COSMO contract.",
        "`denout_checkpoint` is an immediate controlled fallback at the requested",
        "CPU `.den` checkpoint; earlier resident GPU iterations may still have run.",
        "Controlled fallback rows are encoded with `full_scf_gpu_ready=0`.",
        "`OLDEN`/`OLDENS`, `DENOUT`, `PINOUT`, and `PKA` are host setup/output routes and",
        "strict proof rejects `strict_olden_host_lmo_restore`, `strict_denout_host_output`,",
        "`strict_pka_host_output`, `[MOZYME CPU pinout]`, and `.den` output artifacts.",
        "`RE-LOCAL` is accepted only when the run emits",
        "successful `[MOZYME GPU relocal]` markers. Final `REORTH` requires a successful",
        "`[MOZYME GPU reorth] status=success resident=1` marker, with no fallback in the",
        "required resident rebuild stages, and CPU final reorthogonalization sections still",
        "reject a complete-SCF claim through `full_scf_gpu_cpu_mutating_sections`.",
        "Use `--require-full-scf-gpu` when a publication run",
        "is meant to claim complete SCF on GPU; the script aborts before the benchmark if",
        "the readiness probe sees fallback, missing SCF markers, or incomplete stage masks.",
        "Before the full benchmark, the script runs one GPU preflight molecule with profiling",
        "enabled. The full benchmark is aborted if the preflight cannot confirm a CUDA device,",
        "final `lgpu=T`, active `MOZYME_GPU`, and success markers for every planned",
        "production MOZYME GPU kernel family.",
        preflight_speed_text,
        "",
        "Accuracy is reported as absolute, relative, and per-atom CPU-vs-GPU",
        "heat-of-formation difference. This is the publication-relevant end-to-end",
        "numerical check. Parallel GPU kernels accumulate floating-point sums in a",
        "different order from the CPU, so large systems are judged by total, relative,",
        "and size-normalized criteria rather than by bitwise identical total energies.",
        f"`PASS` requires absolute difference <= {energy_abs_tol:g} kcal/mol, "
        f"relative difference <= {energy_rel_tol:g}, or per-atom difference <= "
        f"{energy_per_atom_tol:g} kcal/mol/atom.",
        "`INCOMPLETE` means at least one CPU/GPU run did not finish normally within",
        "the configured timeout.",
        "",
        "Existing MOPAC outputs, when available, are included as historical references only.",
        "They are useful for sanity checks, but they are not a replacement for same-session",
        "CPU/GPU comparisons on the publication hardware.",
        "",
        "## Proof Identity",
        "",
        *(proof_identity_lines(proof_identity) if proof_identity else ["- not recorded"]),
        "",
        "## Summary",
        "",
        "| Molecule | Atoms | CPU best s | GPU best s | Production GPU best s | Production speedup | Speedup scope | Production work units | Full SCF GPU | Full SCF ready | full_scf_gpu_device_id | CPU HoF | GPU HoF | Abs diff | Rel diff | Abs/atom | Basis | Energy status | Proof status | Status |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['molecule']} | {row['atoms']} | {fmt(row['cpu_best_s'])} | "
            f"{fmt(row['gpu_best_s'])} | {fmt(row.get('production_gpu_best_s', ''))} | "
            f"{fmt(row.get('production_speedup', row['speedup']))} | {row.get('speedup_scope', '')} | "
            f"{fmt(row.get('production_gpu_work_units', ''))} | {row.get('full_scf_gpu_status', '')} | "
            f"{fmt(row.get('full_scf_gpu_ready', ''))} | "
            f"{row.get('full_scf_gpu_device_id', '')} | "
            f"{fmt(row['cpu_heat_kcal_mol'])} | {fmt(row['gpu_heat_kcal_mol'])} | "
            f"{fmt(row['abs_heat_diff_kcal_mol'])} | {fmt(row['rel_heat_diff'])} | "
            f"{fmt(row['abs_heat_diff_per_atom_kcal_mol'])} | {row['accuracy_basis']} | "
            f"{row.get('energy_status', '')} | {row.get('proof_status', '')} | {row['status']} |"
        )
    gpu_diag_rows = [
        row
        for row in rows
        if row.get("mode") == "GPU"
        and (
            row.get("density_calls") not in ("", None)
            or int(row.get("mozyme_fock1_batch_gpu_success_calls") or 0) > 0
            or int(row.get("mozyme_fock2_4x1_batch_gpu_success_calls") or 0) > 0
            or int(row.get("density_batch_gpu_success_calls") or 0) > 0
            or int(row.get("mozyme_sparse_fock_setup_calls") or 0) > 0
            or int(row.get("mozyme_sparse_fock_run_calls") or 0) > 0
            or int(row.get("mozyme_eimp_gpu_success_calls") or 0) > 0
            or int(row.get("mozyme_diagg1_construct_gpu_success_calls") or 0) > 0
            or int(row.get("mozyme_diagg1_aocc_gpu_success_calls") or 0) > 0
            or int(row.get("mozyme_diagg1_avir_gpu_success_calls") or 0) > 0
            or int(row.get("mozyme_diagg2_rotate_gpu_success_calls") or 0) > 0
            or int(row.get("mozyme_diagg2_rotprep_gpu_success_calls") or 0) > 0
            or int(row.get("mozyme_cnvgz_gpu_success_calls") or 0) > 0
            or int(row.get("mozyme_helecz_gpu_success_calls") or 0) > 0
        )
    ]
    if gpu_diag_rows:
        lines.extend(
            [
                "",
                "## MOZYME GPU Diagnostics",
                "",
                "These counters are emitted only when GPU profiling is enabled. They show whether",
                "the molecular run actually offloaded MOZYME density blocks, SCF substages,",
                "legacy batched Fock work, or the production resident sparse Fock path to",
                "GPU kernels. Sparse Fock run calls are the main production signal for the",
                "resident MOZYME route.",
                "",
                "| Molecule | Rep | Resident runs | Resident one | Resident pair | Resident 4x1 | Resident point | Point dipole | Point monopole | Resident zero-work calls | Resident ms | Fock1 calls | Fock2 4x1 calls | Density batch calls | Density batch blocks | Density batch terms | Density batch ms | Density calls | GPU SYRK | GPU GEMM | Skipped diag | Skipped offdiag | Max diag | Max offdiag j | Max offdiag k |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in gpu_diag_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_sparse_fock_run_calls', ''))} | "
                f"{fmt(row.get('mozyme_sparse_fock_run_one_tasks', ''))} | "
                f"{fmt(row.get('mozyme_sparse_fock_run_pair_tasks', ''))} | "
                f"{fmt(row.get('mozyme_sparse_fock_run_4x1_tasks', ''))} | "
                f"{fmt(row.get('mozyme_sparse_fock_run_point_tasks', ''))} | "
                f"{fmt(row.get('mozyme_sparse_fock_run_point_dipole_tasks', ''))} | "
                f"{fmt(row.get('mozyme_sparse_fock_run_point_monopole_tasks', ''))} | "
                f"{fmt(row.get('mozyme_sparse_fock_run_zero_work_calls', ''))} | "
                f"{fmt(row.get('mozyme_sparse_fock_run_ms', ''))} | "
                f"{fmt(row.get('mozyme_fock1_batch_gpu_success_calls', ''))} | "
                f"{fmt(row.get('mozyme_fock2_4x1_batch_gpu_success_calls', ''))} | "
                f"{fmt(row.get('density_batch_gpu_success_calls', ''))} | "
                f"{fmt(row.get('density_batch_gpu_last_blocks', ''))} | "
                f"{fmt(row.get('density_batch_gpu_last_terms', ''))} | "
                f"{fmt(row.get('density_batch_gpu_last_ms', ''))} | "
                f"{fmt(row.get('density_calls', ''))} | {fmt(row.get('density_gpu_syrk_calls', ''))} | "
                f"{fmt(row.get('density_gpu_gemm_calls', ''))} | "
                f"{fmt(row.get('density_skipped_diag_blocks', ''))} | "
                f"{fmt(row.get('density_skipped_offdiag_blocks', ''))} | "
                f"{fmt(row.get('density_max_diag_block', ''))} | "
                f"{fmt(row.get('density_max_offdiag_j', ''))} | "
                f"{fmt(row.get('density_max_offdiag_k', ''))} |"
            )
        total_gpu_kernel_calls = sum(int(row.get("density_gpu_syrk_calls") or 0) for row in gpu_diag_rows)
        total_gpu_kernel_calls += sum(int(row.get("density_gpu_gemm_calls") or 0) for row in gpu_diag_rows)
        total_gpu_kernel_calls += sum(int(row.get("density_batch_gpu_success_calls") or 0) for row in gpu_diag_rows)
        total_gpu_kernel_calls += sum(
            int(row.get("mozyme_fock1_batch_gpu_success_calls") or 0) for row in gpu_diag_rows
        )
        total_gpu_kernel_calls += sum(
            int(row.get("mozyme_fock2_4x1_batch_gpu_success_calls") or 0) for row in gpu_diag_rows
        )
        total_gpu_kernel_calls += sum(int(row.get("mozyme_sparse_fock_run_calls") or 0) for row in gpu_diag_rows)
        total_gpu_kernel_calls += sum(
            int(row.get("mozyme_diagg1_construct_gpu_success_calls") or 0) for row in gpu_diag_rows
        )
        total_gpu_kernel_calls += sum(
            int(row.get("mozyme_diagg2_rotate_gpu_success_calls") or 0) for row in gpu_diag_rows
        )
        total_gpu_kernel_calls += sum(
            int(row.get("mozyme_diagg2_rotprep_gpu_success_calls") or 0) for row in gpu_diag_rows
        )
        total_gpu_kernel_calls += sum(int(row.get("mozyme_fock1_gpu_success_seen") or 0) for row in gpu_diag_rows)
        total_gpu_kernel_calls += sum(int(row.get("mozyme_fock2_gpu_success_seen") or 0) for row in gpu_diag_rows)
        total_skipped = sum(int(row.get("density_skipped_diag_blocks") or 0) for row in gpu_diag_rows)
        total_skipped += sum(int(row.get("density_skipped_offdiag_blocks") or 0) for row in gpu_diag_rows)
        lines.extend(["", "Diagnostic interpretation:"])
        if total_gpu_kernel_calls == 0:
            lines.append(
                "No MOZYME density blocks, batched Fock tasks, or resident sparse Fock tasks were offloaded to GPU kernels in the profiled runs."
            )
        else:
            lines.append(
                f"Profiled GPU runs emitted {total_gpu_kernel_calls} MOZYME GPU kernel success markers."
            )
        if total_skipped:
            lines.append(
                f"{total_skipped} MOZYME density blocks were skipped by the GPU threshold and handled on CPU."
            )
    resident_coverage_rows = [
        row
        for row in rows
        if row.get("mode") == "GPU" and row.get("mozyme_resident_fock_real_pairs") not in ("", None)
    ]
    if resident_coverage_rows:
        lines.extend(
            [
                "",
                "## Resident Fock Coverage",
                "",
                "These diagnostics split real atom-pair Fock terms into GPU-covered,",
                "inactive/no-op, and CPU-fallback categories.",
                "",
                "| Molecule | Rep | Mode | real pairs | GPU real pairs | CPU real pairs | inactive real pairs | basis-limit fallback | direct-basis fallback | other fallback |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in resident_coverage_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_resident_fock_coverage_mode', ''))} | "
                f"{fmt(row.get('mozyme_resident_fock_real_pairs', ''))} | "
                f"{fmt(row.get('mozyme_resident_fock_gpu_real_pairs', ''))} | "
                f"{fmt(row.get('mozyme_resident_fock_cpu_real_pairs', ''))} | "
                f"{fmt(row.get('mozyme_resident_fock_inactive_real_pairs', ''))} | "
                f"{fmt(row.get('mozyme_resident_fock_basis_limit_fallback_pairs', ''))} | "
                f"{fmt(row.get('mozyme_resident_fock_direct_basis_fallback_pairs', ''))} | "
                f"{fmt(row.get('mozyme_resident_fock_other_fallback_pairs', ''))} |"
            )
    eimp_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and int(row.get("mozyme_eimp_gpu_success_calls") or 0) > 0
    ]
    if eimp_rows:
        lines.extend(
            [
                "",
                "## MOZYME EIMP GPU",
                "",
                "`eimp` prepares the Fock-interaction scalar used by the MOZYME",
                "diagonalizer. These rows count the exact GPU implementation that",
                "updates the packed `p` entries before `diagg` runs.",
                "",
                "| Molecule | Rep | Success calls | Fallback calls | Last code | Last pairs | Last ms |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in eimp_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_eimp_gpu_success_calls', ''))} | "
                f"{fmt(row.get('mozyme_eimp_gpu_fallback_calls', ''))} | "
                f"{fmt(row.get('mozyme_eimp_gpu_last_code', ''))} | "
                f"{fmt(row.get('mozyme_eimp_gpu_last_pairs', ''))} | "
                f"{fmt(row.get('mozyme_eimp_gpu_last_ms', ''))} |"
            )
    diagg1_construct_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and int(row.get("mozyme_diagg1_construct_gpu_success_calls") or 0) > 0
    ]
    if diagg1_construct_rows:
        lines.extend(
            [
                "",
                "## MOZYME DIAGG1 CONSTRUCT GPU",
                "",
                "`diagg1_construct` builds the occupied-virtual interaction",
                "`fmo/ifmo` list, pseudo-eigenvalues, and DIAGG control",
                "scalars inside a CUDA kernel, then copies the canonical",
                "Fortran outputs back for `diagg2`.",
                "",
                "| Molecule | Rep | Success calls | Fallback calls | Last code | Last nij | Sumt | Tiny | Last ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in diagg1_construct_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_diagg1_construct_gpu_success_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_construct_gpu_fallback_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_construct_gpu_last_code', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_construct_gpu_last_nij', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_construct_gpu_last_sumt', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_construct_gpu_last_tiny', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_construct_gpu_last_ms', ''))} |"
            )
    diagg1_aocc_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and int(row.get("mozyme_diagg1_aocc_gpu_success_calls") or 0) > 0
    ]
    if diagg1_aocc_rows:
        lines.extend(
            [
                "",
                "## MOZYME DIAGG1 AOCC GPU",
                "",
                "`diagg1_aocc` computes the occupied-LMO atom contribution",
                "screening terms used by `diagg1`. This is a production GPU",
                "substage of `diagg1`; it is not the full LMO rotation path.",
                "",
                "| Molecule | Rep | Success calls | Fallback calls | Last code | Last terms | Last ms |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in diagg1_aocc_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_diagg1_aocc_gpu_success_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_aocc_gpu_fallback_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_aocc_gpu_last_code', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_aocc_gpu_last_terms', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_aocc_gpu_last_ms', ''))} |"
            )
    diagg1_avir_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and int(row.get("mozyme_diagg1_avir_gpu_success_calls") or 0) > 0
    ]
    if diagg1_avir_rows:
        lines.extend(
            [
                "",
                "## MOZYME DIAGG1 AVIR GPU",
                "",
                "`diagg1_avir` computes the virtual-LMO atom contribution",
                "screening terms used by `diagg1`. This is a production GPU",
                "substage of `diagg1`; it is not the full LMO rotation path.",
                "",
                "| Molecule | Rep | Success calls | Fallback calls | Last code | Last terms | Last ms |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in diagg1_avir_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_diagg1_avir_gpu_success_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_avir_gpu_fallback_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_avir_gpu_last_code', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_avir_gpu_last_terms', ''))} | "
                f"{fmt(row.get('mozyme_diagg1_avir_gpu_last_ms', ''))} |"
            )
    diagg2_rotprep_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and int(row.get("mozyme_diagg2_rotprep_gpu_success_calls") or 0) > 0
    ]
    diagg2_rotate_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and int(row.get("mozyme_diagg2_rotate_gpu_success_calls") or 0) > 0
    ]
    if diagg2_rotate_rows:
        lines.extend(
            [
                "",
                "## MOZYME DIAGG2 ROTATE GPU",
                "",
                "`diagg2_rotate` performs the sparse LMO accept/retry rotation",
                "inside a CUDA kernel and copies the mutated LMO arrays back to",
                "the canonical Fortran storage. It is opt-in through",
                "`MOPAC_MOZYME_DIAGG2_ROTATE_GPU=1` until CUDA-side validation",
                "covers the full benchmark set.",
                "",
                "| Molecule | Rep | Success calls | Fallback calls | Last code | Rejections | Sumb | Last ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in diagg2_rotate_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_diagg2_rotate_gpu_success_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg2_rotate_gpu_fallback_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg2_rotate_gpu_last_code', ''))} | "
                f"{fmt(row.get('mozyme_diagg2_rotate_gpu_last_nrej', ''))} | "
                f"{fmt(row.get('mozyme_diagg2_rotate_gpu_last_sumb', ''))} | "
                f"{fmt(row.get('mozyme_diagg2_rotate_gpu_last_ms', ''))} |"
            )
    if diagg2_rotprep_rows:
        lines.extend(
            [
                "",
                "## MOZYME DIAGG2 ROTPREP GPU",
                "",
                "`diagg2_rotprep` prepares the two-by-two LMO rotation",
                "coefficients used by `diagg2`. The CPU still owns the",
                "accept/retry decisions and sparse LMO mutation, so this is a",
                "production substage rather than the full rotation path.",
                "",
                "| Molecule | Rep | Success calls | Fallback calls | Last code | Active rotations | Last ms |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in diagg2_rotprep_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_diagg2_rotprep_gpu_success_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg2_rotprep_gpu_fallback_calls', ''))} | "
                f"{fmt(row.get('mozyme_diagg2_rotprep_gpu_last_code', ''))} | "
                f"{fmt(row.get('mozyme_diagg2_rotprep_gpu_last_active', ''))} | "
                f"{fmt(row.get('mozyme_diagg2_rotprep_gpu_last_ms', ''))} |"
            )
    cnvgz_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and int(row.get("mozyme_cnvgz_gpu_success_calls") or 0) > 0
    ]
    if cnvgz_rows:
        lines.extend(
            [
                "",
                "## MOZYME CNVGZ GPU",
                "",
                "`cnvgz` updates the MOZYME density convergence history. These rows count",
                "the exact GPU implementation that copies the updated density/history back",
                "to the canonical Fortran arrays before the next SCF step.",
                "",
                "| Molecule | Rep | Success calls | Fallback calls | Last code | Last pmax | Last RMS | Last ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in cnvgz_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_cnvgz_gpu_success_calls', ''))} | "
                f"{fmt(row.get('mozyme_cnvgz_gpu_fallback_calls', ''))} | "
                f"{fmt(row.get('mozyme_cnvgz_gpu_last_code', ''))} | "
                f"{fmt(row.get('mozyme_cnvgz_gpu_last_pmax', ''))} | "
                f"{fmt(row.get('mozyme_cnvgz_gpu_last_rms', ''))} | "
                f"{fmt(row.get('mozyme_cnvgz_gpu_last_ms', ''))} |"
            )
    helecz_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and int(row.get("mozyme_helecz_gpu_success_calls") or 0) > 0
    ]
    if helecz_rows:
        lines.extend(
            [
                "",
                "## MOZYME HELECZ GPU",
                "",
                "`helecz` evaluates the MOZYME electronic energy. These rows count the",
                "exact GPU reduction path used by the SCF loop when sparse `nijbo` data is available.",
                "",
                "| Molecule | Rep | Success calls | Fallback calls | Last code | Last energy | Last ms |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in helecz_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('mozyme_helecz_gpu_success_calls', ''))} | "
                f"{fmt(row.get('mozyme_helecz_gpu_fallback_calls', ''))} | "
                f"{fmt(row.get('mozyme_helecz_gpu_last_code', ''))} | "
                f"{fmt(row.get('mozyme_helecz_gpu_last_energy', ''))} | "
                f"{fmt(row.get('mozyme_helecz_gpu_last_ms', ''))} |"
            )
    if section_summary:
        lines.extend(
            [
                "",
                "## MOZYME Section Timings",
                "",
                "These rows aggregate stdout markers of the form "
                "`[PROFILE] MOZYME_SECTION name=<name> calls=<n> ms=<total>`.",
                "Full per-run marker data is written to `mozyme_section_times.csv`.",
                "",
                "| Mode | Section | Runs | Calls | Total ms | ms/call |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in section_summary:
            lines.append(
                f"| {row['mode']} | {row['name']} | {fmt(row['runs'])} | "
                f"{fmt(row['calls'])} | {fmt(row['ms'])} | {fmt(row['ms_per_call'])} |"
            )
    scf_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and int(row.get("mozyme_scf_experimental_executed") or 0) > 0
    ]
    if scf_rows:
        lines.extend(
            [
                "",
                "## Complete SCF GPU Readiness",
                "",
                "These markers come from the opt-in `MOPAC_MOZYME_SCF_EXPERIMENTAL=1` boundary.",
                "`complete` means the backend reported `status=success`, `ready=1`, `resident=1`, "
                "`stage_missing=0`, `isitsc_okscf=1`, a final iteration count of at least 1, "
                "`final_density=current_resident`, and no CPU MOZYME state-mutating setup or bookend sections.",
                "`resident_step` means one or more resident iterations completed on GPU but control returned to Fortran.",
                "`fallback_cpu` means the experimental boundary returned to the validated CPU SCF path.",
                "`strict_abort` means strict proof failed closed before any CPU SCF continuation was allowed.",
                "Resident-step, fallback, and strict-abort rows are diagnostics only and are not counted in production speedup or complete-SCF claims.",
                "",
                (
                    "| Molecule | Rep | Requested | Executed | Ready | Status | Reason | Backend ready | "
                    "Resident | full_scf_gpu_device_id | Stage completed | Stage missing | Missing stages | ISITSC ok | Wall ms | Density max | "
                    "Density RMS | Final iterations | Final density resident | COSMO | COSMO Fock calls | COSMO matvec calls | CPU mutating sections | CPU section count |"
                ),
                (
                    "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"
                    "---:|---:|---:|---:|---:|---:|---|---:|"
                ),
            ]
        )
        for row in scf_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | "
                f"{fmt(row.get('full_scf_gpu_requested', ''))} | "
                f"{fmt(row.get('full_scf_gpu_executed', ''))} | "
                f"{fmt(row.get('full_scf_gpu_ready', ''))} | "
                f"{row.get('full_scf_gpu_status', '')} | "
                f"{row.get('full_scf_gpu_reason', '')} | "
                f"{fmt(row.get('full_scf_gpu_backend_ready', ''))} | "
                f"{fmt(row.get('full_scf_gpu_resident', ''))} | "
                f"{fmt(row.get('full_scf_gpu_device_id', ''))} | "
                f"{fmt(row.get('full_scf_gpu_stage_completed', ''))} | "
                f"{fmt(row.get('full_scf_gpu_stage_missing', ''))} | "
                f"{row.get('full_scf_gpu_stage_missing_names', '')} | "
                f"{fmt(row.get('full_scf_gpu_isitsc_okscf', ''))} | "
                f"{fmt(row.get('full_scf_gpu_wall_ms', ''))} | "
                f"{fmt(row.get('full_scf_gpu_density_max', ''))} | "
                f"{fmt(row.get('full_scf_gpu_density_rms', ''))} | "
                f"{fmt(row.get('full_scf_gpu_final_iterations', ''))} | "
                f"{fmt(row.get('full_scf_gpu_final_density_resident', ''))} |"
                f"{fmt(row.get('full_scf_gpu_cosmo_enabled', ''))} | "
                f"{fmt(row.get('full_scf_gpu_cosmo_fock_calls', ''))} | "
                f"{fmt(row.get('full_scf_gpu_cosmo_matvec_calls', ''))} | "
                f"{row.get('full_scf_gpu_cpu_mutating_sections', '')} | "
                f"{fmt(row.get('full_scf_gpu_cpu_mutating_section_count', ''))} |"
            )
    gpu_debug_rows = [
        row for row in rows
        if row.get("mode") == "GPU" and row.get("gpu_lgpu_final") not in ("", None)
    ]
    if gpu_debug_rows:
        lines.extend(
            [
                "",
                "## GPU Runtime State",
                "",
                "| Molecule | Rep | hasGPU | devices | final lgpu | requested | enabled | reason | resident SCF | resident Fock | Fock1 batch | Fock2 4x1 batch | Fock GPU | Check GPU | Plan max block | Plan eligible density pairs | Resident pairs | Point pairs | Point dipole | Point monopole | Fock candidate tasks | Fock production tasks |",
                "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in gpu_debug_rows:
            lines.append(
                f"| {row['molecule']} | {row['repeat']} | {row.get('gpu_has_device', '')} | "
                f"{fmt(row.get('gpu_device_count', ''))} | {row.get('gpu_lgpu_final', '')} | "
                f"{row.get('mozyme_gpu_requested', '')} | {row.get('mozyme_plan_enabled', '')} | "
                f"{row.get('mozyme_plan_disable_reason_text', '')} | "
                f"{row.get('gpu_resident_scf_final', '')} | {row.get('mozyme_resident_fock_gpu_profile', '')} | "
                f"{row.get('mozyme_fock1_batch_gpu_profile', '')} | "
                f"{row.get('mozyme_fock2_4x1_batch_gpu_profile', '')} | "
                f"{row.get('mozyme_fock_gpu_profile', '')} | "
                f"{row.get('mozyme_check_gpu_profile', '')} | "
                f"{fmt(row.get('mozyme_plan_max_block', ''))} | "
                f"{fmt(row.get('mozyme_plan_density_pairs_meeting_minblk', ''))} | "
                f"{fmt(row.get('mozyme_fock_resident_supported_pairs', ''))} | "
                f"{fmt(row.get('mozyme_fock_plan_point_charge_pairs', ''))} | "
                f"{fmt(row.get('mozyme_fock_plan_point_dipole_pairs', ''))} | "
                f"{fmt(row.get('mozyme_fock_plan_point_monopole_pairs', ''))} | "
                f"{fmt(row.get('mozyme_fock_candidate_gpu_tasks', ''))} | "
                f"{fmt(row.get('mozyme_fock_production_gpu_tasks', ''))} |"
            )
    if legacy_rows:
        lines.extend(
            [
                "",
                "## Existing Reference Outputs",
                "",
                "These rows were parsed from outputs already present in the repository before this benchmark run.",
                "",
                "| Name | Source | HoF kcal/mol | Wall s | Total job s | Notes |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for row in legacy_rows[:30]:
            lines.append(
                f"| {row.get('name', '')} | `{row.get('source_path', '')}` | "
                f"{fmt(row.get('heat_kcal_mol', ''))} | {fmt(row.get('wall_clock_s', ''))} | "
                f"{fmt(row.get('total_job_s', ''))} | {row.get('notes', '')} |"
            )
    lines.extend(
        [
            "",
            "## Data Files",
            "",
            "- `molecule_runs.csv`: every CPU/GPU repeat with return code, wall time, parsed heat, and log path.",
            "  With `--gpu-profile`, it also includes MOZYME GPU offload counters and `full_scf_gpu_*` readiness fields.",
            "- `mozyme_section_times.csv`: per-run MOZYME section timing markers; header-only when the executable does not emit them.",
            "- `gpu_preflight.json`: single-molecule preflight result used to decide whether to continue.",
            "- `gpu_preflight_failure.txt`: written only when the benchmark is aborted before the full run.",
            "- `full_scf_gpu_readiness.json`: written when `--require-full-scf-gpu` is used; records the strict readiness probe.",
            "- `full_scf_gpu_readiness_failure.txt`: written only when the strict complete-SCF probe fails.",
            "- `molecule_summary.csv`: best CPU/GPU times, production GPU speedup, and energy agreement per molecule.",
            "- `summary.json`: machine-readable copy of all rows and summaries.",
            "- `existing_mopac_references.csv/json`: historical outputs already present in the repository, if available.",
            "- `logs/`: raw stdout logs from each run.",
            "- `*.png`: generated plots for publication and analysis.",
            "",
            "## Plots",
            "",
        ]
    )
    for plot in plots:
        lines.append(f"- `{plot.name}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.6g}"


def read_legacy_reference_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_legacy_references(csv_path: Path, json_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    legacy_rows = read_legacy_reference_csv(csv_path)
    if csv_path.exists():
        shutil.copy2(csv_path, out_dir / "existing_mopac_references.csv")
    if json_path.exists():
        shutil.copy2(json_path, out_dir / "existing_mopac_references.json")
    return legacy_rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cmake_cache_value(cache_text: str, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}(?::[^=]*)?=(.*)$", re.MULTILINE)
    match = pattern.search(cache_text)
    return match.group(1).strip() if match else ""


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def parse_source_manifest(text: str) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    violations: list[str] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\n")
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            violations.append(f"invalid manifest line {line_no}")
            continue
        digest, rel = match.groups()
        rel_path = Path(rel)
        if rel.startswith("/") or "\\" in rel or ".." in rel_path.parts:
            violations.append(f"unsafe manifest path {rel!r}")
            continue
        if rel in {SOURCE_MANIFEST_NAME, SOURCE_FEATURES_NAME, SOURCE_PROVENANCE_NAME}:
            violations.append(f"manifest unexpectedly lists metadata file {rel!r}")
            continue
        if rel in entries:
            violations.append(f"duplicate manifest path {rel!r}")
            continue
        entries[rel] = digest
    return entries, violations


def preview_items(values: list[str], limit: int = 5) -> str:
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f", ... (+{len(values) - limit} more)"


def verify_manifest_source_files(source_root: Path, entries: dict[str, str]) -> tuple[int, list[str], list[str]]:
    checked = 0
    missing: list[str] = []
    mismatched: list[str] = []
    for rel, expected_sha in entries.items():
        path = source_root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        checked += 1
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            mismatched.append(rel)
    return checked, missing, mismatched


def verify_critical_source_files(
    source_root: Path,
    features: dict[str, Any],
    manifest_entries: dict[str, str],
) -> tuple[int, int, list[str]]:
    critical = features.get("critical_files")
    if not isinstance(critical, dict):
        return 0, 0, ["features critical_files is missing or not an object"]

    verified = 0
    violations: list[str] = []
    for rel, metadata in sorted(critical.items()):
        if not isinstance(metadata, dict):
            violations.append(f"{rel}: critical metadata is not an object")
            continue
        path = source_root / rel
        if not path.is_file():
            violations.append(f"{rel}: missing critical source file")
            continue
        expected_sha = str(metadata.get("sha256") or "")
        expected_size = parse_int_value(metadata.get("size"))
        actual_sha = file_sha256(path)
        actual_size = path.stat().st_size
        if manifest_entries.get(rel) != expected_sha:
            violations.append(f"{rel}: critical sha is not represented in source manifest")
            continue
        if actual_sha != expected_sha:
            violations.append(f"{rel}: critical source sha mismatch")
            continue
        if expected_size is None or actual_size != expected_size:
            violations.append(f"{rel}: critical source size mismatch")
            continue
        verified += 1
    return len(critical), verified, violations


def source_marker_present(text: str, flat_text: str, fragment: object) -> bool:
    value = str(fragment)
    return value in text or " ".join(value.split()) in flat_text


def ordered_source_marker_missing(flat_text: str, fragments: list[Any]) -> str | None:
    cursor = 0
    for fragment in fragments:
        normalized = " ".join(str(fragment).split())
        index = flat_text.find(normalized, cursor)
        if index < 0:
            return normalized
        cursor = index + len(normalized)
    return None


def verify_required_source_markers(
    source_root: Path,
    features: dict[str, Any],
    manifest_entries: dict[str, str],
    archive: zipfile.ZipFile | None = None,
) -> tuple[int, int, list[str]]:
    markers = features.get("required_source_markers")
    if not isinstance(markers, list):
        return 0, 0, ["required_source_markers is missing or not a list"]
    if not markers:
        return 0, 0, ["required_source_markers is empty"]

    archive_names = set(archive.namelist()) if archive is not None else set()
    names_seen: set[str] = set()
    verified = 0
    violations: list[str] = []
    for index, marker in enumerate(markers):
        if not isinstance(marker, dict):
            violations.append(f"required_source_markers[{index}] is not an object")
            continue
        name = str(marker.get("name") or f"#{index}")
        if name in names_seen:
            violations.append(f"{name}: duplicate source marker")
            continue
        names_seen.add(name)
        rel = marker.get("path")
        if not isinstance(rel, str) or not rel:
            violations.append(f"{name}: marker path is missing")
            continue
        if rel not in manifest_entries:
            violations.append(f"{name}: marker path is not represented in source manifest")
            continue

        try:
            if archive is not None:
                if rel not in archive_names:
                    violations.append(f"{name}: marker source file missing from source zip")
                    continue
                text = archive.read(rel).decode("utf-8", errors="ignore")
            else:
                path = source_root / rel
                if not path.is_file():
                    violations.append(f"{name}: marker source file missing")
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, KeyError):
            violations.append(f"{name}: marker source file could not be read")
            continue

        marker_violations: list[str] = []
        flat_text = " ".join(text.split())
        fragments = marker.get("fragments")
        if not isinstance(fragments, list) or not fragments:
            marker_violations.append("fragments missing")
        else:
            for fragment in fragments:
                if not source_marker_present(text, flat_text, fragment):
                    marker_violations.append(f"missing fragment {str(fragment)!r}")
                    break

        ordered_fragments = marker.get("ordered_fragments") or []
        if not isinstance(ordered_fragments, list):
            marker_violations.append("ordered_fragments is not a list")
        else:
            missing = ordered_source_marker_missing(flat_text, ordered_fragments)
            if missing is not None:
                marker_violations.append(f"ordered fragment missing or out of order {missing!r}")

        forbidden_fragments = marker.get("forbidden_fragments") or []
        if not isinstance(forbidden_fragments, list):
            marker_violations.append("forbidden_fragments is not a list")
        else:
            for fragment in forbidden_fragments:
                if source_marker_present(text, flat_text, fragment):
                    marker_violations.append(f"forbidden fragment {str(fragment)!r}")
                    break

        if marker_violations:
            violations.append(f"{name}: " + ", ".join(marker_violations))
        else:
            verified += 1

    if "mozyme_gpu_relocal_fortran_wrapper" not in names_seen:
        violations.append("mozyme_gpu_relocal_fortran_wrapper marker is missing")
    return len(markers), verified, violations


def zip_member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_zip_files(
    archive: zipfile.ZipFile,
    entries: dict[str, str],
) -> tuple[int, list[str], list[str]]:
    names = set(archive.namelist())
    checked = 0
    missing: list[str] = []
    mismatched: list[str] = []
    for rel, expected_sha in entries.items():
        if rel not in names:
            missing.append(rel)
            continue
        checked += 1
        if zip_member_sha256(archive, rel) != expected_sha:
            mismatched.append(rel)
    return checked, missing, mismatched


def verify_critical_zip_files(
    archive: zipfile.ZipFile,
    features: dict[str, Any],
    manifest_entries: dict[str, str],
) -> tuple[int, int, list[str]]:
    critical = features.get("critical_files")
    if not isinstance(critical, dict):
        return 0, 0, ["features critical_files is missing or not an object"]

    names = set(archive.namelist())
    verified = 0
    violations: list[str] = []
    for rel, metadata in sorted(critical.items()):
        if not isinstance(metadata, dict):
            violations.append(f"{rel}: critical metadata is not an object")
            continue
        if rel not in names:
            violations.append(f"{rel}: missing critical zip member")
            continue
        expected_sha = str(metadata.get("sha256") or "")
        expected_size = parse_int_value(metadata.get("size"))
        actual_sha = zip_member_sha256(archive, rel)
        actual_size = archive.getinfo(rel).file_size
        if manifest_entries.get(rel) != expected_sha:
            violations.append(f"{rel}: critical sha is not represented in source manifest")
            continue
        if actual_sha != expected_sha:
            violations.append(f"{rel}: critical zip sha mismatch")
            continue
        if expected_size is None or actual_size != expected_size:
            violations.append(f"{rel}: critical zip size mismatch")
            continue
        verified += 1
    return len(critical), verified, violations


def collect_packaged_source_identity() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[1]
    manifest_path = source_root / SOURCE_MANIFEST_NAME
    features_path = source_root / SOURCE_FEATURES_NAME
    provenance_path = source_root / SOURCE_PROVENANCE_NAME
    source_zip_path = os.environ.get("MOPAC_COLAB_SOURCE_ZIP_PATH", "")
    source_zip = Path(source_zip_path) if source_zip_path else None
    metadata_bytes: dict[str, bytes] = {}
    metadata_loaded_from_zip = False
    zip_file_set_violations: list[str] = []
    if source_zip is not None and source_zip.is_file():
        try:
            with zipfile.ZipFile(source_zip, "r") as archive:
                names = archive.namelist()
                duplicate_names = sorted({name for name in names if names.count(name) > 1})
                for name in (SOURCE_MANIFEST_NAME, SOURCE_FEATURES_NAME, SOURCE_PROVENANCE_NAME):
                    if name in names:
                        metadata_bytes[name] = archive.read(name)
                if SOURCE_MANIFEST_NAME in metadata_bytes:
                    entries_for_set, _ = parse_source_manifest(
                        metadata_bytes[SOURCE_MANIFEST_NAME].decode("utf-8", errors="replace")
                    )
                    expected_names = set(entries_for_set)
                    expected_names.update(
                        {SOURCE_MANIFEST_NAME, SOURCE_FEATURES_NAME, SOURCE_PROVENANCE_NAME}
                    )
                    actual_names = set(names)
                    missing_zip_members = sorted(expected_names - actual_names)
                    extra_zip_members = sorted(actual_names - expected_names)
                    if missing_zip_members:
                        zip_file_set_violations.append(
                            "zip missing manifest member(s): "
                            + preview_items(missing_zip_members)
                        )
                    if extra_zip_members:
                        zip_file_set_violations.append(
                            "zip contains unmanifested member(s): "
                            + preview_items(extra_zip_members)
                        )
                if duplicate_names:
                    zip_file_set_violations.append(
                        "zip contains duplicate member(s): " + preview_items(duplicate_names)
                    )
                metadata_loaded_from_zip = bool(metadata_bytes)
        except (OSError, zipfile.BadZipFile, KeyError):
            metadata_bytes = {}
            zip_file_set_violations.append("source zip metadata could not be read")
    if not metadata_bytes:
        if manifest_path.exists():
            metadata_bytes[SOURCE_MANIFEST_NAME] = manifest_path.read_bytes()
        if features_path.exists():
            metadata_bytes[SOURCE_FEATURES_NAME] = features_path.read_bytes()
        if provenance_path.exists():
            metadata_bytes[SOURCE_PROVENANCE_NAME] = provenance_path.read_bytes()
    metadata_present = all(
        name in metadata_bytes
        for name in (SOURCE_MANIFEST_NAME, SOURCE_FEATURES_NAME, SOURCE_PROVENANCE_NAME)
    )
    identity: dict[str, Any] = {
        "source_metadata_present": int(metadata_present),
        "source_metadata_valid": 0,
        "source_metadata_from_zip": int(metadata_loaded_from_zip),
        "source_manifest_file_sha256": hashlib.sha256(metadata_bytes.get(SOURCE_MANIFEST_NAME, b"")).hexdigest()
        if SOURCE_MANIFEST_NAME in metadata_bytes
        else "",
        "source_features_file_sha256": hashlib.sha256(metadata_bytes.get(SOURCE_FEATURES_NAME, b"")).hexdigest()
        if SOURCE_FEATURES_NAME in metadata_bytes
        else "",
        "source_provenance_file_sha256": hashlib.sha256(metadata_bytes.get(SOURCE_PROVENANCE_NAME, b"")).hexdigest()
        if SOURCE_PROVENANCE_NAME in metadata_bytes
        else "",
        "source_provenance_manifest_sha256": "",
        "source_features_manifest_sha256": "",
        "source_provenance_git_commit": "",
        "source_provenance_git_dirty": "",
        "source_provenance_dirty_status_sha256": "",
        "source_provenance_generated_at_utc": "",
        "source_provenance_contract_version": "",
        "source_features_contract_version": "",
        "source_provenance_marker_contract_version": "",
        "source_features_marker_contract_version": "",
        "source_provenance_marker_contract_sha256": "",
        "source_features_marker_contract_sha256": "",
        "source_features_feature_set": "",
        "source_manifest_entry_count": 0,
        "source_manifest_verified_files": 0,
        "source_manifest_missing_file_count": 0,
        "source_manifest_hash_mismatch_count": 0,
        "source_critical_file_count": 0,
        "source_critical_files_verified": 0,
        "source_required_marker_count": 0,
        "source_required_markers_verified": 0,
        "source_required_marker_violation_count": 0,
        "source_required_critical_file_count": len(REQUIRED_CRITICAL_SOURCE_FILES),
        "source_required_critical_missing_count": len(REQUIRED_CRITICAL_SOURCE_FILES),
        "source_required_critical_missing_files": ";".join(REQUIRED_CRITICAL_SOURCE_FILES),
        "source_metadata_violation_reasons": "",
    }
    if not metadata_present:
        identity["source_metadata_violation_reasons"] = "missing packaged source metadata"
        return identity

    try:
        features_data = json.loads(metadata_bytes[SOURCE_FEATURES_NAME].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        features_data = {}
    try:
        provenance_data = json.loads(metadata_bytes[SOURCE_PROVENANCE_NAME].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        provenance_data = {}
    features = features_data if isinstance(features_data, dict) else {}
    provenance = provenance_data if isinstance(provenance_data, dict) else {}
    try:
        manifest_text = metadata_bytes[SOURCE_MANIFEST_NAME].decode("utf-8")
    except UnicodeDecodeError:
        manifest_text = ""
    manifest_entries, manifest_violations = parse_source_manifest(manifest_text)
    critical_manifest = features.get("critical_files")
    if isinstance(critical_manifest, dict):
        critical_names = {str(name) for name in critical_manifest}
    else:
        critical_names = set()
    required_critical_missing = sorted(set(REQUIRED_CRITICAL_SOURCE_FILES) - critical_names)
    if source_zip is not None and source_zip.is_file():
        try:
            with zipfile.ZipFile(source_zip, "r") as archive:
                verified_files, missing_files, mismatched_files = verify_manifest_zip_files(
                    archive, manifest_entries
                )
                critical_count, critical_verified, critical_violations = verify_critical_zip_files(
                    archive, features, manifest_entries
                )
                marker_count, marker_verified, marker_violations = verify_required_source_markers(
                    source_root, features, manifest_entries, archive
                )
        except (OSError, zipfile.BadZipFile, KeyError):
            verified_files, missing_files, mismatched_files = 0, list(manifest_entries), []
            critical_count, critical_verified, critical_violations = 0, 0, [
                "source zip members could not be verified"
            ]
            marker_count, marker_verified, marker_violations = 0, 0, [
                "source zip markers could not be verified"
            ]
    else:
        verified_files, missing_files, mismatched_files = verify_manifest_source_files(
            source_root, manifest_entries
        )
        critical_count, critical_verified, critical_violations = verify_critical_source_files(
            source_root, features, manifest_entries
        )
        marker_count, marker_verified, marker_violations = verify_required_source_markers(
            source_root, features, manifest_entries
        )
    feature_manifest = features.get("manifest") if isinstance(features.get("manifest"), dict) else {}
    provenance_manifest = (
        provenance.get("manifest") if isinstance(provenance.get("manifest"), dict) else {}
    )
    provenance_git = provenance.get("git") if isinstance(provenance.get("git"), dict) else {}
    identity.update(
        {
            "source_provenance_manifest_sha256": str(provenance_manifest.get("sha256") or ""),
            "source_features_manifest_sha256": str(feature_manifest.get("sha256") or ""),
            "source_provenance_git_commit": str(provenance_git.get("commit") or ""),
            "source_provenance_git_dirty": str(bool(provenance_git.get("dirty"))).lower()
            if "dirty" in provenance_git
            else "",
            "source_provenance_dirty_status_sha256": str(
                provenance_git.get("dirty_status_sha256") or ""
            ),
            "source_provenance_generated_at_utc": str(provenance.get("generated_at_utc") or ""),
            "source_provenance_contract_version": str(
                provenance.get("full_scf_contract_version") or ""
            ),
            "source_features_contract_version": str(
                features.get("full_scf_contract_version") or ""
            ),
            "source_provenance_marker_contract_version": str(
                provenance.get("source_marker_contract_version") or ""
            ),
            "source_features_marker_contract_version": str(
                features.get("source_marker_contract_version") or ""
            ),
            "source_provenance_marker_contract_sha256": str(
                provenance.get("source_marker_contract_sha256") or ""
            ),
            "source_features_marker_contract_sha256": str(
                features.get("source_marker_contract_sha256") or ""
            ),
            "source_features_feature_set": str(features.get("feature_set") or ""),
            "source_manifest_entry_count": len(manifest_entries),
            "source_manifest_verified_files": verified_files,
            "source_manifest_missing_file_count": len(missing_files),
            "source_manifest_hash_mismatch_count": len(mismatched_files),
            "source_critical_file_count": critical_count,
            "source_critical_files_verified": critical_verified,
            "source_required_marker_count": marker_count,
            "source_required_markers_verified": marker_verified,
            "source_required_marker_violation_count": len(marker_violations),
            "source_required_critical_file_count": len(REQUIRED_CRITICAL_SOURCE_FILES),
            "source_required_critical_missing_count": len(required_critical_missing),
            "source_required_critical_missing_files": ";".join(required_critical_missing),
        }
    )

    violations = []
    violations.extend(zip_file_set_violations)
    if manifest_violations:
        violations.append("manifest parse failed: " + preview_items(manifest_violations))
    expected_file_count = parse_int_value(provenance_manifest.get("file_count"))
    if expected_file_count is None or expected_file_count != len(manifest_entries):
        violations.append(
            "provenance manifest file_count mismatch "
            f"({provenance_manifest.get('file_count')!r} != {len(manifest_entries)})"
        )
    feature_file_count = parse_int_value(feature_manifest.get("file_count"))
    if feature_file_count is None or feature_file_count != len(manifest_entries):
        violations.append(
            "features manifest file_count mismatch "
            f"({feature_manifest.get('file_count')!r} != {len(manifest_entries)})"
        )
    if missing_files:
        violations.append("manifest source files missing: " + preview_items(missing_files))
    if mismatched_files:
        violations.append("manifest source hashes mismatch: " + preview_items(mismatched_files))
    if critical_violations:
        violations.append("critical source files invalid: " + preview_items(critical_violations))
    if marker_violations:
        violations.append("required source markers invalid: " + preview_items(marker_violations))
    if required_critical_missing:
        violations.append(
            "required critical source files are missing from the feature manifest: "
            + preview_items(required_critical_missing)
        )
    if provenance.get("schema") != SOURCE_PROVENANCE_SCHEMA:
        violations.append("provenance schema mismatch")
    if features.get("schema") != SOURCE_FEATURES_SCHEMA:
        violations.append("features schema mismatch")
    if provenance.get("zip_source_tree_marker") != ZIP_SOURCE_TREE_MARKER:
        violations.append("provenance source marker mismatch")
    if features.get("zip_source_tree_marker") != ZIP_SOURCE_TREE_MARKER:
        violations.append("features source marker mismatch")
    if identity["source_provenance_contract_version"] != MOPAC_GPU_READINESS_CONTRACT_VERSION:
        violations.append("provenance contract version mismatch")
    if identity["source_features_contract_version"] != MOPAC_GPU_READINESS_CONTRACT_VERSION:
        violations.append("features contract version mismatch")
    if identity["source_provenance_marker_contract_version"] != SOURCE_MARKER_CONTRACT_VERSION:
        violations.append("provenance source marker contract version mismatch")
    if identity["source_features_marker_contract_version"] != SOURCE_MARKER_CONTRACT_VERSION:
        violations.append("features source marker contract version mismatch")
    if identity["source_provenance_marker_contract_sha256"] != SOURCE_MARKER_CONTRACT_SHA256:
        violations.append("provenance source marker contract sha mismatch")
    if identity["source_features_marker_contract_sha256"] != SOURCE_MARKER_CONTRACT_SHA256:
        violations.append("features source marker contract sha mismatch")
    if identity["source_features_feature_set"] != MOPAC_GPU_FEATURE_SET:
        violations.append("features feature set mismatch")
    if identity["source_provenance_manifest_sha256"] != identity["source_manifest_file_sha256"]:
        violations.append("provenance manifest sha mismatch")
    if identity["source_features_manifest_sha256"] != identity["source_manifest_file_sha256"]:
        violations.append("features manifest sha mismatch")
    identity["source_metadata_valid"] = int(not violations)
    identity["source_metadata_violation_reasons"] = "; ".join(violations)
    return identity


def collect_proof_identity(mopac: Path) -> dict[str, Any]:
    build_dir = mopac.parent
    cache_text = ""
    cache_path = build_dir / "CMakeCache.txt"
    if cache_path.exists():
        cache_text = cache_path.read_text(encoding="utf-8", errors="ignore")
    packaged_identity = collect_packaged_source_identity()
    source_manifest_file_sha = str(packaged_identity.get("source_manifest_file_sha256") or "")
    source_provenance_git_commit = str(
        packaged_identity.get("source_provenance_git_commit") or ""
    )
    source_provenance_git_dirty = str(
        packaged_identity.get("source_provenance_git_dirty") or ""
    )
    source_provenance_dirty_sha = str(
        packaged_identity.get("source_provenance_dirty_status_sha256") or ""
    )
    source_provenance_generated_at = str(
        packaged_identity.get("source_provenance_generated_at_utc") or ""
    )
    source_zip_path = os.environ.get("MOPAC_COLAB_SOURCE_ZIP_PATH", "")
    source_zip_env_sha = os.environ.get("MOPAC_COLAB_SOURCE_ZIP_SHA256", "")
    source_zip_verified = 0
    source_zip_computed_sha = ""
    if source_zip_path:
        path = Path(source_zip_path)
        if path.exists() and path.is_file():
            source_zip_computed_sha = file_sha256(path)
            source_zip_verified = 1
    identity: dict[str, Any] = {
        "source_zip_sha256": source_zip_computed_sha or source_zip_env_sha,
        "source_zip_sha256_env": source_zip_env_sha,
        "source_zip_path": source_zip_path,
        "source_zip_verified": source_zip_verified,
        "source_manifest_sha256": os.environ.get(
            "MOPAC_COLAB_SOURCE_MANIFEST_SHA256",
            source_manifest_file_sha,
        ),
        "source_git_commit": os.environ.get(
            "MOPAC_COLAB_SOURCE_GIT_COMMIT",
            source_provenance_git_commit,
        ),
        "source_git_dirty": os.environ.get(
            "MOPAC_COLAB_SOURCE_GIT_DIRTY",
            source_provenance_git_dirty,
        ),
        "source_dirty_status_sha256": os.environ.get(
            "MOPAC_COLAB_SOURCE_DIRTY_STATUS_SHA256",
            source_provenance_dirty_sha,
        ),
        "source_generated_at_utc": os.environ.get(
            "MOPAC_COLAB_SOURCE_GENERATED_AT_UTC",
            source_provenance_generated_at,
        ),
        "mopac_executable": str(mopac),
        "mopac_executable_sha256": file_sha256(mopac) if mopac.exists() else "",
        "cmake_gpu_bool": read_cmake_cache_value(cache_text, "GPU"),
        "cmake_cuda_architectures": read_cmake_cache_value(cache_text, "CMAKE_CUDA_ARCHITECTURES"),
        "cmake_cuda_archs": read_cmake_cache_value(cache_text, "CUDA_ARCHS"),
    }
    identity.update(packaged_identity)
    return identity


def proof_identity_lines(identity: dict[str, Any]) -> list[str]:
    keys = (
        "source_zip_sha256",
        "source_zip_sha256_env",
        "source_zip_path",
        "source_zip_verified",
        "source_manifest_sha256",
        "source_manifest_file_sha256",
        "source_metadata_present",
        "source_metadata_valid",
        "source_metadata_from_zip",
        "source_metadata_violation_reasons",
        "source_manifest_entry_count",
        "source_manifest_verified_files",
        "source_manifest_missing_file_count",
        "source_manifest_hash_mismatch_count",
        "source_critical_file_count",
        "source_critical_files_verified",
        "source_required_marker_count",
        "source_required_markers_verified",
        "source_required_marker_violation_count",
        "source_required_critical_file_count",
        "source_required_critical_missing_count",
        "source_required_critical_missing_files",
        "source_git_commit",
        "source_git_dirty",
        "source_dirty_status_sha256",
        "source_generated_at_utc",
        "source_provenance_git_commit",
        "source_provenance_git_dirty",
        "source_provenance_dirty_status_sha256",
        "source_provenance_generated_at_utc",
        "mopac_executable_sha256",
        "cmake_cuda_architectures",
        "cmake_cuda_archs",
        "cmake_gpu_bool",
    )
    return [f"- {key}: {identity.get(key, '')}" for key in keys]


def create_manifest(
    path: Path,
    files: list[Path],
    title: str = "MOPAC molecule benchmark publication/analysis bundle",
    usage_lines: list[str] | None = None,
    proof_identity: dict[str, Any] | None = None,
) -> None:
    lines = [
        title,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Files:",
    ]
    for item in sorted(files):
        lines.append(f"- {item.relative_to(path.parent)}")
    lines.append("")
    if proof_identity:
        lines.extend(("Proof identity:", *proof_identity_lines(proof_identity), ""))
    if usage_lines is None:
        usage_lines = [
            "Use molecule_summary.csv for figure-ready molecule speedups and energy differences.",
            "Use molecule_runs.csv and logs/ for reproducibility and deeper analysis.",
        ]
    lines.extend(usage_lines)
    path.write_text("\n".join(lines), encoding="utf-8")


def create_bundle_zip(bundle_path: Path, out_dir: Path) -> Path:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if not path.is_file() or path.resolve() == bundle_path.resolve():
                continue
            zf.write(path, arcname=f"{out_dir.name}/{path.relative_to(out_dir)}")
    return bundle_path


def input_requests_mozyme_gpu(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    head = "\n".join(text.splitlines()[:8]).upper()
    return "MOZYME" in head


def select_mozyme_preflight_input(inputs: list[Path]) -> Path:
    for path in inputs:
        if input_requests_mozyme_gpu(path):
            return path
    return inputs[0]


def stop_process(proc: subprocess.Popen[str], timed_out: bool) -> int:
    if proc.poll() is None:
        if timed_out:
            proc.kill()
        else:
            proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
    return proc.returncode if proc.returncode is not None else 124


def run_full_scf_gpu_readiness_probe(
    mopac: Path,
    input_path: Path,
    out_dir: Path,
    timeout: float,
    bundle_path: Path | None,
    mozyme_section_profile: bool,
    require_direct_cosmo_gpu: bool = False,
) -> dict[str, Any]:
    print("")
    print(f"Full SCF GPU readiness probe: {input_path.name}", flush=True)
    run_root = out_dir / "runs"
    run_dir = run_root / input_path.stem / "gpu_full_scf_probe" / "rep0"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    staged_input = stage_input(input_path, run_dir)
    forced_probe_keywords = force_full_scf_probe_keywords(staged_input)
    forced_probe_env = ";".join(FULL_SCF_PROBE_FORCED_ENV)
    env = build_mopac_env(
        MODES[1],
        verbose_gpu=True,
        gpu_profile=True,
        mozyme_section_profile=mozyme_section_profile,
        full_scf_gpu=True,
    )

    cmd = [str(mopac), staged_input.name]
    print(f"[GPU full-SCF probe] {input_path.name}: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    if proc.stdout is None:
        raise SystemExit("Could not capture full SCF GPU readiness probe output.")

    chunks: list[str] = []
    decision = ""
    saw_success = False
    saw_scf_code = False
    saw_scf_stage = False
    saw_scf_decision_complete = False
    saw_scf_isitsc = False
    saw_scf_status_iterations = False
    saw_scf_final_density = False
    saw_makvec_success = False
    saw_makvec_existing = False
    saw_makvec_fallback = False
    timed_out = False
    while True:
        now = time.perf_counter()
        if now - t0 >= timeout:
            timed_out = True
            break
        if proc.poll() is not None:
            rest = proc.stdout.read()
            if rest:
                chunks.append(rest)
                print(rest, end="")
            break

        wait_s = min(0.25, max(0.0, timeout - (now - t0)))
        ready, _, _ = select.select([proc.stdout], [], [], wait_s)
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            continue
        chunks.append(line)
        print(line, end="")
        saw_scf_code = saw_scf_code or bool(MOZYME_SCF_CODE_RE.search(line))
        saw_scf_stage = saw_scf_stage or bool(MOZYME_SCF_STAGE_RE.search(line))
        decision_match = MOZYME_SCF_DECISION_RE.search(line)
        if decision_match:
            saw_scf_decision_complete = (
                parse_int_value(decision_match.group(1))
                == MOZYME_SCF_RESIDENT_DECISION_COMPLETE
            )
        saw_scf_isitsc = saw_scf_isitsc or bool(MOZYME_SCF_ISITSC_RE.search(line))
        saw_scf_status_iterations = saw_scf_status_iterations or bool(MOZYME_SCF_STATUS_ITERATIONS_RE.search(line))
        saw_scf_final_density = saw_scf_final_density or bool(MOZYME_SCF_FINAL_DENSITY_RE.search(line))
        saw_makvec_success = saw_makvec_success or bool(MOZYME_MAKVEC_SUCCESS_RE.search(line))
        saw_makvec_existing = saw_makvec_existing or bool(MOZYME_MAKVEC_EXISTING_RE.search(line))
        saw_makvec_fallback = saw_makvec_fallback or bool(MOZYME_MAKVEC_FALLBACK_RE.search(line))
        if saw_makvec_fallback:
            decision = "makvec_fallback_cpu"
            continue
        if (
            saw_success
            and saw_scf_code
            and saw_scf_stage
            and saw_scf_decision_complete
            and saw_scf_isitsc
            and saw_scf_status_iterations
            and saw_scf_final_density
            and saw_makvec_success
        ):
            decision = "complete"
            continue
        if saw_makvec_existing and not saw_makvec_success:
            decision = "makvec_existing_lmo_not_gpu_proof"
            continue
        if (
            saw_success
            and saw_scf_code
            and saw_scf_stage
            and saw_scf_decision_complete
            and saw_scf_isitsc
            and saw_scf_status_iterations
            and saw_scf_final_density
        ):
            decision = "success_pending_makvec"
            continue
        if (
            saw_success
            and saw_scf_code
            and saw_scf_stage
            and saw_scf_decision_complete
            and saw_scf_isitsc
            and saw_scf_status_iterations
        ):
            decision = "success_pending_final_density"
            continue
        if saw_success and saw_scf_code and saw_scf_stage and saw_scf_decision_complete and saw_scf_isitsc:
            decision = "success_pending_final_iterations"
            continue
        status_match = MOZYME_SCF_STATUS_RE.search(line)
        if not status_match:
            continue
        raw_status = status_match.group(1).lower()
        if raw_status == "success":
            saw_success = True
            if (
                saw_scf_code
                and saw_scf_stage
                and saw_scf_decision_complete
                and saw_scf_isitsc
                and saw_scf_status_iterations
                and saw_scf_final_density
                and saw_makvec_success
            ):
                decision = "complete"
            elif saw_makvec_existing and not saw_makvec_success:
                decision = "makvec_existing_lmo_not_gpu_proof"
            elif (
                saw_scf_code
                and saw_scf_stage
                and saw_scf_decision_complete
                and saw_scf_isitsc
                and saw_scf_status_iterations
                and saw_scf_final_density
            ):
                decision = "success_pending_makvec"
            elif (
                saw_scf_code
                and saw_scf_stage
                and saw_scf_decision_complete
                and saw_scf_isitsc
                and saw_scf_status_iterations
            ):
                decision = "success_pending_final_density"
            elif saw_scf_code and saw_scf_stage and saw_scf_decision_complete and saw_scf_isitsc:
                decision = "success_pending_final_iterations"
            elif saw_scf_code and saw_scf_stage:
                decision = "success_pending_resident_decision"
            else:
                decision = "success_pending_masks"
        elif raw_status == "resident_step":
            decision = "resident_step"
        elif raw_status == "fallback_cpu":
            decision = "fallback_cpu"
        elif raw_status == "strict_abort":
            decision = "strict_abort"

    returncode = stop_process(proc, timed_out)
    rest = proc.stdout.read()
    if rest:
        chunks.append(rest)
        print(rest, end="")
    elapsed = time.perf_counter() - t0
    stdout = "".join(chunks)

    log_path = out_dir / "logs" / f"{input_path.stem}.gpu_full_scf_probe.rep0.stdout.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout, encoding="utf-8", errors="ignore")

    combined_text = stdout + "\n" + read_outputs(run_dir)
    decision = classify_full_scf_probe_decision(combined_text) or decision
    archived_outputs = archive_output_files(run_dir, log_path.parent, f"{input_path.stem}.gpu_full_scf_probe.rep0")
    gpu_error_markers = gpu_error_markers_in_text(combined_text)
    heat = parse_heat(combined_text)
    reported_s = parse_reported_time(combined_text)
    mozyme_diag = parse_mozyme_gpu_diagnostics(combined_text)
    mozyme_diag["full_scf_gpu_requested"] = 1
    if mozyme_diag.get("full_scf_gpu_status") == "not_requested":
        mozyme_diag["full_scf_gpu_status"] = "not_executed"
    proof_identity = collect_proof_identity(mopac)
    mozyme_section_times = parse_mozyme_section_times(combined_text)
    (
        cpu_mutating_sections,
        cpu_mutating_call_count,
        cpu_mutating_ms,
    ) = mozyme_disallowed_strict_section_stats(mozyme_section_times, combined_text)
    normal_marker = "JOB ENDED NORMALLY" in combined_text or "== MOPAC DONE ==" in combined_text
    normal_end = normal_marker and heat is not None
    staged_input_eps = parse_input_eps(staged_input)
    shutil.rmtree(run_dir, ignore_errors=True)

    row = {
        "molecule": input_path.stem,
        "input": str(input_path),
        "input_eps": staged_input_eps or "",
        "requires_direct_cosmo_gpu": int(require_direct_cosmo_gpu),
        "mode": "GPU",
        "repeat": 0,
        "returncode": returncode,
        "timed_out": timed_out,
        "normal_end": normal_end,
        "wall_s": elapsed,
        "reported_s": reported_s if reported_s is not None else "",
        "heat_kcal_mol": heat if heat is not None else "",
        "atoms": count_atoms(input_path),
        **mozyme_diag,
        "gpu_error_markers": ";".join(gpu_error_markers),
        "gpu_error_marker_count": len(gpu_error_markers),
        "mozyme_section_times": mozyme_section_times,
        "full_scf_gpu_cpu_mutating_sections": ";".join(cpu_mutating_sections),
        "full_scf_gpu_cpu_mutating_section_count": len(cpu_mutating_sections),
        "full_scf_gpu_cpu_mutating_call_count": cpu_mutating_call_count,
        "full_scf_gpu_cpu_mutating_ms": cpu_mutating_ms,
        "log_path": str(log_path),
        "output_files": ";".join(str(path) for path in archived_outputs),
        "run_dir": "",
        "full_scf_probe_decision": decision,
        "full_scf_probe_forced_keywords": forced_probe_keywords,
        "full_scf_probe_forced_env": forced_probe_env,
        **proof_identity,
    }
    finalize_full_scf_gpu_contract(row)
    reasons = diagnose_full_scf_gpu_readiness(row)
    payload = {
        "contract_version": MOPAC_GPU_READINESS_CONTRACT_VERSION,
        "feature_set": MOPAC_GPU_FEATURE_SET,
        "benchmark_scope": MOPAC_GPU_BENCHMARK_SCOPE,
        "publication_claim": MOPAC_GPU_PUBLICATION_CLAIM,
        **proof_identity,
        "proof_identity": proof_identity,
        "input": str(input_path),
        "timeout_s": timeout,
        "row": row,
        "reasons": reasons,
        "strict_requirement": (
            "MOPAC must exit 0 and finish normally; full_scf_gpu_status must be complete; "
            f"full_scf_gpu_code must be 0; backend_ready and resident must be 1; "
            f"device_id must be nonnegative; stage_required must be {MOZYME_SCF_STAGE_FULL}; "
            "all stage bits must complete; "
            "stage_missing must be 0; ISITSC must converge; at least one resident iteration must complete; "
            "raw stage-name and strict_proof markers must confirm strict resident/no-fallback execution; "
            "strict resident host synchronization and control-poll counters must be reported as zero; "
            "resident Fock full/partial coverage and required/covered masks must prove every required sparse plan; "
            "final_density=current_resident must be reported; "
            "typed final publication must emit `[MOZYME GPU SCF] final_publication_done=1` "
            "with positive arrays and bytes; "
            "resident PLS restart must be unnecessary or completed on GPU; "
            "MOZYME makvec initial LMO construction must report GPU success, with no makvec fallback "
            "and no OLD_SCF existing-LMO shortcut; "
            "MOZYME setupk atom-list construction must report GPU success with initial_setup=1 "
            "and all_initial_setup_paths=1 and no fallback; "
            "resident SCF success count, runtime, energy, and DIAGG metrics must be parseable; "
            "MOZYME section profile markers must be present; "
            "CPU MOZYME state-mutating setup or bookend sections must not run; "
            "OLDEN/OLDENS host restore, CPU pinout, and .den host checkpoint artifacts must be absent; "
            "final host state publication must emit `[MOZYME GPU SCF] host_commit_only=1 "
            "phase=final_publication` with positive arrays and bytes; "
            "forced final reorthogonalization probes must emit a `[MOZYME GPU reorth] status=success resident=1` "
            "marker, and CPU final reorthogonalization sections must be absent; "
            "no resident-SCF fallback_cpu, strict_abort, or resident_step marker may appear anywhere in the log; "
            "resident sparse Fock must run real GPU work every resident iteration; "
            "point-charge/dipole work must be semantically covered when present; real-pair coverage must be all-GPU; "
            "ordinary EPS/COSMO rows must report resident COSMO Fock, matrix-free CG matvec calls, "
            "resident CG control/convergence, and zero host CG syncs; "
            "and all parsed fallback counters "
            "must be zero; raw output must not contain CUDA/cuBLAS/cuSOLVER error markers."
        ),
    }
    (out_dir / "full_scf_gpu_readiness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if reasons:
        lines = [
            "Complete GPU SCF readiness probe failed.",
            "",
            f"Input: {input_path}",
            f"Log: {row.get('log_path', '')}",
            "",
            "Publication claim:",
            MOPAC_GPU_PUBLICATION_CLAIM,
            "",
            "Reasons:",
        ]
        lines.extend(f"- {reason}" for reason in reasons)
        lines.extend(
            [
                "",
                "Parseable fields:",
                f"- full_scf_gpu_requested: {row.get('full_scf_gpu_requested', '')}",
                f"- full_scf_gpu_executed: {row.get('full_scf_gpu_executed', '')}",
                f"- full_scf_gpu_ready: {row.get('full_scf_gpu_ready', '')}",
                f"- full_scf_gpu_status: {row.get('full_scf_gpu_status', '')}",
                f"- full_scf_gpu_reason: {row.get('full_scf_gpu_reason', '')}",
                f"- full_scf_gpu_contract_violations: {row.get('full_scf_gpu_contract_violations', '')}",
                f"- full_scf_gpu_fallback: {row.get('full_scf_gpu_fallback', '')}",
                f"- full_scf_gpu_code: {row.get('full_scf_gpu_code', '')}",
                f"- full_scf_gpu_backend_ready: {row.get('full_scf_gpu_backend_ready', '')}",
                f"- full_scf_gpu_resident: {row.get('full_scf_gpu_resident', '')}",
                f"- full_scf_gpu_device_id: {row.get('full_scf_gpu_device_id', '')}",
                f"- gpu_error_marker_count: {row.get('gpu_error_marker_count', '')}",
                f"- gpu_error_markers: {row.get('gpu_error_markers', '')}",
                f"- full_scf_gpu_stage_completed: {row.get('full_scf_gpu_stage_completed', '')}",
                f"- full_scf_gpu_stage_required: {row.get('full_scf_gpu_stage_required', '')}",
                f"- full_scf_gpu_stage_missing: {row.get('full_scf_gpu_stage_missing', '')}",
                f"- full_scf_gpu_resident_decision: {row.get('full_scf_gpu_resident_decision', '')}",
                f"- full_scf_gpu_stage_completed_names: {row.get('full_scf_gpu_stage_completed_names', '')}",
                f"- full_scf_gpu_stage_missing_names: {row.get('full_scf_gpu_stage_missing_names', '')}",
                f"- full_scf_gpu_stage_completed_names_raw: {row.get('full_scf_gpu_stage_completed_names_raw', '')}",
                f"- full_scf_gpu_stage_missing_names_raw: {row.get('full_scf_gpu_stage_missing_names_raw', '')}",
                f"- full_scf_gpu_strict_resident: {row.get('full_scf_gpu_strict_resident', '')}",
                f"- full_scf_gpu_no_fallback_required: {row.get('full_scf_gpu_no_fallback_required', '')}",
                f"- full_scf_gpu_full_stage_mask: {row.get('full_scf_gpu_full_stage_mask', '')}",
                f"- full_scf_gpu_resident_decision_complete: {row.get('full_scf_gpu_resident_decision_complete', '')}",
                f"- full_scf_gpu_strict_resident_host_syncs: {row.get('full_scf_gpu_strict_resident_host_syncs', '')}",
                f"- full_scf_gpu_strict_resident_control_polls: {row.get('full_scf_gpu_strict_resident_control_polls', '')}",
                f"- full_scf_gpu_resident_fock_plan_id: {row.get('full_scf_gpu_resident_fock_plan_id', '')}",
                f"- full_scf_gpu_resident_fock_plan_full_coverage: {row.get('full_scf_gpu_resident_fock_plan_full_coverage', '')}",
                f"- full_scf_gpu_resident_fock_plan_partial_coverage: {row.get('full_scf_gpu_resident_fock_plan_partial_coverage', '')}",
                f"- full_scf_gpu_resident_fock_plan_required_mask: {row.get('full_scf_gpu_resident_fock_plan_required_mask', '')}",
                f"- full_scf_gpu_resident_fock_plan_covered_mask: {row.get('full_scf_gpu_resident_fock_plan_covered_mask', '')}",
                f"- full_scf_gpu_isitsc_okscf: {row.get('full_scf_gpu_isitsc_okscf', '')}",
                f"- full_scf_gpu_isitsc_iscf: {row.get('full_scf_gpu_isitsc_iscf', '')}",
                f"- full_scf_gpu_pls_supervisor_calls: {row.get('full_scf_gpu_pls_supervisor_calls', '')}",
                f"- full_scf_gpu_pls_restart_required: {row.get('full_scf_gpu_pls_restart_required', '')}",
                f"- full_scf_gpu_final_iterations: {row.get('full_scf_gpu_final_iterations', '')}",
                f"- full_scf_gpu_final_density_resident: {row.get('full_scf_gpu_final_density_resident', '')}",
                f"- mozyme_makvec_gpu_success_calls: {row.get('mozyme_makvec_gpu_success_calls', '')}",
                f"- mozyme_makvec_gpu_existing_lmo_calls: {row.get('mozyme_makvec_gpu_existing_lmo_calls', '')}",
                f"- mozyme_makvec_gpu_fallback_calls: {row.get('mozyme_makvec_gpu_fallback_calls', '')}",
                f"- mozyme_makvec_gpu_last_ms: {row.get('mozyme_makvec_gpu_last_ms', '')}",
                f"- mozyme_reorth_gpu_resident_success_calls: {row.get('mozyme_reorth_gpu_resident_success_calls', '')}",
                f"- mozyme_reorth_gpu_fallback_calls: {row.get('mozyme_reorth_gpu_fallback_calls', '')}",
                f"- full_scf_gpu_cpu_mutating_sections: {row.get('full_scf_gpu_cpu_mutating_sections', '')}",
                f"- full_scf_gpu_cpu_mutating_call_count: {row.get('full_scf_gpu_cpu_mutating_call_count', '')}",
                f"- full_scf_gpu_cpu_mutating_ms: {row.get('full_scf_gpu_cpu_mutating_ms', '')}",
                f"- full_scf_gpu_diagg_sumt: {row.get('full_scf_gpu_diagg_sumt', '')}",
                f"- full_scf_gpu_diagg_sumb: {row.get('full_scf_gpu_diagg_sumb', '')}",
                f"- full_scf_probe_decision: {row.get('full_scf_probe_decision', '')}",
                "",
                "Proof identity:",
                *proof_identity_lines(proof_identity),
            ]
        )
        output_tail = text_tail(combined_text, 180)
        if output_tail:
            try:
                proof_index = lines.index("Proof identity:")
            except ValueError:
                proof_index = len(lines)
            lines[proof_index:proof_index] = [
                "",
                "MOPAC stdout/output tail:",
                output_tail,
                "",
            ]
        failure_path = out_dir / "full_scf_gpu_readiness_failure.txt"
        failure_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        readme_path = out_dir / "README.md"
        readme_path.write_text("\n".join(["# MOPAC Complete GPU SCF Readiness", "", *lines]) + "\n", encoding="utf-8")
        files = [path for path in out_dir.rglob("*") if path.is_file()]
        manifest_path = out_dir / "MANIFEST.txt"
        create_manifest(
            manifest_path,
            files,
            title="MOPAC complete GPU SCF readiness diagnostic bundle",
            usage_lines=[
                "Use full_scf_gpu_readiness.json for machine-readable readiness fields and failure reasons.",
                "Use full_scf_gpu_readiness_failure.txt and logs/ for diagnostics.",
            ],
            proof_identity=proof_identity,
        )
        if bundle_path is not None:
            create_bundle_zip(bundle_path, out_dir)
            lines.append("")
            lines.append(f"Wrote full-SCF readiness diagnostic bundle: {bundle_path}")
        print("\n".join(lines), flush=True)
        raise SystemExit(2)

    print(
        "Full SCF GPU readiness passed: "
        f"status={row.get('full_scf_gpu_status', '')} "
        f"ready={row.get('full_scf_gpu_ready', '')} "
        f"resident={row.get('full_scf_gpu_resident', '')} "
        f"device_id={row.get('full_scf_gpu_device_id', '')} "
        f"isitsc_okscf={row.get('full_scf_gpu_isitsc_okscf', '')} "
        f"pls_restart_required={row.get('full_scf_gpu_pls_restart_required', '')} "
        f"iterations={row.get('full_scf_gpu_final_iterations', '')} "
        f"final_density_resident={row.get('full_scf_gpu_final_density_resident', '')}",
        flush=True,
    )
    success_lines = [
        "# MOPAC Complete GPU SCF Readiness",
        "",
        "Complete GPU SCF readiness probe passed.",
        "",
        f"Input: {input_path}",
        f"Log: {row.get('log_path', '')}",
        "",
        "Publication claim:",
        MOPAC_GPU_PUBLICATION_CLAIM,
        "",
        "Parseable fields:",
        f"- full_scf_gpu_status: {row.get('full_scf_gpu_status', '')}",
        f"- full_scf_gpu_ready: {row.get('full_scf_gpu_ready', '')}",
        f"- full_scf_gpu_contract_violations: {row.get('full_scf_gpu_contract_violations', '')}",
        f"- full_scf_gpu_backend_ready: {row.get('full_scf_gpu_backend_ready', '')}",
        f"- full_scf_gpu_resident: {row.get('full_scf_gpu_resident', '')}",
        f"- full_scf_gpu_device_id: {row.get('full_scf_gpu_device_id', '')}",
        f"- gpu_error_marker_count: {row.get('gpu_error_marker_count', '')}",
        f"- full_scf_gpu_stage_completed: {row.get('full_scf_gpu_stage_completed', '')}",
        f"- full_scf_gpu_stage_required: {row.get('full_scf_gpu_stage_required', '')}",
        f"- full_scf_gpu_stage_missing: {row.get('full_scf_gpu_stage_missing', '')}",
        f"- full_scf_gpu_resident_decision: {row.get('full_scf_gpu_resident_decision', '')}",
        f"- full_scf_gpu_stage_completed_names: {row.get('full_scf_gpu_stage_completed_names', '')}",
        f"- full_scf_gpu_stage_missing_names: {row.get('full_scf_gpu_stage_missing_names', '')}",
        f"- full_scf_gpu_stage_completed_names_raw: {row.get('full_scf_gpu_stage_completed_names_raw', '')}",
        f"- full_scf_gpu_stage_missing_names_raw: {row.get('full_scf_gpu_stage_missing_names_raw', '')}",
        f"- full_scf_gpu_strict_resident: {row.get('full_scf_gpu_strict_resident', '')}",
        f"- full_scf_gpu_no_fallback_required: {row.get('full_scf_gpu_no_fallback_required', '')}",
        f"- full_scf_gpu_full_stage_mask: {row.get('full_scf_gpu_full_stage_mask', '')}",
        f"- full_scf_gpu_resident_decision_complete: {row.get('full_scf_gpu_resident_decision_complete', '')}",
        f"- full_scf_gpu_strict_resident_host_syncs: {row.get('full_scf_gpu_strict_resident_host_syncs', '')}",
        f"- full_scf_gpu_strict_resident_control_polls: {row.get('full_scf_gpu_strict_resident_control_polls', '')}",
        f"- full_scf_gpu_resident_fock_plan_id: {row.get('full_scf_gpu_resident_fock_plan_id', '')}",
        f"- full_scf_gpu_resident_fock_plan_full_coverage: {row.get('full_scf_gpu_resident_fock_plan_full_coverage', '')}",
        f"- full_scf_gpu_resident_fock_plan_partial_coverage: {row.get('full_scf_gpu_resident_fock_plan_partial_coverage', '')}",
        f"- full_scf_gpu_resident_fock_plan_required_mask: {row.get('full_scf_gpu_resident_fock_plan_required_mask', '')}",
        f"- full_scf_gpu_resident_fock_plan_covered_mask: {row.get('full_scf_gpu_resident_fock_plan_covered_mask', '')}",
        f"- full_scf_gpu_isitsc_okscf: {row.get('full_scf_gpu_isitsc_okscf', '')}",
        f"- full_scf_gpu_pls_supervisor_calls: {row.get('full_scf_gpu_pls_supervisor_calls', '')}",
        f"- full_scf_gpu_pls_restart_required: {row.get('full_scf_gpu_pls_restart_required', '')}",
        f"- full_scf_gpu_final_iterations: {row.get('full_scf_gpu_final_iterations', '')}",
        f"- full_scf_gpu_final_density_resident: {row.get('full_scf_gpu_final_density_resident', '')}",
        f"- mozyme_makvec_gpu_success_calls: {row.get('mozyme_makvec_gpu_success_calls', '')}",
        f"- mozyme_makvec_gpu_existing_lmo_calls: {row.get('mozyme_makvec_gpu_existing_lmo_calls', '')}",
        f"- mozyme_makvec_gpu_fallback_calls: {row.get('mozyme_makvec_gpu_fallback_calls', '')}",
        f"- mozyme_makvec_gpu_last_ms: {row.get('mozyme_makvec_gpu_last_ms', '')}",
        f"- mozyme_reorth_gpu_resident_success_calls: {row.get('mozyme_reorth_gpu_resident_success_calls', '')}",
        f"- mozyme_reorth_gpu_fallback_calls: {row.get('mozyme_reorth_gpu_fallback_calls', '')}",
        f"- full_scf_gpu_cpu_mutating_sections: {row.get('full_scf_gpu_cpu_mutating_sections', '')}",
        f"- full_scf_gpu_cpu_mutating_call_count: {row.get('full_scf_gpu_cpu_mutating_call_count', '')}",
        f"- full_scf_gpu_cpu_mutating_ms: {row.get('full_scf_gpu_cpu_mutating_ms', '')}",
        f"- full_scf_probe_decision: {row.get('full_scf_probe_decision', '')}",
        "",
        "Resident SCF stage counters:",
        *(
            f"- {stage_name}: calls={row.get(f'full_scf_gpu_stage_{stage_name}_calls', '')} "
            f"ms={row.get(f'full_scf_gpu_stage_{stage_name}_ms', '')}"
            for stage_name in MOZYME_SCF_STAGE_NAMES
        ),
        "",
        "Proof identity:",
        *proof_identity_lines(proof_identity),
    ]
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(success_lines) + "\n", encoding="utf-8")
    files = [path for path in out_dir.rglob("*") if path.is_file()]
    manifest_path = out_dir / "MANIFEST.txt"
    create_manifest(
        manifest_path,
        files,
        title="MOPAC complete GPU SCF readiness bundle",
        usage_lines=[
            "Use README.md for the strict readiness summary.",
            "Use full_scf_gpu_readiness.json and logs/ for reproducibility and parsed fields.",
        ],
        proof_identity=proof_identity,
    )
    if bundle_path is not None:
        create_bundle_zip(bundle_path, out_dir)
    return row


def run_full_scf_readiness_cpu_compare(
    mopac: Path,
    input_path: Path,
    gpu_row: dict[str, Any],
    out_dir: Path,
    timeout: float,
    bundle_path: Path | None,
    mozyme_section_profile: bool,
    energy_abs_tol: float,
    energy_rel_tol: float,
    energy_per_atom_tol: float,
) -> dict[str, Any]:
    print("")
    print(f"Full SCF readiness CPU companion: {input_path.name}", flush=True)
    compare_input_dir = out_dir / "full_scf_readiness_cpu_compare_input"
    if compare_input_dir.exists():
        shutil.rmtree(compare_input_dir)
    compare_input_dir.mkdir(parents=True, exist_ok=True)
    cpu_probe_input = stage_input(input_path, compare_input_dir)
    forced_probe_keywords = force_full_scf_probe_keywords(cpu_probe_input)

    cpu_row = run_mopac(
        mopac,
        cpu_probe_input,
        MODES[0],
        -1,
        out_dir,
        timeout,
        keep_run_dirs=False,
        verbose_gpu=False,
        gpu_profile=False,
        mozyme_section_profile=mozyme_section_profile,
        full_scf_gpu=False,
    )
    summary_rows = summarize(
        [cpu_row, gpu_row],
        energy_abs_tol,
        energy_rel_tol,
        energy_per_atom_tol,
    )
    summary_row = summary_rows[0] if summary_rows else {}

    reasons: list[str] = []
    if cpu_row.get("returncode") != 0:
        reasons.append(f"CPU companion returned non-zero exit code {cpu_row.get('returncode')}")
    if cpu_row.get("normal_end") is not True:
        reasons.append("CPU companion did not finish normally with a parsed heat of formation")
    if cpu_row.get("heat_kcal_mol") == "":
        reasons.append("CPU companion heat of formation was not parseable")
    if gpu_row.get("heat_kcal_mol") == "":
        reasons.append("GPU readiness heat of formation was not parseable")
    cpu_only_reasons = cpu_companion_gpu_leakage_reasons(cpu_row)
    reasons.extend(cpu_only_reasons)
    if summary_row.get("energy_status") != "PASS":
        reasons.append(
            "strict readiness CPU/GPU heat comparison failed: "
            f"energy_status={summary_row.get('energy_status', '')} "
            f"abs_diff={summary_row.get('abs_heat_diff_kcal_mol', '')} "
            f"rel_diff={summary_row.get('rel_heat_diff', '')} "
            f"abs_per_atom={summary_row.get('abs_heat_diff_per_atom_kcal_mol', '')}"
        )
    else:
        accuracy_bases = {
            part
            for part in str(summary_row.get("accuracy_basis") or "").split("+")
            if part
        }
        if not (accuracy_bases & {"absolute", "relative"}):
            reasons.append(
                "strict readiness CPU/GPU heat comparison cannot pass on "
                "per-atom tolerance alone"
            )

    proof_identity = collect_proof_identity(mopac)
    payload = {
        "contract_version": MOPAC_GPU_READINESS_CONTRACT_VERSION,
        "feature_set": MOPAC_GPU_FEATURE_SET,
        "benchmark_scope": MOPAC_GPU_BENCHMARK_SCOPE,
        "publication_claim": MOPAC_GPU_PUBLICATION_CLAIM,
        **proof_identity,
        "proof_identity": proof_identity,
        "comparison_kind": "strict_full_scf_readiness_cpu_companion",
        "input": str(input_path),
        "forced_probe_keywords": forced_probe_keywords,
        "forced_probe_env": ";".join(FULL_SCF_PROBE_FORCED_ENV),
        "energy_abs_tol": energy_abs_tol,
        "energy_rel_tol": energy_rel_tol,
        "energy_per_atom_tol": energy_per_atom_tol,
        "status": "PASS" if not reasons else "FAIL",
        "energy_status": summary_row.get("energy_status", ""),
        "accuracy_basis": summary_row.get("accuracy_basis", ""),
        "cpu_heat_kcal_mol": summary_row.get("cpu_heat_kcal_mol", ""),
        "gpu_heat_kcal_mol": summary_row.get("gpu_heat_kcal_mol", ""),
        "abs_heat_diff_kcal_mol": summary_row.get("abs_heat_diff_kcal_mol", ""),
        "rel_heat_diff": summary_row.get("rel_heat_diff", ""),
        "abs_heat_diff_per_atom_kcal_mol": summary_row.get("abs_heat_diff_per_atom_kcal_mol", ""),
        "readiness_binding": {
            field: gpu_row.get(field, "")
            for field in FULL_SCF_READINESS_CPU_COMPARE_BINDING_FIELDS
        },
        "summary": summary_row,
        "cpu_row": cpu_row,
        "gpu_row": gpu_row,
        "cpu_only_reasons": cpu_only_reasons,
        "reasons": reasons,
    }
    compare_path = out_dir / "full_scf_gpu_readiness_cpu_compare.json"
    compare_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    status_line = (
        "Full SCF readiness CPU companion passed: "
        if not reasons
        else "Full SCF readiness CPU companion failed: "
    )
    print(
        status_line
        + f"energy_status={payload['energy_status']} "
        + f"basis={payload['accuracy_basis']} "
        + f"cpu_heat={fmt(payload['cpu_heat_kcal_mol'])} "
        + f"gpu_heat={fmt(payload['gpu_heat_kcal_mol'])} "
        + f"abs_diff={fmt(payload['abs_heat_diff_kcal_mol'])}",
        flush=True,
    )

    lines = [
        "# MOPAC Complete GPU SCF Readiness CPU Companion",
        "",
        "Strict readiness CPU companion comparison " + ("passed." if not reasons else "failed."),
        "",
        f"Input: {input_path}",
        f"Forced keywords: {forced_probe_keywords}",
        f"CPU log: {cpu_row.get('log_path', '')}",
        f"GPU log: {gpu_row.get('log_path', '')}",
        "",
        "Accuracy:",
        f"- energy_status: {payload['energy_status']}",
        f"- accuracy_basis: {payload['accuracy_basis']}",
        f"- cpu_heat_kcal_mol: {fmt(payload['cpu_heat_kcal_mol'])}",
        f"- gpu_heat_kcal_mol: {fmt(payload['gpu_heat_kcal_mol'])}",
        f"- abs_heat_diff_kcal_mol: {fmt(payload['abs_heat_diff_kcal_mol'])}",
        f"- rel_heat_diff: {fmt(payload['rel_heat_diff'])}",
        f"- abs_heat_diff_per_atom_kcal_mol: {fmt(payload['abs_heat_diff_per_atom_kcal_mol'])}",
        f"- energy_abs_tol: {energy_abs_tol:g}",
        f"- energy_rel_tol: {energy_rel_tol:g}",
        f"- energy_per_atom_tol: {energy_per_atom_tol:g}",
    ]
    if reasons:
        lines.extend(["", "Reasons:"])
        lines.extend(f"- {reason}" for reason in reasons)
    compare_readme = out_dir / "full_scf_gpu_readiness_cpu_compare.md"
    compare_readme.write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme_path = out_dir / "README.md"
    if readme_path.exists():
        existing = readme_path.read_text(encoding="utf-8", errors="ignore").rstrip()
        readme_path.write_text(existing + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    else:
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = [path for path in out_dir.rglob("*") if path.is_file()]
    manifest_path = out_dir / "MANIFEST.txt"
    create_manifest(
        manifest_path,
        files,
        title="MOPAC complete GPU SCF readiness and CPU companion bundle",
        usage_lines=[
            "Use full_scf_gpu_readiness.json for strict resident GPU execution evidence.",
            "Use full_scf_gpu_readiness_cpu_compare.json for same-input CPU/GPU energy comparison.",
            "Use logs/ for raw MOPAC output.",
        ],
        proof_identity=proof_identity,
    )
    if bundle_path is not None:
        create_bundle_zip(bundle_path, out_dir)

    if reasons:
        raise SystemExit(2)
    return payload


def diagnose_full_scf_gpu_readiness(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    returncode = parse_int_value(row.get("returncode"))
    if row.get("timed_out"):
        reasons.append(f"readiness probe timed out after {float(row.get('wall_s') or 0.0):.1f} s")
    if returncode != 0:
        reasons.append(f"MOPAC returned non-zero exit code {row.get('returncode')}")
    gpu_error_marker_count = parse_int_value(row.get("gpu_error_marker_count"))
    gpu_error_markers = str(row.get("gpu_error_markers") or "")
    if gpu_error_marker_count is None:
        reasons.append("gpu_error_marker_count was not reported")
    elif gpu_error_marker_count != 0:
        suffix = f": {gpu_error_markers}" if gpu_error_markers else ""
        reasons.append(f"raw output contained {gpu_error_marker_count} GPU error marker(s){suffix}")
    elif gpu_error_markers:
        reasons.append(f"raw output contained GPU error marker(s): {gpu_error_markers}")
    if not row.get("normal_end"):
        reasons.append("MOPAC did not finish normally with a parsed heat of formation")
    section_rows = row.get("mozyme_section_times")
    if not isinstance(section_rows, list) or not section_rows:
        reasons.append(
            "MOZYME section profile markers were not present; cannot prove that CPU mutating sections were absent"
        )
    else:
        final_reorth_sections = mozyme_final_reorth_section_names(section_rows)
        if final_reorth_sections:
            reasons.append(
                "CPU final reorthogonalization sections ran in strict proof: "
                + ";".join(final_reorth_sections)
            )
    if int(row.get("full_scf_gpu_executed") or 0) <= 0:
        reasons.append("the opt-in MOZYME GPU SCF boundary did not emit markers")
    status = str(row.get("full_scf_gpu_status") or "not_requested")
    if status != "complete":
        reason = row.get("full_scf_gpu_reason") or "no_reason"
        reasons.append(f"complete GPU SCF is not available: full_scf_gpu_status={status} reason={reason}")
    if parse_int_value(row.get("full_scf_gpu_requested")) != 1:
        reasons.append("full_scf_gpu_requested is not 1")
    if parse_int_value(row.get("full_scf_gpu_executed")) != 1:
        reasons.append("full_scf_gpu_executed is not 1")
    if parse_int_value(row.get("mozyme_scf_experimental_executed")) != 1:
        reasons.append("mozyme_scf_experimental_executed is not 1")
    if int(row.get("full_scf_gpu_ready") or 0) != 1:
        reasons.append("full_scf_gpu_ready is not 1")
    forced_probe_keywords = str(row.get("full_scf_probe_forced_keywords") or "")
    if "RE-LOCAL=1" in forced_probe_keywords:
        if (parse_int_value(row.get("mozyme_relocal_gpu_occupied_success_calls")) or 0) <= 0:
            reasons.append("forced RE-LOCAL=1 probe did not report occupied GPU relocalization success")
        if (parse_int_value(row.get("mozyme_relocal_gpu_virtual_success_calls")) or 0) <= 0:
            reasons.append("forced RE-LOCAL=1 probe did not report virtual GPU relocalization success")
    forced_probe_env = str(row.get("full_scf_probe_forced_env") or "")
    if "MOPAC_MOZYME_SCF_FORCE_FINAL_REORTH=1" in forced_probe_env:
        if (parse_int_value(row.get("mozyme_reorth_gpu_resident_success_calls")) or 0) <= 0:
            reasons.append("forced REORTH probe did not report a resident=1 final reorth success marker")
    if int(row.get("full_scf_gpu_fallback") or 0) > 0:
        reasons.append("the experimental resident-SCF boundary reported fallback or strict abort")
    probe_decision = row.get("full_scf_probe_decision")
    if probe_decision != "complete":
        reasons.append(f"strict readiness probe decision was not complete: {probe_decision}")
    scf_fallback_calls = parse_int_value(row.get("full_scf_gpu_scf_fallback_calls"))
    if scf_fallback_calls is None:
        reasons.append("full_scf_gpu_scf_fallback_calls was not reported")
    elif scf_fallback_calls != 0:
        reasons.append(
            f"resident-SCF emitted {scf_fallback_calls} fallback/strict-abort status marker(s)"
        )
    scf_success_calls = parse_int_value(row.get("full_scf_gpu_scf_success_calls"))
    if scf_success_calls is None:
        reasons.append("full_scf_gpu_scf_success_calls was not reported")
    elif scf_success_calls <= 0:
        reasons.append("resident-SCF emitted no success status marker")
    resident_step_calls = parse_int_value(row.get("full_scf_gpu_resident_step_calls"))
    if resident_step_calls is None:
        reasons.append("full_scf_gpu_resident_step_calls was not reported")
    elif resident_step_calls != 0:
        reasons.append(f"resident-SCF emitted {resident_step_calls} resident_step status marker(s)")
    backend_code = parse_int_value(row.get("full_scf_gpu_code"))
    if backend_code != 0:
        reasons.append(f"resident backend code is not 0 (full_scf_gpu_code={row.get('full_scf_gpu_code')})")
    validate_strict_host_route_contract(row, reasons)
    validate_full_scf_host_commit_contract(row, reasons)
    backend_ready = parse_int_value(row.get("full_scf_gpu_backend_ready"))
    if backend_ready != 1:
        reasons.append("backend ready marker is not 1")
    resident = parse_int_value(row.get("full_scf_gpu_resident"))
    if resident != 1:
        reasons.append("resident execution marker is not 1")
    device_id = parse_int_value(row.get("full_scf_gpu_device_id"))
    if device_id is None or device_id < 0:
        reasons.append(f"resident CUDA device_id was not reported as a nonnegative device ({row.get('full_scf_gpu_device_id')})")
    wall_ms = parse_float(row.get("full_scf_gpu_wall_ms"))
    if wall_ms is None:
        reasons.append("full_scf_gpu_wall_ms was not reported")
    elif wall_ms < 0.0:
        reasons.append(f"full_scf_gpu_wall_ms is negative ({wall_ms})")
    if parse_float(row.get("full_scf_gpu_energy_total")) is None:
        reasons.append("full_scf_gpu_energy_total was not parseable")
    if parse_float(row.get("full_scf_gpu_diagg_sumt")) is None:
        reasons.append("full_scf_gpu_diagg_sumt was not parseable")
    if parse_float(row.get("full_scf_gpu_diagg_sumb")) is None:
        reasons.append("full_scf_gpu_diagg_sumb was not parseable")
    validate_direct_cosmo_contract(row, reasons)
    validate_full_scf_cosmo_fields(
        row,
        reasons,
        require_enabled=row_uses_eps(row) or row_requires_direct_cosmo_proof(row),
    )
    stage_completed = parse_int_value(row.get("full_scf_gpu_stage_completed"))
    if stage_completed is None:
        reasons.append("resident-SCF stage_completed was not reported")
    elif (stage_completed & MOZYME_SCF_STAGE_FULL) != MOZYME_SCF_STAGE_FULL:
        names = mozyme_scf_stage_names(MOZYME_SCF_STAGE_FULL & ~stage_completed)
        reasons.append(f"resident-SCF stage_completed lacks required stages: {names}")
    elif stage_completed != MOZYME_SCF_STAGE_FULL:
        reasons.append(
            "resident-SCF stage_completed has extra bits: "
            f"{stage_completed & ~MOZYME_SCF_STAGE_FULL}"
        )
    stage_missing = parse_int_value(row.get("full_scf_gpu_stage_missing"))
    if stage_missing is None:
        reasons.append("resident-SCF stage_missing was not reported")
    elif stage_missing != 0:
        names = row.get("full_scf_gpu_stage_missing_names") or mozyme_scf_stage_names(stage_missing)
        reasons.append(f"resident-SCF stage_missing={stage_missing} ({names})")
    stage_required = parse_int_value(row.get("full_scf_gpu_stage_required"))
    if stage_required is None:
        reasons.append("resident-SCF stage_required was not reported")
    elif stage_required != MOZYME_SCF_STAGE_FULL:
        names = mozyme_scf_stage_names(stage_required)
        reasons.append(
            f"resident-SCF stage_required={stage_required} ({names}); expected {MOZYME_SCF_STAGE_FULL}"
        )
    resident_decision = parse_int_value(row.get("full_scf_gpu_resident_decision"))
    if resident_decision is None:
        reasons.append("resident-SCF resident_decision was not reported")
    elif resident_decision != MOZYME_SCF_RESIDENT_DECISION_COMPLETE:
        reasons.append(
            "resident-SCF did not finish with CompleteAndPublish "
            f"(resident_decision={resident_decision})"
        )
    isitsc_okscf = parse_int_value(row.get("full_scf_gpu_isitsc_okscf"))
    if isitsc_okscf is None:
        reasons.append("resident-SCF ISITSC convergence marker was not reported")
    elif isitsc_okscf != 1:
        reasons.append(f"resident-SCF ISITSC did not report convergence (isitsc_okscf={isitsc_okscf})")
    pls_restart_required = parse_int_value(row.get("full_scf_gpu_pls_restart_required")) or 0
    if pls_restart_required != 0:
        reasons.append("resident-SCF detected a PLS restart requirement that was not completed on GPU")
    final_iterations = parse_int_value(row.get("full_scf_gpu_final_iterations"))
    if final_iterations is None:
        reasons.append("resident-SCF final iteration count was not reported")
    elif final_iterations < 1:
        reasons.append(f"resident-SCF final iteration count is below 1 (iterations={final_iterations})")
    min_stage_calls = final_iterations or 1
    for stage_name in MOZYME_SCF_STAGE_NAMES:
        expected_calls = required_resident_stage_calls(stage_name, min_stage_calls)
        calls = parse_int_value(row.get(f"full_scf_gpu_stage_{stage_name}_calls"))
        ms = parse_float(row.get(f"full_scf_gpu_stage_{stage_name}_ms"))
        if calls is None:
            reasons.append(f"resident-SCF stage {stage_name} did not report a call counter")
        elif calls < expected_calls:
            reasons.append(
                f"resident-SCF stage {stage_name} calls={calls}; expected at least {expected_calls}"
            )
        if ms is None:
            reasons.append(f"resident-SCF stage {stage_name} did not report runtime ms")
        elif ms < 0.0:
            reasons.append(f"resident-SCF stage {stage_name} runtime is negative ({ms})")
    cnvgz_active_calls = parse_int_value(row.get("full_scf_gpu_cnvgz_active_calls"))
    cnvgz_noop_calls = parse_int_value(row.get("full_scf_gpu_cnvgz_noop_calls"))
    append_cnvgz_active_noop_reasons(reasons, cnvgz_active_calls, cnvgz_noop_calls)
    if parse_int_value(row.get("full_scf_gpu_final_density_resident")) != 1:
        reasons.append("final MOZYME density was not reported as the resident GPU density")
    makvec_success_calls = parse_int_value(row.get("mozyme_makvec_gpu_success_calls")) or 0
    makvec_existing_calls = parse_int_value(row.get("mozyme_makvec_gpu_existing_lmo_calls")) or 0
    if makvec_success_calls <= 0:
        reasons.append("MOZYME makvec initial LMO construction did not report GPU success")
    if makvec_existing_calls > 0:
        reasons.append("OLD_SCF existing-LMO marker is not accepted as complete GPU makvec proof")
    makvec_ms = parse_float(row.get("mozyme_makvec_gpu_last_ms"))
    if makvec_success_calls > 0 and makvec_ms is None:
        reasons.append("MOZYME makvec GPU did not report runtime")
    elif makvec_ms is not None and makvec_ms < 0.0:
        reasons.append(f"MOZYME makvec GPU runtime is negative ({makvec_ms})")
    setupk_success_calls = parse_int_value(row.get("mozyme_setupk_gpu_success_calls"))
    if setupk_success_calls is None:
        reasons.append("mozyme_setupk_gpu_success_calls was not reported")
    elif setupk_success_calls <= 0:
        reasons.append("MOZYME setupk GPU did not report a success marker")
    setupk_fallback_calls = parse_int_value(row.get("mozyme_setupk_gpu_fallback_calls"))
    if setupk_fallback_calls is None:
        reasons.append("mozyme_setupk_gpu_fallback_calls was not reported")
    elif setupk_fallback_calls != 0:
        reasons.append(f"mozyme_setupk_gpu_fallback_calls is nonzero ({setupk_fallback_calls})")
    setupk_initial_success_calls = parse_int_value(row.get("mozyme_setupk_gpu_initial_setup_success_calls"))
    if setupk_initial_success_calls is None:
        reasons.append("mozyme_setupk_gpu_initial_setup_success_calls was not reported")
    elif setupk_initial_success_calls <= 0:
        reasons.append("MOZYME setupk GPU did not report an initial_setup=1 success marker")
    setupk_initial_fallback_calls = parse_int_value(row.get("mozyme_setupk_gpu_initial_setup_fallback_calls"))
    if setupk_initial_fallback_calls is None:
        reasons.append("mozyme_setupk_gpu_initial_setup_fallback_calls was not reported")
    elif setupk_initial_fallback_calls != 0:
        reasons.append("MOZYME setupk GPU reported an initial_setup=1 fallback marker")
    setupk_initial_all_paths_calls = parse_int_value(
        row.get("mozyme_setupk_gpu_initial_setup_all_paths_calls")
    )
    if setupk_initial_all_paths_calls is None:
        reasons.append("mozyme_setupk_gpu_initial_setup_all_paths_calls was not reported")
    elif setupk_initial_all_paths_calls <= 0:
        reasons.append("MOZYME setupk all-initial-setup path marker was not reported")
    setupk_initial_ms = parse_float(row.get("mozyme_setupk_gpu_initial_setup_last_ms"))
    if setupk_initial_ms is None:
        reasons.append("MOZYME setupk initial_setup=1 GPU did not report runtime")
    elif setupk_initial_ms < 0.0:
        reasons.append(f"MOZYME setupk initial_setup=1 GPU runtime is negative ({setupk_initial_ms})")
    cpu_mutating_sections = str(row.get("full_scf_gpu_cpu_mutating_sections") or "")
    if cpu_mutating_sections:
        reasons.append(
            "CPU MOZYME state-mutating setup or bookend sections ran outside the resident GPU boundary: "
            + cpu_mutating_sections.replace(";", ", ")
        )
    cpu_mutating_call_count = parse_int_value(row.get("full_scf_gpu_cpu_mutating_call_count"))
    if cpu_mutating_call_count is None:
        reasons.append("full_scf_gpu_cpu_mutating_call_count was not reported")
    elif cpu_mutating_call_count != 0:
        reasons.append(f"full_scf_gpu_cpu_mutating_call_count is {cpu_mutating_call_count}")
    cpu_mutating_ms = parse_float(row.get("full_scf_gpu_cpu_mutating_ms"))
    if cpu_mutating_ms is None:
        reasons.append("full_scf_gpu_cpu_mutating_ms was not reported")
    elif cpu_mutating_ms != 0.0:
        reasons.append(f"full_scf_gpu_cpu_mutating_ms is {cpu_mutating_ms}")
    if row.get("gpu_has_device") != "T":
        reasons.append("GPU debug state did not confirm hasGPU=T")
    if row.get("gpu_lgpu_final") != "T":
        reasons.append("final MOPAC GPU switch lgpu is not T")
    if row.get("mozyme_plan_resident_fock_gpu") != "T":
        reasons.append("resident sparse Fock GPU was not enabled in the MOZYME GPU plan")
    if row.get("mozyme_plan_fock_gpu") == "T" and row.get("mozyme_plan_f2_gpu") != "T":
        reasons.append(
            "legacy MOZYME fock_gpu was enabled without f2_gpu; the plan must not count legacy Fock as GPU production"
        )
    reasons.extend(proof_identity_violation_reasons(row))
    sparse_calls = parse_int_value(row.get("mozyme_sparse_fock_run_calls")) or 0
    sparse_work = sum(
        parse_int_value(row.get(key)) or 0
        for key in (
            "mozyme_sparse_fock_run_one_tasks",
            "mozyme_sparse_fock_run_pair_tasks",
            "mozyme_sparse_fock_run_4x1_tasks",
            "mozyme_sparse_fock_run_point_tasks",
        )
    )
    sparse_ms = parse_float(row.get("mozyme_sparse_fock_run_ms"))
    if sparse_calls <= 0:
        reasons.append("resident sparse Fock GPU reported zero run calls")
    if final_iterations is not None and sparse_calls < final_iterations:
        reasons.append(
            f"resident sparse Fock GPU ran {sparse_calls} time(s), below resident iteration count {final_iterations}"
        )
    if sparse_work <= 0:
        reasons.append("resident sparse Fock GPU reported zero work tasks")
    sparse_zero_work_calls = parse_int_value(row.get("mozyme_sparse_fock_run_zero_work_calls")) or 0
    if sparse_zero_work_calls != 0:
        reasons.append(f"resident sparse Fock GPU reported {sparse_zero_work_calls} zero-work run call(s)")
    validate_resident_point_charge_coverage(row, sparse_calls, reasons)
    if sparse_ms is None:
        reasons.append("resident sparse Fock GPU did not report runtime")
    elif sparse_ms < 0.0:
        reasons.append(f"resident sparse Fock GPU runtime is negative ({sparse_ms})")
    validate_resident_plan_coverage(row, reasons)
    validate_resident_real_pair_coverage(row, reasons)
    for key in RESIDENT_FOCK_FALLBACK_KEYS:
        value = parse_int_value(row.get(key)) or 0
        if value != 0:
            reasons.append(f"{key} is nonzero ({value}); resident Fock is not all-GPU")
    for key in FULL_SCF_GPU_FALLBACK_KEYS:
        value = parse_int_value(row.get(key)) or 0
        if value != 0:
            reasons.append(f"{key} is nonzero ({value})")
    return reasons


def enforce_full_scf_gpu_rows(rows: list[dict[str, Any]], out_dir: Path) -> None:
    failures: list[str] = []
    for row in rows:
        if row.get("mode") != "GPU":
            continue
        row_reasons = diagnose_full_scf_gpu_readiness(row)
        if not row_reasons:
            continue
        header = (
            f"{row.get('molecule', '')} rep={row.get('repeat', '')} "
            f"status={row.get('full_scf_gpu_status', '')} "
            f"ready={row.get('full_scf_gpu_ready', '')} "
            f"device_id={row.get('full_scf_gpu_device_id', '')} "
            f"gpu_error_marker_count={row.get('gpu_error_marker_count', '')} "
            f"log={row.get('log_path', '')}"
        )
        failures.append(header)
        failures.extend(f"  - {reason}" for reason in row_reasons)

    if not failures:
        return

    failure_path = out_dir / "full_scf_gpu_molecule_failure.txt"
    failure_path.write_text(
        "\n".join(
            [
                "Full molecule benchmark failed the complete GPU SCF contract.",
                "",
                *failures,
                "",
            ]
        ),
        encoding="utf-8",
    )
    raise SystemExit(f"Complete GPU SCF contract failed for molecule benchmark rows. See {failure_path}")


def gpu_work_units(row: dict[str, Any]) -> int:
    keys = [
        "density_gpu_syrk_calls",
        "density_gpu_gemm_calls",
        "mozyme_sparse_fock_run_calls",
        "mozyme_makvec_gpu_success_calls",
        "mozyme_relocal_gpu_success_calls",
        "mozyme_reorth_gpu_success_calls",
        "mozyme_fock1_batch_gpu_success_calls",
        "mozyme_fock2_4x1_batch_gpu_success_calls",
        "density_batch_gpu_success_calls",
        "mozyme_cnvgz_gpu_success_calls",
        "mozyme_helecz_gpu_success_calls",
        "mozyme_eimp_gpu_success_calls",
        "mozyme_diagg1_construct_gpu_success_calls",
        "mozyme_diagg1_aocc_gpu_success_calls",
        "mozyme_diagg1_avir_gpu_success_calls",
        "mozyme_diagg2_rotate_gpu_success_calls",
        "mozyme_diagg2_rotprep_gpu_success_calls",
        "mozyme_fock1_gpu_success_seen",
        "mozyme_fock2_gpu_success_seen",
    ]
    total = 0
    for key in keys:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        total += int(value)
    return total


def diagnose_gpu_preflight(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row["timed_out"]:
        reasons.append(f"preflight timed out after {row['wall_s']:.1f} s")
    if row["returncode"] != 0:
        reasons.append(f"MOPAC returned non-zero exit code {row['returncode']}")
        if int(row.get("mozyme_fock1_batch_gpu_success_calls") or 0) > 0:
            reasons.append("batched MOZYME one-center Fock GPU work executed before the process stopped")
            planned_tasks = int(row.get("mozyme_fock_plan_one_center") or 0)
            success_tasks = int(row.get("mozyme_fock1_batch_gpu_success_tasks") or 0)
            if planned_tasks and success_tasks < planned_tasks:
                reasons.append(
                    f"only {success_tasks}/{planned_tasks} planned one-center Fock GPU tasks emitted success markers"
                )
        if int(row.get("mozyme_fock2_4x1_batch_gpu_success_calls") or 0) > 0:
            reasons.append("batched MOZYME 4x1 two-center Fock GPU work executed before the process stopped")
    gpu_error_markers = str(row.get("gpu_error_markers") or "")
    if gpu_error_markers:
        reasons.append(f"raw output contained GPU error marker(s): {gpu_error_markers}")
    fallback_keys = tuple(
        dict.fromkeys((*FULL_SCF_GPU_FALLBACK_KEYS, *RESIDENT_FOCK_FALLBACK_KEYS))
    )
    for key in fallback_keys:
        value = parse_int_value(row.get(key)) or 0
        if value:
            reasons.append(f"{key} is nonzero ({value})")
    if not row["normal_end"]:
        reasons.append("MOPAC did not finish normally with a parsed heat of formation")
    if row.get("gpu_has_device") != "T":
        reasons.append("GPU debug state did not confirm hasGPU=T")
    if row.get("gpu_lgpu_final") != "T":
        reasons.append("final MOPAC GPU switch lgpu is not T")
    if row.get("mozyme_gpu_profile") not in ("", "T"):
        reasons.append("MOZYME_GPU is not active in the profiled run")
    if row.get("mozyme_gpu_requested") == "T" and row.get("mozyme_plan_enabled") == "F":
        reason_text = row.get("mozyme_plan_disable_reason_text") or "unknown"
        reasons.append(f"MOZYME GPU was requested but the production planner disabled it: {reason_text}")
    if (
        row.get("mozyme_plan_resident_fock_gpu") == "T"
        and int(row.get("mozyme_fock_candidate_gpu_tasks") or 0) > 0
        and int(row.get("mozyme_sparse_fock_run_calls") or 0) == 0
    ):
        reasons.append("resident MOZYME sparse Fock GPU path was planned but no run marker was emitted")
    validate_resident_plan_coverage(row, reasons)
    if (
        row.get("mozyme_plan_resident_fock_gpu") == "T"
        and int(row.get("mozyme_fock_plan_point_charge_pairs") or 0) > 0
        and int(row.get("mozyme_sparse_fock_run_point_tasks") or 0) == 0
    ):
        reasons.append("resident MOZYME point-charge/dipole Fock work was planned but no point tasks ran")
    validate_resident_point_charge_coverage(
        row, parse_int_value(row.get("mozyme_sparse_fock_run_calls")) or 0, reasons
    )
    if (
        row.get("mozyme_plan_fock1_batch_gpu") == "T"
        and int(row.get("mozyme_fock_plan_one_center") or 0) > 0
        and int(row.get("mozyme_fock1_batch_gpu_success_calls") or 0) == 0
    ):
        reasons.append("batched MOZYME one-center Fock GPU path was planned but no success marker was emitted")
    if (
        row.get("mozyme_plan_fock2_4x1_batch_gpu") == "T"
        and int(row.get("mozyme_fock_plan_pairs_4x1") or 0) > 0
        and int(row.get("mozyme_fock2_4x1_batch_gpu_success_calls") or 0) == 0
    ):
        reasons.append("batched MOZYME 4x1 two-center Fock GPU path was planned but no success marker was emitted")
    required_stage_success = [
        ("MOZYME density_batch GPU", "density_batch_gpu_success_calls", "density_batch_gpu_fallback_calls"),
        ("MOZYME eimp GPU", "mozyme_eimp_gpu_success_calls", "mozyme_eimp_gpu_fallback_calls"),
        (
            "MOZYME diagg1_construct GPU",
            "mozyme_diagg1_construct_gpu_success_calls",
            "mozyme_diagg1_construct_gpu_fallback_calls",
        ),
        ("MOZYME diagg1_aocc GPU", "mozyme_diagg1_aocc_gpu_success_calls", "mozyme_diagg1_aocc_gpu_fallback_calls"),
        ("MOZYME diagg1_avir GPU", "mozyme_diagg1_avir_gpu_success_calls", "mozyme_diagg1_avir_gpu_fallback_calls"),
        ("MOZYME diagg2_rotprep GPU", "mozyme_diagg2_rotprep_gpu_success_calls", "mozyme_diagg2_rotprep_gpu_fallback_calls"),
        ("MOZYME diagg2_rotate GPU", "mozyme_diagg2_rotate_gpu_success_calls", "mozyme_diagg2_rotate_gpu_fallback_calls"),
        ("MOZYME cnvgz GPU", "mozyme_cnvgz_gpu_success_calls", "mozyme_cnvgz_gpu_fallback_calls"),
        ("MOZYME helecz GPU", "mozyme_helecz_gpu_success_calls", "mozyme_helecz_gpu_fallback_calls"),
        ("MOZYME isitsc GPU", "mozyme_isitsc_gpu_success_calls", "mozyme_isitsc_gpu_fallback_calls"),
    ]
    if row.get("normal_end"):
        for label, success_key, fallback_key in required_stage_success:
            if int(row.get(success_key) or 0) == 0:
                if int(row.get(fallback_key) or 0) > 0:
                    reasons.append(f"{label} reported fallback but no success marker")
                else:
                    reasons.append(f"{label} is enabled for production preflight but emitted no success marker")
    if gpu_work_units(row) <= 0:
        reasons.append("no profiled MOZYME GPU kernel executed")
        if row.get("returncode") < 0 and row.get("mozyme_plan_max_block") in ("", None):
            reasons.append("process died before the MOZYME GPU plan was emitted; rebuild with the latest two-phase planner")
        if row.get("mozyme_plan_disabled_no_work") == "T":
            reasons.append("MOZYME GPU was disabled by the pre-SCF plan because no production GPU work is eligible")
            if row.get("returncode") == 2:
                reasons.append("preflight stopped intentionally before running a CPU-only molecule benchmark")
        if row.get("mozyme_plan_disable_reason_text") == "no_production_work":
            reasons.append("structured MOZYME plan reports no production GPU work")
        if row.get("mozyme_plan_disable_reason_text") == "no_device":
            reasons.append("structured MOZYME plan reports no usable GPU device for this run")
        if row.get("mozyme_plan_disable_reason_text") == "device_policy":
            reasons.append("structured MOZYME plan reports the device/policy gate disabled MOZYME GPU")
        candidate_tasks = row.get("mozyme_fock_candidate_gpu_tasks")
        production_tasks = row.get("mozyme_fock_production_gpu_tasks")
        if candidate_tasks not in ("", None) and int(candidate_tasks or 0) > 0 and int(production_tasks or 0) == 0:
            reasons.append(
                f"MOZYME Fock has {candidate_tasks} candidate sparse tasks, but no production GPU Fock kernel is enabled"
            )
        if row.get("mozyme_fock_gpu_profile") == "F" or row.get("mozyme_plan_fock_gpu") == "F":
            reasons.append("legacy MOZYME Fock GPU is disabled by production safety policy")
        if row.get("mozyme_fock1_batch_gpu_profile") == "F" or row.get("mozyme_plan_fock1_batch_gpu") == "F":
            reasons.append("batched MOZYME one-center Fock GPU path is disabled")
        if row.get("mozyme_fock2_4x1_batch_gpu_profile") == "F" or row.get("mozyme_plan_fock2_4x1_batch_gpu") == "F":
            reasons.append("batched MOZYME 4x1 two-center Fock GPU path is disabled")
        if int(row.get("mozyme_fock1_batch_gpu_fallback_calls") or 0) > 0:
            reasons.append("batched MOZYME one-center Fock GPU path reported fallback to CPU")
        if int(row.get("mozyme_fock2_4x1_batch_gpu_fallback_calls") or 0) > 0:
            reasons.append("batched MOZYME 4x1 two-center Fock GPU path reported fallback to CPU")
        if row.get("mozyme_check_gpu_profile") == "F":
            reasons.append("experimental MOZYME LMO check/retry path is disabled by production safety policy")
        if row.get("mozyme_plan_density_pairs_meeting_minblk") not in ("", None):
            if int(row.get("mozyme_plan_density_pairs_meeting_minblk") or 0) == 0:
                reasons.append("MOZYME density plan has no atom pairs meeting the current GPU block threshold")
        minblk = row.get("mozyme_minblk")
        max_diag = row.get("density_max_diag_block")
        max_j = row.get("density_max_offdiag_j")
        max_k = row.get("density_max_offdiag_k")
        if minblk not in ("", None) and max_diag not in ("", None):
            try:
                max_block = max(int(max_diag), int(max_j or 0), int(max_k or 0))
                if max_block < int(minblk):
                    reasons.append(
                        f"largest MOZYME density block is {max_block}, below MOZYME_MINBLK={minblk}"
                    )
            except (TypeError, ValueError):
                pass
        if row.get("mozyme_fock1_gpu_fallback_seen") or row.get("mozyme_fock2_gpu_fallback_seen"):
            reasons.append("MOZYME Fock GPU wrapper reported fallback to CPU")
        if row["returncode"] < 0 and (row.get("mozyme_fock1_gpu_attempt_seen") or row.get("mozyme_fock2_gpu_attempt_seen")):
            reasons.append(
                "process died after attempting a MOZYME Fock GPU wrapper; this path is unsafe and must stay out of production"
            )
    return reasons


def run_gpu_preflight(
    mopac: Path,
    input_path: Path,
    out_dir: Path,
    timeout: float,
    bundle_path: Path | None,
    min_speedup: float,
    mozyme_section_profile: bool,
) -> dict[str, Any]:
    print("")
    print(f"GPU preflight: {input_path.name}", flush=True)
    cpu_row: dict[str, Any] | None = None
    preflight_speedup: float | None = None
    if min_speedup > 0.0:
        cpu_row = run_mopac(
            mopac,
            input_path,
            MODES[0],
            0,
            out_dir,
            timeout,
            keep_run_dirs=False,
            verbose_gpu=False,
            gpu_profile=False,
            mozyme_section_profile=mozyme_section_profile,
        )
    row = run_mopac(
        mopac,
        input_path,
        MODES[1],
        0,
        out_dir,
        timeout,
        keep_run_dirs=False,
        verbose_gpu=True,
        gpu_profile=True,
        mozyme_section_profile=mozyme_section_profile,
        gpu_preflight_stop=True,
    )
    reasons = diagnose_gpu_preflight(row)
    if cpu_row is not None:
        row["preflight_cpu_wall_s"] = cpu_row.get("wall_s", "")
        if cpu_row["timed_out"]:
            reasons.append(f"CPU preflight timed out after {cpu_row['wall_s']:.1f} s")
        elif cpu_row["returncode"] != 0:
            reasons.append(f"CPU preflight returned non-zero exit code {cpu_row['returncode']}")
        elif row["wall_s"] > 0.0:
            preflight_speedup = float(cpu_row["wall_s"]) / float(row["wall_s"])
            row["preflight_speedup"] = preflight_speedup
            if preflight_speedup < min_speedup:
                reasons.append(
                    f"GPU preflight speedup {preflight_speedup:.4g} is below required {min_speedup:.4g}; "
                    "aborting before the full molecule benchmark"
                )
    payload = {
        "input": str(input_path),
        "cpu_row": cpu_row,
        "row": row,
        "gpu_work_units": gpu_work_units(row),
        "preflight_speedup": preflight_speedup,
        "min_speedup": min_speedup,
        "reasons": reasons,
    }
    (out_dir / "gpu_preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if reasons:
        lines = [
            "GPU preflight failed.",
            "",
            f"Input: {input_path}",
            f"Log: {row.get('log_path', '')}",
            "",
            "Reasons:",
        ]
        lines.extend(f"- {reason}" for reason in reasons)
        lines.extend(
            [
                "",
                "Useful fields:",
                f"- hasGPU: {row.get('gpu_has_device', '')}",
                f"- CPU preflight wall s: {row.get('preflight_cpu_wall_s', '')}",
                f"- GPU preflight wall s: {row.get('wall_s', '')}",
                f"- preflight speedup: {row.get('preflight_speedup', '')}",
                f"- final lgpu: {row.get('gpu_lgpu_final', '')}",
                f"- plan lgpu: {row.get('mozyme_plan_lgpu', '')}",
                f"- plan ready: {row.get('mozyme_plan_ready', '')}",
                f"- plan enabled: {row.get('mozyme_plan_enabled', '')}",
                f"- plan disable reason: {row.get('mozyme_plan_disable_reason_text', '')}",
                f"- MOZYME_GPU: {row.get('mozyme_gpu_profile', '')}",
                f"- MOZYME_GPU requested: {row.get('mozyme_gpu_requested', '')}",
                f"- MOZYME_GPU after plan: {row.get('mozyme_gpu_after_plan', '')}",
                f"- MOZYME_RESIDENT_FOCK_GPU: {row.get('mozyme_resident_fock_gpu_profile', '')}",
                f"- MOZYME_FOCK1_BATCH_GPU: {row.get('mozyme_fock1_batch_gpu_profile', '')}",
                f"- MOZYME_FOCK2_4X1_BATCH_GPU: {row.get('mozyme_fock2_4x1_batch_gpu_profile', '')}",
                f"- MOZYME_FOCK_GPU: {row.get('mozyme_fock_gpu_profile', '')}",
                f"- MOZYME_CHECK_GPU: {row.get('mozyme_check_gpu_profile', '')}",
                f"- plan disabled no work: {row.get('mozyme_plan_disabled_no_work', '')}",
                f"- plan direct mode: {row.get('mozyme_plan_direct_mode', '')}",
                f"- plan resident sparse Fock: {row.get('mozyme_plan_resident_fock_gpu', '')}",
                f"- MOZYME_MINBLK: {row.get('mozyme_minblk', '')}",
                f"- plan max block: {row.get('mozyme_plan_max_block', '')}",
                f"- plan density pairs meeting minblk: {row.get('mozyme_plan_density_pairs_meeting_minblk', '')}",
                f"- Fock candidate GPU tasks: {row.get('mozyme_fock_candidate_gpu_tasks', '')}",
                f"- Fock production GPU tasks: {row.get('mozyme_fock_production_gpu_tasks', '')}",
                f"- resident Fock full coverage planned: {row.get('mozyme_fock_resident_full_coverage_planned', '')}",
                f"- resident Fock executable tasks: {row.get('mozyme_fock_resident_executable_tasks', '')}",
                f"- resident-supported sparse Fock pairs: {row.get('mozyme_fock_resident_supported_pairs', '')}",
                f"- resident unsupported real pairs basis/direct/other/noop: "
                f"{row.get('mozyme_fock_resident_basis_limit_unsupported_pairs', '')}/"
                f"{row.get('mozyme_fock_resident_direct_unsupported_pairs', '')}/"
                f"{row.get('mozyme_fock_resident_other_unsupported_pairs', '')}/"
                f"{row.get('mozyme_fock_resident_noop_pairs', '')}",
                f"- resident unsupported point pairs basis/direct/other: "
                f"{row.get('mozyme_fock_resident_basis_limit_unsupported_point_pairs', '')}/"
                f"{row.get('mozyme_fock_resident_direct_unsupported_point_pairs', '')}/"
                f"{row.get('mozyme_fock_resident_other_unsupported_point_pairs', '')}",
                f"- Fock point-charge/dipole/monopole pairs: "
                f"{row.get('mozyme_fock_plan_point_charge_pairs', '')}/"
                f"{row.get('mozyme_fock_plan_point_dipole_pairs', '')}/"
                f"{row.get('mozyme_fock_plan_point_monopole_pairs', '')}",
                f"- Fock one-center/two-center tasks: {row.get('mozyme_fock_plan_one_center', '')}/{row.get('mozyme_fock_plan_two_center', '')}",
                f"- Fock 4x4/4x1/9x4/9x9 pairs: {row.get('mozyme_fock_plan_pairs_4x4', '')}/"
                f"{row.get('mozyme_fock_plan_pairs_4x1', '')}/{row.get('mozyme_fock_plan_pairs_9x4', '')}/"
                f"{row.get('mozyme_fock_plan_pairs_9x9', '')}",
                f"- resident sparse Fock setup calls/one/pair/4x1/point/dipole/monopole: "
                f"{row.get('mozyme_sparse_fock_setup_calls', '')}/"
                f"{row.get('mozyme_sparse_fock_setup_one_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_setup_pair_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_setup_4x1_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_setup_point_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_setup_point_dipole_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_setup_point_monopole_tasks', '')}",
                f"- resident sparse Fock run calls/one/pair/4x1/point/dipole/monopole/zero-work/ms: "
                f"{row.get('mozyme_sparse_fock_run_calls', '')}/"
                f"{row.get('mozyme_sparse_fock_run_one_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_run_pair_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_run_4x1_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_run_point_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_run_point_dipole_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_run_point_monopole_tasks', '')}/"
                f"{row.get('mozyme_sparse_fock_run_zero_work_calls', '')}/"
                f"{row.get('mozyme_sparse_fock_run_ms', '')}",
                f"- density GPU SYRK/GEMM: {row.get('density_gpu_syrk_calls', '')}/{row.get('density_gpu_gemm_calls', '')}",
                f"- density batch GPU success/fallback/blocks/terms/ms: "
                f"{row.get('density_batch_gpu_success_calls', '')}/"
                f"{row.get('density_batch_gpu_fallback_calls', '')}/"
                f"{row.get('density_batch_gpu_last_blocks', '')}/"
                f"{row.get('density_batch_gpu_last_terms', '')}/"
                f"{row.get('density_batch_gpu_last_ms', '')}",
                f"- EIMP GPU success/fallback/pairs/ms: "
                f"{row.get('mozyme_eimp_gpu_success_calls', '')}/"
                f"{row.get('mozyme_eimp_gpu_fallback_calls', '')}/"
                f"{row.get('mozyme_eimp_gpu_last_pairs', '')}/"
                f"{row.get('mozyme_eimp_gpu_last_ms', '')}",
                f"- DIAGG1 CONSTRUCT GPU success/fallback/nij/sumt/tiny/ms: "
                f"{row.get('mozyme_diagg1_construct_gpu_success_calls', '')}/"
                f"{row.get('mozyme_diagg1_construct_gpu_fallback_calls', '')}/"
                f"{row.get('mozyme_diagg1_construct_gpu_last_nij', '')}/"
                f"{row.get('mozyme_diagg1_construct_gpu_last_sumt', '')}/"
                f"{row.get('mozyme_diagg1_construct_gpu_last_tiny', '')}/"
                f"{row.get('mozyme_diagg1_construct_gpu_last_ms', '')}",
                f"- DIAGG1 AOCC GPU success/fallback/terms/ms: "
                f"{row.get('mozyme_diagg1_aocc_gpu_success_calls', '')}/"
                f"{row.get('mozyme_diagg1_aocc_gpu_fallback_calls', '')}/"
                f"{row.get('mozyme_diagg1_aocc_gpu_last_terms', '')}/"
                f"{row.get('mozyme_diagg1_aocc_gpu_last_ms', '')}",
                f"- DIAGG1 AVIR GPU success/fallback/terms/ms: "
                f"{row.get('mozyme_diagg1_avir_gpu_success_calls', '')}/"
                f"{row.get('mozyme_diagg1_avir_gpu_fallback_calls', '')}/"
                f"{row.get('mozyme_diagg1_avir_gpu_last_terms', '')}/"
                f"{row.get('mozyme_diagg1_avir_gpu_last_ms', '')}",
                f"- DIAGG2 ROTATE GPU success/fallback/nrej/sumb/ms: "
                f"{row.get('mozyme_diagg2_rotate_gpu_success_calls', '')}/"
                f"{row.get('mozyme_diagg2_rotate_gpu_fallback_calls', '')}/"
                f"{row.get('mozyme_diagg2_rotate_gpu_last_nrej', '')}/"
                f"{row.get('mozyme_diagg2_rotate_gpu_last_sumb', '')}/"
                f"{row.get('mozyme_diagg2_rotate_gpu_last_ms', '')}",
                f"- DIAGG2 ROTPREP GPU success/fallback/active/ms: "
                f"{row.get('mozyme_diagg2_rotprep_gpu_success_calls', '')}/"
                f"{row.get('mozyme_diagg2_rotprep_gpu_fallback_calls', '')}/"
                f"{row.get('mozyme_diagg2_rotprep_gpu_last_active', '')}/"
                f"{row.get('mozyme_diagg2_rotprep_gpu_last_ms', '')}",
                f"- ISITSC GPU success/fallback/okscf/ms: "
                f"{row.get('mozyme_isitsc_gpu_success_calls', '')}/"
                f"{row.get('mozyme_isitsc_gpu_fallback_calls', '')}/"
                f"{row.get('mozyme_isitsc_gpu_last_okscf', '')}/"
                f"{row.get('mozyme_isitsc_gpu_last_ms', '')}",
                f"- Fock1 batch GPU attempt calls/tasks/pairs: "
                f"{row.get('mozyme_fock1_batch_gpu_attempt_calls', '')}/"
                f"{row.get('mozyme_fock1_batch_gpu_attempt_tasks', '')}/"
                f"{row.get('mozyme_fock1_batch_gpu_attempt_pairs', '')}",
                f"- Fock1 batch GPU success calls/tasks/pairs: "
                f"{row.get('mozyme_fock1_batch_gpu_success_calls', '')}/"
                f"{row.get('mozyme_fock1_batch_gpu_success_tasks', '')}/"
                f"{row.get('mozyme_fock1_batch_gpu_success_pairs', '')}",
                f"- Fock1 batch GPU fallback calls/tasks/pairs: "
                f"{row.get('mozyme_fock1_batch_gpu_fallback_calls', '')}/"
                f"{row.get('mozyme_fock1_batch_gpu_fallback_tasks', '')}/"
                f"{row.get('mozyme_fock1_batch_gpu_fallback_pairs', '')}",
                f"- Fock2 4x1 batch GPU attempt calls/tasks: "
                f"{row.get('mozyme_fock2_4x1_batch_gpu_attempt_calls', '')}/"
                f"{row.get('mozyme_fock2_4x1_batch_gpu_attempt_tasks', '')}",
                f"- Fock2 4x1 batch GPU success calls/tasks: "
                f"{row.get('mozyme_fock2_4x1_batch_gpu_success_calls', '')}/"
                f"{row.get('mozyme_fock2_4x1_batch_gpu_success_tasks', '')}",
                f"- Fock2 4x1 batch GPU fallback calls/tasks: "
                f"{row.get('mozyme_fock2_4x1_batch_gpu_fallback_calls', '')}/"
                f"{row.get('mozyme_fock2_4x1_batch_gpu_fallback_tasks', '')}",
                f"- Fock1/Fock2 GPU attempt seen: {row.get('mozyme_fock1_gpu_attempt_seen', '')}/{row.get('mozyme_fock2_gpu_attempt_seen', '')}",
                f"- Fock1/Fock2 GPU success seen: {row.get('mozyme_fock1_gpu_success_seen', '')}/{row.get('mozyme_fock2_gpu_success_seen', '')}",
                f"- max density block: {row.get('density_max_diag_block', '')}",
            ]
        )
        failure_path = out_dir / "gpu_preflight_failure.txt"
        failure_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        readme_path = out_dir / "README.md"
        readme_path.write_text("\n".join(["# MOPAC Molecule GPU Preflight", "", *lines]) + "\n", encoding="utf-8")
        files = [path for path in out_dir.rglob("*") if path.is_file()]
        manifest_path = out_dir / "MANIFEST.txt"
        create_manifest(manifest_path, files)
        if bundle_path is not None:
            create_bundle_zip(bundle_path, out_dir)
            lines.append("")
            lines.append(f"Wrote preflight diagnostic bundle: {bundle_path}")
        print("\n".join(lines), flush=True)
        raise SystemExit(2)
    print(
        "GPU preflight passed: "
        f"lgpu={row.get('gpu_lgpu_final', '')} "
        f"MOZYME_GPU={row.get('mozyme_gpu_profile', '')} "
        f"ResidentFockCalls={row.get('mozyme_sparse_fock_run_calls', '')} "
        f"ResidentFockTasks={row.get('mozyme_sparse_fock_run_one_tasks', '')}/"
        f"{row.get('mozyme_sparse_fock_run_pair_tasks', '')}/"
        f"{row.get('mozyme_sparse_fock_run_4x1_tasks', '')}/"
        f"{row.get('mozyme_sparse_fock_run_point_tasks', '')} "
        f"ResidentFockPointKinds={row.get('mozyme_sparse_fock_run_point_dipole_tasks', '')}/"
        f"{row.get('mozyme_sparse_fock_run_point_monopole_tasks', '')} "
        f"ResidentFockZeroWork={row.get('mozyme_sparse_fock_run_zero_work_calls', '')} "
        f"Fock1BatchCalls={row.get('mozyme_fock1_batch_gpu_success_calls', '')} "
        f"Fock2_4x1_BatchCalls={row.get('mozyme_fock2_4x1_batch_gpu_success_calls', '')} "
        f"DensityBatchCalls={row.get('density_batch_gpu_success_calls', '')} "
        f"Diagg1ConstructCalls={row.get('mozyme_diagg1_construct_gpu_success_calls', '')} "
        f"Diagg2RotateCalls={row.get('mozyme_diagg2_rotate_gpu_success_calls', '')} "
        f"Diagg2RotprepCalls={row.get('mozyme_diagg2_rotprep_gpu_success_calls', '')} "
        f"work_units={gpu_work_units(row)} "
        f"speedup={row.get('preflight_speedup', '')}",
        flush=True,
    )
    return row


def main() -> int:
    args = parse_args()
    if args.full_scf_readiness_only and not args.require_full_scf_gpu:
        raise SystemExit("--full-scf-readiness-only requires --require-full-scf-gpu.")
    if args.full_scf_readiness_cpu_compare and not args.require_full_scf_gpu:
        raise SystemExit("--full-scf-readiness-cpu-compare requires --require-full-scf-gpu.")
    if args.require_full_scf_gpu and not args.full_scf_readiness_cpu_compare:
        raise SystemExit(
            "--require-full-scf-gpu requires --full-scf-readiness-cpu-compare "
            "for strict full-SCF proof runs."
        )
    if args.require_full_scf_gpu and not args.require_direct_cosmo_gpu:
        raise SystemExit(
            "--require-full-scf-gpu requires --require-direct-cosmo-gpu for the v58 "
            "COSMO-direct point-kind proof contract."
        )
    mopac = Path(args.mopac).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not mopac.exists():
        raise SystemExit(f"MOPAC executable not found: {mopac}")
    raw_inputs = args.inputs or DEFAULT_PROFILES[args.profile]
    if args.full_scf_readiness_only:
        if args.full_scf_probe_input:
            inputs = [Path(args.full_scf_probe_input).resolve()]
        elif args.preflight_input:
            inputs = [Path(args.preflight_input).resolve()]
        else:
            profile_inputs = [Path(item).resolve() for item in raw_inputs]
            inputs = [select_mozyme_preflight_input(profile_inputs)]
    else:
        inputs = [Path(item).resolve() for item in raw_inputs]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        raise SystemExit("Missing input files: " + ", ".join(str(path) for path in missing))

    out_dir.mkdir(parents=True, exist_ok=True)
    proof_identity = collect_proof_identity(mopac)
    bundle_path: Path | None = None
    if not args.no_bundle_zip:
        bundle_path = Path(args.bundle_zip).resolve() if args.bundle_zip else out_dir.with_name(
            f"{out_dir.name}_publication_data.zip"
        )
    if args.require_full_scf_gpu:
        if args.full_scf_probe_input:
            full_scf_probe_input = Path(args.full_scf_probe_input).resolve()
        elif args.preflight_input:
            full_scf_probe_input = Path(args.preflight_input).resolve()
        else:
            full_scf_probe_input = select_mozyme_preflight_input(inputs)
        if not full_scf_probe_input.exists():
            raise SystemExit(f"Full SCF GPU readiness input not found: {full_scf_probe_input}")
        eps = parse_input_eps(full_scf_probe_input)
        if eps is None or abs(eps - 78.4) > 1.0e-9:
            raise SystemExit(
                "--require-direct-cosmo-gpu requires a full SCF probe input with EPS=78.4."
            )
        full_scf_readiness_row = run_full_scf_gpu_readiness_probe(
            mopac,
            full_scf_probe_input,
            out_dir,
            args.full_scf_probe_timeout if args.full_scf_probe_timeout > 0.0 else args.preflight_timeout,
            bundle_path,
            args.mozyme_section_profile,
            args.require_direct_cosmo_gpu,
        )
        if args.full_scf_readiness_cpu_compare:
            run_full_scf_readiness_cpu_compare(
                mopac,
                full_scf_probe_input,
                full_scf_readiness_row,
                out_dir,
                args.full_scf_probe_timeout if args.full_scf_probe_timeout > 0.0 else args.preflight_timeout,
                bundle_path,
                args.mozyme_section_profile,
                args.energy_abs_tol,
                args.energy_rel_tol,
                args.energy_per_atom_tol,
            )
        if args.full_scf_readiness_only:
            return 0
    if not args.no_gpu_preflight:
        preflight_input = (
            Path(args.preflight_input).resolve()
            if args.preflight_input
            else select_mozyme_preflight_input(inputs)
        )
        if not preflight_input.exists():
            raise SystemExit(f"GPU preflight input not found: {preflight_input}")
        run_gpu_preflight(
            mopac,
            preflight_input,
            out_dir,
            args.preflight_timeout,
            bundle_path,
            args.preflight_min_speedup,
            args.mozyme_section_profile,
        )

    rows: list[dict[str, Any]] = []
    for input_path in inputs:
        for mode in MODES:
            for repeat in range(1, args.repeats + 1):
                row = run_mopac(
                    mopac,
                    input_path,
                    mode,
                    repeat,
                    out_dir,
                    args.timeout,
                    args.keep_run_dirs,
                    args.verbose_gpu,
                    args.gpu_profile,
                    args.mozyme_section_profile,
                    full_scf_gpu=args.require_full_scf_gpu,
                    proof_identity=proof_identity,
                )
                rows.append(row)

    if args.require_full_scf_gpu:
        enforce_full_scf_gpu_rows(rows, out_dir)

    summary = summarize(rows, args.energy_abs_tol, args.energy_rel_tol, args.energy_per_atom_tol)
    mozyme_section_rows = collect_mozyme_section_rows(rows)
    mozyme_section_summary = summarize_mozyme_sections(mozyme_section_rows)
    write_csv(
        out_dir / "molecule_runs.csv",
        rows,
        [
            "molecule",
            "input",
            "input_eps",
            "requires_direct_cosmo_gpu",
            "source_zip_sha256",
            "source_zip_sha256_env",
            "source_zip_path",
            "source_zip_verified",
            "source_manifest_sha256",
            "source_manifest_file_sha256",
            "source_features_file_sha256",
            "source_provenance_file_sha256",
            "source_provenance_manifest_sha256",
            "source_features_manifest_sha256",
            "source_metadata_present",
            "source_metadata_valid",
            "source_metadata_from_zip",
            "source_metadata_violation_reasons",
            "source_manifest_entry_count",
            "source_manifest_verified_files",
            "source_manifest_missing_file_count",
            "source_manifest_hash_mismatch_count",
            "source_critical_file_count",
            "source_critical_files_verified",
            "source_required_marker_count",
            "source_required_markers_verified",
            "source_required_marker_violation_count",
            "source_required_critical_file_count",
            "source_required_critical_missing_count",
            "source_required_critical_missing_files",
            "source_git_commit",
            "source_git_dirty",
            "source_dirty_status_sha256",
            "source_generated_at_utc",
            "source_provenance_git_commit",
            "source_provenance_git_dirty",
            "source_provenance_dirty_status_sha256",
            "source_provenance_generated_at_utc",
            "source_provenance_contract_version",
            "source_features_contract_version",
            "source_provenance_marker_contract_version",
            "source_features_marker_contract_version",
            "source_provenance_marker_contract_sha256",
            "source_features_marker_contract_sha256",
            "source_features_feature_set",
            "mopac_executable",
            "mopac_executable_sha256",
            "cmake_gpu_bool",
            "cmake_cuda_architectures",
            "cmake_cuda_archs",
            "mode",
            "repeat",
            "returncode",
            "timed_out",
            "normal_end",
            "wall_s",
            "reported_s",
            "heat_kcal_mol",
            "atoms",
            "mozyme_gpu_profile",
            "mozyme_gpu_requested",
            "mozyme_minblk",
            "mozyme_resident_fock_gpu_profile",
            "mozyme_fock1_batch_gpu_profile",
            "mozyme_fock2_4x1_batch_gpu_profile",
            "mozyme_fock_gpu_profile",
            "mozyme_check_gpu_profile",
            "mozyme_plan_lgpu",
            "mozyme_plan_enabled",
            "mozyme_plan_ready",
            "mozyme_plan_direct_mode",
            "mozyme_plan_fock_gpu",
            "mozyme_plan_f2_gpu",
            "mozyme_plan_resident_fock_gpu",
            "mozyme_plan_fock1_batch_gpu",
            "mozyme_plan_fock2_4x1_batch_gpu",
            "mozyme_plan_check_gpu",
            "mozyme_plan_minblk",
            "mozyme_plan_max_block",
            "mozyme_plan_density_pairs_meeting_minblk",
            "mozyme_gpu_after_plan",
            "mozyme_plan_disabled_no_work",
            "mozyme_plan_disable_reason",
            "mozyme_plan_disable_reason_text",
            "mozyme_fock_plan_one_center",
            "mozyme_fock_resident_supported_one_center",
            "mozyme_fock_resident_unsupported_one_center",
            "mozyme_fock_plan_two_center",
            "mozyme_fock_plan_skipped",
            "mozyme_fock_plan_point_charge_pairs",
            "mozyme_fock_plan_point_dipole_pairs",
            "mozyme_fock_plan_point_monopole_pairs",
            "mozyme_fock_plan_d_pairs",
            "mozyme_fock_plan_pairs_4x4",
            "mozyme_fock_plan_pairs_4x1",
            "mozyme_fock_plan_pairs_9x4",
            "mozyme_fock_plan_pairs_9x9",
            "mozyme_fock_resident_supported_pairs",
            "mozyme_fock_resident_unsupported_pairs",
            "mozyme_fock_resident_noop_pairs",
            "mozyme_fock_resident_basis_limit_unsupported_pairs",
            "mozyme_fock_resident_direct_unsupported_pairs",
            "mozyme_fock_resident_other_unsupported_pairs",
            "mozyme_fock_resident_supported_point_pairs",
            "mozyme_fock_resident_unsupported_point_pairs",
            "mozyme_fock_resident_basis_limit_unsupported_point_pairs",
            "mozyme_fock_resident_direct_unsupported_point_pairs",
            "mozyme_fock_resident_other_unsupported_point_pairs",
            "mozyme_fock_resident_full_coverage_planned",
            "mozyme_fock_resident_executable_tasks",
            "mozyme_fock_candidate_gpu_tasks",
            "mozyme_fock_production_gpu_tasks",
            "mozyme_fock_one_center_terms",
            "mozyme_fock_two_center_terms",
            "gpu_has_device",
            "gpu_device_count",
            "gpu_lgpu_final",
            "gpu_error_marker_count",
            "gpu_error_markers",
            "gpu_resident_scf_final",
            "mozyme_scf_experimental_executed",
            "mozyme_scf_experimental_status",
            "mozyme_scf_experimental_reason",
            "full_scf_gpu_requested",
            "full_scf_gpu_executed",
            "full_scf_gpu_ready",
            "full_scf_gpu_status",
            "full_scf_gpu_reason",
            "full_scf_gpu_contract_violations",
            "full_scf_probe_decision",
            "full_scf_gpu_fallback",
            "full_scf_gpu_scf_success_calls",
            "full_scf_gpu_scf_fallback_calls",
            "full_scf_gpu_resident_step_calls",
            "full_scf_gpu_cpu_boundary_calls",
            "mozyme_gpu_helper_fatal_marker_count",
            "mozyme_gpu_helper_fatal_markers",
            "full_scf_gpu_strict_host_route_marker_count",
            "full_scf_gpu_strict_host_route_markers",
            "full_scf_gpu_pls_restart_required_calls",
            "full_scf_gpu_code",
            "full_scf_gpu_backend_ready",
            "full_scf_gpu_resident",
            "full_scf_gpu_compact_index_route",
            "full_scf_gpu_use_nijbo",
            "full_scf_gpu_device_id",
            "full_scf_gpu_stage_completed",
            "full_scf_gpu_stage_required",
            "full_scf_gpu_stage_missing",
            "full_scf_gpu_resident_decision",
            "full_scf_gpu_stage_completed_names",
            "full_scf_gpu_stage_missing_names",
            "full_scf_gpu_stage_completed_names_raw",
            "full_scf_gpu_stage_missing_names_raw",
            "full_scf_gpu_strict_resident",
            "full_scf_gpu_no_fallback_required",
            "full_scf_gpu_full_stage_mask",
            "full_scf_gpu_resident_decision_complete",
            "full_scf_gpu_strict_resident_host_syncs",
            "full_scf_gpu_strict_resident_control_polls",
            "full_scf_gpu_resident_fock_plan_id",
            "full_scf_gpu_resident_fock_plan_full_coverage",
            "full_scf_gpu_resident_fock_plan_partial_coverage",
            "full_scf_gpu_resident_fock_plan_required_mask",
            "full_scf_gpu_resident_fock_plan_covered_mask",
            *MOZYME_SCF_STAGE_CALL_FIELDS,
            *MOZYME_SCF_STAGE_MS_FIELDS,
            "full_scf_gpu_cnvgz_active_calls",
            "full_scf_gpu_cnvgz_noop_calls",
            "full_scf_gpu_isitsc_okscf",
            "full_scf_gpu_isitsc_iscf",
            "full_scf_gpu_isitsc_iemin",
            "full_scf_gpu_isitsc_iemax",
            "full_scf_gpu_isitsc_scf1",
            "full_scf_gpu_pls_supervisor_calls",
            "full_scf_gpu_pls_restart_required",
            "full_scf_gpu_pls_history_count",
            "full_scf_gpu_pls_ovmax_delta",
            "full_scf_gpu_pls_energy_delta",
            "full_scf_gpu_pls_restart_reset_device_calls",
            "full_scf_gpu_pls_restart_done",
            "full_scf_gpu_final_iterations",
            "full_scf_gpu_final_density_resident",
            "full_scf_gpu_final_publication_done",
            "full_scf_gpu_final_publication_arrays",
            "full_scf_gpu_final_publication_bytes",
            "full_scf_gpu_final_publication_cosmo",
            "full_scf_gpu_olden_setup_only",
            "full_scf_gpu_fillij_gpu_count_calls",
            "full_scf_gpu_fillij_gpu_fill_calls",
            "full_scf_gpu_fillij_gpu_last_mpack",
            "full_scf_gpu_fillij_gpu_last_n2elec",
            "full_scf_gpu_fillij_gpu_last_ij_dim",
            "full_scf_gpu_resident_fock_gpu_count_calls",
            "full_scf_gpu_resident_fock_gpu_count_plan_id",
            "full_scf_gpu_resident_fock_gpu_count_one",
            "full_scf_gpu_resident_fock_gpu_count_pair",
            "full_scf_gpu_resident_fock_gpu_count_pair4x1",
            "full_scf_gpu_resident_fock_gpu_count_point",
            "full_scf_gpu_resident_fock_gpu_count_full_coverage",
            "full_scf_gpu_resident_fock_gpu_pack_calls",
            "full_scf_gpu_resident_fock_gpu_pack_plan_id",
            "full_scf_gpu_resident_fock_gpu_pack_one",
            "full_scf_gpu_resident_fock_gpu_pack_pair",
            "full_scf_gpu_resident_fock_gpu_pack_pair4x1",
            "full_scf_gpu_resident_fock_gpu_pack_point",
            "full_scf_gpu_resident_fock_gpu_pack_full_coverage",
            "full_scf_gpu_resident_fock_gpu_point_weight_calls",
            "full_scf_gpu_resident_fock_gpu_point_weight_point",
            "full_scf_gpu_resident_fock_gpu_point_weight_max_abs_diff",
            "full_scf_gpu_cpu_mozyme_setup_only_calls",
            "full_scf_gpu_cpu_resident_fock_plan_setup_calls",
            "full_scf_gpu_cpu_resident_fock_plan_setup_plan_id",
            "full_scf_gpu_cpu_resident_fock_plan_setup_one",
            "full_scf_gpu_cpu_resident_fock_plan_setup_pair",
            "full_scf_gpu_cpu_resident_fock_plan_setup_pair4x1",
            "full_scf_gpu_cpu_resident_fock_plan_setup_point",
            "full_scf_gpu_cpu_resident_fock_plan_setup_full_coverage",
            "full_scf_gpu_host_commit_only_calls",
            "full_scf_gpu_host_commit_phase",
            "full_scf_gpu_host_commit_arrays",
            "full_scf_gpu_host_commit_bytes",
            "full_scf_gpu_host_commit_cosmo",
            "full_scf_gpu_cpu_pinout_calls",
            "full_scf_gpu_cpu_mutating_sections",
            "full_scf_gpu_cpu_mutating_section_count",
            "full_scf_gpu_cpu_mutating_call_count",
            "full_scf_gpu_cpu_mutating_ms",
            "full_scf_gpu_wall_ms",
            "full_scf_gpu_density_max",
            "full_scf_gpu_density_rms",
            "full_scf_gpu_diagg_sumt",
            "full_scf_gpu_diagg_sumb",
            "full_scf_gpu_energy_total",
            "full_scf_gpu_cosmo_enabled",
            "full_scf_gpu_cosmo_fock_calls",
            "full_scf_gpu_cosmo_matvec_calls",
            "full_scf_gpu_cosmo_cg_iterations",
            "full_scf_gpu_cosmo_nps",
            "full_scf_gpu_cosmo_lm61",
            "full_scf_gpu_cosmo_pair_count",
            "full_scf_gpu_cosmo_solv_energy",
            "full_scf_gpu_cosmo_ediel",
            "full_scf_gpu_cosmo_last_residual",
            "full_scf_gpu_cosmo_cg_control_resident",
            "full_scf_gpu_cosmo_cg_converged",
            "full_scf_gpu_cosmo_cg_breakdown",
            "full_scf_gpu_cosmo_cg_host_syncs",
            "full_scf_gpu_cosmo_cg_target_tol",
            "mozyme_makvec_gpu_success_calls",
            "mozyme_makvec_gpu_existing_lmo_calls",
            "mozyme_makvec_gpu_existing_lmo_last_reason",
            "mozyme_makvec_gpu_fallback_calls",
            "mozyme_makvec_gpu_last_ms",
            "mozyme_makvec_gpu_last_fallback_reason",
            "mozyme_makvec_gpu_last_code",
            "mozyme_relocal_gpu_success_calls",
            "mozyme_relocal_gpu_fallback_calls",
            "mozyme_relocal_gpu_occupied_success_calls",
            "mozyme_relocal_gpu_virtual_success_calls",
            "mozyme_reorth_gpu_success_calls",
            "mozyme_reorth_gpu_resident_success_calls",
            "mozyme_reorth_gpu_fallback_calls",
            "mozyme_tidy_gpu_success_calls",
            "mozyme_tidy_gpu_occupied_success_calls",
            "mozyme_tidy_gpu_virtual_success_calls",
            "mozyme_tidy_gpu_selmos_success_calls",
            "mozyme_tidy_gpu_fallback_calls",
            "mozyme_setupk_gpu_success_calls",
            "mozyme_setupk_gpu_fallback_calls",
            "mozyme_setupk_gpu_last_code",
            "mozyme_setupk_gpu_last_ms",
            "mozyme_setupk_gpu_initial_setup_success_calls",
            "mozyme_setupk_gpu_initial_setup_fallback_calls",
            "mozyme_setupk_gpu_initial_setup_all_paths_calls",
            "mozyme_setupk_gpu_initial_setup_last_ms",
            "mozyme_cnvgz_gpu_success_calls",
            "mozyme_cnvgz_gpu_fallback_calls",
            "mozyme_cnvgz_gpu_last_code",
            "mozyme_cnvgz_gpu_last_pmax",
            "mozyme_cnvgz_gpu_last_rms",
            "mozyme_cnvgz_gpu_last_ms",
            "mozyme_helecz_gpu_success_calls",
            "mozyme_helecz_gpu_fallback_calls",
            "mozyme_helecz_gpu_last_code",
            "mozyme_helecz_gpu_last_energy",
            "mozyme_helecz_gpu_last_ms",
            "mozyme_eimp_gpu_success_calls",
            "mozyme_eimp_gpu_fallback_calls",
            "mozyme_eimp_gpu_last_code",
            "mozyme_eimp_gpu_last_pairs",
            "mozyme_eimp_gpu_last_ms",
            "mozyme_diagg1_construct_gpu_success_calls",
            "mozyme_diagg1_construct_gpu_fallback_calls",
            "mozyme_diagg1_construct_gpu_last_code",
            "mozyme_diagg1_construct_gpu_last_nij",
            "mozyme_diagg1_construct_gpu_last_sumt",
            "mozyme_diagg1_construct_gpu_last_tiny",
            "mozyme_diagg1_construct_gpu_last_ms",
            "mozyme_diagg1_aocc_gpu_success_calls",
            "mozyme_diagg1_aocc_gpu_fallback_calls",
            "mozyme_diagg1_aocc_gpu_last_code",
            "mozyme_diagg1_aocc_gpu_last_terms",
            "mozyme_diagg1_aocc_gpu_last_ms",
        "mozyme_diagg1_avir_gpu_success_calls",
        "mozyme_diagg1_avir_gpu_fallback_calls",
        "mozyme_diagg1_avir_gpu_last_code",
        "mozyme_diagg1_avir_gpu_last_terms",
        "mozyme_diagg1_avir_gpu_last_ms",
        "mozyme_diagg2_rotate_gpu_success_calls",
        "mozyme_diagg2_rotate_gpu_fallback_calls",
        "mozyme_diagg2_rotate_gpu_last_code",
        "mozyme_diagg2_rotate_gpu_last_nrej",
        "mozyme_diagg2_rotate_gpu_last_sumb",
        "mozyme_diagg2_rotate_gpu_last_ms",
        "mozyme_diagg2_rotprep_gpu_success_calls",
        "mozyme_diagg2_rotprep_gpu_fallback_calls",
        "mozyme_diagg2_rotprep_gpu_last_code",
            "mozyme_diagg2_rotprep_gpu_last_active",
            "mozyme_diagg2_rotprep_gpu_last_ms",
            "mozyme_isitsc_gpu_success_calls",
            "mozyme_isitsc_gpu_fallback_calls",
            "mozyme_isitsc_gpu_last_code",
            "mozyme_isitsc_gpu_last_okscf",
            "mozyme_isitsc_gpu_last_ms",
            "mozyme_fock1_gpu_success_seen",
            "mozyme_fock2_gpu_success_seen",
            "mozyme_fock1_gpu_attempt_seen",
            "mozyme_fock2_gpu_attempt_seen",
            "mozyme_fock1_gpu_fallback_seen",
            "mozyme_fock2_gpu_fallback_seen",
            "mozyme_fock1_batch_gpu_success_calls",
            "mozyme_fock1_batch_gpu_attempt_calls",
            "mozyme_fock1_batch_gpu_attempt_tasks",
            "mozyme_fock1_batch_gpu_attempt_pairs",
            "mozyme_fock1_batch_gpu_success_tasks",
            "mozyme_fock1_batch_gpu_success_pairs",
            "mozyme_fock1_batch_gpu_fallback_calls",
            "mozyme_fock1_batch_gpu_fallback_tasks",
            "mozyme_fock1_batch_gpu_fallback_pairs",
            "mozyme_fock2_4x1_batch_gpu_success_calls",
            "mozyme_fock2_4x1_batch_gpu_attempt_calls",
            "mozyme_fock2_4x1_batch_gpu_attempt_tasks",
            "mozyme_fock2_4x1_batch_gpu_success_tasks",
            "mozyme_fock2_4x1_batch_gpu_fallback_calls",
            "mozyme_fock2_4x1_batch_gpu_fallback_tasks",
            "mozyme_sparse_fock_setup_calls",
            "mozyme_sparse_fock_setup_one_tasks",
            "mozyme_sparse_fock_setup_pair_tasks",
            "mozyme_sparse_fock_setup_4x1_tasks",
            "mozyme_sparse_fock_setup_point_tasks",
            "mozyme_sparse_fock_setup_point_dipole_tasks",
            "mozyme_sparse_fock_setup_point_monopole_tasks",
            "mozyme_sparse_fock_run_calls",
            "mozyme_sparse_fock_run_one_tasks",
            "mozyme_sparse_fock_run_pair_tasks",
            "mozyme_sparse_fock_run_4x1_tasks",
            "mozyme_sparse_fock_run_point_tasks",
            "mozyme_sparse_fock_run_point_dipole_tasks",
            "mozyme_sparse_fock_run_point_monopole_tasks",
            "mozyme_sparse_fock_run_zero_work_calls",
            "mozyme_sparse_fock_run_ms",
            "mozyme_resident_fock_coverage_mode",
            "mozyme_resident_fock_coverage_use_nijbo",
            "mozyme_resident_fock_real_pairs",
            "mozyme_resident_fock_gpu_real_pairs",
            "mozyme_resident_fock_cpu_real_pairs",
            "mozyme_resident_fock_inactive_real_pairs",
            "mozyme_resident_fock_fallback_pairs",
            "mozyme_resident_fock_basis_limit_fallback_pairs",
            "mozyme_resident_fock_direct_basis_fallback_pairs",
            "mozyme_resident_fock_other_fallback_pairs",
            "mozyme_resident_fock_point_pairs",
            "mozyme_resident_fock_gpu_point_pairs",
            "mozyme_resident_fock_cpu_point_pairs",
            "mozyme_resident_fock_basis_limit_point_fallback_pairs",
            "mozyme_resident_fock_direct_basis_point_fallback_pairs",
            "mozyme_resident_fock_other_point_fallback_pairs",
            "density_calls",
            "density_gpu_syrk_calls",
            "density_gpu_gemm_calls",
            "density_batch_gpu_success_calls",
            "density_batch_gpu_fallback_calls",
            "density_batch_gpu_last_code",
            "density_batch_gpu_last_mode",
            "density_batch_gpu_last_blocks",
            "density_batch_gpu_last_terms",
            "density_batch_gpu_last_ms",
            "density_skipped_diag_blocks",
            "density_skipped_offdiag_blocks",
            "density_cpu_diag_blocks",
            "density_cpu_offdiag_blocks",
            "density_max_diag_block",
            "density_max_offdiag_j",
            "density_max_offdiag_k",
            "log_path",
            "output_files",
            "run_dir",
        ],
    )
    write_csv(
        out_dir / "mozyme_section_times.csv",
        mozyme_section_rows,
        [
            "molecule",
            "input",
            "input_eps",
            "mode",
            "repeat",
            "name",
            "calls",
            "ms",
            "ms_per_call",
            "log_path",
        ],
    )
    write_csv(
        out_dir / "molecule_summary.csv",
        summary,
        [
            "molecule",
            "atoms",
            "cpu_best_s",
            "gpu_best_s",
            "production_gpu_best_s",
            "speedup",
            "production_speedup",
            "speedup_scope",
            "production_gpu_work_units",
            "experimental_scf_status",
            "full_scf_gpu_status",
            "full_scf_gpu_ready",
            "full_scf_gpu_fallback",
            "full_scf_gpu_device_id",
            "cpu_heat_kcal_mol",
            "gpu_heat_kcal_mol",
            "abs_heat_diff_kcal_mol",
            "rel_heat_diff",
            "abs_heat_diff_per_atom_kcal_mol",
            "accuracy_basis",
            "energy_status",
            "proof_status",
            "status",
        ],
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "feature_set": MOPAC_GPU_FEATURE_SET,
                "full_scf_contract_version": MOPAC_GPU_READINESS_CONTRACT_VERSION,
                "benchmark_scope": MOPAC_GPU_BENCHMARK_SCOPE,
                "publication_claim": MOPAC_GPU_PUBLICATION_CLAIM,
                **proof_identity,
                "proof_identity": proof_identity,
                "rows": rows,
                "summary": summary,
                "mozyme_section_times": mozyme_section_rows,
                "mozyme_section_summary": mozyme_section_summary,
                "requirements": {
                    "require_full_scf_gpu": bool(args.require_full_scf_gpu),
                    "full_scf_gpu_contract": (
                        "full_scf_gpu_status must be complete, full_scf_gpu_ready must be 1, "
                        "full_scf_gpu_code must be 0, full_scf_gpu_backend_ready must be 1, "
                        "full_scf_gpu_resident must be 1, full_scf_gpu_device_id must be nonnegative, "
                        f"full_scf_gpu_stage_required must be {MOZYME_SCF_STAGE_FULL}, "
                        f"full_scf_gpu_stage_completed must exactly equal {MOZYME_SCF_STAGE_FULL}, "
                        "full_scf_gpu_stage_missing must be 0, "
                        "full_scf_gpu_resident_decision must be CompleteAndPublish, "
                        "full_scf_gpu_isitsc_okscf must be 1, "
                        "full_scf_gpu_scf_fallback_calls, full_scf_gpu_resident_step_calls, "
                        "and full_scf_gpu_cpu_boundary_calls must be 0, "
                        "full_scf_gpu_final_iterations must be at least 1, "
                        "full_scf_gpu_final_density_resident must be 1, "
                        "full_scf_gpu_final_publication_done must be 1 with positive "
                        "full_scf_gpu_final_publication_arrays and full_scf_gpu_final_publication_bytes, "
                        "full_scf_gpu_pls_restart_required must finish at 0, "
                        "MOZYME makvec initial LMO construction must report GPU success with no fallback, "
                        "MOZYME setupk atom-list construction must report GPU success with initial_setup=1 "
                        "and all_initial_setup_paths=1 and no fallback, "
                        "resident SCF success count, runtime, energy, and DIAGG metrics must be parseable, "
                        "MOZYME section profile markers must be present, "
                        "timed final reorthogonalization sections require a resident=1 reorth success marker "
                        "and CPU final reorthogonalization section calls must be absent, "
                        "full_scf_gpu_cpu_mutating_section_count, call_count, and ms must be 0, raw output must not contain "
                        "GPU error markers, the MOZYME plan must report complete resident Fock coverage, "
                        "resident sparse Fock calls/work must be positive, timing must be nonnegative, "
                        "resident sparse Fock point-charge/dipole work must be semantically covered when present, "
                        "resident sparse Fock real-pair coverage must be all-GPU, "
                        "ordinary EPS/COSMO rows must report resident COSMO Fock, matrix-free CG matvec calls, "
                        "resident CG control/convergence, and zero host CG syncs, "
                        "all resident Fock and MOZYME GPU fallback counters must be zero, "
                        "and SCF DIAGG sumt/sumb must be parseable when emitted"
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plots = make_plots(summary, mozyme_section_summary, out_dir)
    legacy_rows = copy_legacy_references(Path(args.legacy_reference_csv), Path(args.legacy_reference_json), out_dir)
    report_path = out_dir / "README.md"
    write_report(
        report_path,
        rows,
        summary,
        mozyme_section_summary,
        plots,
        legacy_rows,
        args.energy_abs_tol,
        args.energy_rel_tol,
        args.energy_per_atom_tol,
        args.preflight_min_speedup,
        proof_identity,
    )
    files = [path for path in out_dir.rglob("*") if path.is_file()]
    manifest_path = out_dir / "MANIFEST.txt"
    create_manifest(manifest_path, files, proof_identity=proof_identity)

    if not args.no_bundle_zip:
        create_bundle_zip(bundle_path, out_dir)

    print("")
    print("Molecule benchmark summary:")
    for row in summary:
        print(
            f"  {row['molecule']}: CPU={fmt(row['cpu_best_s'])}s "
            f"GPU={fmt(row['gpu_best_s'])}s "
            f"production_speedup={fmt(row.get('production_speedup', row['speedup']))} "
            f"scope={row.get('speedup_scope', '')} "
            f"full_scf_gpu={row.get('full_scf_gpu_status', '')} "
            f"full_scf_ready={fmt(row.get('full_scf_gpu_ready', ''))} "
            f"abs_dH={fmt(row['abs_heat_diff_kcal_mol'])} kcal/mol "
            f"rel_dH={fmt(row['rel_heat_diff'])} "
            f"dH_atom={fmt(row['abs_heat_diff_per_atom_kcal_mol'])} "
            f"status={row['status']} basis={row['accuracy_basis']}"
        )
    print("")
    print(f"Wrote molecule report directory: {out_dir}")
    print(f"  {report_path}")
    for plot in plots:
        print(f"  {plot}")
    if bundle_path is not None:
        print(f"Wrote molecule publication/analysis bundle: {bundle_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

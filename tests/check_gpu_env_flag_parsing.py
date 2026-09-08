#!/usr/bin/env python3
"""Static guards for GPU environment-variable parsing.

The GPU proof path depends on explicit opt-in/opt-out flags.  This test keeps
common regressions from treating arbitrary nonempty values such as ``off`` or
``false`` as enabled in active GPU code.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRUE_TOKENS_FORTRAN = "case ('1','T','TRUE','Y','YES','ON')"
FALSE_TOKENS_FORTRAN = "case ('0','F','FALSE','N','NO','OFF')"
FALSE_TOKENS_FORTRAN_ALT = "case ('0','N','NO','F','FALSE','OFF')"
FALSE_TOKENS_FORTRAN_ALT_COMPACT = "case('0','N','NO','F','FALSE','OFF')"


REQUIRED_FRAGMENTS = {
    "src/matrix/eigenvectors_LAPACK.F90": (
        "call get_environment_variable('MOPAC_FASTGPU'",
        "call get_environment_variable('MOPAC_EIG2HOST'",
        "call get_environment_variable('MOPAC_ORTHO_GPU'",
        "call get_environment_variable('MOPAC_PARTIAL_EIG'",
        "call get_environment_variable('MOPAC_EIG_MG'",
        TRUE_TOKENS_FORTRAN,
    ),
    "src/forces/dcart.F90": (
        "call get_environment_variable('MOPAC_GPU_GRAD'",
        TRUE_TOKENS_FORTRAN,
    ),
    "src/forces/deri1.F90": (
        "call get_environment_variable('MOPAC_NO_GPU_GRAD'",
        "call get_environment_variable('MOPAC_FORCE_GPU_GRAD'",
        TRUE_TOKENS_FORTRAN,
    ),
    "src/SCF/iter.F90": (
        "call get_environment_variable('MOPAC_GPU_SCF_EXPERIMENTAL'",
        "call get_environment_variable('MOPAC_SCF_HYBRID'",
        "call get_environment_variable('MOPAC_GPU_RESIDENT_DEBUG'",
        FALSE_TOKENS_FORTRAN_ALT_COMPACT,
    ),
    "src/SCF/pulay.F90": (
        "call get_environment_variable('MOPAC_DIIS_GEN'",
        "call get_environment_variable('MOPAC_DIIS_GPU'",
        "call get_environment_variable('MOPAC_DIIS_GPU_BUF'",
        "call get_environment_variable('MOPAC_DIIS_GPU_BFULL'",
        "call get_environment_variable('MOPAC_DIIS_GPU_BMAT'",
        TRUE_TOKENS_FORTRAN,
    ),
    "src/run_mopac.F90": (
        "call get_environment_variable('MOPAC_MOZYME_RESIDENT_FOCK_GPU'",
        "call get_environment_variable('MOPAC_MOZYME_FOCK1_BATCH_GPU'",
        "call get_environment_variable('MOPAC_MOZYME_FOCK2_4X1_BATCH_GPU'",
        "call get_environment_variable('MOPAC_MOZYME_FOCK_GPU'",
        "call get_environment_variable('MOPAC_MOZYME_F2_GPU'",
        "call get_environment_variable('MOPAC_MOZYME_CHECK_GPU'",
        "call get_environment_variable('MOPAC_GPU_AUTOPOLICY_OFF'",
        FALSE_TOKENS_FORTRAN,
        TRUE_TOKENS_FORTRAN,
    ),
    "src/corrections/disp_DnX.F90": (
        "call get_environment_variable('MOPAC_DISP_GPU'",
        FALSE_TOKENS_FORTRAN,
    ),
    "src/SCF/fock2.F90": (
        "call get_environment_variable('MOPAC_GPU_EXACT_SC'",
        FALSE_TOKENS_FORTRAN_ALT,
    ),
    "src/matrix/density_for_GPU.F90": (
        "call get_environment_variable('MOPAC_GPU_EXACT_SC'",
        "call get_environment_variable('MOPAC_RESIDENT_SCF'",
        FALSE_TOKENS_FORTRAN_ALT,
    ),
    "src/gpu/cuda_wrappers.cu": (
        "static inline bool env_token_ci",
        "static inline bool env_truthy_ci",
        "static inline bool env_true_ci",
        'std::getenv("MOPAC_RESIDENT_SCF")',
        "g_resident_mode = env_truthy_ci(env) ? 1 : 0;",
        "g_resident_mode = 1; // default on when not specified",
        'std::getenv("MOPAC_STREAMS")',
        'std::getenv("MOPAC_PIN_USER")',
        'std::getenv("MOPAC_MOZYME_F2_GPU")',
        "g_mozyme_f2_flag = env_truthy_ci(env) ? 1 : 0;",
        'std::getenv("MOPAC_SKIP_GPU_DESTROY")',
        "if (env_truthy_ci(skip)) return;",
    ),
    "src/gpu/fock_kernels.cu": (
        "static inline bool env_token_ci",
        "static inline bool env_truthy_ci",
        'std::getenv("MOPAC_GPU_RESIDENT_DEBUG")',
        'std::getenv("MOPAC_GPU_PROFILE")',
        'std::getenv("MOPAC_GPU_VERIFY_FOCK")',
        'std::getenv("MOPAC_GPU_VERIFY_PER_PAIR")',
        "enabled = env_truthy_ci(s) ? 1 : 0;",
        "prof_collect = env_truthy_ci(s) ? 1 : 0;",
        "verify_fock_enabled = env_truthy_ci(s) ? 1 : 0;",
    ),
    "src/gpu/grad_kernels.cu": (
        "static bool env_truthy",
        'std::getenv("MOPAC_GPU_GRAD_EXPERIMENTAL")',
        'std::strcmp(value, "false") != 0',
        'std::strcmp(value, "no") != 0',
        'std::strcmp(value, "off") != 0',
    ),
}


FORBIDDEN_REGEXES = {
    "src/matrix/eigenvectors_LAPACK.F90": (
        r"case\s*\([^)]*'O'[^)]*\)",
        r"case\s*\([^)]*'o'[^)]*\)",
    ),
    "src/run_mopac.F90": (
        r"get_environment_variable\('MOPAC_GPU_AUTOPOLICY_OFF'[\s\S]{0,300}"
        r"case\s*\([^)]*'0'[^)]*\)[\s\S]{0,120}auto_policy\s*=\s*\.false\.",
    ),
    "src/gpu/cuda_wrappers.cu": (
        r"if\s*\(\s*skip\s*&&\s*\*skip\s*\)\s*return\s*;",
    ),
}


def read_source(path: str) -> str:
    source_path = ROOT / path
    try:
        return source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssertionError(f"{path}: could not read file: {exc}") from exc


def main() -> int:
    failures: list[str] = []
    for path, fragments in REQUIRED_FRAGMENTS.items():
        text = read_source(path)
        for fragment in fragments:
            if fragment not in text:
                failures.append(f"{path}: missing required fragment {fragment!r}")
        for pattern in FORBIDDEN_REGEXES.get(path, ()):
            if re.search(pattern, text, re.IGNORECASE):
                failures.append(f"{path}: forbidden GPU env parsing pattern matched {pattern!r}")

    if failures:
        print("GPU environment flag parsing static check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"GPU environment flag parsing static check passed for {len(REQUIRED_FRAGMENTS)} active files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

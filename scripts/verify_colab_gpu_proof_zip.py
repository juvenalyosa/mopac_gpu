#!/usr/bin/env python3
"""Validate a MOZYME GPU Colab proof source zip before upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


SOURCE_MANIFEST_NAME = "MOPAC_COLAB_SOURCE_MANIFEST.sha256"
SOURCE_FEATURES_NAME = "MOPAC_COLAB_FEATURES.json"
SOURCE_PROVENANCE_NAME = "MOPAC_COLAB_SOURCE_PROVENANCE.json"
SOURCE_FEATURES_SCHEMA = "mopac-colab-feature-manifest-v1"
SOURCE_PROVENANCE_SCHEMA = "mopac-colab-source-provenance-v1"
ZIP_SOURCE_TREE_MARKER = "mopac-colab-gpu-proof-source-v1"
SOURCE_MARKER_CONTRACT_VERSION = "mopac-colab-source-markers-explicit-proof-v83"
SOURCE_MARKER_CONTRACT_SHA256 = "17aa11cd4550db1b4d2bbf2f4b15bfc13b9c6e06faad28f6609ee42c7ab69240"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_bytes(data: bytes, label: str, failures: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - report concrete validation failure
        failures.append(f"{label}: invalid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"{label}: expected JSON object")
        return {}
    return value


def safe_zip_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(name)
    return (
        not posix_path.is_absolute()
        and not windows_path.is_absolute()
        and ".." not in posix_path.parts
    )


def has_source_fragment(text: str, flat_text: str, fragment: object) -> bool:
    value = str(fragment)
    return value in text or " ".join(value.split()) in flat_text


def ordered_source_fragments_missing(flat_text: str, fragments: object) -> str | None:
    start = 0
    for fragment in fragments:
        normalized = " ".join(str(fragment).split())
        index = flat_text.find(normalized, start)
        if index < 0:
            return normalized
        start = index + len(normalized)
    return None


def source_marker_contract_sha256(markers: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        {
            "version": SOURCE_MARKER_CONTRACT_VERSION,
            "required_source_markers": markers,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_source_markers(
    zf: zipfile.ZipFile, markers: list[dict[str, Any]], failures: list[str]
) -> None:
    for marker in markers:
        name = str(marker.get("name") or "<unnamed>")
        rel = str(marker.get("path") or "")
        if not rel:
            failures.append(f"{name}: marker is missing path")
            continue
        try:
            text = zf.read(rel).decode("utf-8", errors="ignore")
        except KeyError:
            failures.append(f"{name}: missing {rel}")
            continue
        flat_text = " ".join(text.split())
        for fragment in marker.get("fragments", ()):
            if not has_source_fragment(text, flat_text, fragment):
                failures.append(f"{name}: missing fragment {fragment!r}")
        for fragment in marker.get("forbidden_fragments", ()):
            if has_source_fragment(text, flat_text, fragment):
                failures.append(f"{name}: forbidden fragment {fragment!r}")
        ordered_missing = ordered_source_fragments_missing(
            flat_text, marker.get("ordered_fragments", ())
        )
        if ordered_missing is not None:
            failures.append(f"{name}: ordered fragment missing or out of order {ordered_missing!r}")


def verify_manifest(
    zf: zipfile.ZipFile,
    manifest_text: str,
    expected_manifest_sha256: str,
    failures: list[str],
) -> set[str]:
    manifest_names: set[str] = set()
    actual_manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    if expected_manifest_sha256 and actual_manifest_sha256 != expected_manifest_sha256:
        failures.append(
            "manifest sha256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    names = set(zf.namelist())
    for line_number, raw_line in enumerate(manifest_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected_hash, rel = line.split(None, 1)
        except ValueError:
            failures.append(f"manifest line {line_number}: malformed")
            continue
        rel_path = PurePosixPath(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts or "\\" in rel:
            failures.append(f"manifest line {line_number}: unsafe path {rel!r}")
            continue
        manifest_names.add(rel)
        if rel not in names:
            failures.append(f"manifest line {line_number}: missing zip member {rel}")
            continue
        actual_hash = sha256_bytes(zf.read(rel))
        if actual_hash != expected_hash:
            failures.append(
                f"manifest line {line_number}: sha256 mismatch for {rel}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return manifest_names


def verify_proof_zip(zip_path: Path, sidecar_path: Path | None, expected_sha: str | None) -> list[str]:
    failures: list[str] = []
    if not zip_path.exists():
        return [f"zip does not exist: {zip_path}"]
    actual_zip_sha = sha256_file(zip_path)
    if expected_sha and actual_zip_sha != expected_sha:
        failures.append(f"zip sha256 mismatch: expected {expected_sha}, got {actual_zip_sha}")

    sidecar: dict[str, Any] = {}
    if sidecar_path is not None:
        if not sidecar_path.exists():
            failures.append(f"sidecar does not exist: {sidecar_path}")
        else:
            sidecar = load_json_bytes(sidecar_path.read_bytes(), str(sidecar_path), failures)
            if sidecar.get("source_zip_sha256") != actual_zip_sha:
                failures.append("sidecar source_zip_sha256 does not match zip")
            if sidecar.get("source_marker_contract_version") != SOURCE_MARKER_CONTRACT_VERSION:
                failures.append("sidecar source_marker_contract_version mismatch")
            if sidecar.get("source_marker_contract_sha256") != SOURCE_MARKER_CONTRACT_SHA256:
                failures.append("sidecar source_marker_contract_sha256 mismatch")
            if sidecar.get("proof_eligible") is not True:
                failures.append("sidecar is not proof_eligible")
            if sidecar.get("source_git_dirty") is not False:
                failures.append("sidecar source_git_dirty is not false")

    with zipfile.ZipFile(zip_path) as zf:
        normalized_seen: set[str] = set()
        duplicates: list[str] = []
        unsafe: list[str] = []
        for member in zf.infolist():
            normalized = member.filename.replace("\\", "/")
            if normalized in normalized_seen:
                duplicates.append(member.filename)
            normalized_seen.add(normalized)
            if not safe_zip_member(member.filename):
                unsafe.append(member.filename)
        if unsafe:
            failures.append("unsafe zip member path(s): " + ", ".join(unsafe[:20]))
        if duplicates:
            failures.append("duplicate zip member path(s): " + ", ".join(duplicates[:20]))

        names = set(zf.namelist())
        for required in (SOURCE_MANIFEST_NAME, SOURCE_FEATURES_NAME, SOURCE_PROVENANCE_NAME):
            if required not in names:
                failures.append(f"missing embedded proof metadata: {required}")
        if failures:
            return failures

        manifest_text = zf.read(SOURCE_MANIFEST_NAME).decode("utf-8")
        features = load_json_bytes(zf.read(SOURCE_FEATURES_NAME), SOURCE_FEATURES_NAME, failures)
        provenance = load_json_bytes(zf.read(SOURCE_PROVENANCE_NAME), SOURCE_PROVENANCE_NAME, failures)

        expected_manifest_sha = str(
            sidecar.get("source_manifest_sha256")
            or provenance.get("manifest", {}).get("sha256")
            or features.get("manifest", {}).get("sha256")
            or ""
        )
        manifest_names = verify_manifest(zf, manifest_text, expected_manifest_sha, failures)
        expected_names = manifest_names | {
            SOURCE_MANIFEST_NAME,
            SOURCE_FEATURES_NAME,
            SOURCE_PROVENANCE_NAME,
        }
        missing_names = sorted(expected_names - names)
        extra_names = sorted(names - expected_names)
        if missing_names or extra_names:
            details = []
            if missing_names:
                details.append("missing=" + ", ".join(missing_names[:20]))
            if extra_names:
                details.append("extra=" + ", ".join(extra_names[:20]))
            failures.append("zip member set does not match manifest: " + "; ".join(details))

        for label, payload, schema in (
            (SOURCE_FEATURES_NAME, features, SOURCE_FEATURES_SCHEMA),
            (SOURCE_PROVENANCE_NAME, provenance, SOURCE_PROVENANCE_SCHEMA),
        ):
            if payload.get("schema") != schema:
                failures.append(f"{label}: schema mismatch")
            if payload.get("zip_source_tree_marker") != ZIP_SOURCE_TREE_MARKER:
                failures.append(f"{label}: zip source tree marker mismatch")
            if payload.get("source_marker_contract_version") != SOURCE_MARKER_CONTRACT_VERSION:
                failures.append(f"{label}: source marker contract version mismatch")
            if payload.get("source_marker_contract_sha256") != SOURCE_MARKER_CONTRACT_SHA256:
                failures.append(f"{label}: source marker contract sha256 mismatch")

        if provenance.get("proof_eligible") is not True:
            failures.append(f"{SOURCE_PROVENANCE_NAME}: proof_eligible is not true")
        git_info = provenance.get("git")
        if not isinstance(git_info, dict) or git_info.get("dirty") is not False:
            failures.append(f"{SOURCE_PROVENANCE_NAME}: git dirty is not false")

        critical_files = features.get("critical_files")
        if not isinstance(critical_files, dict) or not critical_files:
            failures.append(f"{SOURCE_FEATURES_NAME}: critical_files is missing")
        elif "scripts/verify_colab_gpu_proof_zip.py" not in critical_files:
            failures.append("proof zip verifier is not listed as a critical file")
        else:
            for rel, metadata in critical_files.items():
                if rel not in names:
                    failures.append(f"critical file missing from zip: {rel}")
                    continue
                if isinstance(metadata, dict):
                    expected_file_hash = metadata.get("sha256")
                    if expected_file_hash and sha256_bytes(zf.read(rel)) != expected_file_hash:
                        failures.append(f"critical file sha256 mismatch: {rel}")

        markers = features.get("required_source_markers")
        if not isinstance(markers, list) or not markers:
            failures.append(f"{SOURCE_FEATURES_NAME}: required_source_markers is missing")
        else:
            actual_marker_sha = source_marker_contract_sha256(markers)
            if actual_marker_sha != SOURCE_MARKER_CONTRACT_SHA256:
                failures.append(
                    f"{SOURCE_FEATURES_NAME}: required_source_markers sha256 mismatch: "
                    f"expected {SOURCE_MARKER_CONTRACT_SHA256}, got {actual_marker_sha}"
                )
            verifier_markers = [
                marker for marker in markers if marker.get("name") == "colab_proof_zip_verifier"
            ]
            if not verifier_markers:
                failures.append("colab_proof_zip_verifier source marker is missing")
            verify_source_markers(zf, markers, failures)

        notebook = zf.read("colab/mopac_cublas_gpu_bench_colab.ipynb").decode(
            "utf-8", errors="ignore"
        )
        stale_zip_member_check = "or " + repr("..") + " in normalized"
        stale_manifest_check = "or " + repr("..") + " in rel:"
        if stale_zip_member_check in notebook or stale_manifest_check in notebook:
            failures.append("notebook contains stale unsafe-path substring check")
        if "or '..' in posix_path.parts" not in notebook or "or '..' in rel_path.parts" not in notebook:
            failures.append("notebook does not use path-component unsafe-path checks")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path, nargs="?", default=Path("mopac_colab_gpu_bench_proof.zip"))
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--expected-sha256", default="")
    args = parser.parse_args(argv)

    sidecar = args.sidecar
    if sidecar is None:
        candidate = args.zip.with_name(args.zip.name + ".expected.json")
        sidecar = candidate if candidate.exists() else None
    expected = args.expected_sha256.strip().lower()
    if expected and not re.fullmatch(r"[0-9a-f]{64}", expected):
        print("expected SHA-256 must be 64 lowercase hex characters", file=sys.stderr)
        return 2

    failures = verify_proof_zip(args.zip, sidecar, expected or None)
    if failures:
        for failure in failures:
            print(f"Colab proof zip verification failure: {failure}", file=sys.stderr)
        return 1
    print(f"Colab proof zip verification passed: {args.zip}")
    print(f"source_zip_sha256={sha256_file(args.zip)}")
    if sidecar:
        print(f"sidecar={sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

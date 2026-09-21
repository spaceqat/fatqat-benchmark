"""Lazy, integrity-checked access to materialized OpenQASM fixtures."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpora" / "qasm"
_MANIFEST = _CORPUS_ROOT / "manifest.toml"


@dataclass(frozen=True, slots=True)
class QasmFixture:
    """One verified QASM source and its immutable corpus metadata."""

    fixture_id: str
    source: str
    path: str
    qasm_format: str
    profiles: tuple[str, ...]
    qubits: int
    tags: tuple[str, ...]
    sha256: str
    origin: str
    license_id: str
    generated_by: str
    exporter_fatqat_commit: str
    workload: str
    workload_parameters: Mapping[str, Any]
    equivalent_native_case: str | None

    @property
    def source_bytes(self) -> int:
        return len(self.source.encode("utf-8"))


def _entries() -> dict[str, Mapping[str, Any]]:
    document = tomllib.loads(_MANIFEST.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("QASM manifest schema_version must be 1")
    raw_entries = document.get("fixture")
    if not isinstance(raw_entries, list):
        raise TypeError("QASM manifest fixture entries must be a list")
    entries: dict[str, Mapping[str, Any]] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise TypeError("QASM manifest fixture entry must be a table")
        fixture_id = entry.get("id")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ValueError("QASM fixture id must be a non-empty string")
        if fixture_id in entries:
            raise ValueError(f"duplicate QASM fixture id {fixture_id!r}")
        entries[fixture_id] = entry
    return entries


def _required(entry: Mapping[str, Any], key: str, expected_type):
    value = entry.get(key)
    if not isinstance(value, expected_type):
        raise TypeError(f"QASM fixture {entry.get('id')!r} field {key!r} is invalid")
    return value


def load_qasm_fixture(fixture_id: str) -> QasmFixture:
    """Load one manifest entry, reject path escape, and verify its digest."""

    try:
        entry = _entries()[fixture_id]
    except KeyError as exc:
        raise KeyError(f"unknown QASM fixture {fixture_id!r}") from exc

    relative_path = Path(_required(entry, "path", str))
    corpus_root = _CORPUS_ROOT.resolve()
    source_path = (_CORPUS_ROOT / relative_path).resolve()
    if not source_path.is_relative_to(corpus_root):
        raise ValueError(f"QASM fixture {fixture_id!r} escapes the corpus root")
    source_bytes = source_path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    expected_digest = _required(entry, "sha256", str)
    if digest != expected_digest:
        raise ValueError(
            f"QASM fixture {fixture_id!r} SHA-256 mismatch: "
            f"expected {expected_digest}, got {digest}"
        )
    qasm_format = _required(entry, "format", str)
    if qasm_format not in ("openqasm2", "openqasm3"):
        raise ValueError(f"QASM fixture {fixture_id!r} has unknown format")
    profiles = tuple(_required(entry, "profiles", list))
    if not profiles or any(
        not isinstance(profile, str) or profile not in ("quick", "full")
        for profile in profiles
    ):
        raise ValueError(f"QASM fixture {fixture_id!r} has invalid profiles")
    qubits = _required(entry, "qubits", int)
    if isinstance(qubits, bool) or qubits <= 0:
        raise ValueError(f"QASM fixture {fixture_id!r} has invalid qubit count")
    tags = tuple(_required(entry, "tags", list))
    if not tags or any(not isinstance(tag, str) or not tag for tag in tags):
        raise ValueError(f"QASM fixture {fixture_id!r} has invalid tags")
    workload_parameters = _required(entry, "workload_parameters", dict)
    equivalent = entry.get("equivalent_native_case")
    if equivalent is not None and not isinstance(equivalent, str):
        raise TypeError("equivalent_native_case must be a string when present")

    return QasmFixture(
        fixture_id=fixture_id,
        source=source_bytes.decode("utf-8"),
        path=relative_path.as_posix(),
        qasm_format=qasm_format,
        profiles=profiles,
        qubits=qubits,
        tags=tags,
        sha256=digest,
        origin=_required(entry, "origin", str),
        license_id=_required(entry, "license", str),
        generated_by=_required(entry, "generated_by", str),
        exporter_fatqat_commit=_required(entry, "exporter_fatqat_commit", str),
        workload=_required(entry, "workload", str),
        workload_parameters=workload_parameters,
        equivalent_native_case=equivalent,
    )


def validate_fixture_metadata(
    fixture: QasmFixture,
    *,
    profiles: frozenset[str],
    qubits: int,
    qasm_format: str,
) -> None:
    """Reject drift between the manifest and a statically declared case."""

    mismatches = []
    if frozenset(fixture.profiles) != profiles:
        mismatches.append(
            f"profiles manifest={fixture.profiles!r} case={tuple(sorted(profiles))!r}"
        )
    if fixture.qubits != qubits:
        mismatches.append(f"qubits manifest={fixture.qubits!r} case={qubits!r}")
    if fixture.qasm_format != qasm_format:
        mismatches.append(
            f"format manifest={fixture.qasm_format!r} case={qasm_format!r}"
        )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            f"QASM fixture {fixture.fixture_id!r} metadata drift: {details}"
        )


def fixture_diagnostics(fixture: QasmFixture) -> dict[str, Any]:
    """Return provenance fields shared by all QASM result groups."""

    return {
        "fixture_id": fixture.fixture_id,
        "path": fixture.path,
        "format": fixture.qasm_format,
        "source_bytes": fixture.source_bytes,
        "sha256": fixture.sha256,
        "origin": fixture.origin,
        "license": fixture.license_id,
        "generated_by": fixture.generated_by,
        "exporter_fatqat_commit": fixture.exporter_fatqat_commit,
        "workload": fixture.workload,
        "workload_parameters": dict(fixture.workload_parameters),
        "equivalent_native_case": fixture.equivalent_native_case,
    }

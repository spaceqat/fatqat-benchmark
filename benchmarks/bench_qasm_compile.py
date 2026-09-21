"""OpenQASM-to-SC/NA compilation benchmarks."""

from __future__ import annotations

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._qasm_corpus import (
        fixture_diagnostics,
        load_qasm_fixture,
        validate_fixture_metadata,
    )
    from ._workloads import grid_couplings
    from .bench_compile_na import _diagnostics as na_diagnostics
    from .bench_compile_sc import _diagnostics as sc_diagnostics
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from _qasm_corpus import (
        fixture_diagnostics,
        load_qasm_fixture,
        validate_fixture_metadata,
    )
    from _workloads import grid_couplings
    from bench_compile_na import _diagnostics as na_diagnostics
    from bench_compile_sc import _diagnostics as sc_diagnostics

import fatqat as fq
from fatqat.compiler.algorithms.zap import load_architecture


def _sc_setup(
    fixture_id: str,
    profiles: frozenset[str],
    width: int,
    rows: int,
    columns: int,
    qasm_format: str,
) -> PreparedCase:
    fixture = load_qasm_fixture(fixture_id)
    validate_fixture_metadata(
        fixture, profiles=profiles, qubits=width, qasm_format=qasm_format
    )
    backend = fq.simulator.SCQubitSimulator(
        num_qubits=width,
        couplings=grid_couplings(rows, columns),
        runtime="numpy",
    )

    def diagnostics(compiled) -> dict[str, object]:
        values = fixture_diagnostics(fixture)
        values.update(sc_diagnostics(compiled))
        return values

    return PreparedCase(
        lambda: fq.compiler.compile_qasm_to_sc(fixture.source, backend, seed=0),
        diagnostics,
    )


def _na_setup(
    fixture_id: str,
    profiles: frozenset[str],
    width: int,
    qasm_format: str,
) -> PreparedCase:
    fixture = load_qasm_fixture(fixture_id)
    validate_fixture_metadata(
        fixture, profiles=profiles, qubits=width, qasm_format=qasm_format
    )
    architecture = load_architecture("default")

    def diagnostics(compiled) -> dict[str, object]:
        values = fixture_diagnostics(fixture)
        values.update(na_diagnostics(compiled))
        return values

    return PreparedCase(
        lambda: fq.compiler.compile_qasm_to_na(fixture.source, architecture),
        diagnostics,
    )


def _sc_case(fixture_id, profiles, width, rows, columns, qasm_format):
    profile_set = frozenset(profiles)
    return BenchmarkCase(
        name=fixture_id,
        group="qasm.compile.sc",
        profiles=profile_set,
        setup=lambda: _sc_setup(
            fixture_id, profile_set, width, rows, columns, qasm_format
        ),
        parameters={
            "fixture_id": fixture_id,
            "format": qasm_format,
            "qubits": width,
            "topology": f"{rows}x{columns}-grid",
            "seed": 0,
        },
        full_repeats=5,
    )


def _na_case(fixture_id, profiles, width, qasm_format):
    profile_set = frozenset(profiles)
    return BenchmarkCase(
        name=fixture_id,
        group="qasm.compile.na",
        profiles=profile_set,
        setup=lambda: _na_setup(fixture_id, profile_set, width, qasm_format),
        parameters={
            "fixture_id": fixture_id,
            "format": qasm_format,
            "atoms": width,
            "architecture": "default",
        },
        full_repeats=5,
    )


CASES = (
    _sc_case(
        "sc_grid_nonlocal_12q_8l_qasm2",
        {"quick", "full"},
        12,
        3,
        4,
        "openqasm2",
    ),
    _sc_case(
        "sc_grid_nonlocal_12q_8l_qasm3",
        {"quick", "full"},
        12,
        3,
        4,
        "openqasm3",
    ),
    _na_case(
        "na_round_robin_20q_4l_qasm2",
        {"quick", "full"},
        20,
        "openqasm2",
    ),
    _na_case(
        "na_round_robin_20q_4l_qasm3",
        {"quick", "full"},
        20,
        "openqasm3",
    ),
    _sc_case(
        "sc_grid_nonlocal_16q_24l_qasm2",
        {"full"},
        16,
        4,
        4,
        "openqasm2",
    ),
    _sc_case(
        "sc_grid_nonlocal_16q_24l_qasm3",
        {"full"},
        16,
        4,
        4,
        "openqasm3",
    ),
    _na_case("na_round_robin_50q_4l_qasm2", {"full"}, 50, "openqasm2"),
    _na_case("na_round_robin_50q_4l_qasm3", {"full"}, 50, "openqasm3"),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="qasm-compile"))

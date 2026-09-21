"""OpenQASM import benchmarks from in-memory source to FatQat Program."""

from __future__ import annotations

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._qasm_corpus import (
        fixture_diagnostics,
        load_qasm_fixture,
        validate_fixture_metadata,
    )
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from _qasm_corpus import (
        fixture_diagnostics,
        load_qasm_fixture,
        validate_fixture_metadata,
    )

from fatqat.qasm import from_qasm


def _setup(
    fixture_id: str,
    profiles: frozenset[str],
    qubits: int,
    qasm_format: str,
) -> PreparedCase:
    fixture = load_qasm_fixture(fixture_id)
    validate_fixture_metadata(
        fixture, profiles=profiles, qubits=qubits, qasm_format=qasm_format
    )

    def diagnostics(program) -> dict[str, object]:
        values = fixture_diagnostics(fixture)
        values.update(
            {
                "quantum_registers": len(program.quantum_registers),
                "qubits": sum(register.size for register in program.quantum_registers),
                "classical_registers": len(program.classical_registers),
                "dag_nodes": len(program.dag().nodes),
            }
        )
        return values

    return PreparedCase(lambda: from_qasm(fixture.source), diagnostics)


def _case(fixture_id: str, profiles: set[str], qubits: int, qasm_format: str):
    profile_set = frozenset(profiles)
    return BenchmarkCase(
        name=fixture_id,
        group="qasm.import",
        profiles=profile_set,
        setup=lambda: _setup(fixture_id, profile_set, qubits, qasm_format),
        parameters={"fixture_id": fixture_id, "format": qasm_format, "qubits": qubits},
    )


CASES = (
    _case("import_layered_12q_10l_qasm2", {"quick", "full"}, 12, "openqasm2"),
    _case("import_layered_12q_10l_qasm3", {"quick", "full"}, 12, "openqasm3"),
    _case("import_layered_32q_64l_qasm2", {"full"}, 32, "openqasm2"),
    _case("import_layered_32q_64l_qasm3", {"full"}, 32, "openqasm3"),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="qasm-import"))

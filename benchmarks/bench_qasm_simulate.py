"""OpenQASM import-plus-simulation benchmarks."""

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

import fatqat as fq
from fatqat.qasm import from_qasm

_SERIAL = {
    "seed": 31,
    "shot_parallelism": "serial",
    "kernel_parallelism": "serial",
    "max_workers": 1,
}


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
    backend = fq.simulator.Simulator(method="statevector", runtime="numba")

    def run():
        program = from_qasm(fixture.source)
        return backend.run(
            program,
            shots=0,
            simulation_config=_SERIAL,
            result_config={"counts": False, "final_state": True},
        ).result()

    def diagnostics(result) -> dict[str, object]:
        values = fixture_diagnostics(fixture)
        values.update(
            {
                "method": result.metadata["method"],
                "runtime": result.metadata["runtime"],
                "state_size": int(result.get_statevector().size),
            }
        )
        return values

    return PreparedCase(run, diagnostics)


def _case(fixture_id: str, profiles: set[str], qubits: int, qasm_format: str):
    profile_set = frozenset(profiles)
    return BenchmarkCase(
        name=fixture_id,
        group="qasm.simulate",
        profiles=profile_set,
        setup=lambda: _setup(fixture_id, profile_set, qubits, qasm_format),
        parameters={
            "fixture_id": fixture_id,
            "format": qasm_format,
            "qubits": qubits,
            "method": "statevector",
            "runtime": "numba",
        },
    )


CASES = (
    _case("simulate_layered_8q_4l_qasm2", {"quick", "full"}, 8, "openqasm2"),
    _case("simulate_layered_8q_4l_qasm3", {"quick", "full"}, 8, "openqasm3"),
    _case("simulate_layered_12q_8l_qasm2", {"full"}, 12, "openqasm2"),
    _case("simulate_layered_12q_8l_qasm3", {"full"}, 12, "openqasm3"),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="qasm-simulate"))

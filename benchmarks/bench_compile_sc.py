"""Superconducting compiler benchmarks using Python LogicalProgram inputs."""

from __future__ import annotations

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._workloads import (
        grid_couplings,
        line_couplings,
        sc_line_local_program,
        sc_nonlocal_program,
    )
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from _workloads import (
        grid_couplings,
        line_couplings,
        sc_line_local_program,
        sc_nonlocal_program,
    )

import fatqat as fq
from fatqat.compiler.dialects.sc_native import NativeGate


def _native_depth(operations) -> int:
    depth_by_site: dict[object, int] = {}
    maximum = 0
    for operation in operations:
        sites = operation.sites if hasattr(operation, "sites") else (operation.site,)
        layer = max((depth_by_site.get(site, 0) for site in sites), default=0) + 1
        for site in sites:
            depth_by_site[site] = layer
        maximum = max(maximum, layer)
    return maximum


def _diagnostics(compiled) -> dict[str, int]:
    operations = compiled.output.operations
    gates = [operation for operation in operations if isinstance(operation, NativeGate)]
    generated = [gate for gate in gates if gate.generated_by is not None]
    return {
        "native_operations": len(operations),
        "native_gates": len(gates),
        "native_two_qubit_gates": sum(len(gate.sites) == 2 for gate in gates),
        "generated_route_gates": len(generated),
        "route_swaps": len({gate.generated_by for gate in generated}),
        "native_depth": _native_depth(operations),
    }


def _line_setup(width: int, layers: int) -> PreparedCase:
    program = sc_line_local_program(width, layers)
    backend = fq.simulator.SCQubitSimulator(
        num_qubits=width,
        couplings=line_couplings(width),
        runtime="numpy",
    )
    return PreparedCase(
        lambda: fq.compiler.compile_to_sc(program, backend, seed=0), _diagnostics
    )


def _grid_setup(width: int, layers: int, rows: int, columns: int) -> PreparedCase:
    program = sc_nonlocal_program(width, layers, seed=23)
    backend = fq.simulator.SCQubitSimulator(
        num_qubits=width,
        couplings=grid_couplings(rows, columns),
        runtime="numpy",
    )
    return PreparedCase(
        lambda: fq.compiler.compile_to_sc(program, backend, seed=0), _diagnostics
    )


CASES = (
    BenchmarkCase(
        name="line_local_16q_8l",
        group="compiler.sc",
        profiles=frozenset({"quick", "full"}),
        setup=lambda: _line_setup(16, 8),
        parameters={"qubits": 16, "layers": 8, "topology": "line", "seed": 0},
        full_repeats=5,
    ),
    BenchmarkCase(
        name="grid_nonlocal_12q_8l",
        group="compiler.sc",
        profiles=frozenset({"quick", "full"}),
        setup=lambda: _grid_setup(12, 8, 3, 4),
        parameters={"qubits": 12, "layers": 8, "topology": "3x4-grid", "seed": 0},
        full_repeats=5,
    ),
    BenchmarkCase(
        name="line_local_32q_16l",
        group="compiler.sc",
        profiles=frozenset({"full"}),
        setup=lambda: _line_setup(32, 16),
        parameters={"qubits": 32, "layers": 16, "topology": "line", "seed": 0},
        full_repeats=5,
    ),
    BenchmarkCase(
        name="grid_nonlocal_16q_24l",
        group="compiler.sc",
        profiles=frozenset({"full"}),
        setup=lambda: _grid_setup(16, 24, 4, 4),
        parameters={"qubits": 16, "layers": 24, "topology": "4x4-grid", "seed": 0},
        full_repeats=5,
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="compiler-sc"))

"""Unstable microbenchmark for the private SABRE routing boundary."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .._harness import BenchmarkCase, PreparedCase, main
    from .._workloads import grid_couplings, sc_nonlocal_program
else:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks._harness import BenchmarkCase, PreparedCase, main
    from benchmarks._workloads import grid_couplings, sc_nonlocal_program

from fatqat.compiler.algorithms import sabre_map
from fatqat.compiler.passes import normalize_sc_program, snapshot_program


def _setup() -> PreparedCase:
    source = sc_nonlocal_program(16, 12, seed=29, logical=False)
    program = normalize_sc_program(snapshot_program(source))
    sites = tuple(range(16))
    couplings = frozenset(grid_couplings(4, 4))
    return PreparedCase(lambda: sabre_map(program, sites, couplings, seed=0))


CASES = (
    BenchmarkCase(
        name="grid_16q_12l",
        group="micro.sabre",
        profiles=frozenset({"quick", "full"}),
        setup=_setup,
        parameters={"qubits": 16, "layers": 12, "topology": "4x4-grid"},
        unstable=True,
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="micro-sabre"))

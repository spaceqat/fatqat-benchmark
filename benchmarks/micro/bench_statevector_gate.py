"""Unstable microbenchmark for one Numba statevector gate application."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .._harness import BenchmarkCase, PreparedCase, main
else:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks._harness import BenchmarkCase, PreparedCase, main

import numpy as np

from fatqat._backends.steps import ApplyMatrixStep, BuiltinKernelKey
from fatqat.simulator._engine.nb import NumbaSVEngine


def _setup() -> PreparedCase:
    engine = NumbaSVEngine()
    engine.initialize((2,) * 16)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    step = ApplyMatrixStep(matrix, (7,), kernel_key=BuiltinKernelKey.X)
    return PreparedCase(lambda: engine.apply(step))


CASES = (
    BenchmarkCase(
        name="numba_x_16q",
        group="micro.statevector",
        profiles=frozenset({"quick", "full"}),
        setup=_setup,
        parameters={"qubits": 16, "gate": "X", "runtime": "numba"},
        unstable=True,
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="micro-statevector"))

"""Unstable microbenchmark for one Numba density-matrix channel."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .._harness import BenchmarkCase, PreparedCase, main
else:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks._harness import BenchmarkCase, PreparedCase, main

import numpy as np

from fatqat._backends.steps import ApplyChannelStep
from fatqat.simulator._engine.nb import NumbaDMEngine


def _setup() -> PreparedCase:
    engine = NumbaDMEngine()
    engine.initialize((2,) * 8)
    probability = 0.02
    first = np.array(
        [[1.0, 0.0], [0.0, np.sqrt(1.0 - probability)]], dtype=np.complex128
    )
    second = np.array([[0.0, np.sqrt(probability)], [0.0, 0.0]], dtype=np.complex128)
    step = ApplyChannelStep((first, second), (3,))
    rng = np.random.default_rng(19)
    return PreparedCase(lambda: engine.apply_channel(step, rng))


CASES = (
    BenchmarkCase(
        name="numba_amplitude_damping_8q",
        group="micro.density",
        profiles=frozenset({"quick", "full"}),
        setup=_setup,
        parameters={"qubits": 8, "channel": "amplitude_damping", "runtime": "numba"},
        unstable=True,
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="micro-density"))

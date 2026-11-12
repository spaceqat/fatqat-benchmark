"""Expectation evaluation on fixed states, without circuit evolution."""

from __future__ import annotations

import math

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from .bench_sweep_estimator import SERIAL
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from bench_sweep_estimator import SERIAL

import fatqat as fq
from fatqat.observable import NumberOperator
import numpy as np


def expectation_inputs(dimensions, kind, terms):
    """Build a seeded state and one observable shared by public/private timings."""
    width = len(dimensions)
    rng = np.random.default_rng(17)
    state = rng.normal(size=math.prod(dimensions)) + 1j * rng.normal(
        size=math.prod(dimensions)
    )
    state /= np.linalg.norm(state)
    if kind in {"pauli_sum", "projector_sum"}:
        factors = []
        for index in range(terms):
            operators = (
                ("ZERO", "ONE")
                if kind == "projector_sum"
                else ("XY", "YZ", "ZX")[(index // width) % 3]
            )
            factors.append((operators, (index % width, (index + 1) % width), 1 / terms))
    elif kind == "mixed_local":
        factors = [([NumberOperator(), "Z"], (0, 1), 1.0)]
    elif kind == "mixed_global":
        factors = [([NumberOperator()] + ["Z"] * (width - 1), range(width), 1.0)]
    elif kind == "mixed_projector":
        factors = [(["ZERO"] * width, range(width), 1.0)]
    else:
        raise ValueError(f"Unknown expectation kind: {kind}")
    return state, fq.Observable.from_sparse(factors)


def prepare_expectation(dimensions, kind, terms):
    """Time public request handling and contraction, with inputs built once."""
    state, observable = expectation_inputs(dimensions, kind, terms)
    program = fq.Program(
        [
            fq.QuantumRegister(1, dim=dim, name=f"q{i}")
            for i, dim in enumerate(dimensions)
        ]
    )
    estimator = fq.Estimator(
        fq.simulator.Simulator(method="statevector", runtime="numba")
    )
    return PreparedCase(
        lambda: estimator.run(
            program,
            observable,
            initial_state=state,
            shots=0,
            simulation_config=SERIAL,
        ).result(),
        lambda result: {"expectation": float(result.get_expectation())},
    )


def _case(name, dimensions, kind, terms, *, guard=False):
    return BenchmarkCase(
        name=name,
        group="simulator.estimator",
        profiles=frozenset({"guard", "full"} if guard else {"full"}),
        setup=lambda: prepare_expectation(dimensions, kind, terms),
        parameters={
            "dimensions": dimensions,
            "kind": kind,
            "terms": terms,
            "observables": 1,
            "runtime": "numba",
            "method": "statevector",
            "shots": 0,
            "seed": 17,
            "initial_state": True,
            "shot_parallelism": "serial",
            "kernel_parallelism": "serial",
            "measurement": "public_fixed_state",
        },
    )


CASES = (
    _case("fixed_pauli_numba_16q_32terms", (2,) * 16, "pauli_sum", 32, guard=True),
    _case(
        "fixed_mixed_local_numba_1t14q",
        (3,) + (2,) * 14,
        "mixed_local",
        1,
        guard=True,
    ),
    # PR 81: eight distinct neighboring projector pairs repeated 32 times.
    _case("fixed_projector_numba_8q_256terms", (2,) * 8, "projector_sum", 256),
    # PR 89: many named factors leave little work to share through a table.
    _case("fixed_mixed_global_numba_1t14q", (3,) + (2,) * 14, "mixed_global", 1),
    _case("fixed_mixed_projector_numba_1t19q", (3,) + (2,) * 19, "mixed_projector", 1),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="expectation"))

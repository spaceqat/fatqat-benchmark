"""Optional exact expectation timings, excluding public request handling."""

from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import math
from pathlib import Path
import sys

if not __package__:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks._harness import PreparedCase, main
from benchmarks.bench_expectation import CASES as PUBLIC_CASES, expectation_inputs


def prepare_contraction(dimensions, kind, terms):
    """Include term conversion and table construction on every timed call."""
    # PR 89 moved the compiled evaluator out of estimator._exact. Keep the
    # import here so unsupported private APIs cannot block other benchmarks.
    module = "fatqat.simulator._engine.numerics.numba_kernels._expectation"
    if importlib.util.find_spec(module) is None:
        module = "fatqat.estimator._exact"
    evaluator = importlib.import_module(module)
    if not getattr(evaluator, "USING_COMPILED_KERNEL", True):
        raise RuntimeError("This benchmark requires compiled expectation kernels")
    kernel = evaluator.expectation_statevector
    state, observable = expectation_inputs(dimensions, kind, terms)
    # Public inputs use subsystem order; the simulator's native buffers are
    # little-endian. Convert once during setup, as the public API does.
    state = state.reshape(dimensions).transpose().copy().reshape(-1)
    factors = observable.terms
    strides = tuple(math.prod(dimensions[:i]) for i in range(len(dimensions)))
    return PreparedCase(
        lambda: kernel(state, factors, dimensions, strides),
        lambda value: {"expectation": float(value)},
    )


CASES = tuple(
    replace(
        case,
        group="micro.expectation",
        profiles=frozenset({"full"}),
        unstable=True,
        setup=lambda p=case.parameters: prepare_contraction(
            p["dimensions"], p["kind"], p["terms"]
        ),
        parameters={**case.parameters, "measurement": "contraction_only"},
    )
    for case in PUBLIC_CASES
    if case.parameters["kind"] in {"pauli_sum", "projector_sum", "mixed_global"}
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="micro-expectation"))

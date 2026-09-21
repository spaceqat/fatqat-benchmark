"""Parameter-sweep and observable-estimation benchmarks."""

from __future__ import annotations

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._workloads import (
        estimator_observables,
        layered_program,
        parameter_bindings,
        parameterized_program,
    )
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from _workloads import (
        estimator_observables,
        layered_program,
        parameter_bindings,
        parameterized_program,
    )

import fatqat as fq

SERIAL = {
    "seed": 17,
    "shot_parallelism": "serial",
    "kernel_parallelism": "serial",
    "max_workers": 1,
}


def _sweep_setup(width: int, layers: int, rows: int) -> PreparedCase:
    program, angles = parameterized_program(width, layers)
    bindings = parameter_bindings(angles, rows, width)
    backend = fq.simulator.Simulator(method="statevector", runtime="numba")
    return PreparedCase(
        lambda: backend.run_sweep(
            program,
            bindings,
            shots=0,
            simulation_config=SERIAL,
            result_config={"counts": False, "final_state": True},
        ).result()
    )


def _estimator_setup(width: int, layers: int, terms: int, shots: int) -> PreparedCase:
    program = layered_program(width, layers)
    observables = estimator_observables(width, terms)
    estimator = fq.Estimator(
        fq.simulator.Simulator(method="statevector", runtime="numba")
    )
    return PreparedCase(
        lambda: estimator.run(
            program,
            observables,
            shots=shots,
            simulation_config=SERIAL,
        ).result()
    )


CASES = (
    BenchmarkCase(
        name="numba_8q_4l_16rows",
        group="simulator.sweep",
        profiles=frozenset({"quick", "full"}),
        setup=lambda: _sweep_setup(8, 4, 16),
        parameters={"qubits": 8, "layers": 4, "rows": 16},
    ),
    BenchmarkCase(
        name="numba_10q_8l_64rows",
        group="simulator.sweep",
        profiles=frozenset({"full"}),
        setup=lambda: _sweep_setup(10, 8, 64),
        parameters={"qubits": 10, "layers": 8, "rows": 64},
    ),
    BenchmarkCase(
        name="exact_numba_8q_4terms",
        group="simulator.estimator",
        profiles=frozenset({"quick", "full"}),
        setup=lambda: _estimator_setup(8, 4, 4, 0),
        parameters={"qubits": 8, "layers": 4, "terms": 4, "shots": 0},
    ),
    BenchmarkCase(
        name="sampled_numba_8q_4terms_256s",
        group="simulator.estimator",
        profiles=frozenset({"quick", "full"}),
        setup=lambda: _estimator_setup(8, 4, 4, 256),
        parameters={"qubits": 8, "layers": 4, "terms": 4, "shots": 256},
    ),
    BenchmarkCase(
        name="exact_numba_12q_16terms",
        group="simulator.estimator",
        profiles=frozenset({"full"}),
        setup=lambda: _estimator_setup(12, 8, 16, 0),
        parameters={"qubits": 12, "layers": 8, "terms": 16, "shots": 0},
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="sweep-estimator"))

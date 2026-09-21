"""Dynamic-shot and parallel-scaling simulator benchmarks."""

from __future__ import annotations

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._workloads import dynamic_program, measured_ghz_program
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from _workloads import dynamic_program, measured_ghz_program

import fatqat as fq


def _dynamic_setup(width: int, rounds: int, shots: int) -> PreparedCase:
    program = dynamic_program(width, rounds)
    backend = fq.simulator.Simulator(method="statevector", runtime="numba")
    config = {
        "seed": 11,
        "shot_parallelism": "serial",
        "kernel_parallelism": "serial",
        "max_workers": 1,
    }
    return PreparedCase(
        lambda: backend.run(
            program,
            shots=shots,
            simulation_config=config,
            result_config={"counts": True, "final_state": False},
        ).result()
    )


def _parallel_setup(width: int, shots: int, strategy: str) -> PreparedCase:
    program = measured_ghz_program(width)
    backend = fq.simulator.Simulator(method="statevector", runtime="numba")
    config = {
        "seed": 13,
        "shot_parallelism": strategy,
        "kernel_parallelism": "serial",
        "max_workers": 1 if strategy == "serial" else 2,
    }
    return PreparedCase(
        lambda: backend.run(
            program,
            shots=shots,
            simulation_config=config,
            result_config={"counts": True, "final_state": False},
        ).result()
    )


def _parallel_workers(strategy: str) -> int:
    return 1 if strategy == "serial" else 2


CASES = (
    BenchmarkCase(
        name="dynamic_numba_6q_4r_256s",
        group="simulator.shots",
        profiles=frozenset({"quick", "full"}),
        setup=lambda: _dynamic_setup(6, 4, 256),
        parameters={"qubits": 6, "rounds": 4, "shots": 256, "strategy": "serial"},
    ),
    BenchmarkCase(
        name="dynamic_numba_8q_8r_1024s",
        group="simulator.shots",
        profiles=frozenset({"full"}),
        setup=lambda: _dynamic_setup(8, 8, 1024),
        parameters={"qubits": 8, "rounds": 8, "shots": 1024, "strategy": "serial"},
    ),
    *tuple(
        BenchmarkCase(
            name=f"terminal_numba_12q_4096s_{strategy}",
            group="simulator.parallel",
            profiles=frozenset({"full"}),
            setup=lambda strategy=strategy: _parallel_setup(12, 4096, strategy),
            parameters={
                "qubits": 12,
                "shots": 4096,
                "strategy": strategy,
                "workers": _parallel_workers(strategy),
            },
        )
        for strategy in ("serial", "threads", "processes")
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="simulator-shots"))

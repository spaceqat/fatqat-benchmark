"""Generic state, density-matrix, operator, and qudit benchmarks."""

from __future__ import annotations

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._workloads import layered_program, mixed_dimension_program
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from _workloads import layered_program, mixed_dimension_program

import fatqat as fq
import fatqat.operations as ops

SERIAL = {
    "seed": 7,
    "shot_parallelism": "serial",
    "kernel_parallelism": "serial",
    "max_workers": 1,
}


def _state_setup(
    method: str,
    runtime: str,
    width: int,
    layers: int,
    *,
    noisy: bool = False,
    fusion: bool = False,
) -> PreparedCase:
    program = layered_program(width, layers)
    noise = None
    if noisy:
        noise = fq.NoiseModel()
        noise.add(fq.noise.AmplitudeDamping(p=0.015), operation=ops.RY)
        noise.add(fq.noise.Depolarizing(p=0.01), operation=ops.CZ)
    backend = fq.simulator.Simulator(method=method, runtime=runtime, noise=noise)
    simulation_config = dict(SERIAL)
    if fusion:
        simulation_config["fusion"] = True

    def run():
        return backend.run(
            program,
            shots=0,
            simulation_config=simulation_config,
            result_config={"counts": False, "final_state": True},
        ).result()

    return PreparedCase(run)


def _mixed_setup() -> PreparedCase:
    program = mixed_dimension_program()
    backend = fq.simulator.Simulator(method="statevector", runtime="numba")
    return PreparedCase(
        lambda: backend.run(
            program,
            shots=0,
            simulation_config=SERIAL,
            result_config={"counts": False, "final_state": True},
        ).result()
    )


def _case(name, profiles, method, runtime, width, layers, **options):
    parameters = {
        "method": method,
        "runtime": runtime,
        "qubits": width,
        "layers": layers,
        **options,
    }
    return BenchmarkCase(
        name=name,
        group="simulator.state",
        profiles=frozenset(profiles),
        setup=lambda: _state_setup(method, runtime, width, layers, **options),
        parameters=parameters,
    )


CASES = (
    _case("sv_numpy_12q_10l", {"quick", "full"}, "statevector", "numpy", 12, 10),
    _case("sv_numba_12q_10l", {"quick", "full"}, "statevector", "numba", 12, 10),
    _case(
        "dm_numba_noisy_6q_8l",
        {"quick", "full"},
        "density_matrix",
        "numba",
        6,
        8,
        noisy=True,
    ),
    _case("sv_numpy_16q_20l", {"full"}, "statevector", "numpy", 16, 20),
    _case("sv_numba_16q_20l", {"full"}, "statevector", "numba", 16, 20),
    _case(
        "dm_numba_noisy_7q_12l",
        {"full"},
        "density_matrix",
        "numba",
        7,
        12,
        noisy=True,
    ),
    _case(
        "dm_numba_fusion_off_9q_20l",
        {"full"},
        "density_matrix",
        "numba",
        9,
        20,
    ),
    _case(
        "dm_numba_fusion_on_9q_20l",
        {"full"},
        "density_matrix",
        "numba",
        9,
        20,
        fusion=True,
    ),
    _case("unitary_numba_8q_8l", {"full"}, "unitary", "numba", 8, 8),
    _case("superop_numba_4q_6l", {"full"}, "superop", "numba", 4, 6),
    BenchmarkCase(
        name="mixed_qudit_numba",
        group="simulator.state",
        profiles=frozenset({"full"}),
        setup=_mixed_setup,
        parameters={"dimensions": [2, 2, 3, 3], "layers": 8},
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="simulator-state"))

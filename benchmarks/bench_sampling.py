"""Counts-only sampling of fixed dense states and channel-noise circuits."""

from __future__ import annotations

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._circuit_catalog import (
        dense_initial_state,
        mid_measure_reset_fq,
        efficient_su2_ring_fq,
    )
    from .bench_circuit_catalog import SERIAL, _diagnostics
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from _circuit_catalog import (
        dense_initial_state,
        mid_measure_reset_fq,
        efficient_su2_ring_fq,
    )
    from bench_circuit_catalog import SERIAL, _diagnostics

import fatqat as fq
import fatqat.operations as ops


def prepare_sampling(
    width, shots, runtime, *, kind="fixed_state", method="statevector"
):
    initial = None
    noise = None
    if kind == "fixed_state":
        # Only measurement instructions: no state preparation or unitary gates
        # in the timed operation. The immutable input is allocated once here.
        initial = dense_initial_state(width)
        program = fq.Program(width, width)
        program.measure_all()
    elif kind == "terminal_noise":
        program = efficient_su2_ring_fq(width)
        noise = fq.NoiseModel()
        noise.add(fq.noise.AmplitudeDamping(p=0.07), operation=ops.RY)
    elif kind == "dynamic_noise":
        program = mid_measure_reset_fq(width)
        noise = fq.NoiseModel()
        noise.add(fq.noise.Depolarizing(p=0.03), operation=ops.CX)
    else:
        raise ValueError(f"Unknown sampling kind: {kind}")
    backend = fq.simulator.Simulator(method=method, runtime=runtime, noise=noise)

    def run():
        return backend.run(
            program,
            shots=shots,
            initial_state=initial,
            simulation_config=SERIAL,
            result_config={"counts": True, "final_state": False},
        ).result()

    return PreparedCase(
        run, lambda result: _diagnostics(result, shots=shots, method=method)
    )


def _case(kind, width, shots, runtime, method="statevector", *, quick=False):
    return BenchmarkCase(
        name=f"{kind}_{method}_{runtime}_{width}q_{shots}s",
        group="simulator.sampling",
        profiles=frozenset({"quick", "full"} if quick else {"full"}),
        setup=lambda: prepare_sampling(width, shots, runtime, kind=kind, method=method),
        parameters={
            "kind": kind,
            "qubits": width,
            "shots": shots,
            "runtime": runtime,
            "method": method,
            "counts": True,
            "final_state": False,
            "shot_parallelism": "serial",
            "kernel_parallelism": "serial",
            "initial_state": kind == "fixed_state",
            "noise": {
                "terminal_noise": "amplitude_damping_0.07_on_RY",
                "dynamic_noise": "depolarizing_0.03_on_CX",
            }.get(kind),
        },
    )


CASES = (
    *(
        _case(
            "fixed_state",
            width,
            shots,
            runtime,
            quick=width == 5 and runtime == "numba",
        )
        for width in (5, 12, 20)
        for shots in (1024, 65536)
        for runtime in ("numpy", "numba")
    ),
    *(
        _case(kind, width, 32, runtime, method, quick=width == 5 and runtime == "numba")
        for kind in ("terminal_noise", "dynamic_noise")
        for method in ("statevector", "density_matrix")
        for width in (5, 8)
        for runtime in ("numpy", "numba")
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="sampling"))

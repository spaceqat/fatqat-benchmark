"""Public guard workflows and full-only regression probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._workloads import layered_program
except ImportError:  # direct execution, matching the other benchmark entrypoints
    from _harness import BenchmarkCase, PreparedCase, main
    from _workloads import layered_program

import fatqat as fq
import fatqat.operations as ops
import numpy as np


@dataclass(frozen=True)
class StructureSwitch(ops.Operation):
    """Select a diagonal, permutation or dense matrix when parameters bind."""

    theta: object
    name: ClassVar[str] = "StructureSwitch"
    num_subsystems: ClassVar[int] = 2


def structure_sweep_inputs(width, rows):
    """Reuse the review's three matrix structures on reversed target orders."""
    theta = fq.Parameter("structure")
    rng = np.random.default_rng(17)
    dense, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    matrices = (
        np.diag(np.exp(1j * np.array([0.2, 0.6, 1.0, 1.5]))),
        np.eye(4, dtype=complex)[[1, 3, 0, 2]],
        dense,
    )
    implementation = fq.implementation.default_matrix_implementation_map()
    implementation.add(StructureSwitch, lambda op: matrices[int(op.theta) % 3])
    program = layered_program(width, 1)
    program.add(StructureSwitch(theta), (width - 2, 1))
    program.add(ops.RY(0.37), 1)
    program.add(StructureSwitch(theta), (1, width - 2))
    return program, theta, np.arange(rows, dtype=float) % 3, implementation


def sweep_inputs(width, layers, rows, parameters):
    """Keep gate positions fixed while varying parameter sharing and batch size."""
    angles = fq.ParameterVector("angles", parameters)
    program = fq.Program(width, width)
    position = 0
    for layer in range(layers):
        for qubit in range(width):
            program.add(ops.RY(angles[position % parameters]), qubit)
            program.add(ops.RZ(angles[(position + 1) % parameters]), qubit)
            position += 2
        for qubit in range(layer % 2, width - 1, 2):
            program.add(ops.CX, (qubit + 1, qubit))
    if parameters > position:
        raise ValueError("More independent parameters than gate positions")
    values = np.random.default_rng(17).uniform(-1, 1, (rows, parameters))
    return program, angles, values


def _diagnostics(results, shots=0):
    if not isinstance(results, list):
        results = [results]
    for result in results:
        if shots:
            if sum(result.get_counts().values()) != shots:
                raise ValueError("Incorrect shot total")
        else:
            state = result.get_statevector()
            if not np.isfinite(state).all() or not np.isclose(np.linalg.norm(state), 1):
                raise ValueError("Invalid statevector")
    return {"rows": len(results), "shots_per_row": shots}


def prepare_sweep(
    rows=8,
    layers=3,
    parameters=5,
    noise=None,
    shots=0,
    workers=1,
    runtime="numba",
    width=8,
    kernel_mode="serial",
    structure_changing=False,
):
    backend_options = {}
    if structure_changing:
        program, angles, values, implementation = structure_sweep_inputs(width, rows)
        backend_options["implementation_map"] = implementation
    else:
        program, angles, values = sweep_inputs(width, layers, rows, parameters)
    model = None
    if noise:
        channel = {
            "amplitude": fq.noise.AmplitudeDamping,
            "phase": fq.noise.PhaseDamping,
            "depolarizing": fq.noise.Depolarizing,
        }[noise]
        model = fq.NoiseModel()
        model.add(channel(p=0.03), operation=ops.RY)
        program.measure_all()
    backend = fq.simulator.Simulator(
        method="statevector", runtime=runtime, noise=model, **backend_options
    )
    config = {
        "seed": 17,
        "shot_parallelism": "threads" if shots and workers > 1 else "serial",
        "kernel_parallelism": kernel_mode,
        "max_workers": workers,
    }
    return PreparedCase(
        lambda: backend.run_sweep(
            program,
            {angles: values},
            shots=shots,
            simulation_config=config,
            result_config={"counts": bool(shots), "final_state": not shots},
        ).result(),
        lambda results: _diagnostics(results, shots),
    )


def prepare_state(
    width=16, layers=1, mode="serial", workers=1, ccx=False, runtime="numba"
):
    program = layered_program(width, 1 if ccx else layers)
    if ccx:
        for layer in range(layers):
            targets = (0, 2, 4) if layer % 2 else (3, 1, 5)
            program.add(ops.CCX, targets)
            program.add(ops.RZ(0.1 + layer * 0.01), targets[-1])
    backend = fq.simulator.Simulator(method="statevector", runtime=runtime)
    return PreparedCase(
        lambda: backend.run(
            program,
            shots=0,
            simulation_config={
                "seed": 17,
                "shot_parallelism": "serial",
                "kernel_parallelism": mode,
                "max_workers": workers,
            },
            result_config={"counts": False, "final_state": True},
        ).result(),
        _diagnostics,
    )


def _case(name, setup, *, guard=True, **parameters):
    return BenchmarkCase(
        name=name,
        group="simulator.guard",
        profiles=frozenset({"guard", "full"} if guard else {"full"}),
        setup=lambda: setup(**parameters),
        parameters={"runtime": "numba", "seed": 17, **parameters},
    )


CASES = (
    *(
        _case(
            f"sweep_8q_3l_5p_{rows}rows",
            prepare_sweep,
            rows=rows,
            layers=3,
            parameters=5,
        )
        for rows in (1, 8, 64, 256)
    ),
    *(
        _case(
            f"sweep_8q_16l_{count}p_64rows",
            prepare_sweep,
            guard=count == 5,
            rows=64,
            layers=16,
            parameters=count,
        )
        for count in (5, 128)
    ),
    *(
        _case(
            f"sweep_{noise}_8rows_2shots", prepare_sweep, noise=noise, shots=2, rows=8
        )
        for noise in ("amplitude", "phase", "depolarizing")
    ),
    _case(
        "sweep_amplitude_8rows_128shots_threads2",
        prepare_sweep,
        noise="amplitude",
        shots=128,
        rows=8,
        workers=2,
    ),
    _case(
        "sweep_13q_1l_5p_2049rows_auto4",
        prepare_sweep,
        guard=False,
        width=13,
        layers=1,
        rows=2049,
        parameters=5,
        kernel_mode="auto",
        workers=4,
    ),
    *(
        _case(
            f"sweep_amplitude_{rows}rows_2shots",
            prepare_sweep,
            guard=False,
            noise="amplitude",
            shots=2,
            rows=rows,
        )
        for rows in (64, 256)
    ),
    _case(
        "sweep_amplitude_64rows_128shots_threads2",
        prepare_sweep,
        guard=False,
        noise="amplitude",
        shots=128,
        rows=64,
        workers=2,
    ),
    _case(
        "sweep_structure_8q_64rows",
        prepare_sweep,
        guard=False,
        width=8,
        rows=64,
        structure_changing=True,
    ),
    *(
        _case(f"state_16q_{layers}l_serial", prepare_state, width=16, layers=layers)
        for layers in (1, 64)
    ),
    _case(
        "state_16q_64l_auto4",
        prepare_state,
        width=16,
        layers=64,
        mode="auto",
        workers=4,
    ),
    _case(
        "state_20q_3l_threads4",
        prepare_state,
        width=20,
        layers=3,
        mode="threads",
        workers=4,
    ),
    _case(
        "local_ccx_14q_64g_threads8",
        prepare_state,
        guard=False,
        width=14,
        layers=64,
        mode="threads",
        workers=8,
        ccx=True,
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="guard"))

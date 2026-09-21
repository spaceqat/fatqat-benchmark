"""Deterministic Python workload builders for the FatQat benchmarks."""

from __future__ import annotations

import random

import fatqat as fq
import fatqat.operations as ops
import numpy as np


def layered_program(width: int, layers: int, *, logical: bool = False):
    """Build alternating rotation and nearest-neighbor entangling layers."""

    program = fq.LogicalProgram(width) if logical else fq.Program(width)
    for layer in range(layers):
        for qubit in range(width):
            angle = 0.03 * (1 + ((layer + qubit) % 17))
            program.add(ops.RY(angle), qubit)
            program.add(ops.RZ(angle / 2), qubit)
        start = layer % 2
        for qubit in range(start, width - 1, 2):
            program.add(ops.CZ, (qubit, qubit + 1))
    return program


def dynamic_program(width: int, rounds: int) -> fq.Program:
    """Build a per-shot path with measurement, reset, and feed-forward."""

    program = fq.Program(width, width)
    for round_index in range(rounds):
        for qubit in range(width):
            program.add(ops.RY(0.07 * (round_index + qubit + 1)), qubit)
            program.measure(qubit, qubit)
            program.add(ops.Reset, qubit)
            target = (qubit + 1) % width
            program.add(ops.X, target, condition=(qubit, 1))
    program.measure_all()
    return program


def measured_ghz_program(width: int) -> fq.Program:
    """Build a terminal-measurement workload eligible for shot parallelism."""

    program = fq.Program(width, width)
    program.add(ops.H, 0)
    for qubit in range(width - 1):
        program.add(ops.CX, (qubit, qubit + 1))
    program.measure_all()
    return program


def parameterized_program(width: int, layers: int):
    """Return a parameterized ansatz and its vector parameter."""

    angles = fq.ParameterVector("theta", width)
    program = fq.Program(width)
    for layer in range(layers):
        for qubit in range(width):
            program.add(ops.RY(angles[qubit]), qubit)
            program.add(ops.RZ(angles[(qubit + layer + 1) % width]), qubit)
        for qubit in range(layer % 2, width - 1, 2):
            program.add(ops.CZ, (qubit, qubit + 1))
    return program, angles


def parameter_bindings(parameter, rows: int, width: int):
    values = np.empty((rows, width), dtype=float)
    for row in range(rows):
        for column in range(width):
            values[row, column] = 0.01 * (1 + ((row * width + column) % 101))
    return {parameter: values}


def estimator_observables(width: int, terms: int = 4) -> tuple[fq.Observable, ...]:
    """Build fixed-width Pauli observables without random state."""

    observables = []
    letters = "XYZ"
    for term in range(terms):
        label = ["I"] * width
        first = term % width
        second = (term * 3 + 1) % width
        label[first] = letters[term % len(letters)]
        label[second] = letters[(term + 1) % len(letters)]
        observables.append(fq.Observable([("".join(label), 1.0)]))
    return tuple(observables)


def mixed_dimension_program(layers: int = 8) -> fq.Program:
    """Build a small mixed qubit/qutrit statevector workload."""

    qubits = fq.QuantumRegister(2, dim=2, name="q")
    qutrits = fq.QuantumRegister(2, dim=3, name="t")
    program = fq.Program([qubits, qutrits])
    for layer in range(layers):
        program.add(ops.RY(0.1 + layer * 0.01), qubits[0])
        program.add(ops.CX, (qubits[0], qubits[1]))
        program.add(ops.Shift(1 + layer % 2), qutrits[0])
        program.add(ops.Clock(1), qutrits[1])
        program.add(ops.Sum, (qutrits[0], qutrits[1]))
    return program


def line_couplings(width: int) -> tuple[tuple[int, int], ...]:
    return tuple((site, site + 1) for site in range(width - 1))


def grid_couplings(rows: int, columns: int) -> tuple[tuple[int, int], ...]:
    edges = []
    for row in range(rows):
        for column in range(columns):
            site = row * columns + column
            if column + 1 < columns:
                edges.append((site, site + 1))
            if row + 1 < rows:
                edges.append((site, site + columns))
    return tuple(edges)


def sc_line_local_program(width: int, layers: int) -> fq.LogicalProgram:
    program = fq.LogicalProgram(width)
    for layer in range(layers):
        for qubit in range(width):
            program.add(ops.RX(0.05 * (layer + 1)), qubit)
            program.add(ops.RZ(0.02 * (qubit + 1)), qubit)
        for qubit in range(layer % 2, width - 1, 2):
            program.add(ops.CZ, (qubit, qubit + 1))
    return program


def sc_nonlocal_program(
    width: int, layers: int, *, seed: int = 0, logical: bool = True
):
    program = fq.LogicalProgram(width) if logical else fq.Program(width)
    rng = random.Random(seed)
    for layer in range(layers):
        qubits = list(range(width))
        rng.shuffle(qubits)
        for first, second in zip(qubits[::2], qubits[1::2], strict=True):
            program.add(ops.CZ, (first, second))
        for qubit in range(width):
            program.add(ops.RZ(0.01 * (layer + qubit + 1)), qubit)
    return program


def round_robin_matchings(width: int, layers: int):
    if width % 2:
        raise ValueError("round-robin workloads require an even width")
    order = list(range(width))
    matchings = []
    for _ in range(layers):
        matchings.append(tuple(zip(order[: width // 2], reversed(order[width // 2 :]))))
        order = [order[0], order[-1], *order[1:-1]]
    return tuple(matchings)


def na_round_robin_program(width: int, layers: int) -> fq.LogicalProgram:
    program = fq.LogicalProgram(width)
    for layer, matching in enumerate(round_robin_matchings(width, layers)):
        for qubit in range(width):
            program.add(ops.RY(0.01 * (layer + qubit + 1)), qubit)
        for first, second in matching:
            program.add(ops.CZ, (first, second))
    return program

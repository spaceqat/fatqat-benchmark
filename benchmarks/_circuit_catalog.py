"""Native circuit recipes for the scalable circuit catalog.

Built for the setup/cold/warm harness; circuit construction is never timed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi, sqrt
from typing import Callable

import fatqat as fq
import numpy as np

SHOTS_STATIC = 1024


SHOTS_DYNAMIC = 32


SPINE = (5, 10, 15, 20, 24)


SPINE_DEEP = (5, 10, 15, 20)


SPINE_DYNAMIC = (5, 10, 15, 20)


SPINE_DYNAMIC_ODD = (5, 9, 13, 17)


SPINE_ITERATIVE = (5, 10, 15)


SPINE_QUTRIT = (3, 6, 9, 12)


SPINE_QUTRIT_DEEP = (3, 6, 9)


SPINE_UNITARY = (2, 3, 4, 5, 6, 7, 8)


SPINE_SUPEROP = (1, 2, 3, 4, 5)


SPINE_INITIAL_STATE = (5, 20)


SPINE_METHOD_SV = (2, 4, 6, 8, 10, 12)


SPINE_METHOD_DM = (2, 4, 6, 8, 10)


SPINE_METHOD_UNITARY = (2, 4, 6, 8)


SPINE_METHOD_SUPEROP = (2, 3, 4, 5)


_METHOD_PROBE_LAYERS = 4


_INITIAL_STATE_SEED = 20260824


_INITIAL_STATE_SHALLOW_LAYERS = 1


_INITIAL_STATE_DEEP_LAYERS = 8


def _layers(n: int) -> int:
    """Layer-count formula for Section B: grows with n, keeps runs bounded."""
    return max(2, n // 4)


def _require_odd_size(n: int, family: str) -> None:
    """Reject widths that would leave one qubit unused in a pairwise layout."""
    if n % 2 == 0:
        raise ValueError(f"{family} requires an odd size, got n={n}")


def _ghz_chain_body_fq(p: fq.Program, n: int) -> None:
    p.add(fq.operations.H, 0)
    for q in range(n - 1):
        p.add(fq.operations.CX, (q, q + 1))


def ghz_chain_fq(n: int) -> fq.Program:
    p = fq.Program(n, n)
    _ghz_chain_body_fq(p, n)
    p.measure_all()
    return p


def graph_state_line_fq(n: int) -> fq.Program:
    p = fq.Program(n, n)
    for q in range(n):
        p.add(fq.operations.H, q)
    for q in range(n - 1):
        p.add(fq.operations.CZ, (q, q + 1))
    p.measure_all()
    return p


def _w_angles(n: int) -> list[float]:
    return [2 * acos(1 / sqrt(n - k + 1)) for k in range(1, n)]


def w_state_fq(n: int) -> fq.Program:
    # CRY is decomposed identically on both sides (RY/2 - CX - RY/-2 - CX)
    # so the twins stay gate-for-gate comparable.
    p = fq.Program(n, n)
    p.add(fq.operations.X, 0)
    for k, theta in enumerate(_w_angles(n), start=1):
        c, t = k - 1, k
        p.add(fq.operations.RY(theta / 2), t)
        p.add(fq.operations.CX, (c, t))
        p.add(fq.operations.RY(-theta / 2), t)
        p.add(fq.operations.CX, (c, t))
        p.add(fq.operations.CX, (t, c))
    p.measure_all()
    return p


def _su2_angles(layer: int, q: int) -> tuple[float, float, float]:
    return (
        0.31 * (layer + 1) + 0.17 * (q + 1),
        0.11 * (layer + 2) + 0.07 * (q + 1),
        0.05 * (layer + 3) + 0.03 * (q + 1),
    )


def efficient_su2_ring_fq(n: int) -> fq.Program:
    p = fq.Program(n, n)
    for layer in range(_layers(n)):
        for q in range(n):
            a, b, _ = _su2_angles(layer, q)
            p.add(fq.operations.RY(a), q)
            p.add(fq.operations.RZ(b), q)
        for q in range(n):
            p.add(fq.operations.CX, (q, (q + 1) % n))
    p.measure_all()
    return p


def vqe_brickwall_fq(n: int) -> fq.Program:
    p = fq.Program(n, n)
    for layer in range(_layers(n)):
        for q in range(n):
            a, b, c = _su2_angles(layer, q)
            p.add(fq.operations.RX(a), q)
            p.add(fq.operations.RY(b), q)
            p.add(fq.operations.RZ(c), q)
        for q in range(0, n - 1, 2):
            p.add(fq.operations.CX, (q, q + 1))
        for q in range(1, n - 1, 2):
            p.add(fq.operations.CX, (q, q + 1))
    p.measure_all()
    return p


def real_amplitudes_fq(n: int) -> fq.Program:
    p = fq.Program(n, n)
    for layer in range(_layers(n)):
        for q in range(n):
            p.add(fq.operations.RY(_su2_angles(layer, q)[0]), q)
        for q in range(n - 1):
            p.add(fq.operations.CX, (q, q + 1))
    p.measure_all()
    return p


def _qft_body_fq(p: fq.Program, n: int) -> None:
    for j in reversed(range(n)):
        p.add(fq.operations.H, j)
        for k in range(j):
            p.add(fq.operations.CPhase(pi / 2 ** (j - k)), (k, j))
    for j in range(n // 2):
        p.add(fq.operations.Swap, (j, n - 1 - j))


def qft_fq(n: int) -> fq.Program:
    p = fq.Program(n, n)
    _qft_body_fq(p, n)
    p.measure_all()
    return p


def qft_entangled_fq(n: int) -> fq.Program:
    p = fq.Program(n, n)
    p.add(fq.operations.H, 0)
    for q in range(n - 1):
        p.add(fq.operations.CX, (q, q + 1))
    _qft_body_fq(p, n)
    p.measure_all()
    return p


_QAOA_REPS = 3


def qaoa_maxcut_fq(n: int) -> fq.Program:
    p = fq.Program(n, n)
    for q in range(n):
        p.add(fq.operations.H, q)
    for rep in range(_QAOA_REPS):
        gamma, beta = 0.4 + 0.1 * rep, 0.7 - 0.1 * rep
        for q in range(n):  # ring cost layer: ZZ via CX-RZ-CX
            a, b = q, (q + 1) % n
            p.add(fq.operations.CX, (a, b))
            p.add(fq.operations.RZ(2 * gamma), b)
            p.add(fq.operations.CX, (a, b))
        for q in range(n):
            p.add(fq.operations.RX(2 * beta), q)
    p.measure_all()
    return p


def mid_measure_reset_fq(n: int) -> fq.Program:
    half = n // 2
    p = fq.Program(n, n)
    p.add(fq.operations.H, 0)
    for q in range(n - 1):
        p.add(fq.operations.CX, (q, q + 1))
    for q in range(half):
        p.measure(q, q)
    p.add(fq.operations.Reset, tuple(range(half)))
    for q in range(n - 1):
        p.add(fq.operations.CX, (q, q + 1))
    for q in range(half, n):
        p.measure(q, q)
    return p


def teleport_feedforward_fq(n: int) -> fq.Program:
    _require_odd_size(n, "teleport_feedforward")
    # chained repeater: the q0 state hops two qubits per teleportation;
    # exactly 2*hops+1 clbits are written, so exactly that many are declared.
    hops = (n - 1) // 2
    p = fq.Program(n, 2 * hops + 1)
    p.add(fq.operations.RY(0.7), 0)
    i = 0
    while i + 2 < n:
        p.add(fq.operations.H, i + 1)
        p.add(fq.operations.CX, (i + 1, i + 2))
        p.add(fq.operations.CX, (i, i + 1))
        p.add(fq.operations.H, i)
        p.measure(i, i)
        p.measure(i + 1, i + 1)
        p.add(fq.operations.X, i + 2, condition=(i + 1, 1))
        p.add(fq.operations.Z, i + 2, condition=(i, 1))
        i += 2
    p.measure(i, 2 * hops)
    return p


_CODE_ROUNDS = 3


def _code_split(n: int) -> tuple[int, int]:
    _require_odd_size(n, "repetition_code")
    data = (n + 1) // 2
    ancilla = min(n - data, data - 1)
    return data, ancilla


def repetition_code_fq(n: int) -> fq.Program:
    data, anc = _code_split(n)
    n_clbits = _CODE_ROUNDS * anc + data
    p = fq.Program(n, n_clbits)
    p.add(fq.operations.X, 1 % data)
    for r in range(_CODE_ROUNDS):
        base = r * anc
        for j in range(anc):
            a = data + j
            p.add(fq.operations.CX, (j, a))
            p.add(fq.operations.CX, (j + 1, a))
            p.measure(a, base + j)
            p.add(fq.operations.X, j + 1, condition=(base + j, 1))
            if r < _CODE_ROUNDS - 1:
                p.add(fq.operations.Reset, a)
    for j in range(data):
        p.measure(j, _CODE_ROUNDS * anc + j)
    return p


def iterative_feedback_fq(n: int) -> fq.Program:
    rounds = n
    p = fq.Program(n, rounds)
    for q in range(1, n):
        p.add(fq.operations.H, q)
    for k in range(rounds):
        target = 1 + k % (n - 1)
        p.add(fq.operations.H, 0)
        p.add(fq.operations.CPhase(pi / 2 ** (k % 4 + 1)), (0, target))
        p.add(fq.operations.H, 0)
        p.measure(0, k)
        p.add(fq.operations.Reset, 0)
        p.add(fq.operations.RZ(pi / 2 ** (k % 4 + 1)), target, condition=(k, 1))
    return p


_COVERAGE_LAYERS = 30


_COVERAGE_SEED = 2026


_GATE_TABLE = [
    ("i", lambda a: fq.operations.I, 1),
    ("h", lambda a: fq.operations.H, 1),
    ("s", lambda a: fq.operations.S, 1),
    ("sdg", lambda a: fq.operations.Sdg, 1),
    ("sx", lambda a: fq.operations.SX, 1),
    ("t", lambda a: fq.operations.T, 1),
    ("tdg", lambda a: fq.operations.Tdg, 1),
    ("x", lambda a: fq.operations.X, 1),
    ("y", lambda a: fq.operations.Y, 1),
    ("z", lambda a: fq.operations.Z, 1),
    ("rx", fq.operations.RX, 1),
    ("ry", fq.operations.RY, 1),
    ("rz", fq.operations.RZ, 1),
    ("p", fq.operations.Phase, 1),
    ("u", lambda a: fq.operations.U(a, a / 2, -a / 3), 1),
    ("u1", fq.operations.U1, 1),
    ("u2", lambda a: fq.operations.U2(a / 2, -a / 3), 1),
    ("u3", lambda a: fq.operations.U3(a, a / 2, -a / 3), 1),
    ("cx", lambda a: fq.operations.CX, 2),
    ("cz", lambda a: fq.operations.CZ, 2),
    ("cy", lambda a: fq.operations.CY, 2),
    ("cs", lambda a: fq.operations.CS, 2),
    ("swap", lambda a: fq.operations.Swap, 2),
    ("iswap", lambda a: fq.operations.iSwap, 2),
    ("cp", fq.operations.CPhase, 2),
    ("ccx", lambda a: fq.operations.CCX, 3),
    ("cswap", lambda a: fq.operations.CSwap, 3),
]


def _coverage_instructions(n: int) -> list[tuple[int, tuple[int, ...], float]]:
    """One seeded instruction stream shared verbatim by both twins."""
    rng = np.random.default_rng(_COVERAGE_SEED + n)
    instructions = []
    for _ in range(_COVERAGE_LAYERS):
        for _ in range(n):
            g = int(rng.integers(0, len(_GATE_TABLE)))
            arity = _GATE_TABLE[g][2]
            if arity > n:
                continue
            targets = tuple(int(t) for t in rng.choice(n, size=arity, replace=False))
            angle = float(rng.uniform(0.1, pi))
            instructions.append((g, targets, angle))
    return instructions


def gate_coverage_random_fq(n: int) -> fq.Program:
    p = fq.Program(n, n)
    for g, targets, angle in _coverage_instructions(n):
        _, make_fq, arity = _GATE_TABLE[g]
        p.add(make_fq(angle), targets if arity > 1 else targets[0])
    p.measure_all()
    return p


def _micro(gate: str) -> Callable[[int], fq.Program]:
    _, make_fq, arity = next(e for e in _GATE_TABLE if e[0] == gate)

    def build(n: int) -> fq.Program:
        p = fq.Program(n, n)
        for layer in range(_COVERAGE_LAYERS):
            for start in range(0, n - arity + 1, arity):
                targets = tuple(range(start, start + arity))
                p.add(
                    make_fq(0.37 + 0.01 * layer), targets if arity > 1 else targets[0]
                )
        p.measure_all()
        return p

    return build


def qutrit_ghz_fq(n: int) -> fq.Program:
    reg = fq.QuantumRegister(n, dim=3)
    p = fq.Program([reg])
    p.add(fq.operations.Fourier, reg[0])
    for q in range(n - 1):
        p.add(fq.operations.Sum, (reg[q], reg[q + 1]))
    return p


def qutrit_hea_fq(n: int) -> fq.Program:
    reg = fq.QuantumRegister(n, dim=3)
    p = fq.Program([reg])
    for layer in range(max(2, n // 3)):
        for q in range(n):
            a, b, c = _su2_angles(layer, q)
            p.add(fq.operations.SubspaceRX(a, (0, 1)), reg[q])
            p.add(fq.operations.SubspaceRY(b, (1, 2)), reg[q])
            p.add(fq.operations.SubspaceRZ(c, (0, 2)), reg[q])
        for q in range(n):
            p.add(fq.operations.CClock(1), (reg[q], reg[(q + 1) % n]))
    return p


def small_operator_fq(n: int) -> fq.Program:
    """Return the unmeasured QFT workload shared by both operator methods."""
    p = fq.Program(n)
    _qft_body_fq(p, n)
    return p


def small_superop_fq(n: int) -> fq.Program:
    """Return a compact non-unitary map with reset between gate layers."""
    p = fq.Program(n)
    _ghz_chain_body_fq(p, n)
    middle = n // 2
    p.add(fq.operations.Reset, middle)
    p.add(fq.operations.RY(0.37), middle)
    for q in reversed(range(n - 1)):
        p.add(fq.operations.CX, (q, q + 1))
    return p


def dense_initial_state(n: int) -> np.ndarray:
    """Build the reproducible normalized state used by initial-state cases."""
    rng = np.random.Generator(np.random.PCG64(_INITIAL_STATE_SEED + n))
    state = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    return np.ascontiguousarray(state / np.linalg.norm(state), dtype=np.complex128)


def _initial_state_body_fq(p: fq.Program, n: int, layers: int) -> None:
    for layer in range(layers):
        for q in range(n):
            p.add(fq.operations.RY(_su2_angles(layer, q)[0]), q)
        for q in range(layer % 2, n - 1, 2):
            p.add(fq.operations.CX, (q, q + 1))


def initial_state_shallow_fq(n: int) -> fq.Program:
    p = fq.Program(n)
    _initial_state_body_fq(p, n, _INITIAL_STATE_SHALLOW_LAYERS)
    return p


def initial_state_deep_fq(n: int) -> fq.Program:
    p = fq.Program(n)
    _initial_state_body_fq(p, n, _INITIAL_STATE_DEEP_LAYERS)
    return p


def _method_probe_angles(layer: int, q: int) -> tuple[float, float]:
    return 0.30 * layer + 0.10 * q + 0.05, 0.20 * layer - 0.07 * q + 0.11


def method_probe_fq(n: int) -> fq.Program:
    p = fq.Program(n)
    for layer in range(_METHOD_PROBE_LAYERS):
        for q in range(n):
            ry, rz = _method_probe_angles(layer, q)
            p.add(fq.operations.RY(ry), q)
            p.add(fq.operations.RZ(rz), q)
        for q in range(n if n > 2 else n - 1):
            p.add(fq.operations.CX, (q, (q + 1) % n))
    return p


@dataclass(frozen=True)
class Family:
    name: str
    section: str
    sizes: tuple[int, ...]
    build_fq: Callable[[int], fq.Program]
    notes: str
    dynamic: bool = False
    qutrit: bool = False
    method: str = "statevector"
    initial_state: bool = False
    export: bool = False

    @property
    def export_only(self) -> bool:
        return (
            self.export
            or self.qutrit
            or self.initial_state
            or self.method != "statevector"
        )

    @property
    def shots(self) -> int:
        return (
            0 if self.export_only else SHOTS_DYNAMIC if self.dynamic else SHOTS_STATIC
        )


_MICROS = {g: _micro(g) for g in ("cx", "cp", "ccx", "rz", "swap")}


CATALOG: tuple[Family, ...] = (
    Family("ghz_chain", "A", SPINE, ghz_chain_fq, "H + CX ladder GHZ baseline"),
    Family(
        "graph_state_line",
        "A",
        SPINE,
        graph_state_line_fq,
        "H wall + nearest-neighbor CZ chain",
    ),
    Family(
        "w_state",
        "A",
        SPINE,
        w_state_fq,
        "single-excitation cascade (decomposed CRY + CX)",
    ),
    Family(
        "efficient_su2_ring",
        "B",
        SPINE_DEEP,
        efficient_su2_ring_fq,
        "RY+RZ layers with CX ring; L = max(2, n//4)",
    ),
    Family(
        "vqe_brickwall",
        "B",
        SPINE_DEEP,
        vqe_brickwall_fq,
        "RX+RY+RZ layers with CX brickwall; L = max(2, n//4)",
    ),
    Family(
        "real_amplitudes",
        "B",
        SPINE_DEEP,
        real_amplitudes_fq,
        "RY-only layers with CX ladder; L = max(2, n//4)",
    ),
    Family("qft", "C", SPINE, qft_fq, "H + controlled-phase cascade + swap reversal"),
    Family(
        "qft_entangled", "C", SPINE, qft_entangled_fq, "GHZ preparation followed by QFT"
    ),
    Family(
        "qaoa_maxcut",
        "C",
        SPINE,
        qaoa_maxcut_fq,
        f"ring ZZ cost + RX mixer, p = {_QAOA_REPS}",
    ),
    Family(
        "mid_measure_reset",
        "D",
        SPINE_DYNAMIC,
        mid_measure_reset_fq,
        "entangle, measure half, reset, re-entangle",
        dynamic=True,
    ),
    Family(
        "teleport_feedforward",
        "D",
        SPINE_DYNAMIC_ODD,
        teleport_feedforward_fq,
        "chained teleportation with conditioned X/Z corrections",
        dynamic=True,
    ),
    Family(
        "repetition_code",
        "D",
        SPINE_DYNAMIC_ODD,
        repetition_code_fq,
        f"{_CODE_ROUNDS} syndrome rounds, conditioned corrections, ancilla reset",
        dynamic=True,
    ),
    Family(
        "iterative_feedback",
        "D",
        SPINE_ITERATIVE,
        iterative_feedback_fq,
        "H-prepared targets; measure -> reset -> conditioned-RZ loop",
        dynamic=True,
    ),
    Family(
        "gate_coverage_random",
        "E",
        SPINE_DEEP,
        gate_coverage_random_fq,
        f"seeded random layers over the full qubit gate set, {_COVERAGE_LAYERS} layers",
    ),
    *(
        Family(
            f"micro_{g}",
            "E",
            SPINE_DEEP,
            _MICROS[g],
            f"single-gate microbenchmark: {g} x {_COVERAGE_LAYERS} layers",
        )
        for g in _MICROS
    ),
    Family(
        "qutrit_ghz",
        "F",
        SPINE_QUTRIT,
        qutrit_ghz_fq,
        "Fourier + Sum ladder (dim=3, statevector export)",
        qutrit=True,
    ),
    Family(
        "qutrit_hea",
        "F",
        SPINE_QUTRIT_DEEP,
        qutrit_hea_fq,
        "SubspaceR* layers + CClock ring (dim=3, statevector export)",
        qutrit=True,
    ),
    Family(
        "small_unitary",
        "G",
        SPINE_UNITARY,
        small_operator_fq,
        "unmeasured QFT map; bounded full-unitary export",
        method="unitary",
    ),
    Family(
        "small_superop",
        "G",
        SPINE_SUPEROP,
        small_superop_fq,
        "entangle + reset + rotation map; bounded full-super-operator export",
        method="superop",
    ),
    Family(
        "initial_state_shallow",
        "H",
        SPINE_INITIAL_STATE,
        initial_state_shallow_fq,
        f"dense initial state evolved through {_INITIAL_STATE_SHALLOW_LAYERS} RY+CX layer",
        initial_state=True,
    ),
    Family(
        "initial_state_deep",
        "H",
        SPINE_INITIAL_STATE,
        initial_state_deep_fq,
        f"dense initial state evolved through {_INITIAL_STATE_DEEP_LAYERS} RY+CX layers",
        initial_state=True,
    ),
    Family(
        "method_statevector",
        "M",
        SPINE_METHOD_SV,
        method_probe_fq,
        f"{_METHOD_PROBE_LAYERS}-layer RY+RZ+CX-ring probe; final-state export",
        export=True,
    ),
    Family(
        "method_density_matrix",
        "M",
        SPINE_METHOD_DM,
        method_probe_fq,
        f"{_METHOD_PROBE_LAYERS}-layer RY+RZ+CX-ring probe; density-matrix export",
        method="density_matrix",
    ),
    Family(
        "method_unitary",
        "M",
        SPINE_METHOD_UNITARY,
        method_probe_fq,
        f"{_METHOD_PROBE_LAYERS}-layer RY+RZ+CX-ring probe; full-unitary export",
        method="unitary",
    ),
    Family(
        "method_superop",
        "M",
        SPINE_METHOD_SUPEROP,
        method_probe_fq,
        f"{_METHOD_PROBE_LAYERS}-layer RY+RZ+CX-ring probe; super-operator export",
        method="superop",
    ),
)

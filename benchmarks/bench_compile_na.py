"""Neutral-atom compiler benchmarks using Python LogicalProgram inputs."""

from __future__ import annotations

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._workloads import na_round_robin_program
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from _workloads import na_round_robin_program

import fatqat as fq
from fatqat.compiler.algorithms.zap import load_architecture
from fatqat.compiler.dialects.na_zoned import (
    CrosstalkEvent,
    GateBatch,
    MoveEvent,
    TransferEvent,
)


def _event_duration(event) -> float:
    if isinstance(event, GateBatch):
        return float(event.duration)
    if isinstance(event, (MoveEvent, TransferEvent, CrosstalkEvent)):
        return max((float(value) for value in event.durations), default=0.0)
    return 0.0


def _diagnostics(compiled) -> dict[str, int | float]:
    events = compiled.output.events
    moves = [event for event in events if isinstance(event, MoveEvent)]
    transfers = [event for event in events if isinstance(event, TransferEvent)]
    batches = [event for event in events if isinstance(event, GateBatch)]
    return {
        "events": len(events),
        "move_events": len(moves),
        "moved_atoms": sum(len(event.atoms) for event in moves),
        "transfer_events": len(transfers),
        "transferred_atoms": sum(len(event.atoms) for event in transfers),
        "total_move_distance": sum(
            sum(float(distance) for distance in event.distances) for event in moves
        ),
        "gate_batches": len(batches),
        "scheduled_gates": sum(len(event.gates) for event in batches),
        "max_batch_width": max((len(event.gates) for event in batches), default=0),
        "makespan": sum(_event_duration(event) for event in events),
    }


def _setup(width: int, layers: int) -> PreparedCase:
    program = na_round_robin_program(width, layers)
    architecture = load_architecture("default")
    return PreparedCase(
        lambda: fq.compiler.compile_to_na(program, architecture), _diagnostics
    )


CASES = (
    BenchmarkCase(
        name="round_robin_20q_4l",
        group="compiler.na",
        profiles=frozenset({"quick", "full"}),
        setup=lambda: _setup(20, 4),
        parameters={"atoms": 20, "layers": 4, "architecture": "default"},
        full_repeats=5,
    ),
    BenchmarkCase(
        name="round_robin_20q_10l",
        group="compiler.na",
        profiles=frozenset({"full"}),
        setup=lambda: _setup(20, 10),
        parameters={"atoms": 20, "layers": 10, "architecture": "default"},
        full_repeats=5,
    ),
    BenchmarkCase(
        name="round_robin_50q_4l",
        group="compiler.na",
        profiles=frozenset({"full"}),
        setup=lambda: _setup(50, 4),
        parameters={"atoms": 50, "layers": 4, "architecture": "default"},
        full_repeats=5,
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="compiler-na"))

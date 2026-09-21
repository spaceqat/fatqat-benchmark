"""Unstable microbenchmark for the private ZAP scheduler."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .._harness import BenchmarkCase, PreparedCase, main
    from .._workloads import round_robin_matchings
else:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks._harness import BenchmarkCase, PreparedCase, main
    from benchmarks._workloads import round_robin_matchings

from fatqat.compiler.algorithms.zap.scheduler import Scheduler


def _setup() -> PreparedCase:
    gates = [gate for layer in round_robin_matchings(50, 8) for gate in layer]

    def run():
        results = {"n_q": 50}
        scheduler = Scheduler(gates, results)
        scheduler.asap_joint()
        return scheduler

    return PreparedCase(run)


CASES = (
    BenchmarkCase(
        name="round_robin_50q_8l",
        group="micro.zap",
        profiles=frozenset({"quick", "full"}),
        setup=_setup,
        parameters={"atoms": 50, "layers": 8},
        unstable=True,
    ),
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="micro-zap"))

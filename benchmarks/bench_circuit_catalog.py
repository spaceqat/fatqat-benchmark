"""Scalable Python-native circuit catalog."""

from __future__ import annotations

try:
    from ._harness import BenchmarkCase, PreparedCase, main
    from ._circuit_catalog import CATALOG, Family, dense_initial_state
except ImportError:  # direct execution
    from _harness import BenchmarkCase, PreparedCase, main
    from _circuit_catalog import CATALOG, Family, dense_initial_state

import fatqat as fq

SERIAL = {
    "seed": 7,
    "shot_parallelism": "serial",
    "kernel_parallelism": "serial",
    "max_workers": 1,
}


def _diagnostics(result, *, shots: int, method: str):
    if shots:
        counts = result.get_counts()
        if sum(counts.values()) != shots:
            raise ValueError("Simulator counts do not sum to the requested shots")
        return {"shots_observed": sum(counts.values()), "outcomes": len(counts)}
    state = getattr(result, f"get_{method}")()
    return {"output_shape": list(state.shape), "output_bytes": state.nbytes}


def prepare_family(family: Family, width: int, runtime: str) -> PreparedCase:
    program = family.build_fq(width)
    backend = fq.simulator.Simulator(method=family.method, runtime=runtime)
    initial = dense_initial_state(width) if family.initial_state else None

    # Reuse the same program, backend and input across cold/warm operations.
    # Sampling cases explicitly suppress final-state materialization.
    def run():
        return backend.run(
            program,
            shots=family.shots,
            initial_state=initial,
            simulation_config=SERIAL,
            result_config={
                "counts": not family.export_only,
                "final_state": family.export_only,
            },
        ).result()

    return PreparedCase(
        run,
        lambda result: _diagnostics(result, shots=family.shots, method=family.method),
    )


def _case(family: Family, width: int, runtime: str) -> BenchmarkCase:
    return BenchmarkCase(
        name=f"{family.name}_{runtime}_{width}q",
        group="simulator.catalog",
        profiles=frozenset(
            {"quick", "full"}
            if width == family.sizes[0] and runtime == "numba"
            else {"full"}
        ),
        setup=lambda: prepare_family(family, width, runtime),
        parameters={
            "family": family.name,
            "section": family.section,
            "qudits" if family.qutrit else "qubits": width,
            "dimension": 3 if family.qutrit else 2,
            "method": family.method,
            "runtime": runtime,
            "shots": family.shots,
            "counts": not family.export_only,
            "final_state": family.export_only,
            "initial_state": family.initial_state,
            "sampling": (
                "none"
                if family.export_only
                else "dynamic" if family.dynamic else "terminal"
            ),
            "shot_parallelism": "serial",
            "kernel_parallelism": "serial",
        },
    )


CASES = tuple(
    _case(family, width, runtime)
    for family in CATALOG
    for width in family.sizes
    for runtime in ("numpy", "numba")
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="circuit-catalog"))

"""Run the complete FatQat benchmark suite."""

from dataclasses import replace

from benchmarks._harness import main
from benchmarks.bench_guard import CASES as GUARD_CASES
from benchmarks.bench_circuit_catalog import CASES as CATALOG_CASES
from benchmarks.bench_sampling import CASES as SAMPLING_CASES
from benchmarks.bench_compile_na import CASES as COMPILE_NA_CASES
from benchmarks.bench_compile_sc import CASES as COMPILE_SC_CASES
from benchmarks.bench_simulator_shots import CASES as SIMULATOR_SHOT_CASES
from benchmarks.bench_simulator_state import CASES as SIMULATOR_STATE_CASES
from benchmarks.bench_sweep_estimator import CASES as SWEEP_ESTIMATOR_CASES
from benchmarks.micro.bench_density_channel import CASES as MICRO_DM_CASES
from benchmarks.micro.bench_sabre import CASES as MICRO_SABRE_CASES
from benchmarks.micro.bench_statevector_gate import CASES as MICRO_SV_CASES
from benchmarks.micro.bench_zap import CASES as MICRO_ZAP_CASES

CASES = (
    *CATALOG_CASES,
    *SAMPLING_CASES,
    *SIMULATOR_STATE_CASES,
    *SIMULATOR_SHOT_CASES,
    *SWEEP_ESTIMATOR_CASES,
    *COMPILE_SC_CASES,
    *COMPILE_NA_CASES,
    *MICRO_SV_CASES,
    *MICRO_DM_CASES,
    *MICRO_SABRE_CASES,
    *MICRO_ZAP_CASES,
    *GUARD_CASES,
)

# Reuse representative cases instead of maintaining a second circuit catalog.
_GUARD_EXISTING = {
    "simulator.shots.dynamic_numba_6q_4r_256s",
    "simulator.state.dm_numba_noisy_6q_8l",
    "simulator.state.mixed_qudit_numba",
    "simulator.catalog.method_unitary_numba_2q",
    "simulator.sampling.fixed_state_statevector_numba_5q_65536s",
}
CASES = tuple(
    (
        replace(case, profiles=case.profiles | {"guard"})
        if case.key in _GUARD_EXISTING
        else case
    )
    for case in CASES
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="fatqat"))

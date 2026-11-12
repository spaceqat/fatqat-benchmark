"""Run the complete FatQat benchmark suite."""

from dataclasses import replace

from benchmarks._harness import main
from benchmarks.bench_guard import CASES as WORKFLOW_CASES
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

_ALL_CASES = (
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
    *WORKFLOW_CASES,
)

# Reuse representative cases instead of maintaining a second circuit catalog.
_GUARD_CATALOG_SIZES = {
    "ghz_chain": (5, 15, 24),
    "qft": (5, 15, 24),
    "qaoa_maxcut": (5, 15, 24),
    "efficient_su2_ring": (5, 15, 20),
    "vqe_brickwall": (5, 15, 20),
    "real_amplitudes": (5, 15, 20),
}
_GUARD_CASE_IDS = {
    "simulator.shots.dynamic_numba_6q_4r_256s",
    "simulator.state.dm_numba_noisy_6q_8l",
    "simulator.state.mixed_qudit_numba",
    "simulator.catalog.method_unitary_numba_8q",
    "simulator.catalog.small_superop_numba_4q",
    "simulator.sampling.fixed_state_statevector_numba_12q_65536s",
    "simulator.sampling.terminal_noise_density_matrix_numba_8q_32s",
    "simulator.sampling.dynamic_noise_density_matrix_numba_5q_32s",
} | {
    f"simulator.catalog.{family}_numba_{width}q"
    for family, widths in _GUARD_CATALOG_SIZES.items()
    for width in widths
}
CASES = tuple(
    (
        replace(case, profiles=case.profiles | {"guard"})
        if case.key in _GUARD_CASE_IDS
        else case
    )
    for case in _ALL_CASES
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="fatqat"))

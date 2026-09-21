"""Run the complete FatQat benchmark suite."""

from benchmarks._harness import main
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
    *SIMULATOR_STATE_CASES,
    *SIMULATOR_SHOT_CASES,
    *SWEEP_ESTIMATOR_CASES,
    *COMPILE_SC_CASES,
    *COMPILE_NA_CASES,
    *MICRO_SV_CASES,
    *MICRO_DM_CASES,
    *MICRO_SABRE_CASES,
    *MICRO_ZAP_CASES,
)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="fatqat"))

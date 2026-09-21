"""Run the independent OpenQASM benchmark track."""

from benchmarks._harness import main
from benchmarks.bench_qasm_compile import CASES as COMPILE_CASES
from benchmarks.bench_qasm_import import CASES as IMPORT_CASES
from benchmarks.bench_qasm_simulate import CASES as SIMULATE_CASES

CASES = (*IMPORT_CASES, *SIMULATE_CASES, *COMPILE_CASES)


if __name__ == "__main__":
    raise SystemExit(main(CASES, suite_name="qasm"))

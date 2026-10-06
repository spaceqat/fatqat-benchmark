# FatQat Benchmark

Standalone Python performance benchmarks for
[FatQat](https://github.com/spaceqat/fatqat). The suite measures the FatQat
installation visible to the current Python interpreter. The main runners do not
modify `sys.path`, create an environment, or run the FatQat correctness tests.

The benchmark suite covers the generic simulator, parameter sweeps and
estimators, superconducting compilation, and neutral-atom compilation.
The [circuit catalog](docs/circuit-catalog.md) adds 29 scalable families,
separate statevector/density-matrix/unitary/superoperator probes, dynamic and
noisy sampling, and counts-only sampling from a fixed dense statevector.
Correctness remains the responsibility of FatQat's existing test suite and
should be run separately before accepting a performance result.

An independent OpenQASM track measures import, import-plus-simulation, and
QASM-to-SC/NA compilation. It uses committed QASM 2/3 fixtures materialized
offline with FatQat's exporter. The QASM track remains separate so its text
frontend does not change the established Python-native timing and memory
baseline.

## Run

Activate an environment containing the FatQat checkout or wheel that should be
measured, then run either the full suite or one benchmark file:

```sh
python runner.py --profile quick
python benchmarks/bench_compile_sc.py --profile quick
python benchmarks/bench_circuit_catalog.py --profile quick
python benchmarks/bench_sampling.py --profile quick
python runner.py --profile full
python qasm_runner.py --profile quick
python qasm_runner.py --profile full
```

Every executable benchmark file supports `--profile quick|full`, `--help`, and
`--list`. Useful overrides include `--warmups`, `--repeats`, `--timeout`, and
`--output`. Diagnostic microbenchmarks use FatQat private APIs, are marked
unstable, and run only with `--include-micro`.

The terminal receives a compact summary. Complete metadata, raw nanosecond
samples, summary statistics, compiler diagnostics, failures, and timeouts are
written to one JSON document under `results/` unless `--output` is supplied.
`configuration.repeats` is the profile or command-line default;
`configuration.case_repeat_overrides` records cases that use a smaller
profile-specific count, and each result records its actual repeat count.

## Measurement model

- Workload construction is outside the primary timed operation and is reported
  separately as `setup_ns`.
- The first operation in a fresh worker process with an isolated Numba cache is
  reported as `cold_ns`.
- Warm measurements follow untimed warm-up calls and use the median as the
  primary summary.
- Each case runs in its own subprocess. A timeout is recorded without stopping
  the remaining cases.
- The primary baseline fixes simulator shot and kernel parallelism to serial.
  Parallel scaling cases are separate.
- Unix workers report process peak RSS with `resource.getrusage`; unsupported
  platforms report `null`. Peak RSS is platform-specific and must not be
  compared across operating systems.
- Simulator, SC compiler, and NA compiler results remain separate. The runner
  deliberately does not emit a single aggregate score.

The suite requires Python 3.12 or newer and otherwise uses only the standard
library plus FatQat and its installed dependencies.
When the optional `git` executable is unavailable, Git commit and dirty-state
metadata are reported as `null` rather than preventing benchmark output.

## QASM corpus

QASM files and their provenance live under `corpora/qasm/`. The TOML manifest
records each stable fixture ID, format, profile, SHA-256 digest, originating
workload, exporter, FatQat revision, license, and paired Python-native case.
Manifest parsing, file reads, and digest verification happen during case setup,
outside the primary timed operation. Benchmark runs never generate or download
QASM input.

## Validate the benchmark infrastructure

Run the standard-library regression suite before accepting benchmark changes:

```sh
python -m unittest discover -s tests -v
```

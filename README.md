# FatQat Benchmark

Run selected [FatQat](https://github.com/spaceqat/fatqat) circuits or compare
two local revisions for performance regressions. Requires Python 3.12+ and
FatQat installed in each environment being measured.

## Choose a profile

| Profile | When to use it | Cases |
| --- | --- | ---: |
| `guard` | Review a performance-sensitive PR. Small, medium and large circuits, all with Numba. Default for `compare.py`. | 38 |
| `quick` | Check an installation or try the catalog at smaller sizes. Default for `runner.py`. | 45 |
| `full` | Investigate a wider change or a specific boundary case. Includes all sizes; filter it to limit time and memory. | 322 |

Guard includes GHZ, QFT, QAOA and variational circuits, SV/DM/unitary/superoperator
methods, sampling, noise, threading and 8/64/256-row parameter sweeps.
Quick is a different selection, not a smaller guard. Full adds larger batches,
custom parameter bindings and other targeted cases.
See the [case catalog](docs/circuit-catalog.md) for sizes and exact selections.
Counts exclude optional private microbenchmarks. Quick/full include
[legacy compiler cases](docs/running-benchmarks.md#compatibility).

```sh
python runner.py --profile guard --list
python runner.py --profile guard
python runner.py --profile guard --case "*sweep*"
```

## Standard options

| Option | Meaning |
| --- | --- |
| `--profile guard\|quick\|full` | Choose a set of cases |
| `--case "*sweep*"` | Select matching case IDs within the profile; repeat to include more |
| `--exclude "*threads*"` | Remove matching case IDs |
| `--list` | Show the selection without running it |
| `--output results/run.json` | Set the result path; otherwise saved under `results/` |

Patterns are case-sensitive; exclusions win. Use `--help` for timing options.

## Compare a PR

Install base and candidate FatQat in separate environments with matching Python
and dependency versions. Pass their Python executable paths:

```sh
python compare.py --base-python BASE_PYTHON --candidate-python PR_PYTHON --output results/pr-check.json
```

Find each path with `python -c "import sys; print(sys.executable)"` in that
environment; quote paths containing spaces. Comparison uses `guard` by default
and accepts the same profile and case filters. It writes JSON and HTML reports
with a default 5% slowdown budget. Exit codes: **0** pass, **1** regression,
**2** error, **3** inconclusive. See the [running guide](docs/running-benchmarks.md)
for timing, uncertainty and longer follow-up runs.

## Code layout

- `benchmarks/bench_*.py` define circuits and the operations to time.
- `runner.py` collects cases and selects existing cases for guard.
- `benchmarks/_harness.py` filters cases, runs each in a subprocess and saves timings.
- `compare.py` measures the same cases in two installed environments. It does not install packages or switch checkouts.
- `qasm_runner.py` runs the separate OpenQASM benchmarks, including parsing and import costs.

Run this repository's tests with `python -m unittest discover -s tests -v`.
Run FatQat's correctness tests separately before accepting performance results.

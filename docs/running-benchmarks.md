# Running benchmarks

Start with [profile choice and PR comparison commands](../README.md).
The [catalog](circuit-catalog.md) lists circuits, sizes and parameter sweeps.

## Select cases

```sh
python runner.py --profile full --case "simulator.catalog.qft_*" --list
python runner.py --profile guard --case "*sweep*" --exclude "*128shots*"
python benchmarks/bench_sampling.py --profile quick
python qasm_runner.py --profile quick
```

`--case` and `--exclude` match case-sensitive patterns over case IDs. Repeat
`--case` to include more matches; exclusions then remove matches. The profile
still limits selection, and an empty selection fails. `--describe` prints case
parameters and environment details as JSON without running workloads.

Individual benchmark files and the QASM runner accept the same filters. QASM
has quick/full profiles only. Private microbenchmarks require `--include-micro`
on the main runner and may break when FatQat's internal APIs change.

## What gets timed

Each case runs in its own subprocess. Ordinary runs print a summary and save
JSON under `results/`, or at `--output`. Failures and timeouts are recorded
without stopping the remaining cases.

| Measurement | Includes |
| --- | --- |
| `setup_ns` | Building the circuit, backend, noise model and inputs |
| `cold_ns` | First operation in the worker, excluding interpreter startup and imports |
| Warm samples | Complete calls after untimed warm-ups, including result retrieval |

Ordinary runs use a fresh Numba cache per case. Comparisons keep separate caches
for base and candidate and reuse them across rounds, so their first-call times
are not a fresh-cache JIT comparison.

Use `--warmups`, `--repeats` and `--timeout` to change measurement settings.
Ordinary warm samples time one call by default; `--min-sample-time` averages
complete calls over a minimum duration. Results contain raw samples, actual
repeat and call counts, environment details and diagnostics. Median warm time
is the main summary; there is no combined score across unrelated cases.

Unix workers also report peak process RSS. Unsupported platforms report `null`;
RSS should not be compared across operating systems.

## Read a comparison

By default, `compare.py` runs five paired rounds in seeded, randomized
base/candidate order. Workers run sequentially, warm up, then collect five
samples of at least 50 ms each. Both environments must use matching Python
and dependency versions.

The HTML report shows per-case timings, source hashes, environment details and
an estimated 95% confidence interval for the slowdown, calculated by resampling
paired rounds. The linked JSON contains raw samples and worker diagnostics.

- **Pass (exit 0):** all selected cases stay within the slowdown budget.
- **Regression (exit 1):** a case's interval is entirely above the budget.
- **Error (exit 2):** a case fails or environments are incompatible.
- **Inconclusive (exit 3):** timing uncertainty crosses the budget, or there are too few rounds.

`--threshold 0.05` sets the default 5% budget. `--min-change-ms` adds an absolute
minimum slowdown per call (default zero). For a flagged or uncertain case,
repeat with more rounds and longer samples:

```sh
python compare.py --base-python BASE_PYTHON --candidate-python PR_PYTHON --case "*sweep*" --rounds 11 --min-sample-time 0.2 --output results/pr-sweeps-long.json
```

Use `--runner qasm_runner.py --profile quick` to compare QASM cases.
Completed pairs are saved as the comparison progresses. An interrupted report
has `complete: false` and is not a pass. Existing output files are never
overwritten. A passing run covers only the selected cases on that machine.

Use an idle machine and the same CPU affinity and BLAS thread limits for both
environments. Commands inherit and record thread settings; they do not change
them. Explicit threaded cases need a compatible Numba build.

## Compatibility

Legacy Python SC/NA compiler cases require older FatQat compiler APIs. They
are outside guard and can fail on newer installations. Unsupported modes remain
visible as failures, not performance wins. Use case filters to select supported
workloads.

The suite uses the standard library plus FatQat and its dependencies. Git is
optional: without it, commit and dirty-state metadata are `null`.

## QASM inputs

Fixtures and provenance are in `corpora/qasm/`. Its TOML manifest records IDs,
formats, profiles, file hashes, source workloads, exporters, FatQat revisions,
licenses and paired Python cases. Reading files and checking hashes happen
during setup. The timed operations are defined by each QASM case; runs never
generate or download inputs.

## Correctness checks

Run `python -m unittest discover -s tests -v` after changing the benchmarks.
Tests check selection, reporting and representative circuit outputs, including
Numba/NumPy agreement and sweeps against individual bindings. Post-run
diagnostics check state norms or shot totals where provided. These checks
supplement FatQat's own correctness tests.

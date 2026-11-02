# FatQat Benchmark

Run selected [FatQat](https://github.com/spaceqat/fatqat) workloads or compare
two local revisions for performance regressions. Requires Python 3.12+ and
FatQat installed in the environment being measured.

The suite covers simulation, parameter sweeps, estimators, SC/NA compilation
and a separate OpenQASM track. Each run measures the FatQat installation
available to its Python interpreter.

## Choose a profile

| Profile | Use it for | Cases |
| --- | --- | ---: |
| `guard` | Local PR checks: sweeps, noise, threading and representative simulations | 25 |
| `quick` | Small examples across the existing catalog; runner default | 45 |
| `full` | All registered sizes and workloads | 322 |

Start with **`guard` when reviewing a performance-sensitive PR**. It focuses on
representative execution paths and regression risks, including parameter batches,
noise and parallel execution. This is the default for `compare.py`.

Choose **`quick` for a broad first look** at the existing catalog, using smaller
workloads to check that an environment works or explore which cases matter.
It is the default for `runner.py`; it is not a smaller version of `guard`.

Use **`full` for a broader investigation**, such as changes affecting many
simulation or compiler paths. It includes larger workloads, so expect more time
and memory use. Start with `--list` and filter to the relevant cases if needed.
No profile guarantees coverage of every regression.

Counts exclude optional private microbenchmarks. `quick` and `full` also include
legacy compiler cases; see [compatibility notes](docs/running-benchmarks.md#compatibility).

```sh
python runner.py --profile guard --list
python runner.py --profile guard
python runner.py --profile guard --case "*sweep*"
```

## Standard options

| Option | Meaning |
| --- | --- |
| `--profile guard\|quick\|full` | Choose the workload set above |
| `--case "*sweep*"` | Run matching case IDs within that profile; repeat to include more |
| `--exclude "*threads*"` | Remove matching case IDs |
| `--list` | Preview the selection without running it |
| `--output results/run.json` | Choose the result file; otherwise saved under `results/` |

Patterns are case-sensitive; exclusions win. An empty selection fails.
Use `--help` for timing controls and diagnostic options.

## Compare a PR

Install base and candidate FatQat in separate environments with matching Python
and dependency versions. Substitute their Python executable paths below:

```sh
python compare.py --base-python BASE_PYTHON --candidate-python PR_PYTHON --output results/pr-check.json
```

To find each executable path, run `python -c "import sys; print(sys.executable)"`
in that environment. Quote paths containing spaces.
Comparison defaults to `guard` and accepts `--profile`, `--case` and `--exclude`.
It writes JSON and HTML reports, using a 5% slowdown budget by default. Exit codes:
**0** pass, **1** regression, **2** error, **3** inconclusive. Confirm flagged
cases with a longer run; see [measurement and comparison details](docs/running-benchmarks.md).

## How it fits together

**Case files** build circuits and define the timed operation. The **harness**
selects cases and measures them in subprocesses. **compare.py** runs the same
selection sequentially against two environments. It does not install packages
or switch Git checkouts. OpenQASM uses a separate `qasm_runner.py` because its
timings include frontend work.

See the [circuit catalog](docs/circuit-catalog.md) for workloads. Validate this
repository with `python -m unittest discover -s tests -v`; run FatQat's own
correctness tests separately before accepting performance results.
